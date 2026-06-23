//! Compaction adapter — bridges the rich [`context::StandardContextManager`]
//! onto the harness-loop compaction seam.
//!
//! Implements issue #55. Issue #46 wired the verify→retry→warn *machinery*
//! into the harness loop and proved it with test-double context managers. The
//! rich `StandardContextManager` from #29 implements compaction against the
//! *rich* `context::SessionState` / `CompactionResult` API and was never
//! reachable from the loop seam. This module is the production bridge.
//!
//! ## Adapter type
//!
//! [`StandardCompactionAdapter`] wraps an
//! `Arc<context::StandardContextManager<M>>` and implements the harness-side
//! [`harness::ContextManager`] trait so a [`harness::HarnessBuilder`] can be
//! built with a rich manager that *actually compacts*.
//!
//! ## Seam methods satisfied
//!
//! - [`assemble`](harness::ContextManager::assemble) — minimal pass-through
//!   (NOT load-bearing for compaction; builds a `Context` from the session
//!   messages, mirroring the loop's test doubles).
//! - [`append_tool_result`](harness::ContextManager::append_tool_result) /
//!   [`append_user_message`](harness::ContextManager::append_user_message) —
//!   minimal: append to `harness::SessionState.messages`.
//! - [`should_compact`](harness::ContextManager::should_compact) — reconstruct
//!   rich state from `session.extras`, delegate to rich `should_compact`.
//! - [`prepare_compaction_turn`](harness::ContextManager::prepare_compaction_turn)
//!   — reconstruct rich state → rich `prepare_compaction`; `None` when there is
//!   nothing to compact, else project hints + verification state + count.
//! - [`inject_missing_items`] — inherited default (the fixture asserts that
//!   exact prompt; NOT overridden).
//! - [`apply_compaction`](harness::ContextManager::apply_compaction) —
//!   reconstruct rich state, delegate to rich `apply_compaction`, log+swallow
//!   any `Err` (the loop must never halt on compaction), write the mutated
//!   rich state back into the session.
//!
//! ## Rules enforced
//!
//! 1. STATELESS bridge — the adapter holds no session state. Rich
//!    `context::SessionState` is serialized into `harness::SessionState.extras`
//!    under [`RICH_STATE_KEY`] on every mutating seam call and re-read on every
//!    read. No `Mutex`/field carries session state.
//! 2. Compaction never halts the loop — `apply_compaction` swallows the rich
//!    `Err` (logged), and a malformed/absent rich-state blob degrades to a
//!    safe default (no compaction) rather than panicking.
//! 3. The summary is wrapped as `Message { role: Assistant, .. }` for the rich
//!    `CompactionResult` so the rich manager prepends it as the summary turn.
//!
//! No `// SPEC QUESTION:` markers — both design ambiguities (stateless bridge,
//! reuse of the existing `compaction_loop` fixture) are resolved by the issue.

use std::sync::Arc;

use crate::agent::Context as AgentContext;
use crate::context::{
    CompactionResult, ContextManager as RichContextManager, SessionState as RichSessionState,
    StandardContextManager,
};
use crate::harness::{
    BoxFut, CompactionTurn, ContextManager as HarnessContextManager, SessionState as HarnessState,
    Task, ToolOutput, ToolResult,
};
use crate::model::{Content, Message, ModelInterface, ModelParams, Role};

/// Reserved key under `harness::SessionState.extras` holding the serialized
/// rich `context::SessionState`. The adapter is the only writer/reader.
pub const RICH_STATE_KEY: &str = "spore.compaction_adapter.rich_state";

/// Rough token estimate for a single message: the byte length of its textual
/// content divided by four (the same chars/4 proxy `StandardContextManager`
/// uses for cache-marker placement). Used by the adapter to compute real
/// `tokens_reclaimed` from the messages a compaction drops, since the
/// synchronous harness seam cannot call the async `count_tokens`.
pub fn estimate_message_tokens(message: &Message) -> u32 {
    let bytes = match &message.content {
        Content::Text { text } => text.len(),
        Content::ToolCall(tc) => tc.name.len() + tc.input.to_string().len(),
        Content::ToolResult(tr) => tr.content.len(),
        Content::Image { data, .. } => data.len(),
    };
    // chars/4 proxy, at least 1 token for any non-empty message so a drop is
    // never accounted as zero reclamation.
    ((bytes / 4) as u32).max(if bytes > 0 { 1 } else { 0 })
}

/// Sum [`estimate_message_tokens`] over a slice of messages.
pub fn estimate_tokens(messages: &[Message]) -> u32 {
    messages.iter().map(estimate_message_tokens).sum()
}

/// Stateless bridge from the rich [`StandardContextManager`] onto the
/// harness-loop compaction seam ([`harness::ContextManager`]).
///
/// Construct via [`StandardCompactionAdapter::new`] or
/// [`HarnessContextManagerExt::into_harness_adapter`], then inject the result
/// into a [`crate::harness::HarnessBuilder`].
pub struct StandardCompactionAdapter<M: ModelInterface> {
    inner: Arc<StandardContextManager<M>>,
}

impl<M: ModelInterface> StandardCompactionAdapter<M> {
    /// Wrap a rich [`StandardContextManager`] as a harness-seam context manager.
    pub fn new(inner: Arc<StandardContextManager<M>>) -> Self {
        Self { inner }
    }

    /// Reconstruct the rich session state from `extras`. Returns `None` when no
    /// rich state has been seeded yet or the blob is malformed — callers treat
    /// that as "nothing to compact" so the loop is never blocked.
    fn read_rich_state(session: &HarnessState) -> Option<RichSessionState> {
        let value = session.extras.get(RICH_STATE_KEY)?;
        serde_json::from_value::<RichSessionState>(value.clone()).ok()
    }

    /// Serialize the rich session state back into `extras` and project its
    /// `message_history` onto the harness-side `messages`.
    fn write_rich_state(session: &mut HarnessState, rich: &RichSessionState) {
        session.messages = rich.message_history.clone();
        if let Ok(value) = serde_json::to_value(rich) {
            session.extras.insert(RICH_STATE_KEY.to_string(), value);
        }
    }
}

impl<M: ModelInterface + 'static> HarnessContextManager for StandardCompactionAdapter<M> {
    fn assemble<'a>(
        &'a self,
        session: &'a HarnessState,
        _task: &'a Task,
    ) -> BoxFut<'a, AgentContext> {
        // NOT load-bearing for compaction. The rich `assemble` requires
        // `ContextSources` the seam does not supply, so we produce a minimal
        // context straight from the session messages (mirrors the loop's
        // test-double managers).
        let messages = session.messages.clone();
        Box::pin(async move {
            AgentContext {
                messages,
                tools: Vec::new(),
                params: ModelParams::default(),
            }
        })
    }

    fn append_tool_result<'a>(
        &'a self,
        session: &'a mut HarnessState,
        result: &'a ToolResult,
    ) -> BoxFut<'a, ()> {
        let text = match &result.output {
            ToolOutput::Success { content, .. } => content.clone(),
            ToolOutput::Error { message, .. } => message.clone(),
            // Normally normalized into an `Error` by the harness before being
            // appended; record the violation text defensively if it reaches here.
            ToolOutput::SandboxViolation { violation } => format!("sandbox violation: {violation:?}"),
            ToolOutput::WaitingForHuman { .. } => String::new(),
            ToolOutput::Escalate { .. } => String::new(),
            ToolOutput::AwaitingClarification { .. } => String::new(),
            ToolOutput::Consult { .. } => String::new(),
        };
        Box::pin(async move {
            session.messages.push(Message {
                role: Role::Tool,
                content: Content::Text { text },
            });
        })
    }

    fn append_assistant_message<'a>(
        &'a self,
        session: &'a mut HarnessState,
        message: &'a Message,
    ) -> BoxFut<'a, ()> {
        let message = message.clone();
        Box::pin(async move {
            session.messages.push(message);
        })
    }

    fn append_user_message<'a>(
        &'a self,
        session: &'a mut HarnessState,
        text: &'a str,
    ) -> BoxFut<'a, ()> {
        let text = text.to_string();
        Box::pin(async move {
            session.messages.push(Message {
                role: Role::User,
                content: Content::Text { text },
            });
        })
    }

    fn should_compact(&self, session: &HarnessState) -> bool {
        match Self::read_rich_state(session) {
            Some(rich) => self.inner.should_compact(&rich),
            None => false,
        }
    }

    fn prepare_compaction_turn(&self, session: &HarnessState) -> Option<CompactionTurn> {
        let rich = Self::read_rich_state(session)?;
        let request = self.inner.prepare_compaction(&rich).ok()?;
        if request.messages_to_compact.is_empty() {
            return None;
        }
        // The exact dropped set, computed ONCE here and carried on the turn so
        // `apply_compaction` reuses it (no 2nd `prepare_compaction`).
        let dropped_messages = request.messages_to_compact.clone();
        let messages_removed = dropped_messages.len() as u32;

        // Build the summarization context: the messages to compact, followed by
        // the summarization instruction. `inject_missing_items` (inherited
        // default) appends the retry instruction on verification failure.
        let mut messages = dropped_messages.clone();
        messages.push(Message {
            role: Role::User,
            content: Content::Text {
                text: "Summarize the conversation above, preserving the items \
                       in the preservation hints."
                    .to_string(),
            },
        });

        Some(CompactionTurn {
            context: AgentContext {
                messages,
                tools: Vec::new(),
                params: ModelParams::default(),
            },
            preserve_hints: request.preserve_hints,
            verification_state: rich,
            messages_removed,
            dropped_messages,
        })
    }

    // `inject_missing_items` is intentionally NOT overridden: the inherited
    // default produces the exact "Your summary is missing these items: …"
    // prompt the `compaction_loop` fixture asserts.

    fn apply_compaction(&self, session: &mut HarnessState, summary: String, dropped: &[Message]) {
        let Some(mut rich) = Self::read_rich_state(session) else {
            // No rich state to apply against — degrade safely; never panic.
            return;
        };
        // `dropped` is the set `prepare_compaction_turn` already computed (carried
        // on the `CompactionTurn`) — one source of truth, no re-derivation.
        let messages_removed = dropped.len() as u32;

        let summary_message = Message {
            role: Role::Assistant,
            content: Content::Text { text: summary },
        };

        // Real token accounting (Known Deviation #2 fix): reclaim the tokens of
        // the messages we drop, net of the summary that replaces them, and clamp
        // to the live budget so `token_budget_used` never underflows. The rich
        // `apply_compaction` (context.rs) decrements `token_budget_used` by this
        // amount, so utilization actually falls below threshold after a
        // compaction and a long session can compact repeatedly.
        let dropped_tokens = estimate_tokens(dropped);
        let summary_tokens = estimate_message_tokens(&summary_message);
        let net_reclaimed = dropped_tokens.saturating_sub(summary_tokens);
        let tokens_reclaimed = net_reclaimed.min(rich.token_budget_used);

        let result = CompactionResult {
            summary_message,
            tokens_reclaimed,
            messages_removed,
        };
        if let Err(err) = self.inner.apply_compaction(&mut rich, result) {
            // Compaction must never halt the loop — log to stderr and swallow.
            // (spore-core has no logging facade dependency; the harness owns
            // structured observability, the adapter only needs to not panic.)
            eprintln!(
                "spore.compaction: rich apply_compaction failed, leaving session unchanged: {err}"
            );
            return;
        }
        Self::write_rich_state(session, &rich);
    }

    fn token_budget_used(&self, session: &HarnessState) -> Option<u32> {
        Self::read_rich_state(session).map(|rich| rich.token_budget_used)
    }
}

/// Ergonomic constructor: turn an `Arc<StandardContextManager>` into the
/// harness-seam adapter for injection into a `HarnessBuilder`.
pub trait HarnessContextManagerExt<M: ModelInterface> {
    /// Wrap `self` as an `Arc<dyn HarnessContextManager>` ready for
    /// [`crate::harness::HarnessBuilder::new`].
    fn into_harness_adapter(self) -> Arc<dyn HarnessContextManager>;
}

impl<M: ModelInterface + 'static> HarnessContextManagerExt<M> for Arc<StandardContextManager<M>> {
    fn into_harness_adapter(self) -> Arc<dyn HarnessContextManager> {
        Arc::new(StandardCompactionAdapter::new(self))
    }
}

/// Seed `extras` with a serialized rich session state. Callers that drive the
/// harness with [`StandardCompactionAdapter`] use this to project the rich
/// state into the harness session before the first turn.
pub fn seed_rich_state(session: &mut HarnessState, rich: &RichSessionState) {
    session.messages = rich.message_history.clone();
    if let Ok(value) = serde_json::to_value(rich) {
        session.extras.insert(RICH_STATE_KEY.to_string(), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_provider::NullCacheProvider;
    use crate::context::CompactionConfig;
    use crate::harness::{
        AggregateUsage, HarnessBuilder, SandboxProvider, SessionId, StandardHarness, TaskId,
        TerminationPolicy, ToolRegistry,
    };
    use crate::model::{
        ModelError, ModelRequest, ModelResponse, ModelStream, ProviderInfo, StopReason, TokenUsage,
    };
    use crate::observability::{InMemoryObservabilityProvider, ObservabilityProvider};
    use std::sync::Mutex;

    // ---- minimal model stub (BoxFut ModelInterface) ----------------------

    struct StubModel;
    impl ModelInterface for StubModel {
        fn call<'a>(
            &'a self,
            _req: ModelRequest,
        ) -> BoxFut<'a, Result<ModelResponse, ModelError>> {
            Box::pin(async move {
            Ok(ModelResponse {
                content: vec![],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            })
            })
        }
        fn call_streaming<'a>(
            &'a self,
            _req: ModelRequest,
        ) -> BoxFut<'a, Result<ModelStream, ModelError>> {
            Box::pin(async move { Err(ModelError::Timeout) })
        }
        fn count_tokens<'a>(
            &'a self,
            _req: &'a ModelRequest,
        ) -> BoxFut<'a, Result<u32, ModelError>> {
            Box::pin(async move { Ok(0) })
        }
        fn provider(&self) -> ProviderInfo {
            ProviderInfo {
                name: "stub".into(),
                model_id: "stub".into(),
                context_window: 200_000,
            }
        }
    }

    struct AlwaysContinue;
    impl TerminationPolicy for AlwaysContinue {
        fn evaluate<'a>(
            &'a self,
            _session: &'a HarnessState,
            _budget: &'a crate::harness::BudgetSnapshot,
        ) -> BoxFut<'a, crate::harness::TerminationDecision> {
            Box::pin(async { crate::harness::TerminationDecision::Continue })
        }
    }

    fn rich_manager() -> Arc<StandardContextManager<StubModel>> {
        let cfg = CompactionConfig {
            threshold: 0.80,
            preserve_recent_n: 2,
            head_tail_tokens: 64,
            offload_path: std::path::PathBuf::from(".spore/offload"),
            max_compaction_attempts: 2,
            context_length: None,
        };
        Arc::new(StandardContextManager::new(
            Arc::new(StubModel),
            Arc::new(NullCacheProvider),
            cfg,
        ))
    }

    fn msg(role: Role, text: &str) -> Message {
        Message {
            role,
            content: Content::Text { text: text.into() },
        }
    }

    fn rich_state(messages: usize, used: u32, limit: u32) -> RichSessionState {
        let mut s = RichSessionState::new(
            SessionId::new("s1"),
            TaskId::new("t1"),
            "deploy the payment service",
        );
        s.window_limit = limit;
        s.token_budget_used = used;
        s.message_history = (0..messages)
            .map(|i| msg(Role::User, &format!("m{i}")))
            .collect();
        s
    }

    fn session_with(rich: &RichSessionState) -> HarnessState {
        let mut s = HarnessState::default();
        seed_rich_state(&mut s, rich);
        s
    }

    // ---- should_compact threshold ---------------------------------------

    #[test]
    fn should_compact_below_threshold_is_false() {
        let adapter = StandardCompactionAdapter::new(rich_manager());
        let session = session_with(&rich_state(10, 70, 100)); // 0.70 < 0.80
        assert!(!adapter.should_compact(&session));
    }

    #[test]
    fn should_compact_at_threshold_is_true() {
        let adapter = StandardCompactionAdapter::new(rich_manager());
        let session = session_with(&rich_state(10, 80, 100)); // 0.80 >= 0.80
        assert!(adapter.should_compact(&session));
    }

    #[test]
    fn should_compact_above_threshold_is_true() {
        let adapter = StandardCompactionAdapter::new(rich_manager());
        let session = session_with(&rich_state(10, 95, 100));
        assert!(adapter.should_compact(&session));
    }

    #[test]
    fn should_compact_false_without_rich_state() {
        let adapter = StandardCompactionAdapter::new(rich_manager());
        let session = HarnessState::default();
        assert!(!adapter.should_compact(&session));
    }

    // ---- prepare_compaction_turn ----------------------------------------

    #[test]
    fn prepare_returns_none_for_short_history() {
        let adapter = StandardCompactionAdapter::new(rich_manager());
        // preserve_recent_n = 2, history = 2 -> nothing to compact.
        let session = session_with(&rich_state(2, 95, 100));
        assert!(adapter.prepare_compaction_turn(&session).is_none());
    }

    #[test]
    fn prepare_projects_hints_state_and_count() {
        let adapter = StandardCompactionAdapter::new(rich_manager());
        let rich = rich_state(10, 95, 100); // 10 - 2 preserved = 8 removed
        let session = session_with(&rich);
        let turn = adapter
            .prepare_compaction_turn(&session)
            .expect("turn present");
        assert_eq!(turn.messages_removed, 8);
        // verification_state mirrors the rich state.
        assert_eq!(
            turn.verification_state.task_instruction,
            rich.task_instruction
        );
        assert_eq!(turn.verification_state.token_budget_used, 95);
        // default hints projected.
        assert!(turn.preserve_hints.keep_current_task_state);
        // the summarization instruction is appended after the compacted msgs.
        assert!(turn
            .context
            .messages
            .iter()
            .any(|m| matches!(&m.content, Content::Text { text } if text.contains("Summarize"))));
    }

    // ---- apply_compaction shrinks the session ---------------------------

    #[test]
    fn apply_compaction_shrinks_session() {
        let adapter = StandardCompactionAdapter::new(rich_manager());
        let mut session = session_with(&rich_state(10, 95, 100));
        let before = session.messages.len();
        let turn = adapter
            .prepare_compaction_turn(&session)
            .expect("turn present");
        adapter.apply_compaction(
            &mut session,
            "summary preserving payment deploy".into(),
            &turn.dropped_messages,
        );
        // 2 preserved + 1 summary = 3.
        assert!(session.messages.len() < before);
        assert_eq!(session.messages.len(), 3);
        // round-tripped rich state also shrank.
        let rich = StandardCompactionAdapter::<StubModel>::read_rich_state(&session).unwrap();
        assert_eq!(rich.message_history.len(), 3);
        assert_eq!(rich.message_history[0].role, Role::Assistant);
    }

    #[test]
    fn apply_compaction_reclaims_real_tokens_and_drops_budget() {
        // Known Deviation #2 fix: dropping messages must reclaim real tokens so
        // `token_budget_used` falls. Seed long, content-heavy messages so the
        // dropped span carries a non-trivial token estimate.
        let adapter = StandardCompactionAdapter::new(rich_manager());
        let mut rich = RichSessionState::new(
            SessionId::new("s1"),
            TaskId::new("t1"),
            "deploy the payment service",
        );
        rich.window_limit = 100;
        rich.token_budget_used = 95;
        // 10 messages of ~80 bytes each ⇒ ~20 tokens each by the chars/4 proxy.
        rich.message_history = (0..10)
            .map(|i| {
                msg(
                    Role::User,
                    &format!(
                        "message number {i} with a fair amount of content to estimate tokens from"
                    ),
                )
            })
            .collect();
        let mut session = session_with(&rich);

        let before = StandardCompactionAdapter::<StubModel>::read_rich_state(&session)
            .unwrap()
            .token_budget_used;
        let turn = adapter
            .prepare_compaction_turn(&session)
            .expect("turn present");
        adapter.apply_compaction(
            &mut session,
            "summary preserving payment deploy".into(),
            &turn.dropped_messages,
        );
        let after = StandardCompactionAdapter::<StubModel>::read_rich_state(&session)
            .unwrap()
            .token_budget_used;

        assert!(
            after < before,
            "token_budget_used must drop after a real reclamation: {before} -> {after}"
        );
        // And the seam reports the post-compaction budget for span stamping.
        assert_eq!(adapter.token_budget_used(&session), Some(after));
    }

    #[test]
    fn apply_compaction_multi_compaction_keeps_dropping_budget() {
        // Healthy multi-compaction: after compacting, growing history and
        // budget again must let a second compaction reclaim more tokens.
        let adapter = StandardCompactionAdapter::new(rich_manager());
        let mk_history = |n: usize| -> Vec<Message> {
            (0..n)
                .map(|i| {
                    msg(
                        Role::User,
                        &format!("turn {i} produced some output worth a handful of tokens here"),
                    )
                })
                .collect()
        };
        let mut rich = RichSessionState::new(
            SessionId::new("s1"),
            TaskId::new("t1"),
            "deploy the payment service",
        );
        rich.window_limit = 100;
        rich.token_budget_used = 95;
        rich.message_history = mk_history(10);
        let mut session = session_with(&rich);

        let turn1 = adapter
            .prepare_compaction_turn(&session)
            .expect("turn present");
        adapter.apply_compaction(
            &mut session,
            "first summary about payment deploy".into(),
            &turn1.dropped_messages,
        );
        let after_first = adapter.token_budget_used(&session).unwrap();
        assert!(after_first < 95);

        // Simulate the session growing again past threshold.
        let mut grown = StandardCompactionAdapter::<StubModel>::read_rich_state(&session).unwrap();
        grown.token_budget_used = 95;
        grown.message_history = mk_history(10);
        seed_rich_state(&mut session, &grown);

        let turn2 = adapter
            .prepare_compaction_turn(&session)
            .expect("turn present");
        adapter.apply_compaction(
            &mut session,
            "second summary about payment deploy".into(),
            &turn2.dropped_messages,
        );
        let after_second = adapter.token_budget_used(&session).unwrap();
        assert!(
            after_second < 95,
            "second compaction also reclaims real tokens"
        );
    }

    #[test]
    fn apply_compaction_swallows_error_without_rich_state() {
        let adapter = StandardCompactionAdapter::new(rich_manager());
        let mut session = HarnessState::default();
        // No rich state -> no-op, no panic.
        adapter.apply_compaction(&mut session, "summary".into(), &[]);
        assert!(session.messages.is_empty());
    }

    // ---- end-to-end mock-model harness test -----------------------------

    /// Agent that returns a fixed summary as a FinalResponse.
    struct SummaryAgent {
        summary: String,
    }
    impl crate::agent::Agent for SummaryAgent {
        fn turn<'a>(&'a self, _ctx: AgentContext) -> BoxFut<'a, crate::agent::TurnResult> {
            let content = self.summary.clone();
            Box::pin(async move {
                crate::agent::TurnResult::FinalResponse {
                    reasoning: None,
                    content,
                    usage: TokenUsage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cache_read_tokens: None,
                        cache_write_tokens: None,
                    },
                }
            })
        }
        fn id(&self) -> crate::agent::AgentId {
            crate::agent::AgentId::new("summary")
        }
    }

    struct NoopTools;
    impl ToolRegistry for NoopTools {
        fn dispatch<'a>(
            &'a self,
            _call: crate::model::ToolCall,
        ) -> BoxFut<'a, crate::harness::ToolOutput> {
            Box::pin(async {
                ToolOutput::Success {
                    content: String::new(),
                    truncated: false,
                }
            })
        }
    }

    struct AllowAll;
    impl SandboxProvider for AllowAll {
        fn validate<'a>(
            &'a self,
            _call: &'a crate::model::ToolCall,
        ) -> BoxFut<'a, Result<(), crate::harness::SandboxViolation>> {
            Box::pin(async { Ok(()) })
        }
    }

    fn build_harness(
        cm: Arc<dyn HarnessContextManager>,
        agent: Arc<dyn crate::agent::Agent>,
        obs: Arc<dyn ObservabilityProvider>,
        verifier: Arc<dyn crate::context::CompactionVerifier>,
        max_attempts: u32,
    ) -> StandardHarness {
        HarnessBuilder::new(
            agent,
            Arc::new(NoopTools),
            Arc::new(AllowAll),
            cm,
            Arc::new(AlwaysContinue),
        )
        .observability(obs)
        .compaction_verifier(verifier)
        .max_compaction_attempts(max_attempts)
        .build()
    }

    #[tokio::test]
    async fn end_to_end_drives_compaction_through_seam() {
        // A summary that contains the key term "payment" so the default
        // KeyTermVerifier (sources task_instruction "deploy the payment
        // service") passes on the first attempt.
        let adapter: Arc<dyn HarnessContextManager> = rich_manager().into_harness_adapter();
        let agent: Arc<dyn crate::agent::Agent> = Arc::new(SummaryAgent {
            summary: "we are working on the deploy of the payment service".into(),
        });
        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let h = build_harness(
            adapter.clone(),
            agent.clone(),
            obs.clone(),
            Arc::new(crate::context::KeyTermVerifier),
            2,
        );

        // Drive utilization over threshold: 10 messages, 95/100 budget.
        let mut session = session_with(&rich_state(10, 95, 100));
        let before = session.messages.len();
        assert!(h.config().context_manager.should_compact(&session));

        let mut usage = AggregateUsage::default();
        let mut span_seq = 0u64;
        h.run_compaction_for_test(
            &mut session,
            &SessionId::new("s1"),
            &TaskId::new("t1"),
            &mut span_seq,
            &mut usage,
            agent,
        )
        .await;

        assert!(session.messages.len() < before, "session shrank");
        assert_eq!(session.messages.len(), 3);
        // a compaction span was emitted (derived metric counts it). The metric
        // accessor needs a recorded outcome to materialize, so stamp one.
        obs.set_session_outcome(
            &SessionId::new("s1"),
            crate::guide_registry::SessionOutcome::Success,
        );
        let metrics = obs
            .get_session_metrics(&SessionId::new("s1"))
            .await
            .expect("metrics present");
        assert_eq!(metrics.compactions, 1, "one compaction span emitted");
        // verification passed first time -> no warn.
        assert!(obs.warn_spans(&SessionId::new("s1")).is_empty());
    }

    // ---- compaction_loop fixture parity (verify->retry->warn) -----------

    #[derive(serde::Deserialize)]
    struct FixtureFile {
        cases: Vec<FixtureCase>,
    }
    #[derive(serde::Deserialize)]
    struct FixtureCase {
        name: String,
        max_compaction_attempts: u32,
        verdicts: Vec<Verdict>,
        expected: Expected,
    }
    #[derive(serde::Deserialize, Clone)]
    struct Verdict {
        passed: bool,
        missing_items: Vec<String>,
    }
    #[derive(serde::Deserialize)]
    struct Expected {
        apply_compaction_calls: u32,
        warn_emitted: bool,
        #[serde(default)]
        retry_injected_missing: Vec<String>,
    }

    /// Verifier scripted from the fixture's `verdicts` (last entry repeats).
    struct FixtureVerifier {
        verdicts: Vec<Verdict>,
        idx: Mutex<usize>,
    }
    impl crate::context::CompactionVerifier for FixtureVerifier {
        fn verify(
            &self,
            _summary: &str,
            _hints: &crate::context::CompactionPreserveHints,
            _state: &RichSessionState,
        ) -> crate::context::CompactionVerificationResult {
            let mut idx = self.idx.lock().unwrap();
            let v = self
                .verdicts
                .get(*idx)
                .or_else(|| self.verdicts.last())
                .cloned()
                .unwrap();
            *idx += 1;
            crate::context::CompactionVerificationResult {
                passed: v.passed,
                missing_items: v.missing_items,
                detail: "fixture".into(),
            }
        }
    }

    /// Agent that records the contexts it sees so we can assert retry injection.
    struct RecordingAgent {
        seen: Mutex<Vec<AgentContext>>,
    }
    impl crate::agent::Agent for RecordingAgent {
        fn turn<'a>(&'a self, ctx: AgentContext) -> BoxFut<'a, crate::agent::TurnResult> {
            self.seen.lock().unwrap().push(ctx);
            Box::pin(async {
                crate::agent::TurnResult::FinalResponse {
                    reasoning: None,
                    content: "summary".into(),
                    usage: TokenUsage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cache_read_tokens: None,
                        cache_write_tokens: None,
                    },
                }
            })
        }
        fn id(&self) -> crate::agent::AgentId {
            crate::agent::AgentId::new("rec")
        }
    }

    #[tokio::test]
    async fn compaction_loop_fixture_parity_with_real_adapter() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../fixtures/compaction_loop/cases.json"
        );
        let raw = std::fs::read_to_string(path).expect("read fixture");
        let fixture: FixtureFile = serde_json::from_str(&raw).expect("parse fixture");

        for case in &fixture.cases {
            let adapter: Arc<dyn HarnessContextManager> = rich_manager().into_harness_adapter();
            let agent = Arc::new(RecordingAgent {
                seen: Mutex::new(Vec::new()),
            });
            let obs = Arc::new(InMemoryObservabilityProvider::new());
            let verifier = Arc::new(FixtureVerifier {
                verdicts: case.verdicts.clone(),
                idx: Mutex::new(0),
            });
            let h = build_harness(
                adapter,
                agent.clone(),
                obs.clone(),
                verifier,
                case.max_compaction_attempts,
            );

            // 10 messages, over threshold -> a real CompactionTurn is offered.
            let mut session = session_with(&rich_state(10, 95, 100));
            let mut usage = AggregateUsage::default();
            let mut span_seq = 0u64;
            let worker: Arc<dyn crate::agent::Agent> = agent.clone();
            h.run_compaction_for_test(
                &mut session,
                &SessionId::new("s1"),
                &TaskId::new("t1"),
                &mut span_seq,
                &mut usage,
                worker,
            )
            .await;

            // apply_compaction always runs exactly once (accept or accept-anyway).
            let rich = StandardCompactionAdapter::<StubModel>::read_rich_state(&session).unwrap();
            assert_eq!(rich.message_history.len(), 3, "case {}: applied", case.name);
            assert_eq!(
                case.expected.apply_compaction_calls, 1,
                "case {}: fixture sanity",
                case.name
            );

            // warn parity.
            let warns = obs.warn_spans(&SessionId::new("s1"));
            assert_eq!(
                warns.is_empty(),
                !case.expected.warn_emitted,
                "case {}: warn parity",
                case.name
            );

            // retry injection parity: if the fixture expects an injected retry,
            // the second context the agent saw carries the missing-items prompt.
            if !case.expected.retry_injected_missing.is_empty() {
                let seen = agent.seen.lock().unwrap();
                assert!(seen.len() >= 2, "case {}: retry occurred", case.name);
                let retry = &seen[1];
                for item in &case.expected.retry_injected_missing {
                    assert!(
                        retry.messages.iter().any(|m| matches!(&m.content,
                            Content::Text { text } if text.contains("missing these items")
                                && text.contains(item.as_str()))),
                        "case {}: retry carries missing item {item}",
                        case.name
                    );
                }
            }
        }
    }
}
