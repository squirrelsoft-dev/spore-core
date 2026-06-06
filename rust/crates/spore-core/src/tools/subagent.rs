//! SubagentTool — wraps a child `Harness` and exposes it as a `Tool`.
//!
//! Per spec (issue #5), subagents cannot spawn their own subagents. The
//! restriction is enforced at construction time by inspecting the child's
//! `ToolRegistry` via `has_subagent_tools()`.
//!
//! ## Mid-loop consult mediation — seam A1 (issue #114)
//!
//! This is the ORCHESTRATOR side of the consult primitive (the worker side and
//! the type/resume seams live in [`crate::harness`]; see its consult doc-hub).
//! `SubagentTool::execute` drives the FULL consult cycle internally:
//!
//! 1. It runs the child worker.
//! 2. On a child [`RunResult::Consult`] it does the mediation ITSELF — it never
//!    bubbles the consult to the parent orchestrator's model. It routes by
//!    `request.kind` to the matching [`ConsultHandlerEntry`] in its
//!    `consult_handlers` map, checks the per-kind budget, runs the handler
//!    harness on the request, builds a [`ConsultResponse::Answer`] from the
//!    handler's output, and calls `child.resume_consult(..)` to continue the
//!    worker.
//! 3. It repeats until the child reaches a terminal result, then returns the
//!    appropriate terminal [`ToolOutput`] to the parent.
//!
//! Rules enforced here: R2 (mediate, do not bubble), R3 (route by kind, no
//! parent model, parent sees `Success`), R4 (per-kind budget), R5a (`SoftFail`
//! overflow → `BudgetExhausted` resume), R5b (`EscalateToHuman` overflow →
//! `ToolOutput::WaitingForHuman`), R6 (no matching kind → `Escalate`), R7
//! (depth-1: the handler is the orchestrator's direct child, run via
//! `handler.run(..)`, never nested under the worker).
//!
//! The handlers reach this tool through [`SubagentTool::with_consult_handlers`]
//! (the orchestrator builds them from its `HarnessConfig::consult_handlers`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::harness::{
    BoxFut, ChildPausedState, ConsultHandlerEntry, ConsultOverflowPolicy, ConsultRequest,
    ConsultResponse, Harness, HarnessRunOptions, HarnessSignal, HumanRequest, PausedState,
    RunResult, SandboxProvider, SessionId, SessionState, StreamEvent, Task, ToolOutput,
};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolRegistry};

/// How the subagent inherits / does not inherit context from its parent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContextSharing {
    Isolated,
    SharedSession { session_id: SessionId },
    SummaryHandoff { summary: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum BuildError {
    #[error("invalid configuration: {reason}")]
    InvalidConfiguration { reason: String },
}

pub struct SubagentTool {
    name: String,
    description: String,
    input_schema: Value,
    timeout: Duration,
    context_sharing: ContextSharing,
    harness: Arc<dyn Harness>,
    /// Per-kind consult handlers (issue #114, seam A1). Empty (the default)
    /// means consults are NOT mediated here — a child `RunResult::Consult`
    /// degrades gracefully per R6 (no matching kind → `Escalate`). Populated
    /// via [`with_consult_handlers`](Self::with_consult_handlers).
    consult_handlers: HashMap<String, ConsultHandlerEntry>,
    /// Optional stream sink for the CHILD run's `StreamEvent`s, so a host can
    /// observe the subagent's internal turns/tool-calls (otherwise the child
    /// loop is opaque — only its final answer surfaces). Stored as an `Arc` (not
    /// the `Box`-based [`StreamSink`](crate::harness::StreamSink)) because
    /// `execute` runs once per dispatch and must hand each child run a fresh
    /// sink. Installed via [`with_stream`](Self::with_stream).
    child_stream: Option<Arc<dyn Fn(StreamEvent) + Send + Sync>>,
}

impl SubagentTool {
    /// Construct a SubagentTool. Returns `BuildError::InvalidConfiguration`
    /// if the child registry already contains subagent tools (depth-1 rule).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        timeout: Duration,
        context_sharing: ContextSharing,
        harness: Arc<dyn Harness>,
        child_registry: &dyn ToolRegistry,
    ) -> Result<Self, BuildError> {
        if child_registry.has_subagent_tools() {
            return Err(BuildError::InvalidConfiguration {
                reason: "child harness must not contain SubagentTool (depth-1 rule)".into(),
            });
        }
        Ok(Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            timeout,
            context_sharing,
            harness,
            consult_handlers: HashMap::new(),
            child_stream: None,
        })
    }

    /// Install a stream sink for the CHILD run, making the subagent's internal
    /// turns and tool calls observable to the host. Each dispatch wraps this in
    /// a fresh [`StreamSink`](crate::harness::StreamSink) for the child run.
    pub fn with_stream(mut self, sink: Arc<dyn Fn(StreamEvent) + Send + Sync>) -> Self {
        self.child_stream = Some(sink);
        self
    }

    /// Install the per-kind consult handlers (issue #114, seam A1). Typically
    /// the orchestrator passes a clone of its
    /// [`HarnessConfig::consult_handlers`](crate::harness::HarnessConfig::consult_handlers).
    /// With handlers installed, this tool MEDIATES a child consult internally
    /// (R2/R3) instead of letting it surface; without them, a child consult
    /// degrades to `Escalate` (R6).
    pub fn with_consult_handlers(
        mut self,
        consult_handlers: HashMap<String, ConsultHandlerEntry>,
    ) -> Self {
        self.consult_handlers = consult_handlers;
        self
    }

    pub fn description(&self) -> &str {
        &self.description
    }
    pub fn input_schema(&self) -> &Value {
        &self.input_schema
    }
    pub fn context_sharing(&self) -> &ContextSharing {
        &self.context_sharing
    }
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

impl Tool for SubagentTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn is_subagent_tool(&self) -> bool {
        true
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let instruction = match call.input.get("instruction").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => {
                    return ToolOutput::Error {
                        message: "invalid parameters: missing `instruction`".into(),
                        recoverable: true,
                    }
                }
            };

            // Build task and (where applicable) seed session state.
            let (session_id, seeded_session): (SessionId, Option<SessionState>) =
                match &self.context_sharing {
                    ContextSharing::Isolated => (SessionId::generate(), None),
                    ContextSharing::SharedSession { session_id } => (session_id.clone(), None),
                    ContextSharing::SummaryHandoff { summary } => {
                        // Inject the summary as a synthetic extras entry; the
                        // ContextManager (issue #7) decides how to surface it.
                        let mut state = SessionState::default();
                        state.extras.insert(
                            "subagent_handoff_summary".into(),
                            serde_json::Value::String(summary.clone()),
                        );
                        (SessionId::generate(), Some(state))
                    }
                };

            let task = Task::new(
                instruction,
                session_id,
                crate::harness::LoopStrategy::ReAct { max_iterations: 16 },
            );

            let mut options = HarnessRunOptions::new(task);
            options.session_state = seeded_session;
            // Make the child loop observable: each dispatch gets a fresh
            // `Box` sink that forwards to the shared `Arc` installed via
            // `with_stream`.
            if let Some(sink) = self.child_stream.clone() {
                options = options.with_stream(Box::new(move |event| sink(event)));
            }

            let fut = self.harness.run(options);
            let mut result = match tokio::time::timeout(self.timeout, fut).await {
                Ok(r) => r,
                Err(_) => {
                    return ToolOutput::Error {
                        message: format!("subagent timed out after {}s", self.timeout.as_secs()),
                        recoverable: true,
                    }
                }
            };

            // Per-kind consult counters (issue #114, R4). Each consult of a
            // given kind decrements its remaining budget; the (budget+1)th
            // triggers the overflow policy.
            let mut consult_counts: HashMap<String, u32> = HashMap::new();

            // A1 mediation loop: drive the full consult cycle internally. On a
            // child `RunResult::Consult`, mediate (route → run handler → resume)
            // and continue until the child reaches a terminal result.
            loop {
                match result {
                    RunResult::Success { output, .. } => {
                        return ToolOutput::Success {
                            content: output,
                            truncated: false,
                        };
                    }
                    RunResult::Failure { reason, .. } => {
                        return ToolOutput::Error {
                            message: format!("subagent failed: {reason:?}"),
                            recoverable: true,
                        };
                    }
                    RunResult::WaitingForHuman { state, request } => {
                        let child = child_state_from_paused(*state, call.id.clone());
                        return ToolOutput::WaitingForHuman {
                            child_state: Box::new(child),
                            request,
                        };
                    }
                    // A subagent escalation (issue #80) propagates as a tool-side
                    // escalation: the parent harness terminates cleanly and hands
                    // the signal up to its own caller.
                    RunResult::Escalate { signal, .. } => {
                        return ToolOutput::Escalate { signal };
                    }
                    // Mid-loop consult (issue #114, R2): mediate it here — never
                    // bubble it to the parent orchestrator's model.
                    RunResult::Consult { request, state, .. } => {
                        match self
                            .mediate_consult(*state, request, &mut consult_counts, &call.id)
                            .await
                        {
                            // The handler answered (or soft-failed): resume the
                            // worker and loop again on the new result (R3/R5a).
                            MediateOutcome::Resume(next) => {
                                result = next;
                                continue;
                            }
                            // Terminal mapping surfaced to the parent (R5b/R6).
                            MediateOutcome::Terminal(output) => return output,
                        }
                    }
                }
            }
        })
    }
}

/// Outcome of one mediation step (issue #114).
enum MediateOutcome {
    /// The worker was resumed; carry the new `RunResult` for the next loop turn.
    Resume(RunResult),
    /// A terminal `ToolOutput` to surface to the parent (overflow-escalate or
    /// misconfiguration).
    Terminal(ToolOutput),
}

impl SubagentTool {
    /// Mediate one child consult (issue #114, seam A1). Routes by `kind`,
    /// enforces the per-kind budget, runs the handler as the ORCHESTRATOR's
    /// direct child (R7), and resumes the worker — OR applies the overflow
    /// policy / graceful degradation.
    async fn mediate_consult(
        &self,
        state: PausedState,
        request: ConsultRequest,
        counts: &mut HashMap<String, u32>,
        parent_call_id: &str,
    ) -> MediateOutcome {
        // R6: no matching handler (empty map or unknown kind) → Escalate. Loud,
        // not silent. The parent harness terminates cleanly.
        let entry = match self.consult_handlers.get(&request.kind) {
            Some(e) => e,
            None => {
                return MediateOutcome::Terminal(ToolOutput::Escalate {
                    signal: HarnessSignal::Abort {
                        reason: format!(
                            "no consult handler registered for kind {:?}",
                            request.kind
                        ),
                    },
                });
            }
        };

        // R4: per-kind budget. `used` is the number of consults of this kind
        // ALREADY mediated. The handler runs while `used < budget`; the
        // (budget+1)th consult overflows.
        let used = counts.entry(request.kind.clone()).or_insert(0);
        if *used >= entry.budget {
            // R5: overflow policy.
            return match entry.overflow {
                // R5a: resume the worker with a BudgetExhausted response so it
                // finishes with what it has.
                ConsultOverflowPolicy::SoftFail => {
                    let response = ConsultResponse::BudgetExhausted {
                        message: format!(
                            "consult budget for kind {:?} exhausted; proceed without further help",
                            request.kind
                        ),
                    };
                    let next = self.harness.resume_consult(state, response, None).await;
                    MediateOutcome::Resume(next)
                }
                // R5b: convert the over-budget consult into a human pause so the
                // host decides. The parent sees ToolOutput::WaitingForHuman.
                ConsultOverflowPolicy::EscalateToHuman => {
                    let child = child_state_from_paused(state, parent_call_id.to_string());
                    MediateOutcome::Terminal(ToolOutput::WaitingForHuman {
                        child_state: Box::new(child),
                        request: HumanRequest::Review {
                            content: format!(
                                "consult budget for kind {:?} exhausted. situation: {} | question: {}",
                                request.kind, request.situation, request.question
                            ),
                        },
                    })
                }
            };
        }

        // R3/R7: run the handler harness as the orchestrator's direct child
        // (depth-1), WITHOUT the orchestrator model. The handler's instruction
        // is the consult request rendered to text.
        *used += 1;
        let instruction = render_consult_instruction(&request);
        let task = Task::new(
            instruction,
            SessionId::generate(),
            crate::harness::LoopStrategy::ReAct { max_iterations: 16 },
        );
        let handler_result = entry.handler.run(HarnessRunOptions::new(task)).await;
        let answer = match handler_result {
            RunResult::Success { output, .. } => output,
            // A handler that does not cleanly complete still must not stall the
            // worker — feed its failure text back as the consult answer so the
            // worker can adapt. (The orchestrator model is never involved.)
            other => format!("consult handler did not complete cleanly: {other:?}"),
        };
        let response = ConsultResponse::Answer { text: answer };
        let next = self.harness.resume_consult(state, response, None).await;
        MediateOutcome::Resume(next)
    }
}

/// Render a [`ConsultRequest`] to a handler instruction string (issue #114).
fn render_consult_instruction(request: &ConsultRequest) -> String {
    format!(
        "A worker agent is requesting help (kind: {kind}).\n\nSituation: {situation}\n\nAttempts so far: {attempts}\n\nQuestion: {question}",
        kind = request.kind,
        situation = request.situation,
        attempts = request.attempts,
        question = request.question,
    )
}

fn child_state_from_paused(
    state: crate::harness::PausedState,
    parent_tool_call_id: String,
) -> ChildPausedState {
    ChildPausedState {
        session_id: state.session_id,
        task_id: state.task_id,
        turn_number: state.turn_number,
        session_state: state.session_state,
        pending_tool_calls: state.pending_tool_calls,
        approved_results: state.approved_results,
        human_request: state.human_request,
        task: state.task,
        budget_used: state.budget_used,
        parent_tool_call_id,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{
        AggregateUsage, ConsultHandlerEntry, ConsultOverflowPolicy, ConsultRequest,
        ConsultResponse, HaltReason, HarnessRunOptions, HumanRequest, HumanResponse, PausedState,
        RunResult,
    };
    use crate::tool_registry::mock::{test_ctx, AllowAllSandbox};
    use crate::tool_registry::StandardToolRegistry;
    use serde_json::json;
    use std::sync::Mutex;

    /// Scripted harness — returns a queue of `RunResult`s. `resume_consult`
    /// logs the `ConsultResponse` it was resumed with and pops the next result.
    struct ScriptedHarness {
        results: Mutex<Vec<RunResult>>,
        resume_log: Arc<Mutex<Vec<ConsultResponse>>>,
    }
    impl ScriptedHarness {
        fn new(rs: Vec<RunResult>) -> Self {
            Self {
                results: Mutex::new(rs),
                resume_log: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }
    impl Harness for ScriptedHarness {
        fn run<'a>(&'a self, _opts: HarnessRunOptions) -> BoxFut<'a, RunResult> {
            Box::pin(async move {
                let mut g = self.results.lock().unwrap();
                if g.is_empty() {
                    return RunResult::Failure {
                        reason: HaltReason::AgentError {
                            error: crate::agent::AgentError::EmptyResponse,
                        },
                        session_id: SessionId::new("s"),
                        usage: AggregateUsage::default(),
                        turns: 0,
                        session_state: SessionState::default(),
                    };
                }
                g.remove(0)
            })
        }
        fn resume<'a>(
            &'a self,
            _state: PausedState,
            _response: HumanResponse,
            _on_stream: Option<crate::harness::StreamSink>,
        ) -> BoxFut<'a, RunResult> {
            Box::pin(async move {
                RunResult::Failure {
                    reason: HaltReason::HumanHalted,
                    session_id: SessionId::new("s"),
                    usage: AggregateUsage::default(),
                    turns: 0,
                    session_state: SessionState::default(),
                }
            })
        }
        fn resume_consult<'a>(
            &'a self,
            _state: PausedState,
            response: ConsultResponse,
            _on_stream: Option<crate::harness::StreamSink>,
        ) -> BoxFut<'a, RunResult> {
            // Record the response text and pop the next scripted RunResult — this
            // lets a test assert exactly what the worker was resumed with.
            self.resume_log.lock().unwrap().push(response);
            Box::pin(async move {
                let mut g = self.results.lock().unwrap();
                if g.is_empty() {
                    return RunResult::Success {
                        output: "child done after consult".into(),
                        session_id: SessionId::new("s"),
                        usage: AggregateUsage::default(),
                        turns: 1,
                        session_state: SessionState::default(),
                    };
                }
                g.remove(0)
            })
        }
    }

    /// A handler harness that records each instruction it is run with and
    /// returns a fixed answer. Used to assert depth-1 routing (R3/R7).
    struct RecordingHandler {
        answer: String,
        seen: Arc<Mutex<Vec<String>>>,
    }
    impl Harness for RecordingHandler {
        fn run<'a>(&'a self, opts: HarnessRunOptions) -> BoxFut<'a, RunResult> {
            self.seen
                .lock()
                .unwrap()
                .push(opts.task.instruction.clone());
            let answer = self.answer.clone();
            Box::pin(async move {
                RunResult::Success {
                    output: answer,
                    session_id: SessionId::new("handler"),
                    usage: AggregateUsage::default(),
                    turns: 1,
                    session_state: SessionState::default(),
                }
            })
        }
        fn resume<'a>(
            &'a self,
            _s: PausedState,
            _r: HumanResponse,
            _o: Option<crate::harness::StreamSink>,
        ) -> BoxFut<'a, RunResult> {
            Box::pin(async move {
                RunResult::Failure {
                    reason: HaltReason::HumanHalted,
                    session_id: SessionId::new("handler"),
                    usage: AggregateUsage::default(),
                    turns: 0,
                    session_state: SessionState::default(),
                }
            })
        }
    }

    fn consult_paused() -> PausedState {
        PausedState {
            session_id: SessionId::new("worker"),
            task_id: crate::harness::TaskId::new("t"),
            turn_number: 1,
            session_state: SessionState::default(),
            pending_tool_calls: vec![ToolCall {
                id: "consult-call".into(),
                name: "ask_advice".into(),
                input: serde_json::json!({"kind": "advice"}),
            }],
            approved_results: vec![],
            human_request: None,
            task: Task::new(
                "audit",
                SessionId::new("worker"),
                crate::harness::LoopStrategy::ReAct { max_iterations: 4 },
            ),
            budget_used: crate::harness::BudgetSnapshot::default(),
            child_state: None,
        }
    }

    fn consult_request(kind: &str) -> ConsultRequest {
        ConsultRequest {
            kind: kind.into(),
            situation: "drowning".into(),
            attempts: 2,
            question: "what now?".into(),
        }
    }

    fn consult_result(kind: &str) -> RunResult {
        RunResult::Consult {
            request: consult_request(kind),
            state: Box::new(consult_paused()),
            session_id: SessionId::new("worker"),
            usage: AggregateUsage::default(),
            turns: 1,
        }
    }

    fn handlers(
        kind: &str,
        answer: &str,
        budget: u32,
        overflow: ConsultOverflowPolicy,
        seen: Arc<Mutex<Vec<String>>>,
    ) -> std::collections::HashMap<String, ConsultHandlerEntry> {
        let mut m = std::collections::HashMap::new();
        m.insert(
            kind.to_string(),
            ConsultHandlerEntry {
                handler: Arc::new(RecordingHandler {
                    answer: answer.into(),
                    seen,
                }),
                budget,
                overflow,
            },
        );
        m
    }

    fn call(input: Value) -> ToolCall {
        ToolCall {
            id: "parent-call-1".into(),
            name: "subagent".into(),
            input,
        }
    }

    #[tokio::test]
    async fn subagent_success_maps_to_tool_success() {
        let h = Arc::new(ScriptedHarness::new(vec![RunResult::Success {
            output: "child done".into(),
            session_id: SessionId::new("s"),
            usage: AggregateUsage::default(),
            turns: 1,
            session_state: SessionState::default(),
        }]));
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(5),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap();
        let sb = AllowAllSandbox;
        let r = sub
            .execute(&call(json!({"instruction": "do it"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, "child done"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn subagent_failure_maps_to_recoverable_error() {
        let h = Arc::new(ScriptedHarness::new(vec![RunResult::Failure {
            reason: HaltReason::HumanHalted,
            session_id: SessionId::new("s"),
            usage: AggregateUsage::default(),
            turns: 1,
            session_state: SessionState::default(),
        }]));
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(5),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap();
        let sb = AllowAllSandbox;
        let r = sub
            .execute(&call(json!({"instruction": "x"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn subagent_waiting_for_human_propagates_with_parent_call_id() {
        let paused = PausedState {
            session_id: SessionId::new("s"),
            task_id: crate::harness::TaskId::new("t"),
            turn_number: 1,
            session_state: SessionState::default(),
            pending_tool_calls: vec![],
            approved_results: vec![],
            human_request: Some(HumanRequest::Clarification {
                question: "yes?".into(),
                options: None,
            }),
            task: Task::new(
                "x",
                SessionId::new("s"),
                crate::harness::LoopStrategy::ReAct { max_iterations: 1 },
            ),
            budget_used: crate::harness::BudgetSnapshot::default(),
            child_state: None,
        };
        let h = Arc::new(ScriptedHarness::new(vec![RunResult::WaitingForHuman {
            state: Box::new(paused),
            request: HumanRequest::Clarification {
                question: "yes?".into(),
                options: None,
            },
        }]));
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(5),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap();
        let sb = AllowAllSandbox;
        let r = sub
            .execute(&call(json!({"instruction": "x"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::WaitingForHuman { child_state, .. } => {
                assert_eq!(child_state.parent_tool_call_id, "parent-call-1");
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn construction_rejects_child_with_subagent_tools() {
        use crate::tool_registry::mock::SubagentMock;
        use crate::tool_registry::{ToolAnnotations, ToolSchema};
        let child_reg = StandardToolRegistry::new();
        child_reg
            .register(
                Box::new(SubagentMock::new("nested")),
                ToolSchema {
                    name: "nested".into(),
                    description: "n".into(),
                    parameters: json!({"type": "object"}),
                    annotations: ToolAnnotations::default(),
                },
            )
            .unwrap();
        let h = Arc::new(ScriptedHarness::new(vec![]));
        let r = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(1),
            ContextSharing::Isolated,
            h,
            &child_reg,
        );
        assert!(matches!(r, Err(BuildError::InvalidConfiguration { .. })));
    }

    // R2/R3: child Consult is MEDIATED here (not bubbled). With a registered
    // handler, the handler runs (no parent model), the worker is resumed, and
    // the parent ultimately sees Success.
    #[tokio::test]
    async fn consult_is_mediated_and_resumed_to_success() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        // First run() => Consult; resume_consult (default scripted) => Success.
        let h = Arc::new(ScriptedHarness::new(vec![consult_result("advice")]));
        let resume_log = h.resume_log.clone();
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(5),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap()
        .with_consult_handlers(handlers(
            "advice",
            "try plan B",
            3,
            ConsultOverflowPolicy::SoftFail,
            seen.clone(),
        ));
        let sb = AllowAllSandbox;
        let r = sub
            .execute(&call(json!({"instruction": "x"})), &sb, &test_ctx())
            .await;
        // R3: parent sees Success (the consult never reached its model).
        match r {
            ToolOutput::Success { content, .. } => {
                assert_eq!(content, "child done after consult")
            }
            other => panic!("expected Success, got {other:?}"),
        }
        // R3/R7: the handler ran exactly once, on the rendered consult request.
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert!(seen[0].contains("advice"));
        assert!(seen[0].contains("what now?"));
        // R3: worker resumed with the handler's answer.
        let log = resume_log.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(
            log[0],
            ConsultResponse::Answer {
                text: "try plan B".into()
            }
        );
    }

    // R4 + R5a: handler runs up to `budget` times; the (budget+1)th consult
    // overflows. With SoftFail, the worker is resumed with BudgetExhausted and
    // finishes.
    #[tokio::test]
    async fn budget_overflow_soft_fail_resumes_with_budget_exhausted() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        // budget = 1: run() => Consult; resume_consult => Consult again
        // (over-budget); resume_consult => Success.
        let h = Arc::new(ScriptedHarness::new(vec![
            consult_result("advice"),
            consult_result("advice"),
        ]));
        let resume_log = h.resume_log.clone();
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(5),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap()
        .with_consult_handlers(handlers(
            "advice",
            "advice answer",
            1,
            ConsultOverflowPolicy::SoftFail,
            seen.clone(),
        ));
        let sb = AllowAllSandbox;
        let r = sub
            .execute(&call(json!({"instruction": "x"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Success { .. } => {}
            other => panic!("expected Success, got {other:?}"),
        }
        // R4: handler ran exactly once (budget = 1).
        assert_eq!(seen.lock().unwrap().len(), 1);
        // R5a: first resume = Answer, second resume = BudgetExhausted.
        let log = resume_log.lock().unwrap();
        assert_eq!(log.len(), 2);
        assert!(matches!(log[0], ConsultResponse::Answer { .. }));
        assert!(matches!(log[1], ConsultResponse::BudgetExhausted { .. }));
    }

    // R5b: budget overflow with EscalateToHuman → ToolOutput::WaitingForHuman.
    #[tokio::test]
    async fn budget_overflow_escalate_to_human() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        // budget = 0: the FIRST consult is already over budget → escalate.
        let h = Arc::new(ScriptedHarness::new(vec![consult_result("advice")]));
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(5),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap()
        .with_consult_handlers(handlers(
            "advice",
            "x",
            0,
            ConsultOverflowPolicy::EscalateToHuman,
            seen.clone(),
        ));
        let sb = AllowAllSandbox;
        let r = sub
            .execute(&call(json!({"instruction": "x"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::WaitingForHuman {
                child_state,
                request,
            } => {
                assert_eq!(child_state.parent_tool_call_id, "parent-call-1");
                assert!(matches!(request, HumanRequest::Review { .. }));
            }
            other => panic!("expected WaitingForHuman, got {other:?}"),
        }
        // Handler never ran (over budget from the start).
        assert_eq!(seen.lock().unwrap().len(), 0);
    }

    // R6: a consult with NO matching handler (map present, wrong kind) →
    // ToolOutput::Escalate (loud, not silent).
    #[tokio::test]
    async fn consult_no_matching_kind_escalates() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let h = Arc::new(ScriptedHarness::new(vec![consult_result("research")]));
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(5),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap()
        .with_consult_handlers(handlers(
            "advice",
            "x",
            3,
            ConsultOverflowPolicy::SoftFail,
            seen,
        ));
        let sb = AllowAllSandbox;
        let r = sub
            .execute(&call(json!({"instruction": "x"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Escalate {
                signal: HarnessSignal::Abort { reason },
            } => assert!(reason.contains("research")),
            other => panic!("expected Escalate, got {other:?}"),
        }
    }

    // R6 (degradation): with NO handlers installed at all, a child consult is
    // treated as the no-matching-kind case → Escalate.
    #[tokio::test]
    async fn consult_with_no_handlers_escalates() {
        let h = Arc::new(ScriptedHarness::new(vec![consult_result("advice")]));
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(5),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap();
        let sb = AllowAllSandbox;
        let r = sub
            .execute(&call(json!({"instruction": "x"})), &sb, &test_ctx())
            .await;
        assert!(matches!(r, ToolOutput::Escalate { .. }));
    }

    #[tokio::test]
    async fn missing_instruction_returns_recoverable_error() {
        let h = Arc::new(ScriptedHarness::new(vec![]));
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(1),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap();
        let sb = AllowAllSandbox;
        let r = sub.execute(&call(json!({})), &sb, &test_ctx()).await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }
}
