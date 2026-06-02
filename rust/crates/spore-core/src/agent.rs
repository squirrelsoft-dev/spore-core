//! Agent — executes a single turn against a `ModelInterface`.
//!
//! The [`Agent`] trait is **dyn-compatible**: `turn` returns a hand-rolled
//! [`BoxFut`] (`Pin<Box<dyn Future + Send>>`) rather than using an async fn,
//! matching every other component trait in `harness.rs`. The harness therefore
//! holds the agent erased behind `Arc<dyn Agent>` (see issue #45).
//!
//! Implements issue #2. The agent is **one turn**: it accepts a fully
//! assembled [`Context`] (produced upstream by the `ContextManager`, issue #7)
//! and returns a [`TurnResult`] classifying what the model wants to do next.
//!
//! ## What this component does
//!
//! - Translate `Context` → [`crate::model::ModelRequest`]
//! - Invoke `ModelInterface::call`
//! - Classify the response as `ToolCallRequested`, `FinalResponse`, or `Error`
//!
//! ## What this component does NOT do
//!
//! - Assemble context (that is the `ContextManager`'s job — issue #7)
//! - Execute tool calls or dispatch them (the harness loop dispatches via
//!   `ToolRegistry` — issues #3, #4)
//! - Validate tool call parameters against tool schemas (the `ToolRegistry`
//!   does that on dispatch — issue #4)
//! - Decide termination (`TerminationPolicy` — issue #13)
//! - Retry on transient errors (lives in the `ModelInterface` impl)
//!
//! ## Rules enforced here
//!
//! 1. One call to [`Agent::turn`] performs exactly one model call.
//! 2. `TurnResult::ToolCallRequested` may carry **multiple** tool calls when
//!    the model requests parallel tool use; the harness dispatches all of
//!    them.
//! 3. A response that yields neither text nor tool calls is reported as
//!    [`AgentError::EmptyResponse`] — never silently swallowed.
//! 4. Classification uses the model's `stop_reason`:
//!    - `StopReason::ToolUse` → `ToolCallRequested` (all `ToolUse` blocks).
//!    - `StopReason::EndTurn | MaxTokens | StopSequence` → `FinalResponse`
//!      (concatenated `Text` blocks).
//! 5. `ModelError` is surfaced wrapped in `AgentError::ModelError`, with any
//!    partial usage information preserved.
//!
//! ## Delta-level streaming (issue #103)
//!
//! New surface added by #103:
//!
//! - [`AgentStreamSink`] — an owned callback sink of **raw**
//!   [`crate::model::StreamEvent`] values. Per the resolved spec decision
//!   **Q1**, the agent emits raw model stream events and does NOT depend on the
//!   harness `StreamEvent` type. The harness owns the
//!   `model::StreamEvent → harness::StreamEvent` mapping.
//! - [`Agent::turn_streaming`] — a defaulted trait method that takes an
//!   `AgentStreamSink`. The default ignores the sink and delegates to
//!   [`Agent::turn`], so every existing `Agent` impl (e.g. `MockAgent`)
//!   compiles unchanged. [`ModelAgent`] overrides it to call
//!   [`crate::model::ModelInterface::call_streaming`], forward each
//!   `model::StreamEvent` to the sink, accumulate the response, and run the
//!   exact same classification logic as [`Agent::turn`].
//! - `reasoning: Option<String>` on the `FinalResponse` and `ToolCallRequested`
//!   variants of [`TurnResult`] (resolved spec decision **Q4**, option (a)).
//!   Thinking is streamed as raw `ThinkingDelta` events AND accumulated into
//!   this field. We do NOT add a `Content::Thinking` variant nor preserve
//!   thinking in `SessionState.messages` — that is deferred to follow-on issue
//!   **#104**. The non-streaming `turn` path also populates `reasoning` now,
//!   so it is no longer "observability only".
//!
//! `turn` and `turn_streaming` share [`classify_response`] so there is exactly
//! one classification code path.
//!
//! ### Tool name / id in streamed turns
//!
//! `model::StreamEvent::ToolUseStart { index, id, name }` carries the tool name
//! and call id at block start — every provider emits it from its block-start
//! frame (Anthropic `content_block_start`, Ollama / OpenAI's first `tool_calls`
//! chunk) — followed by `ToolUseDelta` for the argument JSON. The streaming
//! accumulator records the name/id from `ToolUseStart`, so a tool call
//! reconstructed from a stream is identical to one from the coarse response.
//!
//! ## Cross-language note
//!
//! The shape of `TurnResult`, `AgentError`, and `Context` is mirrored
//! byte-for-byte in the TypeScript / Python / Go packages. The Rust
//! implementation is the spec reference (`/.claude/skills/implement`).

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::harness::BoxFut;
use crate::model::{
    ContentBlock, Message, ModelError, ModelInterface, ModelParams, ModelRequest, ModelResponse,
    StopReason, StreamEvent, TokenUsage, ToolCall, ToolSchema,
};

// ============================================================================
// Identity
// ============================================================================

/// Caller-assigned agent configuration name, used to correlate turns in
/// traces when multiple agents run in the same session (e.g. an initializer
/// agent and a coding agent).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ============================================================================
// Context — the assembled input handed to the agent
// ============================================================================

/// Fully assembled per-turn input produced by the `ContextManager`
/// (issue #7).
///
/// The agent never modifies this — it is treated as an immutable snapshot.
/// Once issue #7 lands the canonical type will live there; this module's
/// `Context` is the minimum surface the agent requires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Context {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<ToolSchema>,
    #[serde(default)]
    pub params: ModelParams,
}

impl Context {
    pub fn into_request(self) -> ModelRequest {
        self.into_request_with_stream(false)
    }

    /// Build a [`ModelRequest`], setting the `stream` flag. The streaming turn
    /// path (issue #103) builds the request with `stream = true`.
    pub fn into_request_with_stream(self, stream: bool) -> ModelRequest {
        ModelRequest {
            messages: self.messages,
            tools: self.tools,
            params: self.params,
            stream,
        }
    }
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[non_exhaustive]
pub enum AgentError {
    #[error(transparent)]
    ModelError(#[from] ModelError),

    #[error("model returned neither text nor tool calls")]
    EmptyResponse,

    #[error("malformed tool call from model (tool={tool_name}): {reason}")]
    MalformedToolCall { tool_name: String, reason: String },
}

// ============================================================================
// TurnResult
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TurnResult {
    /// The model wants the harness to dispatch one or more tool calls.
    ToolCallRequested {
        calls: Vec<ToolCall>,
        usage: TokenUsage,
        /// Accumulated reasoning (`Thinking`) text produced in this turn, if
        /// any (issue #103, Q4). `#[serde(default)]` keeps pre-#103 serialized
        /// `TurnResult`s back-compatible.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
    /// The model produced a terminal textual response.
    FinalResponse {
        content: String,
        usage: TokenUsage,
        /// Accumulated reasoning (`Thinking`) text produced in this turn, if
        /// any (issue #103, Q4). `#[serde(default)]` keeps pre-#103 serialized
        /// `TurnResult`s back-compatible.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
    /// The turn could not be classified into a tool call or a final response.
    Error {
        error: AgentError,
        usage: Option<TokenUsage>,
    },
}

// ============================================================================
// The trait
// ============================================================================

/// An owned callback that receives **raw** [`crate::model::StreamEvent`]s as
/// the agent drains a streaming model call (issue #103, Q1).
///
/// The agent boundary deals only in `model::StreamEvent`; it does NOT depend on
/// the harness `StreamEvent` type. The harness wraps `on_stream` in an adapter
/// that maps `model::StreamEvent → harness::StreamEvent`.
///
/// Owned (`Box<dyn Fn ...>`) rather than borrowed: the trait method is
/// dyn-compatible (returns a `BoxFut`), and an owned sink can be `move`d into
/// the returned future without lifetime gymnastics.
pub type AgentStreamSink = Box<dyn Fn(StreamEvent) + Send + Sync>;

/// Executes a single turn given a fully assembled [`Context`].
///
/// Dyn-compatible via the hand-rolled [`BoxFut`] pattern, so the harness can
/// hold it as `Arc<dyn Agent>`.
pub trait Agent: Send + Sync {
    fn turn<'a>(&'a self, context: Context) -> BoxFut<'a, TurnResult>;

    /// Execute one turn while forwarding each raw [`crate::model::StreamEvent`]
    /// to `sink` as it arrives (issue #103).
    ///
    /// The default implementation **ignores** the sink and delegates to
    /// [`Agent::turn`], so every existing `Agent` impl keeps working with zero
    /// changes. Model-backed agents override this to call
    /// [`crate::model::ModelInterface::call_streaming`] and emit deltas.
    fn turn_streaming<'a>(
        &'a self,
        context: Context,
        sink: AgentStreamSink,
    ) -> BoxFut<'a, TurnResult> {
        let _ = sink;
        self.turn(context)
    }

    fn id(&self) -> AgentId;
}

/// Classify an accumulated [`ModelResponse`] into a [`TurnResult`].
///
/// Single source of truth shared by [`Agent::turn`] and
/// [`Agent::turn_streaming`] (issue #103) — both the blocking and streaming
/// paths buffer a complete `ModelResponse` and then run this identical logic so
/// classification can never diverge between them.
///
/// `Thinking` blocks are accumulated into the `reasoning` field (Q4) instead of
/// being discarded.
pub fn classify_response(response: ModelResponse) -> TurnResult {
    let usage = response.usage;

    // Extract any tool-use blocks regardless of stop_reason; the model may, in
    // principle, request tool use without setting StopReason (different
    // providers normalise this differently). The stop_reason determines the
    // *classification*, but missing tool calls when stop_reason == ToolUse is
    // itself a malformed response.
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut text_parts: Vec<String> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    for block in &response.content {
        match block {
            ContentBlock::ToolUse(tc) => tool_calls.push(tc.clone()),
            ContentBlock::Text { text } => text_parts.push(text.clone()),
            // Q4: accumulate thinking text instead of discarding it.
            ContentBlock::Thinking { text } => reasoning_parts.push(text.clone()),
        }
    }
    let reasoning = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join(""))
    };

    match response.stop_reason {
        StopReason::ToolUse => {
            if tool_calls.is_empty() {
                return TurnResult::Error {
                    error: AgentError::MalformedToolCall {
                        tool_name: String::new(),
                        reason: "stop_reason=ToolUse but no ToolUse blocks present".into(),
                    },
                    usage: Some(usage),
                };
            }
            TurnResult::ToolCallRequested {
                calls: tool_calls,
                usage,
                reasoning,
            }
        }
        StopReason::EndTurn | StopReason::MaxTokens | StopReason::StopSequence => {
            // If the provider returned no tool-use blocks AND no text, that is
            // an empty response — surface it explicitly. (Thinking-only output
            // is still empty: thinking is not a terminal response.)
            if text_parts.is_empty() && tool_calls.is_empty() {
                return TurnResult::Error {
                    error: AgentError::EmptyResponse,
                    usage: Some(usage),
                };
            }
            // If we somehow received tool-use blocks but stop_reason did not
            // indicate tool use, prefer dispatching them — silently dropping a
            // tool call is worse than a slightly surprising classification.
            if !tool_calls.is_empty() {
                return TurnResult::ToolCallRequested {
                    calls: tool_calls,
                    usage,
                    reasoning,
                };
            }
            TurnResult::FinalResponse {
                content: text_parts.join(""),
                usage,
                reasoning,
            }
        }
    }
}

// ============================================================================
// ModelAgent — the standard implementation
// ============================================================================

/// Standard `Agent` impl: forwards [`Context`] to a `ModelInterface` and
/// classifies the response per the rules in the module header.
///
/// Generic over the model implementation because `ModelInterface` uses
/// return-position `impl Trait` (via `trait_variant`) and is therefore not
/// `dyn`-compatible. Inject `Arc<ConcreteModel>` at construction; the harness
/// holds the agent itself behind `Arc<dyn Agent>`.
pub struct ModelAgent<M: ModelInterface> {
    id: AgentId,
    model: Arc<M>,
}

impl<M: ModelInterface> ModelAgent<M> {
    pub fn new(id: AgentId, model: Arc<M>) -> Self {
        Self { id, model }
    }
}

impl<M: ModelInterface + 'static> Agent for ModelAgent<M> {
    fn turn<'a>(&'a self, context: Context) -> BoxFut<'a, TurnResult> {
        Box::pin(async move {
            let request = context.into_request();
            match self.model.call(request).await {
                Ok(r) => classify_response(r),
                Err(e) => TurnResult::Error {
                    error: AgentError::ModelError(e),
                    usage: None,
                },
            }
        })
    }

    /// Streaming turn (issue #103). Builds a streaming request, drains the
    /// model stream forwarding each raw [`StreamEvent`] to `sink`, reassembles
    /// a complete [`ModelResponse`], then runs the EXACT SAME
    /// [`classify_response`] logic as [`Agent::turn`].
    ///
    /// Reassembly rules:
    /// - `ContentBlockDelta { index, .. }` text deltas are concatenated per
    ///   block index into a `Text` block.
    /// - `ThinkingDelta { index, .. }` deltas accumulate into a `Thinking`
    ///   block (Q4 — surfaced via `reasoning`, NOT discarded).
    /// - `ToolUseStart { index, id, name }` opens a `ToolUse` block, recording
    ///   the tool id + name; `ToolUseDelta { index, partial_json }` fragments
    ///   then concatenate and parse into that block's `input`.
    /// - `MessageStop` carries the final `usage` + `stop_reason`.
    ///
    /// Block ordering is preserved by sorting reassembled blocks on their
    /// stream index.
    fn turn_streaming<'a>(
        &'a self,
        context: Context,
        sink: AgentStreamSink,
    ) -> BoxFut<'a, TurnResult> {
        Box::pin(async move {
            use futures_util::StreamExt;

            let request = context.into_request_with_stream(true);
            let mut stream = match self.model.call_streaming(request).await {
                Ok(s) => s,
                Err(e) => {
                    return TurnResult::Error {
                        error: AgentError::ModelError(e),
                        usage: None,
                    };
                }
            };

            // Per-block accumulators keyed by stream index.
            let mut acc = StreamAccumulator::default();

            while let Some(item) = stream.next().await {
                let event = match item {
                    Ok(ev) => ev,
                    Err(e) => {
                        return TurnResult::Error {
                            error: AgentError::ModelError(e),
                            usage: None,
                        };
                    }
                };
                // Forward the RAW model event to the sink first (Q1), then fold
                // it into the in-progress response.
                sink(event.clone());
                acc.fold(event);
            }

            classify_response(acc.into_response())
        })
    }

    fn id(&self) -> AgentId {
        self.id.clone()
    }
}

/// Reassembles streamed [`StreamEvent`]s into a [`ModelResponse`] (issue #103).
///
/// Tracks an ordered set of partial blocks keyed by their stream `index` so the
/// reconstructed `content` preserves emission order regardless of interleaving.
#[derive(Default)]
struct StreamAccumulator {
    /// (index, partial-block) in first-seen order.
    blocks: Vec<(u32, PartialBlock)>,
    usage: TokenUsage,
    stop_reason: Option<StopReason>,
}

enum PartialBlock {
    Text(String),
    Thinking(String),
    ToolJson { id: String, name: String, json: String },
}

impl StreamAccumulator {
    fn entry(&mut self, index: u32, make: impl FnOnce() -> PartialBlock) -> &mut PartialBlock {
        if let Some(pos) = self.blocks.iter().position(|(i, _)| *i == index) {
            &mut self.blocks[pos].1
        } else {
            self.blocks.push((index, make()));
            &mut self.blocks.last_mut().expect("just pushed").1
        }
    }

    fn fold(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::MessageStart => {}
            StreamEvent::ContentBlockDelta { index, delta } => {
                if let PartialBlock::Text(s) =
                    self.entry(index, || PartialBlock::Text(String::new()))
                {
                    s.push_str(&delta);
                }
            }
            StreamEvent::ThinkingDelta { index, delta } => {
                if let PartialBlock::Thinking(s) =
                    self.entry(index, || PartialBlock::Thinking(String::new()))
                {
                    s.push_str(&delta);
                }
            }
            StreamEvent::ToolUseStart { index, id, name } => {
                if let PartialBlock::ToolJson {
                    id: bid,
                    name: bname,
                    ..
                } = self.entry(index, || PartialBlock::ToolJson {
                    id: String::new(),
                    name: String::new(),
                    json: String::new(),
                }) {
                    *bid = id;
                    *bname = name;
                }
            }
            StreamEvent::ToolUseDelta {
                index,
                partial_json,
            } => {
                if let PartialBlock::ToolJson { json, .. } =
                    self.entry(index, || PartialBlock::ToolJson {
                        id: String::new(),
                        name: String::new(),
                        json: String::new(),
                    })
                {
                    json.push_str(&partial_json);
                }
            }
            StreamEvent::ContentBlockStop { .. } => {}
            StreamEvent::MessageStop { usage, stop_reason } => {
                self.usage = usage;
                self.stop_reason = Some(stop_reason);
            }
        }
    }

    fn into_response(self) -> ModelResponse {
        let content: Vec<ContentBlock> = self
            .blocks
            .into_iter()
            .map(|(index, block)| match block {
                PartialBlock::Text(text) => ContentBlock::Text { text },
                PartialBlock::Thinking(text) => ContentBlock::Thinking { text },
                PartialBlock::ToolJson { id, name, json } => {
                    let input: serde_json::Value =
                        serde_json::from_str(&json).unwrap_or(serde_json::Value::Null);
                    ContentBlock::ToolUse(ToolCall {
                        // `id` / `name` come from the `ToolUseStart` event every
                        // provider emits at block start. Fall back to a stable
                        // per-index id only if a stream somehow omitted the start
                        // frame, so reconstruction is always well-formed.
                        id: if id.is_empty() {
                            format!("call_{index}")
                        } else {
                            id
                        },
                        name,
                        input,
                    })
                }
            })
            .collect();
        ModelResponse {
            content,
            usage: self.usage,
            // Default to EndTurn if the stream ended without MessageStop.
            stop_reason: self.stop_reason.unwrap_or(StopReason::EndTurn),
        }
    }
}

// ============================================================================
// Mock implementation (test-only)
// ============================================================================

#[cfg(any(test, feature = "test-utils"))]
pub mod mock {
    use super::*;
    use std::sync::Mutex;

    /// Programmable mock for unit tests. Each call to `turn` pops the next
    /// queued [`TurnResult`].
    pub struct MockAgent {
        id: AgentId,
        results: Mutex<std::collections::VecDeque<TurnResult>>,
        pub call_count: std::sync::atomic::AtomicUsize,
    }

    impl MockAgent {
        pub fn new(id: AgentId) -> Self {
            Self {
                id,
                results: Mutex::new(Default::default()),
                call_count: Default::default(),
            }
        }

        pub fn push(&self, r: TurnResult) -> &Self {
            self.results.lock().unwrap().push_back(r);
            self
        }
    }

    impl Agent for MockAgent {
        fn turn<'a>(&'a self, _context: Context) -> BoxFut<'a, TurnResult> {
            Box::pin(async move {
                self.call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                self.results
                    .lock()
                    .unwrap()
                    .pop_front()
                    .unwrap_or(TurnResult::Error {
                        error: AgentError::EmptyResponse,
                        usage: None,
                    })
            })
        }

        fn id(&self) -> AgentId {
            self.id.clone()
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mock::MockModelInterface;
    use crate::model::{ModelResponse, ProviderInfo, Role};

    fn provider() -> ProviderInfo {
        ProviderInfo {
            name: "test".into(),
            model_id: "test-1".into(),
            context_window: 1000,
        }
    }

    fn ctx_user(text: &str) -> Context {
        Context {
            messages: vec![Message {
                role: Role::User,
                content: crate::model::Content::Text { text: text.into() },
            }],
            tools: vec![],
            params: ModelParams::default(),
        }
    }

    fn usage(in_t: u32, out_t: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: in_t,
            output_tokens: out_t,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }
    }

    fn text_resp(text: &str) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            usage: usage(3, 4),
            stop_reason: StopReason::EndTurn,
        }
    }

    fn tool_resp(calls: Vec<ToolCall>) -> ModelResponse {
        ModelResponse {
            content: calls.into_iter().map(ContentBlock::ToolUse).collect(),
            usage: usage(5, 6),
            stop_reason: StopReason::ToolUse,
        }
    }

    fn make_agent(model: Arc<MockModelInterface>) -> ModelAgent<MockModelInterface> {
        ModelAgent::new(AgentId::new("coding-agent"), model)
    }

    // Rule: one turn = one model call.
    #[tokio::test]
    async fn turn_makes_exactly_one_model_call() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(text_resp("ok")));
        let agent = make_agent(m.clone());
        let _ = agent.turn(ctx_user("hi")).await;
        assert_eq!(
            m.call_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "agent.turn must invoke the model exactly once"
        );
    }

    // Rule: FinalResponse classification on stop_reason=EndTurn with text.
    #[tokio::test]
    async fn final_response_on_end_turn_with_text() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(text_resp("hello world")));
        let agent = make_agent(m);
        match agent.turn(ctx_user("hi")).await {
            TurnResult::FinalResponse { content, usage, .. } => {
                assert_eq!(content, "hello world");
                assert_eq!(usage.input_tokens, 3);
                assert_eq!(usage.output_tokens, 4);
            }
            other => panic!("expected FinalResponse, got {other:?}"),
        }
    }

    // Rule: ToolCallRequested classification on stop_reason=ToolUse.
    #[tokio::test]
    async fn tool_call_requested_on_tool_use_stop() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(tool_resp(vec![ToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            input: serde_json::json!({"path": "/x"}),
        }])));
        let agent = make_agent(m);
        match agent.turn(ctx_user("read /x")).await {
            TurnResult::ToolCallRequested { calls, usage, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_1");
                assert_eq!(calls[0].name, "read_file");
                assert_eq!(usage.input_tokens, 5);
            }
            other => panic!("expected ToolCallRequested, got {other:?}"),
        }
    }

    // Rule: ToolCallRequested may carry multiple parallel tool calls.
    #[tokio::test]
    async fn tool_call_requested_carries_multiple_calls() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(tool_resp(vec![
            ToolCall {
                id: "a".into(),
                name: "read_file".into(),
                input: serde_json::json!({"path": "/a"}),
            },
            ToolCall {
                id: "b".into(),
                name: "read_file".into(),
                input: serde_json::json!({"path": "/b"}),
            },
        ])));
        let agent = make_agent(m);
        match agent.turn(ctx_user("read both")).await {
            TurnResult::ToolCallRequested { calls, .. } => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].id, "a");
                assert_eq!(calls[1].id, "b");
            }
            other => panic!("expected multi ToolCallRequested, got {other:?}"),
        }
    }

    // Rule: EmptyResponse when model returns neither text nor tool calls.
    #[tokio::test]
    async fn empty_response_when_no_content_blocks() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(ModelResponse {
            content: vec![],
            usage: usage(1, 0),
            stop_reason: StopReason::EndTurn,
        }));
        let agent = make_agent(m);
        match agent.turn(ctx_user("?")).await {
            TurnResult::Error {
                error: AgentError::EmptyResponse,
                usage: Some(u),
            } => {
                assert_eq!(u.input_tokens, 1);
            }
            other => panic!("expected EmptyResponse error, got {other:?}"),
        }
    }

    // Rule: Thinking blocks are discarded — text-only thinking is still empty.
    #[tokio::test]
    async fn thinking_blocks_do_not_satisfy_final_response() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(ModelResponse {
            content: vec![ContentBlock::Thinking {
                text: "musing".into(),
            }],
            usage: usage(1, 2),
            stop_reason: StopReason::EndTurn,
        }));
        let agent = make_agent(m);
        assert!(matches!(
            agent.turn(ctx_user("?")).await,
            TurnResult::Error {
                error: AgentError::EmptyResponse,
                ..
            }
        ));
    }

    // Rule: ModelError surfaces wrapped in AgentError::ModelError.
    #[tokio::test]
    async fn model_error_surfaces_wrapped() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Err(ModelError::Timeout));
        let agent = make_agent(m);
        match agent.turn(ctx_user("hi")).await {
            TurnResult::Error {
                error: AgentError::ModelError(ModelError::Timeout),
                usage: None,
            } => {}
            other => panic!("expected wrapped ModelError::Timeout, got {other:?}"),
        }
    }

    // Rule: stop_reason=ToolUse without ToolUse blocks → MalformedToolCall.
    #[tokio::test]
    async fn malformed_when_tool_use_stop_but_no_tool_blocks() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(ModelResponse {
            content: vec![ContentBlock::Text { text: "hmm".into() }],
            usage: usage(2, 2),
            stop_reason: StopReason::ToolUse,
        }));
        let agent = make_agent(m);
        match agent.turn(ctx_user("?")).await {
            TurnResult::Error {
                error: AgentError::MalformedToolCall { .. },
                usage: Some(_),
            } => {}
            other => panic!("expected MalformedToolCall, got {other:?}"),
        }
    }

    // Tool calls present despite non-ToolUse stop_reason are still dispatched
    // — silently dropping a tool call is worse than the surprise.
    #[tokio::test]
    async fn tool_calls_dispatched_even_when_stop_is_end_turn() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(ModelResponse {
            content: vec![ContentBlock::ToolUse(ToolCall {
                id: "x".into(),
                name: "noop".into(),
                input: serde_json::json!({}),
            })],
            usage: usage(1, 1),
            stop_reason: StopReason::EndTurn,
        }));
        let agent = make_agent(m);
        assert!(matches!(
            agent.turn(ctx_user("?")).await,
            TurnResult::ToolCallRequested { .. }
        ));
    }

    // Rule: agent identity is reported (used for tracing).
    #[test]
    fn agent_id_reported() {
        let m = Arc::new(MockModelInterface::new(provider()));
        let agent = ModelAgent::new(AgentId::new("initializer"), m);
        assert_eq!(agent.id(), AgentId::new("initializer"));
        assert_eq!(agent.id().as_str(), "initializer");
    }

    // Rule: MaxTokens / StopSequence classify as FinalResponse with text.
    #[tokio::test]
    async fn max_tokens_stop_classifies_as_final_response() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(ModelResponse {
            content: vec![ContentBlock::Text {
                text: "truncated".into(),
            }],
            usage: usage(2, 5),
            stop_reason: StopReason::MaxTokens,
        }));
        let agent = make_agent(m);
        assert!(matches!(
            agent.turn(ctx_user("?")).await,
            TurnResult::FinalResponse { .. }
        ));
    }

    #[tokio::test]
    async fn stop_sequence_classifies_as_final_response() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(ModelResponse {
            content: vec![ContentBlock::Text {
                text: "done.".into(),
            }],
            usage: usage(2, 1),
            stop_reason: StopReason::StopSequence,
        }));
        let agent = make_agent(m);
        assert!(matches!(
            agent.turn(ctx_user("?")).await,
            TurnResult::FinalResponse { .. }
        ));
    }

    // Multiple text blocks are concatenated into a single FinalResponse.
    #[tokio::test]
    async fn multiple_text_blocks_are_concatenated() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(ModelResponse {
            content: vec![
                ContentBlock::Text { text: "foo".into() },
                ContentBlock::Text { text: "bar".into() },
            ],
            usage: usage(1, 1),
            stop_reason: StopReason::EndTurn,
        }));
        let agent = make_agent(m);
        match agent.turn(ctx_user("?")).await {
            TurnResult::FinalResponse { content, .. } => assert_eq!(content, "foobar"),
            other => panic!("expected FinalResponse, got {other:?}"),
        }
    }

    // Serde round-trip for cross-language fixture portability.
    #[test]
    fn turn_result_roundtrips_json() {
        let r = TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: "1".into(),
                name: "x".into(),
                input: serde_json::json!({"a": 1}),
            }],
            usage: usage(2, 3),
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: TurnResult = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);

        let r2 = TurnResult::FinalResponse {
            reasoning: None,
            content: "hi".into(),
            usage: usage(1, 1),
        };
        let s2 = serde_json::to_string(&r2).unwrap();
        let back2: TurnResult = serde_json::from_str(&s2).unwrap();
        assert_eq!(r2, back2);

        let r3 = TurnResult::Error {
            error: AgentError::EmptyResponse,
            usage: None,
        };
        let s3 = serde_json::to_string(&r3).unwrap();
        let back3: TurnResult = serde_json::from_str(&s3).unwrap();
        assert_eq!(r3, back3);
    }

    #[test]
    fn agent_error_variants_constructible_and_displayable() {
        let variants = [
            AgentError::ModelError(ModelError::Timeout),
            AgentError::EmptyResponse,
            AgentError::MalformedToolCall {
                tool_name: "x".into(),
                reason: "y".into(),
            },
        ];
        for v in &variants {
            assert!(!v.to_string().is_empty());
        }
    }

    // Fixture-replay coverage: an Agent driven by ReplayModelInterface should
    // produce the same classifications across a recorded session. Mirrors how
    // every language exercises fixtures/model_responses/*.jsonl through the
    // Agent layer.
    #[tokio::test]
    async fn agent_replay_against_fixture_jsonl() {
        use crate::model::ReplayModelInterface;
        let jsonl = r#"{"request":{"messages":[{"role":"user","content":{"type":"text","text":"call tool"}}],"tools":[],"params":{},"stream":false},"response":{"content":[{"type":"tool_use","id":"c1","name":"echo","input":{"v":1}}],"usage":{"input_tokens":3,"output_tokens":2},"stop_reason":"tool_use"},"provider":"anthropic"}
{"request":{"messages":[{"role":"user","content":{"type":"text","text":"finish"}}],"tools":[],"params":{},"stream":false},"response":{"content":[{"type":"text","text":"all done"}],"usage":{"input_tokens":4,"output_tokens":2},"stop_reason":"end_turn"},"provider":"anthropic"}"#;
        let replay = Arc::new(
            ReplayModelInterface::from_jsonl(
                jsonl,
                crate::model::ProviderInfo {
                    name: "anthropic".into(),
                    model_id: "replay".into(),
                    context_window: 200_000,
                },
            )
            .unwrap(),
        );
        let agent = ModelAgent::new(AgentId::new("replay-agent"), replay);

        match agent.turn(ctx_user("call tool")).await {
            TurnResult::ToolCallRequested { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "echo");
            }
            other => panic!("turn 1 expected ToolCallRequested, got {other:?}"),
        }
        match agent.turn(ctx_user("finish")).await {
            TurnResult::FinalResponse { content, .. } => assert_eq!(content, "all done"),
            other => panic!("turn 2 expected FinalResponse, got {other:?}"),
        }
    }

    // ── #103: delta-level streaming through the agent ───────────────────────

    use crate::model::{ProviderInfo as PInfo, ReplayModelInterface, StreamEvent as MEvent};
    use std::sync::Mutex as StdMutex;

    fn replay_provider() -> PInfo {
        PInfo {
            name: "anthropic".into(),
            model_id: "replay".into(),
            context_window: 200_000,
        }
    }

    /// A response carrying a Thinking + Text + ToolUse block in one turn, so
    /// `ReplayModelInterface::call_streaming` synthesizes all three delta kinds.
    fn mixed_response() -> ModelResponse {
        ModelResponse {
            content: vec![
                ContentBlock::Thinking {
                    text: "let me think".into(),
                },
                ContentBlock::Text {
                    text: "the answer is".into(),
                },
                ContentBlock::ToolUse(ToolCall {
                    id: "toolu_1".into(),
                    name: "lookup".into(),
                    input: serde_json::json!({"q": "rust"}),
                }),
            ],
            usage: usage(7, 11),
            stop_reason: StopReason::ToolUse,
        }
    }

    fn replay_agent(resp: ModelResponse) -> ModelAgent<ReplayModelInterface> {
        let exchanges = vec![crate::model::RecordedExchange {
            request_hash: None,
            request: ModelRequest {
                messages: vec![],
                tools: vec![],
                params: ModelParams::default(),
                stream: true,
            },
            response: resp,
            provider: "anthropic".into(),
            model_id: None,
            recorded_at: None,
            duration_ms: None,
        }];
        let replay = Arc::new(ReplayModelInterface::new(exchanges, replay_provider()));
        ModelAgent::new(AgentId::new("stream-agent"), replay)
    }

    // Rule: turn_streaming forwards raw model::StreamEvents to the sink in
    // order, and TextDelta-equivalent fragments concatenate into the final
    // FinalResponse/ToolCallRequested content.
    #[tokio::test]
    async fn turn_streaming_forwards_text_and_thinking_deltas() {
        let agent = replay_agent(ModelResponse {
            content: vec![
                ContentBlock::Thinking {
                    text: "reasoning here".into(),
                },
                ContentBlock::Text {
                    text: "final text".into(),
                },
            ],
            usage: usage(3, 4),
            stop_reason: StopReason::EndTurn,
        });
        let seen: Arc<StdMutex<Vec<MEvent>>> = Arc::new(StdMutex::new(Vec::new()));
        let s = seen.clone();
        let sink: AgentStreamSink = Box::new(move |ev| s.lock().unwrap().push(ev));
        let result = agent.turn_streaming(ctx_user("hi"), sink).await;

        // Reasoning surfaced in TurnResult (Q4) AND streamed as ThinkingDelta.
        match result {
            TurnResult::FinalResponse {
                content, reasoning, ..
            } => {
                assert_eq!(content, "final text");
                assert_eq!(reasoning.as_deref(), Some("reasoning here"));
            }
            other => panic!("expected FinalResponse, got {other:?}"),
        }
        let events = seen.lock().unwrap();
        assert!(matches!(events.first(), Some(MEvent::MessageStart)));
        assert!(events.iter().any(
            |e| matches!(e, MEvent::ThinkingDelta { delta, .. } if delta == "reasoning here")
        ));
        assert!(events.iter().any(
            |e| matches!(e, MEvent::ContentBlockDelta { delta, .. } if delta == "final text")
        ));
        assert!(matches!(events.last(), Some(MEvent::MessageStop { .. })));
    }

    // Rule: tool-use streaming yields ToolUseDelta with the full args JSON, and
    // turn_streaming reassembles the ToolCall.
    #[tokio::test]
    async fn turn_streaming_reassembles_tool_call() {
        let agent = replay_agent(mixed_response());
        let seen: Arc<StdMutex<Vec<MEvent>>> = Arc::new(StdMutex::new(Vec::new()));
        let s = seen.clone();
        let sink: AgentStreamSink = Box::new(move |ev| s.lock().unwrap().push(ev));
        let result = agent.turn_streaming(ctx_user("hi"), sink).await;
        match result {
            TurnResult::ToolCallRequested {
                calls, reasoning, ..
            } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].input, serde_json::json!({"q": "rust"}));
                assert_eq!(reasoning.as_deref(), Some("let me think"));
            }
            other => panic!("expected ToolCallRequested, got {other:?}"),
        }
        let events = seen.lock().unwrap();
        assert!(events
            .iter()
            .any(|e| matches!(e, MEvent::ToolUseDelta { partial_json, .. } if partial_json.contains("rust"))));
    }

    // Back-compat: the default turn_streaming (e.g. MockAgent) ignores the sink
    // and delegates to `turn`, producing the same TurnResult with no events.
    #[tokio::test]
    async fn default_turn_streaming_ignores_sink() {
        use crate::agent::mock::MockAgent;
        let a = MockAgent::new(AgentId::new("mock"));
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(1, 1),
            reasoning: None,
        });
        let count: Arc<StdMutex<usize>> = Arc::new(StdMutex::new(0));
        let c = count.clone();
        let sink: AgentStreamSink = Box::new(move |_ev| *c.lock().unwrap() += 1);
        let result = a.turn_streaming(ctx_user("hi"), sink).await;
        assert!(matches!(result, TurnResult::FinalResponse { content, .. } if content == "done"));
        assert_eq!(
            *count.lock().unwrap(),
            0,
            "default impl must not emit events"
        );
    }

    // turn and turn_streaming classify identically for the same response.
    #[tokio::test]
    async fn turn_and_turn_streaming_classify_identically() {
        let blocking = replay_agent(ModelResponse {
            content: vec![ContentBlock::Text {
                text: "same".into(),
            }],
            usage: usage(2, 2),
            stop_reason: StopReason::EndTurn,
        });
        let r_block = blocking.turn(ctx_user("x")).await;

        let streaming = replay_agent(ModelResponse {
            content: vec![ContentBlock::Text {
                text: "same".into(),
            }],
            usage: usage(2, 2),
            stop_reason: StopReason::EndTurn,
        });
        let noop: AgentStreamSink = Box::new(|_| {});
        let r_stream = streaming.turn_streaming(ctx_user("x"), noop).await;
        assert_eq!(r_block, r_stream);
    }

    // Serde back-compat: a pre-#103 TurnResult JSON (no `reasoning`) still
    // deserializes, with reasoning defaulting to None.
    #[test]
    fn turn_result_deserializes_without_reasoning_field() {
        let json = r#"{"kind":"final_response","content":"hi","usage":{"input_tokens":1,"output_tokens":1}}"#;
        let back: TurnResult = serde_json::from_str(json).unwrap();
        assert!(matches!(
            back,
            TurnResult::FinalResponse {
                reasoning: None,
                ..
            }
        ));
    }
}
