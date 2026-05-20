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
//!      (concatenated `Text` blocks; `Thinking` blocks discarded — they are
//!      observability, not output).
//! 5. `ModelError` is surfaced wrapped in `AgentError::ModelError`, with any
//!    partial usage information preserved.
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
    ContentBlock, Message, ModelError, ModelInterface, ModelParams, ModelRequest, StopReason,
    TokenUsage, ToolCall, ToolSchema,
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
        ModelRequest {
            messages: self.messages,
            tools: self.tools,
            params: self.params,
            stream: false,
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
    },
    /// The model produced a terminal textual response.
    FinalResponse { content: String, usage: TokenUsage },
    /// The turn could not be classified into a tool call or a final response.
    Error {
        error: AgentError,
        usage: Option<TokenUsage>,
    },
}

// ============================================================================
// The trait
// ============================================================================

/// Executes a single turn given a fully assembled [`Context`].
///
/// Dyn-compatible via the hand-rolled [`BoxFut`] pattern, so the harness can
/// hold it as `Arc<dyn Agent>`.
pub trait Agent: Send + Sync {
    fn turn<'a>(&'a self, context: Context) -> BoxFut<'a, TurnResult>;

    fn id(&self) -> AgentId;
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
            let response = match self.model.call(request).await {
                Ok(r) => r,
                Err(e) => {
                    return TurnResult::Error {
                        error: AgentError::ModelError(e),
                        usage: None,
                    };
                }
            };

            let usage = response.usage;

            // Extract any tool-use blocks regardless of stop_reason; the model
            // may, in principle, request tool use without setting StopReason
            // (different providers normalise this differently). The stop_reason
            // determines the *classification*, but missing tool calls when
            // stop_reason == ToolUse is itself a malformed response.
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut text_parts: Vec<String> = Vec::new();
            for block in &response.content {
                match block {
                    ContentBlock::ToolUse(tc) => tool_calls.push(tc.clone()),
                    ContentBlock::Text { text } => text_parts.push(text.clone()),
                    ContentBlock::Thinking { .. } => { /* observability only */ }
                }
            }

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
                    }
                }
                StopReason::EndTurn | StopReason::MaxTokens | StopReason::StopSequence => {
                    // If the provider returned no tool-use blocks AND no text,
                    // that is an empty response — surface it explicitly.
                    if text_parts.is_empty() && tool_calls.is_empty() {
                        return TurnResult::Error {
                            error: AgentError::EmptyResponse,
                            usage: Some(usage),
                        };
                    }
                    // If we somehow received tool-use blocks but stop_reason did
                    // not indicate tool use, prefer dispatching them — silently
                    // dropping a tool call is worse than a slightly surprising
                    // classification.
                    if !tool_calls.is_empty() {
                        return TurnResult::ToolCallRequested {
                            calls: tool_calls,
                            usage,
                        };
                    }
                    TurnResult::FinalResponse {
                        content: text_parts.join(""),
                        usage,
                    }
                }
            }
        })
    }

    fn id(&self) -> AgentId {
        self.id.clone()
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
            TurnResult::FinalResponse { content, usage } => {
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
            TurnResult::ToolCallRequested { calls, usage } => {
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
}
