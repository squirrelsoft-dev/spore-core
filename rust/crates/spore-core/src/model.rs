//! ModelInterface — boundary between the harness and the underlying LLM.
//!
//! Implements issue #1. The harness only ever talks to a model through this
//! trait; provider-specific concerns (Anthropic, OpenAI, Ollama, replay) live
//! behind concrete implementations.
//!
//! ## Rules enforced here
//!
//! 1. `TokenUsage` is reported on every successful call (`call` and the final
//!    summary of `call_streaming`). It is not optional.
//! 2. `ContextLimitExceeded` is reported by the implementation *before* the
//!    provider is contacted whenever `count_tokens` exceeds the provider's
//!    context window. The base impl `enforce_context_limit` does the check.
//! 3. `BudgetExceeded` is a harness-side check against
//!    `ModelRequest.params.max_tokens` budget tracking, surfaced as a typed
//!    error so the harness loop can halt with a useful reason.
//! 4. Provider-specific retry / backoff for transient errors (429, 529,
//!    timeouts) lives in the implementation, not in the trait.
//!
//! Cross-language note: the type names here are mirrored byte-for-byte in
//! the TypeScript / Python / Go packages. See `fixtures/README.md`.

use std::pin::Pin;
use std::time::Duration;

use futures_core::Stream;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// Roles, content, messages
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text { text: String },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    Image { media_type: String, data: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Content,
}

// ============================================================================
// Tool schema (subset — the canonical type lives with ToolRegistry, #4)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

// ============================================================================
// Request / params / response
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ModelParams {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub reasoning_budget: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<ToolSchema>,
    #[serde(default)]
    pub params: ModelParams,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    ToolUse,
    EndTurn,
    MaxTokens,
    StopSequence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    Thinking { text: String },
    ToolUse(ToolCall),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_read_tokens: Option<u32>,
    #[serde(default)]
    pub cache_write_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelResponse {
    pub content: Vec<ContentBlock>,
    pub usage: TokenUsage,
    pub stop_reason: StopReason,
}

// ============================================================================
// Streaming
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart,
    ContentBlockDelta {
        index: u32,
        delta: String,
    },
    ThinkingDelta {
        index: u32,
        delta: String,
    },
    ToolUseDelta {
        index: u32,
        partial_json: String,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageStop {
        usage: TokenUsage,
        stop_reason: StopReason,
    },
}

pub type ModelStream =
    Pin<Box<dyn Stream<Item = Result<StreamEvent, ModelError>> + Send + 'static>>;

// ============================================================================
// Provider identity
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderInfo {
    pub name: String,
    pub model_id: String,
    pub context_window: u32,
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[non_exhaustive]
pub enum ModelError {
    #[error("provider error {code}: {message}")]
    ProviderError { code: u16, message: String },

    #[error("rate limited (retry_after={retry_after:?})")]
    RateLimited {
        #[serde(default, with = "duration_secs_opt")]
        retry_after: Option<Duration>,
    },

    #[error("context limit exceeded: {actual} tokens > limit {limit}")]
    ContextLimitExceeded { limit: u32, actual: u32 },

    #[error("budget exceeded: {used} > budget {budget}")]
    BudgetExceeded { budget: u32, used: u32 },

    #[error("model call timed out")]
    Timeout,
}

mod duration_secs_opt {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(v: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        v.map(|d| d.as_secs()).serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
        Ok(Option::<u64>::deserialize(d)?.map(Duration::from_secs))
    }
}

// ============================================================================
// The trait
// ============================================================================

/// Boundary between the harness and the underlying LLM.
///
/// Implementors must observe the rules documented in the module header.
#[trait_variant::make(Send)]
pub trait ModelInterface: Send + Sync {
    /// One blocking model call. `TokenUsage` must be populated on success.
    async fn call(&self, request: ModelRequest) -> Result<ModelResponse, ModelError>;

    /// Streaming variant. Yields `StreamEvent`s as they arrive; the final
    /// `MessageStop` event carries the accumulated `TokenUsage`.
    async fn call_streaming(&self, request: ModelRequest) -> Result<ModelStream, ModelError>;

    /// Pre-call token count for context-size estimation. Used by the harness
    /// to detect `ContextLimitExceeded` before contacting the provider.
    async fn count_tokens(&self, request: &ModelRequest) -> Result<u32, ModelError>;

    /// Provider identity for tracing and routing decisions.
    fn provider(&self) -> ProviderInfo;
}

/// Shared pre-call validation. Implementations should call this before
/// contacting the provider.
pub fn enforce_context_limit(request_tokens: u32, context_window: u32) -> Result<(), ModelError> {
    if request_tokens > context_window {
        return Err(ModelError::ContextLimitExceeded {
            limit: context_window,
            actual: request_tokens,
        });
    }
    Ok(())
}

/// Shared post-call budget check. `max_tokens` is the per-call budget the
/// harness injected via `ModelParams.max_tokens`; if more were produced, the
/// implementation surfaces it as a typed harness error.
pub fn enforce_budget(used: u32, budget: Option<u32>) -> Result<(), ModelError> {
    if let Some(b) = budget {
        if used > b {
            return Err(ModelError::BudgetExceeded { budget: b, used });
        }
    }
    Ok(())
}

// ============================================================================
// Fixture-replay implementation
// ============================================================================

/// A `(ModelRequest, ModelResponse)` pair as serialised in the shared
/// `fixtures/model_responses/` JSONL files. See `fixtures/README.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedExchange {
    pub request: ModelRequest,
    pub response: ModelResponse,
    pub provider: String,
    #[serde(default)]
    pub recorded_at: Option<String>,
}

/// In-order replay of recorded `(request, response)` pairs. Used by tests in
/// every language; matching is purely positional — the n-th `call` returns
/// the n-th recorded response.
///
/// Positional replay (rather than request-hash matching) is the contract
/// shared across all four language implementations so that fixtures remain
/// portable; deeper request-equality matching is a follow-up.
pub struct ReplayModelInterface {
    exchanges: Vec<RecordedExchange>,
    cursor: tokio::sync::Mutex<usize>,
    provider: ProviderInfo,
}

impl ReplayModelInterface {
    pub fn new(exchanges: Vec<RecordedExchange>, provider: ProviderInfo) -> Self {
        Self {
            exchanges,
            cursor: tokio::sync::Mutex::new(0),
            provider,
        }
    }

    pub fn from_jsonl(jsonl: &str, provider: ProviderInfo) -> Result<Self, serde_json::Error> {
        let exchanges = jsonl
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str::<RecordedExchange>)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(exchanges, provider))
    }

    pub fn remaining(&self) -> usize {
        let c = self.cursor.try_lock().map(|g| *g).unwrap_or(0);
        self.exchanges.len().saturating_sub(c)
    }
}

impl ModelInterface for ReplayModelInterface {
    async fn call(&self, _request: ModelRequest) -> Result<ModelResponse, ModelError> {
        let mut cursor = self.cursor.lock().await;
        let exchange = self
            .exchanges
            .get(*cursor)
            .ok_or(ModelError::ProviderError {
                code: 0,
                message: "replay exhausted: no more recorded exchanges".into(),
            })?;
        *cursor += 1;
        Ok(exchange.response.clone())
    }

    async fn call_streaming(&self, request: ModelRequest) -> Result<ModelStream, ModelError> {
        let response = self.call(request).await?;
        let stream = async_stream::stream! {
            yield Ok(StreamEvent::MessageStart);
            for (i, block) in response.content.iter().enumerate() {
                let idx = i as u32;
                match block {
                    ContentBlock::Text { text } => {
                        yield Ok(StreamEvent::ContentBlockDelta { index: idx, delta: text.clone() });
                    }
                    ContentBlock::Thinking { text } => {
                        yield Ok(StreamEvent::ThinkingDelta { index: idx, delta: text.clone() });
                    }
                    ContentBlock::ToolUse(call) => {
                        let json = serde_json::to_string(&call.input).unwrap_or_else(|_| "{}".into());
                        yield Ok(StreamEvent::ToolUseDelta { index: idx, partial_json: json });
                    }
                }
                yield Ok(StreamEvent::ContentBlockStop { index: idx });
            }
            yield Ok(StreamEvent::MessageStop {
                usage: response.usage,
                stop_reason: response.stop_reason,
            });
        };
        Ok(Box::pin(stream))
    }

    async fn count_tokens(&self, request: &ModelRequest) -> Result<u32, ModelError> {
        // Cheap deterministic estimate sufficient for fixture replay: sum the
        // textual content of every message. Real providers override this.
        let n: usize = request
            .messages
            .iter()
            .map(|m| match &m.content {
                Content::Text { text } => text.len(),
                Content::ToolCall(tc) => tc.name.len() + tc.input.to_string().len(),
                Content::ToolResult(tr) => tr.content.len(),
                Content::Image { .. } => 0,
            })
            .sum();
        Ok((n / 4) as u32) // ~4 chars/token rule-of-thumb
    }

    fn provider(&self) -> ProviderInfo {
        self.provider.clone()
    }
}

// ============================================================================
// Mock implementation (test-only)
// ============================================================================

#[cfg(any(test, feature = "test-utils"))]
pub mod mock {
    use super::*;
    use std::sync::Mutex;

    /// Programmable mock for unit tests. Each entry in `responses` is yielded
    /// on successive calls; `errors` can interleave to test error paths.
    pub struct MockModelInterface {
        responses: Mutex<std::collections::VecDeque<Result<ModelResponse, ModelError>>>,
        token_counts: Mutex<std::collections::VecDeque<Result<u32, ModelError>>>,
        provider: ProviderInfo,
        pub call_count: std::sync::atomic::AtomicUsize,
    }

    impl MockModelInterface {
        pub fn new(provider: ProviderInfo) -> Self {
            Self {
                responses: Mutex::new(Default::default()),
                token_counts: Mutex::new(Default::default()),
                provider,
                call_count: Default::default(),
            }
        }

        pub fn push_response(&self, r: Result<ModelResponse, ModelError>) -> &Self {
            self.responses.lock().unwrap().push_back(r);
            self
        }

        pub fn push_token_count(&self, r: Result<u32, ModelError>) -> &Self {
            self.token_counts.lock().unwrap().push_back(r);
            self
        }
    }

    impl ModelInterface for MockModelInterface {
        async fn call(&self, _request: ModelRequest) -> Result<ModelResponse, ModelError> {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(Err(ModelError::ProviderError {
                    code: 0,
                    message: "mock: no response queued".into(),
                }))
        }

        async fn call_streaming(&self, request: ModelRequest) -> Result<ModelStream, ModelError> {
            let response = self.call(request).await?;
            let stream = async_stream::stream! {
                yield Ok(StreamEvent::MessageStart);
                yield Ok(StreamEvent::MessageStop {
                    usage: response.usage,
                    stop_reason: response.stop_reason,
                });
            };
            Ok(Box::pin(stream))
        }

        async fn count_tokens(&self, _request: &ModelRequest) -> Result<u32, ModelError> {
            self.token_counts
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(Ok(0))
        }

        fn provider(&self) -> ProviderInfo {
            self.provider.clone()
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::mock::MockModelInterface;
    use super::*;
    use futures_util::StreamExt;

    fn provider() -> ProviderInfo {
        ProviderInfo {
            name: "test".into(),
            model_id: "test-1".into(),
            context_window: 1000,
        }
    }

    fn empty_request() -> ModelRequest {
        ModelRequest {
            messages: vec![],
            tools: vec![],
            params: ModelParams::default(),
            stream: false,
        }
    }

    fn text_response(text: &str, in_tok: u32, out_tok: u32) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            usage: TokenUsage {
                input_tokens: in_tok,
                output_tokens: out_tok,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
            stop_reason: StopReason::EndTurn,
        }
    }

    // Rule: call returns response.
    #[tokio::test]
    async fn call_returns_queued_response() {
        let m = MockModelInterface::new(provider());
        m.push_response(Ok(text_response("hi", 3, 1)));
        let r = m.call(empty_request()).await.unwrap();
        assert_eq!(r.content.len(), 1);
        assert_eq!(r.stop_reason, StopReason::EndTurn);
    }

    // Rule: token counts reported on every call, not optional.
    #[tokio::test]
    async fn token_usage_reported_on_every_call() {
        let m = MockModelInterface::new(provider());
        m.push_response(Ok(text_response("a", 5, 7)))
            .push_response(Ok(text_response("b", 11, 13)));
        let r1 = m.call(empty_request()).await.unwrap();
        let r2 = m.call(empty_request()).await.unwrap();
        assert_eq!(r1.usage.input_tokens, 5);
        assert_eq!(r1.usage.output_tokens, 7);
        assert_eq!(r2.usage.input_tokens, 11);
        assert_eq!(r2.usage.output_tokens, 13);
    }

    // Rule: ContextLimitExceeded thrown before API call when detected.
    #[test]
    fn context_limit_enforced_pre_call() {
        let err = enforce_context_limit(1500, 1000).unwrap_err();
        match err {
            ModelError::ContextLimitExceeded { limit, actual } => {
                assert_eq!(limit, 1000);
                assert_eq!(actual, 1500);
            }
            _ => panic!("expected ContextLimitExceeded, got {err:?}"),
        }
    }

    #[test]
    fn context_limit_passes_when_under() {
        assert!(enforce_context_limit(999, 1000).is_ok());
        assert!(enforce_context_limit(1000, 1000).is_ok());
    }

    // Rule: BudgetExceeded is a harness-side check, surfaced as a typed error.
    #[test]
    fn budget_enforced_against_max_tokens() {
        let err = enforce_budget(101, Some(100)).unwrap_err();
        assert!(matches!(
            err,
            ModelError::BudgetExceeded {
                budget: 100,
                used: 101
            }
        ));
    }

    #[test]
    fn budget_passes_when_under_or_unset() {
        assert!(enforce_budget(99, Some(100)).is_ok());
        assert!(enforce_budget(100, Some(100)).is_ok());
        assert!(enforce_budget(1_000_000, None).is_ok());
    }

    // Error variant coverage: every variant is constructible & displayable.
    #[test]
    fn every_error_variant_is_constructible() {
        let variants = [
            ModelError::ProviderError {
                code: 500,
                message: "boom".into(),
            },
            ModelError::RateLimited {
                retry_after: Some(Duration::from_secs(5)),
            },
            ModelError::ContextLimitExceeded {
                limit: 1,
                actual: 2,
            },
            ModelError::BudgetExceeded { budget: 1, used: 2 },
            ModelError::Timeout,
        ];
        for v in &variants {
            assert!(!v.to_string().is_empty());
        }
    }

    // Rule: provider identity is reported.
    #[test]
    fn provider_identity_reported() {
        let m = MockModelInterface::new(provider());
        let p = m.provider();
        assert_eq!(p.name, "test");
        assert_eq!(p.model_id, "test-1");
        assert_eq!(p.context_window, 1000);
    }

    // Rule: streaming accumulates usage from stream events and returns final summary.
    #[tokio::test]
    async fn streaming_yields_message_stop_with_usage() {
        let m = MockModelInterface::new(provider());
        m.push_response(Ok(text_response("hello", 4, 2)));
        let mut stream = m.call_streaming(empty_request()).await.unwrap();
        let mut saw_start = false;
        let mut final_usage: Option<TokenUsage> = None;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::MessageStart => saw_start = true,
                StreamEvent::MessageStop { usage, .. } => final_usage = Some(usage),
                _ => {}
            }
        }
        assert!(saw_start);
        let u = final_usage.expect("MessageStop must carry final usage");
        assert_eq!(u.input_tokens, 4);
        assert_eq!(u.output_tokens, 2);
    }

    // Rule: provider errors surface as typed harness errors.
    #[tokio::test]
    async fn provider_errors_surface_typed() {
        let m = MockModelInterface::new(provider());
        m.push_response(Err(ModelError::ProviderError {
            code: 503,
            message: "unavailable".into(),
        }));
        let err = m.call(empty_request()).await.unwrap_err();
        assert!(matches!(err, ModelError::ProviderError { code: 503, .. }));
    }

    #[tokio::test]
    async fn rate_limit_surface_with_retry_after() {
        let m = MockModelInterface::new(provider());
        m.push_response(Err(ModelError::RateLimited {
            retry_after: Some(Duration::from_secs(2)),
        }));
        let err = m.call(empty_request()).await.unwrap_err();
        match err {
            ModelError::RateLimited { retry_after } => {
                assert_eq!(retry_after, Some(Duration::from_secs(2)))
            }
            _ => panic!("expected RateLimited"),
        }
    }

    #[tokio::test]
    async fn timeout_surface() {
        let m = MockModelInterface::new(provider());
        m.push_response(Err(ModelError::Timeout));
        assert!(matches!(
            m.call(empty_request()).await.unwrap_err(),
            ModelError::Timeout
        ));
    }

    // Serde round-trip: every public type must round-trip JSON cleanly so that
    // fixtures shared across languages are byte-portable.
    #[test]
    fn model_request_roundtrips_json() {
        let req = ModelRequest {
            messages: vec![Message {
                role: Role::User,
                content: Content::Text { text: "hi".into() },
            }],
            tools: vec![ToolSchema {
                name: "echo".into(),
                description: "echoes input".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            params: ModelParams {
                temperature: Some(0.7),
                max_tokens: Some(1024),
                ..Default::default()
            },
            stream: false,
        };
        let s = serde_json::to_string(&req).unwrap();
        let back: ModelRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn model_response_roundtrips_json() {
        let resp = ModelResponse {
            content: vec![
                ContentBlock::Text { text: "ok".into() },
                ContentBlock::ToolUse(ToolCall {
                    id: "1".into(),
                    name: "x".into(),
                    input: serde_json::json!({"a":1}),
                }),
            ],
            usage: TokenUsage {
                input_tokens: 3,
                output_tokens: 4,
                cache_read_tokens: Some(1),
                cache_write_tokens: Some(2),
            },
            stop_reason: StopReason::ToolUse,
        };
        let s = serde_json::to_string(&resp).unwrap();
        let back: ModelResponse = serde_json::from_str(&s).unwrap();
        assert_eq!(resp, back);
    }

    // Fixture-replay coverage. Mirrors how every language exercises
    // fixtures/model_responses/*.jsonl.
    #[tokio::test]
    async fn replay_returns_recorded_responses_in_order() {
        let jsonl = r#"{"request":{"messages":[{"role":"user","content":{"type":"text","text":"a"}}],"tools":[],"params":{},"stream":false},"response":{"content":[{"type":"text","text":"A"}],"usage":{"input_tokens":1,"output_tokens":1},"stop_reason":"end_turn"},"provider":"anthropic"}
{"request":{"messages":[{"role":"user","content":{"type":"text","text":"b"}}],"tools":[],"params":{},"stream":false},"response":{"content":[{"type":"text","text":"B"}],"usage":{"input_tokens":2,"output_tokens":2},"stop_reason":"end_turn"},"provider":"anthropic"}"#;
        let replay = ReplayModelInterface::from_jsonl(
            jsonl,
            ProviderInfo {
                name: "anthropic".into(),
                model_id: "replay".into(),
                context_window: 200_000,
            },
        )
        .unwrap();
        assert_eq!(replay.remaining(), 2);

        let r1 = replay.call(empty_request()).await.unwrap();
        assert_eq!(r1.content, vec![ContentBlock::Text { text: "A".into() }]);
        let r2 = replay.call(empty_request()).await.unwrap();
        assert_eq!(r2.content, vec![ContentBlock::Text { text: "B".into() }]);

        // Exhaustion is a ProviderError, never a panic.
        let err = replay.call(empty_request()).await.unwrap_err();
        assert!(matches!(err, ModelError::ProviderError { code: 0, .. }));
    }

    #[tokio::test]
    async fn replay_count_tokens_is_deterministic() {
        let replay = ReplayModelInterface::new(
            vec![],
            ProviderInfo {
                name: "x".into(),
                model_id: "y".into(),
                context_window: 100,
            },
        );
        let req = ModelRequest {
            messages: vec![Message {
                role: Role::User,
                content: Content::Text {
                    text: "a".repeat(40),
                },
            }],
            tools: vec![],
            params: ModelParams::default(),
            stream: false,
        };
        let n = replay.count_tokens(&req).await.unwrap();
        assert_eq!(n, 10);
    }
}
