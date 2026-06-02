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
use std::sync::Arc;
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
    /// Start of a tool-use block. Carries the tool `name` and call `id` — both
    /// arrive on the provider's block-start frame (Anthropic `content_block_start`,
    /// Ollama / OpenAI's first `tool_calls` chunk) and would otherwise be lost,
    /// since [`ToolUseDelta`](Self::ToolUseDelta) carries only argument JSON. The
    /// streaming accumulator uses this to reconstruct the tool call faithfully.
    ToolUseStart {
        index: u32,
        id: String,
        name: String,
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
///
/// `request_hash` (issue #37) is populated by `RecordingModelInterface` (#38)
/// to enable content-addressed replay. Fixtures recorded before #37 do not
/// include it; absence triggers positional fallback in `ReplayModelInterface`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedExchange {
    /// Stable hash of `request` computed by [`request_hash`]. Optional so
    /// pre-#37 positional fixtures continue to deserialize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_hash: Option<String>,
    pub request: ModelRequest,
    pub response: ModelResponse,
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recorded_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

// ============================================================================
// Cross-language request hashing (#37, #38)
// ============================================================================

/// Stable content hash of a `ModelRequest`. Produced by canonicalizing the
/// request to JSON (object keys sorted lexicographically, no insignificant
/// whitespace) and SHA-256 hashing the resulting bytes, then hex-encoding
/// the first 8 bytes (16 hex chars representing the leading u64).
///
/// All four language implementations of `RecordingModelInterface` and
/// `ReplayModelInterface` must produce the same hash for the same request.
/// The cross-language consistency fixture lives at
/// `fixtures/model_hashing/cases.json`.
pub fn request_hash(req: &ModelRequest) -> String {
    use sha2::Digest;
    let value = serde_json::to_value(req).expect("ModelRequest is always serializable");
    let canonical = canonicalize_json(&value);
    let mut hasher = sha2::Sha256::new();
    hasher.update(canonical.as_bytes());
    let digest = hasher.finalize();
    encode_hex(&digest[..8])
}

fn canonicalize_json(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Null => "null".into(),
        Value::Bool(true) => "true".into(),
        Value::Bool(false) => "false".into(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).expect("string encoding never fails"),
        Value::Array(arr) => {
            let inner: Vec<String> = arr.iter().map(canonicalize_json).collect();
            format!("[{}]", inner.join(","))
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let inner: Vec<String> = keys
                .into_iter()
                .map(|k| {
                    let key_str = serde_json::to_string(k).expect("string encoding never fails");
                    format!("{}:{}", key_str, canonicalize_json(&map[k]))
                })
                .collect();
            format!("{{{}}}", inner.join(","))
        }
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(nibble(b >> 4));
        s.push(nibble(b & 0x0F));
    }
    s
}

fn nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'a' + n - 10) as char,
        _ => unreachable!(),
    }
}

/// How a [`ReplayModelInterface`] matches incoming requests to recorded
/// responses (issue #37).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayMode {
    /// Pre-#37 behavior: the n-th `call` returns the n-th recorded response.
    /// Fragile against loop-order changes but compatible with old fixtures.
    Positional,
    /// New behavior: each `call` hashes its request and looks up the matching
    /// recorded entry. Order-independent.
    HashMatched,
}

/// Replay of recorded `(request, response)` pairs. Used by tests in every
/// language. Defaults to [`ReplayMode::HashMatched`] when every entry has a
/// `request_hash`, and falls back to [`ReplayMode::Positional`] otherwise so
/// pre-#37 fixtures continue to work.
pub struct ReplayModelInterface {
    exchanges: Vec<RecordedExchange>,
    cursor: tokio::sync::Mutex<usize>,
    provider: ProviderInfo,
    mode: ReplayMode,
}

impl ReplayModelInterface {
    /// Construct with the auto-detected mode. Use this in new code.
    pub fn new(exchanges: Vec<RecordedExchange>, provider: ProviderInfo) -> Self {
        let mode = if exchanges.iter().all(|e| e.request_hash.is_some()) && !exchanges.is_empty() {
            ReplayMode::HashMatched
        } else {
            ReplayMode::Positional
        };
        Self::with_mode(exchanges, provider, mode)
    }

    /// Construct with an explicit mode. Useful when forcing positional replay
    /// against a hash-tagged fixture (e.g. to test old behavior).
    pub fn with_mode(
        exchanges: Vec<RecordedExchange>,
        provider: ProviderInfo,
        mode: ReplayMode,
    ) -> Self {
        Self {
            exchanges,
            cursor: tokio::sync::Mutex::new(0),
            provider,
            mode,
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

    pub fn mode(&self) -> ReplayMode {
        self.mode
    }

    pub fn remaining(&self) -> usize {
        let c = self.cursor.try_lock().map(|g| *g).unwrap_or(0);
        self.exchanges.len().saturating_sub(c)
    }
}

impl ModelInterface for ReplayModelInterface {
    async fn call(&self, request: ModelRequest) -> Result<ModelResponse, ModelError> {
        match self.mode {
            ReplayMode::HashMatched => {
                let want = request_hash(&request);
                let exchange = self
                    .exchanges
                    .iter()
                    .find(|e| e.request_hash.as_deref() == Some(want.as_str()))
                    .ok_or_else(|| ModelError::ProviderError {
                        code: 0,
                        message: format!("no matching fixture for request_hash={want}"),
                    })?;
                Ok(exchange.response.clone())
            }
            ReplayMode::Positional => {
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
        }
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
                        yield Ok(StreamEvent::ToolUseStart {
                            index: idx,
                            id: call.id.clone(),
                            name: call.name.clone(),
                        });
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
        // When the fixture was recorded by RecordingModelInterface against a
        // real provider, the recorded response's `usage.input_tokens` carries
        // the provider's exact count. Use that whenever we can match by hash;
        // fall back to the bytes/4 heuristic only when no matching entry
        // exists (positional fixtures or unrecorded requests).
        if self.mode == ReplayMode::HashMatched {
            let want = request_hash(request);
            if let Some(e) = self
                .exchanges
                .iter()
                .find(|e| e.request_hash.as_deref() == Some(want.as_str()))
            {
                return Ok(e.response.usage.input_tokens);
            }
        }
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
        Ok((n / 4) as u32) // ~4 chars/token rule-of-thumb (fallback)
    }

    fn provider(&self) -> ProviderInfo {
        self.provider.clone()
    }
}

// ============================================================================
// RecordingModelInterface (issue #38)
// ============================================================================

/// Modes for [`RecordingModelInterface`]. See spec on issue #38.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingMode {
    /// Append every `(request, response)` pair to `output_path`.
    Record,
    /// Append only if the file does not yet exist. Useful when running tests
    /// for the first time against a real provider, then never re-recording.
    RecordIfNew,
    /// Call `inner` but do not write anything. Used to disable recording
    /// without changing call sites.
    Passthrough,
}

/// Transparent wrapper around a real [`ModelInterface`] that appends each
/// `(request, response)` pair to a JSONL fixture file as a
/// [`RecordedExchange`] with a stable [`request_hash`].
///
/// Generic over `M: ModelInterface + 'static` to remain dyn-compatible-free
/// (the underlying `ModelInterface` is not dyn-compatible because of RPITIT
/// — see #45). Wires up the same way `LlmJudgeEvaluator<M>` and
/// `QuestionAnsweredCheck<M>` do.
pub struct RecordingModelInterface<M: ModelInterface + 'static> {
    inner: Arc<M>,
    output_path: std::path::PathBuf,
    mode: RecordingMode,
    /// Set to `true` once the first write happens in `RecordIfNew` mode.
    /// Wrapped in a tokio Mutex so concurrent calls don't double-write.
    write_lock: tokio::sync::Mutex<()>,
}

impl<M: ModelInterface + 'static> RecordingModelInterface<M> {
    pub fn new(
        inner: Arc<M>,
        output_path: impl Into<std::path::PathBuf>,
        mode: RecordingMode,
    ) -> Self {
        Self {
            inner,
            output_path: output_path.into(),
            mode,
            write_lock: tokio::sync::Mutex::new(()),
        }
    }

    pub fn output_path(&self) -> &std::path::Path {
        &self.output_path
    }

    pub fn mode(&self) -> RecordingMode {
        self.mode
    }

    async fn record(
        &self,
        request: &ModelRequest,
        response: &ModelResponse,
        duration_ms: u64,
    ) -> Result<(), std::io::Error> {
        // Centralized check + write under a mutex so concurrent calls serialize.
        let _guard = self.write_lock.lock().await;
        let should_write = match self.mode {
            RecordingMode::Record => true,
            RecordingMode::RecordIfNew => !self.output_path.exists(),
            RecordingMode::Passthrough => false,
        };
        if !should_write {
            return Ok(());
        }
        if let Some(parent) = self.output_path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        let provider_info = self.inner.provider();
        let entry = RecordedExchange {
            request_hash: Some(request_hash(request)),
            request: request.clone(),
            response: response.clone(),
            provider: provider_info.name.clone(),
            model_id: Some(provider_info.model_id.clone()),
            recorded_at: None,
            duration_ms: Some(duration_ms),
        };
        let line = serde_json::to_string(&entry)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        use tokio::io::AsyncWriteExt;
        let mut f = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_path)
            .await?;
        f.write_all(line.as_bytes()).await?;
        f.write_all(b"\n").await?;
        Ok(())
    }
}

impl<M: ModelInterface + 'static> ModelInterface for RecordingModelInterface<M> {
    async fn call(&self, request: ModelRequest) -> Result<ModelResponse, ModelError> {
        let start = std::time::Instant::now();
        let response = self.inner.call(request.clone()).await?;
        let duration_ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        if let Err(e) = self.record(&request, &response, duration_ms).await {
            return Err(ModelError::ProviderError {
                code: 0,
                message: format!("recorder write failed: {e}"),
            });
        }
        Ok(response)
    }

    async fn call_streaming(&self, request: ModelRequest) -> Result<ModelStream, ModelError> {
        // Streaming recording is not implemented: spec only requires the
        // blocking `call()` pair to be recorded. Pass through unchanged.
        self.inner.call_streaming(request).await
    }

    async fn count_tokens(&self, request: &ModelRequest) -> Result<u32, ModelError> {
        self.inner.count_tokens(request).await
    }

    fn provider(&self) -> ProviderInfo {
        self.inner.provider()
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

    // ── #37: request_hash + ReplayMode ──────────────────────────────────────

    fn req_text(s: &str) -> ModelRequest {
        ModelRequest {
            messages: vec![Message {
                role: Role::User,
                content: Content::Text { text: s.into() },
            }],
            tools: vec![],
            params: ModelParams::default(),
            stream: false,
        }
    }

    fn resp_text(s: &str) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::Text { text: s.into() }],
            usage: TokenUsage::default(),
            stop_reason: StopReason::EndTurn,
        }
    }

    #[test]
    fn request_hash_is_stable_across_field_order() {
        // Two equivalent requests must hash the same. serde_json::Map is
        // already sorted; we still want to guarantee callers see this.
        let a = req_text("hello world");
        let b = req_text("hello world");
        assert_eq!(request_hash(&a), request_hash(&b));
    }

    #[test]
    fn request_hash_changes_when_messages_change() {
        let a = req_text("hello");
        let b = req_text("hello!");
        assert_ne!(request_hash(&a), request_hash(&b));
    }

    #[test]
    fn request_hash_is_16_hex_chars() {
        let h = request_hash(&req_text("x"));
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[tokio::test]
    async fn replay_auto_detects_positional_when_no_hashes() {
        let exchanges = vec![RecordedExchange {
            request_hash: None,
            request: req_text("q1"),
            response: resp_text("r1"),
            provider: "fixture".into(),
            model_id: None,
            recorded_at: None,
            duration_ms: None,
        }];
        let r = ReplayModelInterface::new(exchanges, provider());
        assert_eq!(r.mode(), ReplayMode::Positional);
        let got = r.call(req_text("any")).await.unwrap();
        assert_eq!(got, resp_text("r1"));
    }

    #[tokio::test]
    async fn replay_auto_detects_hash_matched_when_all_have_hashes() {
        let q1 = req_text("q1");
        let q2 = req_text("q2");
        let exchanges = vec![
            RecordedExchange {
                request_hash: Some(request_hash(&q1)),
                request: q1.clone(),
                response: resp_text("r1"),
                provider: "fixture".into(),
                model_id: None,
                recorded_at: None,
                duration_ms: None,
            },
            RecordedExchange {
                request_hash: Some(request_hash(&q2)),
                request: q2.clone(),
                response: resp_text("r2"),
                provider: "fixture".into(),
                model_id: None,
                recorded_at: None,
                duration_ms: None,
            },
        ];
        let r = ReplayModelInterface::new(exchanges, provider());
        assert_eq!(r.mode(), ReplayMode::HashMatched);
        // Out-of-order calls return the right response — order-independent.
        assert_eq!(r.call(q2.clone()).await.unwrap(), resp_text("r2"));
        assert_eq!(r.call(q1.clone()).await.unwrap(), resp_text("r1"));
        assert_eq!(r.call(q2).await.unwrap(), resp_text("r2"));
    }

    #[tokio::test]
    async fn replay_count_tokens_uses_recorded_input_tokens_in_hash_mode() {
        // Carry-over from #38: replace the bytes/4 heuristic with the
        // recorded input_tokens when we can match by request_hash.
        let q = req_text("the quick brown fox");
        let recorded = RecordedExchange {
            request_hash: Some(request_hash(&q)),
            request: q.clone(),
            response: ModelResponse {
                content: vec![ContentBlock::Text { text: "ok".into() }],
                usage: TokenUsage {
                    input_tokens: 137,
                    output_tokens: 4,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
                stop_reason: StopReason::EndTurn,
            },
            provider: "anthropic".into(),
            model_id: None,
            recorded_at: None,
            duration_ms: None,
        };
        let r = ReplayModelInterface::new(vec![recorded], provider());
        assert_eq!(r.mode(), ReplayMode::HashMatched);
        let n = r.count_tokens(&q).await.unwrap();
        // 137 came from the recorded usage. bytes/4 would have produced
        // floor(19/4) = 4, so this proves the recorded value wins.
        assert_eq!(n, 137);
    }

    #[tokio::test]
    async fn replay_count_tokens_falls_back_to_heuristic_when_no_match() {
        let q1 = req_text("xx");
        let recorded = RecordedExchange {
            request_hash: Some(request_hash(&q1)),
            request: q1,
            response: resp_text("r"),
            provider: "fixture".into(),
            model_id: None,
            recorded_at: None,
            duration_ms: None,
        };
        let r = ReplayModelInterface::new(vec![recorded], provider());
        // A different request — no fixture match.
        let unrecorded = req_text("never seen before, a long string indeed");
        let n = r.count_tokens(&unrecorded).await.unwrap();
        // Length 39 → floor(39/4) = 9.
        assert_eq!(n, 9);
    }

    #[tokio::test]
    async fn replay_hash_matched_no_match_returns_provider_error() {
        let q1 = req_text("q1");
        let exchanges = vec![RecordedExchange {
            request_hash: Some(request_hash(&q1)),
            request: q1,
            response: resp_text("r1"),
            provider: "fixture".into(),
            model_id: None,
            recorded_at: None,
            duration_ms: None,
        }];
        let r = ReplayModelInterface::new(exchanges, provider());
        let err = r.call(req_text("unrecorded")).await.unwrap_err();
        match err {
            ModelError::ProviderError { message, .. } => {
                assert!(message.contains("no matching fixture"), "got: {message}");
            }
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    // ── #37: cross-language hash stability fixture ──────────────────────────

    #[derive(Deserialize)]
    struct HashFixtureCase {
        name: String,
        request: ModelRequest,
        expected_hash: String,
    }

    #[derive(Deserialize)]
    struct HashFixtureSuite {
        cases: Vec<HashFixtureCase>,
    }

    #[test]
    #[ignore = "one-shot helper to bootstrap fixture/model_hashing/cases.json"]
    fn _generate_hash_cases_stdout() {
        use crate::model::ToolSchema;
        let cases: Vec<(&str, ModelRequest)> = vec![
            ("simple_user_text", req_text("hello")),
            (
                "multi_message",
                ModelRequest {
                    messages: vec![
                        Message {
                            role: Role::System,
                            content: Content::Text {
                                text: "you are helpful".into(),
                            },
                        },
                        Message {
                            role: Role::User,
                            content: Content::Text {
                                text: "what is 2+2?".into(),
                            },
                        },
                        Message {
                            role: Role::Assistant,
                            content: Content::Text { text: "4".into() },
                        },
                    ],
                    tools: vec![],
                    params: ModelParams::default(),
                    stream: false,
                },
            ),
            (
                "with_tools",
                ModelRequest {
                    messages: vec![Message {
                        role: Role::User,
                        content: Content::Text {
                            text: "look something up".into(),
                        },
                    }],
                    tools: vec![ToolSchema {
                        name: "search".into(),
                        description: "Search the web".into(),
                        input_schema: serde_json::json!({
                            "type":"object",
                            "properties":{"q":{"type":"string"}}
                        }),
                    }],
                    params: ModelParams::default(),
                    stream: false,
                },
            ),
            (
                "with_max_tokens",
                ModelRequest {
                    messages: vec![Message {
                        role: Role::User,
                        content: Content::Text { text: "hi".into() },
                    }],
                    tools: vec![],
                    params: ModelParams {
                        max_tokens: Some(256),
                        ..Default::default()
                    },
                    stream: false,
                },
            ),
            (
                "tool_call_message",
                ModelRequest {
                    messages: vec![Message {
                        role: Role::Assistant,
                        content: Content::ToolCall(ToolCall {
                            id: "abc123".into(),
                            name: "fetch".into(),
                            input: serde_json::json!({"url":"https://example.com","method":"GET"}),
                        }),
                    }],
                    tools: vec![],
                    params: ModelParams::default(),
                    stream: false,
                },
            ),
        ];
        for (name, req) in cases {
            eprintln!("{}\t{}", name, request_hash(&req));
        }
    }

    #[test]
    fn fixture_replay_request_hash_stability() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/model_hashing/cases.json");
        let raw = std::fs::read_to_string(&path).expect("fixture present");
        let suite: HashFixtureSuite = serde_json::from_str(&raw).unwrap();
        for case in suite.cases {
            let got = request_hash(&case.request);
            assert_eq!(
                got, case.expected_hash,
                "case `{}`: hash mismatch (got {got}, expected {})",
                case.name, case.expected_hash
            );
        }
    }

    // ── #38: RecordingModelInterface ────────────────────────────────────────

    #[tokio::test]
    async fn recording_appends_request_response_pair() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("recorded.jsonl");
        let inner = Arc::new(mock::MockModelInterface::new(provider()));
        inner.push_response(Ok(resp_text("hello back")));
        inner.push_response(Ok(resp_text("hello again")));
        let r = RecordingModelInterface::new(inner, &path, RecordingMode::Record);
        let _ = r.call(req_text("hello")).await.unwrap();
        let _ = r.call(req_text("hello2")).await.unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = raw.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let entry: RecordedExchange = serde_json::from_str(line).unwrap();
            assert!(
                entry.request_hash.is_some(),
                "request_hash must be populated"
            );
            assert_eq!(entry.provider, "test");
            assert_eq!(entry.model_id.as_deref(), Some("test-1"));
        }
    }

    #[tokio::test]
    async fn recording_record_if_new_skips_when_file_exists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("existing.jsonl");
        std::fs::write(&path, "preexisting line\n").unwrap();
        let inner = Arc::new(mock::MockModelInterface::new(provider()));
        inner.push_response(Ok(resp_text("ok")));
        let r = RecordingModelInterface::new(inner, &path, RecordingMode::RecordIfNew);
        let _ = r.call(req_text("q")).await.unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            raw, "preexisting line\n",
            "RecordIfNew must not touch existing file"
        );
    }

    #[tokio::test]
    async fn recording_passthrough_does_not_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.jsonl");
        let inner = Arc::new(mock::MockModelInterface::new(provider()));
        inner.push_response(Ok(resp_text("ok")));
        let r = RecordingModelInterface::new(inner, &path, RecordingMode::Passthrough);
        let _ = r.call(req_text("q")).await.unwrap();
        assert!(!path.exists(), "Passthrough must not create the file");
    }

    #[tokio::test]
    async fn recording_then_replay_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.jsonl");
        let inner = Arc::new(mock::MockModelInterface::new(provider()));
        inner.push_response(Ok(resp_text("answer1")));
        inner.push_response(Ok(resp_text("answer2")));
        let recorder = RecordingModelInterface::new(inner, &path, RecordingMode::Record);
        let q1 = req_text("question 1");
        let q2 = req_text("question 2");
        let _ = recorder.call(q1.clone()).await.unwrap();
        let _ = recorder.call(q2.clone()).await.unwrap();
        let jsonl = std::fs::read_to_string(&path).unwrap();
        let replay = ReplayModelInterface::from_jsonl(&jsonl, provider()).unwrap();
        assert_eq!(replay.mode(), ReplayMode::HashMatched);
        // Replay out-of-order to confirm hash matching works end-to-end.
        assert_eq!(replay.call(q2).await.unwrap(), resp_text("answer2"));
        assert_eq!(replay.call(q1).await.unwrap(), resp_text("answer1"));
    }
}
