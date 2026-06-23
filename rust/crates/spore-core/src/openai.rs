//! Issue #40 — `OpenAIModelInterface`: real OpenAI Chat Completions client.
//!
//! Implements [`ModelInterface`] against `${base_url}/v1/chat/completions`.
//! Translates [`ModelRequest`] / [`ModelResponse`] to and from the OpenAI
//! wire format, parses the OpenAI SSE event stream for `call_streaming`,
//! handles tool-call delta accumulation, and maps HTTP errors to typed
//! [`ModelError`] variants with retry/backoff for transient failures.
//!
//! ## Provider-specific shape
//! - System messages become `{role: "system", content: ...}` entries in the
//!   `messages` array (Anthropic extracts them into a top-level field — OpenAI
//!   does not).
//! - Assistant tool calls travel in a `tool_calls` array on the assistant
//!   message. Tool results travel as standalone messages with `role: "tool"`
//!   and a `tool_call_id` linking back to the call.
//! - Reasoning models (`o1`, `o3`, `o4*`) do not accept `temperature` and
//!   replace `max_tokens` with `max_completion_tokens`. Detection is by
//!   model-id prefix, overridable via [`OpenAICompat`] / [`with_compat`]
//!   ([`OpenAIModelInterface::with_compat`]) for models the prefix heuristic
//!   does not recognize — also the vehicle for the `developer` role and the
//!   `reasoning_effort` request field (SC-27).
//! - Streaming SSE chunks contain `delta.content` (text), `delta.tool_calls`
//!   (partial tool calls indexed and accumulated across chunks), and end with
//!   a literal `data: [DONE]` line. The final usage block only appears when
//!   the request set `stream_options: {include_usage: true}`.
//!
//! ## Token counting
//! OpenAI does not expose a counter endpoint. We use the bytes/4 heuristic
//! consistent with [`ReplayModelInterface::count_tokens`]; accuracy is
//! sufficient for compaction decisions, exact counts come from response
//! `usage`. A future revision may pull in `tiktoken-rs`.

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::harness::BoxFut;
use crate::model::{
    Content, ContentBlock, ModelError, ModelInterface, ModelRequest, ModelResponse, ModelStream,
    ProviderInfo, Role, StopReason, StreamEvent, TokenUsage, ToolCall, ToolSchema,
};

// ============================================================================
// OpenAIModelInterface
// ============================================================================

/// Capability declarations that BEAT OpenAI's id-prefix heuristics (SC-27).
///
/// The built-in [`is_reasoning_model`](OpenAIModelInterface::is_reasoning_model)
/// check only recognizes the `o1`/`o3`/`o4` families. A model served behind an
/// OpenAI-compatible endpoint (a local server, a renamed deployment, a newer
/// family the heuristic predates) gets no way to declare that it wants
/// reasoning-model request shaping. `OpenAICompat`, supplied via
/// [`with_compat`](OpenAIModelInterface::with_compat), is that vehicle: every
/// field is OR'd over the id heuristic, never subtracted from it, so the default
/// (all `false`) leaves the recognized o-series behavior byte-identical.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAICompat {
    /// Treat this model as a reasoning model regardless of its id (beats the
    /// `o1`/`o3`/`o4` heuristic): send `max_completion_tokens` instead of
    /// `max_tokens` and drop `temperature`/`top_p`/`stop`, which reasoning models
    /// reject.
    pub reasoning_model: bool,
    /// Route the system message to the `developer` role instead of `system` —
    /// OpenAI's reasoning-model convention, which some compatible servers require.
    pub developer_role: bool,
    /// Emit a `reasoning_effort` field carrying
    /// [`ModelParams::reasoning_effort`](crate::model::ModelParams::reasoning_effort)
    /// (`"low"|"medium"|"high"|"max"`) when it is set. No-op when the caller
    /// leaves `reasoning_effort` unset.
    pub supports_reasoning_effort: bool,
}

pub struct OpenAIModelInterface {
    api_key: String,
    model_id: String,
    base_url: String,
    timeout: Duration,
    max_retries: u32,
    /// Explicit override for the window reported by [`provider`](ModelInterface::provider)
    /// (SC-6). `None` defers to the static [`context_window`](Self::context_window)
    /// table; `Some(n)` pins it.
    context_window_override: Option<u32>,
    /// Capability declarations that override the id heuristics (SC-27).
    compat: OpenAICompat,
    http_client: Arc<reqwest::Client>,
}

impl std::fmt::Debug for OpenAIModelInterface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIModelInterface")
            .field("api_key", &"<redacted>")
            .field("model_id", &self.model_id)
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("context_window_override", &self.context_window_override)
            .field("compat", &self.compat)
            .finish()
    }
}

impl OpenAIModelInterface {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
    pub const DEFAULT_MAX_RETRIES: u32 = 3;

    pub fn new(api_key: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model_id: model_id.into(),
            base_url: Self::DEFAULT_BASE_URL.into(),
            timeout: Self::DEFAULT_TIMEOUT,
            max_retries: Self::DEFAULT_MAX_RETRIES,
            context_window_override: None,
            compat: OpenAICompat::default(),
            http_client: Arc::new(reqwest::Client::new()),
        }
    }

    pub fn from_env(env_var: &str, model_id: impl Into<String>) -> Result<Self, ModelError> {
        let key = std::env::var(env_var).map_err(|_| ModelError::ProviderError {
            code: 0,
            message: format!("env var `{env_var}` not set"),
        })?;
        if key.trim().is_empty() {
            return Err(ModelError::ProviderError {
                code: 0,
                message: format!("env var `{env_var}` is empty"),
            });
        }
        Ok(Self::new(key, model_id))
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Override the window reported by [`provider`](ModelInterface::provider)
    /// (SC-6 / SC-4): the value the harness's compaction budget sizes itself to
    /// (via the context manager's `resolve_context_length`, issue #141). Use it to
    /// pin the window for a model the static table predates, or for a local
    /// OpenAI-compatible deployment whose id the table does not recognize. OpenAI
    /// has no `num_ctx`-style knob, so this affects reporting (and thus the
    /// compaction budget) only.
    pub fn with_context_window(mut self, n: u32) -> Self {
        self.context_window_override = Some(n);
        self
    }

    /// Declare model capabilities that BEAT the id-prefix heuristics (SC-27).
    ///
    /// See [`OpenAICompat`]. Use it for a reasoning model the `o1`/`o3`/`o4`
    /// [`is_reasoning_model`](Self::is_reasoning_model) check does not recognize —
    /// e.g. a local server or a renamed deployment — so the request still carries
    /// the developer role, `max_completion_tokens`, and `reasoning_effort`.
    pub fn with_compat(mut self, compat: OpenAICompat) -> Self {
        self.compat = compat;
        self
    }

    pub fn with_http_client(mut self, c: Arc<reqwest::Client>) -> Self {
        self.http_client = c;
        self
    }

    /// Context window for known model ids.
    pub fn context_window(model_id: &str) -> u32 {
        match model_id {
            id if id.starts_with("gpt-4o") => 128_000,
            id if id.starts_with("gpt-4.1") => 1_000_000,
            id if id.starts_with("o3") || id.starts_with("o4") => 200_000,
            id if id.starts_with("o1") => 128_000,
            _ => 0,
        }
    }

    /// True if this is an o-series reasoning model (different parameter
    /// constraints — no `temperature`, uses `max_completion_tokens`).
    pub fn is_reasoning_model(model_id: &str) -> bool {
        let id = model_id;
        id.starts_with("o1") || id.starts_with("o3") || id.starts_with("o4")
    }
}

// ============================================================================
// Wire-format types (OpenAI Chat Completions API)
// ============================================================================

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    /// SC-27: `"low"|"medium"|"high"|"max"`, emitted only when the model is
    /// treated as reasoning-capable AND `OpenAICompat.supports_reasoning_effort`
    /// is set. Absent (skipped) otherwise, so non-reasoning requests stay
    /// byte-identical.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAITool>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAIToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str, // always "function"
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize)]
struct OpenAIFunctionCall {
    name: String,
    /// Arguments are a JSON-encoded string per OpenAI's wire format.
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    kind: &'static str, // always "function"
    function: OpenAIToolFunction,
}

#[derive(Debug, Serialize)]
struct OpenAIToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    #[serde(default)]
    choices: Vec<OpenAIChoice>,
    #[serde(default)]
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    #[serde(default)]
    message: OpenAIResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAIResponseToolCall>,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponseToolCall {
    id: String,
    #[serde(default)]
    function: OpenAIResponseFunctionCall,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIResponseFunctionCall {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Debug, Default, Deserialize, Clone, Copy)]
struct OpenAIUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Default, Deserialize, Clone, Copy)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorBody {
    #[serde(default)]
    error: Option<OpenAIErrorInner>,
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorInner {
    #[serde(default)]
    message: Option<String>,
}

// ============================================================================
// Conversions
// ============================================================================

fn build_request(
    model_id: &str,
    req: &ModelRequest,
    stream: bool,
    compat: &OpenAICompat,
) -> OpenAIRequest {
    let messages: Vec<OpenAIMessage> = req
        .messages
        .iter()
        .map(|m| message_to_openai(m, compat))
        .collect();

    let tools: Vec<OpenAITool> = req
        .tools
        .iter()
        .map(|t: &ToolSchema| OpenAITool {
            kind: "function",
            function: OpenAIToolFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect();

    // SC-27: `OpenAICompat.reasoning_model` is OR'd OVER the id heuristic, so a
    // model the `o1`/`o3`/`o4` prefix check misses still gets reasoning shaping.
    let is_reasoning =
        OpenAIModelInterface::is_reasoning_model(model_id) || compat.reasoning_model;
    let (max_tokens, max_completion_tokens) = if is_reasoning {
        (None, req.params.max_tokens)
    } else {
        (req.params.max_tokens, None)
    };
    let temperature = if is_reasoning {
        None
    } else {
        req.params.temperature
    };
    // o-series reasoning models reject sampling-control params: drop `top_p` and
    // `stop` for them, exactly as `temperature` is dropped above.
    let (top_p, stop) = if is_reasoning {
        (None, Vec::new())
    } else {
        (req.params.top_p, req.params.stop_sequences.clone())
    };
    // SC-27: emit `reasoning_effort` only for a reasoning model whose compat
    // opted in AND whose caller set an effort level. Absent otherwise so
    // non-reasoning requests stay byte-identical.
    let reasoning_effort = if is_reasoning && compat.supports_reasoning_effort {
        req.params.reasoning_effort.map(reasoning_effort_str)
    } else {
        None
    };

    OpenAIRequest {
        model: model_id.into(),
        messages,
        max_tokens,
        max_completion_tokens,
        reasoning_effort,
        temperature,
        top_p,
        stop,
        tools,
        stream,
        stream_options: if stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        },
    }
}

/// SC-27: the OpenAI `reasoning_effort` wire string. Mirrors Ollama's `think`
/// level mapping so the two providers agree on the level vocabulary.
fn reasoning_effort_str(effort: crate::model::ReasoningEffort) -> &'static str {
    use crate::model::ReasoningEffort::*;
    match effort {
        Low => "low",
        Medium => "medium",
        High => "high",
        Max => "max",
    }
}

fn message_to_openai(m: &crate::model::Message, compat: &OpenAICompat) -> OpenAIMessage {
    let role = match m.role {
        // SC-27: reasoning models use the `developer` role for system content.
        Role::System if compat.developer_role => "developer",
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    match &m.content {
        Content::Text { text } => OpenAIMessage {
            role: role.into(),
            content: Some(text.clone()),
            tool_calls: vec![],
            tool_call_id: None,
        },
        Content::ToolCall(call) => OpenAIMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: vec![OpenAIToolCall {
                id: call.id.clone(),
                kind: "function",
                function: OpenAIFunctionCall {
                    name: call.name.clone(),
                    arguments: serde_json::to_string(&call.input).unwrap_or_else(|_| "{}".into()),
                },
            }],
            tool_call_id: None,
        },
        Content::ToolResult(r) => OpenAIMessage {
            role: "tool".into(),
            content: Some(r.content.clone()),
            tool_calls: vec![],
            tool_call_id: Some(r.tool_use_id.clone()),
        },
        // OpenAI's chat-completions image input uses a content-parts array
        // (`{type: "image_url", image_url: {url: "data:..."}}`). The harness
        // does not currently emit image content into requests, so we serialize
        // a textual placeholder rather than introduce a heterogeneous shape.
        Content::Image { media_type, .. } => OpenAIMessage {
            role: role.into(),
            content: Some(format!("[image: {media_type}]")),
            tool_calls: vec![],
            tool_call_id: None,
        },
    }
}

fn parse_response(body: OpenAIResponse) -> ModelResponse {
    let choice = body.choices.into_iter().next().unwrap_or(OpenAIChoice {
        message: OpenAIResponseMessage::default(),
        finish_reason: None,
    });

    let mut content: Vec<ContentBlock> = Vec::new();
    if let Some(reasoning) = choice.message.reasoning {
        if !reasoning.is_empty() {
            content.push(ContentBlock::Thinking { text: reasoning });
        }
    }
    if let Some(text) = choice.message.content {
        if !text.is_empty() {
            content.push(ContentBlock::Text { text });
        }
    }
    for tc in choice.message.tool_calls {
        let input: serde_json::Value = if tc.function.arguments.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(&tc.function.arguments)
                .unwrap_or_else(|_| serde_json::Value::String(tc.function.arguments.clone()))
        };
        content.push(ContentBlock::ToolUse(ToolCall {
            id: tc.id,
            name: tc.function.name,
            input,
        }));
    }

    let cache_read = body
        .usage
        .prompt_tokens_details
        .and_then(|d| d.cached_tokens);

    ModelResponse {
        content,
        usage: TokenUsage {
            input_tokens: body.usage.prompt_tokens,
            output_tokens: body.usage.completion_tokens,
            cache_read_tokens: cache_read,
            // OpenAI does not report cache writes directly.
            cache_write_tokens: None,
        },
        stop_reason: parse_stop_reason(choice.finish_reason.as_deref()),
    }
}

fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("stop") => StopReason::EndTurn,
        _ => StopReason::EndTurn,
    }
}

// ============================================================================
// HTTP plumbing with retry
// ============================================================================

async fn send_with_retry(
    request_builder: impl Fn() -> reqwest::RequestBuilder,
    max_retries: u32,
    timeout: Duration,
) -> Result<reqwest::Response, ModelError> {
    let mut attempt: u32 = 0;
    loop {
        let req = request_builder().timeout(timeout);
        let result = req.send().await;
        match result {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(resp);
                }
                let code = status.as_u16();
                let retryable = matches!(code, 408 | 425 | 429 | 500 | 502 | 503 | 504);
                if retryable && attempt < max_retries {
                    let retry_after = resp
                        .headers()
                        .get("retry-after")
                        .and_then(|h| h.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok())
                        .map(Duration::from_secs);
                    let delay = retry_after.unwrap_or_else(|| backoff_delay(attempt));
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
                return Err(map_status_error(resp).await);
            }
            Err(e) if e.is_timeout() && attempt < max_retries => {
                tokio::time::sleep(backoff_delay(attempt)).await;
                attempt += 1;
                continue;
            }
            Err(e) if e.is_timeout() => return Err(ModelError::Timeout),
            Err(e) => {
                return Err(ModelError::Transport {
                    message: format!("HTTP transport error: {e}"),
                })
            }
        }
    }
}

fn backoff_delay(attempt: u32) -> Duration {
    let base_ms: u64 = 500u64.saturating_mul(1u64 << attempt.min(6));
    Duration::from_millis(base_ms.min(30_000))
}

async fn map_status_error(resp: reqwest::Response) -> ModelError {
    let status = resp.status();
    let code = status.as_u16();
    let retry_after = resp
        .headers()
        .get("retry-after")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs);
    let body_text = resp.text().await.unwrap_or_default();
    let message = serde_json::from_str::<OpenAIErrorBody>(&body_text)
        .ok()
        .and_then(|b| b.error.and_then(|e| e.message))
        .unwrap_or_else(|| body_text.chars().take(500).collect());
    match code {
        429 => ModelError::RateLimited { retry_after },
        408 | 504 => ModelError::Timeout,
        _ => ModelError::ProviderError { code, message },
    }
}

// ============================================================================
// ModelInterface impl
// ============================================================================

impl ModelInterface for OpenAIModelInterface {
    fn call<'a>(&'a self, request: ModelRequest) -> BoxFut<'a, Result<ModelResponse, ModelError>> {
        Box::pin(async move {
        let body = build_request(&self.model_id, &request, false, &self.compat);
        let url = format!("{}/chat/completions", self.base_url);
        let api_key = self.api_key.clone();
        let body = serde_json::to_string(&body).map_err(|e| ModelError::ProviderError {
            code: 0,
            message: format!("request encode failed: {e}"),
        })?;
        let client = self.http_client.clone();
        let resp = send_with_retry(
            || {
                client
                    .post(&url)
                    .header("authorization", format!("Bearer {api_key}"))
                    .header("content-type", "application/json")
                    .body(body.clone())
            },
            self.max_retries,
            self.timeout,
        )
        .await?;
        let parsed: OpenAIResponse = resp.json().await.map_err(|e| ModelError::ProviderError {
            code: 0,
            message: format!("response decode failed: {e}"),
        })?;
        Ok(parse_response(parsed))
        })
    }

    fn call_streaming<'a>(
        &'a self,
        request: ModelRequest,
    ) -> BoxFut<'a, Result<ModelStream, ModelError>> {
        Box::pin(async move {
        let body = build_request(&self.model_id, &request, true, &self.compat);
        let url = format!("{}/chat/completions", self.base_url);
        let body = serde_json::to_string(&body).map_err(|e| ModelError::ProviderError {
            code: 0,
            message: format!("request encode failed: {e}"),
        })?;
        let client = self.http_client.clone();
        let api_key = self.api_key.clone();
        // Route through `send_with_retry` so opening a stream gets the same
        // retry/backoff on transient failures (timeouts, 429/5xx) as the
        // non-streaming `call` path. Retries happen BEFORE the body is consumed;
        // the closure rebuilds the request each attempt (`body.clone()`).
        let resp = send_with_retry(
            || {
                client
                    .post(&url)
                    .header("authorization", format!("Bearer {api_key}"))
                    .header("content-type", "application/json")
                    .header("accept", "text/event-stream")
                    .body(body.clone())
            },
            self.max_retries,
            self.timeout,
        )
        .await?;
        Ok::<ModelStream, ModelError>(Box::pin(sse_to_events(resp)))
        })
    }

    fn count_tokens<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxFut<'a, Result<u32, ModelError>> {
        Box::pin(async move {
        // OpenAI has no count_tokens endpoint. Use the bytes/4 heuristic
        // consistent with ReplayModelInterface — sufficient for compaction
        // decisions; exact counts come back via response usage.
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
        Ok((n / 4) as u32)
        })
    }

    fn provider(&self) -> ProviderInfo {
        ProviderInfo {
            name: "openai".into(),
            model_id: self.model_id.clone(),
            // SC-6: an explicit override wins over the static table.
            context_window: self
                .context_window_override
                .unwrap_or_else(|| Self::context_window(&self.model_id)),
        }
    }
}

// ============================================================================
// SSE stream parsing — OpenAI Chat Completions
// ============================================================================

/// OpenAI streams chat-completion delta chunks. Each `data:` line carries a
/// JSON object shaped like:
///
/// ```json
/// {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}
/// ```
///
/// Tool calls arrive as partial entries in `delta.tool_calls`, indexed; the
/// `id` and `function.name` arrive on the first chunk for a given index, and
/// subsequent chunks for the same index carry incremental `function.arguments`
/// JSON-fragment strings. The stream ends with `data: [DONE]`. When
/// `stream_options.include_usage` was set, the final non-`[DONE]` chunk also
/// carries `usage`.
fn sse_to_events(
    resp: reqwest::Response,
) -> impl futures_core::Stream<Item = Result<StreamEvent, ModelError>> + Send + 'static {
    async_stream::stream! {
        let stream = resp.bytes_stream();
        futures_util::pin_mut!(stream);
        let mut buf = crate::model::ByteLineBuffer::new();
        let mut usage = TokenUsage::default();
        let mut stop_reason = StopReason::EndTurn;
        let mut started = false;
        let mut tool_indices_seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut content_index_emitted = false;
        let mut content_index: u32 = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Err(ModelError::StreamInterrupted {
                        message: format!("stream chunk error: {e}"),
                    });
                    return;
                }
            };
            buf.push(&chunk);
            while let Some(raw_line) = buf.next_line(b"\n") {
                let line = raw_line.trim_end_matches('\r');
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim_start();
                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    yield Ok(StreamEvent::MessageStop { usage, stop_reason });
                    return;
                }
                let value: serde_json::Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if !started {
                    started = true;
                    yield Ok(StreamEvent::MessageStart);
                }
                if let Some(u) = value.get("usage") {
                    if let Some(pt) = u.get("prompt_tokens").and_then(|v| v.as_u64()) {
                        usage.input_tokens = pt as u32;
                    }
                    if let Some(ct) = u.get("completion_tokens").and_then(|v| v.as_u64()) {
                        usage.output_tokens = ct as u32;
                    }
                    if let Some(d) = u.get("prompt_tokens_details") {
                        if let Some(c) = d.get("cached_tokens").and_then(|v| v.as_u64()) {
                            usage.cache_read_tokens = Some(c as u32);
                        }
                    }
                }
                let Some(choice) = value.get("choices").and_then(|c| c.get(0)) else {
                    continue;
                };
                if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    stop_reason = parse_stop_reason(Some(fr));
                }
                let Some(delta) = choice.get("delta") else { continue };
                if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        if !content_index_emitted {
                            content_index_emitted = true;
                        }
                        yield Ok(StreamEvent::ContentBlockDelta {
                            index: content_index,
                            delta: text.into(),
                        });
                    }
                }
                if let Some(reasoning) = delta.get("reasoning").and_then(|v| v.as_str()) {
                    if !reasoning.is_empty() {
                        yield Ok(StreamEvent::ThinkingDelta {
                            index: content_index,
                            delta: reasoning.into(),
                        });
                    }
                }
                if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                    // Tool call indices are independent of the text content
                    // index; shift by 1 to keep them disjoint from index 0
                    // (which conventionally carries text).
                    for tc in tcs {
                        let i = tc
                            .get("index")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as u32;
                        let event_index = i + 1;
                        if !tool_indices_seen.contains(&event_index) {
                            tool_indices_seen.insert(event_index);
                            if content_index_emitted {
                                yield Ok(StreamEvent::ContentBlockStop { index: content_index });
                                content_index_emitted = false;
                                content_index = event_index;
                            }
                            // The id + function.name arrive on this first chunk for
                            // the index; emit ToolUseStart so they aren't lost when
                            // only argument fragments follow.
                            let name = tc
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or_default()
                                .to_string();
                            let id = tc
                                .get("id")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| format!("call_{event_index}"));
                            yield Ok(StreamEvent::ToolUseStart { index: event_index, id, name });
                        }
                        let arg_delta = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if !arg_delta.is_empty() {
                            yield Ok(StreamEvent::ToolUseDelta {
                                index: event_index,
                                partial_json: arg_delta.into(),
                            });
                        }
                    }
                }
            }
        }
        // If the stream ended without an explicit [DONE] marker we still emit
        // MessageStop so consumers see a terminator.
        yield Ok(StreamEvent::MessageStop { usage, stop_reason });
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Content, Message, ModelParams, ModelRequest, Role, ToolResult};

    fn user(text: &str) -> Message {
        Message {
            role: Role::User,
            content: Content::Text { text: text.into() },
        }
    }

    fn sys(text: &str) -> Message {
        Message {
            role: Role::System,
            content: Content::Text { text: text.into() },
        }
    }

    fn req(messages: Vec<Message>) -> ModelRequest {
        ModelRequest {
            messages,
            tools: vec![],
            params: ModelParams::default(),
            stream: false,
        }
    }

    // ── build_request ───────────────────────────────────────────────────────

    #[test]
    fn build_request_keeps_system_in_messages() {
        let r = req(vec![sys("be helpful"), user("hi")]);
        let body = build_request("gpt-4o", &r, false, &OpenAICompat::default());
        assert_eq!(body.messages.len(), 2);
        assert_eq!(body.messages[0].role, "system");
        assert_eq!(body.messages[1].role, "user");
    }

    #[test]
    fn build_request_sets_max_tokens_for_chat_models() {
        let mut r = req(vec![user("hi")]);
        r.params.max_tokens = Some(256);
        let body = build_request("gpt-4o", &r, false, &OpenAICompat::default());
        assert_eq!(body.max_tokens, Some(256));
        assert_eq!(body.max_completion_tokens, None);
    }

    #[test]
    fn build_request_o_series_uses_max_completion_tokens_and_no_temperature() {
        let mut r = req(vec![user("hi")]);
        r.params.max_tokens = Some(512);
        r.params.temperature = Some(0.7);
        let body = build_request("o3", &r, false, &OpenAICompat::default());
        assert_eq!(body.max_tokens, None);
        assert_eq!(body.max_completion_tokens, Some(512));
        assert_eq!(body.temperature, None);
    }

    #[test]
    fn build_request_o_series_drops_top_p_and_stop() {
        let mut r = req(vec![user("hi")]);
        r.params.top_p = Some(0.9);
        r.params.stop_sequences = vec!["STOP".into()];
        // Reasoning model: both sampling-control params are dropped.
        let reasoning = build_request("o3", &r, false, &OpenAICompat::default());
        assert_eq!(reasoning.top_p, None);
        assert!(reasoning.stop.is_empty());
        // Chat model: both are preserved.
        let chat = build_request("gpt-4o", &r, false, &OpenAICompat::default());
        assert_eq!(chat.top_p, Some(0.9));
        assert_eq!(chat.stop, vec!["STOP".to_string()]);
    }

    #[test]
    fn build_request_o4_is_reasoning() {
        assert!(OpenAIModelInterface::is_reasoning_model("o4-mini"));
        assert!(OpenAIModelInterface::is_reasoning_model("o3"));
        assert!(OpenAIModelInterface::is_reasoning_model("o1-pro"));
        assert!(!OpenAIModelInterface::is_reasoning_model("gpt-4o"));
    }

    #[test]
    fn build_request_maps_tool_call_to_assistant_tool_calls() {
        let r = req(vec![Message {
            role: Role::Assistant,
            content: Content::ToolCall(ToolCall {
                id: "call-1".into(),
                name: "fetch".into(),
                input: serde_json::json!({"url": "x"}),
            }),
        }]);
        let body = build_request("gpt-4o", &r, false, &OpenAICompat::default());
        let s = serde_json::to_string(&body.messages[0]).unwrap();
        assert!(s.contains("\"role\":\"assistant\""), "wire: {s}");
        assert!(s.contains("\"tool_calls\""), "wire: {s}");
        assert!(s.contains("\"id\":\"call-1\""));
        // arguments must be a JSON-encoded string, not a nested object
        assert!(s.contains("\"arguments\":\"{"), "wire: {s}");
    }

    #[test]
    fn build_request_maps_tool_result_to_tool_role_message() {
        let r = req(vec![Message {
            role: Role::Tool,
            content: Content::ToolResult(ToolResult {
                tool_use_id: "call-1".into(),
                content: "ok".into(),
                is_error: false,
            }),
        }]);
        let body = build_request("gpt-4o", &r, false, &OpenAICompat::default());
        assert_eq!(body.messages[0].role, "tool");
        let s = serde_json::to_string(&body.messages[0]).unwrap();
        assert!(s.contains("\"tool_call_id\":\"call-1\""), "wire: {s}");
        assert!(s.contains("\"content\":\"ok\""), "wire: {s}");
    }

    #[test]
    fn build_request_streaming_sets_include_usage() {
        let r = req(vec![user("hi")]);
        let body = build_request("gpt-4o", &r, true, &OpenAICompat::default());
        assert!(body.stream);
        assert!(body.stream_options.is_some());
    }

    // ── parse_response ──────────────────────────────────────────────────────

    #[test]
    fn parse_response_extracts_text_and_usage() {
        let body: OpenAIResponse = serde_json::from_value(serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": "hi there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2}
        }))
        .unwrap();
        let r = parse_response(body);
        assert_eq!(
            r.content,
            vec![ContentBlock::Text {
                text: "hi there".into()
            }]
        );
        assert_eq!(r.usage.input_tokens, 4);
        assert_eq!(r.usage.output_tokens, 2);
        assert_eq!(r.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn parse_response_extracts_tool_calls() {
        let body: OpenAIResponse = serde_json::from_value(serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{\"q\":\"rust\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        }))
        .unwrap();
        let r = parse_response(body);
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        match &r.content[0] {
            ContentBlock::ToolUse(tc) => {
                assert_eq!(tc.id, "c1");
                assert_eq!(tc.name, "search");
                assert_eq!(tc.input, serde_json::json!({"q": "rust"}));
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn parse_response_extracts_reasoning_as_thinking() {
        let body: OpenAIResponse = serde_json::from_value(serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning": "let me think...",
                    "content": "the answer is 4"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        }))
        .unwrap();
        let r = parse_response(body);
        assert!(matches!(r.content[0], ContentBlock::Thinking { .. }));
        assert!(matches!(r.content[1], ContentBlock::Text { .. }));
    }

    #[test]
    fn parse_response_extracts_cache_read_only() {
        let body: OpenAIResponse = serde_json::from_value(serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 50}
            }
        }))
        .unwrap();
        let r = parse_response(body);
        assert_eq!(r.usage.cache_read_tokens, Some(50));
        assert_eq!(r.usage.cache_write_tokens, None);
    }

    // ── stop reason ─────────────────────────────────────────────────────────

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(parse_stop_reason(Some("stop")), StopReason::EndTurn);
        assert_eq!(parse_stop_reason(Some("tool_calls")), StopReason::ToolUse);
        assert_eq!(parse_stop_reason(Some("length")), StopReason::MaxTokens);
        assert_eq!(parse_stop_reason(None), StopReason::EndTurn);
        assert_eq!(parse_stop_reason(Some("???")), StopReason::EndTurn);
    }

    // ── context window ─────────────────────────────────────────────────────

    #[test]
    fn context_window_known_and_unknown() {
        assert_eq!(OpenAIModelInterface::context_window("gpt-4o"), 128_000);
        assert_eq!(OpenAIModelInterface::context_window("gpt-4o-mini"), 128_000);
        assert_eq!(OpenAIModelInterface::context_window("o3"), 200_000);
        assert_eq!(OpenAIModelInterface::context_window("o4-mini"), 200_000);
        assert_eq!(OpenAIModelInterface::context_window("claude-x"), 0);
    }

    // ── provider() ──────────────────────────────────────────────────────────

    #[test]
    fn provider_info_uses_model_id() {
        let c = OpenAIModelInterface::new("test-key", "gpt-4o");
        let p = c.provider();
        assert_eq!(p.name, "openai");
        assert_eq!(p.model_id, "gpt-4o");
        assert_eq!(p.context_window, 128_000);
    }

    #[test]
    fn with_context_window_overrides_reported_window() {
        // SC-6: an unrecognized id reports 0; the override pins it.
        let bare = OpenAIModelInterface::new("k", "local-llama");
        assert_eq!(bare.provider().context_window, 0);
        let pinned = OpenAIModelInterface::new("k", "local-llama").with_context_window(32_768);
        assert_eq!(pinned.provider().context_window, 32_768);
    }

    // ── SC-27: with_compat beats the id heuristic ────────────────────────────

    #[test]
    fn compat_reasoning_model_beats_id_heuristic() {
        // An unrecognized id is NOT reasoning by the heuristic, so by default it
        // gets chat shaping (max_tokens, temperature kept).
        let mut r = req(vec![user("hi")]);
        r.params.max_tokens = Some(512);
        r.params.temperature = Some(0.7);
        let chat = build_request("local-reasoner", &r, false, &OpenAICompat::default());
        assert_eq!(chat.max_tokens, Some(512));
        assert_eq!(chat.max_completion_tokens, None);
        assert_eq!(chat.temperature, Some(0.7));

        // Declaring it reasoning flips the shaping even though the id is unknown.
        let compat = OpenAICompat {
            reasoning_model: true,
            ..Default::default()
        };
        let reasoning = build_request("local-reasoner", &r, false, &compat);
        assert_eq!(reasoning.max_tokens, None);
        assert_eq!(reasoning.max_completion_tokens, Some(512));
        assert_eq!(reasoning.temperature, None);
    }

    #[test]
    fn compat_developer_role_routes_system_message() {
        let r = req(vec![sys("be terse"), user("hi")]);
        // Default: system stays `system`.
        let plain = build_request("local-reasoner", &r, false, &OpenAICompat::default());
        assert_eq!(plain.messages[0].role, "system");
        // developer_role: the system message routes to `developer`.
        let compat = OpenAICompat {
            developer_role: true,
            ..Default::default()
        };
        let dev = build_request("local-reasoner", &r, false, &compat);
        assert_eq!(dev.messages[0].role, "developer");
        // User messages are untouched.
        assert_eq!(dev.messages[1].role, "user");
    }

    #[test]
    fn compat_emits_reasoning_effort_for_reasoning_model() {
        let mut r = req(vec![sys("x"), user("hi")]);
        r.params.reasoning_effort = Some(crate::model::ReasoningEffort::High);

        // SC-27 acceptance: an unrecognized model with reasoning + effort support
        // carries reasoning_effort AND the developer role on the wire.
        let compat = OpenAICompat {
            reasoning_model: true,
            developer_role: true,
            supports_reasoning_effort: true,
        };
        let body = build_request("local-reasoner", &r, false, &compat);
        assert_eq!(body.reasoning_effort, Some("high"));
        assert_eq!(body.messages[0].role, "developer");

        // Opt-in is required: without supports_reasoning_effort, the field is absent.
        let no_effort = OpenAICompat {
            reasoning_model: true,
            ..Default::default()
        };
        let body2 = build_request("local-reasoner", &r, false, &no_effort);
        assert_eq!(body2.reasoning_effort, None);

        // And it's gated on the model being reasoning at all (effort alone on a
        // chat model does nothing).
        let effort_only = OpenAICompat {
            supports_reasoning_effort: true,
            ..Default::default()
        };
        let body3 = build_request("gpt-4o", &r, false, &effort_only);
        assert_eq!(body3.reasoning_effort, None);
    }

    #[test]
    fn compat_reasoning_effort_serialized_on_wire() {
        let mut r = req(vec![user("hi")]);
        r.params.reasoning_effort = Some(crate::model::ReasoningEffort::Medium);
        let compat = OpenAICompat {
            reasoning_model: true,
            supports_reasoning_effort: true,
            ..Default::default()
        };
        let body = build_request("local-reasoner", &r, false, &compat);
        let s = serde_json::to_string(&body).unwrap();
        assert!(s.contains("\"reasoning_effort\":\"medium\""), "wire: {s}");
        // A bare (default-compat) request must NOT carry the field.
        let bare = build_request("gpt-4o", &r, false, &OpenAICompat::default());
        let sb = serde_json::to_string(&bare).unwrap();
        assert!(!sb.contains("reasoning_effort"), "absent by default: {sb}");
    }

    // ── from_env ────────────────────────────────────────────────────────────

    #[test]
    fn from_env_errors_when_unset() {
        std::env::remove_var("__SPORE_TEST_OPENAI_KEY_UNSET__");
        let err = OpenAIModelInterface::from_env("__SPORE_TEST_OPENAI_KEY_UNSET__", "gpt-4o")
            .unwrap_err();
        match err {
            ModelError::ProviderError { message, .. } => assert!(message.contains("not set")),
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    // ── End-to-end wiremock call() ──────────────────────────────────────────

    #[tokio::test]
    async fn call_against_mock_returns_response() {
        let server = wiremock::MockServer::start().await;
        let body = serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        });
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/chat/completions"))
            .and(wiremock::matchers::header(
                "authorization",
                "Bearer test-key",
            ))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        let client = OpenAIModelInterface::new("test-key", "gpt-4o").with_base_url(server.uri());
        let r = client.call(req(vec![user("hi")])).await.unwrap();
        assert_eq!(
            r.content,
            vec![ContentBlock::Text {
                text: "hello there".into()
            }]
        );
        assert_eq!(r.usage.input_tokens, 5);
    }

    #[tokio::test]
    async fn call_maps_429_to_rate_limited() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/chat/completions"))
            .respond_with(wiremock::ResponseTemplate::new(429).insert_header("retry-after", "7"))
            .mount(&server)
            .await;
        let client = OpenAIModelInterface::new("k", "gpt-4o")
            .with_base_url(server.uri())
            .with_max_retries(0);
        let err = client.call(req(vec![user("hi")])).await.unwrap_err();
        match err {
            ModelError::RateLimited { retry_after } => {
                assert_eq!(retry_after, Some(Duration::from_secs(7)));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn call_maps_400_to_provider_error_with_openai_message() {
        let server = wiremock::MockServer::start().await;
        let body = serde_json::json!({
            "error": {"type": "invalid_request_error", "message": "max_tokens must be > 0"}
        });
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/chat/completions"))
            .respond_with(wiremock::ResponseTemplate::new(400).set_body_json(body))
            .mount(&server)
            .await;
        let client = OpenAIModelInterface::new("k", "gpt-4o")
            .with_base_url(server.uri())
            .with_max_retries(0);
        let err = client.call(req(vec![user("hi")])).await.unwrap_err();
        match err {
            ModelError::ProviderError { code, message } => {
                assert_eq!(code, 400);
                assert!(message.contains("max_tokens"), "msg: {message}");
            }
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn call_retries_429_then_succeeds() {
        let server = wiremock::MockServer::start().await;
        let success_body = serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": "after retry"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        });
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/chat/completions"))
            .respond_with(wiremock::ResponseTemplate::new(429).insert_header("retry-after", "0"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/chat/completions"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&server)
            .await;
        let client = OpenAIModelInterface::new("k", "gpt-4o").with_base_url(server.uri());
        let r = client.call(req(vec![user("hi")])).await.unwrap();
        assert_eq!(
            r.content[0],
            ContentBlock::Text {
                text: "after retry".into()
            }
        );
    }

    // ── count_tokens fallback ──────────────────────────────────────────────

    #[tokio::test]
    async fn count_tokens_uses_bytes_over_four_heuristic() {
        let c = OpenAIModelInterface::new("k", "gpt-4o");
        let req = req(vec![user(&"a".repeat(40))]);
        assert_eq!(c.count_tokens(&req).await.unwrap(), 10);
    }

    // ── Streaming ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn streaming_emits_text_delta_then_stop() {
        let sse = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":5}}\n\n",
            "data: [DONE]\n\n",
        );
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/chat/completions"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_string(sse)
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&server)
            .await;
        let client = OpenAIModelInterface::new("k", "gpt-4o").with_base_url(server.uri());
        let mut stream = client.call_streaming(req(vec![user("hi")])).await.unwrap();
        let mut events: Vec<StreamEvent> = vec![];
        while let Some(ev) = stream.next().await {
            events.push(ev.unwrap());
        }
        // start, text, text, stop(=MessageStop)
        assert!(matches!(events[0], StreamEvent::MessageStart));
        let text_deltas: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text_deltas, vec!["hello", " world"]);
        let last = events.last().unwrap();
        match last {
            StreamEvent::MessageStop { usage, stop_reason } => {
                assert_eq!(usage.input_tokens, 3);
                assert_eq!(usage.output_tokens, 5);
                assert_eq!(*stop_reason, StopReason::EndTurn);
            }
            other => panic!("expected MessageStop, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_interruption_is_typed_and_retryable() {
        // SC-3: a connection dropped mid-stream surfaces as the typed, retryable
        // `StreamInterrupted` variant — a consumer drives its retry off
        // `retryable()`, not a substring match on the error text. A raw TCP
        // server promises a 200-byte body (Content-Length) but sends a few bytes
        // then closes the socket, so the client's body stream errors mid-read.
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 2048];
                let _ = sock.read(&mut buf); // drain the request headers
                // 200 OK so `call_streaming` returns Ok (headers arrived), then
                // promise 200 body bytes but deliver only a partial SSE line and
                // drop the socket — EOF before Content-Length errors the stream.
                let _ = sock.write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: 200\r\n\r\ndata: partial",
                );
                let _ = sock.flush();
                // `sock` dropped here -> connection closes mid-body.
            }
        });

        let client = OpenAIModelInterface::new("k", "gpt-4o").with_base_url(format!("http://{addr}"));
        let mut stream = client
            .call_streaming(req(vec![user("hi")]))
            .await
            .expect("headers (200) arrive before the body is truncated");

        let mut last_err = None;
        while let Some(ev) = stream.next().await {
            if let Err(e) = ev {
                last_err = Some(e);
                break;
            }
        }
        let err = last_err.expect("the truncated body must error the stream");
        assert!(
            matches!(err, ModelError::StreamInterrupted { .. }),
            "expected StreamInterrupted, got {err:?}",
        );
        assert!(err.retryable(), "a mid-stream interruption is retryable");
        server.join().unwrap();
    }

    #[tokio::test]
    async fn streaming_accumulates_tool_call_deltas() {
        // Three partial chunks for the same tool call (index=0): the first
        // carries id+name; subsequent chunks carry incremental arguments
        // fragments. Consumer joins ToolUseDelta.partial_json strings to
        // reconstruct the full JSON.
        let sse = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"fetch\",\"arguments\":\"{\\\"u\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"rl\\\":\\\"x\\\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n",
        );
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/chat/completions"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_string(sse)
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&server)
            .await;
        let client = OpenAIModelInterface::new("k", "gpt-4o").with_base_url(server.uri());
        let mut stream = client.call_streaming(req(vec![user("hi")])).await.unwrap();
        let mut tool_fragments: Vec<String> = vec![];
        let mut final_stop = StopReason::EndTurn;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::ToolUseDelta { partial_json, .. } => {
                    tool_fragments.push(partial_json);
                }
                StreamEvent::MessageStop { stop_reason, .. } => {
                    final_stop = stop_reason;
                }
                _ => {}
            }
        }
        let joined: String = tool_fragments.concat();
        let parsed: serde_json::Value = serde_json::from_str(&joined).unwrap();
        assert_eq!(parsed, serde_json::json!({"url": "x"}));
        assert_eq!(final_stop, StopReason::ToolUse);
    }

    // ── #[ignore]-tagged live API tests ────────────────────────────────────
    //
    // Run with: OPENAI_API_KEY=... cargo test -p spore-core openai_live -- --ignored

    #[tokio::test]
    #[ignore = "live-API; needs OPENAI_API_KEY"]
    async fn openai_live_call_returns_response() {
        let client = OpenAIModelInterface::from_env("OPENAI_API_KEY", "gpt-4o-mini")
            .expect("OPENAI_API_KEY set");
        let r = client
            .call(req(vec![user("Reply with the word 'pong'.")]))
            .await
            .unwrap();
        assert!(r.usage.input_tokens > 0);
        assert!(r.usage.output_tokens > 0);
    }

    #[tokio::test]
    #[ignore = "live-API; needs OPENAI_API_KEY"]
    async fn openai_live_streaming_emits_events() {
        let client = OpenAIModelInterface::from_env("OPENAI_API_KEY", "gpt-4o-mini")
            .expect("OPENAI_API_KEY set");
        let mut s = client
            .call_streaming(req(vec![user("Reply with the word 'pong'.")]))
            .await
            .unwrap();
        let mut saw_stop = false;
        while let Some(ev) = s.next().await {
            if matches!(ev.unwrap(), StreamEvent::MessageStop { .. }) {
                saw_stop = true;
            }
        }
        assert!(saw_stop, "stream did not produce MessageStop");
    }

    #[tokio::test]
    #[ignore = "live-API; needs OPENAI_API_KEY"]
    async fn openai_live_reasoning_returns_thinking() {
        let client = OpenAIModelInterface::from_env("OPENAI_API_KEY", "o3-mini")
            .expect("OPENAI_API_KEY set");
        let r = client
            .call(req(vec![user("What is 17 * 23? Think step by step.")]))
            .await
            .unwrap();
        assert!(r.usage.output_tokens > 0);
    }
}
