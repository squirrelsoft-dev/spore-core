//! Issue #39 — `AnthropicModelInterface`: real Anthropic Messages API client.
//!
//! Implements [`ModelInterface`] against `https://api.anthropic.com/v1/messages`.
//! Translates [`ModelRequest`] / [`ModelResponse`] to and from Anthropic's wire
//! format, parses the SSE event stream for `call_streaming`, hits
//! `/v1/messages/count_tokens` for accurate token counts, and maps HTTP errors
//! to typed [`ModelError`] variants with retry/backoff for transient failures.
//!
//! ## What lives here
//! - [`AnthropicModelInterface`] — the client struct + `ModelInterface` impl
//! - [`AnthropicError`] — implementation-private error type (collapsed into
//!   `ModelError` at the public boundary)
//! - Wire-format types (`AnthropicRequest`, `AnthropicResponse`, etc.)
//!
//! ## What's deferred (carry-over from #38, see #39 issue comments)
//! - Re-recording `fixtures/model_responses/model_interface/basic_text.jsonl`
//!   against the real API — needs `ANTHROPIC_API_KEY` and live network
//! - Live-API integration tests run as `#[ignore]`; CI does not invoke them
//!
//! ## Cache-cost wiring
//! Per-model cache pricing (USD per million tokens) lives in
//! [`cache_provider::AnthropicCacheProvider::with_model_pricing`].

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::model::{
    Content, ContentBlock, ModelError, ModelInterface, ModelRequest, ModelResponse, ModelStream,
    ProviderInfo, Role, StopReason, StreamEvent, TokenUsage, ToolCall, ToolSchema,
};

// ============================================================================
// AnthropicModelInterface
// ============================================================================

/// Reference Anthropic client. Constructed with an API key and a model id;
/// callers can override base URL (for proxying or mocking) and tune retry
/// behavior.
///
/// `http_client` is `Arc`'d so the same client (with its connection pool)
/// can back many `AnthropicModelInterface` instances. Tests inject a
/// pre-configured client pointing at `wiremock`.
pub struct AnthropicModelInterface {
    api_key: String,
    model_id: String,
    base_url: String,
    timeout: Duration,
    max_retries: u32,
    http_client: Arc<reqwest::Client>,
}

impl std::fmt::Debug for AnthropicModelInterface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Redact api_key — it must never appear in logs or traces.
        f.debug_struct("AnthropicModelInterface")
            .field("api_key", &"<redacted>")
            .field("model_id", &self.model_id)
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl AnthropicModelInterface {
    /// Default base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.anthropic.com";

    /// Default request timeout.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

    /// Default retry count for transient 429/529/Timeout responses.
    pub const DEFAULT_MAX_RETRIES: u32 = 3;

    /// API version pin. Anthropic requires this header on every request.
    pub const ANTHROPIC_VERSION: &'static str = "2023-06-01";

    pub fn new(api_key: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model_id: model_id.into(),
            base_url: Self::DEFAULT_BASE_URL.into(),
            timeout: Self::DEFAULT_TIMEOUT,
            max_retries: Self::DEFAULT_MAX_RETRIES,
            http_client: Arc::new(reqwest::Client::new()),
        }
    }

    /// Read API key from environment variable. Returns
    /// `ModelError::ProviderError` if the variable is unset or empty.
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

    pub fn with_http_client(mut self, c: Arc<reqwest::Client>) -> Self {
        self.http_client = c;
        self
    }

    /// Context window for known model ids. Falls back to 200k for any
    /// `claude-*` id (current family default) and to 0 otherwise so that
    /// callers can detect "unknown model" rather than silently getting a
    /// plausible-but-wrong value.
    pub fn context_window(model_id: &str) -> u32 {
        match model_id {
            "claude-sonnet-4-5"
            | "claude-sonnet-4-6"
            | "claude-opus-4-5"
            | "claude-opus-4-6"
            | "claude-opus-4-7"
            | "claude-haiku-4-5"
            | "claude-haiku-4-5-20251001" => 200_000,
            id if id.starts_with("claude-") => 200_000,
            _ => 0,
        }
    }
}

// ============================================================================
// Wire-format types (Anthropic Messages API)
// ============================================================================

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
    Image {
        source: AnthropicImageSource,
    },
}

#[derive(Debug, Serialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    kind: &'static str, // always "base64"
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    content: Vec<AnthropicResponseBlock>,
    stop_reason: Option<String>,
    #[serde(default)]
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicResponseBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Default, Deserialize, Clone, Copy)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct AnthropicCountTokensResponse {
    input_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorBody {
    #[serde(default)]
    error: Option<AnthropicErrorInner>,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorInner {
    #[serde(default)]
    message: Option<String>,
}

// ============================================================================
// Conversions (ModelRequest -> Anthropic wire, Anthropic wire -> ModelResponse)
// ============================================================================

/// Translate a Spore `ModelRequest` to the Anthropic Messages API body.
/// System-role messages are extracted into the top-level `system` field; all
/// other messages keep their role and have their `Content` wrapped in a
/// single-element content-block array.
fn build_request(model_id: &str, req: &ModelRequest, stream: bool) -> AnthropicRequest {
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<AnthropicMessage> = Vec::new();
    for m in &req.messages {
        if m.role == Role::System {
            if let Content::Text { text } = &m.content {
                system_parts.push(text.clone());
            }
            continue;
        }
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "user", // Anthropic carries tool_result in user-role messages
            Role::System => unreachable!(),
        };
        let content = vec![content_to_anthropic(&m.content)];
        messages.push(AnthropicMessage {
            role: role.into(),
            content,
        });
    }
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    let tools: Vec<AnthropicTool> = req
        .tools
        .iter()
        .map(|t: &ToolSchema| AnthropicTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect();
    AnthropicRequest {
        model: model_id.into(),
        // Anthropic requires max_tokens; pick a sane default if unset.
        max_tokens: req.params.max_tokens.unwrap_or(4096),
        messages,
        system,
        temperature: req.params.temperature,
        top_p: req.params.top_p,
        stop_sequences: req.params.stop_sequences.clone(),
        tools,
        stream,
    }
}

fn content_to_anthropic(c: &Content) -> AnthropicContent {
    match c {
        Content::Text { text } => AnthropicContent::Text { text: text.clone() },
        Content::ToolCall(call) => AnthropicContent::ToolUse {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.input.clone(),
        },
        Content::ToolResult(r) => AnthropicContent::ToolResult {
            tool_use_id: r.tool_use_id.clone(),
            content: r.content.clone(),
            is_error: r.is_error,
        },
        Content::Image { media_type, data } => AnthropicContent::Image {
            source: AnthropicImageSource {
                kind: "base64",
                media_type: media_type.clone(),
                data: data.clone(),
            },
        },
    }
}

fn parse_response(body: AnthropicResponse) -> ModelResponse {
    let content: Vec<ContentBlock> = body
        .content
        .into_iter()
        .map(|b| match b {
            AnthropicResponseBlock::Text { text } => ContentBlock::Text { text },
            AnthropicResponseBlock::Thinking { thinking } => {
                ContentBlock::Thinking { text: thinking }
            }
            AnthropicResponseBlock::ToolUse { id, name, input } => {
                ContentBlock::ToolUse(ToolCall { id, name, input })
            }
        })
        .collect();
    ModelResponse {
        content,
        usage: TokenUsage {
            input_tokens: body.usage.input_tokens,
            output_tokens: body.usage.output_tokens,
            cache_read_tokens: body.usage.cache_read_input_tokens,
            cache_write_tokens: body.usage.cache_creation_input_tokens,
        },
        stop_reason: parse_stop_reason(body.stop_reason.as_deref()),
    }
}

fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        _ => StopReason::EndTurn,
    }
}

// ============================================================================
// HTTP plumbing with retry
// ============================================================================

/// Quietly suppress two clippy nits relevant inside the loop.
#[allow(clippy::too_many_lines)]
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
                // Decide whether to retry based on status code.
                let retryable = matches!(code, 408 | 425 | 429 | 500 | 502 | 503 | 504 | 529);
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
                return Err(ModelError::ProviderError {
                    code: 0,
                    message: format!("HTTP transport error: {e}"),
                })
            }
        }
    }
}

/// Exponential backoff with a small jitter ceiling. Pure function so it's
/// easy to test.
fn backoff_delay(attempt: u32) -> Duration {
    // 0.5s, 1s, 2s, 4s, ... up to 30s.
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
    let message = serde_json::from_str::<AnthropicErrorBody>(&body_text)
        .ok()
        .and_then(|b| b.error.and_then(|e| e.message))
        .unwrap_or_else(|| body_text.chars().take(500).collect());
    match code {
        429 => ModelError::RateLimited { retry_after },
        529 => ModelError::RateLimited { retry_after: None },
        408 | 504 => ModelError::Timeout,
        _ => ModelError::ProviderError { code, message },
    }
}

// ============================================================================
// ModelInterface impl
// ============================================================================

impl ModelInterface for AnthropicModelInterface {
    async fn call(&self, request: ModelRequest) -> Result<ModelResponse, ModelError> {
        let body = build_request(&self.model_id, &request, false);
        let url = format!("{}/v1/messages", self.base_url);
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
                    .header("x-api-key", &api_key)
                    .header("anthropic-version", Self::ANTHROPIC_VERSION)
                    .header("content-type", "application/json")
                    .body(body.clone())
            },
            self.max_retries,
            self.timeout,
        )
        .await?;
        let parsed: AnthropicResponse =
            resp.json().await.map_err(|e| ModelError::ProviderError {
                code: 0,
                message: format!("response decode failed: {e}"),
            })?;
        Ok(parse_response(parsed))
    }

    async fn call_streaming(&self, request: ModelRequest) -> Result<ModelStream, ModelError> {
        let body = build_request(&self.model_id, &request, true);
        let url = format!("{}/v1/messages", self.base_url);
        let body = serde_json::to_string(&body).map_err(|e| ModelError::ProviderError {
            code: 0,
            message: format!("request encode failed: {e}"),
        })?;
        let client = self.http_client.clone();
        let api_key = self.api_key.clone();
        // Route through `send_with_retry` so opening a stream gets the same
        // retry/backoff on transient failures (timeouts, 429/5xx/529) as the
        // non-streaming paths. Retries happen BEFORE the body is consumed; the
        // closure rebuilds the request each attempt (`body.clone()`).
        let resp = send_with_retry(
            || {
                client
                    .post(&url)
                    .header("x-api-key", &api_key)
                    .header("anthropic-version", Self::ANTHROPIC_VERSION)
                    .header("content-type", "application/json")
                    .header("accept", "text/event-stream")
                    .body(body.clone())
            },
            self.max_retries,
            self.timeout,
        )
        .await?;
        Ok(Box::pin(sse_to_events(resp)))
    }

    async fn count_tokens(&self, request: &ModelRequest) -> Result<u32, ModelError> {
        let body = build_request(&self.model_id, request, false);
        let url = format!("{}/v1/messages/count_tokens", self.base_url);
        let body = serde_json::to_string(&body).map_err(|e| ModelError::ProviderError {
            code: 0,
            message: format!("count_tokens encode failed: {e}"),
        })?;
        let api_key = self.api_key.clone();
        let client = self.http_client.clone();
        let resp = send_with_retry(
            || {
                client
                    .post(&url)
                    .header("x-api-key", &api_key)
                    .header("anthropic-version", Self::ANTHROPIC_VERSION)
                    .header("content-type", "application/json")
                    .body(body.clone())
            },
            self.max_retries,
            self.timeout,
        )
        .await?;
        let parsed: AnthropicCountTokensResponse =
            resp.json().await.map_err(|e| ModelError::ProviderError {
                code: 0,
                message: format!("count_tokens decode failed: {e}"),
            })?;
        Ok(parsed.input_tokens)
    }

    fn provider(&self) -> ProviderInfo {
        ProviderInfo {
            name: "anthropic".into(),
            model_id: self.model_id.clone(),
            context_window: Self::context_window(&self.model_id),
        }
    }
}

// ============================================================================
// SSE stream parsing
// ============================================================================

/// Convert an Anthropic SSE response into a stream of [`StreamEvent`]s.
///
/// We translate the subset of event types the harness consumes:
/// - `message_start` → no emit (header-only)
/// - `content_block_start` (tool_use) → `ToolUseStart` carrying id + name
/// - `content_block_delta.text_delta` → `ContentBlockDelta`
/// - `content_block_delta.thinking_delta` → `ThinkingDelta`
/// - `content_block_delta.input_json_delta` → `ToolUseDelta`
/// - `content_block_stop` → `ContentBlockStop`
/// - `message_delta.usage` → captured into the final `MessageStop`
/// - `message_stop` → `MessageStop` with accumulated usage & stop_reason
fn sse_to_events(
    resp: reqwest::Response,
) -> impl futures_core::Stream<Item = Result<StreamEvent, ModelError>> + Send + 'static {
    async_stream::stream! {
        let stream = resp.bytes_stream();
        futures_util::pin_mut!(stream);
        let mut buf = crate::model::ByteLineBuffer::new();
        let mut usage = TokenUsage::default();
        let mut stop_reason = StopReason::EndTurn;
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Err(ModelError::ProviderError {
                        code: 0,
                        message: format!("stream chunk error: {e}"),
                    });
                    return;
                }
            };
            buf.push(&chunk);
            // SSE events are separated by blank lines (\n\n). Process complete
            // events; leave any partial trailing event in the buffer.
            while let Some(raw) = buf.next_line(b"\n\n") {
                let event = parse_sse_event(&raw);
                if let Some((event_name, data)) = event {
                    // Skip non-JSON keepalive lines and unrecognized event types.
                    if data.is_empty() || data == "{}" {
                        continue;
                    }
                    let value: serde_json::Value = match serde_json::from_str(&data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    match event_name.as_str() {
                        "content_block_start" => {
                            // A tool_use block opens here with its id + name; emit
                            // ToolUseStart so the accumulator captures them before
                            // the input_json_delta arg fragments arrive.
                            let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            let block = value.get("content_block");
                            let is_tool = block
                                .and_then(|b| b.get("type"))
                                .and_then(|v| v.as_str())
                                == Some("tool_use");
                            if is_tool {
                                let id = block.and_then(|b| b.get("id")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                                let name = block.and_then(|b| b.get("name")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                                yield Ok(StreamEvent::ToolUseStart { index, id, name });
                            }
                        }
                        "content_block_delta" => {
                            let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            let delta = value.get("delta");
                            let kind = delta.and_then(|d| d.get("type")).and_then(|v| v.as_str()).unwrap_or("");
                            match kind {
                                "text_delta" => {
                                    let text = delta.and_then(|d| d.get("text")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    yield Ok(StreamEvent::ContentBlockDelta { index, delta: text });
                                }
                                "thinking_delta" => {
                                    let text = delta.and_then(|d| d.get("thinking")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    yield Ok(StreamEvent::ThinkingDelta { index, delta: text });
                                }
                                "input_json_delta" => {
                                    let partial = delta.and_then(|d| d.get("partial_json")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    yield Ok(StreamEvent::ToolUseDelta { index, partial_json: partial });
                                }
                                _ => {}
                            }
                        }
                        "content_block_stop" => {
                            let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            yield Ok(StreamEvent::ContentBlockStop { index });
                        }
                        "message_start" => {
                            if let Some(msg) = value.get("message") {
                                if let Some(u) = msg.get("usage") {
                                    if let Some(it) = u.get("input_tokens").and_then(|v| v.as_u64()) {
                                        usage.input_tokens = it as u32;
                                    }
                                }
                            }
                            yield Ok(StreamEvent::MessageStart);
                        }
                        "message_delta" => {
                            if let Some(d) = value.get("delta") {
                                if let Some(s) = d.get("stop_reason").and_then(|v| v.as_str()) {
                                    stop_reason = parse_stop_reason(Some(s));
                                }
                            }
                            if let Some(u) = value.get("usage") {
                                if let Some(ot) = u.get("output_tokens").and_then(|v| v.as_u64()) {
                                    usage.output_tokens = ot as u32;
                                }
                            }
                        }
                        "message_stop" => {
                            yield Ok(StreamEvent::MessageStop { usage, stop_reason });
                            return;
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

/// Parse one SSE event block (`event: name\ndata: {...}`). Returns
/// `(event_name, data_payload)` or `None` if the block doesn't follow the
/// expected shape.
fn parse_sse_event(raw: &str) -> Option<(String, String)> {
    let mut event_name: Option<String> = None;
    let mut data: Vec<String> = Vec::new();
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data.push(rest.trim_start().to_string());
        }
    }
    let name = event_name?;
    Some((name, data.join("\n")))
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
    fn build_request_extracts_system_message() {
        let r = req(vec![sys("be helpful"), user("hi")]);
        let body = build_request("claude-sonnet-4-6", &r, false);
        assert_eq!(body.system.as_deref(), Some("be helpful"));
        assert_eq!(body.messages.len(), 1);
        assert_eq!(body.messages[0].role, "user");
    }

    #[test]
    fn build_request_joins_multiple_system_messages() {
        let r = req(vec![sys("first"), sys("second"), user("hi")]);
        let body = build_request("claude-sonnet-4-6", &r, false);
        assert_eq!(body.system.as_deref(), Some("first\n\nsecond"));
    }

    #[test]
    fn build_request_defaults_max_tokens_when_unset() {
        let r = req(vec![user("hi")]);
        let body = build_request("claude-sonnet-4-6", &r, false);
        assert_eq!(body.max_tokens, 4096);
    }

    #[test]
    fn build_request_respects_max_tokens() {
        let mut r = req(vec![user("hi")]);
        r.params.max_tokens = Some(256);
        let body = build_request("claude-sonnet-4-6", &r, false);
        assert_eq!(body.max_tokens, 256);
    }

    #[test]
    fn build_request_maps_tool_call_message() {
        let r = req(vec![Message {
            role: Role::Assistant,
            content: Content::ToolCall(ToolCall {
                id: "call-1".into(),
                name: "fetch".into(),
                input: serde_json::json!({"url": "x"}),
            }),
        }]);
        let body = build_request("claude-sonnet-4-6", &r, false);
        let s = serde_json::to_string(&body).unwrap();
        assert!(s.contains("\"type\":\"tool_use\""), "wire: {s}");
        assert!(s.contains("\"id\":\"call-1\""));
    }

    #[test]
    fn build_request_maps_tool_result_to_user_role() {
        let r = req(vec![Message {
            role: Role::Tool,
            content: Content::ToolResult(ToolResult {
                tool_use_id: "call-1".into(),
                content: "ok".into(),
                is_error: false,
            }),
        }]);
        let body = build_request("claude-sonnet-4-6", &r, false);
        assert_eq!(body.messages[0].role, "user");
        let s = serde_json::to_string(&body.messages[0].content).unwrap();
        assert!(s.contains("\"type\":\"tool_result\""), "wire: {s}");
    }

    // ── parse_response ──────────────────────────────────────────────────────

    #[test]
    fn parse_response_extracts_text_and_usage() {
        let body: AnthropicResponse = serde_json::from_value(serde_json::json!({
            "id": "msg_x",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 2}
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
    fn parse_response_extracts_tool_use() {
        let body: AnthropicResponse = serde_json::from_value(serde_json::json!({
            "content": [{"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "rust"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .unwrap();
        let r = parse_response(body);
        match &r.content[0] {
            ContentBlock::ToolUse(tc) => {
                assert_eq!(tc.id, "c1");
                assert_eq!(tc.name, "search");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        assert_eq!(r.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn parse_response_extracts_thinking_block() {
        let body: AnthropicResponse = serde_json::from_value(serde_json::json!({
            "content": [{"type": "thinking", "thinking": "let me reason..."}, {"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })).unwrap();
        let r = parse_response(body);
        assert!(matches!(r.content[0], ContentBlock::Thinking { .. }));
        assert!(matches!(r.content[1], ContentBlock::Text { .. }));
    }

    #[test]
    fn parse_response_extracts_cache_usage() {
        let body: AnthropicResponse = serde_json::from_value(serde_json::json!({
            "content": [{"type": "text", "text": "x"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 2,
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 30
            }
        }))
        .unwrap();
        let r = parse_response(body);
        assert_eq!(r.usage.cache_read_tokens, Some(50));
        assert_eq!(r.usage.cache_write_tokens, Some(30));
    }

    // ── parse_stop_reason ───────────────────────────────────────────────────

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(parse_stop_reason(Some("end_turn")), StopReason::EndTurn);
        assert_eq!(parse_stop_reason(Some("tool_use")), StopReason::ToolUse);
        assert_eq!(parse_stop_reason(Some("max_tokens")), StopReason::MaxTokens);
        assert_eq!(
            parse_stop_reason(Some("stop_sequence")),
            StopReason::StopSequence
        );
        // Unknown / null → EndTurn (safe default).
        assert_eq!(parse_stop_reason(None), StopReason::EndTurn);
        assert_eq!(parse_stop_reason(Some("???")), StopReason::EndTurn);
    }

    // ── backoff ─────────────────────────────────────────────────────────────

    #[test]
    fn backoff_grows_then_caps() {
        let d0 = backoff_delay(0);
        let d3 = backoff_delay(3);
        let dmax = backoff_delay(20);
        assert!(d3 > d0);
        assert!(dmax <= Duration::from_millis(30_000));
    }

    // ── context_window ──────────────────────────────────────────────────────

    #[test]
    fn context_window_known_and_unknown() {
        assert_eq!(
            AnthropicModelInterface::context_window("claude-sonnet-4-6"),
            200_000
        );
        assert_eq!(
            AnthropicModelInterface::context_window("claude-opus-4-7"),
            200_000
        );
        // Unknown claude- prefixed model still defaults to 200k.
        assert_eq!(
            AnthropicModelInterface::context_window("claude-imaginary-9"),
            200_000
        );
        // Completely foreign model id returns 0 so callers detect it.
        assert_eq!(AnthropicModelInterface::context_window("gpt-4o"), 0);
    }

    // ── provider() ──────────────────────────────────────────────────────────

    #[test]
    fn provider_info_uses_model_id() {
        let c = AnthropicModelInterface::new("test-key", "claude-sonnet-4-6");
        let p = c.provider();
        assert_eq!(p.name, "anthropic");
        assert_eq!(p.model_id, "claude-sonnet-4-6");
        assert_eq!(p.context_window, 200_000);
    }

    // ── from_env ────────────────────────────────────────────────────────────

    #[test]
    fn from_env_errors_when_unset() {
        std::env::remove_var("__SPORE_TEST_ANTHROPIC_KEY_UNSET__");
        let err = AnthropicModelInterface::from_env(
            "__SPORE_TEST_ANTHROPIC_KEY_UNSET__",
            "claude-sonnet-4-6",
        )
        .unwrap_err();
        match err {
            ModelError::ProviderError { message, .. } => assert!(message.contains("not set")),
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    // ── SSE parser ──────────────────────────────────────────────────────────

    #[test]
    fn parse_sse_event_basic() {
        let raw = "event: message_start\ndata: {\"type\":\"message_start\"}";
        let (name, data) = parse_sse_event(raw).unwrap();
        assert_eq!(name, "message_start");
        assert_eq!(data, "{\"type\":\"message_start\"}");
    }

    #[test]
    fn parse_sse_event_multiline_data() {
        let raw = "event: message_delta\ndata: {\"first\":1}\ndata: continuation";
        let (name, data) = parse_sse_event(raw).unwrap();
        assert_eq!(name, "message_delta");
        assert_eq!(data, "{\"first\":1}\ncontinuation");
    }

    // ── End-to-end wiremock call() ──────────────────────────────────────────

    #[tokio::test]
    async fn call_against_mock_returns_response() {
        let server = wiremock::MockServer::start().await;
        let body = serde_json::json!({
            "id": "msg_x",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2}
        });
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .and(wiremock::matchers::header("x-api-key", "test-key"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        let client = AnthropicModelInterface::new("test-key", "claude-sonnet-4-6")
            .with_base_url(server.uri());
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
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(wiremock::ResponseTemplate::new(429).insert_header("retry-after", "7"))
            .mount(&server)
            .await;
        let client = AnthropicModelInterface::new("k", "claude-sonnet-4-6")
            .with_base_url(server.uri())
            .with_max_retries(0); // disable retries so error surfaces
        let err = client.call(req(vec![user("hi")])).await.unwrap_err();
        match err {
            ModelError::RateLimited { retry_after } => {
                assert_eq!(retry_after, Some(Duration::from_secs(7)));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn call_maps_529_to_rate_limited_no_retry_after() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(wiremock::ResponseTemplate::new(529))
            .mount(&server)
            .await;
        let client = AnthropicModelInterface::new("k", "claude-sonnet-4-6")
            .with_base_url(server.uri())
            .with_max_retries(0);
        let err = client.call(req(vec![user("hi")])).await.unwrap_err();
        assert!(matches!(err, ModelError::RateLimited { retry_after: None }));
    }

    #[tokio::test]
    async fn call_maps_400_to_provider_error_with_anthropic_message() {
        let server = wiremock::MockServer::start().await;
        let body = serde_json::json!({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "max_tokens must be > 0"}
        });
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(wiremock::ResponseTemplate::new(400).set_body_json(body))
            .mount(&server)
            .await;
        let client = AnthropicModelInterface::new("k", "claude-sonnet-4-6")
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
            "id": "msg_x",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "after retry"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(wiremock::ResponseTemplate::new(429).insert_header("retry-after", "0"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&server)
            .await;
        let client =
            AnthropicModelInterface::new("k", "claude-sonnet-4-6").with_base_url(server.uri());
        let r = client.call(req(vec![user("hi")])).await.unwrap();
        assert_eq!(
            r.content[0],
            ContentBlock::Text {
                text: "after retry".into()
            }
        );
    }

    #[tokio::test]
    async fn count_tokens_uses_real_endpoint() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages/count_tokens"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "input_tokens": 42
                })),
            )
            .mount(&server)
            .await;
        let client =
            AnthropicModelInterface::new("k", "claude-sonnet-4-6").with_base_url(server.uri());
        let n = client.count_tokens(&req(vec![user("hi")])).await.unwrap();
        assert_eq!(n, 42);
    }

    // ── End-to-end wiremock call_streaming() ────────────────────────────────

    #[tokio::test]
    async fn streaming_emits_text_delta_then_stop() {
        let sse = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n",
            "\n",
            "event: content_block_stop\n",
            "data: {\"index\":0}\n",
            "\n",
            "event: message_delta\n",
            "data: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n",
            "\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n",
            "\n",
        );
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_string(sse)
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&server)
            .await;
        let client =
            AnthropicModelInterface::new("k", "claude-sonnet-4-6").with_base_url(server.uri());
        let mut stream = client.call_streaming(req(vec![user("hi")])).await.unwrap();
        let mut events: Vec<StreamEvent> = vec![];
        while let Some(ev) = stream.next().await {
            events.push(ev.unwrap());
        }
        let kinds: Vec<&'static str> = events
            .iter()
            .map(|e| match e {
                StreamEvent::MessageStart => "start",
                StreamEvent::ContentBlockDelta { .. } => "text",
                StreamEvent::ThinkingDelta { .. } => "think",
                StreamEvent::ToolUseStart { .. } => "tool_start",
                StreamEvent::ToolUseDelta { .. } => "tool",
                StreamEvent::ContentBlockStop { .. } => "stop_block",
                StreamEvent::MessageStop { .. } => "stop",
            })
            .collect();
        assert_eq!(kinds, vec!["start", "text", "text", "stop_block", "stop"]);
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

    // ── #[ignore]-tagged live API tests ─────────────────────────────────────
    //
    // Run with: ANTHROPIC_API_KEY=... cargo test -p spore-core anthropic_live -- --ignored

    #[tokio::test]
    #[ignore = "live-API; needs ANTHROPIC_API_KEY"]
    async fn anthropic_live_call_returns_response() {
        let client = AnthropicModelInterface::from_env("ANTHROPIC_API_KEY", "claude-sonnet-4-6")
            .expect("ANTHROPIC_API_KEY set");
        let r = client
            .call(req(vec![user("Reply with the word 'pong'.")]))
            .await
            .unwrap();
        assert!(r.usage.input_tokens > 0);
        assert!(r.usage.output_tokens > 0);
    }

    #[tokio::test]
    #[ignore = "live-API; needs ANTHROPIC_API_KEY"]
    async fn anthropic_live_count_tokens_is_nonzero() {
        let client = AnthropicModelInterface::from_env("ANTHROPIC_API_KEY", "claude-sonnet-4-6")
            .expect("ANTHROPIC_API_KEY set");
        let n = client
            .count_tokens(&req(vec![user("count my tokens please")]))
            .await
            .unwrap();
        assert!(n > 0);
    }

    #[tokio::test]
    #[ignore = "live-API; needs ANTHROPIC_API_KEY"]
    async fn anthropic_live_streaming_emits_events() {
        let client = AnthropicModelInterface::from_env("ANTHROPIC_API_KEY", "claude-sonnet-4-6")
            .expect("ANTHROPIC_API_KEY set");
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
}
