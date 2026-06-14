//! Issue #41 — `OllamaModelInterface`: real Ollama HTTP client.
//!
//! Implements [`ModelInterface`] against a local Ollama server's `/api/chat`,
//! `/api/tags`, and `/api/embed` endpoints. Translates [`ModelRequest`] /
//! [`ModelResponse`] to and from the Ollama wire format, parses Ollama's
//! NDJSON stream (one JSON object per line — not SSE) for `call_streaming`,
//! and maps HTTP/transport errors to typed [`ModelError`] variants. Unlike
//! the Anthropic and OpenAI clients there is **no retry**: spec says fail
//! fast on connection errors with a helpful message ("ollama not running",
//! "run ollama pull <model>").
//!
//! ## Types
//! - [`OllamaModelInterface`] — concrete `ModelInterface`
//! - private wire-format structs: `OllamaRequest`, `OllamaMessage`,
//!   `OllamaResponse`, `OllamaResponseMessage`, `OllamaTool`, `OllamaToolCall`,
//!   `OllamaOptions`, `TagsResponse`, `ShowRequest`, `ShowResponse`,
//!   `ModelMeta`, `EmbedRequest`, `EmbedResponse`
//!
//! ## Trait methods
//! - `call(request)`           — POST `/api/chat` with `stream: false`
//! - `call_streaming(request)` — POST `/api/chat` with `stream: true`, parse NDJSON
//! - `count_tokens(request)`   — POST `/api/embed`, fallback to bytes/4 heuristic
//! - `provider()`              — `ProviderInfo { name: "ollama", ... }`
//!
//! ## Provider-specific shape
//! - No API key; default `base_url` is `http://localhost:11434`.
//! - Sampling parameters (`num_predict`, `temperature`, `top_p`, `stop`) are
//!   nested under `options` rather than being top-level keys.
//! - `keep_alive` (default `"5m"`) controls how long Ollama keeps the model
//!   loaded after the call returns.
//! - Tool-call arguments are a JSON **object** in the wire format, not a
//!   JSON-encoded string like OpenAI.
//! - Ollama does not return tool-call ids; we synthesize `call-{i}` per index
//!   so downstream `ToolResult.tool_use_id` round-trips work.
//! - Thinking blocks are silently omitted from outgoing requests — Ollama
//!   has no structured reasoning shape.
//!
//! ## Rules enforced here
//! 1. `TokenUsage` reported on every successful `call` and final `MessageStop`
//!    (`prompt_eval_count` → input, `eval_count` → output, cache fields `None`).
//! 2. Connection refused → `ProviderError { code: 0, message: "Ollama not
//!    running at {base_url}" }` (no retry).
//! 3. 404 from `/api/chat` or model missing from `/api/tags` →
//!    `ProviderError { code: 404, message: "Model {id} not found. Run:
//!    ollama pull {id}" }`.
//! 4. Timeout → `ModelError::Timeout`.
//! 5. Other 4xx/5xx → `ProviderError { code, message }`.
//! 6. Lazy model availability check on first call; cached for instance lifetime.
//! 7. On first call, a one-time `POST /api/show` discovery runs alongside the
//!    `/api/tags` availability check: it reads the model's `*.context_length`
//!    from `model_info` and the top-level `capabilities` array. `provider()`
//!    returns the discovered context window when available (falling back to the
//!    static [`OllamaModelInterface::context_window`] table), and `call` /
//!    `call_streaming` reject tool-bearing requests with a
//!    `does not support tool calling` `ProviderError` when the model lacks the
//!    `"tools"` capability. The `/api/show` `capabilities` array is the sole
//!    authority for tool support: empty or unavailable metadata ⟹ NOT
//!    tool-capable (fail closed). `/api/show` failures otherwise degrade
//!    gracefully — they never fail the call.

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::model::{
    Content, ContentBlock, ModelError, ModelInterface, ModelRequest, ModelResponse, ModelStream,
    ProviderInfo, Role, StopReason, StreamEvent, TokenUsage, ToolCall, ToolSchema,
};

// ============================================================================
// OllamaModelInterface
// ============================================================================

pub struct OllamaModelInterface {
    model_id: String,
    base_url: String,
    timeout: Duration,
    keep_alive: Option<String>,
    http_client: Arc<reqwest::Client>,
    /// Lazily set after the first availability + discovery probe. Holds the
    /// `/api/show`-discovered metadata (context length + capabilities); empty
    /// `ModelMeta` when discovery failed but availability succeeded.
    model_checked: Arc<OnceCell<ModelMeta>>,
}

/// `/api/show`-discovered metadata for the model. Populated once, alongside the
/// `/api/tags` availability check. All fields are best-effort — `/api/show`
/// failures leave them unset rather than failing the call.
#[derive(Debug, Default, Clone)]
struct ModelMeta {
    /// Discovered context window (`*.context_length` in `model_info`).
    context_length: Option<u32>,
    /// Top-level `capabilities` array (may contain `"tools"`).
    capabilities: Vec<String>,
}

impl ModelMeta {
    fn supports_tools(&self) -> bool {
        self.capabilities.iter().any(|c| c == "tools")
    }
}

impl std::fmt::Debug for OllamaModelInterface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaModelInterface")
            .field("model_id", &self.model_id)
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("keep_alive", &self.keep_alive)
            .finish()
    }
}

impl OllamaModelInterface {
    pub const DEFAULT_BASE_URL: &'static str = "http://localhost:11434";
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
    pub const DEFAULT_KEEP_ALIVE: &'static str = "5m";

    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            base_url: Self::DEFAULT_BASE_URL.into(),
            timeout: Self::DEFAULT_TIMEOUT,
            keep_alive: Some(Self::DEFAULT_KEEP_ALIVE.into()),
            http_client: Arc::new(reqwest::Client::new()),
            model_checked: Arc::new(OnceCell::new()),
        }
    }

    pub fn with_base_url(model_id: impl Into<String>, base_url: impl Into<String>) -> Self {
        let mut s = Self::new(model_id);
        s.base_url = base_url.into();
        s
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    pub fn with_http_client(mut self, c: Arc<reqwest::Client>) -> Self {
        self.http_client = c;
        self
    }

    /// Context window for known Ollama model id prefixes.
    pub fn context_window(model_id: &str) -> u32 {
        match model_id {
            id if id.starts_with("llama3.2") => 128_000,
            id if id.starts_with("qwen2.5-coder") => 128_000,
            id if id.starts_with("mistral") => 32_000,
            id if id.starts_with("gemma") => 8_192,
            _ => 0,
        }
    }

    /// One-time availability + discovery probe. Cached via [`OnceCell`] so
    /// subsequent calls are free. Checks `/api/tags` (surfacing a helpful
    /// "ollama pull" message when the model is missing), then — best-effort —
    /// fetches `/api/show` for the context window and capabilities. Returns the
    /// discovered [`ModelMeta`] (empty when `/api/show` was unavailable).
    async fn ensure_model_available(&self) -> Result<&ModelMeta, ModelError> {
        self.model_checked
            .get_or_try_init(|| async {
                let url = format!("{}/api/tags", self.base_url);
                let resp = self
                    .http_client
                    .get(&url)
                    .timeout(self.timeout)
                    .send()
                    .await
                    .map_err(|e| transport_error(e, &self.base_url))?;
                if !resp.status().is_success() {
                    return Err(map_status_error(resp, &self.model_id).await);
                }
                let body: TagsResponse =
                    resp.json().await.map_err(|e| ModelError::ProviderError {
                        code: 0,
                        message: format!("tags decode failed: {e}"),
                    })?;
                let found = body
                    .models
                    .iter()
                    .any(|m| name_matches(&m.name, &self.model_id));
                if !found {
                    return Err(ModelError::ProviderError {
                        code: 404,
                        message: format!(
                            "Model {} not found. Run: ollama pull {}",
                            self.model_id, self.model_id
                        ),
                    });
                }
                // Best-effort discovery — never fails the call.
                Ok(self.discover_meta().await)
            })
            .await
    }

    /// Best-effort `POST /api/show` discovery. Returns an empty [`ModelMeta`]
    /// on any failure (404, transport error, decode error, missing fields) so
    /// discovery being unavailable never errors the whole call.
    async fn discover_meta(&self) -> ModelMeta {
        let url = format!("{}/api/show", self.base_url);
        let body = ShowRequest {
            model: self.model_id.clone(),
        };
        let resp = match self
            .http_client
            .post(&url)
            .header("content-type", "application/json")
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => r,
            _ => return ModelMeta::default(),
        };
        let parsed: ShowResponse = match resp.json().await {
            Ok(p) => p,
            Err(_) => return ModelMeta::default(),
        };
        let context_length = parsed
            .model_info
            .iter()
            .find(|(k, _)| k.ends_with(".context_length"))
            .and_then(|(_, v)| v.as_u64())
            .map(|n| n as u32);
        ModelMeta {
            context_length,
            capabilities: parsed.capabilities,
        }
    }
}

/// Ollama tag names often look like `"llama3.2:latest"` or `"llama3.2:3b"`.
/// Match if the request id equals the full tag or its bare-name prefix.
fn name_matches(tag: &str, requested: &str) -> bool {
    if tag == requested {
        return true;
    }
    let bare = tag.split(':').next().unwrap_or(tag);
    bare == requested
}

// ============================================================================
// Wire-format types (Ollama Chat API)
// ============================================================================

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OllamaTool>,
    /// Constrained-decoding JSON schema (Ollama's `format` parameter). Set only
    /// in structured-tool-calls mode (`ModelParams.structured_tool_calls`); when
    /// present, native `tools` are dropped and the model is forced to emit a
    /// single schema-constrained JSON object instead of routing tool calls
    /// through Llama's `<|python_tag|>` channel (which Ollama does not parse,
    /// causing the call to leak into `message.content`).
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Default)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

impl OllamaOptions {
    fn is_empty(&self) -> bool {
        self.num_predict.is_none()
            && self.temperature.is_none()
            && self.top_p.is_none()
            && self.stop.is_empty()
    }
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OllamaToolCall>,
    /// Ollama uses `tool_call_id` for tool-result messages (mirrors OpenAI).
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Debug, Serialize)]
struct OllamaFunctionCall {
    name: String,
    /// Ollama wants arguments as a JSON object (NOT a JSON-encoded string).
    arguments: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    kind: &'static str, // always "function"
    function: OllamaToolFunction,
}

#[derive(Debug, Serialize)]
struct OllamaToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaResponse {
    #[serde(default)]
    message: OllamaResponseMessage,
    #[serde(default)]
    #[allow(dead_code)]
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaResponseMessage {
    #[serde(default)]
    #[allow(dead_code)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OllamaResponseToolCall>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponseToolCall {
    #[serde(default)]
    function: OllamaResponseFunctionCall,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaResponseFunctionCall {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagModel>,
}

#[derive(Debug, Deserialize)]
struct TagModel {
    name: String,
}

#[derive(Debug, Serialize)]
struct ShowRequest {
    model: String,
}

#[derive(Debug, Default, Deserialize)]
struct ShowResponse {
    /// Map of architecture-specific keys; we look for `*.context_length`.
    #[serde(default)]
    model_info: serde_json::Map<String, serde_json::Value>,
    /// Top-level capabilities array (may contain `"tools"`).
    #[serde(default)]
    capabilities: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EmbedRequest {
    model: String,
    input: String,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    #[serde(default)]
    prompt_eval_count: Option<u32>,
}

// ============================================================================
// Conversions
// ============================================================================

fn build_request(
    model_id: &str,
    keep_alive: &Option<String>,
    req: &ModelRequest,
    stream: bool,
) -> OllamaRequest {
    // Structured-tool-calls mode (opt-in): constrained decoding via `format`.
    // We send NO native tools — describing them in a system message instead —
    // and force the model to emit a single schema-constrained JSON object. The
    // `tool` enum is the critical constraint: it eliminates the `<|python_tag|>`
    // leak that small local models (llama3.2) otherwise produce.
    let structured = req.params.structured_tool_calls && !req.tools.is_empty();

    let mut messages: Vec<OllamaMessage> = req.messages.iter().map(message_to_ollama).collect();

    let (tools, format) = if structured {
        // Inject a system message describing the tools. Merge into an existing
        // leading system message when present; otherwise prepend a new one.
        let preamble = structured_tools_preamble(&req.tools);
        if let Some(first) = messages.first_mut() {
            if first.role == "system" {
                first.content = format!("{}\n\n{}", first.content, preamble);
            } else {
                messages.insert(0, system_message(preamble));
            }
        } else {
            messages.insert(0, system_message(preamble));
        }

        let mut tool_names: Vec<serde_json::Value> = req
            .tools
            .iter()
            .map(|t| serde_json::Value::String(t.name.clone()))
            .collect();
        tool_names.push(serde_json::Value::String("final".into()));
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": tool_names},
                "arguments": {"type": "object"},
                "content": {"type": "string"}
            },
            "required": ["tool"]
        });
        (Vec::new(), Some(schema))
    } else {
        let tools: Vec<OllamaTool> = req
            .tools
            .iter()
            .map(|t: &ToolSchema| OllamaTool {
                kind: "function",
                function: OllamaToolFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                },
            })
            .collect();
        // Issue #139: the harness sets `params.output_schema` for the terminal
        // turn of an output-schema-enforced ReAct leaf. Route it into the same
        // `format` constrained-decoding channel the structured-tool-calls path
        // uses, so the model is forced onto the schema. (When structured tool
        // calls ARE active, that schema wins — the `if structured` arm above —
        // since the leaf is still requesting tools, not emitting its terminal.)
        let format = req.params.output_schema.clone();
        (tools, format)
    };

    let options = OllamaOptions {
        num_predict: req.params.max_tokens,
        temperature: req.params.temperature,
        top_p: req.params.top_p,
        stop: req.params.stop_sequences.clone(),
    };

    OllamaRequest {
        model: model_id.into(),
        messages,
        stream,
        keep_alive: keep_alive.clone(),
        options: if options.is_empty() {
            None
        } else {
            Some(options)
        },
        tools,
        format,
    }
}

fn system_message(content: String) -> OllamaMessage {
    OllamaMessage {
        role: "system".into(),
        content,
        tool_calls: vec![],
        tool_call_id: None,
    }
}

/// Build the system-message preamble for structured-tool-calls mode. Describes
/// each tool's name, description, and parameter property names/types (read from
/// `t.input_schema`), then gives the model explicit single-JSON-object output
/// instructions. This — together with the `format` schema's `tool` enum — keeps
/// tool calls on the constrained-JSON channel and away from `<|python_tag|>`.
fn structured_tools_preamble(tools: &[ToolSchema]) -> String {
    let mut out = String::from("You have access to the following tools:\n");
    for t in tools {
        out.push_str(&format!("\n- {}: {}", t.name, t.description));
        if let Some(props) = t.input_schema.get("properties").and_then(|p| p.as_object()) {
            if !props.is_empty() {
                let params: Vec<String> = props
                    .iter()
                    .map(|(name, schema)| {
                        let ty = schema.get("type").and_then(|v| v.as_str()).unwrap_or("any");
                        format!("{name} ({ty})")
                    })
                    .collect();
                out.push_str(&format!("\n  parameters: {}", params.join(", ")));
            }
        }
    }
    out.push_str(
        "\n\nRespond with a SINGLE JSON object and nothing else. To call a tool, \
set \"tool\" to the tool name and \"arguments\" to its inputs. When the task is \
fully done, set \"tool\" to \"final\" and put your reply in \"content\".",
    );
    out
}

fn message_to_ollama(m: &crate::model::Message) -> OllamaMessage {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    match &m.content {
        Content::Text { text } => OllamaMessage {
            role: role.into(),
            content: text.clone(),
            tool_calls: vec![],
            tool_call_id: None,
        },
        Content::ToolCall(call) => OllamaMessage {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: vec![OllamaToolCall {
                function: OllamaFunctionCall {
                    name: call.name.clone(),
                    arguments: call.input.clone(),
                },
            }],
            tool_call_id: None,
        },
        Content::ToolResult(r) => OllamaMessage {
            role: "tool".into(),
            content: r.content.clone(),
            tool_calls: vec![],
            tool_call_id: Some(r.tool_use_id.clone()),
        },
        Content::Image { media_type, .. } => OllamaMessage {
            role: role.into(),
            content: format!("[image: {media_type}]"),
            tool_calls: vec![],
            tool_call_id: None,
        },
    }
}

fn parse_response(body: OllamaResponse, structured: bool) -> ModelResponse {
    let usage = TokenUsage {
        input_tokens: body.prompt_eval_count.unwrap_or(0),
        output_tokens: body.eval_count.unwrap_or(0),
        cache_read_tokens: None,
        cache_write_tokens: None,
    };

    if structured {
        // In structured mode the assistant content is a single constrained JSON
        // object — never a native tool_calls array. Parsing it back into a
        // tool-use block (rather than treating the raw JSON as answer text) is
        // precisely what avoids the `<|python_tag|>` leak: the tool call never
        // touches the native channel.
        let raw = body.message.content.unwrap_or_default();
        let (content, stop_reason) = parse_structured_content(&raw, 0);
        return ModelResponse {
            content,
            usage,
            stop_reason,
        };
    }

    let mut content: Vec<ContentBlock> = Vec::new();
    if let Some(text) = body.message.content {
        if !text.is_empty() {
            content.push(ContentBlock::Text { text });
        }
    }
    for (i, tc) in body.message.tool_calls.into_iter().enumerate() {
        let input = if tc.function.arguments.is_null() {
            serde_json::json!({})
        } else {
            tc.function.arguments
        };
        content.push(ContentBlock::ToolUse(ToolCall {
            id: format!("call-{i}"),
            name: tc.function.name,
            input,
        }));
    }

    ModelResponse {
        content,
        usage,
        stop_reason: parse_stop_reason(body.done_reason.as_deref()),
    }
}

/// Parse the constrained-decoding JSON object produced in structured-tool-calls
/// mode into `(content blocks, stop reason)`. `index` is used to synthesize the
/// tool-call id, reusing this file's `call-{i}` convention.
///
/// Defensive: if `raw` is missing/empty/not valid JSON/lacks a `tool` field,
/// fall back to a single `Text` block with the raw content and `EndTurn` —
/// weak models occasionally violate even constrained decoding, and we never
/// panic on their output.
fn parse_structured_content(raw: &str, index: usize) -> (Vec<ContentBlock>, StopReason) {
    let fallback = || {
        (
            vec![ContentBlock::Text { text: raw.into() }],
            StopReason::EndTurn,
        )
    };
    // Capable/cloud models often ignore the constrained-decoding grammar and
    // wrap the JSON tool call in a markdown code fence. Reuse the plan parser's
    // fence stripping so a fenced `{"tool":...}` still dispatches instead of
    // being mis-read as a final text answer.
    let parse_input = crate::plan::strip_code_fence(raw.trim());
    let value: serde_json::Value = match serde_json::from_str(parse_input) {
        Ok(v) => v,
        Err(_) => return fallback(),
    };
    let tool = match value.get("tool").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return fallback(),
    };
    if tool == "final" {
        let text = value
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        return (vec![ContentBlock::Text { text }], StopReason::EndTurn);
    }
    let input = match value.get("arguments") {
        Some(a) if a.is_object() => a.clone(),
        _ => serde_json::json!({}),
    };
    (
        vec![ContentBlock::ToolUse(ToolCall {
            id: format!("call-{index}"),
            name: tool.to_string(),
            input,
        })],
        StopReason::ToolUse,
    )
}

fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("stop") => StopReason::EndTurn,
        _ => StopReason::EndTurn,
    }
}

// ============================================================================
// HTTP error helpers (no retry — fail fast per spec)
// ============================================================================

fn transport_error(e: reqwest::Error, base_url: &str) -> ModelError {
    if e.is_timeout() {
        return ModelError::Timeout;
    }
    if e.is_connect() || e.is_request() {
        return ModelError::ProviderError {
            code: 0,
            message: format!("Ollama not running at {base_url}"),
        };
    }
    ModelError::ProviderError {
        code: 0,
        message: format!("HTTP transport error: {e}"),
    }
}

async fn map_status_error(resp: reqwest::Response, model_id: &str) -> ModelError {
    let status = resp.status();
    let code = status.as_u16();
    let body_text = resp.text().await.unwrap_or_default();
    if code == 404 {
        let combined = body_text.to_ascii_lowercase();
        if combined.contains("not found") || combined.contains("model") || combined.is_empty() {
            return ModelError::ProviderError {
                code: 404,
                message: format!("Model {model_id} not found. Run: ollama pull {model_id}"),
            };
        }
    }
    let message = if body_text.is_empty() {
        status.canonical_reason().unwrap_or("error").to_string()
    } else {
        body_text.chars().take(500).collect()
    };
    if code == 408 || code == 504 {
        return ModelError::Timeout;
    }
    ModelError::ProviderError { code, message }
}

// ============================================================================
// ModelInterface impl
// ============================================================================

impl ModelInterface for OllamaModelInterface {
    async fn call(&self, request: ModelRequest) -> Result<ModelResponse, ModelError> {
        let meta = self.ensure_model_available().await?;
        self.guard_tool_support(&request, meta)?;
        let body = build_request(&self.model_id, &self.keep_alive, &request, false);
        let url = format!("{}/api/chat", self.base_url);
        let encoded = serde_json::to_string(&body).map_err(|e| ModelError::ProviderError {
            code: 0,
            message: format!("request encode failed: {e}"),
        })?;
        let resp = self
            .http_client
            .post(&url)
            .header("content-type", "application/json")
            .timeout(self.timeout)
            .body(encoded)
            .send()
            .await
            .map_err(|e| transport_error(e, &self.base_url))?;
        if !resp.status().is_success() {
            return Err(map_status_error(resp, &self.model_id).await);
        }
        let parsed: OllamaResponse = resp.json().await.map_err(|e| ModelError::ProviderError {
            code: 0,
            message: format!("response decode failed: {e}"),
        })?;
        let structured = request.params.structured_tool_calls && !request.tools.is_empty();
        Ok(parse_response(parsed, structured))
    }

    async fn call_streaming(&self, request: ModelRequest) -> Result<ModelStream, ModelError> {
        let meta = self.ensure_model_available().await?;
        self.guard_tool_support(&request, meta)?;
        let body = build_request(&self.model_id, &self.keep_alive, &request, true);
        let url = format!("{}/api/chat", self.base_url);
        let encoded = serde_json::to_string(&body).map_err(|e| ModelError::ProviderError {
            code: 0,
            message: format!("request encode failed: {e}"),
        })?;
        let base_url = self.base_url.clone();
        let model_id = self.model_id.clone();
        let resp = self
            .http_client
            .post(&url)
            .header("content-type", "application/json")
            .timeout(self.timeout)
            .body(encoded)
            .send()
            .await
            .map_err(|e| transport_error(e, &base_url))?;
        if !resp.status().is_success() {
            return Err(map_status_error(resp, &model_id).await);
        }
        let structured = request.params.structured_tool_calls && !request.tools.is_empty();
        Ok(Box::pin(ndjson_to_events(resp, structured)))
    }

    async fn count_tokens(&self, request: &ModelRequest) -> Result<u32, ModelError> {
        // Try the embed endpoint; fall back to bytes/4 on missing field or
        // any transport failure. Matches the openai.rs fallback strategy.
        let text = concat_request_text(request);
        if let Some(n) = self.try_embed_count(&text).await {
            return Ok(n);
        }
        Ok((text.len() / 4) as u32)
    }

    fn provider(&self) -> ProviderInfo {
        // `provider()` is synchronous, so it cannot await `/api/show`. Read the
        // probe cache non-blockingly: prefer a discovered context length if the
        // probe has already run; otherwise fall back to the static table.
        let context_window = self
            .model_checked
            .get()
            .and_then(|m| m.context_length)
            .unwrap_or_else(|| Self::context_window(&self.model_id));
        ProviderInfo {
            name: "ollama".into(),
            model_id: self.model_id.clone(),
            context_window,
        }
    }
}

impl OllamaModelInterface {
    /// Reject tool-bearing requests when the model does not support tools.
    /// Capability is determined solely by the `/api/show` `capabilities` array:
    /// the model is tool-capable iff `capabilities` contains `"tools"`. Empty or
    /// unavailable `/api/show` metadata ⟹ NOT tool-capable (fail closed).
    fn guard_tool_support(
        &self,
        request: &ModelRequest,
        meta: &ModelMeta,
    ) -> Result<(), ModelError> {
        if request.tools.is_empty() {
            return Ok(());
        }
        let supported = meta.supports_tools();
        if !supported {
            return Err(ModelError::ProviderError {
                code: 0,
                message: format!("Model {} does not support tool calling", self.model_id),
            });
        }
        Ok(())
    }

    async fn try_embed_count(&self, text: &str) -> Option<u32> {
        let url = format!("{}/api/embed", self.base_url);
        let body = EmbedRequest {
            model: self.model_id.clone(),
            input: text.into(),
        };
        let resp = self
            .http_client
            .post(&url)
            .header("content-type", "application/json")
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let parsed: EmbedResponse = resp.json().await.ok()?;
        parsed.prompt_eval_count
    }
}

fn concat_request_text(request: &ModelRequest) -> String {
    let mut out = String::new();
    for m in &request.messages {
        match &m.content {
            Content::Text { text } => out.push_str(text),
            Content::ToolCall(tc) => {
                out.push_str(&tc.name);
                out.push(' ');
                out.push_str(&tc.input.to_string());
            }
            Content::ToolResult(tr) => out.push_str(&tr.content),
            Content::Image { .. } => {}
        }
        out.push('\n');
    }
    out
}

// ============================================================================
// NDJSON stream parsing — Ollama chat streaming
// ============================================================================

/// Ollama streams chat results as **newline-delimited JSON** (one full JSON
/// object per line, NOT SSE). Each line carries an incremental
/// `message.content` delta; `tool_calls` arrive as full argument objects per
/// chunk (not partial-fragment strings); the terminator line carries
/// `done: true` plus `prompt_eval_count` and `eval_count`.
fn ndjson_to_events(
    resp: reqwest::Response,
    structured: bool,
) -> impl futures_core::Stream<Item = Result<StreamEvent, ModelError>> + Send + 'static {
    async_stream::stream! {
        let stream = resp.bytes_stream();
        futures_util::pin_mut!(stream);
        let mut buf = String::new();
        let mut started = false;
        let mut tool_indices_seen: std::collections::HashSet<u32> =
            std::collections::HashSet::new();
        let mut content_index: u32 = 0;
        let mut content_open = false;
        // Structured mode: the constrained JSON object arrives as `message.content`
        // text deltas spread across chunks. We must NOT surface that raw JSON to
        // the user — instead buffer it for the whole response and parse it at the
        // `done` chunk exactly like `parse_response`, emitting reconstructable
        // tool / text events. This keeps the tool call off the native channel and
        // is what prevents the `<|python_tag|>` leak in streaming mode too.
        let mut structured_content = String::new();

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
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(idx) = buf.find('\n') {
                let raw_line = buf[..idx].to_string();
                buf = buf[idx + 1..].to_string();
                let line = raw_line.trim_end_matches('\r').trim();
                if line.is_empty() {
                    continue;
                }
                let value: serde_json::Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if !started {
                    started = true;
                    yield Ok(StreamEvent::MessageStart);
                }
                if structured {
                    // Buffer content deltas; defer all emission to the `done` chunk.
                    if let Some(text) = value
                        .get("message")
                        .and_then(|m| m.get("content"))
                        .and_then(|v| v.as_str())
                    {
                        structured_content.push_str(text);
                    }
                    if value.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
                        let (content, stop_reason) =
                            parse_structured_content(&structured_content, 0);
                        for block in content {
                            match block {
                                ContentBlock::ToolUse(call) => {
                                    let partial = serde_json::to_string(&call.input)
                                        .unwrap_or_else(|_| "{}".into());
                                    yield Ok(StreamEvent::ToolUseStart {
                                        index: 1,
                                        id: call.id,
                                        name: call.name,
                                    });
                                    yield Ok(StreamEvent::ToolUseDelta {
                                        index: 1,
                                        partial_json: partial,
                                    });
                                }
                                ContentBlock::Text { text } => {
                                    yield Ok(StreamEvent::ContentBlockDelta {
                                        index: 0,
                                        delta: text,
                                    });
                                }
                                ContentBlock::Thinking { text } => {
                                    yield Ok(StreamEvent::ThinkingDelta {
                                        index: 0,
                                        delta: text,
                                    });
                                }
                            }
                        }
                        let usage = TokenUsage {
                            input_tokens: value
                                .get("prompt_eval_count")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as u32,
                            output_tokens: value
                                .get("eval_count")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as u32,
                            cache_read_tokens: None,
                            cache_write_tokens: None,
                        };
                        yield Ok(StreamEvent::MessageStop { usage, stop_reason });
                        return;
                    }
                    continue;
                }
                let message = value.get("message");
                if let Some(text) = message
                    .and_then(|m| m.get("content"))
                    .and_then(|v| v.as_str())
                {
                    if !text.is_empty() {
                        content_open = true;
                        yield Ok(StreamEvent::ContentBlockDelta {
                            index: content_index,
                            delta: text.into(),
                        });
                    }
                }
                if let Some(tcs) = message
                    .and_then(|m| m.get("tool_calls"))
                    .and_then(|v| v.as_array())
                {
                    for (i, tc) in tcs.iter().enumerate() {
                        // Ollama identifies a distinct tool call by `function.index`,
                        // which is stable across chunks. A single response with
                        // multiple calls streams them in SEPARATE chunks, each a
                        // one-element `tool_calls` array — so the array position `i`
                        // is 0 for every call and must NOT be used as the index, or
                        // every call collapses onto the same block and their argument
                        // JSON fragments concatenate into garbage. Fall back to `i`
                        // only when `function.index` is absent.
                        let model_index = tc
                            .get("function")
                            .and_then(|f| f.get("index"))
                            .and_then(|v| v.as_u64())
                            .map(|n| n as u32)
                            .unwrap_or(i as u32);
                        let event_index = model_index + 1;
                        let first_seen = !tool_indices_seen.contains(&event_index);
                        if first_seen {
                            tool_indices_seen.insert(event_index);
                            if content_open {
                                yield Ok(StreamEvent::ContentBlockStop {
                                    index: content_index,
                                });
                                content_open = false;
                                content_index = event_index;
                            }
                            // Ollama delivers the full call (id + name + complete
                            // args) on the chunk — emit a ToolUseStart carrying the
                            // name and id so the accumulator can reconstruct the
                            // call faithfully. A missing id is synthesized stably.
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
                            yield Ok(StreamEvent::ToolUseStart {
                                index: event_index,
                                id,
                                name,
                            });
                        }
                        if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")) {
                            let partial = serde_json::to_string(args)
                                .unwrap_or_else(|_| "{}".into());
                            yield Ok(StreamEvent::ToolUseDelta {
                                index: event_index,
                                partial_json: partial,
                            });
                        }
                    }
                }
                if value.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
                    let usage = TokenUsage {
                        input_tokens: value
                            .get("prompt_eval_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as u32,
                        output_tokens: value
                            .get("eval_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as u32,
                        cache_read_tokens: None,
                        cache_write_tokens: None,
                    };
                    let stop_reason = parse_stop_reason(
                        value.get("done_reason").and_then(|v| v.as_str()),
                    );
                    yield Ok(StreamEvent::MessageStop { usage, stop_reason });
                    return;
                }
            }
        }
        // Defensive: if the connection drops without `done:true`, still emit
        // a MessageStop so consumers see a terminator.
        yield Ok(StreamEvent::MessageStop {
            usage: TokenUsage::default(),
            stop_reason: StopReason::EndTurn,
        });
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

    fn req(messages: Vec<Message>) -> ModelRequest {
        ModelRequest {
            messages,
            tools: vec![],
            params: ModelParams::default(),
            stream: false,
        }
    }

    // ── constructors / defaults ─────────────────────────────────────────────

    #[test]
    fn new_uses_localhost_defaults() {
        let c = OllamaModelInterface::new("llama3.2");
        assert_eq!(c.base_url, "http://localhost:11434");
        assert_eq!(c.model_id, "llama3.2");
        assert_eq!(c.timeout, Duration::from_secs(300));
        assert_eq!(c.keep_alive.as_deref(), Some("5m"));
    }

    #[test]
    fn with_base_url_overrides() {
        let c = OllamaModelInterface::with_base_url("mistral", "http://remote:9999");
        assert_eq!(c.base_url, "http://remote:9999");
        assert_eq!(c.model_id, "mistral");
    }

    #[test]
    fn defaults_match_spec() {
        assert_eq!(
            OllamaModelInterface::DEFAULT_BASE_URL,
            "http://localhost:11434"
        );
        assert_eq!(
            OllamaModelInterface::DEFAULT_TIMEOUT,
            Duration::from_secs(300)
        );
        assert_eq!(OllamaModelInterface::DEFAULT_KEEP_ALIVE, "5m");
    }

    // ── build_request ───────────────────────────────────────────────────────

    #[test]
    fn build_request_serializes_options_and_keep_alive() {
        let mut r = req(vec![user("hi")]);
        r.params.max_tokens = Some(256);
        r.params.temperature = Some(0.7);
        r.params.top_p = Some(0.9);
        r.params.stop_sequences = vec!["END".into()];
        let body = build_request("llama3.2", &Some("10m".into()), &r, false);
        let s = serde_json::to_string(&body).unwrap();
        assert!(s.contains("\"keep_alive\":\"10m\""), "wire: {s}");
        assert!(s.contains("\"num_predict\":256"), "wire: {s}");
        assert!(s.contains("\"temperature\":0.7"), "wire: {s}");
        assert!(s.contains("\"top_p\":0.9"), "wire: {s}");
        assert!(s.contains("\"stop\":[\"END\"]"), "wire: {s}");
        assert!(!s.contains("\"stream\":true"), "default stream=false");
    }

    #[test]
    fn build_request_serializes_tools() {
        let mut r = req(vec![user("hi")]);
        r.tools.push(ToolSchema {
            name: "search".into(),
            description: "search the web".into(),
            input_schema: serde_json::json!({"type":"object"}),
        });
        let body = build_request("llama3.2", &None, &r, false);
        let s = serde_json::to_string(&body).unwrap();
        assert!(s.contains("\"tools\":["), "wire: {s}");
        assert!(s.contains("\"name\":\"search\""), "wire: {s}");
        assert!(s.contains("\"type\":\"function\""), "wire: {s}");
    }

    #[test]
    fn build_request_tool_call_uses_object_arguments() {
        let r = req(vec![Message {
            role: Role::Assistant,
            content: Content::ToolCall(ToolCall {
                id: "call-0".into(),
                name: "fetch".into(),
                input: serde_json::json!({"url":"x"}),
            }),
        }]);
        let body = build_request("llama3.2", &None, &r, false);
        let s = serde_json::to_string(&body.messages[0]).unwrap();
        // arguments must be an object, NOT a JSON-encoded string
        assert!(s.contains("\"arguments\":{\"url\":\"x\"}"), "wire: {s}");
        assert!(!s.contains("\"arguments\":\""), "wire: {s}");
    }

    #[test]
    fn build_request_tool_result_maps_to_tool_role() {
        let r = req(vec![Message {
            role: Role::Tool,
            content: Content::ToolResult(ToolResult {
                tool_use_id: "call-0".into(),
                content: "ok".into(),
                is_error: false,
            }),
        }]);
        let body = build_request("llama3.2", &None, &r, false);
        let m = &body.messages[0];
        assert_eq!(m.role, "tool");
        assert_eq!(m.content, "ok");
        assert_eq!(m.tool_call_id.as_deref(), Some("call-0"));
    }

    #[test]
    fn thinking_block_omitted_in_request() {
        // Thinking blocks live in ContentBlock (response side), not Content
        // (request side). Confirm a normal request round-trips with no
        // "thinking" key, which proves we never emit one.
        let r = req(vec![user("hi")]);
        let body = build_request("llama3.2", &None, &r, false);
        let s = serde_json::to_string(&body).unwrap();
        assert!(!s.contains("thinking"), "wire: {s}");
    }

    // ── structured tool calls (opt-in constrained decoding) ────────────────
    //
    // This path describes the tools in a system message and constrains decoding
    // via Ollama's `format` schema instead of sending native `tools`. Because
    // the tool call rides the constrained-JSON channel — never the native
    // tool_calls / `<|python_tag|>` path — small local models can no longer leak
    // a tool call into `message.content` as unparsed text.

    fn structured_tool_req() -> ModelRequest {
        let mut r = req(vec![user("write a summary file")]);
        r.params.structured_tool_calls = true;
        r.tools.push(ToolSchema {
            name: "write_file".into(),
            description: "write a file".into(),
            input_schema: serde_json::json!({
                "type":"object",
                "properties":{"path":{"type":"string"},"content":{"type":"string"}}
            }),
        });
        r.tools.push(ToolSchema {
            name: "read_file".into(),
            description: "read a file".into(),
            input_schema: serde_json::json!({
                "type":"object",
                "properties":{"path":{"type":"string"}}
            }),
        });
        r
    }

    #[test]
    fn build_request_structured_sets_format_drops_tools_adds_system() {
        let r = structured_tool_req();
        let body = build_request("llama3.2", &None, &r, false);
        // Native tools dropped in structured mode.
        assert!(body.tools.is_empty(), "native tools must be empty");
        // format schema present with tool enum = tool names + "final".
        let format = body.format.expect("format schema must be present");
        let enum_vals = format
            .pointer("/properties/tool/enum")
            .and_then(|v| v.as_array())
            .expect("tool enum present");
        let names: Vec<&str> = enum_vals.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"final"));
        // A system message describing the tools is prepended.
        assert_eq!(body.messages[0].role, "system");
        assert!(body.messages[0].content.contains("write_file"));
        assert!(body.messages[0].content.contains("read_file"));
        assert!(body.messages[0].content.contains("SINGLE JSON object"));
    }

    #[test]
    fn build_request_structured_merges_into_existing_system_message() {
        let mut r = structured_tool_req();
        r.messages.insert(
            0,
            Message {
                role: Role::System,
                content: Content::Text {
                    text: "You are terse.".into(),
                },
            },
        );
        let body = build_request("llama3.2", &None, &r, false);
        // Only one system message; original content preserved + preamble merged.
        let system_count = body.messages.iter().filter(|m| m.role == "system").count();
        assert_eq!(system_count, 1);
        assert!(body.messages[0].content.contains("You are terse."));
        assert!(body.messages[0].content.contains("write_file"));
    }

    #[test]
    fn build_request_structured_off_when_no_tools() {
        // Flag on but no tools → unchanged behavior, no format.
        let mut r = req(vec![user("hi")]);
        r.params.structured_tool_calls = true;
        let body = build_request("llama3.2", &None, &r, false);
        assert!(body.format.is_none());
    }

    #[test]
    fn build_request_structured_off_by_default() {
        // Flag default off with tools present → native tools, no format.
        let mut r = req(vec![user("hi")]);
        r.tools.push(ToolSchema {
            name: "search".into(),
            description: "search the web".into(),
            input_schema: serde_json::json!({"type":"object"}),
        });
        let body = build_request("llama3.2", &None, &r, false);
        assert!(body.format.is_none());
        assert_eq!(body.tools.len(), 1);
    }

    // ── output-schema constrained decoding (issue #139) ────────────────────

    #[test]
    fn build_request_output_schema_populates_format_channel() {
        // #139 AC1: `params.output_schema` (set by the harness for an
        // output-schema-enforced terminal turn) routes into the Ollama `format`
        // constrained-decoding channel verbatim, even with NO tools.
        let mut r = req(vec![user("answer")]);
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ok", "error"]}},
            "required": ["status"]
        });
        r.params.output_schema = Some(schema.clone());
        let body = build_request("llama3.2", &None, &r, false);
        assert_eq!(
            body.format.as_ref(),
            Some(&schema),
            "output_schema must populate the Ollama `format` channel"
        );
    }

    #[test]
    fn build_request_no_output_schema_leaves_format_none() {
        // Absent output_schema (the default) keeps `format` None — byte-identical
        // to pre-#139.
        let r = req(vec![user("hi")]);
        let body = build_request("llama3.2", &None, &r, false);
        assert!(body.format.is_none());
    }

    #[test]
    fn parse_response_structured_tool_call() {
        let body: OllamaResponse = serde_json::from_value(serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "{\"tool\":\"write_file\",\"arguments\":{\"path\":\"SUMMARY.md\",\"content\":\"hi\"}}"
            },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1
        }))
        .unwrap();
        let r = parse_response(body, true);
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.content.len(), 1);
        match &r.content[0] {
            ContentBlock::ToolUse(tc) => {
                assert_eq!(tc.name, "write_file");
                assert_eq!(
                    tc.input,
                    serde_json::json!({"path":"SUMMARY.md","content":"hi"})
                );
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn parse_response_structured_final() {
        let body: OllamaResponse = serde_json::from_value(serde_json::json!({
            "message": {"role":"assistant","content":"{\"tool\":\"final\",\"content\":\"all done\"}"},
            "done": true,
            "done_reason": "stop"
        }))
        .unwrap();
        let r = parse_response(body, true);
        assert_eq!(r.stop_reason, StopReason::EndTurn);
        assert_eq!(
            r.content,
            vec![ContentBlock::Text {
                text: "all done".into()
            }]
        );
    }

    #[test]
    fn parse_response_structured_malformed_falls_back_to_text() {
        // Weak model violates constrained decoding: not valid JSON. We must not
        // panic — fall back to a Text block with the raw content and EndTurn.
        let body: OllamaResponse = serde_json::from_value(serde_json::json!({
            "message": {"role":"assistant","content":"oops not json"},
            "done": true,
            "done_reason": "stop"
        }))
        .unwrap();
        let r = parse_response(body, true);
        assert_eq!(r.stop_reason, StopReason::EndTurn);
        assert_eq!(
            r.content,
            vec![ContentBlock::Text {
                text: "oops not json".into()
            }]
        );
    }

    // ── structured fence stripping (capable/cloud models wrap JSON) ─────────

    // Regression for the exact gemma-cloud output: the constrained JSON tool
    // call arrives inside a ```json fence. Must dispatch, not fall back to Text.
    #[test]
    fn parse_structured_json_fenced_tool_call_dispatches() {
        let raw = "```json\n{\"tool\":\"web_search\",\"arguments\":{\"query\":\"x\"}}\n```";
        let (content, stop) = parse_structured_content(raw, 0);
        assert_eq!(stop, StopReason::ToolUse);
        match &content[0] {
            ContentBlock::ToolUse(tc) => {
                assert_eq!(tc.name, "web_search");
                assert_eq!(tc.input, serde_json::json!({"query":"x"}));
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    // A bare ``` fence (no language tag) also strips and dispatches.
    #[test]
    fn parse_structured_bare_fenced_tool_call_dispatches() {
        let raw = "```\n{\"tool\":\"web_search\",\"arguments\":{\"query\":\"y\"}}\n```";
        let (content, stop) = parse_structured_content(raw, 0);
        assert_eq!(stop, StopReason::ToolUse);
        match &content[0] {
            ContentBlock::ToolUse(tc) => assert_eq!(tc.name, "web_search"),
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    // A fenced `final` envelope still resolves to a Text/EndTurn answer.
    #[test]
    fn parse_structured_fenced_final_is_text() {
        let raw = "```json\n{\"tool\":\"final\",\"content\":\"done\"}\n```";
        let (content, stop) = parse_structured_content(raw, 0);
        assert_eq!(stop, StopReason::EndTurn);
        assert_eq!(
            content,
            vec![ContentBlock::Text {
                text: "done".into()
            }]
        );
    }

    // Un-fenced tool calls (grammar-honoring models) still dispatch — no regression.
    #[test]
    fn parse_structured_raw_tool_call_still_dispatches() {
        let raw = "{\"tool\":\"web_search\",\"arguments\":{\"query\":\"z\"}}";
        let (content, stop) = parse_structured_content(raw, 0);
        assert_eq!(stop, StopReason::ToolUse);
        match &content[0] {
            ContentBlock::ToolUse(tc) => assert_eq!(tc.name, "web_search"),
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    // Genuine garbage still falls back to a Text block with EndTurn.
    #[test]
    fn parse_structured_garbage_falls_back_to_text() {
        let raw = "not json at all";
        let (content, stop) = parse_structured_content(raw, 0);
        assert_eq!(stop, StopReason::EndTurn);
        assert_eq!(
            content,
            vec![ContentBlock::Text {
                text: "not json at all".into()
            }]
        );
    }

    // ── stop reason ─────────────────────────────────────────────────────────

    #[test]
    fn stop_reason_mapping_stop() {
        assert_eq!(parse_stop_reason(Some("stop")), StopReason::EndTurn);
        assert_eq!(parse_stop_reason(None), StopReason::EndTurn);
        assert_eq!(parse_stop_reason(Some("???")), StopReason::EndTurn);
    }

    #[test]
    fn stop_reason_mapping_tool_calls() {
        assert_eq!(parse_stop_reason(Some("tool_calls")), StopReason::ToolUse);
        assert_eq!(parse_stop_reason(Some("length")), StopReason::MaxTokens);
    }

    // ── parse_response ──────────────────────────────────────────────────────

    #[test]
    fn parse_response_extracts_usage() {
        let body: OllamaResponse = serde_json::from_value(serde_json::json!({
            "message": {"role": "assistant", "content": "hi"},
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 7,
            "eval_count": 2
        }))
        .unwrap();
        let r = parse_response(body, false);
        assert_eq!(r.usage.input_tokens, 7);
        assert_eq!(r.usage.output_tokens, 2);
        assert_eq!(r.stop_reason, StopReason::EndTurn);
        assert_eq!(r.content, vec![ContentBlock::Text { text: "hi".into() }]);
    }

    #[test]
    fn parse_response_cache_fields_none() {
        let body: OllamaResponse = serde_json::from_value(serde_json::json!({
            "message": {"role": "assistant", "content": "x"},
            "done": true,
            "prompt_eval_count": 1,
            "eval_count": 1
        }))
        .unwrap();
        let r = parse_response(body, false);
        assert_eq!(r.usage.cache_read_tokens, None);
        assert_eq!(r.usage.cache_write_tokens, None);
    }

    #[test]
    fn parse_response_tool_call_synthesizes_id() {
        let body: OllamaResponse = serde_json::from_value(serde_json::json!({
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "fetch", "arguments": {"url": "x"}}},
                    {"function": {"name": "search", "arguments": {"q": "y"}}}
                ]
            },
            "done": true,
            "done_reason": "tool_calls",
            "prompt_eval_count": 1,
            "eval_count": 1
        }))
        .unwrap();
        let r = parse_response(body, false);
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        match &r.content[0] {
            ContentBlock::ToolUse(tc) => {
                assert_eq!(tc.id, "call-0");
                assert_eq!(tc.name, "fetch");
                assert_eq!(tc.input, serde_json::json!({"url": "x"}));
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        match &r.content[1] {
            ContentBlock::ToolUse(tc) => assert_eq!(tc.id, "call-1"),
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    // ── context_window / provider ──────────────────────────────────────────

    #[test]
    fn provider_info_uses_table() {
        let c = OllamaModelInterface::new("llama3.2");
        let p = c.provider();
        assert_eq!(p.name, "ollama");
        assert_eq!(p.model_id, "llama3.2");
        assert_eq!(p.context_window, 128_000);

        assert_eq!(
            OllamaModelInterface::context_window("qwen2.5-coder-7b"),
            128_000
        );
        assert_eq!(OllamaModelInterface::context_window("mistral"), 32_000);
        assert_eq!(OllamaModelInterface::context_window("gemma"), 8_192);
        assert_eq!(OllamaModelInterface::context_window("unknown"), 0);
    }

    // ── name matching for /api/tags ────────────────────────────────────────

    #[test]
    fn name_matches_handles_latest_tag() {
        assert!(name_matches("llama3.2:latest", "llama3.2"));
        assert!(name_matches("llama3.2", "llama3.2"));
        assert!(name_matches("llama3.2:3b", "llama3.2"));
        assert!(!name_matches("llama3.1", "llama3.2"));
    }

    // ── End-to-end wiremock call() ──────────────────────────────────────────

    async fn mock_tags_ok(server: &wiremock::MockServer, model: &str) {
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/tags"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"models": [{"name": format!("{model}:latest")}]}),
            ))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn call_against_mock_returns_response() {
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        let body = serde_json::json!({
            "message": {"role": "assistant", "content": "hello there"},
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 2
        });
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let r = client.call(req(vec![user("hi")])).await.unwrap();
        assert_eq!(
            r.content,
            vec![ContentBlock::Text {
                text: "hello there".into()
            }]
        );
        assert_eq!(r.usage.input_tokens, 5);
        assert_eq!(r.usage.output_tokens, 2);
    }

    #[tokio::test]
    async fn connection_refused_helpful_message() {
        // Point at a closed port: connect should fail immediately.
        let client = OllamaModelInterface::with_base_url("llama3.2", "http://127.0.0.1:1")
            .with_timeout(Duration::from_secs(2));
        let err = client.call(req(vec![user("hi")])).await.unwrap_err();
        match err {
            ModelError::ProviderError { code, message } => {
                assert_eq!(code, 0);
                assert!(message.contains("Ollama not running"), "msg: {message}");
            }
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn connection_error_does_not_retry() {
        // Verify by timing: a connect failure to a closed port returns
        // immediately, well under any plausible retry-backoff window.
        let client = OllamaModelInterface::with_base_url("llama3.2", "http://127.0.0.1:1")
            .with_timeout(Duration::from_secs(5));
        let start = std::time::Instant::now();
        let _ = client.call(req(vec![user("hi")])).await;
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "expected fail-fast (<500ms); took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn model_not_found_suggests_pull() {
        let server = wiremock::MockServer::start().await;
        // /api/tags returns a different model
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/tags"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"models": [{"name": "mistral:latest"}]})),
            )
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let err = client.call(req(vec![user("hi")])).await.unwrap_err();
        match err {
            ModelError::ProviderError { code, message } => {
                assert_eq!(code, 404);
                assert!(message.contains("ollama pull llama3.2"), "msg: {message}");
            }
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_404_maps_to_pull_suggestion() {
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(wiremock::ResponseTemplate::new(404).set_body_string(
                "{\"error\":\"model 'llama3.2' not found, try pulling it first\"}",
            ))
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let err = client.call(req(vec![user("hi")])).await.unwrap_err();
        match err {
            ModelError::ProviderError { code, message } => {
                assert_eq!(code, 404);
                assert!(message.contains("ollama pull llama3.2"), "msg: {message}");
            }
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn timeout_maps_to_timeout() {
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_delay(Duration::from_secs(2)))
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri())
            .with_timeout(Duration::from_millis(200));
        let err = client.call(req(vec![user("hi")])).await.unwrap_err();
        assert!(matches!(err, ModelError::Timeout), "got: {err:?}");
    }

    #[tokio::test]
    async fn model_check_cached_after_first_call() {
        let server = wiremock::MockServer::start().await;
        // /api/tags allowed AT MOST ONCE — second call would fail since no
        // additional matcher is registered (wiremock returns 404 on miss).
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/tags"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"models": [{"name": "llama3.2:latest"}]})),
            )
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "message": {"role": "assistant", "content": "ok"},
                    "done": true,
                    "done_reason": "stop",
                    "prompt_eval_count": 1,
                    "eval_count": 1
                })),
            )
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        client.call(req(vec![user("a")])).await.unwrap();
        client.call(req(vec![user("b")])).await.unwrap();
        // wiremock's `.expect(1)` is verified on drop.
    }

    // ── Streaming ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn streaming_emits_text_delta_then_stop() {
        let ndjson = concat!(
            "{\"message\":{\"role\":\"assistant\",\"content\":\"hello\"},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"content\":\" world\"},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":3,\"eval_count\":5}\n",
        );
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_string(ndjson)
                    .insert_header("content-type", "application/x-ndjson"),
            )
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let mut stream = client.call_streaming(req(vec![user("hi")])).await.unwrap();
        let mut events: Vec<StreamEvent> = vec![];
        while let Some(ev) = stream.next().await {
            events.push(ev.unwrap());
        }
        assert!(matches!(events[0], StreamEvent::MessageStart));
        let text_deltas: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text_deltas, vec!["hello", " world"]);
        match events.last().unwrap() {
            StreamEvent::MessageStop { usage, stop_reason } => {
                assert_eq!(usage.input_tokens, 3);
                assert_eq!(usage.output_tokens, 5);
                assert_eq!(*stop_reason, StopReason::EndTurn);
            }
            other => panic!("expected MessageStop, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_parses_ndjson_lines() {
        // Verify two JSON objects on the same line buffer (split mid-stream)
        // are both parsed correctly.
        let part1 = "{\"message\":{\"role\":\"assistant\",\"content\":\"ab\"},\"done\":false}\n{\"message\":{\"role\":\"assistant\",\"content\":\"cd\"},\"done\":false}\n";
        let part2 = "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":1,\"eval_count\":1}\n";
        let ndjson = format!("{part1}{part2}");
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(ndjson))
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let mut stream = client.call_streaming(req(vec![user("hi")])).await.unwrap();
        let mut deltas: Vec<String> = vec![];
        while let Some(ev) = stream.next().await {
            if let StreamEvent::ContentBlockDelta { delta, .. } = ev.unwrap() {
                deltas.push(delta);
            }
        }
        assert_eq!(deltas, vec!["ab", "cd"]);
    }

    #[tokio::test]
    async fn streaming_done_carries_usage() {
        let ndjson = "{\"message\":{\"role\":\"assistant\",\"content\":\"x\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":42,\"eval_count\":7}\n";
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(ndjson))
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let mut stream = client.call_streaming(req(vec![user("hi")])).await.unwrap();
        let mut final_usage: Option<TokenUsage> = None;
        while let Some(ev) = stream.next().await {
            if let StreamEvent::MessageStop { usage, .. } = ev.unwrap() {
                final_usage = Some(usage);
            }
        }
        let u = final_usage.unwrap();
        assert_eq!(u.input_tokens, 42);
        assert_eq!(u.output_tokens, 7);
    }

    #[tokio::test]
    async fn streaming_accumulates_tool_calls() {
        // Ollama returns the full arguments object per chunk (not partial
        // strings). Verify we emit one ToolUseDelta carrying the serialized
        // object and a MessageStop with stop_reason=ToolUse.
        let ndjson = concat!(
            "{\"message\":{\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"name\":\"fetch\",\"arguments\":{\"url\":\"x\"}}}]},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"tool_calls\",\"prompt_eval_count\":1,\"eval_count\":1}\n",
        );
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(ndjson))
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let mut stream = client.call_streaming(req(vec![user("hi")])).await.unwrap();
        let mut tool_jsons: Vec<String> = vec![];
        let mut final_stop = StopReason::EndTurn;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::ToolUseDelta { partial_json, .. } => tool_jsons.push(partial_json),
                StreamEvent::MessageStop { stop_reason, .. } => final_stop = stop_reason,
                _ => {}
            }
        }
        assert_eq!(tool_jsons.len(), 1);
        let parsed: serde_json::Value = serde_json::from_str(&tool_jsons[0]).unwrap();
        assert_eq!(parsed, serde_json::json!({"url": "x"}));
        assert_eq!(final_stop, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn streaming_keeps_multiple_tool_calls_distinct() {
        // A single response with three tool calls streams them in SEPARATE
        // chunks, each a one-element `tool_calls` array distinguished only by
        // `function.index`. Each call must land on its own stream index so its
        // argument JSON stays well-formed — keying off the array position would
        // collapse all three onto index 1 and concatenate their args into
        // invalid JSON (regression: example 03 saw `calculator(null)`).
        let ndjson = concat!(
            "{\"message\":{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_a\",\"function\":{\"index\":0,\"name\":\"calculator\",\"arguments\":{\"a\":\"144\",\"b\":\"12\",\"op\":\"/\"}}}]},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_b\",\"function\":{\"index\":1,\"name\":\"get_current_time\",\"arguments\":{}}}]},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_c\",\"function\":{\"index\":2,\"name\":\"reverse_string\",\"arguments\":{\"text\":\"harness\"}}}]},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"tool_calls\",\"prompt_eval_count\":1,\"eval_count\":1}\n",
        );
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(ndjson))
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let mut stream = client.call_streaming(req(vec![user("hi")])).await.unwrap();
        // Collect (index -> name) from starts and concatenated json per index.
        let mut names: std::collections::BTreeMap<u32, String> = Default::default();
        let mut jsons: std::collections::BTreeMap<u32, String> = Default::default();
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::ToolUseStart { index, name, .. } => {
                    names.insert(index, name);
                }
                StreamEvent::ToolUseDelta {
                    index,
                    partial_json,
                } => jsons.entry(index).or_default().push_str(&partial_json),
                _ => {}
            }
        }
        // Three distinct calls, each with a start frame and valid argument JSON.
        assert_eq!(names.len(), 3);
        assert_eq!(jsons.len(), 3);
        for (idx, json) in &jsons {
            let parsed: serde_json::Value = serde_json::from_str(json)
                .unwrap_or_else(|e| panic!("index {idx} json {json:?} did not parse: {e}"));
            assert!(
                parsed.is_object(),
                "index {idx} args not an object: {parsed}"
            );
        }
        let collected: Vec<&str> = names.values().map(String::as_str).collect();
        assert_eq!(
            collected,
            vec!["calculator", "get_current_time", "reverse_string"]
        );
    }

    #[tokio::test]
    async fn streaming_structured_buffers_json_then_reconstructs_tool_call() {
        // In structured mode the constrained JSON object arrives split across
        // multiple content chunks. The raw JSON must never be surfaced as answer
        // text; at `done` it is parsed into a reconstructable ToolUse. This is
        // the streaming analogue of the `<|python_tag|>`-leak fix: the tool call
        // is reconstructed from the constrained-JSON channel, never the native
        // tool_calls path.
        let ndjson = concat!(
            "{\"message\":{\"role\":\"assistant\",\"content\":\"{\\\"tool\\\":\\\"write_file\\\",\"},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"content\":\"\\\"arguments\\\":{\\\"path\\\":\\\"S.md\\\",\\\"content\\\":\\\"hi\\\"}}\"},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":1,\"eval_count\":1}\n",
        );
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        mock_show(
            &server,
            serde_json::json!({
                "model_info": {"llama.context_length": 128_000},
                "capabilities": ["tools"]
            }),
            None,
        )
        .await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(ndjson))
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let mut stream = client.call_streaming(structured_tool_req()).await.unwrap();

        let mut name: Option<String> = None;
        let mut json = String::new();
        let mut text_deltas: Vec<String> = vec![];
        let mut final_stop = StopReason::EndTurn;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::ToolUseStart { name: n, .. } => name = Some(n),
                StreamEvent::ToolUseDelta { partial_json, .. } => json.push_str(&partial_json),
                StreamEvent::ContentBlockDelta { delta, .. } => text_deltas.push(delta),
                StreamEvent::MessageStop { stop_reason, .. } => final_stop = stop_reason,
                _ => {}
            }
        }
        // Raw JSON was NOT surfaced as answer text.
        assert!(
            text_deltas.is_empty(),
            "raw JSON leaked as text: {text_deltas:?}"
        );
        // The tool call was reconstructed with the right name + valid arg JSON.
        assert_eq!(name.as_deref(), Some("write_file"));
        let parsed: serde_json::Value = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("tool args {json:?} did not parse: {e}"));
        assert_eq!(parsed, serde_json::json!({"path":"S.md","content":"hi"}));
        assert_eq!(final_stop, StopReason::ToolUse);
    }

    // ── count_tokens ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn count_tokens_uses_embed_endpoint() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/embed"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"prompt_eval_count": 123})),
            )
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let n = client
            .count_tokens(&req(vec![user("hello world")]))
            .await
            .unwrap();
        assert_eq!(n, 123);
    }

    #[tokio::test]
    async fn count_tokens_falls_back_to_heuristic() {
        let server = wiremock::MockServer::start().await;
        // Embed returns 500 — must fall back, never error.
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/embed"))
            .respond_with(wiremock::ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        // 40 chars of 'a' + 1 newline = 41 → 41/4 = 10
        let n = client
            .count_tokens(&req(vec![user(&"a".repeat(40))]))
            .await
            .unwrap();
        assert_eq!(n, 10);
    }

    // ── /api/show discovery + tool-capability guard ─────────────────────────

    fn tool_req() -> ModelRequest {
        let mut r = req(vec![user("use a tool")]);
        r.tools.push(ToolSchema {
            name: "search".into(),
            description: "search the web".into(),
            input_schema: serde_json::json!({"type":"object"}),
        });
        r
    }

    async fn mock_show(server: &wiremock::MockServer, body: serde_json::Value, times: Option<u64>) {
        let mut m = wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/show"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body));
        if let Some(n) = times {
            m = m.up_to_n_times(n).expect(n);
        }
        m.mount(server).await;
    }

    async fn mock_chat_ok(server: &wiremock::MockServer) {
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "message": {"role": "assistant", "content": "ok"},
                    "done": true,
                    "done_reason": "stop",
                    "prompt_eval_count": 1,
                    "eval_count": 1
                })),
            )
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn provider_reflects_discovered_context_window() {
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        mock_show(
            &server,
            serde_json::json!({
                "model_info": {"llama.context_length": 16_384},
                "capabilities": ["tools"]
            }),
            None,
        )
        .await;
        mock_chat_ok(&server).await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        // Before the probe runs, provider() falls back to the static table.
        assert_eq!(client.provider().context_window, 128_000);
        client.call(req(vec![user("hi")])).await.unwrap();
        // After the probe, provider() reflects the discovered value.
        assert_eq!(client.provider().context_window, 16_384);
    }

    #[tokio::test]
    async fn provider_falls_back_when_show_404s() {
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/show"))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .mount(&server)
            .await;
        mock_chat_ok(&server).await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        client.call(req(vec![user("hi")])).await.unwrap();
        assert_eq!(client.provider().context_window, 128_000);
    }

    #[tokio::test]
    async fn provider_falls_back_when_context_length_missing() {
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        // /api/show succeeds but has no *.context_length entry.
        mock_show(
            &server,
            serde_json::json!({"model_info": {"general.architecture": "llama"}, "capabilities": ["tools"]}),
            None,
        )
        .await;
        mock_chat_ok(&server).await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        client.call(req(vec![user("hi")])).await.unwrap();
        assert_eq!(client.provider().context_window, 128_000);
    }

    #[tokio::test]
    async fn tool_request_rejected_when_capability_absent() {
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "gemma").await;
        // capabilities lacks "tools"; no /api/chat mock — a call would 404.
        mock_show(
            &server,
            serde_json::json!({
                "model_info": {"gemma.context_length": 8_192},
                "capabilities": ["completion"]
            }),
            None,
        )
        .await;
        let client = OllamaModelInterface::with_base_url("gemma", server.uri());
        let err = client.call(tool_req()).await.unwrap_err();
        match err {
            ModelError::ProviderError { code, message } => {
                assert_eq!(code, 0);
                assert!(
                    message.contains("does not support tool calling"),
                    "msg: {message}"
                );
            }
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_request_proceeds_when_capability_present() {
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        mock_show(
            &server,
            serde_json::json!({
                "model_info": {"llama.context_length": 128_000},
                "capabilities": ["completion", "tools"]
            }),
            None,
        )
        .await;
        mock_chat_ok(&server).await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        // Proceeds to /api/chat and returns a normal response.
        let r = client.call(tool_req()).await.unwrap();
        assert_eq!(r.content, vec![ContentBlock::Text { text: "ok".into() }]);
    }

    #[tokio::test]
    async fn tool_request_rejected_when_capabilities_empty() {
        // `/api/show` returns an empty `capabilities` array. With the static
        // whitelist removed, empty capabilities fail closed — even for a model
        // id (`llama3.2`) that the old prefix table would have allowed.
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        // No /api/chat mock — a call would 404 if the guard let it through.
        mock_show(
            &server,
            serde_json::json!({
                "model_info": {"llama.context_length": 128_000},
                "capabilities": []
            }),
            None,
        )
        .await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let err = client.call(tool_req()).await.unwrap_err();
        match err {
            ModelError::ProviderError { code, message } => {
                assert_eq!(code, 0);
                assert!(
                    message.contains("does not support tool calling"),
                    "msg: {message}"
                );
            }
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_request_rejected_when_show_404s() {
        // `/api/show` 404s ⟹ empty ModelMeta ⟹ NOT tool-capable (fail closed).
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/show"))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .mount(&server)
            .await;
        // No /api/chat mock — a call would 404 if the guard let it through.
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        let err = client.call(tool_req()).await.unwrap_err();
        match err {
            ModelError::ProviderError { code, message } => {
                assert_eq!(code, 0);
                assert!(
                    message.contains("does not support tool calling"),
                    "msg: {message}"
                );
            }
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn show_fetched_at_most_once() {
        let server = wiremock::MockServer::start().await;
        mock_tags_ok(&server, "llama3.2").await;
        // /api/show expected exactly once across multiple call()s.
        mock_show(
            &server,
            serde_json::json!({
                "model_info": {"llama.context_length": 32_000},
                "capabilities": ["tools"]
            }),
            Some(1),
        )
        .await;
        mock_chat_ok(&server).await;
        let client = OllamaModelInterface::with_base_url("llama3.2", server.uri());
        client.call(req(vec![user("a")])).await.unwrap();
        client.call(req(vec![user("b")])).await.unwrap();
        // wiremock's `.expect(1)` is verified on drop.
    }

    // ── #[ignore]-tagged live API tests ────────────────────────────────────
    //
    // Run with: cargo test -p spore-core ollama_live -- --ignored
    // Requires a local ollama with `llama3.2` pulled.

    #[tokio::test]
    #[ignore = "live-API; needs local ollama with llama3.2"]
    async fn ollama_live_call() {
        let client = OllamaModelInterface::new("llama3.2");
        let r = client
            .call(req(vec![user("Reply with the word 'pong'.")]))
            .await
            .unwrap();
        assert!(r.usage.input_tokens > 0);
        assert!(r.usage.output_tokens > 0);
    }

    #[tokio::test]
    #[ignore = "live-API; needs local ollama with llama3.2"]
    async fn ollama_live_streaming() {
        let client = OllamaModelInterface::new("llama3.2");
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
        assert!(saw_stop);
    }

    #[tokio::test]
    #[ignore = "live-API; needs local ollama with llama3.2"]
    async fn ollama_live_tool_call() {
        let mut r = req(vec![user("Use the echo tool with text='hi'.")]);
        r.tools.push(ToolSchema {
            name: "echo".into(),
            description: "echoes the given text".into(),
            input_schema: serde_json::json!({
                "type":"object",
                "properties":{"text":{"type":"string"}},
                "required":["text"]
            }),
        });
        let client = OllamaModelInterface::new("llama3.2");
        let resp = client.call(r).await.unwrap();
        assert!(matches!(
            resp.stop_reason,
            StopReason::ToolUse | StopReason::EndTurn
        ));
    }
}
