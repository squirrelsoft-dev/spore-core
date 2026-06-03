//! Prompt-based tool calling — an adaptive fallback for models that do not
//! reliably emit native tool calls.
//!
//! Some models (small local ones especially) respond in prose even when a tool
//! call is the right action. Rather than maintaining a list of known-bad models
//! or asking callers to wrap them manually, the harness discovers this at
//! runtime: native tool calling is tried first, and when a turn comes back as
//! prose while tools were advertised (see
//! [`detect_prose_response`](crate::tool_call_repair::detect_prose_response)),
//! the harness flips a session-scoped flag that activates
//! [`PromptBasedToolCallModelInterface`] for the rest of the run.
//!
//! ## Two wrappers
//!
//! - [`PromptBasedToolCallModelInterface`] — an *always-on* transparent wrapper.
//!   It injects a tool-definition block into the system prompt and parses
//!   `<tool_call>` markers out of the model's text response into native
//!   [`ToolCall`]s. Construct it directly for advanced use.
//! - [`AdaptiveToolCallModelInterface`] — a *flag-gated* wrapper installed
//!   automatically by [`HarnessBuilder::conversational`](crate::HarnessBuilder::conversational).
//!   While its shared flag is unset it delegates natively (byte-for-byte); once
//!   the harness sets the flag it behaves exactly like the always-on wrapper.
//!
//! Both share the free functions [`inject_tool_prompt`] and
//! [`parse_prose_response`] so injection and parsing can never diverge between
//! them. Injection is idempotent — double-wrapping (e.g. an `Adaptive` around a
//! `PromptBased`) never appends the block twice.
//!
//! ## Streaming
//!
//! `call_streaming` buffers the full inner stream, parses it for markers, then
//! re-emits the reconstructed response as a stream. Streaming and marker
//! parsing do not compose cleanly; buffering is the accepted trade-off.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;

use crate::model::{
    Content, ContentBlock, Message, ModelError, ModelInterface, ModelRequest, ModelResponse,
    ModelStream, ProviderInfo, Role, StopReason, StreamEvent, TokenUsage, ToolCall, ToolSchema,
};

/// Sentinel that marks an already-injected tool-prompt block. Used to make
/// [`inject_tool_prompt`] idempotent.
const TOOLS_BLOCK_OPEN: &str = "<available_tools>";

// ============================================================================
// System-prompt injection
// ============================================================================

/// Render the tool-definition + response-format block appended to the system
/// prompt when prompt-based tool calling is active.
fn build_tool_prompt(tools: &[ToolSchema]) -> String {
    let mut s = String::new();
    s.push_str(
        "You have access to the following tools. Use them when they would help \
complete the task.\n\n",
    );
    s.push_str("<available_tools>\n");
    for tool in tools {
        s.push_str("<tool>\n");
        s.push_str(&format!("  <name>{}</name>\n", tool.name));
        s.push_str(&format!(
            "  <description>{}</description>\n",
            tool.description
        ));
        let schema_json =
            serde_json::to_string(&tool.input_schema).unwrap_or_else(|_| "{}".to_string());
        s.push_str(&format!("  <input_schema>{schema_json}</input_schema>\n"));
        s.push_str("</tool>\n");
    }
    s.push_str("</available_tools>\n\n");
    s.push_str(
        "When you want to use a tool, respond with ONLY the following format and \
nothing else:\n",
    );
    s.push_str("<tool_call>\n  <name>tool_name_here</name>\n  <input>{\"key\": \"value\"}</input>\n</tool_call>\n\n");
    s.push_str(
        "When you have a final answer that does not require a tool, respond \
normally in prose.",
    );
    s
}

/// Append the tool-definition block to a request's system prompt, in place.
///
/// - No-op when the request advertises no tools (nothing to describe).
/// - Idempotent: if a system message already contains the
///   [`TOOLS_BLOCK_OPEN`] sentinel, nothing is appended (so wrapping a wrapper
///   does not double-inject).
/// - Appends to an existing leading [`Role::System`] text message when present,
///   otherwise inserts a new one at the front — never clobbering the caller's
///   system prompt.
pub fn inject_tool_prompt(request: &mut ModelRequest) {
    if request.tools.is_empty() {
        return;
    }
    let block = build_tool_prompt(&request.tools);

    match request.messages.first_mut() {
        Some(Message {
            role: Role::System,
            content: Content::Text { text },
        }) => {
            if text.contains(TOOLS_BLOCK_OPEN) {
                return; // already injected — idempotent.
            }
            text.push_str("\n\n");
            text.push_str(&block);
        }
        _ => {
            request.messages.insert(
                0,
                Message {
                    role: Role::System,
                    content: Content::Text { text: block },
                },
            );
        }
    }
}

// ============================================================================
// Response parsing
// ============================================================================

/// Extract `(name, input)` pairs from `<tool_call>` markers in model text.
///
/// A marker is `<tool_call><name>..</name><input>{json}</input></tool_call>`.
/// Markers whose `<input>` is not valid JSON are skipped (the caller decides
/// what to do when nothing parses). Supports multiple markers in one response.
fn extract_tool_calls(text: &str) -> Vec<(String, serde_json::Value)> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(
            r"(?s)<tool_call>\s*<name>(.*?)</name>\s*<input>(.*?)</input>\s*</tool_call>",
        )
        .expect("tool-call marker regex is valid")
    });

    let mut out = Vec::new();
    for caps in re.captures_iter(text) {
        let name = caps.get(1).map(|m| m.as_str().trim()).unwrap_or_default();
        let raw_input = caps.get(2).map(|m| m.as_str().trim()).unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(raw_input) {
            Ok(input) => out.push((name.to_string(), input)),
            // Malformed input JSON: skip this marker. If no marker parses, the
            // whole response falls through as prose (graceful degradation).
            Err(_) => continue,
        }
    }
    out
}

/// Rewrite a model response so `<tool_call>` markers in its text become native
/// [`ContentBlock::ToolUse`] blocks.
///
/// - If the response already carries native tool-use blocks, it is returned
///   unchanged (native tool calling succeeded — never second-guess it).
/// - Otherwise text blocks are scanned for markers. When at least one parses,
///   the response's content becomes any `Thinking` blocks followed by the
///   synthesized tool-use blocks, and `stop_reason` becomes
///   [`StopReason::ToolUse`].
/// - When no marker parses, the response is returned unchanged (prose as-is).
pub fn parse_prose_response(response: ModelResponse) -> ModelResponse {
    let has_native_tool_use = response
        .content
        .iter()
        .any(|b| matches!(b, ContentBlock::ToolUse(_)));
    if has_native_tool_use {
        return response;
    }

    let text: String = response
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");

    let parsed = extract_tool_calls(&text);
    if parsed.is_empty() {
        return response; // no tool markers — genuine prose response.
    }

    // Preserve reasoning, replace text with synthesized tool-use blocks.
    let mut content: Vec<ContentBlock> = response
        .content
        .iter()
        .filter(|b| matches!(b, ContentBlock::Thinking { .. }))
        .cloned()
        .collect();
    for (i, (name, input)) in parsed.into_iter().enumerate() {
        content.push(ContentBlock::ToolUse(ToolCall {
            id: format!("ptc_call_{i}"),
            name,
            input,
        }));
    }

    ModelResponse {
        content,
        usage: response.usage,
        stop_reason: StopReason::ToolUse,
    }
}

// ============================================================================
// Stream buffering helpers
// ============================================================================

/// Reassemble a [`ModelResponse`] from a buffered stream of [`StreamEvent`]s.
/// Mirrors the agent's accumulator, kept local so this module owns its own
/// buffering for the parse-then-re-emit path.
#[derive(Default)]
struct ResponseBuffer {
    blocks: Vec<(u32, BufBlock)>,
    usage: TokenUsage,
    stop_reason: Option<StopReason>,
}

enum BufBlock {
    Text(String),
    Thinking(String),
    Tool {
        id: String,
        name: String,
        json: String,
    },
}

impl ResponseBuffer {
    fn entry(&mut self, index: u32, make: impl FnOnce() -> BufBlock) -> &mut BufBlock {
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
                if let BufBlock::Text(s) = self.entry(index, || BufBlock::Text(String::new())) {
                    s.push_str(&delta);
                }
            }
            StreamEvent::ThinkingDelta { index, delta } => {
                if let BufBlock::Thinking(s) =
                    self.entry(index, || BufBlock::Thinking(String::new()))
                {
                    s.push_str(&delta);
                }
            }
            StreamEvent::ToolUseStart { index, id, name } => {
                if let BufBlock::Tool {
                    id: bid,
                    name: bname,
                    ..
                } = self.entry(index, || BufBlock::Tool {
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
                if let BufBlock::Tool { json, .. } = self.entry(index, || BufBlock::Tool {
                    id: String::new(),
                    name: String::new(),
                    json: String::new(),
                }) {
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
        let content = self
            .blocks
            .into_iter()
            .map(|(index, block)| match block {
                BufBlock::Text(text) => ContentBlock::Text { text },
                BufBlock::Thinking(text) => ContentBlock::Thinking { text },
                BufBlock::Tool { id, name, json } => {
                    let input = serde_json::from_str(&json).unwrap_or(serde_json::Value::Null);
                    ContentBlock::ToolUse(ToolCall {
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
            stop_reason: self.stop_reason.unwrap_or(StopReason::EndTurn),
        }
    }
}

/// Re-emit a [`ModelResponse`] as a stream of [`StreamEvent`]s the harness
/// accumulator understands. The inverse of [`ResponseBuffer`].
fn response_to_stream(response: ModelResponse) -> ModelStream {
    let mut events: Vec<Result<StreamEvent, ModelError>> = vec![Ok(StreamEvent::MessageStart)];
    for (index, block) in response.content.into_iter().enumerate() {
        let index = index as u32;
        match block {
            ContentBlock::Text { text } => {
                events.push(Ok(StreamEvent::ContentBlockDelta { index, delta: text }));
            }
            ContentBlock::Thinking { text } => {
                events.push(Ok(StreamEvent::ThinkingDelta { index, delta: text }));
            }
            ContentBlock::ToolUse(tc) => {
                events.push(Ok(StreamEvent::ToolUseStart {
                    index,
                    id: tc.id,
                    name: tc.name,
                }));
                let partial_json = serde_json::to_string(&tc.input).unwrap_or_else(|_| "{}".into());
                events.push(Ok(StreamEvent::ToolUseDelta {
                    index,
                    partial_json,
                }));
            }
        }
        events.push(Ok(StreamEvent::ContentBlockStop { index }));
    }
    events.push(Ok(StreamEvent::MessageStop {
        usage: response.usage,
        stop_reason: response.stop_reason,
    }));
    Box::pin(futures_util::stream::iter(events))
}

/// Shared streaming path: inject, buffer the inner stream, parse, re-emit.
async fn streaming_prompt_call<M: ModelInterface + 'static>(
    inner: &Arc<M>,
    mut request: ModelRequest,
) -> Result<ModelStream, ModelError> {
    use futures_util::StreamExt;
    inject_tool_prompt(&mut request);
    let mut stream = inner.call_streaming(request).await?;
    let mut buf = ResponseBuffer::default();
    while let Some(item) = stream.next().await {
        buf.fold(item?);
    }
    let parsed = parse_prose_response(buf.into_response());
    Ok(response_to_stream(parsed))
}

// ============================================================================
// PromptBasedToolCallModelInterface — always-on wrapper
// ============================================================================

/// Transparent, *always-on* prompt-based tool-calling wrapper around any
/// [`ModelInterface`].
///
/// Every call injects the tool-definition block into the system prompt and
/// parses `<tool_call>` markers from the response into native [`ToolCall`]s.
/// `count_tokens` and `provider` delegate to the inner model unchanged.
///
/// Generic over `M` because [`ModelInterface`] is not dyn-compatible (RPITIT via
/// `trait_variant`); inject `Arc<ConcreteModel>` at construction, exactly like
/// [`RecordingModelInterface`](crate::model::RecordingModelInterface).
pub struct PromptBasedToolCallModelInterface<M: ModelInterface + 'static> {
    inner: Arc<M>,
}

impl<M: ModelInterface + 'static> PromptBasedToolCallModelInterface<M> {
    /// Wrap a model so it always uses prompt-based tool calling.
    pub fn new(inner: M) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Wrap an already-`Arc`'d model (shares ownership with other holders, e.g.
    /// a context manager that needs the same inner model).
    pub fn from_arc(inner: Arc<M>) -> Self {
        Self { inner }
    }

    /// Borrow the wrapped model.
    pub fn inner(&self) -> &Arc<M> {
        &self.inner
    }
}

impl<M: ModelInterface + 'static> ModelInterface for PromptBasedToolCallModelInterface<M> {
    async fn call(&self, mut request: ModelRequest) -> Result<ModelResponse, ModelError> {
        inject_tool_prompt(&mut request);
        let response = self.inner.call(request).await?;
        Ok(parse_prose_response(response))
    }

    async fn call_streaming(&self, request: ModelRequest) -> Result<ModelStream, ModelError> {
        streaming_prompt_call(&self.inner, request).await
    }

    async fn count_tokens(&self, request: &ModelRequest) -> Result<u32, ModelError> {
        self.inner.count_tokens(request).await
    }

    fn provider(&self) -> ProviderInfo {
        self.inner.provider()
    }
}

// ============================================================================
// AdaptiveToolCallModelInterface — flag-gated wrapper
// ============================================================================

/// Flag-gated prompt-based wrapper. While `flag` is `false` it delegates to the
/// inner model byte-for-byte (native tool calling). Once the harness sets `flag`
/// — on detecting a prose response where a tool call was expected — it behaves
/// exactly like [`PromptBasedToolCallModelInterface`] for the rest of the run.
///
/// Installed automatically by
/// [`HarnessBuilder::conversational`](crate::HarnessBuilder::conversational); the
/// harness holds a clone of the same `Arc<AtomicBool>` and flips it from the run
/// loop. Not normally constructed directly.
pub struct AdaptiveToolCallModelInterface<M: ModelInterface + 'static> {
    inner: Arc<M>,
    flag: Arc<AtomicBool>,
}

impl<M: ModelInterface + 'static> AdaptiveToolCallModelInterface<M> {
    /// Wrap `inner`, sharing `flag` with the harness loop.
    pub fn new(inner: Arc<M>, flag: Arc<AtomicBool>) -> Self {
        Self { inner, flag }
    }

    /// `true` once prompt-based mode has been activated for the run.
    pub fn is_active(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

impl<M: ModelInterface + 'static> ModelInterface for AdaptiveToolCallModelInterface<M> {
    async fn call(&self, mut request: ModelRequest) -> Result<ModelResponse, ModelError> {
        if !self.flag.load(Ordering::Acquire) {
            return self.inner.call(request).await;
        }
        inject_tool_prompt(&mut request);
        let response = self.inner.call(request).await?;
        Ok(parse_prose_response(response))
    }

    async fn call_streaming(&self, request: ModelRequest) -> Result<ModelStream, ModelError> {
        if !self.flag.load(Ordering::Acquire) {
            return self.inner.call_streaming(request).await;
        }
        streaming_prompt_call(&self.inner, request).await
    }

    async fn count_tokens(&self, request: &ModelRequest) -> Result<u32, ModelError> {
        self.inner.count_tokens(request).await
    }

    fn provider(&self) -> ProviderInfo {
        self.inner.provider()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mock::MockModelInterface;
    use crate::model::{Content, Message, ModelParams, Role};

    fn provider() -> ProviderInfo {
        ProviderInfo {
            name: "test".into(),
            model_id: "test-1".into(),
            context_window: 4096,
        }
    }

    fn tool_schema() -> ToolSchema {
        ToolSchema {
            name: "calculator".into(),
            description: "evaluate math".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "expression": { "type": "string" } },
                "required": ["expression"],
            }),
        }
    }

    fn req_with_tools(system: Option<&str>) -> ModelRequest {
        let mut messages = Vec::new();
        if let Some(s) = system {
            messages.push(Message {
                role: Role::System,
                content: Content::Text { text: s.into() },
            });
        }
        messages.push(Message {
            role: Role::User,
            content: Content::Text {
                text: "what is 2+2?".into(),
            },
        });
        ModelRequest {
            messages,
            tools: vec![tool_schema()],
            params: ModelParams::default(),
            stream: false,
        }
    }

    fn usage() -> TokenUsage {
        TokenUsage {
            input_tokens: 1,
            output_tokens: 1,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }
    }

    fn prose(text: &str, stop: StopReason) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            usage: usage(),
            stop_reason: stop,
        }
    }

    // --- injection -----------------------------------------------------------

    #[test]
    fn injection_appends_to_existing_system_prompt() {
        let mut req = req_with_tools(Some("You are a helpful assistant."));
        inject_tool_prompt(&mut req);
        let sys = match &req.messages[0].content {
            Content::Text { text } => text,
            _ => panic!("system message must be text"),
        };
        assert!(sys.starts_with("You are a helpful assistant."));
        assert!(sys.contains("<available_tools>"));
        assert!(sys.contains("<name>calculator</name>"));
        // User message preserved after the system message.
        assert!(matches!(req.messages[1].role, Role::User));
    }

    #[test]
    fn injection_inserts_system_prompt_when_absent() {
        let mut req = req_with_tools(None);
        inject_tool_prompt(&mut req);
        assert!(matches!(req.messages[0].role, Role::System));
        let sys = match &req.messages[0].content {
            Content::Text { text } => text,
            _ => panic!(),
        };
        assert!(sys.contains("<available_tools>"));
    }

    #[test]
    fn injection_is_idempotent() {
        let mut req = req_with_tools(Some("base"));
        inject_tool_prompt(&mut req);
        let once = match &req.messages[0].content {
            Content::Text { text } => text.clone(),
            _ => panic!(),
        };
        inject_tool_prompt(&mut req);
        let twice = match &req.messages[0].content {
            Content::Text { text } => text.clone(),
            _ => panic!(),
        };
        assert_eq!(once, twice, "second injection must be a no-op");
    }

    #[test]
    fn injection_noop_without_tools() {
        let mut req = req_with_tools(Some("base"));
        req.tools.clear();
        let before = req.messages.clone();
        inject_tool_prompt(&mut req);
        assert_eq!(req.messages, before);
    }

    // --- parsing -------------------------------------------------------------

    #[test]
    fn parses_single_tool_call_marker() {
        let resp = prose(
            "<tool_call><name>calculator</name><input>{\"expression\": \"2+2\"}</input></tool_call>",
            StopReason::EndTurn,
        );
        let out = parse_prose_response(resp);
        assert_eq!(out.stop_reason, StopReason::ToolUse);
        match &out.content[..] {
            [ContentBlock::ToolUse(tc)] => {
                assert_eq!(tc.name, "calculator");
                assert_eq!(tc.input, serde_json::json!({"expression": "2+2"}));
            }
            other => panic!("expected one tool-use block, got {other:?}"),
        }
    }

    #[test]
    fn parses_multiple_tool_call_markers() {
        let text = "<tool_call><name>a</name><input>{\"x\":1}</input></tool_call>\n\
                    some chatter\n\
                    <tool_call><name>b</name><input>{\"y\":2}</input></tool_call>";
        let out = parse_prose_response(prose(text, StopReason::EndTurn));
        let names: Vec<_> = out
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse(tc) => Some(tc.name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(out.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn malformed_input_json_falls_through_as_prose() {
        let resp = prose(
            "<tool_call><name>calculator</name><input>{not valid json}</input></tool_call>",
            StopReason::EndTurn,
        );
        let out = parse_prose_response(resp.clone());
        // No valid markers → unchanged prose response.
        assert_eq!(out.stop_reason, StopReason::EndTurn);
        assert!(matches!(out.content[0], ContentBlock::Text { .. }));
    }

    #[test]
    fn plain_prose_returned_as_is() {
        let resp = prose("The answer is 4.", StopReason::EndTurn);
        let out = parse_prose_response(resp.clone());
        assert_eq!(out, resp);
    }

    #[test]
    fn native_tool_use_left_untouched() {
        let resp = ModelResponse {
            content: vec![ContentBlock::ToolUse(ToolCall {
                id: "native".into(),
                name: "calculator".into(),
                input: serde_json::json!({"expression": "1"}),
            })],
            usage: usage(),
            stop_reason: StopReason::ToolUse,
        };
        let out = parse_prose_response(resp.clone());
        assert_eq!(out, resp);
    }

    #[test]
    fn thinking_blocks_preserved_alongside_synthesized_calls() {
        let resp = ModelResponse {
            content: vec![
                ContentBlock::Thinking {
                    text: "reasoning".into(),
                },
                ContentBlock::Text {
                    text: "<tool_call><name>t</name><input>{}</input></tool_call>".into(),
                },
            ],
            usage: usage(),
            stop_reason: StopReason::EndTurn,
        };
        let out = parse_prose_response(resp);
        assert!(matches!(out.content[0], ContentBlock::Thinking { .. }));
        assert!(matches!(out.content[1], ContentBlock::ToolUse(_)));
    }

    // --- always-on wrapper ---------------------------------------------------

    #[tokio::test]
    async fn always_on_wrapper_injects_and_parses() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(prose(
            "<tool_call><name>calculator</name><input>{\"expression\":\"2+2\"}</input></tool_call>",
            StopReason::EndTurn,
        )));
        let wrapper = PromptBasedToolCallModelInterface::from_arc(m.clone());
        let resp = wrapper.call(req_with_tools(Some("base"))).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert!(matches!(resp.content[0], ContentBlock::ToolUse(_)));
    }

    // --- adaptive wrapper ----------------------------------------------------

    #[tokio::test]
    async fn adaptive_wrapper_delegates_natively_when_flag_unset() {
        let m = Arc::new(MockModelInterface::new(provider()));
        // A prose response containing a marker — but the flag is OFF, so the
        // wrapper must NOT parse it; it delegates verbatim.
        m.push_response(Ok(prose(
            "<tool_call><name>x</name><input>{}</input></tool_call>",
            StopReason::EndTurn,
        )));
        let flag = Arc::new(AtomicBool::new(false));
        let wrapper = AdaptiveToolCallModelInterface::new(m.clone(), flag.clone());
        let resp = wrapper.call(req_with_tools(Some("base"))).await.unwrap();
        // Untouched: still a text block, stop_reason still EndTurn.
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert!(matches!(resp.content[0], ContentBlock::Text { .. }));
    }

    #[tokio::test]
    async fn adaptive_wrapper_parses_when_flag_set() {
        let m = Arc::new(MockModelInterface::new(provider()));
        m.push_response(Ok(prose(
            "<tool_call><name>x</name><input>{\"k\":1}</input></tool_call>",
            StopReason::EndTurn,
        )));
        let flag = Arc::new(AtomicBool::new(true));
        let wrapper = AdaptiveToolCallModelInterface::new(m.clone(), flag.clone());
        let resp = wrapper.call(req_with_tools(Some("base"))).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        match &resp.content[0] {
            ContentBlock::ToolUse(tc) => {
                assert_eq!(tc.name, "x");
                assert_eq!(tc.input, serde_json::json!({"k": 1}));
            }
            other => panic!("expected tool use, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn adaptive_wrapper_provider_delegates() {
        let m = Arc::new(MockModelInterface::new(provider()));
        let flag = Arc::new(AtomicBool::new(false));
        let wrapper = AdaptiveToolCallModelInterface::new(m, flag);
        assert_eq!(wrapper.provider().model_id, "test-1");
    }
}
