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

use crate::harness::BoxFut;
use crate::model::{
    Content, ContentBlock, Message, ModelError, ModelInterface, ModelRequest, ModelResponse,
    ModelStream, ProviderInfo, Role, StopReason, StreamEvent, ToolCall, ToolSchema,
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

/// Append the tool-definition block to a request's system prompt, in place, and
/// clear the request's native `tools` array.
///
/// - No-op when the request advertises no tools (nothing to describe).
/// - Idempotent: if a system message already contains the
///   [`TOOLS_BLOCK_OPEN`] sentinel, the block is not appended a second time (so
///   wrapping a wrapper does not double-inject).
/// - Appends to an existing leading [`Role::System`] text message when present,
///   otherwise inserts a new one at the front — never clobbering the caller's
///   system prompt.
/// - **Clears `request.tools`.** Prompt-based tool calling SUPERSEDES the native
///   channel: the tools are now described in the system prompt and the model is
///   asked to emit `<tool_call>` markers as plain text. Leaving the native
///   `tools` array populated would make the provider ALSO render its own
///   (different) tool-call format from the chat template — two conflicting
///   formats in one request — and, for servers that run a native tool-call
///   parser (e.g. vLLM `qwen3_coder`), would route the model's output through
///   that parser instead of letting our markers pass through as text for
///   [`parse_prose_response`] to read. Clearing makes the prompt block the SOLE
///   tool interface.
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
            // Append unless already present (idempotent); either way the native
            // tools are cleared below so the invariant "block present ⇒ no native
            // tools" always holds.
            if !text.contains(TOOLS_BLOCK_OPEN) {
                text.push_str("\n\n");
                text.push_str(&block);
            }
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

    request.tools.clear();
}

// ============================================================================
// Response parsing
// ============================================================================

/// Extract `(name, input)` pairs from `<tool_call>` blocks in model text.
///
/// One LENIENT pass, because Qwen-family models do not commit to a single
/// encoding inside a `<tool_call>` — even when asked for the injected one they
/// interleave it with their trained `qwen3_coder`/Hermes format, and produce
/// hybrids that mix the two. Rather than match a fixed shape, this splits on
/// `<tool_call>` openings and, within each block, pulls:
///
/// - the **name** from `<name>NAME</name>` OR `<function=NAME>` (whichever is
///   present), and
/// - the **arguments** from a JSON `<input>{…}</input>` OR a set of
///   `<parameter=KEY>value</parameter>` tags (values taken as trimmed strings;
///   `coerce_tool_args` fixes non-string schema fields downstream on dispatch).
///
/// This accepts all three observed encodings with one code path:
/// - injected:  `<tool_call><name>X</name><input>{…}</input></tool_call>`
/// - qwen3_coder: `<tool_call><function=X><parameter=K>v</parameter>…</function></tool_call>`
/// - hybrid:    `<tool_call><name>X</name><parameter=K>v</parameter></function></tool_call>`
///
/// Malformed/dangling close tags (`</function>`, missing `</tool_call>`) are
/// ignored. A block whose `<input>` is present but not valid JSON, with no
/// `<parameter>` fallback, is skipped — so a genuinely prose response with no
/// real call still falls through as prose (graceful degradation).
fn extract_tool_calls(text: &str) -> Vec<(String, serde_json::Value)> {
    let mut out = Vec::new();
    // Each call begins at a `<tool_call>` opening; its body runs to the matching
    // `</tool_call>` when present (trailing prose is dropped), else to the next
    // opening / end of text.
    for raw in text.split("<tool_call>").skip(1) {
        let block = match raw.find("</tool_call>") {
            Some(i) => &raw[..i],
            None => raw,
        };
        if let Some(call) = parse_tool_call_block(block) {
            out.push(call);
        }
    }
    out
}

/// Parse ONE `<tool_call>` block's inner text into `(name, input)`, tolerant of
/// the injected, qwen3_coder, and hybrid encodings. See [`extract_tool_calls`].
fn parse_tool_call_block(block: &str) -> Option<(String, serde_json::Value)> {
    static NAME: OnceLock<regex::Regex> = OnceLock::new();
    static FUNC: OnceLock<regex::Regex> = OnceLock::new();
    static INPUT: OnceLock<regex::Regex> = OnceLock::new();
    static PARAM: OnceLock<regex::Regex> = OnceLock::new();
    let name_re = NAME.get_or_init(|| regex::Regex::new(r"(?s)<name>(.*?)</name>").unwrap());
    let func_re = FUNC.get_or_init(|| regex::Regex::new(r"<function=(.*?)>").unwrap());
    let input_re = INPUT.get_or_init(|| regex::Regex::new(r"(?s)<input>(.*?)</input>").unwrap());
    let param_re =
        PARAM.get_or_init(|| regex::Regex::new(r"(?s)<parameter=(.*?)>(.*?)</parameter>").unwrap());

    // Name: `<name>…</name>` (injected/hybrid) else `<function=…>` (qwen3_coder).
    let name = name_re
        .captures(block)
        .and_then(|c| c.get(1))
        .or_else(|| func_re.captures(block).and_then(|c| c.get(1)))
        .map(|m| m.as_str().trim().to_string())
        .filter(|n| !n.is_empty())?;

    // Args, preferring an explicit JSON `<input>` block.
    let has_input = input_re.is_match(block);
    if let Some(caps) = input_re.captures(block) {
        if let Some(m) = caps.get(1) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(m.as_str().trim()) {
                return Some((name, v));
            }
        }
    }
    // Else reconstruct from `<parameter=KEY>value</parameter>` pairs.
    let mut map = serde_json::Map::new();
    for p in param_re.captures_iter(block) {
        let (Some(k), Some(v)) = (p.get(1), p.get(2)) else {
            continue;
        };
        let key = k.as_str().trim();
        if key.is_empty() {
            continue;
        }
        map.insert(key.to_string(), serde_json::Value::String(v.as_str().trim().to_string()));
    }
    if !map.is_empty() {
        return Some((name, serde_json::Value::Object(map)));
    }
    // No usable args. If an `<input>` was present but unparseable (and no params
    // rescued it), treat the whole thing as prose. Otherwise it is a zero-arg
    // call — let it fire so a no-argument tool still dispatches.
    if has_input {
        return None;
    }
    Some((name, serde_json::Value::Object(serde_json::Map::new())))
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
// Stream re-emission helper
// ============================================================================

/// Re-emit a [`ModelResponse`] as a stream of [`StreamEvent`]s the harness
/// accumulator understands. Lets the prompt-based `call_streaming` path satisfy
/// the streaming interface from a single non-streaming inner call.
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
    inject_tool_prompt(&mut request);
    // Prompt-based mode buffers the WHOLE response before it can parse `<tool_call>`
    // markers, so per-token streaming offers nothing here — and it actively hurts on
    // some servers. With a server-side reasoning parser active (vLLM + a Qwen
    // reasoning model), a STREAMED request routes the model's entire output into the
    // reasoning channel: the model "thinks" the tool call and stops, and the injected
    // markers never reach the text content this function parses (observed:
    // `reasoning_len≈350, content_len=0`). The SAME request issued NON-streaming
    // returns the assistant message whole with the reasoning/content split applied
    // server-side, so the markers land in text content (observed: reliable marker in
    // `content`). Issue one non-streaming call, parse, then re-emit as a stream for
    // the harness accumulator.
    request.stream = false;
    let response = inner.call(request).await?;
    let parsed = parse_prose_response(response);
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
/// Generic over `M` so it monomorphizes onto a concrete model; inject
/// `Arc<ConcreteModel>` (or, since [`ModelInterface`] is now object-safe and
/// `Arc<dyn ModelInterface>` impls it, a boxed model) at construction, exactly
/// like [`RecordingModelInterface`](crate::model::RecordingModelInterface).
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
    fn call<'a>(
        &'a self,
        mut request: ModelRequest,
    ) -> BoxFut<'a, Result<ModelResponse, ModelError>> {
        Box::pin(async move {
        inject_tool_prompt(&mut request);
        let response = self.inner.call(request).await?;
        Ok(parse_prose_response(response))
        })
    }

    fn call_streaming<'a>(
        &'a self,
        request: ModelRequest,
    ) -> BoxFut<'a, Result<ModelStream, ModelError>> {
        Box::pin(async move { streaming_prompt_call(&self.inner, request).await })
    }

    fn count_tokens<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxFut<'a, Result<u32, ModelError>> {
        Box::pin(async move { self.inner.count_tokens(request).await })
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
    fn call<'a>(
        &'a self,
        mut request: ModelRequest,
    ) -> BoxFut<'a, Result<ModelResponse, ModelError>> {
        Box::pin(async move {
        if !self.flag.load(Ordering::Acquire) {
            return self.inner.call(request).await;
        }
        inject_tool_prompt(&mut request);
        let response = self.inner.call(request).await?;
        Ok(parse_prose_response(response))
        })
    }

    fn call_streaming<'a>(
        &'a self,
        request: ModelRequest,
    ) -> BoxFut<'a, Result<ModelStream, ModelError>> {
        Box::pin(async move {
        if !self.flag.load(Ordering::Acquire) {
            return self.inner.call_streaming(request).await;
        }
        streaming_prompt_call(&self.inner, request).await
        })
    }

    fn count_tokens<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxFut<'a, Result<u32, ModelError>> {
        Box::pin(async move { self.inner.count_tokens(request).await })
    }

    fn provider(&self) -> ProviderInfo {
        self.inner.provider()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mock::MockModelInterface;
    use crate::model::{Content, Message, ModelParams, Role, TokenUsage};

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

    #[test]
    fn injection_clears_native_tools() {
        // Prompt-based mode supersedes the native channel: after injecting the
        // prompt block, the native `tools` array must be empty so the provider's
        // chat template / server-side parser does not ALSO emit (and intercept) a
        // second, conflicting tool-call format. The tool stays described in the
        // system prompt block.
        let mut req = req_with_tools(Some("base"));
        assert_eq!(req.tools.len(), 1);
        inject_tool_prompt(&mut req);
        assert!(req.tools.is_empty(), "native tools must be cleared");
        let sys = match &req.messages[0].content {
            Content::Text { text } => text,
            _ => panic!("system message must be text"),
        };
        assert!(sys.contains("<name>calculator</name>"), "tool still described in prompt");
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

    // --- qwen3_coder / Hermes <function=..><parameter=..> format -------------

    #[test]
    fn parses_xml_function_format_single_param() {
        // Qwen emits its native trained format even when asked for <name>/<input>.
        let text = "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/App.jsx\n</parameter>\n</function>\n</tool_call>";
        let out = parse_prose_response(prose(text, StopReason::EndTurn));
        assert_eq!(out.stop_reason, StopReason::ToolUse);
        match &out.content[..] {
            [ContentBlock::ToolUse(tc)] => {
                assert_eq!(tc.name, "read_file");
                assert_eq!(tc.input, serde_json::json!({ "path": "src/App.jsx" }));
            }
            other => panic!("expected one tool-use block, got {other:?}"),
        }
    }

    #[test]
    fn parses_xml_function_format_multiline_params() {
        // edit_file with a multi-line new_string — the exact shape that broke
        // the live Qwen run before this parser was added.
        let text = "<tool_call>\n<function=edit_file>\n\
<parameter=path>\nsrc/App.jsx\n</parameter>\n\
<parameter=old_string>\nfunction App() {\n</parameter>\n\
<parameter=new_string>\n/**\n * Renders a todo app.\n */\nfunction App() {\n</parameter>\n\
</function>\n</tool_call>";
        let out = parse_prose_response(prose(text, StopReason::EndTurn));
        assert_eq!(out.stop_reason, StopReason::ToolUse);
        match &out.content[..] {
            [ContentBlock::ToolUse(tc)] => {
                assert_eq!(tc.name, "edit_file");
                assert_eq!(tc.input["path"], serde_json::json!("src/App.jsx"));
                assert_eq!(tc.input["old_string"], serde_json::json!("function App() {"));
                assert_eq!(
                    tc.input["new_string"],
                    serde_json::json!("/**\n * Renders a todo app.\n */\nfunction App() {")
                );
            }
            other => panic!("expected one tool-use block, got {other:?}"),
        }
    }

    #[test]
    fn parses_hybrid_name_plus_parameter_format() {
        // The exact mangle Qwen leaked to the REPL: <name> (injected) for the tool
        // name but <parameter=> + a dangling </function> (qwen3_coder) for args.
        let text = "<tool_call>\n  <name>list_dir</name>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>";
        let out = parse_prose_response(prose(text, StopReason::EndTurn));
        assert_eq!(out.stop_reason, StopReason::ToolUse);
        match &out.content[..] {
            [ContentBlock::ToolUse(tc)] => {
                assert_eq!(tc.name, "list_dir");
                assert_eq!(tc.input, serde_json::json!({ "path": "." }));
            }
            other => panic!("expected one tool-use block, got {other:?}"),
        }
    }

    #[test]
    fn parses_two_hybrid_calls_in_one_response() {
        let text = "<tool_call>\n  <name>list_dir</name>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>\n\
<tool_call>\n  <name>read_file</name>\n<parameter=path>\npackage.json\n</parameter>\n</function>\n</tool_call>";
        let out = parse_prose_response(prose(text, StopReason::EndTurn));
        let calls: Vec<_> = out
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse(tc) => Some((tc.name.clone(), tc.input.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls,
            vec![
                ("list_dir".to_string(), serde_json::json!({ "path": "." })),
                ("read_file".to_string(), serde_json::json!({ "path": "package.json" })),
            ]
        );
    }

    #[test]
    fn parses_zero_arg_call() {
        // A named call with no <input> and no <parameter> → zero-arg dispatch.
        let out = parse_prose_response(prose(
            "<tool_call><name>list_dir</name></tool_call>",
            StopReason::EndTurn,
        ));
        match &out.content[..] {
            [ContentBlock::ToolUse(tc)] => {
                assert_eq!(tc.name, "list_dir");
                assert_eq!(tc.input, serde_json::json!({}));
            }
            other => panic!("expected one tool-use block, got {other:?}"),
        }
    }

    #[test]
    fn marker_format_wins_when_both_present() {
        // A clean injected marker is never reinterpreted as the xml-function form.
        let text = "<tool_call><name>a</name><input>{\"x\":1}</input></tool_call>";
        let out = parse_prose_response(prose(text, StopReason::EndTurn));
        match &out.content[..] {
            [ContentBlock::ToolUse(tc)] => {
                assert_eq!(tc.name, "a");
                assert_eq!(tc.input, serde_json::json!({ "x": 1 }));
            }
            other => panic!("expected one tool-use block, got {other:?}"),
        }
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
