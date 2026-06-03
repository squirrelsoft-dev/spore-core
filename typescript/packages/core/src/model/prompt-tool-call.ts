/**
 * Prompt-based tool calling — an adaptive fallback for models that do not
 * reliably emit native tool calls.
 *
 * Some models (small local ones especially) respond in prose even when a tool
 * call is the right action. Rather than maintaining a list of known-bad models
 * or asking callers to wrap them manually, the harness discovers this at
 * runtime: native tool calling is tried first, and when a turn comes back as
 * prose while tools were advertised (see {@link detectProseResponse}), the
 * harness flips a session-scoped flag that activates
 * {@link PromptBasedToolCallModelInterface} for the rest of the run.
 *
 * ## Two wrappers
 *
 * - {@link PromptBasedToolCallModelInterface} — an *always-on* transparent
 *   wrapper. It injects a tool-definition block into the system prompt and
 *   parses `<tool_call>` markers out of the model's text response into native
 *   tool-use blocks. Construct it directly for advanced use.
 * - {@link AdaptiveToolCallModelInterface} — a *flag-gated* wrapper installed
 *   automatically by `HarnessBuilder.conversational`. While its shared flag is
 *   unset it delegates natively (byte-for-byte); once the harness sets the flag
 *   it behaves exactly like the always-on wrapper.
 *
 * Both share the free functions {@link injectToolPrompt} and
 * {@link parseProseResponse} so injection and parsing can never diverge between
 * them. Injection is idempotent — double-wrapping (e.g. an `Adaptive` around a
 * `PromptBased`) never appends the block twice.
 *
 * ## Streaming
 *
 * `callStreaming` buffers the full inner stream, parses it for markers, then
 * re-emits the reconstructed response as a stream. Streaming and marker parsing
 * do not compose cleanly; buffering is the accepted trade-off.
 */

import type { ModelInterface } from "./interface.js";
import type {
  ContentBlock,
  ModelRequest,
  ModelResponse,
  ProviderInfo,
  StopReason,
  StreamEvent,
  ToolSchema,
} from "./schemas.js";

/**
 * Shared mutable boolean holder. The idiomatic-TS analogue of Rust's
 * `Arc<AtomicBool>`: a small object captured by reference so the harness loop
 * and the {@link AdaptiveToolCallModelInterface} observe the same flag.
 */
export interface SharedFlag {
  value: boolean;
}

/** Construct a fresh, unset {@link SharedFlag}. */
export function newSharedFlag(): SharedFlag {
  return { value: false };
}

/**
 * Sentinel that marks an already-injected tool-prompt block. Used to make
 * {@link injectToolPrompt} idempotent.
 */
const TOOLS_BLOCK_OPEN = "<available_tools>";

// ============================================================================
// System-prompt injection
// ============================================================================

/**
 * Recursively sort object keys alphabetically at every nesting level, leaving
 * array element order untouched. Mirrors how Rust's `serde_json::Value`
 * (BTreeMap-backed, no `preserve_order`) and Go's `json.Marshal` of a map
 * serialize: object KEYS are emitted in sorted order, array ELEMENTS in
 * insertion order. Canonicalizing here keeps the `<input_schema>` JSON
 * byte-identical across languages even for multi-property schemas.
 */
function sortKeysDeep(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(sortKeysDeep);
  if (v !== null && typeof v === "object") {
    const obj = v as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(obj)
        .sort()
        .map((k) => [k, sortKeysDeep(obj[k])]),
    );
  }
  return v;
}

/**
 * Render the tool-definition + response-format block appended to the system
 * prompt when prompt-based tool calling is active. Byte-for-byte identical to
 * Rust's `build_tool_prompt` (and Go's): the `<input_schema>` JSON renders with
 * object keys sorted alphabetically at every level (see {@link sortKeysDeep}).
 */
export function buildToolPrompt(tools: ToolSchema[]): string {
  let s = "";
  s +=
    "You have access to the following tools. Use them when they would help " +
    "complete the task.\n\n";
  s += "<available_tools>\n";
  for (const tool of tools) {
    s += "<tool>\n";
    s += `  <name>${tool.name}</name>\n`;
    s += `  <description>${tool.description}</description>\n`;
    let schemaJson: string;
    try {
      schemaJson = JSON.stringify(sortKeysDeep(tool.input_schema));
      if (schemaJson === undefined) schemaJson = "{}";
    } catch {
      schemaJson = "{}";
    }
    s += `  <input_schema>${schemaJson}</input_schema>\n`;
    s += "</tool>\n";
  }
  s += "</available_tools>\n\n";
  s +=
    "When you want to use a tool, respond with ONLY the following format and " + "nothing else:\n";
  s +=
    '<tool_call>\n  <name>tool_name_here</name>\n  <input>{"key": "value"}</input>\n</tool_call>\n\n';
  s += "When you have a final answer that does not require a tool, respond normally in prose.";
  return s;
}

/**
 * Append the tool-definition block to a request's system prompt, in place.
 *
 * - No-op when the request advertises no tools (nothing to describe).
 * - Idempotent: if a system message already contains the {@link TOOLS_BLOCK_OPEN}
 *   sentinel, nothing is appended (so wrapping a wrapper does not double-inject).
 * - Appends to an existing leading `system` text message when present, otherwise
 *   inserts a new one at the front — never clobbering the caller's system prompt.
 */
export function injectToolPrompt(request: ModelRequest): void {
  if (request.tools.length === 0) {
    return;
  }
  const block = buildToolPrompt(request.tools);

  const first = request.messages[0];
  if (first != null && first.role === "system" && first.content.type === "text") {
    if (first.content.text.includes(TOOLS_BLOCK_OPEN)) {
      return; // already injected — idempotent.
    }
    first.content = { type: "text", text: `${first.content.text}\n\n${block}` };
    return;
  }

  request.messages.unshift({
    role: "system",
    content: { type: "text", text: block },
  });
}

// ============================================================================
// Response parsing
// ============================================================================

/**
 * `<tool_call>` marker regex. DOTALL via `[\s\S]`, global, non-greedy. Mirrors
 * Rust's `(?s)<tool_call>\s*<name>(.*?)</name>\s*<input>(.*?)</input>\s*</tool_call>`.
 */
const TOOL_CALL_RE =
  /<tool_call>\s*<name>([\s\S]*?)<\/name>\s*<input>([\s\S]*?)<\/input>\s*<\/tool_call>/g;

/**
 * Extract `(name, input)` pairs from `<tool_call>` markers in model text.
 *
 * A marker is `<tool_call><name>..</name><input>{json}</input></tool_call>`.
 * Markers whose `<input>` is not valid JSON are skipped (the caller decides what
 * to do when nothing parses). Supports multiple markers in one response.
 */
function extractToolCalls(text: string): Array<{ name: string; input: unknown }> {
  const out: Array<{ name: string; input: unknown }> = [];
  // Fresh regex state per call: `lastIndex` is stateful on a global regex.
  const re = new RegExp(TOOL_CALL_RE.source, TOOL_CALL_RE.flags);
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) != null) {
    const name = match[1]!.trim();
    const rawInput = match[2]!.trim();
    if (name.length === 0) {
      continue;
    }
    let input: unknown;
    try {
      input = JSON.parse(rawInput);
    } catch {
      // Malformed input JSON: skip this marker. If no marker parses, the whole
      // response falls through as prose (graceful degradation).
      continue;
    }
    out.push({ name, input });
  }
  return out;
}

/**
 * Rewrite a model response so `<tool_call>` markers in its text become native
 * tool-use blocks.
 *
 * - If the response already carries native tool-use blocks, it is returned
 *   unchanged (native tool calling succeeded — never second-guess it).
 * - Otherwise text blocks are scanned for markers. When at least one parses, the
 *   response's content becomes any `thinking` blocks followed by the synthesized
 *   tool-use blocks, and `stop_reason` becomes `tool_use`.
 * - When no marker parses, the response is returned unchanged (prose as-is).
 */
export function parseProseResponse(response: ModelResponse): ModelResponse {
  const hasNativeToolUse = response.content.some((b) => b.type === "tool_use");
  if (hasNativeToolUse) {
    return response;
  }

  const text = response.content
    .filter((b): b is Extract<ContentBlock, { type: "text" }> => b.type === "text")
    .map((b) => b.text)
    .join("");

  const parsed = extractToolCalls(text);
  if (parsed.length === 0) {
    return response; // no tool markers — genuine prose response.
  }

  // Preserve reasoning, replace text with synthesized tool-use blocks.
  const content: ContentBlock[] = response.content.filter((b) => b.type === "thinking");
  parsed.forEach(({ name, input }, i) => {
    content.push({ type: "tool_use", id: `ptc_call_${i}`, name, input });
  });

  return {
    content,
    usage: response.usage,
    stop_reason: "tool_use",
  };
}

// ============================================================================
// Prose detection (escalation heuristic)
// ============================================================================

/**
 * Curated action-intent phrases. Lower-cased substring match — conservative and
 * cheap. Each phrase strongly implies "I am about to use a tool". Order and
 * contents mirror Rust's `ACTION_PHRASES` verbatim.
 */
const ACTION_PHRASES = [
  "i'll use",
  "i will use",
  "i'll call",
  "i will call",
  "i'll run",
  "i will run",
  "let me use",
  "let me call",
  "let me run",
  "i need to use",
  "i need to call",
  "i should use",
  "i should call",
  "i'll invoke",
  "i will invoke",
  "using the",
  "i can use the",
  "i'm going to use",
  "i am going to use",
] as const;

/**
 * Conservative heuristic: did the model respond in prose when a tool call was
 * the expected next step?
 *
 * Returns the trimmed prose text only when **both**:
 * 1. tools were advertised this turn (`toolsAdvertised`), and
 * 2. the response text contains an explicit action-intent phrase.
 *
 * Otherwise returns `null`. The bias is deliberately toward false negatives: a
 * missed prose response costs one extra turn, but a false positive activates
 * prompt-based mode for a model that was simply giving a final answer.
 */
export function detectProseResponse(text: string, toolsAdvertised: boolean): string | null {
  if (!toolsAdvertised) {
    return null;
  }
  const trimmed = text.trim();
  if (trimmed.length === 0) {
    return null;
  }
  const lower = trimmed.toLowerCase();
  if (ACTION_PHRASES.some((p) => lower.includes(p))) {
    return trimmed;
  }
  return null;
}

/**
 * The corrective nudge appended as a user message when the harness escalates to
 * prompt-based tool calling. Byte-for-byte identical to Rust's `nudge`.
 */
export const PROMPT_TOOL_CALL_NUDGE =
  "You described an action but did not call a tool. Use the provided tool-call " +
  "format to actually invoke the tool.";

// ============================================================================
// Stream buffering helpers
// ============================================================================

/** Reassemble a {@link ModelResponse} from a buffered stream of events. */
class ResponseBuffer {
  private readonly blocks: Array<{ index: number; block: BufBlock }> = [];
  private usage: ModelResponse["usage"] = { input_tokens: 0, output_tokens: 0 };
  private stopReason: StopReason | undefined = undefined;

  private entry(index: number, make: () => BufBlock): BufBlock {
    const found = this.blocks.find((b) => b.index === index);
    if (found) return found.block;
    const block = make();
    this.blocks.push({ index, block });
    return block;
  }

  fold(event: StreamEvent): void {
    switch (event.type) {
      case "message_start":
        break;
      case "content_block_delta": {
        const b = this.entry(event.index, () => ({ kind: "text", text: "" }));
        if (b.kind === "text") b.text += event.delta;
        break;
      }
      case "thinking_delta": {
        const b = this.entry(event.index, () => ({ kind: "thinking", text: "" }));
        if (b.kind === "thinking") b.text += event.delta;
        break;
      }
      case "tool_use_start": {
        const b = this.entry(event.index, () => ({ kind: "tool", id: "", name: "", json: "" }));
        if (b.kind === "tool") {
          b.id = event.id;
          b.name = event.name;
        }
        break;
      }
      case "tool_use_delta": {
        const b = this.entry(event.index, () => ({ kind: "tool", id: "", name: "", json: "" }));
        if (b.kind === "tool") b.json += event.partial_json;
        break;
      }
      case "content_block_stop":
        break;
      case "message_stop":
        this.usage = event.usage;
        this.stopReason = event.stop_reason;
        break;
      default: {
        const _exhaustive: never = event;
        return _exhaustive;
      }
    }
  }

  intoResponse(): ModelResponse {
    const content: ContentBlock[] = this.blocks.map(({ index, block }) => {
      switch (block.kind) {
        case "text":
          return { type: "text", text: block.text };
        case "thinking":
          return { type: "thinking", text: block.text };
        case "tool": {
          let input: unknown;
          try {
            input = JSON.parse(block.json);
          } catch {
            input = null;
          }
          return {
            type: "tool_use",
            id: block.id === "" ? `call_${index}` : block.id,
            name: block.name,
            input,
          };
        }
        default: {
          const _exhaustive: never = block;
          return _exhaustive;
        }
      }
    });
    return {
      content,
      usage: this.usage,
      stop_reason: this.stopReason ?? "end_turn",
    };
  }
}

type BufBlock =
  | { kind: "text"; text: string }
  | { kind: "thinking"; text: string }
  | { kind: "tool"; id: string; name: string; json: string };

/** Re-emit a {@link ModelResponse} as a stream of events. Inverse of {@link ResponseBuffer}. */
async function* responseToStream(response: ModelResponse): AsyncIterable<StreamEvent> {
  yield { type: "message_start" };
  let index = 0;
  for (const block of response.content) {
    switch (block.type) {
      case "text":
        yield { type: "content_block_delta", index, delta: block.text };
        break;
      case "thinking":
        yield { type: "thinking_delta", index, delta: block.text };
        break;
      case "tool_use": {
        yield { type: "tool_use_start", index, id: block.id, name: block.name };
        let partialJson: string;
        try {
          partialJson = JSON.stringify(block.input);
          if (partialJson === undefined) partialJson = "{}";
        } catch {
          partialJson = "{}";
        }
        yield { type: "tool_use_delta", index, partial_json: partialJson };
        break;
      }
      default: {
        const _exhaustive: never = block;
        return _exhaustive;
      }
    }
    yield { type: "content_block_stop", index };
    index += 1;
  }
  yield { type: "message_stop", usage: response.usage, stop_reason: response.stop_reason };
}

/** Shared streaming path: inject, buffer the inner stream, parse, re-emit. */
async function* streamingPromptCall(
  inner: ModelInterface,
  request: ModelRequest,
  signal?: AbortSignal,
): AsyncIterable<StreamEvent> {
  injectToolPrompt(request);
  const buf = new ResponseBuffer();
  for await (const event of inner.callStreaming(request, signal)) {
    buf.fold(event);
  }
  const parsed = parseProseResponse(buf.intoResponse());
  yield* responseToStream(parsed);
}

// ============================================================================
// PromptBasedToolCallModelInterface — always-on wrapper
// ============================================================================

/**
 * Transparent, *always-on* prompt-based tool-calling wrapper around any
 * {@link ModelInterface}.
 *
 * Every call injects the tool-definition block into the system prompt and parses
 * `<tool_call>` markers from the response into native tool-use blocks.
 * `countTokens` and `provider` delegate to the inner model unchanged.
 */
export class PromptBasedToolCallModelInterface implements ModelInterface {
  constructor(private readonly inner: ModelInterface) {}

  async call(request: ModelRequest, signal?: AbortSignal): Promise<ModelResponse> {
    injectToolPrompt(request);
    const response = await this.inner.call(request, signal);
    return parseProseResponse(response);
  }

  callStreaming(request: ModelRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    return streamingPromptCall(this.inner, request, signal);
  }

  countTokens(request: ModelRequest, signal?: AbortSignal): Promise<number> {
    return this.inner.countTokens(request, signal);
  }

  provider(): ProviderInfo {
    return this.inner.provider();
  }
}

// ============================================================================
// AdaptiveToolCallModelInterface — flag-gated wrapper
// ============================================================================

/**
 * Flag-gated prompt-based wrapper. While `flag.value` is `false` it delegates to
 * the inner model byte-for-byte (native tool calling). Once the harness sets the
 * flag — on detecting a prose response where a tool call was expected — it
 * behaves exactly like {@link PromptBasedToolCallModelInterface} for the rest of
 * the run.
 *
 * Installed automatically by `HarnessBuilder.conversational`; the harness holds
 * the SAME {@link SharedFlag} instance and flips it from the run loop.
 */
export class AdaptiveToolCallModelInterface implements ModelInterface {
  constructor(
    private readonly inner: ModelInterface,
    private readonly flag: SharedFlag,
  ) {}

  /** `true` once prompt-based mode has been activated for the run. */
  isActive(): boolean {
    return this.flag.value;
  }

  async call(request: ModelRequest, signal?: AbortSignal): Promise<ModelResponse> {
    if (!this.flag.value) {
      return this.inner.call(request, signal);
    }
    injectToolPrompt(request);
    const response = await this.inner.call(request, signal);
    return parseProseResponse(response);
  }

  callStreaming(request: ModelRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    if (!this.flag.value) {
      return this.inner.callStreaming(request, signal);
    }
    return streamingPromptCall(this.inner, request, signal);
  }

  countTokens(request: ModelRequest, signal?: AbortSignal): Promise<number> {
    return this.inner.countTokens(request, signal);
  }

  provider(): ProviderInfo {
    return this.inner.provider();
  }
}
