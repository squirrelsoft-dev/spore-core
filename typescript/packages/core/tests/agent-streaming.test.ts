/**
 * Unit tests for delta-level streaming through the Agent (spore-core #103).
 *
 * Mirrors `rust/crates/spore-core/src/agent.rs#tests` (the #103 section):
 * raw model `StreamEvent`s are forwarded to the sink in order, deltas are
 * reassembled into a `ModelResponse`, and the SAME classification runs as
 * `turn` — including the documented limitation that streamed tool-use blocks
 * carry an EMPTY name and a synthesized `call_{index}` id.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  MockAgent,
  ModelAgent,
  ReplayModelInterface,
  classifyResponse,
  turnStreaming,
  type AgentStreamSink,
  type Context,
  type ModelResponse,
  type ProviderInfo,
  type RecordedExchange,
  type StreamEvent,
  type TurnResult,
} from "../src/index.js";

const provider: ProviderInfo = {
  name: "anthropic",
  model_id: "replay",
  context_window: 200_000,
};

function ctxUser(text: string): Context {
  return {
    messages: [{ role: "user", content: { type: "text", text } }],
    tools: [],
    params: { stop_sequences: [] },
  };
}

/** A ReplayModelInterface whose single recorded response is `resp`. */
function replayAgent(resp: ModelResponse): ModelAgent {
  const exchange: RecordedExchange = {
    request: { messages: [], tools: [], params: { stop_sequences: [] }, stream: true },
    response: resp,
    provider: "anthropic",
  };
  return new ModelAgent(AgentId.of("stream-agent"), new ReplayModelInterface([exchange], provider));
}

function collector(): { sink: AgentStreamSink; seen: StreamEvent[] } {
  const seen: StreamEvent[] = [];
  return { sink: (ev) => seen.push(ev), seen };
}

describe("Agent streaming — text + reasoning deltas (#103)", () => {
  it("forwards reasoning then text deltas and surfaces reasoning in TurnResult (Q4)", async () => {
    const agent = replayAgent({
      content: [
        { type: "thinking", text: "reasoning here" },
        { type: "text", text: "final text" },
      ],
      usage: { input_tokens: 3, output_tokens: 4 },
      stop_reason: "end_turn",
    });
    const { sink, seen } = collector();
    const result = await agent.turnStreaming(ctxUser("hi"), sink);

    expect(result.kind).toBe("final_response");
    if (result.kind === "final_response") {
      expect(result.content).toBe("final text");
      expect(result.reasoning).toBe("reasoning here");
    }

    // Raw model events, in order: message_start … thinking_delta …
    // content_block_delta … message_stop.
    expect(seen[0]).toEqual({ type: "message_start" });
    expect(seen.some((e) => e.type === "thinking_delta" && e.delta === "reasoning here")).toBe(
      true,
    );
    expect(seen.some((e) => e.type === "content_block_delta" && e.delta === "final text")).toBe(
      true,
    );
    expect(seen[seen.length - 1]!.type).toBe("message_stop");
  });

  it("concatenates ordered text deltas across two text blocks", async () => {
    const agent = replayAgent({
      content: [
        { type: "text", text: "foo" },
        { type: "text", text: "bar" },
      ],
      usage: { input_tokens: 1, output_tokens: 1 },
      stop_reason: "end_turn",
    });
    const { sink } = collector();
    const result = await agent.turnStreaming(ctxUser("hi"), sink);
    expect(result.kind === "final_response" && result.content).toBe("foobar");
  });
});

describe("Agent streaming — tool-use reassembly (#103)", () => {
  it("reassembles tool args; coarse ToolCall has accumulated args + EMPTY name + call_{index} id", async () => {
    const agent = replayAgent({
      content: [
        { type: "thinking", text: "let me think" },
        { type: "tool_use", id: "toolu_1", name: "lookup", input: { q: "rust" } },
      ],
      usage: { input_tokens: 7, output_tokens: 11 },
      stop_reason: "tool_use",
    });
    const { sink, seen } = collector();
    const result = await agent.turnStreaming(ctxUser("hi"), sink);

    expect(result.kind).toBe("tool_call_requested");
    if (result.kind === "tool_call_requested") {
      expect(result.calls).toHaveLength(1);
      expect(result.calls[0]!.input).toEqual({ q: "rust" });
      expect(result.reasoning).toBe("let me think");
      // Documented limitation: model stream drops the tool name/id; the
      // accumulator synthesizes call_{index} and an empty name. The tool_use
      // block is at index 1 (thinking is index 0).
      expect(result.calls[0]!.name).toBe("");
      expect(result.calls[0]!.id).toBe("call_1");
    }
    expect(seen.some((e) => e.type === "tool_use_delta" && e.partial_json.includes("rust"))).toBe(
      true,
    );
  });
});

describe("Agent streaming — parity + back-compat (#103)", () => {
  it("turn and turnStreaming classify identically for the same response", async () => {
    const resp: ModelResponse = {
      content: [{ type: "text", text: "same" }],
      usage: { input_tokens: 2, output_tokens: 2 },
      stop_reason: "end_turn",
    };
    const blocking = await replayAgent(resp).turn(ctxUser("x"));
    const streaming = await replayAgent(resp).turnStreaming(ctxUser("x"), () => {});
    expect(streaming).toEqual(blocking);
  });

  it("the turnStreaming() default helper ignores the sink and delegates to turn", async () => {
    const mock = new MockAgent(AgentId.of("mock"));
    mock.push({
      kind: "final_response",
      content: "done",
      usage: { input_tokens: 1, output_tokens: 1 },
    } satisfies TurnResult);
    let count = 0;
    const result = await turnStreaming(mock, ctxUser("hi"), () => {
      count += 1;
    });
    expect(result.kind === "final_response" && result.content).toBe("done");
    expect(count).toBe(0); // default path must not emit events
  });

  it("classifyResponse is the shared classification source of truth", () => {
    const r = classifyResponse({
      content: [{ type: "text", text: "hi" }],
      usage: { input_tokens: 1, output_tokens: 1 },
      stop_reason: "end_turn",
    });
    expect(r.kind).toBe("final_response");
  });

  it("a pre-#103 TurnResult JSON (no reasoning) round-trips with reasoning undefined", () => {
    const json = `{"kind":"final_response","content":"hi","usage":{"input_tokens":1,"output_tokens":1}}`;
    const parsed = JSON.parse(json) as TurnResult;
    expect(parsed.kind).toBe("final_response");
    expect(parsed.kind === "final_response" && parsed.reasoning).toBeUndefined();
  });

  it("omits reasoning from the wire when absent", () => {
    const r = classifyResponse({
      content: [{ type: "text", text: "hi" }],
      usage: { input_tokens: 1, output_tokens: 1 },
      stop_reason: "end_turn",
    });
    expect(JSON.stringify(r)).not.toContain("reasoning");
  });
});
