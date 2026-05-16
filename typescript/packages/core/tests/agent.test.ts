/**
 * Unit tests for the `Agent` component (spore-core issue #2).
 *
 * Mirrors `rust/crates/spore-core/src/agent.rs#tests` — same rules, same
 * verdicts, parallel structure.
 */

import { describe, expect, it } from "vitest";
import {
  AgentId,
  AgentModelError,
  EmptyResponse,
  MalformedToolCall,
  MockModelInterface,
  ModelAgent,
  ReplayModelInterface,
  Timeout,
  type Context,
  type ContentBlock,
  type ModelResponse,
  type ProviderInfo,
  type StopReason,
  type TokenUsage,
  type ToolCall,
} from "../src/index.js";

const provider: ProviderInfo = {
  name: "test",
  model_id: "test-1",
  context_window: 1000,
};

function usage(inT: number, outT: number): TokenUsage {
  return {
    input_tokens: inT,
    output_tokens: outT,
    cache_read_tokens: null,
    cache_write_tokens: null,
  };
}

function ctxUser(text: string): Context {
  return {
    messages: [{ role: "user", content: { type: "text", text } }],
    tools: [],
    params: { stop_sequences: [] },
  };
}

function textResp(text: string, stop: StopReason = "end_turn"): ModelResponse {
  return {
    content: [{ type: "text", text }],
    usage: usage(3, 4),
    stop_reason: stop,
  };
}

function toolResp(calls: ToolCall[]): ModelResponse {
  const content: ContentBlock[] = calls.map((c) => ({
    type: "tool_use" as const,
    id: c.id,
    name: c.name,
    input: c.input,
  }));
  return {
    content,
    usage: usage(5, 6),
    stop_reason: "tool_use",
  };
}

function makeAgent(): { agent: ModelAgent; model: MockModelInterface } {
  const model = new MockModelInterface(provider);
  const agent = new ModelAgent(AgentId.of("coding-agent"), model);
  return { agent, model };
}

describe("Agent — single-turn classification", () => {
  it("rule: one turn = exactly one model call", async () => {
    const { agent, model } = makeAgent();
    model.pushResponse(textResp("ok"));
    await agent.turn(ctxUser("hi"));
    expect(model.callCount).toBe(1);
  });

  it("classifies stop_reason=end_turn with text as FinalResponse", async () => {
    const { agent, model } = makeAgent();
    model.pushResponse(textResp("hello world"));
    const result = await agent.turn(ctxUser("hi"));
    expect(result.kind).toBe("final_response");
    if (result.kind === "final_response") {
      expect(result.content).toBe("hello world");
      expect(result.usage.input_tokens).toBe(3);
      expect(result.usage.output_tokens).toBe(4);
    }
  });

  it("classifies stop_reason=tool_use as ToolCallRequested", async () => {
    const { agent, model } = makeAgent();
    model.pushResponse(toolResp([{ id: "call_1", name: "read_file", input: { path: "/x" } }]));
    const result = await agent.turn(ctxUser("read /x"));
    expect(result.kind).toBe("tool_call_requested");
    if (result.kind === "tool_call_requested") {
      expect(result.calls).toHaveLength(1);
      expect(result.calls[0]!.id).toBe("call_1");
      expect(result.calls[0]!.name).toBe("read_file");
      expect(result.usage.input_tokens).toBe(5);
    }
  });

  it("ToolCallRequested may carry multiple parallel tool calls", async () => {
    const { agent, model } = makeAgent();
    model.pushResponse(
      toolResp([
        { id: "a", name: "read_file", input: { path: "/a" } },
        { id: "b", name: "read_file", input: { path: "/b" } },
      ]),
    );
    const result = await agent.turn(ctxUser("read both"));
    expect(result.kind).toBe("tool_call_requested");
    if (result.kind === "tool_call_requested") {
      expect(result.calls).toHaveLength(2);
      expect(result.calls[0]!.id).toBe("a");
      expect(result.calls[1]!.id).toBe("b");
    }
  });

  it("returns EmptyResponse when model returns no content blocks", async () => {
    const { agent, model } = makeAgent();
    model.pushResponse({
      content: [],
      usage: usage(1, 0),
      stop_reason: "end_turn",
    });
    const result = await agent.turn(ctxUser("?"));
    expect(result.kind).toBe("error");
    if (result.kind === "error") {
      expect(result.error).toBeInstanceOf(EmptyResponse);
      expect(result.error.kind).toBe("empty_response");
      expect(result.usage?.input_tokens).toBe(1);
    }
  });

  it("discards thinking blocks — thinking-only stays EmptyResponse", async () => {
    const { agent, model } = makeAgent();
    model.pushResponse({
      content: [{ type: "thinking", text: "musing" }],
      usage: usage(1, 2),
      stop_reason: "end_turn",
    });
    const result = await agent.turn(ctxUser("?"));
    expect(result.kind).toBe("error");
    if (result.kind === "error") {
      expect(result.error.kind).toBe("empty_response");
    }
  });

  it("surfaces ModelError wrapped in AgentError.ModelError with usage=null", async () => {
    const { agent, model } = makeAgent();
    model.pushError(new Timeout());
    const result = await agent.turn(ctxUser("hi"));
    expect(result.kind).toBe("error");
    if (result.kind === "error") {
      expect(result.error).toBeInstanceOf(AgentModelError);
      expect(result.error.kind).toBe("model_error");
      expect(result.usage).toBeNull();
      if (result.error instanceof AgentModelError) {
        expect(result.error.error.kind).toBe("timeout");
      }
    }
  });

  it("malformed when stop_reason=tool_use but no tool_use blocks", async () => {
    const { agent, model } = makeAgent();
    model.pushResponse({
      content: [{ type: "text", text: "hmm" }],
      usage: usage(2, 2),
      stop_reason: "tool_use",
    });
    const result = await agent.turn(ctxUser("?"));
    expect(result.kind).toBe("error");
    if (result.kind === "error") {
      expect(result.error).toBeInstanceOf(MalformedToolCall);
      expect(result.error.kind).toBe("malformed_tool_call");
      expect(result.usage).not.toBeNull();
    }
  });

  it("dispatches tool calls even when stop_reason=end_turn", async () => {
    const { agent, model } = makeAgent();
    model.pushResponse({
      content: [{ type: "tool_use", id: "x", name: "noop", input: {} }],
      usage: usage(1, 1),
      stop_reason: "end_turn",
    });
    const result = await agent.turn(ctxUser("?"));
    expect(result.kind).toBe("tool_call_requested");
  });

  it("classifies stop_reason=max_tokens with text as FinalResponse", async () => {
    const { agent, model } = makeAgent();
    model.pushResponse(textResp("truncated", "max_tokens"));
    const result = await agent.turn(ctxUser("?"));
    expect(result.kind).toBe("final_response");
  });

  it("classifies stop_reason=stop_sequence with text as FinalResponse", async () => {
    const { agent, model } = makeAgent();
    model.pushResponse(textResp("done.", "stop_sequence"));
    const result = await agent.turn(ctxUser("?"));
    expect(result.kind).toBe("final_response");
  });

  it("concatenates multiple text blocks", async () => {
    const { agent, model } = makeAgent();
    model.pushResponse({
      content: [
        { type: "text", text: "foo" },
        { type: "text", text: "bar" },
      ],
      usage: usage(1, 1),
      stop_reason: "end_turn",
    });
    const result = await agent.turn(ctxUser("?"));
    expect(result.kind).toBe("final_response");
    if (result.kind === "final_response") {
      expect(result.content).toBe("foobar");
    }
  });

  it("reports agent identity for tracing", () => {
    const model = new MockModelInterface(provider);
    const agent = new ModelAgent(AgentId.of("initializer"), model);
    expect(agent.id().asString()).toBe("initializer");
    expect(agent.id().equals(AgentId.of("initializer"))).toBe(true);
  });
});

describe("Agent — error JSON wire shape", () => {
  it("AgentError variants serialise with snake_case kind", () => {
    expect(new EmptyResponse().toJSON()).toEqual({ kind: "empty_response" });
    expect(new MalformedToolCall("x", "y").toJSON()).toEqual({
      kind: "malformed_tool_call",
      tool_name: "x",
      reason: "y",
    });
    const wrapped = new AgentModelError(new Timeout()).toJSON();
    expect(wrapped.kind).toBe("model_error");
    expect((wrapped.error as { kind: string }).kind).toBe("timeout");
  });
});

describe("Agent — replay against inline fixture", () => {
  it("classifies recorded exchanges consistently", async () => {
    const jsonl =
      `{"request":{"messages":[{"role":"user","content":{"type":"text","text":"call tool"}}],"tools":[],"params":{},"stream":false},"response":{"content":[{"type":"tool_use","id":"c1","name":"echo","input":{"v":1}}],"usage":{"input_tokens":3,"output_tokens":2},"stop_reason":"tool_use"},"provider":"anthropic"}\n` +
      `{"request":{"messages":[{"role":"user","content":{"type":"text","text":"finish"}}],"tools":[],"params":{},"stream":false},"response":{"content":[{"type":"text","text":"all done"}],"usage":{"input_tokens":4,"output_tokens":2},"stop_reason":"end_turn"},"provider":"anthropic"}`;
    const replay = ReplayModelInterface.fromJsonl(jsonl, {
      name: "anthropic",
      model_id: "replay",
      context_window: 200_000,
    });
    const agent = new ModelAgent(AgentId.of("replay-agent"), replay);

    const r1 = await agent.turn(ctxUser("call tool"));
    expect(r1.kind).toBe("tool_call_requested");
    if (r1.kind === "tool_call_requested") {
      expect(r1.calls).toHaveLength(1);
      expect(r1.calls[0]!.name).toBe("echo");
    }

    const r2 = await agent.turn(ctxUser("finish"));
    expect(r2.kind).toBe("final_response");
    if (r2.kind === "final_response") {
      expect(r2.content).toBe("all done");
    }
  });
});

describe("MockAgent", () => {
  it("returns queued results in FIFO order and counts calls", async () => {
    const { MockAgent } = await import("../src/index.js");
    const agent = new MockAgent(AgentId.of("mock"));
    agent.push({
      kind: "final_response",
      content: "one",
      usage: usage(1, 1),
    });
    const r = await agent.turn(ctxUser("?"));
    expect(r.kind).toBe("final_response");
    expect(agent.callCount).toBe(1);
    // Falls back to EmptyResponse when queue is empty.
    const r2 = await agent.turn(ctxUser("?"));
    expect(r2.kind).toBe("error");
    if (r2.kind === "error") {
      expect(r2.error.kind).toBe("empty_response");
    }
  });
});
