/**
 * Unit tests for adaptive prompt-based tool calling (#111) — parity with the
 * inline Rust tests in `rust/crates/spore-core/src/prompt_tool_call.rs` and the
 * `detect_prose_response` tests in `tool_call_repair.rs`.
 */

import { describe, expect, it } from "vitest";

import {
  AdaptiveToolCallModelInterface,
  PromptBasedToolCallModelInterface,
  buildToolPrompt,
  detectProseResponse,
  injectToolPrompt,
  newSharedFlag,
  parseProseResponse,
} from "../src/model/index.js";
import { MockModelInterface } from "../src/model/mock.js";
import type {
  ContentBlock,
  ModelRequest,
  ModelResponse,
  ProviderInfo,
  StopReason,
  ToolSchema,
} from "../src/model/schemas.js";

function provider(): ProviderInfo {
  return { name: "test", model_id: "test-1", context_window: 4096 };
}

function toolSchema(): ToolSchema {
  return {
    name: "calculator",
    description: "evaluate math",
    input_schema: {
      type: "object",
      properties: { expression: { type: "string" } },
      required: ["expression"],
    },
  };
}

function reqWithTools(system?: string): ModelRequest {
  const messages: ModelRequest["messages"] = [];
  if (system != null) {
    messages.push({ role: "system", content: { type: "text", text: system } });
  }
  messages.push({ role: "user", content: { type: "text", text: "what is 2+2?" } });
  return { messages, tools: [toolSchema()], params: { stop_sequences: [] }, stream: false };
}

function usage(): ModelResponse["usage"] {
  return { input_tokens: 1, output_tokens: 1 };
}

function prose(text: string, stop: StopReason): ModelResponse {
  return { content: [{ type: "text", text }], usage: usage(), stop_reason: stop };
}

function systemText(req: ModelRequest): string {
  const c = req.messages[0]?.content;
  if (c == null || c.type !== "text") throw new Error("system message must be text");
  return c.text;
}

// --- build_tool_prompt byte-for-byte -----------------------------------------

describe("buildToolPrompt", () => {
  it("matches the Rust build_tool_prompt output byte-for-byte", () => {
    const expected =
      "You have access to the following tools. Use them when they would help complete the task.\n\n" +
      "<available_tools>\n" +
      "<tool>\n" +
      "  <name>calculator</name>\n" +
      "  <description>evaluate math</description>\n" +
      '  <input_schema>{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}</input_schema>\n' +
      "</tool>\n" +
      "</available_tools>\n\n" +
      "When you want to use a tool, respond with ONLY the following format and nothing else:\n" +
      '<tool_call>\n  <name>tool_name_here</name>\n  <input>{"key": "value"}</input>\n</tool_call>\n\n' +
      "When you have a final answer that does not require a tool, respond normally in prose.";
    expect(buildToolPrompt([toolSchema()])).toBe(expected);
  });
});

// --- injection ---------------------------------------------------------------

describe("injectToolPrompt", () => {
  it("appends to an existing system prompt", () => {
    const req = reqWithTools("You are a helpful assistant.");
    injectToolPrompt(req);
    const sys = systemText(req);
    expect(sys.startsWith("You are a helpful assistant.")).toBe(true);
    expect(sys).toContain("<available_tools>");
    expect(sys).toContain("<name>calculator</name>");
    expect(req.messages[1]?.role).toBe("user");
  });

  it("inserts a system prompt when absent", () => {
    const req = reqWithTools();
    injectToolPrompt(req);
    expect(req.messages[0]?.role).toBe("system");
    expect(systemText(req)).toContain("<available_tools>");
  });

  it("is idempotent", () => {
    const req = reqWithTools("base");
    injectToolPrompt(req);
    const once = systemText(req);
    injectToolPrompt(req);
    expect(systemText(req)).toBe(once);
  });

  it("is a no-op without tools", () => {
    const req = reqWithTools("base");
    req.tools = [];
    const before = structuredClone(req.messages);
    injectToolPrompt(req);
    expect(req.messages).toEqual(before);
  });
});

// --- parsing -----------------------------------------------------------------

describe("parseProseResponse", () => {
  it("parses a single tool-call marker into a tool-use block", () => {
    const out = parseProseResponse(
      prose(
        '<tool_call><name>calculator</name><input>{"expression": "2+2"}</input></tool_call>',
        "end_turn",
      ),
    );
    expect(out.stop_reason).toBe("tool_use");
    expect(out.content).toHaveLength(1);
    const block = out.content[0]!;
    expect(block.type).toBe("tool_use");
    if (block.type !== "tool_use") throw new Error("expected tool_use");
    expect(block.name).toBe("calculator");
    expect(block.id).toBe("ptc_call_0");
    expect(block.input).toEqual({ expression: "2+2" });
  });

  it("parses multiple tool-call markers", () => {
    const text =
      '<tool_call><name>a</name><input>{"x":1}</input></tool_call>\n' +
      "some chatter\n" +
      '<tool_call><name>b</name><input>{"y":2}</input></tool_call>';
    const out = parseProseResponse(prose(text, "end_turn"));
    const names = out.content
      .filter((b): b is Extract<ContentBlock, { type: "tool_use" }> => b.type === "tool_use")
      .map((b) => b.name);
    expect(names).toEqual(["a", "b"]);
    expect(out.stop_reason).toBe("tool_use");
  });

  it("falls through as prose when input JSON is malformed", () => {
    const resp = prose(
      "<tool_call><name>calculator</name><input>{not valid json}</input></tool_call>",
      "end_turn",
    );
    const out = parseProseResponse(resp);
    expect(out.stop_reason).toBe("end_turn");
    expect(out.content[0]?.type).toBe("text");
  });

  it("returns plain prose unchanged", () => {
    const resp = prose("The answer is 4.", "end_turn");
    expect(parseProseResponse(resp)).toEqual(resp);
  });

  it("leaves native tool-use untouched", () => {
    const resp: ModelResponse = {
      content: [{ type: "tool_use", id: "native", name: "calculator", input: { expression: "1" } }],
      usage: usage(),
      stop_reason: "tool_use",
    };
    expect(parseProseResponse(resp)).toEqual(resp);
  });

  it("preserves thinking blocks alongside synthesized calls", () => {
    const resp: ModelResponse = {
      content: [
        { type: "thinking", text: "reasoning" },
        { type: "text", text: "<tool_call><name>t</name><input>{}</input></tool_call>" },
      ],
      usage: usage(),
      stop_reason: "end_turn",
    };
    const out = parseProseResponse(resp);
    expect(out.content[0]?.type).toBe("thinking");
    expect(out.content[1]?.type).toBe("tool_use");
  });
});

// --- always-on wrapper -------------------------------------------------------

describe("PromptBasedToolCallModelInterface", () => {
  it("injects the tool prompt and parses markers", async () => {
    const m = new MockModelInterface(provider());
    m.pushResponse(
      prose(
        '<tool_call><name>calculator</name><input>{"expression":"2+2"}</input></tool_call>',
        "end_turn",
      ),
    );
    const wrapper = new PromptBasedToolCallModelInterface(m);
    const resp = await wrapper.call(reqWithTools("base"));
    expect(resp.stop_reason).toBe("tool_use");
    expect(resp.content[0]?.type).toBe("tool_use");
  });

  it("delegates provider() to the inner model", () => {
    const m = new MockModelInterface(provider());
    expect(new PromptBasedToolCallModelInterface(m).provider().model_id).toBe("test-1");
  });
});

// --- adaptive wrapper --------------------------------------------------------

describe("AdaptiveToolCallModelInterface", () => {
  it("delegates natively when the flag is unset", async () => {
    const m = new MockModelInterface(provider());
    // A marker-bearing prose response — but the flag is OFF, so the wrapper must
    // NOT parse it; it delegates verbatim.
    m.pushResponse(prose("<tool_call><name>x</name><input>{}</input></tool_call>", "end_turn"));
    const flag = newSharedFlag();
    const wrapper = new AdaptiveToolCallModelInterface(m, flag);
    const resp = await wrapper.call(reqWithTools("base"));
    expect(resp.stop_reason).toBe("end_turn");
    expect(resp.content[0]?.type).toBe("text");
  });

  it("injects and parses when the flag is set", async () => {
    const m = new MockModelInterface(provider());
    m.pushResponse(
      prose('<tool_call><name>x</name><input>{"k":1}</input></tool_call>', "end_turn"),
    );
    const flag = newSharedFlag();
    flag.value = true;
    const wrapper = new AdaptiveToolCallModelInterface(m, flag);
    const resp = await wrapper.call(reqWithTools("base"));
    expect(resp.stop_reason).toBe("tool_use");
    const block = resp.content[0]!;
    if (block.type !== "tool_use") throw new Error("expected tool_use");
    expect(block.name).toBe("x");
    expect(block.input).toEqual({ k: 1 });
  });

  it("delegates provider() to the inner model", () => {
    const m = new MockModelInterface(provider());
    const wrapper = new AdaptiveToolCallModelInterface(m, newSharedFlag());
    expect(wrapper.provider().model_id).toBe("test-1");
  });
});

// --- prose detection ---------------------------------------------------------

describe("detectProseResponse", () => {
  it("detects action intent when tools are advertised", () => {
    expect(
      detectProseResponse("Sure, I'll use the calculator tool to add these.", true),
    ).not.toBeNull();
  });

  it("is case-insensitive", () => {
    expect(detectProseResponse("LET ME CALL the search tool now.", true)).not.toBeNull();
  });

  it("does not detect without tools advertised", () => {
    expect(detectProseResponse("I'll use the calculator.", false)).toBeNull();
  });

  it("does not detect a plain final answer", () => {
    expect(detectProseResponse("The answer is 42.", true)).toBeNull();
  });

  it("does not detect empty text", () => {
    expect(detectProseResponse("   ", true)).toBeNull();
  });
});
