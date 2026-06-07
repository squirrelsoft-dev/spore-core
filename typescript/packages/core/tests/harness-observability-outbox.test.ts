/**
 * Hermetic end-to-end test: the harness emits real spans through the durable
 * observability outbox (spore-core issues #3 + #12 + #33).
 *
 * Mirrors `rust/crates/spore-core/src/harness.rs#harness_emits_spans_through_outbox_jsonl`.
 *
 * Builds a `StandardHarness` via `HarnessBuilder.withObservabilityOutbox`,
 * scripts a `MockAgent` to do one tool call then a final response, runs it,
 * then reads `{root}/sessions/{sessionId}/trace.jsonl` and asserts the span
 * kinds, the trailing `session` summary, a shared `trace_id`, and the
 * `.flushed` marker. `SPORE_OTLP_ENDPOINT` is left unset (hermetic).
 */

import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  AgentId,
  HarnessBuilder,
  MockAgent,
  SessionId,
  newTask,
  observability,
  type LoopStrategy,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
} from "../src/index.js";

import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

function usage(): TokenUsage {
  return { input_tokens: 10, output_tokens: 5, cache_read_tokens: null, cache_write_tokens: null };
}

function toolCall(id: string, name = "x"): ToolCall {
  return { id, name, input: { a: 1 } };
}

function tcr(call: ToolCall): TurnResult {
  return { kind: "tool_call_requested", calls: [call], usage: usage() };
}

function fr(content: string): TurnResult {
  return { kind: "final_response", content, usage: usage() };
}

const react: LoopStrategy = {
  kind: "react",
  budget: { kind: "per_loop", value: 5 },
  agent: "",
  toolset: "",
};

describe("Harness — spans through durable outbox (hermetic)", () => {
  it("harness emits turn + tool_call spans and a session summary through the JSONL outbox", async () => {
    const root = mkdtempSync(join(tmpdir(), "spore-outbox-"));

    const agent = new MockAgent(AgentId.of("test"));
    agent.push(tcr(toolCall("call-1")));
    agent.push(fr("all done"));

    const tools = new ScriptedToolRegistry();
    tools.push({ kind: "success", content: "tool output", truncated: false });

    const sessionId = SessionId.of("sess-outbox-1");
    const harness = new HarnessBuilder(
      agent,
      tools,
      new AllowAllSandbox(),
      new NoopContextManager(),
      new AlwaysContinuePolicy(),
    )
      .withObservabilityOutbox(root)
      .build();

    const result = await harness.run({ task: newTask("do it", sessionId, react) });
    expect(result.kind).toBe("success");

    const dir = join(root, "sessions", sessionId.asString());
    const tracePath = join(dir, "trace.jsonl");
    expect(existsSync(tracePath)).toBe(true);

    const lines = readFileSync(tracePath, "utf8")
      .split("\n")
      .filter((l) => l.trim().length > 0)
      .map((l) => JSON.parse(l) as Record<string, unknown>);

    const kinds = lines.map((l) => l.kind as string);
    expect(kinds).toContain("turn");
    expect(kinds).toContain("tool_call");

    // Last line is the session summary with the terminal outcome.
    const last = lines[lines.length - 1]!;
    expect(last.kind).toBe("session");
    const attrs = last.attributes as Record<string, unknown>;
    expect(attrs.outcome).toBe("success");
    expect(attrs.total_turns).toBe(2);

    // All lines share a single trace_id.
    const traceIds = new Set(lines.map((l) => l.trace_id as string));
    expect(traceIds.size).toBe(1);

    // The `.flushed` marker exists.
    expect(existsSync(join(dir, ".flushed"))).toBe(true);
  });

  // ── LLM-native content capture (issue #64) ────────────────────────────────

  it("content capture ON writes gen_ai.* content to the JSONL", async () => {
    const root = mkdtempSync(join(tmpdir(), "spore-cc-on-"));

    const agent = new MockAgent(AgentId.of("test"));
    agent.push(tcr(toolCall("call-1", "shell")));
    agent.push(fr("all done"));

    const tools = new ScriptedToolRegistry();
    tools.push({ kind: "success", content: "tool output", truncated: false });

    const sessionId = SessionId.of("sess-cc-on");
    const harness = new HarnessBuilder(
      agent,
      tools,
      new AllowAllSandbox(),
      new NoopContextManager(),
      new AlwaysContinuePolicy(),
    )
      .withObservabilityOutbox(root)
      .contentCapture({ enabled: true, maxFieldLen: 8192 })
      .build();

    const result = await harness.run({ task: newTask("do it", sessionId, react) });
    expect(result.kind).toBe("success");

    const lines = readFileSync(join(root, "sessions", sessionId.asString(), "trace.jsonl"), "utf8")
      .split("\n")
      .filter((l) => l.trim().length > 0)
      .map((l) => JSON.parse(l) as Record<string, unknown>);

    // The turn requesting a tool call carries gen_ai.response.tool_calls.
    const turnWithCalls = lines.find((l) => {
      const a = l.attributes as Record<string, unknown>;
      return l.kind === "turn" && a["gen_ai.response.tool_calls"] != null;
    });
    expect(turnWithCalls).toBeDefined();
    const calls = (turnWithCalls!.attributes as Record<string, unknown>)[
      "gen_ai.response.tool_calls"
    ] as Array<Record<string, unknown>>;
    expect(calls[0].name).toBe("shell");

    // The final-response turn carries gen_ai.response.content.
    const turnFinal = lines.find((l) => {
      const a = l.attributes as Record<string, unknown>;
      return l.kind === "turn" && a["gen_ai.response.content"] === "all done";
    });
    expect(turnFinal).toBeDefined();

    // The tool_call span carries the args + result content.
    const toolLine = lines.find((l) => l.kind === "tool_call")!;
    const ta = toolLine.attributes as Record<string, unknown>;
    expect(ta["gen_ai.tool.name"]).toBe("shell");
    expect(ta["gen_ai.tool.message.content"]).toBe("tool output");
  });

  it("content capture OFF (default) writes no gen_ai.* content", async () => {
    const root = mkdtempSync(join(tmpdir(), "spore-cc-off-"));

    const agent = new MockAgent(AgentId.of("test"));
    agent.push(tcr(toolCall("call-1", "shell")));
    agent.push(fr("all done"));

    const tools = new ScriptedToolRegistry();
    tools.push({ kind: "success", content: "tool output", truncated: false });

    const sessionId = SessionId.of("sess-cc-off");
    // Default builder → content capture OFF; also confirm fromEnv default is OFF.
    expect(observability.ContentCaptureConfig.fromEnv({}).enabled).toBe(false);
    const harness = new HarnessBuilder(
      agent,
      tools,
      new AllowAllSandbox(),
      new NoopContextManager(),
      new AlwaysContinuePolicy(),
    )
      .withObservabilityOutbox(root)
      .build();

    const result = await harness.run({ task: newTask("do it", sessionId, react) });
    expect(result.kind).toBe("success");

    const raw = readFileSync(join(root, "sessions", sessionId.asString(), "trace.jsonl"), "utf8");
    // No gen_ai.* key appears anywhere on disk (byte-identical to pre-#64).
    expect(raw.includes("gen_ai.")).toBe(false);
  });
});
