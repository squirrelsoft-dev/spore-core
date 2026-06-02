/**
 * Delta-level streaming through the harness (spore-core #103).
 *
 * Covers the harness `StreamEvent` mapping and ordering:
 *   - model `StreamEvent` → harness event mapping (block_start/stop bracketing,
 *     text/reasoning/tool-args deltas, message_start/stop dropped per Q3)
 *   - tool lifecycle: tool_call_start → tool_args_delta → coarse tool_call with
 *     accumulated args; enriched tool_result content (Q5)
 *   - no-sink baseline parity: identical RunResult whether or not a sink runs
 *   - fixture-replay against the shared golden `streaming_events.json`
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  AgentId,
  ModelAgent,
  ReplayModelInterface,
  SessionId,
  StandardHarness,
  TurnStreamState,
  mapModelStreamEvent,
  newTask,
  type HarnessConfig,
  type HarnessStreamEvent,
  type ProviderInfo,
  type StreamEvent,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const turnFixture = resolve(repoRoot, "fixtures/model_responses/harness/streaming_turn.jsonl");
const eventsFixture = resolve(repoRoot, "fixtures/harness/streaming_events.json");

const provider: ProviderInfo = {
  name: "anthropic",
  model_id: "fixture",
  context_window: 200_000,
};

// ── mapModelStreamEvent unit rules ──────────────────────────────────────────

describe("mapModelStreamEvent (#103)", () => {
  function mapAll(events: StreamEvent[]): HarnessStreamEvent[] {
    const state = new TurnStreamState();
    return events.flatMap((e) => mapModelStreamEvent(e, state));
  }

  it("drops message_start / message_stop (Q3)", () => {
    expect(mapAll([{ type: "message_start" }])).toEqual([]);
    expect(
      mapAll([
        {
          type: "message_stop",
          usage: { input_tokens: 1, output_tokens: 1 },
          stop_reason: "end_turn",
        },
      ]),
    ).toEqual([]);
  });

  it("emits block_start exactly once per index, bracketed by block_stop (Q2)", () => {
    const out = mapAll([
      { type: "content_block_delta", index: 0, delta: "a" },
      { type: "content_block_delta", index: 0, delta: "b" },
      { type: "content_block_stop", index: 0 },
    ]);
    expect(out).toEqual([
      { kind: "block_start", index: 0, block: "text" },
      { kind: "text_delta", content: "a" },
      { kind: "text_delta", content: "b" },
      { kind: "block_stop", index: 0 },
    ]);
  });

  it("maps thinking deltas to reasoning_delta with a reasoning block frame (Q4)", () => {
    const out = mapAll([
      { type: "thinking_delta", index: 0, delta: "ponder" },
      { type: "content_block_stop", index: 0 },
    ]);
    expect(out).toEqual([
      { kind: "block_start", index: 0, block: "reasoning" },
      { kind: "reasoning_delta", content: "ponder" },
      { kind: "block_stop", index: 0 },
    ]);
  });

  it("tool lifecycle: tool_use_start carries real id + name, correlating tool_args_delta", () => {
    const out = mapAll([
      { type: "tool_use_start", index: 2, id: "toolu_stream_1", name: "lookup" },
      { type: "tool_use_delta", index: 2, partial_json: '{"q":' },
      { type: "tool_use_delta", index: 2, partial_json: '"rust"}' },
      { type: "content_block_stop", index: 2 },
    ]);
    expect(out).toEqual([
      { kind: "block_start", index: 2, block: "tool_use" },
      { kind: "tool_call_start", index: 2, call_id: "toolu_stream_1", name: "lookup" },
      { kind: "tool_args_delta", call_id: "toolu_stream_1", partial_json: '{"q":' },
      { kind: "tool_args_delta", call_id: "toolu_stream_1", partial_json: '"rust"}' },
      { kind: "block_stop", index: 2 },
    ]);
  });

  it("fallback: tool_use_delta without a start frame synthesizes call_{index} + empty name", () => {
    const out = mapAll([
      { type: "tool_use_delta", index: 2, partial_json: '{"q":' },
      { type: "tool_use_delta", index: 2, partial_json: '"rust"}' },
      { type: "content_block_stop", index: 2 },
    ]);
    expect(out).toEqual([
      { kind: "block_start", index: 2, block: "tool_use" },
      { kind: "tool_call_start", index: 2, call_id: "call_2", name: "" },
      { kind: "tool_args_delta", call_id: "call_2", partial_json: '{"q":' },
      { kind: "tool_args_delta", call_id: "call_2", partial_json: '"rust"}' },
      { kind: "block_stop", index: 2 },
    ]);
  });
});

// ── fixture-replay against the shared golden ────────────────────────────────

describe("Harness streaming fixture replay (#103)", () => {
  function harnessWith(onStream?: (e: HarnessStreamEvent) => void): {
    harness: StandardHarness;
    toolRegistry: ScriptedToolRegistry;
  } {
    const jsonl = readFileSync(turnFixture, "utf-8");
    const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
    const agent = new ModelAgent(AgentId.of("fixture-agent"), replay);
    const toolRegistry = new ScriptedToolRegistry().push({
      kind: "success",
      content: "found it",
    });
    const config: HarnessConfig = {
      agent,
      toolRegistry,
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
    };
    void onStream;
    return { harness: new StandardHarness(config), toolRegistry };
  }

  it("emits the delta StreamEvents in the order recorded in streaming_events.json", async () => {
    const seen: HarnessStreamEvent[] = [];
    const { harness } = harnessWith();
    const task = newTask("look up rust and explain", SessionId.of("stream-session"), {
      kind: "re_act",
      max_iterations: 1,
    });
    await harness.run({ task, on_stream: (e) => seen.push(e) });

    // The golden lists only the delta/frame events of the first turn (coarse
    // events are emitted after dispatch and are NOT in the golden — Q3 note).
    const golden = JSON.parse(readFileSync(eventsFixture, "utf-8")) as {
      events: HarnessStreamEvent[];
    };
    const deltaKinds = new Set([
      "block_start",
      "block_stop",
      "text_delta",
      "reasoning_delta",
      "tool_args_delta",
      "tool_call_start",
    ]);
    // First turn's deltas appear before the first coarse tool_call.
    const firstCoarse = seen.findIndex((e) => e.kind === "tool_call");
    const deltaSlice = seen
      .slice(0, firstCoarse === -1 ? seen.length : firstCoarse)
      .filter((e) => deltaKinds.has(e.kind));
    expect(deltaSlice).toEqual(golden.events);
  });

  it("emits enriched coarse tool_call (args) and tool_result (content) (Q5)", async () => {
    const seen: HarnessStreamEvent[] = [];
    const { harness } = harnessWith();
    const task = newTask("look up rust and explain", SessionId.of("stream-session-2"), {
      kind: "re_act",
      max_iterations: 1,
    });
    await harness.run({ task, on_stream: (e) => seen.push(e) });

    const toolCall = seen.find((e) => e.kind === "tool_call");
    expect(toolCall).toBeDefined();
    if (toolCall && toolCall.kind === "tool_call") {
      // Coarse ToolCall correlates by the real call id from tool_use_start and
      // carries the accumulated args (Q5) plus the real tool name.
      expect(toolCall.call_id).toBe("toolu_stream_1");
      expect(toolCall.name).toBe("lookup");
      expect(toolCall.args).toEqual({ q: "rust" });
    }
    const toolResult = seen.find((e) => e.kind === "tool_result");
    expect(toolResult).toBeDefined();
    if (toolResult && toolResult.kind === "tool_result") {
      expect(toolResult.content).toBe("found it");
      expect(toolResult.is_error).toBe(false);
    }
  });

  it("no-sink baseline parity: identical RunResult with and without a sink", async () => {
    const withSink = harnessWith();
    const taskA = newTask("look up rust and explain", SessionId.of("parity"), {
      kind: "re_act",
      max_iterations: 1,
    });
    const resultWith = await withSink.harness.run({ task: taskA, on_stream: () => {} });

    const without = harnessWith();
    const taskB = newTask("look up rust and explain", SessionId.of("parity"), {
      kind: "re_act",
      max_iterations: 1,
    });
    const resultWithout = await without.harness.run({ task: taskB });

    expect(resultWith).toEqual(resultWithout);
  });
});
