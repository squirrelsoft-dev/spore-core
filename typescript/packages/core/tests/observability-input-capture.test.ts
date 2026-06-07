/**
 * Tests for assembled INPUT-message capture on the turn span (spore-core
 * issue #64). Mirrors the Rust reference's semantics + wire contract:
 *   - `TurnSpan.input_messages` → the `gen_ai.prompt` attribute (ordered
 *     `{ role, content, truncated }`, system first then history).
 *   - guard OFF → no `input_messages` / no `gen_ai.prompt` (byte-identical).
 *   - `emitGenaiEvents` emits one event per INPUT message FIRST, in order,
 *     before the output event(s).
 *   - end-to-end through `StandardHarness`: roles + order, image→placeholder
 *     (no base64), tool-call/tool-result rendering, truncation applied.
 *   - fixture-replay against the shared GROUND TRUTH
 *     `fixtures/observability/trace_line_turn_with_input.json`.
 *
 * Hermetic: temp dirs + temp env only, no live OTLP / network.
 */

import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  AgentId,
  HarnessBuilder,
  MockAgent,
  SessionId,
  newTask,
  observability,
  type LoopStrategy,
  type Message,
  type SessionState,
  type StopReason,
  type TokenUsage,
  type TurnResult,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

const { TraceLine, emitGenaiEvents, TRUNCATION_MARKER } = observability;

type TurnSpan = observability.TurnSpan;
type SpanBase = observability.SpanBase;
type GenAiMessage = observability.GenAiMessage;

const TRACE_ID = "0af7651916cd43dd8448eb211c80319c";

function baseOf(): SpanBase {
  return {
    span_id: "b7ad6b7169203331" as unknown as SpanBase["span_id"],
    parent_span_id: null,
    session_id: "sess_a" as unknown as SpanBase["session_id"],
    task_id: "task_a" as unknown as SpanBase["task_id"],
    kind: "turn",
    started_at: "2026-05-26T18:00:00.0Z" as unknown as SpanBase["started_at"],
    ended_at: "2026-05-26T18:00:02.1Z" as unknown as SpanBase["ended_at"],
    duration_ms: 2100,
    status: { kind: "ok" },
  };
}

function turnSpan(overrides: Partial<TurnSpan> = {}): TurnSpan {
  return {
    base: baseOf(),
    turn_number: 1,
    input_tokens: 10,
    output_tokens: 5,
    cache_read_tokens: null,
    cache_write_tokens: null,
    cost_usd: 0,
    stop_reason: "end_turn",
    tool_calls_requested: 0,
    ...overrides,
  };
}

const PROMPT: GenAiMessage[] = [
  { role: "system", content: "be helpful", truncated: false },
  { role: "user", content: "hi", truncated: false },
  { role: "assistant", content: 'shell {"command":"ls"}', truncated: false },
  { role: "tool", content: "file.txt", truncated: false },
];

// ── fromTurn: input_messages → gen_ai.prompt ─────────────────────────────────

describe("TraceLine.fromTurn input capture", () => {
  it("turn with input_messages emits an ordered gen_ai.prompt attribute", () => {
    const line = TraceLine.fromTurn(turnSpan({ input_messages: PROMPT }), TRACE_ID);
    expect(line.attributes["gen_ai.prompt"]).toEqual(PROMPT);
  });

  it("guard OFF (no input_messages) carries no gen_ai.prompt key", () => {
    const line = TraceLine.fromTurn(turnSpan(), TRACE_ID);
    expect("gen_ai.prompt" in line.attributes).toBe(false);
  });

  it("input_messages: null carries no gen_ai.prompt key", () => {
    const line = TraceLine.fromTurn(turnSpan({ input_messages: null }), TRACE_ID);
    expect("gen_ai.prompt" in line.attributes).toBe(false);
  });
});

// ── emitGenaiEvents: input messages emitted first, in order ──────────────────

describe("emitGenaiEvents input prompt events", () => {
  it("emits one event per INPUT message, in order, with conventional names", () => {
    const line = TraceLine.fromTurn(turnSpan({ input_messages: PROMPT }), TRACE_ID);
    const events = emitGenaiEvents(line);
    expect(events.map((e) => e.name)).toEqual([
      "gen_ai.system.message",
      "gen_ai.user.message",
      "gen_ai.assistant.message",
      "gen_ai.tool.message",
    ]);
    expect(events[0]!.attributes["gen_ai.message.role"]).toBe("system");
    expect(events[0]!.attributes["gen_ai.message.content"]).toBe("be helpful");
    expect(events[3]!.attributes["gen_ai.message.content"]).toBe("file.txt");
  });

  it("emits INPUT messages FIRST, before the output event", () => {
    const line = TraceLine.fromTurn(
      turnSpan({
        input_messages: PROMPT,
        output_text: { role: "assistant", content: "done", truncated: false },
      }),
      TRACE_ID,
    );
    const events = emitGenaiEvents(line);
    // 4 input + 1 output.
    expect(events).toHaveLength(5);
    expect(events[0]!.attributes["gen_ai.message.role"]).toBe("system");
    // The output event is last and carries the response content.
    expect(events[4]!.name).toBe("gen_ai.assistant.message");
    expect(events[4]!.attributes["gen_ai.message.content"]).toBe("done");
  });
});

// ── Fixture replay against the shared ground truth ───────────────────────────

interface InputFixture {
  trace_id: string;
  span: {
    base: Record<string, unknown>;
    turn_number: number;
    input_tokens: number;
    output_tokens: number;
    cache_read_tokens: number | null;
    cache_write_tokens: number | null;
    cost_usd: number;
    stop_reason: StopReason;
    tool_calls_requested: number;
    tool_calls?: observability.ToolCallContent[];
    input_messages?: GenAiMessage[];
  };
  expected_line: { attributes: Record<string, unknown> };
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(
  here,
  "../../../../fixtures/observability/trace_line_turn_with_input.json",
);

function spanFromFixture(fixture: InputFixture): TurnSpan {
  return {
    base: fixture.span.base as unknown as SpanBase,
    turn_number: fixture.span.turn_number,
    input_tokens: fixture.span.input_tokens,
    output_tokens: fixture.span.output_tokens,
    cache_read_tokens: fixture.span.cache_read_tokens,
    cache_write_tokens: fixture.span.cache_write_tokens,
    cost_usd: fixture.span.cost_usd,
    stop_reason: fixture.span.stop_reason,
    tool_calls_requested: fixture.span.tool_calls_requested,
    tool_calls: fixture.span.tool_calls ?? null,
    input_messages: fixture.span.input_messages ?? null,
  };
}

describe("Input-capture fixture replay", () => {
  it("trace_line_turn_with_input.json: fromTurn reproduces the expected attributes", () => {
    const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as InputFixture;
    const line = TraceLine.fromTurn(spanFromFixture(fixture), fixture.trace_id);
    expect(line.attributes).toEqual(fixture.expected_line.attributes);
  });

  it("trace_line_turn_with_input.json: input prompt events come first, in order", () => {
    const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as InputFixture;
    const events = emitGenaiEvents(TraceLine.fromTurn(spanFromFixture(fixture), fixture.trace_id));
    expect(events.slice(0, 4).map((e) => e.name)).toEqual([
      "gen_ai.system.message",
      "gen_ai.user.message",
      "gen_ai.assistant.message",
      "gen_ai.tool.message",
    ]);
  });
});

// ── End-to-end through StandardHarness ───────────────────────────────────────

const react: LoopStrategy = {
  kind: "react",
  budget: { kind: "per_loop", value: 1 },
  agent: "",
  toolset: "",
};

function usage(): TokenUsage {
  return { input_tokens: 5, output_tokens: 2, cache_read_tokens: null, cache_write_tokens: null };
}
function fr(content: string): TurnResult {
  return { kind: "final_response", content, usage: usage() };
}

function seededState(messages: Message[]): SessionState {
  return { messages, extras: {} };
}

async function runWithMessages(
  root: string,
  sessionId: string,
  messages: Message[],
  cc: observability.ContentCaptureConfig,
): Promise<Record<string, unknown>[]> {
  const agent = new MockAgent(AgentId.of("test"));
  agent.push(fr("all done"));
  const tools = new ScriptedToolRegistry();
  const sid = SessionId.of(sessionId);
  const harness = new HarnessBuilder(
    agent,
    tools,
    new AllowAllSandbox(),
    new NoopContextManager(),
    new AlwaysContinuePolicy(),
  )
    .withObservabilityOutbox(root)
    .contentCapture(cc)
    .build();
  await harness.run({ task: newTask("do it", sid, react), session_state: seededState(messages) });
  const lines = readFileSync(join(root, "sessions", sid.asString(), "trace.jsonl"), "utf8")
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l) as Record<string, unknown>);
  return lines.filter((l) => l.kind === "turn");
}

describe("StandardHarness input capture (end-to-end)", () => {
  const messages: Message[] = [
    { role: "system", content: { type: "text", text: "be helpful" } },
    { role: "user", content: { type: "text", text: "list files" } },
    {
      role: "assistant",
      content: { type: "tool_call", id: "c1", name: "shell", input: { command: "ls" } },
    },
    {
      role: "tool",
      content: { type: "tool_result", tool_use_id: "c1", content: "file.txt", is_error: false },
    },
    { role: "user", content: { type: "image", media_type: "image/png", data: "QUJDREVG_base64" } },
  ];

  it("guard ON: assembled prompt rides as gen_ai.prompt with roles + order, no base64", async () => {
    const root = mkdtempSync(join(tmpdir(), "spore-input-on-"));
    const turns = await runWithMessages(root, "s_on", messages, {
      enabled: true,
      maxFieldLen: 8192,
    });
    const turn = turns[0]!;
    const prompt = (turn.attributes as Record<string, unknown>)["gen_ai.prompt"] as GenAiMessage[];
    // The harness appends the task description as a trailing user message, so
    // the captured prompt is the 5 seeded messages followed by the task prompt.
    // The implementation faithfully captures whatever the model saw, in order.
    expect(prompt.map((m) => m.role)).toEqual([
      "system",
      "user",
      "assistant",
      "tool",
      "user",
      "user",
    ]);
    expect(prompt[0]!.content).toBe("be helpful");
    expect(prompt[1]!.content).toBe("list files");
    expect(prompt[2]!.content).toBe('shell {"command":"ls"}');
    expect(prompt[3]!.content).toBe("file.txt");
    // image → placeholder, NEVER the base64 data.
    expect(prompt[4]!.content).toBe("[image image/png]");
    expect(prompt[5]!.content).toBe("do it");
    expect(JSON.stringify(turn)).not.toContain("QUJDREVG_base64");
  });

  it("guard OFF: no input_messages / no gen_ai.prompt", async () => {
    const root = mkdtempSync(join(tmpdir(), "spore-input-off-"));
    const turns = await runWithMessages(root, "s_off", messages, {
      enabled: false,
      maxFieldLen: 8192,
    });
    const turn = turns[0]!;
    expect("gen_ai.prompt" in (turn.attributes as Record<string, unknown>)).toBe(false);
    expect("input_messages" in turn).toBe(false);
  });

  it("truncation is applied per input message", async () => {
    const root = mkdtempSync(join(tmpdir(), "spore-input-trunc-"));
    const longMsgs: Message[] = [
      { role: "user", content: { type: "text", text: "x".repeat(100) } },
    ];
    const turns = await runWithMessages(root, "s_trunc", longMsgs, {
      enabled: true,
      maxFieldLen: 10,
    });
    const turn = turns[0]!;
    const prompt = (turn.attributes as Record<string, unknown>)["gen_ai.prompt"] as GenAiMessage[];
    expect(prompt[0]!.truncated).toBe(true);
    expect(prompt[0]!.content).toBe("x".repeat(10) + TRUNCATION_MARKER);
  });
});
