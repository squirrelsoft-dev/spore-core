/**
 * Unit tests for LLM-native content capture (spore-core issue #64).
 *
 * Covers every rule mirrored from the Rust reference
 * (`rust/crates/spore-core/src/observability.rs` + `observability_outbox.rs`):
 *   - truncateField: within budget / byte-boundary clip+mark / multibyte back-off.
 *   - ContentCaptureConfig.default + fromEnv (default OFF; env parsing).
 *   - GenAiRole → conventional event-name mapping.
 *   - turn / tool-call content on/off serialization (gen_ai.* attributes).
 *   - emitGenaiEvents: one conventional span event per captured message.
 *
 * Hermetic: temp env only, no live OTLP / network.
 */

import { describe, expect, it } from "vitest";

import { observability } from "../src/index.js";

const {
  ContentCaptureConfig,
  GenAiRole,
  TRUNCATION_MARKER,
  truncateField,
  TraceLine,
  emitGenaiEvents,
} = observability;

type SpanBase = observability.SpanBase;
type TurnSpan = observability.TurnSpan;
type ToolCallSpan = observability.ToolCallSpan;

const TRACE_ID = "0af7651916cd43dd8448eb211c80319c";

function utf8Len(s: string): number {
  return new TextEncoder().encode(s).length;
}

function baseOf(kind: observability.SpanKind, spanId = "sp1"): SpanBase {
  return {
    span_id: spanId as unknown as SpanBase["span_id"],
    parent_span_id: null,
    session_id: "sess_a" as unknown as SpanBase["session_id"],
    task_id: "task_a" as unknown as SpanBase["task_id"],
    kind,
    started_at: "2026-05-26T18:00:00.0Z" as unknown as SpanBase["started_at"],
    ended_at: "2026-05-26T18:00:02.1Z" as unknown as SpanBase["ended_at"],
    duration_ms: 2100,
    status: { kind: "ok" },
  };
}

// ── truncateField ───────────────────────────────────────────────────────────

describe("truncateField", () => {
  it("returns the string unchanged when within the byte budget", () => {
    const [out, truncated] = truncateField("hello", 8192);
    expect(out).toBe("hello");
    expect(truncated).toBe(false);
  });

  it("returns unchanged at exactly the budget (byte length == max)", () => {
    const s = "abcdef"; // 6 bytes
    const [out, truncated] = truncateField(s, 6);
    expect(out).toBe(s);
    expect(truncated).toBe(false);
  });

  it("clips at an ASCII byte boundary and appends the marker", () => {
    const s = "the quick brown fox jumps";
    const [out, truncated] = truncateField(s, 19);
    expect(truncated).toBe(true);
    expect(out).toBe("the quick brown fox" + TRUNCATION_MARKER);
    expect(out.endsWith("...[truncated]")).toBe(true);
  });

  it("never splits a multibyte char — backs off to the previous boundary", () => {
    // "é" is 2 UTF-8 bytes (0xC3 0xA9). "aé" is 3 bytes.
    const s = "aéb"; // bytes: 61 C3 A9 62
    // max=2 would land mid-é (byte index 2 is a continuation byte) → back off to 1.
    const [out, truncated] = truncateField(s, 2);
    expect(truncated).toBe(true);
    expect(out).toBe("a" + TRUNCATION_MARKER);
  });

  it("keeps a whole multibyte char when the budget lands exactly on its boundary", () => {
    const s = "aéb"; // bytes: 61 C3 A9 62 (4 bytes)
    // max=3 lands exactly after "é" (a valid boundary).
    const [out, truncated] = truncateField(s, 3);
    expect(truncated).toBe(true);
    expect(out).toBe("aé" + TRUNCATION_MARKER);
  });

  it("measures bytes, not UTF-16 code units (multibyte content over budget)", () => {
    // 10 emoji; each is 4 UTF-8 bytes = 40 bytes, but 20 UTF-16 code units.
    const s = "😀".repeat(10);
    expect(utf8Len(s)).toBe(40);
    const [out, truncated] = truncateField(s, 8);
    expect(truncated).toBe(true);
    // 8 bytes = exactly 2 emoji; never a partial emoji.
    expect(out).toBe("😀😀" + TRUNCATION_MARKER);
  });
});

// ── ContentCaptureConfig ─────────────────────────────────────────────────────

describe("ContentCaptureConfig", () => {
  it("default() is OFF with an 8192-byte cap", () => {
    const cfg = ContentCaptureConfig.default();
    expect(cfg.enabled).toBe(false);
    expect(cfg.maxFieldLen).toBe(8192);
  });

  it("fromEnv() with no env vars is OFF / 8192", () => {
    const cfg = ContentCaptureConfig.fromEnv({});
    expect(cfg.enabled).toBe(false);
    expect(cfg.maxFieldLen).toBe(8192);
  });

  it.each(["1", "true", "yes", "on", "TRUE", "On", " yes "])("fromEnv() enables on %j", (val) => {
    expect(ContentCaptureConfig.fromEnv({ SPORE_TRACE_CONTENT: val }).enabled).toBe(true);
  });

  it.each(["0", "false", "no", "off", "", "maybe"])("fromEnv() stays OFF on %j", (val) => {
    expect(ContentCaptureConfig.fromEnv({ SPORE_TRACE_CONTENT: val }).enabled).toBe(false);
  });

  it("fromEnv() honors a valid SPORE_TRACE_CONTENT_MAX_LEN", () => {
    expect(ContentCaptureConfig.fromEnv({ SPORE_TRACE_CONTENT_MAX_LEN: "256" }).maxFieldLen).toBe(
      256,
    );
  });

  it("fromEnv() falls back to 8192 on an unparseable max-len", () => {
    expect(ContentCaptureConfig.fromEnv({ SPORE_TRACE_CONTENT_MAX_LEN: "abc" }).maxFieldLen).toBe(
      8192,
    );
    expect(ContentCaptureConfig.fromEnv({ SPORE_TRACE_CONTENT_MAX_LEN: "-5" }).maxFieldLen).toBe(
      8192,
    );
  });
});

// ── GenAiRole → event name ───────────────────────────────────────────────────

describe("GenAiRole.eventName", () => {
  it.each([
    ["system", "gen_ai.system.message"],
    ["user", "gen_ai.user.message"],
    ["assistant", "gen_ai.assistant.message"],
    ["tool", "gen_ai.tool.message"],
  ] as const)("maps %s → %s", (role, expected) => {
    expect(GenAiRole.eventName(role)).toBe(expected);
  });
});

// ── Content on/off serialization (gen_ai.* attributes) ───────────────────────

describe("TraceLine content serialization", () => {
  function turnSpan(overrides: Partial<TurnSpan> = {}): TurnSpan {
    return {
      base: baseOf("turn"),
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

  it("turn with NO content has no gen_ai.* keys (byte-identical to pre-#64)", () => {
    const line = TraceLine.fromTurn(turnSpan(), TRACE_ID);
    const keys = Object.keys(line.attributes);
    expect(keys.some((k) => k.startsWith("gen_ai."))).toBe(false);
  });

  it("turn with output_text emits gen_ai.response.* attributes", () => {
    const line = TraceLine.fromTurn(
      turnSpan({ output_text: { role: "assistant", content: "hi there", truncated: false } }),
      TRACE_ID,
    );
    expect(line.attributes["gen_ai.response.role"]).toBe("assistant");
    expect(line.attributes["gen_ai.response.content"]).toBe("hi there");
    expect(line.attributes["gen_ai.response.content_truncated"]).toBe(false);
  });

  it("turn with tool_calls emits gen_ai.response.tool_calls", () => {
    const line = TraceLine.fromTurn(
      turnSpan({
        tool_calls: [{ name: "shell", arguments: { command: "ls" }, arguments_truncated: false }],
      }),
      TRACE_ID,
    );
    expect(line.attributes["gen_ai.response.tool_calls"]).toEqual([
      { name: "shell", arguments: { command: "ls" }, arguments_truncated: false },
    ]);
  });

  it("tool_call with NO content has no gen_ai.* keys", () => {
    const span: ToolCallSpan = {
      base: baseOf("tool_call"),
      tool_name: "shell",
      call_id: "c1",
      parameters_size_bytes: 4,
      output_size_bytes: 2,
      truncated: false,
      sandbox_mode: "",
      sandbox_violations: [],
    };
    const line = TraceLine.fromToolCall(span, TRACE_ID);
    expect(Object.keys(line.attributes).some((k) => k.startsWith("gen_ai."))).toBe(false);
  });

  it("tool_call with content emits gen_ai.tool.* attributes", () => {
    const span: ToolCallSpan = {
      base: baseOf("tool_call"),
      tool_name: "shell",
      call_id: "c1",
      parameters_size_bytes: 4,
      output_size_bytes: 2,
      truncated: false,
      sandbox_mode: "",
      sandbox_violations: [],
      arguments: { name: "shell", arguments: { command: "ls" }, arguments_truncated: false },
      result: { content: "ok", truncated: false },
    };
    const line = TraceLine.fromToolCall(span, TRACE_ID);
    expect(line.attributes["gen_ai.tool.name"]).toBe("shell");
    expect(line.attributes["gen_ai.tool.call.arguments"]).toEqual({ command: "ls" });
    expect(line.attributes["gen_ai.tool.call.arguments_truncated"]).toBe(false);
    expect(line.attributes["gen_ai.tool.message.content"]).toBe("ok");
    expect(line.attributes["gen_ai.tool.message.content_truncated"]).toBe(false);
  });
});

// ── emitGenaiEvents: one conventional span event per message ─────────────────

describe("emitGenaiEvents", () => {
  it("returns no events when content capture is off", () => {
    const line = TraceLine.fromTurn(
      {
        base: baseOf("turn"),
        turn_number: 1,
        input_tokens: 10,
        output_tokens: 5,
        cache_read_tokens: null,
        cache_write_tokens: null,
        cost_usd: 0,
        stop_reason: "end_turn",
        tool_calls_requested: 0,
      },
      TRACE_ID,
    );
    expect(emitGenaiEvents(line)).toEqual([]);
  });

  it("emits one assistant.message event for turn output text", () => {
    const line = TraceLine.fromTurn(
      {
        base: baseOf("turn"),
        turn_number: 1,
        input_tokens: 10,
        output_tokens: 5,
        cache_read_tokens: null,
        cache_write_tokens: null,
        cost_usd: 0,
        stop_reason: "end_turn",
        tool_calls_requested: 0,
        output_text: { role: "assistant", content: "done", truncated: false },
      },
      TRACE_ID,
    );
    const events = emitGenaiEvents(line);
    expect(events).toHaveLength(1);
    expect(events[0].name).toBe("gen_ai.assistant.message");
    expect(events[0].attributes["gen_ai.message.role"]).toBe("assistant");
    expect(events[0].attributes["gen_ai.message.content"]).toBe("done");
  });

  it("emits one assistant.message event per requested tool call", () => {
    const line = TraceLine.fromTurn(
      {
        base: baseOf("turn"),
        turn_number: 1,
        input_tokens: 10,
        output_tokens: 5,
        cache_read_tokens: null,
        cache_write_tokens: null,
        cost_usd: 0,
        stop_reason: "tool_use",
        tool_calls_requested: 2,
        tool_calls: [
          { name: "shell", arguments: { command: "ls" }, arguments_truncated: false },
          { name: "read", arguments: { path: "/x" }, arguments_truncated: false },
        ],
      },
      TRACE_ID,
    );
    const events = emitGenaiEvents(line);
    expect(events).toHaveLength(2);
    for (const ev of events) {
      expect(ev.name).toBe("gen_ai.assistant.message");
      expect(ev.attributes["gen_ai.message.role"]).toBe("assistant");
    }
    expect(events[0].attributes["gen_ai.tool.name"]).toBe("shell");
    expect(events[0].attributes["gen_ai.tool.call.arguments"]).toBe(
      JSON.stringify({ command: "ls" }),
    );
    expect(events[1].attributes["gen_ai.tool.name"]).toBe("read");
  });

  it("emits one tool.message event for a tool result", () => {
    const span: ToolCallSpan = {
      base: baseOf("tool_call"),
      tool_name: "shell",
      call_id: "c1",
      parameters_size_bytes: 4,
      output_size_bytes: 2,
      truncated: false,
      sandbox_mode: "",
      sandbox_violations: [],
      arguments: { name: "shell", arguments: { command: "ls" }, arguments_truncated: false },
      result: { content: "total 0", truncated: false },
    };
    const line = TraceLine.fromToolCall(span, TRACE_ID);
    const events = emitGenaiEvents(line);
    expect(events).toHaveLength(1);
    expect(events[0].name).toBe("gen_ai.tool.message");
    expect(events[0].attributes["gen_ai.message.role"]).toBe("tool");
    expect(events[0].attributes["gen_ai.message.content"]).toBe("total 0");
  });
});
