/**
 * Output-schema delivery + enforcement harness tests (spore-core issue #139).
 *
 * Mirrors the Rust `os_*` test module in
 * `rust/crates/spore-core/src/harness.rs` — same types, same rules, same
 * frozen literals.
 *
 * Types/rules under test:
 *   - `HarnessConfig.enforceOutputSchemas` (MIGRATION GATE, default OFF).
 *   - `HarnessConfig.outputSchemaMaxRetries` (N, default 2).
 *   - `HaltReason.output_schema_violation { schema, attempts, last_error }`.
 *   - AC1: the resolved schema is DELIVERED to the leaf's directive seed
 *     (key-sorted) AND set on `ModelParams.output_schema` (the Ollama `format`
 *     population is unit-tested in `output-schema.test.ts`).
 *   - AC2: terminal validated; valid ⇒ Success; invalid ⇒ feed the frozen error
 *     back + retry within budget.
 *   - AC3: after N retries WITH budget remaining ⇒ output_schema_violation
 *     (distinct from budget; turns < budget). AND budget-precedence: a retry
 *     colliding with the budget cap surfaces budget_exceeded, NOT a violation.
 *   - AC4: flag OFF ⇒ an invalid terminal is accepted as Success.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyToolRegistry,
  ExecutionRegistry,
  MockAgent,
  SessionId,
  StandardHarness,
  autonomous,
  newTask,
  type HarnessConfig,
  type HarnessStreamEvent,
  type LoopStrategy,
  type SessionState,
  type Task,
  type TokenUsage,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

// ── helpers ──────────────────────────────────────────────────────────────────

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function makeAgent(): MockAgent {
  return new MockAgent(AgentId.of("fixture-agent"));
}

/** The schema the #139 tests enforce: an object requiring `status` (one of
 *  `ok`/`error`) and a `count` integer. */
function osSchema(): unknown {
  return {
    type: "object",
    required: ["status", "count"],
    properties: {
      status: { type: "string", enum: ["ok", "error"] },
      count: { type: "integer" },
    },
  };
}

/** Canonical (key-sorted) form of {@link osSchema} — the bytes delivered to the
 *  directive seed + reported on the violation terminal. Pinned so the four ports
 *  byte-compare. */
const EXPECTED_SCHEMA =
  '{"properties":{"count":{"type":"integer"},"status":{"enum":["ok","error"],"type":"string"}},' +
  '"required":["status","count"],"type":"object"}';

const OS_VALID = '{"status":"ok","count":3}';
const OS_INVALID = "{}";

/** A registry with the real schema under the default empty key. */
function osRegistry(agent: MockAgent): ExecutionRegistry {
  return ExecutionRegistry.builder()
    .agent("", agent)
    .toolset("", new EmptyToolRegistry())
    .schema("", osSchema())
    .build();
}

/** A config with output-schema enforcement ON and {@link osSchema} registered. */
function osConfig(agent: MockAgent, maxRetries: number): HarnessConfig {
  return {
    registry: osRegistry(agent),
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    escalationMode: autonomous,
    enforceOutputSchemas: true,
    outputSchemaMaxRetries: maxRetries,
  };
}

/** A bare ReAct leaf carrying `output: ""` and a turn budget. */
function osLeaf(budget: number): Task {
  const strategy: LoopStrategy = {
    kind: "react",
    budget: { kind: "per_loop", value: budget },
    behavior: { kind: "fail" },
    agent: "",
    toolset: "",
    output: "",
  };
  return newTask("produce a status report", SessionId.of("output-schema-session"), strategy);
}

function userMessages(state: SessionState): string[] {
  return state.messages
    .filter((m) => m.role === "user" && m.content.type === "text")
    .map((m) => (m.content.type === "text" ? m.content.text : ""));
}

// ── AC1: schema delivered to the directive seed ──────────────────────────────

describe("Output-schema enforcement (#139) — AC1 delivery", () => {
  it("delivers the KEY-SORTED schema to the directive seed and accepts on turn 1", async () => {
    const a = makeAgent();
    a.push({ kind: "final_response", content: OS_VALID, usage: usage() });
    const r = await new StandardHarness(osConfig(a, 2)).run({ task: osLeaf(10) });
    expect(r.kind).toBe("success");
    if (r.kind !== "success") return;
    expect(r.turns).toBe(1);
    const users = userMessages(r.session_state ?? { messages: [], extras: {} });
    expect(users.some((m) => m.includes(EXPECTED_SCHEMA))).toBe(true);
    // The directive is delivered AFTER the task instruction.
    const directiveIdx = users.findIndex((m) => m.includes(EXPECTED_SCHEMA));
    const instructionIdx = users.findIndex((m) => m === "produce a status report");
    expect(instructionIdx).toBeGreaterThanOrEqual(0);
    expect(directiveIdx).toBeGreaterThan(instructionIdx);
  });
});

// ── AC2: accept + retry ──────────────────────────────────────────────────────

describe("Output-schema enforcement (#139) — AC2 accept + retry", () => {
  it("a valid terminal on turn 1 succeeds with NO feedback message", async () => {
    const a = makeAgent();
    a.push({ kind: "final_response", content: OS_VALID, usage: usage() });
    const r = await new StandardHarness(osConfig(a, 2)).run({ task: osLeaf(10) });
    expect(r.kind).toBe("success");
    if (r.kind !== "success") return;
    expect(r.output).toBe(OS_VALID);
    expect(r.turns).toBe(1);
    const fedBack = userMessages(r.session_state ?? { messages: [], extras: {} }).some((m) =>
      m.includes("did not match the required output schema"),
    );
    expect(fedBack).toBe(false);
  });

  it("turn 1 invalid → feed the frozen error back → turn 2 valid → Success (turns == 2)", async () => {
    const a = makeAgent();
    a.push({ kind: "final_response", content: OS_INVALID, usage: usage() });
    a.push({ kind: "final_response", content: OS_VALID, usage: usage() });
    const r = await new StandardHarness(osConfig(a, 2)).run({ task: osLeaf(10) });
    expect(r.kind).toBe("success");
    if (r.kind !== "success") return;
    expect(r.output).toBe(OS_VALID);
    expect(r.turns).toBe(2);
    // The frozen feedback for the FIRST-failing rule (missing required `status`,
    // array order) must be present, exact bytes.
    const expectedFeedback =
      "Your previous response did not match the required output schema. " +
      'Missing required property "status". Reply with only a JSON value that satisfies the schema.';
    const users = userMessages(r.session_state ?? { messages: [], extras: {} });
    expect(users.some((m) => m === expectedFeedback)).toBe(true);
  });
});

// ── AC3: fail after retries exhausted + budget precedence ────────────────────

describe("Output-schema enforcement (#139) — AC3 fail + budget precedence", () => {
  it("N+1 invalid terminals with a generous budget → output_schema_violation, distinct from budget", async () => {
    const a = makeAgent();
    for (let i = 0; i < 3; i += 1) {
      a.push({ kind: "final_response", content: OS_INVALID, usage: usage() });
    }
    const r = await new StandardHarness(osConfig(a, 2)).run({ task: osLeaf(50) });
    expect(r.kind).toBe("failure");
    if (r.kind !== "failure") return;
    expect(r.reason.kind).toBe("output_schema_violation");
    if (r.reason.kind !== "output_schema_violation") return;
    expect(r.reason.attempts).toBe(3); // 1 + N == 1 + 2
    expect(r.turns).toBe(3); // exactly 1 + N turns
    expect(r.turns).toBeLessThan(50); // budget NOT exhausted (distinct)
    expect(r.reason.last_error).toBe('Missing required property "status".');
    expect(r.reason.schema).toBe(EXPECTED_SCHEMA);
  });

  it("budget precedence: a retry colliding with a tiny turn cap surfaces budget_exceeded", async () => {
    const a = makeAgent();
    for (let i = 0; i < 5; i += 1) {
      a.push({ kind: "final_response", content: OS_INVALID, usage: usage() });
    }
    // N == 5 (large), budget == 2 turns. After 2 invalid terminals the 3rd retry
    // re-enters the loop where the turn-budget gate fires first.
    const r = await new StandardHarness(osConfig(a, 5)).run({ task: osLeaf(2) });
    expect(r.kind).toBe("failure");
    if (r.kind !== "failure") return;
    expect(r.reason.kind).toBe("budget_exceeded");
    expect(r.turns).toBe(2);
  });
});

// ── AC4: flag OFF keeps the invalid terminal as success ──────────────────────

describe("Output-schema enforcement (#139) — AC4 migration gate OFF", () => {
  it("an INVALID terminal is accepted as Success and NO directive is seeded", async () => {
    const a = makeAgent();
    a.push({ kind: "final_response", content: OS_INVALID, usage: usage() });
    // Enforcement OFF; register the schema anyway to prove it is IGNORED.
    const cfg: HarnessConfig = {
      registry: osRegistry(a),
      toolRegistry: new ScriptedToolRegistry(),
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      escalationMode: autonomous,
      // enforceOutputSchemas omitted ⇒ OFF.
    };
    const r = await new StandardHarness(cfg).run({ task: osLeaf(10) });
    expect(r.kind).toBe("success");
    if (r.kind !== "success") return;
    expect(r.output).toBe(OS_INVALID);
    expect(r.turns).toBe(1);
    const seeded = userMessages(r.session_state ?? { messages: [], extras: {} }).some((m) =>
      m.includes("conforms to this JSON schema"),
    );
    expect(seeded).toBe(false);
  });
});

// ── Stream events ────────────────────────────────────────────────────────────

describe("Output-schema enforcement (#139) — stream events", () => {
  it("emits the retry then the violation event", async () => {
    const a = makeAgent();
    for (let i = 0; i < 2; i += 1) {
      a.push({ kind: "final_response", content: OS_INVALID, usage: usage() });
    }
    const events: HarnessStreamEvent[] = [];
    // N == 1 ⇒ 1 retry then violation (2 attempts).
    await new StandardHarness(osConfig(a, 1)).run({
      task: osLeaf(50),
      on_stream: (e) => events.push(e),
    });
    const retries = events
      .filter((e) => e.kind === "output_schema_retry")
      .map((e) => (e.kind === "output_schema_retry" ? e.attempt : -1));
    const violations = events
      .filter((e) => e.kind === "output_schema_violation")
      .map((e) => (e.kind === "output_schema_violation" ? e.attempts : -1));
    expect(retries).toEqual([1]);
    expect(violations).toEqual([2]);
  });
});
