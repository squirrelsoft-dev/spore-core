/**
 * Consecutive-recoverable-tool-error breaker tests (spore-core issue #137).
 *
 * Mirrors the Rust `tel_*` test module in
 * `rust/crates/spore-core/src/harness.rs` plus the shared fixture-replay
 * integration test (`rust/.../tests/tool_error_loop_fixture_replay.rs`) — same
 * rules, same verdicts, parallel structure.
 *
 * Rules under test:
 *   - `HarnessConfig.errorLoopThreshold` (N, default 3); hard stop at 2N.
 *   - Per-tool `ErrorRun` loop-local counter:
 *       * identical-args recoverable error -> count += 1
 *       * args change OR first error -> fresh run at count 1
 *       * ANY success for the tool -> run removed (AC1 reset)
 *   - At N: ONE corrective USER message (enrichToolError schema+hint), AC2.
 *   - At 2N: stop -> resolve node `BudgetExhaustedBehavior`, terminal carries
 *     `HaltReason.tool_error_loop` (never budget_exceeded), AC3.
 *   - At BOTH thresholds: a HarnessStreamEvent + a ContextOperation, AC4.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  AgentId,
  ModelAgent,
  MockAgent,
  ReplayModelInterface,
  SessionId,
  StandardHarness,
  autonomous,
  newTask,
  surfaceToHuman,
  type BudgetExhaustedBehavior,
  type HarnessConfig,
  type HarnessStreamEvent,
  type LoopStrategy,
  type ProviderInfo,
  type Task,
  type TokenUsage,
  type ToolCall,
  type ToolSchema,
} from "../src/index.js";
import { InMemoryObservabilityProvider } from "../src/observability/in-memory.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

// ── helpers ──────────────────────────────────────────────────────────────────

const TEL_BAD_MSG = "missing required parameter `description`";

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function makeAgent(): MockAgent {
  return new MockAgent(AgentId.of("test"));
}

function badArgs(): Record<string, unknown> {
  return { task_list_id: "tl1" };
}

/** The malformed `add_task` call the weak model repeats. */
function badCall(args: Record<string, unknown>): ToolCall {
  return { id: "call_bad", name: "add_task", input: args };
}

/** Push `k` identical malformed `add_task` tool-call turns. */
function pushBad(a: MockAgent, k: number, args: Record<string, unknown>): void {
  for (let i = 0; i < k; i += 1) {
    a.push({ kind: "tool_call_requested", calls: [badCall(args)], usage: usage() });
  }
}

/** A tool registry that always returns the same recoverable error and advertises
 *  the `add_task` schema (so `enrichToolError` can render the schema+hint). */
function errRegistry(): ScriptedToolRegistry {
  const schema: ToolSchema = {
    name: "add_task",
    description: "add a task to a task list",
    input_schema: {
      type: "object",
      properties: {
        task_list_id: { type: "string" },
        description: { type: "string" },
      },
      required: ["task_list_id", "description"],
    },
  };
  return new ScriptedToolRegistry().alwaysRecoverableError(TEL_BAD_MSG).withSchema(schema);
}

/** A `surface_to_human` config — budget/error-loop escalation PAUSES (HITL). */
function surfaceConfig(agent: MockAgent): HarnessConfig {
  return {
    registry: registryWith({ agent }),
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    escalationMode: surfaceToHuman,
  };
}

/** An `autonomous` config — escalation PROPAGATES (AFK). */
function standardConfig(agent: MockAgent): HarnessConfig {
  return { ...surfaceConfig(agent), escalationMode: autonomous };
}

function react(max: number): Task {
  return newTask("add a task to the task list", SessionId.of("s1"), {
    kind: "react",
    budget: { kind: "per_loop", value: max },
    agent: "",
    toolset: "",
  });
}

function leaf(behavior: BudgetExhaustedBehavior, budget: number): Task {
  const strategy: LoopStrategy = {
    kind: "react",
    budget: { kind: "per_loop", value: budget },
    behavior,
    agent: "",
    toolset: "",
  };
  return newTask("add a task to the task list", SessionId.of("s1"), strategy);
}

function userMessages(
  messages: { role: string; content: { type: string; text?: string } }[],
): string[] {
  return messages
    .filter((m) => m.role === "user" && m.content.type === "text")
    .map((m) => m.content.text ?? "");
}

// ── AC1: success / args-change reset ─────────────────────────────────────────

describe("Tool-error-loop breaker (#137) — AC1 reset", () => {
  it("a success in the middle resets the counter; the breaker never trips", async () => {
    // error, error, SUCCESS, error, error -> 4 errors but the longest
    // identical-args run is 2 (< N), so no trip.
    const a = makeAgent();
    pushBad(a, 2, badArgs());
    a.push({
      kind: "tool_call_requested",
      calls: [{ id: "ok", name: "add_task", input: badArgs() }],
      usage: usage(),
    });
    pushBad(a, 2, badArgs());
    a.push({ kind: "final_response", content: "done", usage: usage() });

    const cfg = standardConfig(a);
    const reg = new ScriptedToolRegistry();
    for (const recoverable of [true, true, false, true, true]) {
      reg.push(
        recoverable
          ? { kind: "error", message: TEL_BAD_MSG, recoverable: true }
          : { kind: "success", content: "added" },
      );
    }
    cfg.toolRegistry = reg;
    cfg.errorLoopThreshold = 3;
    const r = await new StandardHarness(cfg).run({ task: react(20) });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.output).toBe("done");
  });

  it("an args change starts a fresh run, so different-args errors never trip", async () => {
    // error(argsX), error(argsY) -> the second is a FRESH run at count 1, so two
    // different-args errors never trip even at N == 2.
    const a = makeAgent();
    a.push({
      kind: "tool_call_requested",
      calls: [badCall({ task_list_id: "X" })],
      usage: usage(),
    });
    a.push({
      kind: "tool_call_requested",
      calls: [badCall({ task_list_id: "Y" })],
      usage: usage(),
    });
    a.push({ kind: "final_response", content: "stopped trying", usage: usage() });

    const cfg = standardConfig(a);
    cfg.toolRegistry = errRegistry();
    cfg.errorLoopThreshold = 2; // 2N == 4; longest identical run is 1.
    const r = await new StandardHarness(cfg).run({ task: react(20) });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.output).toBe("stopped trying");
  });
});

// ── AC2: exactly one corrective at N ─────────────────────────────────────────

describe("Tool-error-loop breaker (#137) — AC2 corrective injection", () => {
  it("injects exactly ONE corrective user message at N and never re-injects", async () => {
    // 4 identical errors (N=3 injects at the 3rd; 4th must not re-inject), then a
    // final response so the run ends cleanly (2N would be 6).
    const a = makeAgent();
    pushBad(a, 4, badArgs());
    a.push({ kind: "final_response", content: "gave up", usage: usage() });

    const cfg = standardConfig(a);
    cfg.toolRegistry = errRegistry();
    cfg.errorLoopThreshold = 3;
    const r = await new StandardHarness(cfg).run({ task: react(20) });
    expect(r.kind).toBe("success");
    if (r.kind !== "success") return;

    const users = userMessages(r.session_state.messages);
    const correctives = users.filter((m) => m.includes("Expected parameter schema"));
    expect(correctives.length).toBe(1);
    const corrective = correctives[0]!;
    expect(corrective).toContain(TEL_BAD_MSG); // carries the bare error
    expect(corrective).toContain('"required"'); // carries the parameter schema
    expect(corrective).toContain("correctly-typed JSON"); // carries the hint
  });
});

// ── AC3: 2N hard stop routed through BudgetExhaustedBehavior ──────────────────

describe("Tool-error-loop breaker (#137) — AC3 hard stop", () => {
  it("Fail behavior at 2N → Failure { tool_error_loop }, budget NOT fully burned", async () => {
    const a = makeAgent();
    pushBad(a, 8, badArgs()); // trip is at 2N == 6.
    const cfg = standardConfig(a);
    cfg.toolRegistry = errRegistry();
    cfg.errorLoopThreshold = 3;
    const r = await new StandardHarness(cfg).run({ task: leaf({ kind: "fail" }, 50) });
    expect(r.kind).toBe("failure");
    if (r.kind !== "failure") return;
    expect(r.reason.kind).toBe("tool_error_loop");
    if (r.reason.kind === "tool_error_loop") {
      expect(r.reason.tool).toBe("add_task");
      expect(r.reason.consecutive_errors).toBe(6); // 2N == 6
    }
    expect(r.turns).toBeLessThan(50); // budget NOT fully burned
  });

  it("Escalate (surface_to_human) → WaitingForHuman", async () => {
    const a = makeAgent();
    pushBad(a, 8, badArgs());
    const cfg = surfaceConfig(a);
    cfg.toolRegistry = errRegistry();
    cfg.errorLoopThreshold = 3;
    const r = await new StandardHarness(cfg).run({ task: leaf({ kind: "escalate" }, 50) });
    expect(r.kind).toBe("waiting_for_human");
  });

  it("Escalate (autonomous) → propagated Failure carrying tool_error_loop", async () => {
    const a = makeAgent();
    pushBad(a, 8, badArgs());
    const cfg = standardConfig(a); // autonomous
    cfg.toolRegistry = errRegistry();
    cfg.errorLoopThreshold = 3;
    const r = await new StandardHarness(cfg).run({ task: leaf({ kind: "escalate" }, 50) });
    expect(r.kind).toBe("failure");
    if (r.kind !== "failure") return;
    expect(r.reason.kind).toBe("tool_error_loop"); // NOT budget_exceeded
    if (r.reason.kind === "tool_error_loop") expect(r.reason.tool).toBe("add_task");
    expect(r.turns).toBeLessThan(50);
  });

  it("Continue grants one window, then a terminal carrying tool_error_loop", async () => {
    // Continue{max_continues:1, on_exhausted:Fail}: the first 2N trip grants a
    // continue (fresh window), the second 2N trip falls through to Fail with
    // tool_error_loop. 2N (6) + 2N (6) = 12 calls needed.
    const a = makeAgent();
    pushBad(a, 14, badArgs());
    const cfg = standardConfig(a);
    cfg.toolRegistry = errRegistry();
    cfg.errorLoopThreshold = 3;
    const behavior: BudgetExhaustedBehavior = {
      kind: "continue",
      max_continues: 1,
      on_exhausted: { kind: "fail" },
    };
    const r = await new StandardHarness(cfg).run({ task: leaf(behavior, 50) });
    expect(r.kind).toBe("failure");
    if (r.kind !== "failure") return;
    expect(r.reason.kind).toBe("tool_error_loop");
    expect(r.turns).toBeLessThan(50);
  });
});

// ── AC4: stream + observability events at both thresholds ─────────────────────

describe("Tool-error-loop breaker (#137) — AC4 events", () => {
  it("emits the detected pair at N and the broken pair at 2N", async () => {
    const a = makeAgent();
    pushBad(a, 8, badArgs());
    const obs = new InMemoryObservabilityProvider();
    const cfg = standardConfig(a);
    cfg.toolRegistry = errRegistry();
    cfg.errorLoopThreshold = 3;
    cfg.observability = obs;

    const captured: HarnessStreamEvent[] = [];
    const sessionId = SessionId.of("s1");
    await new StandardHarness(cfg).run({
      task: leaf({ kind: "fail" }, 50),
      on_stream: (ev) => captured.push(ev),
    });

    const detected = captured
      .filter((e) => e.kind === "tool_error_loop_detected" && e.tool === "add_task")
      .map((e) => (e as { consecutive_errors: number }).consecutive_errors);
    const broken = captured
      .filter((e) => e.kind === "tool_error_loop_broken" && e.tool === "add_task")
      .map((e) => (e as { consecutive_errors: number }).consecutive_errors);
    expect(detected).toEqual([3]); // one detected at N, count == N
    expect(broken).toEqual([6]); // one broken at 2N, count == 2N

    const spans = obs.contextSpans(sessionId);
    const obsDetected = spans
      .filter(
        (s) =>
          s.operation.kind === "tool_error_loop_detected" &&
          (s.operation as { tool_name: string }).tool_name === "add_task",
      )
      .map((s) => (s.operation as { consecutive_errors: number }).consecutive_errors);
    const obsBroken = spans
      .filter(
        (s) =>
          s.operation.kind === "tool_error_loop_broken" &&
          (s.operation as { tool_name: string }).tool_name === "add_task",
      )
      .map((s) => (s.operation as { consecutive_errors: number }).consecutive_errors);
    expect(obsDetected).toEqual([3]);
    expect(obsBroken).toEqual([6]);
  });
});

// ── Breaker disabled when threshold is 0 ─────────────────────────────────────

describe("Tool-error-loop breaker (#137) — disabled", () => {
  it("threshold 0 disables the breaker; the run completes", async () => {
    const a = makeAgent();
    pushBad(a, 5, badArgs());
    a.push({ kind: "final_response", content: "fin", usage: usage() });
    const cfg = standardConfig(a);
    cfg.toolRegistry = errRegistry();
    cfg.errorLoopThreshold = 0;
    const r = await new StandardHarness(cfg).run({ task: react(20) });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.output).toBe("fin");
  });
});

// ── Shared fixture replay — must match the Rust outcome ───────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const loopFixture = resolve(repoRoot, "fixtures/model_responses/harness/tool_error_loop.jsonl");

const provider: ProviderInfo = { name: "ollama", model_id: "fixture", context_window: 200_000 };

describe("Tool-error-loop fixture — tool_error_loop.jsonl (#137)", () => {
  it("the breaker hard-stops at 2N with the same outcome as Rust", async () => {
    const jsonl = readFileSync(loopFixture, "utf-8");
    const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
    const agent = new ModelAgent(AgentId.of("fixture-agent"), replay);

    // Every dispatch of the malformed `add_task` call returns the same
    // recoverable error, regardless of args.
    const toolRegistry = new ScriptedToolRegistry().alwaysRecoverableError(
      "missing required parameter `description`",
    );

    const config: HarnessConfig = {
      registry: registryWith({ agent }),
      toolRegistry,
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      // N == 3 → inject at 3 identical errors, hard-stop at 6.
      errorLoopThreshold: 3,
      // Autonomous so the leaf's Fail behavior produces a terminal Failure (not a
      // HITL pause) at the 2N hard stop.
      escalationMode: autonomous,
    };
    const harness = new StandardHarness(config);

    // A bare ReAct leaf with a generous budget (50) and Fail behavior — so the
    // ONLY thing that can stop the run early is the error-loop breaker.
    const task = newTask("add a task to the task list", SessionId.of("tool-error-loop-session"), {
      kind: "react",
      budget: { kind: "per_loop", value: 50 },
      behavior: { kind: "fail" },
      agent: "",
      toolset: "",
    });

    const r = await harness.run({ task });
    expect(r.kind).toBe("failure");
    if (r.kind !== "failure") return;
    expect(r.reason.kind).toBe("tool_error_loop");
    if (r.reason.kind === "tool_error_loop") {
      expect(r.reason.tool).toBe("add_task");
      expect(r.reason.consecutive_errors).toBe(6); // hard stop at 2N == 6
    }
    expect(r.session_id.asString()).toBe("tool-error-loop-session");
    // The breaker stopped EARLY: exactly 2N turns, far below the budget.
    expect(r.turns).toBe(6);
    expect(r.turns).toBeLessThan(50);
    // The breaker stops AT the 2N dispatch — the registry saw exactly 2N == 6
    // calls (the 7th fixture line is unused headroom).
    expect(toolRegistry.callCount).toBe(6);
  });
});
