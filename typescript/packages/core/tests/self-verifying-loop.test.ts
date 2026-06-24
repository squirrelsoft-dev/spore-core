/**
 * Unit tests for the SelfVerifying loop strategy (issue #61).
 *
 * Mirrors the inline `run_self_verifying` tests in
 * `rust/crates/spore-core/src/harness.rs` — same rules, same verdicts.
 *
 * Rules covered (R1–R11):
 *   R1  build loop runs to agent-done.
 *   R2  evaluate uses a FRESH session id distinct from build.
 *   R3  evaluate read-only sandbox (scripted Write → read_only_violation;
 *       build sandbox unaffected).
 *   R4  role-evaluator chunk present in the evaluate seed (presence-only).
 *   R5  Default-FAIL indeterminate evaluator → loop continues.
 *   R6  fail iter0 reason X / pass iter1 → iter1 build context contains X,
 *       final Success.
 *   R7  always-Fail verifier → exactly maxIterations cycles → exhausted.
 *   R8  budgets fold both phases.
 *   R9  build vs evaluate distinguishable (distinct session ids).
 *   R10 SelfVerifying no longer returns strategy_not_yet_implemented.
 *   R11 verifier absent → self_verify_misconfigured (not a throw).
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyResponse,
  EmptyToolRegistry,
  ExecutionRegistry,
  MockAgent,
  ReadOnlySandbox,
  SessionId,
  StandardHarness,
  loopStrategyFromJson,
  loopStrategyToJson,
  newTask,
  type Agent,
  type Context,
  type HarnessConfig,
  type HumanRequest,
  type LoopStrategy,
  type SandboxViolation,
  type SelfVerifyingConfig,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
} from "../src/index.js";
import { InMemoryChunkProvider, promptChunk } from "../src/prompt-assembly/index.js";
import type { Verifier, VerifierInput, VerifierVerdict } from "../src/verifier/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedMiddleware,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

const SV_STRATEGY: LoopStrategy = {
  kind: "self_verifying",
  inner: {
    kind: "react",
    // #124: under genuine recursion the build phase is the inner ReAct's own
    // loop; a `per_loop` cap of 1 would stop it after a single turn. MAX lets the
    // build run until the worker claims done (mirrors Rust's react_structured).
    budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
    agent: "",
    toolset: "",
    output: "",
  },
  evaluator: "",
};

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function fr(content: string): TurnResult {
  return { kind: "final_response", content, usage: usage() };
}

function tcr(name = "x"): TurnResult {
  const call: ToolCall = { id: `${Math.random()}`, name, input: {} };
  return { kind: "tool_call_requested", calls: [call], usage: usage() };
}

/** An agent that always claims done with a fixed final response. Records the
 *  full sequence of assembled contexts so seed/injection assertions are exact. */
class AlwaysDoneAgent implements Agent {
  ran = 0;
  readonly contexts: Context[] = [];
  constructor(
    private readonly agentId: AgentId,
    private readonly output = "done",
  ) {}
  id(): AgentId {
    return this.agentId;
  }
  async turn(ctx: Context, _signal?: AbortSignal): Promise<TurnResult> {
    this.ran += 1;
    this.contexts.push(ctx);
    return fr(this.output);
  }
}

/** A queue-driven agent: pops the next scripted result per turn. */
class ScriptedAgent implements Agent {
  ran = 0;
  readonly contexts: Context[] = [];
  private readonly results: TurnResult[] = [];
  constructor(private readonly agentId: AgentId) {}
  push(r: TurnResult): this {
    this.results.push(r);
    return this;
  }
  id(): AgentId {
    return this.agentId;
  }
  async turn(ctx: Context, _signal?: AbortSignal): Promise<TurnResult> {
    this.ran += 1;
    this.contexts.push(ctx);
    const next = this.results.shift();
    if (next == null) return { kind: "error", error: new EmptyResponse(), usage: null };
    return next;
  }
}

/** Scripts a sequence of verdicts; records every VerifierInput it sees. */
class ScriptedVerifier implements Verifier {
  readonly seen: VerifierInput[] = [];
  private i = 0;
  constructor(
    private readonly verdicts: VerifierVerdict[],
    private readonly _maxIterations: number,
  ) {}
  async verify(input: VerifierInput, _signal?: AbortSignal): Promise<VerifierVerdict> {
    this.seen.push(input);
    const v = this.verdicts[this.i] ?? { kind: "failed", reason: "no more verdicts" };
    this.i += 1;
    return v;
  }
  maxIterations(): number {
    return this._maxIterations;
  }
}

function pass(): VerifierVerdict {
  return { kind: "passed" };
}
function fail(reason: string): VerifierVerdict {
  return { kind: "failed", reason };
}

function configWith(
  agent: Agent,
  overrides: Partial<HarnessConfig> & { verifier?: Verifier } = {},
): HarnessConfig {
  // #124 Q1: the evaluate-phase agent defaults to the inner worker's agent — so
  // ONE worker agent drives BOTH build and evaluate (no separate evaluatorAgent).
  // The verifier folds into the registry under the default "" key.
  const { verifier, ...rest } = overrides;
  return {
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    ...rest,
    registry: registryWith({ agent, verifier }),
  };
}

const SID = SessionId.of("build-session");

function svTask() {
  return newTask("implement the feature", SID, SV_STRATEGY, { max_turns: 10 });
}

/** Flatten a Context's text content for substring assertions. */
function contextText(ctx: Context): string {
  return ctx.messages
    .map((m) => {
      const c = m.content;
      if (Array.isArray(c)) return c.map((p) => ("text" in p ? p.text : "")).join(" ");
      if (typeof c === "object" && c != null && "text" in c) return (c as { text: string }).text;
      return typeof c === "string" ? c : "";
    })
    .join("\n");
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

describe("SelfVerifying loop strategy (issue #61)", () => {
  it("R10/R1: pass on first iteration → Success (no strategy_not_yet_implemented)", async () => {
    const agent = new AlwaysDoneAgent(AgentId.of("a"), "built it");
    const verifier = new ScriptedVerifier([pass()], 3);
    const h = new StandardHarness(configWith(agent, { verifier }));
    const r = await h.run({ task: svTask() });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.output).toBe("built it"); // build output reused
    }
    // exactly one build + one evaluate run (both used the agent).
    expect(verifier.seen.length).toBe(1);
  });

  it("R11 (#124): verifier absent → typed STARTUP halt (unresolved verifier handle), not a throw", async () => {
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const h = new StandardHarness(configWith(agent)); // no verifier registered
    const r = await h.run({ task: svTask() });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      // #124: resolution is the single path — validation rejects the unresolved
      // verifier handle at startup as a `configuration_error`.
      expect(r.reason.kind).toBe("configuration_error");
      if (r.reason.kind === "configuration_error") {
        expect(r.reason.error.kind).toBe("UnresolvedHandle");
      }
    }
    // never invoked the agent — it short-circuits before the build phase.
    expect(agent.ran).toBe(0);
  });

  it("R2/R9: evaluate run uses a FRESH session id distinct from the build session", async () => {
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const verifier = new ScriptedVerifier([pass()], 3);
    const h = new StandardHarness(configWith(agent, { verifier }));
    await h.run({ task: svTask() });
    const input = verifier.seen[0]!;
    expect(input.build_result.kind).toBe("success");
    expect(input.eval_result.kind).toBe("success");
    if (input.build_result.kind === "success" && input.eval_result.kind === "success") {
      const buildSid = input.build_result.session_id.asString();
      const evalSid = input.eval_result.session_id.asString();
      expect(buildSid).toBe("build-session");
      expect(evalSid).not.toBe(buildSid); // R9: distinguishable
      expect(evalSid.startsWith("sess-")).toBe(true); // freshly generated
    }
  });

  it("R3: evaluate runs over a read-only sandbox; the evaluate write is blocked, the build write is not", async () => {
    // #124 Q1c: ONE worker agent drives both phases. The build phase writes
    // (allowed by the writable sandbox) then claims done; the evaluate phase
    // (same agent, read-only sandbox) tries to write — that write MUST be
    // rejected — then claims a verdict. Exactly ONE read_only_violation appears.
    const verifier = new ScriptedVerifier([pass()], 3);
    const worker = new ScriptedAgent(AgentId.of("worker"))
      .push(tcr("write_file")) // build: write allowed
      .push(fr("done")) // build: done
      .push(tcr("write_file")) // evaluate: write rejected (read-only)
      .push(fr("review done")); // evaluate: verdict

    const h = new StandardHarness(configWith(worker, { verifier }));
    const r = await h.run({ task: svTask() });

    expect(r.kind).toBe("success");
    // The worker took 4 turns (2 build + 2 evaluate).
    expect(worker.ran).toBe(4);
    // Exactly the evaluate-phase write (over the read-only sandbox) surfaces a
    // recoverable read_only_violation fed back as a tool result; the build-phase
    // write (over the writable sandbox) was NOT rejected.
    const violations = worker.contexts
      .map((ctx) => contextText(ctx))
      .join("\n")
      .match(/read_only_violation/g);
    expect(violations?.length ?? 0).toBe(1);
  });

  it("R3 (direct): ReadOnlySandbox blocks mutating tools, delegates reads", async () => {
    const inner = new AllowAllSandbox();
    const ro = new ReadOnlySandbox(inner);
    const w = (name: string): ToolCall => ({ id: "1", name, input: {} });
    const writeV = (await ro.validate(w("write_file"))) as SandboxViolation;
    expect(writeV?.kind).toBe("read_only_violation");
    const editV = (await ro.validate(w("edit_file"))) as SandboxViolation;
    expect(editV?.kind).toBe("read_only_violation");
    // A read tool delegates to inner (AllowAll → null).
    expect(await ro.validate(w("read_file"))).toBeNull();
  });

  it("R4: the role-evaluator chunk content is present in the evaluate seed (presence-only)", async () => {
    // #124 Q1c: ONE worker agent; the evaluate-phase turn (its 2nd turn) seeds
    // the `role-evaluator` chunk marker. The build turn (1st) does not.
    const verifier = new ScriptedVerifier([pass()], 3);
    const worker = new ScriptedAgent(AgentId.of("worker"))
      .push(fr("done")) // build turn
      .push(fr("reviewed")); // evaluate turn
    const chunkProvider = new InMemoryChunkProvider([
      promptChunk("role-evaluator", "YOU-ARE-A-FRESH-EVALUATOR-MARKER"),
    ]);
    const h = new StandardHarness(configWith(worker, { verifier, chunkProvider }));
    await h.run({ task: svTask() });
    expect(worker.contexts.length).toBe(2);
    const evalSeed = contextText(worker.contexts[1]!);
    expect(evalSeed).toContain("YOU-ARE-A-FRESH-EVALUATOR-MARKER");
  });

  it("R5: Default-FAIL — an indeterminate evaluator keeps looping then exhausts", async () => {
    const builder = new AlwaysDoneAgent(AgentId.of("builder"));
    // Two indeterminate (failed) verdicts, cap 2 → exhausts.
    const verifier = new ScriptedVerifier([fail("indeterminate"), fail("indeterminate")], 2);
    const h = new StandardHarness(configWith(builder, { verifier }));
    const r = await h.run({ task: svTask() });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("self_verify_exhausted");
      if (r.reason.kind === "self_verify_exhausted") {
        expect(r.reason.iterations).toBe(2);
      }
    }
    expect(verifier.seen.length).toBe(2);
  });

  it("R6: fail iter0 with reason X / pass iter1 → iter1 build context contains X, final Success", async () => {
    // Builder records contexts: iter0 build turn, then evaluator turn, then iter1
    // build turn. We assert the iter1 build context contains the injected reason.
    const builder = new ScriptedAgent(AgentId.of("builder"))
      .push(fr("attempt 1")) // iter0 build
      .push(fr("eval 0")) // iter0 evaluate (default-agent evaluator)
      .push(fr("attempt 2")) // iter1 build (must see injected reason)
      .push(fr("eval 1")); // iter1 evaluate
    const verifier = new ScriptedVerifier([fail("ADD-A-NULL-CHECK"), pass()], 3);
    const h = new StandardHarness(configWith(builder, { verifier }));
    const r = await h.run({ task: svTask() });
    expect(r.kind).toBe("success");
    // The iter1 build turn is the 3rd recorded context (index 2).
    const iter1Build = contextText(builder.contexts[2]!);
    expect(iter1Build).toContain("ADD-A-NULL-CHECK");
  });

  it("R7: always-Fail verifier → exactly maxIterations cycles → self_verify_exhausted", async () => {
    const builder = new AlwaysDoneAgent(AgentId.of("builder"));
    const verifier = new ScriptedVerifier([fail("a"), fail("b"), fail("c")], 3);
    const h = new StandardHarness(configWith(builder, { verifier }));
    const r = await h.run({ task: svTask() });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "self_verify_exhausted") {
      expect(r.reason.iterations).toBe(3);
      expect(r.reason.last_reason).toBe("c"); // last verdict reason
    } else {
      expect.unreachable("expected self_verify_exhausted");
    }
    expect(verifier.seen.length).toBe(3); // exactly maxIterations round-trips
  });

  it("R8: budgets fold BOTH build and evaluate usage across all iterations", async () => {
    // 2 iterations (fail then pass). Each iteration: 1 build turn + 1 evaluate
    // turn, every turn = 1 input + 1 output token. So 4 turns ⇒ 4 input/4 output.
    const builder = new AlwaysDoneAgent(AgentId.of("builder"));
    const verifier = new ScriptedVerifier([fail("again"), pass()], 3);
    const h = new StandardHarness(configWith(builder, { verifier }));
    const r = await h.run({ task: svTask() });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.usage.input_tokens).toBe(4);
      expect(r.usage.output_tokens).toBe(4);
    }
  });

  it("self_verifying_runs_non_react_inner_worker (#124): the inner PlanExecute drives the build per iteration", async () => {
    // #124: the SelfVerifying build phase GENUINELY recurses into `inner`. With a
    // non-ReAct inner (PlanExecute[ReAct, ReAct]) the inner plan turn must fire
    // per iteration. Worker turns for ONE iteration over a 1-task plan: plan JSON,
    // execute step, then the evaluate-phase turn.
    const worker = new ScriptedAgent(AgentId.of("worker"))
      .push(fr('{"tasks":["only"],"rationale":"r"}')) // inner plan turn
      .push(fr("did the step")) // inner execute step
      .push(fr("PASS")); // evaluate phase
    const verifier = new ScriptedVerifier([pass()], 3);
    const strategy: LoopStrategy = {
      kind: "self_verifying",
      inner: {
        kind: "plan_execute",
        plan: {
          kind: "react",
          budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
          agent: "",
          toolset: "",
          output: "",
        },
        execute: {
          kind: "react",
          budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
          agent: "",
          toolset: "",
        },
      },
      evaluator: "",
    };
    const h = new StandardHarness(configWith(worker, { verifier }));
    const r = await h.run({ task: newTask("build a CLI", SID, strategy, { max_turns: 50 }) });
    expect(r.kind).toBe("success");
    // The inner PlanExecute fired its plan turn ⇒ the worker saw the plan
    // directive. A hardcoded-ReAct build would record ZERO plan turns.
    const planTurns = worker.contexts.filter((ctx) =>
      contextText(ctx).includes("step-by-step plan"),
    ).length;
    expect(planTurns).toBeGreaterThanOrEqual(1);
    // The verifier fired (the SelfVerifying loop ran its evaluate phase).
    expect(verifier.seen.length).toBe(1);
  });

  // ── eval-phase reviewer-context regression tests (#151) ──────────────────

  // BLOCKER FIX (#151): the evaluate phase must NOT inherit the caller's approval
  // middleware. A nested review run is non-interactive (the caller never sees its
  // generated session id), so a `surface_to_human` before_tool decision would
  // pause it with no human able to resume — the reviewer's first tool call would
  // never dispatch and the review half would silently never run. With the caller
  // middleware dropped for the read-only eval phase, the reviewer's read
  // dispatches and the run reaches a verdict.
  it("self_verifying_eval_phase_drops_caller_middleware", async () => {
    // Build emits a tool-free final_response (so the BUILD phase, which DOES run
    // the caller middleware, never trips it); the eval phase (same worker per
    // Q1c) then issues a read — the call the middleware would have gated.
    const worker = new ScriptedAgent(AgentId.of("worker"))
      .push(fr("built")) // build: tool-free done
      .push(tcr("read_file")) // evaluate: the gated read
      .push(fr("reviewed: PASS")); // evaluate: verdict after dispatch
    const verifier = new ScriptedVerifier([pass()], 3);
    // The dispatch count is the discriminator: it only increments if the eval
    // phase's tool call actually dispatched (i.e. was NOT paused at before_tool).
    const reg = new ScriptedToolRegistry();
    // The caller's approval middleware: surface_to_human at before_tool. Without
    // the eval-phase drop, it pauses the reviewer's read and the run never
    // dispatches the tool.
    const request: HumanRequest = {
      kind: "tool_approval",
      calls: [{ id: "c1", name: "read_file", input: {} }],
      risk_level: "medium",
    };
    const middleware = new ScriptedMiddleware().push("before_tool", {
      kind: "surface_to_human",
      request,
    });
    const h = new StandardHarness(configWith(worker, { verifier, toolRegistry: reg, middleware }));
    const r = await h.run({ task: svTask() });
    expect(r.kind).toBe("success");
    // Load-bearing: the reviewer's read actually dispatched. With the caller
    // middleware NOT dropped, the eval phase pauses at before_tool BEFORE
    // dispatch and this count stays 0.
    expect(reg.callCount).toBeGreaterThanOrEqual(1);
  });

  // The evaluate phase runs the configured `evalAgent` (a dedicated reviewer),
  // not the inner worker's agent — so the model reviewing the work is not the one
  // that wrote it (a builder reviewing itself tends to rubber-stamp).
  it("eval_agent_override_runs_distinct_reviewer", async () => {
    const build = new MockAgent(AgentId.of("builder")).push(fr("built"));
    const reviewer = new MockAgent(AgentId.of("reviewer")).push(fr("reviewed"));
    const verifier = new ScriptedVerifier([pass()], 3);
    // A registry with BOTH the default worker (under "") and a separate "reviewer"
    // agent; the SelfVerifying `evalAgent` handle resolves to the latter.
    const registry = ExecutionRegistry.builder()
      .agent("", build)
      .agent("reviewer", reviewer)
      .toolset("", new EmptyToolRegistry())
      .schema("", {})
      .verifier("", verifier)
      .build();
    const config: HarnessConfig = {
      toolRegistry: new ScriptedToolRegistry(),
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      registry,
    };
    const strategy: LoopStrategy = {
      kind: "self_verifying",
      inner: {
        kind: "react",
        budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
        agent: "",
        toolset: "",
        output: "",
      },
      evaluator: "",
      evalAgent: "reviewer",
    };
    const h = new StandardHarness(config);
    const r = await h.run({
      task: newTask("implement the feature", SID, strategy, { max_turns: 10 }),
    });
    expect(r.kind).toBe("success");
    // The reviewer ran the evaluate phase exactly once; the builder ran only the
    // build phase. Without the override the worker would serve BOTH and the
    // reviewer would never be called.
    expect(reviewer.callCount).toBe(1);
    expect(build.callCount).toBe(1);
  });

  // Wire-format contract (cross-language parity, #151): the two optional eval-phase
  // overrides are OMITTED when unset (existing configs stay byte-identical) and
  // serialize as bare string handles in declaration order (evaluator < evalAgent <
  // evalToolset < behavior) when set, round-tripping intact.
  it("config_eval_overrides_serde", () => {
    // Unset ⇒ both override fields are OMITTED from the wire (existing configs +
    // fixtures stay byte-identical).
    const bare: LoopStrategy = {
      kind: "self_verifying",
      inner: { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
      evaluator: "v",
    };
    const bareJson = loopStrategyToJson(bare);
    expect("evalAgent" in bareJson).toBe(false);
    expect("evalToolset" in bareJson).toBe(false);

    // Set ⇒ both serialize as BARE STRING handles, in declaration order.
    const withOverrides: SelfVerifyingConfig = {
      kind: "self_verifying",
      inner: {
        kind: "react",
        budget: { kind: "per_loop", value: 1 },
        behavior: { kind: "escalate" },
        agent: "",
        toolset: "",
      },
      evaluator: "v",
      evalAgent: "reviewer",
      evalToolset: "ro-tools",
      behavior: { kind: "escalate" },
    };
    const json = loopStrategyToJson(withOverrides);
    expect(json.evalAgent).toBe("reviewer"); // bare string handle
    expect(json.evalToolset).toBe("ro-tools"); // bare string handle

    // Field order locked for parity: evaluator < evalAgent < evalToolset <
    // behavior. `behavior` appears twice (inner ReAct + this config); the
    // SelfVerifying one is LAST, so `lastIndexOf` targets it.
    const s = JSON.stringify(json);
    const ev = s.indexOf("evaluator");
    const ea = s.indexOf("evalAgent");
    const et = s.indexOf("evalToolset");
    const bh = s.lastIndexOf("behavior");
    expect(ev).toBeGreaterThanOrEqual(0);
    expect(ev).toBeLessThan(ea);
    expect(ea).toBeLessThan(et);
    expect(et).toBeLessThan(bh);

    // Round-trips intact.
    const back = loopStrategyFromJson(JSON.parse(s));
    expect(back).toEqual(withOverrides);
  });

  // #147: the SelfVerifying combinator must charge the EVALUATOR's turns against
  // its budget scope, not just the build phase. A 2-iteration loop (build, eval,
  // build, eval = 4 turns) succeeds at a cap of 4 (just fits all 4 turns) but is
  // EXHAUSTED at a cap of 3 — the second iteration's evaluator turn overruns it.
  // The two build turns alone are only 2, so without charging the evaluator the
  // cap-3 run would wrongly succeed.
  async function runSvTwoItersWithCap(cap: number): Promise<{ kind: string }> {
    // One worker drives both phases (Q1c); each phase is a single turn.
    const worker = new ScriptedAgent(AgentId.of("worker"))
      .push(fr("build0"))
      .push(fr("eval0"))
      .push(fr("build1"))
      .push(fr("eval1"));
    // Iter 0 fails (loop continues), iter 1 passes.
    const verifier = new ScriptedVerifier([fail("retry"), pass()], 3);
    const h = new StandardHarness(configWith(worker, { verifier }));
    return h.run({ task: newTask("implement the feature", SID, SV_STRATEGY, { max_turns: cap }) });
  }

  it("self_verifying_charges_evaluator_turns_against_scope", async () => {
    const fits = await runSvTwoItersWithCap(4);
    expect(fits.kind).toBe("success");
    // cap=3 must be exhausted once evaluator turns are charged (without the #147
    // charge it would wrongly succeed, since the two build turns alone are 2 <= 3).
    const exhausted = await runSvTwoItersWithCap(3);
    expect(exhausted.kind).not.toBe("success");
  });
});
