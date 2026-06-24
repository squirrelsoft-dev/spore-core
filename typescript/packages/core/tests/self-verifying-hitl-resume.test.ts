/**
 * Regression tests for SC-BUG-1 (issue #156): a HITL approval / clarification /
 * deny resume raised INSIDE a SelfVerifying frame must RE-ENTER the frame, not
 * run only the bare build leaf.
 *
 * Mirrors the Rust `self_verifying_hitl_{resume,deny,clarification}_reenters_eval_frame`
 * tests in `rust/crates/spore-core/src/harness.rs` — same scenario, same
 * load-bearing assertion (the eval-phase verifier runs >= 1 time after resume;
 * before the fix it ran 0 times because the resume returned the bare leaf's
 * Success directly).
 *
 * The original run pauses on the BUILD phase's first tool call. Before the fix,
 * `resumeInner` ran `runReact` on the paused ReAct leaf and returned its Success
 * — the evaluate phase + verifier never ran, so the looper's eval-frame reviewer
 * (SC-30 read-only eval toolset) was silently skipped. After the fix the pause
 * carries the FULL composed task (`finishCombinator` rewrites it on the way up,
 * exactly as it already does for `consult`) and the resume re-drives the whole
 * SelfVerifying strategy from the approved/answered worker session, so the
 * verifier runs.
 *
 * The EVAL-phase worker turn is scripted TOOL-FREE (a plain `final_response`) so
 * it never re-trips the approval gate — keeping the test independent of SC-29
 * (eval-phase caller-middleware drop) parity.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  MockAgent,
  SessionId,
  StandardHarness,
  newTask,
  type HarnessConfig,
  type HumanRequest,
  type LoopStrategy,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
} from "../src/index.js";
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
// Helpers (mirror the self-verifying-loop test scaffolding)
// --------------------------------------------------------------------------

const SID = SessionId.of("build-session");

const SV_STRATEGY: LoopStrategy = {
  kind: "self_verifying",
  inner: {
    kind: "react",
    // The build phase is the inner ReAct's own loop; let it run until the worker
    // claims done (mirrors the self-verifying-loop test's SV_STRATEGY).
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

function finalResp(content: string): TurnResult {
  return { kind: "final_response", content, usage: usage() };
}

function toolCall(name: string): TurnResult {
  const call: ToolCall = { id: "c1", name, input: {} };
  return { kind: "tool_call_requested", calls: [call], usage: usage() };
}

function pass(): VerifierVerdict {
  return { kind: "passed" };
}

/** Scripts a sequence of verdicts; records every VerifierInput it sees in
 *  `.seen` (the load-bearing "did the eval phase run?" counter). */
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

/**
 * The build worker drives BOTH phases (#124 Q1c): build emits a tool call (gated
 * → pause) then a `final_response`; the eval phase emits a TOOL-FREE
 * `final_response` so it never re-trips the gate. `buildToolName` is the gated
 * tool; the eval verdict text is unused at the wire level (the verifier owns the
 * verdict).
 */
function buildWorker(buildToolName: string): MockAgent {
  return new MockAgent(AgentId.of("worker"))
    .push(toolCall(buildToolName)) // build: tool call (gated → pause)
    .push(finalResp("built")) // build: done (after the resume dispatches the call)
    .push(finalResp("reviewed: PASS")); // evaluate: tool-free verdict turn
}

function svConfig(worker: MockAgent, verifier: Verifier): HarnessConfig {
  const cfg: HarnessConfig = {
    registry: registryWith({ agent: worker, verifier }),
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
  };
  return cfg;
}

function svTask() {
  return newTask("implement the feature", SID, SV_STRATEGY, { max_turns: 50 });
}

/** A caller approval middleware: SurfaceToHuman at BeforeTool for the given
 *  gated tool call (AlwaysAsk-style). */
function approvalMiddleware(
  toolName: string,
  riskLevel: "low" | "medium" | "high",
): ScriptedMiddleware {
  const request: HumanRequest = {
    kind: "tool_approval",
    calls: [{ id: "c1", name: toolName, input: {} }],
    risk_level: riskLevel,
  };
  return new ScriptedMiddleware().push("before_tool", { kind: "surface_to_human", request });
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

describe("SC-BUG-1 (#156): HITL resume re-enters the SelfVerifying frame", () => {
  // resume (Allow) arm: an approval pause on the BUILD phase's first tool call;
  // resuming with Allow must re-enter the SelfVerifying frame so the eval-phase
  // verifier runs (0 calls before the fix), and the approved tool dispatches.
  it("self_verifying_hitl_resume_reenters_eval_frame", async () => {
    const worker = buildWorker("write_file");
    const verifier = new ScriptedVerifier([pass()], 3);
    const reg = new ScriptedToolRegistry();
    const cfg = svConfig(worker, verifier);
    cfg.toolRegistry = reg;
    cfg.middleware = approvalMiddleware("write_file", "medium");
    const h = new StandardHarness(cfg);

    // 1) The run pauses on the build-phase tool call.
    const paused = await h.run({ task: svTask() });
    expect(paused.kind).toBe("waiting_for_human");
    if (paused.kind !== "waiting_for_human") return;
    const state = paused.state;
    // Part-1 check: the pause must carry the COMPOSED task (rewritten on the way
    // up by finishCombinator) so the resume can re-enter the frame — not the bare
    // ReAct build leaf.
    expect(state.task.loop_strategy.kind).toBe("self_verifying");

    // 2) Approve and resume.
    const resumed = await h.resume(state, { kind: "allow" });
    expect(resumed.kind).toBe("success");

    // Load-bearing: the eval-phase verifier ran AFTER the resume — the
    // SelfVerifying frame was re-entered. Before the fix this count stays 0.
    expect(verifier.seen.length).toBeGreaterThanOrEqual(1);
    // And the approved build tool actually dispatched on resume.
    expect(reg.callCount).toBeGreaterThanOrEqual(1);
  });

  // deny arm: a DENIED tool-approval resume must also re-enter the frame. Deny
  // appends a recoverable error tool result for the gated call and then re-drives
  // the strategy (the same final-match tail as Allow).
  it("self_verifying_hitl_deny_reenters_eval_frame", async () => {
    const worker = buildWorker("write_file");
    const verifier = new ScriptedVerifier([pass()], 3);
    const cfg = svConfig(worker, verifier);
    cfg.middleware = approvalMiddleware("write_file", "high");
    const h = new StandardHarness(cfg);

    const paused = await h.run({ task: svTask() });
    expect(paused.kind).toBe("waiting_for_human");
    if (paused.kind !== "waiting_for_human") return;
    expect(paused.state.task.loop_strategy.kind).toBe("self_verifying");

    const resumed = await h.resume(paused.state, { kind: "deny", reason: "not allowed" });
    expect(resumed.kind).toBe("success");
    // The eval-phase verifier ran after the DENY resume.
    expect(verifier.seen.length).toBeGreaterThanOrEqual(1);
  });

  // clarification (Answer) arm: the SEPARATE resumeInner clarification tail. The
  // build worker's tool returns awaiting_clarification; the human's Answer is
  // injected as that call's tool result and the strategy is re-driven, so the
  // evaluate phase runs. No approval middleware — the tool itself pauses.
  it("self_verifying_hitl_clarification_reenters_eval_frame", async () => {
    const worker = buildWorker("ask");
    const verifier = new ScriptedVerifier([pass()], 3);
    const cfg = svConfig(worker, verifier);
    const reg = new ScriptedToolRegistry();
    reg.push({ kind: "awaiting_clarification", question: "which file?" });
    cfg.toolRegistry = reg;
    const h = new StandardHarness(cfg);

    const paused = await h.run({ task: svTask() });
    expect(paused.kind).toBe("waiting_for_human");
    if (paused.kind !== "waiting_for_human") return;
    expect(paused.request.kind).toBe("clarification");
    expect(paused.state.task.loop_strategy.kind).toBe("self_verifying");

    const resumed = await h.resume(paused.state, { kind: "answer", text: "the config file" });
    expect(resumed.kind).toBe("success");
    // The eval-phase verifier ran after the clarification resume.
    expect(verifier.seen.length).toBeGreaterThanOrEqual(1);
  });
});
