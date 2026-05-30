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
  ReadOnlySandbox,
  SessionId,
  StandardHarness,
  newTask,
  type Agent,
  type Context,
  type HarnessConfig,
  type LoopStrategy,
  type SandboxProvider,
  type SandboxViolation,
  type SessionState,
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
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

const SV_STRATEGY: LoopStrategy = { kind: "self_verifying" };

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

function configWith(agent: Agent, overrides: Partial<HarnessConfig> = {}): HarnessConfig {
  return {
    agent,
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    ...overrides,
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

  it("R11: verifier absent → self_verify_misconfigured (typed halt, not a throw)", async () => {
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const h = new StandardHarness(configWith(agent)); // no verifier
    const r = await h.run({ task: svTask() });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("self_verify_misconfigured");
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

  it("R3: evaluate runs over a read-only sandbox; a write is blocked, the build sandbox is untouched", async () => {
    // The evaluate agent attempts a write_file tool call; the read-only sandbox
    // must reject it with read_only_violation. The build sandbox (AllowAllSandbox)
    // is never asked to validate a write — the build agent only claims done.
    const verifier = new ScriptedVerifier([pass()], 3);

    // Evaluator: requests a write on its first turn, then claims done.
    const evaluator = new ScriptedAgent(AgentId.of("evaluator"))
      .push(tcr("write_file"))
      .push(fr("review done"));
    const builder = new AlwaysDoneAgent(AgentId.of("builder"));

    // Spy sandbox wrapping AllowAll, to prove the build sandbox saw no write.
    const buildSandbox = new AllowAllSandbox();
    let buildValidatedWrite = false;
    const spyBuild: SandboxProvider = {
      async validate(call) {
        if (call.name === "write_file") buildValidatedWrite = true;
        return buildSandbox.validate(call);
      },
    };

    const h = new StandardHarness(
      configWith(builder, {
        evaluatorAgent: evaluator,
        verifier,
        sandbox: spyBuild,
      }),
    );
    const r = await h.run({ task: svTask() });

    expect(r.kind).toBe("success");
    // The evaluate write was blocked at the read-only sandbox — surfaced as a
    // recoverable tool error, so the evaluate run continued to "review done"
    // (the verifier still saw a successful eval run).
    expect(evaluator.ran).toBe(2);
    // The build sandbox never validated a write (evaluator ran on the read-only
    // decorator, not the build sandbox).
    expect(buildValidatedWrite).toBe(false);
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
    const verifier = new ScriptedVerifier([pass()], 3);
    const builder = new AlwaysDoneAgent(AgentId.of("builder"));
    const evaluator = new AlwaysDoneAgent(AgentId.of("evaluator"), "reviewed");
    const chunkProvider = new InMemoryChunkProvider([
      promptChunk("role-evaluator", "YOU-ARE-A-FRESH-EVALUATOR-MARKER"),
    ]);
    const h = new StandardHarness(
      configWith(builder, { evaluatorAgent: evaluator, verifier, chunkProvider }),
    );
    await h.run({ task: svTask() });
    expect(evaluator.contexts.length).toBeGreaterThan(0);
    const seed = contextText(evaluator.contexts[0]!);
    expect(seed).toContain("YOU-ARE-A-FRESH-EVALUATOR-MARKER");
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
});
