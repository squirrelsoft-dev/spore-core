/**
 * Harness ⇆ Stop-hook integration (spore-core issue #69, R12–R14, R26).
 *
 * Mirrors the Rust `harness.rs` Stop-hook tests: a registered `stop` hook fires
 * at the loop's completion gate; a `block` injects a reason and continues; an
 * all-`continue` terminates normally; and the per-run `maxStopBlocks` cap
 * terminates a perpetually-blocking hook.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  MockAgent,
  SessionId,
  StandardHarness,
  hooks,
  newTask,
  type HarnessConfig,
  type LoopStrategy,
  type TokenUsage,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

const { StandardHookChain, FunctionHook } = hooks;

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function react(max: number): LoopStrategy {
  return { kind: "re_act", max_iterations: max };
}

function baseConfig(agent: MockAgent): HarnessConfig {
  return {
    agent,
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
  };
}

describe("harness Stop hooks", () => {
  it("a Stop block injects + continues, then a continue terminates", async () => {
    // Agent: first turn final, second turn final.
    const agent = new MockAgent(AgentId.of("a"));
    agent
      .push({ kind: "final_response", content: "first", usage: usage() })
      .push({ kind: "final_response", content: "second", usage: usage() });

    // Stop hook: block once, then continue.
    let calls = 0;
    const chain = new StandardHookChain();
    chain.register(
      new FunctionHook("verify", ["stop"], () => {
        calls += 1;
        return calls === 1
          ? { decision: "block", reason: "not done yet" }
          : { decision: "continue" };
      }),
    );

    const cfg = baseConfig(agent);
    cfg.hooks = chain;
    const harness = new StandardHarness(cfg);
    const result = await harness.run({
      task: newTask("do it", SessionId.of("s1"), react(10)),
    });

    expect(result.kind).toBe("success");
    if (result.kind === "success") {
      expect(result.output).toBe("second");
      // Two turns ran: the block forced a second turn.
      expect(result.turns).toBe(2);
    }
    expect(calls).toBe(2);
  });

  it("all-continue terminates on the first completion", async () => {
    const agent = new MockAgent(AgentId.of("a"));
    agent.push({ kind: "final_response", content: "done", usage: usage() });

    const chain = new StandardHookChain();
    chain.register(new FunctionHook("ok", ["stop"], () => ({ decision: "continue" })));

    const cfg = baseConfig(agent);
    cfg.hooks = chain;
    const harness = new StandardHarness(cfg);
    const result = await harness.run({
      task: newTask("do it", SessionId.of("s1"), react(10)),
    });

    expect(result.kind).toBe("success");
    if (result.kind === "success") expect(result.output).toBe("done");
  });

  it("no hook chain terminates normally", async () => {
    const agent = new MockAgent(AgentId.of("a"));
    agent.push({ kind: "final_response", content: "done", usage: usage() });
    const harness = new StandardHarness(baseConfig(agent));
    const result = await harness.run({
      task: newTask("do it", SessionId.of("s1"), react(10)),
    });
    expect(result.kind).toBe("success");
  });

  it("terminates after maxStopBlocks consecutive blocks (R14)", async () => {
    // Agent always returns a final response; the hook always blocks. The cap
    // (3) bounds how many extra turns the block forces.
    const agent = new MockAgent(AgentId.of("a"));
    for (let i = 0; i < 10; i += 1) {
      agent.push({ kind: "final_response", content: `t${i}`, usage: usage() });
    }

    const chain = new StandardHookChain();
    chain.register(
      new FunctionHook("always", ["stop"], () => ({ decision: "block", reason: "again" })),
    );

    const cfg = baseConfig(agent);
    cfg.hooks = chain;
    cfg.maxStopBlocks = 3;
    const harness = new StandardHarness(cfg);
    const result = await harness.run({
      task: newTask("do it", SessionId.of("s1"), react(50)),
    });

    expect(result.kind).toBe("success");
    if (result.kind === "success") {
      // 3 blocks force 3 extra turns; the 4th completion gate hits the cap and
      // terminates → 4 turns total.
      expect(result.turns).toBe(4);
    }
  });
});
