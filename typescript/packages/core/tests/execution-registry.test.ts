/**
 * ExecutionRegistry — runtime resolution of serializable strategy handles
 * (Composable Execution A.3, issue #120).
 *
 * Cross-language parity with the Rust reference
 * (`rust/crates/spore-core/src/execution_registry.rs`) and the SHARED fixtures
 * (`fixtures/strategy/cordyceps_tree.json`,
 * `fixtures/harness/registry_errors.json` — authored by the Rust agent, NOT
 * modified here).
 *
 * Rules covered (one-to-one with the spec):
 *   - unresolved handle → startup error before the first turn
 *   - resume: round-trip a Task with refs through serde, re-resolve all
 *   - missing StrategyRef custom key → recoverable StrategyNotFound, no crash
 *   - registerStrategy then resolveStrategy(custom) → ok
 *   - EscalationMode present/selectable/readable on config
 *   - each resolve_* happy + miss; tree-walk over the cordyceps fixture
 *   - fixture-replay: round-trip registry_errors.json byte-identically
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyToolRegistry,
  ExecutionRegistry,
  HarnessBuilder,
  InvalidConfiguration,
  SessionId,
  StandardHarness,
  StrategyNotFound,
  UnresolvedHandle,
  autonomous,
  harnessErrorFromJson,
  loopStrategyFromJson,
  newTask,
  surfaceToHuman,
  type Agent,
  type ExecutionContext,
  type HarnessConfig,
  type LoopStrategy,
  type ModelInterface,
  type ModelRequest,
  type ModelResponse,
  type ReactConfig,
  type RunStrategy,
  type StrategyOutcome,
  type StrategyRef,
  type TurnResult,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
} from "../src/harness/testing.js";
import type { Context } from "../src/agent/types.js";
import type { Verifier, VerifierVerdict } from "../src/verifier/types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");

// ── Test-only stubs ─────────────────────────────────────────────────────────

class StubAgent implements Agent {
  async turn(_context: Context): Promise<TurnResult> {
    throw new Error("validate() must fail before any agent turn");
  }
  id(): AgentId {
    return AgentId.of("stub");
  }
}

class StubVerifier implements Verifier {
  async verify(): Promise<VerifierVerdict> {
    throw new Error("verifier not invoked in registry tests");
  }
  maxIterations(): number {
    return 3;
  }
}

class StubStrategy implements RunStrategy {
  async run(_cx: ExecutionContext): Promise<StrategyOutcome> {
    return { kind: "complete", output: "" };
  }
}

function reactLeaf(agent: string, toolset: string, output?: string): ReactConfig {
  return {
    kind: "react",
    budget: { kind: "per_loop", value: 4 },
    agent,
    toolset,
    output,
  };
}

function fullyWiredRegistry(): ExecutionRegistry {
  return ExecutionRegistry.builder()
    .agent("a1", new StubAgent())
    .toolset("t1", new EmptyToolRegistry())
    .schema("s1", { type: "object" })
    .verifier("v1", new StubVerifier())
    .build();
}

function cordycepsTree(): LoopStrategy {
  const raw = readFileSync(resolve(repoRoot, "fixtures/strategy/cordyceps_tree.json"), "utf-8");
  return loopStrategyFromJson(JSON.parse(raw));
}

// ── resolve_* happy path + miss ─────────────────────────────────────────────

describe("ExecutionRegistry resolution", () => {
  it("resolves each ref type on hit and returns undefined on miss", () => {
    const reg = fullyWiredRegistry();

    expect(reg.resolveAgent("a1")).toBeDefined();
    expect(reg.resolveAgent("nope")).toBeUndefined();

    expect(reg.resolveToolset("t1")).toBeDefined();
    expect(reg.resolveToolset("nope")).toBeUndefined();

    expect(reg.resolveSchema("s1")).toEqual({ type: "object" });
    expect(reg.resolveSchema("nope")).toBeUndefined();

    expect(reg.resolveVerifier("v1")).toBeDefined();
    expect(reg.resolveVerifier("nope")).toBeUndefined();
  });

  it("empty() is empty; a populated registry is not", () => {
    expect(ExecutionRegistry.empty().isEmpty()).toBe(true);
    expect(fullyWiredRegistry().isEmpty()).toBe(false);
  });

  it("builder is last-wins on a duplicate key", () => {
    const reg = ExecutionRegistry.builder().schema("s", { v: 1 }).schema("s", { v: 2 }).build();
    expect(reg.resolveSchema("s")).toEqual({ v: 2 });
  });

  it("toBuilder() copies entries without aliasing the source registry", () => {
    const base = ExecutionRegistry.builder().agent("a1", new StubAgent()).build();
    const extended = base.toBuilder().agent("a2", new StubAgent()).build();
    // The extension has both; the original is untouched.
    expect(extended.resolveAgent("a1")).toBeDefined();
    expect(extended.resolveAgent("a2")).toBeDefined();
    expect(base.resolveAgent("a2")).toBeUndefined();
  });
});

// ── resolveStrategy: built-in + custom + miss ────────────────────────────────

describe("ExecutionRegistry.resolveStrategy", () => {
  it("registerStrategy then resolveStrategy(custom) → ok", () => {
    const reg = ExecutionRegistry.empty();
    reg.registerStrategy("mine::Custom", new StubStrategy());

    const ref: StrategyRef = { kind: "custom", value: "mine::Custom" };
    const res = reg.resolveStrategy(ref);
    expect(res.kind).toBe("custom");
    if (res.kind === "custom") {
      expect(res.strategy).toBeInstanceOf(StubStrategy);
    }
  });

  it("a built-in ref resolves to the borrowed tree", () => {
    const reg = ExecutionRegistry.empty();
    const ref: StrategyRef = { kind: "built_in", value: reactLeaf("a1", "t1") };
    const res = reg.resolveStrategy(ref);
    expect(res.kind).toBe("built_in");
    if (res.kind === "built_in") {
      expect(res.strategy.kind).toBe("react");
    }
  });

  it("a missing custom key is a recoverable StrategyNotFound, not a crash", () => {
    const reg = ExecutionRegistry.empty();
    const ref: StrategyRef = { kind: "custom", value: "absent" };
    let caught: unknown;
    try {
      reg.resolveStrategy(ref);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(StrategyNotFound);
    expect((caught as StrategyNotFound).key).toBe("absent");
    expect((caught as StrategyNotFound).message).toBe("custom strategy not found: absent");
  });
});

// ── validate(): unresolved handle → UnresolvedHandle ─────────────────────────

describe("ExecutionRegistry.validate", () => {
  it("reports an unresolved agent handle", () => {
    const reg = ExecutionRegistry.empty();
    const task = newTask("do it", SessionId.generate(), reactLeaf("missing-agent", "t1"));
    expect(() => reg.validate(task)).toThrow(UnresolvedHandle);
    try {
      reg.validate(task);
    } catch (e) {
      expect(e).toBeInstanceOf(UnresolvedHandle);
      expect((e as UnresolvedHandle).handleKind).toBe("agent");
      expect((e as UnresolvedHandle).key).toBe("missing-agent");
    }
  });

  it("reports an unresolved toolset handle", () => {
    const reg = ExecutionRegistry.builder().agent("a1", new StubAgent()).build();
    const task = newTask("do it", SessionId.generate(), reactLeaf("a1", "missing-tools"));
    try {
      reg.validate(task);
      throw new Error("expected throw");
    } catch (e) {
      expect(e).toBeInstanceOf(UnresolvedHandle);
      expect((e as UnresolvedHandle).handleKind).toBe("toolset");
      expect((e as UnresolvedHandle).key).toBe("missing-tools");
    }
  });

  it("reports an unresolved schema handle", () => {
    const reg = ExecutionRegistry.builder()
      .agent("a1", new StubAgent())
      .toolset("t1", new EmptyToolRegistry())
      .build();
    const task = newTask("do it", SessionId.generate(), reactLeaf("a1", "t1", "missing-schema"));
    try {
      reg.validate(task);
      throw new Error("expected throw");
    } catch (e) {
      expect(e).toBeInstanceOf(UnresolvedHandle);
      expect((e as UnresolvedHandle).handleKind).toBe("schema");
      expect((e as UnresolvedHandle).key).toBe("missing-schema");
    }
  });

  it("passes a fully-wired ReAct leaf", () => {
    const reg = fullyWiredRegistry();
    const task = newTask("ok", SessionId.generate(), reactLeaf("a1", "t1", "s1"));
    expect(() => reg.validate(task)).not.toThrow();
  });
});

// ── tree-walk over the nested cordyceps fixture ──────────────────────────────

describe("ExecutionRegistry tree-walk over cordyceps_tree.json", () => {
  it("reports the FIRST unresolved handle depth-first (agent 'planner')", () => {
    const reg = ExecutionRegistry.empty();
    const task = newTask("nested", SessionId.generate(), cordycepsTree());
    try {
      reg.validate(task);
      throw new Error("expected throw");
    } catch (e) {
      expect(e).toBeInstanceOf(UnresolvedHandle);
      expect((e as UnresolvedHandle).handleKind).toBe("agent");
      expect((e as UnresolvedHandle).key).toBe("planner");
    }
  });

  it("passes when the whole tree is wired", () => {
    const reg = ExecutionRegistry.builder()
      .agent("planner", new StubAgent())
      .agent("executor", new StubAgent())
      .agent("ralph-agent", new StubAgent())
      .toolset("plan-tools", new EmptyToolRegistry())
      .toolset("exec-tools", new EmptyToolRegistry())
      .schema("exec-evaluator", {})
      .schema("plan-schema", {})
      .schema("worker-schema", {})
      .build();
    const task = newTask("nested", SessionId.generate(), cordycepsTree());
    expect(() => reg.validate(task)).not.toThrow();
  });
});

// ── A.5 output-contract enforcement (#124, Q3) ───────────────────────────────

describe("A.5 output-contract enforcement (#124)", () => {
  it("a bare ReAct in the structured `plan` slot without output is rejected", () => {
    const reg = ExecutionRegistry.builder()
      .agent("a1", new StubAgent())
      .toolset("t1", new EmptyToolRegistry())
      .build();
    const tree: LoopStrategy = {
      kind: "plan_execute",
      plan: reactLeaf("a1", "t1"), // output: undefined
      execute: reactLeaf("a1", "t1"),
    };
    const task = newTask("contract", SessionId.generate(), tree);
    try {
      reg.validate(task);
      throw new Error("expected throw");
    } catch (e) {
      expect(e).toBeInstanceOf(InvalidConfiguration);
      expect((e as InvalidConfiguration).message).toContain("plan");
    }
  });

  it("a bare ReAct worker slot (self_verifying.inner) without output is rejected", () => {
    const reg = ExecutionRegistry.builder()
      .agent("a1", new StubAgent())
      .toolset("t1", new EmptyToolRegistry())
      .schema("eval", {})
      .build();
    const tree: LoopStrategy = {
      kind: "self_verifying",
      inner: reactLeaf("a1", "t1"),
      evaluator: "eval",
    };
    const task = newTask("contract", SessionId.generate(), tree);
    try {
      reg.validate(task);
      throw new Error("expected throw");
    } catch (e) {
      expect(e).toBeInstanceOf(InvalidConfiguration);
      expect((e as InvalidConfiguration).message).toContain("worker");
    }
  });

  it("a bare ReAct propose slot (hill_climbing.inner) without output is rejected", () => {
    const reg = ExecutionRegistry.builder()
      .agent("a1", new StubAgent())
      .toolset("t1", new EmptyToolRegistry())
      .build();
    const tree: LoopStrategy = {
      kind: "hill_climbing",
      inner: reactLeaf("a1", "t1"),
      direction: "minimize",
      max_stagnation: 1,
      revert_on_no_improvement: false,
      min_improvement_delta: 0,
      evaluator: "a1",
    };
    const task = newTask("contract", SessionId.generate(), tree);
    try {
      reg.validate(task);
      throw new Error("expected throw");
    } catch (e) {
      expect(e).toBeInstanceOf(InvalidConfiguration);
      expect((e as InvalidConfiguration).message).toContain("propose");
    }
  });

  it("accepts a ReAct in a structured slot WITH an output schema", () => {
    const reg = ExecutionRegistry.builder()
      .agent("a1", new StubAgent())
      .toolset("t1", new EmptyToolRegistry())
      .schema("plan-schema", {})
      .build();
    const tree: LoopStrategy = {
      kind: "plan_execute",
      plan: reactLeaf("a1", "t1", "plan-schema"),
      execute: reactLeaf("a1", "t1"),
    };
    const task = newTask("contract", SessionId.generate(), tree);
    expect(() => reg.validate(task)).not.toThrow();
  });

  it("accepts a COMBINATOR child in a structured slot (carries its own contract)", () => {
    const reg = ExecutionRegistry.builder()
      .agent("a1", new StubAgent())
      .toolset("t1", new EmptyToolRegistry())
      .schema("worker-schema", {})
      .schema("eval-schema", {})
      .build();
    const innerSv: LoopStrategy = {
      kind: "self_verifying",
      inner: reactLeaf("a1", "t1", "worker-schema"),
      evaluator: "eval-schema",
    };
    const tree: LoopStrategy = {
      kind: "plan_execute",
      plan: innerSv,
      execute: reactLeaf("a1", "t1"),
    };
    const task = newTask("contract", SessionId.generate(), tree);
    expect(() => reg.validate(task)).not.toThrow();
  });
});

// ── resume: round-trip a Task through serde, re-resolve all ───────────────────

describe("ExecutionRegistry resume", () => {
  it("re-resolves every handle after a serde round-trip with no reconfiguration", () => {
    const task = newTask("resume me", SessionId.generate(), reactLeaf("a1", "t1", "s1"));

    // Trait objects never enter the wire — only string handles do.
    const wire = JSON.stringify(task);
    const restored = JSON.parse(wire) as { loop_strategy: LoopStrategy };
    const restoredStrategy = loopStrategyFromJson(restored.loop_strategy);
    const restoredTask = newTask("resume me", task.session_id, restoredStrategy);

    // A freshly-built registry (as on resume) re-resolves all.
    const reg = fullyWiredRegistry();
    expect(() => reg.validate(restoredTask)).not.toThrow();

    expect(restoredStrategy.kind).toBe("react");
    if (restoredStrategy.kind === "react") {
      expect(reg.resolveAgent(restoredStrategy.agent)).toBeDefined();
      expect(reg.resolveToolset(restoredStrategy.toolset)).toBeDefined();
      expect(reg.resolveSchema(restoredStrategy.output!)).toBeDefined();
    }
  });
});

// ── unresolved handle → startup error before the first turn ───────────────────

describe("StandardHarness startup validation (issue #120)", () => {
  function configWith(registry?: ExecutionRegistry): HarnessConfig {
    return {
      agent: new StubAgent(),
      toolRegistry: new EmptyToolRegistry(),
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      registry,
    };
  }

  it("a populated registry with an unresolved handle fails before the first turn", async () => {
    const registry = ExecutionRegistry.builder().agent("a1", new StubAgent()).build();
    const harness = new StandardHarness(configWith(registry));
    // toolset 'missing' is unresolved.
    const task = newTask("go", SessionId.of("start"), reactLeaf("a1", "missing"));

    const result = await harness.run({ task });
    expect(result.kind).toBe("failure");
    if (result.kind === "failure") {
      expect(result.reason.kind).toBe("configuration_error");
      // No turn was taken.
      expect(result.turns).toBe(0);
      if (result.reason.kind === "configuration_error") {
        expect(result.reason.error).toBeInstanceOf(UnresolvedHandle);
        const err = result.reason.error as UnresolvedHandle;
        expect(err.handleKind).toBe("toolset");
        expect(err.key).toBe("missing");
      }
    }
  });

  it("an empty registry skips validation (legacy callers unaffected)", async () => {
    // The StubAgent throws if a turn is ever attempted. An empty registry must
    // NOT short-circuit with a configuration_error — it falls through to the
    // normal run path and reaches the agent turn (which throws here). Reaching
    // the agent proves validation was skipped, and the error is the agent's, not
    // a configuration_error.
    const harness = new StandardHarness(configWith(undefined));
    const task = newTask("go", SessionId.of("legacy"), reactLeaf("anything", "anything"));
    let reached = false;
    try {
      await harness.run({ task });
    } catch (e) {
      reached = true;
      expect((e as Error).message).toBe("validate() must fail before any agent turn");
    }
    expect(reached).toBe(true);
  });
});

// ── EscalationMode present/selectable/readable on config ──────────────────────

describe("EscalationMode config knob (issue #120)", () => {
  it("the builder defaults escalationMode to surfaceToHuman", () => {
    const config = HarnessBuilder.conversational(fakeModel()).buildConfig();
    expect(config.escalationMode).toEqual(surfaceToHuman);
  });

  it("escalationMode is selectable and readable", () => {
    const config = HarnessBuilder.conversational(fakeModel())
      .escalationMode(autonomous)
      .buildConfig();
    expect(config.escalationMode).toEqual(autonomous);
  });

  it("the builder carries a populated registry into the config", () => {
    const config = HarnessBuilder.conversational(fakeModel())
      .agentRef("a1", new StubAgent())
      .buildConfig();
    expect(config.registry).toBeDefined();
    expect(config.registry!.resolveAgent("a1")).toBeDefined();
  });
});

// ── fixture replay: registry_errors.json round-trips byte-identically ─────────

describe("registry_errors.json fixture replay", () => {
  const doc = JSON.parse(
    readFileSync(resolve(repoRoot, "fixtures/harness/registry_errors.json"), "utf-8"),
  ) as Record<string, unknown>;

  it("StrategyNotFound round-trips byte-identically", () => {
    const snf = harnessErrorFromJson(doc.strategy_not_found);
    expect(snf).toBeInstanceOf(StrategyNotFound);
    expect((snf as StrategyNotFound).key).toBe("my-harness::DoubleVerify");
    expect(snf.toJSON()).toEqual(doc.strategy_not_found);
  });

  it("UnresolvedHandle round-trips byte-identically (handle_kind on the wire)", () => {
    const uh = harnessErrorFromJson(doc.unresolved_handle);
    expect(uh).toBeInstanceOf(UnresolvedHandle);
    expect((uh as UnresolvedHandle).handleKind).toBe("agent");
    expect((uh as UnresolvedHandle).key).toBe("planner");
    expect(uh.toJSON()).toEqual(doc.unresolved_handle);
  });
});

// A minimal ModelInterface double for the builder tests (no calls are made).
function fakeModel(): ModelInterface {
  return {
    async call(_req: ModelRequest): Promise<ModelResponse> {
      throw new Error("not called");
    },
    async countTokens(_req: ModelRequest): Promise<number> {
      return 0;
    },
  };
}
