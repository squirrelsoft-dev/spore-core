/**
 * Unit tests for the per-leaf `system_prompt` override in `ReactConfig` (SC-10,
 * #161).
 *
 * Mirrors the inline `plan_and_execute_leaves_see_only_their_own_system_prompt`
 * and `leaf_system_prompt_overrides_global_and_falls_back` tests in
 * `rust/crates/spore-core/src/harness.rs` — same rules, same verdicts.
 *
 * SC-10 (consumer-friction plan §4): the PlanExecute plan + execute phases ran
 * under one global `HarnessConfig.systemPrompt`, with no way to give the two
 * phases distinct system prompts. The TOOLSET half was already per-leaf
 * (`ReactConfig.toolset`); this adds the matching PROMPT half. Because both
 * PlanExecute phases bottom out in `ReactConfig` leaves, the per-leaf override
 * rides the existing recursion — each phase sees ONLY its own prompt.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyResponse,
  SessionId,
  StandardHarness,
  newTask,
  type Agent,
  type Context,
  type HarnessConfig,
  type LoopStrategy,
  type TokenUsage,
  type TurnResult,
} from "../src/index.js";
import { ProjectId } from "../src/storage/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function fr(content: string): TurnResult {
  return { kind: "final_response", content, usage: usage() };
}

/**
 * A recording agent that, in ADDITION to popping the next scripted result,
 * snapshots the FULLY ASSEMBLED context it saw on each turn (post system-prompt
 * merge). `seenText()[i]` is the serialized text of turn `i`'s context, so a
 * marker `includes` check proves exactly which system prompt the turn carried.
 * Mirrors Rust's `RecordingTurnAgent::seen_text`.
 */
class RecordingTurnAgent implements Agent {
  ran = 0;
  private readonly results: TurnResult[] = [];
  private readonly contexts: string[] = [];
  constructor(private readonly agentId: AgentId) {}
  push(r: TurnResult): this {
    this.results.push(r);
    return this;
  }
  seenText(): string[] {
    return this.contexts;
  }
  id(): AgentId {
    return this.agentId;
  }
  async turn(ctx: Context, _signal?: AbortSignal): Promise<TurnResult> {
    this.ran += 1;
    // Snapshot the assembled prompt the model saw (all messages — the leading
    // System block is where the operating system prompt lands).
    this.contexts.push(JSON.stringify(ctx.messages));
    const next = this.results.shift();
    if (next == null) return { kind: "error", error: new EmptyResponse(), usage: null };
    return next;
  }
}

const TEST_PROJECT = ProjectId.fromCanonicalPath("/test-project");
const SID = SessionId.of("s1");

function configWith(agent: Agent, overrides: Partial<HarnessConfig> = {}): HarnessConfig {
  return {
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    projectId: TEST_PROJECT,
    ...overrides,
    registry: registryWith({ agent }),
  };
}

/** A PlanExecute whose plan/execute leaves carry the given per-leaf prompts. */
function planExecuteWithLeafPrompts(
  planPrompt: string | undefined,
  executePrompt: string | undefined,
): LoopStrategy {
  return {
    kind: "plan_execute",
    plan: {
      kind: "react",
      budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
      agent: "",
      toolset: "",
      // A.5: a bare ReAct in the structured `plan` slot must declare an `output`
      // schema so the slot yields a typed plan. `registryWith` registers a
      // default schema under "". (Orthogonal to SC-10 — the override is the
      // `system_prompt` field below.)
      output: "",
      ...(planPrompt !== undefined ? { system_prompt: planPrompt } : {}),
    },
    execute: {
      kind: "react",
      budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
      agent: "",
      toolset: "",
      ...(executePrompt !== undefined ? { system_prompt: executePrompt } : {}),
    },
  };
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

describe("per-leaf system_prompt override (SC-10, #161)", () => {
  // A PlanExecute whose plan and execute leaves carry DISTINCT system prompts
  // runs each phase under ONLY its own prompt — neither leaks into the other.
  // The global `config.systemPrompt` is unset here, so the ONLY system text a
  // turn can see is what its own leaf supplied; any cross-phase appearance is an
  // unambiguous leak.
  it("plan_and_execute_leaves_see_only_their_own_system_prompt", async () => {
    const PLAN_SYS = "PLAN_SYSTEM_PROMPT_MARKER";
    const EXEC_SYS = "EXECUTE_SYSTEM_PROMPT_MARKER";

    const agent = new RecordingTurnAgent(AgentId.of("rec"))
      // Plan turn: a single-step plan.
      .push(fr('{"tasks":["only step"],"rationale":"r"}'))
      // Execute step: finalize directly.
      .push(fr("did the step"));

    // No global prompt ⇒ the ONLY system text a turn can see is its leaf's.
    const config = configWith(agent);
    expect(config.systemPrompt).toBeUndefined();
    const h = new StandardHarness(config);

    const t = newTask("goal", SID, planExecuteWithLeafPrompts(PLAN_SYS, EXEC_SYS), {
      max_turns: null,
    });
    const r = await h.run({ task: t });
    expect(r.kind === "success" && r.output).toBe("did the step");

    const contexts = agent.seenText();
    expect(contexts.length).toBe(2); // one plan turn + one execute turn

    // Plan turn (index 0): sees ONLY the plan leaf's prompt.
    expect(contexts[0]).toContain(PLAN_SYS);
    expect(contexts[0]).not.toContain(EXEC_SYS);

    // Execute turn (index 1): sees ONLY the execute leaf's prompt.
    expect(contexts[1]).toContain(EXEC_SYS);
    expect(contexts[1]).not.toContain(PLAN_SYS);
  });

  // The per-leaf override WINS over the global `config.systemPrompt` — a leaf
  // that supplies one sees ONLY its own (the global does not leak in), while a
  // leaf WITHOUT an override falls back to the global prompt (byte-identical to
  // pre-SC-10).
  it("leaf_system_prompt_overrides_global_and_falls_back", async () => {
    const GLOBAL_SYS = "GLOBAL_SYSTEM_PROMPT_MARKER";
    const PLAN_SYS = "PLAN_ONLY_SYSTEM_PROMPT_MARKER";

    const agent = new RecordingTurnAgent(AgentId.of("rec"))
      .push(fr('{"tasks":["only step"],"rationale":"r"}'))
      .push(fr("did the step"));

    // A global prompt IS configured this time.
    const h = new StandardHarness(configWith(agent, { systemPrompt: GLOBAL_SYS }));

    // Plan leaf overrides the global; execute leaf carries no override (falls
    // back to the global prompt).
    const t = newTask("goal", SID, planExecuteWithLeafPrompts(PLAN_SYS, undefined), {
      max_turns: null,
    });
    const r = await h.run({ task: t });
    expect(r.kind).toBe("success");

    const contexts = agent.seenText();
    expect(contexts.length).toBe(2); // one plan turn + one execute turn

    // Plan turn: its override WINS — only the plan prompt, NOT the global one.
    expect(contexts[0]).toContain(PLAN_SYS);
    expect(contexts[0]).not.toContain(GLOBAL_SYS);

    // Execute turn: no override ⇒ the global prompt applies (back-compat).
    expect(contexts[1]).toContain(GLOBAL_SYS);
    expect(contexts[1]).not.toContain(PLAN_SYS);
  });
});
