/**
 * #140 — `PausedState` (and `ChildPausedState`) carry the pausing leaf's toolset
 * handle, so resume routes pending per-node tool calls back through that leaf's
 * scoped catalogue instead of the global-catalogue fallback.
 *
 * Mirrors the Rust `#140` tests in `rust/crates/spore-core/src/harness.rs`
 * (`paused_state_toolset_back_compat_and_round_trip`,
 * `consult_pause_carries_leaf_toolset_handle`,
 * `clarification_pause_carries_leaf_toolset_handle`,
 * `resume_consult_routes_pending_calls_through_carried_toolset`) — same rules,
 * same verdicts. Covers:
 *   - AC1 (back-compat + always-serialize): a paused-state blob WITHOUT a
 *     `toolset` key hydrates to `""`; the empty handle ALWAYS serializes; a
 *     non-empty handle round-trips by value. Holds for `ChildPausedState` too.
 *   - AC2a (populate): a Consult / Clarification pause from a leaf carrying
 *     `toolset = "scoped"` returns a state whose `toolset` is that handle.
 *   - AC2b (load-bearing): a resumed consult dispatches a pending call to a tool
 *     registered ONLY under the `"scoped"` catalogue when the carried handle is
 *     `"scoped"`; with the EMPTY handle (negative control) the same call is
 *     unknown → a recoverable error.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyToolRegistry,
  ExecutionRegistry,
  HumanRequestSchema,
  MockAgent,
  SessionId,
  StandardHarness,
  TaskId,
  loadCheckpoint,
  newTask,
  serializeCheckpoint,
  toolRegistry as toolRegistryNs,
  type ChildPausedState,
  type HarnessConfig,
  type LoopStrategy,
  type Message,
  type PausedState,
  type Task,
  type TokenUsage,
  type ToolCall,
  type ToolOutput,
  type TurnResult,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

const { StandardToolRegistry, toolRegistryMock } = toolRegistryNs;
const { EchoTool } = toolRegistryMock;

// ── helpers ─────────────────────────────────────────────────────────────────

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function emptySession(): PausedState["session_state"] {
  return { messages: [], extras: {} };
}

function emptyBudget(): PausedState["budget_used"] {
  return { turns: 0, input_tokens: 0, output_tokens: 0, wall_time: null, cost_usd: 0 };
}

/** A bare ReAct task whose leaf carries the given `toolset` handle. */
function reactScoped(max: number, handle: string): Task {
  const strategy: LoopStrategy = {
    kind: "react",
    budget: { kind: "per_loop", value: max },
    behavior: { kind: "escalate" },
    agent: "",
    toolset: handle,
  };
  return newTask("do something", SessionId.of("s1"), strategy);
}

/** A worker that requests one tool call (so the leaf reaches dispatch) then ends. */
function agentRequestingThenDone(): MockAgent {
  const a = new MockAgent(AgentId.of("test"));
  a.push({
    kind: "tool_call_requested",
    calls: [{ id: "c0", name: "probe", input: {} } as ToolCall],
    usage: usage(),
  } as TurnResult);
  a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);
  return a;
}

/**
 * A registry wiring the default-"" worker agent plus a presence entry under the
 * `"scoped"` toolset handle so `validate()` at run entry passes for a leaf that
 * declares `toolset: "scoped"`.
 */
function registryWithScopedHandle(agent: MockAgent): ExecutionRegistry {
  return ExecutionRegistry.builder()
    .agent("", agent)
    .toolset("", new EmptyToolRegistry())
    .toolset("scoped", new EmptyToolRegistry())
    .schema("", {})
    .build();
}

// ============================================================================
// AC1 — back-compat (missing key → "") + always-serialize + round-trip
// ============================================================================

describe("#140 AC1 — toolset back-compat and round-trip", () => {
  it("a paused-state blob WITHOUT a toolset key hydrates to the empty handle", () => {
    // A pre-#140 PausedState JSON (no `toolset` key) — must default to "".
    const pre140 = {
      session_id: "s",
      task_id: "t",
      turn_number: 1,
      session_state: { messages: [], extras: {} },
      pending_tool_calls: [],
      approved_results: [],
      human_request: null,
      task: JSON.parse(JSON.stringify(reactScoped(5, ""))),
      budget_used: emptyBudget(),
      child_state: null,
      // note: NO "toolset" key
    };
    const parsed = loadCheckpoint(JSON.stringify(pre140));
    expect(parsed.toolset).toBe("");
  });

  it("the empty handle ALWAYS serializes explicitly (never skipped)", () => {
    const state = loadCheckpoint(
      JSON.stringify({
        session_id: "s",
        task_id: "t",
        turn_number: 1,
        session_state: { messages: [], extras: {} },
        pending_tool_calls: [],
        approved_results: [],
        human_request: null,
        task: JSON.parse(JSON.stringify(reactScoped(5, ""))),
        budget_used: emptyBudget(),
        child_state: null,
      }),
    );
    const wire = serializeCheckpoint(state);
    expect(wire).toContain('"toolset":""');
  });

  it("a non-empty handle round-trips by value through serialize/loadCheckpoint", () => {
    const base = loadCheckpoint(
      JSON.stringify({
        session_id: "s",
        task_id: "t",
        turn_number: 1,
        session_state: { messages: [], extras: {} },
        pending_tool_calls: [],
        approved_results: [],
        human_request: null,
        task: JSON.parse(JSON.stringify(reactScoped(5, "scoped"))),
        budget_used: emptyBudget(),
        child_state: null,
      }),
    );
    const scoped: PausedState = { ...base, toolset: "scoped" };
    const back = loadCheckpoint(serializeCheckpoint(scoped));
    expect(back.toolset).toBe("scoped");
    expect(back).toEqual(scoped);
  });

  it("the same contract holds for ChildPausedState (default + always-serialize)", () => {
    // A pre-#140 child blob (no `toolset` key) hydrates to "" via the parent.
    const childPre140 = {
      session_id: "c",
      task_id: "ct",
      turn_number: 1,
      session_state: { messages: [], extras: {} },
      pending_tool_calls: [],
      approved_results: [],
      human_request: null,
      task: JSON.parse(JSON.stringify(reactScoped(1, ""))),
      budget_used: emptyBudget(),
      parent_tool_call_id: "p",
      // note: NO "toolset" key
    };
    const withChild = loadCheckpoint(
      JSON.stringify({
        session_id: "s",
        task_id: "t",
        turn_number: 1,
        session_state: { messages: [], extras: {} },
        pending_tool_calls: [],
        approved_results: [],
        human_request: null,
        task: JSON.parse(JSON.stringify(reactScoped(5, ""))),
        budget_used: emptyBudget(),
        child_state: childPre140,
      }),
    );
    const child = withChild.child_state as ChildPausedState;
    expect(child.toolset).toBe("");
    // The empty child handle ALWAYS serializes.
    expect(serializeCheckpoint(withChild)).toContain('"parent_tool_call_id":"p","toolset":""');
  });
});

// ============================================================================
// AC2a — leaf-pause sites populate `toolset` with the leaf's handle
// ============================================================================

describe("#140 AC2a — leaf pause carries its toolset handle", () => {
  function configWithScoped(agent: MockAgent): HarnessConfig {
    return {
      registry: registryWithScopedHandle(agent),
      toolRegistry: new ScriptedToolRegistry(),
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
    };
  }

  it("a Consult pause carries the leaf's toolset handle", async () => {
    const a = agentRequestingThenDone();
    const cfg = configWithScoped(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({
      kind: "consult",
      request: { kind: "advice", situation: "stuck", attempts: 1, question: "help?" },
    } as ToolOutput);
    const h = new StandardHarness(cfg);

    const r = await h.run({ task: reactScoped(5, "scoped") });
    expect(r.kind).toBe("consult");
    if (r.kind !== "consult") return;
    expect(r.state.toolset).toBe("scoped");
  });

  it("a Clarification pause carries the leaf's toolset handle", async () => {
    const a = agentRequestingThenDone();
    const cfg = configWithScoped(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({
      kind: "awaiting_clarification",
      question: "which?",
    } as ToolOutput);
    const h = new StandardHarness(cfg);

    const r = await h.run({ task: reactScoped(5, "scoped") });
    expect(r.kind).toBe("waiting_for_human");
    if (r.kind !== "waiting_for_human") return;
    expect(r.state.toolset).toBe("scoped");
  });
});

// ============================================================================
// AC2b — the load-bearing regression guard: resume routes through the handle
// ============================================================================

describe("#140 AC2b — resumeConsult routes pending calls through the carried toolset", () => {
  /** Bundle an `EchoTool` into a `StandardTool` under `name`. */
  function echoStandardTool(name: string) {
    return {
      implementation: new EchoTool(name),
      schema: {
        name,
        description: "echo",
        parameters: { type: "object" },
        annotations: {
          read_only: true,
          destructive: false,
          idempotent: false,
          open_world: false,
        },
      },
    };
  }

  /**
   * A paused state with TWO pending calls: the head consult call (gets the answer
   * injected) and a trailing call to the scoped-only tool (dispatched through
   * `effectiveToolRegistry(session_id, state.toolset)`). The bare ReAct leaf
   * carries `handle`, so resume takes the dispatch-then-re-enter-window path.
   */
  function makeState(handle: string): PausedState {
    const strategy: LoopStrategy = {
      kind: "react",
      budget: { kind: "per_loop", value: 5 },
      behavior: { kind: "escalate" },
      agent: "",
      toolset: handle,
    };
    return {
      session_id: SessionId.of("s"),
      task_id: TaskId.of("t"),
      turn_number: 1,
      session_state: emptySession(),
      pending_tool_calls: [
        { id: "consult", name: "ask_advice", input: { kind: "advice" } },
        { id: "scoped", name: "scoped_only", input: { probe: 1 } },
      ],
      approved_results: [],
      task: { ...newTask("audit", SessionId.of("s"), strategy), id: TaskId.of("t") },
      budget_used: emptyBudget(),
      child_state: null,
      toolset: handle,
    };
  }

  /**
   * Was the scoped-only tool's pending call dispatched successfully? Scan the
   * resumed session for a tool-result message. `NoopContextManager` prefixes a
   * recoverable error with "[error]"; `EchoTool` echoes the call input (so a
   * success contains "probe").
   */
  function scopedDispatchedOk(messages: Message[]): boolean {
    return messages.some(
      (m) =>
        m.role === "tool" &&
        m.content.type === "text" &&
        m.content.text.includes("probe") &&
        !m.content.text.includes("[error]"),
    );
  }

  /**
   * A harness with a SCOPED catalogue ("scoped" → scoped_only) AND a GLOBAL
   * catalogue ("global_only", which does NOT contain scoped_only), plus a worker
   * agent that emits a final response so the re-entered ReAct window terminates
   * cleanly. The global catalogue is what the EMPTY handle falls back to — making
   * `scoped_only` a genuine unknown-tool there.
   */
  function harnessWithScopedCatalogue(): StandardHarness {
    const a = new MockAgent(AgentId.of("worker"));
    a.push({ kind: "final_response", content: "resumed-done", usage: usage() } as TurnResult);

    const globalCatalogue = new StandardToolRegistry();
    const gErr = globalCatalogue.tools([echoStandardTool("global_only")]);
    expect(gErr).toBeNull();

    const scopedCatalogue = new StandardToolRegistry();
    const sErr = scopedCatalogue.tools([echoStandardTool("scoped_only")]);
    expect(sErr).toBeNull();

    const cfg: HarnessConfig = {
      registry: registryWith({ agent: a }),
      toolRegistry: new ScriptedToolRegistry(),
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      catalogueRegistry: globalCatalogue,
      toolsetCatalogues: new Map([["scoped", scopedCatalogue]]),
    };
    return new StandardHarness(cfg);
  }

  it("the carried handle 'scoped' routes the scoped-only call to the scoped catalogue", async () => {
    const h = harnessWithScopedCatalogue();
    const result = await h.resumeConsult(makeState("scoped"), { kind: "answer", text: "ok" });
    expect(result.kind).toBe("success");
    if (result.kind !== "success") return;
    expect(scopedDispatchedOk(result.session_state?.messages ?? [])).toBe(true);
  });

  it("the EMPTY handle (negative control) falls back to the global catalogue → unknown tool", async () => {
    const h = harnessWithScopedCatalogue();
    const result = await h.resumeConsult(makeState(""), { kind: "answer", text: "ok" });
    expect(result.kind).toBe("success");
    if (result.kind !== "success") return;
    const messages = result.session_state?.messages ?? [];
    // The scoped-only tool is unknown under the global catalogue → no success.
    expect(scopedDispatchedOk(messages)).toBe(false);
    // And it surfaced the recoverable error result.
    const hasError = messages.some(
      (m) => m.role === "tool" && m.content.type === "text" && m.content.text.includes("[error]"),
    );
    expect(hasError).toBe(true);
  });
});

// A small smoke check that the existing `HumanRequestSchema` import stays used —
// the AC1 fixtures embed a null human_request, exercised above via loadCheckpoint.
describe("#140 — paused-state schema sanity", () => {
  it("a clarification human_request still parses (unrelated to toolset)", () => {
    const req = HumanRequestSchema.parse({ kind: "clarification", question: "q" });
    expect(req.kind).toBe("clarification");
  });
});
