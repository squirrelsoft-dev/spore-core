/**
 * SubagentTool tests — mirror `rust/crates/spore-core/src/tools/subagent.rs#tests`.
 * Uses a scripted Harness that returns a queue of `RunResult` values.
 */

import {
  emptyAggregateUsage,
  emptyBudgetSnapshot,
  emptySessionState,
  harnessTesting,
  newTask,
  SessionId,
  TaskId,
  toolRegistry,
  type Harness,
  type HarnessRunOptions,
  type HumanResponse,
  type PausedState,
  type RunResult,
  type StreamSink,
  type ToolCall,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import { SubagentTool } from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;
const { StandardToolRegistry, toolRegistryMock } = toolRegistry;
// Storage seam (#75): SubagentTool ignores ctx, but the signature requires one.
const ctx = toolRegistryMock.testCtx();

class ScriptedHarness implements Harness {
  constructor(private readonly results: RunResult[]) {}
  async run(_opts: HarnessRunOptions): Promise<RunResult> {
    const r = this.results.shift();
    if (!r) {
      return {
        kind: "failure",
        reason: { kind: "human_halted" },
        session_id: SessionId.of("s"),
        usage: emptyAggregateUsage(),
        turns: 0,
      };
    }
    return r;
  }
  async resume(
    _state: PausedState,
    _response: HumanResponse,
    _onStream?: StreamSink,
  ): Promise<RunResult> {
    return {
      kind: "failure",
      reason: { kind: "human_halted" },
      session_id: SessionId.of("s"),
      usage: emptyAggregateUsage(),
      turns: 0,
    };
  }
}

function callWith(input: unknown, id = "parent-call-1"): ToolCall {
  return { id, name: "subagent", input };
}

function buildSubagent(harness: Harness): SubagentTool {
  return SubagentTool.buildOrThrow({
    name: "subagent",
    description: "child",
    inputSchema: { type: "object" },
    timeoutMs: 5_000,
    contextSharing: { kind: "isolated" },
    harness,
    childRegistry: new StandardToolRegistry(),
  });
}

describe("SubagentTool", () => {
  it("maps child Success to ToolOutput.success", async () => {
    const harness = new ScriptedHarness([
      {
        kind: "success",
        output: "child done",
        session_id: SessionId.of("s"),
        usage: emptyAggregateUsage(),
        turns: 1,
      },
    ]);
    const sub = buildSubagent(harness);
    const r = await sub.execute(
      callWith({ instruction: "do it" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toBe("child done");
  });

  it("maps child Failure to recoverable Error", async () => {
    const harness = new ScriptedHarness([
      {
        kind: "failure",
        reason: { kind: "human_halted" },
        session_id: SessionId.of("s"),
        usage: emptyAggregateUsage(),
        turns: 1,
      },
    ]);
    const sub = buildSubagent(harness);
    const r = await sub.execute(
      callWith({ instruction: "x" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it("propagates WaitingForHuman with parent_tool_call_id", async () => {
    const sessionId = SessionId.of("s");
    const paused: PausedState = {
      session_id: sessionId,
      task_id: TaskId.of("t"),
      turn_number: 1,
      session_state: emptySessionState(),
      pending_tool_calls: [],
      approved_results: [],
      human_request: { kind: "clarification", question: "yes?" },
      task: newTask("x", sessionId, { kind: "re_act", max_iterations: 1 }),
      budget_used: emptyBudgetSnapshot(),
      child_state: null,
    };
    const harness = new ScriptedHarness([
      {
        kind: "waiting_for_human",
        state: paused,
        request: { kind: "clarification", question: "yes?" },
      },
    ]);
    const sub = buildSubagent(harness);
    const r = await sub.execute(
      callWith({ instruction: "x" }, "parent-call-1"),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("waiting_for_human");
    if (r.kind !== "waiting_for_human") throw new Error("unreachable");
    expect(r.child_state.parent_tool_call_id).toBe("parent-call-1");
  });

  it("rejects construction when child registry has SubagentTool", () => {
    const childReg = new StandardToolRegistry();
    const err = childReg.register(
      new toolRegistryMock.SubagentMockTool("nested"),
      {
        name: "nested",
        description: "n",
        parameters: { type: "object" },
        annotations: {
          read_only: false,
          destructive: false,
          idempotent: false,
          open_world: false,
        },
      },
    );
    expect(err).toBeNull();
    const r = SubagentTool.build({
      name: "subagent",
      description: "child",
      inputSchema: { type: "object" },
      timeoutMs: 1_000,
      contextSharing: { kind: "isolated" },
      harness: new ScriptedHarness([]),
      childRegistry: childReg,
    });
    expect(r.ok).toBe(false);
    if (r.ok) throw new Error("unreachable");
    expect(r.error.kind).toBe("invalid_configuration");
  });

  it("missing instruction returns recoverable error", async () => {
    const sub = buildSubagent(new ScriptedHarness([]));
    const r = await sub.execute(callWith({}), new AllowAllSandbox(), ctx);
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });
});
