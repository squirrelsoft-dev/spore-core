/**
 * SubagentTool consult-mediation tests (spore-core issue #114) — mirror
 * `rust/crates/spore-core/src/tools/subagent.rs#tests` (the consult block).
 *
 * The SubagentTool drives the FULL consult cycle internally (seam A1): on a
 * child `RunResult.consult` it routes by `kind`, enforces the per-kind budget,
 * runs the handler as the orchestrator's direct child, and resumes the worker —
 * the parent orchestrator's model never sees the consult.
 *
 * Rules covered: R2 (mediate, not bubble), R3 (route by kind, no parent model,
 * parent sees success), R4 (per-kind budget), R5a (soft_fail → budget_exhausted),
 * R5b (escalate_to_human → waiting_for_human), R6 (no matching kind → escalate;
 * no handlers → escalate), R7 (depth-1: the handler runs via handler.run).
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
  type ConsultHandlerEntry,
  type ConsultOverflowPolicy,
  type ConsultRequest,
  type ConsultResponse,
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
const ctx = toolRegistryMock.testCtx();

/** Scripted harness whose `resumeConsult` logs the response and pops the next
 *  scripted RunResult — so a test can assert exactly what the worker was
 *  resumed with. An empty queue resolves to a terminal success. */
class ScriptedHarness implements Harness {
  readonly resumeLog: ConsultResponse[] = [];
  constructor(private readonly results: RunResult[]) {}

  async run(_opts: HarnessRunOptions): Promise<RunResult> {
    return this.pop();
  }
  async resume(
    _state: PausedState,
    _response: HumanResponse,
    _onStream?: StreamSink,
  ): Promise<RunResult> {
    return this.terminal();
  }
  async resumeConsult(
    _state: PausedState,
    response: ConsultResponse,
    _onStream?: StreamSink,
  ): Promise<RunResult> {
    this.resumeLog.push(response);
    return this.pop();
  }
  private pop(): RunResult {
    const r = this.results.shift();
    return r ?? this.terminal();
  }
  private terminal(): RunResult {
    return {
      kind: "success",
      output: "child done after consult",
      session_id: SessionId.of("worker"),
      usage: emptyAggregateUsage(),
      turns: 1,
    };
  }
}

/** Handler harness that records each instruction it is run with and returns a
 *  fixed answer. Used to assert depth-1 routing (R3/R7). */
class RecordingHandler implements Harness {
  readonly seen: string[] = [];
  constructor(private readonly answer: string) {}
  async run(opts: HarnessRunOptions): Promise<RunResult> {
    this.seen.push(opts.task.instruction);
    return {
      kind: "success",
      output: this.answer,
      session_id: SessionId.of("handler"),
      usage: emptyAggregateUsage(),
      turns: 1,
    };
  }
  async resume(
    _state: PausedState,
    _response: HumanResponse,
    _onStream?: StreamSink,
  ): Promise<RunResult> {
    return {
      kind: "failure",
      reason: { kind: "human_halted" },
      session_id: SessionId.of("handler"),
      usage: emptyAggregateUsage(),
      turns: 0,
    };
  }
}

function consultPaused(): PausedState {
  const sessionId = SessionId.of("worker");
  return {
    session_id: sessionId,
    task_id: TaskId.of("t"),
    turn_number: 1,
    session_state: emptySessionState(),
    pending_tool_calls: [
      { id: "consult-call", name: "ask_advice", input: { kind: "advice" } },
    ],
    approved_results: [],
    task: newTask("audit", sessionId, { kind: "re_act", max_iterations: 4 }),
    budget_used: emptyBudgetSnapshot(),
    child_state: null,
  };
}

function consultRequest(kind: string): ConsultRequest {
  return { kind, situation: "drowning", attempts: 2, question: "what now?" };
}

function consultResult(kind: string): RunResult {
  return {
    kind: "consult",
    request: consultRequest(kind),
    state: consultPaused(),
    session_id: SessionId.of("worker"),
    usage: emptyAggregateUsage(),
    turns: 1,
  };
}

function handlers(
  kind: string,
  handler: RecordingHandler,
  budget: number,
  overflow: ConsultOverflowPolicy,
): Map<string, ConsultHandlerEntry> {
  return new Map([[kind, { handler, budget, overflow }]]);
}

function callWith(input: unknown, id = "parent-call-1"): ToolCall {
  return { id, name: "subagent", input };
}

function buildSubagent(
  harness: Harness,
  consultHandlers?: Map<string, ConsultHandlerEntry>,
): SubagentTool {
  return SubagentTool.buildOrThrow({
    name: "subagent",
    description: "child",
    inputSchema: { type: "object" },
    timeoutMs: 5_000,
    contextSharing: { kind: "isolated" },
    harness,
    childRegistry: new StandardToolRegistry(),
    consultHandlers,
  });
}

describe("SubagentTool consult mediation (#114)", () => {
  // R2/R3: child consult is MEDIATED here (not bubbled). With a registered
  // handler, the handler runs (no parent model), the worker is resumed, and the
  // parent ultimately sees success.
  it("R2/R3: mediates and resumes to success without bubbling", async () => {
    const handler = new RecordingHandler("try plan B");
    const harness = new ScriptedHarness([consultResult("advice")]);
    const sub = buildSubagent(
      harness,
      handlers("advice", handler, 3, { kind: "soft_fail" }),
    );
    const r = await sub.execute(
      callWith({ instruction: "x" }),
      new AllowAllSandbox(),
      ctx,
    );

    // R3: parent sees success (the consult never reached its model).
    expect(r.kind).toBe("success");
    if (r.kind === "success")
      expect(r.content).toBe("child done after consult");
    // R3/R7: the handler ran exactly once, on the rendered consult request.
    expect(handler.seen.length).toBe(1);
    expect(handler.seen[0]).toContain("advice");
    expect(handler.seen[0]).toContain("what now?");
    // R3: worker resumed with the handler's answer.
    expect(harness.resumeLog.length).toBe(1);
    expect(harness.resumeLog[0]).toEqual({
      kind: "answer",
      text: "try plan B",
    });
  });

  // R4 + R5a: handler runs up to `budget` times; the (budget+1)th consult
  // overflows. With soft_fail, the worker is resumed with budget_exhausted.
  it("R4/R5a: budget overflow soft_fail resumes with budget_exhausted", async () => {
    const handler = new RecordingHandler("advice answer");
    // budget = 1: run → consult; resumeConsult → consult again (over budget);
    // resumeConsult → success (queue empty).
    const harness = new ScriptedHarness([
      consultResult("advice"),
      consultResult("advice"),
    ]);
    const sub = buildSubagent(
      harness,
      handlers("advice", handler, 1, { kind: "soft_fail" }),
    );
    const r = await sub.execute(
      callWith({ instruction: "x" }),
      new AllowAllSandbox(),
      ctx,
    );

    expect(r.kind).toBe("success");
    // R4: handler ran exactly once (budget = 1).
    expect(handler.seen.length).toBe(1);
    // R5a: first resume = answer, second resume = budget_exhausted.
    expect(harness.resumeLog.length).toBe(2);
    expect(harness.resumeLog[0]!.kind).toBe("answer");
    expect(harness.resumeLog[1]!.kind).toBe("budget_exhausted");
  });

  // R5b: budget overflow with escalate_to_human → ToolOutput.waiting_for_human.
  it("R5b: budget overflow escalate_to_human surfaces waiting_for_human", async () => {
    const handler = new RecordingHandler("x");
    // budget = 0: the FIRST consult is already over budget → escalate.
    const harness = new ScriptedHarness([consultResult("advice")]);
    const sub = buildSubagent(
      harness,
      handlers("advice", handler, 0, { kind: "escalate_to_human" }),
    );
    const r = await sub.execute(
      callWith({ instruction: "x" }),
      new AllowAllSandbox(),
      ctx,
    );

    expect(r.kind).toBe("waiting_for_human");
    if (r.kind === "waiting_for_human") {
      expect(r.child_state.parent_tool_call_id).toBe("parent-call-1");
      expect(r.request.kind).toBe("review");
    }
    // Handler never ran (over budget from the start).
    expect(handler.seen.length).toBe(0);
  });

  // R6: a consult with NO matching handler (map present, wrong kind) → escalate.
  it("R6: consult with no matching kind escalates", async () => {
    const handler = new RecordingHandler("x");
    const harness = new ScriptedHarness([consultResult("research")]);
    const sub = buildSubagent(
      harness,
      handlers("advice", handler, 3, { kind: "soft_fail" }),
    );
    const r = await sub.execute(
      callWith({ instruction: "x" }),
      new AllowAllSandbox(),
      ctx,
    );

    expect(r.kind).toBe("escalate");
    if (r.kind === "escalate") {
      expect(r.signal.kind).toBe("abort");
      if (r.signal.kind === "abort")
        expect(r.signal.reason).toContain("research");
    }
  });

  // R6 (degradation): with NO handlers installed at all, a child consult is
  // treated as the no-matching-kind case → escalate.
  it("R6: consult with no handlers installed escalates", async () => {
    const harness = new ScriptedHarness([consultResult("advice")]);
    const sub = buildSubagent(harness);
    const r = await sub.execute(
      callWith({ instruction: "x" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("escalate");
  });
});
