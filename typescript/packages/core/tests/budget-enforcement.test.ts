/**
 * Per-node budget enforcement + failure isolation (spore-core issue #125).
 *
 * Mirrors the Rust `budget_enforcement_tests` module in
 * `rust/crates/spore-core/src/harness.rs`. Makes {@link BudgetContext.charge} the
 * REAL per-node enforcement point and a {@link StrategyOutcome} `budget_exhausted`
 * a real, isolated, parent-inspectable value.
 *
 * Covers every rule from the slice plan:
 *   1. A node capped at N stops at N WITHOUT killing siblings.
 *   2. In-process Continue resets the counter, honors max_continues, then falls
 *      through; session/messages unchanged across resets.
 *   3. Fail yields partialOutput = undefined.
 *   4. A child budget_exhausted reaches the parent as a StrategyOutcome, never
 *      auto-propagated (parent's own scope unaffected; parent can then Complete).
 *   5. partialOutput concrete per node (4 shapes).
 *   6. The ReAct leaf does not carry its own behavior; it propagates to parent.
 *   7. Never auto-cascade a child exhaustion into a parent exhaustion.
 *
 * No new fixtures (fork #3): every type here is runtime-only.
 */

import { describe, expect, it } from "vitest";

import {
  BudgetContext,
  BudgetStack,
  ExecutionRegistry,
  SessionId,
  StandardHarness,
  budgetPolicyAllowanceValue,
  chargeCurrent,
  hillClimbingPartialJson,
  lastFinalResponseText,
  newExecutionContext,
  newTask,
  planExecutePartialJson,
  popBudget,
  promoteBudgetExhausted,
  pushBudget,
  reactPartialJson,
  resolveCurrent,
  selfVerifyingPartialJson,
  type BudgetExhausted,
  type BudgetExhaustedBehavior,
  type RunResult,
  type Task,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
} from "../src/index.js";
import { MockAgent } from "../src/agent/mock.js";
import { AgentId } from "../src/agent/types.js";
import { addTask, defaultTaskList, updateTask } from "../src/tasklist/types.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

function continueThenFail(maxContinues: number): BudgetExhaustedBehavior {
  return { kind: "continue", max_continues: maxContinues, on_exhausted: { kind: "fail" } };
}

// ── budgetPolicyAllowanceValue ──────────────────────────────────────────────

describe("budgetPolicyAllowanceValue", () => {
  it("returns the value for capped policies, undefined for unlimited", () => {
    expect(budgetPolicyAllowanceValue({ kind: "unlimited" })).toBeUndefined();
    expect(budgetPolicyAllowanceValue({ kind: "total_steps", value: 7 })).toBe(7);
    expect(budgetPolicyAllowanceValue({ kind: "per_loop", value: 3 })).toBe(3);
    expect(budgetPolicyAllowanceValue({ kind: "per_attempt", value: 9 })).toBe(9);
  });
});

// ── resolveExhausted: Fail → fail; Escalate → escalate ──────────────────────

describe("BudgetContext.resolveExhausted", () => {
  it("Fail → fail; Escalate → escalate (terminal behaviors)", () => {
    const fail = new BudgetContext({ kind: "per_loop", value: 1 }, { kind: "fail" }, "p");
    expect(fail.resolveExhausted()).toBe("fail");

    const esc = new BudgetContext({ kind: "per_loop", value: 1 }, { kind: "escalate" }, "p");
    expect(esc.resolveExhausted()).toBe("escalate");
  });

  // ── Rule 2: Continue resets counter, honors max_continues, falls through ──
  it("Continue resets the counter, then falls through to Fail (rule 2)", () => {
    // ReAct under a parent Continue { max_continues: 2, on_exhausted: Fail }.
    const cx = new BudgetContext({ kind: "total_steps", value: 3 }, continueThenFail(2), "phase");

    // 1st exhaustion → continue (counter resets to 0, continuesUsed = 1).
    expect(cx.charge(3).ok).toBe(true);
    expect(cx.charge(1).ok).toBe(false);
    expect(cx.resolveExhausted()).toBe("continue");
    expect(cx.stepsTaken).toBe(0); // counter reset on continue #1
    expect(cx.continuesUsed).toBe(1);

    // 2nd exhaustion → continue (counter resets again, continuesUsed = 2).
    expect(cx.charge(3).ok).toBe(true);
    expect(cx.charge(1).ok).toBe(false);
    expect(cx.resolveExhausted()).toBe("continue");
    expect(cx.stepsTaken).toBe(0); // counter reset on continue #2
    expect(cx.continuesUsed).toBe(2);

    // 3rd exhaustion → continues spent → fall through to Fail.
    expect(cx.charge(3).ok).toBe(true);
    expect(cx.charge(1).ok).toBe(false);
    expect(cx.resolveExhausted()).toBe("fail");
    // continuesUsed does NOT advance past max_continues on the fall-through.
    expect(cx.continuesUsed).toBe(2);
  });

  it("Continue chain shares ONE counter, then escalates", () => {
    // Outer Continue{2}: grants 2 continues; once spent, the SHARED counter (2)
    // >= the nested Continue{2}'s max, so the nested layer grants nothing and
    // falls straight through to Escalate.
    const behavior: BudgetExhaustedBehavior = {
      kind: "continue",
      max_continues: 2,
      on_exhausted: {
        kind: "continue",
        max_continues: 2,
        on_exhausted: { kind: "escalate" },
      },
    };
    const cx = new BudgetContext({ kind: "per_loop", value: 1 }, behavior, "chain");

    expect(cx.resolveExhausted()).toBe("continue");
    expect(cx.continuesUsed).toBe(1);
    expect(cx.resolveExhausted()).toBe("continue");
    expect(cx.continuesUsed).toBe(2);
    // Outer spent → fall through to nested Continue{2}; the SHARED counter is
    // already 2 so the nested layer grants nothing → escalate.
    expect(cx.resolveExhausted()).toBe("escalate");
  });
});

// ── consumeContinue resets only the in-memory step counter ───────────────────

describe("BudgetContext.consumeContinue", () => {
  it("resets stepsTaken and bumps continuesUsed (session untouched)", () => {
    const cx = new BudgetContext({ kind: "per_loop", value: 4 }, continueThenFail(5), "c");
    expect(cx.charge(4).ok).toBe(true);
    expect(cx.stepsTaken).toBe(4);
    cx.consumeContinue();
    expect(cx.stepsTaken).toBe(0); // step counter rewound
    expect(cx.continuesUsed).toBe(1); // continue counted
    // The scope's allowance is intact — a fresh round can charge again.
    expect(cx.charge(4).ok).toBe(true);
  });
});

// ── Rule 3: Fail → partialOutput = undefined; Escalate → Some(partial) ──────

describe("promoteBudgetExhausted (rule 3)", () => {
  const err: BudgetExhausted = {
    policy: { kind: "per_loop", value: 2 },
    behavior: { kind: "fail" },
    stepsTaken: 2,
    continuesUsed: 0,
    phase: "react",
  };

  it("Fail boundary drops the partial (partialOutput = undefined)", () => {
    const failed = promoteBudgetExhausted(err, undefined);
    expect(failed.kind).toBe("budget_exhausted");
    if (failed.kind === "budget_exhausted") {
      expect(failed.partialOutput).toBeUndefined();
      expect(failed.stepsTaken).toBe(2);
    }
  });

  it("Escalate boundary keeps the partial", () => {
    const escalated = promoteBudgetExhausted(err, reactPartialJson("the answer so far"));
    expect(escalated.kind).toBe("budget_exhausted");
    if (escalated.kind === "budget_exhausted") {
      expect(escalated.partialOutput).toBeDefined();
      expect(escalated.partialOutput).toContain("the answer so far");
    }
  });
});

// ── Rule 5: each node's partialOutput has its documented shape ───────────────

describe("partial_output shapes (rule 5)", () => {
  it("react partial: last final response", () => {
    const json = JSON.parse(reactPartialJson("hello world"));
    expect(json.node).toBe("react");
    expect(json.last_final_response).toBe("hello world");
  });

  it("plan_execute partial: task list + statuses + ledger", () => {
    const tl = defaultTaskList();
    const a = addTask(tl, "task a");
    addTask(tl, "task b");
    if (a.ok) updateTask(tl, a.id, "completed");
    const json = JSON.parse(planExecutePartialJson(tl));
    expect(json.node).toBe("plan_execute");
    expect(json.tasks).toBe(2);
    expect(json.ledger).toHaveLength(2); // one row per task
    expect(json.ledger[0].description).toBe("task a");
    expect(json.ledger[0].status).toBe("completed");
    expect(json.ledger[1].status).toBe("pending");
  });

  it("self_verifying partial: last worker result + last verdict", () => {
    const json = JSON.parse(selfVerifyingPartialJson("worker output", "verdict: not yet"));
    expect(json.node).toBe("self_verifying");
    expect(json.last_worker_result).toBe("worker output");
    expect(json.last_verdict).toBe("verdict: not yet");
  });

  it("hill_climbing partial: best candidate + score", () => {
    const json = JSON.parse(hillClimbingPartialJson(0.875));
    expect(json.node).toBe("hill_climbing");
    expect(json.best_candidate).toBe(0.875);
    expect(json.score).toBe(0.875);
  });
});

// ── lastFinalResponseText ────────────────────────────────────────────────────

describe("lastFinalResponseText", () => {
  it("returns the success output", () => {
    const r: RunResult = {
      kind: "success",
      output: "the result",
      session_id: SessionId.of("s"),
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        cost_usd: 0,
      },
      turns: 1,
    };
    expect(lastFinalResponseText(r)).toBe("the result");
  });

  it("returns the last assistant text message of a failure's session state", () => {
    const r: RunResult = {
      kind: "failure",
      reason: { kind: "budget_exceeded", limit_type: "turns" },
      session_id: SessionId.of("s"),
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        cost_usd: 0,
      },
      turns: 2,
      session_state: {
        messages: [
          { role: "user", content: { type: "text", text: "go" } },
          { role: "assistant", content: { type: "text", text: "partial answer" } },
        ],
        extras: {},
      },
    };
    expect(lastFinalResponseText(r)).toBe("partial answer");
  });
});

// ── Rule 1: a node capped at N stops at N without killing siblings ───────────

describe("sibling isolation (rule 1)", () => {
  it("each sibling node gets a FRESH BudgetContext", () => {
    const budgets = new BudgetStack();
    // Sibling A: capped at 2, exhausts.
    budgets.push(
      new BudgetContext({ kind: "per_loop", value: 2 }, { kind: "escalate" }, "child-a"),
    );
    expect(budgets.current()!.charge(2).ok).toBe(true);
    expect(budgets.current()!.charge(1).ok).toBe(false); // A exhausts at its own cap N=2
    const a = budgets.pop()!;
    expect(a.stepsTaken).toBe(2);

    // Sibling B gets a FRESH BudgetContext — A's exhaustion did not bleed in.
    budgets.push(
      new BudgetContext({ kind: "per_loop", value: 4 }, { kind: "escalate" }, "child-b"),
    );
    const b = budgets.current()!;
    expect(b.stepsTaken).toBe(0); // sibling B starts fresh (rule 1)
    expect(b.charge(3).ok).toBe(true); // B runs with its own untouched allowance
  });
});

// ── Rule 4 & 7: a child exhaustion does NOT auto-cascade to the parent ───────

describe("child exhaustion isolation (rule 4 & 7)", () => {
  it("a child's exhaustion does NOT charge the parent scope; parent can Complete", () => {
    const budgets = new BudgetStack();
    // Parent scope (total_steps{5}) that is ALREADY nearly exhausted: it has
    // spent 4 of its 5 steps, leaving EXACTLY ONE remaining. This is the
    // adversarial case for rule 4/7 — if a child's exhaustion auto-cascaded even
    // a single step onto the parent, the parent would be pushed over its own cap
    // and could no longer Complete.
    budgets.push(new BudgetContext({ kind: "total_steps", value: 5 }, { kind: "fail" }, "parent"));
    expect(budgets.current()!.charge(4).ok).toBe(true);
    expect(budgets.current()!.remaining()).toBe(1); // parent starts with exactly 1 step left

    // Child descends with its OWN scope (capped at 1) and exhausts.
    budgets.push(new BudgetContext({ kind: "per_loop", value: 1 }, { kind: "escalate" }, "child"));
    expect(budgets.current()!.charge(1).ok).toBe(true);
    expect(budgets.current()!.charge(1).ok).toBe(false);
    // The child surfaces a budget_exhausted value — modelled here by popping its
    // scope. Crucially the parent scope is UNCHARGED by this.
    budgets.pop();

    const parent = budgets.current()!;
    // rule 4/7: child exhaustion did NOT auto-charge the parent — its stepsTaken
    // is unchanged at 4 (not bumped to 5).
    expect(parent.stepsTaken).toBe(4);
    // The parent STILL has its 1 remaining step after the child exhausted.
    expect(parent.remaining()).toBe(1);
    // The parent can spend its last step and Complete — proving the child's
    // exhaustion did NOT push it over its own cap.
    expect(parent.charge(1).ok).toBe(true);
    // And only now is the parent itself exhausted (at its own 5, by its own work
    // — never by the child's).
    expect(parent.charge(1).ok).toBe(false);
  });
});

// ── ExecutionContext helpers: push / charge / resolve / pop round-trip ───────

describe("ExecutionContext budget helpers", () => {
  it("push / charge / resolve / pop round-trip", () => {
    const cx = newExecutionContext(ExecutionRegistry.empty());
    expect(cx.budgets.depth()).toBe(0);

    pushBudget(cx, { kind: "per_loop", value: 2 }, continueThenFail(1), "node");
    expect(cx.budgets.depth()).toBe(1);
    expect(chargeCurrent(cx, 2).ok).toBe(true);
    expect(chargeCurrent(cx, 1).ok).toBe(false); // scope exhausts at its cap
    expect(resolveCurrent(cx)).toBe("continue");
    // After the continue, the counter reset → charging is possible again.
    expect(chargeCurrent(cx, 2).ok).toBe(true);
    expect(chargeCurrent(cx, 1).ok).toBe(false);
    expect(resolveCurrent(cx)).toBe("fail");
    const popped = popBudget(cx);
    expect(popped).toBeDefined();
    expect(cx.budgets.depth()).toBe(0);
  });

  it("chargeCurrent with no scope is a no-op ok; resolveCurrent → fail", () => {
    const cx = newExecutionContext(ExecutionRegistry.empty());
    expect(chargeCurrent(cx, Number.MAX_SAFE_INTEGER).ok).toBe(true);
    expect(resolveCurrent(cx)).toBe("fail");
  });
});

// ── Rule 6: the ReAct leaf propagates, no leaf behavior (integration) ────────

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function tcr(id: string): TurnResult {
  const call: ToolCall = { id, name: "x", input: {} };
  return { kind: "tool_call_requested", calls: [call], usage: usage() };
}

function react(max: number): Task {
  return newTask("do something", SessionId.of("s1"), {
    kind: "react",
    budget: { kind: "per_loop", value: max },
    agent: "",
    toolset: "",
  });
}

describe("ReAct leaf cap binding (rule 6)", () => {
  it("the leaf propagates when its OWN policy is the binding cap", async () => {
    // #125 rule 6: when the ReAct LEAF's OWN policy is the binding cap (no smaller
    // global backstop), the leaf PROPAGATES a typed budget_exhausted — it never
    // self-resolves Continue/Fail at the leaf. driveStrategy surfaces it as a
    // budget_exceeded terminal whose turn count is the leaf's stepsTaken.
    const a = new MockAgent(AgentId.of("test"));
    // The leaf cap is 2; push 3 tool-call turns so the window runs 2 turns and
    // hits the leaf's own cap (no global max_turns set).
    for (let i = 0; i < 3; i += 1) a.push(tcr(`c${i}`));
    const reg = new ScriptedToolRegistry();
    for (let i = 0; i < 3; i += 1) reg.push({ kind: "success", content: "ok" });
    const h = new StandardHarness({
      registry: registryWith({ agent: a }),
      toolRegistry: reg,
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
    });
    // Leaf per_loop{2}, NO global cap → the leaf policy is the binding cap.
    const r = await h.run({ task: react(2) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("budget_exceeded");
      if (r.reason.kind === "budget_exceeded") expect(r.reason.limit_type).toBe("turns");
      // The turn count is the EXHAUSTED LEAF's own stepsTaken (#125
      // budget_exhausted path), which equals the leaf cap N=2 here.
      expect(r.turns).toBe(2); // leaf stopped at its own cap N=2

      // The #125 discriminator: the partial flowed through
      // StrategyOutcome `budget_exhausted` { partialOutput } → driveStrategy's
      // budget_exhausted arm, which materializes the partial as a single
      // assistant Text message. On PRE-#125 code the leaf surfaced the window's
      // RAW session (the tool-call / observation messages), NOT this
      // node-concrete partial JSON — so these assertions FAIL on the old path
      // and PROVE the new budget_exhausted machinery is exercised end-to-end.
      expect(r.session_state).toBeDefined();
      const messages = r.session_state!.messages;
      expect(messages).toHaveLength(1);
      const msg = messages[0]!;
      expect(msg.role).toBe("assistant");
      expect(msg.content.type).toBe("text");
      const text = msg.content.type === "text" ? msg.content.text : "";
      // The documented ReAct partial shape (fork #2): the last FinalResponse
      // text as JSON. This window produced no FinalResponse, so the documented
      // shape carries an empty `last_final_response`.
      expect(text).toBe(reactPartialJson(""));
      // Sanity: the materialized text is genuinely the partial helper's output,
      // i.e. valid JSON tagged with the node — not a free-form transcript.
      const parsed = JSON.parse(text);
      expect(parsed.node).toBe("react");
      expect(parsed.last_final_response).toBe("");
    }
  });
});

// ── Rule 4: a child budget_exhausted reaches the parent, parent unaffected ───

describe("child budget_exhausted reaches the parent as a StrategyOutcome (rule 4)", () => {
  it("the parent's own BudgetContext is unaffected and it can then Complete", () => {
    // The parent pushes its own scope, then a child pushes + exhausts its own.
    const cx = newExecutionContext(ExecutionRegistry.empty());
    pushBudget(cx, { kind: "total_steps", value: 5 }, { kind: "fail" }, "parent");

    // Child scope: capped at 1, exhausts, then propagates a budget_exhausted (as
    // the leaf does in rule 6).
    pushBudget(cx, { kind: "per_loop", value: 1 }, { kind: "escalate" }, "child");
    expect(chargeCurrent(cx, 1).ok).toBe(true);
    const exhausted = chargeCurrent(cx, 1);
    expect(exhausted.ok).toBe(false);
    const childOutcome = exhausted.ok
      ? undefined
      : promoteBudgetExhausted(exhausted.error, reactPartialJson("child partial"));
    // The parent RECEIVES the child outcome as an inspectable StrategyOutcome —
    // never auto-propagated as the parent's own exhaustion.
    expect(childOutcome?.kind).toBe("budget_exhausted");
    popBudget(cx);

    // The parent's scope is untouched (rule 4/7).
    expect(cx.budgets.current()!.stepsTaken).toBe(0);
    // The parent can decide to proceed and Complete within its own budget.
    expect(chargeCurrent(cx, 5).ok).toBe(true);
    expect(resolveCurrent(cx)).toBe("fail");
  });
});
