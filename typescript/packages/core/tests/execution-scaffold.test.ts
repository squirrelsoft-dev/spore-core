/**
 * Composable Execution runtime scaffold (spore-core issue #123).
 *
 * StrategyOutcome + ExecutionContext / BudgetContext / BudgetStack / SpanStack.
 * Scaffold-only slice: these tests exercise the type surface and the threading,
 * NOT enforcement semantics. They mirror the Rust `execution_scaffold_tests`
 * module in `rust/crates/spore-core/src/harness.rs`:
 *   - charge within allowance increments stepsTaken,
 *   - overflow returns the error with correct fields and no mutation,
 *   - unlimited never exhausts,
 *   - remaining() capped + unlimited,
 *   - continuesRemaining() for continue / escalate / fail,
 *   - StrategyOutcome budget_exhausted vs failed discrimination,
 *   - SpanStack push/pop depth,
 *   - a recursive stub threads the shared ExecutionContext + BudgetStack and
 *     returns to baseline depth,
 *   - siblings get distinct BudgetContexts.
 */

import { describe, expect, it } from "vitest";

import {
  BudgetContext,
  BudgetStack,
  ExecutionRegistry,
  InvalidConfiguration,
  SpanStack,
  newExecutionContext,
  type BudgetExhaustedBehavior,
  type BudgetPolicy,
  type ExecutionContext,
  type RunStrategy,
  type StrategyOutcome,
} from "../src/index.js";
import { SpanId } from "../src/observability/types.js";

function continueBehavior(maxContinues: number): BudgetExhaustedBehavior {
  return { kind: "continue", max_continues: maxContinues, on_exhausted: { kind: "fail" } };
}

const fail: BudgetExhaustedBehavior = { kind: "fail" };
const escalate: BudgetExhaustedBehavior = { kind: "escalate" };

// ── BudgetContext.charge ─────────────────────────────────────────────────────

describe("BudgetContext.charge", () => {
  it("within allowance increments stepsTaken", () => {
    const cx = new BudgetContext({ kind: "per_loop", value: 5 }, fail, "loop");
    expect(cx.charge(2).ok).toBe(true);
    expect(cx.stepsTaken).toBe(2);
    expect(cx.charge(3).ok).toBe(true); // exactly to the cap
    expect(cx.stepsTaken).toBe(5);
    expect(cx.remaining()).toBe(0);
  });

  it("overflow returns the error with correct fields and no mutation", () => {
    const policy: BudgetPolicy = { kind: "total_steps", value: 3 };
    const behavior = continueBehavior(2);
    const cx = new BudgetContext(policy, behavior, "phase-x");
    expect(cx.charge(3).ok).toBe(true);

    const res = cx.charge(1);
    expect(res.ok).toBe(false);
    if (res.ok) throw new Error("expected exhaustion");
    expect(res.error).toEqual({
      policy,
      behavior,
      stepsTaken: 3,
      continuesUsed: 0,
      phase: "phase-x",
    });
    // state unchanged on failure
    expect(cx.stepsTaken).toBe(3);
  });

  it("unlimited never exhausts", () => {
    const cx = new BudgetContext({ kind: "unlimited" }, fail, "root");
    expect(cx.charge(Number.MAX_SAFE_INTEGER).ok).toBe(true);
    expect(cx.charge(100).ok).toBe(true);
    expect(cx.remaining()).toBeUndefined();
  });
});

// ── BudgetContext.remaining ──────────────────────────────────────────────────

describe("BudgetContext.remaining", () => {
  it("is capped for a bounded policy and undefined for unlimited", () => {
    const capped = new BudgetContext({ kind: "per_attempt", value: 10 }, fail, "attempt");
    expect(capped.remaining()).toBe(10);
    capped.charge(4);
    expect(capped.remaining()).toBe(6);

    const unlimited = new BudgetContext({ kind: "unlimited" }, fail, "u");
    expect(unlimited.remaining()).toBeUndefined();
  });
});

// ── BudgetContext.continuesRemaining ─────────────────────────────────────────

describe("BudgetContext.continuesRemaining", () => {
  it("counts continues for `continue` and is 0 for escalate / fail", () => {
    const cont = new BudgetContext({ kind: "unlimited" }, continueBehavior(3), "c");
    expect(cont.continuesRemaining()).toBe(3);
    cont.continuesUsed = 2;
    expect(cont.continuesRemaining()).toBe(1);
    cont.continuesUsed = 5; // saturates
    expect(cont.continuesRemaining()).toBe(0);

    const esc = new BudgetContext({ kind: "unlimited" }, escalate, "e");
    expect(esc.continuesRemaining()).toBe(0);

    const f = new BudgetContext({ kind: "unlimited" }, fail, "f");
    expect(f.continuesRemaining()).toBe(0);
  });
});

// ── StrategyOutcome discrimination ───────────────────────────────────────────

describe("StrategyOutcome", () => {
  it("budget_exhausted is distinguishable from failed and complete", () => {
    const exhausted: StrategyOutcome = {
      kind: "budget_exhausted",
      policy: { kind: "per_loop", value: 1 },
      behavior: fail,
      stepsTaken: 1,
      continuesUsed: 0,
      phase: "p",
      partialOutput: "partial",
    };
    const failed: StrategyOutcome = {
      kind: "failed",
      error: new InvalidConfiguration("boom"),
    };
    const complete: StrategyOutcome = { kind: "complete", output: "done" };

    // A budget_exhausted is NOT a failed — callers can distinguish by `kind`.
    const kinds = [exhausted, failed, complete].map((o): string => o.kind);
    expect(kinds).toEqual(["budget_exhausted", "failed", "complete"]);
    expect(kinds).not.toContain("pending"); // the old #119 placeholder is gone
    if (exhausted.kind === "budget_exhausted") {
      expect(exhausted.partialOutput).toBe("partial");
    }
  });
});

// ── SpanStack ────────────────────────────────────────────────────────────────

describe("SpanStack", () => {
  it("push/pop tracks depth", () => {
    const spans = new SpanStack();
    expect(spans.depth()).toBe(0);
    spans.push(new SpanId("a"));
    spans.push(new SpanId("b"));
    expect(spans.depth()).toBe(2);
    expect(spans.pop()).toEqual(new SpanId("b"));
    expect(spans.depth()).toBe(1);
  });
});

// ── Recursive stub threading the shared ExecutionContext ─────────────────────

/**
 * A recursive stub: pushes its own scope, optionally recurses, then pops —
 * modeling how a combinator threads the shared context and gives each node
 * (incl. siblings) its OWN BudgetContext.
 */
class RecursiveStub implements RunStrategy {
  constructor(
    private readonly value: number,
    private readonly children: RecursiveStub[],
  ) {}

  async run(cx: ExecutionContext): Promise<StrategyOutcome> {
    const baseline = cx.budgets.depth();
    cx.budgets.push(new BudgetContext({ kind: "per_loop", value: this.value }, fail, "node"));
    // charge against this node's own scope
    cx.budgets.current()?.charge(1);
    for (const child of this.children) {
      await child.run(cx);
    }
    // pop our scope; depth returns to baseline
    cx.budgets.pop();
    expect(cx.budgets.depth()).toBe(baseline);
    return { kind: "complete", output: "" };
  }
}

describe("recursive stub threading", () => {
  it("threads one ExecutionContext + BudgetStack and returns to baseline depth", async () => {
    const cx = newExecutionContext(ExecutionRegistry.empty());
    expect(cx.budgets.depth()).toBe(0);

    const tree = new RecursiveStub(4, [new RecursiveStub(2, []), new RecursiveStub(3, [])]);

    const outcome = await tree.run(cx);
    expect(outcome.kind).toBe("complete");
    // stack fully unwound after the recursive run
    expect(cx.budgets.depth()).toBe(0);
    // the shared usage/session/spans are reachable through one context
    expect(cx.spans.depth()).toBe(0);
  });
});

// ── Siblings get distinct BudgetContexts ─────────────────────────────────────

describe("BudgetStack", () => {
  it("gives siblings distinct, non-shared BudgetContexts", () => {
    const budgets = new BudgetStack();
    budgets.push(new BudgetContext({ kind: "per_loop", value: 5 }, fail, "sib-a"));
    // mutate sibling A
    budgets.current()?.charge(2);
    const a = budgets.pop();
    expect(a?.stepsTaken).toBe(2);

    // sibling B starts fresh — no shared state with A
    budgets.push(new BudgetContext({ kind: "per_loop", value: 5 }, fail, "sib-b"));
    const b = budgets.current();
    expect(b?.stepsTaken).toBe(0);
    expect(a?.phase).not.toBe(b?.phase);
  });
});
