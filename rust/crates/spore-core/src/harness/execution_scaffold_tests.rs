// ============================================================================
// Composable Execution runtime scaffold tests (issue #123)
// ============================================================================
//
// NOTE: "runtime types are never serialized" (no `Serialize`/`Deserialize`
// impl) cannot be asserted with a Rust negative trait bound — there is no
// stable way to test for the ABSENCE of a trait impl. It is enforced
// structurally (the types simply derive no serde traits), not by a test here.
// No fixtures are added for this slice: every type defined here is runtime-only
// and never crosses the wire, so there is nothing to byte-compare across
// languages.
use super::*;
use crate::execution_registry::ExecutionRegistry;
use crate::observability::SpanId;

fn continue_behavior(max_continues: u32) -> BudgetExhaustedBehavior {
    BudgetExhaustedBehavior::Continue {
        max_continues,
        on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
    }
}

// ---- charge: within allowance increments steps_taken -------------------

#[test]
fn charge_within_allowance_increments_steps() {
    let mut cx = BudgetContext::new(
        BudgetPolicy::PerLoop { value: 5 },
        BudgetExhaustedBehavior::Fail,
        "loop",
    );
    assert!(cx.charge(2).is_ok());
    assert_eq!(cx.steps_taken, 2);
    assert!(cx.charge(3).is_ok()); // exactly to the cap
    assert_eq!(cx.steps_taken, 5);
    assert_eq!(cx.remaining(), Some(0));
}

// ---- charge: overflow returns Err with correct fields, no mutation -----

#[test]
fn charge_overflow_returns_err_with_state_and_no_mutation() {
    let mut cx = BudgetContext::new(
        BudgetPolicy::TotalSteps { value: 3 },
        continue_behavior(2),
        "phase-x",
    );
    assert!(cx.charge(3).is_ok());
    let err = cx.charge(1).expect_err("debit past cap must exhaust");
    assert_eq!(
        err,
        BudgetExhausted {
            policy: BudgetPolicy::TotalSteps { value: 3 },
            behavior: continue_behavior(2),
            steps_taken: 3,
            continues_used: 0,
            phase: "phase-x".to_string(),
        }
    );
    // state unchanged on failure
    assert_eq!(cx.steps_taken, 3);
}

// ---- charge: Unlimited never exhausts ----------------------------------

#[test]
fn charge_unlimited_never_exhausts() {
    let mut cx = BudgetContext::new(
        BudgetPolicy::Unlimited,
        BudgetExhaustedBehavior::Fail,
        "root",
    );
    assert!(cx.charge(u32::MAX).is_ok());
    assert!(cx.charge(100).is_ok());
    assert_eq!(cx.remaining(), None);
}

// ---- remaining(): capped + Unlimited -----------------------------------

#[test]
fn remaining_capped_and_unlimited() {
    let mut capped = BudgetContext::new(
        BudgetPolicy::PerAttempt { value: 10 },
        BudgetExhaustedBehavior::Fail,
        "attempt",
    );
    assert_eq!(capped.remaining(), Some(10));
    capped.charge(4).unwrap();
    assert_eq!(capped.remaining(), Some(6));

    let unlimited = BudgetContext::new(BudgetPolicy::Unlimited, BudgetExhaustedBehavior::Fail, "u");
    assert_eq!(unlimited.remaining(), None);
}

// ---- continues_remaining(): Continue / Escalate / Fail -----------------

#[test]
fn continues_remaining_per_behavior() {
    let mut cont = BudgetContext::new(BudgetPolicy::Unlimited, continue_behavior(3), "c");
    assert_eq!(cont.continues_remaining(), 3);
    cont.continues_used = 2;
    assert_eq!(cont.continues_remaining(), 1);
    cont.continues_used = 5; // saturates
    assert_eq!(cont.continues_remaining(), 0);

    let esc = BudgetContext::new(
        BudgetPolicy::Unlimited,
        BudgetExhaustedBehavior::Escalate,
        "e",
    );
    assert_eq!(esc.continues_remaining(), 0);

    let fail = BudgetContext::new(BudgetPolicy::Unlimited, BudgetExhaustedBehavior::Fail, "f");
    assert_eq!(fail.continues_remaining(), 0);
}

// ---- StrategyOutcome variant discrimination ----------------------------

#[test]
fn strategy_outcome_budget_exhausted_distinct_from_failed() {
    let exhausted = StrategyOutcome::BudgetExhausted {
        policy: BudgetPolicy::PerLoop { value: 1 },
        behavior: BudgetExhaustedBehavior::Fail,
        steps_taken: 1,
        continues_used: 0,
        phase: "p".to_string(),
        partial_output: Some("partial".to_string()),
        cause: ExhaustionCause::Budget,
    };
    let failed = StrategyOutcome::Failed(HarnessError::InvalidConfiguration("boom".into()));
    let complete = StrategyOutcome::Complete("done".to_string());

    // A BudgetExhausted is NOT a Failed — callers can distinguish.
    assert!(matches!(exhausted, StrategyOutcome::BudgetExhausted { .. }));
    assert!(!matches!(exhausted, StrategyOutcome::Failed(_)));
    assert!(matches!(failed, StrategyOutcome::Failed(_)));
    assert!(matches!(complete, StrategyOutcome::Complete(_)));
}

// ---- SpanStack push/pop depth ------------------------------------------

#[test]
fn span_stack_push_pop_depth() {
    let mut spans = SpanStack::default();
    assert_eq!(spans.depth(), 0);
    spans.push(SpanId::new("a"));
    spans.push(SpanId::new("b"));
    assert_eq!(spans.depth(), 2);
    assert_eq!(spans.pop(), Some(SpanId::new("b")));
    assert_eq!(spans.depth(), 1);
}

// ---- recursive stub strategy threads ExecutionContext + BudgetStack ----

/// A recursive stub: pushes its own scope, optionally recurses, then pops —
/// modeling how a combinator threads the shared context and gives each node
/// (incl. siblings) its OWN BudgetContext.
struct RecursiveStub {
    children: Vec<RecursiveStub>,
    value: u32,
}

impl RunStrategy for RecursiveStub {
    fn run<'a>(&'a self, cx: &'a mut ExecutionContext<'_>) -> BoxFut<'a, StrategyOutcome> {
        Box::pin(async move {
            let baseline = cx.budgets.depth();
            cx.budgets.push(BudgetContext::new(
                BudgetPolicy::PerLoop { value: self.value },
                BudgetExhaustedBehavior::Fail,
                "node",
            ));
            // charge against this node's own scope
            if let Some(scope) = cx.budgets.current_mut() {
                let _ = scope.charge(1);
            }
            for child in &self.children {
                let _ = child.run(cx).await;
            }
            // pop our scope; depth returns to baseline
            cx.budgets.pop();
            assert_eq!(cx.budgets.depth(), baseline);
            StrategyOutcome::Complete(String::new())
        })
    }
}

#[tokio::test]
async fn recursive_stub_threads_context_stack_returns_to_baseline() {
    let registry = ExecutionRegistry::empty();
    let mut cx = ExecutionContext::new(&registry);
    assert_eq!(cx.budgets.depth(), 0);

    let tree = RecursiveStub {
        value: 4,
        children: vec![
            RecursiveStub {
                value: 2,
                children: vec![],
            },
            RecursiveStub {
                value: 3,
                children: vec![],
            },
        ],
    };

    let outcome = tree.run(&mut cx).await;
    assert!(matches!(outcome, StrategyOutcome::Complete(_)));
    // stack fully unwound after the recursive run
    assert_eq!(cx.budgets.depth(), 0);
    // the shared usage/session/spans are reachable through one context
    assert_eq!(cx.spans.depth(), 0);
}

// ---- siblings get distinct BudgetContexts ------------------------------

#[test]
fn siblings_get_distinct_budget_contexts() {
    let mut budgets = BudgetStack::default();
    budgets.push(BudgetContext::new(
        BudgetPolicy::PerLoop { value: 5 },
        BudgetExhaustedBehavior::Fail,
        "sib-a",
    ));
    // mutate sibling A
    budgets.current_mut().unwrap().charge(2).unwrap();
    let a = budgets.pop().unwrap();
    assert_eq!(a.steps_taken, 2);

    // sibling B starts fresh — no shared state with A
    budgets.push(BudgetContext::new(
        BudgetPolicy::PerLoop { value: 5 },
        BudgetExhaustedBehavior::Fail,
        "sib-b",
    ));
    let b = budgets.current_mut().unwrap();
    assert_eq!(b.steps_taken, 0);
    assert_ne!(a.phase, b.phase);
}
