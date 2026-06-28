// ============================================================================
// Per-node budget enforcement + failure isolation tests (issue #125)
// ============================================================================
//
// Covers every rule from the slice plan:
//   1. A node capped at N stops at N WITHOUT killing siblings.
//   2. In-process Continue resets the counter, honors max_continues, then falls
//      through; session/messages unchanged across resets.
//   3. Fail yields partial_output = None.
//   4. A child BudgetExhausted reaches the parent as a StrategyOutcome, never
//      auto-propagated (parent's own scope unaffected; parent can then Complete).
//   5. partial_output concrete per node (4 shapes).
//   6. The ReAct leaf does not carry its own behavior; it propagates to parent.
//   7. Never auto-cascade a child exhaustion into a parent exhaustion.
//
// No new fixtures (fork #3): every type here is runtime-only.
use super::*;

fn continue_then_fail(max_continues: u32) -> BudgetExhaustedBehavior {
    BudgetExhaustedBehavior::Continue {
        max_continues,
        on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
    }
}

// ── resolve_exhausted: Fail → Fail; Escalate → Escalate ──────────────────

#[test]
fn resolve_fail_and_escalate_terminal() {
    let mut fail = BudgetContext::new(
        BudgetPolicy::PerLoop { value: 1 },
        BudgetExhaustedBehavior::Fail,
        "p",
    );
    assert_eq!(fail.resolve_exhausted(), ExhaustedResolution::Fail);

    let mut esc = BudgetContext::new(
        BudgetPolicy::PerLoop { value: 1 },
        BudgetExhaustedBehavior::Escalate,
        "p",
    );
    assert_eq!(esc.resolve_exhausted(), ExhaustedResolution::Escalate);
}

// ── Rule 2: Continue resets counter, honors max_continues, falls through ──

#[test]
fn continue_resets_counter_then_falls_through_to_fail() {
    // ReAct under a parent Continue { max_continues: 2, on_exhausted: Fail }.
    let mut cx = BudgetContext::new(
        BudgetPolicy::TotalSteps { value: 3 },
        continue_then_fail(2),
        "phase",
    );

    // 1st exhaustion → Continue (counter resets to 0, continues_used = 1).
    cx.charge(3).unwrap();
    assert!(cx.charge(1).is_err());
    assert_eq!(cx.resolve_exhausted(), ExhaustedResolution::Continue);
    assert_eq!(cx.steps_taken, 0, "counter reset on continue #1");
    assert_eq!(cx.continues_used, 1);

    // 2nd exhaustion → Continue (counter resets again, continues_used = 2).
    cx.charge(3).unwrap();
    assert!(cx.charge(1).is_err());
    assert_eq!(cx.resolve_exhausted(), ExhaustedResolution::Continue);
    assert_eq!(cx.steps_taken, 0, "counter reset on continue #2");
    assert_eq!(cx.continues_used, 2);

    // 3rd exhaustion → continues spent → fall through to Fail.
    cx.charge(3).unwrap();
    assert!(cx.charge(1).is_err());
    assert_eq!(cx.resolve_exhausted(), ExhaustedResolution::Fail);
    // continues_used does NOT advance past max_continues on the fall-through.
    assert_eq!(cx.continues_used, 2);
}

// ── Continue chain: Continue{1, on: Continue{1, on: Escalate}} ───────────

#[test]
fn continue_chain_shares_counter_then_escalates() {
    // The chain shares ONE `continues_used` counter (it is not reset on a
    // fall-through). Outer Continue{2}: grants 2 continues; once spent, the
    // counter (2) >= the nested Continue{2}'s max, so the nested layer grants
    // nothing and falls straight through to Escalate.
    let behavior = BudgetExhaustedBehavior::Continue {
        max_continues: 2,
        on_exhausted: Box::new(BudgetExhaustedBehavior::Continue {
            max_continues: 2,
            on_exhausted: Box::new(BudgetExhaustedBehavior::Escalate),
        }),
    };
    let mut cx = BudgetContext::new(BudgetPolicy::PerLoop { value: 1 }, behavior, "chain");

    // continue #1 and #2 from the outer Continue's allowance.
    assert_eq!(cx.resolve_exhausted(), ExhaustedResolution::Continue);
    assert_eq!(cx.continues_used, 1);
    assert_eq!(cx.resolve_exhausted(), ExhaustedResolution::Continue);
    assert_eq!(cx.continues_used, 2);
    // Outer spent → fall through to nested Continue{2}; the SHARED counter is
    // already 2 so the nested layer grants nothing → Escalate.
    assert_eq!(cx.resolve_exhausted(), ExhaustedResolution::Escalate);
}

// ── consume_continue resets only the in-memory step counter ──────────────

#[test]
fn consume_continue_resets_steps_and_bumps_counter() {
    let mut cx = BudgetContext::new(
        BudgetPolicy::PerLoop { value: 4 },
        continue_then_fail(5),
        "c",
    );
    cx.charge(4).unwrap();
    assert_eq!(cx.steps_taken, 4);
    cx.consume_continue();
    assert_eq!(cx.steps_taken, 0, "step counter rewound");
    assert_eq!(cx.continues_used, 1, "continue counted");
    // The scope's allowance is intact — a fresh round can charge again.
    assert!(cx.charge(4).is_ok());
}

// ── Rule 3: Fail → partial_output = None; Escalate → Some(partial) ───────

#[test]
fn promote_fail_drops_partial_escalate_keeps_it() {
    let err = BudgetExhausted {
        policy: BudgetPolicy::PerLoop { value: 2 },
        behavior: BudgetExhaustedBehavior::Fail,
        steps_taken: 2,
        continues_used: 0,
        phase: "react".to_string(),
    };
    // The Fail boundary supplies None.
    let failed = promote_budget_exhausted(&err, None);
    match failed {
        StrategyOutcome::BudgetExhausted {
            partial_output,
            steps_taken,
            ..
        } => {
            assert_eq!(partial_output, None, "Fail discards the partial");
            assert_eq!(steps_taken, 2);
        }
        other => panic!("expected BudgetExhausted, got {other:?}"),
    }
    // The Escalate boundary supplies Some(partial).
    let escalated = promote_budget_exhausted(&err, Some(react_partial_json("the answer so far")));
    match escalated {
        StrategyOutcome::BudgetExhausted {
            partial_output: Some(p),
            ..
        } => assert!(p.contains("the answer so far")),
        other => panic!("expected BudgetExhausted with partial, got {other:?}"),
    }
}

// ── Rule 5: each node's partial_output has its documented shape ──────────

#[test]
fn react_partial_shape() {
    let json: serde_json::Value = serde_json::from_str(&react_partial_json("hello world")).unwrap();
    assert_eq!(json["node"], "react");
    assert_eq!(json["last_final_response"], "hello world");
}

#[test]
fn plan_execute_partial_shape() {
    use crate::tasklist::{TaskList, TaskStatus};
    let mut tl = TaskList::default();
    let a = tl.add("task a".to_string(), vec![]).unwrap();
    let _b = tl.add("task b".to_string(), vec![]).unwrap();
    let _ = tl.update(a, Some(TaskStatus::Completed), None);
    let json: serde_json::Value = serde_json::from_str(&plan_execute_partial_json(&tl)).unwrap();
    assert_eq!(json["node"], "plan_execute");
    assert_eq!(json["tasks"], 2);
    let ledger = json["ledger"].as_array().unwrap();
    assert_eq!(ledger.len(), 2, "ledger has one row per task");
    assert_eq!(ledger[0]["description"], "task a");
    assert_eq!(ledger[0]["status"], "Completed");
    assert_eq!(ledger[1]["status"], "Pending");
}

#[test]
fn self_verifying_partial_shape() {
    let json: serde_json::Value = serde_json::from_str(&self_verifying_partial_json(
        "worker output",
        "verdict: not yet",
    ))
    .unwrap();
    assert_eq!(json["node"], "self_verifying");
    assert_eq!(json["last_worker_result"], "worker output");
    assert_eq!(json["last_verdict"], "verdict: not yet");
}

#[test]
fn hill_climbing_partial_shape() {
    let json: serde_json::Value =
        serde_json::from_str(&hill_climbing_partial_json(0.875, Some("abc123"))).unwrap();
    assert_eq!(json["node"], "hill_climbing");
    assert_eq!(json["best_candidate"], 0.875);
    assert_eq!(json["score"], 0.875);
    assert_eq!(json["best_sha"], "abc123");
    // No usable VCS ⇒ best_sha serializes as JSON null.
    let no_vcs: serde_json::Value =
        serde_json::from_str(&hill_climbing_partial_json(0.5, None)).unwrap();
    assert!(no_vcs["best_sha"].is_null());
}

// ── Rule 1: a node capped at N stops at N without killing siblings ───────

#[test]
fn sibling_isolation_fresh_context_per_node() {
    let mut budgets = BudgetStack::default();
    // Sibling A: capped at 2, exhausts.
    budgets.push(BudgetContext::new(
        BudgetPolicy::PerLoop { value: 2 },
        BudgetExhaustedBehavior::Escalate,
        "child-a",
    ));
    budgets.current_mut().unwrap().charge(2).unwrap();
    assert!(
        budgets.current_mut().unwrap().charge(1).is_err(),
        "A exhausts at its own cap N=2"
    );
    let a = budgets.pop().unwrap();
    assert_eq!(a.steps_taken, 2);

    // Sibling B gets a FRESH BudgetContext — A's exhaustion did not bleed in.
    budgets.push(BudgetContext::new(
        BudgetPolicy::PerLoop { value: 4 },
        BudgetExhaustedBehavior::Escalate,
        "child-b",
    ));
    let b = budgets.current_mut().unwrap();
    assert_eq!(b.steps_taken, 0, "sibling B starts fresh (rule 1)");
    assert!(
        b.charge(3).is_ok(),
        "B runs with its own untouched allowance"
    );
}

// ── Rule 4 & 7: a child exhaustion does NOT auto-cascade to the parent ───

#[test]
fn child_exhaustion_does_not_charge_parent_scope() {
    let mut budgets = BudgetStack::default();
    // Parent scope (TotalSteps{5}) that is ALREADY nearly exhausted: it has
    // spent 4 of its 5 steps, leaving EXACTLY ONE remaining. This is the
    // adversarial case for rule 4/7 — if a child's exhaustion auto-cascaded
    // even a single step onto the parent, the parent would be pushed over its
    // own cap and could no longer Complete.
    budgets.push(BudgetContext::new(
        BudgetPolicy::TotalSteps { value: 5 },
        BudgetExhaustedBehavior::Fail,
        "parent",
    ));
    budgets.current_mut().unwrap().charge(4).unwrap();
    assert_eq!(
        budgets.current_mut().unwrap().remaining(),
        Some(1),
        "parent starts with exactly 1 step left"
    );

    // Child descends with its OWN scope (capped at 1) and exhausts.
    budgets.push(BudgetContext::new(
        BudgetPolicy::PerLoop { value: 1 },
        BudgetExhaustedBehavior::Escalate,
        "child",
    ));
    budgets.current_mut().unwrap().charge(1).unwrap();
    assert!(budgets.current_mut().unwrap().charge(1).is_err());
    // The child surfaces a BudgetExhausted value — modelled here by popping
    // its scope. Crucially the parent scope is UNCHARGED by this.
    budgets.pop();

    let parent = budgets.current_mut().unwrap();
    assert_eq!(
        parent.steps_taken, 4,
        "rule 4/7: child exhaustion did NOT auto-charge the parent — its \
             steps_taken is unchanged at 4 (not bumped to 5)"
    );
    assert_eq!(
        parent.remaining(),
        Some(1),
        "the parent STILL has its 1 remaining step after the child exhausted"
    );
    // The parent can spend its last step and Complete — proving the child's
    // exhaustion did NOT push it over its own cap.
    assert!(
        parent.charge(1).is_ok(),
        "parent spends its final step and Completes within its own budget"
    );
    assert!(
        parent.charge(1).is_err(),
        "and only now is the parent itself exhausted (at its own 5, by its own \
             work — never by the child's)"
    );
}

// ── ExecutionContext helpers: push / charge / resolve / pop round-trip ───

#[test]
fn execution_context_charge_and_resolve_round_trip() {
    let registry = crate::execution_registry::ExecutionRegistry::empty();
    let mut cx = ExecutionContext::new(&registry);
    assert_eq!(cx.budgets.depth(), 0);

    cx.push_budget(
        BudgetPolicy::PerLoop { value: 2 },
        continue_then_fail(1),
        "node",
    );
    assert_eq!(cx.budgets.depth(), 1);
    assert!(cx.charge_current(2).is_ok());
    assert!(cx.charge_current(1).is_err(), "scope exhausts at its cap");
    assert_eq!(cx.resolve_current(), ExhaustedResolution::Continue);
    // After the continue, the counter reset → charging is possible again.
    assert!(cx.charge_current(2).is_ok());
    assert!(cx.charge_current(1).is_err());
    assert_eq!(cx.resolve_current(), ExhaustedResolution::Fail);
    let popped = cx.pop_budget();
    assert!(popped.is_some());
    assert_eq!(cx.budgets.depth(), 0);
}

// ── charge_current with no scope is a no-op Ok (scaffold contexts) ───────

#[test]
fn charge_with_no_scope_never_exhausts() {
    let registry = crate::execution_registry::ExecutionRegistry::empty();
    let mut cx = ExecutionContext::new(&registry);
    assert!(cx.charge_current(u32::MAX).is_ok());
    assert_eq!(cx.resolve_current(), ExhaustedResolution::Fail);
}

// ── SC-5: AutoContinue grant mechanics ───────────────────────────────────

#[test]
fn grant_auto_continue_is_additive_and_counted() {
    // The grant raises the scope cap to `steps_taken + steps_per_grant`
    // (additive, NO `steps_taken` rewind), so the loop gets exactly
    // `steps_per_grant` more steps — distinct from `consume_continue`.
    let mut scope = BudgetContext::new(
        BudgetPolicy::PerLoop { value: 2 },
        BudgetExhaustedBehavior::Escalate,
        "react",
    );
    scope.charge(2).unwrap();
    assert!(scope.charge(1).is_err(), "exhausts at the cap");
    assert_eq!(scope.auto_grants_remaining(3), 3);
    scope.grant_auto_continue(2); // cap → steps_taken(2) + 2 = 4
    assert_eq!(scope.auto_grants_used, 1);
    assert_eq!(scope.auto_grants_remaining(3), 2);
    assert!(
        scope.charge(2).is_ok(),
        "two more steps fit after the grant"
    );
    assert!(
        scope.charge(1).is_err(),
        "and it exhausts again at the raised cap"
    );
}

#[test]
fn try_auto_continue_grants_until_capped_then_false() {
    let registry = crate::execution_registry::ExecutionRegistry::empty();
    let mut cx = ExecutionContext::new(&registry);
    cx.push_budget(
        BudgetPolicy::PerLoop { value: 2 },
        BudgetExhaustedBehavior::Escalate,
        "react",
    );

    let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let c2 = count.clone();
    let mode = EscalationMode::AutoContinue {
        max_grants: 2,
        steps_per_grant: 2,
        on_grant: Some(Arc::new(move |info: AutoGrantInfo| {
            assert_eq!(info.steps_granted, 2);
            assert_eq!(info.phase, "react");
            c2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })),
    };

    assert!(cx.charge_current(2).is_ok());
    assert!(cx.charge_current(1).is_err());
    assert!(cx.try_auto_continue(&mode), "grant 1");
    assert!(cx.charge_current(2).is_ok(), "two more steps after grant 1");
    assert!(cx.charge_current(1).is_err());
    assert!(cx.try_auto_continue(&mode), "grant 2");
    assert!(
        !cx.try_auto_continue(&mode),
        "grants spent (max_grants = 2) → falls through to the autonomous terminal"
    );
    assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 2);

    // Other modes never auto-continue.
    assert!(!cx.try_auto_continue(&EscalationMode::Autonomous));
    assert!(!cx.try_auto_continue(&EscalationMode::SurfaceToHuman));
    // max_grants = 0 behaves like Autonomous (no grant).
    assert!(!cx.try_auto_continue(&EscalationMode::AutoContinue {
        max_grants: 0,
        steps_per_grant: 5,
        on_grant: None,
    }));
}
