//! Top-level strategy dispatch + the shared `ExecutionContext` driver.
//!
//! `impl RunStrategy for LoopStrategy` fans a serializable strategy node out to
//! its per-variant body (now living in the matching strategy file —
//! `react.rs` / `plan_execute.rs` / `self_verifying.rs` / `ralph.rs` /
//! `hill_climbing.rs`), and `ExecutionContext` holds the budget/scratch/executor
//! state every body threads. The strategy/budget-promotion helper functions live
//! in `strategy_support.rs`. Pure move — no behavior change.

use super::*;

impl RunStrategy for LoopStrategy {
    fn run<'a>(&'a self, cx: &'a mut ExecutionContext<'_>) -> BoxFut<'a, StrategyOutcome> {
        // The ONLY `match` site: the enum→config delegation (AC1). No central
        // dispatch match remains — each config owns its loop.
        match self {
            LoopStrategy::ReAct(c) => c.run(cx),
            LoopStrategy::PlanExecute(c) => c.run(cx),
            LoopStrategy::SelfVerifying(c) => c.run(cx),
            LoopStrategy::Ralph(c) => c.run(cx),
            LoopStrategy::HillClimbing(c) => c.run(cx),
        }
    }
}

impl ExecutionContext<'_> {
    /// Record a terminal/pause [`RunResult`] from a whole-loop primitive
    /// (ReAct / SelfVerifying / Ralph / HillClimbing): carry the post-run session
    /// into the scratch (so a parent resumes losslessly) and stash the FULL
    /// result in `terminal_override` so the harness entry returns it VERBATIM —
    /// preserving the strategy's typed [`HaltReason`] and accounting. Returns the
    /// matchable [`StrategyOutcome`] for any combinator that recurses into this
    /// node (a wrapping combinator CLEARS the override and builds its own
    /// terminal; see [`take_child_override`](Self::take_child_override)).
    ///
    /// Usage is NOT folded into `self.usage` here: the whole-loop primitive's
    /// RunResult already carries the cumulative usage for its subtree, and it is
    /// returned verbatim as the override, so folding would double-count.
    pub(crate) fn record_terminal(&mut self, result: RunResult) -> StrategyOutcome {
        match &result {
            RunResult::Success { session_state, .. } | RunResult::Failure { session_state, .. } => {
                self.scratch.run_session = session_state.clone();
            }
            RunResult::WaitingForHuman { .. }
            | RunResult::Consult { .. }
            | RunResult::Escalate { .. } => {}
        }
        let outcome = outcome_from_run_result_ref(&result);
        self.scratch.terminal_override = Some(result);
        outcome
    }

    /// Take the FULL terminal [`RunResult`] a child strategy stashed into
    /// `terminal_override` when it returned from `.run(cx)` (#124). A combinator
    /// that recurses per-phase / per-task calls this immediately after each child
    /// `.run(cx)` to fold the child's usage / turns / session back into the shared
    /// execute context. Clearing the override is REQUIRED: the combinator builds
    /// its OWN terminal once the whole loop finishes (via [`finish`](Self::finish)),
    /// and a stale child override would otherwise propagate verbatim and mask it.
    pub(crate) fn take_child_override(&mut self) -> Option<RunResult> {
        self.scratch.terminal_override.take()
    }

    /// A combinator's terminal seam: finalize observability for `result`, restore
    /// the parent `task` into scratch, stash `result` as the override so the
    /// harness entry returns it VERBATIM, and return the matching outcome.
    pub(crate) async fn finish(
        &mut self,
        executor: &dyn StrategyExecutor,
        parent_task: Task,
        result: RunResult,
    ) -> StrategyOutcome {
        executor.finalize(&result).await;
        // #131: a `Consult` propagated from a worker leaf carries the LEAF task,
        // so a host `resume_consult` would resume only that leaf and lose the
        // surrounding walk. As the pause unwinds through each combinator's
        // `finish`, rewrite its `state.task` to the combinator's OWN composed
        // task; by the top it carries the FULL tree, so `resume_consult` re-drives
        // the whole strategy (the in-progress task resumes from `resume_seed`).
        //
        // SC-BUG-1: a HITL pause (`WaitingForHuman`) propagated from a worker leaf
        // ALSO carries the LEAF task (a leaf records its terminal via
        // `record_terminal`, which never rewrites it). Without the same rewrite, a
        // host `resume` runs only that bare leaf and loses the surrounding frame —
        // the SelfVerifying evaluate phase / PlanExecute walk never re-runs (so
        // under `AlwaysAsk` the verify gate silently degrades to a plain executor).
        // Rewrite it on the way up exactly like `Consult`, so `resume_inner`
        // re-drives the whole composed strategy from the approved worker session.
        let result = match result {
            RunResult::Consult {
                request,
                mut state,
                session_id,
                usage,
                turns,
            } => {
                state.task = parent_task.clone();
                RunResult::Consult {
                    request,
                    state,
                    session_id,
                    usage,
                    turns,
                }
            }
            RunResult::WaitingForHuman { request, mut state } => {
                state.task = parent_task.clone();
                RunResult::WaitingForHuman { request, state }
            }
            other => other,
        };
        self.scratch.task = Some(parent_task);
        match &result {
            RunResult::Success { session_state, .. } | RunResult::Failure { session_state, .. } => {
                self.scratch.run_session = session_state.clone();
            }
            _ => {}
        }
        let outcome = outcome_from_run_result_ref(&result);
        self.scratch.terminal_override = Some(result);
        outcome
    }

    /// Push a fresh per-node [`BudgetContext`] scope for `policy`/`behavior`/
    /// `phase` onto `cx.budgets` (#125). Each node — including a sibling — gets
    /// its OWN scope (`steps_taken = 0`), so a node capped at N never spends a
    /// sibling's allowance (rule 1) and a child's exhaustion never touches the
    /// parent scope (rule 4/7). Returns the depth AFTER the push for symmetry
    /// debugging.
    pub fn push_budget(
        &mut self,
        policy: BudgetPolicy,
        behavior: BudgetExhaustedBehavior,
        phase: impl Into<String>,
    ) -> usize {
        let phase = phase.into();
        // #129 (AC2): if a resumed `Continue` checkpoint seed is waiting for THIS
        // phase, reconstruct the scope with its prior `continues_used` (consuming
        // the seed once) instead of zeroing it. The root resumed node pushes
        // first, and the request's `phase` names that node, so the FIRST matching
        // push restores the count. Any other push (or a fresh run) is unaffected.
        let seeded = match &self.scratch.resume_continues {
            Some((seed_phase, _)) if *seed_phase == phase => {
                self.scratch.resume_continues.take().map(|(_, c)| c)
            }
            _ => None,
        };
        let scope = match seeded {
            Some(continues_used) => BudgetContext::resumed(policy, behavior, phase, continues_used),
            None => BudgetContext::new(policy, behavior, phase),
        };
        self.budgets.push(scope);
        self.budgets.depth()
    }

    /// Pop the current per-node budget scope (#125). Always paired with
    /// [`push_budget`](Self::push_budget) so the stack returns to its parent
    /// baseline on ascent.
    pub fn pop_budget(&mut self) -> Option<BudgetContext> {
        self.budgets.pop()
    }

    /// Charge `turns` steps against the CURRENT (innermost) budget scope (#125):
    /// the real enforcement point. `Ok(())` when within allowance; `Err(..)`
    /// carries the budget state at exhaustion. A node with no pushed scope (the
    /// scaffold contexts) never exhausts — charging is a no-op `Ok(())`.
    pub fn charge_current(&mut self, turns: u32) -> Result<(), BudgetExhausted> {
        match self.budgets.current_mut() {
            Some(scope) => scope.charge(turns),
            None => Ok(()),
        }
    }

    /// Resolve the current scope's exhaustion behavior (#125). Walks the chain
    /// (Continue grants a reset; spent continues fall through). A node with no
    /// pushed scope resolves to `Fail` (defensive — should not happen in a wired
    /// run).
    pub fn resolve_current(&mut self) -> ExhaustedResolution {
        match self.budgets.current_mut() {
            Some(scope) => scope.resolve_exhausted(),
            None => ExhaustedResolution::Fail,
        }
    }

    /// SC-5: attempt one [`EscalationMode::AutoContinue`] auto-grant at the
    /// CURRENT scope. Returns `true` when the mode is `AutoContinue`, grants
    /// remain (`auto_grants_used < max_grants`), and the scope was refreshed with
    /// `steps_per_grant` more steps — the caller should then `continue` its loop
    /// IN-PROCESS (the scope is still on the stack). Fires `on_grant` for the
    /// grant. Returns `false` for any other mode, when grants are spent, or when
    /// there is no current scope — the caller then falls through to its existing
    /// pause (`SurfaceToHuman`) / abort (`Autonomous`) handling.
    pub fn try_auto_continue(&mut self, mode: &EscalationMode) -> bool {
        let EscalationMode::AutoContinue {
            max_grants,
            steps_per_grant,
            on_grant,
        } = mode
        else {
            return false;
        };
        let Some(scope) = self.budgets.current_mut() else {
            return false;
        };
        if scope.auto_grants_remaining(*max_grants) == 0 {
            return false;
        }
        scope.grant_auto_continue(*steps_per_grant);
        if let Some(cb) = on_grant {
            cb(AutoGrantInfo {
                grant_number: scope.auto_grants_used,
                steps_granted: *steps_per_grant,
                phase: scope.phase.clone(),
            });
        }
        true
    }
}
