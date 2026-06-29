//! Strategy/budget-promotion helper functions shared across the `RunStrategy`
//! bodies and `ExecutionContext` (worker-key/toolset resolution, partial-JSON
//! synthesis, budget-exhausted / tool-error-loop promotion, budget grants,
//! stream-view + usage folding). Extracted from `strategies.rs`. Pure move.

use super::*;

/// The worker-agent key for a SelfVerifying inner (Q1c: the evaluate-phase agent
/// defaults to the inner worker's resolved agent). Descends to the worker leaf.
pub(crate) fn executor_worker_key(cfg: &SelfVerifyingConfig) -> String {
    worker_agent_key_of(&cfg.inner)
}

/// Descend a [`LoopStrategy`] tree to the worker leaf's agent key (#124).
pub(crate) fn worker_agent_key_of(ls: &LoopStrategy) -> String {
    match ls {
        LoopStrategy::ReAct(c) => c.agent.0.clone(),
        LoopStrategy::PlanExecute(c) => worker_agent_key_of(&c.execute),
        LoopStrategy::SelfVerifying(c) => worker_agent_key_of(&c.inner),
        LoopStrategy::Ralph(c) => {
            if c.agent.0.is_empty() {
                worker_agent_key_of(&c.inner)
            } else {
                c.agent.0.clone()
            }
        }
        LoopStrategy::HillClimbing(c) => worker_agent_key_of(&c.inner),
    }
}

/// Descend a [`LoopStrategy`] tree to the worker (execute) leaf's toolset handle
/// (#138 AC4-a). Mirrors [`worker_agent_key_of`]: a combinator descends into its
/// EXECUTE child (PlanExecute) / inner (SelfVerifying/Ralph/HillClimbing) so a
/// budget-exhausted pause records the same handle #140 would have on the leaf's
/// own pause (e.g. `"exec-tools"`), not the empty default.
pub(crate) fn worker_toolset_of(ls: &LoopStrategy) -> ToolsetRef {
    match ls {
        LoopStrategy::ReAct(c) => c.toolset.clone(),
        LoopStrategy::PlanExecute(c) => worker_toolset_of(&c.execute),
        LoopStrategy::SelfVerifying(c) => worker_toolset_of(&c.inner),
        LoopStrategy::Ralph(c) => worker_toolset_of(&c.inner),
        LoopStrategy::HillClimbing(c) => worker_toolset_of(&c.inner),
    }
}

/// Clone `ls` with its driving leaf's ReAct toolset re-pointed at
/// [`NO_TOOLS_TOOLSET`], so a plan re-ask run through it advertises NO tools and
/// the planner is forced to answer in message content. Recurses into wrappers
/// (the PLAN side of a nested `PlanExecute`).
pub(crate) fn plan_with_tools_suppressed(ls: &LoopStrategy) -> LoopStrategy {
    let mut cloned = ls.clone();
    match &mut cloned {
        LoopStrategy::ReAct(c) => c.toolset = ToolsetRef(NO_TOOLS_TOOLSET.to_string()),
        LoopStrategy::PlanExecute(c) => *c.plan = plan_with_tools_suppressed(&c.plan),
        LoopStrategy::SelfVerifying(c) => *c.inner = plan_with_tools_suppressed(&c.inner),
        LoopStrategy::Ralph(c) => *c.inner = plan_with_tools_suppressed(&c.inner),
        LoopStrategy::HillClimbing(c) => *c.inner = plan_with_tools_suppressed(&c.inner),
    }
    cloned
}

/// The JSON schema the plan-empty REPAIR forces on a re-ask (#139): the canonical
/// `{ tasks: [string], rationale: string }` plan contract. Delivered as the
/// OpenAI/vLLM `response_format` (or Ollama `format`) so the planner is
/// guided-decoded into valid JSON instead of free prose.
pub(crate) fn plan_output_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "tasks": { "type": "array", "items": { "type": "string" } },
            "rationale": { "type": "string" }
        },
        "required": ["tasks", "rationale"]
    })
}

/// Fill the worker leaf's agent handle of `ls` with `agent` ONLY where the leaf
/// left it empty — Ralph's bare-leaf convenience. An explicitly-declared leaf
/// agent is authoritative and is never replaced (the architect's node wins).
/// Descends the worker child chain to the innermost leaf.
pub(crate) fn fill_empty_worker_agent(ls: &mut LoopStrategy, agent: &AgentRef) {
    match ls {
        LoopStrategy::ReAct(c) => {
            if c.agent.0.is_empty() {
                c.agent = agent.clone();
            }
        }
        LoopStrategy::PlanExecute(c) => fill_empty_worker_agent(&mut c.execute, agent),
        LoopStrategy::SelfVerifying(c) => fill_empty_worker_agent(&mut c.inner, agent),
        LoopStrategy::Ralph(c) => fill_empty_worker_agent(&mut c.inner, agent),
        LoopStrategy::HillClimbing(c) => fill_empty_worker_agent(&mut c.inner, agent),
    }
}

/// The last FinalResponse text from a ReAct window terminal (#125, fork #2): the
/// `Success.output`, or for a `Failure` the last assistant text message on its
/// post-run session state (the partial captured before exhaustion). `None` for
/// non-terminal pauses.
pub(crate) fn last_final_response_text(result: &RunResult) -> Option<String> {
    match result {
        RunResult::Success { output, .. } => Some(output.clone()),
        RunResult::Failure { session_state, .. } => {
            session_state
                .messages
                .iter()
                .rev()
                .find_map(|m| match (&m.role, &m.content) {
                    (crate::model::Role::Assistant, crate::model::Content::Text { text }) => {
                        Some(text.clone())
                    }
                    _ => None,
                })
        }
        _ => None,
    }
}

/// The structured per-node partial used to build the `partial_output` JSON
/// attached to a `StrategyOutcome::BudgetExhausted` on `Escalate` (#125, fork
/// #2). Each node serializes its own shape; `Fail` discards the partial entirely
/// (`partial_output = None`).
pub(crate) fn react_partial_json(last_final_response: &str) -> String {
    serde_json::json!({
        "node": "react",
        "last_final_response": last_final_response,
    })
    .to_string()
}

/// PlanExecute partial: the task list + per-task statuses + ledger (#125, fork
/// #2). `ledger` is the per-task `(id, description, status)` rows already walked.
pub(crate) fn plan_execute_partial_json(task_list: &crate::tasklist::TaskList) -> String {
    let ledger: Vec<_> = task_list
        .tasks
        .iter()
        .map(|t| {
            serde_json::json!({
                "id": t.id.to_string(),
                "description": t.description,
                "status": format!("{:?}", t.status),
            })
        })
        .collect();
    serde_json::json!({
        "node": "plan_execute",
        "tasks": task_list.tasks.len(),
        "ledger": ledger,
    })
    .to_string()
}

/// SelfVerifying partial: the last worker result summary + the last verdict
/// reason (#125, fork #2).
pub(crate) fn self_verifying_partial_json(last_worker_output: &str, last_verdict: &str) -> String {
    serde_json::json!({
        "node": "self_verifying",
        "last_worker_result": last_worker_output,
        "last_verdict": last_verdict,
    })
    .to_string()
}

/// HillClimbing partial: the best candidate value + its score (#125, fork #2),
/// plus the `best_sha` of the commit holding that best artifact
/// (`hill-climbing-loop-correctness` §4.6) so a caller resuming a
/// budget/stagnation/escalation exit knows which commit to inspect. `best_sha`
/// is `null` when the workspace has no usable VCS.
pub(crate) fn hill_climbing_partial_json(best_score: f64, best_sha: Option<&str>) -> String {
    serde_json::json!({
        "node": "hill_climbing",
        "best_candidate": best_score,
        "score": best_score,
        "best_sha": best_sha,
    })
    .to_string()
}

/// Promote a charge-time [`BudgetExhausted`] to a [`StrategyOutcome::BudgetExhausted`]
/// (#125 promotion boundary), attaching `partial_output`. Per fork #1: an
/// `Escalate`-resolved exhaustion carries `Some(partial)`; a `Fail`-resolved one
/// carries `None`. The caller supplies the node-concrete partial JSON; this
/// helper threads the budget state from the charge error.
pub(crate) fn promote_budget_exhausted(
    err: &BudgetExhausted,
    partial_output: Option<String>,
) -> StrategyOutcome {
    StrategyOutcome::BudgetExhausted {
        policy: err.policy.clone(),
        behavior: err.behavior.clone(),
        steps_taken: err.steps_taken,
        continues_used: err.continues_used,
        phase: err.phase.clone(),
        partial_output,
        cause: ExhaustionCause::Budget,
    }
}

/// Promote a ReAct tool-error-loop hard-stop (issue #137) to a
/// [`StrategyOutcome::BudgetExhausted`] carrying [`ExhaustionCause::ToolErrorLoop`]
/// so it flows through the SAME single budget-exhaustion resolution site as a
/// real budget exhaustion, but the Fail / Escalate→Fail terminals report
/// [`HaltReason::ToolErrorLoop`] instead of [`HaltReason::BudgetExceeded`]. The
/// leaf's CONFIGURED `behavior` is carried so the resolution site can honor
/// Fail / Escalate / Continue (a granted `Continue` re-drives the window with a
/// fresh per-tool error allowance, since the counter is loop-local to
/// `run_react_inner`). `steps_taken` is the window's turn count at the break, so
/// the terminal's `turns` reflects work actually done — the breaker does NOT
/// burn the rest of the budget.
pub(crate) fn promote_tool_error_loop(
    leaf_budget: &BudgetPolicy,
    leaf_behavior: &BudgetExhaustedBehavior,
    steps_taken: u32,
    tool: String,
    consecutive_errors: u32,
    partial_output: Option<String>,
) -> StrategyOutcome {
    StrategyOutcome::BudgetExhausted {
        policy: leaf_budget.clone(),
        behavior: leaf_behavior.clone(),
        steps_taken,
        continues_used: 0,
        phase: "react".to_string(),
        partial_output,
        cause: ExhaustionCause::ToolErrorLoop {
            tool,
            consecutive_errors,
        },
    }
}

/// The advisory `available_actions` a COMBINATOR offers on a budget-exhaustion
/// pause (#130, fork C): `[ContinueWithBudget, Skip, Fail]`. The suggested
/// `steps` defaults to the scope's own allowance (or 1 for an uncapped scope).
pub(crate) fn combinator_escalation_actions(err: &BudgetExhausted) -> Vec<EscalationAction> {
    let steps = err.policy.allowance_value().unwrap_or(1);
    vec![
        EscalationAction::ContinueWithBudget { steps },
        EscalationAction::Skip,
        EscalationAction::Fail,
    ]
}

/// The advisory `available_actions` a BARE LEAF offers on a budget-exhaustion
/// pause (#130, fork C): `[ContinueWithBudget, Fail]` — a leaf has no sibling
/// tasks to advance to, so `Skip` is OMITTED.
pub(crate) fn leaf_escalation_actions(err: &BudgetExhausted) -> Vec<EscalationAction> {
    let steps = err.policy.allowance_value().unwrap_or(1);
    vec![
        EscalationAction::ContinueWithBudget { steps },
        EscalationAction::Fail,
    ]
}

/// Promote a charge-time [`BudgetExhausted`] to a `RunResult::WaitingForHuman`
/// (#130 HITL pause boundary). Built ONLY when a node's `Escalate` resolution is
/// consulted under [`EscalationMode::SurfaceToHuman`]; it carries the node's
/// `partial_output` and the advisory `available_actions` (combinators pass
/// `[ContinueWithBudget, Skip, Fail]`; a bare leaf passes
/// `[ContinueWithBudget, Fail]` — fork C). The `PausedState` records the node's
/// `steps_taken` / `continues_used` (on the request) so `resume_inner` can
/// reconstruct the node's [`BudgetContext`] from the request alone (fork E).
///
/// #138 AC2-a: the paused `session_state` is the FULL stalled worker session
/// (`worker_session`) — parallel to how the consult pause preserves the worker
/// conversation — so a budget RESUME re-attaches it as the in-progress task's
/// seed and the worker continues with its real context instead of a partial-only
/// stub. When `worker_session` is empty (a bare-leaf propagate site that has no
/// richer conversation to carry, or a legacy caller), it falls back to the
/// single `partial_output` assistant message so the pre-#138 behavior is intact.
/// #138 AC4-a: `toolset` carries the stalled worker leaf's handle (e.g.
/// `"exec-tools"`), consistent with #140, instead of staying `""`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn promote_budget_exhausted_to_human(
    err: &BudgetExhausted,
    partial_output: Option<String>,
    available_actions: Vec<EscalationAction>,
    session_id: SessionId,
    task: Task,
    budget_used: BudgetSnapshot,
    turn_number: u32,
    // #138 AC2-a: the FULL stalled worker session to carry across the pause.
    worker_session: SessionState,
    // #138 AC4-a: the stalled worker leaf's toolset handle (#140 parity).
    toolset: ToolsetRef,
) -> RunResult {
    // #138 AC2-a: prefer the FULL worker conversation; fall back to the
    // partial-only stub when the carried session is empty (back-compat).
    let session_state = if worker_session.messages.is_empty() {
        SessionState {
            messages: partial_output
                .clone()
                .map(|t| {
                    vec![Message {
                        role: crate::model::Role::Assistant,
                        content: crate::model::Content::Text { text: t },
                    }]
                })
                .unwrap_or_default(),
            ..Default::default()
        }
    } else {
        worker_session
    };
    let request = HumanRequest::BudgetExhausted {
        phase: err.phase.clone(),
        policy: err.policy.clone(),
        steps_taken: err.steps_taken,
        continues_used: err.continues_used,
        partial_output,
        available_actions,
    };
    let task_id = task.id.clone();
    let state = PausedState {
        session_id,
        task_id,
        turn_number,
        session_state,
        pending_tool_calls: Vec::new(),
        approved_results: Vec::new(),
        human_request: Some(request.clone()),
        task,
        budget_used,
        child_state: None,
        // #138 AC4-a: carry the stalled worker leaf's toolset handle (#140
        // parity) so a resume routes the re-driven worker through its scoped
        // catalogue.
        toolset,
    };
    RunResult::WaitingForHuman {
        state: Box::new(state),
        request,
    }
}

/// Raise a budget [`BudgetPolicy`]'s per-scope cap to at least `granted` (#130
/// `ContinueWithBudget` grant). `Unlimited` is left untouched (already
/// uncapped); a `TotalSteps`/`PerLoop`/`PerAttempt` value below `granted` is
/// raised to `granted`. Lower grants are no-ops (never SHRINKS an allowance).
pub(crate) fn grant_budget_policy(policy: &mut BudgetPolicy, granted: u32) {
    match policy {
        BudgetPolicy::Unlimited => {}
        BudgetPolicy::TotalSteps { value }
        | BudgetPolicy::PerLoop { value }
        | BudgetPolicy::PerAttempt { value } => {
            if *value < granted {
                *value = granted;
            }
        }
    }
}

/// Reconstruct a resumed task's strategy tree with its budget caps raised to
/// `granted` (#130 `ContinueWithBudget`). The ReAct leaf caps live on each
/// node's own `budget` [`BudgetPolicy`]; the combinator nodes derive their cap
/// from `task.budget.max_turns`, so BOTH are raised. Fork E: `granted` is
/// `request.steps_taken + steps`, so the restored scope has room for `steps`
/// more steps after the checkpoint.
pub(crate) fn grant_task_budget(task: &mut Task, granted: u32) {
    match task.budget.max_turns {
        Some(max) if max < granted => task.budget.max_turns = Some(granted),
        None => task.budget.max_turns = Some(granted),
        _ => {}
    }
    grant_strategy_budget(&mut task.loop_strategy, granted);
}

/// Recurse a [`LoopStrategy`] tree raising every ReAct leaf's `budget` cap to at
/// least `granted` (#130). The combinator nodes carry no inline policy (they
/// derive it from `task.budget.max_turns`, raised by
/// [`grant_task_budget`]), so this only touches the leaves.
pub(crate) fn grant_strategy_budget(ls: &mut LoopStrategy, granted: u32) {
    match ls {
        LoopStrategy::ReAct(c) => grant_budget_policy(&mut c.budget, granted),
        LoopStrategy::PlanExecute(c) => {
            grant_strategy_budget(&mut c.plan, granted);
            grant_strategy_budget(&mut c.execute, granted);
        }
        LoopStrategy::SelfVerifying(c) => grant_strategy_budget(&mut c.inner, granted),
        LoopStrategy::Ralph(c) => grant_strategy_budget(&mut c.inner, granted),
        LoopStrategy::HillClimbing(c) => grant_strategy_budget(&mut c.inner, granted),
    }
}

/// The `(is_error, content)` view a [`ToolOutput`] surfaces on the coarse
/// [`StreamEvent::ToolResult`] (issue #103, Q5). Mirrors the per-call derivation
/// at the initial emit site so the AfterTool re-emit (SC-9 stream parity) snapshots
/// and compares the SAME projection — by the time `fire_after_tool` runs, every
/// output is a `Success` or `Error` (the pause/escalate variants returned earlier),
/// so the catch-all is defensive (`false`, empty) and never taken on that path.
pub(crate) fn stream_view(output: &ToolOutput) -> (bool, String) {
    match output {
        ToolOutput::Success { content, .. } => (false, content.clone()),
        ToolOutput::Error { message, .. } => (true, message.clone()),
        _ => (false, String::new()),
    }
}

/// Translate a terminal [`RunResult`] into a [`StrategyOutcome`] (#124, Q5):
/// `Success → Complete(output)`, every non-success terminal → `Failed`. A
/// `BudgetExceeded` failure maps to `Failed` here (the budget-enforcement
/// `StrategyOutcome::BudgetExhausted` value is produced by `BudgetContext::charge`
/// at the boundary; full HITL-through-recursion is #130). Operates over a borrow
/// so the source `RunResult` can also be stashed verbatim as the override; the
/// pause variants are handled separately (the override path) and degrade to a
/// typed failure only if they ever reach this mapping.
pub(crate) fn outcome_from_run_result_ref(result: &RunResult) -> StrategyOutcome {
    match result {
        RunResult::Success { output, .. } => StrategyOutcome::Complete(output.clone()),
        RunResult::Failure { reason, .. } => {
            StrategyOutcome::Failed(HarnessError::InvalidConfiguration(format!("{reason:?}")))
        }
        RunResult::WaitingForHuman { .. }
        | RunResult::Consult { .. }
        | RunResult::Escalate { .. } => {
            StrategyOutcome::Failed(HarnessError::InvalidConfiguration(
                "non-terminal outcome reached strategy boundary".to_string(),
            ))
        }
    }
}

/// Fold a sub-run's token usage / turn count into the cumulative `total_usage`
/// and the shared `carried` budget snapshot (#124, R8). `carried.turns` becomes
/// the running MAX of the sub-runs' absolute turn counts. A non-terminal pause
/// (`WaitingForHuman`/`Consult`) carries no fold.
pub(crate) fn fold_usage(
    total_usage: &mut AggregateUsage,
    carried: &mut BudgetSnapshot,
    r: &RunResult,
) {
    let (usage, turns) = match r {
        RunResult::Success { usage, turns, .. }
        | RunResult::Failure { usage, turns, .. }
        | RunResult::Escalate { usage, turns, .. } => (usage, *turns),
        RunResult::WaitingForHuman { .. } => return,
        RunResult::Consult { .. } => return,
    };
    total_usage.input_tokens += usage.input_tokens;
    total_usage.output_tokens += usage.output_tokens;
    total_usage.cache_read_tokens += usage.cache_read_tokens;
    total_usage.cache_write_tokens += usage.cache_write_tokens;
    total_usage.cost_usd += usage.cost_usd;
    carried.input_tokens += usage.input_tokens;
    carried.output_tokens += usage.output_tokens;
    carried.turns = carried.turns.max(turns);
}
