//! Strategy execution logic extracted from the harness god-module (Phase 6.1):
//! the `RunStrategy` implementations for every built-in strategy
//! (ReAct/PlanExecute/SelfVerifying/Ralph/HillClimbing), the `ExecutionContext`
//! driver methods they share, and the strategy/budget-promotion helper
//! functions. Pure move — no behavior change.


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

impl RunStrategy for ReactConfig {
    /// The leaf: a bounded ReAct turn-loop window. Reads the per-run scratch
    /// (`task`, `run_session`, `run_budget`) and drives one ReAct window through
    /// the executor primitive, threading the result back into the shared usage.
    ///
    /// ## Budget enforcement (#125)
    ///
    /// The leaf pushes its OWN [`BudgetContext`] scope for `self.budget` and
    /// charges the turns the window consumed against it. Per Open Q A-2 (rule 6)
    /// the leaf does NOT carry its own behavior and NEVER self-resolves a
    /// Continue/Fail/Escalate at the leaf — on charge exhaustion it PROPAGATES a
    /// `StrategyOutcome::BudgetExhausted` to its parent (which owns the single
    /// recovery site). The propagated `partial_output` is the window's last
    /// FinalResponse text as JSON (fork #2). A clean (non-budget) terminal is
    /// recorded verbatim through [`record_terminal`](ExecutionContext::record_terminal).
    fn run<'a>(&'a self, cx: &'a mut ExecutionContext<'_>) -> BoxFut<'a, StrategyOutcome> {
        Box::pin(async move {
            let executor = match cx.executor() {
                Ok(e) => e,
                Err(o) => return o,
            };
            let task = cx.current_task();
            let max_iterations = self.max_iterations();
            let mut session_state = std::mem::take(&mut cx.scratch.run_session);
            let budget_used = cx.scratch.run_budget.clone();
            // #124: resolve the worker agent from the registry by THIS leaf's
            // handle (genuine recursion — no `config.agent`). A missing handle is
            // a typed terminal failure.
            let agent = match executor.resolve_agent_ref(&self.agent, &task.session_id) {
                Ok(a) => a,
                Err(result) => {
                    let result = *result;
                    executor.finalize(&result).await;
                    return cx.record_terminal(result);
                }
            };
            // Output-schema delivery + enforcement (issue #139). MIGRATION GATE:
            // only when `enforce_output_schemas` is ON AND this leaf carries
            // `output = Some(..)` do we resolve the schema, DELIVER it (directive
            // seed + the constrained-decoding channel), and pass it (plus the
            // retry budget `N`) into the window for terminal validation. When the
            // gate is OFF or there is no `output`, `output_schema` is `None` and
            // the window behaves byte-identically to pre-#139 (no resolve, no
            // delivery, no validation). The schema is canonicalized to compact
            // key-sorted JSON so its delivered/reported bytes are identical
            // across the four language ports.
            // A FORCED schema (the plan-empty repair re-ask, #139) wins over the
            // global enforce gate and is consumed one-shot; otherwise fall back to
            // the normal `enforce_output_schemas` + `self.output` resolution.
            let output_schema: Option<serde_json::Value> =
                cx.scratch.forced_output_schema.take().or_else(|| {
                    if executor.enforce_output_schemas() {
                        self.output
                            .as_ref()
                            .and_then(|r| cx.registry.resolve_schema(r))
                            .cloned()
                    } else {
                        None
                    }
                });
            let output_schema_max_retries = executor.output_schema_max_retries();
            if let Some(schema) = output_schema.as_ref() {
                // AC1: append the resolved schema to the leaf's directive/system
                // context as a USER message, key-sorted via `canonicalize_json`
                // so the seeded bytes are identical across languages.
                let directive = format!(
                    "Your final response must be a JSON value that conforms to this \
JSON schema: {}",
                    crate::model::canonicalize_json(schema)
                );
                executor
                    .seed_user_message(&mut session_state, &directive)
                    .await;
            }
            // #125/#129: push this leaf's OWN budget scope carrying its CONFIGURED
            // `behavior`. The leaf still never RESOLVES it in the nested case (rule
            // 6: it PROPAGATES a `BudgetExhausted` to its parent, which owns the
            // single recovery site). Carrying the real `behavior` only means the
            // propagated error reports it, so the TOP-LEVEL/bare-leaf resolution
            // site (`drive_strategy`) can honor it (Q1 — a bare leaf self-resolves,
            // a nested leaf does not).
            cx.push_budget(self.budget.clone(), self.behavior.clone(), "react");
            // The leaf CLONES (not takes) the run's stream sink so sibling phases
            // of a combinator parent still inherit it for their own turns, and
            // WRAPS it so every event this leaf emits is attributed to THIS node
            // (kind `react` + the resolved agent handle). The wrapper is what the
            // live agent adapter forwards to, so token deltas carry attribution too.
            let agent_id = (!self.agent.0.is_empty()).then(|| self.agent.0.clone());
            let on_stream = wrap_node_sink(cx.stream.clone(), "react", agent_id);
            // Consume the one-shot execute-step marker (set by the execute walk) so
            // the behavioral tool-call nudge scopes to execute steps only and never
            // leaks into a nested leaf.
            let execute_step = std::mem::take(&mut cx.scratch.in_execute_step);
            let result = executor
                .react_window(
                    task,
                    max_iterations,
                    session_state,
                    budget_used,
                    on_stream,
                    agent,
                    // Issue 2: thread THIS leaf's toolset handle down so the
                    // window dispatches the per-node scoped catalogue (empty
                    // handle ⇒ global-catalogue fallback). Mirrors `agent`.
                    self.toolset.clone(),
                    // Issue #139: thread the resolved output schema (or `None`)
                    // and the retry budget so the window validates the terminal.
                    output_schema,
                    output_schema_max_retries,
                    // SC-10: thread THIS leaf's per-node system-prompt override
                    // (None ⇒ the window keeps using the global prompt).
                    self.system_prompt.clone(),
                    execute_step,
                )
                .await;
            executor.finalize(&result).await;

            // #125: charge the window's turns against this leaf's OWN scope. The
            // leaf POLICY (`self.budget`) — not the global `BudgetLimits` backstop —
            // is the per-node enforcement point. A window stops at
            // `min(max_iterations, task.budget.max_turns)`: when the LEAF cap is the
            // binding constraint (the window consumed >= the leaf policy value), the
            // leaf is exhausted and PROPAGATES a typed `BudgetExhausted` to its
            // parent (rule 6 — the leaf never self-resolves). When the smaller
            // GLOBAL backstop trips first, the legacy `BudgetExceeded` terminal is
            // recorded VERBATIM (the global cap is unchanged, #117 backstop).
            let window_turns = match &result {
                RunResult::Success { turns, .. } | RunResult::Failure { turns, .. } => *turns,
                _ => 0,
            };
            // #137: the window hit the consecutive-tool-error breaker's 2N hard
            // stop. PROPAGATE it through the SAME single budget-exhaustion
            // resolution site (so the node's `behavior` governs Fail/Escalate/
            // Continue), but carry the `ToolErrorLoop` cause so the terminal
            // reports `HaltReason::ToolErrorLoop`, never `BudgetExceeded`. The
            // window's turns are still CHARGED against the leaf scope (accurate
            // accounting), but the breaker stopped EARLY — the remaining budget
            // is NOT burned. Detection is independent of `leaf_cap_binding`.
            if let RunResult::Failure {
                reason:
                    HaltReason::ToolErrorLoop {
                        tool,
                        consecutive_errors,
                    },
                ..
            } = &result
            {
                let tool = tool.clone();
                let consecutive_errors = *consecutive_errors;
                let last_final = last_final_response_text(&result).unwrap_or_default();
                if let RunResult::Failure { session_state, .. } = &result {
                    cx.scratch.run_session = session_state.clone();
                }
                let _ = cx.charge_current(window_turns);
                cx.pop_budget();
                return promote_tool_error_loop(
                    &self.budget,
                    &self.behavior,
                    window_turns,
                    tool,
                    consecutive_errors,
                    Some(react_partial_json(&last_final)),
                );
            }
            let window_hit_budget = matches!(
                &result,
                RunResult::Failure {
                    reason: HaltReason::BudgetExceeded { .. },
                    ..
                }
            );
            // The leaf cap is binding when the window hit a budget terminal AND it
            // consumed at least the leaf's own allowance (so the leaf policy — not a
            // smaller global backstop — is what stopped it).
            let leaf_cap_binding = window_hit_budget
                && matches!(self.budget.allowance_value(), Some(cap) if window_turns >= cap);
            let charge = cx.charge_current(window_turns);
            if leaf_cap_binding || charge.is_err() {
                // The leaf partial is the window's last FinalResponse text.
                let last_final = last_final_response_text(&result).unwrap_or_default();
                // Carry the post-run session so a parent resumes losslessly.
                if let RunResult::Success { session_state, .. }
                | RunResult::Failure { session_state, .. } = &result
                {
                    cx.scratch.run_session = session_state.clone();
                }
                let err = charge.err().unwrap_or_else(|| {
                    // The window itself hit the cap; synthesize the charge error
                    // from the current scope state.
                    cx.budgets
                        .current_mut()
                        .map(|s| BudgetExhausted {
                            policy: s.policy.clone(),
                            behavior: s.behavior.clone(),
                            steps_taken: s.steps_taken,
                            continues_used: s.continues_used,
                            phase: s.phase.clone(),
                        })
                        .unwrap_or_else(|| BudgetExhausted {
                            policy: self.budget.clone(),
                            behavior: self.behavior.clone(),
                            steps_taken: window_turns,
                            continues_used: 0,
                            phase: "react".to_string(),
                        })
                });
                cx.pop_budget();
                // Rule 6: the leaf PROPAGATES — partial carries the last
                // FinalResponse (Escalate semantics, fork #1/#2).
                return promote_budget_exhausted(&err, Some(react_partial_json(&last_final)));
            }
            cx.pop_budget();
            cx.record_terminal(result)
        })
    }
}

impl RunStrategy for PlanExecuteConfig {
    /// Plan→execute DAG executor (#124 recursion seam, #126 ready-set walk).
    ///
    /// GENUINELY recursive: the plan phase dispatches `self.plan.run(cx)` and the
    /// execute phase dispatches `self.execute.run(cx)` ONCE PER TASK. The child
    /// strategy's full loop runs for each phase.
    ///
    /// # #126 — what this body now does
    /// - **Task list source (decision C):** after the plan phase persists the
    ///   captured artifact (so the legacy plan→tasklist replay path is intact),
    ///   the executor LOADS its runnable [`TaskList`] from the persisted
    ///   `task_list` tool store (`load_task_list`) — the ONE authoring path that
    ///   can carry real `blockers`. The plan artifact only seeds an initial
    ///   linear list when nothing was authored via the tool.
    /// - **Cycle re-check (AC5):** `task_list.has_cycle()` is re-checked at
    ///   execute entry (defense in depth — `add_task` already rejects cycles) →
    ///   [`HaltReason::TaskGraphCycle`].
    /// - **Ready-set walk (AC1):** repeatedly pick the LOWEST-id `Pending` task
    ///   whose blockers are all `Completed` (`next_ready`), run it, mark
    ///   `Completed`, repeat — honoring the blocker DAG, deterministic id
    ///   tiebreak. v1 runs ready tasks SEQUENTIALLY.
    /// - **Two-tier context (AC1 isolation):** each step is seeded with Tier-1
    ///   (its TRANSITIVE blockers' final outputs + their ledger rows, decision D)
    ///   plus Tier-2 (the global running ledger, every step). Independent
    ///   branches never appear in a task's seed.
    /// - **Ledger (decision B):** on completion a
    ///   [`StepLedgerEntry`](crate::tasklist::StepLedgerEntry) is appended whose
    ///   `files_touched` is HARNESS-OBSERVED from write/edit tool calls
    ///   (`take_observed_writes`, AC2) — never self-reported. The ledger is
    ///   bounded at [`STEP_LEDGER_MAX_ENTRIES`](crate::tasklist::STEP_LEDGER_MAX_ENTRIES)
    ///   via drop-oldest.
    /// - **Failure cascade (decision A/E, AC3/AC4):** a terminal task failure
    ///   (unrecoverable error OR a `BudgetExhausted` resolving to `Fail`) marks
    ///   its transitive dependents `Blocked` (tracked in the run-local
    ///   `blocked_by_failure` set) and KEEPS scheduling unrelated tasks; at drain
    ///   the run returns [`HaltReason::TasksBlockedByFailure`] with the full
    ///   completed/blocked partition. A run where every task completes is
    ///   `Success`, as before.
    ///
    /// # #138 — resume seeding + skip re-planning
    /// - **AC1 (skip re-planning):** the durable `task_list` (project-scoped, so it
    ///   survives Ralph's per-window session reset, #142) is probed BEFORE the plan
    ///   dispatch. A non-empty list ⇒ the WHOLE plan phase is skipped (no
    ///   `plan_directive` / `run_plan_subtree` / `capture_plan_artifact`); the run
    ///   goes straight to reconcile + the ready-set walk. The plan artifact may be
    ///   ABSENT (AC1-a artifact-optional) — nothing downstream needs it once a list
    ///   exists. `total_usage` starts empty and `carried.turns` stays at the incoming
    ///   `budget_used.turns` when the plan phase is skipped.
    /// - **AC2 (seed the stalled worker):** a budget/consult resume carries the
    ///   stalled worker session in the phase-agnostic [`RunScratch::resume_seed`].
    ///   For execute-phase exhaustion the single `InProgress` task is reset to
    ///   `Pending` and its step is seeded from the carried session (the
    ///   InProgress→Pending→complete machinery), so the worker continues mid-loop.
    /// - **AC3 (plan-phase exhaustion):** when the seed is present AND the durable
    ///   list has NO `InProgress` task (option iii — the exhaustion was the plan
    ///   phase, list still empty per AC3-b), the PLAN session is seeded from the
    ///   carried conversation instead of a fresh base session.
    ///
    /// No `// SPEC QUESTION:` markers — all five forks (A–E) were resolved.
    fn run<'a>(&'a self, cx: &'a mut ExecutionContext<'_>) -> BoxFut<'a, StrategyOutcome> {
        Box::pin(async move {
            use crate::tasklist::{
                push_step_ledger, render_step_ledger, StepLedgerEntry, TaskStatus,
            };
            use std::collections::{HashMap, HashSet};

            let executor = match cx.executor() {
                Ok(e) => e,
                Err(o) => return o,
            };
            let task = cx.current_task();
            let session_id = task.session_id.clone();
            // The incoming shared execute session ( `[user: task.instruction]` ).
            let base_session = std::mem::take(&mut cx.scratch.run_session);
            let budget_used = cx.scratch.run_budget.clone();
            // Save the parent sink and install a node-attributing wrapper for the
            // phases (#stream): children inherit `cx.stream` and stream through it,
            // each event gaining a `plan_execute` hop in its node path. Restored to
            // the parent on every exit (the `cx.stream = on_stream;` sites below).
            let on_stream = cx.stream.take();
            cx.stream = wrap_node_sink(on_stream.clone(), "plan_execute", None);

            // #131/#138: the phase-agnostic resume seed (the stalled worker
            // conversation carried from a resumed consult or budget pause). Taken
            // BEFORE the plan phase so AC3 (plan-phase exhaustion) can seed the
            // plan session from it, and AC2 (execute-phase exhaustion) can
            // re-attach it to the single `InProgress` task.
            let mut resume_seed_session = cx.scratch.resume_seed.take();

            // #138 AC1 (skip re-planning): probe the DURABLE task_list BEFORE the
            // plan phase. If a non-empty list already exists (e.g. a prior window
            // authored it, #142 makes it survive Ralph's per-window session reset),
            // the plan phase is REDUNDANT — go straight to reconcile + the ready-set
            // walk. This is the core fix: a budget/consult re-entry no longer burns
            // its grant re-planning a graph that is already durable. The plan
            // artifact under `PLAN_EXECUTE_EXTRAS_KEY` may be ABSENT in this case
            // (AC1-a artifact-optional) — nothing downstream requires it once a
            // task_list exists.
            let preexisting = executor.load_task_list(&session_id).await;
            // Skip the plan phase ONLY when this is an explicit CONTINUATION that
            // still has UNFINISHED work to resume — never on the mere presence of a
            // durable list. The durable task_list is keyed by the stable project
            // namespace (#142), so it survives across independent re-entries that
            // share a workspace: separate `run()` calls on one long-lived harness
            // (a conversational host), and HillClimbing iterations (each a fresh
            // neighbor candidate). Treating "a list exists" as "resume it" made
            // those re-plan into nothing — a second `run()`/iteration found the
            // prior, fully-`Completed` graph, walked to no ready task, and drained
            // to a silent empty `Success`. Continuation is therefore SIGNALLED, not
            // inferred: a `resume_seed` (consult/budget/clarification/HITL resume)
            // or an explicit `continue_task_list` flag set by Ralph (cross-window
            // continuation of one logical task) / the `Skip` resolution (advance
            // past the skipped task). The signal is CONSUMED here so it never leaks
            // into a nested per-task PlanExecute (which must plan fresh).
            let is_continuation =
                resume_seed_session.is_some() || std::mem::take(&mut cx.scratch.continue_task_list);
            let has_unfinished_work = preexisting.as_ref().is_some_and(|l| {
                l.tasks
                    .iter()
                    .any(|t| matches!(t.status, TaskStatus::Pending | TaskStatus::InProgress))
            });
            let skip_plan = is_continuation && has_unfinished_work;

            let (mut task_list, mut total_usage, mut carried) = if skip_plan {
                // ── AC1: skip-plan path. The list is the durable source of truth.
                let mut task_list =
                    preexisting.expect("skip_plan implies a non-empty preexisting list");

                // AC5 (defense in depth): re-check the durable graph for cycles.
                if task_list.has_cycle() {
                    cx.stream = on_stream;
                    let result = RunResult::Failure {
                        reason: HaltReason::TaskGraphCycle {
                            reason: "persisted task graph contains a directed cycle".into(),
                        },
                        session_id,
                        usage: AggregateUsage::default(),
                        turns: budget_used.turns,
                        session_state: SessionState::default(),
                    };
                    return cx.finish(executor, task, result).await;
                }

                // A.6 deep-resume: reconcile already-Completed tasks so they are
                // NOT re-run (handles the dedup the plan path would have done).
                executor
                    .reconcile_completed_tasks(&session_id, &mut task_list)
                    .await;
                executor.persist_task_list(&session_id, &task_list).await;

                // No plan turn ran: usage starts empty and the shared budget is
                // carried unchanged (turns stay at the incoming `budget_used.turns`).
                (task_list, AggregateUsage::default(), budget_used.clone())
            } else {
                // ── AC3 / fresh-plan path: run the plan phase. ──────────────────
                // #138 AC3 (plan-phase exhaustion, infer-from-task_list option iii):
                // when a resume seed is present AND the durable list has no
                // `InProgress` task, the exhaustion happened in the PLAN phase
                // before authoring any task (AC3-b: the list is still empty). Seed
                // the PLAN session from the carried conversation so the planner
                // CONTINUES on it instead of starting fresh — and consume the seed
                // here so the execute-phase re-attach below does not also fire.
                // Fresh plan: clear any stale durable list lingering in this
                // project-scoped namespace so the freshly-authored list is the sole
                // source of truth. The post-plan `load_task_list` below prefers the
                // durable list while non-empty, so a stale completed/abandoned list
                // would otherwise re-suppress the walk and drain to empty. Harmless
                // no-op when the list is already empty (e.g. a plan-phase resume
                // that only seeds the planner conversation).
                executor.clear_task_list(&session_id).await;
                let plan_resume = resume_seed_session.is_some()
                    && preexisting
                        .as_ref()
                        .is_none_or(|l| l.tasks.iter().all(|t| t.status != TaskStatus::InProgress));
                let directive = executor.plan_directive(&task.instruction);
                // The resume seed (if any) is consumed ONCE here; the re-ask loop
                // below re-plans from this same base session each attempt.
                let plan_base = if plan_resume {
                    resume_seed_session
                        .take()
                        .unwrap_or_else(|| base_session.clone())
                } else {
                    base_session.clone()
                };

                // Plan-empty re-ask loop. A reasoning model can route its plan into
                // the reasoning channel, leaving the message CONTENT empty so the
                // strict + prose capture yields no tasks. Rather than halting on the
                // FIRST empty plan, re-ask up to `plan_empty_max_retries` more times
                // — nudging the planner to emit the plan as message content — before
                // falling through to `EmptyPlan`. A cap of `0` (the default)
                // preserves the historical single-shot behaviour byte-for-byte: the
                // loop runs once and the `attempt >= max` guard fires immediately on
                // an empty plan, with zero discarded usage to fold.
                let plan_empty_max_retries = executor.plan_empty_max_retries();
                // The re-ask runs with tools SUPPRESSED: a reasoning model offered
                // the full toolset tends to derail into a tool call (or pure
                // reasoning) instead of emitting the plan; with no tools it must
                // answer in content. The first attempt keeps tools (it may research).
                // Built ONLY when re-asks are enabled — at the default cap of `0` the
                // loop is single-shot and this clone would be wasted (keeps the
                // historical no-retry path byte-identical and allocation-free).
                let retry_plan =
                    (plan_empty_max_retries > 0).then(|| plan_with_tools_suppressed(&self.plan));
                // Usage from discarded (empty) attempts, folded into whichever
                // artifact terminates the loop so the carried budget counts them.
                let mut discarded_usage = AggregateUsage::default();
                let mut attempt: u32 = 0;
                let (mut outcome, mut task_list) = loop {
                    let active_plan: &LoopStrategy = if attempt == 0 {
                        &self.plan
                    } else {
                        retry_plan.as_ref().unwrap_or(&self.plan)
                    };
                    // Per-attempt plan session: the seeded directive, plus — on a
                    // retry — a corrective nudge demanding the plan as content.
                    let mut plan_session = plan_base.clone();
                    executor
                        .seed_user_message(&mut plan_session, &directive)
                        .await;
                    if attempt > 0 {
                        // The trailing `/no_think` is a soft thinking-suppression
                        // hint: reasoning models that honor it (e.g. Qwen) stop
                        // routing the plan into the reasoning channel and emit it as
                        // content; models that don't simply ignore the token.
                        let nudge = "Your previous response produced no usable plan \
(no JSON object, and no tasks were authored). Output the plan NOW as a single JSON \
object in your MESSAGE CONTENT: {\"tasks\": [<ordered step strings>], \"rationale\": \
<string>}. Do NOT put the plan only in your reasoning, and do NOT call any tools. \
/no_think";
                        executor.seed_user_message(&mut plan_session, nudge).await;
                    }
                    // The plan phase runs under the plan sub-strategy's OWN declared
                    // budget (e.g. a ReAct `PerLoop{4}`); the global `max_turns` is
                    // only the outer backstop.
                    let plan_task = Task {
                        id: task.id.clone(),
                        instruction: directive.clone(),
                        session_id: session_id.clone(),
                        budget: task.budget.clone(),
                        loop_strategy: active_plan.clone(),
                    };
                    // #139: the trusted first attempt runs UNCONSTRAINED; each
                    // repair re-ask FORCES the plan schema (→ `response_format`), so
                    // a model that answered in prose is now made to emit valid
                    // `{tasks, rationale}` JSON.
                    let forced_schema = (attempt > 0).then(plan_output_schema);
                    let plan_result = executor
                        .run_plan_subtree(
                            active_plan,
                            plan_task,
                            plan_session,
                            budget_used.clone(),
                            cx.stream.clone(),
                            forced_schema,
                        )
                        .await;
                    let (plan_output, plan_usage, plan_turns) = match plan_result {
                        Some(RunResult::Success {
                            output,
                            usage,
                            turns,
                            ..
                        }) => (output, usage, turns),
                        Some(other) => {
                            cx.stream = on_stream.clone();
                            return cx.finish(executor, task.clone(), other).await;
                        }
                        None => {
                            cx.stream = on_stream.clone();
                            let result = RunResult::Failure {
                                reason: HaltReason::PlanPhaseFailed {
                                    error: crate::plan::PlanPhaseError::PlanningTurnFailed {
                                        message: "plan sub-strategy produced no terminal".into(),
                                    },
                                },
                                session_id: session_id.clone(),
                                usage: AggregateUsage::default(),
                                turns: budget_used.turns,
                                session_state: SessionState::default(),
                            };
                            return cx.finish(executor, task.clone(), result).await;
                        }
                    };

                    // Capture + persist the artifact from the plan child's output
                    // (R3/R4/R11) — the harness-side machinery, no model turn.
                    let outcome = match executor
                        .capture_plan_artifact(&session_id, &plan_output, plan_usage, plan_turns)
                        .await
                    {
                        Ok(outcome) => outcome,
                        Err(failure) => {
                            cx.stream = on_stream.clone();
                            return cx.finish(executor, task.clone(), failure).await;
                        }
                    };

                    // #126 decision C: the runnable task list comes from the
                    // persisted `task_list` tool store (the ONE authoring path — it
                    // can carry real blockers). Fall back to the linear plan-artifact
                    // bridge only when nothing was authored via the tool (back-compat
                    // with the #59/#124 plan-only path and its replay fixtures).
                    let task_list = match executor.load_task_list(&session_id).await {
                        Some(persisted) if !persisted.tasks.is_empty() => persisted,
                        _ =>
                        {
                            #[allow(deprecated)]
                            crate::tasklist::plan_artifact_to_task_list(&outcome.artifact)
                        }
                    };

                    if !task_list.tasks.is_empty() {
                        break (outcome, task_list);
                    }

                    // Empty plan: re-ask while attempts remain, else halt with
                    // `EmptyPlan` (reporting every attempt's usage).
                    if attempt >= plan_empty_max_retries {
                        cx.stream = on_stream;
                        let mut usage = outcome.usage;
                        usage.input_tokens += discarded_usage.input_tokens;
                        usage.output_tokens += discarded_usage.output_tokens;
                        let result = RunResult::Failure {
                            reason: HaltReason::EmptyPlan,
                            session_id,
                            usage,
                            turns: outcome.turns,
                            session_state: SessionState::default(),
                        };
                        return cx.finish(executor, task, result).await;
                    }
                    discarded_usage.input_tokens += outcome.usage.input_tokens;
                    discarded_usage.output_tokens += outcome.usage.output_tokens;
                    attempt += 1;
                };

                // Fold any discarded-attempt usage into the successful artifact so
                // the carried budget reflects EVERY plan turn that actually ran.
                outcome.usage.input_tokens += discarded_usage.input_tokens;
                outcome.usage.output_tokens += discarded_usage.output_tokens;

                // #126 AC5: re-check the WHOLE graph for cycles at execute entry.
                if task_list.has_cycle() {
                    cx.stream = on_stream;
                    let result = RunResult::Failure {
                        reason: HaltReason::TaskGraphCycle {
                            reason: "persisted task graph contains a directed cycle".into(),
                        },
                        session_id,
                        usage: outcome.usage,
                        turns: outcome.turns,
                        session_state: SessionState::default(),
                    };
                    return cx.finish(executor, task, result).await;
                }

                executor.persist_task_list(&session_id, &task_list).await;

                // Carry the shared budget past the plan turn.
                let mut carried = budget_used.clone();
                carried.turns = outcome.turns;
                carried.input_tokens += outcome.usage.input_tokens;
                carried.output_tokens += outcome.usage.output_tokens;

                // A.6 deep-resume (Q2): reconcile against the durable checkpoint so
                // already-Completed tasks are not re-run.
                executor
                    .reconcile_completed_tasks(&session_id, &mut task_list)
                    .await;

                (task_list, outcome.usage, carried)
            };

            // #131 consult / #138 budget re-drive: a resumed pause carries the
            // stalled worker conversation in `resume_seed_session`. The stalled task
            // is the single `InProgress` task in the durable list (PlanExecute marks
            // a task InProgress before running it). Reset it to `Pending` so
            // `next_ready` re-schedules it, and remember its id so its step uses the
            // carried session instead of a fresh one — the worker then continues
            // mid-loop and its evaluator still runs. (When the seed was already
            // consumed by the AC3 plan-resume path above, this is a no-op.)
            let mut consult_resume_session = resume_seed_session;
            let consult_resume_task: Option<u32> = if consult_resume_session.is_some() {
                let in_progress = task_list
                    .tasks
                    .iter()
                    .find(|t| t.status == TaskStatus::InProgress)
                    .map(|t| t.id);
                if let Some(tid) = in_progress {
                    // Reset directly (bypassing `update`'s forward-only transition
                    // guard, which rejects InProgress->Pending): the resume
                    // legitimately re-schedules the in-flight task.
                    if let Some(t) = task_list.tasks.iter_mut().find(|t| t.id == tid) {
                        t.status = TaskStatus::Pending;
                    }
                } else {
                    // No in-progress task to resume (out-of-contract); drop the
                    // seed so the walk proceeds normally rather than stalling.
                    consult_resume_session = None;
                }
                in_progress
            } else {
                None
            };

            let mut last_output = String::new();
            let mut last_state = SessionState::default();

            // #126 Tier-1/Tier-2 + cascade run-local state.
            // - `final_outputs`: each completed task's `Success.output` (Tier-1).
            // - `ledger`: the Tier-2 global running ledger (bounded, drop-oldest).
            // - `ledger_elided`: sticky flag set the first time entries are dropped.
            // - `blocked_by_failure`: tasks cascade-Blocked by a terminal failure
            //   (decision E — the run scratch, NOT a TaskStatus variant).
            let mut final_outputs: HashMap<u32, String> = HashMap::new();
            let mut ledger: Vec<StepLedgerEntry> = Vec::new();
            let mut ledger_elided = false;
            let mut blocked_by_failure: HashSet<u32> = HashSet::new();
            // The first terminal failure that triggered a cascade (decision A).
            let mut first_failure: Option<(u32, String)> = None;

            // #125: PlanExecute owns a budget scope for its execute phase.
            let plan_policy = match task.budget.max_turns {
                Some(max) => BudgetPolicy::TotalSteps { value: max },
                None => BudgetPolicy::Unlimited,
            };
            cx.push_budget(plan_policy, self.behavior.clone(), "plan_execute");

            // Positional count for the OnTaskAdvance hook. No longer "stable": the
            // execute agent can append tasks mid-walk via the `task_list` tool, so
            // this is refreshed after each step reload below.
            let mut total_tasks = task_list.tasks.len();

            // ── Phase 2: ready-set DAG walk (#126). ─────────────────────────
            //
            // Repeatedly pick the lowest-id Pending task whose blockers are all
            // Completed. A task whose blocker FAILED is cascade-Blocked (so it is
            // no longer Pending and never becomes ready). When no Pending task is
            // ready, the walk drains.
            loop {
                let Some(task_id) = task_list.next_ready() else {
                    break;
                };
                let index = task_list
                    .tasks
                    .iter()
                    .position(|t| t.id == task_id)
                    .unwrap_or(0);
                let instruction = task_list.tasks[index].description.clone();

                // Mark InProgress and re-persist (Q4).
                let _ = task_list.update(task_id, Some(TaskStatus::InProgress), None);
                executor.persist_task_list(&session_id, &task_list).await;

                // Fire OnTaskAdvance (pre, mutable). The hook may rewrite the
                // step instruction; the (possibly mutated) instruction seeds the
                // execute sub-strategy.
                let mut step_task = Task {
                    id: task.id.clone(),
                    instruction,
                    session_id: session_id.clone(),
                    budget: task.budget.clone(),
                    loop_strategy: (*self.execute).clone(),
                };
                executor
                    .fire_task_advance(&session_id, &mut step_task, index, total_tasks)
                    .await;

                // #131 consult re-drive: if THIS is the task that was consulting,
                // resume its worker from the carried conversation (answer already
                // injected) instead of a fresh instruction-seeded session — the
                // worker continues mid-loop and its evaluator still runs. Otherwise
                // build the normal fresh, Tier-1/Tier-2-seeded step session.
                let step_session = if Some(task_id) == consult_resume_task {
                    consult_resume_session
                        .take()
                        .unwrap_or_else(|| base_session.clone())
                } else {
                    // #126 Tier-1 scoped context: seed this step from a FRESH copy
                    // of the base session (NOT a forward-folded shared transcript —
                    // that breaks on a DAG, #126) plus, for THIS task's transitive
                    // blockers only, their final outputs + their ledger rows.
                    // Independent branches never appear (AC1 isolation).
                    let mut step_session = base_session.clone();
                    let blockers = task_list.transitive_blockers(task_id);
                    if !blockers.is_empty() {
                        let blocker_set: HashSet<u32> = blockers.iter().copied().collect();
                        // Tier-1: transitive blockers' final outputs (ascending id).
                        let mut tier1_lines: Vec<String> = Vec::new();
                        for b in &blockers {
                            if let Some(out) = final_outputs.get(b) {
                                tier1_lines.push(format!("#{b} result: {out}"));
                            }
                        }
                        if !tier1_lines.is_empty() {
                            let block =
                                format!("Results from upstream tasks:\n{}", tier1_lines.join("\n"));
                            executor.seed_user_message(&mut step_session, &block).await;
                        }
                        // Tier-1 ledger: the Tier-2 ledger rows for this set.
                        let scoped: Vec<StepLedgerEntry> = ledger
                            .iter()
                            .filter(|e| blocker_set.contains(&e.task_id))
                            .cloned()
                            .collect();
                        if let Some(block) = render_step_ledger(&scoped, false) {
                            executor.seed_user_message(&mut step_session, &block).await;
                        }
                    }

                    // #126 Tier-2: inject the FULL global running ledger into EVERY
                    // step (with the static elision marker once entries dropped).
                    if let Some(block) = render_step_ledger(&ledger, ledger_elided) {
                        executor.seed_user_message(&mut step_session, &block).await;
                    }

                    // Finally seed this step's own instruction.
                    executor
                        .seed_user_message(&mut step_session, &step_task.instruction)
                        .await;
                    step_session
                };

                // #126 AC2: clear the observed-write accumulator so this task's
                // files_touched reflect ONLY the writes this step issues.
                executor.clear_observed_writes();

                cx.scratch.task = Some(step_task);
                cx.scratch.run_session = step_session;
                cx.scratch.run_budget = carried.clone();
                // Mark this as an EXECUTE step so the leaf's behavioral tool-call
                // nudge applies — but ONLY when other tasks remain Pending. The
                // FINAL task is the synthesis/report step, which legitimately calls
                // no tools (it answers from the ledger); nudging it there discards a
                // good report and pushes the model into unparseable "calls".
                let more_tasks_pending = task_list
                    .tasks
                    .iter()
                    .any(|t| t.status == TaskStatus::Pending);
                cx.scratch.in_execute_step = more_tasks_pending;
                let carried_before = carried.turns;
                let step_outcome = self.execute.run(cx).await;

                // The execute sub-strategy's agent may have mutated the SHARED
                // durable task list during this step via the `task_list` tool
                // (notably `add_task`) — that tool and this walk read/write the
                // SAME project-namespaced RunStore key. Reload it before applying
                // this step's outcome so those edits survive. The walk's own
                // in-flight state is already on disk (the `InProgress` mark was
                // persisted before the step ran), so the reloaded list is the
                // UNION of "what the walk persisted" and "what the agent added";
                // re-adopting it lets `next_ready` schedule the new tasks. WITHOUT
                // this, the walk's stale in-memory copy would overwrite the agent's
                // additions on the very next persist (mark-complete / cascade-block)
                // and the added tasks would never run. `add` validates blockers at
                // tool time (no self-block / unknown-id / cycle), so the merged
                // graph stays acyclic — no re-check needed. No durable store ⇒
                // `None` ⇒ keep the in-memory list (the agent couldn't have
                // persisted anything either, so there is nothing to merge).
                if let Some(reloaded) = executor.load_task_list(&session_id).await {
                    task_list = reloaded;
                    total_tasks = task_list.tasks.len();
                }

                // ── BudgetExhausted (#125 rule 4/7) — resolve THIS scope. ────
                if let StrategyOutcome::BudgetExhausted {
                    steps_taken: leaf_steps,
                    ..
                } = &step_outcome
                {
                    // The execute LEAF already counted the worker's true consumed
                    // turns (ABSOLUTE — including the incoming `carried_before`
                    // floor) and carried them in the outcome's `steps_taken`. Use
                    // THAT for the grant accounting, NOT the parent `plan_execute`
                    // scope's `steps_taken`: that scope is `Unlimited` when the Task
                    // has `max_turns = None` (so its `steps_taken` is 0), which made
                    // the resume grant `granted = 0 + steps` a no-op and stalled the
                    // worker on the same window every grant. Everything ELSE on `err`
                    // stays the parent scope's: `phase` must remain "plan_execute" so
                    // the resume-seed `continues_used` restore matches in
                    // `push_budget`, and `continues_used` is the combinator's
                    // (`max_continues`) count, not the leaf's.
                    let leaf_steps = *leaf_steps;
                    let err = cx
                        .budgets
                        .current_mut()
                        .map(|s| BudgetExhausted {
                            policy: s.policy.clone(),
                            behavior: s.behavior.clone(),
                            steps_taken: leaf_steps,
                            continues_used: s.continues_used,
                            phase: s.phase.clone(),
                        })
                        .expect("plan_execute scope pushed above");
                    let resolution = cx.resolve_current();
                    match resolution {
                        // #126 AC4: a budget-`Fail` task cascades IDENTICALLY to an
                        // error-failed one — block its transitive dependents and
                        // keep scheduling unrelated tasks.
                        ExhaustedResolution::Fail => {
                            let _ = task_list.update(task_id, Some(TaskStatus::Blocked), None);
                            for dep in task_list.transitive_dependents(task_id) {
                                let _ = task_list.update(dep, Some(TaskStatus::Blocked), None);
                                blocked_by_failure.insert(dep);
                            }
                            blocked_by_failure.insert(task_id);
                            executor.persist_task_list(&session_id, &task_list).await;
                            if first_failure.is_none() {
                                first_failure = Some((
                                    task_id,
                                    format!("budget exhausted (Fail): {:?}", err.policy),
                                ));
                            }
                            continue;
                        }
                        // #129: a granted `Continue` loops IN-PROCESS — the scope's
                        // `resolve_current` already reset `steps_taken` and bumped
                        // `continues_used`. Reset this task to `Pending` and re-enter
                        // the ready-set walk so it runs again under the refreshed
                        // scope allowance (NO serialization — AC3). `max_continues`
                        // bounds the loop: once continues are spent, the chain falls
                        // through to `Fail`/`Escalate`.
                        ExhaustedResolution::Continue => {
                            let _ = task_list.update(task_id, Some(TaskStatus::Pending), None);
                            executor.persist_task_list(&session_id, &task_list).await;
                            continue;
                        }
                        // Escalate: under `Autonomous`, surface the partial and abort
                        // the run. Under `SurfaceToHuman` (#130) the node PAUSES with
                        // a `BudgetExhausted` request via the `terminal_override`
                        // seam instead of propagating up.
                        ExhaustedResolution::Escalate => {
                            let mode = executor.escalation_mode();
                            // SC-5: AutoContinue auto-grants this scope in-process
                            // (bounded by `max_grants`, fires `on_grant`) and
                            // re-walks the ready set — mirroring the `Continue` arm
                            // above. Once grants are spent it falls through to the
                            // surface/abort handling below.
                            if cx.try_auto_continue(&mode) {
                                let _ = task_list.update(task_id, Some(TaskStatus::Pending), None);
                                executor.persist_task_list(&session_id, &task_list).await;
                                continue;
                            }
                            let surface = matches!(mode, EscalationMode::SurfaceToHuman);
                            // #138 AC2-b: under SurfaceToHuman the task is NOT
                            // permanently failed — it pauses awaiting a budget grant.
                            // Leave it `InProgress` (the consult path's invariant) so
                            // the resume's seed re-attaches via the SAME
                            // InProgress→Pending→complete machinery. Under Autonomous
                            // the run aborts, so the task is `Blocked` as before.
                            let _ = task_list.update(
                                task_id,
                                Some(if surface {
                                    TaskStatus::InProgress
                                } else {
                                    TaskStatus::Blocked
                                }),
                                None,
                            );
                            executor.persist_task_list(&session_id, &task_list).await;
                            let partial = plan_execute_partial_json(&task_list);
                            // #138 AC2-a: carry the FULL stalled worker session
                            // (the execute child left it in scratch) into the pause
                            // so a budget RESUME re-attaches it as the in-progress
                            // task's seed — parallel to the consult pause — instead
                            // of discarding it for a partial-only stub.
                            let worker_session = std::mem::take(&mut cx.scratch.run_session);
                            let worker_toolset = worker_toolset_of(&self.execute);
                            cx.pop_budget();
                            cx.scratch.task = Some(task.clone());
                            cx.stream = on_stream;
                            // Advance the run-wide turn cursor to the worker's
                            // consumed turns BEFORE building the pause (mirrors the
                            // Success branch's `carried.turns = turns`). Otherwise the
                            // cursor stays frozen at `carried_before`, so the paused
                            // `budget_used` / `turn_number` and the displayed
                            // `TurnStart { turn }` regress, and the resumed window
                            // re-seeds its floor to the same value and re-runs the
                            // same turns. `leaf_steps` is absolute (>= carried_before).
                            carried.turns = leaf_steps;
                            if surface {
                                let waiting = promote_budget_exhausted_to_human(
                                    &err,
                                    Some(partial),
                                    combinator_escalation_actions(&err),
                                    session_id.clone(),
                                    task,
                                    carried.clone(),
                                    carried.turns,
                                    worker_session,
                                    worker_toolset,
                                );
                                return cx.record_terminal(waiting);
                            }
                            return promote_budget_exhausted(&err, Some(partial));
                        }
                    }
                }

                let sub_result = cx.take_child_override();
                match sub_result {
                    Some(RunResult::Success {
                        output,
                        usage,
                        turns,
                        session_state: step_state,
                        ..
                    }) => {
                        carried.turns = turns;
                        last_state = step_state;
                        carried.input_tokens += usage.input_tokens;
                        carried.output_tokens += usage.output_tokens;
                        total_usage.input_tokens += usage.input_tokens;
                        total_usage.output_tokens += usage.output_tokens;
                        total_usage.cache_read_tokens += usage.cache_read_tokens;
                        total_usage.cache_write_tokens += usage.cache_write_tokens;
                        total_usage.cost_usd += usage.cost_usd;
                        last_output = output.clone();

                        let _ = task_list.complete(task_id);
                        executor.persist_task_list(&session_id, &task_list).await;

                        // #126: record this task's final output (Tier-1) and append
                        // a ledger entry whose files_touched is HARNESS-OBSERVED.
                        final_outputs.insert(task_id, output.clone());
                        let files_touched = executor.take_observed_writes();
                        let entry = StepLedgerEntry {
                            task_id,
                            summary: output.clone(),
                            files_touched,
                        };
                        if push_step_ledger(&mut ledger, entry) {
                            ledger_elided = true;
                        }

                        // #stream: emit through the WRAPPED `cx.stream` (not the
                        // saved parent `on_stream`) so this combinator-surfaced step
                        // output is attributed to the `plan_execute` node.
                        StandardHarness::emit(
                            &cx.stream,
                            StreamEvent::FinalResponse {
                                content: last_output.clone(),
                                node: None,
                            },
                        );

                        // #125: charge this step's turns against the PlanExecute scope.
                        if let Err(err) = cx.charge_current(turns.saturating_sub(carried_before)) {
                            let resolution = cx.resolve_current();
                            // #129: a granted `Continue` refreshes the scope and
                            // keeps scheduling the remaining ready tasks IN-PROCESS
                            // (this step already completed). NO serialization (AC3).
                            if matches!(resolution, ExhaustedResolution::Continue) {
                                continue;
                            }
                            // SC-5: AutoContinue auto-grants in-process at this
                            // Escalate point (scope still pushed) before surfacing.
                            if matches!(resolution, ExhaustedResolution::Escalate)
                                && cx.try_auto_continue(&executor.escalation_mode())
                            {
                                continue;
                            }
                            let partial = plan_execute_partial_json(&task_list);
                            cx.pop_budget();
                            cx.scratch.task = Some(task.clone());
                            cx.stream = on_stream;
                            return match resolution {
                                ExhaustedResolution::Fail => promote_budget_exhausted(&err, None),
                                ExhaustedResolution::Continue | ExhaustedResolution::Escalate => {
                                    if matches!(
                                        executor.escalation_mode(),
                                        EscalationMode::SurfaceToHuman
                                    ) {
                                        // #138 AC2-a: this step already COMPLETED;
                                        // carry its post-run session (`last_state`)
                                        // so a budget resume re-attaches the real
                                        // worker context, not a partial-only stub.
                                        let waiting = promote_budget_exhausted_to_human(
                                            &err,
                                            Some(partial),
                                            combinator_escalation_actions(&err),
                                            session_id.clone(),
                                            task,
                                            carried.clone(),
                                            carried.turns,
                                            last_state.clone(),
                                            worker_toolset_of(&self.execute),
                                        );
                                        cx.record_terminal(waiting)
                                    } else {
                                        promote_budget_exhausted(&err, Some(partial))
                                    }
                                }
                            };
                        }
                    }
                    // A GLOBAL turn-budget hard stop (#117 backstop) surfaces as a
                    // `BudgetExceeded` Failure from the leaf. That is a WHOLE-RUN
                    // hard stop, NOT a single-task terminal failure — it aborts the
                    // run verbatim (preserving the pre-#126 mid-execute budget
                    // behavior). It is distinct from a per-NODE `BudgetExhausted`
                    // resolving to `Fail`, which DOES cascade (AC4, handled above).
                    Some(RunResult::Failure {
                        reason: reason @ HaltReason::BudgetExceeded { .. },
                        usage,
                        turns,
                        ..
                    }) => {
                        total_usage.input_tokens += usage.input_tokens;
                        total_usage.output_tokens += usage.output_tokens;
                        total_usage.cache_read_tokens += usage.cache_read_tokens;
                        total_usage.cache_write_tokens += usage.cache_write_tokens;
                        total_usage.cost_usd += usage.cost_usd;
                        let _ = task_list.update(task_id, Some(TaskStatus::Blocked), None);
                        executor.persist_task_list(&session_id, &task_list).await;
                        cx.pop_budget();
                        cx.stream = on_stream;
                        let result = RunResult::Failure {
                            reason,
                            session_id,
                            usage: total_usage,
                            turns,
                            session_state: last_state,
                        };
                        return cx.finish(executor, task, result).await;
                    }
                    // #126 AC3: a terminal task FAILURE cascade-blocks its
                    // transitive dependents and KEEPS scheduling unrelated tasks
                    // (replaces the Q5 blanket abort).
                    Some(RunResult::Failure {
                        reason,
                        usage,
                        turns,
                        ..
                    }) => {
                        carried.turns = turns;
                        total_usage.input_tokens += usage.input_tokens;
                        total_usage.output_tokens += usage.output_tokens;
                        total_usage.cache_read_tokens += usage.cache_read_tokens;
                        total_usage.cache_write_tokens += usage.cache_write_tokens;
                        total_usage.cost_usd += usage.cost_usd;

                        let _ = task_list.update(task_id, Some(TaskStatus::Blocked), None);
                        for dep in task_list.transitive_dependents(task_id) {
                            let _ = task_list.update(dep, Some(TaskStatus::Blocked), None);
                            blocked_by_failure.insert(dep);
                        }
                        blocked_by_failure.insert(task_id);
                        executor.persist_task_list(&session_id, &task_list).await;
                        if first_failure.is_none() {
                            first_failure = Some((task_id, format!("{reason:?}")));
                        }
                        continue;
                    }
                    // A pause / consult / escalate propagates the whole run verbatim.
                    Some(pause) => {
                        cx.pop_budget();
                        cx.stream = on_stream;
                        return cx.finish(executor, task, pause).await;
                    }
                    None => {
                        // No terminal from the child: treat as a terminal failure
                        // of this task and cascade (same as a Failure).
                        let _ = task_list.update(task_id, Some(TaskStatus::Blocked), None);
                        for dep in task_list.transitive_dependents(task_id) {
                            let _ = task_list.update(dep, Some(TaskStatus::Blocked), None);
                            blocked_by_failure.insert(dep);
                        }
                        blocked_by_failure.insert(task_id);
                        executor.persist_task_list(&session_id, &task_list).await;
                        if first_failure.is_none() {
                            first_failure = Some((
                                task_id,
                                "execute sub-strategy produced no terminal".to_string(),
                            ));
                        }
                        continue;
                    }
                }
            }

            cx.pop_budget();
            cx.stream = on_stream;

            // ── Drain (#126, decision A). ───────────────────────────────────
            //
            // A run where a terminal failure cascade-blocked any task returns a
            // PARTIAL terminal `Failure` reporting the full partition. A run where
            // every task completed returns `Success` (output = last step's text).
            if let Some((failed_task, reason)) = first_failure {
                let mut completed: Vec<u32> = task_list
                    .tasks
                    .iter()
                    .filter(|t| t.status == TaskStatus::Completed)
                    .map(|t| t.id)
                    .collect();
                completed.sort_unstable();
                let mut blocked: Vec<u32> = blocked_by_failure.into_iter().collect();
                blocked.sort_unstable();
                let result = RunResult::Failure {
                    reason: HaltReason::TasksBlockedByFailure {
                        completed,
                        blocked,
                        failed_task,
                        reason,
                    },
                    session_id,
                    usage: total_usage,
                    turns: carried.turns,
                    session_state: last_state,
                };
                return cx.finish(executor, task, result).await;
            }

            let result = RunResult::Success {
                output: last_output,
                session_id,
                usage: total_usage,
                turns: carried.turns,
                session_state: last_state,
            };
            cx.finish(executor, task, result).await
        })
    }
}

impl RunStrategy for SelfVerifyingConfig {
    /// SelfVerifying (#124): GENUINELY recursive build↔evaluate loop. Each
    /// iteration dispatches `self.inner.run(cx)` for the build phase (a non-ReAct
    /// inner — e.g. PlanExecute — really runs its whole loop per iteration), then
    /// runs a fresh evaluate phase on the inner worker's resolved agent (Q1c) and
    /// consults the verifier resolved from `self.evaluator`'s key (Q1a). Passed ⇒
    /// Success; Failed ⇒ append the reason (Default-FAIL) and loop; exhausted ⇒
    /// `SelfVerifyExhausted`.
    fn run<'a>(&'a self, cx: &'a mut ExecutionContext<'_>) -> BoxFut<'a, StrategyOutcome> {
        Box::pin(async move {
            let executor = match cx.executor() {
                Ok(e) => e,
                Err(o) => return o,
            };
            let task = cx.current_task();
            let build_session_id = task.session_id.clone();
            // SC-BUG-1: a HITL resume re-drives the whole SelfVerifying strategy
            // with the stalled build (worker) conversation carried in the
            // phase-agnostic resume seed. Consume it as the FIRST build iteration's
            // session so the build phase CONTINUES the approved/answered worker turn
            // (which already carries the original instruction + the dispatched tool
            // result) instead of restarting from an empty top-level session — then
            // the evaluate phase + verifier run, reaching the looper's eval-frame
            // reviewer. On a fresh run the seed is `None` and this is the incoming
            // `run_session`, byte-identical to before. When SelfVerifying is NESTED
            // under a PlanExecute walk, that outer walk `.take()`s the seed BEFORE
            // recursing, so this sees `None` and the nested behavior is unchanged.
            let mut session_state = cx
                .scratch
                .resume_seed
                .take()
                .unwrap_or_else(|| std::mem::take(&mut cx.scratch.run_session));
            let mut carried = cx.scratch.run_budget.clone();
            // #stream: save the parent sink, install a `self_verifying`-attributing
            // wrapper so the build (worker) and evaluate phases stream through it;
            // restored to the parent on every exit (`cx.stream = on_stream;`).
            let on_stream = cx.stream.take();
            cx.stream = wrap_node_sink(on_stream.clone(), "self_verifying", None);

            // Q1a: resolve the verifier from `evaluator`'s key (NO wire change).
            let verifier = match cx.registry.resolve_verifier(&self.evaluator.0) {
                Some(v) => v.clone(),
                None => {
                    let result = RunResult::Failure {
                        reason: HaltReason::SelfVerifyMisconfigured {
                            reason: format!(
                                "SelfVerifying requires a verifier registered under key {:?}",
                                self.evaluator.0
                            ),
                        },
                        session_id: build_session_id,
                        usage: AggregateUsage::default(),
                        turns: 0,
                        session_state: SessionState::default(),
                    };
                    cx.stream = on_stream;
                    return cx.finish(executor, task, result).await;
                }
            };
            // Q1c: the evaluate-phase agent defaults to the inner worker's agent;
            // an explicit `eval_agent` override (a dedicated reviewer, distinct
            // from the builder) takes precedence.
            let eval_agent_ref = self
                .eval_agent
                .clone()
                .unwrap_or_else(|| AgentRef(executor_worker_key(self)));
            let eval_agent = match executor.resolve_agent_ref(&eval_agent_ref, &build_session_id) {
                Ok(a) => a,
                Err(result) => {
                    cx.stream = on_stream;
                    return cx.finish(executor, task, *result).await;
                }
            };
            // The evaluate phase runs against `eval_toolset` (a read-only/
            // inspection catalogue) when set; `None` ⇒ the empty handle (global-
            // catalogue fallback), byte-identical to pre-override behavior.
            let eval_toolset = self
                .eval_toolset
                .clone()
                .unwrap_or_else(|| ToolsetRef(String::new()));

            let max_iterations = verifier.max_iterations();
            let mut total_usage = AggregateUsage::default();
            let mut last_reason = String::new();
            let mut last_worker_output = String::new();

            // #125: SelfVerifying owns a budget scope for its build↔evaluate loop.
            // POLICY is the task's global turn ceiling (TotalSteps); behavior is
            // `Escalate` (in-process placeholder; serialized behavior is #129).
            let sv_policy = match task.budget.max_turns {
                Some(max) => BudgetPolicy::TotalSteps { value: max },
                None => BudgetPolicy::Unlimited,
            };
            cx.push_budget(sv_policy, self.behavior.clone(), "self_verifying");

            for iteration in 0..max_iterations {
                // ── Build phase: recurse `self.inner.run(cx)`.
                let build_task = Task {
                    id: task.id.clone(),
                    instruction: task.instruction.clone(),
                    session_id: build_session_id.clone(),
                    budget: task.budget.clone(),
                    loop_strategy: (*self.inner).clone(),
                };
                cx.scratch.task = Some(build_task);
                cx.scratch.run_session = session_state.clone();
                cx.scratch.run_budget = carried.clone();
                let carried_before = carried.turns;
                let build_outcome = self.inner.run(cx).await;
                // #125 rule 4/7: a child's BudgetExhausted reaches THIS parent as
                // a `StrategyOutcome`, never auto-cascaded. SelfVerifying surfaces
                // its own typed BudgetExhausted (partial = last worker result +
                // last verdict) without charging the child's exhaustion against
                // its own scope.
                if let StrategyOutcome::BudgetExhausted { .. } = &build_outcome {
                    let partial = self_verifying_partial_json(&last_worker_output, &last_reason);
                    let err = cx
                        .budgets
                        .current_mut()
                        .map(|s| BudgetExhausted {
                            policy: s.policy.clone(),
                            behavior: s.behavior.clone(),
                            steps_taken: s.steps_taken,
                            continues_used: s.continues_used,
                            phase: s.phase.clone(),
                        })
                        .expect("self_verifying scope pushed above");
                    cx.pop_budget();
                    cx.scratch.task = Some(task);
                    cx.stream = on_stream;
                    return promote_budget_exhausted(&err, Some(partial));
                }
                let build_result = match cx.take_child_override() {
                    Some(r) => r,
                    None => RunResult::Failure {
                        reason: HaltReason::SelfVerifyMisconfigured {
                            reason: "build sub-strategy produced no terminal".into(),
                        },
                        session_id: build_session_id.clone(),
                        usage: AggregateUsage::default(),
                        turns: carried.turns,
                        session_state: session_state.clone(),
                    },
                };
                fold_usage(&mut total_usage, &mut carried, &build_result);

                // A paused / escalated build propagates verbatim.
                match &build_result {
                    RunResult::WaitingForHuman { .. } | RunResult::Consult { .. } => {
                        cx.pop_budget();
                        cx.stream = on_stream;
                        return cx.finish(executor, task, build_result).await;
                    }
                    RunResult::Escalate { .. } => {
                        cx.pop_budget();
                        cx.stream = on_stream;
                        return cx.finish(executor, task, build_result).await;
                    }
                    _ => {}
                }
                // Capture the build's output for the partial (last worker result).
                if let RunResult::Success { output, .. } = &build_result {
                    last_worker_output = output.clone();
                }
                // Carry the build's post-run session forward for the next round.
                if let RunResult::Success {
                    session_state: st, ..
                }
                | RunResult::Failure {
                    session_state: st, ..
                } = &build_result
                {
                    session_state = st.clone();
                }

                // #125: charge this iteration's build turns against the
                // SelfVerifying scope. If the global cap is spent, the node
                // surfaces its OWN typed BudgetExhausted (partial = last worker
                // result + last verdict).
                if let Err(err) = cx.charge_current(carried.turns.saturating_sub(carried_before)) {
                    let partial = self_verifying_partial_json(&last_worker_output, &last_reason);
                    let resolution = cx.resolve_current();
                    // #129: a granted `Continue` resets the scope and RE-RUNS the
                    // build↔evaluate iteration IN-PROCESS (do NOT pop the scope; the
                    // loop continues under the refreshed allowance). NO serialization
                    // (AC3). `max_continues` bounds the loop.
                    if matches!(resolution, ExhaustedResolution::Continue) {
                        continue;
                    }
                    // SC-5: AutoContinue auto-grants in-process at this Escalate
                    // point (scope still pushed) before surfacing/aborting.
                    if matches!(resolution, ExhaustedResolution::Escalate)
                        && cx.try_auto_continue(&executor.escalation_mode())
                    {
                        continue;
                    }
                    cx.pop_budget();
                    cx.scratch.task = Some(task.clone());
                    cx.stream = on_stream;
                    return match resolution {
                        ExhaustedResolution::Fail => promote_budget_exhausted(&err, None),
                        // #129: the in-process `Continue` is handled above; this arm
                        // is `Escalate`-only now (kept combined for the surface/
                        // propagate shape).
                        ExhaustedResolution::Continue | ExhaustedResolution::Escalate => {
                            if matches!(executor.escalation_mode(), EscalationMode::SurfaceToHuman)
                            {
                                // #138 AC2-a: carry the FULL build (worker) session
                                // — the SelfVerifying loop carries it forward in
                                // `session_state` after each iteration — so a budget
                                // resume re-attaches the real worker context.
                                let waiting = promote_budget_exhausted_to_human(
                                    &err,
                                    Some(partial),
                                    combinator_escalation_actions(&err),
                                    build_session_id.clone(),
                                    task,
                                    carried.clone(),
                                    carried.turns,
                                    session_state.clone(),
                                    worker_toolset_of(&self.inner),
                                );
                                cx.record_terminal(waiting)
                            } else {
                                promote_budget_exhausted(&err, Some(partial))
                            }
                        }
                    };
                }

                // ── Evaluate phase: a fresh evaluator run on `eval_agent`.
                let eval_result = executor
                    .evaluate_phase(
                        &task,
                        eval_agent.clone(),
                        eval_toolset.clone(),
                        &mut carried,
                        &mut total_usage,
                    )
                    .await;

                // #C1: charge the evaluator's OWN turns against the SelfVerifying
                // scope, mirroring the build charge above. `run_evaluate_phase`
                // runs the evaluator in an isolated harness from a fresh
                // `BudgetSnapshot`, so `eval_result.turns` is the evaluator's
                // standalone turn count — and `fold_usage` folds it via `.max()`
                // (NOT a sum), so it never reaches `carried.turns` once build
                // turns dominate. Charging the build delta alone therefore missed
                // the evaluate phase entirely; charge it explicitly here so both
                // phases count against the budget.
                let eval_turns = match &eval_result {
                    RunResult::Success { turns, .. }
                    | RunResult::Failure { turns, .. }
                    | RunResult::Escalate { turns, .. } => *turns,
                    RunResult::WaitingForHuman { .. } | RunResult::Consult { .. } => 0,
                };
                if let Err(err) = cx.charge_current(eval_turns) {
                    let partial = self_verifying_partial_json(&last_worker_output, &last_reason);
                    let resolution = cx.resolve_current();
                    // A granted `Continue` resets the scope and re-runs the
                    // build↔evaluate iteration in-process (mirrors the build path).
                    if matches!(resolution, ExhaustedResolution::Continue) {
                        continue;
                    }
                    // SC-5: AutoContinue auto-grants in-process at this Escalate
                    // point (scope still pushed) before surfacing/aborting.
                    if matches!(resolution, ExhaustedResolution::Escalate)
                        && cx.try_auto_continue(&executor.escalation_mode())
                    {
                        continue;
                    }
                    cx.pop_budget();
                    cx.scratch.task = Some(task.clone());
                    cx.stream = on_stream;
                    return match resolution {
                        ExhaustedResolution::Fail => promote_budget_exhausted(&err, None),
                        ExhaustedResolution::Continue | ExhaustedResolution::Escalate => {
                            if matches!(executor.escalation_mode(), EscalationMode::SurfaceToHuman)
                            {
                                let waiting = promote_budget_exhausted_to_human(
                                    &err,
                                    Some(partial),
                                    combinator_escalation_actions(&err),
                                    build_session_id.clone(),
                                    task,
                                    carried.clone(),
                                    carried.turns,
                                    session_state.clone(),
                                    worker_toolset_of(&self.inner),
                                );
                                cx.record_terminal(waiting)
                            } else {
                                promote_budget_exhausted(&err, Some(partial))
                            }
                        }
                    };
                }

                let input = VerifierInput {
                    build_result,
                    eval_result,
                    workspace: self.sandbox_root(executor),
                    iteration,
                };
                match verifier.verify(&input).await {
                    VerifierVerdict::Passed => {
                        let (output, turns, final_state) = match input.build_result {
                            RunResult::Success {
                                output,
                                turns,
                                session_state: st,
                                ..
                            } => (output, turns, st),
                            _ => (String::new(), carried.turns, session_state.clone()),
                        };
                        let result = RunResult::Success {
                            output,
                            session_id: build_session_id,
                            usage: total_usage,
                            turns,
                            session_state: final_state,
                        };
                        cx.pop_budget();
                        cx.stream = on_stream;
                        return cx.finish(executor, task, result).await;
                    }
                    VerifierVerdict::Failed { reason } => {
                        last_reason = reason.clone();
                        executor
                            .append_user_message(&mut session_state, &reason)
                            .await;
                    }
                }
            }

            cx.pop_budget();
            let result = RunResult::Failure {
                reason: HaltReason::SelfVerifyExhausted {
                    iterations: max_iterations,
                    last_reason,
                },
                session_id: build_session_id,
                usage: total_usage,
                turns: carried.turns,
                session_state,
            };
            cx.stream = on_stream;
            cx.finish(executor, task, result).await
        })
    }
}

impl SelfVerifyingConfig {
    /// The workspace root for a verifier input. Delegated to the executor so the
    /// config body stays harness-agnostic.
    fn sandbox_root(&self, executor: &dyn StrategyExecutor) -> std::path::PathBuf {
        executor.workspace_root()
    }
}

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

impl RunStrategy for RalphConfig {
    /// Ralph (#124): GENUINELY recursive continuation wrapper. Each context
    /// window seeds a FRESH session from the `.spore/` checkpoint, then recurses
    /// `self.inner.run(cx)` (a non-ReAct inner — e.g. SelfVerifying — really runs
    /// its whole loop per window). Ralph is an OUTER LOOP that re-runs `inner` as
    /// the architect declared it — it does NOT replace nodes the inner tree
    /// already assigned. When `self.agent` is set it only FILLS a worker leaf that
    /// left its own agent handle EMPTY (the bare-leaf Ralph convenience); an
    /// explicit leaf agent is authoritative and never overwritten.
    /// `ralph_completion_status` drives the OUTER reset loop; exhaustion ⇒
    /// `RalphCompletionUnmet`.
    fn run<'a>(&'a self, cx: &'a mut ExecutionContext<'_>) -> BoxFut<'a, StrategyOutcome> {
        Box::pin(async move {
            let executor = match cx.executor() {
                Ok(e) => e,
                Err(o) => return o,
            };
            let task = cx.current_task();
            // #stream: save the parent sink, install a `ralph`-attributing wrapper
            // so each reset window's turns stream through it; restored on exit.
            let on_stream = cx.stream.take();
            cx.stream = wrap_node_sink(on_stream.clone(), "ralph", None);
            // Ralph discards the incoming session state by design (each window is
            // a fresh start re-seeded from the filesystem checkpoint).
            let _ = std::mem::take(&mut cx.scratch.run_session);
            let max_resets = self.max_resets(executor);

            // Ralph fills — never replaces. When `self.agent` is set it supplies a
            // worker agent ONLY where the inner leaf left its handle empty; an
            // explicitly-declared leaf agent (the architect's node) is authoritative.
            let inner_for_window: LoopStrategy = if self.agent.0.is_empty() {
                (*self.inner).clone()
            } else {
                let mut cloned = (*self.inner).clone();
                fill_empty_worker_agent(&mut cloned, &self.agent);
                cloned
            };

            let mut total_usage = AggregateUsage::default();
            let mut cumulative_turns: u32 = 0;
            let mut last_reason = String::from(".spore/progress.json missing");
            let mut last_session_id = task.session_id.clone();

            for iteration in 0..max_resets.max(1) {
                let window_session_id = if iteration == 0 {
                    task.session_id.clone()
                } else {
                    SessionId::generate()
                };
                last_session_id = window_session_id.clone();

                // R2/R3: a FRESH session seeded from the `.spore/` checkpoint.
                let session_state = executor.ralph_seed_session(&task.instruction).await;

                let window_task = Task {
                    id: task.id.clone(),
                    instruction: task.instruction.clone(),
                    session_id: window_session_id.clone(),
                    budget: task.budget.clone(),
                    loop_strategy: inner_for_window.clone(),
                };
                cx.scratch.task = Some(window_task);
                cx.scratch.run_session = session_state;
                // FRESH per-window budget (the reset discards the turn budget).
                cx.scratch.run_budget = BudgetSnapshot::default();
                // Model (A): Ralph is the continuation loop — each window continues
                // the ONE logical task across the per-window session reset. Signal
                // it so a PlanExecute inner resumes its durable task list (window 0
                // / fresh-start has an empty list ⇒ the inner's "unfinished work"
                // check still plans fresh; a window that finished its whole plan but
                // is not yet "done" likewise has no unfinished work ⇒ re-plans for
                // the remaining work instead of no-op draining).
                cx.scratch.continue_task_list = true;
                let window_outcome = inner_for_window.run(cx).await;
                // #125 rule 4/7: a window child's BudgetExhausted reaches Ralph as
                // a `StrategyOutcome`, never auto-cascaded. Ralph's recovery
                // semantics: a budget-exhausted window is treated as "window
                // incomplete" — RESET the context window and retry (next outer
                // iteration). After `max_resets` this falls through to
                // `RalphCompletionUnmet`. Ralph's own scope is unaffected.
                if let StrategyOutcome::BudgetExhausted { partial_output, .. } = &window_outcome {
                    last_reason = format!(
                        "window {} budget-exhausted: {}",
                        iteration + 1,
                        partial_output.as_deref().unwrap_or("<no partial>")
                    );
                    continue;
                }
                let window_result = match cx.take_child_override() {
                    Some(r) => r,
                    None => RunResult::Failure {
                        reason: HaltReason::RalphCompletionUnmet {
                            iterations: iteration + 1,
                            last_reason: "window sub-strategy produced no terminal".into(),
                        },
                        session_id: window_session_id.clone(),
                        usage: AggregateUsage::default(),
                        turns: 0,
                        session_state: SessionState::default(),
                    },
                };
                let mut window_budget = BudgetSnapshot::default();
                fold_usage(&mut total_usage, &mut window_budget, &window_result);
                cumulative_turns += window_budget.turns;

                // A paused / escalated window propagates verbatim.
                if window_result.is_pause() {
                    cx.stream = on_stream;
                    return cx.finish(executor, task, window_result).await;
                }

                match executor.ralph_completion_status().await {
                    None => {
                        let (output, final_state) = match window_result {
                            RunResult::Success {
                                output,
                                session_state,
                                ..
                            } => (output, session_state),
                            _ => (String::new(), SessionState::default()),
                        };
                        let result = RunResult::Success {
                            output,
                            session_id: window_session_id,
                            usage: total_usage,
                            turns: cumulative_turns,
                            session_state: final_state,
                        };
                        cx.stream = on_stream;
                        return cx.finish(executor, task, result).await;
                    }
                    Some(reason) => {
                        last_reason = reason;
                    }
                }
            }

            let result = RunResult::Failure {
                reason: HaltReason::RalphCompletionUnmet {
                    iterations: max_resets.max(1),
                    last_reason,
                },
                session_id: last_session_id,
                usage: total_usage,
                turns: cumulative_turns,
                session_state: SessionState::default(),
            };
            cx.stream = on_stream;
            cx.finish(executor, task, result).await
        })
    }
}

impl RalphConfig {
    /// The configured `max_resets` (the harness `config.max_resets`).
    fn max_resets(&self, executor: &dyn StrategyExecutor) -> u32 {
        executor.ralph_max_resets()
    }
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

impl RunStrategy for HillClimbingConfig {
    /// HillClimbing (#124): GENUINELY recursive optimization loop. Iteration 0 is
    /// a pure baseline (no agent turn). Iterations 1.. recurse `self.inner.run(cx)`
    /// to propose a change (a non-ReAct inner — e.g. PlanExecute — really runs its
    /// whole loop per iteration), then evaluate the metric (resolved via
    /// `resolve_metric_evaluator`, Q2) and keep/revert. Bounded by
    /// `max_stagnation` and the turn budget.
    fn run<'a>(&'a self, cx: &'a mut ExecutionContext<'_>) -> BoxFut<'a, StrategyOutcome> {
        Box::pin(async move {
            let executor = match cx.executor() {
                Ok(e) => e,
                Err(o) => return o,
            };
            let task = cx.current_task();
            let session_id = task.session_id.clone();
            let task_id = task.id.clone();
            // #stream: save the parent sink, install a `hill_climbing`-attributing
            // wrapper so each iteration's turns stream through it; restored on exit.
            let on_stream = cx.stream.take();
            cx.stream = wrap_node_sink(on_stream.clone(), "hill_climbing", None);
            let mut carried = cx.scratch.run_budget.clone();
            let _ = std::mem::take(&mut cx.scratch.run_session);
            let direction = self.direction;
            let revert = self.revert_on_no_improvement;
            let min_delta = Some(self.min_improvement_delta);
            // `u32::MAX` sentinel ⇒ no stagnation cap.
            let max_stagnation = (self.max_stagnation != u32::MAX).then_some(self.max_stagnation);

            // Q2: resolve the metric evaluator from `evaluator`'s key.
            let evaluator = match executor.resolve_metric_evaluator(&self.evaluator.0, &session_id)
            {
                Ok(e) => e,
                Err(result) => {
                    cx.stream = on_stream;
                    return cx.finish(executor, task, *result).await;
                }
            };

            let mut total_usage = AggregateUsage::default();
            let mut rows: Vec<crate::metric::ResultsEntry> = Vec::new();
            let mut span_seq: u64 = 0;

            // #125: HillClimbing owns a budget scope for its optimization loop.
            // POLICY is the task's global turn ceiling (TotalSteps); this REPLACES
            // the ad-hoc `turn_cap` / `carried.turns >= turn_cap` gate that #124
            // used. Behavior is `Escalate` (in-process placeholder; #129).
            let hc_policy = match task.budget.max_turns {
                Some(max) => BudgetPolicy::TotalSteps { value: max },
                None => BudgetPolicy::Unlimited,
            };
            cx.push_budget(hc_policy, self.behavior.clone(), "hill_climbing");

            // ── Iteration 0: pure baseline (no agent turn). D4: also commits the
            // starting tree and returns its SHA, seeding the durable candidate
            // pointers (hill-climbing-loop-correctness §4.1).
            let (baseline_score, start_sha) = match executor
                .hill_baseline(
                    &evaluator,
                    &session_id,
                    &task_id,
                    direction,
                    &mut rows,
                    &mut span_seq,
                    total_usage.clone(),
                    carried.turns,
                )
                .await
            {
                Ok(v) => v,
                Err(result) => {
                    cx.pop_budget();
                    cx.stream = on_stream;
                    return cx.finish(executor, task, result).await;
                }
            };
            // `current_*` is where the next neighbor branches from; `best_*` is
            // the best artifact ever accepted. Equal under strict greedy, tracked
            // separately from day one (D2).
            let mut cand = HillCandidate {
                current_sha: start_sha.clone(),
                best_sha: start_sha,
                current_best: baseline_score,
                best_score: baseline_score,
            };

            let mut stagnation: u32 = 0;
            let mut iteration: u32 = 1;

            loop {
                // #125: charge-based budget gate before the iteration's agent
                // turn. A spent `TotalSteps` cap surfaces this node's OWN typed
                // BudgetExhausted (partial = best candidate + score), resolving its
                // behavior — replacing the legacy `BudgetExceeded` Failure.
                if let Err(err) = cx.charge_current(1) {
                    let resolution = cx.resolve_current();
                    // #129: a granted `Continue` resets the scope and KEEPS ITERATING
                    // the climb IN-PROCESS (do NOT pop; the refreshed allowance lets
                    // the next charge pass). NO serialization (AC3). `max_continues`
                    // bounds the loop.
                    if matches!(resolution, ExhaustedResolution::Continue) {
                        continue;
                    }
                    // SC-5: AutoContinue auto-grants in-process at this Escalate
                    // point (scope still pushed) before surfacing/aborting.
                    if matches!(resolution, ExhaustedResolution::Escalate)
                        && cx.try_auto_continue(&executor.escalation_mode())
                    {
                        continue;
                    }
                    executor.hill_write_tsv(&task_id, &rows).await;
                    let partial =
                        hill_climbing_partial_json(cand.best_score, cand.best_sha.as_deref());
                    cx.pop_budget();
                    cx.scratch.task = Some(task.clone());
                    cx.stream = on_stream;
                    return match resolution {
                        ExhaustedResolution::Fail => promote_budget_exhausted(&err, None),
                        // #129: the in-process `Continue` is handled above; this arm
                        // is `Escalate`-only now (kept combined for the surface/
                        // propagate shape).
                        //
                        // #130: under `SurfaceToHuman`, PAUSE with a
                        // `BudgetExhausted` request instead of propagating up.
                        ExhaustedResolution::Continue | ExhaustedResolution::Escalate => {
                            if matches!(executor.escalation_mode(), EscalationMode::SurfaceToHuman)
                            {
                                // #138 AC2-a: HillClimbing iterates on a METRIC, not
                                // a worker conversation, so there is no richer
                                // session to carry — pass the empty default to fall
                                // back to the partial-only stub (pre-#138 behavior).
                                // The worker leaf's toolset handle is still carried
                                // (#140 parity, AC4-a).
                                let waiting = promote_budget_exhausted_to_human(
                                    &err,
                                    Some(partial),
                                    combinator_escalation_actions(&err),
                                    session_id.clone(),
                                    task,
                                    carried.clone(),
                                    carried.turns,
                                    SessionState::default(),
                                    worker_toolset_of(&self.inner),
                                );
                                cx.record_terminal(waiting)
                            } else {
                                promote_budget_exhausted(&err, Some(partial))
                            }
                        }
                    };
                }
                if let Some(limit_type) =
                    StandardHarness::budget_exceeded(&task.budget, &carried, Instant::now())
                {
                    executor.hill_write_tsv(&task_id, &rows).await;
                    // Winner application (§4.6): a wall/cost/token budget halt is a
                    // terminal exit — leave the best accepted candidate on disk.
                    executor.hill_apply_winner(cand.best_sha.as_deref()).await;
                    let result = RunResult::Failure {
                        reason: HaltReason::BudgetExceeded { limit_type },
                        session_id,
                        usage: total_usage,
                        turns: carried.turns,
                        session_state: SessionState::default(),
                    };
                    cx.pop_budget();
                    cx.stream = on_stream;
                    return cx.finish(executor, task, result).await;
                }

                // ── One agent turn proposes a change: recurse `self.inner`.
                let iter_task = Task {
                    id: task.id.clone(),
                    instruction: task.instruction.clone(),
                    session_id: session_id.clone(),
                    budget: task.budget.clone(),
                    loop_strategy: (*self.inner).clone(),
                };
                cx.scratch.task = Some(iter_task);
                let mut iter_state = SessionState::default();
                executor
                    .append_user_message(&mut iter_state, &task.instruction)
                    .await;
                cx.scratch.run_session = iter_state;
                cx.scratch.run_budget = carried.clone();
                // Model (A): each HillClimbing iteration generates a FRESH neighbor
                // candidate — never a continuation of the prior iteration's plan.
                // Clear the continue signal so a PlanExecute inner re-plans each
                // iteration (the load-bearing requirement of hill climbing), and so
                // an enclosing Ralph window's continue signal does not leak in.
                cx.scratch.continue_task_list = false;
                let iter_outcome = self.inner.run(cx).await;
                // #125 rule 4/7: a child's BudgetExhausted reaches HillClimbing as
                // a `StrategyOutcome`, never auto-cascaded. Surface this node's own
                // typed BudgetExhausted (partial = best candidate + score).
                if let StrategyOutcome::BudgetExhausted { .. } = &iter_outcome {
                    executor.hill_write_tsv(&task_id, &rows).await;
                    let partial =
                        hill_climbing_partial_json(cand.best_score, cand.best_sha.as_deref());
                    let err = cx
                        .budgets
                        .current_mut()
                        .map(|s| BudgetExhausted {
                            policy: s.policy.clone(),
                            behavior: s.behavior.clone(),
                            steps_taken: s.steps_taken,
                            continues_used: s.continues_used,
                            phase: s.phase.clone(),
                        })
                        .expect("hill_climbing scope pushed above");
                    cx.pop_budget();
                    cx.scratch.task = Some(task);
                    cx.stream = on_stream;
                    return promote_budget_exhausted(&err, Some(partial));
                }
                let turn_result = match cx.take_child_override() {
                    Some(r) => r,
                    None => RunResult::Failure {
                        reason: HaltReason::BudgetExceeded {
                            limit_type: BudgetLimitType::Turns,
                        },
                        session_id: session_id.clone(),
                        usage: AggregateUsage::default(),
                        turns: carried.turns,
                        session_state: SessionState::default(),
                    },
                };
                fold_usage(&mut total_usage, &mut carried, &turn_result);

                // A paused / escalated turn propagates verbatim.
                if turn_result.is_pause() {
                    executor.hill_write_tsv(&task_id, &rows).await;
                    cx.pop_budget();
                    cx.stream = on_stream;
                    return cx.finish(executor, task, turn_result).await;
                }

                // ── Evaluate the metric + keep/revert decision. `hill_iteration`
                // mutates the durable candidate pointers in `cand` (commit-on-keep,
                // reset-to-current_sha-on-discard) and returns the non-improvement
                // flag.
                let non_improvement = executor
                    .hill_iteration(
                        &evaluator,
                        &session_id,
                        &task_id,
                        iteration,
                        direction,
                        revert,
                        min_delta,
                        &mut cand,
                        &mut rows,
                        &mut span_seq,
                    )
                    .await;
                if non_improvement {
                    stagnation += 1;
                } else {
                    stagnation = 0;
                }

                if let Some(limit) = max_stagnation {
                    if stagnation >= limit {
                        executor.hill_write_tsv(&task_id, &rows).await;
                        // Winner application (§4.6): stagnation is the canonical
                        // HillClimbing termination — leave the best accepted
                        // candidate on disk before returning.
                        executor.hill_apply_winner(cand.best_sha.as_deref()).await;
                        let result = RunResult::Failure {
                            reason: HaltReason::StagnationLimitReached {
                                iterations: stagnation,
                                best_metric: cand.best_score,
                            },
                            session_id,
                            usage: total_usage,
                            turns: carried.turns,
                            session_state: SessionState::default(),
                        };
                        cx.pop_budget();
                        cx.stream = on_stream;
                        return cx.finish(executor, task, result).await;
                    }
                }

                iteration = iteration.saturating_add(1);
            }
        })
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
    fn record_terminal(&mut self, result: RunResult) -> StrategyOutcome {
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
    async fn finish(
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

// ============================================================================
// Per-node budget enforcement + failure isolation (issue #125)
// ============================================================================
//
// This slice makes `BudgetContext::charge` the REAL per-node enforcement point
// and `StrategyOutcome::BudgetExhausted` a real, isolated, parent-inspectable
// value. #123 built `BudgetContext`/`BudgetStack`/`charge` as pure-arithmetic
// scaffold (dead outside tests); #124 wired enforcement through the LEGACY path
// (`task.budget.max_turns` + `BudgetSnapshot`) and mapped every `BudgetExceeded`
// terminal to `StrategyOutcome::Failed`. #125 replaces that with charge-based
// enforcement at each `*Config::run` and a typed, non-cascading exhaustion.
//
// New methods (this slice):
//   - `BudgetContext::consume_continue` — in-process continue: `continues_used +=
//     1`, `steps_taken = 0` (session/messages untouched).
//   - `BudgetContext::resolve_exhausted` -> `ExhaustedResolution` — walks the
//     behavior chain (Fail→Fail; Escalate→Escalate; Continue→grant-or-fall-through).
//   - `ExecutionContext::push_budget` / `pop_budget` / `charge_current` /
//     `resolve_current` — the per-scope stack ops. There is NO single
//     `charge_step` / `run_with_budget` / `execute_budget` wrapper and NO central
//     Continue-loop: enforcement is INLINED per `*Config::run` body. Each body
//     pushes its own scope (`push_budget`), runs the attempt (recurse / window),
//     charges the turns consumed (`charge_current`), and on exhaustion resolves +
//     promotes to a `StrategyOutcome::BudgetExhausted` before popping. The bodies
//     push an `Escalate` placeholder behavior (the serialized
//     `BudgetExhaustedBehavior` field is a wire change DEFERRED to #129), so a live
//     `resolve_current` returns `Escalate`/`Fail` today — never `Continue` — and
//     the partial follows fork #1 (`Escalate` → `Some(partial)`; `Fail` → `None`).
//     The in-process `Continue`-loop wiring (reset + re-run on a granted Continue)
//     and the serialized behavior field that would make it reachable in a live run
//     are DEFERRED to #129; the Continue chain is exercised today only by the
//     direct `BudgetContext`/`resolve_exhausted` unit tests.
//   - The per-node partial helpers (`react_partial_json`,
//     `plan_execute_partial_json`, `self_verifying_partial_json`,
//     `hill_climbing_partial_json`) build the node-concrete `partial_output` JSON.
//   - `promote_budget_exhausted` — the `BudgetExhausted -> StrategyOutcome`
//     promotion boundary helper that attaches `partial_output`.
//
// Enforcement rules (ALL implemented + tested):
//   1. A node capped at N stops at N WITHOUT killing siblings (each sibling gets
//      a FRESH `BudgetContext`; `steps_taken` starts at 0).
//   2. In-process `Continue` resets the counter and resumes with context intact,
//      honors `max_continues`, then falls through to `on_exhausted`.
//   3. `Fail` yields `partial_output = None` (discarded by contract).
//   4. A child `BudgetExhausted` reaches the parent as a `StrategyOutcome`,
//      NEVER auto-propagated (the parent's OWN `BudgetContext` is unaffected).
//   5. `partial_output` is concrete per node: ReAct → last FinalResponse text;
//      PlanExecute → JSON of task list + per-task statuses + ledger; SelfVerifying
//      → JSON of last worker result + verdict; HillClimbing → JSON of best
//      candidate + score.
//   6. The ReAct LEAF does NOT carry its own behavior — it PROPAGATES exhaustion
//      to its parent (one recovery site per node; Open Q A-2).
//   7. Never auto-cascade a child exhaustion into a parent exhaustion.
//
// Resolved spec forks (DECIDED by the maintainer — do NOT re-litigate):
//   - `Escalate` carries `partial_output = Some(partial)`; `Fail` carries `None`.
//   - `partial_output: Option<String>` is a JSON-serialized string of the
//     structured per-node partial (shapes per rule 5 above).
//   - `continues_used` persistence stays DEFERRED to #129 — in-process Continue
//     ONLY, no serialization, no new fixtures.
//
// No `// SPEC QUESTION:` markers — all three forks are resolved.

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
