//! PlanExecute strategy: the `StandardHarness` task-list / plan-artifact /
//! deep-resume helpers plus `impl RunStrategy for PlanExecuteConfig` (the
//! plan→execute DAG walk).

use super::*;

/// Build the `<files>` handoff from a plan phase's conversation: every file the
/// planner READ, deduped by path (last read wins — most recent contents), so the
/// execute steps can use them WITHOUT re-reading (plan→execute context threading).
/// Returns an empty string when the plan read nothing. The combined body is
/// bounded so a large file cannot blow the execute context — the overflowing file
/// is truncated at a char boundary and any further files are noted as omitted.
pub(crate) fn render_plan_read_files(session: &SessionState) -> String {
    use crate::model::{Content, Role};

    // Calls and results are both stored structurally, and a result names the call
    // it answers (`ToolResult::tool_use_id`). Join on that id. This used to pair
    // the Nth call with the Nth result by POSITION, because results were flat
    // `Content::Text` with no back-link — unsound the moment a turn issues
    // parallel calls or a middleware rewrites one.
    let mut read_path_by_call: std::collections::HashMap<&str, &str> =
        std::collections::HashMap::new();
    for m in &session.messages {
        if let Content::ToolCall(tc) = &m.content {
            if tc.name == "read_file" {
                if let Some(path) = tc.input.get("path").and_then(|v| v.as_str()) {
                    read_path_by_call.insert(tc.id.as_str(), path);
                }
            }
        }
    }
    if read_path_by_call.is_empty() {
        return String::new();
    }

    // path -> most-recent read content, preserving first-seen order. A failed
    // read contributes nothing: handing the planner an error body as if it were
    // file contents is worse than omitting the file.
    let mut order: Vec<&str> = Vec::new();
    let mut by_path: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for m in &session.messages {
        if !matches!(m.role, Role::Tool) {
            continue;
        }
        let Content::ToolResult(tr) = &m.content else {
            continue;
        };
        if tr.is_error {
            continue;
        }
        let Some(&path) = read_path_by_call.get(tr.tool_use_id.as_str()) else {
            continue;
        };
        if !by_path.contains_key(path) {
            order.push(path);
        }
        by_path.insert(path, tr.content.as_str());
    }
    if order.is_empty() {
        return String::new();
    }

    const MAX_TOTAL: usize = 24_000;
    let mut out = String::from(
        "<files>\nThese files were already read while planning (contents as of \
         planning). Use them instead of re-reading — only re-read a file you have \
         since modified.\n",
    );
    let mut used = 0usize;
    for path in order {
        let content = by_path[path];
        let remaining = MAX_TOTAL.saturating_sub(used);
        if remaining == 0 {
            out.push_str(&format!(
                "<file path=\"{path}\">(omitted — plan-handoff size limit)</file>\n"
            ));
            continue;
        }
        if content.len() <= remaining {
            used += content.len();
            out.push_str(&format!("<file path=\"{path}\">\n{content}\n</file>\n"));
        } else {
            // Truncate at a char boundary <= remaining.
            let mut cut = remaining;
            while cut > 0 && !content.is_char_boundary(cut) {
                cut -= 1;
            }
            used = MAX_TOTAL;
            out.push_str(&format!(
                "<file path=\"{path}\">\n{}\n… (truncated — plan-handoff size limit)\n</file>\n",
                &content[..cut]
            ));
        }
    }
    out.push_str("</files>");
    out
}

impl StandardHarness {
    /// Persist the parsed [`TaskList`](crate::tasklist::TaskList) for the run
    /// (Q4). The DURABLE write goes through the
    /// [`RunStore`](crate::storage::RunStore) seam under
    /// [`TASK_LIST_EXTRAS_KEY`](crate::tasklist::TASK_LIST_EXTRAS_KEY); the #71
    /// sandbox-filesystem path (`.spore/task_list.json`) is intentionally NOT
    /// used — one source of truth. The RunStore write is the single source of
    /// truth (#76 removed the redundant `SessionState.extras` mirror).
    /// Serialization failures are swallowed: a successful plan must not be lost
    /// to a storage hiccup (the default no-op/in-memory provider never fails).
    pub(crate) async fn persist_task_list(
        &self,
        session_id: &SessionId,
        task_list: &crate::tasklist::TaskList,
    ) {
        // #142: the task_list is DURABLE — key it by the STABLE project namespace
        // (`project_id.namespace()`), NOT the per-window `session_id` the Ralph
        // wrapper regenerates each context window. Namespace-reuse keeps the
        // RunStore trait keyed by `&SessionId` while the value keyed is the
        // stable project id. The incoming `session_id` remains the EPHEMERAL key
        // and is intentionally unused for this durable write.
        let _ = session_id;
        let durable_ns = self.config.project_id.namespace();
        if let Ok(value) = serde_json::to_value(task_list) {
            // Durable write through the storage seam (RunStore).
            let _ = self
                .config
                .storage
                .run()
                .put(&durable_ns, crate::tasklist::TASK_LIST_EXTRAS_KEY, value)
                .await;
        }
    }

    /// The planning directive seeded before the constrained plan turn (R1): the
    /// "record your plan via the `set_plan` tool" instruction wrapped around the
    /// task — the canonical authoring path (#126 decision C). Seeded by the
    /// recursive [`PlanExecuteConfig::run`] plan dispatch before `self.plan.run(cx)`
    /// (#124). The retry nudge still asks for JSON message content because the
    /// re-ask runs with tools SUPPRESSED (set_plan is then unavailable).
    pub(crate) fn plan_directive(instruction: &str) -> String {
        format!(
            "Produce a step-by-step plan for the following task, then RECORD it by \
             calling the `set_plan` tool with `tasks` = the ordered list of concrete \
             step strings (plus an optional one-line `rationale`). Record the plan \
             ONLY through `set_plan` — do NOT write it as a JSON or text message.\n\n\
             Task:\n{instruction}"
        )
    }

    /// Capture + persist a [`PlanArtifact`] from the plan turn's output text
    /// (#124): R3 (parse), R11 (fire `OnPlanCreated`, mutable), R4 (persist to the
    /// RunStore under `PLAN_EXECUTE_EXTRAS_KEY`). The model turn that produced
    /// `plan_output` ran elsewhere — the recursive `self.plan` child via
    /// `self.plan.run(cx)` — so this carries no agent call. Returns the captured
    /// artifact + accounting or a terminal failure to propagate.
    ///
    /// # SC-28 — a free-text / markdown plan is NOT a hard failure
    /// The strict canonical grammar (+ prose repair) runs first, so a JSON plan
    /// captures exactly as before — `tasks`/`rationale` come straight from the
    /// object. When BOTH fail the planner emitted prose rather than the JSON
    /// object; rather than aborting the whole run we capture it as a free-text
    /// artifact: the verbatim prose is preserved in `rationale`, and the runnable
    /// `tasks` are sourced from the durable `task_list` tool store — the ONE
    /// authoring path (#126 decision C) that the execute phase already prefers
    /// over the artifact, so JSON was never the only source of executable steps.
    /// Mirroring it here keeps the `OnPlanCreated` payload's `tasks` populated for
    /// panel consumers (looper `plan_tracker`, cordyceps `plan_announcer`). A
    /// prose plan that authored no `task_list` yields empty `tasks` — the execute
    /// phase then halts with `EmptyPlan`, exactly as a JSON `{"tasks": []}` would.
    /// The pure [`capture_plan_artifact`](crate::plan::capture_plan_artifact)
    /// grammar stays strict and byte-identical across languages; only this harness
    /// driver is tolerant.
    pub(crate) async fn capture_and_persist_plan(
        &self,
        session_id: &SessionId,
        plan_output: &str,
        usage: AggregateUsage,
        turns: u32,
    ) -> Result<PlanPhaseOutcome, RunResult> {
        // R3: capture the artifact from the response text. Uses the prose-repair
        // fallback: a planner that wraps its plan JSON in prose still captures
        // (the strict canonical grammar runs first; repair only rescues a failure).
        let mut artifact = match crate::plan::capture_plan_artifact_with_repair(plan_output) {
            Ok(a) => a,
            // SC-28: not JSON → capture as free-text. `tasks` from the task_list
            // tool store (load_task_list reads the durable, project-scoped list
            // the plan leaf authored via the tool); prose preserved verbatim.
            Err(_strict_err) => {
                let tasks = self
                    .load_task_list(session_id)
                    .await
                    .map(|list| {
                        list.tasks
                            .iter()
                            .map(|t| t.description.clone())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                crate::plan::PlanArtifact {
                    tasks,
                    rationale: plan_output.to_string(),
                }
            }
        };

        // R11: fire OnPlanCreated synchronously; the hook may rewrite `artifact`
        // in place. The stored artifact reflects any mutation.
        if let Some(chain) = self.config.hooks.as_ref() {
            let mut ctx = crate::hooks::HookContext::OnPlanCreated {
                session_id,
                plan: &mut artifact,
            };
            // Errors are non-fatal: an observability/handler error must not lose
            // a successfully-captured plan. The (possibly mutated) artifact is
            // still stored.
            let _ = chain.fire(&mut ctx).await;
        }

        // R4: persist the produced artifact to the RunStore seam under
        // PLAN_EXECUTE_EXTRAS_KEY (#76 — the durable single source of truth;
        // no longer mirrored into SessionState.extras). #142: the plan artifact
        // is DURABLE — key it by the STABLE project namespace, NOT the per-window
        // `session_id` (so a Ralph window reset re-reads the prior window's
        // plan). The put result is swallowed (matching the execute-phase
        // persist): a successfully captured plan must not be lost to a storage
        // hiccup.
        match serde_json::to_value(&artifact) {
            Ok(value) => {
                let _ = self
                    .config
                    .storage
                    .run()
                    .put(
                        &self.config.project_id.namespace(),
                        crate::plan::PLAN_EXECUTE_EXTRAS_KEY,
                        value,
                    )
                    .await;
            }
            Err(e) => {
                return Err(RunResult::Failure {
                    reason: HaltReason::PlanPhaseFailed {
                        error: crate::plan::PlanPhaseError::UnparseablePlan {
                            message: format!("failed to serialize plan artifact: {e}"),
                        },
                    },
                    session_id: session_id.clone(),
                    usage,
                    turns,
                    session_state: SessionState::default(),
                });
            }
        }

        Ok(PlanPhaseOutcome {
            artifact,
            usage,
            turns,
        })
    }

    /// A.6 deep-resume reconcile (#124, Q2): mark every task already `Completed`
    /// on the DURABLE RunStore checkpoint as `Completed` in the freshly-parsed
    /// `task_list` so it is NOT re-run. Tasks are matched positionally by `id`
    /// (the task list is regenerated deterministically from the same artifact).
    /// Extracted from the legacy `run_execute_phase` so the recursive
    /// [`PlanExecuteConfig::run`] owns the per-task loop while the durable-state
    /// reconcile stays here.
    pub(crate) async fn reconcile_deep_resume(
        &self,
        session_id: &SessionId,
        task_list: &mut crate::tasklist::TaskList,
    ) {
        use crate::tasklist::TaskStatus;
        // #142: the durable checkpoint is keyed by the STABLE project namespace,
        // NOT the per-window `session_id` — so deep-resume reads the SAME list a
        // prior Ralph window persisted. The ephemeral `session_id` is unused for
        // this durable read.
        let _ = session_id;
        if let Ok(Some(value)) = self
            .config
            .storage
            .run()
            .get(
                &self.config.project_id.namespace(),
                crate::tasklist::TASK_LIST_EXTRAS_KEY,
            )
            .await
        {
            if let Ok(saved) = serde_json::from_value::<crate::tasklist::TaskList>(value) {
                for t in &mut task_list.tasks {
                    if saved
                        .tasks
                        .iter()
                        .any(|s| s.id == t.id && s.status == TaskStatus::Completed)
                    {
                        t.status = TaskStatus::Completed;
                    }
                }
            }
        }
    }
}

// ── RunStrategy: the PlanExecute body (#124, moved from strategies.rs) ──
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

            // Plan→execute context threading: the `<files>` handoff built from the
            // plan phase's `read_file` results and seeded into each execute step so it
            // does not re-read what planning already read. Empty on the skip-plan path
            // (no plan turn ran) or when the planner read nothing.
            let mut plan_files_block = String::new();

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
                            session_state: plan_session,
                            ..
                        }) => {
                            // Capture the files the planner read so the execute steps
                            // inherit them (plan→execute threading). Guarded so a
                            // tools-suppressed retry (which reads nothing) cannot
                            // clobber a prior non-empty capture.
                            let block = render_plan_read_files(&plan_session);
                            if !block.is_empty() {
                                plan_files_block = block;
                            }
                            (output, usage, turns)
                        }
                        Some(RunResult::Escalate {
                            signal: HarnessSignal::PlanRecorded,
                            state,
                            usage,
                            turns,
                            ..
                        }) => {
                            // `set_plan` recorded the plan and terminated the plan
                            // phase (Fix A — its terminal signal, the plan-phase
                            // analogue of `plan_completed`, scoped to the plan
                            // toolset). Thread the reads it gathered, then proceed
                            // with an EMPTY plan_output: `capture_and_persist_plan`'s
                            // SC-28 fallback loads the runnable plan from the durable
                            // `task_list` store.
                            let block = render_plan_read_files(&state.session_state);
                            if !block.is_empty() {
                                plan_files_block = block;
                            }
                            (String::new(), usage, turns)
                        }
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

                    // Plan→execute threading: seed the files the plan phase already
                    // read so this step uses them instead of re-reading (edits update
                    // via the post-edit echo). Seeded before the step instruction.
                    if !plan_files_block.is_empty() {
                        executor
                            .seed_user_message(&mut step_session, &plan_files_block)
                            .await;
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

#[cfg(test)]
mod plan_read_files_tests {
    use super::render_plan_read_files;
    use crate::harness::types::SessionState;
    use crate::model::{Content, Message, Role, ToolCall};

    fn read_call(id: &str, path: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: Content::ToolCall(ToolCall {
                id: id.into(),
                name: "read_file".into(),
                input: serde_json::json!({ "path": path }),
            }),
        }
    }

    fn other_call(id: &str, name: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: Content::ToolCall(ToolCall {
                id: id.into(),
                name: name.into(),
                input: serde_json::json!({}),
            }),
        }
    }

    // Results are stored structurally and name the call they answer, mirroring
    // the context manager.
    fn result(call_id: &str, text: &str) -> Message {
        Message {
            role: Role::Tool,
            content: Content::ToolResult(crate::model::ToolResult {
                tool_use_id: call_id.into(),
                content: text.into(),
                is_error: false,
            }),
        }
    }

    fn error_result(call_id: &str, message: &str) -> Message {
        Message {
            role: Role::Tool,
            content: Content::ToolResult(crate::model::ToolResult {
                tool_use_id: call_id.into(),
                content: message.into(),
                is_error: true,
            }),
        }
    }

    fn session(messages: Vec<Message>) -> SessionState {
        SessionState {
            messages,
            extras: Default::default(),
        }
    }

    #[test]
    fn empty_when_the_plan_read_nothing() {
        assert_eq!(render_plan_read_files(&session(vec![])), "");
        // A non-read tool with a result yields nothing.
        assert_eq!(
            render_plan_read_files(&session(vec![
                other_call("b1", "bash_command"),
                result("b1", "ls output"),
            ])),
            ""
        );
    }

    #[test]
    fn threads_reads_dedupes_by_path_last_read_wins() {
        // One call per turn: call, result, call, result … (the real ordering). A
        // non-read tool (grep) is interleaved to check it is not paired in.
        let out = render_plan_read_files(&session(vec![
            read_call("c1", "src/util.rs"),
            result("c1", "fn a() {}"),
            other_call("c2", "grep"),
            result("c2", "2 matches"),
            read_call("c3", "src/main.rs"),
            result("c3", "fn main() {}"),
            // Re-read util.rs with newer content — last read wins, single entry.
            read_call("c4", "src/util.rs"),
            result("c4", "fn a() {} // v2"),
        ]));
        assert!(out.starts_with("<files>") && out.ends_with("</files>"));
        assert!(out.contains("path=\"src/main.rs\"") && out.contains("fn main() {}"));
        // util.rs appears once, with the LATEST content; grep output is NOT paired in.
        assert_eq!(out.matches("path=\"src/util.rs\"").count(), 1);
        assert!(out.contains("fn a() {} // v2"));
        assert!(!out.contains("2 matches"));
    }

    // The case ordinal pairing could not express: a turn issues several calls at
    // once and the results come back in a different order. Joining on the id is
    // what makes this correct — by position, main.rs would get util.rs's body.
    #[test]
    fn pairs_parallel_results_by_id_not_position() {
        let out = render_plan_read_files(&session(vec![
            read_call("c1", "src/util.rs"),
            read_call("c2", "src/main.rs"),
            result("c2", "fn main() {}"),
            result("c1", "fn a() {}"),
        ]));
        assert!(out.contains("<file path=\"src/util.rs\">\nfn a() {}"));
        assert!(out.contains("<file path=\"src/main.rs\">\nfn main() {}"));
    }

    // A failed read must not be handed to the planner as if it were the file's
    // contents — previously indistinguishable, since both were bare text.
    #[test]
    fn a_failed_read_contributes_no_file() {
        let out = render_plan_read_files(&session(vec![
            read_call("c1", "src/gone.rs"),
            error_result("c1", "read failed: No such file or directory"),
        ]));
        assert_eq!(out, "");
    }
}
