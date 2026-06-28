//! StandardHarness — PlanExecute task-list / plan-artifact / deep-resume helpers (Phase 6.1 impl split). Pure move.

use super::*;

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
    /// "respond with a single JSON plan object" instruction wrapped around the
    /// task. Seeded by the recursive [`PlanExecuteConfig::run`] plan dispatch
    /// before `self.plan.run(cx)` (#124).
    pub(crate) fn plan_directive(instruction: &str) -> String {
        format!(
            "Produce a step-by-step plan for the following task. Respond with a \
             single JSON object: {{\"tasks\": [<ordered step strings>], \
             \"rationale\": <string>}}.\n\nTask:\n{instruction}"
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
