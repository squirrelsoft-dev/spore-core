//! StandardHarness — StrategyExecutor trait impl (Phase 6.1 impl split). Pure move.

use super::*;

/// [`StrategyExecutor`] impl (#124): the harness-side primitives the recursive
/// per-variant [`RunStrategy::run`] bodies delegate to. Each primitive wraps an
/// existing, tested orchestration method so behavior stays at parity — the only
/// structural change is that the per-variant bodies now own their loops and the
/// central dispatch `match` is gone.
impl StrategyExecutor for StandardHarness {
    fn react_window<'a>(
        &'a self,
        task: Task,
        max_iterations: u32,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
        agent: Arc<dyn Agent>,
        toolset: ToolsetRef,
        output_schema: Option<serde_json::Value>,
        output_schema_max_retries: u32,
        system_prompt: Option<String>,
        execute_step: bool,
    ) -> BoxFut<'a, RunResult> {
        Box::pin(async move {
            self.run_react_inner(
                task,
                max_iterations,
                session_state,
                budget_used,
                on_stream,
                agent,
                toolset,
                output_schema,
                output_schema_max_retries,
                system_prompt,
                execute_step,
            )
            .await
        })
    }

    fn resolve_agent_ref(
        &self,
        r: &AgentRef,
        session_id: &SessionId,
    ) -> Result<Arc<dyn Agent>, Box<RunResult>> {
        self.config
            .registry
            .resolve_agent(r)
            .cloned()
            .ok_or_else(|| {
                Box::new(RunResult::Failure {
                    reason: HaltReason::ConfigurationError {
                        error: HarnessError::UnresolvedHandle {
                            kind: "agent".to_string(),
                            key: r.0.clone(),
                        },
                    },
                    session_id: session_id.clone(),
                    usage: AggregateUsage::default(),
                    turns: 0,
                    session_state: SessionState::default(),
                })
            })
    }

    fn evaluate_phase<'a>(
        &'a self,
        task: &'a Task,
        eval_agent: Arc<dyn Agent>,
        eval_toolset: ToolsetRef,
        carried: &'a mut BudgetSnapshot,
        total_usage: &'a mut AggregateUsage,
    ) -> BoxFut<'a, RunResult> {
        Box::pin(async move {
            self.run_evaluate_phase(task, eval_agent, eval_toolset, carried, total_usage)
                .await
        })
    }

    fn append_user_message<'a>(
        &'a self,
        session_state: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            self.config
                .context_manager
                .append_user_message(session_state, text)
                .await;
        })
    }

    fn workspace_root(&self) -> std::path::PathBuf {
        self.config.sandbox.workspace_root().to_path_buf()
    }

    fn seed_user_message<'a>(
        &'a self,
        session_state: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            self.config
                .context_manager
                .append_user_message(session_state, text)
                .await;
        })
    }

    fn capture_plan_artifact<'a>(
        &'a self,
        session_id: &'a SessionId,
        plan_output: &'a str,
        usage: AggregateUsage,
        turns: u32,
    ) -> BoxFut<'a, Result<PlanPhaseOutcome, RunResult>> {
        Box::pin(async move {
            self.capture_and_persist_plan(session_id, plan_output, usage, turns)
                .await
        })
    }

    fn plan_directive(&self, instruction: &str) -> String {
        StandardHarness::plan_directive(instruction)
    }

    fn plan_empty_max_retries(&self) -> u32 {
        self.config.plan_empty_max_retries
    }

    fn run_plan_subtree<'a>(
        &'a self,
        plan: &'a LoopStrategy,
        plan_task: Task,
        plan_session: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
        forced_output_schema: Option<serde_json::Value>,
    ) -> BoxFut<'a, Option<RunResult>> {
        Box::pin(async move {
            // #124 Q1: the planner concept is DROPPED — the plan child's leaf
            // `ReactConfig.agent` is authoritative and resolved by the recursing
            // leaf itself. `plan_task.loop_strategy == plan` (the caller stamps it),
            // so the WHOLE plan loop runs by driving that task.
            //
            // Drive it through `drive_strategy_with_resume_seed` — the SAME single
            // budget-resolution site the top level uses — instead of a bare
            // `plan.run(cx)` that read only `terminal_override`. A plan phase that
            // exhausts its turn budget propagates a `BudgetExhausted` StrategyOutcome
            // (rule 6: the leaf never self-resolves), which the bare drive DISCARDED,
            // returning `None` → the opaque `PlanPhaseFailed{"plan sub-strategy
            // produced no terminal"}`. Routing through the resolver makes the plan
            // phase auto-continue (Continue / AutoContinue) and then surface a
            // `WaitingForHuman{BudgetExhausted}` under `SurfaceToHuman` — exactly
            // like the execute phase — so the host can grant more budget and the
            // run continues (the plan-phase resume seed, #138 AC3, re-attaches the
            // stalled planner conversation). A plan that COMPLETES still returns its
            // `Success` terminal verbatim (byte-identical to before). `plan` is
            // unused now (the task carries the same strategy); kept for the trait
            // signature / call-site symmetry.
            let _ = plan;
            let result = self
                .drive_strategy_with_resume_seed(
                    plan_task,
                    plan_session,
                    budget_used,
                    on_stream,
                    None,  // resume_continues
                    None,  // resume_seed (the plan child plans fresh each attempt)
                    false, // continue_task_list
                    forced_output_schema,
                )
                .await;
            Some(result)
        })
    }

    fn reconcile_completed_tasks<'a>(
        &'a self,
        session_id: &'a SessionId,
        task_list: &'a mut crate::tasklist::TaskList,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            self.reconcile_deep_resume(session_id, task_list).await;
        })
    }

    fn fire_task_advance<'a>(
        &'a self,
        session_id: &'a SessionId,
        step_task: &'a mut Task,
        task_index: usize,
        total_tasks: usize,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            if let Some(chain) = self.config.hooks.as_ref() {
                let mut ctx = crate::hooks::HookContext::OnTaskAdvance {
                    session_id,
                    task: step_task,
                    task_index,
                    total_tasks,
                };
                let _ = chain.fire(&mut ctx).await;
            }
        })
    }

    fn persist_task_list<'a>(
        &'a self,
        session_id: &'a SessionId,
        task_list: &'a crate::tasklist::TaskList,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            // Delegate to the existing private method (same RunStore seam).
            StandardHarness::persist_task_list(self, session_id, task_list).await
        })
    }

    fn load_task_list<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Option<crate::tasklist::TaskList>> {
        // #142: the task_list is DURABLE — read it from the STABLE project
        // namespace, NOT the per-window `session_id` (so the execute phase of a
        // fresh Ralph window sees the prior window's list). The ephemeral
        // `session_id` is unused for this durable read.
        let _ = session_id;
        Box::pin(async move {
            // #126 (decision C): the one authoring path — read the persisted
            // TaskList (with real blockers) from the RunStore under
            // TASK_LIST_EXTRAS_KEY. A storage miss / deserialize failure yields
            // None (the executor then falls back to the plan artifact).
            match self
                .config
                .storage
                .run()
                .get(
                    &self.config.project_id.namespace(),
                    crate::tasklist::TASK_LIST_EXTRAS_KEY,
                )
                .await
            {
                Ok(Some(value)) => serde_json::from_value::<crate::tasklist::TaskList>(value).ok(),
                _ => None,
            }
        })
    }

    fn clear_task_list<'a>(&'a self, session_id: &'a SessionId) -> BoxFut<'a, ()> {
        // #142: the durable task_list is keyed by the STABLE project namespace,
        // NOT the ephemeral per-run `session_id` — so the delete must target the
        // same project-scoped key as `persist_task_list`/`load_task_list`.
        let _ = session_id;
        Box::pin(async move {
            let _ = self
                .config
                .storage
                .run()
                .delete(
                    &self.config.project_id.namespace(),
                    crate::tasklist::TASK_LIST_EXTRAS_KEY,
                )
                .await;
        })
    }

    fn take_observed_writes(&self) -> Vec<String> {
        match self.observed_writes.lock() {
            Ok(mut acc) => std::mem::take(&mut *acc),
            Err(_) => Vec::new(),
        }
    }

    fn clear_observed_writes(&self) {
        if let Ok(mut acc) = self.observed_writes.lock() {
            acc.clear();
        }
    }

    fn ralph_seed_session<'a>(&'a self, instruction: &'a str) -> BoxFut<'a, SessionState> {
        Box::pin(async move {
            let mut session_state = SessionState::default();
            self.config
                .context_manager
                .append_user_message(&mut session_state, instruction)
                .await;
            // #142: the reload context now comes from the durable project-id
            // RunStore checkpoint, not the `.spore/` filesystem.
            if let Some(reload) =
                Self::ralph_reload_context(self.config.storage.run(), &self.config.project_id).await
            {
                self.config
                    .context_manager
                    .append_user_message(&mut session_state, &reload)
                    .await;
            }
            // R3 (issue #58 v2): inject git history when a `VcsProvider` is wired.
            if let Some(vcs) = &self.config.vcs_provider {
                let args = VcsLogArgs {
                    max_entries: 20,
                    since_ref: None,
                    format: None,
                };
                if let Ok(log) = vcs.log(&args).await {
                    let trimmed = log.trim();
                    if !trimmed.is_empty() {
                        let block = format!("Recent VCS history:\n{trimmed}");
                        self.config
                            .context_manager
                            .append_user_message(&mut session_state, &block)
                            .await;
                    }
                }
            }
            session_state
        })
    }

    fn ralph_completion_status<'a>(&'a self) -> BoxFut<'a, Option<String>> {
        Box::pin(async move {
            // #142: read the checkpoint from the durable project-id RunStore.
            Self::ralph_completion_status(self.config.storage.run(), &self.config.project_id).await
        })
    }

    fn ralph_max_resets(&self) -> u32 {
        self.config.max_resets
    }

    fn escalation_mode(&self) -> EscalationMode {
        // No longer `Copy` (SC-5's `AutoContinue` holds an `Arc` callback); clone
        // out — the callback is refcounted, so this is cheap.
        self.config.escalation_mode.clone()
    }

    fn enforce_output_schemas(&self) -> bool {
        self.config.enforce_output_schemas
    }

    fn output_schema_max_retries(&self) -> u32 {
        self.config.output_schema_max_retries
    }

    fn resolve_metric_evaluator(
        &self,
        key: &str,
        session_id: &SessionId,
    ) -> Result<Arc<dyn crate::metric::MetricEvaluator>, Box<RunResult>> {
        self.config
            .registry
            .resolve_metric_evaluator(key)
            .cloned()
            .ok_or_else(|| {
                Box::new(RunResult::Failure {
                    reason: HaltReason::HillClimbingMisconfigured {
                        reason: format!(
                            "HillClimbing requires a metric evaluator registered under key {key:?}"
                        ),
                    },
                    session_id: session_id.clone(),
                    usage: AggregateUsage::default(),
                    turns: 0,
                    session_state: SessionState::default(),
                })
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn hill_baseline<'a>(
        &'a self,
        evaluator: &'a Arc<dyn crate::metric::MetricEvaluator>,
        session_id: &'a SessionId,
        task_id: &'a TaskId,
        direction: HillClimbingDirection,
        rows: &'a mut Vec<crate::metric::ResultsEntry>,
        span_seq: &'a mut u64,
        total_usage: AggregateUsage,
        turns: u32,
    ) -> BoxFut<'a, Result<(f64, Option<String>), RunResult>> {
        Box::pin(async move {
            use crate::metric::{IterationStatus, MetricResult, ResultsEntry};
            use crate::termination::SessionStateSnapshot;
            let workspace_root = self.config.sandbox.workspace_root().to_path_buf();
            let snapshot = SessionStateSnapshot::new(
                session_id.clone(),
                task_id.clone(),
                SessionState::default(),
                workspace_root.clone(),
            );
            let baseline = evaluator
                .evaluate(self.config.sandbox.as_ref(), &snapshot)
                .await;
            match baseline {
                Ok(MetricResult {
                    value, duration, ..
                }) => {
                    // D4: commit the starting tree as the baseline candidate so
                    // `current_sha`/reverts are always well-defined (handles a
                    // dirty start). `None` when the workspace has no usable VCS,
                    // in which case the loop degrades to its pre-fix behaviour.
                    let start_sha = self
                        .hill_climbing_commit_accepted(&Self::hill_commit_label(0))
                        .await;
                    rows.push(ResultsEntry {
                        iteration: 0,
                        commit_hash: start_sha.clone(),
                        metric_value: value,
                        direction,
                        status: IterationStatus::Kept,
                        duration,
                        description: evaluator.description(),
                        metadata: Default::default(),
                    });
                    self.emit_hill_climbing_span(
                        session_id,
                        task_id,
                        span_seq,
                        0,
                        Some(value),
                        None,
                        IterationStatus::Kept,
                        false,
                    );
                    Ok((value, start_sha))
                }
                Err(err) => {
                    let status = IterationStatus::from_error(&err);
                    rows.push(ResultsEntry {
                        iteration: 0,
                        // Baseline eval failed ⇒ no baseline candidate to commit.
                        commit_hash: None,
                        metric_value: f64::NAN,
                        direction,
                        status,
                        duration: Duration::ZERO,
                        description: evaluator.description(),
                        metadata: Default::default(),
                    });
                    self.emit_hill_climbing_span(
                        session_id, task_id, span_seq, 0, None, None, status, false,
                    );
                    self.write_hill_climbing_tsv(&workspace_root, task_id, rows)
                        .await;
                    Err(RunResult::Failure {
                        reason: HaltReason::HillClimbingMisconfigured {
                            reason: format!("baseline evaluation failed: {err}"),
                        },
                        session_id: session_id.clone(),
                        usage: total_usage,
                        turns,
                        session_state: SessionState::default(),
                    })
                }
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn hill_iteration<'a>(
        &'a self,
        evaluator: &'a Arc<dyn crate::metric::MetricEvaluator>,
        session_id: &'a SessionId,
        task_id: &'a TaskId,
        iteration: u32,
        direction: HillClimbingDirection,
        revert_on_no_improvement: bool,
        min_improvement_delta: Option<f64>,
        cand: &'a mut HillCandidate,
        rows: &'a mut Vec<crate::metric::ResultsEntry>,
        span_seq: &'a mut u64,
    ) -> BoxFut<'a, bool> {
        Box::pin(async move {
            use crate::metric::{should_keep, IterationStatus, MetricResult, ResultsEntry};
            use crate::termination::SessionStateSnapshot;
            let workspace_root = self.config.sandbox.workspace_root().to_path_buf();
            let snapshot = SessionStateSnapshot::new(
                session_id.clone(),
                task_id.clone(),
                SessionState::default(),
                workspace_root,
            );
            let eval = evaluator
                .evaluate(self.config.sandbox.as_ref(), &snapshot)
                .await;
            match eval {
                Ok(MetricResult {
                    value, duration, ..
                }) => {
                    let kept =
                        should_keep(value, cand.current_best, direction, min_improvement_delta);
                    let delta = match direction {
                        HillClimbingDirection::Minimize => cand.current_best - value,
                        HillClimbingDirection::Maximize => value - cand.current_best,
                    };
                    if kept {
                        // Commit the accepted neighbor → durable pointer (§4.2).
                        // `None` ⇒ no usable VCS: the keep still advances the
                        // score high-water mark (pre-fix behaviour), but there is
                        // no SHA to revert a later discard to.
                        let nbr_sha = self
                            .hill_climbing_commit_accepted(&Self::hill_commit_label(iteration))
                            .await;
                        cand.current_sha = nbr_sha.clone();
                        cand.current_best = value;
                        // Update the best-ever pointer when this strictly beats it
                        // (D2: tracked separately even though equal under greedy).
                        let beats_best = match direction {
                            HillClimbingDirection::Minimize => value < cand.best_score,
                            HillClimbingDirection::Maximize => value > cand.best_score,
                        };
                        if beats_best {
                            cand.best_sha = nbr_sha.clone();
                            cand.best_score = value;
                        }
                        rows.push(ResultsEntry {
                            iteration,
                            commit_hash: nbr_sha,
                            metric_value: value,
                            direction,
                            status: IterationStatus::Kept,
                            duration,
                            description: evaluator.description(),
                            metadata: Default::default(),
                        });
                        self.emit_hill_climbing_span(
                            session_id,
                            task_id,
                            span_seq,
                            iteration,
                            Some(value),
                            Some(delta),
                            IterationStatus::Kept,
                            false,
                        );
                        false
                    } else {
                        // Revert to the last ACCEPTED state (§4.3), NOT HEAD.
                        let reverted = if revert_on_no_improvement {
                            self.hill_climbing_reset_to(cand.current_sha.as_deref())
                                .await;
                            true
                        } else {
                            false
                        };
                        rows.push(ResultsEntry {
                            iteration,
                            // The discard row records the surviving accepted SHA
                            // (== HEAD after the reset; == HEAD anyway when revert
                            // is off, since uncommitted neighbor edits don't move
                            // HEAD).
                            commit_hash: cand.current_sha.clone(),
                            metric_value: value,
                            direction,
                            status: IterationStatus::Discarded,
                            duration,
                            description: evaluator.description(),
                            metadata: Default::default(),
                        });
                        self.emit_hill_climbing_span(
                            session_id,
                            task_id,
                            span_seq,
                            iteration,
                            Some(value),
                            Some(delta),
                            IterationStatus::Discarded,
                            reverted,
                        );
                        true
                    }
                }
                Err(err) => {
                    let status = IterationStatus::from_error(&err);
                    let reverted = if revert_on_no_improvement {
                        self.hill_climbing_reset_to(cand.current_sha.as_deref())
                            .await;
                        true
                    } else {
                        false
                    };
                    rows.push(ResultsEntry {
                        iteration,
                        commit_hash: cand.current_sha.clone(),
                        metric_value: f64::NAN,
                        direction,
                        status,
                        duration: Duration::ZERO,
                        description: evaluator.description(),
                        metadata: Default::default(),
                    });
                    self.emit_hill_climbing_span(
                        session_id, task_id, span_seq, iteration, None, None, status, reverted,
                    );
                    true
                }
            }
        })
    }

    fn hill_apply_winner<'a>(&'a self, best_sha: Option<&'a str>) -> BoxFut<'a, ()> {
        Box::pin(async move {
            // Winner application (§4.6): reset to the best accepted candidate
            // before a terminal return. No-op when there is no committed best.
            if let Some(sha) = best_sha.filter(|s| !s.is_empty()) {
                self.hill_climbing_reset_to(Some(sha)).await;
            }
        })
    }

    fn hill_write_tsv<'a>(
        &'a self,
        task_id: &'a TaskId,
        rows: &'a [crate::metric::ResultsEntry],
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            let workspace_root = self.config.sandbox.workspace_root().to_path_buf();
            self.write_hill_climbing_tsv(&workspace_root, task_id, rows)
                .await;
        })
    }

    fn finalize<'a>(&'a self, result: &'a RunResult) -> BoxFut<'a, ()> {
        Box::pin(async move {
            match result {
                RunResult::Success { session_id, .. } => {
                    self.finalize_observability(session_id, SessionOutcome::Success)
                        .await;
                }
                RunResult::Failure {
                    session_id, reason, ..
                } => {
                    self.finalize_observability(
                        session_id,
                        SessionOutcome::Failure {
                            reason: format!("{reason:?}"),
                        },
                    )
                    .await;
                }
                RunResult::Escalate { session_id, .. } => {
                    self.finalize_observability(session_id, SessionOutcome::Escalated)
                        .await;
                }
                // Non-terminal pauses are never finalized.
                RunResult::WaitingForHuman { .. } | RunResult::Consult { .. } => {}
            }
        })
    }
}
