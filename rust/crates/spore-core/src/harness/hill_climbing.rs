//! HillClimbing strategy: the `StandardHarness` reset/commit/metric/TSV helpers
//! plus `impl RunStrategy for HillClimbingConfig` (the keep/revert climb loop).

use super::*;

impl StandardHarness {
    /// Reset the working tree for a no-improvement iteration
    /// (`hill-climbing-loop-correctness` §4.3). The load-bearing fix: revert
    /// targets the last ACCEPTED candidate (`target_sha`), NOT the original
    /// `HEAD`. Routing precedence (SC-14 — the harness NEVER spawns git
    /// directly):
    ///   - a wired [`VcsProvider`] owns the reset: `reset_to(sha)` for a known
    ///     SHA, else `revert()` (its revert-to-last-known-good);
    ///   - otherwise git-through-sandbox: `git reset --hard <sha>` for a known
    ///     SHA, else `git reset --hard HEAD` — byte-identical to the pre-fix
    ///     fallback when the workspace has no usable VCS (`target_sha == None`).
    ///
    /// Best-effort: an error / non-zero exit is swallowed and the loop continues.
    pub(crate) async fn hill_climbing_reset_to(&self, target_sha: Option<&str>) {
        let known = target_sha.filter(|s| !s.is_empty());
        if let Some(vcs) = &self.config.vcs_provider {
            match known {
                Some(sha) => {
                    let _ = vcs.reset_to(sha).await;
                }
                // No durable SHA (no-op/degenerate VCS) ⇒ fall back to the
                // provider's revert-to-last-known-good, preserving pre-fix
                // behaviour for providers that only implement `revert`.
                None => {
                    let _ = vcs.revert().await;
                }
            }
            return;
        }
        let target = known.unwrap_or("HEAD").to_string();
        let args = ["reset".to_string(), "--hard".to_string(), target];
        let _ = self
            .config
            .sandbox
            .execute_command("git", &args, None, None)
            .await;
    }

    /// Resolve the workspace's current commit SHA
    /// (`hill-climbing-loop-correctness` §4.4). A wired [`VcsProvider`] supplies
    /// it (`head()`); otherwise `git rev-parse HEAD` THROUGH the sandbox. Returns
    /// `None` when there is no usable VCS — a blocked spawn, a non-zero exit, or
    /// empty stdout (the case for a fake/no-op sandbox). A `None` here is what
    /// makes the loop degrade gracefully to its pre-fix behaviour rather than
    /// recording a bogus SHA.
    pub(crate) async fn hill_climbing_head_sha(&self) -> Option<String> {
        if let Some(vcs) = &self.config.vcs_provider {
            return vcs
                .head()
                .await
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
        }
        let args = ["rev-parse".to_string(), "HEAD".to_string()];
        let out = self
            .config
            .sandbox
            .execute_command("git", &args, None, None)
            .await
            .ok()?;
        if out.exit_code != 0 {
            return None;
        }
        let sha = out.stdout.trim().to_string();
        (!sha.is_empty()).then_some(sha)
    }

    /// Commit the current working tree as an accepted candidate and return its
    /// SHA (`hill-climbing-loop-correctness` §4.4, Decision D1: commit through
    /// the sandbox in the default path — no `VcsProvider` required). A wired
    /// provider's `commit_checkpoint` wins; otherwise `git add -A && git commit
    /// -m <label>` THROUGH the sandbox, then [`hill_climbing_head_sha`]. Returns
    /// `None` when no SHA could be obtained (no usable VCS) — the caller then
    /// records an empty `commit_hash` and keeps the pre-fix score-only behaviour
    /// (there is nothing durable to revert to, which is inherent without a VCS).
    ///
    /// [`hill_climbing_head_sha`]: Self::hill_climbing_head_sha
    pub(crate) async fn hill_climbing_commit_accepted(&self, label: &str) -> Option<String> {
        if let Some(vcs) = &self.config.vcs_provider {
            return vcs
                .commit_checkpoint(label)
                .await
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
        }
        // Stage everything EXCEPT the harness's own `.spore/` diagnostics: it
        // must stay UNTRACKED so a later `git reset --hard` (revert / winner
        // apply) leaves the live results ledger on disk (`git reset --hard` never
        // touches untracked files). Committing `.spore/` would let a revert
        // clobber the in-progress TSV with a stale committed copy.
        let add = [
            "add".to_string(),
            "-A".to_string(),
            "--".to_string(),
            HILL_CLIMB_COMMIT_EXCLUDE.to_string(),
        ];
        let _ = self
            .config
            .sandbox
            .execute_command("git", &add, None, None)
            .await;
        let commit = ["commit".to_string(), "-m".to_string(), label.to_string()];
        let _ = self
            .config
            .sandbox
            .execute_command("git", &commit, None, None)
            .await;
        self.hill_climbing_head_sha().await
    }

    /// The commit message for an accepted HillClimbing candidate
    /// (`hill-climbing-loop-correctness` §4.4). Kept as a single helper so the
    /// exact string is byte-identical across the four reference implementations
    /// (per `consumer-friction-port-patterns`). Iteration 0 is the baseline.
    pub(crate) fn hill_commit_label(iteration: u32) -> String {
        if iteration == 0 {
            "hill-climb baseline".to_string()
        } else {
            format!("hill-climb iter {iteration}")
        }
    }

    /// Emit one fire-and-forget per-iteration observability span for a
    /// HillClimbing run (issue #60). No-op when no provider is configured.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_hill_climbing_span(
        &self,
        session_id: &SessionId,
        task_id: &TaskId,
        span_seq: &mut u64,
        iteration: u32,
        metric_value: Option<f64>,
        delta: Option<f64>,
        status: crate::metric::IterationStatus,
        reverted: bool,
    ) {
        if let Some(obs) = self.config.observability.as_ref() {
            let base = SpanBase::new_root(
                SpanId::new(format!("{}-hill-{}", session_id.as_str(), *span_seq)),
                session_id.clone(),
                task_id.clone(),
                SpanKind::Warn,
                Timestamp::now(),
            );
            let status_str = serde_json::to_value(status)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_else(|| format!("{status:?}").to_lowercase());
            obs.emit_warn(WarnSpan::new(
                base,
                WarnEvent::HillClimbingIteration {
                    iteration,
                    metric_value,
                    delta,
                    status: status_str,
                    reverted,
                },
            ));
            *span_seq += 1;
        }
    }

    /// Serialize the HillClimbing results log and write it to
    /// `{workspace_root}/.spore/results/{task_id}.tsv` (issue #60, Decisions 2/3).
    /// Tab-separated, REQUIRED header, one row per iteration in ascending order.
    /// Floats use exactly 6 decimal places for cross-language byte-identity.
    /// `metric_value` is the empty string on crashed/timeout rows. `metadata` is
    /// excluded. Best-effort: a filesystem error is swallowed (the run outcome is
    /// authoritative, the TSV is a diagnostic artifact).
    pub(crate) async fn write_hill_climbing_tsv(
        &self,
        workspace_root: &std::path::Path,
        task_id: &TaskId,
        rows: &[crate::metric::ResultsEntry],
    ) {
        let body = Self::render_hill_climbing_tsv(rows);
        let dir = workspace_root.join(".spore/results");
        let _ = tokio::fs::create_dir_all(&dir).await;
        let path = dir.join(format!("{}.tsv", task_id.as_str()));
        let _ = tokio::fs::write(&path, body).await;
    }

    /// Render the HillClimbing results-log TSV body (issue #60, Decisions 2/3).
    /// Pure function over the rows so the exact byte content is unit-testable and
    /// cross-language-comparable. Trailing newline after every row (including the
    /// last) so appends and diffs stay line-oriented.
    pub(crate) fn render_hill_climbing_tsv(rows: &[crate::metric::ResultsEntry]) -> String {
        use crate::metric::IterationStatus;
        let mut out = String::from(
            "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n",
        );
        for r in rows {
            // Decision 3: metric_value is EMPTY on crashed/timeout rows.
            let metric_value = match r.status {
                IterationStatus::Crashed | IterationStatus::Timeout => String::new(),
                _ => format!("{:.6}", r.metric_value),
            };
            let commit_hash = r.commit_hash.clone().unwrap_or_default();
            let direction = match r.direction {
                HillClimbingDirection::Minimize => "minimize",
                HillClimbingDirection::Maximize => "maximize",
            };
            let status = match r.status {
                IterationStatus::Kept => "kept",
                IterationStatus::Discarded => "discarded",
                IterationStatus::Crashed => "crashed",
                IterationStatus::Timeout => "timeout",
            };
            let duration_secs = format!("{:.6}", r.duration.as_secs_f64());
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                r.iteration,
                commit_hash,
                metric_value,
                direction,
                status,
                duration_secs,
                r.description
            ));
        }
        out
    }
}

// ── RunStrategy: the HillClimbing body (#124, moved from strategies.rs) ──
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
                    StandardHarness::budget_exceeded(&task.budget, &carried, Instant::now(), None)
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
