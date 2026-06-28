//! StandardHarness — HillClimbing reset/commit/metric/TSV helpers (Phase 6.1 impl split). Pure move.

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
