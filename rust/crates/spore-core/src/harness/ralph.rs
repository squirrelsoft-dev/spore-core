//! StandardHarness — Ralph progress/feature-list/completion helpers (Phase 6.1 impl split). Pure move.

use super::*;

impl StandardHarness {
    /// Ralph external completion check (issue #58, B1). Reads the deterministic
    /// `.spore/` files under `workspace_root` and reports whether the task is
    /// complete. Returns `None` when complete, `Some(reason)` when tasks remain
    /// (the reason describes what is left). This is the SAME logic the registered
    /// [`RalphStopHook`] applies — one source of truth for the completion
    /// mechanism.
    ///
    /// Contract (#142, decision 3 — the checkpoint MOVED off the `.spore/`
    /// filesystem onto the durable project-id [`RunStore`](crate::storage::RunStore)):
    ///   - [`RALPH_PROGRESS_KEY`](crate::storage::RALPH_PROGRESS_KEY) →
    ///     `{ "complete": bool, "remaining": [string] }`. `complete: true` with an
    ///     empty `remaining` ⇒ progress satisfied. Absent/unreadable/invalid ⇒
    ///     incomplete (so the agent learns to write it).
    ///   - [`RALPH_FEATURE_LIST_KEY`](crate::storage::RALPH_FEATURE_LIST_KEY) → a
    ///     JSON array of `{ "name", "passes" }` (the
    ///     [`FeatureListCheck`](crate::termination::FeatureListCheck) schema). Any
    ///     `passes: false` ⇒ incomplete. An ABSENT feature list is tolerated here
    ///     (progress is the primary signal); an invalid one is not.
    ///
    /// The reason strings stay byte-identical to the legacy `.spore/`-path
    /// messages so the surfaced [`HaltReason::RalphCompletionUnmet`] text is
    /// stable across the storage relocation.
    pub(crate) async fn ralph_completion_status(
        run_store: &Arc<dyn crate::storage::RunStore>,
        project: &crate::storage::ProjectId,
    ) -> Option<String> {
        let ns = project.namespace();
        let progress_value = match run_store.get(&ns, crate::storage::RALPH_PROGRESS_KEY).await {
            Ok(Some(v)) => v,
            // Absent OR a storage error ⇒ incomplete (the agent must write it).
            _ => return Some(".spore/progress.json missing".to_string()),
        };
        let progress: RalphProgress = match serde_json::from_value(progress_value) {
            Ok(p) => p,
            Err(e) => return Some(format!(".spore/progress.json invalid JSON: {e}")),
        };
        if !progress.complete {
            let detail = if progress.remaining.is_empty() {
                "task not marked complete".to_string()
            } else {
                format!("remaining: {}", progress.remaining.join(", "))
            };
            return Some(detail);
        }
        if !progress.remaining.is_empty() {
            return Some(format!("remaining: {}", progress.remaining.join(", ")));
        }

        // Progress says done — corroborate against the feature list when present.
        if let Ok(Some(value)) = run_store
            .get(&ns, crate::storage::RALPH_FEATURE_LIST_KEY)
            .await
        {
            let entries: Vec<RalphFeatureEntry> = match serde_json::from_value(value) {
                Ok(v) => v,
                Err(e) => return Some(format!(".spore/feature_list.json invalid JSON: {e}")),
            };
            let incomplete: Vec<String> = entries
                .into_iter()
                .filter(|e| !e.passes)
                .map(|e| e.name)
                .collect();
            if !incomplete.is_empty() {
                return Some(format!("incomplete features: {}", incomplete.join(", ")));
            }
        }
        None
    }

    /// Build the reload context block injected into each fresh context window
    /// (issue #58, R3). #142: reads the checkpoint from the durable project-id
    /// [`RunStore`](crate::storage::RunStore) (not the `.spore/` filesystem) and
    /// returns the verbatim progress + feature-list JSON (when present) so the
    /// re-seeded window knows what is done and what remains. Returns `None` when
    /// neither checkpoint key is set. The "Reloaded .spore/…" prefix is retained
    /// so the seeded prompt text is byte-stable across the relocation.
    pub(crate) async fn ralph_reload_context(
        run_store: &Arc<dyn crate::storage::RunStore>,
        project: &crate::storage::ProjectId,
    ) -> Option<String> {
        let ns = project.namespace();
        let mut parts: Vec<String> = Vec::new();
        if let Ok(Some(v)) = run_store.get(&ns, crate::storage::RALPH_PROGRESS_KEY).await {
            if let Ok(raw) = serde_json::to_string(&v) {
                parts.push(format!("Reloaded .spore/progress.json:\n{}", raw.trim()));
            }
        }
        if let Ok(Some(v)) = run_store
            .get(&ns, crate::storage::RALPH_FEATURE_LIST_KEY)
            .await
        {
            if let Ok(raw) = serde_json::to_string(&v) {
                parts.push(format!(
                    "Reloaded .spore/feature_list.json:\n{}",
                    raw.trim()
                ));
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }

    /// Write the Ralph progress checkpoint to the durable project-id
    /// [`RunStore`](crate::storage::RunStore) (issue #142, decision 3 — the WRITE
    /// path the relocated checkpoint needs; nothing wrote `progress.json` before).
    /// `complete: true` + empty `remaining` ⇒
    /// [`ralph_completion_status`](Self::ralph_completion_status) reports done.
    pub async fn write_ralph_progress(
        &self,
        complete: bool,
        remaining: Vec<String>,
    ) -> Result<(), crate::storage::StorageError> {
        let progress = RalphProgress {
            complete,
            remaining,
        };
        let value = serde_json::to_value(&progress)
            .map_err(|e| crate::storage::StorageError::Serialization(e.to_string()))?;
        self.config
            .storage
            .run()
            .put(
                &self.config.project_id.namespace(),
                crate::storage::RALPH_PROGRESS_KEY,
                value,
            )
            .await
    }

    /// Write the Ralph feature-list checkpoint to the durable project-id
    /// [`RunStore`](crate::storage::RunStore) (issue #142, decision 3). Each entry
    /// is `{ "name", "passes" }`; any `passes: false` keeps the run incomplete.
    pub async fn write_ralph_feature_list(
        &self,
        features: &[(String, bool)],
    ) -> Result<(), crate::storage::StorageError> {
        let entries: Vec<serde_json::Value> = features
            .iter()
            .map(|(name, passes)| serde_json::json!({ "name": name, "passes": passes }))
            .collect();
        let value = serde_json::Value::Array(entries);
        self.config
            .storage
            .run()
            .put(
                &self.config.project_id.namespace(),
                crate::storage::RALPH_FEATURE_LIST_KEY,
                value,
            )
            .await
    }

    // ========================================================================
    // HillClimbing loop strategy (issue #60)
    // ========================================================================
    //
    // Iterative optimization loop. Establishes a baseline metric over the
    // starting workspace, then repeatedly: runs ONE agent turn proposing a
    // change, evaluates the metric, and KEEPS or REVERTS based on whether the
    // new value strictly beats the running best (per `should_keep`). Generalizes
    // the autoresearch pattern.
    //
    // ## Loop semantics
    //   - Iteration 0 is a PURE baseline measurement of the starting workspace —
    //     NO agent turn. Its `ResultsEntry` has `status: kept`, and its
    //     `duration_secs` is the wall-clock time of the baseline evaluator call
    //     only. It seeds `current_best`. Agent turns begin at iteration 1.
    //   - Iterations 1.. : run one bounded ReAct agent turn (proposes a change),
    //     then evaluate. On a successful evaluation, route through
    //     `should_keep(new, current_best, payload_direction, min_improvement_delta)`:
    //       * keep   → status Kept, commit the accepted neighbor (advancing
    //                  `current_sha`/`current_best`, and `best_sha`/`best_score`
    //                  if it beats the best), reset the stagnation counter to 0.
    //       * no keep → status Discarded; if `revert_on_no_improvement`, run
    //                  `git reset --hard <current_sha>` via the sandbox (the last
    //                  ACCEPTED state, NOT the original HEAD — hill-climbing-loop-
    //                  correctness §4.3); increment the stagnation counter.
    //     On a MetricError (Crashed/Timeout/etc.), the iteration status is
    //     `IterationStatus::from_error` (Crashed/Timeout), the metric value is
    //     EMPTY in the TSV, the iteration counts as a non-improvement (increment
    //     the stagnation counter), and the loop continues.
    //   - `max_stagnation: Some(N)` ⇒ after N consecutive non-improvements, halt
    //     `StagnationLimitReached { iterations, best_metric }`. Improvement resets
    //     the counter. `max_stagnation: None` ⇒ run until the budget/turn limit
    //     (no stagnation halt).
    //   - Budget gates (turns/tokens/wall/cost) are honored per iteration exactly
    //     as the other run_* methods do; each iteration emits a fire-and-forget
    //     observability span carrying the metric value/delta.
    //   - The harness writes the TSV after the run to
    //     `{workspace_root}/.spore/results/{task_id}.tsv`.
    //
    // ## Six pinned spec decisions (resolved with the user — NOT relitigated)
    //   1. REVERT/COMMIT (revised by hill-climbing-loop-correctness): the harness
    //      now COMMITS each accepted candidate through the sandbox (`git add -A &&
    //      git commit`, D1) so `commit_hash` carries the real accepted SHA, and
    //      `revert_on_no_improvement: true` reverts via `git reset --hard
    //      <current_sha>` — the last accepted state, not the original HEAD. A
    //      wired `VcsProvider` owns commit/head/reset; with no provider AND no
    //      usable git (e.g. a fake sandbox), the SHAs are `None` and the loop
    //      degrades to the pre-fix behaviour (empty `commit_hash`, reset to HEAD).
    //   2. TSV FLOAT FORMAT: both `metric_value` and `duration_secs` are
    //      formatted with exactly 6 decimal places (`format!("{:.6}", x)`) for
    //      cross-language byte-identity.
    //   3. TSV SCHEMA: written to `{workspace_root}/.spore/results/{task_id}.tsv`,
    //      tab-separated, REQUIRED header then one row per iteration ascending.
    //      Columns: iteration, commit_hash, metric_value, direction, status,
    //      duration_secs, description. `commit_hash` empty when no VCS wired;
    //      `metadata` EXCLUDED; `metric_value` EMPTY on crashed/timeout rows;
    //      `direction` snake_case; `status` snake_case; `duration_secs` is
    //      `MetricResult.duration.as_secs_f64()` formatted `{:.6}`.
    //   4. DIRECTION: the strategy payload `direction` is authoritative for the
    //      keep/revert decision via `should_keep`; the evaluator's `direction()`
    //      is descriptive only. The TSV `direction` column records the payload
    //      direction.
    //   5. BASELINE: iteration 0 is a pure baseline (no agent turn); its row has
    //      `status: kept` and `duration_secs` = the baseline evaluator-call time.
    //   6. MISCONFIGURATION: a `None` `metric_evaluator` ⇒ halt
    //      `HillClimbingMisconfigured { reason }` (peer to SelfVerifyMisconfigured).
    //      No panic.
    //
    // No `// SPEC QUESTION:` markers — all six decisions are resolved.

}
