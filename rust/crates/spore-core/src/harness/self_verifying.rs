//! StandardHarness — SelfVerifying evaluate phase (Phase 6.1 impl split). Pure move.

use super::*;

impl StandardHarness {
    /// Run the SelfVerifying evaluate phase (issue #61): a fresh evaluator RUN
    /// over a read-only sandbox in a never-shared session.
    ///
    /// Builds a child [`StandardHarness`] from a clone of `self.config` with the
    /// `agent` swapped to the evaluator agent (D2 defaulting) and the `sandbox`
    /// wrapped in a [`ReadOnlySandbox`] (R3). The evaluator runs a fresh ReAct
    /// loop seeded with the `role-evaluator` chunk (R4, presence-only) plus a
    /// review directive, in a freshly [`generate`](SessionId::generate)d session
    /// (R2/R9). Folds the evaluate run's usage into `total_usage`/`carried` (R8)
    /// and returns its terminal [`RunResult`].
    pub(crate) async fn run_evaluate_phase(
        &self,
        task: &Task,
        evaluator: Arc<dyn Agent>,
        eval_toolset: ToolsetRef,
        carried: &mut BudgetSnapshot,
        total_usage: &mut AggregateUsage,
    ) -> RunResult {
        // R3: derive a read-only sandbox internally from the build sandbox.
        let read_only_sandbox: Arc<dyn SandboxProvider> =
            Arc::new(ReadOnlySandbox::new(self.config.sandbox.clone()));

        // R2/R9: fresh, never-shared session id for the evaluate run.
        let eval_session_id = SessionId::generate();

        // R4 (presence-only): prepend the `role-evaluator` chunk content (if the
        // configured provider supplies it) to the review directive.
        let role_chunk = self.role_evaluator_chunk().await;
        let directive = match role_chunk {
            Some(content) => format!(
                "{content}\n\nReview the work produced for the following task and \
                 report whether it is correct. You did NOT write this code; default \
                 to FAIL unless you can confirm it is right.\n\nTask:\n{}",
                task.instruction
            ),
            None => format!(
                "Review the work produced for the following task and report whether \
                 it is correct. You did NOT write this code; default to FAIL unless \
                 you can confirm it is right.\n\nTask:\n{}",
                task.instruction
            ),
        };

        let eval_task = Task {
            id: TaskId::generate(),
            instruction: directive.clone(),
            session_id: eval_session_id.clone(),
            budget: task.budget.clone(),
            loop_strategy: LoopStrategy::ReAct(ReactConfig::per_loop(
                task.budget.max_turns.unwrap_or(u32::MAX),
            )),
        };

        // Child harness: clone the config, swap the sandbox to read-only. Cloning
        // shares the same observability/storage seams so the evaluate run's spans
        // land in the SAME trace stream (distinguished by its distinct session
        // id). The evaluate agent is passed to `run_react` directly (#124 — no
        // `config.agent`).
        let mut eval_config = self.config.clone();
        eval_config.sandbox = read_only_sandbox;
        // The evaluate phase is a NON-INTERACTIVE nested review run: the caller
        // never sees its (freshly generated) session id, so an approval/HITL
        // middleware that resolves `SurfaceToHuman` would pause this run with no
        // human able to resume it — the reviewer's first `read_file` would yield
        // `WaitingForHuman`, which the verifier reads as a misconfiguration, and
        // the review half would silently never run. The read-only sandbox already
        // enforces the only safety property a reviewer needs (no writes, no
        // command execution), so the caller's approval middleware is both
        // redundant and harmful here — drop it for the evaluate phase. (Pair with
        // an `eval_toolset` scoped to read-only tools so non-filesystem
        // side-effecting tools the sandbox does not gate stay out of reach.)
        eval_config.middleware = None;
        // SC-30: when the eval phase falls back to the GLOBAL catalogue (empty
        // `eval_toolset`), auto-derive a read-only VIEW of it — advertise +
        // dispatch only the intersection with `StandardTools::readonly_set()`, so
        // the reviewer cannot reach write / exec / side-effecting tools (incl.
        // web/MCP the read-only sandbox does not gate) WITHOUT the consumer
        // registering a scoped read-only toolset. A non-empty `eval_toolset` is an
        // explicit opt-in and is left untouched (it resolves via its own catalogue).
        if eval_toolset.0.is_empty() {
            if let Some(catalogue) = eval_config.catalogue_registry.take() {
                let allow: std::collections::HashSet<String> =
                    crate::tools::StandardTools::readonly_set()
                        .into_iter()
                        .map(|t| t.schema.name)
                        .collect();
                let real: Arc<dyn ToolRegistry> =
                    Arc::new(crate::tool_registry::RealToolRegistry::new(
                        catalogue,
                        eval_config.sandbox.clone(),
                        eval_session_id.clone(),
                        eval_config.project_id.clone(),
                        eval_config.storage.run().clone(),
                        eval_config.storage.memory().clone(),
                    ));
                // `catalogue_registry` is now None (taken), so the empty-handle
                // path of `effective_tool_registry` returns this filtered
                // `tool_registry` for the reviewer's dispatch + schema advertising.
                eval_config.tool_registry = Arc::new(ReadOnlyToolView::new(real, allow));
            }
        }
        let eval_harness = StandardHarness::new(eval_config);

        let mut eval_state = SessionState::default();
        eval_harness
            .config
            .context_manager
            .append_user_message(&mut eval_state, &directive)
            .await;

        let cap = task.budget.max_turns.unwrap_or(u32::MAX);
        let eval_result = eval_harness
            .run_react(
                eval_task,
                cap,
                eval_state,
                BudgetSnapshot::default(),
                None,
                evaluator,
                eval_toolset,
            )
            .await;

        fold_usage(total_usage, carried, &eval_result);
        eval_result
    }

    /// Look up the `role-evaluator` chunk content from the configured
    /// [`ChunkProvider`](crate::prompt_assembly::ChunkProvider) (R4,
    /// presence-only). Returns `None` if the provider has no such chunk or fails
    /// to load.
    pub(crate) async fn role_evaluator_chunk(&self) -> Option<String> {
        let chunks = self.config.chunk_provider.load().await.ok()?;
        chunks
            .into_iter()
            .find(|c| c.id == "role-evaluator")
            .map(|c| c.content)
    }

    // ========================================================================
    // Ralph loop strategy (issue #58)
    // ========================================================================
    //
    // Multi-context-window continuation loop. The model's exit attempt is
    // intercepted — instead of terminating, the harness RESETS the context
    // window (discards the prior `SessionState`, builds a FRESH
    // `SessionState::default()`), RELOADS state from the filesystem (the
    // deterministic `.spore/progress.json` + `.spore/feature_list.json` files —
    // B4: no git-log read in v1), and resumes — until an external completion
    // check passes. The filesystem is what makes multi-context-window work
    // possible: each window starts from nothing but the instruction plus the
    // reloaded `.spore/` state.
    //
    // ## Completion mechanism (B1): driven off the `Stop` hook (issue #69).
    //   At construction [`StandardHarness::new`] registers a [`RalphStopHook`]
    //   that reads `.spore/progress.json` (+ `.spore/feature_list.json`): while
    //   tasks remain incomplete it returns [`HookDecision::Block`] (the reason
    //   describes what is left) and the harness loops into a new context window;
    //   when all tasks are complete it returns [`HookDecision::Continue`] and the
    //   loop terminates with success. There is NO dedicated `completion_check`
    //   config field and the deprecated `CompletionCheck` trait is NOT reused.
    //   The harness fires the Stop hook (`fire_stop_hooks` on `FinalResponse`);
    //   this strategy ALSO evaluates the SAME filesystem check between windows
    //   ([`ralph_completion_status`]) to drive the OUTER reset loop and to decide
    //   the terminal outcome.
    //
    // ## Config fields this strategy reads:
    //   - `config.max_resets` (B3, default 3) — the OUTER loop cap: the maximum
    //     number of context-window RESETS. Independent from `max_turns` (the
    //     per-window ReAct turn budget) and `max_stop_blocks` (the per-sub-loop
    //     Stop-block cap).
    //
    // ## Canonical filesystem paths (B2, `.spore/`-prefixed):
    //   - `{workspace_root}/.spore/progress.json`
    //   - `{workspace_root}/.spore/feature_list.json`
    //   (matches [`FeatureListCheck::new`](crate::termination::FeatureListCheck)).
    //
    // ## Terminal `HaltReason` variant this strategy produces (peer to
    //    [`SelfVerifyExhausted`](HaltReason::SelfVerifyExhausted)):
    //   - [`RalphCompletionUnmet`](HaltReason::RalphCompletionUnmet)
    //     `{ iterations, last_reason }` — reached `max_resets` context windows
    //     with tasks still incomplete.
    //
    // ## Rules enforced (each maps to a test):
    //   - R1  the model's exit attempt RESETS the context window instead of
    //         terminating, while tasks remain incomplete.
    //   - R2  each reset builds a FRESH `SessionState::default()` — no message
    //         carryover between context windows.
    //   - R3  the filesystem reload injects the reloaded `.spore/progress.json` +
    //         `.spore/feature_list.json` content into the fresh seed.
    //   - R4  completion pattern `incomplete,incomplete,complete` ⇒ `Success` at
    //         iteration 3.
    //   - R5  always-incomplete ⇒ exactly `max_resets` iterations ⇒
    //         `Failure { RalphCompletionUnmet }`.
    //   - R6  budgets fold across ALL context windows (`fold_usage`).
    //   - R7  each reset is traceable (a distinct generated session id per
    //         window, finalized via observability).
    //
    // No `// SPEC QUESTION:` markers — B1–B4 are resolved.
}
