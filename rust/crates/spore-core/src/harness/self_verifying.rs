//! SelfVerifying strategy: the `StandardHarness` evaluate-phase helper plus
//! `impl RunStrategy for SelfVerifyingConfig` (the build↔evaluate↔verify loop).

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
            .run_react(ReactParams::legacy(
                eval_task,
                cap,
                eval_state,
                BudgetSnapshot::default(),
                None,
                evaluator,
                eval_toolset,
            ))
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

// ── RunStrategy: the SelfVerifying body (#124, moved from strategies.rs) ──
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
