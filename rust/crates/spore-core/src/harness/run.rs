//! StandardHarness — run/resume/consult orchestration (Phase 6.1 impl split). Pure move.

use super::*;

impl StandardHarness {
    /// Consult resume seam (issue #114). Mirrors the `resume_inner`
    /// clarification branch: the [`ConsultResponse`] text is injected as the
    /// tool RESULT for the head pending (consult) call — NOT appended as a
    /// free-standing user message (R10) — then any remaining pending calls are
    /// dispatched and the ReAct loop resumes.
    pub(crate) async fn resume_consult_inner(
        &self,
        state: PausedState,
        response: ConsultResponse,
        on_stream: Option<StreamSink>,
    ) -> RunResult {
        let PausedState {
            session_id,
            task_id: _,
            turn_number: _,
            mut session_state,
            pending_tool_calls,
            approved_results: _,
            human_request: _,
            task,
            budget_used,
            child_state: _,
            // #140: the pausing leaf's toolset handle, carried through the pause.
            toolset,
        } = state;

        let (text, answered) = match &response {
            ConsultResponse::Answer { text } => (text.clone(), true),
            ConsultResponse::BudgetExhausted { message } => (message.clone(), false),
        };

        // Observability: lightweight consult-resume event.
        if let Some(obs) = self.config.observability.as_ref() {
            // Recover the consult `kind` from the head pending call's args, if
            // present, else fall back to a generic label.
            let kind = pending_tool_calls
                .first()
                .and_then(|c| c.input.get("kind"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let base = SpanBase::new_root(
                SpanId::new(format!("{}-consult-resume", session_id.as_str())),
                session_id.clone(),
                task.id.clone(),
                SpanKind::ContextAssembly,
                Timestamp::now(),
            );
            obs.emit_context(ContextSpan {
                base,
                operation: ContextOperation::ConsultResumed {
                    consult_kind: kind,
                    answered,
                },
                tokens_before: 0,
                tokens_after: 0,
                utilization_before: 0.0,
                utilization_after: 0.0,
            });
        }

        // #140: resume routes the preserved consulting call (and any remaining
        // pending calls) through the pausing leaf's own toolset handle, restoring
        // its scoped per-node catalogue. An empty handle (the default) still falls
        // back to the global catalogue, so pre-#140 blobs behave unchanged.
        let tool_registry = self.effective_tool_registry(&session_id, &toolset);

        // Inject the consult answer as the RESULT of the head pending (consult)
        // call, then dispatch the remaining pending calls in the same batch.
        let mut pending = pending_tool_calls.into_iter();
        if let Some(consult_call) = pending.next() {
            let tr = ToolResult {
                call_id: consult_call.id.clone(),
                output: ToolOutput::Success {
                    content: text,
                    truncated: false,
                },
            };
            self.config
                .context_manager
                .append_tool_result(&mut session_state, &tr)
                .await;
        }
        for call in pending {
            let output =
                self.normalize_sandbox_violation(tool_registry.dispatch(call.clone()).await);
            let tr = ToolResult {
                call_id: call.id,
                output,
            };
            self.config
                .context_manager
                .append_tool_result(&mut session_state, &tr)
                .await;
        }

        // #131: a consult that surfaced from inside a composed tree carries the
        // FULL strategy in `task.loop_strategy` (each combinator's `finish`
        // rewrote the pause's task on the way up). Re-DRIVE that strategy rather
        // than resuming only the worker leaf: the PlanExecute walk resumes its
        // in-progress task from the injected worker session (`consult_resume`
        // seed), so the worker finishes mid-loop, its SelfVerifying evaluator runs,
        // the task is marked Completed, and the remaining ready-set is walked. A
        // BARE worker leaf (depth-1, e.g. a SubagentTool-mediated consult) has no
        // surrounding walk, so it keeps the original leaf-only resume (back-compat).
        if !matches!(task.loop_strategy, LoopStrategy::ReAct(_)) {
            return self
                .drive_strategy_with_resume_seed(
                    task,
                    // Top-level session starts fresh; the worker conversation is
                    // threaded into the in-progress task via the consult seed.
                    SessionState::default(),
                    budget_used,
                    on_stream,
                    None,
                    Some(session_state),
                    // resume_seed carries the continuation signal.
                    false,
                    None,
                )
                .await;
        }

        let max_iterations = match &task.loop_strategy {
            LoopStrategy::ReAct(c) => c.max_iterations(),
            _ => u32::MAX,
        };
        let agent = match self.resolve_worker_agent(&task.loop_strategy) {
            Ok(a) => a,
            Err(error) => {
                return RunResult::Failure {
                    reason: HaltReason::ConfigurationError { error },
                    session_id,
                    usage: AggregateUsage::default(),
                    turns: budget_used.turns,
                    session_state,
                };
            }
        };
        self.run_react(ReactParams::legacy(
            task,
            max_iterations,
            session_state,
            budget_used,
            on_stream,
            agent,
            // legacy wrapper: empty handle ⇒ global-catalogue fallback.
            ToolsetRef(String::new()),
        ))
        .await
    }

    /// Issue #102 auto-persist seam: write the terminal run state to the
    /// [`SessionStore`](crate::storage::SessionStore) when
    /// [`auto_persist_sessions`](HarnessConfig::auto_persist_sessions) is enabled.
    ///
    /// One write per `run()`/`resume()`, at the same terminal seam as the
    /// observability flush. For `Success`/`Failure` a [`PausedState`] is
    /// synthesized carrying the final `session_state` with empty pending fields
    /// (D4); for `WaitingForHuman`/`Escalate` the carried `PausedState` is
    /// persisted (D6 — the cross-process pause case). Storage errors are
    /// swallowed-and-logged (D8): a put failure must never lose the run nor
    /// surface as a [`HaltReason`].
    ///
    /// When disabled (the default), this returns immediately WITHOUT touching the
    /// store — the off-by-default zero-I/O contract (#102).
    pub(crate) async fn auto_persist_terminal(&self, result: &RunResult) {
        if !self.config.auto_persist_sessions {
            return;
        }
        // Build the `(SessionId, PausedState)` to persist for this terminal.
        let (session_id, state) = match result {
            RunResult::Success {
                session_id,
                session_state,
                turns,
                ..
            }
            | RunResult::Failure {
                session_id,
                session_state,
                turns,
                ..
            } => {
                // Synthesize a completed-run PausedState: empty pending fields,
                // no human request, no child — it carries only the final history
                // so a later `get_session(..).session_state` resumes losslessly.
                let state = PausedState {
                    session_id: session_id.clone(),
                    task_id: TaskId::new(session_id.as_str()),
                    turn_number: *turns,
                    session_state: session_state.clone(),
                    pending_tool_calls: vec![],
                    approved_results: vec![],
                    human_request: None,
                    task: Task::new(
                        String::new(),
                        session_id.clone(),
                        LoopStrategy::ReAct(ReactConfig::per_loop(0)),
                    ),
                    budget_used: BudgetSnapshot::default(),
                    child_state: None,
                    // #140: a synthesized completed-run state has empty pending
                    // fields; the handle is behaviorally irrelevant here.
                    toolset: ToolsetRef::default(),
                };
                (session_id.clone(), state)
            }
            // Persist the carried pause state directly (D6).
            RunResult::WaitingForHuman { state, .. } => {
                (state.session_id.clone(), (**state).clone())
            }
            RunResult::Escalate {
                state, session_id, ..
            } => (session_id.clone(), (**state).clone()),
            // Consult (issue #114) is non-terminal — persist the carried pause
            // state directly (same as WaitingForHuman) so a cross-process host
            // can later `resume_consult` it.
            RunResult::Consult {
                state, session_id, ..
            } => (session_id.clone(), (**state).clone()),
        };
        // Swallow-and-log on Err (D8): a storage hiccup must not lose the run.
        let _ = self
            .config
            .storage
            .session()
            .put_session(&session_id, &state)
            .await;
    }

    pub(crate) async fn run_inner(&self, options: HarnessRunOptions) -> RunResult {
        let HarnessRunOptions {
            task,
            on_stream,
            session_state,
            continue_task_list,
        } = options;

        // #124 startup validation: every serializable handle in the task's
        // strategy tree must resolve against the configured ExecutionRegistry,
        // BEFORE the first turn. The legacy single-collaborator fields are gone —
        // resolution is now the SINGLE path, so validation ALWAYS runs (the
        // `!is_empty()` gate is dropped). An unresolved handle is a startup error.
        if let Err(error) = self.config.registry.validate(&task) {
            return RunResult::Failure {
                reason: HaltReason::ConfigurationError { error },
                session_id: task.session_id.clone(),
                usage: AggregateUsage::default(),
                turns: 0,
                session_state: SessionState::default(),
            };
        }

        // Issue #102 auto-load: when enabled AND no explicit session_state was
        // provided AND the strategy seeds incoming state (ReAct / SelfVerifying —
        // Ralph/HillClimbing discard it by design, D7), load the prior session for
        // this `session_id` from the SessionStore so a caller can resume by id.
        // Explicit `with_session_state` always wins (D5). Errors are
        // swallowed-and-logged: a load failure starts fresh (D8).
        let session_state = match session_state {
            Some(explicit) => Some(explicit),
            None if self.config.auto_persist_sessions
                && matches!(
                    task.loop_strategy,
                    LoopStrategy::ReAct(_) | LoopStrategy::SelfVerifying(_)
                ) =>
            {
                match self
                    .config
                    .storage
                    .session()
                    .get_session(&task.session_id)
                    .await
                {
                    Ok(Some(prior)) => Some(prior.session_state),
                    Ok(None) => None,
                    Err(_e) => {
                        // Swallow-and-log: start fresh on a load failure (D8).
                        // (No logging facade is wired into spore-core; the error
                        // is intentionally dropped rather than surfaced as a
                        // HaltReason.)
                        None
                    }
                }
            }
            None => None,
        };

        let mut session_state = session_state.unwrap_or_default();
        let budget_used = BudgetSnapshot::default();

        // #124: the central dispatch `match` is GONE — the only `match` left is
        // the enum→config delegation inside `LoopStrategy::run`. The harness entry
        // collapses to `task.loop_strategy.run(cx).await` via the recursive
        // executor. Strategies that BUILD ON incoming state (ReAct / PlanExecute /
        // SelfVerifying) get the instruction seeded here on the FRESH run (the
        // compaction adapter ignores `task`, so the harness owns prompt delivery);
        // Ralph / HillClimbing re-seed a fresh window internally (D7), so their
        // incoming state is discarded by the config body.
        if matches!(
            task.loop_strategy,
            LoopStrategy::ReAct(_) | LoopStrategy::PlanExecute(_) | LoopStrategy::SelfVerifying(_)
        ) {
            self.config
                .context_manager
                .append_user_message(&mut session_state, &task.instruction)
                .await;
        }
        self.drive_strategy_with_resume_seed(
            task,
            session_state,
            budget_used,
            on_stream,
            None,
            None,
            // Model (A): a fresh `run()` plans fresh unless the caller explicitly
            // opts in to continuing a prior run's durable task list.
            continue_task_list,
            None,
        )
        .await
    }

    /// The recursive-executor entry (#124): build the shared [`ExecutionContext`],
    /// seed the per-run scratch (task / session / budget), drive
    /// `task.loop_strategy.run(cx)`, and translate the outcome back into a
    /// terminal [`RunResult`] (Q5). A non-terminal pause / escalate stashed in
    /// `scratch.terminal_override` propagates VERBATIM (it never collapses into a
    /// `StrategyOutcome`).
    /// Drive a task's strategy through the recursive executor, with an optional
    /// cross-process Continue checkpoint seed
    /// (#129, AC2): `resume_continues = Some((phase, continues_used))` seeds the
    /// FIRST matching budget scope's `continues_used` (via [`BudgetContext::resumed`])
    /// so a `Continue` that spanned a process pause resumes with the correct
    /// continue count instead of a zeroed one. `None` is the fresh-run path.
    ///
    /// `resume_seed = Some(session)` (#131 consult + #138 budget) seeds the FIRST
    /// PlanExecute walk from `session` (the stalled worker conversation). For a
    /// consult resume the consult answer is already injected; for a budget resume
    /// (#138) it is the worker's full post-exhaustion session. The walk re-attaches
    /// it to the single `InProgress` task (execute-phase exhaustion) or, when the
    /// durable task_list has no `InProgress` task, to the PLAN session (#138 AC3
    /// plan-phase exhaustion). `None` on every fresh / non-resume path.
    // The resume/continuation surface is genuinely multi-axis (session, budget,
    // stream, continue-checkpoint, resume-seed, continue-task-list); grouping it
    // into a struct would not improve clarity at the handful of call sites.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn drive_strategy_with_resume_seed(
        &self,
        task: Task,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
        resume_continues: Option<(String, u32)>,
        resume_seed: Option<SessionState>,
        // Model (A): an explicit "continue the durable task list" signal for a
        // SEEDLESS PlanExecute continuation (Ralph window re-entry / `Skip`
        // resolution). The seed-carrying resume paths pass `false` — a
        // `resume_seed` already implies continuation; a fresh run passes `false`.
        continue_task_list: bool,
        // #139: a one-shot FORCED output schema for the FIRST round's leaf (the
        // plan-empty repair re-ask, threaded by `run_plan_subtree`). Consumed by
        // the first round only — mirrors `resume_seed`/`resume_continues`. `None`
        // on every other drive path (byte-identical to before).
        forced_output_schema: Option<serde_json::Value>,
    ) -> RunResult {
        // #129: the BARE-LEAF resolution site is `drive_strategy` (a bare leaf
        // never self-resolves inside its own body — rule 6 — it PROPAGATES a typed
        // `BudgetExhausted` here, the single recovery site for a top-level leaf).
        // When the leaf's CONFIGURED `behavior` resolves to `Continue`, grant the
        // leaf one more round IN-PROCESS (bump `continues_used`, refresh the
        // step cap) and re-drive WITHOUT any serialization (AC3) — looping until
        // the behavior resolves to `Fail`/`Escalate` or the strategy completes.
        // `behavior_for_resolution` carries the resolution chain's mutated state
        // across in-process continues so `max_continues` is honored.
        let session_id = task.session_id.clone();
        let mut task = task;
        let mut session_state = session_state;
        let mut budget_used = budget_used;
        // #stream: read-only now — cloned into `cx.stream` each round, never moved.
        let on_stream = on_stream;
        // The Continue resolution state threaded across in-process rounds: the
        // (possibly fallen-through) behavior + how many continues have been spent.
        let mut behavior_for_resolution: Option<(BudgetExhaustedBehavior, u32)> = None;
        // #129 (AC2): the cross-process checkpoint seed is consumed by the FIRST
        // round only (the resumed node's scope); later in-process rounds carry
        // their continue count via `behavior_for_resolution`.
        let mut resume_continues = resume_continues;
        // #131/#138: the resume seed is consumed by the FIRST round's
        // PlanExecute walk only; later in-process rounds run normally.
        let mut resume_seed = resume_seed;
        // Model (A): the seedless continue signal is likewise consumed by the
        // FIRST round (an in-process `Continue` re-drive is a bare-leaf ReAct and
        // never reads it, so this is belt-and-suspenders against re-asserting it).
        let mut continue_task_list = continue_task_list;
        // #139: the forced output schema is consumed by the FIRST round's leaf
        // only (the plan-empty repair re-ask); later in-process continue rounds
        // run without it.
        let mut forced_output_schema = forced_output_schema;
        // SC-5: auto-grants spent at the BARE-LEAF Escalate site under
        // `EscalationMode::AutoContinue`. The scope is rebuilt each round here
        // (unlike the in-loop combinator scopes, which carry their own
        // `auto_grants_used`), so the counter lives across rounds in this local.
        let mut auto_grants_used: u32 = 0;

        loop {
            let mut cx = ExecutionContext::new(&self.config.registry);
            cx.executor = Some(self);
            // #stream: CLONE (not take) the Arc sink so the binding survives every
            // in-process re-drive round (a `Continue` rebuilds `cx` at the top of
            // this loop and re-installs the same sink). Nested nodes wrap it.
            cx.stream = on_stream.clone();
            cx.scratch.run_session = std::mem::take(&mut session_state);
            cx.scratch.run_budget = budget_used.clone();
            cx.scratch.task = Some(task.clone());
            cx.scratch.resume_continues = resume_continues.take();
            cx.scratch.resume_seed = resume_seed.take();
            cx.scratch.continue_task_list = std::mem::take(&mut continue_task_list);
            // #139: one-shot forced schema for the first round's leaf (the
            // plan-empty repair re-ask); `take()` so continue rounds run without it.
            cx.scratch.forced_output_schema = forced_output_schema.take();

            let outcome = task.loop_strategy.run(&mut cx).await;

            // A pause / escalate propagates verbatim (preserves the HITL / consult /
            // escalation contract through the recursive executor).
            if let Some(terminal) = cx.scratch.terminal_override.take() {
                return terminal;
            }
            match outcome {
                StrategyOutcome::Complete(output) => {
                    return RunResult::Success {
                        output,
                        session_id,
                        usage: cx.usage.clone(),
                        turns: cx.scratch.run_budget.turns,
                        session_state: std::mem::take(&mut cx.scratch.run_session),
                    }
                }
                StrategyOutcome::Failed(error) => {
                    return RunResult::Failure {
                        reason: HaltReason::ConfigurationError { error },
                        session_id,
                        usage: cx.usage.clone(),
                        turns: cx.scratch.run_budget.turns,
                        session_state: std::mem::take(&mut cx.scratch.run_session),
                    }
                }
                StrategyOutcome::BudgetExhausted {
                    policy,
                    behavior,
                    steps_taken,
                    continues_used,
                    phase,
                    partial_output,
                    cause,
                } => {
                    // #129: a BARE LEAF exhaustion propagates here carrying its
                    // CONFIGURED `behavior` (Q1 — the bare-leaf resolution site
                    // honors it; the leaf body never did). Resolve it ONCE: a
                    // `Continue` with continues left re-drives in-process; a spent
                    // `Continue` falls through to `Fail`/`Escalate`. Carry the
                    // resolution chain's state across rounds so `max_continues` is
                    // respected. (A combinator under `SurfaceToHuman` never reaches
                    // this arm — it sets `terminal_override`, returned above.)
                    let turns = if steps_taken > 0 {
                        steps_taken
                    } else {
                        cx.scratch.run_budget.turns
                    };
                    // #137: the terminal `HaltReason` for a Fail / Escalate→Fail
                    // resolution depends on the CAUSE — a genuine budget
                    // exhaustion reports `BudgetExceeded`; an error-loop break
                    // reports `ToolErrorLoop`. (A granted `Continue` re-drives
                    // the window, whose loop-local error counter starts fresh —
                    // parallel to how `consume_continue` resets `steps_taken`.)
                    let terminal_reason = match &cause {
                        ExhaustionCause::Budget => HaltReason::BudgetExceeded {
                            limit_type: BudgetLimitType::Turns,
                        },
                        ExhaustionCause::ToolErrorLoop {
                            tool,
                            consecutive_errors,
                        } => HaltReason::ToolErrorLoop {
                            tool: tool.clone(),
                            consecutive_errors: *consecutive_errors,
                        },
                    };
                    // Reconstruct the resolution scope: the FIRST round uses the
                    // leaf's propagated behavior + continues_used; later in-process
                    // rounds reuse the threaded (possibly fallen-through) state.
                    let (resolve_behavior, resolve_continues) = behavior_for_resolution
                        .take()
                        .unwrap_or((behavior, continues_used));
                    let mut scope = BudgetContext::resumed(
                        policy.clone(),
                        resolve_behavior,
                        phase.clone(),
                        resolve_continues,
                    );
                    let resolution = scope.resolve_exhausted();
                    match resolution {
                        ExhaustedResolution::Continue => {
                            // In-process continue (AC3: NO serialization). Refresh
                            // the leaf's step cap and re-enter the loop carrying the
                            // post-run session so the conversation context survives
                            // (Continue PRESERVES context — AC4). The granted cap is
                            // `steps_taken + policy.value` so the leaf gets a fresh
                            // window after the checkpoint.
                            let granted =
                                steps_taken.saturating_add(policy.allowance_value().unwrap_or(1));
                            grant_task_budget(&mut task, granted);
                            session_state = std::mem::take(&mut cx.scratch.run_session);
                            budget_used = cx.scratch.run_budget.clone();
                            // #stream: no reclaim needed — `on_stream` (Arc) is
                            // cloned, never moved, into `cx.stream` each round.
                            // Thread the resolution chain's post-continue state so a
                            // subsequent exhaustion sees the bumped `continues_used`.
                            behavior_for_resolution =
                                Some((scope.behavior.clone(), scope.continues_used));
                            continue;
                        }
                        ExhaustedResolution::Escalate => {
                            // SC-5: AutoContinue auto-grants `steps_per_grant` more
                            // steps and re-drives IN-PROCESS (mirroring the
                            // `Continue` arm above), up to `max_grants` times,
                            // firing `on_grant` per grant. This is the bare-leaf /
                            // top-level keep-working-but-capped site — where
                            // consumers otherwise hand-roll a drive loop. Once the
                            // grants are spent it falls through to the
                            // surface/abort handling below (the `Autonomous`
                            // terminal).
                            if let EscalationMode::AutoContinue {
                                max_grants,
                                steps_per_grant,
                                on_grant,
                            } = &self.config.escalation_mode
                            {
                                if auto_grants_used < *max_grants {
                                    auto_grants_used += 1;
                                    if let Some(cb) = on_grant {
                                        cb(AutoGrantInfo {
                                            grant_number: auto_grants_used,
                                            steps_granted: *steps_per_grant,
                                            phase: phase.clone(),
                                        });
                                    }
                                    let granted = steps_taken.saturating_add(*steps_per_grant);
                                    grant_task_budget(&mut task, granted);
                                    session_state = std::mem::take(&mut cx.scratch.run_session);
                                    budget_used = cx.scratch.run_budget.clone();
                                    continue;
                                }
                            }
                            if matches!(self.config.escalation_mode, EscalationMode::SurfaceToHuman)
                            {
                                let err = BudgetExhausted {
                                    policy,
                                    behavior: scope.behavior.clone(),
                                    steps_taken,
                                    continues_used: scope.continues_used,
                                    phase,
                                };
                                let actions = leaf_escalation_actions(&err);
                                // #138 AC2-a: carry the FULL stalled leaf session
                                // (it propagated into scratch with its conversation)
                                // and the leaf's own toolset handle (#140/AC4-a).
                                let worker_session = std::mem::take(&mut cx.scratch.run_session);
                                let worker_toolset = worker_toolset_of(&task.loop_strategy);
                                return promote_budget_exhausted_to_human(
                                    &err,
                                    partial_output,
                                    actions,
                                    session_id,
                                    task,
                                    cx.scratch.run_budget.clone(),
                                    turns,
                                    worker_session,
                                    worker_toolset,
                                );
                            }
                            return RunResult::Failure {
                                // #137: ToolErrorLoop vs BudgetExceeded per cause.
                                reason: terminal_reason,
                                session_id,
                                usage: cx.usage.clone(),
                                turns,
                                session_state: SessionState {
                                    messages: partial_output
                                        .map(|t| {
                                            vec![Message {
                                                role: crate::model::Role::Assistant,
                                                content: crate::model::Content::Text { text: t },
                                            }]
                                        })
                                        .unwrap_or_default(),
                                    ..Default::default()
                                },
                            };
                        }
                        ExhaustedResolution::Fail => {
                            // Fail contract: the partial is DISCARDED.
                            return RunResult::Failure {
                                // #137: ToolErrorLoop vs BudgetExceeded per cause.
                                reason: terminal_reason,
                                session_id,
                                usage: cx.usage.clone(),
                                turns,
                                session_state: SessionState::default(),
                            };
                        }
                    }
                }
            }
        }
    }

    /// Emit a [`ToolCallSpan`] for a tool call dispatched on the RESUME path
    /// (human-approved gated calls: writes, edits, bash, gated reads). The inline
    /// turn loop emits tool spans at a single site in `react.rs`; gated calls run
    /// HERE on resume instead, so without this they are ABSENT from the trace —
    /// making observability under-count tool use (doctor / judge / eval go blind
    /// to the very writes that do the work). Mirrors the inline emit: same
    /// content-capture guard, parented to the turn span via the deterministic
    /// `{session}-turn-{turn}` id so resume tool spans nest under their turn.
    #[allow(clippy::too_many_arguments)]
    fn emit_resumed_tool_call(
        &self,
        session_id: &SessionId,
        task: &Task,
        turn_number: u32,
        started_at: Timestamp,
        elapsed_ms: u64,
        call: &ToolCall,
        output: &ToolOutput,
    ) {
        let Some(obs) = self.config.observability.as_ref() else {
            return;
        };
        let cc = &self.config.content_capture;
        let (arguments, result) = if cc.enabled {
            let args = Self::capture_tool_call_args(call, cc.max_field_len);
            let result = match output {
                ToolOutput::Success { content, .. } => {
                    let (content, t) = truncate_field(content, cc.max_field_len);
                    Some(ToolResultContent { content, truncated: t })
                }
                ToolOutput::Error { message, .. } => {
                    let (content, t) = truncate_field(message, cc.max_field_len);
                    Some(ToolResultContent { content, truncated: t })
                }
                _ => None,
            };
            (Some(args), result)
        } else {
            (None, None)
        };
        let (output_size_bytes, truncated) = match output {
            ToolOutput::Success { content, truncated } => (content.len(), *truncated),
            ToolOutput::Error { message, .. } => (message.len(), false),
            _ => (0, false),
        };
        // Reconstruct the turn's SpanBase only to carry its deterministic id as
        // the parent; this throwaway base is never emitted.
        let parent = SpanBase::new_root(
            SpanId::new(format!("{}-turn-{}", session_id.as_str(), turn_number)),
            session_id.clone(),
            task.id.clone(),
            SpanKind::Turn,
            started_at.clone(),
        );
        let mut base = SpanBase::new_child(
            SpanId::new(format!("{}-tool-{}", session_id.as_str(), call.id)),
            &parent,
            SpanKind::ToolCall,
            started_at,
        );
        let status = if matches!(output, ToolOutput::Error { .. }) {
            SpanStatus::Error { message: "tool returned a recoverable error".into() }
        } else {
            SpanStatus::Ok
        };
        base.finish(Timestamp::now(), status, elapsed_ms);
        obs.emit_tool_call(ToolCallSpan {
            base,
            tool_name: call.name.clone(),
            call_id: call.id.clone(),
            parameters_size_bytes: call.input.to_string().len(),
            output_size_bytes,
            truncated,
            sandbox_mode: String::new(),
            sandbox_violations: Vec::new(),
            arguments,
            result,
        });
    }

    pub(crate) async fn resume_inner(
        &self,
        state: PausedState,
        response: HumanResponse,
        on_stream: Option<StreamSink>,
    ) -> RunResult {
        let PausedState {
            session_id,
            task_id: _,
            turn_number,
            mut session_state,
            pending_tool_calls,
            approved_results: _approved_results,
            human_request: hr,
            task,
            budget_used,
            child_state,
            // #140: the pausing leaf's toolset handle, carried through the pause.
            toolset,
        } = state;

        // Budget-escalation resume (#130/#129): if this pause came from
        // `HumanRequest::BudgetExhausted`, map the operator's `EscalationAction`
        // BEFORE the generic `match response` (mirrors the Clarification branch).
        // The node's `BudgetContext` is reconstructed from the request fields:
        // `steps_taken` raises the granted cap, and `continues_used` (#129, AC2)
        // is the SOLE `BudgetContext` field that rides a process pause — it is
        // seeded back into the rebuilt scope so a `Continue` spanning the pause
        // resumes with the correct continue count (it cannot exceed
        // `max_continues`). Q3: `continues_used` rides the REQUEST payload, NOT a
        // new serialized `BudgetContext`/`PausedState` field.
        if let Some(HumanRequest::BudgetExhausted {
            steps_taken,
            continues_used,
            phase,
            ..
        }) = &hr
        {
            let steps_taken = *steps_taken;
            let resume_continues = (phase.clone(), *continues_used);
            match &response {
                // Grant `steps` ADDITIONAL allowance and re-enter the loop from the
                // restored checkpoint. The strategy tree is rebuilt with the node's
                // budget policy raised to `steps_taken + steps` so the restored
                // scope has room for `steps` more steps, and the resumed scope's
                // `continues_used` is seeded from the request (#129, AC2).
                HumanResponse::Escalate {
                    action: EscalationAction::ContinueWithBudget { steps },
                } => {
                    let granted = steps_taken.saturating_add(*steps);
                    let mut resumed_task = task.clone();
                    grant_task_budget(&mut resumed_task, granted);
                    // #138 AC2-b: for a COMPOSED task (PlanExecute etc.) thread the
                    // carried worker session through the phase-agnostic resume seed
                    // and start the TOP-LEVEL session EMPTY — exactly mirroring
                    // `resume_consult_inner`. The PlanExecute walk re-attaches the
                    // seed to the single `InProgress` task (execute-phase
                    // exhaustion) via the InProgress→Pending→complete machinery, or
                    // to the PLAN session (AC3 plan-phase exhaustion) — so the
                    // stalled worker continues mid-loop instead of re-planning. A
                    // BARE leaf has no surrounding walk, so it resumes from
                    // `session_state` directly (the seed has nowhere to attach).
                    let (top_session, resume_seed) =
                        if matches!(resumed_task.loop_strategy, LoopStrategy::ReAct(_)) {
                            (session_state, None)
                        } else {
                            (SessionState::default(), Some(session_state))
                        };
                    return self
                        .drive_strategy_with_resume_seed(
                            resumed_task,
                            top_session,
                            budget_used,
                            on_stream,
                            Some(resume_continues),
                            resume_seed,
                            // resume_seed (when present) carries the continuation
                            // signal; a bare ReAct leaf needs none.
                            false,
                            None,
                        )
                        .await;
                }
                // Skip: the node is marked skipped and the outer loop advances. For
                // a combinator (PlanExecute) the per-task partial already recorded
                // the blocked task and persisted the list; re-entering the loop from
                // the checkpoint advances to the remaining ready tasks. For a leaf
                // there is no sibling, so a skip resolves to a clean (empty)
                // Success carrying whatever partial history was captured.
                HumanResponse::Escalate {
                    action: EscalationAction::Skip,
                } => {
                    if matches!(task.loop_strategy, LoopStrategy::PlanExecute(_)) {
                        // Model (A): a Skip carries NO resume_seed (the task is
                        // abandoned, not continued), so it must explicitly SIGNAL
                        // continuation — otherwise PlanExecute would now treat the
                        // re-entry as a fresh run, clear the durable list, and
                        // re-plan instead of walking past the skipped task to the
                        // remaining ready ones.
                        return self
                            .drive_strategy_with_resume_seed(
                                task,
                                session_state,
                                budget_used,
                                on_stream,
                                None,
                                None,
                                true,
                                None,
                            )
                            .await;
                    }
                    return RunResult::Success {
                        output: String::new(),
                        session_id,
                        usage: AggregateUsage::default(),
                        turns: turn_number,
                        session_state,
                    };
                }
                // Fail: abort the node and propagate `BudgetExceeded`; the partial
                // is discarded (the `Fail` resolution contract).
                HumanResponse::Escalate {
                    action: EscalationAction::Fail,
                } => {
                    return RunResult::Failure {
                        reason: HaltReason::BudgetExceeded {
                            limit_type: BudgetLimitType::Turns,
                        },
                        session_id,
                        usage: AggregateUsage::default(),
                        turns: turn_number,
                        session_state: SessionState::default(),
                    };
                }
                // A non-`Escalate` response to a budget pause is out of contract;
                // treat it conservatively as a `Fail` (abort + propagate).
                _ => {
                    return RunResult::Failure {
                        reason: HaltReason::BudgetExceeded {
                            limit_type: BudgetLimitType::Turns,
                        },
                        session_id,
                        usage: AggregateUsage::default(),
                        turns: turn_number,
                        session_state: SessionState::default(),
                    };
                }
            }
        }

        // Resolve the effective tool registry for this resumed session — bridges
        // catalogue tools the same way the turn loop does, so pending tool calls
        // dispatched during resume thread the run's storage + sandbox. #140:
        // resume now routes through the pausing leaf's own toolset handle,
        // restoring its scoped per-node catalogue. An empty handle (the default)
        // still falls back to the global catalogue, so pre-#140 blobs and root
        // pauses behave unchanged. NOTE: the budget-escalation branch above
        // returned early via `drive_strategy_with_resume_seed`, which re-resolves
        // per-leaf toolsets during the re-drive — so this line is only reached by
        // the Clarification / direct-resume paths whose pending calls need the
        // carried handle.
        let tool_registry = self.effective_tool_registry(&session_id, &toolset);

        // Clarification resume (issue #81, Q4b): if this pause came from
        // `ToolOutput::AwaitingClarification`, the human's answer is injected as
        // the tool RESULT for the clarifying call (the head of
        // `pending_tool_calls`) — NOT appended as a free-standing user message.
        // Any remaining pending calls after the clarifying one are then
        // dispatched normally before the loop resumes.
        if matches!(hr, Some(HumanRequest::Clarification { .. })) {
            if let HumanResponse::Answer { text }
            | HumanResponse::ApproveWithFeedback { feedback: text } = &response
            {
                let mut pending = pending_tool_calls.into_iter();
                if let Some(clarifying_call) = pending.next() {
                    let tr = ToolResult {
                        call_id: clarifying_call.id.clone(),
                        output: ToolOutput::Success {
                            content: text.clone(),
                            truncated: false,
                        },
                    };
                    self.config
                        .context_manager
                        .append_tool_result(&mut session_state, &tr)
                        .await;
                }
                // Dispatch any remaining pending calls from the same batch.
                for call in pending {
                    let tool_started_at = Timestamp::now();
                    let tool_clock = std::time::Instant::now();
                    let output = self
                        .normalize_sandbox_violation(tool_registry.dispatch(call.clone()).await);
                    self.emit_resumed_tool_call(
                        &session_id,
                        &task,
                        turn_number,
                        tool_started_at,
                        tool_clock.elapsed().as_millis() as u64,
                        &call,
                        &output,
                    );
                    let tr = ToolResult {
                        call_id: call.id,
                        output,
                    };
                    self.config
                        .context_manager
                        .append_tool_result(&mut session_state, &tr)
                        .await;
                }
                // SC-BUG-1: a clarification that surfaced from inside a composed
                // tree carries the FULL strategy in `task.loop_strategy` (each
                // combinator's `finish` rewrote the pause's task on the way up).
                // Re-DRIVE that strategy from the answered worker session — exactly
                // as the consult resume does — so the SelfVerifying evaluate phase /
                // PlanExecute walk runs instead of only the bare worker leaf. A
                // BARE ReAct leaf has no surrounding frame, so it keeps the
                // leaf-only resume below (back-compat).
                if !matches!(task.loop_strategy, LoopStrategy::ReAct(_)) {
                    return self
                        .drive_strategy_with_resume_seed(
                            task,
                            // Top-level session starts fresh; the answered worker
                            // conversation threads in as the resume seed.
                            SessionState::default(),
                            budget_used,
                            on_stream,
                            None,
                            Some(session_state),
                            // resume_seed carries the continuation signal.
                            false,
                            None,
                        )
                        .await;
                }
                let max_iterations = match &task.loop_strategy {
                    LoopStrategy::ReAct(c) => c.max_iterations(),
                    _ => u32::MAX,
                };
                let agent = match self.resolve_worker_agent(&task.loop_strategy) {
                    Ok(a) => a,
                    Err(error) => {
                        return RunResult::Failure {
                            reason: HaltReason::ConfigurationError { error },
                            session_id,
                            usage: AggregateUsage::default(),
                            turns: budget_used.turns,
                            session_state,
                        };
                    }
                };
                return self
                    .run_react(ReactParams::legacy(
                        task,
                        max_iterations,
                        session_state,
                        budget_used,
                        on_stream,
                        agent,
                        // legacy wrapper: empty handle ⇒ global-catalogue fallback.
                        ToolsetRef(String::new()),
                    ))
                    .await;
            }
        }

        // Subagent depth: if there's a child, route the response through it.
        // The actual child harness is owned by the caller-installed
        // `SubagentTool`; without a tool registry hook for `resume_child`,
        // the harness surfaces a tool result indicating the child's outcome
        // is the caller's responsibility. Wiring lands with #4/#5.
        if child_state.is_some() {
            // Append a synthetic tool result and continue the parent loop.
            // The full child.resume() dispatch lives with SubagentTool (#5).
        }

        match response {
            HumanResponse::Halt => {
                return RunResult::Failure {
                    reason: HaltReason::HumanHalted,
                    session_id,
                    usage: AggregateUsage::default(),
                    turns: turn_number,
                    session_state,
                };
            }
            HumanResponse::Deny { reason } => {
                // Append failed tool results for each pending call, continue loop.
                for call in &pending_tool_calls {
                    let tr = ToolResult {
                        call_id: call.id.clone(),
                        output: ToolOutput::Error {
                            message: reason.clone(),
                            recoverable: true,
                        },
                    };
                    self.config
                        .context_manager
                        .append_tool_result(&mut session_state, &tr)
                        .await;
                }
            }
            HumanResponse::Reject { reason } => {
                self.config
                    .context_manager
                    .append_user_message(&mut session_state, &reason)
                    .await;
            }
            HumanResponse::Answer { text }
            | HumanResponse::ApproveWithFeedback { feedback: text } => {
                self.config
                    .context_manager
                    .append_user_message(&mut session_state, &text)
                    .await;
            }
            HumanResponse::Allow => {
                // Dispatch remaining pending calls before resuming the loop.
                // Mirror the turn-loop's stream events (ToolCall before dispatch,
                // ToolResult after) so an approved tool's execution and output
                // render in the UI sink. Without these emits the resume path runs
                // silently, leaving every human-gated tool (bash, writes, gated
                // reads) with an approval line but no visible result.
                for call in pending_tool_calls {
                    Self::emit(
                        &on_stream,
                        StreamEvent::ToolCall {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            args: call.input.clone(),
                            node: None,
                        },
                    );
                    let tool_started_at = Timestamp::now();
                    let tool_clock = std::time::Instant::now();
                    let output = self
                        .normalize_sandbox_violation(tool_registry.dispatch(call.clone()).await);
                    // Observability span — gated tools dispatch on the resume path;
                    // emit here too, else they are missing from the trace (the
                    // single inline emit in `react.rs` never runs for them).
                    self.emit_resumed_tool_call(
                        &session_id,
                        &task,
                        turn_number,
                        tool_started_at,
                        tool_clock.elapsed().as_millis() as u64,
                        &call,
                        &output,
                    );
                    Self::emit(
                        &on_stream,
                        StreamEvent::ToolResult {
                            call_id: call.id.clone(),
                            is_error: matches!(output, ToolOutput::Error { .. }),
                            content: match &output {
                                ToolOutput::Success { content, .. } => content.clone(),
                                ToolOutput::Error { message, .. } => message.clone(),
                                _ => String::new(),
                            },
                            revised: false,
                            node: None,
                        },
                    );
                    let tr = ToolResult {
                        call_id: call.id,
                        output,
                    };
                    self.config
                        .context_manager
                        .append_tool_result(&mut session_state, &tr)
                        .await;
                }
            }
            HumanResponse::AllowWithModification { calls } => {
                // Same stream-event mirroring as `Allow` (see above) so the
                // modified-and-approved calls render their output in the UI sink.
                for call in calls {
                    Self::emit(
                        &on_stream,
                        StreamEvent::ToolCall {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            args: call.input.clone(),
                            node: None,
                        },
                    );
                    let tool_started_at = Timestamp::now();
                    let tool_clock = std::time::Instant::now();
                    let output = self
                        .normalize_sandbox_violation(tool_registry.dispatch(call.clone()).await);
                    // Observability span — gated tools dispatch on the resume path;
                    // emit here too, else they are missing from the trace (the
                    // single inline emit in `react.rs` never runs for them).
                    self.emit_resumed_tool_call(
                        &session_id,
                        &task,
                        turn_number,
                        tool_started_at,
                        tool_clock.elapsed().as_millis() as u64,
                        &call,
                        &output,
                    );
                    Self::emit(
                        &on_stream,
                        StreamEvent::ToolResult {
                            call_id: call.id.clone(),
                            is_error: matches!(output, ToolOutput::Error { .. }),
                            content: match &output {
                                ToolOutput::Success { content, .. } => content.clone(),
                                ToolOutput::Error { message, .. } => message.clone(),
                                _ => String::new(),
                            },
                            revised: false,
                            node: None,
                        },
                    );
                    let tr = ToolResult {
                        call_id: call.id,
                        output,
                    };
                    self.config
                        .context_manager
                        .append_tool_result(&mut session_state, &tr)
                        .await;
                }
            }
            // An `Escalate` response delivered to a non-budget pause is out of
            // contract (#130): the budget-escalation resume is handled by the
            // dedicated `HumanRequest::BudgetExhausted` branch ABOVE, which
            // returns early. Reaching here means the operator sent an
            // `EscalationAction` for a `ToolApproval`/`Review`/`None` pause —
            // halt cleanly rather than mis-resuming the loop.
            HumanResponse::Escalate { .. } => {
                return RunResult::Failure {
                    reason: HaltReason::HumanHalted,
                    session_id,
                    usage: AggregateUsage::default(),
                    turns: turn_number,
                    session_state,
                };
            }
        }

        // SC-BUG-1: an Allow / Deny / Answer / Reject resume that surfaced from
        // inside a composed tree carries the FULL strategy in `task.loop_strategy`
        // (each combinator's `finish` rewrote the pause's task on the way up). The
        // human response has already been applied to `session_state` above (the
        // approved calls dispatched, or the deny/answer message appended). Re-DRIVE
        // the whole strategy from that mutated worker session — mirroring the
        // consult resume — so the SelfVerifying evaluate phase (the looper's
        // eval-frame reviewer) / PlanExecute walk re-runs instead of degrading to a
        // plain executor. A BARE ReAct leaf keeps the leaf-only resume below.
        if !matches!(task.loop_strategy, LoopStrategy::ReAct(_)) {
            return self
                .drive_strategy_with_resume_seed(
                    task,
                    SessionState::default(),
                    budget_used,
                    on_stream,
                    None,
                    Some(session_state),
                    // resume_seed carries the continuation signal.
                    false,
                    None,
                )
                .await;
        }

        // Resume the ReAct loop from where we left off.
        let max_iterations = match &task.loop_strategy {
            LoopStrategy::ReAct(c) => c.max_iterations(),
            _ => u32::MAX,
        };
        let agent = match self.resolve_worker_agent(&task.loop_strategy) {
            Ok(a) => a,
            Err(error) => {
                return RunResult::Failure {
                    reason: HaltReason::ConfigurationError { error },
                    session_id,
                    usage: AggregateUsage::default(),
                    turns: budget_used.turns,
                    session_state,
                };
            }
        };
        self.run_react(ReactParams::legacy(
            task,
            max_iterations,
            session_state,
            budget_used,
            on_stream,
            agent,
            // legacy wrapper: empty handle ⇒ global-catalogue fallback.
            ToolsetRef(String::new()),
        ))
        .await
    }
}
