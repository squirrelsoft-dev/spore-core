//! StandardHarness — context compaction driver (Phase 6.1 impl split). Pure move.

use super::*;

impl StandardHarness {
    /// Run the post-compaction verify→retry→warn loop (issue #46/#29).
    ///
    /// Drives one compaction turn through the agent, verifies the summary,
    /// and either accepts it, retries with the missing items injected, or —
    /// after `max_compaction_attempts` — emits a warn event and accepts the
    /// summary anyway. A blocked compaction is worse than an imperfect one, so
    /// this method NEVER returns an error or halts the run; the worst case is
    /// an accepted-anyway summary plus one warn span.
    ///
    /// Token usage from compaction turns folds into the run-level
    /// [`AggregateUsage`]; each compaction turn that produces a summary is
    /// surfaced as a `Compaction` [`ContextSpan`]. The
    /// `compaction_verification_failures` metric is derived from the emitted
    /// [`WarnSpan`].
    pub(crate) async fn run_compaction(
        &self,
        session_state: &mut SessionState,
        session_id: &SessionId,
        task_id: &TaskId,
        span_seq: &mut u64,
        usage: &mut AggregateUsage,
        // #124: the compaction summary turn runs on the SAME resolved worker
        // agent driving the ReAct window (no `config.agent`).
        agent: &Arc<dyn Agent>,
    ) {
        let Some(mut turn) = self
            .config
            .context_manager
            .prepare_compaction_turn(session_state)
        else {
            // Nothing to compact (e.g. history shorter than preserve window).
            return;
        };
        let tokens_before = turn.verification_state.token_budget_used;
        let max_attempts = self.config.max_compaction_attempts.max(1);
        let mut attempt: u32 = 0;

        loop {
            attempt += 1;
            // Run one compaction turn through the agent to produce a summary.
            let result = agent.turn(turn.context.clone()).await;
            let summary = match result {
                TurnResult::FinalResponse {
                    content, usage: u, ..
                } => {
                    usage.add_turn(&u);
                    content
                }
                TurnResult::ToolCallRequested { usage: u, .. } => {
                    // A compaction turn is expected to yield a summary, not a
                    // tool call. Treat the (empty) response as the summary so
                    // verification can run and the loop terminates predictably.
                    usage.add_turn(&u);
                    String::new()
                }
                TurnResult::Error { usage: u, .. } => {
                    if let Some(u) = u.as_ref() {
                        usage.add_turn(u);
                    }
                    String::new()
                }
            };

            let verification = self.config.compaction_verifier.verify(
                &summary,
                &turn.preserve_hints,
                &turn.verification_state,
            );

            if verification.passed {
                self.accept_compaction(
                    session_state,
                    summary,
                    turn.messages_removed,
                    &turn.dropped_messages,
                    tokens_before,
                    session_id,
                    task_id,
                    span_seq,
                );
                return;
            }

            if attempt < max_attempts {
                // Inject the missing items and retry.
                self.config
                    .context_manager
                    .inject_missing_items(&mut turn.context, &verification.missing_items);
                continue;
            }

            // Exhausted attempts: warn, then accept anyway.
            if let Some(obs) = self.config.observability.as_ref() {
                let base = SpanBase::new_root(
                    SpanId::new(format!("{}-warn-{}", session_id.as_str(), *span_seq)),
                    session_id.clone(),
                    task_id.clone(),
                    SpanKind::Warn,
                    Timestamp::now(),
                );
                obs.emit_warn(WarnSpan::new(
                    base,
                    WarnEvent::CompactionVerificationFailed {
                        missing_items: verification.missing_items.clone(),
                        accepted_anyway: true,
                    },
                ));
                *span_seq += 1;
            }
            self.accept_compaction(
                session_state,
                summary,
                turn.messages_removed,
                &turn.dropped_messages,
                tokens_before,
                session_id,
                task_id,
                span_seq,
            );
            return;
        }
    }

    /// Apply an accepted summary and emit the `Compaction` context span.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accept_compaction(
        &self,
        session_state: &mut SessionState,
        summary: String,
        messages_removed: u32,
        dropped: &[Message],
        tokens_before: u32,
        session_id: &SessionId,
        task_id: &TaskId,
        span_seq: &mut u64,
    ) {
        self.config
            .context_manager
            .apply_compaction(session_state, summary, dropped);

        // Real token accounting (issue #57): read the post-compaction budget the
        // manager tracks. Managers that do not track tokens report `None`, in
        // which case we fall back to the pre-compaction value (the old behavior).
        let tokens_after = self
            .config
            .context_manager
            .token_budget_used(session_state)
            .unwrap_or(tokens_before);
        let tokens_reclaimed = tokens_before.saturating_sub(tokens_after);

        if let Some(obs) = self.config.observability.as_ref() {
            let base = SpanBase::new_root(
                SpanId::new(format!("{}-compaction-{}", session_id.as_str(), *span_seq)),
                session_id.clone(),
                task_id.clone(),
                SpanKind::Compaction,
                Timestamp::now(),
            );
            obs.emit_context(crate::observability::ContextSpan {
                base,
                operation: crate::observability::ContextOperation::Compaction {
                    messages_removed,
                    tokens_reclaimed,
                },
                tokens_before,
                tokens_after,
                utilization_before: 0.0,
                utilization_after: 0.0,
            });
            *span_seq += 1;
        }
    }
}
