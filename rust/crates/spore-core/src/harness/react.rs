//! ReAct strategy: the `StandardHarness` turn-loop primitives (`run_react` /
//! `run_react_inner`) plus `impl RunStrategy for ReactConfig` (the leaf body).

use super::*;

impl StandardHarness {
    /// Dispatch a tool call, applying the configured [`ToolCallRepair`] on a
    /// recoverable error: coerce the arguments and re-dispatch up to
    /// `max_repair_attempts`, enriching the error (schema + hint) when repair
    /// gives up. Returns the call ACTUALLY dispatched (the repaired variant when
    /// repair fired, so the caller can stamp spans/history from it) and its final
    /// output.
    ///
    /// Shared by the inline ReAct turn-loop AND the resume path (`run.rs`), so
    /// human-GATED tool calls (bash/write/edit) — which dispatch only after
    /// approval on the resume path — get the same argument repair as auto-approved
    /// calls. Without it a malformed gated arg (e.g. `timeout: "60"` where a `u64`
    /// is expected) is never coerced and surfaces as a hard tool failure the model
    /// has to retry by hand.
    pub(crate) async fn dispatch_with_repair(
        &self,
        tool_registry: &dyn ToolRegistry,
        call: ToolCall,
    ) -> (ToolCall, ToolOutput) {
        let mut effective_call = call;
        let mut output = tool_registry.dispatch(effective_call.clone()).await;
        if let Some(repair) = self.config.tool_call_repair.as_ref() {
            let mut attempts_left = self.config.max_repair_attempts;
            while attempts_left > 0 {
                let ToolOutput::Error { ref message, recoverable: true } = output else {
                    break;
                };
                let schema =
                    tool_registry.schemas().into_iter().find(|s| s.name == effective_call.name);
                let Some(repaired_call) = repair.repair(&effective_call, message, schema.as_ref())
                else {
                    // Repair gave up. Enrich with the schema+hint ONLY for a
                    // parameter-shaped error (one the hint can actually help); a
                    // plain command/runtime failure (e.g. a non-zero `cargo test`
                    // exit) is left raw, so the hint doesn't mislead the model — or
                    // an offline judge — into seeing a "schema violation" that
                    // never happened.
                    if Self::is_param_error(message) {
                        output = Self::enrich_tool_error(message, schema.as_ref());
                    }
                    break;
                };
                attempts_left -= 1;
                effective_call = repaired_call;
                output = tool_registry.dispatch(effective_call.clone()).await;
                if !matches!(output, ToolOutput::Error { recoverable: true, .. }) {
                    break;
                }
                if attempts_left == 0 {
                    if let ToolOutput::Error { message: ref m, recoverable: true } = output {
                        if Self::is_param_error(m) {
                            let schema = tool_registry
                                .schemas()
                                .into_iter()
                                .find(|s| s.name == effective_call.name);
                            output = Self::enrich_tool_error(m, schema.as_ref());
                        }
                    }
                }
            }
        }
        (effective_call, output)
    }

    /// Whether a recoverable tool-error message is a PARAMETER problem (a
    /// malformed call) rather than a command/runtime failure (e.g. a non-zero
    /// `cargo test` exit). Only parameter-shaped errors benefit from the
    /// schema+hint enrichment; enriching a command failure misleads the model —
    /// and offline judges — into seeing a "schema violation" that did not happen.
    pub(crate) fn is_param_error(message: &str) -> bool {
        let m = message.to_ascii_lowercase();
        m.contains("invalid parameters")
            || m.contains("invalid type")
            || m.contains("missing required field")
            || m.contains("missing field")
            || m.contains("schema validation failed")
            || m.contains("unknown field")
            || m.contains("unknown variant")
    }

    /// Legacy single-shot / DAG execute-step / SelfVerifying-evaluate entry to a
    /// ReAct window. Wraps [`run_react_inner`](Self::run_react_inner) with the
    /// pre-#139 / pre-SC-10 defaults (built via [`ReactParams::legacy`]) and adds
    /// the terminal observability finalization the inner loop deliberately omits.
    pub(crate) async fn run_react(&self, params: ReactParams) -> RunResult {
        let result = self.run_react_inner(params).await;
        match &result {
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
            RunResult::WaitingForHuman { .. } => {}
            // Consult (issue #114) is NOT terminal — like WaitingForHuman, the
            // worker is paused awaiting a resume. Do not finalize observability.
            RunResult::Consult { .. } => {}
            // Escalation (issue #80) IS a terminal outcome — the harness has
            // run to a clean stop and handed a signal up. Finalize
            // observability with the dedicated `Escalated` outcome (NOT
            // `Partial`). Contrast WaitingForHuman, which is NOT terminal.
            RunResult::Escalate { session_id, .. } => {
                self.finalize_observability(session_id, SessionOutcome::Escalated)
                    .await;
            }
        }
        result
    }

    /// The harness-loop tool registry to use for a run keyed by `session_id`.
    ///
    /// When catalogue tools were added via [`HarnessBuilder::tool`] / `tools`,
    /// this bridges the folded [`StandardToolRegistry`] through
    /// [`RealToolRegistry`](crate::tool_registry::RealToolRegistry) — built
    /// fresh per run so the run's `SessionId` + storage thread into every tool
    /// dispatch. Otherwise it returns the injected
    /// [`tool_registry`](HarnessConfig::tool_registry) seam unchanged.
    /// Issue 2 (per-node toolset scoping): `toolset` is the resolving leaf's
    /// `ToolsetRef` handle. When it is NON-EMPTY and a per-key catalogue was
    /// registered via [`HarnessBuilder::toolset_tools`], THAT catalogue is
    /// bridged (so the node dispatches ONLY its own tools — strict scoping).
    /// Otherwise (empty handle, or non-empty handle with no per-key catalogue)
    /// the existing global-catalogue / `tool_registry` seam fallback applies, so
    /// examples 01–11 that use `.tools()` keep working byte-for-byte.
    pub(crate) fn effective_tool_registry(
        &self,
        session_id: &SessionId,
        toolset: &ToolsetRef,
    ) -> Arc<dyn ToolRegistry> {
        // Reserved handle: advertise NO tools (plan-empty re-ask). Intercepted
        // before the catalogue lookup so it never falls through to the full
        // registry. Dispatch is unreachable — the model is offered nothing.
        if toolset.0 == NO_TOOLS_TOOLSET {
            return Arc::new(EmptyToolRegistry);
        }
        // Strict per-node scoping: a non-empty handle with its own catalogue.
        if !toolset.0.is_empty() {
            if let Some(catalogue) = self.config.toolset_catalogues.get(&toolset.0) {
                return self.maybe_tool_search(Arc::new(
                    crate::tool_registry::RealToolRegistry::new(
                        catalogue.clone(),
                        self.config.sandbox.clone(),
                        session_id.clone(),
                        self.config.project_id.clone(),
                        self.config.storage.run().clone(),
                        self.config.storage.memory().clone(),
                    ),
                ));
            }
        }
        // Fallback: empty handle (back-compat) or unregistered non-empty handle.
        match &self.config.catalogue_registry {
            Some(catalogue) => {
                self.maybe_tool_search(Arc::new(crate::tool_registry::RealToolRegistry::new(
                    catalogue.clone(),
                    self.config.sandbox.clone(),
                    session_id.clone(),
                    self.config.project_id.clone(),
                    self.config.storage.run().clone(),
                    self.config.storage.memory().clone(),
                )))
            }
            None => self.config.tool_registry.clone(),
        }
    }

    /// Wrap a resolved catalogue registry in a
    /// [`ToolSearchRegistry`](crate::tool_registry::ToolSearchRegistry) when
    /// [`HarnessConfig::tool_search`] is configured; otherwise return it unchanged.
    /// Built per turn-loop window (alongside the registry it wraps), so a searched
    /// tool stays loaded for the rest of the window and resets on the next.
    pub(crate) fn maybe_tool_search(&self, reg: Arc<dyn ToolRegistry>) -> Arc<dyn ToolRegistry> {
        match &self.config.tool_search {
            Some(cfg) => Arc::new(crate::tool_registry::ToolSearchRegistry::new(
                reg,
                cfg.clone(),
            )),
            None => reg,
        }
    }

    pub(crate) async fn run_react_inner(&self, params: ReactParams) -> RunResult {
        // The fully-resolved per-window inputs, bundled into `ReactParams` so the
        // turn-loop entry points stay under the argument-count lint. Destructured
        // straight back into the loop-body locals so the body below is unchanged.
        let ReactParams {
            task,
            max_iterations,
            mut session_state,
            mut budget_used,
            on_stream,
            // #124: the worker agent is RESOLVED by the caller (the recursing
            // `ReactConfig::run` resolves `self.agent` from the registry; Ralph
            // may override it per window). The leaf no longer reads `config.agent`.
            agent,
            // Issue 2: the leaf's RESOLVED `toolset` handle (mirrors `agent`).
            // Empty (`""`) ⇒ global-catalogue fallback; a non-empty handle with
            // its own per-key catalogue ⇒ strict per-node scoping.
            toolset,
            // Issue #139: the leaf's RESOLVED output schema (`None` ⇒ no
            // enforcement, identical to pre-#139). When `Some`, the terminal is
            // validated against it (frozen validator subset) and a validation
            // failure feeds the error back + retries up to
            // `output_schema_max_retries` extra turns WITHIN budget; on exhaustion
            // WITH budget remaining the window returns
            // `HaltReason::OutputSchemaViolation`. Also set on every turn's
            // `ModelParams.output_schema` so the Ollama `format` channel
            // constrains decoding (Anthropic/OpenAI ignore it).
            output_schema,
            output_schema_max_retries,
            // SC-10: the leaf's per-node system-prompt OVERRIDE. `None` ⇒ the
            // global `config.system_prompt` is used (byte-identical to pre-SC-10).
            // `Some` ⇒ it REPLACES the global prompt for every turn of this window.
            system_prompt,
            // Whether this window is a PlanExecute EXECUTE step — scopes the
            // behavioral nudge so plan-JSON / synthesis turns are not nudged.
            execute_step,
        } = params;
        let session_id = task.session_id.clone();
        // Resolve the effective tool registry once per turn-loop window (all
        // strategies funnel through here). Bridges the per-node toolset catalogue
        // (or the global catalogue when the handle is empty) per-run.
        let tool_registry = self.effective_tool_registry(&session_id, &toolset);
        // Reset the adaptive prompt-based-tool-calling escalation flag at the
        // start of this turn-loop window so detection is scoped to the window
        // and does not leak across `run()` calls (the flag is shared with the
        // model wrapper for the harness's lifetime). No-op unless a
        // `conversational` harness installed the adaptive wrapper.
        if let Some(flag) = self.config.prompt_tool_call_flag.as_ref() {
            flag.store(false, std::sync::atomic::Ordering::Release);
        }
        let started_at = Instant::now();
        let mut usage = AggregateUsage::default();
        // Per-run Stop-hook block counter (issue #69, Decision 4). Resets on
        // every run() — resume starts fresh. After `max_stop_blocks` consecutive
        // blocks the loop terminates anyway (R14).
        let mut stop_blocks: u32 = 0;
        // Consecutive-recoverable-tool-error breaker state (issue #137), keyed
        // by tool name. Loop-local to this window: a fresh `run()`/re-driven
        // `Continue` window starts with a clean N/2N allowance (AC3 Continue
        // reset). `N` is `error_loop_threshold`; the hard stop is at `2 * N`.
        let error_loop_n = self.config.error_loop_threshold;
        let mut error_runs: std::collections::HashMap<String, ErrorRun> =
            std::collections::HashMap::new();
        // Repeated-identical-SUCCESSFUL-call breaker state (the no-progress twin
        // of the error breaker above), loop-local to this window like `error_runs`.
        // Tracks the LAST successful (tool, args) signature and how many times in a
        // row it has recurred; a different successful call resets the run. Shares
        // `error_loop_n`'s `N`/`2N` thresholds. The error breaker owns ERRORS; this
        // owns SUCCESSES, so the two partition cleanly and never double-count.
        let mut last_success_sig: Option<(String, serde_json::Value)> = None;
        let mut repeat_count: u32 = 0;
        let mut repeat_injected = false;
        // Output-schema enforcement state (issue #139), loop-local to this
        // window. `output_schema_retries_used` counts the EXTRA retry turns spent
        // on validation feedback; the budget for them is `output_schema_max_retries`
        // (the `N`). `last_schema_error` holds the most recent frozen validator
        // error so a final exhaustion can report it. Both are inert when
        // `output_schema` is `None` (enforcement OFF / no schema).
        let mut output_schema_retries_used: u32 = 0;
        // Set on every validation failure before it is read in the violation
        // terminal; the initial empty value is never observed.
        #[allow(unused_assignments)]
        let mut last_schema_error: String = String::new();
        // Monotonic per-run span counter for turn / tool-call span ids, and the
        // most recent turn span — parent for the tool-call spans of that turn.
        let mut span_seq: u64 = 0;
        // Set on every turn before any tool dispatch; the initial `None` only
        // covers the (unreachable) case of a tool span with no preceding turn.
        #[allow(unused_assignments)]
        let mut current_turn_base: Option<SpanBase> = None;
        let effective_turn_cap = match task.budget.max_turns {
            Some(t) => t.min(max_iterations),
            None => max_iterations,
        };
        // Per-window cap for the behavioral tool-call nudge (see
        // `HarnessConfig::tool_call_nudge`) so a model that keeps answering without
        // acting can't loop forever, plus a count of tools actually called this
        // window — the signal is "execute turn finished having called ZERO tools".
        const MAX_TOOL_CALL_NUDGES: u32 = 2;
        let mut tool_call_nudges_used: u32 = 0;
        let mut tools_called_this_window: u32 = 0;
        // Send_message-ONLY turns are pure narration (#81) and must not consume the
        // productive-turn budget. Each refunds one turn of the cap in the gate below,
        // keeping `budget_used.turns` monotonic (span ids stay unique) while the
        // budget gate discounts narration. Capped at `effective_turn_cap` so a
        // pure-narration loop still terminates — the window runs at most ~2× budget
        // (productive turns are still bounded at `effective_turn_cap`).
        let mut narration_refund: u32 = 0;

        loop {
            // Mid-run steering (pi-style, `HarnessConfig::steering_source`):
            // drain any user messages queued since the last turn boundary and
            // inject them BEFORE the next model call — after the previous
            // turn's tool calls fully executed, so steering redirects the next
            // turn without ever cancelling in-flight work. Each is recorded on
            // the session via the ContextManager seam (it persists with the
            // transcript) and emitted as `StreamEvent::UserMessage` so UIs
            // render what was injected.
            if let Some(source) = self.config.steering_source.as_ref() {
                for text in source() {
                    self.config
                        .context_manager
                        .append_user_message(&mut session_state, &text)
                        .await;
                    Self::emit(
                        &on_stream,
                        StreamEvent::UserMessage {
                            content: text,
                            steered: true,
                            node: None,
                        },
                    );
                }
            }

            // Layer-1 budget gate before the turn. The single `budget_exceeded`
            // site folds in the turn-cap check: `effective_turn_cap` (the lower of
            // `budget.max_turns` and `max_iterations`) is passed as the hard cap, so
            // `Turns` is still surfaced first and at the same threshold as the old
            // two-gate form.
            if let Some(limit_type) = Self::budget_exceeded(
                &task.budget,
                &budget_used,
                started_at,
                Some(effective_turn_cap.saturating_add(narration_refund)),
            ) {
                return RunResult::Failure {
                    reason: HaltReason::BudgetExceeded { limit_type },
                    session_id,
                    usage,
                    turns: budget_used.turns,
                    session_state: session_state.clone(),
                };
            }

            // Middleware: BeforeTurn (rich chain, issue #11). The chain may mutate
            // `session_state` in place (priority-ordered fan-out); a
            // `ContinueWithModification` is the modified-but-proceed signal.
            if let Some(mw) = self.config.middleware.as_ref() {
                match mw
                    .fire_before_turn(&mut session_state, budget_used.turns)
                    .await
                {
                    MiddlewareDecision::Continue | MiddlewareDecision::ContinueWithModification => {
                    }
                    MiddlewareDecision::Halt { reason } => {
                        return RunResult::Failure {
                            reason: HaltReason::MiddlewareHalt {
                                hook: HookPoint::BeforeTurn,
                                reason,
                            },
                            session_id,
                            usage,
                            turns: budget_used.turns,
                            session_state: session_state.clone(),
                        };
                    }
                    MiddlewareDecision::SurfaceToHuman { request } => {
                        let state = PausedState {
                            session_id: session_id.clone(),
                            task_id: task.id.clone(),
                            turn_number: budget_used.turns,
                            session_state,
                            pending_tool_calls: vec![],
                            approved_results: vec![],
                            human_request: Some(request.clone()),
                            task,
                            budget_used,
                            child_state: None,
                            // #140: carry this leaf's toolset handle so resume
                            // routes through its scoped catalogue.
                            toolset: toolset.clone(),
                        };
                        return RunResult::WaitingForHuman {
                            state: Box::new(state),
                            request,
                        };
                    }
                    // `ForceAnotherTurn` is valid only at `BeforeCompletion`; the
                    // StandardMiddlewareChain converts it to `Halt` here. Handle it
                    // defensively as a halt for any custom chain that emits it.
                    MiddlewareDecision::ForceAnotherTurn { inject } => {
                        return RunResult::Failure {
                            reason: HaltReason::MiddlewareHalt {
                                hook: HookPoint::BeforeTurn,
                                reason: format!(
                                    "ForceAnotherTurn is not valid at BeforeTurn: {inject}"
                                ),
                            },
                            session_id,
                            usage,
                            turns: budget_used.turns,
                            session_state: session_state.clone(),
                        };
                    }
                }
            }

            // Assemble + invoke agent for one turn. `sources` (issue #115 / SC-26)
            // carries the rich ContextSources into the structural assemble seam.
            // Slice 1 threads the seam with tool schemas only; guides/memory/skills
            // populate in later slices. An empty `sources` renders to nothing, so
            // the no-source path stays byte-identical.
            let sources = self
                .build_context_sources(tool_registry.schemas(), &task.instruction)
                .await;
            let mut context = self
                .config
                .context_manager
                .assemble(&session_state, &task, &sources)
                .await;
            // Deliver the registry's tool schemas to the model. Tool schemas are
            // owned by the `ToolRegistry`; the harness wires them into the
            // assembled context so the model knows what it can call. Only fill
            // when the context manager left `tools` empty (the standard
            // compaction adapter does), so a context manager that deliberately
            // sets a phase-specific tool subset is preserved.
            if context.tools.is_empty() {
                context.tools = tool_registry.schemas();
            }
            // Prepend the configured operating system prompt (issue #91), MERGING
            // it with any structural #115 context block the manager already placed
            // as a leading System message (guides/memory/composed prompt). The
            // system prompt always leads. When the manager produced NO System
            // message (the common case — empty sources, or a test double), this
            // inserts a fresh System message exactly as before, so the
            // no-structural-block path stays byte-identical. The `starts_with`
            // guard keeps a resumed/seeded session that already leads with the
            // system prompt from being given it twice.
            //
            // SC-10: a leaf's per-node `system_prompt` override WINS over the
            // global `config.system_prompt`. When this window's leaf carries one,
            // it is used EXCLUSIVELY (the global prompt does not leak in), so the
            // plan and execute phases of a `PlanExecuteConfig` can run under
            // distinct prompts. With no leaf override (`None`) the global prompt
            // applies exactly as before — byte-identical.
            let effective_system_prompt = system_prompt
                .as_ref()
                .or(self.config.system_prompt.as_ref());
            if let Some(system_prompt) = effective_system_prompt {
                match context.messages.first_mut() {
                    Some(Message {
                        role: Role::System,
                        content: Content::Text { text },
                    }) if !text.starts_with(system_prompt.as_str()) => {
                        *text = format!("{system_prompt}\n\n{text}");
                    }
                    Some(Message {
                        role: Role::System, ..
                    }) => {
                        // Already leads with the system prompt (or a non-text
                        // System block) — leave it.
                    }
                    _ => {
                        context.messages.insert(
                            0,
                            Message {
                                role: Role::System,
                                content: Content::Text {
                                    text: system_prompt.clone(),
                                },
                            },
                        );
                    }
                }
            }
            // Per-run model params win unconditionally (issue #93). The agent
            // copies `Context.params` verbatim into the `ModelRequest`, so this
            // is the single seam that delivers configured params (e.g.
            // structured tool calls) to every tool-requesting ReAct/execute/
            // streaming turn.
            context.params = self.config.model_params.clone();
            // Issue #139: route the enforced output schema into this turn's
            // constrained-decoding channel (`ModelParams.output_schema`). Ollama
            // honors it via `format`; Anthropic/OpenAI ignore it (no-op, like
            // `structured_tool_calls`). `None` leaves the params byte-identical.
            if output_schema.is_some() {
                context.params.output_schema = output_schema.clone();
            }
            // Whether tools were advertised to the model this turn — a
            // precondition for classifying a prose final response as a missed
            // tool call (adaptive prompt-based escalation). Captured before
            // `context` is moved into the turn.
            let tools_advertised = !context.tools.is_empty();
            Self::emit(
                &on_stream,
                StreamEvent::TurnStart {
                    turn: budget_used.turns + 1,
                    node: None,
                },
            );
            let turn_started_at = Timestamp::now();
            let turn_clock = Instant::now();
            // LLM-native content capture (issue #64): snapshot the assembled
            // INPUT messages (the full prompt the model saw) BEFORE `context`
            // is moved into `agent.turn`. Zero-cost when the guard is off (no
            // clone of the message history).
            let input_messages = if self.config.content_capture.enabled {
                Some(Self::capture_input_messages(
                    &context.messages,
                    self.config.content_capture.max_field_len,
                ))
            } else {
                None
            };
            // Delta-level streaming (issue #103): when a user-facing stream sink
            // is attached, drive the turn through `turn_streaming` and forward each
            // raw `model::StreamEvent` mapped to harness `StreamEvent`s.
            //
            // #stream: the sink is now an `Arc` (cloneable), so the adapter forwards
            // each mapped event to it LIVE — the instant the model emits it — rather
            // than buffering and flushing post-turn (the old `Box`-can't-be-shared
            // workaround). This is what makes token-by-token SSE / mid-turn cancel
            // observable. `TurnStreamState` (block + call-id correlation) lives
            // behind a Mutex shared with the adapter; we map UNDER the lock, RELEASE
            // it, then forward — never running the user sink under the lock (a
            // re-entrant sink would deadlock). Order is preserved: TurnStart (emitted
            // below, before the turn) → live deltas → TurnEnd → coarse events.
            // When no sink is attached we keep the plain `turn` path so the baseline
            // RunResult is byte-identical (back-compat).
            let result = if let Some(sink) = on_stream.clone() {
                let state = std::sync::Arc::new(std::sync::Mutex::new(TurnStreamState::default()));
                let adapter: crate::agent::AgentStreamSink = Box::new(move |ev| {
                    let mapped = {
                        let mut guard = state.lock().expect("stream state poisoned");
                        Self::map_model_stream_event(ev, &mut guard)
                    };
                    for m in mapped {
                        sink(m);
                    }
                });
                agent.turn_streaming(context, adapter).await
            } else {
                agent.turn(context).await
            };
            budget_used.turns += 1;
            // Refund a send_message-ONLY turn (pure narration, #81): it advanced no
            // task work, so it must not count against the productive-turn budget. The
            // turn span above still uses the incremented `budget_used.turns` (ids stay
            // unique); only the budget GATE (next iteration) discounts it. Bounded by
            // `effective_turn_cap` so narration alone cannot loop forever.
            if let TurnResult::ToolCallRequested { calls, .. } = &result {
                if !calls.is_empty()
                    && narration_refund < effective_turn_cap
                    && calls
                        .iter()
                        .all(|c| c.name == crate::tools::SendMessageTool::NAME)
                {
                    narration_refund += 1;
                }
            }
            // Emit a turn span for every model call (issue #12). Fire-and-forget;
            // it never affects control flow. The span base is retained as the
            // parent for any tool-call spans dispatched this turn.
            {
                let zero = TokenUsage::default();
                let u = match &result {
                    TurnResult::ToolCallRequested { usage, .. }
                    | TurnResult::FinalResponse { usage, .. } => usage,
                    TurnResult::Error { usage, .. } => usage.as_ref().unwrap_or(&zero),
                };
                let (stop_reason, tool_calls_requested) = match &result {
                    TurnResult::FinalResponse { .. } => (StopReason::EndTurn, 0),
                    TurnResult::ToolCallRequested { calls, .. } => {
                        (StopReason::ToolUse, calls.len() as u32)
                    }
                    TurnResult::Error { .. } => (StopReason::EndTurn, 0),
                };
                tools_called_this_window += tool_calls_requested;
                let mut base = SpanBase::new_root(
                    SpanId::new(format!(
                        "{}-turn-{}",
                        session_id.as_str(),
                        budget_used.turns
                    )),
                    session_id.clone(),
                    task.id.clone(),
                    SpanKind::Turn,
                    turn_started_at,
                );
                let status = match &result {
                    TurnResult::Error { error, .. } => SpanStatus::Error {
                        message: format!("{error:?}"),
                    },
                    _ => SpanStatus::Ok,
                };
                base.finish(
                    Timestamp::now(),
                    status,
                    turn_clock.elapsed().as_millis() as u64,
                );
                if let Some(obs) = self.config.observability.as_ref() {
                    // LLM-native content capture (issue #64): output text +
                    // requested tool calls, ONLY when the guard is enabled.
                    // Decision 4: turn span carries output + tool calls; no
                    // assembled input-message history.
                    let cc = &self.config.content_capture;
                    let (output_text, tool_calls) = if cc.enabled {
                        let output_text = match &result {
                            TurnResult::FinalResponse { content, .. } => {
                                let (content, truncated) =
                                    truncate_field(content, cc.max_field_len);
                                Some(GenAiMessage {
                                    role: GenAiRole::Assistant,
                                    content,
                                    truncated,
                                })
                            }
                            _ => None,
                        };
                        let tool_calls = match &result {
                            TurnResult::ToolCallRequested { calls, .. } => Some(
                                calls
                                    .iter()
                                    .map(|c| Self::capture_tool_call_args(c, cc.max_field_len))
                                    .collect(),
                            ),
                            _ => None,
                        };
                        (output_text, tool_calls)
                    } else {
                        (None, None)
                    };
                    obs.emit_turn(TurnSpan {
                        base: base.clone(),
                        turn_number: budget_used.turns,
                        input_tokens: u.input_tokens,
                        output_tokens: u.output_tokens,
                        cache_read_tokens: u.cache_read_tokens,
                        cache_write_tokens: u.cache_write_tokens,
                        cost_usd: self.config.pricing.cost_for(
                            u.input_tokens,
                            u.output_tokens,
                            u.cache_read_tokens,
                            u.cache_write_tokens,
                        ),
                        stop_reason,
                        tool_calls_requested,
                        output_text,
                        tool_calls,
                        input_messages,
                    });
                }
                span_seq += 1;
                current_turn_base = Some(base);
            }
            Self::emit(
                &on_stream,
                StreamEvent::TurnEnd {
                    turn: budget_used.turns,
                    // This turn's usage, so a live consumer updates the token chip
                    // PER-TURN (not only at end-of-run). Re-derived from `result`.
                    usage: match &result {
                        TurnResult::ToolCallRequested { usage, .. }
                        | TurnResult::FinalResponse { usage, .. } => usage.clone(),
                        TurnResult::Error { usage, .. } => usage.clone().unwrap_or_default(),
                    },
                    node: None,
                },
            );

            match result {
                TurnResult::FinalResponse {
                    content, usage: u, ..
                } => {
                    usage.add_turn(&u);
                    budget_used.input_tokens += u.input_tokens as u64;
                    budget_used.output_tokens += u.output_tokens as u64;

                    // Adaptive prompt-based tool-calling escalation. Two signals,
                    // both meaning "tools were available but the model produced NO
                    // native tool call":
                    //   (a) PHRASE — the prose carries action-intent language ("I'll
                    //       read …"), so it *meant* to act. Always active; cheap and
                    //       conservative (biased to false negatives). This is the
                    //       original trigger and is unchanged for every consumer.
                    //   (b) STRUCTURAL — an EXECUTE step reached a terminal answer
                    //       having called ZERO tools while tools were advertised and
                    //       no output schema constrained it. This is a behavioral
                    //       signal (no phrase-matching) that catches the failure modes
                    //       the phrase list misses: markdown command fences
                    //       (```bash\ncat …```), bare narration, and empty content —
                    //       exactly how Qwen drops out of native tool-call mode at the
                    //       cold start of an isolated execute sub-session. Gated on
                    //       `tool_call_nudge` so it ONLY affects consumers that opted
                    //       into behavioral repair (local models); native-tool models
                    //       (Claude, …) keep phrase-only behavior and never flip the
                    //       flag on a legitimately tool-free turn.
                    // On either signal: set the session flag so the wrapped model
                    // switches to prompt-based tool calling (the next request describes
                    // the tools in the system prompt and clears the native channel),
                    // nudge the model, and force another turn instead of completing.
                    // Guarded on the flag being unset so it fires at most once per
                    // window (bounded — one extra turn) and only on the
                    // `conversational` adaptive path.
                    if let Some(flag) = self.config.prompt_tool_call_flag.as_ref() {
                        let already_escalated = flag.load(std::sync::atomic::Ordering::Acquire);
                        let phrase = crate::tool_call_repair::detect_prose_response(
                            &content,
                            tools_advertised,
                        )
                        .is_some();
                        let structural = self.config.tool_call_nudge
                            && execute_step
                            && tools_advertised
                            && tools_called_this_window == 0
                            && output_schema.is_none();
                        if !already_escalated && (phrase || structural) {
                            flag.store(true, std::sync::atomic::Ordering::Release);
                            // Record the model's last (failed) output, then a
                            // corrective nudge, so the next turn has coherent context.
                            let assistant = Message {
                                role: Role::Assistant,
                                content: Content::Text {
                                    text: content.clone(),
                                },
                            };
                            self.config
                                .context_manager
                                .append_assistant_message(&mut session_state, &assistant)
                                .await;
                            let nudge = "No tool call was detected in your last \
response. The tools available to you and the exact tool-call format are described in \
the system prompt — use that format now to actually invoke a tool, not prose or a code \
block."
                                .to_string();
                            self.config
                                .context_manager
                                .append_user_message(&mut session_state, &nudge)
                                .await;
                            continue;
                        }
                    }

                    // Behavioral tool-call nudge (opt-in via `tool_call_nudge`). This
                    // does NOT inspect the response text for phrases — it keys on
                    // ACTION: an EXECUTE-phase task window finished (a terminal
                    // answer) having called ZERO tools while tools were advertised,
                    // i.e. the model "reported findings" without ever reading
                    // anything. Record that answer, nudge it to actually use a tool
                    // before reporting, and force another turn. Scoped to execute
                    // steps so the legitimately tool-free plan-JSON turn is untouched,
                    // skips schema-constrained turns, and bounded per window so a
                    // model that refuses to act can't loop. Off (default) →
                    // byte-identical.
                    if self.config.tool_call_nudge
                        && execute_step
                        && tools_advertised
                        && tools_called_this_window == 0
                        && output_schema.is_none()
                        && tool_call_nudges_used < MAX_TOOL_CALL_NUDGES
                    {
                        tool_call_nudges_used += 1;
                        let assistant = Message {
                            role: Role::Assistant,
                            content: Content::Text {
                                text: content.clone(),
                            },
                        };
                        self.config
                            .context_manager
                            .append_assistant_message(&mut session_state, &assistant)
                            .await;
                        let nudge = "You have not called any tool yet, so you have not \
actually inspected anything — your answer would be guessed, not grounded. Use the \
available tools (list_dir, read_file, grep, …) to gather the real data NOW, then \
report only what the tools returned."
                            .to_string();
                        self.config
                            .context_manager
                            .append_user_message(&mut session_state, &nudge)
                            .await;
                        continue;
                    }

                    if let Some(mw) = self.config.middleware.as_ref() {
                        match mw
                            .fire_before_completion(&content, budget_used.turns, &session_state)
                            .await
                        {
                            MiddlewareDecision::Continue
                            | MiddlewareDecision::ContinueWithModification => {}
                            // The chain concatenates every middleware's injection
                            // into one `ForceAnotherTurn` (issue #11). Record the
                            // model's final text, then the injection as a user
                            // message, and force another turn instead of completing
                            // — the same channel the Stop-block breaker uses, so the
                            // conversation stays well-formed (assistant final text →
                            // user injection).
                            MiddlewareDecision::ForceAnotherTurn { inject } => {
                                let assistant = Message {
                                    role: Role::Assistant,
                                    content: Content::Text {
                                        text: content.clone(),
                                    },
                                };
                                self.config
                                    .context_manager
                                    .append_assistant_message(&mut session_state, &assistant)
                                    .await;
                                self.config
                                    .context_manager
                                    .append_user_message(&mut session_state, &inject)
                                    .await;
                                continue;
                            }
                            MiddlewareDecision::Halt { reason } => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::BeforeCompletion,
                                        reason,
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
                            MiddlewareDecision::SurfaceToHuman { request } => {
                                let state = PausedState {
                                    session_id: session_id.clone(),
                                    task_id: task.id.clone(),
                                    turn_number: budget_used.turns,
                                    session_state,
                                    pending_tool_calls: vec![],
                                    approved_results: vec![],
                                    human_request: Some(request.clone()),
                                    task,
                                    budget_used,
                                    child_state: None,
                                    // #140: carry this leaf's toolset handle so
                                    // resume routes through its scoped catalogue.
                                    toolset: toolset.clone(),
                                };
                                return RunResult::WaitingForHuman {
                                    state: Box::new(state),
                                    request,
                                };
                            }
                        }
                    }

                    match self
                        .config
                        .termination_policy
                        .evaluate(&session_state, &budget_used)
                        .await
                    {
                        TerminationDecision::Halt { reason } => {
                            return RunResult::Failure {
                                reason: HaltReason::TerminationPolicyHalt { reason },
                                session_id,
                                usage,
                                turns: budget_used.turns,
                                session_state: session_state.clone(),
                            };
                        }
                        TerminationDecision::Continue => {
                            // Record the assistant's final text in history so a
                            // continued session reflects what the agent said.
                            let msg = Message {
                                role: Role::Assistant,
                                content: Content::Text {
                                    text: content.clone(),
                                },
                            };
                            self.config
                                .context_manager
                                .append_assistant_message(&mut session_state, &msg)
                                .await;

                            // Stop hook (issue #69, Decision 5). The strategy
                            // believes the task is done; fire registered Stop
                            // hooks synchronously. If any blocks (and we are
                            // under `max_stop_blocks`), inject the reason via
                            // the same path `ForceAnotherTurn` uses and continue
                            // the loop instead of terminating.
                            if let Some(reason) = self
                                .fire_stop_hooks(
                                    &session_id,
                                    &task,
                                    budget_used.turns,
                                    &content,
                                    &mut stop_blocks,
                                )
                                .await
                            {
                                self.config
                                    .context_manager
                                    .append_user_message(&mut session_state, &reason)
                                    .await;
                                continue;
                            }

                            // Output-schema enforcement gate (issue #139).
                            // Additive to and EARLIER than the Success terminal:
                            // when a schema is active, validate this terminal
                            // `FinalResponse`. On a validation FAILURE, feed the
                            // frozen error back as a USER message and retry one
                            // more turn (up to `output_schema_max_retries` extra
                            // turns). The retried turn re-enters the loop top,
                            // where the budget gate fires FIRST — so a retry that
                            // would exceed the turn/budget backstop surfaces the
                            // existing `BudgetExceeded` terminal, NOT
                            // `OutputSchemaViolation` (budget-cap-wins, Q2). Only
                            // when the N retries are exhausted WITH budget
                            // remaining does the window terminate with
                            // `OutputSchemaViolation`. `None` schema ⇒ no
                            // validation (migration gate OFF / no `output`), so
                            // the terminal is byte-identical to pre-#139.
                            if let Some(schema) = output_schema.as_ref() {
                                if let Err(verr) =
                                    crate::output_schema::validate_output(&content, schema)
                                {
                                    last_schema_error = verr.clone();
                                    if output_schema_retries_used < output_schema_max_retries {
                                        // Grant one more turn: feed the validator
                                        // error back as a USER message (the
                                        // assistant's invalid text is already in
                                        // history above) and loop. The budget gate
                                        // at the loop top enforces Q2 precedence.
                                        output_schema_retries_used += 1;
                                        Self::emit(
                                            &on_stream,
                                            StreamEvent::OutputSchemaRetry {
                                                attempt: output_schema_retries_used,
                                                error: verr.clone(),
                                                node: None,
                                            },
                                        );
                                        if let Some(obs) = self.config.observability.as_ref() {
                                            let base = SpanBase::new_root(
                                                SpanId::new(format!(
                                                    "{}-output-schema-retry-{}",
                                                    session_id.as_str(),
                                                    span_seq
                                                )),
                                                session_id.clone(),
                                                task.id.clone(),
                                                SpanKind::ContextAssembly,
                                                Timestamp::now(),
                                            );
                                            obs.emit_context(ContextSpan {
                                                base,
                                                operation: ContextOperation::OutputSchemaRetry {
                                                    attempt: output_schema_retries_used,
                                                    error: verr.clone(),
                                                },
                                                tokens_before: 0,
                                                tokens_after: 0,
                                                utilization_before: 0.0,
                                                utilization_after: 0.0,
                                            });
                                        }
                                        let feedback =
                                            crate::output_schema::feedback_message(&verr);
                                        self.config
                                            .context_manager
                                            .append_user_message(&mut session_state, &feedback)
                                            .await;
                                        continue;
                                    }
                                    // Retries exhausted WITH budget remaining
                                    // (the budget gate did not fire) → the typed
                                    // schema-violation terminal (Q2/AC3). Total
                                    // attempts = 1 + max_retries.
                                    let attempts = output_schema_max_retries + 1;
                                    Self::emit(
                                        &on_stream,
                                        StreamEvent::OutputSchemaViolation {
                                            attempts,
                                            error: last_schema_error.clone(),
                                            node: None,
                                        },
                                    );
                                    if let Some(obs) = self.config.observability.as_ref() {
                                        let base = SpanBase::new_root(
                                            SpanId::new(format!(
                                                "{}-output-schema-violation-{}",
                                                session_id.as_str(),
                                                span_seq
                                            )),
                                            session_id.clone(),
                                            task.id.clone(),
                                            SpanKind::ContextAssembly,
                                            Timestamp::now(),
                                        );
                                        obs.emit_context(ContextSpan {
                                            base,
                                            operation: ContextOperation::OutputSchemaViolation {
                                                attempts,
                                                error: last_schema_error.clone(),
                                            },
                                            tokens_before: 0,
                                            tokens_after: 0,
                                            utilization_before: 0.0,
                                            utilization_after: 0.0,
                                        });
                                    }
                                    return RunResult::Failure {
                                        reason: HaltReason::OutputSchemaViolation {
                                            schema: crate::model::canonicalize_json(schema),
                                            attempts,
                                            last_error: last_schema_error.clone(),
                                        },
                                        session_id,
                                        usage,
                                        turns: budget_used.turns,
                                        session_state: session_state.clone(),
                                    };
                                }
                            }

                            Self::emit(
                                &on_stream,
                                StreamEvent::FinalResponse {
                                    content: content.clone(),
                                    node: None,
                                },
                            );
                            return RunResult::Success {
                                output: content,
                                session_id,
                                usage,
                                turns: budget_used.turns,
                                session_state: session_state.clone(),
                            };
                        }
                    }
                }

                TurnResult::ToolCallRequested {
                    mut calls,
                    usage: u,
                    ..
                } => {
                    usage.add_turn(&u);
                    budget_used.input_tokens += u.input_tokens as u64;
                    budget_used.output_tokens += u.output_tokens as u64;

                    // Always-halt short circuit (Layer 1).
                    if let Some(c) = calls.iter().find(|c| tool_registry.is_always_halt(&c.name)) {
                        return RunResult::Failure {
                            reason: HaltReason::UnrecoverableToolError {
                                tool: c.name.clone(),
                                error: "tool is annotated always_halt".into(),
                            },
                            session_id,
                            usage,
                            turns: budget_used.turns,
                            session_state: session_state.clone(),
                        };
                    }

                    // Record the assistant's turn (the tool calls the model
                    // requested) as soon as the calls are known — BEFORE the
                    // BeforeTool middleware (which may pause via SurfaceToHuman)
                    // and before any tool result. This keeps the conversation
                    // well-formed (assistant tool_use precedes its tool result)
                    // on every path, including human-in-the-loop resume, so
                    // resume_inner never has to append it. The recorded turn
                    // reflects the model's original request; a middleware or
                    // human modification changes only what is dispatched.
                    for call in &calls {
                        let msg = Message {
                            role: Role::Assistant,
                            content: Content::ToolCall(call.clone()),
                        };
                        self.config
                            .context_manager
                            .append_assistant_message(&mut session_state, &msg)
                            .await;
                    }

                    // Middleware: BeforeTool (rich chain, issue #11 / SC-11). The
                    // chain mutates `calls` IN PLACE via a priority-ordered
                    // fan-out; `ContinueWithModification` is the modified-but-proceed
                    // signal. The assistant turn recorded just above keeps the
                    // model's ORIGINAL request — only what is dispatched changes.
                    if let Some(mw) = self.config.middleware.as_ref() {
                        match mw.fire_before_tool(&mut calls, budget_used.turns).await {
                            MiddlewareDecision::Continue
                            | MiddlewareDecision::ContinueWithModification => {}
                            MiddlewareDecision::Halt { reason } => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::BeforeTool,
                                        reason,
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
                            MiddlewareDecision::SurfaceToHuman { request } => {
                                let state = PausedState {
                                    session_id: session_id.clone(),
                                    task_id: task.id.clone(),
                                    turn_number: budget_used.turns,
                                    session_state,
                                    pending_tool_calls: calls,
                                    approved_results: vec![],
                                    human_request: Some(request.clone()),
                                    task,
                                    budget_used,
                                    child_state: None,
                                    // #140: carry this leaf's toolset handle so
                                    // resume routes through its scoped catalogue.
                                    toolset: toolset.clone(),
                                };
                                return RunResult::WaitingForHuman {
                                    state: Box::new(state),
                                    request,
                                };
                            }
                            // `ForceAnotherTurn` is valid only at `BeforeCompletion`;
                            // the StandardMiddlewareChain converts it to `Halt` here.
                            // Defensive for a custom chain that emits it.
                            MiddlewareDecision::ForceAnotherTurn { inject } => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::BeforeTool,
                                        reason: format!(
                                            "ForceAnotherTurn is not valid at BeforeTool: {inject}"
                                        ),
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
                        }
                    }

                    let mut approved_results: Vec<ToolResult> = Vec::new();
                    // SC-9: the `session_state.messages` index of each appended
                    // tool result, recorded 1:1 with `approved_results`. The
                    // AfterTool middleware hook uses these to re-render any result
                    // it rewrites in place (via `ContextManager::replace_tool_result`).
                    let mut result_msg_indices: Vec<usize> = Vec::new();
                    for (i, call) in calls.iter().enumerate() {
                        // Sandbox validation. The pre-dispatch `validate` seam
                        // honours the SAME policy as tool-surfaced violations
                        // (`sandbox_violation_halts`): halt only under `Halt` for
                        // an always-halt-eligible violation; otherwise recoverable.
                        if let Err(violation) = self.config.sandbox.validate(call).await {
                            if self.sandbox_violation_halts(&violation) {
                                return RunResult::Failure {
                                    reason: HaltReason::SandboxViolation { violation },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
                            // Recoverable — append as tool error.
                            let tr = ToolResult {
                                call_id: call.id.clone(),
                                output: ToolOutput::Error {
                                    message: format!("sandbox: {violation}"),
                                    recoverable: true,
                                },
                            };
                            Self::emit(
                                &on_stream,
                                StreamEvent::ToolResult {
                                    call_id: call.id.clone(),
                                    is_error: true,
                                    // Q5: carry the result content.
                                    content: format!("sandbox: {violation}"),
                                    revised: false,
                                    node: None,
                                },
                            );
                            self.config
                                .context_manager
                                .append_tool_result(&mut session_state, &tr)
                                .await;
                            result_msg_indices.push(session_state.messages.len().saturating_sub(1));
                            approved_results.push(tr);
                            continue;
                        }

                        Self::emit(
                            &on_stream,
                            StreamEvent::ToolCall {
                                call_id: call.id.clone(),
                                name: call.name.clone(),
                                // Q5: carry the final tool-call arguments.
                                args: call.input.clone(),
                                node: None,
                            },
                        );
                        let tool_started_at = Timestamp::now();
                        let tool_clock = Instant::now();
                        // `effective_call` is the call actually dispatched. It
                        // starts as the agent's call and may be replaced by a
                        // repaired variant below. Spans/results are stamped from it.
                        // Dispatch + tool-call repair (additive): coerce a
                        // recoverable-error call's args and re-dispatch, enriching
                        // the error if repair gives up. Extracted to
                        // `dispatch_with_repair` so the resume path (gated calls)
                        // shares it. `effective_call` is the call actually
                        // dispatched (repaired variant when repair fired); spans /
                        // results are stamped from it.
                        let (effective_call, mut output) =
                            self.dispatch_with_repair(tool_registry.as_ref(), call.clone()).await;

                        // Tool-surfaced sandbox violation: the HARNESS decides
                        // halt-vs-recoverable per `sandbox_violation_policy`, not
                        // the tool. Under `Halt` an always-halt-eligible violation
                        // ends the run with a typed `HaltReason::SandboxViolation`;
                        // otherwise (the default) it becomes a recoverable error
                        // the model retries from — uniform across every tool.
                        if let ToolOutput::SandboxViolation { violation } = &output {
                            if self.sandbox_violation_halts(violation) {
                                return RunResult::Failure {
                                    reason: HaltReason::SandboxViolation {
                                        violation: violation.clone(),
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
                            let message = format!("sandbox violation: {violation}");
                            output = ToolOutput::Error {
                                message,
                                recoverable: true,
                            };
                        }
                        let call = &effective_call;

                        // #126 (AC2): record harness-OBSERVED write/edit paths
                        // from the call ACTUALLY dispatched. This is the single
                        // file-observation point the PlanExecute DAG executor reads
                        // via `take_observed_writes` to build a task's ledger
                        // `files_touched` — never a model-self-reported field.
                        self.observe_write_call(call);

                        // Pause propagation: WaitingForHuman from a subagent tool.
                        if let ToolOutput::WaitingForHuman {
                            child_state,
                            request,
                        } = output
                        {
                            let remaining = calls[i + 1..].to_vec();
                            let state = PausedState {
                                session_id: session_id.clone(),
                                task_id: task.id.clone(),
                                turn_number: budget_used.turns,
                                session_state,
                                pending_tool_calls: remaining,
                                approved_results,
                                human_request: Some(request.clone()),
                                task,
                                budget_used,
                                child_state: Some(*child_state),
                                // #140: the parent leaf's toolset handle (the
                                // child carries its own inside `child_state`).
                                toolset: toolset.clone(),
                            };
                            return RunResult::WaitingForHuman {
                                state: Box::new(state),
                                request,
                            };
                        }

                        // Escalation propagation (issue #80): a tool requests a
                        // structural state change. The harness is a pure
                        // intermediary — it does NOT act on the signal. It
                        // terminates cleanly, preserves session state for a
                        // possible resume, and returns RunResult::Escalate.
                        // The escalation is a control signal, NOT a conversation
                        // turn: it is never appended to message history, and the
                        // remaining tool calls in this batch are preserved into
                        // pending_tool_calls (mirroring WaitingForHuman). The
                        // signal is NOT stored in PausedState, so it is
                        // discarded on resume — the harness never re-acts on it.
                        if let ToolOutput::Escalate { signal } = output {
                            let remaining = calls[i + 1..].to_vec();
                            let turns = budget_used.turns;
                            let state = PausedState {
                                session_id: session_id.clone(),
                                task_id: task.id.clone(),
                                turn_number: budget_used.turns,
                                session_state,
                                pending_tool_calls: remaining,
                                approved_results,
                                human_request: None,
                                task,
                                budget_used,
                                child_state: None,
                                // #140: carry this leaf's toolset handle so resume
                                // routes pending per-node calls through its catalogue.
                                toolset: toolset.clone(),
                            };
                            return RunResult::Escalate {
                                signal,
                                state: Box::new(state),
                                session_id,
                                usage,
                                turns,
                            };
                        }

                        // Clarification pause (issue #81, Q4b): a tool (e.g.
                        // `ask_user_question`) needs a human answer before it can
                        // produce a result. UNLIKE the subagent `WaitingForHuman`
                        // path, there is NO `ChildPausedState`: the loop builds a
                        // `PausedState` directly with `human_request` set to
                        // `Clarification`. The CLARIFYING call itself is preserved
                        // as the head of `pending_tool_calls` (followed by the
                        // remaining batch) so that, on resume, the human's answer
                        // is injected as the tool RESULT for that pending call.
                        if let ToolOutput::AwaitingClarification { question, options } = output {
                            let mut pending = vec![call.clone()];
                            pending.extend_from_slice(&calls[i + 1..]);
                            let request = HumanRequest::Clarification { question, options };
                            let state = PausedState {
                                session_id: session_id.clone(),
                                task_id: task.id.clone(),
                                turn_number: budget_used.turns,
                                session_state,
                                pending_tool_calls: pending,
                                approved_results,
                                human_request: Some(request.clone()),
                                task,
                                budget_used,
                                child_state: None,
                                // #140: carry this leaf's toolset handle so the
                                // preserved clarifying call resumes against its
                                // scoped catalogue.
                                toolset: toolset.clone(),
                            };
                            return RunResult::WaitingForHuman {
                                state: Box::new(state),
                                request,
                            };
                        }

                        // Consult pause (issue #114, R1/R10): a worker-side tool
                        // returns `ToolOutput::Consult { child_state: None, .. }`
                        // to ask for mid-loop help. Like the
                        // `AwaitingClarification` arm, there is NO
                        // `ChildPausedState` at this level — the loop builds a
                        // `PausedState` directly with `human_request: None` and
                        // preserves the CONSULTING call as the head of
                        // `pending_tool_calls` (followed by the remaining batch),
                        // so that on `resume_consult` the helper's answer is
                        // injected as the tool RESULT for that pending call. The
                        // consult is a control signal, NOT a conversation turn:
                        // it is never appended to message history here (R10).
                        if let ToolOutput::Consult { request, .. } = output {
                            // Observability: lightweight consult-spawn event
                            // alongside `SkillInjected`.
                            if let Some(obs) = self.config.observability.as_ref() {
                                let base = SpanBase::new_root(
                                    SpanId::new(format!(
                                        "{}-consult-spawn-{}",
                                        session_id.as_str(),
                                        span_seq
                                    )),
                                    session_id.clone(),
                                    task.id.clone(),
                                    SpanKind::ContextAssembly,
                                    Timestamp::now(),
                                );
                                obs.emit_context(ContextSpan {
                                    base,
                                    operation: ContextOperation::ConsultSpawned {
                                        consult_kind: request.kind.clone(),
                                    },
                                    tokens_before: 0,
                                    tokens_after: 0,
                                    utilization_before: 0.0,
                                    utilization_after: 0.0,
                                });
                                // No `span_seq += 1` here: we return immediately
                                // below, so the increment would be dead.
                            }
                            let mut pending = vec![call.clone()];
                            pending.extend_from_slice(&calls[i + 1..]);
                            let turns = budget_used.turns;
                            let state = PausedState {
                                session_id: session_id.clone(),
                                task_id: task.id.clone(),
                                turn_number: budget_used.turns,
                                session_state,
                                pending_tool_calls: pending,
                                approved_results,
                                human_request: None,
                                task,
                                budget_used,
                                child_state: None,
                                // #140 (THE load-bearing path): carry this leaf's
                                // toolset handle so `resume_consult` routes the
                                // preserved consulting call through its scoped
                                // catalogue instead of the global fallback.
                                toolset: toolset.clone(),
                            };
                            return RunResult::Consult {
                                request,
                                state: Box::new(state),
                                session_id,
                                usage,
                                turns,
                            };
                        }

                        // SendMessage (issue #81): the `send_message` tool surfaces
                        // an out-of-band message to the user. The loop emits a
                        // `StreamEvent::UserMessage` rather than collapsing the
                        // content into a normal tool result, then records a minimal
                        // success result so the loop continues.
                        let output = if call.name == crate::tools::SendMessageTool::NAME {
                            if let ToolOutput::Success { content, .. } = &output {
                                Self::emit(
                                    &on_stream,
                                    StreamEvent::UserMessage {
                                        content: content.clone(),
                                        steered: false,
                                        node: None,
                                    },
                                );
                            }
                            output
                        } else {
                            output
                        };

                        let is_error = matches!(output, ToolOutput::Error { .. });
                        // Layer-2: unrecoverable tool error halts immediately.
                        if let ToolOutput::Error {
                            ref message,
                            recoverable: false,
                        } = output
                        {
                            return RunResult::Failure {
                                reason: HaltReason::UnrecoverableToolError {
                                    tool: call.name.clone(),
                                    error: message.clone(),
                                },
                                session_id,
                                usage,
                                turns: budget_used.turns,
                                session_state: session_state.clone(),
                            };
                        }

                        // ── Consecutive-recoverable-tool-error breaker (#137) ──
                        // `output` here is either a Success or a RECOVERABLE
                        // Error (every other variant early-returned above). On a
                        // success the tool's error run resets (AC1); on a
                        // recoverable error we increment the identical-args run
                        // (AC1 args-variant) and check the N / 2N thresholds.
                        // At N we stash the corrective USER message here and
                        // append it AFTER the tool result below (well-formed
                        // conversation: assistant tool_use → tool result →
                        // corrective user message).
                        let mut pending_corrective: Option<String> = None;
                        if let ToolOutput::Error {
                            recoverable: true, ..
                        } = &output
                        {
                            let run = match error_runs.get_mut(&call.name) {
                                // Same tool, structurally-identical args → extend
                                // the run.
                                Some(existing) if existing.args == call.input => {
                                    existing.count += 1;
                                    existing
                                }
                                // First error, or the args changed → fresh run.
                                _ => error_runs
                                    .entry(call.name.clone())
                                    .insert_entry(ErrorRun {
                                        args: call.input.clone(),
                                        count: 1,
                                        injected: false,
                                    })
                                    .into_mut(),
                            };
                            let count = run.count;
                            let two_n = error_loop_n.saturating_mul(2);

                            // 2N: HARD STOP (AC3). Do NOT append this last tool
                            // result or continue — return a typed terminal that
                            // `ReactConfig::run` routes through the node's
                            // `BudgetExhaustedBehavior` WITHOUT burning the rest
                            // of the budget. The breaker is disabled when N == 0.
                            if error_loop_n > 0 && count >= two_n {
                                // AC4: emit the "broken" pair (stream + obs).
                                Self::emit(
                                    &on_stream,
                                    StreamEvent::ToolErrorLoopBroken {
                                        tool: call.name.clone(),
                                        consecutive_errors: count,
                                        node: None,
                                    },
                                );
                                if let Some(obs) = self.config.observability.as_ref() {
                                    let base = SpanBase::new_root(
                                        SpanId::new(format!(
                                            "{}-tool-error-loop-broken-{}",
                                            session_id.as_str(),
                                            span_seq
                                        )),
                                        session_id.clone(),
                                        task.id.clone(),
                                        SpanKind::ContextAssembly,
                                        Timestamp::now(),
                                    );
                                    obs.emit_context(ContextSpan {
                                        base,
                                        operation: ContextOperation::ToolErrorLoopBroken {
                                            tool_name: call.name.clone(),
                                            consecutive_errors: count,
                                        },
                                        tokens_before: 0,
                                        tokens_after: 0,
                                        utilization_before: 0.0,
                                        utilization_after: 0.0,
                                    });
                                }
                                return RunResult::Failure {
                                    reason: HaltReason::ToolErrorLoop {
                                        tool: call.name.clone(),
                                        consecutive_errors: count,
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }

                            // N: inject ONE corrective message (AC2). Render the
                            // schema+hint via `enrich_tool_error` (reused
                            // verbatim) and inject it as a USER-role message —
                            // the SAME channel the Stop-block breaker uses — once
                            // per run (do not re-inject between N and 2N).
                            if error_loop_n > 0 && count >= error_loop_n && !run.injected {
                                run.injected = true;
                                let schema = tool_registry
                                    .schemas()
                                    .into_iter()
                                    .find(|s| s.name == call.name);
                                // `enrich_tool_error` renders the bare message +
                                // schema + hint; the bare message is this run's
                                // own error text.
                                let bare = match &output {
                                    ToolOutput::Error { message, .. } => message.as_str(),
                                    _ => "",
                                };
                                let corrective =
                                    match Self::enrich_tool_error(bare, schema.as_ref()) {
                                        ToolOutput::Error { message, .. } => message,
                                        // enrich_tool_error always returns Error.
                                        _ => bare.to_string(),
                                    };
                                // AC4: emit the "detected/warning" pair (stream +
                                // obs) BEFORE the corrective message is appended.
                                Self::emit(
                                    &on_stream,
                                    StreamEvent::ToolErrorLoopDetected {
                                        tool: call.name.clone(),
                                        consecutive_errors: count,
                                        node: None,
                                    },
                                );
                                if let Some(obs) = self.config.observability.as_ref() {
                                    let base = SpanBase::new_root(
                                        SpanId::new(format!(
                                            "{}-tool-error-loop-detected-{}",
                                            session_id.as_str(),
                                            span_seq
                                        )),
                                        session_id.clone(),
                                        task.id.clone(),
                                        SpanKind::ContextAssembly,
                                        Timestamp::now(),
                                    );
                                    obs.emit_context(ContextSpan {
                                        base,
                                        operation: ContextOperation::ToolErrorLoopDetected {
                                            tool_name: call.name.clone(),
                                            consecutive_errors: count,
                                        },
                                        tokens_before: 0,
                                        tokens_after: 0,
                                        utilization_before: 0.0,
                                        utilization_after: 0.0,
                                    });
                                }
                                // Append after the normal tool-result append below
                                // so the conversation stays well-formed (assistant
                                // tool_use → tool result → corrective user msg).
                                // Stash it for injection just after the tool
                                // result is recorded.
                                pending_corrective = Some(corrective);
                            }
                        } else {
                            // AC1: ANY success for this tool resets its error run.
                            error_runs.remove(&call.name);

                            // ── Repeated-identical-successful-call breaker ──
                            // A model that calls the SAME tool with the SAME args
                            // and keeps SUCCEEDING is spinning: it makes no
                            // progress and never reaches a terminal answer (the
                            // plan-phase loop that silently burned the budget).
                            // Track consecutive identical successes; nudge once at
                            // `N`, hard-stop at `2N`. `send_message` is exempt
                            // (benign narration a model may legitimately repeat).
                            if error_loop_n > 0 && call.name != crate::tools::SendMessageTool::NAME
                            {
                                let sig = (call.name.clone(), call.input.clone());
                                if last_success_sig.as_ref() == Some(&sig) {
                                    repeat_count += 1;
                                } else {
                                    last_success_sig = Some(sig);
                                    repeat_count = 1;
                                    repeat_injected = false;
                                }
                                let two_n = error_loop_n.saturating_mul(2);
                                // 2N: HARD STOP. Do NOT append this result or
                                // continue — end with a typed terminal so the
                                // caller/operator can redirect. The run does NOT
                                // auto-continue (more budget would just feed the
                                // same loop), and no remaining budget is burned.
                                if repeat_count >= two_n {
                                    return RunResult::Failure {
                                        reason: HaltReason::RepeatedToolCall {
                                            tool: call.name.clone(),
                                            count: repeat_count,
                                        },
                                        session_id,
                                        usage,
                                        turns: budget_used.turns,
                                        session_state: session_state.clone(),
                                    };
                                }
                                // N: inject ONE corrective USER message (the same
                                // channel the error breaker / Stop-block breaker
                                // use), once per run, appended after this result.
                                if repeat_count >= error_loop_n && !repeat_injected {
                                    repeat_injected = true;
                                    pending_corrective = Some(format!(
                                        "You have called the `{tool}` tool with the same arguments \
{count} times in a row, each returning the same result. Repeating an identical call will not make \
progress. Take a DIFFERENT action now — or, if you already have what you need, produce your final \
response (finalize the plan / answer the task).",
                                        tool = call.name,
                                        count = repeat_count,
                                    ));
                                }
                            }
                        }

                        // Tool-call span (issue #12), child of the current turn.
                        if let Some(obs) = self.config.observability.as_ref() {
                            // LLM-native content capture (issue #64): tool args +
                            // tool result, ONLY when the guard is enabled.
                            let cc = &self.config.content_capture;
                            let (tool_args_content, tool_result_content) = if cc.enabled {
                                let args = Self::capture_tool_call_args(call, cc.max_field_len);
                                let result = match &output {
                                    ToolOutput::Success { content, .. } => {
                                        let (content, t) =
                                            truncate_field(content, cc.max_field_len);
                                        Some(ToolResultContent {
                                            content,
                                            truncated: t,
                                        })
                                    }
                                    ToolOutput::Error { message, .. } => {
                                        let (content, t) =
                                            truncate_field(message, cc.max_field_len);
                                        Some(ToolResultContent {
                                            content,
                                            truncated: t,
                                        })
                                    }
                                    ToolOutput::WaitingForHuman { .. } => None,
                                    // Escalate / AwaitingClarification / Consult
                                    // are handled and returned above; they never
                                    // reach the capture path.
                                    ToolOutput::Escalate { .. } => None,
                                    ToolOutput::AwaitingClarification { .. } => None,
                                    ToolOutput::Consult { .. } => None,
                                    // Normalized into an `Error` (or a halt)
                                    // before this point; never reaches here.
                                    ToolOutput::SandboxViolation { .. } => None,
                                };
                                (Some(args), result)
                            } else {
                                (None, None)
                            };
                            let (output_size_bytes, truncated) = match &output {
                                ToolOutput::Success { content, truncated } => {
                                    (content.len(), *truncated)
                                }
                                ToolOutput::Error { message, .. } => (message.len(), false),
                                ToolOutput::WaitingForHuman { .. } => (0, false),
                                ToolOutput::Escalate { .. } => (0, false),
                                ToolOutput::AwaitingClarification { .. } => (0, false),
                                ToolOutput::Consult { .. } => (0, false),
                                // Normalized away before this point.
                                ToolOutput::SandboxViolation { .. } => (0, false),
                            };
                            let span_id =
                                SpanId::new(format!("{}-tool-{}", session_id.as_str(), span_seq));
                            let mut base = match &current_turn_base {
                                Some(parent) => SpanBase::new_child(
                                    span_id,
                                    parent,
                                    SpanKind::ToolCall,
                                    tool_started_at,
                                ),
                                None => SpanBase::new_root(
                                    span_id,
                                    session_id.clone(),
                                    task.id.clone(),
                                    SpanKind::ToolCall,
                                    tool_started_at,
                                ),
                            };
                            let status = if is_error {
                                SpanStatus::Error {
                                    message: "tool returned a recoverable error".into(),
                                }
                            } else {
                                SpanStatus::Ok
                            };
                            base.finish(
                                Timestamp::now(),
                                status,
                                tool_clock.elapsed().as_millis() as u64,
                            );
                            obs.emit_tool_call(ToolCallSpan {
                                base,
                                tool_name: call.name.clone(),
                                call_id: call.id.clone(),
                                parameters_size_bytes: call.input.to_string().len(),
                                output_size_bytes,
                                truncated,
                                sandbox_mode: String::new(),
                                sandbox_violations: Vec::new(),
                                arguments: tool_args_content,
                                result: tool_result_content,
                            });
                            span_seq += 1;
                        }

                        let tr = ToolResult {
                            call_id: call.id.clone(),
                            output,
                        };
                        // Q5: surface the result content on the coarse event.
                        let result_content = match &tr.output {
                            ToolOutput::Success { content, .. } => content.clone(),
                            ToolOutput::Error { message, .. } => message.clone(),
                            _ => String::new(),
                        };
                        Self::emit(
                            &on_stream,
                            StreamEvent::ToolResult {
                                call_id: call.id.clone(),
                                is_error,
                                content: result_content,
                                revised: false,
                                node: None,
                            },
                        );
                        self.config
                            .context_manager
                            .append_tool_result(&mut session_state, &tr)
                            .await;
                        result_msg_indices.push(session_state.messages.len().saturating_sub(1));
                        approved_results.push(tr);

                        // #137 (AC2): inject the ONE corrective user message at
                        // the N threshold, AFTER this call's tool result is
                        // recorded — same `append_user_message` channel the
                        // Stop-block breaker uses.
                        if let Some(corrective) = pending_corrective.take() {
                            self.config
                                .context_manager
                                .append_user_message(&mut session_state, &corrective)
                                .await;
                        }
                    }

                    // Middleware: AfterTool (rich chain, issue #11 / SC-9). The
                    // chain receives the batch's results as `&mut` and may rewrite
                    // any of them in place (priority-ordered, descending). On
                    // `ContinueWithModification`, re-render the affected tool-result
                    // messages so the rewrite reaches the next model turn — this is
                    // what lets an after-tool middleware turn a landed write into a
                    // model-visible error (or vice versa) without the tool itself
                    // having to invert its output.
                    if let Some(mw) = self.config.middleware.as_ref() {
                        // Snapshot each result's stream view BEFORE the chain runs,
                        // so on `ContinueWithModification` we re-emit ONLY the ones
                        // it actually rewrote.
                        let pre_views: Vec<(bool, String)> = approved_results
                            .iter()
                            .map(|r| stream_view(&r.output))
                            .collect();
                        match mw.fire_after_tool(&calls, &mut approved_results).await {
                            MiddlewareDecision::Continue => {}
                            MiddlewareDecision::ContinueWithModification => {
                                for ((res, &idx), pre) in approved_results
                                    .iter()
                                    .zip(result_msg_indices.iter())
                                    .zip(pre_views.iter())
                                {
                                    self.config
                                        .context_manager
                                        .replace_tool_result(&mut session_state, idx, res)
                                        .await;
                                    // Symmetric stream-side correction: the model
                                    // side was patched above, but the `ToolResult`
                                    // already on the stream still shows the stale
                                    // content. Re-emit with `revised: true` ONLY for
                                    // results the chain actually changed, so consumers
                                    // correct in place rather than double-count by
                                    // `call_id`.
                                    let (is_error, content) = stream_view(&res.output);
                                    if is_error != pre.0 || content != pre.1 {
                                        Self::emit(
                                            &on_stream,
                                            StreamEvent::ToolResult {
                                                call_id: res.call_id.clone(),
                                                is_error,
                                                content,
                                                revised: true,
                                                node: None,
                                            },
                                        );
                                    }
                                }
                            }
                            MiddlewareDecision::Halt { reason } => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::AfterTool,
                                        reason,
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
                            // `SurfaceToHuman` / `ForceAnotherTurn` are not valid at
                            // `AfterTool`; the StandardMiddlewareChain converts them
                            // to `Halt`. Defensive for a custom chain that emits one.
                            other => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::AfterTool,
                                        reason: format!("illegal AfterTool decision: {other:?}"),
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
                        }
                    }

                    // Compaction (issue #46): after tool results are appended,
                    // before the loop restarts — matches the concepts-doc loop
                    // diagram's "compact if should_compact()" placement. Runs the
                    // verify→retry→warn loop; never halts the run.
                    if self.config.context_manager.should_compact(&session_state) {
                        self.run_compaction(
                            &mut session_state,
                            &session_id,
                            &task.id,
                            &mut span_seq,
                            &mut usage,
                            &agent,
                        )
                        .await;
                    }

                    continue;
                }

                TurnResult::Error { error, usage: u } => {
                    if let Some(u) = u.as_ref() {
                        usage.add_turn(u);
                        budget_used.input_tokens += u.input_tokens as u64;
                        budget_used.output_tokens += u.output_tokens as u64;
                    }
                    return RunResult::Failure {
                        reason: HaltReason::AgentError { error },
                        session_id,
                        usage,
                        turns: budget_used.turns,
                        session_state: session_state.clone(),
                    };
                }
            }
        }
    }

    // ========================================================================
    // SelfVerifying loop strategy (issue #61)
    // ========================================================================
    //
    // Loop-within-a-loop. Each round-trip runs a bounded BUILD ReAct sub-loop
    // (the agent does work until it claims done), then a fresh EVALUATE run
    // (a separate evaluator agent on a read-only sandbox in a never-shared
    // session), then asks the injected [`Verifier`] to translate
    // `(build_result, eval_result)` into a verdict. `Passed` ⇒ Success;
    // `Failed { reason }` ⇒ inject `reason` into the build context and loop.
    //
    // ## Config fields this strategy reads (both default `None`):
    //   - `config.verifier` — the oracle. REQUIRED: `None` ⇒
    //     `Failure { SelfVerifyMisconfigured }` (D4) — a typed halt, NOT a panic.
    //     Its `max_iterations()` (default 3) caps the round-trips (D3); per-run
    //     `max_stop_blocks` does NOT enter the picture for this strategy.
    //   - `config.evaluator_agent` — the evaluate-phase agent. Defaulting
    //     contract (D2): `config.evaluator_agent.as_ref().unwrap_or(&config.agent)`
    //     (identical to `planner_agent`). The read-only sandbox and the fresh
    //     `SessionId::generate()` for the evaluate phase are derived INTERNALLY.
    //
    // ## Terminal `HaltReason` variants this strategy produces (peers, D4):
    //   - `SelfVerifyExhausted { iterations, last_reason }` — ran out of
    //     `max_iterations` round-trips without a pass (clean exhaustion).
    //   - `SelfVerifyMisconfigured { reason }` — `config.verifier` is `None`.
    //
    // ## Rules enforced (each maps to a test):
    //   - R1  build phase runs a ReAct loop until the agent claims done.
    //   - R2  evaluate phase uses a FRESH SessionId never shared with build.
    //   - R3  evaluate phase uses a read-only sandbox (writes ⇒ ReadOnlyViolation;
    //         the build sandbox is unaffected).
    //   - R4  the evaluator carries the `role-evaluator` chunk. PRESENCE-ONLY:
    //         the chunk's content (if the provider supplies it) is prepended to
    //         the evaluate seed message; the chunk-condition machinery (#79) is
    //         NOT otherwise wired into the harness loop.
    //   - R5  Default-FAIL: an indeterminate evaluator verdict keeps looping.
    //   - R6  on findings, the verdict reason is injected into the build context
    //         and the build loop resumes (same path the Stop-block injection uses).
    //   - R7  stagnation guard: always-Failed ⇒ exactly `max_iterations` cycles ⇒
    //         `Failure { SelfVerifyExhausted }`.
    //   - R8  budgets fold BOTH phases across ALL iterations.
    //   - R9  build vs evaluate are distinguishable in traces (distinct session
    //         ids: the build session vs each evaluate's generated session).
    //   - R11 `verifier == None` ⇒ `Failure { SelfVerifyMisconfigured }`.
    //
    // No `// SPEC QUESTION:` markers — all four forks (D1–D4) are resolved.
}


// ── RunStrategy: the ReAct leaf body (#124, moved from strategies.rs) ──
impl RunStrategy for ReactConfig {
    /// The leaf: a bounded ReAct turn-loop window. Reads the per-run scratch
    /// (`task`, `run_session`, `run_budget`) and drives one ReAct window through
    /// the executor primitive, threading the result back into the shared usage.
    ///
    /// ## Budget enforcement (#125)
    ///
    /// The leaf pushes its OWN [`BudgetContext`] scope for `self.budget` and
    /// charges the turns the window consumed against it. Per Open Q A-2 (rule 6)
    /// the leaf does NOT carry its own behavior and NEVER self-resolves a
    /// Continue/Fail/Escalate at the leaf — on charge exhaustion it PROPAGATES a
    /// `StrategyOutcome::BudgetExhausted` to its parent (which owns the single
    /// recovery site). The propagated `partial_output` is the window's last
    /// FinalResponse text as JSON (fork #2). A clean (non-budget) terminal is
    /// recorded verbatim through [`record_terminal`](ExecutionContext::record_terminal).
    fn run<'a>(&'a self, cx: &'a mut ExecutionContext<'_>) -> BoxFut<'a, StrategyOutcome> {
        Box::pin(async move {
            let executor = match cx.executor() {
                Ok(e) => e,
                Err(o) => return o,
            };
            let task = cx.current_task();
            let max_iterations = self.max_iterations();
            let mut session_state = std::mem::take(&mut cx.scratch.run_session);
            let budget_used = cx.scratch.run_budget.clone();
            // #124: resolve the worker agent from the registry by THIS leaf's
            // handle (genuine recursion — no `config.agent`). A missing handle is
            // a typed terminal failure.
            let agent = match executor.resolve_agent_ref(&self.agent, &task.session_id) {
                Ok(a) => a,
                Err(result) => {
                    let result = *result;
                    executor.finalize(&result).await;
                    return cx.record_terminal(result);
                }
            };
            // Output-schema delivery + enforcement (issue #139). MIGRATION GATE:
            // only when `enforce_output_schemas` is ON AND this leaf carries
            // `output = Some(..)` do we resolve the schema, DELIVER it (directive
            // seed + the constrained-decoding channel), and pass it (plus the
            // retry budget `N`) into the window for terminal validation. When the
            // gate is OFF or there is no `output`, `output_schema` is `None` and
            // the window behaves byte-identically to pre-#139 (no resolve, no
            // delivery, no validation). The schema is canonicalized to compact
            // key-sorted JSON so its delivered/reported bytes are identical
            // across the four language ports.
            // A FORCED schema (the plan-empty repair re-ask, #139) wins over the
            // global enforce gate and is consumed one-shot; otherwise fall back to
            // the normal `enforce_output_schemas` + `self.output` resolution.
            let output_schema: Option<serde_json::Value> =
                cx.scratch.forced_output_schema.take().or_else(|| {
                    if executor.enforce_output_schemas() {
                        self.output
                            .as_ref()
                            .and_then(|r| cx.registry.resolve_schema(r))
                            .cloned()
                    } else {
                        None
                    }
                });
            let output_schema_max_retries = executor.output_schema_max_retries();
            if let Some(schema) = output_schema.as_ref() {
                // AC1: append the resolved schema to the leaf's directive/system
                // context as a USER message, key-sorted via `canonicalize_json`
                // so the seeded bytes are identical across languages.
                let directive = format!(
                    "Your final response must be a JSON value that conforms to this \
JSON schema: {}",
                    crate::model::canonicalize_json(schema)
                );
                executor
                    .seed_user_message(&mut session_state, &directive)
                    .await;
            }
            // #125/#129: push this leaf's OWN budget scope carrying its CONFIGURED
            // `behavior`. The leaf still never RESOLVES it in the nested case (rule
            // 6: it PROPAGATES a `BudgetExhausted` to its parent, which owns the
            // single recovery site). Carrying the real `behavior` only means the
            // propagated error reports it, so the TOP-LEVEL/bare-leaf resolution
            // site (`drive_strategy`) can honor it (Q1 — a bare leaf self-resolves,
            // a nested leaf does not).
            cx.push_budget(self.budget.clone(), self.behavior.clone(), "react");
            // The leaf CLONES (not takes) the run's stream sink so sibling phases
            // of a combinator parent still inherit it for their own turns, and
            // WRAPS it so every event this leaf emits is attributed to THIS node
            // (kind `react` + the resolved agent handle). The wrapper is what the
            // live agent adapter forwards to, so token deltas carry attribution too.
            let agent_id = (!self.agent.0.is_empty()).then(|| self.agent.0.clone());
            let on_stream = wrap_node_sink(cx.stream.clone(), "react", agent_id);
            // Consume the one-shot execute-step marker (set by the execute walk) so
            // the behavioral tool-call nudge scopes to execute steps only and never
            // leaks into a nested leaf.
            let execute_step = std::mem::take(&mut cx.scratch.in_execute_step);
            let result = executor
                .react_window(ReactParams {
                    task,
                    max_iterations,
                    session_state,
                    budget_used,
                    on_stream,
                    agent,
                    // Issue 2: thread THIS leaf's toolset handle down so the
                    // window dispatches the per-node scoped catalogue (empty
                    // handle ⇒ global-catalogue fallback). Mirrors `agent`.
                    toolset: self.toolset.clone(),
                    // Issue #139: thread the resolved output schema (or `None`)
                    // and the retry budget so the window validates the terminal.
                    output_schema,
                    output_schema_max_retries,
                    // SC-10: thread THIS leaf's per-node system-prompt override
                    // (None ⇒ the window keeps using the global prompt).
                    system_prompt: self.system_prompt.clone(),
                    execute_step,
                })
                .await;
            executor.finalize(&result).await;

            // #125: charge the window's turns against this leaf's OWN scope. The
            // leaf POLICY (`self.budget`) — not the global `BudgetLimits` backstop —
            // is the per-node enforcement point. A window stops at
            // `min(max_iterations, task.budget.max_turns)`: when the LEAF cap is the
            // binding constraint (the window consumed >= the leaf policy value), the
            // leaf is exhausted and PROPAGATES a typed `BudgetExhausted` to its
            // parent (rule 6 — the leaf never self-resolves). When the smaller
            // GLOBAL backstop trips first, the legacy `BudgetExceeded` terminal is
            // recorded VERBATIM (the global cap is unchanged, #117 backstop).
            let window_turns = match &result {
                RunResult::Success { turns, .. } | RunResult::Failure { turns, .. } => *turns,
                _ => 0,
            };
            // #137: the window hit the consecutive-tool-error breaker's 2N hard
            // stop. PROPAGATE it through the SAME single budget-exhaustion
            // resolution site (so the node's `behavior` governs Fail/Escalate/
            // Continue), but carry the `ToolErrorLoop` cause so the terminal
            // reports `HaltReason::ToolErrorLoop`, never `BudgetExceeded`. The
            // window's turns are still CHARGED against the leaf scope (accurate
            // accounting), but the breaker stopped EARLY — the remaining budget
            // is NOT burned. Detection is independent of `leaf_cap_binding`.
            if let RunResult::Failure {
                reason:
                    HaltReason::ToolErrorLoop {
                        tool,
                        consecutive_errors,
                    },
                ..
            } = &result
            {
                let tool = tool.clone();
                let consecutive_errors = *consecutive_errors;
                let last_final = last_final_response_text(&result).unwrap_or_default();
                if let RunResult::Failure { session_state, .. } = &result {
                    cx.scratch.run_session = session_state.clone();
                }
                let _ = cx.charge_current(window_turns);
                cx.pop_budget();
                return promote_tool_error_loop(
                    &self.budget,
                    &self.behavior,
                    window_turns,
                    tool,
                    consecutive_errors,
                    Some(react_partial_json(&last_final)),
                );
            }
            let window_hit_budget = matches!(
                &result,
                RunResult::Failure {
                    reason: HaltReason::BudgetExceeded { .. },
                    ..
                }
            );
            // The leaf cap is binding when the window hit a budget terminal AND it
            // consumed at least the leaf's own allowance (so the leaf policy — not a
            // smaller global backstop — is what stopped it).
            let leaf_cap_binding = window_hit_budget
                && matches!(self.budget.allowance_value(), Some(cap) if window_turns >= cap);
            let charge = cx.charge_current(window_turns);
            if leaf_cap_binding || charge.is_err() {
                // The leaf partial is the window's last FinalResponse text.
                let last_final = last_final_response_text(&result).unwrap_or_default();
                // Carry the post-run session so a parent resumes losslessly.
                if let RunResult::Success { session_state, .. }
                | RunResult::Failure { session_state, .. } = &result
                {
                    cx.scratch.run_session = session_state.clone();
                }
                let err = charge.err().unwrap_or_else(|| {
                    // The window itself hit the cap; synthesize the charge error
                    // from the current scope state.
                    cx.budgets
                        .current_mut()
                        .map(|s| BudgetExhausted {
                            policy: s.policy.clone(),
                            behavior: s.behavior.clone(),
                            steps_taken: s.steps_taken,
                            continues_used: s.continues_used,
                            phase: s.phase.clone(),
                        })
                        .unwrap_or_else(|| BudgetExhausted {
                            policy: self.budget.clone(),
                            behavior: self.behavior.clone(),
                            steps_taken: window_turns,
                            continues_used: 0,
                            phase: "react".to_string(),
                        })
                });
                cx.pop_budget();
                // Rule 6: the leaf PROPAGATES — partial carries the last
                // FinalResponse (Escalate semantics, fork #1/#2).
                return promote_budget_exhausted(&err, Some(react_partial_json(&last_final)));
            }
            cx.pop_budget();
            cx.record_terminal(result)
        })
    }
}
