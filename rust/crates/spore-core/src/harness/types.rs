//! Value types and small component traits exchanged by the harness, extracted
//! from the god-module (Phase 6.1): run/turn/tool data types, sandbox/termination
//! enums, escalation/human/consult types, `PausedState`, `RunResult`, and the
//! `ContextManager`/`TerminationPolicy`/`Harness` traits. Pure move.

use super::*;
impl RunResult {
    /// True for the non-terminal pause / clean-handoff variants
    /// (`WaitingForHuman` / `Consult` / `Escalate`) that a combinator must
    /// propagate VERBATIM rather than fold into its own terminal (#124).
    pub fn is_pause(&self) -> bool {
        matches!(
            self,
            RunResult::WaitingForHuman { .. }
                | RunResult::Consult { .. }
                | RunResult::Escalate { .. }
        )
    }
}


/// Lightweight placeholder for an alternate planner model selection. Full
/// shape lives wherever provider routing lands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub provider: String,
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub instruction: String,
    pub session_id: SessionId,
    pub budget: BudgetLimits,
    pub loop_strategy: LoopStrategy,
}

impl Task {
    pub fn new(
        instruction: impl Into<String>,
        session_id: SessionId,
        loop_strategy: LoopStrategy,
    ) -> Self {
        Self {
            id: TaskId::generate(),
            instruction: instruction.into(),
            session_id,
            budget: BudgetLimits::default(),
            loop_strategy,
        }
    }
    pub fn with_budget(mut self, budget: BudgetLimits) -> Self {
        self.budget = budget;
        self
    }

    /// A one-shot task from just an instruction: a fresh [`SessionId`] and a
    /// default `ReAct` loop. Use [`new`](Self::new) when you need to control the
    /// session id (e.g. multi-turn) or the loop strategy.
    pub fn simple(instruction: impl Into<String>) -> Self {
        Self::new(
            instruction,
            SessionId::generate(),
            LoopStrategy::ReAct(ReactConfig::per_loop(8)),
        )
    }
}

// ============================================================================
// Streaming events emitted by the harness
// ============================================================================

/// The kind of content block a [`StreamEvent::BlockStart`] opens (issue #103,
/// resolved spec decision **Q2**: a single generic frame marker carrying a
/// `BlockKind` rather than typed-per-kind markers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    /// A text content block (`model::StreamEvent::ContentBlockDelta`).
    Text,
    /// A reasoning / thinking block (`model::StreamEvent::ThinkingDelta`).
    Reasoning,
    /// A tool-use block whose arguments arrive as
    /// `model::StreamEvent::ToolUseDelta` fragments.
    ToolUse,
}

/// Events the harness surfaces on [`StreamSink`].
///
/// ## Delta-level streaming (issue #103)
///
/// The harness maps each raw [`crate::model::StreamEvent`] produced by the
/// agent through [`Harness::map_model_stream_event`] into zero or one of the
/// delta/frame variants below, alongside the existing coarse lifecycle events.
/// Resolution notes for the six spec ambiguities:
///
/// - **Q2**: frame markers are the generic [`StreamEvent::BlockStart`] /
///   [`StreamEvent::BlockStop`] carrying a [`BlockKind`], NOT typed-per-kind
///   markers.
/// - **Q3**: `model::StreamEvent::MessageStart` / `MessageStop` are DROPPED at
///   the harness boundary (mapped to `None`). `TurnStart` / `TurnEnd` already
///   cover message lifecycle.
/// - **Q5**: the coarse [`StreamEvent::ToolCall`] now also carries the final
///   `args`, and [`StreamEvent::ToolResult`] the result `content`. Both new
///   fields use `#[serde(default)]` so pre-#103 serialized events round-trip.
///
/// Tool lifecycle ordering per call:
/// [`StreamEvent::ToolCallStart`] → [`StreamEvent::ToolArgsDelta`]\* →
/// (`BlockStop`) → coarse [`StreamEvent::ToolCall`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamEvent {
    TurnStart {
        turn: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    TurnEnd {
        turn: u32,
        /// This turn's token usage, so a live consumer (the UI token chip) can
        /// update PER-TURN instead of only at end-of-run. `#[serde(default)]` keeps
        /// pre-existing serialized events / other-language consumers compatible.
        #[serde(default)]
        usage: crate::model::TokenUsage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    ToolCall {
        call_id: String,
        name: String,
        /// Final, fully-accumulated tool-call arguments (issue #103, Q5).
        /// `#[serde(default)]` keeps pre-#103 serialized events back-compatible.
        #[serde(default)]
        args: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    ToolResult {
        call_id: String,
        is_error: bool,
        /// The tool result content (issue #103, Q5). `#[serde(default)]` keeps
        /// pre-#103 serialized events back-compatible.
        #[serde(default)]
        content: String,
        /// `true` when this event REPLACES a `ToolResult` already emitted for the
        /// same `call_id` because an `AfterTool` middleware rewrote the result in
        /// place (`ContinueWithModification`, issue #11 / SC-9). The model side is
        /// patched via [`ContextManager::replace_tool_result`]; this is the
        /// symmetric stream-side signal so consumers correct the displayed result
        /// instead of double-counting it as a second call. `#[serde(default)]`
        /// keeps consumers that ignore the field back-compatible — they simply see
        /// the corrected content, which is strictly better than the stale one.
        #[serde(default)]
        revised: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    FinalResponse {
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    BudgetWarning {
        limit_type: BudgetLimitType,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    /// Emitted when a tool wants to surface a message to the user out-of-band
    /// (issue #81 — `SendMessageTool`), OR when a queued steering message is
    /// injected into the run (`HarnessConfig::steering_source`). The harness
    /// loop recognizes the `send_message` tool, emits this event instead of
    /// collapsing the content into a normal tool result, and records a minimal
    /// success tool result so the loop continues.
    UserMessage {
        content: String,
        /// `true` when this is a USER-authored steering message injected at a
        /// turn boundary (rendered as user input), `false` for the agent's
        /// `send_message` narration (rendered as agent output). The roles are
        /// opposite, so consumers must branch on it. `#[serde(default)]` keeps
        /// pre-steering serialized events (all narration) deserializing.
        #[serde(default)]
        steered: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    // ── Delta-level streaming (issue #103) ──────────────────────────────────
    /// Streamed text fragment (`model::StreamEvent::ContentBlockDelta`).
    TextDelta {
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    /// Streamed reasoning/thinking fragment
    /// (`model::StreamEvent::ThinkingDelta`). Q4.
    ReasoningDelta {
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    /// Streamed tool-argument JSON fragment
    /// (`model::StreamEvent::ToolUseDelta`), correlated to a `call_id` via the
    /// open-block index.
    ToolArgsDelta {
        call_id: String,
        partial_json: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    /// A content block opened (Q2). Emitted the first time a delta for an index
    /// is seen.
    BlockStart {
        index: u32,
        block: BlockKind,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    /// A content block closed (`model::StreamEvent::ContentBlockStop`). Q2.
    BlockStop {
        index: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    /// A tool-use block opened (issue #103). Emitted so consumers can correlate
    /// the subsequent [`StreamEvent::ToolArgsDelta`] fragments and the final
    /// coarse [`StreamEvent::ToolCall`] by `call_id`. The `name` may be empty
    /// when the underlying model stream does not surface it before args (a
    /// documented `model::StreamEvent` limitation — name is recovered on the
    /// coarse `ToolCall`).
    ToolCallStart {
        index: u32,
        call_id: String,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    /// The consecutive-recoverable-tool-error breaker DETECTED a loop (issue
    /// #137): a tool hit `error_loop_threshold` (`N`) consecutive
    /// identical-argument recoverable errors and ONE corrective message was
    /// injected. A warning — the loop continues. Carries the tool name and the
    /// consecutive-error count (`= N`).
    ToolErrorLoopDetected {
        tool: String,
        consecutive_errors: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    /// The consecutive-recoverable-tool-error breaker TRIPPED (issue #137): the
    /// same tool reached `2 * error_loop_threshold` (`2 * N`) consecutive
    /// identical-argument errors and the loop STOPPED to resolve the node's
    /// `BudgetExhaustedBehavior`. Carries the tool name and the count (`= 2*N`).
    ToolErrorLoopBroken {
        tool: String,
        consecutive_errors: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    /// Output-schema enforcement (issue #139) fed a validation error back and
    /// RETRIED: the terminal `FinalResponse` failed validation against the
    /// leaf's `output` schema and a retry turn was granted (within budget). A
    /// warning — the loop continues. Carries the extra-retry count so far
    /// (`= attempt`) and the frozen validator error fed back.
    OutputSchemaRetry {
        attempt: u32,
        error: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
    /// Output-schema enforcement (issue #139) EXHAUSTED its retries: the
    /// terminal still failed validation after `output_schema_max_retries` extra
    /// turns (with budget remaining), so the run terminates with
    /// [`HaltReason::OutputSchemaViolation`]. Carries the total attempt count
    /// (`= 1 + max_retries`) and the final frozen validator error.
    OutputSchemaViolation {
        attempts: u32,
        error: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node: Option<NodeAttr>,
    },
}

impl StreamEvent {
    /// Uniform mutable access to the optional [`NodeAttr`] on any variant — the
    /// single seam the node-wrapping sinks stamp through ([`wrap_node_sink`]).
    /// One or-pattern binds the common `node` field across every variant.
    pub fn node_mut(&mut self) -> &mut Option<NodeAttr> {
        match self {
            StreamEvent::TurnStart { node, .. }
            | StreamEvent::TurnEnd { node, .. }
            | StreamEvent::ToolCall { node, .. }
            | StreamEvent::ToolResult { node, .. }
            | StreamEvent::FinalResponse { node, .. }
            | StreamEvent::BudgetWarning { node, .. }
            | StreamEvent::UserMessage { node, .. }
            | StreamEvent::TextDelta { node, .. }
            | StreamEvent::ReasoningDelta { node, .. }
            | StreamEvent::ToolArgsDelta { node, .. }
            | StreamEvent::BlockStart { node, .. }
            | StreamEvent::BlockStop { node, .. }
            | StreamEvent::ToolCallStart { node, .. }
            | StreamEvent::ToolErrorLoopDetected { node, .. }
            | StreamEvent::ToolErrorLoopBroken { node, .. }
            | StreamEvent::OutputSchemaRetry { node, .. }
            | StreamEvent::OutputSchemaViolation { node, .. } => node,
        }
    }

    /// The emitting node's attribution, if stamped.
    pub fn node(&self) -> Option<&NodeAttr> {
        match self {
            StreamEvent::TurnStart { node, .. }
            | StreamEvent::TurnEnd { node, .. }
            | StreamEvent::ToolCall { node, .. }
            | StreamEvent::ToolResult { node, .. }
            | StreamEvent::FinalResponse { node, .. }
            | StreamEvent::BudgetWarning { node, .. }
            | StreamEvent::UserMessage { node, .. }
            | StreamEvent::TextDelta { node, .. }
            | StreamEvent::ReasoningDelta { node, .. }
            | StreamEvent::ToolArgsDelta { node, .. }
            | StreamEvent::BlockStart { node, .. }
            | StreamEvent::BlockStop { node, .. }
            | StreamEvent::ToolCallStart { node, .. }
            | StreamEvent::ToolErrorLoopDetected { node, .. }
            | StreamEvent::ToolErrorLoopBroken { node, .. }
            | StreamEvent::OutputSchemaRetry { node, .. }
            | StreamEvent::OutputSchemaViolation { node, .. } => node.as_ref(),
        }
    }
}

/// A user-facing stream sink. `Arc` (not `Box`) so the single sink installed at
/// the top of a run can be CLONED into every nested node's context and into the
/// live per-turn agent adapter without being consumed — this is what lets a
/// composed strategy tree stream every node's events to one channel (the API
/// the harness backs subscribes to exactly this sink and forwards over SSE).
pub type StreamSink = Arc<dyn Fn(StreamEvent) + Send + Sync>;

/// The fully-resolved inputs to one ReAct turn-loop window. Bundles what used
/// to be the long positional argument list of `run_react_inner` /
/// [`StrategyExecutor::react_window`] into a single value so the turn-loop
/// entry points stay under the argument-count lint without an allow, and so the
/// per-leaf knobs (#139 output schema, SC-10 prompt override, Issue 2 toolset,
/// the execute-step marker) travel together rather than as a row of `None`s and
/// booleans. Constructed by the caller (`ReactConfig::run`, the legacy
/// `run_react` wrapper); consumed once when the window starts.
pub struct ReactParams {
    /// The task this window runs (carries `session_id`, instruction, budget).
    pub task: Task,
    /// Hard turn ceiling for the window (the leaf's `max_iterations`).
    pub max_iterations: u32,
    /// Conversation state threaded in; mutated in place across turns.
    pub session_state: SessionState,
    /// Budget consumed so far, carried across nested windows.
    pub budget_used: BudgetSnapshot,
    /// Optional user-facing stream sink (cloned per turn for live deltas).
    pub on_stream: Option<StreamSink>,
    /// The RESOLVED worker agent for this leaf (#124).
    pub agent: Arc<dyn Agent>,
    /// The RESOLVED per-node toolset handle (Issue 2). Empty ⇒ global fallback.
    pub toolset: ToolsetRef,
    /// The RESOLVED output schema to enforce (#139). `None` ⇒ no enforcement.
    pub output_schema: Option<serde_json::Value>,
    /// Extra validation-retry turns allowed within budget (#139).
    pub output_schema_max_retries: u32,
    /// Per-node system-prompt OVERRIDE (SC-10). `None` ⇒ the global prompt.
    pub system_prompt: Option<String>,
    /// Whether this window is a PlanExecute EXECUTE step (scopes the nudge).
    pub execute_step: bool,
}

impl ReactParams {
    /// Build the inputs for a LEGACY `run_react` window — the single-shot / DAG
    /// execute-step / SelfVerifying-evaluate path that does NOT enforce output
    /// schemas, carries no per-leaf prompt override, and is not an execute step.
    /// Fills the #139 / SC-10 / execute-step knobs with their byte-for-byte
    /// pre-feature defaults so callers pass only the seven values they vary.
    pub fn legacy(
        task: Task,
        max_iterations: u32,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
        agent: Arc<dyn Agent>,
        toolset: ToolsetRef,
    ) -> Self {
        Self {
            task,
            max_iterations,
            session_state,
            budget_used,
            on_stream,
            agent,
            toolset,
            output_schema: None,
            output_schema_max_retries: 0,
            system_prompt: None,
            execute_step: false,
        }
    }
}

/// Attribution stamped onto every [`StreamEvent`] as it flows up through the
/// node-wrapping sinks (see [`wrap_node_sink`]). Lets a multi-agent UI route a
/// frame to the node that produced it. Optional + `skip_serializing_if` so an
/// un-attributed event (a bare/single-leaf run, or any pre-attribution
/// consumer) serializes byte-identically to before this field existed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeAttr {
    /// The emitting node's kind: `"react"` (a leaf turn loop), or a combinator
    /// kind (`"plan_execute"`, `"self_verifying"`, `"ralph"`, `"hill_climbing"`).
    /// The leaf's kind is the innermost (most specific) — set first.
    pub kind: String,
    /// The leaf agent's handle (e.g. `"planner"`/`"executor"`/`"ralph-agent"`),
    /// when the emitting node is a `react` leaf. `None` for combinator frames.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Nesting depth of the emitting leaf below the root (root combinator's leaf
    /// children are progressively deeper). Accumulated as the event passes
    /// through each combinator wrapper.
    pub depth: u32,
    /// Root→leaf node-kind path, accumulated by the wrapper chain (outermost
    /// first). E.g. `["ralph", "plan_execute", "self_verifying", "react"]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub path: Vec<String>,
}

/// Wrap a parent sink so every event passing through is stamped for ONE node.
/// The leaf installs the innermost wrapper (it sets `kind`/`agent_id`/seeds the
/// path); each enclosing combinator's wrapper PREPENDS its kind to `path` and
/// bumps `depth`. Stamping is set-once for `kind`/`agent_id` (the innermost leaf
/// wins) so the event reports who actually spoke, with the full nesting in
/// `path`. Returns `None` when there is no parent sink (no-op, baseline path).
pub(crate) fn wrap_node_sink(
    parent: Option<StreamSink>,
    kind: &str,
    agent_id: Option<String>,
) -> Option<StreamSink> {
    let parent = parent?;
    let kind = kind.to_string();
    Some(Arc::new(move |mut ev: StreamEvent| {
        match ev.node_mut() {
            slot @ None => {
                *slot = Some(NodeAttr {
                    kind: kind.clone(),
                    agent_id: agent_id.clone(),
                    depth: 0,
                    path: vec![kind.clone()],
                });
            }
            Some(attr) => {
                attr.path.insert(0, kind.clone());
                attr.depth += 1;
            }
        }
        parent(ev);
    }))
}

/// Per-turn state threaded through [`Harness::map_model_stream_event`] (issue
/// #103). Tracks which block indices are open and their kind so
/// `ToolUseDelta` / `ContentBlockStop` events can be correlated back to a
/// `call_id`, and so each block's `BlockStart` is emitted exactly once.
#[derive(Debug, Default)]
pub struct TurnStreamState {
    /// Open block index → its [`BlockKind`].
    pub(crate) open_blocks: std::collections::HashMap<u32, BlockKind>,
    /// Tool-use block index → its derived `call_id`. The `call_id` is derived
    /// as `call_{index}`, matching the id synthesized by the agent's streaming
    /// accumulator so the coarse `ToolCall` correlates.
    pub(crate) tool_calls: std::collections::HashMap<u32, String>,
}

impl TurnStreamState {
    pub(crate) fn call_id_for(index: u32) -> String {
        format!("call_{index}")
    }
}

// ============================================================================
// Forward-declared sibling component types
// (full surfaces live in their owning issues)
// ============================================================================

/// Tool Escalation Protocol — the typed channel by which a tool signals the
/// harness to terminate cleanly and pass a *structural* state change up to its
/// caller (issue #80).
///
/// The harness is a pure intermediary: it never acts on a signal itself. Mode
/// switching, plan approval, and graceful abort are the caller's concern. The
/// harness terminates cleanly, surfaces the signal via
/// [`RunResult::Escalate`], and the caller (CLI, chat UI, REST API, parent
/// harness) owns the orchestration. This mirrors the [`RunResult::WaitingForHuman`]
/// model — the harness does not resume itself either.
///
/// ## Variants
/// - [`HarnessSignal::EnterPlanMode`] — agent requests entry into plan mode,
///   carrying accumulated context as a seed for the planning harness.
/// - [`HarnessSignal::ExitPlanMode`] — planning agent's terminal signal,
///   carrying the produced [`PlanArtifact`](crate::plan::PlanArtifact) for
///   human approval before an execution harness is instantiated.
/// - [`HarnessSignal::SwitchMode`] — agent requests a mode switch; carries the
///   target [`Mode`](crate::prompt_chunk_registry::Mode) (the EXISTING mode
///   enum — there is no separate `HarnessMode` type).
/// - [`HarnessSignal::Abort`] — agent requests a graceful, intentional stop
///   with a reason. Distinct from `HaltReason::AgentError` — it surfaces as
///   `RunResult::Escalate`, NOT `RunResult::Failure`.
///
/// Wire format: serde-tagged on `kind`, snake_case, byte-identical across the
/// four language implementations. Round-tripped by
/// `fixtures/harness/escalation_signals.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HarnessSignal {
    /// Agent requests entry into plan mode. Carries context the agent has
    /// accumulated so far as seed for the planning harness.
    EnterPlanMode { context: String },
    /// Planning agent has produced a plan and requests exit from plan mode.
    /// Carries the plan artifact for human approval before the execution
    /// harness is instantiated. This is the planning agent's terminal signal.
    ExitPlanMode { plan: crate::plan::PlanArtifact },
    /// Agent requests a mode switch. The caller instantiates the appropriate
    /// harness for the new mode.
    SwitchMode {
        mode: crate::prompt_chunk_registry::Mode,
    },
    /// Agent requests a graceful abort with a reason surfaced to the user.
    /// Distinct from `HaltReason::AgentError` — this is an intentional,
    /// agent-initiated stop and surfaces as `RunResult::Escalate`.
    Abort { reason: String },
    /// The planner recorded its plan via `set_plan` and is done planning — END the
    /// PlanExecute plan phase (the plan-phase analogue of `plan_completed`) so the
    /// DAG walk proceeds to EXECUTE. Carries no artifact: the runnable plan is
    /// already in the durable `task_list` store the plan phase reads. Unlike
    /// `Abort`, it is CONSUMED inside PlanExecute and never ends the whole run.
    PlanRecorded,
}

/// Output of a single tool dispatch. Full type lives in issue #4 (ToolRegistry)
/// / #5 (Tool). The variants below cover what the harness loop needs to
/// route; richer payloads are additive.
///
/// The [`ToolOutput::Escalate`] variant is the tool-side entry point of the
/// Tool Escalation Protocol (issue #80): when a dispatched tool returns it, the
/// harness terminates cleanly (NOT appending the escalation to message
/// history — it is a control signal, not a conversation turn) and returns
/// [`RunResult::Escalate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolOutput {
    Success {
        content: String,
        /// `true` only when the tool itself clipped its output to fit an inline
        /// budget (large outputs routed through
        /// [`SandboxProvider::handle_large_output`] set this). Plain tool
        /// authors should leave it `false` — use [`ToolOutput::success`].
        #[serde(default)]
        truncated: bool,
    },
    Error {
        message: String,
        /// `true` if the agent may sensibly retry or adapt: the loop appends the
        /// error as a tool result and continues. `false` halts the run. Most
        /// tool failures are recoverable — prefer [`ToolOutput::error`]; reach
        /// for [`ToolOutput::fatal`] only when continuing is pointless.
        recoverable: bool,
    },
    /// Boxed because `ChildPausedState` is significantly larger than the other
    /// variants; keeps `ToolOutput` cheap to clone on the happy path.
    WaitingForHuman {
        child_state: Box<ChildPausedState>,
        request: HumanRequest,
    },
    /// Tool requests a structural state change from the harness's parent
    /// (issue #80). The harness terminates cleanly and passes the signal to
    /// the caller via [`RunResult::Escalate`]. The escalation is NOT appended
    /// to message history.
    Escalate { signal: HarnessSignal },
    /// Tool requests a clarifying answer from the human before it can produce a
    /// result (issue #81, Q4b — `AskUserQuestionTool`). UNLIKE
    /// [`ToolOutput::WaitingForHuman`], there is NO [`ChildPausedState`]: the
    /// harness loop builds a [`PausedState`] directly, sets its `human_request`
    /// to [`HumanRequest::Clarification`], and returns
    /// [`RunResult::WaitingForHuman`]. The clarifying tool call is preserved in
    /// `pending_tool_calls`; on resume the human's [`HumanResponse::Answer`]
    /// text is injected as the tool RESULT for that pending call (not appended
    /// as a free-standing user message) and the loop continues.
    AwaitingClarification {
        question: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        options: Option<Vec<String>>,
    },
    /// Mid-loop consult signal (issue #114). A worker-side tool returns it with
    /// `child_state: None`; the worker harness pauses and returns
    /// [`RunResult::Consult`] with the consult call preserved as the head of
    /// `pending_tool_calls` (`human_request: None`). At the subagent boundary,
    /// [`SubagentTool`](crate::tools::SubagentTool) populates `child_state:
    /// Some(..)` — but in the A1 mediation seam it consumes the signal itself
    /// rather than bubbling it, so a parent orchestrator never observes this
    /// variant on the happy path. Mirrors [`ToolOutput::WaitingForHuman`]: the
    /// boxed `ChildPausedState` keeps the common case cheap to clone.
    Consult {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        child_state: Option<Box<ChildPausedState>>,
        request: ConsultRequest,
    },
    /// A tool was denied by the sandbox (e.g. a path escaped the workspace root,
    /// or a command was disallowed). Carries the typed [`SandboxViolation`] so
    /// the harness — not the tool — decides whether it halts the run or is fed
    /// back to the model as a recoverable error, per
    /// [`HarnessConfig::sandbox_violation_policy`]. Produced by the
    /// `From<ToolExecutionError>` conversion; the harness normalizes it into a
    /// halt or a recoverable [`ToolOutput::Error`] immediately after dispatch,
    /// so it never reaches message history.
    SandboxViolation { violation: SandboxViolation },
}

impl ToolOutput {
    /// A successful, non-truncated result. The common case for a tool that
    /// returns its full output — saves spelling out `truncated: false`.
    pub fn success(content: impl Into<String>) -> Self {
        ToolOutput::Success {
            content: content.into(),
            truncated: false,
        }
    }

    /// A **recoverable** error: the harness loop appends it as a tool result and
    /// lets the agent adapt or retry. This is the right default for almost every
    /// tool failure (bad arguments, missing file, transient I/O).
    pub fn error(message: impl Into<String>) -> Self {
        ToolOutput::Error {
            message: message.into(),
            recoverable: true,
        }
    }

    /// A **fatal** error: continuing is pointless, so the run halts. Reserve for
    /// genuinely unrecoverable conditions; prefer [`ToolOutput::error`] when the
    /// agent could reasonably do something different next turn.
    pub fn fatal(message: impl Into<String>) -> Self {
        ToolOutput::Error {
            message: message.into(),
            recoverable: false,
        }
    }

    /// A **worker-side consult** signal (issue #114): the tool asks for mid-loop
    /// help. `child_state` is `None` — the harness loop builds the
    /// [`RunResult::Consult`] pause; only `SubagentTool` populates
    /// `child_state` at the boundary.
    pub fn consult(request: ConsultRequest) -> Self {
        ToolOutput::Consult {
            child_state: None,
            request,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub output: ToolOutput,
}

/// Sandbox-side violation. Full enum lives in issue #6 (SandboxProvider).
/// `PathEscape` and `NetworkViolation` are the halt-ELIGIBLE Layer-1 variants
/// (see [`is_always_halt`](Self::is_always_halt)); whether they actually halt is
/// decided by [`SandboxViolationPolicy`]. The remaining variants are always
/// recoverable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SandboxViolation {
    PathEscape {
        path: String,
    },
    NetworkViolation {
        host: String,
    },
    PathDenied {
        path: String,
        #[serde(default)]
        matched_rule: String,
    },
    ReadOnlyViolation {
        path: String,
    },
    ExtensionDenied {
        path: String,
        extension: String,
    },
    FileSizeExceeded {
        path: String,
        size: u64,
        limit: u64,
    },
    DisallowedCommand {
        command: String,
    },
    /// The process could not be spawned at all (e.g. the command binary does
    /// not exist, or the OS refused the spawn). SC-15: this is reported as a
    /// typed violation rather than a fake `CommandOutput { exit_code: -1 }`, so
    /// "the command ran and exited -1" and "the command never started" are no
    /// longer conflated. Always recoverable (never halt-eligible).
    ExecSpawnFailed {
        command: String,
        message: String,
    },
}

impl SandboxViolation {
    /// The halt-ELIGIBLE Layer-1 violations: a path escaping the workspace root
    /// or a blocked network host. Whether such a violation actually halts the
    /// run is governed by [`SandboxViolationPolicy`]; under the default
    /// (`Recoverable`) even these are fed back to the model. The Layer-2
    /// variants (`PathDenied`, `ReadOnlyViolation`, …) are never halt-eligible —
    /// they are always recoverable regardless of policy.
    pub fn is_always_halt(&self) -> bool {
        matches!(
            self,
            SandboxViolation::PathEscape { .. } | SandboxViolation::NetworkViolation { .. }
        )
    }
}

impl std::fmt::Display for SandboxViolation {
    /// Stable, model-facing rendering. Used wherever a violation is surfaced to
    /// the model as a tool-error message (instead of leaking the Rust `Debug`
    /// enum syntax). This output is a cross-language contract — keep it in sync
    /// with the TS/Py/Go ports.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxViolation::PathEscape { path } => {
                write!(f, "path escapes the workspace: {path}")
            }
            SandboxViolation::NetworkViolation { host } => {
                write!(f, "network access blocked: {host}")
            }
            SandboxViolation::PathDenied { path, matched_rule } => {
                if matched_rule.is_empty() {
                    write!(f, "path denied: {path}")
                } else {
                    write!(f, "path denied: {path} (rule: {matched_rule})")
                }
            }
            SandboxViolation::ReadOnlyViolation { path } => {
                write!(f, "write to read-only path: {path}")
            }
            SandboxViolation::ExtensionDenied { path, extension } => {
                write!(f, "file extension '{extension}' not allowed: {path}")
            }
            SandboxViolation::FileSizeExceeded { path, size, limit } => {
                write!(f, "file size {size} exceeds limit {limit}: {path}")
            }
            SandboxViolation::DisallowedCommand { command } => {
                write!(f, "command not allowed: {command}")
            }
            SandboxViolation::ExecSpawnFailed { command, message } => {
                write!(f, "failed to spawn command '{command}': {message}")
            }
        }
    }
}

/// How the harness treats a [`SandboxViolation`] surfaced by a tool (or the
/// pre-dispatch `validate` check): coach the model and let it retry, or halt the
/// run outright.
///
/// This is the single switch for sandbox-violation handling, applied uniformly
/// across every tool (filesystem, bash/exec, …) and both surfacing paths. The
/// default is [`Recoverable`](Self::Recoverable): a mis-targeted write or a
/// blocked command is a benign mistake for an agent, so the violation is
/// appended as a recoverable tool error and the model self-corrects (the
/// boundary still holds — the access was refused). Opt into
/// [`Halt`](Self::Halt) to restore the original Layer-1 always-halt behavior for
/// path escapes / network violations.
///
/// Set it via [`HarnessBuilder::sandbox_violation_policy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SandboxViolationPolicy {
    /// Surface the violation to the model as a recoverable tool error it can
    /// retry from. Halt-eligible or not, nothing halts the run. The default.
    #[default]
    Recoverable,
    /// Halt the run when a halt-eligible violation
    /// ([`SandboxViolation::is_always_halt`]) is encountered. Layer-2 violations
    /// stay recoverable. Restores the pre-policy Layer-1 behavior.
    Halt,
}

/// Termination-policy decision. Full enum lives in #13.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TerminationDecision {
    Continue,
    Halt { reason: String },
}

/// Session state handed back and forth across pause/resume. The harness does
/// not interpret its contents; it round-trips opaquely so that #7
/// (ContextManager) and #8 (MemoryProvider) own the schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SessionState {
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub extras: serde_json::Map<String, serde_json::Value>,
}


/// Head+tail-truncated output. `full_ref` is `Some` when the sandbox offloads
/// the original content to a backing file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TruncatedOutput {
    /// The (possibly truncated) head+tail content surfaced to the agent.
    pub content: String,
    /// `true` if the original content was truncated.
    #[serde(default)]
    pub truncated: bool,
    /// Backing file holding the full original content, if offloaded.
    #[serde(default)]
    pub full_ref: Option<FileRef>,
    /// Original size of the input content in bytes.
    #[serde(default)]
    pub original_size: u64,
}

/// Reference to a file holding offloaded tool output.
///
/// `path` is a string and `byte_len` is a `u64` to keep the wire format
/// portable across the four reference implementations — `PathBuf` serde is
/// platform-specific and `usize` width differs by target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRef {
    pub path: String,
    pub byte_len: u64,
}

/// Inputs the harness compaction loop (issue #46) needs to run one
/// compaction turn and verify its result.
///
/// The harness loop operates on the opaque [`SessionState`] above; the rich
/// compaction/verification API ([`crate::context::ContextManager`],
/// [`CompactionVerifier`]) operates on [`crate::context::SessionState`]. This
/// struct is the bridge: a [`ContextManager`] that supports compaction
/// projects everything the loop needs into one value, so the loop never has
/// to know which concrete state type its manager uses internally.
///
/// `context` is fed straight to `Agent::turn` to produce the summary;
/// `preserve_hints` and `verification_state` are passed to
/// [`CompactionVerifier::verify`]. On a verification failure the loop re-runs
/// the turn with [`ContextManager::inject_missing_items`] applied to `context`.
pub struct CompactionTurn {
    /// Context to feed `Agent::turn` to elicit the summary.
    pub context: Context,
    /// Preservation hints to hand the verifier.
    pub preserve_hints: CompactionPreserveHints,
    /// Verifier-facing session state (rich `context::SessionState`).
    pub verification_state: ContextSessionState,
    /// Messages about to be removed — used to stamp the compaction span.
    pub messages_removed: u32,
    /// The exact messages this turn will drop, computed once in
    /// `prepare_compaction_turn`. Threaded into [`ContextManager::apply_compaction`]
    /// so the manager reuses this set for token accounting instead of
    /// re-deriving it (one source of truth).
    pub dropped_messages: Vec<Message>,
}

/// Issue #7 — ContextManager: assembles per-turn context.
///
/// Issue #46 adds the optional compaction-loop surface
/// ([`prepare_compaction_turn`](Self::prepare_compaction_turn),
/// [`inject_missing_items`](Self::inject_missing_items),
/// [`apply_compaction`](Self::apply_compaction)). All three have defaults so
/// managers that do not compact (the default `should_compact` returns `false`)
/// need not implement them.
pub trait ContextManager: Send + Sync {
    /// Build one turn's model input. `sources` (issue #115 / SC-26) carries the
    /// rich [`ContextSources`] — guides, merged memory, tool schemas, and the
    /// composed static prompt — so a manager can place them in structural slots
    /// instead of the caller injecting them ad-hoc as User messages. Managers
    /// that do not consume sources may ignore the argument (the pre-#115
    /// behaviour); the production [`StandardCompactionAdapter`] renders them.
    fn assemble<'a>(
        &'a self,
        session: &'a SessionState,
        task: &'a Task,
        sources: &'a ContextSources,
    ) -> BoxFut<'a, Context>;

    fn append_tool_result<'a>(
        &'a self,
        session: &'a mut SessionState,
        result: &'a ToolResult,
    ) -> BoxFut<'a, ()>;

    /// Replace the tool-result message previously appended at `message_index`
    /// with a fresh rendering of `result`. The harness loop calls this from the
    /// `AfterTool` middleware hook (issue #11 / SC-9) when a middleware rewrote a
    /// result in place, so the rewrite reaches the next model turn. `message_index`
    /// is the position in `session.messages` recorded right after the original
    /// `append_tool_result`. The default is a no-op — a manager that does not
    /// store tool results as standalone messages need not act (the rewrite simply
    /// does not propagate, which is the pre-#11 behaviour).
    fn replace_tool_result<'a>(
        &'a self,
        session: &'a mut SessionState,
        message_index: usize,
        result: &'a ToolResult,
    ) -> BoxFut<'a, ()> {
        let _ = (session, message_index, result);
        Box::pin(async {})
    }

    /// Append the assistant's turn (model output: text and/or the tool calls it
    /// requested) to the conversation so the next assemble() reflects what the
    /// agent already did. Without this the model loses track of its own actions.
    fn append_assistant_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        message: &'a Message,
    ) -> BoxFut<'a, ()> {
        let _ = (session, message);
        Box::pin(async {})
    }

    fn append_user_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()>;

    fn should_compact(&self, session: &SessionState) -> bool {
        let _ = session;
        false
    }

    /// Build the inputs for one compaction turn (issue #46). Returns `None`
    /// when there is nothing to compact (e.g. history shorter than the
    /// preserve window), in which case the harness skips compaction entirely.
    ///
    /// Default: `None` — managers that never compact need not override this.
    fn prepare_compaction_turn(&self, session: &SessionState) -> Option<CompactionTurn> {
        let _ = session;
        None
    }

    /// Mutate a compaction [`Context`] in place to request a revised summary
    /// on retry (issue #46). The harness calls this with the items the prior
    /// summary failed to preserve. Default: append the standard "missing
    /// these items … please revise" instruction as a user message.
    fn inject_missing_items(&self, context: &mut Context, missing: &[String]) {
        context.messages.push(Message {
            role: crate::model::Role::User,
            content: crate::model::Content::Text {
                text: format!(
                    "Your summary is missing these items: {}. Please revise.",
                    missing.join(", ")
                ),
            },
        });
    }

    /// Accept a verified (or accepted-anyway) summary into the session,
    /// replacing the compacted span (issue #46). `dropped` is the message set
    /// computed by [`prepare_compaction_turn`](Self::prepare_compaction_turn)
    /// (carried on the [`CompactionTurn`]); managers use it for token accounting
    /// instead of re-deriving the dropped set. Default: no-op — only
    /// compaction-capable managers implement it.
    fn apply_compaction(&self, session: &mut SessionState, summary: String, dropped: &[Message]) {
        let _ = (session, summary, dropped);
    }

    /// Current token budget used for this session, if the manager tracks one
    /// (issue #57 — Known Deviation #2 token-accounting fix). The harness reads
    /// this after [`apply_compaction`](Self::apply_compaction) to stamp the
    /// compaction span with the real post-compaction utilization. Default:
    /// `None` — managers that do not track tokens fall back to the pre-value.
    fn token_budget_used(&self, session: &SessionState) -> Option<u32> {
        let _ = session;
        None
    }
}

/// Issue #13 — TerminationPolicy.
pub trait TerminationPolicy: Send + Sync {
    fn evaluate<'a>(
        &'a self,
        session: &'a SessionState,
        budget_used: &'a BudgetSnapshot,
    ) -> BoxFut<'a, TerminationDecision>;
}

/// Termination policy that lets the loop complete as soon as the agent produces
/// a final response (always `Continue`, which the harness interprets as "accept
/// the final response and succeed"). This is the default policy wired by
/// [`HarnessBuilder::conversational`] — a tool-less chat agent halts naturally
/// on its first final response, with no extra completion criteria to satisfy.
pub struct CompleteOnFinalResponse;

impl TerminationPolicy for CompleteOnFinalResponse {
    fn evaluate<'a>(
        &'a self,
        _session: &'a SessionState,
        _budget_used: &'a BudgetSnapshot,
    ) -> BoxFut<'a, TerminationDecision> {
        Box::pin(async { TerminationDecision::Continue })
    }
}

// Issue #11 — Middleware chain. The canonical `MiddlewareChain` trait,
// `MiddlewareDecision`, and `HookPoint` live in [`crate::middleware`] (the rich,
// fixtured surface). The harness loop wires that chain directly; the former
// harness-local stubs were deleted in Phase 3 (Q2 anchor decision).

// Issue #12 — `ObservabilityProvider` is no longer a no-op stub here. The
// canonical trait lives in [`crate::observability`]; the harness loop emits
// real spans through it (see `run_react`). Re-exported below for ergonomics.
pub use crate::observability::ObservabilityProvider;

// ============================================================================
// Human-in-the-loop
// ============================================================================
//
// ## Budget escalation HITL resume (issue #130)
//
// This section also owns the wiring that connects the `Escalate` budget
// resolution (#123/#125) to the existing HITL pause/resume seam
// (`RunResult::WaitingForHuman` + `PausedState` + `resume_inner`). The
// `escalation_mode` config knob (#120, previously UNCONSUMED) is finally read
// here.
//
// ### New types (#130)
//   - [`EscalationAction`] — the operator's choice on a budget-exhaustion
//     pause: `ContinueWithBudget { steps: u32 }` (grant N more steps, resume
//     from checkpoint) | `Skip` (mark the task skipped, the outer loop
//     advances) | `Fail` (abort the node, propagate `BudgetExceeded`). Named
//     `u32` field on `ContinueWithBudget` (serde internally-tagged enums cannot
//     carry tuple variants — mirrors `BudgetPolicy { value: u32 }`).
//   - [`HumanRequest::BudgetExhausted`] — a new variant carrying the node's
//     phase / policy / `steps_taken` / `continues_used` / `partial_output` and
//     the advisory `available_actions: Vec<EscalationAction>`. The existing
//     `ToolApproval` / `Clarification` / `Review` variants are UNCHANGED.
//   - [`HumanResponse::Escalate`] — a new variant delivering the operator's
//     [`EscalationAction`] on resume. It does NOT overload `Allow` / `Halt` /
//     `Deny`.
//
// ### The `escalation_mode` accessor
//   - [`StrategyExecutor::escalation_mode`] returns the configured
//     [`EscalationMode`]. `StandardHarness` returns `self.config.escalation_mode`
//     (mirrors `ralph_max_resets` / `workspace_root`). The `*Config::run` bodies
//     only hold `&dyn StrategyExecutor`, so this accessor is required to consult
//     the knob from a config body.
//
// ### Rules enforced (#130)
//   - At each `ExhaustedResolution::Escalate` site the body consults
//     `executor.escalation_mode()`:
//       * `EscalationMode::Autonomous`     → EXISTING behavior unchanged
//         (surface the partial, propagate `BudgetExhausted` up).
//       * `EscalationMode::SurfaceToHuman` → build a `PausedState` whose
//         `human_request = Some(HumanRequest::BudgetExhausted { .. })`, stash it
//         as `RunResult::WaitingForHuman` through the `terminal_override` seam so
//         it propagates VERBATIM.
//   - `available_actions` is ADVISORY for v1 (fork D): `resume` does NOT
//     hard-reject an out-of-set action. Combinators offer
//     `[ContinueWithBudget, Skip, Fail]`; a bare leaf OMITS `Skip` and offers
//     `[ContinueWithBudget, Fail]` (fork C).
//   - `resume_inner` reconstructs the node's `BudgetContext` from the request's
//     own `steps_taken` + `continues_used` (fork E — NOT #129's durable
//     cross-process checkpoint). `ContinueWithBudget { steps }` grants `steps`
//     additional allowance and re-enters the loop; `Skip` resolves the node
//     cleanly so the outer loop advances; `Fail` propagates
//     `Failure { BudgetExceeded }` with the partial discarded.
//
// No `// SPEC QUESTION:` markers — all forks are resolved in the #130 plan.

/// The operator's choice on a budget-exhaustion [`HumanRequest::BudgetExhausted`]
/// pause (#130). Tagged on `kind`, snake_case — internally-tagged, so the
/// data-carrying variant uses a NAMED `u32` field (tuple variants are not
/// representable; mirrors [`BudgetPolicy`]'s `value: u32` convention).
///
///   - `ContinueWithBudget { steps }` — grant `steps` additional steps to the
///     node's budget scope and resume from the checkpoint.
///   - `Skip` — mark the current task skipped; a combinator's outer loop
///     advances to its sibling tasks. Offered only by combinators (fork C).
///   - `Fail` — abort the node and propagate `BudgetExceeded` (partial
///     discarded, mirroring the `Fail` resolution contract).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EscalationAction {
    ContinueWithBudget { steps: u32 },
    Skip,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HumanRequest {
    ToolApproval {
        calls: Vec<ToolCall>,
        risk_level: RiskLevel,
    },
    Clarification {
        question: String,
        /// Optional fixed-choice options offered to the human (issue #81, Q4b).
        /// `None` for a free-form clarification. `#[serde(default)]` keeps older
        /// `Clarification` blobs (no `options` field) deserializing.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        options: Option<Vec<String>>,
    },
    Review {
        content: String,
    },
    /// A node's budget scope resolved to `Escalate` under
    /// [`EscalationMode::SurfaceToHuman`] (#130): the run pauses and surfaces the
    /// exhaustion to the operator. Carries the node's `phase`, its `policy`, the
    /// `steps_taken` / `continues_used` counters (so `resume_inner` can
    /// reconstruct the node's [`BudgetContext`] — fork E), any `partial_output`
    /// produced before exhaustion, and the ADVISORY `available_actions` the
    /// author offers (fork C/D). The operator answers with
    /// [`HumanResponse::Escalate`].
    BudgetExhausted {
        phase: String,
        policy: BudgetPolicy,
        steps_taken: u32,
        continues_used: u32,
        partial_output: Option<String>,
        available_actions: Vec<EscalationAction>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HumanResponse {
    Allow,
    AllowWithModification {
        calls: Vec<ToolCall>,
    },
    Deny {
        reason: String,
    },
    Halt,
    Answer {
        text: String,
    },
    ApproveWithFeedback {
        feedback: String,
    },
    Reject {
        reason: String,
    },
    /// The operator's resolution of a [`HumanRequest::BudgetExhausted`] pause
    /// (#130): the chosen [`EscalationAction`]. Distinct from `Allow`/`Halt`/
    /// `Deny` so the budget-escalation resume path is unambiguous.
    Escalate {
        action: EscalationAction,
    },
}

// ============================================================================
// Mid-loop consult primitive (issue #114)
// ============================================================================
//
// PRIMARY CONSULT LOGIC. This is the doc-hub for the mid-loop consult feature
// (issue #114). The signal/route/resume machinery is tightly coupled to the
// ReAct loop and the subagent boundary, so the implementation is split between
// this module (`harness.rs`) — which owns the worker-side pause, the resume
// seam, and the config — and [`crate::tools::subagent`] — which owns the
// orchestrator-side deterministic mediation.
//
// ## Types (all defined here unless noted)
// - [`ConsultRequest`]            — the worker's free-form ask (kind/situation/
//                                   attempts/question). All fields REQUIRED, no
//                                   serde defaults.
// - [`ConsultResponse`]           — the resume input (parallel to
//                                   [`HumanResponse`]): `Answer` | `BudgetExhausted`.
// - [`ConsultOverflowPolicy`]     — per-kind budget-overflow behaviour:
//                                   `SoftFail` | `EscalateToHuman`.
// - [`ConsultHandlerEntry`]       — a handler harness + budget + overflow policy,
//                                   keyed by `kind` in
//                                   [`HarnessConfig::consult_handlers`].
// - [`ToolOutput::Consult`]       — ONE variant. Worker-side tools emit it with
//                                   `child_state: None`; `SubagentTool`
//                                   populates `Some(..)` at the boundary
//                                   (mirrors [`ToolOutput::WaitingForHuman`]).
// - [`RunResult::Consult`]        — sibling of [`RunResult::WaitingForHuman`];
//                                   its `state.human_request` is `None`.
//
// ## Trait / loop methods touched
// - [`StandardHarness::run_react`] dispatch loop — pauses on
//   `ToolOutput::Consult` (R1, R10), preserving the consult call as the head of
//   `pending_tool_calls` exactly like the `AwaitingClarification` arm.
// - [`Harness::resume_consult`] — NEW trait method (the resume seam, below).
//   `StandardHarness` implements it via [`StandardHarness::resume_consult_inner`],
//   mirroring the `resume_inner` clarification branch.
// - [`crate::tools::subagent::SubagentTool::execute`] — the mediation seam
//   (below).
//
// ## Mediation seam — A1 (DECIDED)
// `SubagentTool::execute` drives the FULL consult cycle internally. On a child
// `RunResult::Consult` it routes by `request.kind` to the matching
// [`ConsultHandlerEntry`], checks/decrements the per-kind budget, runs the
// handler harness, builds a [`ConsultResponse::Answer`], and calls
// `child.resume_consult(..)` to continue the worker loop — repeating until the
// child reaches a terminal result. The parent orchestrator's model NEVER sees
// that a consult happened. Depth-1 holds: the handler is the orchestrator's
// direct child, never the worker's. The handlers reach `SubagentTool` through a
// constructor argument (see `SubagentTool::with_consult_handlers`).
//
// ## Resume seam (DECIDED)
// `ConsultResponse` is a distinct type (parallel to [`HumanResponse`]) and the
// resume entry point is a dedicated [`Harness::resume_consult`] trait method
// (NOT an overload of `resume`, which is typed on `HumanResponse`). It mirrors
// the `resume_inner` clarification branch: it injects the response text as the
// `ToolOutput::Success` RESULT of the head pending (consult) tool call, then
// dispatches any remaining pending calls and resumes the ReAct loop. The
// consult is therefore appended to message history ONLY as a tool result, only
// on resume (R10) — never as a free-standing turn.
//
// ## Rules enforced (see issue #114 / fixtures/harness/consult.json)
// R1, R5b, R6 (empty-map standalone), R10 here; R2–R5a, R6 (no-kind), R7 in
// `subagent.rs`; R8/R9 by the fixture-replay + existing-fixture tests.
//
// No `// SPEC QUESTION:` markers — every ambiguity is resolved in the plan.

/// The worker's free-form ask when it pauses mid-loop to consult a
/// parent-spawned helper (issue #114). `kind` selects the handler;
/// `situation`, `attempts`, and `question` carry the free-form context the
/// handler needs. All fields are REQUIRED — there are deliberately no serde
/// defaults, so a malformed/partial request fails to deserialize rather than
/// silently defaulting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ConsultRequest {
    /// Routing key — selects the [`ConsultHandlerEntry`] in
    /// [`HarnessConfig::consult_handlers`].
    pub kind: String,
    /// Free-form description of where the worker is stuck.
    pub situation: String,
    /// How many times the worker has already tried (advisory; the harness
    /// enforces the per-kind budget independently).
    pub attempts: u32,
    /// The concrete question the worker wants answered.
    pub question: String,
}

/// The resume input handed back to a paused worker after a consult (issue
/// #114). Parallel to [`HumanResponse`]; tagged on `kind`, snake_case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConsultResponse {
    /// The handler produced an answer; `text` is injected as the tool RESULT
    /// for the pending consult call.
    Answer { text: String },
    /// The per-kind budget is exhausted under a `SoftFail` overflow policy: the
    /// worker is resumed with this message and finishes with what it has.
    BudgetExhausted { message: String },
}

/// Per-kind budget-overflow behaviour (issue #114). Tagged on `kind`,
/// snake_case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConsultOverflowPolicy {
    /// Resume the worker with [`ConsultResponse::BudgetExhausted`] so it
    /// finishes without further help.
    SoftFail,
    /// Convert the over-budget consult into [`RunResult::WaitingForHuman`] so
    /// the host decides.
    EscalateToHuman,
}

/// A registered consult handler: the helper harness to run, the per-kind
/// budget (max consults of this kind before overflow), and the overflow policy
/// (issue #114). Held by `kind` in [`HarnessConfig::consult_handlers`].
///
/// The `handler` is run by `SubagentTool` as the ORCHESTRATOR's direct child
/// (depth-1, R7), never nested under the worker.
#[derive(Clone)]
pub struct ConsultHandlerEntry {
    /// The helper harness run on the [`ConsultRequest`].
    pub handler: Arc<dyn Harness>,
    /// Max number of consults of this kind before the overflow policy applies.
    pub budget: u32,
    /// What to do once the budget is exhausted.
    pub overflow: ConsultOverflowPolicy,
}

impl std::fmt::Debug for ConsultHandlerEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsultHandlerEntry")
            .field("handler", &"<dyn Harness>")
            .field("budget", &self.budget)
            .field("overflow", &self.overflow)
            .finish()
    }
}

// ============================================================================
// PausedState / ChildPausedState
// ============================================================================

/// The durable snapshot of a paused leaf. Carries everything needed to resume:
/// the in-progress session, the pending tool calls, the human/consult request,
/// and — since #140 — the pausing leaf's [`toolset`](Self::toolset) handle, so
/// resume routes pending per-node tool calls back through that leaf's scoped
/// catalogue instead of the global-catalogue fallback.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PausedState {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub turn_number: u32,
    pub session_state: SessionState,
    pub pending_tool_calls: Vec<ToolCall>,
    pub approved_results: Vec<ToolResult>,
    /// `None` for an escalation-derived state (issue #80) — an escalation has
    /// no human request. `WaitingForHuman` construction paths always set
    /// `Some(..)`. `#[serde(default)]` keeps old `WaitingForHuman` blobs
    /// deserializing (field present) while escalation blobs omit it.
    #[serde(default)]
    pub human_request: Option<HumanRequest>,
    pub task: Task,
    pub budget_used: BudgetSnapshot,
    #[serde(default)]
    pub child_state: Option<ChildPausedState>,
    /// The toolset handle of the leaf that paused (#140). Resume routes pending
    /// per-node tool calls through this handle's scoped catalogue via
    /// [`effective_tool_registry`](StandardHarness::effective_tool_registry); an
    /// empty handle (the default) falls back to the global catalogue. `#[serde(default)]`
    /// keeps pre-#140 paused-state blobs (no `toolset` key) deserializing — they
    /// restore as `ToolsetRef("")`. The field ALWAYS serializes (even when empty,
    /// as `"toolset":""`) for cross-language byte-parity; never add
    /// `skip_serializing_if`.
    #[serde(default)]
    pub toolset: ToolsetRef,
}

impl PausedState {
    /// The SHARED durable checkpoint round-trip (#129, AC1). Both cross-process
    /// `Continue` (a `HumanRequest::BudgetExhausted` pause whose request carries
    /// `continues_used`) and Ralph's pause-propagation hand the SAME
    /// [`PausedState`] to the caller for persistence; this is the one seam they
    /// share. It is JUST the `PausedState` serialize/deserialize — NOT a
    /// unification of their CONTEXT policies (Q2): a `Continue` resumes preserving
    /// `session_state.messages`; Ralph re-seeds a fresh window from its
    /// filesystem `.spore/progress.json` checkpoint, which stays Ralph-specific
    /// ([`RalphProgress`]/[`ralph_seed_session`](StrategyExecutor::ralph_seed_session)).
    ///
    /// `serialize_checkpoint` produces the durable blob the caller persists;
    /// [`load_checkpoint`](Self::load_checkpoint) restores it on resume.
    pub fn serialize_checkpoint(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Restore a [`PausedState`] from a durable checkpoint blob (#129, AC1). The
    /// resume side of [`serialize_checkpoint`](Self::serialize_checkpoint).
    pub fn load_checkpoint(blob: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(blob)
    }
}

/// Child paused state. **Deliberately has no `child_state` field** — the type
/// system enforces a one-level subagent depth (spec rule).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChildPausedState {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub turn_number: u32,
    pub session_state: SessionState,
    pub pending_tool_calls: Vec<ToolCall>,
    pub approved_results: Vec<ToolResult>,
    /// `None` for an escalation-derived state (issue #80). `WaitingForHuman`
    /// construction paths always set `Some(..)`.
    #[serde(default)]
    pub human_request: Option<HumanRequest>,
    pub task: Task,
    pub budget_used: BudgetSnapshot,
    pub parent_tool_call_id: String,
    /// The toolset handle of the child leaf that paused (#140); same semantics
    /// and serialization contract as [`PausedState::toolset`]. ALWAYS serializes
    /// (`"toolset":""` when empty); `#[serde(default)]` keeps pre-#140 child
    /// blobs deserializing.
    #[serde(default)]
    pub toolset: ToolsetRef,
}

// ============================================================================
// Halt reasons / RunResult
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum HaltReason {
    BudgetExceeded {
        limit_type: BudgetLimitType,
    },
    TerminationPolicyHalt {
        reason: String,
    },
    MiddlewareHalt {
        hook: HookPoint,
        reason: String,
    },
    AgentError {
        error: AgentError,
    },
    /// A [`ContextError`] surfaced by the [`crate::context::ContextManager`]
    /// during assembly halts the run (e.g. a cache-hash mismatch — both Block 1
    /// `Static` and, as of #32, Block 2 `PerSession` halt mid-session). This is
    /// the routing type; mirrors [`HaltReason::AgentError`]. The live
    /// [`StandardHarness`] loop does not yet trigger it because its placeholder
    /// `ContextManager::assemble` is infallible pending the #7 migration.
    ContextError {
        error: ContextError,
    },
    SandboxViolation {
        violation: SandboxViolation,
    },
    UnrecoverableToolError {
        tool: String,
        error: String,
    },
    HumanHalted,
    StagnationLimitReached {
        iterations: u32,
        best_metric: f64,
    },
    /// Returned by [`StandardHarness`] for [`LoopStrategy`] variants whose
    /// concrete trait dependencies (e.g. `CompletionCheck`, `Verifier`,
    /// `MetricEvaluator`) ship with later component issues. Marker only; will
    /// be removed once those issues land.
    StrategyNotYetImplemented {
        strategy: String,
    },
    /// Returned by [`StandardHarness`] for the `PlanExecute` strategy (issue
    /// #59) when the accepted plan parsed into an EMPTY task list (`tasks: []`).
    /// Per Q3, an empty plan is a failure — the run does NOT silently succeed.
    EmptyPlan,
    /// Returned by [`StandardHarness`] for the `PlanExecute` strategy (issue
    /// #59) when an execute step's bounded ReAct sub-loop errored or the agent
    /// returned a blocked/failed outcome (Q5). A plan is a dependency chain by
    /// assumption, so the whole run aborts at the failing step — execution does
    /// NOT continue to the next task. Carries the failing step's positional
    /// index, its instruction, and a human-readable reason derived from the
    /// underlying [`HaltReason`].
    StepFailed {
        task_index: usize,
        task: String,
        reason: String,
    },
    /// The PlanExecute plan phase (issue #70) failed before producing an
    /// artifact: the planner's response was unparseable, the planner requested
    /// a tool call in the one-shot turn, or the artifact could not be
    /// serialized for storage. Carries the underlying
    /// [`PlanPhaseError`](crate::plan::PlanPhaseError).
    PlanPhaseFailed {
        error: crate::plan::PlanPhaseError,
    },
    /// Returned by [`StandardHarness`] for the `SelfVerifying` strategy (issue
    /// #61, D4) when the build↔evaluate loop ran out of the verifier's
    /// `max_iterations` round-trips without an explicit `Passed` verdict. A
    /// RUNTIME limit — the work was attempted in good faith but never verified;
    /// a caller might retry with a different task decomposition. Carries the
    /// number of round-trips run and the last failure reason the verifier gave.
    /// PEER to [`SelfVerifyMisconfigured`](Self::SelfVerifyMisconfigured) (NOT a
    /// sub-case of it).
    SelfVerifyExhausted {
        iterations: u32,
        last_reason: String,
    },
    /// Returned by [`StandardHarness`] for the `SelfVerifying` strategy (issue
    /// #61, D4) when the strategy cannot run because it is misconfigured — e.g.
    /// `config.verifier` is `None`. Likely a BUILD-TIME bug in the caller's
    /// wiring. Surfaced as a typed halt, NOT a panic. PEER to
    /// [`SelfVerifyExhausted`](Self::SelfVerifyExhausted) (NOT a sub-case of it).
    SelfVerifyMisconfigured {
        reason: String,
    },
    /// Returned by [`StandardHarness`] for the `Ralph` strategy (issue #58, B3)
    /// when the multi-context-window continuation loop reached its `max_resets`
    /// cap with tasks still incomplete (the Ralph analogue of
    /// [`SelfVerifyExhausted`](Self::SelfVerifyExhausted)). A RUNTIME limit — the
    /// work was attempted across `iterations` context windows but the
    /// filesystem-backed completion check (the registered `Stop` hook reading
    /// `.spore/progress.json`) never reported done. Carries the number of
    /// context-window resets performed and the last incompletion reason.
    RalphCompletionUnmet {
        iterations: u32,
        last_reason: String,
    },
    /// Returned by [`StandardHarness`] for the `HillClimbing` strategy (issue
    /// #60) when the strategy cannot run because it is misconfigured — i.e.
    /// `config.metric_evaluator` is `None`. Likely a BUILD-TIME bug in the
    /// caller's wiring. Surfaced as a typed halt, NOT a panic. PEER to
    /// [`SelfVerifyMisconfigured`](Self::SelfVerifyMisconfigured).
    HillClimbingMisconfigured {
        reason: String,
    },
    /// Returned by [`StandardHarness`] when [`ExecutionRegistry::validate`](crate::ExecutionRegistry::validate)
    /// fails at run entry: a handle referenced by the task's strategy tree is
    /// unresolved against the configured registry, or a `StrategyRef::Custom`
    /// key is missing. A STARTUP error surfaced before the first turn (issue
    /// #120). Carries the underlying [`HarnessError`].
    ConfigurationError {
        error: HarnessError,
    },
    /// Returned by the PlanExecute DAG executor (#126, decision A) for a PARTIAL
    /// run: a task failed terminally (unrecoverable error or `BudgetExhausted`
    /// resolving to `Fail`) and its transitive dependents were cascade-`Blocked`,
    /// while unrelated branches still completed. The run as a whole is a
    /// `Failure`, but the full partition is reported: which tasks `completed`,
    /// which were `blocked` by the cascade, the `failed_task` that triggered it,
    /// and the human-readable `reason`. (A run where EVERY task completes is a
    /// `Success`, as before.)
    TasksBlockedByFailure {
        completed: Vec<u32>,
        blocked: Vec<u32>,
        failed_task: u32,
        reason: String,
    },
    /// Returned by the PlanExecute DAG executor (#126) when the persisted task
    /// graph contains a directed cycle, re-checked at EXECUTE ENTRY as
    /// defense-in-depth (`add_task` already rejects cycles, but the `task_list`
    /// tool path could in principle persist a cyclic graph out of band). No task
    /// is run. Carries a human-readable description.
    TaskGraphCycle {
        reason: String,
    },
    /// Returned by [`StandardHarness`] (issue #137) when the ReAct turn loop's
    /// consecutive-recoverable-tool-error breaker tripped: a tool returned
    /// `2 * error_loop_threshold` consecutive recoverable
    /// [`ToolOutput::Error`]s with identical arguments, so the loop STOPPED
    /// grinding and resolved the node's [`BudgetExhaustedBehavior`] WITHOUT
    /// burning the remaining budget. This is DISTINCT from
    /// [`BudgetExceeded`](Self::BudgetExceeded) (genuine budget exhaustion): the
    /// loop halted early on a detected error loop, not because the step budget
    /// ran out. Carries the offending tool name and the consecutive-error count
    /// at the moment of the hard stop (`2 * N`).
    ToolErrorLoop {
        tool: String,
        consecutive_errors: u32,
    },
    /// The repeated-identical-call breaker tripped: the model called the SAME
    /// tool with structurally-identical arguments `2 * error_loop_threshold`
    /// times in a row, each SUCCEEDING — i.e. it is spinning (repeating an action
    /// that makes no progress and never reaching a terminal answer). DISTINCT
    /// from [`ToolErrorLoop`](Self::ToolErrorLoop) (which counts recoverable
    /// ERRORS) and from [`BudgetExceeded`](Self::BudgetExceeded) (genuine budget
    /// exhaustion): the loop halted EARLY on a detected no-progress repeat, not on
    /// errors or a spent step budget. A single corrective nudge is injected at the
    /// `N` threshold first; this terminal fires only if the model keeps repeating
    /// to `2 * N`. Carries the offending tool name and the consecutive-repeat
    /// count at the hard stop. The run does NOT auto-continue (more budget would
    /// just feed the same loop) — it ends so the caller/operator can redirect.
    RepeatedToolCall {
        tool: String,
        count: u32,
    },
    /// Returned by [`StandardHarness`] (issue #139) when output-schema
    /// enforcement is ON (`HarnessConfig::enforce_output_schemas`) for a
    /// `ReactConfig` leaf carrying `output = Some(..)`, and the leaf's terminal
    /// `FinalResponse` STILL failed validation after the configured
    /// `output_schema_max_retries` extra turns were exhausted WITH budget
    /// remaining. DISTINCT from [`BudgetExceeded`](Self::BudgetExceeded): a
    /// budget/turn cap that a retry would exceed surfaces the budget terminal
    /// instead (budget-cap-wins precedence) — `OutputSchemaViolation` fires ONLY
    /// on a genuine validation exhaustion. Carries the resolved `schema` (the
    /// canonical compact key-sorted JSON), the total number of `attempts`
    /// (`1 + max_retries`), and the `last_error` (the frozen validator error
    /// string from the final attempt).
    OutputSchemaViolation {
        schema: String,
        attempts: u32,
        last_error: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunResult {
    Success {
        output: String,
        session_id: SessionId,
        usage: AggregateUsage,
        turns: u32,
        /// The post-run conversation history (issue #102). Carries the full
        /// `SessionState.messages` the loop produced — assistant tool-call turns
        /// and tool-result turns included — so an in-process caller can resume
        /// losslessly via `HarnessRunOptions::with_session_state(..)` without
        /// reconstructing history from `output`. `#[serde(default)]` keeps old
        /// serialized blobs (and other languages mid-migration) deserializing.
        #[serde(default)]
        session_state: SessionState,
    },
    Failure {
        reason: HaltReason,
        session_id: SessionId,
        usage: AggregateUsage,
        turns: u32,
        /// The post-run conversation history at the point of failure (issue
        /// #102). Same contract as on [`Success`](Self::Success): lossless
        /// resume, `#[serde(default)]` for back-compat.
        #[serde(default)]
        session_state: SessionState,
    },
    /// Boxed because `PausedState` is significantly larger than the other
    /// variants; keeps `RunResult` cheap to clone on the happy path.
    WaitingForHuman {
        state: Box<PausedState>,
        request: HumanRequest,
    },
    /// Harness terminated cleanly due to a tool escalation signal (issue #80).
    /// The caller handles the `signal` and decides whether to resume the
    /// original harness, instantiate a new one, or present UI to the user. The
    /// `state` preserves the full [`PausedState`] (with `human_request: None`)
    /// so `harness.resume(state, ..)` continues the original session; the
    /// signal is NOT stored in the state, so it is naturally discarded on
    /// resume — the harness does not re-act on it.
    Escalate {
        signal: HarnessSignal,
        state: Box<PausedState>,
        session_id: SessionId,
        usage: AggregateUsage,
        turns: u32,
    },
    /// Worker paused mid-loop to consult a parent-spawned helper (issue #114).
    /// Sibling of [`WaitingForHuman`](Self::WaitingForHuman), but it stops at
    /// the ORCHESTRATOR (via `SubagentTool`'s A1 mediation), not the human. The
    /// `state` preserves the full [`PausedState`] with `human_request: None`
    /// and the consult call as the head of `pending_tool_calls`, so
    /// `harness.resume_consult(state, response, ..)` continues the worker. With
    /// no consult handlers registered, a standalone worker simply returns this
    /// unchanged to its caller (R6 graceful degradation).
    Consult {
        request: ConsultRequest,
        state: Box<PausedState>,
        session_id: SessionId,
        usage: AggregateUsage,
        turns: u32,
    },
}

/// Result of a successful PlanExecute plan phase (issue #70). Carries the
/// produced artifact plus the run accounting so the caller can build the terminal
/// `RunResult`. Surfaced on the [`StrategyExecutor::capture_plan_artifact`]
/// primitive (#124); the artifact is also persisted under
/// `PLAN_EXECUTE_EXTRAS_KEY` on the RunStore.
#[derive(Debug)]
pub struct PlanPhaseOutcome {
    pub artifact: crate::plan::PlanArtifact,
    pub usage: AggregateUsage,
    pub turns: u32,
}

// ============================================================================
// Ralph loop strategy support types (issue #58)
// ============================================================================

/// Deserialized `.spore/progress.json` for the Ralph loop strategy (issue #58,
/// B1/B2). The handoff artifact between context windows: `complete` is the
/// primary completion signal, `remaining` lists outstanding work so an
/// incompletion reason can name what is left. Tolerant by default (`#[serde]`
/// defaults) so a partially-written file deserializes to "incomplete" rather
/// than erroring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct RalphProgress {
    #[serde(default)]
    pub(crate) complete: bool,
    #[serde(default)]
    pub(crate) remaining: Vec<String>,
}

/// One `.spore/feature_list.json` entry, mirroring
/// [`FeatureListCheck`](crate::termination::FeatureListCheck)'s schema so the
/// two sources agree (issue #58, B2).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RalphFeatureEntry {
    pub(crate) name: String,
    pub(crate) passes: bool,
}

/// The `Stop` lifecycle hook (issue #69) the Ralph loop strategy registers at
/// construction (issue #58, B1). Drives multi-context-window continuation off
/// `.spore/progress.json`: while tasks remain incomplete it returns
/// [`HookDecision::Block`](crate::hooks::HookDecision::Block) (the reason
/// describes what is left) so the harness loops into a new context window; when
/// complete it returns [`Continue`](crate::hooks::HookDecision::Continue) so the
/// loop terminates with success.
///
/// Registration is harmless for non-Ralph strategies: when `.spore/progress.json`
/// is ABSENT the hook returns `Continue` and does not interfere with ReAct /
/// PlanExecute / SelfVerifying runs. It only blocks when a progress file is
/// PRESENT and reports incomplete tasks — the Ralph contract.
pub(crate) struct RalphStopHook {
    /// #142: the Ralph checkpoint moved off `.spore/` onto the durable project-id
    /// [`RunStore`](crate::storage::RunStore). The hook reads the checkpoint from
    /// the store under the stable [`ProjectId`](crate::storage::ProjectId).
    pub(crate) run_store: Arc<dyn crate::storage::RunStore>,
    pub(crate) project_id: crate::storage::ProjectId,
}

impl crate::hooks::Hook for RalphStopHook {
    fn handle<'c>(
        &'c self,
        ctx: &'c mut crate::hooks::HookContext<'_>,
    ) -> BoxFut<'c, Result<crate::hooks::HookDecision, crate::hooks::HookError>> {
        Box::pin(async move {
            // Only act on `Stop`; any other event is a no-op `Continue`.
            if !matches!(ctx, crate::hooks::HookContext::Stop { .. }) {
                return Ok(crate::hooks::HookDecision::Continue);
            }
            // Absent checkpoint ⇒ do not interfere with non-Ralph runs. The
            // checkpoint now lives in the project-id RunStore (#142); when the
            // progress key is UNSET this is not a Ralph run, so `Continue`.
            let present = matches!(
                self.run_store
                    .get(
                        &self.project_id.namespace(),
                        crate::storage::RALPH_PROGRESS_KEY,
                    )
                    .await,
                Ok(Some(_))
            );
            if !present {
                return Ok(crate::hooks::HookDecision::Continue);
            }
            match StandardHarness::ralph_completion_status(&self.run_store, &self.project_id).await
            {
                None => Ok(crate::hooks::HookDecision::Continue),
                Some(reason) => Ok(crate::hooks::HookDecision::Block { reason }),
            }
        })
    }
    fn events(&self) -> Vec<crate::hooks::HookEvent> {
        vec![crate::hooks::HookEvent::Stop]
    }
    fn name(&self) -> String {
        "ralph-stop".into()
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[non_exhaustive]
pub enum HarnessError {
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    /// A `StrategyRef::Custom(key)` referenced a custom strategy that is not
    /// registered in [`ExecutionRegistry::custom`](crate::ExecutionRegistry).
    /// RECOVERABLE — returned, never panics (same pattern as a missing
    /// `AgentRef`). Issue #120.
    #[error("custom strategy not found: {key}")]
    StrategyNotFound { key: String },
    /// A serializable handle (`AgentRef`/`ToolsetRef`/`SchemaRef`) referenced an
    /// entry absent from the [`ExecutionRegistry`](crate::ExecutionRegistry).
    /// The STARTUP-validation error: surfaced before the first turn. Issue #120.
    ///
    /// The handle category is the Rust field `kind`, but it serializes as
    /// `handle_kind` to avoid colliding with the enum's `#[serde(tag = "kind")]`
    /// discriminant.
    #[error("unresolved {kind} handle: {key}")]
    UnresolvedHandle {
        #[serde(rename = "handle_kind")]
        kind: String,
        key: String,
    },
}

// ============================================================================
// HarnessRunOptions
// ============================================================================

pub struct HarnessRunOptions {
    pub task: Task,
    pub on_stream: Option<StreamSink>,
    /// Optional starting session state (e.g. resumed conversation history).
    pub session_state: Option<SessionState>,
    /// Opt in to CONTINUING the durable, project-scoped task list from a prior
    /// run rather than planning fresh (model A). Default `false`: a `PlanExecute`
    /// `run()` plans fresh and clears any stale list — the correct default for a
    /// conversational host issuing one independent `run()` per user turn. Set
    /// `true` only to deliberately resume an in-flight plan across `run()` calls
    /// (e.g. a cross-process resume of a top-level `PlanExecute` by project id);
    /// the in-process HITL / budget / consult resume paths and the Ralph window
    /// loop signal continuation on their own and do NOT need this. Has effect only
    /// when there is unfinished (`Pending`/`InProgress`) work to resume.
    pub continue_task_list: bool,
}

impl HarnessRunOptions {
    pub fn new(task: Task) -> Self {
        Self {
            task,
            on_stream: None,
            session_state: None,
            continue_task_list: false,
        }
    }

    /// Carry an existing session state into the run (e.g. a resumed
    /// conversation history for multi-turn scenarios — issue #57 S2).
    pub fn with_session_state(mut self, session_state: SessionState) -> Self {
        self.session_state = Some(session_state);
        self
    }

    /// Opt in to continuing a prior run's durable task list (model A). See
    /// [`continue_task_list`](HarnessRunOptions::continue_task_list).
    pub fn continue_task_list(mut self, yes: bool) -> Self {
        self.continue_task_list = yes;
        self
    }

    /// Attach a streaming sink for `StreamEvent`s emitted during the run.
    ///
    /// Generic over `impl Fn` so callers can pass a bare closure, an existing
    /// `Box<dyn Fn>` (which itself implements `Fn` — back-compat for pre-`Arc`
    /// callers), or an `Arc<dyn Fn>`; it is wrapped once into the shared
    /// [`StreamSink`].
    pub fn with_stream<F>(mut self, on_stream: F) -> Self
    where
        F: Fn(StreamEvent) + Send + Sync + 'static,
    {
        self.on_stream = Some(Arc::new(on_stream));
        self
    }
}

// ============================================================================
// The trait
// ============================================================================

pub trait Harness: Send + Sync {
    fn run<'a>(&'a self, options: HarnessRunOptions) -> BoxFut<'a, RunResult>;

    fn resume<'a>(
        &'a self,
        state: PausedState,
        response: HumanResponse,
        on_stream: Option<StreamSink>,
    ) -> BoxFut<'a, RunResult>;

    /// Resume a worker paused by [`RunResult::Consult`] (issue #114). The
    /// resume seam parallel to [`resume`](Self::resume): it injects the
    /// [`ConsultResponse`] as the tool RESULT of the head pending consult call,
    /// dispatches any remaining pending calls, and resumes the loop. A default
    /// impl is provided so callers that never use consults need not implement
    /// it; [`StandardHarness`] overrides it with the real behaviour.
    fn resume_consult<'a>(
        &'a self,
        state: PausedState,
        response: ConsultResponse,
        on_stream: Option<StreamSink>,
    ) -> BoxFut<'a, RunResult> {
        let _ = (response, on_stream);
        Box::pin(async move {
            RunResult::Failure {
                reason: HaltReason::HumanHalted,
                session_id: state.session_id,
                usage: AggregateUsage::default(),
                turns: state.turn_number,
                session_state: state.session_state,
            }
        })
    }
}
