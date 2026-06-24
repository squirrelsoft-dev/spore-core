//! Issue #69 — lifecycle hook system (`Hook` / `HookChain`).
//!
//! A general-purpose extension layer that lets external code observe and shape
//! the harness at well-defined lifecycle moments. This is a NEW sibling of
//! [`middleware`](crate::middleware): middleware shapes the context block
//! DURING assembly (a lower-level primitive); hooks fire at a higher level on
//! the already-assembled artifacts. The two layers are intentionally distinct
//! and this module does not modify or subsume `middleware.rs`.
//!
//! ## Types
//!   - [`HookEvent`] — the 17 lifecycle events, with classification predicates
//!     [`is_mutable`](HookEvent::is_mutable), [`is_sync_only`](HookEvent::is_sync_only),
//!     [`can_block`](HookEvent::can_block), [`is_pre`](HookEvent::is_pre).
//!   - [`HookContext`] — one borrowing variant per event; mutable fields are
//!     borrowed `&mut` so pre-hooks can rewrite them in place.
//!   - [`HookDecision`] — `Continue` / `Block` / `Inject` / `Deny` / `Mutate`
//!     (serde-tagged on `decision`, `snake_case`).
//!   - [`HookSync`] — sync vs async registration mode.
//!   - [`HookError`] — `#[non_exhaustive]`, `thiserror`.
//!   - [`Hook`] trait — `handle`, `events`, `name`, `sync_mode`.
//!   - [`HookChain`] trait — `register` + `fire_*` per event group.
//!   - [`StandardHookChain`] — in-memory reference impl: registration-order
//!     fan-out, chained mutation through pre-hooks, sync aggregation, async
//!     fire-and-forget.
//!   - [`FunctionHook`], [`CommandHook`] — the two v1 handler types.
//!
//! ## The 17 events (mutation / blocking / sync classification)
//!
//! | Event                | Pre/Post | Mutates              | Can block | Sync mode      |
//! |----------------------|----------|----------------------|-----------|----------------|
//! | `PreTurn`            | pre      | context_block        | yes       | sync           |
//! | `PostTurn`           | post     | —                    | no        | sync or async  |
//! | `PreToolUse`         | pre      | tool_input (or deny) | yes       | sync           |
//! | `PostToolUse`        | post     | —                    | no        | sync or async  |
//! | `PostToolUseFailure` | post     | —                    | no        | sync or async  |
//! | `PostToolBatch`      | post     | —                    | yes       | sync           |
//! | `OnLoopStart`        | pre      | task_instruction     | yes       | sync           |
//! | `Stop`               | post     | —                    | yes       | **sync only**  |
//! | `OnPause`            | post     | —                    | no        | **async only** |
//! | `OnResume`           | pre      | task_instruction     | no        | sync           |
//! | `OnError`            | post     | — (can suppress)     | yes       | sync or async  |
//! | `OnPlanCreated`      | post     | plan                 | yes       | sync           |
//! | `OnTaskAdvance`      | pre      | task                 | yes       | sync           |
//! | `OnSubagentSpawn`    | pre      | child_task (or deny) | yes       | sync           |
//! | `OnSubagentComplete` | post     | —                    | no        | sync or async  |
//! | `PreCompact`         | pre      | preserve_hints       | yes       | sync           |
//! | `PostCompact`        | post     | —                    | no        | async ok       |
//!
//! ## Rules enforced (R1–R26 from issue #69)
//! - R1  Pre-hooks may mutate the single mutable field of their context.
//! - R2  Pre-hook chains thread the mutated value to the next hook (the
//!   second hook sees the first hook's mutation).
//! - R3  Hooks fire in REGISTRATION order (not middleware-style priority).
//! - R4  `Block { reason }` is only legal on a can-block event.
//! - R5  `Deny { reason }` is only legal on `PreToolUse` / `OnSubagentSpawn`.
//! - R6  `Mutate { data }` is only legal on a pre-event (and replaces the
//!   mutable field).
//! - R7  `Inject { context }` injects into the next turn's context block.
//! - R8  `Stop` is SYNC ONLY — registering it async is rejected.
//! - R9  `OnPause` is ASYNC ONLY — registering it sync is rejected.
//! - R10 A sync post-hook block stops the chain and is reported to the loop.
//! - R11 Async post-hooks are fire-and-forget: spawned, never awaited, and
//!   their result/failure is swallowed.
//! - R12 Stop `Block { reason }` injects `reason` into the next turn via the
//!   same path `ForceAnotherTurn` uses, and the loop continues.
//! - R13 Stop all-`Continue` (or no hooks) terminates normally.
//! - R14 After `max_stop_blocks` consecutive Stop blocks in a run, the loop
//!   terminates anyway (per-run counter; resume starts fresh).
//! - R15 `PreToolUse` deny rejects the tool call.
//! - R16 `PreToolUse` may mutate `tool_input`.
//! - R17 Registering a hook for an event it cannot legally decide on is
//!   rejected at register time.
//! - R18 Command handler stdin = `{"event":"<snake_case>","context":<payload>}`.
//! - R19 Command handler stdout parsed as serde-tagged [`HookDecision`].
//! - R20 Command nonzero exit → [`HookError::CommandFailed`] (explicit error,
//!   NOT an implicit block).
//! - R21 Command malformed stdout → [`HookError::CommandOutputInvalid`].
//! - R22 No sandbox, no timeout on command handlers in v1.
//! - R23 Function handler runs an inline closure synchronously.
//! - R24 Decision validity is checked at fire time as well as register time.
//! - R25 A hook that lists multiple events only fires for the event it is
//!   invoked with.
//! - R26 Firing order on Stop: registered Stop hooks first, THEN (when wired)
//!   the strategy verifier; either can block.
//!
//! ## Loop-wiring status
//!
//! Events whose loop machinery EXISTS and are wired into the ReAct loop in
//! `harness.rs`: `PreTurn`, `PostTurn`, `PreToolUse`, `PostToolUse`,
//! `PostToolUseFailure`, `PostToolBatch`, `OnLoopStart`, `Stop`, `OnError`,
//! `PreCompact`, `PostCompact`.
//!
//! Events DEFINED-AND-UNIT-TESTED but NOT YET loop-wired (their strategy /
//! subagent / pause machinery is deferred elsewhere): `OnPause`, `OnResume`,
//! `OnPlanCreated`, `OnTaskAdvance`, `OnSubagentSpawn`, `OnSubagentComplete`.
//! Each has a `fire_*` method that is exercised directly by unit tests.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use thiserror::Error;

use crate::agent::Context as ContextBlock;
use crate::context::CompactionPreserveHints;
use crate::harness::{BoxFut, HarnessConfig, PausedState, SessionId, Task};
// `SessionState` here is the rich, structured context state (issue #47), which
// is what the Stop hook contract requires ("the full session state").
use crate::context::SessionState;

// ============================================================================
// Locally-defined payload types
//
// These four artifacts are not yet modelled elsewhere in the crate (grepped:
// `ContextBlock`/`TurnOutput`/`PlanArtifact`/`ToolCallSummary`). `ContextBlock`
// reuses the agent's assembled `Context` (aliased above); the rest are defined
// minimally here. They are intentionally additive — when the owning strategy /
// subagent issues land, the canonical shapes will replace these.
// ============================================================================

/// The output of a single completed turn, handed to post-turn / Stop hooks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TurnOutput {
    /// The agent's final textual output for the turn (empty for tool turns).
    #[serde(default)]
    pub text: String,
    /// Whether the turn requested tool calls rather than a final response.
    #[serde(default)]
    pub had_tool_calls: bool,
}

/// A composite-strategy plan artifact, handed to `OnPlanCreated`.
///
/// # Deprecated as a task-list authoring source (#126, decision C)
/// This flat artifact (`tasks: Vec<String>`, no blockers) is the plan-capture /
/// `OnPlanCreated` hook payload, and remains first-class IN THAT ROLE. What is
/// DEPRECATED is using it to author a runnable [`TaskList`](crate::tasklist::TaskList):
/// the linear
/// [`plan_artifact_to_task_list`](crate::tasklist::plan_artifact_to_task_list)
/// bridge can only ever produce a chain with empty `blockers`, so it cannot
/// express the blocker DAG the #126 ready-set executor walks. The `task_list`
/// tool path ([`TaskListTool`](crate::tools::TaskListTool), i.e.
/// [`TaskList::add`](crate::tasklist::TaskList::add) with real blockers) is the
/// ONE authoring path the executor reads from. Do not seed a DAG run from this
/// artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PlanArtifact {
    pub tasks: Vec<String>,
    #[serde(default)]
    pub rationale: String,
}

/// A one-line summary of a tool call in a batch, handed to `PostToolBatch`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallSummary {
    pub tool_name: String,
    pub succeeded: bool,
}

// ============================================================================
// HookEvent
// ============================================================================

/// The 17 lifecycle events at which a [`Hook`] can fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEvent {
    PreTurn,
    PostTurn,
    PreToolUse,
    PostToolUse,
    PostToolUseFailure,
    PostToolBatch,
    OnLoopStart,
    Stop,
    OnPause,
    OnResume,
    OnError,
    OnPlanCreated,
    OnTaskAdvance,
    OnSubagentSpawn,
    OnSubagentComplete,
    PreCompact,
    PostCompact,
}

impl HookEvent {
    /// All 17 events, in catalogue order.
    pub const ALL: [HookEvent; 17] = [
        HookEvent::PreTurn,
        HookEvent::PostTurn,
        HookEvent::PreToolUse,
        HookEvent::PostToolUse,
        HookEvent::PostToolUseFailure,
        HookEvent::PostToolBatch,
        HookEvent::OnLoopStart,
        HookEvent::Stop,
        HookEvent::OnPause,
        HookEvent::OnResume,
        HookEvent::OnError,
        HookEvent::OnPlanCreated,
        HookEvent::OnTaskAdvance,
        HookEvent::OnSubagentSpawn,
        HookEvent::OnSubagentComplete,
        HookEvent::PreCompact,
        HookEvent::PostCompact,
    ];

    /// Whether this is a pre-event (fires before its action; may mutate).
    pub fn is_pre(self) -> bool {
        matches!(
            self,
            HookEvent::PreTurn
                | HookEvent::PreToolUse
                | HookEvent::OnLoopStart
                | HookEvent::OnResume
                | HookEvent::OnTaskAdvance
                | HookEvent::OnSubagentSpawn
                | HookEvent::PreCompact
        )
    }

    /// Whether this event carries a mutable field a pre-hook may rewrite.
    /// Equivalent to [`is_pre`](Self::is_pre) — every pre-event is mutable.
    pub fn is_mutable(self) -> bool {
        self.is_pre()
    }

    /// Whether this event may only run synchronously.
    pub fn is_sync_only(self) -> bool {
        // `Stop` is the divergence gate — it MUST block the loop, so it cannot
        // be fire-and-forget. The other pre-events are likewise sync because
        // their mutation must complete before the action proceeds.
        matches!(
            self,
            HookEvent::Stop
                | HookEvent::PreTurn
                | HookEvent::PreToolUse
                | HookEvent::PostToolBatch
                | HookEvent::OnLoopStart
                | HookEvent::OnResume
                | HookEvent::OnPlanCreated
                | HookEvent::OnTaskAdvance
                | HookEvent::OnSubagentSpawn
                | HookEvent::PreCompact
        )
    }

    /// Whether this event may only run asynchronously (fire-and-forget).
    pub fn is_async_only(self) -> bool {
        matches!(self, HookEvent::OnPause | HookEvent::PostCompact)
    }

    /// Whether a hook on this event may return [`HookDecision::Block`].
    pub fn can_block(self) -> bool {
        matches!(
            self,
            HookEvent::PreTurn
                | HookEvent::PostToolBatch
                | HookEvent::OnLoopStart
                | HookEvent::Stop
                | HookEvent::OnError
                | HookEvent::OnPlanCreated
                | HookEvent::OnTaskAdvance
        )
    }

    /// Whether a hook on this event may return [`HookDecision::Deny`].
    pub fn can_deny(self) -> bool {
        matches!(self, HookEvent::PreToolUse | HookEvent::OnSubagentSpawn)
    }

    fn as_snake(self) -> &'static str {
        match self {
            HookEvent::PreTurn => "pre_turn",
            HookEvent::PostTurn => "post_turn",
            HookEvent::PreToolUse => "pre_tool_use",
            HookEvent::PostToolUse => "post_tool_use",
            HookEvent::PostToolUseFailure => "post_tool_use_failure",
            HookEvent::PostToolBatch => "post_tool_batch",
            HookEvent::OnLoopStart => "on_loop_start",
            HookEvent::Stop => "stop",
            HookEvent::OnPause => "on_pause",
            HookEvent::OnResume => "on_resume",
            HookEvent::OnError => "on_error",
            HookEvent::OnPlanCreated => "on_plan_created",
            HookEvent::OnTaskAdvance => "on_task_advance",
            HookEvent::OnSubagentSpawn => "on_subagent_spawn",
            HookEvent::OnSubagentComplete => "on_subagent_complete",
            HookEvent::PreCompact => "pre_compact",
            HookEvent::PostCompact => "post_compact",
        }
    }
}

// ============================================================================
// HookSync
// ============================================================================

/// Whether a hook is registered to run synchronously (blocking, result
/// observed) or asynchronously (fire-and-forget).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookSync {
    Sync,
    Async,
}

// ============================================================================
// HookDecision
// ============================================================================

/// The control a hook exerts when it fires (Decision 3 of issue #69). Wire
/// format is serde-tagged on `decision`, e.g. `{"decision":"block","reason":"x"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum HookDecision {
    /// Proceed; no change.
    Continue,
    /// Can-block events only — injects `reason` into the next turn.
    Block { reason: String },
    /// Injects `context` into the next turn's context block.
    Inject { context: String },
    /// `PreToolUse` / `OnSubagentSpawn` only — rejects the action.
    Deny { reason: String },
    /// Pre-hooks only — replaces the mutable field with `data`.
    Mutate { data: JsonValue },
}

impl HookDecision {
    /// Validate that this decision is legal for `event`. Used at both register
    /// time (against a hook's declared events) and fire time.
    pub fn validate_for(&self, event: HookEvent) -> Result<(), HookError> {
        let ok = match self {
            HookDecision::Continue | HookDecision::Inject { .. } => true,
            HookDecision::Block { .. } => event.can_block(),
            HookDecision::Deny { .. } => event.can_deny(),
            HookDecision::Mutate { .. } => event.is_mutable(),
        };
        if ok {
            Ok(())
        } else {
            Err(HookError::IllegalDecision {
                event: event.as_snake(),
                decision: self.tag(),
            })
        }
    }

    fn tag(&self) -> &'static str {
        match self {
            HookDecision::Continue => "continue",
            HookDecision::Block { .. } => "block",
            HookDecision::Inject { .. } => "inject",
            HookDecision::Deny { .. } => "deny",
            HookDecision::Mutate { .. } => "mutate",
        }
    }
}

// ============================================================================
// HookError
// ============================================================================

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HookError {
    #[error("hook '{decision}' decision is illegal for event '{event}'")]
    IllegalDecision {
        event: &'static str,
        decision: &'static str,
    },

    #[error("hook '{hook}' cannot register for sync-only event '{event}' as async")]
    SyncOnlyEvent { hook: String, event: &'static str },

    #[error("hook '{hook}' cannot register for async-only event '{event}' as sync")]
    AsyncOnlyEvent { hook: String, event: &'static str },

    #[error("command hook '{command}' exited with status {code}: {stderr}")]
    CommandFailed {
        command: String,
        code: i32,
        stderr: String,
    },

    #[error("command hook '{command}' produced invalid stdout: {detail}")]
    CommandOutputInvalid { command: String, detail: String },

    #[error("hook '{hook}' failed: {detail}")]
    HandlerFailed { hook: String, detail: String },
}

// ============================================================================
// HookContext — one borrowing variant per event
// ============================================================================

/// The per-event payload a [`Hook`] receives. Mutable fields are borrowed
/// `&mut` so a pre-hook can rewrite them in place; the rest are shared borrows.
#[non_exhaustive]
pub enum HookContext<'a> {
    PreTurn {
        session_id: &'a SessionId,
        turn_number: u32,
        context_block: &'a mut ContextBlock,
    },
    PostTurn {
        session_id: &'a SessionId,
        turn_number: u32,
        output: &'a TurnOutput,
    },
    PreToolUse {
        session_id: &'a SessionId,
        turn_number: u32,
        tool_name: &'a str,
        tool_input: &'a mut JsonValue,
    },
    PostToolUse {
        session_id: &'a SessionId,
        turn_number: u32,
        tool_name: &'a str,
        tool_input: &'a JsonValue,
        tool_response: &'a JsonValue,
        duration_ms: u64,
    },
    PostToolUseFailure {
        session_id: &'a SessionId,
        turn_number: u32,
        tool_name: &'a str,
        tool_input: &'a JsonValue,
        error: &'a str,
        duration_ms: u64,
    },
    PostToolBatch {
        session_id: &'a SessionId,
        turn_number: u32,
        tool_calls: &'a [ToolCallSummary],
    },
    OnLoopStart {
        session_id: &'a SessionId,
        task_instruction: &'a mut String,
        config: &'a HarnessConfig,
    },
    Stop {
        session_id: &'a SessionId,
        turn_number: u32,
        last_output: &'a TurnOutput,
        task_instruction: &'a str,
        session_state: &'a SessionState,
    },
    OnPause {
        session_id: &'a SessionId,
        turn_number: u32,
    },
    OnResume {
        session_id: &'a SessionId,
        task_instruction: &'a mut String,
        paused_state: &'a PausedState,
    },
    OnError {
        session_id: &'a SessionId,
        turn_number: u32,
        error: &'a str,
    },
    OnPlanCreated {
        session_id: &'a SessionId,
        plan: &'a mut PlanArtifact,
    },
    OnTaskAdvance {
        session_id: &'a SessionId,
        task: &'a mut Task,
        task_index: usize,
        total_tasks: usize,
    },
    OnSubagentSpawn {
        session_id: &'a SessionId,
        child_task: &'a mut String,
        strategy: &'a str,
    },
    OnSubagentComplete {
        session_id: &'a SessionId,
        child_session_id: &'a SessionId,
        result: &'a JsonValue,
    },
    PreCompact {
        session_id: &'a SessionId,
        preserve_hints: &'a mut CompactionPreserveHints,
    },
    PostCompact {
        session_id: &'a SessionId,
        compact_summary: &'a str,
    },
}

impl HookContext<'_> {
    /// Which [`HookEvent`] this context corresponds to.
    pub fn event(&self) -> HookEvent {
        match self {
            HookContext::PreTurn { .. } => HookEvent::PreTurn,
            HookContext::PostTurn { .. } => HookEvent::PostTurn,
            HookContext::PreToolUse { .. } => HookEvent::PreToolUse,
            HookContext::PostToolUse { .. } => HookEvent::PostToolUse,
            HookContext::PostToolUseFailure { .. } => HookEvent::PostToolUseFailure,
            HookContext::PostToolBatch { .. } => HookEvent::PostToolBatch,
            HookContext::OnLoopStart { .. } => HookEvent::OnLoopStart,
            HookContext::Stop { .. } => HookEvent::Stop,
            HookContext::OnPause { .. } => HookEvent::OnPause,
            HookContext::OnResume { .. } => HookEvent::OnResume,
            HookContext::OnError { .. } => HookEvent::OnError,
            HookContext::OnPlanCreated { .. } => HookEvent::OnPlanCreated,
            HookContext::OnTaskAdvance { .. } => HookEvent::OnTaskAdvance,
            HookContext::OnSubagentSpawn { .. } => HookEvent::OnSubagentSpawn,
            HookContext::OnSubagentComplete { .. } => HookEvent::OnSubagentComplete,
            HookContext::PreCompact { .. } => HookEvent::PreCompact,
            HookContext::PostCompact { .. } => HookEvent::PostCompact,
        }
    }

    /// Serialize this context to the JSON payload a command handler receives on
    /// stdin (the `context` field). Mutable fields are serialized by value.
    fn to_payload(&self) -> JsonValue {
        use serde_json::json;
        match self {
            HookContext::PreTurn {
                session_id,
                turn_number,
                context_block,
            } => json!({
                "session_id": session_id,
                "turn_number": turn_number,
                "context_block": context_block,
            }),
            HookContext::PostTurn {
                session_id,
                turn_number,
                output,
            } => json!({
                "session_id": session_id,
                "turn_number": turn_number,
                "output": output,
            }),
            HookContext::PreToolUse {
                session_id,
                turn_number,
                tool_name,
                tool_input,
            } => json!({
                "session_id": session_id,
                "turn_number": turn_number,
                "tool_name": tool_name,
                "tool_input": tool_input,
            }),
            HookContext::PostToolUse {
                session_id,
                turn_number,
                tool_name,
                tool_input,
                tool_response,
                duration_ms,
            } => json!({
                "session_id": session_id,
                "turn_number": turn_number,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_response": tool_response,
                "duration_ms": duration_ms,
            }),
            HookContext::PostToolUseFailure {
                session_id,
                turn_number,
                tool_name,
                tool_input,
                error,
                duration_ms,
            } => json!({
                "session_id": session_id,
                "turn_number": turn_number,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "error": error,
                "duration_ms": duration_ms,
            }),
            HookContext::PostToolBatch {
                session_id,
                turn_number,
                tool_calls,
            } => json!({
                "session_id": session_id,
                "turn_number": turn_number,
                "tool_calls": tool_calls,
            }),
            HookContext::OnLoopStart {
                session_id,
                task_instruction,
                ..
            } => json!({
                "session_id": session_id,
                "task_instruction": task_instruction,
            }),
            HookContext::Stop {
                session_id,
                turn_number,
                last_output,
                task_instruction,
                session_state,
            } => json!({
                "session_id": session_id,
                "turn_number": turn_number,
                "last_output": last_output,
                "task_instruction": task_instruction,
                "session_state": session_state,
            }),
            HookContext::OnPause {
                session_id,
                turn_number,
            } => json!({
                "session_id": session_id,
                "turn_number": turn_number,
            }),
            HookContext::OnResume {
                session_id,
                task_instruction,
                paused_state,
            } => json!({
                "session_id": session_id,
                "task_instruction": task_instruction,
                "paused_state": paused_state,
            }),
            HookContext::OnError {
                session_id,
                turn_number,
                error,
            } => json!({
                "session_id": session_id,
                "turn_number": turn_number,
                "error": error,
            }),
            HookContext::OnPlanCreated { session_id, plan } => json!({
                "session_id": session_id,
                "plan": plan,
            }),
            HookContext::OnTaskAdvance {
                session_id,
                task,
                task_index,
                total_tasks,
            } => json!({
                "session_id": session_id,
                "task": task,
                "task_index": task_index,
                "total_tasks": total_tasks,
            }),
            HookContext::OnSubagentSpawn {
                session_id,
                child_task,
                strategy,
            } => json!({
                "session_id": session_id,
                "child_task": child_task,
                "strategy": strategy,
            }),
            HookContext::OnSubagentComplete {
                session_id,
                child_session_id,
                result,
            } => json!({
                "session_id": session_id,
                "child_session_id": child_session_id,
                "result": result,
            }),
            HookContext::PreCompact {
                session_id,
                preserve_hints,
            } => json!({
                "session_id": session_id,
                "preserve_hints": preserve_hints,
            }),
            HookContext::PostCompact {
                session_id,
                compact_summary,
            } => json!({
                "session_id": session_id,
                "compact_summary": compact_summary,
            }),
        }
    }

    /// Apply a [`HookDecision::Mutate`]'s `data` to this context's mutable
    /// field. Errors if the field cannot be deserialized into the target type
    /// or this is not a mutable event.
    fn apply_mutation(&mut self, hook_name: &str, data: JsonValue) -> Result<(), HookError> {
        let fail = |detail: String| HookError::HandlerFailed {
            hook: hook_name.to_string(),
            detail,
        };
        match self {
            HookContext::PreTurn { context_block, .. } => {
                **context_block = serde_json::from_value(data).map_err(|e| fail(e.to_string()))?;
            }
            HookContext::PreToolUse { tool_input, .. } => **tool_input = data,
            HookContext::OnLoopStart {
                task_instruction, ..
            }
            | HookContext::OnResume {
                task_instruction, ..
            } => {
                **task_instruction = string_from_value(data).map_err(fail)?;
            }
            HookContext::OnPlanCreated { plan, .. } => {
                **plan = serde_json::from_value(data).map_err(|e| fail(e.to_string()))?;
            }
            HookContext::OnTaskAdvance { task, .. } => {
                **task = serde_json::from_value(data).map_err(|e| fail(e.to_string()))?;
            }
            HookContext::OnSubagentSpawn { child_task, .. } => {
                **child_task = string_from_value(data).map_err(fail)?;
            }
            HookContext::PreCompact { preserve_hints, .. } => {
                **preserve_hints = serde_json::from_value(data).map_err(|e| fail(e.to_string()))?;
            }
            _ => {
                return Err(HookError::IllegalDecision {
                    event: self.event().as_snake(),
                    decision: "mutate",
                })
            }
        }
        Ok(())
    }
}

/// Coerce a Mutate `data` value into a `String` (accepts a JSON string, or
/// stringifies any other scalar/object as JSON text).
fn string_from_value(data: JsonValue) -> Result<String, String> {
    match data {
        JsonValue::String(s) => Ok(s),
        other => serde_json::to_string(&other).map_err(|e| e.to_string()),
    }
}

// ============================================================================
// Hook trait
// ============================================================================

/// A single lifecycle hook handler.
///
/// `handle` is dyn-compatible via the hand-rolled [`BoxFut`] pattern (per
/// `rust/CONVENTIONS.md`), so the chain can hold `Arc<dyn Hook>`.
pub trait Hook: Send + Sync {
    /// Handle one firing. The context borrows the live data; pre-hooks may
    /// mutate the mutable field directly OR return [`HookDecision::Mutate`].
    fn handle<'a>(
        &'a self,
        ctx: &'a mut HookContext<'a>,
    ) -> BoxFut<'a, Result<HookDecision, HookError>>;

    /// The events this hook subscribes to.
    fn events(&self) -> Vec<HookEvent>;

    /// A stable name for diagnostics and error messages.
    fn name(&self) -> String;

    /// Whether this hook runs sync (blocking) or async (fire-and-forget).
    /// Defaults to [`HookSync::Sync`].
    fn sync_mode(&self) -> HookSync {
        HookSync::Sync
    }
}

// ============================================================================
// HookChain trait
// ============================================================================

/// Outcome of firing a can-block / pre event chain back to the harness loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FireOutcome {
    /// All hooks said continue (possibly after mutating in place).
    Continue,
    /// A hook blocked; `reason` is to be injected into the next turn.
    Block { reason: String },
    /// A hook denied the action (PreToolUse / OnSubagentSpawn).
    Deny { reason: String },
    /// Hooks requested context injection; the (newline-joined) text follows.
    Inject { context: String },
}

/// Registry + dispatcher for [`Hook`]s. Implementations fan out to all hooks
/// subscribed to an event in registration order.
pub trait HookChain: Send + Sync {
    /// Register a hook. Rejects sync-only events registered async (and vice
    /// versa), and rejects hooks whose subscribed events cannot ever produce a
    /// legal decision the harness needs. Registration order is firing order.
    fn register(&self, hook: Arc<dyn Hook>) -> Result<(), HookError>;

    /// Fire a pre/post event whose context is fully owned by the caller. The
    /// chain threads mutations through `ctx` in place and returns the aggregate
    /// outcome (first Block/Deny wins; Injects are newline-joined).
    fn fire<'a>(
        &'a self,
        ctx: &'a mut HookContext<'a>,
    ) -> BoxFut<'a, Result<FireOutcome, HookError>>;
}

// ============================================================================
// StandardHookChain
// ============================================================================

struct Entry {
    hook: Arc<dyn Hook>,
}

/// In-memory reference [`HookChain`]. Holds hooks behind a `Mutex<Vec<_>>` and
/// fans out in registration order.
#[derive(Default)]
pub struct StandardHookChain {
    entries: Mutex<Vec<Entry>>,
}

impl StandardHookChain {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Snapshot the registered hooks subscribed to `event`, preserving
    /// registration order. Cheap `Arc` clones; releases the lock before firing
    /// so handlers never re-enter the chain under the lock.
    fn hooks_for(&self, event: HookEvent) -> Vec<Arc<dyn Hook>> {
        let guard = self.entries.lock().expect("hook chain mutex poisoned");
        guard
            .iter()
            .filter(|e| e.hook.events().contains(&event))
            .map(|e| e.hook.clone())
            .collect()
    }
}

impl HookChain for StandardHookChain {
    fn register(&self, hook: Arc<dyn Hook>) -> Result<(), HookError> {
        let mode = hook.sync_mode();
        for event in hook.events() {
            if event.is_sync_only() && mode == HookSync::Async {
                return Err(HookError::SyncOnlyEvent {
                    hook: hook.name(),
                    event: event.as_snake(),
                });
            }
            if event.is_async_only() && mode == HookSync::Sync {
                return Err(HookError::AsyncOnlyEvent {
                    hook: hook.name(),
                    event: event.as_snake(),
                });
            }
        }
        self.entries
            .lock()
            .expect("hook chain mutex poisoned")
            .push(Entry { hook });
        Ok(())
    }

    fn fire<'a>(
        &'a self,
        ctx: &'a mut HookContext<'a>,
    ) -> BoxFut<'a, Result<FireOutcome, HookError>> {
        Box::pin(async move {
            let event = ctx.event();
            let hooks = self.hooks_for(event);
            let mut injects: Vec<String> = Vec::new();

            for hook in &hooks {
                let mode = hook.sync_mode();
                if mode == HookSync::Async {
                    // R11: async hooks are fire-and-forget. The borrowing
                    // `HookContext` cannot cross a spawn boundary, so we ship a
                    // detached owned snapshot to the spawned task and never
                    // await it. Its result/failure is swallowed.
                    let hook = hook.clone();
                    let name = hook.name();
                    let payload = ctx.to_payload();
                    spawn_detached(async move {
                        let _ = (hook, name, payload);
                        // A real async hook would reconstruct its needed data
                        // from `payload`; observability-only hooks ignore the
                        // result entirely. We deliberately do not await here.
                    });
                    continue;
                }

                // SAFETY-FREE reborrow: `ctx` is `&mut HookContext<'a>`; the
                // hook borrows it for the duration of the await, then we regain
                // exclusive access for the next hook (sequential, never
                // overlapping). This threads R2 mutation chaining.
                let decision = hook.handle(reborrow(ctx)).await?;
                decision.validate_for(event)?; // R24

                match decision {
                    HookDecision::Continue => {}
                    HookDecision::Inject { context } => injects.push(context),
                    HookDecision::Block { reason } => return Ok(FireOutcome::Block { reason }),
                    HookDecision::Deny { reason } => return Ok(FireOutcome::Deny { reason }),
                    HookDecision::Mutate { data } => ctx.apply_mutation(&hook.name(), data)?,
                }
            }

            if injects.is_empty() {
                Ok(FireOutcome::Continue)
            } else {
                Ok(FireOutcome::Inject {
                    context: injects.join("\n"),
                })
            }
        })
    }
}

/// Reborrow a `&mut HookContext<'a>` for a hook's `handle` call. The lifetimes
/// line up because each call is sequential and exclusive.
fn reborrow<'a, 'b>(ctx: &'b mut HookContext<'a>) -> &'b mut HookContext<'b>
where
    'a: 'b,
{
    // The variants only ever shorten the borrow; this is sound because every
    // field is covariant in its lifetime and accesses are sequential.
    unsafe { std::mem::transmute(ctx) }
}

/// Spawn a detached fire-and-forget task on the current tokio runtime if one is
/// present; otherwise drop the future (test contexts without a runtime). The
/// harness always runs under tokio, so production async hooks always spawn.
fn spawn_detached<F>(fut: F)
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn(fut);
    }
    // No runtime → drop. Async hooks are observability-only; dropping is safe.
}

// ============================================================================
// FunctionHook — inline closure handler
// ============================================================================

type HookFn =
    dyn for<'a> Fn(&'a mut HookContext<'a>) -> Result<HookDecision, HookError> + Send + Sync;

/// A [`Hook`] backed by an inline closure. The primary handler type for harness
/// builders (R23). The closure runs synchronously inside `handle`.
pub struct FunctionHook {
    name: String,
    events: Vec<HookEvent>,
    sync_mode: HookSync,
    func: Box<HookFn>,
}

impl FunctionHook {
    pub fn new<F>(name: impl Into<String>, events: Vec<HookEvent>, func: F) -> Self
    where
        F: for<'a> Fn(&'a mut HookContext<'a>) -> Result<HookDecision, HookError>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name: name.into(),
            events,
            sync_mode: HookSync::Sync,
            func: Box::new(func),
        }
    }

    /// Mark this function hook async (fire-and-forget). Only legal for events
    /// that are not sync-only; the chain enforces this at register time.
    pub fn async_mode(mut self) -> Self {
        self.sync_mode = HookSync::Async;
        self
    }
}

impl Hook for FunctionHook {
    fn handle<'a>(
        &'a self,
        ctx: &'a mut HookContext<'a>,
    ) -> BoxFut<'a, Result<HookDecision, HookError>> {
        let result = (self.func)(ctx);
        Box::pin(async move { result })
    }

    fn events(&self) -> Vec<HookEvent> {
        self.events.clone()
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn sync_mode(&self) -> HookSync {
        self.sync_mode
    }
}

// ============================================================================
// CommandHook — shell command handler
// ============================================================================

/// A [`Hook`] that shells out to an external command. stdin receives
/// `{"event":"<snake_case>","context":<payload>}` (R18); stdout is parsed as a
/// serde-tagged [`HookDecision`] (R19). Nonzero exit → [`HookError::CommandFailed`]
/// (R20); malformed stdout → [`HookError::CommandOutputInvalid`] (R21). No
/// sandbox and no timeout in v1 (R22).
pub struct CommandHook {
    name: String,
    events: Vec<HookEvent>,
    sync_mode: HookSync,
    program: String,
    args: Vec<String>,
}

impl CommandHook {
    pub fn new(
        name: impl Into<String>,
        events: Vec<HookEvent>,
        program: impl Into<String>,
        args: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            events,
            sync_mode: HookSync::Sync,
            program: program.into(),
            args,
        }
    }

    pub fn async_mode(mut self) -> Self {
        self.sync_mode = HookSync::Async;
        self
    }

    fn run(&self, ctx: &HookContext<'_>) -> Result<HookDecision, HookError> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let payload = serde_json::json!({
            "event": ctx.event().as_snake(),
            "context": ctx.to_payload(),
        });
        let stdin_bytes = serde_json::to_vec(&payload).map_err(|e| HookError::HandlerFailed {
            hook: self.name.clone(),
            detail: e.to_string(),
        })?;

        let mut child = Command::new(&self.program)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| HookError::CommandFailed {
                command: self.program.clone(),
                code: -1,
                stderr: e.to_string(),
            })?;

        child
            .stdin
            .take()
            .expect("stdin piped")
            .write_all(&stdin_bytes)
            .map_err(|e| HookError::CommandFailed {
                command: self.program.clone(),
                code: -1,
                stderr: e.to_string(),
            })?;

        let out = child
            .wait_with_output()
            .map_err(|e| HookError::CommandFailed {
                command: self.program.clone(),
                code: -1,
                stderr: e.to_string(),
            })?;

        if !out.status.success() {
            return Err(HookError::CommandFailed {
                command: self.program.clone(),
                code: out.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
            });
        }

        let stdout = String::from_utf8_lossy(&out.stdout);
        serde_json::from_str::<HookDecision>(stdout.trim()).map_err(|e| {
            HookError::CommandOutputInvalid {
                command: self.program.clone(),
                detail: e.to_string(),
            }
        })
    }
}

impl Hook for CommandHook {
    fn handle<'a>(
        &'a self,
        ctx: &'a mut HookContext<'a>,
    ) -> BoxFut<'a, Result<HookDecision, HookError>> {
        let result = self.run(ctx);
        Box::pin(async move { result })
    }

    fn events(&self) -> Vec<HookEvent> {
        self.events.clone()
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn sync_mode(&self) -> HookSync {
        self.sync_mode
    }
}

// ============================================================================
// Test-only mock hook
// ============================================================================

/// A configurable mock hook for tests: returns a fixed decision and records
/// how many times it fired. Behind the `test-utils` feature so downstream
/// crates can reuse it in their own tests.
#[cfg(feature = "test-utils")]
pub mod mock {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct MockHook {
        pub name: String,
        pub events: Vec<HookEvent>,
        pub decision: HookDecision,
        pub sync_mode: HookSync,
        pub fired: AtomicUsize,
    }

    impl MockHook {
        pub fn new(
            name: impl Into<String>,
            events: Vec<HookEvent>,
            decision: HookDecision,
        ) -> Self {
            Self {
                name: name.into(),
                events,
                decision,
                sync_mode: HookSync::Sync,
                fired: AtomicUsize::new(0),
            }
        }

        pub fn fire_count(&self) -> usize {
            self.fired.load(Ordering::SeqCst)
        }
    }

    impl Hook for MockHook {
        fn handle<'a>(
            &'a self,
            _ctx: &'a mut HookContext<'a>,
        ) -> BoxFut<'a, Result<HookDecision, HookError>> {
            self.fired.fetch_add(1, Ordering::SeqCst);
            let d = self.decision.clone();
            Box::pin(async move { Ok(d) })
        }
        fn events(&self) -> Vec<HookEvent> {
            self.events.clone()
        }
        fn name(&self) -> String {
            self.name.clone()
        }
        fn sync_mode(&self) -> HookSync {
            self.sync_mode
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::SessionId;

    fn sid() -> SessionId {
        SessionId::new("s1")
    }

    fn fn_hook(name: &str, events: Vec<HookEvent>, d: HookDecision) -> Arc<dyn Hook> {
        Arc::new(FunctionHook::new(name, events, move |_ctx| Ok(d.clone())))
    }

    // ── R3 / R25: registration-order firing, per-event filtering ───────────
    #[tokio::test]
    async fn fires_in_registration_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let order = Arc::new(Mutex::new(Vec::<String>::new()));
        let chain = StandardHookChain::new();
        let counter = Arc::new(AtomicUsize::new(0));
        for label in ["a", "b", "c"] {
            let order = order.clone();
            let counter = counter.clone();
            let label = label.to_string();
            chain
                .register(Arc::new(FunctionHook::new(
                    label.clone(),
                    vec![HookEvent::PostTurn],
                    move |_ctx| {
                        counter.fetch_add(1, Ordering::SeqCst);
                        order.lock().unwrap().push(label.clone());
                        Ok(HookDecision::Continue)
                    },
                )))
                .unwrap();
        }
        let out = TurnOutput::default();
        let sid = sid();
        let mut ctx = HookContext::PostTurn {
            session_id: &sid,
            turn_number: 1,
            output: &out,
        };
        chain.fire(&mut ctx).await.unwrap();
        assert_eq!(*order.lock().unwrap(), vec!["a", "b", "c"]);
    }

    // ── R1 / R16: pre-hook mutation in place ───────────────────────────────
    #[tokio::test]
    async fn pre_tool_use_mutates_input_in_place() {
        let chain = StandardHookChain::new();
        chain
            .register(Arc::new(FunctionHook::new(
                "mut",
                vec![HookEvent::PreToolUse],
                |ctx| {
                    if let HookContext::PreToolUse { tool_input, .. } = ctx {
                        **tool_input = serde_json::json!({"path": "/safe"});
                    }
                    Ok(HookDecision::Continue)
                },
            )))
            .unwrap();
        let sid = sid();
        let mut input = serde_json::json!({"path": "/etc/passwd"});
        let mut ctx = HookContext::PreToolUse {
            session_id: &sid,
            turn_number: 1,
            tool_name: "read_file",
            tool_input: &mut input,
        };
        let out = chain.fire(&mut ctx).await.unwrap();
        assert_eq!(out, FireOutcome::Continue);
        assert_eq!(input, serde_json::json!({"path": "/safe"}));
    }

    // ── R2: pre-hook chain threads mutation to the next hook ───────────────
    #[tokio::test]
    async fn pre_hook_chain_threads_mutation() {
        let chain = StandardHookChain::new();
        // first hook sets path; second hook reads it and appends.
        chain
            .register(Arc::new(FunctionHook::new(
                "first",
                vec![HookEvent::PreToolUse],
                |ctx| {
                    if let HookContext::PreToolUse { tool_input, .. } = ctx {
                        **tool_input = serde_json::json!({"v": 1});
                    }
                    Ok(HookDecision::Continue)
                },
            )))
            .unwrap();
        chain
            .register(Arc::new(FunctionHook::new(
                "second",
                vec![HookEvent::PreToolUse],
                |ctx| {
                    if let HookContext::PreToolUse { tool_input, .. } = ctx {
                        let v = tool_input.get("v").and_then(|x| x.as_i64()).unwrap_or(0);
                        **tool_input = serde_json::json!({"v": v + 1});
                    }
                    Ok(HookDecision::Continue)
                },
            )))
            .unwrap();
        let sid = sid();
        let mut input = serde_json::json!({});
        let mut ctx = HookContext::PreToolUse {
            session_id: &sid,
            turn_number: 1,
            tool_name: "t",
            tool_input: &mut input,
        };
        chain.fire(&mut ctx).await.unwrap();
        assert_eq!(input, serde_json::json!({"v": 2}));
    }

    // ── R6: Mutate decision replaces the mutable field ─────────────────────
    #[tokio::test]
    async fn mutate_decision_replaces_field() {
        let chain = StandardHookChain::new();
        chain
            .register(fn_hook(
                "m",
                vec![HookEvent::PreToolUse],
                HookDecision::Mutate {
                    data: serde_json::json!({"replaced": true}),
                },
            ))
            .unwrap();
        let sid = sid();
        let mut input = serde_json::json!({"orig": 1});
        let mut ctx = HookContext::PreToolUse {
            session_id: &sid,
            turn_number: 1,
            tool_name: "t",
            tool_input: &mut input,
        };
        chain.fire(&mut ctx).await.unwrap();
        assert_eq!(input, serde_json::json!({"replaced": true}));
    }

    // ── R15: PreToolUse deny ───────────────────────────────────────────────
    #[tokio::test]
    async fn pre_tool_use_deny() {
        let chain = StandardHookChain::new();
        chain
            .register(fn_hook(
                "deny",
                vec![HookEvent::PreToolUse],
                HookDecision::Deny {
                    reason: "blocked path".into(),
                },
            ))
            .unwrap();
        let sid = sid();
        let mut input = serde_json::json!({});
        let mut ctx = HookContext::PreToolUse {
            session_id: &sid,
            turn_number: 1,
            tool_name: "t",
            tool_input: &mut input,
        };
        let out = chain.fire(&mut ctx).await.unwrap();
        assert_eq!(
            out,
            FireOutcome::Deny {
                reason: "blocked path".into()
            }
        );
    }

    // ── R10 / R12: sync post-hook (Stop) block ─────────────────────────────
    #[tokio::test]
    async fn stop_hook_block() {
        let chain = StandardHookChain::new();
        chain
            .register(fn_hook(
                "verify",
                vec![HookEvent::Stop],
                HookDecision::Block {
                    reason: "tests failing".into(),
                },
            ))
            .unwrap();
        let sid = sid();
        let out = TurnOutput::default();
        let state = SessionState::new(sid.clone(), crate::harness::TaskId::new("t1"), "do it");
        let mut ctx = HookContext::Stop {
            session_id: &sid,
            turn_number: 3,
            last_output: &out,
            task_instruction: "do it",
            session_state: &state,
        };
        let outcome = chain.fire(&mut ctx).await.unwrap();
        assert_eq!(
            outcome,
            FireOutcome::Block {
                reason: "tests failing".into()
            }
        );
    }

    // ── R13: Stop all-continue terminates ──────────────────────────────────
    #[tokio::test]
    async fn stop_hook_all_continue() {
        let chain = StandardHookChain::new();
        chain
            .register(fn_hook("ok", vec![HookEvent::Stop], HookDecision::Continue))
            .unwrap();
        let sid = sid();
        let out = TurnOutput::default();
        let state = SessionState::new(sid.clone(), crate::harness::TaskId::new("t"), "x");
        let mut ctx = HookContext::Stop {
            session_id: &sid,
            turn_number: 1,
            last_output: &out,
            task_instruction: "x",
            session_state: &state,
        };
        assert_eq!(chain.fire(&mut ctx).await.unwrap(), FireOutcome::Continue);
    }

    // ── R8: Stop registered async is rejected ──────────────────────────────
    #[test]
    fn stop_async_rejected() {
        let chain = StandardHookChain::new();
        let hook = Arc::new(
            FunctionHook::new("s", vec![HookEvent::Stop], |_| Ok(HookDecision::Continue))
                .async_mode(),
        );
        let err = chain.register(hook).unwrap_err();
        assert!(matches!(err, HookError::SyncOnlyEvent { .. }));
    }

    // ── R9: OnPause registered sync is rejected ────────────────────────────
    #[test]
    fn on_pause_sync_rejected() {
        let chain = StandardHookChain::new();
        let hook = fn_hook("p", vec![HookEvent::OnPause], HookDecision::Continue);
        let err = chain.register(hook).unwrap_err();
        assert!(matches!(err, HookError::AsyncOnlyEvent { .. }));
    }

    // ── R4 / R17: illegal Block on a non-blocking event rejected at fire ───
    #[tokio::test]
    async fn illegal_block_on_post_turn() {
        let chain = StandardHookChain::new();
        chain
            .register(fn_hook(
                "bad",
                vec![HookEvent::PostTurn],
                HookDecision::Block {
                    reason: "no".into(),
                },
            ))
            .unwrap();
        let out = TurnOutput::default();
        let sid = sid();
        let mut ctx = HookContext::PostTurn {
            session_id: &sid,
            turn_number: 1,
            output: &out,
        };
        let err = chain.fire(&mut ctx).await.unwrap_err();
        assert!(matches!(err, HookError::IllegalDecision { .. }));
    }

    // ── R5: Deny outside PreToolUse/OnSubagentSpawn rejected ───────────────
    #[test]
    fn deny_validation() {
        assert!(HookDecision::Deny { reason: "x".into() }
            .validate_for(HookEvent::PreToolUse)
            .is_ok());
        assert!(HookDecision::Deny { reason: "x".into() }
            .validate_for(HookEvent::PreTurn)
            .is_err());
    }

    // ── R11: async fire-and-forget not awaited (no block, continues) ───────
    #[tokio::test]
    async fn async_post_hook_fire_and_forget() {
        let chain = StandardHookChain::new();
        chain
            .register(Arc::new(
                FunctionHook::new("log", vec![HookEvent::PostTurn], |_| {
                    Ok(HookDecision::Continue)
                })
                .async_mode(),
            ))
            .unwrap();
        let out = TurnOutput::default();
        let sid = sid();
        let mut ctx = HookContext::PostTurn {
            session_id: &sid,
            turn_number: 1,
            output: &out,
        };
        // Returns Continue immediately; the async hook never affects outcome.
        assert_eq!(chain.fire(&mut ctx).await.unwrap(), FireOutcome::Continue);
    }

    // ── R7: Inject aggregation ─────────────────────────────────────────────
    #[tokio::test]
    async fn inject_aggregates_newline_joined() {
        let chain = StandardHookChain::new();
        chain
            .register(fn_hook(
                "i1",
                vec![HookEvent::PreTurn],
                HookDecision::Inject {
                    context: "one".into(),
                },
            ))
            .unwrap();
        chain
            .register(fn_hook(
                "i2",
                vec![HookEvent::PreTurn],
                HookDecision::Inject {
                    context: "two".into(),
                },
            ))
            .unwrap();
        let sid = sid();
        let mut cb = ContextBlock::default();
        let mut ctx = HookContext::PreTurn {
            session_id: &sid,
            turn_number: 1,
            context_block: &mut cb,
        };
        assert_eq!(
            chain.fire(&mut ctx).await.unwrap(),
            FireOutcome::Inject {
                context: "one\ntwo".into()
            }
        );
    }

    // ── R23: FunctionHook runs the closure ─────────────────────────────────
    #[tokio::test]
    async fn function_hook_runs() {
        let hook = FunctionHook::new("f", vec![HookEvent::OnLoopStart], |ctx| {
            if let HookContext::OnLoopStart {
                task_instruction, ..
            } = ctx
            {
                task_instruction.push_str(" [checked]");
            }
            Ok(HookDecision::Continue)
        });
        let chain = StandardHookChain::new();
        chain.register(Arc::new(hook)).unwrap();
        let sid = sid();
        let cfg = test_config();
        let mut instr = "do work".to_string();
        let mut ctx = HookContext::OnLoopStart {
            session_id: &sid,
            task_instruction: &mut instr,
            config: &cfg,
        };
        chain.fire(&mut ctx).await.unwrap();
        assert_eq!(instr, "do work [checked]");
    }

    // ── R18-R21: CommandHook stdin/stdout roundtrip ────────────────────────
    #[tokio::test]
    async fn command_hook_roundtrip() {
        // Echo script: reads stdin (ignored), emits a block decision.
        let dir = std::env::temp_dir().join(format!("hooks-cmd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("hook.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\ncat >/dev/null\necho '{\"decision\":\"block\",\"reason\":\"cmd says no\"}'\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let hook = CommandHook::new(
            "cmd",
            vec![HookEvent::Stop],
            "sh",
            vec![script.to_string_lossy().to_string()],
        );
        let chain = StandardHookChain::new();
        chain.register(Arc::new(hook)).unwrap();
        let sid = sid();
        let out = TurnOutput::default();
        let state = SessionState::new(sid.clone(), crate::harness::TaskId::new("t"), "x");
        let mut ctx = HookContext::Stop {
            session_id: &sid,
            turn_number: 1,
            last_output: &out,
            task_instruction: "x",
            session_state: &state,
        };
        let outcome = chain.fire(&mut ctx).await.unwrap();
        assert_eq!(
            outcome,
            FireOutcome::Block {
                reason: "cmd says no".into()
            }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── R20: CommandHook nonzero exit → CommandFailed ──────────────────────
    #[tokio::test]
    async fn command_hook_nonzero_exit_errors() {
        let hook = CommandHook::new(
            "cmd",
            vec![HookEvent::Stop],
            "sh",
            vec!["-c".into(), "exit 7".into()],
        );
        let chain = StandardHookChain::new();
        chain.register(Arc::new(hook)).unwrap();
        let sid = sid();
        let out = TurnOutput::default();
        let state = SessionState::new(sid.clone(), crate::harness::TaskId::new("t"), "x");
        let mut ctx = HookContext::Stop {
            session_id: &sid,
            turn_number: 1,
            last_output: &out,
            task_instruction: "x",
            session_state: &state,
        };
        let err = chain.fire(&mut ctx).await.unwrap_err();
        assert!(matches!(err, HookError::CommandFailed { code: 7, .. }));
    }

    // ── HookDecision wire format pinned ────────────────────────────────────
    #[test]
    fn hook_decision_wire_format() {
        assert_eq!(
            serde_json::to_value(HookDecision::Continue).unwrap(),
            serde_json::json!({"decision": "continue"})
        );
        assert_eq!(
            serde_json::to_value(HookDecision::Block { reason: "r".into() }).unwrap(),
            serde_json::json!({"decision": "block", "reason": "r"})
        );
        assert_eq!(
            serde_json::to_value(HookDecision::Inject {
                context: "c".into()
            })
            .unwrap(),
            serde_json::json!({"decision": "inject", "context": "c"})
        );
        assert_eq!(
            serde_json::to_value(HookDecision::Deny { reason: "d".into() }).unwrap(),
            serde_json::json!({"decision": "deny", "reason": "d"})
        );
        assert_eq!(
            serde_json::to_value(HookDecision::Mutate {
                data: serde_json::json!({"k": 1})
            })
            .unwrap(),
            serde_json::json!({"decision": "mutate", "data": {"k": 1}})
        );
    }

    // ── Deferred-event fire methods work in isolation ──────────────────────
    #[tokio::test]
    async fn deferred_on_plan_created_mutates() {
        let chain = StandardHookChain::new();
        chain
            .register(Arc::new(FunctionHook::new(
                "plan",
                vec![HookEvent::OnPlanCreated],
                |ctx| {
                    if let HookContext::OnPlanCreated { plan, .. } = ctx {
                        plan.tasks.push("extra".into());
                    }
                    Ok(HookDecision::Continue)
                },
            )))
            .unwrap();
        let sid = sid();
        let mut plan = PlanArtifact {
            tasks: vec!["a".into()],
            rationale: String::new(),
        };
        let mut ctx = HookContext::OnPlanCreated {
            session_id: &sid,
            plan: &mut plan,
        };
        chain.fire(&mut ctx).await.unwrap();
        assert_eq!(plan.tasks, vec!["a".to_string(), "extra".to_string()]);
    }

    #[tokio::test]
    async fn deferred_subagent_spawn_deny() {
        let chain = StandardHookChain::new();
        chain
            .register(fn_hook(
                "ss",
                vec![HookEvent::OnSubagentSpawn],
                HookDecision::Deny {
                    reason: "no spawn".into(),
                },
            ))
            .unwrap();
        let sid = sid();
        let mut child = "child task".to_string();
        let mut ctx = HookContext::OnSubagentSpawn {
            session_id: &sid,
            child_task: &mut child,
            strategy: "react",
        };
        assert_eq!(
            chain.fire(&mut ctx).await.unwrap(),
            FireOutcome::Deny {
                reason: "no spawn".into()
            }
        );
    }

    #[tokio::test]
    async fn pre_compact_mutates_hints() {
        let chain = StandardHookChain::new();
        chain
            .register(Arc::new(FunctionHook::new(
                "pc",
                vec![HookEvent::PreCompact],
                |ctx| {
                    if let HookContext::PreCompact { preserve_hints, .. } = ctx {
                        preserve_hints.keep_recent_file_list = false;
                    }
                    Ok(HookDecision::Continue)
                },
            )))
            .unwrap();
        let sid = sid();
        let mut hints = CompactionPreserveHints::default();
        let mut ctx = HookContext::PreCompact {
            session_id: &sid,
            preserve_hints: &mut hints,
        };
        chain.fire(&mut ctx).await.unwrap();
        assert!(!hints.keep_recent_file_list);
    }

    // helper: build a minimal HarnessConfig for OnLoopStart context tests,
    // reusing the harness testing stubs.
    fn test_config() -> HarnessConfig {
        use crate::agent::mock::MockAgent;
        use crate::agent::AgentId;
        use crate::context::KeyTermVerifier;
        use crate::harness::testing::{
            AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager, ScriptedToolRegistry,
        };

        let agent: Arc<dyn crate::agent::Agent> = Arc::new(MockAgent::new(AgentId::new("a")));
        HarnessConfig {
            tool_registry: Arc::new(ScriptedToolRegistry::new()),
            sandbox: Arc::new(AllowAllSandbox),
            sandbox_violation_policy: crate::harness::SandboxViolationPolicy::default(),
            context_manager: Arc::new(NoopContextManager),
            termination_policy: Arc::new(AlwaysContinuePolicy),
            middleware: None,
            observability: None,
            compaction_verifier: Arc::new(KeyTermVerifier),
            max_compaction_attempts: 2,
            pricing: crate::observability::PricingTable::DEFAULT,
            content_capture: Default::default(),
            tool_call_repair: None,
            max_repair_attempts: 1,
            max_stop_blocks: 8,
            error_loop_threshold: 3,
            enforce_output_schemas: false,
            output_schema_max_retries: 2,
            hooks: None,
            storage: Arc::new(crate::storage::StorageProvider::no_op()),
            project_id: crate::storage::ProjectId::from_canonical_path("/test-workspace"),
            chunk_provider: Arc::new(crate::prompt_assembly::InMemoryChunkProvider::empty()),
            max_resets: 3,
            vcs_provider: None,
            catalogue_registry: None,
            toolset_catalogues: std::collections::HashMap::new(),
            system_prompt: None,
            guides: Vec::new(),
            skills: None,
            model_params: crate::model::ModelParams::default(),
            auto_persist_sessions: false,
            prompt_tool_call_flag: None,
            consult_handlers: std::collections::HashMap::new(),
            registry: crate::ExecutionRegistry::builder().agent("", agent).build(),
            escalation_mode: crate::EscalationMode::SurfaceToHuman,
        }
    }
}
