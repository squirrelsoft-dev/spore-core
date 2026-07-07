//! ToolRegistry — maintains available tools and dispatches tool calls.
//!
//! Implements issue #4. The registry holds the catalog of `Tool`
//! implementations, validates their JSON schemas at registration time,
//! and dispatches `ToolCall`s coming in from the agent — passing every
//! tool a `SandboxProvider` so that no tool ever touches the environment
//! directly.
//!
//! ## What this component does
//!
//! - Register tools with their schemas (validated up-front)
//! - Manage named `ToolSet` groupings keyed by `TaskPhase`
//! - Return the active schemas for a given phase (sorted by name for
//!   cache-stability)
//! - Dispatch a single call (sandbox-aware) or many calls (concurrent
//!   where `ToolAnnotations` permit)
//! - Expose `has_subagent_tools()` so `SubagentTool::new()` can enforce
//!   the depth-1 rule at construction time
//!
//! ## Storage seam — `ToolContext` (#75)
//!
//! Tools receive a [`ToolContext`] on every dispatch, *in addition to* the
//! `SandboxProvider`. `ToolContext` is the storage seam: `{ session_id:
//! SessionId, run_store: Arc<dyn RunStore> }`. The new `Tool::execute`
//! signature is:
//!
//! ```ignore
//! fn execute<'a>(
//!     &'a self,
//!     call: &'a ToolCall,
//!     sandbox: &'a (dyn SandboxProvider + 'a),
//!     ctx: &'a ToolContext,
//! ) -> BoxFut<'a, ToolOutput>;
//! ```
//!
//! `StandardToolRegistry::dispatch`/`dispatch_all` thread the `ToolContext`
//! through to every tool. The canonical registry takes the `ToolContext` as a
//! dispatch argument; the harness-side [`RealToolRegistry`] bridge (defined
//! below) is constructed per-run with the `SessionId` + `Arc<dyn RunStore>` +
//! `Arc<dyn MemoryStore>` and builds the `ToolContext` itself before forwarding.
//! The harness-loop `ToolRegistry::dispatch(call)` signature is UNCHANGED. `ToolContext` is a
//! struct so future fields are non-breaking; `SandboxProvider` is NOT folded in
//! (storage is additive).
//!
//! ## What this component does NOT do
//!
//! - Retry recoverable failures (middleware concern — issue #11)
//! - Maintain conversation state, budgets, or termination policy
//! - Interpret `ToolOutput::WaitingForHuman` — the registry returns it
//!   verbatim; the harness loop assembles the combined `PausedState`
//!
//! ## Rules enforced here
//!
//! 1. Tools are always dispatched via the registry — never directly.
//! 2. Schemas are validated at registration (basic structural check on
//!    the JSON Schema document).
//! 3. Duplicate tool names → **last-wins upsert** (issue #81, Q1): a second
//!    `register()` for the same name OVERWRITES the first. (Duplicate tool-SET
//!    names still error with `RegistrationError::DuplicateName`.)
//! 4. `ToolAnnotations { destructive: true, read_only: true }` is
//!    contradictory → `RegistrationError::ConflictingAnnotations`.
//! 5. Active `ToolSet` can change between turns (selected by `TaskPhase`).
//! 6. An unregistered tool call → `DispatchError::UnregisteredTool`.
//! 7. Parameters that do not satisfy the schema's declared `required`
//!    fields → `DispatchError::SchemaValidationFailed`.
//! 8. `dispatch_all`:
//!    - Calls whose tools are all `read_only: true` may execute
//!      concurrently.
//!    - Calls whose tools are `destructive: true` or `open_world: true`
//!      execute sequentially.
//!    - Mixed batches are partitioned: the concurrent prefix runs first,
//!      then the sequential remainder, preserving caller-visible order.
//! 9. Subagent depth: `has_subagent_tools()` is wired to a single
//!    `is_subagent_tool` flag on each `Tool` so the rule can be checked at
//!    construction time, not at dispatch time.
//!
//! ## Cross-language note
//!
//! - `ToolCall` is reused from [`crate::model::ToolCall`] (issue #1). The
//!   spec field names map as `tool_name` → `name`, `parameters` → `input`.
//!   This avoids two parallel `ToolCall` shapes shared across the four
//!   language packages.
//! - `ToolOutput` and `ToolResult` are reused from [`crate::harness`].
//!   The harness loop already routes these through `RunResult` and
//!   `PausedState`.
//! - `SandboxProvider` and `SandboxViolation` come from [`crate::harness`]
//!   stubs (canonical type lands with issue #6).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::sync_util::RwLockExt;

use crate::harness::{
    BoxFut, SandboxProvider, SandboxViolation, SessionId, ToolOutput,
    ToolRegistry as HarnessToolRegistry, ToolResult,
};
use crate::model::{ToolCall, ToolSchema as ModelToolSchema};
use crate::storage::{MemoryStore, RunStore};

// ============================================================================
// ToolContext — the storage seam handed to every tool (#75)
// ============================================================================

/// The per-dispatch storage seam handed to every [`Tool::execute`] call,
/// alongside (but separate from) the [`SandboxProvider`]. It carries the
/// minimum a tool needs to persist durable state via the storage layer:
///
///   - `session_id`   — the run's [`SessionId`], the key namespace for
///     EPHEMERAL state that SHOULD reset per Ralph window (conversation/session
///     state, `active_skills`).
///   - `project_id`   — the stable [`ProjectId`] (#142), the key namespace for
///     DURABLE artifacts (the `task_list`, plan artifact, …) that must survive a
///     Ralph window reset (fresh `SessionId` per window) and process restarts.
///     Durable tools key the [`RunStore`] by `project_id().namespace()` instead
///     of `session_id` so the prior window's list is re-read, not orphaned.
///   - `run_store`    — the [`RunStore`] domain of the configured provider.
///   - `memory_store` — the [`MemoryStore`] domain (#78). Scope-aware: the
///     tool passes a [`crate::storage::StorageScope`] on every call. For a
///     composite provider this is the scope-routing memory slot; for the
///     never-null contract it is at worst a [`NoOpStorageProvider`]. `MemoryTool`
///     (#82) picks up this already-threaded seam.
///
/// It is a **struct** (not a tuple/pair) so future fields can be added without
/// breaking the trait signature again. The `SandboxProvider` is intentionally
/// NOT folded in here — storage is additive; tools still receive the sandbox as
/// its own parameter (some tools need the filesystem sandbox and no storage).
#[derive(Clone)]
pub struct ToolContext {
    session_id: SessionId,
    project_id: crate::storage::ProjectId,
    run_store: Arc<dyn RunStore>,
    memory_store: Arc<dyn MemoryStore>,
    /// Optional run-level live-progress sink (pi-style bash streaming). Set by
    /// the harness bridge ([`RealToolRegistry::with_progress`]) when the run
    /// has a stream sink; tools derive a per-call emitter via
    /// [`command_progress`](Self::command_progress). `None` (default) means no
    /// consumer is listening — tools skip streaming entirely.
    progress: Option<ToolProgressSink>,
}

/// One live-progress update from a RUNNING tool, labeled with the call it
/// belongs to. The bridge maps it onto
/// [`StreamEvent::ToolProgress`](crate::harness::StreamEvent) for UIs.
#[derive(Debug, Clone)]
pub struct ToolProgressUpdate {
    /// Tool name (e.g. `bash_command`).
    pub tool: String,
    /// The originating `ToolCall::id`, so a UI can update the right block.
    pub call_id: String,
    /// Sanitized tail snapshot of the output so far — REPLACES the previous
    /// preview (see [`CommandProgressSink`](crate::harness::CommandProgressSink)).
    pub preview: String,
}

/// Run-level sink for [`ToolProgressUpdate`]s. Must not block (called from the
/// executor's ~100ms wait-loop tick).
pub type ToolProgressSink = Arc<dyn Fn(ToolProgressUpdate) + Send + Sync>;

impl ToolContext {
    /// Build a context from the run's session id, the stable project id (#142),
    /// and the storage seams.
    pub fn new(
        session_id: SessionId,
        project_id: crate::storage::ProjectId,
        run_store: Arc<dyn RunStore>,
        memory_store: Arc<dyn MemoryStore>,
    ) -> Self {
        Self {
            session_id,
            project_id,
            run_store,
            memory_store,
            progress: None,
        }
    }

    /// Install the run-level live-progress sink (builder-style). Wired by the
    /// harness bridge when the run has a stream sink; see the field doc.
    pub fn with_progress(mut self, sink: ToolProgressSink) -> Self {
        self.progress = Some(sink);
        self
    }

    /// Derive the per-call command-progress emitter for `call`: wraps the
    /// run-level sink (when one is installed) with the call's identity so the
    /// executor only ever deals in plain preview strings. `None` when no
    /// consumer is listening — pass it straight to
    /// [`SandboxProvider::execute_command_streaming`](crate::harness::SandboxProvider::execute_command_streaming).
    pub fn command_progress(&self, call: &ToolCall) -> Option<crate::harness::CommandProgressSink> {
        let sink = self.progress.clone()?;
        let tool = call.name.clone();
        let call_id = call.id.clone();
        Some(Arc::new(move |preview: String| {
            sink(ToolProgressUpdate {
                tool: tool.clone(),
                call_id: call_id.clone(),
                preview,
            })
        }))
    }

    /// The session id keying this run's EPHEMERAL state (resets per Ralph window).
    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    /// The stable project id (#142) keying this run's DURABLE artifacts — stable
    /// across Ralph window resets and process restarts. Durable tools key the
    /// [`RunStore`] by `project_id().namespace()`.
    pub fn project_id(&self) -> &crate::storage::ProjectId {
        &self.project_id
    }

    /// The run-store domain a tool persists durable state through.
    pub fn run_store(&self) -> &Arc<dyn RunStore> {
        &self.run_store
    }

    /// The memory-store domain a tool reads/writes episodic memory through
    /// (#78). Scope-aware — the caller passes a
    /// [`crate::storage::StorageScope`] on each call.
    pub fn memory_store(&self) -> &Arc<dyn MemoryStore> {
        &self.memory_store
    }
}

// ============================================================================
// ToolAnnotations & ToolSchema
// ============================================================================

/// Behavioural annotations attached to a registered tool. They drive the
/// `dispatch_all` concurrency split and the auto-derived `RiskLevel` used by
/// `PermissionMiddleware`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ToolAnnotations {
    #[serde(default)]
    pub read_only: bool,
    #[serde(default)]
    pub destructive: bool,
    #[serde(default)]
    pub idempotent: bool,
    #[serde(default)]
    pub open_world: bool,
}

/// Canonical schema for a registered tool. Distinct from
/// [`crate::model::ToolSchema`] (which is the minimal subset shipped to the
/// LLM) — this one carries `ToolAnnotations` and is the registry-side type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    #[serde(default)]
    pub annotations: ToolAnnotations,
}

impl ToolSchema {
    /// Project to the slimmer `model::ToolSchema` used in `ModelRequest`.
    pub fn to_model_schema(&self) -> crate::model::ToolSchema {
        crate::model::ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.parameters.clone(),
        }
    }
}

// ============================================================================
// TaskPhase & ToolSet
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskPhase {
    Initialization,
    Planning,
    Execution,
    Verification,
    Cleanup,
}

/// A named grouping of tools. `phase` is `None` if the set is always active.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSet {
    pub name: String,
    pub tools: Vec<String>,
    #[serde(default)]
    pub phase: Option<TaskPhase>,
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[non_exhaustive]
pub enum RegistrationError {
    #[error("invalid schema for tool {tool}: {reason}")]
    InvalidSchema { tool: String, reason: String },

    #[error("tool {tool} already registered")]
    DuplicateName { tool: String },

    #[error("conflicting annotations for tool {tool}: {reason}")]
    ConflictingAnnotations { tool: String, reason: String },
}

#[derive(Debug, Error, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[non_exhaustive]
pub enum DispatchError {
    #[error("unregistered tool: {name}")]
    UnregisteredTool { name: String },

    #[error("schema validation failed for {tool}: {reason}")]
    SchemaValidationFailed { tool: String, reason: String },

    #[error("sandbox violation: {0:?}")]
    SandboxViolation(SandboxViolation),

    #[error("tool {tool} failed: {error}")]
    ToolExecutionFailed { tool: String, error: String },
}

// ============================================================================
// Tool trait
// ============================================================================

/// A single tool implementation. Tools are stateless and receive a
/// `SandboxProvider` (environment seam) and a [`ToolContext`] (storage seam) on
/// every dispatch. The trait is `dyn`-compatible: concrete impls return
/// `BoxFut` so `Box<dyn Tool>` works.
pub trait Tool: Send + Sync {
    /// Tool name — must match the registered `ToolSchema.name`.
    fn name(&self) -> &str;

    /// `true` for `SubagentTool`. Defaults to `false`. Used by
    /// `ToolRegistry::has_subagent_tools()` to enforce the depth-1 rule
    /// at construction time.
    fn is_subagent_tool(&self) -> bool {
        false
    }

    /// `true` if this tool's output may exceed inline budgets and should be
    /// routed through `SandboxProvider::handle_large_output`. Defaults to
    /// `false`. Standard read/exec/search/git/http tools override to `true`.
    fn may_produce_large_output(&self) -> bool {
        false
    }

    /// Execute the tool with validated input. The `SandboxProvider` is the only
    /// path to the environment; the [`ToolContext`] is the only path to durable
    /// storage (`RunStore`, keyed by the run's `SessionId`).
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        sandbox: &'a (dyn SandboxProvider + 'a),
        ctx: &'a ToolContext,
    ) -> BoxFut<'a, ToolOutput>;
}

// ============================================================================
// ToolRegistry trait
// ============================================================================

/// Canonical registry trait. Object-safe via `BoxFut`.
pub trait ToolRegistry: Send + Sync {
    fn register(&self, tool: Box<dyn Tool>, schema: ToolSchema) -> Result<(), RegistrationError>;

    fn register_set(&self, set: ToolSet) -> Result<(), RegistrationError>;

    /// Schemas active in the given phase (sorted by name for cache stability).
    /// `None` → all registered schemas.
    fn active_schemas(&self, phase: Option<TaskPhase>) -> Vec<ToolSchema>;

    fn dispatch<'a>(
        &'a self,
        call: ToolCall,
        sandbox: &'a (dyn SandboxProvider + 'a),
        ctx: &'a ToolContext,
    ) -> BoxFut<'a, Result<ToolResult, DispatchError>>;

    fn dispatch_all<'a>(
        &'a self,
        calls: Vec<ToolCall>,
        sandbox: &'a (dyn SandboxProvider + 'a),
        ctx: &'a ToolContext,
    ) -> BoxFut<'a, Vec<Result<ToolResult, DispatchError>>>;

    fn has_subagent_tools(&self) -> bool;
}

// ============================================================================
// StandardToolRegistry — canonical implementation
// ============================================================================

struct Registered {
    tool: Arc<dyn Tool>,
    schema: ToolSchema,
}

/// Default in-memory registry. Concurrency-safe: register/lookup go through a
/// `std::sync::RwLock`. The lock is held briefly (lookup + clone of `Arc`) and
/// never across an `.await`, so a blocking lock is correct and avoids the
/// panic-on-contention of `try_read()`/`try_write()`; the tool itself executes
/// lock-free.
pub struct StandardToolRegistry {
    tools: RwLock<HashMap<String, Registered>>,
    sets: RwLock<Vec<ToolSet>>,
}

impl StandardToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
            sets: RwLock::new(Vec::new()),
        }
    }

    fn validate_schema(schema: &ToolSchema) -> Result<(), RegistrationError> {
        if schema.name.is_empty() {
            return Err(RegistrationError::InvalidSchema {
                tool: schema.name.clone(),
                reason: "name must not be empty".into(),
            });
        }
        // Basic structural check: parameters must be a JSON object with a
        // `type` key. Full JSON Schema validation is intentionally not
        // bundled here — see `validate_input` for per-call enforcement.
        let obj =
            schema
                .parameters
                .as_object()
                .ok_or_else(|| RegistrationError::InvalidSchema {
                    tool: schema.name.clone(),
                    reason: "parameters must be a JSON object".into(),
                })?;
        if !obj.contains_key("type") {
            return Err(RegistrationError::InvalidSchema {
                tool: schema.name.clone(),
                reason: "parameters must declare a top-level `type`".into(),
            });
        }
        Ok(())
    }

    fn validate_annotations(schema: &ToolSchema) -> Result<(), RegistrationError> {
        let a = schema.annotations;
        if a.read_only && a.destructive {
            return Err(RegistrationError::ConflictingAnnotations {
                tool: schema.name.clone(),
                reason: "read_only and destructive are mutually exclusive".into(),
            });
        }
        Ok(())
    }

    /// Best-effort per-call schema validation. Checks that any `required`
    /// fields declared on the parameter schema are present in the call's
    /// `input` object. Deeper JSON Schema validation can be plugged in later.
    fn validate_input(schema: &ToolSchema, call: &ToolCall) -> Result<(), DispatchError> {
        let Some(input_obj) = call.input.as_object() else {
            return Err(DispatchError::SchemaValidationFailed {
                tool: schema.name.clone(),
                reason: "input must be a JSON object".into(),
            });
        };
        let Some(params_obj) = schema.parameters.as_object() else {
            return Ok(());
        };
        if let Some(required) = params_obj.get("required").and_then(|v| v.as_array()) {
            for field in required {
                if let Some(name) = field.as_str() {
                    if !input_obj.contains_key(name) {
                        return Err(DispatchError::SchemaValidationFailed {
                            tool: schema.name.clone(),
                            reason: format!("missing required field `{name}`"),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn lookup(&self, name: &str) -> Option<(Arc<dyn Tool>, ToolSchema)> {
        let guard = self.tools.read_recover();
        guard.get(name).map(|r| (r.tool.clone(), r.schema.clone()))
    }
}

impl Default for StandardToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry for StandardToolRegistry {
    fn register(&self, tool: Box<dyn Tool>, schema: ToolSchema) -> Result<(), RegistrationError> {
        if tool.name() != schema.name {
            return Err(RegistrationError::InvalidSchema {
                tool: schema.name.clone(),
                reason: format!(
                    "tool name `{}` does not match schema name `{}`",
                    tool.name(),
                    schema.name
                ),
            });
        }
        Self::validate_schema(&schema)?;
        Self::validate_annotations(&schema)?;
        let mut guard = self.tools.write_recover();
        // Last-wins upsert (issue #81, Q1): registering a tool whose name is
        // already present OVERWRITES the prior registration. This is what lets
        // an architect override a standard tool by registering their own after
        // `StandardTools::coding_set()` / `full_set()`. `RegistrationError`
        // remains for the schema/annotation/name-mismatch cases above (and for
        // `register_set`), but a duplicate name is no longer an error.
        let arc: Arc<dyn Tool> = Arc::from(tool);
        guard.insert(schema.name.clone(), Registered { tool: arc, schema });
        Ok(())
    }

    fn register_set(&self, set: ToolSet) -> Result<(), RegistrationError> {
        if set.name.is_empty() {
            return Err(RegistrationError::InvalidSchema {
                tool: set.name.clone(),
                reason: "tool set name must not be empty".into(),
            });
        }
        let mut guard = self.sets.write_recover();
        if guard.iter().any(|s| s.name == set.name) {
            return Err(RegistrationError::DuplicateName {
                tool: set.name.clone(),
            });
        }
        guard.push(set);
        Ok(())
    }

    fn active_schemas(&self, phase: Option<TaskPhase>) -> Vec<ToolSchema> {
        let tools = self.tools.read_recover();
        let sets = self.sets.read_recover();

        let mut out: Vec<ToolSchema> = match phase {
            None => tools.values().map(|r| r.schema.clone()).collect(),
            Some(p) => {
                // Union of: sets matching this phase OR sets with no phase
                // (always-active). If no set matches at all, fall back to the
                // full catalog — registering zero sets must not silently mask
                // every tool.
                let matching: Vec<&ToolSet> = sets
                    .iter()
                    .filter(|s| s.phase.is_none() || s.phase == Some(p))
                    .collect();
                if matching.is_empty() {
                    tools.values().map(|r| r.schema.clone()).collect()
                } else {
                    let mut names: std::collections::BTreeSet<&str> =
                        std::collections::BTreeSet::new();
                    for s in matching {
                        for t in &s.tools {
                            names.insert(t.as_str());
                        }
                    }
                    names
                        .iter()
                        .filter_map(|n| tools.get(*n).map(|r| r.schema.clone()))
                        .collect()
                }
            }
        };
        // Cache-stability: schemas always sorted by name (see spec, cache rules).
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    fn dispatch<'a>(
        &'a self,
        call: ToolCall,
        sandbox: &'a (dyn SandboxProvider + 'a),
        ctx: &'a ToolContext,
    ) -> BoxFut<'a, Result<ToolResult, DispatchError>> {
        Box::pin(async move {
            let (tool, schema) = match self.lookup(&call.name) {
                Some(t) => t,
                None => {
                    return Err(DispatchError::UnregisteredTool {
                        name: call.name.clone(),
                    });
                }
            };

            // Sandbox validation. PathEscape / NetworkViolation are Layer 1
            // — the registry surfaces the violation as a DispatchError so
            // the harness can route it.
            if let Err(violation) = sandbox.validate(&call).await {
                return Err(DispatchError::SandboxViolation(violation));
            }

            Self::validate_input(&schema, &call)?;

            let output = tool.execute(&call, sandbox, ctx).await;
            Ok(ToolResult {
                call_id: call.id.clone(),
                output,
            })
        })
    }

    fn dispatch_all<'a>(
        &'a self,
        calls: Vec<ToolCall>,
        sandbox: &'a (dyn SandboxProvider + 'a),
        ctx: &'a ToolContext,
    ) -> BoxFut<'a, Vec<Result<ToolResult, DispatchError>>> {
        Box::pin(async move {
            // Classify each call. Unknown tools are scheduled sequentially so
            // their error surfaces deterministically alongside any other
            // sequential failures.
            let mut classifications: Vec<bool> = Vec::with_capacity(calls.len()); // true = concurrent
            for call in &calls {
                let concurrent = match self.lookup(&call.name) {
                    Some((_, schema)) => {
                        let a = schema.annotations;
                        a.read_only && !a.destructive && !a.open_world
                    }
                    None => false,
                };
                classifications.push(concurrent);
            }

            // Partition into (concurrent_indices, sequential_indices) while
            // preserving original ordering on output.
            let mut concurrent_idx: Vec<usize> = Vec::new();
            let mut sequential_idx: Vec<usize> = Vec::new();
            for (i, &c) in classifications.iter().enumerate() {
                if c {
                    concurrent_idx.push(i);
                } else {
                    sequential_idx.push(i);
                }
            }

            let mut results: Vec<Option<Result<ToolResult, DispatchError>>> =
                (0..calls.len()).map(|_| None).collect();

            // Concurrent batch: join_all on the read-only subset.
            if !concurrent_idx.is_empty() {
                let futs = concurrent_idx
                    .iter()
                    .map(|&i| self.dispatch(calls[i].clone(), sandbox, ctx));
                let outs = futures_util::future::join_all(futs).await;
                for (slot, out) in concurrent_idx.iter().zip(outs) {
                    results[*slot] = Some(out);
                }
            }

            // Sequential batch.
            for i in sequential_idx {
                let out = self.dispatch(calls[i].clone(), sandbox, ctx).await;
                results[i] = Some(out);
            }

            results
                .into_iter()
                .map(|o| o.expect("slot filled"))
                .collect()
        })
    }

    fn has_subagent_tools(&self) -> bool {
        let guard = self.tools.read_recover();
        guard.values().any(|r| r.tool.is_subagent_tool())
    }
}

// ============================================================================
// Mock tools (test-only)
// ============================================================================

#[cfg(any(test, feature = "test-utils"))]
pub mod mock {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Echo tool — returns its input as JSON-string content. `read_only: true`.
    pub struct EchoTool {
        name: String,
        pub calls: AtomicUsize,
    }

    impl EchoTool {
        pub fn new(name: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Tool for EchoTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn execute<'a>(
            &'a self,
            call: &'a ToolCall,
            _sandbox: &'a (dyn SandboxProvider + 'a),
            _ctx: &'a ToolContext,
        ) -> BoxFut<'a, ToolOutput> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let content = call.input.to_string();
            Box::pin(async move {
                ToolOutput::Success {
                    content,
                    truncated: false,
                }
            })
        }
    }

    /// Failing tool — returns a recoverable error.
    pub struct FailingTool {
        name: String,
    }
    impl FailingTool {
        pub fn new(name: impl Into<String>) -> Self {
            Self { name: name.into() }
        }
    }
    impl Tool for FailingTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn execute<'a>(
            &'a self,
            _call: &'a ToolCall,
            _sandbox: &'a (dyn SandboxProvider + 'a),
            _ctx: &'a ToolContext,
        ) -> BoxFut<'a, ToolOutput> {
            Box::pin(async move {
                ToolOutput::Error {
                    message: "boom".into(),
                    recoverable: true,
                }
            })
        }
    }

    /// Subagent-flagged tool.
    pub struct SubagentMock {
        name: String,
    }
    impl SubagentMock {
        pub fn new(name: impl Into<String>) -> Self {
            Self { name: name.into() }
        }
    }
    impl Tool for SubagentMock {
        fn name(&self) -> &str {
            &self.name
        }
        fn is_subagent_tool(&self) -> bool {
            true
        }
        fn execute<'a>(
            &'a self,
            _call: &'a ToolCall,
            _sandbox: &'a (dyn SandboxProvider + 'a),
            _ctx: &'a ToolContext,
        ) -> BoxFut<'a, ToolOutput> {
            Box::pin(async move {
                ToolOutput::Success {
                    content: "subagent done".into(),
                    truncated: false,
                }
            })
        }
    }

    /// Build a throwaway [`ToolContext`] for tests: a fresh in-memory run store
    /// and a fixed test session id. Available to every tool's `#[cfg(test)]`
    /// module via `crate::tool_registry::mock::test_ctx`.
    pub fn test_ctx() -> ToolContext {
        use crate::storage::{InMemoryStorageProvider, ProjectId};
        // One in-memory backend serves both the run and (scope-aware) memory
        // seams in tests.
        let backend = Arc::new(InMemoryStorageProvider::new());
        ToolContext::new(
            SessionId::new("test-session"),
            ProjectId::from_canonical_path("/test-project"),
            backend.clone(),
            backend,
        )
    }

    /// Permissive sandbox stub — accepts everything.
    pub struct AllowAllSandbox;
    impl SandboxProvider for AllowAllSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async move { Ok(()) })
        }
    }

    /// Denying sandbox stub — rejects everything with PathEscape.
    pub struct DenyAllSandbox;
    impl SandboxProvider for DenyAllSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async move {
                Err(SandboxViolation::PathEscape {
                    path: "denied".into(),
                })
            })
        }
    }
}

// ============================================================================
// RealToolRegistry — the bridge between the two ToolRegistry traits
// ============================================================================

/// Bridges the harness-loop [`HarnessToolRegistry`](crate::harness::ToolRegistry)
/// onto this canonical [`ToolRegistry`] (a populated [`StandardToolRegistry`]).
///
/// This is **the production wiring** for running catalogue / [`Tool`] -based
/// tools inside the harness — not test scaffolding. The harness loop calls
/// `dispatch(ToolCall) -> ToolOutput` with no sandbox or storage; this bridge
/// forwards to the inner registry's `dispatch(call, sandbox, ctx)`, threading
/// the [`SandboxProvider`] and a per-run [`ToolContext`] (storage seam, #75) it
/// was constructed with. A [`DispatchError`] becomes a **recoverable**
/// [`ToolOutput::Error`] so the loop appends it as a tool result and lets the
/// agent adapt rather than halting.
///
/// It is built **once per run**: `session_id`, `run_store`, and `memory_store`
/// are injected at construction (the run's `SessionId` is only known at
/// `run()`-time) and used to build the [`ToolContext`] forwarded on every
/// dispatch. `HarnessBuilder` wires this automatically when catalogue tools are
/// added via `.tool()` / `.tools()`; construct it directly only when supplying
/// your own [`StandardToolRegistry`].
pub struct RealToolRegistry {
    inner: Arc<StandardToolRegistry>,
    sandbox: Arc<dyn SandboxProvider>,
    ctx: ToolContext,
    schemas: Vec<ModelToolSchema>,
}

impl RealToolRegistry {
    /// Build a bridge over an already-populated [`StandardToolRegistry`].
    ///
    /// `session_id` + `project_id` + `run_store` + `memory_store` are injected
    /// here (the bridge is built once per run) and used to construct the
    /// [`ToolContext`] forwarded on every dispatch — see the storage-seam note on
    /// the type. `project_id` (#142) is the STABLE durable namespace; `session_id`
    /// is the per-window ephemeral key.
    pub fn new(
        inner: Arc<StandardToolRegistry>,
        sandbox: Arc<dyn SandboxProvider>,
        session_id: SessionId,
        project_id: crate::storage::ProjectId,
        run_store: Arc<dyn RunStore>,
        memory_store: Arc<dyn MemoryStore>,
    ) -> Self {
        // Snapshot the model-facing schemas (sorted by name) once at
        // construction; the catalogue is fixed for a run.
        let schemas = inner
            .active_schemas(None)
            .iter()
            .map(ToolSchema::to_model_schema)
            .collect();
        Self {
            inner,
            sandbox,
            ctx: ToolContext::new(session_id, project_id, run_store, memory_store),
            schemas,
        }
    }

    /// Install a run-level live-progress sink on the bridged [`ToolContext`]
    /// (builder-style, `None` is a no-op). The harness wires this from the
    /// run's stream sink so running tools can emit
    /// [`StreamEvent::ToolProgress`](crate::harness::StreamEvent) previews.
    pub fn with_progress(mut self, sink: Option<ToolProgressSink>) -> Self {
        if let Some(sink) = sink {
            self.ctx = self.ctx.clone().with_progress(sink);
        }
        self
    }

    /// The model-facing tool schemas, sorted by name.
    pub fn model_schemas(&self) -> Vec<ModelToolSchema> {
        self.schemas.clone()
    }

    /// The [`ToolContext`] this bridge threads into every dispatch — exposes the
    /// `session_id`, `run_store`, and (#78) `memory_store` seams it was wired
    /// with. Lets callers verify the storage seams are live.
    pub fn tool_context(&self) -> &ToolContext {
        &self.ctx
    }
}

impl HarnessToolRegistry for RealToolRegistry {
    fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            match self.inner.dispatch(call, &*self.sandbox, &self.ctx).await {
                Ok(result) => result.output,
                Err(err) => ToolOutput::Error {
                    message: format!("dispatch failed: {err}"),
                    // Recoverable so the loop appends the error and lets the
                    // agent adapt — S4 depends on this.
                    recoverable: true,
                },
            }
        })
    }

    fn is_always_halt(&self, _tool_name: &str) -> bool {
        // No bridged tool is always-halt — S4 needs recoverable failure.
        false
    }

    fn schemas(&self) -> Vec<ModelToolSchema> {
        self.schemas.clone()
    }
}

// ============================================================================
// ToolSearchRegistry — deferred (search-on-demand) tool exposure
// ============================================================================

/// A named group of deferred tools, advertised as a single cheap HEADER.
///
/// The model sees the header (name + description) in its tool list every turn —
/// the menu of capability groups available — and "opens" the namespace by CALLING
/// it (a zero-arg dispatch) to load the member tools into the active set; it can
/// also reach a member directly via `tool_search`. This mirrors the
/// `toolNamespace` grouping in agent SDKs, realized client-side here because local
/// models have no native tool-search. For an MCP future, one server maps naturally
/// to one namespace.
///
/// Versus advertising a lone `tool_search`, headers give a weaker model the
/// category menu UP FRONT (more reliable discovery) while keeping advertised prose
/// tiny (one terse line per group).
#[derive(Debug, Clone)]
pub struct ToolNamespace {
    /// The namespace handle — advertised as a callable header; calling it loads the
    /// member tools. MUST be distinct from every catalogue tool name (a colliding
    /// namespace is dropped at construction so it never shadows a real tool).
    pub name: String,
    /// One terse line describing the group — what the model sees in its tool list.
    pub description: String,
    /// Member tool names (catalogue tools deferred behind this header).
    pub tools: Vec<String>,
}

impl ToolNamespace {
    /// A namespace grouping `tools` under a `name` header with a one-line
    /// `description`.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        tools: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            tools: tools.into_iter().map(Into::into).collect(),
        }
    }
}

/// Configuration for a [`ToolSearchRegistry`]: which tools are always advertised,
/// how much description prose the advertised set may carry, and how many tools a
/// single `tool_search` call surfaces.
///
/// The motivating constraint (see looper's local-vLLM notes) is that a local
/// model's NATIVE tool-calling collapses once the total volume of natural-language
/// tool-DESCRIPTION text advertised at once crosses a model-specific cliff — NOT
/// the tool count or byte size, the *description prose*. A catalogue with many
/// description-rich tools (or a 75-tool MCP server) buries native tool-calling far
/// past that cliff. The deferred-tool pattern keeps the advertised set small and
/// description-bounded, loading more tools on demand, so native tool-calling keeps
/// working at catalogue scale.
#[derive(Debug, Clone)]
pub struct ToolSearchConfig {
    /// Names of tools ALWAYS advertised (never deferred, never evicted). Tools
    /// outside this list start hidden and are surfaced on demand by `tool_search`
    /// or by opening the [`namespace`](Self::namespaces) they belong to.
    /// Names absent from the wrapped catalogue are ignored.
    pub core: Vec<String>,
    /// Deferred-tool groups, each advertised as a single cheap header the model can
    /// open. Members are catalogue tools NOT in `core`; a tool in a namespace is
    /// still independently findable via `tool_search`. Empty (the default) is a
    /// flat catalogue — every non-core tool is standalone deferred.
    pub namespaces: Vec<ToolNamespace>,
    /// Soft ceiling on the TOTAL description bytes across the advertised set
    /// (core + namespace headers + the search meta-tool + currently-loaded tools).
    /// Loaded tools are LRU-evicted to keep the total at or below this. The
    /// always-on baseline (core + headers + meta-tool) is advertised even if it
    /// alone exceeds the budget (correctness over the cliff — size `core` and the
    /// namespace headers sanely). Calibrate against the consumer's real schemas to
    /// sit BELOW the native-tool-calling description-volume cliff.
    pub description_budget: usize,
    /// Maximum tools a single `tool_search` call — or one namespace open — loads
    /// into the active set.
    pub search_top_k: usize,
    /// Name of the synthetic search meta-tool. Defaults to `tool_search`.
    pub search_tool_name: String,
}

impl ToolSearchConfig {
    /// New config with the given always-on core tool names and conservative
    /// defaults (`description_budget` 4096, `search_top_k` 5, search tool named
    /// `tool_search`). Consumers should calibrate `description_budget` against
    /// their own model/schemas.
    pub fn new(core: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            core: core.into_iter().map(Into::into).collect(),
            namespaces: Vec::new(),
            description_budget: 4096,
            search_top_k: 5,
            search_tool_name: "tool_search".into(),
        }
    }

    /// Add a deferred-tool namespace (group). Chainable; call once per group.
    pub fn namespace(mut self, ns: ToolNamespace) -> Self {
        self.namespaces.push(ns);
        self
    }

    /// Replace the namespace list wholesale.
    pub fn namespaces(mut self, namespaces: impl IntoIterator<Item = ToolNamespace>) -> Self {
        self.namespaces = namespaces.into_iter().collect();
        self
    }

    /// Set the advertised-description byte ceiling.
    pub fn description_budget(mut self, bytes: usize) -> Self {
        self.description_budget = bytes;
        self
    }

    /// Set the max tools surfaced per `tool_search` call.
    pub fn search_top_k(mut self, k: usize) -> Self {
        self.search_top_k = k;
        self
    }

    /// Rename the search meta-tool.
    pub fn search_tool_name(mut self, name: impl Into<String>) -> Self {
        self.search_tool_name = name.into();
        self
    }
}

/// A harness-loop [`ToolRegistry`](crate::harness::ToolRegistry) wrapper that
/// advertises only a small, description-bounded ACTIVE set — the configured core
/// tools plus a `tool_search` meta-tool — and loads the rest of the catalogue on
/// demand.
///
/// `schemas()` is re-read by the harness every turn, so the wrapper grows/shrinks
/// the advertised set turn-to-turn via interior mutability:
/// * `tool_search(query)` keyword-matches the full catalogue (names + descriptions),
///   loads the top-K hits into the active set, and returns their schemas in the
///   tool result so the model can call them next turn.
/// * any tool actually dispatched is touched (kept warm, MRU).
/// * the active set is LRU-evicted to keep total advertised description bytes at or
///   below [`ToolSearchConfig::description_budget`].
///
/// It is constructed once per turn-loop window (alongside the wrapped
/// [`RealToolRegistry`]), so a search loaded mid-window stays loaded for the rest
/// of that window and resets on the next window (fresh task ⇒ re-search).
///
/// Deferred tools may be grouped into [`ToolNamespace`]s: each is advertised as a
/// cheap header the model opens (by calling it) to load the group's tools, giving
/// a weaker model the menu of capability groups up front. Namespaced and standalone
/// deferred tools coexist, and every deferred tool stays findable via `tool_search`.
///
/// `dispatch` of any non-search, non-namespace tool delegates straight to the inner
/// registry, so a model that calls a catalogue tool that is not currently advertised
/// (e.g. one it just learned of, or one that was evicted) STILL succeeds — exposure
/// is a hint to the model, never a hard gate on dispatch.
pub struct ToolSearchRegistry {
    inner: Arc<dyn HarnessToolRegistry>,
    /// Full catalogue snapshot (every tool the inner registry advertises).
    catalogue: Vec<ModelToolSchema>,
    /// Core tool names present in the catalogue, in catalogue order.
    core: Vec<String>,
    /// Validated namespaces: members exist in the catalogue and are non-core, and
    /// the namespace name does not collide with any catalogue tool name.
    namespaces: Vec<ToolNamespace>,
    /// Precomputed synthetic header schemas (one per namespace), advertised every
    /// turn alongside `core` and the search meta-tool.
    namespace_headers: Vec<ModelToolSchema>,
    config: ToolSearchConfig,
    /// Synthetic search meta-tool schema, advertised every turn.
    search_schema: ModelToolSchema,
    /// Loaded (searched-in / namespace-opened) tool names in LRU order: front =
    /// most-recently-used.
    loaded: std::sync::Mutex<std::collections::VecDeque<String>>,
}

impl ToolSearchRegistry {
    /// Wrap an inner harness-loop registry (typically a [`RealToolRegistry`] over
    /// the full catalogue). The inner registry's `schemas()` defines the catalogue
    /// available to search; its `dispatch()` routes every real tool.
    pub fn new(inner: Arc<dyn HarnessToolRegistry>, config: ToolSearchConfig) -> Self {
        let catalogue = inner.schemas();
        let in_catalogue = |n: &str| catalogue.iter().any(|s| s.name == n);
        // Core names that actually exist in the catalogue, catalogue order, deduped.
        let mut core: Vec<String> = Vec::new();
        for s in &catalogue {
            if config.core.iter().any(|c| c == &s.name) && !core.iter().any(|c| c == &s.name) {
                core.push(s.name.clone());
            }
        }
        let is_core = |n: &str| core.iter().any(|c| c == n);
        // Validate namespaces: drop a name that collides with a catalogue tool (it
        // would shadow the real tool on dispatch) or with an earlier namespace;
        // keep only members that exist in the catalogue and are non-core, deduped.
        let mut namespaces: Vec<ToolNamespace> = Vec::new();
        for ns in &config.namespaces {
            if in_catalogue(&ns.name) || namespaces.iter().any(|k| k.name == ns.name) {
                continue;
            }
            let mut tools: Vec<String> = Vec::new();
            for t in &ns.tools {
                if in_catalogue(t) && !is_core(t) && !tools.iter().any(|x| x == t) {
                    tools.push(t.clone());
                }
            }
            if tools.is_empty() {
                continue;
            }
            namespaces.push(ToolNamespace { name: ns.name.clone(), description: ns.description.clone(), tools });
        }
        let namespace_headers = namespaces.iter().map(Self::header_schema).collect();
        let search_schema = Self::build_search_schema(&config.search_tool_name);
        Self {
            inner,
            catalogue,
            core,
            namespaces,
            namespace_headers,
            config,
            search_schema,
            loaded: std::sync::Mutex::new(std::collections::VecDeque::new()),
        }
    }

    /// Synthetic, always-advertised header for a namespace. Zero-arg: calling it
    /// opens the group. The "(call to load …)" hint nudges a weak model to open it.
    fn header_schema(ns: &ToolNamespace) -> ModelToolSchema {
        ModelToolSchema {
            name: ns.name.clone(),
            description: format!("{} (call to load this group's tools)", ns.description),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
        }
    }

    fn build_search_schema(name: &str) -> ModelToolSchema {
        // Kept deliberately TERSE: this description is advertised every turn and
        // counts against the same description-prose budget the cliff cares about,
        // so verbosity here would cancel out the savings from deferring tools.
        ModelToolSchema {
            name: name.to_string(),
            description: "Load more tools by keyword when you need a capability you don't \
                currently have (e.g. run command, web search, task list)."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Keywords for the capability you need." }
                },
                "required": ["query"]
            }),
        }
    }

    /// Whether `name` is a configured (and catalogue-present) core tool.
    fn is_core(&self, name: &str) -> bool {
        self.core.iter().any(|c| c == name)
    }

    /// Move `name` to the front of the LRU (most-recently-used). No-op for the
    /// search tool, core tools, and names absent from the catalogue.
    fn touch(&self, name: &str) {
        if name == self.config.search_tool_name
            || self.is_core(name)
            || !self.catalogue.iter().any(|s| s.name == name)
        {
            return;
        }
        let mut g = self.loaded.lock().expect("loaded lock poisoned");
        g.retain(|n| n != name);
        g.push_front(name.to_string());
    }

    /// Keyword search over the non-core catalogue. Scores `+3` per query term
    /// found in a tool name and `+1` per term in its description; returns up to
    /// `search_top_k` names with a positive score, best first (ties by name).
    fn search(&self, query: &str) -> Vec<String> {
        let terms: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= 2)
            .map(|t| t.to_string())
            .collect();
        if terms.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(i32, &str)> = Vec::new();
        for s in &self.catalogue {
            if s.name == self.config.search_tool_name || self.is_core(&s.name) {
                continue;
            }
            let name_l = s.name.to_lowercase();
            let desc_l = s.description.to_lowercase();
            let mut score = 0i32;
            for t in &terms {
                if name_l.contains(t.as_str()) {
                    score += 3;
                }
                if desc_l.contains(t.as_str()) {
                    score += 1;
                }
            }
            if score > 0 {
                scored.push((score, s.name.as_str()));
            }
        }
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        scored
            .into_iter()
            .take(self.config.search_top_k)
            .map(|(_, n)| n.to_string())
            .collect()
    }

    /// Handle a `tool_search` dispatch: run the search, load the hits, and return
    /// a readable listing of the now-available tools.
    fn run_search(&self, input: &serde_json::Value) -> ToolOutput {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if query.is_empty() {
            return ToolOutput::error(format!(
                "{}: provide a non-empty `query` string describing the capability you need.",
                self.config.search_tool_name
            ));
        }
        let hits = self.search(query);
        if hits.is_empty() {
            return ToolOutput::success(format!(
                "No tools matched \"{query}\". Try different keywords — the tools you already have \
                 are in your current tool list."
            ));
        }
        self.load_and_list(
            &hits,
            format!("Loaded {} tool(s) for \"{query}\". You can now call them directly:", hits.len()),
        )
    }

    /// Handle a namespace-header dispatch: load the group's member tools (capped at
    /// `search_top_k`) and return a readable listing of them.
    fn open_namespace(&self, name: &str) -> ToolOutput {
        let Some(ns) = self.namespaces.iter().find(|n| n.name == name) else {
            // Unreachable on the dispatch path (membership is checked first), but be
            // defensive rather than panic.
            return ToolOutput::error(format!("namespace `{name}` is not registered"));
        };
        let members: Vec<String> = ns
            .tools
            .iter()
            .take(self.config.search_top_k)
            .cloned()
            .collect();
        if members.is_empty() {
            return ToolOutput::success(format!("Namespace \"{name}\" has no available tools."));
        }
        self.load_and_list(
            &members,
            format!(
                "Opened namespace \"{name}\" — loaded {} tool(s). You can now call them directly:",
                members.len()
            ),
        )
    }

    /// Load `names` into the active set (MRU-first) and render a `header` +
    /// per-tool listing (name, description, parameters) the model can act on.
    fn load_and_list(&self, names: &[String], header: String) -> ToolOutput {
        // Push in reverse so the first-listed ends at the LRU front (last evicted).
        for name in names.iter().rev() {
            self.touch(name);
        }
        let by_name: std::collections::HashMap<&str, &ModelToolSchema> =
            self.catalogue.iter().map(|s| (s.name.as_str(), s)).collect();
        let mut lines = vec![header];
        for name in names {
            if let Some(s) = by_name.get(name.as_str()) {
                lines.push(format!("- {}: {}", s.name, s.description));
                lines.push(format!("  parameters: {}", s.input_schema));
            }
        }
        ToolOutput::success(lines.join("\n"))
    }

    /// Total description bytes the registry advertises right now (core + the search
    /// meta-tool + currently-loaded tools, after eviction). Exposed for budget
    /// calibration and tests.
    pub fn advertised_description_bytes(&self) -> usize {
        self.schemas().iter().map(|s| s.description.len()).sum()
    }
}

impl HarnessToolRegistry for ToolSearchRegistry {
    fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput> {
        if call.name == self.config.search_tool_name {
            let out = self.run_search(&call.input);
            return Box::pin(async move { out });
        }
        // A namespace header opens its group (synthetic; never reaches the inner
        // registry). Checked before delegation — a colliding name was dropped at
        // construction, so this can't shadow a real tool.
        if self.namespaces.iter().any(|n| n.name == call.name) {
            let out = self.open_namespace(&call.name);
            return Box::pin(async move { out });
        }
        // Keep an actually-used tool warm, then delegate to the full catalogue.
        self.touch(&call.name);
        self.inner.dispatch(call)
    }

    fn is_always_halt(&self, tool_name: &str) -> bool {
        if tool_name == self.config.search_tool_name
            || self.namespaces.iter().any(|n| n.name == tool_name)
        {
            return false;
        }
        self.inner.is_always_halt(tool_name)
    }

    fn schemas(&self) -> Vec<ModelToolSchema> {
        let by_name: std::collections::HashMap<&str, &ModelToolSchema> =
            self.catalogue.iter().map(|s| (s.name.as_str(), s)).collect();
        let mut out: Vec<ModelToolSchema> = Vec::new();
        let mut used: usize = 0;
        // Core tools — always advertised.
        for name in &self.core {
            if let Some(s) = by_name.get(name.as_str()) {
                used += s.description.len();
                out.push((*s).clone());
            }
        }
        // Namespace headers — always advertised (the capability-group menu).
        for header in &self.namespace_headers {
            used += header.description.len();
            out.push(header.clone());
        }
        // Search meta-tool — always advertised.
        used += self.search_schema.description.len();
        out.push(self.search_schema.clone());
        // Loaded tools (MRU first), included while they fit; the overflow tail is
        // EVICTED from state (true LRU eviction on read).
        let mut guard = self.loaded.lock().expect("loaded lock poisoned");
        let mut kept: std::collections::VecDeque<String> = std::collections::VecDeque::new();
        for name in guard.iter() {
            if self.is_core(name) {
                continue;
            }
            let Some(s) = by_name.get(name.as_str()) else {
                continue;
            };
            let next = used + s.description.len();
            if next > self.config.description_budget {
                break;
            }
            used = next;
            kept.push_back(name.clone());
            out.push((*s).clone());
        }
        *guard = kept;
        drop(guard);
        // Sort by name for cache-stability when the advertised set is unchanged.
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::mock::*;
    use super::*;
    use serde_json::json;

    fn schema(name: &str, annotations: ToolAnnotations) -> ToolSchema {
        ToolSchema {
            name: name.into(),
            description: format!("{name} tool"),
            parameters: json!({"type": "object", "properties": {}}),
            annotations,
        }
    }

    fn schema_with_required(name: &str, required: &[&str]) -> ToolSchema {
        ToolSchema {
            name: name.into(),
            description: name.into(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": required,
            }),
            annotations: ToolAnnotations {
                read_only: true,
                ..Default::default()
            },
        }
    }

    fn call(name: &str, id: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    // Rule 1: tools dispatched via registry.
    #[tokio::test]
    async fn registered_tool_dispatches_through_registry() {
        let reg = StandardToolRegistry::new();
        let echo = EchoTool::new("echo");
        reg.register(
            Box::new(echo),
            schema(
                "echo",
                ToolAnnotations {
                    read_only: true,
                    ..Default::default()
                },
            ),
        )
        .unwrap();
        let sandbox = AllowAllSandbox;
        let out = reg
            .dispatch(call("echo", "c1", json!({"x": 1})), &sandbox, &test_ctx())
            .await
            .unwrap();
        assert_eq!(out.call_id, "c1");
        match out.output {
            ToolOutput::Success { content, .. } => assert_eq!(content, r#"{"x":1}"#),
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // Rule 3 (issue #81, Q1): duplicate tool registration is a last-wins upsert,
    // NOT an error. The second registration overwrites the first; the active
    // schema reflects the latest registration.
    #[tokio::test]
    async fn duplicate_registration_is_last_wins_upsert() {
        let reg = StandardToolRegistry::new();
        reg.register(
            Box::new(EchoTool::new("echo")),
            ToolSchema {
                name: "echo".into(),
                description: "first".into(),
                parameters: json!({"type": "object", "properties": {}}),
                annotations: ToolAnnotations::default(),
            },
        )
        .unwrap();
        // Re-register the same name with a different description — must succeed
        // (no DuplicateName error) and overwrite.
        reg.register(
            Box::new(EchoTool::new("echo")),
            ToolSchema {
                name: "echo".into(),
                description: "second".into(),
                parameters: json!({"type": "object", "properties": {}}),
                annotations: ToolAnnotations {
                    read_only: true,
                    ..Default::default()
                },
            },
        )
        .expect("re-register must upsert, not error");
        let schemas = reg.active_schemas(None);
        assert_eq!(schemas.len(), 1, "upsert must not duplicate the entry");
        assert_eq!(schemas[0].description, "second", "last registration wins");
        assert!(schemas[0].annotations.read_only);
    }

    // Duplicate tool-SET names still error (DuplicateName retained for sets).
    #[tokio::test]
    async fn duplicate_set_name_still_errors() {
        let reg = StandardToolRegistry::new();
        reg.register_set(ToolSet {
            name: "s".into(),
            tools: vec![],
            phase: None,
        })
        .unwrap();
        let err = reg
            .register_set(ToolSet {
                name: "s".into(),
                tools: vec![],
                phase: None,
            })
            .unwrap_err();
        assert!(matches!(err, RegistrationError::DuplicateName { .. }));
    }

    // Rule 2: schema validated at registration (missing top-level type).
    #[tokio::test]
    async fn invalid_schema_rejected_at_registration() {
        let reg = StandardToolRegistry::new();
        let bad = ToolSchema {
            name: "x".into(),
            description: "x".into(),
            parameters: json!({"properties": {}}),
            annotations: ToolAnnotations::default(),
        };
        let err = reg.register(Box::new(EchoTool::new("x")), bad).unwrap_err();
        assert!(matches!(err, RegistrationError::InvalidSchema { .. }));
    }

    // Rule 4: conflicting annotations rejected.
    #[tokio::test]
    async fn read_only_plus_destructive_rejected() {
        let reg = StandardToolRegistry::new();
        let err = reg
            .register(
                Box::new(EchoTool::new("rm")),
                schema(
                    "rm",
                    ToolAnnotations {
                        read_only: true,
                        destructive: true,
                        ..Default::default()
                    },
                ),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            RegistrationError::ConflictingAnnotations { .. }
        ));
    }

    // Tool name vs schema name mismatch.
    #[tokio::test]
    async fn tool_and_schema_name_must_match() {
        let reg = StandardToolRegistry::new();
        let err = reg
            .register(
                Box::new(EchoTool::new("a")),
                schema("b", ToolAnnotations::default()),
            )
            .unwrap_err();
        assert!(matches!(err, RegistrationError::InvalidSchema { .. }));
    }

    // Rule 6: dispatching unknown tool errors.
    #[tokio::test]
    async fn unregistered_tool_call_errors() {
        let reg = StandardToolRegistry::new();
        let sandbox = AllowAllSandbox;
        let err = reg
            .dispatch(call("missing", "c1", json!({})), &sandbox, &test_ctx())
            .await
            .unwrap_err();
        assert!(matches!(err, DispatchError::UnregisteredTool { .. }));
    }

    // Rule 7: schema validation failure on missing required field.
    #[tokio::test]
    async fn missing_required_field_errors() {
        let reg = StandardToolRegistry::new();
        reg.register(
            Box::new(EchoTool::new("read")),
            schema_with_required("read", &["path"]),
        )
        .unwrap();
        let sandbox = AllowAllSandbox;
        let err = reg
            .dispatch(call("read", "c1", json!({})), &sandbox, &test_ctx())
            .await
            .unwrap_err();
        match err {
            DispatchError::SchemaValidationFailed { tool, reason } => {
                assert_eq!(tool, "read");
                assert!(reason.contains("path"));
            }
            other => panic!("expected SchemaValidationFailed, got {other:?}"),
        }
    }

    // Sandbox violation propagates through dispatch as a DispatchError.
    #[tokio::test]
    async fn sandbox_violation_surfaces_as_dispatch_error() {
        let reg = StandardToolRegistry::new();
        reg.register(
            Box::new(EchoTool::new("echo")),
            schema(
                "echo",
                ToolAnnotations {
                    read_only: true,
                    ..Default::default()
                },
            ),
        )
        .unwrap();
        let sandbox = DenyAllSandbox;
        let err = reg
            .dispatch(call("echo", "c1", json!({})), &sandbox, &test_ctx())
            .await
            .unwrap_err();
        match err {
            DispatchError::SandboxViolation(SandboxViolation::PathEscape { .. }) => {}
            other => panic!("expected SandboxViolation, got {other:?}"),
        }
    }

    // Tool returning a recoverable error wraps cleanly.
    #[tokio::test]
    async fn tool_error_returned_as_tool_output() {
        let reg = StandardToolRegistry::new();
        reg.register(
            Box::new(FailingTool::new("fail")),
            schema("fail", ToolAnnotations::default()),
        )
        .unwrap();
        let sandbox = AllowAllSandbox;
        let out = reg
            .dispatch(call("fail", "c1", json!({})), &sandbox, &test_ctx())
            .await
            .unwrap();
        match out.output {
            ToolOutput::Error {
                message,
                recoverable,
            } => {
                assert_eq!(message, "boom");
                assert!(recoverable);
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    // Rule 8: read_only calls execute concurrently; destructive sequentially.
    // We don't have a clean way to assert wall-clock concurrency without
    // sleeps; instead we assert correctness for both annotation classes and
    // that ordering of results matches the input order.
    #[tokio::test]
    async fn dispatch_all_preserves_input_order() {
        let reg = StandardToolRegistry::new();
        reg.register(
            Box::new(EchoTool::new("r")),
            schema(
                "r",
                ToolAnnotations {
                    read_only: true,
                    ..Default::default()
                },
            ),
        )
        .unwrap();
        reg.register(
            Box::new(EchoTool::new("d")),
            schema(
                "d",
                ToolAnnotations {
                    destructive: true,
                    ..Default::default()
                },
            ),
        )
        .unwrap();
        let sandbox = AllowAllSandbox;
        let calls = vec![
            call("d", "1", json!({"v": "a"})),
            call("r", "2", json!({"v": "b"})),
            call("d", "3", json!({"v": "c"})),
            call("r", "4", json!({"v": "d"})),
        ];
        let results = reg.dispatch_all(calls, &sandbox, &test_ctx()).await;
        let ids: Vec<&str> = results
            .iter()
            .map(|r| r.as_ref().unwrap().call_id.as_str())
            .collect();
        assert_eq!(ids, vec!["1", "2", "3", "4"]);
    }

    // dispatch_all surfaces unregistered tool errors per-slot.
    #[tokio::test]
    async fn dispatch_all_surfaces_individual_errors() {
        let reg = StandardToolRegistry::new();
        reg.register(
            Box::new(EchoTool::new("ok")),
            schema(
                "ok",
                ToolAnnotations {
                    read_only: true,
                    ..Default::default()
                },
            ),
        )
        .unwrap();
        let sandbox = AllowAllSandbox;
        let results = reg
            .dispatch_all(
                vec![call("ok", "1", json!({})), call("missing", "2", json!({}))],
                &sandbox,
                &test_ctx(),
            )
            .await;
        assert!(results[0].is_ok());
        assert!(matches!(
            results[1],
            Err(DispatchError::UnregisteredTool { .. })
        ));
    }

    // Rule 9: has_subagent_tools tracks subagent registration.
    #[tokio::test]
    async fn has_subagent_tools_reflects_registration() {
        let reg = StandardToolRegistry::new();
        assert!(!reg.has_subagent_tools());
        reg.register(
            Box::new(EchoTool::new("echo")),
            schema("echo", ToolAnnotations::default()),
        )
        .unwrap();
        assert!(!reg.has_subagent_tools());
        reg.register(
            Box::new(SubagentMock::new("subagent")),
            schema("subagent", ToolAnnotations::default()),
        )
        .unwrap();
        assert!(reg.has_subagent_tools());
    }

    // Rule 5: active_schemas reflects ToolSet phase filtering, sorted by name.
    #[tokio::test]
    async fn active_schemas_filtered_by_phase_and_sorted() {
        let reg = StandardToolRegistry::new();
        for n in &["zeta", "alpha", "beta"] {
            reg.register(
                Box::new(EchoTool::new(*n)),
                schema(n, ToolAnnotations::default()),
            )
            .unwrap();
        }
        reg.register_set(ToolSet {
            name: "plan".into(),
            tools: vec!["alpha".into(), "zeta".into()],
            phase: Some(TaskPhase::Planning),
        })
        .unwrap();
        reg.register_set(ToolSet {
            name: "always".into(),
            tools: vec!["beta".into()],
            phase: None,
        })
        .unwrap();

        let plan = reg.active_schemas(Some(TaskPhase::Planning));
        let names: Vec<&str> = plan.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta", "zeta"]); // sorted

        // Phase with no matching sets falls back to "always" sets only.
        let exec = reg.active_schemas(Some(TaskPhase::Execution));
        let names: Vec<&str> = exec.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["beta"]);
    }

    // active_schemas with no sets registered returns every schema.
    #[tokio::test]
    async fn active_schemas_no_sets_returns_all() {
        let reg = StandardToolRegistry::new();
        reg.register(
            Box::new(EchoTool::new("a")),
            schema("a", ToolAnnotations::default()),
        )
        .unwrap();
        reg.register(
            Box::new(EchoTool::new("b")),
            schema("b", ToolAnnotations::default()),
        )
        .unwrap();
        let all = reg.active_schemas(None);
        let names: Vec<&str> = all.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
        // With no sets, phase-filtered call falls back to all.
        let any = reg.active_schemas(Some(TaskPhase::Execution));
        assert_eq!(any.len(), 2);
    }

    // ToolSchema → model::ToolSchema projection drops annotations.
    #[test]
    fn to_model_schema_drops_annotations() {
        let s = schema(
            "x",
            ToolAnnotations {
                read_only: true,
                ..Default::default()
            },
        );
        let m = s.to_model_schema();
        assert_eq!(m.name, "x");
        assert_eq!(m.description, "x tool");
    }

    // Serde round-trips for fixture portability.
    #[test]
    fn types_roundtrip_json() {
        let s = schema(
            "x",
            ToolAnnotations {
                read_only: true,
                idempotent: true,
                ..Default::default()
            },
        );
        let j = serde_json::to_string(&s).unwrap();
        let back: ToolSchema = serde_json::from_str(&j).unwrap();
        assert_eq!(s, back);

        let set = ToolSet {
            name: "p".into(),
            tools: vec!["a".into()],
            phase: Some(TaskPhase::Planning),
        };
        let j = serde_json::to_string(&set).unwrap();
        let back: ToolSet = serde_json::from_str(&j).unwrap();
        assert_eq!(set, back);

        let errs = vec![
            RegistrationError::InvalidSchema {
                tool: "x".into(),
                reason: "y".into(),
            },
            RegistrationError::DuplicateName { tool: "x".into() },
            RegistrationError::ConflictingAnnotations {
                tool: "x".into(),
                reason: "y".into(),
            },
        ];
        for e in errs {
            let j = serde_json::to_string(&e).unwrap();
            let back: RegistrationError = serde_json::from_str(&j).unwrap();
            assert_eq!(e, back);
        }
    }

    // Fixture-replay: dispatch scenarios from shared fixtures.
    #[derive(Deserialize)]
    struct DispatchScenario {
        name: String,
        register: Vec<ToolSchema>,
        #[serde(default)]
        sets: Vec<ToolSet>,
        call: FixtureCall,
        expected: ExpectedOutcome,
    }
    #[derive(Deserialize)]
    struct FixtureCall {
        id: String,
        name: String,
        input: serde_json::Value,
    }
    #[derive(Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum ExpectedOutcome {
        Ok {
            call_id: String,
        },
        Err {
            error: String, // matches DispatchError variant name
        },
    }

    #[tokio::test]
    async fn fixture_replay_dispatch_scenarios() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/tool_registry/dispatch_scenarios.json");
        let data = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("could not read fixture {path:?}: {e}");
        });
        let scenarios: Vec<DispatchScenario> = serde_json::from_str(&data).unwrap();
        assert!(!scenarios.is_empty(), "expected ≥1 scenario");
        let sandbox = AllowAllSandbox;

        for sc in scenarios {
            let reg = StandardToolRegistry::new();
            for s in &sc.register {
                // Every fixture tool is an Echo (test only cares about
                // schema-driven dispatch behaviour, not tool semantics).
                reg.register(Box::new(EchoTool::new(s.name.clone())), s.clone())
                    .expect("register fixture tool");
            }
            for set in sc.sets {
                reg.register_set(set).expect("register fixture set");
            }
            let result = reg
                .dispatch(
                    ToolCall {
                        id: sc.call.id.clone(),
                        name: sc.call.name.clone(),
                        input: sc.call.input.clone(),
                    },
                    &sandbox,
                    &test_ctx(),
                )
                .await;
            match (result, sc.expected) {
                (Ok(tr), ExpectedOutcome::Ok { call_id }) => {
                    assert_eq!(tr.call_id, call_id, "scenario {}", sc.name);
                }
                (Err(e), ExpectedOutcome::Err { error }) => {
                    let actual = match &e {
                        DispatchError::UnregisteredTool { .. } => "UnregisteredTool",
                        DispatchError::SchemaValidationFailed { .. } => "SchemaValidationFailed",
                        DispatchError::SandboxViolation(_) => "SandboxViolation",
                        DispatchError::ToolExecutionFailed { .. } => "ToolExecutionFailed",
                    };
                    assert_eq!(actual, error, "scenario {}", sc.name);
                }
                (other, expected) => panic!(
                    "scenario {} mismatch: got {:?}, expected {:?}",
                    sc.name,
                    other.map(|r| r.call_id),
                    serde_json::to_string(&match expected {
                        ExpectedOutcome::Ok { call_id } =>
                            json!({"kind": "ok", "call_id": call_id}),
                        ExpectedOutcome::Err { error } => json!({"kind": "err", "error": error}),
                    })
                    .unwrap(),
                ),
            }
        }
    }
}

#[cfg(test)]
mod tool_search_tests {
    use super::*;
    use serde_json::json;

    /// Minimal inner harness-loop registry: advertises a fixed schema list and
    /// records the names it is asked to dispatch.
    struct FixedInner {
        schemas: Vec<ModelToolSchema>,
        dispatched: std::sync::Mutex<Vec<String>>,
    }

    impl FixedInner {
        fn new(schemas: Vec<ModelToolSchema>) -> Arc<Self> {
            Arc::new(Self {
                schemas,
                dispatched: std::sync::Mutex::new(Vec::new()),
            })
        }
    }

    impl HarnessToolRegistry for FixedInner {
        fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput> {
            self.dispatched
                .lock()
                .unwrap()
                .push(call.name.clone());
            Box::pin(async move { ToolOutput::success(format!("ran {}", call.name)) })
        }
        fn schemas(&self) -> Vec<ModelToolSchema> {
            self.schemas.clone()
        }
    }

    fn tool(name: &str, description: &str) -> ModelToolSchema {
        ModelToolSchema {
            name: name.into(),
            description: description.into(),
            input_schema: json!({"type": "object", "properties": {}}),
        }
    }

    fn tcall(name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: format!("c-{name}"),
            name: name.into(),
            input,
        }
    }

    fn names(schemas: &[ModelToolSchema]) -> Vec<String> {
        schemas.iter().map(|s| s.name.clone()).collect()
    }

    /// catalogue: 3 core + 3 deferred.
    fn fixture() -> ToolSearchRegistry {
        let inner = FixedInner::new(vec![
            tool("read_file", "Read a file's contents from the workspace."),
            tool("write_file", "Create or overwrite a file in the workspace."),
            tool("bash_command", "Run a shell command in the workspace."),
            tool("web_search", "Search the web for information via a backend."),
            tool("task_list", "Author and update the structured task DAG."),
            tool("todo_write", "Persist a lightweight todo checklist."),
        ]);
        ToolSearchRegistry::new(
            inner,
            ToolSearchConfig::new(["read_file", "write_file", "bash_command"])
                .description_budget(100_000)
                .search_top_k(5),
        )
    }

    #[test]
    fn advertises_core_plus_search_tool_only_by_default() {
        let reg = fixture();
        let advertised = names(&reg.schemas());
        // Core tools + the synthetic tool_search, nothing deferred yet.
        assert_eq!(
            advertised,
            vec![
                "bash_command".to_string(),
                "read_file".to_string(),
                "tool_search".to_string(),
                "write_file".to_string(),
            ]
        );
        assert!(!advertised.contains(&"web_search".to_string()));
    }

    #[test]
    fn unknown_core_names_are_ignored() {
        let inner = FixedInner::new(vec![tool("read_file", "Read a file.")]);
        let reg = ToolSearchRegistry::new(
            inner,
            ToolSearchConfig::new(["read_file", "does_not_exist"]),
        );
        let advertised = names(&reg.schemas());
        assert_eq!(advertised, vec!["read_file".to_string(), "tool_search".to_string()]);
    }

    #[tokio::test]
    async fn tool_search_loads_matching_tools_into_active_set() {
        let reg = fixture();
        let out = reg
            .dispatch(tcall("tool_search", json!({"query": "search the web"})))
            .await;
        match out {
            ToolOutput::Success { content, .. } => {
                assert!(content.contains("web_search"), "result lists the match: {content}");
            }
            other => panic!("expected success, got {other:?}"),
        }
        // web_search is now advertised; the unrelated deferred tools are not.
        let advertised = names(&reg.schemas());
        assert!(advertised.contains(&"web_search".to_string()));
        assert!(!advertised.contains(&"todo_write".to_string()));
    }

    #[tokio::test]
    async fn search_matches_tool_name_keyword() {
        let reg = fixture();
        reg.dispatch(tcall("tool_search", json!({"query": "task"})))
            .await;
        let advertised = names(&reg.schemas());
        assert!(advertised.contains(&"task_list".to_string()));
    }

    #[tokio::test]
    async fn empty_query_is_a_recoverable_error() {
        let reg = fixture();
        let out = reg
            .dispatch(tcall("tool_search", json!({"query": "   "})))
            .await;
        assert!(matches!(out, ToolOutput::Error { recoverable: true, .. }));
    }

    #[tokio::test]
    async fn no_matches_reports_cleanly_without_loading() {
        let reg = fixture();
        let out = reg
            .dispatch(tcall("tool_search", json!({"query": "quantum teleportation"})))
            .await;
        assert!(matches!(out, ToolOutput::Success { .. }));
        // Nothing new loaded.
        let advertised = names(&reg.schemas());
        assert_eq!(advertised.len(), 4); // 3 core + tool_search
    }

    #[tokio::test]
    async fn dispatch_delegates_real_tools_to_inner() {
        let inner = FixedInner::new(vec![
            tool("read_file", "Read a file."),
            tool("web_search", "Search the web."),
        ]);
        let reg = ToolSearchRegistry::new(inner.clone(), ToolSearchConfig::new(["read_file"]));
        // A core tool dispatches through.
        let out = reg.dispatch(tcall("read_file", json!({"path": "x"}))).await;
        assert!(matches!(out, ToolOutput::Success { .. }));
        // A deferred, never-searched tool STILL dispatches (exposure is a hint,
        // not a hard gate) and gets kept warm afterwards.
        let out = reg.dispatch(tcall("web_search", json!({"query": "x"}))).await;
        assert!(matches!(out, ToolOutput::Success { .. }));
        assert_eq!(*inner.dispatched.lock().unwrap(), vec!["read_file", "web_search"]);
        // web_search is now advertised (it was touched on dispatch).
        assert!(names(&reg.schemas()).contains(&"web_search".to_string()));
    }

    #[tokio::test]
    async fn description_budget_lru_evicts_oldest_loaded() {
        // Each deferred description is 10 bytes ("dddddddddd"). Core is empty, the
        // search tool's description is large, so set a budget that fits the search
        // tool + exactly ONE deferred tool.
        let inner = FixedInner::new(vec![
            tool("alpha", "dddddddddd"),
            tool("bravo", "dddddddddd"),
        ]);
        let search_desc_len = ToolSearchRegistry::build_search_schema("tool_search")
            .description
            .len();
        let reg = ToolSearchRegistry::new(
            inner,
            ToolSearchConfig::new(Vec::<String>::new())
                .description_budget(search_desc_len + 10)
                .search_top_k(5),
        );
        reg.dispatch(tcall("tool_search", json!({"query": "alpha"}))).await;
        reg.dispatch(tcall("tool_search", json!({"query": "bravo"}))).await;
        // Only one deferred tool fits; bravo was loaded most recently, so alpha is
        // evicted.
        let advertised = names(&reg.schemas());
        assert!(advertised.contains(&"bravo".to_string()), "MRU kept: {advertised:?}");
        assert!(!advertised.contains(&"alpha".to_string()), "LRU evicted: {advertised:?}");
        assert!(reg.advertised_description_bytes() <= search_desc_len + 10);
    }

    /// Calibration: print the description-prose volume of looper's Default ceiling
    /// (coding_set − memory = the 13-tool cliff) vs the 9-tool core, and the
    /// ToolSearch-bounded advertised set. Run with `--nocapture` to read the
    /// numbers; asserts the registry materially shrinks the advertised prose.
    #[test]
    fn calibration_default_ceiling_description_volume() {
        use crate::tools::StandardTools;

        let coding_minus_memory: Vec<ModelToolSchema> = StandardTools::coding_set()
            .into_iter()
            .filter(|t| t.schema.name != "memory")
            .map(|t| t.schema.to_model_schema())
            .collect();
        let full_bytes: usize = coding_minus_memory.iter().map(|s| s.description.len()).sum();

        let core = [
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "grep",
            "grep_files",
            "find_files",
            "bash_command",
            "send_message",
        ];
        let core_bytes: usize = coding_minus_memory
            .iter()
            .filter(|s| core.contains(&s.name.as_str()))
            .map(|s| s.description.len())
            .sum();

        let inner = FixedInner::new(coding_minus_memory.clone());
        let reg = ToolSearchRegistry::new(
            inner,
            ToolSearchConfig::new(core).description_budget(100_000),
        );
        let advertised_bytes = reg.advertised_description_bytes();
        let search_bytes = ToolSearchRegistry::build_search_schema("tool_search")
            .description
            .len();

        println!(
            "[tool-search calibration] tools={} full_desc_bytes={} core9_desc_bytes={} \
             search_tool_desc_bytes={} advertised(core9+search)_bytes={}",
            coding_minus_memory.len(),
            full_bytes,
            core_bytes,
            search_bytes,
            advertised_bytes,
        );

        // The bounded advertised set carries materially less description prose
        // than the full 13-tool ceiling that trips the native-tool-calling cliff.
        assert!(
            advertised_bytes < full_bytes,
            "advertised {advertised_bytes} should be < full ceiling {full_bytes}"
        );
        // Core (9) + the search tool only — deferred web/task/todo tools are hidden.
        assert_eq!(reg.schemas().len(), core.len() + 1);
    }

    // ---- namespaces --------------------------------------------------------

    /// catalogue: 1 core (read_file) + a `web` namespace (web_fetch, web_search) +
    /// a standalone deferred tool (task_list).
    fn ns_fixture() -> ToolSearchRegistry {
        let inner = FixedInner::new(vec![
            tool("read_file", "Read a file's contents from the workspace."),
            tool("web_fetch", "Fetch the contents of a URL over HTTP."),
            tool("web_search", "Search the web for information via a backend."),
            tool("task_list", "Author and update the structured task DAG."),
        ]);
        ToolSearchRegistry::new(
            inner,
            ToolSearchConfig::new(["read_file"])
                .namespace(ToolNamespace::new(
                    "web",
                    "Web access tools (fetch a URL, search the web)",
                    ["web_fetch", "web_search"],
                ))
                .description_budget(100_000)
                .search_top_k(5),
        )
    }

    #[test]
    fn namespace_header_is_advertised_but_members_are_hidden() {
        let reg = ns_fixture();
        let advertised = names(&reg.schemas());
        // Header + core + search tool; standalone deferred task_list hidden.
        assert!(advertised.contains(&"web".to_string()), "header advertised: {advertised:?}");
        assert!(advertised.contains(&"read_file".to_string()));
        assert!(advertised.contains(&"tool_search".to_string()));
        // Members and the standalone deferred tool are NOT advertised yet.
        assert!(!advertised.contains(&"web_fetch".to_string()));
        assert!(!advertised.contains(&"web_search".to_string()));
        assert!(!advertised.contains(&"task_list".to_string()));
        // Header carries the namespace description + the open hint.
        let header = reg.schemas().into_iter().find(|s| s.name == "web").unwrap();
        assert!(header.description.contains("Web access tools"));
        assert!(header.description.contains("call to load"));
    }

    #[tokio::test]
    async fn opening_a_namespace_loads_its_members() {
        let reg = ns_fixture();
        let out = reg.dispatch(tcall("web", json!({}))).await;
        match out {
            ToolOutput::Success { content, .. } => {
                assert!(content.contains("web_fetch"), "lists members: {content}");
                assert!(content.contains("web_search"), "lists members: {content}");
            }
            other => panic!("expected success, got {other:?}"),
        }
        // Both members are now advertised alongside the header.
        let advertised = names(&reg.schemas());
        assert!(advertised.contains(&"web_fetch".to_string()));
        assert!(advertised.contains(&"web_search".to_string()));
        assert!(advertised.contains(&"web".to_string()), "header stays");
    }

    #[tokio::test]
    async fn opening_a_namespace_does_not_reach_the_inner_registry() {
        let inner = FixedInner::new(vec![
            tool("read_file", "Read a file."),
            tool("web_fetch", "Fetch a URL."),
        ]);
        let reg = ToolSearchRegistry::new(
            inner.clone(),
            ToolSearchConfig::new(["read_file"]).namespace(ToolNamespace::new(
                "web",
                "Web tools",
                ["web_fetch"],
            )),
        );
        reg.dispatch(tcall("web", json!({}))).await;
        // The synthetic header never dispatches to the catalogue.
        assert!(inner.dispatched.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn namespace_members_are_still_findable_via_search() {
        let reg = ns_fixture();
        reg.dispatch(tcall("tool_search", json!({"query": "fetch url web"}))).await;
        let advertised = names(&reg.schemas());
        assert!(advertised.contains(&"web_fetch".to_string()), "search finds namespaced tool");
    }

    #[tokio::test]
    async fn a_namespace_colliding_with_a_tool_name_is_dropped() {
        // A namespace named after a real catalogue tool must NOT shadow it.
        let inner = FixedInner::new(vec![
            tool("read_file", "Read a file."),
            tool("web_fetch", "Fetch a URL."),
        ]);
        let reg = ToolSearchRegistry::new(
            inner.clone(),
            ToolSearchConfig::new(Vec::<String>::new()).namespace(ToolNamespace::new(
                "read_file", // collides with the real tool
                "bogus",
                ["web_fetch"],
            )),
        );
        // No synthetic header was created; calling read_file dispatches the REAL tool.
        let out = reg.dispatch(tcall("read_file", json!({"path": "x"}))).await;
        assert!(matches!(out, ToolOutput::Success { .. }));
        assert_eq!(*inner.dispatched.lock().unwrap(), vec!["read_file"]);
    }

    #[test]
    fn namespace_with_no_valid_members_is_dropped() {
        let inner = FixedInner::new(vec![tool("read_file", "Read a file.")]);
        let reg = ToolSearchRegistry::new(
            inner,
            ToolSearchConfig::new(["read_file"]).namespace(ToolNamespace::new(
                "empty",
                "nothing here",
                ["does_not_exist", "read_file"], // missing + core ⇒ no valid members
            )),
        );
        let advertised = names(&reg.schemas());
        assert!(!advertised.contains(&"empty".to_string()), "empty namespace dropped: {advertised:?}");
        assert_eq!(advertised, vec!["read_file".to_string(), "tool_search".to_string()]);
    }
}
