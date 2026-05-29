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
//! dispatch argument; the harness-side `RealToolRegistry` bridge (scenarios.rs)
//! is constructed per-run with the `SessionId` + `Arc<dyn RunStore>` and builds
//! the `ToolContext` itself before forwarding. The harness-loop
//! `ToolRegistry::dispatch(call)` signature is UNCHANGED. `ToolContext` is a
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
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

use crate::harness::{
    BoxFut, SandboxProvider, SandboxViolation, SessionId, ToolOutput, ToolResult,
};
use crate::model::ToolCall;
use crate::storage::{MemoryStore, RunStore};

// ============================================================================
// ToolContext — the storage seam handed to every tool (#75)
// ============================================================================

/// The per-dispatch storage seam handed to every [`Tool::execute`] call,
/// alongside (but separate from) the [`SandboxProvider`]. It carries the
/// minimum a tool needs to persist durable state via the storage layer:
///
///   - `session_id`   — the run's [`SessionId`], the key namespace for stores.
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
    run_store: Arc<dyn RunStore>,
    memory_store: Arc<dyn MemoryStore>,
}

impl ToolContext {
    /// Build a context from the run's session id and the storage seams.
    pub fn new(
        session_id: SessionId,
        run_store: Arc<dyn RunStore>,
        memory_store: Arc<dyn MemoryStore>,
    ) -> Self {
        Self {
            session_id,
            run_store,
            memory_store,
        }
    }

    /// The session id keying this run's persisted state.
    pub fn session_id(&self) -> &SessionId {
        &self.session_id
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

/// Default in-memory registry. Concurrency-safe: register/lookup go through
/// an async `RwLock`. The lock is held briefly (lookup + clone of `Arc`); the
/// tool itself executes lock-free.
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

    async fn lookup(&self, name: &str) -> Option<(Arc<dyn Tool>, ToolSchema)> {
        let guard = self.tools.read().await;
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
        let mut guard = self
            .tools
            .try_write()
            .expect("tool registration must not contend with dispatch");
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
        let mut guard = self
            .sets
            .try_write()
            .expect("set registration must not contend with dispatch");
        if guard.iter().any(|s| s.name == set.name) {
            return Err(RegistrationError::DuplicateName {
                tool: set.name.clone(),
            });
        }
        guard.push(set);
        Ok(())
    }

    fn active_schemas(&self, phase: Option<TaskPhase>) -> Vec<ToolSchema> {
        let tools = self
            .tools
            .try_read()
            .expect("active_schemas should not contend with dispatch in tests");
        let sets = self
            .sets
            .try_read()
            .expect("active_schemas should not contend with dispatch in tests");

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
            let (tool, schema) = match self.lookup(&call.name).await {
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
                let concurrent = match self.lookup(&call.name).await {
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
        let guard = self
            .tools
            .try_read()
            .expect("has_subagent_tools should not contend with dispatch in tests");
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
        use crate::storage::InMemoryStorageProvider;
        // One in-memory backend serves both the run and (scope-aware) memory
        // seams in tests.
        let backend = Arc::new(InMemoryStorageProvider::new());
        ToolContext::new(SessionId::new("test-session"), backend.clone(), backend)
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
