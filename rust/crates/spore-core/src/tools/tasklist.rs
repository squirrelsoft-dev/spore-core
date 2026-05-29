//! TaskList tool (#71, storage seam #75): the single mutating tool over the
//! persisted task list.
//!
//! One unit-struct tool, [`TaskListTool`] (`NAME = "task_list"`), dispatched on
//! an `action` discriminator (`add_task`, `update_task`, `complete_task`,
//! `list_tasks`). See [`crate::tasklist`] for the types and the transition
//! matrix.
//!
//! ## Storage seam (#75)
//!
//! The tool persists via the [`ToolContext`]'s [`RunStore`](crate::storage::RunStore)
//! — NOT the sandbox filesystem. It is read-modify-write keyed by the run's
//! [`SessionId`](crate::harness::SessionId) under
//! [`TASK_LIST_EXTRAS_KEY`](crate::tasklist::TASK_LIST_EXTRAS_KEY) (`"task_list"`):
//! 1. parse params (bad input → recoverable error),
//! 2. `ctx.run_store().get(session_id, "task_list")` (absent → [`TaskList::default`]),
//! 3. apply the action (domain errors → recoverable),
//! 4. on a mutating action, `ctx.run_store().put(session_id, "task_list", value)`,
//! 5. return the serialized current list as success content.
//!
//! ### Shared key
//! This standalone tool and the harness-side PlanExecute execute loop persist
//! under the SAME `RunStore` key (`"task_list"`), keyed by `SessionId`. A
//! standalone tool call and a PlanExecute run on the same session intentionally
//! share one blob. The JSON shape is the canonical `serde_json` form of
//! [`TaskList`](crate::tasklist::TaskList) (`{"tasks":[...],"next_id":N}`).
//!
//! ### Behavior change vs the retired sandbox path
//! Previously the tool persisted to `.spore/task_list.json` via the sandbox.
//! That path is GONE. With the library's default storage (`StorageProvider::no_op()`)
//! a standalone tool call persists NOTHING across processes — the no-op run
//! store silently discards writes and returns `None` on read. This is an
//! accepted behavior change: durable cross-process persistence now requires
//! configuring a real `StorageProvider` (e.g. `FileSystemStorageProvider`).
//! There is NO migration shim for old on-disk `.spore/task_list.json` files.
//!
//! ## Storage-error mapping
//! [`StorageError`](crate::storage::StorageError) from a get/put maps to a
//! recoverable [`ToolOutput::Error`]. A present-but-malformed blob (parse
//! failure) is likewise recoverable. `list_tasks` never writes.
//!
//! CRITICAL: this tool is NOT annotated `read_only`. `read_only` tools are run
//! CONCURRENTLY by `dispatch_all`, and a concurrent read-modify-write over the
//! same key would race. Leaving `read_only` false makes the registry dispatch
//! it sequentially. `destructive`/`open_world` are also left false so it is not
//! treated as an irreversible side effect.

use serde_json::json;

use crate::harness::{BoxFut, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tasklist::{TaskList, TASK_LIST_EXTRAS_KEY};
use crate::tool_registry::{Tool, ToolAnnotations, ToolContext, ToolSchema};
use crate::tools::params::{parse_params, TaskListParams};

pub struct TaskListTool;

impl TaskListTool {
    pub const NAME: &'static str = "task_list";

    pub fn new() -> Self {
        Self
    }

    pub fn schema() -> ToolSchema {
        // Fields kept sorted/stable for cache stability (see registry spec):
        // `action` (required) plus the union of per-action fields.
        ToolSchema {
            name: Self::NAME.into(),
            description: "Manage the persisted task list: add, update, complete, or list tasks"
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add_task", "complete_task", "list_tasks", "update_task"],
                    },
                    "description": {"type": "string"},
                    "id": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "enum": ["blocked", "completed", "in_progress", "pending"],
                    },
                },
                "required": ["action"],
            }),
            // Intentionally NOT read_only: this tool mutates shared on-disk
            // state and must dispatch sequentially. See module docs.
            annotations: ToolAnnotations::default(),
        }
    }
}

impl Default for TaskListTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for TaskListTool {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        ctx: &'a ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let session_id = ctx.session_id();
            let run_store = ctx.run_store();

            // 1. Parse params (bad input → recoverable).
            let params: TaskListParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };

            // 2. Load current list from the run store (absent → default).
            //    A storage error or a malformed blob is recoverable.
            let mut list = match run_store.get(session_id, TASK_LIST_EXTRAS_KEY).await {
                Ok(None) => TaskList::default(),
                Ok(Some(value)) => match serde_json::from_value::<TaskList>(value) {
                    Ok(l) => l,
                    Err(e) => {
                        return ToolOutput::Error {
                            message: format!("could not parse task list: {e}"),
                            recoverable: true,
                        };
                    }
                },
                Err(e) => {
                    return ToolOutput::Error {
                        message: format!("could not load task list: {e}"),
                        recoverable: true,
                    };
                }
            };

            // 3. Apply the action. Domain errors → recoverable. `list_tasks`
            //    does not mutate.
            let mutated = match params {
                TaskListParams::AddTask { description } => {
                    list.add(description);
                    true
                }
                TaskListParams::UpdateTask {
                    id,
                    status,
                    description,
                } => {
                    if let Err(e) = list.update(id, status, description) {
                        return ToolOutput::Error {
                            message: e.to_string(),
                            recoverable: true,
                        };
                    }
                    true
                }
                TaskListParams::CompleteTask { id } => {
                    if let Err(e) = list.complete(id) {
                        return ToolOutput::Error {
                            message: e.to_string(),
                            recoverable: true,
                        };
                    }
                    true
                }
                TaskListParams::ListTasks {} => false,
            };

            // 4. Persist the (possibly mutated) list to the run store, keyed by
            //    SessionId under the shared TASK_LIST_EXTRAS_KEY. We always
            //    persist on a mutating action; list_tasks skips the write.
            if mutated {
                let value = match serde_json::to_value(&list) {
                    Ok(v) => v,
                    Err(e) => {
                        return ToolOutput::Error {
                            message: format!("could not serialize task list: {e}"),
                            recoverable: true,
                        };
                    }
                };
                if let Err(e) = run_store.put(session_id, TASK_LIST_EXTRAS_KEY, value).await {
                    return ToolOutput::Error {
                        message: format!("could not persist task list: {e}"),
                        recoverable: true,
                    };
                }
            }

            // 5. Return the serialized current list.
            match serde_json::to_string(&list) {
                Ok(content) => ToolOutput::Success {
                    content,
                    truncated: false,
                },
                Err(e) => ToolOutput::Error {
                    message: format!("could not serialize task list: {e}"),
                    recoverable: true,
                },
            }
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{Operation, SandboxProvider, SandboxViolation, SessionId};
    use crate::storage::{InMemoryStorageProvider, NoOpStorageProvider, RunStore, StorageError};
    use crate::tasklist::{TaskList, TaskStatus};
    use std::path::PathBuf;
    use std::sync::Arc;

    /// Permissive sandbox — the tool no longer touches the filesystem, so the
    /// sandbox is irrelevant to persistence. Used to prove storage works even
    /// when the sandbox would deny every path.
    struct AllowAllSandbox;
    impl SandboxProvider for AllowAllSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async move { Ok(()) })
        }
    }

    /// Sandbox whose `resolve_path` denies everything. Proves the tool persists
    /// to the RunStore, not the sandbox: `add_task` still succeeds even though
    /// the sandbox would reject any filesystem path.
    struct DenyPathSandbox;
    impl SandboxProvider for DenyPathSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async move { Ok(()) })
        }
        fn resolve_path<'a>(
            &'a self,
            path: &'a str,
            _op: Operation,
        ) -> BoxFut<'a, Result<PathBuf, SandboxViolation>> {
            Box::pin(async move {
                Err(SandboxViolation::PathEscape {
                    path: path.to_string(),
                })
            })
        }
    }

    /// A RunStore that always fails, to prove storage errors map to a
    /// recoverable tool error.
    struct FailingRunStore;
    impl RunStore for FailingRunStore {
        fn get<'a>(
            &'a self,
            _session_id: &'a SessionId,
            _key: &'a str,
        ) -> BoxFut<'a, Result<Option<serde_json::Value>, StorageError>> {
            Box::pin(async move {
                Err(StorageError::Backend {
                    message: "boom".into(),
                })
            })
        }
        fn put<'a>(
            &'a self,
            _session_id: &'a SessionId,
            _key: &'a str,
            _value: serde_json::Value,
        ) -> BoxFut<'a, Result<(), StorageError>> {
            Box::pin(async move {
                Err(StorageError::Backend {
                    message: "boom".into(),
                })
            })
        }
        fn delete<'a>(
            &'a self,
            _session_id: &'a SessionId,
            _key: &'a str,
        ) -> BoxFut<'a, Result<(), StorageError>> {
            Box::pin(async move { Ok(()) })
        }
        fn list_keys<'a>(
            &'a self,
            _session_id: &'a SessionId,
        ) -> BoxFut<'a, Result<Vec<String>, StorageError>> {
            Box::pin(async move { Ok(Vec::new()) })
        }
    }

    /// A RunStore whose stored blob for the task_list key is malformed JSON for
    /// a `TaskList`, to prove a parse failure is recoverable.
    struct CorruptRunStore;
    impl RunStore for CorruptRunStore {
        fn get<'a>(
            &'a self,
            _session_id: &'a SessionId,
            _key: &'a str,
        ) -> BoxFut<'a, Result<Option<serde_json::Value>, StorageError>> {
            Box::pin(async move { Ok(Some(json!({"not": "a task list"}))) })
        }
        fn put<'a>(
            &'a self,
            _session_id: &'a SessionId,
            _key: &'a str,
            _value: serde_json::Value,
        ) -> BoxFut<'a, Result<(), StorageError>> {
            Box::pin(async move { Ok(()) })
        }
        fn delete<'a>(
            &'a self,
            _session_id: &'a SessionId,
            _key: &'a str,
        ) -> BoxFut<'a, Result<(), StorageError>> {
            Box::pin(async move { Ok(()) })
        }
        fn list_keys<'a>(
            &'a self,
            _session_id: &'a SessionId,
        ) -> BoxFut<'a, Result<Vec<String>, StorageError>> {
            Box::pin(async move { Ok(Vec::new()) })
        }
    }

    fn ctx_with(run_store: Arc<dyn RunStore>, session: &str) -> ToolContext {
        // tasklist tests exercise the run store only; memory is a no-op seam.
        ToolContext::new(
            SessionId::new(session),
            run_store,
            Arc::new(NoOpStorageProvider),
        )
    }

    /// A ToolContext over a fresh in-memory run store with a default session id.
    fn in_memory_ctx() -> ToolContext {
        ctx_with(Arc::new(InMemoryStorageProvider::new()), "test-session")
    }

    fn call(input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: TaskListTool::NAME.into(),
            input,
        }
    }

    fn parse_list(out: &ToolOutput) -> TaskList {
        match out {
            ToolOutput::Success { content, .. } => serde_json::from_str(content).unwrap(),
            other => panic!("expected Success, got {other:?}"),
        }
    }

    /// Read the persisted blob straight off a RunStore as a `TaskList`.
    async fn load_from_store(run_store: &Arc<dyn RunStore>, session: &str) -> Option<TaskList> {
        run_store
            .get(&SessionId::new(session), TASK_LIST_EXTRAS_KEY)
            .await
            .unwrap()
            .map(|v| serde_json::from_value(v).unwrap())
    }

    #[tokio::test]
    async fn add_then_list_persists_and_assigns_ids() {
        let ctx = in_memory_ctx();
        let sb = AllowAllSandbox;
        let tool = TaskListTool::new();

        let r1 = tool
            .execute(
                &call(json!({"action": "add_task", "description": "a"})),
                &sb,
                &ctx,
            )
            .await;
        let l1 = parse_list(&r1);
        assert_eq!(l1.tasks.len(), 1);
        assert_eq!(l1.tasks[0].id, 1);
        assert_eq!(l1.next_id, 2);

        let r2 = tool
            .execute(
                &call(json!({"action": "add_task", "description": "b"})),
                &sb,
                &ctx,
            )
            .await;
        let l2 = parse_list(&r2);
        assert_eq!(l2.tasks.iter().map(|t| t.id).collect::<Vec<_>>(), [1, 2]);

        // The blob actually exists in the run store under the shared key.
        let persisted = load_from_store(ctx.run_store(), "test-session")
            .await
            .expect("task_list blob present");
        assert_eq!(persisted, l2);

        // list_tasks returns the same list and does not mutate.
        let r3 = tool
            .execute(&call(json!({"action": "list_tasks"})), &sb, &ctx)
            .await;
        let l3 = parse_list(&r3);
        assert_eq!(l3, l2);
    }

    // Storage seam: persists to the RunStore, NOT the sandbox. Even with a
    // sandbox that denies every path, add_task succeeds and persists.
    #[tokio::test]
    async fn persists_to_run_store_not_sandbox() {
        let ctx = in_memory_ctx();
        let sb = DenyPathSandbox;
        let tool = TaskListTool::new();

        let r = tool
            .execute(
                &call(json!({"action": "add_task", "description": "via run store"})),
                &sb,
                &ctx,
            )
            .await;
        let list = parse_list(&r);
        assert_eq!(list.tasks.len(), 1);
        let persisted = load_from_store(ctx.run_store(), "test-session")
            .await
            .expect("persisted despite sandbox path denial");
        assert_eq!(persisted, list);
    }

    // Keyed by SessionId: two sessions over the SAME run store keep separate
    // lists.
    #[tokio::test]
    async fn lists_are_keyed_by_session_id() {
        let run_store: Arc<dyn RunStore> = Arc::new(InMemoryStorageProvider::new());
        let sb = AllowAllSandbox;
        let tool = TaskListTool::new();

        let ctx_a = ctx_with(run_store.clone(), "session-a");
        let ctx_b = ctx_with(run_store.clone(), "session-b");

        tool.execute(
            &call(json!({"action": "add_task", "description": "a1"})),
            &sb,
            &ctx_a,
        )
        .await;
        tool.execute(
            &call(json!({"action": "add_task", "description": "b1"})),
            &sb,
            &ctx_b,
        )
        .await;
        tool.execute(
            &call(json!({"action": "add_task", "description": "b2"})),
            &sb,
            &ctx_b,
        )
        .await;

        let a = load_from_store(&run_store, "session-a").await.unwrap();
        let b = load_from_store(&run_store, "session-b").await.unwrap();
        assert_eq!(a.tasks.len(), 1);
        assert_eq!(a.tasks[0].description, "a1");
        assert_eq!(b.tasks.len(), 2);
        assert_eq!(
            b.tasks
                .iter()
                .map(|t| t.description.as_str())
                .collect::<Vec<_>>(),
            ["b1", "b2"]
        );
    }

    // Persist then reload with a FRESH tool over the SAME ctx yields the
    // identical list.
    #[tokio::test]
    async fn persist_then_reload_yields_identical_list() {
        let ctx = in_memory_ctx();
        let sb = AllowAllSandbox;

        let tool1 = TaskListTool::new();
        tool1
            .execute(
                &call(json!({"action": "add_task", "description": "one"})),
                &sb,
                &ctx,
            )
            .await;
        let r = tool1
            .execute(
                &call(json!({"action": "add_task", "description": "two"})),
                &sb,
                &ctx,
            )
            .await;
        let from_tool = parse_list(&r);

        // Fresh tool instance, same ctx: list_tasks reads back identical state.
        let tool2 = TaskListTool::new();
        let reloaded = tool2
            .execute(&call(json!({"action": "list_tasks"})), &sb, &ctx)
            .await;
        assert_eq!(parse_list(&reloaded), from_tool);
    }

    #[tokio::test]
    async fn update_status_and_complete() {
        let ctx = in_memory_ctx();
        let sb = AllowAllSandbox;
        let tool = TaskListTool::new();
        tool.execute(
            &call(json!({"action": "add_task", "description": "x"})),
            &sb,
            &ctx,
        )
        .await;

        let r = tool
            .execute(
                &call(json!({"action": "update_task", "id": 1, "status": "in_progress"})),
                &sb,
                &ctx,
            )
            .await;
        assert_eq!(parse_list(&r).tasks[0].status, TaskStatus::InProgress);

        let r = tool
            .execute(
                &call(json!({"action": "complete_task", "id": 1})),
                &sb,
                &ctx,
            )
            .await;
        assert_eq!(parse_list(&r).tasks[0].status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn unknown_id_is_recoverable_error() {
        let ctx = in_memory_ctx();
        let r = TaskListTool::new()
            .execute(
                &call(json!({"action": "complete_task", "id": 42})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn invalid_transition_out_of_completed_is_recoverable() {
        let ctx = in_memory_ctx();
        let sb = AllowAllSandbox;
        let tool = TaskListTool::new();
        tool.execute(
            &call(json!({"action": "add_task", "description": "x"})),
            &sb,
            &ctx,
        )
        .await;
        tool.execute(
            &call(json!({"action": "complete_task", "id": 1})),
            &sb,
            &ctx,
        )
        .await;
        let r = tool
            .execute(
                &call(json!({"action": "update_task", "id": 1, "status": "pending"})),
                &sb,
                &ctx,
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn bad_params_is_recoverable_error() {
        let ctx = in_memory_ctx();
        // Unknown action.
        let r = TaskListTool::new()
            .execute(&call(json!({"action": "nope"})), &AllowAllSandbox, &ctx)
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    // Storage failure (get/put) → recoverable error.
    #[tokio::test]
    async fn storage_failure_is_recoverable_error() {
        let ctx = ctx_with(Arc::new(FailingRunStore), "test-session");
        let r = TaskListTool::new()
            .execute(
                &call(json!({"action": "add_task", "description": "x"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("expected recoverable error, got {other:?}"),
        }
    }

    // Malformed persisted blob → recoverable parse error.
    #[tokio::test]
    async fn corrupt_blob_is_recoverable_error() {
        let ctx = ctx_with(Arc::new(CorruptRunStore), "test-session");
        let r = TaskListTool::new()
            .execute(
                &call(json!({"action": "list_tasks"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("expected recoverable error, got {other:?}"),
        }
    }

    // list_tasks does not write: a fresh ctx with a never-written store stays
    // empty after a list_tasks call.
    #[tokio::test]
    async fn list_tasks_does_not_write() {
        let ctx = in_memory_ctx();
        let tool = TaskListTool::new();
        let r = tool
            .execute(
                &call(json!({"action": "list_tasks"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        // Returns the empty default list.
        assert_eq!(parse_list(&r), TaskList::default());
        // Nothing was persisted (list_tasks must not write).
        assert!(load_from_store(ctx.run_store(), "test-session")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn schema_is_not_read_only() {
        let s = TaskListTool::schema();
        assert!(!s.annotations.read_only);
        assert!(!s.annotations.destructive);
        assert!(!s.annotations.open_world);
    }

    // ========================================================================
    // Fixture replay (now driven over an in-memory RunStore, not the sandbox)
    // ========================================================================

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/tasklist")
            .join(name)
    }

    #[derive(serde::Deserialize)]
    struct OpStep {
        action: serde_json::Value,
        expected: OpExpected,
    }
    #[derive(serde::Deserialize)]
    struct OpExpected {
        ok: bool,
        #[serde(default)]
        list: Option<TaskList>,
        #[serde(default)]
        error: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct OpScenario {
        name: String,
        steps: Vec<OpStep>,
    }

    // Replay each operations scenario step-by-step against a read-modify-write
    // over a fresh in-memory RunStore, asserting the resulting list (or error
    // kind) per step. Must replay byte-identically to the retired sandbox path.
    #[tokio::test]
    async fn fixture_replay_operations() {
        let data = std::fs::read_to_string(fixture_path("operations.json")).unwrap();
        let scenarios: Vec<OpScenario> = serde_json::from_str(&data).unwrap();
        assert!(!scenarios.is_empty(), "expected >=1 scenario");
        let tool = TaskListTool::new();
        let sb = AllowAllSandbox;

        for sc in scenarios {
            // Fresh isolated run store per scenario.
            let ctx = in_memory_ctx();
            for (i, step) in sc.steps.iter().enumerate() {
                let out = tool.execute(&call(step.action.clone()), &sb, &ctx).await;
                match (&out, step.expected.ok) {
                    (ToolOutput::Success { content, .. }, true) => {
                        let got: TaskList = serde_json::from_str(content).unwrap();
                        let want = step.expected.list.as_ref().unwrap_or_else(|| {
                            panic!("{} step {i}: ok step missing `list`", sc.name)
                        });
                        assert_eq!(&got, want, "{} step {i}", sc.name);
                    }
                    (
                        ToolOutput::Error {
                            message,
                            recoverable,
                        },
                        false,
                    ) => {
                        assert!(
                            recoverable,
                            "{} step {i}: errors must be recoverable",
                            sc.name
                        );
                        let want = step.expected.error.as_deref().unwrap();
                        let kind = if message.contains("not found") {
                            "task_not_found"
                        } else if message.contains("invalid transition") {
                            "invalid_transition"
                        } else {
                            "other"
                        };
                        assert_eq!(kind, want, "{} step {i}: {message}", sc.name);
                    }
                    (other, expected_ok) => {
                        panic!("{} step {i}: ok={expected_ok} but got {other:?}", sc.name)
                    }
                }
            }
        }
    }

    #[derive(serde::Deserialize)]
    struct TransitionCase {
        from: TaskStatus,
        to: TaskStatus,
        expected: String,
    }

    // Replay the full transition matrix against `validate_transition`.
    #[tokio::test]
    async fn fixture_replay_transitions() {
        use crate::tasklist::validate_transition;
        let data = std::fs::read_to_string(fixture_path("transitions.json")).unwrap();
        let cases: Vec<TransitionCase> = serde_json::from_str(&data).unwrap();
        assert!(!cases.is_empty(), "expected >=1 case");
        for c in cases {
            let got = match validate_transition(1, c.from, c.to) {
                Ok(()) => "ok",
                Err(_) => "invalid_transition",
            };
            assert_eq!(got, c.expected, "{:?} -> {:?}", c.from, c.to);
        }
    }

    #[derive(serde::Deserialize)]
    struct SerCase {
        name: String,
        list: TaskList,
        json: String,
    }

    // Replay canonical serialization blobs: serialize(list) must equal the
    // pinned JSON, and parse(json) must equal the list (byte-identity).
    #[tokio::test]
    async fn fixture_replay_serialization() {
        let data = std::fs::read_to_string(fixture_path("serialization.json")).unwrap();
        let cases: Vec<SerCase> = serde_json::from_str(&data).unwrap();
        assert!(!cases.is_empty(), "expected >=1 case");
        for c in cases {
            let serialized = serde_json::to_string(&c.list).unwrap();
            assert_eq!(serialized, c.json, "serialize {}", c.name);
            let parsed: TaskList = serde_json::from_str(&c.json).unwrap();
            assert_eq!(parsed, c.list, "parse {}", c.name);
        }
    }
}
