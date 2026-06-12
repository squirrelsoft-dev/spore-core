//! Memory tool (#82, storage seam #78): the scope-aware read/write tool over the
//! persisted episodic [`MemoryStore`].
//!
//! One unit-struct tool, [`MemoryTool`] (`NAME = "memory"`), dispatched on an
//! `operation` discriminator (`write`, `read`). It is the agent-facing surface
//! over the scoped memory seam shipped in #78.
//!
//! ## Types
//!   - [`MemoryEntry`](crate::storage::MemoryEntry) — `{ role, content,
//!     timestamp, metadata }`. UNCHANGED by #82 — `metadata` already exists; the
//!     tool exposes it as an optional `write` param (decision C).
//!   - [`StorageScope`](crate::storage::StorageScope) — `{ User, Project, Local }`.
//!     `Local` is rejected at runtime on BOTH ops (see Rules).
//!   - [`MemoryToolParams`](crate::tools::params::MemoryToolParams) — the
//!     `operation`-tagged input enum.
//!
//! ## Trait methods used (storage seam #78)
//!   - [`ToolContext::memory_store()`](crate::tool_registry::ToolContext::memory_store)
//!     — the scope-aware [`MemoryStore`] slot.
//!   - [`MemoryStore::append_memory(scope, session_id, entry)`](crate::storage::MemoryStore::append_memory)
//!     for `write`.
//!   - [`MemoryStore::get_memories(scope, session_id, limit)`](crate::storage::MemoryStore::get_memories)
//!     for a scoped `read`.
//!   - [`MemoryStore::get_memories_merged(session_id, limit)`](crate::storage::MemoryStore::get_memories_merged)
//!     for a merged `read` — the SINGLE merge implementation (#82 D2), a default
//!     trait method (User ∪ Project, newest-first, no dedup, Local excluded).
//!
//! ## Rules enforced
//!   - **R1 write→read roundtrip.** A `write` appends one [`MemoryEntry`] to the
//!     given scope; a subsequent same-scope `read` returns it (newest-first).
//!   - **R2 write success content (decision A).** `write` returns the serialized
//!     just-written [`MemoryEntry`] as the [`ToolOutput::Success`] content.
//!   - **R3 read default limit (decision B).** `limit` defaults to `50`,
//!     overridable; `read` returns the most-recent `limit` entries newest-first.
//!   - **R4 metadata on write (decision C).** `metadata` is optional, defaults to
//!     `{}`; it is stored verbatim on the entry. `MemoryEntry` is NOT changed.
//!   - **R5 scope isolation.** A non-merged `read` of one scope never sees the
//!     other scope's entries.
//!   - **R6 merged read (decision D2).** `read` with `merged: true` returns the
//!     User ∪ Project merge via the trait's single merge method.
//!   - **R7 Local rejected on BOTH ops.** `Local` scope → recoverable
//!     [`ToolOutput::Error`] with the EXACT message
//!     `"Local scope is not supported by MemoryTool — use User or Project."`,
//!     checked BEFORE any storage access (nothing is written).
//!   - **R8 bad params recoverable.** Bad input maps via [`parse_params`] /
//!     [`ToolExecutionError::InvalidParameters`] to a recoverable error.
//!   - **R9 storage error recoverable.** A [`StorageError`](crate::storage::StorageError)
//!     from append/get maps to a recoverable [`ToolOutput::Error`].
//!   - **R10 read does not write.** A `read` performs no append.
//!
//! ## Annotations (decision E)
//! NOT annotated `read_only`. A `read_only` tool would be run CONCURRENTLY by
//! `dispatch_all` and could race the shared append; like [`TaskListTool`](crate::tools::TaskListTool)
//! this tool uses [`ToolAnnotations::default()`] (all false) so the registry
//! dispatches it sequentially.
//!
//! ## Known v1 limitation (#78 Q7)
//! Memory is [`SessionId`](crate::harness::SessionId)-keyed for v1: the tool
//! always uses `ctx.session_id()` and offers NO cross-session addressing param.
//! v2 should add session-independent memory keying — do not introduce it here.

use serde_json::json;

use crate::harness::{BoxFut, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::storage::{MemoryEntry, StorageScope};
use crate::tool_registry::{Tool, ToolAnnotations, ToolContext, ToolSchema};
use crate::tools::params::{parse_params, MemoryToolParams};

/// Exact error message returned when a `Local`-scoped op is attempted.
const LOCAL_REJECTED_MESSAGE: &str =
    "Local scope is not supported by MemoryTool — use User or Project.";

pub struct MemoryTool;

impl MemoryTool {
    pub const NAME: &'static str = "memory";

    pub fn new() -> Self {
        Self
    }

    pub fn schema() -> ToolSchema {
        // Properties kept sorted/stable for cache stability (see registry spec).
        // `scope` advertises only user/project — `local` is rejected at runtime
        // but intentionally omitted from the advertised enum.
        ToolSchema {
            name: Self::NAME.into(),
            description: "Read or write scope-aware episodic memory for this session".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "limit": {"type": "integer"},
                    "merged": {"type": "boolean"},
                    "metadata": {"type": "object"},
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write"],
                    },
                    "role": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "enum": ["project", "user"],
                    },
                },
                "required": ["operation", "scope"],
            }),
            // Intentionally NOT read_only: the shared append must dispatch
            // sequentially. See module docs (decision E).
            annotations: ToolAnnotations::default(),
        }
    }
}

impl Default for MemoryTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MemoryTool {
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
            let memory_store = ctx.memory_store();

            // 1. Parse params (bad input → recoverable).
            let params: MemoryToolParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };

            match params {
                MemoryToolParams::Write {
                    scope,
                    role,
                    content,
                    metadata,
                } => {
                    // R7: reject Local BEFORE touching storage; nothing written.
                    if scope == StorageScope::Local {
                        return ToolOutput::Error {
                            message: LOCAL_REJECTED_MESSAGE.into(),
                            recoverable: true,
                        };
                    }

                    // Build the entry. v1 timestamps are caller-provided via the
                    // entry's serde shape — here the tool stamps "now" through the
                    // metadata-preserving constructor path.
                    let entry = MemoryEntry {
                        role,
                        content,
                        timestamp: now_timestamp(),
                        metadata,
                    };

                    if let Err(e) = memory_store
                        .append_memory(scope, session_id, entry.clone())
                        .await
                    {
                        return ToolOutput::Error {
                            message: format!("could not append memory: {e}"),
                            recoverable: true,
                        };
                    }

                    // R2 (decision A): success content = the serialized entry.
                    match serde_json::to_string(&entry) {
                        Ok(content) => ToolOutput::Success {
                            content,
                            truncated: false,
                        },
                        Err(e) => ToolOutput::Error {
                            message: format!("could not serialize memory entry: {e}"),
                            recoverable: true,
                        },
                    }
                }
                MemoryToolParams::Read {
                    scope,
                    merged,
                    limit,
                } => {
                    // R7: reject Local on read too, before any storage access.
                    if scope == StorageScope::Local {
                        return ToolOutput::Error {
                            message: LOCAL_REJECTED_MESSAGE.into(),
                            recoverable: true,
                        };
                    }

                    // R6 (decision D2): merged read drives the single trait merge.
                    // Otherwise a scoped read (R5 isolation).
                    let result = if merged {
                        memory_store.get_memories_merged(session_id, limit).await
                    } else {
                        memory_store.get_memories(scope, session_id, limit).await
                    };

                    let entries = match result {
                        Ok(e) => e,
                        Err(e) => {
                            return ToolOutput::Error {
                                message: format!("could not read memory: {e}"),
                                recoverable: true,
                            };
                        }
                    };

                    match serde_json::to_string(&entries) {
                        Ok(content) => ToolOutput::Success {
                            content,
                            truncated: false,
                        },
                        Err(e) => ToolOutput::Error {
                            message: format!("could not serialize memory entries: {e}"),
                            recoverable: true,
                        },
                    }
                }
            }
        })
    }
}

/// The wall-clock timestamp stamped on a freshly-written entry: RFC-3339 UTC at
/// seconds precision (`YYYY-MM-DDTHH:MM:SSZ`), derived from the Unix epoch with
/// the civil-date algorithm — no external date crate, matching the `Timestamp`
/// string shape the fixtures use.
fn now_timestamp() -> crate::memory::Timestamp {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    crate::memory::Timestamp::new(format_rfc3339_secs(secs))
}

/// Format a Unix-epoch second count as `YYYY-MM-DDTHH:MM:SSZ`. Pure arithmetic
/// (days-from-civil), byte-identical across languages.
fn format_rfc3339_secs(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // days since 1970-01-01 → civil date (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{BoxFut, SandboxProvider, SandboxViolation, SessionId};
    use crate::memory::Timestamp;
    use crate::storage::{InMemoryStorageProvider, MemoryStore, StorageError};
    use std::sync::Arc;

    /// Permissive sandbox — the tool never touches the filesystem.
    struct AllowAllSandbox;
    impl SandboxProvider for AllowAllSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async move { Ok(()) })
        }
    }

    /// A MemoryStore that always fails, to prove storage errors map to a
    /// recoverable tool error. (Mirrors the `FailingRunStore` pattern.)
    struct FailingMemoryStore;
    impl MemoryStore for FailingMemoryStore {
        fn append_memory<'a>(
            &'a self,
            _scope: StorageScope,
            _session_id: &'a SessionId,
            _entry: MemoryEntry,
        ) -> BoxFut<'a, Result<(), StorageError>> {
            Box::pin(async move {
                Err(StorageError::Backend {
                    message: "boom".into(),
                })
            })
        }
        fn get_memories<'a>(
            &'a self,
            _scope: StorageScope,
            _session_id: &'a SessionId,
            _limit: usize,
        ) -> BoxFut<'a, Result<Vec<MemoryEntry>, StorageError>> {
            Box::pin(async move {
                Err(StorageError::Backend {
                    message: "boom".into(),
                })
            })
        }
    }

    fn ctx_with(memory_store: Arc<dyn MemoryStore>, session: &str) -> ToolContext {
        // MemoryTool exercises the memory seam only; the run store is a no-op
        // in-memory backend here. Memory stays session-keyed (#142 keeps memory
        // ephemeral); the project_id is a placeholder for the seam.
        ToolContext::new(
            SessionId::new(session),
            crate::storage::ProjectId::from_canonical_path("/memory-test-project"),
            Arc::new(InMemoryStorageProvider::new()),
            memory_store,
        )
    }

    fn in_memory_ctx() -> ToolContext {
        ctx_with(Arc::new(InMemoryStorageProvider::new()), "test-session")
    }

    fn call(input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: MemoryTool::NAME.into(),
            input,
        }
    }

    fn parse_entries(out: &ToolOutput) -> Vec<MemoryEntry> {
        match out {
            ToolOutput::Success { content, .. } => serde_json::from_str(content).unwrap(),
            other => panic!("expected Success, got {other:?}"),
        }
    }

    fn parse_entry(out: &ToolOutput) -> MemoryEntry {
        match out {
            ToolOutput::Success { content, .. } => serde_json::from_str(content).unwrap(),
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // R1 + R2: write→read roundtrip; write returns the serialized entry.
    #[tokio::test]
    async fn write_then_read_roundtrip() {
        let ctx = in_memory_ctx();
        let tool = MemoryTool::new();

        let w = tool
            .execute(
                &call(json!({
                    "operation": "write",
                    "scope": "user",
                    "role": "user",
                    "content": "hello",
                })),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        let written = parse_entry(&w);
        assert_eq!(written.role, "user");
        assert_eq!(written.content, "hello");
        assert_eq!(written.metadata, json!({})); // R4 default {}

        let r = tool
            .execute(
                &call(json!({"operation": "read", "scope": "user"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        let entries = parse_entries(&r);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "hello");
    }

    // R4: metadata is stored verbatim on the entry.
    #[tokio::test]
    async fn write_preserves_metadata() {
        let ctx = in_memory_ctx();
        let tool = MemoryTool::new();
        let w = tool
            .execute(
                &call(json!({
                    "operation": "write",
                    "scope": "project",
                    "role": "assistant",
                    "content": "c",
                    "metadata": {"k": "v", "n": 3},
                })),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        assert_eq!(parse_entry(&w).metadata, json!({"k": "v", "n": 3}));
    }

    // R5: non-merged scope isolation.
    #[tokio::test]
    async fn scoped_read_does_not_see_other_scope() {
        let ctx = in_memory_ctx();
        let tool = MemoryTool::new();
        tool.execute(
            &call(json!({"operation": "write", "scope": "user", "role": "user", "content": "u1"})),
            &AllowAllSandbox,
            &ctx,
        )
        .await;
        tool.execute(
            &call(json!({"operation": "write", "scope": "project", "role": "assistant", "content": "p1"})),
            &AllowAllSandbox,
            &ctx,
        )
        .await;

        let r_user = tool
            .execute(
                &call(json!({"operation": "read", "scope": "user"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        let user = parse_entries(&r_user);
        assert_eq!(user.len(), 1);
        assert_eq!(user[0].content, "u1");

        let r_proj = tool
            .execute(
                &call(json!({"operation": "read", "scope": "project"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        let proj = parse_entries(&r_proj);
        assert_eq!(proj.len(), 1);
        assert_eq!(proj[0].content, "p1");
    }

    // R6: merged read drives the shared merge fixture — both `dup` survive,
    // `local` absent, newest-first.
    #[tokio::test]
    async fn merged_read_fixture_replay() {
        use crate::storage::CompositeStorageProvider;
        let raw = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../fixtures/storage/memory_scoped_merge.json"),
        )
        .expect("memory_scoped_merge.json present");
        let f: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let limit = f["limit"].as_u64().unwrap() as usize;

        // Build a scope-routing provider and seed each scope.
        let provider = CompositeStorageProvider::new()
            .memory(StorageScope::User, Arc::new(InMemoryStorageProvider::new()))
            .memory(
                StorageScope::Project,
                Arc::new(InMemoryStorageProvider::new()),
            )
            .memory(
                StorageScope::Local,
                Arc::new(InMemoryStorageProvider::new()),
            )
            .build();
        let memory_store = provider.memory().clone();
        let sid = SessionId::new("s");
        for (key, scope) in [
            ("user", StorageScope::User),
            ("project", StorageScope::Project),
            ("local", StorageScope::Local),
        ] {
            for entry in f[key].as_array().unwrap() {
                let e: MemoryEntry = serde_json::from_value(entry.clone()).unwrap();
                memory_store.append_memory(scope, &sid, e).await.unwrap();
            }
        }

        let ctx = ToolContext::new(
            sid,
            crate::storage::ProjectId::from_canonical_path("/memory-test-project"),
            Arc::new(InMemoryStorageProvider::new()),
            memory_store,
        );
        let tool = MemoryTool::new();
        let out = tool
            .execute(
                &call(
                    json!({"operation": "read", "scope": "user", "merged": true, "limit": limit}),
                ),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        let contents: Vec<String> = parse_entries(&out)
            .iter()
            .map(|e| e.content.clone())
            .collect();
        let expected: Vec<String> = f["expected_merged_contents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(contents, expected);
        assert_eq!(contents.iter().filter(|c| *c == "dup").count(), 2);
        assert!(!contents.iter().any(|c| c.contains("should-not-appear")));
    }

    // R7: Local rejected on write — exact message, nothing written.
    #[tokio::test]
    async fn local_rejected_on_write_writes_nothing() {
        let ctx = in_memory_ctx();
        let tool = MemoryTool::new();
        let out = tool
            .execute(
                &call(json!({
                    "operation": "write",
                    "scope": "local",
                    "role": "user",
                    "content": "x",
                })),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        match &out {
            ToolOutput::Error {
                message,
                recoverable,
            } => {
                assert!(*recoverable);
                assert_eq!(message, LOCAL_REJECTED_MESSAGE);
            }
            other => panic!("expected Error, got {other:?}"),
        }
        // Nothing was written to ANY scope.
        for scope in [
            StorageScope::User,
            StorageScope::Project,
            StorageScope::Local,
        ] {
            let got = ctx
                .memory_store()
                .get_memories(scope, ctx.session_id(), 50)
                .await
                .unwrap();
            assert!(got.is_empty(), "scope {scope:?} should be empty");
        }
    }

    // R7: Local rejected on read — exact message.
    #[tokio::test]
    async fn local_rejected_on_read() {
        let ctx = in_memory_ctx();
        let out = MemoryTool::new()
            .execute(
                &call(json!({"operation": "read", "scope": "local"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        match out {
            ToolOutput::Error {
                message,
                recoverable,
            } => {
                assert!(recoverable);
                assert_eq!(message, LOCAL_REJECTED_MESSAGE);
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    // Session isolation: two sessions over the SAME store keep separate memory.
    #[tokio::test]
    async fn memory_is_keyed_by_session_id() {
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStorageProvider::new());
        let tool = MemoryTool::new();
        let ctx_a = ctx_with(store.clone(), "session-a");
        let ctx_b = ctx_with(store.clone(), "session-b");

        tool.execute(
            &call(json!({"operation": "write", "scope": "user", "role": "user", "content": "a1"})),
            &AllowAllSandbox,
            &ctx_a,
        )
        .await;
        tool.execute(
            &call(json!({"operation": "write", "scope": "user", "role": "user", "content": "b1"})),
            &AllowAllSandbox,
            &ctx_b,
        )
        .await;

        let a = store
            .get_memories(StorageScope::User, &SessionId::new("session-a"), 50)
            .await
            .unwrap();
        let b = store
            .get_memories(StorageScope::User, &SessionId::new("session-b"), 50)
            .await
            .unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].content, "a1");
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].content, "b1");
    }

    // R8: bad params → recoverable error.
    #[tokio::test]
    async fn bad_params_is_recoverable_error() {
        let ctx = in_memory_ctx();
        // Unknown operation.
        let r = MemoryTool::new()
            .execute(&call(json!({"operation": "nope"})), &AllowAllSandbox, &ctx)
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
        // Missing required field on write.
        let r = MemoryTool::new()
            .execute(
                &call(json!({"operation": "write", "scope": "user", "role": "user"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    // R9: storage failure → recoverable error (both write and read paths).
    #[tokio::test]
    async fn storage_failure_is_recoverable_error() {
        let ctx = ctx_with(Arc::new(FailingMemoryStore), "test-session");
        let w = MemoryTool::new()
            .execute(
                &call(json!({
                    "operation": "write",
                    "scope": "user",
                    "role": "user",
                    "content": "x",
                })),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        match w {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("expected recoverable error, got {other:?}"),
        }
        let r = MemoryTool::new()
            .execute(
                &call(json!({"operation": "read", "scope": "user"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("expected recoverable error, got {other:?}"),
        }
    }

    // R10: read does not write — a read against a fresh store leaves it empty.
    #[tokio::test]
    async fn read_does_not_write() {
        let ctx = in_memory_ctx();
        let r = MemoryTool::new()
            .execute(
                &call(json!({"operation": "read", "scope": "user"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        assert!(parse_entries(&r).is_empty());
        let got = ctx
            .memory_store()
            .get_memories(StorageScope::User, ctx.session_id(), 50)
            .await
            .unwrap();
        assert!(got.is_empty());
    }

    // Schema is NOT read_only (decision E).
    #[tokio::test]
    async fn schema_is_not_read_only() {
        let s = MemoryTool::schema();
        assert!(!s.annotations.read_only);
        assert!(!s.annotations.destructive);
        assert!(!s.annotations.open_world);
        assert_eq!(s.name, "memory");
    }

    // merged read respects limit: write 3 user entries, merged read limit=2.
    #[tokio::test]
    async fn merged_read_respects_limit() {
        let provider = crate::storage::CompositeStorageProvider::new()
            .memory(StorageScope::User, Arc::new(InMemoryStorageProvider::new()))
            .memory(
                StorageScope::Project,
                Arc::new(InMemoryStorageProvider::new()),
            )
            .build();
        let memory_store = provider.memory().clone();
        let sid = SessionId::new("s");
        for (i, ts) in [
            "2026-05-01T00:00:00Z",
            "2026-05-02T00:00:00Z",
            "2026-05-03T00:00:00Z",
        ]
        .iter()
        .enumerate()
        {
            memory_store
                .append_memory(
                    StorageScope::User,
                    &sid,
                    MemoryEntry::new("user", format!("u{i}"), Timestamp::new(*ts)),
                )
                .await
                .unwrap();
        }
        let ctx = ToolContext::new(
            sid,
            crate::storage::ProjectId::from_canonical_path("/memory-test-project"),
            Arc::new(InMemoryStorageProvider::new()),
            memory_store,
        );
        let out = MemoryTool::new()
            .execute(
                &call(json!({"operation": "read", "scope": "user", "merged": true, "limit": 2})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        let entries = parse_entries(&out);
        assert_eq!(entries.len(), 2);
        // Newest-first.
        assert_eq!(entries[0].content, "u2");
        assert_eq!(entries[1].content, "u1");
    }

    // read default limit is 50 (decision B): more than 50 entries → 50 returned.
    #[tokio::test]
    async fn read_default_limit_is_50() {
        let ctx = in_memory_ctx();
        let tool = MemoryTool::new();
        for i in 0..60 {
            tool.execute(
                &call(json!({
                    "operation": "write",
                    "scope": "user",
                    "role": "user",
                    "content": format!("m{i}"),
                })),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        }
        let r = tool
            .execute(
                &call(json!({"operation": "read", "scope": "user"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        assert_eq!(parse_entries(&r).len(), 50);
    }

    // ========================================================================
    // Fixture replay — fixtures/tools/memory.json
    // ========================================================================

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/tools")
            .join(name)
    }

    #[derive(serde::Deserialize)]
    struct OpStep {
        input: serde_json::Value,
        expected: OpExpected,
    }
    #[derive(serde::Deserialize)]
    struct OpExpected {
        ok: bool,
        /// For an ok `read`: the expected list of entry `content` values
        /// (ordered, unless `unordered` is set).
        #[serde(default)]
        contents: Option<Vec<String>>,
        /// When true, compare `contents` as a multiset (order-independent). Used
        /// for merged reads, where the relative order of entries written within
        /// the same wall-clock second is not deterministic.
        #[serde(default)]
        unordered: bool,
        /// For an error step: the exact expected message.
        #[serde(default)]
        error: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct OpScenario {
        name: String,
        steps: Vec<OpStep>,
    }

    // Replay each operations scenario step-by-step against a fresh scope-routing
    // provider, asserting per-step outcomes. Byte-identical across languages.
    #[tokio::test]
    async fn fixture_replay_operations() {
        use crate::storage::CompositeStorageProvider;
        let data = std::fs::read_to_string(fixture_path("memory.json")).unwrap();
        let scenarios: Vec<OpScenario> = serde_json::from_str(&data).unwrap();
        assert!(!scenarios.is_empty(), "expected >=1 scenario");
        let tool = MemoryTool::new();
        let sb = AllowAllSandbox;

        for sc in scenarios {
            // Fresh isolated scope-routing provider per scenario.
            let provider = CompositeStorageProvider::new()
                .memory(StorageScope::User, Arc::new(InMemoryStorageProvider::new()))
                .memory(
                    StorageScope::Project,
                    Arc::new(InMemoryStorageProvider::new()),
                )
                .build();
            let ctx = ToolContext::new(
                SessionId::new("fx"),
                crate::storage::ProjectId::from_canonical_path("/memory-fx-project"),
                Arc::new(InMemoryStorageProvider::new()),
                provider.memory().clone(),
            );
            for (i, step) in sc.steps.iter().enumerate() {
                let out = tool.execute(&call(step.input.clone()), &sb, &ctx).await;
                match (&out, step.expected.ok) {
                    (ToolOutput::Success { content, .. }, true) => {
                        // A read asserts its content list; a write only asserts ok.
                        if let Some(want) = &step.expected.contents {
                            let entries: Vec<MemoryEntry> = serde_json::from_str(content).unwrap();
                            let mut got: Vec<String> =
                                entries.iter().map(|e| e.content.clone()).collect();
                            if step.expected.unordered {
                                let mut want_sorted = want.clone();
                                want_sorted.sort();
                                got.sort();
                                assert_eq!(got, want_sorted, "{} step {i}", sc.name);
                            } else {
                                assert_eq!(&got, want, "{} step {i}", sc.name);
                            }
                        }
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
                        assert_eq!(message, want, "{} step {i}", sc.name);
                    }
                    (other, expected_ok) => {
                        panic!("{} step {i}: ok={expected_ok} but got {other:?}", sc.name)
                    }
                }
            }
        }
    }
}
