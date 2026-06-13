//! Inline unit + fixture-replay tests for the [`StorageProvider`] abstraction
//! (issue #73). Covers every pinned rule: no-op fallback, composite per-domain
//! routing, single-provider-fills-all-slots, OTLP parse table, atomic write,
//! append ordering, get_memories recency, run-store roundtrip, session-store
//! roundtrip, flush markers, and cross-language fixture replay.

use super::*;
use crate::harness::{
    BudgetSnapshot, HumanRequest, LoopStrategy, PausedState, ReactConfig, RiskLevel, SessionId,
    SessionState, Task, TaskId, ToolsetRef,
};
use crate::memory::Timestamp;
use serde_json::json;
use std::path::{Path, PathBuf};

fn sid(s: &str) -> SessionId {
    SessionId::new(s)
}
fn ts(s: &str) -> Timestamp {
    Timestamp::new(s)
}

/// Minimal valid PausedState for roundtrip tests.
fn paused(session: &str) -> PausedState {
    PausedState {
        session_id: sid(session),
        task_id: TaskId::new("task1"),
        turn_number: 3,
        session_state: SessionState::default(),
        pending_tool_calls: Vec::new(),
        approved_results: Vec::new(),
        human_request: Some(HumanRequest::ToolApproval {
            calls: Vec::new(),
            risk_level: RiskLevel::Low,
        }),
        task: Task::new(
            "do the thing",
            sid(session),
            LoopStrategy::ReAct(ReactConfig::per_loop(1)),
        ),
        budget_used: BudgetSnapshot::default(),
        child_state: None,
        toolset: ToolsetRef::default(),
    }
}

fn mem(role: &str, content: &str, t: &str) -> MemoryEntry {
    MemoryEntry::new(role, content, ts(t))
}

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../fixtures/storage")
        .join(name)
}

// ── OTLP endpoint parsing (the most important cross-language rule) ───────────

#[test]
fn otlp_parse_table() {
    assert_eq!(parse_otlp_endpoints("a"), vec!["a"]);
    assert_eq!(parse_otlp_endpoints("a,b,c"), vec!["a", "b", "c"]);
    assert_eq!(parse_otlp_endpoints(" a , b "), vec!["a", "b"]);
    assert_eq!(parse_otlp_endpoints("a,,b,"), vec!["a", "b"]);
    assert_eq!(parse_otlp_endpoints(""), Vec::<String>::new());
    assert_eq!(parse_otlp_endpoints("  "), Vec::<String>::new());
}

#[test]
fn otlp_parse_fixture_replay() {
    let raw = std::fs::read_to_string(fixture_path("otlp_endpoints_parse.json"))
        .expect("otlp_endpoints_parse.json present");
    let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap();
    for case in cases {
        let input = case["input"].as_str().unwrap();
        let expected: Vec<String> = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            parse_otlp_endpoints(input),
            expected,
            "mismatch for input {input:?}"
        );
    }
}

// ── No-op fallback ───────────────────────────────────────────────────────────

#[tokio::test]
async fn noop_reads_empty_writes_ok() {
    let p = NoOpStorageProvider;
    assert!(SessionStore::get_session(&p, &sid("s"))
        .await
        .unwrap()
        .is_none());
    assert!(p.list_sessions().await.unwrap().is_empty());
    assert!(p.put_session(&sid("s"), &paused("s")).await.is_ok());
    assert!(
        MemoryStore::get_memories(&p, StorageScope::Project, &sid("s"), 10)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(p
        .append_memory(StorageScope::Project, &sid("s"), mem("user", "hi", "t"))
        .await
        .is_ok());
    assert!(RunStore::get(&p, &sid("s"), "k").await.unwrap().is_none());
    assert!(p.put(&sid("s"), "k", json!(1)).await.is_ok());
    assert!(p.list_keys(&sid("s")).await.unwrap().is_empty());
    assert!(ObservabilityStore::get_spans(&p, &sid("s"))
        .await
        .unwrap()
        .is_empty());
    assert!(p.append_span(&sid("s"), json!({})).await.is_ok());
}

#[test]
fn default_storage_provider_is_noop() {
    // StorageProvider::default() / no_op() must not panic and exposes all slots.
    let p = StorageProvider::default();
    let _ = p.session();
    let _ = p.memory();
    let _ = p.run();
    let _ = p.observability();
}

// ── Single-provider-fills-all-slots ──────────────────────────────────────────

#[tokio::test]
async fn single_fills_all_slots() {
    let backend = Arc::new(InMemoryStorageProvider::new());
    let p = StorageProvider::single(backend);
    // Write through each domain accessor; reads see them — proving all four
    // slots share the one backend.
    p.session()
        .put_session(&sid("s"), &paused("s"))
        .await
        .unwrap();
    p.memory()
        .append_memory(StorageScope::Project, &sid("s"), mem("user", "hi", "t1"))
        .await
        .unwrap();
    p.run()
        .put(&sid("s"), "plan", json!({"x": 1}))
        .await
        .unwrap();
    p.observability()
        .append_span(&sid("s"), json!({"kind": "turn"}))
        .await
        .unwrap();

    assert!(p.session().get_session(&sid("s")).await.unwrap().is_some());
    assert_eq!(
        p.memory()
            .get_memories(StorageScope::Project, &sid("s"), 10)
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        p.run().get(&sid("s"), "plan").await.unwrap(),
        Some(json!({"x": 1}))
    );
    assert_eq!(
        p.observability().get_spans(&sid("s")).await.unwrap().len(),
        1
    );
}

// ── Composite per-domain routing + no-op fallback ────────────────────────────

#[tokio::test]
async fn composite_routes_per_domain_and_falls_back_to_noop() {
    let run_backend = Arc::new(InMemoryStorageProvider::new());
    // Only the run domain is configured; the other three fall back to no-op.
    let p = CompositeStorageProvider::new().run(run_backend).build();

    p.run().put(&sid("s"), "k", json!("v")).await.unwrap();
    assert_eq!(p.run().get(&sid("s"), "k").await.unwrap(), Some(json!("v")));

    // Unconfigured domains silently no-op.
    p.session()
        .put_session(&sid("s"), &paused("s"))
        .await
        .unwrap();
    assert!(p.session().get_session(&sid("s")).await.unwrap().is_none());
    assert!(p
        .memory()
        .get_memories(StorageScope::Project, &sid("s"), 5)
        .await
        .unwrap()
        .is_empty());
    assert!(p
        .observability()
        .get_spans(&sid("s"))
        .await
        .unwrap()
        .is_empty());
}

// ── In-memory: session-store roundtrip + list + delete ───────────────────────

#[tokio::test]
async fn in_memory_session_roundtrip_list_delete() {
    let p = InMemoryStorageProvider::new();
    p.put_session(&sid("b"), &paused("b")).await.unwrap();
    p.put_session(&sid("a"), &paused("a")).await.unwrap();
    let got = p.get_session(&sid("a")).await.unwrap().unwrap();
    assert_eq!(got.session_id, sid("a"));
    // list is sorted.
    assert_eq!(p.list_sessions().await.unwrap(), vec![sid("a"), sid("b")]);
    p.delete_session(&sid("a")).await.unwrap();
    assert!(p.get_session(&sid("a")).await.unwrap().is_none());
    assert_eq!(p.list_sessions().await.unwrap(), vec![sid("b")]);
}

// ── In-memory: run-store opaque-json roundtrip + list_keys + delete ──────────

#[tokio::test]
async fn in_memory_run_roundtrip_list_delete() {
    let p = InMemoryStorageProvider::new();
    let blob = json!({"nested": {"arr": [1, 2, 3], "s": "x"}, "n": 4.5});
    p.put(&sid("s"), "plan", blob.clone()).await.unwrap();
    p.put(&sid("s"), "tasks", json!([1, 2])).await.unwrap();
    // opaque-json roundtrips byte-equal.
    assert_eq!(p.get(&sid("s"), "plan").await.unwrap(), Some(blob));
    // list_keys sorted, scoped to the session.
    assert_eq!(
        p.list_keys(&sid("s")).await.unwrap(),
        vec!["plan".to_string(), "tasks".to_string()]
    );
    p.delete(&sid("s"), "plan").await.unwrap();
    assert!(p.get(&sid("s"), "plan").await.unwrap().is_none());
    assert_eq!(
        p.list_keys(&sid("s")).await.unwrap(),
        vec!["tasks".to_string()]
    );
}

// ── In-memory: memory append ordering + recency limit ────────────────────────

#[tokio::test]
async fn in_memory_memory_recency_and_limit() {
    let p = InMemoryStorageProvider::new();
    for (i, c) in ["m0", "m1", "m2", "m3"].iter().enumerate() {
        p.append_memory(
            StorageScope::Project,
            &sid("s"),
            mem("user", c, &format!("t{i}")),
        )
        .await
        .unwrap();
    }
    // most-recent 2, newest-first.
    let got = p
        .get_memories(StorageScope::Project, &sid("s"), 2)
        .await
        .unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].content, "m3");
    assert_eq!(got[1].content, "m2");
    // limit larger than count returns all, still newest-first.
    let all = p
        .get_memories(StorageScope::Project, &sid("s"), 99)
        .await
        .unwrap();
    assert_eq!(
        all.iter().map(|e| e.content.clone()).collect::<Vec<_>>(),
        vec!["m3", "m2", "m1", "m0"]
    );
}

#[tokio::test]
async fn in_memory_spans_append_ordering() {
    let p = InMemoryStorageProvider::new();
    p.append_span(&sid("s"), json!({"n": 0})).await.unwrap();
    p.append_span(&sid("s"), json!({"n": 1})).await.unwrap();
    let spans = p.get_spans(&sid("s")).await.unwrap();
    assert_eq!(spans, vec![json!({"n": 0}), json!({"n": 1})]);
}

// ── FileSystem: atomic write (no leftover .tmp) ──────────────────────────────

#[tokio::test]
async fn fs_atomic_write_leaves_no_tmp() {
    let tmp = tempfile::tempdir().unwrap();
    let p = FileSystemStorageProvider::new(tmp.path());
    p.put_session(&sid("s"), &paused("s")).await.unwrap();
    p.put(&sid("s"), "k", json!({"a": 1})).await.unwrap();
    // No leftover .tmp anywhere.
    let mut leftovers = Vec::new();
    for entry in walkdir::WalkDir::new(tmp.path()).into_iter().flatten() {
        if entry.file_name().to_string_lossy().ends_with(".tmp") {
            leftovers.push(entry.path().to_path_buf());
        }
    }
    assert!(leftovers.is_empty(), "leftover .tmp files: {leftovers:?}");
    // and the canonical layout is used.
    assert!(tmp.path().join("sessions/s/state.json").exists());
    assert!(tmp.path().join("sessions/s/run/k.json").exists());
}

#[tokio::test]
async fn fs_session_roundtrip_list_delete() {
    let tmp = tempfile::tempdir().unwrap();
    let p = FileSystemStorageProvider::new(tmp.path());
    p.put_session(&sid("a"), &paused("a")).await.unwrap();
    p.put_session(&sid("b"), &paused("b")).await.unwrap();
    let got = p.get_session(&sid("a")).await.unwrap().unwrap();
    assert_eq!(got.turn_number, 3);
    assert_eq!(p.list_sessions().await.unwrap(), vec![sid("a"), sid("b")]);
    p.delete_session(&sid("a")).await.unwrap();
    assert!(p.get_session(&sid("a")).await.unwrap().is_none());
    // delete of missing is Ok.
    assert!(p.delete_session(&sid("missing")).await.is_ok());
}

#[tokio::test]
async fn fs_run_roundtrip_list_delete() {
    let tmp = tempfile::tempdir().unwrap();
    let p = FileSystemStorageProvider::new(tmp.path());
    let blob = json!({"deep": [true, null, "x"]});
    p.put(&sid("s"), "plan", blob.clone()).await.unwrap();
    p.put(&sid("s"), "tasks", json!(7)).await.unwrap();
    assert_eq!(p.get(&sid("s"), "plan").await.unwrap(), Some(blob));
    assert_eq!(
        p.list_keys(&sid("s")).await.unwrap(),
        vec!["plan".to_string(), "tasks".to_string()]
    );
    p.delete(&sid("s"), "plan").await.unwrap();
    assert!(p.get(&sid("s"), "plan").await.unwrap().is_none());
    assert!(p.get(&sid("missing"), "x").await.unwrap().is_none());
}

#[tokio::test]
async fn fs_memory_append_recency_and_jsonl_path() {
    let tmp = tempfile::tempdir().unwrap();
    let p = FileSystemStorageProvider::new(tmp.path());
    for (i, c) in ["a", "b", "c"].iter().enumerate() {
        p.append_memory(
            StorageScope::Project,
            &sid("s"),
            mem("user", c, &format!("t{i}")),
        )
        .await
        .unwrap();
    }
    assert!(tmp.path().join("sessions/s/memory.jsonl").exists());
    let got = p
        .get_memories(StorageScope::Project, &sid("s"), 2)
        .await
        .unwrap();
    assert_eq!(got[0].content, "c");
    assert_eq!(got[1].content, "b");
    // metadata defaults to {}.
    assert_eq!(got[0].metadata, json!({}));
}

#[tokio::test]
async fn fs_spans_append_and_flush_marker() {
    let tmp = tempfile::tempdir().unwrap();
    let p = FileSystemStorageProvider::new(tmp.path());
    p.append_span(&sid("s"), json!({"n": 0})).await.unwrap();
    p.append_span(&sid("s"), json!({"n": 1})).await.unwrap();
    assert!(tmp.path().join("sessions/s/trace.jsonl").exists());
    let spans = p.get_spans(&sid("s")).await.unwrap();
    assert_eq!(spans, vec![json!({"n": 0}), json!({"n": 1})]);
    // flush_session writes the .flushed marker.
    ObservabilityStore::flush_session(&p, &sid("s"))
        .await
        .unwrap();
    assert!(tmp.path().join("sessions/s/.flushed").exists());
}

// ── MemoryEntry default metadata ─────────────────────────────────────────────

#[test]
fn memory_entry_metadata_defaults_to_empty_object() {
    // Deserialize without `metadata` → defaults to {}.
    let e: MemoryEntry = serde_json::from_value(json!({
        "role": "user",
        "content": "hi",
        "timestamp": "2026-05-28T00:00:00Z"
    }))
    .unwrap();
    assert_eq!(e.metadata, json!({}));
    // Roundtrip preserves the shape.
    let v = serde_json::to_value(&e).unwrap();
    assert_eq!(v["role"], "user");
    assert_eq!(v["content"], "hi");
    assert_eq!(v["metadata"], json!({}));
}

// ── Fixture replay: run_store_values + memory_entries ────────────────────────

#[tokio::test]
async fn run_store_values_fixture_replay() {
    let raw = std::fs::read_to_string(fixture_path("run_store_values.json"))
        .expect("run_store_values.json present");
    let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap();
    let p = InMemoryStorageProvider::new();
    let fsp_dir = tempfile::tempdir().unwrap();
    let fsp = FileSystemStorageProvider::new(fsp_dir.path());
    for case in cases {
        let key = case["key"].as_str().unwrap();
        let value = case["value"].clone();
        // In-memory roundtrip.
        p.put(&sid("s"), key, value.clone()).await.unwrap();
        assert_eq!(
            p.get(&sid("s"), key).await.unwrap(),
            Some(value.clone()),
            "in-memory roundtrip mismatch for key {key}"
        );
        // Filesystem roundtrip (opaque JSON survives the atomic write).
        fsp.put(&sid("s"), key, value.clone()).await.unwrap();
        assert_eq!(
            fsp.get(&sid("s"), key).await.unwrap(),
            Some(value),
            "fs roundtrip mismatch for key {key}"
        );
    }
}

#[tokio::test]
async fn memory_entries_fixture_replay() {
    let raw = std::fs::read_to_string(fixture_path("memory_entries.jsonl"))
        .expect("memory_entries.jsonl present");
    let entries: Vec<MemoryEntry> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(entries.len() >= 3, "fixture should carry several entries");

    let p = InMemoryStorageProvider::new();
    for e in &entries {
        p.append_memory(StorageScope::Project, &sid("s"), e.clone())
            .await
            .unwrap();
    }
    // get_memories(limit=2) returns the last two, newest-first.
    let got = p
        .get_memories(StorageScope::Project, &sid("s"), 2)
        .await
        .unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0], *entries.last().unwrap());
    assert_eq!(got[1], entries[entries.len() - 2]);
    // The shape roundtrips byte-identically.
    let reser: Vec<MemoryEntry> = (0..entries.len())
        .rev()
        .map(|i| entries[i].clone())
        .collect();
    let all = p
        .get_memories(StorageScope::Project, &sid("s"), 999)
        .await
        .unwrap();
    assert_eq!(all, reser);
}

// ════════════════════════════════════════════════════════════════════════════
// #78 — scope + workspace-partitioning extension
// ════════════════════════════════════════════════════════════════════════════

// ── R2: WorkspaceId derivation ───────────────────────────────────────────────

#[test]
fn workspace_id_is_deterministic_and_pure() {
    // Same input → same id, on repeated calls.
    let a = WorkspaceId::from_canonical_path("/Users/sbeardsley/dev/spore-core");
    let b = WorkspaceId::from_canonical_path("/Users/sbeardsley/dev/spore-core");
    assert_eq!(a, b);
    // Form is `{sanitized_basename}-{8hex}`.
    assert!(a.as_str().starts_with("spore-core-"));
    assert_eq!(a.as_str().len(), "spore-core-".len() + 8);
}

#[test]
fn workspace_id_root_uses_literal_root_basename() {
    let w = WorkspaceId::from_canonical_path("/");
    assert!(w.as_str().starts_with("root-"));
}

#[test]
fn workspace_id_sanitizes_special_chars_and_collapses_dashes() {
    let w = WorkspaceId::from_canonical_path("/Users/me/My Project (v2)!");
    // spaces/punctuation → `-`, collapsed, trailing punctuation stripped.
    assert!(w.as_str().starts_with("my-project-v2-"));
    assert!(!w.as_str().contains("--"));
}

#[test]
fn workspace_id_ignores_trailing_slash() {
    // No trailing slash in the canonical string → same id.
    let a = WorkspaceId::from_canonical_path("/Users/sbeardsley/dev/spore-core");
    let b = WorkspaceId::from_canonical_path("/Users/sbeardsley/dev/spore-core/");
    assert_eq!(a, b);
}

#[test]
fn workspace_id_windows_path_strips_drive_and_normalizes_sep() {
    let w = WorkspaceId::from_canonical_path("C:\\Users\\dev\\spore-core");
    assert!(w.as_str().starts_with("spore-core-"));
    // Distinct from the posix path (drive prefix stripped but hash differs
    // because the rest of the canonical string differs).
    let posix = WorkspaceId::from_canonical_path("/Users/sbeardsley/dev/spore-core");
    assert_ne!(w, posix);
}

#[tokio::test]
async fn workspace_id_derivation_fixture_replay() {
    let raw = std::fs::read_to_string(fixture_path("workspace_id_derivation.json"))
        .expect("workspace_id_derivation.json present");
    let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap();
    assert!(cases.len() >= 4, "fixture should carry several rows");
    for case in cases {
        let path = case["canonical_path"].as_str().unwrap();
        let expected = case["expected_workspace_id"].as_str().unwrap();
        let got = WorkspaceId::from_canonical_path(path);
        assert_eq!(got.as_str(), expected, "mismatch for path {path:?}");
    }
}

// ── R5: scope isolation — User and Project land in different backends ─────────

#[tokio::test]
async fn scoped_writes_isolated_per_scope() {
    let user = Arc::new(InMemoryStorageProvider::new());
    let project = Arc::new(InMemoryStorageProvider::new());
    let p = CompositeStorageProvider::new()
        .memory(StorageScope::User, user.clone())
        .memory(StorageScope::Project, project.clone())
        .build();

    p.memory()
        .append_memory(StorageScope::User, &sid("s"), mem("user", "U", "t1"))
        .await
        .unwrap();
    p.memory()
        .append_memory(StorageScope::Project, &sid("s"), mem("user", "P", "t1"))
        .await
        .unwrap();

    // Each backend physically holds only its own scope's entry.
    let u = user
        .get_memories(StorageScope::User, &sid("s"), 10)
        .await
        .unwrap();
    assert_eq!(
        u.iter().map(|e| e.content.clone()).collect::<Vec<_>>(),
        vec!["U"]
    );
    let pr = project
        .get_memories(StorageScope::Project, &sid("s"), 10)
        .await
        .unwrap();
    assert_eq!(
        pr.iter().map(|e| e.content.clone()).collect::<Vec<_>>(),
        vec!["P"]
    );

    // Scoped reads through the router return only own-scope entries.
    let ru = p
        .memory()
        .get_memories(StorageScope::User, &sid("s"), 10)
        .await
        .unwrap();
    assert_eq!(ru.len(), 1);
    assert_eq!(ru[0].content, "U");
    let rp = p
        .memory()
        .get_memories(StorageScope::Project, &sid("s"), 10)
        .await
        .unwrap();
    assert_eq!(rp.len(), 1);
    assert_eq!(rp[0].content, "P");
}

// ── R6: merged read = User ∪ Project, newest-first by timestamp, no dedup ─────

#[tokio::test]
async fn merged_read_unions_scopes_newest_first_no_dedup() {
    let p = CompositeStorageProvider::new()
        .memory(StorageScope::User, Arc::new(InMemoryStorageProvider::new()))
        .memory(
            StorageScope::Project,
            Arc::new(InMemoryStorageProvider::new()),
        )
        .build();

    // Identical-content "dup" entry in BOTH scopes (same timestamp) to prove no
    // dedup. Local entry must NOT appear in the merge.
    p.memory()
        .append_memory(
            StorageScope::User,
            &sid("s"),
            mem("user", "u-old", "2026-05-01T00:00:00Z"),
        )
        .await
        .unwrap();
    p.memory()
        .append_memory(
            StorageScope::User,
            &sid("s"),
            mem("user", "dup", "2026-05-03T00:00:00Z"),
        )
        .await
        .unwrap();
    p.memory()
        .append_memory(
            StorageScope::User,
            &sid("s"),
            mem("user", "u-new", "2026-05-05T00:00:00Z"),
        )
        .await
        .unwrap();
    p.memory()
        .append_memory(
            StorageScope::Project,
            &sid("s"),
            mem("a", "p-old", "2026-05-02T00:00:00Z"),
        )
        .await
        .unwrap();
    p.memory()
        .append_memory(
            StorageScope::Project,
            &sid("s"),
            mem("a", "dup", "2026-05-03T00:00:00Z"),
        )
        .await
        .unwrap();
    p.memory()
        .append_memory(
            StorageScope::Project,
            &sid("s"),
            mem("a", "p-new", "2026-05-06T00:00:00Z"),
        )
        .await
        .unwrap();

    let merged = p.get_memories_merged(&sid("s"), 10).await.unwrap();
    let contents: Vec<String> = merged.iter().map(|e| e.content.clone()).collect();
    assert_eq!(
        contents,
        vec!["p-new", "u-new", "dup", "dup", "p-old", "u-old"]
    );
    // No dedup: the identical-content "dup" entry is present twice.
    assert_eq!(contents.iter().filter(|c| *c == "dup").count(), 2);
}

#[tokio::test]
async fn merged_read_fixture_replay() {
    let raw = std::fs::read_to_string(fixture_path("memory_scoped_merge.json"))
        .expect("memory_scoped_merge.json present");
    let f: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let limit = f["limit"].as_u64().unwrap() as usize;

    let p = CompositeStorageProvider::new()
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

    for (key, scope) in [
        ("user", StorageScope::User),
        ("project", StorageScope::Project),
        ("local", StorageScope::Local),
    ] {
        for entry in f[key].as_array().unwrap() {
            let e: MemoryEntry = serde_json::from_value(entry.clone()).unwrap();
            p.memory().append_memory(scope, &sid("s"), e).await.unwrap();
        }
    }

    let merged = p.get_memories_merged(&sid("s"), limit).await.unwrap();
    let contents: Vec<String> = merged.iter().map(|e| e.content.clone()).collect();
    let expected: Vec<String> = f["expected_merged_contents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(contents, expected);
    // Local scope entries are excluded from the merge.
    assert!(!contents.iter().any(|c| c.contains("should-not-appear")));
}

// ── R7: unconfigured (memory, scope) → NoOp returns Ok(vec![]) ───────────────

#[tokio::test]
async fn unconfigured_memory_scope_falls_back_to_noop() {
    // Only User wired; Project + Local fall back to no-op.
    let p = CompositeStorageProvider::new()
        .memory(StorageScope::User, Arc::new(InMemoryStorageProvider::new()))
        .build();

    // Writes to an unconfigured scope silently no-op (Ok).
    p.memory()
        .append_memory(StorageScope::Project, &sid("s"), mem("user", "x", "t"))
        .await
        .unwrap();
    // Reads from an unconfigured scope return Ok(vec![]).
    assert!(p
        .memory()
        .get_memories(StorageScope::Project, &sid("s"), 10)
        .await
        .unwrap()
        .is_empty());
}

// ── R8: scoped read newest-first recency (append 4, limit=2 → newest two) ─────

#[tokio::test]
async fn scoped_read_recency_newest_first() {
    let p = CompositeStorageProvider::new()
        .memory(
            StorageScope::Project,
            Arc::new(InMemoryStorageProvider::new()),
        )
        .build();
    for (i, c) in ["m0", "m1", "m2", "m3"].iter().enumerate() {
        p.memory()
            .append_memory(
                StorageScope::Project,
                &sid("s"),
                mem("user", c, &format!("t{i}")),
            )
            .await
            .unwrap();
    }
    let got = p
        .memory()
        .get_memories(StorageScope::Project, &sid("s"), 2)
        .await
        .unwrap();
    assert_eq!(
        got.iter().map(|e| e.content.clone()).collect::<Vec<_>>(),
        vec!["m3", "m2"]
    );
}

// ── R11: Local falls back to NoOp when not wired ─────────────────────────────

#[tokio::test]
async fn local_scope_defaults_to_noop() {
    // Local intentionally not wired.
    let p = CompositeStorageProvider::new()
        .memory(StorageScope::User, Arc::new(InMemoryStorageProvider::new()))
        .memory(
            StorageScope::Project,
            Arc::new(InMemoryStorageProvider::new()),
        )
        .build();
    p.memory()
        .append_memory(StorageScope::Local, &sid("s"), mem("user", "l", "t"))
        .await
        .unwrap();
    assert!(p
        .memory()
        .get_memories(StorageScope::Local, &sid("s"), 10)
        .await
        .unwrap()
        .is_empty());
}

// ── R9: ToolContext exposes memory_store threaded by the registry ────────────

#[tokio::test]
async fn tool_context_exposes_threaded_memory_store() {
    use crate::tool_registry::mock::AllowAllSandbox;
    use crate::tool_registry::RealToolRegistry;
    use crate::tool_registry::StandardToolRegistry;

    // Wire a real memory backend through the registry and prove the seam is
    // live by writing through ToolContext's memory_store and reading it back.
    let memory: Arc<dyn MemoryStore> = Arc::new(InMemoryStorageProvider::new());
    let inner = Arc::new(StandardToolRegistry::new());
    let bridge = RealToolRegistry::new(
        inner,
        Arc::new(AllowAllSandbox),
        sid("ctx-test"),
        ProjectId::from_canonical_path("/ctx-test-project"),
        Arc::new(InMemoryStorageProvider::new()),
        memory.clone(),
    );
    let ctx = bridge.tool_context();
    ctx.memory_store()
        .append_memory(
            StorageScope::Project,
            ctx.session_id(),
            mem("user", "threaded", "t1"),
        )
        .await
        .unwrap();
    // Read back through the same Arc the registry threaded in.
    let got = memory
        .get_memories(StorageScope::Project, &sid("ctx-test"), 10)
        .await
        .unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].content, "threaded");
}

// ════════════════════════════════════════════════════════════════════════════
// #142 — ProjectId + project-scoped durable storage
// ════════════════════════════════════════════════════════════════════════════

// ── ProjectId pure derivation ────────────────────────────────────────────────

#[test]
fn project_id_is_deterministic_and_matches_workspace_id_algorithm() {
    // Same input → same id, repeatedly.
    let a = ProjectId::from_canonical_path("/Users/sbeardsley/dev/spore-core");
    let b = ProjectId::from_canonical_path("/Users/sbeardsley/dev/spore-core");
    assert_eq!(a, b);
    // The derivation is byte-identical to WorkspaceId (it delegates to the same
    // pure algorithm) — the cross-language anchor.
    let w = WorkspaceId::from_canonical_path("/Users/sbeardsley/dev/spore-core");
    assert_eq!(a.as_str(), w.as_str());
    // Form: `{sanitized_basename}-{8hex}`.
    assert!(a.as_str().starts_with("spore-core-"));
    assert_eq!(a.as_str().len(), "spore-core-".len() + 8);
}

#[test]
fn project_id_root_and_special_chars() {
    assert!(ProjectId::from_canonical_path("/")
        .as_str()
        .starts_with("root-"));
    let p = ProjectId::from_canonical_path("/Users/me/My Project (v2)!");
    assert!(p.as_str().starts_with("my-project-v2-"));
    assert!(!p.as_str().contains("--"));
}

#[test]
fn project_id_ignores_trailing_slash() {
    let a = ProjectId::from_canonical_path("/Users/sbeardsley/dev/spore-core");
    let b = ProjectId::from_canonical_path("/Users/sbeardsley/dev/spore-core/");
    assert_eq!(a, b);
}

// ── The `/a/b` vs `/a_b` collision-resolution case (the spec's headline) ─────

#[test]
fn project_id_resolves_slash_underscore_collision() {
    // A naive "slashes → underscores" slug would map BOTH of these to `a_b`.
    // The 8-hex SHA-256 suffix of the FULL canonical path keeps them distinct.
    let ab = ProjectId::from_canonical_path("/a/b");
    let a_b = ProjectId::from_canonical_path("/a_b");
    assert_ne!(ab, a_b, "/a/b and /a_b must derive DISTINCT project ids");
    // And the pinned exact values (cross-language anchor).
    assert_eq!(ab.as_str(), "b-662b7b62");
    assert_eq!(a_b.as_str(), "a-b-328ff01f");
}

#[test]
fn project_id_namespace_projects_onto_session_axis() {
    // `namespace()` yields a SessionId whose string IS the derived project id —
    // the namespace-reuse seam over the RunStore's `&SessionId` axis.
    let p = ProjectId::from_canonical_path("/work/audit-repo");
    assert_eq!(p.namespace().as_str(), p.as_str());
    assert_eq!(p.namespace().as_str(), "audit-repo-9e8ff6f3");
}

// ── FS-touching constructors: canonicalize FIRST ─────────────────────────────

#[tokio::test]
async fn project_id_from_path_canonicalizes_first() {
    // A real dir + a relative path INTO it must derive the same id as the
    // absolute canonical path — proving from_path canonicalizes before slugging.
    let tmp = tempfile::tempdir().unwrap();
    let nested = tmp.path().join("proj");
    std::fs::create_dir_all(&nested).unwrap();
    let canonical = std::fs::canonicalize(&nested).unwrap();

    let from_path = ProjectId::from_path(&nested).unwrap();
    let from_canon = ProjectId::from_canonical_path(&canonical.to_string_lossy());
    assert_eq!(from_path, from_canon);
}

#[tokio::test]
async fn project_id_from_path_resolves_relative_components() {
    // A path with `..` components canonicalizes to the same id as the direct one.
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a");
    let b = tmp.path().join("b");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    // {tmp}/a/../b  ==  {tmp}/b
    let via_dotdot = a.join("..").join("b");
    let direct = ProjectId::from_path(&b).unwrap();
    let dotted = ProjectId::from_path(&via_dotdot).unwrap();
    assert_eq!(direct, dotted);
}

#[cfg(unix)]
#[tokio::test]
async fn project_id_from_path_resolves_symlink() {
    // A symlink and its target derive the SAME id (symlink resolved by
    // canonicalize). Unix-only: symlink creation differs on Windows.
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("real-proj");
    std::fs::create_dir_all(&target).unwrap();
    let link = tmp.path().join("link-proj");
    std::os::unix::fs::symlink(&target, &link).unwrap();

    let via_target = ProjectId::from_path(&target).unwrap();
    let via_link = ProjectId::from_path(&link).unwrap();
    assert_eq!(
        via_target, via_link,
        "a symlink must resolve to its target's project id"
    );
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn project_id_from_path_macos_case_insensitive() {
    // macOS's default filesystem is case-insensitive: a dir created lowercase is
    // reachable via a different-cased path, and canonicalize returns the
    // ON-DISK casing — so both derive the same id. (#cfg-gated: this is a
    // filesystem-behavior-dependent assertion.)
    let tmp = tempfile::tempdir().unwrap();
    let lower = tmp.path().join("caseproj");
    std::fs::create_dir_all(&lower).unwrap();
    let upper = tmp.path().join("CASEPROJ");
    // On a case-insensitive FS, `upper` resolves to the same inode as `lower`.
    if std::fs::canonicalize(&upper).is_ok() {
        let a = ProjectId::from_path(&lower).unwrap();
        let b = ProjectId::from_path(&upper).unwrap();
        assert_eq!(a, b);
    }
}

#[tokio::test]
async fn project_id_from_path_errors_on_missing_path() {
    // A non-existent path cannot be canonicalized ⇒ Canonicalize error variant.
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("does-not-exist");
    let err = ProjectId::from_path(&missing).unwrap_err();
    assert!(matches!(err, ProjectIdError::Canonicalize(_)));
    // Display does not panic.
    let _ = err.to_string();
}

#[test]
fn project_id_from_cwd_succeeds() {
    // The process cwd exists, so from_cwd derives an id without error.
    let p = ProjectId::from_cwd().expect("cwd canonicalizes");
    assert!(!p.as_str().is_empty());
}

// ── ProjectId derivation fixture replay (cross-language anchor) ───────────────

#[test]
fn project_id_derivation_fixture_replay() {
    let raw = std::fs::read_to_string(fixture_path("project_id_derivation.json"))
        .expect("project_id_derivation.json present");
    let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap();
    assert!(cases.len() >= 6, "fixture should carry the collision rows");
    let mut saw_collision_a = false;
    let mut saw_collision_b = false;
    for case in cases {
        let path = case["canonical_path"].as_str().unwrap();
        let expected = case["expected_project_id"].as_str().unwrap();
        let got = ProjectId::from_canonical_path(path);
        assert_eq!(got.as_str(), expected, "mismatch for path {path:?}");
        if path == "/a/b" {
            saw_collision_a = true;
        }
        if path == "/a_b" {
            saw_collision_b = true;
        }
    }
    assert!(
        saw_collision_a && saw_collision_b,
        "fixture must pin both /a/b and /a_b"
    );
}

// ── task_list visible across DIFFERENT sessions with SAME project_id ──────────

#[tokio::test]
async fn run_store_value_visible_across_sessions_same_project() {
    // The durable seam: write under one session's view but keyed by the project
    // namespace; a DIFFERENT session reading the SAME project namespace sees it.
    let store: Arc<dyn RunStore> = Arc::new(InMemoryStorageProvider::new());
    let project = ProjectId::from_canonical_path("/work/repo");

    store
        .put(&project.namespace(), "task_list", json!({"tasks": [1, 2]}))
        .await
        .unwrap();

    // A fresh session id (mirrors SessionId::generate() per Ralph window) does
    // NOT change what the project namespace returns.
    let _fresh_session = SessionId::generate();
    let got = store.get(&project.namespace(), "task_list").await.unwrap();
    assert_eq!(got, Some(json!({"tasks": [1, 2]})));
}

// ── AC5: cross-window AND cross-process durability via FileSystemStorageProvider

#[tokio::test]
async fn project_durable_survives_window_reset_and_process_restart() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let project = ProjectId::from_canonical_path("/work/audit-repo");

    // Window 1 (session A): write the durable task_list under the project ns.
    let list = json!({
        "tasks": [{"id": 1, "description": "discover", "status": "completed", "blockers": []}],
        "next_id": 2
    });
    {
        let provider: Arc<dyn RunStore> = Arc::new(FileSystemStorageProvider::new(root.clone()));
        provider
            .put(&project.namespace(), "task_list", list.clone())
            .await
            .unwrap();
    }

    // Window 2 (a DIFFERENT, freshly generated session) over the SAME provider
    // root reads window 1's list — cross-window survival.
    {
        let provider: Arc<dyn RunStore> = Arc::new(FileSystemStorageProvider::new(root.clone()));
        let _window_2_session = SessionId::generate();
        let got = provider
            .get(&project.namespace(), "task_list")
            .await
            .unwrap();
        assert_eq!(got, Some(list.clone()), "window 2 must see window 1's list");
    }

    // A BRAND-NEW FileSystemStorageProvider over the same on-disk root (a fresh
    // process) reads the same bytes — cross-process durability.
    {
        let fresh: Arc<dyn RunStore> = Arc::new(FileSystemStorageProvider::new(root.clone()));
        let got = fresh.get(&project.namespace(), "task_list").await.unwrap();
        assert_eq!(
            got,
            Some(list),
            "a fresh provider must read the durable list"
        );
    }
}

#[tokio::test]
async fn project_durable_survival_fixture_replay() {
    let raw = std::fs::read_to_string(fixture_path("project_durable_survival.json"))
        .expect("project_durable_survival.json present");
    let fx: serde_json::Value = serde_json::from_str(&raw).unwrap();

    let project_path = fx["project_canonical_path"].as_str().unwrap();
    let project = ProjectId::from_canonical_path(project_path);
    // The pinned project id must match.
    assert_eq!(
        project.as_str(),
        fx["expected_project_id"].as_str().unwrap()
    );
    let run_key = fx["run_key"].as_str().unwrap();

    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();

    // Window 1 writes the fixture's task_list (under a DISTINCT session id).
    let w1_list = fx["window_1"]["task_list"].clone();
    {
        let provider: Arc<dyn RunStore> = Arc::new(FileSystemStorageProvider::new(root.clone()));
        let _session_a = SessionId::new(fx["window_1"]["session_id"].as_str().unwrap());
        provider
            .put(&project.namespace(), run_key, w1_list.clone())
            .await
            .unwrap();
    }

    // Window 2 (a different session id) reads the expected list cross-window.
    {
        let provider: Arc<dyn RunStore> = Arc::new(FileSystemStorageProvider::new(root.clone()));
        let _session_b = SessionId::new(fx["window_2"]["session_id"].as_str().unwrap());
        let got = provider
            .get(&project.namespace(), run_key)
            .await
            .unwrap()
            .expect("window 2 sees the list");
        assert_eq!(got, fx["window_2"]["expected_task_list"]);
    }

    // A fresh provider (cross-process) reads the expected list.
    {
        let fresh: Arc<dyn RunStore> = Arc::new(FileSystemStorageProvider::new(root));
        let got = fresh
            .get(&project.namespace(), run_key)
            .await
            .unwrap()
            .expect("fresh provider sees the list");
        assert_eq!(got, fx["cross_process"]["expected_task_list"]);
    }
}

// ── Active-run lifecycle: new / resume / complete ────────────────────────────

fn run_store() -> Arc<dyn RunStore> {
    Arc::new(InMemoryStorageProvider::new())
}

#[tokio::test]
async fn active_run_starts_new_when_no_slot() {
    let store = run_store();
    let project = ProjectId::from_canonical_path("/work/p");
    assert!(load_active_run(&store, &project).await.unwrap().is_none());

    let decision =
        start_or_resume_active_run(&store, &project, "tag-1", ts("2026-06-12T00:00:00Z"))
            .await
            .unwrap();
    assert_eq!(decision, ActiveRunDecision::StartedNew);

    let slot = load_active_run(&store, &project).await.unwrap().unwrap();
    assert_eq!(slot.run_tag, "tag-1");
    assert_eq!(slot.status, ActiveRunStatus::Active);
    // started_at is the INJECTED timestamp (deterministic, no clock).
    assert_eq!(slot.started_at, ts("2026-06-12T00:00:00Z"));
}

#[tokio::test]
async fn active_run_resumes_on_matching_tag() {
    let store = run_store();
    let project = ProjectId::from_canonical_path("/work/p");
    start_or_resume_active_run(&store, &project, "tag-1", ts("t1"))
        .await
        .unwrap();

    // Same tag ⇒ resume; the original started_at is preserved (slot untouched).
    let decision = start_or_resume_active_run(&store, &project, "tag-1", ts("t2"))
        .await
        .unwrap();
    assert_eq!(decision, ActiveRunDecision::Resumed);
    let slot = load_active_run(&store, &project).await.unwrap().unwrap();
    assert_eq!(slot.started_at, ts("t1"), "resume must not restamp");
}

#[tokio::test]
async fn active_run_starts_fresh_on_different_tag() {
    let store = run_store();
    let project = ProjectId::from_canonical_path("/work/p");
    start_or_resume_active_run(&store, &project, "tag-1", ts("t1"))
        .await
        .unwrap();

    // A DIFFERENT tag is a genuinely new job in the same repo ⇒ start fresh.
    let decision = start_or_resume_active_run(&store, &project, "tag-2", ts("t2"))
        .await
        .unwrap();
    assert_eq!(decision, ActiveRunDecision::StartedNew);
    let slot = load_active_run(&store, &project).await.unwrap().unwrap();
    assert_eq!(slot.run_tag, "tag-2");
    assert_eq!(slot.started_at, ts("t2"));
}

#[tokio::test]
async fn active_run_complete_then_restart_is_fresh() {
    let store = run_store();
    let project = ProjectId::from_canonical_path("/work/p");
    start_or_resume_active_run(&store, &project, "tag-1", ts("t1"))
        .await
        .unwrap();

    complete_active_run(&store, &project).await.unwrap();
    let slot = load_active_run(&store, &project).await.unwrap().unwrap();
    assert_eq!(slot.status, ActiveRunStatus::Completed);

    // After completion, even the SAME tag starts a fresh Active slot.
    let decision = start_or_resume_active_run(&store, &project, "tag-1", ts("t3"))
        .await
        .unwrap();
    assert_eq!(decision, ActiveRunDecision::StartedNew);
    let slot = load_active_run(&store, &project).await.unwrap().unwrap();
    assert_eq!(slot.status, ActiveRunStatus::Active);
    assert_eq!(slot.started_at, ts("t3"));
}

#[tokio::test]
async fn complete_active_run_is_noop_without_slot() {
    let store = run_store();
    let project = ProjectId::from_canonical_path("/work/empty");
    // No active slot ⇒ completing is a silent no-op (no error).
    complete_active_run(&store, &project).await.unwrap();
    assert!(load_active_run(&store, &project).await.unwrap().is_none());
}

#[tokio::test]
async fn active_run_isolated_per_project() {
    // Two projects over the SAME store keep independent active slots.
    let store = run_store();
    let p1 = ProjectId::from_canonical_path("/work/one");
    let p2 = ProjectId::from_canonical_path("/work/two");
    start_or_resume_active_run(&store, &p1, "a", ts("t1"))
        .await
        .unwrap();
    start_or_resume_active_run(&store, &p2, "b", ts("t1"))
        .await
        .unwrap();
    assert_eq!(
        load_active_run(&store, &p1).await.unwrap().unwrap().run_tag,
        "a"
    );
    assert_eq!(
        load_active_run(&store, &p2).await.unwrap().unwrap().run_tag,
        "b"
    );
}
