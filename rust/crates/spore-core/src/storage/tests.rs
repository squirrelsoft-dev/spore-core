//! Inline unit + fixture-replay tests for the [`StorageProvider`] abstraction
//! (issue #73). Covers every pinned rule: no-op fallback, composite per-domain
//! routing, single-provider-fills-all-slots, OTLP parse table, atomic write,
//! append ordering, get_memories recency, run-store roundtrip, session-store
//! roundtrip, flush markers, and cross-language fixture replay.

use super::*;
use crate::harness::{
    BudgetSnapshot, HumanRequest, LoopStrategy, PausedState, ReactConfig, RiskLevel, SessionId,
    SessionState, Task, TaskId,
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
