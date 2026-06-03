//! `MarkdownMemoryProvider` — a `MemoryStore` that persists agent memory to a
//! single human-readable `memory.md` file on disk.
//!
//! ## What this demonstrates
//! The storage seam. The harness is **stateless** — every byte of durable state
//! lives behind a `StorageProvider`. Memory is one of its four domains
//! (`MemoryStore`). This module implements *only* that domain and composes it
//! with [`NoOpStorageProvider`] for the other three (session / run /
//! observability). That composed provider is what `main.rs` hands to
//! `HarnessBuilder::storage(...)`; the harness then threads `storage.memory()`
//! into the built-in `memory` tool's `ToolContext` on every run. No custom
//! harness plumbing — the seam is the whole integration surface.
//!
//! ## The seam
//! `MemoryStore` is the swap point. The built-in `FileSystemStorageProvider`
//! persists memory as a JSONL log; this provider persists the *same*
//! [`MemoryEntry`] values to readable markdown instead. Same trait, same
//! agent, same tool — different on-disk shape. Anything implementing
//! `MemoryStore` slots in here.
//!
//! ## On-disk format (round-trips exactly)
//! Each [`MemoryEntry`] is one markdown block. The header line carries the
//! round-trip fields; the body is the content:
//!
//! ```text
//! ## [project] 2026-06-02T12:00:00Z — assistant
//!
//! Postgres 15 is the system of record.
//! ```
//!
//! `append_memory` writes such a block; `get_memories` parses them back,
//! filters by scope, sorts newest-first by timestamp, and takes `limit`. A
//! hand-edited file (extra prose, blank lines, reordered blocks) is tolerated:
//! anything that is not a recognized `## [scope] [session] timestamp — role` header is
//! treated as body for the preceding entry, and leading prose before the first
//! header is ignored.
//!
//! ## Pinned-session-id requirement (read this)
//! Memory is keyed by `SessionId`. The `memory` tool always uses
//! `ctx.session_id()`. For Run 2 (recall) to read Run 1's (store) memories,
//! **both runs MUST use the SAME `SessionId`** — see `main.rs`, which pins
//! `SessionId::new("project-ironwood")` rather than `SessionId::generate()`.
//! This provider also stores the session id in each header so a single
//! `memory.md` can hold multiple sessions without cross-talk.
//!
//! There are no `// SPEC QUESTION:` markers in this file.

use std::path::PathBuf;
use std::sync::Mutex;

use spore_core::harness::BoxFut;
use spore_core::{
    MemoryEntry, MemoryStore, NoOpStorageProvider, SessionId, StorageError, StorageProvider,
    StorageScope, Timestamp,
};

/// A `MemoryStore` backed by a single human-readable `memory.md` file.
///
/// The `Mutex` serializes the read-modify-write of `append_memory` so
/// concurrent appends from the harness (which dispatches the `memory` tool
/// sequentially anyway) never interleave a partial write.
pub struct MarkdownMemoryProvider {
    path: PathBuf,
    write_lock: Mutex<()>,
}

impl MarkdownMemoryProvider {
    /// Create a provider over `memory.md` at `path`. The file is created lazily
    /// on the first `append_memory`; a missing file reads as empty.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            write_lock: Mutex::new(()),
        }
    }

    /// Compose this provider into a full [`StorageProvider`]: the real
    /// `MemoryStore` for the memory domain, [`NoOpStorageProvider`] for the
    /// other three. This is exactly what the example hands to the harness.
    pub fn into_storage_provider(self) -> StorageProvider {
        StorageProvider::new(
            std::sync::Arc::new(NoOpStorageProvider),
            std::sync::Arc::new(self),
            std::sync::Arc::new(NoOpStorageProvider),
            std::sync::Arc::new(NoOpStorageProvider),
        )
    }
}

/// The marker for a scope: how `StorageScope` is spelled in a header line.
fn scope_token(scope: StorageScope) -> &'static str {
    match scope {
        StorageScope::User => "user",
        StorageScope::Project => "project",
        StorageScope::Local => "local",
    }
}

fn parse_scope_token(s: &str) -> Option<StorageScope> {
    match s {
        "user" => Some(StorageScope::User),
        "project" => Some(StorageScope::Project),
        "local" => Some(StorageScope::Local),
        _ => None,
    }
}

/// Render one entry as a markdown block (header line + blank line + body).
/// `session_id` is encoded so one file can hold multiple sessions.
fn render_block(scope: StorageScope, session_id: &SessionId, entry: &MemoryEntry) -> String {
    format!(
        "## [{scope}] [{session}] {ts} — {role}\n\n{content}\n",
        scope = scope_token(scope),
        session = session_id.as_str(),
        ts = entry.timestamp.as_str(),
        role = entry.role,
        content = entry.content.trim_end(),
    )
}

/// A parsed entry plus the scope + session it was filed under.
struct ParsedBlock {
    scope: StorageScope,
    session: String,
    entry: MemoryEntry,
}

/// Parse a header line of the form `## [scope] [session] timestamp — role`.
/// Returns `None` for any line that is not a recognized header (so prose and
/// hand-edits are tolerated). The em-dash separator is ` — ` (space, U+2014,
/// space).
fn parse_header(line: &str) -> Option<(StorageScope, String, String, String)> {
    let rest = line.strip_prefix("## ")?;
    let rest = rest.strip_prefix('[')?;
    let (scope_str, rest) = rest.split_once("] ")?;
    let scope = parse_scope_token(scope_str.trim())?;
    let rest = rest.strip_prefix('[')?;
    let (session, rest) = rest.split_once("] ")?;
    let (ts, role) = rest.split_once(" — ")?;
    let ts = ts.trim();
    let role = role.trim();
    if ts.is_empty() || role.is_empty() {
        return None;
    }
    Some((
        scope,
        session.trim().to_string(),
        ts.to_string(),
        role.to_string(),
    ))
}

/// Parse the whole file into blocks. Body lines accumulate under the most
/// recent header; text before the first header is discarded.
fn parse_file(contents: &str) -> Vec<ParsedBlock> {
    let mut blocks: Vec<ParsedBlock> = Vec::new();
    let mut current: Option<(StorageScope, String, String, String, Vec<String>)> = None;

    for line in contents.lines() {
        if let Some((scope, session, ts, role)) = parse_header(line) {
            if let Some((s, sess, t, r, body)) = current.take() {
                blocks.push(finish_block(s, sess, t, r, body));
            }
            current = Some((scope, session, ts, role, Vec::new()));
        } else if line.starts_with("## ") {
            // A `## ` line that is NOT a valid entry header (e.g. a human-added
            // subheading) terminates the current block and is otherwise ignored,
            // so hand-edited headings never pollute an entry's content.
            if let Some((s, sess, t, r, body)) = current.take() {
                blocks.push(finish_block(s, sess, t, r, body));
            }
        } else if let Some((_, _, _, _, body)) = current.as_mut() {
            body.push(line.to_string());
        }
        // else: prose before the first header — ignored.
    }
    if let Some((s, sess, t, r, body)) = current.take() {
        blocks.push(finish_block(s, sess, t, r, body));
    }
    blocks
}

fn finish_block(
    scope: StorageScope,
    session: String,
    ts: String,
    role: String,
    body: Vec<String>,
) -> ParsedBlock {
    // Trim the leading/trailing blank lines that bracket the body, then rejoin.
    let content = body.join("\n").trim().to_string();
    ParsedBlock {
        scope,
        session,
        entry: MemoryEntry::new(role, content, Timestamp::new(ts)),
    }
}

impl MemoryStore for MarkdownMemoryProvider {
    fn append_memory<'a>(
        &'a self,
        scope: StorageScope,
        session_id: &'a SessionId,
        entry: MemoryEntry,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            let _guard = self.write_lock.lock().map_err(|e| StorageError::Backend {
                message: e.to_string(),
            })?;

            // Read-modify-write: load existing text, append a new block.
            let mut existing = match std::fs::read_to_string(&self.path) {
                Ok(s) => s,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    "# Agent Memory\n\nHuman-readable working memory for this agent. \
                     Each `##` block below is one remembered entry.\n"
                        .to_string()
                }
                Err(e) => return Err(StorageError::Io(e)),
            };
            if !existing.ends_with('\n') {
                existing.push('\n');
            }
            existing.push('\n');
            existing.push_str(&render_block(scope, session_id, &entry));

            if let Some(parent) = self.path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(StorageError::Io)?;
                }
            }
            std::fs::write(&self.path, existing).map_err(StorageError::Io)?;
            Ok(())
        })
    }

    fn get_memories<'a>(
        &'a self,
        scope: StorageScope,
        session_id: &'a SessionId,
        limit: usize,
    ) -> BoxFut<'a, Result<Vec<MemoryEntry>, StorageError>> {
        Box::pin(async move {
            let contents = match std::fs::read_to_string(&self.path) {
                Ok(s) => s,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
                Err(e) => return Err(StorageError::Io(e)),
            };

            let mut entries: Vec<MemoryEntry> = parse_file(&contents)
                .into_iter()
                .filter(|b| b.scope == scope && b.session == session_id.as_str())
                .map(|b| b.entry)
                .collect();

            // Newest-first by timestamp. RFC-3339 strings sort lexically; ties
            // keep insertion order (stable sort) which mirrors append order.
            entries.sort_by(|a, b| b.timestamp.as_str().cmp(a.timestamp.as_str()));
            entries.truncate(limit);
            Ok(entries)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let unique = format!(
            "spore07-{tag}-{}-{:?}.md",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        p.push(unique);
        p
    }

    fn sid() -> SessionId {
        SessionId::new("s1")
    }

    fn entry(role: &str, content: &str, ts: &str) -> MemoryEntry {
        MemoryEntry::new(role, content, Timestamp::new(ts))
    }

    #[tokio::test]
    async fn missing_file_reads_empty() {
        let p = temp_path("missing");
        let provider = MarkdownMemoryProvider::new(&p);
        let got = provider
            .get_memories(StorageScope::Project, &sid(), 50)
            .await
            .unwrap();
        assert!(got.is_empty());
    }

    #[tokio::test]
    async fn append_then_get_roundtrips_the_entry() {
        let p = temp_path("roundtrip");
        let provider = MarkdownMemoryProvider::new(&p);
        let e = entry(
            "assistant",
            "Postgres is the system of record.",
            "2026-06-02T10:00:00Z",
        );
        provider
            .append_memory(StorageScope::Project, &sid(), e.clone())
            .await
            .unwrap();

        let got = provider
            .get_memories(StorageScope::Project, &sid(), 50)
            .await
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].role, "assistant");
        assert_eq!(got[0].content, "Postgres is the system of record.");
        assert_eq!(got[0].timestamp.as_str(), "2026-06-02T10:00:00Z");
        // The artifact is real, readable markdown on disk.
        let raw = std::fs::read_to_string(&p).unwrap();
        assert!(raw.contains("## [project] [s1] 2026-06-02T10:00:00Z — assistant"));
        assert!(raw.contains("Postgres is the system of record."));
        let _ = std::fs::remove_file(&p);
    }

    #[tokio::test]
    async fn multiline_content_roundtrips() {
        let p = temp_path("multiline");
        let provider = MarkdownMemoryProvider::new(&p);
        let e = entry(
            "assistant",
            "line one\nline two\n\nline four",
            "2026-06-02T10:00:00Z",
        );
        provider
            .append_memory(StorageScope::Project, &sid(), e)
            .await
            .unwrap();
        let got = provider
            .get_memories(StorageScope::Project, &sid(), 50)
            .await
            .unwrap();
        assert_eq!(got[0].content, "line one\nline two\n\nline four");
        let _ = std::fs::remove_file(&p);
    }

    #[tokio::test]
    async fn scope_filtering_isolates_scopes() {
        let p = temp_path("scope");
        let provider = MarkdownMemoryProvider::new(&p);
        provider
            .append_memory(
                StorageScope::Project,
                &sid(),
                entry("user", "proj", "2026-06-02T10:00:00Z"),
            )
            .await
            .unwrap();
        provider
            .append_memory(
                StorageScope::User,
                &sid(),
                entry("user", "usr", "2026-06-02T10:00:01Z"),
            )
            .await
            .unwrap();

        let proj = provider
            .get_memories(StorageScope::Project, &sid(), 50)
            .await
            .unwrap();
        assert_eq!(proj.len(), 1);
        assert_eq!(proj[0].content, "proj");

        let usr = provider
            .get_memories(StorageScope::User, &sid(), 50)
            .await
            .unwrap();
        assert_eq!(usr.len(), 1);
        assert_eq!(usr[0].content, "usr");
        let _ = std::fs::remove_file(&p);
    }

    #[tokio::test]
    async fn session_filtering_isolates_sessions() {
        let p = temp_path("session");
        let provider = MarkdownMemoryProvider::new(&p);
        let a = SessionId::new("alpha");
        let b = SessionId::new("beta");
        provider
            .append_memory(
                StorageScope::Project,
                &a,
                entry("user", "from-alpha", "2026-06-02T10:00:00Z"),
            )
            .await
            .unwrap();
        provider
            .append_memory(
                StorageScope::Project,
                &b,
                entry("user", "from-beta", "2026-06-02T10:00:01Z"),
            )
            .await
            .unwrap();

        let got_a = provider
            .get_memories(StorageScope::Project, &a, 50)
            .await
            .unwrap();
        assert_eq!(got_a.len(), 1);
        assert_eq!(got_a[0].content, "from-alpha");
        let _ = std::fs::remove_file(&p);
    }

    #[tokio::test]
    async fn get_returns_newest_first() {
        let p = temp_path("order");
        let provider = MarkdownMemoryProvider::new(&p);
        // Appended out of timestamp order on purpose.
        provider
            .append_memory(
                StorageScope::Project,
                &sid(),
                entry("user", "middle", "2026-06-02T11:00:00Z"),
            )
            .await
            .unwrap();
        provider
            .append_memory(
                StorageScope::Project,
                &sid(),
                entry("user", "oldest", "2026-06-02T10:00:00Z"),
            )
            .await
            .unwrap();
        provider
            .append_memory(
                StorageScope::Project,
                &sid(),
                entry("user", "newest", "2026-06-02T12:00:00Z"),
            )
            .await
            .unwrap();

        let got = provider
            .get_memories(StorageScope::Project, &sid(), 50)
            .await
            .unwrap();
        let contents: Vec<&str> = got.iter().map(|e| e.content.as_str()).collect();
        assert_eq!(contents, vec!["newest", "middle", "oldest"]);
        let _ = std::fs::remove_file(&p);
    }

    #[tokio::test]
    async fn limit_takes_most_recent() {
        let p = temp_path("limit");
        let provider = MarkdownMemoryProvider::new(&p);
        for i in 0..5 {
            provider
                .append_memory(
                    StorageScope::Project,
                    &sid(),
                    entry("user", &format!("e{i}"), &format!("2026-06-02T10:00:0{i}Z")),
                )
                .await
                .unwrap();
        }
        let got = provider
            .get_memories(StorageScope::Project, &sid(), 2)
            .await
            .unwrap();
        let contents: Vec<&str> = got.iter().map(|e| e.content.as_str()).collect();
        assert_eq!(contents, vec!["e4", "e3"]);
        let _ = std::fs::remove_file(&p);
    }

    #[tokio::test]
    async fn tolerates_hand_edited_file() {
        let p = temp_path("handedit");
        // A file a human authored/edited: prose before the first header, an
        // extra heading, blank lines, and a normal entry block.
        let hand = "# My Notes\n\nSome rambling prose that is not an entry.\n\n\
                    ## [project] [s1] 2026-06-02T09:00:00Z — user\n\n\
                    Hand-written fact about Ironwood.\n\n\
                    ## A non-entry heading the human added\n\n\
                    more prose\n";
        std::fs::write(&p, hand).unwrap();
        let provider = MarkdownMemoryProvider::new(&p);
        let got = provider
            .get_memories(StorageScope::Project, &sid(), 50)
            .await
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].content, "Hand-written fact about Ironwood.");

        // And we can still append on top of the hand-edited file.
        provider
            .append_memory(
                StorageScope::Project,
                &sid(),
                entry("assistant", "appended", "2026-06-02T10:00:00Z"),
            )
            .await
            .unwrap();
        let got = provider
            .get_memories(StorageScope::Project, &sid(), 50)
            .await
            .unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].content, "appended"); // newest-first
        let _ = std::fs::remove_file(&p);
    }

    #[tokio::test]
    async fn composes_into_storage_provider_memory_slot() {
        let p = temp_path("compose");
        let provider = MarkdownMemoryProvider::new(&p);
        let storage = provider.into_storage_provider();
        // The memory slot is the markdown provider; round-trips through it.
        storage
            .memory()
            .append_memory(
                StorageScope::Project,
                &sid(),
                entry("user", "via-seam", "2026-06-02T10:00:00Z"),
            )
            .await
            .unwrap();
        let got = storage
            .memory()
            .get_memories(StorageScope::Project, &sid(), 50)
            .await
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].content, "via-seam");
        // The other three domains are no-ops: a run read returns nothing.
        assert!(storage.run().get(&sid(), "k").await.unwrap().is_none());
        let _ = std::fs::remove_file(&p);
    }
}
