//! Issue #73 — `StorageProvider`: a pluggable, per-domain persistence layer.
//!
//! This is the **reference implementation**. TypeScript, Python, and Go derive
//! from it. Every cross-language rule is pinned by the issue spec (and its
//! "Spec resolutions for implementation (v1)" comment); there are NO
//! `// SPEC QUESTION:` markers — everything is resolved.
//!
//! ## Types
//!   - [`StorageError`] — `#[non_exhaustive]` thiserror enum:
//!     `Io` (`#[from] std::io::Error`), `Serialization(String)`,
//!     `NotFound { domain, key }`, `Backend { message }`.
//!   - [`MemoryEntry`] — `{ role, content, timestamp, metadata (default {}) }`,
//!     serializable and byte-identical cross-language.
//!   - [`StorageProvider`] — a struct of four `Arc<dyn _>` domain stores
//!     (`session`, `memory`, `run`, `observability`) with `.session()` /
//!     `.memory()` / `.run()` / `.observability()` accessors and a
//!     [`StorageProvider::single`] convenience constructor that clones one
//!     concrete provider into all four slots.
//!   - Providers: [`NoOpStorageProvider`], [`InMemoryStorageProvider`],
//!     [`FileSystemStorageProvider`], [`CompositeStorageProvider`].
//!
//! ## Domain trait method sets (all `Send + Sync`, dyn-compatible via `BoxFut`)
//!   - [`SessionStore`]: `get_session`, `put_session`, `delete_session`,
//!     `list_sessions`.
//!   - [`MemoryStore`]: `append_memory`, `get_memories(limit)` — returns the
//!     **most-recent N, newest-first** (recency semantics).
//!   - [`RunStore`]: `get`, `put`, `delete`, `list_keys` — values are opaque
//!     JSON blobs; the store never knows the schema.
//!   - [`ObservabilityStore`]: `append_span`, `get_spans`, `get_sessions`,
//!     `flush_session` — append-only span storage.
//!
//! ## Rules enforced
//!   - **No-op fallback.** Unconfigured domains fall back to
//!     [`NoOpStorageProvider`]; the harness never null-checks — it always calls
//!     the store and the store decides. No-op reads return `Ok(None)` /
//!     `Ok(vec![])`; writes return `Ok(())`.
//!   - **Single-provider-fills-all-slots.** [`StorageProvider::single`] clones
//!     one `Arc<P: SessionStore + MemoryStore + RunStore + ObservabilityStore>`
//!     into all four slots.
//!   - **Composite per-domain routing.** [`CompositeStorageProvider`] holds an
//!     `Option<Arc<dyn _>>` per domain; `.build()` fills each `None` slot with
//!     a `NoOpStorageProvider`.
//!   - **Atomic write-rename.** [`FileSystemStorageProvider`] non-append writes
//!     ensure the parent dir, write full bytes to a sibling `{target}.tmp`,
//!     flush + fsync, then `rename(tmp, target)`. No leftover `.tmp` on success.
//!     Append writes (memory / observability JSONL) append + flush. Layout:
//!     `{root}/sessions/{id}/state.json` (session),
//!     `{root}/sessions/{id}/run/{key}.json` (run),
//!     `{root}/sessions/{id}/memory.jsonl` (memory, append),
//!     `{root}/sessions/{id}/trace.jsonl` (observability, append).
//!     `flush_session` creates a sibling `.flushed` marker.
//!   - **`get_memories` recency.** Reads the JSONL and returns the most-recent
//!     `limit` entries, newest-first.
//!   - **Last-writer-wins** for FS non-append writes via rename; no per-key
//!     locking contract — atomic rename is the only durability guarantee.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::guide_registry::SessionOutcome;
use crate::harness::{BoxFut, PausedState, SessionId};
use crate::memory::Timestamp;
use crate::observability::SessionMetrics;

// ============================================================================
// StorageError
// ============================================================================

/// Errors surfaced by the storage domain traits. `#[non_exhaustive]` so new
/// variants can be added without breaking downstream matches.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StorageError {
    /// An I/O error from a filesystem-backed store.
    #[error("storage I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A (de)serialization error. Carries the underlying message as a `String`
    /// so the error type stays `Send + Sync` and language-portable.
    #[error("storage serialization error: {0}")]
    Serialization(String),
    /// A keyed lookup found nothing where the caller required a value.
    #[error("storage not found: domain={domain} key={key}")]
    NotFound { domain: String, key: String },
    /// A backend-specific failure that does not map to the variants above.
    #[error("storage backend error: {message}")]
    Backend { message: String },
}

// ============================================================================
// MemoryEntry
// ============================================================================

/// One episodic memory entry. Byte-identical cross-language: `{ role, content,
/// timestamp, metadata }` where `metadata` defaults to an empty object `{}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub role: String,
    pub content: String,
    pub timestamp: Timestamp,
    /// Free-form metadata; defaults to an empty JSON object `{}`.
    #[serde(default = "empty_object")]
    pub metadata: JsonValue,
}

fn empty_object() -> JsonValue {
    JsonValue::Object(serde_json::Map::new())
}

impl MemoryEntry {
    /// Build an entry with an empty `metadata` object.
    pub fn new(role: impl Into<String>, content: impl Into<String>, timestamp: Timestamp) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            timestamp,
            metadata: empty_object(),
        }
    }
}

// ============================================================================
// Domain traits
// ============================================================================

/// Pause/resume lifecycle store. Stores [`PausedState`] keyed by [`SessionId`].
pub trait SessionStore: Send + Sync {
    fn get_session<'a>(
        &'a self,
        id: &'a SessionId,
    ) -> BoxFut<'a, Result<Option<PausedState>, StorageError>>;
    fn put_session<'a>(
        &'a self,
        id: &'a SessionId,
        state: &'a PausedState,
    ) -> BoxFut<'a, Result<(), StorageError>>;
    fn delete_session<'a>(&'a self, id: &'a SessionId) -> BoxFut<'a, Result<(), StorageError>>;
    fn list_sessions(&self) -> BoxFut<'_, Result<Vec<SessionId>, StorageError>>;
}

/// Episodic memory store. Append-only log per session.
pub trait MemoryStore: Send + Sync {
    fn append_memory<'a>(
        &'a self,
        session_id: &'a SessionId,
        entry: MemoryEntry,
    ) -> BoxFut<'a, Result<(), StorageError>>;
    /// Returns the **most-recent `limit` entries, newest-first**.
    fn get_memories<'a>(
        &'a self,
        session_id: &'a SessionId,
        limit: usize,
    ) -> BoxFut<'a, Result<Vec<MemoryEntry>, StorageError>>;
}

/// Per-run structured state keyed by `(SessionId, key)`. Values are opaque JSON
/// blobs — the store does not know the schema; callers own serialization.
pub trait RunStore: Send + Sync {
    fn get<'a>(
        &'a self,
        session_id: &'a SessionId,
        key: &'a str,
    ) -> BoxFut<'a, Result<Option<JsonValue>, StorageError>>;
    fn put<'a>(
        &'a self,
        session_id: &'a SessionId,
        key: &'a str,
        value: JsonValue,
    ) -> BoxFut<'a, Result<(), StorageError>>;
    fn delete<'a>(
        &'a self,
        session_id: &'a SessionId,
        key: &'a str,
    ) -> BoxFut<'a, Result<(), StorageError>>;
    fn list_keys<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<Vec<String>, StorageError>>;
}

/// Append-only span storage. Distinct from the other three: no get-by-key,
/// queried by session and time range.
pub trait ObservabilityStore: Send + Sync {
    fn append_span<'a>(
        &'a self,
        session_id: &'a SessionId,
        span: JsonValue,
    ) -> BoxFut<'a, Result<(), StorageError>>;
    fn get_spans<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<Vec<JsonValue>, StorageError>>;
    fn get_sessions<'a>(
        &'a self,
        since: Timestamp,
        domain: Option<String>,
        outcome: Option<SessionOutcome>,
    ) -> BoxFut<'a, Result<Vec<SessionMetrics>, StorageError>>;
    fn flush_session<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<(), StorageError>>;
}

// ============================================================================
// StorageProvider
// ============================================================================

/// A composed persistence layer: four independent domain stores behind
/// `Arc<dyn _>`. Built either from a single backend (cloned into all four
/// slots via [`StorageProvider::single`]) or per-domain via
/// [`CompositeStorageProvider`].
#[derive(Clone)]
pub struct StorageProvider {
    session: Arc<dyn SessionStore>,
    memory: Arc<dyn MemoryStore>,
    run: Arc<dyn RunStore>,
    observability: Arc<dyn ObservabilityStore>,
}

impl StorageProvider {
    /// Construct from four explicit per-domain stores.
    pub fn new(
        session: Arc<dyn SessionStore>,
        memory: Arc<dyn MemoryStore>,
        run: Arc<dyn RunStore>,
        observability: Arc<dyn ObservabilityStore>,
    ) -> Self {
        Self {
            session,
            memory,
            run,
            observability,
        }
    }

    /// Clone a single concrete provider implementing all four domain traits
    /// into all four slots.
    pub fn single<P>(provider: Arc<P>) -> Self
    where
        P: SessionStore + MemoryStore + RunStore + ObservabilityStore + 'static,
    {
        Self {
            session: provider.clone(),
            memory: provider.clone(),
            run: provider.clone(),
            observability: provider,
        }
    }

    /// All-no-op provider. The default when `.storage(...)` is never set.
    pub fn no_op() -> Self {
        Self::single(Arc::new(NoOpStorageProvider))
    }

    pub fn session(&self) -> &Arc<dyn SessionStore> {
        &self.session
    }
    pub fn memory(&self) -> &Arc<dyn MemoryStore> {
        &self.memory
    }
    pub fn run(&self) -> &Arc<dyn RunStore> {
        &self.run
    }
    pub fn observability(&self) -> &Arc<dyn ObservabilityStore> {
        &self.observability
    }
}

impl Default for StorageProvider {
    fn default() -> Self {
        Self::no_op()
    }
}

// ============================================================================
// NoOpStorageProvider
// ============================================================================

/// Silent-discard provider. Reads return `Ok(None)` / `Ok(vec![])`; writes
/// return `Ok(())`. The default for any unconfigured domain.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpStorageProvider;

impl SessionStore for NoOpStorageProvider {
    fn get_session<'a>(
        &'a self,
        _id: &'a SessionId,
    ) -> BoxFut<'a, Result<Option<PausedState>, StorageError>> {
        Box::pin(async { Ok(None) })
    }
    fn put_session<'a>(
        &'a self,
        _id: &'a SessionId,
        _state: &'a PausedState,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async { Ok(()) })
    }
    fn delete_session<'a>(&'a self, _id: &'a SessionId) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async { Ok(()) })
    }
    fn list_sessions(&self) -> BoxFut<'_, Result<Vec<SessionId>, StorageError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

impl MemoryStore for NoOpStorageProvider {
    fn append_memory<'a>(
        &'a self,
        _session_id: &'a SessionId,
        _entry: MemoryEntry,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async { Ok(()) })
    }
    fn get_memories<'a>(
        &'a self,
        _session_id: &'a SessionId,
        _limit: usize,
    ) -> BoxFut<'a, Result<Vec<MemoryEntry>, StorageError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

impl RunStore for NoOpStorageProvider {
    fn get<'a>(
        &'a self,
        _session_id: &'a SessionId,
        _key: &'a str,
    ) -> BoxFut<'a, Result<Option<JsonValue>, StorageError>> {
        Box::pin(async { Ok(None) })
    }
    fn put<'a>(
        &'a self,
        _session_id: &'a SessionId,
        _key: &'a str,
        _value: JsonValue,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async { Ok(()) })
    }
    fn delete<'a>(
        &'a self,
        _session_id: &'a SessionId,
        _key: &'a str,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async { Ok(()) })
    }
    fn list_keys<'a>(
        &'a self,
        _session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<Vec<String>, StorageError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

impl ObservabilityStore for NoOpStorageProvider {
    fn append_span<'a>(
        &'a self,
        _session_id: &'a SessionId,
        _span: JsonValue,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async { Ok(()) })
    }
    fn get_spans<'a>(
        &'a self,
        _session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<Vec<JsonValue>, StorageError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn get_sessions<'a>(
        &'a self,
        _since: Timestamp,
        _domain: Option<String>,
        _outcome: Option<SessionOutcome>,
    ) -> BoxFut<'a, Result<Vec<SessionMetrics>, StorageError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn flush_session<'a>(
        &'a self,
        _session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async { Ok(()) })
    }
}

// ============================================================================
// InMemoryStorageProvider
// ============================================================================

/// Mutex-guarded in-memory provider. Used in tests and ephemeral runs.
#[derive(Default)]
pub struct InMemoryStorageProvider {
    sessions: Mutex<HashMap<SessionId, PausedState>>,
    memory: Mutex<HashMap<SessionId, Vec<MemoryEntry>>>,
    run: Mutex<HashMap<(SessionId, String), JsonValue>>,
    spans: Mutex<HashMap<SessionId, Vec<JsonValue>>>,
}

impl InMemoryStorageProvider {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SessionStore for InMemoryStorageProvider {
    fn get_session<'a>(
        &'a self,
        id: &'a SessionId,
    ) -> BoxFut<'a, Result<Option<PausedState>, StorageError>> {
        Box::pin(async move {
            let map = self.sessions.lock().unwrap();
            Ok(map.get(id).cloned())
        })
    }
    fn put_session<'a>(
        &'a self,
        id: &'a SessionId,
        state: &'a PausedState,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            self.sessions
                .lock()
                .unwrap()
                .insert(id.clone(), state.clone());
            Ok(())
        })
    }
    fn delete_session<'a>(&'a self, id: &'a SessionId) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            self.sessions.lock().unwrap().remove(id);
            Ok(())
        })
    }
    fn list_sessions(&self) -> BoxFut<'_, Result<Vec<SessionId>, StorageError>> {
        Box::pin(async move {
            let mut keys: Vec<SessionId> = self.sessions.lock().unwrap().keys().cloned().collect();
            keys.sort_by(|a, b| a.as_str().cmp(b.as_str()));
            Ok(keys)
        })
    }
}

impl MemoryStore for InMemoryStorageProvider {
    fn append_memory<'a>(
        &'a self,
        session_id: &'a SessionId,
        entry: MemoryEntry,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            self.memory
                .lock()
                .unwrap()
                .entry(session_id.clone())
                .or_default()
                .push(entry);
            Ok(())
        })
    }
    fn get_memories<'a>(
        &'a self,
        session_id: &'a SessionId,
        limit: usize,
    ) -> BoxFut<'a, Result<Vec<MemoryEntry>, StorageError>> {
        Box::pin(async move {
            let map = self.memory.lock().unwrap();
            let all = map.get(session_id).cloned().unwrap_or_default();
            Ok(most_recent_newest_first(all, limit))
        })
    }
}

impl RunStore for InMemoryStorageProvider {
    fn get<'a>(
        &'a self,
        session_id: &'a SessionId,
        key: &'a str,
    ) -> BoxFut<'a, Result<Option<JsonValue>, StorageError>> {
        Box::pin(async move {
            let map = self.run.lock().unwrap();
            Ok(map.get(&(session_id.clone(), key.to_string())).cloned())
        })
    }
    fn put<'a>(
        &'a self,
        session_id: &'a SessionId,
        key: &'a str,
        value: JsonValue,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            self.run
                .lock()
                .unwrap()
                .insert((session_id.clone(), key.to_string()), value);
            Ok(())
        })
    }
    fn delete<'a>(
        &'a self,
        session_id: &'a SessionId,
        key: &'a str,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            self.run
                .lock()
                .unwrap()
                .remove(&(session_id.clone(), key.to_string()));
            Ok(())
        })
    }
    fn list_keys<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<Vec<String>, StorageError>> {
        Box::pin(async move {
            let map = self.run.lock().unwrap();
            let mut keys: Vec<String> = map
                .keys()
                .filter(|(s, _)| s == session_id)
                .map(|(_, k)| k.clone())
                .collect();
            keys.sort();
            Ok(keys)
        })
    }
}

impl ObservabilityStore for InMemoryStorageProvider {
    fn append_span<'a>(
        &'a self,
        session_id: &'a SessionId,
        span: JsonValue,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            self.spans
                .lock()
                .unwrap()
                .entry(session_id.clone())
                .or_default()
                .push(span);
            Ok(())
        })
    }
    fn get_spans<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<Vec<JsonValue>, StorageError>> {
        Box::pin(async move {
            Ok(self
                .spans
                .lock()
                .unwrap()
                .get(session_id)
                .cloned()
                .unwrap_or_default())
        })
    }
    fn get_sessions<'a>(
        &'a self,
        _since: Timestamp,
        _domain: Option<String>,
        _outcome: Option<SessionOutcome>,
    ) -> BoxFut<'a, Result<Vec<SessionMetrics>, StorageError>> {
        // The in-memory span store does not roll up SessionMetrics — that is the
        // ObservabilityProvider's job. Storage-only query returns empty.
        Box::pin(async { Ok(Vec::new()) })
    }
    fn flush_session<'a>(
        &'a self,
        _session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async { Ok(()) })
    }
}

// ============================================================================
// FileSystemStorageProvider
// ============================================================================

/// Disk-backed provider rooted at `root`. Layout mirrors `.spore/`:
///   - session → `{root}/sessions/{id}/state.json` (atomic write-rename)
///   - run     → `{root}/sessions/{id}/run/{key}.json` (atomic write-rename)
///   - memory  → `{root}/sessions/{id}/memory.jsonl` (append)
///   - obs     → `{root}/sessions/{id}/trace.jsonl` (append)
///
/// `flush_session` creates a sibling `.flushed` marker.
#[derive(Debug, Clone)]
pub struct FileSystemStorageProvider {
    root: PathBuf,
}

impl FileSystemStorageProvider {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn session_dir(&self, id: &SessionId) -> PathBuf {
        self.root.join("sessions").join(id.as_str())
    }
    fn state_path(&self, id: &SessionId) -> PathBuf {
        self.session_dir(id).join("state.json")
    }
    fn run_dir(&self, id: &SessionId) -> PathBuf {
        self.session_dir(id).join("run")
    }
    fn run_path(&self, id: &SessionId, key: &str) -> PathBuf {
        self.run_dir(id).join(format!("{key}.json"))
    }
    fn memory_path(&self, id: &SessionId) -> PathBuf {
        self.session_dir(id).join("memory.jsonl")
    }
    fn trace_path(&self, id: &SessionId) -> PathBuf {
        self.session_dir(id).join("trace.jsonl")
    }
}

/// Atomic write-rename: ensure parent dir, write full bytes to sibling
/// `{target}.tmp`, flush + fsync, then `rename(tmp, target)`. On any failure the
/// `.tmp` is removed so no partial sidecar is left behind. Byte-identical
/// algorithm across all four languages.
fn atomic_write(target: &Path, bytes: &[u8]) -> Result<(), StorageError> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = {
        let mut t = target.as_os_str().to_os_string();
        t.push(".tmp");
        PathBuf::from(t)
    };
    let result = (|| -> std::io::Result<()> {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.flush()?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, target)
    })();
    if let Err(e) = result {
        // Best-effort cleanup; leave no leftover .tmp.
        let _ = fs::remove_file(&tmp);
        return Err(StorageError::Io(e));
    }
    Ok(())
}

/// Append one JSONL line (the value plus a trailing `\n`), flushing the handle.
fn append_jsonl(path: &Path, value: &JsonValue) -> Result<(), StorageError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let line =
        serde_json::to_string(value).map_err(|e| StorageError::Serialization(e.to_string()))?;
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(line.as_bytes())?;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}

/// Read every non-empty JSONL line from `path` as a [`JsonValue`]. Missing file
/// → empty vec.
fn read_jsonl(path: &Path) -> Result<Vec<JsonValue>, StorageError> {
    match fs::read_to_string(path) {
        Ok(raw) => {
            let mut out = Vec::new();
            for line in raw.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                out.push(
                    serde_json::from_str(line)
                        .map_err(|e| StorageError::Serialization(e.to_string()))?,
                );
            }
            Ok(out)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(StorageError::Io(e)),
    }
}

impl SessionStore for FileSystemStorageProvider {
    fn get_session<'a>(
        &'a self,
        id: &'a SessionId,
    ) -> BoxFut<'a, Result<Option<PausedState>, StorageError>> {
        Box::pin(async move {
            match fs::read(self.state_path(id)) {
                Ok(bytes) => {
                    let state = serde_json::from_slice(&bytes)
                        .map_err(|e| StorageError::Serialization(e.to_string()))?;
                    Ok(Some(state))
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(StorageError::Io(e)),
            }
        })
    }
    fn put_session<'a>(
        &'a self,
        id: &'a SessionId,
        state: &'a PausedState,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            let bytes = serde_json::to_vec(state)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            atomic_write(&self.state_path(id), &bytes)
        })
    }
    fn delete_session<'a>(&'a self, id: &'a SessionId) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            match fs::remove_file(self.state_path(id)) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(StorageError::Io(e)),
            }
        })
    }
    fn list_sessions(&self) -> BoxFut<'_, Result<Vec<SessionId>, StorageError>> {
        Box::pin(async move {
            let mut out = Vec::new();
            let sessions_dir = self.root.join("sessions");
            match fs::read_dir(&sessions_dir) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.join("state.json").exists() {
                            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                                out.push(SessionId::new(name));
                            }
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(StorageError::Io(e)),
            }
            out.sort_by(|a, b| a.as_str().cmp(b.as_str()));
            Ok(out)
        })
    }
}

impl MemoryStore for FileSystemStorageProvider {
    fn append_memory<'a>(
        &'a self,
        session_id: &'a SessionId,
        entry: MemoryEntry,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            let value = serde_json::to_value(&entry)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            append_jsonl(&self.memory_path(session_id), &value)
        })
    }
    fn get_memories<'a>(
        &'a self,
        session_id: &'a SessionId,
        limit: usize,
    ) -> BoxFut<'a, Result<Vec<MemoryEntry>, StorageError>> {
        Box::pin(async move {
            let values = read_jsonl(&self.memory_path(session_id))?;
            let mut entries = Vec::with_capacity(values.len());
            for v in values {
                entries.push(
                    serde_json::from_value(v)
                        .map_err(|e| StorageError::Serialization(e.to_string()))?,
                );
            }
            Ok(most_recent_newest_first(entries, limit))
        })
    }
}

impl RunStore for FileSystemStorageProvider {
    fn get<'a>(
        &'a self,
        session_id: &'a SessionId,
        key: &'a str,
    ) -> BoxFut<'a, Result<Option<JsonValue>, StorageError>> {
        Box::pin(async move {
            match fs::read(self.run_path(session_id, key)) {
                Ok(bytes) => {
                    let value = serde_json::from_slice(&bytes)
                        .map_err(|e| StorageError::Serialization(e.to_string()))?;
                    Ok(Some(value))
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(StorageError::Io(e)),
            }
        })
    }
    fn put<'a>(
        &'a self,
        session_id: &'a SessionId,
        key: &'a str,
        value: JsonValue,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            let bytes = serde_json::to_vec(&value)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            atomic_write(&self.run_path(session_id, key), &bytes)
        })
    }
    fn delete<'a>(
        &'a self,
        session_id: &'a SessionId,
        key: &'a str,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            match fs::remove_file(self.run_path(session_id, key)) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(StorageError::Io(e)),
            }
        })
    }
    fn list_keys<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<Vec<String>, StorageError>> {
        Box::pin(async move {
            let mut out = Vec::new();
            match fs::read_dir(self.run_dir(session_id)) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        let name = name.to_string_lossy();
                        if let Some(key) = name.strip_suffix(".json") {
                            out.push(key.to_string());
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(StorageError::Io(e)),
            }
            out.sort();
            Ok(out)
        })
    }
}

impl ObservabilityStore for FileSystemStorageProvider {
    fn append_span<'a>(
        &'a self,
        session_id: &'a SessionId,
        span: JsonValue,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move { append_jsonl(&self.trace_path(session_id), &span) })
    }
    fn get_spans<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<Vec<JsonValue>, StorageError>> {
        Box::pin(async move { read_jsonl(&self.trace_path(session_id)) })
    }
    fn get_sessions<'a>(
        &'a self,
        _since: Timestamp,
        _domain: Option<String>,
        _outcome: Option<SessionOutcome>,
    ) -> BoxFut<'a, Result<Vec<SessionMetrics>, StorageError>> {
        // SessionMetrics roll-up is owned by the ObservabilityProvider, not the
        // raw on-disk span store. Storage-only query returns empty.
        Box::pin(async { Ok(Vec::new()) })
    }
    fn flush_session<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<(), StorageError>> {
        Box::pin(async move {
            let dir = self.session_dir(session_id);
            fs::create_dir_all(&dir)?;
            File::create(dir.join(".flushed"))?;
            Ok(())
        })
    }
}

// ============================================================================
// CompositeStorageProvider
// ============================================================================

/// Builder that routes each domain to its own backend, filling any unset domain
/// with [`NoOpStorageProvider`] on `.build()`.
#[derive(Default)]
pub struct CompositeStorageProvider {
    session: Option<Arc<dyn SessionStore>>,
    memory: Option<Arc<dyn MemoryStore>>,
    run: Option<Arc<dyn RunStore>>,
    observability: Option<Arc<dyn ObservabilityStore>>,
}

impl CompositeStorageProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn session(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.session = Some(store);
        self
    }
    pub fn memory(mut self, store: Arc<dyn MemoryStore>) -> Self {
        self.memory = Some(store);
        self
    }
    pub fn run(mut self, store: Arc<dyn RunStore>) -> Self {
        self.run = Some(store);
        self
    }
    pub fn observability(mut self, store: Arc<dyn ObservabilityStore>) -> Self {
        self.observability = Some(store);
        self
    }

    /// Build a [`StorageProvider`], filling each unset domain with a
    /// [`NoOpStorageProvider`].
    pub fn build(self) -> StorageProvider {
        StorageProvider {
            session: self
                .session
                .unwrap_or_else(|| Arc::new(NoOpStorageProvider)),
            memory: self.memory.unwrap_or_else(|| Arc::new(NoOpStorageProvider)),
            run: self.run.unwrap_or_else(|| Arc::new(NoOpStorageProvider)),
            observability: self
                .observability
                .unwrap_or_else(|| Arc::new(NoOpStorageProvider)),
        }
    }
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Return the most-recent `limit` items, newest-first, given a vector in
/// append (oldest-first) order.
fn most_recent_newest_first<T>(mut items: Vec<T>, limit: usize) -> Vec<T> {
    // Newest-first: reverse the append order.
    items.reverse();
    items.truncate(limit);
    items
}

// ============================================================================
// OTLP endpoint parsing (cross-language ground truth — see fan-out refactor)
// ============================================================================

/// Parse the comma-separated `SPORE_OTLP_ENDPOINT` value: `split(',')`, trim
/// each segment, drop empty segments. This is the single most important
/// cross-language fixture (`fixtures/storage/otlp_endpoints_parse.json`) and
/// MUST be byte-identical in every language.
pub fn parse_otlp_endpoints(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests;
