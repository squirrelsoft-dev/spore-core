//! Issue #8 — `MemoryProvider`: persist and retrieve knowledge across
//! turns and sessions.
//!
//! Stores two distinct kinds of memory:
//!   - **Episodic**: what happened during a specific session (created by the
//!     harness from session observations).
//!   - **Semantic**: generalized knowledge — skills, rules, patterns, domain
//!     facts — distilled from one or more episodic traces.
//!
//! See `docs/harness-engineering-concepts.md` § "MemoryProvider" for the rules
//! this module enforces. The reference implementation [`StandardMemoryProvider`]
//! is in-memory; production deployments swap in `SqliteMemoryProvider` or a
//! durable equivalent without changing this trait.
//!
//! ## Rules enforced
//!   - Episodic and semantic memory live in separate stores (separate
//!     `store_*` / `get_*` methods, separate maps).
//!   - `store_semantic(_, MergeStrategy::Replace)` archives the previous
//!     record under a fresh archival id, links it into `previous_versions`,
//!     and bumps `version`. Hard deletes are not permitted.
//!   - `MergeStrategy::Append` concatenates content into the existing record
//!     in place — same id, no new version. Use only for accumulator memories.
//!   - `MergeStrategy::Reject` returns `MergeConflict` on collision.
//!   - `MetaAgentProposed` memories are forced into `MemoryStatus::PendingReview`
//!     regardless of caller-supplied status. The provider does not trust input.
//!   - Empty content fails with `ValidationFailed` (writes are validated).
//!   - `query` returns items with `relevance_score >= query.min_relevance`,
//!     capped at `max_items`, sorted by score descending. Only `Active`
//!     semantic memories are returned by `query`.
//!   - Versions are retained — `get_version_history` walks the
//!     `previous_versions` chain.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::harness::SessionId;

// ============================================================================
// Identity & time
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(pub String);

impl MemoryId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// RFC 3339 / ISO 8601 timestamp. Stored as a string for cross-language
/// fixture portability — every language target serializes the same bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Timestamp(pub String);

impl Timestamp {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Current wall-clock time as an RFC 3339 UTC string with millisecond
    /// precision (e.g. `2026-05-26T18:00:00.123Z`), matching the trace schema.
    ///
    /// Implemented with `std` only (no `chrono`/`time` dependency) via Howard
    /// Hinnant's `civil_from_days` algorithm. Spans compare lexically, so this
    /// is monotonic enough for ordering; OTLP backends restamp at export time.
    pub fn now() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let dur = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Self::new(Self::format_rfc3339(
            dur.as_secs() as i64,
            dur.subsec_millis(),
        ))
    }

    /// Format `secs` since the Unix epoch plus `millis` as RFC 3339 UTC.
    /// Exposed at crate level so deterministic tests can pin a value.
    pub(crate) fn format_rfc3339(secs: i64, millis: u32) -> String {
        let days = secs.div_euclid(86_400);
        let rem = secs.rem_euclid(86_400);
        let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
        // civil_from_days (Hinnant): days since 1970-01-01 → (y, m, d).
        let z = days + 719_468;
        let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
        let doe = z - era * 146_097; // [0, 146096]
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
        let mp = (5 * doy + 2) / 153; // [0, 11]
        let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
        let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
        let y = if m <= 2 { y + 1 } else { y };
        format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{millis:03}Z")
    }
}

#[cfg(test)]
mod timestamp_now_tests {
    use super::Timestamp;

    #[test]
    fn format_rfc3339_matches_known_epoch() {
        // 2026-05-26T18:00:00.123Z = 1779818400 s since epoch.
        assert_eq!(
            Timestamp::format_rfc3339(1_779_818_400, 123),
            "2026-05-26T18:00:00.123Z"
        );
        // Unix epoch.
        assert_eq!(Timestamp::format_rfc3339(0, 0), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn now_is_rfc3339_shaped() {
        let s = Timestamp::now().0;
        assert_eq!(s.len(), 24, "expected YYYY-MM-DDTHH:MM:SS.mmmZ, got {s}");
        assert!(s.ends_with('Z'));
        assert_eq!(&s[4..5], "-");
    }
}

// ============================================================================
// Spec-defined records
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub id: MemoryId,
    pub session_id: SessionId,
    pub content: String,
    pub created_at: Timestamp,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MemorySource {
    Manual,
    SessionGenerated {
        session_id: SessionId,
    },
    TraceDistilled {
        session_ids: Vec<SessionId>,
    },
    MetaAgentProposed {
        #[serde(default)]
        approved_by: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MemoryStatus {
    Active,
    Deprecated { reason: String, at: Timestamp },
    PendingReview { proposed_at: Timestamp },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticMemory {
    pub id: MemoryId,
    pub content: String,
    pub source: MemorySource,
    #[serde(default)]
    pub domain: Option<String>,
    pub version: u32,
    #[serde(default)]
    pub previous_versions: Vec<MemoryId>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub status: MemoryStatus,
}

/// Scored query result — what `query` returns. Lives here (not in `context`)
/// as the canonical definition; `ContextSources::memory` consumes this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryItem {
    pub memory: SemanticMemory,
    pub relevance_score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryQuery {
    pub task_instruction: String,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub session_id: Option<SessionId>,
    #[serde(default = "default_min_relevance")]
    pub min_relevance: f32,
    #[serde(default = "default_max_items")]
    pub max_items: u32,
}

fn default_min_relevance() -> f32 {
    0.5
}
fn default_max_items() -> u32 {
    10
}

impl MemoryQuery {
    pub fn new(task_instruction: impl Into<String>) -> Self {
        Self {
            task_instruction: task_instruction.into(),
            domain: None,
            session_id: None,
            min_relevance: default_min_relevance(),
            max_items: default_max_items(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    Replace,
    Append,
    Reject,
}

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MemoryError {
    #[error("memory not found: {0:?}")]
    NotFound(MemoryId),
    #[error("merge conflict on {existing:?}: {reason}")]
    MergeConflict { existing: MemoryId, reason: String },
    #[error("validation failed: {reason}")]
    ValidationFailed { reason: String },
    #[error("storage error: {reason}")]
    StorageError { reason: String },
}

// ============================================================================
// Trait
// ============================================================================

#[trait_variant::make(Send)]
pub trait MemoryProvider: Send + Sync {
    // ── Episodic ────────────────────────────────────────────────────────────
    async fn store_episodic(&self, memory: EpisodicMemory) -> Result<MemoryId, MemoryError>;
    async fn get_episodic(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<EpisodicMemory>, MemoryError>;

    // ── Semantic ────────────────────────────────────────────────────────────
    async fn store_semantic(
        &self,
        memory: SemanticMemory,
        on_conflict: MergeStrategy,
    ) -> Result<MemoryId, MemoryError>;
    async fn get_semantic(&self, id: &MemoryId) -> Result<SemanticMemory, MemoryError>;

    /// Primary retrieval path. Returns items with score >= `min_relevance`,
    /// capped at `max_items`, sorted by score descending. Only `Active` memories.
    async fn query(&self, query: MemoryQuery) -> Result<Vec<MemoryItem>, MemoryError>;

    // ── Lifecycle ───────────────────────────────────────────────────────────
    async fn deprecate(&self, id: &MemoryId, reason: &str) -> Result<(), MemoryError>;
    async fn get_version_history(&self, id: &MemoryId) -> Result<Vec<SemanticMemory>, MemoryError>;
    async fn mark_pending_review(&self, id: &MemoryId) -> Result<(), MemoryError>;
}

// ============================================================================
// Standard in-memory implementation
// ============================================================================

/// Reference `MemoryProvider`. In-memory; suitable for tests and short-lived
/// processes. Production deployments use a durable backing store but must
/// preserve the rules above byte-for-byte.
///
/// Relevance scoring: token-overlap (Jaccard) between the query's
/// `task_instruction` and each candidate memory's `content`. This is
/// intentionally simple — the spec calls for embedding similarity in
/// production. The trait does not mandate the scoring algorithm, only that
/// items below `min_relevance` are not returned.
#[derive(Default)]
pub struct StandardMemoryProvider {
    inner: Mutex<Store>,
}

#[derive(Default)]
struct Store {
    /// Episodic memories keyed by session, preserving insertion order.
    episodic: HashMap<SessionId, Vec<EpisodicMemory>>,
    /// All semantic memories ever stored, keyed by id (current + archived).
    semantic: HashMap<MemoryId, SemanticMemory>,
    /// Monotonic counter for generating archival ids on Replace.
    archive_seq: u64,
    /// Optional pinned "now" for deterministic tests (mirrors
    /// [`guide_registry`](crate::guide_registry)'s `now_override`). When `None`,
    /// lifecycle transitions stamp the real system clock via [`Timestamp::now`].
    now_override: Option<Timestamp>,
}

impl StandardMemoryProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pin the "now" timestamp used to stamp lifecycle transitions
    /// (`deprecate` / `mark_pending_review`). Tests use this for determinism.
    pub fn set_now(&self, now: Timestamp) {
        self.inner.lock().unwrap().now_override = Some(now);
    }

    fn next_archive_id(store: &mut Store, original: &MemoryId) -> MemoryId {
        store.archive_seq += 1;
        MemoryId(format!("{}@v{}", original.0, store.archive_seq))
    }
}

fn validate_content(s: &str) -> Result<(), MemoryError> {
    if s.trim().is_empty() {
        return Err(MemoryError::ValidationFailed {
            reason: "content must not be empty".into(),
        });
    }
    Ok(())
}

/// Force `MetaAgentProposed` memories into `PendingReview` regardless of
/// caller-supplied status. The `proposed_at` falls back to `created_at` when
/// the caller did not supply a review timestamp.
fn enforce_meta_agent_review(memory: &mut SemanticMemory) {
    if matches!(memory.source, MemorySource::MetaAgentProposed { .. })
        && !matches!(memory.status, MemoryStatus::PendingReview { .. })
    {
        memory.status = MemoryStatus::PendingReview {
            proposed_at: memory.created_at.clone(),
        };
    }
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

fn jaccard(a: &str, b: &str) -> f32 {
    let ta: std::collections::HashSet<_> = tokenize(a).into_iter().collect();
    let tb: std::collections::HashSet<_> = tokenize(b).into_iter().collect();
    if ta.is_empty() && tb.is_empty() {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count() as f32;
    let union = ta.union(&tb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

impl MemoryProvider for StandardMemoryProvider {
    async fn store_episodic(&self, memory: EpisodicMemory) -> Result<MemoryId, MemoryError> {
        validate_content(&memory.content)?;
        let mut store = self.inner.lock().unwrap();
        let id = memory.id.clone();
        store
            .episodic
            .entry(memory.session_id.clone())
            .or_default()
            .push(memory);
        Ok(id)
    }

    async fn get_episodic(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<EpisodicMemory>, MemoryError> {
        let store = self.inner.lock().unwrap();
        Ok(store.episodic.get(session_id).cloned().unwrap_or_default())
    }

    async fn store_semantic(
        &self,
        mut memory: SemanticMemory,
        on_conflict: MergeStrategy,
    ) -> Result<MemoryId, MemoryError> {
        validate_content(&memory.content)?;
        enforce_meta_agent_review(&mut memory);

        let mut store = self.inner.lock().unwrap();
        let id = memory.id.clone();

        if let Some(existing) = store.semantic.get(&id).cloned() {
            match on_conflict {
                MergeStrategy::Reject => {
                    return Err(MemoryError::MergeConflict {
                        existing: existing.id,
                        reason: "memory exists; on_conflict=Reject".into(),
                    });
                }
                MergeStrategy::Append => {
                    let mut merged = existing;
                    merged.content.push_str(&memory.content);
                    merged.updated_at = memory.updated_at;
                    store.semantic.insert(id.clone(), merged);
                    return Ok(id);
                }
                MergeStrategy::Replace => {
                    // Archive the existing record under a fresh archival id,
                    // link it into the new memory's previous_versions, bump
                    // version, and write the new record under the original id.
                    let archive_id = Self::next_archive_id(&mut store, &existing.id);
                    let mut archived = existing.clone();
                    archived.id = archive_id.clone();
                    store.semantic.insert(archive_id.clone(), archived);

                    memory.version = existing.version + 1;
                    memory.previous_versions = {
                        let mut v = existing.previous_versions.clone();
                        v.push(archive_id);
                        v
                    };
                }
            }
        }

        store.semantic.insert(id.clone(), memory);
        Ok(id)
    }

    async fn get_semantic(&self, id: &MemoryId) -> Result<SemanticMemory, MemoryError> {
        let store = self.inner.lock().unwrap();
        store
            .semantic
            .get(id)
            .cloned()
            .ok_or_else(|| MemoryError::NotFound(id.clone()))
    }

    async fn query(&self, query: MemoryQuery) -> Result<Vec<MemoryItem>, MemoryError> {
        let store = self.inner.lock().unwrap();
        let mut scored: Vec<MemoryItem> = store
            .semantic
            .values()
            .filter(|m| matches!(m.status, MemoryStatus::Active))
            .filter(|m| match (&query.domain, &m.domain) {
                (Some(want), Some(have)) => want == have,
                (Some(_), None) => false,
                _ => true,
            })
            .map(|m| MemoryItem {
                memory: m.clone(),
                relevance_score: jaccard(&query.task_instruction, &m.content),
            })
            .filter(|item| item.relevance_score >= query.min_relevance)
            .collect();
        scored.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(query.max_items as usize);
        Ok(scored)
    }

    async fn deprecate(&self, id: &MemoryId, reason: &str) -> Result<(), MemoryError> {
        let mut store = self.inner.lock().unwrap();
        // Stamp the transition with "now", not the stale `updated_at` (which
        // predates this deprecation).
        let now = store.now_override.clone().unwrap_or_else(Timestamp::now);
        let m = store
            .semantic
            .get_mut(id)
            .ok_or_else(|| MemoryError::NotFound(id.clone()))?;
        m.status = MemoryStatus::Deprecated {
            reason: reason.to_string(),
            at: now,
        };
        Ok(())
    }

    async fn get_version_history(&self, id: &MemoryId) -> Result<Vec<SemanticMemory>, MemoryError> {
        let store = self.inner.lock().unwrap();
        let head = store
            .semantic
            .get(id)
            .ok_or_else(|| MemoryError::NotFound(id.clone()))?;
        let mut chain = vec![head.clone()];
        for prev_id in &head.previous_versions {
            if let Some(prev) = store.semantic.get(prev_id) {
                chain.push(prev.clone());
            }
        }
        Ok(chain)
    }

    async fn mark_pending_review(&self, id: &MemoryId) -> Result<(), MemoryError> {
        let mut store = self.inner.lock().unwrap();
        // Stamp the transition with "now", not the stale `updated_at`.
        let now = store.now_override.clone().unwrap_or_else(Timestamp::now);
        let m = store
            .semantic
            .get_mut(id)
            .ok_or_else(|| MemoryError::NotFound(id.clone()))?;
        m.status = MemoryStatus::PendingReview { proposed_at: now };
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: &str) -> Timestamp {
        Timestamp::new(s)
    }

    fn sem(id: &str, content: &str) -> SemanticMemory {
        SemanticMemory {
            id: MemoryId::new(id),
            content: content.into(),
            source: MemorySource::Manual,
            domain: None,
            version: 1,
            previous_versions: vec![],
            created_at: ts("2026-05-16T00:00:00Z"),
            updated_at: ts("2026-05-16T00:00:00Z"),
            status: MemoryStatus::Active,
        }
    }

    fn epi(id: &str, session: &str, content: &str) -> EpisodicMemory {
        EpisodicMemory {
            id: MemoryId::new(id),
            session_id: SessionId::new(session),
            content: content.into(),
            created_at: ts("2026-05-16T00:00:00Z"),
            tags: vec![],
        }
    }

    // ── Rule: Episodic and semantic stored/retrieved separately ─────────────

    #[tokio::test]
    async fn episodic_and_semantic_use_separate_stores() {
        let mp = StandardMemoryProvider::new();
        mp.store_episodic(epi("e1", "s1", "ran tests"))
            .await
            .unwrap();
        mp.store_semantic(sem("g1", "always run tests"), MergeStrategy::Reject)
            .await
            .unwrap();

        let eps = mp.get_episodic(&SessionId::new("s1")).await.unwrap();
        assert_eq!(eps.len(), 1);
        // Episodic id should not be retrievable as semantic.
        assert!(mp.get_semantic(&MemoryId::new("e1")).await.is_err());
        // Semantic id should not appear under any session's episodics.
        assert!(mp
            .get_episodic(&SessionId::new("g1"))
            .await
            .unwrap()
            .is_empty());
    }

    // ── Rule: Replace creates a new version, retains previous ───────────────

    #[tokio::test]
    async fn replace_archives_previous_and_bumps_version() {
        let mp = StandardMemoryProvider::new();
        mp.store_semantic(sem("g1", "v1 content"), MergeStrategy::Reject)
            .await
            .unwrap();

        let mut v2 = sem("g1", "v2 content");
        v2.version = 1; // caller version is ignored; provider bumps from existing.
        mp.store_semantic(v2, MergeStrategy::Replace).await.unwrap();

        let current = mp.get_semantic(&MemoryId::new("g1")).await.unwrap();
        assert_eq!(current.content, "v2 content");
        assert_eq!(current.version, 2);
        assert_eq!(current.previous_versions.len(), 1);

        let history = mp.get_version_history(&MemoryId::new("g1")).await.unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].content, "v2 content");
        assert_eq!(history[1].content, "v1 content");
    }

    #[tokio::test]
    async fn replace_chains_versions_across_multiple_updates() {
        let mp = StandardMemoryProvider::new();
        mp.store_semantic(sem("g1", "v1"), MergeStrategy::Reject)
            .await
            .unwrap();
        mp.store_semantic(sem("g1", "v2"), MergeStrategy::Replace)
            .await
            .unwrap();
        mp.store_semantic(sem("g1", "v3"), MergeStrategy::Replace)
            .await
            .unwrap();
        let cur = mp.get_semantic(&MemoryId::new("g1")).await.unwrap();
        assert_eq!(cur.version, 3);
        let history = mp.get_version_history(&MemoryId::new("g1")).await.unwrap();
        assert_eq!(history.len(), 3);
    }

    // ── Rule: Reject returns MergeConflict ──────────────────────────────────

    #[tokio::test]
    async fn reject_on_conflict_errors() {
        let mp = StandardMemoryProvider::new();
        mp.store_semantic(sem("g1", "first"), MergeStrategy::Reject)
            .await
            .unwrap();
        let err = mp
            .store_semantic(sem("g1", "second"), MergeStrategy::Reject)
            .await
            .unwrap_err();
        assert!(matches!(err, MemoryError::MergeConflict { .. }));
        // Original untouched.
        assert_eq!(
            mp.get_semantic(&MemoryId::new("g1")).await.unwrap().content,
            "first"
        );
    }

    // ── Rule: Append concatenates in place, no new version ──────────────────

    #[tokio::test]
    async fn append_concatenates_without_new_version() {
        let mp = StandardMemoryProvider::new();
        mp.store_semantic(sem("g1", "a"), MergeStrategy::Reject)
            .await
            .unwrap();
        mp.store_semantic(sem("g1", "b"), MergeStrategy::Append)
            .await
            .unwrap();
        let cur = mp.get_semantic(&MemoryId::new("g1")).await.unwrap();
        assert_eq!(cur.content, "ab");
        assert_eq!(cur.version, 1);
        assert!(cur.previous_versions.is_empty());
    }

    // ── Rule: Writes validated (empty content rejected) ─────────────────────

    #[tokio::test]
    async fn empty_content_fails_validation() {
        let mp = StandardMemoryProvider::new();
        let err = mp
            .store_semantic(sem("g1", "   "), MergeStrategy::Reject)
            .await
            .unwrap_err();
        assert!(matches!(err, MemoryError::ValidationFailed { .. }));
        let err = mp.store_episodic(epi("e1", "s1", "")).await.unwrap_err();
        assert!(matches!(err, MemoryError::ValidationFailed { .. }));
    }

    // ── Rule: MetaAgentProposed forced to PendingReview ─────────────────────

    #[tokio::test]
    async fn meta_agent_memories_forced_to_pending_review() {
        let mp = StandardMemoryProvider::new();
        let mut m = sem("g1", "proposed skill");
        m.source = MemorySource::MetaAgentProposed { approved_by: None };
        // Caller dishonestly sets Active.
        m.status = MemoryStatus::Active;
        mp.store_semantic(m, MergeStrategy::Reject).await.unwrap();
        let stored = mp.get_semantic(&MemoryId::new("g1")).await.unwrap();
        assert!(matches!(stored.status, MemoryStatus::PendingReview { .. }));
    }

    // ── Rule: query returns scored items, filtered by min_relevance ─────────

    #[tokio::test]
    async fn query_scores_filters_and_sorts() {
        let mp = StandardMemoryProvider::new();
        mp.store_semantic(sem("g1", "rust async tokio runtime"), MergeStrategy::Reject)
            .await
            .unwrap();
        mp.store_semantic(sem("g2", "python pytest fixtures"), MergeStrategy::Reject)
            .await
            .unwrap();
        mp.store_semantic(sem("g3", "unrelated cooking recipe"), MergeStrategy::Reject)
            .await
            .unwrap();

        let q = MemoryQuery {
            task_instruction: "rust tokio async".into(),
            min_relevance: 0.1,
            max_items: 10,
            domain: None,
            session_id: None,
        };
        let res = mp.query(q).await.unwrap();
        assert!(!res.is_empty());
        assert_eq!(res[0].memory.id, MemoryId::new("g1"));
        // Sorted descending.
        for w in res.windows(2) {
            assert!(w[0].relevance_score >= w[1].relevance_score);
        }
    }

    #[tokio::test]
    async fn query_excludes_deprecated_memories() {
        let mp = StandardMemoryProvider::new();
        mp.store_semantic(sem("g1", "rust tokio"), MergeStrategy::Reject)
            .await
            .unwrap();
        mp.deprecate(&MemoryId::new("g1"), "obsolete")
            .await
            .unwrap();
        let res = mp
            .query(MemoryQuery {
                task_instruction: "rust tokio".into(),
                min_relevance: 0.0,
                max_items: 10,
                domain: None,
                session_id: None,
            })
            .await
            .unwrap();
        assert!(res.is_empty());
    }

    #[tokio::test]
    async fn query_respects_min_relevance_and_max_items() {
        let mp = StandardMemoryProvider::new();
        for i in 0..5 {
            mp.store_semantic(
                sem(&format!("g{i}"), &format!("alpha beta gamma {i}")),
                MergeStrategy::Reject,
            )
            .await
            .unwrap();
        }
        // Very high threshold returns nothing.
        let res = mp
            .query(MemoryQuery {
                task_instruction: "alpha beta gamma".into(),
                min_relevance: 0.99,
                max_items: 10,
                domain: None,
                session_id: None,
            })
            .await
            .unwrap();
        assert!(res.is_empty());
        // max_items cap.
        let res = mp
            .query(MemoryQuery {
                task_instruction: "alpha beta gamma".into(),
                min_relevance: 0.0,
                max_items: 2,
                domain: None,
                session_id: None,
            })
            .await
            .unwrap();
        assert_eq!(res.len(), 2);
    }

    #[tokio::test]
    async fn query_filters_by_domain() {
        let mp = StandardMemoryProvider::new();
        let mut a = sem("a", "shared content");
        a.domain = Some("rust".into());
        let mut b = sem("b", "shared content");
        b.domain = Some("python".into());
        mp.store_semantic(a, MergeStrategy::Reject).await.unwrap();
        mp.store_semantic(b, MergeStrategy::Reject).await.unwrap();
        let res = mp
            .query(MemoryQuery {
                task_instruction: "shared content".into(),
                domain: Some("rust".into()),
                min_relevance: 0.0,
                max_items: 10,
                session_id: None,
            })
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].memory.id, MemoryId::new("a"));
    }

    // ── Rule: Lifecycle — deprecate ────────────────────────────────────────

    #[tokio::test]
    async fn deprecate_sets_status() {
        let mp = StandardMemoryProvider::new();
        mp.store_semantic(sem("g1", "x"), MergeStrategy::Reject)
            .await
            .unwrap();
        mp.deprecate(&MemoryId::new("g1"), "no longer needed")
            .await
            .unwrap();
        let m = mp.get_semantic(&MemoryId::new("g1")).await.unwrap();
        match m.status {
            MemoryStatus::Deprecated { reason, .. } => assert_eq!(reason, "no longer needed"),
            s => panic!("expected Deprecated, got {s:?}"),
        }
    }

    #[tokio::test]
    async fn deprecate_unknown_id_not_found() {
        let mp = StandardMemoryProvider::new();
        let err = mp.deprecate(&MemoryId::new("nope"), "r").await.unwrap_err();
        assert!(matches!(err, MemoryError::NotFound(_)));
    }

    #[tokio::test]
    async fn deprecate_stamps_now_not_stale_updated_at() {
        // The seeded memory's `updated_at` is 2026-05-16; deprecation happens
        // "later" (pinned to 2026-06-15) and must record THAT, not the stale ts.
        let mp = StandardMemoryProvider::new();
        mp.store_semantic(sem("g1", "x"), MergeStrategy::Reject)
            .await
            .unwrap();
        mp.set_now(ts("2026-06-15T00:00:00Z"));
        mp.deprecate(&MemoryId::new("g1"), "obsolete")
            .await
            .unwrap();
        match mp.get_semantic(&MemoryId::new("g1")).await.unwrap().status {
            MemoryStatus::Deprecated { at, .. } => {
                assert_eq!(at, ts("2026-06-15T00:00:00Z"));
                assert_ne!(at, ts("2026-05-16T00:00:00Z"), "must not reuse updated_at");
            }
            s => panic!("expected Deprecated, got {s:?}"),
        }
    }

    #[tokio::test]
    async fn mark_pending_review_stamps_now_not_stale_updated_at() {
        let mp = StandardMemoryProvider::new();
        mp.store_semantic(sem("g1", "x"), MergeStrategy::Reject)
            .await
            .unwrap();
        mp.set_now(ts("2026-06-15T00:00:00Z"));
        mp.mark_pending_review(&MemoryId::new("g1")).await.unwrap();
        match mp.get_semantic(&MemoryId::new("g1")).await.unwrap().status {
            MemoryStatus::PendingReview { proposed_at } => {
                assert_eq!(proposed_at, ts("2026-06-15T00:00:00Z"));
            }
            s => panic!("expected PendingReview, got {s:?}"),
        }
    }

    // ── Rule: mark_pending_review ───────────────────────────────────────────

    #[tokio::test]
    async fn mark_pending_review_changes_status() {
        let mp = StandardMemoryProvider::new();
        mp.store_semantic(sem("g1", "x"), MergeStrategy::Reject)
            .await
            .unwrap();
        mp.mark_pending_review(&MemoryId::new("g1")).await.unwrap();
        let m = mp.get_semantic(&MemoryId::new("g1")).await.unwrap();
        assert!(matches!(m.status, MemoryStatus::PendingReview { .. }));
    }

    // ── Rule: NotFound errors ───────────────────────────────────────────────

    #[tokio::test]
    async fn get_semantic_unknown_returns_not_found() {
        let mp = StandardMemoryProvider::new();
        let err = mp.get_semantic(&MemoryId::new("nope")).await.unwrap_err();
        assert!(matches!(err, MemoryError::NotFound(_)));
    }

    #[tokio::test]
    async fn get_episodic_unknown_session_returns_empty() {
        let mp = StandardMemoryProvider::new();
        let res = mp.get_episodic(&SessionId::new("none")).await.unwrap();
        assert!(res.is_empty());
    }

    // ── Episodic preserves insertion order across multiple writes ───────────

    #[tokio::test]
    async fn episodic_preserves_order() {
        let mp = StandardMemoryProvider::new();
        for i in 0..5 {
            mp.store_episodic(epi(&format!("e{i}"), "s1", &format!("event {i}")))
                .await
                .unwrap();
        }
        let eps = mp.get_episodic(&SessionId::new("s1")).await.unwrap();
        assert_eq!(eps.len(), 5);
        for (i, e) in eps.iter().enumerate() {
            assert_eq!(e.id, MemoryId::new(format!("e{i}")));
        }
    }
}
