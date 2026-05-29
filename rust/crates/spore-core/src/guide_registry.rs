//! Issue #9 — `GuideRegistry`: manage the lifecycle of guides and skills.
//!
//! Guides are feedforward artifacts injected before the agent acts: system
//! prompt fragments, skills loaded on demand, convention docs (AGENTS.md /
//! CLAUDE.md), schema annotations, and safety rules. The registry is the
//! single source of truth for what guides exist, what state each is in, and
//! how each has performed across sessions.
//!
//! See `docs/harness-engineering-concepts.md` § "GuideRegistry" for the rules
//! this module enforces.
//!
//! ## Rules enforced
//!   - States: `Active`, `PendingReview`, `Deprecated`, `Stale`. Hard delete
//!     is not permitted.
//!   - `register` validates content (non-empty) and runs `check_conflicts`
//!     against existing `Active` guides; conflicts surface as
//!     `ConflictDetected`.
//!   - `GuideSource::MetaAgentProposed` forces `PendingReview { AutomatedProposal }`
//!     regardless of caller-supplied status.
//!   - `select` returns only `Active` guides, filtered by domain and
//!     `guide_types`, ordered by Jaccard relevance to `task_instruction`.
//!   - `record_usage` appends an immutable history record.
//!   - `analyze_performance` emits:
//!       - `GuideDeprecationRecommended` for guides whose failure rate
//!         exceeds the no-guide baseline by `min_failure_rate_delta`.
//!       - `SkillGenerationNeeded` for any failure-reason pattern that
//!         appears at least `min_pattern_occurrences` times across recorded
//!         failures.
//!       - `ConflictResolutionNeeded` for any currently-flagged
//!         `ConflictDetected` pending-review state.
//!   - `promote_to_active` is the only path from `PendingReview` to `Active`.
//!   - Conflict detection (standard impl): same domain + high Jaccard overlap
//!     with an existing active guide but non-identical content.
//!
//! ## Implementor notes
//!   - The standard impl is in-memory. Production deployments swap in
//!     `FilesystemGuideRegistry`.
//!   - `analyze_performance.window` is honored by parsing RFC 3339 timestamps
//!     lexicographically — ISO 8601 UTC timestamps sort correctly. Callers
//!     supply a `now` via [`StandardGuideRegistry::set_now`] for deterministic
//!     tests; otherwise the system clock is used.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::harness::SessionId;
use crate::memory::Timestamp;
use crate::tool_registry::TaskPhase;

// ============================================================================
// Identity
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GuideId(pub String);

impl GuideId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// ============================================================================
// Enums
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuideType {
    SystemPromptFragment,
    Skill,
    ConventionDoc,
    SchemaAnnotation,
    SafetyRule,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GuideSource {
    Manual,
    SessionGenerated { session_id: SessionId },
    TraceDistilled { session_ids: Vec<SessionId> },
    MetaAgentProposed { proposed_at: Timestamp },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PendingReason {
    AutomatedProposal,
    PerformanceDegradation { failure_rate_delta: f32 },
    ConflictDetected { conflicts_with: Vec<GuideId> },
    ManualFlag { note: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GuideStatus {
    Active,
    PendingReview {
        reason: PendingReason,
        since: Timestamp,
    },
    Deprecated {
        reason: String,
        at: Timestamp,
    },
    Stale {
        last_used: Timestamp,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionOutcome {
    Success,
    Failure {
        reason: String,
    },
    Partial,
    /// The session terminated cleanly because a tool escalated a structural
    /// signal to the harness's caller (issue #80, Tool Escalation Protocol).
    /// Distinct from `Partial` — an escalation is an intentional, clean
    /// terminal outcome, not a partial success.
    Escalated,
}

// ============================================================================
// Records
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Guide {
    pub id: GuideId,
    pub name: String,
    pub content: String,
    pub guide_type: GuideType,
    #[serde(default)]
    pub domain: Option<String>,
    pub source: GuideSource,
    pub status: GuideStatus,
    pub created_at: Timestamp,
    #[serde(default)]
    pub last_used: Option<Timestamp>,
    pub version: u32,
}

impl Guide {
    /// Convenience constructor for a `Skill`-type guide with sensible defaults.
    /// Used by the harness and tests that only need `(id, content)` to inject
    /// a skill at turn time.
    pub fn skill(id: impl Into<String>, content: impl Into<String>) -> Self {
        let id = GuideId::new(id);
        Self {
            name: id.0.clone(),
            id,
            content: content.into(),
            guide_type: GuideType::Skill,
            domain: None,
            source: GuideSource::Manual,
            status: GuideStatus::Active,
            created_at: Timestamp::new(""),
            last_used: None,
            version: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuideUsageRecord {
    pub guide_id: GuideId,
    pub session_id: SessionId,
    #[serde(default)]
    pub task_domain: Option<String>,
    pub outcome: SessionOutcome,
    pub recorded_at: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuideQuery {
    pub task_instruction: String,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub phase: Option<TaskPhase>,
    #[serde(default)]
    pub guide_types: Vec<GuideType>,
}

impl GuideQuery {
    pub fn new(task_instruction: impl Into<String>) -> Self {
        Self {
            task_instruction: task_instruction.into(),
            domain: None,
            phase: None,
            guide_types: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuideConflict {
    pub guide_a: GuideId,
    pub guide_b: GuideId,
    pub reason: String,
}

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum GuideRegistryError {
    #[error("guide not found: {0:?}")]
    NotFound(GuideId),
    #[error("conflict detected: {0:?}")]
    ConflictDetected(GuideConflict),
    #[error("validation failed: {reason}")]
    ValidationFailed { reason: String },
    #[error("storage error: {reason}")]
    StorageError { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImprovementSignal {
    SkillGenerationNeeded {
        pattern: String,
        session_ids: Vec<SessionId>,
    },
    GuideDeprecationRecommended {
        guide_id: GuideId,
        reason: String,
    },
    ConflictResolutionNeeded {
        conflict: GuideConflict,
    },
}

// ============================================================================
// Trait
// ============================================================================

#[trait_variant::make(Send)]
pub trait GuideRegistry: Send + Sync {
    async fn register(&self, guide: Guide) -> Result<GuideId, GuideRegistryError>;
    async fn select(&self, query: GuideQuery) -> Result<Vec<Guide>, GuideRegistryError>;
    async fn record_usage(&self, record: GuideUsageRecord) -> Result<(), GuideRegistryError>;
    async fn usage_history(
        &self,
        id: &GuideId,
    ) -> Result<Vec<GuideUsageRecord>, GuideRegistryError>;
    async fn deprecate(&self, id: &GuideId, reason: &str) -> Result<(), GuideRegistryError>;
    async fn mark_pending_review(
        &self,
        id: &GuideId,
        reason: PendingReason,
    ) -> Result<(), GuideRegistryError>;
    async fn promote_to_active(&self, id: &GuideId) -> Result<(), GuideRegistryError>;
    async fn analyze_performance(
        &self,
        window: Duration,
        min_failure_rate_delta: f32,
        min_pattern_occurrences: u32,
    ) -> Vec<ImprovementSignal>;
    async fn check_conflicts(&self, content: &str, domain: Option<&str>) -> Vec<GuideConflict>;
}

// ============================================================================
// Standard implementation
// ============================================================================

/// Reference `GuideRegistry`. In-memory, suitable for tests and short-lived
/// processes. Production deployments use `FilesystemGuideRegistry` (durable).
///
/// Conflict heuristic: two guides conflict when they share a domain and
/// have Jaccard token overlap ≥ `CONFLICT_THRESHOLD` (0.6) but non-identical
/// content. Production deployments can override via a separate implementation
/// of `check_conflicts`.
#[derive(Default)]
pub struct StandardGuideRegistry {
    inner: Mutex<Store>,
}

#[derive(Default)]
struct Store {
    guides: HashMap<GuideId, Guide>,
    usage: Vec<GuideUsageRecord>,
    /// Optional pinned "now" for deterministic tests. When `None`, the
    /// system clock is consulted via RFC 3339 string compare.
    now_override: Option<Timestamp>,
}

const CONFLICT_THRESHOLD: f32 = 0.6;

impl StandardGuideRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pin the "now" timestamp for `analyze_performance` window math. Tests
    /// use this to keep results deterministic without a real clock.
    pub fn set_now(&self, now: Timestamp) {
        self.inner.lock().unwrap().now_override = Some(now);
    }
}

fn validate_content(s: &str) -> Result<(), GuideRegistryError> {
    if s.trim().is_empty() {
        return Err(GuideRegistryError::ValidationFailed {
            reason: "content must not be empty".into(),
        });
    }
    Ok(())
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(String::from)
        .collect()
}

fn jaccard(a: &str, b: &str) -> f32 {
    let ta: HashSet<_> = tokenize(a).into_iter().collect();
    let tb: HashSet<_> = tokenize(b).into_iter().collect();
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

fn enforce_meta_agent_pending(g: &mut Guide) {
    if let GuideSource::MetaAgentProposed { proposed_at } = &g.source {
        g.status = GuideStatus::PendingReview {
            reason: PendingReason::AutomatedProposal,
            since: proposed_at.clone(),
        };
    }
}

/// Lexicographic RFC 3339 comparison: returns true if `t >= cutoff`.
/// Empty cutoff (no window math available) treats every record as in-window.
fn in_window(t: &Timestamp, cutoff: &Option<Timestamp>) -> bool {
    match cutoff {
        Some(c) => t.0.as_str() >= c.0.as_str(),
        None => true,
    }
}

/// Compute a cutoff Timestamp = now - window, using RFC 3339 strings.
/// Returns `None` if `now` doesn't parse as `YYYY-MM-DDTHH:MM:SS[.fff]Z`,
/// in which case the standard impl treats every record as in-window.
fn rfc3339_subtract(now: &Timestamp, window: Duration) -> Option<Timestamp> {
    let secs = parse_rfc3339_secs(&now.0)?;
    let cutoff_secs = secs.checked_sub(window.as_secs() as i64)?;
    Some(Timestamp::new(format_rfc3339_secs(cutoff_secs)))
}

/// Parse `YYYY-MM-DDTHH:MM:SSZ` (and tolerated millisecond variants) to
/// seconds since UNIX epoch. Pure arithmetic — no external dependency.
fn parse_rfc3339_secs(s: &str) -> Option<i64> {
    let s = s.trim_end_matches('Z');
    let (date, time) = s.split_once('T')?;
    let mut date_parts = date.split('-');
    let y: i64 = date_parts.next()?.parse().ok()?;
    let mo: i64 = date_parts.next()?.parse().ok()?;
    let d: i64 = date_parts.next()?.parse().ok()?;
    let time = time.split('.').next()?;
    let mut tp = time.split(':');
    let hh: i64 = tp.next()?.parse().ok()?;
    let mm: i64 = tp.next()?.parse().ok()?;
    let ss: i64 = tp.next()?.parse().ok()?;
    Some(days_from_civil(y, mo, d) * 86400 + hh * 3600 + mm * 60 + ss)
}

fn format_rfc3339_secs(secs: i64) -> String {
    let (y, mo, d, hh, mm, ss) = civil_from_days_secs(secs);
    format!("{y:04}-{mo:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Howard Hinnant's days_from_civil algorithm — proleptic Gregorian.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days_secs(total_secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = total_secs.div_euclid(86400);
    let rem = total_secs.rem_euclid(86400);
    let hh = (rem / 3600) as u32;
    let mm = ((rem % 3600) / 60) as u32;
    let ss = (rem % 60) as u32;
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d, hh, mm, ss)
}

fn now_rfc3339() -> Timestamp {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Timestamp::new(format_rfc3339_secs(secs))
}

impl GuideRegistry for StandardGuideRegistry {
    async fn register(&self, mut guide: Guide) -> Result<GuideId, GuideRegistryError> {
        validate_content(&guide.content)?;
        // MetaAgentProposed forces PendingReview regardless of caller status.
        enforce_meta_agent_pending(&mut guide);

        // Conflict check against existing Active guides.
        let conflicts = self
            .check_conflicts(&guide.content, guide.domain.as_deref())
            .await;
        if let Some(first) = conflicts.into_iter().next() {
            // Rewrite the placeholder `guide_a` (set to a sentinel by
            // `check_conflicts`) with the actual new guide id so callers can
            // identify both sides.
            let conflict = GuideConflict {
                guide_a: guide.id.clone(),
                guide_b: first.guide_b,
                reason: first.reason,
            };
            return Err(GuideRegistryError::ConflictDetected(conflict));
        }

        let id = guide.id.clone();
        self.inner.lock().unwrap().guides.insert(id.clone(), guide);
        Ok(id)
    }

    async fn select(&self, query: GuideQuery) -> Result<Vec<Guide>, GuideRegistryError> {
        let store = self.inner.lock().unwrap();
        let mut scored: Vec<(f32, Guide)> = store
            .guides
            .values()
            .filter(|g| matches!(g.status, GuideStatus::Active))
            .filter(|g| match (&query.domain, &g.domain) {
                (Some(want), Some(have)) => want == have,
                (Some(_), None) => false,
                _ => true,
            })
            .filter(|g| query.guide_types.is_empty() || query.guide_types.contains(&g.guide_type))
            .map(|g| (jaccard(&query.task_instruction, &g.content), g.clone()))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().map(|(_, g)| g).collect())
    }

    async fn record_usage(&self, record: GuideUsageRecord) -> Result<(), GuideRegistryError> {
        let mut store = self.inner.lock().unwrap();
        if !store.guides.contains_key(&record.guide_id) {
            return Err(GuideRegistryError::NotFound(record.guide_id));
        }
        // Update last_used on the guide.
        if let Some(g) = store.guides.get_mut(&record.guide_id) {
            g.last_used = Some(record.recorded_at.clone());
        }
        store.usage.push(record);
        Ok(())
    }

    async fn usage_history(
        &self,
        id: &GuideId,
    ) -> Result<Vec<GuideUsageRecord>, GuideRegistryError> {
        let store = self.inner.lock().unwrap();
        if !store.guides.contains_key(id) {
            return Err(GuideRegistryError::NotFound(id.clone()));
        }
        Ok(store
            .usage
            .iter()
            .filter(|r| &r.guide_id == id)
            .cloned()
            .collect())
    }

    async fn deprecate(&self, id: &GuideId, reason: &str) -> Result<(), GuideRegistryError> {
        let mut store = self.inner.lock().unwrap();
        let now = store.now_override.clone().unwrap_or_else(now_rfc3339);
        let g = store
            .guides
            .get_mut(id)
            .ok_or_else(|| GuideRegistryError::NotFound(id.clone()))?;
        g.status = GuideStatus::Deprecated {
            reason: reason.to_string(),
            at: now,
        };
        Ok(())
    }

    async fn mark_pending_review(
        &self,
        id: &GuideId,
        reason: PendingReason,
    ) -> Result<(), GuideRegistryError> {
        let mut store = self.inner.lock().unwrap();
        let now = store.now_override.clone().unwrap_or_else(now_rfc3339);
        let g = store
            .guides
            .get_mut(id)
            .ok_or_else(|| GuideRegistryError::NotFound(id.clone()))?;
        g.status = GuideStatus::PendingReview { reason, since: now };
        Ok(())
    }

    async fn promote_to_active(&self, id: &GuideId) -> Result<(), GuideRegistryError> {
        let mut store = self.inner.lock().unwrap();
        let g = store
            .guides
            .get_mut(id)
            .ok_or_else(|| GuideRegistryError::NotFound(id.clone()))?;
        if !matches!(g.status, GuideStatus::PendingReview { .. }) {
            return Err(GuideRegistryError::ValidationFailed {
                reason: "promote_to_active requires PendingReview status".into(),
            });
        }
        g.status = GuideStatus::Active;
        Ok(())
    }

    async fn analyze_performance(
        &self,
        window: Duration,
        min_failure_rate_delta: f32,
        min_pattern_occurrences: u32,
    ) -> Vec<ImprovementSignal> {
        let store = self.inner.lock().unwrap();
        let now = store.now_override.clone().unwrap_or_else(now_rfc3339);
        let cutoff = rfc3339_subtract(&now, window);

        let in_win: Vec<&GuideUsageRecord> = store
            .usage
            .iter()
            .filter(|r| in_window(&r.recorded_at, &cutoff))
            .collect();

        let mut signals = Vec::new();

        // Per-guide failure-rate vs baseline.
        let total_failures = in_win
            .iter()
            .filter(|r| matches!(r.outcome, SessionOutcome::Failure { .. }))
            .count();
        let total_records = in_win.len();

        for (gid, guide) in store.guides.iter() {
            // Conflict-derived signal.
            if let GuideStatus::PendingReview {
                reason: PendingReason::ConflictDetected { conflicts_with },
                ..
            } = &guide.status
            {
                if let Some(other) = conflicts_with.first() {
                    signals.push(ImprovementSignal::ConflictResolutionNeeded {
                        conflict: GuideConflict {
                            guide_a: gid.clone(),
                            guide_b: other.clone(),
                            reason: "pending-review conflict".into(),
                        },
                    });
                }
            }

            // Performance-delta signal: only fires for guides with >= 1 usage record.
            let with: Vec<&&GuideUsageRecord> =
                in_win.iter().filter(|r| &r.guide_id == gid).collect();
            if with.is_empty() {
                continue;
            }
            let with_fail = with
                .iter()
                .filter(|r| matches!(r.outcome, SessionOutcome::Failure { .. }))
                .count();
            let with_rate = with_fail as f32 / with.len() as f32;
            let without_count = total_records.saturating_sub(with.len());
            let baseline = if without_count == 0 {
                0.0
            } else {
                (total_failures - with_fail) as f32 / without_count as f32
            };
            if with_rate - baseline >= min_failure_rate_delta {
                signals.push(ImprovementSignal::GuideDeprecationRecommended {
                    guide_id: gid.clone(),
                    reason: format!(
                        "failure-rate delta {:.2} (with={:.2} vs baseline={:.2})",
                        with_rate - baseline,
                        with_rate,
                        baseline
                    ),
                });
            }
        }

        // Pattern detection: failure reasons appearing >= N times.
        let mut pattern_counts: HashMap<String, Vec<SessionId>> = HashMap::new();
        for r in &in_win {
            if let SessionOutcome::Failure { reason } = &r.outcome {
                pattern_counts
                    .entry(reason.clone())
                    .or_default()
                    .push(r.session_id.clone());
            }
        }
        for (pattern, sessions) in pattern_counts {
            if sessions.len() as u32 >= min_pattern_occurrences {
                signals.push(ImprovementSignal::SkillGenerationNeeded {
                    pattern,
                    session_ids: sessions,
                });
            }
        }

        signals
    }

    async fn check_conflicts(&self, content: &str, domain: Option<&str>) -> Vec<GuideConflict> {
        let store = self.inner.lock().unwrap();
        let mut out = Vec::new();
        for g in store.guides.values() {
            if !matches!(g.status, GuideStatus::Active) {
                continue;
            }
            // Same-domain only (None vs Some is not considered a conflict).
            if g.domain.as_deref() != domain {
                continue;
            }
            if g.content == content {
                continue;
            }
            let score = jaccard(&g.content, content);
            if score >= CONFLICT_THRESHOLD {
                out.push(GuideConflict {
                    // Placeholder; `register` rewrites this with the new id.
                    guide_a: GuideId::new("<new>"),
                    guide_b: g.id.clone(),
                    reason: format!("Jaccard overlap {score:.2} ≥ {CONFLICT_THRESHOLD}"),
                });
            }
        }
        out
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

    fn make_guide(id: &str, content: &str) -> Guide {
        Guide {
            id: GuideId::new(id),
            name: id.into(),
            content: content.into(),
            guide_type: GuideType::Skill,
            domain: None,
            source: GuideSource::Manual,
            status: GuideStatus::Active,
            created_at: ts("2026-05-16T00:00:00Z"),
            last_used: None,
            version: 1,
        }
    }

    fn usage(gid: &str, sid: &str, outcome: SessionOutcome) -> GuideUsageRecord {
        GuideUsageRecord {
            guide_id: GuideId::new(gid),
            session_id: SessionId::new(sid),
            task_domain: None,
            outcome,
            recorded_at: ts("2026-05-16T00:00:00Z"),
        }
    }

    // ── Rule: register validates content ────────────────────────────────────

    #[tokio::test]
    async fn register_empty_content_fails() {
        let r = StandardGuideRegistry::new();
        let mut g = make_guide("g1", "   ");
        g.content = "   ".into();
        let err = r.register(g).await.unwrap_err();
        assert!(matches!(err, GuideRegistryError::ValidationFailed { .. }));
    }

    // ── Rule: MetaAgentProposed forces PendingReview ────────────────────────

    #[tokio::test]
    async fn meta_agent_source_forces_pending_review() {
        let r = StandardGuideRegistry::new();
        let mut g = make_guide("g1", "proposed");
        g.source = GuideSource::MetaAgentProposed {
            proposed_at: ts("2026-05-16T01:00:00Z"),
        };
        g.status = GuideStatus::Active; // caller lies; provider corrects
        r.register(g).await.unwrap();
        let sel = r.select(GuideQuery::new("anything")).await.unwrap();
        // Not Active → not in select.
        assert!(sel.iter().all(|g| g.id.0 != "g1"));
        // mark_pending_review trail is correct.
        let store = r.inner.lock().unwrap();
        let g = &store.guides[&GuideId::new("g1")];
        assert!(matches!(
            g.status,
            GuideStatus::PendingReview {
                reason: PendingReason::AutomatedProposal,
                ..
            }
        ));
    }

    // ── Rule: select returns only Active, filtered by domain & type ─────────

    #[tokio::test]
    async fn select_filters_by_status_domain_and_type() {
        let r = StandardGuideRegistry::new();
        let mut a = make_guide("a", "rust async tokio runtime");
        a.domain = Some("rust".into());
        let mut b = make_guide("b", "pytest fixtures python");
        b.domain = Some("python".into());
        b.guide_type = GuideType::ConventionDoc;
        let mut c = make_guide("c", "deprecated content");
        c.status = GuideStatus::Deprecated {
            reason: "old".into(),
            at: ts("2026-05-16T00:00:00Z"),
        };
        r.register(a).await.unwrap();
        r.register(b).await.unwrap();
        // Inject c directly to bypass status validation in register.
        r.inner.lock().unwrap().guides.insert(GuideId::new("c"), c);

        let res = r
            .select(GuideQuery {
                task_instruction: "rust tokio".into(),
                domain: Some("rust".into()),
                phase: None,
                guide_types: vec![GuideType::Skill],
            })
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id.0, "a");
    }

    // ── Rule: select sorted by relevance ────────────────────────────────────

    #[tokio::test]
    async fn select_sorted_by_relevance() {
        let r = StandardGuideRegistry::new();
        r.register(make_guide("a", "alpha beta gamma delta"))
            .await
            .unwrap();
        r.register(make_guide("b", "zebra")).await.unwrap();
        r.register(make_guide("c", "alpha beta")).await.unwrap();
        let res = r.select(GuideQuery::new("alpha beta")).await.unwrap();
        // c is closer than a; b doesn't overlap so should be last.
        assert_eq!(res.first().unwrap().id.0, "c");
        assert_eq!(res.last().unwrap().id.0, "b");
    }

    // ── Rule: conflict detection at registration ────────────────────────────

    #[tokio::test]
    async fn register_detects_conflict_in_same_domain() {
        let r = StandardGuideRegistry::new();
        let mut existing = make_guide("a", "always run tests before commit");
        existing.domain = Some("rust".into());
        r.register(existing).await.unwrap();

        let mut conflicting = make_guide("b", "always run tests before committing");
        conflicting.domain = Some("rust".into());
        let err = r.register(conflicting).await.unwrap_err();
        match err {
            GuideRegistryError::ConflictDetected(c) => {
                assert_eq!(c.guide_a.0, "b");
                assert_eq!(c.guide_b.0, "a");
            }
            e => panic!("expected ConflictDetected, got {e:?}"),
        }
    }

    #[tokio::test]
    async fn no_conflict_across_domains() {
        let r = StandardGuideRegistry::new();
        let mut a = make_guide("a", "always run tests before commit");
        a.domain = Some("rust".into());
        r.register(a).await.unwrap();
        let mut b = make_guide("b", "always run tests before commit");
        b.domain = Some("python".into());
        r.register(b).await.unwrap();
    }

    // ── Rule: record_usage requires existing guide & updates last_used ──────

    #[tokio::test]
    async fn record_usage_requires_known_guide() {
        let r = StandardGuideRegistry::new();
        let err = r
            .record_usage(usage("nope", "s1", SessionOutcome::Success))
            .await
            .unwrap_err();
        assert!(matches!(err, GuideRegistryError::NotFound(_)));
    }

    #[tokio::test]
    async fn record_usage_updates_last_used() {
        let r = StandardGuideRegistry::new();
        r.register(make_guide("a", "x")).await.unwrap();
        let mut u = usage("a", "s1", SessionOutcome::Success);
        u.recorded_at = ts("2026-06-01T00:00:00Z");
        r.record_usage(u).await.unwrap();
        let hist = r.usage_history(&GuideId::new("a")).await.unwrap();
        assert_eq!(hist.len(), 1);
        let store = r.inner.lock().unwrap();
        assert_eq!(
            store.guides[&GuideId::new("a")]
                .last_used
                .as_ref()
                .unwrap()
                .0,
            "2026-06-01T00:00:00Z"
        );
    }

    // ── Rule: deprecate sets status ─────────────────────────────────────────

    #[tokio::test]
    async fn deprecate_sets_status_and_404s_on_missing() {
        let r = StandardGuideRegistry::new();
        r.register(make_guide("a", "x")).await.unwrap();
        r.deprecate(&GuideId::new("a"), "obsolete").await.unwrap();
        {
            let store = r.inner.lock().unwrap();
            match &store.guides[&GuideId::new("a")].status {
                GuideStatus::Deprecated { reason, .. } => assert_eq!(reason, "obsolete"),
                s => panic!("expected Deprecated, got {s:?}"),
            }
        }
        let err = r.deprecate(&GuideId::new("nope"), "x").await.unwrap_err();
        assert!(matches!(err, GuideRegistryError::NotFound(_)));
    }

    // ── Rule: promote_to_active only from PendingReview ─────────────────────

    #[tokio::test]
    async fn promote_to_active_only_from_pending_review() {
        let r = StandardGuideRegistry::new();
        r.register(make_guide("a", "x")).await.unwrap();
        // Active → promote should fail.
        let err = r.promote_to_active(&GuideId::new("a")).await.unwrap_err();
        assert!(matches!(err, GuideRegistryError::ValidationFailed { .. }));
        // Move to PendingReview, then promote.
        r.mark_pending_review(
            &GuideId::new("a"),
            PendingReason::ManualFlag { note: "x".into() },
        )
        .await
        .unwrap();
        r.promote_to_active(&GuideId::new("a")).await.unwrap();
        let store = r.inner.lock().unwrap();
        assert!(matches!(
            store.guides[&GuideId::new("a")].status,
            GuideStatus::Active
        ));
    }

    // ── Rule: analyze_performance flags high-delta failure rate ─────────────

    #[tokio::test]
    async fn analyze_performance_flags_high_failure_rate() {
        let r = StandardGuideRegistry::new();
        r.set_now(ts("2026-05-16T01:00:00Z"));
        r.register(make_guide("a", "x")).await.unwrap();
        r.register(make_guide("b", "y")).await.unwrap();
        // Guide a: 3 failures.
        for i in 0..3 {
            r.record_usage(usage(
                "a",
                &format!("s{i}"),
                SessionOutcome::Failure {
                    reason: "boom".into(),
                },
            ))
            .await
            .unwrap();
        }
        // Guide b: 3 successes.
        for i in 0..3 {
            r.record_usage(usage("b", &format!("sb{i}"), SessionOutcome::Success))
                .await
                .unwrap();
        }
        let signals = r
            .analyze_performance(Duration::from_secs(86_400), 0.5, 100)
            .await;
        // Guide a should be flagged for deprecation.
        let dep_for_a = signals.iter().any(|s| {
            matches!(
                s,
                ImprovementSignal::GuideDeprecationRecommended { guide_id, .. }
                    if guide_id.0 == "a"
            )
        });
        assert!(dep_for_a);
    }

    // ── Rule: analyze_performance emits SkillGenerationNeeded on pattern N ──

    #[tokio::test]
    async fn analyze_performance_emits_skill_generation_for_repeated_pattern() {
        let r = StandardGuideRegistry::new();
        r.set_now(ts("2026-05-16T01:00:00Z"));
        r.register(make_guide("a", "x")).await.unwrap();
        for i in 0..4 {
            r.record_usage(usage(
                "a",
                &format!("s{i}"),
                SessionOutcome::Failure {
                    reason: "panic: index out of bounds".into(),
                },
            ))
            .await
            .unwrap();
        }
        let signals = r
            .analyze_performance(Duration::from_secs(86_400), 999.0, 3)
            .await;
        let gen = signals.iter().any(|s| {
            matches!(
                s,
                ImprovementSignal::SkillGenerationNeeded { pattern, session_ids }
                    if pattern == "panic: index out of bounds" && session_ids.len() == 4
            )
        });
        assert!(gen);
    }

    // ── Rule: analyze_performance honors window ─────────────────────────────

    #[tokio::test]
    async fn analyze_performance_filters_by_window() {
        let r = StandardGuideRegistry::new();
        r.set_now(ts("2026-05-16T00:00:00Z"));
        r.register(make_guide("a", "x")).await.unwrap();
        // One very old failure.
        let mut old = usage(
            "a",
            "s0",
            SessionOutcome::Failure {
                reason: "old-pattern".into(),
            },
        );
        old.recorded_at = ts("2020-01-01T00:00:00Z");
        r.record_usage(old).await.unwrap();
        let signals = r
            .analyze_performance(Duration::from_secs(3600), 0.0, 1)
            .await;
        // Old record outside the 1-hour window — no skill signal.
        let any_pattern = signals
            .iter()
            .any(|s| matches!(s, ImprovementSignal::SkillGenerationNeeded { .. }));
        assert!(!any_pattern);
    }

    // ── Rule: usage_history returns only that guide's records ───────────────

    #[tokio::test]
    async fn usage_history_filters_to_one_guide() {
        let r = StandardGuideRegistry::new();
        r.register(make_guide("a", "x")).await.unwrap();
        r.register(make_guide("b", "y")).await.unwrap();
        r.record_usage(usage("a", "s1", SessionOutcome::Success))
            .await
            .unwrap();
        r.record_usage(usage("b", "s2", SessionOutcome::Success))
            .await
            .unwrap();
        let h = r.usage_history(&GuideId::new("a")).await.unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].guide_id.0, "a");
    }

    // ── Rule: check_conflicts external API ──────────────────────────────────

    #[tokio::test]
    async fn check_conflicts_does_not_flag_identical_content() {
        let r = StandardGuideRegistry::new();
        r.register(make_guide("a", "same exact content"))
            .await
            .unwrap();
        let conflicts = r.check_conflicts("same exact content", None).await;
        assert!(conflicts.is_empty());
    }

    // ── Rule: select returns empty when no guides ───────────────────────────

    #[tokio::test]
    async fn select_empty_when_no_guides() {
        let r = StandardGuideRegistry::new();
        let res = r.select(GuideQuery::new("anything")).await.unwrap();
        assert!(res.is_empty());
    }

    // ── Date arithmetic sanity ──────────────────────────────────────────────

    #[test]
    fn rfc3339_round_trip() {
        let s = parse_rfc3339_secs("2026-05-16T12:34:56Z").unwrap();
        let f = format_rfc3339_secs(s);
        assert_eq!(f, "2026-05-16T12:34:56Z");
    }
}
