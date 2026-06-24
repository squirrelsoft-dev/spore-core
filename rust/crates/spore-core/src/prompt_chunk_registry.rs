//! Issue #24 — `PromptChunkRegistry`: named, cacheable prompt chunks composed
//! into a deterministic system prompt at harness construction time.
//!
//! See `docs/harness-engineering-concepts.md` § "Cache Architecture" and the
//! spec issue for the rules this module enforces.
//!
//! ## Rules enforced
//!   - Each chunk has a unique `ChunkId`. Duplicate registration is
//!     `ChunkError::DuplicateId`.
//!   - `ChunkSlot::Budget` and `ChunkSlot::Ephemeral` are always `PerTurn`.
//!     Registering with a different cache block is
//!     `ChunkError::ConflictingCacheBlock`.
//!   - `ChunkSlot::Role` and `ChunkSlot::Mode` are always `Static`. Anything
//!     else is `ChunkError::ConflictingCacheBlock`.
//!   - Chunk content must not be empty (`ChunkError::InvalidSlot`).
//!   - `compose()` requires the named role chunk to be registered and resolves
//!     the mode chunk via [`Mode::prompt_chunk`]. Capability and skill chunks
//!     must already be registered.
//!   - The composed chunk list is ordered by slot (Role → Mode → Capability →
//!     Skill → Task → Environment → PriorSession → Budget → Ephemeral) and,
//!     within a slot, by registration order.
//!   - `block_1_hash` is the FxHash digest of all `Static` chunk contents.
//!     `block_2_hash` is the digest of all `PerSession` chunk contents.
//!   - `validate()` rejects compositions with a `PerTurn` chunk in Block 1,
//!     missing required slots (Role and Mode), or more than one Mode chunk.
//!   - Mode is permanent for the life of a harness instance — there is no
//!     mutation API. Changing mode means building a new harness.
//!
//! ## `dangerous` feature gate
//!
//! `Mode::Yolo` (full autonomy, no approval gates) is a named safety footgun.
//! It is only compiled when the `dangerous` Cargo feature is enabled. In the
//! default build the variant does not exist, so using it is a compile error
//! rather than a runtime warning. The wire tag stays `"yolo"`. See issue #34.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::tool_registry::TaskPhase;

// ============================================================================
// Identity
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChunkId(pub String);

impl ChunkId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ChunkId {
    fn from(s: &str) -> Self {
        ChunkId(s.to_string())
    }
}

impl From<String> for ChunkId {
    fn from(s: String) -> Self {
        ChunkId(s)
    }
}

// ============================================================================
// Enums
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkSlot {
    Role,
    Mode,
    Capability,
    Skill,
    Task,
    Environment,
    PriorSession,
    Budget,
    Ephemeral,
}

impl ChunkSlot {
    /// Render ordering used by [`PromptChunkRegistry::compose`].
    fn order(&self) -> u8 {
        match self {
            ChunkSlot::Role => 0,
            ChunkSlot::Mode => 1,
            ChunkSlot::Capability => 2,
            ChunkSlot::Skill => 3,
            ChunkSlot::Task => 4,
            ChunkSlot::Environment => 5,
            ChunkSlot::PriorSession => 6,
            ChunkSlot::Budget => 7,
            ChunkSlot::Ephemeral => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheBlock {
    Static,
    PerSession,
    PerTurn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalPolicy {
    /// Every action requires approval before execution.
    AlwaysAsk,
    /// Actions proceed automatically; the agent narrates afterwards.
    AutoExplain,
    /// Planning only — file edits are blocked until the user confirms.
    PlanOnly,
    /// Auto for Low/Medium; High/Critical require approval (middleware).
    SafeAuto,
    /// Full autonomy — no approval gates.
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    AlwaysAsk,
    AutoEdit,
    Plan,
    SafeAuto,
    /// Full autonomy — no approval gates. Gated behind the `dangerous` feature
    /// (issue #34); absent from the default build.
    #[cfg(feature = "dangerous")]
    Yolo,
}

impl Mode {
    /// The standard prompt chunk for this mode. Lands in `ChunkSlot::Mode`
    /// in Block 1. Always `Static`.
    pub fn prompt_chunk(&self) -> PromptChunk {
        let (id, content) = match self {
            Mode::AlwaysAsk => (
                "mode-always-ask",
                "Mode: AlwaysAsk. Describe your plan and wait for explicit approval before taking any action.",
            ),
            Mode::AutoEdit => (
                "mode-auto-edit",
                "Mode: AutoEdit. Edit files freely. Explain the changes you make after they are done.",
            ),
            Mode::Plan => (
                "mode-plan",
                "Mode: Plan. Produce a plan only. Do not edit files or execute mutating tools.",
            ),
            Mode::SafeAuto => (
                "mode-safe-auto",
                "Mode: SafeAuto. Auto-execute Low and Medium risk actions. High and Critical actions require approval.",
            ),
            #[cfg(feature = "dangerous")]
            Mode::Yolo => (
                "mode-yolo",
                "Mode: Yolo. Full autonomy. No approval gates.",
            ),
        };
        PromptChunk::new(id, content, ChunkSlot::Mode, CacheBlock::Static)
    }

    /// Enforcement policy this mode implies. Used by `PermissionMiddleware`.
    pub fn approval_policy(&self) -> ApprovalPolicy {
        match self {
            Mode::AlwaysAsk => ApprovalPolicy::AlwaysAsk,
            Mode::AutoEdit => ApprovalPolicy::AutoExplain,
            Mode::Plan => ApprovalPolicy::PlanOnly,
            Mode::SafeAuto => ApprovalPolicy::SafeAuto,
            #[cfg(feature = "dangerous")]
            Mode::Yolo => ApprovalPolicy::None,
        }
    }

    /// Initial task phase implied by the mode.
    pub fn default_tool_phase(&self) -> TaskPhase {
        match self {
            Mode::Plan => TaskPhase::Planning,
            _ => TaskPhase::Execution,
        }
    }
}

// ============================================================================
// Records
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptChunk {
    pub id: ChunkId,
    pub content: String,
    pub slot: ChunkSlot,
    pub cache_block: CacheBlock,
    pub hash: u64,
}

impl PromptChunk {
    /// Build a chunk and compute its content hash. Use this instead of the
    /// struct literal so the hash stays in sync with `content`.
    pub fn new(
        id: impl Into<ChunkId>,
        content: impl Into<String>,
        slot: ChunkSlot,
        cache_block: CacheBlock,
    ) -> Self {
        let content = content.into();
        let hash = hash_content(&content);
        Self {
            id: id.into(),
            content,
            slot,
            cache_block,
            hash,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposedPrompt {
    pub chunks: Vec<PromptChunk>,
    pub block_1_hash: u64,
    pub block_2_hash: u64,
    /// Cached render of all chunks joined by `\n\n`. `None` until materialized
    /// by `ContextManager.assemble()`. Invalidated when any chunk hash changes.
    pub rendered: Option<String>,
}

impl ComposedPrompt {
    /// Render the chunk list deterministically. Caches result in `rendered`.
    pub fn render(&mut self) -> &str {
        if self.rendered.is_none() {
            let rendered = self
                .chunks
                .iter()
                .map(|c| c.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            self.rendered = Some(rendered);
        }
        self.rendered.as_deref().unwrap_or("")
    }

    /// Returns the cached render or the empty string when not yet rendered.
    /// Useful in read-only contexts where mutating is not appropriate.
    pub fn rendered_str(&self) -> &str {
        self.rendered.as_deref().unwrap_or("")
    }

    /// Recompute block hashes from current chunk state. Use after any chunk
    /// content change to confirm whether the cache should be invalidated.
    pub fn recompute_hashes(&self) -> (u64, u64) {
        compute_block_hashes(&self.chunks)
    }
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChunkError {
    #[error("duplicate chunk id: {id:?}")]
    DuplicateId { id: ChunkId },
    #[error("invalid slot for chunk {id:?}: {reason}")]
    InvalidSlot { id: ChunkId, reason: String },
    #[error(
        "conflicting cache block for chunk {id:?} in slot {slot:?}: expected {expected:?}, got {actual:?}"
    )]
    ConflictingCacheBlock {
        id: ChunkId,
        slot: ChunkSlot,
        expected: CacheBlock,
        actual: CacheBlock,
    },
    #[error("chunk not found: {id:?}")]
    NotFound { id: ChunkId },
}

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChunkValidationError {
    #[error("per-turn chunk {id:?} placed in the Static block")]
    PerTurnChunkInStaticBlock { id: ChunkId },
    #[error("required slot {slot:?} is missing from the composition")]
    MissingRequiredSlot { slot: ChunkSlot },
    #[error("more than one Mode chunk in the composition: {ids:?}")]
    ConflictingModeChunks { ids: Vec<ChunkId> },
}

// ============================================================================
// Trait
// ============================================================================

pub trait PromptChunkRegistry: Send + Sync {
    /// Register a chunk. Validates slot/cache-block compatibility before
    /// storing.
    fn register(&self, chunk: PromptChunk) -> Result<(), ChunkError>;

    /// Compose chunks for a given agent configuration. Called once at harness
    /// construction; the result is cached on the harness instance.
    fn compose(
        &self,
        role: ChunkId,
        mode: Mode,
        capabilities: Vec<ChunkId>,
        skills: Vec<ChunkId>,
    ) -> Result<ComposedPrompt, Vec<ChunkValidationError>>;

    /// Validate a composition. Returns an empty Vec when valid.
    fn validate(&self, composed: &ComposedPrompt) -> Vec<ChunkValidationError>;

    /// Look up a chunk by id.
    fn get(&self, id: &ChunkId) -> Option<PromptChunk>;
}

// ============================================================================
// Standard implementation
// ============================================================================

/// Reference, in-memory `PromptChunkRegistry`. The harness owns a single
/// shared instance; chunks are typically registered at startup and never
/// mutated afterwards.
#[derive(Default)]
pub struct StandardPromptChunkRegistry {
    inner: Mutex<Store>,
}

#[derive(Default)]
struct Store {
    chunks: HashMap<ChunkId, PromptChunk>,
    /// Order in which chunk ids were registered. Drives intra-slot ordering
    /// during compose().
    order: Vec<ChunkId>,
}

impl StandardPromptChunkRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register every chunk in the standard library: role personas, the five
    /// mode chunks, common capabilities, and behavioral skills. Returns the
    /// first registration error if any chunk fails validation (e.g. when this
    /// registry already has a colliding id).
    pub fn register_standard_chunks(&self) -> Result<(), ChunkError> {
        for chunk in standard_chunks() {
            self.register(chunk)?;
        }
        Ok(())
    }
}

impl PromptChunkRegistry for StandardPromptChunkRegistry {
    fn register(&self, chunk: PromptChunk) -> Result<(), ChunkError> {
        validate_slot_and_cache_block(&chunk)?;
        let mut store = self.inner.lock().unwrap();
        if store.chunks.contains_key(&chunk.id) {
            return Err(ChunkError::DuplicateId {
                id: chunk.id.clone(),
            });
        }
        store.order.push(chunk.id.clone());
        store.chunks.insert(chunk.id.clone(), chunk);
        Ok(())
    }

    fn compose(
        &self,
        role: ChunkId,
        mode: Mode,
        capabilities: Vec<ChunkId>,
        skills: Vec<ChunkId>,
    ) -> Result<ComposedPrompt, Vec<ChunkValidationError>> {
        let store = self.inner.lock().unwrap();

        let mut errors: Vec<ChunkValidationError> = Vec::new();
        let mut chosen: Vec<PromptChunk> = Vec::new();
        let mut seen: HashMap<ChunkId, ()> = HashMap::new();

        // Role
        match store.chunks.get(&role) {
            Some(c) if c.slot == ChunkSlot::Role => {
                seen.insert(c.id.clone(), ());
                chosen.push(c.clone());
            }
            Some(_) => {
                errors.push(ChunkValidationError::MissingRequiredSlot {
                    slot: ChunkSlot::Role,
                });
            }
            None => {
                errors.push(ChunkValidationError::MissingRequiredSlot {
                    slot: ChunkSlot::Role,
                });
            }
        }

        // Mode — always sourced from the enum.
        let mode_chunk = mode.prompt_chunk();
        seen.insert(mode_chunk.id.clone(), ());
        chosen.push(mode_chunk);

        // Capabilities
        for id in capabilities {
            match store.chunks.get(&id) {
                Some(c) if c.slot == ChunkSlot::Capability => {
                    seen.insert(c.id.clone(), ());
                    chosen.push(c.clone());
                }
                _ => {
                    errors.push(ChunkValidationError::MissingRequiredSlot {
                        slot: ChunkSlot::Capability,
                    });
                }
            }
        }

        // Skills
        for id in skills {
            match store.chunks.get(&id) {
                Some(c) if c.slot == ChunkSlot::Skill => {
                    seen.insert(c.id.clone(), ());
                    chosen.push(c.clone());
                }
                _ => {
                    errors.push(ChunkValidationError::MissingRequiredSlot {
                        slot: ChunkSlot::Skill,
                    });
                }
            }
        }

        if !errors.is_empty() {
            return Err(errors);
        }

        // Stable sort by slot order; within a slot preserve insertion order
        // (capabilities and skills follow the caller-provided sequence).
        chosen.sort_by_key(|c| c.slot.order());

        let (block_1_hash, block_2_hash) = compute_block_hashes(&chosen);

        let composed = ComposedPrompt {
            chunks: chosen,
            block_1_hash,
            block_2_hash,
            rendered: None,
        };

        let v_errors = self.validate(&composed);
        if !v_errors.is_empty() {
            return Err(v_errors);
        }

        Ok(composed)
    }

    fn validate(&self, composed: &ComposedPrompt) -> Vec<ChunkValidationError> {
        let mut errors = Vec::new();

        // Block 1 must not contain PerTurn chunks.
        for c in &composed.chunks {
            if c.cache_block == CacheBlock::Static
                && matches!(c.slot, ChunkSlot::Budget | ChunkSlot::Ephemeral)
            {
                errors.push(ChunkValidationError::PerTurnChunkInStaticBlock { id: c.id.clone() });
            }
        }

        // Required slots: Role and Mode.
        for required in [ChunkSlot::Role, ChunkSlot::Mode] {
            if !composed.chunks.iter().any(|c| c.slot == required) {
                errors.push(ChunkValidationError::MissingRequiredSlot { slot: required });
            }
        }

        // Exactly one Mode chunk.
        let mode_ids: Vec<ChunkId> = composed
            .chunks
            .iter()
            .filter(|c| c.slot == ChunkSlot::Mode)
            .map(|c| c.id.clone())
            .collect();
        if mode_ids.len() > 1 {
            errors.push(ChunkValidationError::ConflictingModeChunks { ids: mode_ids });
        }

        errors
    }

    fn get(&self, id: &ChunkId) -> Option<PromptChunk> {
        self.inner.lock().unwrap().chunks.get(id).cloned()
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn hash_content(content: &str) -> u64 {
    let mut h = DefaultHasher::new();
    content.hash(&mut h);
    h.finish()
}

fn compute_block_hashes(chunks: &[PromptChunk]) -> (u64, u64) {
    // Group by cache block, then by id (deterministic) — never rely on
    // HashMap iteration order.
    let mut block_1 = BTreeMap::new();
    let mut block_2 = BTreeMap::new();
    for c in chunks {
        match c.cache_block {
            CacheBlock::Static => {
                block_1.insert(c.id.clone(), c.hash);
            }
            CacheBlock::PerSession => {
                block_2.insert(c.id.clone(), c.hash);
            }
            CacheBlock::PerTurn => {}
        }
    }
    let mut h1 = DefaultHasher::new();
    for (id, hash) in &block_1 {
        id.0.hash(&mut h1);
        hash.hash(&mut h1);
    }
    let mut h2 = DefaultHasher::new();
    for (id, hash) in &block_2 {
        id.0.hash(&mut h2);
        hash.hash(&mut h2);
    }
    (h1.finish(), h2.finish())
}

fn validate_slot_and_cache_block(chunk: &PromptChunk) -> Result<(), ChunkError> {
    if chunk.content.trim().is_empty() {
        return Err(ChunkError::InvalidSlot {
            id: chunk.id.clone(),
            reason: "content must not be empty".into(),
        });
    }
    match chunk.slot {
        ChunkSlot::Budget | ChunkSlot::Ephemeral if chunk.cache_block != CacheBlock::PerTurn => {
            return Err(ChunkError::ConflictingCacheBlock {
                id: chunk.id.clone(),
                slot: chunk.slot,
                expected: CacheBlock::PerTurn,
                actual: chunk.cache_block,
            });
        }
        ChunkSlot::Role | ChunkSlot::Mode if chunk.cache_block != CacheBlock::Static => {
            return Err(ChunkError::ConflictingCacheBlock {
                id: chunk.id.clone(),
                slot: chunk.slot,
                expected: CacheBlock::Static,
                actual: chunk.cache_block,
            });
        }
        _ => {}
    }
    Ok(())
}

// ============================================================================
// Standard chunk library
// ============================================================================

/// The standard chunk library shipped by `spore-core`. Users register
/// additional chunks for their domain.
pub fn standard_chunks() -> Vec<PromptChunk> {
    let mut out = Vec::new();
    // Roles
    for (id, content) in [
        (
            "role-coding-agent",
            "You are an expert software engineer. Read carefully, change deliberately, and verify your work.",
        ),
        (
            "role-evaluator",
            "You are a fresh evaluator. You did not write the code under review. Default to FAIL.",
        ),
        (
            "role-planner",
            "You are a planning specialist. Decompose tasks into small, verifiable steps.",
        ),
        (
            "role-rag-assistant",
            "You are a retrieval-augmented assistant. Always cite the source document for any claim.",
        ),
        (
            "role-sql-agent",
            "You are a SQL specialist. Prefer read-only queries; never DROP without explicit approval.",
        ),
    ] {
        out.push(PromptChunk::new(
            id,
            content,
            ChunkSlot::Role,
            CacheBlock::Static,
        ));
    }

    // Modes — derived from the enum so prompt_chunk() and standard_chunks()
    // are guaranteed to agree.
    for mode in [Mode::AlwaysAsk, Mode::AutoEdit, Mode::Plan, Mode::SafeAuto] {
        out.push(mode.prompt_chunk());
    }
    #[cfg(feature = "dangerous")]
    out.push(Mode::Yolo.prompt_chunk());

    // Capabilities
    for (id, content) in [
        (
            "capability-bash",
            "Capability: bash. You can run shell commands inside the sandbox.",
        ),
        (
            "capability-filesystem",
            "Capability: filesystem. You can read and write files inside the workspace.",
        ),
        (
            "capability-git",
            "Capability: git. You can stage, commit, and inspect history.",
        ),
        (
            "capability-browser",
            "Capability: browser. You can fetch web pages and follow links.",
        ),
        (
            "capability-subagent",
            "Capability: subagent. You can delegate work to a child harness.",
        ),
        (
            "capability-sql",
            "Capability: sql. You can issue queries against the configured database.",
        ),
    ] {
        out.push(PromptChunk::new(
            id,
            content,
            ChunkSlot::Capability,
            CacheBlock::Static,
        ));
    }

    // Skills
    for (id, content) in [
        (
            "skill-testing",
            "Skill: always run the test suite after changes and report results.",
        ),
        (
            "skill-decomposition",
            "Skill: break large tasks into small, independently verifiable steps.",
        ),
        (
            "skill-security-review",
            "Skill: review changes for injection, auth, and secret-leak issues before commit.",
        ),
        (
            "skill-citation",
            "Skill: cite the source document for every claim drawn from retrieved context.",
        ),
    ] {
        out.push(PromptChunk::new(
            id,
            content,
            ChunkSlot::Skill,
            CacheBlock::Static,
        ));
    }

    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_with_role(id: &str) -> StandardPromptChunkRegistry {
        let r = StandardPromptChunkRegistry::new();
        r.register(PromptChunk::new(
            id,
            "you are a test agent",
            ChunkSlot::Role,
            CacheBlock::Static,
        ))
        .unwrap();
        r
    }

    // ── Rule: register rejects duplicate ids ─────────────────────────────────

    #[test]
    fn register_rejects_duplicate_id() {
        let r = StandardPromptChunkRegistry::new();
        r.register(PromptChunk::new(
            "x",
            "hello",
            ChunkSlot::Capability,
            CacheBlock::Static,
        ))
        .unwrap();
        let err = r
            .register(PromptChunk::new(
                "x",
                "world",
                ChunkSlot::Capability,
                CacheBlock::Static,
            ))
            .unwrap_err();
        assert!(matches!(err, ChunkError::DuplicateId { .. }));
    }

    // ── Rule: register rejects empty content ─────────────────────────────────

    #[test]
    fn register_rejects_empty_content() {
        let r = StandardPromptChunkRegistry::new();
        let err = r
            .register(PromptChunk::new(
                "x",
                "   ",
                ChunkSlot::Capability,
                CacheBlock::Static,
            ))
            .unwrap_err();
        assert!(matches!(err, ChunkError::InvalidSlot { .. }));
    }

    // ── Rule: Budget/Ephemeral slots must be PerTurn ─────────────────────────

    #[test]
    fn budget_slot_rejects_static_cache_block() {
        let r = StandardPromptChunkRegistry::new();
        let err = r
            .register(PromptChunk::new(
                "b",
                "budget warning",
                ChunkSlot::Budget,
                CacheBlock::Static,
            ))
            .unwrap_err();
        match err {
            ChunkError::ConflictingCacheBlock {
                slot,
                expected,
                actual,
                ..
            } => {
                assert_eq!(slot, ChunkSlot::Budget);
                assert_eq!(expected, CacheBlock::PerTurn);
                assert_eq!(actual, CacheBlock::Static);
            }
            e => panic!("expected ConflictingCacheBlock, got {e:?}"),
        }
    }

    #[test]
    fn ephemeral_slot_rejects_per_session_cache_block() {
        let r = StandardPromptChunkRegistry::new();
        let err = r
            .register(PromptChunk::new(
                "e",
                "ephemeral",
                ChunkSlot::Ephemeral,
                CacheBlock::PerSession,
            ))
            .unwrap_err();
        assert!(matches!(err, ChunkError::ConflictingCacheBlock { .. }));
    }

    // ── Rule: Role/Mode slots must be Static ────────────────────────────────

    #[test]
    fn role_slot_rejects_non_static_cache_block() {
        let r = StandardPromptChunkRegistry::new();
        let err = r
            .register(PromptChunk::new(
                "r",
                "role",
                ChunkSlot::Role,
                CacheBlock::PerSession,
            ))
            .unwrap_err();
        assert!(matches!(err, ChunkError::ConflictingCacheBlock { .. }));
    }

    // ── Rule: compose requires the role chunk to exist ──────────────────────

    #[test]
    fn compose_missing_role_returns_error() {
        let r = StandardPromptChunkRegistry::new();
        let err = r
            .compose(ChunkId::new("missing"), Mode::SafeAuto, vec![], vec![])
            .unwrap_err();
        assert!(err
            .iter()
            .any(|e| matches!(e, ChunkValidationError::MissingRequiredSlot { slot } if *slot == ChunkSlot::Role)));
    }

    // ── Rule: compose includes the mode chunk via Mode::prompt_chunk() ──────

    #[test]
    fn compose_includes_mode_chunk_from_enum() {
        let r = registry_with_role("role-test");
        let composed = r
            .compose(ChunkId::new("role-test"), Mode::Plan, vec![], vec![])
            .unwrap();
        let mode_chunk = composed
            .chunks
            .iter()
            .find(|c| c.slot == ChunkSlot::Mode)
            .unwrap();
        assert_eq!(mode_chunk.id.0, "mode-plan");
    }

    // ── Rule: compose orders chunks by slot ──────────────────────────────────

    #[test]
    fn compose_orders_by_slot() {
        let r = registry_with_role("role-test");
        r.register(PromptChunk::new(
            "cap-1",
            "cap one",
            ChunkSlot::Capability,
            CacheBlock::Static,
        ))
        .unwrap();
        r.register(PromptChunk::new(
            "skill-1",
            "skill one",
            ChunkSlot::Skill,
            CacheBlock::Static,
        ))
        .unwrap();
        let composed = r
            .compose(
                ChunkId::new("role-test"),
                Mode::AutoEdit,
                vec![ChunkId::new("cap-1")],
                vec![ChunkId::new("skill-1")],
            )
            .unwrap();
        let slots: Vec<ChunkSlot> = composed.chunks.iter().map(|c| c.slot).collect();
        assert_eq!(
            slots,
            vec![
                ChunkSlot::Role,
                ChunkSlot::Mode,
                ChunkSlot::Capability,
                ChunkSlot::Skill,
            ]
        );
    }

    // ── Rule: block_1_hash and block_2_hash reflect chunk contents ──────────

    #[test]
    fn block_hashes_are_stable_for_identical_content() {
        let r1 = registry_with_role("role-test");
        let r2 = registry_with_role("role-test");
        let a = r1
            .compose(ChunkId::new("role-test"), Mode::SafeAuto, vec![], vec![])
            .unwrap();
        let b = r2
            .compose(ChunkId::new("role-test"), Mode::SafeAuto, vec![], vec![])
            .unwrap();
        assert_eq!(a.block_1_hash, b.block_1_hash);
        assert_eq!(a.block_2_hash, b.block_2_hash);
    }

    #[test]
    fn block_1_hash_changes_when_content_changes() {
        let a = {
            let r = registry_with_role("role-test");
            r.compose(ChunkId::new("role-test"), Mode::SafeAuto, vec![], vec![])
                .unwrap()
        };
        let b = {
            let r = StandardPromptChunkRegistry::new();
            r.register(PromptChunk::new(
                "role-test",
                "DIFFERENT ROLE CONTENT",
                ChunkSlot::Role,
                CacheBlock::Static,
            ))
            .unwrap();
            r.compose(ChunkId::new("role-test"), Mode::SafeAuto, vec![], vec![])
                .unwrap()
        };
        assert_ne!(a.block_1_hash, b.block_1_hash);
    }

    // ── Rule: validate flags PerTurn chunk in Static block ──────────────────

    #[test]
    fn validate_flags_perturn_chunk_in_static_block() {
        let r = StandardPromptChunkRegistry::new();
        // Build a composition by hand and feed it to validate().
        let composed = ComposedPrompt {
            chunks: vec![
                PromptChunk::new("role-x", "x", ChunkSlot::Role, CacheBlock::Static),
                Mode::SafeAuto.prompt_chunk(),
                // Budget chunk with Static cache block — simulates a bug.
                PromptChunk {
                    id: ChunkId::new("bad-budget"),
                    content: "b".into(),
                    slot: ChunkSlot::Budget,
                    cache_block: CacheBlock::Static,
                    hash: 0,
                },
            ],
            block_1_hash: 0,
            block_2_hash: 0,
            rendered: None,
        };
        let errors = r.validate(&composed);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ChunkValidationError::PerTurnChunkInStaticBlock { id } if id.0 == "bad-budget")));
    }

    // ── Rule: validate flags conflicting Mode chunks ────────────────────────

    #[test]
    fn validate_flags_more_than_one_mode_chunk() {
        let r = StandardPromptChunkRegistry::new();
        let composed = ComposedPrompt {
            chunks: vec![
                PromptChunk::new("role-x", "x", ChunkSlot::Role, CacheBlock::Static),
                Mode::SafeAuto.prompt_chunk(),
                Mode::AlwaysAsk.prompt_chunk(),
            ],
            block_1_hash: 0,
            block_2_hash: 0,
            rendered: None,
        };
        let errors = r.validate(&composed);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ChunkValidationError::ConflictingModeChunks { .. })));
    }

    // ── Rule: validate flags missing required slots ─────────────────────────

    #[test]
    fn validate_flags_missing_role_slot() {
        let r = StandardPromptChunkRegistry::new();
        let composed = ComposedPrompt {
            chunks: vec![Mode::SafeAuto.prompt_chunk()],
            block_1_hash: 0,
            block_2_hash: 0,
            rendered: None,
        };
        let errors = r.validate(&composed);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ChunkValidationError::MissingRequiredSlot { slot } if *slot == ChunkSlot::Role)));
    }

    // ── Rule: get returns registered chunks ─────────────────────────────────

    #[test]
    fn get_returns_registered_chunk() {
        let r = registry_with_role("role-x");
        let c = r.get(&ChunkId::new("role-x")).unwrap();
        assert_eq!(c.id.0, "role-x");
        assert!(r.get(&ChunkId::new("nope")).is_none());
    }

    // ── Mode helpers ────────────────────────────────────────────────────────

    #[test]
    fn mode_approval_policy_matches_spec() {
        assert_eq!(Mode::AlwaysAsk.approval_policy(), ApprovalPolicy::AlwaysAsk);
        assert_eq!(
            Mode::AutoEdit.approval_policy(),
            ApprovalPolicy::AutoExplain
        );
        assert_eq!(Mode::Plan.approval_policy(), ApprovalPolicy::PlanOnly);
        assert_eq!(Mode::SafeAuto.approval_policy(), ApprovalPolicy::SafeAuto);
        #[cfg(feature = "dangerous")]
        assert_eq!(Mode::Yolo.approval_policy(), ApprovalPolicy::None);
    }

    #[test]
    fn mode_default_tool_phase_plan_is_planning() {
        assert_eq!(Mode::Plan.default_tool_phase(), TaskPhase::Planning);
        #[cfg(feature = "dangerous")]
        assert_eq!(Mode::Yolo.default_tool_phase(), TaskPhase::Execution);
    }

    // ── ComposedPrompt rendering ────────────────────────────────────────────

    #[test]
    fn composed_prompt_render_joins_chunks_with_blank_line() {
        let r = registry_with_role("role-test");
        let mut composed = r
            .compose(ChunkId::new("role-test"), Mode::SafeAuto, vec![], vec![])
            .unwrap();
        assert!(composed.rendered.is_none());
        let rendered = composed.render().to_string();
        assert!(rendered.contains("you are a test agent"));
        assert!(rendered.contains("Mode: SafeAuto"));
        assert!(composed.rendered.is_some());
    }

    // ── Standard library bootstraps cleanly ─────────────────────────────────

    #[test]
    fn standard_chunks_register_cleanly() {
        let r = StandardPromptChunkRegistry::new();
        r.register_standard_chunks().unwrap();
        assert!(r.get(&ChunkId::new("role-coding-agent")).is_some());
        assert!(r.get(&ChunkId::new("capability-bash")).is_some());
        assert!(r.get(&ChunkId::new("skill-testing")).is_some());
    }

    // ── Fixture-replay: cross-language consistency ─────────────────────────

    #[derive(Deserialize)]
    struct FixtureFile {
        cases: Vec<FixtureCase>,
    }

    #[derive(Deserialize)]
    struct FixtureCase {
        name: String,
        register_inputs: Vec<FixtureChunk>,
        compose: FixtureCompose,
        expected_chunks: Vec<FixtureExpected>,
        rendered_contains: Vec<String>,
    }

    #[derive(Deserialize)]
    struct FixtureChunk {
        id: String,
        content: String,
        slot: ChunkSlot,
        cache_block: CacheBlock,
    }

    #[derive(Deserialize)]
    struct FixtureCompose {
        role: String,
        mode: Mode,
        capabilities: Vec<String>,
        skills: Vec<String>,
    }

    #[derive(Deserialize)]
    struct FixtureExpected {
        slot: ChunkSlot,
        id: String,
    }

    fn run_fixture_file(rel_path: &str) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel_path);
        let bytes = std::fs::read(&path).expect("read fixture file");
        let file: FixtureFile = serde_json::from_slice(&bytes).expect("parse fixture json");
        for case in file.cases {
            let r = StandardPromptChunkRegistry::new();
            for c in case.register_inputs {
                r.register(PromptChunk::new(c.id, c.content, c.slot, c.cache_block))
                    .unwrap_or_else(|e| panic!("[{}] register failed: {e:?}", case.name));
            }
            let mut composed = r
                .compose(
                    ChunkId::new(case.compose.role),
                    case.compose.mode,
                    case.compose
                        .capabilities
                        .into_iter()
                        .map(ChunkId::new)
                        .collect(),
                    case.compose.skills.into_iter().map(ChunkId::new).collect(),
                )
                .unwrap_or_else(|e| panic!("[{}] compose failed: {e:?}", case.name));
            let actual: Vec<(ChunkSlot, String)> = composed
                .chunks
                .iter()
                .map(|c| (c.slot, c.id.0.clone()))
                .collect();
            let expected: Vec<(ChunkSlot, String)> = case
                .expected_chunks
                .into_iter()
                .map(|e| (e.slot, e.id))
                .collect();
            assert_eq!(actual, expected, "[{}] composed chunks mismatch", case.name);
            assert!(
                r.validate(&composed).is_empty(),
                "[{}] validate should pass",
                case.name
            );
            let rendered = composed.render().to_string();
            for needle in &case.rendered_contains {
                assert!(
                    rendered.contains(needle),
                    "[{}] rendered missing {:?}; got {:?}",
                    case.name,
                    needle,
                    rendered
                );
            }
        }
    }

    #[test]
    fn fixture_replay_basic() {
        run_fixture_file("../../../fixtures/prompt_chunk_registry/basic.json");
    }

    // `dangerous.json` exercises `Mode::Yolo`, which only exists under the
    // `dangerous` feature (issue #34). Replayed only by the dangerous-gated
    // suite; the default suite never touches it.
    #[cfg(feature = "dangerous")]
    #[test]
    fn fixture_replay_dangerous() {
        run_fixture_file("../../../fixtures/prompt_chunk_registry/dangerous.json");
    }

    // ── Compose with standard library smoke-test ────────────────────────────

    #[test]
    fn compose_with_standard_chunks_produces_coding_agent_prompt() {
        let r = StandardPromptChunkRegistry::new();
        r.register_standard_chunks().unwrap();
        let composed = r
            .compose(
                ChunkId::new("role-coding-agent"),
                Mode::SafeAuto,
                vec![
                    ChunkId::new("capability-bash"),
                    ChunkId::new("capability-filesystem"),
                    ChunkId::new("capability-git"),
                ],
                vec![
                    ChunkId::new("skill-testing"),
                    ChunkId::new("skill-security-review"),
                ],
            )
            .unwrap();
        assert_eq!(composed.chunks.len(), 1 + 1 + 3 + 2);
        // First chunk is Role; second is Mode.
        assert_eq!(composed.chunks[0].slot, ChunkSlot::Role);
        assert_eq!(composed.chunks[1].slot, ChunkSlot::Mode);
        assert_eq!(composed.chunks[1].id.0, "mode-safe-auto");
    }
}
