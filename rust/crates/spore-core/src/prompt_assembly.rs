//! Issue #79 — `PromptAssemblyEngine`: conditional, provider-sourced prompt
//! assembly that extends (does not replace) the #24 `PromptChunkRegistry`.
//!
//! The shipped #24 [`prompt_chunk_registry`](crate::prompt_chunk_registry) module
//! composes a static Block-1 [`ComposedPrompt`] once at construction. This module
//! builds *on top of* it: chunks are loaded from pluggable [`ChunkProvider`]s and
//! included conditionally based on mode, active tools, phase, agent type, trigger
//! words, hook events, or arbitrary architect-defined predicates. The Static
//! bucket is folded into a #24 [`ComposedPrompt`] (Block 1); PerSession / PerTurn
//! chunks flow through the existing `composed_prompt`/`PromptSegment` machinery in
//! [`context`](crate::context) (decision A4 — no new public segment vectors on
//! [`ContextSources`](crate::context::ContextSources)).
//!
//! This module owns its OWN [`PromptChunk`] and [`ChunkProviderError`] — distinct
//! from the #24 `prompt_chunk_registry::PromptChunk`/`ChunkError`, which are left
//! untouched (decision A1). It is also the home of the minimal shared
//! [`StorageScope`] enum (decision A2).
//!
//! ## Types
//!   - [`PromptChunk`] — unit of assembly content (id, content, stability,
//!     condition, triggers, tool/agent affinity, cache breakpoint).
//!   - [`ToolAffinity`] — `(tool_name, optional capability)` gating a chunk.
//!   - [`ChunkCondition`] — the condition primitive tree evaluated against an
//!     [`AssemblyContext`].
//!   - [`AssemblyContext`] — per-assembly inputs the framework populates.
//!   - [`StorageScope`] — `User` / `Project` / `Local` (serde snake_case).
//!   - [`ChunkProviderError`] — provider load/parse failures (`#[non_exhaustive]`).
//!   - [`ChunkProvider`] trait + [`EmbeddedChunkProvider`],
//!     [`InMemoryChunkProvider`], [`CompositeChunkProvider`].
//!   - [`ContextSourcesBuilder`] — evaluates conditions, buckets by stability,
//!     derives tool-affinity inclusion, scans triggers, injects events, and
//!     composes a Block-1 [`ComposedPrompt`].
//!
//! ## `ChunkProvider` trait methods
//!   - `load(&self) -> BoxFut<Result<Vec<PromptChunk>, ChunkProviderError>>` —
//!     called at harness construction (every request in stateless deployments,
//!     once at startup in long-lived ones). Uses the crate's hand-rolled
//!     [`BoxFut`](crate::harness::BoxFut) pattern, NOT `trait_variant`, so the
//!     trait stays dyn-compatible behind `Arc<dyn ChunkProvider>`.
//!   - `invalidate(&self)` — default no-op; drops cached state so the next
//!     `load()` fetches fresh. Never called mid-session.
//!
//! ## Rules enforced (see the spec issue for cross-language parity)
//!   - R1  `Always` always matches.
//!   - R2  `WhenMode(m)` iff `ctx.mode == m`.
//!   - R3  `WhenToolActive(t)` iff `ctx.active_tool_names` contains `t`.
//!   - R4  `WhenToolCapability(t,c)` iff `ctx.active_capabilities` contains `(t,c)`.
//!   - R5  `WhenPhase` / `WhenAgentType` / `WhenFeature` (a feature matches iff present in `ctx.features` AND its value is `true`).
//!   - R6  `OnTrigger(words)` iff some word is a substring of `ctx.incoming_message` (a `None` message never matches).
//!   - R7  `OnEvent(e)` iff `ctx.pending_events` contains `e`.
//!   - R8  `All` / `Any` / `Not` compose their children.
//!   - R9  `Custom(f)` is evaluated by calling `f(ctx)`.
//!   - R10 Chunks are bucketed by [`SegmentStability`] (Static / PerSession / PerTurn).
//!   - R11 Registration order is preserved within a bucket.
//!   - R12 A `tool_affinity` chunk is included iff its tool is active AND (capability is `None` OR that capability is active).
//!   - R13 `OnTrigger` chunks whose trigger matches the incoming message are routed to the PerTurn bucket.
//!   - R14 `OnEvent` chunks are injected into PerTurn only when their event is pending.
//!   - R15 The Block-1 hash is stable across two builds of an identical Static set.
//!   - R16 `cache_breakpoint` injects a breakpoint after the chunk.
//!   - R17 A tool that is not active yields no description chunk.
//!
//! ## A3 — `Custom` is first-class but unserialized
//!
//! [`ChunkCondition::Custom`] wraps an `Arc<dyn Fn(&AssemblyContext) -> bool>`.
//! It is the PRIMARY escape hatch for conditions that cannot be expressed with
//! the serializable variants, and it is fully supported in the public API.
//! However it CANNOT serialize and CANNOT derive `Eq`/`Hash`, so:
//!   - [`ChunkCondition`] derives neither `Eq` nor `Hash`.
//!   - `Serialize` skips a `Custom` node (it is omitted from the wire form);
//!     `Deserialize` can therefore never produce a `Custom`.
//!   - Manual `PartialEq` treats `Custom` as NEVER equal to anything (including
//!     another `Custom`), since closure identity is not comparable.
//!   - `Custom` is excluded from the shared byte-identical fixtures. Architects
//!     who reach for `Custom` knowingly opt that chunk out of the cross-language
//!     byte-identical contract — a deliberate, supported choice.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::context::{ContextSources, SegmentStability};
use crate::harness::{BoxFut, SessionId, TaskId};
use crate::hooks::HookEvent;
use crate::prompt_chunk_registry::{
    CacheBlock, ChunkSlot, ComposedPrompt, Mode, PromptChunk as RegistryPromptChunk,
};
use crate::tool_registry::TaskPhase;

// ============================================================================
// StorageScope (A2)
// ============================================================================

/// Minimal shared storage scope. This module is its home (decision A2); the
/// scope-aware `FileSystemChunkProvider` that consumes it is deferred (A6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageScope {
    User,
    #[default]
    Project,
    Local,
}

// ============================================================================
// ToolAffinity
// ============================================================================

/// Binds a chunk to a tool (and optionally a sub-capability). The builder
/// includes the chunk only when the tool — and capability, if any — is active.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolAffinity {
    pub tool_name: String,
    #[serde(default)]
    pub capability: Option<String>,
}

impl ToolAffinity {
    pub fn tool(tool_name: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            capability: None,
        }
    }

    pub fn with_capability(tool_name: impl Into<String>, capability: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            capability: Some(capability.into()),
        }
    }
}

// ============================================================================
// ChunkCondition
// ============================================================================

/// Closure type behind [`ChunkCondition::Custom`].
pub type CustomCondition = Arc<dyn Fn(&AssemblyContext) -> bool + Send + Sync>;

/// The condition primitive tree. Architects compose these; the framework
/// evaluates them against an [`AssemblyContext`] via
/// [`ContextSourcesBuilder::evaluate`].
///
/// All variants serialize EXCEPT [`Custom`](ChunkCondition::Custom) — see the
/// module-level A3 note. The enum derives neither `Eq` nor `Hash`; `PartialEq`
/// is implemented manually with `Custom` treated as never-equal.
#[derive(Clone)]
pub enum ChunkCondition {
    Always,
    WhenMode(Mode),
    WhenToolActive(String),
    WhenToolCapability(String, String),
    WhenPhase(TaskPhase),
    WhenAgentType(String),
    WhenFeature(String),
    OnTrigger(Vec<String>),
    OnEvent(HookEvent),
    All(Vec<ChunkCondition>),
    Any(Vec<ChunkCondition>),
    Not(Box<ChunkCondition>),
    /// Arbitrary predicate (A3). First-class, but not serializable and never
    /// equal under `PartialEq`. Constructible only programmatically.
    Custom(CustomCondition),
}

impl std::fmt::Debug for ChunkCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunkCondition::Always => write!(f, "Always"),
            ChunkCondition::WhenMode(m) => f.debug_tuple("WhenMode").field(m).finish(),
            ChunkCondition::WhenToolActive(t) => f.debug_tuple("WhenToolActive").field(t).finish(),
            ChunkCondition::WhenToolCapability(t, c) => f
                .debug_tuple("WhenToolCapability")
                .field(t)
                .field(c)
                .finish(),
            ChunkCondition::WhenPhase(p) => f.debug_tuple("WhenPhase").field(p).finish(),
            ChunkCondition::WhenAgentType(a) => f.debug_tuple("WhenAgentType").field(a).finish(),
            ChunkCondition::WhenFeature(feat) => f.debug_tuple("WhenFeature").field(feat).finish(),
            ChunkCondition::OnTrigger(words) => f.debug_tuple("OnTrigger").field(words).finish(),
            ChunkCondition::OnEvent(e) => f.debug_tuple("OnEvent").field(e).finish(),
            ChunkCondition::All(cs) => f.debug_tuple("All").field(cs).finish(),
            ChunkCondition::Any(cs) => f.debug_tuple("Any").field(cs).finish(),
            ChunkCondition::Not(c) => f.debug_tuple("Not").field(c).finish(),
            ChunkCondition::Custom(_) => write!(f, "Custom(<fn>)"),
        }
    }
}

impl PartialEq for ChunkCondition {
    fn eq(&self, other: &Self) -> bool {
        use ChunkCondition::*;
        match (self, other) {
            (Always, Always) => true,
            (WhenMode(a), WhenMode(b)) => a == b,
            (WhenToolActive(a), WhenToolActive(b)) => a == b,
            (WhenToolCapability(a1, a2), WhenToolCapability(b1, b2)) => a1 == b1 && a2 == b2,
            (WhenPhase(a), WhenPhase(b)) => a == b,
            (WhenAgentType(a), WhenAgentType(b)) => a == b,
            (WhenFeature(a), WhenFeature(b)) => a == b,
            (OnTrigger(a), OnTrigger(b)) => a == b,
            (OnEvent(a), OnEvent(b)) => a == b,
            (All(a), All(b)) => a == b,
            (Any(a), Any(b)) => a == b,
            (Not(a), Not(b)) => a == b,
            // Custom is never equal to anything (A3) — closure identity is
            // not comparable.
            _ => false,
        }
    }
}

// ── Serialization (A3): all variants except Custom ──────────────────────────
//
// Wire form is an internally-tagged object keyed on `type`. A `Custom` node is
// skipped on serialize (emitted as `null`), and Deserialize never yields one.

impl Serialize for ChunkCondition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let repr = ConditionRepr::from_condition(self);
        match repr {
            Some(r) => r.serialize(serializer),
            // Custom: omit from the wire form entirely.
            None => serializer.serialize_none(),
        }
    }
}

impl<'de> Deserialize<'de> for ChunkCondition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // A `null` node (the wire form a skipped `Custom` produces, A3) has no
        // representable condition; it deserializes to the `Always` default.
        let repr = Option::<ConditionRepr>::deserialize(deserializer)?;
        Ok(repr.map_or(ChunkCondition::Always, ConditionRepr::into_condition))
    }
}

/// Serializable mirror of [`ChunkCondition`] (everything but `Custom`). Kept
/// private; the public type round-trips through it. Field ordering is stable
/// for byte-identical cross-language fixtures.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ConditionRepr {
    Always,
    WhenMode { mode: Mode },
    WhenToolActive { tool: String },
    WhenToolCapability { tool: String, capability: String },
    WhenPhase { phase: TaskPhase },
    WhenAgentType { agent_type: String },
    WhenFeature { feature: String },
    OnTrigger { words: Vec<String> },
    OnEvent { event: HookEvent },
    All { conditions: Vec<ConditionRepr> },
    Any { conditions: Vec<ConditionRepr> },
    Not { condition: Box<ConditionRepr> },
}

impl ConditionRepr {
    /// Returns `None` for a `Custom` node (and prunes `Custom` children out of
    /// the boolean combinators, since they cannot be represented on the wire).
    fn from_condition(c: &ChunkCondition) -> Option<ConditionRepr> {
        match c {
            ChunkCondition::Always => Some(ConditionRepr::Always),
            ChunkCondition::WhenMode(m) => Some(ConditionRepr::WhenMode { mode: *m }),
            ChunkCondition::WhenToolActive(t) => {
                Some(ConditionRepr::WhenToolActive { tool: t.clone() })
            }
            ChunkCondition::WhenToolCapability(t, cap) => Some(ConditionRepr::WhenToolCapability {
                tool: t.clone(),
                capability: cap.clone(),
            }),
            ChunkCondition::WhenPhase(p) => Some(ConditionRepr::WhenPhase { phase: *p }),
            ChunkCondition::WhenAgentType(a) => Some(ConditionRepr::WhenAgentType {
                agent_type: a.clone(),
            }),
            ChunkCondition::WhenFeature(feat) => Some(ConditionRepr::WhenFeature {
                feature: feat.clone(),
            }),
            ChunkCondition::OnTrigger(words) => Some(ConditionRepr::OnTrigger {
                words: words.clone(),
            }),
            ChunkCondition::OnEvent(e) => Some(ConditionRepr::OnEvent { event: *e }),
            ChunkCondition::All(cs) => Some(ConditionRepr::All {
                conditions: cs
                    .iter()
                    .filter_map(ConditionRepr::from_condition)
                    .collect(),
            }),
            ChunkCondition::Any(cs) => Some(ConditionRepr::Any {
                conditions: cs
                    .iter()
                    .filter_map(ConditionRepr::from_condition)
                    .collect(),
            }),
            ChunkCondition::Not(inner) => {
                ConditionRepr::from_condition(inner).map(|r| ConditionRepr::Not {
                    condition: Box::new(r),
                })
            }
            ChunkCondition::Custom(_) => None,
        }
    }

    fn into_condition(self) -> ChunkCondition {
        match self {
            ConditionRepr::Always => ChunkCondition::Always,
            ConditionRepr::WhenMode { mode } => ChunkCondition::WhenMode(mode),
            ConditionRepr::WhenToolActive { tool } => ChunkCondition::WhenToolActive(tool),
            ConditionRepr::WhenToolCapability { tool, capability } => {
                ChunkCondition::WhenToolCapability(tool, capability)
            }
            ConditionRepr::WhenPhase { phase } => ChunkCondition::WhenPhase(phase),
            ConditionRepr::WhenAgentType { agent_type } => {
                ChunkCondition::WhenAgentType(agent_type)
            }
            ConditionRepr::WhenFeature { feature } => ChunkCondition::WhenFeature(feature),
            ConditionRepr::OnTrigger { words } => ChunkCondition::OnTrigger(words),
            ConditionRepr::OnEvent { event } => ChunkCondition::OnEvent(event),
            ConditionRepr::All { conditions } => {
                ChunkCondition::All(conditions.into_iter().map(|r| r.into_condition()).collect())
            }
            ConditionRepr::Any { conditions } => {
                ChunkCondition::Any(conditions.into_iter().map(|r| r.into_condition()).collect())
            }
            ConditionRepr::Not { condition } => {
                ChunkCondition::Not(Box::new(condition.into_condition()))
            }
        }
    }
}

// ============================================================================
// PromptChunk (this module's own — distinct from #24, decision A1)
// ============================================================================

/// The unit of conditional assembly content. Distinct from the #24
/// [`prompt_chunk_registry::PromptChunk`](crate::prompt_chunk_registry::PromptChunk):
/// this carries a [`ChunkCondition`], triggers, affinities, and a stability
/// bucket rather than a slot.
///
/// `PartialEq` is manual (delegating to each field; `ChunkCondition`'s own
/// `PartialEq` treats `Custom` as never-equal). `Serialize` omits a `Custom`
/// condition from the wire form (A3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptChunk {
    pub id: String,
    pub content: String,
    pub stability: SegmentStability,
    #[serde(default = "default_condition")]
    pub condition: ChunkCondition,
    #[serde(default)]
    pub triggers: Vec<String>,
    #[serde(default)]
    pub tool_affinity: Option<ToolAffinity>,
    #[serde(default)]
    pub agent_affinity: Option<String>,
    #[serde(default)]
    pub cache_breakpoint: bool,
}

fn default_condition() -> ChunkCondition {
    ChunkCondition::Always
}

impl PromptChunk {
    /// Build a `Static`, `Always` chunk — the common case.
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            stability: SegmentStability::Static,
            condition: ChunkCondition::Always,
            triggers: Vec::new(),
            tool_affinity: None,
            agent_affinity: None,
            cache_breakpoint: false,
        }
    }

    pub fn with_stability(mut self, stability: SegmentStability) -> Self {
        self.stability = stability;
        self
    }

    pub fn with_condition(mut self, condition: ChunkCondition) -> Self {
        self.condition = condition;
        self
    }

    pub fn with_triggers(mut self, triggers: Vec<String>) -> Self {
        self.triggers = triggers;
        self
    }

    pub fn with_tool_affinity(mut self, affinity: ToolAffinity) -> Self {
        self.tool_affinity = Some(affinity);
        self
    }

    pub fn with_agent_affinity(mut self, agent_type: impl Into<String>) -> Self {
        self.agent_affinity = Some(agent_type.into());
        self
    }

    pub fn with_cache_breakpoint(mut self, breakpoint: bool) -> Self {
        self.cache_breakpoint = breakpoint;
        self
    }
}

// ============================================================================
// AssemblyContext
// ============================================================================

/// Per-assembly inputs the framework populates before each assembly. `Custom`
/// conditions read from it; `features` is the escape hatch for architect flags.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssemblyContext {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub turn_number: u32,
    pub mode: Mode,
    pub phase: TaskPhase,
    #[serde(default)]
    pub agent_type: Option<String>,
    #[serde(default)]
    pub active_tool_names: HashSet<String>,
    #[serde(default)]
    pub active_capabilities: HashSet<(String, String)>,
    #[serde(default)]
    pub incoming_message: Option<String>,
    #[serde(default)]
    pub pending_events: Vec<HookEvent>,
    #[serde(default)]
    pub features: HashMap<String, bool>,
    #[serde(default)]
    pub storage_scope: StorageScope,
}

impl AssemblyContext {
    /// Construct a minimal context. Optional collections start empty.
    pub fn new(
        session_id: SessionId,
        task_id: TaskId,
        turn_number: u32,
        mode: Mode,
        phase: TaskPhase,
    ) -> Self {
        Self {
            session_id,
            task_id,
            turn_number,
            mode,
            phase,
            agent_type: None,
            active_tool_names: HashSet::new(),
            active_capabilities: HashSet::new(),
            incoming_message: None,
            pending_events: Vec::new(),
            features: HashMap::new(),
            storage_scope: StorageScope::default(),
        }
    }
}

// ============================================================================
// ChunkProviderError
// ============================================================================

/// Errors a [`ChunkProvider`] can raise while loading chunks. Kept minimal
/// because the Remote/FileSystem providers are deferred (A6), but
/// `#[non_exhaustive]` so the deferred follow-up can extend it without a
/// breaking change.
#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChunkProviderError {
    #[error("chunk load failed from {provider}: {detail}")]
    LoadFailed { provider: String, detail: String },
    #[error("chunk parse error: {detail}")]
    ParseError { detail: String },
}

// ============================================================================
// ChunkProvider trait
// ============================================================================

/// The pluggable source of chunks. Dyn-compatible via the crate's hand-rolled
/// [`BoxFut`](crate::harness::BoxFut) pattern (not `trait_variant`), so it can
/// be injected as `Arc<dyn ChunkProvider>`.
pub trait ChunkProvider: Send + Sync {
    /// Load all chunks this provider is responsible for. Called at harness
    /// construction.
    fn load<'a>(&'a self) -> BoxFut<'a, Result<Vec<PromptChunk>, ChunkProviderError>>;

    /// Drop any cached state so the next `load()` fetches fresh. No-op by
    /// default; never called mid-session.
    fn invalidate(&self) {}
}

// ── EmbeddedChunkProvider ────────────────────────────────────────────────────

/// Compile-time / construction-time chunks. Immutable; `invalidate` is a no-op
/// and `load` always returns the same set.
pub struct EmbeddedChunkProvider {
    chunks: Vec<PromptChunk>,
}

impl EmbeddedChunkProvider {
    pub fn new(chunks: Vec<PromptChunk>) -> Self {
        Self { chunks }
    }
}

impl ChunkProvider for EmbeddedChunkProvider {
    fn load<'a>(&'a self) -> BoxFut<'a, Result<Vec<PromptChunk>, ChunkProviderError>> {
        Box::pin(async move { Ok(self.chunks.clone()) })
    }
    // invalidate: default no-op (chunks are immutable constants).
}

// ── InMemoryChunkProvider ────────────────────────────────────────────────────

/// Programmatic provider. `invalidate` followed by [`set`](Self::set) replaces
/// the chunk list. Interior mutability via `RwLock` keeps the trait `&self`.
pub struct InMemoryChunkProvider {
    chunks: RwLock<Vec<PromptChunk>>,
}

impl InMemoryChunkProvider {
    pub fn new(chunks: Vec<PromptChunk>) -> Self {
        Self {
            chunks: RwLock::new(chunks),
        }
    }

    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Replace the chunk list. The next `load()` returns the new set.
    pub fn set(&self, chunks: Vec<PromptChunk>) {
        if let Ok(mut guard) = self.chunks.write() {
            *guard = chunks;
        }
    }
}

impl ChunkProvider for InMemoryChunkProvider {
    fn load<'a>(&'a self) -> BoxFut<'a, Result<Vec<PromptChunk>, ChunkProviderError>> {
        Box::pin(async move {
            let guard = self
                .chunks
                .read()
                .map_err(|e| ChunkProviderError::LoadFailed {
                    provider: "in_memory".into(),
                    detail: e.to_string(),
                })?;
            Ok(guard.clone())
        })
    }

    fn invalidate(&self) {
        // Stateless cache; the architect replaces chunks via `set`. Clearing
        // here would discard programmatic registrations, so this is a no-op.
    }
}

// ── CompositeChunkProvider ───────────────────────────────────────────────────

/// Merges N providers into one flat list (in add order) and propagates
/// `invalidate` to every child.
#[derive(Default)]
pub struct CompositeChunkProvider {
    providers: Vec<Arc<dyn ChunkProvider>>,
}

impl CompositeChunkProvider {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, provider: Arc<dyn ChunkProvider>) -> Self {
        self.providers.push(provider);
        self
    }

    pub fn push(&mut self, provider: Arc<dyn ChunkProvider>) {
        self.providers.push(provider);
    }
}

impl ChunkProvider for CompositeChunkProvider {
    fn load<'a>(&'a self) -> BoxFut<'a, Result<Vec<PromptChunk>, ChunkProviderError>> {
        Box::pin(async move {
            let mut out = Vec::new();
            for p in &self.providers {
                out.extend(p.load().await?);
            }
            Ok(out)
        })
    }

    fn invalidate(&self) {
        for p in &self.providers {
            p.invalidate();
        }
    }
}

// ============================================================================
// ContextSourcesBuilder
// ============================================================================

/// Evaluates conditions, buckets chunks by stability, derives tool-affinity
/// inclusion, scans triggers, injects pending events, and composes a Block-1
/// [`ComposedPrompt`] from the Static bucket. The result feeds
/// [`ContextSources`](crate::context::ContextSources) (decision A4).
#[derive(Debug, Clone, Default)]
pub struct ContextSourcesBuilder {
    chunks: Vec<PromptChunk>,
}

/// The bucketed outcome of [`ContextSourcesBuilder::assemble`]. Buckets keep
/// registration order within each stability tier.
#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyBuckets {
    pub static_chunks: Vec<PromptChunk>,
    pub per_session: Vec<PromptChunk>,
    pub per_turn: Vec<PromptChunk>,
}

impl ContextSourcesBuilder {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// Seed the builder with chunks (registration order is preserved).
    pub fn with_chunks(chunks: Vec<PromptChunk>) -> Self {
        Self { chunks }
    }

    /// Append a chunk, preserving registration order.
    pub fn register(&mut self, chunk: PromptChunk) -> &mut Self {
        self.chunks.push(chunk);
        self
    }

    /// The load-bearing primitive: recursively evaluate `condition` against
    /// `ctx`. Rules R1–R9.
    pub fn evaluate(&self, condition: &ChunkCondition, ctx: &AssemblyContext) -> bool {
        match condition {
            // R1
            ChunkCondition::Always => true,
            // R2
            ChunkCondition::WhenMode(m) => ctx.mode == *m,
            // R3
            ChunkCondition::WhenToolActive(t) => ctx.active_tool_names.contains(t),
            // R4
            ChunkCondition::WhenToolCapability(t, c) => {
                ctx.active_capabilities.contains(&(t.clone(), c.clone()))
            }
            // R5
            ChunkCondition::WhenPhase(p) => ctx.phase == *p,
            ChunkCondition::WhenAgentType(a) => ctx.agent_type.as_deref() == Some(a.as_str()),
            ChunkCondition::WhenFeature(f) => ctx.features.get(f).copied().unwrap_or(false),
            // R6 — substring match; None message never matches.
            ChunkCondition::OnTrigger(words) => match &ctx.incoming_message {
                Some(msg) => words.iter().any(|w| msg.contains(w.as_str())),
                None => false,
            },
            // R7
            ChunkCondition::OnEvent(e) => ctx.pending_events.contains(e),
            // R8
            ChunkCondition::All(cs) => cs.iter().all(|c| self.evaluate(c, ctx)),
            ChunkCondition::Any(cs) => cs.iter().any(|c| self.evaluate(c, ctx)),
            ChunkCondition::Not(inner) => !self.evaluate(inner, ctx),
            // R9
            ChunkCondition::Custom(f) => f(ctx),
        }
    }

    /// Whether a chunk's `tool_affinity` gate passes for `ctx`. A chunk with no
    /// affinity always passes this gate. Rule R12 / R17.
    fn tool_affinity_ok(chunk: &PromptChunk, ctx: &AssemblyContext) -> bool {
        match &chunk.tool_affinity {
            None => true,
            Some(aff) => {
                if !ctx.active_tool_names.contains(&aff.tool_name) {
                    return false;
                }
                match &aff.capability {
                    None => true,
                    Some(cap) => ctx
                        .active_capabilities
                        .contains(&(aff.tool_name.clone(), cap.clone())),
                }
            }
        }
    }

    /// Whether a chunk's `agent_affinity` gate passes. A chunk with no
    /// agent_affinity always passes; otherwise it must match `ctx.agent_type`.
    fn agent_affinity_ok(chunk: &PromptChunk, ctx: &AssemblyContext) -> bool {
        match &chunk.agent_affinity {
            None => true,
            Some(agent) => ctx.agent_type.as_deref() == Some(agent.as_str()),
        }
    }

    /// Whether a chunk's `triggers` list matches the incoming message. An empty
    /// trigger list never forces inclusion. Rule R13.
    fn triggers_match(chunk: &PromptChunk, ctx: &AssemblyContext) -> bool {
        if chunk.triggers.is_empty() {
            return false;
        }
        match &ctx.incoming_message {
            Some(msg) => chunk.triggers.iter().any(|t| msg.contains(t.as_str())),
            None => false,
        }
    }

    /// Run the assembly steps (spec § "Assembly steps") and bucket the included
    /// chunks. Registration order is preserved within each bucket (R10/R11).
    ///
    /// A chunk is included when its `condition` evaluates true AND its
    /// `tool_affinity` AND `agent_affinity` gates pass. A chunk whose `triggers`
    /// match the incoming message is forced into the PerTurn bucket regardless
    /// of its declared stability (R13). Bucket assignment otherwise follows the
    /// chunk's `stability`.
    pub fn assemble(&self, ctx: &AssemblyContext) -> AssemblyBuckets {
        let mut static_chunks = Vec::new();
        let mut per_session = Vec::new();
        let mut per_turn = Vec::new();

        for chunk in &self.chunks {
            // Gates that apply to EVERY chunk regardless of condition kind.
            if !Self::tool_affinity_ok(chunk, ctx) {
                continue;
            }
            if !Self::agent_affinity_ok(chunk, ctx) {
                continue;
            }

            let condition_ok = self.evaluate(&chunk.condition, ctx);
            let trigger_forced = Self::triggers_match(chunk, ctx);

            if !condition_ok && !trigger_forced {
                continue;
            }

            // R13: a trigger match routes the chunk into PerTurn no matter its
            // declared stability. R14 falls out of this too: an `OnEvent`
            // chunk is only `condition_ok` when its event is pending, and
            // `OnEvent` chunks are declared PerTurn by convention.
            if trigger_forced {
                per_turn.push(chunk.clone());
                continue;
            }

            match chunk.stability {
                SegmentStability::Static => static_chunks.push(chunk.clone()),
                SegmentStability::PerSession => per_session.push(chunk.clone()),
                SegmentStability::PerTurn => per_turn.push(chunk.clone()),
            }
        }

        AssemblyBuckets {
            static_chunks,
            per_session,
            per_turn,
        }
    }

    /// Compose the Static bucket into a #24 [`ComposedPrompt`] (Block 1). Each
    /// Static chunk maps to a #24 `PromptChunk` in [`ChunkSlot::Environment`]
    /// (a neutral, non-required slot) with [`CacheBlock::Static`], preserving
    /// order. The block hashes are recomputed from the mapped chunks so the
    /// Block-1 hash is stable across identical Static sets (R15).
    pub fn compose_block_1(&self, buckets: &AssemblyBuckets) -> ComposedPrompt {
        let chunks: Vec<RegistryPromptChunk> = buckets
            .static_chunks
            .iter()
            .map(|c| {
                RegistryPromptChunk::new(
                    c.id.clone(),
                    c.content.clone(),
                    ChunkSlot::Environment,
                    CacheBlock::Static,
                )
            })
            .collect();

        let mut composed = ComposedPrompt {
            chunks,
            block_1_hash: 0,
            block_2_hash: 0,
            rendered: None,
        };
        let (b1, b2) = composed.recompute_hashes();
        composed.block_1_hash = b1;
        composed.block_2_hash = b2;
        composed.render();
        composed
    }

    /// Full pipeline: assemble buckets, compose Block 1, and produce a
    /// [`ContextSources`] (decision A4 — PerSession/PerTurn fold through the
    /// existing `composed_prompt`/segment machinery downstream; this builder
    /// supplies the composed Block 1 and the buckets the caller threads into
    /// [`SessionState`](crate::context::SessionState)).
    ///
    /// `guides`, `memory`, and `tool_schemas` are passed through verbatim —
    /// the builder does not synthesize tool description text (decision A5).
    pub fn build_context_sources(
        &self,
        ctx: &AssemblyContext,
        guides: Vec<crate::guide_registry::Guide>,
        memory: Vec<crate::memory::MemoryItem>,
        tool_schemas: Vec<crate::model::ToolSchema>,
    ) -> (ContextSources, AssemblyBuckets) {
        let buckets = self.assemble(ctx);
        let composed_prompt = self.compose_block_1(&buckets);
        let sources = ContextSources {
            guides,
            memory,
            tool_schemas,
            composed_prompt,
        };
        (sources, buckets)
    }
}

/// Render breakpoint info derived from a bucket: an entry per chunk that
/// declared `cache_breakpoint` (R16), in order. Exposed for callers wiring the
/// PerSession/PerTurn segments into the segment machinery.
pub fn breakpoint_ids(buckets: &AssemblyBuckets) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in buckets
        .static_chunks
        .iter()
        .chain(buckets.per_session.iter())
        .chain(buckets.per_turn.iter())
    {
        if chunk.cache_breakpoint {
            out.push(chunk.id.clone());
        }
    }
    out
}

/// Map a bucket of chunks into [`PromptSegment`](crate::context::PromptSegment)s
/// for the #7 context machinery (decision A4). Preserves order and carries
/// each chunk's `cache_breakpoint` (R16).
pub fn chunks_to_segments(chunks: &[PromptChunk]) -> Vec<crate::context::PromptSegment> {
    chunks
        .iter()
        .map(|c| crate::context::PromptSegment {
            name: c.id.clone(),
            content: c.content.clone(),
            stability: c.stability,
            cache_breakpoint: c.cache_breakpoint,
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> AssemblyContext {
        AssemblyContext::new(
            SessionId::new("s1"),
            TaskId::new("t1"),
            1,
            Mode::SafeAuto,
            TaskPhase::Execution,
        )
    }

    fn builder() -> ContextSourcesBuilder {
        ContextSourcesBuilder::new()
    }

    // ── R1: Always always matches ────────────────────────────────────────────
    #[test]
    fn r1_always_matches() {
        assert!(builder().evaluate(&ChunkCondition::Always, &ctx()));
    }

    // ── R2: WhenMode ─────────────────────────────────────────────────────────
    #[test]
    fn r2_when_mode() {
        let b = builder();
        let mut c = ctx();
        c.mode = Mode::Plan;
        assert!(b.evaluate(&ChunkCondition::WhenMode(Mode::Plan), &c));
        assert!(!b.evaluate(&ChunkCondition::WhenMode(Mode::AutoEdit), &c));
    }

    // ── R3: WhenToolActive ───────────────────────────────────────────────────
    #[test]
    fn r3_when_tool_active() {
        let b = builder();
        let mut c = ctx();
        c.active_tool_names.insert("bash".into());
        assert!(b.evaluate(&ChunkCondition::WhenToolActive("bash".into()), &c));
        assert!(!b.evaluate(&ChunkCondition::WhenToolActive("grep".into()), &c));
    }

    // ── R4: WhenToolCapability ───────────────────────────────────────────────
    #[test]
    fn r4_when_tool_capability() {
        let b = builder();
        let mut c = ctx();
        c.active_capabilities
            .insert(("bash".into(), "sandbox".into()));
        assert!(b.evaluate(
            &ChunkCondition::WhenToolCapability("bash".into(), "sandbox".into()),
            &c
        ));
        assert!(!b.evaluate(
            &ChunkCondition::WhenToolCapability("bash".into(), "git".into()),
            &c
        ));
    }

    // ── R5: WhenPhase / WhenAgentType / WhenFeature ─────────────────────────
    #[test]
    fn r5_when_phase_agent_feature() {
        let b = builder();
        let mut c = ctx();
        c.phase = TaskPhase::Planning;
        c.agent_type = Some("planner".into());
        c.features.insert("beta".into(), true);
        c.features.insert("alpha".into(), false);

        assert!(b.evaluate(&ChunkCondition::WhenPhase(TaskPhase::Planning), &c));
        assert!(!b.evaluate(&ChunkCondition::WhenPhase(TaskPhase::Cleanup), &c));

        assert!(b.evaluate(&ChunkCondition::WhenAgentType("planner".into()), &c));
        assert!(!b.evaluate(&ChunkCondition::WhenAgentType("coder".into()), &c));

        // feature true iff present AND true
        assert!(b.evaluate(&ChunkCondition::WhenFeature("beta".into()), &c));
        assert!(!b.evaluate(&ChunkCondition::WhenFeature("alpha".into()), &c));
        assert!(!b.evaluate(&ChunkCondition::WhenFeature("missing".into()), &c));
    }

    // ── R6: OnTrigger ────────────────────────────────────────────────────────
    #[test]
    fn r6_on_trigger() {
        let b = builder();
        let mut c = ctx();
        let cond = ChunkCondition::OnTrigger(vec!["deploy".into(), "rollback".into()]);
        // None message -> false
        assert!(!b.evaluate(&cond, &c));
        c.incoming_message = Some("please deploy the service".into());
        assert!(b.evaluate(&cond, &c));
        c.incoming_message = Some("nothing relevant".into());
        assert!(!b.evaluate(&cond, &c));
    }

    // ── R7: OnEvent ──────────────────────────────────────────────────────────
    #[test]
    fn r7_on_event() {
        let b = builder();
        let mut c = ctx();
        let cond = ChunkCondition::OnEvent(HookEvent::PreCompact);
        assert!(!b.evaluate(&cond, &c));
        c.pending_events.push(HookEvent::PreCompact);
        assert!(b.evaluate(&cond, &c));
    }

    // ── R8: All / Any / Not ──────────────────────────────────────────────────
    #[test]
    fn r8_all_any_not() {
        let b = builder();
        let mut c = ctx();
        c.mode = Mode::Plan;
        c.active_tool_names.insert("bash".into());

        let all = ChunkCondition::All(vec![
            ChunkCondition::WhenMode(Mode::Plan),
            ChunkCondition::WhenToolActive("bash".into()),
        ]);
        assert!(b.evaluate(&all, &c));

        let all_fail = ChunkCondition::All(vec![
            ChunkCondition::WhenMode(Mode::Plan),
            ChunkCondition::WhenToolActive("grep".into()),
        ]);
        assert!(!b.evaluate(&all_fail, &c));

        let any = ChunkCondition::Any(vec![
            ChunkCondition::WhenToolActive("grep".into()),
            ChunkCondition::WhenMode(Mode::Plan),
        ]);
        assert!(b.evaluate(&any, &c));

        let not = ChunkCondition::Not(Box::new(ChunkCondition::WhenMode(Mode::AutoEdit)));
        assert!(b.evaluate(&not, &c));
    }

    // ── R9: Custom ───────────────────────────────────────────────────────────
    #[test]
    fn r9_custom_evaluated_against_ctx() {
        let b = builder();
        let mut c = ctx();
        c.turn_number = 5;
        let cond = ChunkCondition::Custom(Arc::new(|ctx: &AssemblyContext| ctx.turn_number > 3));
        assert!(b.evaluate(&cond, &c));
        c.turn_number = 1;
        assert!(!b.evaluate(&cond, &c));
    }

    // ── R10: bucketed by stability ───────────────────────────────────────────
    #[test]
    fn r10_bucketed_by_stability() {
        let b = ContextSourcesBuilder::with_chunks(vec![
            PromptChunk::new("s", "static").with_stability(SegmentStability::Static),
            PromptChunk::new("ps", "session").with_stability(SegmentStability::PerSession),
            PromptChunk::new("pt", "turn").with_stability(SegmentStability::PerTurn),
        ]);
        let buckets = b.assemble(&ctx());
        assert_eq!(buckets.static_chunks.len(), 1);
        assert_eq!(buckets.per_session.len(), 1);
        assert_eq!(buckets.per_turn.len(), 1);
        assert_eq!(buckets.static_chunks[0].id, "s");
        assert_eq!(buckets.per_session[0].id, "ps");
        assert_eq!(buckets.per_turn[0].id, "pt");
    }

    // ── R11: registration order preserved within bucket ─────────────────────
    #[test]
    fn r11_registration_order_within_bucket() {
        let b = ContextSourcesBuilder::with_chunks(vec![
            PromptChunk::new("a", "a"),
            PromptChunk::new("b", "b"),
            PromptChunk::new("c", "c"),
        ]);
        let buckets = b.assemble(&ctx());
        let ids: Vec<&str> = buckets
            .static_chunks
            .iter()
            .map(|c| c.id.as_str())
            .collect();
        assert_eq!(ids, ["a", "b", "c"]);
    }

    // ── R12: tool-affinity 4-way matrix ──────────────────────────────────────
    #[test]
    fn r12_tool_affinity_matrix() {
        // Chunk gated on (tool=bash, capability=git).
        let chunk = PromptChunk::new("bash-git", "git guide")
            .with_tool_affinity(ToolAffinity::with_capability("bash", "git"));
        let b = ContextSourcesBuilder::with_chunks(vec![chunk]);

        // (1) tool inactive, cap inactive -> excluded
        let mut c = ctx();
        assert!(b.assemble(&c).static_chunks.is_empty());

        // (2) tool active, cap inactive -> excluded (cap required)
        c.active_tool_names.insert("bash".into());
        assert!(b.assemble(&c).static_chunks.is_empty());

        // (3) tool active, cap active -> included
        c.active_capabilities.insert(("bash".into(), "git".into()));
        assert_eq!(b.assemble(&c).static_chunks.len(), 1);

        // (4) tool inactive but cap present -> excluded (tool gate first)
        let mut c2 = ctx();
        c2.active_capabilities.insert(("bash".into(), "git".into()));
        assert!(b.assemble(&c2).static_chunks.is_empty());

        // Capability None: included as soon as the tool is active.
        let chunk2 = PromptChunk::new("bash-any", "bash guide")
            .with_tool_affinity(ToolAffinity::tool("bash"));
        let b2 = ContextSourcesBuilder::with_chunks(vec![chunk2]);
        let mut c3 = ctx();
        assert!(b2.assemble(&c3).static_chunks.is_empty());
        c3.active_tool_names.insert("bash".into());
        assert_eq!(b2.assemble(&c3).static_chunks.len(), 1);
    }

    // ── R13: OnTrigger matches pushed to PerTurn ─────────────────────────────
    #[test]
    fn r13_trigger_match_routes_to_per_turn() {
        // Declared Static, but trigger match forces PerTurn.
        let chunk = PromptChunk::new("playbook", "rollback steps")
            .with_stability(SegmentStability::Static)
            .with_triggers(vec!["rollback".into()]);
        let b = ContextSourcesBuilder::with_chunks(vec![chunk]);

        let mut c = ctx();
        // No message -> not included at all (condition Always still true, but
        // declared Static — wait: Always is true, so it lands Static normally).
        // To isolate trigger routing, gate the chunk so it only appears via
        // trigger.
        let chunk2 = PromptChunk::new("playbook2", "rollback steps")
            .with_stability(SegmentStability::Static)
            .with_condition(ChunkCondition::OnTrigger(vec!["rollback".into()]))
            .with_triggers(vec!["rollback".into()]);
        let b2 = ContextSourcesBuilder::with_chunks(vec![chunk2]);
        assert!(b2.assemble(&c).static_chunks.is_empty());
        assert!(b2.assemble(&c).per_turn.is_empty());

        c.incoming_message = Some("we must rollback now".into());
        let buckets = b2.assemble(&c);
        assert!(buckets.static_chunks.is_empty());
        assert_eq!(buckets.per_turn.len(), 1);
        assert_eq!(buckets.per_turn[0].id, "playbook2");

        // And the simple Static+trigger chunk also routes to PerTurn on match.
        let buckets1 = b.assemble(&c);
        assert!(buckets1.static_chunks.is_empty());
        assert_eq!(buckets1.per_turn.len(), 1);
    }

    // ── R14: OnEvent chunks injected to PerTurn only when event pending ──────
    #[test]
    fn r14_on_event_injected_only_when_pending() {
        let chunk = PromptChunk::new("reminder", "system reminder")
            .with_stability(SegmentStability::PerTurn)
            .with_condition(ChunkCondition::OnEvent(HookEvent::PreCompact));
        let b = ContextSourcesBuilder::with_chunks(vec![chunk]);

        let mut c = ctx();
        assert!(b.assemble(&c).per_turn.is_empty());

        c.pending_events.push(HookEvent::PreCompact);
        let buckets = b.assemble(&c);
        assert_eq!(buckets.per_turn.len(), 1);
        assert_eq!(buckets.per_turn[0].id, "reminder");
    }

    // ── R15: Block-1 hash stable across two builds of identical Static set ──
    #[test]
    fn r15_block_1_hash_stable() {
        let mk = || {
            ContextSourcesBuilder::with_chunks(vec![
                PromptChunk::new("core", "identity rules"),
                PromptChunk::new("style", "be concise"),
            ])
        };
        let b1 = mk();
        let b2 = mk();
        let cp1 = b1.compose_block_1(&b1.assemble(&ctx()));
        let cp2 = b2.compose_block_1(&b2.assemble(&ctx()));
        assert_eq!(cp1.block_1_hash, cp2.block_1_hash);

        // Different static content -> different hash.
        let b3 = ContextSourcesBuilder::with_chunks(vec![
            PromptChunk::new("core", "DIFFERENT identity"),
            PromptChunk::new("style", "be concise"),
        ]);
        let cp3 = b3.compose_block_1(&b3.assemble(&ctx()));
        assert_ne!(cp1.block_1_hash, cp3.block_1_hash);
    }

    // ── R16: cache_breakpoint injects breakpoint after chunk ────────────────
    #[test]
    fn r16_cache_breakpoint() {
        let b = ContextSourcesBuilder::with_chunks(vec![
            PromptChunk::new("a", "a"),
            PromptChunk::new("b", "b").with_cache_breakpoint(true),
            PromptChunk::new("c", "c"),
        ]);
        let buckets = b.assemble(&ctx());
        assert_eq!(breakpoint_ids(&buckets), vec!["b".to_string()]);

        // Segment mapping carries the breakpoint flag.
        let segs = chunks_to_segments(&buckets.static_chunks);
        assert!(
            segs.iter()
                .find(|s| s.name == "b")
                .unwrap()
                .cache_breakpoint
        );
        assert!(
            !segs
                .iter()
                .find(|s| s.name == "a")
                .unwrap()
                .cache_breakpoint
        );
    }

    // ── R17: tool not active yields no description chunk ────────────────────
    #[test]
    fn r17_tool_not_active_no_description() {
        let chunk = PromptChunk::new("bash-desc", "Bash tool: run shell commands")
            .with_tool_affinity(ToolAffinity::tool("bash"));
        let b = ContextSourcesBuilder::with_chunks(vec![chunk]);
        // Tool not active -> excluded.
        assert!(b.assemble(&ctx()).static_chunks.is_empty());
        // Tool active -> included.
        let mut c = ctx();
        c.active_tool_names.insert("bash".into());
        assert_eq!(b.assemble(&c).static_chunks.len(), 1);
    }

    // ── R18: EmbeddedChunkProvider invalidate no-op, load same ──────────────
    #[tokio::test]
    async fn r18_embedded_provider() {
        let p = EmbeddedChunkProvider::new(vec![PromptChunk::new("x", "y")]);
        let a = p.load().await.unwrap();
        p.invalidate();
        let b = p.load().await.unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 1);
    }

    // ── R19: InMemoryChunkProvider returns registered; set replaces ─────────
    #[tokio::test]
    async fn r19_in_memory_provider() {
        let p = InMemoryChunkProvider::new(vec![PromptChunk::new("x", "y")]);
        assert_eq!(p.load().await.unwrap().len(), 1);
        p.set(vec![PromptChunk::new("a", "1"), PromptChunk::new("b", "2")]);
        let after = p.load().await.unwrap();
        assert_eq!(after.len(), 2);
        assert_eq!(after[0].id, "a");
    }

    // ── R21: CompositeChunkProvider merges + propagates invalidate ──────────
    struct CountingProvider {
        invalidated: std::sync::atomic::AtomicU32,
        chunks: Vec<PromptChunk>,
    }
    impl ChunkProvider for CountingProvider {
        fn load<'a>(&'a self) -> BoxFut<'a, Result<Vec<PromptChunk>, ChunkProviderError>> {
            Box::pin(async move { Ok(self.chunks.clone()) })
        }
        fn invalidate(&self) {
            self.invalidated
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn r21_composite_provider() {
        let p1 = Arc::new(CountingProvider {
            invalidated: std::sync::atomic::AtomicU32::new(0),
            chunks: vec![PromptChunk::new("a", "1")],
        });
        let p2 = Arc::new(CountingProvider {
            invalidated: std::sync::atomic::AtomicU32::new(0),
            chunks: vec![PromptChunk::new("b", "2"), PromptChunk::new("c", "3")],
        });
        let comp = CompositeChunkProvider::new()
            .add(p1.clone())
            .add(p2.clone());
        let merged = comp.load().await.unwrap();
        let ids: Vec<&str> = merged.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]); // add order preserved

        comp.invalidate();
        assert_eq!(p1.invalidated.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(p2.invalidated.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    // ── PartialEq: Custom never equal (A3) ──────────────────────────────────
    #[test]
    fn custom_condition_never_equal() {
        let f: CustomCondition = Arc::new(|_| true);
        let a = ChunkCondition::Custom(f.clone());
        let b = ChunkCondition::Custom(f);
        assert_ne!(a, b);
        // Non-custom variants still compare by value.
        assert_eq!(ChunkCondition::Always, ChunkCondition::Always);
        assert_eq!(
            ChunkCondition::WhenMode(Mode::AutoEdit),
            ChunkCondition::WhenMode(Mode::AutoEdit)
        );
    }

    // ── Serialization: Custom skipped (A3) ──────────────────────────────────
    #[test]
    fn custom_condition_serializes_to_null() {
        let cond = ChunkCondition::Custom(Arc::new(|_| true));
        let json = serde_json::to_value(&cond).unwrap();
        assert!(json.is_null());

        // A PromptChunk carrying Custom serializes its condition as null and
        // round-trips back to the Always default.
        let chunk =
            PromptChunk::new("x", "y").with_condition(ChunkCondition::Custom(Arc::new(|_| true)));
        let s = serde_json::to_string(&chunk).unwrap();
        let back: PromptChunk = serde_json::from_str(&s).unwrap();
        assert_eq!(back.condition, ChunkCondition::Always);
    }

    #[test]
    fn condition_round_trips_serializable_variants() {
        let cond = ChunkCondition::All(vec![
            ChunkCondition::WhenMode(Mode::Plan),
            ChunkCondition::Any(vec![
                ChunkCondition::WhenToolActive("bash".into()),
                ChunkCondition::Not(Box::new(ChunkCondition::WhenFeature("beta".into()))),
            ]),
            ChunkCondition::OnEvent(HookEvent::PreTurn),
            ChunkCondition::OnTrigger(vec!["deploy".into()]),
            ChunkCondition::WhenToolCapability("bash".into(), "git".into()),
            ChunkCondition::WhenPhase(TaskPhase::Planning),
            ChunkCondition::WhenAgentType("planner".into()),
        ]);
        let s = serde_json::to_string(&cond).unwrap();
        let back: ChunkCondition = serde_json::from_str(&s).unwrap();
        assert_eq!(cond, back);
    }

    #[test]
    fn custom_pruned_from_combinators_on_serialize() {
        // A Custom child inside All is pruned; the rest survive round-trip.
        let cond = ChunkCondition::All(vec![
            ChunkCondition::WhenMode(Mode::AutoEdit),
            ChunkCondition::Custom(Arc::new(|_| true)),
        ]);
        let s = serde_json::to_string(&cond).unwrap();
        let back: ChunkCondition = serde_json::from_str(&s).unwrap();
        assert_eq!(
            back,
            ChunkCondition::All(vec![ChunkCondition::WhenMode(Mode::AutoEdit)])
        );
    }

    // ── ChunkProviderError variants ──────────────────────────────────────────
    #[test]
    fn chunk_provider_error_variants_render_and_round_trip() {
        let e = ChunkProviderError::LoadFailed {
            provider: "remote".into(),
            detail: "timeout".into(),
        };
        assert!(e.to_string().contains("remote"));
        let s = serde_json::to_string(&e).unwrap();
        let back: ChunkProviderError = serde_json::from_str(&s).unwrap();
        assert_eq!(e, back);

        let p = ChunkProviderError::ParseError {
            detail: "bad json".into(),
        };
        assert!(p.to_string().contains("bad json"));
    }

    // ── build_context_sources passes guides/memory/schemas through (A5) ─────
    #[tokio::test]
    async fn build_context_sources_threads_block_1_and_passthrough() {
        let b = ContextSourcesBuilder::with_chunks(vec![
            PromptChunk::new("core", "rules"),
            PromptChunk::new("ps", "ref").with_stability(SegmentStability::PerSession),
        ]);
        let (sources, buckets) = b.build_context_sources(&ctx(), vec![], vec![], vec![]);
        assert_eq!(sources.composed_prompt.chunks.len(), 1); // only Static in Block 1
        assert_eq!(sources.composed_prompt.chunks[0].id.0, "core");
        assert_eq!(buckets.per_session.len(), 1);
        assert!(sources.guides.is_empty());
    }

    // ── StorageScope serde snake_case ────────────────────────────────────────
    #[test]
    fn storage_scope_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&StorageScope::User).unwrap(),
            "\"user\""
        );
        assert_eq!(
            serde_json::to_string(&StorageScope::Project).unwrap(),
            "\"project\""
        );
        assert_eq!(
            serde_json::to_string(&StorageScope::Local).unwrap(),
            "\"local\""
        );
    }

    // ── agent_affinity gate ──────────────────────────────────────────────────
    #[test]
    fn agent_affinity_gate() {
        let chunk = PromptChunk::new("planner-prompt", "you plan").with_agent_affinity("planner");
        let b = ContextSourcesBuilder::with_chunks(vec![chunk]);
        // No agent type -> excluded.
        assert!(b.assemble(&ctx()).static_chunks.is_empty());
        let mut c = ctx();
        c.agent_type = Some("planner".into());
        assert_eq!(b.assemble(&c).static_chunks.len(), 1);
        c.agent_type = Some("coder".into());
        assert!(b.assemble(&c).static_chunks.is_empty());
    }

    // ── Fixture replay: condition_eval.json (R1–R8) ─────────────────────────
    #[test]
    fn fixture_replay_condition_eval() {
        #[derive(Deserialize)]
        struct Case {
            name: String,
            condition: ChunkCondition,
            assembly_context: AssemblyContext,
            expected: bool,
        }
        #[derive(Deserialize)]
        struct Suite {
            cases: Vec<Case>,
        }
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/prompt_assembly/condition_eval.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let suite: Suite = serde_json::from_str(&raw).expect("parse condition_eval.json");
        assert!(suite.cases.len() >= 8, "expected >=8 cases (R1-R8)");
        let b = builder();
        for case in &suite.cases {
            let got = b.evaluate(&case.condition, &case.assembly_context);
            assert_eq!(got, case.expected, "case `{}` mismatch", case.name);
        }
    }

    // ── Fixture replay: assembly_steps.json (R10–R17) ───────────────────────
    #[test]
    fn fixture_replay_assembly_steps() {
        #[derive(Deserialize)]
        struct Case {
            name: String,
            registered_chunks: Vec<PromptChunk>,
            assembly_context: AssemblyContext,
            expected_static: Vec<String>,
            expected_per_session: Vec<String>,
            expected_per_turn: Vec<String>,
        }
        #[derive(Deserialize)]
        struct Suite {
            cases: Vec<Case>,
        }
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/prompt_assembly/assembly_steps.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let suite: Suite = serde_json::from_str(&raw).expect("parse assembly_steps.json");
        assert!(!suite.cases.is_empty());
        for case in &suite.cases {
            let b = ContextSourcesBuilder::with_chunks(case.registered_chunks.clone());
            let buckets = b.assemble(&case.assembly_context);
            let ids = |v: &[PromptChunk]| v.iter().map(|c| c.id.clone()).collect::<Vec<_>>();
            assert_eq!(
                ids(&buckets.static_chunks),
                case.expected_static,
                "case `{}` static mismatch",
                case.name
            );
            assert_eq!(
                ids(&buckets.per_session),
                case.expected_per_session,
                "case `{}` per_session mismatch",
                case.name
            );
            assert_eq!(
                ids(&buckets.per_turn),
                case.expected_per_turn,
                "case `{}` per_turn mismatch",
                case.name
            );
        }
    }
}
