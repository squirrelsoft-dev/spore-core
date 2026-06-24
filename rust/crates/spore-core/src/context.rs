//! Issue #7 — `ContextManager`: assemble and maintain the context window.
//!
//! Builds per-turn context from a pre-computed Block-1 [`ComposedPrompt`],
//! per-session metadata (Block 2), and per-turn ephemera (Block 3). Tracks
//! token usage, compacts on threshold, offloads large tool results, and
//! injects just-in-time skill chunks.
//!
//! See `docs/harness-engineering-concepts.md` § "ContextManager" and § "Cache
//! Architecture" for the cross-language rules this module enforces.
//!
//! The trait defined here is the canonical interface for issue #7. The
//! placeholder trait of the same name in [`crate::harness`] is a narrower
//! stub used by the in-tree `StandardHarness` while the wider rewrite lands
//! (see the spec issue's "Implementor notes" for the migration plan).
//!
//! # Post-compaction verification (issue #29)
//!
//! [`CompactionVerifier`] is a lightweight, *synchronous* sensor that runs
//! after the agent produces a compaction summary and before the harness
//! accepts it. It MUST NOT call the model — it is purely computational.
//! [`KeyTermVerifier`] is the standard implementation; it extracts key
//! terms from [`SessionState`] according to the enabled
//! [`CompactionPreserveHints`] and checks they appear in the summary,
//! producing a [`CompactionVerificationResult`].
//!
//! All five hints contribute source terms, each gated on its hint and
//! pushed in this fixed order (issue #47) — this order is the cross-language
//! invariant that determines first-occurrence dedup:
//!
//! 1. `keep_current_task_state` → `SessionState::task_instruction`
//! 2. `keep_open_problems` → each `SessionState::open_problems`
//! 3. `keep_architectural_decisions` → each `SessionState::architectural_decisions`
//! 4. `keep_recent_file_list` → each `SessionState::recent_files`
//! 5. `keep_thinking_blocks` → `SessionState::reasoning_summary`
//!
//! Each source string runs through the same `extract_terms` rule; an
//! empty/unset field contributes no terms.
//!
//! Note: [`CompactionConfig`] gains a `max_compaction_attempts` field but
//! *intentionally* does NOT carry a `verifier` trait object — that would
//! break its `Serialize`/`Deserialize`/`PartialEq` derives and fixture
//! round-tripping. The harness-side retry loop that consumes the verifier
//! is deferred to a follow-up issue; this module only provides the
//! verification primitives.
//!
//! # Configurable compaction window (issue #141)
//!
//! Compaction triggers at `CompactionConfig.threshold × SessionState.window_limit`.
//! Historically `window_limit` was hardcoded to `200_000`, so the trigger never
//! fired for small-context local models (an 8K gemma or a 128K model overruns
//! its real context long before 80% of 200K). Two pieces make the window
//! model-configurable:
//!
//! - [`CompactionConfig::context_length`] (`Option<u32>`) — an optional caller
//!   override. Serialized as ABSENT when `None`, so existing serialized configs
//!   stay byte-identical.
//! - [`StandardContextManager::resolve_context_length`] — the resolver. Fallback
//!   order: configured `context_length` (only when `Some(n)` and `n > 0`) →
//!   the model's `provider().context_window` (when `> 0`) →
//!   [`DEFAULT_CONTEXT_LENGTH`] (`8_000`). An explicit `Some(0)` (or `None`)
//!   falls through to model metadata, then to the default. Configured values are
//!   NOT clamped to the model's real window — a larger configured value is used
//!   as-is.
//!
//! The manager seeds the rich [`SessionState.window_limit`] with the resolved
//! value via [`StandardContextManager::seed_session`]; trigger math
//! ([`StandardContextManager::should_compact`]) is unchanged and automatically
//! respects the seeded window. When the real context length is unknown,
//! [`SessionState::new`] now defaults `window_limit` to the conservative
//! `DEFAULT_CONTEXT_LENGTH` (8_000) rather than the dangerous old 200_000.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::harness::{
    BoxFut, FileRef, SandboxProvider, SessionId, TaskId, ToolOutput,
    ToolResult as HarnessToolResult,
};
use crate::model::{Content, Message, ModelInterface, ModelParams, ModelRequest, Role, ToolSchema};
use crate::tool_registry::TaskPhase;

// ============================================================================
// Forward-declared sibling types (issues #8, #9, #14)
// ============================================================================

// `Guide` and `GuideId` are defined in [`crate::guide_registry`] (issue #9).
// Re-exported here so existing context callers keep working.
pub use crate::guide_registry::{Guide, GuideId};

// `MemoryItem` is defined in [`crate::memory`] (issue #8). Re-exported here
// for downstream callers building `ContextSources`.
pub use crate::memory::MemoryItem;

// `ComposedPrompt` is defined by `PromptChunkRegistry` (issue #24).
use crate::prompt_chunk_registry::CacheBlock;
pub use crate::prompt_chunk_registry::ComposedPrompt;

/// Per-block cache hit signal recorded into `ContextMeta` after each
/// model response. Distinct from [`crate::cache_provider::CacheStats`],
/// which carries token counts and costs parsed from the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CacheBlockHits {
    pub static_hit: Option<bool>,
    pub session_hit: Option<bool>,
    pub history_hit: Option<bool>,
}

impl CacheBlockHits {
    pub fn new(
        static_hit: Option<bool>,
        session_hit: Option<bool>,
        history_hit: Option<bool>,
    ) -> Self {
        Self {
            static_hit,
            session_hit,
            history_hit,
        }
    }
}

pub use crate::cache_provider::{CacheProvider, NullCacheProvider};

// ============================================================================
// Spec-defined types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentStability {
    Static,
    PerSession,
    PerTurn,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PromptSegment {
    pub name: String,
    pub content: String,
    pub stability: SegmentStability,
    #[serde(default)]
    pub cache_breakpoint: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BreakpointInfo {
    pub after_segment: String,
    pub token_offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderedSystemPrompt {
    pub content: String,
    pub breakpoints: Vec<BreakpointInfo>,
    pub static_block_hash: u64,
    pub session_block_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CacheBlockStatus {
    pub static_hit: Option<bool>,
    pub session_hit: Option<bool>,
    pub history_hit: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextMeta {
    pub session_id: SessionId,
    pub turn_number: u32,
    pub active_phase: TaskPhase,
    pub guides_loaded: Vec<GuideId>,
    pub skills_injected: Vec<GuideId>,
    pub compacted: bool,
    pub cache_blocks: CacheBlockStatus,
}

/// Assembled per-turn context.
///
/// Distinct from [`crate::agent::Context`], which is the narrower bundle the
/// agent treats as immutable input. [`Context::into_request`] converts to a
/// [`ModelRequest`] for the agent layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Context {
    pub system_prompt: RenderedSystemPrompt,
    pub messages: Vec<Message>,
    pub tool_schemas: Vec<ToolSchema>,
    pub token_count: u32,
    pub window_limit: u32,
    pub utilization: f32,
    pub meta: ContextMeta,
}

impl Context {
    /// Convert to a [`ModelRequest`]. The system prompt is prepended as a
    /// `Role::System` message so existing agents can consume the assembled
    /// context without a new field.
    pub fn into_request(self, params: ModelParams) -> ModelRequest {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        messages.push(Message {
            role: Role::System,
            content: Content::Text {
                text: self.system_prompt.content,
            },
        });
        messages.extend(self.messages);
        ModelRequest {
            messages,
            tools: self.tool_schemas,
            params,
            stream: false,
        }
    }
}

/// Conservative fallback compaction window when neither the caller's
/// [`CompactionConfig::context_length`] nor the model's
/// `provider().context_window` supplies a usable (`> 0`) value (issue #141).
///
/// Deliberately small (8K, gemma-class) rather than the old 200K: when the
/// real context length is unknown, assume a tight window so compaction still
/// fires rather than silently never running.
pub const DEFAULT_CONTEXT_LENGTH: u32 = 8_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionConfig {
    pub threshold: f32,
    pub preserve_recent_n: u32,
    pub head_tail_tokens: u32,
    pub offload_path: PathBuf,
    /// Maximum number of summarization attempts the harness retry loop
    /// (deferred, see module docs / issue #29) makes before accepting a
    /// failed-verification summary as-is and logging a warn-level event.
    pub max_compaction_attempts: u32,
    /// Optional caller override for the resolved compaction window (issue
    /// #141). When `Some(n)` and `n > 0`, the resolver
    /// ([`StandardContextManager::resolve_context_length`]) uses `n` as the
    /// `window_limit`. `None` (the default) and an explicit `Some(0)` both
    /// fall through to the model's `provider().context_window`, then to
    /// [`DEFAULT_CONTEXT_LENGTH`]. Configured values are NOT clamped to the
    /// model's real window.
    ///
    /// Serialized as ABSENT when `None`, so an existing serialized
    /// `CompactionConfig` stays byte-identical (no new key when unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            threshold: 0.80,
            preserve_recent_n: 8,
            head_tail_tokens: 512,
            offload_path: PathBuf::from(".spore/offload"),
            max_compaction_attempts: 2,
            context_length: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub turn_number: u32,
    pub task_instruction: String,
    pub environment: String,
    pub prior_state: String,
    pub operational_instructions: String,
    pub active_phase: TaskPhase,
    pub message_history: Vec<Message>,
    pub token_budget_used: u32,
    pub window_limit: u32,
    pub guides_loaded: Vec<GuideId>,
    /// Skills pending Block-3 injection on the next assemble. Cleared after
    /// each assemble — skills are ephemeral, one turn only.
    pub pending_skill_injections: Vec<Guide>,
    pub budget_warning_active: bool,
    /// Open problems feeding the `keep_open_problems` hint (issue #47).
    #[serde(default)]
    pub open_problems: Vec<String>,
    /// Architectural decisions feeding `keep_architectural_decisions` (#47).
    #[serde(default)]
    pub architectural_decisions: Vec<String>,
    /// Recently touched file paths feeding `keep_recent_file_list` (#47).
    /// Typed as `String`, not `PathBuf` — keeps tokenization byte-identical
    /// across languages (no per-language path semantics).
    #[serde(default)]
    pub recent_files: Vec<String>,
    /// Reasoning summary feeding the `keep_thinking_blocks` hint (issue #47).
    #[serde(default)]
    pub reasoning_summary: String,
}

impl SessionState {
    pub fn new(
        session_id: SessionId,
        task_id: TaskId,
        task_instruction: impl Into<String>,
    ) -> Self {
        Self {
            session_id,
            task_id,
            turn_number: 0,
            task_instruction: task_instruction.into(),
            environment: String::new(),
            prior_state: String::new(),
            operational_instructions: String::new(),
            active_phase: TaskPhase::Execution,
            message_history: Vec::new(),
            token_budget_used: 0,
            window_limit: DEFAULT_CONTEXT_LENGTH,
            guides_loaded: Vec::new(),
            pending_skill_injections: Vec::new(),
            budget_warning_active: false,
            open_problems: Vec::new(),
            architectural_decisions: Vec::new(),
            recent_files: Vec::new(),
            reasoning_summary: String::new(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextSources {
    pub guides: Vec<Guide>,
    pub memory: Vec<MemoryItem>,
    pub tool_schemas: Vec<ToolSchema>,
    pub composed_prompt: ComposedPrompt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionRequest {
    pub messages_to_compact: Vec<Message>,
    pub preserve_hints: CompactionPreserveHints,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionPreserveHints {
    pub keep_architectural_decisions: bool,
    pub keep_open_problems: bool,
    pub keep_current_task_state: bool,
    pub keep_recent_file_list: bool,
    /// Defaults to `true` — never compact active reasoning blocks.
    pub keep_thinking_blocks: bool,
}

impl Default for CompactionPreserveHints {
    fn default() -> Self {
        Self {
            keep_architectural_decisions: true,
            keep_open_problems: true,
            keep_current_task_state: true,
            keep_recent_file_list: true,
            keep_thinking_blocks: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionResult {
    pub summary_message: Message,
    pub tokens_reclaimed: u32,
    pub messages_removed: u32,
}

// ============================================================================
// Post-compaction verification (issue #29)
// ============================================================================

/// Outcome of a [`CompactionVerifier::verify`] check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionVerificationResult {
    pub passed: bool,
    /// Items from the preservation list not found in the summary, in
    /// first-occurrence order (already lowercased/normalized).
    pub missing_items: Vec<String>,
    pub detail: String,
}

/// A lightweight, synchronous post-compaction sensor.
///
/// Implementations run after the agent produces a summary and before the
/// harness accepts it. They are purely computational and MUST NOT call the
/// model.
pub trait CompactionVerifier: Send + Sync {
    fn verify(
        &self,
        summary: &str,
        hints: &CompactionPreserveHints,
        session_state: &SessionState,
    ) -> CompactionVerificationResult;
}

/// Standard [`CompactionVerifier`]: extracts key terms from the session
/// state per the enabled hints and checks they appear in the summary.
///
/// All five hints contribute source terms, each gated on its hint and pushed
/// in a fixed order (issue #47) — `keep_current_task_state` →
/// `keep_open_problems` → `keep_architectural_decisions` →
/// `keep_recent_file_list` → `keep_thinking_blocks`. This order pins the
/// cross-language first-occurrence dedup. See module docs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeyTermVerifier;

impl KeyTermVerifier {
    /// Tokenize a source string into normalized key terms: lowercase, split
    /// on runs of any character that is NOT an ASCII `a-z` letter or `0-9`
    /// digit, and discard tokens shorter than 4 characters.
    fn extract_terms(source: &str) -> Vec<String> {
        source
            .to_lowercase()
            .split(|c: char| !c.is_ascii_lowercase() && !c.is_ascii_digit())
            .filter(|tok| tok.len() >= 4)
            .map(|tok| tok.to_string())
            .collect()
    }
}

impl CompactionVerifier for KeyTermVerifier {
    fn verify(
        &self,
        summary: &str,
        hints: &CompactionPreserveHints,
        session_state: &SessionState,
    ) -> CompactionVerificationResult {
        // Step 1: collect source strings from enabled hints, each gated on
        // its hint and pushed in this fixed order (issue #47). This order is
        // the cross-language invariant that determines first-occurrence dedup.
        let mut sources: Vec<&str> = Vec::new();
        if hints.keep_current_task_state {
            sources.push(session_state.task_instruction.as_str());
        }
        if hints.keep_open_problems {
            for item in &session_state.open_problems {
                sources.push(item.as_str());
            }
        }
        if hints.keep_architectural_decisions {
            for item in &session_state.architectural_decisions {
                sources.push(item.as_str());
            }
        }
        if hints.keep_recent_file_list {
            for item in &session_state.recent_files {
                sources.push(item.as_str());
            }
        }
        if hints.keep_thinking_blocks {
            sources.push(session_state.reasoning_summary.as_str());
        }

        // Step 2: build the term list and dedupe preserving first-occurrence
        // order.
        let mut terms: Vec<String> = Vec::new();
        for source in sources {
            for term in Self::extract_terms(source) {
                if !terms.contains(&term) {
                    terms.push(term);
                }
            }
        }

        // Step 3: a term is present iff the lowercased summary contains it.
        let summary_lower = summary.to_lowercase();
        let missing_items: Vec<String> = terms
            .iter()
            .filter(|term| !summary_lower.contains(term.as_str()))
            .cloned()
            .collect();

        // Step 4 + 5.
        let total_terms = terms.len();
        let passed = missing_items.is_empty();
        let detail = if passed {
            format!("all {total_terms} key term(s) present")
        } else {
            format!(
                "missing {} of {} key term(s): {}",
                missing_items.len(),
                total_terms,
                missing_items.join(", ")
            )
        };

        CompactionVerificationResult {
            passed,
            missing_items,
            detail,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContextError {
    #[error("token count failed")]
    TokenCountFailed,
    #[error("compaction failed: {reason}")]
    CompactionFailed { reason: String },
    #[error("assembly failed: {reason}")]
    AssemblyFailed { reason: String },
    /// A cache block's content hash changed when it was expected to be stable.
    ///
    /// Both Block 1 (`CacheBlock::Static`) and Block 2 (`CacheBlock::PerSession`)
    /// halt the run on a mid-session mismatch — they are treated consistently.
    /// A Block-2 change mid-session means session-stable content mutated and
    /// every subsequent turn would silently pay full input-token cost; rather
    /// than warn, the run stops so the caller can fix the source.
    ///
    /// `turn_number` is the turn on which the mismatch was detected (Block 2
    /// only halts when `turn_number > 1`; the turn-1 assemble records the
    /// baseline). Estimated cache-cost-delta tracking (`UnexpectedMiss`) is a
    /// separate observability concern tracked in issue #90.
    #[error(
        "cache hash mismatch on block {block:?} at turn {turn_number}: expected {expected}, got {actual}"
    )]
    CacheHashMismatch {
        block: CacheBlock,
        expected: u64,
        actual: u64,
        turn_number: u32,
    },
}

// ============================================================================
// Trait
// ============================================================================

#[trait_variant::make(Send)]
pub trait ContextManager: Send + Sync {
    async fn assemble(
        &self,
        state: &SessionState,
        sources: &ContextSources,
    ) -> Result<Context, ContextError>;

    fn append_tool_result<'a>(
        &'a self,
        state: &'a mut SessionState,
        result: &'a HarnessToolResult,
        sandbox: &'a dyn SandboxProvider,
    ) -> BoxFut<'a, Result<(), ContextError>>;

    fn append_response(&self, state: &mut SessionState, response: String);

    fn should_compact(&self, state: &SessionState) -> bool;

    fn prepare_compaction(&self, state: &SessionState) -> Result<CompactionRequest, ContextError>;

    fn apply_compaction(
        &self,
        state: &mut SessionState,
        result: CompactionResult,
    ) -> Result<(), ContextError>;

    fn inject_skill(&self, context: &mut Context, skill: &Guide) -> Result<(), ContextError>;

    fn record_cache_result(&self, context: &mut Context, cache_stats: CacheBlockHits);
}

// ============================================================================
// Standard implementation
// ============================================================================

struct CacheHashMemo {
    static_hash: Option<u64>,
    session_hash: Option<u64>,
}

/// Reference `ContextManager` implementation. Enforces the assembly order
/// from the spec: Block 1 from `ComposedPrompt`, Block 2 from `SessionState`,
/// Block 3 from per-turn ephemera. Tool schemas are sorted by name.
pub struct StandardContextManager<M: ModelInterface> {
    model: Arc<M>,
    cache_provider: Arc<dyn CacheProvider>,
    compaction: CompactionConfig,
    /// Sandbox-side threshold above which `append_tool_result` head+tail
    /// truncates via `SandboxProvider::handle_large_output`.
    offload_threshold_bytes: usize,
    memo: Mutex<CacheHashMemo>,
}

impl<M: ModelInterface> StandardContextManager<M> {
    pub fn new(
        model: Arc<M>,
        cache_provider: Arc<dyn CacheProvider>,
        compaction: CompactionConfig,
    ) -> Self {
        Self {
            model,
            cache_provider,
            compaction,
            offload_threshold_bytes: 32 * 1024,
            memo: Mutex::new(CacheHashMemo {
                static_hash: None,
                session_hash: None,
            }),
        }
    }

    pub fn with_offload_threshold(mut self, bytes: usize) -> Self {
        self.offload_threshold_bytes = bytes;
        self
    }

    /// Resolve the compaction window (issue #141). Fallback order:
    ///
    /// 1. configured [`CompactionConfig::context_length`] when `Some(n)` and
    ///    `n > 0`,
    /// 2. else the model's `provider().context_window` when `> 0`,
    /// 3. else [`DEFAULT_CONTEXT_LENGTH`].
    ///
    /// An explicit `Some(0)` (and `None`) falls through to model metadata, then
    /// to the default. The configured value is NOT clamped to the model's real
    /// window — a larger configured value is used as-is.
    pub fn resolve_context_length(&self) -> u32 {
        if let Some(n) = self.compaction.context_length {
            if n > 0 {
                return n;
            }
        }
        let model_window = self.model.provider().context_window;
        if model_window > 0 {
            return model_window;
        }
        DEFAULT_CONTEXT_LENGTH
    }

    /// Build the initial rich [`SessionState`] for a run, seeding its
    /// `window_limit` from [`Self::resolve_context_length`] (issue #141).
    ///
    /// The manager owns seeding so the resolved window has a single production
    /// seam — callers get a `SessionState` whose `window_limit` already
    /// reflects the configured/model/default resolution rather than the bare
    /// [`SessionState::new`] constructor default.
    pub fn seed_session(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        instruction: impl Into<String>,
    ) -> SessionState {
        let mut state = SessionState::new(session_id, task_id, instruction);
        state.window_limit = self.resolve_context_length();
        state
    }

    fn build_session_segments(&self, state: &SessionState) -> Vec<PromptSegment> {
        // Order is load-bearing for prefix-cache stability.
        vec![
            PromptSegment {
                name: "task".into(),
                content: state.task_instruction.clone(),
                stability: SegmentStability::PerSession,
                cache_breakpoint: false,
            },
            PromptSegment {
                name: "environment".into(),
                content: state.environment.clone(),
                stability: SegmentStability::PerSession,
                cache_breakpoint: false,
            },
            PromptSegment {
                name: "prior_state".into(),
                content: state.prior_state.clone(),
                stability: SegmentStability::PerSession,
                cache_breakpoint: false,
            },
            PromptSegment {
                name: "operational".into(),
                content: state.operational_instructions.clone(),
                stability: SegmentStability::PerSession,
                cache_breakpoint: true,
            },
        ]
    }
}

fn hash_strings<'a, I: IntoIterator<Item = &'a str>>(parts: I) -> u64 {
    let mut h = DefaultHasher::new();
    for p in parts {
        p.hash(&mut h);
        // Length-prefixed-style separator to avoid collisions across
        // concatenations like ["a","bc"] vs ["ab","c"].
        0u8.hash(&mut h);
    }
    h.finish()
}

fn segments_hash(segments: &[PromptSegment]) -> u64 {
    let mut h = DefaultHasher::new();
    for s in segments {
        s.name.hash(&mut h);
        s.content.hash(&mut h);
        (s.stability as u8).hash(&mut h);
        s.cache_breakpoint.hash(&mut h);
    }
    h.finish()
}

fn render_segments(block_1: &str, segments: &[PromptSegment]) -> (String, Vec<BreakpointInfo>) {
    // Rough token offset proxy: chars/4. Real token offsets are reported by
    // the model; this is only useful for cache-marker placement which is
    // counted in segments, not tokens.
    let mut content = String::with_capacity(block_1.len() + 1024);
    content.push_str(block_1);
    let mut breakpoints = Vec::new();
    // Block 1 always ends with an implicit breakpoint (spec: cache_provider
    // inserts breakpoint after Block 1).
    breakpoints.push(BreakpointInfo {
        after_segment: "__block_1__".into(),
        token_offset: (content.len() as u32) / 4,
    });
    for seg in segments {
        if !content.ends_with('\n') {
            content.push('\n');
        }
        content.push_str(&seg.content);
        if seg.cache_breakpoint {
            breakpoints.push(BreakpointInfo {
                after_segment: seg.name.clone(),
                token_offset: (content.len() as u32) / 4,
            });
        }
    }
    (content, breakpoints)
}

impl<M: ModelInterface + 'static> ContextManager for StandardContextManager<M> {
    async fn assemble(
        &self,
        state: &SessionState,
        sources: &ContextSources,
    ) -> Result<Context, ContextError> {
        // ── BLOCK 1 hash check ───────────────────────────────────────────
        let static_hash = sources.composed_prompt.block_1_hash;
        {
            let mut memo = self.memo.lock().unwrap();
            if let Some(prev) = memo.static_hash {
                if prev != static_hash {
                    return Err(ContextError::CacheHashMismatch {
                        block: CacheBlock::Static,
                        expected: prev,
                        actual: static_hash,
                        turn_number: state.turn_number,
                    });
                }
            } else {
                memo.static_hash = Some(static_hash);
            }
        }

        // ── BLOCK 2 (PerSession) ─────────────────────────────────────────
        let mut segments = self.build_session_segments(state);
        let session_hash = segments_hash(&segments);
        {
            let mut memo = self.memo.lock().unwrap();
            if let Some(prev) = memo.session_hash {
                if prev != session_hash && state.turn_number > 1 {
                    // Block 2 (PerSession) is expected to be stable for the
                    // life of the session. A mid-session change means cost
                    // would silently spike; halt consistently with Block 1
                    // (#32). We return BEFORE updating the memo — the run is
                    // halting, so there is no "rest of the session" to track.
                    return Err(ContextError::CacheHashMismatch {
                        block: CacheBlock::PerSession,
                        expected: prev,
                        actual: session_hash,
                        turn_number: state.turn_number,
                    });
                }
            }
            memo.session_hash = Some(session_hash);
        }

        // ── BLOCK 3 (PerTurn, never cached) ──────────────────────────────
        if state.budget_warning_active {
            segments.push(PromptSegment {
                name: "budget_warning".into(),
                content: format!(
                    "[BUDGET] {} of {} tokens used.",
                    state.token_budget_used, state.window_limit
                ),
                stability: SegmentStability::PerTurn,
                cache_breakpoint: false,
            });
        }
        for skill in &state.pending_skill_injections {
            segments.push(PromptSegment {
                name: format!("skill:{}", skill.id.0),
                content: skill.content.clone(),
                stability: SegmentStability::PerTurn,
                cache_breakpoint: false,
            });
        }

        // ── Render ───────────────────────────────────────────────────────
        let (rendered, breakpoints) =
            render_segments(sources.composed_prompt.rendered_str(), &segments);
        let system_prompt = RenderedSystemPrompt {
            content: rendered,
            breakpoints,
            static_block_hash: static_hash,
            session_block_hash: session_hash,
        };

        // ── Tool schemas: sort by name (spec: deterministic ordering) ────
        let mut tool_schemas = sources.tool_schemas.clone();
        tool_schemas.sort_by(|a, b| a.name.cmp(&b.name));

        // ── Message history (cache breakpoint metadata is provider-specific
        //    and applied by CacheProvider::annotate below).
        let messages = state.message_history.clone();

        // ── Token count (from ModelInterface, not estimated) ─────────────
        let req = ModelRequest {
            messages: {
                let mut m = Vec::with_capacity(messages.len() + 1);
                m.push(Message {
                    role: Role::System,
                    content: Content::Text {
                        text: system_prompt.content.clone(),
                    },
                });
                m.extend(messages.iter().cloned());
                m
            },
            tools: tool_schemas.clone(),
            params: ModelParams::default(),
            stream: false,
        };
        let token_count = self
            .model
            .count_tokens(&req)
            .await
            .map_err(|_| ContextError::TokenCountFailed)?;
        let utilization = if state.window_limit == 0 {
            0.0
        } else {
            token_count as f32 / state.window_limit as f32
        };

        let meta = ContextMeta {
            session_id: state.session_id.clone(),
            turn_number: state.turn_number,
            active_phase: state.active_phase,
            guides_loaded: state.guides_loaded.clone(),
            skills_injected: state
                .pending_skill_injections
                .iter()
                .map(|g| g.id.clone())
                .collect(),
            compacted: false,
            cache_blocks: CacheBlockStatus::default(),
        };

        let mut context = Context {
            system_prompt,
            messages,
            tool_schemas,
            token_count,
            window_limit: state.window_limit,
            utilization,
            meta,
        };

        self.cache_provider.annotate(&mut context);
        Ok(context)
    }

    fn append_tool_result<'a>(
        &'a self,
        state: &'a mut SessionState,
        result: &'a HarnessToolResult,
        sandbox: &'a dyn SandboxProvider,
    ) -> BoxFut<'a, Result<(), ContextError>> {
        Box::pin(async move {
            let text = match &result.output {
                ToolOutput::Success { content, .. } => content.clone(),
                ToolOutput::Error { message, .. } => format!("[error] {message}"),
                // Normally normalized into an `Error` before append; defensive.
                ToolOutput::SandboxViolation { violation } => {
                    format!("[error] sandbox violation: {violation:?}")
                }
                ToolOutput::WaitingForHuman { .. } => "[waiting]".into(),
                ToolOutput::Escalate { .. } => "[escalate]".into(),
                ToolOutput::AwaitingClarification { .. } => "[clarification]".into(),
                ToolOutput::Consult { .. } => "[consult]".into(),
            };

            // Spec rule: head+tail truncate, offload full to filesystem.
            let final_text = if text.len() > self.offload_threshold_bytes {
                let truncated = sandbox
                    .handle_large_output(
                        text,
                        &result.call_id,
                        self.compaction.head_tail_tokens,
                        self.compaction.head_tail_tokens,
                    )
                    .await;
                format_truncated(&truncated.content, truncated.full_ref.as_ref())
            } else {
                text
            };

            state.message_history.push(Message {
                role: Role::Tool,
                content: Content::Text { text: final_text },
            });
            Ok(())
        })
    }

    fn append_response(&self, state: &mut SessionState, response: String) {
        state.message_history.push(Message {
            role: Role::Assistant,
            content: Content::Text { text: response },
        });
    }

    fn should_compact(&self, state: &SessionState) -> bool {
        if state.window_limit == 0 {
            return false;
        }
        let util = state.token_budget_used as f32 / state.window_limit as f32;
        util >= self.compaction.threshold
    }

    fn prepare_compaction(&self, state: &SessionState) -> Result<CompactionRequest, ContextError> {
        let n = state.message_history.len();
        let keep = self.compaction.preserve_recent_n as usize;
        if n <= keep {
            return Ok(CompactionRequest {
                messages_to_compact: Vec::new(),
                preserve_hints: CompactionPreserveHints::default(),
            });
        }
        let cut = n - keep;
        let messages_to_compact = state.message_history[..cut].to_vec();
        Ok(CompactionRequest {
            messages_to_compact,
            preserve_hints: CompactionPreserveHints::default(),
        })
    }

    fn apply_compaction(
        &self,
        state: &mut SessionState,
        result: CompactionResult,
    ) -> Result<(), ContextError> {
        let n = state.message_history.len();
        let keep = self.compaction.preserve_recent_n as usize;
        if n <= keep {
            return Err(ContextError::CompactionFailed {
                reason: "history shorter than preserve_recent_n".into(),
            });
        }
        let cut = n - keep;
        let mut new_history = Vec::with_capacity(keep + 1);
        new_history.push(result.summary_message);
        new_history.extend(state.message_history[cut..].iter().cloned());
        state.message_history = new_history;
        state.token_budget_used = state
            .token_budget_used
            .saturating_sub(result.tokens_reclaimed);
        Ok(())
    }

    fn inject_skill(&self, context: &mut Context, skill: &Guide) -> Result<(), ContextError> {
        // Block-3 ephemeral injection: append to system prompt content, do
        // not modify message history, do not invalidate Block 1 or Block 2
        // (their hashes are untouched).
        if !context.system_prompt.content.ends_with('\n') {
            context.system_prompt.content.push('\n');
        }
        context
            .system_prompt
            .content
            .push_str(&format!("[SKILL:{}]\n{}", skill.id.0, skill.content));
        context.meta.skills_injected.push(skill.id.clone());
        Ok(())
    }

    fn record_cache_result(&self, context: &mut Context, cache_stats: CacheBlockHits) {
        context.meta.cache_blocks = CacheBlockStatus {
            static_hit: cache_stats.static_hit,
            session_hit: cache_stats.session_hit,
            history_hit: cache_stats.history_hit,
        };
    }
}

fn format_truncated(head_tail: &str, full_ref: Option<&FileRef>) -> String {
    match full_ref {
        Some(r) => format!(
            "{}\n\n[truncated; full output at {} ({} bytes)]",
            head_tail, r.path, r.byte_len
        ),
        None => format!("{}\n\n[truncated]", head_tail),
    }
}

#[allow(dead_code)]
fn _hash_strings_ref() {
    // Keeps `hash_strings` available for downstream issues without a
    // dead-code warning; PromptChunkRegistry (#14) will use it.
    let _ = hash_strings(std::iter::empty::<&str>());
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{ToolOutput, ToolResult as HarnessToolResult};
    use crate::model::{
        ModelError, ModelRequest, ModelResponse, ModelStream, ProviderInfo, StopReason, TokenUsage,
    };
    #[allow(unused_imports)]
    use crate::tool_registry::TaskPhase as _TaskPhase;
    use std::sync::atomic::{AtomicU32, Ordering};

    // ── Test doubles ─────────────────────────────────────────────────────

    struct FakeModel {
        per_call: AtomicU32,
    }
    impl FakeModel {
        fn new() -> Self {
            Self {
                per_call: AtomicU32::new(100),
            }
        }
    }
    impl ModelInterface for FakeModel {
        fn call<'a>(
            &'a self,
            _req: ModelRequest,
        ) -> BoxFut<'a, Result<ModelResponse, ModelError>> {
            Box::pin(async move {
            Ok(ModelResponse {
                content: vec![],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            })
            })
        }
        fn call_streaming<'a>(
            &'a self,
            _req: ModelRequest,
        ) -> BoxFut<'a, Result<ModelStream, ModelError>> {
            Box::pin(async move { unimplemented!() })
        }
        fn count_tokens<'a>(
            &'a self,
            _req: &'a ModelRequest,
        ) -> BoxFut<'a, Result<u32, ModelError>> {
            Box::pin(async move { Ok(self.per_call.load(Ordering::SeqCst)) })
        }
        fn provider(&self) -> ProviderInfo {
            ProviderInfo {
                name: "fake".into(),
                model_id: "fake".into(),
                context_window: 200_000,
            }
        }
    }

    struct CountingCache {
        calls: std::sync::atomic::AtomicU32,
    }
    impl CountingCache {
        fn new() -> Self {
            Self {
                calls: AtomicU32::new(0),
            }
        }
    }
    impl CacheProvider for CountingCache {
        fn supports_caching(&self) -> bool {
            true
        }
        fn annotate(&self, _context: &mut Context) -> crate::cache_provider::CacheAnnotationResult {
            self.calls.fetch_add(1, Ordering::SeqCst);
            crate::cache_provider::CacheAnnotationResult::default()
        }
        fn provider_name(&self) -> &'static str {
            "counting"
        }
    }

    struct PassthroughSandbox;
    impl SandboxProvider for PassthroughSandbox {
        fn validate<'a>(
            &'a self,
            _call: &'a crate::model::ToolCall,
        ) -> BoxFut<'a, Result<(), crate::harness::SandboxViolation>> {
            Box::pin(async { Ok(()) })
        }
    }

    fn sources(rendered: &str, hash: u64, schemas: Vec<ToolSchema>) -> ContextSources {
        ContextSources {
            guides: vec![],
            memory: vec![],
            tool_schemas: schemas,
            composed_prompt: ComposedPrompt {
                chunks: vec![],
                block_1_hash: hash,
                block_2_hash: 0,
                rendered: Some(rendered.into()),
            },
        }
    }

    fn state() -> SessionState {
        let mut s = SessionState::new(SessionId::new("s1"), TaskId::new("t1"), "do the thing");
        s.window_limit = 1000;
        s.token_budget_used = 100;
        s
    }

    fn mk() -> StandardContextManager<FakeModel> {
        StandardContextManager::new(
            Arc::new(FakeModel::new()),
            Arc::new(NullCacheProvider),
            CompactionConfig::default(),
        )
    }

    // ── Rule: Assemble before every turn ─────────────────────────────────

    #[tokio::test]
    async fn assemble_returns_context_with_token_count_from_model() {
        let mgr = mk();
        let st = state();
        let ctx = mgr
            .assemble(&st, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
        assert_eq!(ctx.token_count, 100); // from FakeModel
        assert_eq!(ctx.window_limit, 1000);
        assert!((ctx.utilization - 0.1).abs() < f32::EPSILON);
    }

    // ── Rule: Block 1 hash invariance ────────────────────────────────────

    #[tokio::test]
    async fn block_1_hash_mismatch_is_an_error() {
        let mgr = mk();
        let st = state();
        mgr.assemble(&st, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
        let err = mgr
            .assemble(&st, &sources("BLOCK1", 0xCD, vec![]))
            .await
            .unwrap_err();
        match err {
            ContextError::CacheHashMismatch { block, .. } => {
                assert_eq!(block, CacheBlock::Static)
            }
            e => panic!("wrong error: {e:?}"),
        }
    }

    // ── Rule: Block 2 (PerSession) hash invariance (#32) ─────────────────

    #[tokio::test]
    async fn block_2_hash_mismatch_mid_session_halts() {
        let mgr = mk();
        // Turn 1 records the Block-2 baseline.
        let mut st = state();
        st.turn_number = 1;
        mgr.assemble(&st, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
        // Turn 2 with mutated session content (changed task instruction)
        // must halt, mirroring Block 1.
        let mut st2 = state();
        st2.turn_number = 2;
        st2.task_instruction = "a different task instruction".into();
        let err = mgr
            .assemble(&st2, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap_err();
        match err {
            ContextError::CacheHashMismatch {
                block, turn_number, ..
            } => {
                assert_eq!(block, CacheBlock::PerSession);
                assert_eq!(turn_number, 2);
            }
            e => panic!("wrong error: {e:?}"),
        }
    }

    #[tokio::test]
    async fn block_2_stable_across_turns_does_not_halt() {
        let mgr = mk();
        let mut st = state();
        st.turn_number = 1;
        mgr.assemble(&st, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
        // Turn 2+ with identical session content → Ok.
        let mut st2 = state();
        st2.turn_number = 2;
        mgr.assemble(&st2, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
        let mut st3 = state();
        st3.turn_number = 3;
        mgr.assemble(&st3, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn block_2_change_at_turn_1_does_not_halt() {
        let mgr = mk();
        // First assemble records the baseline. Even if turn-1 content differs
        // from any prior memo state, the `turn_number > 1` guard means turn 1
        // never halts on a Block-2 mismatch.
        let mut st = state();
        st.turn_number = 1;
        st.task_instruction = "a brand new baseline instruction".into();
        mgr.assemble(&st, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
    }

    // ── Rule: Tool schemas sorted by name ────────────────────────────────

    #[tokio::test]
    async fn tool_schemas_are_sorted_by_name() {
        let mgr = mk();
        let st = state();
        let schemas = vec![
            ToolSchema {
                name: "zebra".into(),
                description: "".into(),
                input_schema: serde_json::json!({}),
            },
            ToolSchema {
                name: "apple".into(),
                description: "".into(),
                input_schema: serde_json::json!({}),
            },
            ToolSchema {
                name: "mango".into(),
                description: "".into(),
                input_schema: serde_json::json!({}),
            },
        ];
        let ctx = mgr
            .assemble(&st, &sources("BLOCK1", 0xAB, schemas))
            .await
            .unwrap();
        let names: Vec<&str> = ctx.tool_schemas.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["apple", "mango", "zebra"]);
    }

    // ── Rule: Compact at threshold (default 80%) ─────────────────────────

    #[test]
    fn should_compact_at_threshold() {
        let mgr = mk();
        let mut st = state();
        st.window_limit = 1000;
        st.token_budget_used = 799;
        assert!(!mgr.should_compact(&st));
        st.token_budget_used = 800;
        assert!(mgr.should_compact(&st));
        st.token_budget_used = 900;
        assert!(mgr.should_compact(&st));
    }

    #[test]
    fn should_compact_handles_zero_window() {
        let mgr = mk();
        let mut st = state();
        st.window_limit = 0;
        assert!(!mgr.should_compact(&st));
    }

    // ── Rule: Compaction preserves recent N + preserve hints ─────────────

    #[test]
    fn prepare_compaction_keeps_recent_n_and_uses_default_hints() {
        let mgr = mk();
        let mut st = state();
        for i in 0..20 {
            st.message_history.push(Message {
                role: Role::Assistant,
                content: Content::Text {
                    text: format!("m{i}"),
                },
            });
        }
        let req = mgr.prepare_compaction(&st).unwrap();
        assert_eq!(req.messages_to_compact.len(), 12);
        assert!(req.preserve_hints.keep_thinking_blocks);
        assert!(req.preserve_hints.keep_architectural_decisions);
        assert!(req.preserve_hints.keep_open_problems);
    }

    #[test]
    fn apply_compaction_replaces_old_with_summary() {
        let mgr = mk();
        let mut st = state();
        for i in 0..20 {
            st.message_history.push(Message {
                role: Role::Assistant,
                content: Content::Text {
                    text: format!("m{i}"),
                },
            });
        }
        st.token_budget_used = 800;
        let summary = Message {
            role: Role::Assistant,
            content: Content::Text {
                text: "summary".into(),
            },
        };
        mgr.apply_compaction(
            &mut st,
            CompactionResult {
                summary_message: summary,
                tokens_reclaimed: 500,
                messages_removed: 12,
            },
        )
        .unwrap();
        // 1 summary + 8 preserved recents
        assert_eq!(st.message_history.len(), 9);
        assert_eq!(st.token_budget_used, 300);
        match &st.message_history[0].content {
            Content::Text { text } => assert_eq!(text, "summary"),
            _ => panic!(),
        }
    }

    #[test]
    fn apply_compaction_fails_when_history_too_short() {
        let mgr = mk();
        let mut st = state();
        // Default preserve_recent_n is 8; 4 messages is too short.
        for i in 0..4 {
            st.message_history.push(Message {
                role: Role::Assistant,
                content: Content::Text {
                    text: format!("m{i}"),
                },
            });
        }
        let err = mgr
            .apply_compaction(
                &mut st,
                CompactionResult {
                    summary_message: Message {
                        role: Role::Assistant,
                        content: Content::Text { text: "x".into() },
                    },
                    tokens_reclaimed: 0,
                    messages_removed: 0,
                },
            )
            .unwrap_err();
        assert!(matches!(err, ContextError::CompactionFailed { .. }));
    }

    // ── Rule: append_tool_result head+tail truncates large content ───────

    #[tokio::test]
    async fn append_tool_result_truncates_large_output() {
        let mgr = mk().with_offload_threshold(64);
        let mut st = state();
        let sb = PassthroughSandbox;
        let big = "x".repeat(8 * 1024);
        let result = HarnessToolResult {
            call_id: "c1".into(),
            output: ToolOutput::Success {
                content: big.clone(),
                truncated: false,
            },
        };
        mgr.append_tool_result(&mut st, &result, &sb).await.unwrap();
        assert_eq!(st.message_history.len(), 1);
        let text = match &st.message_history[0].content {
            Content::Text { text } => text.clone(),
            _ => panic!(),
        };
        // Must have been touched by truncation pipeline.
        assert!(text.contains("[truncated"));
        assert!(text.len() < big.len());
    }

    #[tokio::test]
    async fn append_tool_result_small_output_passes_through() {
        let mgr = mk();
        let mut st = state();
        let sb = PassthroughSandbox;
        let result = HarnessToolResult {
            call_id: "c1".into(),
            output: ToolOutput::Success {
                content: "hello".into(),
                truncated: false,
            },
        };
        mgr.append_tool_result(&mut st, &result, &sb).await.unwrap();
        match &st.message_history[0].content {
            Content::Text { text } => assert_eq!(text, "hello"),
            _ => panic!(),
        }
    }

    // ── Rule: append_response appends an Assistant message ───────────────

    #[test]
    fn append_response_pushes_assistant_message() {
        let mgr = mk();
        let mut st = state();
        mgr.append_response(&mut st, "ack".into());
        assert_eq!(st.message_history.len(), 1);
        assert_eq!(st.message_history[0].role, Role::Assistant);
    }

    // ── Rule: inject_skill is ephemeral, no history mutation, no cache
    //         invalidation (Block 1 and Block 2 hashes unchanged) ─────────

    #[tokio::test]
    async fn inject_skill_does_not_touch_history_or_static_hashes() {
        let mgr = mk();
        let st = state();
        let mut ctx = mgr
            .assemble(&st, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
        let before_static = ctx.system_prompt.static_block_hash;
        let before_session = ctx.system_prompt.session_block_hash;
        let before_messages = ctx.messages.len();
        mgr.inject_skill(&mut ctx, &Guide::skill("rust-style", "prefer iterators"))
            .unwrap();
        assert_eq!(ctx.system_prompt.static_block_hash, before_static);
        assert_eq!(ctx.system_prompt.session_block_hash, before_session);
        assert_eq!(ctx.messages.len(), before_messages);
        assert!(ctx.system_prompt.content.contains("[SKILL:rust-style]"));
        assert_eq!(ctx.meta.skills_injected.len(), 1);
    }

    // ── Rule: record_cache_result updates ContextMeta ────────────────────

    #[tokio::test]
    async fn record_cache_result_updates_meta() {
        let mgr = mk();
        let st = state();
        let mut ctx = mgr
            .assemble(&st, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
        mgr.record_cache_result(
            &mut ctx,
            CacheBlockHits::new(Some(true), Some(false), Some(true)),
        );
        assert_eq!(ctx.meta.cache_blocks.static_hit, Some(true));
        assert_eq!(ctx.meta.cache_blocks.session_hit, Some(false));
        assert_eq!(ctx.meta.cache_blocks.history_hit, Some(true));
    }

    // ── Rule: CacheProvider.annotate is invoked by assemble ──────────────

    #[tokio::test]
    async fn cache_provider_annotate_is_called_each_assemble() {
        let cache = Arc::new(CountingCache::new());
        let mgr = StandardContextManager::new(
            Arc::new(FakeModel::new()),
            cache.clone(),
            CompactionConfig::default(),
        );
        let st = state();
        let srcs = sources("BLOCK1", 0xAB, vec![]);
        mgr.assemble(&st, &srcs).await.unwrap();
        mgr.assemble(&st, &srcs).await.unwrap();
        assert_eq!(cache.calls.load(Ordering::SeqCst), 2);
    }

    // ── Rule: Pending skill injections become Block-3 segments and are
    //         reflected in ContextMeta.skills_injected ───────────────────

    #[tokio::test]
    async fn pending_skill_injections_appear_in_meta() {
        let mgr = mk();
        let mut st = state();
        st.pending_skill_injections.push(Guide::skill("g1", "do x"));
        let ctx = mgr
            .assemble(&st, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
        assert_eq!(ctx.meta.skills_injected, vec![GuideId::new("g1")]);
        assert!(ctx.system_prompt.content.contains("do x"));
    }

    // ── Rule: Budget warning lives in Block 3 only when active ───────────

    #[tokio::test]
    async fn budget_warning_only_when_active() {
        let mgr = mk();
        let mut st = state();
        let ctx_off = mgr
            .assemble(&st, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
        assert!(!ctx_off.system_prompt.content.contains("[BUDGET]"));
        st.budget_warning_active = true;
        let ctx_on = mgr
            .assemble(&st, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap();
        assert!(ctx_on.system_prompt.content.contains("[BUDGET]"));
    }

    // ── Rule: TokenCountFailed surfaces when ModelInterface fails ────────

    struct FailingModel;
    impl ModelInterface for FailingModel {
        fn call<'a>(&'a self, _r: ModelRequest) -> BoxFut<'a, Result<ModelResponse, ModelError>> {
            Box::pin(async move { unimplemented!() })
        }
        fn call_streaming<'a>(
            &'a self,
            _r: ModelRequest,
        ) -> BoxFut<'a, Result<ModelStream, ModelError>> {
            Box::pin(async move { unimplemented!() })
        }
        fn count_tokens<'a>(&'a self, _r: &'a ModelRequest) -> BoxFut<'a, Result<u32, ModelError>> {
            Box::pin(async move { Err(ModelError::Timeout) })
        }
        fn provider(&self) -> ProviderInfo {
            ProviderInfo {
                name: "f".into(),
                model_id: "f".into(),
                context_window: 1,
            }
        }
    }

    #[tokio::test]
    async fn token_count_failure_returns_token_count_failed() {
        let mgr = StandardContextManager::new(
            Arc::new(FailingModel),
            Arc::new(NullCacheProvider),
            CompactionConfig::default(),
        );
        let st = state();
        let err = mgr
            .assemble(&st, &sources("BLOCK1", 0xAB, vec![]))
            .await
            .unwrap_err();
        assert!(matches!(err, ContextError::TokenCountFailed));
    }

    // ── Cache-stability invariant: same inputs ⇒ identical rendered Block
    //   1+2 prefix bytes (this is the contract the prefix cache relies on).

    #[tokio::test]
    async fn deterministic_prefix_across_calls() {
        let mgr = mk();
        let st = state();
        let srcs = sources("BLOCK1-content", 0x11, vec![]);
        let a = mgr.assemble(&st, &srcs).await.unwrap();
        let b = mgr.assemble(&st, &srcs).await.unwrap();
        assert_eq!(a.system_prompt.content, b.system_prompt.content);
        assert_eq!(
            a.system_prompt.static_block_hash,
            b.system_prompt.static_block_hash
        );
        assert_eq!(
            a.system_prompt.session_block_hash,
            b.system_prompt.session_block_hash
        );
    }

    // ── Issue #29: CompactionVerifier / KeyTermVerifier ──────────────────

    fn state_with_task(task: &str) -> SessionState {
        SessionState::new(SessionId::new("s1"), TaskId::new("t1"), task)
    }

    fn all_hints_on() -> CompactionPreserveHints {
        CompactionPreserveHints::default()
    }

    fn only_task_state() -> CompactionPreserveHints {
        CompactionPreserveHints {
            keep_architectural_decisions: false,
            keep_open_problems: false,
            keep_current_task_state: true,
            keep_recent_file_list: false,
            keep_thinking_blocks: false,
        }
    }

    #[test]
    fn config_default_max_compaction_attempts_is_two() {
        assert_eq!(CompactionConfig::default().max_compaction_attempts, 2);
    }

    #[test]
    fn verifier_all_terms_present_passes() {
        let v = KeyTermVerifier;
        let st = state_with_task("Refactor the parser module");
        let r = v.verify(
            "we will refactor the parser module",
            &only_task_state(),
            &st,
        );
        assert!(r.passed);
        assert!(r.missing_items.is_empty());
        assert_eq!(r.detail, "all 3 key term(s) present");
    }

    #[test]
    fn verifier_missing_term_fails_and_lists_it() {
        let v = KeyTermVerifier;
        let st = state_with_task("Refactor the parser module");
        let r = v.verify("we will refactor the parser", &only_task_state(), &st);
        assert!(!r.passed);
        assert_eq!(r.missing_items, vec!["module".to_string()]);
        assert_eq!(r.detail, "missing 1 of 3 key term(s): module");
    }

    #[test]
    fn verifier_no_task_state_means_zero_terms_passes() {
        let v = KeyTermVerifier;
        let st = state_with_task("Refactor the parser module");
        let hints = CompactionPreserveHints {
            keep_current_task_state: false,
            ..CompactionPreserveHints::default()
        };
        let r = v.verify("anything at all", &hints, &st);
        assert!(r.passed);
        assert!(r.missing_items.is_empty());
        assert_eq!(r.detail, "all 0 key term(s) present");
    }

    #[test]
    fn verifier_ignores_tokens_under_length_four() {
        // "do","the","api" (len 3) are dropped; only "thing" survives.
        let v = KeyTermVerifier;
        let st = state_with_task("do the api thing");
        // summary omits "thing" but contains the short tokens.
        let r = v.verify("do the api work", &only_task_state(), &st);
        assert_eq!(r.missing_items, vec!["thing".to_string()]);
        assert!(!r.passed);
    }

    #[test]
    fn verifier_is_case_insensitive() {
        let v = KeyTermVerifier;
        let st = state_with_task("Refactor PARSER");
        let r = v.verify("REFACTOR the parser", &only_task_state(), &st);
        assert!(r.passed);
    }

    #[test]
    fn verifier_dedupes_repeated_terms() {
        let v = KeyTermVerifier;
        let st = state_with_task("Deploy deploy deploy service");
        // "deploy" appears once in the term list; reported once when missing.
        let r = v.verify("nothing relevant here", &only_task_state(), &st);
        assert_eq!(
            r.missing_items,
            vec!["deploy".to_string(), "service".to_string()]
        );
        assert_eq!(r.detail, "missing 2 of 2 key term(s): deploy, service");
    }

    #[test]
    fn verifier_empty_fields_contribute_nothing_even_when_all_hints_on() {
        let v = KeyTermVerifier;
        // All hints on but every structured field empty ⇒ zero terms.
        let mut st = state_with_task("");
        st.task_instruction = String::new();
        let hints = CompactionPreserveHints {
            keep_architectural_decisions: true,
            keep_open_problems: true,
            keep_current_task_state: true,
            keep_recent_file_list: true,
            keep_thinking_blocks: true,
        };
        let r = v.verify("empty summary", &hints, &st);
        assert!(r.passed);
        assert!(r.missing_items.is_empty());
        // Sanity: with all hints on, a non-empty task still drives terms.
        let st2 = state_with_task("Build widget");
        let r2 = v.verify("nothing", &all_hints_on(), &st2);
        assert_eq!(
            r2.missing_items,
            vec!["build".to_string(), "widget".to_string()]
        );
    }

    // ── Issue #47: structured fields feed the four additional hints ──────

    fn only_open_problems() -> CompactionPreserveHints {
        CompactionPreserveHints {
            keep_architectural_decisions: false,
            keep_open_problems: true,
            keep_current_task_state: false,
            keep_recent_file_list: false,
            keep_thinking_blocks: false,
        }
    }

    fn only_architectural_decisions() -> CompactionPreserveHints {
        CompactionPreserveHints {
            keep_architectural_decisions: true,
            keep_open_problems: false,
            keep_current_task_state: false,
            keep_recent_file_list: false,
            keep_thinking_blocks: false,
        }
    }

    fn only_recent_files() -> CompactionPreserveHints {
        CompactionPreserveHints {
            keep_architectural_decisions: false,
            keep_open_problems: false,
            keep_current_task_state: false,
            keep_recent_file_list: true,
            keep_thinking_blocks: false,
        }
    }

    fn only_thinking_blocks() -> CompactionPreserveHints {
        CompactionPreserveHints {
            keep_architectural_decisions: false,
            keep_open_problems: false,
            keep_current_task_state: false,
            keep_recent_file_list: false,
            keep_thinking_blocks: true,
        }
    }

    #[test]
    fn verifier_open_problems_isolated() {
        let v = KeyTermVerifier;
        let mut st = state_with_task("ignored task");
        st.open_problems = vec!["Resolve the deadlock issue".into()];
        let r = v.verify("we noted the deadlock", &only_open_problems(), &st);
        assert_eq!(
            r.missing_items,
            vec!["resolve".to_string(), "issue".to_string()]
        );
        assert!(!r.passed);
    }

    #[test]
    fn verifier_architectural_decisions_isolated() {
        let v = KeyTermVerifier;
        let mut st = state_with_task("ignored task");
        st.architectural_decisions = vec!["Adopt hexagonal architecture".into()];
        let r = v.verify(
            "we will adopt hexagonal architecture",
            &only_architectural_decisions(),
            &st,
        );
        assert!(r.passed);
        assert!(r.missing_items.is_empty());
    }

    #[test]
    fn verifier_recent_files_path_tokenization() {
        let v = KeyTermVerifier;
        let mut st = state_with_task("ignored task");
        st.recent_files = vec!["src/parser/mod.rs".into()];
        // src, mod, rs are <4 chars and dropped; only `parser` survives.
        let r = v.verify("touched the lexer", &only_recent_files(), &st);
        assert_eq!(r.missing_items, vec!["parser".to_string()]);
        assert!(!r.passed);
    }

    #[test]
    fn verifier_reasoning_summary_isolated() {
        let v = KeyTermVerifier;
        let mut st = state_with_task("ignored task");
        st.reasoning_summary = "Considered caching strategy".into();
        let r = v.verify("nothing relevant", &only_thinking_blocks(), &st);
        assert_eq!(
            r.missing_items,
            vec![
                "considered".to_string(),
                "caching".to_string(),
                "strategy".to_string()
            ]
        );
        assert!(!r.passed);
    }

    #[test]
    fn verifier_multi_hint_dedup_ordering() {
        let v = KeyTermVerifier;
        // "parser" reachable via both task_instruction and open_problems;
        // first-occurrence is the task position (pushed first).
        let mut st = state_with_task("Refactor parser");
        st.open_problems = vec!["parser bug remains".into()];
        let hints = CompactionPreserveHints {
            keep_architectural_decisions: false,
            keep_open_problems: true,
            keep_current_task_state: true,
            keep_recent_file_list: false,
            keep_thinking_blocks: false,
        };
        let r = v.verify("nothing matched", &hints, &st);
        // refactor, parser (task), then remains (open_problems). "bug" <4 dropped.
        // parser appears once at its first (task) position.
        assert_eq!(
            r.missing_items,
            vec![
                "refactor".to_string(),
                "parser".to_string(),
                "remains".to_string()
            ]
        );
    }

    #[test]
    fn verifier_empty_list_with_hint_on_passes() {
        let v = KeyTermVerifier;
        let st = state_with_task("ignored task");
        // open_problems empty but its hint on ⇒ contributes nothing ⇒ passes.
        let r = v.verify("anything", &only_open_problems(), &st);
        assert!(r.passed);
        assert!(r.missing_items.is_empty());
    }

    // ── Fixture replay: cross-language consistency (issue #29) ───────────

    #[test]
    fn key_term_verifier_fixture_replay() {
        #[derive(Deserialize)]
        struct Expected {
            passed: bool,
            missing_items: Vec<String>,
        }
        #[derive(Deserialize)]
        struct Case {
            name: String,
            summary: String,
            hints: CompactionPreserveHints,
            task_instruction: String,
            #[serde(default)]
            open_problems: Vec<String>,
            #[serde(default)]
            architectural_decisions: Vec<String>,
            #[serde(default)]
            recent_files: Vec<String>,
            #[serde(default)]
            reasoning_summary: String,
            expected: Expected,
        }
        #[derive(Deserialize)]
        struct Suite {
            cases: Vec<Case>,
        }

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/compaction_verifier/cases.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let suite: Suite = serde_json::from_str(&raw).unwrap();
        assert!(suite.cases.len() >= 5, "expected at least 5 fixture cases");

        let verifier = KeyTermVerifier;
        for case in &suite.cases {
            let mut st = state_with_task(&case.task_instruction);
            st.open_problems = case.open_problems.clone();
            st.architectural_decisions = case.architectural_decisions.clone();
            st.recent_files = case.recent_files.clone();
            st.reasoning_summary = case.reasoning_summary.clone();
            let result = verifier.verify(&case.summary, &case.hints, &st);
            assert_eq!(
                result.passed, case.expected.passed,
                "case `{}`: passed mismatch",
                case.name
            );
            assert_eq!(
                result.missing_items, case.expected.missing_items,
                "case `{}`: missing_items mismatch",
                case.name
            );
        }
    }

    // ── Issue #141: configurable compaction window ───────────────────────

    /// Model stub whose `provider().context_window` is configurable, for
    /// exercising the resolver fallback chain.
    struct WindowModel {
        context_window: u32,
    }
    impl ModelInterface for WindowModel {
        fn call<'a>(
            &'a self,
            _req: ModelRequest,
        ) -> BoxFut<'a, Result<ModelResponse, ModelError>> {
            Box::pin(async move { unimplemented!() })
        }
        fn call_streaming<'a>(
            &'a self,
            _req: ModelRequest,
        ) -> BoxFut<'a, Result<ModelStream, ModelError>> {
            Box::pin(async move { unimplemented!() })
        }
        fn count_tokens<'a>(
            &'a self,
            _req: &'a ModelRequest,
        ) -> BoxFut<'a, Result<u32, ModelError>> {
            Box::pin(async move { Ok(0) })
        }
        fn provider(&self) -> ProviderInfo {
            ProviderInfo {
                name: "window".into(),
                model_id: "window".into(),
                context_window: self.context_window,
            }
        }
    }

    fn mgr_with(
        config_context_length: Option<u32>,
        model_context_window: u32,
    ) -> StandardContextManager<WindowModel> {
        let compaction = CompactionConfig {
            context_length: config_context_length,
            ..CompactionConfig::default()
        };
        StandardContextManager::new(
            Arc::new(WindowModel {
                context_window: model_context_window,
            }),
            Arc::new(NullCacheProvider),
            compaction,
        )
    }

    #[test]
    fn resolver_config_wins_over_model() {
        // config Some(8000) + model 128000 ⇒ 8000 (config wins).
        assert_eq!(mgr_with(Some(8000), 128_000).resolve_context_length(), 8000);
    }

    #[test]
    fn resolver_model_fallback_when_config_none() {
        // config None + model 128000 ⇒ 128000 (model fallback).
        assert_eq!(mgr_with(None, 128_000).resolve_context_length(), 128_000);
    }

    #[test]
    fn resolver_default_when_config_none_and_no_model() {
        // config None + model 0 ⇒ 8000 (default).
        assert_eq!(
            mgr_with(None, 0).resolve_context_length(),
            DEFAULT_CONTEXT_LENGTH
        );
    }

    #[test]
    fn resolver_explicit_zero_falls_through_to_model() {
        // config Some(0) + model 128000 ⇒ 128000 (explicit zero falls through).
        assert_eq!(mgr_with(Some(0), 128_000).resolve_context_length(), 128_000);
    }

    #[test]
    fn resolver_explicit_zero_and_no_model_uses_default() {
        // config Some(0) + model 0 ⇒ 8000.
        assert_eq!(
            mgr_with(Some(0), 0).resolve_context_length(),
            DEFAULT_CONTEXT_LENGTH
        );
    }

    #[test]
    fn resolver_does_not_clamp_config_above_model() {
        // config Some(500000) + model 128000 ⇒ 500000 (no clamp).
        assert_eq!(
            mgr_with(Some(500_000), 128_000).resolve_context_length(),
            500_000
        );
    }

    #[test]
    fn trigger_math_respects_small_window() {
        // window 8000 + threshold 0.80 ⇒ trips at 6400, not at 6399.
        let mgr = mgr_with(Some(8000), 128_000);
        let mut st = state_with_task("t");
        st.window_limit = 8000;
        st.token_budget_used = 6400;
        assert!(mgr.should_compact(&st));
        st.token_budget_used = 6399;
        assert!(!mgr.should_compact(&st));
    }

    #[test]
    fn trigger_math_zero_window_never_compacts() {
        // Existing zero-window guard still holds.
        let mgr = mgr_with(None, 128_000);
        let mut st = state_with_task("t");
        st.window_limit = 0;
        st.token_budget_used = 9999;
        assert!(!mgr.should_compact(&st));
    }

    #[test]
    fn config_default_context_length_is_none() {
        assert_eq!(CompactionConfig::default().context_length, None);
    }

    #[test]
    fn config_default_serializes_without_context_length_key() {
        let json = serde_json::to_string(&CompactionConfig::default()).unwrap();
        assert!(
            !json.contains("context_length"),
            "context_length must be absent when None: {json}"
        );
        // Round-trips cleanly with the key absent.
        let back: CompactionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.context_length, None);
    }

    #[test]
    fn config_some_serializes_with_context_length_key() {
        let cfg = CompactionConfig {
            context_length: Some(8192),
            ..CompactionConfig::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("\"context_length\":8192"), "{json}");
    }

    #[test]
    fn seed_session_sets_window_limit_to_resolved_length() {
        let mgr = mgr_with(Some(8192), 128_000);
        let st = mgr.seed_session(SessionId::new("s1"), TaskId::new("t1"), "do the thing");
        assert_eq!(st.window_limit, mgr.resolve_context_length());
        assert_eq!(st.window_limit, 8192);
        // And the resolver tail still applies when neither source is set.
        let mgr2 = mgr_with(None, 0);
        let st2 = mgr2.seed_session(SessionId::new("s2"), TaskId::new("t2"), "x");
        assert_eq!(st2.window_limit, DEFAULT_CONTEXT_LENGTH);
    }

    #[test]
    fn session_state_new_defaults_to_conservative_window() {
        // Issue #141: the unknown-window default is now 8K, not 200K.
        let st = SessionState::new(SessionId::new("s"), TaskId::new("t"), "x");
        assert_eq!(st.window_limit, DEFAULT_CONTEXT_LENGTH);
        assert_eq!(st.window_limit, 8_000);
    }

    // ── Fixture replay: cross-language consistency (issue #141) ──────────

    #[test]
    fn compaction_window_fixture_replay() {
        #[derive(Deserialize)]
        struct TriggerCase {
            name: String,
            window_limit: u32,
            token_budget_used: u32,
            threshold: f32,
            expected_should_compact: bool,
        }
        #[derive(Deserialize)]
        struct ResolverCase {
            name: String,
            config_context_length: Option<u32>,
            model_context_window: u32,
            expected_resolved: u32,
        }
        #[derive(Deserialize)]
        struct Suite {
            trigger_cases: Vec<TriggerCase>,
            resolver_cases: Vec<ResolverCase>,
        }

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/compaction_window/cases.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let suite: Suite = serde_json::from_str(&raw).unwrap();
        assert!(
            suite.trigger_cases.len() >= 5,
            "expected at least 5 trigger cases"
        );
        assert!(
            suite.resolver_cases.len() >= 6,
            "expected at least 6 resolver cases"
        );

        // ── trigger_cases: threshold × window_limit drives should_compact ──
        for case in &suite.trigger_cases {
            let compaction = CompactionConfig {
                threshold: case.threshold,
                ..CompactionConfig::default()
            };
            let mgr = StandardContextManager::new(
                Arc::new(WindowModel { context_window: 0 }),
                Arc::new(NullCacheProvider),
                compaction,
            );
            let mut st = state_with_task("t");
            st.window_limit = case.window_limit;
            st.token_budget_used = case.token_budget_used;
            assert_eq!(
                mgr.should_compact(&st),
                case.expected_should_compact,
                "trigger case `{}`: should_compact mismatch",
                case.name
            );
        }

        // ── resolver_cases: config → model → default fallback chain ────────
        for case in &suite.resolver_cases {
            let mgr = mgr_with(case.config_context_length, case.model_context_window);
            assert_eq!(
                mgr.resolve_context_length(),
                case.expected_resolved,
                "resolver case `{}`: resolved mismatch",
                case.name
            );
        }
    }
}
