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

/// Stable identifier for a guide or skill (issue #9 — GuideRegistry).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GuideId(pub String);

impl GuideId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// Forward-declared `Guide` (issue #9). Carries the rendered chunk and an
/// identifier; full lifecycle metadata lives with `GuideRegistry`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Guide {
    pub id: GuideId,
    /// Rendered final form. The spec forbids reformatting at assembly time.
    pub content: String,
}

// `MemoryItem` is defined in [`crate::memory`] (issue #8). Re-exported here
// for downstream callers building `ContextSources`.
pub use crate::memory::MemoryItem;

/// Forward-declared `ComposedPrompt` (issue #14 — PromptChunkRegistry).
///
/// Block 1 is computed ONCE at harness startup. `rendered` is the final
/// byte-for-byte content; `block_1_hash` is a stable digest used by the
/// `ContextManager` to detect unexpected cache invalidation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposedPrompt {
    pub rendered: String,
    pub block_1_hash: u64,
}

/// Forward-declared cache stats parsed by a `CacheProvider` (issue spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub static_hit: Option<bool>,
    pub session_hit: Option<bool>,
    pub history_hit: Option<bool>,
}

impl CacheStats {
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

/// Forward-declared `CacheProvider` trait (issue #7 dependency). The default
/// `NullCacheProvider` is the testing default — it never interferes.
pub trait CacheProvider: Send + Sync {
    fn supports_caching(&self) -> bool {
        false
    }
    /// Annotate the assembled context with provider-specific cache markers.
    /// No-op when `supports_caching()` is false.
    fn annotate(&self, _context: &mut Context) {}
}

/// Testing default — no-op for all calls. Never interferes with unit tests.
pub struct NullCacheProvider;
impl CacheProvider for NullCacheProvider {}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionConfig {
    pub threshold: f32,
    pub preserve_recent_n: u32,
    pub head_tail_tokens: u32,
    pub offload_path: PathBuf,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            threshold: 0.80,
            preserve_recent_n: 8,
            head_tail_tokens: 512,
            offload_path: PathBuf::from(".spore/offload"),
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
            window_limit: 200_000,
            guides_loaded: Vec::new(),
            pending_skill_injections: Vec::new(),
            budget_warning_active: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    #[error("cache hash mismatch on block {block}: expected {expected}, got {actual}")]
    CacheHashMismatch {
        block: String,
        expected: u64,
        actual: u64,
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

    fn record_cache_result(&self, context: &mut Context, cache_stats: CacheStats);
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
                        block: "static".into(),
                        expected: prev,
                        actual: static_hash,
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
                    // Spec: warn — but do not fail. Update the memo so we
                    // do not warn every turn for the rest of the session.
                    eprintln!(
                        "warn: session block hash changed mid-session ({} → {})",
                        prev, session_hash
                    );
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
        let (rendered, breakpoints) = render_segments(&sources.composed_prompt.rendered, &segments);
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
                ToolOutput::WaitingForHuman { .. } => "[waiting]".into(),
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

    fn record_cache_result(&self, context: &mut Context, cache_stats: CacheStats) {
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
        async fn call(&self, _req: ModelRequest) -> Result<ModelResponse, ModelError> {
            Ok(ModelResponse {
                content: vec![],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            })
        }
        async fn call_streaming(&self, _req: ModelRequest) -> Result<ModelStream, ModelError> {
            unimplemented!()
        }
        async fn count_tokens(&self, _req: &ModelRequest) -> Result<u32, ModelError> {
            Ok(self.per_call.load(Ordering::SeqCst))
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
        fn annotate(&self, _context: &mut Context) {
            self.calls.fetch_add(1, Ordering::SeqCst);
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
                rendered: rendered.into(),
                block_1_hash: hash,
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
            ContextError::CacheHashMismatch { block, .. } => assert_eq!(block, "static"),
            e => panic!("wrong error: {e:?}"),
        }
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
        mgr.inject_skill(
            &mut ctx,
            &Guide {
                id: GuideId::new("rust-style"),
                content: "prefer iterators".into(),
            },
        )
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
            CacheStats::new(Some(true), Some(false), Some(true)),
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
        st.pending_skill_injections.push(Guide {
            id: GuideId::new("g1"),
            content: "do x".into(),
        });
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
        async fn call(&self, _r: ModelRequest) -> Result<ModelResponse, ModelError> {
            unimplemented!()
        }
        async fn call_streaming(&self, _r: ModelRequest) -> Result<ModelStream, ModelError> {
            unimplemented!()
        }
        async fn count_tokens(&self, _r: &ModelRequest) -> Result<u32, ModelError> {
            Err(ModelError::Timeout)
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
}
