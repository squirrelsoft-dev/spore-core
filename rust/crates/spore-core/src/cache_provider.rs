//! Issue #25 — `CacheProvider`: provider-specific cache annotation and stats.
//!
//! Cache control is provider-specific at the API level: Anthropic uses
//! explicit `cache_control` markers, OpenAI caches automatically above a
//! token threshold, Ollama has no caching. `CacheProvider` is the
//! abstraction that keeps these concerns out of provider-agnostic
//! `ContextManager`.
//!
//! Flow (see `docs/harness-engineering-concepts.md` §"Cache Architecture"):
//!
//! ```text
//! ContextManager.assemble():
//!   ... build and render segments ...
//!   if cache_provider.supports_caching():
//!     cache_provider.annotate(&mut context)
//!   return context
//!
//! // After each model response:
//! stats = cache_provider.parse_cache_stats(&response)
//! observability.emit_cache_stats(session_id, stats)
//! ```

use serde::{Deserialize, Serialize};

use crate::context::{BreakpointInfo, Context};
use crate::model::ModelResponse;

// ============================================================================
// Spec-defined types
// ============================================================================

/// Result of annotating a context with provider-specific cache markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CacheAnnotationResult {
    pub markers_inserted: u32,
    pub estimated_cacheable_tokens: u32,
}

/// Cache token usage parsed from a single model response.
///
/// `None` from `parse_cache_stats` means the response had no cache metadata
/// at all (caching wasn't attempted). A `CacheStats` with all-zero fields
/// means caching was attempted and missed — the distinction matters for
/// observability.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct CacheStats {
    pub cache_read_tokens: u32,
    pub cache_write_tokens: u32,
    pub cache_read_cost_usd: f64,
    pub cache_write_cost_usd: f64,
}

// ============================================================================
// Trait
// ============================================================================

pub trait CacheProvider: Send + Sync {
    /// Whether this provider supports prefix caching.
    fn supports_caching(&self) -> bool {
        false
    }

    /// Annotate a fully assembled context with provider-specific cache
    /// markers. No-op when `supports_caching()` is false.
    fn annotate(&self, _context: &mut Context) -> CacheAnnotationResult {
        CacheAnnotationResult::default()
    }

    /// Parse cache usage from a model response. Returns `None` when the
    /// response has no cache metadata at all.
    fn parse_cache_stats(&self, _response: &ModelResponse) -> Option<CacheStats> {
        None
    }

    /// Provider identity — used for observability and auto-detection.
    fn provider_name(&self) -> &'static str;
}

// ============================================================================
// Standard implementations
// ============================================================================

/// Testing default. All operations are no-ops; `supports_caching()` is false.
///
/// Always use `NullCacheProvider` in unit tests so cache logic never
/// interferes with assertions.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullCacheProvider;

impl CacheProvider for NullCacheProvider {
    fn provider_name(&self) -> &'static str {
        "null"
    }
}

/// Anthropic prefix caching.
///
/// Inserts logical `cache_control: ephemeral` breakpoints after each stable
/// block boundary (Block 1: Static, Block 2: PerSession, plus history and
/// optional tool-schema anchors). Reads `cache_read_tokens` and
/// `cache_write_tokens` from response usage.
#[derive(Debug, Clone, Copy)]
pub struct AnthropicCacheProvider {
    /// Anthropic supports up to 4 breakpoints per request.
    pub max_cache_anchors: u32,
    /// USD per 1M tokens for cache reads. Default matches Sonnet 4.x
    /// published pricing (0.30 USD / 1M cache-read tokens).
    pub cache_read_usd_per_million: f64,
    /// USD per 1M tokens for cache writes (5-minute TTL). Default matches
    /// Sonnet 4.x (3.75 USD / 1M cache-write tokens).
    pub cache_write_usd_per_million: f64,
}

impl Default for AnthropicCacheProvider {
    fn default() -> Self {
        Self {
            max_cache_anchors: 4,
            // Sonnet 4.x default; callers override per model with
            // `with_model_pricing`.
            cache_read_usd_per_million: 0.30,
            cache_write_usd_per_million: 3.75,
        }
    }
}

impl AnthropicCacheProvider {
    /// Override the cache pricing for a specific model. Pricing data lives in
    /// the implementation so callers don't have to import a table — pass the
    /// model id and we look it up. Unknown ids return Sonnet pricing.
    ///
    /// Anthropic cache pricing (USD per 1M tokens, 5-minute TTL):
    /// - opus-4.x:   1.50 read / 18.75 write
    /// - sonnet-4.x: 0.30 read /  3.75 write
    /// - haiku-4.x:  0.08 read /  1.00 write
    pub fn with_model_pricing(mut self, model_id: &str) -> Self {
        let (read, write) = anthropic_cache_pricing(model_id);
        self.cache_read_usd_per_million = read;
        self.cache_write_usd_per_million = write;
        self
    }
}

fn anthropic_cache_pricing(model_id: &str) -> (f64, f64) {
    match model_id {
        id if id.contains("opus") => (1.50, 18.75),
        id if id.contains("haiku") => (0.08, 1.00),
        // sonnet, unknown, default → sonnet pricing
        _ => (0.30, 3.75),
    }
}

impl CacheProvider for AnthropicCacheProvider {
    fn supports_caching(&self) -> bool {
        true
    }

    fn annotate(&self, context: &mut Context) -> CacheAnnotationResult {
        // Anchors are derived from rendered system-prompt breakpoints
        // (Block-1 / Block-2 boundaries) plus an optional history anchor
        // if there are any prior messages. Cap at `max_cache_anchors`.
        let existing: Vec<BreakpointInfo> = context.system_prompt.breakpoints.clone();
        let mut anchors = existing.len() as u32;

        // History anchor: if there are messages, the last message in the
        // history is a logical cache anchor. We do not mutate `messages`
        // (Content is a typed enum), but we record the anchor as another
        // BreakpointInfo with a synthetic name to signal annotation.
        let history_anchor_eligible = !context.messages.is_empty();
        if history_anchor_eligible && anchors < self.max_cache_anchors {
            context.system_prompt.breakpoints.push(BreakpointInfo {
                after_segment: "__history_tail__".into(),
                token_offset: context.token_count,
            });
            anchors += 1;
        }

        let markers = anchors.min(self.max_cache_anchors);
        // Cacheable tokens approximation: everything up to the last anchor
        // is eligible — use the assembled `token_count` as an upper bound.
        let estimated = if markers == 0 { 0 } else { context.token_count };
        CacheAnnotationResult {
            markers_inserted: markers,
            estimated_cacheable_tokens: estimated,
        }
    }

    fn parse_cache_stats(&self, response: &ModelResponse) -> Option<CacheStats> {
        let read = response.usage.cache_read_tokens;
        let write = response.usage.cache_write_tokens;
        if read.is_none() && write.is_none() {
            return None;
        }
        let read_tokens = read.unwrap_or(0);
        let write_tokens = write.unwrap_or(0);
        Some(CacheStats {
            cache_read_tokens: read_tokens,
            cache_write_tokens: write_tokens,
            cache_read_cost_usd: f64::from(read_tokens) / 1_000_000.0
                * self.cache_read_usd_per_million,
            cache_write_cost_usd: f64::from(write_tokens) / 1_000_000.0
                * self.cache_write_usd_per_million,
        })
    }

    fn provider_name(&self) -> &'static str {
        "anthropic"
    }
}

/// OpenAI prefix caching.
///
/// OpenAI caches automatically on prompts above `min_cacheable_tokens` (1024
/// by default) — no explicit markers required. `annotate()` is a no-op
/// (returning zeros). `parse_cache_stats()` reads `cache_read_tokens` from
/// the response. OpenAI does not return a "cache write" count, so writes
/// remain zero.
#[derive(Debug, Clone, Copy)]
pub struct OpenAICacheProvider {
    /// Below this token count OpenAI will not cache.
    pub min_cacheable_tokens: u32,
}

impl Default for OpenAICacheProvider {
    fn default() -> Self {
        Self {
            min_cacheable_tokens: 1024,
        }
    }
}

impl CacheProvider for OpenAICacheProvider {
    fn supports_caching(&self) -> bool {
        true
    }

    fn annotate(&self, context: &mut Context) -> CacheAnnotationResult {
        // No markers needed — OpenAI caches automatically.
        let cacheable = if context.token_count >= self.min_cacheable_tokens {
            context.token_count
        } else {
            0
        };
        CacheAnnotationResult {
            markers_inserted: 0,
            estimated_cacheable_tokens: cacheable,
        }
    }

    fn parse_cache_stats(&self, response: &ModelResponse) -> Option<CacheStats> {
        let read = response.usage.cache_read_tokens?;
        Some(CacheStats {
            cache_read_tokens: read,
            cache_write_tokens: 0,
            cache_read_cost_usd: 0.0,
            cache_write_cost_usd: 0.0,
        })
    }

    fn provider_name(&self) -> &'static str {
        "openai"
    }
}

/// Ollama has no prefix caching. Every method is a no-op.
#[derive(Debug, Default, Clone, Copy)]
pub struct OllamaCacheProvider;

impl CacheProvider for OllamaCacheProvider {
    fn provider_name(&self) -> &'static str {
        "ollama"
    }
}

// ============================================================================
// Auto-detection from model provider name
// ============================================================================

/// Map a `ModelInterface::provider().name` to the appropriate
/// `CacheProvider`. Returns `None` when the provider is unknown — the caller
/// (typically `HarnessBuilder`) should emit a `CacheProviderNotDetected`
/// warning and fall back to `NullCacheProvider`.
pub fn auto_detect(provider_name: &str) -> Option<Box<dyn CacheProvider>> {
    match provider_name.to_ascii_lowercase().as_str() {
        "anthropic" => Some(Box::new(AnthropicCacheProvider::default())),
        "openai" => Some(Box::new(OpenAICacheProvider::default())),
        "ollama" => Some(Box::new(OllamaCacheProvider)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{CacheBlockStatus, ContextMeta, RenderedSystemPrompt};
    use crate::harness::SessionId;
    use crate::model::{ContentBlock, StopReason, TokenUsage};
    use crate::tool_registry::TaskPhase;

    fn ctx(tokens: u32, breakpoints: Vec<BreakpointInfo>, msgs: u32) -> Context {
        Context {
            system_prompt: RenderedSystemPrompt {
                content: "system".into(),
                breakpoints,
                static_block_hash: 0,
                session_block_hash: 0,
            },
            messages: (0..msgs)
                .map(|_| crate::model::Message {
                    role: crate::model::Role::User,
                    content: crate::model::Content::Text { text: "m".into() },
                })
                .collect(),
            tool_schemas: vec![],
            token_count: tokens,
            window_limit: 200_000,
            utilization: 0.0,
            meta: ContextMeta {
                session_id: SessionId::new("s"),
                turn_number: 0,
                active_phase: TaskPhase::Execution,
                guides_loaded: vec![],
                skills_injected: vec![],
                compacted: false,
                cache_blocks: CacheBlockStatus::default(),
            },
        }
    }

    fn response(read: Option<u32>, write: Option<u32>) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::Text { text: "hi".into() }],
            usage: TokenUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_read_tokens: read,
                cache_write_tokens: write,
            },
            stop_reason: StopReason::EndTurn,
        }
    }

    // ── Rule: Null provider is no-op ─────────────────────────────────────

    #[test]
    fn null_provider_does_nothing() {
        let p = NullCacheProvider;
        assert!(!p.supports_caching());
        assert_eq!(p.provider_name(), "null");
        let mut c = ctx(100, vec![], 0);
        let r = p.annotate(&mut c);
        assert_eq!(r, CacheAnnotationResult::default());
        assert!(p.parse_cache_stats(&response(Some(5), None)).is_none());
    }

    // ── Rule: Anthropic supports caching and reports its name ────────────

    #[test]
    fn anthropic_identity() {
        let p = AnthropicCacheProvider::default();
        assert!(p.supports_caching());
        assert_eq!(p.provider_name(), "anthropic");
        assert_eq!(p.max_cache_anchors, 4);
    }

    // ── Rule: Anthropic.annotate inserts markers up to max_cache_anchors ─

    #[test]
    fn anthropic_annotate_caps_at_max_anchors() {
        let p = AnthropicCacheProvider {
            max_cache_anchors: 2,
            ..Default::default()
        };
        let bps = vec![
            BreakpointInfo {
                after_segment: "block_1_static".into(),
                token_offset: 10,
            },
            BreakpointInfo {
                after_segment: "block_2_per_session".into(),
                token_offset: 20,
            },
        ];
        let mut c = ctx(50, bps, 3);
        let r = p.annotate(&mut c);
        // Two existing breakpoints already at the cap → no history anchor.
        assert_eq!(r.markers_inserted, 2);
        assert!(r.estimated_cacheable_tokens > 0);
        assert_eq!(c.system_prompt.breakpoints.len(), 2);
    }

    // ── Rule: Anthropic.annotate adds history anchor when room and history
    //   present ────────────────────────────────────────────────────────────

    #[test]
    fn anthropic_annotate_adds_history_anchor() {
        let p = AnthropicCacheProvider::default();
        let bps = vec![BreakpointInfo {
            after_segment: "block_1_static".into(),
            token_offset: 10,
        }];
        let mut c = ctx(75, bps, 4);
        let r = p.annotate(&mut c);
        assert_eq!(r.markers_inserted, 2);
        assert_eq!(c.system_prompt.breakpoints.len(), 2);
        assert_eq!(
            c.system_prompt.breakpoints[1].after_segment,
            "__history_tail__"
        );
    }

    // ── Rule: Anthropic.annotate returns 0 markers when no history and no
    //   existing breakpoints ─────────────────────────────────────────────

    #[test]
    fn anthropic_annotate_zero_when_empty() {
        let p = AnthropicCacheProvider::default();
        let mut c = ctx(50, vec![], 0);
        let r = p.annotate(&mut c);
        assert_eq!(r.markers_inserted, 0);
        assert_eq!(r.estimated_cacheable_tokens, 0);
    }

    // ── Rule: Anthropic.parse_cache_stats returns None without metadata ──

    #[test]
    fn anthropic_parse_returns_none_without_metadata() {
        let p = AnthropicCacheProvider::default();
        assert!(p.parse_cache_stats(&response(None, None)).is_none());
    }

    // ── Rule: Anthropic.parse_cache_stats reads read/write tokens ────────

    #[test]
    fn anthropic_parse_reads_tokens() {
        let p = AnthropicCacheProvider::default();
        let s = p
            .parse_cache_stats(&response(Some(900), Some(120)))
            .unwrap();
        assert_eq!(s.cache_read_tokens, 900);
        assert_eq!(s.cache_write_tokens, 120);
    }

    // ── Rule: Anthropic.parse_cache_stats treats one-sided metadata as Some
    //   (attempted but only one direction present) ──────────────────────

    #[test]
    fn anthropic_parse_one_sided_is_some() {
        let p = AnthropicCacheProvider::default();
        let s = p.parse_cache_stats(&response(Some(0), None)).unwrap();
        assert_eq!(s.cache_read_tokens, 0);
        assert_eq!(s.cache_write_tokens, 0);
    }

    // ── Rule: Anthropic.parse_cache_stats computes USD cost from per-model
    //   pricing (#39) ───────────────────────────────────────────────────────

    #[test]
    fn anthropic_parse_computes_cost_default_sonnet() {
        let p = AnthropicCacheProvider::default();
        let s = p
            .parse_cache_stats(&response(Some(1_000_000), Some(1_000_000)))
            .unwrap();
        // Sonnet pricing: 0.30 read / 3.75 write per 1M.
        assert!((s.cache_read_cost_usd - 0.30).abs() < 1e-9, "{s:?}");
        assert!((s.cache_write_cost_usd - 3.75).abs() < 1e-9, "{s:?}");
    }

    #[test]
    fn anthropic_parse_with_opus_pricing() {
        let p = AnthropicCacheProvider::default().with_model_pricing("claude-opus-4-7");
        let s = p
            .parse_cache_stats(&response(Some(1_000_000), Some(1_000_000)))
            .unwrap();
        // Opus pricing: 1.50 read / 18.75 write per 1M.
        assert!((s.cache_read_cost_usd - 1.50).abs() < 1e-9, "{s:?}");
        assert!((s.cache_write_cost_usd - 18.75).abs() < 1e-9, "{s:?}");
    }

    #[test]
    fn anthropic_parse_with_haiku_pricing() {
        let p = AnthropicCacheProvider::default().with_model_pricing("claude-haiku-4-5");
        let s = p
            .parse_cache_stats(&response(Some(1_000_000), Some(1_000_000)))
            .unwrap();
        // Haiku pricing: 0.08 read / 1.00 write per 1M.
        assert!((s.cache_read_cost_usd - 0.08).abs() < 1e-9, "{s:?}");
        assert!((s.cache_write_cost_usd - 1.00).abs() < 1e-9, "{s:?}");
    }

    // ── Rule: OpenAI.annotate is a no-op and counts cacheable tokens only
    //   above the threshold ───────────────────────────────────────────────

    #[test]
    fn openai_annotate_threshold() {
        let p = OpenAICacheProvider::default();
        let mut below = ctx(1023, vec![], 0);
        let r = p.annotate(&mut below);
        assert_eq!(r.markers_inserted, 0);
        assert_eq!(r.estimated_cacheable_tokens, 0);

        let mut above = ctx(2048, vec![], 0);
        let r = p.annotate(&mut above);
        assert_eq!(r.markers_inserted, 0);
        assert_eq!(r.estimated_cacheable_tokens, 2048);
    }

    // ── Rule: OpenAI.parse_cache_stats reads cached_tokens; write is zero ─

    #[test]
    fn openai_parse_reads_only_reads() {
        let p = OpenAICacheProvider::default();
        let s = p.parse_cache_stats(&response(Some(512), Some(99))).unwrap();
        assert_eq!(s.cache_read_tokens, 512);
        assert_eq!(s.cache_write_tokens, 0);
        assert!(p.parse_cache_stats(&response(None, None)).is_none());
    }

    // ── Rule: Ollama supports_caching is false; all ops are no-ops ───────

    #[test]
    fn ollama_no_op() {
        let p = OllamaCacheProvider;
        assert!(!p.supports_caching());
        assert_eq!(p.provider_name(), "ollama");
        let mut c = ctx(99, vec![], 0);
        assert_eq!(p.annotate(&mut c), CacheAnnotationResult::default());
        assert!(p.parse_cache_stats(&response(Some(5), Some(5))).is_none());
    }

    // ── Rule: auto_detect maps provider names case-insensitively ─────────

    #[test]
    fn auto_detect_maps_known_providers() {
        assert_eq!(
            auto_detect("anthropic").map(|p| p.provider_name()),
            Some("anthropic")
        );
        assert_eq!(
            auto_detect("OpenAI").map(|p| p.provider_name()),
            Some("openai")
        );
        assert_eq!(
            auto_detect("ollama").map(|p| p.provider_name()),
            Some("ollama")
        );
        assert!(auto_detect("mystery").is_none());
    }

    // ── Fixture-replay test ──────────────────────────────────────────────

    #[derive(serde::Deserialize)]
    struct FixtureFile {
        cases: Vec<FixtureCase>,
    }

    #[derive(serde::Deserialize)]
    struct FixtureCase {
        name: String,
        provider: String,
        usage: FixtureUsage,
        expected: FixtureExpected,
    }

    #[derive(serde::Deserialize)]
    struct FixtureUsage {
        cache_read_tokens: Option<u32>,
        cache_write_tokens: Option<u32>,
    }

    #[derive(serde::Deserialize)]
    struct FixtureExpected {
        is_some: bool,
        #[serde(default)]
        cache_read_tokens: u32,
        #[serde(default)]
        cache_write_tokens: u32,
    }

    #[test]
    fn fixture_parse_cache_stats() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/cache_provider/parse_cache_stats.json");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let file: FixtureFile = serde_json::from_str(&text).unwrap();
        for case in file.cases {
            let resp = response(case.usage.cache_read_tokens, case.usage.cache_write_tokens);
            let stats: Option<CacheStats> = match case.provider.as_str() {
                "anthropic" => AnthropicCacheProvider::default().parse_cache_stats(&resp),
                "openai" => OpenAICacheProvider::default().parse_cache_stats(&resp),
                "ollama" => OllamaCacheProvider.parse_cache_stats(&resp),
                "null" => NullCacheProvider.parse_cache_stats(&resp),
                other => panic!("unknown provider in fixture: {other}"),
            };
            match (case.expected.is_some, stats) {
                (true, Some(s)) => {
                    assert_eq!(
                        s.cache_read_tokens, case.expected.cache_read_tokens,
                        "case {}: read tokens",
                        case.name
                    );
                    assert_eq!(
                        s.cache_write_tokens, case.expected.cache_write_tokens,
                        "case {}: write tokens",
                        case.name
                    );
                }
                (false, None) => {}
                (true, None) => panic!("case {}: expected Some, got None", case.name),
                (false, Some(_)) => panic!("case {}: expected None, got Some", case.name),
            }
        }
    }
}
