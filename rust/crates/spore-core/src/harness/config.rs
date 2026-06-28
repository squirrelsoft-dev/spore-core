//! Harness configuration and builder extracted from the harness god-module
//! (Phase 6.1): `MemoryConfig`, the `HarnessConfig` value object, and the
//! `HarnessBuilder`. Pure move — no behavior change.

use super::*;
/// A configured memory source for the live loop (issue #160 / SC-26 follow-up).
///
/// Bundles an `Arc<dyn MemoryProvider>` (now object-safe — see
/// [`MemoryProvider`](crate::memory::MemoryProvider)) with the per-turn query
/// policy. When present on [`HarnessConfig::memory`], the harness queries the
/// provider each turn in
/// [`build_context_sources`](StandardHarness::build_context_sources) and injects
/// the returned items into [`ContextSources::memory`], which the production
/// [`StandardCompactionAdapter`](crate::compaction_adapter::StandardCompactionAdapter)
/// renders into the leading structural System block — alongside guides and
/// skills, with no consumer-side wrapper. `None` (the default) leaves
/// `ContextSources::memory` empty, byte-identical to pre-#160 behaviour.
#[derive(Clone)]
pub struct MemoryConfig {
    /// The provider queried each turn.
    pub provider: Arc<dyn crate::memory::MemoryProvider>,
    /// Fixed query text. `None` (the default) uses the current task's
    /// `instruction`, so retrieved memory tracks what the agent is working on.
    pub query: Option<String>,
    /// Optional domain filter passed to [`MemoryQuery::domain`](crate::memory::MemoryQuery).
    pub domain: Option<String>,
    /// Minimum relevance score; items below it are dropped. Defaults to `0.5`
    /// (matching [`MemoryQuery`](crate::memory::MemoryQuery)'s default).
    pub min_relevance: f32,
    /// Maximum number of items injected per turn. Defaults to `10`.
    pub max_items: u32,
}

impl MemoryConfig {
    /// A memory source with the default query policy: query by the current
    /// task instruction, `min_relevance = 0.5`, `max_items = 10`, no domain
    /// filter.
    pub fn new(provider: Arc<dyn crate::memory::MemoryProvider>) -> Self {
        Self {
            provider,
            query: None,
            domain: None,
            min_relevance: 0.5,
            max_items: 10,
        }
    }

    /// Override the per-turn query text. Without this the task instruction is
    /// used.
    pub fn query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    /// Restrict retrieval to a single memory `domain`.
    pub fn domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Override the minimum relevance threshold.
    pub fn min_relevance(mut self, min_relevance: f32) -> Self {
        self.min_relevance = min_relevance;
        self
    }

    /// Override the per-turn item cap.
    pub fn max_items(mut self, max_items: u32) -> Self {
        self.max_items = max_items;
        self
    }
}

impl std::fmt::Debug for MemoryConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryConfig")
            .field("query", &self.query)
            .field("domain", &self.domain)
            .field("min_relevance", &self.min_relevance)
            .field("max_items", &self.max_items)
            .finish_non_exhaustive()
    }
}

/// Components injected at construction. Mirrors `HarnessConfig` in the spec.
/// Components not yet covered by their own issue are optional here so the
/// harness can be exercised before #4–#13 land.
///
/// Every component, including the agent, is held erased behind `Arc<dyn _>`.
/// The `Agent` trait (issue #2) is dyn-compatible via the hand-rolled `BoxFut`
/// pattern, so no type parameter is needed (issue #45).
#[derive(Clone)]
pub struct HarnessConfig {
    pub tool_registry: Arc<dyn ToolRegistry>,
    pub sandbox: Arc<dyn SandboxProvider>,
    /// How a [`SandboxViolation`] surfaced by a tool (or the pre-dispatch
    /// `validate` check) is handled: fed back to the model as a recoverable
    /// error (the default) or halted on. See [`SandboxViolationPolicy`].
    pub sandbox_violation_policy: SandboxViolationPolicy,
    pub context_manager: Arc<dyn ContextManager>,
    pub termination_policy: Arc<dyn TerminationPolicy>,
    pub middleware: Option<Arc<dyn MiddlewareChain>>,
    pub observability: Option<Arc<dyn ObservabilityProvider>>,
    /// Post-compaction verifier (issue #29/#46). The harness runs it after
    /// each compaction turn and retries up to `max_compaction_attempts` before
    /// accepting a failing summary. Defaults to [`KeyTermVerifier`].
    pub compaction_verifier: Arc<dyn CompactionVerifier>,
    /// Maximum compaction-summary attempts before accepting a failing summary
    /// anyway (issue #46). Defaults to `2` (mirrors `CompactionConfig`).
    pub max_compaction_attempts: u32,
    /// Token → USD pricing used to stamp `cost_usd` on emitted [`TurnSpan`]s.
    /// Defaults to [`PricingTable::DEFAULT`] (zero cost) when unset.
    pub pricing: PricingTable,
    /// LLM-native content capture guard + truncation limit (issue #64).
    /// Defaults to [`ContentCaptureConfig::default`] (OFF). When disabled the
    /// harness populates no `gen_ai.*` content and the JSONL stays
    /// byte-identical to the pre-#64 metrics-only output.
    pub content_capture: ContentCaptureConfig,
    /// Optional deterministic tool-call repair provider. When set, recoverable
    /// tool-dispatch errors trigger an argument-coercion + re-dispatch attempt
    /// before falling back to error feedback. `None` (the default) preserves
    /// today's behaviour byte-for-byte.
    pub tool_call_repair: Option<Arc<dyn ToolCallRepair>>,
    /// Maximum number of repair-and-re-dispatch attempts per tool call.
    /// Defaults to `1`. Ignored when `tool_call_repair` is `None`.
    pub max_repair_attempts: u32,
    /// PlanExecute plan-empty re-ask cap (see
    /// [`StrategyExecutor::plan_empty_max_retries`]). `0` (default) halts on the
    /// first empty plan; higher values re-ask the planner that many times for the
    /// plan as message content before `EmptyPlan`.
    pub plan_empty_max_retries: u32,
    /// Conservative tool-call nudge: when a turn advertises tools but the model
    /// answers in ACTION-INTENT prose ("I'll list the directory…") WITHOUT calling
    /// a tool, nudge it to actually emit the call and force one more turn (bounded
    /// per window). Unlike [`prompt_tool_call_flag`](Self::prompt_tool_call_flag)
    /// it does NOT switch the model to prompt-based tool calling — it only corrects
    /// narrate-instead-of-act. `false` (default) is byte-identical.
    pub tool_call_nudge: bool,
    /// Maximum number of consecutive Stop-hook blocks within a single `run()`
    /// before the loop terminates anyway (issue #69, Decision 4/5). The
    /// counter is PER-RUN — it resets on every `run()` call, so a resumed
    /// session starts fresh. Defaults to `8` (matching Claude Code).
    pub max_stop_blocks: u32,
    /// Consecutive-recoverable-tool-error breaker threshold `N` (issue #137).
    /// The ReAct turn loop tracks, per tool name, the current run of
    /// consecutive recoverable [`ToolOutput::Error`] results keyed on identical
    /// arguments (any SUCCESS for that tool resets the run). At `N` identical
    /// failures it injects ONE corrective message (the [`enrich_tool_error`]
    /// schema+hint, as a USER-role message); at exactly `2 * N` it STOPS
    /// looping and resolves the node's [`BudgetExhaustedBehavior`] via the
    /// single budget-exhaustion resolution site — WITHOUT burning the remaining
    /// budget — terminating with [`HaltReason::ToolErrorLoop`] (never
    /// [`HaltReason::BudgetExceeded`]). The `2×` hard-stop multiplier is FIXED,
    /// not an independent field. Defaults to `3`.
    ///
    /// The SAME `N`/`2N` thresholds also drive the repeated-identical-call
    /// breaker (the no-progress twin of the error breaker): the model calling one
    /// tool with identical args and SUCCEEDING `N` times in a row is nudged once,
    /// and at `2N` the run hard-stops with [`HaltReason::RepeatedToolCall`]. `0`
    /// disables BOTH breakers.
    pub error_loop_threshold: u32,
    /// MIGRATION GATE (issue #139) — NOT a permanent feature flag. When `true`,
    /// a [`ReactConfig`] leaf carrying `output = Some(..)` has its resolved
    /// output schema DELIVERED to the model (directive seed + the
    /// [`ModelParams::output_schema`] constrained-decoding channel) and its
    /// terminal `FinalResponse` VALIDATED against that schema; a validation
    /// failure feeds the error back and retries up to
    /// [`output_schema_max_retries`](Self::output_schema_max_retries) extra turns
    /// (within budget), then terminates with
    /// [`HaltReason::OutputSchemaViolation`]. Enforcement is UNIFORM — it applies
    /// to structured-slot leaves (plan / worker / propose) too, and is additive
    /// to and earlier than any downstream combinator validation.
    ///
    /// Default `false` (OFF) is NON-NEGOTIABLE: it keeps every existing replay
    /// fixture byte-for-byte green during migration. When OFF, `ReactConfig::run`
    /// behaves EXACTLY as before — no resolve, no delivery, no validation.
    pub enforce_output_schemas: bool,
    /// The `N` extra terminal-validation retry turns granted when
    /// [`enforce_output_schemas`](Self::enforce_output_schemas) is ON and a
    /// terminal fails output-schema validation (issue #139). Total attempts =
    /// `1 + N`. Retried turns COUNT against the turn budget; a retry that would
    /// exceed the budget surfaces the budget terminal instead of
    /// [`HaltReason::OutputSchemaViolation`] (budget-cap-wins precedence).
    /// Defaults to `2` (mirrors [`error_loop_threshold`](Self::error_loop_threshold)'s
    /// builder default-handling idiom).
    pub output_schema_max_retries: u32,
    /// Lifecycle hook chain (issue #69). When set, the harness fires the wired
    /// lifecycle events (`PreTurn`, `Stop`, `OnError`, …) through it.
    /// `None` (the default) preserves today's behaviour byte-for-byte.
    pub hooks: Option<Arc<dyn crate::hooks::HookChain>>,
    /// Pluggable per-domain persistence layer (issue #73). Defaults to an
    /// all-no-op [`StorageProvider`] so existing callers/tests compile and
    /// behave unchanged. v1 is expose-only — the run/resume loop is NOT
    /// modified to read/write sessions internally.
    pub storage: Arc<crate::storage::StorageProvider>,
    /// The STABLE project namespace for DURABLE artifacts (issue #142). Where
    /// the per-window [`SessionId`] is regenerated on every Ralph context-window
    /// reset (`SessionId::generate()`), this [`ProjectId`](crate::storage::ProjectId)
    /// stays constant across windows AND process restarts — which is what lets
    /// the `task_list`, plan artifact, and Ralph checkpoint persist across a
    /// window reset instead of being orphaned under a regenerated session.
    /// Defaults (at build) to a project id derived from
    /// `sandbox.workspace_root()` (decision 5 — NOT process cwd). Durable
    /// `RunStore` call sites key by `project_id.namespace()` (namespace-reuse on
    /// the existing `&SessionId` axis); ephemeral session/conversation state
    /// stays keyed by the per-window `SessionId`.
    pub project_id: crate::storage::ProjectId,
    /// Source of conditional prompt chunks (issue #79). Defaults to an empty
    /// [`InMemoryChunkProvider`](crate::prompt_assembly::InMemoryChunkProvider).
    /// The harness loads chunks from it at construction and feeds them through
    /// a [`ContextSourcesBuilder`](crate::prompt_assembly::ContextSourcesBuilder).
    pub chunk_provider: Arc<dyn crate::prompt_assembly::ChunkProvider>,
    /// Outer-loop cap for the `Ralph` loop strategy (issue #58, B3): the maximum
    /// number of context-window RESETS the continuation loop performs before
    /// halting with [`HaltReason::RalphCompletionUnmet`] when tasks are still
    /// incomplete. Independent from `max_turns` (the per-context-window ReAct
    /// turn budget) and from `max_stop_blocks` (the per-sub-loop Stop-block cap).
    /// Defaults to `3` (matching the `SelfVerifying` verifier `max_iterations`
    /// precedent). Irrelevant for every other loop strategy.
    pub max_resets: u32,
    /// Optional VCS read seam for the `Ralph` loop strategy (issue #58 v2). When
    /// `Some`, Ralph's per-window reload phase ALSO calls
    /// [`VcsProvider::log`] and injects the output into the fresh context window
    /// as a delimited "Recent VCS history:" section — alongside the reloaded
    /// `.spore/progress.json` + `.spore/feature_list.json` content. When `None`
    /// (the default) the git-log section is OMITTED and Ralph behaves
    /// byte-for-byte like v1 (the B4→None decision). Irrelevant for every other
    /// loop strategy.
    pub vcs_provider: Option<Arc<dyn VcsProvider>>,
    /// Catalogue tools accumulated via [`HarnessBuilder::tool`] / `tools`
    /// (issue #81), drained into a populated
    /// [`StandardToolRegistry`](crate::tool_registry::StandardToolRegistry) at
    /// [`build`](HarnessBuilder::build). When `Some`, the run loop bridges it
    /// per-run via [`RealToolRegistry`](crate::tool_registry::RealToolRegistry)
    /// — threading the run's `SessionId`, sandbox, and storage into every tool
    /// dispatch — and uses that instead of [`tool_registry`](Self::tool_registry)
    /// (which stays the harness-loop seam for custom slim registries). `None`
    /// (the default) preserves the `tool_registry`-only path byte-for-byte.
    pub catalogue_registry: Option<Arc<crate::tool_registry::StandardToolRegistry>>,
    /// Per-toolset catalogues (Issue 2: per-node toolset scoping), keyed by the
    /// non-empty `ToolsetRef` handle a leaf carries. Populated by
    /// [`HarnessBuilder::toolset_tools`] — each value is a populated
    /// [`StandardToolRegistry`](crate::tool_registry::StandardToolRegistry)
    /// folded from that key's catalogue tools. At dispatch the run loop bridges
    /// the matching catalogue per-run via
    /// [`RealToolRegistry`](crate::tool_registry::RealToolRegistry) (same
    /// sandbox/session/storage wiring as the global
    /// [`catalogue_registry`](Self::catalogue_registry)), so a node with a
    /// non-empty `toolset` handle dispatches ONLY its own tools. A leaf with an
    /// EMPTY handle (`""`) — or a non-empty handle with no entry here — falls
    /// back to the global catalogue / `tool_registry` seam (back-compat with
    /// examples 01–11 that use `.tools()`). Empty (the default) preserves
    /// today's behaviour byte-for-byte.
    pub toolset_catalogues:
        std::collections::HashMap<String, Arc<crate::tool_registry::StandardToolRegistry>>,
    /// Operating system prompt prepended to each turn's assembled context when
    /// the context manager renders none (issue #91). See
    /// [`HarnessBuilder::system_prompt`]. `None` (the default) preserves
    /// today's behaviour byte-for-byte.
    pub system_prompt: Option<String>,
    /// Guides (skills/playbooks/domain knowledge) injected structurally into
    /// every turn's assembled context via the rich `ContextSources` seam (issue
    /// #115 / SC-26 / #9). The harness clones these into
    /// [`ContextSources::guides`] each turn; the production
    /// [`StandardCompactionAdapter`] renders them into the leading System block,
    /// not as ad-hoc User messages. Empty (the default) preserves today's
    /// behaviour byte-for-byte. See [`HarnessBuilder::guide`].
    pub guides: Vec<Guide>,
    /// Optional skill catalog (issue #115 / SC-26). When set, the harness injects
    /// its manifest + active skill bodies into [`ContextSources::guides`] each turn
    /// (progressive disclosure) and the `load_skill` tool activates skills against
    /// its shared active set. `None` (the default) means no skills. See
    /// [`HarnessBuilder::skills`].
    pub skills: Option<Arc<SkillCatalog>>,
    /// Optional memory source (issue #160 / SC-26 follow-up). When set, the
    /// harness queries the provider each turn and injects the returned items
    /// into [`ContextSources::memory`], rendered into the structural System
    /// block alongside guides + skills. `None` (the default) leaves memory
    /// empty, byte-identical to pre-#160 behaviour. See
    /// [`HarnessBuilder::memory`] and [`MemoryConfig`].
    pub memory: Option<MemoryConfig>,
    /// Authoritative per-run model sampling/decoding parameters (issue #93).
    /// The harness replaces each turn's `Context.params` with this value
    /// UNCONDITIONALLY (builder params win) right before the request is built,
    /// so the configured params reach every agent turn that requests tools.
    /// See [`HarnessBuilder::model_params`]. Defaults to
    /// [`ModelParams::default`].
    pub model_params: ModelParams,
    /// Opt-in automatic conversation-history threading via the
    /// [`SessionStore`](crate::storage::SessionStore) (issue #102). When `true`,
    /// the harness:
    /// - **auto-loads** the prior [`SessionState`] for the run's `session_id`
    ///   from the store at the start of a fresh `run()` (ReAct / SelfVerifying
    ///   strategies ONLY — Ralph/HillClimbing discard incoming state by design),
    ///   so a caller can resume by reusing the same `session_id` instead of
    ///   threading messages by hand. An explicit
    ///   [`with_session_state`](HarnessRunOptions::with_session_state) always
    ///   wins (the load is skipped).
    /// - **auto-persists** the post-run [`SessionState`] back to the store at the
    ///   terminal seam (one write per `run()`/`resume()` — Success/Failure
    ///   synthesize a [`PausedState`] with empty pending fields, while
    ///   `WaitingForHuman`/`Escalate` persist their carried `PausedState`).
    ///
    /// Storage errors are swallowed-and-logged (load failure ⇒ start fresh;
    /// persist failure ⇒ continue) — never surfaced as a [`HaltReason`].
    ///
    /// `false` (the default) preserves today's behaviour byte-for-byte: ZERO
    /// `get_session`/`put_session` calls and identical message flow.
    pub auto_persist_sessions: bool,
    /// Session-scoped prompt-based-tool-calling escalation flag, shared with the
    /// [`AdaptiveToolCallModelInterface`](crate::prompt_tool_call::AdaptiveToolCallModelInterface)
    /// installed by [`HarnessBuilder::conversational`]. `Some` only on that path.
    /// The run loop resets it at the start of each turn-loop window and flips it
    /// to `true` on detecting a prose response where a tool call was expected;
    /// the wrapped model reads it to switch into prompt-based mode. `None`
    /// disables adaptive escalation (every non-`conversational` construction).
    pub prompt_tool_call_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Per-kind consult handlers (issue #114), keyed by `ConsultRequest.kind`.
    /// Empty (the default) means consults are NOT mediated: a worker that pauses
    /// with [`RunResult::Consult`] surfaces it unchanged to its caller (R6
    /// graceful degradation), and existing callers are unaffected byte-for-byte
    /// (R9). When populated, the matching entry's helper harness is run
    /// deterministically by `SubagentTool` (the A1 mediation seam) — the
    /// orchestrator model is never involved.
    pub consult_handlers: std::collections::HashMap<String, ConsultHandlerEntry>,
    /// Runtime resolver for the serializable strategy handles
    /// (`AgentRef`/`ToolsetRef`/`SchemaRef`) and `StrategyRef::Custom` keys held
    /// by a task's `LoopStrategy` tree (issue #120). [`StandardHarness::run`]
    /// calls [`ExecutionRegistry::validate`] at entry, so an unresolved handle is
    /// a startup error before the first turn. This slice is ADDITIVE (Option B):
    /// the registry coexists with the deprecated single-collaborator fields and
    /// is not yet read by the run bodies (that lands in #123/#124). Defaults to
    /// an empty registry.
    pub registry: ExecutionRegistry,
    /// HITL-vs-AFK escalation knob (issue #120, PRD goal #7): whether budget
    /// escalation surfaces to a human or proceeds autonomously. STORED only this
    /// slice — consumed in #130. Builder defaults to
    /// [`EscalationMode::SurfaceToHuman`].
    pub escalation_mode: EscalationMode,
    /// Optional deferred-tool exposure (issue: native-tool-calling description
    /// cliff / MCP scale). When `Some`, the resolved catalogue registry is wrapped
    /// in a [`ToolSearchRegistry`](crate::tool_registry::ToolSearchRegistry): only
    /// the configured core set + a `tool_search` meta-tool are advertised up front,
    /// and the model loads the rest of the catalogue on demand. `None` (the
    /// default) advertises the full catalogue, byte-for-byte today's behaviour.
    pub tool_search: Option<crate::tool_registry::ToolSearchConfig>,
}

/// Fluent assembler for a [`HarnessConfig`] / [`StandardHarness`].
///
/// The harness follows strict inversion of control: every component is
/// injected. `HarnessBuilder` is the canonical assembly point — it takes the
/// required components up front and exposes fluent setters for the optional
/// ones (middleware, observability, pricing). It is the intended caller that
/// wires the [`ObservabilityProvider`] into the loop, including the durable
/// outbox via [`with_observability_outbox`](Self::with_observability_outbox).
///
/// ```no_run
/// # use std::sync::Arc;
/// # use spore_core::{HarnessBuilder, PricingTable};
/// # fn demo(
/// #     agent: Arc<dyn spore_core::Agent>,
/// #     tools: Arc<dyn spore_core::HarnessToolRegistry>,
/// #     sandbox: Arc<dyn spore_core::SandboxProvider>,
/// #     ctx: Arc<dyn spore_core::HarnessContextManager>,
/// #     term: Arc<dyn spore_core::TerminationPolicy>,
/// # ) {
/// let harness = HarnessBuilder::new(agent, tools, sandbox, ctx, term)
///     .with_observability_outbox(".spore")
///     .pricing(PricingTable::DEFAULT)
///     .build();
/// # }
/// ```
pub struct HarnessBuilder {
    agent: Arc<dyn Agent>,
    tool_registry: Arc<dyn ToolRegistry>,
    sandbox: Arc<dyn SandboxProvider>,
    sandbox_violation_policy: SandboxViolationPolicy,
    context_manager: Arc<dyn ContextManager>,
    termination_policy: Arc<dyn TerminationPolicy>,
    middleware: Option<Arc<dyn MiddlewareChain>>,
    observability: Option<Arc<dyn ObservabilityProvider>>,
    compaction_verifier: Arc<dyn CompactionVerifier>,
    max_compaction_attempts: u32,
    pricing: PricingTable,
    content_capture: ContentCaptureConfig,
    tool_call_repair: Option<Arc<dyn ToolCallRepair>>,
    max_repair_attempts: u32,
    plan_empty_max_retries: u32,
    tool_call_nudge: bool,
    max_stop_blocks: u32,
    /// Consecutive-recoverable-tool-error breaker threshold `N` (issue #137).
    /// Defaults to `3`. See [`HarnessConfig::error_loop_threshold`].
    error_loop_threshold: u32,
    /// MIGRATION GATE (issue #139): deliver + enforce `ReactConfig.output`
    /// schemas. Defaults to `false` (OFF). See
    /// [`HarnessConfig::enforce_output_schemas`].
    enforce_output_schemas: bool,
    /// Extra output-schema validation retry turns `N` (issue #139). Defaults to
    /// `2`. See [`HarnessConfig::output_schema_max_retries`].
    output_schema_max_retries: u32,
    hooks: Option<Arc<dyn crate::hooks::HookChain>>,
    /// #124: the default SelfVerifying verifier, folded into the registry under
    /// the empty key at build (no longer a live `HarnessConfig` field).
    verifier: Option<Arc<dyn crate::verifier::Verifier>>,
    storage: Option<Arc<crate::storage::StorageProvider>>,
    /// The STABLE durable-storage project namespace (issue #142). `None` (the
    /// default) resolves at [`build_config`](HarnessBuilder::build_config) to a
    /// [`ProjectId`](crate::storage::ProjectId) derived from
    /// `sandbox.workspace_root()` (decision 5 — NOT process cwd), canonicalizing
    /// the path first; an explicit [`project_id`](HarnessBuilder::project_id)
    /// always wins. See [`HarnessConfig::project_id`].
    project_id: Option<crate::storage::ProjectId>,
    /// Conditional prompt-chunk source (issue #79). `None` resolves to an empty
    /// [`InMemoryChunkProvider`](crate::prompt_assembly::InMemoryChunkProvider)
    /// at build time.
    chunk_provider: Option<Arc<dyn crate::prompt_assembly::ChunkProvider>>,
    /// Outer-loop reset cap for the `Ralph` loop strategy (issue #58, B3).
    /// Defaults to `3`.
    max_resets: u32,
    /// Optional VCS read seam for the `Ralph` loop strategy (issue #58 v2).
    /// `None` (the default) omits the git-log reload section (v1 behavior).
    vcs_provider: Option<Arc<dyn VcsProvider>>,
    /// Scoring oracle for the `HillClimbing` loop strategy (issue #60). `None`
    /// (the default) makes a `HillClimbing` run halt with
    /// `HillClimbingMisconfigured`.
    metric_evaluator: Option<Arc<dyn crate::metric::MetricEvaluator>>,
    /// Standard catalogue tools accumulated via [`HarnessBuilder::tool`] /
    /// [`HarnessBuilder::tools`] (issue #81). They are drained into a populated
    /// [`StandardToolRegistry`](crate::tool_registry::StandardToolRegistry) by
    /// [`HarnessBuilder::build_tool_registry`], applying last-wins upsert.
    standard_tools: Vec<crate::tools::StandardTool>,
    /// Per-toolset catalogue tools (Issue 2: per-node toolset scoping),
    /// accumulated via [`HarnessBuilder::toolset_tools`] and keyed by the
    /// non-empty toolset handle a leaf carries. Each bucket is folded into a
    /// populated [`StandardToolRegistry`](crate::tool_registry::StandardToolRegistry)
    /// (last-wins upsert) at [`build_config`](HarnessBuilder::build_config) and
    /// stored in [`HarnessConfig::toolset_catalogues`]. Empty (the default)
    /// preserves today's single-catalogue behaviour byte-for-byte.
    toolset_tools: std::collections::HashMap<String, Vec<crate::tools::StandardTool>>,
    /// Optional operating system prompt prepended to each turn's assembled
    /// context (issue #91) when the context manager renders none. `None` (the
    /// default) preserves today's behaviour. See [`system_prompt`](HarnessBuilder::system_prompt).
    system_prompt: Option<String>,
    /// Guides injected structurally into every turn via `ContextSources` (issue
    /// #115 / #9). Empty (the default) preserves today's behaviour. See
    /// [`guide`](HarnessBuilder::guide).
    guides: Vec<Guide>,
    /// Optional skill catalog (issue #115 / SC-26). See [`skills`](HarnessBuilder::skills).
    skills: Option<Arc<SkillCatalog>>,
    /// Optional memory source (issue #160). `None` (the default) leaves memory
    /// empty. See [`memory`](HarnessBuilder::memory).
    memory: Option<MemoryConfig>,
    /// Authoritative per-run model sampling/decoding parameters (issue #93).
    /// Defaults to [`ModelParams::default`]. See
    /// [`model_params`](HarnessBuilder::model_params).
    model_params: ModelParams,
    /// Opt-in automatic session-state threading + auto-persist (issue #102).
    /// `false` (the default) preserves today's behaviour byte-for-byte (ZERO
    /// session-store I/O). See [`auto_persist_sessions`](HarnessBuilder::auto_persist_sessions).
    auto_persist_sessions: bool,
    /// Session-scoped escalation flag shared with an
    /// [`AdaptiveToolCallModelInterface`](crate::prompt_tool_call::AdaptiveToolCallModelInterface)
    /// installed by [`conversational`](HarnessBuilder::conversational). `Some`
    /// only on that path; the run loop flips it on detecting a prose response so
    /// the wrapped model switches to prompt-based tool calling for the rest of
    /// the run. `None` for every other construction (no adaptive wrapping).
    prompt_tool_call_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Per-kind consult handlers (issue #114). Empty (the default) preserves
    /// today's behaviour byte-for-byte (no mediation). See
    /// [`consult_handler`](HarnessBuilder::consult_handler).
    consult_handlers: std::collections::HashMap<String, ConsultHandlerEntry>,
    /// Runtime strategy-handle resolver (issue #120). Defaults to an empty
    /// [`ExecutionRegistry`]. See [`registry`](HarnessBuilder::registry).
    registry: ExecutionRegistry,
    /// HITL-vs-AFK escalation knob (issue #120). Defaults to
    /// [`EscalationMode::SurfaceToHuman`]. See
    /// [`escalation_mode`](HarnessBuilder::escalation_mode).
    escalation_mode: EscalationMode,
    /// Optional deferred-tool exposure. `None` (the default) advertises the full
    /// catalogue. See [`tool_search`](HarnessBuilder::tool_search).
    tool_search: Option<crate::tool_registry::ToolSearchConfig>,
}

impl HarnessBuilder {
    /// Start a builder from the five required components. Optional components
    /// default to `None`/[`PricingTable::DEFAULT`] until set.
    pub fn new(
        agent: Arc<dyn Agent>,
        tool_registry: Arc<dyn ToolRegistry>,
        sandbox: Arc<dyn SandboxProvider>,
        context_manager: Arc<dyn ContextManager>,
        termination_policy: Arc<dyn TerminationPolicy>,
    ) -> Self {
        Self {
            agent,
            tool_registry,
            sandbox,
            sandbox_violation_policy: SandboxViolationPolicy::default(),
            context_manager,
            termination_policy,
            middleware: None,
            observability: None,
            compaction_verifier: Arc::new(KeyTermVerifier),
            max_compaction_attempts: 2,
            pricing: PricingTable::DEFAULT,
            content_capture: ContentCaptureConfig::default(),
            tool_call_repair: None,
            max_repair_attempts: 1,
            plan_empty_max_retries: 0,
            tool_call_nudge: false,
            max_stop_blocks: 8,
            error_loop_threshold: 3,
            enforce_output_schemas: false,
            output_schema_max_retries: 2,
            hooks: None,
            verifier: None,
            storage: None,
            project_id: None,
            chunk_provider: None,
            max_resets: 3,
            vcs_provider: None,
            metric_evaluator: None,
            standard_tools: Vec::new(),
            toolset_tools: std::collections::HashMap::new(),
            system_prompt: None,
            guides: Vec::new(),
            skills: None,
            memory: None,
            model_params: ModelParams::default(),
            auto_persist_sessions: false,
            prompt_tool_call_flag: None,
            consult_handlers: std::collections::HashMap::new(),
            registry: ExecutionRegistry::empty(),
            escalation_mode: EscalationMode::SurfaceToHuman,
            tool_search: None,
        }
    }

    /// Assemble a minimal conversational harness from a model — no tools, no
    /// filesystem.
    ///
    /// This is the few-lines path: it defaults every required component so you
    /// can go from a model to a running harness in one call. The defaults are a
    /// [`ModelAgent`](crate::agent::ModelAgent) over `model`, an
    /// [`EmptyToolRegistry`], a [`NullSandbox`], a
    /// [`StandardContextManager`](crate::context::StandardContextManager) with a
    /// null cache provider and default compaction, and
    /// [`CompleteOnFinalResponse`] termination (the model's first final response
    /// is the result).
    ///
    /// Every default is overridable: add catalogue tools with
    /// [`tool`](Self::tool) / [`tools`](Self::tools), or supply your own
    /// component to [`new`](Self::new) directly.
    ///
    /// ```no_run
    /// # use spore_core::{Harness, HarnessBuilder, OllamaModelInterface, HarnessRunOptions, Task};
    /// # async fn run() {
    /// let harness = HarnessBuilder::conversational(OllamaModelInterface::new("llama3.2")).build();
    /// let result = harness
    ///     .run(HarnessRunOptions::new(Task::simple("Reply with a friendly greeting.")))
    ///     .await;
    /// # let _ = result;
    /// # }
    /// ```
    pub fn conversational<M: crate::model::ModelInterface + 'static>(model: M) -> Self {
        use crate::compaction_adapter::HarnessContextManagerExt;
        let model = Arc::new(model);
        // Install the adaptive prompt-based tool-calling wrapper around the
        // agent's model. While its flag is unset it delegates natively
        // (byte-for-byte); the run loop flips the flag on detecting a prose
        // response so the model switches to prompt-based tool calling for the
        // rest of the run. The context manager keeps the *raw* model (its only
        // model use is compaction summarization, which advertises no tools).
        let prompt_tool_call_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let adaptive_model = Arc::new(
            crate::prompt_tool_call::AdaptiveToolCallModelInterface::new(
                model.clone(),
                prompt_tool_call_flag.clone(),
            ),
        );
        let agent: Arc<dyn Agent> = Arc::new(crate::agent::ModelAgent::new(
            crate::agent::AgentId::new("agent"),
            adaptive_model,
        ));
        let tool_registry: Arc<dyn ToolRegistry> = Arc::new(EmptyToolRegistry);
        let sandbox: Arc<dyn SandboxProvider> = Arc::new(NullSandbox);
        let context_manager: Arc<dyn ContextManager> =
            Arc::new(crate::context::StandardContextManager::new(
                model,
                Arc::new(crate::cache_provider::NullCacheProvider),
                crate::context::CompactionConfig::default(),
            ))
            .into_harness_adapter();
        let termination_policy: Arc<dyn TerminationPolicy> = Arc::new(CompleteOnFinalResponse);
        let mut builder = Self::new(
            agent,
            tool_registry,
            sandbox,
            context_manager,
            termination_policy,
        );
        builder.prompt_tool_call_flag = Some(prompt_tool_call_flag);
        builder
    }

    /// Like [`conversational`](Self::conversational), but takes an already-boxed
    /// `Arc<dyn ModelInterface>` (SC-2).
    ///
    /// `ModelInterface` is now object-safe (`BoxFut`), so a consumer can hold a
    /// single `Arc<dyn ModelInterface>` — picking the provider at runtime — and
    /// inject it here without monomorphizing through a concrete model type. This
    /// is the entry point `CONVENTIONS.md` refers to when it says components are
    /// injected as `Arc<dyn ModelInterface>`.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use spore_core::{HarnessBuilder, ModelInterface, OllamaModelInterface};
    /// let model: Arc<dyn ModelInterface> = Arc::new(OllamaModelInterface::new("llama3.2"));
    /// let harness = HarnessBuilder::conversational_arc(model).build();
    /// # let _ = harness;
    /// ```
    pub fn conversational_arc(model: Arc<dyn crate::model::ModelInterface>) -> Self {
        // `Arc<dyn ModelInterface>` is itself a `ModelInterface` (blanket impl),
        // so the generic preset accepts it directly — the boxed model is shared
        // by reference-count, never re-minted.
        Self::conversational(model)
    }

    /// Built-in system prompt for [`coding_agent`](Self::coding_agent): a coding
    /// agent that ACTS through the workspace tools (rather than describing what it
    /// would do) and narrates each step to the user via `send_message`. Exposed so
    /// a consumer can extend it; override it wholesale with
    /// [`system_prompt`](Self::system_prompt).
    pub const CODING_AGENT_SYSTEM_PROMPT: &'static str =
        "You are a coding agent working inside a sandboxed workspace directory. \
         Explore with list_dir, read_file, grep, and find_files; create and change files with \
         write_file and edit_file; run commands with bash. Use relative paths only. \
         Act using tools — do not just describe what you would do. When the task is done, \
         reply with a short summary of what you changed. \
         The user CANNOT see your reasoning or your tool calls — they only see the messages you \
         send with the `send_message` tool and your final reply. So before (or as) you act, \
         call `send_message` with one short sentence saying what you are about to do, in \
         PARALLEL with the tool that does the work, so narration never costs an extra round trip. \
         Keep each message to a single short sentence.";

    /// Default per-scope auto-continue cap for the autonomous presets
    /// ([`coding_agent`](Self::coding_agent) / [`hill_climber`](Self::hill_climber)):
    /// grant up to this many extra step budgets at an `Escalate` point before the
    /// run gives up. Mirrors the hand-rolled drive loop the consumers used (the
    /// `12-cordyceps` example's `MAX_AUTO_CONTINUES`). Override the whole policy
    /// with [`escalation_mode`](Self::escalation_mode).
    pub const PRESET_MAX_AUTO_GRANTS: u32 = 10;
    /// Steps granted on each auto-continue for the autonomous presets (the
    /// `12-cordyceps` example's `CONTINUE_STEPS`). See
    /// [`PRESET_MAX_AUTO_GRANTS`](Self::PRESET_MAX_AUTO_GRANTS).
    pub const PRESET_STEPS_PER_GRANT: u32 = 25;

    /// `EscalationMode::AutoContinue` with the preset defaults
    /// ([`PRESET_MAX_AUTO_GRANTS`](Self::PRESET_MAX_AUTO_GRANTS) ×
    /// [`PRESET_STEPS_PER_GRANT`](Self::PRESET_STEPS_PER_GRANT), no `on_grant`
    /// observer). The "autonomous but capped" policy both autonomous presets share.
    fn preset_auto_continue() -> EscalationMode {
        EscalationMode::AutoContinue {
            max_grants: Self::PRESET_MAX_AUTO_GRANTS,
            steps_per_grant: Self::PRESET_STEPS_PER_GRANT,
            on_grant: None,
        }
    }

    /// Assemble an autonomous **coding agent** over a workspace directory (SC-8) —
    /// the looper preset.
    ///
    /// Builds on [`conversational`](Self::conversational) and wires the bits a
    /// coding agent always needs: a **read-write** [`WorkspaceScopedSandbox`] rooted
    /// at `workspace`, the full [`StandardTools::coding_set`] (read/write/edit/list/
    /// grep/find + `bash` + `send_message` + web/memory/task-list), the built-in
    /// [`CODING_AGENT_SYSTEM_PROMPT`](Self::CODING_AGENT_SYSTEM_PROMPT), and
    /// [`EscalationMode::AutoContinue`] (autonomous-but-capped — it keeps working
    /// through a spent step budget instead of pausing, so there is no consumer
    /// drive loop to hand-roll; SC-5).
    ///
    /// **Window sizing (SC-4/SC-6).** Size the model's context window ONCE on the
    /// model before passing it in (e.g. `OllamaModelInterface::with_context_window`):
    /// the preset's `conversational` context manager auto-derives its compaction
    /// budget from `provider().context_window`, and Ollama's `with_context_window`
    /// also sets `num_ctx` — so one call sizes both halves and no manual
    /// [`context_manager`](Self::context_manager) is needed.
    ///
    /// Returns [`Err`] if the workspace path can't be resolved (it must exist and
    /// canonicalize — the sandbox requirement). The strategy is per-run: pass a
    /// `ReAct` / `PlanExecute` [`Task`] to [`run`](Harness::run); the empty
    /// agent/toolset handles on its leaves resolve to this preset's defaults.
    ///
    /// ```no_run
    /// # use spore_core::{HarnessBuilder, OllamaModelInterface};
    /// # fn demo() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = OllamaModelInterface::new("gemma4:e4b").with_context_window(256_000);
    /// let harness = HarnessBuilder::coding_agent(model, "/path/to/project")?.build();
    /// # let _ = harness;
    /// # Ok(())
    /// # }
    /// ```
    pub fn coding_agent<M: crate::model::ModelInterface + 'static>(
        model: M,
        workspace: impl Into<std::path::PathBuf>,
    ) -> Result<Self, crate::sandbox::BuildError> {
        let sandbox = crate::sandbox::WorkspaceScopedSandbox::new(
            crate::sandbox::WorkspaceConfig::scoped(workspace),
        )?;
        Ok(Self::conversational(model)
            .sandbox(Arc::new(sandbox))
            .tools(crate::tools::StandardTools::coding_set())
            .system_prompt(Self::CODING_AGENT_SYSTEM_PROMPT)
            .escalation_mode(Self::preset_auto_continue()))
    }

    /// Assemble an autonomous **hill-climbing agent** (SC-8) — the cordyceps preset.
    ///
    /// Builds on [`conversational`](Self::conversational) and registers the scoring
    /// `evaluator` (required for the [`HillClimbing`](crate::LoopStrategy::HillClimbing)
    /// loop strategy) under the default handle, plus
    /// [`EscalationMode::AutoContinue`] (autonomous-but-capped; SC-5) so a spent
    /// per-iteration build budget keeps working instead of pausing.
    ///
    /// Unlike [`coding_agent`](Self::coding_agent) this does NOT install a sandbox
    /// or tools — hill-climbing workspaces vary (some climb a prose artifact, some
    /// climb files), and the build task's system prompt is task-specific. Add them
    /// with [`sandbox`](Self::sandbox) / [`tools`](Self::tools) /
    /// [`system_prompt`](Self::system_prompt) as the climb requires. Size the
    /// model's window on the model first (SC-4/SC-6), as in
    /// [`coding_agent`](Self::coding_agent).
    ///
    /// The `HillClimbing` config (direction / `max_stagnation` / per-iteration
    /// budget) lives on the per-run [`Task`]'s strategy; the iteration ceiling is
    /// the task's `max_turns`.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use spore_core::{HarnessBuilder, OllamaModelInterface, MetricEvaluator};
    /// # fn demo(evaluator: Arc<dyn MetricEvaluator>) {
    /// let model = OllamaModelInterface::new("gemma4:e4b").with_context_window(256_000);
    /// // Add a workspace `sandbox(..)` + `tools(..)` for the climb as it requires.
    /// let harness = HarnessBuilder::hill_climber(model, evaluator).build();
    /// # let _ = harness;
    /// # }
    /// ```
    pub fn hill_climber<M: crate::model::ModelInterface + 'static>(
        model: M,
        evaluator: Arc<dyn crate::metric::MetricEvaluator>,
    ) -> Self {
        Self::conversational(model)
            .metric_evaluator(evaluator)
            .escalation_mode(Self::preset_auto_continue())
    }

    /// Override the harness-loop tool registry (issue #4 seam).
    ///
    /// Use this to supply your own [`ToolRegistry`] implementation — e.g. a set
    /// of custom tools — on top of a preset like [`conversational`](Self::new):
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use spore_core::{HarnessBuilder, OllamaModelInterface, EmptyToolRegistry};
    /// let harness = HarnessBuilder::conversational(OllamaModelInterface::new("llama3.2"))
    ///     .tool_registry(Arc::new(EmptyToolRegistry))
    ///     .build();
    /// ```
    ///
    /// The registry's [`schemas`](ToolRegistry::schemas) are delivered to the
    /// model automatically each turn, and [`dispatch`](ToolRegistry::dispatch)
    /// is called when the model requests a tool.
    pub fn tool_registry(mut self, tool_registry: Arc<dyn ToolRegistry>) -> Self {
        self.tool_registry = tool_registry;
        self
    }

    /// Override the [`ContextManager`] that assembles per-turn context and drives
    /// compaction. `conversational` installs a `StandardContextManager` with
    /// `CompactionConfig::default()` — compaction fires at 80% of the *resolved*
    /// context window, which (issue #141) is the configured
    /// `CompactionConfig::context_length`, else the model's
    /// `provider().context_window`, else `DEFAULT_CONTEXT_LENGTH` (8K). Size the
    /// model's window in one call with the provider's `with_context_window`
    /// (SC-4); supply your own manager here (e.g. a lower `threshold`) to make
    /// compaction fire earlier.
    pub fn context_manager(mut self, context_manager: Arc<dyn ContextManager>) -> Self {
        self.context_manager = context_manager;
        self
    }

    /// Override the [`SandboxProvider`] — the only path tools have to the
    /// environment (filesystem, process exec).
    ///
    /// [`conversational`](Self::conversational) defaults to [`NullSandbox`],
    /// which denies environment access — fine for pure-compute tools, but
    /// catalogue file tools (`read_file` / `write_file` / `list_dir`) operate
    /// *through* the sandbox, so an agent that touches a real directory needs a
    /// workspace-scoped sandbox here:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use spore_core::{HarnessBuilder, OllamaModelInterface, StandardTools};
    /// # use spore_core::sandbox::{WorkspaceScopedSandbox, WorkspaceConfig};
    /// # fn demo(workspace: Arc<WorkspaceScopedSandbox>) {
    /// let harness = HarnessBuilder::conversational(OllamaModelInterface::new("llama3.2"))
    ///     .sandbox(workspace)
    ///     .tools(StandardTools::coding_set())
    ///     .build();
    /// # let _ = harness;
    /// # }
    /// ```
    pub fn sandbox(mut self, sandbox: Arc<dyn SandboxProvider>) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Set how a [`SandboxViolation`] surfaced by a tool is handled — recoverable
    /// feedback to the model (the default) or a hard halt. See
    /// [`SandboxViolationPolicy`].
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use spore_core::harness::{HarnessBuilder, SandboxViolationPolicy};
    /// # fn ex(b: HarnessBuilder) -> HarnessBuilder {
    /// b.sandbox_violation_policy(SandboxViolationPolicy::Halt)
    /// # }
    /// ```
    pub fn sandbox_violation_policy(mut self, policy: SandboxViolationPolicy) -> Self {
        self.sandbox_violation_policy = policy;
        self
    }

    /// Add a single [`StandardTool`](crate::tools::StandardTool) to the
    /// catalogue accumulated for this harness (issue #81, Q1/Q2).
    ///
    /// At [`build`](Self::build) the accumulated tools are folded into a
    /// populated [`StandardToolRegistry`](crate::tool_registry::StandardToolRegistry)
    /// (via [`drain_tools_into_registry`](Self::drain_tools_into_registry)) and
    /// the run loop bridges them per-run through
    /// [`RealToolRegistry`](crate::tool_registry::RealToolRegistry), so they run
    /// with sandbox + storage wired in — no manual bridging required.
    /// Registration applies LAST-WINS upsert: a later `.tool()` with the same
    /// name overrides an earlier one (e.g. a custom tool registered after a
    /// preset).
    ///
    /// To author a *custom* sandboxed tool, implement
    /// [`Tool`](crate::tool_registry::Tool) and wrap it with
    /// [`StandardTool::new`](crate::tools::StandardTool::new), then pass it here.
    /// Catalogue tools added this way take precedence over a registry supplied
    /// via [`tool_registry`](Self::tool_registry).
    pub fn tool(mut self, tool: crate::tools::StandardTool) -> Self {
        self.standard_tools.push(tool);
        self
    }

    /// Register a [`SkillCatalog`](crate::skills::SkillCatalog) (issue #115 /
    /// SC-26). This both (a) injects the catalog's manifest + active skill bodies
    /// into every turn's structural context (progressive disclosure) and (b)
    /// registers the `load_skill` tool, sharing the catalog's active set, so the
    /// model can activate a skill on demand. Replaces the architect-side
    /// skill-injecting context-manager shim.
    pub fn skills(mut self, catalog: Arc<SkillCatalog>) -> Self {
        self.standard_tools.push(catalog.load_skill_tool());
        self.skills = Some(catalog);
        self
    }

    /// Set an operating system prompt prepended to each turn's assembled context
    /// (issue #91).
    ///
    /// The standard compaction context manager renders no system prompt, so
    /// without this the model receives only the task as a user message and no
    /// guidance on how to behave. When set, the run loop inserts this text as a
    /// leading [`Role::System`](crate::model::Role) message each turn — but only
    /// when the assembled context does not already start with one, so a context
    /// manager that renders its own system prompt is preserved. `None` (the
    /// default) preserves today's behaviour.
    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Register a [`Guide`] (skill/playbook/domain knowledge) injected
    /// structurally into every turn's assembled context via the rich
    /// `ContextSources` seam (issue #115 / SC-26 / #9). Unlike an ad-hoc
    /// User-message prepend, the guide is rendered into the leading System block
    /// by the production context manager. Call repeatedly to register several;
    /// order is preserved.
    pub fn guide(mut self, guide: Guide) -> Self {
        self.guides.push(guide);
        self
    }

    /// Register several [`Guide`]s at once (issue #115 / SC-26). Appends to any
    /// already registered via [`guide`](Self::guide).
    pub fn guides(mut self, guides: impl IntoIterator<Item = Guide>) -> Self {
        self.guides.extend(guides);
        self
    }

    /// Wire a memory source (issue #160 / SC-26 follow-up). The harness queries
    /// the provider each turn and injects the relevant memories into the
    /// structural System block — alongside guides + skills via the rich
    /// `ContextSources` seam — with no consumer-side context-manager shim.
    ///
    /// Pass a [`MemoryConfig`] to control the query policy, e.g.
    /// `MemoryConfig::new(provider).min_relevance(0.6).max_items(5)`. Without an
    /// explicit query text the current task instruction is used, so retrieved
    /// memory tracks what the agent is working on. Not set (the default) leaves
    /// memory empty, byte-identical to pre-#160 behaviour.
    pub fn memory(mut self, memory: MemoryConfig) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Set the authoritative model sampling/decoding parameters for the whole
    /// run (issue #93).
    ///
    /// These params are authoritative: the harness replaces each turn's
    /// `Context.params` with this value UNCONDITIONALLY (builder params win)
    /// right before the request is built, so the configured params reach every
    /// agent turn that requests tools — the ReAct loop, the PlanExecute plan
    /// phase, the execute sub-loop, and the streaming path alike. (The internal
    /// compaction/summarization turn is intentionally left on defaults; it
    /// requests no tools, so decoding params are a no-op there.)
    ///
    /// Enabling [`structured_tool_calls`](crate::model::ModelParams::structured_tool_calls)
    /// trades interleaved reasoning for one schema-constrained tool call per
    /// turn — useful for small local models that otherwise emit malformed tool
    /// calls. See [`ModelParams::structured_tool_calls`](crate::model::ModelParams::structured_tool_calls)
    /// for the full behaviour contract.
    pub fn model_params(mut self, params: ModelParams) -> Self {
        self.model_params = params;
        self
    }

    /// Add many [`StandardTool`](crate::tools::StandardTool)s at once (e.g. a
    /// preset like [`StandardTools::coding_set`](crate::tools::StandardTools::coding_set)).
    /// Order is preserved, so last-wins upsert still applies across the batch.
    pub fn tools(mut self, tools: impl IntoIterator<Item = crate::tools::StandardTool>) -> Self {
        self.standard_tools.extend(tools);
        self
    }

    /// Register catalogue tools SCOPED to a single named toolset handle (Issue 2:
    /// per-node toolset scoping). Mirrors [`tools`](Self::tools), but instead of
    /// folding into the ONE global catalogue, these tools are folded into a
    /// per-`key` [`StandardToolRegistry`](crate::tool_registry::StandardToolRegistry)
    /// (last-wins upsert across calls and within the batch) and stored in
    /// [`HarnessConfig::toolset_catalogues`].
    ///
    /// At dispatch, a leaf whose `toolset` handle equals `key` resolves ONLY this
    /// catalogue (bridged per-run via
    /// [`RealToolRegistry`](crate::tool_registry::RealToolRegistry) with the run's
    /// sandbox/session/storage), so it cannot reach another node's tools — the
    /// union no longer leaks across nodes. A leaf with an EMPTY (`""`) handle, or
    /// a non-empty handle with no entry here, falls back to the global
    /// [`tools`](Self::tools) catalogue / `tool_registry` seam (back-compat).
    ///
    /// `key` should be a NON-EMPTY handle string; the empty handle is reserved
    /// for the global-catalogue fallback.
    pub fn toolset_tools(
        mut self,
        key: impl Into<String>,
        tools: impl IntoIterator<Item = crate::tools::StandardTool>,
    ) -> Self {
        self.toolset_tools
            .entry(key.into())
            .or_default()
            .extend(tools);
        self
    }

    /// Drain the accumulated catalogue tools into a populated
    /// [`StandardToolRegistry`](crate::tool_registry::StandardToolRegistry),
    /// registering each with last-wins upsert (issue #81, Q1/Q2). Consumes the
    /// accumulated set. The harness-loop [`tool_registry`](Self) seam is
    /// separate (it is the bridged dispatch trait); callers wire the returned
    /// registry into a [`RealToolRegistry`](crate::tool_registry::RealToolRegistry)
    /// per run. Returns an empty registry if no catalogue tools were added.
    pub fn drain_tools_into_registry(&mut self) -> Arc<crate::tool_registry::StandardToolRegistry> {
        use crate::tool_registry::ToolRegistry as _;
        let reg = crate::tool_registry::StandardToolRegistry::new();
        for t in self.standard_tools.drain(..) {
            // Upsert: a duplicate name overwrites (issue #81, Q1). Schema/name
            // validation errors are ignored here since catalogue schemas are
            // well-formed by construction.
            let _ = reg.register(t.implementation, t.schema);
        }
        Arc::new(reg)
    }

    /// Inject a [`StorageProvider`](crate::storage::StorageProvider) (issue #73).
    /// Mirrors [`observability`](Self::observability) — independent injection
    /// point. Defaults to an all-no-op provider so existing callers/tests
    /// compile and behave unchanged.
    pub fn storage(mut self, storage: Arc<crate::storage::StorageProvider>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Override the STABLE durable-storage project namespace (issue #142). When
    /// unset, [`build_config`](Self::build_config) derives it from
    /// `sandbox.workspace_root()` (decision 5), canonicalizing the path first.
    /// Set this explicitly when the workspace root is not a stable on-disk path
    /// (e.g. a fixture-replay tempdir, or to pin a known project across
    /// processes). See [`HarnessConfig::project_id`].
    pub fn project_id(mut self, project_id: crate::storage::ProjectId) -> Self {
        self.project_id = Some(project_id);
        self
    }

    /// Enable opt-in automatic conversation-history threading via the
    /// [`SessionStore`](crate::storage::SessionStore) (issue #102). Defaults to
    /// `false`, which preserves today's behaviour byte-for-byte (ZERO
    /// session-store I/O). When enabled, the harness auto-loads the prior
    /// session for the run's `session_id` (ReAct / SelfVerifying only; an
    /// explicit [`with_session_state`](HarnessRunOptions::with_session_state)
    /// always wins) and auto-persists the post-run state at the terminal seam.
    /// See [`HarnessConfig::auto_persist_sessions`] for the full contract.
    ///
    /// Pair with [`storage`](Self::storage) to wire a real (in-memory or
    /// filesystem) [`SessionStore`]; without one the default all-no-op store
    /// makes auto-load/persist no-ops.
    pub fn auto_persist_sessions(mut self, enabled: bool) -> Self {
        self.auto_persist_sessions = enabled;
        self
    }

    /// Set the conditional prompt-chunk provider (issue #79). Defaults to an
    /// empty [`InMemoryChunkProvider`](crate::prompt_assembly::InMemoryChunkProvider).
    pub fn chunk_provider(
        mut self,
        provider: Arc<dyn crate::prompt_assembly::ChunkProvider>,
    ) -> Self {
        self.chunk_provider = Some(provider);
        self
    }

    /// Convenience: register chunks inline without constructing a provider
    /// (issue #79). Resolves to an
    /// [`InMemoryChunkProvider`](crate::prompt_assembly::InMemoryChunkProvider).
    pub fn chunks(mut self, chunks: Vec<crate::prompt_assembly::PromptChunk>) -> Self {
        self.chunk_provider = Some(Arc::new(
            crate::prompt_assembly::InMemoryChunkProvider::new(chunks),
        ));
        self
    }

    /// Register a per-kind consult handler (issue #114). When a worker (run via
    /// [`SubagentTool`](crate::tools::SubagentTool)) pauses with a
    /// [`ConsultRequest`] whose `kind` matches `kind`, the orchestrator runs
    /// `handler` on the request deterministically (no orchestrator model turn),
    /// up to `budget` consults of this kind before applying `overflow`. Without
    /// any registered handler, consults degrade gracefully (R6): a standalone
    /// worker surfaces [`RunResult::Consult`] unchanged. Empty by default.
    pub fn consult_handler(
        mut self,
        kind: impl Into<String>,
        handler: Arc<dyn Harness>,
        budget: u32,
        overflow: ConsultOverflowPolicy,
    ) -> Self {
        self.consult_handlers.insert(
            kind.into(),
            ConsultHandlerEntry {
                handler,
                budget,
                overflow,
            },
        );
        self
    }

    /// Replace the whole [`ExecutionRegistry`] (issue #120). The registry
    /// resolves a task's serializable strategy handles
    /// (`AgentRef`/`ToolsetRef`/`SchemaRef`) and `StrategyRef::Custom` keys at
    /// run entry. Defaults to an empty registry.
    pub fn registry(mut self, registry: ExecutionRegistry) -> Self {
        self.registry = registry;
        self
    }

    /// Register a named agent in the [`ExecutionRegistry`] (issue #120,
    /// per-key convenience over [`registry`](Self::registry)).
    pub fn registry_agent(mut self, key: impl Into<String>, agent: Arc<dyn Agent>) -> Self {
        self.registry = std::mem::take(&mut self.registry)
            .into_builder()
            .agent(key, agent)
            .build();
        self
    }

    /// Register a named toolset in the [`ExecutionRegistry`] (issue #120).
    pub fn registry_toolset(
        mut self,
        key: impl Into<String>,
        toolset: Arc<dyn ToolRegistry>,
    ) -> Self {
        self.registry = std::mem::take(&mut self.registry)
            .into_builder()
            .toolset(key, toolset)
            .build();
        self
    }

    /// Register a named JSON schema in the [`ExecutionRegistry`] (issue #120).
    pub fn registry_schema(mut self, key: impl Into<String>, schema: serde_json::Value) -> Self {
        self.registry = std::mem::take(&mut self.registry)
            .into_builder()
            .schema(key, schema)
            .build();
        self
    }

    /// Register a named verifier in the [`ExecutionRegistry`] (issue #120).
    pub fn registry_verifier(
        mut self,
        key: impl Into<String>,
        verifier: Arc<dyn crate::verifier::Verifier>,
    ) -> Self {
        self.registry = std::mem::take(&mut self.registry)
            .into_builder()
            .verifier(key, verifier)
            .build();
        self
    }

    /// Register a custom strategy in the [`ExecutionRegistry`] under `key`
    /// (issue #120). Resolvable later via a `StrategyRef::Custom(key)`.
    pub fn register_strategy(
        mut self,
        key: impl Into<String>,
        strategy: Arc<dyn RunStrategy>,
    ) -> Self {
        self.registry.register_strategy(key, strategy);
        self
    }

    /// Select the HITL-vs-AFK escalation mode (issue #120, PRD goal #7).
    /// Defaults to [`EscalationMode::SurfaceToHuman`].
    pub fn escalation_mode(mut self, mode: EscalationMode) -> Self {
        self.escalation_mode = mode;
        self
    }

    /// Inject the default verification oracle for the `SelfVerifying` loop
    /// strategy (issue #61, #124). Folded into the [`ExecutionRegistry`] under
    /// the default empty key at build, so a `SelfVerifying` whose `evaluator`
    /// key is empty resolves to it. Without any resolvable verifier a
    /// `SelfVerifying` run halts with `SelfVerifyMisconfigured` (D4). Its
    /// [`max_iterations`](crate::verifier::Verifier::max_iterations) caps the
    /// build↔evaluate round-trips (D3).
    pub fn verifier(mut self, verifier: Arc<dyn crate::verifier::Verifier>) -> Self {
        self.verifier = Some(verifier);
        self
    }

    /// Inject a lifecycle hook chain (issue #69).
    pub fn hooks(mut self, hooks: Arc<dyn crate::hooks::HookChain>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Set the per-run cap on consecutive Stop-hook blocks (issue #69).
    /// Defaults to `8`.
    pub fn max_stop_blocks(mut self, max: u32) -> Self {
        self.max_stop_blocks = max;
        self
    }

    /// Set the consecutive-recoverable-tool-error breaker threshold `N`
    /// (issue #137). At `N` identical-argument recoverable errors for one tool
    /// the loop injects ONE corrective message; at `2 * N` it stops looping and
    /// resolves the node's [`BudgetExhaustedBehavior`] with
    /// [`HaltReason::ToolErrorLoop`] (the `2×` multiplier is fixed). Defaults to
    /// `3`.
    pub fn error_loop_threshold(mut self, n: u32) -> Self {
        self.error_loop_threshold = n;
        self
    }

    /// Turn ON output-schema delivery + enforcement for `ReactConfig` leaves
    /// carrying `output = Some(..)` (issue #139 — MIGRATION GATE). Defaults to
    /// `false` (OFF), which keeps existing replay fixtures byte-identical. See
    /// [`HarnessConfig::enforce_output_schemas`].
    pub fn enforce_output_schemas(mut self, enforce: bool) -> Self {
        self.enforce_output_schemas = enforce;
        self
    }

    /// Set the number of EXTRA terminal-validation retry turns `N` granted when
    /// [`enforce_output_schemas`](Self::enforce_output_schemas) is ON (issue
    /// #139). Total attempts = `1 + N`. Defaults to `2`. See
    /// [`HarnessConfig::output_schema_max_retries`].
    pub fn output_schema_max_retries(mut self, n: u32) -> Self {
        self.output_schema_max_retries = n;
        self
    }

    /// Set the outer-loop context-window reset cap for the `Ralph` loop strategy
    /// (issue #58, B3). Defaults to `3`.
    pub fn max_resets(mut self, max: u32) -> Self {
        self.max_resets = max;
        self
    }

    /// Inject a [`VcsProvider`] for the `Ralph` loop strategy (issue #58 v2).
    /// When set, Ralph's per-window reload phase also calls
    /// [`VcsProvider::log`] and injects a delimited "Recent VCS history:"
    /// section into the fresh context window. Defaults to `None`, which omits
    /// the git-log section and preserves v1 Ralph behavior byte-for-byte (the
    /// B4→None decision).
    pub fn vcs_provider(mut self, provider: Arc<dyn VcsProvider>) -> Self {
        self.vcs_provider = Some(provider);
        self
    }

    /// Inject the scoring oracle for the `HillClimbing` loop strategy (issue
    /// #60). Required for that strategy: without it a `HillClimbing` run halts
    /// with `HillClimbingMisconfigured` (no panic). The evaluator is called once
    /// at the iteration-0 baseline (no agent turn) and after every subsequent
    /// agent turn; each result is routed through
    /// [`should_keep`](crate::metric::should_keep). Defaults to `None`.
    pub fn metric_evaluator(mut self, evaluator: Arc<dyn crate::metric::MetricEvaluator>) -> Self {
        self.metric_evaluator = Some(evaluator);
        self
    }

    /// Inject a deterministic tool-call repair provider (e.g.
    /// [`StandardToolCallRepair`](crate::tool_call_repair::StandardToolCallRepair)).
    /// When set, recoverable tool-dispatch errors trigger an argument-coercion
    /// and re-dispatch attempt before falling back to enriched error feedback.
    /// Defaults to `None` (behaviour byte-identical to today).
    pub fn tool_call_repair(mut self, repair: Arc<dyn ToolCallRepair>) -> Self {
        self.tool_call_repair = Some(repair);
        self
    }

    /// Set the maximum number of repair-and-re-dispatch attempts per tool call.
    /// Defaults to `1`. Ignored when no repair provider is set.
    pub fn max_repair_attempts(mut self, attempts: u32) -> Self {
        self.max_repair_attempts = attempts;
        self
    }

    /// Set the PlanExecute plan-empty re-ask cap (see
    /// [`StrategyExecutor::plan_empty_max_retries`]). `0` (default) halts on the
    /// first empty plan; `n > 0` re-asks the planner up to `n` more times for the
    /// plan as message content before `EmptyPlan`.
    pub fn plan_empty_max_retries(mut self, retries: u32) -> Self {
        self.plan_empty_max_retries = retries;
        self
    }

    /// Enable the conservative tool-call nudge (see
    /// [`HarnessConfig::tool_call_nudge`]): nudge a model that narrates an action
    /// in prose without calling a tool, and force one more turn. Does NOT switch to
    /// prompt-based tool calling. `false` (default) is byte-identical.
    pub fn tool_call_nudge(mut self, on: bool) -> Self {
        self.tool_call_nudge = on;
        self
    }

    /// Enable deferred-tool exposure (see
    /// [`ToolSearchConfig`](crate::tool_registry::ToolSearchConfig)): advertise
    /// only the configured core tools + a `tool_search` meta-tool, and let the
    /// model load the rest of the catalogue on demand. Keeps the advertised
    /// description prose below a local model's native-tool-calling cliff at
    /// catalogue scale. `None` (default) advertises the full catalogue.
    pub fn tool_search(mut self, config: crate::tool_registry::ToolSearchConfig) -> Self {
        self.tool_search = Some(config);
        self
    }

    /// Inject a post-compaction verifier (issue #46). Defaults to
    /// [`KeyTermVerifier`].
    pub fn compaction_verifier(mut self, verifier: Arc<dyn CompactionVerifier>) -> Self {
        self.compaction_verifier = verifier;
        self
    }

    /// Set the maximum number of compaction-summary attempts before accepting
    /// a failing summary anyway (issue #46). Defaults to `2`.
    pub fn max_compaction_attempts(mut self, attempts: u32) -> Self {
        self.max_compaction_attempts = attempts;
        self
    }

    /// Inject a middleware chain.
    pub fn middleware(mut self, middleware: Arc<dyn MiddlewareChain>) -> Self {
        self.middleware = Some(middleware);
        self
    }

    /// Inject an observability provider. The harness loop emits real spans
    /// through it (turn spans, tool-call spans) and flushes on terminal outcomes.
    pub fn observability(mut self, observability: Arc<dyn ObservabilityProvider>) -> Self {
        self.observability = Some(observability);
        self
    }

    /// Convenience: construct and inject a durable-outbox observability provider
    /// rooted at `root` (typically the `.spore` directory). Honors the
    /// `SPORE_OTLP_ENDPOINT` env var for OTLP forwarding (issue #33).
    pub fn with_observability_outbox(self, root: impl Into<std::path::PathBuf>) -> Self {
        let provider = Arc::new(OutboxObservabilityProvider::new(OutboxConfig::new(root)));
        self.observability(provider)
    }

    /// Set the token→USD pricing table used to stamp `cost_usd` on turn spans.
    pub fn pricing(mut self, pricing: PricingTable) -> Self {
        self.pricing = pricing;
        self
    }

    /// Set the LLM-native content-capture config (issue #64). Defaults to OFF.
    /// Use [`ContentCaptureConfig::from_env`] to honor `SPORE_TRACE_CONTENT` /
    /// `SPORE_TRACE_CONTENT_MAX_LEN`.
    pub fn content_capture(mut self, content_capture: ContentCaptureConfig) -> Self {
        self.content_capture = content_capture;
        self
    }

    /// Assemble the [`HarnessConfig`] without wrapping it in a harness.
    pub fn build_config(mut self) -> HarnessConfig {
        // Fold catalogue tools accumulated via `.tool()` / `.tools()` into a
        // populated `StandardToolRegistry` (last-wins upsert). The run loop
        // bridges it per-run — `build()` can't, because the `ToolContext` is
        // keyed by the run's `SessionId`, unknown until `run()`.
        let catalogue_registry = if self.standard_tools.is_empty() {
            None
        } else {
            Some(self.drain_tools_into_registry())
        };
        // Issue 2: fold each per-toolset bucket into its own populated
        // `StandardToolRegistry` (last-wins upsert), keyed by the toolset handle.
        // Bridged per-run at dispatch — same as the global catalogue.
        let toolset_catalogues: std::collections::HashMap<
            String,
            Arc<crate::tool_registry::StandardToolRegistry>,
        > = {
            use crate::tool_registry::ToolRegistry as _;
            self.toolset_tools
                .drain()
                .map(|(key, tools)| {
                    let reg = crate::tool_registry::StandardToolRegistry::new();
                    for t in tools {
                        let _ = reg.register(t.implementation, t.schema);
                    }
                    (key, Arc::new(reg))
                })
                .collect()
        };
        // When catalogue tools are present and the caller wired no storage,
        // default to an in-memory provider (not the all-no-op default) so that
        // session-aware tools (todo_write, memory, task_list) actually persist
        // within the run. Pure tools (read_file/write_file via the sandbox) are
        // unaffected either way.
        let storage = self.storage.unwrap_or_else(|| {
            if catalogue_registry.is_some() || !toolset_catalogues.is_empty() {
                Arc::new(crate::storage::StorageProvider::single(Arc::new(
                    crate::storage::InMemoryStorageProvider::new(),
                )))
            } else {
                Arc::new(crate::storage::StorageProvider::no_op())
            }
        });
        // #124: the legacy single-collaborator fields are gone — fold the
        // builder's collaborators into the ExecutionRegistry under the DEFAULT
        // empty-string handle (`ReactConfig::per_loop` uses an empty `AgentRef`/
        // `ToolsetRef`; the default `SelfVerifyingConfig.evaluator` /
        // `HillClimbingConfig.evaluator` likewise use the empty key). Explicitly
        // registered handles (via `registry`) take precedence: only fill a key
        // the caller did not already wire.
        let mut reg_builder = self.registry.into_builder();
        reg_builder = reg_builder.fill_default_agent(self.agent);
        reg_builder = reg_builder.fill_default_toolset(self.tool_registry.clone());
        // Issue 2: a leaf carrying a non-empty `toolset` handle must RESOLVE
        // against the registry (`ExecutionRegistry::validate` runs `check_toolset`
        // at run entry). For every per-key catalogue wired via `.toolset_tools`,
        // ensure the registry has a presence entry under that handle so
        // `validate()` passes WITHOUT the caller manually registering a
        // placeholder toolset. The registry VALUE is never dispatched (dispatch
        // goes through `toolset_catalogues`), so a no-op `EmptyToolRegistry` is
        // sufficient; an explicitly-registered toolset under the same key wins.
        for key in toolset_catalogues.keys() {
            reg_builder = reg_builder.fill_toolset(key.clone(), Arc::new(EmptyToolRegistry));
        }
        if let Some(v) = self.verifier {
            reg_builder = reg_builder.fill_default_verifier(v);
        }
        if let Some(m) = self.metric_evaluator {
            reg_builder = reg_builder.fill_default_metric_evaluator(m);
        }
        let registry = reg_builder.build();
        // #142: derive the STABLE project namespace from `sandbox.workspace_root()`
        // (decision 5 — NOT process cwd) unless the caller pinned one explicitly.
        // The FS-touching `ProjectId::from_path` canonicalizes first; if that
        // fails (e.g. the workspace root does not yet exist on disk) fall back to
        // the PURE derivation over the workspace_root string so a `project_id` is
        // always present and deterministic.
        let project_id = self.project_id.unwrap_or_else(|| {
            let workspace_root = self.sandbox.workspace_root();
            crate::storage::ProjectId::from_path(workspace_root).unwrap_or_else(|_| {
                crate::storage::ProjectId::from_canonical_path(&workspace_root.to_string_lossy())
            })
        });
        HarnessConfig {
            tool_registry: self.tool_registry,
            sandbox: self.sandbox,
            sandbox_violation_policy: self.sandbox_violation_policy,
            context_manager: self.context_manager,
            termination_policy: self.termination_policy,
            middleware: self.middleware,
            observability: self.observability,
            compaction_verifier: self.compaction_verifier,
            max_compaction_attempts: self.max_compaction_attempts,
            pricing: self.pricing,
            content_capture: self.content_capture,
            tool_call_repair: self.tool_call_repair,
            max_repair_attempts: self.max_repair_attempts,
            plan_empty_max_retries: self.plan_empty_max_retries,
            tool_call_nudge: self.tool_call_nudge,
            max_stop_blocks: self.max_stop_blocks,
            error_loop_threshold: self.error_loop_threshold,
            enforce_output_schemas: self.enforce_output_schemas,
            output_schema_max_retries: self.output_schema_max_retries,
            hooks: self.hooks,
            storage,
            project_id,
            chunk_provider: self.chunk_provider.unwrap_or_else(|| {
                Arc::new(crate::prompt_assembly::InMemoryChunkProvider::empty())
            }),
            max_resets: self.max_resets,
            vcs_provider: self.vcs_provider,
            catalogue_registry,
            toolset_catalogues,
            system_prompt: self.system_prompt,
            guides: self.guides,
            skills: self.skills,
            memory: self.memory,
            model_params: self.model_params,
            auto_persist_sessions: self.auto_persist_sessions,
            prompt_tool_call_flag: self.prompt_tool_call_flag,
            consult_handlers: self.consult_handlers,
            registry,
            escalation_mode: self.escalation_mode,
            tool_search: self.tool_search,
        }
    }

    /// Assemble a ready-to-run [`StandardHarness`].
    pub fn build(self) -> StandardHarness {
        StandardHarness::new(self.build_config())
    }
}
