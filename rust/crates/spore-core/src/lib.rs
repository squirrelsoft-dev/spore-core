//! spore-core: harness runtime library.
//!
//! Implements the language-agnostic harness specification in
//! `docs/harness-engineering-concepts.md`.
//!
//! Components (one issue per trait/struct):
//!   - #1  ModelInterface         ← implemented in [`model`]
//!   - #2  Agent (one turn)
//!   - #3  Harness runtime loop
//!   - #4  ToolRegistry
//!   - #5  Tool trait and base implementations
//!   - #6  SandboxProvider
//!   - #7  ContextManager
//!   - #8  MemoryProvider
//!   - #9  GuideRegistry
//!   - #10 SensorChain
//!   - #11 Middleware Chain
//!   - #12 ObservabilityProvider
//!   - #13 TerminationPolicy

pub mod agent;
pub mod anthropic;
pub mod cache_provider;
pub mod compaction_adapter;
pub mod context;
pub mod guide_registry;
pub mod harness;
pub mod hooks;
pub mod macros;
pub mod memory;
pub mod metric;
pub mod middleware;
pub mod model;
pub mod observability;
pub mod observability_outbox;
pub mod ollama;
pub mod openai;
pub mod plan;
pub mod prompt_assembly;
pub mod prompt_chunk_registry;
pub mod prompt_tool_call;
pub mod sandbox;
pub mod sensor;
pub mod storage;
pub mod tasklist;
pub mod termination;
pub mod tool_call_repair;
pub mod tool_registry;
pub mod tools;
pub mod verifier;

pub use agent::{Agent, AgentError, AgentId, Context as AgentContext, ModelAgent, TurnResult};
pub use anthropic::AnthropicModelInterface;
pub use cache_provider::{
    auto_detect as auto_detect_cache_provider, AnthropicCacheProvider, CacheAnnotationResult,
    CacheProvider, CacheStats, NullCacheProvider, OllamaCacheProvider, OpenAICacheProvider,
};
pub use compaction_adapter::{
    seed_rich_state, HarnessContextManagerExt, StandardCompactionAdapter, RICH_STATE_KEY,
};
pub use context::{
    BreakpointInfo, CacheBlockHits, CacheBlockStatus, CompactionConfig, CompactionPreserveHints,
    CompactionRequest, CompactionResult, CompactionVerificationResult, CompactionVerifier,
    ComposedPrompt, Context, ContextError, ContextManager, ContextMeta, ContextSources,
    KeyTermVerifier, PromptSegment, RenderedSystemPrompt, SegmentStability,
    SessionState as ContextSessionState, StandardContextManager,
};
pub use guide_registry::{
    Guide, GuideConflict, GuideId, GuideQuery, GuideRegistry, GuideRegistryError, GuideSource,
    GuideStatus, GuideType, GuideUsageRecord, ImprovementSignal, PendingReason, SessionOutcome,
    StandardGuideRegistry,
};
pub use harness::{
    AgentRef, AggregateUsage, BudgetExhaustedBehavior, BudgetLimitType, BudgetLimits, BudgetPolicy,
    BudgetSnapshot, BwrapProfile, ChildPausedState, CommandOutput, CompleteOnFinalResponse,
    ConsultHandlerEntry, ConsultOverflowPolicy, ConsultRequest, ConsultResponse,
    ContextManager as HarnessContextManager, EmptyToolRegistry, ExecutionContext, FileRef,
    GitVcsProvider, HaltReason, Harness, HarnessBuilder, HarnessConfig, HarnessError,
    HarnessRunOptions, HarnessSignal, HillClimbingConfig, HillClimbingDirection, HookPoint,
    HumanRequest, HumanResponse, IsolationMode, LoopStrategy, MiddlewareChain, MiddlewareDecision,
    ModelConfig, NetworkPolicy, NullSandbox, ObservabilityProvider, Operation, PausedState,
    PlanExecuteConfig, RalphConfig, ReactConfig, ReadOnlySandbox, RiskLevel, RunResult,
    RunStrategy, SandboxProvider, SandboxViolation, SchemaRef, SelfVerifyingConfig, SessionId,
    SessionState, StandardHarness, StrategyOutcome, StrategyRef, StreamEvent as HarnessStreamEvent,
    Task, TaskId, TerminationDecision, TerminationPolicy, ToolOutput,
    ToolRegistry as HarnessToolRegistry, ToolResult as HarnessToolResult, ToolsetRef,
    TruncatedOutput, VcsError, VcsLogArgs, VcsProvider,
};
pub use hooks::{
    CommandHook, FireOutcome, FunctionHook, Hook, HookChain, HookContext, HookDecision, HookError,
    HookEvent, HookSync, PlanArtifact, StandardHookChain, ToolCallSummary, TurnOutput,
};
pub use memory::{
    EpisodicMemory, MemoryError, MemoryId, MemoryItem, MemoryProvider, MemoryQuery, MemorySource,
    MemoryStatus, MergeStrategy, SemanticMemory, StandardMemoryProvider, Timestamp,
};
pub use metric::{
    should_keep, CommandMetricEvaluator, IterationStatus, JudgeModelConfig, LatencyEvaluator,
    LlmJudgeEvaluator, MetricError, MetricEvaluator, MetricResult, ResultsEntry,
    TestPassRateEvaluator,
};
pub use middleware::{
    HookContext as MiddlewareHookContext, HookPoint as MiddlewareHookPoint,
    LoopDetectionMiddleware, Middleware, MiddlewareChain as FullMiddlewareChain,
    MiddlewareDecision as FullMiddlewareDecision, MiddlewareError, PatchToolCallsMiddleware,
    PreCompletionChecklistMiddleware, StandardMiddlewareChain, TokenBudgetMiddleware,
    TracingMiddleware,
};
pub use model::{
    enforce_budget, enforce_context_limit, request_hash, Content, ContentBlock, Message,
    ModelError, ModelInterface, ModelParams, ModelRequest, ModelResponse, ModelStream,
    ProviderInfo, RecordedExchange, RecordingMode, RecordingModelInterface, ReplayMode,
    ReplayModelInterface, Role, StopReason, StreamEvent, TokenUsage, ToolCall, ToolResult,
    ToolSchema,
};
pub use observability::{
    truncate_field, ContentCaptureConfig, ContextOperation, ContextSpan, GenAiMessage, GenAiRole,
    InMemoryObservabilityProvider, MiddlewareSpan, ObservabilityError,
    ObservabilityProvider as FullObservabilityProvider, PatchSpan, PatchType, PricingTable,
    SensorSpan, SessionMetrics, Span, SpanBase, SpanId, SpanKind, SpanLevel, SpanStatus,
    ToolCallContent, ToolCallSpan, ToolResultContent, TurnSpan, TRUNCATION_MARKER,
};
pub use observability_outbox::{OutboxConfig, OutboxObservabilityProvider, TraceLine};
pub use ollama::OllamaModelInterface;
pub use openai::OpenAIModelInterface;
// `PlanArtifact` is re-exported via `hooks` above (it is the `OnPlanCreated`
// payload); `plan` reuses that type and adds the capture step + phase error.
pub use plan::{capture_plan_artifact, PlanPhaseError, PLAN_EXECUTE_EXTRAS_KEY};
// Prompt assembly engine (issue #79). `PromptChunk` and the chunk error type
// collide by concept with #24's `prompt_chunk_registry` exports, so the
// assembly-engine versions are re-exported under disambiguated aliases.
pub use prompt_assembly::{
    breakpoint_ids, chunks_to_segments, AssemblyBuckets, AssemblyContext, ChunkCondition,
    ChunkProvider, ChunkProviderError, CompositeChunkProvider, ContextSourcesBuilder,
    CustomCondition, EmbeddedChunkProvider, InMemoryChunkProvider,
    PromptChunk as AssemblyPromptChunk, StorageScope, ToolAffinity,
};
pub use prompt_chunk_registry::{
    standard_chunks, ApprovalPolicy, CacheBlock, ChunkError, ChunkId, ChunkSlot,
    ChunkValidationError, Mode, PromptChunk, PromptChunkRegistry, StandardPromptChunkRegistry,
};
pub use sandbox::{BuildError as SandboxBuildError, WorkspaceConfig, WorkspaceScopedSandbox};
pub use sensor::{
    Sensor, SensorChain, SensorConfig, SensorError, SensorId, SensorInput, SensorKind,
    SensorOutcome, SensorResult, SensorSignalFlag, SensorSignalThresholds, SensorStats,
    SensorTrigger, StandardSensorChain,
};
pub use storage::{
    parse_otlp_endpoints, CompositeStorageProvider, FileSystemStorageProvider,
    InMemoryStorageProvider, MemoryEntry, MemoryStore, NoOpStorageProvider, ObservabilityStore,
    RunStore, ScopedMemoryRouter, SessionStore, StorageError, StorageProvider, WorkspaceId,
};
pub use tasklist::{
    plan_artifact_to_task_list, Task as TaskListTask, TaskList, TaskListError, TaskStatus,
    TASK_LIST_EXTRAS_KEY,
};
pub use tools::{
    AbortTool, AskUserQuestionTool, EditFileTool, EnterPlanModeTool, ExitPlanModeTool, GrepTool,
    SearchMethod, SendMessageTool, StandardTool, StandardTools, TaskListTool, TodoWriteTool,
    WebFetchTool, WebSearchConfig, WebSearchConfigError, WebSearchTool, TODO_STORE_KEY,
};
// `CompletionCheck` is `#[deprecated]` (issue #69) but still publicly
// re-exported for backward compatibility; external callers see the deprecation.
pub use prompt_tool_call::{
    inject_tool_prompt, parse_prose_response, AdaptiveToolCallModelInterface,
    PromptBasedToolCallModelInterface,
};
#[allow(deprecated)]
pub use termination::{
    check_budget_default, AlwaysComplete, BudgetValue, CompletionCheck, FeatureListCheck,
    FixedCompletionCheck, NullCompletionCheck, QuestionAnsweredCheck, SessionStateSnapshot,
    SqlResultCheck, StandardTerminationPolicy, TerminationDecision as FullTerminationDecision,
    TerminationFailureReason, TerminationInput, TerminationPolicy as FullTerminationPolicy,
    TestSuiteCheck,
};
pub use tool_call_repair::{
    coerce_tool_args, detect_prose_response, StandardToolCallRepair, ToolCallFailure,
    ToolCallRepair,
};
pub use tool_registry::{
    DispatchError, RealToolRegistry, RegistrationError, StandardToolRegistry, TaskPhase, Tool,
    ToolAnnotations, ToolContext, ToolRegistry, ToolSchema as RegisteredToolSchema, ToolSet,
};
pub use verifier::{
    CompositeVerifier, EvaluatorResponseVerifier, TestSuiteVerifier, Verifier, VerifierInput,
    VerifierVerdict,
};
