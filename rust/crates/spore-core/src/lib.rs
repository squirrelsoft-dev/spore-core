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
pub mod context;
pub mod guide_registry;
pub mod harness;
pub mod memory;
pub mod middleware;
pub mod model;
pub mod observability;
pub mod sandbox;
pub mod sensor;
pub mod tool_registry;
pub mod tools;

pub use agent::{Agent, AgentError, AgentId, Context as AgentContext, ModelAgent, TurnResult};
pub use context::{
    BreakpointInfo, CacheBlockStatus, CacheProvider, CacheStats, CompactionConfig,
    CompactionPreserveHints, CompactionRequest, CompactionResult, ComposedPrompt, Context,
    ContextError, ContextManager, ContextMeta, ContextSources, NullCacheProvider, PromptSegment,
    RenderedSystemPrompt, SegmentStability, SessionState as ContextSessionState,
    StandardContextManager,
};
pub use guide_registry::{
    Guide, GuideConflict, GuideId, GuideQuery, GuideRegistry, GuideRegistryError, GuideSource,
    GuideStatus, GuideType, GuideUsageRecord, ImprovementSignal, PendingReason, SessionOutcome,
    StandardGuideRegistry,
};
pub use harness::{
    AggregateUsage, BudgetLimitType, BudgetLimits, BudgetSnapshot, BwrapProfile, ChildPausedState,
    CommandOutput, FileRef, HaltReason, Harness, HarnessConfig, HarnessError, HarnessRunOptions,
    HookPoint, HumanRequest, HumanResponse, IsolationMode, LoopStrategy, MiddlewareChain,
    MiddlewareDecision, ModelConfig, NetworkPolicy, ObservabilityProvider, Operation,
    OptimizationDirection, PausedState, RiskLevel, RunResult, SandboxProvider, SandboxViolation,
    SessionId, SessionState, StandardHarness, StreamEvent as HarnessStreamEvent, Task, TaskId,
    TerminationDecision, TerminationPolicy, ToolOutput, ToolRegistry as HarnessToolRegistry,
    ToolResult as HarnessToolResult, TruncatedOutput,
};
pub use memory::{
    EpisodicMemory, MemoryError, MemoryId, MemoryItem, MemoryProvider, MemoryQuery, MemorySource,
    MemoryStatus, MergeStrategy, SemanticMemory, StandardMemoryProvider, Timestamp,
};
pub use middleware::{
    HookContext as MiddlewareHookContext, HookPoint as MiddlewareHookPoint,
    LoopDetectionMiddleware, Middleware, MiddlewareChain as FullMiddlewareChain,
    MiddlewareDecision as FullMiddlewareDecision, MiddlewareError, PatchToolCallsMiddleware,
    PreCompletionChecklistMiddleware, StandardMiddlewareChain, TokenBudgetMiddleware,
    TracingMiddleware,
};
pub use model::{
    enforce_budget, enforce_context_limit, Content, ContentBlock, Message, ModelError,
    ModelInterface, ModelParams, ModelRequest, ModelResponse, ModelStream, ProviderInfo,
    RecordedExchange, ReplayModelInterface, Role, StopReason, StreamEvent, TokenUsage, ToolCall,
    ToolResult, ToolSchema,
};
pub use observability::{
    ContextOperation, ContextSpan, InMemoryObservabilityProvider, MiddlewareSpan,
    ObservabilityProvider as FullObservabilityProvider, PricingTable, SensorSpan, SessionMetrics,
    Span, SpanBase, SpanId, SpanKind, SpanStatus, ToolCallSpan, TurnSpan,
};
pub use sandbox::{BuildError as SandboxBuildError, WorkspaceConfig, WorkspaceScopedSandbox};
pub use sensor::{
    Sensor, SensorChain, SensorConfig, SensorError, SensorId, SensorInput, SensorKind,
    SensorOutcome, SensorResult, SensorSignalFlag, SensorSignalThresholds, SensorStats,
    SensorTrigger, StandardSensorChain,
};
pub use tool_registry::{
    DispatchError, RegistrationError, StandardToolRegistry, TaskPhase, Tool, ToolAnnotations,
    ToolRegistry, ToolSchema as RegisteredToolSchema, ToolSet,
};
