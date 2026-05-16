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
pub mod model;

pub use agent::{Agent, AgentError, AgentId, Context, ModelAgent, TurnResult};
pub use model::{
    enforce_budget, enforce_context_limit, Content, ContentBlock, Message, ModelError,
    ModelInterface, ModelParams, ModelRequest, ModelResponse, ModelStream, ProviderInfo,
    RecordedExchange, ReplayModelInterface, Role, StopReason, StreamEvent, TokenUsage, ToolCall,
    ToolResult, ToolSchema,
};
