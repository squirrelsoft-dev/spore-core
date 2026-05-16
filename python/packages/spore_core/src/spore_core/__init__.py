"""spore_core: harness runtime library.

Implements the language-agnostic spec in
``docs/harness-engineering-concepts.md``.

Components (one issue per trait/class):

* #1  ModelInterface
* #2  Agent (one turn)
* #3  Harness runtime loop
* #4  ToolRegistry
* #5  Tool protocol and base implementations
* #6  SandboxProvider
* #7  ContextManager
* #8  MemoryProvider
* #9  GuideRegistry
* #10 SensorChain
* #11 Middleware Chain
* #12 ObservabilityProvider
* #13 TerminationPolicy
"""

from .agent import (
    Agent,
    AgentError,
    AgentErrorEmpty,
    AgentErrorException,
    AgentErrorMalformed,
    AgentErrorModel,
    AgentId,
    Context,
    EmptyResponseError,
    FinalResponse,
    MalformedToolCallError,
    MockAgent,
    ModelAgent,
    ModelErrorPayload,
    ToolCallRequested,
    TurnError,
    TurnResult,
)
from .errors import AlwaysHaltError, SporeError
from .model import (
    BudgetExceeded,
    Content,
    ContentBlock,
    ContentBlockDelta,
    ContentBlockStop,
    ContextLimitExceeded,
    ImageContent,
    Message,
    MessageStart,
    MessageStop,
    MockModelInterface,
    ModelError,
    ModelInterface,
    ModelParams,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ProviderInfo,
    RateLimited,
    RecordedExchange,
    ReplayModelInterface,
    Role,
    StopReason,
    StreamEvent,
    TextBlock,
    TextContent,
    ThinkingBlock,
    ThinkingDelta,
    TimeoutError,
    TokenUsage,
    ToolCall,
    ToolCallContent,
    ToolResult,
    ToolResultContent,
    ToolSchema,
    ToolUseBlock,
    ToolUseDelta,
    enforce_budget,
    enforce_context_limit,
)

__all__ = [
    "Agent",
    "AgentError",
    "AgentErrorEmpty",
    "AgentErrorException",
    "AgentErrorMalformed",
    "AgentErrorModel",
    "AgentId",
    "AlwaysHaltError",
    "BudgetExceeded",
    "Context",
    "EmptyResponseError",
    "FinalResponse",
    "MalformedToolCallError",
    "MockAgent",
    "ModelAgent",
    "ModelErrorPayload",
    "ToolCallRequested",
    "TurnError",
    "TurnResult",
    "Content",
    "ContentBlock",
    "ContentBlockDelta",
    "ContentBlockStop",
    "ContextLimitExceeded",
    "ImageContent",
    "Message",
    "MessageStart",
    "MessageStop",
    "MockModelInterface",
    "ModelError",
    "ModelInterface",
    "ModelParams",
    "ModelRequest",
    "ModelResponse",
    "ProviderError",
    "ProviderInfo",
    "RateLimited",
    "RecordedExchange",
    "ReplayModelInterface",
    "Role",
    "SporeError",
    "StopReason",
    "StreamEvent",
    "TextBlock",
    "TextContent",
    "ThinkingBlock",
    "ThinkingDelta",
    "TimeoutError",
    "TokenUsage",
    "ToolCall",
    "ToolCallContent",
    "ToolResult",
    "ToolResultContent",
    "ToolSchema",
    "ToolUseBlock",
    "ToolUseDelta",
    "enforce_budget",
    "enforce_context_limit",
]
