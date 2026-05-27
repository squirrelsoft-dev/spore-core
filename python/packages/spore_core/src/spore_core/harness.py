"""Harness — the agent runtime loop (issue #3).

Mirrors the Rust reference at ``rust/crates/spore-core/src/harness.rs``.
The harness owns execution lifecycle and wires all components together.
It is stateless between :meth:`Harness.run` calls; everything the harness
needs comes in via :class:`HarnessRunOptions` or :class:`PausedState`, and
everything it produces goes out via :class:`RunResult`.

What this component does:

* Assemble context (via :class:`ContextManager`) before each turn
* Call the agent for one turn
* Dispatch tool calls to :class:`ToolRegistry`
* Evaluate :class:`TerminationPolicy` after each turn
* Fire middleware lifecycle hooks
* Track iterations, token spend, elapsed time
* Pause and resume for human-in-the-loop interactions

What this component does NOT do:

* Touch the filesystem, execute commands, or call the model directly
* Persist :class:`PausedState` — the caller owns persistence
* Implement individual tools, sandbox policy, or context assembly

Rules enforced here:

1. Harness owns the loop — the agent only executes one turn at a time.
2. Termination is evaluated against external state via
   :class:`TerminationPolicy`.
3. Any budget overrun terminates the loop with an explicit
   :class:`HaltReason`.
4. A turn that yields neither a tool call nor a final response is an error
   (surfaced via :class:`AgentError`).
5. All components are injected at construction; the harness never builds
   them itself.
6. Stateless between pause and resume — the caller owns
   :class:`PausedState`.
7. :class:`WaitingForHuman` returns immediately; no internal timeout.
8. ``approved_results`` prevents double-execution on resume.
9. Subagents cannot spawn their own subagents — :class:`ChildPausedState`
   has no ``child_state`` field (depth-1 enforcement).

Many of the sibling component traits (``ToolRegistry``, ``SandboxProvider``,
``ContextManager``, …) ship in their own component issues (#4–#13). Until
those land, this module defines minimal forward-declared :class:`Protocol`
stubs of the trait surface the loop actually consumes. When a sibling
issue lands its canonical definition will replace the stub here.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Literal, NewType, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .agent import (
    Agent,
    AgentError,
    Context,
    FinalResponse,
    ToolCallRequested,
    TurnError,
    TurnResult,
)
from .model import (
    Message,
    ModelParams,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolSchema,
)

# ============================================================================
# Identity newtypes
# ============================================================================

SessionId = NewType("SessionId", str)
TaskId = NewType("TaskId", str)

_id_counter: int = 0


def _random_id() -> str:
    global _id_counter
    _id_counter += 1
    return f"{_id_counter:016x}"


def new_session_id(s: str | None = None) -> SessionId:
    return SessionId(s if s is not None else f"sess-{_random_id()}")


def new_task_id(s: str | None = None) -> TaskId:
    return TaskId(s if s is not None else f"task-{_random_id()}")


# ============================================================================
# Pydantic base
# ============================================================================


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# Budget tracking
# ============================================================================


class BudgetLimits(_Model):
    max_turns: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    # Wire format: seconds (int) to match Rust's `duration_secs_opt`.
    max_wall_time: int | None = None
    max_cost_usd: float | None = None


BudgetLimitTypeT = Literal["turns", "input_tokens", "output_tokens", "wall_time", "cost_usd"]


class BudgetSnapshot(_Model):
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time: int | None = None
    cost_usd: float = 0.0


class AggregateUsage(_Model):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def add_turn(self, u: TokenUsage) -> None:
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cache_read_tokens += u.cache_read_tokens or 0
        self.cache_write_tokens += u.cache_write_tokens or 0


# ============================================================================
# Task + loop strategy (tagged union on ``kind``)
# ============================================================================


OptimizationDirection = Literal["minimize", "maximize"]


class ModelConfig(_Model):
    provider: str
    model_id: str


class LoopStrategyReAct(_Model):
    kind: Literal["re_act"] = "re_act"
    max_iterations: int


class LoopStrategyPlanExecute(_Model):
    kind: Literal["plan_execute"] = "plan_execute"
    plan_model: ModelConfig | None = None


class LoopStrategyRalph(_Model):
    kind: Literal["ralph"] = "ralph"


class LoopStrategySelfVerifying(_Model):
    kind: Literal["self_verifying"] = "self_verifying"


class LoopStrategyHillClimbing(_Model):
    kind: Literal["hill_climbing"] = "hill_climbing"
    direction: OptimizationDirection
    max_stagnation: int | None = None
    revert_on_no_improvement: bool = False
    min_improvement_delta: float | None = None


LoopStrategy = Annotated[
    LoopStrategyReAct
    | LoopStrategyPlanExecute
    | LoopStrategyRalph
    | LoopStrategySelfVerifying
    | LoopStrategyHillClimbing,
    Field(discriminator="kind"),
]


class Task(_Model):
    id: TaskId
    instruction: str
    session_id: SessionId
    budget: BudgetLimits = Field(default_factory=BudgetLimits)
    loop_strategy: LoopStrategy

    @classmethod
    def new(
        cls,
        instruction: str,
        session_id: SessionId,
        loop_strategy: LoopStrategy,
        *,
        budget: BudgetLimits | None = None,
    ) -> Task:
        return cls(
            id=new_task_id(),
            instruction=instruction,
            session_id=session_id,
            budget=budget or BudgetLimits(),
            loop_strategy=loop_strategy,
        )


# ============================================================================
# Stream events
# ============================================================================


class StreamTurnStart(_Model):
    kind: Literal["turn_start"] = "turn_start"
    turn: int


class StreamTurnEnd(_Model):
    kind: Literal["turn_end"] = "turn_end"
    turn: int


class StreamToolCall(_Model):
    kind: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str


class StreamToolResult(_Model):
    kind: Literal["tool_result"] = "tool_result"
    call_id: str
    is_error: bool


class StreamFinalResponse(_Model):
    kind: Literal["final_response"] = "final_response"
    content: str


class StreamBudgetWarning(_Model):
    kind: Literal["budget_warning"] = "budget_warning"
    limit_type: BudgetLimitTypeT


HarnessStreamEvent = Annotated[
    StreamTurnStart
    | StreamTurnEnd
    | StreamToolCall
    | StreamToolResult
    | StreamFinalResponse
    | StreamBudgetWarning,
    Field(discriminator="kind"),
]

StreamSink = Callable[[HarnessStreamEvent], None]


# ============================================================================
# Forward-declared sibling types
# ============================================================================


class SessionState(_Model):
    """Opaque session state round-tripped across pause/resume.

    The harness does not interpret its contents; #7 (ContextManager) and
    #8 (MemoryProvider) own the schema.
    """

    messages: list[Message] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)


# ----- SandboxViolation (issue #6) -----------------------------------------


class SandboxPathEscape(_Model):
    kind: Literal["path_escape"] = "path_escape"
    path: str


class SandboxNetworkViolation(_Model):
    kind: Literal["network_violation"] = "network_violation"
    host: str


class SandboxPathDenied(_Model):
    kind: Literal["path_denied"] = "path_denied"
    path: str
    matched_rule: str = ""


class SandboxReadOnlyViolation(_Model):
    kind: Literal["read_only_violation"] = "read_only_violation"
    path: str


class SandboxExtensionDenied(_Model):
    kind: Literal["extension_denied"] = "extension_denied"
    path: str
    extension: str


class SandboxFileSizeExceeded(_Model):
    kind: Literal["file_size_exceeded"] = "file_size_exceeded"
    path: str
    size: int
    limit: int


class SandboxDisallowedCommand(_Model):
    kind: Literal["disallowed_command"] = "disallowed_command"
    command: str


SandboxViolation = Annotated[
    SandboxPathEscape
    | SandboxNetworkViolation
    | SandboxPathDenied
    | SandboxReadOnlyViolation
    | SandboxExtensionDenied
    | SandboxFileSizeExceeded
    | SandboxDisallowedCommand,
    Field(discriminator="kind"),
]


def sandbox_violation_is_always_halt(v: SandboxViolation) -> bool:
    """Layer-1 violations cannot be overridden — they always halt."""
    return isinstance(v, SandboxPathEscape | SandboxNetworkViolation)


# ----- HookPoint (issue #11) ----------------------------------------------
# Re-exported from :mod:`spore_core.middleware`. We use a string alias here
# because :class:`HaltReasonMiddlewareHalt` and :class:`ScriptedMiddleware`
# round-trip these values as strings on the wire.

HookPoint = Literal[
    "before_session",
    "before_turn",
    "before_tool",
    "after_tool",
    "before_completion",
    "after_session",
]


# ----- TerminationDecision (issue #13) ------------------------------------


class TerminationContinue(_Model):
    kind: Literal["continue"] = "continue"


class TerminationHalt(_Model):
    kind: Literal["halt"] = "halt"
    reason: str


TerminationDecision = Annotated[
    TerminationContinue | TerminationHalt,
    Field(discriminator="kind"),
]


# ----- Human-in-the-loop --------------------------------------------------


RiskLevel = Literal["low", "medium", "high", "critical"]


class HumanRequestToolApproval(_Model):
    kind: Literal["tool_approval"] = "tool_approval"
    calls: list[ToolCall]
    risk_level: RiskLevel


class HumanRequestClarification(_Model):
    kind: Literal["clarification"] = "clarification"
    question: str


class HumanRequestReview(_Model):
    kind: Literal["review"] = "review"
    content: str


HumanRequest = Annotated[
    HumanRequestToolApproval | HumanRequestClarification | HumanRequestReview,
    Field(discriminator="kind"),
]


class HumanResponseAllow(_Model):
    kind: Literal["allow"] = "allow"


class HumanResponseAllowWithModification(_Model):
    kind: Literal["allow_with_modification"] = "allow_with_modification"
    calls: list[ToolCall]


class HumanResponseDeny(_Model):
    kind: Literal["deny"] = "deny"
    reason: str


class HumanResponseHalt(_Model):
    kind: Literal["halt"] = "halt"


class HumanResponseAnswer(_Model):
    kind: Literal["answer"] = "answer"
    text: str


class HumanResponseApproveWithFeedback(_Model):
    kind: Literal["approve_with_feedback"] = "approve_with_feedback"
    feedback: str


class HumanResponseReject(_Model):
    kind: Literal["reject"] = "reject"
    reason: str


HumanResponse = Annotated[
    HumanResponseAllow
    | HumanResponseAllowWithModification
    | HumanResponseDeny
    | HumanResponseHalt
    | HumanResponseAnswer
    | HumanResponseApproveWithFeedback
    | HumanResponseReject,
    Field(discriminator="kind"),
]


# ----- ToolOutput / ToolResult (issue #4/#5) ------------------------------


class ToolOutputSuccess(_Model):
    kind: Literal["success"] = "success"
    content: str
    truncated: bool = False


class ToolOutputError(_Model):
    kind: Literal["error"] = "error"
    message: str
    recoverable: bool = True


class ToolOutputWaitingForHuman(_Model):
    kind: Literal["waiting_for_human"] = "waiting_for_human"
    child_state: ChildPausedState
    request: HumanRequest


ToolOutput = Annotated[
    ToolOutputSuccess | ToolOutputError | ToolOutputWaitingForHuman,
    Field(discriminator="kind"),
]


class HarnessToolResult(_Model):
    """Result of dispatching a tool call (harness-side).

    Distinct from :class:`spore_core.model.ToolResult` which is the wire
    content block appended to messages.
    """

    call_id: str
    output: ToolOutput


# ----- MiddlewareDecision (issue #11) -------------------------------------
# The canonical types live in :mod:`spore_core.middleware`. They are
# imported below (at the bottom of this module, after PausedState resolves
# its forward references) and re-exported for ergonomic
# ``from spore_core.harness import MiddlewareHalt`` style imports used by
# the existing harness tests.


# ----- Component protocols (forward declarations) -------------------------


@runtime_checkable
class ToolRegistry(Protocol):
    """Issue #4 — dispatches tool calls."""

    async def dispatch(self, call: ToolCall) -> ToolOutput: ...

    def is_always_halt(self, tool_name: str) -> bool: ...

    def schemas(self) -> list[ToolSchema]: ...


class CommandOutput(_Model):
    """Output of a subprocess executed through the sandbox."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    truncated: bool = False


class FileRef(_Model):
    """Reference to a file holding offloaded tool output."""

    path: str
    byte_len: int


class TruncatedOutput(_Model):
    """Head+tail-truncated output. ``full_ref`` is set when the sandbox
    offloads the original content to a backing file."""

    content: str
    truncated: bool = False
    full_ref: FileRef | None = None
    original_size: int = 0


# ----- Operation / IsolationMode / NetworkPolicy (issue #6) ---------------


Operation = Literal["read", "write", "execute"]


class BwrapProfile(_Model):
    """Bubblewrap profile descriptor. Opaque placeholder in v1."""


class NetworkPolicyNone(_Model):
    kind: Literal["none"] = "none"


class NetworkPolicyAllowlist(_Model):
    kind: Literal["allowlist"] = "allowlist"
    hosts: list[str] = Field(default_factory=list)


class NetworkPolicyFull(_Model):
    kind: Literal["full"] = "full"


NetworkPolicy = Annotated[
    NetworkPolicyNone | NetworkPolicyAllowlist | NetworkPolicyFull,
    Field(discriminator="kind"),
]


class IsolationModeNone(_Model):
    kind: Literal["none"] = "none"


class IsolationModeWorkspaceScoped(_Model):
    kind: Literal["workspace_scoped"] = "workspace_scoped"


class IsolationModeBubblewrap(_Model):
    kind: Literal["bubblewrap"] = "bubblewrap"
    profile: BwrapProfile = Field(default_factory=BwrapProfile)


class IsolationModeDocker(_Model):
    kind: Literal["docker"] = "docker"
    image: str
    network: NetworkPolicy


IsolationMode = Annotated[
    IsolationModeNone
    | IsolationModeWorkspaceScoped
    | IsolationModeBubblewrap
    | IsolationModeDocker,
    Field(discriminator="kind"),
]


class WorkspaceConfig(_Model):
    """Configuration injected at harness construction time.

    Mirrors the Rust ``WorkspaceConfig`` struct. ``allowed_paths`` /
    ``denied_paths`` are relative-or-absolute filesystem paths;
    :class:`WorkspaceScopedSandbox` normalizes them at construction time.
    """

    root: Path
    allowed_paths: list[Path] = Field(default_factory=list)
    denied_paths: list[Path] = Field(default_factory=list)
    allowed_extensions: list[str] | None = None
    denied_extensions: list[str] = Field(default_factory=list)
    read_only: bool = False
    max_file_size: int = 0


@runtime_checkable
class SandboxProvider(Protocol):
    """Issue #6 — validates tool calls against sandbox policy.

    ``resolve_path`` takes an :data:`Operation` so the sandbox can apply
    read-only policy and handle missing-file Write/Execute canonicalization.
    """

    async def validate(self, call: ToolCall) -> SandboxViolation | None: ...

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput: ...

    async def handle_large_output(
        self,
        content: str,
        call_id: str,
        head_tokens: int,
        tail_tokens: int,
    ) -> TruncatedOutput: ...

    async def resolve_path(self, path: str, operation: Operation = "read") -> Path: ...

    def isolation_mode(self) -> IsolationMode: ...

    def workspace_root(self) -> Path: ...


class BaseSandboxProvider:
    """Concrete base class providing default implementations of
    ``execute_command``, ``handle_large_output``, and ``resolve_path``.

    **NOT production-safe** — these defaults spawn processes directly with
    no sandboxing, return paths as-is, and truncate inline without
    offloading. Issue #6 will replace this with a real sandbox.

    Subclasses must still implement :meth:`validate`.
    """

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        try:
            proc = await asyncio.create_subprocess_exec(
                command,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(working_dir) if working_dir is not None else None,
            )
        except (FileNotFoundError, OSError) as e:
            return CommandOutput(
                stdout="",
                stderr=f"spawn failed: {e}",
                exit_code=-1,
                timed_out=False,
            )
        try:
            if timeout is not None:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            else:
                stdout_b, stderr_b = await proc.communicate()
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except (ProcessLookupError, asyncio.CancelledError):
                pass
            secs = int(timeout) if timeout is not None else 0
            return CommandOutput(
                stdout="",
                stderr=f"command timed out after {secs}s",
                exit_code=-1,
                timed_out=True,
            )
        return CommandOutput(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            exit_code=proc.returncode if proc.returncode is not None else -1,
            timed_out=False,
        )

    async def handle_large_output(
        self,
        content: str,
        call_id: str,
        head_tokens: int,
        tail_tokens: int,
    ) -> TruncatedOutput:
        head_chars = max(0, head_tokens) * 4
        tail_chars = max(0, tail_tokens) * 4
        total = len(content)
        original = len(content.encode("utf-8"))
        if total <= head_chars + tail_chars:
            return TruncatedOutput(
                content=content, truncated=False, full_ref=None, original_size=original
            )
        head = content[:head_chars]
        tail = content[total - tail_chars :]
        elided = total - head_chars - tail_chars
        summary = f"{head}\n... [{elided} chars elided] ...\n{tail}"
        return TruncatedOutput(
            content=summary, truncated=True, full_ref=None, original_size=original
        )

    async def resolve_path(self, path: str, operation: Operation = "read") -> Path:
        return Path(path)

    def isolation_mode(self) -> IsolationMode:
        return IsolationModeNone()

    def workspace_root(self) -> Path:
        return Path("/")


@runtime_checkable
class ContextManager(Protocol):
    """Issue #7 — assembles per-turn context."""

    async def assemble(self, session: SessionState, task: Task) -> Context: ...

    async def append_tool_result(
        self, session: SessionState, result: HarnessToolResult
    ) -> None: ...

    async def append_user_message(self, session: SessionState, text: str) -> None: ...

    def should_compact(self, session: SessionState) -> bool: ...


@runtime_checkable
class TerminationPolicy(Protocol):
    """Issue #13 — evaluated after each turn."""

    async def evaluate(
        self, session: SessionState, budget_used: BudgetSnapshot
    ) -> TerminationDecision: ...


@runtime_checkable
class HarnessMiddlewareChain(Protocol):
    """Simplified middleware-chain Protocol consumed by
    :class:`StandardHarness`.

    The canonical, spec-rich :class:`spore_core.middleware.MiddlewareChain`
    (issue #11) ships with a per-hook ``fire_before_*`` / ``fire_after_*``
    surface. The harness loop here keeps a thinner ``fire(hook, session)``
    interface so existing ReAct unit tests and the
    :class:`ScriptedMiddleware` test double keep working without an
    adapter. Adapters bridging the two surfaces will land alongside the
    harness-middleware integration test in a future commit.
    """

    async def fire(self, hook: HookPoint, session: SessionState) -> MiddlewareDecision: ...


# Issue #12 — ``ObservabilityProvider`` is no longer a per-turn no-op stub
# here. The canonical Protocol lives in :mod:`spore_core.observability`; the
# harness loop emits real spans (turn spans, child tool-call spans) through it
# and flushes on terminal outcomes. It is imported at the bottom of this module
# (after this module's own types are defined) to avoid a circular import, since
# :mod:`spore_core.observability` imports :class:`SessionId` / :class:`TaskId`
# from here. See the deferred import block alongside the middleware import.


# ============================================================================
# PausedState / ChildPausedState
# ============================================================================


class PausedState(_Model):
    session_id: SessionId
    task_id: TaskId
    turn_number: int
    session_state: SessionState
    pending_tool_calls: list[ToolCall] = Field(default_factory=list)
    approved_results: list[HarnessToolResult] = Field(default_factory=list)
    human_request: HumanRequest
    task: Task
    budget_used: BudgetSnapshot
    child_state: ChildPausedState | None = None


class ChildPausedState(_Model):
    """Child paused state. **Deliberately has no ``child_state`` field** —
    subagents cannot spawn their own subagents (spec depth-1 rule)."""

    session_id: SessionId
    task_id: TaskId
    turn_number: int
    session_state: SessionState
    pending_tool_calls: list[ToolCall] = Field(default_factory=list)
    approved_results: list[HarnessToolResult] = Field(default_factory=list)
    human_request: HumanRequest
    task: Task
    budget_used: BudgetSnapshot
    parent_tool_call_id: str


# Resolve forward refs.
ToolOutputWaitingForHuman.model_rebuild()
PausedState.model_rebuild()


# ============================================================================
# HaltReason / RunResult
# ============================================================================


class HaltReasonBudgetExceeded(_Model):
    kind: Literal["budget_exceeded"] = "budget_exceeded"
    limit_type: BudgetLimitTypeT


class HaltReasonTerminationPolicyHalt(_Model):
    kind: Literal["termination_policy_halt"] = "termination_policy_halt"
    reason: str


class HaltReasonMiddlewareHalt(_Model):
    kind: Literal["middleware_halt"] = "middleware_halt"
    hook: HookPoint
    reason: str


class HaltReasonAgentError(_Model):
    kind: Literal["agent_error"] = "agent_error"
    error: AgentError


class HaltReasonSandboxViolation(_Model):
    kind: Literal["sandbox_violation"] = "sandbox_violation"
    violation: SandboxViolation


class HaltReasonUnrecoverableToolError(_Model):
    kind: Literal["unrecoverable_tool_error"] = "unrecoverable_tool_error"
    tool: str
    error: str


class HaltReasonHumanHalted(_Model):
    kind: Literal["human_halted"] = "human_halted"


class HaltReasonStagnationLimitReached(_Model):
    kind: Literal["stagnation_limit_reached"] = "stagnation_limit_reached"
    iterations: int
    best_metric: float


class HaltReasonStrategyNotYetImplemented(_Model):
    """Returned by :class:`StandardHarness` for non-ReAct strategies whose
    concrete trait dependencies ship with later component issues."""

    kind: Literal["strategy_not_yet_implemented"] = "strategy_not_yet_implemented"
    strategy: str


HaltReason = Annotated[
    HaltReasonBudgetExceeded
    | HaltReasonTerminationPolicyHalt
    | HaltReasonMiddlewareHalt
    | HaltReasonAgentError
    | HaltReasonSandboxViolation
    | HaltReasonUnrecoverableToolError
    | HaltReasonHumanHalted
    | HaltReasonStagnationLimitReached
    | HaltReasonStrategyNotYetImplemented,
    Field(discriminator="kind"),
]


class RunResultSuccess(_Model):
    kind: Literal["success"] = "success"
    output: str
    session_id: SessionId
    usage: AggregateUsage
    turns: int


class RunResultFailure(_Model):
    kind: Literal["failure"] = "failure"
    reason: HaltReason
    session_id: SessionId
    usage: AggregateUsage
    turns: int


class RunResultWaitingForHuman(_Model):
    kind: Literal["waiting_for_human"] = "waiting_for_human"
    state: PausedState
    request: HumanRequest


RunResult = Annotated[
    RunResultSuccess | RunResultFailure | RunResultWaitingForHuman,
    Field(discriminator="kind"),
]


# Import canonical middleware decision types. This import is deliberately
# placed after the harness's own types are defined so
# :mod:`spore_core.middleware` can import :class:`HumanRequest`,
# :class:`RunResult`, :class:`Task`, :class:`SessionState`, etc., from
# this module without circularity.
from .middleware import (  # noqa: E402
    MiddlewareContinue,
    MiddlewareContinueWithModification,
    MiddlewareDecision,
    MiddlewareForceAnotherTurn,
    MiddlewareHalt,
    MiddlewareSurfaceToHuman,
)

# Canonical observability surface (issue #12). Imported here, after this
# module's identity types are defined, because :mod:`spore_core.observability`
# imports :class:`SessionId` / :class:`TaskId` from this module — a top-level
# import would be circular. The harness loop emits real spans through this
# provider and flushes on terminal outcomes (mirrors the Rust ``run_react``
# wrapper). The durable-outbox provider is used by
# :meth:`HarnessBuilder.with_observability_outbox`.
from .guide_registry import (  # noqa: E402
    SessionOutcome,
    SessionOutcomeFailure,
    SessionOutcomeSuccess,
)
from .memory import now as _now  # noqa: E402
from .observability import (  # noqa: E402
    ObservabilityProvider,
    PricingTable,
    SpanBase,
    SpanKind,
    SpanStatusError,
    SpanStatusOk,
    ToolCallSpan,
    TurnSpan,
    new_span_id,
)
from .observability_outbox import (  # noqa: E402
    OutboxConfig,
    OutboxObservabilityProvider,
)

# ============================================================================
# HarnessRunOptions
# ============================================================================


class HarnessRunOptions:
    """Options for :meth:`Harness.run`. Not a pydantic model because
    ``on_stream`` is a callable."""

    def __init__(
        self,
        task: Task,
        *,
        on_stream: StreamSink | None = None,
        session_state: SessionState | None = None,
    ) -> None:
        self.task = task
        self.on_stream = on_stream
        self.session_state = session_state


# ============================================================================
# Harness protocol
# ============================================================================


@runtime_checkable
class Harness(Protocol):
    """Drives the agent loop."""

    async def run(self, options: HarnessRunOptions) -> RunResult: ...

    async def resume(
        self,
        state: PausedState,
        response: HumanResponse,
        on_stream: StreamSink | None = None,
    ) -> RunResult: ...


# ============================================================================
# HarnessConfig + StandardHarness
# ============================================================================


class HarnessConfig:
    """Components injected at construction. Mirrors ``HarnessConfig`` in
    the spec. Optional components default to no-op stubs once the loop is
    actually exercised; only ``agent``, ``tool_registry``, ``sandbox``,
    ``context_manager``, ``termination_policy`` are required by the
    ReAct path."""

    def __init__(
        self,
        *,
        agent: Agent,
        tool_registry: ToolRegistry,
        sandbox: SandboxProvider,
        context_manager: ContextManager,
        termination_policy: TerminationPolicy,
        middleware: HarnessMiddlewareChain | None = None,
        observability: ObservabilityProvider | None = None,
        pricing: PricingTable | None = None,
    ) -> None:
        self.agent = agent
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.context_manager = context_manager
        self.termination_policy = termination_policy
        self.middleware = middleware
        self.observability = observability
        # Token → USD pricing used to stamp ``cost_usd`` on emitted turn spans.
        # Defaults to :attr:`PricingTable.DEFAULT` (zero cost) when unset.
        self.pricing: PricingTable = pricing if pricing is not None else PricingTable.DEFAULT


class HarnessBuilder:
    """Fluent assembler for a :class:`HarnessConfig` / :class:`StandardHarness`.

    Mirrors the Rust ``HarnessBuilder``. The harness follows strict inversion
    of control: every component is injected. The builder takes the five
    required components up front and exposes fluent setters for the optional
    ones (middleware, observability, pricing), including the durable outbox via
    :meth:`with_observability_outbox`.
    """

    def __init__(
        self,
        agent: Agent,
        tool_registry: ToolRegistry,
        sandbox: SandboxProvider,
        context_manager: ContextManager,
        termination_policy: TerminationPolicy,
    ) -> None:
        self._agent = agent
        self._tool_registry = tool_registry
        self._sandbox = sandbox
        self._context_manager = context_manager
        self._termination_policy = termination_policy
        self._middleware: HarnessMiddlewareChain | None = None
        self._observability: ObservabilityProvider | None = None
        self._pricing: PricingTable = PricingTable.DEFAULT

    def middleware(self, middleware: HarnessMiddlewareChain) -> HarnessBuilder:
        """Inject a middleware chain."""
        self._middleware = middleware
        return self

    def observability(self, observability: ObservabilityProvider) -> HarnessBuilder:
        """Inject an observability provider. The harness loop emits real spans
        through it (turn spans, tool-call spans) and flushes on terminal
        outcomes."""
        self._observability = observability
        return self

    def with_observability_outbox(self, root: str | Path) -> HarnessBuilder:
        """Construct and inject a durable-outbox observability provider rooted
        at ``root`` (typically the ``.spore`` directory). Honors the
        ``SPORE_OTLP_ENDPOINT`` env var for OTLP forwarding (issue #33)."""
        provider = OutboxObservabilityProvider(OutboxConfig(root=Path(root)))
        return self.observability(provider)

    def pricing(self, pricing: PricingTable) -> HarnessBuilder:
        """Set the token → USD pricing table used to stamp ``cost_usd`` on
        turn spans."""
        self._pricing = pricing
        return self

    def build_config(self) -> HarnessConfig:
        """Assemble the :class:`HarnessConfig` without wrapping it in a
        harness."""
        return HarnessConfig(
            agent=self._agent,
            tool_registry=self._tool_registry,
            sandbox=self._sandbox,
            context_manager=self._context_manager,
            termination_policy=self._termination_policy,
            middleware=self._middleware,
            observability=self._observability,
            pricing=self._pricing,
        )

    def build(self) -> StandardHarness:
        """Assemble a ready-to-run :class:`StandardHarness`."""
        return StandardHarness(self.build_config())


class StandardHarness:
    """Canonical :class:`Harness` implementation.

    Implements the ReAct loop fully; other :class:`LoopStrategy` variants
    return :class:`HaltReasonStrategyNotYetImplemented` per the Rust
    reference.
    """

    def __init__(self, config: HarnessConfig) -> None:
        self._config = config

    # ---- helpers ----------------------------------------------------

    @staticmethod
    def _emit(stream: StreamSink | None, event: HarnessStreamEvent) -> None:
        if stream is not None:
            stream(event)

    @staticmethod
    def _budget_exceeded(
        budget: BudgetLimits, used: BudgetSnapshot, started_at: float
    ) -> BudgetLimitTypeT | None:
        if budget.max_turns is not None and used.turns >= budget.max_turns:
            return "turns"
        if budget.max_input_tokens is not None and used.input_tokens > budget.max_input_tokens:
            return "input_tokens"
        if budget.max_output_tokens is not None and used.output_tokens > budget.max_output_tokens:
            return "output_tokens"
        if (
            budget.max_wall_time is not None
            and (time.monotonic() - started_at) >= budget.max_wall_time
        ):
            return "wall_time"
        if budget.max_cost_usd is not None and used.cost_usd > budget.max_cost_usd:
            return "cost_usd"
        return None

    def _fail(
        self,
        reason: HaltReason,
        session_id: SessionId,
        usage: AggregateUsage,
        turns: int,
    ) -> RunResultFailure:
        return RunResultFailure(
            reason=reason,
            session_id=session_id,
            usage=usage,
            turns=turns,
        )

    # ---- public API -------------------------------------------------

    async def run(self, options: HarnessRunOptions) -> RunResult:
        task = options.task
        on_stream = options.on_stream
        session_state = options.session_state or SessionState()
        budget_used = BudgetSnapshot()

        strategy = task.loop_strategy
        if isinstance(strategy, LoopStrategyReAct):
            return await self._run_react(
                task, strategy.max_iterations, session_state, budget_used, on_stream
            )
        if isinstance(strategy, LoopStrategyPlanExecute):
            return self._fail(
                HaltReasonStrategyNotYetImplemented(strategy="plan_execute"),
                task.session_id,
                AggregateUsage(),
                0,
            )
        if isinstance(strategy, LoopStrategyRalph):
            return self._fail(
                HaltReasonStrategyNotYetImplemented(strategy="ralph"),
                task.session_id,
                AggregateUsage(),
                0,
            )
        if isinstance(strategy, LoopStrategySelfVerifying):
            return self._fail(
                HaltReasonStrategyNotYetImplemented(strategy="self_verifying"),
                task.session_id,
                AggregateUsage(),
                0,
            )
        if isinstance(strategy, LoopStrategyHillClimbing):
            return self._fail(
                HaltReasonStrategyNotYetImplemented(strategy="hill_climbing"),
                task.session_id,
                AggregateUsage(),
                0,
            )
        raise AssertionError(f"unknown loop strategy: {strategy!r}")

    async def resume(
        self,
        state: PausedState,
        response: HumanResponse,
        on_stream: StreamSink | None = None,
    ) -> RunResult:
        session_state = state.session_state
        pending = state.pending_tool_calls
        task = state.task
        budget_used = state.budget_used
        session_id = state.session_id

        # Subagent depth: a full child.resume() dispatch lives with
        # SubagentTool (#5); for now we surface a placeholder and continue
        # the parent loop — mirrors the Rust reference behavior.
        if state.child_state is not None:
            pass

        if isinstance(response, HumanResponseHalt):
            return self._fail(
                HaltReasonHumanHalted(),
                session_id,
                AggregateUsage(),
                state.turn_number,
            )

        if isinstance(response, HumanResponseDeny):
            for call in pending:
                tr = HarnessToolResult(
                    call_id=call.id,
                    output=ToolOutputError(message=response.reason, recoverable=True),
                )
                await self._config.context_manager.append_tool_result(session_state, tr)

        elif isinstance(response, HumanResponseReject):
            await self._config.context_manager.append_user_message(session_state, response.reason)

        elif isinstance(response, HumanResponseAnswer):
            await self._config.context_manager.append_user_message(session_state, response.text)

        elif isinstance(response, HumanResponseApproveWithFeedback):
            await self._config.context_manager.append_user_message(session_state, response.feedback)

        elif isinstance(response, HumanResponseAllow):
            for call in pending:
                output = await self._config.tool_registry.dispatch(call)
                tr = HarnessToolResult(call_id=call.id, output=output)
                await self._config.context_manager.append_tool_result(session_state, tr)

        elif isinstance(response, HumanResponseAllowWithModification):
            for call in response.calls:
                output = await self._config.tool_registry.dispatch(call)
                tr = HarnessToolResult(call_id=call.id, output=output)
                await self._config.context_manager.append_tool_result(session_state, tr)

        # Resume the ReAct loop from where we paused.
        max_iterations = (
            task.loop_strategy.max_iterations
            if isinstance(task.loop_strategy, LoopStrategyReAct)
            else 2**31 - 1
        )
        return await self._run_react(task, max_iterations, session_state, budget_used, on_stream)

    # ---- observability finalization ---------------------------------

    async def _finalize_observability(self, session_id: SessionId, outcome: SessionOutcome) -> None:
        """Record the terminal outcome and flush the observability session.
        Called at every terminal ``_run_react`` outcome (success or any halt)
        — never on a ``WaitingForHuman`` pause, which is not terminal. No-op
        when no provider is configured. Mirrors Rust's
        ``finalize_observability``."""
        obs = self._config.observability
        if obs is not None:
            obs.set_session_outcome(session_id, outcome)
            await obs.flush_session(session_id)

    # ---- ReAct loop -------------------------------------------------

    async def _run_react(
        self,
        task: Task,
        max_iterations: int,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
    ) -> RunResult:
        """Drive the ReAct loop, then finalize observability for terminal
        outcomes. A ``WaitingForHuman`` pause is not terminal, so it is never
        flushed here — the eventual ``resume`` reaches a terminal outcome and
        flushes then. Mirrors Rust's ``run_react`` wrapper."""
        result = await self._run_react_inner(
            task, max_iterations, session_state, budget_used, on_stream
        )
        if isinstance(result, RunResultSuccess):
            await self._finalize_observability(result.session_id, SessionOutcomeSuccess())
        elif isinstance(result, RunResultFailure):
            await self._finalize_observability(
                result.session_id,
                SessionOutcomeFailure(reason=result.reason.kind),
            )
        # RunResultWaitingForHuman: not terminal, do not finalize.
        return result

    async def _run_react_inner(
        self,
        task: Task,
        max_iterations: int,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
    ) -> RunResult:
        session_id = task.session_id
        started_at = time.monotonic()
        usage = AggregateUsage()
        # Monotonic per-run span counter for tool-call span ids, and the most
        # recent turn span base — the parent for that turn's tool-call spans.
        span_seq = 0
        current_turn_base: SpanBase | None = None
        if task.budget.max_turns is not None:
            effective_turn_cap = min(task.budget.max_turns, max_iterations)
        else:
            effective_turn_cap = max_iterations

        config = self._config

        while True:
            # Layer-1 budget gates before the turn.
            if budget_used.turns >= effective_turn_cap:
                return self._fail(
                    HaltReasonBudgetExceeded(limit_type="turns"),
                    session_id,
                    usage,
                    budget_used.turns,
                )
            limit_type = self._budget_exceeded(task.budget, budget_used, started_at)
            if limit_type is not None:
                return self._fail(
                    HaltReasonBudgetExceeded(limit_type=limit_type),
                    session_id,
                    usage,
                    budget_used.turns,
                )

            # Middleware: BeforeTurn.
            if config.middleware is not None:
                decision = await config.middleware.fire("before_turn", session_state)
                if isinstance(decision, MiddlewareHalt):
                    return self._fail(
                        HaltReasonMiddlewareHalt(hook="before_turn", reason=decision.reason),
                        session_id,
                        usage,
                        budget_used.turns,
                    )
                if isinstance(decision, MiddlewareSurfaceToHuman):
                    paused = PausedState(
                        session_id=session_id,
                        task_id=task.id,
                        turn_number=budget_used.turns,
                        session_state=session_state,
                        pending_tool_calls=[],
                        approved_results=[],
                        human_request=decision.request,
                        task=task,
                        budget_used=budget_used,
                        child_state=None,
                    )
                    return RunResultWaitingForHuman(state=paused, request=decision.request)

            # Assemble + invoke agent for one turn.
            context = await config.context_manager.assemble(session_state, task)
            self._emit(on_stream, StreamTurnStart(turn=budget_used.turns + 1))
            turn_started_at = _now()
            turn_clock = time.monotonic()
            result: TurnResult = await config.agent.turn(context)
            budget_used.turns += 1

            # Emit a turn span for every model call (issue #12). Fire-and-forget;
            # it never affects control flow. The span base is retained as the
            # parent for any tool-call spans dispatched this turn.
            zero = TokenUsage()
            if isinstance(result, ToolCallRequested | FinalResponse):
                u = result.usage
            elif isinstance(result, TurnError):
                u = result.usage or zero
            else:
                u = zero
            if isinstance(result, FinalResponse):
                stop_reason, tool_calls_requested = StopReason.END_TURN, 0
            elif isinstance(result, ToolCallRequested):
                stop_reason, tool_calls_requested = StopReason.TOOL_USE, len(result.calls)
            else:
                stop_reason, tool_calls_requested = StopReason.END_TURN, 0
            turn_base = SpanBase.new_root(
                new_span_id(f"{session_id}-turn-{budget_used.turns}"),
                session_id,
                task.id,
                SpanKind.TURN,
                turn_started_at,
            )
            turn_status: SpanStatusOk | SpanStatusError
            if isinstance(result, TurnError):
                turn_status = SpanStatusError(message=result.error.kind)
            else:
                turn_status = SpanStatusOk()
            turn_base.finish(
                _now(),
                turn_status,
                int((time.monotonic() - turn_clock) * 1000),
            )
            if config.observability is not None:
                config.observability.emit_turn(
                    TurnSpan(
                        base=turn_base,
                        turn_number=budget_used.turns,
                        input_tokens=u.input_tokens,
                        output_tokens=u.output_tokens,
                        cache_read_tokens=u.cache_read_tokens,
                        cache_write_tokens=u.cache_write_tokens,
                        cost_usd=config.pricing.cost_for(
                            u.input_tokens,
                            u.output_tokens,
                            u.cache_read_tokens,
                            u.cache_write_tokens,
                        ),
                        stop_reason=stop_reason,
                        tool_calls_requested=tool_calls_requested,
                    )
                )
            span_seq += 1
            current_turn_base = turn_base

            self._emit(on_stream, StreamTurnEnd(turn=budget_used.turns))

            # ---- FinalResponse --------------------------------------
            if isinstance(result, FinalResponse):
                usage.add_turn(result.usage)
                budget_used.input_tokens += result.usage.input_tokens
                budget_used.output_tokens += result.usage.output_tokens

                if config.middleware is not None:
                    decision = await config.middleware.fire("before_completion", session_state)
                    if isinstance(decision, MiddlewareHalt):
                        return self._fail(
                            HaltReasonMiddlewareHalt(
                                hook="before_completion", reason=decision.reason
                            ),
                            session_id,
                            usage,
                            budget_used.turns,
                        )
                    if isinstance(decision, MiddlewareSurfaceToHuman):
                        paused = PausedState(
                            session_id=session_id,
                            task_id=task.id,
                            turn_number=budget_used.turns,
                            session_state=session_state,
                            pending_tool_calls=[],
                            approved_results=[],
                            human_request=decision.request,
                            task=task,
                            budget_used=budget_used,
                            child_state=None,
                        )
                        return RunResultWaitingForHuman(state=paused, request=decision.request)

                tdecision = await config.termination_policy.evaluate(session_state, budget_used)
                if isinstance(tdecision, TerminationHalt):
                    return self._fail(
                        HaltReasonTerminationPolicyHalt(reason=tdecision.reason),
                        session_id,
                        usage,
                        budget_used.turns,
                    )

                self._emit(on_stream, StreamFinalResponse(content=result.content))
                return RunResultSuccess(
                    output=result.content,
                    session_id=session_id,
                    usage=usage,
                    turns=budget_used.turns,
                )

            # ---- ToolCallRequested ----------------------------------
            if isinstance(result, ToolCallRequested):
                usage.add_turn(result.usage)
                budget_used.input_tokens += result.usage.input_tokens
                budget_used.output_tokens += result.usage.output_tokens

                # Always-halt short-circuit.
                for c in result.calls:
                    if config.tool_registry.is_always_halt(c.name):
                        return self._fail(
                            HaltReasonUnrecoverableToolError(
                                tool=c.name,
                                error="tool is annotated always_halt",
                            ),
                            session_id,
                            usage,
                            budget_used.turns,
                        )

                # Middleware: BeforeTool.
                calls = result.calls
                if config.middleware is not None:
                    decision = await config.middleware.fire("before_tool", session_state)
                    if isinstance(decision, MiddlewareContinueWithModification):
                        if decision.calls is not None:
                            calls = decision.calls
                    elif isinstance(decision, MiddlewareHalt):
                        return self._fail(
                            HaltReasonMiddlewareHalt(hook="before_tool", reason=decision.reason),
                            session_id,
                            usage,
                            budget_used.turns,
                        )
                    elif isinstance(decision, MiddlewareSurfaceToHuman):
                        paused = PausedState(
                            session_id=session_id,
                            task_id=task.id,
                            turn_number=budget_used.turns,
                            session_state=session_state,
                            pending_tool_calls=calls,
                            approved_results=[],
                            human_request=decision.request,
                            task=task,
                            budget_used=budget_used,
                            child_state=None,
                        )
                        return RunResultWaitingForHuman(state=paused, request=decision.request)

                approved_results: list[HarnessToolResult] = []
                for i, call in enumerate(calls):
                    # Sandbox validation.
                    violation = await config.sandbox.validate(call)
                    if violation is not None:
                        if sandbox_violation_is_always_halt(violation):
                            return self._fail(
                                HaltReasonSandboxViolation(violation=violation),
                                session_id,
                                usage,
                                budget_used.turns,
                            )
                        # Layer-2 default: recoverable — append as tool error.
                        tr = HarnessToolResult(
                            call_id=call.id,
                            output=ToolOutputError(
                                message=f"sandbox: {violation.kind}",
                                recoverable=True,
                            ),
                        )
                        self._emit(
                            on_stream,
                            StreamToolResult(call_id=call.id, is_error=True),
                        )
                        await config.context_manager.append_tool_result(session_state, tr)
                        approved_results.append(tr)
                        continue

                    self._emit(on_stream, StreamToolCall(call_id=call.id, name=call.name))
                    tool_started_at = _now()
                    tool_clock = time.monotonic()
                    output = await config.tool_registry.dispatch(call)

                    # Subagent pause propagation.
                    if isinstance(output, ToolOutputWaitingForHuman):
                        remaining = calls[i + 1 :]
                        paused = PausedState(
                            session_id=session_id,
                            task_id=task.id,
                            turn_number=budget_used.turns,
                            session_state=session_state,
                            pending_tool_calls=remaining,
                            approved_results=approved_results,
                            human_request=output.request,
                            task=task,
                            budget_used=budget_used,
                            child_state=output.child_state,
                        )
                        return RunResultWaitingForHuman(state=paused, request=output.request)

                    is_error = isinstance(output, ToolOutputError)
                    # Layer-2: unrecoverable tool error halts immediately.
                    if isinstance(output, ToolOutputError) and not output.recoverable:
                        return self._fail(
                            HaltReasonUnrecoverableToolError(tool=call.name, error=output.message),
                            session_id,
                            usage,
                            budget_used.turns,
                        )

                    # Tool-call span (issue #12), child of the current turn.
                    if config.observability is not None:
                        if isinstance(output, ToolOutputSuccess):
                            output_size_bytes = len(output.content)
                            out_truncated = output.truncated
                        elif isinstance(output, ToolOutputError):
                            output_size_bytes = len(output.message)
                            out_truncated = False
                        else:
                            output_size_bytes = 0
                            out_truncated = False
                        span_id = new_span_id(f"{session_id}-tool-{span_seq}")
                        if current_turn_base is not None:
                            tool_base = SpanBase.new_child(
                                span_id, current_turn_base, SpanKind.TOOL_CALL, tool_started_at
                            )
                        else:
                            tool_base = SpanBase.new_root(
                                span_id,
                                session_id,
                                task.id,
                                SpanKind.TOOL_CALL,
                                tool_started_at,
                            )
                        tool_status: SpanStatusOk | SpanStatusError = (
                            SpanStatusError(message="tool returned a recoverable error")
                            if is_error
                            else SpanStatusOk()
                        )
                        tool_base.finish(
                            _now(),
                            tool_status,
                            int((time.monotonic() - tool_clock) * 1000),
                        )
                        config.observability.emit_tool_call(
                            ToolCallSpan(
                                base=tool_base,
                                tool_name=call.name,
                                call_id=call.id,
                                parameters_size_bytes=len(
                                    json.dumps(call.input, separators=(",", ":"))
                                ),
                                output_size_bytes=output_size_bytes,
                                truncated=out_truncated,
                                sandbox_mode="",
                                sandbox_violations=[],
                            )
                        )
                        span_seq += 1

                    tr = HarnessToolResult(call_id=call.id, output=output)
                    self._emit(
                        on_stream,
                        StreamToolResult(call_id=call.id, is_error=is_error),
                    )
                    await config.context_manager.append_tool_result(session_state, tr)
                    approved_results.append(tr)

                # Middleware: AfterTool.
                if config.middleware is not None:
                    decision = await config.middleware.fire("after_tool", session_state)
                    if isinstance(decision, MiddlewareHalt):
                        return self._fail(
                            HaltReasonMiddlewareHalt(hook="after_tool", reason=decision.reason),
                            session_id,
                            usage,
                            budget_used.turns,
                        )

                continue

            # ---- TurnError ------------------------------------------
            if isinstance(result, TurnError):
                if result.usage is not None:
                    usage.add_turn(result.usage)
                    budget_used.input_tokens += result.usage.input_tokens
                    budget_used.output_tokens += result.usage.output_tokens
                return self._fail(
                    HaltReasonAgentError(error=result.error),
                    session_id,
                    usage,
                    budget_used.turns,
                )

            raise AssertionError(f"unhandled TurnResult variant: {result!r}")


# ============================================================================
# Test-only stub implementations of the sibling component traits.
# Used by the unit tests and the fixture-replay test in this package.
# ============================================================================


class NoopContextManager:
    """Default context manager: passes session messages through unchanged
    and appends tool results / user messages as plain text. Sufficient for
    ReAct unit tests; the canonical impl lands in #7."""

    async def assemble(self, session: SessionState, task: Task) -> Context:
        return Context(
            messages=list(session.messages),
            tools=[],
            params=ModelParams(),
        )

    async def append_tool_result(self, session: SessionState, result: HarnessToolResult) -> None:
        # No-op: harness unit tests do not assert on message-shape; #7
        # owns the canonical wire shape.
        return None

    async def append_user_message(self, session: SessionState, text: str) -> None:
        return None

    def should_compact(self, session: SessionState) -> bool:
        return False


class AllowAllSandbox(BaseSandboxProvider):
    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None


class ScriptedSandbox(BaseSandboxProvider):
    """Pop-front queue of validation outcomes (None for allow)."""

    def __init__(self) -> None:
        self._outcomes: list[SandboxViolation | None] = []

    def push(self, outcome: SandboxViolation | None) -> ScriptedSandbox:
        self._outcomes.append(outcome)
        return self

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        if not self._outcomes:
            return None
        return self._outcomes.pop(0)


class ScriptedToolRegistry:
    def __init__(self) -> None:
        self._outputs: list[ToolOutput] = []
        self._always_halt: set[str] = set()
        self.call_count: int = 0

    def push(self, output: ToolOutput) -> ScriptedToolRegistry:
        self._outputs.append(output)
        return self

    def mark_always_halt(self, name: str) -> None:
        self._always_halt.add(name)

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        self.call_count += 1
        if not self._outputs:
            return ToolOutputSuccess(content="ok")
        return self._outputs.pop(0)

    def is_always_halt(self, tool_name: str) -> bool:
        return tool_name in self._always_halt

    def schemas(self) -> list[ToolSchema]:
        return []


class AlwaysContinuePolicy:
    async def evaluate(
        self, session: SessionState, budget_used: BudgetSnapshot
    ) -> TerminationDecision:
        return TerminationContinue()


class ScriptedTerminationPolicy:
    def __init__(self) -> None:
        self._decisions: list[TerminationDecision] = []

    def push(self, d: TerminationDecision) -> ScriptedTerminationPolicy:
        self._decisions.append(d)
        return self

    async def evaluate(
        self, session: SessionState, budget_used: BudgetSnapshot
    ) -> TerminationDecision:
        if not self._decisions:
            return TerminationContinue()
        return self._decisions.pop(0)


class ScriptedMiddleware:
    """Pop-front (hook, decision) queue. If the next entry's hook doesn't
    match the current fire(), returns Continue without consuming the entry.
    """

    def __init__(self) -> None:
        self._decisions: list[tuple[HookPoint, MiddlewareDecision]] = []

    def push(self, hook: HookPoint, decision: MiddlewareDecision) -> ScriptedMiddleware:
        self._decisions.append((hook, decision))
        return self

    async def fire(self, hook: HookPoint, session: SessionState) -> MiddlewareDecision:
        if not self._decisions:
            return MiddlewareContinue()
        front_hook, _ = self._decisions[0]
        if front_hook != hook:
            return MiddlewareContinue()
        return self._decisions.pop(0)[1]


# ============================================================================
# Internal helper exports
# ============================================================================


# Type aliases re-exported as the canonical names so downstream code can
# write ``isinstance(decision, MiddlewareSurfaceToHuman)`` etc.
__all__ = [
    "AggregateUsage",
    "AllowAllSandbox",
    "AlwaysContinuePolicy",
    "BaseSandboxProvider",
    "CommandOutput",
    "FileRef",
    "TruncatedOutput",
    "BudgetLimits",
    "BudgetLimitTypeT",
    "BudgetSnapshot",
    "ChildPausedState",
    "ContextManager",
    "HaltReason",
    "HaltReasonAgentError",
    "HaltReasonBudgetExceeded",
    "HaltReasonHumanHalted",
    "HaltReasonMiddlewareHalt",
    "HaltReasonSandboxViolation",
    "HaltReasonStagnationLimitReached",
    "HaltReasonStrategyNotYetImplemented",
    "HaltReasonTerminationPolicyHalt",
    "HaltReasonUnrecoverableToolError",
    "Harness",
    "HarnessBuilder",
    "HarnessConfig",
    "HarnessRunOptions",
    "HarnessStreamEvent",
    "HarnessToolResult",
    "HookPoint",
    "HumanRequest",
    "HumanRequestClarification",
    "HumanRequestReview",
    "HumanRequestToolApproval",
    "HumanResponse",
    "HumanResponseAllow",
    "HumanResponseAllowWithModification",
    "HumanResponseAnswer",
    "HumanResponseApproveWithFeedback",
    "HumanResponseDeny",
    "HumanResponseHalt",
    "HumanResponseReject",
    "LoopStrategy",
    "LoopStrategyHillClimbing",
    "LoopStrategyPlanExecute",
    "LoopStrategyRalph",
    "LoopStrategyReAct",
    "LoopStrategySelfVerifying",
    "HarnessMiddlewareChain",
    "MiddlewareContinue",
    "MiddlewareContinueWithModification",
    "MiddlewareDecision",
    "MiddlewareForceAnotherTurn",
    "MiddlewareHalt",
    "MiddlewareSurfaceToHuman",
    "ModelConfig",
    "NoopContextManager",
    "OptimizationDirection",
    "PausedState",
    "RiskLevel",
    "RunResult",
    "RunResultFailure",
    "RunResultSuccess",
    "RunResultWaitingForHuman",
    "BwrapProfile",
    "IsolationMode",
    "IsolationModeBubblewrap",
    "IsolationModeDocker",
    "IsolationModeNone",
    "IsolationModeWorkspaceScoped",
    "NetworkPolicy",
    "NetworkPolicyAllowlist",
    "NetworkPolicyFull",
    "NetworkPolicyNone",
    "Operation",
    "SandboxDisallowedCommand",
    "SandboxExtensionDenied",
    "SandboxFileSizeExceeded",
    "SandboxNetworkViolation",
    "SandboxPathDenied",
    "SandboxPathEscape",
    "SandboxProvider",
    "SandboxReadOnlyViolation",
    "SandboxViolation",
    "WorkspaceConfig",
    "ScriptedMiddleware",
    "ScriptedSandbox",
    "ScriptedTerminationPolicy",
    "ScriptedToolRegistry",
    "SessionId",
    "SessionState",
    "StandardHarness",
    "StreamBudgetWarning",
    "StreamFinalResponse",
    "StreamSink",
    "StreamToolCall",
    "StreamToolResult",
    "StreamTurnEnd",
    "StreamTurnStart",
    "Task",
    "TaskId",
    "TerminationContinue",
    "TerminationDecision",
    "TerminationHalt",
    "TerminationPolicy",
    "ToolOutput",
    "ToolOutputError",
    "ToolOutputSuccess",
    "ToolOutputWaitingForHuman",
    "ToolRegistry",
    "new_session_id",
    "new_task_id",
    "sandbox_violation_is_always_halt",
]

# Avoid unused-import warnings for `Awaitable` (kept for IDE hover usefulness).
_: Awaitable[None] | None = None
