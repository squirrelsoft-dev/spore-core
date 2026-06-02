"""Harness — the agent runtime loop (issue #3).

Mirrors the Rust reference at ``rust/crates/spore-core/src/harness.rs``.
The harness owns execution lifecycle and wires all components together.
It is stateless between :meth:`Harness.run` calls; everything the harness
needs comes in via :class:`HarnessRunOptions` or :class:`PausedState`, and
everything it produces goes out via :class:`RunResult`.

``dangerous`` gate (issue #34): ``IsolationModeNone`` (no path enforcement) is
a named safety footgun. It is not part of the default public surface — it is
not exported from this module or ``spore_core``, and is reachable only via
``from spore_core.dangerous import IsolationModeNone``. Consequently the
default :meth:`SandboxProvider.isolation_mode` body returns
``IsolationModeWorkspaceScoped`` (safe-by-default), never ``None``. The wire
tag for the gated mode stays ``"none"``.

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
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NewType, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .agent import (
    Agent,
    AgentError,
    Context,
    FinalResponse,
    ToolCallRequested,
    TurnError,
    TurnResult,
)
from .errors import SporeError
from .model import (
    ImageContent,
    Message,
    ModelParams,
    Role,
    StopReason,
    TextContent,
    TokenUsage,
    ToolCall,
    ToolCallContent as MsgToolCallContent,
    ToolResultContent as MsgToolResultContent,
    ToolSchema,
)

if TYPE_CHECKING:
    from .context import (
        CompactionPreserveHints,
        CompactionVerifier,
        ContextErrorModel,
    )
    from .context import (
        SessionState as ContextSessionState,
    )
    from .hooks import HookChain
    from .tool_registry import StandardToolRegistry

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


class StreamUserMessage(_Model):
    """Out-of-band, prominent message to the user (issue #81). Emitted by the
    loop when the ``send_message`` tool runs, INSTEAD of collapsing the content
    into a normal tool result. The harness only emits the event — rendering it
    prominently is the architect's UI concern."""

    kind: Literal["user_message"] = "user_message"
    content: str


HarnessStreamEvent = Annotated[
    StreamTurnStart
    | StreamTurnEnd
    | StreamToolCall
    | StreamToolResult
    | StreamFinalResponse
    | StreamBudgetWarning
    | StreamUserMessage,
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
    # Issue #81 (Q4b): an ``ask_user_question`` clarification may carry a fixed
    # set of multiple-choice options. Optional / defaulted so older
    # ``Clarification`` blobs (no ``options`` field) still deserialize —
    # back-compat with the bare pre-#81 shape.
    options: list[str] | None = None


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
    """A successful tool result.

    ``truncated`` is ``True`` only when the tool itself clipped its output to
    fit an inline budget (large outputs routed through
    :meth:`SandboxProvider.handle_large_output` set this). Plain tool authors
    should leave it ``False`` — use :meth:`success`.
    """

    kind: Literal["success"] = "success"
    content: str
    truncated: bool = False

    @classmethod
    def success(cls, content: str) -> ToolOutputSuccess:
        """A successful, non-truncated result. The common case for a tool that
        returns its full output — saves spelling out ``truncated=False``."""
        return cls(content=content, truncated=False)


class ToolOutputError(_Model):
    """A failed tool result.

    ``recoverable`` is ``True`` if the agent may sensibly retry or adapt: the
    loop appends the error as a tool result and continues. ``False`` halts the
    run. Most tool failures are recoverable — prefer :meth:`error`; reach for
    :meth:`fatal` only when continuing is pointless.
    """

    kind: Literal["error"] = "error"
    message: str
    recoverable: bool = True

    @classmethod
    def error(cls, message: str) -> ToolOutputError:
        """A **recoverable** error: the harness loop appends it as a tool result
        and lets the agent adapt or retry. This is the right default for almost
        every tool failure (bad arguments, missing file, transient I/O)."""
        return cls(message=message, recoverable=True)

    @classmethod
    def fatal(cls, message: str) -> ToolOutputError:
        """A **fatal** error: continuing is pointless, so the run halts. Reserve
        for genuinely unrecoverable conditions; prefer :meth:`error` when the
        agent could reasonably do something different next turn."""
        return cls(message=message, recoverable=False)


class ToolOutputWaitingForHuman(_Model):
    kind: Literal["waiting_for_human"] = "waiting_for_human"
    child_state: ChildPausedState
    request: HumanRequest


# ----- HarnessSignal (issue #80, Tool Escalation Protocol) ----------------
# The typed channel by which a tool signals the harness to terminate cleanly
# and pass a structured signal up to its caller. The harness never acts on the
# signal itself — it is a pure intermediary. Tagged union over ``kind`` with
# snake_case discriminators (``enter_plan_mode``, ``exit_plan_mode``,
# ``switch_mode``, ``abort``) byte-identical across the four languages.
#
# ``ExitPlanMode`` carries a :class:`PlanArtifact` (defined in
# :mod:`spore_core.hooks`) and ``SwitchMode`` carries a :class:`Mode` (defined
# in :mod:`spore_core.prompt_chunk_registry`). Both of those modules import from
# this module, so the two types are pulled in via the deferred import block at
# the bottom and these models are rebuilt there (forward references).


class HarnessSignalEnterPlanMode(_Model):
    """Agent requests entry into plan mode. Carries the context the agent has
    accumulated so far as a seed for the planning harness."""

    kind: Literal["enter_plan_mode"] = "enter_plan_mode"
    context: str


class HarnessSignalExitPlanMode(_Model):
    """Planning agent has produced a plan and requests exit from plan mode.
    Carries the plan artifact for human approval. The planning agent's terminal
    signal."""

    kind: Literal["exit_plan_mode"] = "exit_plan_mode"
    plan: PlanArtifact


class HarnessSignalSwitchMode(_Model):
    """Agent requests a mode switch. The caller instantiates the appropriate
    harness for the new mode."""

    kind: Literal["switch_mode"] = "switch_mode"
    mode: Mode


class HarnessSignalAbort(_Model):
    """Agent requests a graceful, intentional stop with a reason surfaced to the
    user. Distinct from :class:`HaltReasonAgentError` — it surfaces as
    :class:`RunResultEscalate`, NOT :class:`RunResultFailure`."""

    kind: Literal["abort"] = "abort"
    reason: str


HarnessSignal = Annotated[
    HarnessSignalEnterPlanMode
    | HarnessSignalExitPlanMode
    | HarnessSignalSwitchMode
    | HarnessSignalAbort,
    Field(discriminator="kind"),
]


class ToolOutputEscalate(_Model):
    """Tool requests a structural state change from the harness's parent. The
    harness loop recognizes this variant, terminates the current run cleanly,
    and passes the signal up to the caller via :class:`RunResultEscalate`."""

    kind: Literal["escalate"] = "escalate"
    signal: HarnessSignal


class ToolOutputAwaitingClarification(_Model):
    """Tool needs a human answer before it can produce a result (issue #81,
    Q4b). UNLIKE :class:`ToolOutputWaitingForHuman` (the subagent-shaped pause
    that carries a :class:`ChildPausedState`), this variant carries NO child
    state: the loop pauses by building a :class:`PausedState` directly with
    ``human_request`` set to :class:`HumanRequestClarification`. On resume the
    human's answer is injected as the clarifying call's tool result."""

    kind: Literal["awaiting_clarification"] = "awaiting_clarification"
    question: str
    options: list[str] | None = None


ToolOutput = Annotated[
    ToolOutputSuccess
    | ToolOutputError
    | ToolOutputWaitingForHuman
    | ToolOutputEscalate
    | ToolOutputAwaitingClarification,
    Field(discriminator="kind"),
]

#: The registered name of the catalogue ``send_message`` tool (issue #81). The
#: harness loop recognizes this name and emits a :class:`StreamUserMessage`
#: event rather than collapsing the tool result. The implementation lives in
#: ``spore_tools`` (``SendMessageTool``); the harness only needs the name.
SEND_MESSAGE_TOOL_NAME = "send_message"


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


# ----- VcsProvider seam (issue #58 v2) — git-log reload for Ralph ----------


@dataclass
class VcsLogArgs:
    """Parameters shaping a :meth:`VcsProvider.log` read. Each field maps to a
    ``git log`` flag in :class:`GitVcsProvider`:

    - ``max_entries`` → ``-n <N>`` (cap the number of commits returned),
    - ``since_ref``   → ``<ref>..`` (only commits AFTER ``<ref>``),
    - ``format``      → ``--format=<fmt>`` (custom pretty format).

    Mirrors Rust's ``VcsLogArgs``.
    """

    max_entries: int
    since_ref: str | None = None
    format: str | None = None


class VcsError(SporeError):
    """Error raised by a :class:`VcsProvider`. One exception class per
    component, inheriting from the package root :class:`SporeError` — mirrors
    Rust's ``VcsError`` (``CommandFailed`` / ``Sandbox`` variants). The
    ``message`` carries the captured stderr or the sandbox-block detail."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@runtime_checkable
class VcsProvider(Protocol):
    """Read-only VCS abstraction the ``Ralph`` loop strategy uses to reload git
    history between context windows (issue #58 v2, decision B4).

    Mirrors how :class:`SandboxProvider` abstracts filesystem/shell access:
    a Protocol, a real implementation (:class:`GitVcsProvider`), and a
    deterministic fixture double (:class:`FixtureVcsProvider`), injected at
    construction via :meth:`HarnessBuilder.vcs_provider`. ``Ralph`` calls
    :meth:`log` during its reload phase and injects the output into the next
    window's seed as a clearly delimited "Recent VCS history:" section. When NO
    provider is wired (``vcs_provider is None``, the default) the git-log
    section is OMITTED and Ralph behaves exactly like v1. Raises
    :class:`VcsError` on failure. Mirrors Rust's ``VcsProvider`` trait.
    """

    async def log(self, args: VcsLogArgs) -> str:
        """Return the project's commit log, shaped by ``args``, verbatim."""
        ...

    async def status(self) -> str:
        """Return the working-tree status (``git status`` stdout), verbatim."""
        ...


class GitVcsProvider:
    """Real :class:`VcsProvider` that shells out to ``git`` THROUGH a
    :class:`SandboxProvider` (issue #58 v2). It wraps the sandbox and calls
    :meth:`SandboxProvider.execute_command` — it never bypasses sandboxing to
    spawn ``git`` directly. The command line is built from :class:`VcsLogArgs`
    (see that type for the flag mapping); :meth:`status` runs ``git status``.
    All commands run in ``workspace_root``. Mirrors Rust's ``GitVcsProvider``.
    """

    def __init__(self, sandbox: SandboxProvider, workspace_root: str | Path) -> None:
        self._sandbox = sandbox
        self._workspace_root = Path(workspace_root)

    @staticmethod
    def _log_args(args: VcsLogArgs) -> list[str]:
        """Build the ``git log`` argument vector from ``args`` (static so the
        flag mapping is testable independently of process execution)."""
        out = ["log", "-n", str(args.max_entries)]
        if args.format is not None:
            out.append(f"--format={args.format}")
        if args.since_ref is not None:
            out.append(f"{args.since_ref}..")
        return out

    async def _run(self, argv: list[str]) -> str:
        from .sandbox import SandboxViolationException

        try:
            out = await self._sandbox.execute_command("git", argv, self._workspace_root)
        except SandboxViolationException as exc:
            raise VcsError(f"vcs command blocked by sandbox: {exc.violation!r}") from exc
        if out.exit_code != 0:
            raise VcsError(out.stderr)
        return out.stdout

    async def log(self, args: VcsLogArgs) -> str:
        return await self._run(self._log_args(args))

    async def status(self) -> str:
        return await self._run(["status"])


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
        # Safe-by-default (issue #34): the default isolation mode is
        # WorkspaceScoped, never None. No-isolation requires the explicit
        # dangerous opt-in (``from spore_core.dangerous import
        # IsolationModeNone``).
        return IsolationModeWorkspaceScoped()

    def workspace_root(self) -> Path:
        return Path("/")


class ReadOnlySandbox:
    """Read-only :class:`SandboxProvider` decorator (issue #61, R3).

    Wraps an inner provider and blocks the standard mutating tools by name —
    any :class:`ToolCall` whose ``name`` is in :attr:`DEFAULT_WRITE_TOOLS` is
    rejected at :meth:`validate` with :class:`SandboxReadOnlyViolation`; every
    other call (the read tools) is delegated to the inner provider. Subprocess
    execution is forbidden outright (commands may have arbitrary write side
    effects), and ``resolve_path`` rejects Write/Execute operations.

    ``ReadOnlyViolation`` is a Layer-2 (recoverable) violation, so in the harness
    loop a blocked write surfaces to the evaluator agent as a recoverable tool
    error — it does NOT halt the evaluate run. Mirrors Rust's ``ReadOnlySandbox``
    decorator in ``harness.rs``."""

    #: Standard-catalogue tool names that MUTATE the workspace and are therefore
    #: blocked by a read-only sandbox.
    DEFAULT_WRITE_TOOLS: frozenset[str] = frozenset(
        {
            "write_file",
            "edit_file",
            "delete_file",
            "move_file",
            "exec",
            "bash_command",
            "run_tests",
        }
    )

    def __init__(
        self,
        inner: SandboxProvider,
        write_tools: Iterable[str] | None = None,
    ) -> None:
        self._inner = inner
        self._write_tools: frozenset[str] = (
            frozenset(write_tools) if write_tools is not None else self.DEFAULT_WRITE_TOOLS
        )

    def _is_write(self, tool_name: str) -> bool:
        return tool_name in self._write_tools

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        if self._is_write(call.name):
            return SandboxReadOnlyViolation(path=call.name)
        return await self._inner.validate(call)

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        # A read-only sandbox forbids subprocess execution outright. Deferred
        # import: :mod:`spore_core.sandbox` imports from this module.
        from .sandbox import SandboxViolationException

        raise SandboxViolationException(SandboxReadOnlyViolation(path=command))

    async def handle_large_output(
        self,
        content: str,
        call_id: str,
        head_tokens: int,
        tail_tokens: int,
    ) -> TruncatedOutput:
        return await self._inner.handle_large_output(content, call_id, head_tokens, tail_tokens)

    async def resolve_path(self, path: str, operation: Operation = "read") -> Path:
        if operation in ("write", "execute"):
            from .sandbox import SandboxViolationException

            raise SandboxViolationException(SandboxReadOnlyViolation(path=path))
        return await self._inner.resolve_path(path, operation)

    def isolation_mode(self) -> IsolationMode:
        return self._inner.isolation_mode()

    def workspace_root(self) -> Path:
        return self._inner.workspace_root()


@dataclass
class CompactionTurn:
    """Inputs the harness compaction loop (issue #46) needs to run one
    compaction turn and verify its result.

    The harness loop operates on the opaque :class:`SessionState`; the rich
    compaction/verification API
    (:class:`spore_core.context.ContextManager`,
    :class:`spore_core.context.CompactionVerifier`) operates on
    :class:`spore_core.context.SessionState`. This struct is the bridge: a
    :class:`ContextManager` that supports compaction projects everything the
    loop needs into one value, so the loop never has to know which concrete
    state type its manager uses internally.

    ``context`` is fed straight to :meth:`Agent.turn` to produce the summary;
    ``preserve_hints`` and ``verification_state`` are passed to
    :meth:`CompactionVerifier.verify`. On a verification failure the loop
    re-runs the turn with :meth:`ContextManager.inject_missing_items` applied
    to ``context``.
    """

    context: Context
    preserve_hints: CompactionPreserveHints
    verification_state: ContextSessionState
    messages_removed: int


@runtime_checkable
class ContextManager(Protocol):
    """Issue #7 — assembles per-turn context.

    Issue #46 adds the optional compaction-loop surface
    (:meth:`prepare_compaction_turn`, :meth:`inject_missing_items`,
    :meth:`apply_compaction`). All three have defaults so managers that do not
    compact (the default :meth:`should_compact` returns ``False``) need not
    implement them.
    """

    async def assemble(self, session: SessionState, task: Task) -> Context: ...

    async def append_tool_result(
        self, session: SessionState, result: HarnessToolResult
    ) -> None: ...

    async def append_assistant_message(self, session: SessionState, message: Message) -> None:
        """Append the assistant's turn (model output: text and/or the tool calls
        it requested) to the conversation so the next :meth:`assemble` reflects
        what the agent already did. Without this the model loses track of its own
        actions and repeats them. Default: no-op — but structural (non-inheriting)
        managers do not pick up this body, so the harness loop probes for the
        method via ``getattr`` before calling it (see ``_run_react_inner``)."""
        _ = (session, message)

    async def append_user_message(self, session: SessionState, text: str) -> None: ...

    def should_compact(self, session: SessionState) -> bool:
        """Whether the session is over its compaction threshold. Default:
        ``False`` — compaction stays opt-in behind this gate."""
        _ = session
        return False

    def prepare_compaction_turn(self, session: SessionState) -> CompactionTurn | None:
        """Build the inputs for one compaction turn (issue #46). Returns
        ``None`` when there is nothing to compact (e.g. history shorter than
        the preserve window), in which case the harness skips compaction
        entirely. Default: ``None`` — managers that never compact need not
        override this."""
        _ = session
        return None

    def inject_missing_items(self, context: Context, missing: list[str]) -> None:
        """Mutate a compaction :class:`Context` in place to request a revised
        summary on retry (issue #46). The harness calls this with the items the
        prior summary failed to preserve. Default: append the standard
        "missing these items … please revise" instruction as a user message."""
        context.messages.append(
            Message(
                role=Role.USER,
                content=TextContent(
                    text=(
                        f"Your summary is missing these items: {', '.join(missing)}. Please revise."
                    )
                ),
            )
        )

    def apply_compaction(self, session: SessionState, summary: str) -> None:
        """Accept a verified (or accepted-anyway) summary into the session,
        replacing the compacted span (issue #46). Default: no-op — only
        compaction-capable managers implement it."""
        _ = (session, summary)

    def token_budget_used(self, session: SessionState) -> int | None:
        """Post-compaction budget seam (#57 Known Deviation #2). The harness
        reads this *after* applying a compaction so the emitted ``Compaction``
        span can stamp the real ``tokens_after`` / ``tokens_reclaimed`` instead
        of reporting zero reclamation. Default: ``None`` — managers that do not
        track a token budget leave the span's pre-compaction estimate in
        place."""
        _ = session
        return None


def _default_inject_missing_items(context: Context, missing: list[str]) -> None:
    """Module-level twin of :meth:`ContextManager.inject_missing_items`'s
    default body. Structural (non-inheriting) managers do not pick up Protocol
    method defaults, so the harness loop falls back to this when a manager does
    not override ``inject_missing_items`` (issue #46)."""
    context.messages.append(
        Message(
            role=Role.USER,
            content=TextContent(
                text=(f"Your summary is missing these items: {', '.join(missing)}. Please revise.")
            ),
        )
    )


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
    # Optional (issue #80): a ``WaitingForHuman`` pause always sets this; an
    # escalation pause sets it to ``None``. Omitted from serialization when
    # ``None`` to match the Rust ``#[serde(default)]`` wire behavior — old
    # ``WaitingForHuman`` blobs (field present) still deserialize.
    human_request: HumanRequest | None = None
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
    # Optional (issue #80): mirrors :class:`PausedState.human_request`.
    human_request: HumanRequest | None = None
    task: Task
    budget_used: BudgetSnapshot
    parent_tool_call_id: str


# Forward refs are resolved at the BOTTOM of the module (after the deferred
# import block) rather than here: ``ToolOutputWaitingForHuman`` and
# ``PausedState`` both transitively reach ``ToolOutputEscalate`` →
# ``HarnessSignal`` → :class:`PlanArtifact` / :class:`Mode` (issue #80), and
# those two types live in modules that import from this one — they are only
# importable once ``HarnessConfig`` is defined. See the ``model_rebuild`` block
# after the ``from .hooks import PlanArtifact`` / ``from .prompt_chunk_registry
# import Mode`` imports.


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


class HaltReasonContextError(_Model):
    """A :class:`spore_core.context.ContextError` surfaced by the
    ``ContextManager`` during assembly halts the run — e.g. a cache-hash
    mismatch, where both Block 1 (``static``) and, as of #32, Block 2
    (``per_session``) halt mid-session. This is the routing TYPE; mirrors
    :class:`HaltReasonAgentError`. The live :class:`StandardHarness` loop does
    not yet trigger it because its placeholder ``ContextManager.assemble`` is
    infallible pending the #7 migration. Mirrors Rust's
    ``HaltReason::ContextError { error: ContextError }``."""

    kind: Literal["context_error"] = "context_error"
    error: ContextErrorModel


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


class HaltReasonEmptyPlan(_Model):
    """Returned by :class:`StandardHarness` for the ``PlanExecute`` strategy
    (issue #59, Q3). The plan phase produced a well-formed artifact whose task
    list is empty (``tasks: []``); the execute loop has nothing to drive, so the
    run fails rather than silently succeeding. Mirrors Rust's
    ``HaltReason::EmptyPlan`` and the Go/TypeScript equivalents."""

    kind: Literal["empty_plan"] = "empty_plan"


class HaltReasonStepFailed(_Model):
    """Returned by :class:`StandardHarness` for the ``PlanExecute`` strategy
    (issue #59, Q5). A per-task execute sub-loop errored or returned a
    blocked/failed outcome; this aborts the whole run (a plan is a dependency
    chain, so later tasks do not run). Carries the failing step's index, its
    description, and a debug rendering of the underlying terminal reason.
    Mirrors Rust's ``HaltReason::StepFailed { task_index, task, reason }``."""

    kind: Literal["step_failed"] = "step_failed"
    task_index: int
    task: str
    reason: str


class PlanPhaseErrorPayload(_Model):
    """Serialized form of a :class:`spore_core.plan.PlanPhaseError` as it appears
    nested under :class:`HaltReasonPlanPhaseFailed`. Wire shape matches Rust's
    ``PlanPhaseError`` and Go's ``PlanPhaseError``:
    ``{"kind": "unparseable_plan"|"planning_turn_failed", "message": "..."}``."""

    kind: Literal["unparseable_plan", "planning_turn_failed"]
    message: str


class HaltReasonPlanPhaseFailed(_Model):
    """The PlanExecute plan phase (issue #70) failed before producing an
    artifact — the planning turn failed (R2: a tool call, or an agent error) or
    the response text could not be parsed under the Q3 grammar.

    Carries the underlying :class:`PlanPhaseError` NESTED under an ``error`` key,
    matching the 3-language majority (Rust's
    ``HaltReason::PlanPhaseFailed { error: PlanPhaseError }`` and Go's
    ``HaltPlanPhaseFailed`` carrying ``*PlanPhaseError`` under ``json:"error"``,
    and the TypeScript nested ``error``)."""

    kind: Literal["plan_phase_failed"] = "plan_phase_failed"
    error: PlanPhaseErrorPayload


class HaltReasonSelfVerifyExhausted(_Model):
    """Returned by :class:`StandardHarness` for the ``SelfVerifying`` strategy
    (issue #61, D4) when the build↔evaluate loop ran out of the verifier's
    ``max_iterations`` round-trips without an explicit ``Passed`` verdict — a
    RUNTIME limit (the Default-FAIL stagnation guard). Carries the number of
    round-trips run and the last failure reason the verifier gave. PEER to
    :class:`HaltReasonSelfVerifyMisconfigured` (NOT a sub-case of it). Mirrors
    Rust's ``HaltReason::SelfVerifyExhausted { iterations, last_reason }``."""

    kind: Literal["self_verify_exhausted"] = "self_verify_exhausted"
    iterations: int
    last_reason: str


class HaltReasonSelfVerifyMisconfigured(_Model):
    """Returned by :class:`StandardHarness` for the ``SelfVerifying`` strategy
    (issue #61, D4) when the strategy cannot run because it is misconfigured —
    e.g. ``config.verifier`` is ``None``. A BUILD-TIME wiring bug, surfaced as a
    typed halt, NOT a raise. PEER to :class:`HaltReasonSelfVerifyExhausted`.
    Mirrors Rust's ``HaltReason::SelfVerifyMisconfigured { reason }``."""

    kind: Literal["self_verify_misconfigured"] = "self_verify_misconfigured"
    reason: str


class HaltReasonHillClimbingMisconfigured(_Model):
    """Returned by :class:`StandardHarness` for the ``HillClimbing`` strategy
    (issue #60) when the strategy cannot run because it is misconfigured — e.g.
    ``config.metric_evaluator`` is ``None`` (Decision 6), or the iteration-0
    baseline evaluation itself errored so there is no current best to climb from
    (Decision 7). A typed halt, NOT a raise. PEER to
    :class:`HaltReasonSelfVerifyMisconfigured`. Mirrors Rust's
    ``HaltReason::HillClimbingMisconfigured { reason }``."""

    kind: Literal["hill_climbing_misconfigured"] = "hill_climbing_misconfigured"
    reason: str


class HaltReasonRalphCompletionUnmet(_Model):
    """Returned by :class:`StandardHarness` for the ``Ralph`` strategy (issue
    #58) when the multi-context-window continuation loop reached its
    ``max_resets`` cap with tasks still incomplete (the Ralph analogue of
    :class:`HaltReasonSelfVerifyExhausted`). A RUNTIME limit — the work was
    attempted across ``iterations`` context windows but the filesystem-backed
    completion check (the registered ``Stop`` hook reading
    ``.spore/progress.json``) never reported done. Carries the number of
    context-window resets performed and the last incompletion reason. Mirrors
    Rust's ``HaltReason::RalphCompletionUnmet { iterations, last_reason }``."""

    kind: Literal["ralph_completion_unmet"] = "ralph_completion_unmet"
    iterations: int
    last_reason: str


HaltReason = Annotated[
    HaltReasonBudgetExceeded
    | HaltReasonTerminationPolicyHalt
    | HaltReasonMiddlewareHalt
    | HaltReasonAgentError
    | HaltReasonContextError
    | HaltReasonSandboxViolation
    | HaltReasonUnrecoverableToolError
    | HaltReasonHumanHalted
    | HaltReasonStagnationLimitReached
    | HaltReasonStrategyNotYetImplemented
    | HaltReasonEmptyPlan
    | HaltReasonStepFailed
    | HaltReasonPlanPhaseFailed
    | HaltReasonSelfVerifyExhausted
    | HaltReasonSelfVerifyMisconfigured
    | HaltReasonHillClimbingMisconfigured
    | HaltReasonRalphCompletionUnmet,
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


class RunResultEscalate(_Model):
    """The harness terminated cleanly because a tool returned
    :class:`ToolOutputEscalate` (issue #80). Carries the signal plus the
    preserved :class:`PausedState` (with ``human_request = None``) so the caller
    can handle the signal and decide whether to resume the original harness,
    instantiate a new one, or present UI to the user. The signal is NOT stored
    in the ``state`` — it is discarded on resume; the harness never re-acts on
    it."""

    kind: Literal["escalate"] = "escalate"
    signal: HarnessSignal
    state: PausedState
    session_id: SessionId
    usage: AggregateUsage
    turns: int


RunResult = Annotated[
    RunResultSuccess | RunResultFailure | RunResultWaitingForHuman | RunResultEscalate,
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
    SessionOutcomeEscalated,
    SessionOutcomeFailure,
    SessionOutcomeSuccess,
)
from .memory import now as _now  # noqa: E402
from .observability import (  # noqa: E402
    ContentCaptureConfig,
    ContextOperationCompaction,
    ContextSpan,
    GenAiMessage,
    GenAiRole,
    ObservabilityProvider,
    PricingTable,
    SpanBase,
    SpanKind,
    SpanStatusError,
    SpanStatusOk,
    ToolCallContent,
    ToolCallSpan,
    ToolResultContent,
    TurnSpan,
    WarnEventCompactionVerificationFailed,
    WarnEventHillClimbingIteration,
    WarnSpan,
    new_span_id,
    truncate_field,
)
from .observability_outbox import (  # noqa: E402
    OutboxConfig,
    OutboxObservabilityProvider,
)
from .storage import (  # noqa: E402
    SessionStore,
    StorageProvider,
)


def _capture_tool_call_args(call: ToolCall, max_len: int) -> ToolCallContent:
    """Capture a requested tool call as :class:`ToolCallContent`, truncating its
    arguments to ``max_len`` UTF-8 bytes (issue #64). The arguments are measured
    by their canonical JSON serialization; when over budget they are clipped and
    stored as a JSON string carrying the truncation marker (a structured value
    cannot be clipped in place), with ``arguments_truncated=True``. Mirrors the
    Rust reference ``capture_tool_call_args``."""
    serialized = json.dumps(call.input, separators=(",", ":"))
    clipped, truncated = truncate_field(serialized, max_len)
    arguments: Any = clipped if truncated else call.input
    return ToolCallContent(
        name=call.name,
        arguments=arguments,
        arguments_truncated=truncated,
    )


_ROLE_TO_GENAI: dict[Role, GenAiRole] = {
    Role.SYSTEM: GenAiRole.SYSTEM,
    Role.USER: GenAiRole.USER,
    Role.ASSISTANT: GenAiRole.ASSISTANT,
    Role.TOOL: GenAiRole.TOOL,
}


def _capture_input_messages(messages: list[Message], max_len: int) -> list[GenAiMessage]:
    """Snapshot the assembled INPUT messages (the full prompt the model saw)
    into :class:`GenAiMessage`s for LLM-native tracing (issue #64). Each
    message's :class:`Role` maps to the conventional :class:`GenAiRole`; the
    :class:`Content` is rendered to a plain string and truncated to ``max_len``
    UTF-8 bytes:

    * ``TextContent``       → the text verbatim
    * ``ToolResultContent`` → its ``content`` body (role stays ``Tool``)
    * ``ToolCallContent``   → ``"<name> <compact-json-args>"`` (assistant)
    * ``ImageContent``      → ``"[image <media_type>]"`` — NEVER the base64 data

    System-first, then history order is preserved because the assembled
    ``messages`` already lead with the :class:`Role.SYSTEM` prompt. Mirrors the
    Rust reference ``capture_input_messages``."""
    out: list[GenAiMessage] = []
    for m in messages:
        content = m.content
        if isinstance(content, TextContent):
            rendered = content.text
        elif isinstance(content, MsgToolResultContent):
            rendered = content.content
        elif isinstance(content, MsgToolCallContent):
            rendered = f"{content.name} {json.dumps(content.input, separators=(',', ':'))}"
        elif isinstance(content, ImageContent):
            # NEVER dump the base64 ``data`` — placeholder only.
            rendered = f"[image {content.media_type}]"
        else:  # pragma: no cover - exhaustive over the Content union
            rendered = ""
        clipped, truncated = truncate_field(rendered, max_len)
        out.append(
            GenAiMessage(
                role=_ROLE_TO_GENAI[m.role],
                content=clipped,
                truncated=truncated,
            )
        )
    return out


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
        compaction_verifier: CompactionVerifier | None = None,
        max_compaction_attempts: int = 2,
        pricing: PricingTable | None = None,
        content_capture: ContentCaptureConfig | None = None,
        max_stop_blocks: int = 8,
        max_resets: int = 3,
        vcs_provider: VcsProvider | None = None,
        hooks: HookChain | None = None,
        planner_agent: Agent | None = None,
        verifier: Any | None = None,
        evaluator_agent: Agent | None = None,
        storage: StorageProvider | None = None,
        chunk_provider: Any | None = None,
        metric_evaluator: Any | None = None,
        catalogue_registry: StandardToolRegistry | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.agent = agent
        # The HillClimbing scoring strategy (issue #60). REQUIRED for the
        # ``HillClimbing`` strategy: when ``None`` the run halts with
        # :class:`HaltReasonHillClimbingMisconfigured` (Decision 6) — a typed
        # halt, never a raise. The harness calls ``evaluate`` once per iteration
        # (iteration 0 is a pure baseline with NO agent turn) and routes the
        # result through :func:`~spore_core.metric.should_keep`. Typed ``Any`` to
        # avoid a top-level import of :mod:`spore_core.metric` (that module
        # imports from this one — a top-level import is circular). Mirrors Rust's
        # ``HarnessConfig::metric_evaluator``.
        self.metric_evaluator: Any | None = metric_evaluator
        # Optional alternate agent used for the PlanExecute plan phase (issue
        # #70, Q1). When the loop strategy is ``PlanExecute`` and this is set,
        # the one-shot plan turn runs on it; otherwise the plan turn runs on
        # ``agent``. ``None`` means "use the default agent".
        self.planner_agent = planner_agent
        # The SelfVerifying oracle (issue #61, D2). REQUIRED for the
        # ``SelfVerifying`` strategy: when ``None`` the run halts with
        # :class:`HaltReasonSelfVerifyMisconfigured` (D4) — a typed halt, never a
        # raise. Its ``max_iterations()`` (default 3) caps the build↔evaluate
        # round-trips (D3); ``max_stop_blocks`` does NOT govern this strategy.
        # Typed ``Any`` to avoid a top-level import of :mod:`spore_core.verifier`
        # (that module imports from this one — a top-level import is circular).
        self.verifier: Any | None = verifier
        # Optional alternate agent for the SelfVerifying evaluate phase (issue
        # #61, D2). Defaulting contract is IDENTICAL to ``planner_agent``: when
        # ``None`` the evaluate phase runs on ``config.agent``
        # (``evaluator_agent or agent``). The read-only sandbox and the fresh
        # never-shared session id are derived INTERNALLY by the strategy — there
        # are deliberately NO evaluator sandbox / chunk-provider config fields.
        self.evaluator_agent = evaluator_agent
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.context_manager = context_manager
        self.termination_policy = termination_policy
        self.middleware = middleware
        self.observability = observability
        # Lifecycle hook chain (issue #69). The harness fires registered Stop
        # hooks when a loop strategy believes it is done; a Stop ``block``
        # injects a reason and continues the loop. ``None`` means no hooks.
        self.hooks: HookChain | None = hooks
        # Maximum consecutive Stop-hook blocks honored per run before the loop
        # terminates anyway (issue #69, R14). Per-run counter; resume starts
        # fresh. Default 8, matching Claude Code's behavior.
        self.max_stop_blocks = max_stop_blocks
        # Ralph outer-loop reset cap (issue #58, B3). The maximum number of
        # context windows the ``Ralph`` strategy runs before halting with
        # :class:`HaltReasonRalphCompletionUnmet` when tasks are still
        # incomplete. Independent of ``budget.max_turns`` (which bounds the
        # per-window ReAct sub-loop). Default ``3``.
        self.max_resets = max_resets
        # VcsProvider seam (issue #58 v2, decision B4). When set, the ``Ralph``
        # reload phase ALSO calls ``vcs_provider.log(args)`` and injects the
        # output into each fresh context window's seed as a delimited
        # "Recent VCS history:" section, exactly the way the reloaded
        # ``.spore/`` progress/feature-list content is injected. When ``None``
        # (the default) the git-log section is OMITTED and Ralph behaves
        # byte-for-byte like v1. Mirrors Rust's ``HarnessConfig::vcs_provider``.
        self.vcs_provider: VcsProvider | None = vcs_provider
        # Post-compaction verifier (issue #29/#46). The harness runs it after
        # each compaction turn and retries up to ``max_compaction_attempts``
        # before accepting a failing summary. Defaults to ``KeyTermVerifier``.
        if compaction_verifier is None:
            from .context import KeyTermVerifier

            compaction_verifier = KeyTermVerifier()
        self.compaction_verifier: CompactionVerifier = compaction_verifier
        # Maximum compaction-summary attempts before accepting a failing summary
        # anyway (issue #46). Defaults to ``2`` (mirrors ``CompactionConfig``).
        self.max_compaction_attempts = max_compaction_attempts
        # Token → USD pricing used to stamp ``cost_usd`` on emitted turn spans.
        # Defaults to :attr:`PricingTable.DEFAULT` (zero cost) when unset.
        self.pricing: PricingTable = pricing if pricing is not None else PricingTable.DEFAULT
        # LLM-native content capture config (issue #64). Defaults to OFF. When
        # disabled the harness populates none of the ``gen_ai.*`` content fields,
        # so the durable JSONL stays byte-identical to the pre-#64 output.
        self.content_capture: ContentCaptureConfig = (
            content_capture if content_capture is not None else ContentCaptureConfig()
        )
        # Pluggable persistence layer (issue #73). Defaults to an all-no-op
        # provider so existing callers/tests are unaffected; v1 is expose-only —
        # the run/resume loop does NOT read/write sessions internally. Callers
        # reach the four domain stores via :meth:`StandardHarness.storage` /
        # :meth:`StandardHarness.session_store`.
        self.storage: StorageProvider = storage if storage is not None else StorageProvider.no_op()
        # Pluggable chunk source for the #79 prompt assembly engine. Defaults to
        # an empty in-memory provider. Typed ``Any`` to avoid importing
        # ``prompt_assembly`` at module load.
        if chunk_provider is None:
            from .prompt_assembly import InMemoryChunkProvider

            chunk_provider = InMemoryChunkProvider.empty()
        self.chunk_provider: Any = chunk_provider
        # Catalogue tools accumulated via :meth:`HarnessBuilder.tool` /
        # ``tools`` (issue #81), drained into a populated
        # :class:`StandardToolRegistry` at :meth:`HarnessBuilder.build_config`.
        # When not ``None`` the run loop bridges it per-run via
        # :class:`~spore_core.tool_registry.RealToolRegistry` — threading the
        # run's :class:`SessionId`, sandbox, and storage into every tool
        # dispatch — and uses that instead of :attr:`tool_registry` (which stays
        # the harness-loop seam for custom slim registries). ``None`` (the
        # default) preserves the ``tool_registry``-only path unchanged.
        self.catalogue_registry: StandardToolRegistry | None = catalogue_registry
        # Operating system prompt prepended to each turn's assembled context
        # when the context manager renders none (issue #91). See
        # :meth:`HarnessBuilder.system_prompt`. ``None`` (the default) preserves
        # today's behaviour.
        self.system_prompt: str | None = system_prompt


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
        self._compaction_verifier: CompactionVerifier | None = None
        self._max_compaction_attempts: int = 2
        self._pricing: PricingTable = PricingTable.DEFAULT
        self._content_capture: ContentCaptureConfig | None = None
        self._max_stop_blocks: int = 8
        self._max_resets: int = 3
        self._vcs_provider: VcsProvider | None = None
        self._hooks: HookChain | None = None
        self._planner_agent: Agent | None = None
        self._verifier: Any | None = None
        self._evaluator_agent: Agent | None = None
        self._metric_evaluator: Any | None = None
        self._storage: StorageProvider | None = None
        # Standard catalogue tools accumulated via :meth:`tool` / :meth:`tools`
        # (issue #81). Each is a ``StandardTool``-shaped object exposing
        # ``implementation`` (a :class:`Tool`) and ``schema`` (a
        # :class:`ToolSchema`). They are drained into a populated
        # :class:`StandardToolRegistry` by :meth:`drain_tools_into_registry`,
        # applying last-wins upsert. Typed structurally (the concrete
        # ``StandardTool`` lives in ``spore_tools`` and must not be imported
        # here — that would invert the package dependency).
        self._standard_tools: list[Any] = []
        # Optional operating system prompt prepended to each turn's assembled
        # context (issue #91) when the context manager renders none. ``None``
        # (the default) preserves today's behaviour. See :meth:`system_prompt`.
        self._system_prompt: str | None = None
        # Pluggable chunk source for the #79 prompt assembly engine. Defaults to
        # an empty in-memory provider so existing callers are unaffected. Typed
        # ``Any`` to avoid importing ``prompt_assembly`` at module load (that
        # module imports ``SessionId``/``TaskId`` from here).
        self._chunk_provider: Any | None = None

    def chunk_provider(self, provider: Any) -> HarnessBuilder:
        """Set the chunk provider for the #79 prompt assembly engine. Defaults
        to an empty :class:`~spore_core.prompt_assembly.InMemoryChunkProvider`
        when unset. Mirrors Rust's ``HarnessBuilder::chunk_provider``."""
        self._chunk_provider = provider
        return self

    def chunks(self, chunks: Iterable[Any]) -> HarnessBuilder:
        """Convenience: register chunks inline without constructing a provider
        (issue #79). Resolves to an
        :class:`~spore_core.prompt_assembly.InMemoryChunkProvider` internally.
        Mirrors Rust's ``HarnessBuilder::chunks``."""
        from .prompt_assembly import InMemoryChunkProvider

        self._chunk_provider = InMemoryChunkProvider(list(chunks))
        return self

    def tool(self, tool: Any) -> HarnessBuilder:
        """Accumulate a single ``StandardTool`` (issue #81, Q1/Q2) — an object
        bundling ``implementation`` + ``schema``. The bundle is destructured
        when the registry is built via :meth:`drain_tools_into_registry`.
        Registration applies LAST-WINS upsert: a later ``.tool()`` with the same
        name overrides an earlier one (e.g. a custom tool after a preset)."""
        self._standard_tools.append(tool)
        return self

    def tools(self, tools: Iterable[Any]) -> HarnessBuilder:
        """Accumulate many ``StandardTool``s at once (e.g. a preset like
        ``StandardTools.coding_set()``). Order is preserved, so last-wins upsert
        still applies across the batch."""
        self._standard_tools.extend(tools)
        return self

    def sandbox(self, sandbox: SandboxProvider) -> HarnessBuilder:
        """Override the :class:`SandboxProvider` — the only path tools have to
        the environment (filesystem, process exec).

        :meth:`conversational` defaults to a null sandbox that denies
        environment access — fine for pure-compute tools, but catalogue file
        tools (``read_file`` / ``write_file`` / ``list_dir``) operate *through*
        the sandbox, so an agent that touches a real directory needs a
        workspace-scoped sandbox here::

            harness = (
                HarnessBuilder.conversational(agent)
                .sandbox(workspace)
                .tools(StandardTools.coding_set())
                .build()
            )
        """
        self._sandbox = sandbox
        return self

    def system_prompt(self, system_prompt: str) -> HarnessBuilder:
        """Set an operating system prompt prepended to each turn's assembled
        context (issue #91).

        The standard compaction context manager renders no system prompt, so
        without this the model receives only the task as a user message and no
        guidance on how to behave. When set, the run loop inserts this text as a
        leading :class:`~spore_core.model.Role.SYSTEM` message each turn — but
        only when the assembled context does not already start with one, so a
        context manager that renders its own system prompt is preserved.
        ``None`` (the default) preserves today's behaviour."""
        self._system_prompt = system_prompt
        return self

    def drain_tools_into_registry(self) -> StandardToolRegistry:
        """Drain the accumulated catalogue tools into a populated
        :class:`StandardToolRegistry`, registering each with last-wins upsert
        (issue #81, Q1/Q2). Consumes the accumulated set. Returns an empty
        registry if no catalogue tools were added. Mirrors Rust's
        ``HarnessBuilder::drain_tools_into_registry``."""
        from .tool_registry import StandardToolRegistry as _Registry

        reg = _Registry()
        for t in self._standard_tools:
            reg.register(t.implementation, t.schema)
        self._standard_tools = []
        return reg

    def storage(self, storage: StorageProvider) -> HarnessBuilder:
        """Inject a :class:`StorageProvider` (issue #73). Defaults to an
        all-no-op provider when unset. v1 is expose-only — the harness loop does
        not read/write sessions internally; callers reach the four domain stores
        via :meth:`StandardHarness.storage` / :meth:`StandardHarness.session_store`."""
        self._storage = storage
        return self

    def planner_agent(self, planner_agent: Agent) -> HarnessBuilder:
        """Inject an alternate agent for the PlanExecute plan phase (issue #70,
        Q1). When set and the loop strategy is ``PlanExecute``, the one-shot
        plan turn runs on this agent instead of the default agent."""
        self._planner_agent = planner_agent
        return self

    def verifier(self, verifier: Any) -> HarnessBuilder:
        """Inject the SelfVerifying oracle (issue #61, D2). REQUIRED for the
        ``SelfVerifying`` strategy: without it the run halts with
        :class:`HaltReasonSelfVerifyMisconfigured` (D4). Its ``max_iterations()``
        caps the build↔evaluate round-trips (D3). Mirrors Rust's
        ``HarnessBuilder::verifier``."""
        self._verifier = verifier
        return self

    def evaluator_agent(self, evaluator_agent: Agent) -> HarnessBuilder:
        """Inject an alternate agent for the SelfVerifying evaluate phase (issue
        #61, D2). Mirrors :meth:`planner_agent`: when set and the loop strategy
        is ``SelfVerifying``, the evaluate phase runs on this agent instead of
        the default agent. Mirrors Rust's ``HarnessBuilder::evaluator_agent``."""
        self._evaluator_agent = evaluator_agent
        return self

    def metric_evaluator(self, evaluator: Any) -> HarnessBuilder:
        """Inject the HillClimbing scoring strategy (issue #60). REQUIRED for the
        ``HillClimbing`` strategy: without it the run halts with
        :class:`HaltReasonHillClimbingMisconfigured` (Decision 6) — a typed halt,
        never a raise. The harness calls ``evaluate`` once per iteration (with
        iteration 0 the pure baseline) and routes the result through
        :func:`~spore_core.metric.should_keep`. Mirrors Rust's
        ``HarnessBuilder::metric_evaluator``."""
        self._metric_evaluator = evaluator
        return self

    def hooks(self, hooks: HookChain) -> HarnessBuilder:
        """Inject a lifecycle hook chain (issue #69). The harness fires its
        registered ``Stop`` hooks when a loop strategy believes it is done."""
        self._hooks = hooks
        return self

    def max_stop_blocks(self, max_blocks: int) -> HarnessBuilder:
        """Set the maximum consecutive Stop-hook blocks honored per run before
        the loop terminates anyway (issue #69). Defaults to ``8``."""
        self._max_stop_blocks = max_blocks
        return self

    def max_resets(self, max_resets: int) -> HarnessBuilder:
        """Set the Ralph outer-loop reset cap (issue #58, B3) — the maximum
        number of context windows the ``Ralph`` strategy runs before halting
        with :class:`HaltReasonRalphCompletionUnmet`. Independent of
        ``budget.max_turns``. Defaults to ``3``. Mirrors Rust's
        ``HarnessBuilder::max_resets``."""
        self._max_resets = max_resets
        return self

    def vcs_provider(self, provider: VcsProvider) -> HarnessBuilder:
        """Inject a :class:`VcsProvider` for the ``Ralph`` loop strategy (issue
        #58 v2). When set, Ralph's reload phase calls :meth:`VcsProvider.log`
        and injects a delimited "Recent VCS history:" section into each fresh
        context window's seed. Unset (the default) ⇒ no git section is injected
        and Ralph behaves exactly like v1. Mirrors Rust's
        ``HarnessBuilder::vcs_provider``."""
        self._vcs_provider = provider
        return self

    def compaction_verifier(self, verifier: CompactionVerifier) -> HarnessBuilder:
        """Inject a post-compaction verifier (issue #46). Defaults to
        ``KeyTermVerifier``."""
        self._compaction_verifier = verifier
        return self

    def max_compaction_attempts(self, attempts: int) -> HarnessBuilder:
        """Set the maximum number of compaction-summary attempts before
        accepting a failing summary anyway (issue #46). Defaults to ``2``."""
        self._max_compaction_attempts = attempts
        return self

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

    def content_capture(self, content_capture: ContentCaptureConfig) -> HarnessBuilder:
        """Set the LLM-native content-capture config (issue #64). OFF by
        default. Use :meth:`ContentCaptureConfig.from_env` to honor
        ``SPORE_TRACE_CONTENT`` / ``SPORE_TRACE_CONTENT_MAX_LEN``."""
        self._content_capture = content_capture
        return self

    def build_config(self) -> HarnessConfig:
        """Assemble the :class:`HarnessConfig` without wrapping it in a
        harness."""
        # Fold catalogue tools accumulated via ``.tool()`` / ``.tools()`` into a
        # populated :class:`StandardToolRegistry` (last-wins upsert). The run
        # loop bridges it per-run — ``build()`` can't, because the
        # :class:`ToolContext` is keyed by the run's :class:`SessionId`, unknown
        # until ``run()``.
        catalogue_registry: StandardToolRegistry | None = (
            self.drain_tools_into_registry() if self._standard_tools else None
        )
        # When catalogue tools are present and the caller wired no storage,
        # default to an in-memory provider (not the all-no-op default) so that
        # session-aware tools (todo_write, memory, task_list) actually persist
        # within the run. Pure tools (read_file/write_file via the sandbox) are
        # unaffected either way.
        storage = self._storage
        if storage is None and catalogue_registry is not None:
            from .storage import InMemoryStorageProvider

            storage = StorageProvider.single(InMemoryStorageProvider())
        return HarnessConfig(
            agent=self._agent,
            tool_registry=self._tool_registry,
            sandbox=self._sandbox,
            context_manager=self._context_manager,
            termination_policy=self._termination_policy,
            middleware=self._middleware,
            observability=self._observability,
            compaction_verifier=self._compaction_verifier,
            max_compaction_attempts=self._max_compaction_attempts,
            pricing=self._pricing,
            content_capture=self._content_capture,
            max_stop_blocks=self._max_stop_blocks,
            max_resets=self._max_resets,
            vcs_provider=self._vcs_provider,
            hooks=self._hooks,
            planner_agent=self._planner_agent,
            verifier=self._verifier,
            evaluator_agent=self._evaluator_agent,
            storage=storage,
            chunk_provider=self._chunk_provider,
            metric_evaluator=self._metric_evaluator,
            catalogue_registry=catalogue_registry,
            system_prompt=self._system_prompt,
        )

    def build(self) -> StandardHarness:
        """Assemble a ready-to-run :class:`StandardHarness`."""
        return StandardHarness(self.build_config())


# ============================================================================
# Ralph loop strategy (issue #58) — filesystem-backed completion contract
# ============================================================================


class RalphProgress(_Model):
    """Deserialized ``.spore/progress.json`` (issue #58, B2/B4). The agent
    writes this each context window to record what it has finished and what
    remains. ``complete: true`` with an empty ``remaining`` ⇒ progress
    satisfied. Mirrors Rust's ``RalphProgress``."""

    complete: bool = False
    remaining: list[str] = Field(default_factory=list)


class RalphFeatureEntry(_Model):
    """One entry of ``.spore/feature_list.json`` — the
    :class:`~spore_core.termination.FeatureListCheck` schema (issue #58, B2).
    Any ``passes: false`` ⇒ incomplete. Mirrors Rust's ``RalphFeatureEntry``."""

    name: str
    passes: bool = False


def _ralph_completion_status(workspace_root: Path) -> str | None:
    """Ralph external completion check (issue #58, B1). Reads the deterministic
    ``.spore/`` files under ``workspace_root`` and reports whether the task is
    complete. Returns ``None`` when complete, the failure reason when tasks
    remain. This is the SAME logic the registered :class:`RalphStopHook`
    applies — one source of truth. Mirrors Rust's ``ralph_completion_status``.

    Contract (B4 — no git):

    * ``.spore/progress.json``: ``{"complete": bool, "remaining": [str]}``.
      ``complete: true`` with an empty ``remaining`` ⇒ progress satisfied.
      Missing / unreadable / invalid ⇒ incomplete (so the agent learns to
      write it).
    * ``.spore/feature_list.json``: a JSON array of ``{"name", "passes"}``. Any
      ``passes: false`` ⇒ incomplete. A MISSING feature list is tolerated
      (progress.json is the primary signal); an invalid one is not.
    """
    progress_path = workspace_root / ".spore" / "progress.json"
    try:
        raw = progress_path.read_text()
    except OSError:
        return ".spore/progress.json missing"
    try:
        progress = RalphProgress.model_validate_json(raw)
    except ValueError as e:
        return f".spore/progress.json invalid JSON: {e}"
    if not progress.complete:
        if not progress.remaining:
            return "task not marked complete"
        return f"remaining: {', '.join(progress.remaining)}"
    if progress.remaining:
        return f"remaining: {', '.join(progress.remaining)}"

    # Progress says done — corroborate against the feature list when present.
    feature_path = workspace_root / ".spore" / "feature_list.json"
    try:
        feature_raw = feature_path.read_text()
    except OSError:
        return None
    try:
        entries = TypeAdapter(list[RalphFeatureEntry]).validate_json(feature_raw)
    except ValueError as e:
        return f".spore/feature_list.json invalid JSON: {e}"
    incomplete = [e.name for e in entries if not e.passes]
    if incomplete:
        return f"incomplete features: {', '.join(incomplete)}"
    return None


def _ralph_reload_context(workspace_root: Path) -> str | None:
    """Build the filesystem-reload context block injected into each fresh Ralph
    context window (issue #58, R3/B4). Returns the verbatim
    ``.spore/progress.json`` and ``.spore/feature_list.json`` contents (when
    present) so the re-seeded window knows what is done and what remains.
    Returns ``None`` when neither file exists. Mirrors Rust's
    ``ralph_reload_context``."""
    parts: list[str] = []
    try:
        raw = (workspace_root / ".spore" / "progress.json").read_text()
        parts.append(f"Reloaded .spore/progress.json:\n{raw.strip()}")
    except OSError:
        pass
    try:
        raw = (workspace_root / ".spore" / "feature_list.json").read_text()
        parts.append(f"Reloaded .spore/feature_list.json:\n{raw.strip()}")
    except OSError:
        pass
    if not parts:
        return None
    return "\n\n".join(parts)


class RalphStopHook:
    """``Stop`` hook driving Ralph's multi-context-window continuation (issue
    #58, B1). At each ``FinalResponse`` it reads ``.spore/progress.json`` under
    ``workspace_root``: incomplete tasks ⇒ :class:`HookBlock` (the loop
    continues), all complete ⇒ :class:`HookContinue` (the loop terminates).

    Registration is harmless for non-Ralph strategies: when
    ``.spore/progress.json`` is ABSENT the hook returns ``Continue`` and does
    not interfere with ReAct / PlanExecute / SelfVerifying runs. It only blocks
    when a progress file is PRESENT and reports incomplete tasks — the Ralph
    contract. Mirrors Rust's ``RalphStopHook``."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root

    async def handle(self, ctx: Any) -> Any:
        from .hooks import HookBlock, HookContinue, StopContext

        # Only act on ``Stop``; any other event is a no-op ``Continue``.
        if not isinstance(ctx, StopContext):
            return HookContinue()
        # Absent progress file ⇒ do not interfere with non-Ralph runs.
        progress_path = self._workspace_root / ".spore" / "progress.json"
        if not progress_path.exists():
            return HookContinue()
        reason = _ralph_completion_status(self._workspace_root)
        if reason is None:
            return HookContinue()
        return HookBlock(reason=reason)

    def events(self) -> list[Any]:
        from .hooks import HookEvent

        return [HookEvent.STOP]

    def name(self) -> str:
        return "ralph-stop"

    def sync_mode(self) -> Any:
        from .hooks import HookSync

        return HookSync.SYNC


@dataclass
class _PlanPhaseOutcome:
    """Internal result of a successful PlanExecute plan phase (issue #70).

    Carries the produced (and possibly hook-mutated) :class:`PlanArtifact` plus
    the run accounting so the ``PlanExecute`` arm can build its terminal
    :class:`RunResult`. Private to the harness."""

    artifact: Any  # PlanArtifact — typed Any to avoid a top-level hooks import
    usage: AggregateUsage
    turns: int


class StandardHarness:
    """Canonical :class:`Harness` implementation.

    Implements the ReAct loop fully and the PlanExecute plan phase (issue #70,
    phase 1 of 2); the remaining :class:`LoopStrategy` variants return
    :class:`HaltReasonStrategyNotYetImplemented` per the Rust reference.
    """

    def __init__(self, config: HarnessConfig) -> None:
        # Ralph completion mechanism (issue #58, B1): register a ``Stop`` hook
        # that drives multi-context-window continuation off
        # ``.spore/progress.json``. Registration is harmless for non-Ralph runs
        # — the hook only BLOCKS when a progress file is PRESENT and reports
        # incomplete tasks; when the file is absent (the common case for ReAct /
        # other strategies) it returns ``Continue``, so existing strategies are
        # unaffected. Mirrors Rust's ``StandardHarness::new``.
        from .hooks import StandardHookChain

        workspace_root = config.sandbox.workspace_root()
        chain = config.hooks if config.hooks is not None else StandardHookChain()
        # Best-effort: a duplicate/invalid registration must never raise out of
        # the constructor. The hook subscribes only to the can-block ``Stop``
        # event, so registration cannot be rejected for sync/async mismatch.
        try:
            chain.register(RalphStopHook(workspace_root))
        except Exception:  # noqa: BLE001 — construction must not fail on a hook
            pass
        config.hooks = chain
        self._config = config

    def storage(self) -> StorageProvider:
        """The injected :class:`StorageProvider` (issue #73). Defaults to an
        all-no-op provider when ``.storage(...)`` was never set."""
        return self._config.storage

    def session_store(self) -> SessionStore:
        """Convenience accessor for the storage layer's :class:`SessionStore`
        (issue #73, expose-only)."""
        return self._config.storage.session()

    def _effective_tool_registry(self, session_id: SessionId) -> ToolRegistry:
        """The harness-loop tool registry to use for a run keyed by
        ``session_id`` (issue #91).

        When catalogue tools were added via :meth:`HarnessBuilder.tool` /
        ``tools``, this bridges the folded :class:`StandardToolRegistry` through
        :class:`~spore_core.tool_registry.RealToolRegistry` — built fresh per run
        so the run's :class:`SessionId` + storage thread into every tool
        dispatch. Otherwise it returns the injected
        :attr:`HarnessConfig.tool_registry` seam unchanged."""
        catalogue = self._config.catalogue_registry
        if catalogue is None:
            return self._config.tool_registry
        from .tool_registry import RealToolRegistry

        return RealToolRegistry(
            catalogue,
            self._config.sandbox,
            session_id,
            self._config.storage.run(),
            self._config.storage.memory(),
        )

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

    async def _fire_stop_hooks(
        self,
        session_id: SessionId,
        task: Task,
        turn_number: int,
        last_output_text: str,
        stop_blocks: int,
    ) -> str | None:
        """Fire registered ``Stop`` hooks (issue #69, R12-R14).

        Returns the block ``reason`` to inject when the loop should continue (a
        hook blocked and the per-run ``max_stop_blocks`` cap has not yet been
        hit). Returns ``None`` to allow normal termination — no hook chain, no
        block, the cap was reached, or a hook errored (a broken hook must not
        loop forever, so its error is treated as a non-blocking outcome).
        """
        config = self._config
        chain = config.hooks
        if chain is None:
            return None

        from .context import SessionState as ContextSessionState
        from .hooks import FireOutcome, StopContext, TurnOutput

        rich_state = ContextSessionState(
            session_id=session_id,
            task_id=task.id,
            task_instruction=task.instruction,
        )
        ctx = StopContext(
            session_id=session_id,
            turn_number=turn_number,
            last_output=TurnOutput(text=last_output_text, had_tool_calls=False),
            task_instruction=task.instruction,
            session_state=rich_state,
        )
        try:
            outcome = await chain.fire(ctx)
        except Exception:  # noqa: BLE001 — a broken Stop hook must not loop forever
            return None

        if isinstance(outcome, FireOutcome) and outcome.kind == "block":
            if stop_blocks >= config.max_stop_blocks:
                return None  # R14: cap reached — terminate anyway.
            return outcome.reason
        # Continue / Inject / Deny → allow normal termination.
        return None

    # ---- public API -------------------------------------------------

    async def run(self, options: HarnessRunOptions) -> RunResult:
        task = options.task
        on_stream = options.on_stream
        session_state = options.session_state or SessionState()
        budget_used = BudgetSnapshot()

        strategy = task.loop_strategy
        if isinstance(strategy, LoopStrategyReAct):
            # Seed the task instruction as the initial user message of this run.
            # The compaction adapter intentionally mirrors ``session.messages``
            # and ignores ``task`` on ``assemble``, so the harness must own
            # delivering the prompt. On a fresh run this turns an otherwise-empty
            # conversation into a real user turn; on multi-turn runs over a
            # carried ``session_state`` each ``run()`` call appends its own
            # follow-up instruction. The resume path does not seed — its
            # conversation already exists.
            await self._config.context_manager.append_user_message(session_state, task.instruction)
            return await self._run_react(
                task, strategy.max_iterations, session_state, budget_used, on_stream
            )
        if isinstance(strategy, LoopStrategyPlanExecute):
            return await self._run_plan_execute(task, session_state, budget_used, on_stream)
        if isinstance(strategy, LoopStrategyRalph):
            # Each context window is a fresh session re-seeded INSIDE
            # ``_run_ralph`` (the context-window reset), so we do NOT pre-seed
            # the instruction here.
            return await self._run_ralph(task, budget_used, on_stream)
        if isinstance(strategy, LoopStrategySelfVerifying):
            # Seed the task instruction as the build loop's initial user message,
            # mirroring the ReAct arm (the build phase is a ReAct sub-loop).
            await self._config.context_manager.append_user_message(session_state, task.instruction)
            return await self._run_self_verifying(task, session_state, budget_used, on_stream)
        if isinstance(strategy, LoopStrategyHillClimbing):
            # Each iteration runs its OWN bounded ReAct sub-run that seeds its own
            # fresh session state (iteration 0 is a pure baseline with NO agent
            # turn), so we do NOT pre-seed the instruction here.
            return await self._run_hill_climbing(
                task,
                strategy.direction,
                strategy.max_stagnation,
                strategy.revert_on_no_improvement,
                strategy.min_improvement_delta,
                budget_used,
                on_stream,
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
        # Resolve the effective tool registry for this resumed session — bridges
        # catalogue tools the same way the turn loop does, so pending tool calls
        # dispatched during resume thread the run's storage + sandbox (issue #91).
        tool_registry = self._effective_tool_registry(session_id)

        # Clarification resume (issue #81, Q4b): if this pause came from
        # :class:`ToolOutputAwaitingClarification`, the human's answer is
        # injected as the tool RESULT for the clarifying call (the HEAD of
        # ``pending_tool_calls``) — NOT appended as a free-standing user message.
        # Any remaining pending calls from the same batch are then dispatched
        # normally before the loop resumes.
        if isinstance(state.human_request, HumanRequestClarification) and isinstance(
            response, HumanResponseAnswer | HumanResponseApproveWithFeedback
        ):
            answer = (
                response.text if isinstance(response, HumanResponseAnswer) else response.feedback
            )
            if pending:
                clarifying_call, *rest = pending
                tr = HarnessToolResult(
                    call_id=clarifying_call.id,
                    output=ToolOutputSuccess(content=answer, truncated=False),
                )
                await self._config.context_manager.append_tool_result(session_state, tr)
                for call in rest:
                    output = await tool_registry.dispatch(call)
                    tr = HarnessToolResult(call_id=call.id, output=output)
                    await self._config.context_manager.append_tool_result(session_state, tr)
            max_iterations = (
                task.loop_strategy.max_iterations
                if isinstance(task.loop_strategy, LoopStrategyReAct)
                else 2**31 - 1
            )
            return await self._run_react(
                task, max_iterations, session_state, budget_used, on_stream
            )

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
                output = await tool_registry.dispatch(call)
                tr = HarnessToolResult(call_id=call.id, output=output)
                await self._config.context_manager.append_tool_result(session_state, tr)

        elif isinstance(response, HumanResponseAllowWithModification):
            for call in response.calls:
                output = await tool_registry.dispatch(call)
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

    # ---- PlanExecute strategy (issues #59 / #70) --------------------

    async def _run_plan_execute(
        self,
        task: Task,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
    ) -> RunResult:
        """Drive the PlanExecute strategy (issue #59) — the two-phase loop.

        Phase 1 (plan, runs EXACTLY once): :meth:`_run_plan_phase` seeds a
        planning directive, runs one constrained planner turn, captures a
        :class:`PlanArtifact`, fires ``OnPlanCreated``, and counts the turn
        against the shared budget. Phase 2 (execute, loops):
        :meth:`_run_execute_phase` drains the parsed task list, giving each task
        its own bounded ReAct sub-loop.

        Resolved spec decisions (issue #59, all FINAL):

        * **Q1** — each task gets its own bounded, isolated, SEQUENTIAL ReAct
          sub-loop; the per-task turn cap is derived at the START of each step as
          ``remaining_turns // remaining_tasks`` (floored at 1). The shared
          budget — turns, tokens, observability, compaction — is carried across
          every step and the global budget is the hard stop.
        * **Q2** — success ``output`` is the LAST completed step's final text.
        * **Q3** — an empty plan (``tasks: []``) ⇒ :class:`HaltReasonEmptyPlan`.
        * **Q4** — the task list/plan persist through the
          :class:`StorageProvider` / :class:`RunStore` seam (plus the ``extras``
          mirror); the #71 sandbox path is NOT used by the execute loop.
        * **Q5** — a step erroring/blocked ABORTS the whole run with
          :class:`HaltReasonStepFailed`; execution does not continue.

        Mirrors Rust's ``run_plan_execute``. Like ``_run_react``, this finalizes
        observability for the terminal outcome.
        """
        from .tasklist import plan_artifact_to_task_list

        session_id = task.session_id

        # Phase 1: plan (runs exactly once).
        outcome = await self._run_plan_phase(task, session_state, budget_used, on_stream)
        if not isinstance(outcome, _PlanPhaseOutcome):
            # Plan-phase failure: propagate unchanged (no task list persisted).
            await self._finalize_plan_execute(outcome)
            return outcome

        # Bridge: parse the accepted plan into a TaskList (#72).
        task_list = plan_artifact_to_task_list(outcome.artifact)

        # Q3: an empty plan is a failure, not a silent success.
        if not task_list.tasks:
            result: RunResult = self._fail(
                HaltReasonEmptyPlan(),
                session_id,
                outcome.usage,
                outcome.turns,
            )
            await self._finalize_plan_execute(result)
            return result

        # Q4: persist through the storage seam (RunStore) — single source of truth.
        await self._persist_task_list(session_id, task_list)

        # Carry the shared budget forward: the plan turn already consumed
        # ``outcome.turns`` turns and ``outcome.usage`` tokens (Q1).
        carried = budget_used.model_copy(deep=True)
        carried.turns = outcome.turns
        carried.input_tokens += outcome.usage.input_tokens
        carried.output_tokens += outcome.usage.output_tokens

        # Phase 2: execute (loops over the task list).
        result = await self._run_execute_phase(
            task, session_state, task_list, carried, outcome.usage, on_stream
        )
        await self._finalize_plan_execute(result)
        return result

    async def _finalize_plan_execute(self, result: RunResult) -> None:
        """Finalize observability for a terminal PlanExecute outcome. Mirrors the
        tail of ``_run_react``: a ``WaitingForHuman`` pause is not terminal and is
        never flushed here. Mirrors Rust's ``finalize_plan_execute``."""
        if isinstance(result, RunResultSuccess):
            await self._finalize_observability(result.session_id, SessionOutcomeSuccess())
        elif isinstance(result, RunResultFailure):
            await self._finalize_observability(
                result.session_id,
                SessionOutcomeFailure(reason=result.reason.kind),
            )
        elif isinstance(result, RunResultEscalate):
            await self._finalize_observability(result.session_id, SessionOutcomeEscalated())
        # RunResultWaitingForHuman: not terminal, do not finalize.

    # ---- SelfVerifying strategy (issue #61) -------------------------

    async def _run_self_verifying(
        self,
        task: Task,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
    ) -> RunResult:
        """Drive the SelfVerifying strategy (issue #61) — the loop-within-a-loop.

        Each round-trip runs a bounded BUILD ReAct sub-loop (the agent works
        until it claims done — R1), then a fresh EVALUATE run (a separate
        evaluator agent on a read-only sandbox in a never-shared session — R2,
        R3, R4), then asks the injected :class:`Verifier` to translate
        ``(build_result, eval_result)`` into a verdict. ``Passed`` ⇒ Success;
        ``Failed { reason }`` ⇒ inject ``reason`` into the build context (the
        same ``append_user_message`` path the Stop-block uses — R5/R6) and loop.

        Config fields read (both default ``None``):

        * ``config.verifier`` — the oracle. REQUIRED: ``None`` ⇒
          :class:`HaltReasonSelfVerifyMisconfigured` (D4, R11) — a typed halt,
          NOT a raise. Its ``max_iterations()`` (default 3) caps the round-trips
          (D3); ``max_stop_blocks`` does NOT enter the picture.
        * ``config.evaluator_agent`` — the evaluate-phase agent. Defaulting (D2):
          ``evaluator_agent or agent``, identical to ``planner_agent``. The
          read-only sandbox and the fresh session id are derived INTERNALLY.

        Terminal halts (peers, D4): :class:`HaltReasonSelfVerifyExhausted`
        (R7 — ran out of round-trips without a pass) and
        :class:`HaltReasonSelfVerifyMisconfigured` (R11). Budgets fold BOTH
        phases across ALL iterations (R8); build vs evaluate are distinguishable
        by their distinct session ids (R9). Mirrors Rust's ``run_self_verifying``.
        """
        build_session_id = task.session_id

        # D4/R11: a missing verifier is a typed halt, not a raise.
        verifier = self._config.verifier
        if verifier is None:
            result: RunResult = self._fail(
                HaltReasonSelfVerifyMisconfigured(
                    reason="SelfVerifying requires `config.verifier`, but it is None"
                ),
                build_session_id,
                AggregateUsage(),
                0,
            )
            await self._finalize_self_verifying(result)
            return result

        max_iterations = verifier.max_iterations()
        # Shared budget threaded across every build + evaluate sub-run (R8).
        carried = budget_used.model_copy(deep=True)
        # Cumulative usage across ALL build + evaluate runs of ALL iterations.
        total_usage = AggregateUsage()
        # The most recent verifier failure reason (for SelfVerifyExhausted).
        last_reason = ""

        from .verifier import VerifierInput, VerifierVerdictPassed

        for iteration in range(max_iterations):
            # Build phase (R1): bounded ReAct sub-loop carrying the shared budget.
            # The first iteration's seed instruction is already in
            # ``session_state``; later iterations have the prior verdict reason
            # injected as a user message (R6).
            build_cap = task.budget.max_turns if task.budget.max_turns is not None else 2**31 - 1
            build_result = await self._run_react_inner(
                task,
                build_cap,
                session_state,
                carried.model_copy(deep=True),
                # Sub-loops run with a suppressed sink (mirrors PlanExecute);
                # terminal observability is finalized by this strategy.
                None,
            )
            self._fold_usage(total_usage, carried, build_result)

            # A build run that paused / escalated is propagated up unchanged —
            # the caller must handle it before verification can resume.
            if isinstance(build_result, RunResultWaitingForHuman):
                return build_result
            if isinstance(build_result, RunResultEscalate):
                await self._finalize_self_verifying(build_result)
                return build_result

            # Evaluate phase (R2/R3/R4): a fresh evaluator RUN with a distinct
            # generated session id (R2/R9), a read-only sandbox derived internally
            # (R3), the evaluator agent (D2 defaulting), and the role-evaluator
            # chunk (R4).
            eval_result = await self._run_evaluate_phase(task, carried, total_usage)

            # Verdict.
            verdict = await verifier.verify(
                VerifierInput(
                    build_result=build_result,
                    eval_result=eval_result,
                    workspace=self._config.sandbox.workspace_root(),
                    iteration=iteration,
                )
            )
            if isinstance(verdict, VerifierVerdictPassed):
                # Reuse the build run's output/turns as the run's handle.
                if isinstance(build_result, RunResultSuccess):
                    output, turns = build_result.output, build_result.turns
                else:
                    output, turns = "", carried.turns
                result = RunResultSuccess(
                    output=output,
                    session_id=build_session_id,
                    usage=total_usage,
                    turns=turns,
                )
                await self._finalize_self_verifying(result)
                return result

            # R5/R6: Default-FAIL keeps looping; inject the reason into the build
            # context via the SAME path the Stop-block uses so the next build
            # iteration sees it.
            last_reason = verdict.reason
            await self._config.context_manager.append_user_message(session_state, verdict.reason)

        # R7: ran out of round-trips without a pass — clean exhaustion.
        result = self._fail(
            HaltReasonSelfVerifyExhausted(
                iterations=max_iterations,
                last_reason=last_reason,
            ),
            build_session_id,
            total_usage,
            carried.turns,
        )
        await self._finalize_self_verifying(result)
        return result

    async def _run_evaluate_phase(
        self,
        task: Task,
        carried: BudgetSnapshot,
        total_usage: AggregateUsage,
    ) -> RunResult:
        """Run the SelfVerifying evaluate phase (issue #61): a fresh evaluator RUN
        over a read-only sandbox in a never-shared session.

        Builds a child :class:`StandardHarness` from a copy of ``self._config``
        with the ``agent`` swapped to the evaluator agent (D2 defaulting) and the
        ``sandbox`` wrapped in a :class:`ReadOnlySandbox` (R3). The evaluator runs
        a fresh ReAct loop seeded with the ``role-evaluator`` chunk (R4,
        presence-only) plus a review directive, in a freshly generated session
        (R2/R9). Folds the evaluate run's usage into ``total_usage`` / ``carried``
        (R8) and returns its terminal :class:`RunResult`. Mirrors Rust's
        ``run_evaluate_phase``."""
        config = self._config
        # D2: evaluator agent defaulting — identical contract to ``planner_agent``.
        evaluator = config.evaluator_agent if config.evaluator_agent is not None else config.agent

        # R3: derive a read-only sandbox internally from the build sandbox.
        read_only_sandbox = ReadOnlySandbox(config.sandbox)

        # R2/R9: fresh, never-shared session id for the evaluate run.
        eval_session_id = new_session_id()

        # R4 (presence-only): prepend the role-evaluator chunk content (if the
        # configured provider supplies it) to the review directive.
        role_chunk = await self._role_evaluator_chunk()
        review = (
            "Review the work produced for the following task and report whether "
            "it is correct. You did NOT write this code; default to FAIL unless "
            f"you can confirm it is right.\n\nTask:\n{task.instruction}"
        )
        directive = f"{role_chunk}\n\n{review}" if role_chunk is not None else review

        eval_task = Task(
            id=new_task_id(),
            instruction=directive,
            session_id=eval_session_id,
            budget=task.budget,
            loop_strategy=LoopStrategyReAct(
                max_iterations=(
                    task.budget.max_turns if task.budget.max_turns is not None else 2**31 - 1
                )
            ),
        )

        # Child harness: copy the config, swap agent + sandbox. The copy shares
        # the same observability / storage seams so the evaluate run's spans land
        # in the SAME trace stream (distinguished by its distinct session id).
        eval_config = self._clone_config_with(agent=evaluator, sandbox=read_only_sandbox)
        eval_harness = StandardHarness(eval_config)

        eval_state = SessionState()
        await eval_config.context_manager.append_user_message(eval_state, directive)

        cap = task.budget.max_turns if task.budget.max_turns is not None else 2**31 - 1
        eval_result = await eval_harness._run_react(
            eval_task, cap, eval_state, BudgetSnapshot(), None
        )

        self._fold_usage(total_usage, carried, eval_result)
        return eval_result

    def _clone_config_with(self, *, agent: Agent, sandbox: SandboxProvider) -> HarnessConfig:
        """Copy ``self._config`` swapping only ``agent`` and ``sandbox`` (issue
        #61). Every other component (context manager, observability, storage,
        verifier, …) is shared by reference so the evaluate run's spans land in
        the same trace stream. Mirrors the Rust ``self.config.clone()`` + field
        swap in ``run_evaluate_phase``."""
        c = self._config
        return HarnessConfig(
            agent=agent,
            tool_registry=c.tool_registry,
            sandbox=sandbox,
            context_manager=c.context_manager,
            termination_policy=c.termination_policy,
            middleware=c.middleware,
            observability=c.observability,
            compaction_verifier=c.compaction_verifier,
            max_compaction_attempts=c.max_compaction_attempts,
            pricing=c.pricing,
            content_capture=c.content_capture,
            max_stop_blocks=c.max_stop_blocks,
            max_resets=c.max_resets,
            vcs_provider=c.vcs_provider,
            hooks=c.hooks,
            planner_agent=c.planner_agent,
            verifier=c.verifier,
            evaluator_agent=c.evaluator_agent,
            storage=c.storage,
            chunk_provider=c.chunk_provider,
            metric_evaluator=c.metric_evaluator,
        )

    async def _role_evaluator_chunk(self) -> str | None:
        """Look up the ``role-evaluator`` chunk content from the configured chunk
        provider (R4, presence-only). Returns ``None`` if the provider has no such
        chunk or fails to load. Mirrors Rust's ``role_evaluator_chunk``."""
        provider = self._config.chunk_provider
        if provider is None:
            return None
        try:
            chunks = await provider.load()
        except Exception:  # noqa: BLE001 — a broken provider must not abort the run
            return None
        for chunk in chunks:
            if chunk.id == "role-evaluator":
                return chunk.content
        return None

    @staticmethod
    def _fold_usage(total_usage: AggregateUsage, carried: BudgetSnapshot, r: RunResult) -> None:
        """Fold a sub-run's token usage / turn count into the cumulative
        ``total_usage`` and the shared ``carried`` budget snapshot (R8). Mirrors
        the PlanExecute budget fold and Rust's ``fold_usage``. ``carried.turns``
        becomes the max of the sub-run's absolute turn count (the build sub-loop
        gates on cumulative turns; the fresh-session evaluate run reports its own
        turns)."""
        if isinstance(r, RunResultWaitingForHuman):
            return
        usage = r.usage
        turns = r.turns
        total_usage.input_tokens += usage.input_tokens
        total_usage.output_tokens += usage.output_tokens
        total_usage.cache_read_tokens += usage.cache_read_tokens
        total_usage.cache_write_tokens += usage.cache_write_tokens
        total_usage.cost_usd += usage.cost_usd
        carried.input_tokens += usage.input_tokens
        carried.output_tokens += usage.output_tokens
        carried.turns = max(carried.turns, turns)

    async def _finalize_self_verifying(self, result: RunResult) -> None:
        """Finalize observability for a terminal SelfVerifying outcome (issue
        #61). Mirrors ``_finalize_plan_execute``: a ``WaitingForHuman`` pause is
        not terminal and is never flushed here. Mirrors Rust's
        ``finalize_self_verifying``."""
        if isinstance(result, RunResultSuccess):
            await self._finalize_observability(result.session_id, SessionOutcomeSuccess())
        elif isinstance(result, RunResultFailure):
            await self._finalize_observability(
                result.session_id,
                SessionOutcomeFailure(reason=result.reason.kind),
            )
        elif isinstance(result, RunResultEscalate):
            await self._finalize_observability(result.session_id, SessionOutcomeEscalated())
        # RunResultWaitingForHuman: not terminal, do not finalize.

    async def _run_hill_climbing(
        self,
        task: Task,
        direction: OptimizationDirection,
        max_stagnation: int | None,
        revert_on_no_improvement: bool,
        min_improvement_delta: float | None,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
    ) -> RunResult:
        """Drive the HillClimbing strategy (issue #60) — the iterative
        optimization loop.

        Iteration 0 is a PURE baseline: the evaluator is called with NO agent
        turn (Decision 5); its value becomes ``current_best`` and the row is
        recorded ``kept``. Iterations 1+ each run a bounded ReAct sub-run (one
        proposed change), then evaluate the metric and route the result through
        :func:`~spore_core.metric.should_keep` using the payload ``direction``
        (Decision 4): an improvement keeps the change (status ``kept``, best
        updated, stagnation reset); a non-improvement discards it (status
        ``discarded``), optionally runs ``git reset --hard HEAD`` through the
        sandbox (Decision 1), and increments the consecutive non-improvement
        counter. A :data:`~spore_core.metric.MetricError` (crash/timeout) counts
        as a non-improvement with an EMPTY metric value (Decision 3). When
        ``max_stagnation`` is set, the run halts with
        :class:`HaltReasonStagnationLimitReached` after that many consecutive
        non-improvements; otherwise it runs until the turn/budget cap. The
        harness writes the results TSV to ``.spore/results/{task_id}.tsv``
        (Decisions 2/3); it NEVER commits.

        Config fields read:

        * ``config.metric_evaluator`` — the scorer. REQUIRED: ``None`` ⇒
          :class:`HaltReasonHillClimbingMisconfigured` (Decision 6), a typed
          halt, never a raise. A baseline (iteration-0) evaluation that itself
          errors is ALSO :class:`HaltReasonHillClimbingMisconfigured` (Decision
          7) — there is no current best to climb from.

        Mirrors Rust's ``run_hill_climbing``.
        """
        from .metric import MetricResult, ResultsEntry, iteration_status_from_error, should_keep
        from .termination import SessionStateSnapshot

        session_id = task.session_id
        workspace_root = self._config.sandbox.workspace_root()

        # Decision 6: a missing evaluator is a typed halt, not a raise.
        evaluator = self._config.metric_evaluator
        if evaluator is None:
            result: RunResult = self._fail(
                HaltReasonHillClimbingMisconfigured(
                    reason="HillClimbing requires `config.metric_evaluator`, but it is None"
                ),
                session_id,
                AggregateUsage(),
                0,
            )
            await self._finalize_self_verifying(result)
            return result

        description = evaluator.description()
        # Per-iteration observability span counter.
        span_seq = 0
        # Cumulative usage + turns across ALL agent-turn iterations.
        total_usage = AggregateUsage()
        # Shared budget threaded across every agent sub-run.
        carried = budget_used.model_copy(deep=True)
        # The TSV rows, in iteration order.
        rows: list[ResultsEntry] = []

        # A snapshot for the evaluator. HillClimbing keeps no carried message
        # state of its own (each iteration is a fresh sub-run), so a default
        # SessionState is the right snapshot to hand the evaluator.
        snapshot = SessionStateSnapshot(
            session_id=session_id,
            task_id=task.id,
            state=SessionState(),
            workspace_root=workspace_root,
        )

        # ── Iteration 0: pure baseline. No agent turn (Decision 5).
        baseline = await evaluator.evaluate(self._config.sandbox, snapshot)
        if isinstance(baseline, MetricResult):
            current_best = baseline.value
            rows.append(
                ResultsEntry(
                    iteration=0,
                    commit_hash=self._hill_climbing_commit_hash(),
                    metric_value=baseline.value,
                    direction=direction,
                    status="kept",
                    duration=baseline.duration,
                    description=description,
                )
            )
            self._emit_hill_climbing_span(
                session_id, task.id, span_seq, 0, baseline.value, None, "kept", False
            )
            span_seq += 1
        else:
            # Decision 7: a baseline that cannot even be measured is a
            # misconfiguration of the experiment, not a non-improvement to climb
            # away from — there is no ``current_best`` to compare against. Record
            # the failed row, write the TSV, and halt.
            status = iteration_status_from_error(baseline)
            rows.append(
                ResultsEntry(
                    iteration=0,
                    commit_hash=self._hill_climbing_commit_hash(),
                    # Sentinel; excluded from the TSV (crashed/timeout ⇒ empty).
                    metric_value=float("nan"),
                    direction=direction,
                    status=status,
                    duration=0.0,
                    description=description,
                )
            )
            self._emit_hill_climbing_span(
                session_id, task.id, span_seq, 0, None, None, status, False
            )
            span_seq += 1
            await self._write_hill_climbing_tsv(workspace_root, task.id, rows)
            result = self._fail(
                HaltReasonHillClimbingMisconfigured(
                    reason=f"baseline evaluation failed: {baseline.kind}"
                ),
                session_id,
                total_usage,
                carried.turns,
            )
            await self._finalize_self_verifying(result)
            return result

        # Consecutive non-improvement counter (Decision-driven stagnation halt).
        stagnation = 0
        # The 0-based iteration index; agent turns begin at 1.
        iteration = 1
        turn_cap = task.budget.max_turns if task.budget.max_turns is not None else 2**31 - 1
        started_at = time.monotonic()

        while True:
            # Budget gate before the iteration's agent turn (mirrors run_react).
            if carried.turns >= turn_cap:
                break
            limit_type = self._budget_exceeded(task.budget, carried, started_at)
            if limit_type is not None:
                await self._write_hill_climbing_tsv(workspace_root, task.id, rows)
                result = self._fail(
                    HaltReasonBudgetExceeded(limit_type=limit_type),
                    session_id,
                    total_usage,
                    carried.turns,
                )
                await self._finalize_self_verifying(result)
                return result

            # ── One bounded agent turn proposes a change. The sub-run carries the
            #    shared budget so per-iteration turns count toward the cap. It
            #    seeds its OWN fresh session state with the instruction.
            iter_state = SessionState()
            await self._config.context_manager.append_user_message(iter_state, task.instruction)
            turn_result = await self._run_react_inner(
                task, turn_cap, iter_state, carried.model_copy(deep=True), None
            )
            self._fold_usage(total_usage, carried, turn_result)

            # A turn that paused / escalated is propagated up unchanged.
            if isinstance(turn_result, RunResultWaitingForHuman):
                return turn_result
            if isinstance(turn_result, RunResultEscalate):
                await self._write_hill_climbing_tsv(workspace_root, task.id, rows)
                await self._finalize_self_verifying(turn_result)
                return turn_result

            # ── Evaluate the metric after the change.
            eval_result = await evaluator.evaluate(self._config.sandbox, snapshot)
            if isinstance(eval_result, MetricResult):
                value = eval_result.value
                kept = should_keep(value, current_best, direction, min_improvement_delta)
                delta = (
                    (current_best - value) if direction == "minimize" else (value - current_best)
                )
                if kept:
                    current_best = value
                    stagnation = 0
                    rows.append(
                        ResultsEntry(
                            iteration=iteration,
                            commit_hash=self._hill_climbing_commit_hash(),
                            metric_value=value,
                            direction=direction,
                            status="kept",
                            duration=eval_result.duration,
                            description=description,
                        )
                    )
                    self._emit_hill_climbing_span(
                        session_id, task.id, span_seq, iteration, value, delta, "kept", False
                    )
                    span_seq += 1
                else:
                    # No improvement (Decision 1: optionally revert).
                    reverted = revert_on_no_improvement
                    if reverted:
                        await self._hill_climbing_revert()
                    stagnation += 1
                    rows.append(
                        ResultsEntry(
                            iteration=iteration,
                            commit_hash=self._hill_climbing_commit_hash(),
                            metric_value=value,
                            direction=direction,
                            status="discarded",
                            duration=eval_result.duration,
                            description=description,
                        )
                    )
                    self._emit_hill_climbing_span(
                        session_id,
                        task.id,
                        span_seq,
                        iteration,
                        value,
                        delta,
                        "discarded",
                        reverted,
                    )
                    span_seq += 1
            else:
                # Crash/timeout/etc.: counts as a non-improvement. Optionally
                # revert, increment stagnation, record an empty-metric row.
                status = iteration_status_from_error(eval_result)
                reverted = revert_on_no_improvement
                if reverted:
                    await self._hill_climbing_revert()
                stagnation += 1
                rows.append(
                    ResultsEntry(
                        iteration=iteration,
                        commit_hash=self._hill_climbing_commit_hash(),
                        # Sentinel; excluded from the TSV (crashed/timeout ⇒ empty).
                        metric_value=float("nan"),
                        direction=direction,
                        status=status,
                        duration=0.0,
                        description=description,
                    )
                )
                self._emit_hill_climbing_span(
                    session_id, task.id, span_seq, iteration, None, None, status, reverted
                )
                span_seq += 1

            # ── Stagnation halt (only when a cap is configured).
            if max_stagnation is not None and stagnation >= max_stagnation:
                await self._write_hill_climbing_tsv(workspace_root, task.id, rows)
                result = self._fail(
                    HaltReasonStagnationLimitReached(
                        iterations=stagnation,
                        best_metric=current_best,
                    ),
                    session_id,
                    total_usage,
                    carried.turns,
                )
                await self._finalize_self_verifying(result)
                return result

            iteration += 1

        # Budget/turn cap reached without a stagnation halt — clean budget halt.
        await self._write_hill_climbing_tsv(workspace_root, task.id, rows)
        result = self._fail(
            HaltReasonBudgetExceeded(limit_type="turns"),
            session_id,
            total_usage,
            carried.turns,
        )
        await self._finalize_self_verifying(result)
        return result

    async def _hill_climbing_revert(self) -> None:
        """Revert the working tree to current HEAD for a no-improvement iteration
        (issue #60, Decision 1). Runs ``git reset --hard HEAD`` THROUGH the
        sandbox; the harness NEVER spawns git directly. A sandbox rejection /
        non-zero exit is best-effort: the loop continues (the next agent turn
        re-derives state). Mirrors Rust's ``hill_climbing_revert``."""
        try:
            await self._config.sandbox.execute_command("git", ["reset", "--hard", "HEAD"])
        except Exception:  # noqa: BLE001 — revert is best-effort; never abort the loop
            pass

    @staticmethod
    def _hill_climbing_commit_hash() -> str | None:
        """Resolve the ``commit_hash`` recorded on a HillClimbing TSV row (issue
        #60, Decision 1). The harness never commits, so this is the EMPTY string
        (``None`` serialized as empty in the TSV) unless a ``VcsProvider`` is
        wired to supply a hash. v1 has no per-keep commit, so we always return
        ``None``. Mirrors Rust's ``hill_climbing_commit_hash``."""
        return None

    def _emit_hill_climbing_span(
        self,
        session_id: SessionId,
        task_id: TaskId,
        span_seq: int,
        iteration: int,
        metric_value: float | None,
        delta: float | None,
        status: str,
        reverted: bool,
    ) -> None:
        """Emit one fire-and-forget per-iteration observability span for a
        HillClimbing run (issue #60). No-op when no provider is configured.
        Mirrors Rust's ``emit_hill_climbing_span``."""
        obs = self._config.observability
        if obs is None:
            return
        emit_warn = getattr(obs, "emit_warn", None)
        if emit_warn is None:
            return
        base = SpanBase.new_root(
            new_span_id(f"{session_id}-hill-{span_seq}"),
            session_id,
            task_id,
            SpanKind.WARN,
            _now(),
        )
        emit_warn(
            WarnSpan(
                base=base,
                event=WarnEventHillClimbingIteration(
                    iteration=iteration,
                    metric_value=metric_value,
                    delta=delta,
                    status=status,
                    reverted=reverted,
                ),
            )
        )

    async def _write_hill_climbing_tsv(
        self,
        workspace_root: Path,
        task_id: TaskId,
        rows: list[Any],
    ) -> None:
        """Serialize the HillClimbing results log and write it to
        ``{workspace_root}/.spore/results/{task_id}.tsv`` (issue #60, Decisions
        2/3). Best-effort: a filesystem error is swallowed (the run outcome is
        authoritative, the TSV is a diagnostic artifact). Mirrors Rust's
        ``write_hill_climbing_tsv``."""
        body = self._render_hill_climbing_tsv(rows)
        try:
            dir_path = workspace_root / ".spore" / "results"
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / f"{task_id}.tsv").write_text(body, encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _render_hill_climbing_tsv(rows: list[Any]) -> str:
        """Render the HillClimbing results-log TSV body (issue #60, Decisions
        2/3). Pure function over the rows so the exact byte content is
        unit-testable and cross-language-comparable. Tab-separated, REQUIRED
        header, one row per iteration in ascending order. Floats use exactly 6
        decimal places for cross-language byte-identity. ``metric_value`` is the
        empty string on crashed/timeout rows; ``commit_hash`` is empty when no
        VCS. ``metadata`` is excluded. Trailing newline after every row.
        Mirrors Rust's ``render_hill_climbing_tsv``."""
        out = [
            "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription"
        ]
        for r in rows:
            # Decision 3: metric_value is EMPTY on crashed/timeout rows.
            if r.status in ("crashed", "timeout"):
                metric_value = ""
            else:
                metric_value = f"{r.metric_value:.6f}"
            commit_hash = r.commit_hash if r.commit_hash is not None else ""
            duration_secs = f"{r.duration:.6f}"
            out.append(
                f"{r.iteration}\t{commit_hash}\t{metric_value}\t{r.direction}\t"
                f"{r.status}\t{duration_secs}\t{r.description}"
            )
        return "\n".join(out) + "\n"

    async def _run_ralph(
        self,
        task: Task,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
    ) -> RunResult:
        """Drive the Ralph strategy (issue #58) — the multi-context-window
        continuation loop.

        The OUTER loop runs up to ``config.max_resets`` context windows (B3).
        Each window is a FRESH :class:`SessionState` (no message carryover —
        R2) re-seeded with the instruction plus the reloaded ``.spore/`` state
        (R3, B4 — progress + feature_list, no git), then a bounded ReAct
        sub-loop runs (the registered :class:`RalphStopHook` fires inside it on
        ``FinalResponse``). After the window, the SAME filesystem completion
        check the Stop hook reads (B1) decides reset vs. success: ``None`` ⇒
        done ⇒ Success; a reason ⇒ tasks remain ⇒ reset into the next window
        unless the cap is reached. Budgets fold across ALL windows (R6); each
        reset is independently traceable via a fresh session id (R7).

        Terminal: :class:`HaltReasonRalphCompletionUnmet` when ``max_resets``
        windows are exhausted with tasks still incomplete. Mirrors Rust's
        ``run_ralph``.
        """
        _ = on_stream  # sub-loops run with a suppressed sink (mirrors PlanExecute).
        workspace_root = self._config.sandbox.workspace_root()
        max_resets = max(self._config.max_resets, 1)
        # Ralph's incoming budget snapshot is irrelevant — each window is a fresh
        # start with its own per-window turn budget (the reset discards it).
        # Token/turn accounting is accumulated separately for terminal reporting.
        _ = budget_used

        # Cumulative usage + turns across ALL context windows (R6).
        total_usage = AggregateUsage()
        cumulative_turns = 0
        # The most recent incompletion reason (for RalphCompletionUnmet).
        last_reason = ".spore/progress.json missing"
        # Session id of the most recent context window (terminal accounting).
        last_session_id = task.session_id

        for iteration in range(max_resets):
            # R7: a fresh, distinct session id per context window so each reset
            # is independently traceable. Iteration 0 reuses the task's id.
            window_session_id = task.session_id if iteration == 0 else new_session_id()
            last_session_id = window_session_id

            # R2: a FRESH SessionState per window — no message carryover; the
            # window is re-seeded from scratch.
            session_state = SessionState()

            # Seed the instruction (R2), then R3: reload the deterministic
            # ``.spore/`` state and inject it so the fresh window knows what is
            # already done / still outstanding.
            await self._config.context_manager.append_user_message(session_state, task.instruction)
            reload = _ralph_reload_context(workspace_root)
            if reload is not None:
                await self._config.context_manager.append_user_message(session_state, reload)
            # R3 (issue #58 v2): when a ``VcsProvider`` is wired, ALSO reload git
            # history and inject it as a delimited "Recent VCS history:" section,
            # exactly as the ``.spore/`` reload content is injected. When the
            # provider is ``None`` (the default), this section is omitted
            # entirely — Ralph's reloaded context is then byte-for-byte the v1
            # behavior (the B4→None decision).
            vcs = self._config.vcs_provider
            if vcs is not None:
                args = VcsLogArgs(max_entries=20)
                try:
                    log = await vcs.log(args)
                except VcsError:
                    log = ""
                trimmed = log.strip()
                if trimmed:
                    block = f"Recent VCS history:\n{trimmed}"
                    await self._config.context_manager.append_user_message(session_state, block)

            window_task = Task(
                id=task.id,
                instruction=task.instruction,
                session_id=window_session_id,
                budget=task.budget,
                loop_strategy=task.loop_strategy,
            )
            # FRESH per-window budget: the context-window reset resets the turn
            # budget too. Token fold is accumulated separately via ``total_usage``.
            window_cap = task.budget.max_turns if task.budget.max_turns is not None else 2**31 - 1
            carried = BudgetSnapshot()
            window_result = await self._run_react_inner(
                window_task,
                window_cap,
                session_state,
                carried,
                None,
            )
            self._fold_usage(total_usage, carried, window_result)
            cumulative_turns += carried.turns

            # A window that paused / escalated is propagated up unchanged.
            if isinstance(window_result, RunResultWaitingForHuman):
                return window_result
            if isinstance(window_result, RunResultEscalate):
                await self._finalize_self_verifying(window_result)
                return window_result

            # External completion check (B1): consult the SAME filesystem state
            # the Stop hook reads. ``None`` ⇒ done ⇒ Success; a reason ⇒ tasks
            # remain ⇒ reset into the next window unless the cap is reached.
            reason = _ralph_completion_status(workspace_root)
            if reason is None:
                output = window_result.output if isinstance(window_result, RunResultSuccess) else ""
                result: RunResult = RunResultSuccess(
                    output=output,
                    session_id=window_session_id,
                    usage=total_usage,
                    turns=cumulative_turns,
                )
                await self._finalize_self_verifying(result)
                return result
            last_reason = reason

        # Ran out of context-window resets without completion.
        result = self._fail(
            HaltReasonRalphCompletionUnmet(
                iterations=max_resets,
                last_reason=last_reason,
            ),
            last_session_id,
            total_usage,
            cumulative_turns,
        )
        await self._finalize_self_verifying(result)
        return result

    async def _persist_task_list(
        self,
        session_id: SessionId,
        task_list: object,
    ) -> None:
        """Persist the parsed :class:`TaskList` for the run (Q4).

        The write goes through the :class:`RunStore` seam under
        ``TASK_LIST_EXTRAS_KEY``; the #71 sandbox-filesystem path
        (``.spore/task_list.json``) is intentionally NOT used — one source of
        truth. The RunStore write is the single source of truth (#76 removed the
        redundant ``SessionState.extras`` mirror). Storage failures are
        swallowed: a successful plan must not be lost to a storage hiccup (the
        default no-op provider never fails). Mirrors Rust's ``persist_task_list``.
        """
        from .tasklist import TASK_LIST_EXTRAS_KEY, TaskList

        assert isinstance(task_list, TaskList)
        value = task_list.to_dict()
        try:
            await self._config.storage.run().put(session_id, TASK_LIST_EXTRAS_KEY, value)
        except Exception:  # noqa: BLE001 — a storage hiccup must not lose the plan
            pass

    async def _run_execute_phase(
        self,
        task: Task,
        session_state: SessionState,
        task_list: object,
        carried: BudgetSnapshot,
        plan_usage: AggregateUsage,
        on_stream: StreamSink | None,
    ) -> RunResult:
        """Drive the PlanExecute execute phase (issue #59), draining ``task_list``.

        Per Q1 each task gets its own bounded, fully-isolated, SEQUENTIAL ReAct
        sub-loop. The per-task turn cap is derived at the START of each step from
        the shared budget: ``per_task_turns = remaining_turns // remaining_tasks``
        floored at 1 (``remaining_tasks`` counts the not-yet-started tasks
        including the current one). The shared budget (``carried``) is threaded
        through every step so early tasks cannot starve later ones and the global
        budget stays the hard stop.

        Before each step the task is marked ``in_progress`` (and ``completed``
        after), the list is re-persisted (Q4), and ``OnTaskAdvance`` fires with
        the correct ``task_index`` / ``total_tasks`` (the hook may rewrite the
        step instruction). Q2: on success ``output`` is the LAST completed step's
        final text. Q5: a step that errors/blocks aborts the run with
        :class:`HaltReasonStepFailed`. Mirrors Rust's ``run_execute_phase``.
        """
        from .hooks import OnTaskAdvanceContext
        from .tasklist import TaskList, TaskStatus

        assert isinstance(task_list, TaskList)
        config = self._config
        session_id = task.session_id
        total_tasks = len(task_list.tasks)
        # Cumulative usage across the plan turn + every execute step (Q1).
        total_usage = plan_usage.model_copy(deep=True)
        # Q2: the success handle is the LAST completed step's final text.
        last_output = ""
        # Global turn cap (the hard stop). ``None`` ⇒ no global turn ceiling.
        global_max_turns = task.budget.max_turns

        for index in range(total_tasks):
            task_id = task_list.tasks[index].id
            instruction = task_list.tasks[index].description

            # Q1: per-task turn allocation, derived at the START of this step.
            # remaining_tasks = not-yet-started tasks including this one.
            remaining_tasks = total_tasks - index
            if global_max_turns is not None:
                remaining_turns = max(0, global_max_turns - carried.turns)
                per_task_turns = max(1, remaining_turns // remaining_tasks)
                # The sub-loop cap is RELATIVE to the carried turns:
                # _run_react_inner gates on cumulative ``budget_used.turns``, so a
                # per-task cap of K means "stop K turns from now" while the global
                # budget (carried forward) remains the hard stop.
                sub_loop_cap = carried.turns + per_task_turns
            else:
                # No global turn cap: each step's sub-loop is bounded only by the
                # other (token / wall / cost) budget gates.
                sub_loop_cap = 2**31 - 1

            # Mark InProgress (pending -> in_progress) and re-persist (Q4).
            task_list.update(task_id, TaskStatus.IN_PROGRESS)
            await self._persist_task_list(session_id, task_list)

            # Fire OnTaskAdvance (pre, mutable). The hook may rewrite the step's
            # instruction via the carried Task; the (possibly mutated) instruction
            # seeds the sub-loop.
            step_task = Task(
                id=task.id,
                instruction=instruction,
                session_id=session_id,
                budget=task.budget,
                loop_strategy=task.loop_strategy,
            )
            if config.hooks is not None:
                ctx = OnTaskAdvanceContext(
                    session_id=session_id,
                    task=step_task,
                    task_index=index,
                    total_tasks=total_tasks,
                )
                try:
                    await config.hooks.fire(ctx)
                except Exception:  # noqa: BLE001 — a broken hook must not abort the run
                    pass
                # The chain threads mutations through ``ctx.task`` in place.
                step_task = ctx.task

            # Seed the step instruction as a user message, then run the bounded
            # ReAct sub-loop carrying the shared budget (Q1).
            await config.context_manager.append_user_message(session_state, step_task.instruction)

            sub_result = await self._run_react_inner(
                step_task,
                sub_loop_cap,
                session_state,
                carried.model_copy(deep=True),
                None,
            )

            if isinstance(sub_result, RunResultSuccess):
                # Carry the shared budget forward (Q1): cumulative turns are the
                # sub-loop's absolute count; fold in its token usage.
                carried.turns = sub_result.turns
                carried.input_tokens += sub_result.usage.input_tokens
                carried.output_tokens += sub_result.usage.output_tokens
                total_usage.input_tokens += sub_result.usage.input_tokens
                total_usage.output_tokens += sub_result.usage.output_tokens
                total_usage.cache_read_tokens += sub_result.usage.cache_read_tokens
                total_usage.cache_write_tokens += sub_result.usage.cache_write_tokens
                total_usage.cost_usd += sub_result.usage.cost_usd
                last_output = sub_result.output

                # Mark Completed and re-persist (Q4).
                task_list.complete(task_id)
                await self._persist_task_list(session_id, task_list)
                # Surface the completed step's final text at the parent-visible
                # step boundary (the sub-loop runs with a suppressed sink).
                self._emit(on_stream, StreamFinalResponse(content=last_output))

            elif isinstance(sub_result, RunResultFailure):
                # Q5: any non-success step aborts the whole run. A budget halt
                # surfaces as BudgetExceeded (mid-execute exhaustion); other
                # failures surface as StepFailed carrying the step context.
                total_usage.input_tokens += sub_result.usage.input_tokens
                total_usage.output_tokens += sub_result.usage.output_tokens
                total_usage.cache_read_tokens += sub_result.usage.cache_read_tokens
                total_usage.cache_write_tokens += sub_result.usage.cache_write_tokens
                total_usage.cost_usd += sub_result.usage.cost_usd

                task_list.update(task_id, TaskStatus.BLOCKED)
                await self._persist_task_list(session_id, task_list)

                if isinstance(sub_result.reason, HaltReasonBudgetExceeded):
                    terminal_reason: HaltReason = sub_result.reason
                else:
                    terminal_reason = HaltReasonStepFailed(
                        task_index=index,
                        task=task_list.tasks[index].description,
                        reason=repr(sub_result.reason),
                    )
                return self._fail(
                    terminal_reason,
                    session_id,
                    total_usage,
                    sub_result.turns,
                )

            else:
                # A step surfacing to a human pauses the whole run; a step
                # escalating (issue #80) terminates it cleanly. Either way,
                # propagate the sub-result up unchanged.
                return sub_result

        # Q2: success output is the LAST completed step's final text.
        return RunResultSuccess(
            output=last_output,
            session_id=session_id,
            usage=total_usage,
            turns=carried.turns,
        )

    async def _run_plan_phase(
        self,
        task: Task,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
    ) -> _PlanPhaseOutcome | RunResultFailure:
        """Run the one-shot PlanExecute plan phase (issue #70).

        Selects the planner agent (Q1: ``config.planner_agent`` if set, else the
        default agent), seeds a planning directive as a user message, runs
        EXACTLY ONE constrained turn (R1), expects a ``FinalResponse`` (a tool
        call is a planning failure — R2 — never a dispatch loop), captures the
        response via :func:`spore_core.plan.capture_plan_artifact` (R3), fires
        ``OnPlanCreated`` (which may rewrite the artifact — R11), stores the
        result in ``extras["plan_execute"]`` (R4), emits the turn span (R8), and
        counts the turn against the shared budget (R7). A budget exhausted
        before the turn returns a budget-exceeded :class:`RunResultFailure` with
        no artifact stored (R10).

        On success returns a :class:`_PlanPhaseOutcome`; on any failure returns
        the terminal :class:`RunResultFailure` to propagate.
        """
        from .plan import PLAN_EXECUTE_EXTRAS_KEY, PlanPhaseError, capture_plan_artifact

        config = self._config
        session_id = task.session_id
        started_at = time.monotonic()
        usage = AggregateUsage()

        # R10: Layer-1 budget gate BEFORE the plan turn. Mirrors _run_react_inner.
        effective_turn_cap = max(task.budget.max_turns, 1) if task.budget.max_turns else 1
        # ``max_turns is None`` ⇒ no explicit cap; one plan turn always allowed.
        if task.budget.max_turns is not None and budget_used.turns >= effective_turn_cap:
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

        # Q1: select the planner agent (alternate if configured, else default).
        planner = config.planner_agent if config.planner_agent is not None else config.agent

        # Seed the planning directive as a user message (reuse ContextManager).
        directive = (
            "Produce a step-by-step plan for the following task. Respond with a "
            'single JSON object: {"tasks": [<ordered step strings>], '
            '"rationale": <string>}.\n\nTask:\n' + task.instruction
        )
        await config.context_manager.append_user_message(session_state, directive)

        # Assemble + invoke the planner for exactly ONE turn (R1).
        context = await config.context_manager.assemble(session_state, task)
        self._emit(on_stream, StreamTurnStart(turn=budget_used.turns + 1))
        turn_started_at = _now()
        turn_clock = time.monotonic()
        result: TurnResult = await planner.turn(context)
        budget_used.turns += 1  # R7: the plan turn counts against the budget.

        # R8: emit exactly one turn span for the plan turn. Mirrors the metrics
        # path of _run_react_inner; content capture is intentionally omitted (the
        # plan turn carries no tool calls and #64 content capture is wired in the
        # ReAct loop only).
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
        turn_base.finish(_now(), turn_status, int((time.monotonic() - turn_clock) * 1000))
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
                    output_text=None,
                    tool_calls=None,
                    input_messages=None,
                )
            )
        self._emit(on_stream, StreamTurnEnd(turn=budget_used.turns))

        # Classify the one-shot turn. R2: a tool call is a planning failure,
        # NOT a dispatch loop.
        if isinstance(result, FinalResponse):
            usage.add_turn(result.usage)
            budget_used.input_tokens += result.usage.input_tokens
            budget_used.output_tokens += result.usage.output_tokens
            final_text = result.content
        elif isinstance(result, ToolCallRequested):
            usage.add_turn(result.usage)
            return self._fail(
                HaltReasonPlanPhaseFailed(
                    error=PlanPhaseErrorPayload(
                        kind="planning_turn_failed",
                        message="planner requested a tool call in the one-shot plan turn",
                    ),
                ),
                session_id,
                usage,
                budget_used.turns,
            )
        else:  # TurnError
            if result.usage is not None:
                usage.add_turn(result.usage)
            return self._fail(
                HaltReasonAgentError(error=result.error),
                session_id,
                usage,
                budget_used.turns,
            )

        # R3: capture the artifact from the response text.
        try:
            artifact = capture_plan_artifact(final_text)
        except PlanPhaseError as e:
            return self._fail(
                HaltReasonPlanPhaseFailed(
                    error=PlanPhaseErrorPayload(kind=e.kind, message=e.message),
                ),
                session_id,
                usage,
                budget_used.turns,
            )

        # R11: fire OnPlanCreated synchronously; the hook may rewrite the
        # artifact. The stored artifact reflects any mutation. Errors are
        # non-fatal: an observability/handler error must not lose a
        # successfully-captured plan, so the (possibly mutated) artifact is
        # still stored.
        if config.hooks is not None:
            from .hooks import OnPlanCreatedContext

            ctx = OnPlanCreatedContext(session_id=session_id, plan=artifact)
            try:
                await config.hooks.fire(ctx)
            except Exception:  # noqa: BLE001 — a broken hook must not lose the plan
                pass
            # The chain threads mutations through ``ctx.plan`` in place.
            artifact = ctx.plan

        # R4: persist the produced artifact to the RunStore seam under
        # PLAN_EXECUTE_EXTRAS_KEY (#76 — the durable single source of truth; no
        # longer mirrored into SessionState.extras). The JSON-safe object matches
        # Rust's ``serde_json::to_value(&artifact)``: ``{"tasks": [...],
        # "rationale": "..."}``. Storage failures are swallowed (matching the
        # execute-phase persist): a successfully-captured plan must not be lost
        # to a storage hiccup (the default no-op provider never fails).
        try:
            await self._config.storage.run().put(
                session_id, PLAN_EXECUTE_EXTRAS_KEY, artifact.model_dump(mode="json")
            )
        except Exception:  # noqa: BLE001 — a storage hiccup must not lose the plan
            pass

        return _PlanPhaseOutcome(artifact=artifact, usage=usage, turns=budget_used.turns)

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
        elif isinstance(result, RunResultEscalate):
            # An escalation is a clean, intentional terminal outcome (issue #80)
            # — finalize with the dedicated ``Escalated`` outcome, NOT
            # ``Partial``. Contrast with ``WaitingForHuman``, which is a pause,
            # not terminal, and is never finalized here.
            await self._finalize_observability(result.session_id, SessionOutcomeEscalated())
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
        # Resolve the effective tool registry once per turn-loop window (all
        # strategies funnel through here). Bridges catalogue tools per-run.
        tool_registry = self._effective_tool_registry(session_id)
        started_at = time.monotonic()
        usage = AggregateUsage()
        # Monotonic per-run span counter for tool-call span ids, and the most
        # recent turn span base — the parent for that turn's tool-call spans.
        span_seq = 0
        current_turn_base: SpanBase | None = None
        # Per-run Stop-hook block counter (issue #69, R14). Resets on every
        # run() — resume starts fresh. After ``max_stop_blocks`` consecutive
        # blocks the loop terminates anyway.
        stop_blocks = 0
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
            # Fill the model-facing tool list from the effective registry only
            # when the context manager rendered none (the compaction adapter
            # does), so a context manager that deliberately sets a phase-specific
            # tool subset is preserved.
            if not context.tools:
                context.tools = tool_registry.schemas()
            # Prepend the configured operating system prompt (issue #91). The
            # standard compaction adapter renders none, so without this the model
            # gets only the task and no guidance. Guard against duplicates so a
            # context manager that already leads with a System message (or a
            # resumed/seeded session) isn't given two.
            if config.system_prompt is not None and not (
                context.messages and context.messages[0].role == Role.SYSTEM
            ):
                context.messages.insert(
                    0,
                    Message(
                        role=Role.SYSTEM,
                        content=TextContent(text=config.system_prompt),
                    ),
                )
            self._emit(on_stream, StreamTurnStart(turn=budget_used.turns + 1))
            turn_started_at = _now()
            turn_clock = time.monotonic()
            # LLM-native content capture (issue #64): snapshot the assembled
            # INPUT messages (the full prompt the model saw) BEFORE the agent
            # turn call. Zero work when the guard is off.
            input_messages: list[GenAiMessage] | None = None
            if config.content_capture.enabled:
                input_messages = _capture_input_messages(
                    context.messages, config.content_capture.max_field_len
                )
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
                # LLM-native content capture (issue #64): output text + requested
                # tool calls, ONLY when the guard is enabled. Decision 4: the turn
                # span carries output + tool calls; no assembled input history.
                cc = config.content_capture
                output_text: GenAiMessage | None = None
                turn_tool_calls: list[ToolCallContent] | None = None
                if cc.enabled:
                    if isinstance(result, FinalResponse):
                        content, content_truncated = truncate_field(
                            result.content, cc.max_field_len
                        )
                        output_text = GenAiMessage(
                            role=GenAiRole.ASSISTANT,
                            content=content,
                            truncated=content_truncated,
                        )
                    elif isinstance(result, ToolCallRequested):
                        turn_tool_calls = [
                            _capture_tool_call_args(c, cc.max_field_len) for c in result.calls
                        ]
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
                        output_text=output_text,
                        tool_calls=turn_tool_calls,
                        input_messages=input_messages,
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

                # Record the assistant's final text in history so a continued
                # session reflects what the agent said. Structural managers do
                # not inherit the Protocol default, so probe via ``getattr``.
                appender = getattr(config.context_manager, "append_assistant_message", None)
                if appender is not None:
                    await appender(
                        session_state,
                        Message(role=Role.ASSISTANT, content=TextContent(text=result.content)),
                    )

                # Stop hook (issue #69). The strategy believes the task is done;
                # fire registered Stop hooks synchronously. If any blocks (and
                # we are under ``max_stop_blocks``), inject the reason as a user
                # message and continue the loop instead of terminating.
                stop_reason = await self._fire_stop_hooks(
                    session_id,
                    task,
                    budget_used.turns,
                    result.content,
                    stop_blocks,
                )
                if stop_reason is not None:
                    stop_blocks += 1
                    await config.context_manager.append_user_message(session_state, stop_reason)
                    continue

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
                    if tool_registry.is_always_halt(c.name):
                        return self._fail(
                            HaltReasonUnrecoverableToolError(
                                tool=c.name,
                                error="tool is annotated always_halt",
                            ),
                            session_id,
                            usage,
                            budget_used.turns,
                        )

                # Record the assistant's turn (the tool calls the model
                # requested) as soon as the calls are known — BEFORE the
                # BeforeTool middleware (which may pause via SurfaceToHuman) and
                # before any tool result. This keeps the conversation well-formed
                # (assistant tool_use precedes its tool result) on every path,
                # including human-in-the-loop resume, so the resume path never
                # has to append it. The recorded turn reflects the model's
                # original request; a middleware or human modification changes
                # only what is dispatched. Structural managers do not inherit the
                # Protocol default, so probe via ``getattr``.
                appender = getattr(config.context_manager, "append_assistant_message", None)
                if appender is not None:
                    for call in result.calls:
                        await appender(
                            session_state,
                            Message(
                                role=Role.ASSISTANT,
                                content=MsgToolCallContent(
                                    id=call.id, name=call.name, input=call.input
                                ),
                            ),
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
                    output = await tool_registry.dispatch(call)

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

                    # Escalation propagation (issue #80): a tool requests a
                    # structural state change. The harness is a pure
                    # intermediary — it does NOT act on the signal. It
                    # terminates cleanly, preserves session state for a possible
                    # resume, and returns :class:`RunResultEscalate`. The
                    # escalation is a control signal, NOT a conversation turn: it
                    # is never appended to message history (we ``return`` before
                    # the ``append_tool_result`` below), and the remaining tool
                    # calls in this batch are preserved into
                    # ``pending_tool_calls`` (mirroring WaitingForHuman). The
                    # signal is NOT stored in ``PausedState`` (``human_request``
                    # is ``None``), so it is discarded on resume — the harness
                    # never re-acts on it.
                    if isinstance(output, ToolOutputEscalate):
                        remaining = calls[i + 1 :]
                        turns = budget_used.turns
                        paused = PausedState(
                            session_id=session_id,
                            task_id=task.id,
                            turn_number=budget_used.turns,
                            session_state=session_state,
                            pending_tool_calls=remaining,
                            approved_results=approved_results,
                            human_request=None,
                            task=task,
                            budget_used=budget_used,
                            child_state=None,
                        )
                        return RunResultEscalate(
                            signal=output.signal,
                            state=paused,
                            session_id=session_id,
                            usage=usage,
                            turns=turns,
                        )

                    # Clarification pause (issue #81, Q4b): a tool (e.g.
                    # ``ask_user_question``) needs a human answer before it can
                    # produce a result. UNLIKE the subagent ``WaitingForHuman``
                    # path, there is NO ``ChildPausedState``: the loop builds a
                    # :class:`PausedState` directly with ``human_request`` set to
                    # :class:`HumanRequestClarification`. The CLARIFYING call
                    # itself is preserved as the HEAD of ``pending_tool_calls``
                    # (followed by the remaining batch) so that, on resume, the
                    # human's answer is injected as the tool RESULT for that call.
                    if isinstance(output, ToolOutputAwaitingClarification):
                        pending = [call, *calls[i + 1 :]]
                        request = HumanRequestClarification(
                            question=output.question, options=output.options
                        )
                        paused = PausedState(
                            session_id=session_id,
                            task_id=task.id,
                            turn_number=budget_used.turns,
                            session_state=session_state,
                            pending_tool_calls=pending,
                            approved_results=approved_results,
                            human_request=request,
                            task=task,
                            budget_used=budget_used,
                            child_state=None,
                        )
                        return RunResultWaitingForHuman(state=paused, request=request)

                    # SendMessage (issue #81): the ``send_message`` tool surfaces
                    # an out-of-band message to the user. The loop emits a
                    # :class:`StreamUserMessage` rather than collapsing the
                    # content into a normal tool result, then records a minimal
                    # success result so the loop continues.
                    if call.name == SEND_MESSAGE_TOOL_NAME and isinstance(
                        output, ToolOutputSuccess
                    ):
                        self._emit(on_stream, StreamUserMessage(content=output.content))

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
                        # LLM-native content capture (issue #64): tool args + tool
                        # result, ONLY when the guard is enabled.
                        cc = config.content_capture
                        tool_args_content: ToolCallContent | None = None
                        tool_result_content: ToolResultContent | None = None
                        if cc.enabled:
                            tool_args_content = _capture_tool_call_args(call, cc.max_field_len)
                            if isinstance(output, ToolOutputSuccess):
                                rc, rt = truncate_field(output.content, cc.max_field_len)
                                tool_result_content = ToolResultContent(content=rc, truncated=rt)
                            elif isinstance(output, ToolOutputError):
                                rc, rt = truncate_field(output.message, cc.max_field_len)
                                tool_result_content = ToolResultContent(content=rc, truncated=rt)
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
                                arguments=tool_args_content,
                                result=tool_result_content,
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

                # Compaction (issue #46): after tool results are appended, before
                # the loop restarts — matches the concepts-doc loop diagram's
                # "compact if should_compact()" placement. Runs the
                # verify→retry→warn loop; never halts the run.
                if config.context_manager.should_compact(session_state):
                    span_seq = await self._run_compaction(
                        session_state,
                        session_id,
                        task.id,
                        span_seq,
                        usage,
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

    # ---- compaction loop (issue #46/#29) ----------------------------

    async def _run_compaction(
        self,
        session_state: SessionState,
        session_id: SessionId,
        task_id: TaskId,
        span_seq: int,
        usage: AggregateUsage,
    ) -> int:
        """Run the post-compaction verify→retry→warn loop (issue #46/#29).

        Drives one compaction turn through the agent, verifies the summary, and
        either accepts it, retries with the missing items injected, or — after
        ``max_compaction_attempts`` — emits a warn event and accepts the summary
        anyway. A blocked compaction is worse than an imperfect one, so this
        method NEVER raises or halts the run; the worst case is an
        accepted-anyway summary plus one warn span.

        Token usage from compaction turns folds into the run-level
        :class:`AggregateUsage`; each accepted summary is surfaced as a
        ``Compaction`` :class:`ContextSpan`. The
        ``compaction_verification_failures`` metric is derived from the emitted
        :class:`WarnSpan`. Returns the advanced ``span_seq``.
        """
        config = self._config
        turn = config.context_manager.prepare_compaction_turn(session_state)
        if turn is None:
            # Nothing to compact (e.g. history shorter than preserve window).
            return span_seq
        tokens_before = turn.verification_state.token_budget_used
        max_attempts = max(1, config.max_compaction_attempts)
        attempt = 0

        while True:
            attempt += 1
            # Run one compaction turn through the agent to produce a summary.
            result = await config.agent.turn(turn.context)
            if isinstance(result, FinalResponse):
                usage.add_turn(result.usage)
                summary = result.content
            elif isinstance(result, ToolCallRequested):
                # A compaction turn is expected to yield a summary, not a tool
                # call. Treat the (empty) response as the summary so
                # verification can run and the loop terminates predictably.
                usage.add_turn(result.usage)
                summary = ""
            else:  # TurnError
                if result.usage is not None:
                    usage.add_turn(result.usage)
                summary = ""

            verification = config.compaction_verifier.verify(
                summary, turn.preserve_hints, turn.verification_state
            )

            if verification.passed:
                return self._accept_compaction(
                    session_state,
                    summary,
                    turn.messages_removed,
                    tokens_before,
                    session_id,
                    task_id,
                    span_seq,
                )

            if attempt < max_attempts:
                # Inject the missing items and retry. Use the manager's override
                # if it provides one; otherwise the standard default body.
                inject = getattr(config.context_manager, "inject_missing_items", None)
                if inject is not None:
                    inject(turn.context, verification.missing_items)
                else:
                    _default_inject_missing_items(turn.context, verification.missing_items)
                continue

            # Exhausted attempts: warn, then accept anyway.
            if config.observability is not None:
                base = SpanBase.new_root(
                    new_span_id(f"{session_id}-warn-{span_seq}"),
                    session_id,
                    task_id,
                    SpanKind.WARN,
                    _now(),
                )
                warn_span = WarnSpan(
                    base=base,
                    event=WarnEventCompactionVerificationFailed(
                        missing_items=list(verification.missing_items),
                        accepted_anyway=True,
                    ),
                )
                # ``emit_warn`` carries a Protocol default no-op (issue #46), but
                # structural providers predating #46 may not define it at all —
                # fall back to the default no-op so they keep working (W4).
                emit_warn = getattr(config.observability, "emit_warn", None)
                if emit_warn is not None:
                    emit_warn(warn_span)
                span_seq += 1
            return self._accept_compaction(
                session_state,
                summary,
                turn.messages_removed,
                tokens_before,
                session_id,
                task_id,
                span_seq,
            )

    def _accept_compaction(
        self,
        session_state: SessionState,
        summary: str,
        messages_removed: int,
        tokens_before: int,
        session_id: SessionId,
        task_id: TaskId,
        span_seq: int,
    ) -> int:
        """Apply an accepted summary and emit the ``Compaction`` context span.
        Returns the advanced ``span_seq``."""
        config = self._config
        config.context_manager.apply_compaction(session_state, summary)

        # Real token accounting (#57 Known Deviation #2): read the
        # post-compaction budget back through the optional ``token_budget_used``
        # seam so the span reports what was actually reclaimed instead of zero.
        # Structural managers do not inherit the Protocol default, so probe via
        # ``getattr`` and fall back to the pre-compaction estimate when absent.
        tokens_after = tokens_before
        budget_seam = getattr(config.context_manager, "token_budget_used", None)
        if budget_seam is not None:
            after = budget_seam(session_state)
            if after is not None:
                tokens_after = after
        tokens_reclaimed = max(0, tokens_before - tokens_after)

        if config.observability is not None:
            base = SpanBase.new_root(
                new_span_id(f"{session_id}-compaction-{span_seq}"),
                session_id,
                task_id,
                SpanKind.COMPACTION,
                _now(),
            )
            util_before = 0.0
            util_after = 0.0
            config.observability.emit_context(
                ContextSpan(
                    base=base,
                    operation=ContextOperationCompaction(
                        messages_removed=messages_removed,
                        tokens_reclaimed=tokens_reclaimed,
                    ),
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    utilization_before=util_before,
                    utilization_after=util_after,
                )
            )
            span_seq += 1
        return span_seq


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


class FixtureVcsProvider:
    """Deterministic :class:`VcsProvider` double for tests and fixture replay
    (issue #58 v2). Returns pre-loaded strings VERBATIM with no process
    spawning, so multi-context-window Ralph continuation can be exercised
    hermetically. :meth:`log` ignores its :class:`VcsLogArgs` and yields
    ``log_output``; :meth:`status` yields ``status_output``. Mirrors Rust's
    ``FixtureVcsProvider``."""

    def __init__(self, log_output: str, status_output: str) -> None:
        self._log_output = log_output
        self._status_output = status_output

    async def log(self, args: VcsLogArgs) -> str:
        return self._log_output

    async def status(self) -> str:
        return self._status_output


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
# Deferred forward-ref resolution (issue #80)
# ============================================================================
# ``PlanArtifact`` (from :mod:`spore_core.hooks`) and ``Mode`` (from
# :mod:`spore_core.prompt_chunk_registry`) live in modules that import
# ``HarnessConfig`` / ``PausedState`` from THIS module, so they can only be
# imported after every class above is defined — hence at the very bottom. The
# escalation types reach them directly, and ``ToolOutputWaitingForHuman`` /
# ``ChildPausedState`` / ``PausedState`` reach them transitively through the
# ``ToolOutput`` union, so all of these models are rebuilt here.
from .hooks import PlanArtifact  # noqa: E402
from .prompt_chunk_registry import Mode  # noqa: E402

# ``ContextErrorModel`` lives in :mod:`spore_core.context`, which imports this
# module at its top — a top-level import here would be circular. It is imported
# after this module is otherwise defined so ``HaltReasonContextError``'s
# forward-ref ``error`` field can be resolved by the rebuild below.
from .context import ContextErrorModel  # noqa: E402, F401

HaltReasonContextError.model_rebuild()
HarnessSignalExitPlanMode.model_rebuild()
HarnessSignalSwitchMode.model_rebuild()
ToolOutputEscalate.model_rebuild()
ToolOutputWaitingForHuman.model_rebuild()
RunResultEscalate.model_rebuild()
ChildPausedState.model_rebuild()
PausedState.model_rebuild()


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
    "HaltReasonContextError",
    "HaltReasonEmptyPlan",
    "HaltReasonHumanHalted",
    "HaltReasonMiddlewareHalt",
    "HaltReasonSandboxViolation",
    "HaltReasonStagnationLimitReached",
    "HaltReasonStepFailed",
    "HaltReasonSelfVerifyExhausted",
    "HaltReasonSelfVerifyMisconfigured",
    "HaltReasonHillClimbingMisconfigured",
    "HaltReasonRalphCompletionUnmet",
    "HaltReasonStrategyNotYetImplemented",
    "HaltReasonTerminationPolicyHalt",
    "HaltReasonUnrecoverableToolError",
    "Harness",
    "HarnessBuilder",
    "HarnessConfig",
    "HarnessRunOptions",
    "HarnessSignal",
    "HarnessSignalAbort",
    "HarnessSignalEnterPlanMode",
    "HarnessSignalExitPlanMode",
    "HarnessSignalSwitchMode",
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
    "RalphFeatureEntry",
    "RalphProgress",
    "RalphStopHook",
    "ReadOnlySandbox",
    "RiskLevel",
    "RunResult",
    "RunResultEscalate",
    "RunResultFailure",
    "RunResultSuccess",
    "RunResultWaitingForHuman",
    "BwrapProfile",
    "IsolationMode",
    "IsolationModeBubblewrap",
    "IsolationModeDocker",
    # IsolationModeNone is intentionally NOT exported (issue #34). It is the
    # no-path-enforcement footgun, reachable only via the dangerous opt-in:
    # ``from spore_core.dangerous import IsolationModeNone``. The class stays
    # defined for the wire discriminated union; only its name is gated.
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
    "StreamUserMessage",
    "SEND_MESSAGE_TOOL_NAME",
    "Task",
    "TaskId",
    "TerminationContinue",
    "TerminationDecision",
    "TerminationHalt",
    "TerminationPolicy",
    "ToolOutput",
    "ToolOutputAwaitingClarification",
    "ToolOutputError",
    "ToolOutputEscalate",
    "ToolOutputSuccess",
    "ToolOutputWaitingForHuman",
    "ToolRegistry",
    "VcsError",
    "VcsLogArgs",
    "VcsProvider",
    "GitVcsProvider",
    "FixtureVcsProvider",
    "new_session_id",
    "new_task_id",
    "sandbox_violation_is_always_halt",
]

# Avoid unused-import warnings for `Awaitable` (kept for IDE hover usefulness).
_: Awaitable[None] | None = None
