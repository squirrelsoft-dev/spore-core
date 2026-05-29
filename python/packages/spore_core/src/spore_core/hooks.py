"""Issue #69 — lifecycle hook system (``Hook`` / ``HookChain``).

A general-purpose extension layer that lets external code observe and shape the
harness at well-defined lifecycle moments. This is a NEW sibling of
:mod:`spore_core.middleware`: middleware shapes the context block DURING
assembly (a lower-level primitive); hooks fire at a higher level on the
already-assembled artifacts. The two layers are intentionally distinct and this
module does not modify or subsume ``middleware.py``.

Types
-----
* :class:`HookEvent` — the 17 lifecycle events, with classification predicates
  :meth:`~HookEvent.is_mutable`, :meth:`~HookEvent.is_sync_only`,
  :meth:`~HookEvent.can_block`, :meth:`~HookEvent.is_pre`.
* :class:`HookContext` (one dataclass per event) — the per-event payload a
  :class:`Hook` receives. Mutable fields are plain attributes a pre-hook may
  rewrite in place.
* :class:`HookDecision` — ``Continue`` / ``Block`` / ``Inject`` / ``Deny`` /
  ``Mutate`` (a discriminated union tagged on ``decision``, snake_case).
* :class:`HookSync` — sync vs async registration mode.
* :class:`HookError` — one exception class for this component.
* :class:`Hook` — Protocol: ``handle`` / ``events`` / ``name`` / ``sync_mode``.
* :class:`HookChain` — Protocol: ``register`` + ``fire``.
* :class:`StandardHookChain` — in-memory reference impl: registration-order
  fan-out, chained mutation through pre-hooks, sync aggregation, async
  fire-and-forget.
* :class:`FunctionHook`, :class:`CommandHook` — the two v1 handler types.

The 17 events (mutation / blocking / sync classification)

==================== ======== ===================== ========= ==============
Event                Pre/Post Mutates               Can block Sync mode
==================== ======== ===================== ========= ==============
``PreTurn``          pre      context_block         yes       sync
``PostTurn``         post     —                     no        sync or async
``PreToolUse``       pre      tool_input (or deny)  yes       sync
``PostToolUse``      post     —                     no        sync or async
``PostToolUseFailure`` post   —                     no        sync or async
``PostToolBatch``    post     —                     yes       sync
``OnLoopStart``      pre      task_instruction      yes       sync
``Stop``             post     —                     yes       **sync only**
``OnPause``          post     —                     no        **async only**
``OnResume``         pre      task_instruction      no        sync
``OnError``          post     — (can suppress)      yes       sync or async
``OnPlanCreated``    post     plan                  yes       sync
``OnTaskAdvance``    pre      task                  yes       sync
``OnSubagentSpawn``  pre      child_task (or deny)  yes       sync
``OnSubagentComplete`` post   —                     no        sync or async
``PreCompact``       pre      preserve_hints        yes       sync
``PostCompact``      post     —                     no        async ok
==================== ======== ===================== ========= ==============

Rules enforced (R1–R26 from issue #69)

* R1  Pre-hooks may mutate the single mutable field of their context.
* R2  Pre-hook chains thread the mutated value to the next hook.
* R3  Hooks fire in REGISTRATION order (not middleware-style priority).
* R4  ``Block`` is only legal on a can-block event.
* R5  ``Deny`` is only legal on ``PreToolUse`` / ``OnSubagentSpawn``.
* R6  ``Mutate`` is only legal on a pre-event (and replaces the mutable field).
* R7  ``Inject`` injects into the next turn's context block.
* R8  ``Stop`` is SYNC ONLY — registering it async is rejected.
* R9  ``OnPause`` is ASYNC ONLY — registering it sync is rejected.
* R10 A sync post-hook block stops the chain and is reported to the loop.
* R11 Async post-hooks are fire-and-forget: spawned, never awaited, result
  and failure swallowed.
* R12 Stop ``Block`` injects ``reason`` into the next turn and the loop
  continues (the same path the legacy ForceAnotherTurn would use).
* R13 Stop all-``Continue`` (or no hooks) terminates normally.
* R14 After ``max_stop_blocks`` Stop blocks in a run, the loop terminates
  anyway (per-run counter; resume starts fresh).
* R15 ``PreToolUse`` deny rejects the tool call.
* R16 ``PreToolUse`` may mutate ``tool_input``.
* R17 Registering a hook for an event it cannot legally decide on is rejected
  at register time.
* R18 Command handler stdin = ``{"event":"<snake_case>","context":<payload>}``.
* R19 Command handler stdout parsed as a tagged :class:`HookDecision`.
* R20 Command nonzero exit → :class:`HookError` (``CommandFailed``); NOT an
  implicit block.
* R21 Command malformed stdout → :class:`HookError` (``CommandOutputInvalid``).
* R22 No sandbox, no timeout on command handlers in v1.
* R23 Function handler runs an inline callable synchronously.
* R24 Decision validity is checked at fire time as well as register time.
* R25 A hook that lists multiple events only fires for the event it is invoked
  with.
* R26 Firing order on Stop: registered Stop hooks first, THEN (when wired) the
  strategy verifier; either can block.

Loop-wiring status
-------------------
Events whose loop machinery EXISTS and are wired into the ReAct loop in
``harness.py``: ``Stop`` (the live one — see
``StandardHarness._fire_stop_hooks``). The remaining turn/tool/compaction
events have ``fire``-able contexts defined here and are exercised by unit
tests; their loop call sites are wired as the corresponding harness machinery
lands, mirroring the Rust reference.

Events DEFINED-AND-UNIT-TESTED but NOT YET loop-wired (their strategy /
subagent / pause machinery is deferred elsewhere): ``OnPause``, ``OnResume``,
``OnPlanCreated``, ``OnTaskAdvance``, ``OnSubagentSpawn``,
``OnSubagentComplete``. Each is exercised directly by unit tests through
:meth:`StandardHookChain.fire`.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .agent import Context as ContextBlock
from .context import CompactionPreserveHints, SessionState as RichSessionState
from .errors import SporeError
from .harness import HarnessConfig, PausedState, SessionId, Task

JsonValue = Any


# ============================================================================
# Locally-defined payload types
#
# These three artifacts are not yet modelled elsewhere in the package
# (``TurnOutput`` / ``PlanArtifact`` / ``ToolCallSummary``). ``ContextBlock``
# reuses the agent's assembled ``Context`` (aliased above). They are
# intentionally additive — when the owning strategy / subagent issues land, the
# canonical shapes will replace these.
# ============================================================================


class TurnOutput(BaseModel):
    """The output of a single completed turn, handed to post-turn / Stop hooks."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    had_tool_calls: bool = False


class PlanArtifact(BaseModel):
    """A composite-strategy plan artifact, handed to ``OnPlanCreated``."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[str] = Field(default_factory=list)
    rationale: str = ""


class ToolCallSummary(BaseModel):
    """A one-line summary of a tool call in a batch, handed to ``PostToolBatch``."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    succeeded: bool


# ============================================================================
# HookEvent
# ============================================================================


class HookEvent(str, Enum):
    """The 17 lifecycle events at which a :class:`Hook` can fire.

    The value is the snake_case wire string used on the command-handler stdin
    payload and anywhere an event is serialized.
    """

    PRE_TURN = "pre_turn"
    POST_TURN = "post_turn"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"
    POST_TOOL_BATCH = "post_tool_batch"
    ON_LOOP_START = "on_loop_start"
    STOP = "stop"
    ON_PAUSE = "on_pause"
    ON_RESUME = "on_resume"
    ON_ERROR = "on_error"
    ON_PLAN_CREATED = "on_plan_created"
    ON_TASK_ADVANCE = "on_task_advance"
    ON_SUBAGENT_SPAWN = "on_subagent_spawn"
    ON_SUBAGENT_COMPLETE = "on_subagent_complete"
    PRE_COMPACT = "pre_compact"
    POST_COMPACT = "post_compact"

    def is_pre(self) -> bool:
        """Whether this is a pre-event (fires before its action; may mutate)."""
        return self in _PRE_EVENTS

    def is_mutable(self) -> bool:
        """Whether this event carries a mutable field a pre-hook may rewrite.

        Equivalent to :meth:`is_pre` — every pre-event is mutable.
        """
        return self.is_pre()

    def is_sync_only(self) -> bool:
        """Whether this event may only run synchronously."""
        return self in _SYNC_ONLY_EVENTS

    def is_async_only(self) -> bool:
        """Whether this event may only run asynchronously (fire-and-forget)."""
        return self in _ASYNC_ONLY_EVENTS

    def can_block(self) -> bool:
        """Whether a hook on this event may return :class:`HookBlock`."""
        return self in _CAN_BLOCK_EVENTS

    def can_deny(self) -> bool:
        """Whether a hook on this event may return :class:`HookDeny`."""
        return self in _CAN_DENY_EVENTS


#: All 17 events, in catalogue order.
ALL_EVENTS: tuple[HookEvent, ...] = (
    HookEvent.PRE_TURN,
    HookEvent.POST_TURN,
    HookEvent.PRE_TOOL_USE,
    HookEvent.POST_TOOL_USE,
    HookEvent.POST_TOOL_USE_FAILURE,
    HookEvent.POST_TOOL_BATCH,
    HookEvent.ON_LOOP_START,
    HookEvent.STOP,
    HookEvent.ON_PAUSE,
    HookEvent.ON_RESUME,
    HookEvent.ON_ERROR,
    HookEvent.ON_PLAN_CREATED,
    HookEvent.ON_TASK_ADVANCE,
    HookEvent.ON_SUBAGENT_SPAWN,
    HookEvent.ON_SUBAGENT_COMPLETE,
    HookEvent.PRE_COMPACT,
    HookEvent.POST_COMPACT,
)

_PRE_EVENTS: frozenset[HookEvent] = frozenset(
    {
        HookEvent.PRE_TURN,
        HookEvent.PRE_TOOL_USE,
        HookEvent.ON_LOOP_START,
        HookEvent.ON_RESUME,
        HookEvent.ON_TASK_ADVANCE,
        HookEvent.ON_SUBAGENT_SPAWN,
        HookEvent.PRE_COMPACT,
    }
)

_SYNC_ONLY_EVENTS: frozenset[HookEvent] = frozenset(
    {
        HookEvent.STOP,
        HookEvent.PRE_TURN,
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_BATCH,
        HookEvent.ON_LOOP_START,
        HookEvent.ON_RESUME,
        HookEvent.ON_PLAN_CREATED,
        HookEvent.ON_TASK_ADVANCE,
        HookEvent.ON_SUBAGENT_SPAWN,
        HookEvent.PRE_COMPACT,
    }
)

_ASYNC_ONLY_EVENTS: frozenset[HookEvent] = frozenset({HookEvent.ON_PAUSE, HookEvent.POST_COMPACT})

_CAN_BLOCK_EVENTS: frozenset[HookEvent] = frozenset(
    {
        HookEvent.PRE_TURN,
        HookEvent.POST_TOOL_BATCH,
        HookEvent.ON_LOOP_START,
        HookEvent.STOP,
        HookEvent.ON_ERROR,
        HookEvent.ON_PLAN_CREATED,
        HookEvent.ON_TASK_ADVANCE,
    }
)

_CAN_DENY_EVENTS: frozenset[HookEvent] = frozenset(
    {HookEvent.PRE_TOOL_USE, HookEvent.ON_SUBAGENT_SPAWN}
)


# ============================================================================
# HookSync
# ============================================================================


class HookSync(str, Enum):
    """Whether a hook runs synchronously (blocking, result observed) or
    asynchronously (fire-and-forget)."""

    SYNC = "sync"
    ASYNC = "async"


# ============================================================================
# HookDecision — tagged on ``decision``, snake_case
# ============================================================================


class HookContinue(BaseModel):
    """Proceed; no change."""

    model_config = ConfigDict(extra="forbid")
    decision: Literal["continue"] = "continue"


class HookBlock(BaseModel):
    """Can-block events only — injects ``reason`` into the next turn."""

    model_config = ConfigDict(extra="forbid")
    decision: Literal["block"] = "block"
    reason: str


class HookInject(BaseModel):
    """Injects ``context`` into the next turn's context block."""

    model_config = ConfigDict(extra="forbid")
    decision: Literal["inject"] = "inject"
    context: str


class HookDeny(BaseModel):
    """``PreToolUse`` / ``OnSubagentSpawn`` only — rejects the action."""

    model_config = ConfigDict(extra="forbid")
    decision: Literal["deny"] = "deny"
    reason: str


class HookMutate(BaseModel):
    """Pre-hooks only — replaces the mutable field with ``data``."""

    model_config = ConfigDict(extra="forbid")
    decision: Literal["mutate"] = "mutate"
    data: JsonValue


HookDecision = Union[HookContinue, HookBlock, HookInject, HookDeny, HookMutate]
"""The control a hook exerts when it fires (Decision 3 of issue #69).

Wire format is tagged on ``decision``, e.g.
``{"decision":"block","reason":"x"}``.
"""

#: Adapter for parsing/serializing the tagged union from/to wire JSON.
HOOK_DECISION_ADAPTER: TypeAdapter[HookDecision] = TypeAdapter(HookDecision)


def parse_hook_decision(data: JsonValue | str | bytes) -> HookDecision:
    """Parse a tagged :data:`HookDecision` from a JSON value, string, or bytes."""
    if isinstance(data, (str, bytes, bytearray)):
        return HOOK_DECISION_ADAPTER.validate_json(data)
    return HOOK_DECISION_ADAPTER.validate_python(data)


def hook_decision_to_dict(decision: HookDecision) -> dict[str, Any]:
    """Serialize a :data:`HookDecision` to its tagged wire ``dict``."""
    return decision.model_dump(mode="json")


def _decision_validate_for(decision: HookDecision, event: HookEvent) -> None:
    """Validate that ``decision`` is legal for ``event`` (register + fire time)."""
    if isinstance(decision, (HookContinue, HookInject)):
        ok = True
    elif isinstance(decision, HookBlock):
        ok = event.can_block()
    elif isinstance(decision, HookDeny):
        ok = event.can_deny()
    elif isinstance(decision, HookMutate):
        ok = event.is_mutable()
    else:  # pragma: no cover - exhaustive over the union
        ok = False
    if not ok:
        raise HookError(
            f"hook '{decision.decision}' decision is illegal for event '{event.value}'",
            kind="illegal_decision",
        )


# ============================================================================
# HookError
# ============================================================================


class HookError(SporeError):
    """The single exception class for the hook system (issue #69).

    Carries an optional ``kind`` discriminator so callers can distinguish
    command-handler failures (``command_failed`` / ``command_output_invalid``)
    from decision-validity errors and sync/async mismatches.
    """

    def __init__(self, message: str, *, kind: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind


# ============================================================================
# HookContext — one dataclass per event
# ============================================================================
#
# Mutable fields are plain attributes; a pre-hook rewrites them in place (R1),
# and the chain threads those mutations to the next hook (R2). Each context
# knows its :class:`HookEvent`, can serialize itself to the command-handler
# payload, and can apply a :class:`HookMutate` to its mutable field.


@dataclass
class PreTurnContext:
    session_id: SessionId
    turn_number: int
    context_block: ContextBlock

    EVENT = HookEvent.PRE_TURN


@dataclass
class PostTurnContext:
    session_id: SessionId
    turn_number: int
    output: TurnOutput

    EVENT = HookEvent.POST_TURN


@dataclass
class PreToolUseContext:
    session_id: SessionId
    turn_number: int
    tool_name: str
    tool_input: JsonValue

    EVENT = HookEvent.PRE_TOOL_USE


@dataclass
class PostToolUseContext:
    session_id: SessionId
    turn_number: int
    tool_name: str
    tool_input: JsonValue
    tool_response: JsonValue
    duration_ms: int

    EVENT = HookEvent.POST_TOOL_USE


@dataclass
class PostToolUseFailureContext:
    session_id: SessionId
    turn_number: int
    tool_name: str
    tool_input: JsonValue
    error: str
    duration_ms: int

    EVENT = HookEvent.POST_TOOL_USE_FAILURE


@dataclass
class PostToolBatchContext:
    session_id: SessionId
    turn_number: int
    tool_calls: list[ToolCallSummary]

    EVENT = HookEvent.POST_TOOL_BATCH


@dataclass
class OnLoopStartContext:
    session_id: SessionId
    task_instruction: str
    config: HarnessConfig

    EVENT = HookEvent.ON_LOOP_START


@dataclass
class StopContext:
    session_id: SessionId
    turn_number: int
    last_output: TurnOutput
    task_instruction: str
    session_state: RichSessionState | None

    EVENT = HookEvent.STOP


@dataclass
class OnPauseContext:
    session_id: SessionId
    turn_number: int

    EVENT = HookEvent.ON_PAUSE


@dataclass
class OnResumeContext:
    session_id: SessionId
    task_instruction: str
    paused_state: PausedState

    EVENT = HookEvent.ON_RESUME


@dataclass
class OnErrorContext:
    session_id: SessionId
    turn_number: int
    error: str

    EVENT = HookEvent.ON_ERROR


@dataclass
class OnPlanCreatedContext:
    session_id: SessionId
    plan: PlanArtifact

    EVENT = HookEvent.ON_PLAN_CREATED


@dataclass
class OnTaskAdvanceContext:
    session_id: SessionId
    task: Task
    task_index: int
    total_tasks: int

    EVENT = HookEvent.ON_TASK_ADVANCE


@dataclass
class OnSubagentSpawnContext:
    session_id: SessionId
    child_task: str
    strategy: str

    EVENT = HookEvent.ON_SUBAGENT_SPAWN


@dataclass
class OnSubagentCompleteContext:
    session_id: SessionId
    child_session_id: SessionId
    result: JsonValue

    EVENT = HookEvent.ON_SUBAGENT_COMPLETE


@dataclass
class PreCompactContext:
    session_id: SessionId
    preserve_hints: CompactionPreserveHints

    EVENT = HookEvent.PRE_COMPACT


@dataclass
class PostCompactContext:
    session_id: SessionId
    compact_summary: str

    EVENT = HookEvent.POST_COMPACT


HookContext = Union[
    PreTurnContext,
    PostTurnContext,
    PreToolUseContext,
    PostToolUseContext,
    PostToolUseFailureContext,
    PostToolBatchContext,
    OnLoopStartContext,
    StopContext,
    OnPauseContext,
    OnResumeContext,
    OnErrorContext,
    OnPlanCreatedContext,
    OnTaskAdvanceContext,
    OnSubagentSpawnContext,
    OnSubagentCompleteContext,
    PreCompactContext,
    PostCompactContext,
]


def _jsonify(value: Any) -> JsonValue:
    """Coerce a context field to a JSON-ready value for the command payload."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, CompactionPreserveHints):
        return {
            "keep_architectural_decisions": value.keep_architectural_decisions,
            "keep_open_problems": value.keep_open_problems,
            "keep_current_task_state": value.keep_current_task_state,
            "keep_recent_file_list": value.keep_recent_file_list,
            "keep_thinking_blocks": value.keep_thinking_blocks,
        }
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return value


def context_event(ctx: HookContext) -> HookEvent:
    """Which :class:`HookEvent` ``ctx`` corresponds to."""
    return ctx.EVENT


def context_to_payload(ctx: HookContext) -> dict[str, Any]:
    """Serialize ``ctx`` to the JSON ``context`` payload a command handler
    receives on stdin. Mutable fields are serialized by value.

    ``session_state`` is rendered only when present; the cross-language
    fixtures pin it as ``null`` for Stop contexts assembled without a rich
    state, so we emit ``null`` rather than omitting the key.
    """
    if isinstance(ctx, PreTurnContext):
        return {
            "session_id": ctx.session_id,
            "turn_number": ctx.turn_number,
            "context_block": _jsonify(ctx.context_block),
        }
    if isinstance(ctx, PostTurnContext):
        return {
            "session_id": ctx.session_id,
            "turn_number": ctx.turn_number,
            "output": _jsonify(ctx.output),
        }
    if isinstance(ctx, PreToolUseContext):
        return {
            "session_id": ctx.session_id,
            "turn_number": ctx.turn_number,
            "tool_name": ctx.tool_name,
            "tool_input": ctx.tool_input,
        }
    if isinstance(ctx, PostToolUseContext):
        return {
            "session_id": ctx.session_id,
            "turn_number": ctx.turn_number,
            "tool_name": ctx.tool_name,
            "tool_input": ctx.tool_input,
            "tool_response": ctx.tool_response,
            "duration_ms": ctx.duration_ms,
        }
    if isinstance(ctx, PostToolUseFailureContext):
        return {
            "session_id": ctx.session_id,
            "turn_number": ctx.turn_number,
            "tool_name": ctx.tool_name,
            "tool_input": ctx.tool_input,
            "error": ctx.error,
            "duration_ms": ctx.duration_ms,
        }
    if isinstance(ctx, PostToolBatchContext):
        return {
            "session_id": ctx.session_id,
            "turn_number": ctx.turn_number,
            "tool_calls": _jsonify(ctx.tool_calls),
        }
    if isinstance(ctx, OnLoopStartContext):
        return {
            "session_id": ctx.session_id,
            "task_instruction": ctx.task_instruction,
        }
    if isinstance(ctx, StopContext):
        return {
            "session_id": ctx.session_id,
            "turn_number": ctx.turn_number,
            "last_output": _jsonify(ctx.last_output),
            "task_instruction": ctx.task_instruction,
            "session_state": (
                _jsonify(ctx.session_state) if ctx.session_state is not None else None
            ),
        }
    if isinstance(ctx, OnPauseContext):
        return {"session_id": ctx.session_id, "turn_number": ctx.turn_number}
    if isinstance(ctx, OnResumeContext):
        return {
            "session_id": ctx.session_id,
            "task_instruction": ctx.task_instruction,
            "paused_state": _jsonify(ctx.paused_state),
        }
    if isinstance(ctx, OnErrorContext):
        return {
            "session_id": ctx.session_id,
            "turn_number": ctx.turn_number,
            "error": ctx.error,
        }
    if isinstance(ctx, OnPlanCreatedContext):
        return {"session_id": ctx.session_id, "plan": _jsonify(ctx.plan)}
    if isinstance(ctx, OnTaskAdvanceContext):
        return {
            "session_id": ctx.session_id,
            "task": _jsonify(ctx.task),
            "task_index": ctx.task_index,
            "total_tasks": ctx.total_tasks,
        }
    if isinstance(ctx, OnSubagentSpawnContext):
        return {
            "session_id": ctx.session_id,
            "child_task": ctx.child_task,
            "strategy": ctx.strategy,
        }
    if isinstance(ctx, OnSubagentCompleteContext):
        return {
            "session_id": ctx.session_id,
            "child_session_id": ctx.child_session_id,
            "result": ctx.result,
        }
    if isinstance(ctx, PreCompactContext):
        return {
            "session_id": ctx.session_id,
            "preserve_hints": _jsonify(ctx.preserve_hints),
        }
    if isinstance(ctx, PostCompactContext):
        return {
            "session_id": ctx.session_id,
            "compact_summary": ctx.compact_summary,
        }
    raise HookError(f"unknown hook context: {type(ctx).__name__!r}")  # pragma: no cover


def _string_from_value(data: JsonValue) -> str:
    """Coerce a ``Mutate`` ``data`` value into a ``str`` (a JSON string passes
    through; any other value is stringified as JSON text)."""
    if isinstance(data, str):
        return data
    return json.dumps(data)


def _apply_mutation(ctx: HookContext, hook_name: str, data: JsonValue) -> None:
    """Apply a :class:`HookMutate`'s ``data`` to ``ctx``'s mutable field (R6).

    Raises :class:`HookError` if ``ctx`` is not a mutable event or the data
    cannot be coerced into the target field type.
    """
    try:
        if isinstance(ctx, PreTurnContext):
            ctx.context_block = ContextBlock.model_validate(data)
        elif isinstance(ctx, PreToolUseContext):
            ctx.tool_input = data
        elif isinstance(ctx, OnLoopStartContext):
            ctx.task_instruction = _string_from_value(data)
        elif isinstance(ctx, OnResumeContext):
            ctx.task_instruction = _string_from_value(data)
        elif isinstance(ctx, OnPlanCreatedContext):
            ctx.plan = PlanArtifact.model_validate(data)
        elif isinstance(ctx, OnTaskAdvanceContext):
            ctx.task = Task.model_validate(data)
        elif isinstance(ctx, OnSubagentSpawnContext):
            ctx.child_task = _string_from_value(data)
        elif isinstance(ctx, PreCompactContext):
            ctx.preserve_hints = _hints_from_value(data)
        else:
            raise HookError(
                f"hook 'mutate' decision is illegal for event '{context_event(ctx).value}'",
                kind="illegal_decision",
            )
    except HookError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface any coercion failure
        raise HookError(f"hook '{hook_name}' failed: {exc}", kind="handler_failed") from exc


def _hints_from_value(data: JsonValue) -> CompactionPreserveHints:
    if not isinstance(data, dict):
        raise HookError("preserve_hints mutation requires an object")
    hints = CompactionPreserveHints()
    for key, value in data.items():
        if not hasattr(hints, key):
            raise HookError(f"unknown preserve hint field: {key!r}")
        setattr(hints, key, value)
    return hints


# ============================================================================
# Hook / HookChain protocols
# ============================================================================


@runtime_checkable
class Hook(Protocol):
    """A single lifecycle hook handler.

    ``handle`` is ``async`` so command/HTTP handlers can do I/O; the in-process
    :class:`FunctionHook` simply returns its synchronous result.
    """

    async def handle(self, ctx: HookContext) -> HookDecision:
        """Handle one firing. Pre-hooks may mutate the mutable field on ``ctx``
        directly OR return :class:`HookMutate`."""
        ...

    def events(self) -> list[HookEvent]:
        """The events this hook subscribes to."""
        ...

    def name(self) -> str:
        """A stable name for diagnostics and error messages."""
        ...

    def sync_mode(self) -> HookSync:
        """Whether this hook runs sync (blocking) or async (fire-and-forget)."""
        ...


@dataclass(frozen=True)
class FireOutcome:
    """Outcome of firing a chain back to the harness loop."""

    kind: Literal["continue", "block", "deny", "inject"]
    reason: str = ""
    context: str = ""

    @staticmethod
    def cont() -> FireOutcome:
        return FireOutcome(kind="continue")

    @staticmethod
    def block(reason: str) -> FireOutcome:
        return FireOutcome(kind="block", reason=reason)

    @staticmethod
    def deny(reason: str) -> FireOutcome:
        return FireOutcome(kind="deny", reason=reason)

    @staticmethod
    def inject(context: str) -> FireOutcome:
        return FireOutcome(kind="inject", context=context)


@runtime_checkable
class HookChain(Protocol):
    """Registry + dispatcher for :class:`Hook`s. Implementations fan out to all
    hooks subscribed to an event in registration order."""

    def register(self, hook: Hook) -> None:
        """Register a hook. Rejects sync-only events registered async (and vice
        versa). Registration order is firing order."""
        ...

    async def fire(self, ctx: HookContext) -> FireOutcome:
        """Fire the chain for ``ctx``. Threads mutations through ``ctx`` in
        place and returns the aggregate outcome (first Block/Deny wins; Injects
        are newline-joined)."""
        ...


# ============================================================================
# StandardHookChain
# ============================================================================


class StandardHookChain:
    """In-memory reference :class:`HookChain`.

    Holds hooks in a registration-ordered list (guarded by a lock for
    register-time mutation) and fans out in that order.
    """

    def __init__(self) -> None:
        self._hooks: list[Hook] = []
        self._lock = threading.Lock()

    def register(self, hook: Hook) -> None:
        mode = hook.sync_mode()
        for event in hook.events():
            if event.is_sync_only() and mode == HookSync.ASYNC:
                raise HookError(
                    f"hook '{hook.name()}' cannot register for sync-only event "
                    f"'{event.value}' as async",
                    kind="sync_only_event",
                )
            if event.is_async_only() and mode == HookSync.SYNC:
                raise HookError(
                    f"hook '{hook.name()}' cannot register for async-only event "
                    f"'{event.value}' as sync",
                    kind="async_only_event",
                )
        with self._lock:
            self._hooks.append(hook)

    def _hooks_for(self, event: HookEvent) -> list[Hook]:
        with self._lock:
            return [h for h in self._hooks if event in h.events()]

    async def fire(self, ctx: HookContext) -> FireOutcome:
        event = context_event(ctx)
        injects: list[str] = []

        for hook in self._hooks_for(event):
            if hook.sync_mode() == HookSync.ASYNC:
                # R11: async hooks are fire-and-forget. Spawn a detached task on
                # a snapshot of the payload; never await it; swallow its result
                # and any failure.
                _spawn_detached(hook, context_to_payload(ctx))
                continue

            decision = await hook.handle(ctx)
            _decision_validate_for(decision, event)  # R24

            if isinstance(decision, HookContinue):
                continue
            if isinstance(decision, HookInject):
                injects.append(decision.context)
                continue
            if isinstance(decision, HookBlock):
                return FireOutcome.block(decision.reason)
            if isinstance(decision, HookDeny):
                return FireOutcome.deny(decision.reason)
            if isinstance(decision, HookMutate):
                _apply_mutation(ctx, hook.name(), decision.data)

        if injects:
            return FireOutcome.inject("\n".join(injects))
        return FireOutcome.cont()


def _spawn_detached(hook: Hook, payload: dict[str, Any]) -> None:
    """Fire-and-forget an async hook (R11).

    The detached task reconstructs nothing from the live context — it gets an
    owned ``payload`` snapshot — and its result/failure is swallowed. If no
    event loop is running (rare in tests that drive ``fire`` directly), the task
    is silently dropped; async hooks are observability-only, so dropping is safe.
    """

    handle_payload = getattr(hook, "handle_payload", None)

    async def _run() -> None:
        try:
            if handle_payload is not None:
                await handle_payload(payload)
            # Observability-only hooks need no reconstruction step.
        except Exception:  # noqa: BLE001 — fire-and-forget swallows everything
            pass

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return
    running.create_task(_run())


# ============================================================================
# FunctionHook — inline callable handler
# ============================================================================


class FunctionHook:
    """A :class:`Hook` backed by an inline callable (R23).

    The primary handler type for harness builders. The callable receives the
    live :class:`HookContext` (so it may mutate the mutable field directly) and
    returns a :class:`HookDecision`. It runs synchronously inside :meth:`handle`.
    The callable may be a plain function or an ``async`` coroutine function.
    """

    def __init__(
        self,
        name: str,
        events: list[HookEvent],
        func: Any,
        *,
        sync_mode: HookSync = HookSync.SYNC,
    ) -> None:
        self._name = name
        self._events = list(events)
        self._func = func
        self._sync_mode = sync_mode

    def async_mode(self) -> FunctionHook:
        """Mark this hook async (fire-and-forget). Only legal for events that
        are not sync-only; the chain enforces this at register time."""
        self._sync_mode = HookSync.ASYNC
        return self

    async def handle(self, ctx: HookContext) -> HookDecision:
        result = self._func(ctx)
        if hasattr(result, "__await__"):
            result = await result
        return result

    def events(self) -> list[HookEvent]:
        return list(self._events)

    def name(self) -> str:
        return self._name

    def sync_mode(self) -> HookSync:
        return self._sync_mode


# ============================================================================
# CommandHook — shell command handler
# ============================================================================


class CommandHook:
    """A :class:`Hook` that shells out to an external command.

    stdin receives ``{"event":"<snake_case>","context":<payload>}`` (R18);
    stdout is parsed as a tagged :class:`HookDecision` (R19). Nonzero exit →
    :class:`HookError` ``command_failed`` (R20); malformed stdout →
    :class:`HookError` ``command_output_invalid`` (R21). No sandbox and no
    timeout in v1 (R22).
    """

    def __init__(
        self,
        name: str,
        events: list[HookEvent],
        program: str,
        args: list[str] | None = None,
        *,
        sync_mode: HookSync = HookSync.SYNC,
    ) -> None:
        self._name = name
        self._events = list(events)
        self._program = program
        self._args = list(args or [])
        self._sync_mode = sync_mode

    def async_mode(self) -> CommandHook:
        self._sync_mode = HookSync.ASYNC
        return self

    def _run(self, ctx: HookContext) -> HookDecision:
        payload = {
            "event": context_event(ctx).value,
            "context": context_to_payload(ctx),
        }
        stdin_bytes = json.dumps(payload).encode("utf-8")

        try:
            proc = subprocess.run(  # noqa: S603 — no sandbox/timeout by spec (R22)
                [self._program, *self._args],
                input=stdin_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise HookError(
                f"command hook '{self._program}' exited with status -1: {exc}",
                kind="command_failed",
            ) from exc

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", "replace").strip()
            raise HookError(
                f"command hook '{self._program}' exited with status {proc.returncode}: {stderr}",
                kind="command_failed",
            )

        stdout = proc.stdout.decode("utf-8", "replace").strip()
        try:
            return parse_hook_decision(stdout)
        except Exception as exc:  # noqa: BLE001 — any parse failure is invalid output
            raise HookError(
                f"command hook '{self._program}' produced invalid stdout: {exc}",
                kind="command_output_invalid",
            ) from exc

    async def handle(self, ctx: HookContext) -> HookDecision:
        return self._run(ctx)

    def events(self) -> list[HookEvent]:
        return list(self._events)

    def name(self) -> str:
        return self._name

    def sync_mode(self) -> HookSync:
        return self._sync_mode


__all__ = [
    "ALL_EVENTS",
    "HOOK_DECISION_ADAPTER",
    "CommandHook",
    "CompactionPreserveHints",
    "FireOutcome",
    "FunctionHook",
    "Hook",
    "HookBlock",
    "HookChain",
    "HookContext",
    "HookContinue",
    "HookDecision",
    "HookDeny",
    "HookError",
    "HookEvent",
    "HookInject",
    "HookMutate",
    "HookSync",
    "OnErrorContext",
    "OnLoopStartContext",
    "OnPauseContext",
    "OnPlanCreatedContext",
    "OnResumeContext",
    "OnSubagentCompleteContext",
    "OnSubagentSpawnContext",
    "OnTaskAdvanceContext",
    "PlanArtifact",
    "PostCompactContext",
    "PostToolBatchContext",
    "PostToolUseContext",
    "PostToolUseFailureContext",
    "PostTurnContext",
    "PreCompactContext",
    "PreToolUseContext",
    "PreTurnContext",
    "StandardHookChain",
    "StopContext",
    "ToolCallSummary",
    "TurnOutput",
    "context_event",
    "context_to_payload",
    "hook_decision_to_dict",
    "parse_hook_decision",
]
