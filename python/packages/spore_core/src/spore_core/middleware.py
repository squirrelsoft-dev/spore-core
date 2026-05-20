"""MiddlewareChain — cross-cutting interception of the agent loop at six
hook points (issue #11).

Mirrors the Rust reference at ``rust/crates/spore-core/src/middleware.rs``.

See ``docs/harness-engineering-concepts.md`` § "Middleware Chain" for the
authoritative rules. This module ships:

* The full :class:`Middleware` Protocol and :class:`HookContext` /
  :class:`HookPoint` / :class:`MiddlewareDecision` surface from the spec.
* :class:`StandardMiddlewareChain` — in-memory reference implementation
  with priority ordering, ForceAnotherTurn concatenation, and first-wins
  SurfaceToHuman semantics.
* A subset of standard middleware referenced by name in the spec
  (:class:`TokenBudgetMiddleware`, :class:`LoopDetectionMiddleware`,
  :class:`PreCompletionChecklistMiddleware`,
  :class:`PatchToolCallsMiddleware`, :class:`TracingMiddleware`).

Rules enforced:

* Before hooks (``BEFORE_SESSION``, ``BEFORE_TURN``, ``BEFORE_TOOL``,
  ``BEFORE_COMPLETION``) run sorted by priority **ascending** — lowest
  priority number first.
* After hooks (``AFTER_TOOL``, ``AFTER_SESSION``) run sorted by priority
  **descending** — highest number first (wrapping pattern).
* First :class:`MiddlewareHalt` or :class:`MiddlewareSurfaceToHuman` stops
  the chain.
* :class:`MiddlewareForceAnotherTurn` is **only valid on
  ``BEFORE_COMPLETION``**. All injections from middleware are concatenated
  (newline-joined) and returned in a single decision. The chain continues
  running.
* :class:`MiddlewareSurfaceToHuman` is valid only on ``BEFORE_TOOL`` and
  ``BEFORE_COMPLETION``. Returning it from any other hook is an
  :class:`IllegalDecisionError` surfaced as :class:`MiddlewareHalt` to the
  loop.
* Middleware **must not** hold session state on ``self`` keyed by
  :class:`SessionId` (use an external dict; clear in ``AfterSession``).
  This is design guidance, not enforced.
* Middleware must not call :class:`ModelInterface` or :class:`ToolRegistry`
  — neither is in scope of any :class:`HookContext` variant.

Priority defaults to ``0``. :class:`TracingMiddleware` registers at
``-sys.maxsize`` so it always runs first on before hooks and last on
after hooks. :class:`PatchToolCallsMiddleware` registers at
``-sys.maxsize + 1`` so it runs before all other ``BEFORE_TOOL``
middleware (per spec).
"""

from __future__ import annotations

import asyncio
import itertools
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .errors import SporeError
from .harness import (
    HumanRequest,
    RunResult,
    SessionId,
    SessionState,
    Task,
    TaskId,
    ToolOutputSuccess,
)
from .harness import (
    HarnessToolResult as ToolResult,
)
from .model import ToolCall

if TYPE_CHECKING:
    # ``observability`` imports ``HookPoint`` / ``MiddlewareDecision`` from this
    # module, so importing it at runtime here would create a cycle. The patch
    # middleware (issue #28) takes an :class:`ObservabilityProvider` purely as a
    # collaborator and imports the concrete span types lazily inside the method
    # that emits them.
    from .observability import ObservabilityProvider

# ============================================================================
# Pydantic base
# ============================================================================


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# HookPoint
# ============================================================================


class HookPoint(str, Enum):
    """The six lifecycle hook points exposed by the harness."""

    BEFORE_SESSION = "before_session"
    BEFORE_TURN = "before_turn"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    BEFORE_COMPLETION = "before_completion"
    AFTER_SESSION = "after_session"

    def is_before(self) -> bool:
        """True for hooks ordered ascending by priority (lowest first)."""
        return self in (
            HookPoint.BEFORE_SESSION,
            HookPoint.BEFORE_TURN,
            HookPoint.BEFORE_TOOL,
            HookPoint.BEFORE_COMPLETION,
        )

    def is_after(self) -> bool:
        """True for hooks ordered descending by priority (highest first)."""
        return self in (HookPoint.AFTER_TOOL, HookPoint.AFTER_SESSION)

    def allows_surface_to_human(self) -> bool:
        """Whether :class:`MiddlewareSurfaceToHuman` is permitted here."""
        return self in (HookPoint.BEFORE_TOOL, HookPoint.BEFORE_COMPLETION)

    def allows_force_another_turn(self) -> bool:
        """Whether :class:`MiddlewareForceAnotherTurn` is permitted here."""
        return self is HookPoint.BEFORE_COMPLETION


# ============================================================================
# HookContext (discriminated dataclasses; mutate lists in place where allowed)
# ============================================================================


@dataclass
class HookContextBeforeSession:
    kind: ClassVar[Literal["before_session"]] = "before_session"
    task: Task
    session_id: SessionId

    def point(self) -> HookPoint:
        return HookPoint.BEFORE_SESSION


@dataclass
class HookContextBeforeTurn:
    kind: ClassVar[Literal["before_turn"]] = "before_turn"
    session: SessionState
    turn_number: int

    def point(self) -> HookPoint:
        return HookPoint.BEFORE_TURN


@dataclass
class HookContextBeforeTool:
    kind: ClassVar[Literal["before_tool"]] = "before_tool"
    calls: list[ToolCall]
    turn_number: int

    def point(self) -> HookPoint:
        return HookPoint.BEFORE_TOOL


@dataclass
class HookContextAfterTool:
    kind: ClassVar[Literal["after_tool"]] = "after_tool"
    calls: list[ToolCall]
    results: list[ToolResult]

    def point(self) -> HookPoint:
        return HookPoint.AFTER_TOOL


@dataclass
class HookContextBeforeCompletion:
    kind: ClassVar[Literal["before_completion"]] = "before_completion"
    response: str
    turn_number: int
    session_state: SessionState

    def point(self) -> HookPoint:
        return HookPoint.BEFORE_COMPLETION


@dataclass
class HookContextAfterSession:
    kind: ClassVar[Literal["after_session"]] = "after_session"
    result: RunResult
    session_id: SessionId

    def point(self) -> HookPoint:
        return HookPoint.AFTER_SESSION


HookContext = (
    HookContextBeforeSession
    | HookContextBeforeTurn
    | HookContextBeforeTool
    | HookContextAfterTool
    | HookContextBeforeCompletion
    | HookContextAfterSession
)


# ============================================================================
# MiddlewareDecision (discriminated union on ``kind``)
# ============================================================================


class MiddlewareContinue(_Model):
    kind: Literal["continue"] = "continue"


class MiddlewareContinueWithModification(_Model):
    """Signals that the middleware mutated the borrowed context.

    Semantically equivalent to :class:`MiddlewareContinue` for chain
    control flow — the harness can use this signal for observability
    without re-diffing. The optional ``calls`` field is a Python-specific
    convenience used by the legacy simple-API harness path; the canonical
    spec-rich chain leaves it ``None`` (mutation is in-place on the
    :class:`HookContextBeforeTool.calls` list).
    """

    kind: Literal["continue_with_modification"] = "continue_with_modification"
    calls: list[ToolCall] | None = None


class MiddlewareForceAnotherTurn(_Model):
    """Valid only on :attr:`HookPoint.BEFORE_COMPLETION`.

    The chain concatenates the ``inject`` strings from every middleware
    that returned this and surfaces one combined decision. The chain
    continues running through the remaining BeforeCompletion middleware.
    """

    kind: Literal["force_another_turn"] = "force_another_turn"
    inject: str


class MiddlewareHalt(_Model):
    kind: Literal["halt"] = "halt"
    reason: str


class MiddlewareSurfaceToHuman(_Model):
    """Valid only on :attr:`HookPoint.BEFORE_TOOL` and
    :attr:`HookPoint.BEFORE_COMPLETION`.

    First occurrence in priority order wins; remaining middleware do not
    run.
    """

    kind: Literal["surface_to_human"] = "surface_to_human"
    request: HumanRequest


MiddlewareDecision = Annotated[
    MiddlewareContinue
    | MiddlewareContinueWithModification
    | MiddlewareForceAnotherTurn
    | MiddlewareHalt
    | MiddlewareSurfaceToHuman,
    Field(discriminator="kind"),
]


# ============================================================================
# Errors
# ============================================================================


class MiddlewareError(SporeError):
    """Root of every error raised by a :class:`MiddlewareChain`."""

    kind: ClassVar[str] = "MiddlewareError"


class AlreadyRegisteredError(MiddlewareError):
    kind: ClassVar[str] = "AlreadyRegistered"

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"middleware already registered: {name!r}")


class NoHooksError(MiddlewareError):
    kind: ClassVar[str] = "NoHooks"

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"middleware {name!r} declared zero hooks")


class IllegalDecisionError(MiddlewareError):
    kind: ClassVar[str] = "IllegalDecision"

    def __init__(self, name: str, hook: HookPoint, decision: str) -> None:
        self.name = name
        self.hook = hook
        self.decision = decision
        super().__init__(
            f"middleware {name!r} returned {decision} from {hook.value} which does not allow it"
        )


# ============================================================================
# Protocols
# ============================================================================


@runtime_checkable
class Middleware(Protocol):
    """A single middleware. Implementations satisfy this Protocol
    structurally; they do not need to subclass it."""

    async def handle(self, ctx: HookContext) -> MiddlewareDecision: ...

    def hooks(self) -> list[HookPoint]: ...

    def name(self) -> str: ...

    # ``priority`` has a default of ``0`` — implementors may omit it.
    # def priority(self) -> int: ...


def _priority_of(m: Middleware) -> int:
    """Read a middleware's priority, defaulting to ``0`` when absent."""
    fn = getattr(m, "priority", None)
    if fn is None:
        return 0
    try:
        return int(fn())
    except TypeError:
        return 0


@runtime_checkable
class MiddlewareChain(Protocol):
    """Registry + fan-out evaluator."""

    async def register(self, middleware: Middleware) -> None: ...

    async def fire_before_session(
        self, task: Task, session_id: SessionId
    ) -> MiddlewareDecision: ...

    async def fire_before_turn(
        self, session: SessionState, turn_number: int
    ) -> MiddlewareDecision: ...

    async def fire_before_tool(
        self, calls: list[ToolCall], turn_number: int
    ) -> MiddlewareDecision: ...

    async def fire_after_tool(
        self, calls: list[ToolCall], results: list[ToolResult]
    ) -> MiddlewareDecision: ...

    async def fire_before_completion(
        self, response: str, turn_number: int, state: SessionState
    ) -> MiddlewareDecision: ...

    async def fire_after_session(self, result: RunResult, session_id: SessionId) -> None: ...


# ============================================================================
# StandardMiddlewareChain — reference in-memory implementation
# ============================================================================


@dataclass
class _Entry:
    name: str
    priority: int
    hooks: list[HookPoint]
    middleware: Middleware


def _validate_decision(
    entry: _Entry, hook: HookPoint, decision: MiddlewareDecision
) -> MiddlewareDecision:
    if isinstance(decision, MiddlewareSurfaceToHuman) and not hook.allows_surface_to_human():
        raise IllegalDecisionError(entry.name, hook, "SurfaceToHuman")
    if isinstance(decision, MiddlewareForceAnotherTurn) and not hook.allows_force_another_turn():
        raise IllegalDecisionError(entry.name, hook, "ForceAnotherTurn")
    return decision


class StandardMiddlewareChain:
    """Reference :class:`MiddlewareChain` implementation. In-memory;
    suitable for tests and short-lived processes."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._entries: list[_Entry] = []

    # ── register ───────────────────────────────────────────────────────────

    async def register(self, middleware: Middleware) -> None:
        name = middleware.name()
        hooks = list(middleware.hooks())
        if not hooks:
            raise NoHooksError(name)
        priority = _priority_of(middleware)
        async with self._lock:
            if any(e.name == name for e in self._entries):
                raise AlreadyRegisteredError(name)
            self._entries.append(
                _Entry(name=name, priority=priority, hooks=hooks, middleware=middleware)
            )

    # ── helpers ────────────────────────────────────────────────────────────

    def _eligible(self, hook: HookPoint) -> list[_Entry]:
        v = [e for e in self._entries if hook in e.hooks]
        # Stable secondary sort by name for deterministic ordering.
        if hook.is_after():
            v.sort(key=lambda e: (-e.priority, e.name))
        else:
            v.sort(key=lambda e: (e.priority, e.name))
        return v

    async def _snapshot(self, hook: HookPoint) -> list[_Entry]:
        async with self._lock:
            return self._eligible(hook)

    # ── fire_before_session ────────────────────────────────────────────────

    async def fire_before_session(self, task: Task, session_id: SessionId) -> MiddlewareDecision:
        entries = await self._snapshot(HookPoint.BEFORE_SESSION)
        for entry in entries:
            decision = await entry.middleware.handle(
                HookContextBeforeSession(task=task, session_id=session_id)
            )
            try:
                decision = _validate_decision(entry, HookPoint.BEFORE_SESSION, decision)
            except IllegalDecisionError as e:
                return MiddlewareHalt(reason=str(e))
            if isinstance(decision, MiddlewareContinue | MiddlewareContinueWithModification):
                continue
            return decision
        return MiddlewareContinue()

    # ── fire_before_turn ───────────────────────────────────────────────────

    async def fire_before_turn(self, session: SessionState, turn_number: int) -> MiddlewareDecision:
        entries = await self._snapshot(HookPoint.BEFORE_TURN)
        any_modified = False
        for entry in entries:
            decision = await entry.middleware.handle(
                HookContextBeforeTurn(session=session, turn_number=turn_number)
            )
            try:
                decision = _validate_decision(entry, HookPoint.BEFORE_TURN, decision)
            except IllegalDecisionError as e:
                return MiddlewareHalt(reason=str(e))
            if isinstance(decision, MiddlewareContinue):
                continue
            if isinstance(decision, MiddlewareContinueWithModification):
                any_modified = True
                continue
            return decision
        return MiddlewareContinueWithModification() if any_modified else MiddlewareContinue()

    # ── fire_before_tool ───────────────────────────────────────────────────

    async def fire_before_tool(self, calls: list[ToolCall], turn_number: int) -> MiddlewareDecision:
        entries = await self._snapshot(HookPoint.BEFORE_TOOL)
        any_modified = False
        for entry in entries:
            decision = await entry.middleware.handle(
                HookContextBeforeTool(calls=calls, turn_number=turn_number)
            )
            try:
                decision = _validate_decision(entry, HookPoint.BEFORE_TOOL, decision)
            except IllegalDecisionError as e:
                return MiddlewareHalt(reason=str(e))
            if isinstance(decision, MiddlewareContinue):
                continue
            if isinstance(decision, MiddlewareContinueWithModification):
                any_modified = True
                continue
            return decision
        return MiddlewareContinueWithModification() if any_modified else MiddlewareContinue()

    # ── fire_after_tool ────────────────────────────────────────────────────

    async def fire_after_tool(
        self, calls: list[ToolCall], results: list[ToolResult]
    ) -> MiddlewareDecision:
        entries = await self._snapshot(HookPoint.AFTER_TOOL)
        any_modified = False
        for entry in entries:
            decision = await entry.middleware.handle(
                HookContextAfterTool(calls=calls, results=results)
            )
            try:
                decision = _validate_decision(entry, HookPoint.AFTER_TOOL, decision)
            except IllegalDecisionError as e:
                return MiddlewareHalt(reason=str(e))
            if isinstance(decision, MiddlewareContinue):
                continue
            if isinstance(decision, MiddlewareContinueWithModification):
                any_modified = True
                continue
            return decision
        return MiddlewareContinueWithModification() if any_modified else MiddlewareContinue()

    # ── fire_before_completion ─────────────────────────────────────────────

    async def fire_before_completion(
        self, response: str, turn_number: int, state: SessionState
    ) -> MiddlewareDecision:
        entries = await self._snapshot(HookPoint.BEFORE_COMPLETION)
        injections: list[str] = []
        for entry in entries:
            decision = await entry.middleware.handle(
                HookContextBeforeCompletion(
                    response=response, turn_number=turn_number, session_state=state
                )
            )
            try:
                decision = _validate_decision(entry, HookPoint.BEFORE_COMPLETION, decision)
            except IllegalDecisionError as e:
                return MiddlewareHalt(reason=str(e))
            if isinstance(decision, MiddlewareContinue | MiddlewareContinueWithModification):
                continue
            if isinstance(decision, MiddlewareForceAnotherTurn):
                injections.append(decision.inject)
                # chain continues per spec
                continue
            # Halt or SurfaceToHuman stops the chain.
            return decision
        if injections:
            return MiddlewareForceAnotherTurn(inject="\n".join(injections))
        return MiddlewareContinue()

    # ── fire_after_session ─────────────────────────────────────────────────

    async def fire_after_session(self, result: RunResult, session_id: SessionId) -> None:
        entries = await self._snapshot(HookPoint.AFTER_SESSION)
        for entry in entries:
            # After-session decisions are ignored — the session is already
            # terminating per spec.
            await entry.middleware.handle(
                HookContextAfterSession(result=result, session_id=session_id)
            )


# ============================================================================
# Standard middleware implementations (representative subset)
# ============================================================================


class TracingMiddleware:
    """Lowest-priority observability middleware. Records every hook
    firing. The real implementation forwards to
    :class:`ObservabilityProvider`; this version keeps an in-memory log so
    tests can assert ordering.
    """

    def __init__(self, name: str = "tracing") -> None:
        self._name = name
        self._log: list[tuple[HookPoint, int]] = []

    def entries(self) -> list[tuple[HookPoint, int]]:
        return list(self._log)

    async def handle(self, ctx: HookContext) -> MiddlewareDecision:
        point = ctx.point()
        turn = 0
        if isinstance(ctx, HookContextBeforeTurn | HookContextBeforeTool):
            turn = ctx.turn_number
        elif isinstance(ctx, HookContextBeforeCompletion):
            turn = ctx.turn_number
        self._log.append((point, turn))
        return MiddlewareContinue()

    def hooks(self) -> list[HookPoint]:
        return [
            HookPoint.BEFORE_SESSION,
            HookPoint.BEFORE_TURN,
            HookPoint.BEFORE_TOOL,
            HookPoint.AFTER_TOOL,
            HookPoint.BEFORE_COMPLETION,
            HookPoint.AFTER_SESSION,
        ]

    def priority(self) -> int:
        return -sys.maxsize

    def name(self) -> str:
        return self._name


class PatchToolCallsMiddleware:
    """Repairs syntactically invalid tool calls before they reach the
    registry. The shipped implementation patches empty or whitespace-only
    tool names to a configurable fallback. Runs at the second-lowest
    ``BEFORE_TOOL`` priority so downstream middleware see clean calls.

    Observability (issue #28)
    -------------------------
    This middleware is an always-on, highest-priority action mutator. To keep
    it from silently rewriting calls, **every patch emits a warn-level
    ``PatchSpan``** via an injected :class:`ObservabilityProvider` before the
    patched call proceeds. The span carries the original and patched
    parameters and a classified ``PatchType`` so the trace shows the diff,
    never just the patched call.

    The shared :class:`HookContextBeforeTool` does not carry ``session_id`` /
    ``task_id`` (widening it would ripple across all four language ports), so
    this middleware captures identity at :attr:`HookPoint.BEFORE_SESSION` into
    a local field and reads it at :attr:`HookPoint.BEFORE_TOOL` — the same
    external-identity pattern used by :class:`LoopDetectionMiddleware`.

    Rules enforced: R1 (one warn span per patch), R2 (no patch → no span),
    R3 (original + patched recorded), R4 (patch_type classified), R9 (one span
    per patched call in a batch), R10 (still runs at the highest ``BEFORE_TOOL``
    priority, ``-sys.maxsize + 1``).
    """

    def __init__(
        self,
        fallback_name: str,
        *,
        name: str = "patch-tool-calls",
        observability: ObservabilityProvider | None = None,
    ) -> None:
        self._name = name
        self.fallback_name = fallback_name
        # Optional observability sink. ``None`` keeps construction
        # test-friendly and makes the middleware a no-op observer when unwired.
        self._observability = observability
        # Captured at BeforeSession, read at BeforeTool: the session/task
        # identity needed to stamp a PatchSpan.
        self._identity: tuple[SessionId, TaskId] | None = None
        self._identity_lock = threading.Lock()
        # Monotonic counter so emitted patch spans get distinct ids.
        self._patch_seq = itertools.count()

    def with_observability(self, obs: ObservabilityProvider) -> PatchToolCallsMiddleware:
        """Inject the observability sink (fluent). Patches emitted after this
        is set produce warn-level patch spans (issue #28)."""
        self._observability = obs
        return self

    def clear(self) -> None:
        """Reset captured identity — tests use this to simulate session
        boundaries."""
        with self._identity_lock:
            self._identity = None

    def _emit_patch_event(
        self,
        call: ToolCall,
        original: dict[str, Any],
        patch_type: Any,
    ) -> None:
        obs = self._observability
        if obs is None:
            return
        # Lazy import to avoid the import cycle (observability imports this
        # module for HookPoint / MiddlewareDecision).
        from .memory import Timestamp
        from .observability import PatchSpan, SpanBase, SpanKind, new_span_id

        with self._identity_lock:
            identity = self._identity
        # Identity captured at BeforeSession; fall back to empty ids if a tool
        # patch somehow fires before any session began (defensive — the span
        # still records the diff).
        session_id, task_id = identity if identity is not None else (SessionId(""), TaskId(""))
        seq = next(self._patch_seq)
        ts = Timestamp("")
        base = SpanBase(
            span_id=new_span_id(f"patch-{seq}"),
            session_id=session_id,
            task_id=task_id,
            kind=SpanKind.PATCH,
            started_at=ts,
            ended_at=ts,
        )
        obs.emit_patch(
            PatchSpan(
                base=base,
                call_id=call.id,
                tool_name=call.name,
                original_parameters=original,
                patched_parameters=dict(call.input),
                patch_type=patch_type,
            )
        )

    async def handle(self, ctx: HookContext) -> MiddlewareDecision:
        if isinstance(ctx, HookContextBeforeSession):
            with self._identity_lock:
                self._identity = (ctx.session_id, ctx.task.id)
            return MiddlewareContinue()
        if not isinstance(ctx, HookContextBeforeTool):
            return MiddlewareContinue()
        # Classify the empty-name repair as a dangling tool call and emit a
        # warn-level event per patched call.
        from .observability import PatchTypeDanglingToolCall

        modified = False
        for call in ctx.calls:
            if not call.name.strip():
                # Capture the original parameters before mutating.
                original = dict(call.input)
                call.name = self.fallback_name
                modified = True
                self._emit_patch_event(
                    call,
                    original,
                    PatchTypeDanglingToolCall(reason="empty tool name"),
                )
        return MiddlewareContinueWithModification() if modified else MiddlewareContinue()

    def hooks(self) -> list[HookPoint]:
        return [HookPoint.BEFORE_SESSION, HookPoint.BEFORE_TOOL]

    def priority(self) -> int:
        return -sys.maxsize + 1

    def name(self) -> str:
        return self._name


class LoopDetectionMiddleware:
    """Tracks per-file edit counts (keyed by tool argument ``path``) and
    annotates the tool result with a warning after ``threshold`` repeated
    edits to the same path. State is held in an external dict keyed by
    path — production middleware would key by :class:`SessionId` and
    clear in ``AfterSession``.
    """

    def __init__(self, tool_name: str, threshold: int, *, name: str = "loop-detection") -> None:
        self._name = name
        self.tool_name = tool_name
        self.threshold = threshold
        self._counts: dict[str, int] = {}

    def clear(self) -> None:
        """Reset all counts — tests use this to simulate session boundaries."""
        self._counts.clear()

    async def handle(self, ctx: HookContext) -> MiddlewareDecision:
        if not isinstance(ctx, HookContextAfterTool):
            return MiddlewareContinue()
        modified = False
        for call, result in zip(ctx.calls, ctx.results, strict=False):
            if call.name != self.tool_name:
                continue
            path = ""
            inp = call.input
            if isinstance(inp, dict):
                v = inp.get("path")
                if isinstance(v, str):
                    path = v
            if not path:
                continue
            self._counts[path] = self._counts.get(path, 0) + 1
            count = self._counts[path]
            if count >= self.threshold and isinstance(result.output, ToolOutputSuccess):
                warning = f"[loop-detection] {path} has been edited {count} times — reconsider"
                if "[loop-detection]" not in result.output.content:
                    result.output.content = result.output.content + "\n\n" + warning
                    modified = True
        return MiddlewareContinueWithModification() if modified else MiddlewareContinue()

    def hooks(self) -> list[HookPoint]:
        return [HookPoint.AFTER_TOOL]

    def name(self) -> str:
        return self._name


class PreCompletionChecklistMiddleware:
    """Forces another turn at :attr:`HookPoint.BEFORE_COMPLETION` if the
    agent's final response fails a configured checklist (a list of
    required substrings)."""

    def __init__(
        self,
        required_substrings: list[str],
        *,
        name: str = "pre-completion-checklist",
    ) -> None:
        self._name = name
        self.required_substrings = list(required_substrings)

    async def handle(self, ctx: HookContext) -> MiddlewareDecision:
        if not isinstance(ctx, HookContextBeforeCompletion):
            return MiddlewareContinue()
        missing = [s for s in self.required_substrings if s not in ctx.response]
        if not missing:
            return MiddlewareContinue()
        return MiddlewareForceAnotherTurn(
            inject=("Verification incomplete. Required items not addressed: " + ", ".join(missing))
        )

    def hooks(self) -> list[HookPoint]:
        return [HookPoint.BEFORE_COMPLETION]

    def name(self) -> str:
        return self._name


class TokenBudgetMiddleware:
    """Halts the session if cumulative token spend exceeds the configured
    limit. Real production wires this to ``BudgetSnapshot``; the
    standalone version reads token counts from an internal counter so
    tests can drive it via :meth:`record`."""

    def __init__(self, limit_tokens: int, *, name: str = "token-budget") -> None:
        self._name = name
        self.limit_tokens = limit_tokens
        self.spent_tokens = 0

    def record(self, tokens: int) -> None:
        self.spent_tokens += tokens

    async def handle(self, ctx: HookContext) -> MiddlewareDecision:
        if self.spent_tokens >= self.limit_tokens:
            return MiddlewareHalt(
                reason=f"token budget exhausted: {self.spent_tokens}/{self.limit_tokens}"
            )
        return MiddlewareContinue()

    def hooks(self) -> list[HookPoint]:
        return [HookPoint.BEFORE_TURN]

    def name(self) -> str:
        return self._name


# Avoid unused-import warnings for `Any` (kept for IDE hover usefulness).
_: Any = None


__all__ = [
    "AlreadyRegisteredError",
    "HookContext",
    "HookContextAfterSession",
    "HookContextAfterTool",
    "HookContextBeforeCompletion",
    "HookContextBeforeSession",
    "HookContextBeforeTool",
    "HookContextBeforeTurn",
    "HookPoint",
    "IllegalDecisionError",
    "LoopDetectionMiddleware",
    "Middleware",
    "MiddlewareChain",
    "MiddlewareContinue",
    "MiddlewareContinueWithModification",
    "MiddlewareDecision",
    "MiddlewareError",
    "MiddlewareForceAnotherTurn",
    "MiddlewareHalt",
    "MiddlewareSurfaceToHuman",
    "NoHooksError",
    "PatchToolCallsMiddleware",
    "PreCompletionChecklistMiddleware",
    "StandardMiddlewareChain",
    "TokenBudgetMiddleware",
    "TracingMiddleware",
]
