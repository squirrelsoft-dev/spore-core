"""SubagentTool — wraps a child :class:`spore_core.Harness` as a Tool.

Per spec (issue #5) subagents cannot spawn their own subagents. The depth-1
rule is enforced at construction time by inspecting the child harness's
:class:`ToolRegistry` via :meth:`has_subagent_tools`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field

from spore_core.errors import SporeError
from spore_core.harness import (
    ChildPausedState,
    ConsultHandlerEntry,
    ConsultOverflowPolicyEscalateToHuman,
    ConsultOverflowPolicySoftFail,
    ConsultRequest,
    ConsultResponse,
    ConsultResponseAnswer,
    ConsultResponseBudgetExhausted,
    Harness,
    HarnessRunOptions,
    HarnessSignalAbort,
    HumanRequestReview,
    PausedState,
    RunResult,
    RunResultConsult,
    RunResultEscalate,
    RunResultFailure,
    RunResultSuccess,
    RunResultWaitingForHuman,
    SandboxProvider,
    SessionId,
    SessionState,
    Task,
    ToolOutput,
    ToolOutputError,
    ToolOutputEscalate,
    ToolOutputSuccess,
    ToolOutputWaitingForHuman,
    new_session_id,
)
from spore_core.harness import (
    ReactConfig,
    _Model,  # type: ignore[attr-defined]  # internal pydantic base
)
from spore_core.model import ToolCall
from spore_core.tool_registry import ToolContext, ToolRegistry


# ============================================================================
# ContextSharing — discriminated union on ``kind``
# ============================================================================


class ContextSharingIsolated(_Model):
    kind: Literal["isolated"] = "isolated"


class ContextSharingSharedSession(_Model):
    kind: Literal["shared_session"] = "shared_session"
    session_id: SessionId


class ContextSharingSummaryHandoff(_Model):
    kind: Literal["summary_handoff"] = "summary_handoff"
    summary: str


ContextSharing = Annotated[
    ContextSharingIsolated | ContextSharingSharedSession | ContextSharingSummaryHandoff,
    Field(discriminator="kind"),
]


# ============================================================================
# BuildError
# ============================================================================


class BuildError(SporeError, ValueError):
    """Raised when constructing a :class:`SubagentTool` with an invalid
    configuration — e.g. child registry already contains subagent tools."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"invalid configuration: {reason}")


# ============================================================================
# SubagentTool
# ============================================================================


@dataclass
class SubagentTool:
    """A tool that runs a child :class:`Harness` for a single instruction."""

    _name: str
    description: str
    input_schema: dict[str, Any]
    timeout_seconds: float
    context_sharing: Any  # ContextSharing instance
    harness: Harness
    # Per-kind consult handlers (issue #114, seam A1). Empty (the default) means
    # consults are NOT mediated here — a child :class:`RunResultConsult` surfaces
    # as :class:`ToolOutputConsult` to the parent (R6). Populated via
    # :meth:`with_consult_handlers` (typically from the orchestrator's
    # ``HarnessConfig.consult_handlers``).
    consult_handlers: dict[str, ConsultHandlerEntry] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        timeout_seconds: float,
        context_sharing: Any,
        harness: Harness,
        child_registry: ToolRegistry,
        consult_handlers: dict[str, ConsultHandlerEntry] | None = None,
    ) -> SubagentTool:
        """Construct a :class:`SubagentTool`.

        Raises :class:`BuildError` if ``child_registry`` already contains a
        subagent-flagged tool (depth-1 rule).
        """

        if child_registry.has_subagent_tools():
            raise BuildError(reason="child harness must not contain SubagentTool (depth-1 rule)")
        return cls(
            _name=name,
            description=description,
            input_schema=input_schema,
            timeout_seconds=timeout_seconds,
            context_sharing=context_sharing,
            harness=harness,
            consult_handlers=dict(consult_handlers) if consult_handlers is not None else {},
        )

    def with_consult_handlers(
        self, consult_handlers: dict[str, ConsultHandlerEntry]
    ) -> SubagentTool:
        """Install the per-kind consult handlers (issue #114, seam A1). With
        handlers installed, this tool MEDIATES a child consult internally (R2/R3)
        instead of letting it surface; without them, a child consult surfaces as
        :class:`ToolOutputConsult` (R6). Returns ``self`` for chaining."""
        self.consult_handlers = dict(consult_handlers)
        return self

    # ---- Tool protocol --------------------------------------------------

    def name(self) -> str:
        return self._name

    def is_subagent_tool(self) -> bool:
        return True

    def may_produce_large_output(self) -> bool:
        return False

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        instruction = call.input.get("instruction") if isinstance(call.input, dict) else None
        if not isinstance(instruction, str):
            return ToolOutputError(
                message="invalid parameters: missing `instruction`",
                recoverable=True,
            )

        seeded_session: SessionState | None = None
        sharing = self.context_sharing
        if isinstance(sharing, ContextSharingSharedSession):
            session_id: SessionId = sharing.session_id
        elif isinstance(sharing, ContextSharingSummaryHandoff):
            session_id = new_session_id()
            seeded_session = SessionState(extras={"subagent_handoff_summary": sharing.summary})
        else:
            # Isolated (default)
            session_id = new_session_id()

        task = Task.new(
            instruction=instruction,
            session_id=session_id,
            loop_strategy=ReactConfig.per_loop(16),
        )
        options = HarnessRunOptions(task=task, session_state=seeded_session)

        try:
            result = await asyncio.wait_for(self.harness.run(options), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            secs = int(self.timeout_seconds)
            return ToolOutputError(message=f"subagent timed out after {secs}s", recoverable=True)

        # Per-kind consult counters (issue #114, R4). Each consult of a given
        # kind decrements its remaining budget; the (budget+1)th triggers the
        # overflow policy.
        consult_counts: dict[str, int] = {}

        # A1 mediation loop: drive the full consult cycle internally. On a child
        # :class:`RunResultConsult`, mediate (route -> run handler -> resume) and
        # continue until the child reaches a terminal result.
        while True:
            if isinstance(result, RunResultSuccess):
                return ToolOutputSuccess(content=result.output, truncated=False)
            if isinstance(result, RunResultFailure):
                return ToolOutputError(
                    message=f"subagent failed: {result.reason.kind}",
                    recoverable=True,
                )
            if isinstance(result, RunResultWaitingForHuman):
                child = _child_state_from_paused(result.state, call.id)
                return ToolOutputWaitingForHuman(child_state=child, request=result.request)
            # A subagent escalation (issue #80) propagates as a tool-side
            # escalation: the parent harness terminates cleanly and hands the
            # signal up to its own caller.
            if isinstance(result, RunResultEscalate):
                return ToolOutputEscalate(signal=result.signal)
            # Mid-loop consult (issue #114, R2): mediate it here — never bubble it
            # to the parent orchestrator's model.
            if isinstance(result, RunResultConsult):
                outcome = await self._mediate_consult(
                    result.state, result.request, consult_counts, call.id
                )
                if isinstance(outcome, _Resume):
                    # The handler answered (or soft-failed): loop again on the new
                    # result (R3/R5a).
                    result = outcome.result
                    continue
                # Terminal mapping surfaced to the parent (R5b/R6).
                return outcome.output
            raise AssertionError(f"unhandled RunResult: {result!r}")

    async def _mediate_consult(
        self,
        state: PausedState,
        request: ConsultRequest,
        counts: dict[str, int],
        parent_call_id: str,
    ) -> _Resume | _Terminal:
        """Mediate one child consult (issue #114, seam A1). Routes by ``kind``,
        enforces the per-kind budget, runs the handler as the ORCHESTRATOR's
        direct child (R7), and resumes the worker — OR applies the overflow
        policy / graceful degradation."""
        # R6: no matching handler (empty map or unknown kind) -> Escalate. Loud,
        # not silent. The parent harness terminates cleanly.
        entry = self.consult_handlers.get(request.kind)
        if entry is None:
            return _Terminal(
                ToolOutputEscalate(
                    signal=HarnessSignalAbort(
                        reason=f"no consult handler registered for kind {request.kind!r}"
                    )
                )
            )

        # R4: per-kind budget. ``used`` is the number of consults of this kind
        # ALREADY mediated. The handler runs while ``used < budget``; the
        # (budget+1)th consult overflows.
        used = counts.get(request.kind, 0)
        if used >= entry.budget:
            # R5: overflow policy.
            if isinstance(entry.overflow, ConsultOverflowPolicySoftFail):
                # R5a: resume the worker with a BudgetExhausted response so it
                # finishes with what it has.
                response: ConsultResponse = ConsultResponseBudgetExhausted(
                    message=(
                        f"consult budget for kind {request.kind!r} exhausted; "
                        "proceed without further help"
                    )
                )
                nxt = await self.harness.resume_consult(state, response, None)
                return _Resume(nxt)
            if isinstance(entry.overflow, ConsultOverflowPolicyEscalateToHuman):
                # R5b: convert the over-budget consult into a human pause so the
                # host decides. The parent sees ToolOutputWaitingForHuman.
                child = _child_state_from_paused(state, parent_call_id)
                return _Terminal(
                    ToolOutputWaitingForHuman(
                        child_state=child,
                        request=HumanRequestReview(
                            content=(
                                f"consult budget for kind {request.kind!r} exhausted. "
                                f"situation: {request.situation} | question: {request.question}"
                            )
                        ),
                    )
                )
            raise AssertionError(f"unhandled ConsultOverflowPolicy: {entry.overflow!r}")

        # R3/R7: run the handler harness as the orchestrator's direct child
        # (depth-1), WITHOUT the orchestrator model. The handler's instruction is
        # the consult request rendered to text.
        counts[request.kind] = used + 1
        instruction = _render_consult_instruction(request)
        task = Task.new(
            instruction=instruction,
            session_id=new_session_id(),
            loop_strategy=ReactConfig.per_loop(16),
        )
        handler_result = await entry.handler.run(HarnessRunOptions(task=task))
        if isinstance(handler_result, RunResultSuccess):
            answer = handler_result.output
        else:
            # A handler that does not cleanly complete still must not stall the
            # worker — feed its failure text back as the consult answer so the
            # worker can adapt. (The orchestrator model is never involved.)
            answer = f"consult handler did not complete cleanly: {handler_result!r}"
        nxt = await self.harness.resume_consult(state, ConsultResponseAnswer(text=answer), None)
        return _Resume(nxt)


@dataclass
class _Resume:
    """The worker was resumed; carry the new :class:`RunResult` for the next loop
    turn (issue #114)."""

    result: RunResult


@dataclass
class _Terminal:
    """A terminal :class:`ToolOutput` to surface to the parent — overflow-escalate
    or misconfiguration (issue #114)."""

    output: ToolOutput


def _render_consult_instruction(request: ConsultRequest) -> str:
    """Render a :class:`ConsultRequest` to a handler instruction string (issue
    #114). Mirrors the Rust ``render_consult_instruction``."""
    return (
        f"A worker agent is requesting help (kind: {request.kind}).\n\n"
        f"Situation: {request.situation}\n\n"
        f"Attempts so far: {request.attempts}\n\n"
        f"Question: {request.question}"
    )


def _child_state_from_paused(state: PausedState, parent_tool_call_id: str) -> ChildPausedState:
    return ChildPausedState(
        session_id=state.session_id,
        task_id=state.task_id,
        turn_number=state.turn_number,
        session_state=state.session_state,
        pending_tool_calls=list(state.pending_tool_calls),
        approved_results=list(state.approved_results),
        human_request=state.human_request,
        task=state.task,
        budget_used=state.budget_used,
        parent_tool_call_id=parent_tool_call_id,
    )


__all__ = [
    "BuildError",
    "ContextSharing",
    "ContextSharingIsolated",
    "ContextSharingSharedSession",
    "ContextSharingSummaryHandoff",
    "SubagentTool",
]
