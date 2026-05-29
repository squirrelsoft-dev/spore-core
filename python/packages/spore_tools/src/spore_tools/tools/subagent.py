"""SubagentTool — wraps a child :class:`spore_core.Harness` as a Tool.

Per spec (issue #5) subagents cannot spawn their own subagents. The depth-1
rule is enforced at construction time by inspecting the child harness's
:class:`ToolRegistry` via :meth:`has_subagent_tools`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import Field

from spore_core.errors import SporeError
from spore_core.harness import (
    ChildPausedState,
    Harness,
    HarnessRunOptions,
    PausedState,
    RunResultFailure,
    RunResultSuccess,
    RunResultWaitingForHuman,
    SandboxProvider,
    SessionId,
    SessionState,
    Task,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
    ToolOutputWaitingForHuman,
    new_session_id,
)
from spore_core.harness import (
    LoopStrategyReAct,
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
        )

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
            loop_strategy=LoopStrategyReAct(max_iterations=16),
        )
        options = HarnessRunOptions(task=task, session_state=seeded_session)

        try:
            result = await asyncio.wait_for(self.harness.run(options), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            secs = int(self.timeout_seconds)
            return ToolOutputError(message=f"subagent timed out after {secs}s", recoverable=True)

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
        raise AssertionError(f"unhandled RunResult: {result!r}")


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
