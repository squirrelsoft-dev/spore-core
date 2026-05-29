"""Tier-3 control tools (#81): tools that drive the harness via the escalation
/ clarification protocols rather than returning ordinary data.

Mirrors ``rust/crates/spore-core/src/tools/control.rs``.

* :class:`EnterPlanModeTool` (``enter_plan_mode``) →
  :class:`ToolOutputEscalate` ``{ HarnessSignalEnterPlanMode { context } }``.
* :class:`ExitPlanModeTool` (``exit_plan_mode``) →
  :class:`ToolOutputEscalate` ``{ HarnessSignalExitPlanMode { plan } }``. The
  plan is a structured tool param deserialized DIRECTLY into the existing
  :class:`~spore_core.hooks.PlanArtifact` (issue #81, Q4a — no stub).
* :class:`AskUserQuestionTool` (``ask_user_question``) →
  :class:`ToolOutputAwaitingClarification` ``{ question, options }`` (issue #81,
  Q4b). The harness loop pauses with :class:`HumanRequestClarification`.
* :class:`AbortTool` (``abort``) →
  :class:`ToolOutputEscalate` ``{ HarnessSignalAbort { reason } }``.
"""

from __future__ import annotations

from spore_core.harness import (
    HarnessSignalAbort,
    HarnessSignalEnterPlanMode,
    HarnessSignalExitPlanMode,
    SandboxProvider,
    ToolOutput,
    ToolOutputAwaitingClarification,
    ToolOutputEscalate,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

from .error import ToolExecutionError
from .params import (
    AbortParams,
    AskUserQuestionParams,
    EnterPlanModeParams,
    ExitPlanModeParams,
    parse_params,
)


# ============================================================================
# EnterPlanMode
# ============================================================================


class EnterPlanModeTool:
    NAME = "enter_plan_mode"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Request entry into plan mode, seeding the planner with context",
            parameters={
                "type": "object",
                "properties": {"context": {"type": "string"}},
            },
            annotations=ToolAnnotations(),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(EnterPlanModeParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        return ToolOutputEscalate(signal=HarnessSignalEnterPlanMode(context=params.context))


# ============================================================================
# ExitPlanMode
# ============================================================================


class ExitPlanModeTool:
    NAME = "exit_plan_mode"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Submit the produced plan and request exit from plan mode",
            parameters={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "object",
                        "properties": {
                            "tasks": {"type": "array", "items": {"type": "string"}},
                            "rationale": {"type": "string"},
                        },
                        "required": ["tasks"],
                    },
                },
                "required": ["plan"],
            },
            annotations=ToolAnnotations(),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(ExitPlanModeParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        return ToolOutputEscalate(signal=HarnessSignalExitPlanMode(plan=params.plan))


# ============================================================================
# AskUserQuestion
# ============================================================================


class AskUserQuestionTool:
    NAME = "ask_user_question"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Ask the user a clarifying question (optionally with fixed choices)",
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question"],
            },
            annotations=ToolAnnotations(),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(AskUserQuestionParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        return ToolOutputAwaitingClarification(question=params.question, options=params.options)


# ============================================================================
# Abort
# ============================================================================


class AbortTool:
    NAME = "abort"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Request a graceful abort of the run with a reason",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
            annotations=ToolAnnotations(),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(AbortParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        return ToolOutputEscalate(signal=HarnessSignalAbort(reason=params.reason))


__all__ = [
    "AbortTool",
    "AskUserQuestionTool",
    "EnterPlanModeTool",
    "ExitPlanModeTool",
]
