"""The two consult tools the analysis worker calls to escalate mid-loop
(issue #114).

Both lower to :meth:`ToolOutputConsult.consult` with a ``kind`` tag; the
``analysis_worker`` :class:`SubagentTool` mediates by ``kind`` (research →
research_worker, advice → advisor) using the per-kind budgets + overflow
policies installed via :meth:`SubagentTool.with_consult_handlers`.

Neither tool captures any host state — each simply renders its call input into a
:class:`ConsultRequest` and returns :class:`ToolOutputConsult`. The worker
harness pauses (:class:`RunResultConsult`) and ``SubagentTool`` resumes it with
the handler's answer (or a ``BudgetExhausted`` message). So these are defined
with :func:`define_tool` — no closed-over state needed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from spore_core.harness import ConsultRequest, SandboxProvider, ToolOutput, ToolOutputConsult
from spore_core.tool_registry import ToolContext
from spore_tools import StandardTool, define_tool

#: Routing key for the research consult ladder (→ research_worker, web_search).
KIND_RESEARCH = "research"
#: Routing key for the advice consult ladder (→ advisor, cloud model).
KIND_ADVICE = "advice"


class ConsultInput(BaseModel):
    """Shared input shape for both consult tools: the worker describes where it
    is stuck and the concrete question it wants answered. ``attempts`` is
    advisory — the harness enforces the per-kind budget independently."""

    situation: str = Field(description="Free-form description of where you are stuck or uncertain.")
    question: str = Field(description="The concrete question you want answered.")
    attempts: int = Field(
        default=0, description="How many times you have already tried (advisory only)."
    )


async def _research(input: ConsultInput, sandbox: SandboxProvider, ctx: ToolContext) -> ToolOutput:
    _ = (sandbox, ctx)
    return ToolOutputConsult.consult(
        ConsultRequest(
            kind=KIND_RESEARCH,
            situation=input.situation,
            attempts=input.attempts,
            question=input.question,
        )
    )


async def _advice(input: ConsultInput, sandbox: SandboxProvider, ctx: ToolContext) -> ToolOutput:
    _ = (sandbox, ctx)
    return ToolOutputConsult.consult(
        ConsultRequest(
            kind=KIND_ADVICE,
            situation=input.situation,
            attempts=input.attempts,
            question=input.question,
        )
    )


def research_best_practices_tool() -> StandardTool:
    """``research_best_practices`` → ``kind="research"``. Routed to the research
    worker (web_search). Budget 5, overflow ``SoftFail``: on exhaustion the
    worker resumes with ``BudgetExhausted`` and finishes on general knowledge.
    Looking up an idiom is normal, not distress, so it never reaches the human."""
    return define_tool(
        name="research_best_practices",
        description=(
            "Ask a research helper to web-search current best practices or idioms when you are "
            "unsure whether a pattern is a real defect. Pass `situation` and a focused `question`. "
            "Returns cited findings; use sparingly."
        ),
        input_model=ConsultInput,
        execute=_research,
    )


def consult_advisor_tool() -> StandardTool:
    """``consult_advisor`` → ``kind="advice"``. Routed to the advisor (a stronger
    cloud model with ``read_file``/``grep``). Budget 3, overflow
    ``EscalateToHuman``: on exhaustion the consult converts to
    :class:`RunResultWaitingForHuman` and the REPL surfaces the three-choice
    ladder."""
    return define_tool(
        name="consult_advisor",
        description=(
            "Ask a senior advisor agent (a stronger model that can read_file/grep the repo) when "
            "you are stuck on whether a finding is real or how to rank its severity. Pass "
            "`situation` and a concrete `question`. Reserve for genuine uncertainty — the advisor "
            "budget is small."
        ),
        input_model=ConsultInput,
        execute=_advice,
    )


__all__ = [
    "KIND_ADVICE",
    "KIND_RESEARCH",
    "ConsultInput",
    "consult_advisor_tool",
    "research_best_practices_tool",
]
