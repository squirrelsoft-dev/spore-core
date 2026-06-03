"""End-to-end coverage for adaptive prompt-based tool-calling escalation (#111).

Drives a full ``conversational`` harness with a scripted ``MockModelInterface``
to prove the escalation path the unit tests can only exercise in pieces:
native-first, then automatic switch to prompt-based mode after a prose response,
then ``<tool_call>`` markers parsed into a real dispatch — with no model lists
and no manual wrapping. Ports ``tests/prompt_tool_call_escalation.rs``.
"""

from __future__ import annotations

from spore_core import (
    HarnessBuilder,
    HarnessRunOptions,
    ProviderInfo,
    RunResultSuccess,
    Task,
    ToolCall,
)
from spore_core.harness import ToolOutput, ToolOutputSuccess
from spore_core.model import (
    MockModelInterface,
    ModelResponse,
    StopReason,
    TextBlock,
    TokenUsage,
    ToolSchema,
    ToolUseBlock,
)


class _CountingCalculator:
    """Harness-loop tool registry advertising a single ``calculator`` tool and
    counting how many times it is dispatched."""

    def __init__(self) -> None:
        self.hits = 0

    def schemas(self) -> list[ToolSchema]:
        return [
            ToolSchema(
                name="calculator",
                description="Evaluate a math expression",
                input_schema={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            )
        ]

    def is_always_halt(self, tool_name: str) -> bool:
        return False

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        self.hits += 1
        return ToolOutputSuccess.success("4")


def provider() -> ProviderInfo:
    return ProviderInfo(name="mock", model_id="mock-1", context_window=8192)


def usage() -> TokenUsage:
    return TokenUsage(input_tokens=1, output_tokens=1)


def text(t: str) -> ModelResponse:
    return ModelResponse(
        content=[TextBlock(text=t)], usage=usage(), stop_reason=StopReason.END_TURN
    )


async def test_prose_response_escalates_to_prompt_based_tool_call() -> None:
    """Turn 1 prose with action-intent → escalate. Turn 2 (now in prompt mode)
    emits a ``<tool_call>`` marker → parsed + dispatched. Turn 3 final answer."""
    reg = _CountingCalculator()
    model = MockModelInterface(provider())
    model.push_response(text("Sure — I'll use the calculator tool to compute 2+2."))
    model.push_response(
        text('<tool_call><name>calculator</name><input>{"expression":"2+2"}</input></tool_call>')
    )
    model.push_response(text("The answer is 4"))

    harness = HarnessBuilder.conversational(model).tool_registry(reg).build()
    result = await harness.run(HarnessRunOptions(Task.simple("What is 2+2?")))

    assert isinstance(result, RunResultSuccess)
    # Reaching turn 3's answer proves: turn 1 did NOT terminate the run
    # (escalation fired), and turn 2's marker text was parsed into a real tool
    # call rather than being treated as a prose final answer.
    assert result.output == "The answer is 4"
    assert result.turns == 3
    assert reg.hits == 1


async def test_plain_final_answer_does_not_escalate() -> None:
    """Native path unaffected: tools advertised, but the model gives a plain
    final answer with no action-intent language. The conservative heuristic must
    NOT escalate — the run completes on turn 1."""
    reg = _CountingCalculator()
    model = MockModelInterface(provider())
    model.push_response(text("The answer is 4."))

    harness = HarnessBuilder.conversational(model).tool_registry(reg).build()
    result = await harness.run(HarnessRunOptions(Task.simple("What is 2+2?")))

    assert isinstance(result, RunResultSuccess)
    assert result.output == "The answer is 4."
    assert result.turns == 1
    assert reg.hits == 0


async def test_native_tool_call_path_unaffected() -> None:
    """Native tool calling unaffected: a model that emits a native tool-use block
    (tools advertised) dispatches normally through the adaptive wrapper while the
    flag is unset — no prompt injection, no marker parsing involved."""
    reg = _CountingCalculator()
    model = MockModelInterface(provider())
    model.push_response(
        ModelResponse(
            content=[ToolUseBlock(id="c1", name="calculator", input={"expression": "2+2"})],
            usage=usage(),
            stop_reason=StopReason.TOOL_USE,
        )
    )
    model.push_response(text("The answer is 4"))

    harness = HarnessBuilder.conversational(model).tool_registry(reg).build()
    result = await harness.run(HarnessRunOptions(Task.simple("What is 2+2?")))

    assert isinstance(result, RunResultSuccess)
    assert result.output == "The answer is 4"
    assert result.turns == 2
    assert reg.hits == 1
