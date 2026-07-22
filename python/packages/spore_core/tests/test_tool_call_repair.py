"""Tests for the tool-call repair seam (parity plan §5.5).

Mirrors ``rust/crates/spore-core/src/tool_call_repair.rs`` unit tests and the
harness integration test ``tool_call_repair_fixes_bad_bool_arg`` in
``rust/crates/spore-core/src/harness/tests.rs``.
"""

from __future__ import annotations

from typing import Any

from spore_core import (
    AllowAllSandbox,
    AlwaysContinuePolicy,
    HarnessBuilder,
    HarnessRunOptions,
    ModelAgent,
    ModelResponse,
    NoopContextManager,
    AgentId,
    ProviderInfo,
    ReactConfig,
    RecordedExchange,
    ReplayModelInterface,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    StandardToolCallRepair,
    StopReason,
    Task,
    TextBlock,
    TokenUsage,
    ToolOutputError,
    ToolOutputSuccess,
    ToolSchema,
    ToolUseBlock,
    ProseResponse,
    coerce_tool_args,
    detect_prose_tool_call_failure,
)
from spore_core.model import ToolCall as ModelToolCall


def _provider() -> ProviderInfo:
    return ProviderInfo(name="anthropic", model_id="replay", context_window=200_000)


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=3, output_tokens=4)


def _replay_agent(responses: list[ModelResponse]) -> ModelAgent:
    from spore_core import ModelRequest

    exchanges = [
        RecordedExchange(request=ModelRequest(stream=True), response=r, provider="anthropic")
        for r in responses
    ]
    return ModelAgent(AgentId("repair-agent"), ReplayModelInterface(exchanges, _provider()))


# ---------------------------------------------------------------------------
# coerce_tool_args
# ---------------------------------------------------------------------------


def _schema(props: dict[str, dict[str, Any]]) -> ToolSchema:
    return ToolSchema(
        name="t", description="", input_schema={"type": "object", "properties": props}
    )


def test_coerce_bool_string_to_bool() -> None:
    r = coerce_tool_args({"flag": "true"}, _schema({"flag": {"type": "boolean"}}))
    assert r == {"flag": True}


def test_coerce_numeric_string_to_integer() -> None:
    r = coerce_tool_args({"n": "42"}, _schema({"n": {"type": "integer"}}))
    assert r == {"n": 42}


def test_coerce_float_string_to_number() -> None:
    r = coerce_tool_args({"n": "3.14"}, _schema({"n": {"type": "number"}}))
    assert r == {"n": 3.14}


def test_coerce_json_array_string_to_array() -> None:
    r = coerce_tool_args({"items": '["a", "b"]'}, _schema({"items": {"type": "array"}}))
    assert r == {"items": ["a", "b"]}


def test_coerce_already_valid_returns_none() -> None:
    assert coerce_tool_args({"flag": True}, _schema({"flag": {"type": "boolean"}})) is None
    assert coerce_tool_args({"n": 42}, _schema({"n": {"type": "integer"}})) is None


def test_coerce_case_insensitive_bool() -> None:
    assert coerce_tool_args({"flag": "FALSE"}, _schema({"flag": {"type": "boolean"}})) == {
        "flag": False
    }


def test_coerce_quoted_key_normalized() -> None:
    r = coerce_tool_args({'"flag"': "true"}, _schema({"flag": {"type": "boolean"}}))
    assert r == {"flag": True}


def test_coerce_non_dict_returns_none() -> None:
    assert coerce_tool_args("not a dict") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# detect_prose_response
# ---------------------------------------------------------------------------


def test_detect_prose_response_action_phrase() -> None:
    assert isinstance(
        detect_prose_tool_call_failure("I'll use the bash tool now", True), ProseResponse
    )


def test_detect_prose_response_no_tools_advertised() -> None:
    assert detect_prose_tool_call_failure("I'll use the bash tool", False) is None


def test_detect_prose_response_bare_final_answer() -> None:
    assert detect_prose_tool_call_failure("The answer is 42.", True) is None


def test_detect_prose_response_empty() -> None:
    assert detect_prose_tool_call_failure("   ", True) is None


# ---------------------------------------------------------------------------
# StandardToolCallRepair
# ---------------------------------------------------------------------------


def test_standard_repair_repairs_bool_arg() -> None:
    repair = StandardToolCallRepair()
    call = ModelToolCall(id="c1", name="t", input={"flag": "false"})
    out = repair.repair(call, "invalid type", _schema({"flag": {"type": "boolean"}}))
    assert out is not None
    assert out.input == {"flag": False}
    assert out.id == "c1"
    assert out.name == "t"


def test_standard_repair_returns_none_when_no_change() -> None:
    repair = StandardToolCallRepair()
    call = ModelToolCall(id="c1", name="t", input={"flag": True})
    assert repair.repair(call, "err", _schema({"flag": {"type": "boolean"}})) is None


# ---------------------------------------------------------------------------
# Harness integration: repair fixes a bad bool arg mid-loop
# ---------------------------------------------------------------------------


class _BoolTool:
    """A tool that rejects a string bool arg, accepts a real bool."""

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def name(self) -> str:
        return "set_flag"

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    async def execute(self, call: Any, sandbox: Any, ctx: Any) -> Any:
        flag = call.input.get("flag")
        self.seen.append(flag)
        if not isinstance(flag, bool):
            return ToolOutputError(
                message="invalid parameters for tool `set_flag`: flag must be bool",
                recoverable=True,
            )
        return ToolOutputSuccess(content=f"flag={flag}")


async def test_harness_repair_fixes_bad_bool_arg() -> None:
    from dataclasses import dataclass

    from spore_core import RegistryToolSchema, ToolAnnotations

    @dataclass
    class _Bundle:
        implementation: Any
        schema: Any

    schema = RegistryToolSchema(
        name="set_flag",
        description="sets a flag",
        parameters={
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
            "required": ["flag"],
        },
        annotations=ToolAnnotations(),
    )
    bool_tool = _BoolTool()
    agent = _replay_agent(
        [
            ModelResponse(
                content=[ToolUseBlock(id="toolu_1", name="set_flag", input={"flag": "true"})],
                usage=_usage(),
                stop_reason=StopReason.TOOL_USE,
            ),
            ModelResponse(
                content=[TextBlock(text="done")],
                usage=_usage(),
                stop_reason=StopReason.END_TURN,
            ),
        ]
    )
    builder = (
        HarnessBuilder(
            agent=agent,
            tool_registry=ScriptedToolRegistry(),
            sandbox=AllowAllSandbox(),
            context_manager=NoopContextManager(),
            termination_policy=AlwaysContinuePolicy(),
        )
        .tool(_Bundle(implementation=bool_tool, schema=schema))
        .tool_call_repair(StandardToolCallRepair(), max_repair_attempts=1)
    )
    h = builder.build()
    r = await h.run(HarnessRunOptions(Task.new("set it", SessionId("s1"), ReactConfig.per_loop(5))))
    assert isinstance(r, RunResultSuccess), r
    # The repair coerced "true" -> True: the tool first saw the string (rejected),
    # then the repaired bool (accepted) — proving the repair pass fired.
    assert bool_tool.seen == ["true", True], bool_tool.seen
