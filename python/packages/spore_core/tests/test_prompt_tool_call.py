"""Unit tests for the adaptive prompt-based tool-calling fallback (#111).

Ports the Rust inline tests from ``prompt_tool_call.rs`` and the prose-detection
tests from ``tool_call_repair.rs``. Covers injection, parsing, the always-on and
adaptive wrappers, and prose detection. The end-to-end harness escalation lives
in ``test_harness_escalation.py``.
"""

from __future__ import annotations

import pytest

from spore_core.model import (
    Message,
    MockModelInterface,
    ModelParams,
    ModelRequest,
    ModelResponse,
    ProviderInfo,
    Role,
    StopReason,
    TextBlock,
    TextContent,
    ThinkingBlock,
    TokenUsage,
    ToolSchema,
    ToolUseBlock,
)
from spore_core.prompt_tool_call import (
    AdaptiveToolCallModelInterface,
    PromptBasedToolCallModelInterface,
    PromptToolCallFlag,
    build_tool_prompt,
    detect_prose_response,
    inject_tool_prompt,
    parse_prose_response,
)


def provider() -> ProviderInfo:
    return ProviderInfo(name="test", model_id="test-1", context_window=4096)


def tool_schema() -> ToolSchema:
    return ToolSchema(
        name="calculator",
        description="evaluate math",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )


def req_with_tools(system: str | None) -> ModelRequest:
    messages: list[Message] = []
    if system is not None:
        messages.append(Message(role=Role.SYSTEM, content=TextContent(text=system)))
    messages.append(Message(role=Role.USER, content=TextContent(text="what is 2+2?")))
    return ModelRequest(
        messages=messages,
        tools=[tool_schema()],
        params=ModelParams(),
        stream=False,
    )


def usage() -> TokenUsage:
    return TokenUsage(input_tokens=1, output_tokens=1)


def prose(text: str, stop: StopReason) -> ModelResponse:
    return ModelResponse(content=[TextBlock(text=text)], usage=usage(), stop_reason=stop)


# --- build_tool_prompt: exact bytes -----------------------------------------


def test_build_tool_prompt_exact_bytes() -> None:
    schema = ToolSchema(
        name="calculator",
        description="evaluate math",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )
    out = build_tool_prompt([schema])
    expected = (
        "You have access to the following tools. Use them when they would help "
        "complete the task.\n\n"
        "<available_tools>\n"
        "<tool>\n"
        "  <name>calculator</name>\n"
        "  <description>evaluate math</description>\n"
        '  <input_schema>{"properties":{"expression":{"type":"string"}},'
        '"required":["expression"],"type":"object"}</input_schema>\n'
        "</tool>\n"
        "</available_tools>\n\n"
        "When you want to use a tool, respond with ONLY the following format and "
        "nothing else:\n"
        '<tool_call>\n  <name>tool_name_here</name>\n  <input>{"key": "value"}</input>\n'
        "</tool_call>\n\n"
        "When you have a final answer that does not require a tool, respond "
        "normally in prose."
    )
    assert out == expected


def test_build_tool_prompt_sorts_schema_keys_recursively() -> None:
    """A 2+ property schema with keys deliberately out of alphabetical order must
    render with object keys SORTED at every nesting level, matching Rust
    (BTreeMap-backed serde_json) and Go (json.Marshal of a map). Single-property
    schemas coincide regardless; this is the divergence that single-prop tests
    miss (#111)."""
    schema = ToolSchema(
        name="t",
        description="d",
        # Insertion order is intentionally NOT alphabetical, at both levels:
        # top-level "type" before "properties" before "required"; and inside
        # properties "zeta" before "alpha".
        input_schema={
            "type": "object",
            "properties": {
                "zeta": {"type": "number"},
                "alpha": {"type": "string"},
            },
            "required": ["zeta", "alpha"],
        },
    )
    out = build_tool_prompt([schema])
    start = out.index("<input_schema>") + len("<input_schema>")
    end = out.index("</input_schema>")
    rendered = out[start:end]
    # Top-level object keys sorted: properties < required < type. Inside
    # properties: alpha < zeta. The "required" ARRAY order is preserved
    # (["zeta","alpha"], not sorted) — only object keys sort.
    assert rendered == (
        '{"properties":{"alpha":{"type":"string"},"zeta":{"type":"number"}},'
        '"required":["zeta","alpha"],"type":"object"}'
    )


# --- injection ---------------------------------------------------------------


def test_injection_appends_to_existing_system_prompt() -> None:
    req = req_with_tools("You are a helpful assistant.")
    inject_tool_prompt(req)
    sys = req.messages[0].content
    assert isinstance(sys, TextContent)
    assert sys.text.startswith("You are a helpful assistant.")
    assert "<available_tools>" in sys.text
    assert "<name>calculator</name>" in sys.text
    assert req.messages[1].role == Role.USER


def test_injection_inserts_system_prompt_when_absent() -> None:
    req = req_with_tools(None)
    inject_tool_prompt(req)
    assert req.messages[0].role == Role.SYSTEM
    sys = req.messages[0].content
    assert isinstance(sys, TextContent)
    assert "<available_tools>" in sys.text


def test_injection_is_idempotent() -> None:
    req = req_with_tools("base")
    inject_tool_prompt(req)
    once = req.messages[0].content
    assert isinstance(once, TextContent)
    once_text = once.text
    inject_tool_prompt(req)
    twice = req.messages[0].content
    assert isinstance(twice, TextContent)
    assert once_text == twice.text


def test_injection_noop_without_tools() -> None:
    req = req_with_tools("base")
    req.tools = []
    before = [m.model_copy(deep=True) for m in req.messages]
    inject_tool_prompt(req)
    assert req.messages == before


# --- parsing -----------------------------------------------------------------


def test_parses_single_tool_call_marker() -> None:
    resp = prose(
        '<tool_call><name>calculator</name><input>{"expression": "2+2"}</input></tool_call>',
        StopReason.END_TURN,
    )
    out = parse_prose_response(resp)
    assert out.stop_reason == StopReason.TOOL_USE
    assert len(out.content) == 1
    block = out.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.name == "calculator"
    assert block.input == {"expression": "2+2"}
    assert block.id == "ptc_call_0"


def test_parses_multiple_tool_call_markers() -> None:
    text = (
        '<tool_call><name>a</name><input>{"x":1}</input></tool_call>\n'
        "some chatter\n"
        '<tool_call><name>b</name><input>{"y":2}</input></tool_call>'
    )
    out = parse_prose_response(prose(text, StopReason.END_TURN))
    names = [b.name for b in out.content if isinstance(b, ToolUseBlock)]
    assert names == ["a", "b"]
    assert out.stop_reason == StopReason.TOOL_USE
    ids = [b.id for b in out.content if isinstance(b, ToolUseBlock)]
    assert ids == ["ptc_call_0", "ptc_call_1"]


def test_malformed_input_json_falls_through_as_prose() -> None:
    resp = prose(
        "<tool_call><name>calculator</name><input>{not valid json}</input></tool_call>",
        StopReason.END_TURN,
    )
    out = parse_prose_response(resp)
    assert out.stop_reason == StopReason.END_TURN
    assert isinstance(out.content[0], TextBlock)


def test_plain_prose_returned_as_is() -> None:
    resp = prose("The answer is 4.", StopReason.END_TURN)
    out = parse_prose_response(resp)
    assert out == resp


def test_native_tool_use_left_untouched() -> None:
    resp = ModelResponse(
        content=[ToolUseBlock(id="native", name="calculator", input={"expression": "1"})],
        usage=usage(),
        stop_reason=StopReason.TOOL_USE,
    )
    out = parse_prose_response(resp)
    assert out == resp


def test_thinking_blocks_preserved_alongside_synthesized_calls() -> None:
    resp = ModelResponse(
        content=[
            ThinkingBlock(text="reasoning"),
            TextBlock(text="<tool_call><name>t</name><input>{}</input></tool_call>"),
        ],
        usage=usage(),
        stop_reason=StopReason.END_TURN,
    )
    out = parse_prose_response(resp)
    assert isinstance(out.content[0], ThinkingBlock)
    assert isinstance(out.content[1], ToolUseBlock)


# --- always-on wrapper -------------------------------------------------------


async def test_always_on_wrapper_injects_and_parses() -> None:
    m = MockModelInterface(provider())
    m.push_response(
        prose(
            '<tool_call><name>calculator</name><input>{"expression":"2+2"}</input></tool_call>',
            StopReason.END_TURN,
        )
    )
    wrapper = PromptBasedToolCallModelInterface(m)
    resp = await wrapper.call(req_with_tools("base"))
    assert resp.stop_reason == StopReason.TOOL_USE
    assert isinstance(resp.content[0], ToolUseBlock)


# --- adaptive wrapper --------------------------------------------------------


async def test_adaptive_wrapper_delegates_natively_when_flag_unset() -> None:
    m = MockModelInterface(provider())
    m.push_response(
        prose("<tool_call><name>x</name><input>{}</input></tool_call>", StopReason.END_TURN)
    )
    flag = PromptToolCallFlag(value=False)
    wrapper = AdaptiveToolCallModelInterface(m, flag)
    resp = await wrapper.call(req_with_tools("base"))
    assert resp.stop_reason == StopReason.END_TURN
    assert isinstance(resp.content[0], TextBlock)


async def test_adaptive_wrapper_parses_when_flag_set() -> None:
    m = MockModelInterface(provider())
    m.push_response(
        prose('<tool_call><name>x</name><input>{"k":1}</input></tool_call>', StopReason.END_TURN)
    )
    flag = PromptToolCallFlag(value=True)
    wrapper = AdaptiveToolCallModelInterface(m, flag)
    resp = await wrapper.call(req_with_tools("base"))
    assert resp.stop_reason == StopReason.TOOL_USE
    block = resp.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.name == "x"
    assert block.input == {"k": 1}


async def test_adaptive_wrapper_provider_delegates() -> None:
    m = MockModelInterface(provider())
    flag = PromptToolCallFlag(value=False)
    wrapper = AdaptiveToolCallModelInterface(m, flag)
    assert wrapper.provider().model_id == "test-1"


# --- prose detection ---------------------------------------------------------


def test_prose_detected_on_action_intent_with_tools() -> None:
    got = detect_prose_response("Sure, I'll use the calculator tool to add these.", True)
    assert got is not None


def test_prose_detected_case_insensitive() -> None:
    got = detect_prose_response("LET ME CALL the search tool now.", True)
    assert got is not None


def test_prose_not_detected_without_tools_advertised() -> None:
    assert detect_prose_response("I'll use the calculator.", False) is None


def test_prose_not_detected_for_plain_final_answer() -> None:
    assert detect_prose_response("The answer is 42.", True) is None


def test_prose_not_detected_for_empty_text() -> None:
    assert detect_prose_response("   ", True) is None


@pytest.mark.parametrize("_unused", [0])
def test_action_phrases_present(_unused: int) -> None:
    # Sanity: the curated list is non-empty and lowercase (mirrors Rust).
    from spore_core.prompt_tool_call import ACTION_PHRASES

    assert ACTION_PHRASES
    assert all(p == p.lower() for p in ACTION_PHRASES)
