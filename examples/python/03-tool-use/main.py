"""spore-core example 03 — ReAct with local tools.

The agent now *acts*: it thinks, calls a tool, observes the result, and loops
until it can answer. The tools here are deliberately trivial — ``calculator``,
``get_current_time``, ``reverse_string`` — because the star of this example is
the **Think -> Act -> Observe** loop, not the tools.

What it shows
-------------

- Implementing the harness-loop ``ToolRegistry`` directly: ``schemas()``
  advertises the tools to the model, ``dispatch()`` runs them. No filesystem, no
  sandbox needed — these are pure functions, so the
  :meth:`HarnessBuilder.conversational` defaults (incl. the null sandbox) are
  fine; we only override the tool registry.
- The loop itself: the program prints each turn (Think) via the stream sink and
  each tool call + result (Act / Observe), so you can watch the agent work.

Run it::

    ollama serve &
    ollama pull llama3.2
    uv run main.py
    uv run main.py --prompt "reverse the word 'mycelium' and multiply 6 by 7"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from spore_core import (
    HarnessBuilder,
    HarnessRunOptions,
    OllamaModelInterface,
    ReactConfig,
    RunResultSuccess,
    StreamTurnStart,
    Task,
    ToolCall,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
    ToolSchema,
    new_session_id,
)


def _schema(name: str, description: str, properties: dict, required: list[str]) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


class LocalTools:
    """Three trivial, pure-compute tools, exposed through the harness-loop tool
    registry. ``schemas()`` is what the model sees; ``dispatch()`` is what runs.
    """

    def schemas(self) -> list[ToolSchema]:
        return [
            _schema(
                "calculator",
                "Compute a binary arithmetic operation. 'op' is one of + - * /.",
                {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "op": {"type": "string", "enum": ["+", "-", "*", "/"]},
                },
                ["a", "b", "op"],
            ),
            _schema(
                "get_current_time",
                "Return the current time of day as HH:MM:SS UTC. Takes no arguments.",
                {},
                [],
            ),
            _schema(
                "reverse_string",
                "Reverse the characters in a string.",
                {"text": {"type": "string"}},
                ["text"],
            ),
        ]

    def is_always_halt(self, tool_name: str) -> bool:
        return False

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        try:
            if call.name == "calculator":
                content = _calculator(call.input)
            elif call.name == "get_current_time":
                content = _current_time()
            elif call.name == "reverse_string":
                content = _reverse_string(call.input)
            else:
                raise ValueError(f"unknown tool: {call.name}")
        except ValueError as e:
            # Print the failed Act step so the loop is visible.
            print(f"    act    → {call.name}({json.dumps(call.input)}) failed: {e}")
            return ToolOutputError.error(str(e))
        # Print the Act + Observe step so the loop is visible.
        print(f"    act    → {call.name}({json.dumps(call.input)}) = {content}")
        return ToolOutputSuccess.success(content)


def _number(input: dict, key: str) -> float:
    if key not in input:
        raise ValueError(f"missing number '{key}'")
    value = input[key]
    # Models often pass numbers as JSON strings ("144"); accept either.
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise ValueError(f"'{key}' is not a number: {value}")


def _calculator(input: dict) -> str:
    a = _number(input, "a")
    b = _number(input, "b")
    op = input.get("op")
    if not isinstance(op, str):
        raise ValueError("missing string 'op'")
    if op == "+":
        value = a + b
    elif op == "-":
        value = a - b
    elif op == "*":
        value = a * b
    elif op == "/":
        if b == 0.0:
            raise ValueError("division by zero")
        value = a / b
    else:
        raise ValueError(f"unknown op '{op}' (use + - * /)")
    # Render integers without a trailing ".0" to match the Rust output shape.
    if value == int(value):
        return str(int(value))
    return str(value)


def _current_time() -> str:
    secs = int(time.time())
    return f"{(secs // 3600) % 24:02d}:{(secs // 60) % 60:02d}:{secs % 60:02d} UTC"


def _reverse_string(input: dict) -> str:
    text = input.get("text")
    if not isinstance(text, str):
        raise ValueError("missing string 'text'")
    return text[::-1]


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core ReAct tool use")
    parser.add_argument("--model")
    parser.add_argument("--prompt")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)
    prompt = args.prompt or (
        "Use your tools to answer: what is 144 divided by 12, what is the current "
        "time, and what is 'harness' reversed?"
    )

    # Same conversational harness as 01/02 — we only swap in our tool registry.
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    harness = HarnessBuilder.conversational(model).tool_registry(LocalTools()).build()

    task = Task.new(prompt, new_session_id(), ReactConfig.per_loop(6))

    # Print each turn so the "Think" steps are visible alongside the tool calls.
    def on_stream(event: object) -> None:
        if isinstance(event, StreamTurnStart):
            print(f"think  · turn {event.turn}")

    options = HarnessRunOptions(task, on_stream=on_stream)

    print(f"model  : {model_id}")
    print(f"prompt : {prompt}\n")
    result = await harness.run(options)
    if isinstance(result, RunResultSuccess):
        print(f"\nanswer ({result.turns} turn(s)): {result.output}")
        return 0
    print(f"\nrun did not succeed: {result!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
