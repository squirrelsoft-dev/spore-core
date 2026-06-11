"""spore-core example 05 — a custom tool you write yourself.

Examples :doc:`03-tool-use <../03-tool-use/main>` and
:doc:`04-filesystem-agent <../04-filesystem-agent/main>` showed the two *built-in*
tool paths: hand-rolling the harness-loop ``ToolRegistry`` (03) and registering
the shipped catalogue with ``.tools(StandardTools.coding_set())`` (04). This
example shows the third and most important path — **bringing your own tool** —
using the ergonomic :func:`~spore_tools.define_tool` helper.

The two custom tools
--------------------

Both live in :mod:`tools`, each defined with :func:`~spore_tools.define_tool`: a
typed pydantic input model plus an async ``execute`` body. The helper derives
the advertised JSON schema from the input model (so schema and validation can
never drift) and validates the model's arguments into it before calling
``execute``:

- **``remember(key, value)``** — persists a fact into the run store
  (:mod:`tools.remember`). It MUTATES shared state, so it is not ``read_only``.
- **``recall(key)``** — reads a fact back out (:mod:`tools.recall`). It only
  reads, so it is ``read_only`` + ``idempotent`` (passed via ``annotations``).

The pattern: ``define_tool(...)`` → ``.tool()``
-----------------------------------------------

1. Call ``define_tool(name, description, input_model, execute, ...)`` — it
   returns a :class:`~spore_tools.StandardTool` bundling a ``Tool`` impl with
   its derived schema. The ``input_model`` is the single source of truth: the
   advertised schema is generated from it, never hand-written.
2. Register each with ``.tool(...)``. The harness wires the sandbox and a per-run
   :class:`~spore_core.tool_registry.ToolContext` automatically — **the harness
   doesn't change, only what you register does.**

   If the model sends bad arguments, ``define_tool`` returns a *recoverable*
   ``invalid parameters`` error so tool-call repair can retry.

Two builder differences from 04: there is no ``.tools(...)`` catalogue, and no
explicit ``.sandbox(...)`` / ``.storage(...)``. ``build()`` defaults storage to
an in-memory provider whenever ``.tool()`` tools are present, so the run store
works for free.

Run it::

    ollama serve &
    ollama pull llama3.2
    uv run main.py
    uv run main.py --prompt "Research mycelium. Remember a few facts, then recall and summarize them."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from spore_core import (
    HarnessBuilder,
    HarnessRunOptions,
    OllamaModelInterface,
    ReactConfig,
    RunResultSuccess,
    StreamToolCall,
    StreamToolResult,
    StreamTurnStart,
    Task,
    new_session_id,
)
from tools import recall_tool, remember_tool

SYSTEM_PROMPT = (
    "You are a research agent with a memory. Research the topic the user gives "
    "you across several turns. As you discover each fact, call `remember` to "
    "store it under a short, stable key (e.g. 'habitat', 'diet'). Keep track of "
    "the keys you use. When you have gathered enough facts, call `recall` on each "
    "key you remembered, then write a final summary built ONLY from the recalled "
    "facts. Act using tools — do not just describe."
)


def _truncate(text: str, limit: int = 200) -> str:
    """Keep observe lines readable — recalled facts can be long."""
    flat = text.replace("\n", " ")
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core custom-tool research agent")
    parser.add_argument("--model")
    parser.add_argument("--prompt")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    prompt = args.prompt or (
        "Research the common octopus. Remember a few key facts (habitat, diet, "
        "lifespan, intelligence), then recall them and write a short summary."
    )

    # Same ``conversational`` harness as 03 / 04 — the substantive change is that
    # we register two tools WE wrote (``.tool(...)``) instead of a catalogue
    # preset. No ``.sandbox(...)`` (these tools ignore it) and no ``.storage(...)``
    # (``build()`` defaults to in-memory storage when ``.tool()`` tools present).
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    harness = (
        HarnessBuilder.conversational(model)
        .tool(remember_tool())
        .tool(recall_tool())
        .system_prompt(SYSTEM_PROMPT)
        .build()
    )

    task = Task.new(prompt, new_session_id(), ReactConfig.per_loop(12))

    # Print each turn (Think) and each tool call + result (Act / Observe) from
    # harness STREAM events — the builder dispatches our tools internally, just as
    # it does the catalogue in 04.
    def on_stream(event: object) -> None:
        if isinstance(event, StreamTurnStart):
            print(f"think  · turn {event.turn}")
        elif isinstance(event, StreamToolCall):
            print(f"    act    → {event.name}({json.dumps(event.args)})")
        elif isinstance(event, StreamToolResult):
            tag = "obs(err)" if event.is_error else "obs "
            print(f"    {tag}→ {_truncate(event.content)}")

    options = HarnessRunOptions(task, on_stream=on_stream)

    print(f"model  : {model_id}")
    print("tools  : remember, recall")
    print(f"prompt : {prompt}\n")

    try:
        result = await harness.run(options)
    except OSError as e:
        # Ollama unreachable / endpoint refused the connection, etc.
        print(f"\ncould not reach the model — is `ollama serve` running? ({e})", file=sys.stderr)
        return 1

    if isinstance(result, RunResultSuccess):
        print(f"\nsummary ({result.turns} turn(s)): {result.output}")
        return 0

    print(f"\nrun did not succeed: {result!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
