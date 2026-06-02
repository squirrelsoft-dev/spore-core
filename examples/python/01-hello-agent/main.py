"""spore-core example 01 — hello agent.

The smallest real thing you can build with spore-core: turn a model into a
running agent and ask it to say hello. No tools, no filesystem, no multi-turn
state.

``HarnessBuilder.conversational(model)`` defaults every required component (a
model-backed agent, an empty tool registry, a null sandbox, a standard
context manager, and respond-and-stop termination), so the whole thing is a
handful of lines. Later examples override individual defaults — add tools, swap
the sandbox, change the loop strategy — via the builder setters.

Run it::

    ollama serve &            # start a local model server
    ollama pull llama3.2      # pull the default model
    uv run main.py            # or: uv run main.py --model <id>

``SPORE_OLLAMA_MODEL`` / ``SPORE_OLLAMA_BASE_URL`` override the model id and the
Ollama endpoint (default ``http://localhost:11434``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from spore_core import (
    HarnessBuilder,
    HarnessRunOptions,
    OllamaModelInterface,
    RunResultSuccess,
    Task,
)


async def main() -> int:
    # Model id + endpoint come from args/env so you can swap models without an
    # edit.
    parser = argparse.ArgumentParser(description="spore-core hello agent")
    parser.add_argument("--model")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    # A model, a harness, a task — that's the whole setup.
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    harness = HarnessBuilder.conversational(model).build()
    task = Task.simple("Reply with a friendly one-line greeting.")

    print(f"model      : {model_id}")
    result = await harness.run(HarnessRunOptions(task))
    if isinstance(result, RunResultSuccess):
        print(f"result     : Success ({result.turns} turn(s))")
        print(f"greeting   : {result.output}")
        return 0
    print(f"result     : {result!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
