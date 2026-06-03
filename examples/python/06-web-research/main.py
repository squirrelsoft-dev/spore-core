"""spore-core example 06 — web research with an external API tool.

This is the first example whose tools reach **outside the process** to a
third-party HTTP service. The whole point is that this changes *nothing* about
the harness: an external API is just another tool.

Tools wired (all from the built-in catalogue, no custom tool class):

- ``web_search`` — :meth:`StandardTools.web_search_with_endpoint`. The query is
  POSTed to ``endpoint`` as JSON ``{"query": ...}`` and the response body is
  returned to the agent verbatim. The endpoint comes from
  ``SPORE_WEB_SEARCH_ENDPOINT`` (see the README + ``.env.example``).
- ``write_file`` — :meth:`StandardTools.write_file`. The agent writes its
  synthesized, cited answer to ``answer.md``.
- ``read_file`` — :meth:`StandardTools.read_file`. Lets the agent re-read what
  it wrote (e.g. to verify or revise the answer).

Harness + sandbox pattern reused verbatim from
:doc:`04-filesystem-agent <../04-filesystem-agent/main>`:

- ``HarnessBuilder.conversational(model)`` — same builder.
- ``LoopStrategyReAct(max_iterations=...)`` — same loop.
- ``WorkspaceScopedSandbox`` over ``WorkspaceConfig(root=...)`` — same sandbox,
  here scoped to this example's ``workspace/`` dir so ``write_file`` cannot
  escape it. 04 wrote ``SUMMARY.md``; 06 writes ``answer.md``.

The ONLY substantive difference from 04 is the tool set: 04 registers
``coding_set()``, 06 registers ``web_search_with_endpoint(..)`` + ``write_file``
+ ``read_file``. Same harness, different tools.

Run it::

    ollama serve &
    ollama pull llama3.2
    export SPORE_WEB_SEARCH_ENDPOINT=http://localhost:8888/search  # a {"query"}->JSON endpoint
    uv run main.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from spore_core import (
    HarnessBuilder,
    HarnessRunOptions,
    LoopStrategyReAct,
    OllamaModelInterface,
    RunResultSuccess,
    StreamToolCall,
    StreamToolResult,
    StreamTurnStart,
    Task,
    WorkspaceConfig,
    WorkspaceScopedSandbox,
    new_session_id,
)
from spore_tools import StandardTools

SYSTEM_PROMPT = (
    "You are a web-research agent. Use web_search to find current information, "
    "synthesize what you learn into a clear answer, and ALWAYS cite the sources "
    "you used. Write the final answer to answer.md using write_file. Act using "
    "tools — do not answer from memory alone."
)


def _truncate(text: str, limit: int = 200) -> str:
    """Keep observe lines readable — search results can be long."""
    flat = text.replace("\n", " ")
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core web-research agent")
    parser.add_argument("--model")
    parser.add_argument("--prompt")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    # The search backend endpoint. ``web_search`` POSTs ``{"query": ...}`` here
    # and returns the JSON body to the agent. There is no live backend in
    # spore-core, so you must supply one — a self-hosted SearXNG JSON endpoint,
    # or a mock that accepts the ``{"query"}`` shape. Raw Brave/Tavily are NOT
    # yet drop-in: they need a custom auth header, tracked as core issue #108.
    endpoint = (os.environ.get("SPORE_WEB_SEARCH_ENDPOINT") or "").strip()
    if not endpoint:
        print(
            "SPORE_WEB_SEARCH_ENDPOINT is not set.\n"
            'Set it to a search endpoint that accepts a JSON `{"query": ...}` POST '
            "and returns JSON results.\n"
            "See .env.example and the README. (Raw Brave/Tavily need core #108 first.)",
            file=sys.stderr,
        )
        return 2

    # The agent operates inside this example's ``workspace/`` directory. Resolve
    # it relative to this source file so ``uv run main.py`` works from anywhere,
    # and canonicalize it — the sandbox requires a canonical, existing root.
    workspace_root = Path(__file__).parent / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace_root = workspace_root.resolve(strict=True)

    prompt = args.prompt or (
        # A TIMELESS research question: the answer evolves over time but the
        # question stays interesting and is not tied to a single news event.
        "What is the current recommended way to install Rust on macOS, and what are the "
        "main alternatives? Search the web, synthesize the options, cite your sources, "
        "and write the answer to answer.md."
    )

    # Same ``conversational`` harness + ``WorkspaceScopedSandbox`` as 04. The
    # ONLY substantive change is the tool set: ``web_search`` (external API)
    # composes with ``write_file`` / ``read_file`` in one builder chain.
    # ``.tool()`` and ``.tools()`` push into the same registry with last-wins
    # upsert by name.
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    sandbox = WorkspaceScopedSandbox(WorkspaceConfig(root=workspace_root))
    harness = (
        HarnessBuilder.conversational(model)
        .sandbox(sandbox)
        .tool(StandardTools.web_search_with_endpoint(endpoint))
        .tool(StandardTools.write_file())
        .tool(StandardTools.read_file())
        .system_prompt(SYSTEM_PROMPT)
        .build()
    )

    task = Task.new(prompt, new_session_id(), LoopStrategyReAct(max_iterations=10))

    # Print each turn (Think) and each tool call + result (Act / Observe). The
    # search queries and result snippets show up here because ``web_search``
    # dispatches through the harness like any other catalogue tool.
    def on_stream(event: object) -> None:
        if isinstance(event, StreamTurnStart):
            print(f"think  · turn {event.turn}")
        elif isinstance(event, StreamToolCall):
            print(f"    act    → {event.name}({json.dumps(event.args)})")
        elif isinstance(event, StreamToolResult):
            tag = "obs(err)" if event.is_error else "obs "
            print(f"    {tag}→ {_truncate(event.content)}")

    options = HarnessRunOptions(task, on_stream=on_stream)

    print(f"model    : {model_id}")
    print(f"endpoint : {endpoint}")
    print(f"workspace: {workspace_root}")
    print(f"prompt   : {prompt}\n")

    try:
        result = await harness.run(options)
    except OSError as e:
        # Ollama unreachable / endpoint refused the connection, etc.
        print(f"\ncould not reach the model — is `ollama serve` running? ({e})", file=sys.stderr)
        return 1

    if isinstance(result, RunResultSuccess):
        print(f"\nanswer ({result.turns} turn(s)): {result.output}")
        answer = workspace_root / "answer.md"
        if answer.exists():
            print(f"\nanswer.md now exists on disk: {answer}")
        return 0

    print(f"\nrun did not succeed: {result!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
