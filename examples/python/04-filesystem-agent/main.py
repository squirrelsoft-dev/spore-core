"""spore-core example 04 — ReAct with the built-in catalogue file tools.

This is :doc:`03-tool-use <../03-tool-use/main>` with one substantive change. In
03 the agent's tools were hand-rolled: we implemented the harness-loop
``ToolRegistry`` ourselves (``schemas`` + ``dispatch``) and ran each call by
hand. Here we register spore-core's *built-in* catalogue instead — a single
builder line::

    .tools(StandardTools.coding_set())   # read_file, write_file, list_dir, …

Everything else is the same: the same :meth:`HarnessBuilder.conversational`
builder, the same ``ReAct`` loop, the same stream-printed ``think · turn N`` /
tool-call output. The thesis of this example is exactly that: **the harness
doesn't change — only the registration path does.**

What it shows
-------------

- **Catalogue registration.** ``.tools(StandardTools.coding_set())`` advertises
  and dispatches ``read_file`` / ``write_file`` / ``list_dir`` (and friends)
  with no bespoke code.
- **A real sandbox.** Catalogue file tools operate *through* a sandbox, so
  unlike 03's pure-compute tools (which were happy with the
  ``conversational`` default null sandbox) this example wires a
  :class:`WorkspaceScopedSandbox` scoped to ``sample-files/``.
- **A side effect that outlives the process.** The agent writes ``SUMMARY.md``
  into ``sample-files/``. It is still there after the program exits — the first
  example that leaves something behind on disk.

Run it::

    ollama serve &
    ollama pull llama3.2
    uv run main.py
    uv run main.py --prompt "List the files and tell me which one mentions nutrients."
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
    "You are a file-summarizing agent. Use list_dir to find files, "
    "read_file to read each, and write_file to create SUMMARY.md. "
    "Act using tools — do not just describe."
)


def _truncate(text: str, limit: int = 200) -> str:
    """Keep observe lines readable — file contents can be long."""
    flat = text.replace("\n", " ")
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core filesystem agent")
    parser.add_argument("--model")
    parser.add_argument("--prompt")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    # The agent operates inside the shipped ``sample-files/`` directory. Resolve
    # it relative to this source file so ``uv run main.py`` works from anywhere,
    # and canonicalize it — the sandbox requires a canonical, existing root.
    workspace_root = (Path(__file__).parent / "sample-files").resolve(strict=True)

    prompt = args.prompt or (
        "There are several .txt files in this directory. Use list_dir to find them, "
        "read_file to read each one, then write a SUMMARY.md containing a one-sentence "
        "summary of every file. Use write_file to create it."
    )

    # Same ``conversational`` harness as 03 — the ONLY substantive change is that
    # we register the built-in catalogue (``.tools(...)``) over a real sandbox
    # instead of hand-rolling a ``ToolRegistry``.
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    sandbox = WorkspaceScopedSandbox(WorkspaceConfig(root=workspace_root))
    harness = (
        HarnessBuilder.conversational(model)
        .sandbox(sandbox)
        .tools(StandardTools.coding_set())
        .system_prompt(SYSTEM_PROMPT)
        .build()
    )

    task = Task.new(prompt, new_session_id(), LoopStrategyReAct(max_iterations=8))

    # Print each turn (Think) and each catalogue tool call + result (Act /
    # Observe). Because the catalogue dispatches internally, the Act/Observe
    # lines come from harness STREAM events, not from inside a hand-rolled
    # dispatch like 03.
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
    print(f"dir    : {workspace_root}")
    print(f"prompt : {prompt}\n")

    try:
        result = await harness.run(options)
    except OSError as e:
        # Ollama unreachable / endpoint refused the connection, etc.
        print(f"\ncould not reach the model — is `ollama serve` running? ({e})", file=sys.stderr)
        return 1

    if isinstance(result, RunResultSuccess):
        print(f"\nanswer ({result.turns} turn(s)): {result.output}")
        summary = workspace_root / "SUMMARY.md"
        if summary.exists():
            print(f"\nSUMMARY.md now exists on disk: {summary}")
        return 0

    print(f"\nrun did not succeed: {result!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
