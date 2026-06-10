"""spore-core example 07 — the storage seam, via a ``MarkdownMemoryProvider``.

What it demonstrates
====================
The harness is **stateless**: all durable state lives behind the storage seam.
Memory is just one domain of that seam — a ``MemoryStore`` you implement. The
simplest useful implementation is a human-readable markdown file. This example
ships :class:`MarkdownMemoryProvider` (in ``memory_provider.py``), composes it
into a :class:`~spore_core.StorageProvider` (NoOp for the other three domains),
and runs the SAME agent twice against it:

- ``--phase store``  — the agent is given facts about a fictional "Project
  Ironwood" and writes each as a memory via the built-in ``memory`` tool. The
  process exits leaving a readable ``memory.md`` on disk.
- ``--phase recall`` — a fresh process loads ``memory.md`` through the same
  provider and answers questions that restate NONE of the facts. The agent
  recalls them from memory via the ``memory`` tool.

The seam
========
``main`` never calls ``append_memory`` / ``get_memories`` directly. It hands the
composed provider to ``HarnessBuilder.storage(...)``; the harness threads
``storage.memory()`` into the built-in ``memory`` tool's context per run. The
agent drives all reads/writes from inside the ReAct loop. Swap the provider
(e.g. the built-in JSONL :class:`~spore_core.FileSystemStorageProvider`) and
nothing else changes — that is the point of the seam.

Pinned session id (critical)
============================
Memory is keyed by :class:`~spore_core.SessionId`; the ``memory`` tool always
uses the run's session id. Both phases therefore pin the SAME id —
``SessionId("project-ironwood")``, NOT ``new_session_id()``. With a generated id
Run 2 would key a different session and read nothing back.

Scope
=====
All facts use ``StorageScope.PROJECT`` (the ``memory`` tool rejects ``Local``).
The prompts instruct the agent to use ``scope: "project"`` consistently so the
recall read hits the same scope the store writes wrote.

Tool-calling mode: this example uses **native Ollama tool calling by default**
(the real typed tool schema), which works for tool-capable / cloud models like
``gemma4:31b-cloud``. Pass ``--structured`` to opt into
``ModelParams(structured_tool_calls=True)`` — schema-constrained decoding that
helps small local models (e.g. ``llama3.2``) emit one clean ``memory`` tool call
per turn. Structured mode exposes an always-available ``final`` envelope, so a
capable model may emit ``{"tool":"final"}`` prematurely and return an EMPTY
answer without ever calling ``memory``; if you see that (and no ``memory.md``),
drop ``--structured``.

Run it::

    ollama serve &
    ollama pull llama3.2
    uv run main.py --phase store     # writes memory.md (native tool calling)
    cat memory.md                    # inspect the human-readable artifact
    uv run main.py --phase recall    # answers from memory.md alone
    uv run main.py --phase store --structured   # constrained decoding for small models
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
    ModelParams,
    OllamaModelInterface,
    ReactConfig,
    RunResultSuccess,
    SessionId,
    StreamToolCall,
    StreamToolResult,
    StreamTurnStart,
    Task,
)
from spore_tools import StandardTools

from memory_provider import MarkdownMemoryProvider

# The pinned session id shared by BOTH phases. Memory is session-keyed, so store
# and recall MUST agree on this id or recall reads nothing. NOT new_session_id().
SESSION = SessionId("project-ironwood")

STORE_SYSTEM_PROMPT = (
    "You are a memory-keeping agent. You will be given a briefing of facts. For "
    'EACH distinct fact, call the `memory` tool with operation "write", scope '
    '"project", role "assistant", and the fact text as `content`. Write the facts '
    "verbatim and one at a time. Do not summarize or merge facts. When every fact "
    "has been written, reply with a short confirmation of how many you stored."
)

RECALL_SYSTEM_PROMPT = (
    "You are a recall agent. Everything you know about Project Ironwood lives in "
    "memory — nothing is in this prompt. FIRST call the `memory` tool with "
    'operation "read", scope "project" to load what you remember. THEN answer the '
    "user's questions using only the recalled memories. Cite the relevant "
    "remembered fact when you answer. Do not invent facts that are not in memory."
)

RECALL_QUESTIONS = (
    "Answer these about Project Ironwood, using only your memory:\n"
    "1. How many engineers are on the team, and who leads it?\n"
    "2. What database was chosen as the system of record, and why over the alternative?\n"
    "3. What are the two hard constraints?\n"
    "4. What is the known single point of failure?"
)


def _truncate(text: str, limit: int = 160) -> str:
    """Keep stream lines readable — memory reads return a JSON array of entries."""
    flat = text.replace("\n", " ")
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


async def _run_phase(
    model: OllamaModelInterface,
    memory_path: Path,
    system_prompt: str,
    task_prompt: str,
    structured: bool,
) -> str | None:
    """Build a harness over the markdown memory provider + the built-in ``memory``
    tool, pin the shared session id, run one task, and stream the loop. Returns
    the agent's output on success, ``None`` otherwise."""
    # Compose the real markdown MemoryStore with NoOp for the other three storage
    # domains. This is the entire integration: the harness threads
    # ``storage.memory()`` into the ``memory`` tool's context per run.
    storage = MarkdownMemoryProvider(memory_path).into_storage_provider()

    harness = (
        HarnessBuilder.conversational(model)
        .storage(storage)
        .tool(StandardTools.memory())
        .system_prompt(system_prompt)
        # Native tool calling by default; ``--structured`` opts into constrained
        # decoding for small local models. With structured mode the "think" line
        # is just a turn marker (one clean tool call per turn, no interleaved
        # reasoning), but a capable model can bail early via the always-available
        # ``final`` envelope — see the docstring.
        .model_params(ModelParams(structured_tool_calls=structured))
        .build()
    )

    # PIN the session id — both phases pass the same one so recall reads what
    # store wrote.
    task = Task.new(task_prompt, SESSION, ReactConfig.per_loop(20))

    def on_stream(event: object) -> None:
        if isinstance(event, StreamTurnStart):
            print(f"think  · turn {event.turn}")
        elif isinstance(event, StreamToolCall):
            print(f"    act    → {event.name}({_truncate(json.dumps(event.args))})")
        elif isinstance(event, StreamToolResult):
            tag = "obs(err)" if event.is_error else "obs "
            print(f"    {tag}→ {_truncate(event.content)}")

    options = HarnessRunOptions(task, on_stream=on_stream)
    result = await harness.run(options)
    if isinstance(result, RunResultSuccess):
        return result.output
    print(f"\nrun did not succeed: {result!r}", file=sys.stderr)
    return None


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core memory / storage-seam example")
    parser.add_argument("--model")
    parser.add_argument("--phase", choices=["store", "recall"])
    parser.add_argument(
        "--structured",
        action="store_true",
        help=(
            "Opt into schema-constrained (structured) tool calls for small local "
            "models. Default is native Ollama tool calling, which works for "
            "tool-capable / cloud models like gemma4:31b-cloud."
        ),
    )
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    here = Path(__file__).parent
    # ``memory.md`` lives next to this example's sources so the artifact is easy
    # to find and inspect between phases.
    memory_path = here / "memory.md"

    # Default (no --phase): run store, then point the user at recall and exit.
    default_phase = args.phase is None
    phase = args.phase or "store"

    model = OllamaModelInterface.with_base_url(model_id, base_url)

    print(f"model      : {model_id}")
    print(f"memory.md  : {memory_path}")
    print(f"session id : {SESSION}  (pinned — shared by both phases)")
    print(f"phase      : {phase}\n")

    if phase == "store":
        # Read the briefing and feed it to the agent. The agent writes each fact
        # via the ``memory`` tool; ``main`` never writes memory itself.
        briefing = (here / "project-ironwood.md").read_text(encoding="utf-8")
        task_prompt = (
            "Here is the Project Ironwood briefing. Store each fact to memory.\n\n" + briefing
        )
        try:
            output = await _run_phase(
                model, memory_path, STORE_SYSTEM_PROMPT, task_prompt, args.structured
            )
        except OSError as e:
            print(
                f"\ncould not reach the model — is `ollama serve` running? ({e})", file=sys.stderr
            )
            return 1
        if output is None:
            return 1
        print(f"\nstored. agent said: {output}")
        if memory_path.exists():
            print(f"\nmemory.md now exists on disk: {memory_path}")
            print("inspect it, then run:  uv run main.py --phase recall")
        if default_phase:
            print(
                "\n(no --phase given, so we ran `store`. Now run `uv run main.py --phase recall`.)"
            )
        return 0

    # phase == "recall"
    if not memory_path.exists():
        print(
            f"memory.md does not exist yet at {memory_path}.\n"
            "Run `uv run main.py --phase store` first.",
            file=sys.stderr,
        )
        return 2
    try:
        output = await _run_phase(
            model, memory_path, RECALL_SYSTEM_PROMPT, RECALL_QUESTIONS, args.structured
        )
    except OSError as e:
        print(f"\ncould not reach the model — is `ollama serve` running? ({e})", file=sys.stderr)
        return 1
    if output is None:
        return 1
    print(f"\nanswers from memory:\n{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
