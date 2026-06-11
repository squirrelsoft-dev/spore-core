"""spore-core example 02 — conversational REPL.

Takes example 01 one step further: an interactive chat loop where the agent
remembers what you said earlier in the session. Same ``conversational(model)``
harness as 01 — the new idea here is *conversation continuity across runs*.

How memory works
----------------

The harness is stateless between ``run()`` calls: each call takes an optional
starting :class:`SessionState` (the message history) and drives one task to a
final response. As of issue #102, ``RunResultSuccess`` now hands the post-run
:class:`SessionState` back, so the caller resumes the conversation LOSSLESSLY —
no reconstruction. After each turn we feed the returned ``session_state``
straight into the next run via ``HarnessRunOptions(..., session_state=...)``.
The harness appends the new user line on top of that history before calling the
model, so the model sees the whole conversation and can refer back to it.

This works for tool-using agents too: the returned ``session_state`` carries the
tool-call and tool-result messages the loop produced, which the old "reconstruct
history from ``output``" trick could not recover.

Prefer it hands-free? Wire a ``SessionStore`` and call
``HarnessBuilder.auto_persist_sessions(True)``: the harness then auto-loads and
auto-persists by ``session_id``, so you reuse the id instead of threading state
at all (great for a web service that resumes across restarts).

Run it::

    ollama serve &
    ollama pull llama3.2
    uv run main.py                 # then chat; /exit or Ctrl-D to quit
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
    ReactConfig,
    RunResultSuccess,
    SessionState,
    Task,
    new_session_id,
)


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core conversational REPL")
    parser.add_argument("--model")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    # Build the harness once; reuse it for every turn.
    model = OllamaModelInterface.with_base_url(model_id, base_url)
    harness = HarnessBuilder.conversational(model).build()

    # One session id for the whole REPL, and the conversation state we thread
    # back in on each turn. Each run hands the post-run ``SessionState`` back
    # (issue #102), so we just carry it forward — lossless, no reconstruction.
    session_id = new_session_id()
    state = SessionState()
    turns_exchanged = 0

    print(f"conversational REPL — model {model_id}. Type a message; /exit or Ctrl-D to quit.")
    loop = asyncio.get_running_loop()
    while True:
        print("you> ", end="", flush=True)
        # ``input()`` blocks; run it off the event loop so the harness's async
        # I/O is never starved.
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line == "":  # EOF (Ctrl-D)
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break

        # Thread the running state into this turn. The harness appends ``line``
        # as the new user message before calling the model.
        task = Task.new(line, session_id, ReactConfig.per_loop(4))
        options = HarnessRunOptions(task, session_state=state)

        result = await harness.run(options)
        if isinstance(result, RunResultSuccess):
            print(f"bot> {result.output}")
            # Carry the post-run state forward losslessly (issue #102): it
            # already contains this turn's user + assistant messages (and any
            # tool messages a tool-using agent would produce).
            state = result.session_state
            turns_exchanged += 1
        else:
            print(f"bot> [run did not succeed: {result!r}]", file=sys.stderr)

    print(f"bye ({turns_exchanged} turn(s); {len(state.messages)} message(s) in history)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
