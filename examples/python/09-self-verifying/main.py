"""spore-core example 09 — the ``SelfVerifying`` loop strategy.

What this example demonstrates
------------------------------

**Quality loops are a harness concern, not application logic.** The agent
drafts a Python function, a *fresh evaluator run* critiques that draft against
an explicit spec, and a :class:`Verifier` turns the critique into a verdict. If
the verdict is FAIL, the reason is injected back into the build context and the
loop revises. This repeats until the verifier returns ``Passed`` or
``max_iterations`` is exhausted. You write **no loop code** — you wire a strategy
(:class:`LoopStrategySelfVerifying`), a :class:`Verifier`, and an evaluator
agent, and the harness runs the verify -> revise cycle for you.

The task the agent-under-test must solve
----------------------------------------

Write a Python function ``parse_int_list(text: str) -> list[int]`` that parses a
comma-separated list of integers and raises a custom ``ParseIntListError`` on a
bad token. The evaluator checks **five** criteria, each explicitly:

1. SIGNATURE: ``def parse_int_list(text: str) -> list[int]`` with a custom
   ``ParseIntListError(Exception)`` defined in the same file.
2. EDGE CASES: empty/whitespace-only input returns ``[]``; whitespace around each
   number is tolerated (``" 1, 2 ,3 "`` -> ``[1, 2, 3]``); a non-integer token
   raises ``ParseIntListError`` and NEVER raises a bare ``ValueError`` or crashes.
3. DOCSTRING: the function has a docstring describing what it does.
4. NO UNGUARDED CRASHES: the int parse is wrapped so a bad token becomes the typed
   error, not an unhandled exception.
5. USAGE EXAMPLE: at least one usage example in the docstring (e.g. a ``>>>``
   doctest line showing a call and its result).

The spec lives in **one** place — the ``Task`` instruction — so the build agent
and the evaluator see the exact same five criteria.

How the draft reaches the evaluator — and why we need a file tool
-----------------------------------------------------------------

Reading the strategy source (``_run_self_verifying`` / ``_run_evaluate_phase`` in
``harness.py``) settles the tool question. The evaluate phase builds a **fresh**
evaluator run whose context is seeded ONLY with a review directive containing the
``task.instruction`` plus a read-only sandbox. The build agent's draft text is
**not** auto-injected into the evaluator's context. So for the evaluator to
actually read the draft, the draft must live on disk where the (read-only)
evaluator can read it.

Therefore this example wires exactly the minimal file tool set:

- ``write_file`` — the **build** agent saves its draft to
  ``workspace/parse_int_list.py``.
- ``read_file``  — the **evaluator** reads that file back (its ``write_file`` is
  blocked by the internally-derived ``ReadOnlySandbox``).

No ``web_search``, no shell, nothing else. The loop is the point.

The observability seam — ``ReportingVerifier``
----------------------------------------------

Sub-loop streaming is suppressed by design (the build and evaluate sub-runs run
with a ``None`` sink, exactly like PlanExecute). The ONE reliable seam to watch
the verify -> revise cycle is the :class:`Verifier` itself: the harness calls
``verify(input)`` once per iteration, and :class:`VerifierInput` carries the
**draft** (``build_result`` output), the **critique** (``eval_result`` output),
and the 0-indexed ``iteration``. So we wrap :class:`EvaluatorResponseVerifier` in
a small :class:`ReportingVerifier` that prints, each iteration: a 1-based header
with the configured max, the draft, the critique, and the verdict — then
delegates the actual pass/fail decision to the inner verifier.

``EvaluatorResponseVerifier`` matches the evaluator's text against a ``PASS``
pattern and a ``FAIL: <reason>`` pattern; if NEITHER matches it returns FAIL by
contract (default-to-FAIL is baked into the verifier and reinforced by the
harness's evaluator directive — "you did NOT write this code; default to FAIL
unless you can confirm it is right").

Run it::

    ollama serve &
    ollama pull llama3.2
    uv run main.py                       # default model llama3.2, 3 iterations
    uv run main.py --max-iterations 5
    uv run main.py --model qwen2.5-coder:7b

See the README for the honest rough-edges section: SelfVerifying against a small
local model is genuinely flaky (the evaluator may mis-judge, the loop may exhaust
without passing). A larger hosted model gives a cleaner demo.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from spore_core import (
    AgentId,
    BudgetLimits,
    EvaluatorResponseVerifier,
    HaltReasonSelfVerifyExhausted,
    HarnessBuilder,
    HarnessRunOptions,
    LoopStrategySelfVerifying,
    ModelAgent,
    OllamaModelInterface,
    RunResult,
    RunResultFailure,
    RunResultSuccess,
    Task,
    VerifierInput,
    VerifierVerdict,
    VerifierVerdictPassed,
    WorkspaceConfig,
    WorkspaceScopedSandbox,
    new_session_id,
)
from spore_tools import StandardTools

# The spec the agent must satisfy. It is the ``Task`` instruction, so the
# **build** agent sees it directly, and — because the evaluate phase embeds the
# ``task.instruction`` in the evaluator's directive — the **evaluator** sees the
# exact same five criteria. One source of truth for both roles.
TASK_PROMPT = (
    "Write a Python function named `parse_int_list` and save it to the file "
    "`parse_int_list.py` using the write_file tool. It must satisfy ALL of the "
    "following, which you will be graded on criterion-by-criterion:\n"
    "\n"
    "1. SIGNATURE: `def parse_int_list(text: str) -> list[int]` where you also "
    "define a custom exception `class ParseIntListError(Exception)` in the same "
    "file.\n"
    "2. EDGE CASES: empty or whitespace-only input returns `[]`; whitespace "
    'around each number is tolerated (e.g. " 1, 2 ,3 " parses to [1, 2, 3]); a '
    "non-integer token raises `ParseIntListError` (NOT a bare ValueError) and "
    "never crashes with an unhandled exception.\n"
    "3. DOCSTRING: the function has a docstring describing what it does.\n"
    "4. NO UNGUARDED CRASHES: wrap the int() parse so a bad token becomes a "
    "raised `ParseIntListError`, not an unhandled exception.\n"
    "5. USAGE EXAMPLE: include at least one usage example in the docstring — for "
    "instance a `>>>` doctest line showing a call and its result.\n"
    "\n"
    "Write ONLY the file contents (valid Python). Save it with write_file, then "
    "report that you are done."
)

# System prompt shared by the build agent and the evaluator agent (the harness
# ``system_prompt`` is shared across both phases). It is deliberately
# role-neutral: the build/evaluate framing is supplied per-phase (the build agent
# gets the spec as its task; the evaluator gets the harness's built-in review
# directive plus the same spec). It reinforces the file-tool contract and the
# evaluator's default-to-FAIL posture.
SYSTEM_PROMPT = (
    "You work on Python code. Your only tools are write_file (save a file to the "
    "workspace) and read_file (read a file back). You have no shell and cannot run "
    "or import code.\n"
    "\n"
    "When ASKED TO WRITE code: write the file with write_file, then say you are "
    "done.\n"
    "\n"
    "When ASKED TO REVIEW code: first read_file the file under review. Then check "
    "the work against EACH numbered criterion in the task, one at a time. You did "
    "NOT write this code — default to FAIL unless you can positively confirm every "
    "criterion holds. Respond with EXACTLY ONE verdict line as the LAST line of "
    "your reply:\n"
    "  - `PASS` if and only if every criterion holds, or\n"
    "  - `FAIL: <which criteria failed and why>` otherwise.\n"
    "Never emit PASS when unsure."
)


def _run_result_output(result: RunResult) -> str:
    """Reduce a :class:`RunResult` to printable text: the ``Success`` output, or a
    short description of why the run did not complete."""
    if isinstance(result, RunResultSuccess):
        return result.output
    if isinstance(result, RunResultFailure):
        return f"<run did not complete: {result.reason!r}>"
    return f"<run did not complete: {result!r}>"


class ReportingVerifier:
    """A :class:`Verifier` decorator: prints the verify -> revise cycle to stdout,
    then delegates the actual verdict to an inner verifier.

    This is the one reliable observability seam for SelfVerifying — the build and
    evaluate sub-runs are streamed with a suppressed sink, so the verifier call is
    where the draft + critique + verdict become visible. Per iteration it prints:
    a 1-based header with the configured max, the **draft** (``build_result``
    output), the **critique** (``eval_result`` output), and the **verdict**. It
    satisfies the :class:`Verifier` Protocol structurally — no inheritance.
    """

    def __init__(self, inner: object, max_iterations: int) -> None:
        self._inner = inner
        self._max_iterations = max_iterations

    async def verify(self, input: VerifierInput) -> VerifierVerdict:
        # ``iteration`` is 0-indexed on the wire; display it 1-based.
        n = input.iteration + 1
        print(f"\n══════════════ iteration {n}/{self._max_iterations} ══════════════")

        print("\n── draft (what the agent wrote) ──")
        print(_run_result_output(input.build_result))

        print("\n── evaluation (the critique) ──")
        print(_run_result_output(input.eval_result))

        # Delegate the actual decision to the inner verifier.
        verdict: VerifierVerdict = await self._inner.verify(input)  # type: ignore[attr-defined]

        print("\n── verdict ──")
        if isinstance(verdict, VerifierVerdictPassed):
            print("PASS — criteria satisfied; loop halts.")
        else:
            print(f"FAIL — {verdict.reason}")
            if n < self._max_iterations:
                print("(reason injected into next build turn; revising…)")
            else:
                print("(no iterations left; loop will exhaust)")
        print("════════════════════════════════════════════════")
        return verdict

    def max_iterations(self) -> int:
        return self._max_iterations


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core self-verifying agent")
    parser.add_argument("--model")
    parser.add_argument("--prompt")
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Verify -> revise cap (default 3, or SPORE_MAX_ITERATIONS).",
    )
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    # Max iterations: CLI flag wins, then env var, then default 3. A non-positive
    # value falls back to the default.
    max_iterations = args.max_iterations
    if max_iterations is None:
        env_iters = os.environ.get("SPORE_MAX_ITERATIONS")
        max_iterations = int(env_iters) if env_iters and env_iters.isdigit() else 3
    if max_iterations <= 0:
        max_iterations = 3

    prompt = args.prompt or TASK_PROMPT

    # The agents operate inside this example's ``workspace/`` directory. Resolve it
    # relative to this source file so ``uv run main.py`` works from anywhere, and
    # canonicalize it — the sandbox requires a canonical, existing root.
    workspace_root = Path(__file__).parent / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace_root = workspace_root.resolve(strict=True)

    # The evaluator runs on its own agent instance (the ``evaluator_agent`` seam).
    # It shares the harness system prompt and tool set; the harness derives a
    # read-only sandbox for it internally, so its ``write_file`` is blocked but
    # ``read_file`` works — exactly what a reviewer needs.
    evaluator_model = OllamaModelInterface.with_base_url(model_id, base_url)
    evaluator_agent = ModelAgent(AgentId("evaluator"), evaluator_model)

    # The verifier: pattern-match the evaluator's text. ``PASS`` (anchored,
    # case-insensitive, multiline) -> Passed; ``FAIL: <reason>`` -> Failed(reason);
    # neither -> Failed by contract (default-to-FAIL). Wrapped in
    # ``ReportingVerifier`` so the cycle is visible. The harness reads
    # ``max_iterations()`` off the OUTER verifier, so keep both equal.
    inner = EvaluatorResponseVerifier(
        pass_pattern=r"(?im)^\s*PASS\s*$",
        fail_pattern=r"(?im)FAIL:\s*.+",
        max_iterations=max_iterations,
    )
    verifier = ReportingVerifier(inner, max_iterations)

    # Build harness: conversational preset, workspace sandbox, the minimal file
    # tool set (write_file for the builder + read_file for the evaluator), shared
    # system prompt, the evaluator agent, and the verifier.
    build_model = OllamaModelInterface.with_base_url(model_id, base_url)
    sandbox = WorkspaceScopedSandbox(WorkspaceConfig(root=workspace_root))
    harness = (
        HarnessBuilder.conversational(build_model)
        .sandbox(sandbox)
        .tool(StandardTools.write_file())
        .tool(StandardTools.read_file())
        .system_prompt(SYSTEM_PROMPT)
        .evaluator_agent(evaluator_agent)
        .verifier(verifier)
        .build()
    )

    # THE STRATEGY. There is no loop code below — the harness runs the
    # verify -> revise cycle. A generous turn budget per build/evaluate sub-run
    # lets a small model take a few tool calls before claiming done.
    task = Task.new(
        prompt,
        new_session_id(),
        LoopStrategySelfVerifying(),
        budget=BudgetLimits(max_turns=12),
    )

    print(f"model         : {model_id}")
    print(f"base url      : {base_url}")
    print(f"workspace     : {workspace_root}")
    print("strategy      : SelfVerifying (draft -> critique -> revise)")
    print(f"max iterations: {max_iterations}")
    print("verifier      : EvaluatorResponseVerifier (PASS / FAIL:) wrapped in ReportingVerifier")
    print("\nThe agent will draft `parse_int_list`, an evaluator will critique it against the")
    print(
        f"five spec criteria, and the loop revises until PASS or {max_iterations} "
        "iterations elapse.\n"
    )

    draft_path = workspace_root / "parse_int_list.py"
    try:
        result = await harness.run(HarnessRunOptions(task))
    except OSError as e:
        # Ollama unreachable / endpoint refused the connection, etc.
        print(f"\ncould not reach the model — is `ollama serve` running? ({e})", file=sys.stderr)
        return 1

    if isinstance(result, RunResultSuccess):
        print(
            f"\n✓ PASSED — the evaluator accepted the draft (after at most "
            f"{max_iterations} iteration(s), {result.turns} build turn(s) total)."
        )
        if draft_path.exists():
            print(f"\n── final function ({draft_path}) ──\n{draft_path.read_text()}")
        return 0

    if isinstance(result, RunResultFailure) and isinstance(
        result.reason, HaltReasonSelfVerifyExhausted
    ):
        print(f"\n✗ EXHAUSTED — {result.reason.iterations} iteration(s) elapsed without a PASS.")
        print(f"last failure reason: {result.reason.last_reason}")
        if draft_path.exists():
            print(f"\n── last draft on disk ({draft_path}) ──\n{draft_path.read_text()}")
        print(
            "\nThis is an expected rough edge with small local models — see the README. "
            "Try a larger model or raise --max-iterations."
        )
        return 1

    print(f"\nrun did not succeed: {result!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
