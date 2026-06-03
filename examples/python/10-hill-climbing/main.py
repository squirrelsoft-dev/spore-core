"""spore-core example 10 — the ``HillClimbing`` loop strategy.

What this example demonstrates
------------------------------

**Iterative refinement under a scoring oracle is a harness concern, not
application logic.** The agent edits ONE file in place (``workspace/README.md``)
across iterations. After every iteration a custom :class:`MetricEvaluator` reads
that file and asks a *separate judge model* to score it on three dimensions —
Clarity, Completeness, Example quality (0–10 each) — returning the total/30
normalized to ``[0,1]``. The harness applies its keep/revert rule: a strictly
better score is KEPT; anything else is DISCARDED and (because
``revert_on_no_improvement`` is on) the workspace is ``git reset --hard``-ed back
to the best-so-far. The loop halts on **stagnation** (``MAX_STAGNATION``
consecutive non-improvements) or **budget** (``max_turns``). You write **no loop
code** — you wire a strategy, a metric evaluator, and an observability sink.

The contrast with example 09 (SelfVerifying) — the teaching point
-----------------------------------------------------------------

09 has a **binary exit condition**: a :class:`Verifier` returns PASS and the loop
*succeeds*, or it exhausts and *fails*. HillClimbing has **no PASS**. It is an
optimization loop: there is only *best-so-far*. It does not know it is "done" — it
only knows it has stopped improving. The terminal outcome is therefore a
:class:`HaltReasonStagnationLimitReached` or :class:`HaltReasonBudgetExceeded`,
NOT a success/fail verdict on quality.

SPEC NOTE — why this diverges from issue #99's original framing (Option A)
--------------------------------------------------------------------------

The original issue asked the agent to "climb until total ≥ 25/30 or max
iterations". Planning (#99 spec-resolution comment) established that framing does
NOT match the real ``HillClimbing`` strategy in spore-core:

* There is no score-threshold success condition. The loop keeps/reverts on
  *relative* improvement and halts on stagnation/budget — it never compares the
  metric against an absolute target.
* ``MAX_ITERATIONS`` is not a HillClimbing parameter; iterations are bounded by
  :class:`BudgetLimits` ``max_turns``. The ``MAX_ITERATIONS`` constant maps there.
* The shipped ``LlmJudgeEvaluator`` scores a FIXED construction-time string, so it
  cannot see the evolving draft. This example therefore ships a small
  example-local :class:`MetricEvaluator` (:class:`ReadmeQualityEvaluator`) that
  reads ``workspace/README.md`` through the sandbox each iteration before scoring.

Resolution = **Option A** (reframe to real semantics, no core change):

* ``SCORE_THRESHOLD`` (25/30) is kept as a **DISPLAY annotation only**. When a
  draft's total crosses it, the printed line is marked ``★ crossed target
  threshold``. It does **not** terminate the loop. (SPEC NOTE: display-only.)
* The per-iteration print is split across two seams, mirroring how the harness
  actually exposes the run:

  - the evaluator prints the draft + 3 sub-scores + total (it is the only place
    that sees the rubric breakdown), and
  - a custom :class:`ObservabilityProvider` handling
    :class:`WarnEventHillClimbingIteration` prints the kept/discarded/reverted
    decision (iteration, metric value, delta) — the harness emits exactly one
    such event per iteration.

There are no unresolved spec-question markers: every divergence above is resolved
against the source and the #99 resolution comment.

The seams this example wires
----------------------------

* :class:`ReadmeQualityEvaluator` — satisfies the :class:`MetricEvaluator`
  Protocol structurally; reads the file via the ``SandboxProvider``, runs a fresh
  judge-model call, prints the rubric.
* :class:`ReportingObservability` — subclasses
  :class:`InMemoryObservabilityProvider` and prints each
  ``HillClimbingIteration`` decision before delegating to the in-memory base.

Constants (see their inline comments)
-------------------------------------

* ``MAX_ITERATIONS``  — maps to ``BudgetLimits.max_turns`` (default 6).
* ``MAX_STAGNATION``  — consecutive non-improvements before halt (2).
* ``SCORE_THRESHOLD`` — DISPLAY annotation only (25). Never terminates.
* ``DIMENSION_MAX`` / ``TOTAL_MAX`` — 10 per dimension, 30 total.

Run it::

    ollama serve &
    ollama pull llama3.2
    uv run main.py                       # default model llama3.2, 6-iteration budget
    uv run main.py --max-iterations 8
    uv run main.py --model qwen2.5-coder:7b

See the README for the honest rough-edges section.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

from spore_core import (
    BudgetLimits,
    EvaluateResult,
    HaltReasonBudgetExceeded,
    HaltReasonHillClimbingMisconfigured,
    HaltReasonStagnationLimitReached,
    HarnessBuilder,
    HarnessRunOptions,
    LoopStrategyHillClimbing,
    Message,
    MetricErrorExecutionFailed,
    MetricResult,
    ModelParams,
    ModelRequest,
    OllamaModelInterface,
    OptimizationDirection,
    Role,
    RunResultFailure,
    RunResultSuccess,
    SessionStateSnapshot,
    Task,
    TextBlock,
    TextContent,
    WorkspaceConfig,
    WorkspaceScopedSandbox,
    new_session_id,
)
from spore_core.harness import SandboxProvider
from spore_core.model import ModelInterface
from spore_core.observability import (
    InMemoryObservabilityProvider,
    WarnEventHillClimbingIteration,
    WarnSpan,
)
from spore_tools import StandardTools

# ============================================================================
# Constants — UPPER_SNAKE, with the spec semantics in inline comments.
# ============================================================================

# Climbing-iteration ceiling. Maps to ``BudgetLimits.max_turns`` — this is the
# BUDGET, not a success target. The loop may halt EARLIER on stagnation. There is
# no "reached the goal" outcome; HillClimbing always halts on budget or
# stagnation. Six gives a small local model room to make a few real edits.
MAX_ITERATIONS = 6

# Consecutive non-improvements tolerated before the loop halts with
# ``HaltReasonStagnationLimitReached``. The stagnation counter resets to 0 on any
# kept (strictly-improving) iteration. Maps to ``max_stagnation``.
MAX_STAGNATION = 2

# DISPLAY ANNOTATION ONLY. When a draft's total score (0–30) reaches this, the
# evaluator marks the printed line ``★ crossed target threshold``. SPEC NOTE: this
# does NOT terminate the loop — HillClimbing has no score-threshold exit.
SCORE_THRESHOLD = 25

# Max score per rubric dimension (Clarity, Completeness, Example quality).
DIMENSION_MAX = 10

# Max total across the three dimensions (``3 * DIMENSION_MAX``).
TOTAL_MAX = 3 * DIMENSION_MAX

# The file under refinement, relative to the workspace root. The build agent edits
# this in place; the evaluator reads it back through the sandbox.
DRAFT_FILE = "README.md"

# The task the build agent is asked to perform each iteration. It edits ONE file
# in place — the climb is over successive revisions of the same README.
TASK_PROMPT = (
    "You are writing the README.md for a fictional Rust crate called `ironwood`, "
    "a small library for parsing and validating semantic-version strings. Use the "
    "write_file tool to save your README to `README.md`. If a `README.md` already "
    "exists, first read_file it, then improve it and write it back.\n"
    "\n"
    "A great README for this crate has THREE qualities, each scored 0–10 by a "
    "reviewer:\n"
    "  1. CLARITY: a crisp one-line summary, then prose a newcomer can follow.\n"
    "  2. COMPLETENESS: what the crate does, how to add it to Cargo.toml, the main "
    "API surface, and an error/edge-cases note.\n"
    "  3. EXAMPLE QUALITY: at least one fenced ```rust code block showing a real "
    "call and its expected result.\n"
    "\n"
    "Write the BEST README you can, then report that you are done."
)

# System prompt shared by the build agent. (The judge model is prompted separately
# by the evaluator; it does not share this prompt.) Reinforces the minimal
# file-tool contract.
SYSTEM_PROMPT = (
    "You write developer documentation in Markdown. Your only tools are write_file "
    "(save a file to the workspace) and read_file (read a file back). You have no "
    "shell and cannot run code. When asked to write or improve the README: read any "
    "existing file first, write the improved Markdown with write_file, then say you "
    "are done."
)

# The rubric handed to the judge model. Kept separate from the build prompt so the
# judge scores independently of how the writer was instructed.
JUDGE_RUBRIC = (
    "You are a strict technical-documentation reviewer. Score the README below on "
    "THREE dimensions, each an integer from 0 to 10:\n"
    "  - CLARITY: is there a crisp one-line summary and prose a newcomer can "
    "follow?\n"
    "  - COMPLETENESS: does it cover what the crate does, how to add it to "
    "Cargo.toml, the main API, and an error/edge-cases note?\n"
    "  - EXAMPLE_QUALITY: is there at least one fenced ```rust block with a real "
    "call and expected result?\n"
    "\n"
    "Reply with EXACTLY these three lines and nothing else:\n"
    "clarity: <0-10>\n"
    "completeness: <0-10>\n"
    "example_quality: <0-10>"
)


def _parse_dimension(text: str, name: str) -> int:
    """Parse a ``name: <int>`` line from the judge reply, clamped to
    ``[0, DIMENSION_MAX]``. A missing or unparseable line scores 0 — a malformed
    judge reply must not crash the run (it just reads as a poor score, which the
    loop treats as a normal outcome)."""
    prefix = f"{name}:"
    for line in text.splitlines():
        lower = line.strip().lower()
        if lower.startswith(prefix):
            rest = lower[len(prefix) :].split()
            if rest:
                try:
                    return min(int(rest[0]), DIMENSION_MAX)
                except ValueError:
                    return 0
    return 0


class ReadmeQualityEvaluator:
    """Example-local :class:`MetricEvaluator` (satisfies the Protocol
    structurally — no inheritance, per the Python conventions).

    Scores ``workspace/README.md`` by reading it through the ``SandboxProvider``
    then making a SEPARATE judge-model call that returns three sub-scores. The
    value reported to the harness is ``total / TOTAL_MAX``, normalized to
    ``[0,1]``, with ``direction = "maximize"``.

    SPEC NOTE: this replaces the shipped ``LlmJudgeEvaluator``, which scores a
    fixed construction-time string and so cannot observe the evolving draft.
    """

    def __init__(self, judge: ModelInterface) -> None:
        self._judge = judge

    async def evaluate(
        self,
        sandbox: SandboxProvider,
        session_state: SessionStateSnapshot,
    ) -> EvaluateResult:
        start = time.monotonic()

        # Read the current draft through the sandbox root, exactly as the core
        # evaluators do. A missing draft (e.g. the baseline before the agent has
        # written anything) scores 0 rather than erroring.
        draft_path = sandbox.workspace_root() / DRAFT_FILE
        try:
            draft = draft_path.read_text()
        except OSError:
            draft = ""

        if not draft.strip():
            clarity = completeness = example = total = 0
            print(f"\n── evaluator: no draft on disk yet (baseline) — total 0/{TOTAL_MAX} ──")
        else:
            prompt = f"{JUDGE_RUBRIC}\n\n----- README under review -----\n{draft}"
            request = ModelRequest(
                messages=[Message(role=Role.USER, content=TextContent(text=prompt))],
                params=ModelParams(),
            )
            try:
                response = await self._judge.call(request)
            except Exception as e:  # noqa: BLE001 — surface any judge failure as a typed metric error
                return MetricErrorExecutionFailed(reason=f"judge model call failed: {e}")
            text = "\n".join(b.text for b in response.content if isinstance(b, TextBlock))

            clarity = _parse_dimension(text, "clarity")
            completeness = _parse_dimension(text, "completeness")
            example = _parse_dimension(text, "example_quality")
            total = clarity + completeness + example

            print(f"\n── evaluator: scored draft ({len(draft)} bytes) ──")
            print(draft)
            print(f"  clarity        : {clarity}/{DIMENSION_MAX}")
            print(f"  completeness   : {completeness}/{DIMENSION_MAX}")
            print(f"  example quality: {example}/{DIMENSION_MAX}")

        # SPEC NOTE: the threshold is DISPLAY-ONLY. We annotate the line; we do NOT
        # halt the loop here. The harness halts on stagnation/budget.
        crossed = "  ★ crossed target threshold" if total >= SCORE_THRESHOLD else ""
        print(f"  TOTAL          : {total}/{TOTAL_MAX}{crossed}")

        return MetricResult(
            value=total / TOTAL_MAX,
            raw_output=draft,
            duration=time.monotonic() - start,
            metadata={
                "clarity": str(clarity),
                "completeness": str(completeness),
                "example_quality": str(example),
                "total": str(total),
            },
        )

    def direction(self) -> OptimizationDirection:
        return "maximize"

    def description(self) -> str:
        return f"ironwood README quality (clarity+completeness+example, /{TOTAL_MAX})"


class ReportingObservability(InMemoryObservabilityProvider):
    """An :class:`ObservabilityProvider` that PRINTS each
    ``HillClimbingIteration`` decision, then delegates everything to the
    in-memory base.

    This is the seam the harness uses to report the per-iteration keep/revert
    decision — the evaluator prints the scores, this prints what the loop DID with
    them. We subclass :class:`InMemoryObservabilityProvider` (a concrete class) and
    override only :meth:`emit_warn`; every other ``emit_*`` / query method is
    inherited verbatim, so the trace is still recorded as usual.
    """

    def __init__(self, max_iterations: int) -> None:
        super().__init__()
        self._max_iterations = max_iterations

    def emit_warn(self, span: WarnSpan) -> None:
        event = span.event
        if isinstance(event, WarnEventHillClimbingIteration):
            # ``iteration`` is 0-based on the wire (0 = baseline). Display 1-based.
            n = event.iteration + 1
            value = f"{event.metric_value:.3f}" if event.metric_value is not None else "n/a"
            delta = f"{event.delta:+.3f}" if event.delta is not None else "—"
            reverted = " (workspace git-reset to best-so-far)" if event.reverted else ""
            print(
                f"\n══ iteration {n}/{self._max_iterations} — {event.status} ══  "
                f"metric={value} (Δ {delta}){reverted}"
            )
        # Delegate so the warn span is still recorded in the in-memory trace.
        super().emit_warn(span)


def _init_git_workspace(root: Path) -> None:
    """``git init`` the workspace and make an initial commit if it is not already a
    repo, so ``revert_on_no_improvement``'s ``git reset --hard`` has a baseline.
    Best-effort and idempotent: a missing ``git`` or an existing repo is fine."""
    if (root / ".git").exists():
        return

    def run(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    try:
        run("init")
        # Local identity so the initial commit succeeds without global git config.
        run("config", "user.email", "example@spore-core.invalid")
        run("config", "user.name", "spore-core example")
        run("add", "-A")
        # An empty initial commit is fine if the dir is otherwise empty.
        run("commit", "--allow-empty", "-m", "baseline")
    except (OSError, subprocess.CalledProcessError) as e:
        print(
            f"\n(warning: could not git-init the workspace: {e}) — reverts may not work",
            file=sys.stderr,
        )


def _report_best(best_metric: float | None, draft_path: Path) -> None:
    """Print the best-so-far metric (when known) and the final draft on disk."""
    if best_metric is not None:
        total = round(best_metric * TOTAL_MAX)
        print(f"\n── best score seen: {total}/{TOTAL_MAX} (normalized {best_metric:.3f}) ──")
    if draft_path.exists():
        print(f"\n── final draft ({draft_path}) ──\n{draft_path.read_text()}")
    else:
        print(f"\n(no draft was written to {draft_path})")


async def main() -> int:
    parser = argparse.ArgumentParser(description="spore-core hill-climbing agent")
    parser.add_argument("--model")
    parser.add_argument("--prompt")
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Climbing budget ceiling → max_turns (default 6, or SPORE_MAX_ITERATIONS).",
    )
    args = parser.parse_args()

    model_id = args.model or os.environ.get("SPORE_OLLAMA_MODEL") or "llama3.2"
    base_url = os.environ.get("SPORE_OLLAMA_BASE_URL", OllamaModelInterface.DEFAULT_BASE_URL)

    # Iteration budget: CLI flag wins, then env var, then MAX_ITERATIONS. A
    # non-positive value falls back to the default. SPEC NOTE: this is the BUDGET
    # ceiling (max_turns), NOT a success target.
    max_iterations = args.max_iterations
    if max_iterations is None:
        env_iters = os.environ.get("SPORE_MAX_ITERATIONS")
        max_iterations = int(env_iters) if env_iters and env_iters.isdigit() else MAX_ITERATIONS
    if max_iterations <= 0:
        max_iterations = MAX_ITERATIONS

    prompt = args.prompt or TASK_PROMPT

    # The agent edits this example's ``workspace/`` in place. Resolve it relative to
    # this source file so ``uv run main.py`` works from anywhere, and canonicalize
    # it — the sandbox requires a canonical, existing root.
    workspace_root = Path(__file__).parent / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace_root = workspace_root.resolve(strict=True)

    # git-init the workspace so ``revert_on_no_improvement``'s ``git reset --hard``
    # has a clean baseline to return to. Idempotent: skip if already a repo.
    _init_git_workspace(workspace_root)

    # Two model instances on the same Ollama endpoint: one drives the build agent
    # (writing the README), one is the judge the evaluator calls to score it.
    build_model = OllamaModelInterface.with_base_url(model_id, base_url)
    judge_model = OllamaModelInterface.with_base_url(model_id, base_url)
    evaluator = ReadmeQualityEvaluator(judge_model)

    observability = ReportingObservability(max_iterations)

    # Build harness: conversational preset, workspace sandbox, the minimal file
    # tool set (write_file + read_file), shared system prompt, the metric evaluator
    # (required for HillClimbing), and the observability sink.
    sandbox = WorkspaceScopedSandbox(WorkspaceConfig(root=workspace_root))
    harness = (
        HarnessBuilder.conversational(build_model)
        .sandbox(sandbox)
        .tool(StandardTools.write_file())
        .tool(StandardTools.read_file())
        .system_prompt(SYSTEM_PROMPT)
        .metric_evaluator(evaluator)
        .observability(observability)
        .build()
    )

    # THE STRATEGY. No loop code below — the harness runs the climb. ``max_turns``
    # bounds the NUMBER OF ITERATIONS (the budget ceiling), and ``max_stagnation``
    # can halt sooner. SPEC NOTE: there is no score-threshold field — by design.
    task = Task.new(
        prompt,
        new_session_id(),
        LoopStrategyHillClimbing(
            direction="maximize",
            max_stagnation=MAX_STAGNATION,
            revert_on_no_improvement=True,
            min_improvement_delta=None,
        ),
        budget=BudgetLimits(max_turns=max_iterations),
    )

    print(f"model         : {model_id}")
    print(f"base url      : {base_url}")
    print(f"workspace     : {workspace_root}")
    print("strategy      : HillClimbing (score → keep/revert → climb)")
    print("direction     : maximize (higher README score is better)")
    print(f"max iterations: {max_iterations} (budget ceiling — NOT a success target)")
    print(f"max stagnation: {MAX_STAGNATION} (halt after this many non-improvements)")
    print(f"threshold     : {SCORE_THRESHOLD}/{TOTAL_MAX} — DISPLAY ONLY (★ marks it; never halts)")
    print(f"\nThe agent will draft and refine `{DRAFT_FILE}`; each iteration a judge model")
    print("scores it on three dimensions, and the loop keeps the best — reverting the rest —")
    print("until it stops improving (stagnation) or the budget is spent. There is no PASS.\n")

    draft_path = workspace_root / DRAFT_FILE
    try:
        result = await harness.run(HarnessRunOptions(task))
    except OSError as e:
        # Ollama unreachable / endpoint refused the connection, etc.
        print(f"\ncould not reach the model — is `ollama serve` running? ({e})", file=sys.stderr)
        return 1

    if isinstance(result, RunResultFailure):
        reason = result.reason
        if isinstance(reason, HaltReasonStagnationLimitReached):
            _report_best(reason.best_metric, draft_path)
            print(
                f"\n■ HALTED ON STAGNATION — {reason.iterations} consecutive "
                "non-improving iteration(s)."
            )
            print("This is the NORMAL terminal outcome for HillClimbing: it stopped because it")
            print(
                "could not improve, not because it hit a target. The file on disk is best-so-far."
            )
            return 0
        if isinstance(reason, HaltReasonBudgetExceeded):
            _report_best(None, draft_path)
            print(f"\n■ HALTED ON BUDGET — exhausted the iteration ceiling ({reason.limit_type}).")
            print("Also a normal terminal outcome: the climb ran out of budget while still")
            print("(possibly) improving. The file on disk is the best-so-far draft.")
            return 0
        if isinstance(reason, HaltReasonHillClimbingMisconfigured):
            print(f"\nHillClimbing misconfigured: {reason.reason}", file=sys.stderr)
            return 1
        print(f"\nrun did not complete as expected: {reason!r}", file=sys.stderr)
        return 1

    if isinstance(result, RunResultSuccess):
        # HillClimbing does not normally return Success (it has no success
        # condition); surface it honestly if a future core revision does.
        _report_best(None, draft_path)
        print(f"\n■ run returned Success after {result.turns} turn(s) — best-so-far draft on disk.")
        return 0

    print(f"\nrun did not complete as expected: {result!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
