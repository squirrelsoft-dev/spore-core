"""MetricEvaluator — pluggable scoring for the ``HillClimbing`` loop strategy
(issue #23).

Mirrors the Rust reference at ``rust/crates/spore-core/src/metric.rs``.

See ``docs/harness-engineering-concepts.md`` § "Loop Strategies / HillClimbing"
and the issue spec for authoritative rules. This module ships:

* :class:`MetricError` / :class:`MetricResult` — the error and success
  surfaces.
* The :class:`MetricEvaluator` Protocol — used by the harness to score each
  iteration of a ``HillClimbing`` run.
* Standard evaluators: :class:`CommandMetricEvaluator`,
  :class:`TestPassRateEvaluator`, :class:`LatencyEvaluator`,
  :class:`LlmJudgeEvaluator`.
* :class:`ResultsEntry` / :class:`IterationStatus` — the row format the
  harness writes to ``.spore/results/{task_id}.tsv``.
* :func:`should_keep` — the keep/revert decision the harness applies after
  each iteration.

Rules enforced:

* :meth:`MetricEvaluator.evaluate` receives the :class:`SandboxProvider`.
  All subprocess execution goes through it; evaluators never spawn
  processes directly.
* :class:`CommandMetricEvaluator` writes captured stdout+stderr to
  ``log_output_to`` *before* parsing the metric, so a partial run is still
  diagnosable.
* A regex that does not match the captured output is
  :class:`MetricErrorParseFailed`, not a crash.
* Non-zero exit from the subprocess maps to :class:`MetricErrorCrashed`;
  an exceeded timeout maps to :class:`MetricErrorTimeout`. Both are valid
  iteration outcomes — the harness logs them and asks the agent to try a
  different approach.
* :func:`should_keep` strictly compares against ``current_best``: a delta
  of exactly ``min_delta`` (or ``0.0`` when ``None``) does NOT count as
  improvement. Equal scores are discarded.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .harness import OptimizationDirection, SandboxProvider
from .model import ModelInterface, ModelParams, ModelRequest, Role, TextBlock, TextContent
from .model import Message as ModelMessage
from .termination import SessionStateSnapshot


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# MetricError (tagged union on ``kind``)
# ============================================================================


class MetricErrorExecutionFailed(_Model):
    kind: Literal["execution_failed"] = "execution_failed"
    reason: str


class MetricErrorTimeout(_Model):
    """Duration in seconds (float) — matches Rust's ``duration_secs`` adapter."""

    kind: Literal["timeout"] = "timeout"
    after: float


class MetricErrorParseFailed(_Model):
    kind: Literal["parse_failed"] = "parse_failed"
    output: str
    pattern: str


class MetricErrorCrashed(_Model):
    kind: Literal["crashed"] = "crashed"
    log: str


MetricError = Annotated[
    MetricErrorExecutionFailed | MetricErrorTimeout | MetricErrorParseFailed | MetricErrorCrashed,
    Field(discriminator="kind"),
]


# ============================================================================
# MetricResult
# ============================================================================


class MetricResult(_Model):
    value: float
    raw_output: str = ""
    duration: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)


# ============================================================================
# IterationStatus / ResultsEntry
# ============================================================================


IterationStatus = Literal["kept", "discarded", "crashed", "timeout"]


def iteration_status_from_error(err: MetricError) -> IterationStatus:
    """Map an evaluator error to the iteration status the harness records.

    Successful evaluations are routed through :func:`should_keep` to resolve
    ``"kept"`` vs ``"discarded"``.
    """

    if isinstance(err, MetricErrorTimeout):
        return "timeout"
    return "crashed"


class ResultsEntry(_Model):
    iteration: int
    commit_hash: str | None = None
    metric_value: float
    direction: OptimizationDirection
    status: IterationStatus
    duration: float
    description: str
    metadata: dict[str, str] = Field(default_factory=dict)


# ============================================================================
# Evaluation outcome (returned-as-value pattern; pre-3.10 type alias)
# ============================================================================


class _MetricOk:
    __slots__ = ("result",)

    def __init__(self, result: MetricResult) -> None:
        self.result = result


class _MetricErr:
    __slots__ = ("error",)

    def __init__(self, error: MetricError) -> None:
        self.error = error


EvaluateResult = (
    MetricResult
    | MetricErrorExecutionFailed
    | MetricErrorTimeout
    | MetricErrorParseFailed
    | MetricErrorCrashed
)


def is_metric_error(value: EvaluateResult) -> bool:
    return not isinstance(value, MetricResult)


# ============================================================================
# MetricEvaluator Protocol
# ============================================================================


@runtime_checkable
class MetricEvaluator(Protocol):
    """Pluggable scoring strategy for the ``HillClimbing`` loop.

    The harness calls :meth:`evaluate` after the agent completes each
    iteration and feeds the result into :func:`should_keep`.

    :meth:`evaluate` returns either a :class:`MetricResult` on success or
    one of the :data:`MetricError` variants as a value — recoverable
    failures never raise across the agent/harness boundary.
    """

    async def evaluate(
        self,
        sandbox: SandboxProvider,
        session_state: SessionStateSnapshot,
    ) -> EvaluateResult: ...

    def direction(self) -> OptimizationDirection: ...

    def description(self) -> str: ...


# ============================================================================
# should_keep
# ============================================================================


def should_keep(
    new_value: float,
    current_best: float,
    direction: OptimizationDirection,
    min_delta: float | None = None,
) -> bool:
    """The keep-or-revert decision the harness applies after every
    iteration.

    Returns ``True`` only when ``new_value`` strictly beats ``current_best``
    by more than ``min_delta`` (default ``0.0``). Equal scores are
    discarded — a flat run is not progress.
    """

    if direction == "minimize":
        delta = current_best - new_value
    else:
        delta = new_value - current_best
    threshold = min_delta if min_delta is not None else 0.0
    return delta > threshold


# ============================================================================
# Internal helpers
# ============================================================================


def parse_metric(
    output: str, pattern: str
) -> float | MetricErrorExecutionFailed | MetricErrorParseFailed:
    """Compile ``pattern`` and extract its first capture group from
    ``output``, parsed as float. Returns the value, an
    :class:`MetricErrorExecutionFailed` on invalid regex, or a
    :class:`MetricErrorParseFailed` on no-match / unparseable capture.
    """

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return MetricErrorExecutionFailed(reason=f"invalid regex {pattern!r}: {e}")
    m = regex.search(output)
    if m is None:
        return MetricErrorParseFailed(output=output, pattern=pattern)
    try:
        group = m.group(1)
    except IndexError:
        return MetricErrorParseFailed(output=output, pattern=pattern)
    if group is None:
        return MetricErrorParseFailed(output=output, pattern=pattern)
    try:
        return float(group.strip())
    except ValueError:
        return MetricErrorParseFailed(output=output, pattern=pattern)


async def _write_log(sandbox: SandboxProvider, path: Path, body: str) -> None:
    """Write ``body`` to ``workspace_root / path``. Failures are swallowed
    — the log is best-effort diagnosis only."""

    try:
        target = sandbox.workspace_root() / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    except OSError:
        pass


# ============================================================================
# CommandMetricEvaluator
# ============================================================================


class CommandMetricEvaluator:
    """Runs a shell command through the sandbox, parses a numeric metric out
    of its combined stdout+stderr via a single-capture-group regex.

    Models the autoresearch pattern (``uv run train.py`` ⇒ ``val_bpb``).
    """

    def __init__(
        self,
        *,
        command: str,
        args: list[str],
        metric_pattern: str,
        timeout: float,
        log_output_to: Path,
        working_dir: Path | None = None,
        direction: OptimizationDirection,
        description: str,
    ) -> None:
        self.command = command
        self.args = args
        self.metric_pattern = metric_pattern
        self.timeout = timeout
        self.log_output_to = log_output_to
        self.working_dir = working_dir
        self._direction: OptimizationDirection = direction
        self._description = description

    def direction(self) -> OptimizationDirection:
        return self._direction

    def description(self) -> str:
        return self._description

    async def evaluate(
        self,
        sandbox: SandboxProvider,
        session_state: SessionStateSnapshot,
    ) -> EvaluateResult:
        _ = session_state
        start = time.monotonic()
        try:
            out = await sandbox.execute_command(
                self.command,
                self.args,
                self.working_dir,
                self.timeout,
            )
        except Exception as e:  # noqa: BLE001 — surface sandbox rejections as ExecutionFailed
            return MetricErrorExecutionFailed(reason=f"sandbox rejected command: {e!r}")

        combined = f"{out.stdout}{out.stderr}"
        # Always write the log BEFORE parsing — even on failure the log is
        # diagnosable.
        await _write_log(sandbox, self.log_output_to, combined)

        if out.timed_out:
            return MetricErrorTimeout(after=self.timeout)
        if out.exit_code != 0:
            return MetricErrorCrashed(log=combined)

        parsed = parse_metric(combined, self.metric_pattern)
        if isinstance(parsed, (MetricErrorExecutionFailed, MetricErrorParseFailed)):
            return parsed
        metadata = {
            "command": self.command,
            "exit_code": str(out.exit_code),
        }
        return MetricResult(
            value=parsed,
            raw_output=combined,
            duration=time.monotonic() - start,
            metadata=metadata,
        )


# ============================================================================
# TestPassRateEvaluator
# ============================================================================


class TestPassRateEvaluator:
    """Runs a test suite, extracts pass / total counts via two regexes, and
    reports the fraction of passing tests in ``[0.0, 1.0]``. Direction is
    fixed to ``"maximize"``.
    """

    # Tell pytest not to collect this as a test class — the ``Test`` prefix
    # is mandated by the cross-language spec.
    __test__ = False

    def __init__(
        self,
        *,
        command: str,
        args: list[str] | None = None,
        timeout: float,
        pass_pattern: str,
        total_pattern: str,
        working_dir: Path | None = None,
    ) -> None:
        self.command = command
        self.args = args or []
        self.timeout = timeout
        self.pass_pattern = pass_pattern
        self.total_pattern = total_pattern
        self.working_dir = working_dir

    def direction(self) -> OptimizationDirection:
        return "maximize"

    def description(self) -> str:
        return f"test pass rate ({self.command})"

    async def evaluate(
        self,
        sandbox: SandboxProvider,
        session_state: SessionStateSnapshot,
    ) -> EvaluateResult:
        _ = session_state
        start = time.monotonic()
        try:
            out = await sandbox.execute_command(
                self.command,
                self.args,
                self.working_dir,
                self.timeout,
            )
        except Exception as e:  # noqa: BLE001
            return MetricErrorExecutionFailed(reason=f"sandbox rejected command: {e!r}")

        combined = f"{out.stdout}{out.stderr}"
        if out.timed_out:
            return MetricErrorTimeout(after=self.timeout)
        # A failing test run is a normal outcome here — only treat as crashed
        # if we cannot parse it.

        pass_parsed = parse_metric(combined, self.pass_pattern)
        if isinstance(pass_parsed, (MetricErrorExecutionFailed, MetricErrorParseFailed)):
            return pass_parsed
        total_parsed = parse_metric(combined, self.total_pattern)
        if isinstance(total_parsed, (MetricErrorExecutionFailed, MetricErrorParseFailed)):
            return total_parsed
        if total_parsed <= 0.0:
            return MetricErrorParseFailed(output=combined, pattern=self.total_pattern)
        value = pass_parsed / total_parsed
        metadata = {
            "pass": str(pass_parsed),
            "total": str(total_parsed),
        }
        return MetricResult(
            value=value,
            raw_output=combined,
            duration=time.monotonic() - start,
            metadata=metadata,
        )


# ============================================================================
# LatencyEvaluator
# ============================================================================


class LatencyEvaluator:
    """Measures wall-clock latency of ``command``, averaged over
    ``measured_runs`` trials after ``warmup_runs`` warm-ups. Direction is
    fixed to ``"minimize"``.
    """

    def __init__(
        self,
        *,
        command: str,
        args: list[str] | None = None,
        warmup_runs: int,
        measured_runs: int,
        timeout: float,
        working_dir: Path | None = None,
    ) -> None:
        self.command = command
        self.args = args or []
        self.warmup_runs = warmup_runs
        self.measured_runs = measured_runs
        self.timeout = timeout
        self.working_dir = working_dir

    def direction(self) -> OptimizationDirection:
        return "minimize"

    def description(self) -> str:
        return f"latency ({self.command})"

    async def evaluate(
        self,
        sandbox: SandboxProvider,
        session_state: SessionStateSnapshot,
    ) -> EvaluateResult:
        _ = session_state
        if self.measured_runs <= 0:
            return MetricErrorExecutionFailed(reason="measured_runs must be > 0")
        start = time.monotonic()

        for _i in range(self.warmup_runs):
            try:
                await sandbox.execute_command(
                    self.command,
                    self.args,
                    self.working_dir,
                    self.timeout,
                )
            except Exception as e:  # noqa: BLE001
                return MetricErrorExecutionFailed(reason=f"sandbox rejected command: {e!r}")

        total = 0.0
        last_output = ""
        for _i in range(self.measured_runs):
            trial_start = time.monotonic()
            try:
                out = await sandbox.execute_command(
                    self.command,
                    self.args,
                    self.working_dir,
                    self.timeout,
                )
            except Exception as e:  # noqa: BLE001
                return MetricErrorExecutionFailed(reason=f"sandbox rejected command: {e!r}")
            if out.timed_out:
                return MetricErrorTimeout(after=self.timeout)
            if out.exit_code != 0:
                return MetricErrorCrashed(log=f"{out.stdout}{out.stderr}")
            total += time.monotonic() - trial_start
            last_output = f"{out.stdout}{out.stderr}"

        avg = total / self.measured_runs
        metadata = {
            "warmup_runs": str(self.warmup_runs),
            "measured_runs": str(self.measured_runs),
        }
        return MetricResult(
            value=avg,
            raw_output=last_output,
            duration=time.monotonic() - start,
            metadata=metadata,
        )


# ============================================================================
# LlmJudgeEvaluator
# ============================================================================


class JudgeModelConfig(_Model):
    provider: str
    model_id: str
    params: ModelParams = Field(default_factory=ModelParams)


class LlmJudgeEvaluator:
    """Uses an LLM-as-judge to score ``sample_input`` against ``rubric``.

    The judge is expected to emit a ``score: <number>`` line; that number is
    clamped to ``score_range`` and normalized to ``[0.0, 1.0]``. Direction
    is fixed to ``"maximize"``.

    The trait shape from the spec only carries a :class:`JudgeModelConfig`;
    the concrete :class:`ModelInterface` used to dispatch the judge call is
    supplied at construction time, keeping the evaluator independent of
    model routing while letting :class:`JudgeModelConfig` flow through the
    results log for observability.
    """

    _SCORE_RE = re.compile(r"(?i)score\s*:\s*([-+]?\d+(?:\.\d+)?)")

    def __init__(
        self,
        *,
        judge_model: JudgeModelConfig,
        rubric: str,
        score_range: tuple[float, float],
        sample_input: str,
        client: ModelInterface,
    ) -> None:
        self.judge_model = judge_model
        self.rubric = rubric
        self.score_range = score_range
        self.sample_input = sample_input
        self.client = client

    def direction(self) -> OptimizationDirection:
        return "maximize"

    def description(self) -> str:
        return f"llm judge ({self.judge_model.provider}/{self.judge_model.model_id})"

    def _parse_score(
        self, text: str
    ) -> float | MetricErrorExecutionFailed | MetricErrorParseFailed:
        lo, hi = self.score_range
        if hi <= lo:
            return MetricErrorExecutionFailed(reason=f"invalid score_range: ({lo}, {hi})")
        m = self._SCORE_RE.search(text)
        if m is None:
            return MetricErrorParseFailed(output=text, pattern=self._SCORE_RE.pattern)
        try:
            raw = float(m.group(1))
        except ValueError:
            return MetricErrorParseFailed(output=text, pattern=self._SCORE_RE.pattern)
        clamped = max(lo, min(hi, raw))
        return (clamped - lo) / (hi - lo)

    async def evaluate(
        self,
        sandbox: SandboxProvider,
        session_state: SessionStateSnapshot,
    ) -> EvaluateResult:
        _ = sandbox
        _ = session_state
        start = time.monotonic()
        prompt = (
            f"{self.rubric}\n\nInput to evaluate:\n{self.sample_input}\n\n"
            f"Reply with a single line `score: <number>` where the number is within "
            f"{self.score_range}."
        )
        request = ModelRequest(
            messages=[ModelMessage(role=Role.USER, content=TextContent(text=prompt))],
            tools=[],
            params=self.judge_model.params,
            stream=False,
        )
        try:
            response = await self.client.call(request)
        except Exception as e:  # noqa: BLE001
            return MetricErrorExecutionFailed(reason=f"judge model call failed: {e}")

        text = "\n".join(b.text for b in response.content if isinstance(b, TextBlock))
        score = self._parse_score(text)
        if isinstance(score, (MetricErrorExecutionFailed, MetricErrorParseFailed)):
            return score
        metadata = {
            "judge_model": self.judge_model.model_id,
            "judge_provider": self.judge_model.provider,
        }
        return MetricResult(
            value=score,
            raw_output=text,
            duration=time.monotonic() - start,
            metadata=metadata,
        )


__all__ = [
    "CommandMetricEvaluator",
    "EvaluateResult",
    "IterationStatus",
    "JudgeModelConfig",
    "LatencyEvaluator",
    "LlmJudgeEvaluator",
    "MetricError",
    "MetricErrorCrashed",
    "MetricErrorExecutionFailed",
    "MetricErrorParseFailed",
    "MetricErrorTimeout",
    "MetricEvaluator",
    "MetricResult",
    "ResultsEntry",
    "TestPassRateEvaluator",
    "iteration_status_from_error",
    "parse_metric",
    "should_keep",
]
