""":class:`TaskVerifier` Protocol + standard implementations.

Mirrors ``rust/crates/spore-eval/src/verifier.rs``.

Rules enforced here:
  * 9  ``is_deterministic()`` true for test-suite/result verifiers, false for
       the LLM judge.
  * 10 :class:`TestSuiteVerifier`: command pass-rate; passed = score==1.0.
  * 11 :class:`CompositeVerifier`: weighted mean; passed = all required;
       determinism = AND.
  * 12 :class:`MetricEvaluatorVerifier`: wraps a ``MetricEvaluator``,
       normalizes the value.
  * 13 :class:`LlmJudgeVerifier`: thin; non-deterministic; judge injected.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from spore_core.harness import (
    BaseSandboxProvider,
    OptimizationDirection,
    RunResult,
    RunResultSuccess,
    SessionId,
    SessionState,
    TaskId,
)
from spore_core.metric import MetricEvaluator, MetricResult
from spore_core.model import (
    Message,
    ModelInterface,
    ModelParams,
    ModelRequest,
    Role,
    TextBlock,
    TextContent,
)
from spore_core.termination import SessionStateSnapshot

from .task import (
    EvalTask,
    VerificationResult,
    VerifierSpec,
    VerifierSpecAlwaysFail,
    VerifierSpecAlwaysPass,
    VerifierSpecComposite,
    VerifierSpecLlmJudge,
    VerifierSpecMetricEvaluator,
    VerifierSpecTestSuite,
    VerifyError,
)

# ``EPSILON`` for "score == 1.0" comparisons; mirrors Rust's ``f64::EPSILON``.
_EPS = 2.220446049250313e-16


# ============================================================================
# TaskVerifier Protocol
# ============================================================================


@runtime_checkable
class TaskVerifier(Protocol):
    """Verifies whether a task run satisfied its goal."""

    async def verify(
        self,
        task: EvalTask,
        run: RunResult,
        workspace: Path,
    ) -> VerificationResult: ...

    def is_deterministic(self) -> bool: ...


# ============================================================================
# build_verifier — resolve a VerifierSpec into a TaskVerifier
# ============================================================================


def build_verifier(spec: VerifierSpec) -> TaskVerifier:
    """Resolve a :data:`VerifierSpec` to a concrete verifier. ``metric_evaluator``
    specs have no built-in concrete evaluator (it is injected for non-fixture
    use), so they resolve to a normalizing placeholder that scores from the
    run's success flag; ``llm_judge`` resolves to a non-deterministic stub."""
    if isinstance(spec, VerifierSpecTestSuite):
        return TestSuiteVerifier(
            command=spec.command,
            args=list(spec.args),
            timeout=float(spec.timeout_secs) if spec.timeout_secs is not None else 60.0,
        )
    if isinstance(spec, VerifierSpecComposite):
        children = [(build_verifier(c.spec), c.weight, c.required) for c in spec.children]
        return CompositeVerifier(children)
    if isinstance(spec, VerifierSpecMetricEvaluator):
        return _NormalizingSuccessVerifier()
    if isinstance(spec, VerifierSpecLlmJudge):
        return _StubLlmJudgeVerifier(score_range=spec.score_range)
    if isinstance(spec, VerifierSpecAlwaysPass):
        return AlwaysPass()
    if isinstance(spec, VerifierSpecAlwaysFail):
        return AlwaysFail()
    raise AssertionError(f"unhandled verifier spec {spec!r}")  # pragma: no cover


# ============================================================================
# AlwaysPass / AlwaysFail (test scaffolding)
# ============================================================================


class AlwaysPass:
    """Always passes with score 1.0."""

    async def verify(self, task: EvalTask, run: RunResult, workspace: Path) -> VerificationResult:
        return VerificationResult.new(True, 1.0, "always pass")

    def is_deterministic(self) -> bool:
        return True


class AlwaysFail:
    """Always fails with score 0.0."""

    async def verify(self, task: EvalTask, run: RunResult, workspace: Path) -> VerificationResult:
        return VerificationResult.new(False, 0.0, "always fail")

    def is_deterministic(self) -> bool:
        return True


# ============================================================================
# TestSuiteVerifier (Rule 10)
# ============================================================================


class TestSuiteVerifier:
    """Runs a command in the workspace; score = pass rate parsed from the
    output. ``passed`` = (score == 1.0). Deterministic."""

    __test__ = False  # do not collect under pytest

    def __init__(self, *, command: str, args: list[str], timeout: float) -> None:
        self.command = command
        self.args = args
        self.timeout = timeout

    async def verify(self, task: EvalTask, run: RunResult, workspace: Path) -> VerificationResult:
        sandbox = _DirectSandbox(workspace)
        out = await sandbox.execute_command(self.command, self.args, workspace, self.timeout)
        combined = f"{out.stdout}{out.stderr}"
        score = _parse_pass_rate(combined)
        if score is None:
            score = 1.0 if out.exit_code == 0 else 0.0
        passed = abs(score - 1.0) < _EPS
        result = VerificationResult.clamped(
            passed, score, f"exit={out.exit_code} pass_rate={score:.3f}"
        )
        return result.with_signal("exit_code", float(out.exit_code)).with_signal("pass_rate", score)

    def is_deterministic(self) -> bool:
        return True


class _DirectSandbox(BaseSandboxProvider):
    """Minimal sandbox that runs commands directly in a workspace dir. Used by
    :class:`TestSuiteVerifier` for the verification command only."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def validate(self, call: object) -> None:  # type: ignore[override]
        return None

    def workspace_root(self) -> Path:
        return self._root


def _parse_pass_rate(output: str) -> float | None:
    """Parse a pass-rate from common test-runner output. Returns ``None`` if no
    recognizable counts are present."""
    passed = _scan_number_before(output, " passed")
    total = _scan_number_before(output, " total")
    if total is None:
        total = _scan_number_after(output, "of ")
    if passed is not None and total is not None and total > 0.0:
        return min(max(passed / total, 0.0), 1.0)
    return None


def _scan_number_before(s: str, suffix: str) -> float | None:
    idx = s.find(suffix)
    if idx < 0:
        return None
    head = s[:idx]
    digits = ""
    for ch in reversed(head):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


def _scan_number_after(s: str, prefix: str) -> float | None:
    idx = s.find(prefix)
    if idx < 0:
        return None
    tail = s[idx + len(prefix) :]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


# ============================================================================
# CompositeVerifier (Rule 11)
# ============================================================================


class CompositeVerifier:
    """Combines children by weight: score = weighted mean; passed = all
    required children passed; ``is_deterministic`` = AND of children."""

    def __init__(self, children: list[tuple[TaskVerifier, float, bool]]) -> None:
        self._children = children

    async def verify(self, task: EvalTask, run: RunResult, workspace: Path) -> VerificationResult:
        weighted_sum = 0.0
        weight_total = 0.0
        all_required_passed = True
        details: list[str] = []
        for verifier, weight, required in self._children:
            r = await verifier.verify(task, run, workspace)
            weighted_sum += r.score * weight
            weight_total += weight
            if required and not r.passed:
                all_required_passed = False
            details.append(f"[w={weight} req={required} pass={r.passed} score={r.score:.3f}]")
        score = weighted_sum / weight_total if weight_total > 0.0 else 0.0
        return VerificationResult.clamped(all_required_passed, score, " ".join(details))

    def is_deterministic(self) -> bool:
        return all(v.is_deterministic() for v, _, _ in self._children)


# ============================================================================
# MetricEvaluatorVerifier (Rule 12)
# ============================================================================


class MetricEvaluatorVerifier:
    """Wraps a ``MetricEvaluator``: runs ``evaluate``, normalizes the value to
    ``[0,1]`` per ``direction()`` and the configured min/max (or a threshold).
    Deterministic unless explicitly marked otherwise."""

    def __init__(
        self,
        evaluator: MetricEvaluator,
        *,
        min: float | None = None,
        max: float | None = None,
        threshold: float | None = None,
        deterministic: bool = True,
    ) -> None:
        self._evaluator = evaluator
        self._min = min
        self._max = max
        self._threshold = threshold
        self._deterministic = deterministic

    @classmethod
    def with_range(
        cls, evaluator: MetricEvaluator, min: float, max: float
    ) -> MetricEvaluatorVerifier:
        return cls(evaluator, min=min, max=max)

    @classmethod
    def with_threshold(
        cls, evaluator: MetricEvaluator, threshold: float
    ) -> MetricEvaluatorVerifier:
        return cls(evaluator, threshold=threshold)

    def non_deterministic(self) -> MetricEvaluatorVerifier:
        self._deterministic = False
        return self

    def _normalize(self, value: float, direction: OptimizationDirection) -> float:
        if self._threshold is not None:
            beats = (
                value >= self._threshold if direction == "maximize" else value <= self._threshold
            )
            return 1.0 if beats else 0.0
        if self._min is not None and self._max is not None:
            if abs(self._max - self._min) < _EPS:
                return 0.0
            unit = min(max((value - self._min) / (self._max - self._min), 0.0), 1.0)
            return unit if direction == "maximize" else 1.0 - unit
        return min(max(value, 0.0), 1.0)

    async def verify(self, task: EvalTask, run: RunResult, workspace: Path) -> VerificationResult:
        sandbox = _DirectSandbox(workspace)
        session_id = _session_id_of(run)
        snapshot = SessionStateSnapshot(
            session_id=session_id,
            task_id=TaskId(task.id),
            state=SessionState(),
            workspace_root=workspace,
        )
        result = await self._evaluator.evaluate(sandbox, snapshot)
        if not isinstance(result, MetricResult):
            raise VerifyError(f"evaluator failed: {result!r}")
        score = self._normalize(result.value, self._evaluator.direction())
        passed = score >= 1.0 - _EPS
        return VerificationResult.clamped(
            passed, score, f"metric value={result.value} normalized={score:.3f}"
        ).with_signal("metric_value", result.value)

    def is_deterministic(self) -> bool:
        return self._deterministic


class _NormalizingSuccessVerifier:
    """Placeholder for a ``metric_evaluator`` spec resolved from a manifest (no
    concrete evaluator wired). Scores from the run's success flag."""

    async def verify(self, task: EvalTask, run: RunResult, workspace: Path) -> VerificationResult:
        success = isinstance(run, RunResultSuccess)
        value = 1.0 if success else 0.0
        return VerificationResult.new(success, value, "metric-evaluator (manifest placeholder)")

    def is_deterministic(self) -> bool:
        return True


# ============================================================================
# LlmJudgeVerifier (Rule 13)
# ============================================================================

_SCORE_RE = re.compile(r"(?i)score\s*:\s*([-+]?\d+(?:\.\d+)?)")


class LlmJudgeVerifier:
    """A thin LLM-judge verifier. ``is_deterministic() == False``. The concrete
    judge :class:`ModelInterface` is injected at construction."""

    def __init__(
        self,
        *,
        judge: ModelInterface,
        rubric: str,
        score_range: tuple[float, float],
        params: ModelParams | None = None,
    ) -> None:
        self.judge = judge
        self.rubric = rubric
        self.score_range = score_range
        self.params = params or ModelParams()

    async def verify(self, task: EvalTask, run: RunResult, workspace: Path) -> VerificationResult:
        output = run.output if isinstance(run, RunResultSuccess) else ""
        prompt = (
            f"{self.rubric}\n\nAgent output to evaluate:\n{output}\n\n"
            f"Reply with a single line `score: <number>` within {self.score_range}."
        )
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=TextContent(text=prompt))],
            tools=[],
            params=self.params,
            stream=False,
        )
        try:
            response = await self.judge.call(request)
        except Exception as e:  # noqa: BLE001 — surface judge failure as VerifyError
            raise VerifyError(f"judge call failed: {e}") from e
        text = "\n".join(b.text for b in response.content if isinstance(b, TextBlock))
        m = _SCORE_RE.search(text)
        if m is None:
            raise VerifyError(f"no score in judge reply: {text!r}")
        raw = float(m.group(1))
        lo, hi = self.score_range
        if hi <= lo:
            raise VerifyError(f"invalid score_range ({lo},{hi})")
        score = min(max((min(max(raw, lo), hi) - lo) / (hi - lo), 0.0), 1.0)
        return VerificationResult.new(score >= 0.5, score, f"judge score={raw}")

    def is_deterministic(self) -> bool:
        return False


class _StubLlmJudgeVerifier:
    """Stub LLM judge used when a manifest's ``llm_judge`` spec is resolved
    without an injected model. Non-deterministic."""

    def __init__(self, *, score_range: tuple[float, float]) -> None:
        self.score_range = score_range

    async def verify(self, task: EvalTask, run: RunResult, workspace: Path) -> VerificationResult:
        success = isinstance(run, RunResultSuccess)
        score = 1.0 if success else 0.0
        return VerificationResult.new(success, score, "llm-judge (manifest stub)")

    def is_deterministic(self) -> bool:
        return False


def _session_id_of(run: RunResult) -> SessionId:
    if isinstance(run, RunResultSuccess):
        return run.session_id
    state = getattr(run, "state", None)
    if state is not None and getattr(state, "session_id", None) is not None:
        return state.session_id
    return getattr(run, "session_id", SessionId(""))
