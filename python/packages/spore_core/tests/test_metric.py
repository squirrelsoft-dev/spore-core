"""Tests for :mod:`spore_core.metric` — issue #23.

Mirrors the Rust unit tests in ``rust/crates/spore-core/src/metric.rs`` and
replays the cross-language fixtures under
``fixtures/metric_evaluator/``.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from spore_core.dangerous import IsolationModeNone
from spore_core.harness import (
    BaseSandboxProvider,
    CommandOutput,
    IsolationMode,
    SessionId,
    TaskId,
)
from spore_core.metric import (
    CommandMetricEvaluator,
    LatencyEvaluator,
    LlmJudgeEvaluator,
    JudgeModelConfig,
    MetricErrorCrashed,
    MetricErrorExecutionFailed,
    MetricErrorParseFailed,
    MetricErrorTimeout,
    MetricResult,
    TestPassRateEvaluator,
    iteration_status_from_error,
    parse_metric,
    should_keep,
)
from spore_core.model import (
    ModelParams,
    ModelRequest,
    ModelResponse,
    ProviderInfo,
    StopReason,
    StreamEvent,
    TextBlock,
    TokenUsage,
    ToolCall,
)
from spore_core.termination import SessionStateSnapshot


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def _snapshot() -> SessionStateSnapshot:
    return SessionStateSnapshot(session_id=SessionId("sess"), task_id=TaskId("task"))


class _FakeSandbox(BaseSandboxProvider):
    """In-process sandbox that returns a pre-scripted :class:`CommandOutput`
    and writes log files into a real temp directory."""

    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
        timed_out: bool = False,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = exit_code
        self._timed_out = timed_out
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self.calls = 0

    def __del__(self) -> None:
        try:
            self._tmp.cleanup()
        except Exception:  # noqa: BLE001
            pass

    async def validate(self, call: ToolCall) -> None:
        return None

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        self.calls += 1
        return CommandOutput(
            stdout=self._stdout,
            stderr=self._stderr,
            exit_code=self._exit_code,
            timed_out=self._timed_out,
            truncated=False,
        )

    def isolation_mode(self) -> IsolationMode:
        return IsolationModeNone()

    def workspace_root(self) -> Path:
        return self._root


class _FakeModel:
    def __init__(self, text: str) -> None:
        self._text = text

    async def call(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=[TextBlock(text=self._text)],
            usage=TokenUsage(),
            stop_reason=StopReason.END_TURN,
        )

    async def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        # Not used by LlmJudgeEvaluator but required by the protocol.
        if False:
            yield  # type: ignore[unreachable]

    def provider(self) -> ProviderInfo:
        return ProviderInfo(name="fake", model_id="fake", context_window=8000)


# ---------------------------------------------------------------------------
# should_keep
# ---------------------------------------------------------------------------


def test_should_keep_minimize_lower_is_better() -> None:
    assert should_keep(1.0, 2.0, "minimize") is True
    assert should_keep(2.0, 1.0, "minimize") is False


def test_should_keep_maximize_higher_is_better() -> None:
    assert should_keep(2.0, 1.0, "maximize") is True
    assert should_keep(1.0, 2.0, "maximize") is False


def test_should_keep_equal_is_discarded() -> None:
    assert should_keep(1.0, 1.0, "minimize") is False
    assert should_keep(1.0, 1.0, "maximize") is False


def test_should_keep_respects_min_delta() -> None:
    assert should_keep(1.5, 2.0, "minimize", min_delta=0.5) is False
    assert should_keep(1.49, 2.0, "minimize", min_delta=0.5) is True


# ---------------------------------------------------------------------------
# parse_metric
# ---------------------------------------------------------------------------


def test_parse_metric_extracts_capture_group() -> None:
    v = parse_metric("val_bpb:  3.125\nother", r"val_bpb:\s+([\d.]+)")
    assert isinstance(v, float)
    assert abs(v - 3.125) < 1e-9


def test_parse_metric_no_match_is_parse_failed() -> None:
    err = parse_metric("no metric here", r"val_bpb:\s+([\d.]+)")
    assert isinstance(err, MetricErrorParseFailed)


def test_parse_metric_unparseable_capture_is_parse_failed() -> None:
    err = parse_metric("val_bpb: oops", r"val_bpb:\s+(\S+)")
    assert isinstance(err, MetricErrorParseFailed)


def test_parse_metric_invalid_regex_is_execution_failed() -> None:
    err = parse_metric("x", "(unbalanced")
    assert isinstance(err, MetricErrorExecutionFailed)


# ---------------------------------------------------------------------------
# iteration_status_from_error
# ---------------------------------------------------------------------------


def test_iteration_status_from_error_maps_timeout() -> None:
    assert iteration_status_from_error(MetricErrorTimeout(after=1.0)) == "timeout"


def test_iteration_status_from_error_maps_others_to_crashed() -> None:
    for err in (
        MetricErrorCrashed(log="x"),
        MetricErrorExecutionFailed(reason="x"),
        MetricErrorParseFailed(output="", pattern=""),
    ):
        assert iteration_status_from_error(err) == "crashed"


# ---------------------------------------------------------------------------
# CommandMetricEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_evaluator_happy_path_writes_log_and_parses() -> None:
    sb = _FakeSandbox(stdout="val_bpb: 1.234\n")
    ev = CommandMetricEvaluator(
        command="uv",
        args=["run", "train.py"],
        metric_pattern=r"val_bpb:\s+([\d.]+)",
        timeout=60.0,
        log_output_to=Path("run.log"),
        direction="minimize",
        description="autoresearch val_bpb",
    )
    r = await ev.evaluate(sb, _snapshot())
    assert isinstance(r, MetricResult)
    assert abs(r.value - 1.234) < 1e-9
    log = (sb.workspace_root() / "run.log").read_text(encoding="utf-8")
    assert "val_bpb" in log
    assert ev.direction() == "minimize"


@pytest.mark.asyncio
async def test_command_evaluator_timeout_maps_to_timeout_error() -> None:
    sb = _FakeSandbox(timed_out=True)
    ev = CommandMetricEvaluator(
        command="x",
        args=[],
        metric_pattern=r"v:(\d+)",
        timeout=0.001,
        log_output_to=Path("run.log"),
        direction="minimize",
        description="x",
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorTimeout)


@pytest.mark.asyncio
async def test_command_evaluator_nonzero_exit_is_crashed() -> None:
    sb = _FakeSandbox(stdout="boom", exit_code=1)
    ev = CommandMetricEvaluator(
        command="x",
        args=[],
        metric_pattern=r"v:(\d+)",
        timeout=1.0,
        log_output_to=Path("run.log"),
        direction="minimize",
        description="x",
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorCrashed)


@pytest.mark.asyncio
async def test_command_evaluator_parse_failed_when_regex_doesnt_match() -> None:
    sb = _FakeSandbox(stdout="no metric")
    ev = CommandMetricEvaluator(
        command="x",
        args=[],
        metric_pattern=r"v:(\d+)",
        timeout=1.0,
        log_output_to=Path("run.log"),
        direction="minimize",
        description="x",
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorParseFailed)


@pytest.mark.asyncio
async def test_command_evaluator_invalid_regex_is_execution_failed() -> None:
    sb = _FakeSandbox(stdout="anything")
    ev = CommandMetricEvaluator(
        command="x",
        args=[],
        metric_pattern="(unbalanced",
        timeout=1.0,
        log_output_to=Path("run.log"),
        direction="minimize",
        description="x",
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorExecutionFailed)


@pytest.mark.asyncio
async def test_command_evaluator_writes_log_before_parsing_on_parse_failure() -> None:
    sb = _FakeSandbox(stdout="garbage but useful for debugging\n")
    ev = CommandMetricEvaluator(
        command="x",
        args=[],
        metric_pattern=r"v:(\d+)",
        timeout=1.0,
        log_output_to=Path("logs/run.log"),
        direction="minimize",
        description="x",
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorParseFailed)
    # Log should still be present even though parsing failed.
    log = (sb.workspace_root() / "logs" / "run.log").read_text(encoding="utf-8")
    assert "garbage" in log


# ---------------------------------------------------------------------------
# TestPassRateEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_pass_rate_evaluator_returns_fraction() -> None:
    sb = _FakeSandbox(stdout="passed 17 of 20")
    ev = TestPassRateEvaluator(
        command="pytest",
        timeout=60.0,
        pass_pattern=r"passed (\d+)",
        total_pattern=r"of (\d+)",
    )
    r = await ev.evaluate(sb, _snapshot())
    assert isinstance(r, MetricResult)
    assert abs(r.value - 0.85) < 1e-9
    assert ev.direction() == "maximize"


@pytest.mark.asyncio
async def test_test_pass_rate_evaluator_zero_total_is_parse_failed() -> None:
    sb = _FakeSandbox(stdout="passed 0 of 0")
    ev = TestPassRateEvaluator(
        command="pytest",
        timeout=60.0,
        pass_pattern=r"passed (\d+)",
        total_pattern=r"of (\d+)",
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorParseFailed)


@pytest.mark.asyncio
async def test_test_pass_rate_evaluator_timeout() -> None:
    sb = _FakeSandbox(timed_out=True)
    ev = TestPassRateEvaluator(
        command="pytest",
        timeout=1.0,
        pass_pattern=r"passed (\d+)",
        total_pattern=r"of (\d+)",
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorTimeout)


# ---------------------------------------------------------------------------
# LatencyEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_latency_evaluator_averages_runs() -> None:
    sb = _FakeSandbox(stdout="ok")
    ev = LatencyEvaluator(
        command="echo",
        args=["ok"],
        warmup_runs=1,
        measured_runs=2,
        timeout=5.0,
    )
    r = await ev.evaluate(sb, _snapshot())
    assert isinstance(r, MetricResult)
    assert r.value >= 0.0
    assert ev.direction() == "minimize"
    assert r.metadata["measured_runs"] == "2"
    assert sb.calls == 3  # 1 warmup + 2 measured


@pytest.mark.asyncio
async def test_latency_evaluator_zero_measured_runs_rejects() -> None:
    sb = _FakeSandbox()
    ev = LatencyEvaluator(
        command="x",
        warmup_runs=0,
        measured_runs=0,
        timeout=1.0,
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorExecutionFailed)


@pytest.mark.asyncio
async def test_latency_evaluator_crash_on_nonzero_exit() -> None:
    sb = _FakeSandbox(stdout="boom", exit_code=1)
    ev = LatencyEvaluator(
        command="x",
        warmup_runs=0,
        measured_runs=1,
        timeout=1.0,
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorCrashed)


@pytest.mark.asyncio
async def test_latency_evaluator_timeout() -> None:
    sb = _FakeSandbox(timed_out=True)
    ev = LatencyEvaluator(
        command="x",
        warmup_runs=0,
        measured_runs=1,
        timeout=0.001,
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorTimeout)


# ---------------------------------------------------------------------------
# LlmJudgeEvaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_judge_normalizes_score_into_unit_range() -> None:
    sb = _FakeSandbox()
    ev = LlmJudgeEvaluator(
        judge_model=JudgeModelConfig(provider="fake", model_id="judge-1", params=ModelParams()),
        rubric="rate this",
        score_range=(0.0, 10.0),
        sample_input="the answer",
        client=_FakeModel("score: 7.5"),
    )
    r = await ev.evaluate(sb, _snapshot())
    assert isinstance(r, MetricResult)
    assert abs(r.value - 0.75) < 1e-9
    assert ev.direction() == "maximize"


@pytest.mark.asyncio
async def test_llm_judge_clamps_score_above_range() -> None:
    sb = _FakeSandbox()
    ev = LlmJudgeEvaluator(
        judge_model=JudgeModelConfig(provider="fake", model_id="judge-1"),
        rubric="rate this",
        score_range=(0.0, 10.0),
        sample_input="x",
        client=_FakeModel("Score: 42"),
    )
    r = await ev.evaluate(sb, _snapshot())
    assert isinstance(r, MetricResult)
    assert abs(r.value - 1.0) < 1e-9


@pytest.mark.asyncio
async def test_llm_judge_clamps_score_below_range() -> None:
    sb = _FakeSandbox()
    ev = LlmJudgeEvaluator(
        judge_model=JudgeModelConfig(provider="fake", model_id="judge-1"),
        rubric="rate this",
        score_range=(0.0, 10.0),
        sample_input="x",
        client=_FakeModel("score: -5"),
    )
    r = await ev.evaluate(sb, _snapshot())
    assert isinstance(r, MetricResult)
    assert abs(r.value - 0.0) < 1e-9


@pytest.mark.asyncio
async def test_llm_judge_parse_failed_when_no_score() -> None:
    sb = _FakeSandbox()
    ev = LlmJudgeEvaluator(
        judge_model=JudgeModelConfig(provider="fake", model_id="judge-1"),
        rubric="rate this",
        score_range=(0.0, 10.0),
        sample_input="x",
        client=_FakeModel("no score in here"),
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorParseFailed)


@pytest.mark.asyncio
async def test_llm_judge_invalid_score_range_is_execution_failed() -> None:
    sb = _FakeSandbox()
    ev = LlmJudgeEvaluator(
        judge_model=JudgeModelConfig(provider="fake", model_id="judge-1"),
        rubric="rate this",
        score_range=(10.0, 10.0),
        sample_input="x",
        client=_FakeModel("score: 5"),
    )
    err = await ev.evaluate(sb, _snapshot())
    assert isinstance(err, MetricErrorExecutionFailed)


# ---------------------------------------------------------------------------
# Fixture replay
# ---------------------------------------------------------------------------


def _fixtures_dir() -> Path:
    # python/packages/spore_core/tests/test_metric.py
    # -> repo_root/fixtures/metric_evaluator/
    return Path(__file__).resolve().parents[4] / "fixtures" / "metric_evaluator"


def test_should_keep_fixture_replay() -> None:
    body = json.loads((_fixtures_dir() / "should_keep.json").read_text(encoding="utf-8"))
    for case in body["cases"]:
        got = should_keep(
            case["new_value"],
            case["current_best"],
            case["direction"],
            case["min_delta"],
        )
        assert got is case["expected"], f"case {case['name']} mismatched"


def test_parse_metric_fixture_replay() -> None:
    body = json.loads((_fixtures_dir() / "parse_metric.json").read_text(encoding="utf-8"))
    for case in body["cases"]:
        got = parse_metric(case["output"], case["pattern"])
        expected = case["expected"]
        if expected["kind"] == "value":
            assert isinstance(got, float), f"case {case['name']} expected value got {got!r}"
            assert abs(got - expected["value"]) < 1e-9, f"case {case['name']} value mismatch"
        elif expected["kind"] == "parse_failed":
            assert isinstance(got, MetricErrorParseFailed), (
                f"case {case['name']} expected ParseFailed got {got!r}"
            )
        else:
            raise AssertionError(f"unknown fixture expected kind: {expected['kind']!r}")
