"""Tests for :mod:`spore_core.verifier` — issue #44."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from spore_core.harness import (
    AggregateUsage,
    CommandOutput,
    HaltReasonStrategyNotYetImplemented,
    RunResult,
    RunResultFailure,
    RunResultSuccess,
    SessionId,
)
from spore_core.model import ToolCall
from spore_core.verifier import (
    CompositeVerifier,
    EvaluatorResponseVerifier,
    TestSuiteVerifier,
    Verifier,
    VerifierInput,
    VerifierVerdictFailed,
    VerifierVerdictPassed,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _success(output: str) -> RunResultSuccess:
    return RunResultSuccess(
        output=output,
        session_id=SessionId("s"),
        usage=AggregateUsage(),
        turns=1,
    )


def _failure() -> RunResultFailure:
    return RunResultFailure(
        reason=HaltReasonStrategyNotYetImplemented(strategy="x"),
        session_id=SessionId("s"),
        usage=AggregateUsage(),
        turns=0,
    )


def _input_with(build: RunResult, eval_: RunResult) -> VerifierInput:
    return VerifierInput(
        build_result=build,
        eval_result=eval_,
        workspace=Path("/tmp"),
        iteration=0,
    )


def _make_resp_verifier() -> EvaluatorResponseVerifier:
    return EvaluatorResponseVerifier(r"(?i)\bPASS\b", r"(?i)\bFAIL: .+", 3)


# ── EvaluatorResponseVerifier ──────────────────────────────────────────────


async def test_response_verifier_pass_pattern_matches() -> None:
    v = _make_resp_verifier()
    i = _input_with(_success("ok"), _success("all checks PASS, ready to ship"))
    assert isinstance(await v.verify(i), VerifierVerdictPassed)


async def test_response_verifier_fail_pattern_matches_with_reason() -> None:
    v = _make_resp_verifier()
    i = _input_with(
        _success("ok"),
        _success("FAIL: missing edge case in handler.rs"),
    )
    verdict = await v.verify(i)
    assert isinstance(verdict, VerifierVerdictFailed)
    assert "missing edge case" in verdict.reason


async def test_response_verifier_neither_pattern_default_fails() -> None:
    v = _make_resp_verifier()
    i = _input_with(_success("ok"), _success("indeterminate output"))
    verdict = await v.verify(i)
    assert isinstance(verdict, VerifierVerdictFailed)
    assert "matched neither" in verdict.reason
    assert "indeterminate output" in verdict.reason


async def test_response_verifier_build_failure_propagates() -> None:
    v = _make_resp_verifier()
    i = _input_with(_failure(), _success("PASS"))
    verdict = await v.verify(i)
    assert isinstance(verdict, VerifierVerdictFailed)
    assert verdict.reason.startswith("build run halted")


async def test_response_verifier_eval_failure_propagates() -> None:
    v = _make_resp_verifier()
    i = _input_with(_success("ok"), _failure())
    verdict = await v.verify(i)
    assert isinstance(verdict, VerifierVerdictFailed)
    assert verdict.reason.startswith("evaluator run halted")


async def test_response_verifier_default_max_iterations_overrideable() -> None:
    v = _make_resp_verifier()
    assert v.max_iterations() == 3
    v2 = EvaluatorResponseVerifier("a", "b", 10)
    assert v2.max_iterations() == 10


# ── TestSuiteVerifier ──────────────────────────────────────────────────────


@dataclass
class _StubSandbox:
    out: CommandOutput
    root: Path

    async def validate(self, call: ToolCall):  # type: ignore[no-untyped-def]
        return None

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        return self.out

    async def handle_large_output(  # pragma: no cover — unused here
        self, content: str, call_id: str, head_tokens: int, tail_tokens: int
    ):
        raise NotImplementedError

    async def resolve_path(self, path: str, operation: str = "read") -> Path:
        return Path(path)

    def isolation_mode(self):  # type: ignore[no-untyped-def]
        from spore_core.dangerous import IsolationModeNone

        return IsolationModeNone()

    def workspace_root(self) -> Path:
        return self.root


def _stub_sandbox(exit_code: int, stderr: str) -> _StubSandbox:
    return _StubSandbox(
        out=CommandOutput(
            stdout="",
            stderr=stderr,
            exit_code=exit_code,
            timed_out=False,
            truncated=False,
        ),
        root=Path("/"),
    )


async def test_test_suite_verifier_pass() -> None:
    v = TestSuiteVerifier(
        "cargo test",
        Path("/work"),
        60.0,
        _stub_sandbox(0, ""),
        3,
    )
    i = _input_with(_success("ok"), _success(""))
    assert isinstance(await v.verify(i), VerifierVerdictPassed)


async def test_test_suite_verifier_fail_includes_stderr() -> None:
    v = TestSuiteVerifier(
        "cargo test",
        Path("/work"),
        60.0,
        _stub_sandbox(1, "test foo ... FAILED"),
        3,
    )
    i = _input_with(_success("ok"), _success(""))
    verdict = await v.verify(i)
    assert isinstance(verdict, VerifierVerdictFailed)
    assert "FAILED" in verdict.reason


async def test_test_suite_verifier_build_failure_short_circuits() -> None:
    v = TestSuiteVerifier(
        "cargo test",
        Path("/work"),
        60.0,
        _stub_sandbox(0, ""),
        3,
    )
    i = _input_with(_failure(), _success(""))
    verdict = await v.verify(i)
    assert isinstance(verdict, VerifierVerdictFailed)
    assert verdict.reason.startswith("build run halted")


async def test_test_suite_verifier_empty_command_fails() -> None:
    v = TestSuiteVerifier("", Path("/work"), 60.0, _stub_sandbox(0, ""), 3)
    i = _input_with(_success("ok"), _success(""))
    verdict = await v.verify(i)
    assert isinstance(verdict, VerifierVerdictFailed)
    assert "empty test command" in verdict.reason


# ── CompositeVerifier ──────────────────────────────────────────────────────


@dataclass
class _FixedVerifier:
    verdict: VerifierVerdictPassed | VerifierVerdictFailed

    async def verify(self, input: VerifierInput):  # type: ignore[no-untyped-def]
        return self.verdict

    def max_iterations(self) -> int:
        return 3


def _pass_v() -> Verifier:
    return _FixedVerifier(VerifierVerdictPassed())


def _fail_v(reason: str) -> Verifier:
    return _FixedVerifier(VerifierVerdictFailed(reason=reason))


async def test_composite_all_pass_returns_passed() -> None:
    c = CompositeVerifier([_pass_v(), _pass_v(), _pass_v()], 3)
    i = _input_with(_success("ok"), _success("ok"))
    assert isinstance(await c.verify(i), VerifierVerdictPassed)


async def test_composite_one_fail_returns_that_reason() -> None:
    c = CompositeVerifier([_pass_v(), _fail_v("oops"), _pass_v()], 3)
    i = _input_with(_success("ok"), _success("ok"))
    verdict = await c.verify(i)
    assert isinstance(verdict, VerifierVerdictFailed)
    assert "oops" in verdict.reason
    assert "[verifier 1]" in verdict.reason


async def test_composite_many_fails_concatenated() -> None:
    c = CompositeVerifier(
        [_fail_v("first"), _pass_v(), _fail_v("second"), _fail_v("third")],
        3,
    )
    i = _input_with(_success("ok"), _success("ok"))
    verdict = await c.verify(i)
    assert isinstance(verdict, VerifierVerdictFailed)
    assert "first" in verdict.reason
    assert "second" in verdict.reason
    assert "third" in verdict.reason
    assert "[verifier 1]" not in verdict.reason


async def test_composite_truncates_at_2000_chars() -> None:
    long = "x" * 5000
    c = CompositeVerifier([_fail_v(long)], 3)
    i = _input_with(_success("ok"), _success("ok"))
    verdict = await c.verify(i)
    assert isinstance(verdict, VerifierVerdictFailed)
    assert len(verdict.reason) <= 2000 + len("... [truncated]")
    assert verdict.reason.endswith("... [truncated]")


def test_verifier_protocol_satisfied() -> None:
    assert isinstance(EvaluatorResponseVerifier("PASS", "FAIL", 3), Verifier)
    assert isinstance(CompositeVerifier([], 3), Verifier)


# ── Cross-language fixture replay ──────────────────────────────────────────


_FIXTURE_PATH = (
    Path(__file__).resolve().parents[4] / "fixtures" / "verifier" / "evaluator_response.json"
)


@pytest.mark.asyncio
async def test_fixture_replay_evaluator_response_verifier() -> None:
    raw = _FIXTURE_PATH.read_text()
    suite = json.loads(raw)
    run_result_adapter: TypeAdapter[RunResult] = TypeAdapter(RunResult)
    for case in suite["cases"]:
        name = case["name"]
        v = EvaluatorResponseVerifier(
            case["pass_pattern"],
            case["fail_pattern"],
            3,
        )
        build = run_result_adapter.validate_python(case["build_result"])
        evalr = run_result_adapter.validate_python(case["eval_result"])
        i = VerifierInput(
            build_result=build,
            eval_result=evalr,
            workspace=Path("/fixture"),
            iteration=0,
        )
        verdict = await v.verify(i)
        expected = case["expected"]
        if expected["kind"] == "passed":
            assert isinstance(verdict, VerifierVerdictPassed), (
                f"case `{name}`: expected Passed, got {verdict!r}"
            )
        else:
            assert isinstance(verdict, VerifierVerdictFailed), (
                f"case `{name}`: expected Failed, got {verdict!r}"
            )
            assert expected["contains"] in verdict.reason, (
                f"case `{name}`: expected reason to contain "
                f"`{expected['contains']}`, got `{verdict.reason}`"
            )
