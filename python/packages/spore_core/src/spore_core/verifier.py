"""Verifier — the oracle for the ``SelfVerifying`` loop strategy (issue #44).

Mirrors the Rust reference at ``rust/crates/spore-core/src/verifier.rs``.

The :class:`Verifier` sits between an evaluator harness's :class:`RunResult`
and the build loop's halt decision. It translates ``(build_result,
eval_result)`` into a :class:`VerifierVerdict` — either :class:`Passed`
(halt with success) or :class:`Failed` (re-enter the build loop with
``reason`` injected into the next turn's context).

Ambiguity resolutions (see issue #44 comment thread):

1. :class:`EvaluatorResponseVerifier` when neither ``pass_pattern`` nor
   ``fail_pattern`` matches → ``Failed`` with a descriptive reason
   including a truncated copy of the output. Default-FAIL is **not**
   configurable.
2. Any non-:class:`RunResultSuccess` :class:`RunResult` in
   ``build_result`` or ``eval_result`` → ``Failed``.
   :class:`RunResultWaitingForHuman` is treated as a misconfiguration
   signal and surfaced in the reason.
3. :class:`CompositeVerifier` concatenates all child failure reasons
   (joined by ``"\\n"``), capped at 2000 characters total. Children that
   pass are not mentioned.
4. ``LoopStrategy.self_verifying`` wiring is **deferred** — bundled with
   the Ralph wiring and #45. The strategy continues to return
   ``StrategyNotYetImplemented`` in the harness loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .harness import (
    RunResult,
    RunResultFailure,
    RunResultSuccess,
    RunResultWaitingForHuman,
    SandboxProvider,
)

# ============================================================================
# VerifierVerdict
# ============================================================================


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class VerifierVerdictPassed(_Model):
    kind: Literal["passed"] = "passed"


class VerifierVerdictFailed(_Model):
    kind: Literal["failed"] = "failed"
    reason: str


VerifierVerdict = Annotated[
    VerifierVerdictPassed | VerifierVerdictFailed,
    Field(discriminator="kind"),
]


def failed(reason: str) -> VerifierVerdictFailed:
    """Convenience constructor for a :class:`VerifierVerdictFailed`."""
    return VerifierVerdictFailed(reason=reason)


# ============================================================================
# VerifierInput
# ============================================================================


@dataclass
class VerifierInput:
    build_result: RunResult
    eval_result: RunResult
    workspace: Path
    # Which build-evaluate cycle this is (0-indexed).
    iteration: int = 0


# ============================================================================
# Verifier protocol
# ============================================================================


@runtime_checkable
class Verifier(Protocol):
    """Translates ``(build_result, eval_result)`` into a verdict.

    ``max_iterations`` is the maximum number of build-evaluate cycles
    before the harness halts the loop regardless of verdict; the spec
    default is 3. Implementations expose it as a method to match the
    Rust trait surface.
    """

    async def verify(self, input: VerifierInput) -> VerifierVerdict: ...

    def max_iterations(self) -> int: ...


# ============================================================================
# Helpers
# ============================================================================


_COMPOSITE_REASON_CAP = 2000


def _view(label: str, r: RunResult) -> tuple[str, None] | tuple[None, str]:
    """Reduce a :class:`RunResult` to either (output, None) on Success or
    (None, failure_reason) otherwise."""
    if isinstance(r, RunResultSuccess):
        return (r.output, None)
    if isinstance(r, RunResultFailure):
        return (None, f"{label} run halted: {_describe_halt(r.reason)}")
    if isinstance(r, RunResultWaitingForHuman):
        return (
            None,
            f"{label} run is WaitingForHuman — verifier received a paused "
            f"harness; this is a misconfiguration signal (the {label} should "
            f"run to completion before being verified)",
        )
    # Defensive: shouldn't happen given the discriminated union.
    return (None, f"{label} run had unexpected result kind")


def _describe_halt(reason: object) -> str:
    """HaltReason is opaque diagnostic text; mirror Rust's `Debug` fallback."""
    if isinstance(reason, BaseModel):
        return reason.model_dump_json()
    return repr(reason)


def _truncate_for_reason(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "... [truncated]"


def _tail_lines(s: str, n: int) -> str:
    lines = s.splitlines()
    start = max(0, len(lines) - n)
    return "\n".join(lines[start:])


# ============================================================================
# EvaluatorResponseVerifier
# ============================================================================


class EvaluatorResponseVerifier:
    """Pattern-matches the evaluator harness's final text response. The
    simplest verifier — trusts whatever the evaluator wrote.

    Rules:

    * If ``build_result`` is not :class:`RunResultSuccess` → ``Failed``
      with the halt reason.
    * If ``eval_result`` is not :class:`RunResultSuccess` → ``Failed``
      with the halt reason.
    * If ``pass_pattern`` matches the eval output → ``Passed``.
    * If ``fail_pattern`` matches the eval output → ``Failed`` with the
      matched substring as the reason.
    * Neither matches → ``Failed`` with a descriptive default reason
      (Default-FAIL contract; not configurable).
    """

    def __init__(
        self,
        pass_pattern: str,
        fail_pattern: str,
        max_iterations: int = 3,
    ) -> None:
        self._pass_pattern_src = pass_pattern
        self._fail_pattern_src = fail_pattern
        self._pass_pattern = re.compile(pass_pattern)
        self._fail_pattern = re.compile(fail_pattern)
        self._max_iterations = max_iterations

    @property
    def pass_pattern(self) -> re.Pattern[str]:
        return self._pass_pattern

    @property
    def fail_pattern(self) -> re.Pattern[str]:
        return self._fail_pattern

    async def verify(self, input: VerifierInput) -> VerifierVerdict:
        _, build_failure = _view("build", input.build_result)
        if build_failure is not None:
            return failed(build_failure)

        output, eval_failure = _view("evaluator", input.eval_result)
        if eval_failure is not None:
            return failed(eval_failure)
        assert output is not None  # narrow for type checkers

        pass_m = self._pass_pattern.search(output)
        if pass_m is not None:
            return VerifierVerdictPassed()

        fail_m = self._fail_pattern.search(output)
        if fail_m is not None:
            return failed(
                f"evaluator reported failure: {_truncate_for_reason(fail_m.group(0), 500)}"
            )

        return failed(
            f"evaluator output matched neither pass_pattern "
            f"(`{self._pass_pattern_src}`) nor fail_pattern "
            f"(`{self._fail_pattern_src}`). Output was:\n"
            f"{_truncate_for_reason(output, 1000)}"
        )

    def max_iterations(self) -> int:
        return self._max_iterations


# ============================================================================
# TestSuiteVerifier
# ============================================================================


class TestSuiteVerifier:
    """Runs a test command via the injected :class:`SandboxProvider` and
    uses the exit code as the verdict. Ignores the evaluator's text
    output — ground truth is the tests.

    Rules:

    * If ``build_result`` is not :class:`RunResultSuccess` → ``Failed``
      with the halt reason.
    * Run ``command`` in ``working_dir`` via ``sandbox.execute_command``.
    * Exit 0, not timed out → ``Passed``.
    * Anything else → ``Failed`` with a stderr/stdout tail.
    """

    # Tell pytest this class is not a test collection target despite the
    # ``Test`` prefix in its name.
    __test__ = False

    def __init__(
        self,
        command: str,
        working_dir: Path,
        timeout: float,
        sandbox: SandboxProvider,
        max_iterations: int = 3,
    ) -> None:
        self.command = command
        self.working_dir = working_dir
        self.timeout = timeout
        self.sandbox = sandbox
        self._max_iterations = max_iterations

    async def verify(self, input: VerifierInput) -> VerifierVerdict:
        _, build_failure = _view("build", input.build_result)
        if build_failure is not None:
            return failed(build_failure)

        parts = self.command.split()
        if not parts:
            return failed("empty test command")
        program, *args = parts

        try:
            out = await self.sandbox.execute_command(
                program,
                args,
                working_dir=self.working_dir,
                timeout=self.timeout,
            )
        except Exception as e:  # noqa: BLE001 — sandbox is external surface
            return failed(f"sandbox refused test command: {e!r}")

        if out.exit_code == 0 and not out.timed_out:
            return VerifierVerdictPassed()

        tail = _tail_lines(out.stderr, 20)
        if not tail.strip():
            tail = _tail_lines(out.stdout, 20)
        return failed(
            f"test suite failed (exit {out.exit_code}, "
            f"timed_out={str(out.timed_out).lower()}):\n{tail}"
        )

    def max_iterations(self) -> int:
        return self._max_iterations


# ============================================================================
# CompositeVerifier
# ============================================================================


class CompositeVerifier:
    """Passes only when **all** child verifiers pass. On failure,
    concatenates every child's failure reason (joined by ``"\\n"``),
    capped at 2000 characters total. Children that pass are not mentioned
    in the failure reason.
    """

    def __init__(self, verifiers: list[Verifier], max_iterations: int = 3) -> None:
        self.verifiers = verifiers
        self._max_iterations = max_iterations

    async def verify(self, input: VerifierInput) -> VerifierVerdict:
        failures: list[str] = []
        for i, v in enumerate(self.verifiers):
            verdict = await v.verify(input)
            if isinstance(verdict, VerifierVerdictFailed):
                failures.append(f"[verifier {i}] {verdict.reason}")
        if not failures:
            return VerifierVerdictPassed()
        joined = "\n".join(failures)
        return failed(_truncate_for_reason(joined, _COMPOSITE_REASON_CAP))

    def max_iterations(self) -> int:
        return self._max_iterations


__all__ = [
    "CompositeVerifier",
    "EvaluatorResponseVerifier",
    "TestSuiteVerifier",
    "Verifier",
    "VerifierInput",
    "VerifierVerdict",
    "VerifierVerdictFailed",
    "VerifierVerdictPassed",
    "failed",
]
