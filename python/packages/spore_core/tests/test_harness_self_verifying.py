"""Tests for the SelfVerifying loop strategy (issue #61).

Mirrors ``rust/crates/spore-core/src/harness.rs`` SelfVerifying unit tests and
the shared fixture at ``fixtures/harness/self_verifying.json``. Each test maps
to one rule (R1-R12) from the issue spec; the rule lives in the docstring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetLimits,
    FinalResponse,
    HaltReasonSelfVerifyExhausted,
    HarnessConfig,
    HarnessRunOptions,
    SelfVerifyingConfig,
    MockAgent,
    NoopContextManager,
    ReadOnlySandbox,
    RunResult,
    RunResultFailure,
    RunResultSuccess,
    SandboxReadOnlyViolation,
    SandboxViolation,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardHarness,
    Task,
    ToolCall,
    ToolCallRequested,
    TokenUsage,
    VerifierInput,
    VerifierVerdict,
    VerifierVerdictFailed,
    VerifierVerdictPassed,
)
from spore_core.harness import BaseSandboxProvider
from spore_core.prompt_assembly import InMemoryChunkProvider, PromptChunk

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ScriptedVerifier:
    """Pops scripted verdicts in order; runs out -> Failed (defensive)."""

    def __init__(self, verdicts: list[VerifierVerdict], max_iterations: int = 3) -> None:
        self._verdicts = list(verdicts)
        self._max_iterations = max_iterations
        self.calls: list[VerifierInput] = []

    async def verify(self, input: VerifierInput) -> VerifierVerdict:
        self.calls.append(input)
        if not self._verdicts:
            return VerifierVerdictFailed(reason="scripted verifier exhausted")
        return self._verdicts.pop(0)

    def max_iterations(self) -> int:
        return self._max_iterations


def _passed() -> VerifierVerdictPassed:
    return VerifierVerdictPassed()


def _failed(reason: str) -> VerifierVerdictFailed:
    return VerifierVerdictFailed(reason=reason)


class RecordingContextManager:
    """NoopContextManager that records every user message text, so tests can
    assert injected verdict reasons reach the build context (R6)."""

    def __init__(self) -> None:
        self.user_messages: list[str] = []

    async def assemble(self, session: SessionState, task: Task) -> Any:
        from spore_core.agent import Context
        from spore_core.model import ModelParams

        return Context(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: Any) -> None:
        return None

    async def append_user_message(self, session: SessionState, text: str) -> None:
        self.user_messages.append(text)

    def should_compact(self, session: SessionState) -> bool:
        return False


class RecordingSandbox(BaseSandboxProvider):
    """Allow-all sandbox that records every validated tool-call name, so a test
    can confirm the BUILD sandbox is never asked to validate a write that the
    evaluate (read-only) sandbox would have blocked."""

    def __init__(self) -> None:
        self.validated: list[str] = []

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        self.validated.append(call.name)
        return None

    def workspace_root(self) -> Path:
        return Path("/build-ws")


def _build_agent() -> MockAgent:
    """An agent that always claims done on its first turn (R1)."""
    a = MockAgent(AgentId("build"))
    for _ in range(20):
        a.push(
            FinalResponse(content="build done", usage=TokenUsage(input_tokens=1, output_tokens=1))
        )
    return a


def _eval_agent() -> MockAgent:
    a = MockAgent(AgentId("eval"))
    for _ in range(20):
        a.push(
            FinalResponse(content="eval done", usage=TokenUsage(input_tokens=1, output_tokens=1))
        )
    return a


def _config(
    *,
    agent: MockAgent,
    verifier: Any = None,
    evaluator_agent: MockAgent | None = None,
    sandbox: Any = None,
    context_manager: Any = None,
    chunk_provider: Any = None,
) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=sandbox if sandbox is not None else AllowAllSandbox(),
        context_manager=context_manager if context_manager is not None else NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        verifier=verifier,
        evaluator_agent=evaluator_agent,
        chunk_provider=chunk_provider,
    )


def _task(max_turns: int | None = None) -> Task:
    return Task.new(
        "implement the thing",
        SessionId("build-session"),
        SelfVerifyingConfig.simple(),
        budget=BudgetLimits(max_turns=max_turns),
    )


# ---------------------------------------------------------------------------
# R10: SelfVerifying no longer returns StrategyNotYetImplemented.
# ---------------------------------------------------------------------------


async def test_r10_self_verifying_not_strategy_not_yet_implemented() -> None:
    v = ScriptedVerifier([_passed()])
    h = StandardHarness(_config(agent=_build_agent(), verifier=v, evaluator_agent=_eval_agent()))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)


# ---------------------------------------------------------------------------
# R11: verifier is None -> Failure(SelfVerifyMisconfigured), NOT a raise.
# ---------------------------------------------------------------------------


async def test_r11_no_verifier_is_misconfigured() -> None:
    # #124: an unresolved verifier (the SelfVerifying ``evaluator`` key resolves
    # against the verifier map) is a STARTUP ConfigurationError (single resolution
    # path), NOT the legacy SelfVerifyMisconfigured. Mirrors Rust's
    # ``self_verifying_missing_verifier_is_typed_halt``.
    from spore_core import HaltReasonConfigurationError, HarnessErrorUnresolvedHandle

    h = StandardHarness(_config(agent=_build_agent(), verifier=None))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonConfigurationError)
    assert r.reason.error == HarnessErrorUnresolvedHandle(handle_kind="verifier", key="")
    assert r.session_id == SessionId("build-session")


# ---------------------------------------------------------------------------
# R1: the build loop runs to agent-done before the verdict is taken.
# ---------------------------------------------------------------------------


async def test_r1_build_runs_to_agent_done() -> None:
    v = ScriptedVerifier([_passed()])
    build = _build_agent()
    h = StandardHarness(_config(agent=build, verifier=v, evaluator_agent=_eval_agent()))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    # The verifier saw a Success build_result (build claimed done).
    assert isinstance(v.calls[0].build_result, RunResultSuccess)
    assert v.calls[0].build_result.output == "build done"


# ---------------------------------------------------------------------------
# R2 / R9: evaluate uses a FRESH session id, distinct from the build session.
# ---------------------------------------------------------------------------


async def test_r2_r9_evaluate_session_is_fresh_and_distinct() -> None:
    v = ScriptedVerifier([_passed()])
    h = StandardHarness(_config(agent=_build_agent(), verifier=v, evaluator_agent=_eval_agent()))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    inp = v.calls[0]
    assert isinstance(inp.build_result, RunResultSuccess)
    assert isinstance(inp.eval_result, RunResultSuccess)
    build_sid = inp.build_result.session_id
    eval_sid = inp.eval_result.session_id
    assert build_sid == SessionId("build-session")
    assert eval_sid != build_sid


# ---------------------------------------------------------------------------
# R3: evaluate uses a read-only sandbox (write -> ReadOnlyViolation); the build
# sandbox is unaffected. Tested both via the decorator directly and end-to-end.
# ---------------------------------------------------------------------------


async def test_r3_read_only_sandbox_blocks_writes_delegates_reads() -> None:
    inner = RecordingSandbox()
    ro = ReadOnlySandbox(inner)
    # Every mutating tool name is rejected with ReadOnlyViolation.
    for name in ReadOnlySandbox.DEFAULT_WRITE_TOOLS:
        v = await ro.validate(ToolCall(id="c", name=name, input={}))
        assert isinstance(v, SandboxReadOnlyViolation)
        assert v.path == name
    # The inner sandbox was never consulted for a blocked write.
    assert inner.validated == []
    # A read tool delegates to the inner sandbox.
    v = await ro.validate(ToolCall(id="c", name="read_file", input={}))
    assert v is None
    assert inner.validated == ["read_file"]


async def test_r3_evaluate_phase_write_is_blocked_build_unaffected() -> None:
    # Build sandbox records validations; the evaluator scripts a write_file call
    # that the read-only wrapper must reject before the build sandbox sees it.
    build_sandbox = RecordingSandbox()
    evaluator = MockAgent(AgentId("eval"))
    evaluator.push(
        ToolCallRequested(
            calls=[ToolCall(id="w", name="write_file", input={"path": "x"})],
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )
    )
    evaluator.push(
        FinalResponse(content="eval done", usage=TokenUsage(input_tokens=1, output_tokens=1))
    )
    v = ScriptedVerifier([_passed()])
    h = StandardHarness(
        _config(
            agent=_build_agent(),
            verifier=v,
            evaluator_agent=evaluator,
            sandbox=build_sandbox,
        )
    )
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    # The write_file was attempted in the evaluate phase but the build sandbox
    # never validated it (the read-only wrapper short-circuited it).
    assert "write_file" not in build_sandbox.validated


# ---------------------------------------------------------------------------
# R4: the role-evaluator chunk is present in the evaluate seed (presence-only).
# ---------------------------------------------------------------------------


async def test_r4_role_evaluator_chunk_present_in_evaluate_seed() -> None:
    chunk = PromptChunk("role-evaluator", "You are a fresh evaluator. You did not write this code.")
    provider = InMemoryChunkProvider([chunk])
    # Record the evaluate seed by giving the EVALUATOR its own recording context
    # manager via a shared config (the evaluate phase clones the config, so the
    # same recording manager is used for both build and evaluate).
    rec = RecordingContextManager()
    v = ScriptedVerifier([_passed()])
    h = StandardHarness(
        _config(
            agent=_build_agent(),
            verifier=v,
            evaluator_agent=_eval_agent(),
            context_manager=rec,
            chunk_provider=provider,
        )
    )
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    # Some user message seeded into the evaluate phase carries the chunk content.
    assert any("fresh evaluator" in m for m in rec.user_messages)


# ---------------------------------------------------------------------------
# R5: a Default-FAIL indeterminate evaluator verdict keeps looping (-> exhaust).
# ---------------------------------------------------------------------------


async def test_r5_indeterminate_verdict_keeps_looping() -> None:
    v = ScriptedVerifier([_failed("indeterminate"), _failed("indeterminate")], max_iterations=2)
    h = StandardHarness(_config(agent=_build_agent(), verifier=v, evaluator_agent=_eval_agent()))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonSelfVerifyExhausted)
    assert r.reason.iterations == 2
    assert len(v.calls) == 2


# ---------------------------------------------------------------------------
# R6: fail iter0 (reason X) / pass iter1 -> iter1 build context contains X,
# final Success.
# ---------------------------------------------------------------------------


async def test_r6_fail_then_pass_injects_reason_and_succeeds() -> None:
    rec = RecordingContextManager()
    v = ScriptedVerifier([_failed("needs a null check"), _passed()])
    h = StandardHarness(
        _config(
            agent=_build_agent(),
            verifier=v,
            evaluator_agent=_eval_agent(),
            context_manager=rec,
        )
    )
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    # The iter0 failure reason was injected into the build context for iter1.
    assert "needs a null check" in rec.user_messages
    assert len(v.calls) == 2


# ---------------------------------------------------------------------------
# R7: always-Fail verifier -> exactly max_iterations cycles -> Exhausted.
# ---------------------------------------------------------------------------


async def test_r7_always_fail_exhausts_at_max_iterations() -> None:
    v = ScriptedVerifier(
        [_failed("nope"), _failed("nope"), _failed("still nope")], max_iterations=3
    )
    h = StandardHarness(_config(agent=_build_agent(), verifier=v, evaluator_agent=_eval_agent()))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonSelfVerifyExhausted)
    assert r.reason.iterations == 3
    assert r.reason.last_reason == "still nope"
    assert len(v.calls) == 3


# ---------------------------------------------------------------------------
# R8: budgets fold BOTH phases across all iterations.
# ---------------------------------------------------------------------------


async def test_r8_budgets_fold_both_phases() -> None:
    # 2 iterations; each iteration runs one build turn + one evaluate turn, each
    # consuming (1 in, 1 out). 2 iterations * 2 phases = 4 turns of usage.
    v = ScriptedVerifier([_failed("again"), _passed()])
    h = StandardHarness(_config(agent=_build_agent(), verifier=v, evaluator_agent=_eval_agent()))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    # 2 build turns + 2 evaluate turns, each (1,1).
    assert r.usage.input_tokens == 4
    assert r.usage.output_tokens == 4


# ---------------------------------------------------------------------------
# R9: build vs evaluate distinguishable (distinct session ids), shared-agent
# default path also keeps them distinct.
# ---------------------------------------------------------------------------


async def test_r9_build_and_evaluate_distinct_with_default_agent() -> None:
    # No evaluator_agent: the evaluate phase defaults to config.agent (D2). The
    # session ids must still differ so traces are distinguishable.
    v = ScriptedVerifier([_passed()])
    shared = _build_agent()  # enough responses for build + evaluate
    h = StandardHarness(_config(agent=shared, verifier=v, evaluator_agent=None))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    inp = v.calls[0]
    assert isinstance(inp.build_result, RunResultSuccess)
    assert isinstance(inp.eval_result, RunResultSuccess)
    assert inp.build_result.session_id != inp.eval_result.session_id


# ---------------------------------------------------------------------------
# R12: fixture replay of fixtures/harness/self_verifying.json.
# ---------------------------------------------------------------------------


def _fixture_path() -> Path:
    here = Path(__file__).resolve()
    return here.parents[4] / "fixtures" / "harness" / "self_verifying.json"


def _verdicts_from_fixture(raw: list[str]) -> list[VerifierVerdict]:
    return [(_passed() if s == "pass" else _failed(s)) for s in raw]


async def test_r12_fixture_replay() -> None:
    suite = json.loads(_fixture_path().read_text())
    for case in suite["cases"]:
        name = case["name"]
        verdicts = _verdicts_from_fixture(case["verdicts"])
        max_iter = case["max_iterations"]
        expected = case["expected"]

        if expected["kind"] == "misconfigured":
            # #124: an unresolved verifier is now a startup ConfigurationError
            # (single resolution path). The fixture file is unchanged — only the
            # expected runtime mapping. Mirrors Rust's fixture-replay arm.
            from spore_core import HaltReasonConfigurationError, HarnessErrorUnresolvedHandle

            h = StandardHarness(_config(agent=_build_agent(), verifier=None))
            r: RunResult = await h.run(HarnessRunOptions(_task()))
            assert isinstance(r, RunResultFailure), name
            assert isinstance(r.reason, HaltReasonConfigurationError), name
            assert isinstance(r.reason.error, HarnessErrorUnresolvedHandle), name
            continue

        v = ScriptedVerifier(verdicts, max_iterations=max_iter)
        h = StandardHarness(
            _config(agent=_build_agent(), verifier=v, evaluator_agent=_eval_agent())
        )
        r = await h.run(HarnessRunOptions(_task()))

        if expected["kind"] == "success":
            assert isinstance(r, RunResultSuccess), name
            assert len(v.calls) == expected["iterations"], name
        elif expected["kind"] == "exhausted":
            assert isinstance(r, RunResultFailure), name
            assert isinstance(r.reason, HaltReasonSelfVerifyExhausted), name
            assert r.reason.iterations == expected["iterations"], name
            assert len(v.calls) == expected["iterations"], name
        else:  # pragma: no cover
            raise AssertionError(f"unknown expected kind in {name}")


# ---------------------------------------------------------------------------
# ReadOnlySandbox: command execution and write/execute path resolution rejected.
# ---------------------------------------------------------------------------


async def test_read_only_sandbox_forbids_exec_and_write_paths() -> None:
    from spore_core.sandbox import SandboxViolationException

    ro = ReadOnlySandbox(RecordingSandbox())
    raised = False
    try:
        await ro.execute_command("ls", [])
    except SandboxViolationException as e:
        raised = True
        assert isinstance(e.violation, SandboxReadOnlyViolation)
    assert raised

    raised = False
    try:
        await ro.resolve_path("x", "write")
    except SandboxViolationException as e:
        raised = True
        assert isinstance(e.violation, SandboxReadOnlyViolation)
    assert raised

    # A read resolve delegates to the inner sandbox (returns the path).
    p = await ro.resolve_path("x", "read")
    assert isinstance(p, Path)
