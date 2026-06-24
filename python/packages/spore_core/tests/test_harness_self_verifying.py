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
    ScriptedMiddleware,
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

    async def assemble(self, session: SessionState, task: Task, sources: object) -> Any:
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


# ---------------------------------------------------------------------------
# #151 — eval-phase reviewer context: caller-middleware drop + eval_agent /
# eval_toolset overrides. #147 — charge the evaluator's turns against the scope.
#
# Mirrors the Rust regression tests
# ``self_verifying_eval_phase_drops_caller_middleware``,
# ``self_verifying_eval_agent_override_runs_distinct_reviewer``,
# ``self_verifying_config_eval_overrides_serde`` (#151) and
# ``self_verifying_charges_evaluator_turns_against_scope`` (#147) in
# ``rust/crates/spore-core/src/harness.rs``.
# ---------------------------------------------------------------------------


# BLOCKER FIX (#151): the evaluate phase must NOT inherit the caller's approval
# middleware. A nested review run is non-interactive (the caller never sees its
# generated session id), so a ``SurfaceToHuman`` BeforeTool decision would pause
# it with no human able to resume — the reviewer's first tool call would never
# dispatch and the review half would silently never run. With the caller
# middleware dropped for the read-only eval phase, the reviewer's read dispatches
# and the run reaches a verdict.
async def test_self_verifying_eval_phase_drops_caller_middleware() -> None:
    from spore_core import (
        HumanRequestToolApproval,
        MiddlewareSurfaceToHuman,
    )
    from spore_core.middleware import HookPoint

    # Build emits a tool-free FinalResponse (so the BUILD phase, which DOES run
    # the caller middleware, never trips it); the eval phase (same worker per
    # Q1c) then issues a read — the call the middleware would have gated.
    worker = MockAgent(AgentId("build"))
    worker.push(FinalResponse(content="built", usage=TokenUsage(input_tokens=1, output_tokens=1)))
    worker.push(
        ToolCallRequested(
            calls=[ToolCall(id="c1", name="read_file", input={})],
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )
    )
    worker.push(
        FinalResponse(content="reviewed: PASS", usage=TokenUsage(input_tokens=1, output_tokens=1))
    )

    # The dispatch count is the discriminator: it only increments if the eval
    # phase's tool call actually dispatched (i.e. was NOT paused at BeforeTool).
    reg = ScriptedToolRegistry()

    # The caller's approval middleware: SurfaceToHuman at BeforeTool. Without the
    # eval-phase drop, it pauses the reviewer's read and the run never dispatches.
    mw = ScriptedMiddleware()
    mw.push(
        HookPoint.BEFORE_TOOL,
        MiddlewareSurfaceToHuman(
            request=HumanRequestToolApproval(
                calls=[ToolCall(id="c1", name="read_file", input={})],
                risk_level="medium",  # type: ignore[arg-type]
            )
        ),
    )

    cfg = HarnessConfig(
        agent=worker,
        tool_registry=reg,
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        verifier=ScriptedVerifier([_passed()], max_iterations=3),
        middleware=mw,
    )
    h = StandardHarness(cfg)
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess), f"eval phase must run to a verdict, got {r!r}"
    # Load-bearing: the reviewer's read actually dispatched. With the caller
    # middleware NOT dropped, the eval phase pauses at BeforeTool BEFORE dispatch
    # and this count stays 0.
    assert reg.call_count >= 1, (
        "eval-phase reviewer tool must dispatch (caller approval middleware must "
        "be dropped for the read-only review run)"
    )


# The evaluate phase runs the configured ``eval_agent`` (a dedicated reviewer),
# not the inner worker's agent — so the model reviewing the work is not the one
# that wrote it (a builder reviewing itself tends to rubber-stamp).
async def test_eval_agent_override_runs_distinct_reviewer() -> None:
    from spore_core.execution_registry import ExecutionRegistry

    build = MockAgent(AgentId("build"))
    build.push(FinalResponse(content="built", usage=TokenUsage(input_tokens=1, output_tokens=1)))
    reviewer = MockAgent(AgentId("reviewer"))
    reviewer.push(
        FinalResponse(content="reviewed", usage=TokenUsage(input_tokens=1, output_tokens=1))
    )

    # Register the reviewer under the "reviewer" key; ``build`` is folded as the
    # default (empty-key) worker agent.
    registry = ExecutionRegistry.empty().into_builder().agent("reviewer", reviewer).build()
    cfg = HarnessConfig(
        agent=build,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        verifier=ScriptedVerifier([_passed()], max_iterations=3),
        registry=registry,
    )
    h = StandardHarness(cfg)
    simple = SelfVerifyingConfig.simple()
    strategy = simple.model_copy(update={"eval_agent": "reviewer"})
    t = Task.new(
        "implement the thing",
        SessionId("build-session"),
        strategy,
        budget=BudgetLimits(),
    )
    r = await h.run(HarnessRunOptions(t))
    assert isinstance(r, RunResultSuccess), f"expected Success, got {r!r}"
    # The reviewer ran the evaluate phase exactly once; the builder ran only the
    # build phase. Without the override the worker would serve BOTH and the
    # reviewer would never be called.
    assert reviewer.call_count == 1, "eval_agent reviewer runs the evaluate phase"
    assert build.call_count == 1, "builder runs only the build phase"


# Wire-format contract (cross-language parity): the two optional eval-phase
# overrides are OMITTED when unset (existing configs stay byte-identical) and
# serialize as bare string handles in declaration order (evaluator < eval_agent <
# eval_toolset < behavior) when set, round-tripping intact.
def test_config_eval_overrides_serde() -> None:
    bare = SelfVerifyingConfig.simple()
    bare_dump = bare.model_dump()
    assert "eval_agent" not in bare_dump, "eval_agent omitted when None"
    assert "eval_toolset" not in bare_dump, "eval_toolset omitted when None"

    with_ov = bare.model_copy(update={"eval_agent": "reviewer", "eval_toolset": "ro-tools"})
    dump = with_ov.model_dump()
    # Bare string handles (not nested objects).
    assert dump["eval_agent"] == "reviewer", dump
    assert dump["eval_toolset"] == "ro-tools", dump
    # Declaration / serialization order locked for parity.
    keys = list(dump.keys())
    assert (
        keys.index("evaluator")
        < keys.index("eval_agent")
        < keys.index("eval_toolset")
        < keys.index("behavior")
    ), f"field order locked for parity: {keys}"
    # Round-trips intact through JSON.
    back = SelfVerifyingConfig.model_validate(json.loads(json.dumps(dump)))
    assert back == with_ov, "round-trips intact"


# #147: the SelfVerifying combinator must charge the evaluator's turns against
# its budget scope. A 2-iteration loop with build+eval turns passes at a turn
# cap that includes the eval turns, and budget-exhausts at a cap one below.
async def _run_sv_two_iters_with_cap(cap: int) -> RunResult:
    # 2 iterations × (build 1 turn + eval 1 turn) = 4 turns. Same worker serves
    # both phases (Q1c default), so queue four FinalResponses.
    worker = MockAgent(AgentId("build"))
    for content in ("build0", "eval0", "build1", "eval1"):
        worker.push(
            FinalResponse(content=content, usage=TokenUsage(input_tokens=1, output_tokens=1))
        )
    cfg = HarnessConfig(
        agent=worker,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        # iteration 0 fails (retry), iteration 1 passes.
        verifier=ScriptedVerifier([_failed("retry"), _passed()], max_iterations=3),
    )
    h = StandardHarness(cfg)
    return await h.run(HarnessRunOptions(_task(max_turns=cap)))


async def test_self_verifying_charges_evaluator_turns_against_scope() -> None:
    # With the evaluator turns charged, a cap of 4 just fits → Success; a cap of
    # 3 is overrun by the second iteration's EVALUATOR turn (the two build turns
    # alone are only 2, so without charging the evaluator this would wrongly
    # succeed).
    fit = await _run_sv_two_iters_with_cap(4)
    assert isinstance(fit, RunResultSuccess), f"cap=4 should fit all 4 turns, got {fit!r}"
    exhausted = await _run_sv_two_iters_with_cap(3)
    assert not isinstance(exhausted, RunResultSuccess), (
        f"cap=3 must be exhausted once evaluator turns are charged, got {exhausted!r}"
    )
