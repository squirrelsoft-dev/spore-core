"""SC-BUG-1: a HITL approval / clarification / deny resume must RE-ENTER the
SelfVerifying frame, not run only the bare build leaf.

Mirrors the Rust regression tests
``self_verifying_hitl_{resume,deny,clarification}_reenters_eval_frame`` in
``rust/crates/spore-core/src/harness.rs`` (reference commit ``8d1d679``).

The original run pauses on the BUILD phase's first tool call (caller approval
middleware → SurfaceToHuman, or a tool returning AwaitingClarification). Before
the fix, ``_resume_inner`` ran ``_run_react`` on the paused ReAct build leaf and
returned its Success directly — the evaluate phase + verifier never ran, so the
looper's eval-frame reviewer was silently skipped. After the fix the pause
carries the full composed task (``_finish`` rewrites it on the way up, as it
already does for ``Consult``) and the resume re-drives the whole SelfVerifying
strategy from the approved worker session, so the verifier runs.
"""

from __future__ import annotations

from typing import Any

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetLimits,
    FinalResponse,
    HarnessConfig,
    HarnessRunOptions,
    HumanRequestClarification,
    HumanRequestToolApproval,
    HumanResponseAllow,
    HumanResponseAnswer,
    HumanResponseDeny,
    MiddlewareSurfaceToHuman,
    MockAgent,
    NoopContextManager,
    RunResultWaitingForHuman,
    ScriptedMiddleware,
    ScriptedToolRegistry,
    SelfVerifyingConfig,
    SessionId,
    StandardHarness,
    Task,
    ToolCall,
    ToolCallRequested,
    ToolOutputAwaitingClarification,
    TokenUsage,
)
from spore_core.middleware import HookPoint

from .test_harness_self_verifying import ScriptedVerifier, _passed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=1, output_tokens=1)


def _worker(tool_name: str) -> MockAgent:
    """A worker scripted to emit a tool call on the FIRST build turn (which is
    gated → pause), then a FinalResponse once the approved/answered call
    dispatches, then a TOOL-FREE FinalResponse for the eval phase so it never
    re-trips the gate (independent of SC-29 parity)."""
    a = MockAgent(AgentId("build"))
    a.push(
        ToolCallRequested(
            calls=[ToolCall(id="c1", name=tool_name, input={})],
            usage=_usage(),
        )
    )
    a.push(FinalResponse(content="built", usage=_usage()))
    a.push(FinalResponse(content="reviewed: PASS", usage=_usage()))
    return a


def _config(*, agent: MockAgent, verifier: Any, tool_registry: Any, middleware: Any = None) -> Any:
    return HarnessConfig(
        agent=agent,
        tool_registry=tool_registry,
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        verifier=verifier,
        middleware=middleware,
    )


def _task() -> Task:
    # A top-level SelfVerifying(ReAct) task.
    return Task.new(
        "implement the thing",
        SessionId("build-session"),
        SelfVerifyingConfig.simple(),
        budget=BudgetLimits(),
    )


def _approval_middleware(risk_level: str) -> ScriptedMiddleware:
    mw = ScriptedMiddleware()
    mw.push(
        HookPoint.BEFORE_TOOL,
        MiddlewareSurfaceToHuman(
            request=HumanRequestToolApproval(
                calls=[ToolCall(id="c1", name="write_file", input={})],
                risk_level=risk_level,  # type: ignore[arg-type]
            )
        ),
    )
    return mw


# ---------------------------------------------------------------------------
# resume (Allow) arm
# ---------------------------------------------------------------------------


async def test_self_verifying_hitl_resume_reenters_eval_frame() -> None:
    worker = _worker("write_file")
    verifier = ScriptedVerifier([_passed()], max_iterations=3)
    reg = ScriptedToolRegistry()
    h = StandardHarness(
        _config(
            agent=worker,
            verifier=verifier,
            tool_registry=reg,
            middleware=_approval_middleware("medium"),
        )
    )

    # 1) The run pauses on the build-phase tool call.
    paused = await h.run(HarnessRunOptions(_task()))
    assert isinstance(paused, RunResultWaitingForHuman), (
        f"expected WaitingForHuman on the build tool call, got {paused!r}"
    )
    state = paused.state
    # Part-1 check: the pause must carry the COMPOSED task so the resume can
    # re-enter the frame — not the bare ReAct build leaf.
    assert isinstance(state.task.loop_strategy, SelfVerifyingConfig), (
        f"the unwound pause must carry the SelfVerifying task, got {state.task.loop_strategy!r}"
    )

    # 2) Approve and resume.
    resumed = await h.resume(state, HumanResponseAllow())
    from spore_core import RunResultSuccess

    assert isinstance(resumed, RunResultSuccess), f"resume must run to a verdict, got {resumed!r}"

    # Load-bearing: the eval-phase verifier ran AFTER the resume — i.e. the
    # SelfVerifying frame was re-entered. Before the fix the resume returned the
    # bare leaf's Success and this count stays 0.
    assert len(verifier.calls) >= 1, (
        f"SelfVerifying frame must be re-entered on HITL resume so the eval-phase "
        f"verifier runs (got {len(verifier.calls)} verifier calls)"
    )
    # And the approved build tool actually dispatched on resume.
    assert reg.call_count >= 1, "the approved build-phase tool must dispatch on resume"


# ---------------------------------------------------------------------------
# deny arm
# ---------------------------------------------------------------------------


async def test_self_verifying_hitl_deny_reenters_eval_frame() -> None:
    worker = _worker("write_file")
    verifier = ScriptedVerifier([_passed()], max_iterations=3)
    reg = ScriptedToolRegistry()
    h = StandardHarness(
        _config(
            agent=worker,
            verifier=verifier,
            tool_registry=reg,
            middleware=_approval_middleware("high"),
        )
    )

    paused = await h.run(HarnessRunOptions(_task()))
    assert isinstance(paused, RunResultWaitingForHuman), f"expected WaitingForHuman, got {paused!r}"
    state = paused.state
    assert isinstance(state.task.loop_strategy, SelfVerifyingConfig), (
        f"the unwound pause must carry the SelfVerifying task, got {state.task.loop_strategy!r}"
    )

    resumed = await h.resume(state, HumanResponseDeny(reason="not allowed"))
    from spore_core import RunResultSuccess

    assert isinstance(resumed, RunResultSuccess), (
        f"deny resume must run to a verdict, got {resumed!r}"
    )
    assert len(verifier.calls) >= 1, (
        "SelfVerifying frame must be re-entered on a DENY resume so the eval-phase verifier runs"
    )


# ---------------------------------------------------------------------------
# clarification (Answer) arm
# ---------------------------------------------------------------------------


async def test_self_verifying_hitl_clarification_reenters_eval_frame() -> None:
    # No approval middleware: the tool itself returns AwaitingClarification,
    # pausing the build phase with a HumanRequestClarification.
    worker = _worker("ask")
    verifier = ScriptedVerifier([_passed()], max_iterations=3)
    reg = ScriptedToolRegistry()
    reg.push(ToolOutputAwaitingClarification(question="which file?", options=None))
    h = StandardHarness(_config(agent=worker, verifier=verifier, tool_registry=reg))

    paused = await h.run(HarnessRunOptions(_task()))
    assert isinstance(paused, RunResultWaitingForHuman), f"expected WaitingForHuman, got {paused!r}"
    assert isinstance(paused.request, HumanRequestClarification), (
        f"expected a clarification pause, got {paused.request!r}"
    )
    state = paused.state
    assert isinstance(state.task.loop_strategy, SelfVerifyingConfig), (
        f"the unwound pause must carry the SelfVerifying task, got {state.task.loop_strategy!r}"
    )

    resumed = await h.resume(state, HumanResponseAnswer(text="the config file"))
    from spore_core import RunResultSuccess

    assert isinstance(resumed, RunResultSuccess), (
        f"clarification resume must run to a verdict, got {resumed!r}"
    )
    assert len(verifier.calls) >= 1, (
        "SelfVerifying frame must be re-entered on a clarification resume so the "
        "eval-phase verifier runs"
    )
