"""Tests for the harness post-compaction verify->retry->warn loop (issue #46).

Mirrors the ``tests::compaction`` module in
``rust/crates/spore-core/src/harness.rs``. Each test exercises one rule of the
loop; the rule lives in the test docstring. The fixture-replay test drives the
loop against ``fixtures/compaction_loop/cases.json`` for cross-language
consistency.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from spore_core import (
    AggregateUsage,
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    CompactionTurn,
    HarnessConfig,
    InMemoryObservabilityProvider,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    SessionOutcomeSuccess,
    StandardHarness,
    TaskId,
    WarnEventCompactionVerificationFailed,
)
from spore_core.agent import Context, FinalResponse, TurnResult
from spore_core.context import (
    CompactionPreserveHints,
    CompactionVerificationResult,
)
from spore_core.context import SessionState as ContextSessionState
from spore_core.model import Message, ModelParams, Role, TextContent, TokenUsage

_FIXTURE = Path(__file__).resolve().parents[4] / "fixtures" / "compaction_loop" / "cases.json"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class CompactingContextManager:
    """A ``ContextManager`` that always offers a compaction turn. Records how
    many times ``apply_compaction`` ran. Relies on the default
    ``inject_missing_items`` body via the harness fallback."""

    def __init__(self, should: bool) -> None:
        self.applied = 0
        self._should = should

    async def assemble(self, session: SessionState, task: object) -> Context:
        return Context(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: object) -> None:
        session.messages.append(Message(role=Role.TOOL, content=TextContent(text="tool")))

    async def append_user_message(self, session: SessionState, text: str) -> None:
        pass

    def should_compact(self, session: SessionState) -> bool:
        return self._should

    def prepare_compaction_turn(self, session: SessionState) -> CompactionTurn | None:
        vs = ContextSessionState(
            session_id=SessionId("s1"),
            task_id=TaskId("t1"),
            task_instruction="deploy the payment service",
        )
        vs.token_budget_used = 1000
        return CompactionTurn(
            context=Context(
                messages=[Message(role=Role.USER, content=TextContent(text="please summarize"))],
                tools=[],
                params=ModelParams(),
            ),
            preserve_hints=CompactionPreserveHints(),
            verification_state=vs,
            messages_removed=3,
        )

    def apply_compaction(self, session: SessionState, summary: str) -> None:
        self.applied += 1


class RecordingAgent:
    """Agent that records every context it is handed and replays a queue of
    final-response summaries."""

    def __init__(self, summaries: list[str]) -> None:
        self._summaries: deque[str] = deque(summaries)
        self.seen: list[Context] = []

    async def turn(self, context: Context) -> TurnResult:
        self.seen.append(context)
        content = self._summaries.popleft() if self._summaries else ""
        return FinalResponse(
            content=content,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def id(self) -> AgentId:
        return AgentId("recording")


class ScriptedVerifier:
    """Verifier that fails the first ``fail_first`` calls, then passes."""

    def __init__(self, fail_first: int, missing: list[str]) -> None:
        self._fail_first = fail_first
        self._missing = list(missing)

    def verify(
        self,
        summary: str,
        hints: CompactionPreserveHints,
        session_state: ContextSessionState,
    ) -> CompactionVerificationResult:
        if self._fail_first > 0:
            self._fail_first -= 1
            return CompactionVerificationResult(
                passed=False, missing_items=list(self._missing), detail="scripted fail"
            )
        return CompactionVerificationResult(passed=True, missing_items=[], detail="scripted pass")


def _harness_with(
    cm: CompactingContextManager,
    agent: RecordingAgent,
    verifier: object,
    obs: InMemoryObservabilityProvider,
    max_attempts: int,
) -> StandardHarness:
    return StandardHarness(
        HarnessConfig(
            agent=agent,
            tool_registry=ScriptedToolRegistry(),
            sandbox=AllowAllSandbox(),
            context_manager=cm,
            termination_policy=AlwaysContinuePolicy(),
            middleware=None,
            observability=obs,
            compaction_verifier=verifier,
            max_compaction_attempts=max_attempts,
        )
    )


async def _drive(
    h: StandardHarness,
    cm: CompactingContextManager,
) -> None:
    state = SessionState()
    usage = AggregateUsage()
    # Pre-condition: the should_compact gate is honored by the caller.
    if h._config.context_manager.should_compact(state):
        await h._run_compaction(state, SessionId("s1"), TaskId("t1"), 0, usage)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_no_compaction_when_should_compact_false() -> None:
    """should_compact False -> no compaction turn, no apply."""
    cm = CompactingContextManager(should=False)
    agent = RecordingAgent(["summary"])
    obs = InMemoryObservabilityProvider()
    h = _harness_with(cm, agent, ScriptedVerifier(0, []), obs, 2)
    await _drive(h, cm)
    assert len(agent.seen) == 0
    assert cm.applied == 0


async def test_passing_verifier_one_turn_one_apply_no_warn() -> None:
    """Passing verifier -> exactly one turn, one apply, no warn."""
    cm = CompactingContextManager(should=True)
    agent = RecordingAgent(["good summary"])
    obs = InMemoryObservabilityProvider()
    h = _harness_with(cm, agent, ScriptedVerifier(0, []), obs, 2)
    await _drive(h, cm)
    assert len(agent.seen) == 1
    assert cm.applied == 1
    assert obs.warn_spans(SessionId("s1")) == []


async def test_failing_then_passing_retries_and_injects_missing_items() -> None:
    """Fail-then-pass, max=2 -> two turns; retry context carries the injected
    missing-items message with the actual terms."""
    cm = CompactingContextManager(should=True)
    agent = RecordingAgent(["v1", "v2"])
    obs = InMemoryObservabilityProvider()
    h = _harness_with(cm, agent, ScriptedVerifier(1, ["payment", "deploy"]), obs, 2)
    await _drive(h, cm)
    assert len(agent.seen) == 2
    retry_ctx = agent.seen[1]
    injected = any(
        isinstance(m.content, TextContent)
        and "missing these items" in m.content.text
        and "payment" in m.content.text
        and "deploy" in m.content.text
        for m in retry_ctx.messages
    )
    assert injected
    assert cm.applied == 1
    assert obs.warn_spans(SessionId("s1")) == []


async def test_always_failing_warns_and_accepts_anyway() -> None:
    """Always-failing, max=2 -> warn with missing_items + accepted_anyway=True,
    apply still called, failure metric == 1."""
    cm = CompactingContextManager(should=True)
    agent = RecordingAgent(["v1", "v2"])
    obs = InMemoryObservabilityProvider()
    h = _harness_with(cm, agent, ScriptedVerifier(99, ["payment"]), obs, 2)
    await _drive(h, cm)
    assert len(agent.seen) == 2
    assert cm.applied == 1
    sid = SessionId("s1")
    warns = obs.warn_spans(sid)
    assert len(warns) == 1
    event = warns[0].event
    assert isinstance(event, WarnEventCompactionVerificationFailed)
    assert event.missing_items == ["payment"]
    assert event.accepted_anyway is True
    # Seed an outcome so metrics are produced (no turn span emitted in this unit).
    obs.set_session_outcome(sid, SessionOutcomeSuccess())
    m = await obs.get_session_metrics(sid)
    assert m is not None
    assert m.compaction_verification_failures == 1


async def test_max_attempts_one_honored() -> None:
    """max_compaction_attempts=1 -> one attempt then warn + accept."""
    cm = CompactingContextManager(should=True)
    agent = RecordingAgent(["v1", "v2", "v3"])
    obs = InMemoryObservabilityProvider()
    h = _harness_with(cm, agent, ScriptedVerifier(99, ["payment"]), obs, 1)
    await _drive(h, cm)
    assert len(agent.seen) == 1
    assert cm.applied == 1
    assert len(obs.warn_spans(SessionId("s1"))) == 1


async def test_emit_warn_default_noop_does_not_break_bare_provider() -> None:
    """emit_warn default no-op does not break a provider that doesn't override
    it (W4)."""

    class BareProvider:
        """Minimal provider that does NOT override emit_warn; the Protocol
        default body is the only emit_warn it has."""

        def emit_turn(self, span: object) -> None: ...
        def emit_tool_call(self, span: object) -> None: ...
        def emit_sensor(self, span: object) -> None: ...
        def emit_context(self, span: object) -> None: ...
        def emit_middleware(self, span: object) -> None: ...
        def emit_patch(self, span: object) -> None: ...

        async def flush_session(self, session_id: SessionId) -> None: ...

        async def get_session_metrics(self, session_id: SessionId) -> None:
            return None

        async def get_sessions(self, *a: object, **k: object) -> list:
            return []

        async def get_trace(self, session_id: SessionId) -> list:
            return []

    cm = CompactingContextManager(should=True)
    agent = RecordingAgent(["v1", "v2"])
    obs = BareProvider()
    h = _harness_with(cm, agent, ScriptedVerifier(99, ["payment"]), obs, 2)
    # Reaching the warn path must not raise; the bare provider ignores it.
    await _drive(h, cm)
    assert cm.applied == 1


# ---------------------------------------------------------------------------
# Cross-language consistency fixture replay (issue #46)
# ---------------------------------------------------------------------------


class FixtureVerifier:
    """Verifier driven by a fixture verdict queue; repeats the last verdict
    once the queue is exhausted, matching the fixture contract."""

    def __init__(self, verdicts: list[CompactionVerificationResult]) -> None:
        self._verdicts = verdicts
        self._idx = 0

    def verify(
        self,
        summary: str,
        hints: CompactionPreserveHints,
        session_state: ContextSessionState,
    ) -> CompactionVerificationResult:
        i = self._idx
        self._idx += 1
        if i < len(self._verdicts):
            return self._verdicts[i]
        return self._verdicts[-1]


async def test_fixture_replay_loop_outcomes() -> None:
    """Replay fixtures/compaction_loop/cases.json against the loop."""
    suite = json.loads(_FIXTURE.read_text())
    cases = suite["cases"]
    assert len(cases) >= 5

    for case in cases:
        cm = CompactingContextManager(should=True)
        agent = RecordingAgent(["s1", "s2", "s3", "s4"])
        obs = InMemoryObservabilityProvider()
        verifier = FixtureVerifier(
            [
                CompactionVerificationResult(
                    passed=v["passed"], missing_items=list(v["missing_items"]), detail=""
                )
                for v in case["verdicts"]
            ]
        )
        h = _harness_with(cm, agent, verifier, obs, case["max_compaction_attempts"])
        await _drive(h, cm)

        sid = SessionId("s1")
        expected = case["expected"]
        assert len(agent.seen) == expected["attempts"], f"{case['name']}: attempts"
        assert cm.applied == expected["apply_compaction_calls"], (
            f"{case['name']}: apply_compaction_calls"
        )
        warns = obs.warn_spans(sid)
        assert bool(warns) == expected["warn_emitted"], f"{case['name']}: warn_emitted"
        if expected["warn_emitted"]:
            assert len(warns) == 1, f"{case['name']}: exactly one warn"
            event = warns[0].event
            assert isinstance(event, WarnEventCompactionVerificationFailed)
            assert event.missing_items == expected["final_missing_items"], (
                f"{case['name']}: final_missing_items"
            )
            assert event.accepted_anyway is True, f"{case['name']}: accepted_anyway"
        if "retry_injected_missing" in expected:
            joined = ", ".join(expected["retry_injected_missing"])
            found = any(
                isinstance(m.content, TextContent)
                and "missing these items" in m.content.text
                and joined in m.content.text
                for ctx in agent.seen[1:]
                for m in ctx.messages
            )
            assert found, f"{case['name']}: retry injection"
