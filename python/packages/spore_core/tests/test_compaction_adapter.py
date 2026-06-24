"""Tests for the StandardContextManager harness compaction adapter (issue #55).

Mirrors the ``tests`` module in
``rust/crates/spore-core/src/compaction_adapter.rs``. Each test exercises one
adapter rule; the rule lives in the test docstring. An end-to-end harness test
(mock model) drives utilization over threshold and asserts a real compaction
turn fires through the seam and shrinks the session. A fixture-parity test
replays ``fixtures/compaction_loop/cases.json`` through the real adapter for
verify→retry→warn control-flow parity.
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
    CompactionConfig,
    HarnessConfig,
    InMemoryObservabilityProvider,
    KeyTermVerifier,
    RICH_STATE_KEY,
    ScriptedToolRegistry,
    SessionId,
    SessionOutcomeSuccess,
    StandardCompactionAdapter,
    StandardContextManager,
    StandardHarness,
    TaskId,
    ToolOutputError,
    ToolOutputSuccess,
    WarnEventCompactionVerificationFailed,
    into_harness_adapter,
    seed_rich_state,
)
from spore_core.agent import Context, FinalResponse, TurnResult
from spore_core.compaction_adapter import _rich_from_dict
from spore_core.context import (
    CompactionPreserveHints,
    CompactionVerificationResult,
)
from spore_core.context import SessionState as RichSessionState
from spore_core.harness import HarnessToolResult
from spore_core.harness import SessionState as HarnessState
from spore_core.model import Message, ProviderInfo, Role, TextContent, TokenUsage

_FIXTURE = Path(__file__).resolve().parents[4] / "fixtures" / "compaction_loop" / "cases.json"


# ---------------------------------------------------------------------------
# Stub model + builders
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal ``ModelInterface`` — only ``count_tokens`` / ``provider`` are
    reachable from the adapter paths under test, and those return constants."""

    async def call(self, request: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    async def call_streaming(self, request: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    async def count_tokens(self, request: object) -> int:
        return 0

    def provider(self) -> ProviderInfo:
        return ProviderInfo(name="stub", model_id="stub", context_window=200_000)


def _rich_manager() -> StandardContextManager:
    cfg = CompactionConfig(
        threshold=0.80,
        preserve_recent_n=2,
        head_tail_tokens=64,
        max_compaction_attempts=2,
    )
    return StandardContextManager(_StubModel(), compaction=cfg)


def _msg(role: Role, text: str) -> Message:
    return Message(role=role, content=TextContent(text=text))


def _rich_state(messages: int, used: int, limit: int) -> RichSessionState:
    s = RichSessionState(
        session_id=SessionId("s1"),
        task_id=TaskId("t1"),
        task_instruction="deploy the payment service",
    )
    s.window_limit = limit
    s.token_budget_used = used
    s.message_history = [_msg(Role.USER, f"m{i}") for i in range(messages)]
    return s


def _session_with(rich: RichSessionState) -> HarnessState:
    s = HarnessState()
    seed_rich_state(s, rich)
    return s


# ---------------------------------------------------------------------------
# should_compact threshold
# ---------------------------------------------------------------------------


def test_should_compact_below_threshold_is_false() -> None:
    """Utilization below threshold -> no compaction."""
    adapter = StandardCompactionAdapter(_rich_manager())
    session = _session_with(_rich_state(10, 70, 100))  # 0.70 < 0.80
    assert adapter.should_compact(session) is False


def test_should_compact_at_threshold_is_true() -> None:
    """Utilization at threshold -> compaction."""
    adapter = StandardCompactionAdapter(_rich_manager())
    session = _session_with(_rich_state(10, 80, 100))  # 0.80 >= 0.80
    assert adapter.should_compact(session) is True


def test_should_compact_above_threshold_is_true() -> None:
    """Utilization above threshold -> compaction."""
    adapter = StandardCompactionAdapter(_rich_manager())
    session = _session_with(_rich_state(10, 95, 100))
    assert adapter.should_compact(session) is True


def test_should_compact_false_without_rich_state() -> None:
    """No seeded rich state -> safe default (no compaction)."""
    adapter = StandardCompactionAdapter(_rich_manager())
    assert adapter.should_compact(HarnessState()) is False


def test_should_compact_false_with_malformed_rich_state() -> None:
    """Malformed rich-state blob degrades to no compaction, never raises."""
    adapter = StandardCompactionAdapter(_rich_manager())
    session = HarnessState()
    session.extras[RICH_STATE_KEY] = {"not": "a valid state"}
    assert adapter.should_compact(session) is False


# ---------------------------------------------------------------------------
# prepare_compaction_turn
# ---------------------------------------------------------------------------


def test_prepare_returns_none_for_short_history() -> None:
    """History no longer than preserve window -> nothing to compact -> None."""
    adapter = StandardCompactionAdapter(_rich_manager())
    # preserve_recent_n = 2, history = 2 -> nothing to compact.
    session = _session_with(_rich_state(2, 95, 100))
    assert adapter.prepare_compaction_turn(session) is None


def test_prepare_returns_none_without_rich_state() -> None:
    """No seeded rich state -> None."""
    adapter = StandardCompactionAdapter(_rich_manager())
    assert adapter.prepare_compaction_turn(HarnessState()) is None


def test_prepare_projects_hints_state_and_count() -> None:
    """A real turn projects messages_removed, verification state, hints, and the
    summarization instruction."""
    adapter = StandardCompactionAdapter(_rich_manager())
    rich = _rich_state(10, 95, 100)  # 10 - 2 preserved = 8 removed
    session = _session_with(rich)
    turn = adapter.prepare_compaction_turn(session)
    assert turn is not None
    assert turn.messages_removed == 8
    # verification_state mirrors the rich state.
    assert turn.verification_state.task_instruction == rich.task_instruction
    assert turn.verification_state.token_budget_used == 95
    # default hints projected.
    assert turn.preserve_hints.keep_current_task_state is True
    # the summarization instruction is appended after the compacted msgs.
    assert any(
        isinstance(m.content, TextContent) and "Summarize" in m.content.text
        for m in turn.context.messages
    )


# ---------------------------------------------------------------------------
# apply_compaction shrinks the session
# ---------------------------------------------------------------------------


def test_apply_compaction_shrinks_session() -> None:
    """apply_compaction replaces the compacted span with the summary and writes
    the mutated rich state back (messages + extras)."""
    adapter = StandardCompactionAdapter(_rich_manager())
    session = _session_with(_rich_state(10, 95, 100))
    before = len(session.messages)
    adapter.apply_compaction(session, "summary preserving payment deploy")
    # 2 preserved + 1 summary = 3.
    assert len(session.messages) < before
    assert len(session.messages) == 3
    # round-tripped rich state also shrank.
    rich = _rich_from_dict(session.extras[RICH_STATE_KEY])
    assert len(rich.message_history) == 3
    assert rich.message_history[0].role == Role.ASSISTANT


def _content_heavy_state(messages: int, used: int, limit: int) -> RichSessionState:
    s = _rich_state(messages, used, limit)
    s.message_history = [
        _msg(Role.USER, f"message number {i} with a fair amount of content to estimate tokens from")
        for i in range(messages)
    ]
    return s


def test_apply_compaction_reclaims_real_tokens_and_drops_budget() -> None:
    """Known Deviation #2 fix: dropping messages reclaims real tokens so
    ``token_budget_used`` falls, and the ``token_budget_used`` seam reports the
    post-compaction budget for span stamping."""
    adapter = StandardCompactionAdapter(_rich_manager())
    session = _session_with(_content_heavy_state(10, 95, 100))

    before = _rich_from_dict(session.extras[RICH_STATE_KEY]).token_budget_used
    adapter.apply_compaction(session, "summary preserving payment deploy")
    after = _rich_from_dict(session.extras[RICH_STATE_KEY]).token_budget_used

    assert after < before, (
        f"token_budget_used must drop after a real reclamation: {before} -> {after}"
    )
    # The seam reports the post-compaction budget for span stamping.
    assert adapter.token_budget_used(session) == after


def test_apply_compaction_multi_compaction_keeps_dropping_budget() -> None:
    """Healthy multi-compaction: after compacting, growing history and budget
    again must let a second compaction reclaim more tokens."""
    adapter = StandardCompactionAdapter(_rich_manager())
    session = _session_with(_content_heavy_state(10, 95, 100))

    adapter.apply_compaction(session, "first summary about payment deploy")
    after_first = adapter.token_budget_used(session)
    assert after_first is not None and after_first < 95

    # Simulate the session growing again past threshold.
    grown = _content_heavy_state(10, 95, 100)
    seed_rich_state(session, grown)
    adapter.apply_compaction(session, "second summary about payment deploy")
    after_second = adapter.token_budget_used(session)
    assert after_second is not None and after_second < 95, (
        "second compaction also reclaims real tokens"
    )


def test_token_budget_used_none_without_rich_state() -> None:
    """The budget seam returns ``None`` when no rich state has been seeded, so
    the harness leaves the span's pre-compaction estimate in place."""
    adapter = StandardCompactionAdapter(_rich_manager())
    assert adapter.token_budget_used(HarnessState()) is None


def test_apply_compaction_swallows_without_rich_state() -> None:
    """No rich state -> no-op, no raise."""
    adapter = StandardCompactionAdapter(_rich_manager())
    session = HarnessState()
    adapter.apply_compaction(session, "summary")
    assert session.messages == []


def test_apply_compaction_swallows_malformed_state() -> None:
    """Malformed rich state -> no-op, no raise."""
    adapter = StandardCompactionAdapter(_rich_manager())
    session = HarnessState()
    session.extras[RICH_STATE_KEY] = {"bad": True}
    adapter.apply_compaction(session, "summary")
    assert session.messages == []


# ---------------------------------------------------------------------------
# minimal seam methods
# ---------------------------------------------------------------------------


async def test_append_user_message_and_assemble() -> None:
    """append_user_message appends a USER message; assemble passes it through."""
    adapter = StandardCompactionAdapter(_rich_manager())
    session = HarnessState()
    await adapter.append_user_message(session, "hello")
    assert len(session.messages) == 1
    assert session.messages[0].role == Role.USER
    ctx = await adapter.assemble(session, task=None)  # type: ignore[arg-type]
    assert len(ctx.messages) == 1


async def test_replace_tool_result_rerenders_the_recorded_message() -> None:
    """SC-9 (#158): replace_tool_result re-renders the previously appended
    role:tool message at the recorded index (no new message added), and is a
    defensive no-op for an out-of-range index."""
    adapter = StandardCompactionAdapter(_rich_manager())
    state = HarnessState()
    original = HarnessToolResult(call_id="c1", output=ToolOutputSuccess(content="ORIGINAL"))
    await adapter.append_tool_result(state, original)
    idx = len(state.messages) - 1
    assert isinstance(state.messages[idx].content, TextContent)
    assert state.messages[idx].content.text == "ORIGINAL"

    # A middleware rewrote the result; re-render the recorded message.
    rewritten = HarnessToolResult(
        call_id="c1", output=ToolOutputError(message="REWRITTEN", recoverable=True)
    )
    await adapter.replace_tool_result(state, idx, rewritten)
    assert len(state.messages) == 1, "no message added on replace"
    assert state.messages[idx].role == Role.TOOL
    assert isinstance(state.messages[idx].content, TextContent)
    assert state.messages[idx].content.text == "REWRITTEN"

    # Out-of-range index is a defensive no-op.
    await adapter.replace_tool_result(state, 99, original)
    assert len(state.messages) == 1
    assert state.messages[idx].content.text == "REWRITTEN"


# ---------------------------------------------------------------------------
# Test doubles for end-to-end + fixture parity
# ---------------------------------------------------------------------------


class _RecordingAgent:
    """Agent that records every context it is handed and replays a queue of
    final-response summaries."""

    def __init__(self, summaries: list[str]) -> None:
        self._summaries: deque[str] = deque(summaries)
        self.seen: list[Context] = []

    async def turn(self, context: Context) -> TurnResult:
        self.seen.append(context)
        content = self._summaries.popleft() if self._summaries else ""
        return FinalResponse(content=content, usage=TokenUsage(input_tokens=1, output_tokens=1))

    def id(self) -> AgentId:
        return AgentId("recording")


class _FixtureVerifier:
    """Verifier driven by a fixture verdict queue; repeats the last verdict once
    exhausted, matching the fixture contract."""

    def __init__(self, verdicts: list[CompactionVerificationResult]) -> None:
        self._verdicts = verdicts
        self._idx = 0

    def verify(
        self,
        summary: str,
        hints: CompactionPreserveHints,
        session_state: RichSessionState,
    ) -> CompactionVerificationResult:
        i = self._idx
        self._idx += 1
        if i < len(self._verdicts):
            return self._verdicts[i]
        return self._verdicts[-1]


def _harness_with(adapter: object, agent: object, verifier: object, obs: object, max_attempts: int):
    return StandardHarness(
        HarnessConfig(
            agent=agent,
            tool_registry=ScriptedToolRegistry(),
            sandbox=AllowAllSandbox(),
            context_manager=adapter,
            termination_policy=AlwaysContinuePolicy(),
            observability=obs,
            compaction_verifier=verifier,
            max_compaction_attempts=max_attempts,
        )
    )


# ---------------------------------------------------------------------------
# End-to-end: a real compaction turn fires through the seam and shrinks session
# ---------------------------------------------------------------------------


async def test_end_to_end_drives_compaction_through_seam() -> None:
    """A real adapter drives a compaction turn through the loop seam: the
    session shrinks, one compaction span is emitted, and a passing summary warns
    nothing."""
    adapter = into_harness_adapter(_rich_manager())
    agent = _RecordingAgent(["we are working on the deploy of the payment service"])
    obs = InMemoryObservabilityProvider()
    # Default KeyTermVerifier sources the task instruction "deploy the payment
    # service" — the summary contains those terms, so it passes first attempt.
    h = _harness_with(adapter, agent, KeyTermVerifier(), obs, 2)

    session = _session_with(_rich_state(10, 95, 100))
    before = len(session.messages)
    assert h._config.context_manager.should_compact(session) is True

    usage = AggregateUsage()
    await h._run_compaction(
        session, SessionId("s1"), TaskId("t1"), 0, usage, h._config.registry.resolve_agent("")
    )

    assert len(session.messages) < before
    assert len(session.messages) == 3
    sid = SessionId("s1")
    obs.set_session_outcome(sid, SessionOutcomeSuccess())
    metrics = await obs.get_session_metrics(sid)
    assert metrics is not None
    assert metrics.compactions == 1
    assert obs.warn_spans(sid) == []


# ---------------------------------------------------------------------------
# Cross-language fixture parity (verify->retry->warn) with the real adapter
# ---------------------------------------------------------------------------


async def test_compaction_loop_fixture_parity_with_real_adapter() -> None:
    """Replay fixtures/compaction_loop/cases.json through the real adapter."""
    suite = json.loads(_FIXTURE.read_text())
    cases = suite["cases"]
    assert len(cases) >= 5

    for case in cases:
        adapter = into_harness_adapter(_rich_manager())
        agent = _RecordingAgent(["s1", "s2", "s3", "s4"])
        obs = InMemoryObservabilityProvider()
        verifier = _FixtureVerifier(
            [
                CompactionVerificationResult(
                    passed=v["passed"], missing_items=list(v["missing_items"]), detail=""
                )
                for v in case["verdicts"]
            ]
        )
        h = _harness_with(adapter, agent, verifier, obs, case["max_compaction_attempts"])

        # 10 messages, over threshold -> a real CompactionTurn is offered.
        session = _session_with(_rich_state(10, 95, 100))
        usage = AggregateUsage()
        await h._run_compaction(
            session, SessionId("s1"), TaskId("t1"), 0, usage, h._config.registry.resolve_agent("")
        )

        sid = SessionId("s1")
        expected = case["expected"]

        # apply_compaction always runs exactly once -> session shrank to 3.
        rich = _rich_from_dict(session.extras[RICH_STATE_KEY])
        assert len(rich.message_history) == 3, f"{case['name']}: applied"
        assert expected["apply_compaction_calls"] == 1, f"{case['name']}: fixture sanity"

        # attempts parity.
        assert len(agent.seen) == expected["attempts"], f"{case['name']}: attempts"

        # warn parity.
        warns = obs.warn_spans(sid)
        assert bool(warns) == expected["warn_emitted"], f"{case['name']}: warn parity"
        if expected["warn_emitted"]:
            assert len(warns) == 1, f"{case['name']}: exactly one warn"
            event = warns[0].event
            assert isinstance(event, WarnEventCompactionVerificationFailed)
            assert event.missing_items == expected["final_missing_items"], (
                f"{case['name']}: final_missing_items"
            )
            assert event.accepted_anyway is True, f"{case['name']}: accepted_anyway"

        # retry injection parity: the missing-items prompt carries the items.
        if expected.get("retry_injected_missing"):
            joined = ", ".join(expected["retry_injected_missing"])
            found = any(
                isinstance(m.content, TextContent)
                and "missing these items" in m.content.text
                and joined in m.content.text
                for ctx in agent.seen[1:]
                for m in ctx.messages
            )
            assert found, f"{case['name']}: retry injection"
