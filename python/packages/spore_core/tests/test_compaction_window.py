"""Tests for the configurable compaction window (issue #141).

Covers:

- :meth:`StandardContextManager.resolve_context_length` fallback chain
  (config ``context_length`` > 0 → model ``context_window`` > 0 →
  :data:`DEFAULT_CONTEXT_LENGTH`), explicit-zero fall-through, and the no-clamp
  rule.
- :meth:`StandardContextManager.seed_session` wiring the resolved window into a
  fresh :class:`SessionState`.
- The trigger math (``should_compact``) at small windows, proving
  ``threshold x window_limit`` respects the configured window.
- ``context_length`` is OMITTED from a default config's serialized dict/JSON.
- The new conservative :class:`SessionState` constructor default.

A fixture-replay test drives the cross-language ground-truth
``fixtures/compaction_window/cases.json``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator
from pathlib import Path

from spore_core.context import (
    DEFAULT_CONTEXT_LENGTH,
    CompactionConfig,
    SessionState,
    StandardContextManager,
)
from spore_core.harness import SessionId, TaskId
from spore_core.model import (
    ModelRequest,
    ModelResponse,
    ProviderInfo,
    StopReason,
    StreamEvent,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubModel:
    """``ModelInterface`` whose ``provider().context_window`` is configurable.

    A ``context_window`` of ``0`` models "no usable metadata" so the resolver
    falls through to the default.
    """

    def __init__(self, context_window: int) -> None:
        self._context_window = context_window

    async def call(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content=[], usage=TokenUsage(), stop_reason=StopReason.END_TURN)

    async def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]
        raise NotImplementedError

    async def count_tokens(self, request: ModelRequest) -> int:
        return 0

    def provider(self) -> ProviderInfo:
        return ProviderInfo(name="stub", model_id="stub", context_window=self._context_window)


def _mgr(context_length: int | None, model_context_window: int) -> StandardContextManager:
    return StandardContextManager(
        model=StubModel(context_window=model_context_window),
        compaction=CompactionConfig(context_length=context_length),
    )


# ---------------------------------------------------------------------------
# Rule: resolver fallback chain
# ---------------------------------------------------------------------------


def test_resolve_config_wins_over_model() -> None:
    # Configured 8000 wins even though the model advertises a larger window.
    assert (
        _mgr(context_length=8_000, model_context_window=128_000).resolve_context_length() == 8_000
    )


def test_resolve_model_fallback_when_config_none() -> None:
    assert (
        _mgr(context_length=None, model_context_window=128_000).resolve_context_length() == 128_000
    )


def test_resolve_default_when_both_absent() -> None:
    assert (
        _mgr(context_length=None, model_context_window=0).resolve_context_length()
        == DEFAULT_CONTEXT_LENGTH
    )
    assert DEFAULT_CONTEXT_LENGTH == 8_000


def test_resolve_explicit_zero_config_falls_through_to_model() -> None:
    # An explicit 0 is treated like None — only `> 0` configs are honored.
    assert _mgr(context_length=0, model_context_window=128_000).resolve_context_length() == 128_000


def test_resolve_explicit_zero_config_and_no_model_uses_default() -> None:
    assert (
        _mgr(context_length=0, model_context_window=0).resolve_context_length()
        == DEFAULT_CONTEXT_LENGTH
    )


def test_resolve_no_clamp_config_larger_than_model() -> None:
    # A configured value larger than the model's real window is used as-is.
    assert (
        _mgr(context_length=500_000, model_context_window=128_000).resolve_context_length()
        == 500_000
    )


# ---------------------------------------------------------------------------
# Rule: seed_session sets window_limit == resolve_context_length()
# ---------------------------------------------------------------------------


def test_seed_session_sets_resolved_window_limit() -> None:
    mgr = _mgr(context_length=None, model_context_window=128_000)
    state = mgr.seed_session(SessionId("s1"), TaskId("t1"), "do the thing")
    assert state.window_limit == mgr.resolve_context_length() == 128_000
    # The rest of the state is the bare constructor default.
    assert state.token_budget_used == 0
    assert state.task_instruction == "do the thing"


def test_seed_session_honors_config_override() -> None:
    mgr = _mgr(context_length=8_000, model_context_window=128_000)
    state = mgr.seed_session(SessionId("s1"), TaskId("t1"), "x")
    assert state.window_limit == 8_000


# ---------------------------------------------------------------------------
# Rule: trigger math respects the (often small) configured window
# ---------------------------------------------------------------------------


def test_should_compact_at_small_window() -> None:
    mgr = StandardContextManager(
        model=StubModel(context_window=128_000), compaction=CompactionConfig(threshold=0.8)
    )
    st = SessionState(
        session_id=SessionId("s1"),
        task_id=TaskId("t1"),
        task_instruction="x",
        window_limit=8_000,
    )
    # 6400 / 8000 == 0.8 → at threshold → compact.
    st.token_budget_used = 6_400
    assert mgr.should_compact(st)
    # 6399 / 8000 < 0.8 → no compaction.
    st.token_budget_used = 6_399
    assert not mgr.should_compact(st)


def test_should_compact_zero_window_never_compacts() -> None:
    mgr = StandardContextManager(
        model=StubModel(context_window=128_000), compaction=CompactionConfig(threshold=0.8)
    )
    st = SessionState(
        session_id=SessionId("s1"),
        task_id=TaskId("t1"),
        task_instruction="x",
        window_limit=0,
    )
    st.token_budget_used = 9_999
    assert not mgr.should_compact(st)


# ---------------------------------------------------------------------------
# Rule: default config omits context_length from serialized dict/JSON
# ---------------------------------------------------------------------------


def test_default_config_omits_context_length_from_dict() -> None:
    cfg = CompactionConfig()
    assert cfg.context_length is None
    data = cfg.to_dict()
    assert "context_length" not in data
    # The remaining keys serialize exactly as before.
    assert data == {
        "threshold": 0.80,
        "preserve_recent_n": 8,
        "head_tail_tokens": 512,
        "offload_path": ".spore/offload",
        "max_compaction_attempts": 2,
    }


def test_default_config_omits_context_length_from_json() -> None:
    assert "context_length" not in json.dumps(CompactionConfig().to_dict())


def test_set_config_includes_context_length() -> None:
    data = CompactionConfig(context_length=8_000).to_dict()
    assert data["context_length"] == 8_000


# ---------------------------------------------------------------------------
# Rule: SessionState constructor default is now the conservative 8K, not 200K
# ---------------------------------------------------------------------------


def test_session_state_default_window_is_conservative() -> None:
    st = SessionState(session_id=SessionId("s1"), task_id=TaskId("t1"), task_instruction="x")
    assert st.window_limit == DEFAULT_CONTEXT_LENGTH == 8_000


# ---------------------------------------------------------------------------
# Cross-language fixture replay (ground truth — do NOT edit the fixture)
# ---------------------------------------------------------------------------


def _fixture_cases() -> dict[str, object]:
    here = Path(__file__).resolve()
    path = here.parents[4] / "fixtures" / "compaction_window" / "cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_fixture_resolver_cases() -> None:
    cases = _fixture_cases()["resolver_cases"]
    assert isinstance(cases, list) and cases
    for case in cases:
        # config_context_length is null in JSON → None config.
        config_len = case["config_context_length"]
        mgr = StandardContextManager(
            model=StubModel(context_window=case["model_context_window"]),
            compaction=CompactionConfig(context_length=config_len),
        )
        assert mgr.resolve_context_length() == case["expected_resolved"], case["name"]


def test_fixture_trigger_cases() -> None:
    cases = _fixture_cases()["trigger_cases"]
    assert isinstance(cases, list) and cases
    for case in cases:
        mgr = StandardContextManager(
            model=StubModel(context_window=128_000),
            compaction=CompactionConfig(threshold=case["threshold"]),
        )
        st = SessionState(
            session_id=SessionId("s1"),
            task_id=TaskId("t1"),
            task_instruction="x",
            window_limit=case["window_limit"],
            token_budget_used=case["token_budget_used"],
        )
        assert mgr.should_compact(st) is case["expected_should_compact"], case["name"]


# Keep the import used so ruff does not flag it; dataclasses.fields documents
# that context_length is a real field on the frozen public dataclass.
def test_context_length_is_a_real_field() -> None:
    names = {f.name for f in dataclasses.fields(CompactionConfig)}
    assert "context_length" in names
