"""Output-schema delivery + enforcement at the harness level (issue #139).

Mirrors the ``os_*`` tests in ``rust/crates/spore-core/src/harness.rs``:

- AC1: the resolved schema is DELIVERED to the leaf's directive seed (key-sorted)
  AND set on ``ModelParams.output_schema`` (the Ollama ``format`` population +
  non-Ollama no-op are unit-tested in ``test_ollama.py``).
- AC2: terminal validated; valid ⇒ Success; invalid ⇒ feed the frozen error back
  + retry within budget.
- AC3: after N retries WITH budget remaining ⇒ OutputSchemaViolation (distinct
  from budget exhaustion); AND budget-precedence (a retry that would exceed the
  turn cap surfaces ``BudgetExceeded``).
- AC4: flag OFF ⇒ an invalid terminal is accepted as Success.
"""

from __future__ import annotations

from typing import Any

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    EscalationModeAutonomous,
    FinalResponse,
    HaltReasonBudgetExceeded,
    HaltReasonOutputSchemaViolation,
    HarnessConfig,
    HarnessRunOptions,
    MockAgent,
    ReactConfig,
    RunResultFailure,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardCompactionAdapter,
    StandardContextManager,
    StandardHarness,
    StreamOutputSchemaRetry,
    StreamOutputSchemaViolation,
    Task,
    TokenUsage,
)
from spore_core.agent import Context, TurnResult
from spore_core.context import CompactionConfig
from spore_core.execution_registry import ExecutionRegistry
from spore_core.harness import HarnessStreamEvent
from spore_core.model import ProviderInfo, Role, TextContent

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _os_schema() -> dict[str, Any]:
    """The output schema the #139 tests enforce: an object requiring a ``status``
    (one of ``ok``/``error``) and a ``count`` integer."""
    return {
        "type": "object",
        "required": ["status", "count"],
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "count": {"type": "integer"},
        },
    }


#: A valid terminal body for the schema.
OS_VALID = '{"status":"ok","count":3}'
#: An invalid terminal body (missing both required props).
OS_INVALID = "{}"

#: The exact KEY-SORTED canonical schema bytes the directive must carry — pinned
#: so all four ports match. Also the ``schema`` reported by the violation.
EXPECTED_SCHEMA = (
    '{"properties":{"count":{"type":"integer"},'
    '"status":{"enum":["ok","error"],"type":"string"}},'
    '"required":["status","count"],"type":"object"}'
)

#: The exact frozen feedback for the first-failing rule (missing required
#: ``status``, array order).
EXPECTED_FEEDBACK = (
    "Your previous response did not match the required output schema. "
    'Missing required property "status". '
    "Reply with only a JSON value that satisfies the schema."
)


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=1, output_tokens=1)


def _stub_provider() -> ProviderInfo:
    return ProviderInfo(name="stub", model_id="stub", context_window=200_000)


class _StubModel:
    """Minimal ``ModelInterface`` for the StandardContextManager assemble path."""

    async def call(self, request: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    async def call_streaming(self, request: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    async def count_tokens(self, request: object) -> int:
        return 0

    def provider(self) -> ProviderInfo:
        return _stub_provider()


def _rich_adapter() -> StandardCompactionAdapter:
    """A real context manager so seeded user messages (directive + feedback) land
    in ``session_state.messages`` exactly like a live run."""
    cfg = CompactionConfig(
        threshold=0.80,
        preserve_recent_n=2,
        head_tail_tokens=64,
        max_compaction_attempts=2,
    )
    return StandardCompactionAdapter(StandardContextManager(_StubModel(), compaction=cfg))


def _registry_with_schema() -> ExecutionRegistry:
    return ExecutionRegistry.builder().schema("", _os_schema()).build()


def _os_config(
    agent: object,
    *,
    enforce: bool = True,
    max_retries: int = 2,
) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,  # type: ignore[arg-type]
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=_rich_adapter(),
        termination_policy=AlwaysContinuePolicy(),
        enforce_output_schemas=enforce,
        output_schema_max_retries=max_retries,
        registry=_registry_with_schema(),
        # Mirror the Rust ``standard_config`` (Autonomous): a budget escalation
        # propagates as a ``BudgetExceeded`` failure, not a HITL pause — so the
        # budget-precedence test surfaces the budget terminal, not WaitingForHuman.
        escalation_mode=EscalationModeAutonomous(),
    )


def _os_leaf(budget: int = 10) -> Task:
    """A bare ReAct leaf carrying ``output = ""`` and a turn budget."""
    return Task.new(
        "produce a status report",
        SessionId("s1"),
        ReactConfig(
            budget=ReactConfig.per_loop(budget).budget,
            agent="",
            toolset="",
            output="",
        ),
    )


def _user_texts(state: SessionState) -> list[str]:
    return [
        m.content.text
        for m in state.messages
        if m.role == Role.USER and isinstance(m.content, TextContent)
    ]


class _CapturingAgent:
    """Records every ``Context`` it sees and returns a scripted queue of
    ``FinalResponse`` contents in order."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.seen: list[Context] = []

    async def turn(self, context: Context) -> TurnResult:
        self.seen.append(context)
        content = self._contents.pop(0)
        return FinalResponse(content=content, usage=_usage())

    def id(self) -> AgentId:
        return AgentId("capture")


# ---------------------------------------------------------------------------
# AC1 — schema delivered to the directive seed + ModelParams.output_schema
# ---------------------------------------------------------------------------


async def test_os_ac1_schema_delivered_to_directive_seed_and_params() -> None:
    agent = _CapturingAgent([OS_VALID])
    h = StandardHarness(_os_config(agent, max_retries=2))
    result = await h.run(HarnessRunOptions(_os_leaf(10)))
    assert isinstance(result, RunResultSuccess), f"expected Success, got {result!r}"
    assert result.turns == 1, "valid on turn 1"

    # The directive carries the KEY-SORTED schema bytes (pinned across ports).
    users = _user_texts(result.session_state)
    assert any(EXPECTED_SCHEMA in m for m in users), (
        f"directive must carry the key-sorted schema; got {users!r}"
    )

    # The schema also reaches the model's constrained-decoding channel.
    assert agent.seen, "agent must have been invoked"
    assert agent.seen[0].params.output_schema == _os_schema()


async def test_os_ac1_non_enforced_leaves_params_clean() -> None:
    # Flag OFF ⇒ params.output_schema is never set (non-Ollama-ignored is unit
    # tested in test_ollama; here we confirm the harness does not deliver it).
    agent = _CapturingAgent([OS_INVALID])
    h = StandardHarness(_os_config(agent, enforce=False))
    result = await h.run(HarnessRunOptions(_os_leaf(10)))
    assert isinstance(result, RunResultSuccess)
    assert agent.seen[0].params.output_schema is None


# ---------------------------------------------------------------------------
# AC2 — accept on first turn; retry on invalid then accept
# ---------------------------------------------------------------------------


async def test_os_ac2_accept_valid_on_first_turn() -> None:
    a = MockAgent(AgentId("test"))
    a.push(FinalResponse(content=OS_VALID, usage=_usage()))
    h = StandardHarness(_os_config(a, max_retries=2))
    result = await h.run(HarnessRunOptions(_os_leaf(10)))
    assert isinstance(result, RunResultSuccess)
    assert result.output == OS_VALID
    assert result.turns == 1
    fed_back = any(
        "did not match the required output schema" in m for m in _user_texts(result.session_state)
    )
    assert not fed_back, "no feedback message on a turn-1 accept"


async def test_os_ac2_retry_invalid_then_valid() -> None:
    a = MockAgent(AgentId("test"))
    a.push(FinalResponse(content=OS_INVALID, usage=_usage()))
    a.push(FinalResponse(content=OS_VALID, usage=_usage()))
    h = StandardHarness(_os_config(a, max_retries=2))
    result = await h.run(HarnessRunOptions(_os_leaf(10)))
    assert isinstance(result, RunResultSuccess), f"expected Success, got {result!r}"
    assert result.output == OS_VALID
    assert result.turns == 2, "one retry consumed"
    # The exact frozen feedback (with the validator error) must be fed back.
    assert EXPECTED_FEEDBACK in _user_texts(result.session_state), (
        "the exact frozen feedback (with validator error) must be fed back"
    )


# ---------------------------------------------------------------------------
# AC3 — typed violation after retries exhausted; budget precedence
# ---------------------------------------------------------------------------


async def test_os_ac3_fail_after_retries_exhausted() -> None:
    # N == 2 ⇒ 3 attempts; a generous budget so the violation (not budget) fires.
    a = MockAgent(AgentId("test"))
    for _ in range(3):
        a.push(FinalResponse(content=OS_INVALID, usage=_usage()))
    h = StandardHarness(_os_config(a, max_retries=2))
    result = await h.run(HarnessRunOptions(_os_leaf(50)))
    assert isinstance(result, RunResultFailure), f"expected Failure, got {result!r}"
    assert isinstance(result.reason, HaltReasonOutputSchemaViolation), (
        f"expected OutputSchemaViolation, got {result.reason!r}"
    )
    assert result.reason.attempts == 3, "1 + N == 1 + 2 attempts"
    assert result.turns == 3, "exactly 1 + N turns"
    assert result.turns < 50, "budget NOT exhausted (distinct from budget)"
    assert result.reason.last_error == 'Missing required property "status".'
    assert result.reason.schema == EXPECTED_SCHEMA


async def test_os_ac3_budget_precedence_over_schema_violation() -> None:
    # N == 5 (large), budget == 2 turns. After 2 invalid terminals the 3rd retry
    # re-enters the loop where the turn-budget gate fires FIRST — budget wins.
    a = MockAgent(AgentId("test"))
    for _ in range(5):
        a.push(FinalResponse(content=OS_INVALID, usage=_usage()))
    h = StandardHarness(_os_config(a, max_retries=5))
    result = await h.run(HarnessRunOptions(_os_leaf(2)))
    assert isinstance(result, RunResultFailure), f"expected Failure, got {result!r}"
    assert isinstance(result.reason, HaltReasonBudgetExceeded), (
        f"expected BudgetExceeded (budget wins), got {result.reason!r}"
    )
    assert result.turns == 2, "stopped at the turn budget, not on schema"


# ---------------------------------------------------------------------------
# AC4 — flag OFF accepts an invalid terminal as Success
# ---------------------------------------------------------------------------


async def test_os_ac4_flag_off_accepts_invalid_terminal() -> None:
    # The schema is registered (so validate passes) but enforcement is OFF — the
    # invalid terminal must be accepted as-is, with no directive delivered.
    a = MockAgent(AgentId("test"))
    a.push(FinalResponse(content=OS_INVALID, usage=_usage()))
    h = StandardHarness(_os_config(a, enforce=False))
    result = await h.run(HarnessRunOptions(_os_leaf(10)))
    assert isinstance(result, RunResultSuccess), f"expected Success (gate OFF), got {result!r}"
    assert result.output == OS_INVALID, "invalid terminal accepted as-is"
    assert result.turns == 1
    seeded = any("conforms to this" in m for m in _user_texts(result.session_state))
    assert not seeded, "no schema directive delivered when gate is OFF"


# ---------------------------------------------------------------------------
# Stream events: enforcement ON emits the retry + violation events
# ---------------------------------------------------------------------------


async def test_os_emits_retry_and_violation_events() -> None:
    a = MockAgent(AgentId("test"))
    for _ in range(2):
        a.push(FinalResponse(content=OS_INVALID, usage=_usage()))
    captured: list[HarnessStreamEvent] = []
    # N == 1 ⇒ 1 retry then violation (2 attempts).
    h = StandardHarness(_os_config(a, max_retries=1))
    await h.run(HarnessRunOptions(_os_leaf(50), on_stream=captured.append))

    retries = [e.attempt for e in captured if isinstance(e, StreamOutputSchemaRetry)]
    violations = [e.attempts for e in captured if isinstance(e, StreamOutputSchemaViolation)]
    assert retries == [1], "one retry event at attempt 1"
    assert violations == [2], "one violation event, attempts == 2"
