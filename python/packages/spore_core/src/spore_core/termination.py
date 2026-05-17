"""TerminationPolicy — evaluate after each turn whether to continue, halt
with success, halt with failure, or halt because a budget limit was breached
(issue #13).

Mirrors the Rust reference at ``rust/crates/spore-core/src/termination.rs``.

See ``docs/harness-engineering-concepts.md`` § "TerminationPolicy" for the
authoritative rules. This module ships:

* The full :class:`TerminationDecision` / :class:`TerminationFailureReason`
  / :class:`BudgetValue` surface from the spec.
* The :class:`CompletionCheck` Protocol and standard checks
  (:class:`NullCompletionCheck`, :class:`FixedCompletionCheck`).
* :class:`StandardTerminationPolicy` — the reference policy that runs budget
  first, then sensor halts, then the injected :class:`CompletionCheck`.

Rules enforced:

* ``agent_claims_done`` is **one input**, not the decision.
* Budget limits are unconditional hard stops — evaluated before anything
  else and regardless of ``agent_claims_done``.
* :class:`TerminationHaltFailure` carries a typed
  :class:`TerminationFailureReason`; it cannot be a free string.
* The :class:`CompletionCheck` is injected at construction time — the
  policy itself is domain-agnostic.
* If ``not agent_claims_done``, always :class:`TerminationContinueDecision`
  (after the budget check).
* When ``agent_claims_done``, any sensor result with
  :class:`SensorOutcome.HALT` becomes
  :class:`TerminationFailureUnrecoverableSensorHalt`.
* :meth:`CompletionCheck.check` returning a string ⇒
  :class:`TerminationContinueDecision` (the harness re-injects the reason).
* :meth:`CompletionCheck.check` returning ``None`` ⇒
  :class:`TerminationHaltSuccess` using the agent's last response as
  ``summary`` (empty string if absent).
* ``HumanHalted`` is reserved for the harness; the policy never produces
  it. (Captured by
  :class:`TerminationFailureHumanHalted` for completeness of the public
  type.)
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .agent import AgentError
from .harness import (
    BudgetLimits,
    BudgetLimitTypeT,
    BudgetSnapshot,
    SessionId,
    SessionState,
    TaskId,
)
from .middleware import HookPoint
from .sensor import SensorId, SensorOutcome, SensorResult

# ============================================================================
# Pydantic base
# ============================================================================


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# BudgetValue (discriminated union on ``kind``)
# ============================================================================


class BudgetValueTurns(_Model):
    kind: Literal["turns"] = "turns"
    value: int


class BudgetValueTokens(_Model):
    kind: Literal["tokens"] = "tokens"
    value: int


class BudgetValueDuration(_Model):
    """Duration in whole seconds — mirrors Rust's ``duration_secs`` adapter."""

    kind: Literal["duration"] = "duration"
    value: int


class BudgetValueUsd(_Model):
    kind: Literal["usd"] = "usd"
    value: float


BudgetValue = Annotated[
    BudgetValueTurns | BudgetValueTokens | BudgetValueDuration | BudgetValueUsd,
    Field(discriminator="kind"),
]


def budget_value_turns(v: int) -> BudgetValueTurns:
    return BudgetValueTurns(value=v)


def budget_value_tokens(v: int) -> BudgetValueTokens:
    return BudgetValueTokens(value=v)


def budget_value_duration(v: int) -> BudgetValueDuration:
    return BudgetValueDuration(value=v)


def budget_value_usd(v: float) -> BudgetValueUsd:
    return BudgetValueUsd(value=v)


# ============================================================================
# SessionStateSnapshot
# ============================================================================


class SessionStateSnapshot(_Model):
    """Read-only snapshot of session state handed to
    :meth:`CompletionCheck.check`.

    Wraps :class:`SessionState` so the policy can identify the source
    session and task — completion checks frequently key into per-session
    scratchpads (e.g. ``feature_list.json`` under ``.spore/<session>/``).
    """

    session_id: SessionId
    task_id: TaskId
    state: SessionState = Field(default_factory=SessionState)


# ============================================================================
# TerminationFailureReason (discriminated union on ``kind``)
# ============================================================================


class TerminationFailureCompletionCheckFailed(_Model):
    kind: Literal["completion_check_failed"] = "completion_check_failed"
    detail: str


class TerminationFailureMaxRetriesExhausted(_Model):
    kind: Literal["max_retries_exhausted"] = "max_retries_exhausted"
    tool: str
    attempts: int


class TerminationFailureUnrecoverableSensorHalt(_Model):
    kind: Literal["unrecoverable_sensor_halt"] = "unrecoverable_sensor_halt"
    sensor_id: SensorId
    detail: str


class TerminationFailureMiddlewareHalt(_Model):
    kind: Literal["middleware_halt"] = "middleware_halt"
    hook: HookPoint
    reason: str


class TerminationFailureAgentError(_Model):
    kind: Literal["agent_error"] = "agent_error"
    error: AgentError


class TerminationFailurePolicyViolation(_Model):
    kind: Literal["policy_violation"] = "policy_violation"
    detail: str


class TerminationFailureHumanHalted(_Model):
    """Set by the harness when ``HumanResponse::Halt`` is received; the
    policy never produces this variant."""

    kind: Literal["human_halted"] = "human_halted"


TerminationFailureReason = Annotated[
    TerminationFailureCompletionCheckFailed
    | TerminationFailureMaxRetriesExhausted
    | TerminationFailureUnrecoverableSensorHalt
    | TerminationFailureMiddlewareHalt
    | TerminationFailureAgentError
    | TerminationFailurePolicyViolation
    | TerminationFailureHumanHalted,
    Field(discriminator="kind"),
]


# ============================================================================
# TerminationDecision (discriminated union on ``kind``)
# ============================================================================


class TerminationContinueDecision(_Model):
    kind: Literal["continue"] = "continue"


class TerminationHaltSuccess(_Model):
    kind: Literal["halt_success"] = "halt_success"
    summary: str


class TerminationHaltFailure(_Model):
    kind: Literal["halt_failure"] = "halt_failure"
    reason: TerminationFailureReason


class TerminationHaltBudgetExceeded(_Model):
    kind: Literal["halt_budget_exceeded"] = "halt_budget_exceeded"
    limit_type: BudgetLimitTypeT
    used: BudgetValue
    limit: BudgetValue


TerminationDecision = Annotated[
    TerminationContinueDecision
    | TerminationHaltSuccess
    | TerminationHaltFailure
    | TerminationHaltBudgetExceeded,
    Field(discriminator="kind"),
]


# ============================================================================
# TerminationInput
# ============================================================================


class TerminationInput(_Model):
    session_id: SessionId
    task_id: TaskId
    turn_number: int
    agent_claims_done: bool
    agent_response: str | None = None
    budget_used: BudgetSnapshot
    budget_limits: BudgetLimits = Field(default_factory=BudgetLimits)
    sensor_results: list[SensorResult] = Field(default_factory=list)
    session_state: SessionStateSnapshot


# ============================================================================
# CompletionCheck Protocol + standard impls
# ============================================================================


@runtime_checkable
class CompletionCheck(Protocol):
    """Pluggable domain-specific completion check.

    Returns ``None`` if complete, a reason string if not yet done. The
    harness injects the reason into the next turn's context when a string
    is returned.
    """

    async def check(self, state: SessionStateSnapshot) -> str | None: ...

    def description(self) -> str: ...


class NullCompletionCheck:
    """Always-complete check. Causes the policy to halt with success the
    moment the agent claims done."""

    async def check(self, state: SessionStateSnapshot) -> str | None:
        return None

    def description(self) -> str:
        return "null (always complete)"


class FixedCompletionCheck:
    """Test/fixture completion check that returns a configured outcome."""

    def __init__(self, outcome: str | None, label: str) -> None:
        self.outcome = outcome
        self.label = label

    @classmethod
    def complete(cls) -> FixedCompletionCheck:
        return cls(outcome=None, label="fixed:complete")

    @classmethod
    def incomplete(cls, reason: str) -> FixedCompletionCheck:
        return cls(outcome=reason, label="fixed:incomplete")

    async def check(self, state: SessionStateSnapshot) -> str | None:
        return self.outcome

    def description(self) -> str:
        return self.label


# ============================================================================
# Budget check
# ============================================================================


def check_budget_default(
    snapshot: BudgetSnapshot,
    limits: BudgetLimits,
) -> TerminationDecision | None:
    """Default budget check used by :class:`StandardTerminationPolicy` and
    exposed for direct use by the harness loop. Returns
    :class:`TerminationHaltBudgetExceeded` if any limit is breached, else
    ``None``."""
    if limits.max_turns is not None and snapshot.turns >= limits.max_turns:
        return TerminationHaltBudgetExceeded(
            limit_type="turns",
            used=budget_value_turns(snapshot.turns),
            limit=budget_value_turns(limits.max_turns),
        )
    if limits.max_input_tokens is not None and snapshot.input_tokens >= limits.max_input_tokens:
        return TerminationHaltBudgetExceeded(
            limit_type="input_tokens",
            used=budget_value_tokens(snapshot.input_tokens),
            limit=budget_value_tokens(limits.max_input_tokens),
        )
    if limits.max_output_tokens is not None and snapshot.output_tokens >= limits.max_output_tokens:
        return TerminationHaltBudgetExceeded(
            limit_type="output_tokens",
            used=budget_value_tokens(snapshot.output_tokens),
            limit=budget_value_tokens(limits.max_output_tokens),
        )
    if limits.max_wall_time is not None:
        used = snapshot.wall_time if snapshot.wall_time is not None else 0
        if used >= limits.max_wall_time:
            return TerminationHaltBudgetExceeded(
                limit_type="wall_time",
                used=budget_value_duration(used),
                limit=budget_value_duration(limits.max_wall_time),
            )
    if limits.max_cost_usd is not None and snapshot.cost_usd >= limits.max_cost_usd:
        return TerminationHaltBudgetExceeded(
            limit_type="cost_usd",
            used=budget_value_usd(snapshot.cost_usd),
            limit=budget_value_usd(limits.max_cost_usd),
        )
    return None


# ============================================================================
# TerminationPolicy Protocol
# ============================================================================


@runtime_checkable
class TerminationPolicy(Protocol):
    async def evaluate(self, input: TerminationInput) -> TerminationDecision: ...

    def check_budget(
        self,
        snapshot: BudgetSnapshot,
        limits: BudgetLimits,
    ) -> TerminationDecision | None: ...


# ============================================================================
# StandardTerminationPolicy
# ============================================================================


class StandardTerminationPolicy:
    """Reference :class:`TerminationPolicy`. Runs:

    1. Budget check (unconditional).
    2. Continue if ``not input.agent_claims_done``.
    3. ``UnrecoverableSensorHalt`` if any sensor returned
       :class:`SensorOutcome.HALT`.
    4. The injected :class:`CompletionCheck`.
    """

    def __init__(self, completion_check: CompletionCheck) -> None:
        self._check = completion_check

    @classmethod
    def with_null_check(cls) -> StandardTerminationPolicy:
        return cls(NullCompletionCheck())

    @property
    def completion_check(self) -> CompletionCheck:
        return self._check

    def check_budget(
        self,
        snapshot: BudgetSnapshot,
        limits: BudgetLimits,
    ) -> TerminationDecision | None:
        return check_budget_default(snapshot, limits)

    async def evaluate(self, input: TerminationInput) -> TerminationDecision:
        halt = self.check_budget(input.budget_used, input.budget_limits)
        if halt is not None:
            return halt
        if not input.agent_claims_done:
            return TerminationContinueDecision()
        for r in input.sensor_results:
            if r.outcome == SensorOutcome.HALT:
                return TerminationHaltFailure(
                    reason=TerminationFailureUnrecoverableSensorHalt(
                        sensor_id=r.sensor_id,
                        detail=r.detail,
                    )
                )
        result = await self._check.check(input.session_state)
        if result is None:
            return TerminationHaltSuccess(
                summary=input.agent_response if input.agent_response is not None else "",
            )
        return TerminationContinueDecision()


__all__ = [
    "BudgetValue",
    "BudgetValueDuration",
    "BudgetValueTokens",
    "BudgetValueTurns",
    "BudgetValueUsd",
    "CompletionCheck",
    "FixedCompletionCheck",
    "NullCompletionCheck",
    "SessionStateSnapshot",
    "StandardTerminationPolicy",
    "TerminationContinueDecision",
    "TerminationDecision",
    "TerminationFailureAgentError",
    "TerminationFailureCompletionCheckFailed",
    "TerminationFailureHumanHalted",
    "TerminationFailureMaxRetriesExhausted",
    "TerminationFailureMiddlewareHalt",
    "TerminationFailurePolicyViolation",
    "TerminationFailureReason",
    "TerminationFailureUnrecoverableSensorHalt",
    "TerminationHaltBudgetExceeded",
    "TerminationHaltFailure",
    "TerminationHaltSuccess",
    "TerminationInput",
    "TerminationPolicy",
    "budget_value_duration",
    "budget_value_tokens",
    "budget_value_turns",
    "budget_value_usd",
    "check_budget_default",
]
