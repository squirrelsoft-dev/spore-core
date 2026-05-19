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

import json as _json
from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .agent import AgentError
from .harness import (
    BudgetLimits,
    BudgetLimitTypeT,
    BudgetSnapshot,
    CommandOutput,
    SandboxProvider,
    SessionId,
    SessionState,
    TaskId,
)
from .middleware import HookPoint
from .model import (
    Message,
    ModelInterface,
    ModelParams,
    ModelRequest,
    Role,
    TextBlock,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
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
    workspace_root: Path = Field(default_factory=Path)
    """Populated by the harness from
    :meth:`SandboxProvider.workspace_root` so checks like
    :class:`FeatureListCheck` can resolve workspace-relative paths without
    being given a sandbox handle."""


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


#: Spec alias from issue #43. Returns ``None`` immediately — the task is
#: considered done the moment the agent claims done. Use for single-turn
#: tasks where the model's self-assessment is sufficient.
AlwaysComplete = NullCompletionCheck


# ============================================================================
# FeatureListCheck (issue #43)
# ============================================================================


class FeatureListCheck:
    """Reads ``feature_list.json`` under the snapshot's ``workspace_root``.

    Returns a reason string with the list of incomplete feature names if
    any entry has ``passes: false``. Returns ``None`` when all entries pass.

    File schema: a JSON array of ``{"name": str, "passes": bool}``. Missing
    or unreadable file → ``"<path> missing"`` (treated as incomplete so the
    agent learns to create it).
    """

    def __init__(self, path: Path | str = "feature_list.json") -> None:
        self.path: Path = Path(path)

    async def check(self, state: SessionStateSnapshot) -> str | None:
        full = self.path if self.path.is_absolute() else state.workspace_root / self.path
        try:
            raw = full.read_text()
        except OSError:
            return f"{self.path} missing"
        try:
            entries = _json.loads(raw)
        except _json.JSONDecodeError as e:
            return f"{self.path} invalid JSON: {e}"
        if not isinstance(entries, list):
            return f"{self.path} invalid JSON: expected list"
        incomplete: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                return f"{self.path} invalid JSON: entry not an object"
            if not entry.get("passes", False):
                name = entry.get("name", "<unnamed>")
                incomplete.append(str(name))
        if not incomplete:
            return None
        return f"incomplete features: {', '.join(incomplete)}"

    def description(self) -> str:
        return f"feature list at {self.path}"


# ============================================================================
# TestSuiteCheck (issue #43)
# ============================================================================


class TestSuiteCheck:
    """Runs an external test command via the injected
    :class:`SandboxProvider`. Returns ``None`` if exit code is 0, otherwise a
    failure summary containing the trailing portion of stderr/stdout so the
    next turn knows what failed.

    ``command`` is parsed shell-style: the first whitespace-separated token
    is the program, the remainder become args. For more complex invocations,
    callers should build a wrapper script and invoke it instead.
    """

    __test__ = False  # not a pytest collection target

    def __init__(
        self,
        command: str,
        working_dir: Path | str,
        timeout: float,
        sandbox: SandboxProvider,
    ) -> None:
        self.command = command
        self.working_dir = Path(working_dir)
        self.timeout = timeout
        self.sandbox = sandbox

    async def check(self, state: SessionStateSnapshot) -> str | None:
        parts = self.command.split()
        if not parts:
            return "empty test command"
        program, *args = parts
        try:
            out: CommandOutput = await self.sandbox.execute_command(
                program,
                args,
                working_dir=self.working_dir,
                timeout=self.timeout,
            )
        except Exception as e:  # noqa: BLE001 — mirror Rust's catch-all on sandbox refusal
            return f"sandbox refused test command: {e!r}"
        if out.exit_code == 0 and not out.timed_out:
            return None
        tail = _tail_lines(out.stderr, 20)
        if not tail.strip():
            tail = _tail_lines(out.stdout, 20)
        return f"test suite failed (exit {out.exit_code}, timed_out={out.timed_out}):\n{tail}"

    def description(self) -> str:
        return f"test suite: `{self.command}` in {self.working_dir}"


def _tail_lines(s: str, n: int) -> str:
    lines = s.splitlines()
    start = max(0, len(lines) - n)
    return "\n".join(lines[start:])


# ============================================================================
# QuestionAnsweredCheck (issue #43)
# ============================================================================


class QuestionAnsweredCheck:
    """LLM-as-judge: asks a judge model whether the agent's final response
    actually answered the original question.

    Takes a :class:`ModelInterface` directly (not a ``ModelConfig``).
    """

    def __init__(
        self,
        judge: ModelInterface,
        original_question: str,
        rubric: str | None = None,
    ) -> None:
        self.judge = judge
        self.original_question = original_question
        self.rubric = rubric

    def with_rubric(self, rubric: str) -> QuestionAnsweredCheck:
        self.rubric = rubric
        return self

    async def check(self, state: SessionStateSnapshot) -> str | None:
        agent_response = _last_assistant_text(state.state.messages) or "<no agent response>"
        rubric_clause = f"\n\nRubric:\n{self.rubric}" if self.rubric else ""
        user_text = (
            f"Question:\n{self.original_question}\n\n"
            f"Agent's final response:\n{agent_response}\n\n"
            "Did the agent's response answer the question? Reply with the first "
            "line `ANSWERED: YES` or `ANSWERED: NO`, then a brief reason on the "
            f"next line.{rubric_clause}"
        )
        req = ModelRequest(
            messages=[
                Message(
                    role=Role.SYSTEM,
                    content=TextContent(
                        text=(
                            "You are an evaluation judge. Reply with `ANSWERED: YES` or "
                            "`ANSWERED: NO` on the first line, no other prefix."
                        ),
                    ),
                ),
                Message(role=Role.USER, content=TextContent(text=user_text)),
            ],
            tools=[],
            params=ModelParams(),
            stream=False,
        )
        try:
            resp = await self.judge.call(req)
        except Exception as e:  # noqa: BLE001 — judge errors become incomplete reasons
            return f"judge model error: {e}"
        verdict = ""
        for block in resp.content:
            if isinstance(block, TextBlock):
                verdict = block.text
                break
        first = verdict.splitlines()[0].strip().upper() if verdict else ""
        if first.startswith("ANSWERED: YES"):
            return None
        return f"judge says not answered: {verdict}"

    def description(self) -> str:
        return f"LLM-judge: did the response answer `{self.original_question}`"


def _last_assistant_text(messages: list[Message]) -> str | None:
    for m in reversed(messages):
        if m.role == Role.ASSISTANT and isinstance(m.content, TextContent):
            return m.content.text
    return None


# ============================================================================
# SqlResultCheck (issue #43)
# ============================================================================


class SqlResultCheck:
    """Validates the most recent SQL tool result in the session.

    Scans ``state.state.messages`` in reverse for the last
    :class:`ToolResultContent` whose matching :class:`ToolCallContent` has
    ``name == sql_tool_name``, then parses the result content as
    ``{"columns": [str], "rows": [[any]]}``.

    Returns ``None`` when the result satisfies all configured constraints.
    Returns a reason string when no SQL result was found, parsing failed,
    or a constraint was violated.
    """

    def __init__(
        self,
        sql_tool_name: str = "execute_sql",
        expected_columns: list[str] | None = None,
        min_rows: int | None = None,
    ) -> None:
        self.sql_tool_name = sql_tool_name
        self.expected_columns = expected_columns
        self.min_rows = min_rows

    def with_tool_name(self, name: str) -> SqlResultCheck:
        self.sql_tool_name = name
        return self

    def with_expected_columns(self, cols: list[str]) -> SqlResultCheck:
        self.expected_columns = list(cols)
        return self

    def with_min_rows(self, n: int) -> SqlResultCheck:
        self.min_rows = n
        return self

    async def check(self, state: SessionStateSnapshot) -> str | None:
        # Build id -> tool_name map from ToolCalls.
        id_to_name: dict[str, str] = {}
        for m in state.state.messages:
            if isinstance(m.content, ToolCallContent):
                id_to_name[m.content.id] = m.content.name
        # Find most recent ToolResult matching sql_tool_name.
        raw: str | None = None
        for m in reversed(state.state.messages):
            if isinstance(m.content, ToolResultContent):
                name = id_to_name.get(m.content.tool_use_id)
                if name == self.sql_tool_name:
                    raw = m.content.content
                    break
        if raw is None:
            return f"no `{self.sql_tool_name}` tool result found in session"
        try:
            payload = _json.loads(raw)
        except _json.JSONDecodeError as e:
            return f"sql result is not JSON: {e}"
        if not isinstance(payload, dict):
            return "sql result is not JSON: expected object"
        columns = payload.get("columns", []) or []
        rows = payload.get("rows", []) or []
        if not isinstance(columns, list) or not isinstance(rows, list):
            return "sql result is not JSON: columns/rows must be arrays"
        if self.expected_columns is not None and list(columns) != self.expected_columns:
            return f"sql columns mismatch: expected {self.expected_columns!r}, got {columns!r}"
        min_rows = self.min_rows if self.min_rows is not None else 1
        if len(rows) < min_rows:
            return f"sql result has {len(rows)} rows, expected at least {min_rows}"
        return None

    def description(self) -> str:
        return f"sql result check on tool `{self.sql_tool_name}`"


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
    "AlwaysComplete",
    "BudgetValue",
    "BudgetValueDuration",
    "BudgetValueTokens",
    "BudgetValueTurns",
    "BudgetValueUsd",
    "CompletionCheck",
    "FeatureListCheck",
    "FixedCompletionCheck",
    "NullCompletionCheck",
    "QuestionAnsweredCheck",
    "SqlResultCheck",
    "TestSuiteCheck",
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
