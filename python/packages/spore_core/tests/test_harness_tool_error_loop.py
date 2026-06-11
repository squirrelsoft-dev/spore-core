"""Tests for the ReAct consecutive-recoverable-tool-error breaker (issue #137).

Mirrors the ``tel_*`` unit tests in ``rust/crates/spore-core/src/harness.rs``.
Each test exercises one acceptance criterion:

- AC1: a success resets the per-tool counter; an args-change starts a fresh run.
- AC2: at ``N`` identical-argument recoverable errors, exactly ONE corrective
  USER message (the ``enrich_tool_error`` schema+hint) is injected; a later
  identical error does NOT re-inject.
- AC3: at ``2 * N`` the loop stops and resolves the node's
  ``BudgetExhaustedBehavior`` — ``Fail`` / ``Escalate`` / ``Continue`` — carrying
  ``HaltReasonToolErrorLoop`` (never ``BudgetExceeded``), without burning the
  rest of the budget.
- AC4: a ``ToolErrorLoopDetected`` stream + obs pair at ``N`` and a
  ``ToolErrorLoopBroken`` pair at ``2 * N``, each carrying tool + count.
"""

from __future__ import annotations

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetExhaustedContinue,
    BudgetExhaustedEscalate,
    BudgetExhaustedFail,
    EscalationModeAutonomous,
    EscalationModeSurfaceToHuman,
    FinalResponse,
    HaltReasonToolErrorLoop,
    HarnessConfig,
    HarnessRunOptions,
    InMemoryObservabilityProvider,
    MockAgent,
    NoopContextManager,
    ReactConfig,
    RunResultFailure,
    RunResultSuccess,
    RunResultWaitingForHuman,
    SessionId,
    SessionState,
    StandardHarness,
    StreamToolErrorLoopBroken,
    StreamToolErrorLoopDetected,
    Task,
    TokenUsage,
    ToolCall,
    ToolCallRequested,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
    ToolSchema,
)
from spore_core.harness import HarnessStreamEvent
from spore_core.model import Message, Role, TextContent
from spore_core.observability import (
    ContextOperationToolErrorLoopBroken,
    ContextOperationToolErrorLoopDetected,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_TEL_BAD_MSG = "missing required parameter `description`"
_ADD_TASK_SCHEMA = ToolSchema(
    name="add_task",
    description="add a task to a task list",
    input_schema={
        "type": "object",
        "properties": {
            "task_list_id": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["task_list_id", "description"],
    },
)

# The EXACT schema JSON the corrective message must carry — key-sorted compact
# form, byte-identical to Rust's ``serde_json`` BTreeMap serialization (#137
# cross-language consistency). ``required`` element order is preserved (only
# object keys are sorted, not array elements).
_EXPECTED_SCHEMA_JSON = (
    '{"properties":{"description":{"type":"string"},'
    '"task_list_id":{"type":"string"}},'
    '"required":["task_list_id","description"],"type":"object"}'
)


class _ErrRegistry:
    """A tool registry that always returns the same recoverable error and
    advertises the ``add_task`` schema (so ``_enrich_tool_error`` can render the
    schema+hint). Mirrors the Rust ``tel_err_registry`` (``always_recoverable_error``
    + ``with_schema``)."""

    def __init__(self, message: str = _TEL_BAD_MSG) -> None:
        self._message = message
        self.call_count = 0

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        self.call_count += 1
        return ToolOutputError(message=self._message, recoverable=True)

    def is_always_halt(self, tool_name: str) -> bool:
        return False

    def schemas(self) -> list[ToolSchema]:
        return [_ADD_TASK_SCHEMA]


class _ScriptedRegistry:
    """A tool registry that replays a queued sequence of outputs (the Rust AC1
    test scripts a [error, error, success, error, error] sequence)."""

    def __init__(self, outputs: list[ToolOutput]) -> None:
        self._outputs = list(outputs)
        self.call_count = 0

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        self.call_count += 1
        if self._outputs:
            return self._outputs.pop(0)
        return ToolOutputSuccess(content="ok")

    def is_always_halt(self, tool_name: str) -> bool:
        return False

    def schemas(self) -> list[ToolSchema]:
        return [_ADD_TASK_SCHEMA]


class _RecordingContextManager:
    """Records appended user messages into ``session.messages`` so AC2 can assert
    what reached history (mirrors the Rust standard config's recording manager,
    where ``tel_user_msgs`` reads ``state.messages``)."""

    async def assemble(self, session: SessionState, task: Task) -> object:
        from spore_core.agent import Context
        from spore_core.model import ModelParams

        _ = task
        return Context(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: object) -> None:
        _ = (session, result)

    async def append_assistant_message(self, session: SessionState, message: Message) -> None:
        session.messages.append(message)

    async def append_user_message(self, session: SessionState, text: str) -> None:
        session.messages.append(Message(role=Role.USER, content=TextContent(text=text)))

    def should_compact(self, session: SessionState) -> bool:
        _ = session
        return False


def _agent() -> MockAgent:
    return MockAgent(AgentId("test"))


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=1, output_tokens=1)


def _bad_args() -> dict[str, str]:
    return {"task_list_id": "tl1"}


def _bad_call(args: dict[str, str] | None = None) -> ToolCall:
    return ToolCall(id="call_bad", name="add_task", input=args if args is not None else _bad_args())


def _push_bad(a: MockAgent, k: int, args: dict[str, str] | None = None) -> None:
    for _ in range(k):
        a.push(ToolCallRequested(reasoning=None, calls=[_bad_call(args)], usage=_usage()))


def _config(
    agent: MockAgent,
    *,
    tool_registry: object,
    error_loop_threshold: int = 3,
    context_manager: object | None = None,
    observability: object | None = None,
    escalation_mode: object | None = None,
) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=tool_registry,  # type: ignore[arg-type]
        sandbox=AllowAllSandbox(),
        context_manager=context_manager if context_manager is not None else NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        observability=observability,  # type: ignore[arg-type]
        error_loop_threshold=error_loop_threshold,
        escalation_mode=escalation_mode,  # type: ignore[arg-type]
    )


def _react_task(max_iter: int = 20) -> Task:
    return Task.new("add a task to the task list", SessionId("s1"), ReactConfig.per_loop(max_iter))


def _leaf_task(behavior: object, budget: int = 50) -> Task:
    return Task.new(
        "add a task to the task list",
        SessionId("s1"),
        ReactConfig(
            budget=ReactConfig.per_loop(budget).budget,
            behavior=behavior,  # type: ignore[arg-type]
            agent="",
            toolset="",
        ),
    )


def _user_texts(state: SessionState) -> list[str]:
    return [
        m.content.text
        for m in state.messages
        if m.role == Role.USER and isinstance(m.content, TextContent)
    ]


# ---------------------------------------------------------------------------
# AC1 — success / args-change resets
# ---------------------------------------------------------------------------


async def test_ac1_success_resets_counter_no_trip() -> None:
    """error, error, SUCCESS, error, error -> 4 errors but the longest
    identical-args run is 2 (< N), so the breaker never trips."""
    a = _agent()
    _push_bad(a, 2)
    a.push(
        ToolCallRequested(
            reasoning=None,
            calls=[ToolCall(id="ok", name="add_task", input=_bad_args())],
            usage=_usage(),
        )
    )
    _push_bad(a, 2)
    a.push(FinalResponse(reasoning=None, content="done", usage=_usage()))
    reg = _ScriptedRegistry(
        [
            ToolOutputError(message=_TEL_BAD_MSG, recoverable=True),
            ToolOutputError(message=_TEL_BAD_MSG, recoverable=True),
            ToolOutputSuccess(content="added"),
            ToolOutputError(message=_TEL_BAD_MSG, recoverable=True),
            ToolOutputError(message=_TEL_BAD_MSG, recoverable=True),
        ]
    )
    h = StandardHarness(_config(a, tool_registry=reg, error_loop_threshold=3))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "done"


async def test_ac1_args_change_starts_fresh_run() -> None:
    """error(argsX), error(argsY) -> the second is a FRESH run at count 1, so two
    different-args errors never trip even at N == 2."""
    a = _agent()
    a.push(
        ToolCallRequested(reasoning=None, calls=[_bad_call({"task_list_id": "X"})], usage=_usage())
    )
    a.push(
        ToolCallRequested(reasoning=None, calls=[_bad_call({"task_list_id": "Y"})], usage=_usage())
    )
    a.push(FinalResponse(reasoning=None, content="stopped trying", usage=_usage()))
    # 2N == 4; longest identical run is 1.
    h = StandardHarness(_config(a, tool_registry=_ErrRegistry(), error_loop_threshold=2))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "stopped trying"


# ---------------------------------------------------------------------------
# AC2 — one corrective at N, no re-inject
# ---------------------------------------------------------------------------


async def test_ac2_injects_one_corrective_at_n() -> None:
    """4 identical errors (N=3 injects at the 3rd; the 4th must not re-inject),
    then a final response so the run ends cleanly (2N would be 6)."""
    a = _agent()
    _push_bad(a, 4)
    a.push(FinalResponse(reasoning=None, content="gave up", usage=_usage()))
    h = StandardHarness(
        _config(
            a,
            tool_registry=_ErrRegistry(),
            error_loop_threshold=3,
            context_manager=_RecordingContextManager(),
        )
    )
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)
    correctives = [m for m in _user_texts(r.session_state) if "Expected parameter schema" in m]
    assert len(correctives) == 1, (
        f"exactly one corrective injected, got {_user_texts(r.session_state)!r}"
    )
    corrective = correctives[0]
    assert _TEL_BAD_MSG in corrective, "carries the bare error"
    # Cross-language consistency (#137): the schema JSON is key-sorted and
    # byte-identical to Rust's serde_json BTreeMap form — assert the EXACT
    # substring, not just a loose key match, so a re-introduced insertion-order
    # serialization can't pass.
    assert "Expected parameter schema: " + _EXPECTED_SCHEMA_JSON in corrective, (
        f"carries the exact key-sorted parameter schema, got {corrective!r}"
    )
    assert "correctly-typed JSON" in corrective, "carries the hint"


# ---------------------------------------------------------------------------
# AC3 — 2N resolves the node behavior with ToolErrorLoop, not BudgetExceeded
# ---------------------------------------------------------------------------


async def test_ac3_fail_terminal_is_tool_error_loop() -> None:
    a = _agent()
    _push_bad(a, 8)  # trip is at 2N == 6.
    h = StandardHarness(_config(a, tool_registry=_ErrRegistry(), error_loop_threshold=3))
    r = await h.run(HarnessRunOptions(_leaf_task(BudgetExhaustedFail(), budget=50)))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonToolErrorLoop)
    assert r.reason.tool == "add_task"
    assert r.reason.consecutive_errors == 6, "2N == 6"
    assert r.turns < 50, f"budget NOT fully burned, got {r.turns}"


async def test_ac3_escalate_surfaces_to_human() -> None:
    a = _agent()
    _push_bad(a, 8)
    h = StandardHarness(
        _config(
            a,
            tool_registry=_ErrRegistry(),
            error_loop_threshold=3,
            escalation_mode=EscalationModeSurfaceToHuman(),
        )
    )
    r = await h.run(HarnessRunOptions(_leaf_task(BudgetExhaustedEscalate(), budget=50)))
    assert isinstance(r, RunResultWaitingForHuman)


async def test_ac3_escalate_autonomous_is_tool_error_loop() -> None:
    a = _agent()
    _push_bad(a, 8)
    h = StandardHarness(
        _config(
            a,
            tool_registry=_ErrRegistry(),
            error_loop_threshold=3,
            escalation_mode=EscalationModeAutonomous(),
        )
    )
    r = await h.run(HarnessRunOptions(_leaf_task(BudgetExhaustedEscalate(), budget=50)))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonToolErrorLoop)
    assert r.reason.tool == "add_task"
    assert r.turns < 50, "budget NOT fully burned"


async def test_ac3_continue_grants_one_window_then_terminal() -> None:
    """With ``Continue { max_continues: 1, on_exhausted: Fail }`` the first 2N
    trip grants a continue (fresh window), the second 2N trip falls through to
    Fail with ToolErrorLoop. A granted Continue resets the per-tool counter."""
    a = _agent()
    _push_bad(a, 14)  # 2N (6) + 2N (6) = 12 needed.
    behavior = BudgetExhaustedContinue(max_continues=1, on_exhausted=BudgetExhaustedFail())
    h = StandardHarness(_config(a, tool_registry=_ErrRegistry(), error_loop_threshold=3))
    r = await h.run(HarnessRunOptions(_leaf_task(behavior, budget=50)))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonToolErrorLoop)
    assert r.turns < 50, f"budget NOT fully burned, got {r.turns}"


# ---------------------------------------------------------------------------
# AC4 — stream + observability pairs at both thresholds
# ---------------------------------------------------------------------------


async def test_ac4_emits_detected_and_broken_events() -> None:
    a = _agent()
    _push_bad(a, 8)
    obs = InMemoryObservabilityProvider()
    captured: list[HarnessStreamEvent] = []
    h = StandardHarness(
        _config(a, tool_registry=_ErrRegistry(), error_loop_threshold=3, observability=obs)
    )
    task = _leaf_task(BudgetExhaustedFail(), budget=50)
    await h.run(HarnessRunOptions(task, on_stream=captured.append))

    detected = [
        e.consecutive_errors
        for e in captured
        if isinstance(e, StreamToolErrorLoopDetected) and e.tool == "add_task"
    ]
    broken = [
        e.consecutive_errors
        for e in captured
        if isinstance(e, StreamToolErrorLoopBroken) and e.tool == "add_task"
    ]
    assert detected == [3], "one detected at N, count == N"
    assert broken == [6], "one broken at 2N, count == 2N"

    spans = obs.context_spans(task.session_id)
    obs_detected = [
        s.operation.consecutive_errors
        for s in spans
        if isinstance(s.operation, ContextOperationToolErrorLoopDetected)
        and s.operation.tool_name == "add_task"
    ]
    obs_broken = [
        s.operation.consecutive_errors
        for s in spans
        if isinstance(s.operation, ContextOperationToolErrorLoopBroken)
        and s.operation.tool_name == "add_task"
    ]
    assert obs_detected == [3], "obs detected at N"
    assert obs_broken == [6], "obs broken at 2N"


# ---------------------------------------------------------------------------
# Breaker disabled when threshold is 0
# ---------------------------------------------------------------------------


async def test_threshold_zero_disables_breaker() -> None:
    a = _agent()
    _push_bad(a, 5)
    a.push(FinalResponse(reasoning=None, content="fin", usage=_usage()))
    h = StandardHarness(_config(a, tool_registry=_ErrRegistry(), error_loop_threshold=0))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "fin"
