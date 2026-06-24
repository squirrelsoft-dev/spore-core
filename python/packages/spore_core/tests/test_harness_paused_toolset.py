"""Issue #140 — ``PausedState`` carries the pausing leaf's toolset handle.

Mirrors the Rust reference tests in ``rust/crates/spore-core/src/harness.rs``
(``paused_state_toolset_back_compat_and_round_trip``,
``consult_pause_carries_leaf_toolset_handle``,
``clarification_pause_carries_leaf_toolset_handle``,
``resume_consult_routes_pending_calls_through_carried_toolset``) and
``rust/crates/spore-core/src/tools/subagent.rs``
(``child_state_from_paused_carries_toolset``).

The bug (verified in Rust): both resume paths resolved tools with the empty
global-catalogue fallback, so pending tool calls from a node that declared a
per-node toolset (Issue 2 scoping) resumed against the WRONG catalogue. The fix
threads the pausing leaf's toolset handle through :class:`PausedState` /
:class:`ChildPausedState`, and the resume paths route through it.

* AC1 (back-compat + round-trip): a paused-state blob WITHOUT a ``toolset`` key
  deserializes to the empty handle; the empty handle ALWAYS serializes (never
  skipped) for byte-parity; a non-empty handle round-trips by value.
* AC2a (populate): the Consult and Clarification leaf-pause paths capture the
  leaf's toolset handle into the returned :class:`PausedState`.
* AC2b (THE load-bearing path): a resume whose carried handle is ``"scoped"``
  dispatches a pending call to a scoped-only tool through the SCOPED catalogue;
  the negative control (empty handle) routes to the global catalogue where that
  tool is unknown → a recoverable error.
* The subagent ``_child_state_from_paused`` carries the child leaf's own handle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetExhaustedEscalate,
    BudgetPolicyPerLoop,
    BudgetSnapshot,
    ChildPausedState,
    ConsultRequest,
    ConsultResponseAnswer,
    EmptyToolRegistry,
    HarnessBuilder,
    HarnessRunOptions,
    HumanRequestClarification,
    MockAgent,
    NoopContextManager,
    PausedState,
    ReactConfig,
    RunResultConsult,
    RunResultSuccess,
    RunResultWaitingForHuman,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardHarness,
    Task,
    TaskId,
    ToolCall,
    ToolOutputAwaitingClarification,
    ToolOutputConsult,
    ToolOutputSuccess,
    TokenUsage,
)
from spore_core.agent import FinalResponse, ToolCallRequested
from spore_core.harness import HarnessToolResult
from spore_core.model import Message, ModelParams
from spore_core.tool_registry import (
    SandboxProvider,
    Tool,
    ToolAnnotations,
    ToolContext,
    ToolOutput,
    ToolSchema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


def _consult_req() -> ConsultRequest:
    return ConsultRequest(
        kind="advice",
        situation="stuck on auth",
        attempts=1,
        question="what next?",
    )


def _react_scoped(max_iter: int, handle: str, session: str = "s") -> Task:
    """A bare ReAct task whose leaf carries a NON-EMPTY toolset ``handle``. Used
    to prove the leaf's handle threads into the pause state (AC2a)."""
    return Task.new(
        "audit the auth module",
        SessionId(session),
        ReactConfig(
            budget=BudgetPolicyPerLoop(value=max_iter),
            behavior=BudgetExhaustedEscalate(),
            agent="",
            toolset=handle,
        ),
    )


class _RecordingContextManager:
    """Records appended tool results so a test can assert what dispatched.
    Satisfies the ``ContextManager`` Protocol structurally."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.tool_results: list[HarnessToolResult] = []

    async def assemble(self, session: SessionState, task: Task, sources: object) -> object:
        from spore_core.agent import Context

        _ = task
        return Context(messages=list(self.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: HarnessToolResult) -> None:
        _ = session
        self.tool_results.append(result)

    async def append_assistant_message(self, session: SessionState, message: Message) -> None:
        _ = session
        self.messages.append(message)

    async def append_user_message(self, session: SessionState, text: str) -> None:
        _ = session

    def should_compact(self, session: SessionState) -> bool:
        _ = session
        return False


# ----- scoped-catalogue test doubles (mirror test_harness_toolset_scoping) ---


class _ScopedTool:
    """A minimal :class:`Tool` that echoes its input so a successful dispatch is
    distinguishable from a recoverable unknown-tool error."""

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        _ = (sandbox, ctx)
        return ToolOutputSuccess.success(f"ran {self._name} with {json.dumps(call.input)}")


@dataclass
class _StandardTool:
    """``StandardTool``-shaped bundle the builder destructures (issue #81)."""

    implementation: Tool
    schema: ToolSchema


def _catalogue_tool(name: str) -> _StandardTool:
    return _StandardTool(
        implementation=_ScopedTool(name),
        schema=ToolSchema(
            name=name,
            description=f"the {name} tool",
            parameters={"type": "object", "properties": {}},
            annotations=ToolAnnotations(read_only=True),
        ),
    )


# ===========================================================================
# AC1 — back-compat (missing key → empty) + always-serialize + round-trip.
# ===========================================================================


def test_paused_state_toolset_back_compat_and_round_trip() -> None:
    """AC1: a pre-#140 paused-state blob (no ``toolset`` key) deserializes,
    defaulting to the empty handle; the empty handle ALWAYS serializes (never
    skipped); a non-empty handle round-trips by value. The same contract holds
    for :class:`ChildPausedState`."""
    react5 = Task.new("t", SessionId("s"), ReactConfig.per_loop(5))

    # A pre-#140 PausedState JSON (no ``toolset`` key) — must default to "".
    pre_140 = {
        "session_id": "s",
        "task_id": "t",
        "turn_number": 1,
        "session_state": {"messages": [], "extras": {}},
        "pending_tool_calls": [],
        "approved_results": [],
        "human_request": None,
        "task": json.loads(react5.model_dump_json()),
        "budget_used": json.loads(BudgetSnapshot().model_dump_json()),
        "child_state": None,
        # note: NO "toolset" key
    }
    parsed = PausedState.model_validate_json(json.dumps(pre_140))
    assert parsed.toolset == "", "missing toolset must default to the empty handle"

    # The empty handle ALWAYS serializes (never skipped) for byte-parity.
    wire = parsed.model_dump_json()
    assert '"toolset":""' in wire, f"empty toolset must serialize explicitly: {wire}"

    # A non-empty handle round-trips by value.
    scoped = parsed.model_copy(update={"toolset": "scoped"})
    back = PausedState.model_validate_json(scoped.model_dump_json())
    assert back.toolset == "scoped"
    assert back == scoped

    # The same back-compat + always-serialize contract holds for ChildPausedState.
    child_pre_140 = {
        "session_id": "c",
        "task_id": "ct",
        "turn_number": 1,
        "session_state": {"messages": [], "extras": {}},
        "pending_tool_calls": [],
        "approved_results": [],
        "human_request": None,
        "task": json.loads(react5.model_dump_json()),
        "budget_used": json.loads(BudgetSnapshot().model_dump_json()),
        "parent_tool_call_id": "p",
        # note: NO "toolset" key
    }
    child = ChildPausedState.model_validate_json(json.dumps(child_pre_140))
    assert child.toolset == ""
    assert '"toolset":""' in child.model_dump_json()


# ===========================================================================
# AC2a — leaf-pause paths capture the leaf's toolset handle.
# ===========================================================================


async def test_consult_pause_carries_leaf_toolset_handle() -> None:
    """AC2a: a Consult pause from a leaf carrying ``"scoped"`` returns a
    :class:`PausedState` whose ``toolset`` is that handle."""
    agent = MockAgent(AgentId("worker"))
    agent.push(
        ToolCallRequested(
            calls=[ToolCall(id="c0", name="ask_advice", input={"kind": "advice"})],
            usage=_usage(),
        )
    )
    reg = ScriptedToolRegistry().push(ToolOutputConsult.consult(_consult_req()))
    # Register the "scoped" toolset so the tree's handle passes validate() at run
    # entry; an EmptyToolRegistry presence entry is sufficient.
    cfg = (
        HarnessBuilder(
            agent,
            reg,
            AllowAllSandbox(),
            NoopContextManager(),
            AlwaysContinuePolicy(),
        )
        .register_toolset("scoped", EmptyToolRegistry())
        .build_config()
    )
    h = StandardHarness(cfg)
    r = await h.run(HarnessRunOptions(_react_scoped(5, "scoped")))
    assert isinstance(r, RunResultConsult)
    assert r.state.toolset == "scoped", "consult pause must carry the leaf's toolset handle"


async def test_clarification_pause_carries_leaf_toolset_handle() -> None:
    """AC2a: the Clarification (#81) leaf-pause path likewise carries the leaf's
    toolset handle."""
    agent = MockAgent(AgentId("worker"))
    agent.push(
        ToolCallRequested(
            calls=[ToolCall(id="c0", name="ask_user", input={})],
            usage=_usage(),
        )
    )
    reg = ScriptedToolRegistry().push(
        ToolOutputAwaitingClarification(question="which?", options=None)
    )
    cfg = (
        HarnessBuilder(
            agent,
            reg,
            AllowAllSandbox(),
            NoopContextManager(),
            AlwaysContinuePolicy(),
        )
        .register_toolset("scoped", EmptyToolRegistry())
        .build_config()
    )
    h = StandardHarness(cfg)
    r = await h.run(HarnessRunOptions(_react_scoped(5, "scoped")))
    assert isinstance(r, RunResultWaitingForHuman)
    assert isinstance(r.state.human_request, HumanRequestClarification)
    assert r.state.toolset == "scoped"


# ===========================================================================
# AC2b — resume routes pending calls through the carried toolset handle.
# ===========================================================================


def _make_paused(handle: str) -> PausedState:
    """A paused state with TWO pending calls: the head consult call (gets the
    answer injected) and a trailing call to the scoped-only tool (dispatched
    through ``_effective_tool_registry(session_id, state.toolset)``). A BARE
    ReAct leaf, so resume takes the dispatch-then-re-enter path (not re-drive)."""
    return PausedState(
        session_id=SessionId("s"),
        task_id=TaskId("t"),
        turn_number=1,
        session_state=SessionState(),
        pending_tool_calls=[
            ToolCall(id="consult", name="ask_advice", input={"kind": "advice"}),
            ToolCall(id="scoped", name="scoped_only", input={"probe": 1}),
        ],
        approved_results=[],
        human_request=None,
        task=Task.new(
            "audit",
            SessionId("s"),
            ReactConfig(
                budget=BudgetPolicyPerLoop(value=5),
                behavior=BudgetExhaustedEscalate(),
                agent="",
                toolset=handle,
            ),
        ),
        budget_used=BudgetSnapshot(),
        child_state=None,
        toolset=handle,
    )


def _scoped_result(results: list[HarnessToolResult]) -> HarnessToolResult | None:
    for r in results:
        if r.call_id == "scoped":
            return r
    return None


def _harness_with_scoped_catalogue(cm: _RecordingContextManager) -> StandardHarness:
    """A harness whose GLOBAL catalogue has ``global_only`` (NOT ``scoped_only``)
    and whose SCOPED ``"scoped"`` catalogue has ``scoped_only``. A worker agent
    emits a final response so the re-entered ReAct window terminates."""
    agent = MockAgent(AgentId("worker"))
    agent.push(FinalResponse(content="resumed-done", usage=_usage()))
    cfg = (
        HarnessBuilder(
            agent,
            ScriptedToolRegistry(),
            AllowAllSandbox(),
            cm,
            AlwaysContinuePolicy(),
        )
        .tools([_catalogue_tool("global_only")])
        .toolset_tools("scoped", [_catalogue_tool("scoped_only")])
        .build_config()
    )
    return StandardHarness(cfg)


async def test_resume_consult_routes_pending_calls_through_carried_toolset() -> None:
    """AC2b (THE load-bearing regression guard): a paused state whose ``toolset``
    is ``"scoped"`` resumes the trailing scoped-only call against the SCOPED
    catalogue — it dispatches successfully. The negative control (empty handle)
    falls back to the global catalogue, where ``scoped_only`` is unknown → a
    recoverable error. This proves the carried handle is behaviorally
    load-bearing, not cosmetic."""
    # ── Positive: carried handle "scoped" routes to the scoped catalogue ──
    cm = _RecordingContextManager()
    h = _harness_with_scoped_catalogue(cm)
    r = await h.resume_consult(_make_paused("scoped"), ConsultResponseAnswer(text="ok"))
    assert isinstance(r, RunResultSuccess)
    scoped = _scoped_result(cm.tool_results)
    assert scoped is not None, "the scoped-only call must have been dispatched"
    assert scoped.output.kind == "success", (
        "scoped-only tool must dispatch successfully when the carried handle is 'scoped'"
    )
    assert "probe" in scoped.output.content

    # ── Negative control: EMPTY handle falls back to the global catalogue ──
    cm = _RecordingContextManager()
    h = _harness_with_scoped_catalogue(cm)
    r = await h.resume_consult(_make_paused(""), ConsultResponseAnswer(text="ok"))
    assert isinstance(r, RunResultSuccess)
    scoped = _scoped_result(cm.tool_results)
    assert scoped is not None
    assert scoped.output.kind == "error", (
        "with the EMPTY handle the scoped-only tool is unknown → must NOT dispatch successfully"
    )
    assert scoped.output.recoverable is True, (
        "the scoped-only call must surface a RECOVERABLE error under the empty handle"
    )


# ===========================================================================
# Subagent — _child_state_from_paused carries the child leaf's own handle.
# ===========================================================================


def test_child_state_from_paused_carries_toolset() -> None:
    """#140: the subagent ``_child_state_from_paused`` must carry the child
    leaf's OWN toolset handle through to the :class:`ChildPausedState`, so the
    child resumes against its scoped catalogue rather than the parent's / global
    fallback."""
    from spore_tools.tools.subagent import _child_state_from_paused

    paused = _make_paused("worker-tools")
    child = _child_state_from_paused(paused, "parent-call-1")
    assert child.toolset == "worker-tools"
    assert child.parent_tool_call_id == "parent-call-1"
