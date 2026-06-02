"""Session-state threading + opt-in auto-persist (issue #102).

Mirrors the Rust reference ``mod session_threading`` in
``rust/crates/spore-core/src/harness.rs``. Two layers:

* Part 1 — the post-run :class:`SessionState` is returned on Success/Failure so
  an in-process caller can resume losslessly (tool-call + tool-result turns are
  recoverable, not just ``output``).
* Part 2 — opt-in ``auto_persist_sessions`` layers SessionStore auto-load
  (ReAct / SelfVerifying only; explicit session_state wins) + terminal persist,
  giving cross-process continuity by ``session_id``.

OFF by default: ZERO session-store I/O and a byte-identical message flow.
"""

from __future__ import annotations

from pathlib import Path

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetSnapshot,
    FinalResponse,
    HaltReasonUnrecoverableToolError,
    HarnessConfig,
    HarnessRunOptions,
    InMemoryStorageProvider,
    LoopStrategyReAct,
    MockAgent,
    PausedState,
    RunResultFailure,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardHarness,
    StorageProvider,
    Task,
    TokenUsage,
    ToolCall,
    ToolCallRequested,
    ToolOutputSuccess,
)
from spore_core.harness import (
    HarnessToolResult,
    ToolOutputError,
)
from spore_core.storage import FileSystemStorageProvider
from spore_core.model import (
    Message,
    ModelParams,
    Role,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)

# ---------------------------------------------------------------------------
# Helpers (mirror the Rust test doubles)
# ---------------------------------------------------------------------------


def _agent() -> MockAgent:
    return MockAgent(AgentId("test"))


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=1, output_tokens=1)


def _tc(call_id: str = "c1", name: str = "x") -> ToolCall:
    return ToolCall(id=call_id, name=name, input={})


def _react_task(
    session_id: str = "s1", instruction: str = "do something", max_iter: int = 5
) -> Task:
    return Task.new(
        instruction,
        SessionId(session_id),
        LoopStrategyReAct(max_iterations=max_iter),
    )


class RecordingContextManager:
    """Records every appended message into ``session.messages`` so the post-run
    :class:`SessionState` reflects the full conversation the loop built — the
    assistant tool-call turn, the tool result, and the final text. Mirrors the
    Rust reference's message-recording manager."""

    async def assemble(self, session: SessionState, task: Task) -> object:
        from spore_core.agent import Context

        _ = task
        return Context(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: HarnessToolResult) -> None:
        output = result.output
        if isinstance(output, ToolOutputSuccess):
            content, is_error = output.content, False
        elif isinstance(output, ToolOutputError):
            content, is_error = output.message, True
        else:
            content, is_error = "", False
        session.messages.append(
            Message(
                role=Role.TOOL,
                content=ToolResultContent(
                    tool_use_id=result.call_id, content=content, is_error=is_error
                ),
            )
        )

    async def append_assistant_message(self, session: SessionState, message: Message) -> None:
        session.messages.append(message)

    async def append_user_message(self, session: SessionState, text: str) -> None:
        session.messages.append(Message(role=Role.USER, content=TextContent(text=text)))

    def should_compact(self, session: SessionState) -> bool:
        _ = session
        return False


class CountingSessionStore:
    """In-memory :class:`SessionStore` that COUNTS every get/put so a test can
    assert "zero session-store I/O when auto_persist is disabled". Mirrors the
    Rust reference's ``CountingSessionStore``."""

    def __init__(self) -> None:
        self.inner = InMemoryStorageProvider()
        self.gets = 0
        self.puts = 0

    async def get_session(self, session_id: SessionId) -> PausedState | None:
        self.gets += 1
        return await self.inner.get_session(session_id)

    async def put_session(self, session_id: SessionId, state: PausedState) -> None:
        self.puts += 1
        await self.inner.put_session(session_id, state)

    async def delete_session(self, session_id: SessionId) -> None:
        await self.inner.delete_session(session_id)

    async def list_sessions(self) -> list[SessionId]:
        return await self.inner.list_sessions()


def _config(
    agent: MockAgent,
    *,
    tool_registry: ScriptedToolRegistry | None = None,
    context_manager: object | None = None,
    storage: StorageProvider | None = None,
    auto_persist_sessions: bool = False,
) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=tool_registry or ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=context_manager or RecordingContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        storage=storage,
        auto_persist_sessions=auto_persist_sessions,
    )


def _tool_then_final_agent() -> MockAgent:
    """One tool call, then final text. Mirrors ``tool_then_final_agent``."""
    a = _agent()
    a.push(ToolCallRequested(calls=[_tc("c1", "x")], usage=_usage()))
    a.push(FinalResponse(content="after-tool", usage=_usage()))
    return a


def _tool_registry() -> ScriptedToolRegistry:
    return ScriptedToolRegistry().push(ToolOutputSuccess(content="tool-ok"))


def _texts(state: SessionState) -> list[str]:
    return [m.content.text for m in state.messages if isinstance(m.content, TextContent)]


def _has_tool_call(state: SessionState) -> bool:
    return any(isinstance(m.content, ToolCallContent) for m in state.messages)


def _has_tool_result(state: SessionState) -> bool:
    return any(
        m.role == Role.TOOL and isinstance(m.content, ToolResultContent) for m in state.messages
    )


# ---------------------------------------------------------------------------
# (a) Off-by-default: ZERO session-store I/O AND the message flow / outcome is
# identical to today (Success carries the right messages).
# ---------------------------------------------------------------------------


async def test_off_by_default_no_session_io() -> None:
    store = CountingSessionStore()
    cfg = _config(
        _tool_then_final_agent(),
        tool_registry=_tool_registry(),
        storage=StorageProvider.single(store),
        auto_persist_sessions=False,
    )
    assert cfg.auto_persist_sessions is False
    h = StandardHarness(cfg)
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "after-tool"
    # The new field is populated even when persistence is off.
    assert _has_tool_call(r.session_state)
    # The off-by-default zero-I/O contract: no get/put hits the store.
    assert store.gets == 0, "no get_session calls"
    assert store.puts == 0, "no put_session calls"


# ---------------------------------------------------------------------------
# (b) Success.session_state is LOSSLESS for a tool-using run: assistant
# tool-call message, tool-result message, and final text all present.
# ---------------------------------------------------------------------------


async def test_success_session_state_is_lossless_for_tool_run() -> None:
    cfg = _config(_tool_then_final_agent(), tool_registry=_tool_registry())
    h = StandardHarness(cfg)
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultSuccess)
    state = r.session_state
    # user instruction
    assert any(m.role == Role.USER for m in state.messages)
    # assistant tool-call turn
    assert _has_tool_call(state)
    # tool-result turn
    assert _has_tool_result(state)
    # final assistant text — NOT recoverable from any single field but `output`
    assert any(
        m.role == Role.ASSISTANT
        and isinstance(m.content, TextContent)
        and m.content.text == "after-tool"
        for m in state.messages
    )


async def test_success_session_state_lossless_via_fixture_replay() -> None:
    """Replays the shared ``react_loop.jsonl`` fixture and asserts the post-run
    SessionState is lossless (tool-call + tool-result + final text), the part-1
    payoff of issue #102 against the same fixture the Rust reference uses."""
    from spore_core import ProviderInfo, ReplayModelInterface
    from spore_core.agent import ModelAgent

    here = Path(__file__).resolve()
    jsonl = (
        here.parents[4] / "fixtures" / "model_responses" / "harness" / "react_loop.jsonl"
    ).read_text()
    replay = ReplayModelInterface.from_jsonl(
        jsonl,
        ProviderInfo(name="anthropic", model_id="fixture", context_window=200_000),
    )
    agent = ModelAgent(AgentId("fixture-agent"), replay)
    reg = ScriptedToolRegistry().push(ToolOutputSuccess(content="127.0.0.1 localhost"))
    cfg = HarnessConfig(
        agent=agent,
        tool_registry=reg,
        sandbox=AllowAllSandbox(),
        context_manager=RecordingContextManager(),
        termination_policy=AlwaysContinuePolicy(),
    )
    h = StandardHarness(cfg)
    task = Task.new(
        "read /etc/hosts then summarize",
        SessionId("fixture-session"),
        LoopStrategyReAct(max_iterations=5),
    )
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "127.0.0.1 localhost"
    assert _has_tool_call(r.session_state), "assistant tool-call turn recoverable"
    assert _has_tool_result(r.session_state), "tool-result turn recoverable"


# ---------------------------------------------------------------------------
# (c) Auto-load across two runs sharing a session_id (in-memory store): the
# second run sees the first run's history.
# ---------------------------------------------------------------------------


async def test_auto_load_by_session_id_across_runs() -> None:
    provider = StorageProvider.single(InMemoryStorageProvider())
    sid = "shared"

    # Run 1: one final response, auto-persisted.
    a1 = _agent()
    a1.push(FinalResponse(content="first", usage=_usage()))
    h1 = StandardHarness(_config(a1, storage=provider, auto_persist_sessions=True))
    await h1.run(HarnessRunOptions(_react_task(sid, "turn one")))

    # Run 2: same session_id, no explicit state. Loaded history carries forward.
    a2 = _agent()
    a2.push(FinalResponse(content="second", usage=_usage()))
    h2 = StandardHarness(_config(a2, storage=provider, auto_persist_sessions=True))
    r = await h2.run(HarnessRunOptions(_react_task(sid, "turn two")))
    assert isinstance(r, RunResultSuccess)
    texts = _texts(r.session_state)
    assert "first" in texts, "prior turn carried forward"
    assert "turn one" in texts
    assert "turn two" in texts
    assert "second" in texts


# ---------------------------------------------------------------------------
# (d) Auto-persist round-trip: get_session returns the final history with empty
# pending fields and no human request.
# ---------------------------------------------------------------------------


async def test_auto_persist_round_trip() -> None:
    provider = StorageProvider.single(InMemoryStorageProvider())
    cfg = _config(
        _tool_then_final_agent(),
        tool_registry=_tool_registry(),
        storage=provider,
        auto_persist_sessions=True,
    )
    h = StandardHarness(cfg)
    sid = SessionId("s1")
    await h.run(HarnessRunOptions(_react_task("s1")))

    stored = await provider.session().get_session(sid)
    assert stored is not None, "session persisted at terminal"
    assert _has_tool_call(stored.session_state)
    assert stored.pending_tool_calls == []
    assert stored.approved_results == []
    assert stored.human_request is None
    assert stored.child_state is None


# ---------------------------------------------------------------------------
# (e) Cross-process continuity via a filesystem store under a tempdir: a
# brand-new provider + harness over the same dir resumes by session_id.
# ---------------------------------------------------------------------------


async def test_cross_process_continuity_filesystem(tmp_path: Path) -> None:
    sid = "fs-session"

    # "Process 1": its own provider + harness instance.
    provider1 = StorageProvider.single(FileSystemStorageProvider(tmp_path))
    a1 = _agent()
    a1.push(FinalResponse(content="process-one", usage=_usage()))
    h1 = StandardHarness(_config(a1, storage=provider1, auto_persist_sessions=True))
    await h1.run(HarnessRunOptions(_react_task(sid, "first process")))

    # "Process 2": brand-new provider over the SAME dir, brand-new harness.
    provider2 = StorageProvider.single(FileSystemStorageProvider(tmp_path))
    a2 = _agent()
    a2.push(FinalResponse(content="process-two", usage=_usage()))
    h2 = StandardHarness(_config(a2, storage=provider2, auto_persist_sessions=True))
    r = await h2.run(HarnessRunOptions(_react_task(sid, "second process")))
    assert isinstance(r, RunResultSuccess)
    texts = _texts(r.session_state)
    assert "process-one" in texts, "prior process history loaded"
    assert "process-two" in texts


# ---------------------------------------------------------------------------
# (f) Explicit session_state WINS over auto-load (no get_session; explicit state
# seeds the run).
# ---------------------------------------------------------------------------


async def test_explicit_session_state_wins_over_auto_load() -> None:
    store = CountingSessionStore()
    # Pre-seed the store with a DIFFERENT history under the session id.
    prior = SessionState(
        messages=[Message(role=Role.USER, content=TextContent(text="STORED-history"))]
    )
    ps = PausedState(
        session_id=SessionId("s1"),
        task_id="s1",  # type: ignore[arg-type]
        turn_number=0,
        session_state=prior,
        pending_tool_calls=[],
        approved_results=[],
        human_request=None,
        task=Task.new("", SessionId("s1"), LoopStrategyReAct(max_iterations=0)),
        budget_used=BudgetSnapshot(),
        child_state=None,
    )
    await store.inner.put_session(SessionId("s1"), ps)
    store.puts = 0  # reset the pre-seed write

    a = _agent()
    a.push(FinalResponse(content="done", usage=_usage()))
    h = StandardHarness(
        _config(a, storage=StorageProvider.single(store), auto_persist_sessions=True)
    )

    explicit = SessionState(
        messages=[Message(role=Role.USER, content=TextContent(text="EXPLICIT-history"))]
    )
    r = await h.run(HarnessRunOptions(_react_task("s1", "turn"), session_state=explicit))
    assert isinstance(r, RunResultSuccess)
    texts = _texts(r.session_state)
    assert "EXPLICIT-history" in texts
    assert "STORED-history" not in texts, "auto-load skipped"
    assert store.gets == 0, "explicit state skips the auto-load get_session"


# ---------------------------------------------------------------------------
# (g) Failure ALSO carries session_state.
# ---------------------------------------------------------------------------


async def test_failure_carries_session_state() -> None:
    # A tool annotated always_halt fails the run after the assistant tool-call
    # turn was recorded into session_state.
    a = _agent()
    a.push(ToolCallRequested(calls=[_tc("c", "danger")], usage=_usage()))
    reg = ScriptedToolRegistry()
    reg.mark_always_halt("danger")
    h = StandardHarness(_config(a, tool_registry=reg))
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonUnrecoverableToolError)
    # The seeded user instruction is present in the failure state.
    assert any(
        m.role == Role.USER and isinstance(m.content, TextContent) for m in r.session_state.messages
    )


# ---------------------------------------------------------------------------
# (h) Failure auto-persists too (terminal write happens on Failure as well).
# ---------------------------------------------------------------------------


async def test_failure_auto_persists() -> None:
    store = CountingSessionStore()
    a = _agent()
    a.push(ToolCallRequested(calls=[_tc("c", "danger")], usage=_usage()))
    reg = ScriptedToolRegistry()
    reg.mark_always_halt("danger")
    h = StandardHarness(
        _config(
            a,
            tool_registry=reg,
            storage=StorageProvider.single(store),
            auto_persist_sessions=True,
        )
    )
    r = await h.run(HarnessRunOptions(_react_task()))
    assert isinstance(r, RunResultFailure)
    assert store.puts == 1, "a Failure terminal still persists once"
    stored = await store.inner.get_session(SessionId("s1"))
    assert stored is not None
