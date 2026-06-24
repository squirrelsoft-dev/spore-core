"""SC-10 (#161) — per-leaf ``system_prompt`` override in :class:`ReactConfig`.

Mirrors the Rust reference's acceptance tests
(``rust/crates/spore-core/src/harness.rs``):

- ``plan_and_execute_leaves_see_only_their_own_system_prompt``
- ``leaf_system_prompt_overrides_global_and_falls_back``

The toolset half of "per-phase config" was already per-leaf
(:attr:`ReactConfig.toolset`); this adds the matching per-leaf PROMPT half so
each :class:`PlanExecuteConfig` phase sees ONLY its own system prompt. The
override REPLACES the global ``config.system_prompt`` for that leaf's window
(``effective = leaf.or(global)`` — None-based, not falsy); a leaf WITHOUT an
override falls back to the global prompt (byte-identical to pre-SC-10).

A :class:`RecordingAgent` captures the assembled-context text the model saw on
each turn so we can assert which system prompt(s) reached which phase.
"""

from __future__ import annotations

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetPolicyPerLoop,
    Context,
    FinalResponse,
    HarnessConfig,
    HarnessRunOptions,
    InMemoryStorageProvider,
    NoopContextManager,
    PlanExecuteConfig,
    ReactConfig,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    StandardHarness,
    StorageProvider,
    Task,
    TextContent,
    TokenUsage,
    TurnResult,
)


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=1, output_tokens=1)


def _context_text(context: Context) -> str:
    """Flatten every text block of the assembled turn context into one string,
    so a marker that appears in ANY message (esp. the leading System block) is
    detectable. Mirrors the Rust ``RecordingTurnAgent::seen_text`` join."""
    parts: list[str] = []
    for message in context.messages:
        content = message.content
        if isinstance(content, TextContent):
            parts.append(content.text)
    return "\n".join(parts)


class RecordingAgent:
    """Programmable :class:`~spore_core.agent.Agent` that RECORDS the assembled
    context text it saw on each turn (so a test can assert which system prompt
    reached which phase), then yields the next queued :class:`TurnResult`.

    Mirrors the Rust ``RecordingTurnAgent``: the recorded text is the join of
    every turn context's text blocks, in turn order.
    """

    def __init__(self, agent_id: AgentId, results: list[TurnResult]) -> None:
        self._id = agent_id
        self._results = list(results)
        self.seen_text: list[str] = []

    async def turn(self, context: Context) -> TurnResult:
        self.seen_text.append(_context_text(context))
        if not self._results:
            raise AssertionError("RecordingAgent ran out of queued results")
        return self._results.pop(0)

    def id(self) -> AgentId:
        return self._id


def _config(agent: RecordingAgent, *, system_prompt: str | None) -> HarnessConfig:
    # NoopContextManager produces NO structural System block, so the harness's
    # system-prompt prepend path inserts a fresh leading System message — the
    # cleanest place to observe which prompt reached the turn.
    return HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        storage=StorageProvider.single(InMemoryStorageProvider()),
        system_prompt=system_prompt,
    )


def _react(*, system_prompt: str | None = None, output: str | None = None) -> ReactConfig:
    """A bare unbounded ReAct leaf carrying an optional per-leaf system prompt."""
    return ReactConfig(
        budget=BudgetPolicyPerLoop(value=2**31 - 1),
        agent="",
        toolset="",
        output=output,
        system_prompt=system_prompt,
    )


def _plan_execute_task(plan: ReactConfig, execute: ReactConfig) -> Task:
    return Task.new(
        "build it",
        SessionId("sc10-s1"),
        PlanExecuteConfig(plan=plan, execute=execute),
    )


# ---------------------------------------------------------------------------
# Acceptance #1: distinct plan/execute prompts, each phase sees only its own.
# Global prompt unset, so any cross-phase appearance is an unambiguous leak.
# ---------------------------------------------------------------------------


async def test_plan_and_execute_leaves_see_only_their_own_system_prompt() -> None:
    plan_sys = "PLAN_SYSTEM_PROMPT_MARKER"
    exec_sys = "EXECUTE_SYSTEM_PROMPT_MARKER"

    agent = RecordingAgent(
        AgentId("rec"),
        [
            # Plan turn: a single-step plan.
            FinalResponse(content='{"tasks":["only step"],"rationale":"r"}', usage=_usage()),
            # Execute step: finalize directly.
            FinalResponse(content="did the step", usage=_usage()),
        ],
    )
    # No global prompt ⇒ the ONLY system text a turn can see is its leaf's.
    h_config = _config(agent, system_prompt=None)
    assert h_config.system_prompt is None
    h = StandardHarness(h_config)

    task = _plan_execute_task(
        # The PlanExecute ``plan`` slot is STRUCTURED (#124, A.5) — a bare ReAct
        # there MUST declare ``output`` (here the default empty-key schema the
        # builder fills). The per-leaf ``system_prompt`` is the override under test.
        plan=_react(system_prompt=plan_sys, output=""),
        execute=_react(system_prompt=exec_sys),
    )
    result = await h.run(HarnessRunOptions(task))
    assert isinstance(result, RunResultSuccess)
    assert result.output == "did the step"

    contexts = agent.seen_text
    assert len(contexts) == 2, "one plan turn + one execute turn"

    # Plan turn (index 0): sees ONLY the plan leaf's prompt.
    assert plan_sys in contexts[0], contexts[0]
    assert exec_sys not in contexts[0], contexts[0]

    # Execute turn (index 1): sees ONLY the execute leaf's prompt.
    assert exec_sys in contexts[1], contexts[1]
    assert plan_sys not in contexts[1], contexts[1]


# ---------------------------------------------------------------------------
# Acceptance #2: the per-leaf override WINS over the global prompt; a leaf
# WITHOUT an override falls back to the global prompt (byte-identical to
# pre-SC-10).
# ---------------------------------------------------------------------------


async def test_leaf_system_prompt_overrides_global_and_falls_back() -> None:
    global_sys = "GLOBAL_SYSTEM_PROMPT_MARKER"
    plan_sys = "PLAN_ONLY_SYSTEM_PROMPT_MARKER"

    agent = RecordingAgent(
        AgentId("rec"),
        [
            FinalResponse(content='{"tasks":["only step"],"rationale":"r"}', usage=_usage()),
            FinalResponse(content="did the step", usage=_usage()),
        ],
    )
    # A global prompt IS configured this time.
    h = StandardHarness(_config(agent, system_prompt=global_sys))

    task = _plan_execute_task(
        # Plan leaf overrides the global prompt (structured plan slot ⇒ ``output``).
        plan=_react(system_prompt=plan_sys, output=""),
        # Execute leaf carries no override ⇒ falls back to the global prompt.
        execute=_react(system_prompt=None),
    )
    result = await h.run(HarnessRunOptions(task))
    assert isinstance(result, RunResultSuccess)

    contexts = agent.seen_text
    assert len(contexts) == 2, "one plan turn + one execute turn"

    # Plan turn: its override WINS — only the plan prompt, NOT the global one.
    assert plan_sys in contexts[0] and global_sys not in contexts[0], contexts[0]
    # Execute turn: no override ⇒ the global prompt applies (back-compat).
    assert global_sys in contexts[1] and plan_sys not in contexts[1], contexts[1]
