"""Fixture-replay integration tests for the cordyceps composition (#131):
``Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]``, driven by the canonical
``fixtures/strategy/cordyceps_tree.json``.

These exercise the SAME recorded-model harness as
``test_harness_plan_execute_dag.py``, but against the FULL composed tree with its
real handles wired into an :class:`ExecutionRegistry`: agents
``planner``/``executor``/``ralph-agent``, toolsets ``plan-tools``/``exec-tools``,
schemas ``plan-schema``/``worker-schema``, and the Default-FAIL ``exec-evaluator``
verifier. Never edit a fixture to make a failing implementation pass — the
fixtures are ground truth and must stay internally consistent.

Mirror of the Rust reference
``rust/crates/spore-core/tests/cordyceps_composition_fixture_replay.rs``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetPolicyUnlimited,
    ConsultRequest,
    ConsultResponseAnswer,
    EmptyToolRegistry,
    EscalationModeAutonomous,
    EvaluatorResponseVerifier,
    ExecutionRegistry,
    HaltReasonTasksBlockedByFailure,
    HarnessConfig,
    HarnessRunOptions,
    InMemoryStorageProvider,
    LoopStrategy,
    ModelAgent,
    ProviderInfo,
    RalphConfig,
    ReplayModelInterface,
    RunResultConsult,
    RunResultFailure,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardHarness,
    StorageProvider,
    Task,
    ToolOutputConsult,
    Verifier,
)
from spore_core.harness import AggregateUsage, loop_strategy_max_steps
from spore_core.tasklist import TASK_LIST_EXTRAS_KEY, TaskList, TaskStatus
from spore_core.verifier import VerifierInput, VerifierVerdictFailed, VerifierVerdictPassed

_LOOP_STRATEGY_ADAPTER: TypeAdapter[LoopStrategy] = TypeAdapter(LoopStrategy)


def _provider() -> ProviderInfo:
    return ProviderInfo(name="anthropic", model_id="fixture", context_window=200_000)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _fixture_path(name: str) -> Path:
    return _repo_root() / "fixtures" / "model_responses" / "harness" / name


def _exec_evaluator() -> Verifier:
    """The Default-FAIL evaluator the ``exec-evaluator`` handle resolves to — the
    same construction the 12-cordyceps example registers (single read-only turn;
    neither-pattern ⇒ Failed)."""
    return EvaluatorResponseVerifier(r"(?i)\bPASS\b", r"(?i)\bFAIL\b", 1)


def _cordyceps_tree() -> LoopStrategy:
    """The canonical cordyceps tree, deserialized from the shared fixture (the
    same path the example uses)."""
    txt = (_repo_root() / "fixtures" / "strategy" / "cordyceps_tree.json").read_text()
    return _LOOP_STRATEGY_ADAPTER.validate_json(txt)


def _cordyceps_plan_execute() -> LoopStrategy:
    """The PlanExecute subtree of the cordyceps tree (drops the Ralph wrapper) so
    the positional fixture maps 1:1 to one window — Ralph's per-window reset loop
    would otherwise re-enter and re-consume the (exhausted) replay queue."""
    tree = _cordyceps_tree()
    assert isinstance(tree, RalphConfig), "root is Ralph"
    return tree.inner


def _pe_task(session: str) -> Task:
    t = Task.new("audit the repo", SessionId(session), _cordyceps_plan_execute())
    t.budget.max_turns = 64
    return t


def _registry(replay: ReplayModelInterface) -> ExecutionRegistry:
    def agent(idv: str) -> ModelAgent:
        return ModelAgent(AgentId(idv), replay)

    return (
        ExecutionRegistry.builder()
        .agent("planner", agent("planner"))
        .agent("executor", agent("executor"))
        .agent("ralph-agent", agent("ralph-agent"))
        .toolset("plan-tools", EmptyToolRegistry())
        .toolset("exec-tools", EmptyToolRegistry())
        .schema("plan-schema", {"type": "object"})
        .schema("worker-schema", {"type": "array"})
        .verifier("exec-evaluator", _exec_evaluator())
        .build()
    )


def _harness_for(
    fixture: str,
    storage: StorageProvider,
    *,
    tool_registry: ScriptedToolRegistry | None = None,
) -> StandardHarness:
    """Build a harness whose plan/worker/evaluator turns all replay positionally
    from ONE shared :class:`ReplayModelInterface` (a single cursor across the
    whole composed run), with the cordyceps handles wired into the registry."""
    jsonl = _fixture_path(fixture).read_text()
    replay = ReplayModelInterface.from_jsonl(jsonl, _provider())
    registry = _registry(replay)
    return StandardHarness(
        HarnessConfig(
            agent=ModelAgent(AgentId("ralph-agent"), replay),
            tool_registry=tool_registry or ScriptedToolRegistry(),
            sandbox=AllowAllSandbox(),
            context_manager=_StoringContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            storage=storage,
            registry=registry,
            escalation_mode=EscalationModeAutonomous(),
            consult_handlers={},
        )
    )


class _StoringContextManager:
    """A context manager that STORES appended user messages on the session and
    assembles them into the Context, so a recording agent observes the seeded
    text (unlike the no-op default). Same shape as the DAG test's stub."""

    async def assemble(self, session: SessionState, task: object) -> object:
        from spore_core.agent import Context
        from spore_core.model import ModelParams

        return Context(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: object) -> None:
        return None

    async def append_user_message(self, session: SessionState, text: str) -> None:
        from spore_core.model import Message, Role, TextContent

        session.messages.append(Message(role=Role.USER, content=TextContent(text=text)))

    async def append_assistant_message(self, session: SessionState, message: object) -> None:
        session.messages.append(message)  # type: ignore[arg-type]

    def should_compact(self, session: SessionState) -> bool:
        return False


def _new_storage() -> StorageProvider:
    return StorageProvider.single(InMemoryStorageProvider())


async def _seed(storage: StorageProvider, session: SessionId, tl: TaskList) -> None:
    await storage.run().put(session, TASK_LIST_EXTRAS_KEY, tl.to_dict())


async def _stored_list(storage: StorageProvider, session: SessionId) -> TaskList:
    value = await storage.run().get(session, TASK_LIST_EXTRAS_KEY)
    assert value is not None
    return TaskList.from_dict(value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC5 (static): the canonical tree's per-window worst case is computable before
# any run; an Unlimited anywhere collapses it to None.
# ---------------------------------------------------------------------------


def test_cordyceps_max_steps_is_17_unlimited_is_none() -> None:
    tree = _cordyceps_tree()
    assert loop_strategy_max_steps(tree) == 17

    # Swap the worker leaf's PerLoop{12} for Unlimited ⇒ None.
    assert isinstance(tree, RalphConfig)
    pe = tree.inner
    sv = pe.execute  # type: ignore[attr-defined]
    worker = sv.inner
    worker.budget = BudgetPolicyUnlimited()
    assert loop_strategy_max_steps(tree) is None


# ---------------------------------------------------------------------------
# AC6 (handle re-resolution): a paused cordyceps tree resumes by re-resolving
# EVERY handle from a freshly-built registry, with no reconfiguration.
# ---------------------------------------------------------------------------


def test_resume_reresolves_handles() -> None:
    raw = (_repo_root() / "fixtures" / "paused_states" / "cordyceps_budget_exhausted.json").read_text()
    doc = json.loads(raw)

    # The paused state carries the FULL cordyceps tree in task.loop_strategy.
    task = Task.model_validate(doc["task"])

    # Serde round-trip the Task (model_dump/model_validate is the wire).
    restored = Task.model_validate_json(task.model_dump_json())

    # A fresh registry, built independently (as on a cold resume), re-resolves
    # every handle — proving no reconfiguration of the Task is needed. No model
    # backend is needed; handle resolution is structural.
    jsonl = _fixture_path("plan_execute_dag_order.jsonl").read_text()
    replay = ReplayModelInterface.from_jsonl(jsonl, _provider())
    registry = _registry(replay)
    # validate() raises on the first unresolved handle; no raise ⇒ all resolve.
    registry.validate(restored)

    tree = restored.loop_strategy
    assert isinstance(tree, RalphConfig), "root is Ralph after round-trip"
    assert registry.resolve_agent(tree.agent) is not None, "ralph-agent resolves"
    pe = tree.inner
    plan = pe.plan  # type: ignore[attr-defined]
    assert registry.resolve_agent(plan.agent) is not None, "planner resolves"
    assert registry.resolve_toolset(plan.toolset) is not None, "plan-tools resolves"
    assert registry.resolve_schema(plan.output) is not None
    sv = pe.execute  # type: ignore[attr-defined]
    assert registry.resolve_verifier(sv.evaluator) is not None, "exec-evaluator resolves"
    worker = sv.inner
    assert registry.resolve_agent(worker.agent) is not None, "executor resolves"
    assert registry.resolve_toolset(worker.toolset) is not None, "exec-tools resolves"

    # The fixture's available_actions advertise the combinator escalation menu.
    kinds = [a["kind"] for a in doc["human_request"]["available_actions"]]
    assert kinds == ["continue_with_budget", "skip", "fail"]


# ---------------------------------------------------------------------------
# AC2: the plan phase builds a blocker-aware task graph (seeded via task_list)
# and the execute phase walks it as a READY-SET, self-verifying each task. Two
# independent modules both complete in ready-set order; the run succeeds.
# ---------------------------------------------------------------------------


async def test_plan_builds_dag_execute_walks_readyset() -> None:
    storage = _new_storage()
    session = SessionId("cordyceps-pe")
    tl = TaskList()
    tl.add("audit module one", [])  # 1
    tl.add("audit module two", [])  # 2 (independent)
    await _seed(storage, session, tl)
    h = _harness_for("cordyceps_plan_execute_readyset.jsonl", storage)

    r = await h.run(HarnessRunOptions(_pe_task("cordyceps-pe")))
    assert isinstance(r, RunResultSuccess), f"expected Success, got {r!r}"
    after = await _stored_list(storage, session)
    assert all(t.status is TaskStatus.COMPLETED for t in after.tasks), (
        f"all ready-set tasks complete: {[t.status for t in after.tasks]}"
    )


# ---------------------------------------------------------------------------
# AC4: a single runaway worker node exhausts its own PerLoop{12} budget and
# FAILS its task; an INDEPENDENT module still completes. The PlanExecute drains
# to TasksBlockedByFailure with a partition that does NOT cascade to the
# unrelated branch.
# ---------------------------------------------------------------------------


async def test_cordyceps_runaway_bounded() -> None:
    storage = _new_storage()
    session = SessionId("cordyceps-runaway")
    tl = TaskList()
    tl.add("root module", [])  # 1 (completes)
    tl.add("runaway module", [1])  # 2 -> 1 (PerLoop{12} budget-Fail)
    tl.add("dependent of runaway", [2])  # 3 -> 2 (cascade-blocked)
    tl.add("independent module", [])  # 4 (still completes)
    await _seed(storage, session, tl)
    h = _harness_for("cordyceps_runaway_bounded.jsonl", storage)

    r = await h.run(HarnessRunOptions(_pe_task("cordyceps-runaway")))
    assert isinstance(r, RunResultFailure), f"expected Failure, got {r!r}"
    assert isinstance(r.reason, HaltReasonTasksBlockedByFailure)
    assert r.reason.failed_task == 2, "the runaway module is the failed task"
    # The runaway (2) and its transitive dependent (3) are blocked; the root (1)
    # and the UNRELATED independent module (4) both complete — the runaway's
    # bounded failure does NOT cascade to unrelated tasks.
    assert r.reason.completed == [1, 4], "root + independent branch complete"
    assert r.reason.blocked == [2, 3], "runaway + its dependent are blocked"


# ---------------------------------------------------------------------------
# Consult ladder (#114, PRESERVED through the composed tree). A worker leaf
# consult — with NO SubagentTool to mediate it — propagates all the way up to a
# top-level RunResultConsult. The host (this test) injects an answer via
# resume_consult, the worker finishes, the evaluator passes, and the run
# completes. This exercises the host-mediation seam the 12-cordyceps example
# relies on.
# ---------------------------------------------------------------------------


async def test_worker_consult_surfaces_and_host_resumes() -> None:
    # The GLOBAL tool registry returns a worker-side consult on the first
    # dispatch (the worker's consult_advisor call), then defaults to plain
    # success for anything after.
    tool_registry = ScriptedToolRegistry()
    tool_registry.push(
        ToolOutputConsult.consult(
            ConsultRequest(
                kind="advice",
                situation="found a suspicious unwrap in module one",
                attempts=1,
                question="is this a real defect and how severe?",
            )
        )
    )

    storage = _new_storage()
    # NO consult_handlers: the composed tree has no SubagentTool, so the consult
    # must surface to the host (not be mediated inside the harness).
    h = _harness_for("cordyceps_worker_consult.jsonl", storage, tool_registry=tool_registry)

    # Seed ONE ready task so the execute phase runs exactly one worker.
    session = SessionId("cordyceps-consult")
    tl = TaskList()
    tl.add("audit module one", [])
    await _seed(storage, session, tl)

    # First leg: drive to the consult pause.
    first = await h.run(HarnessRunOptions(_pe_task("cordyceps-consult")))
    assert isinstance(first, RunResultConsult), (
        f"expected RunResultConsult to surface to the host, got {first!r}"
    )
    assert first.request.kind == "advice", "the advice consult reached the host"
    assert "real defect" in first.request.question, (
        "the request carries the worker's question verbatim"
    )
    state = first.state

    # Host mediation: inject the advisor's answer and resume the composed tree.
    resumed = await h.resume_consult(
        state,
        ConsultResponseAnswer(
            text="Yes — unwrap on untrusted input is a real high-severity panic risk.",
        ),
    )
    # The worker continued mid-loop AFTER the consult (the finding it emitted
    # post-answer is the run output) — proving the answer was injected and the
    # SelfVerifying evaluator then cleared the task, not a bare leaf resume.
    assert isinstance(resumed, RunResultSuccess), (
        f"expected Success after resume_consult, got {resumed!r}"
    )
    assert "advisor-confirmed" in resumed.output, (
        f"run output is the post-consult worker finding: {resumed.output}"
    )

    # The worker's task self-verified and completed after the consult.
    after = await _stored_list(storage, session)
    assert all(t.status is TaskStatus.COMPLETED for t in after.tasks), (
        f"the consulted task completed: {[t.status for t in after.tasks]}"
    )


# ---------------------------------------------------------------------------
# AC3: the registered exec-evaluator is Default-FAIL — Passed only on an
# explicit PASS, Failed on indeterminate output (proving the worker self-checks
# before a task clears).
# ---------------------------------------------------------------------------


async def test_self_verified_default_fail() -> None:
    v = _exec_evaluator()
    assert v.max_iterations() == 1, "single read-only evaluator turn"

    def success(out: str) -> RunResultSuccess:
        return RunResultSuccess(
            output=out,
            session_id=SessionId("s"),
            usage=AggregateUsage(),
            turns=1,
        )

    def vinput(eval_text: str) -> VerifierInput:
        return VerifierInput(
            build_result=success("audited module"),
            eval_result=success(eval_text),
            workspace=Path("/tmp"),
            iteration=0,
        )

    assert isinstance(await v.verify(vinput("verdict: PASS")), VerifierVerdictPassed)
    assert isinstance(await v.verify(vinput("looks plausible")), VerifierVerdictFailed)
