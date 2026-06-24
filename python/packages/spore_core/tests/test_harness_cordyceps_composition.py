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
    AgentRef,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetExhaustedEscalate,
    BudgetPolicyPerLoop,
    BudgetPolicyUnlimited,
    ConsultRequest,
    ConsultResponseAnswer,
    EmptyToolRegistry,
    EscalationActionContinueWithBudget,
    EscalationModeAutonomous,
    EscalationModeSurfaceToHuman,
    EvaluatorResponseVerifier,
    ExecutionRegistry,
    HaltReasonTasksBlockedByFailure,
    HarnessConfig,
    HarnessRunOptions,
    HumanRequestBudgetExhausted,
    HumanResponseEscalate,
    InMemoryStorageProvider,
    LoopStrategy,
    Message,
    ModelAgent,
    PausedState,
    PlanExecuteConfig,
    ProviderInfo,
    RalphConfig,
    ReactConfig,
    ReplayModelInterface,
    Role,
    RunResultConsult,
    RunResultFailure,
    RunResultSuccess,
    RunResultWaitingForHuman,
    SchemaRef,
    ScriptedToolRegistry,
    SelfVerifyingConfig,
    SessionId,
    SessionState,
    StandardHarness,
    StorageProvider,
    Task,
    TextContent,
    ToolOutputConsult,
    ToolOutputSuccess,
    ToolsetRef,
    Verifier,
)
from spore_core.harness import AggregateUsage, loop_strategy_max_steps
from spore_core.storage import project_id_from_canonical_path, project_namespace
from spore_core.tasklist import TASK_LIST_EXTRAS_KEY, TaskList, TaskStatus
from spore_core.verifier import VerifierInput, VerifierVerdictFailed, VerifierVerdictPassed

# #142: durable artifacts are keyed by the STABLE project namespace, not the run
# session id. These tests all use ``AllowAllSandbox`` (workspace root ``/``), so
# the harness derives this exact project id — the seed/readback helpers key by it.
_DURABLE_NS = project_namespace(project_id_from_canonical_path("/"))

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

    async def assemble(self, session: SessionState, task: object, sources: object) -> object:
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
    _ = session  # #142: durable write keys by the project namespace.
    await storage.run().put(_DURABLE_NS, TASK_LIST_EXTRAS_KEY, tl.to_dict())


async def _stored_list(storage: StorageProvider, session: SessionId) -> TaskList:
    _ = session  # #142: durable readback keys by the project namespace.
    value = await storage.run().get(_DURABLE_NS, TASK_LIST_EXTRAS_KEY)
    assert value is not None
    return TaskList.from_dict(value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC5 (static): the canonical tree's per-window worst case is computable before
# any run; an Unlimited anywhere collapses it to None.
# ---------------------------------------------------------------------------


def test_cordyceps_max_steps_is_25_unlimited_is_none() -> None:
    tree = _cordyceps_tree()
    assert loop_strategy_max_steps(tree) == 25

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
    raw = (
        _repo_root() / "fixtures" / "paused_states" / "cordyceps_budget_exhausted.json"
    ).read_text()
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


# ===========================================================================
# #138 budget-resume (seed the stalled worker, skip re-planning).
#
# Mirror of the Rust reference's ``budget-resume`` block in
# ``cordyceps_composition_fixture_replay.rs``.
# ===========================================================================


def _small_budget_pe() -> LoopStrategy:
    """A small-budget ``PlanExecute[ ReAct(plan), SelfVerifying[ ReAct(PerLoop{2}) ] ]``
    tree whose worker leaf exhausts after exactly TWO turns — so a budget pause is
    reachable with a tiny fixture. Mirrors the cordyceps execute leaf's handles
    (``executor`` / ``exec-tools`` / ``worker-schema`` / ``exec-evaluator``)."""
    worker = ReactConfig(
        behavior=BudgetExhaustedEscalate(),
        budget=BudgetPolicyPerLoop(value=2),
        agent=AgentRef("executor"),
        toolset=ToolsetRef("exec-tools"),
        output=SchemaRef("worker-schema"),
    )
    plan = ReactConfig(
        behavior=BudgetExhaustedEscalate(),
        budget=BudgetPolicyPerLoop(value=12),
        agent=AgentRef("planner"),
        toolset=ToolsetRef("plan-tools"),
        output=SchemaRef("plan-schema"),
    )
    return PlanExecuteConfig(
        behavior=BudgetExhaustedEscalate(),
        plan=plan,
        execute=SelfVerifyingConfig(
            behavior=BudgetExhaustedEscalate(),
            inner=worker,
            evaluator=SchemaRef("exec-evaluator"),
        ),
        plan_model=None,
    )


def _surface_harness_for(
    fixture: str,
    storage: StorageProvider,
) -> StandardHarness:
    """A SurfaceToHuman harness whose plan/worker/evaluate turns replay
    positionally from ONE shared :class:`ReplayModelInterface`, plus a
    :class:`ScriptedToolRegistry` that returns success for the worker's two
    budget-burning tool calls. Mirrors Rust's ``surface_harness_for``."""
    jsonl = _fixture_path(fixture).read_text()
    replay = ReplayModelInterface.from_jsonl(jsonl, _provider())
    # The worker's two tool calls each dispatch to a plain success (content is
    # irrelevant; they only burn the PerLoop{2} budget).
    tool_registry = ScriptedToolRegistry()
    tool_registry.push(ToolOutputSuccess(content="src/one.rs\nsrc/two.rs", truncated=False))
    tool_registry.push(ToolOutputSuccess(content="fn one() { x.unwrap() }", truncated=False))
    registry = _registry(replay)
    return StandardHarness(
        HarnessConfig(
            agent=ModelAgent(AgentId("ralph-agent"), replay),
            tool_registry=tool_registry,
            sandbox=AllowAllSandbox(),
            context_manager=_StoringContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            storage=storage,
            registry=registry,
            # #138/#130: the worker leaf's budget exhaustion PAUSES.
            escalation_mode=EscalationModeSurfaceToHuman(),
            consult_handlers={},
        )
    )


def _small_pe_task(session: str) -> Task:
    from spore_core import BudgetLimits

    t = Task.new("audit the repo", SessionId(session), _small_budget_pe())
    t.budget = BudgetLimits(max_turns=64)
    return t


# #138 AC2 + AC1: a budget-resume of an execute-phase exhaustion SEEDS the stalled
# worker (carries its full session across the pause) and SKIPS re-planning. Leg 1
# drives the worker leaf to its PerLoop{2} cap and PAUSES with a BudgetExhausted
# request whose PausedState carries the FULL worker session (AC2-a) and the
# ``exec-tools`` handle (AC4-a). Leg 2 (ContinueWithBudget) does NOT re-plan (the
# fixture has NO plan turn) and re-attaches the carried session so the worker
# continues mid-loop to a finding that the evaluator clears.
async def test_budget_resume_seeds_stalled_worker_and_skips_replanning() -> None:
    storage = _new_storage()
    session = SessionId("cordyceps-budget")
    h = _surface_harness_for("cordyceps_budget_resume.jsonl", storage)
    # Pre-seed ONE ready task so AC1's skip-replan precondition holds (non-empty
    # durable list) and the execute phase runs exactly one worker.
    tl = TaskList()
    tl.add("audit module one", [])
    await _seed(storage, session, tl)

    # Leg 1: drive to the budget-exhaustion pause.
    first = await h.run(HarnessRunOptions(_small_pe_task("cordyceps-budget")))
    assert isinstance(first, RunResultWaitingForHuman), (
        f"expected WaitingForHuman budget pause, got {first!r}"
    )
    request, state = first.request, first.state
    # The combinator (PlanExecute) resolves the worker leaf's propagated
    # exhaustion, so the pause's ``phase`` is the resolving scope.
    assert isinstance(request, HumanRequestBudgetExhausted), f"got {request!r}"
    assert request.phase == "plan_execute", "the combinator resolved the exhaustion"
    # AC4-a (#140 parity): the pause carries the worker leaf's toolset handle.
    assert state.toolset == "exec-tools", "AC4-a: budget pause carries worker handle"
    # AC2-a: the pause carries the FULL worker session (instruction + the two
    # budget-burning tool-call rounds), NOT a partial-only stub.
    assert len(state.session_state.messages) > 1, (
        f"AC2-a: full worker session carried, got {len(state.session_state.messages)} messages"
    )
    # AC2 parity: the stalled task stays InProgress on the durable list at the
    # pause (the consult path's invariant) — NOT permanently Blocked — so the
    # resume can re-attach the carried session via InProgress->Pending->complete.
    paused_list = await _stored_list(storage, session)
    assert paused_list.tasks[0].status is TaskStatus.IN_PROGRESS, (
        "the stalled task awaits a budget grant (InProgress, not Blocked)"
    )

    # Leg 2: grant more budget and resume. AC1: NO plan turn in the fixture, so a
    # re-plan would exhaust the positional replay and error — Success proves the
    # plan phase was skipped. AC2-b: the carried session re-attaches to the
    # InProgress task, so the worker continues to its finding and self-verifies.
    resumed = await h.resume(
        state,
        HumanResponseEscalate(action=EscalationActionContinueWithBudget(steps=5)),
    )
    assert isinstance(resumed, RunResultSuccess), (
        f"expected Success after budget resume, got {resumed!r}"
    )
    assert "resume-continued" in resumed.output, (
        f"run output is the post-resume worker finding: {resumed.output}"
    )

    # The resumed task self-verified and completed (InProgress->Pending->Completed
    # — the same transition machinery the consult path uses, AC2 parity).
    after = await _stored_list(storage, session)
    assert all(t.status is TaskStatus.COMPLETED for t in after.tasks), (
        f"the resumed task completed: {[t.status for t in after.tasks]}"
    )


# #138 AC4: the budget-exhausted PausedState fixture round-trips byte-structurally
# — the carried worker session (AC2-a) and the ``exec-tools`` handle (AC4-a)
# survive a serde round-trip identically. This is the cross-language wire-parity
# lock for the four-language ports. Mirrors Rust's
# ``budget_exhausted_paused_state_round_trips``.
def test_budget_exhausted_paused_state_round_trips() -> None:
    raw = (
        _repo_root() / "fixtures" / "paused_states" / "cordyceps_budget_exhausted.json"
    ).read_text()
    value = json.loads(raw)

    typed = PausedState.model_validate(value)
    reser = json.loads(typed.model_dump_json())
    assert reser == value, "PausedState round-trips byte-structurally"

    # AC4-a: the toolset handle is the worker leaf's, always serialized.
    assert value["toolset"] == "exec-tools"
    # AC2-a: the carried session grew beyond the single partial-only stub.
    assert len(value["session_state"]["messages"]) > 1, (
        "AC2-a: the budget-exhausted session carries the worker conversation"
    )


# #138 AC3: plan-phase exhaustion resumes the PLAN session. When a budget resume
# carries a worker session AND the durable task_list is EMPTY (no InProgress task
# ⇒ the exhaustion happened in the PLAN phase), ``PlanExecuteConfig`` seeds the
# PLAN session from the carried conversation instead of a fresh base session — so
# the planner CONTINUES on it.
#
# NOTE (per #138 plan): the carried-session→plan seeding is observed via the
# planner agent's RECORDED contexts; the replay harness's NoopContextManager would
# not otherwise surface it. Mirrors Rust's
# ``budget_resume_plan_phase_seeds_plan_session_from_carried``.
async def test_budget_resume_plan_phase_seeds_plan_session_from_carried() -> None:
    from spore_core import BudgetLimits, FinalResponse, TokenUsage

    marker = "CARRIED_PLAN_SESSION_MARKER"

    class _RecordingPlanner:
        """Records every assembled context's text, then authors a one-task plan."""

        def __init__(self) -> None:
            self.seen: list[str] = []

        async def turn(self, context: object) -> FinalResponse:
            parts = [
                getattr(m.content, "text", "")
                for m in context.messages  # type: ignore[attr-defined]
                if getattr(m.content, "text", "")
            ]
            self.seen.append("\n".join(parts))
            return FinalResponse(
                content='{"tasks":["only"],"rationale":"r"}',
                usage=TokenUsage(),
            )

        def id(self) -> AgentId:
            return AgentId("planner")

    class _Worker:
        async def turn(self, context: object) -> FinalResponse:
            return FinalResponse(content="did the work", usage=TokenUsage())

        def id(self) -> AgentId:
            return AgentId("")

    planner = _RecordingPlanner()
    registry = ExecutionRegistry.builder().agent("planner", planner).agent("", _Worker()).build()
    storage = _new_storage()
    h = StandardHarness(
        HarnessConfig(
            agent=_Worker(),
            tool_registry=ScriptedToolRegistry(),
            sandbox=AllowAllSandbox(),
            context_manager=_StoringContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            storage=storage,
            registry=registry,
            escalation_mode=EscalationModeSurfaceToHuman(),
            consult_handlers={},
        )
    )

    # A PlanExecute whose PLAN leaf resolves to "planner"; execute is a bare ReAct
    # on the default key.
    pe = PlanExecuteConfig(
        behavior=BudgetExhaustedEscalate(),
        plan=ReactConfig(
            behavior=BudgetExhaustedEscalate(),
            budget=BudgetPolicyPerLoop(value=12),
            agent=AgentRef("planner"),
            toolset=ToolsetRef(""),
            output=SchemaRef(""),
        ),
        execute=ReactConfig.per_loop(8),
        plan_model=None,
    )
    t = Task.new("audit the repo", SessionId("s1"), pe)
    t.budget = BudgetLimits(max_turns=32)

    # A budget-exhausted pause carrying a worker session with a MARKER, and NO
    # durable task_list persisted (empty ⇒ plan-phase exhaustion, AC3).
    carried = SessionState(
        messages=[Message(role=Role.ASSISTANT, content=TextContent(text=marker))]
    )
    from spore_core import (
        BudgetSnapshot,
        EscalationActionFail,
        EscalationActionSkip,
        BudgetPolicyTotalSteps,
    )

    state = PausedState(
        session_id=SessionId("s1"),
        task_id=t.id,
        turn_number=1,
        session_state=carried,
        pending_tool_calls=[],
        approved_results=[],
        human_request=HumanRequestBudgetExhausted(
            phase="plan_execute",
            policy=BudgetPolicyTotalSteps(value=1),
            steps_taken=1,
            continues_used=0,
            partial_output=None,
            available_actions=[
                EscalationActionContinueWithBudget(steps=1),
                EscalationActionSkip(),
                EscalationActionFail(),
            ],
        ),
        task=t,
        budget_used=BudgetSnapshot(),
        child_state=None,
        toolset="",
    )

    await h.resume(
        state,
        HumanResponseEscalate(action=EscalationActionContinueWithBudget(steps=10)),
    )

    # AC3: the planner's FIRST context was seeded from the CARRIED session — the
    # marker is present, proving the plan session continued on it rather than
    # starting from a fresh base session.
    assert planner.seen and marker in planner.seen[0], (
        f"AC3: the plan session must be seeded from the carried conversation; "
        f"planner saw: {planner.seen!r}"
    )


# #138 AC1: skip-plan reconciles already-Completed tasks (dedup). A non-empty
# durable task_list whose task #1 is already Completed: a fresh run SKIPS the plan
# phase (AC1) and reconcile does NOT re-run the completed task — only the
# still-Pending task #2 runs (one model call, no plan turn). Mirrors Rust's
# ``skip_plan_reconciles_completed_tasks``.
async def test_skip_plan_reconciles_completed_tasks() -> None:
    from spore_core import BudgetLimits, FinalResponse, MockAgent, TokenUsage

    a = MockAgent(AgentId(""))
    # NO plan turn pushed (AC1 skips it). Only task #2 runs.
    a.push(FinalResponse(content="did two", usage=TokenUsage()))
    storage = _new_storage()
    session = SessionId("s1")
    h = StandardHarness(
        HarnessConfig(
            agent=a,
            tool_registry=ScriptedToolRegistry(),
            sandbox=AllowAllSandbox(),
            context_manager=_StoringContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            storage=storage,
        )
    )

    # Pre-seed: task #1 already Completed, task #2 Pending.
    tl = TaskList()
    tl.add("one", [])  # 1
    tl.add("two", [])  # 2
    tl.update(1, TaskStatus.IN_PROGRESS)
    tl.complete(1)
    await _seed(storage, session, tl)

    pe = PlanExecuteConfig.simple()
    t = Task.new("audit the repo", session, pe)
    t.budget = BudgetLimits(max_turns=64)
    r = await h.run(HarnessRunOptions(t))
    assert isinstance(r, RunResultSuccess), f"expected Success, got {r!r}"
    assert r.output == "did two"
    # Exactly ONE model call: task #2 (no plan turn, task #1 not re-run).
    assert a.call_count == 1, "AC1: plan skipped + completed task #1 deduped — only task #2 ran"
    # Both tasks are Completed in the durable store (1 deduped, 2 freshly run).
    stored = await _stored_list(storage, session)
    assert all(t.status is TaskStatus.COMPLETED for t in stored.tasks)


# #138 AC2-a (unit): the pause helper carries the FULL stalled worker session and
# the worker leaf's toolset handle (AC4-a) — not a partial-only stub. A direct
# unit on the boundary helper, decoupled from the surrounding strategy. Mirrors
# Rust's ``promote_budget_pause_carries_full_worker_session_and_handle``.
def test_promote_budget_pause_carries_full_worker_session_and_handle() -> None:
    from spore_core import BudgetLimits
    from spore_core.harness import (
        BudgetExhausted,
        BudgetSnapshot,
        _leaf_escalation_actions,
        _promote_budget_exhausted_to_human,
    )

    err = BudgetExhausted(
        policy=BudgetPolicyPerLoop(value=2),
        behavior=BudgetExhaustedEscalate(),
        steps_taken=2,
        continues_used=0,
        phase="react",
    )
    react = ReactConfig.per_loop(2)
    task = Task.new("worker", SessionId("s1"), react, budget=BudgetLimits(max_turns=2))
    # A realistic worker conversation (instruction + a tool round).
    worker = SessionState(
        messages=[
            Message(role=Role.USER, content=TextContent(text="worker: audit")),
            Message(role=Role.ASSISTANT, content=TextContent(text="looking")),
            Message(role=Role.TOOL, content=TextContent(text="listing")),
        ]
    )
    waiting = _promote_budget_exhausted_to_human(
        err,
        "partial",
        _leaf_escalation_actions(err),
        SessionId("s1"),
        task,
        BudgetSnapshot(),
        2,
        worker,
        ToolsetRef("exec-tools"),
    )
    assert isinstance(waiting, RunResultWaitingForHuman)
    # AC2-a: the FULL worker session is carried (3 messages), NOT the single
    # partial-only assistant stub.
    assert waiting.state.session_state.messages == worker.messages
    # AC4-a: the worker leaf's toolset handle rides the pause (#140 parity).
    assert waiting.state.toolset == "exec-tools"

    # Back-compat: an EMPTY worker session falls back to the partial-only stub (the
    # pre-#138 behavior) so legacy / HillClimbing sites are unchanged.
    waiting2 = _promote_budget_exhausted_to_human(
        err,
        "just-the-partial",
        _leaf_escalation_actions(err),
        SessionId("s1"),
        task,
        BudgetSnapshot(),
        2,
        SessionState(),
        ToolsetRef(""),
    )
    assert isinstance(waiting2, RunResultWaitingForHuman)
    msgs = waiting2.state.session_state.messages
    assert len(msgs) == 1
    assert getattr(msgs[0].content, "text", None) == "just-the-partial"
