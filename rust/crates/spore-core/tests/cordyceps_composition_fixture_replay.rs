//! Fixture-replay integration tests for the cordyceps composition (#131):
//! `Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]`, driven by the canonical
//! `fixtures/strategy/cordyceps_tree.json`.
//!
//! These exercise the SAME recorded-model harness as
//! `plan_execute_dag_fixture_replay.rs`, but against the FULL composed tree with
//! its real handles wired into an `ExecutionRegistry`: agents
//! `planner`/`executor`/`ralph-agent`, toolsets `plan-tools`/`exec-tools`,
//! schemas `plan-schema`/`worker-schema`, and the Default-FAIL `exec-evaluator`
//! verifier. Never edit a fixture to make a failing implementation pass — the
//! fixtures are ground truth and must stay internally consistent.

use std::sync::Arc;

use spore_core::harness::testing::{
    AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager, ScriptedToolRegistry,
};
use spore_core::harness::ToolOutput;
use spore_core::{
    Agent, AgentId, BudgetPolicy, ConsultRequest, ConsultResponse, EscalationMode,
    EvaluatorResponseVerifier, ExecutionRegistry, HaltReason, Harness, HarnessConfig,
    HarnessRunOptions, LoopStrategy, ModelAgent, ProviderInfo, ReplayModelInterface, RunResult,
    SessionId, StandardHarness, StorageProvider, Task, TaskList, TaskStatus, Verifier,
    TASK_LIST_EXTRAS_KEY,
};

fn provider() -> ProviderInfo {
    ProviderInfo {
        name: "anthropic".into(),
        model_id: "fixture".into(),
        context_window: 200_000,
    }
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("fixtures/model_responses/harness")
        .join(name)
}

/// The Default-FAIL evaluator the `exec-evaluator` handle resolves to — the same
/// construction the 12-cordyceps example registers (single read-only turn;
/// neither-pattern ⇒ Failed).
fn exec_evaluator() -> Arc<dyn Verifier> {
    Arc::new(EvaluatorResponseVerifier::new(r"(?i)\bPASS\b", r"(?i)\bFAIL\b", 1).unwrap())
}

/// Build a harness whose plan/worker/evaluator turns all replay positionally
/// from ONE shared `ReplayModelInterface` (a single cursor across the whole
/// composed run), with the cordyceps handles wired into the registry.
fn harness_for(fixture: &str) -> (StandardHarness, Arc<StorageProvider>) {
    let jsonl = std::fs::read_to_string(fixture_path(fixture))
        .unwrap_or_else(|e| panic!("read {fixture}: {e}"));
    // One replay backend, shared by all three node agents so the positional
    // cursor advances across plan → worker → evaluate uniformly.
    let replay = Arc::new(ReplayModelInterface::from_jsonl(&jsonl, provider()).expect("fixture"));
    let agent = |id: &str| -> Arc<dyn Agent> {
        Arc::new(ModelAgent::new(AgentId::new(id), replay.clone()))
    };
    let storage = Arc::new(StorageProvider::single(Arc::new(
        spore_core::storage::InMemoryStorageProvider::new(),
    )));
    let registry = ExecutionRegistry::builder()
        .agent("planner", agent("planner"))
        .agent("executor", agent("executor"))
        .agent("ralph-agent", agent("ralph-agent"))
        .toolset("plan-tools", Arc::new(spore_core::EmptyToolRegistry))
        .toolset("exec-tools", Arc::new(spore_core::EmptyToolRegistry))
        .schema("plan-schema", serde_json::json!({"type": "object"}))
        .schema("worker-schema", serde_json::json!({"type": "array"}))
        .verifier("exec-evaluator", exec_evaluator())
        .build();
    let cfg = HarnessConfig {
        tool_registry: Arc::new(ScriptedToolRegistry::new()),
        sandbox: Arc::new(AllowAllSandbox),
        context_manager: Arc::new(NoopContextManager),
        termination_policy: Arc::new(AlwaysContinuePolicy),
        toolset_catalogues: Default::default(),
        middleware: None,
        observability: None,
        compaction_verifier: Arc::new(spore_core::KeyTermVerifier),
        max_compaction_attempts: 2,
        pricing: spore_core::PricingTable::DEFAULT,
        content_capture: spore_core::ContentCaptureConfig::default(),
        tool_call_repair: None,
        max_repair_attempts: 1,
        max_stop_blocks: 8,
        error_loop_threshold: 3,
        hooks: None,
        storage: storage.clone(),
        // #142: durable artifacts (task_list) are keyed by the project namespace;
        // pin a known project id so `seed`/`stored_list` key the SAME namespace.
        project_id: spore_core::ProjectId::from_canonical_path(CORDYCEPS_PROJECT_PATH),
        chunk_provider: Arc::new(spore_core::prompt_assembly::InMemoryChunkProvider::empty()),
        max_resets: 3,
        vcs_provider: None,
        catalogue_registry: None,
        system_prompt: None,
        model_params: spore_core::ModelParams::default(),
        auto_persist_sessions: false,
        prompt_tool_call_flag: None,
        consult_handlers: std::collections::HashMap::new(),
        registry,
        escalation_mode: EscalationMode::Autonomous,
    };
    (StandardHarness::new(cfg), storage)
}

/// The canonical cordyceps tree, deserialized from the shared fixture (the same
/// `include_str!` path the example uses).
fn cordyceps_tree() -> LoopStrategy {
    let json = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join("fixtures/strategy/cordyceps_tree.json"),
    )
    .expect("cordyceps_tree.json present");
    serde_json::from_str(&json).expect("cordyceps_tree.json deserializes")
}

/// The PlanExecute subtree of the cordyceps tree (drops the Ralph wrapper) so
/// the positional fixture maps 1:1 to one window — Ralph's per-window reset loop
/// would otherwise re-enter and re-consume the (exhausted) replay queue.
fn cordyceps_plan_execute() -> LoopStrategy {
    let LoopStrategy::Ralph(ralph) = cordyceps_tree() else {
        panic!("root is Ralph");
    };
    *ralph.inner
}

fn pe_task(session: &str) -> Task {
    let mut t = Task::new(
        "audit the repo",
        SessionId::new(session),
        cordyceps_plan_execute(),
    );
    t.budget.max_turns = Some(64);
    t
}

/// #142: the fixed canonical path the test harness derives its durable
/// `project_id` from (pinned via the `project_id` field in the harness configs).
const CORDYCEPS_PROJECT_PATH: &str = "/cordyceps-workspace";

/// The durable `RunStore` namespace the harness keys task_list under (#142). The
/// project id projected onto the session-id axis — `seed`/`stored_list` key this,
/// NOT the ephemeral run session, so the seeded list is what the harness reads.
fn project_ns() -> SessionId {
    spore_core::ProjectId::from_canonical_path(CORDYCEPS_PROJECT_PATH).namespace()
}

async fn seed(storage: &StorageProvider, _session: &SessionId, list: &TaskList) {
    storage
        .run()
        .put(
            &project_ns(),
            TASK_LIST_EXTRAS_KEY,
            serde_json::to_value(list).unwrap(),
        )
        .await
        .unwrap();
}

async fn stored_list(storage: &StorageProvider, _session: &SessionId) -> TaskList {
    serde_json::from_value(
        storage
            .run()
            .get(&project_ns(), TASK_LIST_EXTRAS_KEY)
            .await
            .unwrap()
            .unwrap(),
    )
    .unwrap()
}

// AC5 (static): the canonical tree's per-window worst case is computable before
// any run; an Unlimited anywhere collapses it to None.
#[test]
fn cordyceps_max_steps_is_25_unlimited_is_none() {
    let tree = cordyceps_tree();
    assert_eq!(tree.max_steps(), Some(25));

    // Swap the worker leaf's PerLoop{12} for Unlimited ⇒ None.
    let LoopStrategy::Ralph(mut ralph) = tree else {
        unreachable!()
    };
    let LoopStrategy::PlanExecute(pe) = ralph.inner.as_mut() else {
        unreachable!()
    };
    let LoopStrategy::SelfVerifying(sv) = pe.execute.as_mut() else {
        unreachable!()
    };
    let LoopStrategy::ReAct(worker) = sv.inner.as_mut() else {
        unreachable!()
    };
    worker.budget = BudgetPolicy::Unlimited;
    assert_eq!(LoopStrategy::Ralph(ralph).max_steps(), None);
}

// AC6 (handle re-resolution): a paused cordyceps tree resumes by re-resolving
// EVERY handle from a freshly-built registry, with no reconfiguration. Load the
// paused-state fixture carrying the FULL tree, serde round-trip its Task, build
// a fresh registry, and assert validate() is_ok() + every handle resolves.
#[test]
fn resume_reresolves_handles() {
    let raw = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join("fixtures/paused_states/cordyceps_budget_exhausted.json"),
    )
    .expect("cordyceps_budget_exhausted.json present");
    let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();

    // The paused state carries the FULL cordyceps tree in task.loop_strategy.
    let task: Task = serde_json::from_value(doc["task"].clone()).expect("Task deserializes");

    // Serde round-trip the Task (trait objects never enter the wire).
    let wire = serde_json::to_string(&task).unwrap();
    let restored: Task = serde_json::from_str(&wire).unwrap();

    // A fresh registry, built independently (as on a cold resume), re-resolves
    // every handle — proving no reconfiguration of the Task is needed.
    let registry = fresh_registry();
    assert!(
        registry.validate(&restored).is_ok(),
        "every handle must re-resolve"
    );

    // Spot-check the load-bearing handles resolve concretely.
    let LoopStrategy::Ralph(ralph) = &restored.loop_strategy else {
        panic!("root is Ralph after round-trip");
    };
    assert!(
        registry.resolve_agent(&ralph.agent).is_some(),
        "ralph-agent resolves"
    );
    let LoopStrategy::PlanExecute(pe) = ralph.inner.as_ref() else {
        unreachable!()
    };
    let LoopStrategy::ReAct(plan) = pe.plan.as_ref() else {
        unreachable!()
    };
    assert!(
        registry.resolve_agent(&plan.agent).is_some(),
        "planner resolves"
    );
    assert!(
        registry.resolve_toolset(&plan.toolset).is_some(),
        "plan-tools resolves"
    );
    assert!(registry
        .resolve_schema(plan.output.as_ref().unwrap())
        .is_some());
    let LoopStrategy::SelfVerifying(sv) = pe.execute.as_ref() else {
        unreachable!()
    };
    // The evaluator handle resolves against the verifier map.
    assert!(
        registry.resolve_verifier(&sv.evaluator.0).is_some(),
        "exec-evaluator resolves"
    );
    let LoopStrategy::ReAct(worker) = sv.inner.as_ref() else {
        unreachable!()
    };
    assert!(
        registry.resolve_agent(&worker.agent).is_some(),
        "executor resolves"
    );
    assert!(
        registry.resolve_toolset(&worker.toolset).is_some(),
        "exec-tools resolves"
    );

    // The fixture's available_actions advertise the combinator escalation menu.
    let actions = &doc["human_request"]["available_actions"];
    let kinds: Vec<&str> = actions
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, vec!["continue_with_budget", "skip", "fail"]);
}

/// A fresh `ExecutionRegistry` wired with the cordyceps handles (no model
/// backend needed — handle resolution is structural).
fn fresh_registry() -> ExecutionRegistry {
    let jsonl = std::fs::read_to_string(fixture_path("plan_execute_dag_order.jsonl")).unwrap();
    let replay = Arc::new(ReplayModelInterface::from_jsonl(&jsonl, provider()).unwrap());
    let agent = |id: &str| -> Arc<dyn Agent> {
        Arc::new(ModelAgent::new(AgentId::new(id), replay.clone()))
    };
    ExecutionRegistry::builder()
        .agent("planner", agent("planner"))
        .agent("executor", agent("executor"))
        .agent("ralph-agent", agent("ralph-agent"))
        .toolset("plan-tools", Arc::new(spore_core::EmptyToolRegistry))
        .toolset("exec-tools", Arc::new(spore_core::EmptyToolRegistry))
        .schema("plan-schema", serde_json::json!({"type": "object"}))
        .schema("worker-schema", serde_json::json!({"type": "array"}))
        .verifier("exec-evaluator", exec_evaluator())
        .build()
}

// AC2: the plan phase builds a blocker-aware task graph (seeded via task_list,
// the decision-C authoring path) and the execute phase walks it as a READY-SET,
// self-verifying each task. Two independent modules both complete in ready-set
// order; the run succeeds.
#[tokio::test]
async fn plan_builds_dag_execute_walks_readyset() {
    let (h, storage) = harness_for("cordyceps_plan_execute_readyset.jsonl");
    let session = SessionId::new("cordyceps-pe");
    let mut tl = TaskList::default();
    tl.add("audit module one".into(), vec![]).unwrap(); // 1
    tl.add("audit module two".into(), vec![]).unwrap(); // 2 (independent)
    seed(&storage, &session, &tl).await;

    match h.run(HarnessRunOptions::new(pe_task("cordyceps-pe"))).await {
        RunResult::Success { .. } => {}
        other => panic!("expected Success, got {other:?}"),
    }
    // Every ready task was walked and self-verified to completion.
    let after = stored_list(&storage, &session).await;
    assert!(
        after
            .tasks
            .iter()
            .all(|t| t.status == TaskStatus::Completed),
        "all ready-set tasks complete: {:?}",
        after.tasks.iter().map(|t| t.status).collect::<Vec<_>>()
    );
}

// #stream: streaming threads through the COMPOSED tree. ONE top-level sink
// receives every node's events, each stamped with the node that produced it
// (kind / agent_id / depth / root→leaf path). This is the seam an API backing a
// web UI subscribes to and forwards over SSE. Proven here against the same
// worker→evaluator replay as the ready-set test: worker turns carry the
// `executor` leaf nested under `plan_execute → self_verifying`.
//
// #138 AC1: the durable task_list is pre-seeded, so the plan phase is SKIPPED —
// there is NO `planner`-attributed event. The execute (worker) leaf still
// streams with full nesting attribution, which is what this test now asserts.
#[tokio::test]
async fn composed_tree_streams_attributed_events() {
    use spore_core::HarnessStreamEvent;

    let (h, storage) = harness_for("cordyceps_plan_execute_readyset.jsonl");
    let session = SessionId::new("cordyceps-stream");
    let mut tl = TaskList::default();
    tl.add("audit module one".into(), vec![]).unwrap();
    tl.add("audit module two".into(), vec![]).unwrap();
    seed(&storage, &session, &tl).await;

    let events: Arc<std::sync::Mutex<Vec<HarnessStreamEvent>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = events.clone();
    let result = h
        .run(
            HarnessRunOptions::new(pe_task("cordyceps-stream"))
                .with_stream(move |ev| sink.lock().unwrap().push(ev)),
        )
        .await;
    assert!(matches!(result, RunResult::Success { .. }), "{result:?}");

    let events = events.lock().unwrap();
    assert!(!events.is_empty(), "the composed tree streamed events");

    // EVERY event from a node inside the tree is attributed — no anonymous
    // frames reach the sink (the combinators no longer suppress streaming).
    assert!(
        events.iter().all(|e| e.node().is_some()),
        "every streamed event carries node attribution"
    );

    // Distinct leaf agents are distinguishable on the wire (who is speaking).
    let agents: std::collections::HashSet<&str> = events
        .iter()
        .filter_map(|e| e.node().and_then(|n| n.agent_id.as_deref()))
        .collect();
    // #138 AC1: the plan phase is skipped, so there is NO planner event.
    assert!(
        !agents.contains("planner"),
        "#138 AC1: skip-plan emits no planner event: {agents:?}"
    );
    assert!(
        agents.contains("executor"),
        "worker turns attributed to executor: {agents:?}"
    );

    // The worker leaf sits under `plan_execute → self_verifying` — the path/depth
    // carry the full nesting attribution even though the plan phase was skipped.
    let executor = events
        .iter()
        .find_map(|e| {
            e.node()
                .filter(|n| n.agent_id.as_deref() == Some("executor"))
        })
        .expect("an executor-attributed event");

    assert_eq!(executor.kind, "react");
    assert_eq!(
        executor.path,
        vec!["plan_execute", "self_verifying", "react"]
    );
    assert!(
        executor.depth >= 2,
        "the worker leaf carries its full nesting depth: {}",
        executor.depth
    );
}

// AC4: a single runaway worker node exhausts its own PerLoop{12} budget and
// FAILS its task; an INDEPENDENT module still completes. The PlanExecute drains
// to TasksBlockedByFailure with a partition that does NOT cascade to the
// unrelated branch.
#[tokio::test]
async fn cordyceps_runaway_bounded() {
    let (h, storage) = harness_for("cordyceps_runaway_bounded.jsonl");
    let session = SessionId::new("cordyceps-runaway");
    let mut tl = TaskList::default();
    tl.add("root module".into(), vec![]).unwrap(); // 1 (completes)
    tl.add("runaway module".into(), vec![1]).unwrap(); // 2 -> 1 (PerLoop{12} budget-Fail)
    tl.add("dependent of runaway".into(), vec![2]).unwrap(); // 3 -> 2 (cascade-blocked)
    tl.add("independent module".into(), vec![]).unwrap(); // 4 (still completes)
    seed(&storage, &session, &tl).await;

    match h
        .run(HarnessRunOptions::new(pe_task("cordyceps-runaway")))
        .await
    {
        RunResult::Failure {
            reason:
                HaltReason::TasksBlockedByFailure {
                    completed,
                    blocked,
                    failed_task,
                    ..
                },
            ..
        } => {
            assert_eq!(failed_task, 2, "the runaway module is the failed task");
            // The runaway (2) and its transitive dependent (3) are blocked; the
            // root (1) and the UNRELATED independent module (4) both complete —
            // the runaway's bounded failure does NOT cascade to unrelated tasks.
            assert_eq!(completed, vec![1, 4], "root + independent branch complete");
            assert_eq!(blocked, vec![2, 3], "runaway + its dependent are blocked");
        }
        other => panic!("expected TasksBlockedByFailure, got {other:?}"),
    }
}

// Consult ladder (#114, PRESERVED through the composed tree). A worker leaf
// consult — with NO `SubagentTool` to mediate it — propagates all the way up to
// a top-level `RunResult::Consult`. The host (this test) injects an answer via
// `resume_consult`, the worker finishes, the evaluator passes, and the run
// completes. This exercises the host-mediation seam the 12-cordyceps example
// relies on.
#[tokio::test]
async fn worker_consult_surfaces_and_host_resumes() {
    // Build a harness whose GLOBAL tool registry returns a worker-side consult
    // on the first dispatch (the worker's `consult_advisor` call), then defaults
    // to plain success for anything after.
    let tool_registry = Arc::new(spore_core::harness::testing::ScriptedToolRegistry::new());
    tool_registry.push(ToolOutput::Consult {
        child_state: None,
        request: ConsultRequest {
            kind: "advice".into(),
            situation: "found a suspicious unwrap in module one".into(),
            attempts: 1,
            question: "is this a real defect and how severe?".into(),
        },
    });

    let jsonl = std::fs::read_to_string(fixture_path("cordyceps_worker_consult.jsonl"))
        .expect("read consult fixture");
    let replay = Arc::new(ReplayModelInterface::from_jsonl(&jsonl, provider()).expect("fixture"));
    let agent = |id: &str| -> Arc<dyn Agent> {
        Arc::new(ModelAgent::new(AgentId::new(id), replay.clone()))
    };
    let storage = Arc::new(StorageProvider::single(Arc::new(
        spore_core::storage::InMemoryStorageProvider::new(),
    )));
    let registry = ExecutionRegistry::builder()
        .agent("planner", agent("planner"))
        .agent("executor", agent("executor"))
        .agent("ralph-agent", agent("ralph-agent"))
        .toolset("plan-tools", Arc::new(spore_core::EmptyToolRegistry))
        .toolset("exec-tools", Arc::new(spore_core::EmptyToolRegistry))
        .schema("plan-schema", serde_json::json!({"type": "object"}))
        .schema("worker-schema", serde_json::json!({"type": "array"}))
        .verifier("exec-evaluator", exec_evaluator())
        .build();
    let cfg = HarnessConfig {
        tool_registry,
        sandbox: Arc::new(AllowAllSandbox),
        context_manager: Arc::new(NoopContextManager),
        termination_policy: Arc::new(AlwaysContinuePolicy),
        toolset_catalogues: Default::default(),
        middleware: None,
        observability: None,
        compaction_verifier: Arc::new(spore_core::KeyTermVerifier),
        max_compaction_attempts: 2,
        pricing: spore_core::PricingTable::DEFAULT,
        content_capture: spore_core::ContentCaptureConfig::default(),
        tool_call_repair: None,
        max_repair_attempts: 1,
        max_stop_blocks: 8,
        error_loop_threshold: 3,
        hooks: None,
        storage: storage.clone(),
        // #142: durable artifacts (task_list) are keyed by the project namespace;
        // pin a known project id so `seed`/`stored_list` key the SAME namespace.
        project_id: spore_core::ProjectId::from_canonical_path(CORDYCEPS_PROJECT_PATH),
        chunk_provider: Arc::new(spore_core::prompt_assembly::InMemoryChunkProvider::empty()),
        max_resets: 3,
        vcs_provider: None,
        catalogue_registry: None,
        system_prompt: None,
        model_params: spore_core::ModelParams::default(),
        auto_persist_sessions: false,
        prompt_tool_call_flag: None,
        // NO consult_handlers: the composed tree has no SubagentTool, so the
        // consult must surface to the host (not be mediated inside the harness).
        consult_handlers: std::collections::HashMap::new(),
        registry,
        escalation_mode: EscalationMode::Autonomous,
    };
    let h = StandardHarness::new(cfg);

    // Seed ONE ready task so the execute phase runs exactly one worker.
    let session = SessionId::new("cordyceps-consult");
    let mut tl = TaskList::default();
    tl.add("audit module one".into(), vec![]).unwrap();
    seed(&storage, &session, &tl).await;

    // First leg: drive to the consult pause.
    let first = h
        .run(HarnessRunOptions::new(pe_task("cordyceps-consult")))
        .await;
    let (request, state) = match first {
        RunResult::Consult { request, state, .. } => (request, state),
        other => panic!("expected RunResult::Consult to surface to the host, got {other:?}"),
    };
    assert_eq!(
        request.kind, "advice",
        "the advice consult reached the host"
    );
    assert!(
        request.question.contains("real defect"),
        "the request carries the worker's question verbatim"
    );

    // Host mediation: inject the advisor's answer and resume the composed tree.
    let resumed = h
        .resume_consult(
            *state,
            ConsultResponse::Answer {
                text: "Yes — unwrap on untrusted input is a real high-severity panic risk.".into(),
            },
            None,
        )
        .await;
    match resumed {
        // The worker continued mid-loop AFTER the consult (the finding it emitted
        // post-answer is the run output) — proving the answer was injected and the
        // SelfVerifying evaluator then cleared the task, not a bare leaf resume.
        RunResult::Success { output, .. } => assert!(
            output.contains("advisor-confirmed"),
            "run output is the post-consult worker finding: {output}"
        ),
        other => panic!("expected Success after resume_consult, got {other:?}"),
    }

    // The worker's task self-verified and completed after the consult.
    let after = stored_list(&storage, &session).await;
    assert!(
        after
            .tasks
            .iter()
            .all(|t| t.status == TaskStatus::Completed),
        "the consulted task completed: {:?}",
        after.tasks.iter().map(|t| t.status).collect::<Vec<_>>()
    );
}

// ── #138 budget-resume (seed the stalled worker, skip re-planning) ──────────

/// A small-budget `PlanExecute[ ReAct(plan), SelfVerifying[ ReAct(PerLoop{2}) ] ]`
/// tree whose worker leaf exhausts after exactly TWO turns — so a budget pause is
/// reachable with a tiny fixture. Mirrors the cordyceps execute leaf's handles
/// (`executor` / `exec-tools` / `worker-schema` / `exec-evaluator`).
fn small_budget_pe() -> LoopStrategy {
    use spore_core::{
        AgentRef, BudgetExhaustedBehavior, BudgetPolicy, PlanExecuteConfig, ReactConfig, SchemaRef,
        SelfVerifyingConfig, ToolsetRef,
    };
    let worker = LoopStrategy::ReAct(ReactConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        budget: BudgetPolicy::PerLoop { value: 2 },
        agent: AgentRef("executor".into()),
        toolset: ToolsetRef("exec-tools".into()),
        output: Some(SchemaRef("worker-schema".into())),
    });
    let plan = LoopStrategy::ReAct(ReactConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        budget: BudgetPolicy::PerLoop { value: 12 },
        agent: AgentRef("planner".into()),
        toolset: ToolsetRef("plan-tools".into()),
        output: Some(SchemaRef("plan-schema".into())),
    });
    LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        plan: Box::new(plan),
        execute: Box::new(LoopStrategy::SelfVerifying(SelfVerifyingConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            inner: Box::new(worker),
            evaluator: SchemaRef("exec-evaluator".into()),
        })),
        plan_model: None,
    })
}

/// A SurfaceToHuman harness whose plan/worker/evaluate turns replay positionally
/// from ONE shared `ReplayModelInterface` plus a `ScriptedToolRegistry` that
/// returns success for the worker's two budget-burning tool calls.
fn surface_harness_for(fixture: &str) -> (StandardHarness, Arc<StorageProvider>) {
    let jsonl = std::fs::read_to_string(fixture_path(fixture))
        .unwrap_or_else(|e| panic!("read {fixture}: {e}"));
    let replay = Arc::new(ReplayModelInterface::from_jsonl(&jsonl, provider()).expect("fixture"));
    let agent = |id: &str| -> Arc<dyn Agent> {
        Arc::new(ModelAgent::new(AgentId::new(id), replay.clone()))
    };
    // The worker's two tool calls each dispatch to a plain success (content is
    // irrelevant; they only burn the PerLoop{2} budget).
    let tool_registry = Arc::new(spore_core::harness::testing::ScriptedToolRegistry::new());
    tool_registry.push(ToolOutput::Success {
        content: "src/one.rs\nsrc/two.rs".into(),
        truncated: false,
    });
    tool_registry.push(ToolOutput::Success {
        content: "fn one() { x.unwrap() }".into(),
        truncated: false,
    });
    let storage = Arc::new(StorageProvider::single(Arc::new(
        spore_core::storage::InMemoryStorageProvider::new(),
    )));
    let registry = ExecutionRegistry::builder()
        .agent("planner", agent("planner"))
        .agent("executor", agent("executor"))
        .agent("ralph-agent", agent("ralph-agent"))
        .toolset("plan-tools", Arc::new(spore_core::EmptyToolRegistry))
        .toolset("exec-tools", Arc::new(spore_core::EmptyToolRegistry))
        .schema("plan-schema", serde_json::json!({"type": "object"}))
        .schema("worker-schema", serde_json::json!({"type": "array"}))
        .verifier("exec-evaluator", exec_evaluator())
        .build();
    let cfg = HarnessConfig {
        tool_registry,
        sandbox: Arc::new(AllowAllSandbox),
        context_manager: Arc::new(NoopContextManager),
        termination_policy: Arc::new(AlwaysContinuePolicy),
        toolset_catalogues: Default::default(),
        middleware: None,
        observability: None,
        compaction_verifier: Arc::new(spore_core::KeyTermVerifier),
        max_compaction_attempts: 2,
        pricing: spore_core::PricingTable::DEFAULT,
        content_capture: spore_core::ContentCaptureConfig::default(),
        tool_call_repair: None,
        max_repair_attempts: 1,
        max_stop_blocks: 8,
        error_loop_threshold: 3,
        hooks: None,
        storage: storage.clone(),
        project_id: spore_core::ProjectId::from_canonical_path(CORDYCEPS_PROJECT_PATH),
        chunk_provider: Arc::new(spore_core::prompt_assembly::InMemoryChunkProvider::empty()),
        max_resets: 3,
        vcs_provider: None,
        catalogue_registry: None,
        system_prompt: None,
        model_params: spore_core::ModelParams::default(),
        auto_persist_sessions: false,
        prompt_tool_call_flag: None,
        consult_handlers: std::collections::HashMap::new(),
        registry,
        // #138/#130: the worker leaf's budget exhaustion PAUSES (WaitingForHuman).
        escalation_mode: EscalationMode::SurfaceToHuman,
    };
    (StandardHarness::new(cfg), storage)
}

fn small_pe_task(session: &str) -> Task {
    let mut t = Task::new("audit the repo", SessionId::new(session), small_budget_pe());
    t.budget.max_turns = Some(64);
    t
}

// #138 AC2 + AC1: a budget-resume of an execute-phase exhaustion SEEDS the
// stalled worker (carries its full session across the pause) and SKIPS
// re-planning. Leg 1 drives the worker leaf to its PerLoop{2} cap and PAUSES with
// a `BudgetExhausted` request whose `PausedState` carries the FULL worker session
// (AC2-a) and the `exec-tools` handle (AC4-a). Leg 2 (`ContinueWithBudget`) does
// NOT re-plan (the fixture has NO plan turn) and re-attaches the carried session
// so the worker continues mid-loop to a finding that the evaluator clears.
#[tokio::test]
async fn budget_resume_seeds_stalled_worker_and_skips_replanning() {
    use spore_core::{EscalationAction, HumanRequest, HumanResponse};

    let (h, storage) = surface_harness_for("cordyceps_budget_resume.jsonl");
    let session = SessionId::new("cordyceps-budget");
    // Pre-seed ONE ready task so AC1's skip-replan precondition holds (non-empty
    // durable list) and the execute phase runs exactly one worker.
    let mut tl = TaskList::default();
    tl.add("audit module one".into(), vec![]).unwrap();
    seed(&storage, &session, &tl).await;

    // Leg 1: drive to the budget-exhaustion pause.
    let first = h
        .run(HarnessRunOptions::new(small_pe_task("cordyceps-budget")))
        .await;
    let (request, state) = match first {
        RunResult::WaitingForHuman { request, state } => (request, state),
        other => panic!("expected WaitingForHuman budget pause, got {other:?}"),
    };
    match &request {
        // The combinator (PlanExecute) resolves the worker leaf's propagated
        // exhaustion, so the pause's `phase` is the resolving scope.
        HumanRequest::BudgetExhausted { phase, .. } => {
            assert_eq!(
                phase, "plan_execute",
                "the combinator resolved the exhaustion"
            );
        }
        other => panic!("expected BudgetExhausted, got {other:?}"),
    }
    // AC4-a (#140 parity): the pause carries the worker leaf's toolset handle.
    assert_eq!(
        state.toolset.0, "exec-tools",
        "AC4-a: budget pause carries the worker leaf's handle"
    );
    // AC2-a: the pause carries the FULL worker session (instruction + the two
    // budget-burning tool-call rounds), NOT a partial-only stub.
    assert!(
        state.session_state.messages.len() > 1,
        "AC2-a: full worker session carried, got {} messages",
        state.session_state.messages.len()
    );
    // AC2 parity: the stalled task stays `InProgress` on the durable list at the
    // pause (the consult path's invariant) — NOT permanently Blocked — so the
    // resume can re-attach the carried session via InProgress->Pending->complete.
    let paused_list = stored_list(&storage, &session).await;
    assert_eq!(
        paused_list.tasks[0].status,
        TaskStatus::InProgress,
        "the stalled task awaits a budget grant (InProgress, not Blocked)"
    );

    // Leg 2: grant more budget and resume. AC1: NO plan turn in the fixture, so a
    // re-plan would exhaust the positional replay and error — Success proves the
    // plan phase was skipped. AC2-b: the carried session re-attaches to the
    // InProgress task, so the worker continues to its finding and self-verifies.
    let resumed = h
        .resume(
            *state,
            HumanResponse::Escalate {
                action: EscalationAction::ContinueWithBudget { steps: 5 },
            },
            None,
        )
        .await;
    match resumed {
        RunResult::Success { output, .. } => assert!(
            output.contains("resume-continued"),
            "run output is the post-resume worker finding: {output}"
        ),
        other => panic!("expected Success after budget resume, got {other:?}"),
    }

    // The resumed task self-verified and completed (InProgress->Pending->Completed
    // — the same transition machinery the consult path uses, AC2 parity).
    let after = stored_list(&storage, &session).await;
    assert!(
        after
            .tasks
            .iter()
            .all(|t| t.status == TaskStatus::Completed),
        "the resumed task completed: {:?}",
        after.tasks.iter().map(|t| t.status).collect::<Vec<_>>()
    );
}

// #138 AC4: the budget-exhausted `PausedState` fixture round-trips
// byte-structurally — the carried worker session (AC2-a) and the `exec-tools`
// handle (AC4-a) survive a serde round-trip identically. This is the
// cross-language wire-parity lock for the four-language ports.
#[test]
fn budget_exhausted_paused_state_round_trips() {
    use spore_core::PausedState;
    let raw = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join("fixtures/paused_states/cordyceps_budget_exhausted.json"),
    )
    .expect("cordyceps_budget_exhausted.json present");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("valid JSON");

    let typed: PausedState =
        serde_json::from_value(value.clone()).expect("PausedState deserializes");
    let reser = serde_json::to_value(&typed).expect("re-serialize");
    assert_eq!(reser, value, "PausedState round-trips byte-structurally");

    // AC4-a: the toolset handle is the worker leaf's, always serialized.
    assert_eq!(value["toolset"], "exec-tools");
    // AC2-a: the carried session grew beyond the single partial-only stub.
    assert!(
        value["session_state"]["messages"].as_array().unwrap().len() > 1,
        "AC2-a: the budget-exhausted session carries the worker conversation"
    );
}

// AC3: the registered `exec-evaluator` is Default-FAIL — Passed only on an
// explicit PASS, Failed on indeterminate output (proving the worker self-checks
// before a task clears).
#[tokio::test]
async fn self_verified_default_fail() {
    use spore_core::{AggregateUsage, SessionState, VerifierInput, VerifierVerdict};
    let v = exec_evaluator();
    assert_eq!(v.max_iterations(), 1, "single read-only evaluator turn");
    let success = |out: &str| RunResult::Success {
        output: out.into(),
        session_id: SessionId::new("s"),
        usage: AggregateUsage::default(),
        turns: 1,
        session_state: SessionState::default(),
    };
    let input = |eval: &str| VerifierInput {
        build_result: success("audited module"),
        eval_result: success(eval),
        workspace: std::path::PathBuf::from("/tmp"),
        iteration: 0,
    };
    assert_eq!(
        v.verify(&input("verdict: PASS")).await,
        VerifierVerdict::Passed
    );
    assert!(matches!(
        v.verify(&input("looks plausible")).await,
        VerifierVerdict::Failed { .. }
    ));
}
