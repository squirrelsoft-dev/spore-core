//! Fixture-replay integration tests for the PlanExecute DAG executor (issue #126).
//!
//! Each fixture under `fixtures/model_responses/harness/plan_execute_dag_*.jsonl`
//! is a POSITIONAL model-response trace: one plan turn followed by one execute
//! completion per scheduled task, in ready-set scheduling order (lowest-id ready
//! task first). The blocker DAG itself is authored via the persisted `task_list`
//! tool store (the #126 decision-C authoring path) — the test seeds it before the
//! run, exactly as a real `task_list`-tool run would.
//!
//! Mirrors `tests/plan_execute_loop_fixture_replay.rs`. Must produce the same
//! outcome in all four language implementations — never edit the fixture to make
//! a failing implementation pass.

use std::sync::Arc;

use spore_core::harness::testing::{
    AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager, ScriptedToolRegistry,
};
use spore_core::{
    Agent, AgentId, HaltReason, Harness, HarnessConfig, HarnessRunOptions, LoopStrategy,
    ModelAgent, PlanExecuteConfig, ProviderInfo, ReactConfig, ReplayModelInterface, RunResult,
    SessionId, StandardHarness, StorageProvider, Task, TaskList, TaskStatus, TASK_LIST_EXTRAS_KEY,
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

/// Build a harness whose worker agent replays `fixture`, backed by a real
/// in-memory RunStore (so the seeded `task_list` and post-run statuses are
/// observable). Returns the harness and the shared storage provider.
fn harness_for(fixture: &str) -> (StandardHarness, Arc<StorageProvider>) {
    let jsonl = std::fs::read_to_string(fixture_path(fixture))
        .unwrap_or_else(|e| panic!("read {fixture}: {e}"));
    let replay = Arc::new(ReplayModelInterface::from_jsonl(&jsonl, provider()).expect("fixture"));
    let agent: Arc<ModelAgent<ReplayModelInterface>> =
        Arc::new(ModelAgent::new(AgentId::new("dag"), replay));
    let storage = Arc::new(StorageProvider::single(Arc::new(
        spore_core::storage::InMemoryStorageProvider::new(),
    )));
    let cfg = HarnessConfig {
        tool_registry: Arc::new(ScriptedToolRegistry::new()),
        sandbox: Arc::new(AllowAllSandbox),
        sandbox_violation_policy: spore_core::harness::SandboxViolationPolicy::default(),
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
        enforce_output_schemas: false,
        output_schema_max_retries: 2,
        hooks: None,
        storage: storage.clone(),
        // #142: durable artifacts (task_list) are keyed by the project namespace.
        // Pin a known project id so `seed`/`stored_list` key the SAME namespace
        // the harness reads/writes under (see `project_ns`).
        project_id: spore_core::ProjectId::from_canonical_path(DAG_PROJECT_PATH),
        chunk_provider: Arc::new(spore_core::prompt_assembly::InMemoryChunkProvider::empty()),
        max_resets: 3,
        vcs_provider: None,
        catalogue_registry: None,
        system_prompt: None,
        model_params: spore_core::ModelParams::default(),
        auto_persist_sessions: false,
        prompt_tool_call_flag: None,
        consult_handlers: std::collections::HashMap::new(),
        registry: spore_core::ExecutionRegistry::builder()
            .agent("", agent as Arc<dyn Agent>)
            .toolset("", Arc::new(spore_core::EmptyToolRegistry))
            .schema("", serde_json::json!({}))
            .build(),
        escalation_mode: spore_core::EscalationMode::SurfaceToHuman,
    };
    (StandardHarness::new(cfg), storage)
}

fn dag_task() -> Task {
    Task::new(
        "build the DAG",
        SessionId::new("dag-fixture"),
        LoopStrategy::PlanExecute(PlanExecuteConfig {
            plan: Box::new(LoopStrategy::ReAct(ReactConfig {
                budget: spore_core::BudgetPolicy::PerLoop { value: u32::MAX },
                behavior: spore_core::BudgetExhaustedBehavior::Escalate,
                agent: spore_core::AgentRef(String::new()),
                toolset: spore_core::ToolsetRef(String::new()),
                output: Some(spore_core::SchemaRef(String::new())),
            })),
            execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
            plan_model: None,
            behavior: spore_core::BudgetExhaustedBehavior::Escalate,
        }),
    )
}

/// #142: the fixed canonical path the test harness derives its durable
/// `project_id` from (pinned via the `project_id` field in `harness_for`).
const DAG_PROJECT_PATH: &str = "/dag-workspace";

/// The durable `RunStore` namespace the harness keys task_list under (#142) — the
/// project id projected onto the session-id axis. `seed`/`stored_list` key this,
/// NOT the ephemeral run session, so what the test seeds is what the harness
/// reads and vice versa.
fn project_ns() -> SessionId {
    spore_core::ProjectId::from_canonical_path(DAG_PROJECT_PATH).namespace()
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

// AC1: a diamond DAG executes in dependency order; the run succeeds and every
// task completes. Scheduling order 1, 2, 3, 4 matches the fixture's trace order.
#[tokio::test]
async fn dag_order_diamond_all_complete() {
    let (h, storage) = harness_for("plan_execute_dag_order.jsonl");
    let session = SessionId::new("dag-fixture");
    let mut tl = TaskList::default();
    tl.add("one".into(), vec![]).unwrap(); // 1
    tl.add("two".into(), vec![1]).unwrap(); // 2 -> 1
    tl.add("three".into(), vec![1]).unwrap(); // 3 -> 1
    tl.add("four".into(), vec![2, 3]).unwrap(); // 4 -> 2,3
    seed(&storage, &session, &tl).await;

    match h.run(HarnessRunOptions::new(dag_task())).await {
        RunResult::Success { output, .. } => assert_eq!(output, "did 4"),
        other => panic!("expected Success, got {other:?}"),
    }
    let after = stored_list(&storage, &session).await;
    assert!(after
        .tasks
        .iter()
        .all(|t| t.status == TaskStatus::Completed));
}

// AC1 branch isolation: the child sees its transitive blocker's output; the run
// succeeds and all three tasks complete.
#[tokio::test]
async fn dag_branch_isolation_completes() {
    let (h, storage) = harness_for("plan_execute_dag_branch_isolation.jsonl");
    let session = SessionId::new("dag-fixture");
    let mut tl = TaskList::default();
    tl.add("root".into(), vec![]).unwrap(); // 1
    tl.add("child of root".into(), vec![1]).unwrap(); // 2 -> 1
    tl.add("independent".into(), vec![]).unwrap(); // 3 indep
    seed(&storage, &session, &tl).await;

    match h.run(HarnessRunOptions::new(dag_task())).await {
        RunResult::Success { output, .. } => assert_eq!(output, "INDEP_OUTPUT_CCC"),
        other => panic!("expected Success, got {other:?}"),
    }
    let after = stored_list(&storage, &session).await;
    assert!(after
        .tasks
        .iter()
        .all(|t| t.status == TaskStatus::Completed));
}

// AC3: a terminal task failure blocks only its transitive dependents; unrelated
// tasks complete; the run drains to TasksBlockedByFailure with the partition.
#[tokio::test]
async fn dag_failure_cascade_partition() {
    let (h, storage) = harness_for("plan_execute_dag_failure_cascade.jsonl");
    let session = SessionId::new("dag-fixture");
    let mut tl = TaskList::default();
    tl.add("root".into(), vec![]).unwrap(); // 1
    tl.add("mid".into(), vec![1]).unwrap(); // 2 -> 1 (fails)
    tl.add("leaf".into(), vec![2]).unwrap(); // 3 -> 2 (cascade-blocked)
    tl.add("indep".into(), vec![]).unwrap(); // 4 independent
    seed(&storage, &session, &tl).await;

    match h.run(HarnessRunOptions::new(dag_task())).await {
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
            assert_eq!(failed_task, 2);
            assert_eq!(completed, vec![1, 4]);
            assert_eq!(blocked, vec![2, 3]);
        }
        other => panic!("expected TasksBlockedByFailure, got {other:?}"),
    }
}

// AC4: a budget-`Fail` task cascades IDENTICALLY to an error-failed one — same
// cascade arm, same partition shape (structural twin of the error cascade).
#[tokio::test]
async fn dag_budget_fail_cascade_partition() {
    let (h, storage) = harness_for("plan_execute_dag_budget_fail_cascade.jsonl");
    let session = SessionId::new("dag-fixture");
    let mut tl = TaskList::default();
    tl.add("root".into(), vec![]).unwrap(); // 1
    tl.add("mid".into(), vec![1]).unwrap(); // 2 -> 1 (fails)
    tl.add("leaf".into(), vec![2]).unwrap(); // 3 -> 2 (cascade-blocked)
    tl.add("indep".into(), vec![]).unwrap(); // 4 independent
    seed(&storage, &session, &tl).await;

    match h.run(HarnessRunOptions::new(dag_task())).await {
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
            assert_eq!(failed_task, 2);
            assert_eq!(completed, vec![1, 4]);
            assert_eq!(blocked, vec![2, 3]);
        }
        other => panic!("expected TasksBlockedByFailure, got {other:?}"),
    }
}

// AC5: a cyclic persisted graph is rejected at execute entry; no execute step
// runs (only the single plan turn is consumed from the fixture).
#[tokio::test]
async fn dag_cycle_rejected_at_entry() {
    let (h, storage) = harness_for("plan_execute_dag_cycle_rejection.jsonl");
    let session = SessionId::new("dag-fixture");
    let mut tl = TaskList::default();
    tl.add("a".into(), vec![]).unwrap(); // 1
    tl.add("b".into(), vec![1]).unwrap(); // 2 -> 1
    tl.tasks[0].blockers = vec![2]; // 1 -> 2 closes the cycle
    seed(&storage, &session, &tl).await;

    match h.run(HarnessRunOptions::new(dag_task())).await {
        RunResult::Failure {
            reason: HaltReason::TaskGraphCycle { .. },
            ..
        } => {}
        other => panic!("expected TaskGraphCycle, got {other:?}"),
    }
}
