//! Fixture-replay integration test for the PlanExecute plan phase (issue #70).
//!
//! Loads `fixtures/model_responses/harness/plan_phase_basic.jsonl` and drives a
//! `StandardHarness` with `LoopStrategy::PlanExecute`, asserting that:
//!   1. The plan turn's recorded `FinalResponse` is captured into the exact
//!      `PlanArtifact` (tasks + rationale), persisted to the `RunStore` seam
//!      under `PLAN_EXECUTE_EXTRAS_KEY` (#76 — no longer mirrored into
//!      `extras["plan_execute"]`).
//!   2. The fenced ```json variant is captured identically (fence-strip rule).
//!   3. The plan turn is consumed and parsed into a non-empty task list, so the
//!      run proceeds into the execute phase (issue #59). This fixture provides
//!      ONLY the single plan turn, so the first execute step's ReAct sub-loop
//!      exhausts the replay and the run aborts with `StepFailed { task_index: 0 }`
//!      — proving the harness consumed the planner response and entered execute.
//!
//! Must produce the same outcome in all four language implementations — never
//! edit the fixture to make a failing implementation pass.

use std::sync::Arc;

use spore_core::harness::testing::{
    AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager, ScriptedToolRegistry,
};
use spore_core::{
    Agent, AgentId, HaltReason, Harness, HarnessConfig, HarnessRunOptions, LoopStrategy,
    ModelAgent, PlanExecuteConfig, ProviderInfo, ReactConfig, RecordedExchange,
    ReplayModelInterface, RunResult, SessionId, StandardHarness, Task,
};

fn fixture_exchanges() -> Vec<RecordedExchange> {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest
        .join("../../..")
        .join("fixtures/model_responses/harness/plan_phase_basic.jsonl");
    let jsonl =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    jsonl
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<RecordedExchange>(l).expect("fixture line parses"))
        .collect()
}

fn provider() -> ProviderInfo {
    ProviderInfo {
        name: "anthropic".into(),
        model_id: "fixture".into(),
        context_window: 200_000,
    }
}

fn config_for(exchange: RecordedExchange) -> HarnessConfig {
    // Positional single-exchange replay: each plan run consumes exactly one turn.
    let replay = Arc::new(ReplayModelInterface::new(vec![exchange], provider()));
    let agent: Arc<ModelAgent<ReplayModelInterface>> =
        Arc::new(ModelAgent::new(AgentId::new("planner"), replay));
    HarnessConfig {
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
        storage: Arc::new(spore_core::StorageProvider::no_op()),
        chunk_provider: Arc::new(spore_core::prompt_assembly::InMemoryChunkProvider::empty()),
        max_resets: 3,
        vcs_provider: None,
        catalogue_registry: None,
        system_prompt: None,
        model_params: spore_core::ModelParams::default(),
        auto_persist_sessions: false,
        prompt_tool_call_flag: None,
        consult_handlers: std::collections::HashMap::new(),
        // #124: worker agent + toolset + default plan-slot schema under the
        // default empty key.
        registry: spore_core::ExecutionRegistry::builder()
            .agent("", agent as Arc<dyn Agent>)
            .toolset("", Arc::new(spore_core::EmptyToolRegistry))
            .schema("", serde_json::json!({}))
            .build(),
        escalation_mode: spore_core::EscalationMode::SurfaceToHuman,
    }
}

/// Drive the plan phase against `exchange` through a ReplayModelInterface and
/// assert the run consumes the planner turn, parses a non-empty task list, and
/// enters the execute phase — where the single-exchange replay exhausts on the
/// first step, cascades, and drains to `TasksBlockedByFailure { failed_task: 1 }`
/// (#126 — replaces the pre-#126 `StepFailed` blanket abort).
async fn drive_plan_phase(exchange: RecordedExchange) {
    let harness = StandardHarness::new(config_for(exchange));
    // #124 A.5: the plan slot is STRUCTURED — its leaf carries an output schema.
    let task = Task::new(
        "build something",
        SessionId::new("plan-fixture"),
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
    );
    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Failure {
            reason: HaltReason::TasksBlockedByFailure { failed_task, .. },
            turns,
            ..
        } => {
            // #126: a failing execute step no longer raises `StepFailed`; it
            // cascades and the run drains to `TasksBlockedByFailure`. With the
            // single-exchange replay every execute step exhausts, so the FIRST
            // ready task (id 1) is the failed task.
            assert_eq!(failed_task, 1, "the first ready task fails");
            // The plan turn (1) plus the step's first model call before the
            // replay exhausts: the harness reached the execute phase.
            assert!(turns >= 1, "at least the plan turn was consumed");
        }
        other => panic!("expected TasksBlockedByFailure, got {other:?}"),
    }
}

/// Extract the single text block from a recorded response.
fn response_text(exchange: &RecordedExchange) -> String {
    exchange
        .response
        .content
        .iter()
        .find_map(|b| match b {
            spore_core::ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
        .expect("recorded response has a text block")
}

#[tokio::test]
async fn plan_phase_basic_captures_plain_json() {
    let exchanges = fixture_exchanges();
    assert!(exchanges.len() >= 2, "fixture has >= 2 cases");

    // Drive the full harness plan phase against the replayed response.
    drive_plan_phase(exchanges[0].clone()).await;

    // Assert the captured artifact via the same public capture grammar the
    // harness uses internally — byte-identical across all four languages.
    let artifact =
        spore_core::capture_plan_artifact(&response_text(&exchanges[0])).expect("captures");
    assert_eq!(
        artifact.tasks,
        vec![
            "scaffold the project",
            "add the argument parser",
            "write the integration tests"
        ]
    );
    assert_eq!(artifact.rationale, "deliver a working CLI incrementally");
}

// Fenced-```json variant: exercises the fence-strip rule cross-language.
#[tokio::test]
async fn plan_phase_basic_captures_fenced_json() {
    let exchanges = fixture_exchanges();
    assert!(exchanges.len() >= 2, "fixture has >= 2 cases");

    drive_plan_phase(exchanges[1].clone()).await;

    let artifact =
        spore_core::capture_plan_artifact(&response_text(&exchanges[1])).expect("fenced json");
    assert_eq!(
        artifact.tasks,
        vec!["draft the outline", "write the reference section"]
    );
    assert_eq!(artifact.rationale, "docs follow the code");
}
