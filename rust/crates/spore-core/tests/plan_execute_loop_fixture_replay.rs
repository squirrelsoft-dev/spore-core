//! Fixture-replay integration test for the PlanExecute loop (issue #59).
//!
//! Loads `fixtures/model_responses/harness/plan_execute_loop.jsonl` — a full
//! two-phase trace: one plan turn producing K=2 tasks, followed by K execute
//! completions — and drives a `StandardHarness` with
//! `LoopStrategy::PlanExecute`, asserting that:
//!   1. The plan turn is captured and parsed into a 2-task list.
//!   2. The execute phase drains BOTH tasks sequentially (one turn each).
//!   3. The run SUCCEEDS, `output` is the LAST step's FinalResponse (Q2), and
//!      `turns == 3` (one plan turn + one per task — shared budget, Q1).
//!
//! Must produce the same outcome in all four language implementations — never
//! edit the fixture to make a failing implementation pass.

use std::sync::Arc;

use spore_core::harness::testing::{
    AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager, ScriptedToolRegistry,
};
use spore_core::{
    Agent, AgentId, Harness, HarnessConfig, HarnessRunOptions, LoopStrategy, ModelAgent,
    PlanExecuteConfig, ProviderInfo, ReactConfig, ReplayModelInterface, RunResult, SessionId,
    StandardHarness, Task,
};

fn provider() -> ProviderInfo {
    ProviderInfo {
        name: "anthropic".into(),
        model_id: "fixture".into(),
        context_window: 200_000,
    }
}

fn config() -> HarnessConfig {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest
        .join("../../..")
        .join("fixtures/model_responses/harness/plan_execute_loop.jsonl");
    let jsonl =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    // Positional replay: the plan turn (line 1) then the two execute steps.
    let replay = Arc::new(ReplayModelInterface::from_jsonl(&jsonl, provider()).expect("fixture"));
    let agent: Arc<ModelAgent<ReplayModelInterface>> =
        Arc::new(ModelAgent::new(AgentId::new("plan-execute"), replay));
    HarnessConfig {
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
        storage: Arc::new(spore_core::StorageProvider::no_op()),
        project_id: spore_core::ProjectId::from_canonical_path("/test-workspace"),
        chunk_provider: Arc::new(spore_core::prompt_assembly::InMemoryChunkProvider::empty()),
        max_resets: 3,
        vcs_provider: None,
        catalogue_registry: None,
        system_prompt: None,
        guides: Vec::new(),
        model_params: spore_core::ModelParams::default(),
        auto_persist_sessions: false,
        prompt_tool_call_flag: None,
        consult_handlers: std::collections::HashMap::new(),
        // #124: the worker agent + toolset + a default plan-slot schema resolve
        // from the registry under the default empty key.
        registry: spore_core::ExecutionRegistry::builder()
            .agent("", agent as Arc<dyn Agent>)
            .toolset("", Arc::new(spore_core::EmptyToolRegistry))
            .schema("", serde_json::json!({}))
            .build(),
        escalation_mode: spore_core::EscalationMode::SurfaceToHuman,
    }
}

#[tokio::test]
async fn plan_execute_loop_full_trace_succeeds() {
    let harness = StandardHarness::new(config());
    // #124 A.5: the plan slot is STRUCTURED — its leaf carries an output schema.
    let task = Task::new(
        "build a CLI",
        SessionId::new("plan-execute-fixture"),
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
        RunResult::Success { output, turns, .. } => {
            // Q2: the success handle is the LAST completed step's final text.
            assert_eq!(output, "wrote the integration tests");
            // Q1: one plan turn + one turn per task (2) under the shared budget.
            assert_eq!(turns, 3, "one plan turn + one per task");
        }
        other => panic!("expected Success, got {other:?}"),
    }
}
