//! Loop-replay integration test for the consecutive-recoverable-tool-error
//! breaker (issue #137).
//!
//! Loads `fixtures/model_responses/harness/tool_error_loop.jsonl` — a recorded
//! trace in which the model repeatedly emits the SAME malformed `add_task` tool
//! call (the gemma `task_list`/`add_task`-without-`description` scenario) — and
//! drives a `StandardHarness` with `LoopStrategy::ReAct`. The scripted tool
//! registry returns an identical `ToolOutput::Error { recoverable: true }` for
//! every dispatch. With `error_loop_threshold = 3` (N) and a generous turn
//! budget (50), the ONLY thing that can stop the run is the breaker, which
//! hard-stops at 2N (6 identical errors) and resolves the leaf's `Fail`
//! behavior into `RunResult::Failure { reason: ToolErrorLoop }` WITHOUT burning
//! the remaining budget.
//!
//! Must produce the same outcome in all four language implementations — never
//! edit the fixture to make a failing implementation pass (see
//! `fixtures/README.md`).

use std::sync::Arc;

use spore_core::harness::testing::{
    AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager, ScriptedToolRegistry,
};
use spore_core::{
    Agent, AgentId, AgentRef, BudgetExhaustedBehavior, BudgetPolicy, HaltReason, Harness,
    HarnessConfig, HarnessRunOptions, LoopStrategy, ModelAgent, ProviderInfo, ReactConfig,
    ReplayModelInterface, RunResult, SessionId, StandardHarness, Task, ToolsetRef,
};

fn fixture_path() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("../../..")
        .join("fixtures/model_responses/harness/tool_error_loop.jsonl")
}

#[tokio::test]
async fn tool_error_loop_breaker_hard_stops_at_two_n() {
    let jsonl = std::fs::read_to_string(fixture_path()).expect("fixture readable");
    let replay = Arc::new(
        ReplayModelInterface::from_jsonl(
            &jsonl,
            ProviderInfo {
                name: "ollama".into(),
                model_id: "fixture".into(),
                context_window: 200_000,
            },
        )
        .expect("fixture parses"),
    );
    let agent: Arc<ModelAgent<ReplayModelInterface>> =
        Arc::new(ModelAgent::new(AgentId::new("fixture-agent"), replay));

    // Every dispatch of the malformed `add_task` call returns the same
    // recoverable error, regardless of args.
    let tool_registry = Arc::new(ScriptedToolRegistry::new());
    tool_registry.always_recoverable_error("missing required parameter `description`");

    let config = HarnessConfig {
        tool_registry: tool_registry.clone(),
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
        // N == 3 → inject at 3 identical errors, hard-stop at 6.
        error_loop_threshold: 3,
        hooks: None,
        storage: Arc::new(spore_core::StorageProvider::no_op()),
        project_id: spore_core::ProjectId::from_canonical_path("/test-workspace"),
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
            .toolset("", tool_registry.clone())
            .build(),
        // Autonomous so the leaf's `Fail` behavior produces a terminal Failure
        // (not a HITL pause) at the 2N hard stop.
        escalation_mode: spore_core::EscalationMode::Autonomous,
    };
    let harness = StandardHarness::new(config);

    // A bare ReAct leaf with a generous budget (50) and `Fail` behavior — so the
    // ONLY thing that can stop the run early is the error-loop breaker.
    let task = Task::new(
        "add a task to the task list",
        SessionId::new("tool-error-loop-session"),
        LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: 50 },
            behavior: BudgetExhaustedBehavior::Fail,
            agent: AgentRef(String::new()),
            toolset: ToolsetRef(String::new()),
            output: None,
        }),
    );

    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Failure {
            reason:
                HaltReason::ToolErrorLoop {
                    tool,
                    consecutive_errors,
                },
            session_id,
            turns,
            ..
        } => {
            assert_eq!(tool, "add_task");
            assert_eq!(consecutive_errors, 6, "hard stop at 2N == 6");
            assert_eq!(session_id, SessionId::new("tool-error-loop-session"));
            // The breaker stopped EARLY: exactly 2N turns, far below the budget.
            assert_eq!(turns, 6, "exactly 2N turns consumed before the hard stop");
            assert!(turns < 50, "budget NOT fully burned, got {turns}");
            // The breaker stops AT the 2N dispatch — it does not append/continue
            // past it — so the registry saw exactly 2N == 6 calls (the 7th
            // fixture line is unused headroom: the breaker, not fixture
            // exhaustion, ended the run).
            assert_eq!(
                tool_registry
                    .call_count
                    .load(std::sync::atomic::Ordering::SeqCst),
                6,
                "tool dispatched exactly 2N times, then the breaker stopped"
            );
        }
        other => panic!("expected RunResult::Failure {{ ToolErrorLoop }}, got {other:?}"),
    }
}
