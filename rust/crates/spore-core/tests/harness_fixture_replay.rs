//! Fixture-replay integration test for Harness (issue #3).
//!
//! Loads `fixtures/model_responses/harness/react_loop.jsonl` and drives a
//! `StandardHarness` with `LoopStrategy::ReAct`, asserting the loop:
//!   1. Dispatches the recorded tool call
//!   2. Loops to the next agent turn
//!   3. Returns `RunResult::Success` with the recorded final response
//!
//! Must produce the same outcome in all four language implementations.

use std::sync::Arc;

use spore_core::harness::testing::{
    AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager, ScriptedToolRegistry,
};
use spore_core::{
    Agent, AgentId, Harness, HarnessConfig, HarnessRunOptions, LoopStrategy, ModelAgent,
    ProviderInfo, ReactConfig, ReplayModelInterface, RunResult, SessionId, StandardHarness, Task,
    ToolOutput,
};

fn fixture_path() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("../../..")
        .join("fixtures/model_responses/harness/react_loop.jsonl")
}

#[tokio::test]
async fn react_loop_dispatches_tool_then_completes() {
    let jsonl = std::fs::read_to_string(fixture_path()).expect("fixture readable");
    let replay = Arc::new(
        ReplayModelInterface::from_jsonl(
            &jsonl,
            ProviderInfo {
                name: "anthropic".into(),
                model_id: "fixture".into(),
                context_window: 200_000,
            },
        )
        .expect("fixture parses"),
    );
    let agent: Arc<ModelAgent<ReplayModelInterface>> =
        Arc::new(ModelAgent::new(AgentId::new("fixture-agent"), replay));

    let tool_registry = Arc::new(ScriptedToolRegistry::new());
    tool_registry.push(ToolOutput::Success {
        content: "127.0.0.1 localhost".into(),
        truncated: false,
    });

    let config = HarnessConfig {
        tool_registry: tool_registry.clone(),
        sandbox: Arc::new(AllowAllSandbox),
        context_manager: Arc::new(NoopContextManager),
        termination_policy: Arc::new(AlwaysContinuePolicy),
        middleware: None,
        observability: None,
        compaction_verifier: Arc::new(spore_core::KeyTermVerifier),
        max_compaction_attempts: 2,
        pricing: spore_core::PricingTable::DEFAULT,
        content_capture: spore_core::ContentCaptureConfig::default(),
        tool_call_repair: None,
        max_repair_attempts: 1,
        max_stop_blocks: 8,
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
        // #124: the worker agent + toolset resolve from the registry under the
        // default empty key (the task's bare ReAct leaves carry empty handles).
        registry: spore_core::ExecutionRegistry::builder()
            .agent("", agent as Arc<dyn Agent>)
            .toolset("", tool_registry.clone())
            .build(),
        escalation_mode: spore_core::EscalationMode::SurfaceToHuman,
    };
    let harness = StandardHarness::new(config);

    let task = Task::new(
        "read /etc/hosts then summarize",
        SessionId::new("fixture-session"),
        LoopStrategy::ReAct(ReactConfig::per_loop(5)),
    );

    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Success {
            output,
            turns,
            usage,
            ..
        } => {
            assert_eq!(output, "127.0.0.1 localhost");
            assert_eq!(turns, 2, "ReAct loop should run two turns");
            assert_eq!(usage.input_tokens, 30, "12 + 18 input tokens");
            assert_eq!(usage.output_tokens, 14, "8 + 6 output tokens");
            assert_eq!(
                tool_registry
                    .call_count
                    .load(std::sync::atomic::Ordering::SeqCst),
                1,
                "tool registry dispatched exactly once"
            );
        }
        other => panic!("expected Success, got {other:?}"),
    }
}
