//! Loop-replay integration test for the Tool Escalation Protocol (issue #80).
//!
//! Loads `fixtures/model_responses/harness/escalation_loop.jsonl` — a one-turn
//! model trace in which the agent requests a single tool call — and drives a
//! `StandardHarness` with `LoopStrategy::ReAct`. The scripted tool registry
//! returns `ToolOutput::Escalate { Abort }` for that call. The replay asserts:
//!   1. The harness returns `RunResult::Escalate` (NOT Success/Failure).
//!   2. The escalation is NOT appended to message history (no tool-result turn).
//!   3. The carried signal is the expected `Abort`.
//!
//! Must produce the same outcome in all four language implementations — never
//! edit the fixture to make a failing implementation pass (see
//! `fixtures/README.md`).

use std::sync::Arc;

use spore_core::harness::testing::{
    AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager, ScriptedToolRegistry,
};
use spore_core::{
    Agent, AgentId, Harness, HarnessConfig, HarnessRunOptions, HarnessSignal, LoopStrategy,
    ModelAgent, ProviderInfo, ReplayModelInterface, Role, RunResult, SessionId, StandardHarness,
    Task, ToolOutput,
};

fn fixture_path() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("../../..")
        .join("fixtures/model_responses/harness/escalation_loop.jsonl")
}

#[tokio::test]
async fn escalation_loop_returns_escalate_and_skips_history_append() {
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
    tool_registry.push(ToolOutput::Escalate {
        signal: HarnessSignal::Abort {
            reason: "blocked on missing credentials".into(),
        },
    });

    let config = HarnessConfig {
        agent: agent as Arc<dyn Agent>,
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
        planner_agent: None,
        storage: Arc::new(spore_core::StorageProvider::no_op()),
        chunk_provider: Arc::new(spore_core::prompt_assembly::InMemoryChunkProvider::empty()),
    };
    let harness = StandardHarness::new(config);

    let task = Task::new(
        "investigate then decide whether to abort",
        SessionId::new("escalation-loop-session"),
        LoopStrategy::ReAct { max_iterations: 5 },
    );

    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Escalate {
            signal,
            state,
            session_id,
            turns,
            ..
        } => {
            assert_eq!(
                signal,
                HarnessSignal::Abort {
                    reason: "blocked on missing credentials".into(),
                }
            );
            assert_eq!(session_id, SessionId::new("escalation-loop-session"));
            assert_eq!(turns, 1, "one turn consumed before the escalating dispatch");
            // The escalation is a control signal — never appended as a turn.
            let tool_results = state
                .session_state
                .messages
                .iter()
                .filter(|m| m.role == Role::Tool)
                .count();
            assert_eq!(tool_results, 0, "escalation must not append a tool result");
            // No remaining calls in this single-call batch.
            assert!(state.pending_tool_calls.is_empty());
            // The dispatch ran exactly once.
            assert_eq!(
                tool_registry
                    .call_count
                    .load(std::sync::atomic::Ordering::SeqCst),
                1
            );
        }
        other => panic!("expected RunResult::Escalate, got {other:?}"),
    }
}
