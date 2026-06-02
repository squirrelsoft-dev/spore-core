//! Fixture-replay integration test for Agent (issue #2).
//!
//! Loads `fixtures/model_responses/agent/turn_classification.jsonl` and drives
//! a `ModelAgent` backed by `ReplayModelInterface`, asserting the agent
//! classifies each recorded exchange consistently. Must produce the same
//! verdicts in all four language implementations.

use std::sync::Arc;

use spore_core::{
    Agent, AgentContext as Context, AgentError, AgentId, ModelAgent, ProviderInfo,
    ReplayModelInterface, TurnResult,
};

fn fixture_path() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("../../..")
        .join("fixtures/model_responses/agent/turn_classification.jsonl")
}

#[tokio::test]
async fn agent_classifies_recorded_turns_consistently() {
    let jsonl = std::fs::read_to_string(fixture_path()).expect("fixture file readable");
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
    let agent = ModelAgent::new(AgentId::new("fixture-agent"), replay);

    // 1. Plain text response → FinalResponse("hello")
    match agent.turn(Context::default()).await {
        TurnResult::FinalResponse { content, usage, .. } => {
            assert_eq!(content, "hello");
            assert_eq!(usage.input_tokens, 5);
            assert_eq!(usage.output_tokens, 1);
        }
        other => panic!("turn 1 expected FinalResponse, got {other:?}"),
    }

    // 2. Single tool call → ToolCallRequested with one call.
    match agent.turn(Context::default()).await {
        TurnResult::ToolCallRequested { calls, usage, .. } => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "read_file");
            assert_eq!(calls[0].id, "toolu_a");
            assert_eq!(usage.input_tokens, 20);
        }
        other => panic!("turn 2 expected ToolCallRequested, got {other:?}"),
    }

    // 3. Parallel tool calls → ToolCallRequested with two calls.
    match agent.turn(Context::default()).await {
        TurnResult::ToolCallRequested { calls, .. } => {
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].id, "toolu_b1");
            assert_eq!(calls[1].id, "toolu_b2");
        }
        other => panic!("turn 3 expected parallel ToolCallRequested, got {other:?}"),
    }

    // 4. Empty content blocks with EndTurn → AgentError::EmptyResponse.
    match agent.turn(Context::default()).await {
        TurnResult::Error {
            error: AgentError::EmptyResponse,
            usage: Some(u),
        } => {
            assert_eq!(u.input_tokens, 3);
            assert_eq!(u.output_tokens, 0);
        }
        other => panic!("turn 4 expected EmptyResponse, got {other:?}"),
    }
}
