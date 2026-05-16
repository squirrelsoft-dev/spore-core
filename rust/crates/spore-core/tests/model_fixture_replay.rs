//! Fixture-replay integration test for ModelInterface (issue #1).
//!
//! Loads `fixtures/model_responses/model_interface/basic_text.jsonl` from the
//! repo root and asserts the recorded exchanges replay deterministically.
//! This test must pass byte-for-byte in all four language implementations.

use spore_core::{
    ContentBlock, ModelInterface, ModelParams, ModelRequest, ProviderInfo, ReplayModelInterface,
    StopReason,
};

fn fixture_path() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR = .../rust/crates/spore-core
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("../../..")
        .join("fixtures/model_responses/model_interface/basic_text.jsonl")
}

fn empty_request() -> ModelRequest {
    ModelRequest {
        messages: vec![],
        tools: vec![],
        params: ModelParams::default(),
        stream: false,
    }
}

#[tokio::test]
async fn basic_text_fixture_replays_in_order() {
    let jsonl = std::fs::read_to_string(fixture_path()).expect("fixture file readable");
    let replay = ReplayModelInterface::from_jsonl(
        &jsonl,
        ProviderInfo {
            name: "anthropic".into(),
            model_id: "fixture".into(),
            context_window: 200_000,
        },
    )
    .expect("fixture parses");

    assert_eq!(replay.remaining(), 3);

    let r1 = replay.call(empty_request()).await.unwrap();
    assert_eq!(r1.stop_reason, StopReason::EndTurn);
    assert_eq!(r1.usage.input_tokens, 8);
    assert_eq!(r1.usage.output_tokens, 11);
    assert_eq!(
        r1.content,
        vec![ContentBlock::Text {
            text: "Hello! How can I help you today?".into()
        }]
    );

    let r2 = replay.call(empty_request()).await.unwrap();
    assert_eq!(r2.usage.input_tokens, 10);
    assert_eq!(r2.usage.output_tokens, 1);

    let r3 = replay.call(empty_request()).await.unwrap();
    assert_eq!(r3.stop_reason, StopReason::ToolUse);
    match &r3.content[0] {
        ContentBlock::ToolUse(call) => {
            assert_eq!(call.name, "echo");
            assert_eq!(call.input["text"], "hi");
        }
        other => panic!("expected ToolUse, got {other:?}"),
    }
}
