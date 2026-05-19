//! Fixture-replay integration test for `OllamaModelInterface` (issue #41).
//!
//! Loads `fixtures/model_responses/model_interface/ollama_basic_text.jsonl`
//! and asserts that the recorded exchange replays through
//! `ReplayModelInterface` with the expected Ollama-shaped outcome:
//! cache fields `None`, `stop_reason` = `EndTurn`, provider = "ollama".
//! This test must pass byte-for-byte in all four language implementations.

use spore_core::{
    ContentBlock, ModelInterface, ModelParams, ModelRequest, ProviderInfo, ReplayModelInterface,
    StopReason,
};

fn fixture_path() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("../../..")
        .join("fixtures/model_responses/model_interface/ollama_basic_text.jsonl")
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
async fn ollama_basic_text_fixture_replays() {
    let jsonl = std::fs::read_to_string(fixture_path()).expect("ollama fixture readable");
    let replay = ReplayModelInterface::from_jsonl(
        &jsonl,
        ProviderInfo {
            name: "ollama".into(),
            model_id: "llama3.2".into(),
            context_window: 128_000,
        },
    )
    .expect("fixture parses");

    assert_eq!(replay.remaining(), 1);
    let r = replay.call(empty_request()).await.unwrap();
    assert_eq!(r.stop_reason, StopReason::EndTurn);
    assert_eq!(r.usage.input_tokens, 8);
    assert_eq!(r.usage.output_tokens, 11);
    assert_eq!(r.usage.cache_read_tokens, None);
    assert_eq!(r.usage.cache_write_tokens, None);
    assert_eq!(
        r.content,
        vec![ContentBlock::Text {
            text: "Hello! How can I help you today?".into()
        }]
    );
}
