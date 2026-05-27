//! Live observability smoke test (issue #12 / #33 wiring).
//!
//! Runs one real `harness.run()` through a [`HarnessBuilder`]-assembled harness
//! with a durable-outbox [`ObservabilityProvider`], emitting turn and tool-call
//! spans and a terminal session summary. Writes JSONL under `<root>/sessions/`
//! and, when `SPORE_OTLP_ENDPOINT` is set, forwards the same spans to Tempo.
//!
//! Usage (against the local stack in `observability/`):
//!
//! ```sh
//! SPORE_OTLP_ENDPOINT=http://localhost:4317 \
//!   cargo run -p spore-core --example observability_smoke
//! ```
//!
//! It prints the session id, the on-disk trace path, and the `trace_id` — query
//! Tempo with `curl -s http://localhost:3200/api/traces/<trace_id> | jq` to
//! confirm the grouped trace, and check the "Spore" folder in Grafana.

use std::sync::Arc;

use spore_core::agent::mock::MockAgent;
use spore_core::harness::testing::{
    AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager, ScriptedToolRegistry,
};
use spore_core::{
    AgentId, Harness, HarnessBuilder, HarnessRunOptions, LoopStrategy, RunResult, SessionId, Task,
    Timestamp, TokenUsage, ToolCall, ToolOutput, TurnResult,
};

#[tokio::main]
async fn main() {
    // Unique per-run session id so repeated runs don't collide in Tempo.
    let session_id = format!("live-smoke-{}", Timestamp::now().as_str().replace([':', '.'], "-"));
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.spore");

    let endpoint = std::env::var("SPORE_OTLP_ENDPOINT").unwrap_or_default();
    if endpoint.trim().is_empty() {
        eprintln!(
            "note: SPORE_OTLP_ENDPOINT is unset — writing JSONL only (no Tempo forwarding).\n\
             set SPORE_OTLP_ENDPOINT=http://localhost:4317 to forward to the local stack."
        );
    }

    let usage = TokenUsage {
        input_tokens: 1_820,
        output_tokens: 140,
        cache_read_tokens: Some(1_600),
        cache_write_tokens: Some(0),
    };
    let agent = Arc::new(MockAgent::new(AgentId::new("smoke")));
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![ToolCall {
            id: "call-1".into(),
            name: "read_file".into(),
            input: serde_json::json!({ "path": "src/auth.rs" }),
        }],
        usage: usage.clone(),
    });
    agent.push(TurnResult::FinalResponse {
        content: "fixed the failing test".into(),
        usage,
    });

    let tools = Arc::new(ScriptedToolRegistry::new());
    tools.push(ToolOutput::Success {
        content: "file contents elided".into(),
        truncated: false,
    });

    let harness = HarnessBuilder::new(
        agent,
        tools,
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
    .with_observability_outbox(&root)
    .build();

    let task = Task::new(
        "fix the failing tests in src/auth.rs",
        SessionId::new(session_id.clone()),
        LoopStrategy::ReAct { max_iterations: 5 },
    );

    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Success { turns, output, .. } => {
            println!("run succeeded: {turns} turns, output = {output:?}");
        }
        other => {
            eprintln!("unexpected run result: {other:?}");
            std::process::exit(1);
        }
    }

    let trace_path = root.join(format!("sessions/{session_id}/trace.jsonl"));
    let body = std::fs::read_to_string(&trace_path).expect("trace.jsonl written");
    let first: serde_json::Value =
        serde_json::from_str(body.lines().next().expect("at least one span")).unwrap();
    let trace_id = first["trace_id"].as_str().unwrap_or("?");
    let kinds: Vec<String> = body
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .map(|v| v["kind"].as_str().unwrap_or("?").to_string())
        .collect();

    println!("session_id : {session_id}");
    println!("trace_path : {}", trace_path.display());
    println!("trace_id   : {trace_id}");
    println!("span kinds : {kinds:?}");
    if !endpoint.trim().is_empty() {
        println!(
            "verify     : curl -s http://localhost:3200/api/traces/{trace_id} | jq '.batches | length'"
        );
    }
}
