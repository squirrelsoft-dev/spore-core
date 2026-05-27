//! End-to-end CLI agent harness + scenario suite (issue #57).
//!
//! One shared example binary that drives the *complete* harness loop against a
//! real local model (Ollama) through a `HarnessBuilder`-assembled harness with
//! real tools (read/write/list/bash + a deliberately-failing tool), the
//! `StandardCompactionAdapter`, and the durable-outbox observability provider.
//! The scenario is selected by a CLI arg.
//!
//! ## Scenarios
//!
//! - `s1` — multi-step / multi-tool: read input.txt → uppercase → write
//!   output.txt → read back + confirm.
//! - `s2` — multi-turn: run twice with the same `SessionId`, carrying session
//!   state across turns.
//! - `s3` — live compaction: a seeded small window + long history fires the
//!   compaction adapter mid-run; the token-accounting fix lets it compact,
//!   continue, and compact again.
//! - `s4` — tool failure + recovery: call `flaky_op` (recoverable error), then
//!   write recovered.txt explaining the adaptation.
//!
//! ## Run recipe (live, against a local model + observability stack)
//!
//! ```sh
//! # 1. Start Ollama and pull a tool-capable model.
//! ollama serve &              # or run the Ollama app
//! ollama pull llama3.2        # default model; passes the #41 capability guard
//!
//! # 2. (optional) Start the local observability stack and forward traces.
//! #    See observability/ for the compose stack (Tempo + Loki + Grafana).
//! export SPORE_OTLP_ENDPOINT=http://localhost:4317
//!
//! # 3. Run a scenario. Prompt/model/endpoint/workspace come from args+env.
//! cargo run -p spore-core --example e2e_agent -- s1 --model llama3.2
//! cargo run -p spore-core --example e2e_agent -- s2
//! cargo run -p spore-core --example e2e_agent -- s3
//! cargo run -p spore-core --example e2e_agent -- s4
//!
//! # 4. Verify the grouped trace in Tempo (the run prints the trace_id):
//! curl -s http://localhost:3200/api/traces/<trace_id> | jq '.batches | length'
//! #    For S3, spot-check a Compaction span appears mid-trace.
//! ```
//!
//! Environment variables (all optional):
//! - `SPORE_OLLAMA_MODEL`     — default model id (overridden by `--model`).
//! - `SPORE_OLLAMA_BASE_URL`  — Ollama base url (default http://localhost:11434).
//! - `SPORE_OTLP_ENDPOINT`    — when set, forward spans to Tempo (issue #50).
//! - `SPORE_E2E_WORKSPACE`    — workspace root (default: a temp dir per run).
//!
//! ## Offline / hermetic mode
//!
//! `--mock` runs the same scenario builders against a scripted `MockAgent`,
//! requiring no Ollama or network. Mock support is gated behind the
//! `test-utils` feature:
//!
//! ```sh
//! cargo run -p spore-core --features test-utils --example e2e_agent -- s1 --mock
//! ```
//!
//! The hermetic CI assertions live in `tests/e2e_scenarios.rs`, which drive the
//! same `build_scenario` path with mock components.

use std::sync::Arc;

use spore_core::agent::{AgentId, ModelAgent};
use spore_core::cache_provider::OllamaCacheProvider;
use spore_core::context::CompactionConfig;
use spore_core::observability_outbox::{OutboxConfig, OutboxObservabilityProvider};
use spore_core::scenarios::{
    build_real_tool_registry, build_rich_context_manager, build_scenario, seed_compaction_state,
    CompleteOnFinalResponse, RealToolRegistry, ScenarioId,
};
use spore_core::{
    Agent, FullObservabilityProvider, Harness, HarnessContextManager, HarnessRunOptions,
    HarnessToolRegistry, LoopStrategy, OllamaModelInterface, RunResult, SandboxProvider, SessionId,
    SessionState, Task, TerminationPolicy, Timestamp, WorkspaceConfig, WorkspaceScopedSandbox,
};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let scenario = args
        .first()
        .and_then(|s| ScenarioId::parse(s))
        .unwrap_or_else(|| {
            eprintln!("usage: e2e_agent <s1|s2|s3|s4> [--model <id>] [--mock]");
            std::process::exit(2);
        });
    let mock = args.iter().any(|a| a == "--mock");
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());

    // Per-run session id so repeated runs don't collide in Tempo.
    let session_id = SessionId::new(format!(
        "e2e-{:?}-{}",
        scenario,
        Timestamp::now().as_str().replace([':', '.'], "-")
    ));

    // Workspace: a real directory the file tools operate inside.
    let workspace = std::env::var("SPORE_E2E_WORKSPACE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::temp_dir().join(format!("spore-e2e-{}", session_id.as_str()))
        });
    std::fs::create_dir_all(&workspace).expect("create workspace");
    prepare_workspace(scenario, &workspace);
    println!("workspace  : {}", workspace.display());

    // Observability root (honors SPORE_OTLP_ENDPOINT like observability_smoke).
    let obs_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.spore");
    let endpoint = std::env::var("SPORE_OTLP_ENDPOINT").unwrap_or_default();
    if endpoint.trim().is_empty() {
        eprintln!("note: SPORE_OTLP_ENDPOINT is unset — writing JSONL only (no Tempo forwarding).");
    }

    let result = if mock {
        run_mock(scenario, &session_id).await
    } else {
        run_live(
            scenario,
            &session_id,
            &model_id,
            &workspace,
            obs_root.clone(),
        )
        .await
    };

    match result {
        RunResult::Success { turns, output, .. } => {
            println!("result     : Success ({turns} turns)");
            println!("output     : {output:?}");
        }
        other => {
            eprintln!("result     : {other:?}");
            std::process::exit(1);
        }
    }

    // Trace info: derive the on-disk JSONL path the outbox provider wrote and
    // surface the trace_id so the user can jump straight to Grafana/Tempo. Mock
    // runs use a no-op observability provider and write no JSONL, so guard the
    // read and print a note instead of panicking. S2 runs twice under one
    // session_id (one trace dir) — printing the single id once is sufficient.
    let sid = session_id.as_str();
    println!("session_id : {sid}");
    let trace_path = obs_root.join(format!("sessions/{sid}/trace.jsonl"));
    if mock {
        println!("trace_path : (mock run — no trace)");
    } else {
        match std::fs::read_to_string(&trace_path) {
            Ok(body) => {
                let trace_id = body
                    .lines()
                    .next()
                    .and_then(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                    .and_then(|v| v["trace_id"].as_str().map(str::to_string));
                println!("trace_path : {}", trace_path.display());
                match trace_id {
                    Some(id) => {
                        println!("trace_id   : {id}");
                        if !endpoint.trim().is_empty() {
                            println!(
                                "verify     : curl -s http://localhost:3200/api/traces/{id} | jq '.batches | length'"
                            );
                        }
                    }
                    None => println!("trace_id   : (trace file empty or unparseable)"),
                }
            }
            Err(_) => {
                println!("trace_path : {} (not found)", trace_path.display());
            }
        }
    }
}

/// Run the scenario against a live Ollama model.
async fn run_live(
    scenario: ScenarioId,
    session_id: &SessionId,
    model_id: &str,
    workspace: &std::path::Path,
    obs_root: std::path::PathBuf,
) -> RunResult {
    if !OllamaModelInterface::supports_tools(model_id) {
        eprintln!(
            "warning: model `{model_id}` is not in the known tool-capable whitelist; \
             the harness will guard against tool use if the model lacks the capability."
        );
    }
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());
    let model = Arc::new(OllamaModelInterface::with_base_url(model_id, base_url));
    let agent: Arc<dyn Agent> = Arc::new(ModelAgent::new(AgentId::new("e2e-agent"), model.clone()));

    // Real tools + sandbox scoped to the workspace.
    let registry = build_real_tool_registry();
    let sandbox: Arc<dyn SandboxProvider> = Arc::new(
        WorkspaceScopedSandbox::new(WorkspaceConfig {
            root: workspace.to_path_buf(),
            allowed_paths: vec![],
            denied_paths: vec![],
            allowed_extensions: None,
            denied_extensions: vec![],
            read_only: false,
            max_file_size: 0,
        })
        .expect("build sandbox"),
    );
    let bridge = RealToolRegistry::new(registry, sandbox.clone());
    let tool_schemas = bridge.model_schemas();
    let tools: Arc<dyn HarnessToolRegistry> = Arc::new(bridge);

    // Compaction-capable context manager (small window for S3).
    let window_limit = if scenario == ScenarioId::S3 {
        200
    } else {
        128_000
    };
    let cfg = CompactionConfig {
        threshold: 0.80,
        preserve_recent_n: 2,
        head_tail_tokens: 64,
        offload_path: workspace.join(".spore/offload"),
        max_compaction_attempts: 2,
    };
    let context_manager: Arc<dyn HarnessContextManager> =
        build_rich_context_manager(model, Arc::new(OllamaCacheProvider), cfg);

    let termination: Arc<dyn TerminationPolicy> = Arc::new(CompleteOnFinalResponse);

    let obs: Arc<dyn FullObservabilityProvider> = Arc::new(OutboxObservabilityProvider::new(
        OutboxConfig::new(obs_root),
    ));

    let harness = build_scenario(
        scenario,
        agent,
        tools,
        sandbox,
        context_manager,
        termination,
        tool_schemas,
        Some(obs),
    );

    run_scenario(scenario, &harness, session_id, window_limit).await
}

/// Drive the scenario, including the S2 multi-turn and S3 seed.
async fn run_scenario(
    scenario: ScenarioId,
    harness: &impl spore_core::Harness,
    session_id: &SessionId,
    window_limit: u32,
) -> RunResult {
    let strategy = LoopStrategy::ReAct { max_iterations: 8 };

    match scenario {
        ScenarioId::S2 => {
            // Multi-turn: run twice with the same SessionId, carrying state.
            let task1 = Task::new(scenario.prompt(), session_id.clone(), strategy.clone());
            let r1 = harness.run(HarnessRunOptions::new(task1)).await;
            let state = match &r1 {
                RunResult::Success { .. } => SessionState::default(),
                other => {
                    eprintln!("S2 turn 1 did not succeed: {other:?}");
                    return r1;
                }
            };
            let task2 = Task::new(
                "Add a second TODO item to notes.md that references the first item you wrote. \
                 Use write_file with append=true. Reply DONE when finished.",
                session_id.clone(),
                strategy,
            );
            harness
                .run(HarnessRunOptions::new(task2).with_session_state(state))
                .await
        }
        ScenarioId::S3 => {
            let task = Task::new(scenario.prompt(), session_id.clone(), strategy);
            let mut state = SessionState::default();
            seed_compaction_state(
                &mut state,
                "deploy the payment service",
                session_id.clone(),
                task.id.clone(),
                window_limit,
                (window_limit as f32 * 0.82) as u32,
                12,
            );
            harness
                .run(HarnessRunOptions::new(task).with_session_state(state))
                .await
        }
        _ => {
            let task = Task::new(scenario.prompt(), session_id.clone(), strategy);
            harness.run(HarnessRunOptions::new(task)).await
        }
    }
}

/// Seed scenario-specific workspace files.
fn prepare_workspace(scenario: ScenarioId, workspace: &std::path::Path) {
    if scenario == ScenarioId::S1 {
        std::fs::write(
            workspace.join("input.txt"),
            "hello from the spore harness end to end scenario\n",
        )
        .expect("seed input.txt");
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

// ---------------------------------------------------------------------------
// Offline / mock mode (gated behind the test-utils feature).
// ---------------------------------------------------------------------------

#[cfg(feature = "test-utils")]
async fn run_mock(scenario: ScenarioId, session_id: &SessionId) -> RunResult {
    use spore_core::agent::mock::MockAgent;
    use spore_core::harness::testing::{
        AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager, ScriptedToolRegistry,
    };
    use spore_core::{TokenUsage, ToolCall, ToolOutput, TurnResult};

    let _ = scenario;
    let agent = Arc::new(MockAgent::new(AgentId::new("mock")));
    let usage = TokenUsage {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_tokens: None,
        cache_write_tokens: None,
    };
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "read_file".into(),
            input: serde_json::json!({"path": "input.txt"}),
        }],
        usage,
    });
    agent.push(TurnResult::FinalResponse {
        content: "DONE".into(),
        usage,
    });
    let tools = Arc::new(ScriptedToolRegistry::new());
    tools.push(ToolOutput::Success {
        content: "contents".into(),
        truncated: false,
    });

    let harness = build_scenario(
        scenario,
        agent,
        tools,
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
        vec![],
        None,
    );
    let task = Task::new(
        scenario.prompt(),
        session_id.clone(),
        LoopStrategy::ReAct { max_iterations: 5 },
    );
    harness.run(HarnessRunOptions::new(task)).await
}

#[cfg(not(feature = "test-utils"))]
async fn run_mock(_scenario: ScenarioId, _session_id: &SessionId) -> RunResult {
    eprintln!(
        "error: --mock requires the test-utils feature. Re-run with:\n  \
         cargo run -p spore-core --features test-utils --example e2e_agent -- <scenario> --mock"
    );
    std::process::exit(2);
}
