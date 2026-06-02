//! spore-core example 01 — hello agent.
//!
//! The smallest real thing you can build with spore-core: turn a model into a
//! running agent and ask it to say hello. No tools, no filesystem, no
//! multi-turn state.
//!
//! `HarnessBuilder::conversational(model)` defaults every required component (a
//! model-backed agent, an empty tool registry, a null sandbox, a standard
//! context manager, and respond-and-stop termination), so the whole thing is
//! three lines. Later examples override individual defaults — add tools, swap
//! the sandbox, change the loop strategy — via the builder setters.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &            # start a local model server
//! ollama pull llama3.2      # pull the default model
//! cargo run                 # or: cargo run -- --model <id>
//! ```
//!
//! `SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and the
//! Ollama endpoint (default `http://localhost:11434`).

use spore_core::{
    Harness, HarnessBuilder, HarnessRunOptions, OllamaModelInterface, RunResult, Task,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Model id + endpoint come from args/env so you can swap models without a
    // recompile.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    // A model, a harness, a task — that's the whole setup.
    let model = OllamaModelInterface::with_base_url(&model_id, base_url);
    let harness = HarnessBuilder::conversational(model).build();
    let task = Task::simple("Reply with a friendly one-line greeting.");

    println!("model      : {model_id}");
    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Success { output, turns, .. } => {
            println!("result     : Success ({turns} turn(s))");
            println!("greeting   : {output}");
            Ok(())
        }
        other => {
            eprintln!("result     : {other:?}");
            std::process::exit(1);
        }
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
