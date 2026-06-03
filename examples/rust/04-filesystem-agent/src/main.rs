//! spore-core example 04 — ReAct with the built-in catalogue file tools.
//!
//! This is [`03-tool-use`](../../03-tool-use) with one substantive change. In 03
//! the agent's tools were hand-rolled: we implemented the harness-loop
//! `ToolRegistry` ourselves and dispatched each call by hand. Here we register
//! spore-core's *built-in* catalogue instead — a single builder line:
//!
//! ```ignore
//! .tools(StandardTools::coding_set())   // read_file, write_file, list_dir, …
//! ```
//!
//! Everything else is the same: the same `conversational(model)` builder, the
//! same `ReAct` loop, the same stream-printed `think · turn N` / tool-call
//! output. The thesis of this example is exactly that: **the harness doesn't
//! change — only the registration path does.**
//!
//! ## What it shows
//!
//! - **Catalogue registration.** `.tools(StandardTools::coding_set())` advertises
//!   and dispatches `read_file` / `write_file` / `list_dir` (and friends) with no
//!   bespoke code.
//! - **A real sandbox.** Catalogue file tools go through a sandbox, so unlike 03's
//!   pure-compute tools (which were happy with the default `NullSandbox`) this
//!   example wires a [`WorkspaceScopedSandbox`] scoped to `sample-files/`.
//! - **A side effect that outlives the process.** The agent writes `SUMMARY.md`
//!   into `sample-files/`. It is still there after the program exits — the first
//!   example that leaves something behind on disk.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull llama3.2
//! cargo run
//! ```

use std::sync::Arc;

use spore_core::{
    Harness, HarnessBuilder, HarnessRunOptions, HarnessStreamEvent, LoopStrategy,
    OllamaModelInterface, RunResult, SessionId, StandardTools, Task, WorkspaceConfig,
    WorkspaceScopedSandbox,
};

const SYSTEM_PROMPT: &str = "You are a file-summarizing agent. Use list_dir to find files, \
     read_file to read each, and write_file to create SUMMARY.md. \
     Act using tools — do not just describe.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    // The agent operates inside the shipped `sample-files/` directory. Resolve
    // it relative to this source file so `cargo run` works from anywhere, and
    // canonicalize it — the sandbox requires a canonical, existing root.
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("sample-files");
    let workspace_root = std::fs::canonicalize(&workspace_root)?;

    let prompt = arg_value(&args, "--prompt").unwrap_or_else(|| {
        "There are several .txt files in this directory. Use list_dir to find them, \
         read_file to read each one, then write a SUMMARY.md containing a one-sentence \
         summary of every file. Use write_file to create it."
            .to_string()
    });

    // Same `conversational` harness as 03 — the ONLY substantive change is that
    // we register the built-in catalogue (`.tools(...)`) over a real sandbox
    // instead of hand-rolling a `ToolRegistry`.
    let model = OllamaModelInterface::with_base_url(&model_id, base_url);
    let sandbox = WorkspaceScopedSandbox::new(WorkspaceConfig::scoped(workspace_root.clone()))?;
    let harness = HarnessBuilder::conversational(model)
        .sandbox(Arc::new(sandbox))
        .tools(StandardTools::coding_set())
        .system_prompt(SYSTEM_PROMPT)
        .build();

    let task = Task::new(
        prompt.clone(),
        SessionId::generate(),
        LoopStrategy::ReAct { max_iterations: 8 },
    );
    // Print each turn (Think) and each catalogue tool call + result (Act /
    // Observe). Because the catalogue dispatches internally, the Act/Observe
    // lines come from harness STREAM events, not from inside a hand-rolled
    // dispatch like 03.
    let options = HarnessRunOptions::new(task).with_stream(Box::new(
        |event: HarnessStreamEvent| match event {
            HarnessStreamEvent::TurnStart { turn } => println!("think  · turn {turn}"),
            HarnessStreamEvent::ToolCall { name, args, .. } => {
                println!("    act    → {name}({args})");
            }
            HarnessStreamEvent::ToolResult {
                is_error, content, ..
            } => {
                let tag = if is_error { "obs(err)" } else { "obs " };
                println!("    {tag}→ {}", truncate(&content, 200));
            }
            _ => {}
        },
    ));

    println!("model  : {model_id}");
    println!("dir    : {}", workspace_root.display());
    println!("prompt : {prompt}\n");
    match harness.run(options).await {
        RunResult::Success { output, turns, .. } => {
            println!("\nanswer ({turns} turn(s)): {output}");
            let summary = workspace_root.join("SUMMARY.md");
            if summary.exists() {
                println!("\nSUMMARY.md now exists on disk: {}", summary.display());
            }
            Ok(())
        }
        other => {
            eprintln!("\nrun did not succeed: {other:?}");
            std::process::exit(1);
        }
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Keep observe lines readable — file contents can be long.
fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max {
        s
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}
