//! spore-core example 12 — **cordyceps**: a basic ReAct coding agent.
//!
//! This is [`04-filesystem-agent`](../../04-filesystem-agent) with two changes:
//!
//! 1. the workspace sandbox is **read-write** and the agent gets the full
//!    [`StandardTools::coding_set`] (read/write/edit/list/grep/find + `bash`), so
//!    it can actually change code — not just summarize files; and
//! 2. the single `harness.run(...)` is wrapped in a **REPL**: build the harness
//!    once, then read a task per line and run a fresh ReAct loop for each.
//!
//! Everything else is 04 verbatim — the same `conversational(model)` builder, the
//! same `ReAct` loop, the same stream-printed `think · turn N` / `act` / `obs`
//! trace. The thesis is the same as 04's: **the harness doesn't change** — here
//! we only widen the toolset and drive it in a loop.
//!
//! ## What it shows
//!
//! - **A REPL over one harness.** The harness is built once and reused; each line
//!   you type becomes a new [`Task`] with a fresh [`SessionId`] running its own
//!   `ReAct` loop. Side effects live on disk, so the workspace carries state
//!   across turns even though each turn is a fresh session.
//! - **A real coding sandbox.** Catalogue file tools go through a
//!   [`WorkspaceScopedSandbox`] scoped to `workspace/`. Unlike 04 it is NOT
//!   read-only, so `write_file` / `edit_file` / `bash` can build things there.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull gemma4:e4b
//! cargo run        # type a coding task; Ctrl-D to quit
//! ```

use std::io::Write;
use std::sync::Arc;

use spore_core::{
    Harness, HarnessBuilder, HarnessRunOptions, HarnessStreamEvent, LoopStrategy,
    OllamaModelInterface, ReactConfig, RunResult, SessionId, StandardTools, Task, WorkspaceConfig,
    WorkspaceScopedSandbox,
};

const SYSTEM_PROMPT: &str = "You are a coding agent working inside a sandboxed workspace directory. \
     Explore with list_dir, read_file, grep, and find_files; create and change files with \
     write_file and edit_file; run commands with bash. Use `.` and relative paths only. \
     Act using tools — do not just describe what you would do. When the task is done, reply with \
     a short summary of what you changed.";

/// Per-loop ReAct step budget for one REPL turn (04 used 8; a coding task wants
/// more room to explore, edit, and verify).
const MAX_STEPS: u32 = 25;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // A tool-capable model is required — a small model that only narrates tool
    // use (e.g. llama3.2 3B) will never act. Default to gemma4:e4b or better.
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "gemma4:e4b".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    // The agent operates inside a writable `workspace/` directory next to this
    // crate. Resolve it relative to this source file so `cargo run` works from
    // anywhere, create it if missing, and canonicalize it — the sandbox requires
    // a canonical, existing root.
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("workspace");
    std::fs::create_dir_all(&workspace_root)?;
    let workspace_root = std::fs::canonicalize(&workspace_root)?;

    // The SAME conversational ReAct harness as 04 — the only differences are a
    // read-WRITE sandbox and the full coding catalogue. Built once and reused for
    // every REPL turn.
    let model = OllamaModelInterface::with_base_url(&model_id, base_url);
    let sandbox = WorkspaceScopedSandbox::new(WorkspaceConfig::scoped(workspace_root.clone()))?;
    let harness = HarnessBuilder::conversational(model)
        .sandbox(Arc::new(sandbox))
        .tools(StandardTools::coding_set())
        .system_prompt(SYSTEM_PROMPT)
        .build();

    println!("spore-core — basic ReAct coding agent");
    println!("model     : {model_id}");
    println!("workspace : {}", workspace_root.display());
    println!("tools     : read_file, write_file, edit_file, list_dir, grep, find_files, bash, …");
    println!("Type a coding task and press enter. Ctrl-D to quit.\n");

    while let Some(prompt) = read_prompt() {
        if prompt.trim().is_empty() {
            continue;
        }
        // Each REPL turn is a fresh ReAct task over the SAME workspace. Files the
        // agent wrote on a previous turn are still on disk, so it can build on
        // them even though the conversation itself starts clean each time.
        let task = Task::new(
            prompt,
            SessionId::generate(),
            LoopStrategy::ReAct(ReactConfig::per_loop(MAX_STEPS)),
        );
        // Print each turn (Think) and each catalogue tool call + result (Act /
        // Observe) — the exact stream trace from 04.
        let options = HarnessRunOptions::new(task).with_stream(Box::new(
            |event: HarnessStreamEvent| match event {
                HarnessStreamEvent::TurnStart { turn, .. } => println!("think  · turn {turn}"),
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

        match harness.run(options).await {
            RunResult::Success { output, turns, .. } => {
                println!("\nanswer ({turns} turn(s)): {output}\n");
            }
            other => {
                eprintln!("\nrun did not succeed: {other:?}\n");
            }
        }
    }

    println!("\nbye.");
    Ok(())
}

/// Read one task line from the REPL. `Some(line)` to run; `None` on EOF (Ctrl-D),
/// which quits.
fn read_prompt() -> Option<String> {
    print!("code> ");
    let _ = std::io::stdout().flush();
    let mut buf = String::new();
    match std::io::stdin().read_line(&mut buf) {
        Ok(0) => None, // EOF
        Ok(_) => Some(buf.trim_end_matches(['\n', '\r']).to_string()),
        Err(_) => None,
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
