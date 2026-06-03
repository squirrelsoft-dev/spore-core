//! spore-core example 05 — a custom tool you write yourself.
//!
//! Examples [`03`](../../03-tool-use) and [`04`](../../04-filesystem-agent)
//! showed the two *built-in* tool paths: hand-rolling the harness-loop
//! `ToolRegistry` (03) and registering the shipped catalogue with
//! `.tools(StandardTools::coding_set())` (04). This example shows the third and
//! most important path — **bringing your own tool** — and the public extension
//! point that makes it possible: the [`Tool`] trait.
//!
//! ## The two custom tools
//!
//! Both are defined in [`mod tools`](tools) as plain `impl Tool`:
//!
//! - **`remember(key, value)`** — persists a fact into the run store
//!   ([`tools::remember`]). It MUTATES shared state, so it is not `read_only`.
//! - **`recall(key)`** — reads a fact back out ([`tools::recall`]). It only
//!   reads, so it is `read_only` + `idempotent`.
//!
//! ## The trait used, and the seam it exposes
//!
//! Each tool implements [`spore_core::Tool`]. The `execute` method receives two
//! seams: a `SandboxProvider` (the environment — unused here, these tools never
//! touch the filesystem) and a [`ToolContext`](spore_core::ToolContext) (the
//! storage seam: `run_store()` + `session_id()`). Facts are keyed under
//! `fact:{key}` so they cannot collide with reserved catalogue keys.
//!
//! ## The pattern: `impl Tool` → `StandardTool::new` → `.tool()`
//!
//! 1. Implement [`Tool`](spore_core::Tool).
//! 2. Bundle the impl with its schema via
//!    [`StandardTool::new`](spore_core::StandardTool) so the two can never drift.
//! 3. Register each with `.tool(...)`. The harness wires the sandbox and a
//!    per-run `ToolContext` automatically — **the harness doesn't change, only
//!    what you register does.**
//!
//! Two builder differences from 04: there is no `.tools(...)` catalogue, and no
//! explicit `.sandbox(...)` / `.storage(...)`. `build()` defaults storage to an
//! in-memory provider whenever `.tool()` tools are present, so the run store
//! works for free.
//!
//! (No `// SPEC QUESTION:` markers remain — all design points are resolved.)
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull llama3.2
//! cargo run
//! ```

mod tools;

use spore_core::{
    Harness, HarnessBuilder, HarnessRunOptions, HarnessStreamEvent, LoopStrategy,
    OllamaModelInterface, RunResult, SessionId, StandardTool, Task,
};

use tools::recall::RecallTool;
use tools::remember::RememberTool;

const SYSTEM_PROMPT: &str = "You are a research agent with a memory. Research the topic the user \
     gives you across several turns. As you discover each fact, call `remember` to store it under \
     a short, stable key (e.g. 'habitat', 'diet'). Keep track of the keys you use. When you have \
     gathered enough facts, call `recall` on each key you remembered, then write a final summary \
     built ONLY from the recalled facts. Act using tools — do not just describe.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    let prompt = arg_value(&args, "--prompt").unwrap_or_else(|| {
        "Research the common octopus. Remember a few key facts (habitat, diet, lifespan, \
         intelligence), then recall them and write a short summary."
            .to_string()
    });

    // Same `conversational` harness as 03 / 04 — the substantive change is that
    // we register two tools WE wrote (`.tool(...)`) instead of a catalogue
    // preset. No `.sandbox(...)` (these tools ignore it) and no `.storage(...)`
    // (build() defaults to in-memory storage when `.tool()` tools are present).
    let model = OllamaModelInterface::with_base_url(&model_id, base_url);
    let harness = HarnessBuilder::conversational(model)
        .tool(StandardTool::new(
            Box::new(RememberTool::new()),
            RememberTool::schema(),
        ))
        .tool(StandardTool::new(
            Box::new(RecallTool::new()),
            RecallTool::schema(),
        ))
        .system_prompt(SYSTEM_PROMPT)
        .build();

    let task = Task::new(
        prompt.clone(),
        SessionId::generate(),
        LoopStrategy::ReAct { max_iterations: 12 },
    );
    // Print each turn (Think) and each tool call + result (Act / Observe) from
    // harness STREAM events — the builder dispatches our tools internally, just
    // as it does the catalogue in 04.
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
    println!("tools  : remember, recall");
    println!("prompt : {prompt}\n");
    match harness.run(options).await {
        RunResult::Success { output, turns, .. } => {
            println!("\nsummary ({turns} turn(s)): {output}");
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

/// Keep observe lines readable — recalled facts can be long.
fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max {
        s
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}
