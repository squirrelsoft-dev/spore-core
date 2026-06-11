//! spore-core example 07 — the storage seam, via a `MarkdownMemoryProvider`.
//!
//! ## What it demonstrates
//! The harness is **stateless**: all durable state lives behind the storage
//! seam. Memory is just one domain of that seam — a `MemoryStore` you
//! implement. The simplest useful implementation is a human-readable markdown
//! file. This example ships [`MarkdownMemoryProvider`] (in `memory_provider.rs`),
//! composes it into a [`StorageProvider`] (NoOp for the other three domains),
//! and runs the SAME agent twice against it:
//!
//! - `--phase store`  — the agent is given facts about a fictional "Project
//!   Ironwood" and writes each as a memory via the built-in `memory` tool. The
//!   process exits leaving a readable `memory.md` on disk.
//! - `--phase recall` — a fresh process loads `memory.md` through the same
//!   provider and answers questions that restate NONE of the facts. The agent
//!   recalls them from memory via the `memory` tool.
//!
//! ## The seam
//! `main` never calls `append_memory`/`get_memories` directly. It hands the
//! composed provider to `HarnessBuilder::storage(...)`; the harness threads
//! `storage.memory()` into the built-in `memory` tool's `ToolContext` per run.
//! The agent drives all reads/writes from inside the ReAct loop. Swap the
//! provider (e.g. the built-in JSONL `FileSystemStorageProvider`) and nothing
//! else changes — that is the point of the seam.
//!
//! ## Pinned session id (critical)
//! Memory is keyed by `SessionId`; the `memory` tool always uses
//! `ctx.session_id()`. Both phases therefore pin the SAME id —
//! [`SESSION`] = `SessionId::new("project-ironwood")`, NOT
//! `SessionId::generate()`. With a generated id Run 2 would key a different
//! session and read nothing back.
//!
//! ## Scope
//! All facts use `StorageScope::Project` (the `memory` tool rejects `Local`).
//! The prompts instruct the agent to use `scope: "project"` consistently so the
//! recall read hits the same scope the store writes wrote.
//!
//! There are no `// SPEC QUESTION:` markers in this file.
//!
//! Tool calling is native by default (real typed `memory` tool schema, no
//! always-on `final` escape) — what tool-capable models, including hosted
//! `*-cloud` models, want. Pass `--structured` to enable
//! `ModelParams::structured_tool_calls` (schema-constrained decoding) for small
//! local models that otherwise leak `<|python_tag|>` or malformed JSON.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull llama3.2
//! cargo run -- --phase store     # writes memory.md
//! cat memory.md                  # inspect the human-readable artifact
//! cargo run -- --phase recall    # answers from memory.md alone
//! ```

mod memory_provider;

use std::sync::Arc;

use memory_provider::MarkdownMemoryProvider;
use spore_core::{
    Harness, HarnessBuilder, HarnessRunOptions, HarnessStreamEvent, LoopStrategy,
    OllamaModelInterface, ReactConfig, RunResult, SessionId, StandardTools, Task,
};

/// The pinned session id shared by BOTH phases. See the module docs: memory is
/// session-keyed, so store and recall MUST agree on this id or recall reads
/// nothing.
fn session() -> SessionId {
    SessionId::new("project-ironwood")
}

const STORE_SYSTEM_PROMPT: &str =
    "You are a memory-keeping agent. You will be given a briefing of \
    facts. For EACH distinct fact, call the `memory` tool with operation \"write\", scope \
    \"project\", role \"assistant\", and the fact text as `content`. Write the facts verbatim and \
    one at a time. Do not summarize or merge facts. When every fact has been written, reply with a \
    short confirmation of how many you stored.";

const RECALL_SYSTEM_PROMPT: &str = "You are a recall agent. Everything you know about Project \
    Ironwood lives in memory — nothing is in this prompt. FIRST call the `memory` tool with \
    operation \"read\", scope \"project\" to load what you remember. THEN answer the user's \
    questions using only the recalled memories. Cite the relevant remembered fact when you answer. \
    Do not invent facts that are not in memory.";

const RECALL_QUESTIONS: &str = "Answer these about Project Ironwood, using only your memory:\n\
    1. How many engineers are on the team, and who leads it?\n\
    2. What database was chosen as the system of record, and why over the alternative?\n\
    3. What are the two hard constraints?\n\
    4. What is the known single point of failure?";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());
    // Opt-in constrained decoding. OFF by default: tool-capable models (incl.
    // `*-cloud`) use native Ollama tool calling, which gives the `memory` tool a
    // real typed schema and no always-on `final` escape. Small local models that
    // leak `<|python_tag|>` can pass `--structured` to force the JSON-object channel.
    let structured = args.iter().any(|a| a == "--structured");

    // `memory.md` lives next to this example's sources so `cargo run` works from
    // anywhere and the artifact is easy to find and inspect between phases.
    let memory_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("memory.md");

    // Default (no --phase): run store, then point the user at recall and exit.
    let phase = arg_value(&args, "--phase").unwrap_or_else(|| "store".to_string());
    let default_phase = arg_value(&args, "--phase").is_none();

    let model = OllamaModelInterface::with_base_url(&model_id, base_url);

    println!("model      : {model_id}");
    println!("memory.md  : {}", memory_path.display());
    println!(
        "session id : {}  (pinned — shared by both phases)",
        session().as_str()
    );
    println!("phase      : {phase}\n");

    match phase.as_str() {
        "store" => {
            // Read the briefing and feed it to the agent. The agent writes each
            // fact via the `memory` tool; `main` never writes memory itself.
            let briefing = std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("project-ironwood.md"),
            )?;
            // Start each phase from a clean slate is NOT done here on purpose —
            // append is idempotent enough for a demo and re-running store just
            // appends; delete memory.md by hand to reset.

            let task_prompt = format!(
                "Here is the Project Ironwood briefing. Store each fact to memory.\n\n{briefing}"
            );
            let output =
                run_phase(model, &memory_path, STORE_SYSTEM_PROMPT, &task_prompt, structured)
                    .await?;
            println!("\nstored. agent said: {output}");
            if memory_path.exists() {
                println!("\nmemory.md now exists on disk: {}", memory_path.display());
                println!("inspect it, then run:  cargo run -- --phase recall");
            }
            if default_phase {
                println!(
                    "\n(no --phase given, so we ran `store`. Now run `cargo run -- --phase recall`.)"
                );
            }
            Ok(())
        }
        "recall" => {
            if !memory_path.exists() {
                eprintln!(
                    "memory.md does not exist yet at {}.\nRun `cargo run -- --phase store` first.",
                    memory_path.display()
                );
                std::process::exit(2);
            }
            let output = run_phase(
                model,
                &memory_path,
                RECALL_SYSTEM_PROMPT,
                RECALL_QUESTIONS,
                structured,
            )
            .await?;
            println!("\nanswers from memory:\n{output}");
            Ok(())
        }
        other => {
            eprintln!("unknown --phase {other:?}. Use `store` or `recall`.");
            std::process::exit(2);
        }
    }
}

/// Build a harness over the markdown memory provider + the built-in `memory`
/// tool, pin the shared session id, run one task, and stream the loop.
async fn run_phase(
    model: OllamaModelInterface,
    memory_path: &std::path::Path,
    system_prompt: &str,
    task_prompt: &str,
    structured: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    // Compose the real markdown MemoryStore with NoOp for the other three
    // storage domains. This is the entire integration: the harness threads
    // `storage.memory()` into the `memory` tool's context per run.
    let storage = MarkdownMemoryProvider::new(memory_path).into_storage_provider();

    let harness = HarnessBuilder::conversational(model)
        .storage(Arc::new(storage))
        .tool(StandardTools::memory())
        .system_prompt(system_prompt)
        // Native tool calling by default; `--structured` (threaded in from main)
        // flips on constrained decoding for small models. With structured mode the
        // "think · turn N" line is just a turn marker, not model chatter, since
        // each turn emits one clean JSON tool call.
        .model_params(spore_core::ModelParams {
            structured_tool_calls: structured,
            ..Default::default()
        })
        .build();

    // PIN the session id — both phases pass the same one so recall reads what
    // store wrote.
    let task = Task::new(
        task_prompt.to_string(),
        session(),
        LoopStrategy::ReAct(ReactConfig::per_loop(20)),
    );

    let options = HarnessRunOptions::new(task).with_stream(Box::new(
        |event: HarnessStreamEvent| match event {
            HarnessStreamEvent::TurnStart { turn, .. } => println!("think  · turn {turn}"),
            HarnessStreamEvent::ToolCall { name, args, .. } => {
                println!("    act    → {name}({})", truncate(&args.to_string(), 160));
            }
            HarnessStreamEvent::ToolResult {
                is_error, content, ..
            } => {
                let tag = if is_error { "obs(err)" } else { "obs " };
                println!("    {tag}→ {}", truncate(&content, 160));
            }
            _ => {}
        },
    ));

    match harness.run(options).await {
        RunResult::Success { output, .. } => Ok(output),
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

/// Keep stream lines readable — memory reads return a JSON array of entries.
fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max {
        s
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}
