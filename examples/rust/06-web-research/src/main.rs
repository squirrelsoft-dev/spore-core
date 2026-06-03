//! spore-core example 06 — web research with an external API tool.
//!
//! This is the first example whose tools reach **outside the process** to a
//! third-party HTTP service. The whole point is that this changes *nothing*
//! about the harness: an external API is just another tool.
//!
//! Tools wired (all from the built-in catalogue, no custom `impl Tool`):
//!
//! - `web_search` — a [`WebSearchTool::with_config`] backend configured for
//!   SearXNG: the query is issued as `GET <endpoint>?q=<query>` (the
//!   `format=json` selector rides on the endpoint) and the JSON response body
//!   is returned to the agent verbatim. The endpoint comes from
//!   `SPORE_WEB_SEARCH_ENDPOINT` (see the README + `.env.example`).
//! - `write_file` — [`StandardTools::write_file`]. The agent writes its
//!   synthesized, cited answer to `answer.md`.
//! - `read_file` — [`StandardTools::read_file`]. Lets the agent re-read what it
//!   wrote (e.g. to verify or revise the answer).
//!
//! Harness + sandbox pattern reused verbatim from
//! [`04-filesystem-agent`](../../04-filesystem-agent):
//!
//! - `HarnessBuilder::conversational(model)` — same builder.
//! - `LoopStrategy::ReAct { max_iterations }` — same loop.
//! - `WorkspaceScopedSandbox` over `WorkspaceConfig::scoped(root)` — same
//!   sandbox, here scoped to this example's `workspace/` dir so `write_file`
//!   cannot escape it. 04 wrote `SUMMARY.md`; 06 writes `answer.md`.
//!
//! The ONLY substantive difference from 04 is the tool set: 04 registers
//! `coding_set()`, 06 registers a SearXNG-configured `web_search` (GET +
//! `?q=`) + `write_file` + `read_file`. Same harness, different tools.
//!
//! There are no `// SPEC QUESTION:` markers: the backend-adapter, file-write
//! path, and model choices were all resolved before this example was written.
//!
//! This example also enables `ModelParams::structured_tool_calls` via
//! `HarnessBuilder::model_params(..)` — schema-constrained decoding that helps
//! small Ollama models emit one clean tool call per turn instead of malformed
//! or interleaved output.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull llama3.2
//! export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"   # SearXNG JSON API
//! cargo run
//! ```

use std::sync::Arc;

use spore_core::{
    Harness, HarnessBuilder, HarnessRunOptions, HarnessStreamEvent, LoopStrategy,
    OllamaModelInterface, RunResult, SearchMethod, SessionId, StandardTool, StandardTools, Task,
    WebSearchConfig, WebSearchTool, WorkspaceConfig, WorkspaceScopedSandbox,
};

const SYSTEM_PROMPT: &str = "You are a web-research agent. Use web_search to find current \
     information, synthesize what you learn into a clear answer, and ALWAYS cite the sources \
     you used. Write the final answer to answer.md using write_file. Act using tools — do not \
     answer from memory alone.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    // The search backend endpoint. `web_search` issues `GET <endpoint>?q=<query>`
    // and returns the JSON body to the agent. There is no live backend in
    // spore-core, so you must supply one — this example targets a self-hosted
    // SearXNG JSON endpoint (`.../search?format=json`). Raw Brave/Tavily would use
    // auth headers (now supported via `WebSearchConfig::auth_headers`, core #108);
    // this example targets SearXNG, which needs no key. See README.
    let endpoint = match std::env::var("SPORE_WEB_SEARCH_ENDPOINT") {
        Ok(e) if !e.trim().is_empty() => e,
        _ => {
            eprintln!(
                "SPORE_WEB_SEARCH_ENDPOINT is not set.\n\
                 Set it to a SearXNG JSON endpoint, e.g. \
                 http://localhost:8888/search?format=json — the query is appended as \
                 `&q=<query>`.\n\
                 See .env.example and the README."
            );
            std::process::exit(2);
        }
    };

    // SearXNG GET config: `?q=<query>` appended to the endpoint, `format=json`
    // already on the endpoint URL. No auth env vars, so `with_config` cannot fail
    // here — but it returns a `Result`, so surface any error with a clear message.
    let web_search_tool = WebSearchTool::with_config(WebSearchConfig {
        endpoint: endpoint.clone(),
        method: SearchMethod::Get,
        query_param: "q".into(),
        auth_headers: Vec::new(),
        body_auth_params: Vec::new(),
    })
    .expect("web_search backend config is valid (SearXNG needs no auth env vars)");
    let web_search = StandardTool::new(Box::new(web_search_tool), WebSearchTool::schema());

    // The agent operates inside this example's `workspace/` directory. Resolve it
    // relative to this source file so `cargo run` works from anywhere, and
    // canonicalize it — the sandbox requires a canonical, existing root.
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("workspace");
    std::fs::create_dir_all(&workspace_root)?;
    let workspace_root = std::fs::canonicalize(&workspace_root)?;

    let prompt = arg_value(&args, "--prompt").unwrap_or_else(|| {
        // A TIMELESS research question: the answer evolves over time but the
        // question stays interesting and is not tied to a single news event.
        "What is the current recommended way to install Rust on macOS, and what are the main \
         alternatives? Search the web, synthesize the options, cite your sources, and write the \
         answer to answer.md."
            .to_string()
    });

    // Same `conversational` harness + `WorkspaceScopedSandbox` as 04. The ONLY
    // substantive change is the tool set: `web_search` (external API) composes
    // with `write_file` / `read_file` in one builder chain. `.tool()` and
    // `.tools()` push into the same registry with last-wins upsert by name.
    let model = OllamaModelInterface::with_base_url(&model_id, base_url);
    let sandbox = WorkspaceScopedSandbox::new(WorkspaceConfig::scoped(workspace_root.clone()))?;
    let harness = HarnessBuilder::conversational(model)
        .sandbox(Arc::new(sandbox))
        .tool(web_search)
        .tool(StandardTools::write_file())
        .tool(StandardTools::read_file())
        .system_prompt(SYSTEM_PROMPT)
        // Structured mode helps small Ollama models emit clean tool calls (one
        // per turn, no interleaved reasoning — so the "think · turn N" line is
        // just a turn marker, not model chatter).
        .model_params(spore_core::ModelParams {
            structured_tool_calls: true,
            ..Default::default()
        })
        .build();

    let task = Task::new(
        prompt.clone(),
        SessionId::generate(),
        LoopStrategy::ReAct { max_iterations: 10 },
    );
    // Print each turn (Think) and each tool call + result (Act / Observe). The
    // search queries and result snippets show up here because `web_search`
    // dispatches through the harness like any other catalogue tool.
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

    println!("model    : {model_id}");
    println!("endpoint : {endpoint}");
    println!("workspace: {}", workspace_root.display());
    println!("prompt   : {prompt}\n");
    match harness.run(options).await {
        RunResult::Success { output, turns, .. } => {
            println!("\nanswer ({turns} turn(s)): {output}");
            let answer = workspace_root.join("answer.md");
            if answer.exists() {
                println!("\nanswer.md now exists on disk: {}", answer.display());
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

/// Keep observe lines readable — search results can be long.
fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max {
        s
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}
