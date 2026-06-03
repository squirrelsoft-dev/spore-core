//! spore-core example 08 — multi-step goal decomposition with PlanExecute.
//!
//! This is the first example to swap the **loop strategy**. Everything else —
//! the `conversational(model)` builder, the `WorkspaceScopedSandbox`, and the
//! tool set (`web_search` + `write_file` + `read_file`, identical to 06) — is
//! held constant. The ONLY substantive change is one line on the `Task`:
//!
//! ```text
//! // 06 — react step-by-step:
//! LoopStrategy::ReAct { max_iterations: 10 }
//! // 08 — decompose the goal first, then execute each subtask:
//! LoopStrategy::PlanExecute { plan_model: None }
//! ```
//!
//! With `PlanExecute`, the harness runs one constrained planner turn FIRST: the
//! model must return strict JSON `{ "tasks": [...], "rationale": ... }`. That
//! plan is captured into a `PlanArtifact`, surfaced, then each subtask is run in
//! a bounded sub-loop. The turn budget is divided across subtasks (per-task cap
//! = remaining_turns / remaining_tasks), so we set a generous `max_turns` — e.g.
//! an 8-step plan at 64 turns gives each subtask ~8 turns instead of starving.
//!
//! ## Surfacing the plan — via lifecycle HOOKS, not stream events
//!
//! There are no plan/subtask *stream* events; the plan is visible through the
//! hook chain. We register a [`PlanExecuteReporter`] (`impl Hook`) on two events:
//!
//! - `OnPlanCreated { plan: &mut PlanArtifact }` fires post-capture / pre-execute
//!   — we print a `── plan ──` banner: the rationale, then the numbered tasks.
//! - `OnTaskAdvance { task: &mut Task, task_index, total_tasks }` fires before
//!   each subtask — we print `[i/N] <instruction>`.
//!
//! The plan is also persisted to `session_state.extras["plan_execute"]`
//! (`PLAN_EXECUTE_EXTRAS_KEY`); we read it back on success to confirm.
//!
//! Tools wired (all from the built-in catalogue, identical to 06):
//!
//! - `web_search` — a [`WebSearchTool::with_config`] backend configured for
//!   SearXNG: the query is issued as `GET <endpoint>?q=<query>` against
//!   `SPORE_WEB_SEARCH_ENDPOINT` (with `format=json` on the endpoint).
//! - `write_file` — the agent writes `async-comparison.md` into `workspace/`.
//! - `read_file` — lets the agent re-read what it wrote.
//!
//! There are no `// SPEC QUESTION:` markers: the strategy swap, the hook events,
//! and the budget API were all resolved against the source before writing this.
//!
//! This example also enables `ModelParams::structured_tool_calls` via
//! `HarnessBuilder::model_params(..)` — schema-constrained decoding that helps
//! small Ollama models emit one clean tool call per turn across both the plan
//! and execute phases.
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

use spore_core::harness::BoxFut;
use spore_core::{
    BudgetLimits, Harness, HarnessBuilder, HarnessRunOptions, HarnessStreamEvent, Hook, HookChain,
    HookContext, HookDecision, HookError, HookEvent, LoopStrategy, OllamaModelInterface, RunResult,
    SearchMethod, SessionId, StandardHookChain, StandardTool, StandardTools, Task, WebSearchConfig,
    WebSearchTool, WorkspaceConfig, WorkspaceScopedSandbox, PLAN_EXECUTE_EXTRAS_KEY,
};

const SYSTEM_PROMPT: &str = "You are a research-and-writing agent. Your ONLY capabilities are: \
     web_search (find current information online), read_file, and write_file (save your work to \
     the workspace). You have NO shell or terminal — you cannot install software, set up projects \
     or environments, run/compile/build code, or execute commands. Decompose the goal into \
     subtasks that are each achievable with web_search and writing alone; never plan setup, \
     installation, or build steps. For each subtask, use web_search to gather current information, \
     then synthesize a clear, cited comparison and save the final document with write_file. Act \
     using tools — do not answer from memory alone.";

/// Lifecycle hook that prints the PlanExecute plan and each subtask as it runs.
///
/// `OnPlanCreated` fires once, after the planner turn captures the plan and
/// before any subtask executes — the money moment for PlanExecute. `OnTaskAdvance`
/// fires before each subtask. Both are sync, post/pre, plan/task-carrying events.
/// This hook only observes; it always returns [`HookDecision::Continue`].
struct PlanExecuteReporter;

impl Hook for PlanExecuteReporter {
    fn handle<'a>(
        &'a self,
        ctx: &'a mut HookContext<'a>,
    ) -> BoxFut<'a, Result<HookDecision, HookError>> {
        match ctx {
            HookContext::OnPlanCreated { plan, .. } => {
                println!("\n── plan ──");
                if !plan.rationale.trim().is_empty() {
                    println!("rationale: {}", plan.rationale);
                }
                for (i, task) in plan.tasks.iter().enumerate() {
                    println!("  {}. {task}", i + 1);
                }
                println!("──────────\n");
            }
            HookContext::OnTaskAdvance {
                task,
                task_index,
                total_tasks,
                ..
            } => {
                println!("[{}/{}] {}", *task_index + 1, total_tasks, task.instruction);
            }
            _ => {}
        }
        Box::pin(async { Ok(HookDecision::Continue) })
    }

    fn events(&self) -> Vec<HookEvent> {
        vec![HookEvent::OnPlanCreated, HookEvent::OnTaskAdvance]
    }

    fn name(&self) -> String {
        "plan-execute-reporter".to_string()
    }
}

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
        // A multi-step goal that benefits from upfront decomposition: search each
        // runtime, synthesize a comparison, then write the file.
        "Research the Rust async ecosystem, write a comparison of tokio vs async-std vs smol \
         covering performance, ecosystem maturity, and use cases, and save it to \
         async-comparison.md."
            .to_string()
    });

    // Register the plan reporter on a StandardHookChain. The chain is how the plan
    // becomes visible: there are no plan/subtask stream events.
    let chain = StandardHookChain::new();
    chain.register(Arc::new(PlanExecuteReporter))?;

    // Same `conversational` harness + `WorkspaceScopedSandbox` + tool set as 06.
    // The ONLY substantive change vs 06 is the `LoopStrategy` on the Task below.
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
        .hooks(Arc::new(chain))
        .build();

    // THE ONE-LINE SWAP. 06 used `LoopStrategy::ReAct { max_iterations: 10 }`;
    // here we decompose first via PlanExecute. The turn budget is divided across
    // subtasks, so we give it generous headroom.
    let task = Task::new(
        prompt.clone(),
        SessionId::generate(),
        LoopStrategy::PlanExecute { plan_model: None },
    )
    .with_budget(BudgetLimits {
        max_turns: Some(64),
        ..BudgetLimits::default()
    });

    // Print each turn (Think) and each tool call + result (Act / Observe). This is
    // most useful for the plan-phase turn; Rust suppresses the subtask sub-loop
    // stream, so the hooks above are the portable view of execution.
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
    println!("strategy : PlanExecute (06 used ReAct)");
    println!("prompt   : {prompt}\n");
    match harness.run(options).await {
        RunResult::Success {
            output,
            turns,
            session_state,
            ..
        } => {
            println!("\nanswer ({turns} turn(s)): {output}");
            // The captured plan is persisted in extras — confirm it round-tripped.
            if let Some(plan) = session_state.extras.get(PLAN_EXECUTE_EXTRAS_KEY) {
                if let Some(tasks) = plan.get("tasks").and_then(|t| t.as_array()) {
                    println!("\nplan persisted in extras[\"{PLAN_EXECUTE_EXTRAS_KEY}\"] with {} subtask(s)", tasks.len());
                }
            }
            let doc = workspace_root.join("async-comparison.md");
            if doc.exists() {
                println!(
                    "\nasync-comparison.md now exists on disk: {}",
                    doc.display()
                );
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
