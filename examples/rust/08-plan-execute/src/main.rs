//! spore-core example 08 — multi-step goal decomposition with PlanExecute.
//!
//! This is the first example to swap the **loop strategy**. Everything else —
//! the `conversational(model)` builder, the `WorkspaceScopedSandbox`, and the
//! tool set (`web_search` + `write_file` + `read_file`, identical to 06) — is
//! held constant. The substantive change is the `LoopStrategy` on the `Task`:
//!
//! ```text
//! // 06 — react step-by-step:
//! LoopStrategy::ReAct(ReactConfig::per_loop(10))
//! // 08 — decompose the goal first, then execute each subtask:
//! LoopStrategy::PlanExecute(PlanExecuteConfig {
//!     plan: ReAct{ output: Some("plan-schema") },  // structured slot ⇒ schema
//!     execute: ReAct{ .. },
//!     ..PlanExecuteConfig::simple(None)
//! })
//! ```
//!
//! Post-#119 the `LoopStrategy` is a recursive enum of config newtypes, so the
//! strategy is a composed tree rather than a flat literal. `PlanExecute`'s `plan`
//! slot is STRUCTURED — a bare `ReAct` there MUST declare an `output` schema
//! (here `plan-schema`), which `ExecutionRegistry::validate` enforces at run
//! entry. The empty agent / toolset handles on the leaves are default-filled by
//! the builder; only the `plan-schema` handle is registered explicitly.
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
//! ## Staying within small (~128K) context windows
//!
//! Under `PlanExecute`, verbose tool output is retained across every plan step,
//! so a few searches can overflow a model with a ~128K window (e.g. `gemma4:e4b`,
//! 131072 tokens). Two measures keep this example running cleanly on such models:
//!
//! - It **distills `web_search` results**: a [`ConciseWebSearch`] wrapper trims
//!   the verbatim SearXNG JSON (25-40K tokens/call) down to the top 6 results
//!   with only `title` / `url` / `content`, so context stays small.
//! - It **lowers the compaction threshold** to `0.45` (compaction at ≈90K tokens
//!   instead of the default ≈160K), installed via `.context_manager(...)`, so
//!   compaction fires before a 128K-window model overflows.
//!
//! Tool calling is native by default (real typed `write_file` schema, no
//! always-on `final` escape) — what tool-capable models, including hosted
//! `*-cloud` models, want. Pass `--structured` to enable
//! `ModelParams::structured_tool_calls` (schema-constrained decoding) for small
//! local models that otherwise leak `<|python_tag|>` or malformed JSON across
//! the plan and execute phases.
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

use spore_core::harness::{BoxFut, ToolOutput};
use spore_core::{
    BudgetLimits, CompactionConfig, ExecutionRegistry, Harness, HarnessBuilder,
    HarnessContextManagerExt, HarnessRunOptions, HarnessStreamEvent, Hook, HookChain, HookContext,
    HookDecision, HookError, HookEvent, LoopStrategy, NullCacheProvider, OllamaModelInterface,
    PlanExecuteConfig, ReactConfig, RunResult, SandboxProvider, SchemaRef, SearchMethod, SessionId,
    StandardContextManager, StandardHookChain, StandardTool, StandardTools, Task, Tool, ToolCall,
    ToolContext, WebSearchConfig, WebSearchTool, WorkspaceConfig, WorkspaceScopedSandbox,
    PLAN_EXECUTE_EXTRAS_KEY,
};

// The GLOBAL operating prompt — the shared capability contract. It is the
// DEFAULT every leaf falls back to. Under SC-10 the plan and execute leaves
// below each carry their OWN `system_prompt`, which REPLACES this for those
// phases (each phase sees ONLY its own prompt). This global remains the prompt
// any leaf WITHOUT an override would use.
const SYSTEM_PROMPT: &str = "You are a research-and-writing agent. Your ONLY capabilities are: \
     web_search (find current information online), read_file, and write_file (save your work to \
     the workspace). You have NO shell or terminal — you cannot install software, set up projects \
     or environments, run/compile/build code, or execute commands. Act using tools — do not answer \
     from memory alone.";

// SC-10: the PLAN phase's own system prompt. The planner only DECOMPOSES — it
// never executes a subtask, so its prompt is about producing a good plan, not
// about searching/writing. This replaces SYSTEM_PROMPT for the plan leaf only.
const PLAN_SYSTEM_PROMPT: &str = "You are the PLANNER. Your ONLY job is to decompose the goal \
     into an ordered list of subtasks. Each subtask must be achievable with web_search and \
     write_file alone — there is NO shell or terminal, so never plan setup, installation, or \
     build steps. Do not perform any subtask yourself; output ONLY the plan.";

// SC-10: the EXECUTE phase's own system prompt. The executor works ONE subtask
// at a time — it does not re-plan. This replaces SYSTEM_PROMPT for the execute
// leaf only, so plan-phase decomposition guidance never leaks into execution.
const EXECUTE_SYSTEM_PROMPT: &str = "You are the EXECUTOR. You are given ONE subtask at a time. \
     Use web_search to gather current information for it, then synthesize a clear, cited result \
     and save your work with write_file. Do not re-plan or invent new subtasks — complete the one \
     you were given, using tools rather than memory.";

/// The plan-phase output contract (`plan-schema`). Post-#119, `PlanExecute`'s
/// `plan` slot is STRUCTURED: a bare `ReAct` there must declare an `output`
/// schema so the slot yields a typed task graph (`ExecutionRegistry::validate`
/// enforces this via `check_structured_slot`). This is the strict JSON the
/// planner turn returns.
fn plan_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "Ordered subtasks to execute in sequence.",
                "items": { "type": "string" }
            },
            "rationale": { "type": "string" }
        },
        "required": ["tasks"]
    })
}

/// The registry the composed strategy's handles resolve against. The single
/// `plan-schema` slot is the only EXPLICIT handle; the builder default-fills the
/// empty agent/toolset handles (`ReactConfig::per_loop` uses empty handles) from
/// the harness's own model and global tool catalogue at `build`.
fn build_registry() -> ExecutionRegistry {
    ExecutionRegistry::builder()
        .schema("plan-schema", plan_schema())
        .build()
}

/// The post-#119 composed strategy: `PlanExecute(plan: ReAct, execute: ReAct)`.
/// The plan leaf carries the `plan-schema` output contract (required for the
/// structured `plan` slot); both leaves use empty agent/toolset handles that the
/// builder default-fills. Old flat shape was `PlanExecute { plan_model: None }`.
///
/// SC-10 (per-leaf system prompt): the plan and execute leaves each carry their
/// OWN `system_prompt`. The plan phase runs under `PLAN_SYSTEM_PROMPT` (decompose
/// only) and the execute phase under `EXECUTE_SYSTEM_PROMPT` (do one subtask) —
/// each phase sees ONLY its own prompt, so planning guidance never leaks into
/// execution and vice versa. (The per-leaf TOOLSET override is the existing
/// `ReactConfig::toolset` handle; here both phases share the global catalogue.)
fn plan_execute_strategy() -> LoopStrategy {
    let plan = ReactConfig {
        output: Some(SchemaRef("plan-schema".to_string())),
        system_prompt: Some(PLAN_SYSTEM_PROMPT.to_string()),
        ..ReactConfig::per_loop(u32::MAX)
    };
    let execute = ReactConfig {
        system_prompt: Some(EXECUTE_SYSTEM_PROMPT.to_string()),
        ..ReactConfig::per_loop(u32::MAX)
    };
    LoopStrategy::PlanExecute(PlanExecuteConfig {
        plan: Box::new(LoopStrategy::ReAct(plan)),
        execute: Box::new(LoopStrategy::ReAct(execute)),
        ..PlanExecuteConfig::simple(None)
    })
}

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

/// A thin wrapper around the built-in [`WebSearchTool`] that distills its output.
///
/// WHY: the core `web_search` tool returns the SearXNG JSON body VERBATIM (by
/// frozen spec — normalization is out of scope for the core tool). Each search
/// yields ~25-30 results, each carrying the full `content` plus a dozen noise
/// fields (`thumbnail`, `engine`, `score`, `parsed_url`, …) — roughly 25-40K
/// tokens per call. Under `PlanExecute` those dumps are retained across every
/// plan step, so three searches alone can overflow a ~128K-window model. This
/// wrapper keeps only the top results and the fields the agent actually reads,
/// so the conversation context stays small. The model still sees an identical
/// `web_search` tool (same name + schema); only the *result* is trimmed.
struct ConciseWebSearch {
    inner: WebSearchTool,
}

impl Tool for ConciseWebSearch {
    fn name(&self) -> &str {
        "web_search"
    }

    fn may_produce_large_output(&self) -> bool {
        true
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        sandbox: &'a (dyn SandboxProvider + 'a),
        ctx: &'a ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let out = self.inner.execute(call, sandbox, ctx).await;
            match out {
                ToolOutput::Success { content, truncated } => ToolOutput::Success {
                    content: distill_search_results(&content),
                    truncated,
                },
                // Errors and every non-Success variant pass through untouched.
                other => other,
            }
        })
    }
}

/// Keep only the top 6 results, and for each only `title` / `url` / `content`
/// (content clipped to ~500 chars). Drop all other fields and top-level keys
/// (`answers`, `infoboxes`, `suggestions`, `unresponsive_engines`, …), and
/// re-serialize as compact `{"results":[...]}`. Defensive: if the body is not
/// JSON or has no `results` array, the original string is returned unchanged —
/// we never error just because the shape was unexpected.
fn distill_search_results(content: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(content) else {
        return content.to_string();
    };
    let Some(results) = value.get("results").and_then(|r| r.as_array()) else {
        return content.to_string();
    };

    let distilled: Vec<serde_json::Value> = results
        .iter()
        .take(6)
        .map(|r| {
            let title = r.get("title").and_then(|v| v.as_str()).unwrap_or_default();
            let url = r.get("url").and_then(|v| v.as_str()).unwrap_or_default();
            let body = r
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let clipped: String = body.chars().take(500).collect();
            serde_json::json!({ "title": title, "url": url, "content": clipped })
        })
        .collect();

    serde_json::json!({ "results": distilled }).to_string()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());
    // Opt-in constrained decoding. OFF by default: tool-capable models (incl.
    // `*-cloud`) use native Ollama tool calling, which gives `write_file` a real
    // typed schema and no always-on `final` escape. Small local models that leak
    // `<|python_tag|>` can pass `--structured` to force the JSON-object channel.
    let structured = args.iter().any(|a| a == "--structured");

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
    // Wrap the raw `WebSearchTool` in `ConciseWebSearch` so verbose SearXNG JSON
    // is distilled before it enters the conversation (see the struct doc above).
    // Same name + schema, so the model sees an identical `web_search` tool.
    let web_search = StandardTool::new(
        Box::new(ConciseWebSearch {
            inner: web_search_tool,
        }),
        WebSearchTool::schema(),
    );

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
    let model = OllamaModelInterface::with_base_url(&model_id, base_url.clone());
    let sandbox = WorkspaceScopedSandbox::new(WorkspaceConfig::scoped(workspace_root.clone()))?;

    // Lower the compaction threshold so it fires at ~0.45 * 200K ≈ 90K tokens,
    // BEFORE a ~128K-window model (e.g. gemma4:e4b) overflows. `conversational`
    // installs a `StandardContextManager` with `CompactionConfig::default()`
    // (compaction at 80% of a 200K window ≈ 160K), which is too late here. The
    // context manager gets its OWN model instance for summarization turns.
    let cm = Arc::new(StandardContextManager::new(
        Arc::new(OllamaModelInterface::with_base_url(&model_id, base_url)),
        Arc::new(NullCacheProvider),
        CompactionConfig {
            threshold: 0.45,
            ..CompactionConfig::default()
        },
    ))
    .into_harness_adapter();

    let harness = HarnessBuilder::conversational(model)
        .sandbox(Arc::new(sandbox))
        .context_manager(cm)
        .registry(build_registry())
        .tool(web_search)
        .tool(StandardTools::write_file())
        .tool(StandardTools::read_file())
        .system_prompt(SYSTEM_PROMPT)
        // Native tool calling by default; `--structured` flips on constrained
        // decoding for small models (see the `structured` flag above). With
        // structured mode the "think · turn N" line is just a turn marker, not
        // model chatter, since each turn emits one clean JSON tool call.
        .model_params(spore_core::ModelParams {
            structured_tool_calls: structured,
            ..Default::default()
        })
        .hooks(Arc::new(chain))
        .build();

    // THE STRATEGY SWAP. 06 used a bare `LoopStrategy::ReAct(ReactConfig::per_loop(10))`;
    // here we decompose first via the composed `PlanExecute(plan: ReAct, execute:
    // ReAct)` tree (post-#119 recursive enum). The plan leaf carries the
    // `plan-schema` output contract — `PlanExecute`'s `plan` slot is STRUCTURED,
    // so a bare `ReAct` there MUST declare an output schema. The empty agent /
    // toolset handles on both leaves are default-filled by the builder from the
    // harness's own model + global tool catalogue. The turn budget is divided
    // across subtasks, so we give it generous headroom.
    let task = Task::new(prompt.clone(), SessionId::generate(), plan_execute_strategy())
        .with_budget(BudgetLimits {
            max_turns: Some(64),
            ..BudgetLimits::default()
        });

    // Print each turn (Think) and each tool call + result (Act / Observe). This is
    // most useful for the plan-phase turn; Rust suppresses the subtask sub-loop
    // stream, so the hooks above are the portable view of execution.
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

// ============================================================================
// Example-crate test (NO model): the composed PlanExecute strategy resolves
// against the example's registry. This is the regression guard that the
// post-#119 strategy tree stays validation-clean against current core.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use spore_core::{AgentId, EmptyToolRegistry, ModelAgent};

    /// AC: the composed `PlanExecute(plan: ReAct{plan-schema}, execute: ReAct)`
    /// tree passes `ExecutionRegistry::validate` — the plan slot's output schema
    /// resolves and the structured-slot contract is satisfied. The leaves use the
    /// EMPTY agent/toolset handles that `HarnessBuilder::build_config` default-fills
    /// at `build`; here we mirror that fill (empty-key agent + toolset) so the
    /// standalone registry validates exactly as the assembled harness would.
    #[test]
    fn registry_validates() {
        let model = Arc::new(OllamaModelInterface::with_base_url(
            "gemma4:e4b",
            "http://localhost:11434".to_string(),
        ));
        let registry = build_registry()
            .into_builder()
            .fill_default_agent(Arc::new(ModelAgent::new(AgentId::new("default"), model)))
            .fill_default_toolset(Arc::new(EmptyToolRegistry))
            .build();
        let task = Task::new(
            "decompose and execute".to_string(),
            SessionId::generate(),
            plan_execute_strategy(),
        );
        assert!(
            registry.validate(&task).is_ok(),
            "the composed PlanExecute strategy must validate against the registry"
        );
    }
}
