//! spore-core example 11 — multi-agent composition.
//!
//! **The thesis: agents are composable.** The harness does not care whether a
//! "tool" dispatches to a function or to *another agent*. This example builds
//! three agents and wires two of them into the third as ordinary tools.
//!
//! ## The three agents
//!
//! - **research worker** — a [`StandardHarness`] with exactly one tool,
//!   `web_search` (the SearXNG-backed [`WebSearchTool`] from example 06). Given
//!   an instruction string, it searches the web and returns raw, cited findings
//!   as its output string.
//! - **writing worker** — a [`StandardHarness`] with NO tools. Given the
//!   research findings as its instruction, it formats them into a polished
//!   markdown report and returns that markdown as its output string. It never
//!   touches the network — it only shapes prose.
//! - **orchestrator** — a [`StandardHarness`] whose "tools" are the two workers
//!   (wrapped as [`SubagentTool`]s) plus `write_file`. It plans the job, calls
//!   `research_worker`, hands that output to `writing_worker`, then writes the
//!   final markdown to `workspace/report.md`.
//!
//! Three agents, two handoffs (`research → writing`, `writing → report.md`),
//! one output.
//!
//! ## The agent-as-tool mechanism
//!
//! Each worker is a fully-built child [`Harness`] wrapped in a [`SubagentTool`].
//! `SubagentTool` implements [`Tool`](spore_core::tool_registry::Tool): when the
//! orchestrator emits a `ToolCall` for `research_worker`, the tool reads a single
//! `instruction` string from the call, runs the child harness on a fresh
//! [`Task`], and returns the child's final output string as the tool result. The
//! orchestrator cannot tell — and does not need to know — that the "tool" behind
//! `research_worker` is an entire agent with its own loop, its own model, and its
//! own web-search tool.
//!
//! We register each worker on the orchestrator's builder the same way example 06
//! registers `web_search`: build a [`StandardTool`] from the boxed `SubagentTool`
//! plus a [`RegisteredToolSchema`] advertising the `{ instruction: string }`
//! input, then `.tool(...)` it.
//!
//! ## Why this keeps the orchestrator's context clean
//!
//! Both workers use [`ContextSharing::Isolated`]: each runs in a brand-new
//! session with NO shared mutable state with the orchestrator or with each other.
//! The research worker may burn a dozen internal turns issuing search queries and
//! sifting noisy JSON — but the ONLY thing that crosses back into the
//! orchestrator's context is the worker's final output string. The orchestrator
//! never sees the worker's intermediate turns, failed searches, or raw result
//! blobs. The noisy work is encapsulated; the orchestrator's context stays small
//! and on-topic. This is the whole reason to delegate to a subagent rather than
//! inline the work.
//!
//! A direct, visible consequence: the child's internal turns do **not** stream up
//! through the parent. The orchestrator's stream only shows the `ToolCall` to
//! `research_worker` and the `ToolResult` coming back — which is exactly the
//! agent boundary we print. The invisibility of the child's turns is not a
//! limitation; it *is* the context isolation, made observable.
//!
//! ## The strategy split: PlanExecute at the top, ReAct inside
//!
//! The orchestrator runs a composed [`LoopStrategy::PlanExecute`] (post-#119,
//! `PlanExecute(plan: ReAct, execute: ReAct)`): it decomposes the job
//! ("research, then write, then save") into subtasks up front and executes them
//! in order — natural for a coordinator. The plan leaf carries a `plan-schema`
//! output contract — `PlanExecute`'s `plan` slot is STRUCTURED, so a bare `ReAct`
//! there MUST declare an output schema (`ExecutionRegistry::validate` enforces
//! this at run entry). Each worker, by contrast, runs ReAct internally. (The
//! ReAct loop is hardcoded inside `SubagentTool`; a subagent always runs its
//! child as `ReAct`.) So the two layers use two different loop strategies, each
//! fit to its level: deliberate planning at the orchestrator, step-by-step tool
//! use inside the workers.
//!
//! ## Agent boundaries in stdout
//!
//! The point of this example is *legibility*: you should be able to read stdout
//! and see which agent is acting, what it received, and what it returned. The
//! orchestrator's stream fires a `ToolCall{name, args}` and a `ToolResult` for
//! each worker dispatch — we turn those into a boxed banner:
//!
//! ```text
//! ┌─ orchestrator → research_worker
//! │  received: <instruction>
//! └─ research_worker → orchestrator
//!    returned: <truncated findings>
//! ```
//!
//! The orchestrator's **plan itself** is surfaced separately, through the hook
//! chain (there are no plan/subtask stream events). A `PlanExecuteReporter`
//! (`impl Hook`, copied from example 08) prints a `── orchestrator plan ──`
//! banner on `OnPlanCreated` and an `[i/N]` line per subtask on `OnTaskAdvance`,
//! interleaved with the boundary banners above — so "the orchestrator PLANS,
//! then delegates" is legible end to end.
//!
//! There are no `// SPEC QUESTION:` markers: the single-file layout, the
//! three-agent shape, the isolated context sharing, the PlanExecute/ReAct split,
//! and the final-write owner were all resolved before this example was written.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull llama3.2
//! export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"   # SearXNG JSON API
//! cargo run
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use spore_core::harness::BoxFut;
use spore_core::tool_registry::StandardToolRegistry;
use spore_core::tools::{ContextSharing, SubagentTool};
use spore_core::{
    BudgetLimits, ExecutionRegistry, Harness, HarnessBuilder, HarnessRunOptions, HarnessStreamEvent,
    Hook, HookChain, HookContext, HookDecision, HookError, HookEvent, LoopStrategy,
    OllamaModelInterface, PlanExecuteConfig, ReactConfig, RegisteredToolSchema, RunResult, SchemaRef,
    SearchMethod, SessionId, StandardHookChain, StandardTool, StandardTools, Task, ToolAnnotations,
    WebSearchConfig, WebSearchTool, WorkspaceConfig, WorkspaceScopedSandbox,
};

/// Per-worker wall-clock cap. A worker can burn many internal ReAct turns; this
/// bounds how long the orchestrator will wait on any single delegation.
const WORKER_TIMEOUT: Duration = Duration::from_secs(180);

const RESEARCH_PROMPT: &str = "You are a research worker. Use the web_search tool to gather \
     current, factual information on the topic you are given. Issue focused queries, read the \
     results, and return a concise set of findings as plain text — key facts, figures, and \
     definitions — each followed by the source URL it came from. Do NOT format a report; just \
     return the raw, cited findings. Act using web_search — do not answer from memory alone.";

const WRITING_PROMPT: &str = "You are a writing worker. You will be given a set of raw, cited \
     research findings. Turn them into a polished markdown report: a top-level `# ` title, a short \
     intro, well-organized `## ` sections, and a `## Sources` list preserving the URLs from the \
     findings. Return ONLY the markdown of the report — no preamble, no commentary. You have no \
     tools; produce the report directly as your final answer.";

const ORCHESTRATOR_PROMPT: &str = "You are an orchestrator. You coordinate two worker agents, \
     each exposed to you as a tool. Your plan is always the same three steps: (1) call \
     `research_worker` with an `instruction` describing the topic to research; (2) call \
     `writing_worker` with an `instruction` that is the EXACT findings text returned by the \
     research worker, asking it to format a polished markdown report; (3) call `write_file` to \
     save the writing worker's markdown verbatim to `report.md`. Do the research and writing by \
     delegating to the workers — never do it yourself — and always finish by writing report.md.";

/// The orchestrator plan-phase output contract (`plan-schema`). Post-#119,
/// `PlanExecute`'s `plan` slot is STRUCTURED: a bare `ReAct` there must declare
/// an `output` schema so the slot yields a typed task graph
/// (`ExecutionRegistry::validate` enforces this via `check_structured_slot`).
fn plan_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "Ordered subtasks the orchestrator delegates.",
                "items": { "type": "string" }
            },
            "rationale": { "type": "string" }
        },
        "required": ["tasks"]
    })
}

/// The registry the orchestrator's composed strategy resolves against. Only the
/// `plan-schema` handle is EXPLICIT; the builder default-fills the empty
/// agent/toolset handles (`ReactConfig::per_loop`) from the orchestrator's own
/// model + global tool catalogue (the worker-tools) at `build`.
fn build_registry() -> ExecutionRegistry {
    ExecutionRegistry::builder()
        .schema("plan-schema", plan_schema())
        .build()
}

/// The post-#119 composed orchestrator strategy:
/// `PlanExecute(plan: ReAct{plan-schema}, execute: ReAct)`. The plan leaf carries
/// the `plan-schema` output contract required for the structured `plan` slot.
/// Old flat shape was `PlanExecute { plan_model: None }`.
fn plan_execute_strategy() -> LoopStrategy {
    let plan = ReactConfig {
        output: Some(SchemaRef("plan-schema".to_string())),
        ..ReactConfig::per_loop(u32::MAX)
    };
    LoopStrategy::PlanExecute(PlanExecuteConfig {
        plan: Box::new(LoopStrategy::ReAct(plan)),
        ..PlanExecuteConfig::simple(None)
    })
}

/// Lifecycle hook that surfaces the ORCHESTRATOR's PlanExecute plan and each
/// subtask as it advances. This is the "the orchestrator PLANS, then delegates"
/// half of the example made visible — the worker hand-offs print via the stream
/// (the boxed `┌─ … └─` banners); the plan prints here.
///
/// The plan is NOT a stream event — it travels through the hook chain, exactly as
/// in example 08:
/// - `OnPlanCreated` fires once, post-capture / pre-execute — we print a
///   `── orchestrator plan ──` banner: the rationale, then the numbered steps.
/// - `OnTaskAdvance` fires before each subtask — we print `[i/N] <instruction>`.
///
/// This hook only observes; it always returns [`HookDecision::Continue`].
struct PlanExecuteReporter;

impl Hook for PlanExecuteReporter {
    fn handle<'a>(
        &'a self,
        ctx: &'a mut HookContext<'a>,
    ) -> BoxFut<'a, Result<HookDecision, HookError>> {
        match ctx {
            HookContext::OnPlanCreated { plan, .. } => {
                println!("\n── orchestrator plan ──");
                if !plan.rationale.trim().is_empty() {
                    println!("rationale: {}", plan.rationale);
                }
                for (i, task) in plan.tasks.iter().enumerate() {
                    println!("  {}. {task}", i + 1);
                }
                println!("───────────────────────\n");
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

/// The single-parameter input schema every worker tool advertises: the
/// orchestrator passes one `instruction` string, which `SubagentTool` forwards to
/// the child harness as its task. Matches the schema `SubagentTool` reads.
fn instruction_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "The full instruction / task for the worker agent."
            }
        },
        "required": ["instruction"]
    })
}

/// Build the SearXNG-backed `web_search` catalogue tool (identical wiring to
/// example 06). Only the research worker gets this.
fn build_web_search(endpoint: &str) -> StandardTool {
    let tool = WebSearchTool::with_config(WebSearchConfig {
        endpoint: endpoint.to_string(),
        method: SearchMethod::Get,
        query_param: "q".into(),
        auth_headers: Vec::new(),
        body_auth_params: Vec::new(),
    })
    .expect("web_search backend config is valid (SearXNG needs no auth env vars)");
    StandardTool::new(Box::new(tool), WebSearchTool::schema())
}

/// Build the research worker: a child harness whose only tool is `web_search`.
/// Returned as `Arc<dyn Harness>` so it can be embedded in a `SubagentTool`.
fn build_research_harness(model_id: &str, base_url: &str, endpoint: &str) -> Arc<dyn Harness> {
    // Each agent gets its OWN fresh model instance — the workers are genuinely
    // independent and do not share a model object with the orchestrator.
    let model = OllamaModelInterface::with_base_url(model_id, base_url.to_string());
    let harness = HarnessBuilder::conversational(model)
        .tool(build_web_search(endpoint))
        .system_prompt(RESEARCH_PROMPT)
        .build();
    Arc::new(harness)
}

/// Build the writing worker: a child harness with NO tools — it formats prose and
/// returns the report as its final answer.
fn build_writing_harness(model_id: &str, base_url: &str) -> Arc<dyn Harness> {
    let model = OllamaModelInterface::with_base_url(model_id, base_url.to_string());
    let harness = HarnessBuilder::conversational(model)
        .system_prompt(WRITING_PROMPT)
        .build();
    Arc::new(harness)
}

/// Wrap a child harness as a `SubagentTool` and bundle it into a `StandardTool`
/// the orchestrator can register — exactly how example 06 wraps `web_search`,
/// only the "implementation" here is an entire agent.
fn build_worker_tool(name: &str, description: &str, child: Arc<dyn Harness>) -> StandardTool {
    // `child_registry` is used ONLY for the depth-1 `has_subagent_tools()` check.
    // The workers have no subagent tools of their own, so a fresh empty registry
    // passes trivially. The child's REAL tools were wired on its builder above.
    let empty_child_registry = StandardToolRegistry::new();
    let subagent = SubagentTool::new(
        name,
        description,
        instruction_schema(),
        WORKER_TIMEOUT,
        ContextSharing::Isolated,
        child,
        &empty_child_registry,
    )
    .expect("worker child has no subagent tools, so the depth-1 rule is satisfied");

    StandardTool::new(
        Box::new(subagent),
        RegisteredToolSchema {
            name: name.into(),
            description: description.into(),
            parameters: instruction_schema(),
            // `open_world`: a subagent reaches outside the process (it runs a whole
            // agent, and the research worker hits the network), so it is not a
            // closed, read-only computation.
            annotations: ToolAnnotations {
                open_world: true,
                ..Default::default()
            },
        },
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    // Same required search backend as example 06 — only the research worker uses
    // it, but the orchestrator cannot do its job without it, so we fail fast here.
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

    // The orchestrator operates inside this example's `workspace/` directory; it
    // writes the final report there. Resolve relative to this source file and
    // canonicalize — the sandbox requires a canonical, existing root.
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("workspace");
    std::fs::create_dir_all(&workspace_root)?;
    let workspace_root = std::fs::canonicalize(&workspace_root)?;

    let topic = arg_value(&args, "--topic").unwrap_or_else(|| {
        // A TIMELESS, encyclopedic subject so web-search results stay stable and
        // useful across runs (per the issue: keep the topic generic).
        "the history and core ideas of the Rust programming language".to_string()
    });
    let prompt = format!(
        "Research {topic} and produce a polished markdown report saved to report.md. \
         Delegate the research to research_worker and the writing to writing_worker."
    );

    // ---- Build the two workers, then wrap them as orchestrator tools --------
    let research_child = build_research_harness(&model_id, &base_url, &endpoint);
    let writing_child = build_writing_harness(&model_id, &base_url);

    let research_tool = build_worker_tool(
        "research_worker",
        "Delegate to the research agent: pass an `instruction` describing a topic; it web-searches \
         and returns concise, cited findings as text.",
        research_child,
    );
    let writing_tool = build_worker_tool(
        "writing_worker",
        "Delegate to the writing agent: pass an `instruction` containing research findings; it \
         returns a polished markdown report.",
        writing_child,
    );

    // Register the plan reporter on a hook chain. The chain is how the
    // orchestrator's plan becomes visible: there are no plan/subtask stream
    // events (identical mechanism to example 08).
    let chain = StandardHookChain::new();
    chain.register(Arc::new(PlanExecuteReporter))?;

    // ---- Build the orchestrator: workers-as-tools + write_file --------------
    let orchestrator_model = OllamaModelInterface::with_base_url(&model_id, base_url.clone());
    let sandbox = WorkspaceScopedSandbox::new(WorkspaceConfig::scoped(workspace_root.clone()))?;
    let orchestrator = HarnessBuilder::conversational(orchestrator_model)
        .sandbox(Arc::new(sandbox))
        .registry(build_registry())
        .tool(research_tool)
        .tool(writing_tool)
        .tool(StandardTools::write_file())
        .system_prompt(ORCHESTRATOR_PROMPT)
        .hooks(Arc::new(chain))
        .build();

    // The orchestrator plans the three steps up front via PlanExecute, then
    // executes them. The turn budget is divided across subtasks, so give it
    // generous headroom — each worker dispatch may itself be slow.
    let task = Task::new(prompt.clone(), SessionId::generate(), plan_execute_strategy())
        .with_budget(BudgetLimits {
            max_turns: Some(32),
            ..BudgetLimits::default()
        });

    // The orchestrator's stream is where the agent boundaries become visible.
    // A `ToolCall` to a worker IS the "→ worker" boundary; the matching
    // `ToolResult` IS the "← worker" boundary. The child's own internal turns do
    // NOT appear here — that invisibility is the context isolation, made
    // observable (see the module docs).
    //
    // `ToolResult` carries only `call_id` (no tool name), so we remember which
    // `call_id` belonged to which tool when the `ToolCall` fires, then look it up
    // on the result to label the closing half of the boundary.
    let call_names: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
    let options =
        HarnessRunOptions::new(task).with_stream(Box::new(move |event: HarnessStreamEvent| {
            match event {
                HarnessStreamEvent::TurnStart { turn, .. } => {
                    println!("orchestrator · plan/execute turn {turn}");
                }
                HarnessStreamEvent::ToolCall {
                    call_id,
                    name,
                    args,
                    ..
                } => {
                    call_names.lock().unwrap().insert(call_id, name.clone());
                    if is_worker(&name) {
                        let instruction = args
                            .get("instruction")
                            .and_then(|v| v.as_str())
                            .unwrap_or("<no instruction>");
                        println!("┌─ orchestrator → {name}");
                        println!("│  received: {}", truncate(instruction, 200));
                    } else {
                        println!(
                            "  orchestrator → {name}({})",
                            truncate(&args.to_string(), 160)
                        );
                    }
                }
                HarnessStreamEvent::ToolResult {
                    call_id,
                    is_error,
                    content,
                    ..
                } => {
                    let name = call_names
                        .lock()
                        .unwrap()
                        .remove(&call_id)
                        .unwrap_or_else(|| "<tool>".to_string());
                    if is_worker(&name) {
                        let tag = if is_error { "FAILED" } else { "returned" };
                        println!("└─ {name} → orchestrator");
                        println!("   {tag}: {}", truncate(&content, 280));
                    } else {
                        let tag = if is_error { "err" } else { "ok" };
                        println!(
                            "  {name} → orchestrator [{tag}]: {}",
                            truncate(&content, 160)
                        );
                    }
                }
                _ => {}
            }
        }));

    println!("model       : {model_id}");
    println!("endpoint    : {endpoint}");
    println!("workspace   : {}", workspace_root.display());
    println!("strategy    : orchestrator=PlanExecute, workers=ReAct (isolated)");
    println!("agents      : orchestrator → [research_worker, writing_worker]");
    println!("topic       : {topic}\n");

    match orchestrator.run(options).await {
        RunResult::Success { output, turns, .. } => {
            println!(
                "\norchestrator done ({turns} turn(s)): {}",
                truncate(&output, 280)
            );
            let report = workspace_root.join("report.md");
            if report.exists() {
                println!("\nreport.md now exists on disk: {}", report.display());
            } else {
                eprintln!("\nwarning: orchestrator finished but report.md was not written.");
            }
            Ok(())
        }
        other => {
            eprintln!("\nrun did not succeed: {other:?}");
            std::process::exit(1);
        }
    }
}

/// A tool name that maps to one of the two worker agents (vs. `write_file`).
fn is_worker(name: &str) -> bool {
    name == "research_worker" || name == "writing_worker"
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Keep boundary lines readable — findings and reports can be long.
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
// Example-crate test (NO model): the orchestrator's composed PlanExecute
// strategy resolves against the example's registry — the regression guard that
// the post-#119 strategy tree stays validation-clean against current core.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use spore_core::{AgentId, EmptyToolRegistry, ModelAgent};

    /// AC: the composed `PlanExecute(plan: ReAct{plan-schema}, execute: ReAct)`
    /// tree validates — the plan slot's output schema resolves and the
    /// structured-slot contract is satisfied. The leaves use the EMPTY
    /// agent/toolset handles that `HarnessBuilder::build_config` default-fills at
    /// `build`; here we mirror that fill so the standalone registry validates
    /// exactly as the assembled orchestrator harness would.
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
            "research, write, save".to_string(),
            SessionId::generate(),
            plan_execute_strategy(),
        );
        assert!(
            registry.validate(&task).is_ok(),
            "the composed PlanExecute strategy must validate against the registry"
        );
    }
}
