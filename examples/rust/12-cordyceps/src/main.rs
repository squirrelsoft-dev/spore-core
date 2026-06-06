//! spore-core example 12 — **cordyceps**: a fully autonomous task-completion
//! agent (the capstone).
//!
//! **The thesis: you give it a task; it does not stop until the job is done —
//! and when a worker gets stuck or uncertain, it asks for help (a sibling
//! helper, then a human) rather than giving up.**
//!
//! This example composes everything the suite has built — subagents-as-tools
//! (11), custom sandboxed tools (05), `web_search` (06), `memory` (07),
//! `task_list` — plus two new capabilities:
//!
//! - **Architect-side skill loading** (see [`skills`]): a `load_skill` tool
//!   activates the bundled `audit` skill at runtime via a `GuideRegistry`, and a
//!   custom context manager re-injects the skill body every turn
//!   (compaction-proof). This is the pattern issue #115 will absorb into the
//!   harness; the live loop's structural skill-injection path is not wired yet
//!   (see the README and #115).
//! - **A generalized consult / escalation ladder** (issue #114): the analysis
//!   worker escalates mid-loop to a research helper (`kind=research`, budget 5,
//!   soft-fail) and then to a cloud-model advisor (`kind=advice`, budget 3,
//!   escalate-to-human), resuming each time without ending its run.
//!
//! ## Topology (depth-1)
//!
//! ```text
//! orchestrator (ReAct, gemma4:e4b)
//!   tools: list_dir, grep, task_list, memory, write_file, bash_command,
//!          analysis_worker (SubagentTool, with consult handlers)
//!   ├── analysis_worker (Isolated) — audits ONE module
//!   │     tools: read_file, grep, research_best_practices, consult_advisor,
//!   │            load_skill
//!   ├── research_worker (Isolated) — web_search   [consult handler: research]
//!   └── advisor         (Isolated, cloud model)   [consult handler: advice]
//! ```
//!
//! The orchestrator enumerates crates → modules, adds one `task_list` task per
//! module, dispatches the analysis worker per task, accumulates findings in
//! `memory`, finalizes the top 5, writes `workspace/findings.md`, and runs the
//! y/N issue-filing flow. The audit is READ-ONLY; the only writes are
//! `workspace/findings.md` and (approved) GitHub issues.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull gemma4:e4b
//! export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
//! cargo run
//! ```

mod skills;
mod tools;

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use spore_core::storage::InMemoryStorageProvider;
use spore_core::tool_registry::StandardToolRegistry;
use spore_core::tools::{ContextSharing, SubagentTool};
use spore_core::{
    BudgetLimits, ConsultHandlerEntry, ConsultOverflowPolicy, Harness, HarnessBuilder,
    HarnessContextManager, HarnessContextManagerExt, HarnessRunOptions, HarnessStreamEvent,
    HumanRequest, HumanResponse, LoopStrategy, NullCacheProvider, OllamaModelInterface,
    RegisteredToolSchema, RunResult, SearchMethod, SessionId, StandardContextManager, StandardTool,
    StandardTools, StorageProvider, Task, ToolAnnotations, WebSearchConfig, WebSearchTool,
    WorkspaceConfig, WorkspaceScopedSandbox,
};

use crate::skills::{SkillCatalog, SkillInjectingContextManager};
use crate::tools::consult::{
    consult_advisor_tool, research_best_practices_tool, KIND_ADVICE, KIND_RESEARCH,
};
use crate::tools::load_skill::load_skill_tool;

/// The pre-filled audit prompt the user presses enter to accept (from the
/// issue). An empty line at the REPL ⇒ this verbatim.
const DEFAULT_AUDIT_PROMPT: &str =
    "Audit the current repo for the rust language. Work sequentially by identifying each crate, \
     and each module in the crate, and adding a task to the tasklist for a subagent to do the \
     deep dive audit on.";

/// Per-worker wall-clock cap. A worker can burn many internal ReAct turns (and
/// mediated consults); this bounds how long the orchestrator waits on one
/// delegation.
const WORKER_TIMEOUT: Duration = Duration::from_secs(300);

/// Bundled `audit` skill, embedded so the example is self-contained even with
/// an empty `.spore/skills/`.
const BUNDLED_AUDIT_SKILL: &str = include_str!("../skills/audit/SKILL.md");

const ORCHESTRATOR_PROMPT: &str = "You are the cordyceps orchestrator: an autonomous Rust-repo \
     auditor. You do not stop until the audit is complete. Work sequentially: (1) use `list_dir` \
     to enumerate the crates under `rust/crates/`, then the modules (`src/*.rs`) in each crate; \
     (2) for each module, add ONE task to the task list (`task_list`) describing the module to \
     audit; (3) for each task, call `analysis_worker` with an `instruction` naming the ONE module \
     to deep-dive audit; (4) accumulate the findings each worker returns into `memory` under a \
     stable key; (5) when every module is audited, pick the TOP 5 most important findings across \
     all modules and write them as a markdown report to `workspace/findings.md` using \
     `write_file`. The audit is READ-ONLY — never modify source files. Delegate the per-module \
     deep dives to `analysis_worker`; do not audit modules yourself. Finish by writing \
     findings.md.";

const ANALYSIS_WORKER_PROMPT: &str = "You are an analysis worker: you deep-dive audit exactly ONE \
     Rust module for real, actionable defects. BEFORE auditing, call `load_skill` with \
     `skill_id` = \"audit\" and follow the returned procedure and findings schema EXACTLY. Stay \
     inside the one module you were given. Grep first, read only narrow line ranges, and escalate \
     with `research_best_practices` (idiom questions) or `consult_advisor` (severity / is-this-real \
     questions) when genuinely unsure. Your FINAL answer must be a JSON array of \
     {file, line, severity, description} objects — and nothing else.";

const RESEARCH_PROMPT: &str = "You are a research worker. Use the web_search tool to gather \
     current, factual information on the Rust best-practice or idiom question you are given. Issue \
     focused queries, read the results, and return a concise, cited answer as plain text. Act \
     using web_search — do not answer from memory alone.";

const ADVISOR_PROMPT: &str = "You are a senior Rust advisor. A worker is stuck on whether a \
     finding is a real defect, or on how to rank its severity. Use `read_file` and `grep` to \
     investigate the specific code in question, then give a crisp, decisive recommendation: is it \
     a real defect, what severity (low/medium/high/critical), and why. Be concrete and brief.";

/// The single-parameter input schema every subagent tool advertises.
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

/// Build the SearXNG-backed `web_search` catalogue tool (identical to 06/11).
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

/// Build the inner standard compaction adapter (the same one
/// `HarnessBuilder::conversational` installs), so our skill-injecting context
/// manager can wrap it and delegate every non-`assemble` method to it.
fn build_inner_context_manager(model_id: &str, base_url: &str) -> Arc<dyn HarnessContextManager> {
    let model = Arc::new(OllamaModelInterface::with_base_url(
        model_id,
        base_url.to_string(),
    ));
    Arc::new(StandardContextManager::new(
        model,
        Arc::new(NullCacheProvider),
        spore_core::CompactionConfig::default(),
    ))
    .into_harness_adapter()
}

/// Build the research worker harness (web_search only). Used as the
/// `kind=research` consult handler.
fn build_research_harness(model_id: &str, base_url: &str, endpoint: &str) -> Arc<dyn Harness> {
    let model = OllamaModelInterface::with_base_url(model_id, base_url.to_string());
    Arc::new(
        HarnessBuilder::conversational(model)
            .tool(build_web_search(endpoint))
            .system_prompt(RESEARCH_PROMPT)
            .build(),
    )
}

/// Build the advisor harness (cloud model, read_file + grep). Used as the
/// `kind=advice` consult handler. Rides the same Ollama endpoint via
/// `with_base_url`, only the model id differs.
fn build_advisor_harness(
    model_id: &str,
    base_url: &str,
    repo_sandbox: Arc<WorkspaceScopedSandbox>,
) -> Arc<dyn Harness> {
    let model = OllamaModelInterface::with_base_url(model_id, base_url.to_string());
    Arc::new(
        HarnessBuilder::conversational(model)
            .sandbox(repo_sandbox)
            .tool(StandardTools::read_file())
            .tool(StandardTools::grep())
            .system_prompt(ADVISOR_PROMPT)
            .build(),
    )
}

/// Build the analysis worker harness: read_file, grep, the two consult tools,
/// and load_skill. Isolated; audits ONE module.
fn build_analysis_harness(
    model_id: &str,
    base_url: &str,
    repo_sandbox: Arc<WorkspaceScopedSandbox>,
    storage: Arc<StorageProvider>,
    catalog: &SkillCatalog,
) -> Arc<dyn Harness> {
    let model = OllamaModelInterface::with_base_url(model_id, base_url.to_string());
    // The worker shares the SAME storage as the orchestrator so `load_skill`'s
    // active-skill write and the context manager's read rendezvous within the
    // run. (Subagents run Isolated SESSIONS, but the run_store is keyed by the
    // worker's own session id, so each worker activates `audit` for itself.)
    let inner_cm = build_inner_context_manager(model_id, base_url);
    let skill_cm = Arc::new(SkillInjectingContextManager::new(
        inner_cm,
        storage.run().clone(),
        catalog.manifest(),
    ));
    Arc::new(
        HarnessBuilder::conversational(model)
            .sandbox(repo_sandbox)
            .storage(storage)
            .context_manager(skill_cm)
            .tool(StandardTools::read_file())
            .tool(StandardTools::grep())
            .tool(research_best_practices_tool())
            .tool(consult_advisor_tool())
            .tool(load_skill_tool(catalog.registry()))
            .system_prompt(ANALYSIS_WORKER_PROMPT)
            .build(),
    )
}

/// Wrap the analysis worker as a `SubagentTool` with the consult handlers
/// installed. The two handlers mediate by `kind` (research → research_worker,
/// advice → advisor) with the per-kind budgets + overflow policies from #114.
fn build_analysis_tool(
    analysis: Arc<dyn Harness>,
    consult_handlers: HashMap<String, ConsultHandlerEntry>,
) -> StandardTool {
    let empty_child_registry = StandardToolRegistry::new();
    let subagent = SubagentTool::new(
        "analysis_worker",
        "Delegate a deep-dive audit of ONE Rust module: pass an `instruction` naming the module; \
         it loads the `audit` skill, audits the module (escalating via consults when stuck), and \
         returns a JSON array of {file, line, severity, description} findings.",
        instruction_schema(),
        WORKER_TIMEOUT,
        ContextSharing::Isolated,
        analysis,
        &empty_child_registry,
    )
    .expect("analysis worker has no subagent tools (depth-1 holds)")
    .with_consult_handlers(consult_handlers);

    StandardTool::new(
        Box::new(subagent),
        RegisteredToolSchema {
            name: "analysis_worker".into(),
            description: "Deep-dive audit one module via a subagent; returns JSON findings.".into(),
            parameters: instruction_schema(),
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
        .unwrap_or_else(|| "gemma4:e4b".to_string());
    let advisor_model_id = arg_value(&args, "--advisor-model")
        .or_else(|| std::env::var("SPORE_ADVISOR_MODEL").ok())
        .unwrap_or_else(|| "minimax-m3:cloud".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    // Required search backend (research worker) — fail fast like 06/11.
    let endpoint = match arg_value(&args, "--search-url")
        .or_else(|| std::env::var("SPORE_WEB_SEARCH_ENDPOINT").ok())
    {
        Some(e) if !e.trim().is_empty() => e,
        _ => {
            eprintln!(
                "SPORE_WEB_SEARCH_ENDPOINT is not set.\n\
                 Set it to a SearXNG JSON endpoint, e.g. \
                 http://localhost:8888/search?format=json. See .env.example and the README."
            );
            std::process::exit(2);
        }
    };

    // Resolve the repo root (cwd) for the read-only audit sandbox, and this
    // example's workspace/ for the report write.
    let repo_root = std::env::current_dir()?;
    let repo_root = std::fs::canonicalize(&repo_root)?;
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("workspace");
    std::fs::create_dir_all(&workspace_root)?;

    // Seed `.spore/skills/audit/SKILL.md` from the bundled copy if absent, so a
    // user can see the filesystem-registry shape (documented in the README).
    seed_bundled_audit_skill(&repo_root);

    // One in-memory storage provider, shared by the orchestrator and the
    // analysis worker so `load_skill` (worker-side write) and the context
    // manager (read) rendezvous on `run_store["active_skills"]`.
    let storage = Arc::new(StorageProvider::single(Arc::new(
        InMemoryStorageProvider::new(),
    )));

    // Scan + register skills (bundled audit + any project/user skills).
    let catalog = SkillCatalog::bootstrap(&repo_root, BUNDLED_AUDIT_SKILL).await;

    // The orchestrator can read the whole repo + write into its own workspace.
    // A workspace-scoped sandbox rooted at the repo lets read_file/grep/list_dir
    // reach the crates, write_file land findings.md (relative to cwd workspace),
    // and bash_command run `gh`. For the audit's read-only guarantee we rely on
    // the orchestrator prompt + skill discipline, not a read-only sandbox,
    // because it must also write findings.md and (optionally) run `gh`.
    let orchestrator_sandbox = Arc::new(WorkspaceScopedSandbox::new(WorkspaceConfig::scoped(
        repo_root.clone(),
    ))?);
    // Workers and the advisor get a READ-ONLY view: same root, but their tool
    // sets (read_file/grep) never write. A dedicated sandbox keeps them scoped.
    let worker_sandbox = Arc::new(WorkspaceScopedSandbox::new(WorkspaceConfig::scoped(
        repo_root.clone(),
    ))?);

    // ---- Build the consult handlers (research + advice) ---------------------
    let research_handler = build_research_harness(&model_id, &base_url, &endpoint);
    let advisor_handler =
        build_advisor_harness(&advisor_model_id, &base_url, worker_sandbox.clone());
    let mut consult_handlers: HashMap<String, ConsultHandlerEntry> = HashMap::new();
    consult_handlers.insert(
        KIND_RESEARCH.to_string(),
        ConsultHandlerEntry {
            handler: research_handler,
            budget: 5,
            overflow: ConsultOverflowPolicy::SoftFail,
        },
    );
    consult_handlers.insert(
        KIND_ADVICE.to_string(),
        ConsultHandlerEntry {
            handler: advisor_handler.clone(),
            budget: 3,
            overflow: ConsultOverflowPolicy::EscalateToHuman,
        },
    );

    // ---- Build the analysis worker + wrap it (with consult handlers) --------
    let analysis = build_analysis_harness(
        &model_id,
        &base_url,
        worker_sandbox.clone(),
        storage.clone(),
        &catalog,
    );
    let analysis_tool = build_analysis_tool(analysis, consult_handlers);

    // ---- Build the orchestrator --------------------------------------------
    let orchestrator_model = OllamaModelInterface::with_base_url(&model_id, base_url.clone());
    let orchestrator = HarnessBuilder::conversational(orchestrator_model)
        .sandbox(orchestrator_sandbox)
        .storage(storage.clone())
        .tool(StandardTools::list_dir())
        .tool(StandardTools::grep())
        .tool(StandardTools::task_list())
        .tool(StandardTools::memory())
        .tool(StandardTools::write_file())
        .tool(StandardTools::bash_command())
        .tool(analysis_tool)
        .system_prompt(ORCHESTRATOR_PROMPT)
        .build();

    println!("model        : {model_id}");
    println!("advisor model: {advisor_model_id}");
    println!("endpoint     : {endpoint}");
    println!("repo root    : {}", repo_root.display());
    println!("workspace    : {}", workspace_root.display());
    println!(
        "skills       : {}",
        catalog
            .manifest()
            .iter()
            .map(|e| e.name.clone())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("strategy     : orchestrator=ReAct, workers=ReAct (isolated)\n");

    // ---- REPL: pre-filled prompt, enter accepts the default -----------------
    let prompt = read_audit_prompt();

    // Stream banners: mirror 11-multi-agent's `┌─ … └─` boundary style, but for
    // the analysis_worker delegation.
    let call_names: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
    let task = Task::new(
        prompt,
        SessionId::generate(),
        LoopStrategy::ReAct { max_iterations: 64 },
    )
    .with_budget(BudgetLimits {
        max_turns: Some(64),
        ..BudgetLimits::default()
    });
    let options = HarnessRunOptions::new(task).with_stream(stream_sink(call_names));

    // ---- Drive the orchestrator, handling the human-escalation ladder -------
    let mut result = orchestrator.run(options).await;
    loop {
        match result {
            RunResult::Success { output, turns, .. } => {
                println!(
                    "\norchestrator done ({turns} turn(s)): {}",
                    truncate(&output, 400)
                );
                let findings = workspace_root.join("findings.md");
                if findings.exists() {
                    println!("\nfindings.md written: {}", findings.display());
                    run_issue_filing_flow(&orchestrator, &findings).await;
                } else {
                    eprintln!("\nwarning: orchestrator finished but workspace/findings.md was not written.");
                }
                return Ok(());
            }
            RunResult::Failure { reason, turns, .. } => {
                eprintln!("\norchestrator failed after {turns} turn(s): {reason:?}");
                std::process::exit(1);
            }
            RunResult::WaitingForHuman { state, request } => {
                // The advice consult budget (3) was exhausted under
                // EscalateToHuman: the analysis_worker SubagentTool converted the
                // over-budget consult into a human pause, which bubbled up here.
                result =
                    handle_human_escalation(&orchestrator, &advisor_handler, *state, request).await;
            }
            other => {
                eprintln!("\nrun ended unexpectedly: {other:?}");
                std::process::exit(1);
            }
        }
    }
}

/// Build the orchestrator stream sink: boundary banners for the analysis worker,
/// terse lines for the standard tools (mirrors 11-multi-agent).
fn stream_sink(
    call_names: Arc<Mutex<HashMap<String, String>>>,
) -> Box<dyn Fn(HarnessStreamEvent) + Send + Sync> {
    Box::new(move |event: HarnessStreamEvent| match event {
        HarnessStreamEvent::TurnStart { turn } => {
            println!("orchestrator · turn {turn}");
        }
        HarnessStreamEvent::ToolCall {
            call_id,
            name,
            args,
        } => {
            call_names.lock().unwrap().insert(call_id, name.clone());
            if name == "analysis_worker" {
                let instruction = args
                    .get("instruction")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<no instruction>");
                println!("┌─ orchestrator → analysis_worker");
                println!("│  received: {}", truncate(instruction, 200));
            } else {
                println!(
                    "  orchestrator → {name}({})",
                    truncate(&args.to_string(), 140)
                );
            }
        }
        HarnessStreamEvent::ToolResult {
            call_id,
            is_error,
            content,
        } => {
            let name = call_names
                .lock()
                .unwrap()
                .remove(&call_id)
                .unwrap_or_else(|| "<tool>".to_string());
            if name == "analysis_worker" {
                let tag = if is_error { "FAILED" } else { "findings" };
                println!("└─ analysis_worker → orchestrator");
                println!("   {tag}: {}", truncate(&content, 300));
            } else {
                let tag = if is_error { "err" } else { "ok" };
                println!(
                    "  {name} → orchestrator [{tag}]: {}",
                    truncate(&content, 140)
                );
            }
        }
        _ => {}
    })
}

/// Handle an advice-budget-exhausted human escalation with the three-choice
/// ladder. Returns the next `RunResult` (the orchestrator resumed).
///
/// IMPORTANT (honest mechanics): the worker's paused consult lives inside the
/// orchestrator's `PausedState.child_state`, and the harness does NOT yet wire
/// a child-consult resume through the parent (`resume_inner` no-ops the
/// `child_state` branch — that lands with the #5/#115 follow-up). So every
/// choice here resumes the ORCHESTRATOR with the human's decision injected as
/// guidance; the specific module's in-flight worker audit is dropped. "+1
/// advisor turn" re-runs the advisor handler HOST-SIDE and injects its answer as
/// that guidance — the closest we can get to a budget bump without a core
/// primitive.
async fn handle_human_escalation(
    orchestrator: &spore_core::StandardHarness,
    advisor_handler: &Arc<dyn Harness>,
    state: spore_core::PausedState,
    request: HumanRequest,
) -> RunResult {
    let context = match &request {
        HumanRequest::Review { content } => content.clone(),
        HumanRequest::Clarification { question, .. } => question.clone(),
        HumanRequest::ToolApproval { .. } => "tool approval requested".to_string(),
    };
    println!("\n╔═ HUMAN ESCALATION (advisor budget exhausted) ═");
    println!("║ {}", truncate(&context, 400));
    println!("╚═══════════════════════════════════════════════");
    println!("Choose: [1] +1 advisor turn  [2] abort subagent & chat  [3] free-form answer");

    let choice = prompt_line("> ");
    match choice.trim() {
        "1" => {
            // Re-run the advisor handler once, host-side, on the escalation
            // context; inject its output as the orchestrator's guidance.
            println!("(running advisor for one more turn…)");
            let task = Task::new(
                context,
                SessionId::generate(),
                LoopStrategy::ReAct { max_iterations: 16 },
            );
            let advice = match advisor_handler.run(HarnessRunOptions::new(task)).await {
                RunResult::Success { output, .. } => output,
                other => format!("advisor did not complete cleanly: {other:?}"),
            };
            println!("advisor: {}", truncate(&advice, 300));
            orchestrator
                .resume(state, HumanResponse::Answer { text: advice }, None)
                .await
        }
        "2" => {
            println!("(aborting the stuck subagent; returning to the orchestrator…)");
            orchestrator.resume(state, HumanResponse::Halt, None).await
        }
        _ => {
            let text = if choice.trim() == "3" {
                prompt_line("your answer> ")
            } else {
                choice
            };
            orchestrator
                .resume(state, HumanResponse::Answer { text }, None)
                .await
        }
    }
}

/// After a successful audit, present the top-5 and offer to file them as issues.
/// The model drives `gh issue create` via `bash_command` (no `gh` skill).
async fn run_issue_filing_flow(
    orchestrator: &spore_core::StandardHarness,
    findings: &std::path::Path,
) {
    let report = std::fs::read_to_string(findings).unwrap_or_default();
    println!("\n── top findings (workspace/findings.md) ──");
    println!("{}", truncate(&report, 1200));
    println!("──────────────────────────────────────────");

    let answer = prompt_line("File these as GitHub issues? [y/N] ");
    if !answer.trim().eq_ignore_ascii_case("y") {
        println!("Not filing. Done.");
        return;
    }

    println!("(asking the orchestrator to file the top 5 via `gh issue create`…)");
    let task = Task::new(
        "Using `bash_command`, file the TOP 5 findings from workspace/findings.md as GitHub \
         issues via `gh issue create` — one issue per finding, with a clear title and a body \
         containing the file, line, severity, and description. Run `gh` once per finding. Report \
         the issue URLs when done."
            .to_string(),
        SessionId::generate(),
        LoopStrategy::ReAct { max_iterations: 24 },
    );
    match orchestrator.run(HarnessRunOptions::new(task)).await {
        RunResult::Success { output, .. } => {
            println!("\nfiling done: {}", truncate(&output, 400));
        }
        other => eprintln!("\nfiling did not complete cleanly: {other:?}"),
    }
}

/// Seed `.spore/skills/audit/SKILL.md` from the bundled copy if absent.
fn seed_bundled_audit_skill(repo_root: &std::path::Path) {
    let dir = repo_root.join(".spore").join("skills").join("audit");
    let file = dir.join("SKILL.md");
    if file.exists() {
        return;
    }
    if std::fs::create_dir_all(&dir).is_ok() {
        let _ = std::fs::write(&file, BUNDLED_AUDIT_SKILL);
    }
}

/// Read the audit prompt from the REPL: print the default, accept an empty line
/// as the default verbatim.
fn read_audit_prompt() -> String {
    println!("Default audit prompt (press enter to accept, or type your own):");
    println!("  {DEFAULT_AUDIT_PROMPT}");
    let line = prompt_line("audit> ");
    if line.trim().is_empty() {
        DEFAULT_AUDIT_PROMPT.to_string()
    } else {
        line
    }
}

/// Print a prompt and read one line from stdin (trailing newline stripped).
fn prompt_line(prompt: &str) -> String {
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    let mut buf = String::new();
    if std::io::stdin().read_line(&mut buf).is_err() {
        return String::new();
    }
    buf.trim_end_matches(['\n', '\r']).to_string()
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Keep boundary lines readable.
fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max {
        s
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}
