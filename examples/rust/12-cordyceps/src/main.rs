//! spore-core example 12 — **cordyceps**: the capstone of the Composable
//! Execution refactor (#117–#131).
//!
//! **The thesis: you describe a strategy as DATA — a composed `LoopStrategy`
//! tree — wire its string handles to concrete collaborators in an
//! [`ExecutionRegistry`], and the harness runs the whole nested machine under
//! one shared budget / usage / observability context.**
//!
//! The motivating composition is:
//!
//! ```text
//! Ralph[ PlanExecute[ ReAct, SelfVerifying[ ReAct ] ] ]
//! │       │             │      │             │
//! │       │             │      │             └─ worker: audits ONE module
//! │       │             │      └─ Default-FAIL evaluator (single read-only turn)
//! │       │             └─ plan: explores the repo, builds a blocker-aware DAG
//! │       └─ plan→ready-set: walks the DAG in dependency order, self-verifying each task
//! └─ continuation wrapper: resets the window, resumes from durable progress
//! ```
//!
//! ## What changed vs. the pre-#131 example (HONEST note)
//!
//! The old depth-1 example used a hand-built `SubagentTool` orchestrator with a
//! per-node consult mediator (#114) and an architect-side `load_skill` tool
//! (#115). The declarative tree has NO SubagentTool seam, so:
//!
//! - the #114 consult ladder is **PRESERVED, with its mediation seam moved**.
//!   The worker still calls `research_best_practices` / `consult_advisor`, which
//!   lower to `ToolOutput::Consult`. With no `SubagentTool` to mediate, the
//!   worker-leaf consult propagates all the way up to a top-level
//!   [`RunResult::Consult`], and the HOST run loop mediates it — routing by
//!   `kind` to a helper harness with a per-kind budget + overflow policy
//!   (`research` → web_search, budget 5, SoftFail; `advice` → cloud advisor,
//!   budget 3, EscalateToHuman). Identical #114 semantics, host-owned budgets.
//! - `load_skill` is **dropped** — there is no worker-side per-node seam;
//! - the `audit` skill is **kept**, but now rides the single GLOBAL
//!   [`SkillInjectingContextManager`] (the harness's `context_manager`), seeded
//!   ALWAYS-ACTIVE at startup. The audit procedure reaches the model structurally
//!   every turn, compaction-proof, with no `load_skill` round-trip.
//!
//! ## The tree is DATA
//!
//! We do NOT hand-build the [`LoopStrategy`]. We `include_str!` the canonical
//! fixture `fixtures/strategy/cordyceps_tree.json` and deserialize it — so this
//! example proves the canonical fixture deserializes and runs.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull gemma4:e4b
//! cargo run
//! ```

mod skills;
mod tools;

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

use spore_core::storage::InMemoryStorageProvider;
use spore_core::{
    Agent, AgentId, BudgetLimits, ConsultHandlerEntry, ConsultOverflowPolicy, ConsultRequest,
    ConsultResponse, EmptyToolRegistry, EscalationAction, EscalationMode, EvaluatorResponseVerifier,
    ExecutionRegistry, Harness, HarnessBuilder, HarnessContextManager, HarnessContextManagerExt,
    HarnessRunOptions, HarnessStreamEvent, HumanRequest, HumanResponse, LoopStrategy, ModelAgent,
    NullCacheProvider, OllamaModelInterface, ReactConfig, RunResult, SearchMethod, SessionId,
    StandardContextManager, StandardTool, StandardTools, StorageProvider, Task, Verifier,
    WebSearchConfig, WebSearchTool, WorkspaceConfig, WorkspaceScopedSandbox,
};

use crate::skills::{SkillCatalog, SkillInjectingContextManager, ACTIVE_SKILLS_KEY};
use crate::tools::consult::{
    consult_advisor_tool, research_best_practices_tool, KIND_ADVICE, KIND_RESEARCH,
};
use crate::tools::send_message::send_user_message_tool;

/// The canonical composed-strategy fixture, embedded so the example proves the
/// ground-truth tree deserializes (and runs) verbatim — never hand-built.
pub const CORDYCEPS_TREE_JSON: &str =
    include_str!("../../../../fixtures/strategy/cordyceps_tree.json");

/// Bundled `audit` skill (the global context_manager's always-active procedure).
const BUNDLED_AUDIT_SKILL: &str = include_str!("../skills/audit/SKILL.md");

/// The verifier registry key the `SelfVerifying` node's `evaluator` resolves to.
pub const EXEC_EVALUATOR_KEY: &str = "exec-evaluator";

/// The pre-filled audit prompt (press enter to accept).
const DEFAULT_AUDIT_PROMPT: &str =
    "Audit this repository for Rust defects. Discover the crates and their modules, audit each \
     module for real, actionable defects, and write a markdown report of the most important \
     findings to `workspace/findings.md`.";

const EXEC_SYSTEM_PROMPT: &str = "\
You are a cordyceps execution machine. Your strategy is composed declaratively: a Ralph \
continuation wrapper drives a PlanExecute, whose plan phase explores the repo and builds a \
blocker-aware task graph via `task_list`, and whose execute phase walks that graph as a \
ready-set — auditing one module per ready task, each result self-verified by a read-only \
evaluator (Default-FAIL: only an explicit PASS clears a task).

Before each step, call `send_user_message` with one short sentence telling the watching human \
what you are about to do and why.

You are already scoped to the repository root (READ-ONLY). Use `.` for the root and paths \
relative to it (e.g. `rust/crates`); never prefix a path with the repository's own folder name. \
The audit is read-only — you have no write tool; never attempt to modify source files.

Follow the ACTIVE `audit` skill's procedure and output schema exactly: grep first, read narrow, \
and return findings as a JSON array of {file, line, severity, description}.

PLAN phase: explore the repo with `list_dir`/`grep`, then build a blocker-aware task graph with \
`task_list` (one task per module; add dependencies where one audit should wait on another). \
RALPH wrapper: resume from durable `task_list` progress after each context-window reset and keep \
going until every task is done.";

/// `plan-schema` — the task-graph contract the plan phase's ReAct emits.
fn plan_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "Ordered task-graph entries; each names a module to audit.",
                "items": {
                    "type": "object",
                    "properties": {
                        "module": { "type": "string", "description": "Module path to audit." },
                        "blockers": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "description": "1-based ids of tasks this one waits on."
                        }
                    },
                    "required": ["module"]
                }
            },
            "rationale": { "type": "string" }
        },
        "required": ["tasks"]
    })
}

/// `worker-schema` — the per-module finding contract the worker ReAct emits.
fn worker_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "description": "Findings for ONE module.",
        "items": {
            "type": "object",
            "properties": {
                "file": { "type": "string", "description": "Path relative to the repo root." },
                "line": { "type": "integer", "description": "1-based line of the defect." },
                "severity": { "enum": ["low", "medium", "high", "critical"] },
                "description": { "type": "string", "description": "Concrete, actionable defect." }
            },
            "required": ["file", "line", "severity", "description"]
        }
    })
}

/// The `plan-tools` catalogue: explore + author the task graph (read-only).
fn plan_tools() -> Vec<StandardTool> {
    vec![
        StandardTools::list_dir(),
        StandardTools::grep(),
        StandardTools::task_list(),
    ]
}

/// The `exec-tools` catalogue: read-only audit + the #114 consult ladder +
/// human observability. The two consult tools lower to `ToolOutput::Consult`,
/// which the host run loop mediates (the seam moved off `SubagentTool`).
fn exec_tools() -> Vec<StandardTool> {
    vec![
        StandardTools::list_dir(),
        StandardTools::read_file(),
        StandardTools::grep(),
        research_best_practices_tool(),
        consult_advisor_tool(),
        send_user_message_tool("🤖"),
    ]
}

const RESEARCH_PROMPT: &str = "\
You are a research worker. A peer agent needs factual, current information on a Rust best-practice \
or language question.

Use `web_search` to find the answer. Issue focused queries, read the results, and return a \
concise cited answer in plain text. Do not answer from memory alone — always search first.";

const ADVISOR_PROMPT: &str = "\
You are a senior Rust advisor. A worker has escalated a candidate finding to you because they \
need a judgment call.

Use `read_file` and `grep` to examine the specific code in question. Then make a decision: \
is this a real defect, what is the severity (low / medium / high / critical), and why. \
Be decisive. State your verdict in one sentence, your reasoning in two. Do not hedge.";

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

/// Build the research handler harness (web_search only) — the `kind="research"`
/// consult handler. Run host-side on a `ConsultRequest`.
fn build_research_harness(model_id: &str, base_url: &str, endpoint: &str) -> Arc<dyn Harness> {
    let model = OllamaModelInterface::with_base_url(model_id, base_url.to_string());
    Arc::new(
        HarnessBuilder::conversational(model)
            .tool(build_web_search(endpoint))
            .system_prompt(RESEARCH_PROMPT)
            .build(),
    )
}

/// Build the advisor handler harness (cloud model, read_file + grep) — the
/// `kind="advice"` consult handler. Rides the same Ollama endpoint via
/// `with_base_url`; only the model id differs (heterogeneous models).
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

/// Build the HOST-owned `kind → {handler, budget, overflow}` map (#114). The
/// composed tree has no `SubagentTool`, so the host run loop holds these
/// entries and mediates each `RunResult::Consult` against them — the per-kind
/// budget lives for the whole run (see `mediate_consult`).
fn build_consult_handlers(
    research: Arc<dyn Harness>,
    advisor: Arc<dyn Harness>,
) -> HashMap<String, ConsultHandlerEntry> {
    let mut handlers = HashMap::new();
    handlers.insert(
        KIND_RESEARCH.to_string(),
        ConsultHandlerEntry {
            handler: research,
            budget: 5,
            overflow: ConsultOverflowPolicy::SoftFail,
        },
    );
    handlers.insert(
        KIND_ADVICE.to_string(),
        ConsultHandlerEntry {
            handler: advisor,
            budget: 3,
            overflow: ConsultOverflowPolicy::EscalateToHuman,
        },
    );
    handlers
}

/// Build a model agent (`Arc<dyn Agent>`) over the local Ollama model.
fn model_agent(id: &str, model_id: &str, base_url: &str) -> Arc<dyn Agent> {
    let model = Arc::new(OllamaModelInterface::with_base_url(
        model_id,
        base_url.to_string(),
    ));
    Arc::new(ModelAgent::new(AgentId::new(id), model))
}

/// The Default-FAIL self-verification evaluator registered under
/// `exec-evaluator`. A single read-only turn (`max_iterations = 1`); the
/// neither-pattern ⇒ Failed contract is built into [`EvaluatorResponseVerifier`].
pub fn exec_evaluator() -> Arc<dyn Verifier> {
    Arc::new(
        EvaluatorResponseVerifier::new(r"(?i)\bPASS\b", r"(?i)\bFAIL\b", 1)
            .expect("evaluator regexes are valid"),
    )
}

/// Assemble the [`ExecutionRegistry`] the cordyceps tree's handles resolve
/// against: agents `planner`/`executor`/`ralph-agent`, toolsets
/// `plan-tools`/`exec-tools`, schemas `plan-schema`/`worker-schema`, and the
/// `exec-evaluator` verifier. The handle STRINGS are ground truth from the
/// fixture; this is the host-side wiring of those strings to collaborators.
pub fn build_registry(model_id: &str, base_url: &str) -> ExecutionRegistry {
    ExecutionRegistry::builder()
        .agent("planner", model_agent("planner", model_id, base_url))
        .agent("executor", model_agent("executor", model_id, base_url))
        .agent(
            "ralph-agent",
            model_agent("ralph-agent", model_id, base_url),
        )
        // Per-node toolset scoping is now resolved (Issue 2): each leaf dispatches
        // ONLY its own toolset's catalogue. The real tools are wired per-handle on
        // the HarnessBuilder via `.toolset_tools("plan-tools", ..)` /
        // `.toolset_tools("exec-tools", ..)` (see `main`) and bridged per-run, so
        // the planner can no longer reach exec-only tools and vice-versa.
        //
        // These registry slots are now PRESENCE-ONLY: `ExecutionRegistry::validate`
        // checks every handle resolves, but the value is NEVER dispatched (dispatch
        // goes through the per-handle catalogues on the builder). The harness also
        // auto-fills these presence entries from `.toolset_tools`, so wiring them
        // here is just what keeps the standalone `build_registry().validate()`
        // contract self-consistent.
        .toolset("plan-tools", Arc::new(EmptyToolRegistry))
        .toolset("exec-tools", Arc::new(EmptyToolRegistry))
        .schema("plan-schema", plan_schema())
        .schema("worker-schema", worker_schema())
        .verifier(EXEC_EVALUATOR_KEY, exec_evaluator())
        .build()
}

/// Build the GLOBAL skill-injecting context manager with the `audit` skill
/// seeded ALWAYS-ACTIVE for `session`. Wraps the standard compaction adapter.
async fn build_global_context_manager(
    model_id: &str,
    base_url: &str,
    storage: &StorageProvider,
    repo_root: &std::path::Path,
    session: &SessionId,
) -> Arc<dyn HarnessContextManager> {
    let catalog = SkillCatalog::bootstrap(repo_root, BUNDLED_AUDIT_SKILL).await;
    // Seed `audit` always-active: the global context_manager injects its body
    // structurally every turn (no `load_skill` round-trip in the composed tree).
    storage
        .run()
        .put(session, ACTIVE_SKILLS_KEY, serde_json::json!(["audit"]))
        .await
        .expect("seed active_skills");

    let inner_model = Arc::new(OllamaModelInterface::with_base_url(
        model_id,
        base_url.to_string(),
    ));
    let inner: Arc<dyn HarnessContextManager> = Arc::new(StandardContextManager::new(
        inner_model,
        Arc::new(NullCacheProvider),
        spore_core::CompactionConfig::default(),
    ))
    .into_harness_adapter();
    Arc::new(SkillInjectingContextManager::new(
        inner,
        storage.run().clone(),
        catalog.manifest(),
    ))
}

/// Build the cordyceps [`Task`]: the composed tree deserialized from the fixture,
/// under a generous global backstop so the per-node `PerLoop{12}` worker bound
/// fires first.
pub fn build_task(prompt: String, session: SessionId) -> Task {
    let tree: LoopStrategy =
        serde_json::from_str(CORDYCEPS_TREE_JSON).expect("cordyceps_tree.json deserializes");
    Task::new(prompt, session, tree).with_budget(BudgetLimits {
        max_turns: Some(64),
        ..BudgetLimits::default()
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "gemma4:e4b".to_string());
    // The #114 advisor consult handler runs a heterogeneous (cloud) model.
    let advisor_model_id = arg_value(&args, "--advisor-model")
        .or_else(|| std::env::var("SPORE_ADVISOR_MODEL").ok())
        .unwrap_or_else(|| "minimax-m3:cloud".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    // The research consult handler needs a SearXNG JSON endpoint — fail fast
    // (like examples 06/11) so a missing backend is a startup error, not a
    // mid-run surprise.
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

    let repo_root = std::fs::canonicalize(std::env::current_dir()?)?;
    let workspace_root = repo_root.join("workspace");
    std::fs::create_dir_all(&workspace_root)?;

    // AC5: the fully-bounded tree's worst-case per-window turn count is computable
    // BEFORE the run. Ralph[PlanExecute[ReAct{4}, SelfVerifying[ReAct{12}]]]
    // = 4 + (12 + 1) = 17. An `Unlimited` anywhere would collapse this to None.
    let tree_preview: LoopStrategy = serde_json::from_str(CORDYCEPS_TREE_JSON)?;
    println!("model        : {model_id}");
    println!("advisor model: {advisor_model_id}");
    println!("search       : {endpoint}");
    println!("repo root    : {}", repo_root.display());
    println!("strategy     : Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]] (from fixture)");
    println!(
        "max_steps    : {:?}  (per-window worst case; Unlimited anywhere ⇒ None)",
        tree_preview.max_steps()
    );
    println!("consults     : research(web_search, budget 5, soft-fail), advice(advisor, budget 3, escalate)");
    println!();

    while let Some(prompt) = read_audit_prompt() {
        let session = SessionId::generate();
        let storage = Arc::new(StorageProvider::single(Arc::new(
            InMemoryStorageProvider::new(),
        )));

        // Read-only repo sandbox: the audit never writes source files.
        let mut sandbox_cfg = WorkspaceConfig::scoped(repo_root.clone());
        sandbox_cfg.read_only = true;
        let sandbox = Arc::new(WorkspaceScopedSandbox::new(sandbox_cfg)?);

        // The advisor handler gets its own read-only view of the repo (read_file
        // + grep) so it can inspect the code the worker is asking about.
        let mut advisor_cfg = WorkspaceConfig::scoped(repo_root.clone());
        advisor_cfg.read_only = true;
        let advisor_sandbox = Arc::new(WorkspaceScopedSandbox::new(advisor_cfg)?);

        // The HOST-owned consult ladder (#114). The seam moved off `SubagentTool`:
        // the host loop holds these handlers + per-kind budgets for the whole run.
        let consult_handlers = build_consult_handlers(
            build_research_harness(&model_id, &base_url, &endpoint),
            build_advisor_harness(&advisor_model_id, &base_url, advisor_sandbox),
        );

        let registry = build_registry(&model_id, &base_url);
        let context_manager =
            build_global_context_manager(&model_id, &base_url, &storage, &repo_root, &session)
                .await;

        // The harness's own model drives the Ralph wrapper; the per-node agents
        // come from the registry. Compaction/summarization uses this model too.
        let model = OllamaModelInterface::with_base_url(&model_id, base_url.clone());
        // Issue 2: each node dispatches ONLY its own toolset's catalogue. The
        // tools are wired per-handle (`plan-tools` / `exec-tools`) — the planner
        // can no longer reach exec-only tools (read_file/consult_advisor) and the
        // executor can no longer reach the plan-only `task_list` (though `list_dir`
        // and `grep` appear in both catalogues).
        let harness = HarnessBuilder::conversational(model)
            .sandbox(sandbox)
            .storage(storage.clone())
            .registry(registry)
            .escalation_mode(EscalationMode::SurfaceToHuman)
            .system_prompt(EXEC_SYSTEM_PROMPT)
            .context_manager(context_manager)
            .toolset_tools("plan-tools", plan_tools())
            .toolset_tools("exec-tools", exec_tools())
            .build();

        let task = build_task(prompt, session);
        // Per-RUN consult counts (host-owned, #114): how many consults of each
        // `kind` have already been mediated. Persists across every pause/resume
        // of THIS audit so the per-kind budget bounds the whole run, not one turn.
        let mut consult_counts: HashMap<String, u32> = HashMap::new();
        // ONE top-level stream sink now reaches EVERY node of the composed tree:
        // the combinators thread the sink down and stamp each event with the node
        // that produced it (kind / agent_id / depth / path), so the trace shows the
        // plan ReAct, the worker, and the evaluator live, indented by tree depth.
        // (This is the exact seam an API backing a web UI would forward over SSE.)
        let mut result = harness
            .run(HarnessRunOptions::new(task).with_stream(node_stream_sink()))
            .await;
        loop {
            match result {
                RunResult::Success { output, turns, .. } => {
                    println!("\ndone ({turns} turn(s)): {}", truncate(&output, 400));
                    break;
                }
                RunResult::Failure { reason, turns, .. } => {
                    eprintln!("\nfailed after {turns} turn(s): {reason:?}");
                    break;
                }
                RunResult::Consult { request, state, .. } => {
                    // A worker leaf consult propagated up through the composed
                    // tree (no SubagentTool to absorb it). The host mediates.
                    result = mediate_consult(
                        &harness,
                        &consult_handlers,
                        &mut consult_counts,
                        request,
                        *state,
                    )
                    .await;
                }
                RunResult::WaitingForHuman { state, request } => {
                    result = handle_human_escalation(&harness, *state, request).await;
                }
                other => {
                    eprintln!("\nrun ended unexpectedly: {other:?}");
                    break;
                }
            }
        }
    }

    println!("\nbye.");
    Ok(())
}

/// Mediate one worker leaf consult HOST-SIDE (#114, seam relocated off
/// `SubagentTool`). Routes by `kind`, enforces the per-kind budget held in
/// `counts` for the whole run, runs the handler harness as a direct child, and
/// resumes the paused composed tree with the answer — or applies the overflow
/// policy (`SoftFail` resumes with `BudgetExhausted`; `EscalateToHuman` surfaces
/// the advisor ladder to the operator). Identical to the old
/// `SubagentTool::mediate_consult`, only the owner moved.
async fn mediate_consult(
    harness: &spore_core::StandardHarness,
    handlers: &HashMap<String, ConsultHandlerEntry>,
    counts: &mut HashMap<String, u32>,
    request: ConsultRequest,
    state: spore_core::PausedState,
) -> RunResult {
    // No handler for this kind ⇒ resume the worker without help (loud, not a
    // silent stall). Matches SubagentTool R6 graceful degradation in spirit.
    let Some(entry) = handlers.get(&request.kind) else {
        eprintln!("\n(no consult handler for kind {:?}; worker proceeds)", request.kind);
        return harness
            .resume_consult(
                state,
                ConsultResponse::BudgetExhausted {
                    message: format!(
                        "no consult handler for kind {:?}; proceed without further help",
                        request.kind
                    ),
                },
                None,
            )
            .await;
    };

    // Per-kind budget: `used` is how many consults of this kind were already
    // mediated this run. The handler runs while `used < budget`; the
    // (budget+1)th consult overflows.
    let used = counts.entry(request.kind.clone()).or_insert(0);
    if *used >= entry.budget {
        return match entry.overflow {
            ConsultOverflowPolicy::SoftFail => {
                println!(
                    "\n(consult budget for {:?} exhausted — worker finishes with what it has)",
                    request.kind
                );
                harness
                    .resume_consult(
                        state,
                        ConsultResponse::BudgetExhausted {
                            message: format!(
                                "consult budget for kind {:?} exhausted; proceed without further help",
                                request.kind
                            ),
                        },
                        None,
                    )
                    .await
            }
            ConsultOverflowPolicy::EscalateToHuman => {
                handle_consult_overflow(harness, entry, request, state).await
            }
        };
    }

    // Run the handler harness as a direct child (depth-1, never under the
    // worker) on the consult rendered to text, then resume with its answer.
    *used += 1;
    println!(
        "\n┌─ consult ({}) → {} of {} budget",
        request.kind, *used, entry.budget
    );
    let instruction = render_consult_instruction(&request);
    let task = Task::new(
        instruction,
        SessionId::generate(),
        LoopStrategy::ReAct(ReactConfig::per_loop(16)),
    );
    let answer = match entry
        .handler
        .run(HarnessRunOptions::new(task).with_stream(node_stream_sink_prefixed("│  ")))
        .await
    {
        RunResult::Success { output, .. } => output,
        // A handler that does not cleanly complete must not stall the worker —
        // feed its failure text back as the answer so the worker can adapt.
        other => format!("consult handler did not complete cleanly: {other:?}"),
    };
    println!("└─ consult answer: {}", truncate(&answer, 200));
    harness
        .resume_consult(state, ConsultResponse::Answer { text: answer }, None)
        .await
}

/// Render a [`ConsultRequest`] to the handler's instruction text (#114).
fn render_consult_instruction(request: &ConsultRequest) -> String {
    format!(
        "A worker agent is requesting help (kind: {kind}).\n\nSituation: {situation}\n\nAttempts so far: {attempts}\n\nQuestion: {question}",
        kind = request.kind,
        situation = request.situation,
        attempts = request.attempts,
        question = request.question,
    )
}

/// The `advice` consult overflowed its budget under `EscalateToHuman`: present
/// the #114 three-choice ladder to the operator and resume the worker with the
/// decision. Preserves the original ladder semantics host-side.
async fn handle_consult_overflow(
    harness: &spore_core::StandardHarness,
    entry: &ConsultHandlerEntry,
    request: ConsultRequest,
    state: spore_core::PausedState,
) -> RunResult {
    println!("\n╔═ HUMAN ESCALATION (advisor budget exhausted) ═");
    println!("║ situation: {}", truncate(&request.situation, 200));
    println!("║ question : {}", truncate(&request.question, 200));
    println!("║ [1] run the advisor once more (host-side)");
    println!("║ [2] abort this consult — worker proceeds without help");
    println!("║ [3] type a free-form answer yourself");
    println!("╚═════════════════════════════════════════════════");

    let choice = prompt_line("> ");
    match choice.trim() {
        "2" => {
            harness
                .resume_consult(
                    state,
                    ConsultResponse::BudgetExhausted {
                        message: "advisor budget exhausted; proceed without further help".into(),
                    },
                    None,
                )
                .await
        }
        "3" => {
            let text = prompt_line("answer> ");
            harness
                .resume_consult(state, ConsultResponse::Answer { text }, None)
                .await
        }
        // Default ([1] or empty): run the advisor handler once more host-side and
        // inject its answer — a bounded escape hatch past the per-kind budget.
        _ => {
            println!("(running advisor for one more turn…)");
            let task = Task::new(
                render_consult_instruction(&request),
                SessionId::generate(),
                LoopStrategy::ReAct(ReactConfig::per_loop(16)),
            );
            let answer = match entry
                .handler
                .run(HarnessRunOptions::new(task).with_stream(node_stream_sink_prefixed("│  ")))
                .await
            {
                RunResult::Success { output, .. } => output,
                other => format!("advisor did not complete cleanly: {other:?}"),
            };
            println!("advisor: {}", truncate(&answer, 300));
            harness
                .resume_consult(state, ConsultResponse::Answer { text: answer }, None)
                .await
        }
    }
}

/// Present a `BudgetExhausted` pause and resume with the operator's choice. The
/// composed tree surfaces a runaway node here under `SurfaceToHuman`; we offer
/// its `available_actions` and resume by re-resolving handles (no
/// reconfiguration).
async fn handle_human_escalation(
    harness: &spore_core::StandardHarness,
    state: spore_core::PausedState,
    request: HumanRequest,
) -> RunResult {
    let (phase, actions) = match &request {
        HumanRequest::BudgetExhausted {
            phase,
            available_actions,
            ..
        } => (phase.clone(), available_actions.clone()),
        other => {
            // The composed tree only escalates via BudgetExhausted; anything else
            // is unexpected — halt cleanly.
            eprintln!("\nunexpected human request: {other:?}");
            return harness.resume(state, HumanResponse::Halt, None).await;
        }
    };

    println!("\n╔═ BUDGET ESCALATION ({phase}) ═══════════════════");
    for (i, a) in actions.iter().enumerate() {
        println!("║ [{}] {}", i + 1, describe_action(a));
    }
    println!("╚═════════════════════════════════════════════════");

    let choice = prompt_line("> ");
    let idx = choice
        .trim()
        .parse::<usize>()
        .unwrap_or(1)
        .saturating_sub(1);
    let action = actions
        .get(idx)
        .cloned()
        // Default to a small budget bump so an empty line keeps the run going.
        .unwrap_or(EscalationAction::ContinueWithBudget { steps: 12 });

    println!("(resuming with {})", describe_action(&action));
    harness
        .resume(state, HumanResponse::Escalate { action }, None)
        .await
}

fn describe_action(a: &EscalationAction) -> String {
    match a {
        EscalationAction::ContinueWithBudget { steps } => {
            format!("continue with +{steps} steps")
        }
        EscalationAction::Skip => "skip this task".to_string(),
        EscalationAction::Fail => "fail this node".to_string(),
    }
}

/// Read one audit prompt from the REPL. `Some(prompt)` to run (empty line ⇒ the
/// default verbatim); `None` on EOF (Ctrl-D), which quits the REPL.
fn read_audit_prompt() -> Option<String> {
    println!("Default audit prompt (press enter to accept, type your own, or Ctrl-D to quit):");
    println!("  {DEFAULT_AUDIT_PROMPT}");
    print!("audit> ");
    let _ = std::io::stdout().flush();
    let mut buf = String::new();
    match std::io::stdin().read_line(&mut buf) {
        Ok(0) => None,
        Ok(_) => {
            let line = buf.trim_end_matches(['\n', '\r']).to_string();
            if line.trim().is_empty() {
                Some(DEFAULT_AUDIT_PROMPT.to_string())
            } else {
                Some(line)
            }
        }
        Err(_) => None,
    }
}

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

/// The TOP-LEVEL stream sink for the composed run. The declarative tree threads
/// ONE sink through every node and stamps each event with the node that produced
/// it ([`HarnessStreamEvent::node`]: kind / agent_id / depth / root→leaf path),
/// so a single `with_stream` surfaces the WHOLE tree — plan ReAct, worker, and
/// evaluator — indented by tree depth and labelled by the speaking agent. This is
/// the exact channel an API backing a web UI subscribes to and forwards over SSE.
fn node_stream_sink() -> impl Fn(HarnessStreamEvent) + Send + Sync + 'static {
    node_stream_sink_prefixed("")
}

/// As [`node_stream_sink`], with a fixed `base` prefix so a consult HANDLER's
/// (separate, bare-ReAct) run nests visually under its consult banner.
fn node_stream_sink_prefixed(
    base: &'static str,
) -> impl Fn(HarnessStreamEvent) + Send + Sync + 'static {
    let call_names: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
    move |event: HarnessStreamEvent| {
        // Indent by tree depth + a per-node tag (agent_id, else kind) drawn from
        // the event's attribution. `None` (a bare leaf at the top) ⇒ depth 0.
        let (depth, tag) = match event.node() {
            Some(n) => (
                n.depth as usize,
                n.agent_id.clone().unwrap_or_else(|| n.kind.clone()),
            ),
            None => (0, "·".to_string()),
        };
        let indent = format!("{base}{}", "  ".repeat(depth));
        match event {
            HarnessStreamEvent::TurnStart { turn, .. } => {
                println!("{indent}[{tag}] · turn {turn}");
            }
            HarnessStreamEvent::ToolCall {
                call_id, name, args, ..
            } => {
                call_names.lock().unwrap().insert(call_id, name.clone());
                // `send_user_message` renders itself as the 🤖 UserMessage event.
                if name != "send_user_message" {
                    println!("{indent}  → {name}({})", truncate(&args.to_string(), 140));
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
                if name != "send_user_message" {
                    let t = if is_error { "err" } else { "ok" };
                    println!("{indent}  {name} [{t}]: {}", truncate(&content, 140));
                }
            }
            // A worker `send_user_message` surfaces here (issue #81 out-of-band).
            HarnessStreamEvent::UserMessage { content, .. } => {
                println!("{indent}🤖 {}", truncate(&content, 200));
            }
            // A node's final assistant text (plan JSON, worker findings, evaluator
            // verdict) — the substance when a turn emits no tool call.
            HarnessStreamEvent::FinalResponse { content, .. } => {
                println!("{indent}⟐ {}", truncate(&content, 200));
            }
            // Delta-level events (#103), block start/stop, budget warnings: ignored
            // here — the coarse events above are enough for a readable trace.
            _ => {}
        }
    }
}

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
// Example-crate tests (NO model): the tree is data, max_steps is computable,
// and the registry validates the real task.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "gemma4:e4b";
    const BASE: &str = "http://localhost:11434";

    /// AC: the tree is DATA. Deserialize the included canonical fixture,
    /// re-serialize, and assert the value round-trips; then assert the expected
    /// keys / budgets / behaviors are present.
    #[test]
    fn tree_is_byte_identical() {
        let tree: LoopStrategy =
            serde_json::from_str(CORDYCEPS_TREE_JSON).expect("fixture deserializes");
        // Round-trip equality at the JSON-value level (key order independent).
        let reserialized = serde_json::to_value(&tree).unwrap();
        let original: serde_json::Value = serde_json::from_str(CORDYCEPS_TREE_JSON).unwrap();
        assert_eq!(reserialized, original, "tree must round-trip through serde");

        // Structural assertions on the canonical shape.
        let LoopStrategy::Ralph(ralph) = &tree else {
            panic!("root must be Ralph");
        };
        assert_eq!(ralph.agent.0, "ralph-agent");
        assert!(matches!(
            ralph.behavior,
            spore_core::BudgetExhaustedBehavior::Escalate
        ));
        let LoopStrategy::PlanExecute(pe) = ralph.inner.as_ref() else {
            panic!("Ralph inner must be PlanExecute");
        };
        // plan = ReAct{planner, plan-tools, plan-schema, PerLoop{4}}
        let LoopStrategy::ReAct(plan) = pe.plan.as_ref() else {
            panic!("plan must be ReAct");
        };
        assert_eq!(plan.agent.0, "planner");
        assert_eq!(plan.toolset.0, "plan-tools");
        assert_eq!(plan.output.as_ref().unwrap().0, "plan-schema");
        assert_eq!(plan.budget, spore_core::BudgetPolicy::PerLoop { value: 4 });
        // execute = SelfVerifying{ ReAct{executor, exec-tools, worker-schema, 12}, exec-evaluator }
        let LoopStrategy::SelfVerifying(sv) = pe.execute.as_ref() else {
            panic!("execute must be SelfVerifying");
        };
        assert_eq!(sv.evaluator.0, "exec-evaluator");
        let LoopStrategy::ReAct(worker) = sv.inner.as_ref() else {
            panic!("worker must be ReAct");
        };
        assert_eq!(worker.agent.0, "executor");
        assert_eq!(worker.toolset.0, "exec-tools");
        assert_eq!(worker.output.as_ref().unwrap().0, "worker-schema");
        assert_eq!(
            worker.budget,
            spore_core::BudgetPolicy::PerLoop { value: 12 }
        );
    }

    /// AC5: the fully-bounded tree's per-window worst case is `Some(17)`; one
    /// `Unlimited` anywhere collapses it to `None`.
    #[test]
    fn max_steps_is_17() {
        let tree: LoopStrategy = serde_json::from_str(CORDYCEPS_TREE_JSON).unwrap();
        assert_eq!(tree.max_steps(), Some(17));

        // Swap the worker's PerLoop{12} for Unlimited ⇒ None.
        let LoopStrategy::Ralph(mut ralph) = tree else {
            unreachable!()
        };
        let LoopStrategy::PlanExecute(pe) = ralph.inner.as_mut() else {
            unreachable!()
        };
        let LoopStrategy::SelfVerifying(sv) = pe.execute.as_mut() else {
            unreachable!()
        };
        let LoopStrategy::ReAct(worker) = sv.inner.as_mut() else {
            unreachable!()
        };
        worker.budget = spore_core::BudgetPolicy::Unlimited;
        let mutated = LoopStrategy::Ralph(ralph);
        assert_eq!(mutated.max_steps(), None);
    }

    /// AC: handles resolve from the ExecutionRegistry at run entry. Build the
    /// real registry + task and assert `validate().is_ok()`.
    #[test]
    fn registry_validates() {
        let registry = build_registry(MODEL, BASE);
        let task = build_task("audit the repo".to_string(), SessionId::generate());
        assert!(
            registry.validate(&task).is_ok(),
            "every handle in the cordyceps task must resolve"
        );
    }

    /// The Default-FAIL evaluator: PASS clears, indeterminate output fails.
    #[tokio::test]
    async fn exec_evaluator_is_default_fail() {
        use spore_core::{AggregateUsage, RunResult, SessionState, VerifierInput, VerifierVerdict};
        let v = exec_evaluator();
        assert_eq!(v.max_iterations(), 1);

        let success = |out: &str| RunResult::Success {
            output: out.into(),
            session_id: SessionId::new("s"),
            usage: AggregateUsage::default(),
            turns: 1,
            session_state: SessionState::default(),
        };
        let input = |eval: &str| VerifierInput {
            build_result: success("audited"),
            eval_result: success(eval),
            workspace: std::path::PathBuf::from("/tmp"),
            iteration: 0,
        };
        assert_eq!(
            v.verify(&input("verdict: PASS")).await,
            VerifierVerdict::Passed
        );
        assert!(matches!(
            v.verify(&input("hmm, unclear")).await,
            VerifierVerdict::Failed { .. }
        ));
    }
}
