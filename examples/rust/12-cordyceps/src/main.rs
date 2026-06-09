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
//! - the #114 consult ladder is **dropped** — there is no per-node
//!   `ToolOutput::Consult` handler in the composed tree;
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

use std::io::Write;
use std::sync::Arc;

use spore_core::storage::InMemoryStorageProvider;
use spore_core::{
    Agent, AgentId, BudgetLimits, EmptyToolRegistry, EscalationAction, EscalationMode,
    EvaluatorResponseVerifier, ExecutionRegistry, Harness, HarnessBuilder, HarnessContextManager,
    HarnessContextManagerExt, HarnessRunOptions, HumanRequest, HumanResponse, LoopStrategy,
    ModelAgent, NullCacheProvider, OllamaModelInterface, RunResult, SessionId,
    StandardContextManager, StandardTool, StandardTools, StorageProvider, Task, Verifier,
    WorkspaceConfig, WorkspaceScopedSandbox,
};

use crate::skills::{SkillCatalog, SkillInjectingContextManager, ACTIVE_SKILLS_KEY};
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

/// The `exec-tools` catalogue: read-only audit + human observability.
fn exec_tools() -> Vec<StandardTool> {
    vec![
        StandardTools::read_file(),
        StandardTools::grep(),
        send_user_message_tool("🤖"),
    ]
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
        // The toolset HANDLES must resolve for `validate()`. The harness run loop
        // dispatches every node through the single GLOBAL catalogue wired on the
        // HarnessBuilder (`.tools(...)`), not per-node — a known harness scoping
        // limitation — so these registry slots only need to be present, not
        // distinct dispatchers. The real tools live on the builder (see `main`).
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
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    let repo_root = std::fs::canonicalize(std::env::current_dir()?)?;
    let workspace_root = repo_root.join("workspace");
    std::fs::create_dir_all(&workspace_root)?;

    // AC5: the fully-bounded tree's worst-case per-window turn count is computable
    // BEFORE the run. Ralph[PlanExecute[ReAct{4}, SelfVerifying[ReAct{12}]]]
    // = 4 + (12 + 1) = 17. An `Unlimited` anywhere would collapse this to None.
    let tree_preview: LoopStrategy = serde_json::from_str(CORDYCEPS_TREE_JSON)?;
    println!("model      : {model_id}");
    println!("repo root  : {}", repo_root.display());
    println!("strategy   : Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]] (from fixture)");
    println!(
        "max_steps  : {:?}  (per-window worst case; Unlimited anywhere ⇒ None)",
        tree_preview.max_steps()
    );
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

        let registry = build_registry(&model_id, &base_url);
        let context_manager =
            build_global_context_manager(&model_id, &base_url, &storage, &repo_root, &session)
                .await;

        // The harness's own model drives the Ralph wrapper; the per-node agents
        // come from the registry. Compaction/summarization uses this model too.
        let model = OllamaModelInterface::with_base_url(&model_id, base_url.clone());
        // The harness dispatches every node through ONE global catalogue: the
        // union of plan-tools + exec-tools (read_file + grep dedupe by last-wins).
        let harness = HarnessBuilder::conversational(model)
            .sandbox(sandbox)
            .storage(storage.clone())
            .registry(registry)
            .escalation_mode(EscalationMode::SurfaceToHuman)
            .system_prompt(EXEC_SYSTEM_PROMPT)
            .context_manager(context_manager)
            .tools(plan_tools())
            .tools(exec_tools())
            .build();

        let task = build_task(prompt, session);
        let mut result = harness.run(HarnessRunOptions::new(task)).await;
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
