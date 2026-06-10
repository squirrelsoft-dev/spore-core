//! spore-core example 09 — the `SelfVerifying` loop strategy.
//!
//! ## What this example demonstrates
//!
//! **Quality loops are a harness concern, not application logic.** The agent
//! drafts a Rust function, a *fresh evaluator run* critiques that draft against
//! an explicit spec, and a [`Verifier`] turns the critique into a verdict. If the
//! verdict is FAIL, the reason is injected back into the build context and the
//! loop revises. This repeats until the verifier returns `Passed` or
//! `max_iterations` is exhausted. You write **no loop code** — you wire a
//! strategy ([`LoopStrategy::SelfVerifying`]), a [`Verifier`], and an evaluator
//! agent, and the harness runs the verify→revise cycle for you.
//!
//! Post-#119 the strategy is a composed tree, not the old unit literal:
//!
//! ```text
//! // pre-#119: LoopStrategy::SelfVerifying  (unit variant)
//! // post-#119:
//! LoopStrategy::SelfVerifying(SelfVerifyingConfig {
//!     inner: ReAct { output: Some("worker-schema"), budget: PerLoop(max_iterations) },
//!     evaluator: "",   // empty handle ⇒ default-filled from `.verifier(..)`
//!     behavior: Escalate,
//! })
//! ```
//!
//! The `inner` (worker) slot is STRUCTURED — a bare `ReAct` there MUST declare an
//! `output` schema (here `worker-schema`) so its result is evaluable; this is
//! enforced by `ExecutionRegistry::validate` at run entry. The `evaluator` stays
//! the EMPTY handle (`""`), which the builder default-fills from the
//! single-collaborator `.verifier(..)` setter — so the only handle registered
//! explicitly is `worker-schema`.
//!
//! ## The task the agent-under-test must solve
//!
//! Write a Rust function `parse_int_list(&str) -> Result<Vec<i32>, ParseIntListError>`
//! that parses a comma-separated list of integers. The verifier checks **five**
//! criteria, each explicitly:
//!
//! 1. **Signature**: takes `&str`, returns `Result<Vec<i32>, ParseIntListError>`
//!    with a custom error type.
//! 2. **Edge cases**: empty/whitespace-only input → `Ok(vec![])`; whitespace
//!    around each number tolerated (`" 1, 2 ,3 "`); a non-integer token →
//!    `Err(...)`, never a panic.
//! 3. **Doc comments** on the function.
//! 4. **No `unwrap()` / no `panic!`**.
//! 5. **At least one usage example** (a `rust` doctest block or an inline example
//!    in the doc comment).
//!
//! ## How the draft reaches the evaluator — and why we need a file tool
//!
//! Reading the strategy source ([`run_self_verifying`] / `run_evaluate_phase` in
//! `harness.rs`) settles the tool question. The evaluate phase builds a **fresh**
//! evaluator run whose context is seeded ONLY with a directive containing the
//! `task.instruction` plus the read-only sandbox. The build agent's draft text is
//! **not** auto-injected into the evaluator's context. So for the evaluator to
//! actually read the draft, the draft must live on disk where the (read-only)
//! evaluator can read it.
//!
//! Therefore this example wires exactly the minimal file tool set:
//! - `write_file` — the **build** agent saves its draft to `workspace/parse_int_list.rs`.
//! - `read_file`  — the **evaluator** reads that file back (writes are blocked for
//!   it by the internally-derived [`ReadOnlySandbox`]).
//!
//! No `web_search`, no shell, nothing else. The loop is the point. (If a future
//! strategy revision fed the build text straight to the evaluator, this could
//! drop to zero tools — but the current source requires the file seam.)
//!
//! ## The observability seam — `ReportingVerifier`
//!
//! Sub-loop streaming is suppressed by design (the build and evaluate sub-runs
//! run with a `None` sink, exactly like PlanExecute). The ONE reliable seam to
//! watch the verify→revise cycle is the [`Verifier`] itself: the harness calls
//! `verify(&VerifierInput)` once per iteration, and [`VerifierInput`] carries the
//! **draft** (`build_result` output), the **critique** (`eval_result` output),
//! and the 0-indexed `iteration`. So we wrap [`EvaluatorResponseVerifier`] in a
//! small [`ReportingVerifier`] that prints, each iteration: a 1-based header with
//! the configured max, the draft, the critique, and the verdict — then delegates
//! the actual pass/fail decision to the inner verifier.
//!
//! `EvaluatorResponseVerifier` matches the evaluator's text against a `PASS`
//! pattern and a `FAIL: <reason>` pattern; if NEITHER matches it returns FAIL by
//! contract (default-to-FAIL is baked into the verifier and reinforced by the
//! harness's evaluator directive — "you did NOT write this code; default to FAIL
//! unless you can confirm it is right").
//!
//! There are no `// SPEC QUESTION:` markers: the tool decision, the verifier
//! construction, and the evaluator wiring were all resolved against the source.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull llama3.2
//! cargo run                       # default model llama3.2, 3 iterations
//! cargo run -- --max-iterations 5
//! cargo run -- --model qwen2.5-coder:7b
//! ```
//!
//! See the README for the honest rough-edges section: SelfVerifying against a
//! small local model is genuinely flaky (the evaluator may mis-judge, the loop
//! may exhaust without passing). A larger hosted model gives a cleaner demo.

use std::sync::Arc;

use spore_core::harness::BoxFut;
use spore_core::{
    BudgetLimits, EvaluatorResponseVerifier, ExecutionRegistry, HaltReason, Harness, HarnessBuilder,
    HarnessRunOptions, LoopStrategy, OllamaModelInterface, ReactConfig, RunResult, SchemaRef,
    SelfVerifyingConfig, SessionId, StandardTools, Task, Verifier, VerifierInput, VerifierVerdict,
    WorkspaceConfig, WorkspaceScopedSandbox,
};

/// The spec the agent must satisfy. It is the `Task` instruction, so the **build**
/// agent sees it directly, and — because the evaluate phase embeds the
/// `task.instruction` in the evaluator's directive — the **evaluator** sees the
/// exact same five criteria. One source of truth for both roles.
const TASK_PROMPT: &str = "\
Write a Rust function named `parse_int_list` and save it to the file \
`parse_int_list.rs` using the write_file tool. It must satisfy ALL of the \
following, which you will be graded on criterion-by-criterion:\n\
\n\
1. SIGNATURE: `pub fn parse_int_list(input: &str) -> Result<Vec<i32>, ParseIntListError>` \
where `ParseIntListError` is a custom error type you define in the same file.\n\
2. EDGE CASES: empty or whitespace-only input returns `Ok(vec![])`; whitespace \
around each number is tolerated (e.g. \" 1, 2 ,3 \" parses to [1, 2, 3]); a \
non-integer token returns `Err(...)` and NEVER panics.\n\
3. DOC COMMENTS: the function has `///` doc comments describing what it does.\n\
4. NO PANICS: no `unwrap()`, no `expect()`, no `panic!` anywhere in the code.\n\
5. USAGE EXAMPLE: include at least one usage example — either a ```rust doctest \
block in the doc comment or an inline `# Example` showing a call and its result.\n\
\n\
Write ONLY the file contents (valid Rust). Save it with write_file, then report \
that you are done.";

/// System prompt shared by the build agent and the evaluator agent (the harness
/// `system_prompt` is shared across both phases). It is deliberately role-neutral:
/// the build/evaluate framing is supplied per-phase (the build agent gets the
/// spec as its task; the evaluator gets the harness's built-in review directive
/// plus the same spec). It reinforces the file-tool contract and the evaluator's
/// default-to-FAIL posture.
const SYSTEM_PROMPT: &str = "\
You work on Rust code. Your only tools are write_file (save a file to the \
workspace) and read_file (read a file back). You have no shell and cannot run or \
compile code.\n\
\n\
When ASKED TO WRITE code: write the file with write_file, then say you are done.\n\
\n\
When ASKED TO REVIEW code: first read_file the file under review. Then check the \
work against EACH numbered criterion in the task, one at a time. You did NOT \
write this code — default to FAIL unless you can positively confirm every \
criterion holds. Respond with EXACTLY ONE verdict line as the LAST line of your \
reply:\n\
  - `PASS` if and only if every criterion holds, or\n\
  - `FAIL: <which criteria failed and why>` otherwise.\n\
Never emit PASS when unsure.";

/// A [`Verifier`] decorator: prints the verify→revise cycle to stdout, then
/// delegates the actual verdict to an inner verifier.
///
/// This is the one reliable observability seam for SelfVerifying — the build and
/// evaluate sub-runs are streamed with a suppressed sink, so the verifier call is
/// where the draft + critique + verdict become visible. Per iteration it prints:
/// a 1-based header with the configured max, the **draft** (`build_result`
/// output), the **critique** (`eval_result` output), and the **verdict**.
struct ReportingVerifier {
    inner: Arc<dyn Verifier>,
    max_iterations: u32,
}

impl ReportingVerifier {
    fn new(inner: Arc<dyn Verifier>, max_iterations: u32) -> Self {
        Self {
            inner,
            max_iterations,
        }
    }
}

impl Verifier for ReportingVerifier {
    fn verify<'a>(&'a self, input: &'a VerifierInput) -> BoxFut<'a, VerifierVerdict> {
        Box::pin(async move {
            // `iteration` is 0-indexed on the wire; display it 1-based.
            let n = input.iteration + 1;
            println!(
                "\n══════════════ iteration {n}/{} ══════════════",
                self.max_iterations
            );

            println!("\n── draft (what the agent wrote) ──");
            println!("{}", run_result_output(&input.build_result));

            println!("\n── evaluation (the critique) ──");
            println!("{}", run_result_output(&input.eval_result));

            // Delegate the actual decision to the inner verifier.
            let verdict = self.inner.verify(input).await;

            println!("\n── verdict ──");
            match &verdict {
                VerifierVerdict::Passed => println!("PASS — criteria satisfied; loop halts."),
                VerifierVerdict::Failed { reason } => {
                    println!("FAIL — {reason}");
                    if n < self.max_iterations {
                        println!("(reason injected into next build turn; revising…)");
                    } else {
                        println!("(no iterations left; loop will exhaust)");
                    }
                }
            }
            println!("════════════════════════════════════════════════");
            verdict
        })
    }

    fn max_iterations(&self) -> u32 {
        self.max_iterations
    }
}

/// Reduce a [`RunResult`] to printable text: the `Success` output, or a short
/// description of why the run did not complete.
fn run_result_output(r: &RunResult) -> String {
    match r {
        RunResult::Success { output, .. } => output.clone(),
        RunResult::Failure { reason, .. } => format!("<run did not complete: {reason:?}>"),
        RunResult::WaitingForHuman { .. } => "<run paused waiting for human>".to_string(),
        RunResult::Escalate { signal, .. } => format!("<run escalated: {signal:?}>"),
        // This example never wires a consult ladder; a top-level Consult is not
        // expected here, but the match must stay exhaustive against current core.
        RunResult::Consult { request, .. } => format!("<run paused on consult: {:?}>", request.kind),
    }
}

/// The worker (build-phase) output contract (`worker-schema`). Post-#119,
/// `SelfVerifying`'s `inner` (worker) slot is STRUCTURED: a bare `ReAct` there
/// must declare an `output` schema so its result is EVALUABLE
/// (`ExecutionRegistry::validate` enforces this via `check_structured_slot`).
/// The build agent writes `parse_int_list.rs`; this advertises the path it wrote.
fn worker_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "file": { "type": "string", "description": "Path the draft was written to." },
            "summary": { "type": "string", "description": "What was implemented." }
        },
        "required": ["file"]
    })
}

/// The registry the composed strategy's handles resolve against. Only the
/// `worker-schema` is EXPLICIT; the builder default-fills the empty agent /
/// toolset handles (`ReactConfig::per_loop`) AND the empty-key evaluator from the
/// single-collaborator setters (`.evaluator_agent(..)` / `.verifier(..)`) at
/// `build`. So the SelfVerifying `evaluator` stays the EMPTY handle (`""`).
fn build_registry() -> ExecutionRegistry {
    ExecutionRegistry::builder()
        .schema("worker-schema", worker_schema())
        .build()
}

/// The post-#119 composed strategy: `SelfVerifying(inner: ReAct, evaluator)`.
/// The worker leaf carries the `worker-schema` output contract (required for the
/// structured `worker` slot) and a `per_loop(max_iterations)` build budget. The
/// `evaluator` is the EMPTY handle (`""`), which the builder default-fills from
/// `.verifier(..)`. Old flat shape was the unit `LoopStrategy::SelfVerifying`.
fn self_verifying_strategy(max_iterations: u32) -> LoopStrategy {
    let worker = ReactConfig {
        output: Some(SchemaRef("worker-schema".to_string())),
        ..ReactConfig::per_loop(max_iterations)
    };
    LoopStrategy::SelfVerifying(SelfVerifyingConfig {
        inner: Box::new(LoopStrategy::ReAct(worker)),
        // The empty SchemaRef resolves to the default verifier the builder folds
        // in from `.verifier(..)` (#124 single-collaborator migration seam).
        evaluator: SchemaRef(String::new()),
        // Mirrors the shim default (`ReactConfig::per_loop` stamps `Escalate`):
        // a nested combinator propagates exhaustion to its parent.
        behavior: spore_core::BudgetExhaustedBehavior::Escalate,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    // Max iterations: CLI flag wins, then env var, then default 3.
    let max_iterations = arg_value(&args, "--max-iterations")
        .or_else(|| std::env::var("SPORE_MAX_ITERATIONS").ok())
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(3);

    let prompt = arg_value(&args, "--prompt").unwrap_or_else(|| TASK_PROMPT.to_string());

    // The agents operate inside this example's `workspace/` directory. Resolve it
    // relative to this source file so `cargo run` works from anywhere, and
    // canonicalize it — the sandbox requires a canonical, existing root.
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("workspace");
    std::fs::create_dir_all(&workspace_root)?;
    let workspace_root = std::fs::canonicalize(&workspace_root)?;

    // Post-#119/#124 the `.evaluator_agent(..)` single-collaborator seam is GONE.
    // The evaluate phase now defaults to the inner worker's agent (the empty-key
    // agent the builder default-fills from `conversational(model)`), running a
    // FRESH evaluator turn that the harness internally derives a read-only sandbox
    // for — so its `write_file` is blocked but `read_file` works, exactly what a
    // reviewer needs. The judging seam is the [`Verifier`] below (the SelfVerifying
    // `evaluator` handle resolves to it via the empty-key default fill).

    // The verifier: pattern-match the evaluator's text. `PASS` (word-boundaried,
    // case-insensitive) → Passed; `FAIL: <reason>` → Failed(reason); neither →
    // Failed by contract (default-to-FAIL). Wrapped in `ReportingVerifier` so the
    // cycle is visible. The inner verifier's `max_iterations` is what the harness
    // reads via the OUTER verifier — keep both equal to `max_iterations`.
    let inner =
        EvaluatorResponseVerifier::new(r"(?im)^\s*PASS\s*$", r"(?im)FAIL:\s*.+", max_iterations)?;
    let verifier: Arc<dyn Verifier> =
        Arc::new(ReportingVerifier::new(Arc::new(inner), max_iterations));

    // Build harness: conversational preset, workspace sandbox, the minimal file
    // tool set (write_file for the builder + read_file for the evaluator), shared
    // system prompt, the registry (carrying the `worker-schema` output contract),
    // and the verifier (folded into the default-key evaluator handle).
    let build_model = OllamaModelInterface::with_base_url(&model_id, base_url.clone());
    let sandbox = WorkspaceScopedSandbox::new(WorkspaceConfig::scoped(workspace_root.clone()))?;
    let harness = HarnessBuilder::conversational(build_model)
        .sandbox(Arc::new(sandbox))
        .registry(build_registry())
        .tool(StandardTools::write_file())
        .tool(StandardTools::read_file())
        .system_prompt(SYSTEM_PROMPT)
        .verifier(verifier)
        .build();

    // THE STRATEGY. There is no loop code below — the harness runs the
    // verify→revise cycle. A generous turn budget per build/evaluate sub-run lets
    // a small model take a few tool calls before claiming done.
    let task = Task::new(
        prompt.clone(),
        SessionId::generate(),
        self_verifying_strategy(max_iterations),
    )
    .with_budget(BudgetLimits {
        max_turns: Some(12),
        ..BudgetLimits::default()
    });

    println!("model         : {model_id}");
    println!("base url      : {base_url}");
    println!("workspace     : {}", workspace_root.display());
    println!("strategy      : SelfVerifying (draft → critique → revise)");
    println!("max iterations: {max_iterations}");
    println!(
        "verifier      : EvaluatorResponseVerifier (PASS / FAIL:) wrapped in ReportingVerifier"
    );
    println!("\nThe agent will draft `parse_int_list`, an evaluator will critique it against the");
    println!("five spec criteria, and the loop revises until PASS or {max_iterations} iterations elapse.\n");

    let draft_path = workspace_root.join("parse_int_list.rs");
    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Success { turns, .. } => {
            println!("\n✓ PASSED — the evaluator accepted the draft (after at most {max_iterations} iteration(s), {turns} build turn(s) total).");
            if let Ok(code) = std::fs::read_to_string(&draft_path) {
                println!("\n── final function ({}) ──\n{code}", draft_path.display());
            }
            Ok(())
        }
        RunResult::Failure {
            reason:
                HaltReason::SelfVerifyExhausted {
                    iterations,
                    last_reason,
                },
            ..
        } => {
            println!("\n✗ EXHAUSTED — {iterations} iteration(s) elapsed without a PASS.");
            println!("last failure reason: {last_reason}");
            if let Ok(code) = std::fs::read_to_string(&draft_path) {
                println!(
                    "\n── last draft on disk ({}) ──\n{code}",
                    draft_path.display()
                );
            }
            println!(
                "\nThis is an expected rough edge with small local models — see the README. \
                 Try a larger model or raise --max-iterations."
            );
            std::process::exit(1);
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

// ============================================================================
// Example-crate test (NO model): the composed SelfVerifying strategy resolves
// against the example's registry — the regression guard that the post-#119
// strategy tree stays validation-clean against current core.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use spore_core::{AgentId, EmptyToolRegistry, ModelAgent};

    /// AC: the composed `SelfVerifying(inner: ReAct{worker-schema}, evaluator: "")`
    /// tree validates — the worker slot's output schema resolves (structured-slot
    /// contract) and the empty-handle evaluator resolves to the default-filled
    /// verifier. The leaves use EMPTY agent/toolset/evaluator handles that
    /// `HarnessBuilder::build_config` default-fills at `build`; here we mirror
    /// that fill (empty-key agent + toolset + verifier) so the standalone
    /// registry validates exactly as the assembled harness would.
    #[test]
    fn registry_validates() {
        let model = Arc::new(OllamaModelInterface::with_base_url(
            "gemma4:e4b",
            "http://localhost:11434".to_string(),
        ));
        let verifier: Arc<dyn Verifier> = Arc::new(
            EvaluatorResponseVerifier::new(r"(?im)^\s*PASS\s*$", r"(?im)FAIL:\s*.+", 3).unwrap(),
        );
        let registry = build_registry()
            .into_builder()
            .fill_default_agent(Arc::new(ModelAgent::new(AgentId::new("default"), model)))
            .fill_default_toolset(Arc::new(EmptyToolRegistry))
            .fill_default_verifier(verifier)
            .build();
        let task = Task::new(
            "draft parse_int_list".to_string(),
            SessionId::generate(),
            self_verifying_strategy(3),
        );
        assert!(
            registry.validate(&task).is_ok(),
            "the composed SelfVerifying strategy must validate against the registry"
        );
    }
}
