//! spore-core example 10 — the `HillClimbing` loop strategy.
//!
//! ## What this example demonstrates
//!
//! **Iterative refinement under a scoring oracle is a harness concern, not
//! application logic.** The agent edits ONE file in place (`workspace/README.md`)
//! across iterations. After every iteration a custom [`MetricEvaluator`] reads
//! that file and asks a *separate judge model* to score it on three dimensions —
//! Clarity, Completeness, Example quality (0–10 each) — returning the total/30
//! normalized to `[0,1]`. The harness applies [`should_keep`](spore_core::should_keep):
//! a strictly-better score is KEPT; anything else is DISCARDED and (because
//! `revert_on_no_improvement` is on) the workspace is `git reset --hard`-ed back
//! to the best-so-far. The loop halts on **stagnation** (`MAX_STAGNATION`
//! consecutive non-improvements) or **budget** (`max_turns`). You write **no loop
//! code** — you wire a strategy, a metric evaluator, and an observability sink.
//!
//! Post-#119 the strategy is a composed tree, not the old struct-variant literal:
//!
//! ```text
//! // pre-#119:
//! LoopStrategy::HillClimbing {
//!     direction: OptimizationDirection::Maximize,
//!     max_stagnation: Some(MAX_STAGNATION),
//!     revert_on_no_improvement: true,
//!     min_improvement_delta: None,
//! }
//! // post-#119:
//! LoopStrategy::HillClimbing(HillClimbingConfig {
//!     inner: ReAct { output: Some("propose-schema"), budget: PerLoop(PER_ITER_BUDGET) },
//!     direction: HillClimbingDirection::Maximize,   // renamed from OptimizationDirection
//!     max_stagnation: MAX_STAGNATION,               // now a bare u32 (was Some(_))
//!     revert_on_no_improvement: true,
//!     min_improvement_delta: 0.0,                   // now a bare f64 (was None)
//!     evaluator: "",                                // empty handle ⇒ `.metric_evaluator(..)`
//!     behavior: Escalate,
//! })
//! ```
//!
//! The `inner` (propose) slot is STRUCTURED — a bare `ReAct` there MUST declare an
//! `output` schema (here `propose-schema`) so each iteration yields a scorable
//! candidate; `ExecutionRegistry::validate` enforces this at run entry. The
//! `evaluator` stays the EMPTY handle (`""`), default-filled from the
//! single-collaborator `.metric_evaluator(..)` setter — so the only handle
//! registered explicitly is `propose-schema`.
//!
//! ## The contrast with example 09 (SelfVerifying) — the teaching point
//!
//! 09 has a **binary exit condition**: a [`Verifier`](spore_core::Verifier)
//! returns PASS and the loop *succeeds*, or it exhausts and *fails*. HillClimbing
//! has **no PASS**. It is an optimization loop: there is only *best-so-far*. It
//! does not know it is "done" — it only knows it has stopped improving. The
//! terminal outcome is therefore a [`HaltReason::StagnationLimitReached`] or
//! [`HaltReason::BudgetExceeded`], NOT a success/fail verdict on quality.
//!
//! ## SPEC NOTE — why this diverges from issue #99's original framing (Option A)
//!
//! The original issue asked the agent to "climb until total ≥ 25/30 or max
//! iterations". Planning (#99 spec-resolution comment) established that framing
//! does NOT match the real `HillClimbing` strategy in spore-core:
//!   - There is no score-threshold success condition. The loop keeps/reverts on
//!     *relative* improvement and halts on stagnation/budget — it never compares
//!     the metric against an absolute target.
//!   - `MAX_ITERATIONS` is not a HillClimbing parameter; iterations are bounded by
//!     [`BudgetLimits::max_turns`]. The `MAX_ITERATIONS` constant maps there.
//!   - The shipped [`LlmJudgeEvaluator`](spore_core::LlmJudgeEvaluator) scores a
//!     FIXED construction-time string, so it cannot see the evolving draft. This
//!     example therefore ships a small example-local `MetricEvaluator`
//!     ([`ReadmeQualityEvaluator`]) that reads `workspace/README.md` through the
//!     sandbox each iteration before scoring.
//!
//! Resolution = **Option A** (reframe to real semantics, no core change):
//!   - `SCORE_THRESHOLD` (25/30) is kept as a **DISPLAY annotation only**. When a
//!     draft's total crosses it, the printed line is marked `★ crossed target
//!     threshold`. It does **not** terminate the loop. // SPEC NOTE: display-only.
//!   - The per-iteration print is split across two seams, mirroring how the
//!     harness actually exposes the run:
//!       * the evaluator prints the draft + 3 sub-scores + total (it is the only
//!         place that sees the rubric breakdown), and
//!       * a custom [`ObservabilityProvider`] handling
//!         [`WarnEvent::HillClimbingIteration`] prints the kept/discarded/reverted
//!         decision (iteration, metric value, delta) — the harness emits exactly
//!         one such event per iteration.
//!
//! There are no `// SPEC QUESTION:` markers: every divergence above is resolved
//! against the source and the #99 resolution comment.
//!
//! ## Constants (see their doc comments below)
//!   - [`MAX_ITERATIONS`]  — maps to `BudgetLimits.max_turns` (default 6).
//!   - [`MAX_STAGNATION`]  — consecutive non-improvements before halt (2).
//!   - [`SCORE_THRESHOLD`] — DISPLAY annotation only (25). Never terminates.
//!   - [`DIMENSION_MAX`] / [`TOTAL_MAX`] — 10 per dimension, 30 total.
//!
//! ## The seams this example wires
//!   - [`ReadmeQualityEvaluator`]   — `impl MetricEvaluator`; reads the file via
//!     the [`SandboxProvider`], runs a fresh judge model call, prints the rubric.
//!   - [`ReportingObservability`]   — `impl ObservabilityProvider`; wraps an
//!     in-memory provider and prints each `HillClimbingIteration` decision.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull llama3.2
//! cargo run                       # default model llama3.2, 6-iteration budget
//! cargo run -- --max-iterations 8
//! cargo run -- --model qwen2.5-coder:7b
//! ```
//!
//! See the README for the honest rough-edges section.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use spore_core::harness::BoxFut;
use spore_core::observability::{WarnEvent, WarnSpan};
use spore_core::{
    BudgetLimits, Content, ContentBlock, ExecutionRegistry, HaltReason, Harness, HarnessBuilder,
    HarnessRunOptions, HillClimbingConfig, HillClimbingDirection, InMemoryObservabilityProvider,
    LoopStrategy, Message, MetricError, MetricEvaluator, MetricResult, ModelInterface, ModelParams,
    ModelRequest, ObservabilityProvider, OllamaModelInterface, ReactConfig, Role, RunResult,
    SchemaRef, SessionId, SessionMetrics, SessionOutcome, SessionStateSnapshot, Span, StandardTools,
    Task, Timestamp, WorkspaceConfig, WorkspaceScopedSandbox,
};

// ============================================================================
// Constants — all clearly named, with the spec semantics in their doc comments.
// ============================================================================

/// Climbing-iteration ceiling. Maps to [`BudgetLimits::max_turns`] — this is the
/// BUDGET, not a success target. The loop may halt EARLIER on stagnation. There
/// is no "reached the goal" outcome; HillClimbing always halts on budget or
/// stagnation. Six gives a small local model room to make a few real edits.
const MAX_ITERATIONS: u32 = 6;

/// Consecutive non-improvements tolerated before the loop halts with
/// [`HaltReason::StagnationLimitReached`]. The stagnation counter resets to 0 on
/// any kept (strictly-improving) iteration. Post-#119 maps to the bare
/// `HillClimbingConfig.max_stagnation: u32` (was `Some(_)`).
const MAX_STAGNATION: u32 = 2;

/// Per-iteration build budget for the propose leaf. Post-#119, HillClimbing's
/// `inner` is a composed `ReAct(ReactConfig::per_loop(PER_ITER_BUDGET))` — this
/// caps how many tool-using turns the build agent gets to PRODUCE ONE candidate
/// draft per climb step (distinct from `MAX_ITERATIONS`, the count of climb
/// steps, which maps to `BudgetLimits::max_turns`).
const PER_ITER_BUDGET: u32 = 8;

/// DISPLAY ANNOTATION ONLY. When a draft's total score (0–30) reaches this, the
/// evaluator marks the printed line `★ crossed target threshold`. // SPEC NOTE:
/// this does NOT terminate the loop — HillClimbing has no score-threshold exit.
const SCORE_THRESHOLD: u32 = 25;

/// Max score per rubric dimension (Clarity, Completeness, Example quality).
const DIMENSION_MAX: u32 = 10;

/// Max total across the three dimensions (`3 * DIMENSION_MAX`).
const TOTAL_MAX: u32 = 3 * DIMENSION_MAX;

/// The file under refinement, relative to the workspace root. The build agent
/// edits this in place; the evaluator reads it back through the sandbox.
const DRAFT_FILE: &str = "README.md";

/// The task the build agent is asked to perform each iteration. It edits ONE file
/// in place — the climb is over successive revisions of the same README.
const TASK_PROMPT: &str = "\
You are writing the README.md for a fictional Rust crate called `ironwood`, a \
small library for parsing and validating semantic-version strings. Use the \
write_file tool to save your README to `README.md`. If a `README.md` already \
exists, first read_file it, then improve it and write it back.\n\
\n\
A great README for this crate has THREE qualities, each scored 0–10 by a \
reviewer:\n\
  1. CLARITY: a crisp one-line summary, then prose a newcomer can follow.\n\
  2. COMPLETENESS: what the crate does, how to add it to Cargo.toml, the main \
API surface, and an error/edge-cases note.\n\
  3. EXAMPLE QUALITY: at least one fenced ```rust code block showing a real call \
and its expected result.\n\
\n\
Write the BEST README you can, then report that you are done.";

/// System prompt shared by the build agent. (The judge model is prompted
/// separately by the evaluator; it does not share this prompt.) Reinforces the
/// minimal file-tool contract.
const SYSTEM_PROMPT: &str = "\
You write developer documentation in Markdown. Your only tools are write_file \
(save a file to the workspace) and read_file (read a file back). You have no \
shell and cannot run code. When asked to write or improve the README: read any \
existing file first, write the improved Markdown with write_file, then say you \
are done.";

/// The rubric handed to the judge model. Kept separate from the build prompt so
/// the judge scores independently of how the writer was instructed.
const JUDGE_RUBRIC: &str = "\
You are a strict technical-documentation reviewer. Score the README below on \
THREE dimensions, each an integer from 0 to 10:\n\
  - CLARITY: is there a crisp one-line summary and prose a newcomer can follow?\n\
  - COMPLETENESS: does it cover what the crate does, how to add it to \
Cargo.toml, the main API, and an error/edge-cases note?\n\
  - EXAMPLE_QUALITY: is there at least one fenced ```rust block with a real call \
and expected result?\n\
\n\
Reply with EXACTLY these three lines and nothing else:\n\
clarity: <0-10>\n\
completeness: <0-10>\n\
example_quality: <0-10>";

// ============================================================================
// ReadmeQualityEvaluator — the example-local `MetricEvaluator`.
// ============================================================================

/// Scores `workspace/README.md` by reading it through the [`SandboxProvider`]
/// then making a SEPARATE judge-model call that returns three sub-scores. The
/// value reported to the harness is `total / TOTAL_MAX`, normalized to `[0,1]`,
/// with `direction = Maximize`.
///
/// SPEC NOTE: this replaces the shipped [`LlmJudgeEvaluator`], which scores a
/// fixed construction-time string and so cannot observe the evolving draft.
struct ReadmeQualityEvaluator<M: ModelInterface> {
    judge: Arc<M>,
}

impl<M: ModelInterface> ReadmeQualityEvaluator<M> {
    fn new(judge: Arc<M>) -> Self {
        Self { judge }
    }

    /// Parse a `name: <int>` line, clamped to `[0, DIMENSION_MAX]`. A missing or
    /// unparseable line scores 0 — a malformed judge reply must not crash the run
    /// (it just reads as a poor score, which the loop treats as a normal outcome).
    fn parse_dimension(text: &str, name: &str) -> u32 {
        text.lines()
            .find_map(|line| {
                let lower = line.trim().to_lowercase();
                let prefix = format!("{name}:");
                lower.strip_prefix(&prefix).and_then(|rest| {
                    rest.split_whitespace()
                        .next()
                        .and_then(|tok| tok.parse::<u32>().ok())
                })
            })
            .unwrap_or(0)
            .min(DIMENSION_MAX)
    }
}

impl<M: ModelInterface + 'static> MetricEvaluator for ReadmeQualityEvaluator<M> {
    fn evaluate<'a>(
        &'a self,
        sandbox: &'a dyn spore_core::SandboxProvider,
        _session_state: &'a SessionStateSnapshot,
    ) -> BoxFut<'a, Result<MetricResult, MetricError>> {
        Box::pin(async move {
            let start = Instant::now();

            // Read the current draft through the sandbox root, exactly as the core
            // evaluators do. A missing draft (e.g. the baseline before the agent
            // has written anything) scores 0 rather than erroring.
            let draft_path = sandbox.workspace_root().join(DRAFT_FILE);
            let draft = tokio::fs::read_to_string(&draft_path)
                .await
                .unwrap_or_default();

            let (clarity, completeness, example, total);
            if draft.trim().is_empty() {
                clarity = 0;
                completeness = 0;
                example = 0;
                total = 0;
                println!(
                    "\n── evaluator: no draft on disk yet (baseline) — total 0/{TOTAL_MAX} ──"
                );
            } else {
                let prompt = format!("{JUDGE_RUBRIC}\n\n----- README under review -----\n{draft}");
                let request = ModelRequest {
                    messages: vec![Message {
                        role: Role::User,
                        content: Content::Text { text: prompt },
                    }],
                    tools: Vec::new(),
                    params: ModelParams::default(),
                    stream: false,
                };
                let response =
                    self.judge
                        .call(request)
                        .await
                        .map_err(|e| MetricError::ExecutionFailed {
                            reason: format!("judge model call failed: {e}"),
                        })?;
                let text = response
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                clarity = Self::parse_dimension(&text, "clarity");
                completeness = Self::parse_dimension(&text, "completeness");
                example = Self::parse_dimension(&text, "example_quality");
                total = clarity + completeness + example;

                println!("\n── evaluator: scored draft ({} bytes) ──", draft.len());
                println!("{draft}");
                println!("  clarity        : {clarity}/{DIMENSION_MAX}");
                println!("  completeness   : {completeness}/{DIMENSION_MAX}");
                println!("  example quality: {example}/{DIMENSION_MAX}");
            }

            // SPEC NOTE: the threshold is DISPLAY-ONLY. We annotate the line; we do
            // NOT halt the loop here. The harness halts on stagnation/budget.
            let crossed = if total >= SCORE_THRESHOLD {
                "  ★ crossed target threshold"
            } else {
                ""
            };
            println!("  TOTAL          : {total}/{TOTAL_MAX}{crossed}");

            let value = f64::from(total) / f64::from(TOTAL_MAX);
            let mut metadata = BTreeMap::new();
            metadata.insert("clarity".into(), clarity.to_string());
            metadata.insert("completeness".into(), completeness.to_string());
            metadata.insert("example_quality".into(), example.to_string());
            metadata.insert("total".into(), total.to_string());
            Ok(MetricResult {
                value,
                raw_output: draft,
                duration: start.elapsed(),
                metadata,
            })
        })
    }

    fn direction(&self) -> HillClimbingDirection {
        HillClimbingDirection::Maximize
    }

    fn description(&self) -> String {
        format!("ironwood README quality (clarity+completeness+example, /{TOTAL_MAX})")
    }
}

// ============================================================================
// ReportingObservability — prints each `HillClimbingIteration` decision.
// ============================================================================

/// An [`ObservabilityProvider`] that delegates everything to an inner
/// [`InMemoryObservabilityProvider`] but additionally PRINTS each
/// [`WarnEvent::HillClimbingIteration`]. This is the seam the harness uses to
/// report the per-iteration keep/revert decision — the evaluator prints the
/// scores, this prints what the loop DID with them.
struct ReportingObservability {
    inner: Arc<InMemoryObservabilityProvider>,
}

impl ReportingObservability {
    fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryObservabilityProvider::new()),
        }
    }
}

impl ObservabilityProvider for ReportingObservability {
    fn emit_warn(&self, span: WarnSpan) {
        if let WarnEvent::HillClimbingIteration {
            iteration,
            metric_value,
            delta,
            status,
            reverted,
        } = &span.event
        {
            // `iteration` is 0-based on the wire (0 = baseline). Display 1-based.
            let n = iteration + 1;
            let value = metric_value
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "n/a".to_string());
            let delta_str = delta
                .map(|d| format!("{d:+.3}"))
                .unwrap_or_else(|| "—".to_string());
            let reverted_note = if *reverted {
                " (workspace git-reset to best-so-far)"
            } else {
                ""
            };
            println!(
                "\n══ iteration {n}/{MAX_ITERATIONS} — {status} ══  metric={value} (Δ {delta_str}){reverted_note}"
            );
        }
        self.inner.emit_warn(span);
    }

    // ---- everything else delegates verbatim to the in-memory provider ----
    fn emit_turn(&self, span: spore_core::TurnSpan) {
        self.inner.emit_turn(span);
    }
    fn emit_tool_call(&self, span: spore_core::ToolCallSpan) {
        self.inner.emit_tool_call(span);
    }
    fn emit_sensor(&self, span: spore_core::SensorSpan) {
        self.inner.emit_sensor(span);
    }
    fn emit_context(&self, span: spore_core::ContextSpan) {
        self.inner.emit_context(span);
    }
    fn emit_middleware(&self, span: spore_core::MiddlewareSpan) {
        self.inner.emit_middleware(span);
    }
    fn emit_patch(&self, span: spore_core::PatchSpan) {
        self.inner.emit_patch(span);
    }
    fn set_session_outcome(&self, session_id: &SessionId, outcome: SessionOutcome) {
        self.inner.set_session_outcome(session_id, outcome);
    }
    fn flush_session<'a>(&'a self, session_id: &'a SessionId) -> BoxFut<'a, ()> {
        self.inner.flush_session(session_id)
    }
    fn get_session_metrics<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Option<SessionMetrics>> {
        self.inner.get_session_metrics(session_id)
    }
    fn get_sessions<'a>(
        &'a self,
        since: Timestamp,
        domain: Option<String>,
        outcome: Option<SessionOutcome>,
    ) -> BoxFut<'a, Vec<SessionMetrics>> {
        self.inner.get_sessions(since, domain, outcome)
    }
    fn get_trace<'a>(&'a self, session_id: &'a SessionId) -> BoxFut<'a, Vec<Box<dyn Span>>> {
        self.inner.get_trace(session_id)
    }
}

// ============================================================================
// Strategy assembly (post-#119 composed tree)
// ============================================================================

/// The propose-phase output contract (`propose-schema`). Post-#119,
/// `HillClimbing`'s `inner` (propose) slot is STRUCTURED: a bare `ReAct` there
/// must declare an `output` schema so each iteration yields a scorable candidate
/// (`ExecutionRegistry::validate` enforces this via `check_structured_slot`). The
/// build agent rewrites `DRAFT_FILE`; this advertises the path it wrote.
fn propose_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "file": { "type": "string", "description": "Path the candidate draft was written to." },
            "summary": { "type": "string", "description": "What this iteration changed." }
        },
        "required": ["file"]
    })
}

/// The registry the composed strategy's handles resolve against. Only the
/// `propose-schema` is EXPLICIT; the builder default-fills the empty agent /
/// toolset handles (`ReactConfig::per_loop`) AND the empty-key metric evaluator
/// from `.metric_evaluator(..)` at `build`. So the HillClimbing `evaluator` stays
/// the EMPTY handle (`""`).
fn build_registry() -> ExecutionRegistry {
    ExecutionRegistry::builder()
        .schema("propose-schema", propose_schema())
        .build()
}

/// The post-#119 composed strategy: `HillClimbing(inner: ReAct, evaluator)`. The
/// propose leaf carries the `propose-schema` output contract (required for the
/// structured `propose` slot) and a `per_loop(per_iter_budget)` build budget. The
/// `evaluator` is the EMPTY handle (`""`), which the builder default-fills from
/// `.metric_evaluator(..)`. `max_stagnation` / `min_improvement_delta` are now
/// BARE (`u32` / `f64`), not `Option`. Old flat shape was the struct-variant
/// `LoopStrategy::HillClimbing { direction, max_stagnation: Some(_), .. }`.
fn hill_climbing_strategy(per_iter_budget: u32) -> LoopStrategy {
    let propose = ReactConfig {
        output: Some(SchemaRef("propose-schema".to_string())),
        ..ReactConfig::per_loop(per_iter_budget)
    };
    LoopStrategy::HillClimbing(HillClimbingConfig {
        inner: Box::new(LoopStrategy::ReAct(propose)),
        direction: HillClimbingDirection::Maximize,
        max_stagnation: MAX_STAGNATION,
        revert_on_no_improvement: true,
        min_improvement_delta: 0.0,
        // Empty AgentRef ⇒ default-filled metric evaluator from `.metric_evaluator(..)`.
        evaluator: spore_core::AgentRef(String::new()),
        // Mirrors the shim default: a nested combinator propagates exhaustion up.
        behavior: spore_core::BudgetExhaustedBehavior::Escalate,
    })
}

// ============================================================================
// main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    // Iteration budget: CLI flag wins, then env var, then MAX_ITERATIONS.
    let max_iterations = arg_value(&args, "--max-iterations")
        .or_else(|| std::env::var("SPORE_MAX_ITERATIONS").ok())
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(MAX_ITERATIONS);

    let prompt = arg_value(&args, "--prompt").unwrap_or_else(|| TASK_PROMPT.to_string());

    // The agent edits this example's `workspace/` in place. Resolve relative to
    // the source file so `cargo run` works from anywhere, and canonicalize it —
    // the sandbox requires a canonical, existing root.
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("workspace");
    std::fs::create_dir_all(&workspace_root)?;
    let workspace_root = std::fs::canonicalize(&workspace_root)?;

    // git-init the workspace so `revert_on_no_improvement`'s `git reset --hard`
    // has a clean baseline to return to. Idempotent: skip if already a repo.
    init_git_workspace(&workspace_root)?;

    // Two model instances on the same Ollama endpoint: one drives the build agent
    // (writing the README), one is the judge the evaluator calls to score it.
    let build_model = OllamaModelInterface::with_base_url(&model_id, base_url.clone());
    let judge_model = Arc::new(OllamaModelInterface::with_base_url(
        &model_id,
        base_url.clone(),
    ));
    let evaluator: Arc<dyn MetricEvaluator> = Arc::new(ReadmeQualityEvaluator::new(judge_model));

    let observability: Arc<dyn ObservabilityProvider> = Arc::new(ReportingObservability::new());

    // Build harness: conversational preset, workspace sandbox, the minimal file
    // tool set (write_file + read_file), shared system prompt, the metric
    // evaluator (required for HillClimbing), and the observability sink.
    let sandbox = WorkspaceScopedSandbox::new(WorkspaceConfig::scoped(workspace_root.clone()))?;
    let harness = HarnessBuilder::conversational(build_model)
        .sandbox(Arc::new(sandbox))
        .registry(build_registry())
        .tool(StandardTools::write_file())
        .tool(StandardTools::read_file())
        .system_prompt(SYSTEM_PROMPT)
        .metric_evaluator(evaluator)
        .observability(observability)
        .build();

    // THE STRATEGY. No loop code below — the harness runs the climb. The composed
    // `HillClimbing(inner: ReAct{propose-schema}, evaluator)` tree (post-#119)
    // gives the propose leaf a per-iteration build budget (`PER_ITER_BUDGET`);
    // `max_turns` bounds the NUMBER OF ITERATIONS (the budget ceiling), and
    // `max_stagnation` can halt sooner. SPEC NOTE: there is no score-threshold
    // field — by design.
    let task = Task::new(
        prompt.clone(),
        SessionId::generate(),
        hill_climbing_strategy(PER_ITER_BUDGET),
    )
    .with_budget(BudgetLimits {
        max_turns: Some(max_iterations),
        ..BudgetLimits::default()
    });

    println!("model         : {model_id}");
    println!("base url      : {base_url}");
    println!("workspace     : {}", workspace_root.display());
    println!("strategy      : HillClimbing (score → keep/revert → climb)");
    println!("direction     : Maximize (higher README score is better)");
    println!("max iterations: {max_iterations} (budget ceiling — NOT a success target)");
    println!("max stagnation: {MAX_STAGNATION} (halt after this many non-improvements)");
    println!(
        "threshold     : {SCORE_THRESHOLD}/{TOTAL_MAX} — DISPLAY ONLY (★ marks it; never halts)"
    );
    println!("\nThe agent will draft and refine `{DRAFT_FILE}`; each iteration a judge model");
    println!("scores it on three dimensions, and the loop keeps the best — reverting the rest —");
    println!("until it stops improving (stagnation) or the budget is spent. There is no PASS.\n");

    let draft_path = workspace_root.join(DRAFT_FILE);
    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Failure {
            reason:
                HaltReason::StagnationLimitReached {
                    iterations,
                    best_metric,
                },
            ..
        } => {
            report_best(best_metric, &draft_path);
            println!(
                "\n■ HALTED ON STAGNATION — {iterations} consecutive non-improving iteration(s)."
            );
            println!("This is the NORMAL terminal outcome for HillClimbing: it stopped because it");
            println!(
                "could not improve, not because it hit a target. The file on disk is best-so-far."
            );
            Ok(())
        }
        RunResult::Failure {
            reason: HaltReason::BudgetExceeded { limit_type },
            ..
        } => {
            report_best(f64::NAN, &draft_path);
            println!("\n■ HALTED ON BUDGET — exhausted the iteration ceiling ({limit_type:?}).");
            println!("Also a normal terminal outcome: the climb ran out of budget while still");
            println!("(possibly) improving. The file on disk is the best-so-far draft.");
            Ok(())
        }
        RunResult::Failure {
            reason: HaltReason::HillClimbingMisconfigured { reason },
            ..
        } => {
            eprintln!("\nHillClimbing misconfigured: {reason}");
            std::process::exit(1);
        }
        RunResult::Success { turns, .. } => {
            // HillClimbing does not normally return Success (it has no success
            // condition); surface it honestly if a future core revision does.
            report_best(f64::NAN, &draft_path);
            println!("\n■ run returned Success after {turns} turn(s) — best-so-far draft on disk.");
            Ok(())
        }
        other => {
            eprintln!("\nrun did not complete as expected: {other:?}");
            std::process::exit(1);
        }
    }
}

/// Print the best-so-far metric (when known) and the final draft on disk.
fn report_best(best_metric: f64, draft_path: &Path) {
    if best_metric.is_finite() {
        let total = (best_metric * f64::from(TOTAL_MAX)).round() as u32;
        println!("\n── best score seen: {total}/{TOTAL_MAX} (normalized {best_metric:.3}) ──");
    }
    match std::fs::read_to_string(draft_path) {
        Ok(code) => println!("\n── final draft ({}) ──\n{code}", draft_path.display()),
        Err(_) => println!("\n(no draft was written to {})", draft_path.display()),
    }
}

/// `git init` the workspace and make an initial commit if it is not already a
/// repo, so `revert_on_no_improvement`'s `git reset --hard` has a baseline.
/// Best-effort and idempotent: a missing `git` or an existing repo is fine.
fn init_git_workspace(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if root.join(".git").exists() {
        return Ok(());
    }
    let run = |args: &[&str]| -> std::io::Result<()> {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()?;
        if !status.success() {
            return Err(std::io::Error::other(format!("git {args:?} failed")));
        }
        Ok(())
    };
    run(&["init"])?;
    // Local identity so the initial commit succeeds without global git config.
    run(&["config", "user.email", "example@spore-core.invalid"])?;
    run(&["config", "user.name", "spore-core example"])?;
    run(&["add", "-A"])?;
    // An empty initial commit is fine if the dir is otherwise empty.
    run(&["commit", "--allow-empty", "-m", "baseline"])?;
    Ok(())
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

// ============================================================================
// Example-crate test (NO model): the composed HillClimbing strategy resolves
// against the example's registry — the regression guard that the post-#119
// strategy tree stays validation-clean against current core.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use spore_core::{AgentId, EmptyToolRegistry, ModelAgent};

    /// AC: the composed `HillClimbing(inner: ReAct{propose-schema}, evaluator: "")`
    /// tree validates — the propose slot's output schema resolves (structured-slot
    /// contract) and the empty-handle evaluator resolves to the default-filled
    /// metric evaluator. The leaves use EMPTY agent/toolset/evaluator handles that
    /// `HarnessBuilder::build_config` default-fills at `build`; here we mirror that
    /// fill (empty-key agent + toolset + metric evaluator) so the standalone
    /// registry validates exactly as the assembled harness would.
    #[test]
    fn registry_validates() {
        let model = Arc::new(OllamaModelInterface::with_base_url(
            "gemma4:e4b",
            "http://localhost:11434".to_string(),
        ));
        let metric: Arc<dyn MetricEvaluator> =
            Arc::new(ReadmeQualityEvaluator::new(model.clone()));
        let registry = build_registry()
            .into_builder()
            .fill_default_agent(Arc::new(ModelAgent::new(AgentId::new("default"), model)))
            .fill_default_toolset(Arc::new(EmptyToolRegistry))
            .fill_default_metric_evaluator(metric)
            .build();
        let task = Task::new(
            "refine the README".to_string(),
            SessionId::generate(),
            hill_climbing_strategy(PER_ITER_BUDGET),
        );
        assert!(
            registry.validate(&task).is_ok(),
            "the composed HillClimbing strategy must validate against the registry"
        );
    }
}
