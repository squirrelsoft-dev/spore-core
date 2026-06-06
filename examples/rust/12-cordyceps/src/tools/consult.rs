//! The two consult tools the analysis worker calls to escalate mid-loop
//! (issue #114). Both lower to [`ToolOutput::consult`] with a `kind` tag; the
//! `analysis_worker` [`SubagentTool`](spore_core::tools::SubagentTool) mediates
//! by `kind` (research → research_worker, advice → advisor) using the
//! per-kind budgets + overflow policies installed via `with_consult_handlers`.
//!
//! Neither tool captures any host state — each simply renders its call input
//! into a [`ConsultRequest`] and returns [`ToolOutput::consult`]. The worker
//! harness pauses (`RunResult::Consult`) and `SubagentTool` resumes it with the
//! handler's answer (or a `BudgetExhausted` message). So these are defined with
//! the [`tool!`](spore_core::tool) macro — no closed-over `Arc` needed.

use schemars::JsonSchema;
use serde::Deserialize;

use spore_core::harness::{ConsultRequest, ToolOutput};
use spore_core::{tool, StandardTool};

/// Routing key for the research consult ladder (→ research_worker, web_search).
pub const KIND_RESEARCH: &str = "research";
/// Routing key for the advice consult ladder (→ advisor, cloud model).
pub const KIND_ADVICE: &str = "advice";

/// Shared input shape for both consult tools: the worker describes where it is
/// stuck and the concrete question it wants answered. `attempts` is advisory —
/// the harness enforces the per-kind budget independently.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ConsultInput {
    /// Free-form description of where you are stuck or uncertain.
    pub situation: String,
    /// The concrete question you want answered.
    pub question: String,
    /// How many times you have already tried (advisory only).
    #[serde(default)]
    pub attempts: u32,
}

/// `research_best_practices` → `kind="research"`. Routed to the research
/// worker (web_search). Budget 5, overflow `SoftFail`: on exhaustion the worker
/// resumes with `BudgetExhausted` and finishes on general knowledge. Looking up
/// an idiom is normal, not distress, so it never reaches the human.
pub fn research_best_practices_tool() -> StandardTool {
    tool! {
        name: "research_best_practices",
        description: "Ask a research helper to web-search current best practices or idioms when \
                      you are unsure whether a pattern is a real defect. Pass `situation` and a \
                      focused `question`. Returns cited findings; use sparingly.",
        input: ConsultInput,
        execute: |input, _sandbox, _ctx| async move {
            ToolOutput::consult(ConsultRequest {
                kind: KIND_RESEARCH.to_string(),
                situation: input.situation,
                attempts: input.attempts,
                question: input.question,
            })
        },
    }
}

/// `consult_advisor` → `kind="advice"`. Routed to the advisor (near-frontier
/// cloud model with `read_file`/`grep`). Budget 3, overflow `EscalateToHuman`:
/// on exhaustion the consult converts to `RunResult::WaitingForHuman` and the
/// REPL surfaces the three-choice ladder.
pub fn consult_advisor_tool() -> StandardTool {
    tool! {
        name: "consult_advisor",
        description: "Ask a senior advisor agent (a stronger model that can read_file/grep the \
                      repo) when you are stuck on whether a finding is real or how to rank its \
                      severity. Pass `situation` and a concrete `question`. Reserve for genuine \
                      uncertainty — the advisor budget is small.",
        input: ConsultInput,
        execute: |input, _sandbox, _ctx| async move {
            ToolOutput::consult(ConsultRequest {
                kind: KIND_ADVICE.to_string(),
                situation: input.situation,
                attempts: input.attempts,
                question: input.question,
            })
        },
    }
}
