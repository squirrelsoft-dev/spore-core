//! The example's custom tools. After the #131 composition rewrite the surviving
//! hand-written tools are `send_user_message` (observability) and the #114
//! consult ladder (`research_best_practices`, `consult_advisor`) — both part of
//! `exec-tools`. The consult ladder is PRESERVED: the worker leaf consult now
//! propagates to a top-level `RunResult::Consult` and the host run loop mediates
//! it (the seam moved from the old `SubagentTool` to the host loop; see
//! `consult.rs` and `main.rs`). The #115 `load_skill` tool was a worker-side
//! per-node seam the declarative tree does not expose, so it was dropped — the
//! `audit` skill instead rides the GLOBAL `SkillInjectingContextManager`.

pub mod consult;
pub mod send_message;
