//! The example's custom tools. After the #131 composition rewrite the only
//! hand-written tool that survives is `send_user_message` (observability — part
//! of `exec-tools`). The #114 consult ladder (`research_best_practices`,
//! `consult_advisor`) and the #115 `load_skill` tool were SubagentTool /
//! per-node seams that the declarative strategy tree does not expose, so they
//! were dropped (see the README's "what changed" note).

pub mod send_message;
