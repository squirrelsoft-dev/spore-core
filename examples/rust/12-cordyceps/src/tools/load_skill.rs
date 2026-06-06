//! `load_skill(skill_id)` — activate a skill for the rest of the session.
//!
//! This tool closes over the shared [`SkillCatalog`] / registry, so it is a
//! hand-written [`Tool`] impl rather than a `tool!`-macro tool (the macro
//! produces a zero-sized struct that cannot capture an `Arc`). On execute it:
//!
//! 1. confirms the named skill exists in the registry (rejects unknown ids,
//!    recoverably, so the model can pick a real one from the manifest);
//! 2. reads `run_store["active_skills"]` → `Vec<String>`, appends the id
//!    (deduped), and writes it back;
//! 3. returns a short confirmation.
//!
//! The active set is then re-injected every turn by
//! [`SkillInjectingContextManager`](crate::skills::SkillInjectingContextManager)
//! — no new `ToolOutput` variant, all storage-backed (issue #115 flavor B).

use std::sync::Arc;

use serde_json::json;

use spore_core::harness::{BoxFut, SandboxProvider, ToolOutput};
use spore_core::tool_registry::{Tool, ToolContext};
use spore_core::{
    GuideRegistry, RegisteredToolSchema, StandardGuideRegistry, StandardTool, ToolAnnotations,
};

use crate::skills::ACTIVE_SKILLS_KEY;

/// The registered name of the tool.
pub const NAME: &str = "load_skill";

/// `load_skill`, holding the shared registry so it can validate ids.
struct LoadSkillTool {
    registry: Arc<StandardGuideRegistry>,
}

impl Tool for LoadSkillTool {
    fn name(&self) -> &str {
        NAME
    }

    fn execute<'a>(
        &'a self,
        call: &'a spore_core::ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        ctx: &'a ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let skill_id = match call.input.get("skill_id").and_then(|v| v.as_str()) {
                Some(s) if !s.trim().is_empty() => s.trim().to_string(),
                _ => {
                    return ToolOutput::error(
                        "invalid parameters: `skill_id` (string) is required",
                    );
                }
            };

            // 1. Confirm the skill exists. `select` returns only Active guides;
            //    a broad query with the id as the instruction surfaces it (and
            //    ranks it first on exact-term overlap). Reject unknown ids
            //    recoverably so the model can choose a real one.
            let known = self
                .registry
                .select(spore_core::GuideQuery::new(skill_id.clone()))
                .await
                .map(|guides| guides.iter().any(|g| g.id.as_str() == skill_id))
                .unwrap_or(false);
            if !known {
                return ToolOutput::error(format!(
                    "unknown skill '{skill_id}'. Pick one of the skills listed in the manifest."
                ));
            }

            // 2. Append to run_store["active_skills"] (dedup).
            let session_id = ctx.session_id();
            let run_store = ctx.run_store();
            let mut active: Vec<String> = match run_store.get(session_id, ACTIVE_SKILLS_KEY).await {
                Ok(Some(value)) => serde_json::from_value(value).unwrap_or_default(),
                Ok(None) => Vec::new(),
                Err(e) => {
                    return ToolOutput::error(format!(
                        "load_skill: could not read active set: {e}"
                    ));
                }
            };
            if !active.iter().any(|s| s == &skill_id) {
                active.push(skill_id.clone());
            }
            if let Err(e) = run_store
                .put(session_id, ACTIVE_SKILLS_KEY, json!(active))
                .await
            {
                return ToolOutput::error(format!("load_skill: could not persist active set: {e}"));
            }

            // 3. Confirm. The body is now injected every turn by the context
            //    manager, so the procedure is "active" from the next turn on.
            ToolOutput::success(format!(
                "Loaded skill '{skill_id}'. Its procedure is now active — follow it."
            ))
        })
    }
}

/// Build the `load_skill` [`StandardTool`], closing over the shared registry.
pub fn load_skill_tool(registry: Arc<StandardGuideRegistry>) -> StandardTool {
    StandardTool::new(
        Box::new(LoadSkillTool { registry }),
        RegisteredToolSchema {
            name: NAME.into(),
            description: "Activate a skill by id so its full procedure stays in your context for \
                          the rest of the session. Choose an id from the manifest of available \
                          skills."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "The id (name) of the skill to activate, e.g. \"audit\"."
                    }
                },
                "required": ["skill_id"]
            }),
            annotations: ToolAnnotations {
                // Writes a small per-run key; not read-only, but harmless.
                ..Default::default()
            },
        },
    )
}
