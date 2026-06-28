//! SetPlan tool — the planner authors its whole plan in ONE atomic call.
//!
//! `set_plan(tasks=[...], rationale="...")` replaces the persisted task list with
//! the given ordered steps. Because the plan arrives as a TOOL CALL — consumed by
//! the harness, never rendered as a message — it cannot leak as a raw
//! `{"tasks":...}` assistant message, and the execute phase (which is not told to
//! author plans and is not offered this tool) won't emit one either.
//!
//! It writes the SAME RunStore blob the [`task_list`](crate::tools::tasklist)
//! tool authors (durable, project-scoped, under
//! [`TASK_LIST_EXTRAS_KEY`](crate::tasklist::TASK_LIST_EXTRAS_KEY)), so the
//! existing plan-capture fallback (`load_task_list`) and the execute-phase
//! runnable list read it unchanged, and the plan-start `clear_task_list` clears
//! any stale plan for free. Like `task_list` it is NOT `read_only` — it mutates
//! that shared blob and must dispatch sequentially.

use serde_json::json;

use crate::harness::{BoxFut, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tasklist::{TaskList, TASK_LIST_EXTRAS_KEY};
use crate::tool_registry::{Tool, ToolAnnotations, ToolContext, ToolSchema};

pub struct SetPlanTool;

impl SetPlanTool {
    pub const NAME: &'static str = "set_plan";

    pub fn new() -> Self {
        Self
    }

    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Record your plan in ONE call: `tasks` = ordered step strings. \
                          Use this instead of writing the plan as text."
                .into(),
            parameters: json!({
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered, concrete step strings.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Optional one-line reason for the plan.",
                    },
                },
                "required": ["tasks"],
                "type": "object",
            }),
            annotations: ToolAnnotations::default(),
        }
    }
}

impl Default for SetPlanTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for SetPlanTool {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        ctx: &'a ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            // Parse `tasks`: a non-empty array of non-empty strings (bad → recoverable).
            let tasks: Vec<String> = match call.input.get("tasks").and_then(|v| v.as_array()) {
                Some(arr) if !arr.is_empty() => {
                    let mut out = Vec::with_capacity(arr.len());
                    for (i, el) in arr.iter().enumerate() {
                        match el.as_str() {
                            Some(s) if !s.trim().is_empty() => out.push(s.to_string()),
                            _ => {
                                return ToolOutput::Error {
                                    message: format!("task {i} must be a non-empty string"),
                                    recoverable: true,
                                };
                            }
                        }
                    }
                    out
                }
                _ => {
                    return ToolOutput::Error {
                        message: "`tasks` must be a non-empty array of step strings".into(),
                        recoverable: true,
                    };
                }
            };

            // Build a linear TaskList (no blockers) from the ordered steps.
            let mut list = TaskList::default();
            for desc in &tasks {
                if let Err(e) = list.add(desc.clone(), Vec::new()) {
                    return ToolOutput::Error {
                        message: format!("could not record task {desc:?}: {e}"),
                        recoverable: true,
                    };
                }
            }

            // Persist under the SAME durable, project-scoped key `task_list` uses
            // (#142), so the plan-capture fallback and execute phase read it.
            let project_ns = ctx.project_id().namespace();
            let session_id = &project_ns;
            let value = match serde_json::to_value(&list) {
                Ok(v) => v,
                Err(e) => {
                    return ToolOutput::Error {
                        message: format!("could not serialize plan: {e}"),
                        recoverable: true,
                    };
                }
            };
            if let Err(e) = ctx
                .run_store()
                .put(session_id, TASK_LIST_EXTRAS_KEY, value)
                .await
            {
                return ToolOutput::Error {
                    message: format!("could not persist plan: {e}"),
                    recoverable: true,
                };
            }

            ToolOutput::success(format!("Recorded a {}-step plan.", tasks.len()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_requires_tasks_array() {
        let s = SetPlanTool::schema();
        assert_eq!(s.name, "set_plan");
        let required = s.parameters["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "tasks"));
        assert_eq!(s.parameters["properties"]["tasks"]["type"], "array");
    }
}
