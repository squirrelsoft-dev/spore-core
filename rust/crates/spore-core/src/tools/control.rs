//! Tier-3 control tools (#81): tools that drive the harness via the escalation
//! / clarification protocols rather than returning ordinary data.
//!
//! - [`EnterPlanModeTool`] (`enter_plan_mode`) →
//!   [`ToolOutput::Escalate`]`{ HarnessSignal::EnterPlanMode { context } }`.
//! - [`ExitPlanModeTool`] (`exit_plan_mode`) →
//!   [`ToolOutput::Escalate`]`{ HarnessSignal::ExitPlanMode { plan } }`. The plan
//!   is a structured tool param deserialized DIRECTLY into the existing
//!   [`PlanArtifact`](crate::hooks::PlanArtifact) (issue #81, Q4a — no stub).
//! - [`AskUserQuestionTool`] (`ask_user_question`) →
//!   [`ToolOutput::AwaitingClarification`]`{ question, options }` (issue #81,
//!   Q4b). The harness loop pauses with `HumanRequest::Clarification`.
//! - [`AbortTool`] (`abort`) →
//!   [`ToolOutput::Escalate`]`{ HarnessSignal::Abort { reason } }`.

use serde_json::json;

use crate::harness::{BoxFut, HarnessSignal, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::params::{
    parse_params, AbortParams, AskUserQuestionParams, EnterPlanModeParams, ExitPlanModeParams,
};

// ============================================================================
// EnterPlanMode
// ============================================================================

pub struct EnterPlanModeTool;

impl EnterPlanModeTool {
    pub const NAME: &'static str = "enter_plan_mode";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Request entry into plan mode, seeding the planner with context".into(),
            parameters: json!({
                "type": "object",
                "properties": {"context": {"type": "string"}},
            }),
            annotations: ToolAnnotations::default(),
        }
    }
}

impl Default for EnterPlanModeTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for EnterPlanModeTool {
    fn name(&self) -> &str {
        Self::NAME
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: EnterPlanModeParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            ToolOutput::Escalate {
                signal: HarnessSignal::EnterPlanMode {
                    context: params.context,
                },
            }
        })
    }
}

// ============================================================================
// ExitPlanMode
// ============================================================================

pub struct ExitPlanModeTool;

impl ExitPlanModeTool {
    pub const NAME: &'static str = "exit_plan_mode";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Submit the produced plan and request exit from plan mode".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "object",
                        "properties": {
                            "tasks": {"type": "array", "items": {"type": "string"}},
                            "rationale": {"type": "string"},
                        },
                        "required": ["tasks"],
                    },
                },
                "required": ["plan"],
            }),
            annotations: ToolAnnotations::default(),
        }
    }
}

impl Default for ExitPlanModeTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for ExitPlanModeTool {
    fn name(&self) -> &str {
        Self::NAME
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: ExitPlanModeParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            ToolOutput::Escalate {
                signal: HarnessSignal::ExitPlanMode { plan: params.plan },
            }
        })
    }
}

// ============================================================================
// AskUserQuestion
// ============================================================================

pub struct AskUserQuestionTool;

impl AskUserQuestionTool {
    pub const NAME: &'static str = "ask_user_question";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Ask the user a clarifying question (optionally with fixed choices)"
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question"],
            }),
            annotations: ToolAnnotations::default(),
        }
    }
}

impl Default for AskUserQuestionTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for AskUserQuestionTool {
    fn name(&self) -> &str {
        Self::NAME
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: AskUserQuestionParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            ToolOutput::AwaitingClarification {
                question: params.question,
                options: params.options,
            }
        })
    }
}

// ============================================================================
// Abort
// ============================================================================

pub struct AbortTool;

impl AbortTool {
    pub const NAME: &'static str = "abort";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Request a graceful abort of the run with a reason".into(),
            parameters: json!({
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            }),
            annotations: ToolAnnotations::default(),
        }
    }
}

impl Default for AbortTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for AbortTool {
    fn name(&self) -> &str {
        Self::NAME
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: AbortParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            ToolOutput::Escalate {
                signal: HarnessSignal::Abort {
                    reason: params.reason,
                },
            }
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_registry::mock::{test_ctx, AllowAllSandbox};
    use serde_json::json;

    fn call(name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input,
        }
    }

    #[tokio::test]
    async fn enter_plan_mode_escalates() {
        let sb = AllowAllSandbox;
        let r = EnterPlanModeTool::new()
            .execute(
                &call("enter_plan_mode", json!({"context": "seed"})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Escalate {
                signal: HarnessSignal::EnterPlanMode { context },
            } => assert_eq!(context, "seed"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn exit_plan_mode_escalates_with_plan() {
        let sb = AllowAllSandbox;
        let r = ExitPlanModeTool::new()
            .execute(
                &call(
                    "exit_plan_mode",
                    json!({"plan": {"tasks": ["a", "b"], "rationale": "because"}}),
                ),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Escalate {
                signal: HarnessSignal::ExitPlanMode { plan },
            } => {
                assert_eq!(plan.tasks, vec!["a".to_string(), "b".to_string()]);
                assert_eq!(plan.rationale, "because");
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn exit_plan_mode_rationale_defaults() {
        let sb = AllowAllSandbox;
        let r = ExitPlanModeTool::new()
            .execute(
                &call("exit_plan_mode", json!({"plan": {"tasks": ["x"]}})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Escalate {
                signal: HarnessSignal::ExitPlanMode { plan },
            } => {
                assert_eq!(plan.tasks, vec!["x".to_string()]);
                assert_eq!(plan.rationale, "");
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn ask_user_question_awaits_clarification() {
        let sb = AllowAllSandbox;
        let r = AskUserQuestionTool::new()
            .execute(
                &call(
                    "ask_user_question",
                    json!({"question": "which?", "options": ["a", "b"]}),
                ),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::AwaitingClarification { question, options } => {
                assert_eq!(question, "which?");
                assert_eq!(options, Some(vec!["a".to_string(), "b".to_string()]));
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn ask_user_question_options_optional() {
        let sb = AllowAllSandbox;
        let r = AskUserQuestionTool::new()
            .execute(
                &call("ask_user_question", json!({"question": "free form?"})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::AwaitingClarification { question, options } => {
                assert_eq!(question, "free form?");
                assert_eq!(options, None);
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn abort_escalates() {
        let sb = AllowAllSandbox;
        let r = AbortTool::new()
            .execute(&call("abort", json!({"reason": "stop"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Escalate {
                signal: HarnessSignal::Abort { reason },
            } => assert_eq!(reason, "stop"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn abort_missing_reason_is_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = AbortTool::new()
            .execute(&call("abort", json!({})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }
}
