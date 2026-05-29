//! SubagentTool — wraps a child `Harness` and exposes it as a `Tool`.
//!
//! Per spec (issue #5), subagents cannot spawn their own subagents. The
//! restriction is enforced at construction time by inspecting the child's
//! `ToolRegistry` via `has_subagent_tools()`.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::harness::{
    BoxFut, ChildPausedState, Harness, HarnessRunOptions, RunResult, SandboxProvider, SessionId,
    SessionState, Task, ToolOutput,
};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolRegistry};

/// How the subagent inherits / does not inherit context from its parent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContextSharing {
    Isolated,
    SharedSession { session_id: SessionId },
    SummaryHandoff { summary: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum BuildError {
    #[error("invalid configuration: {reason}")]
    InvalidConfiguration { reason: String },
}

pub struct SubagentTool {
    name: String,
    description: String,
    input_schema: Value,
    timeout: Duration,
    context_sharing: ContextSharing,
    harness: Arc<dyn Harness>,
}

impl SubagentTool {
    /// Construct a SubagentTool. Returns `BuildError::InvalidConfiguration`
    /// if the child registry already contains subagent tools (depth-1 rule).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        timeout: Duration,
        context_sharing: ContextSharing,
        harness: Arc<dyn Harness>,
        child_registry: &dyn ToolRegistry,
    ) -> Result<Self, BuildError> {
        if child_registry.has_subagent_tools() {
            return Err(BuildError::InvalidConfiguration {
                reason: "child harness must not contain SubagentTool (depth-1 rule)".into(),
            });
        }
        Ok(Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            timeout,
            context_sharing,
            harness,
        })
    }

    pub fn description(&self) -> &str {
        &self.description
    }
    pub fn input_schema(&self) -> &Value {
        &self.input_schema
    }
    pub fn context_sharing(&self) -> &ContextSharing {
        &self.context_sharing
    }
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

impl Tool for SubagentTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn is_subagent_tool(&self) -> bool {
        true
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let instruction = match call.input.get("instruction").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => {
                    return ToolOutput::Error {
                        message: "invalid parameters: missing `instruction`".into(),
                        recoverable: true,
                    }
                }
            };

            // Build task and (where applicable) seed session state.
            let (session_id, seeded_session): (SessionId, Option<SessionState>) =
                match &self.context_sharing {
                    ContextSharing::Isolated => (SessionId::generate(), None),
                    ContextSharing::SharedSession { session_id } => (session_id.clone(), None),
                    ContextSharing::SummaryHandoff { summary } => {
                        // Inject the summary as a synthetic extras entry; the
                        // ContextManager (issue #7) decides how to surface it.
                        let mut state = SessionState::default();
                        state.extras.insert(
                            "subagent_handoff_summary".into(),
                            serde_json::Value::String(summary.clone()),
                        );
                        (SessionId::generate(), Some(state))
                    }
                };

            let task = Task::new(
                instruction,
                session_id,
                crate::harness::LoopStrategy::ReAct { max_iterations: 16 },
            );

            let mut options = HarnessRunOptions::new(task);
            options.session_state = seeded_session;

            let fut = self.harness.run(options);
            let result = match tokio::time::timeout(self.timeout, fut).await {
                Ok(r) => r,
                Err(_) => {
                    return ToolOutput::Error {
                        message: format!("subagent timed out after {}s", self.timeout.as_secs()),
                        recoverable: true,
                    }
                }
            };

            match result {
                RunResult::Success { output, .. } => ToolOutput::Success {
                    content: output,
                    truncated: false,
                },
                RunResult::Failure { reason, .. } => ToolOutput::Error {
                    message: format!("subagent failed: {reason:?}"),
                    recoverable: true,
                },
                RunResult::WaitingForHuman { state, request } => {
                    let child = child_state_from_paused(*state, call.id.clone());
                    ToolOutput::WaitingForHuman {
                        child_state: Box::new(child),
                        request,
                    }
                }
                // A subagent escalation (issue #80) propagates as a tool-side
                // escalation: the parent harness terminates cleanly and hands
                // the signal up to its own caller.
                RunResult::Escalate { signal, .. } => ToolOutput::Escalate { signal },
            }
        })
    }
}

fn child_state_from_paused(
    state: crate::harness::PausedState,
    parent_tool_call_id: String,
) -> ChildPausedState {
    ChildPausedState {
        session_id: state.session_id,
        task_id: state.task_id,
        turn_number: state.turn_number,
        session_state: state.session_state,
        pending_tool_calls: state.pending_tool_calls,
        approved_results: state.approved_results,
        human_request: state.human_request,
        task: state.task,
        budget_used: state.budget_used,
        parent_tool_call_id,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{
        AggregateUsage, HaltReason, HarnessRunOptions, HumanRequest, HumanResponse, PausedState,
        RunResult,
    };
    use crate::tool_registry::mock::{test_ctx, AllowAllSandbox};
    use crate::tool_registry::StandardToolRegistry;
    use serde_json::json;
    use std::sync::Mutex;

    /// Scripted harness — returns a queue of `RunResult`s.
    struct ScriptedHarness {
        results: Mutex<Vec<RunResult>>,
    }
    impl ScriptedHarness {
        fn new(rs: Vec<RunResult>) -> Self {
            Self {
                results: Mutex::new(rs),
            }
        }
    }
    impl Harness for ScriptedHarness {
        fn run<'a>(&'a self, _opts: HarnessRunOptions) -> BoxFut<'a, RunResult> {
            Box::pin(async move {
                let mut g = self.results.lock().unwrap();
                if g.is_empty() {
                    return RunResult::Failure {
                        reason: HaltReason::AgentError {
                            error: crate::agent::AgentError::EmptyResponse,
                        },
                        session_id: SessionId::new("s"),
                        usage: AggregateUsage::default(),
                        turns: 0,
                    };
                }
                g.remove(0)
            })
        }
        fn resume<'a>(
            &'a self,
            _state: PausedState,
            _response: HumanResponse,
            _on_stream: Option<crate::harness::StreamSink>,
        ) -> BoxFut<'a, RunResult> {
            Box::pin(async move {
                RunResult::Failure {
                    reason: HaltReason::HumanHalted,
                    session_id: SessionId::new("s"),
                    usage: AggregateUsage::default(),
                    turns: 0,
                }
            })
        }
    }

    fn call(input: Value) -> ToolCall {
        ToolCall {
            id: "parent-call-1".into(),
            name: "subagent".into(),
            input,
        }
    }

    #[tokio::test]
    async fn subagent_success_maps_to_tool_success() {
        let h = Arc::new(ScriptedHarness::new(vec![RunResult::Success {
            output: "child done".into(),
            session_id: SessionId::new("s"),
            usage: AggregateUsage::default(),
            turns: 1,
        }]));
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(5),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap();
        let sb = AllowAllSandbox;
        let r = sub
            .execute(&call(json!({"instruction": "do it"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, "child done"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn subagent_failure_maps_to_recoverable_error() {
        let h = Arc::new(ScriptedHarness::new(vec![RunResult::Failure {
            reason: HaltReason::HumanHalted,
            session_id: SessionId::new("s"),
            usage: AggregateUsage::default(),
            turns: 1,
        }]));
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(5),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap();
        let sb = AllowAllSandbox;
        let r = sub
            .execute(&call(json!({"instruction": "x"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn subagent_waiting_for_human_propagates_with_parent_call_id() {
        let paused = PausedState {
            session_id: SessionId::new("s"),
            task_id: crate::harness::TaskId::new("t"),
            turn_number: 1,
            session_state: SessionState::default(),
            pending_tool_calls: vec![],
            approved_results: vec![],
            human_request: Some(HumanRequest::Clarification {
                question: "yes?".into(),
            }),
            task: Task::new(
                "x",
                SessionId::new("s"),
                crate::harness::LoopStrategy::ReAct { max_iterations: 1 },
            ),
            budget_used: crate::harness::BudgetSnapshot::default(),
            child_state: None,
        };
        let h = Arc::new(ScriptedHarness::new(vec![RunResult::WaitingForHuman {
            state: Box::new(paused),
            request: HumanRequest::Clarification {
                question: "yes?".into(),
            },
        }]));
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(5),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap();
        let sb = AllowAllSandbox;
        let r = sub
            .execute(&call(json!({"instruction": "x"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::WaitingForHuman { child_state, .. } => {
                assert_eq!(child_state.parent_tool_call_id, "parent-call-1");
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn construction_rejects_child_with_subagent_tools() {
        use crate::tool_registry::mock::SubagentMock;
        use crate::tool_registry::{ToolAnnotations, ToolSchema};
        let child_reg = StandardToolRegistry::new();
        child_reg
            .register(
                Box::new(SubagentMock::new("nested")),
                ToolSchema {
                    name: "nested".into(),
                    description: "n".into(),
                    parameters: json!({"type": "object"}),
                    annotations: ToolAnnotations::default(),
                },
            )
            .unwrap();
        let h = Arc::new(ScriptedHarness::new(vec![]));
        let r = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(1),
            ContextSharing::Isolated,
            h,
            &child_reg,
        );
        assert!(matches!(r, Err(BuildError::InvalidConfiguration { .. })));
    }

    #[tokio::test]
    async fn missing_instruction_returns_recoverable_error() {
        let h = Arc::new(ScriptedHarness::new(vec![]));
        let child_reg = StandardToolRegistry::new();
        let sub = SubagentTool::new(
            "subagent",
            "child",
            json!({"type": "object"}),
            Duration::from_secs(1),
            ContextSharing::Isolated,
            h,
            &child_reg,
        )
        .unwrap();
        let sb = AllowAllSandbox;
        let r = sub.execute(&call(json!({})), &sb, &test_ctx()).await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }
}
