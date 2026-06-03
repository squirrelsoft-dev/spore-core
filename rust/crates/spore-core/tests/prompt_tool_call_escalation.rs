//! End-to-end coverage for adaptive prompt-based tool-calling escalation
//! (the `AdaptiveToolCallModelInterface` + harness-loop seam).
//!
//! These drive a full `conversational` harness with a scripted model
//! (`MockModelInterface`) to prove the escalation path the unit tests can only
//! exercise in pieces: native-first, then automatic switch to prompt-based mode
//! after a prose response, then `<tool_call>` markers parsed into a real
//! dispatch — with no model lists and no manual wrapping.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use spore_core::harness::{BoxFut, SandboxProvider, ToolOutput};
use spore_core::model::mock::MockModelInterface;
use spore_core::model::{
    ContentBlock, ModelResponse, ProviderInfo, StopReason, TokenUsage, ToolCall,
};
use spore_core::tool_registry::{Tool, ToolAnnotations, ToolContext, ToolSchema};
use spore_core::tools::StandardTool;
use spore_core::{Harness, HarnessBuilder, HarnessRunOptions, RunResult, Task};

/// A pure tool that records how many times it was dispatched.
struct CountingCalculator {
    hits: Arc<AtomicUsize>,
}

impl Tool for CountingCalculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn execute<'a>(
        &'a self,
        _call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        self.hits.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { ToolOutput::success("4") })
    }
}

fn calculator_tool(hits: Arc<AtomicUsize>) -> StandardTool {
    StandardTool::new(
        Box::new(CountingCalculator { hits }),
        ToolSchema {
            name: "calculator".into(),
            description: "Evaluate a math expression".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "expression": { "type": "string" } },
                "required": ["expression"],
            }),
            annotations: ToolAnnotations::default(),
        },
    )
}

fn provider() -> ProviderInfo {
    ProviderInfo {
        name: "mock".into(),
        model_id: "mock-1".into(),
        context_window: 8192,
    }
}

fn usage() -> TokenUsage {
    TokenUsage {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_tokens: None,
        cache_write_tokens: None,
    }
}

fn text(t: &str) -> ModelResponse {
    ModelResponse {
        content: vec![ContentBlock::Text { text: t.into() }],
        usage: usage(),
        stop_reason: StopReason::EndTurn,
    }
}

/// Turn 1 prose with action-intent → escalate. Turn 2 (now in prompt mode)
/// emits a `<tool_call>` marker → parsed + dispatched. Turn 3 final answer.
#[tokio::test]
async fn prose_response_escalates_to_prompt_based_tool_call() {
    let hits = Arc::new(AtomicUsize::new(0));
    let model = MockModelInterface::new(provider());
    model.push_response(Ok(text(
        "Sure — I'll use the calculator tool to compute 2+2.",
    )));
    model.push_response(Ok(text(
        "<tool_call><name>calculator</name><input>{\"expression\":\"2+2\"}</input></tool_call>",
    )));
    model.push_response(Ok(text("The answer is 4")));

    let harness = HarnessBuilder::conversational(model)
        .tool(calculator_tool(hits.clone()))
        .build();

    let result = harness
        .run(HarnessRunOptions::new(Task::simple("What is 2+2?")))
        .await;

    match result {
        RunResult::Success { output, turns, .. } => {
            // Reaching turn 3's answer proves: turn 1 did NOT terminate the run
            // (escalation fired), and turn 2's marker text was parsed into a
            // real tool call rather than being treated as a prose final answer.
            assert_eq!(output, "The answer is 4");
            assert_eq!(turns, 3, "expected prose → tool call → answer");
        }
        other => panic!("expected Success, got {other:?}"),
    }
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "the calculator tool must have been dispatched exactly once"
    );
}

/// Native path unaffected: tools advertised, but the model gives a plain final
/// answer with no action-intent language. The conservative heuristic must NOT
/// escalate — the run completes on turn 1.
#[tokio::test]
async fn plain_final_answer_does_not_escalate() {
    let hits = Arc::new(AtomicUsize::new(0));
    let model = MockModelInterface::new(provider());
    model.push_response(Ok(text("The answer is 4.")));

    let harness = HarnessBuilder::conversational(model)
        .tool(calculator_tool(hits.clone()))
        .build();

    let result = harness
        .run(HarnessRunOptions::new(Task::simple("What is 2+2?")))
        .await;

    match result {
        RunResult::Success { output, turns, .. } => {
            assert_eq!(output, "The answer is 4.");
            assert_eq!(turns, 1, "a plain answer must not trigger escalation");
        }
        other => panic!("expected Success, got {other:?}"),
    }
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "no tool should be dispatched"
    );
}

/// Native tool calling unaffected: a model that emits a native tool-use block
/// (tools advertised) dispatches normally through the adaptive wrapper while the
/// flag is unset — no prompt injection, no marker parsing involved.
#[tokio::test]
async fn native_tool_call_path_unaffected() {
    let hits = Arc::new(AtomicUsize::new(0));
    let model = MockModelInterface::new(provider());
    model.push_response(Ok(ModelResponse {
        content: vec![ContentBlock::ToolUse(ToolCall {
            id: "c1".into(),
            name: "calculator".into(),
            input: serde_json::json!({ "expression": "2+2" }),
        })],
        usage: usage(),
        stop_reason: StopReason::ToolUse,
    }));
    model.push_response(Ok(text("The answer is 4")));

    let harness = HarnessBuilder::conversational(model)
        .tool(calculator_tool(hits.clone()))
        .build();

    let result = harness
        .run(HarnessRunOptions::new(Task::simple("What is 2+2?")))
        .await;

    match result {
        RunResult::Success { output, turns, .. } => {
            assert_eq!(output, "The answer is 4");
            assert_eq!(turns, 2, "native tool call then answer — no extra turn");
        }
        other => panic!("expected Success, got {other:?}"),
    }
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "native tool call dispatched exactly once"
    );
}
