package observability

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// End-to-end coverage for adaptive prompt-based tool-calling escalation (#111):
// the AdaptiveToolCallModelInterface + harness-loop seam, driven through a full
// conversational harness with a scripted MockModel. Mirrors the Rust reference
// tests in rust/crates/spore-core/tests/prompt_tool_call_escalation.rs.

// countingCalculator records how many times it is dispatched. Defined locally
// (not imported from the tools package) to avoid the tools -> observability
// import cycle.
type countingCalculator struct {
	name string
	hits *atomic.Int64
}

func (c countingCalculator) Name() string                { return c.name }
func (c countingCalculator) IsSubagentTool() bool        { return false }
func (c countingCalculator) MayProduceLargeOutput() bool { return false }
func (c countingCalculator) Execute(_ context.Context, _ sporecore.ToolCall, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	c.hits.Add(1)
	return sporecore.NewToolOutputSuccess("4")
}

func calculatorStandardTool(hits *atomic.Int64) sporecore.StandardTool {
	return sporecore.StandardTool{
		Implementation: countingCalculator{name: "calculator", hits: hits},
		Schema: sporecore.RegistryToolSchema{
			Name:        "calculator",
			Description: "Evaluate a math expression",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
		},
	}
}

func escProvider() sporecore.ProviderInfo {
	return sporecore.ProviderInfo{Name: "mock", ModelID: "mock-1", ContextWindow: 8192}
}

func escText(t string) sporecore.ModelResponse {
	return sporecore.ModelResponse{
		Content:    []sporecore.ContentBlock{sporecore.NewTextBlock(t)},
		Usage:      sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1},
		StopReason: sporecore.StopEndTurn,
	}
}

// Turn 1 prose with action-intent → escalate. Turn 2 (now in prompt mode) emits
// a <tool_call> marker → parsed + dispatched. Turn 3 final answer.
func TestProseResponseEscalatesToPromptBasedToolCall(t *testing.T) {
	hits := &atomic.Int64{}
	model := sporecore.NewMockModel(escProvider())
	model.PushResponse(escText("Sure — I'll use the calculator tool to compute 2+2."))
	model.PushResponse(escText(`<tool_call><name>calculator</name><input>{"expression":"2+2"}</input></tool_call>`))
	model.PushResponse(escText("The answer is 4"))

	h := ConversationalBuilder(model).
		Tool(calculatorStandardTool(hits)).
		Build()

	r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(
		sporecore.SimpleTask("What is 2+2?")))

	if r.Kind != sporecore.RunSuccess {
		t.Fatalf("kind = %q reason = %+v", r.Kind, r.Reason)
	}
	// Reaching turn 3's answer proves: turn 1 did NOT terminate the run
	// (escalation fired), and turn 2's marker text was parsed into a real tool
	// call rather than being treated as a prose final answer.
	if r.Output != "The answer is 4" {
		t.Fatalf("output = %q, want %q", r.Output, "The answer is 4")
	}
	if r.Turns != 3 {
		t.Fatalf("turns = %d, want 3 (prose -> tool call -> answer)", r.Turns)
	}
	if hits.Load() != 1 {
		t.Fatalf("calculator dispatched %d times, want 1", hits.Load())
	}
}

// Native path unaffected: tools advertised, but the model gives a plain final
// answer with no action-intent language. The conservative heuristic must NOT
// escalate — the run completes on turn 1.
func TestPlainFinalAnswerDoesNotEscalate(t *testing.T) {
	hits := &atomic.Int64{}
	model := sporecore.NewMockModel(escProvider())
	model.PushResponse(escText("The answer is 4."))

	h := ConversationalBuilder(model).
		Tool(calculatorStandardTool(hits)).
		Build()

	r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(
		sporecore.SimpleTask("What is 2+2?")))

	if r.Kind != sporecore.RunSuccess {
		t.Fatalf("kind = %q reason = %+v", r.Kind, r.Reason)
	}
	if r.Output != "The answer is 4." {
		t.Fatalf("output = %q, want %q", r.Output, "The answer is 4.")
	}
	if r.Turns != 1 {
		t.Fatalf("turns = %d, want 1 (a plain answer must not escalate)", r.Turns)
	}
	if hits.Load() != 0 {
		t.Fatalf("no tool should be dispatched, got %d", hits.Load())
	}
}

// Native tool calling unaffected: a model that emits a native tool-use block
// (tools advertised) dispatches normally through the adaptive wrapper while the
// flag is unset — no prompt injection, no marker parsing involved.
func TestNativeToolCallPathUnaffected(t *testing.T) {
	hits := &atomic.Int64{}
	model := sporecore.NewMockModel(escProvider())
	model.PushResponse(sporecore.ModelResponse{
		Content: []sporecore.ContentBlock{sporecore.NewToolUseBlock(sporecore.ToolCall{
			ID: "c1", Name: "calculator", Input: json.RawMessage(`{"expression":"2+2"}`),
		})},
		Usage:      sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1},
		StopReason: sporecore.StopToolUse,
	})
	model.PushResponse(escText("The answer is 4"))

	h := ConversationalBuilder(model).
		Tool(calculatorStandardTool(hits)).
		Build()

	r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(
		sporecore.SimpleTask("What is 2+2?")))

	if r.Kind != sporecore.RunSuccess {
		t.Fatalf("kind = %q reason = %+v", r.Kind, r.Reason)
	}
	if r.Output != "The answer is 4" {
		t.Fatalf("output = %q, want %q", r.Output, "The answer is 4")
	}
	if r.Turns != 2 {
		t.Fatalf("turns = %d, want 2 (native tool call then answer)", r.Turns)
	}
	if hits.Load() != 1 {
		t.Fatalf("native tool call dispatched %d times, want 1", hits.Load())
	}
}
