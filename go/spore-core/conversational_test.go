package sporecore

import (
	"context"
	"testing"
)

// Rule: NullSandbox approves every Validate (no environment access expected).
func TestNullSandboxValidateApproves(t *testing.T) {
	s := NullSandbox{}
	if v := s.Validate(context.Background(), ToolCall{Name: "anything"}); v != nil {
		t.Fatalf("NullSandbox.Validate returned violation %+v, want nil", v)
	}
}

// Rule: CompleteOnFinalResponse always Continues, so the first final response
// is accepted as the result.
func TestCompleteOnFinalResponseAlwaysContinues(t *testing.T) {
	p := CompleteOnFinalResponse{}
	d := p.Evaluate(context.Background(), &SessionState{}, &BudgetSnapshot{})
	if d.Kind != TerminationContinue {
		t.Fatalf("decision = %q, want %q", d.Kind, TerminationContinue)
	}
}

// Rule: SimpleTask mirrors Task::simple — a fresh session id and a default
// ReAct loop with MaxIterations 8.
func TestSimpleTaskDefaults(t *testing.T) {
	task := SimpleTask("Reply with a friendly greeting.")
	if task.Instruction != "Reply with a friendly greeting." {
		t.Fatalf("instruction = %q", task.Instruction)
	}
	if task.LoopStrategy.Kind != StrategyReAct {
		t.Fatalf("strategy kind = %q, want %q", task.LoopStrategy.Kind, StrategyReAct)
	}
	if task.LoopStrategy.MaxIterations() != 8 {
		t.Fatalf("max iterations = %d, want 8", task.LoopStrategy.MaxIterations())
	}
	if task.SessionID == "" {
		t.Fatalf("session id is empty, want a fresh generated id")
	}

	// Two SimpleTasks must get distinct fresh session ids.
	other := SimpleTask("something else")
	if other.SessionID == task.SessionID {
		t.Fatalf("two SimpleTasks shared session id %q", task.SessionID)
	}
}

// Rule: a tool-less conversational harness wired with NullSandbox +
// CompleteOnFinalResponse succeeds on the agent's first final response.
// (The full preset constructor lives in the observability package to avoid an
// import cycle; here we assert the two primitives compose into a single-turn
// success against a MockAgent, which is what the preset relies on.)
func TestConversationalPrimitivesSingleTurnSuccess(t *testing.T) {
	agent := NewMockAgent("agent")
	agent.Push(NewFinalResponse("hello there!", TokenUsage{InputTokens: 1, OutputTokens: 1}))

	cfg := HarnessConfig{
		Agent:             agent,
		ToolRegistry:      NewStandardToolRegistry(),
		Sandbox:           NullSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: CompleteOnFinalResponse{},
	}
	h := NewStandardHarness(cfg)

	r := h.Run(context.Background(), NewHarnessRunOptions(SimpleTask("Say hi.")))
	if r.Kind != RunSuccess {
		t.Fatalf("kind = %q reason = %+v", r.Kind, r.Reason)
	}
	if r.Output != "hello there!" {
		t.Fatalf("output = %q, want %q", r.Output, "hello there!")
	}
	if r.Turns != 1 {
		t.Fatalf("turns = %d, want 1", r.Turns)
	}
}
