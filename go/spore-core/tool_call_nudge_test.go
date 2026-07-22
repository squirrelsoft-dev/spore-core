package sporecore

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"testing"
)

type nudgeTestTool struct{}

func (nudgeTestTool) Name() string                { return "nudge_tool" }
func (nudgeTestTool) IsSubagentTool() bool        { return false }
func (nudgeTestTool) MayProduceLargeOutput() bool { return false }
func (nudgeTestTool) Execute(context.Context, ToolCall, SandboxProvider, *ToolContext) ToolOutput {
	return ToolOutput{Kind: ToolOutputSuccess, Content: "used"}
}

func nudgeRegistry(t *testing.T) *StandardToolRegistry {
	t.Helper()
	reg := NewStandardToolRegistry()
	if err := reg.Register(nudgeTestTool{}, RegistryToolSchema{
		Name:        "nudge_tool",
		Description: "test tool",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}); err != nil {
		t.Fatal(err)
	}
	return reg
}

func TestToolCallNudgeDefaultOffDoesNotForceAnotherTurn(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("I looked already", turnUsage()))
	cfg := standardCfg(a)
	cfg.CatalogueRegistry = nudgeRegistry(t)
	h := NewStandardHarness(cfg)

	r := h.ReactWindow(context.Background(), reactTask(5), 5, SessionState{}, BudgetSnapshot{}, nil, a, ToolsetRef(""), nil, 0, nil, true)
	if r.Kind != RunSuccess || r.Output != "I looked already" {
		t.Fatalf("result=%+v, want first final success when nudge is default-off", r)
	}
	if got := a.CallCount.Load(); got != 1 {
		t.Fatalf("model call count=%d, want 1", got)
	}
}

func TestToolCallNudgeBehavioralExecuteStepForcesToolUse(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("I looked already", turnUsage()))
	a.Push(NewToolCallRequested([]ToolCall{{ID: "n1", Name: "nudge_tool", Input: json.RawMessage(`{}`)}}, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := standardCfg(a).WithToolCallNudge(true)
	cfg.CatalogueRegistry = nudgeRegistry(t)
	h := NewStandardHarness(cfg)

	r := h.ReactWindow(context.Background(), reactTask(5), 5, SessionState{}, BudgetSnapshot{}, nil, a, ToolsetRef(""), nil, 0, nil, true)
	if r.Kind != RunSuccess || r.Output != "done" {
		t.Fatalf("result=%+v, want success after nudged tool use", r)
	}
	if got := a.CallCount.Load(); got != 3 {
		t.Fatalf("model call count=%d, want 3 (final -> nudge -> tool -> final)", got)
	}
}

func TestToolCallNudgeStructuralFlipsPromptFlag(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("```bash\necho hi\n```", turnUsage()))
	a.Push(NewToolCallRequested([]ToolCall{{ID: "n1", Name: "nudge_tool", Input: json.RawMessage(`{}`)}}, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))
	flag := &atomic.Bool{}
	cfg := standardCfg(a).WithToolCallNudge(true)
	cfg.PromptToolCallFlag = flag
	cfg.CatalogueRegistry = nudgeRegistry(t)
	h := NewStandardHarness(cfg)

	r := h.ReactWindow(context.Background(), reactTask(5), 5, SessionState{}, BudgetSnapshot{}, nil, a, ToolsetRef(""), nil, 0, nil, true)
	if r.Kind != RunSuccess {
		t.Fatalf("result=%+v, want success after structural nudge", r)
	}
	if !flag.Load() {
		t.Fatal("PromptToolCallFlag was not flipped by structural execute-step nudge")
	}
}
