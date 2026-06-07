package tools

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// scriptedHarness yields a queue of pre-scripted RunResults. ResumeConsult logs
// the ConsultResponse it was resumed with and pops the next result (issue #114).
type scriptedHarness struct {
	mu        sync.Mutex
	results   []sporecore.RunResult
	resumeLog []sporecore.ConsultResponse
}

func (s *scriptedHarness) Run(_ context.Context, _ sporecore.HarnessRunOptions) sporecore.RunResult {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.results) == 0 {
		return sporecore.RunResult{
			Kind:   sporecore.RunFailure,
			Reason: sporecore.HaltReason{Kind: sporecore.HaltHumanHalted},
		}
	}
	r := s.results[0]
	s.results = s.results[1:]
	return r
}

func (s *scriptedHarness) Resume(_ context.Context, _ sporecore.PausedState, _ sporecore.HumanResponse, _ sporecore.StreamSink) sporecore.RunResult {
	return sporecore.RunResult{Kind: sporecore.RunFailure, Reason: sporecore.HaltReason{Kind: sporecore.HaltHumanHalted}}
}

func (s *scriptedHarness) ResumeConsult(_ context.Context, _ sporecore.PausedState, response sporecore.ConsultResponse, _ sporecore.StreamSink) sporecore.RunResult {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.resumeLog = append(s.resumeLog, response)
	if len(s.results) == 0 {
		return sporecore.RunResult{Kind: sporecore.RunSuccess, Output: "child done after consult"}
	}
	r := s.results[0]
	s.results = s.results[1:]
	return r
}

func newSubagent(t *testing.T, h sporecore.Harness, sharing ContextSharing) *SubagentTool {
	t.Helper()
	child := sporecore.NewStandardToolRegistry()
	s, err := NewSubagentTool("subagent", "child", json.RawMessage(`{"type":"object"}`),
		5*time.Second, sharing, h, child)
	if err != nil {
		t.Fatalf("NewSubagentTool: %v", err)
	}
	return s
}

func subagentCall(input any) sporecore.ToolCall {
	b, _ := json.Marshal(input)
	return sporecore.ToolCall{ID: "parent-call-1", Name: "subagent", Input: b}
}

func TestSubagentSuccessMaps(t *testing.T) {
	h := &scriptedHarness{results: []sporecore.RunResult{
		{Kind: sporecore.RunSuccess, Output: "child done"},
	}}
	s := newSubagent(t, h, Isolated{})
	r := s.Execute(context.Background(), subagentCall(map[string]any{"instruction": "do it"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "child done" {
		t.Fatalf("%+v", r)
	}
}

func TestSubagentFailureMapsRecoverable(t *testing.T) {
	h := &scriptedHarness{results: []sporecore.RunResult{
		{Kind: sporecore.RunFailure, Reason: sporecore.HaltReason{Kind: sporecore.HaltHumanHalted}},
	}}
	s := newSubagent(t, h, Isolated{})
	r := s.Execute(context.Background(), subagentCall(map[string]any{"instruction": "x"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("%+v", r)
	}
}

func TestSubagentWaitingForHumanPropagatesParentCallID(t *testing.T) {
	paused := &sporecore.PausedState{
		SessionID: "s",
		Task:      sporecore.NewTask("x", "s", sporecore.ReActStrategy(1)),
		HumanRequest: &sporecore.HumanRequest{
			Kind:     sporecore.HumanReqClarification,
			Question: "yes?",
		},
	}
	req := &sporecore.HumanRequest{Kind: sporecore.HumanReqClarification, Question: "yes?"}
	h := &scriptedHarness{results: []sporecore.RunResult{
		{Kind: sporecore.RunWaitingForHuman, State: paused, Request: req},
	}}
	s := newSubagent(t, h, Isolated{})
	r := s.Execute(context.Background(), subagentCall(map[string]any{"instruction": "x"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputWaitingForHuman {
		t.Fatalf("%+v", r)
	}
	if r.ChildState == nil || r.ChildState.ParentToolCallID != "parent-call-1" {
		t.Fatalf("child state wrong: %+v", r.ChildState)
	}
}

// subagentMockTool: a no-op tool that reports IsSubagentTool == true so
// HasSubagentTools() returns true on the child registry.
type subagentMockTool struct{}

func (subagentMockTool) Name() string                { return "nested_sub" }
func (subagentMockTool) IsSubagentTool() bool        { return true }
func (subagentMockTool) MayProduceLargeOutput() bool { return false }
func (subagentMockTool) Execute(context.Context, sporecore.ToolCall, sporecore.SandboxProvider, *sporecore.ToolContext) sporecore.ToolOutput {
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess}
}

func TestSubagentConstructionRejectsNestedChild(t *testing.T) {
	child := sporecore.NewStandardToolRegistry()
	schema := sporecore.RegistryToolSchema{
		Name:        "nested_sub",
		Description: "n",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}
	if err := child.Register(subagentMockTool{}, schema); err != nil {
		t.Fatalf("register: %v", err)
	}
	h := &scriptedHarness{}
	_, err := NewSubagentTool("subagent", "child", json.RawMessage(`{"type":"object"}`),
		time.Second, Isolated{}, h, child)
	if err == nil {
		t.Fatalf("expected build error")
	}
	if !errors.Is(err, ErrInvalidConfiguration) {
		t.Fatalf("expected ErrInvalidConfiguration, got %v", err)
	}
	var be *BuildError
	if !errors.As(err, &be) {
		t.Fatalf("expected *BuildError, got %T", err)
	}
}

func TestSubagentMissingInstructionRecoverable(t *testing.T) {
	h := &scriptedHarness{}
	s := newSubagent(t, h, Isolated{})
	r := s.Execute(context.Background(), subagentCall(map[string]any{}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("%+v", r)
	}
}

func TestContextSharingJSONRoundtrip(t *testing.T) {
	for _, c := range []ContextSharing{
		Isolated{},
		SharedSession{SessionID: "s1"},
		SummaryHandoff{Summary: "did stuff"},
	} {
		b, err := MarshalContextSharing(c)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		back, err := UnmarshalContextSharing(b)
		if err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if back.Kind() != c.Kind() {
			t.Fatalf("kind mismatch: %s vs %s", back.Kind(), c.Kind())
		}
	}
}
