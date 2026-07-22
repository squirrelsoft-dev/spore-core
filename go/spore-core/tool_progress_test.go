package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

type progressTestTool struct{}

func (progressTestTool) Name() string                { return "progress_tool" }
func (progressTestTool) IsSubagentTool() bool        { return false }
func (progressTestTool) MayProduceLargeOutput() bool { return false }
func (progressTestTool) Execute(_ context.Context, call ToolCall, _ SandboxProvider, toolCtx *ToolContext) ToolOutput {
	if progress := toolCtx.CommandProgress(call); progress != nil {
		progress("halfway")
	}
	return ToolOutput{Kind: ToolOutputSuccess, Content: "done"}
}

func TestToolProgressEventBetweenToolCallAndResult(t *testing.T) {
	a := NewMockAgent("t")
	call := ToolCall{ID: "p1", Name: "progress_tool", Input: json.RawMessage(`{}`)}
	a.Push(NewToolCallRequested([]ToolCall{call}, turnUsage()))
	a.Push(NewFinalResponse("ok", turnUsage()))

	reg := NewStandardToolRegistry()
	if err := reg.Register(progressTestTool{}, RegistryToolSchema{
		Name:        "progress_tool",
		Description: "emits one live progress preview",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}); err != nil {
		t.Fatal(err)
	}

	cfg := standardCfg(a)
	cfg.CatalogueRegistry = reg
	h := NewStandardHarness(cfg)

	var kinds []HarnessStreamEventKind
	var progress HarnessStreamEvent
	opts := NewHarnessRunOptions(reactTask(5))
	opts.OnStream = func(ev HarnessStreamEvent) {
		kinds = append(kinds, ev.Kind)
		if ev.Kind == HarnessStreamToolProgress {
			progress = ev
		}
	}
	r := h.Run(context.Background(), opts)
	if r.Kind != RunSuccess {
		t.Fatalf("kind=%q reason=%+v, want success", r.Kind, r.Reason)
	}
	if progress.Kind != HarnessStreamToolProgress {
		t.Fatalf("tool_progress event missing; events=%v", kinds)
	}
	if progress.CallID != "p1" || progress.Name != "progress_tool" || progress.Preview != "halfway" {
		t.Fatalf("unexpected progress event: %+v", progress)
	}
	idx := func(k HarnessStreamEventKind) int {
		for i, got := range kinds {
			if got == k {
				return i
			}
		}
		return -1
	}
	callIdx, progressIdx, resultIdx := idx(HarnessStreamToolCall), idx(HarnessStreamToolProgress), idx(HarnessStreamToolResult)
	if callIdx < 0 || progressIdx < 0 || resultIdx < 0 {
		t.Fatalf("missing expected tool events: %v", kinds)
	}
	if !(callIdx < progressIdx && progressIdx < resultIdx) {
		t.Fatalf("tool_progress should be between tool_call and tool_result; events=%v", kinds)
	}
}

func TestToolProgressMarshalIncludesNode(t *testing.T) {
	ev := HarnessStreamEvent{
		Kind:    HarnessStreamToolProgress,
		CallID:  "c",
		Name:    "bash_command",
		Preview: "tail",
		Node:    &NodeAttr{Kind: "react", Depth: 0, Path: []string{"react"}},
	}
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]json.RawMessage
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"kind", "call_id", "name", "preview", "node"} {
		if _, ok := got[key]; !ok {
			t.Fatalf("missing %q in %s", key, b)
		}
	}
}
