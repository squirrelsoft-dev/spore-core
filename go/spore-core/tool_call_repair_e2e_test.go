package sporecore

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
)

// boolFlagTool is a minimal validating tool: it unmarshals args into a struct
// with a typed bool Flag field, so a string-encoded "false" fails with a
// recoverable "invalid parameters" error — exactly the weak-model defect
// ToolCallRepair exists to fix.
type boolFlagTool struct {
	mu   sync.Mutex
	seen []json.RawMessage // the args each dispatch attempt actually received
}

func (b *boolFlagTool) Name() string                { return "flag_tool" }
func (b *boolFlagTool) IsSubagentTool() bool        { return false }
func (b *boolFlagTool) MayProduceLargeOutput() bool { return false }

func (b *boolFlagTool) Schema() RegistryToolSchema {
	return RegistryToolSchema{
		Name:        "flag_tool",
		Description: "a tool with a bool flag",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"flag":{"type":"boolean"}},"required":["flag"]}`),
		Annotations: ToolAnnotations{},
	}
}

func (b *boolFlagTool) Execute(_ context.Context, call ToolCall, _ SandboxProvider, _ *ToolContext) ToolOutput {
	b.mu.Lock()
	b.seen = append(b.seen, call.Input)
	b.mu.Unlock()
	var params struct {
		Flag bool `json:"flag"`
	}
	if err := json.Unmarshal(call.Input, &params); err != nil {
		return NewToolOutputError("invalid parameters: " + err.Error())
	}
	return NewToolOutputSuccess("flag=" + jsonBoolStr(params.Flag))
}

func jsonBoolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// TestRepairOnDispatchCoercesAndRedispatches is the E2E proof that a configured
// StandardToolCallRepair rescues a weak-model malformed-args call: the first
// dispatch fails validation (string "false" for a bool field), repair coerces
// it to a real bool, and the re-dispatch succeeds — all within a single react
// turn, with no extra model turn.
func TestRepairOnDispatchCoercesAndRedispatches(t *testing.T) {
	tool := &boolFlagTool{}
	reg := NewStandardToolRegistry()
	if err := reg.Register(tool, tool.Schema()); err != nil {
		t.Fatal(err)
	}

	// The model emits ONE turn: a tool call with a string-encoded bool arg,
	// then a final response.
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c1", Name: "flag_tool", Input: json.RawMessage(`{"flag":"false"}`)},
	}, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))

	cfg := standardCfg(a)
	cfg.ToolRegistry = reg
	cfg.ToolCallRepair = StandardToolCallRepair{}
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess {
		t.Fatalf("kind=%q reason=%+v", r.Kind, r.Reason)
	}
	// The tool must have been dispatched TWICE: once with the malformed args
	// (failed validation), once with the coerced args (succeeded).
	tool.mu.Lock()
	got := len(tool.seen)
	tool.mu.Unlock()
	if got != 2 {
		t.Fatalf("expected 2 dispatch attempts (malformed then repaired), got %d", got)
	}
	// The second attempt carries the coerced bool.
	if string(tool.seen[1]) != `{"flag":false}` {
		t.Fatalf("second attempt input = %s, want {\"flag\":false}", tool.seen[1])
	}
}

// TestRepairAbsentByDefaultIsByteIdentical confirms that WITHOUT a repair
// provider, the malformed-args call is NOT re-dispatched (the recoverable error
// is fed straight back to the model) — the default path is byte-identical.
func TestRepairAbsentByDefaultIsByteIdentical(t *testing.T) {
	tool := &boolFlagTool{}
	reg := NewStandardToolRegistry()
	_ = reg.Register(tool, tool.Schema())

	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c1", Name: "flag_tool", Input: json.RawMessage(`{"flag":"false"}`)},
	}, turnUsage()))
	// The model retries with a correct call on the next turn.
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c2", Name: "flag_tool", Input: json.RawMessage(`{"flag":false}`)},
	}, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))

	cfg := standardCfg(a)
	cfg.ToolRegistry = reg
	// No ToolCallRepair configured.
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess {
		t.Fatalf("kind=%q reason=%+v", r.Kind, r.Reason)
	}
	tool.mu.Lock()
	got := len(tool.seen)
	tool.mu.Unlock()
	// Without repair: each call dispatches exactly ONCE (the malformed one
	// fails and is fed back; the model retries manually). Two calls → two
	// dispatches, NOT three.
	if got != 2 {
		t.Fatalf("expected 2 dispatches (no repair re-dispatch), got %d", got)
	}
}

// TestMaxRepairAttemptsZeroDisablesRepair confirms an explicit 0 is honored:
// even with a repair provider set, MaxRepairAttempts=0 means no re-dispatch.
func TestMaxRepairAttemptsZeroDisablesRepair(t *testing.T) {
	tool := &boolFlagTool{}
	reg := NewStandardToolRegistry()
	_ = reg.Register(tool, tool.Schema())

	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c1", Name: "flag_tool", Input: json.RawMessage(`{"flag":"false"}`)},
	}, turnUsage()))
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c2", Name: "flag_tool", Input: json.RawMessage(`{"flag":false}`)},
	}, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))

	cfg := standardCfg(a)
	cfg.ToolRegistry = reg
	cfg.ToolCallRepair = StandardToolCallRepair{}
	zero := uint32(0)
	cfg.MaxRepairAttempts = &zero
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess {
		t.Fatalf("kind=%q reason=%+v", r.Kind, r.Reason)
	}
	tool.mu.Lock()
	got := len(tool.seen)
	tool.mu.Unlock()
	if got != 2 {
		t.Fatalf("MaxRepairAttempts=0 should disable repair: got %d dispatches, want 2", got)
	}
}
