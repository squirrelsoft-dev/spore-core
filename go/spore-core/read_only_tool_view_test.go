package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

// rovInnerRegistry is a ToolRegistry test double for the SC-30 ReadOnlyToolView
// test. It advertises a fixed read + write + exec schema set and records every
// tool name dispatched so the view's pass-through / block behavior is
// observable. Only the ToolRegistry methods exercised by ReadOnlyToolView are
// meaningful; the rest satisfy the interface.
type rovInnerRegistry struct {
	dispatched []string
}

func (r *rovInnerRegistry) Register(_ Tool, _ RegistryToolSchema) error { return nil }
func (r *rovInnerRegistry) RegisterSet(_ ToolSet) error                 { return nil }

func (r *rovInnerRegistry) ActiveSchemas(_ *TaskPhase) []RegistryToolSchema {
	mk := func(name string) RegistryToolSchema {
		return RegistryToolSchema{
			Name:       name,
			Parameters: json.RawMessage(`{"type":"object"}`),
		}
	}
	// read_file is in the read-only allow-list; write_file + bash are not.
	return []RegistryToolSchema{mk("read_file"), mk("write_file"), mk("bash")}
}

func (r *rovInnerRegistry) Dispatch(
	_ context.Context,
	call ToolCall,
	_ SandboxProvider,
) (HarnessToolResult, error) {
	r.dispatched = append(r.dispatched, call.Name)
	return HarnessToolResult{CallID: call.ID, Output: NewToolOutputSuccess("ok")}, nil
}

func (r *rovInnerRegistry) DispatchAll(
	ctx context.Context,
	calls []ToolCall,
	sandbox SandboxProvider,
) []DispatchOutcome {
	out := make([]DispatchOutcome, len(calls))
	for i, c := range calls {
		res, _ := r.Dispatch(ctx, c, sandbox)
		out[i].Result = res
	}
	return out
}

func (r *rovInnerRegistry) HasSubagentTools() bool     { return false }
func (r *rovInnerRegistry) IsAlwaysHalt(_ string) bool { return false }

var _ ToolRegistry = (*rovInnerRegistry)(nil)

// TestReadOnlyToolViewFiltersToReadonlyAllowlist mirrors the Rust unit test
// (SC-30): the eval-phase read-only view advertises + dispatches only the
// INTERSECTION of the wrapped catalogue with the read-only allow-list.
func TestReadOnlyToolViewFiltersToReadonlyAllowlist(t *testing.T) {
	inner := &rovInnerRegistry{}
	view := NewReadOnlyToolView(inner, readonlyEvalAllowSet())

	// Schemas: only read_file survives the intersection; write_file / bash gone.
	names := map[string]bool{}
	for _, s := range view.ActiveSchemas(nil) {
		names[s.Name] = true
	}
	if !names["read_file"] {
		t.Fatalf("read_file should be advertised, got %v", names)
	}
	if names["write_file"] {
		t.Fatalf("write_file should be hidden (not read-only), got %v", names)
	}
	if names["bash"] {
		t.Fatalf("bash should be hidden (not read-only), got %v", names)
	}
	if len(names) != 1 {
		t.Fatalf("only read_file should survive, got %v", names)
	}

	// read_file dispatches through to the inner registry.
	ok, err := view.Dispatch(context.Background(), ToolCall{
		ID:    "1",
		Name:  "read_file",
		Input: json.RawMessage(`{}`),
	}, nil)
	if err != nil {
		t.Fatalf("read_file dispatch returned error: %v", err)
	}
	if ok.Output.Kind != ToolOutputSuccess {
		t.Fatalf("read_file should succeed through inner, got %v", ok.Output.Kind)
	}

	// write_file is blocked with a RECOVERABLE error and never reaches inner.
	blocked, err := view.Dispatch(context.Background(), ToolCall{
		ID:    "2",
		Name:  "write_file",
		Input: json.RawMessage(`{}`),
	}, nil)
	if err != nil {
		t.Fatalf("blocked dispatch should return a recoverable ToolOutput, not a Go error: %v", err)
	}
	if blocked.Output.Kind != ToolOutputError {
		t.Fatalf("write_file should be blocked with an error ToolOutput, got %v", blocked.Output.Kind)
	}
	if !blocked.Output.Recoverable {
		t.Fatalf("blocked write_file error must be RECOVERABLE, got fatal")
	}
	if blocked.CallID != "2" {
		t.Fatalf("blocked result must echo the call id, got %q", blocked.CallID)
	}

	// The inner registry only ever saw read_file; write_file never reached it.
	if len(inner.dispatched) != 1 || inner.dispatched[0] != "read_file" {
		t.Fatalf("inner should have seen only [read_file], got %v", inner.dispatched)
	}
}
