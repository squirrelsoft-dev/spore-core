package middleware

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// recordingEmitter captures PatchEvents for assertions without depending on
// the observability package (which would create an import cycle).
type recordingEmitter struct {
	events []PatchEvent
}

func (r *recordingEmitter) EmitPatch(ev PatchEvent) { r.events = append(r.events, ev) }

// R10 regression: the middleware still registers at the highest BeforeTool
// priority and subscribes to BeforeTool (now also BeforeSession for identity).
func TestPatchMiddlewarePriorityIsHighestBeforeTool(t *testing.T) {
	mw := NewPatchToolCallsMiddleware("noop")
	if mw.Priority() != math.MinInt32+1 {
		t.Fatalf("priority: want %d, got %d", math.MinInt32+1, mw.Priority())
	}
	hasBeforeTool := false
	for _, h := range mw.Hooks() {
		if h == HookBeforeTool {
			hasBeforeTool = true
		}
	}
	if !hasBeforeTool {
		t.Fatalf("middleware must subscribe to BeforeTool: %v", mw.Hooks())
	}
}

// R10: in a chain, PatchToolCalls runs before other BeforeTool middleware, and
// still returns ContinueWithModification when it patches.
func TestPatchRunsBeforeOtherBeforeToolMiddleware(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	observer := newScripted("observer", []HookPoint{HookBeforeTool}, 0)
	if err := chain.Register(NewPatchToolCallsMiddleware("noop")); err != nil {
		t.Fatalf("register patch: %v", err)
	}
	if err := chain.Register(observer); err != nil {
		t.Fatalf("register observer: %v", err)
	}
	calls := []sporecore.ToolCall{{ID: "c1", Name: "", Input: json.RawMessage(`{}`)}}
	d, err := chain.FireBeforeTool(context.Background(), &calls, 1)
	if err != nil {
		t.Fatalf("fire: %v", err)
	}
	if d.Kind != DecisionContinueWithModification {
		t.Fatalf("decision: %v", d.Kind)
	}
	if calls[0].Name != "noop" {
		t.Fatalf("observer should see patched call, name=%q", calls[0].Name)
	}
}

// Identity captured at BeforeSession is stamped on the emitted event; the
// empty-name repair is classified as a dangling tool call.
func TestPatchEmitsEventWithCapturedIdentity(t *testing.T) {
	rec := &recordingEmitter{}
	mw := NewPatchToolCallsMiddleware("noop").WithObservability(rec)
	task := sporecore.NewTask("task", sporecore.SessionID("sess"), sporecore.ReActStrategy(0))
	s := sporecore.SessionID("sess")
	if _, err := mw.Handle(context.Background(), HookContext{
		Point: HookBeforeSession, Task: &task, SessionID: &s,
	}); err != nil {
		t.Fatalf("before session: %v", err)
	}
	calls := []sporecore.ToolCall{{ID: "c1", Name: "", Input: json.RawMessage(`{"x":1}`)}}
	if _, err := mw.Handle(context.Background(), HookContext{
		Point: HookBeforeTool, Calls: &calls, TurnNumber: 1,
	}); err != nil {
		t.Fatalf("before tool: %v", err)
	}
	if len(rec.events) != 1 {
		t.Fatalf("want 1 event, got %d", len(rec.events))
	}
	ev := rec.events[0]
	if ev.SessionID != "sess" || ev.TaskID != task.ID {
		t.Fatalf("identity not captured: sid=%q tid=%q", ev.SessionID, ev.TaskID)
	}
	if ev.Kind != PatchKindDanglingToolCall || ev.Reason != "empty tool name" {
		t.Fatalf("classification: kind=%q reason=%q", ev.Kind, ev.Reason)
	}
	if ev.CallID != "c1" || ev.ToolName != "noop" {
		t.Fatalf("call: id=%q tool=%q", ev.CallID, ev.ToolName)
	}
	if string(ev.Original) != `{"x":1}` {
		t.Fatalf("original params: %s", ev.Original)
	}
}
