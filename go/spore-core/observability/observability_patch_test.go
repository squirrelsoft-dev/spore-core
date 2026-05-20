package observability

import (
	"context"
	"encoding/json"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/middleware"
)

// ─── helpers ────────────────────────────────────────────────────────────────

// patchSpan builds a ParameterCoercion patch span for direct EmitPatch tests.
func patchSpan(session, spanID, callID, tool string) PatchSpan {
	return NewPatchSpan(
		SpanBase{
			SpanID:    SpanID(spanID),
			SessionID: sid(session),
			TaskID:    tid("t1"),
			Kind:      SpanKindPatch,
			StartedAt: ts("2026-05-16T00:00:00Z"),
			EndedAt:   ts("2026-05-16T00:00:00Z"),
			Status:    NewStatusOk(),
		},
		callID,
		tool,
		json.RawMessage(`{"a":"1"}`),
		json.RawMessage(`{"a":1}`),
		NewPatchParameterCoercion("a", "string", "number"),
	)
}

func patchToolCallSpan(session, spanID, tool string) ToolCallSpan {
	return ToolCallSpan{
		Base: SpanBase{
			SpanID:    SpanID(spanID),
			SessionID: sid(session),
			TaskID:    tid("t1"),
			Kind:      SpanKindToolCall,
			StartedAt: ts("2026-05-16T00:00:00Z"),
			EndedAt:   ts("2026-05-16T00:00:00Z"),
			Status:    NewStatusOk(),
		},
		ToolName:    tool,
		CallID:      spanID,
		SandboxMode: "none",
	}
}

func toolCall(id, name string, input json.RawMessage) sporecore.ToolCall {
	return sporecore.ToolCall{ID: id, Name: name, Input: input}
}

// wiredPatch builds a PatchToolCallsMiddleware wired to obs via the adapter.
func wiredPatch(obs *InMemoryObservabilityProvider) *middleware.PatchToolCallsMiddleware {
	return middleware.NewPatchToolCallsMiddleware("noop").
		WithObservability(NewPatchEmitterAdapter(obs))
}

// runPatch drives identity capture (BeforeSession) then a BeforeTool fire
// directly on the middleware so the test owns the calls slice.
func runPatch(t *testing.T, mw *middleware.PatchToolCallsMiddleware, calls *[]sporecore.ToolCall) middleware.MiddlewareDecision {
	t.Helper()
	task := sporecore.NewTask("test task", sid("sess"), sporecore.LoopStrategy{})
	s := sid("sess")
	if _, err := mw.Handle(context.Background(), middleware.HookContext{
		Point: middleware.HookBeforeSession, Task: &task, SessionID: &s,
	}); err != nil {
		t.Fatalf("before session: %v", err)
	}
	d, err := mw.Handle(context.Background(), middleware.HookContext{
		Point: middleware.HookBeforeTool, Calls: calls, TurnNumber: 1,
	})
	if err != nil {
		t.Fatalf("before tool: %v", err)
	}
	return d
}

// ─── Direct EmitPatch (interface-level) ─────────────────────────────────────

// R1/R5: EmitPatch records a warn-level span that appears in the trace.
// R3: both original and patched recorded and differ.
func TestEmitPatchAppearsInTraceAsWarn(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	sp := patchSpan("s1", "p1", "c1", "shell")
	if sp.Level != SpanLevelWarn {
		t.Fatalf("level: want warn, got %q", sp.Level)
	}
	obs.EmitPatch(sp)
	trace, err := obs.GetTrace(context.Background(), sid("s1"))
	if err != nil {
		t.Fatal(err)
	}
	if len(trace) != 1 {
		t.Fatalf("trace len: want 1, got %d", len(trace))
	}
	if trace[0].GetBase().Kind != SpanKindPatch {
		t.Fatalf("kind: %q", trace[0].GetBase().Kind)
	}
	got := obs.PatchSpans(sid("s1"))
	if len(got) != 1 {
		t.Fatalf("patch spans: want 1, got %d", len(got))
	}
	if string(got[0].OriginalParameters) == string(got[0].PatchedParameters) {
		t.Fatalf("original and patched should differ")
	}
}

// R4: all three PatchType variants round-trip through the span store.
func TestPatchTypeVariants(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	mk := func(spanID, callID string, pt PatchType) PatchSpan {
		return NewPatchSpan(SpanBase{
			SpanID: SpanID(spanID), SessionID: sid("s1"), TaskID: tid("t1"),
			Kind: SpanKindPatch, StartedAt: ts("2026-05-16T00:00:00Z"),
			EndedAt: ts("2026-05-16T00:00:00Z"), Status: NewStatusOk(),
		}, callID, "shell", json.RawMessage(`{}`), json.RawMessage(`{}`), pt)
	}
	obs.EmitPatch(mk("p1", "c1", NewPatchMalformedJSON("bad json")))
	obs.EmitPatch(mk("p2", "c2", NewPatchDanglingToolCall("empty tool name")))
	obs.EmitPatch(mk("p3", "c3", NewPatchParameterCoercion("count", "string", "number")))

	got := obs.PatchSpans(sid("s1"))
	if len(got) != 3 {
		t.Fatalf("want 3 patch spans, got %d", len(got))
	}
	if got[0].PatchType.Kind != PatchKindMalformedJSON || got[0].PatchType.Error != "bad json" {
		t.Fatalf("malformed: %+v", got[0].PatchType)
	}
	if got[1].PatchType.Kind != PatchKindDanglingToolCall || got[1].PatchType.Reason != "empty tool name" {
		t.Fatalf("dangling: %+v", got[1].PatchType)
	}
	if got[2].PatchType.Kind != PatchKindParameterCoercion ||
		got[2].PatchType.Field != "count" || got[2].PatchType.From != "string" || got[2].PatchType.To != "number" {
		t.Fatalf("coercion: %+v", got[2].PatchType)
	}
}

// ─── Metrics (R6/R7/R8) ─────────────────────────────────────────────────────

// R6/R7/R8: patch metrics roll up. 2 patches over 4 tool calls = 0.5.
func TestPatchMetricsCountRateAndByTool(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	obs.EmitTurn(turnSpan("s1", "t1", 1, 10, 5))
	obs.EmitToolCall(patchToolCallSpan("s1", "tc1", "shell"))
	obs.EmitToolCall(patchToolCallSpan("s1", "tc2", "shell"))
	obs.EmitToolCall(patchToolCallSpan("s1", "tc3", "edit"))
	obs.EmitToolCall(patchToolCallSpan("s1", "tc4", "edit"))
	obs.EmitPatch(patchSpan("s1", "p1", "tc1", "shell"))
	obs.EmitPatch(patchSpan("s1", "p2", "tc2", "shell"))

	m, err := obs.GetSessionMetrics(context.Background(), sid("s1"))
	if err != nil || m == nil {
		t.Fatalf("metrics: err=%v m=%v", err, m)
	}
	if m.PatchCount != 2 {
		t.Fatalf("patch count: want 2, got %d", m.PatchCount)
	}
	if m.PatchRate != 0.5 {
		t.Fatalf("patch rate: want 0.5, got %v", m.PatchRate)
	}
	if m.PatchesByTool["shell"] != 2 {
		t.Fatalf("patches_by_tool[shell]: want 2, got %d", m.PatchesByTool["shell"])
	}
	if _, ok := m.PatchesByTool["edit"]; ok {
		t.Fatalf("edit should not appear in patches_by_tool: %+v", m.PatchesByTool)
	}
}

// R7: zero tool calls → patch_rate is 0.0, never a divide-by-zero.
func TestPatchRateZeroWhenNoToolCalls(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	obs.EmitTurn(turnSpan("s1", "t1", 1, 10, 5))
	obs.EmitPatch(patchSpan("s1", "p1", "c1", "shell"))

	m, err := obs.GetSessionMetrics(context.Background(), sid("s1"))
	if err != nil || m == nil {
		t.Fatalf("metrics: err=%v m=%v", err, m)
	}
	if m.PatchCount != 1 {
		t.Fatalf("patch count: want 1, got %d", m.PatchCount)
	}
	if m.PatchRate != 0.0 {
		t.Fatalf("patch rate: want 0.0, got %v", m.PatchRate)
	}
}

// ─── End-to-end middleware → adapter → provider (R1-R5, R9, R10, silent) ────

// R1 + R3: every patch emits exactly one warn span recording original+patched.
func TestPatchEmitsOneWarnSpanWithOriginalAndPatched(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	mw := wiredPatch(obs)
	calls := []sporecore.ToolCall{toolCall("c1", "", json.RawMessage(`{"command":"ls"}`))}
	d := runPatch(t, mw, &calls)
	if d.Kind != middleware.DecisionContinueWithModification {
		t.Fatalf("decision: %v", d.Kind)
	}
	got := obs.PatchSpans(sid("sess"))
	if len(got) != 1 {
		t.Fatalf("want 1 patch span, got %d", len(got))
	}
	if got[0].Level != SpanLevelWarn {
		t.Fatalf("level: %q", got[0].Level)
	}
	if string(got[0].OriginalParameters) != `{"command":"ls"}` {
		t.Fatalf("original: %s", got[0].OriginalParameters)
	}
	if string(got[0].PatchedParameters) != `{"command":"ls"}` {
		t.Fatalf("patched: %s", got[0].PatchedParameters)
	}
}

// R2: no patch needed → no span, decision is plain Continue.
func TestNoPatchEmitsNoSpan(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	mw := wiredPatch(obs)
	calls := []sporecore.ToolCall{toolCall("c1", "shell", json.RawMessage(`{}`))}
	d := runPatch(t, mw, &calls)
	if d.Kind != middleware.DecisionContinue {
		t.Fatalf("decision: %v", d.Kind)
	}
	if len(obs.PatchSpans(sid("sess"))) != 0 {
		t.Fatalf("expected no patch spans")
	}
	trace, _ := obs.GetTrace(context.Background(), sid("sess"))
	for _, s := range trace {
		if s.GetBase().Kind == SpanKindPatch {
			t.Fatalf("trace must contain no patch span")
		}
	}
}

// R4: the empty-name repair is classified as DanglingToolCall with the
// patched name and call id on the span.
func TestEmptyNameClassifiedAsDanglingToolCall(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	mw := wiredPatch(obs)
	calls := []sporecore.ToolCall{toolCall("c1", "", json.RawMessage(`{}`))}
	runPatch(t, mw, &calls)
	got := obs.PatchSpans(sid("sess"))
	if len(got) != 1 {
		t.Fatalf("want 1 patch span, got %d", len(got))
	}
	p := got[0]
	if p.PatchType.Kind != PatchKindDanglingToolCall {
		t.Fatalf("kind: %v", p.PatchType.Kind)
	}
	if p.PatchType.Reason != "empty tool name" {
		t.Fatalf("reason: %q", p.PatchType.Reason)
	}
	if p.ToolName != "noop" {
		t.Fatalf("tool name (patched): %q", p.ToolName)
	}
	if p.CallID != "c1" {
		t.Fatalf("call id: %q", p.CallID)
	}
}

// R5: the patch event is present in GetTrace.
func TestTraceContainsPatchEvent(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	mw := wiredPatch(obs)
	calls := []sporecore.ToolCall{toolCall("c1", "", json.RawMessage(`{}`))}
	runPatch(t, mw, &calls)
	trace, err := obs.GetTrace(context.Background(), sid("sess"))
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, s := range trace {
		if s.GetBase().Kind == SpanKindPatch {
			found = true
		}
	}
	if !found {
		t.Fatalf("patch event missing from trace")
	}
}

// R9: a batch of N patched calls emits N patch spans (well-formed calls skip).
func TestBatchEmitsOneSpanPerPatch(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	mw := wiredPatch(obs)
	calls := []sporecore.ToolCall{
		toolCall("c1", "", json.RawMessage(`{}`)),
		toolCall("c2", "shell", json.RawMessage(`{}`)),
		toolCall("c3", "  ", json.RawMessage(`{}`)),
	}
	runPatch(t, mw, &calls)
	if n := len(obs.PatchSpans(sid("sess"))); n != 2 {
		t.Fatalf("want 2 patch spans (c1, c3), got %d", n)
	}
}

// Silent: without an injected emitter, patching still works and emits nothing.
func TestPatchWithoutObservabilityIsSilent(t *testing.T) {
	mw := middleware.NewPatchToolCallsMiddleware("noop")
	calls := []sporecore.ToolCall{toolCall("c1", "", json.RawMessage(`{}`))}
	d := runPatch(t, mw, &calls)
	if d.Kind != middleware.DecisionContinueWithModification {
		t.Fatalf("decision: %v", d.Kind)
	}
	if calls[0].Name != "noop" {
		t.Fatalf("name not patched: %q", calls[0].Name)
	}
}

// SessionMetrics materializes for a patch-only session (no turns) once an
// outcome is recorded; patch_rate is 0.0 by the divide-by-zero guard.
func TestPatchOnlySessionMetrics(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	mw := wiredPatch(obs)
	calls := []sporecore.ToolCall{
		toolCall("c1", "", json.RawMessage(`{}`)),
		toolCall("c2", "   ", json.RawMessage(`{}`)),
	}
	runPatch(t, mw, &calls)
	obs.SetSessionOutcome(sid("sess"), guideregistry.NewOutcomeSuccess())
	m, err := obs.GetSessionMetrics(context.Background(), sid("sess"))
	if err != nil || m == nil {
		t.Fatalf("metrics: err=%v m=%v", err, m)
	}
	if m.PatchCount != 2 {
		t.Fatalf("patch count: want 2, got %d", m.PatchCount)
	}
	if m.PatchRate != 0.0 {
		t.Fatalf("patch rate: want 0.0, got %v", m.PatchRate)
	}
	if m.PatchesByTool["noop"] != 2 {
		t.Fatalf("patches_by_tool[noop]: want 2, got %d", m.PatchesByTool["noop"])
	}
}
