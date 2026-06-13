package sporecore

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// ============================================================================
// #140 — PausedState carries the pausing leaf's toolset handle
//
// Mirrors the Rust #140 tests in harness.rs:
//   - paused_state_toolset_back_compat_and_round_trip (AC1)
//   - consult_pause_carries_leaf_toolset_handle       (AC2a, Consult)
//   - clarification_pause_carries_leaf_toolset_handle (AC2a, Clarification)
//   - resume_consult_routes_pending_calls_through_carried_toolset (AC2b, load-bearing)
// ============================================================================

// reactScoped builds a bare ReAct task whose leaf carries a NON-EMPTY toolset
// handle, so the leaf's handle threads into the pause state (#140 AC2a) and
// resume routes pending calls through it (#140 AC2b).
func reactScoped(max uint32, handle string) Task {
	cfg := ReactPerLoop(max)
	cfg.Toolset = ToolsetRef(handle)
	ls := LoopStrategy{Kind: StrategyReAct, ReActCfg: &cfg}
	return NewTask("do something", SessionID("s1"), ls)
}

// TestPausedStateToolsetBackCompatAndRoundTrip — #140 AC1 (back-compat): a
// paused-state blob WITHOUT a "toolset" key still deserialises, defaulting to
// the empty handle. A non-empty handle round-trips and the field ALWAYS
// serialises (even when empty) for byte-parity.
func TestPausedStateToolsetBackCompatAndRoundTrip(t *testing.T) {
	// A pre-#140 PausedState JSON (no "toolset" key) — must default to "".
	pre140 := []byte(`{
		"session_id": "s",
		"task_id": "t",
		"turn_number": 1,
		"session_state": {"messages": [], "extras": {}},
		"pending_tool_calls": [],
		"approved_results": [],
		"human_request": null,
		"task": ` + taskJSON(t, reactTask(5)) + `,
		"budget_used": ` + budgetJSON(t) + `,
		"child_state": null
	}`)
	var parsed PausedState
	if err := json.Unmarshal(pre140, &parsed); err != nil {
		t.Fatalf("pre-#140 blob must deserialise: %v", err)
	}
	if parsed.Toolset != ToolsetRef("") {
		t.Fatalf("missing toolset must default to the empty handle, got %q", parsed.Toolset)
	}

	// The empty handle ALWAYS serialises (never skipped) for byte-parity.
	wire, err := json.Marshal(parsed)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(wire), `"toolset":""`) {
		t.Fatalf("empty toolset must serialise explicitly, not be skipped: %s", wire)
	}

	// A non-empty handle round-trips by value.
	scoped := parsed
	scoped.Toolset = ToolsetRef("scoped")
	scopedWire, err := json.Marshal(scoped)
	if err != nil {
		t.Fatalf("marshal scoped: %v", err)
	}
	var back PausedState
	if err := json.Unmarshal(scopedWire, &back); err != nil {
		t.Fatalf("unmarshal scoped: %v", err)
	}
	if back.Toolset != ToolsetRef("scoped") {
		t.Fatalf("non-empty toolset must round-trip, got %q", back.Toolset)
	}

	// The same back-compat + always-serialise contract holds for ChildPausedState.
	childPre140 := []byte(`{
		"session_id": "c",
		"task_id": "ct",
		"turn_number": 1,
		"session_state": {"messages": [], "extras": {}},
		"pending_tool_calls": [],
		"approved_results": [],
		"human_request": null,
		"task": ` + taskJSON(t, reactTask(1)) + `,
		"budget_used": ` + budgetJSON(t) + `,
		"parent_tool_call_id": "p"
	}`)
	var child ChildPausedState
	if err := json.Unmarshal(childPre140, &child); err != nil {
		t.Fatalf("pre-#140 child blob must deserialise: %v", err)
	}
	if child.Toolset != ToolsetRef("") {
		t.Fatalf("missing child toolset must default to empty, got %q", child.Toolset)
	}
	childWire, err := json.Marshal(child)
	if err != nil {
		t.Fatalf("marshal child: %v", err)
	}
	if !strings.Contains(string(childWire), `"toolset":""`) {
		t.Fatalf("empty child toolset must serialise explicitly: %s", childWire)
	}
}

// TestConsultPauseCarriesLeafToolsetHandle — #140 AC2a (Consult): a Consult
// pause from a leaf carrying ToolsetRef("scoped") returns a PausedState whose
// Toolset is that handle, proving the leaf's handle is captured at the pause
// site. The Consult path is the load-bearing one.
func TestConsultPauseCarriesLeafToolsetHandle(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c0", Name: "ask_advice", Input: json.RawMessage(`{"kind":"advice"}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	// Register the "scoped" toolset handle (presence only) so Validate() at run
	// entry passes. With NO per-key catalogue, effectiveToolRegistry("scoped")
	// falls back to the injected ToolRegistry, so the scripted ask_advice tool
	// still dispatches — we only need the pause to capture the handle here.
	reg := NewScriptedToolRegistry()
	reg.Push(NewToolOutputConsult(consultReq()))
	cfg.ToolRegistry = reg
	cfg = cfg.WithRegistryToolset("scoped", reg)
	h := NewStandardHarness(cfg)

	r := h.Run(context.Background(), NewHarnessRunOptions(reactScoped(5, "scoped")))
	if r.Kind != RunConsult {
		t.Fatalf("expected RunConsult, got %q (%+v)", r.Kind, r)
	}
	if r.State == nil {
		t.Fatal("nil state")
	}
	if r.State.Toolset != ToolsetRef("scoped") {
		t.Fatalf("consult pause must carry the leaf's toolset handle, got %q", r.State.Toolset)
	}
}

// TestClarificationPauseCarriesLeafToolsetHandle — #140 AC2a (Clarification):
// the AwaitingClarification (#81) leaf-pause path likewise carries the leaf's
// toolset handle.
func TestClarificationPauseCarriesLeafToolsetHandle(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "ask1", Name: "ask_user_question", Input: json.RawMessage(`{"question":"which?"}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputAwaitingClarification, Question: "which?"})
	cfg.ToolRegistry = reg
	// Register the "scoped" handle (presence only) so Validate() passes; with no
	// per-key catalogue, dispatch falls back to the injected ToolRegistry.
	cfg = cfg.WithRegistryToolset("scoped", reg)
	h := NewStandardHarness(cfg)

	r := h.Run(context.Background(), NewHarnessRunOptions(reactScoped(5, "scoped")))
	if r.Kind != RunWaitingForHuman {
		t.Fatalf("expected RunWaitingForHuman, got %q (%+v)", r.Kind, r)
	}
	if r.State == nil {
		t.Fatal("nil state")
	}
	if r.State.Toolset != ToolsetRef("scoped") {
		t.Fatalf("clarification pause must carry the leaf's toolset handle, got %q", r.State.Toolset)
	}
}

// TestResumeConsultRoutesPendingCallsThroughCarriedToolset — #140 AC2b (THE
// load-bearing regression guard): a PausedState whose Toolset is "scoped"
// resumes pending per-node tool calls against the SCOPED catalogue. A tool
// registered ONLY under "scoped" (absent from the global catalogue) dispatches
// successfully on resume.
//
// NEGATIVE CONTROL: the identical state with an EMPTY handle resumes against the
// global catalogue, where the scoped-only tool is unknown → a tool-error result.
// This proves the carried handle is behaviorally load-bearing, not cosmetic.
func TestResumeConsultRoutesPendingCallsThroughCarriedToolset(t *testing.T) {
	// A paused state with TWO pending calls: the head consult call (gets the
	// answer injected) and a trailing call to the scoped-only tool (dispatched
	// through effectiveToolRegistry(sessionID, state.Toolset)).
	makeState := func(handle string) PausedState {
		cfg := ReactPerLoop(5)
		cfg.Toolset = ToolsetRef(handle)
		ls := LoopStrategy{Kind: StrategyReAct, ReActCfg: &cfg}
		return PausedState{
			SessionID:  SessionID("s"),
			TaskID:     TaskID("t"),
			TurnNumber: 1,
			PendingToolCalls: []ToolCall{
				{ID: "consult", Name: "ask_advice", Input: json.RawMessage(`{"kind":"advice"}`)},
				{ID: "scoped", Name: "scoped_only", Input: json.RawMessage(`{}`)},
			},
			Task:    NewTask("do something", SessionID("s"), ls),
			Toolset: ToolsetRef(handle),
		}
	}

	// scopedDispatchedOK reports whether the scoped-only tool's pending call
	// dispatched successfully: the echo catalogue tool returns Success("echo"),
	// recorded as a Tool-role message with text "echo"; an unknown-tool error is
	// recorded with a "[error]" prefix (NoopContextManager.AppendToolResult).
	scopedDispatchedOK := func(messages []Message) bool {
		for _, m := range messages {
			if m.Role == RoleTool && m.Content.Text == "echo" {
				return true
			}
		}
		return false
	}
	scopedErrored := func(messages []Message) bool {
		for _, m := range messages {
			if m.Role == RoleTool && strings.Contains(m.Content.Text, "[error]") {
				return true
			}
		}
		return false
	}

	// harnessWithScopedCatalogue builds a harness with a SCOPED catalogue
	// ("scoped" → scoped_only) AND a GLOBAL catalogue ("global_only", which does
	// NOT contain scoped_only) plus a worker agent that emits a final response so
	// the re-entered ReAct window terminates cleanly. The global catalogue is what
	// the EMPTY handle falls back to — making scoped_only a genuine unknown tool
	// there.
	harnessWithScopedCatalogue := func() *StandardHarness {
		a := NewMockAgent("t")
		a.Push(NewFinalResponse("resumed-done", turnUsage()))
		cfg := standardCfg(a)
		cfg.CatalogueRegistry = catalogueRegistryWith("global_only")
		cfg.ToolsetCatalogues = map[string]*StandardToolRegistry{
			"scoped": catalogueRegistryWith("scoped_only"),
		}
		return NewStandardHarness(cfg)
	}

	// ── Positive: carried handle "scoped" routes to the scoped catalogue ──
	h := harnessWithScopedCatalogue()
	r := h.ResumeConsult(context.Background(), makeState("scoped"), NewConsultAnswer("ok"), nil)
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %q (%+v)", r.Kind, r)
	}
	if !scopedDispatchedOK(r.SessionState.Messages) {
		t.Fatalf("scoped-only tool must dispatch successfully when the carried handle is 'scoped': %+v", r.SessionState.Messages)
	}

	// ── Negative control: EMPTY handle falls back to the global catalogue ──
	h = harnessWithScopedCatalogue()
	r = h.ResumeConsult(context.Background(), makeState(""), NewConsultAnswer("ok"), nil)
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %q (%+v)", r.Kind, r)
	}
	if scopedDispatchedOK(r.SessionState.Messages) {
		t.Fatalf("with the EMPTY handle the scoped-only tool is unknown → must NOT dispatch successfully: %+v", r.SessionState.Messages)
	}
	if !scopedErrored(r.SessionState.Messages) {
		t.Fatalf("the scoped-only call must surface a tool error under the empty handle: %+v", r.SessionState.Messages)
	}
}

// taskJSON marshals a Task to its JSON string for embedding in a raw blob.
func taskJSON(t *testing.T, task Task) string {
	t.Helper()
	b, err := json.Marshal(task)
	if err != nil {
		t.Fatalf("marshal task: %v", err)
	}
	return string(b)
}

// budgetJSON marshals a default BudgetSnapshot to its JSON string.
func budgetJSON(t *testing.T) string {
	t.Helper()
	b, err := json.Marshal(BudgetSnapshot{})
	if err != nil {
		t.Fatalf("marshal budget: %v", err)
	}
	return string(b)
}
