package observability

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/middleware"
)

// patchFixture mirrors the shared JSON shape in
// fixtures/patch/patch_events_basic.json so the same fixture deserialises
// across all language implementations (Rust reference at
// `rust/crates/spore-core/src/middleware.rs::PatchFixture`).
type patchFixture struct {
	FallbackName          string            `json:"fallback_name"`
	InputCalls            []fixtureCall     `json:"input_calls"`
	ExpectedPatches       []expectedPatch   `json:"expected_patches"`
	ExpectedPatchCount    uint32            `json:"expected_patch_count"`
	ExpectedPatchesByTool map[string]uint32 `json:"expected_patches_by_tool"`
}

type fixtureCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

type expectedPatch struct {
	CallID    string          `json:"call_id"`
	ToolName  string          `json:"tool_name"`
	PatchType string          `json:"patch_type"`
	Original  json.RawMessage `json:"original"`
	Patched   json.RawMessage `json:"patched"`
}

// TestPatchEventsFixtureReplay loads patch_events_basic.json, runs each input
// call through the middleware wired to the observability adapter, and asserts
// the emitted patch events plus the rolled-up metrics match the fixture.
func TestPatchEventsFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/observability → ../../../fixtures/patch/...
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "patch", "patch_events_basic.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx patchFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	obs := NewInMemoryObservabilityProvider()
	mw := middleware.NewPatchToolCallsMiddleware(fx.FallbackName).
		WithObservability(NewPatchEmitterAdapter(obs))

	task := sporecore.NewTask("fixture", sid("sess"), sporecore.LoopStrategy{})
	s := sid("sess")
	if _, err := mw.Handle(context.Background(), middleware.HookContext{
		Point: middleware.HookBeforeSession, Task: &task, SessionID: &s,
	}); err != nil {
		t.Fatalf("before session: %v", err)
	}

	calls := make([]sporecore.ToolCall, len(fx.InputCalls))
	for i, c := range fx.InputCalls {
		calls[i] = sporecore.ToolCall{ID: c.ID, Name: c.Name, Input: c.Input}
	}
	if _, err := mw.Handle(context.Background(), middleware.HookContext{
		Point: middleware.HookBeforeTool, Calls: &calls, TurnNumber: 1,
	}); err != nil {
		t.Fatalf("before tool: %v", err)
	}

	patches := obs.PatchSpans(sid("sess"))
	if len(patches) != len(fx.ExpectedPatches) {
		t.Fatalf("patch count: want %d, got %d", len(fx.ExpectedPatches), len(patches))
	}
	for _, exp := range fx.ExpectedPatches {
		var found *PatchSpan
		for i := range patches {
			if patches[i].CallID == exp.CallID {
				found = &patches[i]
				break
			}
		}
		if found == nil {
			t.Fatalf("no patch span for call %q", exp.CallID)
		}
		if found.ToolName != exp.ToolName {
			t.Fatalf("call %q tool: want %q, got %q", exp.CallID, exp.ToolName, found.ToolName)
		}
		if !jsonEqual(t, found.OriginalParameters, exp.Original) {
			t.Fatalf("call %q original: want %s, got %s", exp.CallID, exp.Original, found.OriginalParameters)
		}
		if !jsonEqual(t, found.PatchedParameters, exp.Patched) {
			t.Fatalf("call %q patched: want %s, got %s", exp.CallID, exp.Patched, found.PatchedParameters)
		}
		switch exp.PatchType {
		case "dangling_tool_call":
			if found.PatchType.Kind != PatchKindDanglingToolCall {
				t.Fatalf("call %q patch_type: want dangling_tool_call, got %q", exp.CallID, found.PatchType.Kind)
			}
		default:
			t.Fatalf("unexpected patch_type %q", exp.PatchType)
		}
	}

	// SessionMetrics materializes for this middleware-only replay once an
	// outcome is recorded. No tool-call spans were emitted, so patch_rate is
	// 0.0 by the divide-by-zero guard; the fixture asserts count + by-tool.
	obs.SetSessionOutcome(sid("sess"), guideregistry.NewOutcomeSuccess())
	m, err := obs.GetSessionMetrics(context.Background(), sid("sess"))
	if err != nil || m == nil {
		t.Fatalf("metrics: err=%v m=%v", err, m)
	}
	if m.PatchCount != fx.ExpectedPatchCount {
		t.Fatalf("patch count: want %d, got %d", fx.ExpectedPatchCount, m.PatchCount)
	}
	for tool, n := range fx.ExpectedPatchesByTool {
		if m.PatchesByTool[tool] != n {
			t.Fatalf("patches_by_tool[%q]: want %d, got %d", tool, n, m.PatchesByTool[tool])
		}
	}
}

// jsonEqual compares two raw JSON values structurally (ignoring formatting).
func jsonEqual(t *testing.T, a, b json.RawMessage) bool {
	t.Helper()
	var av, bv any
	if err := json.Unmarshal(a, &av); err != nil {
		return false
	}
	if err := json.Unmarshal(b, &bv); err != nil {
		return false
	}
	ab, _ := json.Marshal(av)
	bb, _ := json.Marshal(bv)
	return string(ab) == string(bb)
}
