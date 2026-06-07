package sporecore

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Mirrors rust/crates/spore-core/tests/plan_phase_fixture_replay.rs.
//
// Loads fixtures/model_responses/harness/plan_phase_basic.jsonl and drives a
// StandardHarness with LoopStrategy PlanExecute against each recorded planner
// response, asserting — identically to Rust — that:
//  1. The plan turn's recorded FinalResponse is captured into the exact
//     PlanArtifact (tasks + rationale), stored under extras["plan_execute"].
//  2. The fenced ```json variant is captured identically (fence-strip rule).
//  3. The plan turn is consumed and parsed into a non-empty task list, so the
//     run proceeds into the execute phase (issue #59). This fixture provides
//     ONLY the single plan turn, so the first execute step's ReAct sub-loop
//     exhausts the replay and the run aborts with StepFailed{TaskIndex: 0} —
//     proving the harness consumed the planner response and entered execute.
//
// Never edit the fixture to make a failing implementation pass.

func planFixtureExchanges(t *testing.T) []RecordedExchange {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	path := filepath.Join(dir, "..", "..", "fixtures", "model_responses", "harness", "plan_phase_basic.jsonl")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var out []RecordedExchange
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		replay, err := ParseReplayJSONL(line, ProviderInfo{
			Name: "anthropic", ModelID: "fixture", ContextWindow: 200_000,
		})
		if err != nil {
			t.Fatalf("parse fixture line: %v", err)
		}
		out = append(out, replay.exchanges...)
	}
	return out
}

// responseText extracts the single text block from a recorded response.
func responseText(t *testing.T, ex RecordedExchange) string {
	t.Helper()
	for _, b := range ex.Response.Content {
		if b.Type == ContentBlockTypeText {
			return b.Text
		}
	}
	t.Fatal("recorded response has no text block")
	return ""
}

// drivePlanFixture replays a single plan exchange through the full harness and
// asserts the run consumes the planner turn, parses a non-empty task list, and
// enters the execute phase — where the single-exchange replay exhausts on the
// first step and aborts with StepFailed{TaskIndex: 0} (issue #59, Q5). Mirrors
// the Rust plan_phase_fixture_replay drive_plan_phase.
func drivePlanFixture(t *testing.T, ex RecordedExchange) {
	t.Helper()
	replay := NewReplayModel([]RecordedExchange{ex}, ProviderInfo{
		Name: "anthropic", ModelID: "fixture", ContextWindow: 200_000,
	})
	agent := NewModelAgent(AgentID("planner"), replay)
	cfg := HarnessConfig{
		Agent:             agent,
		ToolRegistry:      NewScriptedToolRegistry(),
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
	}
	h := NewStandardHarness(cfg)
	// #124: PlanExecute now genuinely recurses into its plan/execute children, so
	// the strategy must carry a real config (both phases default ReAct leaves).
	// A.5: the structured plan slot's leaf carries an output schema (empty key).
	planChild := ReActStrategy(^uint32(0))
	planChild.ReActCfg.Output = func() *SchemaRef { s := SchemaRef(""); return &s }()
	execChild := ReActStrategy(^uint32(0))
	task := NewTask("build something", SessionID("plan-fixture"), PlanExecuteStrategy(PlanExecuteConfig{
		Plan:    &planChild,
		Execute: &execChild,
	}))
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltStepFailed {
		t.Fatalf("expected StepFailed, got %+v", r)
	}
	if r.Reason.TaskIndex != 0 {
		t.Fatalf("task index = %d, want 0", r.Reason.TaskIndex)
	}
	if r.Turns < 1 {
		t.Fatalf("turns = %d, want >= 1 (at least the plan turn)", r.Turns)
	}
}

func TestPlanPhaseFixtureCapturesPlainJSON(t *testing.T) {
	exchanges := planFixtureExchanges(t)
	if len(exchanges) < 2 {
		t.Fatalf("fixture has %d cases, want >= 2", len(exchanges))
	}
	drivePlanFixture(t, exchanges[0])

	a, err := CapturePlanArtifact(responseText(t, exchanges[0]))
	if err != nil {
		t.Fatalf("capture: %v", err)
	}
	want := []string{"scaffold the project", "add the argument parser", "write the integration tests"}
	if !equalStrings(a.Tasks, want) {
		t.Fatalf("tasks = %v, want %v", a.Tasks, want)
	}
	if a.Rationale != "deliver a working CLI incrementally" {
		t.Fatalf("rationale = %q", a.Rationale)
	}
}

func TestPlanPhaseFixtureCapturesFencedJSON(t *testing.T) {
	exchanges := planFixtureExchanges(t)
	if len(exchanges) < 2 {
		t.Fatalf("fixture has %d cases, want >= 2", len(exchanges))
	}
	drivePlanFixture(t, exchanges[1])

	a, err := CapturePlanArtifact(responseText(t, exchanges[1]))
	if err != nil {
		t.Fatalf("capture: %v", err)
	}
	want := []string{"draft the outline", "write the reference section"}
	if !equalStrings(a.Tasks, want) {
		t.Fatalf("tasks = %v, want %v", a.Tasks, want)
	}
	if a.Rationale != "docs follow the code" {
		t.Fatalf("rationale = %q", a.Rationale)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
