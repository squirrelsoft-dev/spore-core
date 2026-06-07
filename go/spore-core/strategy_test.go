package sporecore

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

// strategyFixturePath returns the absolute path to a fixtures/strategy/ file.
func strategyFixturePath(t *testing.T, name string) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	return filepath.Join(dir, "..", "..", "fixtures", "strategy", name)
}

// compact returns the minified form of a JSON document.
func compactJSON(t *testing.T, raw []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		t.Fatalf("compact: %v", err)
	}
	return buf.Bytes()
}

// cordycepsTree builds the canonical Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]
// tree mirrored by fixtures/strategy/cordyceps_tree.json.
func cordycepsTree() LoopStrategy {
	plan := LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget:  BudgetPolicy{Kind: BudgetPerLoop, Value: 4},
		Agent:   AgentRef("planner"),
		Toolset: ToolsetRef("plan-tools"),
	}}
	execInner := LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget:  BudgetPolicy{Kind: BudgetPerLoop, Value: 12},
		Agent:   AgentRef("executor"),
		Toolset: ToolsetRef("exec-tools"),
	}}
	execute := SelfVerifyingStrategy(SelfVerifyingConfig{
		Inner:     &execInner,
		Evaluator: SchemaRef("exec-evaluator"),
	})
	planExec := PlanExecuteStrategy(PlanExecuteConfig{
		Plan:    &plan,
		Execute: &execute,
	})
	return RalphStrategy(RalphConfig{
		Inner: &planExec,
		Agent: AgentRef("ralph-agent"),
	})
}

// ---------------------------------------------------------------------------
// Per-variant round-trip
// ---------------------------------------------------------------------------

func TestLoopStrategyPerVariantRoundTrip(t *testing.T) {
	out := SchemaRef("out")
	cases := []LoopStrategy{
		{Kind: StrategyReAct, ReActCfg: &ReactConfig{
			Budget:  BudgetPolicy{Kind: BudgetPerLoop, Value: 7},
			Agent:   AgentRef("a"),
			Toolset: ToolsetRef("t"),
			Output:  &out,
		}},
		ReActStrategy(3),
		PlanExecuteStrategy(PlanExecuteSimple(nil)),
		PlanExecuteStrategy(PlanExecuteSimple(&ModelConfig{Provider: "anthropic", ModelID: "m"})),
		SelfVerifyingStrategy(SelfVerifyingConfig{
			Inner:     PtrStrategy(ReActStrategy(2)),
			Evaluator: SchemaRef("ev"),
		}),
		RalphStrategy(RalphConfig{
			Inner: PtrStrategy(ReActStrategy(1)),
			Agent: AgentRef("r"),
		}),
		HillClimbingStrategy(HillClimbingConfig{
			Inner:                 PtrStrategy(ReActStrategy(5)),
			Direction:             OptimizationMaximize,
			MaxStagnation:         3,
			RevertOnNoImprovement: true,
			MinImprovementDelta:   0.25,
			Evaluator:             AgentRef("metric"),
		}),
	}
	for i, s := range cases {
		data, err := json.Marshal(s)
		if err != nil {
			t.Fatalf("case %d marshal: %v", i, err)
		}
		var back LoopStrategy
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("case %d unmarshal: %v (json=%s)", i, err, data)
		}
		if !reflect.DeepEqual(s, back) {
			t.Fatalf("case %d round-trip mismatch:\n want %+v\n got  %+v\n json %s", i, s, back, data)
		}
	}
}

// ---------------------------------------------------------------------------
// react tag + omitted output
// ---------------------------------------------------------------------------

func TestReActTagAndOmittedOutput(t *testing.T) {
	data, err := json.Marshal(ReActStrategy(8))
	if err != nil {
		t.Fatal(err)
	}
	want := `{"kind":"react","budget":{"kind":"per_loop","value":8},"agent":"","toolset":""}`
	if string(data) != want {
		t.Fatalf("got %s, want %s", data, want)
	}
}

func TestReActOutputPresentWhenSet(t *testing.T) {
	out := SchemaRef("schema-1")
	s := LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget:  BudgetPolicy{Kind: BudgetPerLoop, Value: 1},
		Agent:   AgentRef("a"),
		Toolset: ToolsetRef("t"),
		Output:  &out,
	}}
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"kind":"react","budget":{"kind":"per_loop","value":1},"agent":"a","toolset":"t","output":"schema-1"}`
	if string(data) != want {
		t.Fatalf("got %s, want %s", data, want)
	}
}

// ---------------------------------------------------------------------------
// Handle newtypes round-trip as bare strings
// ---------------------------------------------------------------------------

func TestHandleRefsRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		val  any
		want string
	}{
		{AgentRef("x"), `"x"`},
		{ToolsetRef("y"), `"y"`},
		{SchemaRef("z"), `"z"`},
	} {
		data, err := json.Marshal(tc.val)
		if err != nil {
			t.Fatal(err)
		}
		if string(data) != tc.want {
			t.Fatalf("got %s, want %s", data, tc.want)
		}
	}
	var a AgentRef
	if err := json.Unmarshal([]byte(`"x"`), &a); err != nil {
		t.Fatal(err)
	}
	if a != AgentRef("x") {
		t.Fatalf("AgentRef round-trip got %q", a)
	}
}

// ---------------------------------------------------------------------------
// Cordyceps tree round-trip + byte-identity
// ---------------------------------------------------------------------------

func TestCordycepsTreeRoundTrip(t *testing.T) {
	tree := cordycepsTree()
	data, err := json.Marshal(tree)
	if err != nil {
		t.Fatal(err)
	}
	var back LoopStrategy
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(tree, back) {
		t.Fatalf("cordyceps round-trip mismatch:\n want %+v\n got %+v", tree, back)
	}
	want := `{"kind":"ralph","inner":{"kind":"plan_execute","plan":{"kind":"react","budget":{"kind":"per_loop","value":4},"agent":"planner","toolset":"plan-tools"},"execute":{"kind":"self_verifying","inner":{"kind":"react","budget":{"kind":"per_loop","value":12},"agent":"executor","toolset":"exec-tools"},"evaluator":"exec-evaluator"}},"agent":"ralph-agent"}`
	if string(data) != want {
		t.Fatalf("cordyceps bytes mismatch:\n got  %s\n want %s", data, want)
	}
}

// ---------------------------------------------------------------------------
// StrategyRef BuiltIn / Custom
// ---------------------------------------------------------------------------

func TestStrategyRefRoundTrip(t *testing.T) {
	builtIn := StrategyRef{Kind: StrategyRefBuiltIn, BuiltIn: PtrStrategy(cordycepsTree())}
	custom := StrategyRef{Kind: StrategyRefCustom, Custom: "my-harness::DoubleVerify"}
	for i, r := range []StrategyRef{builtIn, custom} {
		data, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("case %d marshal: %v", i, err)
		}
		var back StrategyRef
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("case %d unmarshal: %v", i, err)
		}
		if !reflect.DeepEqual(r, back) {
			t.Fatalf("case %d round-trip mismatch:\n want %+v\n got %+v", i, r, back)
		}
	}
	// Adjacent tagging on kind/value, no collision with the nested LoopStrategy kind.
	data, err := json.Marshal(custom)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"kind":"custom","value":"my-harness::DoubleVerify"}` {
		t.Fatalf("custom bytes: %s", data)
	}
}

// ---------------------------------------------------------------------------
// Stub Run returns a placeholder, never panics
// ---------------------------------------------------------------------------

func TestStubRunReturnsPending(t *testing.T) {
	strategies := []LoopStrategy{
		ReActStrategy(1),
		PlanExecuteStrategy(PlanExecuteSimple(nil)),
		SelfVerifyingStrategy(SelfVerifyingConfig{Inner: PtrStrategy(ReActStrategy(1)), Evaluator: SchemaRef("e")}),
		RalphStrategy(RalphConfig{Inner: PtrStrategy(ReActStrategy(1)), Agent: AgentRef("r")}),
		HillClimbingStrategy(HillClimbingConfig{
			Inner:               PtrStrategy(ReActStrategy(1)),
			Direction:           OptimizationMinimize,
			MaxStagnation:       1,
			MinImprovementDelta: 0.0,
			Evaluator:           AgentRef("m"),
		}),
	}
	for i, s := range strategies {
		var cx ExecutionContext
		got := s.Run(&cx)
		if got.Kind != StrategyOutcomePending {
			t.Fatalf("case %d: stub Run returned %v, want pending", i, got.Kind)
		}
	}
}

// ---------------------------------------------------------------------------
// Fixture replay: byte-identical re-marshal + deep-equal parse
// ---------------------------------------------------------------------------

func TestFixtureCordycepsTree(t *testing.T) {
	raw, err := os.ReadFile(strategyFixturePath(t, "cordyceps_tree.json"))
	if err != nil {
		t.Fatal(err)
	}
	var parsed LoopStrategy
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(parsed, cordycepsTree()) {
		t.Fatalf("parsed tree != expected:\n got  %+v\n want %+v", parsed, cordycepsTree())
	}
	got, err := json.Marshal(parsed)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, compactJSON(t, raw)) {
		t.Fatalf("re-marshal not byte-identical:\n got  %s\n want %s", got, compactJSON(t, raw))
	}
}

func TestFixtureStrategyRef(t *testing.T) {
	raw, err := os.ReadFile(strategyFixturePath(t, "strategy_ref.json"))
	if err != nil {
		t.Fatal(err)
	}
	var suite struct {
		BuiltIn StrategyRef `json:"built_in"`
		Custom  StrategyRef `json:"custom"`
	}
	if err := json.Unmarshal(raw, &suite); err != nil {
		t.Fatal(err)
	}
	if suite.BuiltIn.Kind != StrategyRefBuiltIn || !reflect.DeepEqual(*suite.BuiltIn.BuiltIn, cordycepsTree()) {
		t.Fatalf("built_in parse mismatch: %+v", suite.BuiltIn)
	}
	if suite.Custom.Kind != StrategyRefCustom || suite.Custom.Custom != "my-harness::DoubleVerify" {
		t.Fatalf("custom parse mismatch: %+v", suite.Custom)
	}
	got, err := json.Marshal(suite)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, compactJSON(t, raw)) {
		t.Fatalf("strategy_ref re-marshal not byte-identical:\n got  %s\n want %s", got, compactJSON(t, raw))
	}
}

func TestFixturePausedState(t *testing.T) {
	raw, err := os.ReadFile(strategyFixturePath(t, "paused_state.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ps PausedState
	if err := json.Unmarshal(raw, &ps); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ps.Task.LoopStrategy, cordycepsTree()) {
		t.Fatalf("paused_state task.loop_strategy mismatch: %+v", ps.Task.LoopStrategy)
	}
	got, err := json.Marshal(ps)
	if err != nil {
		t.Fatal(err)
	}
	// Semantic JSON equality: the only divergence from the literal fixture bytes
	// is the cross-language float representation of cost_usd (Go emits 0, the
	// fixture writes 0.0) — orthogonal to #119. jsonEqual normalizes both.
	if !jsonEqual(t, got, raw) {
		t.Fatalf("paused_state re-marshal mismatch:\n got  %s\n want %s", got, compactJSON(t, raw))
	}
}

func TestFixtureChildPausedState(t *testing.T) {
	raw, err := os.ReadFile(strategyFixturePath(t, "child_paused_state.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cs ChildPausedState
	if err := json.Unmarshal(raw, &cs); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(cs.Task.LoopStrategy, cordycepsTree()) {
		t.Fatalf("child_paused_state task.loop_strategy mismatch: %+v", cs.Task.LoopStrategy)
	}
	got, err := json.Marshal(cs)
	if err != nil {
		t.Fatal(err)
	}
	// Semantic JSON equality (see TestFixturePausedState): cost_usd float repr
	// (0 vs 0.0) is the only orthogonal divergence.
	if !jsonEqual(t, got, raw) {
		t.Fatalf("child_paused_state re-marshal mismatch:\n got  %s\n want %s", got, compactJSON(t, raw))
	}
}
