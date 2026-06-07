package sporecore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// ============================================================================
// Test-only stubs
// ============================================================================

type stubAgent struct{ id AgentID }

func (a stubAgent) Turn(_ context.Context, _ Context) TurnResult {
	panic("Validate() must fail before any agent turn")
}
func (a stubAgent) ID() AgentID { return a.id }

type regStubVerifier struct{}

func (regStubVerifier) Verify(_ context.Context, _ SelfVerifyInput) SelfVerifyVerdict {
	panic("verifier not invoked in registry tests")
}
func (regStubVerifier) MaxIterations() uint32 { return 3 }

type stubStrategy struct{}

func (stubStrategy) Run(_ *ExecutionContext) StrategyOutcome { return pendingOutcome() }

func reactLeaf(agent, toolset string) LoopStrategy {
	return LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget:  BudgetPolicy{Kind: BudgetPerLoop, Value: 4},
		Agent:   AgentRef(agent),
		Toolset: ToolsetRef(toolset),
	}}
}

func fullyWiredRegistry() ExecutionRegistry {
	return NewExecutionRegistryBuilder().
		Agent("a1", stubAgent{id: "a1"}).
		Toolset("t1", NewScriptedToolRegistry()).
		Schema("s1", json.RawMessage(`{"type":"object"}`)).
		Verifier("v1", regStubVerifier{}).
		Build()
}

// ============================================================================
// Resolve_* happy path + miss
// ============================================================================

func TestResolveEachHappyAndMiss(t *testing.T) {
	reg := fullyWiredRegistry()

	if _, ok := reg.ResolveAgent(AgentRef("a1")); !ok {
		t.Fatal("ResolveAgent(a1) should hit")
	}
	if _, ok := reg.ResolveAgent(AgentRef("nope")); ok {
		t.Fatal("ResolveAgent(nope) should miss")
	}

	if _, ok := reg.ResolveToolset(ToolsetRef("t1")); !ok {
		t.Fatal("ResolveToolset(t1) should hit")
	}
	if _, ok := reg.ResolveToolset(ToolsetRef("nope")); ok {
		t.Fatal("ResolveToolset(nope) should miss")
	}

	if _, ok := reg.ResolveSchema(SchemaRef("s1")); !ok {
		t.Fatal("ResolveSchema(s1) should hit")
	}
	if _, ok := reg.ResolveSchema(SchemaRef("nope")); ok {
		t.Fatal("ResolveSchema(nope) should miss")
	}

	if _, ok := reg.ResolveVerifier("v1"); !ok {
		t.Fatal("ResolveVerifier(v1) should hit")
	}
	if _, ok := reg.ResolveVerifier("nope"); ok {
		t.Fatal("ResolveVerifier(nope) should miss")
	}
}

// ============================================================================
// RegisterStrategy + ResolveStrategy(Custom)
// ============================================================================

func TestRegisterThenResolveCustomStrategy(t *testing.T) {
	reg := NewExecutionRegistry()
	reg.RegisterStrategy("mine::Custom", stubStrategy{})

	res, err := reg.ResolveStrategy(StrategyRef{Kind: StrategyRefCustom, Custom: "mine::Custom"})
	if err != nil {
		t.Fatalf("ResolveStrategy should succeed, got %v", err)
	}
	if res.Kind != ResolutionCustom || res.Custom == nil {
		t.Fatalf("expected Custom resolution, got %+v", res)
	}
}

func TestResolveBuiltInStrategy(t *testing.T) {
	reg := NewExecutionRegistry()
	leaf := reactLeaf("a1", "t1")
	res, err := reg.ResolveStrategy(StrategyRef{Kind: StrategyRefBuiltIn, BuiltIn: &leaf})
	if err != nil {
		t.Fatalf("ResolveStrategy(BuiltIn) should succeed, got %v", err)
	}
	if res.Kind != ResolutionBuiltIn || res.BuiltIn == nil || res.BuiltIn.Kind != StrategyReAct {
		t.Fatalf("expected BuiltIn react, got %+v", res)
	}
}

// ============================================================================
// Missing custom key → recoverable StrategyNotFound, no panic
// ============================================================================

func TestMissingCustomKeyIsRecoverableStrategyNotFound(t *testing.T) {
	reg := NewExecutionRegistry()
	_, err := reg.ResolveStrategy(StrategyRef{Kind: StrategyRefCustom, Custom: "absent"})
	if err == nil {
		t.Fatal("expected an error for a missing custom key")
	}
	var snf *StrategyNotFoundError
	if !errors.As(err, &snf) {
		t.Fatalf("expected *StrategyNotFoundError, got %T", err)
	}
	if snf.Key != "absent" {
		t.Fatalf("expected key=absent, got %q", snf.Key)
	}
	// Reaching here proves it was returned, never panicked.
}

// ============================================================================
// Validate() unresolved handle → UnresolvedHandleError
// ============================================================================

func assertUnresolved(t *testing.T, err error, kind, key string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected unresolved %s handle %q, got nil", kind, key)
	}
	var uh *UnresolvedHandleError
	if !errors.As(err, &uh) {
		t.Fatalf("expected *UnresolvedHandleError, got %T (%v)", err, err)
	}
	if uh.Kind != kind || uh.Key != key {
		t.Fatalf("expected {%s,%s}, got {%s,%s}", kind, key, uh.Kind, uh.Key)
	}
}

func TestValidateUnresolvedAgentHandle(t *testing.T) {
	reg := NewExecutionRegistry()
	task := NewTask("do it", NewSessionID(), reactLeaf("missing-agent", "t1"))
	assertUnresolved(t, reg.Validate(task), "agent", "missing-agent")
}

func TestValidateUnresolvedToolsetHandle(t *testing.T) {
	reg := NewExecutionRegistryBuilder().Agent("a1", stubAgent{id: "a1"}).Build()
	task := NewTask("do it", NewSessionID(), reactLeaf("a1", "missing-tools"))
	assertUnresolved(t, reg.Validate(task), "toolset", "missing-tools")
}

func TestValidateUnresolvedSchemaHandle(t *testing.T) {
	reg := NewExecutionRegistryBuilder().
		Agent("a1", stubAgent{id: "a1"}).
		Toolset("t1", NewScriptedToolRegistry()).
		Build()
	out := SchemaRef("missing-schema")
	leaf := LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget:  BudgetPolicy{Kind: BudgetPerLoop, Value: 4},
		Agent:   AgentRef("a1"),
		Toolset: ToolsetRef("t1"),
		Output:  &out,
	}}
	task := NewTask("do it", NewSessionID(), leaf)
	assertUnresolved(t, reg.Validate(task), "schema", "missing-schema")
}

func TestValidateHappyPathReact(t *testing.T) {
	reg := fullyWiredRegistry()
	out := SchemaRef("s1")
	leaf := LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget:  BudgetPolicy{Kind: BudgetPerLoop, Value: 4},
		Agent:   AgentRef("a1"),
		Toolset: ToolsetRef("t1"),
		Output:  &out,
	}}
	task := NewTask("ok", NewSessionID(), leaf)
	if err := reg.Validate(task); err != nil {
		t.Fatalf("expected validate ok, got %v", err)
	}
}

// ============================================================================
// Tree-walk over the nested cordyceps fixture tree
// ============================================================================

func regCordycepsTree(t *testing.T) LoopStrategy {
	t.Helper()
	_, this, _, _ := runtime.Caller(0)
	path := filepath.Join(filepath.Dir(this), "..", "..", "fixtures", "strategy", "cordyceps_tree.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read cordyceps_tree.json: %v", err)
	}
	var ls LoopStrategy
	if err := json.Unmarshal(raw, &ls); err != nil {
		t.Fatalf("parse cordyceps_tree.json: %v", err)
	}
	return ls
}

func TestValidateTreeWalkReportsFirstUnresolvedInNestedTree(t *testing.T) {
	// The cordyceps tree references agents planner/executor/ralph-agent,
	// toolsets plan-tools/exec-tools, schema exec-evaluator. An empty registry
	// must report the FIRST unresolved handle (depth-first: ralph inner →
	// plan_execute → plan react → agent "planner").
	reg := NewExecutionRegistry()
	task := NewTask("nested", NewSessionID(), regCordycepsTree(t))
	assertUnresolved(t, reg.Validate(task), "agent", "planner")
}

func TestValidateTreeWalkPassesWhenFullyWired(t *testing.T) {
	reg := NewExecutionRegistryBuilder().
		Agent("planner", stubAgent{id: "planner"}).
		Agent("executor", stubAgent{id: "executor"}).
		Agent("ralph-agent", stubAgent{id: "ralph-agent"}).
		Toolset("plan-tools", NewScriptedToolRegistry()).
		Toolset("exec-tools", NewScriptedToolRegistry()).
		Schema("exec-evaluator", json.RawMessage(`{}`)).
		Build()
	task := NewTask("nested", NewSessionID(), regCordycepsTree(t))
	if err := reg.Validate(task); err != nil {
		t.Fatalf("expected fully-wired validate ok, got %v", err)
	}
}

// ============================================================================
// Resume: round-trip a Task through JSON, re-resolve all
// ============================================================================

func TestResumeReResolvesAllHandlesAfterJSONRoundTrip(t *testing.T) {
	out := SchemaRef("s1")
	leaf := LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget:  BudgetPolicy{Kind: BudgetPerLoop, Value: 4},
		Agent:   AgentRef("a1"),
		Toolset: ToolsetRef("t1"),
		Output:  &out,
	}}
	task := NewTask("resume me", NewSessionID(), leaf)

	wire, err := json.Marshal(task)
	if err != nil {
		t.Fatalf("Task serializes: %v", err)
	}
	var restored Task
	if err := json.Unmarshal(wire, &restored); err != nil {
		t.Fatalf("Task deserializes: %v", err)
	}

	// Fresh registry built independently (as on resume) re-resolves all, no
	// reconfiguration of the Task required.
	reg := fullyWiredRegistry()
	if err := reg.Validate(restored); err != nil {
		t.Fatalf("re-resolve after round-trip should succeed, got %v", err)
	}
	c := restored.LoopStrategy.ReActCfg
	if c == nil {
		t.Fatal("expected ReAct leaf after round-trip")
	}
	if _, ok := reg.ResolveAgent(c.Agent); !ok {
		t.Fatal("agent should re-resolve")
	}
	if _, ok := reg.ResolveToolset(c.Toolset); !ok {
		t.Fatal("toolset should re-resolve")
	}
	if _, ok := reg.ResolveSchema(*c.Output); !ok {
		t.Fatal("schema should re-resolve")
	}
}

// ============================================================================
// Startup error: unresolved handle → HaltConfigurationError before first turn
// ============================================================================

func TestUnresolvedHandleIsStartupErrorBeforeFirstTurn(t *testing.T) {
	// stubAgent.Turn panics if reached; the run must fail at startup validation
	// before any turn fires.
	reg := NewExecutionRegistryBuilder().Agent("a1", stubAgent{id: "a1"}).Build()
	cfg := HarnessConfig{
		Agent:             stubAgent{id: "default"},
		ToolRegistry:      NewScriptedToolRegistry(),
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
		Registry:          reg,
	}
	h := NewStandardHarness(cfg)
	task := NewTask("startup", NewSessionID(), reactLeaf("a1", "missing-tools"))

	result := h.Run(context.Background(), HarnessRunOptions{Task: task})
	if result.Kind != RunFailure {
		t.Fatalf("expected RunFailure at startup, got %v", result.Kind)
	}
	if result.Reason.Kind != HaltConfigurationError {
		t.Fatalf("expected HaltConfigurationError, got %v", result.Reason.Kind)
	}
	assertUnresolved(t, result.Reason.ConfigError, "toolset", "missing-tools")
}

func TestEmptyRegistrySkipsStartupValidation(t *testing.T) {
	// Legacy callers with no registry must NOT trigger validation (Option B
	// byte-identity). An empty registry + a handle-bearing task must not produce
	// a HaltConfigurationError; the run proceeds on the legacy path.
	a := NewMockAgent("legacy")
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := HarnessConfig{
		Agent:             a,
		ToolRegistry:      NewScriptedToolRegistry(),
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
	}
	h := NewStandardHarness(cfg)
	task := NewTask("legacy", NewSessionID(), reactLeaf("unresolved", "unresolved"))
	result := h.Run(context.Background(), HarnessRunOptions{Task: task})
	if result.Reason.Kind == HaltConfigurationError {
		t.Fatal("empty registry must skip startup validation")
	}
}

// ============================================================================
// EscalationMode present / selectable / readable on config
// ============================================================================

func TestEscalationModePresentSelectableReadable(t *testing.T) {
	// Default (zero value) reads as surface_to_human.
	cfg := HarnessConfig{}
	if got := cfg.EffectiveEscalationMode(); got.Kind != EscalationSurfaceToHuman {
		t.Fatalf("default escalation should be surface_to_human, got %q", got.Kind)
	}

	// Selectable + readable.
	cfg = cfg.WithEscalationMode(AutonomousEscalation())
	if cfg.EscalationMode.Kind != EscalationAutonomous {
		t.Fatalf("expected autonomous, got %q", cfg.EscalationMode.Kind)
	}
	if got := cfg.EffectiveEscalationMode(); got.Kind != EscalationAutonomous {
		t.Fatalf("effective should pass through explicit autonomous, got %q", got.Kind)
	}
}

func TestEscalationModeJSONRoundTrip(t *testing.T) {
	for _, m := range []EscalationMode{SurfaceToHumanEscalation(), AutonomousEscalation()} {
		b, err := json.Marshal(m)
		if err != nil {
			t.Fatalf("marshal %q: %v", m.Kind, err)
		}
		var back EscalationMode
		if err := json.Unmarshal(b, &back); err != nil {
			t.Fatalf("unmarshal %s: %v", b, err)
		}
		if back.Kind != m.Kind {
			t.Fatalf("round-trip mismatch: %q != %q", back.Kind, m.Kind)
		}
	}
	// Tagged snake_case shape.
	b, _ := json.Marshal(SurfaceToHumanEscalation())
	if string(b) != `{"kind":"surface_to_human"}` {
		t.Fatalf("unexpected wire shape: %s", b)
	}
}

// ============================================================================
// HarnessConfig registry convenience setters do not alias source maps
// ============================================================================

func TestRegistryConvenienceSettersAndChaining(t *testing.T) {
	cfg := HarnessConfig{}.
		WithRegistryAgent("a1", stubAgent{id: "a1"}).
		WithRegistryToolset("t1", NewScriptedToolRegistry()).
		WithRegistrySchema("s1", json.RawMessage(`{}`)).
		WithRegistryVerifier("v1", regStubVerifier{}).
		RegisterStrategy("c1", stubStrategy{})

	if _, ok := cfg.Registry.ResolveAgent(AgentRef("a1")); !ok {
		t.Fatal("agent should be registered via convenience setter")
	}
	if _, ok := cfg.Registry.ResolveToolset(ToolsetRef("t1")); !ok {
		t.Fatal("toolset should be registered")
	}
	if _, ok := cfg.Registry.ResolveSchema(SchemaRef("s1")); !ok {
		t.Fatal("schema should be registered")
	}
	if _, ok := cfg.Registry.ResolveVerifier("v1"); !ok {
		t.Fatal("verifier should be registered")
	}
	if _, err := cfg.Registry.ResolveStrategy(StrategyRef{Kind: StrategyRefCustom, Custom: "c1"}); err != nil {
		t.Fatalf("custom strategy should resolve, got %v", err)
	}
}

func TestBuilderLastWinsOnDuplicateKey(t *testing.T) {
	reg := NewExecutionRegistryBuilder().
		Schema("s", json.RawMessage(`{"v":1}`)).
		Schema("s", json.RawMessage(`{"v":2}`)).
		Build()
	v, ok := reg.ResolveSchema(SchemaRef("s"))
	if !ok || string(v) != `{"v":2}` {
		t.Fatalf("expected last-wins {\"v\":2}, got %s (ok=%v)", v, ok)
	}
}

// ============================================================================
// Fixture replay: registry_errors.json round-trips byte-identically
// ============================================================================

func TestRegistryErrorsFixtureRoundTripsByteIdentical(t *testing.T) {
	_, this, _, _ := runtime.Caller(0)
	path := filepath.Join(filepath.Dir(this), "..", "..", "fixtures", "harness", "registry_errors.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read registry_errors.json: %v", err)
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}

	// StrategyNotFound
	snf, err := UnmarshalHarnessError(doc["strategy_not_found"])
	if err != nil {
		t.Fatalf("decode strategy_not_found: %v", err)
	}
	if got, ok := snf.(*StrategyNotFoundError); !ok || got.Key != "my-harness::DoubleVerify" {
		t.Fatalf("unexpected StrategyNotFound: %#v", snf)
	}
	assertByteIdentical(t, snf, doc["strategy_not_found"])

	// UnresolvedHandle (handle_kind on the wire)
	uh, err := UnmarshalHarnessError(doc["unresolved_handle"])
	if err != nil {
		t.Fatalf("decode unresolved_handle: %v", err)
	}
	if got, ok := uh.(*UnresolvedHandleError); !ok || got.Kind != "agent" || got.Key != "planner" {
		t.Fatalf("unexpected UnresolvedHandle: %#v", uh)
	}
	assertByteIdentical(t, uh, doc["unresolved_handle"])
}

// assertByteIdentical re-marshals he and compares it (key-order-insensitively
// via canonicalization) to the fixture's raw bytes.
func assertByteIdentical(t *testing.T, he HarnessError, want json.RawMessage) {
	t.Helper()
	got, err := json.Marshal(he)
	if err != nil {
		t.Fatalf("marshal %T: %v", he, err)
	}
	if canonical(t, got) != canonical(t, want) {
		t.Fatalf("round-trip not byte-identical:\n got=%s\nwant=%s", got, want)
	}
}

func canonical(t *testing.T, b []byte) string {
	t.Helper()
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatalf("canonicalize %s: %v", b, err)
	}
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("re-marshal: %v", err)
	}
	return string(out)
}
