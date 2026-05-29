package sporecore

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"testing"
)

// ----------------------------------------------------------------------------
// Test helpers — mock tools and sandboxes.
// ----------------------------------------------------------------------------

type echoTool struct {
	name  string
	calls atomic.Int64
}

func (e *echoTool) Name() string                { return e.name }
func (e *echoTool) IsSubagentTool() bool        { return false }
func (e *echoTool) MayProduceLargeOutput() bool { return false }
func (e *echoTool) Execute(_ context.Context, call ToolCall, _ SandboxProvider, _ *ToolContext) ToolOutput {
	e.calls.Add(1)
	return ToolOutput{
		Kind:    ToolOutputSuccess,
		Content: string(call.Input),
	}
}

type subagentTool struct{ name string }

func (s *subagentTool) Name() string                { return s.name }
func (s *subagentTool) IsSubagentTool() bool        { return true }
func (s *subagentTool) MayProduceLargeOutput() bool { return false }
func (s *subagentTool) Execute(_ context.Context, _ ToolCall, _ SandboxProvider, _ *ToolContext) ToolOutput {
	return ToolOutput{Kind: ToolOutputSuccess, Content: "subagent done"}
}

type allowAllSandbox struct{ DefaultSandbox }

func (allowAllSandbox) Validate(context.Context, ToolCall) *SandboxViolation { return nil }

type denyAllSandbox struct{ DefaultSandbox }

func (denyAllSandbox) Validate(context.Context, ToolCall) *SandboxViolation {
	return &SandboxViolation{Kind: SandboxPathEscape, Path: "denied"}
}

func makeSchema(name string, annotations ToolAnnotations) RegistryToolSchema {
	return RegistryToolSchema{
		Name:        name,
		Description: name + " tool",
		Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
		Annotations: annotations,
	}
}

func makeRequiredSchema(name string, required []string) RegistryToolSchema {
	r, _ := json.Marshal(required)
	params := json.RawMessage(`{"type":"object","properties":{},"required":` + string(r) + `}`)
	return RegistryToolSchema{
		Name:        name,
		Description: name,
		Parameters:  params,
		Annotations: ToolAnnotations{ReadOnly: true},
	}
}

func makeCall(name, id string, input string) ToolCall {
	return ToolCall{ID: id, Name: name, Input: json.RawMessage(input)}
}

// ----------------------------------------------------------------------------
// Tests — one per rule.
// ----------------------------------------------------------------------------

// Rule 1: tools dispatched via the registry.
func TestRegistryDispatchesRegisteredTool(t *testing.T) {
	reg := NewStandardToolRegistry()
	if err := reg.Register(&echoTool{name: "echo"}, makeSchema("echo", ToolAnnotations{ReadOnly: true})); err != nil {
		t.Fatalf("register: %v", err)
	}
	res, err := reg.Dispatch(context.Background(), makeCall("echo", "c1", `{"x":1}`), allowAllSandbox{})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if res.CallID != "c1" {
		t.Fatalf("call_id: got %q", res.CallID)
	}
	if res.Output.Kind != ToolOutputSuccess || res.Output.Content != `{"x":1}` {
		t.Fatalf("output: %+v", res.Output)
	}
}

// Rule 3 (issue #81, Q1): duplicate names UPSERT — the last registration wins,
// overwriting the prior tool/schema. This is what lets an architect override a
// standard tool by registering their own under the same name after a preset.
func TestRegistryDuplicateNameUpserts(t *testing.T) {
	reg := NewStandardToolRegistry()
	if err := reg.Register(&echoTool{name: "echo"}, makeSchema("echo", ToolAnnotations{})); err != nil {
		t.Fatal(err)
	}
	// Re-registering the same name must NOT error.
	override := &constTool{name: "echo", content: "override-wins"}
	if err := reg.Register(override, makeSchema("echo", ToolAnnotations{})); err != nil {
		t.Fatalf("upsert must not error, got %v", err)
	}
	// The later registration is the one that dispatches.
	res, err := reg.Dispatch(context.Background(), ToolCall{ID: "1", Name: "echo", Input: json.RawMessage(`{}`)}, allowAllSandbox{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Output.Content != "override-wins" {
		t.Fatalf("last-wins upsert: expected override tool to dispatch, got %q", res.Output.Content)
	}
}

// constTool returns a fixed content string, to prove which registration is live.
type constTool struct {
	name    string
	content string
}

func (c *constTool) Name() string                { return c.name }
func (c *constTool) IsSubagentTool() bool        { return false }
func (c *constTool) MayProduceLargeOutput() bool { return false }
func (c *constTool) Execute(_ context.Context, _ ToolCall, _ SandboxProvider, _ *ToolContext) ToolOutput {
	return ToolOutput{Kind: ToolOutputSuccess, Content: c.content}
}

// Rule 2: invalid schema (no top-level "type") rejected.
func TestRegistryInvalidSchemaRejected(t *testing.T) {
	reg := NewStandardToolRegistry()
	bad := RegistryToolSchema{
		Name:        "x",
		Description: "x",
		Parameters:  json.RawMessage(`{"properties":{}}`),
	}
	err := reg.Register(&echoTool{name: "x"}, bad)
	re, ok := err.(*RegistrationError)
	if !ok || re.Kind != RegErrInvalidSchema {
		t.Fatalf("expected InvalidSchema, got %v", err)
	}
}

// Rule 2: empty name rejected.
func TestRegistryEmptyNameRejected(t *testing.T) {
	reg := NewStandardToolRegistry()
	bad := RegistryToolSchema{
		Name:       "",
		Parameters: json.RawMessage(`{"type":"object"}`),
	}
	err := reg.Register(&echoTool{name: ""}, bad)
	re, ok := err.(*RegistrationError)
	if !ok || re.Kind != RegErrInvalidSchema {
		t.Fatalf("expected InvalidSchema, got %v", err)
	}
}

// Rule 4: read_only + destructive rejected.
func TestRegistryConflictingAnnotationsRejected(t *testing.T) {
	reg := NewStandardToolRegistry()
	err := reg.Register(&echoTool{name: "rm"}, makeSchema("rm", ToolAnnotations{ReadOnly: true, Destructive: true}))
	re, ok := err.(*RegistrationError)
	if !ok || re.Kind != RegErrConflictingAnnotations {
		t.Fatalf("expected ConflictingAnnotations, got %v", err)
	}
}

// Rule 5: tool.Name() must match schema.Name.
func TestRegistryToolSchemaNameMismatch(t *testing.T) {
	reg := NewStandardToolRegistry()
	err := reg.Register(&echoTool{name: "a"}, makeSchema("b", ToolAnnotations{}))
	re, ok := err.(*RegistrationError)
	if !ok || re.Kind != RegErrInvalidSchema {
		t.Fatalf("expected InvalidSchema, got %v", err)
	}
}

// Rule 6: unregistered tool call errors.
func TestRegistryUnregisteredToolErrors(t *testing.T) {
	reg := NewStandardToolRegistry()
	_, err := reg.Dispatch(context.Background(), makeCall("missing", "c1", `{}`), allowAllSandbox{})
	de, ok := err.(*DispatchError)
	if !ok || de.Kind != DispatchErrUnregisteredTool {
		t.Fatalf("expected UnregisteredTool, got %v", err)
	}
}

// Rule 7: missing required field surfaces SchemaValidationFailed.
func TestRegistryMissingRequiredField(t *testing.T) {
	reg := NewStandardToolRegistry()
	if err := reg.Register(&echoTool{name: "read"}, makeRequiredSchema("read", []string{"path"})); err != nil {
		t.Fatal(err)
	}
	_, err := reg.Dispatch(context.Background(), makeCall("read", "c1", `{}`), allowAllSandbox{})
	de, ok := err.(*DispatchError)
	if !ok || de.Kind != DispatchErrSchemaValidationFailed {
		t.Fatalf("expected SchemaValidationFailed, got %v", err)
	}
	if de.Tool != "read" {
		t.Fatalf("tool: %q", de.Tool)
	}
}

// Rule 7 (positive): required field present succeeds.
func TestRegistryRequiredFieldPresent(t *testing.T) {
	reg := NewStandardToolRegistry()
	if err := reg.Register(&echoTool{name: "read"}, makeRequiredSchema("read", []string{"path"})); err != nil {
		t.Fatal(err)
	}
	if _, err := reg.Dispatch(context.Background(), makeCall("read", "c1", `{"path":"/x"}`), allowAllSandbox{}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
}

// Rule 8: sandbox violation surfaces as DispatchError.
func TestRegistrySandboxViolationSurfaces(t *testing.T) {
	reg := NewStandardToolRegistry()
	if err := reg.Register(&echoTool{name: "echo"}, makeSchema("echo", ToolAnnotations{ReadOnly: true})); err != nil {
		t.Fatal(err)
	}
	_, err := reg.Dispatch(context.Background(), makeCall("echo", "c1", `{}`), denyAllSandbox{})
	de, ok := err.(*DispatchError)
	if !ok || de.Kind != DispatchErrSandboxViolation {
		t.Fatalf("expected SandboxViolation, got %v", err)
	}
	if de.Violation == nil || de.Violation.Kind != SandboxPathEscape {
		t.Fatalf("violation: %+v", de.Violation)
	}
}

// Rule 9: ActiveSchemas filters by phase and sorts by name.
func TestActiveSchemasFilteredByPhaseAndSorted(t *testing.T) {
	reg := NewStandardToolRegistry()
	for _, n := range []string{"zeta", "alpha", "beta"} {
		if err := reg.Register(&echoTool{name: n}, makeSchema(n, ToolAnnotations{})); err != nil {
			t.Fatal(err)
		}
	}
	plan := PhasePlanning
	if err := reg.RegisterSet(ToolSet{Name: "plan", Tools: []string{"alpha", "zeta"}, Phase: &plan}); err != nil {
		t.Fatal(err)
	}
	if err := reg.RegisterSet(ToolSet{Name: "always", Tools: []string{"beta"}}); err != nil {
		t.Fatal(err)
	}

	got := reg.ActiveSchemas(&plan)
	names := schemaNames(got)
	want := []string{"alpha", "beta", "zeta"}
	if !orderedEqual(names, want) {
		t.Fatalf("planning: got %v, want %v (sorted)", names, want)
	}

	exec := PhaseExecution
	got = reg.ActiveSchemas(&exec)
	if names := schemaNames(got); !orderedEqual(names, []string{"beta"}) {
		t.Fatalf("execution: got %v, want [beta]", names)
	}
}

// Rule 9 (fallback): no sets registered → ActiveSchemas returns all.
func TestActiveSchemasNoSetsReturnsAll(t *testing.T) {
	reg := NewStandardToolRegistry()
	for _, n := range []string{"a", "b"} {
		if err := reg.Register(&echoTool{name: n}, makeSchema(n, ToolAnnotations{})); err != nil {
			t.Fatal(err)
		}
	}
	got := reg.ActiveSchemas(nil)
	if names := schemaNames(got); !orderedEqual(names, []string{"a", "b"}) {
		t.Fatalf("nil phase: got %v", names)
	}
	exec := PhaseExecution
	got = reg.ActiveSchemas(&exec)
	if names := schemaNames(got); !orderedEqual(names, []string{"a", "b"}) {
		t.Fatalf("phase fallback: got %v", names)
	}
}

// Rule 10: DispatchAll preserves input order across mixed annotations.
func TestDispatchAllPreservesInputOrder(t *testing.T) {
	reg := NewStandardToolRegistry()
	if err := reg.Register(&echoTool{name: "r"}, makeSchema("r", ToolAnnotations{ReadOnly: true})); err != nil {
		t.Fatal(err)
	}
	if err := reg.Register(&echoTool{name: "d"}, makeSchema("d", ToolAnnotations{Destructive: true})); err != nil {
		t.Fatal(err)
	}
	calls := []ToolCall{
		makeCall("d", "1", `{"v":"a"}`),
		makeCall("r", "2", `{"v":"b"}`),
		makeCall("d", "3", `{"v":"c"}`),
		makeCall("r", "4", `{"v":"d"}`),
	}
	out := reg.DispatchAll(context.Background(), calls, allowAllSandbox{})
	if len(out) != 4 {
		t.Fatalf("len: %d", len(out))
	}
	for i, want := range []string{"1", "2", "3", "4"} {
		if out[i].Err != nil {
			t.Fatalf("slot %d err: %v", i, out[i].Err)
		}
		if out[i].Result.CallID != want {
			t.Fatalf("slot %d call_id: got %q, want %q", i, out[i].Result.CallID, want)
		}
	}
}

// DispatchAll surfaces per-slot errors.
func TestDispatchAllPerSlotErrors(t *testing.T) {
	reg := NewStandardToolRegistry()
	if err := reg.Register(&echoTool{name: "ok"}, makeSchema("ok", ToolAnnotations{ReadOnly: true})); err != nil {
		t.Fatal(err)
	}
	calls := []ToolCall{
		makeCall("ok", "1", `{}`),
		makeCall("missing", "2", `{}`),
	}
	out := reg.DispatchAll(context.Background(), calls, allowAllSandbox{})
	if out[0].Err != nil {
		t.Fatalf("slot 0: %v", out[0].Err)
	}
	if out[1].Err == nil || out[1].Err.Kind != DispatchErrUnregisteredTool {
		t.Fatalf("slot 1 err: %+v", out[1].Err)
	}
}

// Rule 11: HasSubagentTools tracks registration.
func TestHasSubagentToolsReflectsRegistration(t *testing.T) {
	reg := NewStandardToolRegistry()
	if reg.HasSubagentTools() {
		t.Fatal("empty registry should not report subagent tools")
	}
	if err := reg.Register(&echoTool{name: "echo"}, makeSchema("echo", ToolAnnotations{})); err != nil {
		t.Fatal(err)
	}
	if reg.HasSubagentTools() {
		t.Fatal("non-subagent registration must not flip HasSubagentTools")
	}
	if err := reg.Register(&subagentTool{name: "sub"}, makeSchema("sub", ToolAnnotations{})); err != nil {
		t.Fatal(err)
	}
	if !reg.HasSubagentTools() {
		t.Fatal("subagent registration should flip HasSubagentTools")
	}
}

// ToolSchema projection drops annotations.
func TestRegistryToolSchemaProjection(t *testing.T) {
	s := makeSchema("x", ToolAnnotations{ReadOnly: true})
	m := s.ToModelSchema()
	if m.Name != "x" || m.Description != "x tool" {
		t.Fatalf("projection: %+v", m)
	}
}

// JSON round-trips for fixture portability.
func TestRegistryTypesRoundtripJSON(t *testing.T) {
	s := makeSchema("x", ToolAnnotations{ReadOnly: true, Idempotent: true})
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	var back RegistryToolSchema
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back.Name != s.Name || back.Description != s.Description {
		t.Fatalf("roundtrip schema: %+v", back)
	}
	if back.Annotations != s.Annotations {
		t.Fatalf("roundtrip annotations: %+v", back.Annotations)
	}

	plan := PhasePlanning
	set := ToolSet{Name: "p", Tools: []string{"a"}, Phase: &plan}
	b, _ = json.Marshal(set)
	var setBack ToolSet
	if err := json.Unmarshal(b, &setBack); err != nil {
		t.Fatal(err)
	}
	if setBack.Name != "p" || setBack.Phase == nil || *setBack.Phase != PhasePlanning {
		t.Fatalf("roundtrip set: %+v", setBack)
	}

	errs := []RegistrationError{
		{Kind: RegErrInvalidSchema, Tool: "x", Reason: "y"},
		{Kind: RegErrDuplicateName, Tool: "x"},
		{Kind: RegErrConflictingAnnotations, Tool: "x", Reason: "y"},
	}
	for _, e := range errs {
		b, err := json.Marshal(e)
		if err != nil {
			t.Fatal(err)
		}
		var back RegistrationError
		if err := json.Unmarshal(b, &back); err != nil {
			t.Fatal(err)
		}
		if back.Kind != e.Kind || back.Tool != e.Tool {
			t.Fatalf("roundtrip err: %+v", back)
		}
	}
}

// RegisterSet duplicate rejected.
func TestRegisterSetDuplicate(t *testing.T) {
	reg := NewStandardToolRegistry()
	if err := reg.RegisterSet(ToolSet{Name: "a", Tools: []string{}}); err != nil {
		t.Fatal(err)
	}
	err := reg.RegisterSet(ToolSet{Name: "a", Tools: []string{}})
	re, ok := err.(*RegistrationError)
	if !ok || re.Kind != RegErrDuplicateName {
		t.Fatalf("expected DuplicateName, got %v", err)
	}
}

// ----------------------------------------------------------------------------
// Helpers.
// ----------------------------------------------------------------------------

func schemaNames(in []RegistryToolSchema) []string {
	out := make([]string, len(in))
	for i, s := range in {
		out[i] = s.Name
	}
	return out
}

func orderedEqual(a, b []string) bool {
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
