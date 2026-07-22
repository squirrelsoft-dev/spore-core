package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

type namedTestTool struct{ name string }

func (t namedTestTool) Name() string                { return t.name }
func (t namedTestTool) IsSubagentTool() bool        { return false }
func (t namedTestTool) MayProduceLargeOutput() bool { return false }
func (t namedTestTool) Execute(context.Context, ToolCall, SandboxProvider, *ToolContext) ToolOutput {
	return ToolOutput{Kind: ToolOutputSuccess, Content: "ran " + t.name}
}

func registerNamedTool(t *testing.T, reg *StandardToolRegistry, name, desc string) {
	t.Helper()
	if err := reg.Register(namedTestTool{name: name}, RegistryToolSchema{
		Name:        name,
		Description: desc,
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}); err != nil {
		t.Fatal(err)
	}
}

func TestToolSearchRegistryAdvertisesCorePlusMeta(t *testing.T) {
	inner := NewStandardToolRegistry()
	registerNamedTool(t, inner, "read_file", "read a file")
	registerNamedTool(t, inner, "write_file", "write a file")
	reg := NewToolSearchRegistry(inner, ToolSearchConfig{CoreTools: []string{"read_file"}})
	schemas := reg.ActiveSchemas(nil)
	var names []string
	for _, s := range schemas {
		names = append(names, s.Name)
	}
	want := []string{"read_file", ToolSearchToolName}
	if len(names) != len(want) || names[0] != want[0] || names[1] != want[1] {
		t.Fatalf("advertised names=%v, want %v", names, want)
	}
}

func TestToolSearchRegistrySearchesDeferredAndDelegatesDispatch(t *testing.T) {
	inner := NewStandardToolRegistry()
	registerNamedTool(t, inner, "read_file", "read a file")
	registerNamedTool(t, inner, "grep", "search text in files")
	reg := NewToolSearchRegistry(inner, ToolSearchConfig{CoreTools: []string{"read_file"}, DescriptionBudget: 20})
	res, err := reg.Dispatch(context.Background(), ToolCall{ID: "s1", Name: ToolSearchToolName, Input: json.RawMessage(`{"query":"search"}`)}, AllowAllSandbox{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Output.Kind != ToolOutputSuccess {
		t.Fatalf("search output=%+v", res.Output)
	}
	var matches []RegistryToolSchema
	if err := json.Unmarshal([]byte(res.Output.Content), &matches); err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 || matches[0].Name != "grep" {
		t.Fatalf("matches=%+v, want grep only", matches)
	}
	deferred, err := reg.Dispatch(context.Background(), ToolCall{ID: "g1", Name: "grep", Input: json.RawMessage(`{}`)}, AllowAllSandbox{})
	if err != nil {
		t.Fatal(err)
	}
	if deferred.Output.Content != "ran grep" {
		t.Fatalf("delegated output=%+v", deferred.Output)
	}
}
