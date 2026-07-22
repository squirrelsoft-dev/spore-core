package sporecore

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
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

func toolSearchSchemaNames(schemas []RegistryToolSchema) []string {
	names := make([]string, len(schemas))
	for i, s := range schemas {
		names[i] = s.Name
	}
	return names
}

func TestToolSearchRegistryAdvertisesCoreNamespaceAndMeta(t *testing.T) {
	inner := NewStandardToolRegistry()
	registerNamedTool(t, inner, "read_file", "read a file")
	registerNamedTool(t, inner, "web_fetch", "fetch web pages")
	reg := NewToolSearchRegistry(inner, ToolSearchConfig{
		Core:       []string{"read_file"},
		Namespaces: []ToolNamespace{NewToolNamespace("web", "Web tools", "web_fetch")},
	})
	names := toolSearchSchemaNames(reg.ActiveSchemas(nil))
	want := []string{"read_file", ToolSearchToolName, "web"}
	if !slices.Equal(names, want) {
		t.Fatalf("advertised names=%v, want %v", names, want)
	}
	for _, s := range reg.ActiveSchemas(nil) {
		if s.Name == "web" && !strings.Contains(s.Description, "call to load") {
			t.Fatalf("namespace header missing open hint: %q", s.Description)
		}
	}
}

func TestToolSearchLoadsDeferredToolIntoActiveSchemas(t *testing.T) {
	inner := NewStandardToolRegistry()
	registerNamedTool(t, inner, "read_file", "read a file")
	registerNamedTool(t, inner, "grep", "search text in files")
	reg := NewToolSearchRegistry(inner, ToolSearchConfig{Core: []string{"read_file"}, SearchTopK: 1})
	res, err := reg.Dispatch(context.Background(), ToolCall{ID: "s1", Name: ToolSearchToolName, Input: json.RawMessage(`{"query":"search"}`)}, AllowAllSandbox{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Output.Kind != ToolOutputSuccess || !strings.Contains(res.Output.Content, "grep") {
		t.Fatalf("search output=%+v, want readable grep listing", res.Output)
	}
	if names := toolSearchSchemaNames(reg.ActiveSchemas(nil)); !slices.Contains(names, "grep") {
		t.Fatalf("loaded tool not advertised after search: %v", names)
	}
	deferred, err := reg.Dispatch(context.Background(), ToolCall{ID: "g1", Name: "grep", Input: json.RawMessage(`{}`)}, AllowAllSandbox{})
	if err != nil {
		t.Fatal(err)
	}
	if deferred.Output.Content != "ran grep" {
		t.Fatalf("delegated output=%+v", deferred.Output)
	}
}

func TestToolSearchNamespaceOpenLoadsMembersWithoutInnerDispatch(t *testing.T) {
	inner := NewStandardToolRegistry()
	registerNamedTool(t, inner, "web_fetch", "fetch web pages")
	registerNamedTool(t, inner, "web_search", "search the web")
	reg := NewToolSearchRegistry(inner, ToolSearchConfig{
		Namespaces: []ToolNamespace{NewToolNamespace("web", "Web tools", "web_fetch", "web_search")},
		SearchTopK: 1,
	})
	res, err := reg.Dispatch(context.Background(), ToolCall{ID: "n1", Name: "web", Input: json.RawMessage(`{}`)}, AllowAllSandbox{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Output.Kind != ToolOutputSuccess || !strings.Contains(res.Output.Content, "Opened namespace") {
		t.Fatalf("namespace output=%+v", res.Output)
	}
	names := toolSearchSchemaNames(reg.ActiveSchemas(nil))
	if !slices.Contains(names, "web_fetch") || slices.Contains(names, "web_search") {
		t.Fatalf("namespace topK loading names=%v, want only web_fetch loaded", names)
	}
}

func TestToolSearchCustomNameAndDescriptionBudgetLRUEvictsOldest(t *testing.T) {
	inner := NewStandardToolRegistry()
	registerNamedTool(t, inner, "alpha", "alpha search capability")
	registerNamedTool(t, inner, "bravo", "bravo search capability")
	reg := NewToolSearchRegistry(inner, ToolSearchConfig{SearchToolName: "find_tool", SearchTopK: 1, DescriptionBudget: 160})
	if names := toolSearchSchemaNames(reg.ActiveSchemas(nil)); !slices.Contains(names, "find_tool") {
		t.Fatalf("custom search tool missing: %v", names)
	}
	_, _ = reg.Dispatch(context.Background(), ToolCall{ID: "a", Name: "find_tool", Input: json.RawMessage(`{"query":"alpha"}`)}, AllowAllSandbox{})
	_, _ = reg.Dispatch(context.Background(), ToolCall{ID: "b", Name: "find_tool", Input: json.RawMessage(`{"query":"bravo"}`)}, AllowAllSandbox{})
	names := toolSearchSchemaNames(reg.ActiveSchemas(nil))
	if slices.Contains(names, "alpha") || !slices.Contains(names, "bravo") {
		t.Fatalf("LRU advertised names=%v, want bravo kept and alpha evicted", names)
	}
}
