package observability

import (
	"context"
	"encoding/json"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// echoTool is a trivial catalogue tool for the builder-fold tests. It is defined
// locally rather than importing the tools package: tools transitively depends on
// this (observability) package, so importing it from an observability test would
// form an import cycle.
type echoTool struct{ name string }

func (t echoTool) Name() string                { return t.name }
func (t echoTool) IsSubagentTool() bool        { return false }
func (t echoTool) MayProduceLargeOutput() bool { return false }
func (t echoTool) Execute(_ context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	return sporecore.NewToolOutputSuccess("echo")
}

func echoStandardTool(name string) sporecore.StandardTool {
	return sporecore.StandardTool{
		Implementation: echoTool{name: name},
		Schema: sporecore.RegistryToolSchema{
			Name:        name,
			Description: name,
			Parameters:  json.RawMessage(`{"type":"object"}`),
		},
	}
}

func builderForCatalogue(agent sporecore.Agent) *HarnessBuilder {
	return NewHarnessBuilder(
		agent,
		sporecore.NewScriptedToolRegistry(),
		sporecore.AllowAllSandbox{},
		sporecore.NoopContextManager{},
		sporecore.AlwaysContinuePolicy{},
	)
}

// sentinelSandbox is a distinct SandboxProvider used to assert the Sandbox()
// setter overrides the sandbox the builder was constructed with. It embeds
// AllowAllSandbox for the SandboxProvider behaviour and carries an id so the
// configured value can be identity-checked.
type sentinelSandbox struct {
	sporecore.AllowAllSandbox
	id int
}

// Mirrors the Rust sandbox_setter_overrides_the_configured_sandbox test: the
// Sandbox() setter overrides the sandbox the builder was constructed with, and
// that overriding value is what lands in the built HarnessConfig.
func TestSandboxSetterOverridesTheConfiguredSandbox(t *testing.T) {
	override := sentinelSandbox{id: 7}
	cfg := builderForCatalogue(sporecore.NewMockAgent("t")).
		Sandbox(override).
		BuildConfig()

	got, ok := cfg.Sandbox.(sentinelSandbox)
	if !ok {
		t.Fatalf("expected the override sandbox, got %T", cfg.Sandbox)
	}
	if got.id != override.id {
		t.Fatalf("sandbox not overridden: got id=%d want %d", got.id, override.id)
	}
}

// Issue #91: catalogue tools added via Tool() are folded into a populated
// CatalogueRegistry, and — because catalogue tools are present and no storage was
// wired — the run store defaults to in-memory (not no-op) so a put/get round-trips.
func TestCatalogueToolsFoldWithInMemoryStorage(t *testing.T) {
	cfg := builderForCatalogue(sporecore.NewMockAgent("t")).
		Tool(echoStandardTool("read_file")).
		Tool(echoStandardTool("write_file")).
		BuildConfig()

	if cfg.CatalogueRegistry == nil {
		t.Fatal("expected a folded CatalogueRegistry")
	}
	names := map[string]bool{}
	for _, s := range cfg.CatalogueRegistry.ActiveSchemas(nil) {
		names[s.Name] = true
	}
	if !names["read_file"] || !names["write_file"] {
		t.Fatalf("catalogue schemas missing: %v", names)
	}

	// Storage defaulted to in-memory because catalogue tools are present: a
	// put/get round-trips on the run store.
	if cfg.ToolRunStore == nil {
		t.Fatal("expected an in-memory run store default")
	}
	sid := sporecore.SessionID("s1")
	if err := cfg.ToolRunStore.Put(context.Background(), sid, "k", json.RawMessage(`{"v":1}`)); err != nil {
		t.Fatalf("put: %v", err)
	}
	v, found, err := cfg.ToolRunStore.Get(context.Background(), sid, "k")
	if err != nil || !found {
		t.Fatalf("get: found=%v err=%v", found, err)
	}
	if string(v) != `{"v":1}` {
		t.Fatalf("run store value: %s", v)
	}
}

// Issue #91: with no catalogue tools, the CatalogueRegistry stays nil (the
// ToolRegistry-only path) and no in-memory storage default is applied.
func TestNoCatalogueToolsKeepsToolRegistrySeam(t *testing.T) {
	cfg := builderForCatalogue(sporecore.NewMockAgent("t")).BuildConfig()
	if cfg.CatalogueRegistry != nil {
		t.Fatal("expected nil CatalogueRegistry when no catalogue tools were added")
	}
	if cfg.ToolRunStore != nil {
		t.Fatal("expected no storage default when no catalogue tools were added")
	}
}

// Issue #91: an explicitly wired run store is preserved (not overridden by the
// in-memory default) even when catalogue tools are present.
func TestExplicitStorageIsPreserved(t *testing.T) {
	custom := sporecore.NewInMemoryToolRunStore()
	cfg := builderForCatalogue(sporecore.NewMockAgent("t")).
		Tool(echoStandardTool("read_file")).
		Storage(custom, nil).
		BuildConfig()
	if cfg.ToolRunStore == nil {
		t.Fatal("expected the wired run store")
	}
}
