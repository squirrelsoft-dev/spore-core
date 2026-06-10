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

// sentinelContextManager is a distinct ContextManager used to assert the
// ContextManager() setter overrides the manager the builder was constructed
// with. It embeds NoopContextManager for the behaviour and carries an id so the
// configured value can be identity-checked.
type sentinelContextManager struct {
	sporecore.NoopContextManager
	id int
}

// Mirrors the Rust context_manager_setter_overrides_the_configured_manager
// test: the ContextManager() setter overrides the manager the builder was
// constructed with, and that overriding value is what lands in the built
// HarnessConfig.
func TestContextManagerSetterOverridesTheConfiguredManager(t *testing.T) {
	override := sentinelContextManager{id: 9}
	cfg := builderForCatalogue(sporecore.NewMockAgent("t")).
		ContextManager(override).
		BuildConfig()

	got, ok := cfg.ContextManager.(sentinelContextManager)
	if !ok {
		t.Fatalf("expected the override context manager, got %T", cfg.ContextManager)
	}
	if got.id != override.id {
		t.Fatalf("context manager not overridden: got id=%d want %d", got.id, override.id)
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

// ---- Issue 2: ToolsetTools builder fold -----------------------------------

// ToolsetTools folds each per-key bucket into its OWN populated catalogue on
// HarnessConfig.ToolsetCatalogues (last-wins upsert, additive across calls), and
// NewStandardHarness auto-fills a registry presence entry for each so a tree
// referencing the handle passes Validate without a manual placeholder. Mirrors
// the Rust toolset_tools_autofill_registry_presence_for_validate test.
func TestToolsetToolsFoldsPerKeyCataloguesAndAutofillsRegistry(t *testing.T) {
	cfg := builderForCatalogue(sporecore.NewMockAgent("t")).
		ToolsetTools("plan-tools", echoStandardTool("list_dir")).
		ToolsetTools("plan-tools", echoStandardTool("task_list")). // additive across calls
		ToolsetTools("exec-tools", echoStandardTool("read_file")).
		BuildConfig()

	plan, ok := cfg.ToolsetCatalogues["plan-tools"]
	if !ok {
		t.Fatal("expected a plan-tools per-key catalogue")
	}
	exec, ok := cfg.ToolsetCatalogues["exec-tools"]
	if !ok {
		t.Fatal("expected an exec-tools per-key catalogue")
	}

	// plan-tools advertises BOTH accumulated tools; exec-tools only its own.
	planNames := schemaNames(plan)
	if !planNames["list_dir"] || !planNames["task_list"] {
		t.Fatalf("plan-tools catalogue must hold list_dir+task_list, got %v", planNames)
	}
	if planNames["read_file"] {
		t.Fatalf("plan-tools catalogue must NOT hold the exec-only read_file, got %v", planNames)
	}
	if execNames := schemaNames(exec); !execNames["read_file"] || execNames["list_dir"] {
		t.Fatalf("exec-tools catalogue must hold only read_file, got %v", execNames)
	}

	// Catalogue tools present ⇒ in-memory run store defaulted (no storage wired),
	// mirroring the global-catalogue default. (The NewStandardHarness registry
	// auto-fill is asserted in the core package's
	// TestToolsetCataloguesAutofillRegistryPresenceForValidate.)
	if cfg.ToolRunStore == nil {
		t.Fatal("expected an in-memory run store default when toolset catalogues are present")
	}
}

func schemaNames(reg *sporecore.StandardToolRegistry) map[string]bool {
	out := map[string]bool{}
	for _, s := range reg.ActiveSchemas(nil) {
		out[s.Name] = true
	}
	return out
}
