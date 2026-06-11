package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

// ---- Issue 2: per-node toolset scoping (effectiveToolRegistry) -------------
//
// These mirror the Rust per_node_toolset_* tests in harness.rs. They build a
// HarnessConfig with the per-toolset catalogues populated directly (the
// .ToolsetTools() builder lives in the observability package, which the tools
// package transitively imports, so it can't be reached from the core package
// without a cycle — exactly like the existing CatalogueRegistry bridge tests).

// toolsetScopingCfg is the shared config for the scoping tests: a mock agent
// over the standard test seams, with the global + per-key catalogues injected by
// the caller.
func toolsetScopingCfg() HarnessConfig {
	return standardCfg(NewMockAgent("t"))
}

// advertises reports whether reg's active schemas include a tool of name.
func advertises(reg ToolRegistry, name string) bool {
	for _, s := range reg.ActiveSchemas(nil) {
		if s.Name == name {
			return true
		}
	}
	return false
}

// dispatchedOK reports whether dispatching name through reg yields a non-error
// output (i.e. the tool was actually available to that node).
func dispatchedOK(reg ToolRegistry, name string) bool {
	out := dispatchAndUnwrap(context.Background(), reg, AllowAllSandbox{}, ToolCall{
		ID: "c", Name: name, Input: json.RawMessage(`{}`),
	})
	return out.Kind != ToolOutputError
}

// A node carrying a NON-EMPTY toolset handle dispatches ONLY that toolset's
// catalogue: the planner (plan-tools = list_dir/task_list) cannot call an
// exec-only tool (read_file), and the executor (exec-tools = read_file) cannot
// call a plan-only tool (task_list/list_dir). The leaked union is closed.
func TestPerNodeToolsetScopingClosesCrossNodeLeaks(t *testing.T) {
	cfg := toolsetScopingCfg()
	cfg.ToolsetCatalogues = map[string]*StandardToolRegistry{
		"plan-tools": catalogueRegistryWith("list_dir", "task_list"),
		"exec-tools": catalogueRegistryWith("read_file"),
	}
	h := NewStandardHarness(cfg)
	sid := SessionID("s1")

	// Planner node: plan-tools only. Its OWN tools advertise; exec-only tools are
	// NOT available, and the leaked dispatch the live run exhibited now fails.
	plan := h.effectiveToolRegistry(sid, ToolsetRef("plan-tools"))
	if !advertises(plan, "list_dir") || !advertises(plan, "task_list") {
		t.Fatal("plan-tools node must advertise its own list_dir/task_list")
	}
	if advertises(plan, "read_file") {
		t.Fatal("plan-tools node must NOT advertise the exec-only read_file")
	}
	if dispatchedOK(plan, "read_file") {
		t.Fatal("plan-tools node must NOT be able to dispatch the exec-only read_file")
	}

	// Executor node: exec-tools only. It cannot reach the plan-only tools.
	exec := h.effectiveToolRegistry(sid, ToolsetRef("exec-tools"))
	if !advertises(exec, "read_file") {
		t.Fatal("exec-tools node must advertise its own read_file")
	}
	if advertises(exec, "task_list") || advertises(exec, "list_dir") {
		t.Fatal("exec-tools node must NOT advertise plan-only tools")
	}
	if dispatchedOK(exec, "task_list") || dispatchedOK(exec, "list_dir") {
		t.Fatal("exec-tools node must NOT be able to dispatch plan-only tools")
	}
}

// An unknown tool called by a scoped node still hits the existing unknown-tool
// error path (no regression of the error path): the scoped catalogue routes the
// dispatch exactly as the global catalogue does. In Go the unregistered-tool
// DispatchError is mapped to an UNRECOVERABLE ToolOutputError by
// dispatchAndUnwrap (a pre-existing Go vs. Rust divergence: Rust's RealToolRegistry
// bridge maps every DispatchError to recoverable=true). This test asserts the
// EXISTING Go contract — the scoped node reaches the same unknown-tool path, not a
// different one — rather than changing that mapping (out of scope for Issue 2).
func TestPerNodeToolsetUnknownToolHitsErrorPath(t *testing.T) {
	cfg := toolsetScopingCfg()
	cfg.ToolsetCatalogues = map[string]*StandardToolRegistry{
		"plan-tools": catalogueRegistryWith("list_dir"),
	}
	h := NewStandardHarness(cfg)
	scoped := h.effectiveToolRegistry(SessionID("s1"), ToolsetRef("plan-tools"))

	scopedOut := dispatchAndUnwrap(context.Background(), scoped, AllowAllSandbox{}, ToolCall{
		ID: "c", Name: "does_not_exist", Input: json.RawMessage(`{}`),
	})
	if scopedOut.Kind != ToolOutputError {
		t.Fatalf("expected an error output from the scoped node, got %q", scopedOut.Kind)
	}

	// The scoped node reaches the SAME unknown-tool error path the global
	// catalogue does — recoverability is identical (not a new/different path).
	cfg2 := toolsetScopingCfg()
	cfg2.CatalogueRegistry = catalogueRegistryWith("list_dir")
	h2 := NewStandardHarness(cfg2)
	global := h2.effectiveToolRegistry(SessionID("s1"), ToolsetRef(""))
	globalOut := dispatchAndUnwrap(context.Background(), global, AllowAllSandbox{}, ToolCall{
		ID: "c", Name: "does_not_exist", Input: json.RawMessage(`{}`),
	})
	if globalOut.Kind != scopedOut.Kind || globalOut.Recoverable != scopedOut.Recoverable {
		t.Fatalf("scoped unknown-tool path (kind=%q recoverable=%v) must match the global path (kind=%q recoverable=%v)",
			scopedOut.Kind, scopedOut.Recoverable, globalOut.Kind, globalOut.Recoverable)
	}
}

// A node with an EMPTY toolset handle still sees the GLOBAL catalogue wired via
// .Tools() / CatalogueRegistry (back-compat with examples 01–11 that never
// scope). The per-key catalogues do NOT leak into the empty-handle fallback.
func TestEmptyToolsetHandleFallsBackToGlobalCatalogue(t *testing.T) {
	cfg := toolsetScopingCfg()
	cfg.CatalogueRegistry = catalogueRegistryWith("read_file") // global catalogue
	cfg.ToolsetCatalogues = map[string]*StandardToolRegistry{
		"plan-tools": catalogueRegistryWith("list_dir"), // scoped
	}
	h := NewStandardHarness(cfg)
	global := h.effectiveToolRegistry(SessionID("s1"), ToolsetRef(""))
	// Empty handle ⇒ global catalogue (read_file), NOT the scoped plan-tools.
	if !advertises(global, "read_file") {
		t.Fatal("empty handle must fall back to the global catalogue (read_file)")
	}
	if advertises(global, "list_dir") {
		t.Fatal("the scoped plan-tools catalogue must NOT leak into the empty-handle fallback")
	}
}

// A non-empty toolset handle with NO registered per-key catalogue falls back to
// the global catalogue / seam (additive change — an unscoped name does not
// strand the node with zero tools).
func TestUnknownToolsetHandleFallsBackToGlobalCatalogue(t *testing.T) {
	cfg := toolsetScopingCfg()
	cfg.CatalogueRegistry = catalogueRegistryWith("read_file")
	h := NewStandardHarness(cfg)
	reg := h.effectiveToolRegistry(SessionID("s1"), ToolsetRef("not-wired"))
	if !advertises(reg, "read_file") {
		t.Fatal("an unwired non-empty handle must fall back to the global catalogue")
	}
}

// NewStandardHarness auto-registers a registry presence entry for every per-key
// catalogue, so a tree referencing that toolset handle passes
// ExecutionRegistry.Validate without the caller wiring a placeholder. Mirrors the
// Rust toolset_tools_autofill_registry_presence_for_validate test.
func TestToolsetCataloguesAutofillRegistryPresenceForValidate(t *testing.T) {
	cfg := toolsetScopingCfg()
	cfg.ToolsetCatalogues = map[string]*StandardToolRegistry{
		"plan-tools": catalogueRegistryWith("list_dir"),
	}
	h := NewStandardHarness(cfg)
	if _, ok := h.config.Registry.ResolveToolset(ToolsetRef("plan-tools")); !ok {
		t.Fatal("expected a registry presence entry for the per-key catalogue handle")
	}
	// And the dispatchable catalogue is present.
	if _, ok := h.config.ToolsetCatalogues["plan-tools"]; !ok {
		t.Fatal("expected the dispatchable per-key catalogue on the config")
	}
}
