package preset

import (
	"path/filepath"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/metric"
)

// ── SC-8 presets: CodingAgent + HillClimber ─────────────────────────────────
//
// Mirrors rust/crates/spore-core/src/harness.rs (6f39933):
// coding_agent_preset_{wires_sandbox_tools_prompt_and_autocontinue,
// errors_on_missing_workspace}, hill_climber_preset_registers_evaluator_and_autocontinue.

func presetMockModel() *sporecore.MockModel {
	return sporecore.NewMockModel(sporecore.ProviderInfo{
		Name: "test", ModelID: "test-1", ContextWindow: 8192,
	})
}

// SC-8: CodingAgent wires the read-write workspace sandbox, the coding tool
// catalogue, the built-in coding system prompt, and AutoContinue (the looper
// preset) — so a consumer collapses to one call.
func TestCodingAgentPresetWiresSandboxToolsPromptAndAutocontinue(t *testing.T) {
	b, err := CodingAgent(presetMockModel(), t.TempDir())
	if err != nil {
		t.Fatalf("CodingAgent over a temp dir must build, got %v", err)
	}
	cfg := b.BuildConfig()

	// Autonomous-but-capped escalation with the preset defaults (SC-5).
	mode := cfg.EscalationMode
	if mode.Kind != sporecore.EscalationAutoContinue {
		t.Fatalf("escalation kind = %q, want auto_continue", mode.Kind)
	}
	if mode.MaxGrants != PresetMaxAutoGrants || mode.StepsPerGrant != PresetStepsPerGrant {
		t.Fatalf("escalation grants = {%d,%d}, want {%d,%d}",
			mode.MaxGrants, mode.StepsPerGrant, PresetMaxAutoGrants, PresetStepsPerGrant)
	}

	// The read-write workspace-scoped sandbox is installed.
	sb, ok := cfg.Sandbox.(*sporecore.WorkspaceScopedSandbox)
	if !ok {
		t.Fatalf("sandbox = %T, want *sporecore.WorkspaceScopedSandbox", cfg.Sandbox)
	}
	if sb.Config().ReadOnly {
		t.Fatal("coding_agent sandbox must be read-write, got read-only")
	}

	// The built-in coding system prompt is installed.
	if cfg.SystemPrompt != CodingAgentSystemPrompt {
		t.Fatalf("system prompt = %q, want the built-in CodingAgentSystemPrompt", cfg.SystemPrompt)
	}

	// The coding catalogue is wired (a representative sample of CodingSet()).
	// .Tools() are folded into the CatalogueRegistry (bridged per-run), not the
	// harness-loop ToolRegistry, which stays the conversational empty registry
	// until the run binds a ToolContext.
	catalogue := cfg.CatalogueRegistry
	if catalogue == nil {
		t.Fatal("coding_agent must wire a catalogue registry from CodingSet()")
	}
	have := map[string]bool{}
	for _, s := range catalogue.ActiveSchemas(nil) {
		have[s.Name] = true
	}
	for _, expected := range []string{"read_file", "write_file", "edit_file", "bash_command", "send_message"} {
		if !have[expected] {
			t.Fatalf("coding_set must include %q; got %v", expected, have)
		}
	}
}

// SC-8: a workspace path that can't be resolved is a typed *BuildError, not a
// panic — the sandbox requires a canonical, existing root.
func TestCodingAgentPresetErrorsOnMissingWorkspace(t *testing.T) {
	missing := filepath.Join("/spore-sc8-does-not-exist-37a1", "nope")
	b, err := CodingAgent(presetMockModel(), missing)
	if err == nil {
		t.Fatal("a missing workspace must surface an error, not build")
	}
	if b != nil {
		t.Fatalf("builder must be nil on error, got %v", b)
	}
	if _, ok := err.(*sporecore.BuildError); !ok {
		t.Fatalf("error = %T, want *sporecore.BuildError", err)
	}
}

// SC-8: HillClimber registers the scoring evaluator (required for the
// HillClimbing strategy) under the default handle and defaults to AutoContinue —
// the cordyceps preset.
func TestHillClimberPresetRegistersEvaluatorAndAutocontinue(t *testing.T) {
	evaluator := &metric.LatencyEvaluator{
		Command:      "true",
		Args:         nil,
		WarmupRuns:   0,
		MeasuredRuns: 1,
		Timeout:      time.Second,
		WorkingDir:   "",
	}
	cfg := HillClimber(presetMockModel(), evaluator).BuildConfig()

	if _, ok := cfg.Registry.ResolveMetricEvaluator(""); !ok {
		t.Fatal("hill_climber must register the metric evaluator under the default handle")
	}
	if cfg.EscalationMode.Kind != sporecore.EscalationAutoContinue {
		t.Fatalf("hill_climber escalation kind = %q, want auto_continue", cfg.EscalationMode.Kind)
	}
}
