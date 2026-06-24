package observability

import (
	"context"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// Rule: ConversationalBuilder wires the conversational defaults — an empty
// *StandardToolRegistry, a NullSandbox, and CompleteOnFinalResponse termination
// — mirroring Rust's HarnessBuilder::conversational.
func TestConversationalBuilderDefaults(t *testing.T) {
	model := sporecore.NewMockModel(sporecore.ProviderInfo{
		Name: "mock", ModelID: "mock-1", ContextWindow: 1000,
	})
	cfg := ConversationalBuilder(model).BuildConfig()

	if _, ok := cfg.Sandbox.(sporecore.NullSandbox); !ok {
		t.Fatalf("sandbox = %T, want sporecore.NullSandbox", cfg.Sandbox)
	}
	if _, ok := cfg.TerminationPolicy.(sporecore.CompleteOnFinalResponse); !ok {
		t.Fatalf("termination = %T, want sporecore.CompleteOnFinalResponse", cfg.TerminationPolicy)
	}
	reg, ok := cfg.ToolRegistry.(*sporecore.StandardToolRegistry)
	if !ok {
		t.Fatalf("tool registry = %T, want *sporecore.StandardToolRegistry", cfg.ToolRegistry)
	}
	if schemas := reg.ActiveSchemas(nil); len(schemas) != 0 {
		t.Fatalf("empty registry advertised %d schemas, want 0", len(schemas))
	}
	if cfg.Agent == nil {
		t.Fatalf("agent is nil")
	}
	if cfg.ContextManager == nil {
		t.Fatalf("context manager is nil")
	}
}

// Rule: a conversational harness built from a model runs a single turn to
// success on the model's first final response — the end-to-end few-lines path.
func TestNewConversationalHarnessSingleTurnSuccess(t *testing.T) {
	model := sporecore.NewMockModel(sporecore.ProviderInfo{
		Name: "mock", ModelID: "mock-1", ContextWindow: 1000,
	})
	model.PushResponse(sporecore.ModelResponse{
		Content:    []sporecore.ContentBlock{sporecore.NewTextBlock("Hello, friend!")},
		Usage:      sporecore.TokenUsage{InputTokens: 3, OutputTokens: 4},
		StopReason: sporecore.StopEndTurn,
	})

	h := NewConversationalHarness(model)
	r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(
		sporecore.SimpleTask("Reply with a friendly one-line greeting.")))

	if r.Kind != sporecore.RunSuccess {
		t.Fatalf("kind = %q reason = %+v", r.Kind, r.Reason)
	}
	if r.Output != "Hello, friend!" {
		t.Fatalf("output = %q, want %q", r.Output, "Hello, friend!")
	}
	if r.Turns != 1 {
		t.Fatalf("turns = %d, want 1", r.Turns)
	}
}

// SC-2 acceptance: a conversational harness can be constructed from a model
// bound to the INTERFACE type (not a concrete struct), and the run dispatches
// against that interface value — the equivalent of Rust's
// conversational_arc(Arc<dyn ModelInterface>). Go interfaces are already
// dynamically dispatched, so the model is held once and shared by the agent and
// context manager with no concrete-type enum or dispatch needed.
func TestConversationalHarnessFromInterfaceTypedModel(t *testing.T) {
	mock := sporecore.NewMockModel(sporecore.ProviderInfo{
		Name: "mock", ModelID: "mock-1", ContextWindow: 1000,
	})
	mock.PushResponse(sporecore.ModelResponse{
		Content:    []sporecore.ContentBlock{sporecore.NewTextBlock("from the interface")},
		Usage:      sporecore.TokenUsage{InputTokens: 2, OutputTokens: 3},
		StopReason: sporecore.StopEndTurn,
	})
	// Explicitly interface-typed — the harness must accept the interface value,
	// not require the concrete *MockModel.
	var model sporecore.ModelInterface = mock

	h := NewConversationalHarness(model)
	r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(
		sporecore.SimpleTask("Reply please.")))

	if r.Kind != sporecore.RunSuccess {
		t.Fatalf("kind = %q reason = %+v", r.Kind, r.Reason)
	}
	if r.Output != "from the interface" {
		t.Fatalf("output = %q, want %q", r.Output, "from the interface")
	}
	if got := mock.CallCount(); got != 1 {
		t.Fatalf("call count = %d, want 1 (one shared model instance dispatched once)", got)
	}
}
