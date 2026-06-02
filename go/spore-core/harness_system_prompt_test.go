package sporecore

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
)

// ---- issue #91: catalogue bridge + system_prompt seam --------------------

// echoBridgeTool is a trivial catalogue tool used to populate a CatalogueRegistry
// in-package (the tools package transitively imports observability, so it can't
// be imported here without a cycle).
type echoBridgeTool struct{ name string }

func (t echoBridgeTool) Name() string                { return t.name }
func (t echoBridgeTool) IsSubagentTool() bool        { return false }
func (t echoBridgeTool) MayProduceLargeOutput() bool { return false }
func (t echoBridgeTool) Execute(_ context.Context, _ ToolCall, _ SandboxProvider, _ *ToolContext) ToolOutput {
	return NewToolOutputSuccess("echo")
}

func catalogueRegistryWith(names ...string) *StandardToolRegistry {
	reg := NewStandardToolRegistry()
	for _, n := range names {
		_ = reg.Register(echoBridgeTool{name: n}, RegistryToolSchema{
			Name:        n,
			Description: n,
			Parameters:  json.RawMessage(`{"type":"object"}`),
		})
	}
	return reg
}

// effectiveToolRegistry returns the CatalogueRegistry (bridge) when catalogue
// tools are present: it advertises the catalogue schemas to the model and maps an
// unknown-tool dispatch to a recoverable error so the loop appends it and the
// agent can adapt.
func TestEffectiveRegistryBridgesCatalogueTools(t *testing.T) {
	cfg := standardCfg(NewMockAgent("t"))
	cfg.CatalogueRegistry = catalogueRegistryWith("read_file")
	h := NewStandardHarness(cfg)

	reg := h.effectiveToolRegistry(SessionID("s1"))

	// Advertises the catalogue schema.
	var advertised bool
	for _, s := range reg.ActiveSchemas(nil) {
		if s.Name == "read_file" {
			advertised = true
		}
	}
	if !advertised {
		t.Fatal("bridge did not advertise the catalogue schema")
	}

	// An unknown-tool dispatch maps to a recoverable error.
	out := dispatchAndUnwrap(context.Background(), reg, AllowAllSandbox{}, ToolCall{
		ID: "c", Name: "does_not_exist", Input: json.RawMessage(`{}`),
	})
	if out.Kind != ToolOutputError {
		t.Fatalf("expected an error output, got %q", out.Kind)
	}
}

// When no catalogue tools were folded, effectiveToolRegistry returns the injected
// ToolRegistry seam unchanged.
func TestEffectiveRegistryKeepsSeamWithoutCatalogue(t *testing.T) {
	cfg := standardCfg(NewMockAgent("t"))
	h := NewStandardHarness(cfg)
	if h.effectiveToolRegistry(SessionID("s1")) != cfg.ToolRegistry {
		t.Fatal("expected the injected ToolRegistry seam when no catalogue is set")
	}
}

// The bridge threads the run's SessionID into the registry's ToolContext so a
// catalogue tool's storage is keyed by the live session.
func TestEffectiveRegistryThreadsSessionAndStorage(t *testing.T) {
	cfg := standardCfg(NewMockAgent("t"))
	cfg.CatalogueRegistry = catalogueRegistryWith("read_file")
	run := NewInMemoryToolRunStore()
	cfg.ToolRunStore = run
	h := NewStandardHarness(cfg)

	reg := h.effectiveToolRegistry(SessionID("sess-xyz")).(*StandardToolRegistry)
	tc := reg.dispatchToolContext()
	if tc.SessionID != SessionID("sess-xyz") {
		t.Fatalf("ToolContext SessionID = %q, want sess-xyz", tc.SessionID)
	}
	if tc.RunStore == nil {
		t.Fatal("ToolContext RunStore was not threaded")
	}
}

// capturingAgent records the messages of the context it was handed, then replies
// with a final response — lets a test assert what the model saw.
type capturingAgent struct {
	id   AgentID
	mu   sync.Mutex
	seen []Message
}

func (a *capturingAgent) Turn(_ context.Context, c Context) TurnResult {
	a.mu.Lock()
	a.seen = append([]Message(nil), c.Messages...)
	a.mu.Unlock()
	return NewFinalResponse("done", turnUsage())
}
func (a *capturingAgent) ID() AgentID { return a.id }

// Issue #91: a configured SystemPrompt is prepended as a leading System message
// to each turn's assembled context.
func TestSystemPromptIsPrepended(t *testing.T) {
	agent := &capturingAgent{id: AgentID("cap")}
	cfg := standardCfg(agent)
	cfg.SystemPrompt = "OPERATING RULES"
	h := NewStandardHarness(cfg)
	if r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(2))); r.Kind != RunSuccess {
		t.Fatalf("run: %+v", r)
	}

	agent.mu.Lock()
	defer agent.mu.Unlock()
	if len(agent.seen) == 0 {
		t.Fatal("context had no messages")
	}
	first := agent.seen[0]
	if first.Role != RoleSystem {
		t.Fatalf("first message role = %q, want system", first.Role)
	}
	if first.Content.Type != ContentTypeText || first.Content.Text != "OPERATING RULES" {
		t.Fatalf("first message content = %+v", first.Content)
	}
}

// Issue #91: with no SystemPrompt set, the context carries no System message
// (today's default behaviour is preserved).
func TestNoSystemPromptLeavesContextWithoutSystem(t *testing.T) {
	agent := &capturingAgent{id: AgentID("cap")}
	cfg := standardCfg(agent) // SystemPrompt stays empty.
	h := NewStandardHarness(cfg)
	if r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(2))); r.Kind != RunSuccess {
		t.Fatalf("run: %+v", r)
	}

	agent.mu.Lock()
	defer agent.mu.Unlock()
	for _, m := range agent.seen {
		if m.Role == RoleSystem {
			t.Fatal("expected no System message when SystemPrompt is unset")
		}
	}
}

// Issue #91: when the assembled context already leads with a System message, the
// SystemPrompt is NOT prepended again (no duplicate).
func TestSystemPromptNotDuplicated(t *testing.T) {
	agent := &capturingAgent{id: AgentID("cap")}
	cfg := standardCfg(agent)
	cfg.SystemPrompt = "OPERATING RULES"
	// Seed the session so Assemble (NoopContextManager copies session messages)
	// already leads with a System message.
	cfg.ContextManager = NoopContextManager{}
	h := NewStandardHarness(cfg)

	task := reactTask(2)
	session := SessionState{Messages: []Message{
		{Role: RoleSystem, Content: NewTextContent("EXISTING SYSTEM")},
		{Role: RoleUser, Content: NewTextContent("hi")},
	}}
	r := h.runReAct(context.Background(), task, 2, session, BudgetSnapshot{}, nil)
	if r.Kind != RunSuccess {
		t.Fatalf("run: %+v", r)
	}

	agent.mu.Lock()
	defer agent.mu.Unlock()
	systemCount := 0
	for _, m := range agent.seen {
		if m.Role == RoleSystem {
			systemCount++
		}
	}
	if systemCount != 1 {
		t.Fatalf("expected exactly one System message, got %d", systemCount)
	}
	if agent.seen[0].Content.Text != "EXISTING SYSTEM" {
		t.Fatalf("existing System message was overwritten: %q", agent.seen[0].Content.Text)
	}
}
