package sporecore_test

// Issue #163 / SC-26 follow-up (the memory half): a MemoryProvider wired onto the
// harness has its relevant items reach the model through the SAME structural
// assemble seam as guides — a leading System block, NOT an ad-hoc User message.
//
// These tests live in an external test package (sporecore_test) because they
// import the memory subpackage, which imports sporecore — an internal (package
// sporecore) test importing memory would be an import cycle. The per-turn query
// path (buildContextSources) is exercised end-to-end through Run: a recording
// context manager captures the ContextSources.Memory the loop built each turn.

import (
	"context"
	"strings"
	"sync"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/memory"
)

// ── shared helpers (exported-API mirrors of the in-package test doubles) ──────

func memTurnUsage() sporecore.TokenUsage {
	return sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1}
}

func memStandardCfg(agent sporecore.Agent) sporecore.HarnessConfig {
	return sporecore.HarnessConfig{
		Agent:             agent,
		ToolRegistry:      sporecore.NewScriptedToolRegistry(),
		Sandbox:           sporecore.AllowAllSandbox{},
		ContextManager:    sporecore.NoopContextManager{},
		TerminationPolicy: sporecore.AlwaysContinuePolicy{},
		EscalationMode:    sporecore.AutonomousEscalation(),
	}
}

func memReactTask(instruction string, max uint32) sporecore.Task {
	return sporecore.NewTask(instruction, sporecore.SessionID("s1"), sporecore.ReActStrategy(max))
}

// memProviderWith returns a StandardMemoryProvider holding a single Active
// semantic memory.
func memProviderWith(t *testing.T, id, content string) *memory.StandardMemoryProvider {
	t.Helper()
	p := memory.NewStandardMemoryProvider()
	sem := memory.SemanticMemory{
		ID:        memory.MemoryID(id),
		Content:   content,
		Source:    memory.NewSourceManual(),
		Version:   1,
		CreatedAt: memory.Timestamp("2026-06-24T00:00:00Z"),
		UpdatedAt: memory.Timestamp("2026-06-24T00:00:00Z"),
		Status:    memory.NewStatusActive(),
	}
	if _, err := p.StoreSemantic(context.Background(), sem, memory.MergeStrategyReject); err != nil {
		t.Fatalf("store semantic: %v", err)
	}
	return p
}

// memCapturingAgent records the messages of the context it was handed, then
// replies with a final response — so a test can assert what the model saw.
type memCapturingAgent struct {
	id   sporecore.AgentID
	mu   sync.Mutex
	seen []sporecore.Message
}

func (a *memCapturingAgent) Turn(_ context.Context, c sporecore.Context) sporecore.TurnResult {
	a.mu.Lock()
	a.seen = append([]sporecore.Message(nil), c.Messages...)
	a.mu.Unlock()
	return sporecore.NewFinalResponse("done", memTurnUsage())
}
func (a *memCapturingAgent) ID() sporecore.AgentID { return a.id }

// memRenderingCM renders sources.Guides then sources.Memory into a leading System
// block — mirroring the production StandardCompactionAdapter's renderContextBlock
// — AND records the ContextSources.Memory it was handed each turn so the per-turn
// query path can be asserted through Run.
type memRenderingCM struct {
	mu       sync.Mutex
	lastSeen []sporecore.MemoryItem
}

func (m *memRenderingCM) Assemble(_ context.Context, session *sporecore.SessionState, _ *sporecore.Task, sources sporecore.ContextSources) sporecore.Context {
	m.mu.Lock()
	m.lastSeen = append([]sporecore.MemoryItem(nil), sources.Memory...)
	m.mu.Unlock()

	messages := append([]sporecore.Message(nil), session.Messages...)
	var parts []string
	for _, g := range sources.Guides {
		parts = append(parts, "# "+g.ID+"\n"+g.Content)
	}
	for _, it := range sources.Memory {
		parts = append(parts, it.Content)
	}
	if block := strings.Join(parts, "\n\n"); block != "" {
		messages = append([]sporecore.Message{{Role: sporecore.RoleSystem, Content: sporecore.NewTextContent(block)}}, messages...)
	}
	return sporecore.Context{Messages: messages}
}

func (m *memRenderingCM) AppendToolResult(_ context.Context, session *sporecore.SessionState, result *sporecore.HarnessToolResult) {
	session.Messages = append(session.Messages, sporecore.Message{Role: sporecore.RoleTool, Content: sporecore.NewTextContent(result.Output.Content)})
}

func (m *memRenderingCM) AppendUserMessage(_ context.Context, session *sporecore.SessionState, text string) {
	session.Messages = append(session.Messages, sporecore.Message{Role: sporecore.RoleUser, Content: sporecore.NewTextContent(text)})
}

func (m *memRenderingCM) ShouldCompact(*sporecore.SessionState) bool { return false }

func (m *memRenderingCM) memory() []sporecore.MemoryItem {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]sporecore.MemoryItem(nil), m.lastSeen...)
}

var _ sporecore.ContextManager = (*memRenderingCM)(nil)

// runWithMemory drives a single-turn ReAct run with the given memory config and
// task instruction, returning the capturing agent + the CM that recorded the
// per-turn ContextSources.Memory. A nil cfg leaves no memory configured.
func runWithMemory(t *testing.T, cfg *sporecore.MemoryConfig, systemPrompt, instruction string) (*memCapturingAgent, *memRenderingCM) {
	t.Helper()
	agent := &memCapturingAgent{id: sporecore.AgentID("cap")}
	hc := memStandardCfg(agent)
	cm := &memRenderingCM{}
	hc.ContextManager = cm
	hc.SystemPrompt = systemPrompt
	hc.Memory = cfg
	h := sporecore.NewStandardHarness(hc)
	if r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(memReactTask(instruction, 2))); r.Kind != sporecore.RunSuccess {
		t.Fatalf("run: %+v", r)
	}
	return agent, cm
}

// Test 1: #163 acceptance — a memory provider registered on the harness has its
// relevant items reach the model through the structural assemble seam (a leading
// System block), NOT as a User message. Exercises the object-safe MemoryProvider
// behind memory.NewMemoryConfig.
func TestMemoryReachesModelViaAssembleSeam(t *testing.T) {
	provider := memProviderWith(t, "m1", "REFUND IDEMPOTENCY: audit the payments refund path")
	cfg := memory.NewMemoryConfig(provider, memory.WithMinRelevance(0.1))

	agent, _ := runWithMemory(t, &cfg, "SYSTEM PROMPT", "audit the payments refund path")

	agent.mu.Lock()
	defer agent.mu.Unlock()
	if len(agent.seen) == 0 {
		t.Fatal("the agent must have been called")
	}
	first := agent.seen[0]
	if first.Role != sporecore.RoleSystem || first.Content.Type != sporecore.ContentTypeText {
		t.Fatalf("expected a leading System text message; got %+v", first)
	}
	if !strings.HasPrefix(first.Content.Text, "SYSTEM PROMPT") {
		t.Fatalf("system prompt must lead the merged System block: %q", first.Content.Text)
	}
	if !strings.Contains(first.Content.Text, "REFUND IDEMPOTENCY") {
		t.Fatalf("the memory must reach the model structurally: %q", first.Content.Text)
	}
	for _, m := range agent.seen {
		if m.Role == sporecore.RoleUser && m.Content.Type == sporecore.ContentTypeText && strings.Contains(m.Content.Text, "REFUND IDEMPOTENCY") {
			t.Fatalf("memory must not be injected as a User message")
		}
	}
}

// Test 2: the per-turn query text defaults to the task instruction — a related
// instruction surfaces the memory; an unrelated one (at the same threshold) does
// not. 0.3 cleanly separates jaccard(≈0.83, related) from jaccard(≈0.11,
// unrelated — a shared stopword).
func TestMemoryDefaultQueryTracksTaskInstruction(t *testing.T) {
	provider := memProviderWith(t, "m1", "REFUND IDEMPOTENCY: audit the payments refund path")
	cfg := memory.NewMemoryConfig(provider, memory.WithMinRelevance(0.3))

	_, related := runWithMemory(t, &cfg, "", "audit the payments refund path")
	if got := related.memory(); len(got) != 1 || !strings.Contains(got[0].Content, "REFUND IDEMPOTENCY") {
		t.Fatalf("the relevant memory must surface for a related instruction; got %+v", got)
	}

	_, unrelated := runWithMemory(t, &cfg, "", "compile the rust workspace")
	if got := unrelated.memory(); len(got) != 0 {
		t.Fatalf("an unrelated instruction must not surface the memory; got %+v", got)
	}
}

// Test 3: a configured query text overrides the task instruction — so even an
// unrelated instruction surfaces the memory when the fixed query matches.
func TestMemoryConfiguredQueryOverridesTaskInstruction(t *testing.T) {
	provider := memProviderWith(t, "m1", "REFUND IDEMPOTENCY: audit the payments refund path")
	cfg := memory.NewMemoryConfig(provider,
		memory.WithQuery("audit the payments refund path"),
		memory.WithMinRelevance(0.1))

	_, cm := runWithMemory(t, &cfg, "", "compile the rust workspace")
	if got := cm.memory(); len(got) != 1 {
		t.Fatalf("the configured query must override the task instruction; got %+v", got)
	}
}

// Test 4: no memory configured leaves ContextSources.Memory empty (the
// byte-identical pre-#163 path), regardless of instruction.
func TestNoMemoryProviderLeavesMemoryEmpty(t *testing.T) {
	_, cm := runWithMemory(t, nil, "", "audit the payments refund path")
	if got := cm.memory(); len(got) != 0 {
		t.Fatalf("no provider configured must leave memory empty; got %+v", got)
	}
}
