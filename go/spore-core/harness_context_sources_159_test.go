package sporecore

import (
	"context"
	"strings"
	"testing"
)

// guideRenderingCM is a minimal ContextManager that renders sources.Guides then
// sources.Memory into a leading System block — mirroring the production
// StandardCompactionAdapter's renderContextBlock so the loop's sources-building +
// system-prompt merge can be asserted without the adapter's model machinery (the
// adapter's own rendering is unit-tested in contextmgr).
type guideRenderingCM struct{}

func (guideRenderingCM) Assemble(_ context.Context, session *SessionState, _ *Task, sources ContextSources) Context {
	messages := append([]Message(nil), session.Messages...)
	var parts []string
	for _, g := range sources.Guides {
		parts = append(parts, "# "+g.ID+"\n"+g.Content)
	}
	for _, m := range sources.Memory {
		parts = append(parts, m.Content)
	}
	if block := strings.Join(parts, "\n\n"); block != "" {
		messages = append([]Message{{Role: RoleSystem, Content: NewTextContent(block)}}, messages...)
	}
	return Context{Messages: messages}
}

func (guideRenderingCM) AppendToolResult(_ context.Context, session *SessionState, result *HarnessToolResult) {
	session.Messages = append(session.Messages, Message{Role: RoleTool, Content: NewTextContent(result.Output.Content)})
}

func (guideRenderingCM) AppendUserMessage(_ context.Context, session *SessionState, text string) {
	session.Messages = append(session.Messages, Message{Role: RoleUser, Content: NewTextContent(text)})
}

func (guideRenderingCM) ShouldCompact(*SessionState) bool { return false }

// SC-26/#115 acceptance (guide half): a guide registered on the harness reaches
// the model through the structural assemble seam — NOT as an ad-hoc User message
// — and the configured system prompt is merged in front of it in a single
// leading System message.
func TestGuideReachesModelViaAssembleSeam(t *testing.T) {
	agent := &capturingAgent{id: AgentID("cap")}
	cfg := standardCfg(agent)
	cfg.ContextManager = guideRenderingCM{}
	cfg.SystemPrompt = "SYSTEM PROMPT"
	cfg.Guides = []Guide{{ID: "audit", Content: "AUDIT PLAYBOOK BODY"}}
	h := NewStandardHarness(cfg)
	if r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(2))); r.Kind != RunSuccess {
		t.Fatalf("run: %+v", r)
	}

	agent.mu.Lock()
	defer agent.mu.Unlock()
	if len(agent.seen) == 0 {
		t.Fatal("the agent must have been called")
	}
	first := agent.seen[0]
	if first.Role != RoleSystem || first.Content.Type != ContentTypeText {
		t.Fatalf("expected a leading System text message; got %+v", first)
	}
	if !strings.HasPrefix(first.Content.Text, "SYSTEM PROMPT") {
		t.Fatalf("system prompt must lead the merged System block: %q", first.Content.Text)
	}
	if !strings.Contains(first.Content.Text, "# audit") || !strings.Contains(first.Content.Text, "AUDIT PLAYBOOK BODY") {
		t.Fatalf("the guide must reach the model structurally: %q", first.Content.Text)
	}
	// The guide is NOT delivered as a stray User message.
	for _, m := range agent.seen {
		if m.Role == RoleUser && m.Content.Type == ContentTypeText && strings.Contains(m.Content.Text, "AUDIT PLAYBOOK BODY") {
			t.Fatalf("guide must not be injected as a User message")
		}
	}
}

// TestBuildContextSourcesAppendsActiveGuides: buildContextSources copies
// config.Guides then appends the catalog's ActiveGuides (manifest + active
// bodies), in that order.
func TestBuildContextSourcesAppendsActiveGuides(t *testing.T) {
	cfg := standardCfg(&capturingAgent{id: AgentID("cap")})
	cfg.Guides = []Guide{{ID: "domain", Content: "DOMAIN GUIDE"}}
	cat := SkillCatalogFromEntries(SkillEntry{Name: "audit", Description: "Audit a module.", Body: "AUDIT BODY"})
	if !cat.Activate("audit") {
		t.Fatal("activate audit")
	}
	cfg.Skills = cat
	h := NewStandardHarness(cfg)

	sources := h.buildContextSources(context.Background(), nil, "")
	if len(sources.Guides) != 3 {
		t.Fatalf("expected config guide + manifest + active body = 3; got %d: %+v", len(sources.Guides), sources.Guides)
	}
	if sources.Guides[0].ID != "domain" {
		t.Fatalf("config guide must come first; got %q", sources.Guides[0].ID)
	}
	if sources.Guides[1].ID != "AVAILABLE SKILLS" {
		t.Fatalf("manifest guide must follow config guides; got %q", sources.Guides[1].ID)
	}
	if sources.Guides[2].ID != "ACTIVE SKILL — audit" || sources.Guides[2].Content != "AUDIT BODY" {
		t.Fatalf("active body guide mismatch; got %+v", sources.Guides[2])
	}
}

// TestBuildContextSourcesEmptyByDefault: with no guides and no catalog, sources
// carry no guides (the byte-identical no-source path).
func TestBuildContextSourcesEmptyByDefault(t *testing.T) {
	h := NewStandardHarness(standardCfg(&capturingAgent{id: AgentID("cap")}))
	sources := h.buildContextSources(context.Background(), nil, "")
	if len(sources.Guides) != 0 {
		t.Fatalf("expected no guides by default; got %+v", sources.Guides)
	}
}
