package sporecore

import (
	"context"
	"strings"
	"testing"
)

// ---- ParseSkillDoc ----------------------------------------------------------

func TestParseSkillDocNameDescriptionBody(t *testing.T) {
	doc := "---\nname: security-review\ndescription: Review code for security issues.\n---\n\n# Procedure\n\nDo the thing.\n"
	entry, ok := ParseSkillDoc(doc)
	if !ok {
		t.Fatal("should parse")
	}
	if entry.Name != "security-review" {
		t.Fatalf("name = %q", entry.Name)
	}
	if entry.Description != "Review code for security issues." {
		t.Fatalf("description = %q", entry.Description)
	}
	if entry.Body != "# Procedure\n\nDo the thing.\n" {
		t.Fatalf("body = %q", entry.Body)
	}
}

func TestParseSkillDocToleratesOptionalAndStripsQuotes(t *testing.T) {
	doc := "---\nname: \"pdf\"\ndescription: 'Handle PDFs.'\nlicense: Apache-2.0\nmetadata:\n  author: me\n---\nbody\n"
	entry, ok := ParseSkillDoc(doc)
	if !ok {
		t.Fatal("should parse")
	}
	if entry.Name != "pdf" {
		t.Fatalf("name = %q", entry.Name)
	}
	if entry.Description != "Handle PDFs." {
		t.Fatalf("description = %q", entry.Description)
	}
	if entry.Body != "body\n" {
		t.Fatalf("body = %q", entry.Body)
	}
}

func TestParseSkillDocRejects(t *testing.T) {
	if _, ok := ParseSkillDoc("---\ndescription: no name\n---\nbody\n"); ok {
		t.Fatal("expected reject: missing name")
	}
	if _, ok := ParseSkillDoc("---\nname: x\ndescription: d\n---\n   \n"); ok {
		t.Fatal("expected reject: empty body")
	}
	if _, ok := ParseSkillDoc("no frontmatter at all"); ok {
		t.Fatal("expected reject: no frontmatter")
	}
}

// ---- SkillCatalog -----------------------------------------------------------

func TestSkillCatalogEmptyYieldsNoGuides(t *testing.T) {
	cat := SkillCatalogFromEntries()
	if !cat.IsEmpty() {
		t.Fatal("expected empty catalog")
	}
	if g := cat.ActiveGuides(); g != nil {
		t.Fatalf("empty catalog must yield no guides; got %+v", g)
	}
}

func TestSkillCatalogManifestThenActiveBody(t *testing.T) {
	cat := SkillCatalogFromEntries(
		SkillEntry{Name: "audit", Description: "Audit a module.", Body: "AUDIT BODY"},
		SkillEntry{Name: "style", Description: "Style guide.", Body: "STYLE BODY"},
	)

	// Before activation: only the manifest guide.
	guides := cat.ActiveGuides()
	if len(guides) != 1 {
		t.Fatalf("expected only the manifest guide; got %d", len(guides))
	}
	if guides[0].ID != "AVAILABLE SKILLS" {
		t.Fatalf("manifest guide ID = %q", guides[0].ID)
	}
	if !strings.Contains(guides[0].Content, "- audit: Audit a module.") {
		t.Fatalf("manifest missing audit line:\n%s", guides[0].Content)
	}
	if !strings.Contains(guides[0].Content, "- style: Style guide.") {
		t.Fatalf("manifest missing style line:\n%s", guides[0].Content)
	}

	// Unknown activation is rejected (no change); known activation is sticky.
	if cat.Activate("nope") {
		t.Fatal("unknown skill must be rejected")
	}
	if len(cat.ActiveGuides()) != 1 {
		t.Fatal("rejected activation must not add a guide")
	}
	if !cat.Activate("audit") {
		t.Fatal("known skill must activate")
	}
	guides = cat.ActiveGuides()
	if len(guides) != 2 {
		t.Fatalf("expected manifest + active body; got %d", len(guides))
	}
	if guides[1].ID != "ACTIVE SKILL — audit" || guides[1].Content != "AUDIT BODY" {
		t.Fatalf("active body guide mismatch: %+v", guides[1])
	}

	// Idempotent (sticky).
	if !cat.Activate("audit") {
		t.Fatal("re-activate must still report true")
	}
	if len(cat.ActiveGuides()) != 2 {
		t.Fatal("re-activation must not duplicate the body guide")
	}

	// ClearActive drops the body but keeps the manifest.
	cat.ClearActive()
	if len(cat.ActiveGuides()) != 1 {
		t.Fatal("ClearActive must drop the active body, leaving only the manifest")
	}
}

func TestSkillCatalogActiveSorted(t *testing.T) {
	cat := SkillCatalogFromEntries(
		SkillEntry{Name: "beta", Description: "b", Body: "B"},
		SkillEntry{Name: "alpha", Description: "a", Body: "A"},
	)
	cat.Activate("beta")
	cat.Activate("alpha")
	got := cat.Active()
	if len(got) != 2 || got[0] != "alpha" || got[1] != "beta" {
		t.Fatalf("Active must be sorted; got %v", got)
	}
}

// TestLoadSkillToolActivates: the load_skill tool validates the name, activates
// against the shared catalog, and returns the verbatim success/error strings.
func TestLoadSkillToolActivates(t *testing.T) {
	cat := SkillCatalogFromEntries(SkillEntry{Name: "audit", Description: "d", Body: "AUDIT BODY"})
	st := cat.LoadSkillTool()
	if st.Schema.Name != LoadSkill {
		t.Fatalf("schema name = %q", st.Schema.Name)
	}
	tool := st.Implementation

	// Missing name → recoverable error.
	out := tool.Execute(context.Background(), ToolCall{Name: LoadSkill, Input: []byte(`{}`)}, AllowAllSandbox{}, nil)
	if out.Kind != ToolOutputError || out.Message != "invalid parameters: `name` (string) is required" {
		t.Fatalf("missing name: %+v", out)
	}

	// Unknown skill → recoverable error.
	out = tool.Execute(context.Background(), ToolCall{Name: LoadSkill, Input: []byte(`{"name":"nope"}`)}, AllowAllSandbox{}, nil)
	if out.Kind != ToolOutputError || out.Message != "unknown skill 'nope'. Choose one listed in AVAILABLE SKILLS." {
		t.Fatalf("unknown skill: %+v", out)
	}

	// Known skill → success + sticky activation on the shared catalog.
	out = tool.Execute(context.Background(), ToolCall{Name: LoadSkill, Input: []byte(`{"name":"audit"}`)}, AllowAllSandbox{}, nil)
	if out.Kind != ToolOutputSuccess || out.Content != "Loaded skill 'audit' — its full procedure is now in your context. Follow it." {
		t.Fatalf("known skill: %+v", out)
	}
	if active := cat.Active(); len(active) != 1 || active[0] != "audit" {
		t.Fatalf("tool must activate on the shared catalog; got %v", active)
	}
}
