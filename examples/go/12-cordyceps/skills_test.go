package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/storage"
)

// testManifest is a two-skill manifest used by the injection test.
func testManifest() []skillEntry {
	return []skillEntry{
		{
			Name:        "audit",
			Description: "Audit one Rust module for real, actionable defects.",
			Body:        "GREP-FIRST PROCEDURE BODY",
		},
		{
			Name:        "other",
			Description: "Some other skill.",
			Body:        "OTHER BODY",
		},
	}
}

// textOf joins all text-content messages in an assembled context.
func textOf(c sporecore.Context) string {
	var parts []string
	for _, m := range c.Messages {
		if m.Content.Type == sporecore.ContentTypeText {
			parts = append(parts, m.Content.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// newPassThroughInner builds a real StandardCompactionAdapter, whose Assemble is
// a pass-through of session.Messages — exactly the live-loop behaviour the
// injecting manager wraps.
func newPassThroughInner() *contextmgr.StandardCompactionAdapter {
	return contextmgr.NewStandardCompactionAdapter(
		contextmgr.NewStandardContextManager(
			ollama.New("test-model"),
			contextmgr.NullCacheProvider{},
			contextmgr.DefaultCompactionConfig(),
		),
	)
}

// TestManifestAlwaysInjectedBodiesOnlyWhenActive mirrors the Rust reference's
// manifest_always_injected_bodies_only_when_active: the manifest is injected
// every turn, but a skill's body appears only after it is activated in the run
// store under "active_skills".
func TestManifestAlwaysInjectedBodiesOnlyWhenActive(t *testing.T) {
	ctx := context.Background()
	store := storage.SingleStorageProvider(storage.NewInMemoryStorageProvider())
	cm := NewSkillInjectingContextManager(newPassThroughInner(), store.Run(), testManifest())

	session := sporecore.SessionState{}
	task := sporecore.NewTask("audit a module", sporecore.NewSessionID(), sporecore.ReActStrategy(8))

	// No active skills yet: manifest present, NO body.
	body := textOf(cm.Assemble(ctx, &session, &task))
	if !strings.Contains(body, "AVAILABLE SKILLS") {
		t.Fatalf("manifest must be injected; got:\n%s", body)
	}
	if !strings.Contains(body, "audit: Audit one Rust module") {
		t.Fatalf("manifest must list audit; got:\n%s", body)
	}
	if !strings.Contains(body, "other: Some other skill") {
		t.Fatalf("manifest must list other; got:\n%s", body)
	}
	if strings.Contains(body, "GREP-FIRST PROCEDURE BODY") {
		t.Fatalf("inactive skill body must NOT be injected; got:\n%s", body)
	}

	// Activate `audit` (as the load_skill tool does) → body appears next turn.
	active, _ := json.Marshal([]string{"audit"})
	if err := store.Run().Put(ctx, task.SessionID, activeSkillsKey, active); err != nil {
		t.Fatalf("failed to seed active skills: %v", err)
	}

	body = textOf(cm.Assemble(ctx, &session, &task))
	if !strings.Contains(body, "AVAILABLE SKILLS") {
		t.Fatalf("manifest still present; got:\n%s", body)
	}
	if !strings.Contains(body, "ACTIVE SKILL — audit") {
		t.Fatalf("active skill body must be injected; got:\n%s", body)
	}
	if !strings.Contains(body, "GREP-FIRST PROCEDURE BODY") {
		t.Fatalf("active audit body must be present; got:\n%s", body)
	}
	if strings.Contains(body, "OTHER BODY") {
		t.Fatalf("only the active skill's body is injected; got:\n%s", body)
	}
}

// TestParsesFrontmatterNameAndDescription checks the minimal frontmatter parser.
func TestParsesFrontmatterNameAndDescription(t *testing.T) {
	doc := "---\nname: audit\ndescription: Audit one module.\n---\n\n# Body\nprocedure"
	entry, ok := parseSkillDoc(doc)
	if !ok {
		t.Fatal("expected the doc to parse")
	}
	if entry.Name != "audit" {
		t.Fatalf("name = %q, want audit", entry.Name)
	}
	if entry.Description != "Audit one module." {
		t.Fatalf("description = %q, want %q", entry.Description, "Audit one module.")
	}
	if !strings.Contains(entry.Body, "procedure") {
		t.Fatalf("body missing procedure; got %q", entry.Body)
	}
}

// TestRejectsMissingNameOrEmptyBody checks the parser's reject paths.
func TestRejectsMissingNameOrEmptyBody(t *testing.T) {
	if _, ok := parseSkillDoc("---\ndescription: x\n---\nbody"); ok {
		t.Fatal("expected reject: missing name")
	}
	if _, ok := parseSkillDoc("---\nname: audit\n---\n"); ok {
		t.Fatal("expected reject: empty body")
	}
}

// TestBundledAuditSkillRegisters confirms the embedded audit skill parses and
// ends up in a bootstrapped catalog (the example must be self-contained).
func TestBundledAuditSkillRegisters(t *testing.T) {
	catalog := BootstrapCatalog(context.Background(), t.TempDir(), bundledAuditSkill)
	found := false
	for _, n := range catalog.Names() {
		if n == "audit" {
			found = true
		}
	}
	if !found {
		t.Fatalf("bundled audit skill not in catalog; names = %v", catalog.Names())
	}
}
