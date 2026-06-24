package main

import (
	"os"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// SKILL.md parse + catalog behaviour now live in the core module
// (go/spore-core/skills_test.go). This test only asserts the example stays
// self-contained: the bundled skills/audit/SKILL.md is discovered by the
// harness-native SkillCatalog (#115 / SC-26) — the migration that deleted the
// example-side SkillInjectingContextManager shim must not lose the audit skill.
func TestBundledAuditSkillDiscovered(t *testing.T) {
	repoRoot, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	catalog := sporecore.Discover([]string{repoRoot + "/skills"}, repoRoot)
	found := false
	for _, n := range catalog.Names() {
		if n == "audit" {
			found = true
		}
	}
	if !found {
		t.Fatalf("bundled audit skill not discovered; names = %v", catalog.Names())
	}
}
