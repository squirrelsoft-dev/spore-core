// Tests for the #78 scope + workspace-partitioning extension: WorkspaceID
// derivation (incl. fixture replay), per-scope routing + isolation, the merged
// cross-scope read (incl. fixture replay), no-op fallback for unconfigured
// (memory, scope) pairs, scoped recency, Local-defaults-to-no-op, and the
// ToolContext memory seam threaded by the registry.

package storage

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ── R2: WorkspaceID derivation ───────────────────────────────────────────────

func TestWorkspaceIDIsDeterministicAndPure(t *testing.T) {
	a := WorkspaceIDFromCanonicalPath("/Users/sbeardsley/dev/spore-core")
	b := WorkspaceIDFromCanonicalPath("/Users/sbeardsley/dev/spore-core")
	if a != b {
		t.Fatalf("non-deterministic: %q != %q", a, b)
	}
	// Form is {sanitized_basename}-{8hex}.
	const prefix = "spore-core-"
	if got := a.String(); got[:len(prefix)] != prefix || len(got) != len(prefix)+8 {
		t.Fatalf("unexpected form: %q", got)
	}
}

func TestWorkspaceIDRootUsesLiteralRootBasename(t *testing.T) {
	w := WorkspaceIDFromCanonicalPath("/")
	if w.String()[:5] != "root-" {
		t.Fatalf("root basename = %q, want root-…", w)
	}
}

func TestWorkspaceIDSanitizesSpecialCharsAndCollapsesDashes(t *testing.T) {
	w := WorkspaceIDFromCanonicalPath("/Users/me/My Project (v2)!")
	s := w.String()
	const prefix = "my-project-v2-"
	if s[:len(prefix)] != prefix {
		t.Fatalf("sanitize/collapse = %q, want %s…", s, prefix)
	}
	if containsSub(s, "--") {
		t.Fatalf("dashes not collapsed: %q", s)
	}
}

func TestWorkspaceIDIgnoresTrailingSlash(t *testing.T) {
	a := WorkspaceIDFromCanonicalPath("/Users/sbeardsley/dev/spore-core")
	b := WorkspaceIDFromCanonicalPath("/Users/sbeardsley/dev/spore-core/")
	if a != b {
		t.Fatalf("trailing slash changed id: %q != %q", a, b)
	}
}

func TestWorkspaceIDWindowsPathStripsDriveAndNormalizesSep(t *testing.T) {
	w := WorkspaceIDFromCanonicalPath(`C:\Users\dev\spore-core`)
	if w.String()[:len("spore-core-")] != "spore-core-" {
		t.Fatalf("windows basename = %q", w)
	}
	// Distinct hash from the posix path (rest of the canonical string differs).
	if posix := WorkspaceIDFromCanonicalPath("/Users/sbeardsley/dev/spore-core"); w == posix {
		t.Fatalf("windows id collided with posix: %q", w)
	}
}

func TestWorkspaceIDDerivationFixtureReplay(t *testing.T) {
	b, err := os.ReadFile(fixturePath("workspace_id_derivation.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		CanonicalPath       string `json:"canonical_path"`
		ExpectedWorkspaceID string `json:"expected_workspace_id"`
	}
	must(t, json.Unmarshal(b, &cases))
	if len(cases) < 4 {
		t.Fatalf("fixture should carry several rows, got %d", len(cases))
	}
	for _, c := range cases {
		got := WorkspaceIDFromCanonicalPath(c.CanonicalPath)
		if got.String() != c.ExpectedWorkspaceID {
			t.Fatalf("path %q: got %q want %q", c.CanonicalPath, got, c.ExpectedWorkspaceID)
		}
	}
}

// ── R5: scope isolation — User and Project land in different backends ─────────

func TestScopedWritesIsolatedPerScope(t *testing.T) {
	user := NewInMemoryStorageProvider()
	project := NewInMemoryStorageProvider()
	p := NewCompositeStorageProvider().
		Memory(StorageScopeUser, user).
		Memory(StorageScopeProject, project).
		Build()

	must(t, p.Memory().AppendMemory(ctx(), StorageScopeUser, sid("s"), mem("user", "U", "t1")))
	must(t, p.Memory().AppendMemory(ctx(), StorageScopeProject, sid("s"), mem("user", "P", "t1")))

	// Each backend physically holds only its own scope's entry.
	u, _ := user.GetMemories(ctx(), StorageScopeUser, sid("s"), 10)
	if !reflect.DeepEqual(contents(u), []string{"U"}) {
		t.Fatalf("user backend = %v", contents(u))
	}
	pr, _ := project.GetMemories(ctx(), StorageScopeProject, sid("s"), 10)
	if !reflect.DeepEqual(contents(pr), []string{"P"}) {
		t.Fatalf("project backend = %v", contents(pr))
	}

	// Scoped reads through the router return only own-scope entries.
	ru, _ := p.Memory().GetMemories(ctx(), StorageScopeUser, sid("s"), 10)
	if len(ru) != 1 || ru[0].Content != "U" {
		t.Fatalf("router user read = %v", contents(ru))
	}
	rp, _ := p.Memory().GetMemories(ctx(), StorageScopeProject, sid("s"), 10)
	if len(rp) != 1 || rp[0].Content != "P" {
		t.Fatalf("router project read = %v", contents(rp))
	}
}

// ── R6: merged read = User ∪ Project, newest-first by timestamp, no dedup ─────

func TestMergedReadUnionsScopesNewestFirstNoDedup(t *testing.T) {
	p := NewCompositeStorageProvider().
		Memory(StorageScopeUser, NewInMemoryStorageProvider()).
		Memory(StorageScopeProject, NewInMemoryStorageProvider()).
		Build()

	// Identical-content "dup" entry in BOTH scopes (same timestamp) proves no
	// dedup. A Local entry must NOT appear in the merge.
	must(t, p.Memory().AppendMemory(ctx(), StorageScopeUser, sid("s"), mem("user", "u-old", "2026-05-01T00:00:00Z")))
	must(t, p.Memory().AppendMemory(ctx(), StorageScopeUser, sid("s"), mem("user", "dup", "2026-05-03T00:00:00Z")))
	must(t, p.Memory().AppendMemory(ctx(), StorageScopeUser, sid("s"), mem("user", "u-new", "2026-05-05T00:00:00Z")))
	must(t, p.Memory().AppendMemory(ctx(), StorageScopeProject, sid("s"), mem("a", "p-old", "2026-05-02T00:00:00Z")))
	must(t, p.Memory().AppendMemory(ctx(), StorageScopeProject, sid("s"), mem("a", "dup", "2026-05-03T00:00:00Z")))
	must(t, p.Memory().AppendMemory(ctx(), StorageScopeProject, sid("s"), mem("a", "p-new", "2026-05-06T00:00:00Z")))

	merged, err := p.GetMemoriesMerged(ctx(), sid("s"), 10)
	must(t, err)
	want := []string{"p-new", "u-new", "dup", "dup", "p-old", "u-old"}
	if !reflect.DeepEqual(contents(merged), want) {
		t.Fatalf("merged = %v want %v", contents(merged), want)
	}
	// No dedup: the identical-content "dup" entry is present twice.
	dupCount := 0
	for _, e := range merged {
		if e.Content == "dup" {
			dupCount++
		}
	}
	if dupCount != 2 {
		t.Fatalf("dup count = %d want 2", dupCount)
	}
}

func TestMergedReadFixtureReplay(t *testing.T) {
	b, err := os.ReadFile(fixturePath("memory_scoped_merge.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f struct {
		Limit    int           `json:"limit"`
		User     []MemoryEntry `json:"user"`
		Project  []MemoryEntry `json:"project"`
		Local    []MemoryEntry `json:"local"`
		Expected []string      `json:"expected_merged_contents"`
	}
	must(t, json.Unmarshal(b, &f))

	p := NewCompositeStorageProvider().
		Memory(StorageScopeUser, NewInMemoryStorageProvider()).
		Memory(StorageScopeProject, NewInMemoryStorageProvider()).
		Memory(StorageScopeLocal, NewInMemoryStorageProvider()).
		Build()

	for _, e := range f.User {
		must(t, p.Memory().AppendMemory(ctx(), StorageScopeUser, sid("s"), e))
	}
	for _, e := range f.Project {
		must(t, p.Memory().AppendMemory(ctx(), StorageScopeProject, sid("s"), e))
	}
	for _, e := range f.Local {
		must(t, p.Memory().AppendMemory(ctx(), StorageScopeLocal, sid("s"), e))
	}

	merged, err := p.GetMemoriesMerged(ctx(), sid("s"), f.Limit)
	must(t, err)
	if !reflect.DeepEqual(contents(merged), f.Expected) {
		t.Fatalf("merged = %v want %v", contents(merged), f.Expected)
	}
	// Local scope entries are excluded from the merge.
	for _, e := range merged {
		if containsSub(e.Content, "should-not-appear") {
			t.Fatalf("local entry leaked into merge: %q", e.Content)
		}
	}
}

// ── R7: unconfigured (memory, scope) → NoOp returns empty ────────────────────

func TestUnconfiguredMemoryScopeFallsBackToNoOp(t *testing.T) {
	// Only User wired; Project + Local fall back to no-op.
	p := NewCompositeStorageProvider().
		Memory(StorageScopeUser, NewInMemoryStorageProvider()).
		Build()

	// Writes to an unconfigured scope silently no-op (no error).
	must(t, p.Memory().AppendMemory(ctx(), StorageScopeProject, sid("s"), mem("user", "x", "t")))
	// Reads from an unconfigured scope return empty.
	if ms, _ := p.Memory().GetMemories(ctx(), StorageScopeProject, sid("s"), 10); len(ms) != 0 {
		t.Fatalf("unconfigured scope read = %v, want empty", contents(ms))
	}
}

// ── R8: scoped read newest-first recency (append 4, limit=2 → newest two) ─────

func TestScopedReadRecencyNewestFirst(t *testing.T) {
	p := NewCompositeStorageProvider().
		Memory(StorageScopeProject, NewInMemoryStorageProvider()).
		Build()
	for i, c := range []string{"m0", "m1", "m2", "m3"} {
		must(t, p.Memory().AppendMemory(ctx(), StorageScopeProject, sid("s"), mem("user", c, "t"+itoa(i))))
	}
	got, _ := p.Memory().GetMemories(ctx(), StorageScopeProject, sid("s"), 2)
	if !reflect.DeepEqual(contents(got), []string{"m3", "m2"}) {
		t.Fatalf("scoped recency = %v want [m3 m2]", contents(got))
	}
}

// ── R11: Local falls back to NoOp when not wired ─────────────────────────────

func TestLocalScopeDefaultsToNoOp(t *testing.T) {
	// Local intentionally not wired.
	p := NewCompositeStorageProvider().
		Memory(StorageScopeUser, NewInMemoryStorageProvider()).
		Memory(StorageScopeProject, NewInMemoryStorageProvider()).
		Build()
	must(t, p.Memory().AppendMemory(ctx(), StorageScopeLocal, sid("s"), mem("user", "l", "t")))
	if ms, _ := p.Memory().GetMemories(ctx(), StorageScopeLocal, sid("s"), 10); len(ms) != 0 {
		t.Fatalf("local read = %v, want empty", contents(ms))
	}
}

// ── R9: ToolContext exposes the memory store threaded by the registry ─────────

// memProbeTool asserts that the ToolContext it receives carries a live
// MemoryStore (threaded by the registry) and writes one entry through it.
type memProbeTool struct{ t *testing.T }

func (memProbeTool) Name() string                { return "mem_probe" }
func (memProbeTool) IsSubagentTool() bool        { return false }
func (memProbeTool) MayProduceLargeOutput() bool { return false }
func (p memProbeTool) Execute(c context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
	ms, ok := toolCtx.MemoryStore.(MemoryStore)
	if !ok {
		p.t.Fatalf("ToolContext.MemoryStore is not a storage.MemoryStore: %T", toolCtx.MemoryStore)
	}
	if err := ms.AppendMemory(c, StorageScopeProject, toolCtx.SessionID, mem("user", "threaded", "t1")); err != nil {
		p.t.Fatalf("append through threaded store: %v", err)
	}
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "ok"}
}

func TestToolContextExposesThreadedMemoryStore(t *testing.T) {
	// Wire a real memory backend through the registry and prove the seam is live
	// by writing through ToolContext's MemoryStore and reading it back off the
	// SAME store the registry threaded in.
	memStore := NewInMemoryStorageProvider()
	reg := sporecore.NewStandardToolRegistry()
	reg.SetToolContext(sporecore.NewToolContext(sid("ctx-test"), NewInMemoryStorageProvider(), memStore))

	schema := sporecore.RegistryToolSchema{
		Name:       "mem_probe",
		Parameters: json.RawMessage(`{"type":"object"}`),
	}
	must(t, reg.Register(memProbeTool{t: t}, schema))

	_, err := reg.Dispatch(context.Background(),
		sporecore.ToolCall{ID: "1", Name: "mem_probe", Input: json.RawMessage(`{}`)},
		sporecore.AllowAllSandbox{})
	must(t, err)

	got, err := memStore.GetMemories(context.Background(), StorageScopeProject, sid("ctx-test"), 10)
	must(t, err)
	if len(got) != 1 || got[0].Content != "threaded" {
		t.Fatalf("threaded read = %v, want [threaded]", contents(got))
	}
}

// containsSub reports whether sub occurs within s.
func containsSub(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
