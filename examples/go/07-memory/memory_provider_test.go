package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/storage"
)

func tempPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "memory.md")
}

func sid() storage.SessionID { return storage.SessionID("s1") }

func entry(role, content, ts string) storage.MemoryEntry {
	return storage.NewMemoryEntry(role, content, storage.Timestamp(ts))
}

func contents(entries []storage.MemoryEntry) []string {
	out := make([]string, len(entries))
	for i, e := range entries {
		out[i] = e.Content
	}
	return out
}

func mustAppend(t *testing.T, p *MarkdownMemoryProvider, scope storage.StorageScope, s storage.SessionID, e storage.MemoryEntry) {
	t.Helper()
	if err := p.AppendMemory(context.Background(), scope, s, e); err != nil {
		t.Fatalf("AppendMemory: %v", err)
	}
}

func mustGet(t *testing.T, p *MarkdownMemoryProvider, scope storage.StorageScope, s storage.SessionID, limit int) []storage.MemoryEntry {
	t.Helper()
	got, err := p.GetMemories(context.Background(), scope, s, limit)
	if err != nil {
		t.Fatalf("GetMemories: %v", err)
	}
	return got
}

func TestMissingFileReadsEmpty(t *testing.T) {
	p := NewMarkdownMemoryProvider(tempPath(t))
	got := mustGet(t, p, storage.StorageScopeProject, sid(), 50)
	if len(got) != 0 {
		t.Fatalf("want empty, got %d entries", len(got))
	}
}

func TestAppendThenGetRoundtrips(t *testing.T) {
	path := tempPath(t)
	p := NewMarkdownMemoryProvider(path)
	mustAppend(t, p, storage.StorageScopeProject, sid(),
		entry("assistant", "Postgres is the system of record.", "2026-06-02T10:00:00Z"))

	got := mustGet(t, p, storage.StorageScopeProject, sid(), 50)
	if len(got) != 1 {
		t.Fatalf("want 1 entry, got %d", len(got))
	}
	if got[0].Role != "assistant" {
		t.Fatalf("role: %q", got[0].Role)
	}
	if got[0].Content != "Postgres is the system of record." {
		t.Fatalf("content: %q", got[0].Content)
	}
	if string(got[0].Timestamp) != "2026-06-02T10:00:00Z" {
		t.Fatalf("timestamp: %q", got[0].Timestamp)
	}

	// The artifact is real, readable markdown on disk.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file: %v", err)
	}
	if !strings.Contains(string(raw), "## [project] [s1] 2026-06-02T10:00:00Z — assistant") {
		t.Fatalf("missing header in file:\n%s", raw)
	}
	if !strings.Contains(string(raw), "Postgres is the system of record.") {
		t.Fatalf("missing content in file:\n%s", raw)
	}
}

func TestMultilineContentRoundtrips(t *testing.T) {
	p := NewMarkdownMemoryProvider(tempPath(t))
	mustAppend(t, p, storage.StorageScopeProject, sid(),
		entry("assistant", "line one\nline two\n\nline four", "2026-06-02T10:00:00Z"))

	got := mustGet(t, p, storage.StorageScopeProject, sid(), 50)
	if len(got) != 1 {
		t.Fatalf("want 1 entry, got %d", len(got))
	}
	if got[0].Content != "line one\nline two\n\nline four" {
		t.Fatalf("content: %q", got[0].Content)
	}
}

func TestScopeFilteringIsolatesScopes(t *testing.T) {
	p := NewMarkdownMemoryProvider(tempPath(t))
	mustAppend(t, p, storage.StorageScopeProject, sid(),
		entry("user", "proj", "2026-06-02T10:00:00Z"))
	mustAppend(t, p, storage.StorageScopeUser, sid(),
		entry("user", "usr", "2026-06-02T10:00:01Z"))

	proj := mustGet(t, p, storage.StorageScopeProject, sid(), 50)
	if len(proj) != 1 || proj[0].Content != "proj" {
		t.Fatalf("project scope: %v", contents(proj))
	}
	usr := mustGet(t, p, storage.StorageScopeUser, sid(), 50)
	if len(usr) != 1 || usr[0].Content != "usr" {
		t.Fatalf("user scope: %v", contents(usr))
	}
}

func TestSessionFilteringIsolatesSessions(t *testing.T) {
	p := NewMarkdownMemoryProvider(tempPath(t))
	a := storage.SessionID("alpha")
	b := storage.SessionID("beta")
	mustAppend(t, p, storage.StorageScopeProject, a,
		entry("user", "from-alpha", "2026-06-02T10:00:00Z"))
	mustAppend(t, p, storage.StorageScopeProject, b,
		entry("user", "from-beta", "2026-06-02T10:00:01Z"))

	gotA := mustGet(t, p, storage.StorageScopeProject, a, 50)
	if len(gotA) != 1 || gotA[0].Content != "from-alpha" {
		t.Fatalf("alpha session: %v", contents(gotA))
	}
}

func TestGetReturnsNewestFirst(t *testing.T) {
	p := NewMarkdownMemoryProvider(tempPath(t))
	// Appended out of timestamp order on purpose.
	mustAppend(t, p, storage.StorageScopeProject, sid(),
		entry("user", "middle", "2026-06-02T11:00:00Z"))
	mustAppend(t, p, storage.StorageScopeProject, sid(),
		entry("user", "oldest", "2026-06-02T10:00:00Z"))
	mustAppend(t, p, storage.StorageScopeProject, sid(),
		entry("user", "newest", "2026-06-02T12:00:00Z"))

	got := mustGet(t, p, storage.StorageScopeProject, sid(), 50)
	want := []string{"newest", "middle", "oldest"}
	if fmt.Sprint(contents(got)) != fmt.Sprint(want) {
		t.Fatalf("order: got %v, want %v", contents(got), want)
	}
}

func TestLimitTakesMostRecent(t *testing.T) {
	p := NewMarkdownMemoryProvider(tempPath(t))
	for i := 0; i < 5; i++ {
		mustAppend(t, p, storage.StorageScopeProject, sid(),
			entry("user", fmt.Sprintf("e%d", i), fmt.Sprintf("2026-06-02T10:00:0%dZ", i)))
	}
	got := mustGet(t, p, storage.StorageScopeProject, sid(), 2)
	want := []string{"e4", "e3"}
	if fmt.Sprint(contents(got)) != fmt.Sprint(want) {
		t.Fatalf("limit: got %v, want %v", contents(got), want)
	}
}

func TestToleratesHandEditedFile(t *testing.T) {
	path := tempPath(t)
	// A file a human authored/edited: prose before the first header, an extra
	// heading, blank lines, and a normal entry block.
	hand := "# My Notes\n\nSome rambling prose that is not an entry.\n\n" +
		"## [project] [s1] 2026-06-02T09:00:00Z — user\n\n" +
		"Hand-written fact about Ironwood.\n\n" +
		"## A non-entry heading the human added\n\n" +
		"more prose\n"
	if err := os.WriteFile(path, []byte(hand), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	p := NewMarkdownMemoryProvider(path)

	got := mustGet(t, p, storage.StorageScopeProject, sid(), 50)
	if len(got) != 1 || got[0].Content != "Hand-written fact about Ironwood." {
		t.Fatalf("hand-edited read: %v", contents(got))
	}

	// And we can still append on top of the hand-edited file.
	mustAppend(t, p, storage.StorageScopeProject, sid(),
		entry("assistant", "appended", "2026-06-02T10:00:00Z"))
	got = mustGet(t, p, storage.StorageScopeProject, sid(), 50)
	if len(got) != 2 {
		t.Fatalf("want 2 entries after append, got %d", len(got))
	}
	if got[0].Content != "appended" { // newest-first
		t.Fatalf("newest after append: %q", got[0].Content)
	}
}

func TestComposesIntoStorageProviderMemorySlot(t *testing.T) {
	p := NewMarkdownMemoryProvider(tempPath(t))
	sp := p.IntoStorageProvider()

	// The memory slot is the markdown provider; round-trips through it.
	if err := sp.Memory().AppendMemory(context.Background(), storage.StorageScopeProject, sid(),
		entry("user", "via-seam", "2026-06-02T10:00:00Z")); err != nil {
		t.Fatalf("AppendMemory via seam: %v", err)
	}
	got, err := sp.Memory().GetMemories(context.Background(), storage.StorageScopeProject, sid(), 50)
	if err != nil {
		t.Fatalf("GetMemories via seam: %v", err)
	}
	if len(got) != 1 || got[0].Content != "via-seam" {
		t.Fatalf("via-seam read: %v", contents(got))
	}

	// The other three domains are no-ops: a run read returns nothing.
	v, found, err := sp.Run().Get(context.Background(), sid(), "k")
	if err != nil {
		t.Fatalf("Run().Get: %v", err)
	}
	if found || v != nil {
		t.Fatalf("expected no-op run store, got found=%v v=%s", found, v)
	}
}

func TestMergedReadUnionsUserAndProject(t *testing.T) {
	p := NewMarkdownMemoryProvider(tempPath(t))
	mustAppend(t, p, storage.StorageScopeProject, sid(),
		entry("assistant", "proj", "2026-06-02T10:00:00Z"))
	mustAppend(t, p, storage.StorageScopeUser, sid(),
		entry("assistant", "usr", "2026-06-02T11:00:00Z"))
	mustAppend(t, p, storage.StorageScopeLocal, sid(),
		entry("assistant", "loc", "2026-06-02T12:00:00Z"))

	// GetMemoriesMerged delegates to storage.MergeMemories: User ∪ Project,
	// newest-first, Local excluded.
	got, err := p.GetMemoriesMerged(context.Background(), sid(), 50)
	if err != nil {
		t.Fatalf("GetMemoriesMerged: %v", err)
	}
	want := []string{"usr", "proj"} // newest-first, no local
	if fmt.Sprint(contents(got)) != fmt.Sprint(want) {
		t.Fatalf("merged: got %v, want %v", contents(got), want)
	}
}

// Compile-time check that the example's session id helper returns the pinned id.
func TestPinnedSessionID(t *testing.T) {
	if session() != sporecore.SessionID("project-ironwood") {
		t.Fatalf("pinned session id changed: %q", session())
	}
}
