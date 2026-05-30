package tools

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/storage"
)

// ── Test helpers ─────────────────────────────────────────────────────────────

func memCall(input map[string]any) sporecore.ToolCall {
	b, _ := json.Marshal(input)
	return sporecore.ToolCall{ID: "c1", Name: MemoryToolName, Input: b}
}

// ctxWith builds a ToolContext over a memory store keyed by session.
func ctxWith(store storage.MemoryStore, session string) *sporecore.ToolContext {
	return sporecore.NewToolContext(sporecore.SessionID(session), nil, store)
}

// inMemoryCtx is a fresh single-backend ToolContext. A single InMemory backend
// is scope-aware, so user and project entries stay isolated.
func inMemoryCtx() *sporecore.ToolContext {
	return ctxWith(storage.NewInMemoryStorageProvider(), "test-session")
}

// scopedCtx builds a ToolContext over a CompositeStorageProvider routing User /
// Project (and optionally Local) to their own InMemory backends — the canonical
// multi-scope wiring whose ScopedMemoryRouter owns the merge.
func scopedCtx(session string, withLocal bool) *sporecore.ToolContext {
	b := storage.NewCompositeStorageProvider().
		Memory(storage.StorageScopeUser, storage.NewInMemoryStorageProvider()).
		Memory(storage.StorageScopeProject, storage.NewInMemoryStorageProvider())
	if withLocal {
		b = b.Memory(storage.StorageScopeLocal, storage.NewInMemoryStorageProvider())
	}
	return sporecore.NewToolContext(sporecore.SessionID(session), nil, b.Build().Memory())
}

func parseEntries(t *testing.T, out sporecore.ToolOutput) []storage.MemoryEntry {
	t.Helper()
	if out.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected Success, got %+v", out)
	}
	var entries []storage.MemoryEntry
	if err := json.Unmarshal([]byte(out.Content), &entries); err != nil {
		t.Fatalf("unmarshal entries: %v", err)
	}
	return entries
}

func parseEntry(t *testing.T, out sporecore.ToolOutput) storage.MemoryEntry {
	t.Helper()
	if out.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected Success, got %+v", out)
	}
	var e storage.MemoryEntry
	if err := json.Unmarshal([]byte(out.Content), &e); err != nil {
		t.Fatalf("unmarshal entry: %v", err)
	}
	return e
}

func entryContents(entries []storage.MemoryEntry) []string {
	out := make([]string, len(entries))
	for i, e := range entries {
		out[i] = e.Content
	}
	return out
}

// failingMemoryStore always fails, proving storage errors map to a recoverable
// tool error (R9). Mirrors the Rust FailingMemoryStore double.
type failingMemoryStore struct{}

func (failingMemoryStore) AppendMemory(context.Context, storage.StorageScope, storage.SessionID, storage.MemoryEntry) error {
	return errors.New("boom")
}
func (failingMemoryStore) GetMemories(context.Context, storage.StorageScope, storage.SessionID, int) ([]storage.MemoryEntry, error) {
	return nil, errors.New("boom")
}
func (f failingMemoryStore) GetMemoriesMerged(ctx context.Context, sessionID storage.SessionID, limit int) ([]storage.MemoryEntry, error) {
	return storage.MergeMemories(ctx, f, sessionID, limit)
}

var _ storage.MemoryStore = failingMemoryStore{}

func run(t *testing.T, ctx *sporecore.ToolContext, input map[string]any) sporecore.ToolOutput {
	t.Helper()
	return NewMemoryTool().Execute(context.Background(), memCall(input), sporecore.AllowAllSandbox{}, ctx)
}

// ── R1 + R2: write→read roundtrip; write returns the serialized entry ─────────

func TestMemoryWriteThenReadRoundtrip(t *testing.T) {
	ctx := inMemoryCtx()
	w := run(t, ctx, map[string]any{"operation": "write", "scope": "user", "role": "user", "content": "hello"})
	written := parseEntry(t, w)
	if written.Role != "user" || written.Content != "hello" {
		t.Fatalf("written entry mismatch: %+v", written)
	}
	if string(written.Metadata) != "{}" {
		t.Fatalf("metadata default = %q, want {}", string(written.Metadata))
	}

	r := run(t, ctx, map[string]any{"operation": "read", "scope": "user"})
	entries := parseEntries(t, r)
	if len(entries) != 1 || entries[0].Content != "hello" {
		t.Fatalf("read = %v", entryContents(entries))
	}
}

// ── R4: metadata preserved verbatim ───────────────────────────────────────────

func TestMemoryWritePreservesMetadata(t *testing.T) {
	ctx := inMemoryCtx()
	w := run(t, ctx, map[string]any{
		"operation": "write", "scope": "project", "role": "assistant", "content": "c",
		"metadata": map[string]any{"k": "v", "n": 3},
	})
	got := parseEntry(t, w)
	var md map[string]any
	if err := json.Unmarshal(got.Metadata, &md); err != nil {
		t.Fatalf("metadata: %v", err)
	}
	if md["k"] != "v" || md["n"].(float64) != 3 {
		t.Fatalf("metadata not preserved: %v", md)
	}
}

// ── R5: non-merged scope isolation ────────────────────────────────────────────

func TestMemoryScopedReadIsolation(t *testing.T) {
	ctx := inMemoryCtx()
	run(t, ctx, map[string]any{"operation": "write", "scope": "user", "role": "user", "content": "u1"})
	run(t, ctx, map[string]any{"operation": "write", "scope": "project", "role": "assistant", "content": "p1"})

	u := parseEntries(t, run(t, ctx, map[string]any{"operation": "read", "scope": "user"}))
	if len(u) != 1 || u[0].Content != "u1" {
		t.Fatalf("user read = %v", entryContents(u))
	}
	p := parseEntries(t, run(t, ctx, map[string]any{"operation": "read", "scope": "project"}))
	if len(p) != 1 || p[0].Content != "p1" {
		t.Fatalf("project read = %v", entryContents(p))
	}
}

// ── R6: merged read drives the shared merge fixture ───────────────────────────

func TestMemoryMergedReadFixtureReplay(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(fixturesStorageDir(t), "memory_scoped_merge.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f struct {
		Limit    int                   `json:"limit"`
		User     []storage.MemoryEntry `json:"user"`
		Project  []storage.MemoryEntry `json:"project"`
		Local    []storage.MemoryEntry `json:"local"`
		Expected []string              `json:"expected_merged_contents"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	ctx := scopedCtx("s", true)
	store := ctx.MemoryStore.(storage.MemoryStore)
	seed := func(scope storage.StorageScope, es []storage.MemoryEntry) {
		for _, e := range es {
			if err := store.AppendMemory(context.Background(), scope, ctx.SessionID, e); err != nil {
				t.Fatalf("seed: %v", err)
			}
		}
	}
	seed(storage.StorageScopeUser, f.User)
	seed(storage.StorageScopeProject, f.Project)
	seed(storage.StorageScopeLocal, f.Local)

	out := run(t, ctx, map[string]any{"operation": "read", "scope": "user", "merged": true, "limit": f.Limit})
	contents := entryContents(parseEntries(t, out))
	if len(contents) != len(f.Expected) {
		t.Fatalf("merged = %v want %v", contents, f.Expected)
	}
	for i := range contents {
		if contents[i] != f.Expected[i] {
			t.Fatalf("merged = %v want %v", contents, f.Expected)
		}
	}
	dup := 0
	for _, c := range contents {
		if c == "dup" {
			dup++
		}
		if c == "l-should-not-appear" {
			t.Fatalf("local leaked into merge")
		}
	}
	if dup != 2 {
		t.Fatalf("expected both dup entries, got %d", dup)
	}
}

// ── merged read respects limit ────────────────────────────────────────────────

func TestMemoryMergedReadRespectsLimit(t *testing.T) {
	ctx := scopedCtx("s", false)
	store := ctx.MemoryStore.(storage.MemoryStore)
	for i, ts := range []string{"2026-05-01T00:00:00Z", "2026-05-02T00:00:00Z", "2026-05-03T00:00:00Z"} {
		e := storage.NewMemoryEntry("user", string(rune('0'+i)), storage.Timestamp(ts))
		e.Content = "u" + string(rune('0'+i))
		if err := store.AppendMemory(context.Background(), storage.StorageScopeUser, ctx.SessionID, e); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	out := run(t, ctx, map[string]any{"operation": "read", "scope": "user", "merged": true, "limit": 2})
	entries := parseEntries(t, out)
	if len(entries) != 2 || entries[0].Content != "u2" || entries[1].Content != "u1" {
		t.Fatalf("limited merged read = %v", entryContents(entries))
	}
}

// ── R7: Local rejected on write — exact message, nothing written ──────────────

func TestMemoryLocalRejectedOnWriteWritesNothing(t *testing.T) {
	ctx := scopedCtx("s", true)
	out := run(t, ctx, map[string]any{"operation": "write", "scope": "local", "role": "user", "content": "x"})
	if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", out)
	}
	if out.Message != LocalScopeRejectedMessage {
		t.Fatalf("message = %q want %q", out.Message, LocalScopeRejectedMessage)
	}
	store := ctx.MemoryStore.(storage.MemoryStore)
	for _, scope := range []storage.StorageScope{storage.StorageScopeUser, storage.StorageScopeProject, storage.StorageScopeLocal} {
		got, _ := store.GetMemories(context.Background(), scope, ctx.SessionID, 50)
		if len(got) != 0 {
			t.Fatalf("scope %q should be empty, got %v", scope, entryContents(got))
		}
	}
}

// ── R7: Local rejected on read — exact message ────────────────────────────────

func TestMemoryLocalRejectedOnRead(t *testing.T) {
	ctx := inMemoryCtx()
	out := run(t, ctx, map[string]any{"operation": "read", "scope": "local"})
	if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", out)
	}
	if out.Message != LocalScopeRejectedMessage {
		t.Fatalf("message = %q want %q", out.Message, LocalScopeRejectedMessage)
	}
}

// ── Session isolation: two sessions over the SAME store ───────────────────────

func TestMemoryKeyedBySessionID(t *testing.T) {
	store := storage.NewInMemoryStorageProvider()
	ctxA := ctxWith(store, "session-a")
	ctxB := ctxWith(store, "session-b")
	run(t, ctxA, map[string]any{"operation": "write", "scope": "user", "role": "user", "content": "a1"})
	run(t, ctxB, map[string]any{"operation": "write", "scope": "user", "role": "user", "content": "b1"})

	a, _ := store.GetMemories(context.Background(), storage.StorageScopeUser, "session-a", 50)
	b, _ := store.GetMemories(context.Background(), storage.StorageScopeUser, "session-b", 50)
	if len(a) != 1 || a[0].Content != "a1" {
		t.Fatalf("session-a = %v", entryContents(a))
	}
	if len(b) != 1 || b[0].Content != "b1" {
		t.Fatalf("session-b = %v", entryContents(b))
	}
}

// ── R8: bad params → recoverable error ────────────────────────────────────────

func TestMemoryBadParamsRecoverable(t *testing.T) {
	ctx := inMemoryCtx()
	// Unknown operation (scope present so scope check passes; op check fails).
	out := run(t, ctx, map[string]any{"operation": "nope", "scope": "user"})
	if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
		t.Fatalf("unknown op: %+v", out)
	}
	// Missing required field on write (no content).
	out = run(t, ctx, map[string]any{"operation": "write", "scope": "user", "role": "user"})
	if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
		t.Fatalf("missing content: %+v", out)
	}
	// Missing scope.
	out = run(t, ctx, map[string]any{"operation": "read"})
	if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
		t.Fatalf("missing scope: %+v", out)
	}
	// Malformed JSON input.
	bad := sporecore.ToolCall{ID: "c", Name: MemoryToolName, Input: json.RawMessage(`{`)}
	out = NewMemoryTool().Execute(context.Background(), bad, sporecore.AllowAllSandbox{}, ctx)
	if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
		t.Fatalf("malformed: %+v", out)
	}
}

// ── R9: storage failure → recoverable error (both write and read) ─────────────

func TestMemoryStorageFailureRecoverable(t *testing.T) {
	ctx := ctxWith(failingMemoryStore{}, "test-session")
	w := run(t, ctx, map[string]any{"operation": "write", "scope": "user", "role": "user", "content": "x"})
	if w.Kind != sporecore.ToolOutputError || !w.Recoverable {
		t.Fatalf("write failure: %+v", w)
	}
	r := run(t, ctx, map[string]any{"operation": "read", "scope": "user"})
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("read failure: %+v", r)
	}
	m := run(t, ctx, map[string]any{"operation": "read", "scope": "user", "merged": true})
	if m.Kind != sporecore.ToolOutputError || !m.Recoverable {
		t.Fatalf("merged read failure: %+v", m)
	}
}

// ── R10: read does not write ──────────────────────────────────────────────────

func TestMemoryReadDoesNotWrite(t *testing.T) {
	ctx := inMemoryCtx()
	r := run(t, ctx, map[string]any{"operation": "read", "scope": "user"})
	if len(parseEntries(t, r)) != 0 {
		t.Fatalf("fresh read should be empty")
	}
	store := ctx.MemoryStore.(storage.MemoryStore)
	got, _ := store.GetMemories(context.Background(), storage.StorageScopeUser, ctx.SessionID, 50)
	if len(got) != 0 {
		t.Fatalf("read mutated the store: %v", entryContents(got))
	}
}

// ── Schema is NOT read_only (decision E) ──────────────────────────────────────

func TestMemorySchemaNotReadOnly(t *testing.T) {
	s := NewMemoryTool().Schema()
	if s.Annotations.ReadOnly || s.Annotations.Destructive || s.Annotations.OpenWorld {
		t.Fatalf("memory annotations must all be false: %+v", s.Annotations)
	}
	if s.Name != "memory" {
		t.Fatalf("name = %q", s.Name)
	}
}

// ── Default read limit is 50 (decision B) ─────────────────────────────────────

func TestMemoryReadDefaultLimit50(t *testing.T) {
	ctx := inMemoryCtx()
	for i := 0; i < 60; i++ {
		run(t, ctx, map[string]any{"operation": "write", "scope": "user", "role": "user", "content": "m" + itoaMem(i)})
	}
	r := run(t, ctx, map[string]any{"operation": "read", "scope": "user"})
	if n := len(parseEntries(t, r)); n != 50 {
		t.Fatalf("default-limit read returned %d, want 50", n)
	}
}

// ── Fixture replay — fixtures/tools/memory.json ───────────────────────────────

type memOpStep struct {
	Input    map[string]any `json:"input"`
	Expected struct {
		OK        bool      `json:"ok"`
		Contents  *[]string `json:"contents"`
		Unordered bool      `json:"unordered"`
		Error     *string   `json:"error"`
	} `json:"expected"`
}

type memOpScenario struct {
	Name  string      `json:"name"`
	Steps []memOpStep `json:"steps"`
}

func TestMemoryFixtureReplayOperations(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(fixturesDir(t), "memory.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var scenarios []memOpScenario
	if err := json.Unmarshal(data, &scenarios); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(scenarios) == 0 {
		t.Fatal("expected >= 1 scenario")
	}

	for _, sc := range scenarios {
		// Fresh isolated scope-routing provider per scenario.
		ctx := scopedCtx("fx", false)
		for i, step := range sc.Steps {
			out := run(t, ctx, step.Input)
			if step.Expected.OK {
				if out.Kind != sporecore.ToolOutputSuccess {
					t.Fatalf("%s step %d: expected ok, got %+v", sc.Name, i, out)
				}
				if step.Expected.Contents == nil {
					continue
				}
				got := entryContents(parseEntries(t, out))
				want := append([]string(nil), *step.Expected.Contents...)
				if step.Expected.Unordered {
					// Honor the `unordered` flag: tool-stamped timestamps can collide
					// within the same wall-clock second, so compare as a multiset.
					sort.Strings(got)
					sort.Strings(want)
				}
				if len(got) != len(want) {
					t.Fatalf("%s step %d: got %v want %v", sc.Name, i, got, want)
				}
				for j := range got {
					if got[j] != want[j] {
						t.Fatalf("%s step %d: got %v want %v", sc.Name, i, got, want)
					}
				}
			} else {
				if out.Kind != sporecore.ToolOutputError {
					t.Fatalf("%s step %d: expected error, got %+v", sc.Name, i, out)
				}
				if !out.Recoverable {
					t.Fatalf("%s step %d: errors must be recoverable", sc.Name, i)
				}
				if step.Expected.Error != nil && out.Message != *step.Expected.Error {
					t.Fatalf("%s step %d: message %q want %q", sc.Name, i, out.Message, *step.Expected.Error)
				}
			}
		}
	}
}

// fixturesStorageDir locates /fixtures/storage relative to this test file.
func fixturesStorageDir(t *testing.T) string {
	t.Helper()
	_, here, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(here), "..", "..", "..", "fixtures", "storage")
}

func itoaMem(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}
