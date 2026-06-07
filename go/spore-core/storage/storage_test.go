package storage

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ── helpers ──────────────────────────────────────────────────────────────────

func sid(s string) SessionID { return SessionID(s) }
func ts(s string) Timestamp  { return Timestamp(s) }

// paused builds a minimal valid PausedState for roundtrip tests.
func paused(session string) *PausedState {
	return &sporecore.PausedState{
		SessionID:    sid(session),
		TaskID:       "task1",
		TurnNumber:   3,
		HumanRequest: &sporecore.HumanRequest{Kind: sporecore.HumanReqToolApproval, RiskLevel: sporecore.RiskLow},
		Task:         sporecore.NewTask("do the thing", sid(session), sporecore.ReActStrategy(1)),
	}
}

func mem(role, content, t string) MemoryEntry { return NewMemoryEntry(role, content, ts(t)) }

func fixturePath(name string) string {
	wd, _ := os.Getwd()
	return filepath.Join(wd, "..", "..", "..", "fixtures", "storage", name)
}

func raw(s string) json.RawMessage { return json.RawMessage(s) }

func ctx() context.Context { return context.Background() }

// ── OTLP endpoint parsing (the most important cross-language rule) ───────────

func TestOTLPParseTable(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"a", []string{"a"}},
		{"a,b,c", []string{"a", "b", "c"}},
		{" a , b ", []string{"a", "b"}},
		{"a,,b,", []string{"a", "b"}},
		{"", []string{}},
		{"  ", []string{}},
	}
	for _, c := range cases {
		got := ParseOTLPEndpoints(c.in)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("ParseOTLPEndpoints(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestOTLPParseFixtureReplay(t *testing.T) {
	b, err := os.ReadFile(fixturePath("otlp_endpoints_parse.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Input    string   `json:"input"`
		Expected []string `json:"expected"`
	}
	if err := json.Unmarshal(b, &cases); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	for _, c := range cases {
		got := ParseOTLPEndpoints(c.Input)
		want := c.Expected
		if want == nil {
			want = []string{}
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("ParseOTLPEndpoints(%q) = %v, want %v", c.Input, got, want)
		}
	}
}

// ── No-op fallback ───────────────────────────────────────────────────────────

func TestNoOpReadsEmptyWritesOK(t *testing.T) {
	p := NoOpStorageProvider{}
	if _, found, err := p.GetSession(ctx(), sid("s")); found || err != nil {
		t.Fatalf("GetSession found=%v err=%v", found, err)
	}
	if list, err := p.ListSessions(ctx()); err != nil || len(list) != 0 {
		t.Fatalf("ListSessions=%v err=%v", list, err)
	}
	if err := p.PutSession(ctx(), sid("s"), paused("s")); err != nil {
		t.Fatalf("PutSession err=%v", err)
	}
	if ms, err := p.GetMemories(ctx(), StorageScopeProject, sid("s"), 10); err != nil || len(ms) != 0 {
		t.Fatalf("GetMemories=%v err=%v", ms, err)
	}
	if err := p.AppendMemory(ctx(), StorageScopeProject, sid("s"), mem("user", "hi", "t")); err != nil {
		t.Fatalf("AppendMemory err=%v", err)
	}
	if _, found, err := p.Get(ctx(), sid("s"), "k"); found || err != nil {
		t.Fatalf("Get found=%v err=%v", found, err)
	}
	if err := p.Put(ctx(), sid("s"), "k", raw("1")); err != nil {
		t.Fatalf("Put err=%v", err)
	}
	if keys, err := p.ListKeys(ctx(), sid("s")); err != nil || len(keys) != 0 {
		t.Fatalf("ListKeys=%v err=%v", keys, err)
	}
	if spans, err := p.GetSpans(ctx(), sid("s")); err != nil || len(spans) != 0 {
		t.Fatalf("GetSpans=%v err=%v", spans, err)
	}
	if err := p.AppendSpan(ctx(), sid("s"), raw("{}")); err != nil {
		t.Fatalf("AppendSpan err=%v", err)
	}
}

func TestDefaultStorageProviderIsNoOp(t *testing.T) {
	p := NoOp()
	// All four slots present and reachable; reads are empty.
	if _, found, _ := p.Session().GetSession(ctx(), sid("s")); found {
		t.Fatal("session slot not no-op")
	}
	if ms, _ := p.Memory().GetMemories(ctx(), StorageScopeProject, sid("s"), 1); len(ms) != 0 {
		t.Fatal("memory slot not no-op")
	}
	if _, found, _ := p.Run().Get(ctx(), sid("s"), "k"); found {
		t.Fatal("run slot not no-op")
	}
	if spans, _ := p.Observability().GetSpans(ctx(), sid("s")); len(spans) != 0 {
		t.Fatal("observability slot not no-op")
	}
}

// ── Single-provider-fills-all-slots ──────────────────────────────────────────

func TestSingleFillsAllSlots(t *testing.T) {
	backend := NewInMemoryStorageProvider()
	p := SingleStorageProvider(backend)

	must(t, p.Session().PutSession(ctx(), sid("s"), paused("s")))
	must(t, p.Memory().AppendMemory(ctx(), StorageScopeProject, sid("s"), mem("user", "hi", "t1")))
	must(t, p.Run().Put(ctx(), sid("s"), "plan", raw(`{"x":1}`)))
	must(t, p.Observability().AppendSpan(ctx(), sid("s"), raw(`{"kind":"turn"}`)))

	if _, found, _ := p.Session().GetSession(ctx(), sid("s")); !found {
		t.Fatal("session not shared")
	}
	if ms, _ := p.Memory().GetMemories(ctx(), StorageScopeProject, sid("s"), 10); len(ms) != 1 {
		t.Fatalf("memory len=%d", len(ms))
	}
	if v, found, _ := p.Run().Get(ctx(), sid("s"), "plan"); !found || string(v) != `{"x":1}` {
		t.Fatalf("run get=%s found=%v", v, found)
	}
	if spans, _ := p.Observability().GetSpans(ctx(), sid("s")); len(spans) != 1 {
		t.Fatalf("spans len=%d", len(spans))
	}
}

// ── Composite per-domain routing + no-op fallback ────────────────────────────

func TestCompositeRoutesPerDomainAndFallsBackToNoOp(t *testing.T) {
	runBackend := NewInMemoryStorageProvider()
	// Only the run domain is configured; the others fall back to no-op.
	p := NewCompositeStorageProvider().Run(runBackend).Build()

	must(t, p.Run().Put(ctx(), sid("s"), "k", raw(`"v"`)))
	if v, found, _ := p.Run().Get(ctx(), sid("s"), "k"); !found || string(v) != `"v"` {
		t.Fatalf("run get=%s found=%v", v, found)
	}

	// Unconfigured domains silently no-op.
	must(t, p.Session().PutSession(ctx(), sid("s"), paused("s")))
	if _, found, _ := p.Session().GetSession(ctx(), sid("s")); found {
		t.Fatal("session should be no-op")
	}
	if ms, _ := p.Memory().GetMemories(ctx(), StorageScopeProject, sid("s"), 5); len(ms) != 0 {
		t.Fatal("memory should be no-op")
	}
	if spans, _ := p.Observability().GetSpans(ctx(), sid("s")); len(spans) != 0 {
		t.Fatal("observability should be no-op")
	}
}

// ── In-memory: session-store roundtrip + list + delete ───────────────────────

func TestInMemorySessionRoundtripListDelete(t *testing.T) {
	p := NewInMemoryStorageProvider()
	must(t, p.PutSession(ctx(), sid("b"), paused("b")))
	must(t, p.PutSession(ctx(), sid("a"), paused("a")))
	got, found, err := p.GetSession(ctx(), sid("a"))
	if err != nil || !found {
		t.Fatalf("get a: found=%v err=%v", found, err)
	}
	if got.SessionID != sid("a") {
		t.Fatalf("got session id %q", got.SessionID)
	}
	if list, _ := p.ListSessions(ctx()); !reflect.DeepEqual(list, []SessionID{sid("a"), sid("b")}) {
		t.Fatalf("list=%v", list)
	}
	must(t, p.DeleteSession(ctx(), sid("a")))
	if _, found, _ := p.GetSession(ctx(), sid("a")); found {
		t.Fatal("a not deleted")
	}
	if list, _ := p.ListSessions(ctx()); !reflect.DeepEqual(list, []SessionID{sid("b")}) {
		t.Fatalf("list after delete=%v", list)
	}
}

// ── In-memory: run-store opaque-json roundtrip + list_keys + delete ──────────

func TestInMemoryRunRoundtripListDelete(t *testing.T) {
	p := NewInMemoryStorageProvider()
	blob := raw(`{"nested":{"arr":[1,2,3],"s":"x"},"n":4.5}`)
	must(t, p.Put(ctx(), sid("s"), "plan", blob))
	must(t, p.Put(ctx(), sid("s"), "tasks", raw(`[1,2]`)))
	if v, found, _ := p.Get(ctx(), sid("s"), "plan"); !found || string(v) != string(blob) {
		t.Fatalf("get plan=%s found=%v", v, found)
	}
	if keys, _ := p.ListKeys(ctx(), sid("s")); !reflect.DeepEqual(keys, []string{"plan", "tasks"}) {
		t.Fatalf("keys=%v", keys)
	}
	must(t, p.Delete(ctx(), sid("s"), "plan"))
	if _, found, _ := p.Get(ctx(), sid("s"), "plan"); found {
		t.Fatal("plan not deleted")
	}
	if keys, _ := p.ListKeys(ctx(), sid("s")); !reflect.DeepEqual(keys, []string{"tasks"}) {
		t.Fatalf("keys after delete=%v", keys)
	}
}

// ── In-memory: memory append ordering + recency limit ────────────────────────

func TestInMemoryMemoryRecencyAndLimit(t *testing.T) {
	p := NewInMemoryStorageProvider()
	for i, c := range []string{"m0", "m1", "m2", "m3"} {
		must(t, p.AppendMemory(ctx(), StorageScopeProject, sid("s"), mem("user", c, "t"+itoa(i))))
	}
	got, _ := p.GetMemories(ctx(), StorageScopeProject, sid("s"), 2)
	if len(got) != 2 || got[0].Content != "m3" || got[1].Content != "m2" {
		t.Fatalf("recency 2 = %+v", contents(got))
	}
	all, _ := p.GetMemories(ctx(), StorageScopeProject, sid("s"), 99)
	if !reflect.DeepEqual(contents(all), []string{"m3", "m2", "m1", "m0"}) {
		t.Fatalf("all newest-first = %v", contents(all))
	}
}

func TestInMemorySpansAppendOrdering(t *testing.T) {
	p := NewInMemoryStorageProvider()
	must(t, p.AppendSpan(ctx(), sid("s"), raw(`{"n":0}`)))
	must(t, p.AppendSpan(ctx(), sid("s"), raw(`{"n":1}`)))
	spans, _ := p.GetSpans(ctx(), sid("s"))
	if len(spans) != 2 || string(spans[0]) != `{"n":0}` || string(spans[1]) != `{"n":1}` {
		t.Fatalf("spans=%v", spans)
	}
}

// ── FileSystem: atomic write (no leftover .tmp) ──────────────────────────────

func TestFSAtomicWriteLeavesNoTmp(t *testing.T) {
	tmp := t.TempDir()
	p := NewFileSystemStorageProvider(tmp)
	must(t, p.PutSession(ctx(), sid("s"), paused("s")))
	must(t, p.Put(ctx(), sid("s"), "k", raw(`{"a":1}`)))

	var leftovers []string
	_ = filepath.Walk(tmp, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && filepath.Ext(path) == ".tmp" {
			leftovers = append(leftovers, path)
		}
		return nil
	})
	if len(leftovers) != 0 {
		t.Fatalf("leftover .tmp files: %v", leftovers)
	}
	if !fsExists(filepath.Join(tmp, "sessions", "s", "state.json")) {
		t.Fatal("state.json missing")
	}
	if !fsExists(filepath.Join(tmp, "sessions", "s", "run", "k.json")) {
		t.Fatal("run/k.json missing")
	}
}

func TestFSSessionRoundtripListDelete(t *testing.T) {
	p := NewFileSystemStorageProvider(t.TempDir())
	must(t, p.PutSession(ctx(), sid("a"), paused("a")))
	must(t, p.PutSession(ctx(), sid("b"), paused("b")))
	got, found, err := p.GetSession(ctx(), sid("a"))
	if err != nil || !found || got.TurnNumber != 3 {
		t.Fatalf("get a: found=%v err=%v state=%+v", found, err, got)
	}
	if list, _ := p.ListSessions(ctx()); !reflect.DeepEqual(list, []SessionID{sid("a"), sid("b")}) {
		t.Fatalf("list=%v", list)
	}
	must(t, p.DeleteSession(ctx(), sid("a")))
	if _, found, _ := p.GetSession(ctx(), sid("a")); found {
		t.Fatal("a not deleted")
	}
	// delete of missing is OK.
	must(t, p.DeleteSession(ctx(), sid("missing")))
}

func TestFSRunRoundtripListDelete(t *testing.T) {
	p := NewFileSystemStorageProvider(t.TempDir())
	blob := raw(`{"deep":[true,null,"x"]}`)
	must(t, p.Put(ctx(), sid("s"), "plan", blob))
	must(t, p.Put(ctx(), sid("s"), "tasks", raw("7")))
	if v, found, _ := p.Get(ctx(), sid("s"), "plan"); !found || string(v) != string(blob) {
		t.Fatalf("get plan=%s found=%v", v, found)
	}
	if keys, _ := p.ListKeys(ctx(), sid("s")); !reflect.DeepEqual(keys, []string{"plan", "tasks"}) {
		t.Fatalf("keys=%v", keys)
	}
	must(t, p.Delete(ctx(), sid("s"), "plan"))
	if _, found, _ := p.Get(ctx(), sid("s"), "plan"); found {
		t.Fatal("plan not deleted")
	}
	if _, found, _ := p.Get(ctx(), sid("missing"), "x"); found {
		t.Fatal("missing should be not-found")
	}
}

func TestFSMemoryAppendRecencyAndJSONLPath(t *testing.T) {
	tmp := t.TempDir()
	p := NewFileSystemStorageProvider(tmp)
	for i, c := range []string{"a", "b", "c"} {
		must(t, p.AppendMemory(ctx(), StorageScopeProject, sid("s"), mem("user", c, "t"+itoa(i))))
	}
	if !fsExists(filepath.Join(tmp, "sessions", "s", "memory.jsonl")) {
		t.Fatal("memory.jsonl missing")
	}
	got, _ := p.GetMemories(ctx(), StorageScopeProject, sid("s"), 2)
	if len(got) != 2 || got[0].Content != "c" || got[1].Content != "b" {
		t.Fatalf("recency = %v", contents(got))
	}
	// metadata defaults to {}.
	if string(got[0].Metadata) != "{}" {
		t.Fatalf("metadata = %s", got[0].Metadata)
	}
}

func TestFSSpansAppendAndFlushMarker(t *testing.T) {
	tmp := t.TempDir()
	p := NewFileSystemStorageProvider(tmp)
	must(t, p.AppendSpan(ctx(), sid("s"), raw(`{"n":0}`)))
	must(t, p.AppendSpan(ctx(), sid("s"), raw(`{"n":1}`)))
	if !fsExists(filepath.Join(tmp, "sessions", "s", "trace.jsonl")) {
		t.Fatal("trace.jsonl missing")
	}
	spans, _ := p.GetSpans(ctx(), sid("s"))
	if len(spans) != 2 || string(spans[0]) != `{"n":0}` || string(spans[1]) != `{"n":1}` {
		t.Fatalf("spans=%v", spans)
	}
	must(t, p.FlushSession(ctx(), sid("s")))
	if !fsExists(filepath.Join(tmp, "sessions", "s", ".flushed")) {
		t.Fatal(".flushed marker missing")
	}
}

// ── MemoryEntry default metadata ─────────────────────────────────────────────

func TestMemoryEntryMetadataDefaultsToEmptyObject(t *testing.T) {
	var e MemoryEntry
	if err := json.Unmarshal([]byte(`{"role":"user","content":"hi","timestamp":"2026-05-28T00:00:00Z"}`), &e); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(e.Metadata) != "{}" {
		t.Fatalf("metadata = %s, want {}", e.Metadata)
	}
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	must(t, json.Unmarshal(b, &m))
	if string(m["role"]) != `"user"` || string(m["content"]) != `"hi"` || string(m["metadata"]) != "{}" {
		t.Fatalf("roundtrip = %s", b)
	}
}

// ── Fixture replay: run_store_values + memory_entries ────────────────────────

func TestRunStoreValuesFixtureReplay(t *testing.T) {
	b, err := os.ReadFile(fixturePath("run_store_values.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Key   string          `json:"key"`
		Value json.RawMessage `json:"value"`
	}
	must(t, json.Unmarshal(b, &cases))

	p := NewInMemoryStorageProvider()
	fsp := NewFileSystemStorageProvider(t.TempDir())
	for _, c := range cases {
		// canonicalize the expected value through a marshal round-trip so the
		// comparison is whitespace-insensitive (both stores return canonical
		// json).
		want := canon(t, c.Value)

		must(t, p.Put(ctx(), sid("s"), c.Key, c.Value))
		if v, found, _ := p.Get(ctx(), sid("s"), c.Key); !found || canon(t, v) != want {
			t.Fatalf("in-memory roundtrip mismatch for %q: got %s want %s", c.Key, v, want)
		}
		must(t, fsp.Put(ctx(), sid("s"), c.Key, c.Value))
		if v, found, _ := fsp.Get(ctx(), sid("s"), c.Key); !found || canon(t, v) != want {
			t.Fatalf("fs roundtrip mismatch for %q: got %s want %s", c.Key, v, want)
		}
	}
}

func TestMemoryEntriesFixtureReplay(t *testing.T) {
	b, err := os.ReadFile(fixturePath("memory_entries.jsonl"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var entries []MemoryEntry
	for _, line := range splitLines(b) {
		if len(line) == 0 {
			continue
		}
		var e MemoryEntry
		must(t, json.Unmarshal(line, &e))
		entries = append(entries, e)
	}
	if len(entries) < 3 {
		t.Fatalf("fixture should carry several entries, got %d", len(entries))
	}

	p := NewInMemoryStorageProvider()
	for _, e := range entries {
		must(t, p.AppendMemory(ctx(), StorageScopeProject, sid("s"), e))
	}
	got, _ := p.GetMemories(ctx(), StorageScopeProject, sid("s"), 2)
	if len(got) != 2 {
		t.Fatalf("len=%d", len(got))
	}
	if got[0].Content != entries[len(entries)-1].Content || got[1].Content != entries[len(entries)-2].Content {
		t.Fatalf("recency mismatch: %v", contents(got))
	}
	all, _ := p.GetMemories(ctx(), StorageScopeProject, sid("s"), 999)
	for i := range all {
		if all[i].Content != entries[len(entries)-1-i].Content {
			t.Fatalf("order mismatch at %d: %v", i, contents(all))
		}
	}
}

// ── small local helpers (avoid extra imports) ───────────────────────────────

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func contents(es []MemoryEntry) []string {
	out := make([]string, len(es))
	for i, e := range es {
		out[i] = e.Content
	}
	return out
}

func itoa(i int) string { return string(rune('0' + i)) }

func canon(t *testing.T, v json.RawMessage) string {
	t.Helper()
	var x any
	must(t, json.Unmarshal(v, &x))
	b, err := json.Marshal(x)
	must(t, err)
	return string(b)
}

func splitLines(b []byte) [][]byte {
	var out [][]byte
	start := 0
	for i := 0; i < len(b); i++ {
		if b[i] == '\n' {
			out = append(out, trimSpace(b[start:i]))
			start = i + 1
		}
	}
	if start < len(b) {
		out = append(out, trimSpace(b[start:]))
	}
	return out
}

func trimSpace(b []byte) []byte {
	for len(b) > 0 && (b[0] == ' ' || b[0] == '\t' || b[0] == '\r') {
		b = b[1:]
	}
	for len(b) > 0 && (b[len(b)-1] == ' ' || b[len(b)-1] == '\t' || b[len(b)-1] == '\r') {
		b = b[:len(b)-1]
	}
	return b
}
