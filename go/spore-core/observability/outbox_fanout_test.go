package observability

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"
)

// countingForwarder is a hermetic fake otlpForwarder: it counts forward() and
// forceFlush() calls so the fan-out routing matrix and failure isolation can be
// asserted without a live OTLP stack. failing is irrelevant — forward() never
// returns an error, mirroring the production fire-and-forget contract.
type countingForwarder struct {
	forwards atomic.Int64
	flushes  atomic.Int64
	// panics, when set, makes forward() panic to prove a misbehaving leg does
	// not bring down the others. The outbox recovers per-leg.
	panics bool
}

func (f *countingForwarder) forward(TraceLine) {
	if f.panics {
		panic("boom")
	}
	f.forwards.Add(1)
}
func (f *countingForwarder) forceFlush() { f.flushes.Add(1) }

// countingStore is a hermetic fake SpanStore counting AppendSpan/FlushSession.
type countingStore struct {
	appends  atomic.Int64
	flushes  atomic.Int64
	failNext bool
}

func (s *countingStore) AppendSpan(_ context.Context, _ SessionID, _ json.RawMessage) error {
	s.appends.Add(1)
	if s.failNext {
		return errors.New("store append failed")
	}
	return nil
}
func (s *countingStore) FlushSession(_ context.Context, _ SessionID) error {
	s.flushes.Add(1)
	return nil
}

// ── ParseOTLPEndpoints (mirror of the storage fixture; observability owns the
// canonical impl) ─────────────────────────────────────────────────────────────

func TestParseOTLPEndpointsTable(t *testing.T) {
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
		if len(got) != len(c.want) {
			t.Fatalf("ParseOTLPEndpoints(%q) = %v, want %v", c.in, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("ParseOTLPEndpoints(%q) = %v, want %v", c.in, got, c.want)
			}
		}
	}
}

// ── Fan-out routing matrix ───────────────────────────────────────────────────
//
// both / otlp-only / store-only / dropped, asserted by the per-leg counters and
// the durable JSONL file (always written regardless of legs).

func TestFanOutMatrixBoth(t *testing.T) {
	tmp := t.TempDir()
	p := newProvider(t, NewOutboxConfig(tmp))
	fwd := &countingForwarder{}
	store := &countingStore{}
	p.otlp = []otlpForwarder{fwd}
	p = p.WithStore(store)

	p.EmitTurn(outboxTurn("s1", "sp1"))

	if fwd.forwards.Load() != 1 {
		t.Fatalf("otlp forwards = %d, want 1", fwd.forwards.Load())
	}
	if store.appends.Load() != 1 {
		t.Fatalf("store appends = %d, want 1", store.appends.Load())
	}
	if got := len(readLines(t, tmp, "s1")); got != 1 {
		t.Fatalf("jsonl lines = %d, want 1", got)
	}
}

func TestFanOutMatrixOTLPOnly(t *testing.T) {
	tmp := t.TempDir()
	p := newProvider(t, NewOutboxConfig(tmp))
	fwd := &countingForwarder{}
	p.otlp = []otlpForwarder{fwd}
	// no store leg (store-only no-op via nil)

	p.EmitTurn(outboxTurn("s1", "sp1"))

	if fwd.forwards.Load() != 1 {
		t.Fatalf("otlp forwards = %d, want 1", fwd.forwards.Load())
	}
	if p.store != nil {
		t.Fatal("store leg should be nil")
	}
	if got := len(readLines(t, tmp, "s1")); got != 1 {
		t.Fatalf("jsonl lines = %d, want 1", got)
	}
}

func TestFanOutMatrixStoreOnly(t *testing.T) {
	tmp := t.TempDir()
	p := newProvider(t, NewOutboxConfig(tmp))
	store := &countingStore{}
	p = p.WithStore(store)
	// no OTLP legs (env cleared by newProvider → empty slice)

	p.EmitTurn(outboxTurn("s1", "sp1"))

	if len(p.otlp) != 0 {
		t.Fatalf("otlp legs = %d, want 0", len(p.otlp))
	}
	if store.appends.Load() != 1 {
		t.Fatalf("store appends = %d, want 1", store.appends.Load())
	}
	if got := len(readLines(t, tmp, "s1")); got != 1 {
		t.Fatalf("jsonl lines = %d, want 1", got)
	}
}

func TestFanOutMatrixDropped(t *testing.T) {
	tmp := t.TempDir()
	p := newProvider(t, NewOutboxConfig(tmp))
	// neither OTLP nor store configured: spans go to JSONL only (the durable
	// source of truth); both fan-out legs are silently dropped.

	p.EmitTurn(outboxTurn("s1", "sp1"))

	if len(p.otlp) != 0 {
		t.Fatalf("otlp legs = %d, want 0", len(p.otlp))
	}
	if p.store != nil {
		t.Fatal("store leg should be nil")
	}
	if got := len(readLines(t, tmp, "s1")); got != 1 {
		t.Fatalf("jsonl lines = %d, want 1", got)
	}
}

// ── Multi-endpoint fan-out: every endpoint receives every span ───────────────

func TestFanOutMultiEndpoint(t *testing.T) {
	tmp := t.TempDir()
	p := newProvider(t, NewOutboxConfig(tmp))
	f1, f2, f3 := &countingForwarder{}, &countingForwarder{}, &countingForwarder{}
	p.otlp = []otlpForwarder{f1, f2, f3}

	p.EmitTurn(outboxTurn("s1", "sp1"))
	p.EmitTurn(outboxTurn("s1", "sp2"))

	for i, f := range []*countingForwarder{f1, f2, f3} {
		if f.forwards.Load() != 2 {
			t.Fatalf("forwarder %d forwards = %d, want 2", i, f.forwards.Load())
		}
	}
}

// ── Fan-out failure isolation: a panicking leg does not block the others ─────

func TestFanOutFailureIsolation(t *testing.T) {
	tmp := t.TempDir()
	p := newProvider(t, NewOutboxConfig(tmp))
	bad := &countingForwarder{panics: true}
	good := &countingForwarder{}
	// bad first so if it weren't isolated, good would never run.
	p.otlp = []otlpForwarder{wrapRecover(bad), good}
	store := &countingStore{failNext: true}
	p = p.WithStore(store)

	// Must not panic out of EmitTurn, and the good leg + store + JSONL all run.
	p.EmitTurn(outboxTurn("s1", "sp1"))

	if good.forwards.Load() != 1 {
		t.Fatalf("good forwarder forwards = %d, want 1 (bad leg not isolated)", good.forwards.Load())
	}
	if store.appends.Load() != 1 {
		t.Fatalf("store appends = %d, want 1 (store error not isolated)", store.appends.Load())
	}
	if got := len(readLines(t, tmp, "s1")); got != 1 {
		t.Fatalf("jsonl lines = %d, want 1 (durable write must always happen)", got)
	}
}

// recoverForwarder wraps a forwarder so a panic in forward() is contained — the
// production OTLP SDK forwarder never panics, but a hostile/buggy leg must not
// abort the fan-out. This mirrors the per-leg isolation guarantee.
type recoverForwarder struct{ inner otlpForwarder }

func wrapRecover(f otlpForwarder) otlpForwarder { return &recoverForwarder{inner: f} }

func (r *recoverForwarder) forward(line TraceLine) {
	defer func() { _ = recover() }()
	r.inner.forward(line)
}
func (r *recoverForwarder) forceFlush() {
	defer func() { _ = recover() }()
	r.inner.forceFlush()
}

// ── Bad endpoint skipped at construction ─────────────────────────────────────

func TestBuildForwardersSkipsUnparseable(t *testing.T) {
	// Empty / whitespace yields no legs.
	if got := buildForwarders(""); len(got) != 0 {
		t.Fatalf("buildForwarders(\"\") = %d legs, want 0", len(got))
	}
	if got := buildForwarders("  "); len(got) != 0 {
		t.Fatalf("buildForwarders(\"  \") = %d legs, want 0", len(got))
	}
	// A list of valid endpoints yields one leg each (lazy gRPC; no socket
	// touched). The "a,,b," empties are dropped by ParseOTLPEndpoints.
	got := buildForwarders("http://c1:4317,,http://c2:4317,")
	if len(got) != 2 {
		t.Fatalf("buildForwarders multi = %d legs, want 2", len(got))
	}
}

// ── FlushSession fans out force-flush + store flush; .flushed marker ─────────

func TestFanOutFlushSession(t *testing.T) {
	tmp := t.TempDir()
	p := newProvider(t, NewOutboxConfig(tmp))
	f1, f2 := &countingForwarder{}, &countingForwarder{}
	p.otlp = []otlpForwarder{f1, f2}
	store := &countingStore{}
	p = p.WithStore(store)

	p.EmitTurn(outboxTurn("s1", "sp1"))
	if err := p.FlushSession(context.Background(), sid("s1")); err != nil {
		t.Fatalf("FlushSession: %v", err)
	}

	if f1.flushes.Load() != 1 || f2.flushes.Load() != 1 {
		t.Fatalf("force-flush not fanned out: f1=%d f2=%d", f1.flushes.Load(), f2.flushes.Load())
	}
	if store.flushes.Load() != 1 {
		t.Fatalf("store flush = %d, want 1", store.flushes.Load())
	}
}

// ── Harness default storage no-op: a builder with no WithStorage leaves the
// outbox store leg nil; emit still writes JSONL and never touches a store. ────

func TestHarnessDefaultStorageNoOp(t *testing.T) {
	t.Setenv("SPORE_OTLP_ENDPOINT", "")
	tmp := t.TempDir()
	b := NewHarnessBuilder(nil, nil, nil, nil, nil).WithObservabilityOutbox(tmp)
	outbox, ok := b.provider.(*OutboxObservabilityProvider)
	if !ok {
		t.Fatalf("provider = %T, want *OutboxObservabilityProvider", b.provider)
	}
	if outbox.store != nil {
		t.Fatal("default outbox store leg should be nil (no-op)")
	}
	// And emit still works, writing the durable JSONL.
	outbox.EmitTurn(outboxTurn("s1", "sp1"))
	if got := len(readLines(t, tmp, "s1")); got != 1 {
		t.Fatalf("jsonl lines = %d, want 1", got)
	}
}

// ── WithStorage wires the store leg into a builder-constructed outbox ────────

func TestWithStorageWiresStoreLeg(t *testing.T) {
	t.Setenv("SPORE_OTLP_ENDPOINT", "")
	tmp := t.TempDir()
	store := &countingStore{}
	b := NewHarnessBuilder(nil, nil, nil, nil, nil).
		WithStorage(store).
		WithObservabilityOutbox(tmp)
	outbox := b.provider.(*OutboxObservabilityProvider)
	if outbox.store == nil {
		t.Fatal("WithStorage store leg not wired into outbox")
	}
	outbox.EmitTurn(outboxTurn("s1", "sp1"))
	if store.appends.Load() != 1 {
		t.Fatalf("store appends = %d, want 1", store.appends.Load())
	}
}
