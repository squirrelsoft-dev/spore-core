package observability

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"testing"
	"time"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
)

// TestNewForwarderEmptyReturnsNull asserts that an empty or whitespace endpoint
// resolves to the no-op nullForwarder (JSONL only).
func TestNewForwarderEmptyReturnsNull(t *testing.T) {
	for _, ep := range []string{"", "   ", "\t\n"} {
		if _, ok := newForwarder(ep).(nullForwarder); !ok {
			t.Fatalf("newForwarder(%q) = %T, want nullForwarder", ep, newForwarder(ep))
		}
	}
}

// TestNewForwarderNonEmptyReturnsRealForwarder asserts that a non-empty
// endpoint constructs the real otlpSdkForwarder (lazy gRPC; no socket touched).
func TestNewForwarderNonEmptyReturnsRealForwarder(t *testing.T) {
	f := newForwarder("http://localhost:4317")
	if _, ok := f.(*otlpSdkForwarder); !ok {
		t.Fatalf("newForwarder(non-empty) = %T, want *otlpSdkForwarder", f)
	}
}

// TestDeriveOTLPSpanIDGolden pins the SHA-256-first-8-bytes derivation to a
// cross-language golden value (matches Rust/TS/Python derive_otlp_span_id).
func TestDeriveOTLPSpanIDGolden(t *testing.T) {
	d := deriveOTLPSpanID("sp1")
	got := hex.EncodeToString(d[:])
	const want = "94b97b295a989a13"
	if got != want {
		t.Fatalf("deriveOTLPSpanID(\"sp1\") = %s, want %s", got, want)
	}
}

// TestStripScheme covers the scheme-stripping used for the gRPC host:port.
func TestStripScheme(t *testing.T) {
	cases := map[string]string{
		"http://localhost:4317":  "localhost:4317",
		"https://tempo:4317":     "tempo:4317",
		"localhost:4317":         "localhost:4317",
		"https://host/path:4317": "host/path:4317",
	}
	for in, want := range cases {
		if got := stripScheme(in); got != want {
			t.Errorf("stripScheme(%q) = %q, want %q", in, got, want)
		}
	}
}

// newInMemoryForwarder builds an otlpSdkForwarder backed by an in-memory
// exporter via a SimpleSpanProcessor — no network/socket. This exercises the
// exact forward() span-mapping the gRPC forwarder uses.
func newInMemoryForwarder() (*otlpSdkForwarder, *tracetest.InMemoryExporter) {
	exp := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sdktrace.NewSimpleSpanProcessor(exp)))
	return &otlpSdkForwarder{provider: provider, tracer: provider.Tracer(otlpTracerName)}, exp
}

// TestOTLPForwardEmitsMappedSpan feeds one TraceLine through forward() and
// asserts the emitted span carries the hex-parsed trace id, the derived span
// id, and the session_id/task_id/level/status/parent_span_id attributes.
func TestOTLPForwardEmitsMappedSpan(t *testing.T) {
	f, exp := newInMemoryForwarder()

	parent := "turn-parent"
	line := TraceLine{
		TraceID:      "0123456789abcdef0123456789abcdef",
		SpanID:       "sp1",
		ParentSpanID: &parent,
		SessionID:    "sess-1",
		TaskID:       "task-1",
		Kind:         "tool_call",
		Level:        "info",
		Status:       "ok",
	}
	f.forward(line)

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("want 1 emitted span, got %d", len(spans))
	}
	s := spans[0]

	wantTrace, err := trace.TraceIDFromHex(line.TraceID)
	if err != nil {
		t.Fatal(err)
	}
	if s.SpanContext.TraceID() != wantTrace {
		t.Errorf("trace id = %s, want %s", s.SpanContext.TraceID(), wantTrace)
	}
	// The emitted span inherits the harness trace id (the only Tempo join key)
	// and is parented to the synthesized remote parent whose span id is the
	// derived 8-byte id. The span's OWN id is freshly minted by the SDK — we do
	// not build a real tree, so this is expected and matches Rust/Python.
	wantParent := trace.SpanID(deriveOTLPSpanID(line.SpanID))
	if s.Parent.SpanID() != wantParent {
		t.Errorf("parent span id = %s, want derived %s", s.Parent.SpanID(), wantParent)
	}
	if !s.Parent.IsRemote() {
		t.Errorf("parent span context should be remote")
	}
	if s.Name != "tool_call" {
		t.Errorf("name = %q, want tool_call", s.Name)
	}
	if s.SpanKind != trace.SpanKindInternal {
		t.Errorf("kind = %v, want Internal", s.SpanKind)
	}

	attrs := map[string]string{}
	for _, kv := range s.Attributes {
		attrs[string(kv.Key)] = kv.Value.AsString()
	}
	for k, want := range map[string]string{
		"session_id":     "sess-1",
		"task_id":        "task-1",
		"level":          "info",
		"status":         "ok",
		"parent_span_id": "turn-parent",
	} {
		if attrs[k] != want {
			t.Errorf("attr %s = %q, want %q", k, attrs[k], want)
		}
	}
}

// TestOTLPForwardSkipsBadTraceID asserts a non-hex trace id is skipped (no span
// emitted) — the JSONL line stays durable regardless.
func TestOTLPForwardSkipsBadTraceID(t *testing.T) {
	f, exp := newInMemoryForwarder()
	f.forward(TraceLine{TraceID: "not-hex", SpanID: "sp1", Kind: "turn", Status: "ok"})
	if got := len(exp.GetSpans()); got != 0 {
		t.Fatalf("want 0 spans for bad trace id, got %d", got)
	}
}

// TestOTLPForwardOmitsParentWhenNil asserts no parent_span_id attribute is set
// for a root span (nil ParentSpanID).
func TestOTLPForwardOmitsParentWhenNil(t *testing.T) {
	f, exp := newInMemoryForwarder()
	f.forward(TraceLine{
		TraceID: "0123456789abcdef0123456789abcdef",
		SpanID:  "root", Kind: "turn", Level: "info", Status: "ok",
	})
	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("want 1 span, got %d", len(spans))
	}
	for _, kv := range spans[0].Attributes {
		if string(kv.Key) == "parent_span_id" {
			t.Errorf("parent_span_id attribute set on root span")
		}
	}
}

// TestForceFlushUnreachableBounded asserts forceFlush completes (never panics)
// even against an unreachable endpoint, within the 2s bound.
func TestForceFlushUnreachableBounded(t *testing.T) {
	f, err := newOTLPSdkForwarder("127.0.0.1:0")
	if err != nil {
		t.Fatalf("construct forwarder: %v", err)
	}
	done := make(chan struct{})
	go func() {
		f.forward(TraceLine{TraceID: "0123456789abcdef0123456789abcdef", SpanID: "sp1", Kind: "turn", Status: "ok"})
		f.forceFlush()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(otlpFlushTimeout + 3*time.Second):
		t.Fatal("forceFlush did not complete within bound")
	}
}

// TestFlushSessionReturnsNilWhenOTLPFails asserts FlushSession returns nil even
// when the OTLP endpoint is unreachable (OTLP is best-effort; JSONL durable).
func TestFlushSessionReturnsNilWhenOTLPFails(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("SPORE_OTLP_ENDPOINT", "http://127.0.0.1:0")
	obs := NewOutboxObservabilityProvider(NewOutboxConfig(tmp))
	obs.EmitTurn(outboxTurn("s1", "sp1"))
	obs.Inner().SetSessionOutcome(sid("s1"), guideregistry.NewOutcomeSuccess())
	if err := obs.FlushSession(context.Background(), sid("s1")); err != nil {
		t.Fatalf("FlushSession returned error despite best-effort OTLP: %v", err)
	}
	// JSONL still durable: turn line + session summary.
	if got := len(readLines(t, tmp, "s1")); got < 2 {
		t.Fatalf("want >=2 durable lines, got %d", got)
	}
}

// TestAttributesToKeyValuesFlattensScalarsAndSkipsNull mirrors the Rust pure
// unit test: a turn-like attributes payload flattens to typed KeyValues, with
// integral numbers as Int64, strings as String, null skipped entirely, and the
// reserved envelope key skipped so the fixed tag wins.
func TestAttributesToKeyValuesFlattensScalarsAndSkipsNull(t *testing.T) {
	raw := json.RawMessage(`{
		"input_tokens": 386,
		"output_tokens": 102,
		"stop_reason": "tool_use",
		"turn_number": 1,
		"cache_read_tokens": null,
		"session_id": "should-be-skipped"
	}`)

	kvs := attributesToKeyValues(raw)
	get := func(k string) (attribute.Value, bool) {
		for _, kv := range kvs {
			if string(kv.Key) == k {
				return kv.Value, true
			}
		}
		return attribute.Value{}, false
	}

	for _, k := range []string{"input_tokens", "output_tokens", "turn_number"} {
		v, ok := get(k)
		if !ok {
			t.Fatalf("%s missing", k)
		}
		if v.Type() != attribute.INT64 {
			t.Fatalf("%s type = %v, want INT64", k, v.Type())
		}
	}
	if v, _ := get("input_tokens"); v.AsInt64() != 386 {
		t.Fatalf("input_tokens = %d, want 386", v.AsInt64())
	}
	if v, _ := get("output_tokens"); v.AsInt64() != 102 {
		t.Fatalf("output_tokens = %d, want 102", v.AsInt64())
	}
	if v, _ := get("turn_number"); v.AsInt64() != 1 {
		t.Fatalf("turn_number = %d, want 1", v.AsInt64())
	}

	if v, ok := get("stop_reason"); !ok || v.Type() != attribute.STRING || v.AsString() != "tool_use" {
		t.Fatalf("stop_reason = %v (ok=%v), want String tool_use", v, ok)
	}

	// Null is skipped entirely.
	if _, ok := get("cache_read_tokens"); ok {
		t.Fatal("cache_read_tokens (null) should be skipped")
	}
	// Reserved envelope key is skipped so the fixed tag wins.
	if _, ok := get("session_id"); ok {
		t.Fatal("session_id (reserved) should be skipped")
	}

	// 4 emitted: input_tokens, output_tokens, stop_reason, turn_number.
	if len(kvs) != 4 {
		t.Fatalf("len(kvs) = %d, want 4", len(kvs))
	}
}
