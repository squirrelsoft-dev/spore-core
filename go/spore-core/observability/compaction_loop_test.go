package observability

import (
	"context"
	"testing"
)

// bareProvider is an ObservabilityProvider that does NOT implement the optional
// WarnEmitter interface (issue #46). It proves the harness adapter's
// type-assert path silently drops warns for predating providers without
// breaking — the Go equivalent of Rust's default-no-op emit_warn.
type bareProvider struct{}

func (bareProvider) EmitTurn(TurnSpan)                             {}
func (bareProvider) EmitToolCall(ToolCallSpan)                     {}
func (bareProvider) EmitSensor(SensorSpan)                         {}
func (bareProvider) EmitContext(ContextSpan)                       {}
func (bareProvider) EmitMiddleware(MiddlewareSpan)                 {}
func (bareProvider) EmitPatch(PatchSpan)                           {}
func (bareProvider) SetSessionOutcome(SessionID, SessionOutcome)   {}
func (bareProvider) FlushSession(context.Context, SessionID) error { return nil }
func (bareProvider) GetSessionMetrics(context.Context, SessionID) (*SessionMetrics, error) {
	return nil, nil
}
func (bareProvider) GetSessions(context.Context, Timestamp, *string, *SessionOutcome) ([]SessionMetrics, error) {
	return nil, nil
}
func (bareProvider) GetTrace(context.Context, SessionID) ([]Span, error)        { return nil, nil }
func (bareProvider) ListUnflushedSessions(context.Context) ([]SessionID, error) { return nil, nil }
func (bareProvider) CleanupSession(context.Context, SessionID) error            { return nil }

var _ ObservabilityProvider = bareProvider{}

// TestEmitWarnOptionalPathDoesNotBreakBareProvider proves the adapter's
// WarnEmitter type-assert silently no-ops on a provider that does not implement
// it (W4), and conversely records the warn + surfaces the metric on a provider
// that does.
func TestEmitWarnOptionalPathDoesNotBreakBareProvider(t *testing.T) {
	// Bare provider: must not panic, simply drops the warn.
	bare := NewHarnessObserver(bareProvider{}, DefaultPricing())
	bare.EmitCompactionVerificationFailed("w1", "s", "t", "2026-01-01T00:00:00Z", []string{"x"}, true)

	// WarnEmitter-capable provider: records the warn and surfaces the metric.
	mem := NewInMemoryObservabilityProvider()
	obs := NewHarnessObserver(mem, DefaultPricing())
	obs.EmitCompactionVerificationFailed("w2", "s2", "t2", "2026-01-01T00:00:00Z", []string{"payment"}, true)
	// A session with only warn spans has no turns/outcome, so metrics are nil
	// until something else is recorded; record a terminal outcome (as a real
	// run does) so GetSessionMetrics can surface the derived failure count.
	obs.SetSessionOutcome("s2", true, "")

	warns := mem.WarnSpans("s2")
	if len(warns) != 1 {
		t.Fatalf("warn spans = %d, want 1", len(warns))
	}
	if warns[0].Event.Kind != WarnKindCompactionVerificationFailed {
		t.Fatalf("warn kind = %q", warns[0].Event.Kind)
	}
	m, _ := mem.GetSessionMetrics(context.Background(), "s2")
	if m == nil || m.CompactionVerificationFailures != 1 {
		t.Fatalf("CompactionVerificationFailures metric not 1: %+v", m)
	}
}
