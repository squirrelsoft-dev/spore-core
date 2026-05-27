// OTLP gRPC forwarder (issue #50) — the real otlpForwarder implementation.
//
// This file isolates the go.opentelemetry.io/otel SDK + otlptracegrpc surface
// from the reliability-critical outbox (outbox.go). The durable JSONL path
// never depends on this code: outbox.go appends the line first, then calls
// forward() best-effort. Behavior mirrors the Rust/TS/Python forwarders so a
// harness session collapses into ONE grouped Tempo trace keyed by the JSONL
// trace_id:
//
//   - OTLP trace id = the 32-hex JSONL TraceID parsed directly via
//     trace.TraceIDFromHex (NO hashing). On parse error the span is skipped
//     (early return); the JSONL line stays durable regardless.
//   - OTLP span id = trace.SpanID(deriveOTLPSpanID(line.SpanID)) — the shared
//     8-byte SHA-256 derivation (outbox.go), byte-for-byte with Rust.
//   - We do NOT build a real OTLP parent/child tree. We synthesize a remote
//     parent SpanContext carrying the harness trace id + derived span id, push
//     it into the context, then Start+End immediately. The harness
//     parent_span_id is carried ONLY as a string attribute (parity with
//     Rust/Python). The shared trace id is the only Tempo join key.
//   - Attributes per span: session_id, task_id, level, status, and
//     parent_span_id (only when non-nil). Span kind = Internal. Status: "ok" =>
//     OTel Ok, else OTel Error with status_detail as the description.
//   - force_flush bounds provider.ForceFlush with a 2s context timeout; errors
//     are swallowed/logged and NEVER propagate (the JSONL file is durable).
package observability

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// otlpFlushTimeout bounds the best-effort force-flush on FlushSession.
const otlpFlushTimeout = 2 * time.Second

// otlpTracerName is the instrumentation scope name for emitted spans.
const otlpTracerName = "spore-core"

// otlpSdkForwarder exports spans over OTLP gRPC via a batch span processor so
// export is buffered and non-blocking. It satisfies otlpForwarder.
type otlpSdkForwarder struct {
	provider *sdktrace.TracerProvider
	tracer   trace.Tracer
}

// stripScheme removes a leading http:// or https:// so the endpoint is the
// bare host:port that otlptracegrpc expects (parity with Python insecure=True).
func stripScheme(endpoint string) string {
	if rest, ok := strings.CutPrefix(endpoint, "https://"); ok {
		return rest
	}
	if rest, ok := strings.CutPrefix(endpoint, "http://"); ok {
		return rest
	}
	return endpoint
}

// newOTLPSdkForwarder builds an OTLP gRPC exporter against endpoint (scheme
// stripped, WithInsecure for the local-stack recipe) and a batch-exporter
// TracerProvider. The exporter is created lazily (no connection established
// until export), so this does not block on an unreachable collector.
func newOTLPSdkForwarder(endpoint string) (*otlpSdkForwarder, error) {
	target := stripScheme(endpoint)
	exporter, err := otlptracegrpc.New(
		context.Background(),
		otlptracegrpc.WithEndpoint(target),
		otlptracegrpc.WithInsecure(),
	)
	if err != nil {
		return nil, fmt.Errorf("otlptracegrpc.New: %w", err)
	}
	provider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
	)
	return &otlpSdkForwarder{
		provider: provider,
		tracer:   provider.Tracer(otlpTracerName),
	}, nil
}

// reservedAttrKeys are owned by the fixed envelope tags; a flattened attribute
// with one of these names is skipped so it never duplicates a fixed tag.
var reservedAttrKeys = map[string]struct{}{
	"session_id":     {},
	"task_id":        {},
	"level":          {},
	"status":         {},
	"parent_span_id": {},
}

// attributesToKeyValues flattens a span's top-level attributes object into OTLP
// KeyValues so the rich per-span detail (tokens, stop_reason, tool_name,
// turn_number, …) reaches Tempo, not just the JSONL.
//
// Rules (shallow, no dotted-key deep-flatten), mirroring the Rust reference:
//   - string → string attribute.
//   - bool → bool attribute.
//   - number → Int64 when integral (and within int64 range), else Float64.
//   - null → skipped (no key emitted).
//   - nested object/array → compact JSON string attribute.
//   - non-object top-level (shouldn't happen) → nothing.
//   - keys colliding with the fixed envelope tags (reservedAttrKeys) are skipped
//     so the fixed tag wins.
//
// Numbers arrive as float64 from JSON unmarshal; integral detection uses
// f == math.Trunc(f) bounded to the int64 range.
func attributesToKeyValues(raw json.RawMessage) []attribute.KeyValue {
	if len(raw) == 0 {
		return nil
	}
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		// Not a JSON object (or invalid) → nothing to flatten.
		return nil
	}
	out := make([]attribute.KeyValue, 0, len(obj))
	for key, value := range obj {
		if _, reserved := reservedAttrKeys[key]; reserved {
			continue
		}
		switch v := value.(type) {
		case nil:
			continue
		case string:
			out = append(out, attribute.String(key, v))
		case bool:
			out = append(out, attribute.Bool(key, v))
		case float64:
			if v == math.Trunc(v) && v >= math.MinInt64 && v <= math.MaxInt64 {
				out = append(out, attribute.Int64(key, int64(v)))
			} else {
				out = append(out, attribute.Float64(key, v))
			}
		default:
			// Nested object or array → compact JSON string.
			if js, err := json.Marshal(v); err == nil {
				out = append(out, attribute.String(key, string(js)))
			}
		}
	}
	return out
}

// forward emits one span carrying the harness trace id + derived span id.
// Best-effort and non-blocking; never errors. A bad trace id skips the span.
func (f *otlpSdkForwarder) forward(line TraceLine) {
	traceID, err := trace.TraceIDFromHex(line.TraceID)
	if err != nil {
		// JSONL stays durable; we just cannot place this span in Tempo.
		return
	}
	spanID := trace.SpanID(deriveOTLPSpanID(line.SpanID))

	// Synthesize a remote parent SpanContext so the emitted span adopts the
	// harness trace id and derived span id without building a real tree.
	parentCtx := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    traceID,
		SpanID:     spanID,
		TraceFlags: trace.FlagsSampled,
		Remote:     true,
	})
	ctx := trace.ContextWithSpanContext(context.Background(), parentCtx)

	attrs := []attribute.KeyValue{
		attribute.String("session_id", line.SessionID),
		attribute.String("task_id", line.TaskID),
		attribute.String("level", line.Level),
		attribute.String("status", line.Status),
	}
	if line.ParentSpanID != nil {
		// Tempo↔Loki join uses trace_id only; carry the readable parent
		// SpanID string as an attribute for cross-referencing (parity).
		attrs = append(attrs, attribute.String("parent_span_id", *line.ParentSpanID))
	}
	// Flatten the rich per-kind payload so token/stop_reason/tool_name/etc.
	// detail reaches Tempo, not just the JSONL. Fixed tags win via the skip list.
	attrs = append(attrs, attributesToKeyValues(line.Attributes)...)

	_, span := f.tracer.Start(
		ctx,
		line.Kind,
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(attrs...),
	)
	if line.Status == "ok" {
		span.SetStatus(codes.Ok, "")
	} else {
		detail := ""
		if line.StatusDetail != nil {
			detail = *line.StatusDetail
		}
		span.SetStatus(codes.Error, detail)
	}
	span.End()
}

// forceFlush best-effort flushes the batch processor, bounded by
// otlpFlushTimeout. Errors are logged and swallowed — never propagated.
func (f *otlpSdkForwarder) forceFlush() {
	ctx, cancel := context.WithTimeout(context.Background(), otlpFlushTimeout)
	defer cancel()
	if err := f.provider.ForceFlush(ctx); err != nil {
		fmt.Fprintf(os.Stderr,
			"[spore-core] OTLP force_flush failed (JSONL is durable): %v\n", err)
	}
}

// Compile-time interface check.
var _ otlpForwarder = (*otlpSdkForwarder)(nil)
