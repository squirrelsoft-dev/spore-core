package observability

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// outboxFixture mirrors the shared cross-language ground truth in
// fixtures/observability/trace_line_*.json. Each fixture carries the input
// span (and a trace_id) plus the expected on-disk JSONL line. Building the
// TraceLine from the input must produce a JSON-value-equal line.
type outboxFixture struct {
	TraceID      string          `json:"trace_id"`
	Span         json.RawMessage `json:"span"`
	ExpectedLine json.RawMessage `json:"expected_line"`
}

func fixturePath(t *testing.T, name string) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(wd, "..", "..", "..", "fixtures", "observability", name)
}

// buildLine constructs the TraceLine for one fixture file from its input span.
func buildLine(t *testing.T, file string, span json.RawMessage, traceID string) TraceLine {
	t.Helper()
	dec := func(v any) {
		if err := json.Unmarshal(span, v); err != nil {
			t.Fatalf("decode span for %s: %v", file, err)
		}
	}
	switch file {
	case "trace_line_turn.json":
		var s TurnSpan
		dec(&s)
		return TraceLineFromTurn(s, traceID)
	case "trace_line_tool_call.json":
		var s ToolCallSpan
		dec(&s)
		return TraceLineFromToolCall(s, traceID)
	case "trace_line_sensor.json":
		var s SensorSpan
		dec(&s)
		return TraceLineFromSensor(s, traceID)
	case "trace_line_context_assembly.json", "trace_line_compaction.json":
		var s ContextSpan
		dec(&s)
		return TraceLineFromContext(s, traceID)
	case "trace_line_middleware.json":
		var s MiddlewareSpan
		dec(&s)
		return TraceLineFromMiddleware(s, traceID)
	case "trace_line_patch.json":
		var s PatchSpan
		dec(&s)
		return TraceLineFromPatch(s, traceID)
	case "trace_line_session_summary.json":
		var wrapper struct {
			Metrics SessionMetrics `json:"metrics"`
			Root    SpanBase       `json:"root"`
		}
		if err := json.Unmarshal(span, &wrapper); err != nil {
			t.Fatalf("decode session_summary span: %v", err)
		}
		return TraceLineSessionSummary(wrapper.Metrics, traceID, wrapper.Root)
	default:
		t.Fatalf("unknown fixture %s", file)
		return TraceLine{}
	}
}

func TestOutboxFixtureReplayAllKinds(t *testing.T) {
	files := []string{
		"trace_line_turn.json",
		"trace_line_tool_call.json",
		"trace_line_sensor.json",
		"trace_line_context_assembly.json",
		"trace_line_compaction.json",
		"trace_line_middleware.json",
		"trace_line_patch.json",
		"trace_line_session_summary.json",
	}
	for _, f := range files {
		t.Run(f, func(t *testing.T) {
			raw, err := os.ReadFile(fixturePath(t, f))
			if err != nil {
				t.Fatalf("read fixture %s: %v", f, err)
			}
			var fx outboxFixture
			if err := json.Unmarshal(raw, &fx); err != nil {
				t.Fatalf("unmarshal fixture %s: %v", f, err)
			}
			line := buildLine(t, f, fx.Span, fx.TraceID)
			gotBytes, err := json.Marshal(line)
			if err != nil {
				t.Fatalf("marshal line: %v", err)
			}
			// Compare as JSON values (avoid key-ordering differences).
			var got, want any
			if err := json.Unmarshal(gotBytes, &got); err != nil {
				t.Fatalf("unmarshal got: %v", err)
			}
			if err := json.Unmarshal(fx.ExpectedLine, &want); err != nil {
				t.Fatalf("unmarshal expected: %v", err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("fixture %s mismatch\n got: %s\nwant: %s", f, gotBytes, fx.ExpectedLine)
			}
		})
	}
}
