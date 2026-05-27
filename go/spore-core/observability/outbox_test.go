package observability

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/middleware"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/sensor"
)

// newProvider returns a hermetic provider: SPORE_OTLP_ENDPOINT is cleared so
// no live OTLP/network is touched.
func newProvider(t *testing.T, cfg OutboxConfig) *OutboxObservabilityProvider {
	t.Helper()
	t.Setenv("SPORE_OTLP_ENDPOINT", "")
	return NewOutboxObservabilityProvider(cfg)
}

func outboxBase(session, spanID string, kind SpanKind, status SpanStatus) SpanBase {
	return SpanBase{
		SpanID:     SpanID(spanID),
		SessionID:  sid(session),
		TaskID:     tid("task1"),
		Kind:       kind,
		StartedAt:  ts("2026-05-26T18:00:00.000Z"),
		EndedAt:    ts("2026-05-26T18:00:02.100Z"),
		DurationMs: 2100,
		Status:     status,
	}
}

func outboxTurn(session, spanID string) TurnSpan {
	cr := uint32(1600)
	cw := uint32(0)
	return TurnSpan{
		Base:               outboxBase(session, spanID, SpanKindTurn, NewStatusOk()),
		TurnNumber:         1,
		InputTokens:        1820,
		OutputTokens:       140,
		CacheReadTokens:    &cr,
		CacheWriteTokens:   &cw,
		CostUSD:            0.0123,
		StopReason:         StopReason("tool_use"),
		ToolCallsRequested: 1,
	}
}

func readLines(t *testing.T, root, session string) []map[string]any {
	t.Helper()
	path := filepath.Join(root, "sessions", session, activeFileName)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read trace.jsonl: %v", err)
	}
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimRight(string(raw), "\n"), "\n") {
		if line == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("unmarshal line %q: %v", line, err)
		}
		out = append(out, m)
	}
	return out
}

func TestOutboxOneLinePerEmit(t *testing.T) {
	tmp := t.TempDir()
	obs := newProvider(t, NewOutboxConfig(tmp))
	obs.EmitTurn(outboxTurn("s1", "sp1"))
	obs.EmitTurn(outboxTurn("s1", "sp2"))
	if got := len(readLines(t, tmp, "s1")); got != 2 {
		t.Fatalf("want 2 lines, got %d", got)
	}
}

func TestOutboxTurnLineMatchesSchema(t *testing.T) {
	tmp := t.TempDir()
	obs := newProvider(t, NewOutboxConfig(tmp))
	obs.EmitTurn(outboxTurn("s1", "sp1"))
	l := readLines(t, tmp, "s1")[0]

	checks := map[string]any{
		"kind":       "turn",
		"level":      "info",
		"span_id":    "sp1",
		"session_id": "s1",
		"task_id":    "task1",
		"timestamp":  "2026-05-26T18:00:02.100Z",
		"started_at": "2026-05-26T18:00:00.000Z",
		"status":     "ok",
	}
	for k, want := range checks {
		if l[k] != want {
			t.Errorf("%s=%v want %v", k, l[k], want)
		}
	}
	if l["parent_span_id"] != nil {
		t.Errorf("parent_span_id=%v want nil", l["parent_span_id"])
	}
	if l["status_detail"] != nil {
		t.Errorf("status_detail=%v want nil", l["status_detail"])
	}
	if l["duration_ms"].(float64) != 2100 {
		t.Errorf("duration_ms=%v", l["duration_ms"])
	}
	attrs := l["attributes"].(map[string]any)
	if attrs["turn_number"].(float64) != 1 || attrs["input_tokens"].(float64) != 1820 {
		t.Errorf("attrs scalars wrong: %v", attrs)
	}
	if attrs["cache_read_tokens"].(float64) != 1600 {
		t.Errorf("cache_read_tokens=%v", attrs["cache_read_tokens"])
	}
	if attrs["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason=%v", attrs["stop_reason"])
	}
	if tid := l["trace_id"].(string); len(tid) != 32 {
		t.Errorf("trace_id len=%d want 32", len(tid))
	}
}

func TestOutboxPatchLineIsWarn(t *testing.T) {
	tmp := t.TempDir()
	obs := newProvider(t, NewOutboxConfig(tmp))
	span := NewPatchSpan(
		outboxBase("s1", "p1", SpanKindPatch, NewStatusOk()),
		"c1", "shell",
		json.RawMessage(`{"a":"1"}`),
		json.RawMessage(`{"a":1}`),
		NewPatchParameterCoercion("a", "string", "number"),
	)
	obs.EmitPatch(span)
	l := readLines(t, tmp, "s1")[0]
	if l["kind"] != "patch" || l["level"] != "warn" {
		t.Fatalf("kind/level wrong: %v / %v", l["kind"], l["level"])
	}
	attrs := l["attributes"].(map[string]any)
	pt := attrs["patch_type"].(map[string]any)
	if pt["kind"] != "parameter_coercion" {
		t.Errorf("patch_type.kind=%v", pt["kind"])
	}
	if attrs["original_parameters"].(map[string]any)["a"] != "1" {
		t.Errorf("original_parameters wrong: %v", attrs["original_parameters"])
	}
	if attrs["patched_parameters"].(map[string]any)["a"].(float64) != 1 {
		t.Errorf("patched_parameters wrong: %v", attrs["patched_parameters"])
	}
}

func TestOutboxStatusErrorAndHalted(t *testing.T) {
	tmp := t.TempDir()
	obs := newProvider(t, NewOutboxConfig(tmp))
	e := outboxTurn("s1", "err")
	e.Base.Status = NewStatusError("boom")
	obs.EmitTurn(e)
	h := outboxTurn("s1", "halt")
	h.Base.Status = NewStatusHalted("stop")
	obs.EmitTurn(h)
	lines := readLines(t, tmp, "s1")
	if lines[0]["status"] != "error" || lines[0]["status_detail"] != "boom" {
		t.Errorf("error line wrong: %v", lines[0])
	}
	if lines[1]["status"] != "halted" || lines[1]["status_detail"] != "stop" {
		t.Errorf("halted line wrong: %v", lines[1])
	}
}

func TestOutboxContextVsCompactionKind(t *testing.T) {
	tmp := t.TempDir()
	obs := newProvider(t, NewOutboxConfig(tmp))
	mk := func(spanID string, op ContextOperation) ContextSpan {
		return ContextSpan{
			Base:              outboxBase("s1", spanID, SpanKindContextAssembly, NewStatusOk()),
			Operation:         op,
			TokensBefore:      100,
			TokensAfter:       50,
			UtilizationBefore: 0.9,
			UtilizationAfter:  0.5,
		}
	}
	obs.EmitContext(mk("asm", NewContextOpAssembly(1, 2, 3)))
	obs.EmitContext(mk("comp", NewContextOpCompaction(5, 50)))
	lines := readLines(t, tmp, "s1")
	if lines[0]["kind"] != "context_assembly" {
		t.Errorf("line0 kind=%v", lines[0]["kind"])
	}
	if lines[1]["kind"] != "compaction" {
		t.Errorf("line1 kind=%v", lines[1]["kind"])
	}
	op0 := lines[0]["attributes"].(map[string]any)["operation"].(map[string]any)
	if op0["kind"] != "assembly" {
		t.Errorf("op0 kind=%v", op0["kind"])
	}
}

func TestOutboxSensorAndMiddlewareLines(t *testing.T) {
	tmp := t.TempDir()
	obs := newProvider(t, NewOutboxConfig(tmp))
	obs.EmitSensor(SensorSpan{
		Base:       outboxBase("s1", "sn1", SpanKindSensorEvaluation, NewStatusOk()),
		SensorID:   sensor.SensorID("test-runner"),
		SensorKind: sensor.SensorKind("computational"),
		Trigger:    sensor.NewTriggerPostTool("shell"),
		Outcome:    sensor.OutcomePass,
		Fired:      true,
	})
	obs.EmitMiddleware(MiddlewareSpan{
		Base:     outboxBase("s1", "mw1", SpanKindMiddlewareHook, NewStatusOk()),
		Hook:     middleware.HookBeforeTurn,
		Decision: middleware.DecisionContinueVal(),
	})
	lines := readLines(t, tmp, "s1")
	sa := lines[0]["attributes"].(map[string]any)
	if lines[0]["kind"] != "sensor_evaluation" || sa["sensor_id"] != "test-runner" {
		t.Errorf("sensor line wrong: %v", lines[0])
	}
	if sa["trigger"].(map[string]any)["kind"] != "post_tool" {
		t.Errorf("trigger wrong: %v", sa["trigger"])
	}
	ma := lines[1]["attributes"].(map[string]any)
	if lines[1]["kind"] != "middleware_hook" {
		t.Errorf("mw kind=%v", lines[1]["kind"])
	}
	if ma["decision"].(map[string]any)["kind"] != "continue" {
		t.Errorf("decision wrong: %v", ma["decision"])
	}
}

func TestOutboxFlushWritesSummaryAndMarker(t *testing.T) {
	tmp := t.TempDir()
	obs := newProvider(t, NewOutboxConfig(tmp))
	obs.EmitTurn(outboxTurn("s1", "sp1"))
	obs.Inner().SetSessionOutcome(sid("s1"), guideregistry.NewOutcomeSuccess())
	if err := obs.FlushSession(context.Background(), sid("s1")); err != nil {
		t.Fatal(err)
	}
	lines := readLines(t, tmp, "s1")
	last := lines[len(lines)-1]
	if last["kind"] != "session" {
		t.Fatalf("last line kind=%v", last["kind"])
	}
	attrs := last["attributes"].(map[string]any)
	if attrs["outcome"] != "success" || attrs["total_turns"].(float64) != 1 {
		t.Errorf("summary attrs wrong: %v", attrs)
	}
	if !fileExists(filepath.Join(tmp, "sessions", "s1", flushedMarkerName)) {
		t.Errorf(".flushed marker missing")
	}
}

func TestOutboxFlushNoSummaryWhenDisabled(t *testing.T) {
	tmp := t.TempDir()
	cfg := NewOutboxConfig(tmp)
	cfg.FlushOnSessionEnd = false
	obs := newProvider(t, cfg)
	obs.EmitTurn(outboxTurn("s1", "sp1"))
	obs.Inner().SetSessionOutcome(sid("s1"), guideregistry.NewOutcomeSuccess())
	if err := obs.FlushSession(context.Background(), sid("s1")); err != nil {
		t.Fatal(err)
	}
	lines := readLines(t, tmp, "s1")
	if len(lines) != 1 || lines[0]["kind"] != "turn" {
		t.Fatalf("expected only turn line, got %v", lines)
	}
	if !fileExists(filepath.Join(tmp, "sessions", "s1", flushedMarkerName)) {
		t.Errorf(".flushed marker missing")
	}
}

func TestOutboxRotationAtTinyMaxSize(t *testing.T) {
	tmp := t.TempDir()
	cfg := NewOutboxConfig(tmp)
	cfg.MaxSizeBytes = 10 // each line far exceeds 10 bytes → rotate each emit
	obs := newProvider(t, cfg)
	obs.EmitTurn(outboxTurn("s1", "sp1"))
	obs.EmitTurn(outboxTurn("s1", "sp2"))
	obs.EmitTurn(outboxTurn("s1", "sp3"))
	dir := filepath.Join(tmp, "sessions", "s1")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var rotated int
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "trace-") && strings.HasSuffix(e.Name(), ".jsonl") {
			rotated++
		}
	}
	if rotated == 0 {
		t.Fatalf("expected at least one rotated segment, got dir %v", entries)
	}
	if !fileExists(filepath.Join(dir, "trace-001.jsonl")) {
		t.Errorf("trace-001.jsonl missing")
	}
}

func TestOutboxJSONLOnlyWhenEnvUnset(t *testing.T) {
	tmp := t.TempDir()
	obs := newProvider(t, NewOutboxConfig(tmp)) // env cleared → nullForwarder
	obs.EmitTurn(outboxTurn("s1", "sp1"))
	if got := len(readLines(t, tmp, "s1")); got != 1 {
		t.Fatalf("want 1 line, got %d", got)
	}
}

func TestOutboxListUnflushedBeforeAndAfterFlush(t *testing.T) {
	tmp := t.TempDir()
	obs := newProvider(t, NewOutboxConfig(tmp))
	obs.EmitTurn(outboxTurn("s1", "sp1"))
	before, err := obs.ListUnflushedSessions(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(before) != 1 || before[0] != sid("s1") {
		t.Fatalf("before=%v want [s1]", before)
	}
	obs.Inner().SetSessionOutcome(sid("s1"), guideregistry.NewOutcomeSuccess())
	if err := obs.FlushSession(context.Background(), sid("s1")); err != nil {
		t.Fatal(err)
	}
	after, err := obs.ListUnflushedSessions(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != 0 {
		t.Fatalf("after=%v want empty", after)
	}
}

func TestOutboxCleanupSessionSuccessAndNotFound(t *testing.T) {
	tmp := t.TempDir()
	obs := newProvider(t, NewOutboxConfig(tmp))
	obs.EmitTurn(outboxTurn("s1", "sp1"))
	if err := obs.CleanupSession(context.Background(), sid("s1")); err != nil {
		t.Fatalf("cleanup s1: %v", err)
	}
	if fileExists(filepath.Join(tmp, "sessions", "s1")) {
		t.Errorf("session dir not removed")
	}
	err := obs.CleanupSession(context.Background(), sid("missing"))
	if !errors.Is(err, ErrSessionNotFound) {
		t.Fatalf("want ErrSessionNotFound, got %v", err)
	}
	var nf *SessionNotFoundError
	if !errors.As(err, &nf) || nf.SessionID != sid("missing") {
		t.Fatalf("want SessionNotFoundError{missing}, got %v", err)
	}
}

func TestOutboxTraceIDStablePerSessionDistinctAcross(t *testing.T) {
	tmp := t.TempDir()
	obs := newProvider(t, NewOutboxConfig(tmp))
	obs.EmitTurn(outboxTurn("s1", "a"))
	obs.EmitTurn(outboxTurn("s1", "b"))
	obs.EmitTurn(outboxTurn("s2", "c"))
	s1 := readLines(t, tmp, "s1")
	s2 := readLines(t, tmp, "s2")
	if s1[0]["trace_id"] != s1[1]["trace_id"] {
		t.Errorf("trace_id not stable within session: %v vs %v", s1[0]["trace_id"], s1[1]["trace_id"])
	}
	if s1[0]["trace_id"] == s2[0]["trace_id"] {
		t.Errorf("trace_id not distinct across sessions")
	}
}

func TestDeriveOTLPSpanIDStable(t *testing.T) {
	a := deriveOTLPSpanID("sp1")
	b := deriveOTLPSpanID("sp1")
	c := deriveOTLPSpanID("sp2")
	if a != b {
		t.Errorf("derive not stable")
	}
	if a == c {
		t.Errorf("derive not distinct")
	}
}
