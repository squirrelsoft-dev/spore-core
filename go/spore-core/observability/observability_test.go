package observability

import (
	"context"
	"math"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/memory"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/middleware"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/sensor"
)

// ─── helpers ────────────────────────────────────────────────────────────────

func ts(s string) Timestamp  { return memory.Timestamp(s) }
func sid(s string) SessionID { return sporecore.SessionID(s) }
func tid(s string) TaskID    { return sporecore.TaskID(s) }

func turnSpan(session, spanID string, turn, input, output uint32) TurnSpan {
	return TurnSpan{
		Base: SpanBase{
			SpanID:     SpanID(spanID),
			SessionID:  sid(session),
			TaskID:     tid("t1"),
			Kind:       SpanKindTurn,
			StartedAt:  ts("2026-05-16T00:00:00Z"),
			EndedAt:    ts("2026-05-16T00:00:01Z"),
			DurationMs: 1000,
			Status:     NewStatusOk(),
		},
		TurnNumber:   turn,
		InputTokens:  input,
		OutputTokens: output,
		StopReason:   sporecore.StopEndTurn,
	}
}

// ─── Rule: EmitTurn is fire-and-forget; metrics aggregate ───────────────────

func TestEmitTurnRecordedAndMetricsAggregate(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	obs.EmitTurn(turnSpan("s1", "sp1", 1, 100, 50))
	obs.EmitTurn(turnSpan("s1", "sp2", 2, 200, 80))
	obs.SetSessionOutcome(sid("s1"), guideregistry.NewOutcomeSuccess())

	m, err := obs.GetSessionMetrics(context.Background(), sid("s1"))
	if err != nil || m == nil {
		t.Fatalf("metrics: err=%v m=%v", err, m)
	}
	if m.TotalTurns != 2 || m.TotalInputTokens != 300 || m.TotalOutputTokens != 130 {
		t.Fatalf("aggregates wrong: %+v", m)
	}
	if m.Outcome.Kind != guideregistry.OutcomeKindSuccess {
		t.Fatalf("outcome: %v", m.Outcome)
	}
}

// ─── Rule: EmitToolCall counted in metrics; duration accumulates ────────────

func TestEmitToolCallCountedInMetrics(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	obs.EmitTurn(turnSpan("s1", "t1", 1, 10, 5))
	obs.EmitToolCall(ToolCallSpan{
		Base: SpanBase{
			SpanID:     "tc1",
			SessionID:  sid("s1"),
			TaskID:     tid("t1"),
			Kind:       SpanKindToolCall,
			StartedAt:  ts("2026-05-16T00:00:00Z"),
			EndedAt:    ts("2026-05-16T00:00:00Z"),
			DurationMs: 250,
			Status:     NewStatusOk(),
		},
		ToolName:    "shell",
		CallID:      "c1",
		SandboxMode: "workspace_scoped",
	})
	m, _ := obs.GetSessionMetrics(context.Background(), sid("s1"))
	if m.ToolCalls != 1 {
		t.Fatalf("tool_calls=%d", m.ToolCalls)
	}
	if m.TotalDurationMs != 1250 {
		t.Fatalf("duration=%d", m.TotalDurationMs)
	}
}

// ─── Rule: Sensor metrics count fires and halts ─────────────────────────────

func TestSensorMetricsCountFiresAndHalts(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	obs.EmitTurn(turnSpan("s1", "t1", 1, 10, 5))
	mk := func(id string, fired bool, outcome SensorOutcome) SensorSpan {
		return SensorSpan{
			Base: SpanBase{
				SpanID:    SpanID(id),
				SessionID: sid("s1"),
				TaskID:    tid("t1"),
				Kind:      SpanKindSensorEvaluation,
				StartedAt: ts("2026-05-16T00:00:00Z"),
				EndedAt:   ts("2026-05-16T00:00:00Z"),
				Status:    NewStatusOk(),
			},
			SensorID:   sensor.SensorID("lint"),
			SensorKind: sensor.SensorKindComputational,
			Trigger:    sensor.NewTriggerPostTurn(),
			Outcome:    outcome,
			Fired:      fired,
		}
	}
	obs.EmitSensor(mk("sn1", true, sensor.OutcomeWarn))
	obs.EmitSensor(mk("sn2", true, sensor.OutcomeHalt))
	obs.EmitSensor(mk("sn3", false, sensor.OutcomePass))
	m, _ := obs.GetSessionMetrics(context.Background(), sid("s1"))
	if m.SensorFires != 2 {
		t.Fatalf("fires=%d", m.SensorFires)
	}
	if m.SensorHalts != 1 {
		t.Fatalf("halts=%d", m.SensorHalts)
	}
}

// ─── Rule: Compactions counted in metrics ───────────────────────────────────

func TestCompactionCountedInMetrics(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	obs.EmitTurn(turnSpan("s1", "t1", 1, 100, 50))
	mkCtx := func(op ContextOperation) ContextSpan {
		return ContextSpan{
			Base: SpanBase{
				SpanID:    "c1",
				SessionID: sid("s1"),
				TaskID:    tid("t1"),
				Kind:      SpanKindCompaction,
				StartedAt: ts("2026-05-16T00:00:00Z"),
				EndedAt:   ts("2026-05-16T00:00:00Z"),
				Status:    NewStatusOk(),
			},
			Operation:    op,
			TokensBefore: 10000, TokensAfter: 5000,
			UtilizationBefore: 0.9, UtilizationAfter: 0.5,
		}
	}
	obs.EmitContext(mkCtx(NewContextOpCompaction(5, 5000)))
	obs.EmitContext(mkCtx(NewContextOpAssembly(2, 3, 5)))
	m, _ := obs.GetSessionMetrics(context.Background(), sid("s1"))
	if m.Compactions != 1 {
		t.Fatalf("compactions=%d", m.Compactions)
	}
}

// ─── Rule: PricingTable computes cost; DefaultPricing returns zero ──────────

func TestPricingTableComputesCost(t *testing.T) {
	table := PricingTable{
		InputPerMillion:      3.0,
		OutputPerMillion:     15.0,
		CacheReadPerMillion:  0.3,
		CacheWritePerMillion: 3.75,
	}
	one := uint32(1_000_000)
	cost := table.CostFor(one, one, &one, &one)
	// 3 + 15 + 0.3 + 3.75 = 22.05
	if math.Abs(cost-22.05) > 1e-9 {
		t.Fatalf("cost=%v want 22.05", cost)
	}
}

func TestDefaultPricingIsZero(t *testing.T) {
	one := uint32(1000)
	cost := DefaultPricing().CostFor(one, one, &one, &one)
	if cost != 0 {
		t.Fatalf("default cost=%v want 0", cost)
	}
}

func TestPricingTableHandlesNilCacheTokens(t *testing.T) {
	table := PricingTable{InputPerMillion: 3.0, OutputPerMillion: 15.0}
	// 3 * 1 + 15 * 1 = 18 (no cache tokens contributed)
	cost := table.CostFor(1_000_000, 1_000_000, nil, nil)
	if math.Abs(cost-18.0) > 1e-9 {
		t.Fatalf("cost=%v want 18", cost)
	}
}

// ─── Rule: FlushSession is idempotent — spans remain queryable ──────────────

func TestFlushSessionIsIdempotent(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	obs.EmitTurn(turnSpan("s1", "t1", 1, 10, 5))
	ctx := context.Background()
	if err := obs.FlushSession(ctx, sid("s1")); err != nil {
		t.Fatal(err)
	}
	if err := obs.FlushSession(ctx, sid("s1")); err != nil {
		t.Fatal(err)
	}
	m, _ := obs.GetSessionMetrics(ctx, sid("s1"))
	if m == nil || m.TotalTurns != 1 {
		t.Fatalf("metrics post-flush: %+v", m)
	}
}

// ─── Rule: GetTrace returns spans in insertion order; ParentSpanID kept ─────

func TestGetTracePreservesInsertionOrder(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	obs.EmitTurn(turnSpan("s1", "sp1", 1, 10, 5))
	parent := SpanID("sp1")
	obs.EmitToolCall(ToolCallSpan{
		Base: SpanBase{
			SpanID:       "sp2",
			ParentSpanID: &parent,
			SessionID:    sid("s1"),
			TaskID:       tid("t1"),
			Kind:         SpanKindToolCall,
			StartedAt:    ts("2026-05-16T00:00:00Z"),
			EndedAt:      ts("2026-05-16T00:00:00Z"),
			Status:       NewStatusOk(),
		},
		ToolName: "shell", CallID: "c1", SandboxMode: "none",
	})
	trace, err := obs.GetTrace(context.Background(), sid("s1"))
	if err != nil {
		t.Fatal(err)
	}
	if len(trace) != 2 {
		t.Fatalf("trace len=%d", len(trace))
	}
	if trace[0].GetBase().SpanID != "sp1" {
		t.Fatalf("ordering[0]=%v", trace[0].GetBase().SpanID)
	}
	if trace[1].GetBase().SpanID != "sp2" {
		t.Fatalf("ordering[1]=%v", trace[1].GetBase().SpanID)
	}
	if trace[1].GetBase().ParentSpanID == nil || *trace[1].GetBase().ParentSpanID != "sp1" {
		t.Fatalf("parent linkage lost: %+v", trace[1].GetBase().ParentSpanID)
	}
}

// ─── Rule: Middleware spans recorded with hook & decision ───────────────────

func TestMiddlewareSpanRecordedInTrace(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	obs.EmitMiddleware(MiddlewareSpan{
		Base: SpanBase{
			SpanID:    "mw1",
			SessionID: sid("s1"),
			TaskID:    tid("t1"),
			Kind:      SpanKindMiddlewareHook,
			StartedAt: ts("2026-05-16T00:00:00Z"),
			EndedAt:   ts("2026-05-16T00:00:00Z"),
			Status:    NewStatusOk(),
		},
		Hook:     middleware.HookBeforeTurn,
		Decision: middleware.DecisionContinueVal(),
	})
	trace, _ := obs.GetTrace(context.Background(), sid("s1"))
	if len(trace) != 1 {
		t.Fatalf("trace len=%d", len(trace))
	}
	if trace[0].GetBase().Kind != SpanKindMiddlewareHook {
		t.Fatalf("kind=%v", trace[0].GetBase().Kind)
	}
}

// ─── Rule: GetSessions filters by outcome ───────────────────────────────────

func TestGetSessionsFiltersByOutcome(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	obs.EmitTurn(turnSpan("good", "sp1", 1, 10, 5))
	obs.EmitTurn(turnSpan("bad", "sp2", 1, 10, 5))
	obs.SetSessionOutcome(sid("good"), guideregistry.NewOutcomeSuccess())
	obs.SetSessionOutcome(sid("bad"), guideregistry.NewOutcomeFailure("x"))
	want := guideregistry.NewOutcomeSuccess()
	got, err := obs.GetSessions(context.Background(), ts("2026-05-16T00:00:00Z"), nil, &want)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("len=%d", len(got))
	}
	if got[0].SessionID != sid("good") {
		t.Fatalf("session=%v", got[0].SessionID)
	}
}

// ─── Rule: GetSessions filters by since timestamp ───────────────────────────

func TestGetSessionsFiltersBySince(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	old := turnSpan("old", "sp1", 1, 10, 5)
	old.Base.StartedAt = ts("2026-01-01T00:00:00Z")
	obs.EmitTurn(old)
	obs.EmitTurn(turnSpan("new", "sp2", 1, 10, 5))
	got, err := obs.GetSessions(context.Background(), ts("2026-05-15T00:00:00Z"), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	ids := map[SessionID]bool{}
	for _, m := range got {
		ids[m.SessionID] = true
	}
	if !ids[sid("new")] {
		t.Fatalf("missing new: %+v", ids)
	}
	if ids[sid("old")] {
		t.Fatalf("should not contain old: %+v", ids)
	}
}

// ─── Rule: passive observer — interface has no behavior-mutating methods ────
//
// Compile-time check: the ObservabilityProvider interface declares only
// EmitX (no return), FlushSession, and GetX accessors. There is no method
// that could change a TurnResult or ToolOutput. We also assert thread
// safety by exercising concurrent emits.

func TestObservabilityProviderIsConcurrencySafe(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	done := make(chan struct{}, 4)
	emit := func(prefix string) {
		for i := 0; i < 25; i++ {
			obs.EmitTurn(turnSpan("s1", prefix+string(rune('a'+i%26)), uint32(i), 1, 1))
		}
		done <- struct{}{}
	}
	go emit("a")
	go emit("b")
	go emit("c")
	go emit("d")
	for i := 0; i < 4; i++ {
		<-done
	}
	m, _ := obs.GetSessionMetrics(context.Background(), sid("s1"))
	if m == nil || m.TotalTurns != 100 {
		t.Fatalf("turns=%v", m)
	}
}

// ─── SpanBase helpers ───────────────────────────────────────────────────────

func TestNewRootAndNewChild(t *testing.T) {
	root := NewRoot(SpanID("r"), sid("s"), tid("t"), SpanKindSession, ts("2026-05-16T00:00:00Z"))
	child := NewChild(SpanID("c"), root, SpanKindTurn, ts("2026-05-16T00:00:01Z"))
	if child.ParentSpanID == nil || *child.ParentSpanID != "r" {
		t.Fatalf("parent=%+v", child.ParentSpanID)
	}
	if child.SessionID != sid("s") {
		t.Fatalf("session=%v", child.SessionID)
	}
	if child.Status.Kind != SpanStatusKindOk {
		t.Fatalf("status=%v", child.Status)
	}
}

func TestSpanBaseFinish(t *testing.T) {
	b := NewRoot("r", sid("s"), tid("t"), SpanKindTurn, ts("2026-05-16T00:00:00Z"))
	b.Finish(ts("2026-05-16T00:00:02Z"), NewStatusError("bad"), 2000)
	if b.DurationMs != 2000 || b.Status.Kind != SpanStatusKindError || b.Status.Message != "bad" {
		t.Fatalf("finish: %+v", b)
	}
}

// ─── Rule: GetSessionMetrics returns nil when nothing recorded ──────────────

func TestGetSessionMetricsReturnsNilWhenEmpty(t *testing.T) {
	obs := NewInMemoryObservabilityProvider()
	m, err := obs.GetSessionMetrics(context.Background(), sid("missing"))
	if err != nil {
		t.Fatal(err)
	}
	if m != nil {
		t.Fatalf("want nil, got %+v", m)
	}
}
