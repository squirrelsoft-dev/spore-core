// Package observability — issue #12 `ObservabilityProvider`: structured
// recording of all harness activity.
//
// Every observable harness operation emits one Span. Spans carry identity
// (session, task, parent span), timing, status, and operation-specific
// payload. Aggregates roll up to SessionMetrics for the improvement loop.
//
// See `docs/harness-engineering-concepts.md` § "ObservabilityProvider" and
// the Rust reference at `rust/crates/spore-core/src/observability.rs`.
//
// Rules enforced
//   - EmitX methods are fire-and-forget — they take no context, return no
//     error, and never block the harness loop. The standard implementation
//     pushes to a sync.Mutex-guarded buffer synchronously.
//   - Every harness operation type has a corresponding EmitX method;
//     nothing is exempt.
//   - CostUSD on TurnSpan is computed at emit time from a PricingTable;
//     the harness does not estimate it. DefaultPricing returns zero cost.
//   - Observability is passive — the interface has no method that
//     affects harness behavior.
//   - GetTrace returns spans in insertion order. ParentSpanID is
//     preserved so the trace analyzer can reconstruct hierarchy.
//   - FlushSession is idempotent — calling it twice for the same
//     session is a no-op the second time; spans remain queryable after.
//
// Implementor notes
//   - SpanBase.NewRoot and SpanBase.NewChild stamp StartedAt and leave
//     EndedAt to the caller via SpanBase.Finish — matches the harness
//     pattern of "open at start, finish at end".
//   - The in-memory backend uses RFC 3339 timestamps so spans compare
//     lexically; production OTLP backends would use nanosecond clocks.
package observability

import (
	"context"
	"sort"
	"sync"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/memory"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/middleware"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/sensor"
)

// ============================================================================
// Re-exported type aliases (so callers don't need to import every sibling
// package just to construct a span).
// ============================================================================

// Timestamp is re-exported from the memory package for wire compatibility.
type Timestamp = memory.Timestamp

// SessionID, TaskID re-exported from sporecore.
type SessionID = sporecore.SessionID

// TaskID re-exported from sporecore.
type TaskID = sporecore.TaskID

// StopReason re-exported from sporecore.
type StopReason = sporecore.StopReason

// GuideID re-exported from guideregistry (matches the Rust GuideId import
// path for this component).
type GuideID = guideregistry.GuideID

// SessionOutcome re-exported from guideregistry.
type SessionOutcome = guideregistry.SessionOutcome

// SensorID, SensorKind, SensorTrigger, SensorOutcome re-exported from sensor.
type SensorID = sensor.SensorID

// SensorKind re-exported from sensor.
type SensorKind = sensor.SensorKind

// SensorTrigger re-exported from sensor.
type SensorTrigger = sensor.SensorTrigger

// SensorOutcome re-exported from sensor.
type SensorOutcome = sensor.SensorOutcome

// HookPoint re-exported from middleware.
type HookPoint = middleware.HookPoint

// MiddlewareDecision re-exported from middleware.
type MiddlewareDecision = middleware.MiddlewareDecision

// ============================================================================
// Identity
// ============================================================================

// SpanID is the stable identifier for a single span.
type SpanID string

// ============================================================================
// SpanKind
// ============================================================================

// SpanKind discriminates the ten observable harness operation types.
type SpanKind string

const (
	SpanKindSession          SpanKind = "session"
	SpanKindTurn             SpanKind = "turn"
	SpanKindToolCall         SpanKind = "tool_call"
	SpanKindSensorEvaluation SpanKind = "sensor_evaluation"
	SpanKindContextAssembly  SpanKind = "context_assembly"
	SpanKindCompaction       SpanKind = "compaction"
	SpanKindMiddlewareHook   SpanKind = "middleware_hook"
	SpanKindGuideSelection   SpanKind = "guide_selection"
	SpanKindMemoryQuery      SpanKind = "memory_query"
	SpanKindMemoryWrite      SpanKind = "memory_write"
)

// ============================================================================
// SpanStatus (tagged via Kind)
// ============================================================================

// SpanStatusKind discriminates SpanStatus variants.
type SpanStatusKind string

const (
	// SpanStatusKindOk — span completed successfully.
	SpanStatusKindOk SpanStatusKind = "ok"
	// SpanStatusKindError — span completed with an error.
	SpanStatusKindError SpanStatusKind = "error"
	// SpanStatusKindHalted — span aborted because the harness halted.
	SpanStatusKindHalted SpanStatusKind = "halted"
)

// SpanStatus is a tagged union. Only the field(s) matching Kind are set.
type SpanStatus struct {
	Kind    SpanStatusKind `json:"kind"`
	Message string         `json:"message,omitempty"`
	Reason  string         `json:"reason,omitempty"`
}

// NewStatusOk returns an Ok status.
func NewStatusOk() SpanStatus { return SpanStatus{Kind: SpanStatusKindOk} }

// NewStatusError returns an Error status.
func NewStatusError(message string) SpanStatus {
	return SpanStatus{Kind: SpanStatusKindError, Message: message}
}

// NewStatusHalted returns a Halted status.
func NewStatusHalted(reason string) SpanStatus {
	return SpanStatus{Kind: SpanStatusKindHalted, Reason: reason}
}

// ============================================================================
// SpanBase
// ============================================================================

// SpanBase is the identity/timing envelope shared by every span payload.
type SpanBase struct {
	SpanID       SpanID     `json:"span_id"`
	ParentSpanID *SpanID    `json:"parent_span_id,omitempty"`
	SessionID    SessionID  `json:"session_id"`
	TaskID       TaskID     `json:"task_id"`
	Kind         SpanKind   `json:"kind"`
	StartedAt    Timestamp  `json:"started_at"`
	EndedAt      Timestamp  `json:"ended_at"`
	DurationMs   uint64     `json:"duration_ms"`
	Status       SpanStatus `json:"status"`
}

// NewRoot stamps a span with no parent. EndedAt mirrors StartedAt until
// Finish is called.
func NewRoot(spanID SpanID, sessionID SessionID, taskID TaskID, kind SpanKind, startedAt Timestamp) SpanBase {
	return SpanBase{
		SpanID:    spanID,
		SessionID: sessionID,
		TaskID:    taskID,
		Kind:      kind,
		StartedAt: startedAt,
		EndedAt:   startedAt,
		Status:    NewStatusOk(),
	}
}

// NewChild stamps a span with parent inherited from the supplied base.
func NewChild(spanID SpanID, parent SpanBase, kind SpanKind, startedAt Timestamp) SpanBase {
	pid := parent.SpanID
	return SpanBase{
		SpanID:       spanID,
		ParentSpanID: &pid,
		SessionID:    parent.SessionID,
		TaskID:       parent.TaskID,
		Kind:         kind,
		StartedAt:    startedAt,
		EndedAt:      startedAt,
		Status:       NewStatusOk(),
	}
}

// Finish records the terminal timestamp, status, and duration.
func (b *SpanBase) Finish(endedAt Timestamp, status SpanStatus, durationMs uint64) {
	b.EndedAt = endedAt
	b.Status = status
	b.DurationMs = durationMs
}

// ============================================================================
// Span payload types
// ============================================================================

// TurnSpan is one agent turn — tokens, cost, stop reason.
type TurnSpan struct {
	Base               SpanBase   `json:"base"`
	TurnNumber         uint32     `json:"turn_number"`
	InputTokens        uint32     `json:"input_tokens"`
	OutputTokens       uint32     `json:"output_tokens"`
	CacheReadTokens    *uint32    `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens   *uint32    `json:"cache_write_tokens,omitempty"`
	CostUSD            float64    `json:"cost_usd"`
	StopReason         StopReason `json:"stop_reason"`
	ToolCallsRequested uint32     `json:"tool_calls_requested"`
}

// ToolCallSpan is one tool dispatch.
type ToolCallSpan struct {
	Base                SpanBase `json:"base"`
	ToolName            string   `json:"tool_name"`
	CallID              string   `json:"call_id"`
	ParametersSizeBytes uint64   `json:"parameters_size_bytes"`
	OutputSizeBytes     uint64   `json:"output_size_bytes"`
	Truncated           bool     `json:"truncated"`
	SandboxMode         string   `json:"sandbox_mode"`
	SandboxViolations   []string `json:"sandbox_violations"`
}

// SensorSpan is one sensor evaluation.
type SensorSpan struct {
	Base       SpanBase      `json:"base"`
	SensorID   SensorID      `json:"sensor_id"`
	SensorKind SensorKind    `json:"sensor_kind"`
	Trigger    SensorTrigger `json:"trigger"`
	Outcome    SensorOutcome `json:"outcome"`
	Fired      bool          `json:"fired"`
}

// ============================================================================
// ContextOperation (tagged via Kind)
// ============================================================================

// ContextOperationKind discriminates ContextOperation variants.
type ContextOperationKind string

const (
	// ContextOpKindAssembly — initial context assembly.
	ContextOpKindAssembly ContextOperationKind = "assembly"
	// ContextOpKindToolResultAppended — tool result appended to context.
	ContextOpKindToolResultAppended ContextOperationKind = "tool_result_appended"
	// ContextOpKindCompaction — context window compacted.
	ContextOpKindCompaction ContextOperationKind = "compaction"
	// ContextOpKindSkillInjected — a guide/skill was injected.
	ContextOpKindSkillInjected ContextOperationKind = "skill_injected"
)

// ContextOperation is a tagged union. Only the field(s) matching Kind are set.
type ContextOperation struct {
	Kind ContextOperationKind `json:"kind"`
	// Assembly
	GuidesLoaded      uint32 `json:"guides_loaded,omitempty"`
	MemoryItemsLoaded uint32 `json:"memory_items_loaded,omitempty"`
	ToolsLoaded       uint32 `json:"tools_loaded,omitempty"`
	// ToolResultAppended
	ToolName  string `json:"tool_name,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
	// Compaction
	MessagesRemoved uint32 `json:"messages_removed,omitempty"`
	TokensReclaimed uint32 `json:"tokens_reclaimed,omitempty"`
	// SkillInjected
	GuideID GuideID `json:"guide_id,omitempty"`
}

// NewContextOpAssembly returns an Assembly operation.
func NewContextOpAssembly(guides, memItems, tools uint32) ContextOperation {
	return ContextOperation{
		Kind:              ContextOpKindAssembly,
		GuidesLoaded:      guides,
		MemoryItemsLoaded: memItems,
		ToolsLoaded:       tools,
	}
}

// NewContextOpToolResultAppended returns a ToolResultAppended operation.
func NewContextOpToolResultAppended(toolName string, truncated bool) ContextOperation {
	return ContextOperation{
		Kind:      ContextOpKindToolResultAppended,
		ToolName:  toolName,
		Truncated: truncated,
	}
}

// NewContextOpCompaction returns a Compaction operation.
func NewContextOpCompaction(messagesRemoved, tokensReclaimed uint32) ContextOperation {
	return ContextOperation{
		Kind:            ContextOpKindCompaction,
		MessagesRemoved: messagesRemoved,
		TokensReclaimed: tokensReclaimed,
	}
}

// NewContextOpSkillInjected returns a SkillInjected operation.
func NewContextOpSkillInjected(guideID GuideID) ContextOperation {
	return ContextOperation{Kind: ContextOpKindSkillInjected, GuideID: guideID}
}

// ContextSpan is one context operation.
type ContextSpan struct {
	Base              SpanBase         `json:"base"`
	Operation         ContextOperation `json:"operation"`
	TokensBefore      uint32           `json:"tokens_before"`
	TokensAfter       uint32           `json:"tokens_after"`
	UtilizationBefore float32          `json:"utilization_before"`
	UtilizationAfter  float32          `json:"utilization_after"`
}

// MiddlewareSpan is one middleware hook firing.
type MiddlewareSpan struct {
	Base     SpanBase           `json:"base"`
	Hook     HookPoint          `json:"hook"`
	Decision MiddlewareDecision `json:"decision"`
}

// ============================================================================
// Span interface (for GetTrace's heterogeneous return)
// ============================================================================

// Span is the common surface used by the trace analyzer to reconstruct
// hierarchy regardless of payload type.
type Span interface {
	GetBase() SpanBase
}

// GetBase implements Span.
func (s TurnSpan) GetBase() SpanBase { return s.Base }

// GetBase implements Span.
func (s ToolCallSpan) GetBase() SpanBase { return s.Base }

// GetBase implements Span.
func (s SensorSpan) GetBase() SpanBase { return s.Base }

// GetBase implements Span.
func (s ContextSpan) GetBase() SpanBase { return s.Base }

// GetBase implements Span.
func (s MiddlewareSpan) GetBase() SpanBase { return s.Base }

// ============================================================================
// SessionMetrics
// ============================================================================

// SessionMetrics is the aggregated roll-up surfaced by GetSessionMetrics and
// GetSessions for the improvement loop.
type SessionMetrics struct {
	SessionID         SessionID      `json:"session_id"`
	TaskID            TaskID         `json:"task_id"`
	TotalTurns        uint32         `json:"total_turns"`
	TotalInputTokens  uint32         `json:"total_input_tokens"`
	TotalOutputTokens uint32         `json:"total_output_tokens"`
	TotalCostUSD      float64        `json:"total_cost_usd"`
	TotalDurationMs   uint64         `json:"total_duration_ms"`
	ToolCalls         uint32         `json:"tool_calls"`
	SensorFires       uint32         `json:"sensor_fires"`
	SensorHalts       uint32         `json:"sensor_halts"`
	Compactions       uint32         `json:"compactions"`
	Outcome           SessionOutcome `json:"outcome"`
	GuidesUsed        []GuideID      `json:"guides_used"`
}

// ============================================================================
// PricingTable
// ============================================================================

// PricingTable is the provider-specific token → USD lookup, injected at
// construction so CostUSD is a first-class span field (per spec).
type PricingTable struct {
	// InputPerMillion is USD per 1M input tokens.
	InputPerMillion float64 `json:"input_per_million"`
	// OutputPerMillion is USD per 1M output tokens.
	OutputPerMillion float64 `json:"output_per_million"`
	// CacheReadPerMillion is USD per 1M cache-read tokens.
	CacheReadPerMillion float64 `json:"cache_read_per_million"`
	// CacheWritePerMillion is USD per 1M cache-write tokens.
	CacheWritePerMillion float64 `json:"cache_write_per_million"`
}

// DefaultPricing is the conservative zero-cost default. Production callers
// inject a real table.
func DefaultPricing() PricingTable { return PricingTable{} }

// CostFor computes USD cost for a turn. Cache tokens count against cache
// pricing, never input/output pricing.
func (p PricingTable) CostFor(input, output uint32, cacheRead, cacheWrite *uint32) float64 {
	per := func(perMillion float64) float64 { return perMillion / 1_000_000.0 }
	var cr, cw uint32
	if cacheRead != nil {
		cr = *cacheRead
	}
	if cacheWrite != nil {
		cw = *cacheWrite
	}
	return per(p.InputPerMillion)*float64(input) +
		per(p.OutputPerMillion)*float64(output) +
		per(p.CacheReadPerMillion)*float64(cr) +
		per(p.CacheWritePerMillion)*float64(cw)
}

// ============================================================================
// Interface
// ============================================================================

// ObservabilityProvider is the structured-observability surface.
//
// All EmitX methods are fire-and-forget: they take no context, return no
// error, and must never block the harness loop. Implementations buffer
// internally and flush asynchronously via FlushSession.
//
// Observability is passive: this interface has no method that affects
// harness behavior — calling EmitX cannot change a TurnResult or a
// ToolOutput.
type ObservabilityProvider interface {
	EmitTurn(span TurnSpan)
	EmitToolCall(span ToolCallSpan)
	EmitSensor(span SensorSpan)
	EmitContext(span ContextSpan)
	EmitMiddleware(span MiddlewareSpan)

	// FlushSession is idempotent — calling it twice for the same session
	// is a no-op the second time. Spans remain queryable after flush.
	FlushSession(ctx context.Context, sessionID SessionID) error

	// GetSessionMetrics returns the aggregated roll-up for one session, or
	// nil if neither spans nor an outcome have been recorded for it.
	GetSessionMetrics(ctx context.Context, sessionID SessionID) (*SessionMetrics, error)

	// GetSessions filters by `since` (lexical timestamp ≥ since on
	// started_at) and optionally by outcome and domain (domain not yet
	// modelled — accepted for forward compatibility).
	GetSessions(ctx context.Context, since Timestamp, domain *string, outcome *SessionOutcome) ([]SessionMetrics, error)

	// GetTrace returns the full span list for a session in insertion
	// order; trace analyzer reconstructs hierarchy via Base.ParentSpanID.
	GetTrace(ctx context.Context, sessionID SessionID) ([]Span, error)
}

// ============================================================================
// InMemoryObservabilityProvider
// ============================================================================

// InMemoryObservabilityProvider is the standard in-memory backend used for
// tests and short-lived processes. Production OTLP / JSONL backends live in
// sibling packages and implement the same interface.
type InMemoryObservabilityProvider struct {
	mu          sync.Mutex
	turns       []TurnSpan
	toolCalls   []ToolCallSpan
	sensors     []SensorSpan
	contexts    []ContextSpan
	middlewares []MiddlewareSpan
	// Per-session insertion-ordered (kind, spanID) feed for GetTrace.
	traceOrder map[SessionID][]traceEntry
	flushed    map[SessionID]bool
	// Per-session terminal outcome, recorded by the harness via
	// SetSessionOutcome after AfterSession.
	outcomes map[SessionID]SessionOutcome
	// Per-session guides used, populated by the harness via
	// RecordGuidesUsed at session start.
	guidesUsed map[SessionID][]GuideID
}

type traceEntry struct {
	kind   SpanKind
	spanID SpanID
}

// NewInMemoryObservabilityProvider returns an empty provider.
func NewInMemoryObservabilityProvider() *InMemoryObservabilityProvider {
	return &InMemoryObservabilityProvider{
		traceOrder: make(map[SessionID][]traceEntry),
		flushed:    make(map[SessionID]bool),
		outcomes:   make(map[SessionID]SessionOutcome),
		guidesUsed: make(map[SessionID][]GuideID),
	}
}

// SetSessionOutcome records the terminal outcome for a session so
// SessionMetrics can surface it. The harness calls this once, after
// AfterSession.
func (p *InMemoryObservabilityProvider) SetSessionOutcome(sessionID SessionID, outcome SessionOutcome) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.outcomes[sessionID] = outcome
}

// RecordGuidesUsed records the guides selected for a session. Called once
// at session start.
func (p *InMemoryObservabilityProvider) RecordGuidesUsed(sessionID SessionID, guides []GuideID) {
	p.mu.Lock()
	defer p.mu.Unlock()
	cp := make([]GuideID, len(guides))
	copy(cp, guides)
	p.guidesUsed[sessionID] = cp
}

// pushOrder appends an entry to the per-session trace feed. Caller holds mu.
func (p *InMemoryObservabilityProvider) pushOrder(sid SessionID, kind SpanKind, id SpanID) {
	p.traceOrder[sid] = append(p.traceOrder[sid], traceEntry{kind: kind, spanID: id})
}

// EmitTurn implements ObservabilityProvider.
func (p *InMemoryObservabilityProvider) EmitTurn(span TurnSpan) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pushOrder(span.Base.SessionID, SpanKindTurn, span.Base.SpanID)
	p.turns = append(p.turns, span)
}

// EmitToolCall implements ObservabilityProvider.
func (p *InMemoryObservabilityProvider) EmitToolCall(span ToolCallSpan) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pushOrder(span.Base.SessionID, SpanKindToolCall, span.Base.SpanID)
	p.toolCalls = append(p.toolCalls, span)
}

// EmitSensor implements ObservabilityProvider.
func (p *InMemoryObservabilityProvider) EmitSensor(span SensorSpan) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pushOrder(span.Base.SessionID, SpanKindSensorEvaluation, span.Base.SpanID)
	p.sensors = append(p.sensors, span)
}

// EmitContext implements ObservabilityProvider.
//
// Routes the trace-feed kind based on the operation: Compaction operations
// surface as SpanKindCompaction; everything else as SpanKindContextAssembly.
func (p *InMemoryObservabilityProvider) EmitContext(span ContextSpan) {
	p.mu.Lock()
	defer p.mu.Unlock()
	kind := SpanKindContextAssembly
	if span.Operation.Kind == ContextOpKindCompaction {
		kind = SpanKindCompaction
	}
	p.pushOrder(span.Base.SessionID, kind, span.Base.SpanID)
	p.contexts = append(p.contexts, span)
}

// EmitMiddleware implements ObservabilityProvider.
func (p *InMemoryObservabilityProvider) EmitMiddleware(span MiddlewareSpan) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pushOrder(span.Base.SessionID, SpanKindMiddlewareHook, span.Base.SpanID)
	p.middlewares = append(p.middlewares, span)
}

// FlushSession implements ObservabilityProvider. Idempotent: the second
// invocation for a given session is a no-op. Spans remain queryable after.
func (p *InMemoryObservabilityProvider) FlushSession(_ context.Context, sessionID SessionID) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.flushed[sessionID] = true
	return nil
}

// GetSessionMetrics implements ObservabilityProvider. Returns nil if no
// spans nor outcome have been recorded for the session.
func (p *InMemoryObservabilityProvider) GetSessionMetrics(_ context.Context, sessionID SessionID) (*SessionMetrics, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.computeMetricsLocked(sessionID), nil
}

func (p *InMemoryObservabilityProvider) computeMetricsLocked(sessionID SessionID) *SessionMetrics {
	var turns []TurnSpan
	for _, t := range p.turns {
		if t.Base.SessionID == sessionID {
			turns = append(turns, t)
		}
	}
	_, hasOutcome := p.outcomes[sessionID]
	if len(turns) == 0 && !hasOutcome {
		return nil
	}
	var taskID TaskID
	if len(turns) > 0 {
		taskID = turns[0].Base.TaskID
	}
	var input, output uint32
	var cost float64
	var duration uint64
	for _, t := range turns {
		input += t.InputTokens
		output += t.OutputTokens
		cost += t.CostUSD
		duration += t.Base.DurationMs
	}
	var toolCalls uint32
	for _, c := range p.toolCalls {
		if c.Base.SessionID == sessionID {
			toolCalls++
			duration += c.Base.DurationMs
		}
	}
	var sensorFires, sensorHalts uint32
	for _, s := range p.sensors {
		if s.Base.SessionID != sessionID {
			continue
		}
		if s.Fired {
			sensorFires++
		}
		if s.Outcome == sensor.OutcomeHalt {
			sensorHalts++
		}
	}
	var compactions uint32
	for _, c := range p.contexts {
		if c.Base.SessionID == sessionID && c.Operation.Kind == ContextOpKindCompaction {
			compactions++
		}
	}
	outcome, ok := p.outcomes[sessionID]
	if !ok {
		outcome = guideregistry.NewOutcomePartial()
	}
	var guides []GuideID
	if g, ok := p.guidesUsed[sessionID]; ok {
		guides = append([]GuideID(nil), g...)
	}
	return &SessionMetrics{
		SessionID:         sessionID,
		TaskID:            taskID,
		TotalTurns:        uint32(len(turns)),
		TotalInputTokens:  input,
		TotalOutputTokens: output,
		TotalCostUSD:      cost,
		TotalDurationMs:   duration,
		ToolCalls:         toolCalls,
		SensorFires:       sensorFires,
		SensorHalts:       sensorHalts,
		Compactions:       compactions,
		Outcome:           outcome,
		GuidesUsed:        guides,
	}
}

// GetSessions implements ObservabilityProvider. Filters by since (lexical
// timestamp ≥) and optionally by outcome. Domain is accepted but not yet
// modelled.
func (p *InMemoryObservabilityProvider) GetSessions(_ context.Context, since Timestamp, _ *string, outcome *SessionOutcome) ([]SessionMetrics, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	// Collect distinct session ids that have at least one turn with
	// started_at ≥ since.
	seen := make(map[SessionID]bool)
	var ids []SessionID
	for _, t := range p.turns {
		if string(t.Base.StartedAt) < string(since) {
			continue
		}
		if seen[t.Base.SessionID] {
			continue
		}
		seen[t.Base.SessionID] = true
		ids = append(ids, t.Base.SessionID)
	}
	sort.Slice(ids, func(i, j int) bool { return string(ids[i]) < string(ids[j]) })
	out := make([]SessionMetrics, 0, len(ids))
	for _, sid := range ids {
		m := p.computeMetricsLocked(sid)
		if m == nil {
			continue
		}
		if outcome != nil && m.Outcome.Kind != outcome.Kind {
			continue
		}
		out = append(out, *m)
	}
	return out, nil
}

// GetTrace implements ObservabilityProvider. Returns spans in insertion
// order. ParentSpanID linkage is preserved through Base.
func (p *InMemoryObservabilityProvider) GetTrace(_ context.Context, sessionID SessionID) ([]Span, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	order, ok := p.traceOrder[sessionID]
	if !ok {
		return nil, nil
	}
	out := make([]Span, 0, len(order))
	for _, entry := range order {
		switch entry.kind {
		case SpanKindTurn:
			for _, t := range p.turns {
				if t.Base.SpanID == entry.spanID {
					out = append(out, t)
					break
				}
			}
		case SpanKindToolCall:
			for _, c := range p.toolCalls {
				if c.Base.SpanID == entry.spanID {
					out = append(out, c)
					break
				}
			}
		case SpanKindSensorEvaluation:
			for _, s := range p.sensors {
				if s.Base.SpanID == entry.spanID {
					out = append(out, s)
					break
				}
			}
		case SpanKindContextAssembly, SpanKindCompaction:
			for _, c := range p.contexts {
				if c.Base.SpanID == entry.spanID {
					out = append(out, c)
					break
				}
			}
		case SpanKindMiddlewareHook:
			for _, m := range p.middlewares {
				if m.Base.SpanID == entry.spanID {
					out = append(out, m)
					break
				}
			}
		}
	}
	return out, nil
}

// Compile-time interface check.
var _ ObservabilityProvider = (*InMemoryObservabilityProvider)(nil)
