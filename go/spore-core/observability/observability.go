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
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"

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

// TerminalOutcome re-exported from sporecore (issue #80): the harness's
// 3-state terminal outcome passed to the adapter's SetSessionOutcome.
type TerminalOutcome = sporecore.TerminalOutcome

// TerminalSuccess, TerminalFailure, TerminalEscalated re-exported from
// sporecore for ergonomic use by observability consumers and tests.
const (
	TerminalSuccess   = sporecore.TerminalSuccess
	TerminalFailure   = sporecore.TerminalFailure
	TerminalEscalated = sporecore.TerminalEscalated
)

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
	// SpanKindPatch — emitted by PatchToolCallsMiddleware whenever it mutates
	// a tool call (issue #28). Always carries a PatchSpan at SpanLevelWarn.
	SpanKindPatch SpanKind = "patch"
	// SpanKindWarn — emitted by the harness compaction loop when a summary is
	// accepted despite failing verification (issue #46). Always carries a
	// WarnSpan at SpanLevelWarn.
	SpanKindWarn SpanKind = "warn"
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
	// OutputText is the model's output text for this turn (issue #64). Captured
	// only when ContentCaptureConfig.Enabled; nil keeps the line pre-#64-identical
	// (omitempty drops the key entirely).
	OutputText *GenAiMessage `json:"output_text,omitempty"`
	// ToolCalls are the tool calls the model requested this turn (issue #64).
	// Captured only when content capture is enabled; nil keeps the line
	// pre-#64-identical.
	ToolCalls []ToolCallContent `json:"tool_calls,omitempty"`
	// InputMessages is the assembled INPUT prompt the model saw this turn,
	// ordered system-first then history order (issue #64). Captured only when
	// content capture is enabled; nil (omitempty) keeps the line
	// pre-#64-identical.
	InputMessages []GenAiMessage `json:"input_messages,omitempty"`
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
	// Arguments is the tool-call arguments (issue #64). Captured only when
	// content capture is enabled; nil keeps the line pre-#64-identical.
	Arguments *ToolCallContent `json:"arguments,omitempty"`
	// Result is the tool result body (issue #64). Captured only when content
	// capture is enabled; nil keeps the line pre-#64-identical.
	Result *ToolResultContent `json:"result,omitempty"`
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
// Patch observability (issue #28)
// ============================================================================
//
// PatchToolCallsMiddleware is an always-on, highest-priority BeforeTool action
// mutator that silently rewrites malformed or dangling tool calls before the
// sandbox and sensors see them. An always-on mutator with no observability is
// a footgun: the trace would show the patched call as if the model had sent
// it. Issue #28 closes that gap.
//
// Types added here:
//   - SpanLevel  — severity tag; patch spans are ALWAYS Warn, never Info.
//     Distinct from SpanStatus so the surrounding trace stays Ok while the
//     patch event itself is flagged.
//   - PatchType  — what kind of repair happened (MalformedJson,
//     DanglingToolCall, ParameterCoercion), a tagged union keyed by "kind".
//   - PatchSpan  — the full event: identity (Base), call id and tool name,
//     the original parameters as the model sent them, the patched parameters
//     that were actually dispatched, the classified PatchType, and the
//     hardcoded Level Warn.
//
// Interface method added: ObservabilityProvider.EmitPatch — fire-and-forget,
// synchronous, mirrors the other EmitX methods.

// SpanLevel is the severity of an emitted span. Patch spans are always
// SpanLevelWarn per issue #28; this enum keeps the level orthogonal to
// SpanStatus so a successful (Ok) trace can still surface warn-level patch
// events.
type SpanLevel string

const (
	// SpanLevelInfo — informational severity.
	SpanLevelInfo SpanLevel = "info"
	// SpanLevelWarn — warning severity. Every patch span uses this.
	SpanLevelWarn SpanLevel = "warn"
)

// PatchTypeKind discriminates PatchType variants.
type PatchTypeKind string

const (
	// PatchKindMalformedJSON — the raw tool-call arguments failed to parse as
	// JSON; a repair was attempted. Error is the parse error recovered from.
	PatchKindMalformedJSON PatchTypeKind = "malformed_json"
	// PatchKindDanglingToolCall — the call was structurally incomplete (e.g.
	// empty tool name) and was completed with defaults. Reason explains what
	// was missing.
	PatchKindDanglingToolCall PatchTypeKind = "dangling_tool_call"
	// PatchKindParameterCoercion — a parameter value was coerced from one type
	// to another to satisfy the tool schema.
	PatchKindParameterCoercion PatchTypeKind = "parameter_coercion"
)

// PatchType classifies a tool-call patch. It is a tagged union keyed by Kind;
// only the field(s) matching Kind are meaningful.
type PatchType struct {
	Kind PatchTypeKind `json:"kind"`
	// MalformedJson
	Error string `json:"error,omitempty"`
	// DanglingToolCall
	Reason string `json:"reason,omitempty"`
	// ParameterCoercion
	Field string `json:"field,omitempty"`
	From  string `json:"from,omitempty"`
	To    string `json:"to,omitempty"`
}

// NewPatchMalformedJSON returns a MalformedJson patch type.
func NewPatchMalformedJSON(parseErr string) PatchType {
	return PatchType{Kind: PatchKindMalformedJSON, Error: parseErr}
}

// NewPatchDanglingToolCall returns a DanglingToolCall patch type.
func NewPatchDanglingToolCall(reason string) PatchType {
	return PatchType{Kind: PatchKindDanglingToolCall, Reason: reason}
}

// NewPatchParameterCoercion returns a ParameterCoercion patch type.
func NewPatchParameterCoercion(field, from, to string) PatchType {
	return PatchType{Kind: PatchKindParameterCoercion, Field: field, From: from, To: to}
}

// PatchSpan is one observability event per tool-call patch (issue #28). It
// carries both the original parameters (what the model sent) and the patched
// parameters (what was dispatched) so the trace shows the diff, never just the
// patched call. Level is always SpanLevelWarn; construct via NewPatchSpan.
type PatchSpan struct {
	Base               SpanBase        `json:"base"`
	CallID             string          `json:"call_id"`
	ToolName           string          `json:"tool_name"`
	OriginalParameters json.RawMessage `json:"original_parameters"`
	PatchedParameters  json.RawMessage `json:"patched_parameters"`
	PatchType          PatchType       `json:"patch_type"`
	Level              SpanLevel       `json:"level"`
}

// NewPatchSpan builds a patch span. Level is forced to SpanLevelWarn; callers
// cannot emit an Info-level patch.
func NewPatchSpan(base SpanBase, callID, toolName string, original, patched json.RawMessage, patchType PatchType) PatchSpan {
	return PatchSpan{
		Base:               base,
		CallID:             callID,
		ToolName:           toolName,
		OriginalParameters: original,
		PatchedParameters:  patched,
		PatchType:          patchType,
		Level:              SpanLevelWarn,
	}
}

// ============================================================================
// WarnEvent / WarnSpan (issue #46)
// ============================================================================
//
// The harness compaction loop verifies every agent-produced summary with a
// CompactionVerifier before accepting it. After MaxCompactionAttempts failed
// verifications the harness accepts the summary anyway — a blocked compaction
// is worse than an imperfect one — and emits exactly one warn-level WarnSpan
// carrying WarnEventCompactionVerificationFailed{missing_items, accepted_anyway:
// true}.
//
// Rules (mirrored by the harness loop tests):
//   W1  a successful (or first-try-passing) compaction emits NO warn span.
//   W2  exhausting attempts emits EXACTLY ONE warn span carrying the final
//       missing_items and accepted_anyway = true.
//   W3  SessionMetrics.CompactionVerificationFailures counts these spans for
//       the session (mirrors how Compactions is derived from spans).
//   W4  EmitWarn is exposed as an OPTIONAL WarnEmitter interface the harness
//       adapter type-asserts, so providers predating #46 keep compiling and
//       behave unchanged (Go equivalent of Rust's default-bodied emit_warn).

// WarnEventKind discriminates WarnEvent variants. The "warn" JSON tag mirrors
// the Rust enum's #[serde(tag = "warn")].
type WarnEventKind string

const (
	// WarnKindCompactionVerificationFailed — a compaction summary failed
	// verification on every attempt and was accepted as-is (issue #46).
	WarnKindCompactionVerificationFailed WarnEventKind = "compaction_verification_failed"
	// WarnKindHillClimbingIteration — one iteration of a HillClimbing loop
	// strategy run (issue #60). Emitted fire-and-forget after each iteration's
	// metric evaluation so the run is traceable per-iteration.
	WarnKindHillClimbingIteration WarnEventKind = "hill_climbing_iteration"
)

// WarnEvent is a warn-level, fire-and-forget observability event. It is a
// tagged union keyed by Kind (wire tag "warn"); only the field(s) matching Kind
// are meaningful. The enum-as-event shape mirrors PatchSpan (always Warn) but
// keeps warns that are not tied to a single tool call in their own type.
type WarnEvent struct {
	Kind WarnEventKind `json:"warn"`
	// CompactionVerificationFailed fields.
	MissingItems   []string `json:"missing_items,omitempty"`
	AcceptedAnyway bool     `json:"accepted_anyway,omitempty"`
	// HillClimbingIteration fields (issue #60). MetricValue/Delta are nil on
	// crashed/timeout iterations (no comparable metric); Delta is also nil for
	// the baseline iteration. Status is the snake_case IterationStatus string.
	Iteration   uint32   `json:"iteration,omitempty"`
	MetricValue *float64 `json:"metric_value,omitempty"`
	Delta       *float64 `json:"delta,omitempty"`
	Status      string   `json:"status,omitempty"`
	Reverted    bool     `json:"reverted,omitempty"`
}

// NewWarnCompactionVerificationFailed builds a CompactionVerificationFailed
// warn event. accepted_anyway is always true for this variant (the harness
// never blocks on compaction).
func NewWarnCompactionVerificationFailed(missingItems []string, acceptedAnyway bool) WarnEvent {
	if missingItems == nil {
		missingItems = []string{}
	}
	return WarnEvent{
		Kind:           WarnKindCompactionVerificationFailed,
		MissingItems:   missingItems,
		AcceptedAnyway: acceptedAnyway,
	}
}

// NewWarnHillClimbingIteration builds a HillClimbingIteration warn event (issue
// #60). metricValue/delta are nil-passed for crashed/timeout iterations (and
// delta nil for the baseline).
func NewWarnHillClimbingIteration(iteration uint32, metricValue, delta *float64, status string, reverted bool) WarnEvent {
	return WarnEvent{
		Kind:        WarnKindHillClimbingIteration,
		Iteration:   iteration,
		MetricValue: metricValue,
		Delta:       delta,
		Status:      status,
		Reverted:    reverted,
	}
}

// WarnSpan is one warn-level observability span (issue #46). It carries a
// SpanBase for trace correlation, the classified WarnEvent, and a hardcoded
// Level SpanLevelWarn. Construct via NewWarnSpan.
type WarnSpan struct {
	Base  SpanBase  `json:"base"`
	Event WarnEvent `json:"event"`
	Level SpanLevel `json:"level"`
}

// NewWarnSpan builds a warn span. Level is forced to SpanLevelWarn.
func NewWarnSpan(base SpanBase, event WarnEvent) WarnSpan {
	return WarnSpan{Base: base, Event: event, Level: SpanLevelWarn}
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

// GetBase implements Span.
func (s PatchSpan) GetBase() SpanBase { return s.Base }

// GetBase implements Span.
func (s WarnSpan) GetBase() SpanBase { return s.Base }

// ============================================================================
// SessionMetrics
// ============================================================================

// SessionMetrics is the aggregated roll-up surfaced by GetSessionMetrics and
// GetSessions for the improvement loop.
type SessionMetrics struct {
	SessionID         SessionID `json:"session_id"`
	TaskID            TaskID    `json:"task_id"`
	TotalTurns        uint32    `json:"total_turns"`
	TotalInputTokens  uint32    `json:"total_input_tokens"`
	TotalOutputTokens uint32    `json:"total_output_tokens"`
	TotalCostUSD      float64   `json:"total_cost_usd"`
	TotalDurationMs   uint64    `json:"total_duration_ms"`
	ToolCalls         uint32    `json:"tool_calls"`
	SensorFires       uint32    `json:"sensor_fires"`
	SensorHalts       uint32    `json:"sensor_halts"`
	Compactions       uint32    `json:"compactions"`
	// CompactionVerificationFailures is the number of compactions whose summary
	// failed verification on every attempt and was accepted anyway (issue #46).
	// Derived by counting WarnSpans carrying WarnEventCompactionVerificationFailed,
	// mirroring how Compactions is derived from compaction spans.
	CompactionVerificationFailures uint32         `json:"compaction_verification_failures"`
	Outcome                        SessionOutcome `json:"outcome"`
	GuidesUsed                     []GuideID      `json:"guides_used"`
	// PatchCount is the number of tool-call patches in the session (issue #28).
	PatchCount uint32 `json:"patch_count"`
	// PatchRate is PatchCount / ToolCalls. 0.0 when there are no tool calls
	// (divide-by-zero guard).
	PatchRate float32 `json:"patch_rate"`
	// PatchesByTool breaks the patch count down per tool name.
	PatchesByTool map[string]uint32 `json:"patches_by_tool"`
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
	// EmitPatch records a warn-level tool-call patch event (issue #28).
	// Fire-and-forget like the other EmitX methods.
	EmitPatch(span PatchSpan)

	// SetSessionOutcome records the terminal outcome for a session so
	// SessionMetrics can surface it. The harness calls this once, after the
	// terminal turn (mirrors the Rust trait's set_session_outcome).
	SetSessionOutcome(sessionID SessionID, outcome SessionOutcome)

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

	// ListUnflushedSessions returns the session ids that have a durable
	// outbox (trace.jsonl) but no .flushed marker (issue #33). Backends
	// without a durable outbox return an empty slice.
	ListUnflushedSessions(ctx context.Context) ([]SessionID, error)

	// CleanupSession deletes a session's durable outbox directory (issue
	// #33). It returns an error matching ErrSessionNotFound (via errors.Is)
	// when the session has no outbox. The provider NEVER auto-deletes.
	// Backends without a durable outbox treat every session as not found.
	CleanupSession(ctx context.Context, sessionID SessionID) error
}

// WarnEmitter is the OPTIONAL warn-emission surface (issue #46). It is kept
// off the core ObservabilityProvider interface deliberately: Go interfaces
// force every implementer to satisfy all methods, so adding EmitWarn there
// would break any provider predating #46. Instead the harness adapter
// type-asserts its wrapped provider to WarnEmitter and only emits a warn span
// when the provider implements it — the Go equivalent of the Rust reference's
// default-no-op emit_warn. The two standard providers (InMemory, Outbox) do
// implement it; a bare provider that does not is simply skipped.
type WarnEmitter interface {
	// EmitWarn records a warn-level event. Fire-and-forget like the other
	// EmitX methods.
	EmitWarn(span WarnSpan)
}

// ErrSessionNotFound is returned by CleanupSession when the requested
// session has no durable outbox directory (issue #33). Match with
// errors.Is; SessionNotFoundError carries the offending session id.
var ErrSessionNotFound = errors.New("observability: session not found")

// SessionNotFoundError is the typed error returned by CleanupSession for a
// missing session. It wraps ErrSessionNotFound so errors.Is matches.
type SessionNotFoundError struct {
	SessionID SessionID
}

// Error implements error.
func (e *SessionNotFoundError) Error() string {
	return fmt.Sprintf("observability: session not found: %s", e.SessionID)
}

// Unwrap lets errors.Is(err, ErrSessionNotFound) match.
func (e *SessionNotFoundError) Unwrap() error { return ErrSessionNotFound }

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
	patches     []PatchSpan
	warns       []WarnSpan
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

// EmitPatch implements ObservabilityProvider.
func (p *InMemoryObservabilityProvider) EmitPatch(span PatchSpan) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pushOrder(span.Base.SessionID, SpanKindPatch, span.Base.SpanID)
	p.patches = append(p.patches, span)
}

// EmitWarn implements WarnEmitter (issue #46). Fire-and-forget.
func (p *InMemoryObservabilityProvider) EmitWarn(span WarnSpan) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pushOrder(span.Base.SessionID, SpanKindWarn, span.Base.SpanID)
	p.warns = append(p.warns, span)
}

// WarnSpans returns all recorded warn spans for a session, in insertion order
// (issue #46).
func (p *InMemoryObservabilityProvider) WarnSpans(sessionID SessionID) []WarnSpan {
	p.mu.Lock()
	defer p.mu.Unlock()
	var out []WarnSpan
	for _, s := range p.warns {
		if s.Base.SessionID == sessionID {
			out = append(out, s)
		}
	}
	return out
}

// PatchSpans returns all recorded patch spans for a session, in insertion
// order (issue #28). Lets callers inspect the original/patched diff and
// classified PatchType without reconstructing them from the trace.
func (p *InMemoryObservabilityProvider) PatchSpans(sessionID SessionID) []PatchSpan {
	p.mu.Lock()
	defer p.mu.Unlock()
	var out []PatchSpan
	for _, s := range p.patches {
		if s.Base.SessionID == sessionID {
			out = append(out, s)
		}
	}
	return out
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
	var compactionVerificationFailures uint32
	for _, w := range p.warns {
		if w.Base.SessionID == sessionID && w.Event.Kind == WarnKindCompactionVerificationFailed {
			compactionVerificationFailures++
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
	var patchCount uint32
	patchesByTool := make(map[string]uint32)
	for _, pt := range p.patches {
		if pt.Base.SessionID != sessionID {
			continue
		}
		patchCount++
		patchesByTool[pt.ToolName]++
	}
	// Divide-by-zero guard: denominator is all tool-call spans.
	var patchRate float32
	if toolCalls != 0 {
		patchRate = float32(patchCount) / float32(toolCalls)
	}
	return &SessionMetrics{
		SessionID:                      sessionID,
		TaskID:                         taskID,
		TotalTurns:                     uint32(len(turns)),
		TotalInputTokens:               input,
		TotalOutputTokens:              output,
		TotalCostUSD:                   cost,
		TotalDurationMs:                duration,
		ToolCalls:                      toolCalls,
		SensorFires:                    sensorFires,
		SensorHalts:                    sensorHalts,
		Compactions:                    compactions,
		CompactionVerificationFailures: compactionVerificationFailures,
		Outcome:                        outcome,
		GuidesUsed:                     guides,
		PatchCount:                     patchCount,
		PatchRate:                      patchRate,
		PatchesByTool:                  patchesByTool,
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
		case SpanKindPatch:
			for _, pt := range p.patches {
				if pt.Base.SpanID == entry.spanID {
					out = append(out, pt)
					break
				}
			}
		case SpanKindWarn:
			for _, w := range p.warns {
				if w.Base.SpanID == entry.spanID {
					out = append(out, w)
					break
				}
			}
		}
	}
	return out, nil
}

// ListUnflushedSessions implements ObservabilityProvider. The in-memory
// backend has no durable outbox, so it returns an empty slice (issue #33).
func (p *InMemoryObservabilityProvider) ListUnflushedSessions(_ context.Context) ([]SessionID, error) {
	return nil, nil
}

// CleanupSession implements ObservabilityProvider. The in-memory backend has
// no durable outbox, so every session is reported as not found (issue #33).
func (p *InMemoryObservabilityProvider) CleanupSession(_ context.Context, sessionID SessionID) error {
	return &SessionNotFoundError{SessionID: sessionID}
}

// Compile-time interface check.
var _ ObservabilityProvider = (*InMemoryObservabilityProvider)(nil)

// Compile-time check: the standard providers implement the optional
// WarnEmitter (issue #46).
var _ WarnEmitter = (*InMemoryObservabilityProvider)(nil)

// ============================================================================
// PatchEmitterAdapter (issue #28)
// ============================================================================

// PatchEmitterAdapter bridges middleware.PatchToolCallsMiddleware to an
// ObservabilityProvider. The middleware emits observability-independent
// middleware.PatchEvent values (it cannot import this package — observability
// imports middleware, not the reverse); this adapter stamps a SpanBase, maps
// the PatchKind to a PatchType, and records a warn-level PatchSpan.
//
// Wire it with:
//
//	obs := NewInMemoryObservabilityProvider()
//	mw := middleware.NewPatchToolCallsMiddleware("noop").
//	    WithObservability(NewPatchEmitterAdapter(obs))
type PatchEmitterAdapter struct {
	provider ObservabilityProvider
	seq      atomic.Uint64
}

// NewPatchEmitterAdapter wraps a provider so it can receive
// middleware.PatchEvent values as warn-level PatchSpans.
func NewPatchEmitterAdapter(provider ObservabilityProvider) *PatchEmitterAdapter {
	return &PatchEmitterAdapter{provider: provider}
}

// EmitPatch implements middleware.PatchEmitter. Fire-and-forget.
func (a *PatchEmitterAdapter) EmitPatch(event middleware.PatchEvent) {
	seq := a.seq.Add(1) - 1
	ts := memory.Timestamp("")
	base := SpanBase{
		SpanID:    SpanID(fmt.Sprintf("patch-%d", seq)),
		SessionID: event.SessionID,
		TaskID:    event.TaskID,
		Kind:      SpanKindPatch,
		StartedAt: ts,
		EndedAt:   ts,
		Status:    NewStatusOk(),
	}
	a.provider.EmitPatch(NewPatchSpan(
		base,
		event.CallID,
		event.ToolName,
		event.Original,
		event.Patched,
		patchTypeFromEvent(event),
	))
}

// patchTypeFromEvent maps a middleware.PatchEvent to the canonical PatchType.
func patchTypeFromEvent(event middleware.PatchEvent) PatchType {
	switch event.Kind {
	case middleware.PatchKindMalformedJSON:
		return NewPatchMalformedJSON(event.Error)
	case middleware.PatchKindParameterCoercion:
		return NewPatchParameterCoercion(event.Field, event.From, event.To)
	default:
		// DanglingToolCall is the default classification.
		return NewPatchDanglingToolCall(event.Reason)
	}
}

// Compile-time interface check.
var _ middleware.PatchEmitter = (*PatchEmitterAdapter)(nil)
