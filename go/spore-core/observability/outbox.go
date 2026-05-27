// Issue #33 — OutboxObservabilityProvider: a durable, JSONL-backed
// ObservabilityProvider that wraps InMemoryObservabilityProvider.
//
// This is the Go port of the Rust reference
// (rust/crates/spore-core/src/observability_outbox.rs). It does NOT
// transliterate Rust; it follows the idioms in go/CONVENTIONS.md.
//
// What it adds on top of the in-memory provider
//  1. Durable outbox. Every EmitX writes exactly ONE JSONL line,
//     synchronously appended and flushed (file Sync), to
//     {root}/sessions/{session_id}/trace.jsonl. The local JSONL file is the
//     source of truth (see observability/TRACE_SCHEMA.md). The wrapped
//     InMemoryObservabilityProvider still handles all buffering, metrics,
//     and query methods — this provider only adds the file write + OTLP hop.
//  2. OTLP forwarding. When the env var SPORE_OTLP_ENDPOINT is set
//     (non-empty / non-whitespace) at construction, each span is ALSO
//     forwarded to OTLP, best-effort and non-blocking. Drops on failure are
//     acceptable because the JSONL file is durable. When unset/empty, JSONL
//     only.
//
// New interface methods (added to ObservabilityProvider, issue #33)
//   - ListUnflushedSessions — session dirs under root/sessions that have a
//     trace.jsonl but no .flushed marker.
//   - CleanupSession(id) — deletes the session dir; returns an error matching
//     ErrSessionNotFound (errors.Is) if it does not exist. NEVER auto-deletes.
//
// Rules enforced
//   - One JSONL line per EmitX, synchronously appended + flushed.
//   - The line matches the schema envelope exactly; attributes is the verbatim
//     JSON serialization of the payload fields (structured unions stay tagged
//     objects; scalars stay scalar; keys are NOT renamed/flattened).
//   - level: patch spans → "warn"; all other kinds → "info".
//   - status/status_detail: Ok → ("ok", null), Error{message} →
//     ("error", message), Halted{reason} → ("halted", reason). The tagged
//     SpanStatus serde is NOT used for the line. status_detail emits explicit
//     null when status == ok.
//   - context_assembly vs compaction envelope kind mirrors EmitContext
//     (Compaction → "compaction", else → "context_assembly").
//   - session summary attributes.outcome is the bare string of SessionOutcome
//     (success/failure/partial).
//   - trace_id: a 32-hex (16 random bytes) string generated ONCE per session,
//     reused in every line for that session and as the OTLP trace id. JSONL
//     span_id/parent_span_id are the harness SpanID string VERBATIM. For OTLP
//     only, an 8-byte span id is derived by hashing (SHA-256, first 8 bytes)
//     the SpanID string.
//   - Rotation: when the active trace.jsonl exceeds MaxSizeBytes after an
//     append, it is renamed to trace-{NNN}.jsonl (zero-padded, increasing) and
//     a fresh trace.jsonl is opened. Rotated segments keep .jsonl.
//   - FlushSession: writes the trailing session summary line (when
//     FlushOnSessionEnd), flushes the file, and creates a sibling .flushed
//     marker. OTLP force-flush is best-effort with a timeout and logs on
//     failure — it NEVER returns an error for OTLP failure.
//
// OTLP forwarding behind an internal interface (issue #50).
// OTLP forwarding is isolated behind the small internal otlpForwarder
// interface with a real go.opentelemetry.io/otel SDK + otlptracegrpc
// implementation (otlpSdkForwarder, see otlp.go) and a no-op default
// (nullForwarder). When SPORE_OTLP_ENDPOINT is set, spans are also exported to
// Tempo over OTLP gRPC, reaching parity with Rust/TS/Python. When unset/empty,
// JSONL only. Per the #50 maintainer decision (Option A), the otel SDK is a
// blessed dependency (see go/CONVENTIONS.md) — the core outbox is no longer
// zero-dep, the accepted tradeoff for cross-language Tempo parity. The
// durable-JSONL path remains fully network-free and is tested WITHOUT any live
// OTLP/network; the JSONL append always happens before (and independent of)
// the best-effort OTLP forward.
package observability

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// DefaultMaxSizeBytes is the default outbox rotation threshold (50 MiB).
const DefaultMaxSizeBytes uint64 = 50 * 1024 * 1024

// activeFileName is the active per-session JSONL outbox file name.
const activeFileName = "trace.jsonl"

// flushedMarkerName is the sibling marker written by FlushSession.
const flushedMarkerName = ".flushed"

// ============================================================================
// Config
// ============================================================================

// OutboxConfig configures the durable outbox. Root is the outbox root
// directory (".spore" by convention); per session the provider derives
// {Root}/sessions/{session_id}/trace.jsonl.
type OutboxConfig struct {
	// Root is the outbox root directory (".spore" by convention).
	Root string
	// MaxSizeBytes rotates the active trace.jsonl once it exceeds this many
	// bytes. Defaults to DefaultMaxSizeBytes when zero.
	MaxSizeBytes uint64
	// FlushOnSessionEnd, when true, makes FlushSession write the trailing
	// session summary line.
	FlushOnSessionEnd bool
}

// NewOutboxConfig builds a config rooted at root with the spec defaults
// (MaxSizeBytes = 50 MiB, FlushOnSessionEnd = true).
func NewOutboxConfig(root string) OutboxConfig {
	return OutboxConfig{
		Root:              root,
		MaxSizeBytes:      DefaultMaxSizeBytes,
		FlushOnSessionEnd: true,
	}
}

func (c OutboxConfig) maxSize() uint64 {
	if c.MaxSizeBytes == 0 {
		return DefaultMaxSizeBytes
	}
	return c.MaxSizeBytes
}

func (c OutboxConfig) sessionsDir() string {
	return filepath.Join(c.Root, "sessions")
}

func (c OutboxConfig) sessionDir(sessionID SessionID) string {
	return filepath.Join(c.sessionsDir(), string(sessionID))
}

// ============================================================================
// Bare status mapping (the line format, NOT the tagged SpanStatus serde)
// ============================================================================

// statusPair maps a SpanStatus to the bare (status, status_detail) pair the
// JSONL schema requires. detail is nil for Ok (emits explicit null).
func statusPair(s SpanStatus) (string, *string) {
	switch s.Kind {
	case SpanStatusKindError:
		m := s.Message
		return "error", &m
	case SpanStatusKindHalted:
		r := s.Reason
		return "halted", &r
	default:
		return "ok", nil
	}
}

// ============================================================================
// TraceLine envelope
// ============================================================================

// TraceLine is one on-disk JSONL line. Common envelope fields are top-level;
// the per-kind payload lives under Attributes. Built by the From* /
// SessionSummary constructors — these are the unit-tested, cross-language
// mapping surface (see fixtures/observability/).
//
// Field order matches the schema for readable lines; cross-language fixture
// comparison is value-based (unmarshal both sides), so key ordering does not
// affect correctness. StatusDetail uses a non-omitempty pointer so it emits
// explicit null when nil.
type TraceLine struct {
	TraceID      string          `json:"trace_id"`
	SpanID       string          `json:"span_id"`
	ParentSpanID *string         `json:"parent_span_id"`
	SessionID    string          `json:"session_id"`
	TaskID       string          `json:"task_id"`
	Kind         string          `json:"kind"`
	Level        string          `json:"level"`
	Timestamp    string          `json:"timestamp"`
	StartedAt    string          `json:"started_at"`
	DurationMs   uint64          `json:"duration_ms"`
	Status       string          `json:"status"`
	StatusDetail *string         `json:"status_detail"`
	Attributes   json.RawMessage `json:"attributes"`
}

// fromBase fills the common envelope fields shared by every span kind.
func fromBase(base SpanBase, traceID, kind, level string, attributes json.RawMessage) TraceLine {
	status, detail := statusPair(base.Status)
	var parent *string
	if base.ParentSpanID != nil {
		p := string(*base.ParentSpanID)
		parent = &p
	}
	return TraceLine{
		TraceID:      traceID,
		SpanID:       string(base.SpanID),
		ParentSpanID: parent,
		SessionID:    string(base.SessionID),
		TaskID:       string(base.TaskID),
		Kind:         kind,
		Level:        level,
		Timestamp:    string(base.EndedAt),
		StartedAt:    string(base.StartedAt),
		DurationMs:   base.DurationMs,
		Status:       status,
		StatusDetail: detail,
		Attributes:   attributes,
	}
}

func mustMarshal(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		// The attribute payloads are plain data; marshaling cannot fail.
		return json.RawMessage("null")
	}
	return json.RawMessage(b)
}

// --- per-kind attribute payloads (ordered structs → verbatim keys) ---

type turnAttrs struct {
	TurnNumber         uint32     `json:"turn_number"`
	InputTokens        uint32     `json:"input_tokens"`
	OutputTokens       uint32     `json:"output_tokens"`
	CacheReadTokens    *uint32    `json:"cache_read_tokens"`
	CacheWriteTokens   *uint32    `json:"cache_write_tokens"`
	CostUSD            float64    `json:"cost_usd"`
	StopReason         StopReason `json:"stop_reason"`
	ToolCallsRequested uint32     `json:"tool_calls_requested"`
}

// TraceLineFromTurn builds the TraceLine for a turn span.
//
// When content capture (issue #64) populated OutputText / ToolCalls, the
// gen_ai.* attributes ride alongside the metrics keys so the same line is
// readable in an LLM-native backend (Phoenix) without code changes. When
// neither is set the attributes object is byte-identical to the pre-#64 output.
func TraceLineFromTurn(span TurnSpan, traceID string) TraceLine {
	attrs := turnAttrs{
		TurnNumber:         span.TurnNumber,
		InputTokens:        span.InputTokens,
		OutputTokens:       span.OutputTokens,
		CacheReadTokens:    span.CacheReadTokens,
		CacheWriteTokens:   span.CacheWriteTokens,
		CostUSD:            span.CostUSD,
		StopReason:         span.StopReason,
		ToolCallsRequested: span.ToolCallsRequested,
	}
	if span.OutputText == nil && span.ToolCalls == nil {
		return fromBase(span.Base, traceID, "turn", "info", mustMarshal(attrs))
	}
	extra := make(map[string]any)
	if msg := span.OutputText; msg != nil {
		extra["gen_ai.response.role"] = string(msg.Role)
		extra["gen_ai.response.content"] = msg.Content
		extra["gen_ai.response.content_truncated"] = msg.Truncated
	}
	if span.ToolCalls != nil {
		extra["gen_ai.response.tool_calls"] = span.ToolCalls
	}
	return fromBase(span.Base, traceID, "turn", "info", mergeAttrs(attrs, extra))
}

// mergeAttrs marshals base, then layers the extra gen_ai.* keys (issue #64) into
// the resulting object. Fixture comparison is value-based, so the merged key
// ordering is irrelevant; correctness is the union of keys and values.
func mergeAttrs(base any, extra map[string]any) json.RawMessage {
	merged := make(map[string]json.RawMessage)
	if err := json.Unmarshal(mustMarshal(base), &merged); err != nil {
		return mustMarshal(base)
	}
	for k, v := range extra {
		merged[k] = mustMarshal(v)
	}
	return mustMarshal(merged)
}

type toolCallAttrs struct {
	ToolName            string   `json:"tool_name"`
	CallID              string   `json:"call_id"`
	ParametersSizeBytes uint64   `json:"parameters_size_bytes"`
	OutputSizeBytes     uint64   `json:"output_size_bytes"`
	Truncated           bool     `json:"truncated"`
	SandboxMode         string   `json:"sandbox_mode"`
	SandboxViolations   []string `json:"sandbox_violations"`
}

// TraceLineFromToolCall builds the TraceLine for a tool_call span.
func TraceLineFromToolCall(span ToolCallSpan, traceID string) TraceLine {
	violations := span.SandboxViolations
	if violations == nil {
		violations = []string{}
	}
	attrs := toolCallAttrs{
		ToolName:            span.ToolName,
		CallID:              span.CallID,
		ParametersSizeBytes: span.ParametersSizeBytes,
		OutputSizeBytes:     span.OutputSizeBytes,
		Truncated:           span.Truncated,
		SandboxMode:         span.SandboxMode,
		SandboxViolations:   violations,
	}
	if span.Arguments == nil && span.Result == nil {
		return fromBase(span.Base, traceID, "tool_call", "info", mustMarshal(attrs))
	}
	extra := make(map[string]any)
	if args := span.Arguments; args != nil {
		extra["gen_ai.tool.name"] = args.Name
		extra["gen_ai.tool.call.arguments"] = args.Arguments
		extra["gen_ai.tool.call.arguments_truncated"] = args.ArgumentsTruncated
	}
	if res := span.Result; res != nil {
		extra["gen_ai.tool.message.content"] = res.Content
		extra["gen_ai.tool.message.content_truncated"] = res.Truncated
	}
	return fromBase(span.Base, traceID, "tool_call", "info", mergeAttrs(attrs, extra))
}

type sensorAttrs struct {
	SensorID   SensorID      `json:"sensor_id"`
	SensorKind SensorKind    `json:"sensor_kind"`
	Trigger    SensorTrigger `json:"trigger"`
	Outcome    SensorOutcome `json:"outcome"`
	Fired      bool          `json:"fired"`
}

// TraceLineFromSensor builds the TraceLine for a sensor_evaluation span.
func TraceLineFromSensor(span SensorSpan, traceID string) TraceLine {
	attrs := sensorAttrs{
		SensorID:   span.SensorID,
		SensorKind: span.SensorKind,
		Trigger:    span.Trigger,
		Outcome:    span.Outcome,
		Fired:      span.Fired,
	}
	return fromBase(span.Base, traceID, "sensor_evaluation", "info", mustMarshal(attrs))
}

type contextAttrs struct {
	Operation         ContextOperation `json:"operation"`
	TokensBefore      uint32           `json:"tokens_before"`
	TokensAfter       uint32           `json:"tokens_after"`
	UtilizationBefore float32          `json:"utilization_before"`
	UtilizationAfter  float32          `json:"utilization_after"`
}

// TraceLineFromContext builds the TraceLine for a context span. The envelope
// kind mirrors EmitContext: Compaction → "compaction", else
// "context_assembly".
func TraceLineFromContext(span ContextSpan, traceID string) TraceLine {
	kind := "context_assembly"
	if span.Operation.Kind == ContextOpKindCompaction {
		kind = "compaction"
	}
	attrs := contextAttrs{
		Operation:         span.Operation,
		TokensBefore:      span.TokensBefore,
		TokensAfter:       span.TokensAfter,
		UtilizationBefore: span.UtilizationBefore,
		UtilizationAfter:  span.UtilizationAfter,
	}
	return fromBase(span.Base, traceID, kind, "info", mustMarshal(attrs))
}

type middlewareAttrs struct {
	Hook     HookPoint          `json:"hook"`
	Decision MiddlewareDecision `json:"decision"`
}

// TraceLineFromMiddleware builds the TraceLine for a middleware_hook span.
func TraceLineFromMiddleware(span MiddlewareSpan, traceID string) TraceLine {
	attrs := middlewareAttrs{Hook: span.Hook, Decision: span.Decision}
	return fromBase(span.Base, traceID, "middleware_hook", "info", mustMarshal(attrs))
}

type patchAttrs struct {
	ToolName           string          `json:"tool_name"`
	CallID             string          `json:"call_id"`
	PatchType          PatchType       `json:"patch_type"`
	OriginalParameters json.RawMessage `json:"original_parameters"`
	PatchedParameters  json.RawMessage `json:"patched_parameters"`
}

// TraceLineFromPatch builds the TraceLine for a patch span. Patch spans are
// ALWAYS warn-level.
func TraceLineFromPatch(span PatchSpan, traceID string) TraceLine {
	attrs := patchAttrs{
		ToolName:           span.ToolName,
		CallID:             span.CallID,
		PatchType:          span.PatchType,
		OriginalParameters: span.OriginalParameters,
		PatchedParameters:  span.PatchedParameters,
	}
	return fromBase(span.Base, traceID, "patch", "warn", mustMarshal(attrs))
}

// TraceLineFromWarn builds the TraceLine for a warn span (issue #46). Warn
// spans are ALWAYS warn-level; the WarnEvent payload is serialized verbatim as
// the attributes object.
func TraceLineFromWarn(span WarnSpan, traceID string) TraceLine {
	return fromBase(span.Base, traceID, "warn", "warn", mustMarshal(span.Event))
}

type sessionAttrs struct {
	Outcome      string  `json:"outcome"`
	TotalTurns   uint32  `json:"total_turns"`
	TotalCostUSD float64 `json:"total_cost_usd"`
	SensorFires  uint32  `json:"sensor_fires"`
	SensorHalts  uint32  `json:"sensor_halts"`
	PatchCount   uint32  `json:"patch_count"`
}

// TraceLineSessionSummary builds the trailing session summary line from
// rolled-up metrics. The envelope identity (span/parent ids, task id, timing,
// status) come from root; attributes.outcome is the bare string of the
// SessionOutcome in metrics.
func TraceLineSessionSummary(metrics SessionMetrics, traceID string, root SpanBase) TraceLine {
	attrs := sessionAttrs{
		Outcome:      string(metrics.Outcome.Kind),
		TotalTurns:   metrics.TotalTurns,
		TotalCostUSD: metrics.TotalCostUSD,
		SensorFires:  metrics.SensorFires,
		SensorHalts:  metrics.SensorHalts,
		PatchCount:   metrics.PatchCount,
	}
	return fromBase(root, traceID, "session", "info", mustMarshal(attrs))
}

func (l TraceLine) toJSONLLine() ([]byte, error) {
	b, err := json.Marshal(l)
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

// ============================================================================
// trace_id / span_id derivation
// ============================================================================

// newTraceID returns a fresh 32-hex (16 random bytes) trace id.
func newTraceID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// Extremely unlikely; fall back to a hash of a process-unique seed.
		h := sha256.Sum256([]byte(fmt.Sprintf("%p%d", &b, os.Getpid())))
		copy(b[:], h[:16])
	}
	return hex.EncodeToString(b[:])
}

// deriveOTLPSpanID derives an 8-byte OTLP span id from the harness SpanID
// string by hashing (SHA-256, first 8 bytes). Matches the Rust reference.
func deriveOTLPSpanID(spanID string) [8]byte {
	h := sha256.Sum256([]byte(spanID))
	var out [8]byte
	copy(out[:], h[:8])
	return out
}

// ============================================================================
// OTLP forwarder (internal interface — see file-header deviation note)
// ============================================================================

// otlpForwarder abstracts the OTLP forwarding hop. The durable JSONL path does
// not depend on it; it exists only so the version-churny OTLP SDK can be
// isolated and so tests run without a network.
type otlpForwarder interface {
	// forward sends one already-built line. Best-effort, non-blocking, never
	// errors.
	forward(line TraceLine)
	// forceFlush best-effort flushes; logs on failure, never returns an error.
	forceFlush()
}

// nullForwarder is used when SPORE_OTLP_ENDPOINT is unset/empty, and as the
// default impl until the OTLP SDK is wired in (see deviation note).
type nullForwarder struct{}

func (nullForwarder) forward(TraceLine) {}
func (nullForwarder) forceFlush()       {}

// newForwarder resolves SPORE_OTLP_ENDPOINT once. Empty/whitespace is treated
// as unset (JSONL only → nullForwarder). A non-empty endpoint constructs the
// real OTLP gRPC forwarder (see otlp.go); if its initialization fails, it logs
// and falls back to nullForwarder so the durable JSONL path is never affected.
func newForwarder(endpoint string) otlpForwarder {
	trimmed := strings.TrimSpace(endpoint)
	if trimmed == "" {
		return nullForwarder{}
	}
	f, err := newOTLPSdkForwarder(trimmed)
	if err != nil {
		fmt.Fprintf(os.Stderr,
			"[spore-core] failed to init OTLP forwarder for %q; JSONL only: %v\n", trimmed, err)
		return nullForwarder{}
	}
	return f
}

// ============================================================================
// sessionWriter — per-session open handle + rotation + trace_id
// ============================================================================

type sessionWriter struct {
	file         *os.File
	activePath   string
	dir          string
	bytesWritten uint64
	maxSizeBytes uint64
	nextSeq      uint32
	traceID      string
}

func openSessionWriter(dir string, maxSizeBytes uint64) (*sessionWriter, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	activePath := filepath.Join(dir, activeFileName)
	f, err := os.OpenFile(activePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	var size uint64
	if info, err := f.Stat(); err == nil {
		size = uint64(info.Size())
	}
	return &sessionWriter{
		file:         f,
		activePath:   activePath,
		dir:          dir,
		bytesWritten: size,
		maxSizeBytes: maxSizeBytes,
		nextSeq:      scanNextSeq(dir),
		traceID:      newTraceID(),
	}, nil
}

// scanNextSeq finds the next rotation sequence by scanning existing
// trace-NNN.jsonl files.
func scanNextSeq(dir string) uint32 {
	var maxSeen uint32
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 1
	}
	for _, e := range entries {
		name := e.Name()
		rest, ok := strings.CutPrefix(name, "trace-")
		if !ok {
			continue
		}
		num, ok := strings.CutSuffix(rest, ".jsonl")
		if !ok {
			continue
		}
		if n, err := strconv.ParseUint(num, 10, 32); err == nil {
			if uint32(n) > maxSeen {
				maxSeen = uint32(n)
			}
		}
	}
	return maxSeen + 1
}

// append writes one line, syncs, then rotates if over size.
func (w *sessionWriter) append(line []byte) error {
	if _, err := w.file.Write(line); err != nil {
		return err
	}
	if err := w.file.Sync(); err != nil {
		return err
	}
	w.bytesWritten += uint64(len(line))
	if w.bytesWritten > w.maxSizeBytes {
		return w.rotate()
	}
	return nil
}

func (w *sessionWriter) rotate() error {
	rotated := filepath.Join(w.dir, fmt.Sprintf("trace-%03d.jsonl", w.nextSeq))
	w.nextSeq++
	if err := w.file.Sync(); err != nil {
		return err
	}
	if err := w.file.Close(); err != nil {
		return err
	}
	if err := os.Rename(w.activePath, rotated); err != nil {
		return err
	}
	f, err := os.OpenFile(w.activePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	w.file = f
	w.bytesWritten = 0
	return nil
}

func (w *sessionWriter) close() {
	if w.file != nil {
		_ = w.file.Sync()
		_ = w.file.Close()
		w.file = nil
	}
}

// ============================================================================
// OutboxObservabilityProvider
// ============================================================================

// OutboxObservabilityProvider is a durable, JSONL-backed ObservabilityProvider
// (issue #33). It wraps an InMemoryObservabilityProvider for all buffering /
// metrics / query behavior and adds: one synchronous JSONL line per EmitX,
// optional best-effort OTLP forwarding, rotation, and flush markers.
type OutboxObservabilityProvider struct {
	inner   *InMemoryObservabilityProvider
	config  OutboxConfig
	otlp    otlpForwarder
	mu      sync.Mutex
	writers map[SessionID]*sessionWriter
}

// NewOutboxObservabilityProvider constructs a provider. It reads
// SPORE_OTLP_ENDPOINT once: empty/whitespace is treated as unset (JSONL only);
// a non-empty value wires the OTLP forwarder to that endpoint (see deviation).
func NewOutboxObservabilityProvider(config OutboxConfig) *OutboxObservabilityProvider {
	return &OutboxObservabilityProvider{
		inner:   NewInMemoryObservabilityProvider(),
		config:  config,
		otlp:    newForwarder(os.Getenv("SPORE_OTLP_ENDPOINT")),
		writers: make(map[SessionID]*sessionWriter),
	}
}

// Inner exposes the wrapped in-memory provider (e.g. to call
// RecordGuidesUsed).
func (p *OutboxObservabilityProvider) Inner() *InMemoryObservabilityProvider {
	return p.inner
}

// SetSessionOutcome records the terminal outcome, forwarding to the inner
// provider so SessionMetrics (and the flushed session summary line) surface
// it. Mirrors the Rust trait method.
func (p *OutboxObservabilityProvider) SetSessionOutcome(sessionID SessionID, outcome SessionOutcome) {
	p.inner.SetSessionOutcome(sessionID, outcome)
}

// TraceIDFor returns the per-session trace id, opening the session writer if
// needed.
func (p *OutboxObservabilityProvider) TraceIDFor(sessionID SessionID) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	w, err := p.writerForLocked(sessionID)
	if err != nil {
		return "", err
	}
	return w.traceID, nil
}

// writerForLocked returns (opening if needed) the session writer. Caller holds
// p.mu.
func (p *OutboxObservabilityProvider) writerForLocked(sessionID SessionID) (*sessionWriter, error) {
	if w, ok := p.writers[sessionID]; ok {
		return w, nil
	}
	w, err := openSessionWriter(p.config.sessionDir(sessionID), p.config.maxSize())
	if err != nil {
		return nil, err
	}
	p.writers[sessionID] = w
	return w, nil
}

// traceIDLocked returns the per-session trace id, opening the writer if
// needed; returns "" on open failure (the line is still best-effort written).
func (p *OutboxObservabilityProvider) traceIDLocked(sessionID SessionID) string {
	w, err := p.writerForLocked(sessionID)
	if err != nil {
		return ""
	}
	return w.traceID
}

// writeLine appends a built line to the session's JSONL file then forwards to
// OTLP. The file write is the reliability guarantee; OTLP is best-effort.
func (p *OutboxObservabilityProvider) writeLine(sessionID SessionID, line TraceLine) {
	jsonl, err := line.toJSONLLine()
	if err != nil {
		fmt.Fprintf(os.Stderr, "[spore-core] outbox marshal failed: %v\n", err)
		return
	}
	p.mu.Lock()
	w, err := p.writerForLocked(sessionID)
	if err != nil {
		p.mu.Unlock()
		fmt.Fprintf(os.Stderr, "[spore-core] outbox open failed: %v\n", err)
		return
	}
	if err := w.append(jsonl); err != nil {
		fmt.Fprintf(os.Stderr, "[spore-core] outbox append failed: %v\n", err)
	}
	p.mu.Unlock()
	p.otlp.forward(line)
}

// --- EmitX (build the line with the session trace id, delegate to inner) ---

// EmitTurn implements ObservabilityProvider.
func (p *OutboxObservabilityProvider) EmitTurn(span TurnSpan) {
	sid := span.Base.SessionID
	p.mu.Lock()
	traceID := p.traceIDLocked(sid)
	p.mu.Unlock()
	line := TraceLineFromTurn(span, traceID)
	p.inner.EmitTurn(span)
	p.writeLine(sid, line)
}

// EmitToolCall implements ObservabilityProvider.
func (p *OutboxObservabilityProvider) EmitToolCall(span ToolCallSpan) {
	sid := span.Base.SessionID
	p.mu.Lock()
	traceID := p.traceIDLocked(sid)
	p.mu.Unlock()
	line := TraceLineFromToolCall(span, traceID)
	p.inner.EmitToolCall(span)
	p.writeLine(sid, line)
}

// EmitSensor implements ObservabilityProvider.
func (p *OutboxObservabilityProvider) EmitSensor(span SensorSpan) {
	sid := span.Base.SessionID
	p.mu.Lock()
	traceID := p.traceIDLocked(sid)
	p.mu.Unlock()
	line := TraceLineFromSensor(span, traceID)
	p.inner.EmitSensor(span)
	p.writeLine(sid, line)
}

// EmitContext implements ObservabilityProvider.
func (p *OutboxObservabilityProvider) EmitContext(span ContextSpan) {
	sid := span.Base.SessionID
	p.mu.Lock()
	traceID := p.traceIDLocked(sid)
	p.mu.Unlock()
	line := TraceLineFromContext(span, traceID)
	p.inner.EmitContext(span)
	p.writeLine(sid, line)
}

// EmitMiddleware implements ObservabilityProvider.
func (p *OutboxObservabilityProvider) EmitMiddleware(span MiddlewareSpan) {
	sid := span.Base.SessionID
	p.mu.Lock()
	traceID := p.traceIDLocked(sid)
	p.mu.Unlock()
	line := TraceLineFromMiddleware(span, traceID)
	p.inner.EmitMiddleware(span)
	p.writeLine(sid, line)
}

// EmitPatch implements ObservabilityProvider.
func (p *OutboxObservabilityProvider) EmitPatch(span PatchSpan) {
	sid := span.Base.SessionID
	p.mu.Lock()
	traceID := p.traceIDLocked(sid)
	p.mu.Unlock()
	line := TraceLineFromPatch(span, traceID)
	p.inner.EmitPatch(span)
	p.writeLine(sid, line)
}

// EmitWarn implements WarnEmitter (issue #46). Mirrors EmitPatch: it writes a
// warn-level trace line and delegates to the inner provider so SessionMetrics
// (CompactionVerificationFailures) surfaces the event. Per the Rust reference,
// the trailing session_summary line does NOT carry the failure counter.
func (p *OutboxObservabilityProvider) EmitWarn(span WarnSpan) {
	sid := span.Base.SessionID
	p.mu.Lock()
	traceID := p.traceIDLocked(sid)
	p.mu.Unlock()
	line := TraceLineFromWarn(span, traceID)
	p.inner.EmitWarn(span)
	p.writeLine(sid, line)
}

// FlushSession implements ObservabilityProvider. It writes the trailing
// session summary line (when FlushOnSessionEnd), flushes the file, force-
// flushes OTLP best-effort, and creates the sibling .flushed marker. It then
// delegates to the inner provider for idempotency bookkeeping. OTLP failures
// never produce an error.
func (p *OutboxObservabilityProvider) FlushSession(ctx context.Context, sessionID SessionID) error {
	if p.config.FlushOnSessionEnd {
		if metrics, _ := p.inner.GetSessionMetrics(ctx, sessionID); metrics != nil {
			p.mu.Lock()
			traceID := p.traceIDLocked(sessionID)
			p.mu.Unlock()
			root := SpanBase{
				SpanID:     SpanID(sessionID),
				SessionID:  sessionID,
				TaskID:     metrics.TaskID,
				Kind:       SpanKindSession,
				DurationMs: metrics.TotalDurationMs,
				Status:     NewStatusOk(),
			}
			line := TraceLineSessionSummary(*metrics, traceID, root)
			p.writeLine(sessionID, line)
		}
	}

	// Flush the JSONL file handle.
	p.mu.Lock()
	if w, ok := p.writers[sessionID]; ok {
		_ = w.file.Sync()
	}
	p.mu.Unlock()

	// Best-effort OTLP force-flush; never errors out of FlushSession.
	p.otlp.forceFlush()

	// Create the sibling .flushed marker.
	dir := p.config.sessionDir(sessionID)
	if _, err := os.Stat(dir); err == nil {
		marker := filepath.Join(dir, flushedMarkerName)
		if f, err := os.Create(marker); err != nil {
			fmt.Fprintf(os.Stderr, "[spore-core] failed to write .flushed marker: %v\n", err)
		} else {
			_ = f.Close()
		}
	}

	// Delegate to inner so its flushed bookkeeping stays consistent.
	return p.inner.FlushSession(ctx, sessionID)
}

// GetSessionMetrics implements ObservabilityProvider (delegated).
func (p *OutboxObservabilityProvider) GetSessionMetrics(ctx context.Context, sessionID SessionID) (*SessionMetrics, error) {
	return p.inner.GetSessionMetrics(ctx, sessionID)
}

// GetSessions implements ObservabilityProvider (delegated).
func (p *OutboxObservabilityProvider) GetSessions(ctx context.Context, since Timestamp, domain *string, outcome *SessionOutcome) ([]SessionMetrics, error) {
	return p.inner.GetSessions(ctx, since, domain, outcome)
}

// GetTrace implements ObservabilityProvider (delegated).
func (p *OutboxObservabilityProvider) GetTrace(ctx context.Context, sessionID SessionID) ([]Span, error) {
	return p.inner.GetTrace(ctx, sessionID)
}

// ListUnflushedSessions implements ObservabilityProvider. Returns session dirs
// under root/sessions that have a trace.jsonl but no .flushed marker, sorted.
func (p *OutboxObservabilityProvider) ListUnflushedSessions(_ context.Context) ([]SessionID, error) {
	entries, err := os.ReadDir(p.config.sessionsDir())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []SessionID
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := filepath.Join(p.config.sessionsDir(), e.Name())
		hasTrace := fileExists(filepath.Join(dir, activeFileName))
		flushed := fileExists(filepath.Join(dir, flushedMarkerName))
		if hasTrace && !flushed {
			out = append(out, SessionID(e.Name()))
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out, nil
}

// CleanupSession implements ObservabilityProvider. It deletes the session's
// outbox directory; returns a SessionNotFoundError (errors.Is ErrSessionNotFound)
// when the directory does not exist. NEVER auto-deletes.
func (p *OutboxObservabilityProvider) CleanupSession(_ context.Context, sessionID SessionID) error {
	dir := p.config.sessionDir(sessionID)
	if _, err := os.Stat(dir); err != nil {
		if os.IsNotExist(err) {
			return &SessionNotFoundError{SessionID: sessionID}
		}
		return err
	}
	// Drop the open writer first so the handle is released.
	p.mu.Lock()
	if w, ok := p.writers[sessionID]; ok {
		w.close()
		delete(p.writers, sessionID)
	}
	p.mu.Unlock()
	return os.RemoveAll(dir)
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// Compile-time interface check.
var _ ObservabilityProvider = (*OutboxObservabilityProvider)(nil)
var _ WarnEmitter = (*OutboxObservabilityProvider)(nil)
