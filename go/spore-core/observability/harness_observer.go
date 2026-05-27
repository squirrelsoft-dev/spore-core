// HarnessObserver adapter + HarnessBuilder (issue #12).
//
// The root `sporecore` package defines the consumer-side
// sporecore.HarnessObserver seam the ReAct loop emits through. It cannot
// import this package (the import edge runs observability -> sporecore, so the
// reverse is a cycle), so this adapter lives here: it implements
// sporecore.HarnessObserver by building real spans (TurnSpan / ToolCallSpan)
// and forwarding to an ObservabilityProvider, exactly mirroring the in-loop
// span construction in the Rust reference (harness.rs `run_react_inner`).
//
// HarnessBuilder is the fluent assembly point — it wires the required harness
// components plus the optional middleware / observability / pricing, then
// builds a *sporecore.StandardHarness whose loop emits through this adapter.

package observability

import (
	"context"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
)

// HarnessObservabilityAdapter implements sporecore.HarnessObserver by building
// spans and forwarding them to a wrapped ObservabilityProvider. Pricing is
// held here (not on HarnessConfig) because PricingTable is defined in this
// package; the loop calls CostFor at emit time so cost is stamped per spec.
type HarnessObservabilityAdapter struct {
	provider ObservabilityProvider
	pricing  PricingTable
}

// NewHarnessObserver wraps an ObservabilityProvider as a
// sporecore.HarnessObserver, stamping turn-span cost via pricing.
func NewHarnessObserver(provider ObservabilityProvider, pricing PricingTable) *HarnessObservabilityAdapter {
	return &HarnessObservabilityAdapter{provider: provider, pricing: pricing}
}

// EmitTurn builds a root TurnSpan and forwards it. Mirrors the Rust loop's
// per-turn span: NewRoot(...turn id...), Finish(duration, status), fields from
// usage, cost via pricing, stop reason + tool-calls-requested from the result.
func (a *HarnessObservabilityAdapter) EmitTurn(
	spanID string,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	turnNumber uint32,
	startedAt string,
	durationMs uint64,
	usage sporecore.TokenUsage,
	costUSD float64,
	stopReason sporecore.StopReason,
	toolCallsRequested uint32,
	errorMessage string,
) {
	base := NewRoot(SpanID(spanID), sessionID, taskID, SpanKindTurn, Timestamp(startedAt))
	status := NewStatusOk()
	if errorMessage != "" {
		status = NewStatusError(errorMessage)
	}
	base.Finish(Timestamp(startedAt), status, durationMs)
	a.provider.EmitTurn(TurnSpan{
		Base:               base,
		TurnNumber:         turnNumber,
		InputTokens:        usage.InputTokens,
		OutputTokens:       usage.OutputTokens,
		CacheReadTokens:    usage.CacheReadTokens,
		CacheWriteTokens:   usage.CacheWriteTokens,
		CostUSD:            costUSD,
		StopReason:         stopReason,
		ToolCallsRequested: toolCallsRequested,
	})
}

// EmitToolCall builds a child ToolCallSpan (parented to the turn span) and
// forwards it. SandboxMode is "" and SandboxViolations nil, mirroring Rust.
func (a *HarnessObservabilityAdapter) EmitToolCall(
	spanID string,
	parentSpanID string,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	toolName string,
	callID string,
	startedAt string,
	durationMs uint64,
	parametersSizeBytes uint64,
	outputSizeBytes uint64,
	truncated bool,
	isError bool,
) {
	// Reconstruct the parent envelope so NewChild can stamp the parent id.
	parent := NewRoot(SpanID(parentSpanID), sessionID, taskID, SpanKindTurn, Timestamp(startedAt))
	base := NewChild(SpanID(spanID), parent, SpanKindToolCall, Timestamp(startedAt))
	status := NewStatusOk()
	if isError {
		status = NewStatusError("tool returned a recoverable error")
	}
	base.Finish(Timestamp(startedAt), status, durationMs)
	a.provider.EmitToolCall(ToolCallSpan{
		Base:                base,
		ToolName:            toolName,
		CallID:              callID,
		ParametersSizeBytes: parametersSizeBytes,
		OutputSizeBytes:     outputSizeBytes,
		Truncated:           truncated,
		SandboxMode:         "",
		SandboxViolations:   nil,
	})
}

// SetSessionOutcome records the terminal outcome on the wrapped provider.
func (a *HarnessObservabilityAdapter) SetSessionOutcome(sessionID sporecore.SessionID, success bool, failureReason string) {
	outcome := guideregistry.NewOutcomeSuccess()
	if !success {
		outcome = guideregistry.NewOutcomeFailure(failureReason)
	}
	a.provider.SetSessionOutcome(sessionID, outcome)
}

// FlushSession flushes the wrapped provider's durable session record.
func (a *HarnessObservabilityAdapter) FlushSession(ctx context.Context, sessionID sporecore.SessionID) {
	// Fire-and-forget per the harness seam; the underlying error is the
	// caller's concern via ListUnflushedSessions, not the loop's.
	_ = a.provider.FlushSession(ctx, sessionID)
}

// CostFor computes USD cost for a turn from the adapter's pricing table.
func (a *HarnessObservabilityAdapter) CostFor(usage sporecore.TokenUsage) float64 {
	return a.pricing.CostFor(usage.InputTokens, usage.OutputTokens, usage.CacheReadTokens, usage.CacheWriteTokens)
}

// EmitCompaction builds the Compaction ContextSpan for an accepted summary and
// forwards it (issue #46). Mirrors the Rust reference's accept_compaction span:
// a root context span with a Compaction operation, tokens_before == tokens_after
// and zero tokens_reclaimed (the bridge does not surface reclaim accounting).
func (a *HarnessObservabilityAdapter) EmitCompaction(
	spanID string,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	startedAt string,
	messagesRemoved uint32,
	tokensBefore uint32,
) {
	base := NewRoot(SpanID(spanID), sessionID, taskID, SpanKindCompaction, Timestamp(startedAt))
	base.Finish(Timestamp(startedAt), NewStatusOk(), 0)
	a.provider.EmitContext(ContextSpan{
		Base:              base,
		Operation:         NewContextOpCompaction(messagesRemoved, 0),
		TokensBefore:      tokensBefore,
		TokensAfter:       tokensBefore,
		UtilizationBefore: 0,
		UtilizationAfter:  0,
	})
}

// EmitCompactionVerificationFailed builds a WarnSpan and forwards it via the
// OPTIONAL WarnEmitter surface (issue #46). If the wrapped provider does not
// implement WarnEmitter, the warn is silently dropped — the Go equivalent of
// the Rust reference's default-no-op emit_warn, keeping providers predating #46
// unaffected.
func (a *HarnessObservabilityAdapter) EmitCompactionVerificationFailed(
	spanID string,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	startedAt string,
	missingItems []string,
	acceptedAnyway bool,
) {
	emitter, ok := a.provider.(WarnEmitter)
	if !ok {
		return
	}
	base := NewRoot(SpanID(spanID), sessionID, taskID, SpanKindWarn, Timestamp(startedAt))
	emitter.EmitWarn(NewWarnSpan(base, NewWarnCompactionVerificationFailed(missingItems, acceptedAnyway)))
}

// Compile-time interface check.
var _ sporecore.HarnessObserver = (*HarnessObservabilityAdapter)(nil)

// ============================================================================
// HarnessBuilder
// ============================================================================

// HarnessBuilder is the fluent assembler for a *sporecore.StandardHarness.
//
// It lives in this package (not the root) because wiring observability into
// the loop means constructing the HarnessObservabilityAdapter, which depends
// on this package's span/pricing types. It takes the five required components
// up front and exposes fluent setters for the optional ones (middleware,
// observability, pricing). Setters return the receiver, the conventional Go
// fluent-builder shape.
type HarnessBuilder struct {
	agent             sporecore.Agent
	toolRegistry      sporecore.ToolRegistry
	sandbox           sporecore.SandboxProvider
	contextManager    sporecore.ContextManager
	terminationPolicy sporecore.TerminationPolicy
	middleware        sporecore.MiddlewareChain
	provider          ObservabilityProvider
	pricing           PricingTable
	// Compaction loop (issue #46). compactionVerifier defaults to
	// contextmgr.NewKeyTermVerifier(); maxCompactionAttempts defaults to 2.
	compactionVerifier    sporecore.CompactionVerifier
	maxCompactionAttempts uint32
}

// NewHarnessBuilder starts a builder from the five required components.
// Optional components default to nil / DefaultPricing until set.
func NewHarnessBuilder(
	agent sporecore.Agent,
	toolRegistry sporecore.ToolRegistry,
	sandbox sporecore.SandboxProvider,
	contextManager sporecore.ContextManager,
	terminationPolicy sporecore.TerminationPolicy,
) *HarnessBuilder {
	return &HarnessBuilder{
		agent:                 agent,
		toolRegistry:          toolRegistry,
		sandbox:               sandbox,
		contextManager:        contextManager,
		terminationPolicy:     terminationPolicy,
		pricing:               DefaultPricing(),
		compactionVerifier:    contextmgr.NewKeyTermVerifier(),
		maxCompactionAttempts: 2,
	}
}

// CompactionVerifier injects a post-compaction verifier (issue #46). Defaults
// to contextmgr.NewKeyTermVerifier().
func (b *HarnessBuilder) CompactionVerifier(v sporecore.CompactionVerifier) *HarnessBuilder {
	b.compactionVerifier = v
	return b
}

// MaxCompactionAttempts sets the maximum number of compaction-summary attempts
// before accepting a failing summary anyway (issue #46). Defaults to 2; the
// loop clamps the effective value to a minimum of 1.
func (b *HarnessBuilder) MaxCompactionAttempts(n uint32) *HarnessBuilder {
	b.maxCompactionAttempts = n
	return b
}

// Middleware injects a middleware chain.
func (b *HarnessBuilder) Middleware(m sporecore.MiddlewareChain) *HarnessBuilder {
	b.middleware = m
	return b
}

// Observability injects an ObservabilityProvider. The harness loop emits real
// spans through it (turn spans, tool-call spans) and flushes on terminal
// outcomes.
func (b *HarnessBuilder) Observability(provider ObservabilityProvider) *HarnessBuilder {
	b.provider = provider
	return b
}

// WithObservabilityOutbox constructs and injects a durable-outbox
// ObservabilityProvider rooted at root (typically the ".spore" directory),
// using the spec defaults. Honors SPORE_OTLP_ENDPOINT for OTLP forwarding to
// Tempo over gRPC (issue #50); unset/empty means JSONL only.
func (b *HarnessBuilder) WithObservabilityOutbox(root string) *HarnessBuilder {
	return b.Observability(NewOutboxObservabilityProvider(NewOutboxConfig(root)))
}

// Pricing sets the token→USD pricing table used to stamp cost on turn spans.
func (b *HarnessBuilder) Pricing(p PricingTable) *HarnessBuilder {
	b.pricing = p
	return b
}

// BuildConfig assembles the HarnessConfig without wrapping it in a harness.
func (b *HarnessBuilder) BuildConfig() sporecore.HarnessConfig {
	cfg := sporecore.HarnessConfig{
		Agent:                 b.agent,
		ToolRegistry:          b.toolRegistry,
		Sandbox:               b.sandbox,
		ContextManager:        b.contextManager,
		TerminationPolicy:     b.terminationPolicy,
		Middleware:            b.middleware,
		CompactionVerifier:    b.compactionVerifier,
		MaxCompactionAttempts: b.maxCompactionAttempts,
	}
	if b.provider != nil {
		cfg.Observability = NewHarnessObserver(b.provider, b.pricing)
	}
	return cfg
}

// Build assembles a ready-to-run *sporecore.StandardHarness.
func (b *HarnessBuilder) Build() *sporecore.StandardHarness {
	return sporecore.NewStandardHarness(b.BuildConfig())
}
