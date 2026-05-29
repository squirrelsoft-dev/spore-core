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
	"encoding/json"

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
	// content is the LLM-native content-capture guard + truncation limit (issue
	// #64). Default OFF, so the adapter populates none of the gen_ai.* span
	// fields and the durable JSONL stays pre-#64-identical.
	content ContentCaptureConfig
}

// NewHarnessObserver wraps an ObservabilityProvider as a
// sporecore.HarnessObserver, stamping turn-span cost via pricing. Content
// capture defaults OFF; use NewHarnessObserverWithContent to enable it.
func NewHarnessObserver(provider ObservabilityProvider, pricing PricingTable) *HarnessObservabilityAdapter {
	return &HarnessObservabilityAdapter{
		provider: provider,
		pricing:  pricing,
		content:  DefaultContentCaptureConfig(),
	}
}

// NewHarnessObserverWithContent is NewHarnessObserver plus an explicit
// ContentCaptureConfig (issue #64). When content.Enabled is true the adapter
// captures the model output text + requested tool calls on the turn span and
// the tool arguments + result on the tool-call span, each truncated to
// content.MaxFieldLen UTF-8 bytes.
func NewHarnessObserverWithContent(provider ObservabilityProvider, pricing PricingTable, content ContentCaptureConfig) *HarnessObservabilityAdapter {
	return &HarnessObservabilityAdapter{provider: provider, pricing: pricing, content: content}
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
	outputText string,
	calls []sporecore.ToolCall,
	inputMessages []sporecore.Message,
) {
	base := NewRoot(SpanID(spanID), sessionID, taskID, SpanKindTurn, Timestamp(startedAt))
	status := NewStatusOk()
	if errorMessage != "" {
		status = NewStatusError(errorMessage)
	}
	base.Finish(Timestamp(startedAt), status, durationMs)
	span := TurnSpan{
		Base:               base,
		TurnNumber:         turnNumber,
		InputTokens:        usage.InputTokens,
		OutputTokens:       usage.OutputTokens,
		CacheReadTokens:    usage.CacheReadTokens,
		CacheWriteTokens:   usage.CacheWriteTokens,
		CostUSD:            costUSD,
		StopReason:         stopReason,
		ToolCallsRequested: toolCallsRequested,
	}
	// Content capture (issue #64): output text + requested tool calls on the
	// turn span, gated on the guard and truncated to MaxFieldLen UTF-8 bytes.
	if a.content.Enabled {
		if outputText != "" {
			clipped, truncated := TruncateField(outputText, a.content.MaxFieldLen)
			span.OutputText = &GenAiMessage{
				Role:      GenAiRoleAssistant,
				Content:   clipped,
				Truncated: truncated,
			}
		}
		if len(calls) > 0 {
			tcs := make([]ToolCallContent, 0, len(calls))
			for _, c := range calls {
				tcs = append(tcs, toolCallContent(c.Name, c.Input, a.content.MaxFieldLen))
			}
			span.ToolCalls = tcs
		}
		if len(inputMessages) > 0 {
			span.InputMessages = captureInputMessages(inputMessages, a.content.MaxFieldLen)
		}
	}
	a.provider.EmitTurn(span)
}

// toolCallContent builds a captured ToolCallContent (issue #64), truncating the
// arguments by their JSON-byte length. A truncated argument cannot be clipped
// in place as JSON, so it is stored as a JSON string value carrying the marker.
func toolCallContent(name string, args json.RawMessage, maxLen int) ToolCallContent {
	if len(args) == 0 {
		return ToolCallContent{Name: name, Arguments: json.RawMessage("null")}
	}
	clipped, truncated := TruncateField(string(args), maxLen)
	if !truncated {
		return ToolCallContent{Name: name, Arguments: append(json.RawMessage(nil), args...)}
	}
	// Store the clipped form as a JSON string value.
	strBytes, err := json.Marshal(clipped)
	if err != nil {
		strBytes = json.RawMessage("null")
	}
	return ToolCallContent{Name: name, Arguments: strBytes, ArgumentsTruncated: true}
}

// captureInputMessages snapshots the assembled INPUT messages (the full prompt
// the model saw) into GenAiMessages (issue #64). Each message's Role maps to the
// conventional GenAiRole; Content is rendered to a plain string and truncated to
// maxLen UTF-8 bytes:
//   - text        → the text verbatim
//   - tool_result → the result body (role stays Tool)
//   - tool_call   → "<name> <compact-json-args>" (assistant)
//   - image       → "[image <media_type>]" — NEVER the base64 data
//
// System-first, then history order is preserved because the assembled messages
// already lead with the RoleSystem prompt; each message is mapped directly (no
// synthesized system entry).
func captureInputMessages(messages []sporecore.Message, maxLen int) []GenAiMessage {
	out := make([]GenAiMessage, 0, len(messages))
	for _, m := range messages {
		role := genAiRoleFor(m.Role)
		var rendered string
		switch m.Content.Type {
		case sporecore.ContentTypeText:
			rendered = m.Content.Text
		case sporecore.ContentTypeToolResult:
			if m.Content.ToolResult != nil {
				rendered = m.Content.ToolResult.Content
			}
		case sporecore.ContentTypeToolCall:
			if m.Content.ToolCall != nil {
				rendered = m.Content.ToolCall.Name + " " + string(m.Content.ToolCall.Input)
			}
		case sporecore.ContentTypeImage:
			// NEVER dump the base64 data — placeholder only.
			rendered = "[image " + m.Content.MediaType + "]"
		}
		clipped, truncated := TruncateField(rendered, maxLen)
		out = append(out, GenAiMessage{Role: role, Content: clipped, Truncated: truncated})
	}
	return out
}

// genAiRoleFor maps a model Role to the conventional GenAiRole (issue #64).
func genAiRoleFor(r sporecore.Role) GenAiRole {
	switch r {
	case sporecore.RoleSystem:
		return GenAiRoleSystem
	case sporecore.RoleUser:
		return GenAiRoleUser
	case sporecore.RoleAssistant:
		return GenAiRoleAssistant
	case sporecore.RoleTool:
		return GenAiRoleTool
	default:
		return GenAiRole(string(r))
	}
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
	arguments json.RawMessage,
	resultContent string,
) {
	// Reconstruct the parent envelope so NewChild can stamp the parent id.
	parent := NewRoot(SpanID(parentSpanID), sessionID, taskID, SpanKindTurn, Timestamp(startedAt))
	base := NewChild(SpanID(spanID), parent, SpanKindToolCall, Timestamp(startedAt))
	status := NewStatusOk()
	if isError {
		status = NewStatusError("tool returned a recoverable error")
	}
	base.Finish(Timestamp(startedAt), status, durationMs)
	span := ToolCallSpan{
		Base:                base,
		ToolName:            toolName,
		CallID:              callID,
		ParametersSizeBytes: parametersSizeBytes,
		OutputSizeBytes:     outputSizeBytes,
		Truncated:           truncated,
		SandboxMode:         "",
		SandboxViolations:   nil,
	}
	// Content capture (issue #64): tool arguments + result body on the tool-call
	// span, gated on the guard and truncated to MaxFieldLen UTF-8 bytes.
	if a.content.Enabled {
		tc := toolCallContent(toolName, arguments, a.content.MaxFieldLen)
		span.Arguments = &tc
		clipped, resTruncated := TruncateField(resultContent, a.content.MaxFieldLen)
		span.Result = &ToolResultContent{Content: clipped, Truncated: resTruncated}
	}
	a.provider.EmitToolCall(span)
}

// SetSessionOutcome records the terminal outcome on the wrapped provider,
// mapping the harness's 3-state TerminalOutcome onto the guideregistry
// SessionOutcome enum (issue #80: TerminalEscalated -> Escalated).
func (a *HarnessObservabilityAdapter) SetSessionOutcome(sessionID sporecore.SessionID, outcome sporecore.TerminalOutcome, failureReason string) {
	var so guideregistry.SessionOutcome
	switch outcome {
	case sporecore.TerminalFailure:
		so = guideregistry.NewOutcomeFailure(failureReason)
	case sporecore.TerminalEscalated:
		so = guideregistry.NewOutcomeEscalated()
	default:
		so = guideregistry.NewOutcomeSuccess()
	}
	a.provider.SetSessionOutcome(sessionID, so)
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
// a root context span with a Compaction operation carrying the real
// tokens_before / tokens_after / tokens_reclaimed the harness computed from the
// manager's post-compaction budget (issue #57). When the manager tracks no
// budget the loop passes tokens_after == tokens_before and zero reclaimed.
func (a *HarnessObservabilityAdapter) EmitCompaction(
	spanID string,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	startedAt string,
	messagesRemoved uint32,
	tokensBefore uint32,
	tokensAfter uint32,
	tokensReclaimed uint32,
) {
	base := NewRoot(SpanID(spanID), sessionID, taskID, SpanKindCompaction, Timestamp(startedAt))
	base.Finish(Timestamp(startedAt), NewStatusOk(), 0)
	a.provider.EmitContext(ContextSpan{
		Base:              base,
		Operation:         NewContextOpCompaction(messagesRemoved, tokensReclaimed),
		TokensBefore:      tokensBefore,
		TokensAfter:       tokensAfter,
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
	// content is the LLM-native content-capture config (issue #64). Defaults to
	// ContentCaptureConfigFromEnv() (OFF unless SPORE_TRACE_CONTENT is set).
	content ContentCaptureConfig
	// Compaction loop (issue #46). compactionVerifier defaults to
	// contextmgr.NewKeyTermVerifier(); maxCompactionAttempts defaults to 2.
	compactionVerifier    sporecore.CompactionVerifier
	maxCompactionAttempts uint32
	// spanStore is the optional ObservabilityStore leg of a StorageProvider
	// (issue #73), set via WithStorage. When present AND the configured
	// observability provider is the durable outbox, the builder wires it into
	// the outbox's fan-out so every span is ALSO appended to the store. Defaults
	// to nil (no-op): the harness never null-checks; an unconfigured store leg
	// simply contributes nothing.
	spanStore SpanStore
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
		content:               ContentCaptureConfigFromEnv(),
		compactionVerifier:    contextmgr.NewKeyTermVerifier(),
		maxCompactionAttempts: 2,
	}
}

// ContentCapture sets the LLM-native content-capture config (issue #64),
// overriding the env-derived default. Use this to enable capture
// programmatically (e.g. ContentCaptureConfig{Enabled: true, MaxFieldLen: 8192}).
func (b *HarnessBuilder) ContentCapture(c ContentCaptureConfig) *HarnessBuilder {
	b.content = c
	return b
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
// using the spec defaults. Honors SPORE_OTLP_ENDPOINT (comma-separated
// multi-endpoint fan-out, issue #73) for OTLP forwarding to Tempo over gRPC
// (issue #50); unset/empty means JSONL only. When a storage span-store leg was
// set via WithStorage, it is wired into the outbox's fan-out here so every span
// is ALSO appended to the store.
func (b *HarnessBuilder) WithObservabilityOutbox(root string) *HarnessBuilder {
	outbox := NewOutboxObservabilityProvider(NewOutboxConfig(root))
	if b.spanStore != nil {
		outbox = outbox.WithStore(b.spanStore)
	}
	return b.Observability(outbox)
}

// WithStorage wires the ObservabilityStore leg of a StorageProvider (issue #73)
// into the builder. Pass storageProvider.Observability(); the observability
// fan-out then ALSO appends every span to that store (in addition to the
// durable JSONL outbox and any OTLP endpoints). The other three storage domains
// (session, memory, run) are expose-only in v1 — the harness loop does not call
// them — so the builder takes only the observability leg, the one with runtime
// behavior. The split avoids an observability → storage import cycle: the store
// is accepted via the SpanStore seam, which storage.ObservabilityStore
// implementations satisfy structurally.
//
// Defaults to nil (no-op): an unconfigured store leg contributes nothing and
// the harness never null-checks. Must be called BEFORE WithObservabilityOutbox
// for the leg to be wired into a builder-constructed outbox.
func (b *HarnessBuilder) WithStorage(observabilityStore SpanStore) *HarnessBuilder {
	b.spanStore = observabilityStore
	return b
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
		cfg.Observability = NewHarnessObserverWithContent(b.provider, b.pricing, b.content)
	}
	return cfg
}

// Build assembles a ready-to-run *sporecore.StandardHarness.
func (b *HarnessBuilder) Build() *sporecore.StandardHarness {
	return sporecore.NewStandardHarness(b.BuildConfig())
}
