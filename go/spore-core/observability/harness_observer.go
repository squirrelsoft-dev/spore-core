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
	"sync/atomic"

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

// EmitHillClimbingIteration builds a WarnSpan and forwards it via the OPTIONAL
// WarnEmitter surface (issue #60). If the wrapped provider does not implement
// WarnEmitter, the warn is silently dropped — same contract as
// EmitCompactionVerificationFailed.
func (a *HarnessObservabilityAdapter) EmitHillClimbingIteration(
	spanID string,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	startedAt string,
	iteration uint32,
	metricValue float64,
	hasMetric bool,
	delta float64,
	hasDelta bool,
	status string,
	reverted bool,
) {
	emitter, ok := a.provider.(WarnEmitter)
	if !ok {
		return
	}
	base := NewRoot(SpanID(spanID), sessionID, taskID, SpanKindWarn, Timestamp(startedAt))
	var mv, d *float64
	if hasMetric {
		v := metricValue
		mv = &v
	}
	if hasDelta {
		dv := delta
		d = &dv
	}
	emitter.EmitWarn(NewWarnSpan(base, NewWarnHillClimbingIteration(iteration, mv, d, status, reverted)))
}

// EmitConsultSpawned builds a context span for a worker pausing mid-loop to
// consult a parent-spawned helper (issue #114) and forwards it. A lightweight
// root ContextAssembly span carrying the consult kind — alongside the existing
// SkillInjected context-operation family.
func (a *HarnessObservabilityAdapter) EmitConsultSpawned(
	spanID string,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	startedAt string,
	consultKind string,
) {
	base := NewRoot(SpanID(spanID), sessionID, taskID, SpanKindContextAssembly, Timestamp(startedAt))
	base.Finish(Timestamp(startedAt), NewStatusOk(), 0)
	a.provider.EmitContext(ContextSpan{
		Base:      base,
		Operation: NewContextOpConsultSpawned(consultKind),
	})
}

// EmitConsultResumed builds a context span for a paused worker resumed after a
// consult (issue #114) and forwards it. answered is false for a budget-exhausted
// soft-fail resume.
func (a *HarnessObservabilityAdapter) EmitConsultResumed(
	spanID string,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	startedAt string,
	consultKind string,
	answered bool,
) {
	base := NewRoot(SpanID(spanID), sessionID, taskID, SpanKindContextAssembly, Timestamp(startedAt))
	base.Finish(Timestamp(startedAt), NewStatusOk(), 0)
	a.provider.EmitContext(ContextSpan{
		Base:      base,
		Operation: NewContextOpConsultResumed(consultKind, answered),
	})
}

// EmitToolErrorLoopDetected builds a context span for the ReAct breaker detecting
// a tool-error loop at N (issue #137) and forwards it. A lightweight root
// ContextAssembly span carrying the offending tool and the consecutive-error
// count.
func (a *HarnessObservabilityAdapter) EmitToolErrorLoopDetected(
	spanID string,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	startedAt string,
	toolName string,
	consecutiveErrors uint32,
) {
	base := NewRoot(SpanID(spanID), sessionID, taskID, SpanKindContextAssembly, Timestamp(startedAt))
	base.Finish(Timestamp(startedAt), NewStatusOk(), 0)
	a.provider.EmitContext(ContextSpan{
		Base:      base,
		Operation: NewContextOpToolErrorLoopDetected(toolName, consecutiveErrors),
	})
}

// EmitToolErrorLoopBroken builds a context span for the ReAct breaker tripping at
// 2*N (issue #137) and forwards it.
func (a *HarnessObservabilityAdapter) EmitToolErrorLoopBroken(
	spanID string,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	startedAt string,
	toolName string,
	consecutiveErrors uint32,
) {
	base := NewRoot(SpanID(spanID), sessionID, taskID, SpanKindContextAssembly, Timestamp(startedAt))
	base.Finish(Timestamp(startedAt), NewStatusOk(), 0)
	a.provider.EmitContext(ContextSpan{
		Base:      base,
		Operation: NewContextOpToolErrorLoopBroken(toolName, consecutiveErrors),
	})
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
	// sandboxViolationPolicy controls how a tool-surfaced (or pre-dispatch
	// validate) SandboxViolation is handled (issue #150). The zero value (empty
	// string) leaves HarnessConfig.SandboxViolationPolicy empty, which the harness
	// reads as Recoverable — the cross-language default. Set via
	// SandboxViolationPolicy.
	sandboxViolationPolicy sporecore.SandboxViolationPolicy
	provider               ObservabilityProvider
	pricing                PricingTable
	// content is the LLM-native content-capture config (issue #64). Defaults to
	// ContentCaptureConfigFromEnv() (OFF unless SPORE_TRACE_CONTENT is set).
	content ContentCaptureConfig
	// Compaction loop (issue #46). compactionVerifier defaults to
	// contextmgr.NewKeyTermVerifier(); maxCompactionAttempts defaults to 2.
	compactionVerifier    sporecore.CompactionVerifier
	maxCompactionAttempts uint32
	// errorLoopThreshold is N, the consecutive-recoverable-tool-error breaker
	// trigger (issue #137). nil (the default) leaves HarnessConfig.ErrorLoopThreshold
	// nil, which the loop reads as the cross-language default of 3; an EXPLICIT
	// value via ErrorLoopThreshold(n) is honored verbatim (0 disables). See
	// ErrorLoopThreshold.
	errorLoopThreshold *uint32
	// spanStore is the optional ObservabilityStore leg of a StorageProvider
	// (issue #73), set via WithStorage. When present AND the configured
	// observability provider is the durable outbox, the builder wires it into
	// the outbox's fan-out so every span is ALSO appended to the store. Defaults
	// to nil (no-op): the harness never null-checks; an unconfigured store leg
	// simply contributes nothing.
	spanStore SpanStore
	// catalogueTools accumulates StandardTools (issue #81) added via Tool() /
	// Tools(). At build time they are folded into a fresh, populated
	// *StandardToolRegistry — a last-wins upsert (issue #81, Q1), so a tool added
	// later (e.g. a custom override) wins over an earlier standard tool of the
	// same name — placed on HarnessConfig.CatalogueRegistry. The run loop then
	// bridges that registry per-run (threading the run's SessionID + storage). This
	// mirrors the Rust builder's drain_tools_into_registry + catalogue_registry
	// seam. Nil accumulator preserves the ToolRegistry-only path.
	catalogueTools []sporecore.StandardTool
	// toolsetTools accumulates per-toolset catalogue StandardTools (Issue 2:
	// per-node toolset scoping), keyed by the non-empty toolset handle a leaf
	// carries. At build time each bucket is folded into its OWN populated
	// *StandardToolRegistry (last-wins upsert) and placed on
	// HarnessConfig.ToolsetCatalogues. A leaf whose toolset handle equals a key
	// dispatches ONLY that catalogue (strict per-node scoping); an EMPTY handle or
	// an unwired non-empty handle falls back to the global catalogueTools /
	// ToolRegistry seam. Mirrors the Rust builder's toolset_tools +
	// toolset_catalogues seam. Nil accumulator preserves the single-catalogue path.
	toolsetTools map[string][]sporecore.StandardTool
	// runStore / memStore are the optional storage seams threaded into catalogue
	// tools' ToolContext (issue #75/#78). When catalogue tools are present and
	// neither was set, the builder defaults runStore to an in-memory store so
	// session-aware tools persist within a run (the Go analog of Rust's in-memory
	// storage default).
	runStore sporecore.ToolRunStore
	memStore sporecore.ToolMemoryStore
	// systemPrompt is the optional operating system prompt prepended to each
	// turn's assembled context when the context manager renders none (issue #91).
	// Empty (the default) preserves today's behaviour.
	systemPrompt string
	// guides are structural Guides injected into every turn via ContextSources
	// (issue #115 / SC-26 / #9). Empty (the default) preserves today's behaviour.
	// See Guide / Guides.
	guides []sporecore.Guide
	// skills is the optional skill catalog (issue #115 / SC-26). Set via Skills,
	// which also registers the load_skill tool. nil (the default) means no
	// skills. See Skills.
	skills *sporecore.SkillCatalog
	// memory is the optional memory source (issue #163 / SC-26 follow-up). Set via
	// Memory. nil (the default) leaves memory empty. See Memory.
	memory *sporecore.MemoryConfig
	// modelParams are the authoritative per-run model sampling/decoding
	// parameters (issue #93). Builder params win: the harness replaces each
	// tool-requesting turn's Context.Params with this value unconditionally
	// right before the request is built. Zero value (the default) preserves
	// today's behaviour. See WithModelParams.
	modelParams sporecore.ModelParams
	// sessionStore / autoPersistSessions are the issue #102 opt-in
	// conversation-history threading seam. sessionStore is the SessionStore the
	// loop auto-loads from / auto-persists to; autoPersistSessions gates the whole
	// feature (default false → zero session-store I/O, byte-for-byte today's
	// behaviour). sessionStore is accepted via the consumer-side
	// sporecore.SessionStore interface, which a *storage.StorageProvider's
	// Session() store satisfies structurally — so the builder never imports the
	// storage package (which would form a cycle: storage imports observability).
	sessionStore        sporecore.SessionStore
	autoPersistSessions bool
	// promptToolCallFlag is the shared session-scoped flag for the adaptive
	// prompt-based tool-calling fallback (#111). Set ONLY by the conversational
	// preset (ConversationalBuilder), which also wraps the agent's model in an
	// AdaptiveToolCallModelInterface over this same pointer. Nil for every other
	// construction path, leaving HarnessConfig.PromptToolCallFlag nil so the
	// escalation seam is disabled and behaviour is byte-for-byte today's.
	promptToolCallFlag *atomic.Bool
	// projectNamespace is the STABLE durable namespace (#142) projected onto the
	// SessionID axis (a storage.ProjectID.Namespace() value). Set via
	// ProjectID(...). Empty (the default) leaves HarnessConfig.ProjectNamespace
	// empty, so durable artifacts fall back to the per-window session id —
	// today's single-process behaviour. When set, the builder ALSO wires it as
	// HarnessConfig.RunStore's durable namespace and, if Storage's run store
	// satisfies the harness-side RunStore interface, threads it through so the
	// task_list / plan / Ralph checkpoint persist under the project id.
	projectNamespace sporecore.SessionID
	// escalationMode is the HITL-vs-AFK escalation knob (#120/#155) threaded onto
	// HarnessConfig.EscalationMode at build time. The zero value (empty Kind) is
	// left unset, so HarnessConfig.EffectiveEscalationMode defaults it to
	// SurfaceToHuman — byte-for-byte today's behaviour. The SC-8 autonomous
	// presets set it to AutoContinueEscalation; a consumer overrides the whole
	// policy via EscalationMode(..). See EscalationMode.
	escalationMode sporecore.EscalationMode
	// metricEvaluator is the optional scoring evaluator (#124/SC-8) registered in
	// the built config's ExecutionRegistry under the default ("") handle, where
	// the HillClimbing loop strategy resolves it. nil (the default) registers
	// nothing. Set via MetricEvaluator (or the hill_climber preset). See
	// MetricEvaluator.
	metricEvaluator sporecore.MetricEvaluator
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
		// errorLoopThreshold left nil: the loop reads nil as the cross-language
		// default of 3 (effectiveErrorLoopThreshold). An EXPLICIT 0 via
		// ErrorLoopThreshold(0) disables the breaker.
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

// ErrorLoopThreshold sets N, the consecutive-recoverable-tool-error breaker
// trigger (issue #137): the ReAct turn loop injects ONE corrective message at N
// identical-argument recoverable errors for a tool and hard-stops at 2*N,
// resolving the node's BudgetExhaustedBehavior with HaltToolErrorLoop (the 2x
// multiplier is fixed). Defaults to 3 when unset. Pass 0 to disable the breaker.
func (b *HarnessBuilder) ErrorLoopThreshold(n uint32) *HarnessBuilder {
	b.errorLoopThreshold = &n
	return b
}

// Middleware injects a middleware chain.
func (b *HarnessBuilder) Middleware(m sporecore.MiddlewareChain) *HarnessBuilder {
	b.middleware = m
	return b
}

// SandboxViolationPolicy sets how a SandboxViolation surfaced by a tool (or the
// pre-dispatch validate check) is handled (issue #150): recoverable feedback to
// the model (the default) or a hard halt for halt-eligible Layer-1 violations
// (path escape / network violation). Layer-2 violations stay recoverable
// regardless. Defaults to sporecore.SandboxViolationRecoverable when unset.
func (b *HarnessBuilder) SandboxViolationPolicy(policy sporecore.SandboxViolationPolicy) *HarnessBuilder {
	b.sandboxViolationPolicy = policy
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

// Tool adds a single catalogue StandardTool (issue #81) to the builder. The
// tool is registered into the configured ToolRegistry at build time via the
// last-wins upsert (issue #81, Q1), so adding a custom tool under the same name
// as a standard tool (e.g. after Tools(StandardTools{}.CodingSet())) overrides
// it. Returns the receiver for fluent chaining.
func (b *HarnessBuilder) Tool(t sporecore.StandardTool) *HarnessBuilder {
	b.catalogueTools = append(b.catalogueTools, t)
	return b
}

// Tools adds many catalogue StandardTools at once (e.g. a preset like
// StandardTools{}.CodingSet()). Registered in order at build time; last-wins on
// name collisions (issue #81, Q1).
func (b *HarnessBuilder) Tools(ts ...sporecore.StandardTool) *HarnessBuilder {
	b.catalogueTools = append(b.catalogueTools, ts...)
	return b
}

// ToolsetTools registers catalogue StandardTools SCOPED to a single named
// toolset handle (Issue 2: per-node toolset scoping). Mirrors Tools, but instead
// of folding into the ONE global catalogue, these tools are folded into a
// per-key *StandardToolRegistry (last-wins upsert across calls and within the
// batch) and placed on HarnessConfig.ToolsetCatalogues.
//
// At dispatch, a leaf whose toolset handle equals key resolves ONLY this
// catalogue (bridged per-run with the run's sandbox/session/storage), so it
// cannot reach another node's tools — the union no longer leaks across nodes. A
// leaf with an EMPTY ("") handle, or a non-empty handle with no entry here,
// falls back to the global Tools catalogue / ToolRegistry seam (back-compat).
//
// key should be a NON-EMPTY handle string; the empty handle is reserved for the
// global-catalogue fallback. Additive across calls. Mirrors the Rust builder's
// toolset_tools. Returns the receiver for fluent chaining.
func (b *HarnessBuilder) ToolsetTools(key string, ts ...sporecore.StandardTool) *HarnessBuilder {
	if b.toolsetTools == nil {
		b.toolsetTools = make(map[string][]sporecore.StandardTool)
	}
	b.toolsetTools[key] = append(b.toolsetTools[key], ts...)
	return b
}

// Sandbox overrides the SandboxProvider — the only path catalogue tools have to
// the environment (filesystem, process exec). The sandbox is a required builder
// component (set at NewHarnessBuilder time), but catalogue file tools
// (read_file / write_file / list_dir) operate *through* the sandbox, so this
// setter lets an agent reach a real workspace-scoped sandbox via fluent
// chaining — e.g. b.Sandbox(workspace).Tools(StandardTools{}.CodingSet()) —
// without reconstructing the builder. Returns the receiver for chaining.
func (b *HarnessBuilder) Sandbox(sandbox sporecore.SandboxProvider) *HarnessBuilder {
	b.sandbox = sandbox
	return b
}

// ContextManager overrides the ContextManager that assembles per-turn context
// and drives compaction. ConversationalBuilder installs a
// StandardContextManager with the default compaction config (compaction at 80%
// of a 200K window); supply your own (e.g. one built with a lower threshold) to
// make compaction fire earlier for models with a smaller context window.
// Returns the receiver for fluent chaining.
func (b *HarnessBuilder) ContextManager(contextManager sporecore.ContextManager) *HarnessBuilder {
	b.contextManager = contextManager
	return b
}

// SystemPrompt sets an operating system prompt prepended to each turn's
// assembled context (issue #91).
//
// A context manager that renders no system prompt (e.g. the standard compaction
// adapter) leaves the model with only the task as a user message and no guidance
// on how to behave. When set, the run loop inserts this text as a leading
// System-role message each turn — but only when the assembled context does not
// already start with one, so a context manager that renders its own system prompt
// is preserved. Empty (the default) preserves today's behaviour. Returns the
// receiver for fluent chaining.
func (b *HarnessBuilder) SystemPrompt(text string) *HarnessBuilder {
	b.systemPrompt = text
	return b
}

// Guide registers a structural Guide (skill/playbook/domain knowledge) injected
// into every turn's assembled context via the ContextSources seam (issue #115 /
// SC-26 / #9). Unlike an ad-hoc User-message prepend, the guide is rendered into
// the leading System block by the production context manager. Call repeatedly to
// register several; order is preserved. Returns the receiver for fluent
// chaining.
func (b *HarnessBuilder) Guide(g sporecore.Guide) *HarnessBuilder {
	b.guides = append(b.guides, g)
	return b
}

// Guides registers several Guides at once (issue #115 / SC-26). Appends to any
// already registered via Guide. Returns the receiver for fluent chaining.
func (b *HarnessBuilder) Guides(gs ...sporecore.Guide) *HarnessBuilder {
	b.guides = append(b.guides, gs...)
	return b
}

// Skills registers a SkillCatalog (issue #115 / SC-26). This both (a) injects
// the catalog's manifest + active skill bodies into every turn's structural
// context (progressive disclosure) and (b) registers the load_skill tool,
// sharing the catalog's active set, so the model can activate a skill on demand.
// Replaces the architect-side skill-injecting context-manager shim. Returns the
// receiver for fluent chaining.
func (b *HarnessBuilder) Skills(catalog *sporecore.SkillCatalog) *HarnessBuilder {
	b.catalogueTools = append(b.catalogueTools, catalog.LoadSkillTool())
	b.skills = catalog
	return b
}

// Memory wires a memory source (issue #163 / SC-26 follow-up). The harness
// queries the provider each turn and injects the relevant memories into the
// structural System block — alongside guides + skills via the rich
// ContextSources seam — with no consumer-side context-manager shim.
//
// Build the config with memory.NewMemoryConfig(provider, opts...) to control the
// query policy, e.g. NewMemoryConfig(provider, memory.WithMinRelevance(0.6),
// memory.WithMaxItems(5)). Without an explicit query text the current task
// instruction is used, so retrieved memory tracks what the agent is working on.
// Not set (the default) leaves memory empty, byte-identical to the pre-#163 path.
// Returns the receiver for fluent chaining.
func (b *HarnessBuilder) Memory(cfg sporecore.MemoryConfig) *HarnessBuilder {
	b.memory = &cfg
	return b
}

// WithModelParams sets the authoritative model sampling/decoding parameters for
// the whole run (issue #93).
//
// These params are authoritative: the harness replaces each turn's
// Context.Params with this value UNCONDITIONALLY (builder params win) right
// before the request is built, so the configured params reach every agent turn
// that requests tools — the ReAct loop, the PlanExecute plan phase, the execute
// sub-loop, and the streaming path alike. (The internal compaction/summarization
// turn is intentionally left on defaults; it requests no tools, so decoding
// params are a no-op there.)
//
// Enabling ModelParams.StructuredToolCalls trades interleaved reasoning for one
// schema-constrained tool call per turn — useful for small local models that
// otherwise emit malformed tool calls. See ModelParams.StructuredToolCalls for
// the full behaviour contract. The zero value (the default) preserves today's
// behaviour. Returns the receiver for fluent chaining.
func (b *HarnessBuilder) WithModelParams(p sporecore.ModelParams) *HarnessBuilder {
	b.modelParams = p
	return b
}

// Storage wires the per-run storage seams threaded into catalogue tools'
// ToolContext (issue #75/#78): runStore is the structured-state store, memStore
// the episodic-memory store. Pass a *storage.StorageProvider's Run() and Memory()
// stores — they satisfy these consumer-side interfaces structurally, so the
// builder (in this package) never imports the storage package (which would form a
// cycle: storage imports observability). When catalogue tools are present and no
// storage was set, the builder defaults runStore to an in-memory store so
// session-aware tools persist within a run. Returns the receiver for chaining.
func (b *HarnessBuilder) Storage(runStore sporecore.ToolRunStore, memStore sporecore.ToolMemoryStore) *HarnessBuilder {
	b.runStore = runStore
	b.memStore = memStore
	return b
}

// ProjectID pins the STABLE project namespace (#142) the DURABLE artifacts (the
// task_list, the plan checkpoint, the Ralph progress / feature-list checkpoint,
// the active-run slot) are keyed by — instead of the per-window SessionID. Pass
// a storage.ProjectID's Namespace() (the project id projected onto the SessionID
// axis); the builder cannot import storage (a cycle), so the caller derives the
// ProjectID from the sandbox workspace root (decision 5) and projects it.
//
// Wiring it pins HarnessConfig.ProjectNamespace, and — when the run store passed
// to Storage(...) also satisfies the harness-side RunStore (Get+Put) — threads
// that SAME store into HarnessConfig.RunStore so the harness-side durable sites
// (PlanExecute task_list/plan persistence, deep-resume reconcile) and the Ralph
// checkpoint readers/writers persist under the project namespace too. Empty (the
// default) preserves today's session-keyed, single-process behaviour. Returns the
// receiver for fluent chaining.
func (b *HarnessBuilder) ProjectID(namespace sporecore.SessionID) *HarnessBuilder {
	b.projectNamespace = namespace
	return b
}

// SessionStore wires the conversation-history persistence store for opt-in
// session-state threading (issue #102). Pass a *storage.StorageProvider's
// Session() store — it satisfies the consumer-side sporecore.SessionStore
// interface structurally, so the builder (in this package) never imports the
// storage package (which would form a cycle: storage imports observability).
// Has no effect unless AutoPersistSessions(true) is also called. Returns the
// receiver for fluent chaining.
func (b *HarnessBuilder) SessionStore(store sporecore.SessionStore) *HarnessBuilder {
	b.sessionStore = store
	return b
}

// AutoPersistSessions opts this harness into the issue #102 session-store
// auto-load + auto-persist contract: the run loop loads the prior SessionState
// for the run's SessionID at the start of Run() (ReAct / SelfVerifying only; an
// explicit HarnessRunOptions.SessionState wins) and persists the post-run
// SessionState back at the terminal seam. Defaults to false — when false there
// is ZERO session-store I/O and the message flow + replay outcomes are
// byte-for-byte identical to today's. Pair with SessionStore to point at a
// concrete store; without one the never-null no-op store makes the flag inert.
// Returns the receiver for fluent chaining.
func (b *HarnessBuilder) AutoPersistSessions(enabled bool) *HarnessBuilder {
	b.autoPersistSessions = enabled
	return b
}

// EscalationMode selects the HITL-vs-AFK escalation mode (#120/#155) threaded
// onto the built config's EscalationMode: whether a spent step budget surfaces
// to a human (SurfaceToHumanEscalation, the default), proceeds without bound
// (AutonomousEscalation), or auto-continues a capped number of times
// (AutoContinueEscalation — what the SC-8 autonomous presets default to).
// Returns the receiver for fluent chaining.
func (b *HarnessBuilder) EscalationMode(mode sporecore.EscalationMode) *HarnessBuilder {
	b.escalationMode = mode
	return b
}

// MetricEvaluator registers the scoring evaluator the HillClimbing loop strategy
// requires (#124/SC-8) under the built config's default ("") evaluator handle —
// the handle a HillClimbing task with an empty Evaluator ref resolves. Pass an
// already-bridged sporecore.MetricEvaluator (e.g. metric.AsHarnessMetricEvaluator
// over a metric.MetricEvaluator). nil leaves the registry untouched. Returns the
// receiver for fluent chaining.
func (b *HarnessBuilder) MetricEvaluator(evaluator sporecore.MetricEvaluator) *HarnessBuilder {
	b.metricEvaluator = evaluator
	return b
}

// foldCatalogueRegistry folds every accumulated catalogue tool into a fresh,
// populated *StandardToolRegistry via Register() (a last-wins upsert) and returns
// it, or nil when no catalogue tools were added. Mirrors the Rust builder's
// drain_tools_into_registry: build() folds the tools, the run loop bridges the
// registry per-run. Best-effort: a registration error (e.g. an invalid custom
// schema) is silently skipped here — the registry's own validation is the gate,
// and the harness loop surfaces an unregistered tool at dispatch time. Drains the
// accumulator so a second build does not double-register.
func (b *HarnessBuilder) foldCatalogueRegistry() *sporecore.StandardToolRegistry {
	if len(b.catalogueTools) == 0 {
		return nil
	}
	reg := sporecore.NewStandardToolRegistry()
	for _, t := range b.catalogueTools {
		_ = reg.Register(t.Implementation, t.Schema)
	}
	b.catalogueTools = nil
	return reg
}

// foldToolsetCatalogues folds each accumulated per-toolset bucket into its OWN
// populated *StandardToolRegistry (last-wins upsert), keyed by the toolset
// handle, and returns the map, or nil when no toolset tools were added. Bridged
// per-run at dispatch — same as the global catalogue. Mirrors the Rust builder's
// build_config toolset_catalogues fold. Drains the accumulator so a second build
// does not double-register.
func (b *HarnessBuilder) foldToolsetCatalogues() map[string]*sporecore.StandardToolRegistry {
	if len(b.toolsetTools) == 0 {
		return nil
	}
	out := make(map[string]*sporecore.StandardToolRegistry, len(b.toolsetTools))
	for key, ts := range b.toolsetTools {
		reg := sporecore.NewStandardToolRegistry()
		for _, t := range ts {
			_ = reg.Register(t.Implementation, t.Schema)
		}
		out[key] = reg
	}
	b.toolsetTools = nil
	return out
}

// BuildConfig assembles the HarnessConfig without wrapping it in a harness.
func (b *HarnessBuilder) BuildConfig() sporecore.HarnessConfig {
	catalogue := b.foldCatalogueRegistry()
	toolsetCatalogues := b.foldToolsetCatalogues()
	// When catalogue tools are present and the caller wired no storage, default
	// the run store to an in-memory provider (not the no-op default) so that
	// session-aware tools (task_list, todo_write, memory) actually persist within
	// the run. Pure tools (read_file/write_file via the sandbox) are unaffected
	// either way. Mirrors the Rust in-memory storage default.
	runStore := b.runStore
	if (catalogue != nil || len(toolsetCatalogues) > 0) && runStore == nil && b.memStore == nil {
		runStore = sporecore.NewInMemoryToolRunStore()
	}
	cfg := sporecore.HarnessConfig{
		Agent:                  b.agent,
		ToolRegistry:           b.toolRegistry,
		Sandbox:                b.sandbox,
		SandboxViolationPolicy: b.sandboxViolationPolicy,
		ContextManager:         b.contextManager,
		TerminationPolicy:      b.terminationPolicy,
		Middleware:             b.middleware,
		CompactionVerifier:     b.compactionVerifier,
		MaxCompactionAttempts:  b.maxCompactionAttempts,
		ErrorLoopThreshold:     b.errorLoopThreshold,
		CatalogueRegistry:      catalogue,
		ToolsetCatalogues:      toolsetCatalogues,
		ToolRunStore:           runStore,
		ToolMemoryStore:        b.memStore,
		SystemPrompt:           b.systemPrompt,
		Guides:                 b.guides,
		Skills:                 b.skills,
		Memory:                 b.memory,
		ModelParams:            b.modelParams,
		SessionStore:           b.sessionStore,
		AutoPersistSessions:    b.autoPersistSessions,
		PromptToolCallFlag:     b.promptToolCallFlag,
		ProjectNamespace:       b.projectNamespace,
	}
	// #142: when a project namespace is pinned, also wire the harness-side durable
	// RunStore (the PlanExecute task_list/plan checkpoint + the Ralph checkpoint)
	// from the SAME store wired via Storage(...). The ToolRunStore (Get+Put)
	// satisfies the harness-side RunStore (also Get+Put) structurally, so the
	// task_list the standalone tool writes and the one the harness loop reads
	// share one backend under the project namespace.
	if b.projectNamespace != "" {
		if rs, ok := runStore.(sporecore.RunStore); ok {
			cfg.RunStore = rs
		}
	}
	// #120/#155: thread the explicit escalation mode (left unset → the config's
	// EffectiveEscalationMode defaults it to SurfaceToHuman).
	if b.escalationMode.Kind != "" {
		cfg = cfg.WithEscalationMode(b.escalationMode)
	}
	// #124/SC-8: register the scoring evaluator under the default ("") handle so a
	// HillClimbing task with an empty Evaluator ref resolves it.
	if b.metricEvaluator != nil {
		cfg = cfg.WithRegistryMetricEvaluator("", b.metricEvaluator)
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
