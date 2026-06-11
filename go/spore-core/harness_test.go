package sporecore

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func turnUsage() TokenUsage { return TokenUsage{InputTokens: 1, OutputTokens: 1} }

func standardCfg(agent Agent) HarnessConfig {
	return HarnessConfig{
		Agent:             agent,
		ToolRegistry:      NewScriptedToolRegistry(),
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
		// #130: the shared test config drives the AUTONOMOUS (propagate-up)
		// escalation path so the pre-#130 budget-exhaustion tests keep asserting
		// their original Failure{BudgetExceeded} / propagate behavior. The new HITL
		// pause path is exercised by tests that build a SurfaceToHuman config
		// explicitly (see surfaceCfg in harness_budget_escalation_130_test.go).
		EscalationMode: AutonomousEscalation(),
		// #137: default N to 3, matching the builder default and the Rust reference
		// standard_config. Tests that exercise the breaker override this explicitly.
		ErrorLoopThreshold: 3,
	}
}

func reactTask(max uint32) Task {
	return NewTask("do something", SessionID("s1"), ReActStrategy(max))
}

// Rule: Harness owns the loop; a single FinalResponse returns Success.
func TestFinalResponseReturnsSuccess(t *testing.T) {
	a := NewMockAgent("test")
	a.Push(NewFinalResponse("done", turnUsage()))
	h := NewStandardHarness(standardCfg(a))
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess {
		t.Fatalf("kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Output != "done" || r.Turns != 1 {
		t.Fatalf("got output=%q turns=%d", r.Output, r.Turns)
	}
}

// Rule: tool calls dispatched, loop continues to the final response.
func TestToolCallThenFinalResponseLoops(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	a.Push(NewFinalResponse("after-tool", turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "tool-ok"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess || r.Output != "after-tool" || r.Turns != 2 {
		t.Fatalf("got %+v", r)
	}
	if reg.CallCount.Load() != 1 {
		t.Fatalf("call count = %d", reg.CallCount.Load())
	}
}

// Rule: parallel tool calls all dispatched in one turn.
func TestParallelToolCallsAllDispatched(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "a", Name: "x", Input: json.RawMessage(`{}`)},
		{ID: "b", Name: "y", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	a.Push(NewFinalResponse("ok", turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "1"})
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "2"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	_ = h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if reg.CallCount.Load() != 2 {
		t.Fatalf("call count = %d", reg.CallCount.Load())
	}
}

// Rule: budget overrun terminates with explicit reason.
func TestBudgetMaxTurnsExceeded(t *testing.T) {
	a := NewMockAgent("t")
	for i := 0; i < 10; i++ {
		a.Push(NewToolCallRequested([]ToolCall{
			{ID: "c", Name: "x", Input: json.RawMessage(`{}`)},
		}, turnUsage()))
	}
	h := NewStandardHarness(standardCfg(a))
	task := reactTask(100)
	two := uint32(2)
	task.Budget.MaxTurns = &two
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltBudgetExceeded || r.Reason.LimitType != BudgetLimitTurns {
		t.Fatalf("got %+v", r)
	}
	if r.Turns != 2 {
		t.Fatalf("turns = %d", r.Turns)
	}
}

// Rule: A turn with neither tool call nor response is an error.
func TestAgentErrorTerminatesWithAgentErrorHaltReason(t *testing.T) {
	a := NewMockAgent("t")
	u := turnUsage()
	a.Push(NewTurnError(NewEmptyResponseError(), &u))
	h := NewStandardHarness(standardCfg(a))
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltAgentError {
		t.Fatalf("got %+v", r)
	}
}

// Rule: Layer-1 SandboxViolation PathEscape halts unconditionally.
func TestLayer1PathEscapeHalts(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c", Name: "read", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	sb := NewScriptedSandbox()
	sb.Push(&SandboxViolation{Kind: SandboxPathEscape, Path: "/etc/passwd"})
	cfg.Sandbox = sb
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltSandboxViolation || r.Reason.Violation.Kind != SandboxPathEscape {
		t.Fatalf("got %+v", r)
	}
}

// Rule: Layer-2 recoverable sandbox violation appends as tool error, loop continues.
func TestLayer2PathDeniedContinuesAsToolError(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c", Name: "read", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	a.Push(NewFinalResponse("ack", turnUsage()))
	cfg := standardCfg(a)
	sb := NewScriptedSandbox()
	sb.Push(&SandboxViolation{Kind: SandboxPathDenied, Path: "/p"})
	cfg.Sandbox = sb
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess || r.Turns != 2 {
		t.Fatalf("got %+v", r)
	}
}

// Rule: TerminationPolicy::Halt overrides final response.
func TestTerminationPolicyHaltOverridesSuccess(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := standardCfg(a)
	tp := NewScriptedTerminationPolicy()
	tp.Push(TerminationDecision{Kind: TerminationHalt, Reason: "not yet"})
	cfg.TerminationPolicy = tp
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltTerminationPolicyHalt || r.Reason.Reason != "not yet" {
		t.Fatalf("got %+v", r)
	}
}

// Rule: Middleware::Halt at BeforeTurn halts before model call.
func TestMiddlewareHaltBeforeTurn(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("unused", turnUsage()))
	cfg := standardCfg(a)
	mw := NewScriptedMiddleware()
	mw.Push(HookBeforeTurn, MiddlewareDecision{Kind: MiddlewareHalt, Reason: "blocked"})
	cfg.Middleware = mw
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltMiddlewareHalt {
		t.Fatalf("got %+v", r)
	}
	if r.Reason.Hook != HookBeforeTurn || r.Reason.Reason != "blocked" || r.Turns != 0 {
		t.Fatalf("hook=%q reason=%q turns=%d", r.Reason.Hook, r.Reason.Reason, r.Turns)
	}
}

// Rule: Middleware::SurfaceToHuman at BeforeTool returns WaitingForHuman with pending calls.
func TestMiddlewareSurfaceToHumanBeforeTool(t *testing.T) {
	a := NewMockAgent("t")
	calls := []ToolCall{{ID: "c", Name: "x", Input: json.RawMessage(`{}`)}}
	a.Push(NewToolCallRequested(calls, turnUsage()))
	cfg := standardCfg(a)
	mw := NewScriptedMiddleware()
	req := HumanRequest{Kind: HumanReqToolApproval, Calls: calls, RiskLevel: RiskMedium}
	mw.Push(HookBeforeTool, MiddlewareDecision{Kind: MiddlewareSurfaceToHuman, Request: &req})
	cfg.Middleware = mw
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunWaitingForHuman || r.State == nil || len(r.State.PendingToolCalls) != 1 {
		t.Fatalf("got %+v", r)
	}
	if r.State.ChildState != nil {
		t.Fatalf("child state must be nil")
	}
}

// Rule: always_halt tool annotation halts the loop.
func TestAlwaysHaltToolHalts(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c", Name: "danger", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.MarkAlwaysHalt("danger")
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltUnrecoverableToolError || r.Reason.Tool != "danger" {
		t.Fatalf("got %+v", r)
	}
}

// Rule: Unrecoverable tool error halts loop immediately.
func TestUnrecoverableToolErrorHalts(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c", Name: "x", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputError, Message: "boom", Recoverable: false})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltUnrecoverableToolError || r.Reason.Error != "boom" {
		t.Fatalf("got %+v", r)
	}
}

// Rule: WaitingForHuman from a tool dispatch propagates to RunResult.
func TestToolWaitingForHumanPropagates(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c", Name: "subagent", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	req := HumanRequest{Kind: HumanReqClarification, Question: "?"}
	child := &ChildPausedState{
		SessionID: "child", TaskID: "ct", TurnNumber: 1,
		HumanRequest: &req, Task: reactTask(1), ParentToolCallID: "c",
	}
	reg.Push(ToolOutput{Kind: ToolOutputWaitingForHuman, ChildState: child, Request: &req})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunWaitingForHuman || r.State == nil || r.State.ChildState == nil {
		t.Fatalf("got %+v", r)
	}
}

// Rule: resume() with Halt returns Failure HumanHalted.
func TestResumeWithHaltReturnsHumanHalted(t *testing.T) {
	a := NewMockAgent("t")
	h := NewStandardHarness(standardCfg(a))
	state := PausedState{
		SessionID: "s", TaskID: "t", TurnNumber: 1,
		HumanRequest: &HumanRequest{Kind: HumanReqClarification, Question: "?"},
		Task:         reactTask(5),
	}
	r := h.Resume(context.Background(), state, HumanResponse{Kind: HumanRespHalt}, nil)
	if r.Kind != RunFailure || r.Reason.Kind != HaltHumanHalted {
		t.Fatalf("got %+v", r)
	}
}

// Rule: resume() with Allow dispatches pending tool calls then continues loop.
func TestResumeWithAllowExecutesPendingAndContinues(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "tool-ok"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	state := PausedState{
		SessionID: "s", TaskID: "t", TurnNumber: 1,
		PendingToolCalls: []ToolCall{{ID: "c", Name: "x", Input: json.RawMessage(`{}`)}},
		HumanRequest:     &HumanRequest{Kind: HumanReqToolApproval, RiskLevel: RiskLow},
		Task:             reactTask(5),
	}
	r := h.Resume(context.Background(), state, HumanResponse{Kind: HumanRespAllow}, nil)
	if r.Kind != RunSuccess || r.Output != "done" {
		t.Fatalf("got %+v", r)
	}
	if reg.CallCount.Load() != 1 {
		t.Fatalf("call count = %d", reg.CallCount.Load())
	}
}

// Rule: an UNKNOWN loop strategy is explicitly marked NotYetImplemented.
// All five spec'd strategies are now fully implemented: ReAct (this file),
// PlanExecute (issues #70 + #59, plan_test.go + execute-phase tests below),
// SelfVerifying (issue #61, self_verifying_test.go), Ralph (issue #58,
// ralph_test.go), and HillClimbing (issue #60, hill_climbing_test.go). The
// NotYetImplemented stub is now reached only by an unrecognized strategy kind —
// the dispatch's default arm.
// #124: with the central dispatch switch removed, an unknown LoopStrategy kind
// (a Go-only edge case — Rust's closed enum cannot express it) is rejected by
// the enum→config delegation in LoopStrategy.Run as a typed Failed outcome,
// which driveStrategy maps to a HaltConfigurationError carrying the typed
// InvalidConfigurationError — never a panic.
func TestUnknownStrategyIsConfigurationError(t *testing.T) {
	a := NewMockAgent("t")
	h := NewStandardHarness(standardCfg(a))
	task := NewTask("do it", SessionID("s"), LoopStrategy{Kind: LoopStrategyKind("not_a_real_strategy")})
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltConfigurationError {
		t.Fatalf("got %+v", r)
	}
	if _, ok := r.Reason.ConfigError.(*InvalidConfigurationError); !ok {
		t.Fatalf("expected InvalidConfigurationError, got %T", r.Reason.ConfigError)
	}
}

// Rule: Aggregate usage accumulates across turns.
func TestAggregateUsageAccumulates(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c", Name: "x", Input: json.RawMessage(`{}`)},
	}, TokenUsage{InputTokens: 10, OutputTokens: 5}))
	a.Push(NewFinalResponse("ok", TokenUsage{InputTokens: 7, OutputTokens: 3}))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "x"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess || r.Usage.InputTokens != 17 || r.Usage.OutputTokens != 8 {
		t.Fatalf("got %+v", r)
	}
}

// Rule: ModelError surfaces as AgentError → HaltReason AgentError.
func TestModelErrorHaltsViaAgentError(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewTurnError(NewModelAgentError(NewTimeout()), nil))
	h := NewStandardHarness(standardCfg(a))
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltAgentError {
		t.Fatalf("got %+v", r)
	}
	if r.Reason.AgentError == nil || r.Reason.AgentError.Kind != AgentErrModelError {
		t.Fatalf("inner agent error = %+v", r.Reason.AgentError)
	}
}

// Serde round-trip — fixture portability.
func TestRunResultRoundtripJSON(t *testing.T) {
	r := RunResult{
		Kind:      RunFailure,
		Reason:    HaltReason{Kind: HaltBudgetExceeded, LimitType: BudgetLimitTurns},
		SessionID: "s",
		Turns:     3,
	}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var back RunResult
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back.Kind != r.Kind || back.Reason.Kind != HaltBudgetExceeded || back.Reason.LimitType != BudgetLimitTurns {
		t.Fatalf("roundtrip mismatch: %s -> %+v", data, back)
	}
	if back.Turns != 3 || back.SessionID != "s" {
		t.Fatalf("roundtrip mismatch: %+v", back)
	}
}

func TestPausedStateRoundtripJSON(t *testing.T) {
	ps := PausedState{
		SessionID: "s", TaskID: "t", TurnNumber: 4,
		PendingToolCalls: []ToolCall{{ID: "c", Name: "x", Input: json.RawMessage(`{"k":1}`)}},
		HumanRequest:     &HumanRequest{Kind: HumanReqClarification, Question: "what?"},
		Task:             reactTask(5),
		BudgetUsed:       BudgetSnapshot{Turns: 4, InputTokens: 100, OutputTokens: 50},
	}
	data, err := json.Marshal(ps)
	if err != nil {
		t.Fatal(err)
	}
	var back PausedState
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back.SessionID != "s" || back.TurnNumber != 4 || back.HumanRequest.Question != "what?" {
		t.Fatalf("roundtrip mismatch: %s -> %+v", data, back)
	}
	if back.Task.LoopStrategy.Kind != StrategyReAct || back.Task.LoopStrategy.MaxIterations() != 5 {
		t.Fatalf("task strategy lost: %+v", back.Task.LoopStrategy)
	}
}

// ChildPausedState cannot nest itself — depth-1 enforcement.
func TestChildPausedStateHasNoChildField(t *testing.T) {
	cs := ChildPausedState{
		SessionID: "c", TaskID: "ct", TurnNumber: 1,
		HumanRequest:     &HumanRequest{Kind: HumanReqClarification, Question: "?"},
		Task:             reactTask(1),
		ParentToolCallID: "p",
	}
	data, err := json.Marshal(cs)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), `"child_state"`) {
		t.Fatalf("ChildPausedState must not have child_state field: %s", data)
	}
}

// JSON tag values match Rust serde snake_case for the loop strategy.
func TestLoopStrategyJSONTags(t *testing.T) {
	s := ReActStrategy(5)
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"kind":"react","budget":{"kind":"per_loop","value":5},"behavior":{"kind":"escalate"},"agent":"","toolset":""}`
	if string(data) != want {
		t.Fatalf("got %s, want %s", data, want)
	}
}

// HaltReason agent_error embeds the AgentError under "error".
func TestHaltReasonAgentErrorShape(t *testing.T) {
	r := HaltReason{Kind: HaltAgentError, AgentError: NewEmptyResponseError()}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"kind":"agent_error","error":{"kind":"EmptyResponse"}}`
	if string(data) != want {
		t.Fatalf("got %s", data)
	}
}

// HaltReason context_error embeds the ContextError under "error" and
// round-trips (issue #32). Mirrors the agent_error shape exactly.
func TestHaltReasonContextErrorShape(t *testing.T) {
	r := HaltReason{Kind: HaltContextError, ContextError: &ContextError{
		Kind:       ContextErrCacheHashMismatch,
		Block:      "per_session",
		Expected:   1,
		Actual:     2,
		TurnNumber: 2,
	}}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"kind":"context_error","error":{"kind":"cache_hash_mismatch","block":"per_session","expected":1,"actual":2,"turn_number":2}}`
	if string(data) != want {
		t.Fatalf("got %s", data)
	}

	var back HaltReason
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back.Kind != HaltContextError || back.ContextError == nil {
		t.Fatalf("round-trip lost variant: %+v", back)
	}
	ce := back.ContextError
	if ce.Kind != ContextErrCacheHashMismatch || ce.Block != "per_session" || ce.TurnNumber != 2 || ce.Expected != 1 || ce.Actual != 2 {
		t.Fatalf("round-trip mismatch: %+v", ce)
	}
}

// ── Assistant-turn recording (regression for lost conversation history) ──────

// recordingContextManager is a ContextManager that records every message the
// loop appends (tool results, user messages, and — via the optional
// AssistantMessageAppender seam — assistant turns) into a flat ordered log so
// tests can assert positional ordering. Mirrors NoopContextManager but exposes
// the message log. The harness loop is single-goroutine, so no locking is
// needed. Each recorded entry preserves its role; tool results keep the call id
// so ordering can be asserted against the assistant tool-call message (Go's
// production adapter renders tool results as plain text without a call_id, so
// the double tracks the id itself — production shape is unchanged).
type recordingContextManager struct {
	roles   []Role
	callIDs []string // tool-call id for assistant ToolCall / tool result rows; "" otherwise
	texts   []string // text body for assistant Text / user rows; "" otherwise
}

func (m *recordingContextManager) Assemble(_ context.Context, session *SessionState, _ *Task) Context {
	return Context{Messages: append([]Message(nil), session.Messages...)}
}

func (m *recordingContextManager) AppendToolResult(_ context.Context, session *SessionState, result *HarnessToolResult) {
	msg := Message{Role: RoleTool, Content: NewTextContent(result.Output.Content)}
	session.Messages = append(session.Messages, msg)
	m.roles = append(m.roles, RoleTool)
	m.callIDs = append(m.callIDs, result.CallID)
	m.texts = append(m.texts, "")
}

func (m *recordingContextManager) AppendUserMessage(_ context.Context, session *SessionState, text string) {
	msg := Message{Role: RoleUser, Content: NewTextContent(text)}
	session.Messages = append(session.Messages, msg)
	m.roles = append(m.roles, RoleUser)
	m.callIDs = append(m.callIDs, "")
	m.texts = append(m.texts, text)
}

func (m *recordingContextManager) ShouldCompact(*SessionState) bool { return false }

func (m *recordingContextManager) AppendAssistantMessage(_ context.Context, session *SessionState, message Message) {
	session.Messages = append(session.Messages, message)
	m.roles = append(m.roles, message.Role)
	id := ""
	text := ""
	if message.Content.ToolCall != nil {
		id = message.Content.ToolCall.ID
	}
	if message.Content.Type == ContentTypeText {
		text = message.Content.Text
	}
	m.callIDs = append(m.callIDs, id)
	m.texts = append(m.texts, text)
}

// assistantToolCallIndex returns the index of the recorded assistant message
// carrying tool-call id, or -1.
func (m *recordingContextManager) assistantToolCallIndex(id string) int {
	for i := range m.roles {
		if m.roles[i] == RoleAssistant && m.callIDs[i] == id {
			return i
		}
	}
	return -1
}

// toolResultIndex returns the index of the recorded tool result for call id, or -1.
func (m *recordingContextManager) toolResultIndex(id string) int {
	for i := range m.roles {
		if m.roles[i] == RoleTool && m.callIDs[i] == id {
			return i
		}
	}
	return -1
}

var _ ContextManager = (*recordingContextManager)(nil)
var _ AssistantMessageAppender = (*recordingContextManager)(nil)

// Regression: a turn that requests a tool call must record the assistant's
// tool-call message in history, positioned BEFORE the tool result, so the next
// turn's assembled context reflects what the agent already did.
func TestToolCallRecordsAssistantMessageBeforeResult(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c1", Name: "read_file", Input: json.RawMessage(`{"path":"a.txt"}`)},
	}, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))
	cm := &recordingContextManager{}
	cfg := standardCfg(a)
	cfg.ContextManager = cm
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "contents"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess {
		t.Fatalf("got %+v", r)
	}
	ai := cm.assistantToolCallIndex("c1")
	ti := cm.toolResultIndex("c1")
	if ai < 0 {
		t.Fatalf("assistant tool-call message must be recorded; roles=%v", cm.roles)
	}
	if ti < 0 {
		t.Fatalf("tool result must be recorded; roles=%v", cm.roles)
	}
	if ai >= ti {
		t.Fatalf("assistant tool_use (idx %d) must precede its tool result (idx %d)", ai, ti)
	}
}

// Regression: a final response must append the assistant's text to history so a
// continued session sees what the agent said.
func TestFinalResponseRecordsAssistantText(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("the final answer", turnUsage()))
	cm := &recordingContextManager{}
	cfg := standardCfg(a)
	cfg.ContextManager = cm
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess {
		t.Fatalf("got %+v", r)
	}
	found := false
	for i := range cm.roles {
		if cm.roles[i] == RoleAssistant && cm.texts[i] == "the final answer" {
			found = true
		}
	}
	if !found {
		t.Fatalf("assistant final text must be recorded; roles=%v texts=%v", cm.roles, cm.texts)
	}
}

// Regression: when a run pauses at BeforeTool (SurfaceToHuman) and is then
// resumed with Allow, the assistant tool-call message must already be in history
// — recorded before the pause — positioned before its tool result, with no
// duplicate from the resume path.
func TestResumeAfterSurfaceToHumanRecordsAssistantOnceBeforeResult(t *testing.T) {
	a := NewMockAgent("t")
	calls := []ToolCall{{ID: "c1", Name: "read_file", Input: json.RawMessage(`{"path":"a.txt"}`)}}
	a.Push(NewToolCallRequested(calls, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))
	cm := &recordingContextManager{}
	cfg := standardCfg(a)
	cfg.ContextManager = cm
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "contents"})
	cfg.ToolRegistry = reg
	mw := NewScriptedMiddleware()
	req := HumanRequest{Kind: HumanReqToolApproval, Calls: calls, RiskLevel: RiskMedium}
	mw.Push(HookBeforeTool, MiddlewareDecision{Kind: MiddlewareSurfaceToHuman, Request: &req})
	cfg.Middleware = mw
	h := NewStandardHarness(cfg)

	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunWaitingForHuman || r.State == nil {
		t.Fatalf("expected WaitingForHuman, got %+v", r)
	}
	r = h.Resume(context.Background(), *r.State, HumanResponse{Kind: HumanRespAllow}, nil)
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success on resume, got %+v", r)
	}

	ai := cm.assistantToolCallIndex("c1")
	ti := cm.toolResultIndex("c1")
	if ai < 0 {
		t.Fatalf("assistant tool-call must be recorded on the resume path; roles=%v", cm.roles)
	}
	if ti < 0 {
		t.Fatalf("tool result must be recorded; roles=%v", cm.roles)
	}
	if ai >= ti {
		t.Fatalf("assistant tool_use (idx %d) must precede its tool result (idx %d)", ai, ti)
	}
	count := 0
	for i := range cm.roles {
		if cm.roles[i] == RoleAssistant && cm.callIDs[i] == "c1" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("assistant tool-call must be recorded exactly once, got %d", count)
	}
}
