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
	}
}

func reactTask(max uint32) Task {
	return NewTask("do something", SessionID("s1"), LoopStrategy{Kind: StrategyReAct, MaxIterations: max})
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
		HumanRequest: req, Task: reactTask(1), ParentToolCallID: "c",
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
		HumanRequest: HumanRequest{Kind: HumanReqClarification, Question: "?"},
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
		HumanRequest:     HumanRequest{Kind: HumanReqToolApproval, RiskLevel: RiskLow},
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

// Rule: non-ReAct strategies are explicitly marked NotYetImplemented.
func TestNonReactStrategiesMarkedNotYetImplemented(t *testing.T) {
	a := NewMockAgent("t")
	h := NewStandardHarness(standardCfg(a))
	strategies := []LoopStrategy{
		{Kind: StrategyRalph},
		{Kind: StrategySelfVerifying},
		{Kind: StrategyPlanExecute},
		{Kind: StrategyHillClimbing, Direction: OptimizationMaximize},
	}
	for _, s := range strategies {
		task := NewTask("do it", SessionID("s"), s)
		r := h.Run(context.Background(), NewHarnessRunOptions(task))
		if r.Kind != RunFailure || r.Reason.Kind != HaltStrategyNotYetImplemented {
			t.Fatalf("strategy %q got %+v", s.Kind, r)
		}
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
		HumanRequest:     HumanRequest{Kind: HumanReqClarification, Question: "what?"},
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
	if back.Task.LoopStrategy.Kind != StrategyReAct || back.Task.LoopStrategy.MaxIterations != 5 {
		t.Fatalf("task strategy lost: %+v", back.Task.LoopStrategy)
	}
}

// ChildPausedState cannot nest itself — depth-1 enforcement.
func TestChildPausedStateHasNoChildField(t *testing.T) {
	cs := ChildPausedState{
		SessionID: "c", TaskID: "ct", TurnNumber: 1,
		HumanRequest:     HumanRequest{Kind: HumanReqClarification, Question: "?"},
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
	s := LoopStrategy{Kind: StrategyReAct, MaxIterations: 5}
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"kind":"re_act","max_iterations":5}`
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
