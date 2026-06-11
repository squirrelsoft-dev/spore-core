package sporecore

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// ============================================================================
// Consecutive-recoverable-tool-error breaker (issue #137).
//
// Mirrors the Rust reference unit tests (the tel_* tests in
// rust/crates/spore-core/src/harness.rs) and the shared replay fixture
// fixtures/model_responses/harness/tool_error_loop.jsonl. Same types, same
// rules, same outcomes.
//
// Rules under test:
//   - HarnessConfig.ErrorLoopThreshold (N, default 3); hard stop at 2N.
//   - Per-tool errorRun{args, count, injected} loop-local counter:
//       * identical-args recoverable error -> count += 1
//       * args change OR first error -> fresh run at count 1
//       * ANY success for the tool -> run removed (AC1 reset)
//   - At N: ONE corrective USER message (enrichToolError schema+hint), AC2.
//   - At 2N: stop -> resolve node BudgetExhaustedBehavior, terminal carries
//     HaltToolErrorLoop (never HaltBudgetExceeded), AC3.
//   - At BOTH thresholds: a StreamEvent + an observability op, AC4.
// ============================================================================

const telBadMsg = "missing required parameter `description`"

// telBadCall is the malformed add_task call the weak model repeats (gemma's
// task_list/add_task-without-description scenario).
func telBadCall(args string) ToolCall {
	return ToolCall{ID: "call_bad", Name: "add_task", Input: json.RawMessage(args)}
}

const telBadArgs = `{"task_list_id":"tl1"}`

// telPushBad pushes k identical malformed add_task tool-call turns.
func telPushBad(a *MockAgent, k int, args string) {
	for i := 0; i < k; i++ {
		a.Push(NewToolCallRequested([]ToolCall{telBadCall(args)}, turnUsage()))
	}
}

// telErrRegistry is a tool registry that always returns the same recoverable
// error and advertises the add_task schema (so enrichToolError can render the
// schema+hint).
func telErrRegistry() *ScriptedToolRegistry {
	reg := NewScriptedToolRegistry()
	reg.AlwaysRecoverableError(telBadMsg)
	reg.WithSchema(RegistryToolSchema{
		Name:        "add_task",
		Description: "add a task to a task list",
		Parameters: json.RawMessage(`{"type":"object","properties":` +
			`{"task_list_id":{"type":"string"},"description":{"type":"string"}},` +
			`"required":["task_list_id","description"]}`),
	})
	return reg
}

// telUserMsgs returns the user-role text messages of a post-run session.
func telUserMsgs(state SessionState) []string {
	var out []string
	for _, m := range state.Messages {
		if m.Role == RoleUser && m.Content.Type == ContentTypeText {
			out = append(out, m.Content.Text)
		}
	}
	return out
}

// telLeaf builds a bare ReAct leaf with the given behavior + PerLoop budget.
func telLeaf(behavior BudgetExhaustedBehavior, budget uint32) Task {
	return NewTask("add a task to the task list", SessionID("s1"),
		LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
			Budget:   BudgetPolicy{Kind: BudgetPerLoop, Value: budget},
			Behavior: behavior,
			Agent:    AgentRef(""),
			Toolset:  ToolsetRef(""),
		}})
}

// AC1: a success in the middle resets the counter; the breaker never trips.
// error, error, SUCCESS, error, error -> 4 errors but the longest identical-args
// run is 2 (< N), so no trip.
func TestTELAC1SuccessResetsCounterNoTrip(t *testing.T) {
	a := NewMockAgent("t")
	telPushBad(a, 2, telBadArgs)
	a.Push(NewToolCallRequested([]ToolCall{telBadCall(telBadArgs)}, turnUsage()))
	telPushBad(a, 2, telBadArgs)
	a.Push(NewFinalResponse("done", turnUsage()))

	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	for _, recoverableErr := range []bool{true, true, false, true, true} {
		if recoverableErr {
			reg.Push(ToolOutput{Kind: ToolOutputError, Message: telBadMsg, Recoverable: true})
		} else {
			reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "added"})
		}
	}
	cfg.ToolRegistry = reg
	cfg.ErrorLoopThreshold = 3

	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(20)))
	if r.Kind != RunSuccess || r.Output != "done" {
		t.Fatalf("expected Success (no trip), got kind=%q reason=%+v output=%q", r.Kind, r.Reason, r.Output)
	}
}

// AC1 args variant: error(argsX), error(argsY) -> the second is a FRESH run at
// count 1, so two different-args errors never trip even at N == 2.
func TestTELAC1ArgsChangeStartsFreshRun(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{telBadCall(`{"task_list_id":"X"}`)}, turnUsage()))
	a.Push(NewToolCallRequested([]ToolCall{telBadCall(`{"task_list_id":"Y"}`)}, turnUsage()))
	a.Push(NewFinalResponse("stopped trying", turnUsage()))

	cfg := standardCfg(a)
	cfg.ToolRegistry = telErrRegistry()
	cfg.ErrorLoopThreshold = 2 // 2N == 4; longest identical run is 1.

	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(20)))
	if r.Kind != RunSuccess || r.Output != "stopped trying" {
		t.Fatalf("expected Success (args changed, no trip), got kind=%q reason=%+v", r.Kind, r.Reason)
	}
}

// AC2: exactly N identical-arg recoverable errors inject exactly ONE corrective
// user message carrying the enrichToolError schema+hint; a 4th identical error
// does NOT re-inject.
func TestTELAC2InjectsOneCorrectiveAtN(t *testing.T) {
	a := NewMockAgent("t")
	// 4 identical errors (N=3 injects at the 3rd; 4th must not re-inject), then a
	// final response so the run ends cleanly (2N would be 6).
	telPushBad(a, 4, telBadArgs)
	a.Push(NewFinalResponse("gave up", turnUsage()))

	cfg := standardCfg(a)
	cfg.ToolRegistry = telErrRegistry()
	cfg.ErrorLoopThreshold = 3

	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(20)))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got kind=%q reason=%+v", r.Kind, r.Reason)
	}

	users := telUserMsgs(r.SessionState)
	var correctives []string
	for _, m := range users {
		if strings.Contains(m, "Expected parameter schema") {
			correctives = append(correctives, m)
		}
	}
	if len(correctives) != 1 {
		t.Fatalf("expected exactly one corrective injected, got %d (%q)", len(correctives), users)
	}
	c := correctives[0]
	if !strings.Contains(c, telBadMsg) {
		t.Fatalf("corrective must carry the bare error: %q", c)
	}
	if !strings.Contains(c, `"required"`) {
		t.Fatalf("corrective must carry the parameter schema: %q", c)
	}
	if !strings.Contains(c, "correctly-typed JSON") {
		t.Fatalf("corrective must carry the hint: %q", c)
	}
}

// AC3 Fail: 2N identical errors under behavior = Fail ->
// RunResult.Failure{ToolErrorLoop}; budget NOT fully burned.
func TestTELAC3FailTerminalIsToolErrorLoop(t *testing.T) {
	a := NewMockAgent("t")
	telPushBad(a, 8, telBadArgs) // trip is at 2N == 6.

	cfg := standardCfg(a)
	cfg.ToolRegistry = telErrRegistry()
	cfg.ErrorLoopThreshold = 3

	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(telLeaf(BudgetExhaustedBehavior{Kind: BehaviorFail}, 50)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltToolErrorLoop {
		t.Fatalf("expected Failure{ToolErrorLoop}, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Reason.Tool != "add_task" {
		t.Fatalf("tool = %q, want add_task", r.Reason.Tool)
	}
	if r.Reason.ConsecutiveErrors != 6 {
		t.Fatalf("consecutive_errors = %d, want 6 (2N)", r.Reason.ConsecutiveErrors)
	}
	if r.Turns >= 50 {
		t.Fatalf("budget NOT fully burned expected, got turns = %d", r.Turns)
	}
}

// AC3 Escalate (SurfaceToHuman) -> WaitingForHuman.
func TestTELAC3EscalateSurfacesToHuman(t *testing.T) {
	a := NewMockAgent("t")
	telPushBad(a, 8, telBadArgs)

	cfg := surfaceCfg(a) // EscalationMode SurfaceToHuman
	cfg.ToolRegistry = telErrRegistry()
	cfg.ErrorLoopThreshold = 3

	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(telLeaf(BudgetExhaustedBehavior{Kind: BehaviorEscalate}, 50)))
	if r.Kind != RunWaitingForHuman {
		t.Fatalf("expected WaitingForHuman, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
}

// AC3 Escalate (Autonomous) -> propagated Failure carrying ToolErrorLoop (NOT
// BudgetExceeded).
func TestTELAC3EscalateAutonomousIsToolErrorLoop(t *testing.T) {
	a := NewMockAgent("t")
	telPushBad(a, 8, telBadArgs)

	cfg := standardCfg(a) // EscalationMode Autonomous
	cfg.ToolRegistry = telErrRegistry()
	cfg.ErrorLoopThreshold = 3

	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(telLeaf(BudgetExhaustedBehavior{Kind: BehaviorEscalate}, 50)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltToolErrorLoop {
		t.Fatalf("expected Failure{ToolErrorLoop}, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Reason.Tool != "add_task" {
		t.Fatalf("tool = %q, want add_task", r.Reason.Tool)
	}
	if r.Turns >= 50 {
		t.Fatalf("budget NOT fully burned expected, got turns = %d", r.Turns)
	}
}

// AC3 Continue: one extra window granted, then a terminal. With
// Continue{MaxContinues: 1, OnExhausted: Fail}, the first 2N trip grants a
// continue (fresh window), the second 2N trip falls through to Fail with
// ToolErrorLoop.
func TestTELAC3ContinueGrantsOneWindowThenTerminal(t *testing.T) {
	a := NewMockAgent("t")
	telPushBad(a, 14, telBadArgs) // 2N (6) + 2N (6) = 12 needed.

	cfg := standardCfg(a)
	cfg.ToolRegistry = telErrRegistry()
	cfg.ErrorLoopThreshold = 3

	behavior := BudgetExhaustedBehavior{
		Kind:         BehaviorContinue,
		MaxContinues: 1,
		OnExhausted:  &BudgetExhaustedBehavior{Kind: BehaviorFail},
	}
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(telLeaf(behavior, 50)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltToolErrorLoop {
		t.Fatalf("expected Failure{ToolErrorLoop} after continue, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Turns >= 50 {
		t.Fatalf("budget NOT fully burned expected, got turns = %d", r.Turns)
	}
}

// telLoopObserver records the EmitToolErrorLoop{Detected,Broken} calls (issue
// #137). All other HarnessObserver methods are no-ops.
type telLoopObserver struct {
	detected []uint32
	broken   []uint32
}

func (o *telLoopObserver) EmitTurn(string, SessionID, TaskID, uint32, string, uint64, TokenUsage, float64, StopReason, uint32, string, string, []ToolCall, []Message) {
}
func (o *telLoopObserver) EmitToolCall(string, string, SessionID, TaskID, string, string, string, uint64, uint64, uint64, bool, bool, json.RawMessage, string) {
}
func (o *telLoopObserver) SetSessionOutcome(SessionID, TerminalOutcome, string) {}
func (o *telLoopObserver) FlushSession(context.Context, SessionID)              {}
func (o *telLoopObserver) CostFor(TokenUsage) float64                           { return 0 }
func (o *telLoopObserver) EmitCompaction(string, SessionID, TaskID, string, uint32, uint32, uint32, uint32) {
}
func (o *telLoopObserver) EmitCompactionVerificationFailed(string, SessionID, TaskID, string, []string, bool) {
}
func (o *telLoopObserver) EmitHillClimbingIteration(string, SessionID, TaskID, string, uint32, float64, bool, float64, bool, string, bool) {
}
func (o *telLoopObserver) EmitConsultSpawned(string, SessionID, TaskID, string, string) {}
func (o *telLoopObserver) EmitConsultResumed(string, SessionID, TaskID, string, string, bool) {
}
func (o *telLoopObserver) EmitToolErrorLoopDetected(_ string, _ SessionID, _ TaskID, _ string, tool string, count uint32) {
	if tool == "add_task" {
		o.detected = append(o.detected, count)
	}
}
func (o *telLoopObserver) EmitToolErrorLoopBroken(_ string, _ SessionID, _ TaskID, _ string, tool string, count uint32) {
	if tool == "add_task" {
		o.broken = append(o.broken, count)
	}
}

var _ HarnessObserver = (*telLoopObserver)(nil)

// AC4: a capturing StreamSink + a recording observer -> the detected pair at N
// and the broken pair at 2N, each carrying tool + count.
func TestTELAC4EmitsDetectedAndBrokenEvents(t *testing.T) {
	a := NewMockAgent("t")
	telPushBad(a, 8, telBadArgs)

	obs := &telLoopObserver{}
	cfg := standardCfg(a)
	cfg.ToolRegistry = telErrRegistry()
	cfg.ErrorLoopThreshold = 3
	cfg.Observability = obs

	var streamDetected, streamBroken []uint32
	sink := func(ev HarnessStreamEvent) {
		switch ev.Kind {
		case HarnessStreamToolErrorLoopDetected:
			if ev.Tool == "add_task" {
				streamDetected = append(streamDetected, ev.ConsecutiveErrors)
			}
		case HarnessStreamToolErrorLoopBroken:
			if ev.Tool == "add_task" {
				streamBroken = append(streamBroken, ev.ConsecutiveErrors)
			}
		}
	}
	opts := NewHarnessRunOptions(telLeaf(BudgetExhaustedBehavior{Kind: BehaviorFail}, 50))
	opts.OnStream = sink

	h := NewStandardHarness(cfg)
	_ = h.Run(context.Background(), opts)

	if len(streamDetected) != 1 || streamDetected[0] != 3 {
		t.Fatalf("stream detected = %v, want [3] (one at N)", streamDetected)
	}
	if len(streamBroken) != 1 || streamBroken[0] != 6 {
		t.Fatalf("stream broken = %v, want [6] (one at 2N)", streamBroken)
	}
	if len(obs.detected) != 1 || obs.detected[0] != 3 {
		t.Fatalf("obs detected = %v, want [3] (at N)", obs.detected)
	}
	if len(obs.broken) != 1 || obs.broken[0] != 6 {
		t.Fatalf("obs broken = %v, want [6] (at 2N)", obs.broken)
	}
}

// Breaker disabled when threshold is 0: no trip, run completes.
func TestTELThresholdZeroDisablesBreaker(t *testing.T) {
	a := NewMockAgent("t")
	telPushBad(a, 5, telBadArgs)
	a.Push(NewFinalResponse("fin", turnUsage()))

	cfg := standardCfg(a)
	cfg.ToolRegistry = telErrRegistry()
	cfg.ErrorLoopThreshold = 0

	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(20)))
	if r.Kind != RunSuccess || r.Output != "fin" {
		t.Fatalf("expected Success (breaker off), got kind=%q reason=%+v", r.Kind, r.Reason)
	}
}
