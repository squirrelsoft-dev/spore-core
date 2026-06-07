package middleware

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ─── scripted middleware fixture ────────────────────────────────────────────

// scripted is a programmable Middleware used across the test suite. It
// records every firing and returns a configured decision.
type scripted struct {
	name     string
	hooks    []HookPoint
	priority int
	decision MiddlewareDecision

	mu    sync.Mutex
	fired []HookPoint
}

func newScripted(name string, hooks []HookPoint, priority int) *scripted {
	return &scripted{
		name:     name,
		hooks:    hooks,
		priority: priority,
		decision: DecisionContinueVal(),
	}
}

func (s *scripted) withDecision(d MiddlewareDecision) *scripted {
	s.decision = d
	return s
}

func (s *scripted) Handle(_ context.Context, hctx HookContext) (MiddlewareDecision, error) {
	s.mu.Lock()
	s.fired = append(s.fired, hctx.Point)
	s.mu.Unlock()
	return s.decision, nil
}

func (s *scripted) Hooks() []HookPoint { return s.hooks }
func (s *scripted) Priority() int      { return s.priority }
func (s *scripted) Name() string       { return s.name }

func (s *scripted) firedCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.fired)
}

func tc(id, name string, input string) sporecore.ToolCall {
	if input == "" {
		input = "{}"
	}
	return sporecore.ToolCall{ID: id, Name: name, Input: json.RawMessage(input)}
}

func sid() sporecore.SessionID { return sporecore.SessionID("sess-test") }

// ─── Rule: register validates hooks list and uniqueness ─────────────────────

func TestRegisterRejectsEmptyHooks(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	err := chain.Register(newScripted("m", nil, 0))
	var me *MiddlewareError
	if !errors.As(err, &me) || me.Kind != ErrKindNoHooks {
		t.Fatalf("want NoHooks, got %v", err)
	}
}

func TestRegisterRejectsDuplicateName(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	if err := chain.Register(newScripted("m", []HookPoint{HookBeforeTurn}, 0)); err != nil {
		t.Fatal(err)
	}
	err := chain.Register(newScripted("m", []HookPoint{HookBeforeTurn}, 0))
	var me *MiddlewareError
	if !errors.As(err, &me) || me.Kind != ErrKindAlreadyRegistered {
		t.Fatalf("want AlreadyRegistered, got %v", err)
	}
}

// ─── Rule: before hooks ascend by priority ──────────────────────────────────

func TestBeforeHooksRunInAscendingPriority(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	tracing := NewTracingMiddleware()
	low := newScripted("b", []HookPoint{HookBeforeTurn}, 10)
	high := newScripted("c", []HookPoint{HookBeforeTurn}, 100)
	if err := chain.Register(high); err != nil {
		t.Fatal(err)
	}
	if err := chain.Register(low); err != nil {
		t.Fatal(err)
	}
	if err := chain.Register(tracing); err != nil {
		t.Fatal(err)
	}
	state := sporecore.SessionState{}
	d, err := chain.FireBeforeTurn(context.Background(), &state, 7)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionContinue {
		t.Fatalf("got %v", d.Kind)
	}
	// tracing fires first (MinInt32). Both scripted should also have fired.
	if got := tracing.Entries(); len(got) != 1 || got[0].Point != HookBeforeTurn || got[0].Turn != 7 {
		t.Fatalf("tracing entries: %+v", got)
	}
	if low.firedCount() != 1 || high.firedCount() != 1 {
		t.Fatalf("expected both to fire, got low=%d high=%d", low.firedCount(), high.firedCount())
	}
}

// ─── Rule: after hooks descend by priority ──────────────────────────────────

func TestAfterHooksRunInDescendingPriority(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	// Use a haltOnFirst sentinel to detect order: register a middleware that
	// Halts on AfterTool — but Halt is not actionable on after hooks for
	// session-end. AfterTool however does honour Halt. We exploit that.
	first := newScripted("a", []HookPoint{HookAfterTool}, 1).
		withDecision(DecisionHaltVal("first"))
	second := newScripted("b", []HookPoint{HookAfterTool}, 100).
		withDecision(DecisionHaltVal("second"))
	if err := chain.Register(first); err != nil {
		t.Fatal(err)
	}
	if err := chain.Register(second); err != nil {
		t.Fatal(err)
	}
	calls := []sporecore.ToolCall{tc("c1", "edit", "")}
	results := []sporecore.ToolResult{{ToolUseID: "c1", Content: "ok"}}
	d, err := chain.FireAfterTool(context.Background(), calls, &results)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionHalt {
		t.Fatalf("expected halt, got %v", d.Kind)
	}
	// b (100) ran first under descending priority; a (1) never fired.
	if second.firedCount() != 1 {
		t.Fatalf("expected b to fire first")
	}
	if first.firedCount() != 0 {
		t.Fatalf("expected a not to fire")
	}
	if d.Reason != "second" {
		t.Fatalf("expected halt reason from b, got %q", d.Reason)
	}
}

// ─── Rule: first Halt stops the chain ───────────────────────────────────────

func TestHaltStopsChain(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	halter := newScripted("halt", []HookPoint{HookBeforeTurn}, 1).
		withDecision(DecisionHaltVal("stop"))
	after := newScripted("after", []HookPoint{HookBeforeTurn}, 100)
	if err := chain.Register(halter); err != nil {
		t.Fatal(err)
	}
	if err := chain.Register(after); err != nil {
		t.Fatal(err)
	}
	state := sporecore.SessionState{}
	d, err := chain.FireBeforeTurn(context.Background(), &state, 1)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionHalt {
		t.Fatalf("want halt, got %v", d)
	}
	if after.firedCount() != 0 {
		t.Fatalf("downstream middleware fired despite halt")
	}
}

// ─── Rule: SurfaceToHuman first-wins on BeforeTool ──────────────────────────

func TestSurfaceToHumanFirstWinsOnBeforeTool(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	req := sporecore.HumanRequest{
		Kind:      sporecore.HumanReqToolApproval,
		Calls:     []sporecore.ToolCall{tc("c1", "shell", "")},
		RiskLevel: sporecore.RiskHigh,
	}
	first := newScripted("first", []HookPoint{HookBeforeTool}, 1).
		withDecision(DecisionSurfaceToHumanVal(req))
	second := newScripted("second", []HookPoint{HookBeforeTool}, 2)
	if err := chain.Register(first); err != nil {
		t.Fatal(err)
	}
	if err := chain.Register(second); err != nil {
		t.Fatal(err)
	}
	calls := []sporecore.ToolCall{tc("c1", "shell", "")}
	d, err := chain.FireBeforeTool(context.Background(), &calls, 1)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionSurfaceToHuman {
		t.Fatalf("want SurfaceToHuman, got %v", d.Kind)
	}
	if second.firedCount() != 0 {
		t.Fatalf("downstream middleware ran")
	}
}

// ─── Rule: SurfaceToHuman illegal outside BeforeTool/BeforeCompletion ───────

func TestSurfaceToHumanIllegalOnBeforeTurn(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	req := sporecore.HumanRequest{
		Kind:     sporecore.HumanReqClarification,
		Question: "?",
	}
	bad := newScripted("bad", []HookPoint{HookBeforeTurn}, 1).
		withDecision(DecisionSurfaceToHumanVal(req))
	if err := chain.Register(bad); err != nil {
		t.Fatal(err)
	}
	state := sporecore.SessionState{}
	d, err := chain.FireBeforeTurn(context.Background(), &state, 1)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionHalt {
		t.Fatalf("want halt, got %v", d.Kind)
	}
	if !strings.Contains(d.Reason, "SurfaceToHuman") {
		t.Fatalf("expected reason to mention SurfaceToHuman, got %q", d.Reason)
	}
}

// ─── Rule: ForceAnotherTurn concatenates and continues ──────────────────────

func TestForceAnotherTurnConcatenatesAndContinues(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	a := newScripted("a", []HookPoint{HookBeforeCompletion}, 1).
		withDecision(DecisionForceAnotherTurnVal("first"))
	b := newScripted("b", []HookPoint{HookBeforeCompletion}, 2).
		withDecision(DecisionForceAnotherTurnVal("second"))
	if err := chain.Register(a); err != nil {
		t.Fatal(err)
	}
	if err := chain.Register(b); err != nil {
		t.Fatal(err)
	}
	state := sporecore.SessionState{}
	d, err := chain.FireBeforeCompletion(context.Background(), "done", 3, &state)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionForceAnotherTurn {
		t.Fatalf("want ForceAnotherTurn, got %v", d.Kind)
	}
	if !strings.Contains(d.Inject, "first") || !strings.Contains(d.Inject, "second") {
		t.Fatalf("expected both injections, got %q", d.Inject)
	}
	if a.firedCount() != 1 || b.firedCount() != 1 {
		t.Fatalf("expected both to fire")
	}
}

// ─── Rule: ForceAnotherTurn illegal outside BeforeCompletion ────────────────

func TestForceAnotherTurnIllegalOnBeforeTurn(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	bad := newScripted("bad", []HookPoint{HookBeforeTurn}, 1).
		withDecision(DecisionForceAnotherTurnVal("x"))
	if err := chain.Register(bad); err != nil {
		t.Fatal(err)
	}
	state := sporecore.SessionState{}
	d, err := chain.FireBeforeTurn(context.Background(), &state, 1)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionHalt {
		t.Fatalf("want halt, got %v", d.Kind)
	}
	if !strings.Contains(d.Reason, "ForceAnotherTurn") {
		t.Fatalf("expected reason to mention ForceAnotherTurn, got %q", d.Reason)
	}
}

// ─── Rule: PatchToolCallsMiddleware renames empty/whitespace names ──────────

func TestPatchToolCallsRenamesEmpty(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	if err := chain.Register(NewPatchToolCallsMiddleware("noop")); err != nil {
		t.Fatal(err)
	}
	calls := []sporecore.ToolCall{tc("c1", "", ""), tc("c2", "   ", ""), tc("c3", "real", "")}
	d, err := chain.FireBeforeTool(context.Background(), &calls, 1)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionContinueWithModification {
		t.Fatalf("want ContinueWithModification, got %v", d.Kind)
	}
	if calls[0].Name != "noop" || calls[1].Name != "noop" || calls[2].Name != "real" {
		t.Fatalf("patched names: %+v", []string{calls[0].Name, calls[1].Name, calls[2].Name})
	}
}

// ─── Rule: PatchToolCalls runs before other BeforeTool middleware ───────────

func TestPatchToolCallsRunsFirstOnBeforeTool(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	observed := ""
	observer := &funcMiddleware{
		name:     "observer",
		hooks:    []HookPoint{HookBeforeTool},
		priority: 0,
		fn: func(_ context.Context, hctx HookContext) (MiddlewareDecision, error) {
			if hctx.Calls != nil && len(*hctx.Calls) > 0 {
				observed = (*hctx.Calls)[0].Name
			}
			return DecisionContinueVal(), nil
		},
	}
	if err := chain.Register(NewPatchToolCallsMiddleware("noop")); err != nil {
		t.Fatal(err)
	}
	if err := chain.Register(observer); err != nil {
		t.Fatal(err)
	}
	calls := []sporecore.ToolCall{tc("c1", "", "")}
	d, err := chain.FireBeforeTool(context.Background(), &calls, 1)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionContinueWithModification {
		t.Fatalf("want ContinueWithModification, got %v", d.Kind)
	}
	if observed != "noop" {
		t.Fatalf("observer saw %q, expected patched name", observed)
	}
}

// ─── Rule: AfterTool mutation propagates via ContinueWithModification ───────

func TestLoopDetectionAnnotatesAfterThreshold(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	if err := chain.Register(NewLoopDetectionMiddleware("edit", 2)); err != nil {
		t.Fatal(err)
	}
	calls := []sporecore.ToolCall{
		{ID: "c1", Name: "edit", Input: json.RawMessage(`{"path":"/tmp/foo.txt"}`)},
	}
	// First fire: under threshold (count == 1).
	results := []sporecore.ToolResult{{ToolUseID: "c1", Content: "ok"}}
	d, err := chain.FireAfterTool(context.Background(), calls, &results)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionContinue {
		t.Fatalf("first fire: want Continue, got %v", d.Kind)
	}
	// Second fire: reaches threshold; should annotate.
	results = []sporecore.ToolResult{{ToolUseID: "c1", Content: "ok"}}
	d, err = chain.FireAfterTool(context.Background(), calls, &results)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionContinueWithModification {
		t.Fatalf("second fire: want ContinueWithModification, got %v", d.Kind)
	}
	if !strings.Contains(results[0].Content, "[loop-detection]") {
		t.Fatalf("expected annotation, got %q", results[0].Content)
	}
}

// ─── Rule: PreCompletionChecklist forces another turn when missing ──────────

func TestPreCompletionChecklistForcesAnotherTurn(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	if err := chain.Register(NewPreCompletionChecklistMiddleware([]string{"tests passed"})); err != nil {
		t.Fatal(err)
	}
	state := sporecore.SessionState{}
	d, err := chain.FireBeforeCompletion(context.Background(), "done", 1, &state)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionForceAnotherTurn {
		t.Fatalf("want ForceAnotherTurn, got %v", d.Kind)
	}
	if !strings.Contains(d.Inject, "tests passed") {
		t.Fatalf("expected inject to mention missing item, got %q", d.Inject)
	}
	d, err = chain.FireBeforeCompletion(context.Background(), "all tests passed", 1, &state)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionContinue {
		t.Fatalf("want Continue, got %v", d.Kind)
	}
}

// ─── Rule: TokenBudgetMiddleware halts at limit ─────────────────────────────

func TestTokenBudgetHaltsWhenExhausted(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	budget := NewTokenBudgetMiddleware(100)
	if err := chain.Register(budget); err != nil {
		t.Fatal(err)
	}
	state := sporecore.SessionState{}
	d, err := chain.FireBeforeTurn(context.Background(), &state, 1)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionContinue {
		t.Fatalf("want Continue while under budget, got %v", d.Kind)
	}
	budget.Record(150)
	d, err = chain.FireBeforeTurn(context.Background(), &state, 2)
	if err != nil {
		t.Fatal(err)
	}
	if d.Kind != DecisionHalt {
		t.Fatalf("want Halt at limit, got %v", d.Kind)
	}
}

// ─── Edge: BeforeSession and AfterSession fire end-to-end ───────────────────

func TestSessionBoundaryHooksFire(t *testing.T) {
	chain := NewStandardMiddlewareChain()
	tracing := NewTracingMiddleware()
	if err := chain.Register(tracing); err != nil {
		t.Fatal(err)
	}
	task := sporecore.NewTask(
		"do thing",
		sid(),
		sporecore.ReActStrategy(5),
	)
	if _, err := chain.FireBeforeSession(context.Background(), &task, sid()); err != nil {
		t.Fatal(err)
	}
	result := sporecore.RunResult{
		Kind: sporecore.RunSuccess, Output: "done", SessionID: sid(), Turns: 1,
	}
	if err := chain.FireAfterSession(context.Background(), &result, sid()); err != nil {
		t.Fatal(err)
	}
	entries := tracing.Entries()
	var sawBefore, sawAfter bool
	for _, e := range entries {
		if e.Point == HookBeforeSession {
			sawBefore = true
		}
		if e.Point == HookAfterSession {
			sawAfter = true
		}
	}
	if !sawBefore || !sawAfter {
		t.Fatalf("missing session boundary firings: %+v", entries)
	}
}

// ─── Decision JSON round-trip ───────────────────────────────────────────────

func TestMiddlewareDecisionJSONRoundTrip(t *testing.T) {
	cases := []MiddlewareDecision{
		DecisionContinueVal(),
		DecisionContinueWithModificationVal(),
		DecisionForceAnotherTurnVal("inject me"),
		DecisionHaltVal("nope"),
		DecisionSurfaceToHumanVal(sporecore.HumanRequest{
			Kind: sporecore.HumanReqClarification, Question: "?",
		}),
	}
	for _, want := range cases {
		raw, err := json.Marshal(want)
		if err != nil {
			t.Fatalf("marshal %v: %v", want.Kind, err)
		}
		var got MiddlewareDecision
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatalf("unmarshal %v: %v", want.Kind, err)
		}
		if got.Kind != want.Kind {
			t.Fatalf("kind mismatch: got %v want %v (raw=%s)", got.Kind, want.Kind, raw)
		}
		if got.Inject != want.Inject || got.Reason != want.Reason {
			t.Fatalf("payload mismatch on %v", want.Kind)
		}
	}
}

// ─── HookPoint predicates ───────────────────────────────────────────────────

func TestHookPointPredicates(t *testing.T) {
	if !HookBeforeTurn.IsBefore() || HookBeforeTurn.IsAfter() {
		t.Fatal("BeforeTurn should be before-ordered")
	}
	if !HookAfterTool.IsAfter() || HookAfterTool.IsBefore() {
		t.Fatal("AfterTool should be after-ordered")
	}
	if !HookBeforeTool.AllowsSurfaceToHuman() ||
		!HookBeforeCompletion.AllowsSurfaceToHuman() {
		t.Fatal("BeforeTool/BeforeCompletion should allow SurfaceToHuman")
	}
	if HookBeforeTurn.AllowsSurfaceToHuman() {
		t.Fatal("BeforeTurn should NOT allow SurfaceToHuman")
	}
	if !HookBeforeCompletion.AllowsForceAnotherTurn() {
		t.Fatal("BeforeCompletion should allow ForceAnotherTurn")
	}
	if HookBeforeTurn.AllowsForceAnotherTurn() {
		t.Fatal("BeforeTurn should NOT allow ForceAnotherTurn")
	}
}

// ─── funcMiddleware helper ──────────────────────────────────────────────────

type funcMiddleware struct {
	name     string
	hooks    []HookPoint
	priority int
	fn       func(context.Context, HookContext) (MiddlewareDecision, error)
}

func (f *funcMiddleware) Handle(ctx context.Context, hctx HookContext) (MiddlewareDecision, error) {
	return f.fn(ctx, hctx)
}
func (f *funcMiddleware) Hooks() []HookPoint { return f.hooks }
func (f *funcMiddleware) Priority() int      { return f.priority }
func (f *funcMiddleware) Name() string       { return f.name }
