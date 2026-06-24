package sporecore_test

// Issue #158 / Phase 3 (Q2): the rich middleware chain is now wired into the
// ReAct loop. These tests exercise the wired *middleware.StandardMiddlewareChain
// (not the ScriptedMiddleware double) end-to-end through the loop: SC-9 (AfterTool
// rewrites a result in place), SC-11 (BeforeTool mutates calls in place +
// priority fan-out), and the BeforeCompletion ForceAnotherTurn injection the
// former harness stub lacked.
//
// They live in an external test package (sporecore_test) because they import the
// middleware package, which imports sporecore — an internal (package sporecore)
// test importing middleware would be an import cycle.

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/middleware"
)

// ── shared helpers ──────────────────────────────────────────────────────────

func mwTurnUsage() sporecore.TokenUsage {
	return sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1}
}

func mwStandardCfg(agent sporecore.Agent) sporecore.HarnessConfig {
	return sporecore.HarnessConfig{
		Agent:             agent,
		ToolRegistry:      sporecore.NewScriptedToolRegistry(),
		Sandbox:           sporecore.AllowAllSandbox{},
		ContextManager:    sporecore.NoopContextManager{},
		TerminationPolicy: sporecore.AlwaysContinuePolicy{},
		EscalationMode:    sporecore.AutonomousEscalation(),
	}
}

func mwReactTask(max uint32) sporecore.Task {
	return sporecore.NewTask("do something", sporecore.SessionID("s1"), sporecore.ReActStrategy(max))
}

// recordingCM stores each tool result as a RoleTool text message AND honours
// ReplaceToolResult — so a SC-9 in-place rewrite is observable in the post-run
// session state. It also appends assistant/user messages (for ForceAnotherTurn).
type recordingCM struct{}

func (recordingCM) render(o sporecore.ToolOutput) string {
	switch o.Kind {
	case sporecore.ToolOutputSuccess:
		return o.Content
	case sporecore.ToolOutputError:
		return o.Message
	}
	return ""
}

func (recordingCM) Assemble(_ context.Context, session *sporecore.SessionState, _ *sporecore.Task, _ sporecore.ContextSources) sporecore.Context {
	return sporecore.Context{Messages: append([]sporecore.Message(nil), session.Messages...)}
}

func (c recordingCM) AppendToolResult(_ context.Context, session *sporecore.SessionState, result *sporecore.HarnessToolResult) {
	session.Messages = append(session.Messages, sporecore.Message{
		Role: sporecore.RoleTool, Content: sporecore.NewTextContent(c.render(result.Output)),
	})
}

func (c recordingCM) ReplaceToolResult(_ context.Context, session *sporecore.SessionState, messageIndex int, result *sporecore.HarnessToolResult) {
	if messageIndex < 0 || messageIndex >= len(session.Messages) {
		return
	}
	session.Messages[messageIndex] = sporecore.Message{
		Role: sporecore.RoleTool, Content: sporecore.NewTextContent(c.render(result.Output)),
	}
}

func (recordingCM) AppendAssistantMessage(_ context.Context, session *sporecore.SessionState, message sporecore.Message) {
	session.Messages = append(session.Messages, message)
}

func (recordingCM) AppendUserMessage(_ context.Context, session *sporecore.SessionState, text string) {
	session.Messages = append(session.Messages, sporecore.Message{
		Role: sporecore.RoleUser, Content: sporecore.NewTextContent(text),
	})
}

func (recordingCM) ShouldCompact(*sporecore.SessionState) bool { return false }

// recordingToolRegistry records every dispatched call so SC-11 can assert what
// was actually dispatched. It embeds *ScriptedToolRegistry for the rest of the
// (large) ToolRegistry surface and overrides Dispatch.
type recordingToolRegistry struct {
	*sporecore.ScriptedToolRegistry
	mu   sync.Mutex
	seen []sporecore.ToolCall
}

func newRecordingToolRegistry() *recordingToolRegistry {
	return &recordingToolRegistry{ScriptedToolRegistry: sporecore.NewScriptedToolRegistry()}
}

func (r *recordingToolRegistry) Dispatch(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) (sporecore.HarnessToolResult, error) {
	r.mu.Lock()
	r.seen = append(r.seen, call)
	r.mu.Unlock()
	return sporecore.HarnessToolResult{
		CallID: call.ID,
		Output: sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "ok"},
	}, nil
}

func (r *recordingToolRegistry) dispatched() []sporecore.ToolCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]sporecore.ToolCall(nil), r.seen...)
}

func toolTexts(session sporecore.SessionState) []string {
	var out []string
	for _, m := range session.Messages {
		if m.Role == sporecore.RoleTool && m.Content.Type == sporecore.ContentTypeText {
			out = append(out, m.Content.Text)
		}
	}
	return out
}

// ── SC-9: AfterTool rewrites a result in place ──────────────────────────────

// rewriteFirstResultMw rewrites the first result's output in place.
type rewriteFirstResultMw struct{ to string }

func (m *rewriteFirstResultMw) Handle(_ context.Context, hctx middleware.HookContext) (middleware.MiddlewareDecision, error) {
	if hctx.Point == middleware.HookAfterTool && hctx.Results != nil && len(*hctx.Results) > 0 {
		(*hctx.Results)[0].Content = m.to
		(*hctx.Results)[0].IsError = true
		return middleware.DecisionContinueWithModificationVal(), nil
	}
	return middleware.DecisionContinueVal(), nil
}
func (m *rewriteFirstResultMw) Hooks() []middleware.HookPoint {
	return []middleware.HookPoint{middleware.HookAfterTool}
}
func (m *rewriteFirstResultMw) Priority() int { return 0 }
func (m *rewriteFirstResultMw) Name() string  { return "rewrite_first_result" }

func TestAfterToolMiddlewareRewritesResultInPlace(t *testing.T) {
	a := sporecore.NewMockAgent("t")
	a.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{
		{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)},
	}, mwTurnUsage()))
	a.Push(sporecore.NewFinalResponse("done", mwTurnUsage()))

	cfg := mwStandardCfg(a)
	cfg.ContextManager = recordingCM{}
	reg := sporecore.NewScriptedToolRegistry()
	reg.Push(sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "ORIGINAL"})
	cfg.ToolRegistry = reg

	chain := middleware.NewStandardMiddlewareChain()
	if err := chain.Register(&rewriteFirstResultMw{to: "REWRITTEN"}); err != nil {
		t.Fatalf("register: %v", err)
	}
	cfg.Middleware = chain

	h := sporecore.NewStandardHarness(cfg)
	r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(mwReactTask(5)))
	if r.Kind != sporecore.RunSuccess {
		t.Fatalf("expected Success, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	texts := toolTexts(r.SessionState)
	var sawRewritten, sawOriginal bool
	for _, txt := range texts {
		if txt == "REWRITTEN" {
			sawRewritten = true
		}
		if txt == "ORIGINAL" {
			sawOriginal = true
		}
	}
	if !sawRewritten {
		t.Fatalf("rewritten result must reach the conversation, got %v", texts)
	}
	if sawOriginal {
		t.Fatalf("original result must not survive the rewrite, got %v", texts)
	}
}

// ── SC-11: BeforeTool mutates calls in place, in priority order ─────────────

// traceTagMw appends tag to calls[0].input["trace"] in place, at the given
// priority.
type traceTagMw struct {
	tag  string
	prio int
}

func (m *traceTagMw) Handle(_ context.Context, hctx middleware.HookContext) (middleware.MiddlewareDecision, error) {
	if hctx.Point == middleware.HookBeforeTool && hctx.Calls != nil && len(*hctx.Calls) > 0 {
		first := &(*hctx.Calls)[0]
		var obj map[string]json.RawMessage
		if len(first.Input) > 0 {
			_ = json.Unmarshal(first.Input, &obj)
		}
		if obj == nil {
			obj = map[string]json.RawMessage{}
		}
		var trace []string
		if raw, ok := obj["trace"]; ok {
			_ = json.Unmarshal(raw, &trace)
		}
		trace = append(trace, m.tag)
		traceRaw, _ := json.Marshal(trace)
		obj["trace"] = traceRaw
		newInput, _ := json.Marshal(obj)
		first.Input = newInput
		return middleware.DecisionContinueWithModificationVal(), nil
	}
	return middleware.DecisionContinueVal(), nil
}
func (m *traceTagMw) Hooks() []middleware.HookPoint {
	return []middleware.HookPoint{middleware.HookBeforeTool}
}
func (m *traceTagMw) Priority() int { return m.prio }
func (m *traceTagMw) Name() string {
	// Distinct names so the chain accepts both registrations.
	if m.prio <= 1 {
		return "trace_first"
	}
	return "trace_second"
}

func TestBeforeToolMiddlewareMutatesCallsInPriorityOrder(t *testing.T) {
	a := sporecore.NewMockAgent("t")
	a.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{
		{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)},
	}, mwTurnUsage()))
	a.Push(sporecore.NewFinalResponse("done", mwTurnUsage()))

	reg := newRecordingToolRegistry()
	cfg := mwStandardCfg(a)
	cfg.ToolRegistry = reg

	chain := middleware.NewStandardMiddlewareChain()
	// Register out of order to prove the chain sorts by priority, not insertion.
	if err := chain.Register(&traceTagMw{tag: "second", prio: 2}); err != nil {
		t.Fatalf("register second: %v", err)
	}
	if err := chain.Register(&traceTagMw{tag: "first", prio: 1}); err != nil {
		t.Fatalf("register first: %v", err)
	}
	cfg.Middleware = chain

	h := sporecore.NewStandardHarness(cfg)
	r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(mwReactTask(5)))
	if r.Kind != sporecore.RunSuccess {
		t.Fatalf("expected Success, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	disp := reg.dispatched()
	if len(disp) != 1 {
		t.Fatalf("expected exactly one tool dispatched, got %d", len(disp))
	}
	var got struct {
		Trace []string `json:"trace"`
	}
	if err := json.Unmarshal(disp[0].Input, &got); err != nil {
		t.Fatalf("unmarshal dispatched input %q: %v", disp[0].Input, err)
	}
	if len(got.Trace) != 2 || got.Trace[0] != "first" || got.Trace[1] != "second" {
		t.Fatalf("BeforeTool middleware must mutate the dispatched call in priority order, got %v", got.Trace)
	}
}

// ── BeforeCompletion ForceAnotherTurn ───────────────────────────────────────

// forceOnceMw forces ONE extra turn at BeforeCompletion, then lets the run
// complete.
type forceOnceMw struct {
	mu     sync.Mutex
	fired  bool
	inject string
}

func (m *forceOnceMw) Handle(_ context.Context, hctx middleware.HookContext) (middleware.MiddlewareDecision, error) {
	if hctx.Point != middleware.HookBeforeCompletion {
		return middleware.DecisionContinueVal(), nil
	}
	m.mu.Lock()
	first := !m.fired
	m.fired = true
	m.mu.Unlock()
	if first {
		return middleware.DecisionForceAnotherTurnVal(m.inject), nil
	}
	return middleware.DecisionContinueVal(), nil
}
func (m *forceOnceMw) Hooks() []middleware.HookPoint {
	return []middleware.HookPoint{middleware.HookBeforeCompletion}
}
func (m *forceOnceMw) Priority() int { return 0 }
func (m *forceOnceMw) Name() string  { return "force_once" }

func TestBeforeCompletionForceAnotherTurnRunsExtraTurn(t *testing.T) {
	a := sporecore.NewMockAgent("t")
	a.Push(sporecore.NewFinalResponse("first", mwTurnUsage()))
	a.Push(sporecore.NewFinalResponse("second", mwTurnUsage()))

	cfg := mwStandardCfg(a)
	cfg.ContextManager = recordingCM{}
	chain := middleware.NewStandardMiddlewareChain()
	if err := chain.Register(&forceOnceMw{inject: "keep going"}); err != nil {
		t.Fatalf("register: %v", err)
	}
	cfg.Middleware = chain

	h := sporecore.NewStandardHarness(cfg)
	r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(mwReactTask(5)))
	if r.Kind != sporecore.RunSuccess {
		t.Fatalf("expected Success, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Output != "second" {
		t.Fatalf("the forced second turn must win, got output=%q", r.Output)
	}
	if r.Turns != 2 {
		t.Fatalf("exactly one extra turn was forced, got turns=%d", r.Turns)
	}
	var injected bool
	for _, msg := range r.SessionState.Messages {
		if msg.Role == sporecore.RoleUser && msg.Content.Type == sporecore.ContentTypeText && msg.Content.Text == "keep going" {
			injected = true
		}
	}
	if !injected {
		t.Fatal("the ForceAnotherTurn injection must be recorded as a user message")
	}
}
