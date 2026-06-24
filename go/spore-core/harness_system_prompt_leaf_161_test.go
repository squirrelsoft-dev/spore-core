package sporecore

import (
	"context"
	"strings"
	"sync"
	"testing"
)

// ============================================================================
// SC-10 (#161): per-leaf system_prompt override in ReactConfig. A PlanExecute
// whose plan and execute leaves carry DISTINCT per-leaf system prompts runs
// each phase under ONLY its own prompt — neither leaks into the other. Mirrors
// the Rust acceptance tests:
//   - plan_and_execute_leaves_see_only_their_own_system_prompt
//   - leaf_system_prompt_overrides_global_and_falls_back
// ============================================================================

// recordingTurnAgent records the full assembled-context text of EVERY turn it is
// handed (the Go analogue of the Rust RecordingTurnAgent.seen_text()), then
// replies with a scripted FinalResponse per turn. The first scripted reply is
// the plan turn's plan JSON; the second is the execute turn's result.
type recordingTurnAgent struct {
	id      AgentID
	mu      sync.Mutex
	replies []string
	calls   int
	seen    []string // joined text of each turn's context, in turn order.
}

func (a *recordingTurnAgent) Turn(_ context.Context, c Context) TurnResult {
	a.mu.Lock()
	defer a.mu.Unlock()
	var b strings.Builder
	for _, m := range c.Messages {
		if m.Content.Type == ContentTypeText {
			b.WriteString(m.Content.Text)
			b.WriteString("\n")
		}
	}
	a.seen = append(a.seen, b.String())
	idx := a.calls
	a.calls++
	if idx >= len(a.replies) {
		idx = len(a.replies) - 1
	}
	return NewFinalResponse(a.replies[idx], TokenUsage{InputTokens: 1, OutputTokens: 1})
}

func (a *recordingTurnAgent) ID() AgentID { return a.id }

func (a *recordingTurnAgent) seenText() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]string(nil), a.seen...)
}

// planExecuteLeafPromptTask builds a PlanExecute whose plan and execute leaves
// carry the supplied per-leaf system prompts (nil ⇒ no override on that leaf).
// The plan leaf declares the structured plan-schema output contract (the
// structured plan slot, A.5), exactly as planTask does; everything else is the
// bare default leaf.
func planExecuteLeafPromptTask(instruction string, planSys, execSys *string) Task {
	plan := ReActStrategy(^uint32(0))
	plan.ReActCfg.Output = func() *SchemaRef { s := SchemaRef(""); return &s }()
	plan.ReActCfg.SystemPrompt = planSys
	exec := ReActStrategy(^uint32(0))
	exec.ReActCfg.SystemPrompt = execSys
	return NewTask(instruction, SessionID("leaf-prompt-sess"), PlanExecuteStrategy(PlanExecuteConfig{
		Plan:     &plan,
		Execute:  &exec,
		Behavior: BudgetExhaustedBehavior{Kind: BehaviorEscalate},
	}))
}

func strptr(s string) *string { return &s }

// SC-10: a PlanExecute whose plan and execute leaves carry DISTINCT system
// prompts runs each phase under ONLY its own prompt. The global SystemPrompt is
// unset here, so the ONLY system text a turn can see is what its own leaf
// supplied; any cross-phase appearance is an unambiguous leak.
func TestPlanAndExecuteLeavesSeeOnlyTheirOwnSystemPrompt(t *testing.T) {
	const planSys = "PLAN_SYSTEM_PROMPT_MARKER"
	const execSys = "EXECUTE_SYSTEM_PROMPT_MARKER"

	agent := &recordingTurnAgent{
		id: AgentID("rec"),
		replies: []string{
			// Plan turn: a single-step plan.
			`{"tasks":["only step"],"rationale":"r"}`,
			// Execute step: finalize directly.
			"did the step",
		},
	}
	cfg := standardCfg(agent)
	// No global prompt ⇒ the ONLY system text a turn can see is its leaf's.
	if cfg.SystemPrompt != "" {
		t.Fatalf("precondition: global SystemPrompt must be unset, got %q", cfg.SystemPrompt)
	}
	h := NewStandardHarness(cfg)

	task := planExecuteLeafPromptTask("decompose and execute", strptr(planSys), strptr(execSys))
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if r.Output != "did the step" {
		t.Fatalf("output = %q, want %q", r.Output, "did the step")
	}

	contexts := agent.seenText()
	if len(contexts) != 2 {
		t.Fatalf("expected 2 turns (one plan + one execute), got %d", len(contexts))
	}

	// Plan turn (index 0): sees ONLY the plan leaf's prompt.
	if !strings.Contains(contexts[0], planSys) {
		t.Fatalf("plan turn must carry the plan leaf's system prompt: %s", contexts[0])
	}
	if strings.Contains(contexts[0], execSys) {
		t.Fatalf("plan turn must NOT see the execute leaf's system prompt: %s", contexts[0])
	}

	// Execute turn (index 1): sees ONLY the execute leaf's prompt.
	if !strings.Contains(contexts[1], execSys) {
		t.Fatalf("execute turn must carry the execute leaf's system prompt: %s", contexts[1])
	}
	if strings.Contains(contexts[1], planSys) {
		t.Fatalf("execute turn must NOT see the plan leaf's system prompt: %s", contexts[1])
	}
}

// SC-10: the per-leaf system_prompt override WINS over the global
// config.SystemPrompt — a leaf that supplies one sees ONLY its own (the global
// does not leak in), while a leaf WITHOUT an override falls back to the global
// prompt (byte-identical to pre-SC-10).
func TestLeafSystemPromptOverridesGlobalAndFallsBack(t *testing.T) {
	const globalSys = "GLOBAL_SYSTEM_PROMPT_MARKER"
	const planSys = "PLAN_ONLY_SYSTEM_PROMPT_MARKER"

	agent := &recordingTurnAgent{
		id: AgentID("rec"),
		replies: []string{
			`{"tasks":["only step"],"rationale":"r"}`,
			"did the step",
		},
	}
	cfg := standardCfg(agent)
	// A global prompt IS configured this time.
	cfg.SystemPrompt = globalSys
	h := NewStandardHarness(cfg)

	// Plan leaf overrides the global prompt; execute leaf has no override ⇒ it
	// falls back to the global prompt.
	task := planExecuteLeafPromptTask("decompose and execute", strptr(planSys), nil)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}

	contexts := agent.seenText()
	if len(contexts) != 2 {
		t.Fatalf("expected 2 turns (one plan + one execute), got %d", len(contexts))
	}

	// Plan turn: its override WINS — only the plan prompt, NOT the global one.
	if !strings.Contains(contexts[0], planSys) || strings.Contains(contexts[0], globalSys) {
		t.Fatalf("plan turn must see ONLY its override, not the global prompt: %s", contexts[0])
	}
	// Execute turn: no override ⇒ the global prompt applies (back-compat).
	if !strings.Contains(contexts[1], globalSys) || strings.Contains(contexts[1], planSys) {
		t.Fatalf("execute turn must fall back to the global prompt: %s", contexts[1])
	}
}
