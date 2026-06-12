package sporecore

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
)

// ============================================================================
// #142: Ralph checkpoint lives in the DURABLE project store, not .spore/ files.
// ============================================================================
//
// The agent and the harness must SHARE one store + project namespace: the agent
// "writes" the progress checkpoint to the SAME store, keyed by the SAME project
// namespace, that the harness's Ralph readers consult. ralphStore is the shared
// backend; ralphProjectNS is the fixed namespace both sides key by.

// ralphProjectNS is the fixed durable project namespace the Ralph tests key the
// checkpoint by — a storage.ProjectID.Namespace() value (here a plain string,
// since the test only needs a stable key both sides agree on). The harness reads
// it from cfg.ProjectNamespace; the scripted agents write under it.
const ralphProjectNS = SessionID("ralph-project")

// rootedSandbox is an AllowAllSandbox with a real workspace root. Retained so the
// VCS-history / workspace-root accessors still resolve; the Ralph checkpoint no
// longer lives under it (#142).
type rootedSandbox struct {
	AllowAllSandbox
	root string
}

func (s rootedSandbox) WorkspaceRoot() string { return s.root }

// ralphWindow scripts one context window's filesystem effect: the
// .spore/progress.json the agent "writes" before returning a FinalResponse.
type ralphWindow struct {
	complete  bool
	remaining []string
}

// ralphScriptedAgent simulates the per-window agent: on each Turn (one per
// context window) it pops the next scripted window and writes the corresponding
// progress checkpoint to the SHARED store under ralphProjectNS, then claims done.
// When the queue is empty it leaves the store untouched and claims done (so an
// over-run window is a no-op rather than a panic). Records every Turn's
// concatenated message text so tests can assert fresh-state and reload injection.
type ralphScriptedAgent struct {
	id      AgentID
	store   *fakeRunStore
	mu      sync.Mutex
	windows []ralphWindow
	idx     int
	seen    []string
}

func newRalphAgent(store *fakeRunStore, windows ...ralphWindow) *ralphScriptedAgent {
	return &ralphScriptedAgent{id: AgentID("ralph"), store: store, windows: windows}
}

func (a *ralphScriptedAgent) Turn(_ context.Context, c Context) TurnResult {
	a.mu.Lock()
	defer a.mu.Unlock()

	var b strings.Builder
	for _, m := range c.Messages {
		b.WriteString(m.Content.Text)
		b.WriteString("\n")
	}
	a.seen = append(a.seen, b.String())

	if a.idx < len(a.windows) {
		w := a.windows[a.idx]
		a.idx++
		writeRalphProgress(a.store, w)
	}
	return NewFinalResponse("window done", TokenUsage{InputTokens: 1, OutputTokens: 1})
}

func (a *ralphScriptedAgent) ID() AgentID { return a.id }

func (a *ralphScriptedAgent) callCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.seen)
}

func (a *ralphScriptedAgent) turnTexts() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]string, len(a.seen))
	copy(out, a.seen)
	return out
}

var _ Agent = (*ralphScriptedAgent)(nil)

// ralphFixedAgent always writes the SAME incomplete progress checkpoint (with a
// fixed remaining list) before claiming done — modelling an agent that never
// finishes, so the outer loop resets until MaxResets is exhausted. Records every
// turn's concatenated message text.
type ralphFixedAgent struct {
	id        AgentID
	store     *fakeRunStore
	remaining []string
	mu        sync.Mutex
	seen      []string
}

func (a *ralphFixedAgent) Turn(_ context.Context, c Context) TurnResult {
	a.mu.Lock()
	defer a.mu.Unlock()
	var b strings.Builder
	for _, m := range c.Messages {
		b.WriteString(m.Content.Text)
		b.WriteString("\n")
	}
	a.seen = append(a.seen, b.String())
	writeRalphProgress(a.store, ralphWindow{complete: false, remaining: a.remaining})
	return NewFinalResponse("window done", TokenUsage{InputTokens: 1, OutputTokens: 1})
}

func (a *ralphFixedAgent) ID() AgentID { return a.id }

func (a *ralphFixedAgent) turnTexts() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]string, len(a.seen))
	copy(out, a.seen)
	return out
}

var _ Agent = (*ralphFixedAgent)(nil)

// ralphToolLoopingAgent NEVER returns a FinalResponse — every turn requests a
// tool call, so a bounded ReAct window can only ever exhaust its turn budget (it
// never terminates cleanly). On its FIRST turn it writes a COMPLETE progress
// checkpoint to the shared store, so any code path that DID consult Ralph's
// external completion after the window would (wrongly) see "complete" and Success.
type ralphToolLoopingAgent struct {
	id    AgentID
	store *fakeRunStore
	mu    sync.Mutex
	calls int
}

func newRalphToolLoopingAgent(store *fakeRunStore) *ralphToolLoopingAgent {
	return &ralphToolLoopingAgent{id: AgentID("ralph-tool-loop"), store: store}
}

func (a *ralphToolLoopingAgent) Turn(_ context.Context, _ Context) TurnResult {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.calls == 0 {
		// Mark COMPLETE in the store up front — if Ralph (wrongly) consulted
		// completion after a budget-exhausted window it would Success.
		writeRalphProgress(a.store, ralphWindow{complete: true})
	}
	n := a.calls
	a.calls++
	return NewToolCallRequested([]ToolCall{{
		ID:    fmt.Sprintf("c%d", n),
		Name:  "x",
		Input: json.RawMessage(`{}`),
	}}, TokenUsage{InputTokens: 1, OutputTokens: 1})
}

func (a *ralphToolLoopingAgent) ID() AgentID { return a.id }

func (a *ralphToolLoopingAgent) callCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.calls
}

var _ Agent = (*ralphToolLoopingAgent)(nil)

// ============================================================================
// Helpers
// ============================================================================

// writeRalphProgress writes the progress checkpoint to the shared store under
// ralphProjectNS / RalphProgressKey (#142 — the checkpoint moved off the
// filesystem onto the durable project store).
func writeRalphProgress(store *fakeRunStore, w ralphWindow) {
	var b strings.Builder
	b.WriteString(`{"complete":`)
	if w.complete {
		b.WriteString("true")
	} else {
		b.WriteString("false")
	}
	b.WriteString(`,"remaining":[`)
	for i, r := range w.remaining {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(`"` + r + `"`)
	}
	b.WriteString("]}")
	_ = store.Put(context.Background(), ralphProjectNS, RalphProgressKey, json.RawMessage(b.String()))
}

func writeRalphFeatureList(t *testing.T, store *fakeRunStore, body string) {
	t.Helper()
	if err := store.Put(context.Background(), ralphProjectNS, RalphFeatureListKey, json.RawMessage(body)); err != nil {
		t.Fatal(err)
	}
}

// ralphCfg wires the shared store + the fixed project namespace into the harness
// config so the harness's Ralph readers and the scripted agents' writes key the
// SAME durable namespace (#142 — agent and harness SHARE one store + project id).
func ralphCfg(agent Agent, store *fakeRunStore) HarnessConfig {
	cfg := standardCfg(agent)
	cfg.RunStore = store
	cfg.ProjectNamespace = ralphProjectNS
	// #124 Q3: ralphTask sets Agent: "ralph-agent" as the per-window override, so
	// register the agent under that key (the worker resolves to it per window).
	cfg = cfg.WithRegistryAgent("ralph-agent", agent)
	return cfg
}

func ralphTask() Task {
	return NewTask("build the thing", SessionID("s1"), RalphStrategy(RalphConfig{Inner: PtrStrategy(ReActStrategy(^uint32(0))), Agent: AgentRef("ralph-agent")}))
}

// ============================================================================
// Unit tests
// ============================================================================

// R0: Ralph is implemented — no longer StrategyNotYetImplemented.
func TestRalphIsImplemented(t *testing.T) {
	store := newFakeRunStore()
	a := newRalphAgent(store, ralphWindow{complete: true})
	cfg := ralphCfg(a, store)
	cfg.MaxResets = 3
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	if r.Kind == RunFailure && r.Reason.Kind == HaltStrategyNotYetImplemented {
		t.Fatalf("Ralph must be implemented, got %+v", r)
	}
}

// A first-turn complete progress file => Success in a single agent turn.
func TestRalphCompleteFirstTurnSucceeds(t *testing.T) {
	store := newFakeRunStore()
	a := newRalphAgent(store, ralphWindow{complete: true})
	cfg := ralphCfg(a, store)
	cfg.MaxResets = 3
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if a.callCount() != 1 {
		t.Fatalf("expected exactly 1 agent turn, got %d", a.callCount())
	}
}

// Exit-interception: the registered Stop hook intercepts the agent's exit on an
// incomplete progress file and re-prompts (within the window) instead of
// terminating; once a later turn writes a complete file the run succeeds.
// incomplete, incomplete, complete => Success after exactly 3 turns.
func TestRalphStopHookInterceptsExitUntilComplete(t *testing.T) {
	store := newFakeRunStore()
	a := newRalphAgent(store,
		ralphWindow{complete: false, remaining: []string{"task A"}},
		ralphWindow{complete: false, remaining: []string{"task B"}},
		ralphWindow{complete: true},
	)
	cfg := ralphCfg(a, store)
	cfg.MaxResets = 3
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if a.callCount() != 3 {
		t.Fatalf("expected exactly 3 turns (2 intercepts + complete), got %d", a.callCount())
	}
}

// R5: always-incomplete => exactly MaxResets context-window resets =>
// RalphCompletionUnmet{Iterations: MaxResets}. The agent never writes a complete
// file, so every inner window exhausts its Stop-block budget and the outer loop
// resets until MaxResets windows are spent.
func TestRalphExhaustsMaxResets(t *testing.T) {
	store := newFakeRunStore()
	// Drip-feed only incomplete bodies; once exhausted the agent leaves the store
	// untouched (still incomplete).
	a := newRalphAgent(store, ralphWindow{complete: false, remaining: []string{"task A"}})
	cfg := ralphCfg(a, store)
	cfg.MaxResets = 3
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	if r.Kind != RunFailure || r.Reason.Kind != HaltRalphCompletionUnmet {
		t.Fatalf("expected RalphCompletionUnmet, got %+v", r)
	}
	if r.Reason.Iterations != 3 {
		t.Fatalf("expected exactly MaxResets (3) iterations, got %d", r.Reason.Iterations)
	}
	if !strings.Contains(r.Reason.Reason, "task A") {
		t.Fatalf("last reason should name the remaining task, got %q", r.Reason.Reason)
	}
}

// #125 F5: a Ralph window whose INNER LEAF exhausts its OWN budget mid-window
// (the leaf's PerLoop policy is the binding cap, no smaller global backstop).
// The window surfaces StrategyOutcome.BudgetExhausted; Ralph must treat it as
// "window incomplete → RESET and retry" — it must NOT consult external
// completion (which is COMPLETE on disk here) and must NOT cascade the child's
// exhaustion into its own terminal. So the run reaches MaxResets windows and
// ends RalphCompletionUnmet, NOT Success.
func TestRalphBudgetExhaustedWindowResetsNoCompletionNoCascade(t *testing.T) {
	store := newFakeRunStore()
	// Seed an incomplete progress checkpoint so the window's reload starts incomplete.
	writeRalphProgress(store, ralphWindow{complete: false, remaining: []string{"task A"}})
	agent := newRalphToolLoopingAgent(store)
	cfg := ralphCfg(agent, store)
	// Provide tool outputs for the looping calls across all windows.
	reg := NewScriptedToolRegistry()
	for i := 0; i < 32; i++ {
		reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	}
	cfg.ToolRegistry = reg
	cfg.MaxResets = 3
	h := NewStandardHarness(cfg)

	// Inner leaf carries its OWN binding cap (PerLoop{2}); NO global MaxTurns
	// backstop, so the leaf policy — not the global cap — exhausts the window and
	// the new BudgetExhausted path is taken.
	task := NewTask("build the thing", SessionID("s1"), RalphStrategy(RalphConfig{
		Inner: PtrStrategy(ReActStrategy(2)),
		Agent: AgentRef("ralph-agent"),
	}))

	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltRalphCompletionUnmet {
		t.Fatalf("expected RalphCompletionUnmet (reset-on-exhaust, no completion, no cascade), got %+v", r)
	}
	// Reached MaxResets — completion was NEVER consulted (despite COMPLETE on
	// disk) and the child's exhaustion did NOT cascade into Ralph's own terminal.
	if r.Reason.Iterations != 3 {
		t.Fatalf("expected exactly MaxResets (3) windows, got %d", r.Reason.Iterations)
	}
	// Three windows × two leaf turns each = six agent turns total — proving each
	// exhausted window fully reset and re-ran, not short-circuited.
	if agent.callCount() != 6 {
		t.Fatalf("expected 6 agent turns (3 windows × 2 leaf turns each), got %d", agent.callCount())
	}
}

// R5: a single-window cap (MaxResets=1) => exactly 1 iteration on
// always-incomplete.
func TestRalphSingleWindowCap(t *testing.T) {
	store := newFakeRunStore()
	a := newRalphAgent(store, ralphWindow{complete: false, remaining: []string{"task A"}})
	cfg := ralphCfg(a, store)
	cfg.MaxResets = 1
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	if r.Kind != RunFailure || r.Reason.Kind != HaltRalphCompletionUnmet {
		t.Fatalf("expected RalphCompletionUnmet, got %+v", r)
	}
	if r.Reason.Iterations != 1 {
		t.Fatalf("expected exactly 1 iteration, got %d", r.Reason.Iterations)
	}
}

// R2/R3 (#142): a context-window RESET re-seeds a FRESH SessionState from the
// DURABLE project store — the reloaded progress + feature-list checkpoint content
// lands in the new window's seed, NOT carried-over messages. Forcing
// MaxStopBlocks=1 makes each outer window short (1 intercept), so window 2 is
// reached via a real reset.
func TestRalphFreshStateReloadsCheckpointPerReset(t *testing.T) {
	store := newFakeRunStore()
	writeRalphFeatureList(t, store, `[{"name":"login","passes":false}]`)
	// Seed an initial incomplete progress checkpoint so window 1's reload already
	// sees prior durable state (mirrors the canonical Ralph bootstrap).
	writeRalphProgress(store, ralphWindow{complete: false, remaining: []string{"finish login"}})
	// Always-incomplete: every turn re-writes an incomplete progress checkpoint
	// naming outstanding work, so the outer loop resets between windows.
	a := &ralphFixedAgent{
		id:        AgentID("ralph-fixed"),
		store:     store,
		remaining: []string{"finish login"},
	}
	cfg := ralphCfg(a, store)
	cfg.MaxResets = 2
	cfg.MaxStopBlocks = 1 // each window: 1 turn + 1 intercept, then reset
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	if r.Kind != RunFailure || r.Reason.Kind != HaltRalphCompletionUnmet {
		t.Fatalf("expected RalphCompletionUnmet, got %+v", r)
	}
	if r.Reason.Iterations != 2 {
		t.Fatalf("expected 2 outer iterations, got %d", r.Reason.Iterations)
	}
	// Every window's seed reloaded BOTH filesystem files from disk.
	texts := a.turnTexts()
	if len(texts) == 0 {
		t.Fatal("agent never ran")
	}
	for i, txt := range texts {
		if !strings.Contains(txt, "Reloaded .spore/progress.json") ||
			!strings.Contains(txt, "Reloaded .spore/feature_list.json") {
			t.Fatalf("turn %d seed missing reload block: %q", i, txt)
		}
	}
}

// R7: each context-window reset is traceable — the second (reset) window runs
// under a freshly generated session id, distinct from the task's seed id.
func TestRalphResetUsesFreshSessionID(t *testing.T) {
	store := newFakeRunStore()
	a := &ralphFixedAgent{id: AgentID("ralph-fixed"), store: store, remaining: []string{"task A"}}
	cfg := ralphCfg(a, store)
	cfg.MaxResets = 2
	cfg.MaxStopBlocks = 1
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	// The terminal result is reported under the LAST window's id, which (after a
	// reset) is a generated id, never the seed "s1".
	if r.SessionID == SessionID("s1") {
		t.Fatalf("reset window must run under a fresh generated session id, got seed id %q", r.SessionID)
	}
}

// R6: budgets/usage fold across ALL context windows. With MaxStopBlocks=1 each of
// the 3 windows runs exactly 2 turns => 6 cumulative turns and folded token usage
// 6/6.
func TestRalphBudgetsFoldAcrossWindows(t *testing.T) {
	store := newFakeRunStore()
	a := &ralphFixedAgent{id: AgentID("ralph-fixed"), store: store, remaining: []string{"task A"}}
	cfg := ralphCfg(a, store)
	cfg.MaxResets = 3
	cfg.MaxStopBlocks = 1
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	if r.Turns != 6 {
		t.Fatalf("expected cumulative 6 turns (3 windows x 2 turns), got %d", r.Turns)
	}
	if r.Usage.InputTokens != 6 || r.Usage.OutputTokens != 6 {
		t.Fatalf("expected folded usage 6/6, got %d/%d", r.Usage.InputTokens, r.Usage.OutputTokens)
	}
}

// B3: MaxResets defaults to 3 when zero.
func TestRalphMaxResetsDefaultsToThree(t *testing.T) {
	store := newFakeRunStore()
	a := &ralphFixedAgent{id: AgentID("ralph-fixed"), store: store, remaining: []string{"x"}}
	cfg := ralphCfg(a, store)
	cfg.MaxStopBlocks = 1
	// MaxResets left zero => default 3.
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	if r.Kind != RunFailure || r.Reason.Kind != HaltRalphCompletionUnmet {
		t.Fatalf("expected RalphCompletionUnmet, got %+v", r)
	}
	if r.Reason.Iterations != 3 {
		t.Fatalf("default MaxResets should be 3, got %d", r.Reason.Iterations)
	}
}

// ============================================================================
// Completion-check unit tests
// ============================================================================

// progress missing => incomplete (so the agent learns to write it).
func TestRalphCompletionStatusMissingProgress(t *testing.T) {
	store := newFakeRunStore()
	reason, incomplete := ralphCompletionStatus(context.Background(), store, ralphProjectNS)
	if !incomplete || !strings.Contains(reason, "missing") {
		t.Fatalf("missing progress must be incomplete: (%q, %v)", reason, incomplete)
	}
}

// progress complete + all features pass => complete.
func TestRalphCompletionStatusFeatureListGate(t *testing.T) {
	store := newFakeRunStore()
	writeRalphProgress(store, ralphWindow{complete: true})
	// A failing feature gates completion even when progress says done.
	writeRalphFeatureList(t, store, `[{"name":"login","passes":false}]`)
	if reason, incomplete := ralphCompletionStatus(context.Background(), store, ralphProjectNS); !incomplete ||
		!strings.Contains(reason, "login") {
		t.Fatalf("failing feature must gate completion: (%q, %v)", reason, incomplete)
	}
	// Flip the feature to passing => complete.
	writeRalphFeatureList(t, store, `[{"name":"login","passes":true}]`)
	if reason, incomplete := ralphCompletionStatus(context.Background(), store, ralphProjectNS); incomplete {
		t.Fatalf("all features passing must be complete: (%q, %v)", reason, incomplete)
	}
}

// complete:true but remaining non-empty => incomplete.
func TestRalphCompletionStatusCompleteButRemaining(t *testing.T) {
	store := newFakeRunStore()
	writeRalphProgress(store, ralphWindow{complete: true, remaining: []string{"leftover"}})
	if reason, incomplete := ralphCompletionStatus(context.Background(), store, ralphProjectNS); !incomplete ||
		!strings.Contains(reason, "leftover") {
		t.Fatalf("complete-with-remaining must be incomplete: (%q, %v)", reason, incomplete)
	}
}

// ============================================================================
// Stop-hook inertness (B1)
// ============================================================================

// The registered Ralph Stop hook must NOT interfere with a non-Ralph (plain
// ReAct) run when there is no Ralph progress checkpoint in the store (#142): the
// hook returns Continue and the run terminates in one turn. A store with the
// project namespace pinned but NO progress checkpoint written exercises the
// "present store, absent checkpoint => inert" path.
func TestRalphStopHookInertWithoutProgressFile(t *testing.T) {
	store := newFakeRunStore() // no progress checkpoint written
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := standardCfg(a)
	cfg.RunStore = store
	cfg.ProjectNamespace = ralphProjectNS
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess || r.Output != "done" {
		t.Fatalf("inert Stop hook must allow normal termination, got %+v", r)
	}
	if r.Turns != 1 {
		t.Fatalf("expected exactly 1 turn (no Stop block), got %d", r.Turns)
	}
}
