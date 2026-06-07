package sporecore

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// ============================================================================
// Test doubles
// ============================================================================

// rootedSandbox is an AllowAllSandbox with a real workspace root, so the Ralph
// strategy reads .spore/ files from a tempdir under test control.
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
// .spore/progress.json under root, then claims done. When the queue is empty it
// leaves the filesystem untouched and claims done (so an over-run window is a
// no-op rather than a panic). Records every Turn's concatenated message text so
// tests can assert fresh-state and reload injection.
type ralphScriptedAgent struct {
	id      AgentID
	root    string
	mu      sync.Mutex
	windows []ralphWindow
	idx     int
	seen    []string
}

func newRalphAgent(root string, windows ...ralphWindow) *ralphScriptedAgent {
	return &ralphScriptedAgent{id: AgentID("ralph"), root: root, windows: windows}
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
		writeRalphProgress(a.root, w)
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

// ralphFixedAgent always writes the SAME incomplete progress file (with a fixed
// remaining list) before claiming done — modelling an agent that never finishes,
// so the outer loop resets until MaxResets is exhausted. Records every turn's
// concatenated message text.
type ralphFixedAgent struct {
	id        AgentID
	root      string
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
	writeRalphProgress(a.root, ralphWindow{complete: false, remaining: a.remaining})
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

// ============================================================================
// Helpers
// ============================================================================

func writeRalphProgress(root string, w ralphWindow) {
	dir := filepath.Join(root, ".spore")
	_ = os.MkdirAll(dir, 0o755)
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
	_ = os.WriteFile(filepath.Join(dir, "progress.json"), []byte(b.String()), 0o644)
}

func writeRalphFeatureList(t *testing.T, root, body string) {
	t.Helper()
	dir := filepath.Join(root, ".spore")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "feature_list.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func ralphCfg(agent Agent, root string) HarnessConfig {
	cfg := standardCfg(agent)
	cfg.Sandbox = rootedSandbox{root: root}
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
	dir := t.TempDir()
	a := newRalphAgent(dir, ralphWindow{complete: true})
	cfg := ralphCfg(a, dir)
	cfg.MaxResets = 3
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	if r.Kind == RunFailure && r.Reason.Kind == HaltStrategyNotYetImplemented {
		t.Fatalf("Ralph must be implemented, got %+v", r)
	}
}

// A first-turn complete progress file => Success in a single agent turn.
func TestRalphCompleteFirstTurnSucceeds(t *testing.T) {
	dir := t.TempDir()
	a := newRalphAgent(dir, ralphWindow{complete: true})
	cfg := ralphCfg(a, dir)
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
	dir := t.TempDir()
	a := newRalphAgent(dir,
		ralphWindow{complete: false, remaining: []string{"task A"}},
		ralphWindow{complete: false, remaining: []string{"task B"}},
		ralphWindow{complete: true},
	)
	cfg := ralphCfg(a, dir)
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
	dir := t.TempDir()
	// Drip-feed only incomplete bodies; once exhausted the agent leaves the file
	// untouched (still incomplete).
	a := newRalphAgent(dir, ralphWindow{complete: false, remaining: []string{"task A"}})
	cfg := ralphCfg(a, dir)
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

// R5: a single-window cap (MaxResets=1) => exactly 1 iteration on
// always-incomplete.
func TestRalphSingleWindowCap(t *testing.T) {
	dir := t.TempDir()
	a := newRalphAgent(dir, ralphWindow{complete: false, remaining: []string{"task A"}})
	cfg := ralphCfg(a, dir)
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

// R2/R3: a context-window RESET re-seeds a FRESH SessionState from the filesystem
// — the reloaded .spore/progress.json + .spore/feature_list.json content lands in
// the new window's seed, NOT carried-over messages. Forcing MaxStopBlocks=1 makes
// each outer window short (1 intercept), so window 2 is reached via a real reset.
func TestRalphFreshStateReloadsFilesystemPerReset(t *testing.T) {
	dir := t.TempDir()
	writeRalphFeatureList(t, dir, `[{"name":"login","passes":false}]`)
	// Seed an initial incomplete progress file so window 1's reload already sees
	// prior filesystem state (mirrors the canonical Ralph bootstrap).
	writeRalphProgress(dir, ralphWindow{complete: false, remaining: []string{"finish login"}})
	// Always-incomplete: every turn re-writes an incomplete progress file naming
	// outstanding work, so the outer loop resets between windows.
	a := &ralphFixedAgent{
		id:        AgentID("ralph-fixed"),
		root:      dir,
		remaining: []string{"finish login"},
	}
	cfg := ralphCfg(a, dir)
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
	dir := t.TempDir()
	a := &ralphFixedAgent{id: AgentID("ralph-fixed"), root: dir, remaining: []string{"task A"}}
	cfg := ralphCfg(a, dir)
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
	dir := t.TempDir()
	a := &ralphFixedAgent{id: AgentID("ralph-fixed"), root: dir, remaining: []string{"task A"}}
	cfg := ralphCfg(a, dir)
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
	dir := t.TempDir()
	a := &ralphFixedAgent{id: AgentID("ralph-fixed"), root: dir, remaining: []string{"x"}}
	cfg := ralphCfg(a, dir)
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

// progress.json missing => incomplete (so the agent learns to write it).
func TestRalphCompletionStatusMissingProgress(t *testing.T) {
	dir := t.TempDir()
	reason, incomplete := ralphCompletionStatus(dir)
	if !incomplete || !strings.Contains(reason, "missing") {
		t.Fatalf("missing progress must be incomplete: (%q, %v)", reason, incomplete)
	}
}

// progress complete + all features pass => complete.
func TestRalphCompletionStatusFeatureListGate(t *testing.T) {
	dir := t.TempDir()
	writeRalphProgress(dir, ralphWindow{complete: true})
	// A failing feature gates completion even when progress says done.
	writeRalphFeatureList(t, dir, `[{"name":"login","passes":false}]`)
	if reason, incomplete := ralphCompletionStatus(dir); !incomplete ||
		!strings.Contains(reason, "login") {
		t.Fatalf("failing feature must gate completion: (%q, %v)", reason, incomplete)
	}
	// Flip the feature to passing => complete.
	writeRalphFeatureList(t, dir, `[{"name":"login","passes":true}]`)
	if reason, incomplete := ralphCompletionStatus(dir); incomplete {
		t.Fatalf("all features passing must be complete: (%q, %v)", reason, incomplete)
	}
}

// complete:true but remaining non-empty => incomplete.
func TestRalphCompletionStatusCompleteButRemaining(t *testing.T) {
	dir := t.TempDir()
	writeRalphProgress(dir, ralphWindow{complete: true, remaining: []string{"leftover"}})
	if reason, incomplete := ralphCompletionStatus(dir); !incomplete ||
		!strings.Contains(reason, "leftover") {
		t.Fatalf("complete-with-remaining must be incomplete: (%q, %v)", reason, incomplete)
	}
}

// ============================================================================
// Stop-hook inertness (B1)
// ============================================================================

// The registered Ralph Stop hook must NOT interfere with a non-Ralph (plain
// ReAct) run when there is no .spore/progress.json: the hook returns Continue and
// the run terminates in one turn.
func TestRalphStopHookInertWithoutProgressFile(t *testing.T) {
	dir := t.TempDir() // no .spore/progress.json
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := standardCfg(a)
	cfg.Sandbox = rootedSandbox{root: dir}
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess || r.Output != "done" {
		t.Fatalf("inert Stop hook must allow normal termination, got %+v", r)
	}
	if r.Turns != 1 {
		t.Fatalf("expected exactly 1 turn (no Stop block), got %d", r.Turns)
	}
}
