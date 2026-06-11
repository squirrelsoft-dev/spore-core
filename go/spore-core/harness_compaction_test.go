package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// ============================================================================
// Test doubles for the compaction verify→retry→warn loop (issue #46).
// ============================================================================

// compactingCM is a ContextManager that also implements
// CompactingContextManager. It always offers a compaction turn (gated by
// ShouldCompact), records how many times ApplyCompaction ran, and captures the
// contexts the agent saw across retries so injected missing-items messages can
// be asserted.
type compactingCM struct {
	NoopContextManager
	shouldCompact bool
	applyCount    int
	// preparedContext seeds the first compaction turn's Context.
	preparedContext Context
	// injectedTurns records each Context the agent was handed, in order.
	injectedTurns []Context
}

func (c *compactingCM) ShouldCompact(*SessionState) bool { return c.shouldCompact }

func (c *compactingCM) PrepareCompactionTurn(*SessionState) (*CompactionTurn, bool) {
	turn := &CompactionTurn{
		Context:           c.preparedContext,
		PreserveHints:     nil,
		VerificationState: nil,
		MessagesRemoved:   3,
	}
	return turn, true
}

func (c *compactingCM) InjectMissingItems(ctx *Context, missing []string) {
	ctx.Messages = append(ctx.Messages, Message{
		Role:    RoleUser,
		Content: NewTextContent(injectMissingItemsText(missing)),
	})
}

func (c *compactingCM) ApplyCompaction(_ *SessionState, _ string) {
	c.applyCount++
}

// injectMissingItemsText mirrors the spec/fixture message format.
func injectMissingItemsText(missing []string) string {
	out := "Your summary is missing these items: "
	for i, m := range missing {
		if i > 0 {
			out += ", "
		}
		out += m
	}
	out += ". Please revise."
	return out
}

// stubVerifier returns scripted verdicts, one per Verify call; the last verdict
// repeats if called more times than scripted. It also records each Context it
// was asked to verify against (via the turn) so retry injection can be
// inspected by the caller through the manager instead.
type stubVerifier struct {
	verdicts []CompactionVerificationResult
	calls    int
}

func (v *stubVerifier) Verify(_ string, _ *CompactionTurn) CompactionVerificationResult {
	idx := v.calls
	v.calls++
	if idx >= len(v.verdicts) {
		idx = len(v.verdicts) - 1
	}
	return v.verdicts[idx]
}

// recordingAgent records each Context it is handed and returns a FinalResponse.
type recordingAgent struct {
	cm      *compactingCM
	summary string
	turns   int
}

func (a *recordingAgent) Turn(_ context.Context, c Context) TurnResult {
	a.turns++
	a.cm.injectedTurns = append(a.cm.injectedTurns, c)
	return NewFinalResponse(a.summary, TokenUsage{InputTokens: 1, OutputTokens: 1})
}

func (a *recordingAgent) ID() AgentID { return "compacting-agent" }

func compactionCfg(agent Agent, cm ContextManager, v CompactionVerifier, maxAttempts uint32) HarnessConfig {
	return HarnessConfig{
		Agent:                 agent,
		ToolRegistry:          NewScriptedToolRegistry(),
		Sandbox:               AllowAllSandbox{},
		ContextManager:        cm,
		TerminationPolicy:     AlwaysContinuePolicy{},
		CompactionVerifier:    v,
		MaxCompactionAttempts: maxAttempts,
	}
}

// runOneCompaction drives runCompaction directly with a fresh span sequence and
// usage accumulator, returning the usage so token folding can be asserted.
func runOneCompaction(h *StandardHarness, cm ContextManager) AggregateUsage {
	var span uint64
	var usage AggregateUsage
	session := SessionState{}
	if cm.ShouldCompact(&session) {
		h.runCompaction(context.Background(), &session, "sess-c", "task-c", &span, &usage, h.config.Agent)
	}
	return usage
}

func pass() CompactionVerificationResult {
	return CompactionVerificationResult{Passed: true, MissingItems: []string{}}
}
func fail(items ...string) CompactionVerificationResult {
	return CompactionVerificationResult{Passed: false, MissingItems: items}
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

// shouldCompact false → no compaction turn at all.
func TestCompactionShouldCompactFalseNoTurn(t *testing.T) {
	cm := &compactingCM{shouldCompact: false}
	agent := &recordingAgent{cm: cm, summary: "summary"}
	v := &stubVerifier{verdicts: []CompactionVerificationResult{pass()}}
	h := NewStandardHarness(compactionCfg(agent, cm, v, 2))
	runOneCompaction(h, cm)
	if agent.turns != 0 {
		t.Fatalf("expected 0 compaction turns, got %d", agent.turns)
	}
	if cm.applyCount != 0 {
		t.Fatalf("expected 0 apply calls, got %d", cm.applyCount)
	}
}

// passing verifier → 1 turn, 1 apply, no warn.
func TestCompactionPassingVerifier(t *testing.T) {
	cm := &compactingCM{shouldCompact: true}
	agent := &recordingAgent{cm: cm, summary: "good summary"}
	v := &stubVerifier{verdicts: []CompactionVerificationResult{pass()}}
	obs := newCapturingObserver()
	cfg := compactionCfg(agent, cm, v, 2)
	cfg.Observability = obs
	h := NewStandardHarness(cfg)
	runOneCompaction(h, cm)

	if agent.turns != 1 {
		t.Fatalf("turns = %d, want 1", agent.turns)
	}
	if cm.applyCount != 1 {
		t.Fatalf("apply = %d, want 1", cm.applyCount)
	}
	if obs.warns != 0 {
		t.Fatalf("warns = %d, want 0", obs.warns)
	}
	if obs.compactions != 1 {
		t.Fatalf("compaction spans = %d, want 1", obs.compactions)
	}
}

// failing-then-passing, max=2 → 2 turns; retry context contains the injected
// missing-items message with the actual terms.
func TestCompactionFailThenPass(t *testing.T) {
	cm := &compactingCM{shouldCompact: true}
	agent := &recordingAgent{cm: cm, summary: "summary"}
	v := &stubVerifier{verdicts: []CompactionVerificationResult{
		fail("payment", "deploy"),
		pass(),
	}}
	h := NewStandardHarness(compactionCfg(agent, cm, v, 2))
	runOneCompaction(h, cm)

	if agent.turns != 2 {
		t.Fatalf("turns = %d, want 2", agent.turns)
	}
	if cm.applyCount != 1 {
		t.Fatalf("apply = %d, want 1", cm.applyCount)
	}
	// The second turn's context must carry the injected message.
	if len(cm.injectedTurns) != 2 {
		t.Fatalf("recorded %d turns, want 2", len(cm.injectedTurns))
	}
	retry := cm.injectedTurns[1]
	want := injectMissingItemsText([]string{"payment", "deploy"})
	found := false
	for _, m := range retry.Messages {
		if m.Content.Text == want {
			found = true
		}
	}
	if !found {
		t.Fatalf("retry context missing injected message %q; got %#v", want, retry.Messages)
	}
}

// always-failing, max=2 → warn with MissingItems + AcceptedAnyway=true, apply
// still called, run proceeds, failure metric == 1.
func TestCompactionAlwaysFailsWarnsAndAccepts(t *testing.T) {
	cm := &compactingCM{shouldCompact: true}
	agent := &recordingAgent{cm: cm, summary: "summary"}
	v := &stubVerifier{verdicts: []CompactionVerificationResult{fail("payment")}}
	obs := newCapturingObserver()
	cfg := compactionCfg(agent, cm, v, 2)
	cfg.Observability = obs
	h := NewStandardHarness(cfg)
	runOneCompaction(h, cm)

	if agent.turns != 2 {
		t.Fatalf("turns = %d, want 2", agent.turns)
	}
	if cm.applyCount != 1 {
		t.Fatalf("apply = %d, want 1", cm.applyCount)
	}
	if obs.warns != 1 {
		t.Fatalf("warns = %d, want 1", obs.warns)
	}
	if !reflect.DeepEqual(obs.lastWarnMissing, []string{"payment"}) {
		t.Fatalf("warn missing = %#v, want [payment]", obs.lastWarnMissing)
	}
	if !obs.lastWarnAccepted {
		t.Fatalf("warn accepted_anyway = false, want true")
	}
}

// MaxCompactionAttempts=1 honored → 1 attempt then warn+accept (no retry).
func TestCompactionMaxAttemptsOne(t *testing.T) {
	cm := &compactingCM{shouldCompact: true}
	agent := &recordingAgent{cm: cm, summary: "summary"}
	v := &stubVerifier{verdicts: []CompactionVerificationResult{fail("payment")}}
	obs := newCapturingObserver()
	cfg := compactionCfg(agent, cm, v, 1)
	cfg.Observability = obs
	h := NewStandardHarness(cfg)
	runOneCompaction(h, cm)

	if agent.turns != 1 {
		t.Fatalf("turns = %d, want 1", agent.turns)
	}
	if cm.applyCount != 1 {
		t.Fatalf("apply = %d, want 1", cm.applyCount)
	}
	if obs.warns != 1 {
		t.Fatalf("warns = %d, want 1", obs.warns)
	}
}

// MaxCompactionAttempts=0 is clamped to 1 → 1 attempt then warn+accept.
func TestCompactionMaxAttemptsZeroClamped(t *testing.T) {
	cm := &compactingCM{shouldCompact: true}
	agent := &recordingAgent{cm: cm, summary: "summary"}
	v := &stubVerifier{verdicts: []CompactionVerificationResult{fail("payment")}}
	obs := newCapturingObserver()
	cfg := compactionCfg(agent, cm, v, 0)
	cfg.Observability = obs
	h := NewStandardHarness(cfg)
	runOneCompaction(h, cm)

	if agent.turns != 1 {
		t.Fatalf("turns = %d, want 1 (clamped)", agent.turns)
	}
	if obs.warns != 1 {
		t.Fatalf("warns = %d, want 1", obs.warns)
	}
}

// A non-compacting ContextManager (does not implement CompactingContextManager)
// is skipped even when ShouldCompact returns true.
func TestCompactionSkippedWhenNotCompactingManager(t *testing.T) {
	cm := alwaysCompactNoopCM{}
	agent := &recordingAgent{cm: &compactingCM{}, summary: "summary"}
	v := &stubVerifier{verdicts: []CompactionVerificationResult{fail("x")}}
	h := NewStandardHarness(compactionCfg(agent, cm, v, 2))
	var span uint64
	var usage AggregateUsage
	session := SessionState{}
	h.runCompaction(context.Background(), &session, "s", "t", &span, &usage, h.config.Agent)
	if agent.turns != 0 {
		t.Fatalf("expected 0 turns for non-compacting manager, got %d", agent.turns)
	}
}

// alwaysCompactNoopCM implements ContextManager (ShouldCompact true) but NOT
// CompactingContextManager.
type alwaysCompactNoopCM struct{ NoopContextManager }

func (alwaysCompactNoopCM) ShouldCompact(*SessionState) bool { return true }

// ----------------------------------------------------------------------------
// capturingObserver — a minimal HarnessObserver test double for the loop.
// ----------------------------------------------------------------------------

type capturingObserver struct {
	compactions      int
	warns            int
	lastWarnMissing  []string
	lastWarnAccepted bool
}

func newCapturingObserver() *capturingObserver { return &capturingObserver{} }

func (o *capturingObserver) EmitTurn(string, SessionID, TaskID, uint32, string, uint64, TokenUsage, float64, StopReason, uint32, string, string, []ToolCall, []Message) {
}
func (o *capturingObserver) EmitToolCall(string, string, SessionID, TaskID, string, string, string, uint64, uint64, uint64, bool, bool, json.RawMessage, string) {
}
func (o *capturingObserver) SetSessionOutcome(SessionID, TerminalOutcome, string) {}
func (o *capturingObserver) FlushSession(context.Context, SessionID)              {}
func (o *capturingObserver) CostFor(TokenUsage) float64                           { return 0 }
func (o *capturingObserver) EmitCompaction(string, SessionID, TaskID, string, uint32, uint32, uint32, uint32) {
	o.compactions++
}
func (o *capturingObserver) EmitCompactionVerificationFailed(_ string, _ SessionID, _ TaskID, _ string, missing []string, accepted bool) {
	o.warns++
	o.lastWarnMissing = missing
	o.lastWarnAccepted = accepted
}
func (o *capturingObserver) EmitHillClimbingIteration(string, SessionID, TaskID, string, uint32, float64, bool, float64, bool, string, bool) {
}
func (o *capturingObserver) EmitConsultSpawned(string, SessionID, TaskID, string, string) {}
func (o *capturingObserver) EmitConsultResumed(string, SessionID, TaskID, string, string, bool) {
}
func (o *capturingObserver) EmitToolErrorLoopDetected(string, SessionID, TaskID, string, string, uint32) {
}
func (o *capturingObserver) EmitToolErrorLoopBroken(string, SessionID, TaskID, string, string, uint32) {
}

var _ HarnessObserver = (*capturingObserver)(nil)

// ----------------------------------------------------------------------------
// Fixture replay against fixtures/compaction_loop/cases.json.
// ----------------------------------------------------------------------------

type loopFixtureVerdict struct {
	Passed       bool     `json:"passed"`
	MissingItems []string `json:"missing_items"`
}

type loopFixtureCase struct {
	Name                  string               `json:"name"`
	MaxCompactionAttempts uint32               `json:"max_compaction_attempts"`
	Verdicts              []loopFixtureVerdict `json:"verdicts"`
	Expected              struct {
		Attempts             int      `json:"attempts"`
		ApplyCompactionCalls int      `json:"apply_compaction_calls"`
		WarnEmitted          bool     `json:"warn_emitted"`
		RetryInjectedMissing []string `json:"retry_injected_missing"`
		FinalMissingItems    []string `json:"final_missing_items"`
		AcceptedAnyway       bool     `json:"accepted_anyway"`
	} `json:"expected"`
}

type loopFixtureFile struct {
	Cases []loopFixtureCase `json:"cases"`
}

// scriptedVerifier returns scripted verdicts from fixture data (last repeats).
type scriptedVerifier struct {
	verdicts []loopFixtureVerdict
	calls    int
}

func (v *scriptedVerifier) Verify(_ string, _ *CompactionTurn) CompactionVerificationResult {
	idx := v.calls
	v.calls++
	if idx >= len(v.verdicts) {
		idx = len(v.verdicts) - 1
	}
	d := v.verdicts[idx]
	missing := d.MissingItems
	if missing == nil {
		missing = []string{}
	}
	return CompactionVerificationResult{Passed: d.Passed, MissingItems: missing}
}

func TestCompactionLoopFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core → ../../fixtures/compaction_loop/cases.json
	path := filepath.Join(wd, "..", "..", "fixtures", "compaction_loop", "cases.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", path, err)
	}
	var file loopFixtureFile
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(file.Cases) == 0 {
		t.Fatal("expected at least one case")
	}

	for _, c := range file.Cases {
		t.Run(c.Name, func(t *testing.T) {
			cm := &compactingCM{shouldCompact: true}
			agent := &recordingAgent{cm: cm, summary: "summary"}
			v := &scriptedVerifier{verdicts: c.Verdicts}
			obs := newCapturingObserver()
			cfg := compactionCfg(agent, cm, v, c.MaxCompactionAttempts)
			cfg.Observability = obs
			h := NewStandardHarness(cfg)

			var span uint64
			var usage AggregateUsage
			session := SessionState{}
			if cm.ShouldCompact(&session) {
				h.runCompaction(context.Background(), &session, SessionID("sess-"+c.Name), "task", &span, &usage, h.config.Agent)
			}

			if agent.turns != c.Expected.Attempts {
				t.Fatalf("attempts = %d, want %d", agent.turns, c.Expected.Attempts)
			}
			if cm.applyCount != c.Expected.ApplyCompactionCalls {
				t.Fatalf("apply = %d, want %d", cm.applyCount, c.Expected.ApplyCompactionCalls)
			}
			if (obs.warns > 0) != c.Expected.WarnEmitted {
				t.Fatalf("warn_emitted = %v, want %v", obs.warns > 0, c.Expected.WarnEmitted)
			}
			if c.Expected.WarnEmitted {
				if !reflect.DeepEqual(obs.lastWarnMissing, c.Expected.FinalMissingItems) {
					t.Fatalf("final missing = %#v, want %#v", obs.lastWarnMissing, c.Expected.FinalMissingItems)
				}
				if obs.lastWarnAccepted != c.Expected.AcceptedAnyway {
					t.Fatalf("accepted_anyway = %v, want %v", obs.lastWarnAccepted, c.Expected.AcceptedAnyway)
				}
			}
			if len(c.Expected.RetryInjectedMissing) > 0 {
				if len(cm.injectedTurns) < 2 {
					t.Fatalf("expected >=2 turns, got %d", len(cm.injectedTurns))
				}
				want := injectMissingItemsText(c.Expected.RetryInjectedMissing)
				found := false
				for _, msg := range cm.injectedTurns[1].Messages {
					if msg.Content.Text == want {
						found = true
					}
				}
				if !found {
					t.Fatalf("retry context missing %q; got %#v", want, cm.injectedTurns[1].Messages)
				}
			}
		})
	}
}
