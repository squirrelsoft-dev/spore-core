package contextmgr_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
)

// ============================================================================
// Helpers
// ============================================================================

func richManager(t *testing.T) *contextmgr.StandardContextManager {
	t.Helper()
	cfg := contextmgr.CompactionConfig{
		Threshold:             0.80,
		PreserveRecentN:       2,
		HeadTailTokens:        64,
		OffloadPath:           ".spore/offload",
		MaxCompactionAttempts: 2,
	}
	model := sporecore.NewMockModel(sporecore.ProviderInfo{Name: "stub", ModelID: "stub", ContextWindow: 200000})
	return contextmgr.NewStandardContextManager(model, nil, cfg)
}

func richState(messages int, used, limit uint32) *contextmgr.SessionState {
	s := contextmgr.NewSessionState("s1", "t1", "deploy the payment service")
	s.WindowLimit = limit
	s.TokenBudgetUsed = used
	hist := make([]sporecore.Message, 0, messages)
	for i := 0; i < messages; i++ {
		hist = append(hist, sporecore.Message{
			Role:    sporecore.RoleUser,
			Content: sporecore.NewTextContent("m" + string(rune('0'+i%10))),
		})
	}
	s.MessageHistory = hist
	return &s
}

func sessionWith(rich *contextmgr.SessionState) sporecore.SessionState {
	var s sporecore.SessionState
	contextmgr.SeedRichState(&s, rich)
	return s
}

func msgsText(msgs []sporecore.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(m.Content.Text)
		b.WriteString("\n")
	}
	return b.String()
}

// ============================================================================
// ShouldCompact threshold
// ============================================================================

func TestShouldCompactBelowThresholdFalse(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	s := sessionWith(richState(10, 70, 100)) // 0.70 < 0.80
	if a.ShouldCompact(&s) {
		t.Fatal("should not compact below threshold")
	}
}

func TestShouldCompactAtThresholdTrue(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	s := sessionWith(richState(10, 80, 100)) // 0.80 >= 0.80
	if !a.ShouldCompact(&s) {
		t.Fatal("should compact at threshold")
	}
}

func TestShouldCompactAboveThresholdTrue(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	s := sessionWith(richState(10, 95, 100))
	if !a.ShouldCompact(&s) {
		t.Fatal("should compact above threshold")
	}
}

func TestShouldCompactFalseWithoutRichState(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	var s sporecore.SessionState
	if a.ShouldCompact(&s) {
		t.Fatal("should not compact without rich state")
	}
}

// ============================================================================
// PrepareCompactionTurn
// ============================================================================

func TestPrepareReturnsNoneForShortHistory(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	// PreserveRecentN = 2, history = 2 -> nothing to compact.
	s := sessionWith(richState(2, 95, 100))
	if _, ok := a.PrepareCompactionTurn(&s); ok {
		t.Fatal("expected no compaction turn for short history")
	}
}

func TestPrepareReturnsNoneWithoutRichState(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	var s sporecore.SessionState
	if _, ok := a.PrepareCompactionTurn(&s); ok {
		t.Fatal("expected no compaction turn without rich state")
	}
}

func TestPrepareProjectsHintsStateAndCount(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	rich := richState(10, 95, 100) // 10 - 2 preserved = 8 removed
	s := sessionWith(rich)
	turn, ok := a.PrepareCompactionTurn(&s)
	if !ok {
		t.Fatal("expected a compaction turn")
	}
	if turn.MessagesRemoved != 8 {
		t.Fatalf("MessagesRemoved = %d, want 8", turn.MessagesRemoved)
	}
	// VerificationState mirrors the rich state.
	vs, ok := turn.VerificationState.(*contextmgr.SessionState)
	if !ok {
		t.Fatalf("VerificationState type = %T, want *contextmgr.SessionState", turn.VerificationState)
	}
	if vs.TaskInstruction != "deploy the payment service" {
		t.Fatalf("verification task = %q", vs.TaskInstruction)
	}
	if vs.TokenBudgetUsed != 95 {
		t.Fatalf("verification token used = %d, want 95", vs.TokenBudgetUsed)
	}
	// Default hints projected.
	hints, ok := turn.PreserveHints.(contextmgr.CompactionPreserveHints)
	if !ok {
		t.Fatalf("PreserveHints type = %T", turn.PreserveHints)
	}
	if !hints.KeepCurrentTaskState {
		t.Fatal("expected KeepCurrentTaskState true")
	}
	// The summarization instruction is appended after the compacted msgs.
	if !strings.Contains(msgsText(turn.Context.Messages), "Summarize") {
		t.Fatal("expected summarization instruction in context")
	}
}

// ============================================================================
// InjectMissingItems text (fixture wording)
// ============================================================================

func TestInjectMissingItemsText(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	c := sporecore.Context{}
	a.InjectMissingItems(&c, []string{"payment", "deploy"})
	last := c.Messages[len(c.Messages)-1]
	want := "Your summary is missing these items: payment, deploy. Please revise."
	if last.Content.Text != want {
		t.Fatalf("inject text = %q, want %q", last.Content.Text, want)
	}
	if last.Role != sporecore.RoleUser {
		t.Fatalf("inject role = %q, want user", last.Role)
	}
}

// ============================================================================
// ApplyCompaction shrinks the session
// ============================================================================

func TestApplyCompactionShrinksSession(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	s := sessionWith(richState(10, 95, 100))
	before := len(s.Messages)
	a.ApplyCompaction(&s, "summary preserving payment deploy")
	// 2 preserved + 1 summary = 3.
	if len(s.Messages) >= before {
		t.Fatalf("session did not shrink: %d >= %d", len(s.Messages), before)
	}
	if len(s.Messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(s.Messages))
	}
	if s.Messages[0].Role != sporecore.RoleAssistant {
		t.Fatalf("first message role = %q, want assistant", s.Messages[0].Role)
	}
	// Round-tripped rich state also shrank to the same shape.
	if _, ok := a.PrepareCompactionTurn(&s); ok {
		// After shrinking to 3 with PreserveRecentN=2, prepare still has 1 to
		// compact; just assert the rich state survived the round-trip below.
	}
}

func TestApplyCompactionSwallowsErrorWithoutRichState(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	var s sporecore.SessionState
	// No rich state -> no-op, no panic.
	a.ApplyCompaction(&s, "summary")
	if len(s.Messages) != 0 {
		t.Fatalf("messages = %d, want 0", len(s.Messages))
	}
}

func TestApplyCompactionRoundTripsRichState(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	s := sessionWith(richState(10, 95, 100))
	a.ApplyCompaction(&s, "summary preserving payment deploy")
	// Serialize/deserialize the harness session (pause/resume) and confirm the
	// rich state blob survives — proving the bridge is stateless on Extras.
	raw, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	var restored sporecore.SessionState
	if err := json.Unmarshal(raw, &restored); err != nil {
		t.Fatal(err)
	}
	if got, ok := a.TokenBudgetUsed(&restored); !ok || got > 95 {
		// The rich state (budget + history) must survive the JSON round-trip.
		t.Fatalf("rich state did not survive pause/resume round-trip: used=%d present=%v", got, ok)
	}
}

// ============================================================================
// AppendUserMessage / AppendToolResult keep rich state in sync
// ============================================================================

func TestAppendKeepsRichStateInSync(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	s := sessionWith(richState(2, 10, 100)) // below threshold
	a.AppendUserMessage(context.Background(), &s, "new turn")
	if len(s.Messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(s.Messages))
	}
	// The prepared turn should now see 3 messages (1 compactable beyond the 2
	// preserved), proving the append synced into the rich state.
	turn, ok := a.PrepareCompactionTurn(&s)
	if !ok {
		t.Fatal("expected a compaction turn after append")
	}
	if turn.MessagesRemoved != 1 {
		t.Fatalf("MessagesRemoved = %d, want 1", turn.MessagesRemoved)
	}
}

// ============================================================================
// ReplaceToolResult (issue #158 / SC-9 in-place AfterTool rewrite)
// ============================================================================

func TestReplaceToolResultRerendersTheRecordedMessage(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	var s sporecore.SessionState

	original := &sporecore.HarnessToolResult{
		CallID: "c1",
		Output: sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "ORIGINAL"},
	}
	a.AppendToolResult(context.Background(), &s, original)
	idx := len(s.Messages) - 1
	if s.Messages[idx].Content.Text != "ORIGINAL" {
		t.Fatalf("appended text = %q, want ORIGINAL", s.Messages[idx].Content.Text)
	}

	// A middleware rewrote the result; re-render the recorded message.
	rewritten := &sporecore.HarnessToolResult{
		CallID: "c1",
		Output: sporecore.ToolOutput{Kind: sporecore.ToolOutputError, Message: "REWRITTEN", Recoverable: true},
	}
	a.ReplaceToolResult(context.Background(), &s, idx, rewritten)
	if len(s.Messages) != 1 {
		t.Fatalf("no message added on replace: len = %d, want 1", len(s.Messages))
	}
	if s.Messages[idx].Role != sporecore.RoleTool || s.Messages[idx].Content.Text != "REWRITTEN" {
		t.Fatalf("replaced message = (%q, %q), want (tool, REWRITTEN)",
			s.Messages[idx].Role, s.Messages[idx].Content.Text)
	}

	// Out-of-range index is a defensive no-op.
	a.ReplaceToolResult(context.Background(), &s, 99, original)
	if len(s.Messages) != 1 {
		t.Fatalf("out-of-range replace must be a no-op: len = %d, want 1", len(s.Messages))
	}
	if s.Messages[idx].Content.Text != "REWRITTEN" {
		t.Fatalf("out-of-range replace must not touch existing messages, got %q", s.Messages[idx].Content.Text)
	}
}

// ============================================================================
// End-to-end through the real harness compaction seam (mock model)
// ============================================================================

// summaryAgent returns a fixed summary when it sees a compaction turn (the
// context carries the "Summarize" instruction), else a tool-call request that
// drives the loop to the compaction check, then a final response to terminate.
type summaryAgent struct {
	summary string
	turns   int
}

func (a *summaryAgent) Turn(_ context.Context, c sporecore.Context) sporecore.TurnResult {
	if strings.Contains(msgsText(c.Messages), "Summarize") {
		return sporecore.NewFinalResponse(a.summary, sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1})
	}
	a.turns++
	if a.turns == 1 {
		return sporecore.NewToolCallRequested(
			[]sporecore.ToolCall{{ID: "c1", Name: "noop", Input: json.RawMessage(`{}`)}},
			sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1},
		)
	}
	return sporecore.NewFinalResponse("done", sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1})
}

func (a *summaryAgent) ID() sporecore.AgentID { return "summary" }

func TestEndToEndDrivesCompactionThroughSeam(t *testing.T) {
	inner := richManager(t)
	agent := &summaryAgent{summary: "we are working on the deploy of the payment service"}
	provider := observability.NewInMemoryObservabilityProvider()
	obs := observability.NewHarnessObserver(provider, observability.DefaultPricing())

	cfg := contextmgr.NewCompactingHarnessConfig(
		inner, agent, sporecore.NewScriptedToolRegistry(),
		sporecore.AllowAllSandbox{}, sporecore.AlwaysContinuePolicy{},
	)
	cfg.Observability = obs
	h := sporecore.NewStandardHarness(cfg)

	rich := richState(10, 95, 100)
	session := sessionWith(rich)

	task := sporecore.NewTask("deploy the payment service", "s1",
		sporecore.ReActStrategy(5))
	opts := sporecore.HarnessRunOptions{Task: task, SessionState: &session}
	result := h.Run(context.Background(), opts)

	if result.Kind == sporecore.RunFailure {
		t.Fatalf("run failed: %+v", result.Reason)
	}

	// A real compaction fired through the seam: turn 1 made a tool call
	// (triggering ShouldCompact via the adapter), compaction ran and shrank the
	// session, then turn 2 ended the run. Assert via the derived compaction
	// metric (the run mutates the harness's own session copy).
	metrics, err := provider.GetSessionMetrics(context.Background(), result.SessionID)
	if err != nil {
		t.Fatalf("metrics: %v", err)
	}
	if metrics.Compactions != 1 {
		t.Fatalf("compactions = %d, want 1", metrics.Compactions)
	}
	// Verification passed first time -> no warn span / no accepted-anyway.
	if metrics.CompactionVerificationFailures != 0 {
		t.Fatalf("verification failures = %d, want 0", metrics.CompactionVerificationFailures)
	}
	if warns := provider.WarnSpans(result.SessionID); len(warns) != 0 {
		t.Fatalf("unexpected warn spans: %d", len(warns))
	}
}

// ============================================================================
// compaction_loop fixture parity (verify->retry->warn) with the real adapter
// ============================================================================

type fixtureFile struct {
	Cases []fixtureCase `json:"cases"`
}

type fixtureCase struct {
	Name                  string    `json:"name"`
	MaxCompactionAttempts uint32    `json:"max_compaction_attempts"`
	Verdicts              []verdict `json:"verdicts"`
	Expected              expected  `json:"expected"`
}

type verdict struct {
	Passed       bool     `json:"passed"`
	MissingItems []string `json:"missing_items"`
}

type expected struct {
	ApplyCompactionCalls uint32   `json:"apply_compaction_calls"`
	WarnEmitted          bool     `json:"warn_emitted"`
	RetryInjectedMissing []string `json:"retry_injected_missing"`
}

// fixtureVerifier returns scripted verdicts (last entry repeats).
type fixtureVerifier struct {
	verdicts []verdict
	idx      int
}

func (v *fixtureVerifier) Verify(_ string, _ *sporecore.CompactionTurn) sporecore.CompactionVerificationResult {
	i := v.idx
	if i >= len(v.verdicts) {
		i = len(v.verdicts) - 1
	}
	v.idx++
	out := v.verdicts[i]
	missing := out.MissingItems
	if missing == nil {
		missing = []string{}
	}
	return sporecore.CompactionVerificationResult{Passed: out.Passed, MissingItems: missing}
}

// replayCompactionLoop mirrors StandardHarness.runCompaction's documented
// verify->retry->warn control flow, driving the REAL adapter + a scripted
// verifier (runCompaction itself is unexported; this is the in-language parity
// harness). Returns apply count, whether a warn fired, and the contexts the
// agent saw in order (to assert retry injection).
func replayCompactionLoop(
	a *contextmgr.StandardCompactionAdapter,
	session *sporecore.SessionState,
	v sporecore.CompactionVerifier,
	maxAttempts uint32,
) (applyCalls int, warnEmitted bool, seen []sporecore.Context) {
	turn, ok := a.PrepareCompactionTurn(session)
	if !ok {
		return 0, false, nil
	}
	if maxAttempts < 1 {
		maxAttempts = 1
	}
	for attempt := uint32(1); ; attempt++ {
		seen = append(seen, turn.Context)
		summary := "summary" // the agent's response stand-in
		res := v.Verify(summary, turn)
		if res.Passed {
			a.ApplyCompaction(session, summary)
			applyCalls++
			return applyCalls, warnEmitted, seen
		}
		if attempt < maxAttempts {
			a.InjectMissingItems(&turn.Context, res.MissingItems)
			continue
		}
		// Exhausted: warn, then accept anyway.
		warnEmitted = true
		a.ApplyCompaction(session, summary)
		applyCalls++
		return applyCalls, warnEmitted, seen
	}
}

func TestCompactionLoopFixtureParity(t *testing.T) {
	path := filepath.Join("..", "..", "..", "fixtures", "compaction_loop", "cases.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fixture fixtureFile
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("no fixture cases")
	}

	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			a := contextmgr.NewStandardCompactionAdapter(richManager(t))
			session := sessionWith(richState(10, 95, 100))
			v := &fixtureVerifier{verdicts: c.Verdicts}

			applyCalls, warnEmitted, seen := replayCompactionLoop(a, &session, v, c.MaxCompactionAttempts)

			// apply_compaction always runs exactly once (accept or accept-anyway).
			if uint32(applyCalls) != c.Expected.ApplyCompactionCalls {
				t.Fatalf("apply calls = %d, want %d", applyCalls, c.Expected.ApplyCompactionCalls)
			}
			// The session actually shrank.
			if len(session.Messages) != 3 {
				t.Fatalf("messages after apply = %d, want 3", len(session.Messages))
			}
			// Warn parity.
			if warnEmitted != c.Expected.WarnEmitted {
				t.Fatalf("warn = %v, want %v", warnEmitted, c.Expected.WarnEmitted)
			}
			// Retry injection parity: the second context carries the missing-items prompt.
			if len(c.Expected.RetryInjectedMissing) > 0 {
				if len(seen) < 2 {
					t.Fatalf("expected a retry turn, saw %d contexts", len(seen))
				}
				retryText := msgsText(seen[1].Messages)
				if !strings.Contains(retryText, "missing these items") {
					t.Fatalf("retry context missing the prompt: %q", retryText)
				}
				for _, item := range c.Expected.RetryInjectedMissing {
					if !strings.Contains(retryText, item) {
						t.Fatalf("retry context missing item %q", item)
					}
				}
			}
		})
	}
}

// ============================================================================
// Real token reclamation + healthy multi-compaction (issue #57)
// ============================================================================

// richStateHeavy builds a rich state whose messages carry enough content to
// produce a non-trivial token estimate (the chars/4 proxy), so a compaction
// that drops them reclaims real tokens.
func richStateHeavy(messages int, used, limit uint32) *contextmgr.SessionState {
	s := contextmgr.NewSessionState("s1", "t1", "deploy the payment service")
	s.WindowLimit = limit
	s.TokenBudgetUsed = used
	hist := make([]sporecore.Message, 0, messages)
	for i := 0; i < messages; i++ {
		hist = append(hist, sporecore.Message{
			Role: sporecore.RoleUser,
			Content: sporecore.NewTextContent(
				"history message with a fair amount of content to estimate tokens from " +
					string(rune('a'+i%26)),
			),
		})
	}
	s.MessageHistory = hist
	return &s
}

func TestApplyCompactionReclaimsRealTokensAndDropsBudget(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	s := sessionWith(richStateHeavy(10, 95, 100))

	before, ok := a.TokenBudgetUsed(&s)
	if !ok || before != 95 {
		t.Fatalf("before budget = %d present=%v, want 95", before, ok)
	}
	a.ApplyCompaction(&s, "summary preserving payment deploy")
	after, ok := a.TokenBudgetUsed(&s)
	if !ok {
		t.Fatal("post-compaction budget not present")
	}
	if after >= before {
		t.Fatalf("TokenBudgetUsed must drop after real reclamation: %d -> %d", before, after)
	}
}

func TestApplyCompactionMultiCompactionKeepsDroppingBudget(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	s := sessionWith(richStateHeavy(10, 95, 100))

	a.ApplyCompaction(&s, "first summary about payment deploy")
	afterFirst, _ := a.TokenBudgetUsed(&s)
	if afterFirst >= 95 {
		t.Fatalf("first compaction did not reclaim: %d", afterFirst)
	}

	// Simulate the session growing again past threshold, then compact again.
	grown := richStateHeavy(10, 95, 100)
	contextmgr.SeedRichState(&s, grown)
	a.ApplyCompaction(&s, "second summary about payment deploy")
	afterSecond, _ := a.TokenBudgetUsed(&s)
	if afterSecond >= 95 {
		t.Fatalf("second compaction did not reclaim: %d", afterSecond)
	}
}

// ============================================================================
// SC-26/#115 — structural ContextSources rendering
// ============================================================================

// TestAssembleEmptyWhenNoSources: empty sources → no System block → the adapter
// adds no System message, so the no-source path is byte-identical to the
// pre-#115 pass-through of session.Messages.
func TestAssembleEmptyWhenNoSources(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	session := sporecore.SessionState{Messages: []sporecore.Message{
		{Role: sporecore.RoleUser, Content: sporecore.NewTextContent("hi")},
	}}
	task := sporecore.NewTask("t", sporecore.NewSessionID(), sporecore.ReActStrategy(4))
	c := a.Assemble(context.Background(), &session, &task, sporecore.ContextSources{})
	if len(c.Messages) != 1 || c.Messages[0].Role != sporecore.RoleUser {
		t.Fatalf("empty sources must not add a System block; got %+v", c.Messages)
	}
}

// TestAssembleFormatsGuidesAsLeadingSystemBlock: guides render name-headed, in
// registration order, joined by blank lines, as a single LEADING System message.
func TestAssembleFormatsGuidesAsLeadingSystemBlock(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	session := sporecore.SessionState{Messages: []sporecore.Message{
		{Role: sporecore.RoleUser, Content: sporecore.NewTextContent("hi")},
	}}
	task := sporecore.NewTask("t", sporecore.NewSessionID(), sporecore.ReActStrategy(4))
	sources := sporecore.ContextSources{Guides: []sporecore.Guide{
		{ID: "audit", Content: "AUDIT BODY"},
		{ID: "style", Content: "STYLE BODY"},
	}}
	c := a.Assemble(context.Background(), &session, &task, sources)
	if len(c.Messages) != 2 {
		t.Fatalf("expected a leading System block + the user message; got %+v", c.Messages)
	}
	if c.Messages[0].Role != sporecore.RoleSystem {
		t.Fatalf("guides must lead as a System message; got role %q", c.Messages[0].Role)
	}
	if got, want := c.Messages[0].Content.Text, "# audit\nAUDIT BODY\n\n# style\nSTYLE BODY"; got != want {
		t.Fatalf("guide block mismatch:\n got %q\nwant %q", got, want)
	}
	if c.Messages[1].Role != sporecore.RoleUser {
		t.Fatalf("the original user message must follow the System block")
	}
}

// TestRenderContextBlockAppendsMemoryAfterGuides: #163 — memory items render
// into the SAME structural System block, after the guides, as plain content
// joined by blank lines (mirrors Rust render_context_block_appends_memory_after_guides).
func TestRenderContextBlockAppendsMemoryAfterGuides(t *testing.T) {
	a := contextmgr.NewStandardCompactionAdapter(richManager(t))
	session := sporecore.SessionState{Messages: []sporecore.Message{
		{Role: sporecore.RoleUser, Content: sporecore.NewTextContent("hi")},
	}}
	task := sporecore.NewTask("t", sporecore.NewSessionID(), sporecore.ReActStrategy(4))
	sources := sporecore.ContextSources{
		Guides: []sporecore.Guide{{ID: "audit", Content: "AUDIT BODY"}},
		Memory: []sporecore.MemoryItem{{Key: "m1", Content: "MEMORY CONTENT"}},
	}
	c := a.Assemble(context.Background(), &session, &task, sources)
	if len(c.Messages) != 2 || c.Messages[0].Role != sporecore.RoleSystem {
		t.Fatalf("expected a leading System block + the user message; got %+v", c.Messages)
	}
	if got, want := c.Messages[0].Content.Text, "# audit\nAUDIT BODY\n\nMEMORY CONTENT"; got != want {
		t.Fatalf("memory must render after guides in the same block:\n got %q\nwant %q", got, want)
	}
}
