package sporecore

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// ============================================================================
// Output-schema delivery + enforcement (issue #139).
//
// Mirrors the Rust reference harness tests (the os_* tests in
// rust/crates/spore-core/src/harness.rs). Same types, same rules, same frozen
// literals, same outcomes.
//
// Types/rules under test:
//   - HarnessConfig.EnforceOutputSchemas (MIGRATION GATE, default OFF).
//   - HarnessConfig.OutputSchemaMaxRetries (N, default 2; *uint32 sentinel).
//   - HaltReason{HaltOutputSchemaViolation, Schema, Attempts, LastError}.
//   - AC1: the resolved schema is DELIVERED to the leaf's directive seed
//     (key-sorted) AND set on ModelParams.OutputSchema (Ollama format tested in
//     ollama_test.go).
//   - AC2: terminal validated; valid ⇒ Success; invalid ⇒ feed the frozen error
//     back + retry within budget.
//   - AC3: after N retries WITH budget remaining ⇒ OutputSchemaViolation
//     (distinct from budget exhaustion; turns < budget). AND budget precedence:
//     a retry that would exceed the turn cap surfaces BudgetExceeded.
//   - AC4: flag OFF ⇒ an invalid terminal is accepted as Success.
// ============================================================================

// osSchemaJSON is the output schema the #139 tests enforce: an object requiring
// a status (one of ok/error) and a count integer.
const osSchemaJSON = `{"type":"object","required":["status","count"],` +
	`"properties":{"status":{"type":"string","enum":["ok","error"]},"count":{"type":"integer"}}}`

// osValid is a valid terminal body for osSchemaJSON.
const osValid = `{"status":"ok","count":3}`

// osInvalid is an invalid terminal body for osSchemaJSON (missing both required
// props).
const osInvalid = `{}`

// The canonical key-sorted schema bytes (delivered in the directive + reported
// on a violation). Pinned so the four ports match.
const osCanonical = `{"properties":{"count":{"type":"integer"},"status":{"enum":["ok","error"],"type":"string"}},"required":["status","count"],"type":"object"}`

// osMaxRetries returns a *uint32 for an explicit OutputSchemaMaxRetries (the N).
func osMaxRetries(n uint32) *uint32 { return &n }

// osConfig builds a standardCfg with output-schema enforcement ON, osSchemaJSON
// registered under the default empty SchemaRef key, and N == maxRetries. A nil
// maxRetries leaves the config field nil (default 2 via the sentinel).
func osConfig(agent Agent, maxRetries *uint32) HarnessConfig {
	cfg := standardCfg(agent)
	cfg.EnforceOutputSchemas = true
	cfg.OutputSchemaMaxRetries = maxRetries
	// Register the real schema under the empty key (replacing the default {} the
	// fill-only fillDefaultSchema would otherwise install).
	cfg.Registry = NewExecutionRegistryBuilder().
		Schema("", json.RawMessage(osSchemaJSON)).
		Build()
	return cfg
}

// osLeaf builds a bare ReAct leaf carrying Output = &SchemaRef("") and a PerLoop
// turn budget.
func osLeaf(budget uint32) Task {
	out := SchemaRef("")
	return NewTask("produce a status report", SessionID("s1"),
		LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
			Budget:   BudgetPolicy{Kind: BudgetPerLoop, Value: budget},
			Behavior: BudgetExhaustedBehavior{Kind: BehaviorFail},
			Agent:    AgentRef(""),
			Toolset:  ToolsetRef(""),
			Output:   &out,
		}})
}

// osUserMsgs returns the user-role text messages of a post-run session.
func osUserMsgs(state SessionState) []string {
	var out []string
	for _, m := range state.Messages {
		if m.Role == RoleUser && m.Content.Type == ContentTypeText {
			out = append(out, m.Content.Text)
		}
	}
	return out
}

// AC1: the resolved schema is delivered to the directive seed (key-sorted), and
// a valid terminal accepts on turn 1.
func TestOSAc1SchemaDeliveredToDirectiveSeed(t *testing.T) {
	a := NewMockAgent("test")
	a.Push(NewFinalResponse(osValid, turnUsage()))
	h := NewStandardHarness(osConfig(a, osMaxRetries(2)))
	r := h.Run(context.Background(), NewHarnessRunOptions(osLeaf(10)))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Turns != 1 {
		t.Fatalf("turns = %d, want 1 (valid on turn 1)", r.Turns)
	}
	delivered := false
	for _, m := range osUserMsgs(r.SessionState) {
		if strings.Contains(m, osCanonical) {
			delivered = true
		}
	}
	if !delivered {
		t.Fatalf("directive must carry the key-sorted schema; got %q", osUserMsgs(r.SessionState))
	}
}

// AC2 accept: schema ON, turn 1 valid → Success, turns == 1, NO feedback.
func TestOSAc2AcceptValidOnFirstTurn(t *testing.T) {
	a := NewMockAgent("test")
	a.Push(NewFinalResponse(osValid, turnUsage()))
	h := NewStandardHarness(osConfig(a, osMaxRetries(2)))
	r := h.Run(context.Background(), NewHarnessRunOptions(osLeaf(10)))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Output != osValid {
		t.Fatalf("output = %q, want %q", r.Output, osValid)
	}
	if r.Turns != 1 {
		t.Fatalf("turns = %d, want 1", r.Turns)
	}
	for _, m := range osUserMsgs(r.SessionState) {
		if strings.Contains(m, "did not match the required output schema") {
			t.Fatalf("no feedback message expected on a turn-1 accept; got %q", m)
		}
	}
}

// AC2 retry: turn 1 invalid → feed error back → turn 2 valid → Success,
// turns == 2, the frozen feedback (with the validator error) is present.
func TestOSAc2RetryInvalidThenValid(t *testing.T) {
	a := NewMockAgent("test")
	a.Push(NewFinalResponse(osInvalid, turnUsage()))
	a.Push(NewFinalResponse(osValid, turnUsage()))
	h := NewStandardHarness(osConfig(a, osMaxRetries(2)))
	r := h.Run(context.Background(), NewHarnessRunOptions(osLeaf(10)))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success after retry, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Output != osValid {
		t.Fatalf("output = %q, want %q", r.Output, osValid)
	}
	if r.Turns != 2 {
		t.Fatalf("turns = %d, want 2 (one retry consumed)", r.Turns)
	}
	// The frozen feedback for the FIRST-failing rule (missing required status,
	// array order) must be present, exact bytes.
	const wantFeedback = `Your previous response did not match the required output schema. Missing required property "status". Reply with only a JSON value that satisfies the schema.`
	found := false
	for _, m := range osUserMsgs(r.SessionState) {
		if m == wantFeedback {
			found = true
		}
	}
	if !found {
		t.Fatalf("exact frozen feedback must be fed back; got %q", osUserMsgs(r.SessionState))
	}
}

// AC3 fail: N+1 invalid terminals (N == 2 ⇒ 3 attempts) with a generous budget
// → OutputSchemaViolation, DISTINCT from budget, turns < budget.
func TestOSAc3FailAfterRetriesExhausted(t *testing.T) {
	a := NewMockAgent("test")
	for i := 0; i < 3; i++ {
		a.Push(NewFinalResponse(osInvalid, turnUsage()))
	}
	h := NewStandardHarness(osConfig(a, osMaxRetries(2)))
	r := h.Run(context.Background(), NewHarnessRunOptions(osLeaf(50)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltOutputSchemaViolation {
		t.Fatalf("expected Failure{OutputSchemaViolation}, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Reason.Attempts != 3 {
		t.Fatalf("attempts = %d, want 3 (1 + N == 1 + 2)", r.Reason.Attempts)
	}
	if r.Turns != 3 {
		t.Fatalf("turns = %d, want 3 (exactly 1 + N turns)", r.Turns)
	}
	if r.Turns >= 50 {
		t.Fatalf("budget NOT exhausted expected (distinct from budget), got turns = %d", r.Turns)
	}
	if r.Reason.LastError != `Missing required property "status".` {
		t.Fatalf("last_error = %q", r.Reason.LastError)
	}
	if r.Reason.Schema != osCanonical {
		t.Fatalf("schema = %s, want %s", r.Reason.Schema, osCanonical)
	}
}

// AC3 budget precedence: a tiny turn budget (2) is exhausted by retries BEFORE
// the N==5 retries run out → the BUDGET terminal wins, NOT OutputSchemaViolation
// (budget-cap-wins).
func TestOSAc3BudgetPrecedenceOverSchemaViolation(t *testing.T) {
	a := NewMockAgent("test")
	for i := 0; i < 5; i++ {
		a.Push(NewFinalResponse(osInvalid, turnUsage()))
	}
	// N == 5 (large), budget == 2 turns. After 2 invalid terminals the 3rd retry
	// re-enters the loop where the turn-budget gate fires.
	h := NewStandardHarness(osConfig(a, osMaxRetries(5)))
	r := h.Run(context.Background(), NewHarnessRunOptions(osLeaf(2)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltBudgetExceeded {
		t.Fatalf("expected Failure{BudgetExceeded} (budget wins), got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Turns != 2 {
		t.Fatalf("turns = %d, want 2 (stopped at the turn budget, not on schema)", r.Turns)
	}
}

// AC4: flag OFF ⇒ an INVALID terminal is accepted as Success (no resolve, no
// delivery, no validation) — the migration-gate guarantee.
func TestOSAc4FlagOffAcceptsInvalidTerminal(t *testing.T) {
	a := NewMockAgent("test")
	a.Push(NewFinalResponse(osInvalid, turnUsage()))
	// standardCfg has EnforceOutputSchemas = false; register the schema anyway to
	// prove it is IGNORED when the gate is OFF.
	cfg := standardCfg(a)
	cfg.Registry = NewExecutionRegistryBuilder().
		Schema("", json.RawMessage(osSchemaJSON)).
		Build()
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(osLeaf(10)))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success (gate OFF), got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Output != osInvalid {
		t.Fatalf("output = %q, want %q (invalid terminal accepted as-is)", r.Output, osInvalid)
	}
	if r.Turns != 1 {
		t.Fatalf("turns = %d, want 1", r.Turns)
	}
	for _, m := range osUserMsgs(r.SessionState) {
		if strings.Contains(m, "conforms to this") {
			t.Fatalf("no schema directive should be delivered when gate is OFF; got %q", m)
		}
	}
}

// AC4 + events: enforcement ON emits the retry + violation stream events.
func TestOSEmitsRetryAndViolationEvents(t *testing.T) {
	a := NewMockAgent("test")
	for i := 0; i < 2; i++ {
		a.Push(NewFinalResponse(osInvalid, turnUsage()))
	}
	var events []HarnessStreamEvent
	// N == 1 ⇒ 1 retry then violation (2 attempts).
	h := NewStandardHarness(osConfig(a, osMaxRetries(1)))
	opts := NewHarnessRunOptions(osLeaf(50))
	opts.OnStream = func(ev HarnessStreamEvent) { events = append(events, ev) }
	_ = h.Run(context.Background(), opts)

	var retries []uint32
	var violations []uint32
	for _, e := range events {
		switch e.Kind {
		case HarnessStreamOutputSchemaRetry:
			retries = append(retries, e.Attempt)
		case HarnessStreamOutputSchemaViolation:
			violations = append(violations, e.Attempts)
		}
	}
	if len(retries) != 1 || retries[0] != 1 {
		t.Fatalf("retries = %v, want [1] (one retry event at attempt 1)", retries)
	}
	if len(violations) != 1 || violations[0] != 2 {
		t.Fatalf("violations = %v, want [2] (one violation event, attempts == 2)", violations)
	}
}

// The *uint32 sentinel: nil → default 2; explicit 0 → first invalid is a
// violation; explicit n → n.
func TestOSMaxRetriesSentinel(t *testing.T) {
	if got := (HarnessConfig{}).effectiveOutputSchemaMaxRetries(); got != 2 {
		t.Fatalf("nil sentinel default = %d, want 2", got)
	}
	if got := (HarnessConfig{OutputSchemaMaxRetries: osMaxRetries(0)}).effectiveOutputSchemaMaxRetries(); got != 0 {
		t.Fatalf("explicit 0 = %d, want 0", got)
	}
	if got := (HarnessConfig{OutputSchemaMaxRetries: osMaxRetries(5)}).effectiveOutputSchemaMaxRetries(); got != 5 {
		t.Fatalf("explicit 5 = %d, want 5", got)
	}
}

// Explicit N == 0: the FIRST invalid terminal is a violation (no retry).
func TestOSMaxRetriesZeroFirstInvalidIsViolation(t *testing.T) {
	a := NewMockAgent("test")
	a.Push(NewFinalResponse(osInvalid, turnUsage()))
	h := NewStandardHarness(osConfig(a, osMaxRetries(0)))
	r := h.Run(context.Background(), NewHarnessRunOptions(osLeaf(10)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltOutputSchemaViolation {
		t.Fatalf("expected Failure{OutputSchemaViolation}, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Reason.Attempts != 1 {
		t.Fatalf("attempts = %d, want 1 (1 + 0)", r.Reason.Attempts)
	}
	if r.Turns != 1 {
		t.Fatalf("turns = %d, want 1", r.Turns)
	}
}
