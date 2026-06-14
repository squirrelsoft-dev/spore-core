package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Loop-replay integration tests for output-schema delivery + enforcement
// (issue #139).
//
// Three recorded traces under fixtures/model_responses/harness/:
//   - output_schema_accept.jsonl — turn-1 terminal already satisfies the schema
//     ⇒ RunSuccess in ONE turn.
//   - output_schema_retry.jsonl — turn-1 terminal is INVALID, the harness feeds
//     the FROZEN feedback message back, turn-2 terminal is valid ⇒ Success in TWO
//     turns. The fixture's SECOND request carries the exact frozen feedback text
//     (hash-load-bearing).
//   - output_schema_fail.jsonl — every terminal is invalid; after
//     OutputSchemaMaxRetries == 2 extra turns (3 attempts) WITH budget remaining
//     ⇒ RunFailure{OutputSchemaViolation}.
//
// Replay is POSITIONAL (no request_hash in these fixtures): the responses drive
// the flow in order. The four languages must produce the same outcome + turn
// count — never edit a fixture to make a failing implementation pass (see
// fixtures/README.md).

func osHarnessFor(t *testing.T, fixture string, maxRetries uint32) *StandardHarness {
	t.Helper()
	path := filepath.Join(fixtureRoot(t), "model_responses", "harness", fixture)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	replay, err := ParseReplayJSONL(string(raw), ProviderInfo{
		Name: "ollama", ModelID: "fixture", ContextWindow: 200_000,
	})
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	agent := NewModelAgent(AgentID("fixture-agent"), replay)

	cfg := HarnessConfig{
		Agent:             agent,
		ToolRegistry:      NewScriptedToolRegistry(),
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
		EscalationMode:    AutonomousEscalation(),
		// #139: enforcement ON; N == maxRetries; schema registered under the empty
		// SchemaRef key (the leaf carries Output = &SchemaRef("")).
		EnforceOutputSchemas:   true,
		OutputSchemaMaxRetries: &maxRetries,
		Registry: NewExecutionRegistryBuilder().
			Schema("", json.RawMessage(osSchemaJSON)).
			Build(),
	}
	return NewStandardHarness(cfg)
}

func osFixtureTask(budget uint32) Task {
	out := SchemaRef("")
	return NewTask("produce a status report", SessionID("output-schema-session"),
		LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
			Budget:   BudgetPolicy{Kind: BudgetPerLoop, Value: budget},
			Behavior: BudgetExhaustedBehavior{Kind: BehaviorFail},
			Agent:    AgentRef(""),
			Toolset:  ToolsetRef(""),
			Output:   &out,
		}})
}

const osFixtureFeedback = `Your previous response did not match the required output schema. Missing required property "status". Reply with only a JSON value that satisfies the schema.`

func TestOutputSchemaAcceptSucceedsInOneTurn(t *testing.T) {
	h := osHarnessFor(t, "output_schema_accept.jsonl", 2)
	r := h.Run(context.Background(), NewHarnessRunOptions(osFixtureTask(10)))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Output != osValid {
		t.Fatalf("output = %q, want %q", r.Output, osValid)
	}
	if r.Turns != 1 {
		t.Fatalf("turns = %d, want 1 (valid terminal accepted on turn 1)", r.Turns)
	}
}

func TestOutputSchemaRetryFeedsFrozenMessageThenSucceeds(t *testing.T) {
	h := osHarnessFor(t, "output_schema_retry.jsonl", 2)
	r := h.Run(context.Background(), NewHarnessRunOptions(osFixtureTask(10)))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success after retry, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Output != osValid {
		t.Fatalf("output = %q, want %q", r.Output, osValid)
	}
	if r.Turns != 2 {
		t.Fatalf("turns = %d, want 2 (one retry consumed)", r.Turns)
	}
	// The harness fed the EXACT frozen feedback (with the validator error) back as
	// a user message — the same bytes the fixture's second request embeds.
	fed := false
	for _, m := range osUserMsgs(r.SessionState) {
		if m == osFixtureFeedback {
			fed = true
		}
	}
	if !fed {
		t.Fatalf("exact frozen feedback message must be fed back; got %q", osUserMsgs(r.SessionState))
	}
}

func TestOutputSchemaFailTerminatesWithViolation(t *testing.T) {
	h := osHarnessFor(t, "output_schema_fail.jsonl", 2)
	r := h.Run(context.Background(), NewHarnessRunOptions(osFixtureTask(50)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltOutputSchemaViolation {
		t.Fatalf("expected Failure{OutputSchemaViolation}, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Reason.Attempts != 3 {
		t.Fatalf("attempts = %d, want 3 (1 + N == 1 + 2)", r.Reason.Attempts)
	}
	if r.Turns != 3 {
		t.Fatalf("turns = %d, want 3 (exactly 1 + N turns; budget not exhausted)", r.Turns)
	}
	if r.Turns >= 50 {
		t.Fatalf("distinct from budget exhaustion expected, got turns = %d", r.Turns)
	}
	if r.Reason.LastError != `Missing required property "status".` {
		t.Fatalf("last_error = %q", r.Reason.LastError)
	}
	if r.SessionID != "output-schema-session" {
		t.Fatalf("session id = %q", r.SessionID)
	}
}
