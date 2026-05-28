package contextmgr

import (
	"reflect"
	"testing"
)

// taskHintsOnly returns hints with only KeepCurrentTaskState enabled.
func taskHintsOnly() CompactionPreserveHints {
	return CompactionPreserveHints{KeepCurrentTaskState: true}
}

func stateWith(instruction string) *SessionState {
	s := NewSessionState("sess", "task", instruction)
	return &s
}

func TestDefaultMaxCompactionAttemptsIsTwo(t *testing.T) {
	if got := DefaultCompactionConfig().MaxCompactionAttempts; got != 2 {
		t.Fatalf("MaxCompactionAttempts = %d, want 2", got)
	}
}

func TestKeyTermVerifierAllPresent(t *testing.T) {
	v := KeyTermVerifier{}
	res := v.Verify("We will refactor the parser module to be faster.", taskHintsOnly(), stateWith("Refactor the parser module"))
	if !res.Passed {
		t.Fatalf("expected pass, got %+v", res)
	}
	if !reflect.DeepEqual(res.MissingItems, []string{}) {
		t.Fatalf("MissingItems = %#v, want []", res.MissingItems)
	}
	if res.Detail != "all 3 key term(s) present" {
		t.Fatalf("Detail = %q", res.Detail)
	}
}

func TestKeyTermVerifierMissingListed(t *testing.T) {
	v := KeyTermVerifier{}
	res := v.Verify("We will refactor the parser.", taskHintsOnly(), stateWith("Refactor the parser module"))
	if res.Passed {
		t.Fatalf("expected fail, got %+v", res)
	}
	if !reflect.DeepEqual(res.MissingItems, []string{"module"}) {
		t.Fatalf("MissingItems = %#v, want [module]", res.MissingItems)
	}
	if res.Detail != "missing 1 of 3 key term(s): module" {
		t.Fatalf("Detail = %q", res.Detail)
	}
}

func TestKeyTermVerifierTaskStateOffNoTerms(t *testing.T) {
	v := KeyTermVerifier{}
	hints := DefaultPreserveHints()
	hints.KeepCurrentTaskState = false
	res := v.Verify("Nothing in particular.", hints, stateWith("Refactor the parser module"))
	if !res.Passed {
		t.Fatalf("expected pass, got %+v", res)
	}
	if !reflect.DeepEqual(res.MissingItems, []string{}) {
		t.Fatalf("MissingItems = %#v, want []", res.MissingItems)
	}
	if res.Detail != "all 0 key term(s) present" {
		t.Fatalf("Detail = %q", res.Detail)
	}
}

func TestKeyTermVerifierShortTokensIgnored(t *testing.T) {
	v := KeyTermVerifier{}
	// "the", "api" are < 4 chars and dropped; "test", "endpoint" remain.
	res := v.Verify("Wrote a test for the endpoint.", taskHintsOnly(), stateWith("Test the api endpoint"))
	if !res.Passed {
		t.Fatalf("expected pass (short tokens ignored), got %+v", res)
	}
}

func TestKeyTermVerifierCaseInsensitive(t *testing.T) {
	v := KeyTermVerifier{}
	res := v.Verify("REFACTOR THE PARSER MODULE", taskHintsOnly(), stateWith("refactor the parser module"))
	if !res.Passed {
		t.Fatalf("expected pass (case-insensitive), got %+v", res)
	}
}

func TestKeyTermVerifierDedupe(t *testing.T) {
	v := KeyTermVerifier{}
	// "deploy" and "service" each appear; both missing, but listed once each.
	res := v.Verify("An unrelated note.", taskHintsOnly(), stateWith("Deploy deploy the service"))
	if res.Passed {
		t.Fatalf("expected fail, got %+v", res)
	}
	if !reflect.DeepEqual(res.MissingItems, []string{"deploy", "service"}) {
		t.Fatalf("MissingItems = %#v, want [deploy service]", res.MissingItems)
	}
}

func TestKeyTermVerifierNonTaskHintsAreNoOps(t *testing.T) {
	v := KeyTermVerifier{}
	hints := CompactionPreserveHints{
		KeepArchitecturalDecisions: true,
		KeepOpenProblems:           true,
		KeepCurrentTaskState:       false,
		KeepRecentFileList:         true,
		KeepThinkingBlocks:         true,
	}
	res := v.Verify("Nothing here at all.", hints, stateWith("Refactor the parser module"))
	if !res.Passed || len(res.MissingItems) != 0 {
		t.Fatalf("expected zero terms from non-task hints, got %+v", res)
	}
	if res.Detail != "all 0 key term(s) present" {
		t.Fatalf("Detail = %q", res.Detail)
	}
}

// ── Issue #47: structured fields feed the four additional hints ──────────────

func hintOnly(set func(*CompactionPreserveHints)) CompactionPreserveHints {
	h := CompactionPreserveHints{}
	set(&h)
	return h
}

func TestKeyTermVerifierOpenProblemsIsolated(t *testing.T) {
	v := KeyTermVerifier{}
	st := stateWith("ignored task")
	st.OpenProblems = []string{"Resolve the deadlock issue"}
	hints := hintOnly(func(h *CompactionPreserveHints) { h.KeepOpenProblems = true })
	res := v.Verify("we noted the deadlock", hints, st)
	if res.Passed {
		t.Fatalf("expected fail, got %+v", res)
	}
	if !reflect.DeepEqual(res.MissingItems, []string{"resolve", "issue"}) {
		t.Fatalf("MissingItems = %#v, want [resolve issue]", res.MissingItems)
	}
}

func TestKeyTermVerifierArchitecturalDecisionsIsolated(t *testing.T) {
	v := KeyTermVerifier{}
	st := stateWith("ignored task")
	st.ArchitecturalDecisions = []string{"Adopt hexagonal architecture"}
	hints := hintOnly(func(h *CompactionPreserveHints) { h.KeepArchitecturalDecisions = true })
	res := v.Verify("we will adopt hexagonal architecture", hints, st)
	if !res.Passed {
		t.Fatalf("expected pass, got %+v", res)
	}
	if !reflect.DeepEqual(res.MissingItems, []string{}) {
		t.Fatalf("MissingItems = %#v, want []", res.MissingItems)
	}
}

func TestKeyTermVerifierRecentFilesPathTokenization(t *testing.T) {
	v := KeyTermVerifier{}
	st := stateWith("ignored task")
	st.RecentFiles = []string{"src/parser/mod.rs"}
	// src, mod, rs are < 4 chars and dropped; only "parser" survives.
	hints := hintOnly(func(h *CompactionPreserveHints) { h.KeepRecentFileList = true })
	res := v.Verify("touched the lexer", hints, st)
	if res.Passed {
		t.Fatalf("expected fail, got %+v", res)
	}
	if !reflect.DeepEqual(res.MissingItems, []string{"parser"}) {
		t.Fatalf("MissingItems = %#v, want [parser]", res.MissingItems)
	}
}

func TestKeyTermVerifierReasoningSummaryIsolated(t *testing.T) {
	v := KeyTermVerifier{}
	st := stateWith("ignored task")
	st.ReasoningSummary = "Considered caching strategy"
	hints := hintOnly(func(h *CompactionPreserveHints) { h.KeepThinkingBlocks = true })
	res := v.Verify("nothing relevant", hints, st)
	if res.Passed {
		t.Fatalf("expected fail, got %+v", res)
	}
	if !reflect.DeepEqual(res.MissingItems, []string{"considered", "caching", "strategy"}) {
		t.Fatalf("MissingItems = %#v, want [considered caching strategy]", res.MissingItems)
	}
}

func TestKeyTermVerifierMultiHintDedupOrdering(t *testing.T) {
	v := KeyTermVerifier{}
	// "parser" reachable via both task_instruction and open_problems;
	// first-occurrence is the task position (pushed first). "bug" < 4 dropped.
	st := stateWith("Refactor parser")
	st.OpenProblems = []string{"parser bug remains"}
	hints := CompactionPreserveHints{
		KeepOpenProblems:     true,
		KeepCurrentTaskState: true,
	}
	res := v.Verify("nothing matched", hints, st)
	if !reflect.DeepEqual(res.MissingItems, []string{"refactor", "parser", "remains"}) {
		t.Fatalf("MissingItems = %#v, want [refactor parser remains]", res.MissingItems)
	}
}

func TestKeyTermVerifierEmptyListWithHintOn(t *testing.T) {
	v := KeyTermVerifier{}
	st := stateWith("ignored task")
	// open_problems empty but its hint on ⇒ contributes nothing ⇒ passes.
	hints := hintOnly(func(h *CompactionPreserveHints) { h.KeepOpenProblems = true })
	res := v.Verify("anything", hints, st)
	if !res.Passed || len(res.MissingItems) != 0 {
		t.Fatalf("expected pass with no missing items, got %+v", res)
	}
}
