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
