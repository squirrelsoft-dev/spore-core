package sporecore

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// R8: serde round-trip is byte-identical for every consult type (issue #114).
// For each section of fixtures/harness/consult.json we parse the raw JSON into
// the typed Go value, re-serialize it, and assert it equals the original parsed
// value — proving the wire shape is stable and the SAME shared fixture replays
// identically across the four languages. The fixtures are ground truth: NEVER
// edit them to make a test pass.
func TestConsultFixtureReplayRoundTrips(t *testing.T) {
	path := filepath.Join(fixtureRoot(t), "harness", "consult.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}

	// check round-trips each element of a named array through T.
	check := func(key string, newT func() any) {
		arrRaw, ok := root[key]
		if !ok {
			t.Fatalf("missing array %q", key)
		}
		var arr []json.RawMessage
		if err := json.Unmarshal(arrRaw, &arr); err != nil {
			t.Fatalf("%q: not an array: %v", key, err)
		}
		if len(arr) == 0 {
			t.Fatalf("%q must have cases", key)
		}
		for i, caseRaw := range arr {
			typed := newT()
			if err := json.Unmarshal(caseRaw, typed); err != nil {
				t.Fatalf("%q[%d] deserialize: %v", key, i, err)
			}
			back, err := json.Marshal(typed)
			if err != nil {
				t.Fatalf("%q[%d] serialize: %v", key, i, err)
			}
			if !jsonEqual(t, caseRaw, back) {
				t.Fatalf("%q[%d] not byte-identical:\n want %s\n  got %s", key, i, caseRaw, back)
			}
		}
	}

	check("run_result_cases", func() any { return new(RunResult) })
	check("worker_tool_output_cases", func() any { return new(ToolOutput) })
	check("subagent_tool_output_cases", func() any { return new(ToolOutput) })
	check("consult_request_cases", func() any { return new(ConsultRequest) })
	check("consult_response_cases", func() any { return new(ConsultResponse) })
	check("consult_overflow_policy_cases", func() any { return new(ConsultOverflowPolicy) })

	// Spot-check the documented invariants on the structured cases.
	var rrCases []map[string]json.RawMessage
	if err := json.Unmarshal(root["run_result_cases"], &rrCases); err != nil {
		t.Fatal(err)
	}
	rr := rrCases[0]
	if string(rr["kind"]) != `"consult"` {
		t.Fatalf("run_result[0] kind = %s", rr["kind"])
	}
	var state map[string]json.RawMessage
	if err := json.Unmarshal(rr["state"], &state); err != nil {
		t.Fatal(err)
	}
	if string(state["human_request"]) != "null" {
		t.Fatalf("human_request must be null, got %s", state["human_request"])
	}
	if string(state["child_state"]) != "null" {
		t.Fatalf("child_state must be null at RunResult level, got %s", state["child_state"])
	}

	// Worker-side ToolOutput.Consult omits child_state (skip_serializing).
	var workerCases []map[string]json.RawMessage
	if err := json.Unmarshal(root["worker_tool_output_cases"], &workerCases); err != nil {
		t.Fatal(err)
	}
	if _, present := workerCases[0]["child_state"]; present {
		t.Fatalf("worker-side consult must omit child_state")
	}

	// Subagent-boundary ToolOutput.Consult carries a populated child_state.
	var subCases []map[string]json.RawMessage
	if err := json.Unmarshal(root["subagent_tool_output_cases"], &subCases); err != nil {
		t.Fatal(err)
	}
	cs, present := subCases[0]["child_state"]
	if !present || string(cs) == "null" {
		t.Fatalf("subagent boundary must populate child_state, got present=%v val=%s", present, cs)
	}
}
