package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Fixture-replay test: load the shared dispatch_scenarios.json from
// fixtures/tool_registry/ and assert each outcome.
//
// The fixture file is byte-identical to the one consumed by the Rust,
// TypeScript, and Python implementations. Do not edit the fixture to
// make this test pass — fix the implementation instead.

type fixtureCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

type fixtureExpected struct {
	Kind   string `json:"kind"`
	CallID string `json:"call_id,omitempty"`
	Error  string `json:"error,omitempty"`
}

type fixtureScenario struct {
	Name     string               `json:"name"`
	Register []RegistryToolSchema `json:"register"`
	Sets     []ToolSet            `json:"sets,omitempty"`
	Call     fixtureCall          `json:"call"`
	Expected fixtureExpected      `json:"expected"`
}

func TestToolRegistryFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core → ../../fixtures/tool_registry/...
	path := filepath.Join(wd, "..", "..", "fixtures", "tool_registry", "dispatch_scenarios.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", path, err)
	}
	var scenarios []fixtureScenario
	if err := json.Unmarshal(data, &scenarios); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(scenarios) == 0 {
		t.Fatal("expected at least one scenario")
	}

	for _, sc := range scenarios {
		sc := sc
		t.Run(sc.Name, func(t *testing.T) {
			reg := NewStandardToolRegistry()
			for _, s := range sc.Register {
				if err := reg.Register(&echoTool{name: s.Name}, s); err != nil {
					t.Fatalf("register %q: %v", s.Name, err)
				}
			}
			for _, set := range sc.Sets {
				if err := reg.RegisterSet(set); err != nil {
					t.Fatalf("register set %q: %v", set.Name, err)
				}
			}
			result, err := reg.Dispatch(
				context.Background(),
				ToolCall{ID: sc.Call.ID, Name: sc.Call.Name, Input: sc.Call.Input},
				allowAllSandbox{},
			)
			switch sc.Expected.Kind {
			case "ok":
				if err != nil {
					t.Fatalf("expected ok, got error %v", err)
				}
				if result.CallID != sc.Expected.CallID {
					t.Fatalf("call_id: got %q, want %q", result.CallID, sc.Expected.CallID)
				}
			case "err":
				if err == nil {
					t.Fatalf("expected error %q, got ok %+v", sc.Expected.Error, result)
				}
				de, ok := err.(*DispatchError)
				if !ok {
					t.Fatalf("expected *DispatchError, got %T: %v", err, err)
				}
				actual := string(de.Kind)
				if actual != sc.Expected.Error {
					t.Fatalf("error kind: got %q, want %q", actual, sc.Expected.Error)
				}
			default:
				t.Fatalf("unknown expected kind %q", sc.Expected.Kind)
			}
		})
	}
}
