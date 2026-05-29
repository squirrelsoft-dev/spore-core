package tools

import (
	"context"
	"encoding/json"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func TestEnterPlanModeEscalates(t *testing.T) {
	r := NewEnterPlanModeTool().Execute(context.Background(),
		call("enter_plan_mode", "c1", map[string]any{"context": "seed"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputEscalate || r.Signal == nil {
		t.Fatalf("expected escalate, got %+v", r)
	}
	if r.Signal.Kind != sporecore.SignalEnterPlanMode || r.Signal.Context != "seed" {
		t.Fatalf("signal: %+v", r.Signal)
	}
}

func TestExitPlanModeEscalatesWithPlan(t *testing.T) {
	r := NewExitPlanModeTool().Execute(context.Background(),
		call("exit_plan_mode", "c1", map[string]any{"plan": map[string]any{"tasks": []string{"a", "b"}, "rationale": "because"}}),
		sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputEscalate || r.Signal == nil || r.Signal.Kind != sporecore.SignalExitPlanMode {
		t.Fatalf("expected exit_plan_mode escalate, got %+v", r)
	}
	if r.Signal.Plan == nil || len(r.Signal.Plan.Tasks) != 2 || r.Signal.Plan.Rationale != "because" {
		t.Fatalf("plan: %+v", r.Signal.Plan)
	}
}

func TestExitPlanModeRationaleDefaults(t *testing.T) {
	r := NewExitPlanModeTool().Execute(context.Background(),
		call("exit_plan_mode", "c1", map[string]any{"plan": map[string]any{"tasks": []string{"x"}}}),
		sporecore.AllowAllSandbox{}, nil)
	if r.Signal == nil || r.Signal.Plan == nil || r.Signal.Plan.Rationale != "" {
		t.Fatalf("expected empty rationale, got %+v", r.Signal)
	}
}

func TestAskUserQuestionAwaitsClarification(t *testing.T) {
	r := NewAskUserQuestionTool().Execute(context.Background(),
		call("ask_user_question", "c1", map[string]any{"question": "which?", "options": []string{"a", "b"}}),
		sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputAwaitingClarification || r.Question != "which?" {
		t.Fatalf("expected awaiting_clarification, got %+v", r)
	}
	if r.Options == nil || len(*r.Options) != 2 {
		t.Fatalf("options: %+v", r.Options)
	}
}

func TestAskUserQuestionOptionsOptional(t *testing.T) {
	r := NewAskUserQuestionTool().Execute(context.Background(),
		call("ask_user_question", "c1", map[string]any{"question": "free form?"}),
		sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputAwaitingClarification || r.Question != "free form?" {
		t.Fatalf("expected awaiting_clarification, got %+v", r)
	}
	if r.Options != nil {
		t.Fatalf("expected nil options, got %+v", r.Options)
	}
}

func TestAbortEscalates(t *testing.T) {
	r := NewAbortTool().Execute(context.Background(),
		call("abort", "c1", map[string]any{"reason": "stop"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputEscalate || r.Signal == nil || r.Signal.Kind != sporecore.SignalAbort || r.Signal.Reason != "stop" {
		t.Fatalf("expected abort escalate, got %+v", r)
	}
}

func TestAbortMissingReasonRecoverable(t *testing.T) {
	r := NewAbortTool().Execute(context.Background(),
		call("abort", "c1", map[string]any{}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

// Fixture replay: escalation_tools.json.
func TestEscalationToolsFixtureReplay(t *testing.T) {
	type expected struct {
		ToolOutputKind string          `json:"tool_output_kind"`
		Signal         json.RawMessage `json:"signal"`
		Question       string          `json:"question"`
		Options        *[]string       `json:"options"`
	}
	type escCase struct {
		Name     string          `json:"name"`
		Tool     string          `json:"tool"`
		Input    json.RawMessage `json:"input"`
		Expected expected        `json:"expected"`
	}
	data := readFixture(t, "escalation_tools.json")
	var cases []escCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	dispatch := func(tool string, input json.RawMessage) sporecore.ToolOutput {
		c := sporecore.ToolCall{ID: "c1", Name: tool, Input: input}
		sb := sporecore.AllowAllSandbox{}
		switch tool {
		case "enter_plan_mode":
			return NewEnterPlanModeTool().Execute(context.Background(), c, sb, nil)
		case "exit_plan_mode":
			return NewExitPlanModeTool().Execute(context.Background(), c, sb, nil)
		case "ask_user_question":
			return NewAskUserQuestionTool().Execute(context.Background(), c, sb, nil)
		case "abort":
			return NewAbortTool().Execute(context.Background(), c, sb, nil)
		default:
			t.Fatalf("unknown tool %q", tool)
			return sporecore.ToolOutput{}
		}
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			out := dispatch(c.Tool, c.Input)
			switch c.Expected.ToolOutputKind {
			case "escalate":
				if out.Kind != sporecore.ToolOutputEscalate || out.Signal == nil {
					t.Fatalf("expected escalate, got %+v", out)
				}
				// Round-trip the signal to JSON and compare to the fixture's signal.
				gotSig, err := json.Marshal(*out.Signal)
				if err != nil {
					t.Fatal(err)
				}
				if !jsonEqual(t, gotSig, c.Expected.Signal) {
					t.Fatalf("signal: got %s want %s", gotSig, c.Expected.Signal)
				}
			case "awaiting_clarification":
				if out.Kind != sporecore.ToolOutputAwaitingClarification {
					t.Fatalf("expected awaiting_clarification, got %+v", out)
				}
				if out.Question != c.Expected.Question {
					t.Fatalf("question: got %q want %q", out.Question, c.Expected.Question)
				}
				if (out.Options == nil) != (c.Expected.Options == nil) {
					t.Fatalf("options presence: got %+v want %+v", out.Options, c.Expected.Options)
				}
				if out.Options != nil && c.Expected.Options != nil {
					if len(*out.Options) != len(*c.Expected.Options) {
						t.Fatalf("options len: got %v want %v", *out.Options, *c.Expected.Options)
					}
				}
			default:
				t.Fatalf("unexpected kind %q", c.Expected.ToolOutputKind)
			}
		})
	}
}

// jsonEqual reports whether two JSON blobs are semantically equal.
func jsonEqual(t *testing.T, a, b []byte) bool {
	t.Helper()
	var av, bv any
	if err := json.Unmarshal(a, &av); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &bv); err != nil {
		t.Fatal(err)
	}
	am, _ := json.Marshal(av)
	bm, _ := json.Marshal(bv)
	return string(am) == string(bm)
}
