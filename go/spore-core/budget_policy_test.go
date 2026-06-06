package sporecore

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// assertJSONBytes round-trips v to JSON and asserts exact serialized bytes.
func assertJSONBytes(t *testing.T, v any, want string) {
	t.Helper()
	got, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(got) != want {
		t.Fatalf("serialized bytes mismatch\n got: %s\nwant: %s", got, want)
	}
}

func TestBudgetPolicyExactBytes(t *testing.T) {
	cases := []struct {
		name string
		val  BudgetPolicy
		want string
	}{
		{"unlimited", BudgetPolicy{Kind: BudgetUnlimited}, `{"kind":"unlimited"}`},
		{"total_steps", BudgetPolicy{Kind: BudgetTotalSteps, Value: 100}, `{"kind":"total_steps","value":100}`},
		{"per_loop", BudgetPolicy{Kind: BudgetPerLoop, Value: 10}, `{"kind":"per_loop","value":10}`},
		{"per_attempt", BudgetPolicy{Kind: BudgetPerAttempt, Value: 3}, `{"kind":"per_attempt","value":3}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			assertJSONBytes(t, c.val, c.want)
		})
	}
}

func TestBudgetPolicyRoundTrip(t *testing.T) {
	vals := []BudgetPolicy{
		{Kind: BudgetUnlimited},
		{Kind: BudgetTotalSteps, Value: 100},
		{Kind: BudgetPerLoop, Value: 10},
		{Kind: BudgetPerAttempt, Value: 3},
	}
	for _, v := range vals {
		t.Run(string(v.Kind), func(t *testing.T) {
			data, err := json.Marshal(v)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got BudgetPolicy
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			// Unlimited ignores Value; normalise before comparing.
			if v.Kind == BudgetUnlimited {
				got.Value = 0
			}
			if !reflect.DeepEqual(got, v) {
				t.Fatalf("round-trip mismatch: got %+v want %+v", got, v)
			}
		})
	}
}

func TestBudgetPolicyRejectsUnknownAndMissingKind(t *testing.T) {
	bad := []string{
		`{"kind":"bogus","value":1}`,
		`{"value":5}`,
		`{}`,
	}
	for _, b := range bad {
		t.Run(b, func(t *testing.T) {
			var p BudgetPolicy
			if err := json.Unmarshal([]byte(b), &p); err == nil {
				t.Fatalf("expected error for %q, got none (decoded %+v)", b, p)
			}
		})
	}
}

func TestBudgetExhaustedBehaviorExactBytes(t *testing.T) {
	cases := []struct {
		name string
		val  BudgetExhaustedBehavior
		want string
	}{
		{"escalate", BudgetExhaustedBehavior{Kind: BehaviorEscalate}, `{"kind":"escalate"}`},
		{"fail", BudgetExhaustedBehavior{Kind: BehaviorFail}, `{"kind":"fail"}`},
		{
			"continue_then_fail",
			BudgetExhaustedBehavior{
				Kind:         BehaviorContinue,
				MaxContinues: 2,
				OnExhausted:  &BudgetExhaustedBehavior{Kind: BehaviorFail},
			},
			`{"kind":"continue","max_continues":2,"on_exhausted":{"kind":"fail"}}`,
		},
		{
			"nested_continue_continue_fail",
			BudgetExhaustedBehavior{
				Kind:         BehaviorContinue,
				MaxContinues: 1,
				OnExhausted: &BudgetExhaustedBehavior{
					Kind:         BehaviorContinue,
					MaxContinues: 2,
					OnExhausted:  &BudgetExhaustedBehavior{Kind: BehaviorFail},
				},
			},
			`{"kind":"continue","max_continues":1,"on_exhausted":{"kind":"continue","max_continues":2,"on_exhausted":{"kind":"fail"}}}`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			assertJSONBytes(t, c.val, c.want)
		})
	}
}

func TestBudgetExhaustedBehaviorRoundTrip(t *testing.T) {
	vals := []BudgetExhaustedBehavior{
		{Kind: BehaviorEscalate},
		{Kind: BehaviorFail},
		{Kind: BehaviorContinue, MaxContinues: 2, OnExhausted: &BudgetExhaustedBehavior{Kind: BehaviorFail}},
		{
			Kind:         BehaviorContinue,
			MaxContinues: 1,
			OnExhausted: &BudgetExhaustedBehavior{
				Kind:         BehaviorContinue,
				MaxContinues: 2,
				OnExhausted:  &BudgetExhaustedBehavior{Kind: BehaviorFail},
			},
		},
	}
	for i, v := range vals {
		t.Run(string(rune('a'+i)), func(t *testing.T) {
			data, err := json.Marshal(v)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got BudgetExhaustedBehavior
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if !reflect.DeepEqual(got, v) {
				t.Fatalf("round-trip mismatch: got %+v want %+v", got, v)
			}
		})
	}
}

func TestBudgetExhaustedBehaviorRejectsUnknownAndMissingKind(t *testing.T) {
	bad := []string{
		`{"kind":"bogus"}`,
		`{"max_continues":1,"on_exhausted":{"kind":"fail"}}`, // missing kind: must NOT default to continue
		`{}`,
		`{"kind":"continue","max_continues":1}`, // continue without on_exhausted is invalid
		`{"kind":"continue","on_exhausted":{"kind":"fail"}}`, // continue without max_continues: no silent default to 0
	}
	for _, b := range bad {
		t.Run(b, func(t *testing.T) {
			var be BudgetExhaustedBehavior
			if err := json.Unmarshal([]byte(b), &be); err == nil {
				t.Fatalf("expected error for %q, got none (decoded %+v)", b, be)
			}
		})
	}
}

// budgetFixture mirrors fixtures/budget_policy/cases.json.
type budgetFixture struct {
	Policies  []json.RawMessage `json:"policies"`
	Behaviors []json.RawMessage `json:"behaviors"`
}

// TestBudgetPolicyFixtureReplay loads the shared budget_policy fixture and
// asserts a byte-identity round-trip: each raw case unmarshals into the Go
// type and re-marshals to a structurally identical JSON value.
func TestBudgetPolicyFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core → ../../fixtures/budget_policy/cases.json
	path := filepath.Join(wd, "..", "..", "fixtures", "budget_policy", "cases.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx budgetFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	if len(fx.Policies) == 0 || len(fx.Behaviors) == 0 {
		t.Fatalf("fixture missing cases: %d policies, %d behaviors", len(fx.Policies), len(fx.Behaviors))
	}

	for i, rawCase := range fx.Policies {
		t.Run("policy_"+string(rune('0'+i)), func(t *testing.T) {
			var p BudgetPolicy
			if err := json.Unmarshal(rawCase, &p); err != nil {
				t.Fatalf("unmarshal policy: %v", err)
			}
			assertByteIdentity(t, rawCase, p)
		})
	}
	for i, rawCase := range fx.Behaviors {
		t.Run("behavior_"+string(rune('0'+i)), func(t *testing.T) {
			var be BudgetExhaustedBehavior
			if err := json.Unmarshal(rawCase, &be); err != nil {
				t.Fatalf("unmarshal behavior: %v", err)
			}
			assertByteIdentity(t, rawCase, be)
		})
	}
}

// assertByteIdentity re-marshals v and asserts it is structurally identical to
// the original fixture bytes (whitespace/key-order insensitive).
func assertByteIdentity(t *testing.T, original json.RawMessage, v any) {
	t.Helper()
	got, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var gotAny, wantAny any
	if err := json.Unmarshal(got, &gotAny); err != nil {
		t.Fatalf("re-unmarshal got: %v", err)
	}
	if err := json.Unmarshal(original, &wantAny); err != nil {
		t.Fatalf("unmarshal original: %v", err)
	}
	if !reflect.DeepEqual(gotAny, wantAny) {
		t.Fatalf("byte-identity mismatch\n got: %s\nwant: %s", got, original)
	}
}
