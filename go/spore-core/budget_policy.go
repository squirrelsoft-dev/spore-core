// BudgetPolicy + BudgetExhaustedBehavior — composable-execution budget
// vocabulary (issue #117, PRD Part B).
//
// These are pure, serializable value types — no executor wiring. Later slices
// thread them through the strategy tree. They layer *on top of* BudgetLimits
// (the global turns/tokens/wall/cost backstop), which is unchanged.
//
// BudgetPolicy — a per-scope step allowance. A step is one model turn (matches
// BudgetSnapshot.Turns). PerGoal is intentionally excluded in v1.
//   - Unlimited   — no per-scope cap.
//   - TotalSteps  — cap across the whole run.
//   - PerLoop     — cap per loop iteration.
//   - PerAttempt  — cap per attempt.
//
// BudgetExhaustedBehavior — what to do when a policy's allowance is spent. No
// node silently defaults to Continue (there is deliberately no Default).
//   - Continue { MaxContinues, OnExhausted } — grant up to MaxContinues extra
//     rounds, then fall through to the nested OnExhausted behavior.
//     MaxContinues == 0 means immediate fall-through.
//   - Escalate — hand off to a parent/escalation path.
//   - Fail     — terminate with failure.
//
// Serialized forms (internally tagged on "kind", snake_case):
//   - {"kind":"unlimited"}
//   - {"kind":"total_steps","value":N}
//   - {"kind":"per_loop","value":N}
//   - {"kind":"per_attempt","value":N}
//   - {"kind":"escalate"}
//   - {"kind":"fail"}
//   - {"kind":"continue","max_continues":N,"on_exhausted":{...nested...}}
//
// Cross-language note: the wire format is byte-identical across Rust,
// TypeScript, Python, and Go. value and max_continues are uint32.
//
// Go divergence: Rust models these as enums; Go uses the established
// flat-tagged-struct + custom Marshal/Unmarshal pattern (see LoopStrategy in
// harness.go). The recursive OnExhausted is a pointer, mirroring Rust's Box.

package sporecore

import (
	"encoding/json"
	"fmt"
)

// BudgetPolicyKind is the discriminator tag for a BudgetPolicy.
type BudgetPolicyKind string

const (
	// BudgetUnlimited is no per-scope cap.
	BudgetUnlimited BudgetPolicyKind = "unlimited"
	// BudgetTotalSteps caps steps across the whole run.
	BudgetTotalSteps BudgetPolicyKind = "total_steps"
	// BudgetPerLoop caps steps per loop iteration.
	BudgetPerLoop BudgetPolicyKind = "per_loop"
	// BudgetPerAttempt caps steps per attempt.
	BudgetPerAttempt BudgetPolicyKind = "per_attempt"
)

// BudgetPolicy is a per-scope step allowance tagged-union. Value applies to the
// TotalSteps, PerLoop, and PerAttempt variants; it is ignored for Unlimited.
type BudgetPolicy struct {
	Kind  BudgetPolicyKind `json:"kind"`
	Value uint32           `json:"-"`
}

// MarshalJSON serialises BudgetPolicy as a flat tagged object.
func (p BudgetPolicy) MarshalJSON() ([]byte, error) {
	switch p.Kind {
	case BudgetUnlimited:
		return json.Marshal(struct {
			Kind BudgetPolicyKind `json:"kind"`
		}{p.Kind})
	case BudgetTotalSteps, BudgetPerLoop, BudgetPerAttempt:
		return json.Marshal(struct {
			Kind  BudgetPolicyKind `json:"kind"`
			Value uint32           `json:"value"`
		}{p.Kind, p.Value})
	default:
		return nil, fmt.Errorf("BudgetPolicy: unknown kind %q", p.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form, rejecting unknown/missing kinds.
func (p *BudgetPolicy) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind  BudgetPolicyKind `json:"kind"`
		Value uint32           `json:"value"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	switch probe.Kind {
	case BudgetUnlimited, BudgetTotalSteps, BudgetPerLoop, BudgetPerAttempt:
		p.Kind = probe.Kind
		p.Value = probe.Value
		return nil
	default:
		return fmt.Errorf("BudgetPolicy: unknown kind %q", probe.Kind)
	}
}

// BudgetExhaustedBehaviorKind is the discriminator tag for a
// BudgetExhaustedBehavior.
type BudgetExhaustedBehaviorKind string

const (
	// BehaviorContinue grants extra rounds, then falls through to OnExhausted.
	BehaviorContinue BudgetExhaustedBehaviorKind = "continue"
	// BehaviorEscalate hands off to a parent/escalation path.
	BehaviorEscalate BudgetExhaustedBehaviorKind = "escalate"
	// BehaviorFail terminates with failure.
	BehaviorFail BudgetExhaustedBehaviorKind = "fail"
)

// BudgetExhaustedBehavior is a tagged-union describing what to do when a
// policy's allowance is spent. MaxContinues and OnExhausted apply only to the
// Continue variant; OnExhausted is a pointer so the type can nest recursively
// (mirrors Rust's Box<BudgetExhaustedBehavior>).
type BudgetExhaustedBehavior struct {
	Kind         BudgetExhaustedBehaviorKind `json:"kind"`
	MaxContinues uint32                      `json:"-"`
	OnExhausted  *BudgetExhaustedBehavior    `json:"-"`
}

// MarshalJSON serialises BudgetExhaustedBehavior as a flat tagged object.
func (b BudgetExhaustedBehavior) MarshalJSON() ([]byte, error) {
	switch b.Kind {
	case BehaviorContinue:
		if b.OnExhausted == nil {
			return nil, fmt.Errorf("BudgetExhaustedBehavior: continue requires on_exhausted")
		}
		return json.Marshal(struct {
			Kind         BudgetExhaustedBehaviorKind `json:"kind"`
			MaxContinues uint32                      `json:"max_continues"`
			OnExhausted  *BudgetExhaustedBehavior    `json:"on_exhausted"`
		}{b.Kind, b.MaxContinues, b.OnExhausted})
	case BehaviorEscalate, BehaviorFail:
		return json.Marshal(struct {
			Kind BudgetExhaustedBehaviorKind `json:"kind"`
		}{b.Kind})
	default:
		return nil, fmt.Errorf("BudgetExhaustedBehavior: unknown kind %q", b.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form, recursively for the Continue
// variant, and rejects unknown/missing kinds (never silently defaults to
// Continue).
func (b *BudgetExhaustedBehavior) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind         BudgetExhaustedBehaviorKind `json:"kind"`
		MaxContinues *uint32                     `json:"max_continues"`
		OnExhausted  *BudgetExhaustedBehavior    `json:"on_exhausted"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	switch probe.Kind {
	case BehaviorContinue:
		// max_continues is required with no silent default (spec contract);
		// a *uint32 probe distinguishes absent from an explicit 0.
		if probe.MaxContinues == nil {
			return fmt.Errorf("BudgetExhaustedBehavior: continue requires max_continues")
		}
		if probe.OnExhausted == nil {
			return fmt.Errorf("BudgetExhaustedBehavior: continue requires on_exhausted")
		}
		b.Kind = probe.Kind
		b.MaxContinues = *probe.MaxContinues
		b.OnExhausted = probe.OnExhausted
		return nil
	case BehaviorEscalate, BehaviorFail:
		b.Kind = probe.Kind
		b.MaxContinues = 0
		b.OnExhausted = nil
		return nil
	default:
		return fmt.Errorf("BudgetExhaustedBehavior: unknown kind %q", probe.Kind)
	}
}
