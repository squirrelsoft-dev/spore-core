// Composable Execution Part A (issue #119): recursive LoopStrategy config
// newtypes + per-node collaborator handles + StrategyRef + RunStrategy.
//
// This file owns the strategy node shapes and the composition seam — types and
// the runtime trait only. Per-variant run bodies are STUBS (they return a
// placeholder StrategyOutcome and never panic); the real bodies land in #124.
// The live StandardHarness dispatch in harness.go is intentionally left in
// place (it also migrates in #124).
//
// Wire format (byte-identical across Rust, TypeScript, Python, Go):
//   - LoopStrategy is internally tagged on "kind" (snake_case). The ReAct
//     variant's tag is "react" (NOT "re_act").
//   - The leaf ReAct config flattens its fields next to "kind"; the combinators
//     nest their recursive children (plan / execute / inner) as tagged objects.
//   - JSON field order follows the Rust struct declaration order (cross-language
//     byte-identity target), so each variant is hand-marshalled in declaration
//     order rather than via a plain map (which would alphabetize keys).
//   - StrategyRef is adjacently tagged on "kind"/"value":
//       {"kind":"built_in","value":{...LoopStrategy...}}
//       {"kind":"custom","value":"my-harness::DoubleVerify"}
//
// Go divergence: Rust models LoopStrategy as a closed enum of config newtypes
// with trait/dyn dispatch. Go has no native sum type, so LoopStrategy is the
// established flat-tagged struct (a Kind discriminant carrying the active config
// pointer) with hand-written Marshal/Unmarshal, mirroring BudgetPolicy. Runtime
// polymorphism is the RunStrategy interface; the single dispatch is one switch
// in LoopStrategy.Run delegating to each per-config Run. Recursive children are
// *LoopStrategy (mirrors Rust's Box<LoopStrategy>).

package sporecore

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// ============================================================================
// Per-node collaborator handles
// ============================================================================

// AgentRef is a per-node handle to a named agent definition. It serializes as a
// bare JSON string. Resolution to a concrete agent lands with the registry
// slice (#120).
type AgentRef string

// ToolsetRef is a per-node handle to a named toolset. It serializes as a bare
// JSON string. Resolution lands with the registry slice (#120).
type ToolsetRef string

// SchemaRef is a per-node handle to a named output/evaluator schema. It
// serializes as a bare JSON string. Resolution lands with the registry slice
// (#120).
type SchemaRef string

// ============================================================================
// Strategy config newtypes
// ============================================================================

// ReactConfig is the leaf ReAct node config. Budget is the renamed
// max_iterations (semantically PerLoop(n)). It serializes flat next to
// "kind":"react": kind, budget, agent, toolset, output (output omitted when
// empty/absent).
type ReactConfig struct {
	Budget BudgetPolicy
	// Behavior is what this node does when its Budget is spent (#129). Canonical
	// wire position: IMMEDIATELY after budget. A leaf honors its Behavior ONLY at
	// the top-level/bare-leaf resolution site (driveStrategy); in the normal
	// NESTED case the leaf still PROPAGATES exhaustion to its parent (#125 rule
	// 6 — a nested leaf never self-resolves). Always serialized (Q1).
	Behavior BudgetExhaustedBehavior
	Agent    AgentRef
	Toolset  ToolsetRef
	// Output is omitted from JSON when nil (matches Rust Option + skip-if-none).
	Output *SchemaRef
}

// ReactPerLoop builds a bare ReAct leaf with a PerLoop{value} budget and empty
// agent / toolset handles (resolution lands with the registry slice, #120).
// This is the migration shim for the old ReAct{max_iterations} shape.
func ReactPerLoop(value uint32) ReactConfig {
	return ReactConfig{
		Budget:   BudgetPolicy{Kind: BudgetPerLoop, Value: value},
		Behavior: defaultBudgetBehavior(),
		Agent:    AgentRef(""),
		Toolset:  ToolsetRef(""),
		Output:   nil,
	}
}

// defaultBudgetBehavior is the default BudgetExhaustedBehavior for a config
// node's serialized behavior field (#129): Escalate. Two roles:
//   - the value UnmarshalJSON seeds when a strategy tree serialized BEFORE #129
//     omits the behavior key, preserving backward-compat reads;
//   - the value the migration shims stamp so a bare leaf keeps its pre-#129
//     propagate-to-parent contract by default.
//
// The field is NOT omitempty: it ALWAYS serializes (uniform wire shape across
// all five config structs, Q1), so the cross-language fixtures carry an explicit
// "behavior":{"kind":"escalate"} on every node.
func defaultBudgetBehavior() BudgetExhaustedBehavior {
	return BudgetExhaustedBehavior{Kind: BehaviorEscalate}
}

// behaviorOrDefault returns the decoded behavior, or defaultBudgetBehavior() if
// the field was absent (a pre-#129 tree). The behavior probe is a pointer so an
// absent key (nil) is distinguished from a present-but-Escalate one (#129
// backward-compat reads).
func behaviorOrDefault(b *BudgetExhaustedBehavior) BudgetExhaustedBehavior {
	if b == nil {
		return defaultBudgetBehavior()
	}
	return *b
}

// MaxIterations extracts the max_iterations value from a PerLoop budget; any
// other budget shape yields math.MaxUint32 (matching the legacy executor's
// "unbounded" fall-through for non-PerLoop strategies).
func (c ReactConfig) MaxIterations() uint32 {
	if c.Budget.Kind == BudgetPerLoop {
		return c.Budget.Value
	}
	return ^uint32(0)
}

// PlanExecuteConfig is the PlanExecute combinator: a plan sub-strategy feeds an
// execute sub-strategy. PlanModel stays optional/omittable.
type PlanExecuteConfig struct {
	Plan    *LoopStrategy
	Execute *LoopStrategy
	// PlanModel is omitted from JSON when nil.
	PlanModel *ModelConfig
	// Behavior is what this combinator does when its execute-phase budget is
	// spent (#129). Canonical wire position on a combinator: the LAST field.
	// Always serialized (Q1).
	Behavior BudgetExhaustedBehavior
}

// PlanExecuteSimple builds a PlanExecute whose plan and execute phases are both
// bare ReAct leaves (migration shim for the old PlanExecute{plan_model} shape).
// As of #124 the executor genuinely dispatches both children: the plan child via
// Plan.Run and the execute child via Execute.Run per task.
func PlanExecuteSimple(planModel *ModelConfig) PlanExecuteConfig {
	plan := ReActStrategy(^uint32(0))
	exec := ReActStrategy(^uint32(0))
	return PlanExecuteConfig{
		Plan:      &plan,
		Execute:   &exec,
		PlanModel: planModel,
		Behavior:  defaultBudgetBehavior(),
	}
}

// SelfVerifyingConfig is the SelfVerifying combinator: run inner, then judge it
// against evaluator.
type SelfVerifyingConfig struct {
	Inner     *LoopStrategy
	Evaluator SchemaRef
	// Behavior is what this combinator does when its build↔evaluate budget is
	// spent (#129). Canonical wire position on a combinator: the LAST field.
	// Always serialized (Q1).
	Behavior BudgetExhaustedBehavior
}

// RalphConfig is the Ralph combinator: re-run inner under a fixed agent across
// context-window resets.
type RalphConfig struct {
	Inner *LoopStrategy
	Agent AgentRef
	// Behavior is what Ralph does when its own scope is spent (#129). Canonical
	// wire position on a combinator: the LAST field. Always serialized (Q1).
	// NOTE: Ralph's window recovery (reset + retry) is independent of this field
	// — it governs Ralph's OWN budget scope, not the per-window child exhaustion
	// (which Ralph already absorbs as "window incomplete" and retries).
	Behavior BudgetExhaustedBehavior
}

// HillClimbingConfig is the HillClimbing combinator: iterate inner, keeping or
// reverting per the metric evaluator and direction. MaxStagnation and
// MinImprovementDelta are required (#119).
type HillClimbingConfig struct {
	Inner                 *LoopStrategy
	Direction             OptimizationDirection
	MaxStagnation         uint32
	RevertOnNoImprovement bool
	MinImprovementDelta   float64
	Evaluator             AgentRef
	// Behavior is what this combinator does when its optimization-loop budget is
	// spent (#129). Canonical wire position on a combinator: the LAST field.
	// Always serialized (Q1).
	Behavior BudgetExhaustedBehavior
}

// ============================================================================
// LoopStrategy — recursive closed tagged union
// ============================================================================

// LoopStrategyKind discriminates LoopStrategy variants.
type LoopStrategyKind string

const (
	// StrategyReAct is the leaf ReAct node. Wire tag "react".
	StrategyReAct LoopStrategyKind = "react"
	// StrategyPlanExecute feeds a plan sub-strategy into an execute sub-strategy.
	StrategyPlanExecute LoopStrategyKind = "plan_execute"
	// StrategySelfVerifying runs an inner strategy, then judges it.
	StrategySelfVerifying LoopStrategyKind = "self_verifying"
	// StrategyRalph re-runs an inner strategy across context-window resets.
	StrategyRalph LoopStrategyKind = "ralph"
	// StrategyHillClimbing iterates an inner strategy against a metric.
	StrategyHillClimbing LoopStrategyKind = "hill_climbing"
)

// LoopStrategy is a closed, recursive tagged union of config newtypes. ReAct is
// the leaf; the rest are combinators holding *LoopStrategy children. Exactly one
// config field is populated, selected by Kind. See the file header for the full
// type/wire-format/rule documentation.
type LoopStrategy struct {
	Kind         LoopStrategyKind
	ReActCfg     *ReactConfig
	PlanExecute  *PlanExecuteConfig
	SelfVerify   *SelfVerifyingConfig
	Ralph        *RalphConfig
	HillClimbing *HillClimbingConfig
}

// ReActStrategy builds a leaf ReAct LoopStrategy with a PerLoop{maxIterations}
// budget. Migration shim for the old LoopStrategy{Kind:StrategyReAct,
// MaxIterations:n} literal.
func ReActStrategy(maxIterations uint32) LoopStrategy {
	c := ReactPerLoop(maxIterations)
	return LoopStrategy{Kind: StrategyReAct, ReActCfg: &c}
}

// PtrStrategy returns a pointer to a copy of s — a small helper for building
// recursive combinator children (Plan / Execute / Inner) inline.
func PtrStrategy(s LoopStrategy) *LoopStrategy { return &s }

// PlanExecuteStrategy wraps a PlanExecuteConfig into a LoopStrategy.
func PlanExecuteStrategy(c PlanExecuteConfig) LoopStrategy {
	return LoopStrategy{Kind: StrategyPlanExecute, PlanExecute: &c}
}

// SelfVerifyingStrategy wraps a SelfVerifyingConfig into a LoopStrategy.
func SelfVerifyingStrategy(c SelfVerifyingConfig) LoopStrategy {
	return LoopStrategy{Kind: StrategySelfVerifying, SelfVerify: &c}
}

// RalphStrategy wraps a RalphConfig into a LoopStrategy.
func RalphStrategy(c RalphConfig) LoopStrategy {
	return LoopStrategy{Kind: StrategyRalph, Ralph: &c}
}

// HillClimbingStrategy wraps a HillClimbingConfig into a LoopStrategy.
func HillClimbingStrategy(c HillClimbingConfig) LoopStrategy {
	return LoopStrategy{Kind: StrategyHillClimbing, HillClimbing: &c}
}

// MaxIterations returns the leaf ReAct budget's iteration cap, or
// math.MaxUint32 for any non-ReAct strategy (the legacy executor's unbounded
// fall-through).
func (s LoopStrategy) MaxIterations() uint32 {
	if s.Kind == StrategyReAct && s.ReActCfg != nil {
		return s.ReActCfg.MaxIterations()
	}
	return ^uint32(0)
}

// MarshalJSON serializes LoopStrategy as a flat tagged object, emitting keys in
// Rust struct-declaration order (NOT alphabetical) for cross-language
// byte-identity.
func (s LoopStrategy) MarshalJSON() ([]byte, error) {
	switch s.Kind {
	case StrategyReAct:
		if s.ReActCfg == nil {
			return nil, fmt.Errorf("LoopStrategy: react requires config")
		}
		c := s.ReActCfg
		// Key order: kind, budget, behavior, agent, toolset, [output].
		type reactFlat struct {
			Kind     LoopStrategyKind        `json:"kind"`
			Budget   BudgetPolicy            `json:"budget"`
			Behavior BudgetExhaustedBehavior `json:"behavior"`
			Agent    AgentRef                `json:"agent"`
			Toolset  ToolsetRef              `json:"toolset"`
			Output   *SchemaRef              `json:"output,omitempty"`
		}
		return json.Marshal(reactFlat{s.Kind, c.Budget, c.Behavior, c.Agent, c.Toolset, c.Output})
	case StrategyPlanExecute:
		if s.PlanExecute == nil {
			return nil, fmt.Errorf("LoopStrategy: plan_execute requires config")
		}
		c := s.PlanExecute
		// Key order: kind, plan, execute, [plan_model], behavior (LAST).
		type planExecuteFlat struct {
			Kind      LoopStrategyKind        `json:"kind"`
			Plan      *LoopStrategy           `json:"plan"`
			Execute   *LoopStrategy           `json:"execute"`
			PlanModel *ModelConfig            `json:"plan_model,omitempty"`
			Behavior  BudgetExhaustedBehavior `json:"behavior"`
		}
		return json.Marshal(planExecuteFlat{s.Kind, c.Plan, c.Execute, c.PlanModel, c.Behavior})
	case StrategySelfVerifying:
		if s.SelfVerify == nil {
			return nil, fmt.Errorf("LoopStrategy: self_verifying requires config")
		}
		c := s.SelfVerify
		// Key order: kind, inner, evaluator, behavior (LAST).
		type selfVerifyFlat struct {
			Kind      LoopStrategyKind        `json:"kind"`
			Inner     *LoopStrategy           `json:"inner"`
			Evaluator SchemaRef               `json:"evaluator"`
			Behavior  BudgetExhaustedBehavior `json:"behavior"`
		}
		return json.Marshal(selfVerifyFlat{s.Kind, c.Inner, c.Evaluator, c.Behavior})
	case StrategyRalph:
		if s.Ralph == nil {
			return nil, fmt.Errorf("LoopStrategy: ralph requires config")
		}
		c := s.Ralph
		// Key order: kind, inner, agent, behavior (LAST).
		type ralphFlat struct {
			Kind     LoopStrategyKind        `json:"kind"`
			Inner    *LoopStrategy           `json:"inner"`
			Agent    AgentRef                `json:"agent"`
			Behavior BudgetExhaustedBehavior `json:"behavior"`
		}
		return json.Marshal(ralphFlat{s.Kind, c.Inner, c.Agent, c.Behavior})
	case StrategyHillClimbing:
		if s.HillClimbing == nil {
			return nil, fmt.Errorf("LoopStrategy: hill_climbing requires config")
		}
		c := s.HillClimbing
		// Key order: kind, inner, direction, max_stagnation,
		// revert_on_no_improvement, min_improvement_delta, evaluator, behavior
		// (LAST).
		type hillFlat struct {
			Kind                  LoopStrategyKind        `json:"kind"`
			Inner                 *LoopStrategy           `json:"inner"`
			Direction             OptimizationDirection   `json:"direction"`
			MaxStagnation         uint32                  `json:"max_stagnation"`
			RevertOnNoImprovement bool                    `json:"revert_on_no_improvement"`
			MinImprovementDelta   float64                 `json:"min_improvement_delta"`
			Evaluator             AgentRef                `json:"evaluator"`
			Behavior              BudgetExhaustedBehavior `json:"behavior"`
		}
		return json.Marshal(hillFlat{
			s.Kind, c.Inner, c.Direction, c.MaxStagnation,
			c.RevertOnNoImprovement, c.MinImprovementDelta, c.Evaluator, c.Behavior,
		})
	default:
		return nil, fmt.Errorf("LoopStrategy: unknown kind %q", s.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form, recursively for the combinators.
func (s *LoopStrategy) UnmarshalJSON(data []byte) error {
	var kindProbe struct {
		Kind LoopStrategyKind `json:"kind"`
	}
	if err := json.Unmarshal(data, &kindProbe); err != nil {
		return err
	}
	*s = LoopStrategy{Kind: kindProbe.Kind}
	switch kindProbe.Kind {
	case StrategyReAct:
		var probe struct {
			Budget   BudgetPolicy             `json:"budget"`
			Behavior *BudgetExhaustedBehavior `json:"behavior"`
			Agent    AgentRef                 `json:"agent"`
			Toolset  ToolsetRef               `json:"toolset"`
			Output   *SchemaRef               `json:"output"`
		}
		if err := json.Unmarshal(data, &probe); err != nil {
			return err
		}
		s.ReActCfg = &ReactConfig{
			Budget:   probe.Budget,
			Behavior: behaviorOrDefault(probe.Behavior),
			Agent:    probe.Agent,
			Toolset:  probe.Toolset,
			Output:   probe.Output,
		}
		return nil
	case StrategyPlanExecute:
		var probe struct {
			Plan      *LoopStrategy            `json:"plan"`
			Execute   *LoopStrategy            `json:"execute"`
			PlanModel *ModelConfig             `json:"plan_model"`
			Behavior  *BudgetExhaustedBehavior `json:"behavior"`
		}
		if err := json.Unmarshal(data, &probe); err != nil {
			return err
		}
		s.PlanExecute = &PlanExecuteConfig{
			Plan:      probe.Plan,
			Execute:   probe.Execute,
			PlanModel: probe.PlanModel,
			Behavior:  behaviorOrDefault(probe.Behavior),
		}
		return nil
	case StrategySelfVerifying:
		var probe struct {
			Inner     *LoopStrategy            `json:"inner"`
			Evaluator SchemaRef                `json:"evaluator"`
			Behavior  *BudgetExhaustedBehavior `json:"behavior"`
		}
		if err := json.Unmarshal(data, &probe); err != nil {
			return err
		}
		s.SelfVerify = &SelfVerifyingConfig{
			Inner:     probe.Inner,
			Evaluator: probe.Evaluator,
			Behavior:  behaviorOrDefault(probe.Behavior),
		}
		return nil
	case StrategyRalph:
		var probe struct {
			Inner    *LoopStrategy            `json:"inner"`
			Agent    AgentRef                 `json:"agent"`
			Behavior *BudgetExhaustedBehavior `json:"behavior"`
		}
		if err := json.Unmarshal(data, &probe); err != nil {
			return err
		}
		s.Ralph = &RalphConfig{
			Inner:    probe.Inner,
			Agent:    probe.Agent,
			Behavior: behaviorOrDefault(probe.Behavior),
		}
		return nil
	case StrategyHillClimbing:
		var probe struct {
			Inner                 *LoopStrategy            `json:"inner"`
			Direction             OptimizationDirection    `json:"direction"`
			MaxStagnation         uint32                   `json:"max_stagnation"`
			RevertOnNoImprovement bool                     `json:"revert_on_no_improvement"`
			MinImprovementDelta   float64                  `json:"min_improvement_delta"`
			Evaluator             AgentRef                 `json:"evaluator"`
			Behavior              *BudgetExhaustedBehavior `json:"behavior"`
		}
		if err := json.Unmarshal(data, &probe); err != nil {
			return err
		}
		s.HillClimbing = &HillClimbingConfig{
			Inner:                 probe.Inner,
			Direction:             probe.Direction,
			MaxStagnation:         probe.MaxStagnation,
			RevertOnNoImprovement: probe.RevertOnNoImprovement,
			MinImprovementDelta:   probe.MinImprovementDelta,
			Evaluator:             probe.Evaluator,
			Behavior:              behaviorOrDefault(probe.Behavior),
		}
		return nil
	default:
		return fmt.Errorf("LoopStrategy: unknown kind %q", kindProbe.Kind)
	}
}

// ============================================================================
// StrategyRef — serializable strategy identity
// ============================================================================

// StrategyRefKind discriminates StrategyRef variants.
type StrategyRefKind string

const (
	// StrategyRefBuiltIn carries a closed built-in LoopStrategy tree.
	StrategyRefBuiltIn StrategyRefKind = "built_in"
	// StrategyRefCustom carries an opaque string key resolved at runtime (#120).
	StrategyRefCustom StrategyRefKind = "custom"
)

// StrategyRef is the serializable identity of a strategy: either a closed
// built-in LoopStrategy tree or an opaque Custom string key resolved at runtime
// (registry: #120). It is adjacently tagged on "kind"/"value" to avoid a tag
// collision with the nested LoopStrategy's own "kind":
//
//	{"kind":"built_in","value":{"kind":"react",...}}
//	{"kind":"custom","value":"my-harness::DoubleVerify"}
type StrategyRef struct {
	Kind    StrategyRefKind
	BuiltIn *LoopStrategy
	Custom  string
}

// MarshalJSON serializes StrategyRef adjacently tagged on kind/value.
func (r StrategyRef) MarshalJSON() ([]byte, error) {
	switch r.Kind {
	case StrategyRefBuiltIn:
		if r.BuiltIn == nil {
			return nil, fmt.Errorf("StrategyRef: built_in requires value")
		}
		return json.Marshal(struct {
			Kind  StrategyRefKind `json:"kind"`
			Value *LoopStrategy   `json:"value"`
		}{r.Kind, r.BuiltIn})
	case StrategyRefCustom:
		return json.Marshal(struct {
			Kind  StrategyRefKind `json:"kind"`
			Value string          `json:"value"`
		}{r.Kind, r.Custom})
	default:
		return nil, fmt.Errorf("StrategyRef: unknown kind %q", r.Kind)
	}
}

// UnmarshalJSON decodes the adjacently-tagged form.
func (r *StrategyRef) UnmarshalJSON(data []byte) error {
	var kindProbe struct {
		Kind StrategyRefKind `json:"kind"`
	}
	if err := json.Unmarshal(data, &kindProbe); err != nil {
		return err
	}
	*r = StrategyRef{Kind: kindProbe.Kind}
	switch kindProbe.Kind {
	case StrategyRefBuiltIn:
		var probe struct {
			Value *LoopStrategy `json:"value"`
		}
		if err := json.Unmarshal(data, &probe); err != nil {
			return err
		}
		r.BuiltIn = probe.Value
		return nil
	case StrategyRefCustom:
		var probe struct {
			Value string `json:"value"`
		}
		if err := json.Unmarshal(data, &probe); err != nil {
			return err
		}
		r.Custom = probe.Value
		return nil
	default:
		return fmt.Errorf("StrategyRef: unknown kind %q", kindProbe.Kind)
	}
}

// ============================================================================
// MaxSteps — advisory worst-case turn bound (#122)
// ============================================================================

// MaxSteps returns an advisory worst-case TURN count for a fully-bounded
// strategy tree, computed BEFORE a run. It is a pre-run advisory figure logged
// at startup, NOT an enforcement mechanism — the per-node budget ceiling is the
// safety mechanism. The returned bool is false ("no finite advisory bound")
// when the tree is not fully bounded; this mirrors BudgetPolicy.AllowanceValue's
// (value, ok) optional idiom.
//
// The bound is derived multiplicatively/additively down the tree and is
// option-monadic: any Unlimited node anywhere collapses the whole figure to
// (0, false). It is a runtime-only computation and is NEVER serialized.
//
// Per-variant rules:
//   - ReAct(c)         ⇒ c.Budget.AllowanceValue() (Unlimited ⇒ (0, false)).
//   - SelfVerifying(c) ⇒ inner + 1 — the single read-only evaluator turn is
//     exactly one extra turn.
//   - PlanExecute(c)   ⇒ plan + execute. PER-TASK bound: a full run's total is
//     plan + executePerTask × taskCount, where taskCount is data-dependent (the
//     plan phase builds the task graph at runtime), so it is intentionally NOT
//     part of this static figure.
//   - Ralph(c)         ⇒ inner — PER-WINDOW bound, mirroring PlanExecute's
//     per-task treatment. A full run's total is perWindow × maxWindows, where
//     maxWindows derives from HarnessConfig max resets (default 3) at runtime
//     and is intentionally NOT part of this static figure.
//   - HillClimbing(c)  ⇒ inner × (MaxStagnation + 1) — the +1 is the one
//     productive pass; MaxStagnation non-improving passes follow. The
//     MaxStagnation == math.MaxUint32 sentinel means unbounded windows ⇒
//     (0, false) (a semantic unbounded rule, distinct from arithmetic overflow).
//
// Arithmetic guards add/mul against uint32 overflow; an unrepresentable bound
// (overflow) yields (0, false) — an unrepresentable figure is "no finite
// advisory bound".
func (s LoopStrategy) MaxSteps() (uint32, bool) {
	switch s.Kind {
	case StrategyReAct:
		if s.ReActCfg == nil {
			return 0, false
		}
		return s.ReActCfg.Budget.AllowanceValue()
	case StrategySelfVerifying:
		if s.SelfVerify == nil || s.SelfVerify.Inner == nil {
			return 0, false
		}
		inner, ok := s.SelfVerify.Inner.MaxSteps()
		if !ok {
			return 0, false
		}
		return checkedAddU32(inner, 1)
	case StrategyPlanExecute:
		if s.PlanExecute == nil || s.PlanExecute.Plan == nil || s.PlanExecute.Execute == nil {
			return 0, false
		}
		plan, ok := s.PlanExecute.Plan.MaxSteps()
		if !ok {
			return 0, false
		}
		exec, ok := s.PlanExecute.Execute.MaxSteps()
		if !ok {
			return 0, false
		}
		return checkedAddU32(plan, exec)
	case StrategyRalph:
		if s.Ralph == nil || s.Ralph.Inner == nil {
			return 0, false
		}
		return s.Ralph.Inner.MaxSteps()
	case StrategyHillClimbing:
		if s.HillClimbing == nil || s.HillClimbing.Inner == nil {
			return 0, false
		}
		if s.HillClimbing.MaxStagnation == ^uint32(0) {
			// Unbounded-windows sentinel ⇒ no finite advisory bound.
			return 0, false
		}
		inner, ok := s.HillClimbing.Inner.MaxSteps()
		if !ok {
			return 0, false
		}
		windows, ok := checkedAddU32(s.HillClimbing.MaxStagnation, 1)
		if !ok {
			return 0, false
		}
		return checkedMulU32(inner, windows)
	default:
		return 0, false
	}
}

// MaxSteps is the advisory worst-case turn bound for this strategy reference
// (#122). Custom is opaque to the framework (it cannot be introspected), so it
// yields (0, false); BuiltIn delegates to LoopStrategy.MaxSteps.
func (r StrategyRef) MaxSteps() (uint32, bool) {
	switch r.Kind {
	case StrategyRefBuiltIn:
		if r.BuiltIn == nil {
			return 0, false
		}
		return r.BuiltIn.MaxSteps()
	default:
		// Custom (and any unknown kind): opaque ⇒ no finite advisory bound.
		return 0, false
	}
}

// checkedAddU32 returns a+b and true, or (0, false) on uint32 overflow.
func checkedAddU32(a, b uint32) (uint32, bool) {
	sum := a + b
	if sum < a {
		return 0, false
	}
	return sum, true
}

// checkedMulU32 returns a*b and true, or (0, false) on uint32 overflow.
func checkedMulU32(a, b uint32) (uint32, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	prod := a * b
	if prod/b != a {
		return 0, false
	}
	return prod, true
}

// ============================================================================
// RunStrategy — the runtime composition seam
// ============================================================================

// RunStrategy is the runtime composition seam: every strategy node knows how to
// run itself given an *ExecutionContext. Implemented on LoopStrategy as the ONLY
// dispatch site (one switch, one-line delegation per arm) and on each *Config
// with a REAL per-variant loop (#124).
//
// Each *Config.Run OWNS its loop completely: a combinator recurses via
// cx.runChild(ctx, self.inner) (or self.plan / self.execute), and the leaf
// (*ReactConfig).Run drives one bounded ReAct window through the
// StrategyExecutor primitive on the context. The harness entry collapses to
// task.LoopStrategy.Run(ctx, cx) — there is NO central dispatch switch anymore;
// the only switch left in the system is the enum→config delegation below.
//
// Without a wired StrategyExecutor (the scaffold-only contexts that exercise the
// runtime context with no real harness) every body returns a TYPED Failed
// outcome — never a panic.
//
// The full ExecutionContext / StrategyOutcome / BudgetContext / BudgetStack /
// SpanStack runtime scaffold these methods thread is defined in
// execution_scaffold.go (#123); the StrategyExecutor seam and RunScratch are
// in executor.go (#124).
//
// Cross-language note: Rust uses trait/dyn dispatch; Go uses this interface as
// its runtime-polymorphism idiom, and threads context.Context as the first arg
// (Go CONVENTIONS — never store a Context in a struct). The serialized
// LoopStrategy / StrategyRef stay byte-identical across languages.
type RunStrategy interface {
	Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome
}

// Run dispatches to the active config's Run. This is the single switch site in
// the system (mirrors the Rust enum-delegation match). AC1: no central dispatch
// switch remains — each config owns its loop.
func (s LoopStrategy) Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome {
	switch s.Kind {
	case StrategyReAct:
		return s.ReActCfg.Run(ctx, cx)
	case StrategyPlanExecute:
		return s.PlanExecute.Run(ctx, cx)
	case StrategySelfVerifying:
		return s.SelfVerify.Run(ctx, cx)
	case StrategyRalph:
		return s.Ralph.Run(ctx, cx)
	case StrategyHillClimbing:
		return s.HillClimbing.Run(ctx, cx)
	default:
		return StrategyFailed(&InvalidConfigurationError{
			Message: fmt.Sprintf("unknown loop strategy kind %q", s.Kind),
		})
	}
}

// Run is the leaf: one bounded ReAct turn-loop window. It reads the per-run
// scratch (task / session / budget) and drives a single ReAct window through the
// executor primitive, recording the terminal back into the shared context.
func (c *ReactConfig) Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome {
	executor, fail := cx.executor()
	if executor == nil {
		return fail
	}
	task := cx.currentTask()
	// #124: the leaf resolves its worker agent from the registry (the leaf no
	// longer reads a default config agent). An unresolved handle is a typed
	// failure recorded verbatim.
	agent, agentFail := executor.ResolveWorkerAgent(&task.LoopStrategy)
	if agentFail != nil {
		_ = cx.takeSession()
		_ = cx.takeStream()
		return cx.recordTerminal(*agentFail)
	}
	session := cx.takeSession()
	budget := cx.Scratch.RunBudget

	// Output-schema delivery + enforcement (issue #139). MIGRATION GATE: only when
	// EnforceOutputSchemas is ON AND this leaf carries Output != nil do we resolve
	// the schema, DELIVER it (directive seed + the constrained-decoding channel),
	// and pass it (plus the retry budget N) into the window for terminal
	// validation. When the gate is OFF or there is no Output, outputSchema is nil
	// and the window behaves byte-identically to pre-#139 (no resolve, no delivery,
	// no validation). The schema is canonicalized to compact key-sorted JSON so its
	// delivered/reported bytes are identical across the four language ports.
	var outputSchema json.RawMessage
	if executor.EnforceOutputSchemas() && c.Output != nil {
		if resolved, ok := cx.Registry.ResolveSchema(*c.Output); ok {
			outputSchema = resolved
		}
	}
	outputSchemaMaxRetries := executor.OutputSchemaMaxRetries()
	if outputSchema != nil {
		// AC1: append the resolved schema to the leaf's directive/system context as a
		// USER message, key-sorted via canonicalizeSchema so the seeded bytes are
		// identical across languages.
		directive := "Your final response must be a JSON value that conforms to this " +
			"JSON schema: " + canonicalizeSchema(outputSchema)
		executor.SeedUserMessage(ctx, &session, directive)
	}

	// #125/#129: push this leaf's OWN budget scope carrying its CONFIGURED
	// Behavior. The leaf still never RESOLVES it in the nested case (rule 6: it
	// PROPAGATES a BudgetExhausted to its parent, which owns the single recovery
	// site). Carrying the real Behavior only means the propagated error reports
	// it, so the TOP-LEVEL/bare-leaf resolution site (driveStrategy) can honor it
	// (Q1 — a bare leaf self-resolves, a nested leaf does not).
	cx.PushBudget(c.Budget, c.Behavior, "react")
	// The leaf takes the run's stream sink for the window; combinators that
	// recurse per-phase suppress it (they take it before recursing).
	onStream := cx.takeStream()
	// Issue 2: thread THIS leaf's toolset handle down so the window dispatches the
	// per-node scoped catalogue (empty handle ⇒ global-catalogue fallback).
	// Mirrors agent resolution.
	// Issue #139: thread the resolved output schema (or nil) and the retry budget
	// so the window validates the terminal.
	result := executor.ReactWindow(ctx, task, c.MaxIterations(), session, budget, onStream, agent, c.Toolset, outputSchema, outputSchemaMaxRetries)
	executor.Finalize(ctx, result)

	// #125: charge the window's turns against this leaf's OWN scope. The leaf
	// POLICY (c.Budget) — not the global BudgetLimits backstop — is the per-node
	// enforcement point. When the LEAF cap is the binding constraint (the window
	// consumed >= the leaf policy value) the leaf is exhausted and PROPAGATES a
	// typed BudgetExhausted to its parent (rule 6 — the leaf never self-resolves).
	// When the smaller GLOBAL backstop trips first, the legacy terminal is recorded
	// VERBATIM (the global cap is unchanged, #117 backstop).
	var windowTurns uint32
	if result.Kind == RunSuccess || result.Kind == RunFailure {
		windowTurns = result.Turns
	}

	// #137: the window hit the consecutive-tool-error breaker's 2N hard stop.
	// PROPAGATE it through the SAME single budget-exhaustion resolution site (so
	// the node's Behavior governs Fail/Escalate/Continue), but carry the
	// ToolErrorLoop cause so the terminal reports HaltToolErrorLoop, never
	// HaltBudgetExceeded. The window's turns are still CHARGED against the leaf
	// scope (accurate accounting), but the breaker stopped EARLY — the remaining
	// budget is NOT burned. Detection is independent of leafCapBinding.
	if result.Kind == RunFailure && result.Reason.Kind == HaltToolErrorLoop {
		tool := result.Reason.Tool
		consecutiveErrors := result.Reason.ConsecutiveErrors
		// Carry the post-run session so a parent resumes losslessly.
		cx.Scratch.RunSession = result.SessionState
		_ = cx.ChargeCurrent(windowTurns)
		cx.PopBudget()
		partial := reactPartialJSON(lastFinalResponseText(result))
		return promoteToolErrorLoop(c.Budget, c.Behavior, windowTurns, tool, consecutiveErrors, &partial)
	}
	windowHitBudget := result.Kind == RunFailure && result.Reason.Kind == HaltBudgetExceeded
	leafCapBinding := false
	if windowHitBudget {
		if cap, capped := c.Budget.AllowanceValue(); capped && windowTurns >= cap {
			leafCapBinding = true
		}
	}
	charge := cx.ChargeCurrent(windowTurns)
	if leafCapBinding || charge != nil {
		// Carry the post-run session so a parent resumes losslessly.
		if result.Kind == RunSuccess || result.Kind == RunFailure {
			cx.Scratch.RunSession = result.SessionState
		}
		err := charge
		if err == nil {
			// The window itself hit the cap; synthesize the charge error from the
			// current scope state.
			err = cx.currentExhausted()
			if err == nil {
				err = &BudgetExhausted{
					Policy:     c.Budget,
					Behavior:   c.Behavior,
					StepsTaken: windowTurns,
					Phase:      "react",
				}
			}
		}
		cx.PopBudget()
		// Rule 6: the leaf PROPAGATES — partial carries the last FinalResponse
		// (Escalate semantics, fork #1/#2).
		partial := reactPartialJSON(lastFinalResponseText(result))
		return promoteBudgetExhausted(err, &partial)
	}
	cx.PopBudget()
	return cx.recordTerminal(result)
}

// Run is the plan→execute combinator (#124). GENUINELY recursive: the plan phase
// dispatches c.Plan.Run (seeding the planning directive + a one-turn budget on
// the scratch first) and the execute phase dispatches c.Execute.Run ONCE PER
// TASK. The child strategy's full loop runs for each phase — a non-ReAct execute
// child (SelfVerifying / HillClimbing) really executes per task, not a hardcoded
// flat ReAct (the defeated-the-point bug this fixes).
//
// This config body OWNS the orchestration: per-task turn/budget allocation (Q1),
// the OnTaskAdvance hook (pre, mutable), seeding each step instruction as a user
// message, A.6 deep-resume against the durable RunStore checkpoint, task-list
// persistence after each transition (Q4), and cumulative usage / last-output /
// last-state carry. The harness keeps only LEAF primitives: the constrained-plan
// capture/persist machinery, the deep-resume reconcile, and the OnTaskAdvance
// fire — none of which touch the per-task model loop. The ready-set walk lands in
// #126 (execute runs per task sequentially for now).
func (c *PlanExecuteConfig) Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome {
	executor, fail := cx.executor()
	if executor == nil {
		return fail
	}
	task := cx.currentTask()
	sessionID := task.SessionID
	// The incoming shared execute session ( [user: task.instruction] ).
	baseSession := cx.takeSession()
	budget := cx.Scratch.RunBudget
	// PlanExecute suppresses the run's stream sink for its phases (parent-visible
	// step boundaries are re-emitted on the caller's sink). Take it now and keep
	// it OUT of cx.Stream so the recursive children run with a suppressed sink.
	onStream := cx.takeStream()

	// #131/#138: the phase-agnostic resume seed (the stalled worker conversation
	// carried from a resumed consult or budget pause). Taken BEFORE the plan phase
	// so AC3 (plan-phase exhaustion) can seed the plan session from it, and AC2
	// (execute-phase exhaustion) can re-attach it to the single InProgress task.
	resumeSeedSession := cx.Scratch.ResumeSeed
	cx.Scratch.ResumeSeed = nil

	// #138 AC1 (skip re-planning): probe the DURABLE task_list BEFORE the plan
	// phase. If a non-empty list already exists (e.g. a prior window authored it,
	// #142 makes it survive Ralph's per-window session reset), the plan phase is
	// REDUNDANT — go straight to reconcile + the ready-set walk. This is the core
	// fix: a budget/consult re-entry no longer burns its grant re-planning a graph
	// that is already durable. The plan artifact under PlanExecuteExtrasKey may be
	// ABSENT in this case (AC1-a artifact-optional) — nothing downstream requires
	// it once a task_list exists.
	preexisting, havePreexisting := executor.LoadTaskList(ctx, sessionID)
	skipPlan := havePreexisting && len(preexisting.Tasks) > 0

	var taskList TaskList
	var totalUsage AggregateUsage
	var carried BudgetSnapshot
	if skipPlan {
		// ── AC1: skip-plan path. The list is the durable source of truth.
		taskList = preexisting

		// AC5 (defense in depth): re-check the durable graph for cycles. No task
		// runs.
		if taskList.HasCycle() {
			result := RunResult{
				Kind: RunFailure,
				Reason: HaltReason{
					Kind:   HaltTaskGraphCycle,
					Reason: "persisted task graph contains a directed cycle",
				},
				SessionID: sessionID,
			}
			return cx.finish(ctx, executor, task, result)
		}

		// A.6 deep-resume: reconcile already-Completed tasks so they are NOT
		// re-run (handles the dedup the plan path would have done).
		executor.ReconcileCompletedTasks(ctx, sessionID, &taskList)
		executor.PersistTaskList(ctx, sessionID, taskList)

		// No plan turn ran: usage starts empty and the shared budget is carried
		// unchanged (turns stay at the incoming budget.Turns).
		totalUsage = AggregateUsage{}
		carried = budget
	} else {
		// ── AC3 / fresh-plan path: run the plan phase. ──────────────────────────
		//
		// Seed the planning directive onto a CLONE of the base session so the shared
		// execute context stays [user: task.instruction] (#93 — a leaked directive
		// would make every execute step re-emit a plan). The plan sub-strategy's own
		// declared budget governs the plan phase (R1): an authored task_list may take
		// more than one turn. The task's global turn ceiling remains the outer
		// backstop (R10), enforced by the shared budget — not a turns+1 clamp.
		//
		// #138 AC3 (plan-phase exhaustion, infer-from-task_list option iii): when a
		// resume seed is present AND the durable list has no InProgress task, the
		// exhaustion happened in the PLAN phase before authoring any task (AC3-b: the
		// list is still empty). Seed the PLAN session from the carried conversation
		// so the planner CONTINUES on it instead of starting fresh — and consume the
		// seed here so the execute-phase re-attach below does not also fire.
		planResume := resumeSeedSession != nil && !preexistingHasInProgress(preexisting)
		directive := executor.PlanDirective(task.Instruction)
		var planSession SessionState
		if planResume {
			planSession = *resumeSeedSession
			resumeSeedSession = nil
		} else {
			planSession = baseSession
		}
		executor.SeedUserMessage(ctx, &planSession, directive)
		planBudget := task.Budget
		planTask := Task{
			ID:           task.ID,
			Instruction:  directive,
			SessionID:    sessionID,
			Budget:       planBudget,
			LoopStrategy: *c.Plan,
		}
		planResult := executor.RunPlanSubtree(ctx, c.Plan, planTask, planSession, budget)
		if planResult == nil {
			result := RunResult{
				Kind: RunFailure,
				Reason: HaltReason{
					Kind: HaltPlanPhaseFailed,
					PlanError: &PlanPhaseError{
						Kind:    PlanErrorPlanningTurnFailed,
						Message: "plan sub-strategy produced no terminal",
					},
				},
				SessionID: sessionID,
				Turns:     budget.Turns,
			}
			return cx.finish(ctx, executor, task, result)
		}
		if planResult.Kind != RunSuccess {
			// A non-success plan terminal (budget / agent error / pause) propagates
			// verbatim — the run never reaches execute.
			return cx.finish(ctx, executor, task, *planResult)
		}
		planOutput := planResult.Output
		planUsageAgg := planResult.Usage
		planTurns := planResult.Turns

		// Capture + persist the artifact from the plan child's output (R3/R4/R11) —
		// the harness-side machinery, no model turn.
		outcome, failure := executor.CapturePlanArtifact(ctx, sessionID, planOutput, planUsageAgg, planTurns)
		if failure != nil {
			return cx.finish(ctx, executor, task, *failure)
		}

		// #126 decision C: the runnable task list comes from the persisted task_list
		// tool store (the ONE authoring path — it can carry real blockers). Fall back
		// to the linear plan-artifact bridge only when nothing was authored via the
		// tool (back-compat with the #59/#124 plan-only path and its replay fixtures).
		if persisted, ok := executor.LoadTaskList(ctx, sessionID); ok && len(persisted.Tasks) > 0 {
			taskList = persisted
		} else {
			taskList = PlanArtifactToTaskList(outcome.Artifact)
		}
		if len(taskList.Tasks) == 0 {
			result := RunResult{
				Kind:      RunFailure,
				Reason:    HaltReason{Kind: HaltEmptyPlan},
				SessionID: sessionID,
				Usage:     outcome.Usage,
				Turns:     outcome.Turns,
			}
			return cx.finish(ctx, executor, task, result)
		}

		// #126 AC5: re-check the WHOLE graph for cycles at execute entry (defense in
		// depth — add_task already rejects cycles, but the persisted store could be
		// cyclic out of band). No task runs.
		if taskList.HasCycle() {
			result := RunResult{
				Kind: RunFailure,
				Reason: HaltReason{
					Kind:   HaltTaskGraphCycle,
					Reason: "persisted task graph contains a directed cycle",
				},
				SessionID: sessionID,
				Usage:     outcome.Usage,
				Turns:     outcome.Turns,
			}
			return cx.finish(ctx, executor, task, result)
		}
		executor.PersistTaskList(ctx, sessionID, taskList)

		// Carry the shared budget past the plan turn.
		carried = budget
		carried.Turns = outcome.Turns
		carried.InputTokens += outcome.Usage.InputTokens
		carried.OutputTokens += outcome.Usage.OutputTokens

		// A.6 deep-resume (Q2): reconcile against the durable checkpoint so
		// already-Completed tasks are not re-run.
		executor.ReconcileCompletedTasks(ctx, sessionID, &taskList)

		totalUsage = outcome.Usage
	}

	// ── Phase 2: execute (dispatch c.Execute PER TASK). ─────────────────────
	//
	// The shared execute context starts from baseSession (NOT the plan child's
	// polluted session) so the directive never leaks (#93).
	result, exhausted := c.runExecuteLoop(ctx, cx, executor, &task, baseSession, taskList, carried, totalUsage, resumeSeedSession, onStream)
	if exhausted != nil {
		// #125: a BudgetExhausted is surfaced as a typed StrategyOutcome (NOT
		// collapsed through finish into a Failure override). Restore the parent task
		// and ensure no stale terminal override masks it on ascent.
		pt := task
		cx.Scratch.Task = &pt
		cx.Scratch.TerminalOverride = nil
		cx.Stream = onStream
		return *exhausted
	}
	return cx.finish(ctx, executor, task, result)
}

// runExecuteLoop drains taskList by dispatching c.Execute.Run ONCE PER TASK
// (#124). It owns the per-task orchestration: Q1 per-task turn allocation, the
// OnTaskAdvance hook, seeding each step instruction as a user message on the
// SHARED execute context, task-list persistence after each transition (Q4), and
// cumulative usage / last-output / last-state carry. The A.6 deep-resume
// reconcile already ran in Run before this call. Returns the terminal RunResult
// for the execute phase.
//
// #131 consult / #138 budget re-drive: resumeSeedSession is the stalled worker
// conversation carried from a resumed pause (nil on a fresh run, or when the AC3
// plan-resume path already consumed it). The stalled task is the single
// InProgress task in the durable list; this resets it to Pending so NextReady
// re-schedules it and seeds its step from the carried session instead of a fresh
// one — the worker continues mid-loop and its evaluator still runs.
func (c *PlanExecuteConfig) runExecuteLoop(
	ctx context.Context,
	cx *ExecutionContext,
	executor StrategyExecutor,
	task *Task,
	session SessionState,
	taskList TaskList,
	carried BudgetSnapshot,
	planUsage AggregateUsage,
	resumeSeedSession *SessionState,
	onStream StreamSink,
) (RunResult, *StrategyOutcome) {
	sessionID := task.SessionID

	// #131 consult / #138 budget re-drive: the stalled task is the single
	// InProgress task in the durable list (PlanExecute marks a task InProgress
	// before running it). Reset it to Pending so NextReady re-schedules it, and
	// remember its id so its step uses the carried session instead of a fresh one.
	consultResumeSession := resumeSeedSession
	var consultResumeTask uint32
	var haveConsultResumeTask bool
	if consultResumeSession != nil {
		for i := range taskList.Tasks {
			if taskList.Tasks[i].Status == TaskStatusInProgress {
				// Reset directly (bypassing Update's forward-only transition guard,
				// which rejects InProgress->Pending): the resume legitimately
				// re-schedules the in-flight task.
				taskList.Tasks[i].Status = TaskStatusPending
				consultResumeTask = taskList.Tasks[i].ID
				haveConsultResumeTask = true
				break
			}
		}
		if !haveConsultResumeTask {
			// No in-progress task to resume (out-of-contract); drop the seed so the
			// walk proceeds normally rather than stalling.
			consultResumeSession = nil
		}
	}

	totalTasks := len(taskList.Tasks)
	totalUsage := planUsage
	var lastOutput string
	var lastState SessionState

	// #126 Tier-1/Tier-2 + cascade run-local state (decision D/E):
	//   - finalOutputs: each completed task's Success.Output (Tier-1).
	//   - ledger: the Tier-2 global running ledger (bounded, drop-oldest).
	//   - ledgerElided: sticky flag set the first time entries are dropped.
	//   - blockedByFailure: tasks cascade-Blocked by a terminal failure (decision
	//     E — run scratch, NOT a TaskStatus variant).
	//   - firstFailure*: the first terminal failure that triggered a cascade (A).
	finalOutputs := make(map[uint32]string)
	var ledger []StepLedgerEntry
	ledgerElided := false
	blockedByFailure := make(map[uint32]struct{})
	var (
		firstFailureID     uint32
		firstFailureReason string
		hadFailure         bool
	)

	// #125: PlanExecute owns a budget scope for its execute phase. Its POLICY is
	// the task's global turn ceiling (TotalSteps). Behavior is Escalate (in-process
	// placeholder; the serialized behavior field is #129). Enforcement is
	// charge-based per node.
	var planPolicy BudgetPolicy
	if task.Budget.MaxTurns != nil {
		planPolicy = BudgetPolicy{Kind: BudgetTotalSteps, Value: *task.Budget.MaxTurns}
	} else {
		planPolicy = BudgetPolicy{Kind: BudgetUnlimited}
	}
	cx.PushBudget(planPolicy, c.Behavior, "plan_execute")

	// cascade marks taskID and its transitive dependents Blocked, records the
	// first failure, and persists (#126 AC3/AC4, decision A/E).
	cascade := func(taskID uint32, reason string) {
		blocked := TaskStatusBlocked
		_ = taskList.Update(taskID, &blocked, nil)
		for _, dep := range taskList.TransitiveDependents(taskID) {
			_ = taskList.Update(dep, &blocked, nil)
			blockedByFailure[dep] = struct{}{}
		}
		blockedByFailure[taskID] = struct{}{}
		executor.PersistTaskList(ctx, sessionID, taskList)
		if !hadFailure {
			firstFailureID = taskID
			firstFailureReason = reason
			hadFailure = true
		}
	}

	// ── Phase 2: ready-set DAG walk (#126). ─────────────────────────────────
	//
	// Repeatedly pick the lowest-id Pending task whose blockers are all Completed.
	// A task whose blocker FAILED is cascade-Blocked (so it is no longer Pending
	// and never becomes ready). When no Pending task is ready, the walk drains.
	for {
		taskID, ok := taskList.NextReady()
		if !ok {
			break
		}
		index := 0
		for i := range taskList.Tasks {
			if taskList.Tasks[i].ID == taskID {
				index = i
				break
			}
		}
		instruction := taskList.Tasks[index].Description

		// Mark InProgress and re-persist (Q4).
		ip := TaskStatusInProgress
		_ = taskList.Update(taskID, &ip, nil)
		executor.PersistTaskList(ctx, sessionID, taskList)

		// Fire OnTaskAdvance (pre, mutable). The hook may rewrite the step
		// instruction; the (possibly mutated) instruction seeds the execute
		// sub-strategy.
		stepTask := Task{
			ID:           task.ID,
			Instruction:  instruction,
			SessionID:    sessionID,
			Budget:       task.Budget,
			LoopStrategy: *c.Execute,
		}
		executor.FireTaskAdvance(ctx, sessionID, &stepTask, index, totalTasks)

		// #131 consult re-drive: if THIS is the task that was consulting, resume its
		// worker from the carried conversation (answer already injected) instead of a
		// fresh instruction-seeded session — the worker continues mid-loop and its
		// evaluator still runs. Otherwise build the normal fresh, Tier-1/Tier-2-seeded
		// step session.
		var stepSession *SessionState
		if haveConsultResumeTask && taskID == consultResumeTask && consultResumeSession != nil {
			stepSession = consultResumeSession
			consultResumeSession = nil
		} else {
			// #126 Tier-1 scoped context (decision D): seed this step from a FRESH copy
			// of the base session (NOT a forward-folded shared transcript — that breaks
			// on a DAG) plus, for THIS task's transitive blockers ONLY, their final
			// outputs + their ledger rows. Independent branches never appear (AC1).
			stepSession = cloneSessionState(&session)
			blockers := taskList.TransitiveBlockers(taskID)
			if len(blockers) > 0 {
				blockerSet := make(map[uint32]struct{}, len(blockers))
				for _, b := range blockers {
					blockerSet[b] = struct{}{}
				}
				// Tier-1: transitive blockers' final outputs (ascending id).
				var tier1Lines []string
				for _, b := range blockers {
					if out, ok := finalOutputs[b]; ok {
						tier1Lines = append(tier1Lines, fmt.Sprintf("#%d result: %s", b, out))
					}
				}
				if len(tier1Lines) > 0 {
					executor.SeedUserMessage(ctx, stepSession, "Results from upstream tasks:\n"+strings.Join(tier1Lines, "\n"))
				}
				// Tier-1 ledger: the Tier-2 ledger rows for this transitive set.
				var scoped []StepLedgerEntry
				for _, e := range ledger {
					if _, ok := blockerSet[e.TaskID]; ok {
						scoped = append(scoped, e)
					}
				}
				if block, ok := RenderStepLedger(scoped, false); ok {
					executor.SeedUserMessage(ctx, stepSession, block)
				}
			}

			// #126 Tier-2: inject the FULL global running ledger into EVERY step (with
			// the static elision marker once entries were dropped).
			if block, ok := RenderStepLedger(ledger, ledgerElided); ok {
				executor.SeedUserMessage(ctx, stepSession, block)
			}

			// Finally seed this step's own instruction.
			executor.SeedUserMessage(ctx, stepSession, stepTask.Instruction)
		}

		// #126 AC2: clear the observed-write accumulator so this task's
		// files_touched reflect ONLY the writes this step issues.
		executor.ClearObservedWrites()

		stCopy := stepTask
		cx.Scratch.Task = &stCopy
		cx.Scratch.RunSession = *stepSession
		cx.Scratch.RunBudget = carried
		// #125: absolute turn count BEFORE this step, so the success path can
		// charge only the DELTA against the PlanExecute scope.
		carriedBefore := carried.Turns
		stepOutcome := c.Execute.Run(ctx, cx)
		// #125 rule 4/7: a child's BudgetExhausted reaches THIS parent as a
		// StrategyOutcome. #126 AC4: a budget-Fail resolution cascades IDENTICALLY
		// to an error-failed one; an Escalate/Continue resolution surfaces the
		// partial and aborts the run.
		if stepOutcome.Kind == StrategyOutcomeBudgetExhausted {
			err := cx.currentExhausted()
			resolution := cx.ResolveCurrent()
			_ = cx.takeChildOverride()
			switch resolution {
			case ExhaustedResolutionContinue:
				// #129: a granted Continue loops IN-PROCESS — ResolveCurrent already
				// reset StepsTaken and bumped ContinuesUsed. Reset this task to
				// Pending and re-enter the ready-set walk so it runs again under the
				// refreshed scope allowance (NO serialization — AC3). MaxContinues
				// bounds the loop: once continues are spent, the chain falls through
				// to Fail/Escalate.
				pending := TaskStatusPending
				_ = taskList.Update(taskID, &pending, nil)
				executor.PersistTaskList(ctx, sessionID, taskList)
				continue
			case ExhaustedResolutionFail:
				cascade(taskID, "budget exhausted (Fail)")
				continue
			default: // Escalate: surface the partial and abort.
				// #138 AC2-b: under SurfaceToHuman the task is NOT permanently failed
				// — it pauses awaiting a budget grant. Leave it InProgress (the consult
				// path's invariant) so the resume's seed re-attaches via the SAME
				// InProgress->Pending->complete machinery. Under Autonomous the run
				// aborts, so the task is Blocked as before.
				surface := executor.EscalationMode().Kind == EscalationSurfaceToHuman
				newStatus := TaskStatusBlocked
				if surface {
					newStatus = TaskStatusInProgress
				}
				_ = taskList.Update(taskID, &newStatus, nil)
				executor.PersistTaskList(ctx, sessionID, taskList)
				partial := planExecutePartialJSON(taskList)
				// #138 AC2-a: carry the FULL stalled worker session (the execute child
				// left it in scratch) into the pause so a budget RESUME re-attaches it
				// as the in-progress task's seed — parallel to the consult pause —
				// instead of discarding it for a partial-only stub.
				workerSession := cx.takeSession()
				workerToolset := workerToolsetOf(c.Execute)
				cx.PopBudget()
				cx.Scratch.RunSession = session
				// #130: under SurfaceToHuman, PAUSE with a BudgetExhausted request
				// (combinator actions: [ContinueWithBudget, Skip, Fail]) instead of
				// propagating up. The waiting RunResult is returned as the terminal
				// result so Run finishes it as the verbatim override.
				if surface {
					waiting := promoteBudgetExhaustedToHuman(
						err, &partial, combinatorEscalationActions(err),
						sessionID, *task, carried, carried.Turns,
						workerSession, workerToolset,
					)
					return waiting, nil
				}
				outcome := promoteBudgetExhausted(err, &partial)
				return RunResult{}, &outcome
			}
		}
		subResult := cx.takeChildOverride()

		if subResult == nil {
			// No terminal from the child: treat as a terminal failure of this task
			// and cascade (same as a Failure).
			cascade(taskID, "execute sub-strategy produced no terminal")
			continue
		}

		switch subResult.Kind {
		case RunSuccess:
			carried.Turns = subResult.Turns
			lastState = subResult.SessionState
			carried.InputTokens += subResult.Usage.InputTokens
			carried.OutputTokens += subResult.Usage.OutputTokens
			totalUsage.InputTokens += subResult.Usage.InputTokens
			totalUsage.OutputTokens += subResult.Usage.OutputTokens
			totalUsage.CacheReadTokens += subResult.Usage.CacheReadTokens
			totalUsage.CacheWriteTokens += subResult.Usage.CacheWriteTokens
			totalUsage.CostUSD += subResult.Usage.CostUSD
			lastOutput = subResult.Output

			_ = taskList.Complete(taskID)
			executor.PersistTaskList(ctx, sessionID, taskList)

			// #126: record this task's final output (Tier-1) and append a ledger
			// entry whose files_touched is HARNESS-OBSERVED (AC2).
			finalOutputs[taskID] = subResult.Output
			var dropped bool
			ledger, dropped = PushStepLedger(ledger, StepLedgerEntry{
				TaskID:       taskID,
				Summary:      subResult.Output,
				FilesTouched: executor.TakeObservedWrites(),
			})
			if dropped {
				ledgerElided = true
			}

			emit(onStream, HarnessStreamEvent{Kind: HarnessStreamFinalResponse, Content: lastOutput})

			// #125: charge this step's turns against the PlanExecute scope. If the
			// global TotalSteps cap is now spent, the PlanExecute node surfaces its
			// OWN typed BudgetExhausted (partial = tasklist + ledger).
			if chErr := cx.ChargeCurrent(saturatingSubU32(subResult.Turns, carriedBefore)); chErr != nil {
				resolution := cx.ResolveCurrent()
				// #129: a granted Continue refreshes the scope and keeps scheduling
				// the remaining ready tasks IN-PROCESS (this step already completed).
				// NO serialization (AC3).
				if resolution == ExhaustedResolutionContinue {
					continue
				}
				partial := planExecutePartialJSON(taskList)
				cx.PopBudget()
				cx.Scratch.RunSession = lastState
				// #130: under SurfaceToHuman, an Escalate-resolving exhaustion PAUSES
				// with a BudgetExhausted request (combinator actions) instead of
				// propagating up.
				if resolution != ExhaustedResolutionFail &&
					executor.EscalationMode().Kind == EscalationSurfaceToHuman {
					// #138 AC2-a: this step already COMPLETED; carry its post-run
					// session (lastState) so a budget resume re-attaches the real worker
					// context, not a partial-only stub.
					waiting := promoteBudgetExhaustedToHuman(
						chErr, &partial, combinatorEscalationActions(chErr),
						sessionID, *task, carried, carried.Turns,
						lastState, workerToolsetOf(c.Execute),
					)
					return waiting, nil
				}
				var outcome StrategyOutcome
				switch resolution {
				case ExhaustedResolutionFail:
					outcome = promoteBudgetExhausted(chErr, nil)
				default:
					outcome = promoteBudgetExhausted(chErr, &partial)
				}
				return RunResult{}, &outcome
			}

		case RunFailure:
			// A GLOBAL turn-budget hard stop (#117 backstop) surfaces as a
			// BudgetExceeded Failure from the leaf. That is a WHOLE-RUN hard stop,
			// NOT a single-task terminal failure — it aborts the run verbatim
			// (preserving the pre-#126 mid-execute budget behavior). It is distinct
			// from a per-NODE BudgetExhausted resolving to Fail, which DOES cascade.
			totalUsage.InputTokens += subResult.Usage.InputTokens
			totalUsage.OutputTokens += subResult.Usage.OutputTokens
			totalUsage.CacheReadTokens += subResult.Usage.CacheReadTokens
			totalUsage.CacheWriteTokens += subResult.Usage.CacheWriteTokens
			totalUsage.CostUSD += subResult.Usage.CostUSD

			if subResult.Reason.Kind == HaltBudgetExceeded {
				blocked := TaskStatusBlocked
				_ = taskList.Update(taskID, &blocked, nil)
				executor.PersistTaskList(ctx, sessionID, taskList)
				cx.PopBudget()
				return RunResult{
					Kind:         RunFailure,
					Reason:       subResult.Reason,
					SessionID:    sessionID,
					Usage:        totalUsage,
					Turns:        subResult.Turns,
					SessionState: lastState,
				}, nil
			}

			// #126 AC3: a terminal task FAILURE cascade-blocks its transitive
			// dependents and KEEPS scheduling unrelated tasks (replaces the Q5
			// blanket abort).
			carried.Turns = subResult.Turns
			cascade(taskID, haltReasonString(subResult.Reason))
			continue

		default:
			// A pause / consult / escalate propagates the whole run verbatim.
			cx.PopBudget()
			return *subResult, nil
		}
	}

	cx.PopBudget()

	// ── Drain (#126, decision A). ───────────────────────────────────────────
	//
	// A run where a terminal failure cascade-blocked any task returns a PARTIAL
	// terminal Failure reporting the full partition. A run where every task
	// completed returns Success (output = last step's text).
	if hadFailure {
		var completed []uint32
		for i := range taskList.Tasks {
			if taskList.Tasks[i].Status == TaskStatusCompleted {
				completed = append(completed, taskList.Tasks[i].ID)
			}
		}
		sort.Slice(completed, func(i, j int) bool { return completed[i] < completed[j] })
		blocked := make([]uint32, 0, len(blockedByFailure))
		for id := range blockedByFailure {
			blocked = append(blocked, id)
		}
		sort.Slice(blocked, func(i, j int) bool { return blocked[i] < blocked[j] })
		return RunResult{
			Kind: RunFailure,
			Reason: HaltReason{
				Kind:       HaltTasksBlockedByFailure,
				Completed:  completed,
				Blocked:    blocked,
				FailedTask: firstFailureID,
				Reason:     firstFailureReason,
			},
			SessionID:    sessionID,
			Usage:        totalUsage,
			Turns:        carried.Turns,
			SessionState: lastState,
		}, nil
	}

	return RunResult{
		Kind:         RunSuccess,
		Output:       lastOutput,
		SessionID:    sessionID,
		Usage:        totalUsage,
		Turns:        carried.Turns,
		SessionState: lastState,
	}, nil
}

// Run is the SelfVerifying combinator (#124): GENUINELY recursive build↔evaluate
// loop. Each iteration dispatches c.Inner.Run(ctx, cx) for the build phase (a
// non-ReAct inner — e.g. PlanExecute — really runs its whole loop per
// iteration), then runs a fresh evaluate phase on the inner worker's resolved
// agent (Q1c) and consults the verifier resolved from c.Evaluator's key (Q1a).
// Passed => Success; Failed => append the reason (Default-FAIL) and loop;
// exhausted => SelfVerifyExhausted.
func (c *SelfVerifyingConfig) Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome {
	executor, fail := cx.executor()
	if executor == nil {
		return fail
	}
	task := cx.currentTask()
	buildSessionID := task.SessionID
	session := cx.takeSession()
	carried := cx.Scratch.RunBudget
	// Suppress the run's stream sink for the recursive child phases.
	onStream := cx.takeStream()

	// Q1a: resolve the verifier from c.Evaluator's key (NO wire change).
	verifier, ok := cx.Registry.ResolveVerifier(string(c.Evaluator))
	if !ok {
		result := RunResult{
			Kind: RunFailure,
			Reason: HaltReason{
				Kind: HaltSelfVerifyMisconfigured,
				Reason: fmt.Sprintf(
					"SelfVerifying requires a verifier registered under key %q", string(c.Evaluator),
				),
			},
			SessionID: buildSessionID,
		}
		cx.Stream = onStream
		return cx.finish(ctx, executor, task, result)
	}
	// Q1c: the evaluate-phase agent defaults to the inner worker's resolved agent.
	evalAgent, evalFail := executor.ResolveWorkerAgent(c.Inner)
	if evalFail != nil {
		cx.Stream = onStream
		return cx.finish(ctx, executor, task, *evalFail)
	}

	maxIterations := verifier.MaxIterations()
	var totalUsage AggregateUsage
	var lastReason string
	var lastWorkerOutput string

	// #125: SelfVerifying owns a budget scope for its build↔evaluate loop. POLICY
	// is the task's global turn ceiling (TotalSteps); behavior is Escalate
	// (in-process placeholder; serialized behavior is #129).
	var svPolicy BudgetPolicy
	if task.Budget.MaxTurns != nil {
		svPolicy = BudgetPolicy{Kind: BudgetTotalSteps, Value: *task.Budget.MaxTurns}
	} else {
		svPolicy = BudgetPolicy{Kind: BudgetUnlimited}
	}
	cx.PushBudget(svPolicy, c.Behavior, "self_verifying")

	for iteration := uint32(0); iteration < maxIterations; iteration++ {
		// ── Build phase: recurse c.Inner.Run.
		buildTask := Task{
			ID:           task.ID,
			Instruction:  task.Instruction,
			SessionID:    buildSessionID,
			Budget:       task.Budget,
			LoopStrategy: *c.Inner,
		}
		btCopy := buildTask
		cx.Scratch.Task = &btCopy
		cx.Scratch.RunSession = session
		cx.Scratch.RunBudget = carried
		carriedBefore := carried.Turns
		buildOutcome := c.Inner.Run(ctx, cx)
		// #125 rule 4/7: a child's BudgetExhausted reaches THIS parent as a
		// StrategyOutcome, never auto-cascaded. SelfVerifying surfaces its own typed
		// BudgetExhausted (partial = last worker result + last verdict) without
		// charging the child's exhaustion against its own scope.
		if buildOutcome.Kind == StrategyOutcomeBudgetExhausted {
			partial := selfVerifyingPartialJSON(lastWorkerOutput, lastReason)
			err := cx.currentExhausted()
			cx.PopBudget()
			_ = cx.takeChildOverride()
			pt := task
			cx.Scratch.Task = &pt
			cx.Scratch.TerminalOverride = nil
			cx.Stream = onStream
			return promoteBudgetExhausted(err, &partial)
		}
		bo := cx.takeChildOverride()
		var buildResult RunResult
		if bo == nil {
			buildResult = RunResult{
				Kind: RunFailure,
				Reason: HaltReason{
					Kind:   HaltSelfVerifyMisconfigured,
					Reason: "build sub-strategy produced no terminal",
				},
				SessionID:    buildSessionID,
				Turns:        carried.Turns,
				SessionState: session,
			}
		} else {
			buildResult = *bo
		}
		foldSelfVerifyUsage(&totalUsage, &carried, buildResult)

		// A paused / escalated build propagates verbatim.
		switch buildResult.Kind {
		case RunWaitingForHuman, RunConsult, RunEscalate:
			cx.PopBudget()
			cx.Stream = onStream
			return cx.finish(ctx, executor, task, buildResult)
		}
		// Capture the build's output for the partial (last worker result).
		if buildResult.Kind == RunSuccess {
			lastWorkerOutput = buildResult.Output
		}
		// Carry the build's post-run session forward for the next round.
		if buildResult.Kind == RunSuccess || buildResult.Kind == RunFailure {
			session = buildResult.SessionState
		}

		// #125: charge this iteration's build turns against the SelfVerifying scope.
		// If the global cap is spent, the node surfaces its OWN typed
		// BudgetExhausted (partial = last worker result + last verdict).
		if chErr := cx.ChargeCurrent(saturatingSubU32(carried.Turns, carriedBefore)); chErr != nil {
			resolution := cx.ResolveCurrent()
			// #129: a granted Continue resets the scope and RE-RUNS the
			// build↔evaluate iteration IN-PROCESS (do NOT pop the scope; the loop
			// continues under the refreshed allowance). NO serialization (AC3).
			// MaxContinues bounds the loop.
			if resolution == ExhaustedResolutionContinue {
				continue
			}
			partial := selfVerifyingPartialJSON(lastWorkerOutput, lastReason)
			cx.PopBudget()
			pt := task
			cx.Scratch.Task = &pt
			cx.Stream = onStream
			// #130: under SurfaceToHuman, an Escalate-resolving exhaustion PAUSES
			// with a BudgetExhausted request (combinator actions) recorded as the
			// verbatim terminal override instead of propagating up.
			if resolution != ExhaustedResolutionFail &&
				executor.EscalationMode().Kind == EscalationSurfaceToHuman {
				// #138 AC2-a: carry the FULL build (worker) session — the
				// SelfVerifying loop carries it forward in session after each
				// iteration — so a budget resume re-attaches the real worker context.
				waiting := promoteBudgetExhaustedToHuman(
					chErr, &partial, combinatorEscalationActions(chErr),
					buildSessionID, task, carried, carried.Turns,
					session, workerToolsetOf(c.Inner),
				)
				return cx.recordTerminal(waiting)
			}
			cx.Scratch.TerminalOverride = nil
			switch resolution {
			case ExhaustedResolutionFail:
				return promoteBudgetExhausted(chErr, nil)
			default:
				// #129: the in-process Continue is handled above; this is
				// Escalate-only now (the surface/propagate shape).
				return promoteBudgetExhausted(chErr, &partial)
			}
		}

		// ── Evaluate phase: a fresh evaluator run on evalAgent.
		evalResult := executor.EvaluatePhase(ctx, &task, evalAgent, &carried, &totalUsage)

		verdict := verifier.Verify(ctx, SelfVerifyInput{
			BuildResult: buildResult,
			EvalResult:  evalResult,
			Workspace:   executor.WorkspaceRoot(),
			Iteration:   iteration,
		})
		switch verdict.Kind {
		case SelfVerifyPassed:
			output := ""
			turns := carried.Turns
			finalState := session
			if buildResult.Kind == RunSuccess {
				output = buildResult.Output
				turns = buildResult.Turns
				finalState = buildResult.SessionState
			}
			result := RunResult{
				Kind:         RunSuccess,
				Output:       output,
				SessionID:    buildSessionID,
				Usage:        totalUsage,
				Turns:        turns,
				SessionState: finalState,
			}
			cx.PopBudget()
			cx.Stream = onStream
			return cx.finish(ctx, executor, task, result)
		default:
			// SelfVerifyFailed — Default-FAIL: inject the reason into the build
			// context so the next iteration sees it.
			lastReason = verdict.Reason
			executor.AppendUserMessage(ctx, &session, verdict.Reason)
		}
	}

	result := RunResult{
		Kind: RunFailure,
		Reason: HaltReason{
			Kind:       HaltSelfVerifyExhausted,
			Iterations: maxIterations,
			Reason:     lastReason,
		},
		SessionID:    buildSessionID,
		Usage:        totalUsage,
		Turns:        carried.Turns,
		SessionState: session,
	}
	cx.PopBudget()
	cx.Stream = onStream
	return cx.finish(ctx, executor, task, result)
}

// Run is the Ralph continuation wrapper (#124): GENUINELY recursive. Each
// context window seeds a FRESH session from the .spore/ checkpoint, then recurses
// c.Inner.Run(ctx, cx) (a non-ReAct inner — e.g. SelfVerifying — really runs its
// whole loop per window). Q3: when c.Agent is set it FILLS the inner leaf's
// agent per window only where that leaf's handle is empty (an explicit leaf
// agent stays authoritative); when unset the worker resolves via the inner leaf.
// RalphCompletionStatus drives the OUTER reset loop; exhaustion =>
// RalphCompletionUnmet. Ralph discards the incoming session state by design.
func (c *RalphConfig) Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome {
	executor, fail := cx.executor()
	if executor == nil {
		return fail
	}
	task := cx.currentTask()
	onStream := cx.takeStream()
	_ = cx.takeSession() // discarded: each window re-seeds from the checkpoint.
	maxResets := executor.RalphMaxResets()

	// Q3: when c.Agent is set, FILL the inner leaf's agent for every window only
	// where the worker leaf's handle is empty — an explicitly-declared leaf agent
	// is authoritative and is never shadowed by Ralph's per-window agent.
	innerForWindow := *c.Inner
	if string(c.Agent) != "" {
		cloned := *c.Inner
		fillEmptyWorkerAgent(&cloned, c.Agent)
		innerForWindow = cloned
	}

	var totalUsage AggregateUsage
	var cumulativeTurns uint32
	lastReason := ".spore/progress.json missing"
	lastSessionID := task.SessionID
	var lastSessionState SessionState

	for iteration := uint32(0); iteration < maxResets; iteration++ {
		windowSessionID := task.SessionID
		if iteration > 0 {
			windowSessionID = NewSessionID()
		}
		lastSessionID = windowSessionID

		// R2/R3: a FRESH session seeded from the .spore/ checkpoint.
		session := executor.RalphSeedSession(ctx, task.Instruction)

		windowTask := Task{
			ID:           task.ID,
			Instruction:  task.Instruction,
			SessionID:    windowSessionID,
			Budget:       task.Budget,
			LoopStrategy: innerForWindow,
		}
		wtCopy := windowTask
		cx.Scratch.Task = &wtCopy
		cx.Scratch.RunSession = session
		// FRESH per-window budget (the reset discards the turn budget).
		cx.Scratch.RunBudget = BudgetSnapshot{}
		windowOutcome := innerForWindow.Run(ctx, cx)
		// #125 rule 4/7: a window child's BudgetExhausted reaches Ralph as a
		// StrategyOutcome, never auto-cascaded. Ralph's recovery semantics: a
		// budget-exhausted window is treated as "window incomplete" — RESET the
		// context window and retry (next outer iteration). After maxResets this
		// falls through to RalphCompletionUnmet. Ralph's own scope is unaffected.
		if windowOutcome.Kind == StrategyOutcomeBudgetExhausted {
			_ = cx.takeChildOverride()
			partial := "<no partial>"
			if windowOutcome.Exhausted != nil && windowOutcome.Exhausted.PartialOutput != nil {
				partial = *windowOutcome.Exhausted.PartialOutput
			}
			lastReason = fmt.Sprintf("window %d budget-exhausted: %s", iteration+1, partial)
			continue
		}
		wo := cx.takeChildOverride()
		var windowResult RunResult
		if wo == nil {
			windowResult = RunResult{
				Kind: RunFailure,
				Reason: HaltReason{
					Kind:       HaltRalphCompletionUnmet,
					Iterations: iteration + 1,
					Reason:     "window sub-strategy produced no terminal",
				},
				SessionID: windowSessionID,
			}
		} else {
			windowResult = *wo
		}
		windowBudget := BudgetSnapshot{}
		foldSelfVerifyUsage(&totalUsage, &windowBudget, windowResult)
		cumulativeTurns += windowBudget.Turns

		// A paused / escalated window propagates verbatim.
		switch windowResult.Kind {
		case RunWaitingForHuman, RunConsult, RunEscalate:
			cx.Stream = onStream
			return cx.finish(ctx, executor, task, windowResult)
		}

		reason, incomplete := executor.RalphCompletionStatus(ctx)
		if !incomplete {
			output := ""
			finalState := SessionState{}
			if windowResult.Kind == RunSuccess {
				output = windowResult.Output
				finalState = windowResult.SessionState
			}
			result := RunResult{
				Kind:         RunSuccess,
				Output:       output,
				SessionID:    windowSessionID,
				Usage:        totalUsage,
				Turns:        cumulativeTurns,
				SessionState: finalState,
			}
			cx.Stream = onStream
			return cx.finish(ctx, executor, task, result)
		}
		lastReason = reason
		lastSessionState = windowResult.SessionState
	}

	result := RunResult{
		Kind: RunFailure,
		Reason: HaltReason{
			Kind:       HaltRalphCompletionUnmet,
			Iterations: maxResets,
			Reason:     lastReason,
		},
		SessionID:    lastSessionID,
		Usage:        totalUsage,
		Turns:        cumulativeTurns,
		SessionState: lastSessionState,
	}
	cx.Stream = onStream
	return cx.finish(ctx, executor, task, result)
}

// Run is the HillClimbing combinator (#124): GENUINELY recursive optimization
// loop. Iteration 0 is a pure baseline (no agent turn). Iterations 1.. recurse
// c.Inner.Run(ctx, cx) to propose a change (a non-ReAct inner — e.g. PlanExecute
// — really runs its whole loop per iteration), then evaluate the metric
// (resolved via cx.Registry.ResolveMetricEvaluator, Q2) and keep/revert. Bounded
// by max_stagnation and the turn budget. The incoming session is discarded.
func (c *HillClimbingConfig) Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome {
	executor, fail := cx.executor()
	if executor == nil {
		return fail
	}
	task := cx.currentTask()
	sessionID := task.SessionID
	onStream := cx.takeStream()
	carried := cx.Scratch.RunBudget
	_ = cx.takeSession()
	workspaceRoot := executor.WorkspaceRoot()

	direction := c.Direction
	revert := c.RevertOnNoImprovement
	minDelta := c.MinImprovementDelta
	var maxStagnation *uint32
	if c.MaxStagnation != ^uint32(0) {
		v := c.MaxStagnation
		maxStagnation = &v
	}

	// Q2: resolve the metric evaluator from c.Evaluator's key.
	evaluator, ok := cx.Registry.ResolveMetricEvaluator(string(c.Evaluator))
	if !ok {
		result := RunResult{
			Kind: RunFailure,
			Reason: HaltReason{
				Kind: HaltHillClimbingMisconfigured,
				Reason: fmt.Sprintf(
					"HillClimbing requires a metric evaluator registered under key %q", string(c.Evaluator),
				),
			},
			SessionID: sessionID,
		}
		cx.Stream = onStream
		return cx.finish(ctx, executor, task, result)
	}
	description := evaluator.Description()

	var totalUsage AggregateUsage
	var rows []HillClimbRow
	var spanSeq uint64

	// #125: HillClimbing owns a budget scope for its optimization loop. POLICY is
	// the task's global turn ceiling (TotalSteps); this REPLACES the ad-hoc
	// turn_cap / carried.Turns >= turnCap gate that #124 used. Behavior is Escalate
	// (in-process placeholder; #129).
	var hcPolicy BudgetPolicy
	if task.Budget.MaxTurns != nil {
		hcPolicy = BudgetPolicy{Kind: BudgetTotalSteps, Value: *task.Budget.MaxTurns}
	} else {
		hcPolicy = BudgetPolicy{Kind: BudgetUnlimited}
	}
	cx.PushBudget(hcPolicy, c.Behavior, "hill_climbing")

	// ── Iteration 0: pure baseline. No agent turn (Decision 5).
	baseValue, baseDur, baseStatus, baseMsg, baseOK := executor.HillEvaluate(ctx, evaluator, sessionID, task.ID)
	if !baseOK {
		// Decision 7: a baseline that cannot be measured is a misconfiguration.
		rows = append(rows, HillClimbRow{
			Iteration:   0,
			CommitHash:  executor.HillCommitHash(ctx),
			HasMetric:   false,
			Direction:   direction,
			Status:      baseStatus,
			Duration:    0,
			Description: description,
		})
		executor.HillEmitIteration(ctx, sessionID, task.ID, &spanSeq, 0, 0, false, 0, false, baseStatus, false)
		executor.HillWriteTSV(workspaceRoot, task.ID, rows)
		result := RunResult{
			Kind: RunFailure,
			Reason: HaltReason{
				Kind:   HaltHillClimbingMisconfigured,
				Reason: fmt.Sprintf("baseline evaluation failed: %s", baseMsg),
			},
			SessionID: sessionID,
			Usage:     totalUsage,
			Turns:     carried.Turns,
		}
		cx.PopBudget()
		cx.Stream = onStream
		return cx.finish(ctx, executor, task, result)
	}
	currentBest := baseValue
	rows = append(rows, HillClimbRow{
		Iteration:   0,
		CommitHash:  executor.HillCommitHash(ctx),
		MetricValue: baseValue,
		HasMetric:   true,
		Direction:   direction,
		Status:      HillClimbKept,
		Duration:    baseDur,
		Description: description,
	})
	executor.HillEmitIteration(ctx, sessionID, task.ID, &spanSeq, 0, baseValue, true, 0, false, HillClimbKept, false)

	var stagnation uint32
	iteration := uint32(1)

	for {
		// #125: charge-based budget gate before the iteration's agent turn. A spent
		// TotalSteps cap surfaces this node's OWN typed BudgetExhausted (partial =
		// best candidate + score), resolving its behavior — replacing the legacy
		// BudgetExceeded Failure.
		if chErr := cx.ChargeCurrent(1); chErr != nil {
			resolution := cx.ResolveCurrent()
			// #129: a granted Continue resets the scope and KEEPS ITERATING the
			// climb IN-PROCESS (do NOT pop; the refreshed allowance lets the next
			// charge pass). NO serialization (AC3). MaxContinues bounds the loop.
			if resolution == ExhaustedResolutionContinue {
				continue
			}
			executor.HillWriteTSV(workspaceRoot, task.ID, rows)
			partial := hillClimbingPartialJSON(currentBest)
			cx.PopBudget()
			pt := task
			cx.Scratch.Task = &pt
			cx.Stream = onStream
			// #130: under SurfaceToHuman, an Escalate-resolving exhaustion PAUSES
			// with a BudgetExhausted request (combinator actions) recorded as the
			// verbatim terminal override instead of propagating up.
			if resolution != ExhaustedResolutionFail &&
				executor.EscalationMode().Kind == EscalationSurfaceToHuman {
				// #138 AC2-a: HillClimbing iterates on a METRIC, not a worker
				// conversation, so there is no richer session to carry — pass the empty
				// default to fall back to the partial-only stub (pre-#138 behavior). The
				// worker leaf's toolset handle is still carried (#140 parity, AC4-a).
				waiting := promoteBudgetExhaustedToHuman(
					chErr, &partial, combinatorEscalationActions(chErr),
					sessionID, task, carried, carried.Turns,
					SessionState{}, workerToolsetOf(c.Inner),
				)
				return cx.recordTerminal(waiting)
			}
			cx.Scratch.TerminalOverride = nil
			switch resolution {
			case ExhaustedResolutionFail:
				return promoteBudgetExhausted(chErr, nil)
			default:
				// #129: the in-process Continue is handled above; this is
				// Escalate-only now (the surface/propagate shape).
				return promoteBudgetExhausted(chErr, &partial)
			}
		}
		if lt, exceeded := budgetExceeded(task.Budget, carried, time.Now()); exceeded {
			executor.HillWriteTSV(workspaceRoot, task.ID, rows)
			result := RunResult{
				Kind:      RunFailure,
				Reason:    HaltReason{Kind: HaltBudgetExceeded, LimitType: lt},
				SessionID: sessionID,
				Usage:     totalUsage,
				Turns:     carried.Turns,
			}
			cx.PopBudget()
			cx.Stream = onStream
			return cx.finish(ctx, executor, task, result)
		}

		// ── One agent turn proposes a change: recurse c.Inner.Run.
		iterTask := Task{
			ID:           task.ID,
			Instruction:  task.Instruction,
			SessionID:    sessionID,
			Budget:       task.Budget,
			LoopStrategy: *c.Inner,
		}
		var iterState SessionState
		executor.AppendUserMessage(ctx, &iterState, task.Instruction)
		itCopy := iterTask
		cx.Scratch.Task = &itCopy
		cx.Scratch.RunSession = iterState
		cx.Scratch.RunBudget = carried
		iterOutcome := c.Inner.Run(ctx, cx)
		// #125 rule 4/7: a child's BudgetExhausted reaches HillClimbing as a
		// StrategyOutcome, never auto-cascaded. Surface this node's own typed
		// BudgetExhausted (partial = best candidate + score).
		if iterOutcome.Kind == StrategyOutcomeBudgetExhausted {
			executor.HillWriteTSV(workspaceRoot, task.ID, rows)
			_ = cx.takeChildOverride()
			partial := hillClimbingPartialJSON(currentBest)
			err := cx.currentExhausted()
			cx.PopBudget()
			pt := task
			cx.Scratch.Task = &pt
			cx.Scratch.TerminalOverride = nil
			cx.Stream = onStream
			return promoteBudgetExhausted(err, &partial)
		}
		to := cx.takeChildOverride()
		var turnResult RunResult
		if to == nil {
			turnResult = RunResult{
				Kind:      RunFailure,
				Reason:    HaltReason{Kind: HaltBudgetExceeded, LimitType: BudgetLimitTurns},
				SessionID: sessionID,
				Turns:     carried.Turns,
			}
		} else {
			turnResult = *to
		}
		foldSelfVerifyUsage(&totalUsage, &carried, turnResult)

		// A paused / escalated turn propagates verbatim.
		switch turnResult.Kind {
		case RunWaitingForHuman, RunConsult, RunEscalate:
			executor.HillWriteTSV(workspaceRoot, task.ID, rows)
			cx.PopBudget()
			cx.Stream = onStream
			return cx.finish(ctx, executor, task, turnResult)
		}

		// ── Evaluate the metric after the change.
		value, dur, evalStatus, _, evalOK := executor.HillEvaluate(ctx, evaluator, sessionID, task.ID)
		if !evalOK {
			reverted := false
			if revert {
				executor.HillRevert(ctx)
				reverted = true
			}
			stagnation++
			rows = append(rows, HillClimbRow{
				Iteration:   iteration,
				CommitHash:  executor.HillCommitHash(ctx),
				HasMetric:   false,
				Direction:   direction,
				Status:      evalStatus,
				Duration:    0,
				Description: description,
			})
			executor.HillEmitIteration(ctx, sessionID, task.ID, &spanSeq, iteration, 0, false, 0, false, evalStatus, reverted)
		} else {
			kept := hillClimbShouldKeep(value, currentBest, direction, &minDelta)
			var delta float64
			switch direction {
			case OptimizationMinimize:
				delta = currentBest - value
			default:
				delta = value - currentBest
			}
			if kept {
				currentBest = value
				stagnation = 0
				rows = append(rows, HillClimbRow{
					Iteration:   iteration,
					CommitHash:  executor.HillCommitHash(ctx),
					MetricValue: value,
					HasMetric:   true,
					Direction:   direction,
					Status:      HillClimbKept,
					Duration:    dur,
					Description: description,
				})
				executor.HillEmitIteration(ctx, sessionID, task.ID, &spanSeq, iteration, value, true, delta, true, HillClimbKept, false)
			} else {
				reverted := false
				if revert {
					executor.HillRevert(ctx)
					reverted = true
				}
				stagnation++
				rows = append(rows, HillClimbRow{
					Iteration:   iteration,
					CommitHash:  executor.HillCommitHash(ctx),
					MetricValue: value,
					HasMetric:   true,
					Direction:   direction,
					Status:      HillClimbDiscarded,
					Duration:    dur,
					Description: description,
				})
				executor.HillEmitIteration(ctx, sessionID, task.ID, &spanSeq, iteration, value, true, delta, true, HillClimbDiscarded, reverted)
			}
		}

		// ── Stagnation halt (only when a cap is configured).
		if maxStagnation != nil && stagnation >= *maxStagnation {
			executor.HillWriteTSV(workspaceRoot, task.ID, rows)
			result := RunResult{
				Kind: RunFailure,
				Reason: HaltReason{
					Kind:       HaltStagnationLimitReached,
					Iterations: stagnation,
					BestMetric: currentBest,
				},
				SessionID: sessionID,
				Usage:     totalUsage,
				Turns:     carried.Turns,
			}
			cx.PopBudget()
			cx.Stream = onStream
			return cx.finish(ctx, executor, task, result)
		}

		if iteration < allTurns {
			iteration++
		}
	}
	// #125: the loop never falls out — the charge-based budget gate at the top
	// surfaces a typed BudgetExhausted, and stagnation / pause / misconfig all
	// return from inside. The legacy post-loop "clean budget halt" is now dead and
	// removed (the turn_cap break it depended on is gone).
}

// workerAgentKeyOf descends a LoopStrategy tree to the worker leaf's agent key
// (#124). The worker is the agent on the leaf reached by descending the primary
// worker child chain. A Ralph with a non-empty agent override resolves THAT (Q3).
func workerAgentKeyOf(ls *LoopStrategy) string {
	if ls == nil {
		return ""
	}
	switch ls.Kind {
	case StrategyReAct:
		if ls.ReActCfg == nil {
			return ""
		}
		return string(ls.ReActCfg.Agent)
	case StrategyPlanExecute:
		if ls.PlanExecute == nil {
			return ""
		}
		return workerAgentKeyOf(ls.PlanExecute.Execute)
	case StrategySelfVerifying:
		if ls.SelfVerify == nil {
			return ""
		}
		return workerAgentKeyOf(ls.SelfVerify.Inner)
	case StrategyRalph:
		if ls.Ralph == nil {
			return ""
		}
		if string(ls.Ralph.Agent) != "" {
			return string(ls.Ralph.Agent)
		}
		return workerAgentKeyOf(ls.Ralph.Inner)
	case StrategyHillClimbing:
		if ls.HillClimbing == nil {
			return ""
		}
		return workerAgentKeyOf(ls.HillClimbing.Inner)
	default:
		return ""
	}
}

// preexistingHasInProgress reports whether the durable task_list (as probed
// before the plan phase) has any InProgress task (#138 AC3). An empty / absent
// list (no InProgress task) means a present resume seed came from a PLAN-phase
// exhaustion — the planner is re-seeded from the carried conversation. A list
// WITH an InProgress task means an EXECUTE-phase exhaustion — the seed re-attaches
// to that task in the ready-set walk instead.
func preexistingHasInProgress(list TaskList) bool {
	for i := range list.Tasks {
		if list.Tasks[i].Status == TaskStatusInProgress {
			return true
		}
	}
	return false
}

// workerToolsetOf descends a LoopStrategy tree to the worker (execute) leaf's
// toolset handle (#138 AC4-a). Mirrors workerAgentKeyOf: a combinator descends
// into its EXECUTE child (PlanExecute) / inner (SelfVerifying/Ralph/HillClimbing)
// so a budget-exhausted pause records the same handle #140 would have on the
// leaf's own pause (e.g. "exec-tools"), not the empty default.
func workerToolsetOf(ls *LoopStrategy) ToolsetRef {
	if ls == nil {
		return ToolsetRef("")
	}
	switch ls.Kind {
	case StrategyReAct:
		if ls.ReActCfg == nil {
			return ToolsetRef("")
		}
		return ls.ReActCfg.Toolset
	case StrategyPlanExecute:
		if ls.PlanExecute == nil {
			return ToolsetRef("")
		}
		return workerToolsetOf(ls.PlanExecute.Execute)
	case StrategySelfVerifying:
		if ls.SelfVerify == nil {
			return ToolsetRef("")
		}
		return workerToolsetOf(ls.SelfVerify.Inner)
	case StrategyRalph:
		if ls.Ralph == nil {
			return ToolsetRef("")
		}
		return workerToolsetOf(ls.Ralph.Inner)
	case StrategyHillClimbing:
		if ls.HillClimbing == nil {
			return ToolsetRef("")
		}
		return workerToolsetOf(ls.HillClimbing.Inner)
	default:
		return ToolsetRef("")
	}
}

// fillEmptyWorkerAgent fills the worker leaf's agent handle of ls with agent
// ONLY when that leaf's handle is empty (#124 Q3 — Ralph's per-window agent
// fill). An explicitly-declared leaf agent is authoritative and is left
// untouched. Mutates the leaf reached by descending the worker child chain.
// Operates on the *LoopStrategy in place (combinator children are pointers, so
// descending mutates the shared subtree — callers pass a CLONE).
func fillEmptyWorkerAgent(ls *LoopStrategy, agent AgentRef) {
	if ls == nil {
		return
	}
	switch ls.Kind {
	case StrategyReAct:
		if ls.ReActCfg != nil {
			if string(ls.ReActCfg.Agent) == "" {
				cfg := *ls.ReActCfg
				cfg.Agent = agent
				ls.ReActCfg = &cfg
			}
		}
	case StrategyPlanExecute:
		if ls.PlanExecute != nil {
			child := *ls.PlanExecute.Execute
			fillEmptyWorkerAgent(&child, agent)
			cfg := *ls.PlanExecute
			cfg.Execute = &child
			ls.PlanExecute = &cfg
		}
	case StrategySelfVerifying:
		if ls.SelfVerify != nil {
			child := *ls.SelfVerify.Inner
			fillEmptyWorkerAgent(&child, agent)
			cfg := *ls.SelfVerify
			cfg.Inner = &child
			ls.SelfVerify = &cfg
		}
	case StrategyRalph:
		if ls.Ralph != nil {
			child := *ls.Ralph.Inner
			fillEmptyWorkerAgent(&child, agent)
			cfg := *ls.Ralph
			cfg.Inner = &child
			ls.Ralph = &cfg
		}
	case StrategyHillClimbing:
		if ls.HillClimbing != nil {
			child := *ls.HillClimbing.Inner
			fillEmptyWorkerAgent(&child, agent)
			cfg := *ls.HillClimbing
			cfg.Inner = &child
			ls.HillClimbing = &cfg
		}
	}
}

// Compile-time interface checks.
var (
	_ RunStrategy = LoopStrategy{}
	_ RunStrategy = (*ReactConfig)(nil)
	_ RunStrategy = (*PlanExecuteConfig)(nil)
	_ RunStrategy = (*SelfVerifyingConfig)(nil)
	_ RunStrategy = (*RalphConfig)(nil)
	_ RunStrategy = (*HillClimbingConfig)(nil)
)
