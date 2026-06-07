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
	Budget  BudgetPolicy
	Agent   AgentRef
	Toolset ToolsetRef
	// Output is omitted from JSON when nil (matches Rust Option + skip-if-none).
	Output *SchemaRef
}

// ReactPerLoop builds a bare ReAct leaf with a PerLoop{value} budget and empty
// agent / toolset handles (resolution lands with the registry slice, #120).
// This is the migration shim for the old ReAct{max_iterations} shape.
func ReactPerLoop(value uint32) ReactConfig {
	return ReactConfig{
		Budget:  BudgetPolicy{Kind: BudgetPerLoop, Value: value},
		Agent:   AgentRef(""),
		Toolset: ToolsetRef(""),
		Output:  nil,
	}
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
}

// PlanExecuteSimple builds a PlanExecute whose plan and execute phases are both
// bare ReAct leaves (migration shim for the old PlanExecute{plan_model} shape;
// the live executor does not read the children yet — #124).
func PlanExecuteSimple(planModel *ModelConfig) PlanExecuteConfig {
	plan := ReActStrategy(^uint32(0))
	exec := ReActStrategy(^uint32(0))
	return PlanExecuteConfig{
		Plan:      &plan,
		Execute:   &exec,
		PlanModel: planModel,
	}
}

// SelfVerifyingConfig is the SelfVerifying combinator: run inner, then judge it
// against evaluator.
type SelfVerifyingConfig struct {
	Inner     *LoopStrategy
	Evaluator SchemaRef
}

// RalphConfig is the Ralph combinator: re-run inner under a fixed agent across
// context-window resets.
type RalphConfig struct {
	Inner *LoopStrategy
	Agent AgentRef
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
		// Key order: kind, budget, agent, toolset, [output].
		type reactFlat struct {
			Kind    LoopStrategyKind `json:"kind"`
			Budget  BudgetPolicy     `json:"budget"`
			Agent   AgentRef         `json:"agent"`
			Toolset ToolsetRef       `json:"toolset"`
			Output  *SchemaRef       `json:"output,omitempty"`
		}
		return json.Marshal(reactFlat{s.Kind, c.Budget, c.Agent, c.Toolset, c.Output})
	case StrategyPlanExecute:
		if s.PlanExecute == nil {
			return nil, fmt.Errorf("LoopStrategy: plan_execute requires config")
		}
		c := s.PlanExecute
		// Key order: kind, plan, execute, [plan_model].
		type planExecuteFlat struct {
			Kind      LoopStrategyKind `json:"kind"`
			Plan      *LoopStrategy    `json:"plan"`
			Execute   *LoopStrategy    `json:"execute"`
			PlanModel *ModelConfig     `json:"plan_model,omitempty"`
		}
		return json.Marshal(planExecuteFlat{s.Kind, c.Plan, c.Execute, c.PlanModel})
	case StrategySelfVerifying:
		if s.SelfVerify == nil {
			return nil, fmt.Errorf("LoopStrategy: self_verifying requires config")
		}
		c := s.SelfVerify
		// Key order: kind, inner, evaluator.
		type selfVerifyFlat struct {
			Kind      LoopStrategyKind `json:"kind"`
			Inner     *LoopStrategy    `json:"inner"`
			Evaluator SchemaRef        `json:"evaluator"`
		}
		return json.Marshal(selfVerifyFlat{s.Kind, c.Inner, c.Evaluator})
	case StrategyRalph:
		if s.Ralph == nil {
			return nil, fmt.Errorf("LoopStrategy: ralph requires config")
		}
		c := s.Ralph
		// Key order: kind, inner, agent.
		type ralphFlat struct {
			Kind  LoopStrategyKind `json:"kind"`
			Inner *LoopStrategy    `json:"inner"`
			Agent AgentRef         `json:"agent"`
		}
		return json.Marshal(ralphFlat{s.Kind, c.Inner, c.Agent})
	case StrategyHillClimbing:
		if s.HillClimbing == nil {
			return nil, fmt.Errorf("LoopStrategy: hill_climbing requires config")
		}
		c := s.HillClimbing
		// Key order: kind, inner, direction, max_stagnation,
		// revert_on_no_improvement, min_improvement_delta, evaluator.
		type hillFlat struct {
			Kind                  LoopStrategyKind      `json:"kind"`
			Inner                 *LoopStrategy         `json:"inner"`
			Direction             OptimizationDirection `json:"direction"`
			MaxStagnation         uint32                `json:"max_stagnation"`
			RevertOnNoImprovement bool                  `json:"revert_on_no_improvement"`
			MinImprovementDelta   float64               `json:"min_improvement_delta"`
			Evaluator             AgentRef              `json:"evaluator"`
		}
		return json.Marshal(hillFlat{
			s.Kind, c.Inner, c.Direction, c.MaxStagnation,
			c.RevertOnNoImprovement, c.MinImprovementDelta, c.Evaluator,
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
			Budget  BudgetPolicy `json:"budget"`
			Agent   AgentRef     `json:"agent"`
			Toolset ToolsetRef   `json:"toolset"`
			Output  *SchemaRef   `json:"output"`
		}
		if err := json.Unmarshal(data, &probe); err != nil {
			return err
		}
		s.ReActCfg = &ReactConfig{
			Budget:  probe.Budget,
			Agent:   probe.Agent,
			Toolset: probe.Toolset,
			Output:  probe.Output,
		}
		return nil
	case StrategyPlanExecute:
		var probe struct {
			Plan      *LoopStrategy `json:"plan"`
			Execute   *LoopStrategy `json:"execute"`
			PlanModel *ModelConfig  `json:"plan_model"`
		}
		if err := json.Unmarshal(data, &probe); err != nil {
			return err
		}
		s.PlanExecute = &PlanExecuteConfig{
			Plan:      probe.Plan,
			Execute:   probe.Execute,
			PlanModel: probe.PlanModel,
		}
		return nil
	case StrategySelfVerifying:
		var probe struct {
			Inner     *LoopStrategy `json:"inner"`
			Evaluator SchemaRef     `json:"evaluator"`
		}
		if err := json.Unmarshal(data, &probe); err != nil {
			return err
		}
		s.SelfVerify = &SelfVerifyingConfig{Inner: probe.Inner, Evaluator: probe.Evaluator}
		return nil
	case StrategyRalph:
		var probe struct {
			Inner *LoopStrategy `json:"inner"`
			Agent AgentRef      `json:"agent"`
		}
		if err := json.Unmarshal(data, &probe); err != nil {
			return err
		}
		s.Ralph = &RalphConfig{Inner: probe.Inner, Agent: probe.Agent}
		return nil
	case StrategyHillClimbing:
		var probe struct {
			Inner                 *LoopStrategy         `json:"inner"`
			Direction             OptimizationDirection `json:"direction"`
			MaxStagnation         uint32                `json:"max_stagnation"`
			RevertOnNoImprovement bool                  `json:"revert_on_no_improvement"`
			MinImprovementDelta   float64               `json:"min_improvement_delta"`
			Evaluator             AgentRef              `json:"evaluator"`
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
	session := cx.takeSession()
	budget := cx.Scratch.RunBudget
	// The leaf takes the run's stream sink for the window; combinators that
	// recurse per-phase suppress it (they take it before recursing).
	onStream := cx.takeStream()
	result := executor.ReactWindow(ctx, task, c.MaxIterations(), session, budget, onStream)
	executor.Finalize(ctx, result)
	return cx.recordTerminal(result)
}

// Run is the plan→execute combinator (#124). The plan phase recurses through the
// plan sub-strategy via the executor primitive (Q4 — a real loop building the
// task graph, not a one-shot dispatch); the execute phase recurses through the
// execute sub-strategy per task. The model-touching plan/execute machinery stays
// on the harness behind StrategyExecutor so behavior is at parity with the
// ported runPlanExecute body (AC6). The ready-set walk lands in #126 (execute
// runs per task sequentially for now).
func (c *PlanExecuteConfig) Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome {
	executor, fail := cx.executor()
	if executor == nil {
		return fail
	}
	task := cx.currentTask()
	sessionID := task.SessionID
	session := cx.takeSession()
	budget := cx.Scratch.RunBudget
	onStream := cx.takeStream()

	// ── Phase 1: plan (recurse through the plan sub-strategy). ──────────────
	outcome, failure := executor.PlanPhase(ctx, &task, &session, budget, onStream)
	if failure != nil {
		return cx.finish(ctx, executor, task, *failure)
	}

	taskList := PlanArtifactToTaskList(outcome.Artifact)
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
	executor.PersistTaskList(ctx, sessionID, taskList)

	// Carry the shared budget past the plan turn.
	carried := budget
	carried.Turns = outcome.Turns
	carried.InputTokens += outcome.Usage.InputTokens
	carried.OutputTokens += outcome.Usage.OutputTokens

	// ── Phase 2: execute (recurse through the execute sub-strategy). ────────
	result := executor.ExecutePhase(ctx, &task, &session, taskList, carried, outcome.Usage, onStream)
	return cx.finish(ctx, executor, task, result)
}

// Run is the SelfVerifying combinator: drive the build↔evaluate loop
// (Default-FAIL; bounded by the verifier's iteration cap / the run budget — Q1).
// The build phase recurses through the inner sub-strategy.
func (c *SelfVerifyingConfig) Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome {
	executor, fail := cx.executor()
	if executor == nil {
		return fail
	}
	task := cx.currentTask()
	session := cx.takeSession()
	budget := cx.Scratch.RunBudget
	onStream := cx.takeStream()
	result := executor.SelfVerifyingLoop(ctx, task, session, budget, onStream)
	return cx.recordTerminal(result)
}

// Run is the Ralph continuation wrapper: reset the context window per window and
// resume from the durable .spore/ checkpoint (A.6 deep-resume). Each window
// recurses through the inner sub-strategy. Ralph discards the incoming session
// state by design (each window is a fresh start re-seeded from the filesystem
// checkpoint).
func (c *RalphConfig) Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome {
	executor, fail := cx.executor()
	if executor == nil {
		return fail
	}
	task := cx.currentTask()
	budget := cx.Scratch.RunBudget
	onStream := cx.takeStream()
	_ = cx.takeSession() // discarded: each window re-seeds from the checkpoint.
	result := executor.RalphLoop(ctx, task, budget, onStream)
	return cx.recordTerminal(result)
}

// Run is the HillClimbing combinator: iterate the inner sub-strategy, scoring
// each candidate with the metric; bounded by max_stagnation (Q1). The incoming
// session is discarded — each iteration re-seeds a fresh window internally.
func (c *HillClimbingConfig) Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome {
	executor, fail := cx.executor()
	if executor == nil {
		return fail
	}
	task := cx.currentTask()
	budget := cx.Scratch.RunBudget
	onStream := cx.takeStream()
	_ = cx.takeSession()
	result := executor.HillClimbingLoop(ctx, task, budget, onStream)
	return cx.recordTerminal(result)
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
