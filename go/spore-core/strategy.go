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
	// #125: push this leaf's OWN budget scope. The leaf carries only its POLICY
	// (the cap); behavior is Escalate as a leaf placeholder — the leaf never
	// RESOLVES it (rule 6: propagate to parent), it is only the scope shape so the
	// charge below enforces the cap and any exhaustion promotes to a
	// parent-inspectable BudgetExhausted.
	cx.PushBudget(c.Budget, BudgetExhaustedBehavior{Kind: BehaviorEscalate}, "react")
	// The leaf takes the run's stream sink for the window; combinators that
	// recurse per-phase suppress it (they take it before recursing).
	onStream := cx.takeStream()
	result := executor.ReactWindow(ctx, task, c.MaxIterations(), session, budget, onStream, agent)
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
					Behavior:   BudgetExhaustedBehavior{Kind: BehaviorEscalate},
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

	// ── Phase 1: plan (dispatch through c.Plan). ────────────────────────────
	//
	// Seed the planning directive onto a CLONE of the base session so the shared
	// execute context stays [user: task.instruction] (#93 — a leaked directive
	// would make every execute step re-emit a plan). Cap the plan child at ONE
	// turn (R1): the plan is a single constrained turn that yields the JSON
	// artifact, but never beyond the task's global turn ceiling (R10).
	directive := executor.PlanDirective(task.Instruction)
	planSession := baseSession
	executor.SeedUserMessage(ctx, &planSession, directive)
	planCap := saturatingAddU32(budget.Turns, 1)
	if task.Budget.MaxTurns != nil && *task.Budget.MaxTurns < planCap {
		planCap = *task.Budget.MaxTurns
	}
	planBudget := task.Budget
	planBudget.MaxTurns = &planCap
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

	// ── Phase 2: execute (dispatch c.Execute PER TASK). ─────────────────────
	//
	// The shared execute context starts from baseSession (NOT the plan child's
	// polluted session) so the directive never leaks (#93).
	result, exhausted := c.runExecuteLoop(ctx, cx, executor, &task, baseSession, taskList, carried, outcome.Usage, onStream)
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
// (#124). It owns the per-task orchestration: A.6 deep-resume reconcile, Q1
// per-task turn allocation, the OnTaskAdvance hook, seeding each step instruction
// as a user message on the SHARED execute context, task-list persistence after
// each transition (Q4), and cumulative usage / last-output / last-state carry.
// Returns the terminal RunResult for the execute phase.
func (c *PlanExecuteConfig) runExecuteLoop(
	ctx context.Context,
	cx *ExecutionContext,
	executor StrategyExecutor,
	task *Task,
	session SessionState,
	taskList TaskList,
	carried BudgetSnapshot,
	planUsage AggregateUsage,
	onStream StreamSink,
) (RunResult, *StrategyOutcome) {
	sessionID := task.SessionID

	// A.6 deep-resume (Q2): reconcile against the durable checkpoint so
	// already-Completed tasks are not re-run.
	executor.ReconcileCompletedTasks(ctx, sessionID, &taskList)

	totalTasks := len(taskList.Tasks)
	totalUsage := planUsage
	var lastOutput string
	var lastState SessionState

	// #125: PlanExecute owns a budget scope for its execute phase. Its POLICY is
	// the task's global turn ceiling (TotalSteps) — the root node's TotalSteps
	// subsumes the OLD per-task remaining_turns / remaining_tasks / step_cap
	// derivation, which is now DEAD and removed. Behavior is Escalate (in-process
	// placeholder; the serialized behavior field is #129). Enforcement is now
	// charge-based per node.
	var planPolicy BudgetPolicy
	if task.Budget.MaxTurns != nil {
		planPolicy = BudgetPolicy{Kind: BudgetTotalSteps, Value: *task.Budget.MaxTurns}
	} else {
		planPolicy = BudgetPolicy{Kind: BudgetUnlimited}
	}
	cx.PushBudget(planPolicy, BudgetExhaustedBehavior{Kind: BehaviorEscalate}, "plan_execute")

	for index := 0; index < totalTasks; index++ {
		taskID := taskList.Tasks[index].ID
		instruction := taskList.Tasks[index].Description

		// A.6 deep-resume: a task already Completed is skipped.
		if taskList.Tasks[index].Status == TaskStatusCompleted {
			lastOutput = instruction
			continue
		}

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

		// Seed the step instruction as a user message on the SHARED execute
		// context, then dispatch the execute sub-strategy.
		executor.SeedUserMessage(ctx, &session, stepTask.Instruction)

		stCopy := stepTask
		cx.Scratch.Task = &stCopy
		cx.Scratch.RunSession = session
		cx.Scratch.RunBudget = carried
		// #125: absolute turn count BEFORE this step, so the success path can
		// charge only the DELTA against the PlanExecute scope.
		carriedBefore := carried.Turns
		stepOutcome := c.Execute.Run(ctx, cx)
		// #125 rule 4/7: a child's BudgetExhausted reaches THIS parent as a
		// StrategyOutcome, never auto-cascaded. PlanExecute does NOT charge the
		// child's exhaustion against its OWN scope; it surfaces its own typed
		// BudgetExhausted with the PlanExecute partial (tasklist + statuses +
		// ledger) and aborts the run.
		if stepOutcome.Kind == StrategyOutcomeBudgetExhausted {
			blocked := TaskStatusBlocked
			_ = taskList.Update(taskID, &blocked, nil)
			executor.PersistTaskList(ctx, sessionID, taskList)
			partial := planExecutePartialJSON(taskList)
			err := cx.currentExhausted()
			cx.PopBudget()
			_ = cx.takeChildOverride()
			cx.Scratch.RunSession = session
			outcome := promoteBudgetExhausted(err, &partial)
			return RunResult{}, &outcome
		}
		subResult := cx.takeChildOverride()

		if subResult == nil {
			cx.PopBudget()
			return RunResult{
				Kind: RunFailure,
				Reason: HaltReason{
					Kind:      HaltStepFailed,
					TaskIndex: index,
					Task:      taskList.Tasks[index].Description,
					Reason:    "execute sub-strategy produced no terminal",
				},
				SessionID:    sessionID,
				Usage:        totalUsage,
				Turns:        carried.Turns,
				SessionState: lastState,
			}, nil
		}

		switch subResult.Kind {
		case RunSuccess:
			// Carry the shared budget forward (Q1) and fold this step's
			// conversation back into the SHARED context so the next step builds on
			// its results.
			carried.Turns = subResult.Turns
			session = subResult.SessionState
			lastState = session
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
			emit(onStream, HarnessStreamEvent{Kind: HarnessStreamFinalResponse, Content: lastOutput})

			// #125: charge this step's turns against the PlanExecute scope. If the
			// global TotalSteps cap is now spent, the PlanExecute node surfaces its
			// OWN typed BudgetExhausted (partial = tasklist + ledger), resolving its
			// behavior.
			if chErr := cx.ChargeCurrent(saturatingSubU32(subResult.Turns, carriedBefore)); chErr != nil {
				partial := planExecutePartialJSON(taskList)
				resolution := cx.ResolveCurrent()
				cx.PopBudget()
				cx.Scratch.RunSession = session
				var outcome StrategyOutcome
				switch resolution {
				case ExhaustedResolutionFail:
					outcome = promoteBudgetExhausted(chErr, nil)
				case ExhaustedResolutionContinue, ExhaustedResolutionEscalate:
					// #129: a granted Continue must reset this scope
					// (ResolveCurrent already did via ConsumeContinue) and RE-RUN
					// this execute step — the in-process loop wiring lands in #129.
					// It is UNREACHABLE today: live bodies push an Escalate
					// placeholder behavior (the serialized BudgetExhaustedBehavior
					// field is a wire change deferred to #129), so ResolveCurrent
					// only ever yields Escalate/Fail here. Until that loop exists,
					// an (impossible) Continue is handled EXPLICITLY as a lossless
					// surface-with-partial rather than a silent default fall-through.
					outcome = promoteBudgetExhausted(chErr, &partial)
				}
				return RunResult{}, &outcome
			}

		case RunFailure:
			// Q5: any non-success step aborts the whole run.
			totalUsage.InputTokens += subResult.Usage.InputTokens
			totalUsage.OutputTokens += subResult.Usage.OutputTokens
			totalUsage.CacheReadTokens += subResult.Usage.CacheReadTokens
			totalUsage.CacheWriteTokens += subResult.Usage.CacheWriteTokens
			totalUsage.CostUSD += subResult.Usage.CostUSD

			blocked := TaskStatusBlocked
			_ = taskList.Update(taskID, &blocked, nil)
			executor.PersistTaskList(ctx, sessionID, taskList)

			var terminalReason HaltReason
			if subResult.Reason.Kind == HaltBudgetExceeded {
				terminalReason = subResult.Reason
			} else {
				terminalReason = HaltReason{
					Kind:      HaltStepFailed,
					TaskIndex: index,
					Task:      taskList.Tasks[index].Description,
					Reason:    haltReasonString(subResult.Reason),
				}
			}
			cx.PopBudget()
			return RunResult{
				Kind:         RunFailure,
				Reason:       terminalReason,
				SessionID:    sessionID,
				Usage:        totalUsage,
				Turns:        subResult.Turns,
				SessionState: lastState,
			}, nil

		default:
			// A pause / consult / escalate propagates the whole run verbatim.
			cx.PopBudget()
			return *subResult, nil
		}
	}

	cx.PopBudget()
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
	cx.PushBudget(svPolicy, BudgetExhaustedBehavior{Kind: BehaviorEscalate}, "self_verifying")

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
			partial := selfVerifyingPartialJSON(lastWorkerOutput, lastReason)
			resolution := cx.ResolveCurrent()
			cx.PopBudget()
			pt := task
			cx.Scratch.Task = &pt
			cx.Scratch.TerminalOverride = nil
			cx.Stream = onStream
			switch resolution {
			case ExhaustedResolutionFail:
				return promoteBudgetExhausted(chErr, nil)
			case ExhaustedResolutionContinue, ExhaustedResolutionEscalate:
				// #129: a granted Continue must reset + RE-RUN this
				// build↔evaluate iteration (the in-process loop wiring lands in
				// #129). UNREACHABLE today — live bodies push an Escalate
				// placeholder, so ResolveCurrent never yields Continue here.
				// Handle it EXPLICITLY (surface-with-partial) rather than via a
				// silent default fall-through.
				return promoteBudgetExhausted(chErr, &partial)
			}
			return promoteBudgetExhausted(chErr, &partial)
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
// whole loop per window). Q3: when c.Agent is set it OVERRIDES the inner leaf's
// agent per window; when unset the worker resolves via the inner leaf.
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

	// Q3: when c.Agent is set, override the inner leaf's agent for every window
	// by rewriting the inner tree's worker leaf handle.
	innerForWindow := *c.Inner
	if string(c.Agent) != "" {
		cloned := *c.Inner
		overrideWorkerAgent(&cloned, c.Agent)
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

		reason, incomplete := executor.RalphCompletionStatus()
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
	cx.PushBudget(hcPolicy, BudgetExhaustedBehavior{Kind: BehaviorEscalate}, "hill_climbing")

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
			executor.HillWriteTSV(workspaceRoot, task.ID, rows)
			partial := hillClimbingPartialJSON(currentBest)
			resolution := cx.ResolveCurrent()
			cx.PopBudget()
			pt := task
			cx.Scratch.Task = &pt
			cx.Scratch.TerminalOverride = nil
			cx.Stream = onStream
			switch resolution {
			case ExhaustedResolutionFail:
				return promoteBudgetExhausted(chErr, nil)
			case ExhaustedResolutionContinue, ExhaustedResolutionEscalate:
				// #129: a granted Continue must reset + keep iterating the climb
				// (the in-process loop wiring lands in #129). UNREACHABLE today —
				// live bodies push an Escalate placeholder, so ResolveCurrent
				// never yields Continue here. Handle it EXPLICITLY
				// (surface-with-partial) rather than via a silent default
				// fall-through.
				return promoteBudgetExhausted(chErr, &partial)
			}
			return promoteBudgetExhausted(chErr, &partial)
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

// overrideWorkerAgent rewrites the worker leaf's agent handle of ls to agent
// (#124 Q3 — Ralph's per-window agent override). Mutates the leaf reached by
// descending the worker child chain. Operates on the *LoopStrategy in place
// (combinator children are pointers, so descending mutates the shared subtree —
// callers pass a CLONE).
func overrideWorkerAgent(ls *LoopStrategy, agent AgentRef) {
	if ls == nil {
		return
	}
	switch ls.Kind {
	case StrategyReAct:
		if ls.ReActCfg != nil {
			cfg := *ls.ReActCfg
			cfg.Agent = agent
			ls.ReActCfg = &cfg
		}
	case StrategyPlanExecute:
		if ls.PlanExecute != nil {
			child := *ls.PlanExecute.Execute
			overrideWorkerAgent(&child, agent)
			cfg := *ls.PlanExecute
			cfg.Execute = &child
			ls.PlanExecute = &cfg
		}
	case StrategySelfVerifying:
		if ls.SelfVerify != nil {
			child := *ls.SelfVerify.Inner
			overrideWorkerAgent(&child, agent)
			cfg := *ls.SelfVerify
			cfg.Inner = &child
			ls.SelfVerify = &cfg
		}
	case StrategyRalph:
		if ls.Ralph != nil {
			child := *ls.Ralph.Inner
			overrideWorkerAgent(&child, agent)
			cfg := *ls.Ralph
			cfg.Inner = &child
			ls.Ralph = &cfg
		}
	case StrategyHillClimbing:
		if ls.HillClimbing != nil {
			child := *ls.HillClimbing.Inner
			overrideWorkerAgent(&child, agent)
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
