// Composable Execution runtime scaffold (issue #123): StrategyOutcome +
// ExecutionContext / BudgetContext / BudgetStack / SpanStack.
//
// SCAFFOLD ONLY. This slice establishes the typed strategy outcome and the
// shared, mutable runtime context that threads through a nested strategy tree.
// Charge here is PURE ARITHMETIC against a per-scope step allowance — the
// behavior-chain walk, continue-consumption, and persistence are the later
// budget-enforcement slice (#124+).
//
// Types defined in this file:
//   - StrategyOutcome — typed result a strategy node returns (sealed sum type:
//     Complete | BudgetExhausted | Failed), mirroring the LoopStrategy/StrategyRef
//     tagged-union idiom established in strategy.go.
//   - BudgetExhausted — the signal Charge returns when a debit overflows.
//   - BudgetContext — one budget scope (Charge / Remaining / ContinuesRemaining).
//   - BudgetStack — runtime push/pop stack of BudgetContext, one node per
//     recursion frame (siblings do NOT share).
//   - SpanStack — runtime push/pop stack of span ids.
//   - ExecutionContext — the one shared, mutable context for a whole nested tree.
//
// Rules enforced:
//   - A child's StrategyOutcome BudgetExhausted is an INSPECTABLE value the
//     parent reads; it does NOT auto-propagate.
//   - Charge is pure arithmetic: it debits turns steps; on success increments
//     StepsTaken and returns nil; on overflow returns a *BudgetExhausted from
//     current state WITHOUT mutating. It does NOT walk the behavior chain or
//     consume continues. BudgetUnlimited never exhausts.
//   - Each BudgetContext represents ONE scope; the allowance is the policy's own
//     Value (Unlimited = no cap).
//   - All runtime types here are NEVER serialized (no JSON tags / Marshal impls).
//
// Resolved spec ambiguities (DECIDED — see issue #123):
//  1. ExecutionContext holds a pointer to the ExecutionRegistry; RunStrategy.Run
//     threads the *ExecutionContext through. The serializable LoopStrategy /
//     StrategyRef are unaffected.
//  2. Charge is pure arithmetic (above); BudgetExhausted is a dedicated signal
//     type; the StrategyOutcome BudgetExhausted variant mirrors those fields and
//     adds PartialOutput. Output maps to string.
//  3. ContinuesUsed is an IN-MEMORY field ONLY this slice; its checkpoint
//     persistence is DEFERRED to the enforcement slice — no serialization types
//     are touched here.
//  4. Go-specific divergence (recorded on the issue): SpanStack is a stack of
//     plain string. The typed observability.SpanID lives in the observability
//     subpackage, which imports sporecore; importing it back here would form an
//     import cycle. sporecore already represents span ids as plain string in the
//     harness loop (harness.go), so SpanStack matches that. SpanStack is
//     runtime-only and never serialized, so the element type never crosses the
//     wire and cross-language consistency is preserved.

package sporecore

// ============================================================================
// StrategyOutcome — the typed result a strategy node returns
// ============================================================================

// StrategyOutcomeKind discriminates StrategyOutcome variants.
type StrategyOutcomeKind string

const (
	// StrategyOutcomeComplete: the strategy completed and produced its output.
	StrategyOutcomeComplete StrategyOutcomeKind = "complete"
	// StrategyOutcomeBudgetExhausted: the strategy's budget scope ran out of
	// allowance. Inspectable, NOT auto-propagating.
	StrategyOutcomeBudgetExhausted StrategyOutcomeKind = "budget_exhausted"
	// StrategyOutcomeFailed: the strategy halted with a harness error.
	StrategyOutcomeFailed StrategyOutcomeKind = "failed"
)

// StrategyOutcome is the typed result a strategy node returns. Runtime-only —
// NEVER serialized (a strategy outcome is an in-process value, never persisted).
//
// It is a sealed sum type with exactly one populated payload selected by Kind,
// mirroring the LoopStrategy / StrategyRef tagged-union idiom in strategy.go
// (Go has no native sum type). BudgetExhausted is distinguishable from Failed by
// callers — a child's BudgetExhausted is a value the parent INSPECTS (e.g. to
// grant a continue or escalate); it does NOT auto-propagate as a failure.
//
// Output maps to string in this codebase (mirrors RunResult).
type StrategyOutcome struct {
	Kind StrategyOutcomeKind
	// Complete is the final output, valid when Kind == StrategyOutcomeComplete.
	Complete string
	// Exhausted carries the exhaustion details, valid when
	// Kind == StrategyOutcomeBudgetExhausted.
	Exhausted *StrategyOutcomeExhausted
	// Failed is the harness error, valid when Kind == StrategyOutcomeFailed.
	Failed HarnessError
}

// StrategyOutcomeExhausted is the payload of the BudgetExhausted outcome
// variant. It mirrors the BudgetExhausted charge-signal fields and adds
// PartialOutput — any output produced before exhaustion.
type StrategyOutcomeExhausted struct {
	Policy        BudgetPolicy
	Behavior      BudgetExhaustedBehavior
	StepsTaken    uint32
	ContinuesUsed uint32
	Phase         string
	PartialOutput *string
}

// StrategyComplete builds a Complete outcome carrying output.
func StrategyComplete(output string) StrategyOutcome {
	return StrategyOutcome{Kind: StrategyOutcomeComplete, Complete: output}
}

// StrategyFailed builds a Failed outcome carrying a harness error.
func StrategyFailed(err HarnessError) StrategyOutcome {
	return StrategyOutcome{Kind: StrategyOutcomeFailed, Failed: err}
}

// StrategyBudgetExhausted promotes a charge-signal BudgetExhausted into a
// BudgetExhausted outcome, attaching any partial output produced before the
// scope ran out (partialOutput may be nil).
func StrategyBudgetExhausted(e BudgetExhausted, partialOutput *string) StrategyOutcome {
	return StrategyOutcome{
		Kind: StrategyOutcomeBudgetExhausted,
		Exhausted: &StrategyOutcomeExhausted{
			Policy:        e.Policy,
			Behavior:      e.Behavior,
			StepsTaken:    e.StepsTaken,
			ContinuesUsed: e.ContinuesUsed,
			Phase:         e.Phase,
			PartialOutput: partialOutput,
		},
	}
}

// ============================================================================
// BudgetExhausted — the signal BudgetContext.Charge returns on overflow
// ============================================================================

// BudgetExhausted is the signal BudgetContext.Charge returns when a debit would
// exceed the scope's step allowance. It captures the budget state at the moment
// of exhaustion. Runtime-only (NOT serialized).
//
// It implements error so callers may treat it as a Go error value; it promotes
// to a StrategyOutcome BudgetExhausted variant (which adds PartialOutput) at the
// strategy boundary via StrategyBudgetExhausted.
type BudgetExhausted struct {
	Policy        BudgetPolicy
	Behavior      BudgetExhaustedBehavior
	StepsTaken    uint32
	ContinuesUsed uint32
	Phase         string
}

// Error implements error.
func (e *BudgetExhausted) Error() string {
	return "budget exhausted in phase " + e.Phase
}

// ============================================================================
// BudgetContext — one budget scope
// ============================================================================

// BudgetContext is one budget scope in the strategy tree. Each recursion node
// gets its OWN BudgetContext; siblings do NOT share. Runtime-only (NOT
// serialized).
//
// The per-scope step allowance is the policy's own Value: TotalSteps / PerLoop /
// PerAttempt all expose Value as the cap for this scope; Unlimited is uncapped.
//
// ContinuesUsed is an in-memory field ONLY in this slice; its checkpoint
// persistence is deferred to the enforcement slice (see file header, #3).
type BudgetContext struct {
	Policy        BudgetPolicy
	Behavior      BudgetExhaustedBehavior
	StepsTaken    uint32
	ContinuesUsed uint32
	Phase         string
}

// NewBudgetContext constructs a fresh scope (zeroed counters) for
// policy / behavior / phase.
func NewBudgetContext(policy BudgetPolicy, behavior BudgetExhaustedBehavior, phase string) BudgetContext {
	return BudgetContext{
		Policy:   policy,
		Behavior: behavior,
		Phase:    phase,
	}
}

// allowance returns the per-scope step allowance and whether the scope is
// capped (false for Unlimited).
func (c *BudgetContext) allowance() (uint32, bool) {
	switch c.Policy.Kind {
	case BudgetUnlimited:
		return 0, false
	default:
		// TotalSteps / PerLoop / PerAttempt all expose Value as the cap.
		return c.Policy.Value, true
	}
}

// Charge debits turns steps against the scope allowance (pure arithmetic — see
// file header, #2). On success it increments StepsTaken and returns nil. If the
// debit would exceed the allowance, it returns a non-nil *BudgetExhausted from
// current state WITHOUT mutating. It does NOT walk the behavior chain or consume
// continues. Unlimited never exhausts.
func (c *BudgetContext) Charge(turns uint32) *BudgetExhausted {
	if allowance, capped := c.allowance(); capped {
		if saturatingAddU32(c.StepsTaken, turns) > allowance {
			return &BudgetExhausted{
				Policy:        c.Policy,
				Behavior:      c.Behavior,
				StepsTaken:    c.StepsTaken,
				ContinuesUsed: c.ContinuesUsed,
				Phase:         c.Phase,
			}
		}
	}
	c.StepsTaken = saturatingAddU32(c.StepsTaken, turns)
	return nil
}

// Remaining returns the steps left in this scope (allowance - StepsTaken,
// saturating) and whether the scope is capped. For Unlimited it returns
// (0, false): no cap.
func (c *BudgetContext) Remaining() (uint32, bool) {
	allowance, capped := c.allowance()
	if !capped {
		return 0, false
	}
	return saturatingSubU32(allowance, c.StepsTaken), true
}

// ContinuesRemaining returns the continues left before fall-through. For a
// Continue behavior this is MaxContinues - ContinuesUsed (saturating); for
// Escalate / Fail there are no continues, so 0.
func (c *BudgetContext) ContinuesRemaining() uint32 {
	switch c.Behavior.Kind {
	case BehaviorContinue:
		return saturatingSubU32(c.Behavior.MaxContinues, c.ContinuesUsed)
	default:
		// Escalate / Fail: no continues.
		return 0
	}
}

// ============================================================================
// BudgetStack — runtime push/pop stack of BudgetContext
// ============================================================================

// BudgetStack is a runtime push/pop stack of BudgetContext scopes — one node per
// recursion frame, pushed on descent and popped on ascent. Runtime-only (NOT
// serialized). Siblings get DISTINCT contexts and do not share state.
type BudgetStack struct {
	Stack []BudgetContext
}

// Push pushes a new scope onto the stack.
func (s *BudgetStack) Push(cx BudgetContext) {
	s.Stack = append(s.Stack, cx)
}

// Pop pops the current scope, returning it and true; (zero, false) when empty.
func (s *BudgetStack) Pop() (BudgetContext, bool) {
	n := len(s.Stack)
	if n == 0 {
		return BudgetContext{}, false
	}
	cx := s.Stack[n-1]
	s.Stack = s.Stack[:n-1]
	return cx, true
}

// Current returns a pointer to the innermost scope (mutable), or nil when empty.
func (s *BudgetStack) Current() *BudgetContext {
	n := len(s.Stack)
	if n == 0 {
		return nil
	}
	return &s.Stack[n-1]
}

// Depth returns the current stack depth (active recursion frames).
func (s *BudgetStack) Depth() int { return len(s.Stack) }

// ============================================================================
// SpanStack — runtime push/pop stack of span ids
// ============================================================================

// SpanStack is a runtime push/pop stack of span ids for observability nesting.
// Runtime-only (NOT serialized).
//
// Go-specific divergence (see file header, #4): the element type is plain string
// rather than observability.SpanID, because that typed id lives in the
// observability subpackage which imports sporecore — importing it back would
// form an import cycle. sporecore already uses plain string span ids in the
// harness loop, and SpanStack never crosses the wire.
type SpanStack struct {
	Stack []string
}

// Push pushes a span id onto the stack.
func (s *SpanStack) Push(id string) {
	s.Stack = append(s.Stack, id)
}

// Pop pops the current span id, returning it and true; ("", false) when empty.
func (s *SpanStack) Pop() (string, bool) {
	n := len(s.Stack)
	if n == 0 {
		return "", false
	}
	id := s.Stack[n-1]
	s.Stack = s.Stack[:n-1]
	return id, true
}

// Depth returns the current stack depth.
func (s *SpanStack) Depth() int { return len(s.Stack) }

// ============================================================================
// ExecutionContext — the one shared, mutable runtime context
// ============================================================================

// ExecutionContext is the one shared, mutable runtime context threaded through a
// whole nested strategy tree. It holds a pointer to the ExecutionRegistry for
// the duration of the run. Runtime-only — NEVER serialized.
//
// Stream is an optional StreamSink (nil when absent).
type ExecutionContext struct {
	// Registry is the borrowed handle registry
	// (agents/toolsets/schemas/custom strategies). #120.
	Registry *ExecutionRegistry
	// Budgets is the per-scope budget stack, pushed/popped through recursion.
	Budgets BudgetStack
	// Usage is the aggregated token/cost usage across the whole tree.
	Usage AggregateUsage
	// Session is the conversation/session state round-tripped across the tree.
	Session SessionState
	// Spans is the span stack for observability nesting.
	Spans SpanStack
	// Stream is an optional streaming sink for emitted events (nil = none). It is
	// single-use: a leaf / combinator takes it once via takeStream when driving a
	// sub-loop, suppressing it for the rest of the recursion (#124).
	Stream StreamSink
	// Executor is the harness primitives the per-variant Run bodies delegate to
	// (#124). Nil only for the scaffold/unit fixtures that exercise the runtime
	// context without a real harness (the recursion stub tests).
	Executor StrategyExecutor
	// Scratch is the per-run mutable orchestration state threaded across the
	// recursive strategy tree (#124). Runtime-only.
	Scratch RunScratch
}

// NewExecutionContext returns a fresh context bound to registry, with empty
// stacks and zero-value usage/session and no stream sink.
func NewExecutionContext(registry *ExecutionRegistry) *ExecutionContext {
	return &ExecutionContext{Registry: registry}
}

// ============================================================================
// saturating uint32 arithmetic helpers
// ============================================================================

// saturatingAddU32 returns a+b, clamped at math.MaxUint32 on overflow.
func saturatingAddU32(a, b uint32) uint32 {
	sum := a + b
	if sum < a {
		return ^uint32(0)
	}
	return sum
}

// saturatingSubU32 returns a-b, clamped at 0 on underflow.
func saturatingSubU32(a, b uint32) uint32 {
	if b > a {
		return 0
	}
	return a - b
}
