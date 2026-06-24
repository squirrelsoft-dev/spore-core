// ExecutionRegistry — runtime resolution of serializable strategy handles
// (Composable Execution A.3, issue #120; part of the #117–#131 refactor).
//
// # Types
//   - ExecutionRegistry — six maps of concrete collaborators keyed by string:
//     agents, toolsets, schemas, verifiers, metricEvaluators (#124, Q2), custom
//     (custom strategies). Trait objects never serialize, so the registry is NOT
//     JSON-(un)marshalled.
//   - ExecutionRegistryBuilder — fluent assembler mirroring the Rust builder.
//   - StrategyResolution — the result of resolving a StrategyRef: either a
//     built-in LoopStrategy or a custom RunStrategy.
//   - EscalationMode — the HITL-vs-AFK config knob (PRD goal #7).
//   - HarnessError — the recoverable/startup error interface added this slice:
//     StrategyNotFoundError (recoverable) + UnresolvedHandleError (startup
//     validation). Both are JSON-(un)marshalled byte-identically to the Rust
//     serde shapes (fixtures/harness/registry_errors.json).
//
// # Methods
//   - ResolveAgent / ResolveToolset / ResolveSchema / ResolveVerifier — pure
//     lookups, each Ref type maps to exactly ONE map (SchemaRef → schemas).
//     They use the (value, ok) idiom.
//   - ResolveStrategy — (StrategyResolution, error); a missing Custom key
//     returns the recoverable StrategyNotFoundError.
//   - RegisterStrategy — register a custom RunStrategy at startup.
//   - Validate — walks a Task's strategy tree, returning the FIRST unresolved
//     handle as UnresolvedHandleError (or StrategyNotFoundError for a missing
//     custom key). Called at the entry of StandardHarness.Run so an unresolved
//     handle is a STARTUP error, before the first turn.
//
// # Rules enforced
//   - Unresolved handle (missing agent/toolset/schema) → startup error before
//     the first turn (UnresolvedHandleError).
//   - A missing StrategyRef.Custom key → recoverable StrategyNotFoundError,
//     never a panic.
//   - Resume re-resolves every handle from the registry with no
//     reconfiguration: trait objects never enter the serialized Task, only
//     string handles do.
//   - RegisterStrategy makes a custom strategy resolvable by key.
//   - HITL-vs-AFK escalation is selectable via EscalationMode on HarnessConfig,
//     not hardcoded.
//
// # Resolutions applied (do not re-litigate — pinned in #120)
//   - Scope = ADDITIVE (Option B). This slice ADDS the registry +
//     EscalationMode to HarnessConfig; it does NOT remove the four
//     single-collaborator fields (Agent, Verifier, PlannerAgent,
//     EvaluatorAgent) nor touch the executor consumption sites. Those four
//     carry a "superseded by ExecutionRegistry" doc note; physical removal +
//     executor migration to registry resolution lands in #124.
//   - The registry has exactly FIVE maps (no sixth).
//   - EscalationMode has NO baked-in default; the harness wiring picks an
//     explicit default (SurfaceToHuman).
//   - EscalationMode is STORED only this slice (#130 consumes it) and is NOT
//     part of the serialized Task → no fixture.

package sporecore

import (
	"encoding/json"
	"fmt"
)

// ============================================================================
// EscalationMode — HITL-vs-AFK config knob
// ============================================================================

// EscalationMode is the HITL-vs-AFK escalation knob (PRD goal #7: local vs.
// prod differ only by config). It selects what budget escalation does: surface
// to a human, fail autonomously, or keep working under an auto-granted cap
// (SC-5). Stored on HarnessConfig; consumed at every Escalate resolution site
// (#130, SC-5).
//
// No baked-in default by design (mirrors the budget-types discipline); the
// harness wiring picks an explicit default (EscalationSurfaceToHuman). It has a
// tagged JSON shape ({"kind":"surface_to_human"} / {"kind":"autonomous"} /
// {"kind":"auto_continue","max_grants":N,"steps_per_grant":M}) for symmetry with
// the other harness enums, but it is NOT placed on the serialized Task (no
// fixture serializes the mode).
//
// SC-5: the AutoContinue kind carries MaxGrants / StepsPerGrant plus an OnGrant
// callback. The callback is runtime-only — NEVER serialized (funcs aren't
// JSON-serializable). EscalationMode is therefore compared on Kind ONLY (funcs
// aren't comparable — never use == on the struct).
type EscalationMode struct {
	Kind EscalationModeKind
	// MaxGrants is the maximum number of auto-grants per exhausted scope before
	// falling through to the autonomous terminal (EscalationAutoContinue). 0
	// behaves like Autonomous (no grants).
	MaxGrants uint32
	// StepsPerGrant is the steps granted each time (EscalationAutoContinue). A
	// grant raises the exhausted scope's cap to steps_taken + steps_per_grant so
	// the loop gets exactly this many more steps after the exhaustion point.
	StepsPerGrant uint32
	// OnGrant is an optional per-grant observer fired once per auto-grant
	// (EscalationAutoContinue). Runtime-only — NEVER serialized.
	OnGrant func(AutoGrantInfo)
}

// EscalationModeKind discriminates EscalationMode variants. Wire values are
// snake_case to match the cross-language serde shape.
type EscalationModeKind string

const (
	// EscalationSurfaceToHuman pauses and surfaces budget escalation to a human
	// (HITL).
	EscalationSurfaceToHuman EscalationModeKind = "surface_to_human"
	// EscalationAutonomous fails the run autonomously (AFK / prod): the partial
	// is propagated and the run stops.
	EscalationAutonomous EscalationModeKind = "autonomous"
	// EscalationAutoContinue is "autonomous but capped" (SC-5): at an escalation
	// point, auto-grant StepsPerGrant more steps and KEEP WORKING in-process, up
	// to MaxGrants times, firing OnGrant per grant. Once the grants are spent it
	// falls through to the same terminal as EscalationAutonomous. This is the
	// keep-working-to-completion-but-cap-at-N policy consumers otherwise
	// hand-roll around the harness.
	EscalationAutoContinue EscalationModeKind = "auto_continue"
)

// AutoGrantInfo is the detail handed to an EscalationAutoContinue OnGrant
// callback each time the harness auto-grants more budget at an escalation point
// (SC-5).
type AutoGrantInfo struct {
	// GrantNumber is the 1-based index of this grant within the exhausted scope
	// (1..=MaxGrants).
	GrantNumber uint32
	// StepsGranted is the steps granted this round (the mode's StepsPerGrant).
	StepsGranted uint32
	// Phase is the budget scope phase that exhausted (e.g. "react",
	// "plan_execute").
	Phase string
}

// SurfaceToHumanEscalation returns the HITL escalation mode.
func SurfaceToHumanEscalation() EscalationMode {
	return EscalationMode{Kind: EscalationSurfaceToHuman}
}

// AutonomousEscalation returns the AFK / autonomous escalation mode.
func AutonomousEscalation() EscalationMode {
	return EscalationMode{Kind: EscalationAutonomous}
}

// AutoContinueEscalation returns the "autonomous but capped" escalation mode
// (SC-5): at each Escalate site, auto-grant stepsPerGrant more steps and keep
// working in-process up to maxGrants times, firing onGrant per grant. onGrant
// may be nil.
func AutoContinueEscalation(maxGrants, stepsPerGrant uint32, onGrant func(AutoGrantInfo)) EscalationMode {
	return EscalationMode{
		Kind:          EscalationAutoContinue,
		MaxGrants:     maxGrants,
		StepsPerGrant: stepsPerGrant,
		OnGrant:       onGrant,
	}
}

// MarshalJSON serialises EscalationMode as a "kind"-tagged object. The
// AutoContinue kind also emits max_grants / steps_per_grant; OnGrant is NEVER
// serialized (funcs aren't JSON-serializable).
func (m EscalationMode) MarshalJSON() ([]byte, error) {
	switch m.Kind {
	case EscalationSurfaceToHuman, EscalationAutonomous:
		return json.Marshal(struct {
			Kind EscalationModeKind `json:"kind"`
		}{m.Kind})
	case EscalationAutoContinue:
		return json.Marshal(struct {
			Kind          EscalationModeKind `json:"kind"`
			MaxGrants     uint32             `json:"max_grants"`
			StepsPerGrant uint32             `json:"steps_per_grant"`
		}{m.Kind, m.MaxGrants, m.StepsPerGrant})
	default:
		return nil, fmt.Errorf("EscalationMode: unknown kind %q", m.Kind)
	}
}

// UnmarshalJSON decodes the "kind"-tagged form. OnGrant is never restored (it is
// not serialized).
func (m *EscalationMode) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind          EscalationModeKind `json:"kind"`
		MaxGrants     uint32             `json:"max_grants"`
		StepsPerGrant uint32             `json:"steps_per_grant"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	switch probe.Kind {
	case EscalationSurfaceToHuman, EscalationAutonomous:
		m.Kind = probe.Kind
		return nil
	case EscalationAutoContinue:
		m.Kind = probe.Kind
		m.MaxGrants = probe.MaxGrants
		m.StepsPerGrant = probe.StepsPerGrant
		return nil
	default:
		return fmt.Errorf("EscalationMode: unknown kind %q", probe.Kind)
	}
}

// ============================================================================
// HarnessError — recoverable / startup-validation errors (issue #120)
// ============================================================================

// HarnessError is the marker interface for the #120 registry errors. Both
// concrete variants implement error; callers match with errors.As against the
// concrete types. Mirrors the Rust HarnessError enum (serde tag "kind",
// variant names emitted verbatim — PascalCase).
type HarnessError interface {
	error
	isHarnessError()
}

// StrategyNotFoundError is returned when a StrategyRef.Custom(key) references a
// custom strategy that is not registered in the ExecutionRegistry's custom map.
// RECOVERABLE — returned, never panics (same pattern as a missing AgentRef).
//
// Serializes as {"kind":"StrategyNotFound","key":"<key>"}.
type StrategyNotFoundError struct {
	Key string
}

func (e *StrategyNotFoundError) isHarnessError() {}

// Error implements error. Message mirrors the Rust display impl.
func (e *StrategyNotFoundError) Error() string {
	return fmt.Sprintf("custom strategy not found: %s", e.Key)
}

// MarshalJSON serialises as {"kind":"StrategyNotFound","key":...}.
func (e *StrategyNotFoundError) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Kind string `json:"kind"`
		Key  string `json:"key"`
	}{"StrategyNotFound", e.Key})
}

// UnmarshalJSON decodes the "StrategyNotFound" form.
func (e *StrategyNotFoundError) UnmarshalJSON(data []byte) error {
	var probe struct {
		Key string `json:"key"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	e.Key = probe.Key
	return nil
}

// UnresolvedHandleError is returned when a serializable handle
// (AgentRef/ToolsetRef/SchemaRef) references an entry absent from the
// ExecutionRegistry. The STARTUP-validation error: surfaced before the first
// turn.
//
// The handle category is the Go field Kind, but it serializes as the JSON key
// "handle_kind" (matching Rust's rename) so it does not collide with the
// variant discriminant "kind". Serializes as
// {"kind":"UnresolvedHandle","handle_kind":"<kind>","key":"<key>"}.
type UnresolvedHandleError struct {
	// Kind is the handle category: "agent", "toolset", or "schema".
	Kind string
	Key  string
}

func (e *UnresolvedHandleError) isHarnessError() {}

// Error implements error. Message mirrors the Rust display impl.
func (e *UnresolvedHandleError) Error() string {
	return fmt.Sprintf("unresolved %s handle: %s", e.Kind, e.Key)
}

// MarshalJSON serialises as {"kind":"UnresolvedHandle","handle_kind":...,"key":...}.
func (e *UnresolvedHandleError) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Kind       string `json:"kind"`
		HandleKind string `json:"handle_kind"`
		Key        string `json:"key"`
	}{"UnresolvedHandle", e.Kind, e.Key})
}

// UnmarshalJSON decodes the "UnresolvedHandle" form (handle_kind → Kind).
func (e *UnresolvedHandleError) UnmarshalJSON(data []byte) error {
	var probe struct {
		HandleKind string `json:"handle_kind"`
		Key        string `json:"key"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	e.Kind = probe.HandleKind
	e.Key = probe.Key
	return nil
}

// UnmarshalHarnessError decodes a #120 HarnessError from its "kind"-tagged JSON
// form, dispatching to the concrete variant. Used by fixture replay.
func UnmarshalHarnessError(data []byte) (HarnessError, error) {
	var probe struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, err
	}
	switch probe.Kind {
	case "InvalidConfiguration":
		e := &InvalidConfigurationError{}
		if err := json.Unmarshal(data, e); err != nil {
			return nil, err
		}
		return e, nil
	case "StrategyNotFound":
		e := &StrategyNotFoundError{}
		if err := json.Unmarshal(data, e); err != nil {
			return nil, err
		}
		return e, nil
	case "UnresolvedHandle":
		e := &UnresolvedHandleError{}
		if err := json.Unmarshal(data, e); err != nil {
			return nil, err
		}
		return e, nil
	default:
		return nil, fmt.Errorf("HarnessError: unknown kind %q", probe.Kind)
	}
}

// Compile-time interface checks.
var (
	_ HarnessError = (*StrategyNotFoundError)(nil)
	_ HarnessError = (*UnresolvedHandleError)(nil)
)

// ============================================================================
// StrategyResolution — the result of resolving a StrategyRef
// ============================================================================

// StrategyResolutionKind discriminates StrategyResolution variants.
type StrategyResolutionKind string

const (
	// ResolutionBuiltIn marks a built-in LoopStrategy tree resolution.
	ResolutionBuiltIn StrategyResolutionKind = "built_in"
	// ResolutionCustom marks a custom RunStrategy resolution.
	ResolutionCustom StrategyResolutionKind = "custom"
)

// StrategyResolution is the result of resolving a StrategyRef against an
// ExecutionRegistry: either the built-in LoopStrategy tree (BuiltIn) or the
// custom RunStrategy looked up in ExecutionRegistry.custom (Custom). Exactly one
// field is populated, selected by Kind.
type StrategyResolution struct {
	Kind    StrategyResolutionKind
	BuiltIn *LoopStrategy
	Custom  RunStrategy
}

// ============================================================================
// ExecutionRegistry — runtime resolver
// ============================================================================

// ExecutionRegistry maps serializable string handles (and StrategyRef.Custom
// keys) to concrete collaborators. See the file header for the full
// type/method/rule documentation.
//
// Trait objects never serialize, so this type is NOT JSON-(un)marshalled. Build
// one with NewExecutionRegistry (empty) or an ExecutionRegistryBuilder.
type ExecutionRegistry struct {
	agents    map[string]Agent
	toolsets  map[string]ToolRegistry
	schemas   map[string]json.RawMessage
	verifiers map[string]Verifier
	// metricEvaluators is the SIXTH map (#124, Q2): HillClimbing metric
	// evaluators, keyed by the same string HillClimbingConfig.Evaluator carries on
	// the wire. Runtime-only (never serialized) like the other maps; keeping it
	// distinct from agents preserves the metric-evaluator wire string while
	// resolving it to a MetricEvaluator rather than an Agent.
	metricEvaluators map[string]MetricEvaluator
	custom           map[string]RunStrategy
}

// NewExecutionRegistry returns an empty registry (no entries in any of the five
// maps). Maps are lazily allocated on first write.
func NewExecutionRegistry() ExecutionRegistry {
	return ExecutionRegistry{}
}

// IsEmpty reports whether no entries exist in any of the five maps. Lets the
// harness skip startup validation for callers that never wire a registry (Option
// B additive scope — they still use the superseded single-collaborator fields).
func (r ExecutionRegistry) IsEmpty() bool {
	return len(r.agents) == 0 &&
		len(r.toolsets) == 0 &&
		len(r.schemas) == 0 &&
		len(r.verifiers) == 0 &&
		len(r.metricEvaluators) == 0 &&
		len(r.custom) == 0
}

// ResolveAgent resolves an AgentRef to its registered agent. ok is false when
// absent.
func (r ExecutionRegistry) ResolveAgent(ref AgentRef) (Agent, bool) {
	a, ok := r.agents[string(ref)]
	return a, ok
}

// ResolveToolset resolves a ToolsetRef to its registered toolset. ok is false
// when absent.
func (r ExecutionRegistry) ResolveToolset(ref ToolsetRef) (ToolRegistry, bool) {
	t, ok := r.toolsets[string(ref)]
	return t, ok
}

// ResolveSchema resolves a SchemaRef to its registered JSON schema. ok is false
// when absent. (SchemaRef maps to the schemas map.)
func (r ExecutionRegistry) ResolveSchema(ref SchemaRef) (json.RawMessage, bool) {
	s, ok := r.schemas[string(ref)]
	return s, ok
}

// ResolveVerifier resolves a verifier key to its registered verifier. ok is
// false when absent.
func (r ExecutionRegistry) ResolveVerifier(key string) (Verifier, bool) {
	v, ok := r.verifiers[key]
	return v, ok
}

// ResolveMetricEvaluator resolves a metric-evaluator key (the string
// HillClimbingConfig.Evaluator carries, #124 Q2) to its registered
// MetricEvaluator. ok is false when absent. The wire string is identical to the
// legacy AgentRef; only the resolution target differs (the sixth
// metricEvaluators map).
func (r ExecutionRegistry) ResolveMetricEvaluator(key string) (MetricEvaluator, bool) {
	m, ok := r.metricEvaluators[key]
	return m, ok
}

// ResolveStrategy resolves a StrategyRef: a BuiltIn ref yields the built-in
// tree; a Custom ref looks up the custom map and returns the recoverable
// *StrategyNotFoundError when the key is absent (never panics).
func (r ExecutionRegistry) ResolveStrategy(ref StrategyRef) (StrategyResolution, error) {
	switch ref.Kind {
	case StrategyRefBuiltIn:
		return StrategyResolution{Kind: ResolutionBuiltIn, BuiltIn: ref.BuiltIn}, nil
	case StrategyRefCustom:
		s, ok := r.custom[ref.Custom]
		if !ok {
			return StrategyResolution{}, &StrategyNotFoundError{Key: ref.Custom}
		}
		return StrategyResolution{Kind: ResolutionCustom, Custom: s}, nil
	default:
		return StrategyResolution{}, fmt.Errorf("StrategyRef: unknown kind %q", ref.Kind)
	}
}

// RegisterStrategy registers (or replaces, last-wins) a custom strategy under
// key.
func (r *ExecutionRegistry) RegisterStrategy(key string, s RunStrategy) {
	if r.custom == nil {
		r.custom = make(map[string]RunStrategy)
	}
	r.custom[key] = s
}

// Validate checks that every handle referenced by task.LoopStrategy resolves
// against this registry. It walks the strategy tree and returns the FIRST
// unresolved handle as *UnresolvedHandleError (or *StrategyNotFoundError for a
// missing custom key). Returns nil when the whole tree resolves. Called at the
// entry of StandardHarness.Run so an unresolved handle is a startup error.
func (r ExecutionRegistry) Validate(task Task) error {
	return r.walkStrategy(&task.LoopStrategy)
}

// walkStrategy recursively walks a LoopStrategy, checking every child handle.
func (r ExecutionRegistry) walkStrategy(ls *LoopStrategy) error {
	if ls == nil {
		return nil
	}
	switch ls.Kind {
	case StrategyReAct:
		if ls.ReActCfg == nil {
			return nil
		}
		if err := r.checkAgent(ls.ReActCfg.Agent); err != nil {
			return err
		}
		if err := r.checkToolset(ls.ReActCfg.Toolset); err != nil {
			return err
		}
		if ls.ReActCfg.Output != nil {
			if err := r.checkSchema(*ls.ReActCfg.Output); err != nil {
				return err
			}
		}
		return nil
	case StrategyPlanExecute:
		if ls.PlanExecute == nil {
			return nil
		}
		// A.5 (#124, Q3): the plan slot is STRUCTURED — it must yield a task
		// graph. A bare ReAct there needs an output schema.
		if err := checkStructuredSlot(ls.PlanExecute.Plan, "plan"); err != nil {
			return err
		}
		if err := r.walkStrategy(ls.PlanExecute.Plan); err != nil {
			return err
		}
		return r.walkStrategy(ls.PlanExecute.Execute)
	case StrategySelfVerifying:
		if ls.SelfVerify == nil {
			return nil
		}
		// A.5: the inner (worker) slot is STRUCTURED — its result must be
		// evaluable. A bare ReAct worker needs an output schema.
		if err := checkStructuredSlot(ls.SelfVerify.Inner, "worker"); err != nil {
			return err
		}
		if err := r.walkStrategy(ls.SelfVerify.Inner); err != nil {
			return err
		}
		// #124 Q1: the evaluator's wire string (a SchemaRef) is the VERIFIER
		// registry key — resolved against the verifiers map (NO wire change).
		return r.checkVerifier(ls.SelfVerify.Evaluator)
	case StrategyRalph:
		if ls.Ralph == nil {
			return nil
		}
		if err := r.walkStrategy(ls.Ralph.Inner); err != nil {
			return err
		}
		return r.checkAgent(ls.Ralph.Agent)
	case StrategyHillClimbing:
		if ls.HillClimbing == nil {
			return nil
		}
		// A.5: the inner (propose) slot is STRUCTURED — it must yield a
		// candidate. A bare ReAct proposer needs an output schema.
		if err := checkStructuredSlot(ls.HillClimbing.Inner, "propose"); err != nil {
			return err
		}
		if err := r.walkStrategy(ls.HillClimbing.Inner); err != nil {
			return err
		}
		// #124 Q2: the evaluator's wire string is resolved against the sixth
		// metricEvaluators map (not agents).
		return r.checkMetricEvaluator(ls.HillClimbing.Evaluator)
	default:
		return nil
	}
}

// checkStructuredSlot enforces the A.5 output contract (#124, Q3): a bare ReAct
// feeding a STRUCTURED slot (plan ⇒ task graph, propose ⇒ candidate, worker ⇒
// evaluable result) MUST declare output = Some(schema). A combinator child
// carries its own contract, so this check applies only to the leaf. Returns an
// InvalidConfigurationError naming the offending slot.
func checkStructuredSlot(slot *LoopStrategy, slotName string) error {
	if slot != nil && slot.Kind == StrategyReAct && slot.ReActCfg != nil && slot.ReActCfg.Output == nil {
		return &InvalidConfigurationError{
			Message: fmt.Sprintf(
				"a bare ReAct in the structured `%s` slot requires `output = Some(schema)` so the slot yields a typed result",
				slotName,
			),
		}
	}
	return nil
}

func (r ExecutionRegistry) checkAgent(ref AgentRef) error {
	if _, ok := r.agents[string(ref)]; ok {
		return nil
	}
	return &UnresolvedHandleError{Kind: "agent", Key: string(ref)}
}

func (r ExecutionRegistry) checkToolset(ref ToolsetRef) error {
	if _, ok := r.toolsets[string(ref)]; ok {
		return nil
	}
	return &UnresolvedHandleError{Kind: "toolset", Key: string(ref)}
}

func (r ExecutionRegistry) checkSchema(ref SchemaRef) error {
	if _, ok := r.schemas[string(ref)]; ok {
		return nil
	}
	return &UnresolvedHandleError{Kind: "schema", Key: string(ref)}
}

// checkVerifier enforces #124 Q1: a SelfVerifying evaluator (a SchemaRef on the
// wire) resolves against the verifiers map.
func (r ExecutionRegistry) checkVerifier(ref SchemaRef) error {
	if _, ok := r.verifiers[string(ref)]; ok {
		return nil
	}
	return &UnresolvedHandleError{Kind: "verifier", Key: string(ref)}
}

// checkMetricEvaluator enforces #124 Q2: a HillClimbing evaluator (an AgentRef
// on the wire) resolves against the sixth metricEvaluators map.
func (r ExecutionRegistry) checkMetricEvaluator(ref AgentRef) error {
	if _, ok := r.metricEvaluators[string(ref)]; ok {
		return nil
	}
	return &UnresolvedHandleError{Kind: "metric_evaluator", Key: string(ref)}
}

// ============================================================================
// ExecutionRegistryBuilder — fluent assembler
// ============================================================================

// ExecutionRegistryBuilder is the fluent assembler for an ExecutionRegistry.
type ExecutionRegistryBuilder struct {
	registry ExecutionRegistry
}

// intoBuilder returns a builder seeded with a SHALLOW COPY of this registry's
// maps, so a caller (e.g. HarnessConfig's per-key convenience setters) can add
// more entries before re-Build-ing without mutating the source registry's maps.
// Mirrors the Rust into_builder seam.
func (r ExecutionRegistry) intoBuilder() *ExecutionRegistryBuilder {
	b := &ExecutionRegistryBuilder{}
	if len(r.agents) > 0 {
		b.registry.agents = make(map[string]Agent, len(r.agents))
		for k, v := range r.agents {
			b.registry.agents[k] = v
		}
	}
	if len(r.toolsets) > 0 {
		b.registry.toolsets = make(map[string]ToolRegistry, len(r.toolsets))
		for k, v := range r.toolsets {
			b.registry.toolsets[k] = v
		}
	}
	if len(r.schemas) > 0 {
		b.registry.schemas = make(map[string]json.RawMessage, len(r.schemas))
		for k, v := range r.schemas {
			b.registry.schemas[k] = v
		}
	}
	if len(r.verifiers) > 0 {
		b.registry.verifiers = make(map[string]Verifier, len(r.verifiers))
		for k, v := range r.verifiers {
			b.registry.verifiers[k] = v
		}
	}
	if len(r.metricEvaluators) > 0 {
		b.registry.metricEvaluators = make(map[string]MetricEvaluator, len(r.metricEvaluators))
		for k, v := range r.metricEvaluators {
			b.registry.metricEvaluators[k] = v
		}
	}
	if len(r.custom) > 0 {
		b.registry.custom = make(map[string]RunStrategy, len(r.custom))
		for k, v := range r.custom {
			b.registry.custom[k] = v
		}
	}
	return b
}

// NewExecutionRegistryBuilder starts a fluent builder over an empty registry.
func NewExecutionRegistryBuilder() *ExecutionRegistryBuilder {
	return &ExecutionRegistryBuilder{registry: NewExecutionRegistry()}
}

// Agent registers an agent under key (last-wins).
func (b *ExecutionRegistryBuilder) Agent(key string, agent Agent) *ExecutionRegistryBuilder {
	if b.registry.agents == nil {
		b.registry.agents = make(map[string]Agent)
	}
	b.registry.agents[key] = agent
	return b
}

// Toolset registers a toolset under key (last-wins).
func (b *ExecutionRegistryBuilder) Toolset(key string, toolset ToolRegistry) *ExecutionRegistryBuilder {
	if b.registry.toolsets == nil {
		b.registry.toolsets = make(map[string]ToolRegistry)
	}
	b.registry.toolsets[key] = toolset
	return b
}

// Schema registers a JSON schema under key (last-wins).
func (b *ExecutionRegistryBuilder) Schema(key string, schema json.RawMessage) *ExecutionRegistryBuilder {
	if b.registry.schemas == nil {
		b.registry.schemas = make(map[string]json.RawMessage)
	}
	b.registry.schemas[key] = schema
	return b
}

// Verifier registers a verifier under key (last-wins).
func (b *ExecutionRegistryBuilder) Verifier(key string, verifier Verifier) *ExecutionRegistryBuilder {
	if b.registry.verifiers == nil {
		b.registry.verifiers = make(map[string]Verifier)
	}
	b.registry.verifiers[key] = verifier
	return b
}

// MetricEvaluator registers a metric evaluator under key (#124, Q2 — the sixth
// map; last-wins).
func (b *ExecutionRegistryBuilder) MetricEvaluator(key string, evaluator MetricEvaluator) *ExecutionRegistryBuilder {
	if b.registry.metricEvaluators == nil {
		b.registry.metricEvaluators = make(map[string]MetricEvaluator)
	}
	b.registry.metricEvaluators[key] = evaluator
	return b
}

// RegisterStrategy registers a custom strategy under key (last-wins).
func (b *ExecutionRegistryBuilder) RegisterStrategy(key string, strategy RunStrategy) *ExecutionRegistryBuilder {
	b.registry.RegisterStrategy(key, strategy)
	return b
}

// fillDefaultAgent registers agent under the DEFAULT empty-string key ONLY if
// that key is not already wired (#124 migration seam). NewStandardHarness folds
// its default config.Agent here so bare ReactConfig leaves (empty AgentRef)
// resolve to it. An explicitly-registered "" agent wins.
func (b *ExecutionRegistryBuilder) fillDefaultAgent(agent Agent) *ExecutionRegistryBuilder {
	if agent == nil {
		return b
	}
	if b.registry.agents == nil {
		b.registry.agents = make(map[string]Agent)
	}
	if _, ok := b.registry.agents[""]; !ok {
		b.registry.agents[""] = agent
	}
	return b
}

// fillDefaultToolset registers toolset under the empty key only if absent
// (#124), as fillDefaultAgent but for the default config.ToolRegistry.
func (b *ExecutionRegistryBuilder) fillDefaultToolset(toolset ToolRegistry) *ExecutionRegistryBuilder {
	if toolset == nil {
		return b
	}
	if b.registry.toolsets == nil {
		b.registry.toolsets = make(map[string]ToolRegistry)
	}
	if _, ok := b.registry.toolsets[""]; !ok {
		b.registry.toolsets[""] = toolset
	}
	return b
}

// fillToolset registers toolset under key ONLY if that key is not already wired
// (Issue 2: per-node toolset scoping). NewStandardHarness calls this for each
// per-key catalogue wired via HarnessBuilder.ToolsetTools, so a leaf carrying
// that non-empty toolset handle RESOLVES against the registry (Validate runs
// checkToolset at run entry) without the caller manually registering a
// placeholder. The registry VALUE is never dispatched (dispatch goes through
// HarnessConfig.ToolsetCatalogues); an explicitly-registered toolset under the
// same key wins. Mirrors Rust ExecutionRegistryBuilder.fill_toolset.
func (b *ExecutionRegistryBuilder) fillToolset(key string, toolset ToolRegistry) *ExecutionRegistryBuilder {
	if toolset == nil {
		return b
	}
	if b.registry.toolsets == nil {
		b.registry.toolsets = make(map[string]ToolRegistry)
	}
	if _, ok := b.registry.toolsets[key]; !ok {
		b.registry.toolsets[key] = toolset
	}
	return b
}

// fillDefaultSchema registers an empty JSON schema under the empty key only if
// absent (#124), so a structured-slot leaf carrying output SchemaRef("") passes
// A.5 validation. Mirrors the Rust default-key schema fold.
func (b *ExecutionRegistryBuilder) fillDefaultSchema() *ExecutionRegistryBuilder {
	if b.registry.schemas == nil {
		b.registry.schemas = make(map[string]json.RawMessage)
	}
	if _, ok := b.registry.schemas[""]; !ok {
		b.registry.schemas[""] = json.RawMessage(`{}`)
	}
	return b
}

// fillDefaultVerifier registers verifier under the empty key only if absent
// (#124), for a default SelfVerifying verifier (the config.Verifier fold).
func (b *ExecutionRegistryBuilder) fillDefaultVerifier(verifier Verifier) *ExecutionRegistryBuilder {
	if verifier == nil {
		return b
	}
	if b.registry.verifiers == nil {
		b.registry.verifiers = make(map[string]Verifier)
	}
	if _, ok := b.registry.verifiers[""]; !ok {
		b.registry.verifiers[""] = verifier
	}
	return b
}

// fillDefaultMetricEvaluator registers evaluator under the empty key only if
// absent (#124), for a default HillClimbing metric evaluator (the
// config.MetricEvaluator fold).
func (b *ExecutionRegistryBuilder) fillDefaultMetricEvaluator(evaluator MetricEvaluator) *ExecutionRegistryBuilder {
	if evaluator == nil {
		return b
	}
	if b.registry.metricEvaluators == nil {
		b.registry.metricEvaluators = make(map[string]MetricEvaluator)
	}
	if _, ok := b.registry.metricEvaluators[""]; !ok {
		b.registry.metricEvaluators[""] = evaluator
	}
	return b
}

// Build finishes and returns the assembled ExecutionRegistry.
func (b *ExecutionRegistryBuilder) Build() ExecutionRegistry {
	return b.registry
}
