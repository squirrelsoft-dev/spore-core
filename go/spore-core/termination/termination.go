// Package termination — issue #13 `TerminationPolicy`: evaluate after each
// turn whether to continue, halt with success, halt with failure, or halt
// because a budget limit was breached.
//
// See `docs/harness-engineering-concepts.md` § "TerminationPolicy" for the
// authoritative rules. This package ships:
//   - The full TerminationDecision / TerminationFailureReason / BudgetValue
//     surface from the spec.
//   - The CompletionCheck interface and standard checks
//     (NullCompletionCheck, FixedCompletionCheck).
//   - StandardTerminationPolicy — the reference policy that runs budget
//     first, then sensor halts, then the injected CompletionCheck.
//
// Rules enforced
//   - The model's `AgentClaimsDone` is one input, not the decision.
//   - Budget limits are unconditional hard stops — evaluated before anything
//     else and regardless of `AgentClaimsDone`.
//   - HaltFailure carries a typed TerminationFailureReason; it cannot be a
//     free string.
//   - The CompletionCheck is injected at construction time — the policy
//     itself is domain-agnostic.
//   - If `!AgentClaimsDone`, always Continue (after budget check).
//   - When `AgentClaimsDone`, any sensor result with SensorOutcome `halt`
//     becomes TerminationFailureReason UnrecoverableSensorHalt.
//   - CompletionCheck.Check returning (reason, false) ⇒ Continue (the harness
//     re-injects the reason).
//   - CompletionCheck.Check returning (_, true) ⇒ HaltSuccess using the
//     agent's last response as summary (empty string if absent).
//   - HumanHalted is reserved for the harness; the policy never produces it.
package termination

import (
	"context"
	"encoding/json"
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/middleware"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/sensor"
)

// ============================================================================
// BudgetValue (tagged union via Kind)
// ============================================================================

// BudgetValueKind discriminates BudgetValue variants. Wire tag values match
// the Rust reference's serde rename_all = "snake_case".
type BudgetValueKind string

const (
	// BudgetValueTurns — a turn count.
	BudgetValueTurns BudgetValueKind = "turns"
	// BudgetValueTokens — a token count (input, output, or total).
	BudgetValueTokens BudgetValueKind = "tokens"
	// BudgetValueDuration — a wall-clock duration, encoded as integer seconds.
	BudgetValueDuration BudgetValueKind = "duration"
	// BudgetValueUSD — a cost in USD.
	BudgetValueUSD BudgetValueKind = "usd"
)

// BudgetValue carries the measured budget quantity at the moment of halt so
// observers can compute overshoot against the configured limit.
//
// Exactly one of TurnsValue / TokensValue / DurationSecs / USDValue is
// meaningful, selected by Kind.
type BudgetValue struct {
	Kind         BudgetValueKind
	TurnsValue   uint32
	TokensValue  uint64
	DurationSecs uint64
	USDValue     float64
}

// NewBudgetValueTurns builds a Turns budget value.
func NewBudgetValueTurns(v uint32) BudgetValue {
	return BudgetValue{Kind: BudgetValueTurns, TurnsValue: v}
}

// NewBudgetValueTokens builds a Tokens budget value.
func NewBudgetValueTokens(v uint64) BudgetValue {
	return BudgetValue{Kind: BudgetValueTokens, TokensValue: v}
}

// NewBudgetValueDuration builds a Duration budget value (seconds on the wire).
func NewBudgetValueDuration(secs uint64) BudgetValue {
	return BudgetValue{Kind: BudgetValueDuration, DurationSecs: secs}
}

// NewBudgetValueUSD builds a USD budget value.
func NewBudgetValueUSD(v float64) BudgetValue {
	return BudgetValue{Kind: BudgetValueUSD, USDValue: v}
}

// MarshalJSON serialises as a flat tagged object matching the Rust shape:
// {"kind": "...", "value": ...}.
func (b BudgetValue) MarshalJSON() ([]byte, error) {
	switch b.Kind {
	case BudgetValueTurns:
		return json.Marshal(struct {
			Kind  BudgetValueKind `json:"kind"`
			Value uint32          `json:"value"`
		}{b.Kind, b.TurnsValue})
	case BudgetValueTokens:
		return json.Marshal(struct {
			Kind  BudgetValueKind `json:"kind"`
			Value uint64          `json:"value"`
		}{b.Kind, b.TokensValue})
	case BudgetValueDuration:
		return json.Marshal(struct {
			Kind  BudgetValueKind `json:"kind"`
			Value uint64          `json:"value"`
		}{b.Kind, b.DurationSecs})
	case BudgetValueUSD:
		return json.Marshal(struct {
			Kind  BudgetValueKind `json:"kind"`
			Value float64         `json:"value"`
		}{b.Kind, b.USDValue})
	default:
		return nil, fmt.Errorf("BudgetValue: unknown kind %q", b.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (b *BudgetValue) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind  BudgetValueKind `json:"kind"`
		Value json.RawMessage `json:"value"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	b.Kind = probe.Kind
	switch probe.Kind {
	case BudgetValueTurns:
		return json.Unmarshal(probe.Value, &b.TurnsValue)
	case BudgetValueTokens:
		return json.Unmarshal(probe.Value, &b.TokensValue)
	case BudgetValueDuration:
		return json.Unmarshal(probe.Value, &b.DurationSecs)
	case BudgetValueUSD:
		return json.Unmarshal(probe.Value, &b.USDValue)
	default:
		return fmt.Errorf("BudgetValue: unknown kind %q", probe.Kind)
	}
}

// ============================================================================
// SessionStateSnapshot
// ============================================================================

// SessionStateSnapshot is a read-only snapshot of session state handed to
// CompletionCheck.Check. The session/task ids let domain-specific checks key
// into per-session scratchpads.
//
// WorkspaceRoot is populated by the harness from
// SandboxProvider.WorkspaceRoot() so checks like FeatureListCheck can
// resolve workspace-relative paths without being given a sandbox handle.
type SessionStateSnapshot struct {
	SessionID     sporecore.SessionID    `json:"session_id"`
	TaskID        sporecore.TaskID       `json:"task_id"`
	State         sporecore.SessionState `json:"state"`
	WorkspaceRoot string                 `json:"workspace_root"`
}

// NewSessionStateSnapshot constructs a SessionStateSnapshot with an empty
// workspace root. Use NewSessionStateSnapshotWithRoot when the workspace
// root matters.
func NewSessionStateSnapshot(sid sporecore.SessionID, tid sporecore.TaskID, state sporecore.SessionState) SessionStateSnapshot {
	return SessionStateSnapshot{SessionID: sid, TaskID: tid, State: state}
}

// NewSessionStateSnapshotWithRoot constructs a SessionStateSnapshot with
// the supplied workspace root.
func NewSessionStateSnapshotWithRoot(sid sporecore.SessionID, tid sporecore.TaskID, state sporecore.SessionState, workspaceRoot string) SessionStateSnapshot {
	return SessionStateSnapshot{SessionID: sid, TaskID: tid, State: state, WorkspaceRoot: workspaceRoot}
}

// ============================================================================
// TerminationFailureReason (tagged union)
// ============================================================================

// TerminationFailureReasonKind discriminates TerminationFailureReason variants.
type TerminationFailureReasonKind string

const (
	// ReasonCompletionCheckFailed — completion check returned a hard failure.
	ReasonCompletionCheckFailed TerminationFailureReasonKind = "completion_check_failed"
	// ReasonMaxRetriesExhausted — repeated tool errors beyond retry budget.
	ReasonMaxRetriesExhausted TerminationFailureReasonKind = "max_retries_exhausted"
	// ReasonUnrecoverableSensorHalt — a sensor returned Halt.
	ReasonUnrecoverableSensorHalt TerminationFailureReasonKind = "unrecoverable_sensor_halt"
	// ReasonMiddlewareHalt — middleware returned Halt.
	ReasonMiddlewareHalt TerminationFailureReasonKind = "middleware_halt"
	// ReasonAgentError — agent loop produced an AgentError.
	ReasonAgentError TerminationFailureReasonKind = "agent_error"
	// ReasonPolicyViolation — a governance policy was violated.
	ReasonPolicyViolation TerminationFailureReasonKind = "policy_violation"
	// ReasonHumanHalted — set by the harness when HumanResponse::Halt arrives.
	// The TerminationPolicy never produces this variant.
	ReasonHumanHalted TerminationFailureReasonKind = "human_halted"
)

// TerminationFailureReason is the typed failure reason carried on HaltFailure.
//
// Exactly one variant's fields are populated, selected by Kind. Field naming
// mirrors the Rust serde tags so the JSON wire format is byte-equivalent.
type TerminationFailureReason struct {
	Kind TerminationFailureReasonKind

	// completion_check_failed / middleware_halt / unrecoverable_sensor_halt /
	// policy_violation
	Detail string
	// max_retries_exhausted
	Tool     string
	Attempts uint32
	// unrecoverable_sensor_halt
	SensorID sensor.SensorID
	// middleware_halt
	Hook   middleware.HookPoint
	Reason string
	// agent_error
	AgentError *sporecore.AgentError
}

// NewReasonCompletionCheckFailed builds a CompletionCheckFailed reason.
func NewReasonCompletionCheckFailed(detail string) TerminationFailureReason {
	return TerminationFailureReason{Kind: ReasonCompletionCheckFailed, Detail: detail}
}

// NewReasonMaxRetriesExhausted builds a MaxRetriesExhausted reason.
func NewReasonMaxRetriesExhausted(tool string, attempts uint32) TerminationFailureReason {
	return TerminationFailureReason{Kind: ReasonMaxRetriesExhausted, Tool: tool, Attempts: attempts}
}

// NewReasonUnrecoverableSensorHalt builds an UnrecoverableSensorHalt reason.
func NewReasonUnrecoverableSensorHalt(id sensor.SensorID, detail string) TerminationFailureReason {
	return TerminationFailureReason{Kind: ReasonUnrecoverableSensorHalt, SensorID: id, Detail: detail}
}

// NewReasonMiddlewareHalt builds a MiddlewareHalt reason.
func NewReasonMiddlewareHalt(hook middleware.HookPoint, reason string) TerminationFailureReason {
	return TerminationFailureReason{Kind: ReasonMiddlewareHalt, Hook: hook, Reason: reason}
}

// NewReasonAgentError builds an AgentError reason.
func NewReasonAgentError(err *sporecore.AgentError) TerminationFailureReason {
	return TerminationFailureReason{Kind: ReasonAgentError, AgentError: err}
}

// NewReasonPolicyViolation builds a PolicyViolation reason.
func NewReasonPolicyViolation(detail string) TerminationFailureReason {
	return TerminationFailureReason{Kind: ReasonPolicyViolation, Detail: detail}
}

// NewReasonHumanHalted builds a HumanHalted reason.
func NewReasonHumanHalted() TerminationFailureReason {
	return TerminationFailureReason{Kind: ReasonHumanHalted}
}

// MarshalJSON serialises as flat tagged JSON matching the Rust shape.
func (r TerminationFailureReason) MarshalJSON() ([]byte, error) {
	switch r.Kind {
	case ReasonCompletionCheckFailed, ReasonPolicyViolation:
		return json.Marshal(struct {
			Kind   TerminationFailureReasonKind `json:"kind"`
			Detail string                       `json:"detail"`
		}{r.Kind, r.Detail})
	case ReasonMaxRetriesExhausted:
		return json.Marshal(struct {
			Kind     TerminationFailureReasonKind `json:"kind"`
			Tool     string                       `json:"tool"`
			Attempts uint32                       `json:"attempts"`
		}{r.Kind, r.Tool, r.Attempts})
	case ReasonUnrecoverableSensorHalt:
		return json.Marshal(struct {
			Kind     TerminationFailureReasonKind `json:"kind"`
			SensorID sensor.SensorID              `json:"sensor_id"`
			Detail   string                       `json:"detail"`
		}{r.Kind, r.SensorID, r.Detail})
	case ReasonMiddlewareHalt:
		return json.Marshal(struct {
			Kind   TerminationFailureReasonKind `json:"kind"`
			Hook   middleware.HookPoint         `json:"hook"`
			Reason string                       `json:"reason"`
		}{r.Kind, r.Hook, r.Reason})
	case ReasonAgentError:
		return json.Marshal(struct {
			Kind  TerminationFailureReasonKind `json:"kind"`
			Error *sporecore.AgentError        `json:"error"`
		}{r.Kind, r.AgentError})
	case ReasonHumanHalted:
		return json.Marshal(struct {
			Kind TerminationFailureReasonKind `json:"kind"`
		}{r.Kind})
	default:
		return nil, fmt.Errorf("TerminationFailureReason: unknown kind %q", r.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (r *TerminationFailureReason) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind     TerminationFailureReasonKind `json:"kind"`
		Detail   string                       `json:"detail"`
		Tool     string                       `json:"tool"`
		Attempts uint32                       `json:"attempts"`
		SensorID sensor.SensorID              `json:"sensor_id"`
		Hook     middleware.HookPoint         `json:"hook"`
		Reason   string                       `json:"reason"`
		Error    *sporecore.AgentError        `json:"error"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	r.Kind = probe.Kind
	r.Detail = probe.Detail
	r.Tool = probe.Tool
	r.Attempts = probe.Attempts
	r.SensorID = probe.SensorID
	r.Hook = probe.Hook
	r.Reason = probe.Reason
	r.AgentError = probe.Error
	return nil
}

// ============================================================================
// TerminationDecision (tagged union)
// ============================================================================

// TerminationDecisionKind discriminates TerminationDecision variants.
type TerminationDecisionKind string

const (
	// DecisionContinue — the loop should run another turn.
	DecisionContinue TerminationDecisionKind = "continue"
	// DecisionHaltSuccess — the loop satisfied the completion criteria.
	DecisionHaltSuccess TerminationDecisionKind = "halt_success"
	// DecisionHaltFailure — the loop halted due to a typed failure reason.
	DecisionHaltFailure TerminationDecisionKind = "halt_failure"
	// DecisionHaltBudgetExceeded — a budget limit was exceeded.
	DecisionHaltBudgetExceeded TerminationDecisionKind = "halt_budget_exceeded"
)

// TerminationDecision is the output of TerminationPolicy.Evaluate.
//
// Exactly one variant's payload is meaningful, selected by Kind.
type TerminationDecision struct {
	Kind TerminationDecisionKind

	// halt_success
	Summary string
	// halt_failure
	Reason TerminationFailureReason
	// halt_budget_exceeded
	LimitType sporecore.BudgetLimitType
	Used      BudgetValue
	Limit     BudgetValue
}

// NewDecisionContinue builds a Continue decision.
func NewDecisionContinue() TerminationDecision {
	return TerminationDecision{Kind: DecisionContinue}
}

// NewDecisionHaltSuccess builds a HaltSuccess decision.
func NewDecisionHaltSuccess(summary string) TerminationDecision {
	return TerminationDecision{Kind: DecisionHaltSuccess, Summary: summary}
}

// NewDecisionHaltFailure builds a HaltFailure decision.
func NewDecisionHaltFailure(reason TerminationFailureReason) TerminationDecision {
	return TerminationDecision{Kind: DecisionHaltFailure, Reason: reason}
}

// NewDecisionHaltBudgetExceeded builds a HaltBudgetExceeded decision.
func NewDecisionHaltBudgetExceeded(lt sporecore.BudgetLimitType, used, limit BudgetValue) TerminationDecision {
	return TerminationDecision{Kind: DecisionHaltBudgetExceeded, LimitType: lt, Used: used, Limit: limit}
}

// MarshalJSON serialises as flat tagged JSON matching the Rust shape.
func (d TerminationDecision) MarshalJSON() ([]byte, error) {
	switch d.Kind {
	case DecisionContinue:
		return json.Marshal(struct {
			Kind TerminationDecisionKind `json:"kind"`
		}{d.Kind})
	case DecisionHaltSuccess:
		return json.Marshal(struct {
			Kind    TerminationDecisionKind `json:"kind"`
			Summary string                  `json:"summary"`
		}{d.Kind, d.Summary})
	case DecisionHaltFailure:
		return json.Marshal(struct {
			Kind   TerminationDecisionKind  `json:"kind"`
			Reason TerminationFailureReason `json:"reason"`
		}{d.Kind, d.Reason})
	case DecisionHaltBudgetExceeded:
		return json.Marshal(struct {
			Kind      TerminationDecisionKind   `json:"kind"`
			LimitType sporecore.BudgetLimitType `json:"limit_type"`
			Used      BudgetValue               `json:"used"`
			Limit     BudgetValue               `json:"limit"`
		}{d.Kind, d.LimitType, d.Used, d.Limit})
	default:
		return nil, fmt.Errorf("TerminationDecision: unknown kind %q", d.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (d *TerminationDecision) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind      TerminationDecisionKind   `json:"kind"`
		Summary   string                    `json:"summary"`
		Reason    *TerminationFailureReason `json:"reason"`
		LimitType sporecore.BudgetLimitType `json:"limit_type"`
		Used      *BudgetValue              `json:"used"`
		Limit     *BudgetValue              `json:"limit"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	d.Kind = probe.Kind
	d.Summary = probe.Summary
	if probe.Reason != nil {
		d.Reason = *probe.Reason
	}
	d.LimitType = probe.LimitType
	if probe.Used != nil {
		d.Used = *probe.Used
	}
	if probe.Limit != nil {
		d.Limit = *probe.Limit
	}
	return nil
}

// ============================================================================
// TerminationInput
// ============================================================================

// TerminationInput is the snapshot evaluated by TerminationPolicy.Evaluate.
type TerminationInput struct {
	SessionID       sporecore.SessionID      `json:"session_id"`
	TaskID          sporecore.TaskID         `json:"task_id"`
	TurnNumber      uint32                   `json:"turn_number"`
	AgentClaimsDone bool                     `json:"agent_claims_done"`
	AgentResponse   *string                  `json:"agent_response,omitempty"`
	BudgetUsed      sporecore.BudgetSnapshot `json:"budget_used"`
	BudgetLimits    sporecore.BudgetLimits   `json:"budget_limits"`
	SensorResults   []sensor.SensorResult    `json:"sensor_results"`
	SessionState    SessionStateSnapshot     `json:"session_state"`
}

// ============================================================================
// CompletionCheck
// ============================================================================

// CompletionCheck is the pluggable domain-specific completion check.
//
// Check returns (reason, complete). When complete is true, the policy halts
// with success. When complete is false, the policy returns Continue and the
// harness injects reason into the next turn's context.
//
// Returning a non-nil error indicates the check itself failed (distinct from
// "task is incomplete"); the policy surfaces this to the caller without
// mapping it to a TerminationDecision.
type CompletionCheck interface {
	Check(ctx context.Context, state *SessionStateSnapshot) (reason string, complete bool, err error)
	Description() string
}

// NullCompletionCheck always reports complete. Useful when the policy should
// halt with success the moment the agent claims done.
type NullCompletionCheck struct{}

// Check always returns ("", true, nil).
func (NullCompletionCheck) Check(context.Context, *SessionStateSnapshot) (string, bool, error) {
	return "", true, nil
}

// Description returns the canonical label.
func (NullCompletionCheck) Description() string { return "null (always complete)" }

// FixedCompletionCheck returns a configured outcome. Intended for tests and
// fixtures.
type FixedCompletionCheck struct {
	// Complete drives the return value of Check: when true, Check reports
	// complete; when false, Check returns IncompleteReason.
	Complete         bool
	IncompleteReason string
	Label            string
}

// NewFixedComplete returns a FixedCompletionCheck that reports complete.
func NewFixedComplete() *FixedCompletionCheck {
	return &FixedCompletionCheck{Complete: true, Label: "fixed:complete"}
}

// NewFixedIncomplete returns a FixedCompletionCheck that reports incomplete
// with the given reason.
func NewFixedIncomplete(reason string) *FixedCompletionCheck {
	return &FixedCompletionCheck{Complete: false, IncompleteReason: reason, Label: "fixed:incomplete"}
}

// Check returns the configured outcome.
func (f *FixedCompletionCheck) Check(context.Context, *SessionStateSnapshot) (string, bool, error) {
	if f.Complete {
		return "", true, nil
	}
	return f.IncompleteReason, false, nil
}

// Description returns the configured label.
func (f *FixedCompletionCheck) Description() string { return f.Label }

// ============================================================================
// TerminationPolicy
// ============================================================================

// TerminationPolicy decides whether the loop continues after a turn.
type TerminationPolicy interface {
	// Evaluate runs the full policy. Returns the decision; a non-nil error
	// indicates the policy itself failed to evaluate (distinct from a Halt
	// decision).
	Evaluate(ctx context.Context, input *TerminationInput) (TerminationDecision, error)

	// CheckBudget is a cheap budget-only poll the harness may call every
	// turn before assembling the rest of TerminationInput. Returns
	// (decision, true) when a budget limit is breached; otherwise the
	// decision is zero-valued and the bool is false.
	CheckBudget(snapshot *sporecore.BudgetSnapshot, limits *sporecore.BudgetLimits) (TerminationDecision, bool)
}

// CheckBudgetDefault is the default budget check used by
// StandardTerminationPolicy and exposed for direct use by the harness loop.
// Returns (decision, true) when any limit is breached.
func CheckBudgetDefault(snapshot *sporecore.BudgetSnapshot, limits *sporecore.BudgetLimits) (TerminationDecision, bool) {
	if limits.MaxTurns != nil && snapshot.Turns >= *limits.MaxTurns {
		return NewDecisionHaltBudgetExceeded(
			sporecore.BudgetLimitTurns,
			NewBudgetValueTurns(snapshot.Turns),
			NewBudgetValueTurns(*limits.MaxTurns),
		), true
	}
	if limits.MaxInputTokens != nil && snapshot.InputTokens >= uint64(*limits.MaxInputTokens) {
		return NewDecisionHaltBudgetExceeded(
			sporecore.BudgetLimitInputTokens,
			NewBudgetValueTokens(snapshot.InputTokens),
			NewBudgetValueTokens(uint64(*limits.MaxInputTokens)),
		), true
	}
	if limits.MaxOutputTokens != nil && snapshot.OutputTokens >= uint64(*limits.MaxOutputTokens) {
		return NewDecisionHaltBudgetExceeded(
			sporecore.BudgetLimitOutputTokens,
			NewBudgetValueTokens(snapshot.OutputTokens),
			NewBudgetValueTokens(uint64(*limits.MaxOutputTokens)),
		), true
	}
	if limits.MaxWallTime != nil {
		var usedSecs uint64
		if snapshot.WallTime != nil {
			usedSecs = uint64(snapshot.WallTime.Seconds())
		}
		limitSecs := uint64(limits.MaxWallTime.Seconds())
		if usedSecs >= limitSecs {
			return NewDecisionHaltBudgetExceeded(
				sporecore.BudgetLimitWallTime,
				NewBudgetValueDuration(usedSecs),
				NewBudgetValueDuration(limitSecs),
			), true
		}
	}
	if limits.MaxCostUSD != nil && snapshot.CostUSD >= *limits.MaxCostUSD {
		return NewDecisionHaltBudgetExceeded(
			sporecore.BudgetLimitCostUSD,
			NewBudgetValueUSD(snapshot.CostUSD),
			NewBudgetValueUSD(*limits.MaxCostUSD),
		), true
	}
	return TerminationDecision{}, false
}

// ============================================================================
// StandardTerminationPolicy
// ============================================================================

// StandardTerminationPolicy is the reference TerminationPolicy implementing
// the spec algorithm:
//  1. Budget check (unconditional).
//  2. Continue if !AgentClaimsDone.
//  3. UnrecoverableSensorHalt if any sensor result has SensorOutcome Halt.
//  4. The injected CompletionCheck.
type StandardTerminationPolicy struct {
	check CompletionCheck
}

// NewStandardTerminationPolicy returns a StandardTerminationPolicy with the
// given CompletionCheck.
func NewStandardTerminationPolicy(check CompletionCheck) *StandardTerminationPolicy {
	return &StandardTerminationPolicy{check: check}
}

// NewStandardTerminationPolicyWithNullCheck returns a policy backed by
// NullCompletionCheck — i.e. it halts with success the moment the agent
// claims done (subject to budget and sensor rules).
func NewStandardTerminationPolicyWithNullCheck() *StandardTerminationPolicy {
	return NewStandardTerminationPolicy(NullCompletionCheck{})
}

// CompletionCheck returns the injected completion check.
func (p *StandardTerminationPolicy) CompletionCheck() CompletionCheck { return p.check }

// CheckBudget delegates to CheckBudgetDefault.
func (p *StandardTerminationPolicy) CheckBudget(snapshot *sporecore.BudgetSnapshot, limits *sporecore.BudgetLimits) (TerminationDecision, bool) {
	return CheckBudgetDefault(snapshot, limits)
}

// Evaluate runs the policy algorithm.
func (p *StandardTerminationPolicy) Evaluate(ctx context.Context, input *TerminationInput) (TerminationDecision, error) {
	if d, ok := p.CheckBudget(&input.BudgetUsed, &input.BudgetLimits); ok {
		return d, nil
	}
	if !input.AgentClaimsDone {
		return NewDecisionContinue(), nil
	}
	for i := range input.SensorResults {
		r := &input.SensorResults[i]
		if r.Outcome == sensor.OutcomeHalt {
			return NewDecisionHaltFailure(NewReasonUnrecoverableSensorHalt(r.SensorID, r.Detail)), nil
		}
	}
	_, complete, err := p.check.Check(ctx, &input.SessionState)
	if err != nil {
		return TerminationDecision{}, fmt.Errorf("completion check: %w", err)
	}
	if !complete {
		// CompletionCheck reported incomplete — harness re-injects the
		// reason into next turn's context; the policy returns Continue.
		return NewDecisionContinue(), nil
	}
	var summary string
	if input.AgentResponse != nil {
		summary = *input.AgentResponse
	}
	return NewDecisionHaltSuccess(summary), nil
}

// Compile-time interface check.
var _ TerminationPolicy = (*StandardTerminationPolicy)(nil)
