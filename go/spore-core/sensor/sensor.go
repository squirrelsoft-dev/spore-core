// Package sensor — issue #10 `SensorChain`: post-action feedback controls
// and output quality evaluation.
//
// Sensors observe the agent's actions (tool calls, tool results, agent
// responses) at defined trigger points and emit SensorResults. The chain is
// a registry plus a fan-out evaluator: it runs every sensor registered for
// a trigger and returns all results without short-circuiting. The harness
// decides routing (Warn -> inject observation; Halt -> stop).
//
// See `docs/harness-engineering-concepts.md` § "SensorChain" for the rules
// this package enforces.
//
// Rules enforced
//   - `Fire` runs every sensor whose Config.Triggers contains the trigger
//     and returns all results — the chain never short-circuits.
//   - Computational sensors run on every matching trigger.
//   - Inferential sensors are gated by RunEveryNTurns (modulo
//     SensorInput.TurnNumber) and RunOnPhases (if set, input Phase must
//     match).
//   - `Stats` aggregates fire history; fire_rate = total_fires /
//     sessions_observed clamped to [0.0, 1.0].
//   - `SignalQualityReport` flags:
//   - NeverFired — chain has been asked to fire ≥ min_sessions distinct
//     sessions yet this sensor never fired.
//   - AlwaysFiring — sensor's fire-rate > low_signal_threshold
//     always_fired_rate.
//   - Trigger matching for PostTool: a sensor configured with
//     PostTool{ToolName: ""} (empty) matches any tool. Non-empty ToolName
//     matches exact equality only.
package sensor

import (
	"context"
	"encoding/json"
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/memory"
)

// ============================================================================
// Identity & time
// ============================================================================

// SensorID is the stable identifier for a sensor.
type SensorID string

// Timestamp is re-exported from the memory package for wire compatibility.
type Timestamp = memory.Timestamp

// ============================================================================
// SensorTrigger (tagged via Kind)
// ============================================================================

// SensorTriggerKind discriminates SensorTrigger variants.
type SensorTriggerKind string

const (
	// TriggerKindPostTool — after a specific tool executes.
	TriggerKindPostTool SensorTriggerKind = "post_tool"
	// TriggerKindPostTurn — after every agent turn.
	TriggerKindPostTurn SensorTriggerKind = "post_turn"
	// TriggerKindPostSession — after the loop ends.
	TriggerKindPostSession SensorTriggerKind = "post_session"
	// TriggerKindContinuous — runs outside the change lifecycle.
	TriggerKindContinuous SensorTriggerKind = "continuous"
	// TriggerKindOnToolError — when a tool returns an error.
	TriggerKindOnToolError SensorTriggerKind = "on_tool_error"
	// TriggerKindOnCompaction — after context is compacted.
	TriggerKindOnCompaction SensorTriggerKind = "on_compaction"
)

// SensorTrigger is a tagged union. ToolName is only meaningful for
// TriggerKindPostTool; an empty ToolName matches any tool.
type SensorTrigger struct {
	Kind     SensorTriggerKind `json:"kind"`
	ToolName string            `json:"tool_name,omitempty"`
}

// NewTriggerPostTool returns a PostTool trigger. Empty toolName matches any tool.
func NewTriggerPostTool(toolName string) SensorTrigger {
	return SensorTrigger{Kind: TriggerKindPostTool, ToolName: toolName}
}

// NewTriggerPostTurn returns a PostTurn trigger.
func NewTriggerPostTurn() SensorTrigger { return SensorTrigger{Kind: TriggerKindPostTurn} }

// NewTriggerPostSession returns a PostSession trigger.
func NewTriggerPostSession() SensorTrigger { return SensorTrigger{Kind: TriggerKindPostSession} }

// NewTriggerContinuous returns a Continuous trigger.
func NewTriggerContinuous() SensorTrigger { return SensorTrigger{Kind: TriggerKindContinuous} }

// NewTriggerOnToolError returns an OnToolError trigger.
func NewTriggerOnToolError() SensorTrigger { return SensorTrigger{Kind: TriggerKindOnToolError} }

// NewTriggerOnCompaction returns an OnCompaction trigger.
func NewTriggerOnCompaction() SensorTrigger { return SensorTrigger{Kind: TriggerKindOnCompaction} }

// Matches reports whether this configured trigger matches a fired trigger.
// PostTool{ToolName: ""} is a wildcard matching any PostTool.
func (t SensorTrigger) Matches(fired SensorTrigger) bool {
	if t.Kind != fired.Kind {
		return false
	}
	if t.Kind == TriggerKindPostTool {
		return t.ToolName == "" || t.ToolName == fired.ToolName
	}
	return true
}

// ============================================================================
// SensorKind
// ============================================================================

// SensorKind discriminates the kind of check a sensor performs.
type SensorKind string

const (
	// SensorKindComputational — deterministic, fast (linter, type check, etc).
	SensorKindComputational SensorKind = "computational"
	// SensorKindInferential — probabilistic, slow (LLM-as-judge).
	SensorKindInferential SensorKind = "inferential"
)

// ============================================================================
// SensorOutcome
// ============================================================================

// SensorOutcome is the outcome of a single sensor evaluation.
type SensorOutcome string

const (
	// OutcomePass — no concern.
	OutcomePass SensorOutcome = "pass"
	// OutcomeWarn — continue but inject observation into next turn.
	OutcomeWarn SensorOutcome = "warn"
	// OutcomeHalt — stop execution and surface to human/middleware.
	OutcomeHalt SensorOutcome = "halt"
)

// ============================================================================
// Records
// ============================================================================

// SensorInput is the snapshot a sensor evaluates.
type SensorInput struct {
	SessionID     sporecore.SessionID    `json:"session_id"`
	TurnNumber    *uint32                `json:"turn_number,omitempty"`
	Phase         *sporecore.TaskPhase   `json:"phase,omitempty"`
	ToolCall      *sporecore.ToolCall    `json:"tool_call,omitempty"`
	ToolResult    *sporecore.ToolResult  `json:"tool_result,omitempty"`
	AgentResponse *string                `json:"agent_response,omitempty"`
	SessionState  sporecore.SessionState `json:"session_state"`
}

// NewSensorInput constructs a SensorInput with sensible defaults.
func NewSensorInput(sessionID sporecore.SessionID, state sporecore.SessionState) SensorInput {
	return SensorInput{SessionID: sessionID, SessionState: state}
}

// SensorResult is what a sensor returns.
type SensorResult struct {
	SensorID    SensorID      `json:"sensor_id"`
	Outcome     SensorOutcome `json:"outcome"`
	Observation *string       `json:"observation,omitempty"`
	Detail      string        `json:"detail"`
	FiredAt     Timestamp     `json:"fired_at"`
}

// SensorSignalThresholds tunes the signal-quality report.
type SensorSignalThresholds struct {
	NeverFiredAfterNSessions uint32  `json:"never_fired_after_n_sessions"`
	AlwaysFiredRate          float32 `json:"always_fired_rate"`
}

// DefaultSensorSignalThresholds returns the standard thresholds.
func DefaultSensorSignalThresholds() SensorSignalThresholds {
	return SensorSignalThresholds{
		NeverFiredAfterNSessions: 10,
		AlwaysFiredRate:          0.9,
	}
}

// SensorConfig is the registration shape for a sensor.
type SensorConfig struct {
	ID                 SensorID               `json:"id"`
	Name               string                 `json:"name"`
	Kind               SensorKind             `json:"kind"`
	Triggers           []SensorTrigger        `json:"triggers"`
	RunEveryNTurns     *uint32                `json:"run_every_n_turns,omitempty"`
	RunOnPhases        []sporecore.TaskPhase  `json:"run_on_phases,omitempty"`
	LowSignalThreshold SensorSignalThresholds `json:"low_signal_threshold"`
}

// SensorStats is per-sensor aggregated firing history.
type SensorStats struct {
	SensorID      SensorID   `json:"sensor_id"`
	TotalFires    uint32     `json:"total_fires"`
	WarnCount     uint32     `json:"warn_count"`
	HaltCount     uint32     `json:"halt_count"`
	PassCount     uint32     `json:"pass_count"`
	FireRate      float32    `json:"fire_rate"`
	LastFired     *Timestamp `json:"last_fired,omitempty"`
	LowSignalFlag bool       `json:"low_signal_flag"`
}

// ============================================================================
// SensorSignalFlag (tagged via Kind)
// ============================================================================

// SensorSignalFlagKind discriminates SensorSignalFlag variants.
type SensorSignalFlagKind string

const (
	// FlagKindNeverFired — sensor never fired across observed sessions.
	FlagKindNeverFired SensorSignalFlagKind = "never_fired"
	// FlagKindAlwaysFiring — sensor fires above threshold rate.
	FlagKindAlwaysFiring SensorSignalFlagKind = "always_firing"
)

// SensorSignalFlag is a tagged union emitted by SignalQualityReport.
type SensorSignalFlag struct {
	Kind             SensorSignalFlagKind `json:"kind"`
	SensorID         SensorID             `json:"sensor_id"`
	SessionsObserved uint32               `json:"sessions_observed,omitempty"`
	FireRate         float32              `json:"fire_rate,omitempty"`
}

// NewFlagNeverFired constructs a NeverFired flag.
func NewFlagNeverFired(id SensorID, sessionsObserved uint32) SensorSignalFlag {
	return SensorSignalFlag{Kind: FlagKindNeverFired, SensorID: id, SessionsObserved: sessionsObserved}
}

// NewFlagAlwaysFiring constructs an AlwaysFiring flag.
func NewFlagAlwaysFiring(id SensorID, fireRate float32) SensorSignalFlag {
	return SensorSignalFlag{Kind: FlagKindAlwaysFiring, SensorID: id, FireRate: fireRate}
}

// ============================================================================
// Errors
// ============================================================================

// SensorErrorKind discriminates SensorError variants.
type SensorErrorKind string

const (
	// ErrKindAlreadyRegistered — duplicate sensor id.
	ErrKindAlreadyRegistered SensorErrorKind = "already_registered"
	// ErrKindValidationFailed — invalid sensor config.
	ErrKindValidationFailed SensorErrorKind = "validation_failed"
)

// SensorError is the typed error returned by SensorChain methods.
type SensorError struct {
	Kind     SensorErrorKind `json:"kind"`
	SensorID SensorID        `json:"sensor_id,omitempty"`
	Reason   string          `json:"reason,omitempty"`
}

// Error implements error.
func (e *SensorError) Error() string {
	switch e.Kind {
	case ErrKindAlreadyRegistered:
		return fmt.Sprintf("sensor already registered: %q", string(e.SensorID))
	case ErrKindValidationFailed:
		return fmt.Sprintf("validation failed: %s", e.Reason)
	default:
		return fmt.Sprintf("sensor error: %s", e.Kind)
	}
}

// MarshalJSON emits the Rust-compatible shape: {kind, ...variant fields}.
func (e *SensorError) MarshalJSON() ([]byte, error) {
	switch e.Kind {
	case ErrKindAlreadyRegistered:
		// Rust variant is a tuple struct: AlreadyRegistered(SensorId). Serde
		// serializes this as {"kind":"already_registered","0":SensorId} by
		// default — but we expose it via a named field for Go ergonomics.
		return json.Marshal(struct {
			Kind     SensorErrorKind `json:"kind"`
			SensorID SensorID        `json:"sensor_id"`
		}{e.Kind, e.SensorID})
	case ErrKindValidationFailed:
		return json.Marshal(struct {
			Kind   SensorErrorKind `json:"kind"`
			Reason string          `json:"reason"`
		}{e.Kind, e.Reason})
	}
	return json.Marshal(struct {
		Kind SensorErrorKind `json:"kind"`
	}{e.Kind})
}

// ============================================================================
// Interfaces
// ============================================================================

// Sensor is a single observation unit registered with a SensorChain.
type Sensor interface {
	// Evaluate inspects input and returns a result. Must not mutate input.
	Evaluate(ctx context.Context, input *SensorInput) SensorResult
	// Config returns the sensor's registration config.
	Config() SensorConfig
}

// SensorChain is the registry + fan-out evaluator.
type SensorChain interface {
	// Register validates and inserts a sensor. Duplicate ids return
	// AlreadyRegistered; empty triggers return ValidationFailed.
	Register(ctx context.Context, s Sensor) error

	// Fire runs every sensor whose triggers match and returns every result.
	// Inferential sensors are gated by RunEveryNTurns / RunOnPhases.
	Fire(ctx context.Context, trigger SensorTrigger, input *SensorInput) []SensorResult

	// Stats returns aggregated firing stats. If since is non-nil, only
	// records at-or-after that timestamp are included.
	Stats(ctx context.Context, since *Timestamp) []SensorStats

	// SignalQualityReport flags low-signal sensors. Returns empty when the
	// chain has observed fewer than minSessions sessions.
	SignalQualityReport(ctx context.Context, minSessions uint32) []SensorSignalFlag
}
