// Package guideregistry — issue #9 `GuideRegistry`: manage the lifecycle of
// guides and skills (feedforward artifacts injected before the agent acts).
//
// Guides include system prompt fragments, skills loaded on demand,
// convention docs (AGENTS.md / CLAUDE.md), schema annotations, and safety
// rules. The registry is the single source of truth for what guides
// exist, what state each is in, and how each has performed across
// sessions.
//
// See `docs/harness-engineering-concepts.md` § "GuideRegistry" for the
// rules this package enforces.
//
// Rules enforced
//   - States: Active, PendingReview, Deprecated, Stale. Hard delete is
//     not permitted.
//   - `Register` validates content (non-empty) and runs `CheckConflicts`
//     against existing Active guides; conflicts surface as
//     ConflictDetected.
//   - `GuideSource` of Kind `meta_agent_proposed` forces
//     PendingReview{AutomatedProposal, Since=ProposedAt} regardless of
//     caller-supplied status.
//   - `Select` returns only Active guides, filtered by domain and
//     `GuideTypes`, ordered by Jaccard relevance to `TaskInstruction`.
//   - `RecordUsage` appends an immutable history record.
//   - `AnalyzePerformance` emits:
//   - GuideDeprecationRecommended for guides whose failure rate
//     exceeds the no-guide baseline by `MinFailureRateDelta`.
//   - SkillGenerationNeeded for any failure-reason pattern that
//     appears at least `MinPatternOccurrences` times across failures.
//   - ConflictResolutionNeeded for any currently-flagged
//     ConflictDetected pending-review state.
//   - `PromoteToActive` is the only path from PendingReview to Active.
//   - Conflict detection (standard impl): same domain + high Jaccard
//     overlap with an existing active guide but non-identical content.
package guideregistry

import (
	"context"
	"fmt"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/memory"
)

// ============================================================================
// Identity & time
// ============================================================================

// GuideID is the stable identifier for a guide.
type GuideID string

// Timestamp is re-exported from the memory package so the wire format
// matches and types remain consistent across components.
type Timestamp = memory.Timestamp

// ============================================================================
// GuideType
// ============================================================================

// GuideType discriminates the kind of feedforward artifact.
type GuideType string

const (
	// GuideTypeSystemPromptFragment is appended to the system prompt.
	GuideTypeSystemPromptFragment GuideType = "system_prompt_fragment"
	// GuideTypeSkill is loaded on demand via progressive disclosure.
	GuideTypeSkill GuideType = "skill"
	// GuideTypeConventionDoc is an AGENTS.md / CLAUDE.md equivalent.
	GuideTypeConventionDoc GuideType = "convention_doc"
	// GuideTypeSchemaAnnotation is domain-specific knowledge.
	GuideTypeSchemaAnnotation GuideType = "schema_annotation"
	// GuideTypeSafetyRule is always-injected regardless of task.
	GuideTypeSafetyRule GuideType = "safety_rule"
)

// ============================================================================
// GuideSource (tagged via Kind)
// ============================================================================

// GuideSourceKind discriminates GuideSource variants.
type GuideSourceKind string

const (
	// SourceKindManual — written by a human.
	SourceKindManual GuideSourceKind = "manual"
	// SourceKindSessionGenerated — produced during a session.
	SourceKindSessionGenerated GuideSourceKind = "session_generated"
	// SourceKindTraceDistilled — synthesized from multiple sessions.
	SourceKindTraceDistilled GuideSourceKind = "trace_distilled"
	// SourceKindMetaAgentProposed — proposed by the meta-agent, pending review.
	SourceKindMetaAgentProposed GuideSourceKind = "meta_agent_proposed"
)

// GuideSource is a tagged union. Only the field(s) matching Kind are set.
type GuideSource struct {
	Kind       GuideSourceKind       `json:"kind"`
	SessionID  *sporecore.SessionID  `json:"session_id,omitempty"`
	SessionIDs []sporecore.SessionID `json:"session_ids,omitempty"`
	ProposedAt Timestamp             `json:"proposed_at,omitempty"`
}

// NewSourceManual returns a Manual source.
func NewSourceManual() GuideSource {
	return GuideSource{Kind: SourceKindManual}
}

// NewSourceSessionGenerated returns a SessionGenerated source.
func NewSourceSessionGenerated(id sporecore.SessionID) GuideSource {
	return GuideSource{Kind: SourceKindSessionGenerated, SessionID: &id}
}

// NewSourceTraceDistilled returns a TraceDistilled source.
func NewSourceTraceDistilled(ids []sporecore.SessionID) GuideSource {
	return GuideSource{Kind: SourceKindTraceDistilled, SessionIDs: ids}
}

// NewSourceMetaAgentProposed returns a MetaAgentProposed source.
func NewSourceMetaAgentProposed(proposedAt Timestamp) GuideSource {
	return GuideSource{Kind: SourceKindMetaAgentProposed, ProposedAt: proposedAt}
}

// ============================================================================
// PendingReason (tagged via Kind)
// ============================================================================

// PendingReasonKind discriminates PendingReason variants.
type PendingReasonKind string

const (
	// PendingReasonAutomatedProposal — proposed by the meta-agent.
	PendingReasonAutomatedProposal PendingReasonKind = "automated_proposal"
	// PendingReasonPerformanceDegradation — flagged by analyze_performance.
	PendingReasonPerformanceDegradation PendingReasonKind = "performance_degradation"
	// PendingReasonConflictDetected — conflicts with another active guide.
	PendingReasonConflictDetected PendingReasonKind = "conflict_detected"
	// PendingReasonManualFlag — manually flagged for review.
	PendingReasonManualFlag PendingReasonKind = "manual_flag"
)

// PendingReason is a tagged union.
type PendingReason struct {
	Kind             PendingReasonKind `json:"kind"`
	FailureRateDelta float32           `json:"failure_rate_delta,omitempty"`
	ConflictsWith    []GuideID         `json:"conflicts_with,omitempty"`
	Note             string            `json:"note,omitempty"`
}

// NewPendingReasonAutomatedProposal returns an AutomatedProposal reason.
func NewPendingReasonAutomatedProposal() PendingReason {
	return PendingReason{Kind: PendingReasonAutomatedProposal}
}

// NewPendingReasonPerformanceDegradation returns a PerformanceDegradation reason.
func NewPendingReasonPerformanceDegradation(delta float32) PendingReason {
	return PendingReason{Kind: PendingReasonPerformanceDegradation, FailureRateDelta: delta}
}

// NewPendingReasonConflictDetected returns a ConflictDetected reason.
func NewPendingReasonConflictDetected(ids []GuideID) PendingReason {
	return PendingReason{Kind: PendingReasonConflictDetected, ConflictsWith: ids}
}

// NewPendingReasonManualFlag returns a ManualFlag reason.
func NewPendingReasonManualFlag(note string) PendingReason {
	return PendingReason{Kind: PendingReasonManualFlag, Note: note}
}

// ============================================================================
// GuideStatus (tagged via Kind)
// ============================================================================

// GuideStatusKind discriminates GuideStatus variants.
type GuideStatusKind string

const (
	// StatusKindActive — usable for selection.
	StatusKindActive GuideStatusKind = "active"
	// StatusKindPendingReview — awaiting promotion to Active.
	StatusKindPendingReview GuideStatusKind = "pending_review"
	// StatusKindDeprecated — no longer recommended.
	StatusKindDeprecated GuideStatusKind = "deprecated"
	// StatusKindStale — used but not recently.
	StatusKindStale GuideStatusKind = "stale"
)

// GuideStatus is a tagged union.
type GuideStatus struct {
	Kind     GuideStatusKind `json:"kind"`
	Reason   *PendingReason  `json:"reason,omitempty"`
	At       Timestamp       `json:"at,omitempty"`
	Since    Timestamp       `json:"since,omitempty"`
	LastUsed Timestamp       `json:"last_used,omitempty"`
	// DeprecatedReason holds the free-form Deprecated reason string. (Renamed
	// from the Rust `reason: String` to avoid colliding with the
	// PendingReview `Reason` pointer field above.)
	DeprecatedReason string `json:"deprecated_reason,omitempty"`
}

// NewStatusActive returns an Active status.
func NewStatusActive() GuideStatus {
	return GuideStatus{Kind: StatusKindActive}
}

// NewStatusPendingReview returns a PendingReview status.
func NewStatusPendingReview(reason PendingReason, since Timestamp) GuideStatus {
	return GuideStatus{Kind: StatusKindPendingReview, Reason: &reason, Since: since}
}

// NewStatusDeprecated returns a Deprecated status.
func NewStatusDeprecated(reason string, at Timestamp) GuideStatus {
	return GuideStatus{Kind: StatusKindDeprecated, DeprecatedReason: reason, At: at}
}

// NewStatusStale returns a Stale status.
func NewStatusStale(lastUsed Timestamp) GuideStatus {
	return GuideStatus{Kind: StatusKindStale, LastUsed: lastUsed}
}

// ============================================================================
// SessionOutcome (tagged via Kind)
// ============================================================================

// SessionOutcomeKind discriminates SessionOutcome variants.
type SessionOutcomeKind string

const (
	// OutcomeKindSuccess — session succeeded.
	OutcomeKindSuccess SessionOutcomeKind = "success"
	// OutcomeKindFailure — session failed.
	OutcomeKindFailure SessionOutcomeKind = "failure"
	// OutcomeKindPartial — session partially succeeded.
	OutcomeKindPartial SessionOutcomeKind = "partial"
	// OutcomeKindEscalated — session terminated cleanly via a tool escalation
	// signal (issue #80). Distinct from Partial: an escalation is an
	// intentional, clean termination handing a structured signal to the caller,
	// not a partial success.
	OutcomeKindEscalated SessionOutcomeKind = "escalated"
)

// SessionOutcome is a tagged union.
type SessionOutcome struct {
	Kind   SessionOutcomeKind `json:"kind"`
	Reason string             `json:"reason,omitempty"`
}

// NewOutcomeSuccess returns a Success outcome.
func NewOutcomeSuccess() SessionOutcome {
	return SessionOutcome{Kind: OutcomeKindSuccess}
}

// NewOutcomeFailure returns a Failure outcome.
func NewOutcomeFailure(reason string) SessionOutcome {
	return SessionOutcome{Kind: OutcomeKindFailure, Reason: reason}
}

// NewOutcomePartial returns a Partial outcome.
func NewOutcomePartial() SessionOutcome {
	return SessionOutcome{Kind: OutcomeKindPartial}
}

// NewOutcomeEscalated returns an Escalated outcome (issue #80).
func NewOutcomeEscalated() SessionOutcome {
	return SessionOutcome{Kind: OutcomeKindEscalated}
}

// ============================================================================
// Records
// ============================================================================

// Guide is a feedforward artifact.
type Guide struct {
	ID        GuideID     `json:"id"`
	Name      string      `json:"name"`
	Content   string      `json:"content"`
	GuideType GuideType   `json:"guide_type"`
	Domain    *string     `json:"domain,omitempty"`
	Source    GuideSource `json:"source"`
	Status    GuideStatus `json:"status"`
	CreatedAt Timestamp   `json:"created_at"`
	LastUsed  *Timestamp  `json:"last_used,omitempty"`
	Version   uint32      `json:"version"`
}

// GuideUsageRecord is the outcome of a session that used a guide.
type GuideUsageRecord struct {
	GuideID    GuideID             `json:"guide_id"`
	SessionID  sporecore.SessionID `json:"session_id"`
	TaskDomain *string             `json:"task_domain,omitempty"`
	Outcome    SessionOutcome      `json:"outcome"`
	RecordedAt Timestamp           `json:"recorded_at"`
}

// GuideQuery selects relevant guides.
type GuideQuery struct {
	TaskInstruction string               `json:"task_instruction"`
	Domain          *string              `json:"domain,omitempty"`
	Phase           *sporecore.TaskPhase `json:"phase,omitempty"`
	GuideTypes      []GuideType          `json:"guide_types,omitempty"`
}

// NewGuideQuery constructs a GuideQuery with sensible defaults.
func NewGuideQuery(taskInstruction string) GuideQuery {
	return GuideQuery{TaskInstruction: taskInstruction}
}

// GuideConflict reports a detected conflict between two guides.
type GuideConflict struct {
	GuideA GuideID `json:"guide_a"`
	GuideB GuideID `json:"guide_b"`
	Reason string  `json:"reason"`
}

// ============================================================================
// Errors
// ============================================================================

// GuideRegistryErrorKind discriminates GuideRegistryError variants.
type GuideRegistryErrorKind string

const (
	// ErrKindNotFound — guide id not present.
	ErrKindNotFound GuideRegistryErrorKind = "not_found"
	// ErrKindConflictDetected — registration conflicted with an active guide.
	ErrKindConflictDetected GuideRegistryErrorKind = "conflict_detected"
	// ErrKindValidationFailed — input failed validation.
	ErrKindValidationFailed GuideRegistryErrorKind = "validation_failed"
	// ErrKindStorageError — backing store failure.
	ErrKindStorageError GuideRegistryErrorKind = "storage_error"
)

// GuideRegistryError is the typed error returned by GuideRegistry methods.
type GuideRegistryError struct {
	Kind     GuideRegistryErrorKind `json:"kind"`
	ID       GuideID                `json:"id,omitempty"`
	Conflict *GuideConflict         `json:"conflict,omitempty"`
	Reason   string                 `json:"reason,omitempty"`
}

// Error implements error.
func (e *GuideRegistryError) Error() string {
	switch e.Kind {
	case ErrKindNotFound:
		return fmt.Sprintf("guide not found: %q", string(e.ID))
	case ErrKindConflictDetected:
		if e.Conflict != nil {
			return fmt.Sprintf("conflict detected: %q vs %q: %s",
				string(e.Conflict.GuideA), string(e.Conflict.GuideB), e.Conflict.Reason)
		}
		return "conflict detected"
	case ErrKindValidationFailed:
		return fmt.Sprintf("validation failed: %s", e.Reason)
	case ErrKindStorageError:
		return fmt.Sprintf("storage error: %s", e.Reason)
	default:
		return fmt.Sprintf("guide registry error: %s", e.Kind)
	}
}

// ============================================================================
// ImprovementSignal (tagged via Kind)
// ============================================================================

// ImprovementSignalKind discriminates ImprovementSignal variants.
type ImprovementSignalKind string

const (
	// SignalKindSkillGenerationNeeded — pattern observed N+ times.
	SignalKindSkillGenerationNeeded ImprovementSignalKind = "skill_generation_needed"
	// SignalKindGuideDeprecationRecommended — failure-rate delta exceeded.
	SignalKindGuideDeprecationRecommended ImprovementSignalKind = "guide_deprecation_recommended"
	// SignalKindConflictResolutionNeeded — pending-review conflict open.
	SignalKindConflictResolutionNeeded ImprovementSignalKind = "conflict_resolution_needed"
)

// ImprovementSignal is a tagged union emitted by AnalyzePerformance.
type ImprovementSignal struct {
	Kind       ImprovementSignalKind `json:"kind"`
	Pattern    string                `json:"pattern,omitempty"`
	SessionIDs []sporecore.SessionID `json:"session_ids,omitempty"`
	GuideID    GuideID               `json:"guide_id,omitempty"`
	Reason     string                `json:"reason,omitempty"`
	Conflict   *GuideConflict        `json:"conflict,omitempty"`
}

// ============================================================================
// Interface
// ============================================================================

// GuideRegistry manages the lifecycle of guides.
type GuideRegistry interface {
	// Register a new guide — validates content and checks for conflicts.
	Register(ctx context.Context, guide Guide) (GuideID, error)

	// Select returns Active guides relevant to the query, ordered by
	// relevance descending.
	Select(ctx context.Context, query GuideQuery) ([]Guide, error)

	// RecordUsage appends a usage record. Errors with NotFound if the
	// guide is unknown.
	RecordUsage(ctx context.Context, record GuideUsageRecord) error

	// UsageHistory returns the full usage history for a guide.
	UsageHistory(ctx context.Context, id GuideID) ([]GuideUsageRecord, error)

	// Deprecate sets the status to Deprecated.
	Deprecate(ctx context.Context, id GuideID, reason string) error

	// MarkPendingReview transitions the guide to PendingReview.
	MarkPendingReview(ctx context.Context, id GuideID, reason PendingReason) error

	// PromoteToActive transitions a guide from PendingReview to Active.
	PromoteToActive(ctx context.Context, id GuideID) error

	// AnalyzePerformance emits ImprovementSignals.
	AnalyzePerformance(
		ctx context.Context,
		window time.Duration,
		minFailureRateDelta float32,
		minPatternOccurrences uint32,
	) []ImprovementSignal

	// CheckConflicts returns conflicts between proposed content and
	// existing Active guides in the same domain.
	CheckConflicts(ctx context.Context, content string, domain *string) []GuideConflict
}
