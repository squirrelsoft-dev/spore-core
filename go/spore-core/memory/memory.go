// Package memory — issue #8 `MemoryProvider`: persist and retrieve
// knowledge across turns and sessions.
//
// Stores two distinct kinds of memory:
//   - Episodic: what happened during a specific session (created by the
//     harness from session observations).
//   - Semantic: generalized knowledge — skills, rules, patterns, domain
//     facts — distilled from one or more episodic traces.
//
// See `docs/harness-engineering-concepts.md` § "MemoryProvider" for the
// rules this package enforces. The reference implementation
// `StandardMemoryProvider` is in-memory; production deployments swap in a
// durable equivalent without changing this interface.
//
// Rules enforced
//   - Episodic and semantic memory live in separate stores.
//   - `StoreSemantic(_, MergeStrategyReplace)` archives the previous record
//     under a fresh archival id, links it into `PreviousVersions`, and bumps
//     `Version`. Hard deletes are not permitted.
//   - `MergeStrategyAppend` concatenates content into the existing record in
//     place — same id, no new version.
//   - `MergeStrategyReject` returns `MergeConflict` on collision.
//   - `MetaAgentProposed` memories are forced into `PendingReview` regardless
//     of caller-supplied status.
//   - Empty content fails with `ValidationFailed`.
//   - `Query` returns items with relevance >= `MinRelevance`, capped at
//     `MaxItems`, sorted desc. Only `Active` semantic memories are returned.
//   - Versions are retained — `GetVersionHistory` walks `PreviousVersions`.
package memory

import (
	"context"
	"encoding/json"
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// Identity & time
// ============================================================================

// MemoryID is the stable identifier for a memory record.
type MemoryID string

// Timestamp is an RFC 3339 / ISO 8601 timestamp stored as a string for
// cross-language fixture portability.
type Timestamp string

// ============================================================================
// Episodic & semantic records
// ============================================================================

// EpisodicMemory is what happened during a specific session.
type EpisodicMemory struct {
	ID        MemoryID            `json:"id"`
	SessionID sporecore.SessionID `json:"session_id"`
	Content   string              `json:"content"`
	CreatedAt Timestamp           `json:"created_at"`
	Tags      []string            `json:"tags"`
}

// SemanticMemory is generalized knowledge distilled across episodes.
type SemanticMemory struct {
	ID               MemoryID     `json:"id"`
	Content          string       `json:"content"`
	Source           MemorySource `json:"source"`
	Domain           *string      `json:"domain,omitempty"`
	Version          uint32       `json:"version"`
	PreviousVersions []MemoryID   `json:"previous_versions"`
	CreatedAt        Timestamp    `json:"created_at"`
	UpdatedAt        Timestamp    `json:"updated_at"`
	Status           MemoryStatus `json:"status"`
}

// ============================================================================
// MemorySource (tagged via Kind field; matches Rust serde tag = "kind")
// ============================================================================

// MemorySourceKind discriminates MemorySource variants.
type MemorySourceKind string

const (
	// SourceKindManual — written by a human.
	SourceKindManual MemorySourceKind = "manual"
	// SourceKindSessionGenerated — produced during a session.
	SourceKindSessionGenerated MemorySourceKind = "session_generated"
	// SourceKindTraceDistilled — synthesized from multiple sessions.
	SourceKindTraceDistilled MemorySourceKind = "trace_distilled"
	// SourceKindMetaAgentProposed — proposed by the meta-agent, pending review.
	SourceKindMetaAgentProposed MemorySourceKind = "meta_agent_proposed"
)

// MemorySource is a tagged union. Only one of {SessionID, SessionIDs,
// ApprovedBy} is populated, determined by Kind.
type MemorySource struct {
	Kind       MemorySourceKind      `json:"kind"`
	SessionID  *sporecore.SessionID  `json:"session_id,omitempty"`
	SessionIDs []sporecore.SessionID `json:"session_ids,omitempty"`
	ApprovedBy *string               `json:"approved_by,omitempty"`
}

// NewSourceManual returns a Manual source.
func NewSourceManual() MemorySource {
	return MemorySource{Kind: SourceKindManual}
}

// NewSourceSessionGenerated returns a SessionGenerated source.
func NewSourceSessionGenerated(id sporecore.SessionID) MemorySource {
	return MemorySource{Kind: SourceKindSessionGenerated, SessionID: &id}
}

// NewSourceTraceDistilled returns a TraceDistilled source.
func NewSourceTraceDistilled(ids []sporecore.SessionID) MemorySource {
	return MemorySource{Kind: SourceKindTraceDistilled, SessionIDs: ids}
}

// NewSourceMetaAgentProposed returns a MetaAgentProposed source.
func NewSourceMetaAgentProposed(approvedBy *string) MemorySource {
	return MemorySource{Kind: SourceKindMetaAgentProposed, ApprovedBy: approvedBy}
}

// ============================================================================
// MemoryStatus (tagged via Kind)
// ============================================================================

// MemoryStatusKind discriminates MemoryStatus variants.
type MemoryStatusKind string

const (
	// StatusKindActive — usable for retrieval.
	StatusKindActive MemoryStatusKind = "active"
	// StatusKindDeprecated — no longer recommended.
	StatusKindDeprecated MemoryStatusKind = "deprecated"
	// StatusKindPendingReview — awaiting human review.
	StatusKindPendingReview MemoryStatusKind = "pending_review"
)

// MemoryStatus is a tagged union.
type MemoryStatus struct {
	Kind       MemoryStatusKind `json:"kind"`
	Reason     string           `json:"reason,omitempty"`
	At         Timestamp        `json:"at,omitempty"`
	ProposedAt Timestamp        `json:"proposed_at,omitempty"`
}

// NewStatusActive returns an Active status.
func NewStatusActive() MemoryStatus {
	return MemoryStatus{Kind: StatusKindActive}
}

// NewStatusDeprecated returns a Deprecated status.
func NewStatusDeprecated(reason string, at Timestamp) MemoryStatus {
	return MemoryStatus{Kind: StatusKindDeprecated, Reason: reason, At: at}
}

// NewStatusPendingReview returns a PendingReview status.
func NewStatusPendingReview(proposedAt Timestamp) MemoryStatus {
	return MemoryStatus{Kind: StatusKindPendingReview, ProposedAt: proposedAt}
}

// ============================================================================
// Query & retrieval
// ============================================================================

// MemoryItem is a scored query result.
type MemoryItem struct {
	Memory         SemanticMemory `json:"memory"`
	RelevanceScore float32        `json:"relevance_score"`
}

// MemoryQuery is the retrieval request.
type MemoryQuery struct {
	TaskInstruction string               `json:"task_instruction"`
	Domain          *string              `json:"domain,omitempty"`
	SessionID       *sporecore.SessionID `json:"session_id,omitempty"`
	MinRelevance    float32              `json:"min_relevance"`
	MaxItems        uint32               `json:"max_items"`
}

// NewMemoryQuery constructs a MemoryQuery with spec defaults
// (MinRelevance=0.5, MaxItems=10).
func NewMemoryQuery(taskInstruction string) MemoryQuery {
	return MemoryQuery{
		TaskInstruction: taskInstruction,
		MinRelevance:    0.5,
		MaxItems:        10,
	}
}

// ============================================================================
// MergeStrategy
// ============================================================================

// MergeStrategy controls write conflict behavior.
type MergeStrategy string

const (
	// MergeStrategyReplace overwrites the previous version, archiving it.
	MergeStrategyReplace MergeStrategy = "replace"
	// MergeStrategyAppend concatenates content in place; no new version.
	MergeStrategyAppend MergeStrategy = "append"
	// MergeStrategyReject errors on collision.
	MergeStrategyReject MergeStrategy = "reject"
)

// ============================================================================
// Errors
// ============================================================================

// MemoryErrorKind discriminates MemoryError variants.
type MemoryErrorKind string

const (
	// ErrKindNotFound — id not present in the store.
	ErrKindNotFound MemoryErrorKind = "not_found"
	// ErrKindMergeConflict — write conflicted with an existing record.
	ErrKindMergeConflict MemoryErrorKind = "merge_conflict"
	// ErrKindValidationFailed — input failed validation.
	ErrKindValidationFailed MemoryErrorKind = "validation_failed"
	// ErrKindStorageError — backing store failure.
	ErrKindStorageError MemoryErrorKind = "storage_error"
)

// MemoryError is the typed error returned by MemoryProvider methods.
type MemoryError struct {
	Kind     MemoryErrorKind `json:"kind"`
	ID       MemoryID        `json:"id,omitempty"`
	Existing MemoryID        `json:"existing,omitempty"`
	Reason   string          `json:"reason,omitempty"`
}

// Error implements error.
func (e *MemoryError) Error() string {
	switch e.Kind {
	case ErrKindNotFound:
		return fmt.Sprintf("memory not found: %q", string(e.ID))
	case ErrKindMergeConflict:
		return fmt.Sprintf("merge conflict on %q: %s", string(e.Existing), e.Reason)
	case ErrKindValidationFailed:
		return fmt.Sprintf("validation failed: %s", e.Reason)
	case ErrKindStorageError:
		return fmt.Sprintf("storage error: %s", e.Reason)
	default:
		return fmt.Sprintf("memory error: %s", e.Kind)
	}
}

// MarshalJSON ensures stable wire format alignment with the other languages.
func (e *MemoryError) MarshalJSON() ([]byte, error) {
	type alias MemoryError
	return json.Marshal((*alias)(e))
}

// ============================================================================
// Interface
// ============================================================================

// MemoryProvider persists and retrieves episodic and semantic memory.
type MemoryProvider interface {
	// ── Episodic ────────────────────────────────────────────────────────────
	StoreEpisodic(ctx context.Context, memory EpisodicMemory) (MemoryID, error)
	GetEpisodic(ctx context.Context, sessionID sporecore.SessionID) ([]EpisodicMemory, error)

	// ── Semantic ────────────────────────────────────────────────────────────
	StoreSemantic(ctx context.Context, memory SemanticMemory, onConflict MergeStrategy) (MemoryID, error)
	GetSemantic(ctx context.Context, id MemoryID) (SemanticMemory, error)

	// Primary retrieval path. Returns items with score >= MinRelevance,
	// capped at MaxItems, sorted by score descending. Only Active memories.
	Query(ctx context.Context, query MemoryQuery) ([]MemoryItem, error)

	// ── Lifecycle ───────────────────────────────────────────────────────────
	Deprecate(ctx context.Context, id MemoryID, reason string) error
	GetVersionHistory(ctx context.Context, id MemoryID) ([]SemanticMemory, error)
	MarkPendingReview(ctx context.Context, id MemoryID) error
}
