package guideregistry

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// conflictThreshold is the Jaccard cutoff for flagging two same-domain
// guides as conflicting.
const conflictThreshold float32 = 0.6

// StandardGuideRegistry is the reference in-memory GuideRegistry.
//
// Conflict heuristic: two guides conflict when they share a domain and
// have Jaccard token overlap >= conflictThreshold (0.6) but non-identical
// content. Production deployments may override via a separate
// implementation.
type StandardGuideRegistry struct {
	mu          sync.Mutex
	guides      map[GuideID]Guide
	usage       []GuideUsageRecord
	nowOverride *Timestamp
}

// NewStandardGuideRegistry constructs an empty provider.
func NewStandardGuideRegistry() *StandardGuideRegistry {
	return &StandardGuideRegistry{
		guides: make(map[GuideID]Guide),
	}
}

// SetNow pins the "now" timestamp for AnalyzePerformance and lifecycle
// transitions. Tests use this for deterministic results.
func (r *StandardGuideRegistry) SetNow(now Timestamp) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.nowOverride = &now
}

// ── Helpers ────────────────────────────────────────────────────────────────

func validateContent(s string) error {
	if strings.TrimSpace(s) == "" {
		return &GuideRegistryError{
			Kind:   ErrKindValidationFailed,
			Reason: "content must not be empty",
		}
	}
	return nil
}

func tokenize(s string) []string {
	lower := strings.ToLower(s)
	out := make([]string, 0)
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range lower {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return out
}

func jaccard(a, b string) float32 {
	ta := map[string]struct{}{}
	for _, t := range tokenize(a) {
		ta[t] = struct{}{}
	}
	tb := map[string]struct{}{}
	for _, t := range tokenize(b) {
		tb[t] = struct{}{}
	}
	if len(ta) == 0 && len(tb) == 0 {
		return 0
	}
	inter := 0
	for t := range ta {
		if _, ok := tb[t]; ok {
			inter++
		}
	}
	union := len(ta) + len(tb) - inter
	if union == 0 {
		return 0
	}
	return float32(inter) / float32(union)
}

// enforceMetaAgentPending forces MetaAgentProposed sources into
// PendingReview{AutomatedProposal, Since=ProposedAt}.
func enforceMetaAgentPending(g *Guide) {
	if g.Source.Kind == SourceKindMetaAgentProposed {
		g.Status = NewStatusPendingReview(
			NewPendingReasonAutomatedProposal(),
			g.Source.ProposedAt,
		)
	}
}

// nowTimestamp returns the override if set, else system clock as RFC3339.
func (r *StandardGuideRegistry) nowTimestamp() Timestamp {
	if r.nowOverride != nil {
		return *r.nowOverride
	}
	return Timestamp(time.Now().UTC().Format(time.RFC3339))
}

// rfc3339Subtract returns now - window as an RFC 3339 timestamp.
// If now does not parse, returns ok=false.
func rfc3339Subtract(now Timestamp, window time.Duration) (Timestamp, bool) {
	t, err := time.Parse(time.RFC3339, string(now))
	if err != nil {
		// Try a few tolerated variants by trimming fractional seconds.
		t, err = time.Parse("2006-01-02T15:04:05Z", string(now))
		if err != nil {
			return "", false
		}
	}
	cutoff := t.Add(-window)
	return Timestamp(cutoff.UTC().Format("2006-01-02T15:04:05Z")), true
}

// inWindow returns true if t >= cutoff using lexicographic compare on
// RFC 3339 strings. If !haveCutoff, every record is considered in-window.
func inWindow(t Timestamp, cutoff Timestamp, haveCutoff bool) bool {
	if !haveCutoff {
		return true
	}
	return string(t) >= string(cutoff)
}

// ── Register / Select ──────────────────────────────────────────────────────

// Register validates content, enforces meta-agent pending, checks for
// conflicts, and inserts the guide.
func (r *StandardGuideRegistry) Register(ctx context.Context, guide Guide) (GuideID, error) {
	if err := validateContent(guide.Content); err != nil {
		return "", err
	}
	enforceMetaAgentPending(&guide)

	conflicts := r.CheckConflicts(ctx, guide.Content, guide.Domain)
	if len(conflicts) > 0 {
		first := conflicts[0]
		return "", &GuideRegistryError{
			Kind: ErrKindConflictDetected,
			Conflict: &GuideConflict{
				GuideA: guide.ID,
				GuideB: first.GuideB,
				Reason: first.Reason,
			},
		}
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.guides[guide.ID] = guide
	return guide.ID, nil
}

// Select returns Active guides matching domain & types, ordered by
// Jaccard relevance descending.
func (r *StandardGuideRegistry) Select(_ context.Context, q GuideQuery) ([]Guide, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	type scored struct {
		score float32
		guide Guide
	}
	out := make([]scored, 0)

	typeFilter := func(t GuideType) bool {
		if len(q.GuideTypes) == 0 {
			return true
		}
		for _, want := range q.GuideTypes {
			if want == t {
				return true
			}
		}
		return false
	}

	for _, g := range r.guides {
		if g.Status.Kind != StatusKindActive {
			continue
		}
		// Domain filter: Some(want) requires Some(have)=want; None matches all.
		if q.Domain != nil {
			if g.Domain == nil || *g.Domain != *q.Domain {
				continue
			}
		}
		if !typeFilter(g.GuideType) {
			continue
		}
		s := jaccard(q.TaskInstruction, g.Content)
		out = append(out, scored{s, g})
	}
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].score > out[j].score
	})
	res := make([]Guide, len(out))
	for i, s := range out {
		res[i] = s.guide
	}
	return res, nil
}

// ── Usage ──────────────────────────────────────────────────────────────────

// RecordUsage appends a usage record and updates LastUsed on the guide.
func (r *StandardGuideRegistry) RecordUsage(_ context.Context, rec GuideUsageRecord) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	g, ok := r.guides[rec.GuideID]
	if !ok {
		return &GuideRegistryError{Kind: ErrKindNotFound, ID: rec.GuideID}
	}
	ts := rec.RecordedAt
	g.LastUsed = &ts
	r.guides[rec.GuideID] = g
	r.usage = append(r.usage, rec)
	return nil
}

// UsageHistory returns this guide's records.
func (r *StandardGuideRegistry) UsageHistory(_ context.Context, id GuideID) ([]GuideUsageRecord, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.guides[id]; !ok {
		return nil, &GuideRegistryError{Kind: ErrKindNotFound, ID: id}
	}
	out := make([]GuideUsageRecord, 0)
	for _, u := range r.usage {
		if u.GuideID == id {
			out = append(out, u)
		}
	}
	return out, nil
}

// ── Lifecycle ──────────────────────────────────────────────────────────────

// Deprecate transitions the guide to Deprecated.
func (r *StandardGuideRegistry) Deprecate(_ context.Context, id GuideID, reason string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	g, ok := r.guides[id]
	if !ok {
		return &GuideRegistryError{Kind: ErrKindNotFound, ID: id}
	}
	g.Status = NewStatusDeprecated(reason, r.nowTimestamp())
	r.guides[id] = g
	return nil
}

// MarkPendingReview transitions to PendingReview with the given reason.
func (r *StandardGuideRegistry) MarkPendingReview(_ context.Context, id GuideID, reason PendingReason) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	g, ok := r.guides[id]
	if !ok {
		return &GuideRegistryError{Kind: ErrKindNotFound, ID: id}
	}
	g.Status = NewStatusPendingReview(reason, r.nowTimestamp())
	r.guides[id] = g
	return nil
}

// PromoteToActive transitions PendingReview -> Active. Errors otherwise.
func (r *StandardGuideRegistry) PromoteToActive(_ context.Context, id GuideID) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	g, ok := r.guides[id]
	if !ok {
		return &GuideRegistryError{Kind: ErrKindNotFound, ID: id}
	}
	if g.Status.Kind != StatusKindPendingReview {
		return &GuideRegistryError{
			Kind:   ErrKindValidationFailed,
			Reason: "promote_to_active requires PendingReview status",
		}
	}
	g.Status = NewStatusActive()
	r.guides[id] = g
	return nil
}

// ── Analysis ───────────────────────────────────────────────────────────────

// AnalyzePerformance emits improvement signals across the registry.
func (r *StandardGuideRegistry) AnalyzePerformance(
	_ context.Context,
	window time.Duration,
	minFailureRateDelta float32,
	minPatternOccurrences uint32,
) []ImprovementSignal {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := r.nowTimestamp()
	cutoff, haveCutoff := rfc3339Subtract(now, window)

	inWin := make([]GuideUsageRecord, 0, len(r.usage))
	for _, u := range r.usage {
		if inWindow(u.RecordedAt, cutoff, haveCutoff) {
			inWin = append(inWin, u)
		}
	}

	signals := make([]ImprovementSignal, 0)

	totalFailures := 0
	for _, u := range inWin {
		if u.Outcome.Kind == OutcomeKindFailure {
			totalFailures++
		}
	}
	totalRecords := len(inWin)

	// Per-guide signals: deterministic iteration by sorted id.
	ids := make([]GuideID, 0, len(r.guides))
	for id := range r.guides {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return string(ids[i]) < string(ids[j]) })

	for _, gid := range ids {
		g := r.guides[gid]

		// Conflict-derived signal.
		if g.Status.Kind == StatusKindPendingReview &&
			g.Status.Reason != nil &&
			g.Status.Reason.Kind == PendingReasonConflictDetected &&
			len(g.Status.Reason.ConflictsWith) > 0 {
			other := g.Status.Reason.ConflictsWith[0]
			signals = append(signals, ImprovementSignal{
				Kind: SignalKindConflictResolutionNeeded,
				Conflict: &GuideConflict{
					GuideA: gid,
					GuideB: other,
					Reason: "pending-review conflict",
				},
			})
		}

		// Performance-delta signal.
		var withCount, withFail int
		for _, u := range inWin {
			if u.GuideID == gid {
				withCount++
				if u.Outcome.Kind == OutcomeKindFailure {
					withFail++
				}
			}
		}
		if withCount == 0 {
			continue
		}
		withRate := float32(withFail) / float32(withCount)
		withoutCount := totalRecords - withCount
		var baseline float32
		if withoutCount > 0 {
			baseline = float32(totalFailures-withFail) / float32(withoutCount)
		}
		if withRate-baseline >= minFailureRateDelta {
			signals = append(signals, ImprovementSignal{
				Kind:    SignalKindGuideDeprecationRecommended,
				GuideID: gid,
				Reason: fmt.Sprintf(
					"failure-rate delta %.2f (with=%.2f vs baseline=%.2f)",
					withRate-baseline, withRate, baseline,
				),
			})
		}
	}

	// Pattern detection: failure reasons appearing >= N times.
	patternSessions := make(map[string][]sporecore.SessionID)
	for _, u := range inWin {
		if u.Outcome.Kind == OutcomeKindFailure {
			patternSessions[u.Outcome.Reason] = append(patternSessions[u.Outcome.Reason], u.SessionID)
		}
	}
	// Deterministic order: sort pattern keys.
	keys := make([]string, 0, len(patternSessions))
	for k := range patternSessions {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, p := range keys {
		sessions := patternSessions[p]
		if uint32(len(sessions)) >= minPatternOccurrences {
			signals = append(signals, ImprovementSignal{
				Kind:       SignalKindSkillGenerationNeeded,
				Pattern:    p,
				SessionIDs: append([]sporecore.SessionID(nil), sessions...),
			})
		}
	}

	return signals
}

// ── CheckConflicts ─────────────────────────────────────────────────────────

// CheckConflicts returns same-domain Active guides whose content has
// Jaccard overlap >= conflictThreshold with the proposed content (and is
// not identical).
func (r *StandardGuideRegistry) CheckConflicts(_ context.Context, content string, domain *string) []GuideConflict {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]GuideConflict, 0)
	// Deterministic order by id.
	ids := make([]GuideID, 0, len(r.guides))
	for id := range r.guides {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return string(ids[i]) < string(ids[j]) })

	for _, gid := range ids {
		g := r.guides[gid]
		if g.Status.Kind != StatusKindActive {
			continue
		}
		// Same domain only: nil vs Some(d) is not a conflict.
		if !domainEqual(g.Domain, domain) {
			continue
		}
		if g.Content == content {
			continue
		}
		score := jaccard(g.Content, content)
		if score >= conflictThreshold {
			out = append(out, GuideConflict{
				GuideA: GuideID("<new>"),
				GuideB: g.ID,
				Reason: fmt.Sprintf("Jaccard overlap %.2f >= %.2f", score, conflictThreshold),
			})
		}
	}
	return out
}

func domainEqual(a, b *string) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return *a == *b
}

// Compile-time interface check.
var _ GuideRegistry = (*StandardGuideRegistry)(nil)
