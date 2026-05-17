package guideregistry

import (
	"context"
	"errors"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func ts(s string) Timestamp { return Timestamp(s) }

func strPtr(s string) *string { return &s }

func makeGuide(id, content string) Guide {
	return Guide{
		ID:        GuideID(id),
		Name:      id,
		Content:   content,
		GuideType: GuideTypeSkill,
		Domain:    nil,
		Source:    NewSourceManual(),
		Status:    NewStatusActive(),
		CreatedAt: ts("2026-05-16T00:00:00Z"),
		Version:   1,
	}
}

func usage(gid, sid string, outcome SessionOutcome) GuideUsageRecord {
	return GuideUsageRecord{
		GuideID:    GuideID(gid),
		SessionID:  sporecore.SessionID(sid),
		Outcome:    outcome,
		RecordedAt: ts("2026-05-16T00:00:00Z"),
	}
}

// 1) register validates content
func TestRegisterEmptyContentFails(t *testing.T) {
	r := NewStandardGuideRegistry()
	g := makeGuide("g1", "   ")
	g.Content = "   "
	_, err := r.Register(context.Background(), g)
	var e *GuideRegistryError
	if !errors.As(err, &e) || e.Kind != ErrKindValidationFailed {
		t.Fatalf("expected ValidationFailed, got %v", err)
	}
}

// 2) MetaAgentProposed forces PendingReview
func TestMetaAgentSourceForcesPendingReview(t *testing.T) {
	r := NewStandardGuideRegistry()
	g := makeGuide("g1", "proposed")
	g.Source = NewSourceMetaAgentProposed(ts("2026-05-16T01:00:00Z"))
	g.Status = NewStatusActive() // caller lies; provider corrects
	if _, err := r.Register(context.Background(), g); err != nil {
		t.Fatalf("register: %v", err)
	}
	sel, err := r.Select(context.Background(), NewGuideQuery("anything"))
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	for _, sg := range sel {
		if sg.ID == "g1" {
			t.Fatalf("g1 should not appear in Active select results")
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	got := r.guides[GuideID("g1")]
	if got.Status.Kind != StatusKindPendingReview {
		t.Fatalf("expected PendingReview, got %s", got.Status.Kind)
	}
	if got.Status.Reason == nil || got.Status.Reason.Kind != PendingReasonAutomatedProposal {
		t.Fatalf("expected AutomatedProposal reason, got %+v", got.Status.Reason)
	}
}

// 3) select filters by status, domain, and type
func TestSelectFiltersByStatusDomainAndType(t *testing.T) {
	r := NewStandardGuideRegistry()
	a := makeGuide("a", "rust async tokio runtime")
	a.Domain = strPtr("rust")
	b := makeGuide("b", "pytest fixtures python")
	b.Domain = strPtr("python")
	b.GuideType = GuideTypeConventionDoc
	c := makeGuide("c", "deprecated content")
	c.Status = NewStatusDeprecated("old", ts("2026-05-16T00:00:00Z"))

	if _, err := r.Register(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Register(context.Background(), b); err != nil {
		t.Fatal(err)
	}
	// Inject c bypassing register's conflict check.
	r.mu.Lock()
	r.guides[GuideID("c")] = c
	r.mu.Unlock()

	res, err := r.Select(context.Background(), GuideQuery{
		TaskInstruction: "rust tokio",
		Domain:          strPtr("rust"),
		GuideTypes:      []GuideType{GuideTypeSkill},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 {
		t.Fatalf("expected 1 result, got %d", len(res))
	}
	if res[0].ID != "a" {
		t.Fatalf("expected a, got %s", res[0].ID)
	}
}

// 4) select sorted by relevance
func TestSelectSortedByRelevance(t *testing.T) {
	r := NewStandardGuideRegistry()
	ctx := context.Background()
	if _, err := r.Register(ctx, makeGuide("a", "alpha beta gamma delta")); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Register(ctx, makeGuide("b", "zebra")); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Register(ctx, makeGuide("c", "alpha beta")); err != nil {
		t.Fatal(err)
	}
	res, err := r.Select(ctx, NewGuideQuery("alpha beta"))
	if err != nil {
		t.Fatal(err)
	}
	if res[0].ID != "c" {
		t.Fatalf("expected first=c, got %s", res[0].ID)
	}
	if res[len(res)-1].ID != "b" {
		t.Fatalf("expected last=b, got %s", res[len(res)-1].ID)
	}
}

// 5) conflict detection at registration
func TestRegisterDetectsConflictInSameDomain(t *testing.T) {
	r := NewStandardGuideRegistry()
	existing := makeGuide("a", "always run tests before commit")
	existing.Domain = strPtr("rust")
	if _, err := r.Register(context.Background(), existing); err != nil {
		t.Fatal(err)
	}
	conflicting := makeGuide("b", "always run tests before committing")
	conflicting.Domain = strPtr("rust")
	_, err := r.Register(context.Background(), conflicting)
	var e *GuideRegistryError
	if !errors.As(err, &e) || e.Kind != ErrKindConflictDetected {
		t.Fatalf("expected ConflictDetected, got %v", err)
	}
	if e.Conflict == nil || e.Conflict.GuideA != "b" || e.Conflict.GuideB != "a" {
		t.Fatalf("unexpected conflict: %+v", e.Conflict)
	}
}

// 6) no conflict across domains
func TestNoConflictAcrossDomains(t *testing.T) {
	r := NewStandardGuideRegistry()
	a := makeGuide("a", "always run tests before commit")
	a.Domain = strPtr("rust")
	if _, err := r.Register(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	b := makeGuide("b", "always run tests before commit")
	b.Domain = strPtr("python")
	if _, err := r.Register(context.Background(), b); err != nil {
		t.Fatalf("expected no conflict across domains, got %v", err)
	}
}

// 7) record_usage requires existing guide
func TestRecordUsageRequiresKnownGuide(t *testing.T) {
	r := NewStandardGuideRegistry()
	err := r.RecordUsage(context.Background(), usage("nope", "s1", NewOutcomeSuccess()))
	var e *GuideRegistryError
	if !errors.As(err, &e) || e.Kind != ErrKindNotFound {
		t.Fatalf("expected NotFound, got %v", err)
	}
}

// 8) record_usage updates last_used
func TestRecordUsageUpdatesLastUsed(t *testing.T) {
	r := NewStandardGuideRegistry()
	ctx := context.Background()
	if _, err := r.Register(ctx, makeGuide("a", "x")); err != nil {
		t.Fatal(err)
	}
	u := usage("a", "s1", NewOutcomeSuccess())
	u.RecordedAt = ts("2026-06-01T00:00:00Z")
	if err := r.RecordUsage(ctx, u); err != nil {
		t.Fatal(err)
	}
	hist, err := r.UsageHistory(ctx, GuideID("a"))
	if err != nil {
		t.Fatal(err)
	}
	if len(hist) != 1 {
		t.Fatalf("expected 1 record, got %d", len(hist))
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	g := r.guides[GuideID("a")]
	if g.LastUsed == nil || string(*g.LastUsed) != "2026-06-01T00:00:00Z" {
		t.Fatalf("unexpected last_used: %+v", g.LastUsed)
	}
}

// 9) deprecate sets status and 404s
func TestDeprecateSetsStatusAndNotFound(t *testing.T) {
	r := NewStandardGuideRegistry()
	ctx := context.Background()
	if _, err := r.Register(ctx, makeGuide("a", "x")); err != nil {
		t.Fatal(err)
	}
	if err := r.Deprecate(ctx, GuideID("a"), "obsolete"); err != nil {
		t.Fatal(err)
	}
	r.mu.Lock()
	g := r.guides[GuideID("a")]
	r.mu.Unlock()
	if g.Status.Kind != StatusKindDeprecated || g.Status.DeprecatedReason != "obsolete" {
		t.Fatalf("expected Deprecated/obsolete, got %+v", g.Status)
	}
	err := r.Deprecate(ctx, GuideID("nope"), "x")
	var e *GuideRegistryError
	if !errors.As(err, &e) || e.Kind != ErrKindNotFound {
		t.Fatalf("expected NotFound, got %v", err)
	}
}

// 10) promote_to_active only from PendingReview
func TestPromoteToActiveOnlyFromPendingReview(t *testing.T) {
	r := NewStandardGuideRegistry()
	ctx := context.Background()
	if _, err := r.Register(ctx, makeGuide("a", "x")); err != nil {
		t.Fatal(err)
	}
	err := r.PromoteToActive(ctx, GuideID("a"))
	var e *GuideRegistryError
	if !errors.As(err, &e) || e.Kind != ErrKindValidationFailed {
		t.Fatalf("expected ValidationFailed, got %v", err)
	}
	if err := r.MarkPendingReview(ctx, GuideID("a"), NewPendingReasonManualFlag("x")); err != nil {
		t.Fatal(err)
	}
	if err := r.PromoteToActive(ctx, GuideID("a")); err != nil {
		t.Fatal(err)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.guides[GuideID("a")].Status.Kind != StatusKindActive {
		t.Fatalf("expected Active after promote")
	}
}

// 11) analyze_performance flags high failure rate
func TestAnalyzePerformanceFlagsHighFailureRate(t *testing.T) {
	r := NewStandardGuideRegistry()
	ctx := context.Background()
	r.SetNow(ts("2026-05-16T01:00:00Z"))
	if _, err := r.Register(ctx, makeGuide("a", "x")); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Register(ctx, makeGuide("b", "y")); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if err := r.RecordUsage(ctx, usage("a", "s"+itoa(i), NewOutcomeFailure("boom"))); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 3; i++ {
		if err := r.RecordUsage(ctx, usage("b", "sb"+itoa(i), NewOutcomeSuccess())); err != nil {
			t.Fatal(err)
		}
	}
	signals := r.AnalyzePerformance(ctx, 24*time.Hour, 0.5, 100)
	found := false
	for _, s := range signals {
		if s.Kind == SignalKindGuideDeprecationRecommended && s.GuideID == "a" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected GuideDeprecationRecommended for a, got %+v", signals)
	}
}

// 12) analyze_performance emits SkillGenerationNeeded for repeated pattern
func TestAnalyzePerformanceEmitsSkillGenerationForRepeatedPattern(t *testing.T) {
	r := NewStandardGuideRegistry()
	ctx := context.Background()
	r.SetNow(ts("2026-05-16T01:00:00Z"))
	if _, err := r.Register(ctx, makeGuide("a", "x")); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		if err := r.RecordUsage(ctx, usage("a", "s"+itoa(i), NewOutcomeFailure("panic: index out of bounds"))); err != nil {
			t.Fatal(err)
		}
	}
	signals := r.AnalyzePerformance(ctx, 24*time.Hour, 999.0, 3)
	found := false
	for _, s := range signals {
		if s.Kind == SignalKindSkillGenerationNeeded &&
			s.Pattern == "panic: index out of bounds" &&
			len(s.SessionIDs) == 4 {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected SkillGenerationNeeded, got %+v", signals)
	}
}

// 13) analyze_performance honors window
func TestAnalyzePerformanceFiltersByWindow(t *testing.T) {
	r := NewStandardGuideRegistry()
	ctx := context.Background()
	r.SetNow(ts("2026-05-16T00:00:00Z"))
	if _, err := r.Register(ctx, makeGuide("a", "x")); err != nil {
		t.Fatal(err)
	}
	old := usage("a", "s0", NewOutcomeFailure("old-pattern"))
	old.RecordedAt = ts("2020-01-01T00:00:00Z")
	if err := r.RecordUsage(ctx, old); err != nil {
		t.Fatal(err)
	}
	signals := r.AnalyzePerformance(ctx, time.Hour, 0.0, 1)
	for _, s := range signals {
		if s.Kind == SignalKindSkillGenerationNeeded {
			t.Fatalf("expected no SkillGenerationNeeded for old record, got %+v", s)
		}
	}
}

// 14) usage_history filters to one guide
func TestUsageHistoryFiltersToOneGuide(t *testing.T) {
	r := NewStandardGuideRegistry()
	ctx := context.Background()
	if _, err := r.Register(ctx, makeGuide("a", "x")); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Register(ctx, makeGuide("b", "y")); err != nil {
		t.Fatal(err)
	}
	if err := r.RecordUsage(ctx, usage("a", "s1", NewOutcomeSuccess())); err != nil {
		t.Fatal(err)
	}
	if err := r.RecordUsage(ctx, usage("b", "s2", NewOutcomeSuccess())); err != nil {
		t.Fatal(err)
	}
	h, err := r.UsageHistory(ctx, GuideID("a"))
	if err != nil {
		t.Fatal(err)
	}
	if len(h) != 1 || h[0].GuideID != "a" {
		t.Fatalf("expected only a's history, got %+v", h)
	}
}

// 15) check_conflicts does not flag identical content
func TestCheckConflictsDoesNotFlagIdenticalContent(t *testing.T) {
	r := NewStandardGuideRegistry()
	ctx := context.Background()
	if _, err := r.Register(ctx, makeGuide("a", "same exact content")); err != nil {
		t.Fatal(err)
	}
	conflicts := r.CheckConflicts(ctx, "same exact content", nil)
	if len(conflicts) != 0 {
		t.Fatalf("expected no conflicts, got %+v", conflicts)
	}
}

// 16) select empty when no guides
func TestSelectEmptyWhenNoGuides(t *testing.T) {
	r := NewStandardGuideRegistry()
	res, err := r.Select(context.Background(), NewGuideQuery("anything"))
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 0 {
		t.Fatalf("expected empty, got %+v", res)
	}
}

// 17) RFC3339 round-trip sanity
func TestRFC3339RoundTrip(t *testing.T) {
	cutoff, ok := rfc3339Subtract(ts("2026-05-16T12:34:56Z"), 0)
	if !ok {
		t.Fatalf("expected parse ok")
	}
	if string(cutoff) != "2026-05-16T12:34:56Z" {
		t.Fatalf("expected round-trip, got %s", cutoff)
	}
}

// ── helpers ────────────────────────────────────────────────────────────────

func itoa(i int) string {
	// simple itoa for test session ids
	if i == 0 {
		return "0"
	}
	neg := false
	if i < 0 {
		neg = true
		i = -i
	}
	buf := make([]byte, 0, 6)
	for i > 0 {
		buf = append([]byte{byte('0' + i%10)}, buf...)
		i /= 10
	}
	if neg {
		return "-" + string(buf)
	}
	return string(buf)
}
