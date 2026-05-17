package memory

import (
	"context"
	"errors"
	"fmt"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func ts(s string) Timestamp { return Timestamp(s) }

func sem(id, content string) SemanticMemory {
	return SemanticMemory{
		ID:               MemoryID(id),
		Content:          content,
		Source:           NewSourceManual(),
		Domain:           nil,
		Version:          1,
		PreviousVersions: nil,
		CreatedAt:        ts("2026-05-16T00:00:00Z"),
		UpdatedAt:        ts("2026-05-16T00:00:00Z"),
		Status:           NewStatusActive(),
	}
}

func epi(id, session, content string) EpisodicMemory {
	return EpisodicMemory{
		ID:        MemoryID(id),
		SessionID: sporecore.SessionID(session),
		Content:   content,
		CreatedAt: ts("2026-05-16T00:00:00Z"),
		Tags:      nil,
	}
}

func isKind(err error, kind MemoryErrorKind) bool {
	var me *MemoryError
	if !errors.As(err, &me) {
		return false
	}
	return me.Kind == kind
}

// ── Rule: Episodic and semantic stored/retrieved separately ─────────────────

func TestEpisodicAndSemanticUseSeparateStores(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	if _, err := mp.StoreEpisodic(ctx, epi("e1", "s1", "ran tests")); err != nil {
		t.Fatal(err)
	}
	if _, err := mp.StoreSemantic(ctx, sem("g1", "always run tests"), MergeStrategyReject); err != nil {
		t.Fatal(err)
	}

	eps, err := mp.GetEpisodic(ctx, sporecore.SessionID("s1"))
	if err != nil {
		t.Fatal(err)
	}
	if len(eps) != 1 {
		t.Fatalf("want 1 episodic, got %d", len(eps))
	}
	if _, err := mp.GetSemantic(ctx, MemoryID("e1")); err == nil {
		t.Fatalf("expected NotFound for episodic id in semantic store")
	}
	other, _ := mp.GetEpisodic(ctx, sporecore.SessionID("g1"))
	if len(other) != 0 {
		t.Fatalf("semantic id leaked into episodic")
	}
}

// ── Rule: Replace creates a new version, retains previous ───────────────────

func TestReplaceArchivesPreviousAndBumpsVersion(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	if _, err := mp.StoreSemantic(ctx, sem("g1", "v1 content"), MergeStrategyReject); err != nil {
		t.Fatal(err)
	}
	v2 := sem("g1", "v2 content")
	v2.Version = 1
	if _, err := mp.StoreSemantic(ctx, v2, MergeStrategyReplace); err != nil {
		t.Fatal(err)
	}

	cur, err := mp.GetSemantic(ctx, MemoryID("g1"))
	if err != nil {
		t.Fatal(err)
	}
	if cur.Content != "v2 content" {
		t.Fatalf("content: %q", cur.Content)
	}
	if cur.Version != 2 {
		t.Fatalf("version: %d", cur.Version)
	}
	if len(cur.PreviousVersions) != 1 {
		t.Fatalf("previous_versions: %d", len(cur.PreviousVersions))
	}

	hist, err := mp.GetVersionHistory(ctx, MemoryID("g1"))
	if err != nil {
		t.Fatal(err)
	}
	if len(hist) != 2 || hist[0].Content != "v2 content" || hist[1].Content != "v1 content" {
		t.Fatalf("unexpected history: %+v", hist)
	}
}

func TestReplaceChainsVersionsAcrossMultipleUpdates(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	if _, err := mp.StoreSemantic(ctx, sem("g1", "v1"), MergeStrategyReject); err != nil {
		t.Fatal(err)
	}
	if _, err := mp.StoreSemantic(ctx, sem("g1", "v2"), MergeStrategyReplace); err != nil {
		t.Fatal(err)
	}
	if _, err := mp.StoreSemantic(ctx, sem("g1", "v3"), MergeStrategyReplace); err != nil {
		t.Fatal(err)
	}
	cur, _ := mp.GetSemantic(ctx, MemoryID("g1"))
	if cur.Version != 3 {
		t.Fatalf("version: %d", cur.Version)
	}
	hist, _ := mp.GetVersionHistory(ctx, MemoryID("g1"))
	if len(hist) != 3 {
		t.Fatalf("history len: %d", len(hist))
	}
}

// ── Rule: Reject returns MergeConflict ──────────────────────────────────────

func TestRejectOnConflictErrors(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	if _, err := mp.StoreSemantic(ctx, sem("g1", "first"), MergeStrategyReject); err != nil {
		t.Fatal(err)
	}
	_, err := mp.StoreSemantic(ctx, sem("g1", "second"), MergeStrategyReject)
	if !isKind(err, ErrKindMergeConflict) {
		t.Fatalf("want MergeConflict, got %v", err)
	}
	cur, _ := mp.GetSemantic(ctx, MemoryID("g1"))
	if cur.Content != "first" {
		t.Fatalf("original mutated: %q", cur.Content)
	}
}

// ── Rule: Append concatenates in place, no new version ─────────────────────

func TestAppendConcatenatesWithoutNewVersion(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	if _, err := mp.StoreSemantic(ctx, sem("g1", "a"), MergeStrategyReject); err != nil {
		t.Fatal(err)
	}
	if _, err := mp.StoreSemantic(ctx, sem("g1", "b"), MergeStrategyAppend); err != nil {
		t.Fatal(err)
	}
	cur, _ := mp.GetSemantic(ctx, MemoryID("g1"))
	if cur.Content != "ab" {
		t.Fatalf("content: %q", cur.Content)
	}
	if cur.Version != 1 {
		t.Fatalf("version: %d", cur.Version)
	}
	if len(cur.PreviousVersions) != 0 {
		t.Fatalf("previous_versions: %d", len(cur.PreviousVersions))
	}
}

// ── Rule: Writes validated ─────────────────────────────────────────────────

func TestEmptyContentFailsValidation(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	_, err := mp.StoreSemantic(ctx, sem("g1", "   "), MergeStrategyReject)
	if !isKind(err, ErrKindValidationFailed) {
		t.Fatalf("want ValidationFailed, got %v", err)
	}
	_, err = mp.StoreEpisodic(ctx, epi("e1", "s1", ""))
	if !isKind(err, ErrKindValidationFailed) {
		t.Fatalf("want ValidationFailed for episodic, got %v", err)
	}
}

// ── Rule: MetaAgentProposed forced to PendingReview ────────────────────────

func TestMetaAgentMemoriesForcedToPendingReview(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	m := sem("g1", "proposed skill")
	m.Source = NewSourceMetaAgentProposed(nil)
	m.Status = NewStatusActive() // caller dishonestly sets Active
	if _, err := mp.StoreSemantic(ctx, m, MergeStrategyReject); err != nil {
		t.Fatal(err)
	}
	stored, _ := mp.GetSemantic(ctx, MemoryID("g1"))
	if stored.Status.Kind != StatusKindPendingReview {
		t.Fatalf("want PendingReview, got %v", stored.Status.Kind)
	}
}

// ── Rule: query scores, filters, sorts ─────────────────────────────────────

func TestQueryScoresFiltersAndSorts(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	for _, m := range []SemanticMemory{
		sem("g1", "rust async tokio runtime"),
		sem("g2", "python pytest fixtures"),
		sem("g3", "unrelated cooking recipe"),
	} {
		if _, err := mp.StoreSemantic(ctx, m, MergeStrategyReject); err != nil {
			t.Fatal(err)
		}
	}
	q := MemoryQuery{TaskInstruction: "rust tokio async", MinRelevance: 0.1, MaxItems: 10}
	res, err := mp.Query(ctx, q)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("expected results")
	}
	if res[0].Memory.ID != MemoryID("g1") {
		t.Fatalf("top result: %v", res[0].Memory.ID)
	}
	for i := 1; i < len(res); i++ {
		if res[i-1].RelevanceScore < res[i].RelevanceScore {
			t.Fatalf("not sorted desc")
		}
	}
}

func TestQueryExcludesDeprecatedMemories(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	if _, err := mp.StoreSemantic(ctx, sem("g1", "rust tokio"), MergeStrategyReject); err != nil {
		t.Fatal(err)
	}
	if err := mp.Deprecate(ctx, MemoryID("g1"), "obsolete"); err != nil {
		t.Fatal(err)
	}
	res, _ := mp.Query(ctx, MemoryQuery{TaskInstruction: "rust tokio", MinRelevance: 0, MaxItems: 10})
	if len(res) != 0 {
		t.Fatalf("deprecated returned: %+v", res)
	}
}

func TestQueryRespectsMinRelevanceAndMaxItems(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	for i := 0; i < 5; i++ {
		_, err := mp.StoreSemantic(ctx, sem(fmt.Sprintf("g%d", i), fmt.Sprintf("alpha beta gamma %d", i)), MergeStrategyReject)
		if err != nil {
			t.Fatal(err)
		}
	}
	res, _ := mp.Query(ctx, MemoryQuery{TaskInstruction: "alpha beta gamma", MinRelevance: 0.99, MaxItems: 10})
	if len(res) != 0 {
		t.Fatalf("min_relevance not enforced: %d", len(res))
	}
	res, _ = mp.Query(ctx, MemoryQuery{TaskInstruction: "alpha beta gamma", MinRelevance: 0, MaxItems: 2})
	if len(res) != 2 {
		t.Fatalf("max_items not enforced: %d", len(res))
	}
}

func TestQueryFiltersByDomain(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	rust, py := "rust", "python"
	a := sem("a", "shared content")
	a.Domain = &rust
	b := sem("b", "shared content")
	b.Domain = &py
	if _, err := mp.StoreSemantic(ctx, a, MergeStrategyReject); err != nil {
		t.Fatal(err)
	}
	if _, err := mp.StoreSemantic(ctx, b, MergeStrategyReject); err != nil {
		t.Fatal(err)
	}
	res, _ := mp.Query(ctx, MemoryQuery{
		TaskInstruction: "shared content",
		Domain:          &rust,
		MinRelevance:    0,
		MaxItems:        10,
	})
	if len(res) != 1 || res[0].Memory.ID != MemoryID("a") {
		t.Fatalf("domain filter failed: %+v", res)
	}
}

// ── Rule: Lifecycle — deprecate ────────────────────────────────────────────

func TestDeprecateSetsStatus(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	if _, err := mp.StoreSemantic(ctx, sem("g1", "x"), MergeStrategyReject); err != nil {
		t.Fatal(err)
	}
	if err := mp.Deprecate(ctx, MemoryID("g1"), "no longer needed"); err != nil {
		t.Fatal(err)
	}
	m, _ := mp.GetSemantic(ctx, MemoryID("g1"))
	if m.Status.Kind != StatusKindDeprecated || m.Status.Reason != "no longer needed" {
		t.Fatalf("status: %+v", m.Status)
	}
}

func TestDeprecateUnknownIDNotFound(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	err := mp.Deprecate(ctx, MemoryID("nope"), "r")
	if !isKind(err, ErrKindNotFound) {
		t.Fatalf("want NotFound, got %v", err)
	}
}

// ── Rule: mark_pending_review ──────────────────────────────────────────────

func TestMarkPendingReviewChangesStatus(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	if _, err := mp.StoreSemantic(ctx, sem("g1", "x"), MergeStrategyReject); err != nil {
		t.Fatal(err)
	}
	if err := mp.MarkPendingReview(ctx, MemoryID("g1")); err != nil {
		t.Fatal(err)
	}
	m, _ := mp.GetSemantic(ctx, MemoryID("g1"))
	if m.Status.Kind != StatusKindPendingReview {
		t.Fatalf("status: %+v", m.Status)
	}
}

// ── Rule: NotFound errors ──────────────────────────────────────────────────

func TestGetSemanticUnknownReturnsNotFound(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	_, err := mp.GetSemantic(ctx, MemoryID("nope"))
	if !isKind(err, ErrKindNotFound) {
		t.Fatalf("want NotFound, got %v", err)
	}
}

func TestGetEpisodicUnknownSessionReturnsEmpty(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	res, err := mp.GetEpisodic(ctx, sporecore.SessionID("none"))
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 0 {
		t.Fatalf("expected empty, got %d", len(res))
	}
}

// ── Episodic preserves insertion order ─────────────────────────────────────

func TestEpisodicPreservesOrder(t *testing.T) {
	ctx := context.Background()
	mp := NewStandardMemoryProvider()
	for i := 0; i < 5; i++ {
		if _, err := mp.StoreEpisodic(ctx, epi(fmt.Sprintf("e%d", i), "s1", fmt.Sprintf("event %d", i))); err != nil {
			t.Fatal(err)
		}
	}
	eps, _ := mp.GetEpisodic(ctx, sporecore.SessionID("s1"))
	if len(eps) != 5 {
		t.Fatalf("len: %d", len(eps))
	}
	for i, e := range eps {
		if e.ID != MemoryID(fmt.Sprintf("e%d", i)) {
			t.Fatalf("order broken at %d: %v", i, e.ID)
		}
	}
}

// ── NewMemoryQuery defaults ────────────────────────────────────────────────

func TestNewMemoryQueryDefaults(t *testing.T) {
	q := NewMemoryQuery("hello")
	if q.TaskInstruction != "hello" {
		t.Fatalf("instr: %q", q.TaskInstruction)
	}
	if q.MinRelevance != 0.5 {
		t.Fatalf("min: %v", q.MinRelevance)
	}
	if q.MaxItems != 10 {
		t.Fatalf("max: %v", q.MaxItems)
	}
}
