package memory

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"unicode"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// StandardMemoryProvider is the reference in-memory MemoryProvider.
//
// Relevance scoring uses token-overlap (Jaccard) between the query's
// TaskInstruction and each candidate memory's Content. The spec calls for
// embedding similarity in production; the interface does not mandate the
// scoring algorithm, only that items below MinRelevance are not returned.
type StandardMemoryProvider struct {
	mu         sync.Mutex
	episodic   map[sporecore.SessionID][]EpisodicMemory
	semantic   map[MemoryID]SemanticMemory
	archiveSeq uint64
}

// NewStandardMemoryProvider constructs an empty provider.
func NewStandardMemoryProvider() *StandardMemoryProvider {
	return &StandardMemoryProvider{
		episodic: make(map[sporecore.SessionID][]EpisodicMemory),
		semantic: make(map[MemoryID]SemanticMemory),
	}
}

// ── Helpers ────────────────────────────────────────────────────────────────

func validateContent(s string) error {
	if strings.TrimSpace(s) == "" {
		return &MemoryError{Kind: ErrKindValidationFailed, Reason: "content must not be empty"}
	}
	return nil
}

// enforceMetaAgentReview forces MetaAgentProposed memories into PendingReview
// regardless of caller-supplied status.
func enforceMetaAgentReview(m *SemanticMemory) {
	if m.Source.Kind == SourceKindMetaAgentProposed && m.Status.Kind != StatusKindPendingReview {
		m.Status = NewStatusPendingReview(m.CreatedAt)
	}
}

func (p *StandardMemoryProvider) nextArchiveID(original MemoryID) MemoryID {
	p.archiveSeq++
	return MemoryID(fmt.Sprintf("%s@v%d", string(original), p.archiveSeq))
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

// ── Episodic ───────────────────────────────────────────────────────────────

// StoreEpisodic persists an episodic memory.
func (p *StandardMemoryProvider) StoreEpisodic(_ context.Context, m EpisodicMemory) (MemoryID, error) {
	if err := validateContent(m.Content); err != nil {
		return "", err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.episodic[m.SessionID] = append(p.episodic[m.SessionID], m)
	return m.ID, nil
}

// GetEpisodic returns all episodic memories for a session, preserving insertion order.
func (p *StandardMemoryProvider) GetEpisodic(_ context.Context, sessionID sporecore.SessionID) ([]EpisodicMemory, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	src := p.episodic[sessionID]
	out := make([]EpisodicMemory, len(src))
	copy(out, src)
	return out, nil
}

// ── Semantic ───────────────────────────────────────────────────────────────

// StoreSemantic persists a semantic memory, honoring onConflict on collisions.
func (p *StandardMemoryProvider) StoreSemantic(_ context.Context, memory SemanticMemory, onConflict MergeStrategy) (MemoryID, error) {
	if err := validateContent(memory.Content); err != nil {
		return "", err
	}
	enforceMetaAgentReview(&memory)

	p.mu.Lock()
	defer p.mu.Unlock()

	id := memory.ID
	if existing, ok := p.semantic[id]; ok {
		switch onConflict {
		case MergeStrategyReject:
			return "", &MemoryError{
				Kind:     ErrKindMergeConflict,
				Existing: existing.ID,
				Reason:   "memory exists; on_conflict=Reject",
			}
		case MergeStrategyAppend:
			merged := existing
			merged.Content = merged.Content + memory.Content
			merged.UpdatedAt = memory.UpdatedAt
			p.semantic[id] = merged
			return id, nil
		case MergeStrategyReplace:
			archiveID := p.nextArchiveID(existing.ID)
			archived := existing
			archived.ID = archiveID
			p.semantic[archiveID] = archived

			memory.Version = existing.Version + 1
			prev := make([]MemoryID, 0, len(existing.PreviousVersions)+1)
			prev = append(prev, existing.PreviousVersions...)
			prev = append(prev, archiveID)
			memory.PreviousVersions = prev
		}
	}
	p.semantic[id] = memory
	return id, nil
}

// GetSemantic returns a single semantic memory by id.
func (p *StandardMemoryProvider) GetSemantic(_ context.Context, id MemoryID) (SemanticMemory, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	m, ok := p.semantic[id]
	if !ok {
		return SemanticMemory{}, &MemoryError{Kind: ErrKindNotFound, ID: id}
	}
	return m, nil
}

// Query returns scored Active semantic memories above MinRelevance, sorted desc.
func (p *StandardMemoryProvider) Query(_ context.Context, q MemoryQuery) ([]MemoryItem, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	scored := make([]MemoryItem, 0)
	for _, m := range p.semantic {
		if m.Status.Kind != StatusKindActive {
			continue
		}
		// domain filter: Some(d) only matches Some(d); None matches both
		if q.Domain != nil {
			if m.Domain == nil || *m.Domain != *q.Domain {
				continue
			}
		}
		score := jaccard(q.TaskInstruction, m.Content)
		if score < q.MinRelevance {
			continue
		}
		scored = append(scored, MemoryItem{Memory: m, RelevanceScore: score})
	}
	sort.SliceStable(scored, func(i, j int) bool {
		return scored[i].RelevanceScore > scored[j].RelevanceScore
	})
	if uint32(len(scored)) > q.MaxItems {
		scored = scored[:q.MaxItems]
	}
	return scored, nil
}

// Deprecate sets the status to Deprecated{Reason, At=UpdatedAt}.
func (p *StandardMemoryProvider) Deprecate(_ context.Context, id MemoryID, reason string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	m, ok := p.semantic[id]
	if !ok {
		return &MemoryError{Kind: ErrKindNotFound, ID: id}
	}
	m.Status = NewStatusDeprecated(reason, m.UpdatedAt)
	p.semantic[id] = m
	return nil
}

// GetVersionHistory returns [current, prev_chain_in_order].
func (p *StandardMemoryProvider) GetVersionHistory(_ context.Context, id MemoryID) ([]SemanticMemory, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	head, ok := p.semantic[id]
	if !ok {
		return nil, &MemoryError{Kind: ErrKindNotFound, ID: id}
	}
	chain := []SemanticMemory{head}
	for _, prevID := range head.PreviousVersions {
		if prev, ok := p.semantic[prevID]; ok {
			chain = append(chain, prev)
		}
	}
	return chain, nil
}

// MarkPendingReview sets the status to PendingReview{ProposedAt=UpdatedAt}.
func (p *StandardMemoryProvider) MarkPendingReview(_ context.Context, id MemoryID) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	m, ok := p.semantic[id]
	if !ok {
		return &MemoryError{Kind: ErrKindNotFound, ID: id}
	}
	m.Status = NewStatusPendingReview(m.UpdatedAt)
	p.semantic[id] = m
	return nil
}

// Compile-time interface check.
var _ MemoryProvider = (*StandardMemoryProvider)(nil)
