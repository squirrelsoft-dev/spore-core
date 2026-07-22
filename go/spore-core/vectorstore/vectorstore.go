// Package vectorstore provides the VectorStore boundary — store and
// similarity-search embedding vectors with arbitrary JSON metadata — plus a
// reference in-memory cosine implementation that is the trait's executable
// contract.
//
// The companion to EmbeddingModel: the embedder turns text into vectors, the
// store persists them with metadata and answers top-k nearest-neighbour
// queries. The trait is deliberately vector-only — "embed this text, then
// store/search it" is a composition the caller performs, keeping the storage
// backend independent of any particular embedding provider.
//
// Mirrors the Rust reference (vector_store.rs). The concrete sqlite-vec
// backend (spore-sqlvec) is a documented known gap in Go (not created).
package vectorstore

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
)

// VectorRecord is a vector to store: a caller-chosen id, the vector itself, and
// arbitrary JSON metadata (a JSON object by convention — e.g. the source text,
// a stack id, a kind discriminator). Re-adding an existing id is an upsert.
type VectorRecord struct {
	ID       string          `json:"id"`
	Vector   []float32       `json:"vector"`
	Metadata json.RawMessage `json:"metadata,omitempty"`
}

// NewVectorRecord constructs a VectorRecord.
func NewVectorRecord(id string, vector []float32, metadata json.RawMessage) VectorRecord {
	return VectorRecord{ID: id, Vector: vector, Metadata: metadata}
}

// VectorHit is one search result: the stored id, its score (HIGHER = more
// similar — backends normalise their native distance into this convention), and
// the record's metadata.
type VectorHit struct {
	ID       string          `json:"id"`
	Score    float32         `json:"score"`
	Metadata json.RawMessage `json:"metadata,omitempty"`
}

// MetadataFilter is an AND-ed set of equality predicates over top-level metadata
// keys. It expresses the common "scope this search to kind = "chunk" AND
// stack_id = "x"" need. Disjunctions are expressed by running multiple searches
// and merging.
type MetadataFilter struct {
	// Equals is the (key, value) pairs; a record matches when every key equals
	// its value.
	Equals []MetadataEqual `json:"equals,omitempty"`
}

// MetadataEqual is one (key, value) equality predicate.
type MetadataEqual struct {
	Key   string          `json:"key"`
	Value json.RawMessage `json:"value"`
}

// NewMetadataFilter returns an empty MetadataFilter.
func NewMetadataFilter() MetadataFilter { return MetadataFilter{} }

// Eq adds an equality predicate (builder style).
func (f MetadataFilter) Eq(key string, value json.RawMessage) MetadataFilter {
	f.Equals = append(f.Equals, MetadataEqual{Key: key, Value: value})
	return f
}

// IsEmpty reports whether the filter has no predicates.
func (f MetadataFilter) IsEmpty() bool { return len(f.Equals) == 0 }

// Matches reports whether a record's metadata satisfies every predicate.
// Backends that cannot push a predicate down to storage can apply it with this.
func (f MetadataFilter) Matches(metadata json.RawMessage) bool {
	if f.IsEmpty() {
		return true
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(metadata, &obj); err != nil {
		return false
	}
	for _, p := range f.Equals {
		got, ok := obj[p.Key]
		if !ok {
			return false
		}
		if string(got) != string(p.Value) {
			return false
		}
	}
	return true
}

// VectorStoreErrorKind discriminates VectorStoreError variants.
type VectorStoreErrorKind string

const (
	// VectorStoreErrDimensionMismatch: a record (or query) vector length did
	// not match the store's configured dimensionality.
	VectorStoreErrDimensionMismatch VectorStoreErrorKind = "dimension_mismatch"
	// VectorStoreErrBackend: any backend/transport failure (SQL error, I/O,
	// serialization).
	VectorStoreErrBackend VectorStoreErrorKind = "backend"
)

// VectorStoreError is the typed error for VectorStore operations. Tagged on
// "kind", snake_case, mirroring the Rust #[non_exhaustive] enum.
type VectorStoreError struct {
	Kind     VectorStoreErrorKind `json:"kind"`
	Expected int                  `json:"expected,omitempty"`
	Actual   int                  `json:"actual,omitempty"`
	Message  string               `json:"message,omitempty"`
}

// Error implements error.
func (e *VectorStoreError) Error() string {
	switch e.Kind {
	case VectorStoreErrDimensionMismatch:
		return fmt.Sprintf("vector dimension mismatch: expected %d, got %d", e.Expected, e.Actual)
	case VectorStoreErrBackend:
		return "vector store backend error: " + e.Message
	default:
		return string(e.Kind)
	}
}

// NewDimensionMismatchError constructs a DimensionMismatch error.
func NewDimensionMismatchError(expected, actual int) *VectorStoreError {
	return &VectorStoreError{Kind: VectorStoreErrDimensionMismatch, Expected: expected, Actual: actual}
}

// NewBackendError constructs a Backend error.
func NewBackendError(message string) *VectorStoreError {
	return &VectorStoreError{Kind: VectorStoreErrBackend, Message: message}
}

// VectorStore stores and similarity-searches embedding vectors.
type VectorStore interface {
	// Add inserts or upserts records (matched by id). Rejects a record whose
	// vector length differs from the store's dimensionality with a
	// DimensionMismatch error.
	Add(ctx context.Context, records []VectorRecord) error

	// Search returns up to topK nearest records to query, highest score first.
	// When filter is non-empty, only records whose metadata satisfies it are
	// considered (the filter is applied as part of the search, not as a
	// post-truncation drop).
	Search(ctx context.Context, query []float32, topK int, filter MetadataFilter) ([]VectorHit, error)

	// Delete removes every record whose metadata matches filter and returns the
	// count removed. An empty filter matches all records (a full clear).
	Delete(ctx context.Context, filter MetadataFilter) (uint64, error)

	// Len returns the number of stored vectors.
	Len(ctx context.Context) (uint64, error)

	// IsEmpty reports whether the store holds no vectors. Defaults to Len == 0
	// for implementations that do not override it.
	IsEmpty(ctx context.Context) (bool, error)
}

// ============================================================================
// InMemoryVectorStore — the reference implementation (the trait's executable
// contract). Mirrors the Rust InMemoryVectorStore test impl: cosine over a
// slice, dimension-mismatch rejection, filter pushdown into the candidate set.
// ============================================================================

// InMemoryVectorStore is an in-memory cosine VectorStore.
type InMemoryVectorStore struct {
	dimensions int
	mu         sync.Mutex
	rows       []VectorRecord
}

// NewInMemoryVectorStore constructs an InMemoryVectorStore with the given
// vector dimensionality (0 = accept any length).
func NewInMemoryVectorStore(dimensions int) *InMemoryVectorStore {
	return &InMemoryVectorStore{dimensions: dimensions}
}

// Add inserts or upserts records (matched by id).
func (s *InMemoryVectorStore) Add(_ context.Context, records []VectorRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, rec := range records {
		if s.dimensions > 0 && len(rec.Vector) != s.dimensions {
			return NewDimensionMismatchError(s.dimensions, len(rec.Vector))
		}
		// Upsert by id.
		replaced := false
		for i, existing := range s.rows {
			if existing.ID == rec.ID {
				s.rows[i] = rec
				replaced = true
				break
			}
		}
		if !replaced {
			s.rows = append(s.rows, rec)
		}
	}
	return nil
}

// Search returns up to topK nearest records to query, highest score first,
// filtered by metadata.
func (s *InMemoryVectorStore) Search(_ context.Context, query []float32, topK int, filter MetadataFilter) ([]VectorHit, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.dimensions > 0 && len(query) != s.dimensions {
		return nil, NewDimensionMismatchError(s.dimensions, len(query))
	}
	type scored struct {
		hit VectorHit
		idx int
	}
	var hits []scored
	for i, rec := range s.rows {
		if !filter.Matches(rec.Metadata) {
			continue
		}
		score := cosineSimilarity(query, rec.Vector)
		hits = append(hits, scored{hit: VectorHit{ID: rec.ID, Score: score, Metadata: rec.Metadata}, idx: i})
	}
	sort.SliceStable(hits, func(i, j int) bool {
		return hits[i].hit.Score > hits[j].hit.Score // higher = more similar first
	})
	if topK < len(hits) {
		hits = hits[:topK]
	}
	out := make([]VectorHit, len(hits))
	for i, h := range hits {
		out[i] = h.hit
	}
	return out, nil
}

// Delete removes every record whose metadata matches filter and returns the
// count removed. An empty filter matches all records (a full clear).
func (s *InMemoryVectorStore) Delete(_ context.Context, filter MetadataFilter) (uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var kept []VectorRecord
	var removed uint64
	for _, rec := range s.rows {
		if filter.Matches(rec.Metadata) {
			removed++
			continue
		}
		kept = append(kept, rec)
	}
	s.rows = kept
	return removed, nil
}

// Len returns the number of stored vectors.
func (s *InMemoryVectorStore) Len(_ context.Context) (uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return uint64(len(s.rows)), nil
}

// IsEmpty reports whether the store holds no vectors.
func (s *InMemoryVectorStore) IsEmpty(ctx context.Context) (bool, error) {
	n, err := s.Len(ctx)
	return n == 0, err
}

// cosineSimilarity is the cosine of two equal-length vectors in [-1, 1]; 0.0
// on length mismatch or zero magnitude (never NaN). Mirrors the embedding
// package's CosineSimilarity (kept local to avoid an import cycle).
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (sqrt32(na) * sqrt32(nb))
}

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
