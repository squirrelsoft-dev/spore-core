package vectorstore

import (
	"context"
	"encoding/json"
	"testing"
)

func TestAddBatchAndLen(t *testing.T) {
	store := NewInMemoryVectorStore(3)
	records := []VectorRecord{
		{ID: "a", Vector: []float32{1, 0, 0}, Metadata: rawJSON(`{"kind":"doc"}`)},
		{ID: "b", Vector: []float32{0, 1, 0}, Metadata: rawJSON(`{"kind":"doc"}`)},
		{ID: "c", Vector: []float32{0, 0, 1}, Metadata: rawJSON(`{"kind":"chunk"}`)},
	}
	if err := store.Add(context.Background(), records); err != nil {
		t.Fatal(err)
	}
	n, _ := store.Len(context.Background())
	if n != 3 {
		t.Fatalf("len = %d, want 3", n)
	}
}

func TestAddUpsertsByID(t *testing.T) {
	store := NewInMemoryVectorStore(2)
	_ = store.Add(context.Background(), []VectorRecord{{ID: "x", Vector: []float32{1, 0}}})
	_ = store.Add(context.Background(), []VectorRecord{{ID: "x", Vector: []float32{0, 1}}})
	n, _ := store.Len(context.Background())
	if n != 1 {
		t.Fatalf("len = %d, want 1 (upsert)", n)
	}
	hits, _ := store.Search(context.Background(), []float32{0, 1}, 1, MetadataFilter{})
	if len(hits) != 1 || hits[0].ID != "x" {
		t.Fatalf("hits = %v", hits)
	}
}

func TestAddDimensionMismatch(t *testing.T) {
	store := NewInMemoryVectorStore(3)
	err := store.Add(context.Background(), []VectorRecord{{ID: "x", Vector: []float32{1, 0}}})
	if err == nil {
		t.Fatal("expected a dimension mismatch")
	}
	vse, ok := err.(*VectorStoreError)
	if !ok || vse.Kind != VectorStoreErrDimensionMismatch || vse.Expected != 3 || vse.Actual != 2 {
		t.Fatalf("err = %v", err)
	}
}

func TestSearchTopKOrdersByScore(t *testing.T) {
	store := NewInMemoryVectorStore(3)
	_ = store.Add(context.Background(), []VectorRecord{
		{ID: "a", Vector: []float32{1, 0, 0}},
		{ID: "b", Vector: []float32{0.9, 0.1, 0}},
		{ID: "c", Vector: []float32{0, 1, 0}},
	})
	hits, err := store.Search(context.Background(), []float32{1, 0, 0}, 2, MetadataFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 {
		t.Fatalf("len = %d, want 2", len(hits))
	}
	if hits[0].ID != "a" {
		t.Errorf("first hit = %q, want a (highest score)", hits[0].ID)
	}
	if hits[1].ID != "b" {
		t.Errorf("second hit = %q, want b", hits[1].ID)
	}
}

func TestSearchFilterPushdown(t *testing.T) {
	store := NewInMemoryVectorStore(3)
	_ = store.Add(context.Background(), []VectorRecord{
		{ID: "a", Vector: []float32{1, 0, 0}, Metadata: rawJSON(`{"kind":"doc"}`)},
		{ID: "b", Vector: []float32{0.99, 0, 0}, Metadata: rawJSON(`{"kind":"chunk"}`)},
	})
	// Filter to kind=chunk: only "b" is a candidate, so "a" (the exact match)
	// must NOT appear even though it scores higher.
	filter := NewMetadataFilter().Eq("kind", rawJSON(`"chunk"`))
	hits, err := store.Search(context.Background(), []float32{1, 0, 0}, 5, filter)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].ID != "b" {
		t.Fatalf("hits = %v, want only b", hits)
	}
}

func TestSearchNegativeTopKReturnsEmpty(t *testing.T) {
	store := NewInMemoryVectorStore(2)
	_ = store.Add(context.Background(), []VectorRecord{{ID: "a", Vector: []float32{1, 0}}})
	hits, err := store.Search(context.Background(), []float32{1, 0}, -1, MetadataFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Fatalf("hits=%v, want empty for negative topK", hits)
	}
}

func TestMetadataFilterUsesSemanticJSONEquality(t *testing.T) {
	store := NewInMemoryVectorStore(2)
	_ = store.Add(context.Background(), []VectorRecord{{
		ID:       "a",
		Vector:   []float32{1, 0},
		Metadata: rawJSON(`{"attrs":{"b":2,"a":1}}`),
	}})
	filter := NewMetadataFilter().Eq("attrs", rawJSON(`{"a":1, "b":2}`))
	hits, err := store.Search(context.Background(), []float32{1, 0}, 1, filter)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].ID != "a" {
		t.Fatalf("hits=%v, want semantic JSON object match", hits)
	}
}

func TestDeleteByFilterReturnsCount(t *testing.T) {
	store := NewInMemoryVectorStore(3)
	_ = store.Add(context.Background(), []VectorRecord{
		{ID: "a", Vector: []float32{1, 0, 0}, Metadata: rawJSON(`{"kind":"doc"}`)},
		{ID: "b", Vector: []float32{0, 1, 0}, Metadata: rawJSON(`{"kind":"chunk"}`)},
		{ID: "c", Vector: []float32{0, 0, 1}, Metadata: rawJSON(`{"kind":"chunk"}`)},
	})
	removed, err := store.Delete(context.Background(), NewMetadataFilter().Eq("kind", rawJSON(`"chunk"`)))
	if err != nil {
		t.Fatal(err)
	}
	if removed != 2 {
		t.Fatalf("removed = %d, want 2", removed)
	}
	n, _ := store.Len(context.Background())
	if n != 1 {
		t.Fatalf("len after delete = %d, want 1", n)
	}
}

func TestDeleteEmptyFilterIsFullClear(t *testing.T) {
	store := NewInMemoryVectorStore(2)
	_ = store.Add(context.Background(), []VectorRecord{
		{ID: "a", Vector: []float32{1, 0}},
		{ID: "b", Vector: []float32{0, 1}},
	})
	removed, err := store.Delete(context.Background(), MetadataFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if removed != 2 {
		t.Fatalf("removed = %d, want 2", removed)
	}
	empty, _ := store.IsEmpty(context.Background())
	if !empty {
		t.Fatal("expected empty after full clear")
	}
}

func TestMetadataFilterMatches(t *testing.T) {
	f := NewMetadataFilter().Eq("kind", rawJSON(`"doc"`)).Eq("stack", rawJSON(`"x"`))
	if !f.Matches(rawJSON(`{"kind":"doc","stack":"x"}`)) {
		t.Error("expected match")
	}
	if f.Matches(rawJSON(`{"kind":"chunk","stack":"x"}`)) {
		t.Error("expected no match (kind mismatch)")
	}
	if f.Matches(rawJSON(`{"kind":"doc"}`)) {
		t.Error("expected no match (missing stack)")
	}
	emptyFilter := MetadataFilter{}
	if !emptyFilter.Matches(rawJSON(`{}`)) {
		t.Error("empty filter matches everything")
	}
}

func rawJSON(s string) json.RawMessage { return json.RawMessage(s) }
