package promptchunkregistry

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Fixture-replay test: load the shared basic.json from
// fixtures/prompt_chunk_registry/ and assert each case.
//
// The fixture file is byte-identical to the one consumed by the Rust,
// TypeScript, and Python implementations. Do not edit the fixture to make
// this test pass — fix the implementation instead.

type fixtureChunk struct {
	ID         ChunkID    `json:"id"`
	Content    string     `json:"content"`
	Slot       ChunkSlot  `json:"slot"`
	CacheBlock CacheBlock `json:"cache_block"`
}

type fixtureCompose struct {
	Role         ChunkID   `json:"role"`
	Mode         Mode      `json:"mode"`
	Capabilities []ChunkID `json:"capabilities"`
	Skills       []ChunkID `json:"skills"`
}

type fixtureExpected struct {
	Slot ChunkSlot `json:"slot"`
	ID   ChunkID   `json:"id"`
}

type fixtureCase struct {
	Name             string            `json:"name"`
	RegisterInputs   []fixtureChunk    `json:"register_inputs"`
	Compose          fixtureCompose    `json:"compose"`
	ExpectedChunks   []fixtureExpected `json:"expected_chunks"`
	RenderedContains []string          `json:"rendered_contains"`
}

type fixtureFile struct {
	Cases []fixtureCase `json:"cases"`
}

func TestPromptChunkRegistryFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/promptchunkregistry → ../../../fixtures/prompt_chunk_registry/basic.json
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "prompt_chunk_registry", "basic.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", path, err)
	}
	var file fixtureFile
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(file.Cases) == 0 {
		t.Fatal("expected at least one case")
	}

	for _, c := range file.Cases {
		c := c
		t.Run(c.Name, func(t *testing.T) {
			r := NewStandardPromptChunkRegistry()
			for _, in := range c.RegisterInputs {
				if err := r.Register(NewPromptChunk(in.ID, in.Content, in.Slot, in.CacheBlock)); err != nil {
					t.Fatalf("register %q: %v", in.ID, err)
				}
			}
			composed, err := r.Compose(c.Compose.Role, c.Compose.Mode, c.Compose.Capabilities, c.Compose.Skills)
			if err != nil {
				t.Fatalf("compose: %v", err)
			}
			if got, want := len(composed.Chunks), len(c.ExpectedChunks); got != want {
				t.Fatalf("composed len = %d, want %d (got %+v)", got, want, composed.Chunks)
			}
			for i, want := range c.ExpectedChunks {
				got := composed.Chunks[i]
				if got.Slot != want.Slot || got.ID != want.ID {
					t.Errorf("chunks[%d] = (%q, %q), want (%q, %q)", i, got.Slot, got.ID, want.Slot, want.ID)
				}
			}
			if vErrs := r.Validate(&composed); len(vErrs) != 0 {
				t.Errorf("validate should pass, got %v", vErrs)
			}
			rendered := composed.Render()
			for _, needle := range c.RenderedContains {
				if !strings.Contains(rendered, needle) {
					t.Errorf("rendered missing %q; got %q", needle, rendered)
				}
			}
		})
	}
}
