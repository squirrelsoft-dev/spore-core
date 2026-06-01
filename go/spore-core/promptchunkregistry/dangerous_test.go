//go:build dangerous

package promptchunkregistry

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// These tests exercise the gated ModeYolo footgun (issue #34). They compile and
// run ONLY under `go test -tags dangerous ./...`; the default suite never sees
// them because ModeYolo does not exist without the build tag.

func TestDangerousModeYolo_ApprovalPolicy(t *testing.T) {
	if got := ModeYolo.ApprovalPolicy(); got != ApprovalPolicyNone {
		t.Fatalf("ModeYolo.ApprovalPolicy() = %q, want %q", got, ApprovalPolicyNone)
	}
}

func TestDangerousModeYolo_DefaultToolPhase(t *testing.T) {
	if got := ModeYolo.DefaultToolPhase(); got != sporecore.PhaseExecution {
		t.Fatalf("ModeYolo.DefaultToolPhase() = %q, want %q", got, sporecore.PhaseExecution)
	}
}

func TestDangerousModeYolo_PromptChunk(t *testing.T) {
	pc := ModeYolo.PromptChunk()
	if pc.ID != "mode-yolo" {
		t.Errorf("id = %q, want %q", pc.ID, "mode-yolo")
	}
	if !strings.HasPrefix(pc.Content, "Mode: Yolo.") {
		t.Errorf("content %q missing prefix %q", pc.Content, "Mode: Yolo.")
	}
	if pc.Slot != ChunkSlotMode || pc.CacheBlock != CacheBlockStatic {
		t.Errorf("slot=%q cache=%q", pc.Slot, pc.CacheBlock)
	}
}

func TestDangerousModeYolo_WireTag(t *testing.T) {
	if ModeYolo != Mode("yolo") {
		t.Fatalf("ModeYolo = %q, want %q", ModeYolo, "yolo")
	}
}

func TestDangerousStandardChunks_IncludeYolo(t *testing.T) {
	r := NewStandardPromptChunkRegistry()
	if err := r.RegisterStandardChunks(); err != nil {
		t.Fatalf("register standard: %v", err)
	}
	if _, ok := r.Get("mode-yolo"); !ok {
		t.Fatal("standard library missing mode-yolo under dangerous build")
	}
}

// TestDangerousFixtureReplay replays the shared dangerous.json fixture, which
// exercises ModeYolo. It is byte-identical to the file consumed by the other
// language implementations under their own dangerous gates. The default suite
// must NEVER load this file. Do not edit the fixture to make this pass.
func TestDangerousFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "prompt_chunk_registry", "dangerous.json")
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
