package contextmgr

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// Fixture-replay test: load the shared parse_cache_stats.json from
// fixtures/cache_provider/ and assert each case.
//
// The fixture is byte-identical across Rust / TypeScript / Python / Go. The
// `is_some` field is the cross-language way of expressing the Go pattern
// `(_, ok)`; we adapt it via the fixtureExpected struct below. Do not edit
// the fixture to make this test pass — fix the implementation instead.

type fixtureUsage struct {
	CacheReadTokens  *uint32 `json:"cache_read_tokens"`
	CacheWriteTokens *uint32 `json:"cache_write_tokens"`
}

type fixtureExpected struct {
	IsSome           bool   `json:"is_some"`
	CacheReadTokens  uint32 `json:"cache_read_tokens"`
	CacheWriteTokens uint32 `json:"cache_write_tokens"`
}

type fixtureCase struct {
	Name     string          `json:"name"`
	Provider string          `json:"provider"`
	Usage    fixtureUsage    `json:"usage"`
	Expected fixtureExpected `json:"expected"`
}

type fixtureFile struct {
	Cases []fixtureCase `json:"cases"`
}

func TestCacheProviderFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/contextmgr → ../../../fixtures/cache_provider/parse_cache_stats.json
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "cache_provider", "parse_cache_stats.json")
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
			resp := &sporecore.ModelResponse{
				Content: []sporecore.ContentBlock{sporecore.NewTextBlock("hi")},
				Usage: sporecore.TokenUsage{
					CacheReadTokens:  c.Usage.CacheReadTokens,
					CacheWriteTokens: c.Usage.CacheWriteTokens,
				},
				StopReason: sporecore.StopEndTurn,
			}

			var (
				stats CacheStats
				ok    bool
			)
			switch c.Provider {
			case "anthropic":
				stats, ok = NewAnthropicCacheProvider().ParseCacheStats(resp)
			case "openai":
				stats, ok = NewOpenAICacheProvider().ParseCacheStats(resp)
			case "ollama":
				stats, ok = OllamaCacheProvider{}.ParseCacheStats(resp)
			case "null":
				stats, ok = NullCacheProvider{}.ParseCacheStats(resp)
			default:
				t.Fatalf("unknown provider in fixture: %q", c.Provider)
			}

			if ok != c.Expected.IsSome {
				t.Fatalf("ok = %v, want %v", ok, c.Expected.IsSome)
			}
			if !ok {
				return
			}
			if stats.CacheReadTokens != c.Expected.CacheReadTokens {
				t.Errorf("CacheReadTokens = %d, want %d",
					stats.CacheReadTokens, c.Expected.CacheReadTokens)
			}
			if stats.CacheWriteTokens != c.Expected.CacheWriteTokens {
				t.Errorf("CacheWriteTokens = %d, want %d",
					stats.CacheWriteTokens, c.Expected.CacheWriteTokens)
			}
		})
	}
}
