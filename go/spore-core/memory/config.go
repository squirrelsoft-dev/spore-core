package memory

import (
	"context"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// MemoryConfigOption customizes the MemoryConfig built by NewMemoryConfig,
// mirroring the fluent setters on the Rust MemoryConfig (query / domain /
// min_relevance / max_items). Apply them as trailing arguments:
//
//	memory.NewMemoryConfig(provider,
//		memory.WithMinRelevance(0.6),
//		memory.WithMaxItems(5))
type MemoryConfigOption func(*sporecore.MemoryConfig)

// WithQuery overrides the per-turn query text. Without it the current task
// instruction is used, so retrieved memory tracks what the agent is working on.
func WithQuery(query string) MemoryConfigOption {
	return func(c *sporecore.MemoryConfig) { c.QueryText = &query }
}

// WithDomain restricts retrieval to a single memory domain.
func WithDomain(domain string) MemoryConfigOption {
	return func(c *sporecore.MemoryConfig) { c.Domain = &domain }
}

// WithMinRelevance overrides the minimum relevance threshold (default 0.5).
func WithMinRelevance(minRelevance float32) MemoryConfigOption {
	return func(c *sporecore.MemoryConfig) { c.MinRelevance = minRelevance }
}

// WithMaxItems overrides the per-turn item cap (default 10).
func WithMaxItems(maxItems uint32) MemoryConfigOption {
	return func(c *sporecore.MemoryConfig) { c.MaxItems = maxItems }
}

// NewMemoryConfig builds a sporecore.MemoryConfig wiring an object-safe
// MemoryProvider into the harness (issue #163 / SC-26 follow-up).
//
// The memory subpackage imports the root sporecore package, so the root cannot
// reference MemoryProvider directly without an import cycle. This constructor
// bridges the two: it closes the provider's Query over a function field on the
// returned config. Each turn the harness calls that closure with the effective
// query text (the harness applies the QueryText override over the task
// instruction before calling), the domain filter, and the min-relevance /
// max-items policy; the closure builds a MemoryQuery, calls provider.Query, and
// maps each scored MemoryItem onto the root sporecore.MemoryItem the structural
// System block renders.
//
// The default query policy matches MemoryQuery: query by the current task
// instruction, MinRelevance = 0.5, MaxItems = 10, no domain filter. Override any
// of these with the MemoryConfigOption setters.
func NewMemoryConfig(p MemoryProvider, opts ...MemoryConfigOption) sporecore.MemoryConfig {
	cfg := sporecore.MemoryConfig{
		MinRelevance: 0.5,
		MaxItems:     10,
		Query: func(ctx context.Context, taskInstruction string, domain *string, minRelevance float32, maxItems uint32) ([]sporecore.MemoryItem, error) {
			q := MemoryQuery{
				TaskInstruction: taskInstruction,
				Domain:          domain,
				MinRelevance:    minRelevance,
				MaxItems:        maxItems,
			}
			items, err := p.Query(ctx, q)
			if err != nil {
				return nil, err
			}
			out := make([]sporecore.MemoryItem, len(items))
			for i, it := range items {
				out[i] = sporecore.MemoryItem{
					Key:     string(it.Memory.ID),
					Content: it.Memory.Content,
				}
			}
			return out, nil
		},
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}
