package sporecore

import "context"

// MemoryConfig is a configured memory source for the live loop (issue #163 /
// SC-26 follow-up). It bundles a per-turn query closure (over an object-safe
// memory.MemoryProvider) with the query policy. When present on
// HarnessConfig.Memory, the harness queries the provider each turn in
// buildContextSources and injects the returned items into ContextSources.Memory,
// which the production StandardCompactionAdapter renders into the leading
// structural System block — alongside guides + skills, with no consumer-side
// wrapper. nil (the default) leaves ContextSources.Memory empty, byte-identical
// to the pre-#163 path.
//
// The memory subpackage imports this root package, so the root cannot import it
// back (that would form a cycle). MemoryProvider therefore stays in memory; this
// config holds a function field instead, and the memory subpackage's
// NewMemoryConfig constructor closes a concrete MemoryProvider over it. Because
// the field is a closure, MemoryConfig is inherently non-serializable — adding it
// to HarnessConfig (which is not serialized) is purely additive, with no fixture
// or wire impact.
type MemoryConfig struct {
	// Query is the per-turn retrieval closure. The harness calls it each turn
	// with the effective query text, the optional domain filter, and the
	// min-relevance / max-items policy, and assigns the returned items to
	// ContextSources.Memory. A non-nil error is swallowed by the caller (empty
	// memory) — memory is best-effort context, never a halt. Constructed by
	// memory.NewMemoryConfig; nil is treated as no provider.
	Query func(ctx context.Context, taskInstruction string, domain *string, minRelevance float32, maxItems uint32) ([]MemoryItem, error)

	// QueryText, when non-nil, overrides the task instruction as the query text,
	// so retrieved memory tracks a fixed concern rather than the current task.
	// The NewMemoryConfig closure applies this override.
	QueryText *string

	// Domain optionally restricts retrieval to a single memory domain.
	Domain *string

	// MinRelevance is the minimum relevance score; items below it are dropped.
	// Defaults to 0.5 (matching memory.MemoryQuery).
	MinRelevance float32

	// MaxItems caps the number of items injected per turn. Defaults to 10.
	MaxItems uint32
}
