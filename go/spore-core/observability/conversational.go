// Conversational harness preset (mirrors Rust's HarnessBuilder::conversational,
// rust/crates/spore-core/src/harness.rs).
//
// This is the few-lines path: it defaults every required component so you can go
// from a model to a running harness in one call. It lives in the observability
// package — not the root sporecore package — because assembling the standard
// context-manager default requires contextmgr.StandardContextManager, and the
// contextmgr package imports sporecore (so sporecore cannot import contextmgr
// without a cycle). The observability package already imports BOTH, so it is the
// cycle-free home that keeps the API ergonomic and close to the reference.

package observability

import (
	"sync/atomic"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
)

// ConversationalBuilder returns a HarnessBuilder pre-wired with the
// conversational defaults, so callers can override individual components (add a
// tool registry, swap the sandbox, attach observability) before Build().
//
// The defaults mirror Rust's HarnessBuilder::conversational:
//
//   - agent:              a ModelAgent over model, with AgentID "agent"
//   - tool registry:      an empty *StandardToolRegistry (no tools advertised)
//   - sandbox:            NullSandbox (permits validation; no isolation)
//   - context manager:    a StandardContextManager built from the model with a
//     null cache provider and the default compaction config, bridged onto the
//     harness loop seam via the standard compaction adapter
//   - termination policy: CompleteOnFinalResponse (the model's first final
//     response is the result)
//
// Every default is overridable through the returned builder's setters.
func ConversationalBuilder(model sporecore.ModelInterface) *HarnessBuilder {
	// Adaptive prompt-based tool-calling fallback (#111). Create ONE shared flag
	// (false) and wrap the agent's model in an AdaptiveToolCallModelInterface over
	// it; the run loop flips the SAME flag on detecting a prose response. While
	// the flag is unset the wrapper delegates natively, so this is byte-for-byte
	// the prior behaviour until escalation fires. The context manager keeps the
	// RAW model (its summarization calls must never be wrapped). This fallback is
	// enabled in the conversational preset ONLY.
	flag := &atomic.Bool{}
	agentModel := sporecore.NewAdaptiveToolCallModelInterface(model, flag)

	agent := sporecore.NewModelAgent(sporecore.AgentID("agent"), agentModel)
	toolRegistry := sporecore.NewStandardToolRegistry() // empty: ActiveSchemas -> none
	sandbox := sporecore.NullSandbox{}
	contextManager := contextmgr.NewStandardCompactionAdapter(
		contextmgr.NewStandardContextManager(
			model,
			contextmgr.NullCacheProvider{},
			contextmgr.DefaultCompactionConfig(),
		),
	)
	termination := sporecore.CompleteOnFinalResponse{}
	b := NewHarnessBuilder(agent, toolRegistry, sandbox, contextManager, termination)
	b.promptToolCallFlag = flag
	return b
}

// NewConversationalHarness assembles a minimal conversational harness from a
// model — no tools, no filesystem. It is the one-call path equivalent to
// ConversationalBuilder(model).Build(). See ConversationalBuilder for the full
// list of defaults.
//
// Example:
//
//	model := ollama.New("llama3.2")
//	h := observability.NewConversationalHarness(model)
//	r := h.Run(ctx, sporecore.NewHarnessRunOptions(sporecore.SimpleTask("Say hello.")))
func NewConversationalHarness(model sporecore.ModelInterface) *sporecore.StandardHarness {
	return ConversationalBuilder(model).Build()
}
