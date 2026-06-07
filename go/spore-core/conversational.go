package sporecore

import "context"

// conversational.go holds the production primitives the conversational harness
// preset wires by default, mirroring Rust's HarnessBuilder::conversational
// defaults (rust/crates/spore-core/src/harness.rs).
//
// The preset CONSTRUCTOR itself lives in the observability package
// (NewConversationalHarness / ConversationalConfig), not here: assembling the
// preset requires a contextmgr.StandardContextManager, and the contextmgr
// package imports sporecore — so sporecore cannot import contextmgr without a
// cycle. The observability package already imports BOTH sporecore and
// contextmgr (it bridges the rich compaction manager onto the loop seam), so it
// is the cycle-free home for the preset. The two leaf primitives below have no
// such dependency, so they stay in the root package where the docs reference
// them (and where the empty registry / sandbox / termination siblings live).

// NullSandbox is a SandboxProvider that PERMITS every tool-call validation and
// applies no path or process isolation. It embeds DefaultSandbox to inherit the
// interface's ExecuteCommand / HandleLargeOutput / ResolvePath methods — those
// paths are never reached for a pure-compute / tool-less agent, where no tool is
// ever dispatched and the environment boundary is never exercised.
//
// This is the sandbox wired by the conversational preset
// (observability.NewConversationalHarness): it is the right starting point for a
// tool-less agent. Agents that actually touch the filesystem or shell must use a
// real sandbox such as WorkspaceScopedSandbox.
//
// Mirrors Rust's spore_core::NullSandbox.
type NullSandbox struct{ DefaultSandbox }

// Validate always returns nil (no violation): the boundary is never exercised
// for a tool-less agent.
func (NullSandbox) Validate(context.Context, ToolCall) *SandboxViolation { return nil }

// CompleteOnFinalResponse is a TerminationPolicy that lets the loop complete as
// soon as the agent produces a final response: it always returns Continue,
// which the harness interprets as "accept the final response and succeed". This
// is the policy wired by the conversational preset — a tool-less chat agent
// halts naturally on its first final response, with no extra completion
// criteria to satisfy.
//
// Mirrors Rust's spore_core::CompleteOnFinalResponse. Behaviourally identical to
// AlwaysContinuePolicy; the distinct type names the intent (and matches the
// reference implementation's surface).
type CompleteOnFinalResponse struct{}

// Evaluate always returns Continue.
func (CompleteOnFinalResponse) Evaluate(context.Context, *SessionState, *BudgetSnapshot) TerminationDecision {
	return TerminationDecision{Kind: TerminationContinue}
}

// SimpleTask builds a one-shot task from just an instruction: a fresh SessionID
// and a default ReAct loop (MaxIterations 8). Use NewTask when you need to
// control the session id (e.g. multi-turn) or the loop strategy.
//
// Mirrors Rust's Task::simple.
func SimpleTask(instruction string) Task {
	return NewTask(instruction, NewSessionID(), ReActStrategy(8))
}

// Compile-time interface checks.
var (
	_ SandboxProvider   = NullSandbox{}
	_ TerminationPolicy = CompleteOnFinalResponse{}
)
