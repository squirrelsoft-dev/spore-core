// Compaction adapter (issue #55) — bridges the rich StandardContextManager
// onto the harness-loop compaction seam introduced in #46.
//
// #46 wired the verify→retry→warn machinery into the harness loop and proved it
// with test-double context managers. The rich StandardContextManager (#29/#7)
// implements compaction against the rich SessionState / CompactionResult API
// and was never reachable from the loop seam. This file is the production
// bridge: StandardCompactionAdapter implements sporecore.ContextManager AND the
// optional sporecore.CompactingContextManager, so a harness configured with a
// rich manager actually compacts out of the box.
//
// Design decisions (resolved by #55, NOT relitigated here):
//
//  1. STATELESS bridge — the adapter holds no session state. The rich
//     SessionState is serialized into sporecore.SessionState.Extras under
//     RichStateKey on every mutating seam call and re-read on every read. No
//     struct field carries session state.
//  2. Compaction never halts the loop — ApplyCompaction swallows (logs) any
//     rich Err, and a malformed/absent rich-state blob degrades to a safe
//     default (no compaction) rather than panicking.
//  3. The summary is wrapped as a Role==Assistant message for the rich
//     CompactionResult so the rich manager prepends it as the summary turn.

package contextmgr

import (
	"context"
	"encoding/json"
	"log"
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// RichStateKey is the reserved key under sporecore.SessionState.Extras holding
// the serialized rich SessionState. The adapter is the only writer/reader.
const RichStateKey = "spore.compaction_adapter.rich_state"

// EstimateMessageTokens is a rough token estimate for a single message: the
// byte length of its textual content divided by four (the same chars/4 proxy
// StandardContextManager uses for cache-marker placement). Used by the adapter
// to compute real TokensReclaimed from the messages a compaction drops, since
// the synchronous harness seam cannot call the async CountTokens. A non-empty
// message is never accounted as zero tokens so a drop always reclaims something.
func EstimateMessageTokens(m sporecore.Message) uint32 {
	var n int
	switch m.Content.Type {
	case sporecore.ContentTypeText:
		n = len(m.Content.Text)
	case sporecore.ContentTypeToolCall:
		if m.Content.ToolCall != nil {
			n = len(m.Content.ToolCall.Name) + len(m.Content.ToolCall.Input)
		}
	case sporecore.ContentTypeToolResult:
		if m.Content.ToolResult != nil {
			n = len(m.Content.ToolResult.Content)
		}
	case sporecore.ContentTypeImage:
		n = len(m.Content.Data)
	}
	t := uint32(n / 4)
	if n > 0 && t == 0 {
		return 1
	}
	return t
}

// EstimateTokens sums EstimateMessageTokens over a slice of messages.
func EstimateTokens(msgs []sporecore.Message) uint32 {
	var sum uint32
	for _, m := range msgs {
		sum += EstimateMessageTokens(m)
	}
	return sum
}

// summarizeInstruction is appended after the messages-to-compact to elicit the
// summary from the agent during a compaction turn.
const summarizeInstruction = "Summarize the conversation above, preserving the items in the preservation hints."

// StandardCompactionAdapter is a stateless bridge from the rich
// StandardContextManager onto the harness-loop compaction seam.
//
// Construct via NewStandardCompactionAdapter, then inject the result as the
// HarnessConfig.ContextManager. It implements sporecore.ContextManager (the
// per-turn seam) and sporecore.CompactingContextManager (the optional
// compaction seam the harness type-asserts).
type StandardCompactionAdapter struct {
	inner *StandardContextManager
}

// NewStandardCompactionAdapter wraps a rich StandardContextManager as a
// harness-seam context manager.
func NewStandardCompactionAdapter(inner *StandardContextManager) *StandardCompactionAdapter {
	return &StandardCompactionAdapter{inner: inner}
}

// SeedRichState projects a rich SessionState into a harness SessionState:
// it copies the message history onto Messages and serializes the rich state
// into Extras under RichStateKey. Callers driving the harness with this adapter
// use it to seed the session before the first turn.
func SeedRichState(session *sporecore.SessionState, rich *SessionState) {
	writeRichState(session, rich)
}

// readRichState reconstructs the rich SessionState from Extras. Returns nil
// when no rich state has been seeded yet or the blob is malformed — callers
// treat that as "nothing to compact" so the loop is never blocked.
func readRichState(session *sporecore.SessionState) *SessionState {
	if session == nil || session.Extras == nil {
		return nil
	}
	raw, ok := session.Extras[RichStateKey]
	if !ok {
		return nil
	}
	// Extras is map[string]any; the blob may be the original *SessionState
	// (same-process seeding) or a decoded JSON value (post pause/resume).
	switch v := raw.(type) {
	case *SessionState:
		return v
	case SessionState:
		s := v
		return &s
	default:
		data, err := json.Marshal(raw)
		if err != nil {
			return nil
		}
		var s SessionState
		if err := json.Unmarshal(data, &s); err != nil {
			return nil
		}
		return &s
	}
}

// writeRichState serializes the rich SessionState back into Extras and projects
// its MessageHistory onto the harness-side Messages.
func writeRichState(session *sporecore.SessionState, rich *SessionState) {
	if session == nil || rich == nil {
		return
	}
	session.Messages = append([]sporecore.Message(nil), rich.MessageHistory...)
	if session.Extras == nil {
		session.Extras = make(map[string]any)
	}
	// Round-trip through JSON so the stored blob is plain data (decodable after
	// pause/resume), matching the Rust adapter's serde_json::to_value spirit.
	data, err := json.Marshal(rich)
	if err != nil {
		return
	}
	var decoded any
	if err := json.Unmarshal(data, &decoded); err != nil {
		return
	}
	session.Extras[RichStateKey] = decoded
}

// ============================================================================
// sporecore.ContextManager (per-turn seam) — minimal, NOT load-bearing for
// compaction. Mirrors the loop's test-double managers.
// ============================================================================

// Assemble produces a minimal Context straight from the session messages. The
// rich Assemble requires ContextSources the seam does not supply, so this is a
// pass-through (mirrors the Rust adapter + the loop's test doubles).
func (a *StandardCompactionAdapter) Assemble(_ context.Context, session *sporecore.SessionState, _ *sporecore.Task) sporecore.Context {
	return sporecore.Context{
		Messages: append([]sporecore.Message(nil), session.Messages...),
	}
}

// AppendToolResult appends a tool message to the session, keeping the rich
// state's message history in sync.
func (a *StandardCompactionAdapter) AppendToolResult(_ context.Context, session *sporecore.SessionState, result *sporecore.HarnessToolResult) {
	var text string
	switch result.Output.Kind {
	case sporecore.ToolOutputSuccess:
		text = result.Output.Content
	case sporecore.ToolOutputError:
		text = result.Output.Message
	}
	msg := sporecore.Message{Role: sporecore.RoleTool, Content: sporecore.NewTextContent(text)}
	session.Messages = append(session.Messages, msg)
	a.syncMessagesIntoRichState(session)
}

// AppendUserMessage appends a user message to the session, keeping the rich
// state's message history in sync.
func (a *StandardCompactionAdapter) AppendUserMessage(_ context.Context, session *sporecore.SessionState, text string) {
	msg := sporecore.Message{Role: sporecore.RoleUser, Content: sporecore.NewTextContent(text)}
	session.Messages = append(session.Messages, msg)
	a.syncMessagesIntoRichState(session)
}

// syncMessagesIntoRichState mirrors the harness Messages back into the rich
// state blob so the next ShouldCompact / PrepareCompactionTurn sees the growth.
// When no rich state has been seeded the append is a plain Messages mutation.
func (a *StandardCompactionAdapter) syncMessagesIntoRichState(session *sporecore.SessionState) {
	rich := readRichState(session)
	if rich == nil {
		return
	}
	rich.MessageHistory = append([]sporecore.Message(nil), session.Messages...)
	writeRichState(session, rich)
}

// ShouldCompact reconstructs the rich state and delegates to the rich
// ShouldCompact. Absent/malformed rich state degrades to false.
func (a *StandardCompactionAdapter) ShouldCompact(session *sporecore.SessionState) bool {
	rich := readRichState(session)
	if rich == nil {
		return false
	}
	return a.inner.ShouldCompact(rich)
}

// ============================================================================
// sporecore.CompactingContextManager (optional compaction seam)
// ============================================================================

// PrepareCompactionTurn reconstructs the rich state, runs the rich
// PrepareCompaction, and projects the preserve hints + verification state +
// removed-count into a CompactionTurn. Returns (nil, false) when there is
// nothing to compact (no rich state, prepare error, or empty slice).
func (a *StandardCompactionAdapter) PrepareCompactionTurn(session *sporecore.SessionState) (*sporecore.CompactionTurn, bool) {
	rich := readRichState(session)
	if rich == nil {
		return nil, false
	}
	request, err := a.inner.PrepareCompaction(rich)
	if err != nil || request == nil || len(request.MessagesToCompact) == 0 {
		return nil, false
	}

	// Build the summarization context: the messages to compact followed by the
	// summarization instruction. InjectMissingItems appends the retry
	// instruction on a verification failure.
	messages := make([]sporecore.Message, 0, len(request.MessagesToCompact)+1)
	messages = append(messages, request.MessagesToCompact...)
	messages = append(messages, sporecore.Message{
		Role:    sporecore.RoleUser,
		Content: sporecore.NewTextContent(summarizeInstruction),
	})

	return &sporecore.CompactionTurn{
		Context:           sporecore.Context{Messages: messages},
		PreserveHints:     request.PreserveHints,
		VerificationState: rich,
		MessagesRemoved:   uint32(len(request.MessagesToCompact)),
	}, true
}

// InjectMissingItems appends the standard retry instruction to the compaction
// Context, requesting a revised summary. The exact wording matches the
// compaction_loop fixture: "Your summary is missing these items: {missing}.
// Please revise."
func (a *StandardCompactionAdapter) InjectMissingItems(c *sporecore.Context, missing []string) {
	text := "Your summary is missing these items: " + strings.Join(missing, ", ") + ". Please revise."
	c.Messages = append(c.Messages, sporecore.Message{
		Role:    sporecore.RoleUser,
		Content: sporecore.NewTextContent(text),
	})
}

// ApplyCompaction reconstructs the rich state, builds a CompactionResult
// (summary as an Assistant message), delegates to the rich ApplyCompaction,
// logs+swallows any error (the loop must never halt), and writes the mutated
// rich state back into the session. A malformed/absent rich-state blob is a
// safe no-op.
func (a *StandardCompactionAdapter) ApplyCompaction(session *sporecore.SessionState, summary string) {
	rich := readRichState(session)
	if rich == nil {
		// No rich state to apply against — degrade safely; never panic.
		return
	}

	// Recompute the dropped messages from a fresh prepare so token accounting
	// reflects exactly what ApplyCompaction will remove.
	var messagesRemoved uint32
	var dropped []sporecore.Message
	if req, err := a.inner.PrepareCompaction(rich); err == nil && req != nil {
		dropped = req.MessagesToCompact
		messagesRemoved = uint32(len(dropped))
	}

	summaryMessage := sporecore.Message{Role: sporecore.RoleAssistant, Content: sporecore.NewTextContent(summary)}

	// Real token accounting (issue #57 / Known Deviation #2 fix): reclaim the
	// tokens of the messages we drop, net of the summary that replaces them, and
	// clamp to the live budget so TokenBudgetUsed never underflows. The rich
	// ApplyCompaction decrements TokenBudgetUsed by this amount, so utilization
	// actually falls below threshold after a compaction and a long session can
	// compact, continue, drop below threshold, and compact again.
	droppedTokens := EstimateTokens(dropped)
	summaryTokens := EstimateMessageTokens(summaryMessage)
	var netReclaimed uint32
	if droppedTokens > summaryTokens {
		netReclaimed = droppedTokens - summaryTokens
	}
	tokensReclaimed := netReclaimed
	if tokensReclaimed > rich.TokenBudgetUsed {
		tokensReclaimed = rich.TokenBudgetUsed
	}

	result := CompactionResult{
		SummaryMessage:  summaryMessage,
		TokensReclaimed: tokensReclaimed,
		MessagesRemoved: messagesRemoved,
	}

	if err := a.inner.ApplyCompaction(rich, result); err != nil {
		// Compaction must never halt the loop — log and swallow, leaving the
		// session unchanged.
		log.Printf("spore.compaction: rich ApplyCompaction failed, leaving session unchanged: %v", err)
		return
	}
	writeRichState(session, rich)
}

// TokenBudgetUsed reports the rich state's post-compaction token budget so the
// harness can stamp the real TokensAfter / TokensReclaimed on the Compaction
// span (issue #57). Returns (0, false) when no rich state has been seeded.
func (a *StandardCompactionAdapter) TokenBudgetUsed(session *sporecore.SessionState) (uint32, bool) {
	rich := readRichState(session)
	if rich == nil {
		return 0, false
	}
	return rich.TokenBudgetUsed, true
}

// ============================================================================
// Convenience constructor
// ============================================================================

// NewCompactingHarnessConfig builds a sporecore.HarnessConfig wired with the
// StandardCompactionAdapter (so the harness actually compacts) and the standard
// KeyTermVerifier. Callers supply the rest of the components; optional fields
// (Middleware, Observability) are left to the caller. This is the convenience
// path so a harness can be built with a StandardContextManager that compacts.
func NewCompactingHarnessConfig(
	inner *StandardContextManager,
	agent sporecore.Agent,
	tools sporecore.ToolRegistry,
	sandbox sporecore.SandboxProvider,
	termination sporecore.TerminationPolicy,
) sporecore.HarnessConfig {
	return sporecore.HarnessConfig{
		Agent:                 agent,
		ToolRegistry:          tools,
		Sandbox:               sandbox,
		ContextManager:        NewStandardCompactionAdapter(inner),
		TerminationPolicy:     termination,
		CompactionVerifier:    NewKeyTermVerifier(),
		MaxCompactionAttempts: inner.compaction.MaxCompactionAttempts,
	}
}

// Compile-time interface checks.
var (
	_ sporecore.ContextManager           = (*StandardCompactionAdapter)(nil)
	_ sporecore.CompactingContextManager = (*StandardCompactionAdapter)(nil)
)
