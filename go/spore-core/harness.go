// Harness — the agent runtime loop.
//
// Implements issue #3. The harness owns execution lifecycle and wires all
// components together. It is stateless between Run() calls; everything the
// harness needs comes in via HarnessRunOptions or PausedState, and
// everything it produces goes out via RunResult.
//
// What this component does:
//   - Assemble context (via ContextManager) before each turn
//   - Call the agent for one turn
//   - Dispatch tool calls to ToolRegistry
//   - Evaluate TerminationPolicy after each turn
//   - Fire middleware lifecycle hooks
//   - Track iterations, token spend, elapsed time
//   - Pause and resume for human-in-the-loop interactions
//
// What this component does NOT do:
//   - Touch the filesystem, execute commands, or call the model directly
//   - Persist PausedState — the caller owns persistence
//   - Implement individual tools, sandbox policy, or context assembly
//
// Rules enforced here:
//
//  1. Harness owns the loop — the agent only executes one turn at a time.
//  2. Termination is evaluated against external state via TerminationPolicy.
//  3. Any budget overrun terminates the loop with an explicit HaltReason.
//  4. A turn that yields neither a tool call nor a final response is an error
//     (surfaced via AgentError, routed through error-propagation rules).
//  5. All components are injected at construction; the harness never builds
//     them itself.
//  6. Stateless between pause and resume — the caller owns PausedState.
//  7. WaitingForHuman returns immediately; no internal timeout.
//  8. approved_results prevents double-execution on resume.
//  9. Subagents cannot spawn their own subagents — ChildPausedState has no
//     child_state field (compile-time depth-1 enforcement).
//
// Component dependencies (forward declarations):
//
// Many of the trait dependencies of the harness (ToolRegistry,
// SandboxProvider, ContextManager, ...) ship in their own component issues
// (#4–#13). Until those land, this file defines minimal forward
// declarations of the interface surface the loop actually consumes. When a
// sibling issue lands its canonical definition will replace the stub here.
//
// Cross-language note: the shape of Task, BudgetLimits, RunResult,
// HaltReason, PausedState, ChildPausedState, HumanRequest, and
// HumanResponse mirrors byte-for-byte across Rust, TypeScript, and Python.

package sporecore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ============================================================================
// Identity newtypes
// ============================================================================

// SessionID is an opaque session identifier.
type SessionID string

// TaskID is an opaque task identifier.
type TaskID string

var idCounter uint64

func randomID() string {
	n := atomic.AddUint64(&idCounter, 1)
	return fmt.Sprintf("%016x", n)
}

// NewSessionID generates a fresh, opaque session id.
func NewSessionID() SessionID { return SessionID("sess-" + randomID()) }

// NewTaskID generates a fresh, opaque task id.
func NewTaskID() TaskID { return TaskID("task-" + randomID()) }

// ============================================================================
// Budget tracking
// ============================================================================

// BudgetLimits caps loop resource consumption. Nil pointer fields mean
// "no limit" for that resource (matches Rust Option<>).
type BudgetLimits struct {
	MaxTurns        *uint32 `json:"max_turns"`
	MaxInputTokens  *uint32 `json:"max_input_tokens"`
	MaxOutputTokens *uint32 `json:"max_output_tokens"`
	// MaxWallTime is encoded as seconds (or null) to match Rust
	// duration_secs_opt.
	MaxWallTime *time.Duration `json:"-"`
	MaxCostUSD  *float64       `json:"max_cost_usd"`
}

// MarshalJSON encodes MaxWallTime as a uint64 second count (or null).
func (b BudgetLimits) MarshalJSON() ([]byte, error) {
	var wall *uint64
	if b.MaxWallTime != nil {
		s := uint64(b.MaxWallTime.Seconds())
		wall = &s
	}
	return json.Marshal(struct {
		MaxTurns        *uint32  `json:"max_turns"`
		MaxInputTokens  *uint32  `json:"max_input_tokens"`
		MaxOutputTokens *uint32  `json:"max_output_tokens"`
		MaxWallTime     *uint64  `json:"max_wall_time"`
		MaxCostUSD      *float64 `json:"max_cost_usd"`
	}{b.MaxTurns, b.MaxInputTokens, b.MaxOutputTokens, wall, b.MaxCostUSD})
}

// UnmarshalJSON decodes the seconds-encoded MaxWallTime.
func (b *BudgetLimits) UnmarshalJSON(data []byte) error {
	var probe struct {
		MaxTurns        *uint32  `json:"max_turns"`
		MaxInputTokens  *uint32  `json:"max_input_tokens"`
		MaxOutputTokens *uint32  `json:"max_output_tokens"`
		MaxWallTime     *uint64  `json:"max_wall_time"`
		MaxCostUSD      *float64 `json:"max_cost_usd"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	b.MaxTurns = probe.MaxTurns
	b.MaxInputTokens = probe.MaxInputTokens
	b.MaxOutputTokens = probe.MaxOutputTokens
	b.MaxCostUSD = probe.MaxCostUSD
	if probe.MaxWallTime != nil {
		d := time.Duration(*probe.MaxWallTime) * time.Second
		b.MaxWallTime = &d
	}
	return nil
}

// BudgetLimitType discriminates which budget triggered a halt.
type BudgetLimitType string

const (
	BudgetLimitTurns        BudgetLimitType = "turns"
	BudgetLimitInputTokens  BudgetLimitType = "input_tokens"
	BudgetLimitOutputTokens BudgetLimitType = "output_tokens"
	BudgetLimitWallTime     BudgetLimitType = "wall_time"
	BudgetLimitCostUSD      BudgetLimitType = "cost_usd"
)

// BudgetSnapshot records resource consumption so far.
type BudgetSnapshot struct {
	Turns        uint32         `json:"turns"`
	InputTokens  uint64         `json:"input_tokens"`
	OutputTokens uint64         `json:"output_tokens"`
	WallTime     *time.Duration `json:"-"`
	CostUSD      float64        `json:"cost_usd"`
}

// MarshalJSON encodes WallTime as seconds.
func (b BudgetSnapshot) MarshalJSON() ([]byte, error) {
	var wall *uint64
	if b.WallTime != nil {
		s := uint64(b.WallTime.Seconds())
		wall = &s
	}
	return json.Marshal(struct {
		Turns        uint32  `json:"turns"`
		InputTokens  uint64  `json:"input_tokens"`
		OutputTokens uint64  `json:"output_tokens"`
		WallTime     *uint64 `json:"wall_time"`
		CostUSD      float64 `json:"cost_usd"`
	}{b.Turns, b.InputTokens, b.OutputTokens, wall, b.CostUSD})
}

// UnmarshalJSON decodes the seconds-encoded WallTime.
func (b *BudgetSnapshot) UnmarshalJSON(data []byte) error {
	var probe struct {
		Turns        uint32  `json:"turns"`
		InputTokens  uint64  `json:"input_tokens"`
		OutputTokens uint64  `json:"output_tokens"`
		WallTime     *uint64 `json:"wall_time"`
		CostUSD      float64 `json:"cost_usd"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	b.Turns = probe.Turns
	b.InputTokens = probe.InputTokens
	b.OutputTokens = probe.OutputTokens
	b.CostUSD = probe.CostUSD
	if probe.WallTime != nil {
		d := time.Duration(*probe.WallTime) * time.Second
		b.WallTime = &d
	}
	return nil
}

// AggregateUsage is the token-and-cost totals reported on every RunResult.
type AggregateUsage struct {
	InputTokens      uint64  `json:"input_tokens"`
	OutputTokens     uint64  `json:"output_tokens"`
	CacheReadTokens  uint64  `json:"cache_read_tokens"`
	CacheWriteTokens uint64  `json:"cache_write_tokens"`
	CostUSD          float64 `json:"cost_usd"`
}

// AddTurn folds one TokenUsage into the running totals.
func (a *AggregateUsage) AddTurn(u TokenUsage) {
	a.InputTokens += uint64(u.InputTokens)
	a.OutputTokens += uint64(u.OutputTokens)
	if u.CacheReadTokens != nil {
		a.CacheReadTokens += uint64(*u.CacheReadTokens)
	}
	if u.CacheWriteTokens != nil {
		a.CacheWriteTokens += uint64(*u.CacheWriteTokens)
	}
}

// ============================================================================
// Task and loop strategy
// ============================================================================

// OptimizationDirection chooses minimise vs maximise for HillClimbing.
type OptimizationDirection string

const (
	OptimizationMinimize OptimizationDirection = "minimize"
	OptimizationMaximize OptimizationDirection = "maximize"
)

// ModelConfig is a lightweight placeholder for an alternate planner model.
type ModelConfig struct {
	Provider string `json:"provider"`
	ModelID  string `json:"model_id"`
}

// LoopStrategy, LoopStrategyKind, the strategy config newtypes, StrategyRef,
// and the RunStrategy seam live in strategy.go (issue #119).

// Task is the input to a harness run.
type Task struct {
	ID           TaskID       `json:"id"`
	Instruction  string       `json:"instruction"`
	SessionID    SessionID    `json:"session_id"`
	Budget       BudgetLimits `json:"budget"`
	LoopStrategy LoopStrategy `json:"loop_strategy"`
}

// NewTask constructs a Task with a fresh ID and default budget.
func NewTask(instruction string, sessionID SessionID, strategy LoopStrategy) Task {
	return Task{
		ID:           NewTaskID(),
		Instruction:  instruction,
		SessionID:    sessionID,
		Budget:       BudgetLimits{},
		LoopStrategy: strategy,
	}
}

// WithBudget returns a copy of the task with the supplied budget.
func (t Task) WithBudget(b BudgetLimits) Task {
	t.Budget = b
	return t
}

// ============================================================================
// Streaming
// ============================================================================

// BlockKind is the kind of content block a BlockStart stream event opens
// (issue #103, resolved spec decision Q2: a single generic frame marker
// carrying a BlockKind rather than typed-per-kind markers). Wire values are
// snake_case to match the cross-language fixtures.
type BlockKind string

const (
	// BlockText is a text content block (model content_block_delta).
	BlockText BlockKind = "text"
	// BlockReasoning is a reasoning/thinking block (model thinking_delta).
	BlockReasoning BlockKind = "reasoning"
	// BlockToolUse is a tool-use block whose arguments arrive as model
	// tool_use_delta fragments.
	BlockToolUse BlockKind = "tool_use"
)

// HarnessStreamEventKind discriminates harness-level stream events.
type HarnessStreamEventKind string

const (
	HarnessStreamTurnStart     HarnessStreamEventKind = "turn_start"
	HarnessStreamTurnEnd       HarnessStreamEventKind = "turn_end"
	HarnessStreamToolCall      HarnessStreamEventKind = "tool_call"
	HarnessStreamToolResult    HarnessStreamEventKind = "tool_result"
	HarnessStreamFinalResponse HarnessStreamEventKind = "final_response"
	HarnessStreamBudgetWarning HarnessStreamEventKind = "budget_warning"
	// HarnessStreamUserMessage — the send_message tool (#81) surfaces an
	// out-of-band, prominent message to the user. The loop emits this instead
	// of collapsing the content into a normal tool result.
	HarnessStreamUserMessage HarnessStreamEventKind = "user_message"

	// ── Delta-level streaming (issue #103) ──────────────────────────────────

	// HarnessStreamTextDelta is a streamed text fragment (model
	// content_block_delta).
	HarnessStreamTextDelta HarnessStreamEventKind = "text_delta"
	// HarnessStreamReasoningDelta is a streamed reasoning/thinking fragment
	// (model thinking_delta). Q4.
	HarnessStreamReasoningDelta HarnessStreamEventKind = "reasoning_delta"
	// HarnessStreamToolArgsDelta is a streamed tool-argument JSON fragment
	// (model tool_use_delta), correlated to a call_id via the open-block index.
	HarnessStreamToolArgsDelta HarnessStreamEventKind = "tool_args_delta"
	// HarnessStreamBlockStart marks a content block opening (Q2). Emitted the
	// first time a delta for an index is seen.
	HarnessStreamBlockStart HarnessStreamEventKind = "block_start"
	// HarnessStreamBlockStop marks a content block closing (model
	// content_block_stop). Q2.
	HarnessStreamBlockStop HarnessStreamEventKind = "block_stop"
	// HarnessStreamToolCallStart marks a tool-use block opening so consumers can
	// correlate the subsequent ToolArgsDelta fragments and the final coarse
	// ToolCall by call_id. The name may be empty when the underlying model
	// stream does not surface it before args (a documented model StreamEvent
	// limitation — name is recovered on the coarse ToolCall).
	HarnessStreamToolCallStart HarnessStreamEventKind = "tool_call_start"

	// ── Consecutive-recoverable-tool-error breaker (issue #137) ─────────────────

	// HarnessStreamToolErrorLoopDetected: the breaker DETECTED a loop — a tool hit
	// ErrorLoopThreshold (N) consecutive identical-argument recoverable errors and
	// ONE corrective message was injected. A warning — the loop continues. Carries
	// the Tool name and ConsecutiveErrors (= N).
	HarnessStreamToolErrorLoopDetected HarnessStreamEventKind = "tool_error_loop_detected"
	// HarnessStreamToolErrorLoopBroken: the breaker TRIPPED — the same tool reached
	// 2*ErrorLoopThreshold (2*N) consecutive identical-argument errors and the loop
	// STOPPED to resolve the node's BudgetExhaustedBehavior. Carries the Tool name
	// and ConsecutiveErrors (= 2*N).
	HarnessStreamToolErrorLoopBroken HarnessStreamEventKind = "tool_error_loop_broken"

	// ── Output-schema enforcement (issue #139) ──────────────────────────────────

	// HarnessStreamOutputSchemaRetry: output-schema enforcement fed a validation
	// error back and RETRIED — the terminal FinalResponse failed validation against
	// the leaf's output schema and a retry turn was granted (within budget). A
	// warning — the loop continues. Carries the extra-retry count so far
	// (= Attempt) and the frozen validator error fed back (= Content).
	HarnessStreamOutputSchemaRetry HarnessStreamEventKind = "output_schema_retry"
	// HarnessStreamOutputSchemaViolation: output-schema enforcement EXHAUSTED its
	// retries — the terminal still failed validation after OutputSchemaMaxRetries
	// extra turns (with budget remaining), so the run terminates with
	// HaltOutputSchemaViolation. Carries the total attempt count (= Attempts =
	// 1 + max_retries) and the final frozen validator error (= Content).
	HarnessStreamOutputSchemaViolation HarnessStreamEventKind = "output_schema_violation"
)

// HarnessStreamEvent is one event emitted while the loop runs.
//
// ## Delta-level streaming (issue #103)
//
// The harness maps each raw model StreamEvent produced by the agent through
// mapModelStreamEvent into zero or more of the delta/frame variants below,
// alongside the existing coarse lifecycle events. Resolution notes:
//
//   - Q2: frame markers are the generic block_start / block_stop carrying a
//     BlockKind, NOT typed-per-kind markers.
//   - Q3: model message_start / message_stop are DROPPED at the harness
//     boundary (mapped to nothing). turn_start / turn_end already cover message
//     lifecycle.
//   - Q5: the coarse tool_call now also carries the final Args, and tool_result
//     the result Content. Both new fields serialise with defaults so pre-#103
//     serialized events round-trip.
//
// Tool lifecycle ordering per call: tool_call_start → tool_args_delta* →
// (block_stop) → coarse tool_call.
type HarnessStreamEvent struct {
	Kind HarnessStreamEventKind `json:"kind"`
	// turn_start, turn_end
	Turn uint32 `json:"-"`
	// tool_call, tool_result, tool_args_delta, tool_call_start
	CallID string `json:"-"`
	// tool_call, tool_call_start
	Name string `json:"-"`
	// tool_result
	IsError bool `json:"-"`
	// final_response, user_message, text_delta, reasoning_delta
	Content string `json:"-"`
	// budget_warning
	LimitType BudgetLimitType `json:"-"`
	// tool_call (Q5): final, fully-accumulated tool-call arguments (parsed JSON
	// value carried as RawMessage). Defaults to "null" on the wire.
	Args json.RawMessage `json:"-"`
	// tool_result (Q5): the tool result content. Defaults to "" on the wire.
	ResultContent string `json:"-"`
	// tool_args_delta
	PartialJSON string `json:"-"`
	// block_start, block_stop, tool_call_start
	Index uint32 `json:"-"`
	// block_start
	Block BlockKind `json:"-"`
	// tool_error_loop_detected, tool_error_loop_broken (issue #137)
	Tool              string `json:"-"`
	ConsecutiveErrors uint32 `json:"-"`
	// output_schema_retry (issue #139): the extra-retry count so far. The frozen
	// validator error fed back rides the Content field above.
	Attempt uint32 `json:"-"`
	// output_schema_violation (issue #139): the total attempt count (1 + N). The
	// final frozen validator error rides the Content field above.
	Attempts uint32 `json:"-"`
}

// MarshalJSON serialises as a flat tagged object.
func (e HarnessStreamEvent) MarshalJSON() ([]byte, error) {
	switch e.Kind {
	case HarnessStreamTurnStart, HarnessStreamTurnEnd:
		return json.Marshal(struct {
			Kind HarnessStreamEventKind `json:"kind"`
			Turn uint32                 `json:"turn"`
		}{e.Kind, e.Turn})
	case HarnessStreamToolCall:
		// Q5: carry the final args. Default to JSON null so pre-#103 consumers
		// (and fixtures that omit it) round-trip.
		args := e.Args
		if len(args) == 0 {
			args = json.RawMessage("null")
		}
		return json.Marshal(struct {
			Kind   HarnessStreamEventKind `json:"kind"`
			CallID string                 `json:"call_id"`
			Name   string                 `json:"name"`
			Args   json.RawMessage        `json:"args"`
		}{e.Kind, e.CallID, e.Name, args})
	case HarnessStreamToolResult:
		return json.Marshal(struct {
			Kind    HarnessStreamEventKind `json:"kind"`
			CallID  string                 `json:"call_id"`
			IsError bool                   `json:"is_error"`
			Content string                 `json:"content"`
		}{e.Kind, e.CallID, e.IsError, e.ResultContent})
	case HarnessStreamFinalResponse, HarnessStreamUserMessage:
		return json.Marshal(struct {
			Kind    HarnessStreamEventKind `json:"kind"`
			Content string                 `json:"content"`
		}{e.Kind, e.Content})
	case HarnessStreamBudgetWarning:
		return json.Marshal(struct {
			Kind      HarnessStreamEventKind `json:"kind"`
			LimitType BudgetLimitType        `json:"limit_type"`
		}{e.Kind, e.LimitType})
	case HarnessStreamTextDelta, HarnessStreamReasoningDelta:
		return json.Marshal(struct {
			Kind    HarnessStreamEventKind `json:"kind"`
			Content string                 `json:"content"`
		}{e.Kind, e.Content})
	case HarnessStreamToolArgsDelta:
		return json.Marshal(struct {
			Kind        HarnessStreamEventKind `json:"kind"`
			CallID      string                 `json:"call_id"`
			PartialJSON string                 `json:"partial_json"`
		}{e.Kind, e.CallID, e.PartialJSON})
	case HarnessStreamBlockStart:
		return json.Marshal(struct {
			Kind  HarnessStreamEventKind `json:"kind"`
			Index uint32                 `json:"index"`
			Block BlockKind              `json:"block"`
		}{e.Kind, e.Index, e.Block})
	case HarnessStreamBlockStop:
		return json.Marshal(struct {
			Kind  HarnessStreamEventKind `json:"kind"`
			Index uint32                 `json:"index"`
		}{e.Kind, e.Index})
	case HarnessStreamToolCallStart:
		return json.Marshal(struct {
			Kind   HarnessStreamEventKind `json:"kind"`
			Index  uint32                 `json:"index"`
			CallID string                 `json:"call_id"`
			Name   string                 `json:"name"`
		}{e.Kind, e.Index, e.CallID, e.Name})
	case HarnessStreamToolErrorLoopDetected, HarnessStreamToolErrorLoopBroken:
		return json.Marshal(struct {
			Kind              HarnessStreamEventKind `json:"kind"`
			Tool              string                 `json:"tool"`
			ConsecutiveErrors uint32                 `json:"consecutive_errors"`
		}{e.Kind, e.Tool, e.ConsecutiveErrors})
	case HarnessStreamOutputSchemaRetry:
		return json.Marshal(struct {
			Kind    HarnessStreamEventKind `json:"kind"`
			Attempt uint32                 `json:"attempt"`
			Error   string                 `json:"error"`
		}{e.Kind, e.Attempt, e.Content})
	case HarnessStreamOutputSchemaViolation:
		return json.Marshal(struct {
			Kind     HarnessStreamEventKind `json:"kind"`
			Attempts uint32                 `json:"attempts"`
			Error    string                 `json:"error"`
		}{e.Kind, e.Attempts, e.Content})
	default:
		return nil, fmt.Errorf("HarnessStreamEvent: unknown kind %q", e.Kind)
	}
}

// StreamSink consumes harness stream events. May be nil.
type StreamSink func(HarnessStreamEvent)

// turnStreamState is the per-turn mutable state threaded through
// mapModelStreamEvent (issue #103). It correlates a block index to its
// BlockKind so tool_use_delta / content_block_stop events can be mapped back to
// a call_id, and so each block's block_start is emitted exactly once.
//
// The model tool_use_start event carries the real call id + tool name at block
// start, which the harness threads onto tool_call_start. The synthesized
// "call_{index}" id + empty name remain only as a fallback when a stream omits
// the start frame and opens the block on a tool_use_delta instead. See agent.go.
type turnStreamState struct {
	openBlocks map[uint32]BlockKind
	toolCalls  map[uint32]string
}

func newTurnStreamState() *turnStreamState {
	return &turnStreamState{
		openBlocks: make(map[uint32]BlockKind),
		toolCalls:  make(map[uint32]string),
	}
}

func streamCallIDFor(index uint32) string {
	return fmt.Sprintf("call_%d", index)
}

// mapModelStreamEvent maps one raw model StreamEvent to zero or more harness
// HarnessStreamEvents (issue #103), threading turnStreamState so blocks and
// tool calls are correlated across events.
//
// Rules enforced here:
//   - Q2: a block's block_start is emitted exactly once, the first time a delta
//     for that index is observed; content_block_stop maps to block_stop.
//   - Q3: message_start / message_stop map to nothing (dropped).
//   - A tool-use block additionally emits tool_call_start on open, then each
//     fragment as tool_args_delta keyed by the derived call_id.
func mapModelStreamEvent(ev StreamEvent, state *turnStreamState) []HarnessStreamEvent {
	switch ev.Type {
	case StreamMessageStart, StreamMessageStop:
		// Q3: dropped at the harness boundary.
		return nil
	case StreamContentBlockDelta:
		var out []HarnessStreamEvent
		if _, open := state.openBlocks[ev.Index]; !open {
			state.openBlocks[ev.Index] = BlockText
			out = append(out, HarnessStreamEvent{
				Kind: HarnessStreamBlockStart, Index: ev.Index, Block: BlockText,
			})
		}
		out = append(out, HarnessStreamEvent{Kind: HarnessStreamTextDelta, Content: ev.Delta})
		return out
	case StreamThinkingDelta:
		var out []HarnessStreamEvent
		if _, open := state.openBlocks[ev.Index]; !open {
			state.openBlocks[ev.Index] = BlockReasoning
			out = append(out, HarnessStreamEvent{
				Kind: HarnessStreamBlockStart, Index: ev.Index, Block: BlockReasoning,
			})
		}
		out = append(out, HarnessStreamEvent{Kind: HarnessStreamReasoningDelta, Content: ev.Delta})
		return out
	case StreamToolUseStart:
		var out []HarnessStreamEvent
		if _, open := state.openBlocks[ev.Index]; !open {
			state.openBlocks[ev.Index] = BlockToolUse
			// Use the real call id from the model; consumers correlate
			// subsequent tool_args_delta by it.
			state.toolCalls[ev.Index] = ev.ID
			out = append(out, HarnessStreamEvent{
				Kind: HarnessStreamBlockStart, Index: ev.Index, Block: BlockToolUse,
			})
			out = append(out, HarnessStreamEvent{
				Kind: HarnessStreamToolCallStart, Index: ev.Index, CallID: ev.ID, Name: ev.Name,
			})
		}
		return out
	case StreamToolUseDelta:
		var out []HarnessStreamEvent
		if _, open := state.openBlocks[ev.Index]; !open {
			state.openBlocks[ev.Index] = BlockToolUse
			callID := streamCallIDFor(ev.Index)
			state.toolCalls[ev.Index] = callID
			out = append(out, HarnessStreamEvent{
				Kind: HarnessStreamBlockStart, Index: ev.Index, Block: BlockToolUse,
			})
			// Fallback: if a stream omitted tool_use_start, open the block here
			// with a synthesized id and empty name so args still surface.
			out = append(out, HarnessStreamEvent{
				Kind: HarnessStreamToolCallStart, Index: ev.Index, CallID: callID, Name: "",
			})
		}
		callID, ok := state.toolCalls[ev.Index]
		if !ok {
			callID = streamCallIDFor(ev.Index)
		}
		out = append(out, HarnessStreamEvent{
			Kind: HarnessStreamToolArgsDelta, CallID: callID, PartialJSON: ev.PartialJSON,
		})
		return out
	case StreamContentBlockStop:
		delete(state.openBlocks, ev.Index)
		return []HarnessStreamEvent{{Kind: HarnessStreamBlockStop, Index: ev.Index}}
	default:
		return nil
	}
}

// sendMessageToolName is the registered name of the catalogue's send_message
// tool (issue #81). The harness loop recognizes this name to emit a
// UserMessage stream event instead of collapsing the tool result. It is
// duplicated here (rather than imported from the tools package) because the
// tools package imports this one — importing it back would form a cycle. It
// must stay in sync with tools.SendMessageToolName.
const sendMessageToolName = "send_message"

// ============================================================================
// Forward-declared sibling types (full surfaces in their owning issues)
// ============================================================================

// Mode is the harness operating mode targeted by HarnessSignal.SwitchMode
// (issue #80). It mirrors the existing prompt-chunk-registry Mode enum: the
// wire format is the identical bare snake_case string, so a Mode value
// round-trips byte-for-byte across the four language implementations. It is
// defined here (not imported from promptchunkregistry) because that package
// imports this one — taking the dependency the other way would form a cycle.
type Mode string

const (
	ModeAlwaysAsk Mode = "always_ask"
	ModeAutoEdit  Mode = "auto_edit"
	ModePlan      Mode = "plan"
	ModeSafeAuto  Mode = "safe_auto"
	// ModeYolo (full autonomy, no approval gates) is a named safety footgun and
	// is gated behind the `dangerous` build tag (issue #34). It is defined in
	// dangerous.go and absent from the default build, so SwitchMode cannot
	// target it by name without the explicit dangerous opt-in. Wire tag "yolo".
)

// HarnessSignalKind discriminates HarnessSignal variants.
type HarnessSignalKind string

const (
	SignalEnterPlanMode HarnessSignalKind = "enter_plan_mode"
	SignalExitPlanMode  HarnessSignalKind = "exit_plan_mode"
	SignalSwitchMode    HarnessSignalKind = "switch_mode"
	SignalAbort         HarnessSignalKind = "abort"
)

// HarnessSignal is the set of signals a tool can escalate to the harness's
// parent via ToolOutput.Escalate (issue #80). The harness is a pure
// intermediary: it never acts on a signal itself. It terminates cleanly,
// surfaces the signal via RunResult.Escalate, and the caller (CLI, chat UI,
// REST API, parent harness) owns the orchestration. This mirrors the
// WaitingForHuman model — the harness does not resume itself either.
//
// Wire format: serde-tagged on "kind", snake_case, byte-identical across the
// four language implementations. Round-tripped by
// fixtures/harness/escalation_signals.json.
type HarnessSignal struct {
	Kind HarnessSignalKind `json:"kind"`
	// enter_plan_mode: context the agent has accumulated so far as a seed for
	// the planning harness.
	Context string `json:"-"`
	// exit_plan_mode: the produced plan artifact for human approval before the
	// execution harness is instantiated. This is the planning agent's terminal
	// signal.
	Plan *PlanArtifact `json:"-"`
	// switch_mode: the target mode. The caller instantiates the appropriate
	// harness for the new mode.
	Mode Mode `json:"-"`
	// abort: a graceful-abort reason surfaced to the user. Distinct from
	// HaltAgentError — this is an intentional, agent-initiated stop and
	// surfaces as RunResult.Escalate, NOT RunResult.Failure.
	Reason string `json:"-"`
}

// MarshalJSON serialises HarnessSignal as a flat "kind"-tagged object.
func (s HarnessSignal) MarshalJSON() ([]byte, error) {
	switch s.Kind {
	case SignalEnterPlanMode:
		return json.Marshal(struct {
			Kind    HarnessSignalKind `json:"kind"`
			Context string            `json:"context"`
		}{s.Kind, s.Context})
	case SignalExitPlanMode:
		return json.Marshal(struct {
			Kind HarnessSignalKind `json:"kind"`
			Plan *PlanArtifact     `json:"plan"`
		}{s.Kind, s.Plan})
	case SignalSwitchMode:
		return json.Marshal(struct {
			Kind HarnessSignalKind `json:"kind"`
			Mode Mode              `json:"mode"`
		}{s.Kind, s.Mode})
	case SignalAbort:
		return json.Marshal(struct {
			Kind   HarnessSignalKind `json:"kind"`
			Reason string            `json:"reason"`
		}{s.Kind, s.Reason})
	default:
		return nil, fmt.Errorf("HarnessSignal: unknown kind %q", s.Kind)
	}
}

// UnmarshalJSON decodes the flat "kind"-tagged form.
func (s *HarnessSignal) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind    HarnessSignalKind `json:"kind"`
		Context string            `json:"context"`
		Plan    *PlanArtifact     `json:"plan"`
		Mode    Mode              `json:"mode"`
		Reason  string            `json:"reason"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	s.Kind = probe.Kind
	s.Context = probe.Context
	s.Plan = probe.Plan
	s.Mode = probe.Mode
	s.Reason = probe.Reason
	return nil
}

// ToolOutputKind discriminates ToolOutput variants.
type ToolOutputKind string

const (
	ToolOutputSuccess         ToolOutputKind = "success"
	ToolOutputError           ToolOutputKind = "error"
	ToolOutputWaitingForHuman ToolOutputKind = "waiting_for_human"
	// ToolOutputEscalate — tool requests a structural state change from the
	// harness's parent (issue #80). The harness terminates cleanly and passes
	// the signal to the caller via RunResult.Escalate. The escalation is NOT
	// appended to message history.
	ToolOutputEscalate ToolOutputKind = "escalate"
	// ToolOutputAwaitingClarification — a tool (e.g. ask_user_question, #81 Q4b)
	// needs a human answer before it can produce a result. UNLIKE the subagent
	// WaitingForHuman path there is NO ChildPausedState: the loop pauses with a
	// HumanRequest.Clarification, preserving the clarifying call as the head of
	// PendingToolCalls, and on resume injects the answer as that call's result.
	ToolOutputAwaitingClarification ToolOutputKind = "awaiting_clarification"
	// ToolOutputConsult — mid-loop consult signal (issue #114). A worker-side
	// tool returns it with ChildState nil; the worker harness pauses and returns
	// RunResult.Consult with the consult call as the head of PendingToolCalls
	// (HumanRequest nil). At the subagent boundary SubagentTool populates
	// ChildState — but in the A1 mediation seam it consumes the signal itself
	// rather than bubbling it, so a parent orchestrator never observes this
	// variant on the happy path. Mirrors ToolOutputWaitingForHuman. ONE variant:
	// the optional ChildState distinguishes worker-side (nil) from the
	// subagent boundary (populated).
	ToolOutputConsult ToolOutputKind = "consult"
)

// ToolOutput is the result of dispatching one tool call. Full shape lives
// in issue #4 (ToolRegistry) / #5 (Tool). The variants below cover what
// the harness loop needs to route.
type ToolOutput struct {
	Kind ToolOutputKind `json:"kind"`
	// success
	Content string `json:"-"`
	// Truncated is true ONLY when the tool itself clipped its output to fit an
	// inline budget (large outputs routed through SandboxProvider.HandleLargeOutput
	// set this). Plain tool authors should leave it false — use NewToolOutputSuccess.
	Truncated bool `json:"-"`
	// error
	Message string `json:"-"`
	// Recoverable is true if the agent may sensibly retry or adapt: the loop
	// appends the error as a tool result and continues. False halts the run. Most
	// tool failures are recoverable — prefer NewToolOutputError; reach for
	// NewToolOutputFatal only when continuing is pointless.
	Recoverable bool `json:"-"`
	// waiting_for_human
	ChildState *ChildPausedState `json:"-"`
	Request    *HumanRequest     `json:"-"`
	// escalate (issue #80)
	Signal *HarnessSignal `json:"-"`
	// awaiting_clarification (issue #81, Q4b)
	Question string `json:"-"`
	// Options is nil for a free-form clarification.
	Options *[]string `json:"-"`
	// consult (issue #114). ConsultRequest is the worker's ask. ChildState
	// (above) is reused: nil for the worker-side signal, populated by
	// SubagentTool at the subagent boundary.
	ConsultRequest *ConsultRequest `json:"-"`
}

// MarshalJSON serialises ToolOutput as a flat tagged object.
func (o ToolOutput) MarshalJSON() ([]byte, error) {
	switch o.Kind {
	case ToolOutputSuccess:
		return json.Marshal(struct {
			Kind      ToolOutputKind `json:"kind"`
			Content   string         `json:"content"`
			Truncated bool           `json:"truncated,omitempty"`
		}{o.Kind, o.Content, o.Truncated})
	case ToolOutputError:
		return json.Marshal(struct {
			Kind        ToolOutputKind `json:"kind"`
			Message     string         `json:"message"`
			Recoverable bool           `json:"recoverable"`
		}{o.Kind, o.Message, o.Recoverable})
	case ToolOutputWaitingForHuman:
		return json.Marshal(struct {
			Kind       ToolOutputKind    `json:"kind"`
			ChildState *ChildPausedState `json:"child_state"`
			Request    *HumanRequest     `json:"request"`
		}{o.Kind, o.ChildState, o.Request})
	case ToolOutputEscalate:
		// Nested tagged union: signal is itself a "kind"-tagged HarnessSignal.
		return json.Marshal(struct {
			Kind   ToolOutputKind `json:"kind"`
			Signal *HarnessSignal `json:"signal"`
		}{o.Kind, o.Signal})
	case ToolOutputAwaitingClarification:
		// Options serialised as null (not omitted) when absent, matching the
		// Rust Option<Vec<String>> wire form and the escalation_tools fixture.
		var opts []string
		if o.Options != nil {
			opts = *o.Options
		}
		return json.Marshal(struct {
			Kind     ToolOutputKind `json:"kind"`
			Question string         `json:"question"`
			Options  []string       `json:"options"`
		}{o.Kind, o.Question, opts})
	case ToolOutputConsult:
		// ChildState is OMITTED (skip_serializing_if = "Option::is_none") for
		// the worker-side signal and POPULATED at the subagent boundary, so the
		// two fixture shapes round-trip byte-identically.
		if o.ChildState == nil {
			return json.Marshal(struct {
				Kind    ToolOutputKind  `json:"kind"`
				Request *ConsultRequest `json:"request"`
			}{o.Kind, o.ConsultRequest})
		}
		return json.Marshal(struct {
			Kind       ToolOutputKind    `json:"kind"`
			ChildState *ChildPausedState `json:"child_state"`
			Request    *ConsultRequest   `json:"request"`
		}{o.Kind, o.ChildState, o.ConsultRequest})
	default:
		return nil, fmt.Errorf("ToolOutput: unknown kind %q", o.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (o *ToolOutput) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind        ToolOutputKind    `json:"kind"`
		Content     string            `json:"content"`
		Truncated   bool              `json:"truncated"`
		Message     string            `json:"message"`
		Recoverable bool              `json:"recoverable"`
		ChildState  *ChildPausedState `json:"child_state"`
		Request     *HumanRequest     `json:"request"`
		Signal      *HarnessSignal    `json:"signal"`
		Question    string            `json:"question"`
		Options     *[]string         `json:"options"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	o.Kind = probe.Kind
	o.Content = probe.Content
	o.Truncated = probe.Truncated
	o.Message = probe.Message
	o.Recoverable = probe.Recoverable
	o.ChildState = probe.ChildState
	o.Request = probe.Request
	o.Signal = probe.Signal
	o.Question = probe.Question
	o.Options = probe.Options
	// consult (issue #114): the "request" key here is a ConsultRequest, not a
	// HumanRequest. Re-read it into the dedicated field for that variant so the
	// two "request"-keyed variants do not collide on one probe field.
	if probe.Kind == ToolOutputConsult {
		var cr struct {
			Request *ConsultRequest `json:"request"`
		}
		if err := json.Unmarshal(data, &cr); err != nil {
			return err
		}
		o.ConsultRequest = cr.Request
		o.Request = nil
	}
	return nil
}

// NewToolOutputSuccess returns a successful, non-truncated result. The common
// case for a tool that returns its full output — saves spelling out the Kind +
// Truncated fields. Mirrors Rust's ToolOutput::success.
func NewToolOutputSuccess(content string) ToolOutput {
	return ToolOutput{Kind: ToolOutputSuccess, Content: content, Truncated: false}
}

// NewToolOutputConsult returns a WORKER-SIDE consult signal (issue #114): the
// tool asks for mid-loop help. ChildState is nil — the harness loop builds the
// RunResult.Consult pause; only SubagentTool populates ChildState at the
// subagent boundary. Mirrors Rust's ToolOutput::consult.
func NewToolOutputConsult(request ConsultRequest) ToolOutput {
	r := request
	return ToolOutput{Kind: ToolOutputConsult, ConsultRequest: &r}
}

// NewToolOutputError returns a RECOVERABLE error: the harness loop appends it as
// a tool result and lets the agent adapt or retry. This is the right default for
// almost every tool failure (bad arguments, missing file, transient I/O).
// Mirrors Rust's ToolOutput::error.
func NewToolOutputError(message string) ToolOutput {
	return ToolOutput{Kind: ToolOutputError, Message: message, Recoverable: true}
}

// NewToolOutputFatal returns a FATAL error: continuing is pointless, so the run
// halts. Reserve for genuinely unrecoverable conditions; prefer NewToolOutputError
// when the agent could reasonably do something different next turn. Mirrors
// Rust's ToolOutput::fatal.
func NewToolOutputFatal(message string) ToolOutput {
	return ToolOutput{Kind: ToolOutputError, Message: message, Recoverable: false}
}

// HarnessToolResult is the recorded outcome of one tool call dispatch by the
// harness. Distinct from the model-side ToolResult in model.go which carries
// tool_use_id; this one wraps the discriminated ToolOutput.
type HarnessToolResult struct {
	CallID string     `json:"call_id"`
	Output ToolOutput `json:"output"`
}

// SandboxViolationKind discriminates SandboxViolation variants.
type SandboxViolationKind string

const (
	SandboxPathEscape        SandboxViolationKind = "path_escape"
	SandboxNetworkViolation  SandboxViolationKind = "network_violation"
	SandboxPathDenied        SandboxViolationKind = "path_denied"
	SandboxReadOnly          SandboxViolationKind = "read_only_violation"
	SandboxExtensionDenied   SandboxViolationKind = "extension_denied"
	SandboxFileSizeExceeded  SandboxViolationKind = "file_size_exceeded"
	SandboxDisallowedCommand SandboxViolationKind = "disallowed_command"
)

// SandboxViolation is a sandbox-side rejection. Issue #6 expands the variant
// set; the on-wire shape is a flat tagged object discriminated by Kind.
//
// Per-variant fields (only the relevant ones are populated):
//   - path_escape:         Path
//   - path_denied:         Path, MatchedRule
//   - read_only_violation: Path
//   - extension_denied:    Path, Extension
//   - file_size_exceeded:  Path, Size, Limit
//   - disallowed_command:  Command
//   - network_violation:   Host
type SandboxViolation struct {
	Kind        SandboxViolationKind `json:"kind"`
	Path        string               `json:"-"`
	Host        string               `json:"-"`
	MatchedRule string               `json:"-"`
	Extension   string               `json:"-"`
	Size        uint64               `json:"-"`
	Limit       uint64               `json:"-"`
	Command     string               `json:"-"`
}

// MarshalJSON serialises as a flat tagged object.
func (s SandboxViolation) MarshalJSON() ([]byte, error) {
	switch s.Kind {
	case SandboxPathEscape, SandboxReadOnly:
		return json.Marshal(struct {
			Kind SandboxViolationKind `json:"kind"`
			Path string               `json:"path"`
		}{s.Kind, s.Path})
	case SandboxPathDenied:
		return json.Marshal(struct {
			Kind        SandboxViolationKind `json:"kind"`
			Path        string               `json:"path"`
			MatchedRule string               `json:"matched_rule"`
		}{s.Kind, s.Path, s.MatchedRule})
	case SandboxExtensionDenied:
		return json.Marshal(struct {
			Kind      SandboxViolationKind `json:"kind"`
			Path      string               `json:"path"`
			Extension string               `json:"extension"`
		}{s.Kind, s.Path, s.Extension})
	case SandboxFileSizeExceeded:
		return json.Marshal(struct {
			Kind  SandboxViolationKind `json:"kind"`
			Path  string               `json:"path"`
			Size  uint64               `json:"size"`
			Limit uint64               `json:"limit"`
		}{s.Kind, s.Path, s.Size, s.Limit})
	case SandboxDisallowedCommand:
		return json.Marshal(struct {
			Kind    SandboxViolationKind `json:"kind"`
			Command string               `json:"command"`
		}{s.Kind, s.Command})
	case SandboxNetworkViolation:
		return json.Marshal(struct {
			Kind SandboxViolationKind `json:"kind"`
			Host string               `json:"host"`
		}{s.Kind, s.Host})
	default:
		return nil, fmt.Errorf("SandboxViolation: unknown kind %q", s.Kind)
	}
}

// UnmarshalJSON decodes the flat form.
func (s *SandboxViolation) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind        SandboxViolationKind `json:"kind"`
		Path        string               `json:"path"`
		Host        string               `json:"host"`
		MatchedRule string               `json:"matched_rule"`
		Extension   string               `json:"extension"`
		Size        uint64               `json:"size"`
		Limit       uint64               `json:"limit"`
		Command     string               `json:"command"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	s.Kind = probe.Kind
	s.Path = probe.Path
	s.Host = probe.Host
	s.MatchedRule = probe.MatchedRule
	s.Extension = probe.Extension
	s.Size = probe.Size
	s.Limit = probe.Limit
	s.Command = probe.Command
	return nil
}

// Error implements error.
func (s *SandboxViolation) Error() string {
	switch s.Kind {
	case SandboxPathEscape:
		return fmt.Sprintf("sandbox: path escape: %s", s.Path)
	case SandboxNetworkViolation:
		return fmt.Sprintf("sandbox: network violation: %s", s.Host)
	case SandboxPathDenied:
		return fmt.Sprintf("sandbox: path denied: %s (rule=%s)", s.Path, s.MatchedRule)
	case SandboxReadOnly:
		return fmt.Sprintf("sandbox: read-only violation: %s", s.Path)
	case SandboxExtensionDenied:
		return fmt.Sprintf("sandbox: extension denied: %s (.%s)", s.Path, s.Extension)
	case SandboxFileSizeExceeded:
		return fmt.Sprintf("sandbox: file size exceeded: %s (%d > %d)", s.Path, s.Size, s.Limit)
	case SandboxDisallowedCommand:
		return fmt.Sprintf("sandbox: disallowed command: %s", s.Command)
	default:
		return fmt.Sprintf("sandbox violation: %s", s.Kind)
	}
}

// IsAlwaysHalt reports whether this violation is Layer-1 always-halt.
func (s SandboxViolation) IsAlwaysHalt() bool {
	return s.Kind == SandboxPathEscape || s.Kind == SandboxNetworkViolation
}

// ============================================================================
// Operation — read | write | execute
// ============================================================================

// Operation tags the intent of a path resolution. Read operations are
// permitted in read-only mode; Write and Execute are not.
type Operation string

const (
	OperationRead    Operation = "read"
	OperationWrite   Operation = "write"
	OperationExecute Operation = "execute"
)

// ============================================================================
// WorkspaceConfig — sandbox construction inputs
// ============================================================================

// WorkspaceConfig is the input to NewWorkspaceScopedSandbox. Paths are kept
// as strings to stay portable across the cross-language fixtures; the
// canonical, OS-specific resolution happens inside the sandbox.
type WorkspaceConfig struct {
	Root              string   `json:"root"`
	AllowedPaths      []string `json:"allowed_paths,omitempty"`
	DeniedPaths       []string `json:"denied_paths,omitempty"`
	AllowedExtensions []string `json:"allowed_extensions,omitempty"`
	DeniedExtensions  []string `json:"denied_extensions,omitempty"`
	ReadOnly          bool     `json:"read_only,omitempty"`
	MaxFileSize       uint64   `json:"max_file_size,omitempty"`
}

// ============================================================================
// IsolationMode — sealed interface
// ============================================================================

// IsolationMode is a sealed interface with concrete impls
// IsolationWorkspaceScoped, IsolationBubblewrap, IsolationDocker, and — gated
// behind the `dangerous` build tag (issue #34) — IsolationNone. The unexported
// sealedIsolationMode() method seals the type so external implementations
// cannot satisfy it.
//
// IsolationNone (no path enforcement) is a named safety footgun: it is defined
// only in dangerous.go (`//go:build dangerous`) and absent from the default
// build, so it cannot be selected by accident. The safe-by-default isolation
// mode is IsolationWorkspaceScoped.
type IsolationMode interface {
	sealedIsolationMode()
	Kind() string
}

type IsolationWorkspaceScoped struct{}
type IsolationBubblewrap struct {
	Profile BwrapProfile `json:"profile"`
}
type IsolationDocker struct {
	Image   string        `json:"image"`
	Network NetworkPolicy `json:"network"`
}

func (IsolationWorkspaceScoped) sealedIsolationMode() {}
func (IsolationBubblewrap) sealedIsolationMode()      {}
func (IsolationDocker) sealedIsolationMode()          {}

func (IsolationWorkspaceScoped) Kind() string { return "workspace_scoped" }
func (IsolationBubblewrap) Kind() string      { return "bubblewrap" }
func (IsolationDocker) Kind() string          { return "docker" }

// BwrapProfile is a placeholder for the bubblewrap configuration. Fields
// will be filled in when a bubblewrap backend lands.
type BwrapProfile struct{}

// ============================================================================
// NetworkPolicy — sealed interface
// ============================================================================

// NetworkPolicy is a sealed interface with NetworkNone, NetworkAllowlist,
// NetworkFull concrete impls.
type NetworkPolicy interface {
	sealedNetworkPolicy()
	Kind() string
}

type NetworkNone struct{}
type NetworkAllowlist struct {
	Hosts []string `json:"hosts"`
}
type NetworkFull struct{}

func (NetworkNone) sealedNetworkPolicy()      {}
func (NetworkAllowlist) sealedNetworkPolicy() {}
func (NetworkFull) sealedNetworkPolicy()      {}

func (NetworkNone) Kind() string      { return "none" }
func (NetworkAllowlist) Kind() string { return "allowlist" }
func (NetworkFull) Kind() string      { return "full" }

// HookPoint indicates where in the lifecycle a middleware hook fired.
//
// Issue #158 / Phase 3 (Q2): the rich middleware surface in package middleware
// is now canonical. HookPoint is the six-variant set it fans out over; the
// former four-variant harness-local stub was widened to match. The middleware
// package aliases its HookPoint to this type so *middleware.StandardMiddlewareChain
// satisfies the MiddlewareChain interface below structurally.
type HookPoint string

const (
	HookBeforeSession    HookPoint = "before_session"
	HookBeforeTurn       HookPoint = "before_turn"
	HookBeforeTool       HookPoint = "before_tool"
	HookAfterTool        HookPoint = "after_tool"
	HookBeforeCompletion HookPoint = "before_completion"
	HookAfterSession     HookPoint = "after_session"
)

// IsBefore reports whether the hook is ordered ascending by priority.
func (h HookPoint) IsBefore() bool {
	switch h {
	case HookBeforeSession, HookBeforeTurn, HookBeforeTool, HookBeforeCompletion:
		return true
	}
	return false
}

// IsAfter reports whether the hook is ordered descending by priority.
func (h HookPoint) IsAfter() bool {
	return h == HookAfterTool || h == HookAfterSession
}

// AllowsSurfaceToHuman reports whether SurfaceToHuman is legal at this hook.
func (h HookPoint) AllowsSurfaceToHuman() bool {
	return h == HookBeforeTool || h == HookBeforeCompletion
}

// AllowsForceAnotherTurn reports whether ForceAnotherTurn is legal at this hook.
func (h HookPoint) AllowsForceAnotherTurn() bool {
	return h == HookBeforeCompletion
}

// TerminationDecisionKind discriminates TerminationDecision variants.
type TerminationDecisionKind string

const (
	TerminationContinue TerminationDecisionKind = "continue"
	TerminationHalt     TerminationDecisionKind = "halt"
)

// TerminationDecision is the output of TerminationPolicy.Evaluate.
type TerminationDecision struct {
	Kind   TerminationDecisionKind `json:"kind"`
	Reason string                  `json:"reason,omitempty"`
}

// SessionState is round-tripped opaquely across pause/resume. The harness
// does not interpret its contents; ContextManager (#7) and MemoryProvider
// (#8) own the schema.
type SessionState struct {
	Messages []Message      `json:"messages"`
	Extras   map[string]any `json:"extras"`
}

// MarshalJSON ensures Messages serialises as [] and Extras as {} rather than
// null/omitted. Extras is always emitted (matching the Rust serde-default and
// Python default-factory siblings, both of which serialise an empty extras as
// {}) so PausedState / SessionState round-trips are byte-identical across the
// four languages — see the shared escalation fixture (issue #80).
func (s SessionState) MarshalJSON() ([]byte, error) {
	type alias SessionState
	a := alias(s)
	if a.Messages == nil {
		a.Messages = []Message{}
	}
	if a.Extras == nil {
		a.Extras = map[string]any{}
	}
	return json.Marshal(a)
}

// cloneSessionState returns a shallow copy of s whose Messages slice is a fresh
// backing array (not aliased to s.Messages). Used by the PlanExecute plan phase
// (#93) to seed the planning directive onto a throwaway state without mutating
// the shared session that the execute phase threads through. Appending to the
// clone's Messages must not touch the caller's slice, so the slice is copied;
// Extras is shared by reference (the plan turn never mutates it).
func cloneSessionState(s *SessionState) *SessionState {
	if s == nil {
		return &SessionState{}
	}
	return &SessionState{
		Messages: append([]Message(nil), s.Messages...),
		Extras:   s.Extras,
	}
}

// ============================================================================
// Forward-declared sibling interfaces
// ============================================================================

// ToolRegistry (#4) is defined in tool_registry.go.

// SandboxProvider (#6) validates tool calls against sandbox policy.
//
// Issue #5 adds ExecuteCommand, HandleLargeOutput, and ResolvePath so the
// standard tool catalogue can be built before #6 lands its canonical
// sandbox. The DefaultSandbox struct (sandbox_defaults.go) implements those
// three methods with non-sandboxed defaults; embed it in any stub
// implementation that only cares about Validate.
//
// These defaults are NOT production-safe: ExecuteCommand spawns processes
// directly with no sandboxing, ResolvePath returns the input as-is, and
// HandleLargeOutput truncates inline without offloading. Issue #6 must
// override these.
type SandboxProvider interface {
	Validate(ctx context.Context, call ToolCall) *SandboxViolation

	// ExecuteCommand runs a subprocess. working_dir may be "" to inherit.
	// A non-zero timeout (Duration > 0) bounds execution.
	ExecuteCommand(ctx context.Context, command string, args []string, workingDir string, timeout time.Duration) (CommandOutput, *SandboxViolation)

	// HandleLargeOutput head+tail-truncates content and may offload the full
	// content to a backing file. Truncated == false means the input was
	// returned unchanged.
	HandleLargeOutput(ctx context.Context, content string, callID string, headTokens uint32, tailTokens uint32) TruncatedOutput

	// ResolvePath canonicalizes a raw path against the sandbox root and
	// validates it against the workspace policy for the given operation.
	ResolvePath(ctx context.Context, path string, op Operation) (string, *SandboxViolation)

	// IsolationMode returns the active isolation mode (used by middleware /
	// observability).
	IsolationMode() IsolationMode

	// WorkspaceRoot returns the canonical workspace root path.
	WorkspaceRoot() string
}

// CommandOutput is the result of a subprocess executed via the sandbox.
type CommandOutput struct {
	Stdout    string `json:"stdout"`
	Stderr    string `json:"stderr"`
	ExitCode  int    `json:"exit_code"`
	TimedOut  bool   `json:"timed_out,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
}

// TruncatedOutput is the result of HandleLargeOutput. Content equals the
// original input iff Truncated == false. FullRef is populated when the
// sandbox offloads the original to a backing file.
type TruncatedOutput struct {
	Content      string   `json:"content"`
	Truncated    bool     `json:"truncated"`
	FullRef      *FileRef `json:"full_ref,omitempty"`
	OriginalSize uint64   `json:"original_size"`
}

// FileRef references a file holding offloaded tool output.
type FileRef struct {
	Path    string `json:"path"`
	ByteLen uint64 `json:"byte_len"`
}

// ----------------------------------------------------------------------------
// ContextSources bridge types (issue #115 / SC-26)
// ----------------------------------------------------------------------------
//
// The rich ContextSources/Guide/MemoryItem/ComposedPrompt types live in the
// contextmgr subpackage, which imports this root package — so they cannot be
// threaded through the root ContextManager.Assemble seam without forming an
// import cycle. These lightweight root-package mirrors are the harness-loop's
// view of the rich types, exactly the bridge-type pattern CompactionTurn and
// CompactionVerificationResult already use. The harness builds a ContextSources
// per turn (build_context_sources) and threads it into Assemble; the production
// StandardCompactionAdapter renders it into a leading System block. The empty
// (zero) value renders to nothing, so a harness with no sources stays
// byte-identical to the pre-#115 pass-through.

// Guide is the harness-loop mirror of contextmgr.Guide (issue #115 / SC-26 /
// #9): a unit of structural context — a skill body, a playbook, domain
// knowledge — injected into the assembled context. Heading is derived from ID;
// Content is the final rendered body.
type Guide struct {
	ID      string `json:"id"`
	Content string `json:"content"`
}

// MemoryItem is the harness-loop mirror of contextmgr.MemoryItem (issue #115 /
// SC-26 / #8). #163 wired the MemoryProvider into the live loop: when a
// MemoryConfig is set, the harness queries the provider each turn and populates
// ContextSources.Memory with these items (Key = source memory id, Content =
// memory body), rendered into the structural System block after guides.
type MemoryItem struct {
	Key     string `json:"key"`
	Content string `json:"content"`
}

// ComposedPrompt is the harness-loop mirror of contextmgr.ComposedPrompt (issue
// #115 / SC-26 / #14): the composed static system prompt. Empty until the
// chunk-provider path is wired live.
type ComposedPrompt struct {
	Rendered string `json:"rendered"`
}

// ContextSources is the harness-loop mirror of contextmgr.ContextSources (issue
// #115 / SC-26): the rich per-turn inputs — guides, merged memory, tool
// schemas, and the composed static prompt — threaded into the structural
// ContextManager.Assemble seam so a manager can place them in structural slots
// instead of the caller injecting them ad-hoc as User messages. The zero value
// is the empty path (renders to nothing).
type ContextSources struct {
	Guides         []Guide        `json:"guides"`
	Memory         []MemoryItem   `json:"memory"`
	ToolSchemas    []ToolSchema   `json:"tool_schemas"`
	ComposedPrompt ComposedPrompt `json:"composed_prompt"`
}

// ContextManager (#7) assembles per-turn context and appends results.
type ContextManager interface {
	// Assemble builds one turn's model input. sources (issue #115 / SC-26)
	// carries the rich ContextSources — guides, merged memory, tool schemas,
	// and the composed static prompt — so a manager can place them in
	// structural slots instead of the caller injecting them ad-hoc as User
	// messages. Managers that do not consume sources may ignore the argument
	// (the pre-#115 behaviour); the production StandardCompactionAdapter renders
	// them into a leading System block.
	Assemble(ctx context.Context, session *SessionState, task *Task, sources ContextSources) Context
	AppendToolResult(ctx context.Context, session *SessionState, result *HarnessToolResult)
	AppendUserMessage(ctx context.Context, session *SessionState, text string)
	ShouldCompact(session *SessionState) bool
}

// CompactionTurn bundles everything the harness compaction loop (issue #46)
// needs to run one compaction turn and verify its result.
//
// The harness loop operates on the opaque SessionState; the rich
// compaction/verification API (contextmgr.ContextManager, CompactionVerifier)
// operates on contextmgr.SessionState. This struct is the bridge: a
// CompactingContextManager projects everything the loop needs into one value,
// so the loop never has to know which concrete state type its manager uses
// internally — mirroring the Rust reference's CompactionTurn.
//
// Context is fed straight to Agent.Turn to produce the summary; PreserveHints
// and VerificationState are opaque payloads handed to the CompactionVerifier
// (the concrete verifier type-asserts them back to its own types, keeping the
// root package free of an import cycle on contextmgr). On a verification
// failure the loop re-runs the turn with InjectMissingItems applied to Context.
type CompactionTurn struct {
	// Context to feed Agent.Turn to elicit the summary.
	Context Context
	// PreserveHints are the preservation hints handed to the verifier. Opaque
	// to the loop; the concrete verifier type-asserts the concrete type.
	PreserveHints any
	// VerificationState is the verifier-facing session state (the rich
	// contextmgr.SessionState). Opaque to the loop.
	VerificationState any
	// MessagesRemoved is the count of messages about to be removed — used to
	// stamp the compaction span.
	MessagesRemoved uint32
}

// CompactingContextManager is the OPTIONAL compaction surface a ContextManager
// may also implement (issue #46). The harness loop type-asserts its held
// ContextManager to this interface; a manager that does NOT implement it has
// compaction skipped entirely — Go's equivalent of the Rust reference's
// default-bodied trait methods (should_compact defaults false / skip).
type CompactingContextManager interface {
	// PrepareCompactionTurn builds the inputs for one compaction turn. The
	// bool is false when there is nothing to compact (e.g. history shorter
	// than the preserve window), in which case the harness skips compaction.
	PrepareCompactionTurn(session *SessionState) (*CompactionTurn, bool)
	// InjectMissingItems mutates a compaction Context in place to request a
	// revised summary on retry, appending the standard "Your summary is
	// missing these items: {missing}. Please revise." user message.
	InjectMissingItems(c *Context, missing []string)
	// ApplyCompaction accepts a verified (or accepted-anyway) summary into the
	// session, replacing the compacted span.
	ApplyCompaction(session *SessionState, summary string)
}

// AssistantMessageAppender is the OPTIONAL seam a ContextManager may also
// implement to record the assistant's turn (the model's output text and/or the
// tool calls it requested) in the conversation, so the next Assemble() reflects
// what the agent already did. The harness loop type-asserts its held
// ContextManager to this interface; a manager that does NOT implement it simply
// skips the append — Go's equivalent of the Rust reference's default-no-op
// ContextManager::append_assistant_message trait method. Without this the model
// loses track of its own prior actions and the conversation is malformed (a tool
// result with no preceding assistant tool_use).
type AssistantMessageAppender interface {
	AppendAssistantMessage(ctx context.Context, session *SessionState, message Message)
}

// ToolResultReplacer is the OPTIONAL seam (issue #158 / SC-9) a ContextManager
// may also implement to re-render the tool-result message previously appended at
// messageIndex with a fresh rendering of result. The harness loop calls this
// from the AfterTool middleware hook when a middleware rewrote a result in place,
// so the rewrite reaches the next model turn. messageIndex is the position in
// session.Messages recorded right after the original AppendToolResult.
//
// The harness loop type-asserts its held ContextManager to this interface; a
// manager that does NOT implement it simply skips the re-render (the rewrite
// does not propagate, which is the pre-#158 behaviour) — Go's equivalent of the
// Rust reference's default-no-op ContextManager::replace_tool_result trait
// method. An out-of-range messageIndex must be a defensive no-op.
type ToolResultReplacer interface {
	ReplaceToolResult(ctx context.Context, session *SessionState, messageIndex int, result *HarnessToolResult)
}

// lastMessageIndex returns the index of the most-recently-appended message, or
// -1 if the session has none. Mirrors the Rust reference's saturating_sub(1) at
// the SC-9 index-recording sites (it is always called right after an append, so
// the slice is non-empty in practice).
func lastMessageIndex(session *SessionState) int {
	return len(session.Messages) - 1
}

// harnessResultsToToolResults projects the harness-side []HarnessToolResult onto
// the model-side []ToolResult the rich AfterTool middleware hook expects
// (issue #158 / SC-9). A Success becomes a non-error result carrying its
// content; an Error becomes an is_error result carrying its message; other
// (non-terminal) outputs render as empty content. The Go middleware chain
// operates on model.ToolResult (Content/IsError) rather than the harness's
// discriminated HarnessToolResult, so this bridge is required (Rust has a single
// ToolResult type and needs no conversion).
func harnessResultsToToolResults(approved []HarnessToolResult) []ToolResult {
	out := make([]ToolResult, len(approved))
	for i, r := range approved {
		tr := ToolResult{ToolUseID: r.CallID}
		switch r.Output.Kind {
		case ToolOutputSuccess:
			tr.Content = r.Output.Content
		case ToolOutputError:
			tr.Content = r.Output.Message
			tr.IsError = true
		}
		out[i] = tr
	}
	return out
}

// applyToolResultRewrites folds the (possibly middleware-rewritten) model-side
// results back onto the harness-side approved slice in place (issue #158 / SC-9).
// An is_error result becomes a recoverable ToolOutputError carrying the message;
// otherwise a ToolOutputSuccess carrying the content. Indices align 1:1 (the
// chain mutates the same slice it was handed); extra results are ignored.
func applyToolResultRewrites(approved []HarnessToolResult, results []ToolResult) {
	for i := range approved {
		if i >= len(results) {
			break
		}
		r := results[i]
		if r.IsError {
			approved[i].Output = ToolOutput{
				Kind:        ToolOutputError,
				Message:     r.Content,
				Recoverable: true,
			}
		} else {
			approved[i].Output = ToolOutput{
				Kind:    ToolOutputSuccess,
				Content: r.Content,
			}
		}
	}
}

// TokenBudgetReader is the OPTIONAL seam (issue #57) a ContextManager may also
// implement so the harness can stamp the real post-compaction token budget onto
// the Compaction span. The loop type-asserts its held ContextManager to this
// interface after ApplyCompaction; a manager that does NOT implement it falls
// back to TokensAfter == TokensBefore (the old behavior). The bool is false when
// the manager tracks no budget for this session.
type TokenBudgetReader interface {
	TokenBudgetUsed(session *SessionState) (uint32, bool)
}

// CompactionVerificationResult is the verdict of a CompactionVerifier check,
// in root-package terms (mirrors contextmgr.CompactionVerificationResult).
type CompactionVerificationResult struct {
	Passed       bool
	MissingItems []string
}

// CompactionVerifier is the harness-loop seam for post-compaction verification
// (issue #29/#46). It is a lightweight, synchronous, computational sensor that
// runs after the agent produces a compaction summary and before the summary is
// applied. Defined in root-package terms so the loop needs no contextmgr
// import; the standard KeyTermVerifier is adapted into this seam by
// contextmgr.NewKeyTermVerifier (the adapter type-asserts the opaque bridge
// fields back to its concrete types).
type CompactionVerifier interface {
	Verify(summary string, turn *CompactionTurn) CompactionVerificationResult
}

// TerminationPolicy (#13) decides whether the loop continues after a final
// response.
type TerminationPolicy interface {
	Evaluate(ctx context.Context, session *SessionState, budgetUsed *BudgetSnapshot) TerminationDecision
}

// MiddlewareDecisionKind discriminates MiddlewareDecision variants.
//
// Issue #158: collapsed onto the rich (issue #11) five-kind shape. The dropped
// stub field was Calls — BeforeTool now mutates the dispatched calls in place
// (SC-11) rather than handing back a replacement slice. The middleware package
// aliases its MiddlewareDecisionKind / MiddlewareDecision to these types.
type MiddlewareDecisionKind string

const (
	MiddlewareContinue                 MiddlewareDecisionKind = "continue"
	MiddlewareContinueWithModification MiddlewareDecisionKind = "continue_with_modification"
	MiddlewareForceAnotherTurn         MiddlewareDecisionKind = "force_another_turn"
	MiddlewareHalt                     MiddlewareDecisionKind = "halt"
	MiddlewareSurfaceToHuman           MiddlewareDecisionKind = "surface_to_human"
)

// MiddlewareDecision is the output of the rich MiddlewareChain fan-out (#11,
// wired by #158). Per-variant fields (only the relevant one is populated):
//   - force_another_turn: Inject (the concatenated injection text)
//   - halt:               Reason
//   - surface_to_human:   Request
type MiddlewareDecision struct {
	Kind    MiddlewareDecisionKind `json:"kind"`
	Inject  string                 `json:"-"`
	Reason  string                 `json:"-"`
	Request *HumanRequest          `json:"-"`
}

// MarshalJSON serialises as a flat tagged object matching the Rust shape.
func (d MiddlewareDecision) MarshalJSON() ([]byte, error) {
	switch d.Kind {
	case MiddlewareContinue, MiddlewareContinueWithModification:
		return json.Marshal(struct {
			Kind MiddlewareDecisionKind `json:"kind"`
		}{d.Kind})
	case MiddlewareForceAnotherTurn:
		return json.Marshal(struct {
			Kind   MiddlewareDecisionKind `json:"kind"`
			Inject string                 `json:"inject"`
		}{d.Kind, d.Inject})
	case MiddlewareHalt:
		return json.Marshal(struct {
			Kind   MiddlewareDecisionKind `json:"kind"`
			Reason string                 `json:"reason"`
		}{d.Kind, d.Reason})
	case MiddlewareSurfaceToHuman:
		return json.Marshal(struct {
			Kind    MiddlewareDecisionKind `json:"kind"`
			Request *HumanRequest          `json:"request"`
		}{d.Kind, d.Request})
	default:
		return nil, fmt.Errorf("MiddlewareDecision: unknown kind %q", d.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (d *MiddlewareDecision) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind    MiddlewareDecisionKind `json:"kind"`
		Inject  string                 `json:"inject"`
		Reason  string                 `json:"reason"`
		Request *HumanRequest          `json:"request"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	d.Kind = probe.Kind
	d.Inject = probe.Inject
	d.Reason = probe.Reason
	d.Request = probe.Request
	return nil
}

// MiddlewareChain (#11) is the canonical, priority-ordered, six-hook middleware
// fan-out the ReAct loop wires directly (issue #158). The interface lives here
// (consumer side, so the harness can name it) but the concrete implementation —
// *middleware.StandardMiddlewareChain — lives in package middleware, which
// imports this package; it satisfies this interface structurally (the same
// pattern as CompactionVerifier / HarnessObserver). The harness loop fires only
// BeforeTurn / BeforeTool / AfterTool / BeforeCompletion; BeforeSession /
// AfterSession are part of the surface but not yet wired (deferred, matching
// the Rust reference).
type MiddlewareChain interface {
	Register(m Middleware) error
	FireBeforeSession(ctx context.Context, task *Task, sid SessionID) (MiddlewareDecision, error)
	FireBeforeTurn(ctx context.Context, session *SessionState, turn uint32) (MiddlewareDecision, error)
	FireBeforeTool(ctx context.Context, calls *[]ToolCall, turn uint32) (MiddlewareDecision, error)
	FireAfterTool(ctx context.Context, calls []ToolCall, results *[]ToolResult) (MiddlewareDecision, error)
	FireBeforeCompletion(ctx context.Context, response string, turn uint32, state *SessionState) (MiddlewareDecision, error)
	FireAfterSession(ctx context.Context, result *RunResult, sid SessionID) error
}

// MiddlewareHookContext is the per-firing payload handed to Middleware.Handle.
//
// Issue #158: moved here (from package middleware) so the harness-side
// MiddlewareChain interface can name the Middleware contract without an import
// cycle. The middleware package aliases HookContext to this type. It is a
// tagged union — exactly one variant payload is meaningful per firing,
// selected by Point. Mutable fields are passed by pointer where the spec allows
// modification (BeforeTurn Session, BeforeTool Calls, AfterTool Results).
type MiddlewareHookContext struct {
	Point HookPoint

	// BeforeSession
	Task      *Task
	SessionID *SessionID

	// BeforeTurn
	Session    *SessionState
	TurnNumber uint32

	// BeforeTool — Calls is mutable (in-place modification permitted).
	Calls *[]ToolCall

	// AfterTool — Results is mutable; CallsRO is a read-only snapshot.
	CallsRO []ToolCall
	Results *[]ToolResult

	// BeforeCompletion
	Response     string
	SessionState *SessionState

	// AfterSession
	Result *RunResult
}

// Middleware is a single interceptor registrable on a MiddlewareChain.
//
// Issue #158: the interface lives here (consumer side) so the harness-side
// MiddlewareChain.Register signature can name it without importing package
// middleware (which would be a cycle — middleware imports this package). The
// middleware package aliases Middleware / MiddlewareHookContext to these types,
// so its concrete middleware and *StandardMiddlewareChain satisfy these
// interfaces structurally. Register is unused by the loop (only the standard
// chain constructor / tests register), but is part of the canonical surface.
type Middleware interface {
	Handle(ctx context.Context, hctx MiddlewareHookContext) (MiddlewareDecision, error)
	Hooks() []HookPoint
	Priority() int
	Name() string
}

// HarnessObserver (#12) is the consumer-side observability seam the ReAct
// loop emits through.
//
// The canonical, full ObservabilityProvider (EmitTurn/EmitToolCall/
// FlushSession/...) and its span types live in the `observability`
// subpackage. That package imports this root package (it aliases SessionID,
// TaskID, StopReason from here), so the root package CANNOT import
// `observability` back — doing so is a compile-time import cycle. Following
// the project's consumer-side-interface convention, the loop therefore emits
// through this narrow interface, expressed entirely in root-package types.
// The `observability` package supplies the adapter
// (observability.NewHarnessObserver) that builds the real spans and forwards
// to an observability.ObservabilityProvider.
//
// Mirrors the Rust reference's in-loop span emission: a TurnSpan after every
// agent turn, a child ToolCallSpan after each tool dispatch, and a terminal
// SetSessionOutcome + FlushSession finalize on Success/Failure (never on a
// WaitingForHuman pause).
//
// Methods that take no context are fire-and-forget and must never block or
// affect control flow.
type HarnessObserver interface {
	// EmitTurn records one agent turn. spanID is the caller-chosen,
	// stable turn span id (mirrors Rust's "{session}-turn-{n}"); the adapter
	// returns it as the parent handle for this turn's tool-call spans.
	// errorMessage is non-empty only when the turn errored.
	EmitTurn(
		spanID string,
		sessionID SessionID,
		taskID TaskID,
		turnNumber uint32,
		startedAt string,
		durationMs uint64,
		usage TokenUsage,
		costUSD float64,
		stopReason StopReason,
		toolCallsRequested uint32,
		errorMessage string,
		// content (issue #64): the model's output text and the tool calls it
		// requested this turn. Captured only when the observer's content-capture
		// guard is ON; the observer truncates and gates these. outputText is ""
		// when the turn produced no final text; calls is nil when no tool calls.
		outputText string,
		calls []ToolCall,
		// inputMessages (issue #64): the assembled INPUT prompt the model saw
		// this turn (system-first, then history order). Captured only when the
		// observer's content-capture guard is ON; the observer maps roles,
		// renders/truncates content, and gates these. nil when unavailable.
		inputMessages []Message,
	)

	// EmitToolCall records one tool dispatch as a child of parentSpanID.
	EmitToolCall(
		spanID string,
		parentSpanID string,
		sessionID SessionID,
		taskID TaskID,
		toolName string,
		callID string,
		startedAt string,
		durationMs uint64,
		parametersSizeBytes uint64,
		outputSizeBytes uint64,
		truncated bool,
		isError bool,
		// content (issue #64): the tool-call arguments and the tool result body.
		// Captured only when the observer's content-capture guard is ON; the
		// observer truncates and gates these. arguments is nil when unavailable;
		// resultContent is "" when the tool produced no result body.
		arguments json.RawMessage,
		resultContent string,
	)

	// SetSessionOutcome records the terminal outcome. outcome is the harness's
	// 3-state terminal signal (success / failure / escalated, issue #80);
	// failureReason is non-empty only for TerminalFailure.
	SetSessionOutcome(sessionID SessionID, outcome TerminalOutcome, failureReason string)

	// FlushSession flushes the durable session record.
	FlushSession(ctx context.Context, sessionID SessionID)

	// CostFor computes USD cost for a turn using the observer's pricing.
	// The loop calls this so cost is stamped at emit time (per spec); kept
	// on the seam so the root package needs no PricingTable type.
	CostFor(usage TokenUsage) float64

	// EmitCompaction records an accepted-summary compaction as a context span
	// (issue #46). spanID is the caller-chosen, stable span id.
	EmitCompaction(
		spanID string,
		sessionID SessionID,
		taskID TaskID,
		startedAt string,
		messagesRemoved uint32,
		tokensBefore uint32,
		tokensAfter uint32,
		tokensReclaimed uint32,
	)

	// EmitCompactionVerificationFailed records a warn-level event for a
	// compaction summary that failed verification on every attempt and was
	// accepted anyway (issue #46). acceptedAnyway is always true for this
	// event. Fire-and-forget; never affects control flow.
	EmitCompactionVerificationFailed(
		spanID string,
		sessionID SessionID,
		taskID TaskID,
		startedAt string,
		missingItems []string,
		acceptedAnyway bool,
	)

	// EmitHillClimbingIteration records one iteration of a HillClimbing loop
	// strategy run as a warn-level span (issue #60). Emitted fire-and-forget
	// after each iteration's metric evaluation so the run is traceable
	// per-iteration with its metric value and delta. iteration is the 0-based
	// index (0 = the pure baseline). hasMetric is false on crashed/timeout rows
	// (no comparable metric) — when false, metricValue/delta are ignored.
	// hasDelta is false for the baseline and for crashed/timeout rows. status is
	// the snake_case IterationStatus string the harness recorded
	// (kept/discarded/crashed/timeout). reverted is true when the harness ran a
	// git reset for this iteration. Never affects control flow.
	EmitHillClimbingIteration(
		spanID string,
		sessionID SessionID,
		taskID TaskID,
		startedAt string,
		iteration uint32,
		metricValue float64,
		hasMetric bool,
		delta float64,
		hasDelta bool,
		status string,
		reverted bool,
	)

	// EmitConsultSpawned records a worker pausing mid-loop to consult a
	// parent-spawned helper (issue #114), emitted when the loop returns
	// RunResult.Consult. Lightweight — alongside the SkillInjected event family.
	// Fire-and-forget; never affects control flow.
	EmitConsultSpawned(
		spanID string,
		sessionID SessionID,
		taskID TaskID,
		startedAt string,
		consultKind string,
	)

	// EmitConsultResumed records a paused worker being resumed after a consult
	// (issue #114), emitted by the ResumeConsult seam. answered is false when the
	// resume carried a budget-exhausted soft-fail rather than a handler answer.
	// Fire-and-forget; never affects control flow.
	EmitConsultResumed(
		spanID string,
		sessionID SessionID,
		taskID TaskID,
		startedAt string,
		consultKind string,
		answered bool,
	)

	// EmitToolErrorLoopDetected records the ReAct consecutive-recoverable-tool-error
	// breaker DETECTING a loop at N (issue #137): a tool hit ErrorLoopThreshold
	// consecutive identical-argument recoverable errors and ONE corrective message
	// was injected. A warning — the loop continues. Carries the tool name and the
	// consecutive-error count (= N). Fire-and-forget; never affects control flow.
	EmitToolErrorLoopDetected(
		spanID string,
		sessionID SessionID,
		taskID TaskID,
		startedAt string,
		toolName string,
		consecutiveErrors uint32,
	)

	// EmitToolErrorLoopBroken records the ReAct breaker TRIPPING at 2*N (issue
	// #137): the same tool reached 2*ErrorLoopThreshold consecutive
	// identical-argument errors and the loop STOPPED to resolve the node's
	// BudgetExhaustedBehavior. Carries the tool name and the consecutive-error
	// count (= 2*N). Fire-and-forget; never affects control flow.
	EmitToolErrorLoopBroken(
		spanID string,
		sessionID SessionID,
		taskID TaskID,
		startedAt string,
		toolName string,
		consecutiveErrors uint32,
	)
}

// ============================================================================
// Human-in-the-loop
// ============================================================================

// RiskLevel is the severity hint attached to a ToolApproval request.
type RiskLevel string

const (
	RiskLow      RiskLevel = "low"
	RiskMedium   RiskLevel = "medium"
	RiskHigh     RiskLevel = "high"
	RiskCritical RiskLevel = "critical"
)

// ============================================================================
// EscalationAction — the operator's choice on a budget-exhaustion pause (#130)
// ============================================================================

// EscalationActionKind discriminates EscalationAction variants. Wire values are
// snake_case to match the cross-language serde shape.
type EscalationActionKind string

const (
	// EscalationContinueWithBudget grants N more steps and resumes from the
	// checkpoint. Carries a NAMED Steps field on the wire
	// ({"kind":"continue_with_budget","steps":N}) — internally-tagged unions
	// cannot carry a positional/tuple field, mirroring BudgetPolicy's value.
	EscalationContinueWithBudget EscalationActionKind = "continue_with_budget"
	// EscalationSkip marks the current task skipped; a combinator's outer loop
	// advances to its sibling tasks. Offered only by combinators (fork C).
	EscalationSkip EscalationActionKind = "skip"
	// EscalationFail aborts the node and propagates BudgetExceeded (the partial
	// is discarded, mirroring the Fail resolution contract).
	EscalationFail EscalationActionKind = "fail"
)

// EscalationAction is the operator's choice on a budget-exhaustion
// HumanRequest.BudgetExhausted pause (#130). Tagged on Kind, snake_case. Go
// idiom: a single struct with a Kind discriminator and the data-carrying field
// (Steps) valid only for the ContinueWithBudget variant — matching the existing
// HumanRequest / HumanResponse / EscalationMode union shape rather than Rust's
// enum syntax. The WIRE FORMAT is byte-identical to the Rust serde form.
type EscalationAction struct {
	Kind EscalationActionKind
	// Steps is the granted additional step allowance, valid only when
	// Kind == EscalationContinueWithBudget.
	Steps uint32
}

// ContinueWithBudgetAction builds a ContinueWithBudget escalation action.
func ContinueWithBudgetAction(steps uint32) EscalationAction {
	return EscalationAction{Kind: EscalationContinueWithBudget, Steps: steps}
}

// SkipAction builds a Skip escalation action.
func SkipAction() EscalationAction { return EscalationAction{Kind: EscalationSkip} }

// FailAction builds a Fail escalation action.
func FailAction() EscalationAction { return EscalationAction{Kind: EscalationFail} }

// MarshalJSON serialises as a "kind"-tagged object. ContinueWithBudget carries
// a named "steps" field; Skip / Fail are bare.
func (a EscalationAction) MarshalJSON() ([]byte, error) {
	switch a.Kind {
	case EscalationContinueWithBudget:
		return json.Marshal(struct {
			Kind  EscalationActionKind `json:"kind"`
			Steps uint32               `json:"steps"`
		}{a.Kind, a.Steps})
	case EscalationSkip, EscalationFail:
		return json.Marshal(struct {
			Kind EscalationActionKind `json:"kind"`
		}{a.Kind})
	default:
		return nil, fmt.Errorf("EscalationAction: unknown kind %q", a.Kind)
	}
}

// UnmarshalJSON decodes the "kind"-tagged form.
func (a *EscalationAction) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind  EscalationActionKind `json:"kind"`
		Steps uint32               `json:"steps"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	switch probe.Kind {
	case EscalationContinueWithBudget, EscalationSkip, EscalationFail:
		a.Kind = probe.Kind
		a.Steps = probe.Steps
		return nil
	default:
		return fmt.Errorf("EscalationAction: unknown kind %q", probe.Kind)
	}
}

// HumanRequestKind discriminates HumanRequest variants.
type HumanRequestKind string

const (
	HumanReqToolApproval  HumanRequestKind = "tool_approval"
	HumanReqClarification HumanRequestKind = "clarification"
	HumanReqReview        HumanRequestKind = "review"
	// HumanReqBudgetExhausted surfaces a node's budget-scope exhaustion that
	// resolved to Escalate under EscalationMode.SurfaceToHuman (#130). The
	// operator answers with a HumanResponse.Escalate carrying an EscalationAction.
	HumanReqBudgetExhausted HumanRequestKind = "budget_exhausted"
)

// HumanRequest is the question surfaced to the human-in-the-loop.
type HumanRequest struct {
	Kind HumanRequestKind `json:"kind"`
	// tool_approval
	Calls     []ToolCall `json:"-"`
	RiskLevel RiskLevel  `json:"-"`
	// clarification
	Question string `json:"-"`
	// Options are optional fixed choices for a clarification (issue #81, Q4b).
	// nil for a free-form clarification; omitted from the wire form when nil so
	// older Clarification blobs (no `options` field) round-trip unchanged.
	Options *[]string `json:"-"`
	// review
	Content string `json:"-"`
	// budget_exhausted (#130): the exhausted node's context. Phase / Policy /
	// StepsTaken / ContinuesUsed let resumeInner reconstruct the node's budget
	// scope from the request alone (fork E — no durable cross-process
	// checkpoint). PartialOutput is the node's pre-exhaustion output (optional;
	// omitted from the wire when nil to mirror Rust's Option). AvailableActions
	// is the ADVISORY action set the author offers (fork C/D).
	Phase            string
	Policy           BudgetPolicy
	StepsTaken       uint32
	ContinuesUsed    uint32
	PartialOutput    *string
	AvailableActions []EscalationAction
}

// MarshalJSON serialises as a flat tagged object.
func (h HumanRequest) MarshalJSON() ([]byte, error) {
	switch h.Kind {
	case HumanReqToolApproval:
		calls := h.Calls
		if calls == nil {
			calls = []ToolCall{}
		}
		return json.Marshal(struct {
			Kind      HumanRequestKind `json:"kind"`
			Calls     []ToolCall       `json:"calls"`
			RiskLevel RiskLevel        `json:"risk_level"`
		}{h.Kind, calls, h.RiskLevel})
	case HumanReqClarification:
		// Options omitted (not null) when absent, matching the Rust
		// skip_serializing_if so back-compat Clarification blobs are byte-identical.
		if h.Options == nil {
			return json.Marshal(struct {
				Kind     HumanRequestKind `json:"kind"`
				Question string           `json:"question"`
			}{h.Kind, h.Question})
		}
		return json.Marshal(struct {
			Kind     HumanRequestKind `json:"kind"`
			Question string           `json:"question"`
			Options  []string         `json:"options"`
		}{h.Kind, h.Question, *h.Options})
	case HumanReqReview:
		return json.Marshal(struct {
			Kind    HumanRequestKind `json:"kind"`
			Content string           `json:"content"`
		}{h.Kind, h.Content})
	case HumanReqBudgetExhausted:
		actions := h.AvailableActions
		if actions == nil {
			actions = []EscalationAction{}
		}
		// partial_output is emitted as null when nil (matching the Rust
		// Option<String> with no skip-if-none) — the *string is marshalled
		// directly so a nil pointer becomes JSON null.
		return json.Marshal(struct {
			Kind             HumanRequestKind   `json:"kind"`
			Phase            string             `json:"phase"`
			Policy           BudgetPolicy       `json:"policy"`
			StepsTaken       uint32             `json:"steps_taken"`
			ContinuesUsed    uint32             `json:"continues_used"`
			PartialOutput    *string            `json:"partial_output"`
			AvailableActions []EscalationAction `json:"available_actions"`
		}{h.Kind, h.Phase, h.Policy, h.StepsTaken, h.ContinuesUsed, h.PartialOutput, actions})
	default:
		return nil, fmt.Errorf("HumanRequest: unknown kind %q", h.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (h *HumanRequest) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind             HumanRequestKind   `json:"kind"`
		Calls            []ToolCall         `json:"calls"`
		RiskLevel        RiskLevel          `json:"risk_level"`
		Question         string             `json:"question"`
		Options          *[]string          `json:"options"`
		Content          string             `json:"content"`
		Phase            string             `json:"phase"`
		Policy           BudgetPolicy       `json:"policy"`
		StepsTaken       uint32             `json:"steps_taken"`
		ContinuesUsed    uint32             `json:"continues_used"`
		PartialOutput    *string            `json:"partial_output"`
		AvailableActions []EscalationAction `json:"available_actions"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	h.Kind = probe.Kind
	h.Calls = probe.Calls
	h.RiskLevel = probe.RiskLevel
	h.Question = probe.Question
	h.Options = probe.Options
	h.Content = probe.Content
	h.Phase = probe.Phase
	h.Policy = probe.Policy
	h.StepsTaken = probe.StepsTaken
	h.ContinuesUsed = probe.ContinuesUsed
	h.PartialOutput = probe.PartialOutput
	h.AvailableActions = probe.AvailableActions
	return nil
}

// HumanResponseKind discriminates HumanResponse variants.
type HumanResponseKind string

const (
	HumanRespAllow                 HumanResponseKind = "allow"
	HumanRespAllowWithModification HumanResponseKind = "allow_with_modification"
	HumanRespDeny                  HumanResponseKind = "deny"
	HumanRespHalt                  HumanResponseKind = "halt"
	HumanRespAnswer                HumanResponseKind = "answer"
	HumanRespApproveWithFeedback   HumanResponseKind = "approve_with_feedback"
	HumanRespReject                HumanResponseKind = "reject"
	// HumanRespEscalate delivers the operator's resolution of a
	// HumanRequest.BudgetExhausted pause (#130): the chosen EscalationAction.
	// Distinct from Allow/Halt/Deny so the budget-escalation resume is
	// unambiguous.
	HumanRespEscalate HumanResponseKind = "escalate"
)

// HumanResponse is the human's reply to a HumanRequest.
type HumanResponse struct {
	Kind     HumanResponseKind `json:"kind"`
	Calls    []ToolCall        `json:"-"`
	Reason   string            `json:"-"`
	Text     string            `json:"-"`
	Feedback string            `json:"-"`
	// Action carries the operator's choice, valid only when
	// Kind == HumanRespEscalate (#130).
	Action EscalationAction `json:"-"`
}

// EscalateResponse builds a HumanResponse.Escalate carrying an EscalationAction.
func EscalateResponse(action EscalationAction) HumanResponse {
	return HumanResponse{Kind: HumanRespEscalate, Action: action}
}

// MarshalJSON serialises as a flat tagged object.
func (h HumanResponse) MarshalJSON() ([]byte, error) {
	switch h.Kind {
	case HumanRespAllow, HumanRespHalt:
		return json.Marshal(struct {
			Kind HumanResponseKind `json:"kind"`
		}{h.Kind})
	case HumanRespAllowWithModification:
		calls := h.Calls
		if calls == nil {
			calls = []ToolCall{}
		}
		return json.Marshal(struct {
			Kind  HumanResponseKind `json:"kind"`
			Calls []ToolCall        `json:"calls"`
		}{h.Kind, calls})
	case HumanRespDeny, HumanRespReject:
		return json.Marshal(struct {
			Kind   HumanResponseKind `json:"kind"`
			Reason string            `json:"reason"`
		}{h.Kind, h.Reason})
	case HumanRespAnswer:
		return json.Marshal(struct {
			Kind HumanResponseKind `json:"kind"`
			Text string            `json:"text"`
		}{h.Kind, h.Text})
	case HumanRespApproveWithFeedback:
		return json.Marshal(struct {
			Kind     HumanResponseKind `json:"kind"`
			Feedback string            `json:"feedback"`
		}{h.Kind, h.Feedback})
	case HumanRespEscalate:
		return json.Marshal(struct {
			Kind   HumanResponseKind `json:"kind"`
			Action EscalationAction  `json:"action"`
		}{h.Kind, h.Action})
	default:
		return nil, fmt.Errorf("HumanResponse: unknown kind %q", h.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (h *HumanResponse) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind     HumanResponseKind `json:"kind"`
		Calls    []ToolCall        `json:"calls"`
		Reason   string            `json:"reason"`
		Text     string            `json:"text"`
		Feedback string            `json:"feedback"`
		Action   EscalationAction  `json:"action"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	h.Kind = probe.Kind
	h.Calls = probe.Calls
	h.Reason = probe.Reason
	h.Text = probe.Text
	h.Feedback = probe.Feedback
	h.Action = probe.Action
	return nil
}

// ============================================================================
// Mid-loop consult primitive (issue #114)
// ============================================================================
//
// A third pause/resume path that STOPS AT THE ORCHESTRATOR instead of bubbling
// to the human. A worker-side tool returns ToolOutput.Consult; the worker
// harness pauses and returns RunResult.Consult (sibling of WaitingForHuman,
// HumanRequest nil). At the subagent boundary the SubagentTool (tools/subagent.go)
// drives the full run -> consult -> route-by-kind -> budget-check -> run-handler
// -> resume-worker loop internally (mediation seam A1), so the parent
// orchestrator's model never sees the consult and depth-1 is preserved.
//
// Per go/CONVENTIONS.md the consult-handler config is a HarnessConfig struct
// FIELD (ConsultHandlers), not a builder setter — matching the established
// PlannerAgent / Verifier / VcsProvider divergence.

// ConsultRequest is the worker's free-form ask when it pauses mid-loop to
// consult a parent-spawned helper (issue #114). Kind selects the handler;
// Situation / Attempts / Question carry the free-form context the handler
// needs. All fields are REQUIRED on the wire (no omitempty) so a malformed
// request fails to round-trip rather than silently defaulting.
type ConsultRequest struct {
	// Kind is the routing key — selects the ConsultHandlerEntry in
	// HarnessConfig.ConsultHandlers.
	Kind string `json:"kind"`
	// Situation is a free-form description of where the worker is stuck.
	Situation string `json:"situation"`
	// Attempts is how many times the worker has already tried (advisory; the
	// harness enforces the per-kind budget independently).
	Attempts uint32 `json:"attempts"`
	// Question is the concrete question the worker wants answered.
	Question string `json:"question"`
}

// ConsultResponseKind discriminates ConsultResponse variants.
type ConsultResponseKind string

const (
	// ConsultRespAnswer — the handler produced an answer; Text is injected as
	// the tool RESULT for the pending consult call.
	ConsultRespAnswer ConsultResponseKind = "answer"
	// ConsultRespBudgetExhausted — the per-kind budget is exhausted under a
	// SoftFail overflow policy: the worker is resumed with Message and finishes
	// with what it has.
	ConsultRespBudgetExhausted ConsultResponseKind = "budget_exhausted"
)

// ConsultResponse is the resume input handed back to a paused worker after a
// consult (issue #114). Parallel to HumanResponse; tagged on Kind, snake_case.
type ConsultResponse struct {
	Kind ConsultResponseKind `json:"kind"`
	// answer
	Text string `json:"-"`
	// budget_exhausted
	Message string `json:"-"`
}

// MarshalJSON serialises ConsultResponse as a flat tagged object.
func (c ConsultResponse) MarshalJSON() ([]byte, error) {
	switch c.Kind {
	case ConsultRespAnswer:
		return json.Marshal(struct {
			Kind ConsultResponseKind `json:"kind"`
			Text string              `json:"text"`
		}{c.Kind, c.Text})
	case ConsultRespBudgetExhausted:
		return json.Marshal(struct {
			Kind    ConsultResponseKind `json:"kind"`
			Message string              `json:"message"`
		}{c.Kind, c.Message})
	default:
		return nil, fmt.Errorf("ConsultResponse: unknown kind %q", c.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (c *ConsultResponse) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind    ConsultResponseKind `json:"kind"`
		Text    string              `json:"text"`
		Message string              `json:"message"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	c.Kind = probe.Kind
	c.Text = probe.Text
	c.Message = probe.Message
	return nil
}

// NewConsultAnswer builds an Answer response.
func NewConsultAnswer(text string) ConsultResponse {
	return ConsultResponse{Kind: ConsultRespAnswer, Text: text}
}

// NewConsultBudgetExhausted builds a BudgetExhausted response.
func NewConsultBudgetExhausted(message string) ConsultResponse {
	return ConsultResponse{Kind: ConsultRespBudgetExhausted, Message: message}
}

// ConsultOverflowPolicyKind discriminates per-kind budget-overflow behaviours.
type ConsultOverflowPolicyKind string

const (
	// ConsultOverflowSoftFail — resume the worker with ConsultResponse.BudgetExhausted
	// so it finishes without further help.
	ConsultOverflowSoftFail ConsultOverflowPolicyKind = "soft_fail"
	// ConsultOverflowEscalateToHuman — convert the over-budget consult into
	// RunResult.WaitingForHuman so the host decides.
	ConsultOverflowEscalateToHuman ConsultOverflowPolicyKind = "escalate_to_human"
)

// ConsultOverflowPolicy is the per-kind budget-overflow behaviour (issue #114).
// Tagged on Kind, snake_case.
type ConsultOverflowPolicy struct {
	Kind ConsultOverflowPolicyKind `json:"kind"`
}

// ConsultHandlerEntry is a registered consult handler: the helper harness to
// run, the per-kind budget (max consults of this kind before overflow), and the
// overflow policy (issue #114). Held by kind in HarnessConfig.ConsultHandlers.
// The Handler is run by SubagentTool as the ORCHESTRATOR's direct child
// (depth-1, R7), never nested under the worker.
type ConsultHandlerEntry struct {
	// Handler is the helper harness run on the ConsultRequest.
	Handler Harness
	// Budget is the max number of consults of this kind before the overflow
	// policy applies.
	Budget uint32
	// Overflow is what to do once the budget is exhausted.
	Overflow ConsultOverflowPolicy
}

// ============================================================================
// PausedState / ChildPausedState
// ============================================================================

// PausedState is the harness state captured when a run pauses for a human.
// The caller persists this opaquely; the harness round-trips it through
// resume().
type PausedState struct {
	SessionID        SessionID           `json:"session_id"`
	TaskID           TaskID              `json:"task_id"`
	TurnNumber       uint32              `json:"turn_number"`
	SessionState     SessionState        `json:"session_state"`
	PendingToolCalls []ToolCall          `json:"pending_tool_calls"`
	ApprovedResults  []HarnessToolResult `json:"approved_results"`
	// HumanRequest is nil for an escalation-derived state (issue #80) — an
	// escalation has no human request. The WaitingForHuman construction paths
	// always set it. Serialised as null (not omitted) so the wire form matches
	// the Rust Option<HumanRequest> with #[serde(default)] byte-for-byte.
	HumanRequest *HumanRequest  `json:"human_request"`
	Task         Task           `json:"task"`
	BudgetUsed   BudgetSnapshot `json:"budget_used"`
	// ChildState is serialised as null (not omitted) when absent, matching the
	// Rust Option<ChildPausedState> with #[serde(default)] and the Python
	// sibling so PausedState round-trips byte-identically across languages.
	ChildState *ChildPausedState `json:"child_state"`
	// Toolset is the toolset handle of the leaf that paused (#140). Resume routes
	// pending per-node tool calls through this handle's scoped catalogue via
	// effectiveToolRegistry; an empty handle (the zero value) falls back to the
	// global catalogue. A missing "toolset" key in pre-#140 blobs unmarshals to
	// the empty handle, so old paused-state blobs deserialise unchanged. The
	// field ALWAYS serialises (even when empty, as "toolset":"") for
	// cross-language byte-parity — NO omitempty. Declared LAST so encoding/json
	// emits it last, byte-matching the fixtures.
	Toolset ToolsetRef `json:"toolset"`
}

// MarshalJSON ensures slice fields serialise as [].
func (p PausedState) MarshalJSON() ([]byte, error) {
	type alias PausedState
	a := alias(p)
	if a.PendingToolCalls == nil {
		a.PendingToolCalls = []ToolCall{}
	}
	if a.ApprovedResults == nil {
		a.ApprovedResults = []HarnessToolResult{}
	}
	return json.Marshal(a)
}

// SerializeCheckpoint produces the durable checkpoint blob the caller persists
// (#129, AC1). The SHARED pause/resume round-trip: both cross-process Continue
// (a HumanRequest BudgetExhausted pause whose request carries continues_used)
// and Ralph's pause-propagation hand the SAME PausedState to the caller for
// persistence; this is the one seam they share. It is JUST the PausedState
// serialize/deserialize — NOT a unification of their CONTEXT policies (Q2): a
// Continue resumes preserving session_state.messages; Ralph re-seeds a fresh
// window from its filesystem .spore/progress.json checkpoint, which stays
// Ralph-specific. LoadCheckpoint is the resume side.
func (p PausedState) SerializeCheckpoint() ([]byte, error) {
	return json.Marshal(p)
}

// LoadCheckpoint restores a PausedState from a durable checkpoint blob (#129,
// AC1) — the resume side of SerializeCheckpoint.
func LoadCheckpoint(blob []byte) (PausedState, error) {
	var p PausedState
	if err := json.Unmarshal(blob, &p); err != nil {
		return PausedState{}, err
	}
	return p, nil
}

// ChildPausedState is the paused state for a subagent. Deliberately has no
// child_state field — the type system enforces a one-level subagent depth.
type ChildPausedState struct {
	SessionID        SessionID           `json:"session_id"`
	TaskID           TaskID              `json:"task_id"`
	TurnNumber       uint32              `json:"turn_number"`
	SessionState     SessionState        `json:"session_state"`
	PendingToolCalls []ToolCall          `json:"pending_tool_calls"`
	ApprovedResults  []HarnessToolResult `json:"approved_results"`
	// HumanRequest is nil for an escalation-derived state (issue #80). The
	// WaitingForHuman construction paths always set it. Serialised as null
	// (not omitted) to match the Rust Option<HumanRequest> wire form.
	HumanRequest     *HumanRequest  `json:"human_request"`
	Task             Task           `json:"task"`
	BudgetUsed       BudgetSnapshot `json:"budget_used"`
	ParentToolCallID string         `json:"parent_tool_call_id"`
	// Toolset is the toolset handle of the child leaf that paused (#140); same
	// semantics and serialization contract as PausedState.Toolset. ALWAYS
	// serialises ("toolset":"" when empty); a missing key in pre-#140 child blobs
	// unmarshals to the empty handle. Declared LAST to byte-match the fixtures.
	Toolset ToolsetRef `json:"toolset"`
}

// MarshalJSON ensures slice fields serialise as [].
func (c ChildPausedState) MarshalJSON() ([]byte, error) {
	type alias ChildPausedState
	a := alias(c)
	if a.PendingToolCalls == nil {
		a.PendingToolCalls = []ToolCall{}
	}
	if a.ApprovedResults == nil {
		a.ApprovedResults = []HarnessToolResult{}
	}
	return json.Marshal(a)
}

// ============================================================================
// HaltReason and RunResult
// ============================================================================

// HaltReasonKind discriminates HaltReason variants.
type HaltReasonKind string

const (
	HaltBudgetExceeded        HaltReasonKind = "budget_exceeded"
	HaltTerminationPolicyHalt HaltReasonKind = "termination_policy_halt"
	HaltMiddlewareHalt        HaltReasonKind = "middleware_halt"
	HaltAgentError            HaltReasonKind = "agent_error"
	// HaltContextError (issue #32) routes a ContextError surfaced by the
	// ContextManager during assembly (e.g. a Block-1 or Block-2 cache-hash
	// mismatch) into a halt. Mirrors HaltAgentError. This is the routing type;
	// the live StandardHarness loop does not yet trigger it because its
	// placeholder ContextManager.Assemble is infallible pending the #7
	// migration.
	HaltContextError              HaltReasonKind = "context_error"
	HaltSandboxViolation          HaltReasonKind = "sandbox_violation"
	HaltUnrecoverableToolError    HaltReasonKind = "unrecoverable_tool_error"
	HaltHumanHalted               HaltReasonKind = "human_halted"
	HaltStagnationLimitReached    HaltReasonKind = "stagnation_limit_reached"
	HaltStrategyNotYetImplemented HaltReasonKind = "strategy_not_yet_implemented"
	// HaltEmptyPlan (issue #59, Q3) is returned by StandardHarness for the
	// PlanExecute strategy when the accepted plan parsed into an EMPTY task list
	// (tasks: []). An empty plan is a failure — the run does NOT silently
	// succeed.
	HaltEmptyPlan HaltReasonKind = "empty_plan"
	// HaltStepFailed (issue #59, Q5) is returned for the PlanExecute strategy
	// when an execute step's bounded ReAct sub-loop errored or the agent
	// returned a blocked/failed outcome. A plan is a dependency chain by
	// assumption, so the whole run aborts at the failing step — execution does
	// NOT continue to the next task. Carries the failing step's positional
	// index, its instruction, and a human-readable reason derived from the
	// underlying HaltReason.
	HaltStepFailed HaltReasonKind = "step_failed"
	// HaltPlanPhaseFailed (issue #70) is returned when the PlanExecute plan
	// phase fails before producing an artifact: the planner's response was
	// unparseable, the planner requested a tool call in the one-shot turn, or
	// the artifact could not be serialized for storage. Carries the underlying
	// PlanPhaseError.
	HaltPlanPhaseFailed HaltReasonKind = "plan_phase_failed"
	// HaltSelfVerifyExhausted (issue #61, D4) is returned by the SelfVerifying
	// strategy when the build<->evaluate loop ran out of the verifier's
	// MaxIterations round-trips without an explicit Passed verdict — the
	// stagnation guard / Default-FAIL contract. A RUNTIME limit. Carries the
	// number of round-trips run (Iterations) and the last failure reason the
	// verifier gave (Reason). PEER to HaltSelfVerifyMisconfigured (NOT a sub-case
	// of it).
	HaltSelfVerifyExhausted HaltReasonKind = "self_verify_exhausted"
	// HaltSelfVerifyMisconfigured (issue #61, D4) is returned by the
	// SelfVerifying strategy when config.Verifier is nil. Likely a BUILD-TIME bug
	// in the caller's wiring. Carries a human-readable Reason. PEER to
	// HaltSelfVerifyExhausted (NOT a sub-case of it).
	HaltSelfVerifyMisconfigured HaltReasonKind = "self_verify_misconfigured"
	// HaltRalphCompletionUnmet (issue #58, B3) is returned by the Ralph loop
	// strategy when the multi-context-window continuation loop reached its
	// MaxResets cap with tasks still incomplete (the Ralph analogue of
	// HaltSelfVerifyExhausted). A RUNTIME limit — the work was attempted across
	// Iterations context windows but the filesystem-backed completion check (the
	// registered Stop hook reading .spore/progress.json) never reported done.
	// Carries the number of context-window resets performed (Iterations) and the
	// last incompletion reason (Reason, serialized as last_reason).
	HaltRalphCompletionUnmet HaltReasonKind = "ralph_completion_unmet"
	// HaltHillClimbingMisconfigured (issue #60) is returned by the HillClimbing
	// loop strategy when it cannot run because it is misconfigured — i.e.
	// config.MetricEvaluator is nil, or the iteration-0 baseline evaluation
	// itself errored (no current best to climb from). Likely a BUILD-TIME bug in
	// the caller's wiring. Surfaced as a typed halt, NOT a panic. PEER to
	// HaltSelfVerifyMisconfigured.
	HaltHillClimbingMisconfigured HaltReasonKind = "hill_climbing_misconfigured"
	// HaltConfigurationError (issue #120) is returned when ExecutionRegistry.Validate
	// fails at run entry: a handle referenced by the task's strategy tree is
	// unresolved against the configured registry, or a StrategyRef.Custom key is
	// missing. A STARTUP error surfaced before the first turn. Carries the
	// underlying HarnessError (UnresolvedHandleError or StrategyNotFoundError).
	HaltConfigurationError HaltReasonKind = "configuration_error"
	// HaltTasksBlockedByFailure (issue #126, decision A) is returned by the
	// PlanExecute DAG executor for a PARTIAL run: a task failed terminally
	// (unrecoverable error or a per-node BudgetExhausted resolving to Fail) and
	// its transitive dependents were cascade-Blocked, while unrelated branches
	// still completed. The run as a whole is a Failure, but the full partition is
	// reported: which tasks Completed, which were Blocked by the cascade, the
	// FailedTask that triggered it, and the human-readable Reason. (A run where
	// EVERY task completes is a Success, as before.)
	HaltTasksBlockedByFailure HaltReasonKind = "tasks_blocked_by_failure"
	// HaltTaskGraphCycle (issue #126) is returned by the PlanExecute DAG executor
	// when the persisted task graph contains a directed cycle, re-checked at
	// EXECUTE ENTRY as defense-in-depth (add_task already rejects cycles, but the
	// task_list tool path could in principle persist a cyclic graph out of band).
	// No task is run. Carries a human-readable Reason.
	HaltTaskGraphCycle HaltReasonKind = "task_graph_cycle"
	// HaltToolErrorLoop (issue #137) is returned when the ReAct turn loop's
	// consecutive-recoverable-tool-error breaker tripped: a tool returned
	// 2*ErrorLoopThreshold consecutive recoverable ToolOutput.Error results with
	// identical arguments, so the loop STOPPED grinding and resolved the node's
	// BudgetExhaustedBehavior WITHOUT burning the remaining budget. DISTINCT from
	// HaltBudgetExceeded (genuine budget exhaustion): the loop halted early on a
	// detected error loop, not because the step budget ran out. Carries the
	// offending Tool name and the ConsecutiveErrors count at the hard stop (2*N).
	HaltToolErrorLoop HaltReasonKind = "tool_error_loop"
	// HaltOutputSchemaViolation (issue #139) is returned when output-schema
	// enforcement is ON (HarnessConfig.EnforceOutputSchemas) for a ReactConfig
	// leaf carrying Output != nil, and the leaf's terminal FinalResponse STILL
	// failed validation after the configured OutputSchemaMaxRetries extra turns
	// were exhausted WITH budget remaining. DISTINCT from HaltBudgetExceeded: a
	// budget/turn cap that a retry would exceed surfaces the budget terminal
	// instead (budget-cap-wins precedence) — OutputSchemaViolation fires ONLY on a
	// genuine validation exhaustion. Carries the resolved Schema (canonical
	// compact key-sorted JSON), the total Attempts (1 + max_retries), and the
	// LastError (the frozen validator error string from the final attempt).
	HaltOutputSchemaViolation HaltReasonKind = "output_schema_violation"
)

// HaltReason carries the explicit reason a loop halted.
type HaltReason struct {
	Kind HaltReasonKind `json:"kind"`
	// budget_exceeded
	LimitType BudgetLimitType `json:"-"`
	// termination_policy_halt, middleware_halt
	Reason string    `json:"-"`
	Hook   HookPoint `json:"-"`
	// agent_error
	AgentError *AgentError `json:"-"`
	// context_error (issue #32)
	ContextError *ContextError `json:"-"`
	// sandbox_violation
	Violation *SandboxViolation `json:"-"`
	// unrecoverable_tool_error
	Tool  string `json:"-"`
	Error string `json:"-"`
	// stagnation_limit_reached
	Iterations uint32  `json:"-"`
	BestMetric float64 `json:"-"`
	// strategy_not_yet_implemented
	Strategy string `json:"-"`
	// plan_phase_failed (issue #70)
	PlanError *PlanPhaseError `json:"-"`
	// step_failed (issue #59, Q5)
	TaskIndex int    `json:"-"`
	Task      string `json:"-"`
	// configuration_error (issue #120): the underlying registry validation error
	// (UnresolvedHandleError or StrategyNotFoundError).
	ConfigError HarnessError `json:"-"`
	// tasks_blocked_by_failure (issue #126, decision A): the cascade partition.
	Completed  []uint32 `json:"-"`
	Blocked    []uint32 `json:"-"`
	FailedTask uint32   `json:"-"`
	// tool_error_loop (issue #137): the consecutive identical-argument
	// recoverable-error count at the 2*N hard stop. The offending tool name reuses
	// the Tool field above.
	ConsecutiveErrors uint32 `json:"-"`
	// output_schema_violation (issue #139): the resolved Schema (canonical compact
	// key-sorted JSON), the total Attempts (1 + max_retries), and the LastError
	// (the frozen validator error from the final attempt).
	Schema    string `json:"-"`
	Attempts  uint32 `json:"-"`
	LastError string `json:"-"`
}

// MarshalJSON serialises as a flat tagged object.
func (h HaltReason) MarshalJSON() ([]byte, error) {
	switch h.Kind {
	case HaltBudgetExceeded:
		return json.Marshal(struct {
			Kind      HaltReasonKind  `json:"kind"`
			LimitType BudgetLimitType `json:"limit_type"`
		}{h.Kind, h.LimitType})
	case HaltTerminationPolicyHalt:
		return json.Marshal(struct {
			Kind   HaltReasonKind `json:"kind"`
			Reason string         `json:"reason"`
		}{h.Kind, h.Reason})
	case HaltMiddlewareHalt:
		return json.Marshal(struct {
			Kind   HaltReasonKind `json:"kind"`
			Hook   HookPoint      `json:"hook"`
			Reason string         `json:"reason"`
		}{h.Kind, h.Hook, h.Reason})
	case HaltAgentError:
		return json.Marshal(struct {
			Kind  HaltReasonKind `json:"kind"`
			Error *AgentError    `json:"error"`
		}{h.Kind, h.AgentError})
	case HaltContextError:
		return json.Marshal(struct {
			Kind  HaltReasonKind `json:"kind"`
			Error *ContextError  `json:"error"`
		}{h.Kind, h.ContextError})
	case HaltSandboxViolation:
		return json.Marshal(struct {
			Kind      HaltReasonKind    `json:"kind"`
			Violation *SandboxViolation `json:"violation"`
		}{h.Kind, h.Violation})
	case HaltUnrecoverableToolError:
		return json.Marshal(struct {
			Kind  HaltReasonKind `json:"kind"`
			Tool  string         `json:"tool"`
			Error string         `json:"error"`
		}{h.Kind, h.Tool, h.Error})
	case HaltHumanHalted:
		return json.Marshal(struct {
			Kind HaltReasonKind `json:"kind"`
		}{h.Kind})
	case HaltStagnationLimitReached:
		return json.Marshal(struct {
			Kind       HaltReasonKind `json:"kind"`
			Iterations uint32         `json:"iterations"`
			BestMetric float64        `json:"best_metric"`
		}{h.Kind, h.Iterations, h.BestMetric})
	case HaltStrategyNotYetImplemented:
		return json.Marshal(struct {
			Kind     HaltReasonKind `json:"kind"`
			Strategy string         `json:"strategy"`
		}{h.Kind, h.Strategy})
	case HaltEmptyPlan:
		return json.Marshal(struct {
			Kind HaltReasonKind `json:"kind"`
		}{h.Kind})
	case HaltStepFailed:
		return json.Marshal(struct {
			Kind      HaltReasonKind `json:"kind"`
			TaskIndex int            `json:"task_index"`
			Task      string         `json:"task"`
			Reason    string         `json:"reason"`
		}{h.Kind, h.TaskIndex, h.Task, h.Reason})
	case HaltPlanPhaseFailed:
		return json.Marshal(struct {
			Kind  HaltReasonKind  `json:"kind"`
			Error *PlanPhaseError `json:"error"`
		}{h.Kind, h.PlanError})
	case HaltSelfVerifyExhausted:
		return json.Marshal(struct {
			Kind       HaltReasonKind `json:"kind"`
			Iterations uint32         `json:"iterations"`
			LastReason string         `json:"last_reason"`
		}{h.Kind, h.Iterations, h.Reason})
	case HaltSelfVerifyMisconfigured:
		return json.Marshal(struct {
			Kind   HaltReasonKind `json:"kind"`
			Reason string         `json:"reason"`
		}{h.Kind, h.Reason})
	case HaltRalphCompletionUnmet:
		return json.Marshal(struct {
			Kind       HaltReasonKind `json:"kind"`
			Iterations uint32         `json:"iterations"`
			LastReason string         `json:"last_reason"`
		}{h.Kind, h.Iterations, h.Reason})
	case HaltHillClimbingMisconfigured:
		return json.Marshal(struct {
			Kind   HaltReasonKind `json:"kind"`
			Reason string         `json:"reason"`
		}{h.Kind, h.Reason})
	case HaltConfigurationError:
		// The nested "error" is itself a "kind"-tagged HarnessError.
		return json.Marshal(struct {
			Kind  HaltReasonKind `json:"kind"`
			Error HarnessError   `json:"error"`
		}{h.Kind, h.ConfigError})
	case HaltTasksBlockedByFailure:
		completed := h.Completed
		if completed == nil {
			completed = []uint32{}
		}
		blocked := h.Blocked
		if blocked == nil {
			blocked = []uint32{}
		}
		return json.Marshal(struct {
			Kind       HaltReasonKind `json:"kind"`
			Completed  []uint32       `json:"completed"`
			Blocked    []uint32       `json:"blocked"`
			FailedTask uint32         `json:"failed_task"`
			Reason     string         `json:"reason"`
		}{h.Kind, completed, blocked, h.FailedTask, h.Reason})
	case HaltTaskGraphCycle:
		return json.Marshal(struct {
			Kind   HaltReasonKind `json:"kind"`
			Reason string         `json:"reason"`
		}{h.Kind, h.Reason})
	case HaltToolErrorLoop:
		return json.Marshal(struct {
			Kind              HaltReasonKind `json:"kind"`
			Tool              string         `json:"tool"`
			ConsecutiveErrors uint32         `json:"consecutive_errors"`
		}{h.Kind, h.Tool, h.ConsecutiveErrors})
	case HaltOutputSchemaViolation:
		return json.Marshal(struct {
			Kind      HaltReasonKind `json:"kind"`
			Schema    string         `json:"schema"`
			Attempts  uint32         `json:"attempts"`
			LastError string         `json:"last_error"`
		}{h.Kind, h.Schema, h.Attempts, h.LastError})
	default:
		return nil, fmt.Errorf("HaltReason: unknown kind %q", h.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (h *HaltReason) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind              HaltReasonKind    `json:"kind"`
		LimitType         BudgetLimitType   `json:"limit_type"`
		Reason            string            `json:"reason"`
		Hook              HookPoint         `json:"hook"`
		Error             json.RawMessage   `json:"error"`
		Violation         *SandboxViolation `json:"violation"`
		Tool              string            `json:"tool"`
		Iterations        uint32            `json:"iterations"`
		BestMetric        float64           `json:"best_metric"`
		Strategy          string            `json:"strategy"`
		TaskIndex         int               `json:"task_index"`
		Task              string            `json:"task"`
		LastReason        string            `json:"last_reason"`
		Completed         []uint32          `json:"completed"`
		Blocked           []uint32          `json:"blocked"`
		FailedTask        uint32            `json:"failed_task"`
		ConsecutiveErrors uint32            `json:"consecutive_errors"`
		Schema            string            `json:"schema"`
		Attempts          uint32            `json:"attempts"`
		LastError         string            `json:"last_error"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	h.Kind = probe.Kind
	h.LimitType = probe.LimitType
	h.Reason = probe.Reason
	h.Hook = probe.Hook
	h.Violation = probe.Violation
	h.Tool = probe.Tool
	h.Iterations = probe.Iterations
	h.BestMetric = probe.BestMetric
	h.Strategy = probe.Strategy
	h.TaskIndex = probe.TaskIndex
	h.Task = probe.Task
	h.Completed = probe.Completed
	h.Blocked = probe.Blocked
	h.FailedTask = probe.FailedTask
	h.ConsecutiveErrors = probe.ConsecutiveErrors
	h.Schema = probe.Schema
	h.Attempts = probe.Attempts
	h.LastError = probe.LastError

	switch probe.Kind {
	case HaltAgentError:
		if len(probe.Error) > 0 && string(probe.Error) != "null" {
			ae := &AgentError{}
			if err := ae.UnmarshalJSON(probe.Error); err != nil {
				return err
			}
			h.AgentError = ae
		}
	case HaltContextError:
		// "error" is a ContextError object here (issue #32).
		if len(probe.Error) > 0 && string(probe.Error) != "null" {
			ce := &ContextError{}
			if err := json.Unmarshal(probe.Error, ce); err != nil {
				return err
			}
			h.ContextError = ce
		}
	case HaltUnrecoverableToolError:
		// "error" is a string here, not an object — re-read as string.
		var v struct {
			Error string `json:"error"`
		}
		if err := json.Unmarshal(data, &v); err != nil {
			return err
		}
		h.Error = v.Error
	case HaltPlanPhaseFailed:
		// "error" is a PlanPhaseError object here (issue #70).
		if len(probe.Error) > 0 && string(probe.Error) != "null" {
			pe := &PlanPhaseError{}
			if err := json.Unmarshal(probe.Error, pe); err != nil {
				return err
			}
			h.PlanError = pe
		}
	case HaltConfigurationError:
		// "error" is a "kind"-tagged HarnessError object here (issue #120).
		if len(probe.Error) > 0 && string(probe.Error) != "null" {
			ce, err := UnmarshalHarnessError(probe.Error)
			if err != nil {
				return err
			}
			h.ConfigError = ce
		}
	case HaltSelfVerifyExhausted:
		// "last_reason" carries the verifier's final failure reason (#61).
		h.Reason = probe.LastReason
	case HaltRalphCompletionUnmet:
		// "last_reason" carries the final incompletion reason (#58).
		h.Reason = probe.LastReason
	}
	return nil
}

// RunResultKind discriminates RunResult variants.
type RunResultKind string

const (
	RunSuccess         RunResultKind = "success"
	RunFailure         RunResultKind = "failure"
	RunWaitingForHuman RunResultKind = "waiting_for_human"
	// RunEscalate — harness terminated cleanly due to a tool escalation signal
	// (issue #80). Carries the signal plus the preserved session state so the
	// caller can resume the original harness, instantiate a new one, or present
	// UI to the user.
	RunEscalate RunResultKind = "escalate"
	// RunConsult — worker paused mid-loop to consult a parent-spawned helper
	// (issue #114). Sibling of RunWaitingForHuman, but it stops at the
	// ORCHESTRATOR (via SubagentTool's A1 mediation), not the human. State
	// preserves the full PausedState with HumanRequest nil and the consult call
	// as the head of PendingToolCalls, so ResumeConsult continues the worker.
	// With no consult handlers registered, a standalone worker returns this
	// unchanged to its caller (R6 graceful degradation).
	RunConsult RunResultKind = "consult"
)

// RunResult is the outcome of Harness.Run / Harness.Resume.
type RunResult struct {
	Kind RunResultKind `json:"kind"`
	// success
	Output string `json:"-"`
	// failure
	Reason HaltReason `json:"-"`
	// success, failure
	SessionID SessionID      `json:"-"`
	Usage     AggregateUsage `json:"-"`
	Turns     uint32         `json:"-"`
	// SessionState is the post-run conversation history carried on success and
	// failure (issue #102). It holds the full SessionState.Messages the loop
	// produced — assistant tool-call turns and tool-result turns included — so an
	// in-process caller can resume losslessly via HarnessRunOptions.SessionState
	// without reconstructing history from Output. On the wire it is the Go analog
	// of Rust's #[serde(default)]: always emitted on success/failure, and
	// tolerant of absence on decode so old serialized RunResult blobs still parse.
	SessionState SessionState `json:"-"`
	// waiting_for_human, escalate (both carry State)
	State   *PausedState  `json:"-"`
	Request *HumanRequest `json:"-"`
	// escalate (issue #80)
	Signal *HarnessSignal `json:"-"`
	// consult (issue #114): the worker's ConsultRequest. State (above) carries
	// the paused worker; SessionID/Usage/Turns carry the pause snapshot.
	ConsultRequest *ConsultRequest `json:"-"`
}

// MarshalJSON serialises as a flat tagged object.
func (r RunResult) MarshalJSON() ([]byte, error) {
	switch r.Kind {
	case RunSuccess:
		return json.Marshal(struct {
			Kind         RunResultKind  `json:"kind"`
			Output       string         `json:"output"`
			SessionID    SessionID      `json:"session_id"`
			Usage        AggregateUsage `json:"usage"`
			Turns        uint32         `json:"turns"`
			SessionState SessionState   `json:"session_state"`
		}{r.Kind, r.Output, r.SessionID, r.Usage, r.Turns, r.SessionState})
	case RunFailure:
		return json.Marshal(struct {
			Kind         RunResultKind  `json:"kind"`
			Reason       HaltReason     `json:"reason"`
			SessionID    SessionID      `json:"session_id"`
			Usage        AggregateUsage `json:"usage"`
			Turns        uint32         `json:"turns"`
			SessionState SessionState   `json:"session_state"`
		}{r.Kind, r.Reason, r.SessionID, r.Usage, r.Turns, r.SessionState})
	case RunWaitingForHuman:
		return json.Marshal(struct {
			Kind    RunResultKind `json:"kind"`
			State   *PausedState  `json:"state"`
			Request *HumanRequest `json:"request"`
		}{r.Kind, r.State, r.Request})
	case RunEscalate:
		// Nested tagged union: signal is itself a "kind"-tagged HarnessSignal.
		return json.Marshal(struct {
			Kind      RunResultKind  `json:"kind"`
			Signal    *HarnessSignal `json:"signal"`
			State     *PausedState   `json:"state"`
			SessionID SessionID      `json:"session_id"`
			Usage     AggregateUsage `json:"usage"`
			Turns     uint32         `json:"turns"`
		}{r.Kind, r.Signal, r.State, r.SessionID, r.Usage, r.Turns})
	case RunConsult:
		return json.Marshal(struct {
			Kind      RunResultKind   `json:"kind"`
			Request   *ConsultRequest `json:"request"`
			State     *PausedState    `json:"state"`
			SessionID SessionID       `json:"session_id"`
			Usage     AggregateUsage  `json:"usage"`
			Turns     uint32          `json:"turns"`
		}{r.Kind, r.ConsultRequest, r.State, r.SessionID, r.Usage, r.Turns})
	default:
		return nil, fmt.Errorf("RunResult: unknown kind %q", r.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (r *RunResult) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind      RunResultKind  `json:"kind"`
		Output    string         `json:"output"`
		Reason    HaltReason     `json:"reason"`
		SessionID SessionID      `json:"session_id"`
		Usage     AggregateUsage `json:"usage"`
		Turns     uint32         `json:"turns"`
		// SessionState is tolerant of absence (issue #102): old serialized
		// RunResult blobs that predate this field decode to the zero value, the
		// Go analog of Rust's #[serde(default)].
		SessionState SessionState   `json:"session_state"`
		State        *PausedState   `json:"state"`
		Request      *HumanRequest  `json:"request"`
		Signal       *HarnessSignal `json:"signal"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	r.Kind = probe.Kind
	r.Output = probe.Output
	r.Reason = probe.Reason
	r.SessionID = probe.SessionID
	r.Usage = probe.Usage
	r.Turns = probe.Turns
	r.SessionState = probe.SessionState
	r.State = probe.State
	r.Request = probe.Request
	r.Signal = probe.Signal
	// consult (issue #114): the "request" key here is a ConsultRequest, not a
	// HumanRequest. Re-read it into the dedicated field for that variant.
	if probe.Kind == RunConsult {
		var cr struct {
			Request *ConsultRequest `json:"request"`
		}
		if err := json.Unmarshal(data, &cr); err != nil {
			return err
		}
		r.ConsultRequest = cr.Request
		r.Request = nil
	}
	return nil
}

// ============================================================================
// HarnessRunOptions
// ============================================================================

// HarnessRunOptions is the input to Harness.Run.
type HarnessRunOptions struct {
	Task         Task
	OnStream     StreamSink
	SessionState *SessionState
}

// NewHarnessRunOptions builds run options with a default empty session.
func NewHarnessRunOptions(t Task) HarnessRunOptions {
	return HarnessRunOptions{Task: t}
}

// ============================================================================
// The Harness interface
// ============================================================================

// Harness drives the agent loop.
type Harness interface {
	Run(ctx context.Context, options HarnessRunOptions) RunResult
	Resume(ctx context.Context, state PausedState, response HumanResponse, onStream StreamSink) RunResult
	// ResumeConsult resumes a worker paused by RunResult.Consult (issue #114).
	// The resume seam parallel to Resume: it injects the ConsultResponse as the
	// tool RESULT of the head pending consult call, dispatches any remaining
	// pending calls, and resumes the loop. SubagentTool calls this internally
	// during A1 mediation; the orchestrator model is never involved.
	ResumeConsult(ctx context.Context, state PausedState, response ConsultResponse, onStream StreamSink) RunResult
}

// ============================================================================
// StandardHarness — the canonical implementation
// ============================================================================

// HarnessConfig is the bag of components injected into StandardHarness.
// Middleware and Observability are optional; the rest are required.
type HarnessConfig struct {
	// Agent is the default agent the loop drives.
	//
	// Superseded by ExecutionRegistry (issue #120): per-node AgentRef handles
	// resolved via Registry replace this single collaborator. Kept this slice
	// (Option B, additive scope) so existing callers and executor consumption
	// sites stay byte-identical; physical removal + executor migration to registry
	// resolution lands in #124.
	Agent             Agent
	ToolRegistry      ToolRegistry
	Sandbox           SandboxProvider
	ContextManager    ContextManager
	TerminationPolicy TerminationPolicy
	Middleware        MiddlewareChain // optional
	// Observability is the loop's span-emission seam. Optional; when nil
	// the loop emits nothing. The token→USD pricing used to stamp cost on
	// turn spans is held by the observer (see HarnessObserver.CostFor) — it
	// cannot live here because PricingTable is defined in the
	// `observability` package, which the root package cannot import (cycle).
	// The builder in `observability` wires pricing into the observer.
	Observability HarnessObserver // optional

	// CompactionVerifier is the post-compaction verifier (issue #29/#46). The
	// loop runs it after each compaction turn and retries up to
	// MaxCompactionAttempts before accepting a failing summary. Optional; when
	// nil every summary is treated as passing (no verification gate). The
	// builder defaults this to contextmgr.NewKeyTermVerifier().
	CompactionVerifier CompactionVerifier // optional
	// MaxCompactionAttempts bounds compaction-summary attempts before accepting
	// a failing summary anyway (issue #46). Clamped to a minimum of 1 by the
	// loop. The builder defaults this to 2 (mirrors CompactionConfig).
	MaxCompactionAttempts uint32

	// Hooks is the lifecycle hook chain (issue #69). Optional; when nil no hooks
	// fire. Of the 17 events, only Stop is wired into the ReAct loop in Go (its
	// machinery exists) — Stop replaces the old ForceAnotherTurn / completion
	// re-prompt path. See hooks.go for the full event catalogue and which events
	// are defined-and-unit-tested but not yet loop-wired.
	Hooks HookChain // optional

	// MaxStopBlocks caps how many consecutive Stop-hook blocks the loop honours
	// in a single Run before terminating anyway (issue #69). The counter is
	// per-run (resets each Run call). Zero is treated as the default of 8.
	MaxStopBlocks uint32

	// ErrorLoopThreshold is N, the consecutive-recoverable-tool-error breaker
	// trigger (issue #137). The ReAct turn loop tracks, per tool name, the run of
	// consecutive recoverable ToolOutput.Error results carrying identical
	// arguments. At N identical-argument errors it injects ONE corrective message
	// (the enrichToolError schema+hint); at 2*N it STOPS looping and resolves the
	// node's BudgetExhaustedBehavior WITHOUT burning the remaining budget,
	// terminating with HaltToolErrorLoop (never HaltBudgetExceeded). The 2x
	// hard-stop multiplier is FIXED.
	//
	// Pointer sentinel so a default-constructed config gets the cross-language
	// default of 3 while an EXPLICIT 0 disables the breaker (matching the Rust /
	// TS / Python builder default + explicit-0-disables contract). Read it via
	// effectiveErrorLoopThreshold: nil -> 3 (the default), *=0 -> disabled, *=n ->
	// n. Mirrors the house idiom for optional caps (e.g. BudgetLimits.MaxTurns).
	ErrorLoopThreshold *uint32

	// EnforceOutputSchemas is the output-schema enforcement MIGRATION GATE (issue
	// #139) — NOT a permanent feature flag. When true, a ReactConfig leaf carrying
	// Output != nil has its resolved output schema DELIVERED to the model
	// (directive seed + the ModelParams.OutputSchema constrained-decoding channel)
	// and its terminal FinalResponse VALIDATED against that schema; a validation
	// failure feeds the frozen error back and retries up to
	// OutputSchemaMaxRetries extra turns (within budget), then terminates with
	// HaltOutputSchemaViolation. Enforcement is UNIFORM — it applies to
	// structured-slot leaves (plan / worker / propose) too, and is additive to and
	// earlier than any downstream combinator validation.
	//
	// Default false (OFF) is NON-NEGOTIABLE: it keeps every existing replay
	// fixture byte-for-byte green during migration. When OFF, ReactConfig.Run
	// behaves EXACTLY as before — no resolve, no delivery, no validation.
	EnforceOutputSchemas bool

	// OutputSchemaMaxRetries is the N extra terminal-validation retry turns
	// granted when EnforceOutputSchemas is ON and a terminal fails output-schema
	// validation (issue #139). Total attempts = 1 + N. Retried turns COUNT against
	// the turn budget; a retry that would exceed the budget surfaces the budget
	// terminal instead of HaltOutputSchemaViolation (budget-cap-wins precedence).
	//
	// Pointer sentinel (mirrors ErrorLoopThreshold, #137): a nil field on a
	// default-constructed config yields the cross-language default of 2; an
	// EXPLICIT 0 is honored (0 extra turns ⇒ the first invalid terminal is a
	// violation). Read it via effectiveOutputSchemaMaxRetries.
	OutputSchemaMaxRetries *uint32

	// MaxResets is the outer-loop context-window reset cap for the Ralph loop
	// strategy (issue #58, B3): the maximum number of context windows the
	// multi-context-window continuation loop runs before halting with
	// HaltRalphCompletionUnmet when tasks are still incomplete. Independent of
	// MaxTurns (which bounds turns WITHIN a single window). Zero is treated as
	// the default of 3.
	MaxResets uint32

	// VcsProvider is the optional VCS seam for the Ralph loop strategy (issue #58
	// v2, B4). When non-nil, Ralph's per-window reload phase ALSO calls
	// VcsProvider.Log and injects the output into the fresh context window's seed
	// as a delimited "Recent VCS history:" section — alongside the reloaded
	// .spore/ state. When nil (the default) the git-log section is omitted and
	// Ralph behaves exactly like v1 (the B4→nil decision). This is a
	// consumer-side interface (go/CONVENTIONS.md): GitVcsProvider is the real
	// impl, FixtureVcsProvider the hermetic test double.
	VcsProvider VcsProvider // optional

	// #124: the legacy single-collaborator fields PlannerAgent / Verifier /
	// EvaluatorAgent are GONE. The SelfVerifying verifier resolves from the
	// Registry verifiers map by the SelfVerifying evaluator key (Q1a); the
	// PlanExecute plan-phase agent is the plan child's leaf ReactConfig.Agent
	// (Q1 — the planner_agent concept is dropped); the SelfVerifying evaluate
	// agent defaults to the inner worker's resolved agent (Q1c). A default
	// Verifier passed in via the (#124-folded) registry seam resolves under the
	// empty key — see NewStandardHarness.

	// RunStore is the durable per-run structured-state seam (issue #59, Q4).
	// The PlanExecute plan phase writes the plan artifact under
	// PlanExecuteExtrasKey and the execute loop writes the parsed task list
	// under TaskListExtrasKey, both keyed by SessionID (#76 — the single source
	// of truth; no SessionState.Extras mirror). Optional; when nil the durable
	// write is skipped.
	// This is a consumer-side interface (go/CONVENTIONS.md): the concrete
	// storage.StorageProvider.Run() satisfies it structurally without the
	// sporecore package importing the storage package (which would be a cycle).
	RunStore RunStore // optional

	// ProjectNamespace is the STABLE durable namespace (#142) the DURABLE
	// RunStore call sites — the task_list / plan checkpoint (persistTaskList,
	// LoadTaskList, captureAndPersistPlan, reconcileDeepResume), the threaded-in
	// catalogue tools' task_list, and the Ralph progress / feature-list
	// checkpoint — key by, instead of the per-window SessionID. It is a
	// storage.ProjectID.Namespace() value carried as a SessionID (the root
	// sporecore package cannot import storage; the project id is projected onto
	// the existing SessionID axis — namespace-reuse, NOT interface widening).
	// Derived from the sandbox workspace root (decision 5), it stays constant
	// across Ralph window resets (NewSessionID() per window) and process restarts,
	// so a window reset re-reads the prior window's durable artifacts. Empty (the
	// default) falls back to the per-window SessionID, preserving today's
	// single-process behaviour byte-for-byte.
	ProjectNamespace SessionID // optional

	// #124: the legacy MetricEvaluator single-collaborator field is GONE. The
	// HillClimbing metric evaluator resolves from the Registry's SIXTH
	// metricEvaluators map by the HillClimbing evaluator key (Q2); a default
	// MetricEvaluator passed via the (#124-folded) registry seam resolves under
	// the empty key — see NewStandardHarness.

	// CatalogueRegistry holds the catalogue tools accumulated via
	// HarnessBuilder.Tool / Tools (issue #81), folded into a populated
	// *StandardToolRegistry at build time. When non-nil, the run loop bridges it
	// per-run via effectiveToolRegistry — threading the run's SessionID + the
	// ToolRunStore / ToolMemoryStore seams into every tool dispatch through
	// SetToolContext — and uses it instead of ToolRegistry (which stays the
	// harness-loop seam for custom slim registries). Nil (the default) preserves
	// the ToolRegistry-only path byte-for-byte. Unlike Rust, Go has no separate
	// RealToolRegistry bridge type: *StandardToolRegistry IS the harness
	// ToolRegistry, so the per-run "bridge" is the same registry with its
	// ToolContext re-injected each run.
	CatalogueRegistry *StandardToolRegistry // optional

	// ToolsetCatalogues holds the per-toolset catalogues (Issue 2: per-node
	// toolset scoping), keyed by the non-empty ToolsetRef handle a leaf carries.
	// Populated by HarnessBuilder.ToolsetTools — each value is a populated
	// *StandardToolRegistry folded from that key's catalogue tools. At dispatch
	// effectiveToolRegistry re-injects the matching catalogue's ToolContext with
	// the run's SessionID + storage (same wiring as the global CatalogueRegistry),
	// so a node with a non-empty toolset handle dispatches ONLY its own tools. A
	// leaf with an EMPTY handle ("") — or a non-empty handle with no entry here —
	// falls back to the global CatalogueRegistry / ToolRegistry seam (back-compat
	// with examples 01–11 that use .Tools()). Mirrors Rust
	// HarnessConfig.toolset_catalogues. Nil/empty (the default) preserves today's
	// single-catalogue behaviour byte-for-byte.
	ToolsetCatalogues map[string]*StandardToolRegistry // optional

	// ToolRunStore is the per-run structured-state seam threaded into catalogue
	// tools' ToolContext (issue #75). Optional; nil means catalogue tools persist
	// nothing across processes (no-op store). The builder defaults this to an
	// in-memory store when catalogue tools are present and no storage was wired.
	ToolRunStore ToolRunStore // optional
	// ToolMemoryStore is the episodic-memory seam threaded into catalogue tools'
	// ToolContext (issue #78/#82). Optional; nil is the never-null library default
	// (a no-op for memory-aware tools).
	ToolMemoryStore ToolMemoryStore // optional

	// SystemPrompt is the operating system prompt prepended to each turn's
	// assembled context when the context manager renders none (issue #91). See
	// HarnessBuilder.SystemPrompt. Empty (the default) preserves today's behaviour
	// byte-for-byte.
	SystemPrompt string // optional

	// Guides (skills/playbooks/domain knowledge) injected structurally into
	// every turn's assembled context via the ContextSources seam (issue #115 /
	// SC-26 / #9). The harness clones these into ContextSources.Guides each turn;
	// the production StandardCompactionAdapter renders them into the leading
	// System block, not as ad-hoc User messages. Empty (the default) preserves
	// today's behaviour byte-for-byte. See HarnessBuilder.Guide.
	Guides []Guide // optional

	// Skills is the optional skill catalog (issue #115 / SC-26). When set, the
	// harness injects its manifest + active skill bodies into
	// ContextSources.Guides each turn (progressive disclosure) and the load_skill
	// tool activates skills against its shared active set. nil (the default)
	// means no skills. See HarnessBuilder.Skills.
	Skills *SkillCatalog // optional

	// Memory is the optional memory source (issue #163 / SC-26 follow-up). When
	// set, the harness queries the provider each turn (via buildContextSources)
	// and injects the returned items into ContextSources.Memory, rendered into the
	// structural System block alongside guides + skills. nil (the default) leaves
	// memory empty, byte-identical to the pre-#163 path. See HarnessBuilder.Memory
	// and MemoryConfig.
	Memory *MemoryConfig // optional

	// ModelParams are the authoritative per-run model sampling/decoding
	// parameters (issue #93). The harness replaces each tool-requesting turn's
	// Context.Params with this value UNCONDITIONALLY (builder params win) right
	// before the request is built, so the configured params reach every agent
	// turn that requests tools — the ReAct loop, the PlanExecute plan phase, the
	// execute sub-loop, and the streaming path alike. The internal
	// compaction/summarization turn is intentionally left on defaults. See
	// HarnessBuilder.WithModelParams. The zero value (the default) preserves
	// today's behaviour byte-for-byte.
	ModelParams ModelParams // optional

	// SessionStore is the opt-in conversation-history persistence seam (issue
	// #102). When AutoPersistSessions is true the run loop:
	//   - auto-LOADS the prior SessionState for the run's SessionID from this
	//     store at the start of Run() (ReAct / SelfVerifying only — Ralph /
	//     HillClimbing discard incoming state by design), so a caller resumes by
	//     reusing the same SessionID instead of threading SessionState; an
	//     explicit HarnessRunOptions.SessionState always wins (the load is
	//     skipped), and
	//   - auto-PERSISTS the post-run SessionState back to the store at the
	//     terminal seam (one write per Run()/Resume()).
	// Optional; nil is the never-null library default (a no-op store, so the loop
	// never null-checks). This is a consumer-side interface (go/CONVENTIONS.md):
	// a *storage.StorageProvider's Session() store satisfies it structurally
	// without sporecore importing the storage package (which would be a cycle).
	SessionStore SessionStore // optional

	// AutoPersistSessions opts this harness into the issue #102 session-store
	// auto-load + auto-persist contract above. Per-harness builder config (NOT a
	// per-run flag). Defaults to false: when false there is ZERO session-store
	// I/O and the message flow + replay outcomes are byte-for-byte identical to
	// today's.
	AutoPersistSessions bool

	// PromptToolCallFlag is the shared session-scoped flag for the adaptive
	// prompt-based tool-calling fallback (#111). The conversational preset wraps
	// the agent's model in an AdaptiveToolCallModelInterface over THIS SAME
	// pointer and sets it here; the run loop resets it to false at each window
	// start and flips it to true when it detects a prose response where a tool
	// call was expected (DetectProseResponse). Nil (the default for
	// non-conversational construction) disables the escalation seam entirely —
	// the loop never reads or writes it.
	PromptToolCallFlag *atomic.Bool // optional (set by the conversational preset)

	// ConsultHandlers is the per-kind consult-handler map (issue #114), keyed by
	// ConsultRequest.Kind. Per go/CONVENTIONS.md this is a struct FIELD, not a
	// builder setter — matching the established PlannerAgent / Verifier /
	// VcsProvider divergence. Empty (the default) means consults are NOT mediated:
	// a worker that pauses with RunResult.Consult surfaces it unchanged to its
	// caller (R6 graceful degradation), and existing callers are unaffected
	// byte-for-byte (R9). When populated, the orchestrator passes a copy to its
	// SubagentTool (via SubagentTool.WithConsultHandlers); SubagentTool runs the
	// matching entry's helper harness deterministically (the A1 mediation seam) —
	// the orchestrator model is never involved.
	ConsultHandlers map[string]ConsultHandlerEntry // optional

	// Registry is the runtime resolver for the serializable strategy handles
	// (AgentRef/ToolsetRef/SchemaRef) and StrategyRef.Custom keys held by a
	// task's LoopStrategy tree (issue #120). StandardHarness.Run calls
	// Registry.Validate at entry, so an unresolved handle is a STARTUP error
	// before the first turn. This slice is ADDITIVE (Option B): the registry
	// coexists with the superseded single-collaborator fields and is not yet read
	// by the run bodies (that lands in #123/#124). The zero value is an empty
	// registry, in which case startup validation is skipped (legacy callers stay
	// byte-identical). Set it directly, or via the WithRegistry* / RegisterStrategy
	// convenience methods on HarnessConfig.
	Registry ExecutionRegistry // optional

	// EscalationMode is the HITL-vs-AFK escalation knob (issue #120, PRD goal
	// #7): whether budget escalation surfaces to a human or proceeds
	// autonomously. STORED only this slice — consumed in #130. The zero value
	// (empty Kind) is treated as EscalationSurfaceToHuman by EffectiveEscalationMode;
	// set it explicitly to select AFK / autonomous behaviour.
	EscalationMode EscalationMode // optional (defaults to surface_to_human)
}

// EffectiveEscalationMode returns the configured EscalationMode, defaulting the
// zero value (empty Kind) to EscalationSurfaceToHuman — the explicit default
// (EscalationMode itself has no baked-in default, mirroring the budget-types
// discipline). Issue #120; consumed in #130.
func (c HarnessConfig) EffectiveEscalationMode() EscalationMode {
	if c.EscalationMode.Kind == "" {
		return SurfaceToHumanEscalation()
	}
	return c.EscalationMode
}

// WithRegistry replaces the whole ExecutionRegistry (issue #120) and returns the
// updated config (value-receiver fluent style). The registry resolves a task's
// serializable strategy handles and StrategyRef.Custom keys at run entry.
func (c HarnessConfig) WithRegistry(registry ExecutionRegistry) HarnessConfig {
	c.Registry = registry
	return c
}

// WithRegistryAgent registers a named agent in the ExecutionRegistry (issue
// #120, per-key convenience over WithRegistry) and returns the updated config.
func (c HarnessConfig) WithRegistryAgent(key string, agent Agent) HarnessConfig {
	c.Registry = c.Registry.intoBuilder().Agent(key, agent).Build()
	return c
}

// WithRegistryToolset registers a named toolset in the ExecutionRegistry (issue
// #120) and returns the updated config.
func (c HarnessConfig) WithRegistryToolset(key string, toolset ToolRegistry) HarnessConfig {
	c.Registry = c.Registry.intoBuilder().Toolset(key, toolset).Build()
	return c
}

// WithRegistrySchema registers a named JSON schema in the ExecutionRegistry
// (issue #120) and returns the updated config.
func (c HarnessConfig) WithRegistrySchema(key string, schema json.RawMessage) HarnessConfig {
	c.Registry = c.Registry.intoBuilder().Schema(key, schema).Build()
	return c
}

// WithRegistryVerifier registers a named verifier in the ExecutionRegistry
// (issue #120) and returns the updated config.
func (c HarnessConfig) WithRegistryVerifier(key string, verifier Verifier) HarnessConfig {
	c.Registry = c.Registry.intoBuilder().Verifier(key, verifier).Build()
	return c
}

// WithRegistryMetricEvaluator registers a named metric evaluator in the
// ExecutionRegistry's SIXTH map (#124, Q2) and returns the updated config. The
// key matches the HillClimbingConfig.Evaluator string on the wire.
func (c HarnessConfig) WithRegistryMetricEvaluator(key string, evaluator MetricEvaluator) HarnessConfig {
	c.Registry = c.Registry.intoBuilder().MetricEvaluator(key, evaluator).Build()
	return c
}

// RegisterStrategy registers a custom strategy in the ExecutionRegistry under
// key (issue #120). Resolvable later via a StrategyRef.Custom(key). Returns the
// updated config.
func (c HarnessConfig) RegisterStrategy(key string, strategy RunStrategy) HarnessConfig {
	c.Registry = c.Registry.intoBuilder().RegisterStrategy(key, strategy).Build()
	return c
}

// WithEscalationMode selects the HITL-vs-AFK escalation mode (issue #120, PRD
// goal #7) and returns the updated config.
func (c HarnessConfig) WithEscalationMode(mode EscalationMode) HarnessConfig {
	c.EscalationMode = mode
	return c
}

// SessionStore is the consumer-side view of the pause/resume lifecycle store
// the harness reads/writes for opt-in conversation-history threading (issue
// #102). It mirrors the read/write methods of storage.SessionStore so a
// *storage.StorageProvider's Session() store can be dropped straight into
// HarnessConfig.SessionStore without an import cycle (storage imports
// sporecore). State is the opaque *PausedState keyed by SessionID; found=false
// means the lookup hit nothing.
type SessionStore interface {
	GetSession(ctx context.Context, id SessionID) (state *PausedState, found bool, err error)
	PutSession(ctx context.Context, id SessionID, state *PausedState) error
}

// effectiveSessionStore returns the configured SessionStore or a no-op so the
// loop never null-checks (the never-null library default — D8).
func (c HarnessConfig) effectiveSessionStore() SessionStore {
	if c.SessionStore == nil {
		return noopSessionStore{}
	}
	return c.SessionStore
}

// noopSessionStore is the never-null default: reads find nothing, writes
// discard. Used when no SessionStore was wired so the loop can call the store
// unconditionally.
type noopSessionStore struct{}

func (noopSessionStore) GetSession(context.Context, SessionID) (*PausedState, bool, error) {
	return nil, false, nil
}
func (noopSessionStore) PutSession(context.Context, SessionID, *PausedState) error { return nil }

// RunStore is the consumer-side view of the per-run structured-state store the
// PlanExecute execute loop writes through (issue #59, Q4). It mirrors the Put
// method of storage.RunStore so a *storage.StorageProvider's Run() store can be
// dropped straight into HarnessConfig.RunStore without an import cycle. Values
// are opaque JSON blobs keyed by (SessionID, key); the store never knows the
// schema — the harness owns serialization.
type RunStore interface {
	// Get returns the stored value and found=false when absent (#124 deep-resume
	// reads the durable task-list checkpoint through this seam). Mirrors the
	// storage.RunStore Get.
	Get(ctx context.Context, sessionID SessionID, key string) (value json.RawMessage, found bool, err error)
	Put(ctx context.Context, sessionID SessionID, key string, value json.RawMessage) error
}

// effectiveMaxStopBlocks returns the Stop-block cap, defaulting 0 to 8.
func (c HarnessConfig) effectiveMaxStopBlocks() uint32 {
	if c.MaxStopBlocks == 0 {
		return 8
	}
	return c.MaxStopBlocks
}

// effectiveMaxResets returns the Ralph outer-loop reset cap, defaulting 0 to 3
// (issue #58, B3).
func (c HarnessConfig) effectiveMaxResets() uint32 {
	if c.MaxResets == 0 {
		return 3
	}
	return c.MaxResets
}

// effectiveErrorLoopThreshold returns N, the consecutive-recoverable-tool-error
// breaker trigger (issue #137). A nil ErrorLoopThreshold (a default-constructed
// config) yields the cross-language default of 3; a non-nil pointer is honored
// verbatim, so an EXPLICIT 0 disables the breaker. This is the read-time default
// pattern of effectiveMaxStopBlocks / effectiveMaxResets, extended with a pointer
// sentinel so explicit-0 is distinguishable from unset (matching the Rust / TS /
// Python builder-default + explicit-0-disables contract).
func (c HarnessConfig) effectiveErrorLoopThreshold() uint32 {
	if c.ErrorLoopThreshold == nil {
		return 3
	}
	return *c.ErrorLoopThreshold
}

// effectiveOutputSchemaMaxRetries returns N, the extra terminal-validation retry
// turns granted under output-schema enforcement (issue #139; total attempts =
// 1 + N). A nil OutputSchemaMaxRetries (a default-constructed config) yields the
// cross-language default of 2; a non-nil pointer is honored verbatim, so an
// EXPLICIT 0 means the first invalid terminal is a violation. Mirrors the
// effectiveErrorLoopThreshold pointer-sentinel idiom (#137).
func (c HarnessConfig) effectiveOutputSchemaMaxRetries() uint32 {
	if c.OutputSchemaMaxRetries == nil {
		return 2
	}
	return *c.OutputSchemaMaxRetries
}

// durableNamespace returns the key axis for DURABLE artifacts (#142): the pinned
// stable ProjectNamespace when set, otherwise the per-window sessionID. The
// sessionID param is intentionally retained (the namespace-reuse seam) so the
// durable call sites keep a stable signature and ephemeral session-keyed state
// elsewhere is unaffected; when no project namespace is wired the durable sites
// behave exactly as before (keyed by the session id).
func (c HarnessConfig) durableNamespace(sessionID SessionID) SessionID {
	if c.ProjectNamespace != "" {
		return c.ProjectNamespace
	}
	return sessionID
}

// StandardHarness is the canonical Harness implementation.
type StandardHarness struct {
	config HarnessConfig
	// observedWrites accumulates the harness-OBSERVED write/edit tool-call paths
	// for the current PlanExecute execute step (#126, AC2). The dispatch seam
	// (observeWriteCall, in the ReAct turn loop) appends here as write_file /
	// edit_file calls are ACTUALLY dispatched; the DAG executor drains it
	// (TakeObservedWrites) on task completion to build a StepLedgerEntry. Never a
	// model-self-reported field.
	observedWrites   []string
	observedWritesMu sync.Mutex
}

// NewStandardHarness constructs a StandardHarness.
//
// Ralph completion mechanism (issue #58, B1): at construction a Stop hook is
// registered that drives multi-context-window continuation off
// .spore/progress.json. Registration is harmless for non-Ralph runs — the hook
// only BLOCKS when a progress file is PRESENT and reports incomplete tasks; when
// the file is absent it returns Continue and does not interfere with ReAct /
// PlanExecute / SelfVerifying runs.
func NewStandardHarness(c HarnessConfig) *StandardHarness {
	// #142: the STABLE project namespace (HarnessConfig.ProjectNamespace) is
	// supplied by the storage-aware layer that can reach the derivation algorithm
	// (the example / a builder ProjectID(...) setter derives storage.ProjectID
	// from the sandbox workspace root and passes its .Namespace()). The root
	// sporecore package cannot import storage (a cycle), so it does NOT derive the
	// namespace here. When ProjectNamespace is empty the durable sites fall back
	// to the per-window session id — today's single-process behaviour.
	if c.Hooks == nil {
		c.Hooks = NewStandardHookChain()
	}
	// #124: the per-strategy single-collaborator fields are gone — collaborator
	// resolution now goes through the ExecutionRegistry. Fold the default
	// collaborators (the config's Agent / ToolRegistry) into the registry under
	// the DEFAULT empty-string handle so a bare ReactConfig leaf (empty
	// AgentRef/ToolsetRef) and a bare SelfVerifying/HillClimbing whose evaluator
	// key is empty resolve to them. Explicitly-registered handles win (fill-only).
	// This keeps NewStandardHarness's signature stable: callers still pass Agent /
	// ToolRegistry on the config, and the recursive executor resolves them via the
	// registry. A nil Agent (scaffold-only configs) is left unfolded.
	if c.Agent != nil {
		// A real harness: fold the default agent + toolset + a default empty-key
		// output schema (so a structured slot's leaf carrying output SchemaRef("")
		// resolves under A.5). A nil Agent (scaffold-only config) is left unfolded
		// so its registry stays empty and startup validation is skipped.
		rb := c.Registry.intoBuilder().
			fillDefaultAgent(c.Agent).
			fillDefaultToolset(c.ToolRegistry).
			fillDefaultSchema()
		// Issue 2: a leaf carrying a non-empty toolset handle must RESOLVE against
		// the registry (Validate runs checkToolset at run entry). For every per-key
		// catalogue wired via HarnessBuilder.ToolsetTools, ensure the registry has a
		// presence entry under that handle so Validate passes WITHOUT the caller
		// manually registering a placeholder. The registry VALUE is never dispatched
		// (dispatch goes through ToolsetCatalogues), so an empty *StandardToolRegistry
		// is sufficient; an explicitly-registered toolset under the same key wins.
		// Mirrors Rust build_config's fill_toolset loop.
		for key := range c.ToolsetCatalogues {
			rb = rb.fillToolset(key, NewStandardToolRegistry())
		}
		c.Registry = rb.Build()
	}
	// Best-effort registration: a Stop hook can only fail registration on an
	// event-class mismatch, which never applies to a sync Stop hook, so the
	// error is intentionally ignored. #142: the hook now reads the Ralph
	// checkpoint from the durable project store (keyed by the project namespace),
	// not the sandbox filesystem.
	_ = c.Hooks.Register(newRalphStopHook(c.RunStore, c.ProjectNamespace))
	return &StandardHarness{config: c}
}

func emit(sink StreamSink, event HarnessStreamEvent) {
	if sink != nil {
		sink(event)
	}
}

// runStreamingTurn executes one user-facing turn (issue #103). When a stream
// sink is attached, it drives the turn through TurnStreamingOrDelegate with an
// adapter that maps each raw model StreamEvent into harness StreamEvents (via
// mapModelStreamEvent, threading turnStreamState) and forwards them to onStream
// in arrival order. When no sink is attached it uses plain Agent.Turn so the
// baseline RunResult is byte-identical (back-compat). Either way the returned
// TurnResult is classified by the shared agent logic.
//
// Unlike the Rust reference, the Go StreamSink is an ordinary closure (no
// 'static / Send+Sync constraints), so the adapter forwards mapped events
// synchronously as they arrive rather than buffering and flushing after the
// turn. Ordering is preserved: TurnStart (emitted by the caller) → deltas →
// TurnEnd → coarse events.
func (h *StandardHarness) runStreamingTurn(ctx context.Context, agent Agent, c Context, onStream StreamSink) TurnResult {
	if onStream == nil {
		return agent.Turn(ctx, c)
	}
	state := newTurnStreamState()
	adapter := func(ev StreamEvent) {
		for _, mapped := range mapModelStreamEvent(ev, state) {
			emit(onStream, mapped)
		}
	}
	return TurnStreamingOrDelegate(ctx, agent, c, adapter)
}

// budgetExceeded returns the BudgetLimitType that just tripped, if any.
func budgetExceeded(b BudgetLimits, used BudgetSnapshot, startedAt time.Time) (BudgetLimitType, bool) {
	if b.MaxTurns != nil && used.Turns >= *b.MaxTurns {
		return BudgetLimitTurns, true
	}
	if b.MaxInputTokens != nil && used.InputTokens > uint64(*b.MaxInputTokens) {
		return BudgetLimitInputTokens, true
	}
	if b.MaxOutputTokens != nil && used.OutputTokens > uint64(*b.MaxOutputTokens) {
		return BudgetLimitOutputTokens, true
	}
	if b.MaxWallTime != nil && time.Since(startedAt) >= *b.MaxWallTime {
		return BudgetLimitWallTime, true
	}
	if b.MaxCostUSD != nil && used.CostUSD > *b.MaxCostUSD {
		return BudgetLimitCostUSD, true
	}
	return "", false
}

// Run executes a task to completion (or pause).
//
// Issue #102: Run is the thin auto-persist seam. It delegates the loop to
// runInner, then — when AutoPersistSessions is enabled — writes the terminal
// run state to the SessionStore (one write per Run, at the same terminal point
// as the observability flush). When disabled (the default) it is byte-for-byte
// the prior behaviour: runInner does no session-store I/O and the persist step
// returns immediately.
func (h *StandardHarness) Run(ctx context.Context, options HarnessRunOptions) RunResult {
	result := h.runInner(ctx, options)
	h.autoPersistTerminal(ctx, &result)
	return result
}

// runInner is the strategy dispatch (the body of the former Run). It performs
// the issue #102 auto-LOAD before dispatching: when AutoPersistSessions is on,
// no explicit SessionState was provided, and the strategy seeds incoming state
// (ReAct / SelfVerifying — Ralph / HillClimbing discard it by design, D7), it
// loads the prior session for this SessionID from the SessionStore so a caller
// can resume by id. An explicit HarnessRunOptions.SessionState always wins (the
// load is skipped, D5). A load failure is swallowed-and-logged: the run starts
// fresh (D8).
func (h *StandardHarness) runInner(ctx context.Context, options HarnessRunOptions) RunResult {
	task := options.Task

	// #124 startup validation (UNGATED): every serializable handle in the task's
	// strategy tree must resolve against the ExecutionRegistry, BEFORE the first
	// turn. The IsEmpty gate is gone — collaborator resolution now goes through
	// the registry for every run (the default Agent / ToolRegistry / Verifier /
	// MetricEvaluator are folded under the empty key by NewStandardHarness), so
	// validation is a single resolution path. A scaffold-only config with no
	// folded agent leaves an empty registry; validation is skipped there so those
	// tests still reach the no-executor leaf failure rather than a spurious
	// unresolved-handle halt.
	if !h.config.Registry.IsEmpty() {
		if err := h.config.Registry.Validate(task); err != nil {
			he, ok := err.(HarnessError)
			if !ok {
				he = &StrategyNotFoundError{Key: err.Error()}
			}
			return RunResult{
				Kind:      RunFailure,
				Reason:    HaltReason{Kind: HaltConfigurationError, ConfigError: he},
				SessionID: task.SessionID,
			}
		}
	}

	var session SessionState
	switch {
	case options.SessionState != nil:
		session = *options.SessionState
	case h.config.AutoPersistSessions && seedsIncomingSessionState(task.LoopStrategy.Kind):
		// Auto-load by session id. errors.Is-style swallow: any read error (or a
		// miss) starts fresh — never surfaced as a HaltReason (D8). No logging
		// facade is wired into spore-core, mirroring the Rust reference.
		if prior, found, err := h.config.effectiveSessionStore().GetSession(ctx, task.SessionID); err == nil && found && prior != nil {
			session = prior.SessionState
		}
	}
	budget := BudgetSnapshot{}

	// #124: the central dispatch switch is GONE — the only switch left is the
	// enum→config delegation inside LoopStrategy.Run. The harness entry collapses
	// to driveStrategy, which builds the shared ExecutionContext and calls
	// task.LoopStrategy.Run(ctx, cx) via the recursive executor.
	//
	// Strategies that BUILD ON incoming state (ReAct / PlanExecute /
	// SelfVerifying) get the instruction seeded HERE on the FRESH run — the
	// compaction adapter ignores task on Assemble, so the harness owns prompt
	// delivery; on a fresh run this turns an otherwise-empty conversation into a
	// real user turn; on multi-turn runs each Run call appends its follow-up. The
	// resume path is intentionally excluded — its conversation already exists.
	// Ralph / HillClimbing re-seed a fresh window internally (D7), so their
	// incoming state is discarded by the config body and the seed is skipped.
	if seedsIncomingSessionState(task.LoopStrategy.Kind) {
		h.config.ContextManager.AppendUserMessage(ctx, &session, task.Instruction)
	}
	return h.driveStrategy(ctx, task, session, budget, options.OnStream)
}

// driveStrategy is the recursive-executor entry (#124): it builds the shared
// ExecutionContext, seeds the per-run scratch (task / session / budget), drives
// task.LoopStrategy.Run(ctx, cx), and translates the outcome back into a terminal
// RunResult (Q5). A non-terminal pause / escalate stashed in
// Scratch.TerminalOverride propagates VERBATIM (it never collapses into a
// StrategyOutcome).
func (h *StandardHarness) driveStrategy(
	ctx context.Context,
	task Task,
	session SessionState,
	budget BudgetSnapshot,
	onStream StreamSink,
) RunResult {
	return h.driveStrategyWithResumeSeed(ctx, task, session, budget, onStream, nil, nil)
}

// driveStrategyWithResumeSeed is driveStrategy with an optional cross-process
// Continue checkpoint seed (#129, AC2): resumeSeed = (phase, continuesUsed)
// seeds the FIRST matching budget scope's ContinuesUsed (via
// ResumedBudgetContext) so a Continue that spanned a process pause resumes with
// the correct continue count instead of a zeroed one. nil is the fresh-run path.
//
// #129: the BARE-LEAF resolution site is driveStrategy (a bare leaf never
// self-resolves inside its own body — rule 6 — it PROPAGATES a typed
// BudgetExhausted here, the single recovery site for a top-level leaf). When the
// leaf's CONFIGURED behavior resolves to Continue, grant the leaf one more round
// IN-PROCESS (bump ContinuesUsed, refresh the step cap) and re-drive WITHOUT any
// serialization (AC3) — looping until the behavior resolves to Fail/Escalate or
// the strategy completes. behaviorForResolution carries the resolution chain's
// mutated state across in-process continues so MaxContinues is honored.
//
// #131/#138: resumeSeed = &session seeds the FIRST PlanExecute walk from session
// (the stalled worker conversation). For a consult resume the consult answer is
// already injected; for a budget resume (#138) it is the worker's full
// post-exhaustion session. The walk re-attaches it to the single InProgress task
// (execute-phase exhaustion) or, when the durable task_list has no InProgress
// task, to the PLAN session (#138 AC3 plan-phase exhaustion). nil on every
// fresh / non-resume path.
func (h *StandardHarness) driveStrategyWithResumeSeed(
	ctx context.Context,
	task Task,
	session SessionState,
	budget BudgetSnapshot,
	onStream StreamSink,
	resumeContinues *ResumeContinueSeed,
	resumeSeed *SessionState,
) RunResult {
	sessionID := task.SessionID
	// The Continue resolution state threaded across in-process rounds: the
	// (possibly fallen-through) behavior + how many continues have been spent.
	var behaviorForResolution *BudgetContext
	// #129 (AC2): the cross-process checkpoint seed is consumed by the FIRST
	// round only (the resumed node's scope); later in-process rounds carry their
	// continue count via behaviorForResolution.
	for {
		cx := NewExecutionContext(&h.config.Registry)
		cx.Executor = h
		cx.Stream = onStream
		cx.Scratch.RunSession = session
		cx.Scratch.RunBudget = budget
		taskCopy := task
		cx.Scratch.Task = &taskCopy
		cx.Scratch.ResumeContinues = resumeContinues
		resumeContinues = nil
		// #131/#138: the resume seed is consumed by the FIRST round's PlanExecute
		// walk only; later in-process rounds run normally.
		cx.Scratch.ResumeSeed = resumeSeed
		resumeSeed = nil

		outcome := task.LoopStrategy.Run(ctx, cx)

		// A pause / escalate propagates verbatim (preserves the HITL / consult /
		// escalation contract through the recursive executor).
		if cx.Scratch.TerminalOverride != nil {
			return *cx.Scratch.TerminalOverride
		}
		switch outcome.Kind {
		case StrategyOutcomeComplete:
			return RunResult{
				Kind:         RunSuccess,
				Output:       outcome.Complete,
				SessionID:    sessionID,
				Usage:        cx.Usage,
				Turns:        cx.Scratch.RunBudget.Turns,
				SessionState: cx.Scratch.RunSession,
			}
		case StrategyOutcomeBudgetExhausted:
			// #129: a BARE LEAF exhaustion propagates here carrying its CONFIGURED
			// behavior (Q1 — the bare-leaf resolution site honors it; the leaf body
			// never did). Resolve it ONCE: a Continue with continues left re-drives
			// in-process; a spent Continue falls through to Fail/Escalate. Carry the
			// resolution chain's state across rounds so MaxContinues is respected.
			// (A combinator under SurfaceToHuman never reaches this arm — it sets
			// the terminal override, returned above.)
			ex := outcome.Exhausted
			// #125: the exhausted node's own StepsTaken is the turn count it
			// reached (the scratch budget is not written back on the propagate
			// path). Fall back to the scratch turns if it is somehow 0.
			turns := cx.Scratch.RunBudget.Turns
			if ex != nil && ex.StepsTaken > 0 {
				turns = ex.StepsTaken
			}
			if ex == nil {
				// Defensive: no exhaustion payload — preserve the legacy Failure.
				return RunResult{
					Kind:         RunFailure,
					Reason:       HaltReason{Kind: HaltBudgetExceeded, LimitType: BudgetLimitTurns},
					SessionID:    sessionID,
					Usage:        cx.Usage,
					Turns:        turns,
					SessionState: SessionState{},
				}
			}
			// #137: the terminal HaltReason for a Fail / Escalate->Fail resolution
			// depends on the CAUSE — a genuine budget exhaustion reports
			// BudgetExceeded; an error-loop break reports ToolErrorLoop. (A granted
			// Continue re-drives the window, whose loop-local error counter starts
			// fresh.)
			terminalReason := HaltReason{Kind: HaltBudgetExceeded, LimitType: BudgetLimitTurns}
			if ex.Cause.Kind == ExhaustionCauseToolErrorLoop {
				terminalReason = HaltReason{
					Kind:              HaltToolErrorLoop,
					Tool:              ex.Cause.Tool,
					ConsecutiveErrors: ex.Cause.ConsecutiveErrors,
				}
			}
			// Reconstruct the resolution scope: the FIRST round uses the leaf's
			// propagated behavior + ContinuesUsed; later in-process rounds reuse
			// the threaded (possibly fallen-through) state.
			var scope BudgetContext
			if behaviorForResolution != nil {
				scope = ResumedBudgetContext(ex.Policy, behaviorForResolution.Behavior, ex.Phase, behaviorForResolution.ContinuesUsed)
				behaviorForResolution = nil
			} else {
				scope = ResumedBudgetContext(ex.Policy, ex.Behavior, ex.Phase, ex.ContinuesUsed)
			}
			resolution := scope.ResolveExhausted()
			switch resolution {
			case ExhaustedResolutionContinue:
				// In-process continue (AC3: NO serialization). Refresh the leaf's
				// step cap and re-enter the loop carrying the post-run session so
				// the conversation context survives (Continue PRESERVES context —
				// AC4). The granted cap is StepsTaken + policy.value so the leaf
				// gets a fresh window after the checkpoint.
				allowance, capped := ex.Policy.AllowanceValue()
				if !capped {
					allowance = 1
				}
				granted := saturatingAddU32(ex.StepsTaken, allowance)
				grantTaskBudget(&task, granted)
				session = cx.Scratch.RunSession
				budget = cx.Scratch.RunBudget
				// Thread the resolution chain's post-continue state so a
				// subsequent exhaustion sees the bumped ContinuesUsed.
				threaded := scope
				behaviorForResolution = &threaded
				continue
			case ExhaustedResolutionEscalate:
				// #130: a BARE LEAF exhaustion under SurfaceToHuman PAUSES with a
				// BudgetExhausted request (a combinator under SurfaceToHuman never
				// reaches this arm — it sets the terminal override at its escalate
				// site and is returned above). A bare leaf offers
				// [ContinueWithBudget, Fail] (fork C — no sibling to Skip to).
				if h.config.EffectiveEscalationMode().Kind == EscalationSurfaceToHuman {
					err := &BudgetExhausted{
						Policy:        ex.Policy,
						Behavior:      scope.Behavior,
						StepsTaken:    ex.StepsTaken,
						ContinuesUsed: scope.ContinuesUsed,
						Phase:         ex.Phase,
					}
					// #138 AC2-a: carry the FULL stalled leaf session (it propagated
					// into scratch with its conversation) and the leaf's own toolset
					// handle (#140/AC4-a).
					workerSession := cx.takeSession()
					workerToolset := workerToolsetOf(&task.LoopStrategy)
					return promoteBudgetExhaustedToHuman(
						err,
						ex.PartialOutput,
						leafEscalationActions(err),
						sessionID,
						task,
						cx.Scratch.RunBudget,
						turns,
						workerSession,
						workerToolset,
					)
				}
				var messages []Message
				if ex.PartialOutput != nil {
					messages = []Message{{
						Role:    RoleAssistant,
						Content: NewTextContent(*ex.PartialOutput),
					}}
				}
				return RunResult{
					Kind: RunFailure,
					// #137: ToolErrorLoop vs BudgetExceeded per cause.
					Reason:       terminalReason,
					SessionID:    sessionID,
					Usage:        cx.Usage,
					Turns:        turns,
					SessionState: SessionState{Messages: messages},
				}
			default: // ExhaustedResolutionFail
				// Fail contract: the partial is DISCARDED.
				return RunResult{
					Kind: RunFailure,
					// #137: ToolErrorLoop vs BudgetExceeded per cause.
					Reason:       terminalReason,
					SessionID:    sessionID,
					Usage:        cx.Usage,
					Turns:        turns,
					SessionState: SessionState{},
				}
			}
		default: // StrategyOutcomeFailed
			var configErr HarnessError = &InvalidConfigurationError{Message: "strategy failed"}
			if outcome.Failed != nil {
				configErr = outcome.Failed
			}
			return RunResult{
				Kind:         RunFailure,
				Reason:       HaltReason{Kind: HaltConfigurationError, ConfigError: configErr},
				SessionID:    sessionID,
				Usage:        cx.Usage,
				Turns:        cx.Scratch.RunBudget.Turns,
				SessionState: cx.Scratch.RunSession,
			}
		}
	}
}

// ============================================================================
// StrategyExecutor impl (#124): the harness-side primitives the recursive
// per-variant RunStrategy.Run bodies delegate to. Each primitive wraps an
// existing, tested orchestration method so behavior stays at parity — the only
// structural change is that the per-variant bodies now own their loops and the
// central dispatch switch is gone.
// ============================================================================

// ReactWindow runs one bounded ReAct turn-loop window (the leaf primitive) on
// the RESOLVED worker agent (#124 — threaded by ReactConfig.Run). Issue 2: the
// resolved toolset handle is threaded down alongside the agent so the window
// dispatches the per-node scoped catalogue (empty handle ⇒ global-catalogue
// fallback).
func (h *StandardHarness) ReactWindow(ctx context.Context, task Task, maxIterations uint32, session SessionState, budget BudgetSnapshot, onStream StreamSink, agent Agent, toolset ToolsetRef, outputSchema json.RawMessage, outputSchemaMaxRetries uint32, systemPrompt *string) RunResult {
	return h.runReActInner(ctx, task, maxIterations, session, budget, onStream, agent, toolset, outputSchema, outputSchemaMaxRetries, systemPrompt)
}

// ResolveWorkerAgent resolves the worker agent for a LoopStrategy tree from the
// ExecutionRegistry (#124): the agent on the LEAF reached by descending the
// worker child chain (workerAgentKeyOf). Returns a typed UnresolvedHandle
// failure RunResult when the key is absent.
func (h *StandardHarness) ResolveWorkerAgent(ls *LoopStrategy) (Agent, *RunResult) {
	key := workerAgentKeyOf(ls)
	agent, ok := h.config.Registry.ResolveAgent(AgentRef(key))
	if !ok {
		he := &UnresolvedHandleError{Kind: "agent", Key: key}
		return nil, &RunResult{
			Kind:   RunFailure,
			Reason: HaltReason{Kind: HaltConfigurationError, ConfigError: he},
		}
	}
	return agent, nil
}

// WorkspaceRoot returns the sandbox workspace root (#124). Empty when no sandbox
// is wired.
func (h *StandardHarness) WorkspaceRoot() string {
	if h.config.Sandbox == nil {
		return ""
	}
	return h.config.Sandbox.WorkspaceRoot()
}

// AppendUserMessage seeds a user message onto session via the ContextManager
// seam (#124 — alias of SeedUserMessage for the combinator bodies).
func (h *StandardHarness) AppendUserMessage(ctx context.Context, session *SessionState, text string) {
	h.config.ContextManager.AppendUserMessage(ctx, session, text)
}

// HillEvaluate runs one HillClimbing metric evaluation on the resolved evaluator
// over a fresh SessionState (#124). On success ok is true; on failure ok is
// false and (errStatus, errMsg) carry the typed failure.
func (h *StandardHarness) HillEvaluate(ctx context.Context, evaluator MetricEvaluator, sessionID SessionID, taskID TaskID) (float64, time.Duration, HillClimbIterationStatus, string, bool) {
	res, err := evaluator.Evaluate(ctx, h.config.Sandbox, sessionID, taskID, SessionState{})
	if err != nil {
		return 0, 0, err.Status, err.Message, false
	}
	return res.Value, res.Duration, "", "", true
}

// HillRevert reverts the working tree to HEAD through the sandbox (#124,
// Decision 1). Best-effort.
func (h *StandardHarness) HillRevert(ctx context.Context) {
	h.hillClimbingRevert(ctx)
}

// HillCommitHash resolves the commit_hash recorded on a TSV row (#124,
// Decision 1; v1 always empty).
func (h *StandardHarness) HillCommitHash(ctx context.Context) string {
	return h.hillClimbingCommitHash(ctx)
}

// HillEmitIteration emits one fire-and-forget per-iteration HillClimbing
// observability span (#124).
func (h *StandardHarness) HillEmitIteration(ctx context.Context, sessionID SessionID, taskID TaskID, spanSeq *uint64, iteration uint32, metricValue float64, hasMetric bool, delta float64, hasDelta bool, status HillClimbIterationStatus, reverted bool) {
	h.emitHillClimbingIteration(ctx, sessionID, taskID, spanSeq, iteration, metricValue, hasMetric, delta, hasDelta, status, reverted)
}

// HillWriteTSV serializes the HillClimbing results log (#124, Decisions 2/3).
func (h *StandardHarness) HillWriteTSV(workspaceRoot string, taskID TaskID, rows []HillClimbRow) {
	h.writeHillClimbingTSV(workspaceRoot, taskID, rows)
}

// SeedUserMessage seeds a user message onto session (the ContextManager seam).
func (h *StandardHarness) SeedUserMessage(ctx context.Context, session *SessionState, text string) {
	h.config.ContextManager.AppendUserMessage(ctx, session, text)
}

// PlanDirective returns the planning directive seeded before the plan
// sub-strategy runs (R1) — the "respond with a single JSON plan" instruction
// wrapped around the task. Dispatched by the recursive PlanExecuteConfig.Run
// before c.Plan.Run (#124).
func (h *StandardHarness) PlanDirective(instruction string) string {
	return planDirective(instruction)
}

// RunPlanSubtree dispatches the plan sub-strategy plan for planTask over
// planSession, returning its terminal RunResult (#124). Genuinely recursive —
// the child's Run drives its WHOLE loop and resolves its own worker agent from
// the registry via the plan child's leaf ReactConfig.Agent (Q1 — the separate
// planner_agent concept is dropped). Returns nil if the child produced no
// terminal.
func (h *StandardHarness) RunPlanSubtree(ctx context.Context, plan *LoopStrategy, planTask Task, planSession SessionState, budget BudgetSnapshot) *RunResult {
	// #124 Q1: the separate planner_agent concept is DROPPED — the plan child's
	// leaf ReactConfig.Agent is authoritative and the plan child resolves its own
	// worker agent from the registry via plan.Run.
	cx := NewExecutionContext(&h.config.Registry)
	cx.Executor = h
	cx.Scratch.RunSession = planSession
	cx.Scratch.RunBudget = budget
	pt := planTask
	cx.Scratch.Task = &pt
	_ = plan.Run(ctx, cx)
	return cx.takeChildOverride()
}

// CapturePlanArtifact captures + persists a PlanArtifact from the plan
// sub-strategy's output text (R3/R4/R11). Adapts the private capture helper to
// the public seam shape.
func (h *StandardHarness) CapturePlanArtifact(ctx context.Context, sessionID SessionID, planOutput string, usage AggregateUsage, turns uint32) (PlanPhaseOutcome, *RunResult) {
	outcome, failure := h.captureAndPersistPlan(ctx, sessionID, planOutput, usage, turns)
	if failure != nil {
		return PlanPhaseOutcome{}, failure
	}
	return PlanPhaseOutcome{
		Artifact: outcome.artifact,
		Usage:    outcome.usage,
		Turns:    outcome.turns,
	}, nil
}

// ReconcileCompletedTasks marks every task already Completed on the durable
// RunStore checkpoint as Completed in taskList (A.6 deep-resume).
func (h *StandardHarness) ReconcileCompletedTasks(ctx context.Context, sessionID SessionID, taskList *TaskList) {
	h.reconcileDeepResume(ctx, sessionID, taskList)
}

// FireTaskAdvance fires the OnTaskAdvance hook (pre, mutable) for an execute
// step. The hook may rewrite stepTask.Instruction.
func (h *StandardHarness) FireTaskAdvance(ctx context.Context, sessionID SessionID, stepTask *Task, taskIndex, totalTasks int) {
	if h.config.Hooks != nil {
		hctx := &HookContext{
			Event:      HookEventOnTaskAdvance,
			SessionID:  sessionID,
			Task:       stepTask,
			TaskIndex:  taskIndex,
			TotalTasks: totalTasks,
		}
		_, _ = h.config.Hooks.Fire(ctx, hctx)
	}
}

// PersistTaskList persists a parsed task list through the RunStore seam.
func (h *StandardHarness) PersistTaskList(ctx context.Context, sessionID SessionID, taskList TaskList) {
	h.persistTaskList(ctx, sessionID, taskList)
}

// LoadTaskList reads the persisted TaskList (with real blockers) for sessionID
// from the RunStore under TaskListExtrasKey (#126, decision C — the ONE
// authoring path). A storage miss / decode failure yields (TaskList{}, false),
// and the executor then falls back to the linear plan-artifact bridge.
func (h *StandardHarness) LoadTaskList(ctx context.Context, sessionID SessionID) (TaskList, bool) {
	if h.config.RunStore == nil {
		return TaskList{}, false
	}
	// #142: the task_list is DURABLE — keyed by the stable project namespace, not
	// the per-window sessionID (retained as the seam param).
	raw, found, err := h.config.RunStore.Get(ctx, h.config.durableNamespace(sessionID), TaskListExtrasKey)
	if err != nil || !found {
		return TaskList{}, false
	}
	var list TaskList
	if json.Unmarshal(raw, &list) != nil {
		return TaskList{}, false
	}
	return list, true
}

// observeWriteCall records a harness-OBSERVED write/edit at the tool-dispatch
// seam (#126, AC2). Only write_file / edit_file calls carrying a string "path"
// are recorded; de-duplicated against what is already accumulated for the
// current step. Called from the ReAct turn loop for the call ACTUALLY
// dispatched, so the path comes from the real tool call — never a
// model-self-reported field.
func (h *StandardHarness) observeWriteCall(call ToolCall) {
	if call.Name != "write_file" && call.Name != "edit_file" {
		return
	}
	var probe struct {
		Path string `json:"path"`
	}
	if json.Unmarshal(call.Input, &probe) != nil || probe.Path == "" {
		return
	}
	h.observedWritesMu.Lock()
	defer h.observedWritesMu.Unlock()
	for _, p := range h.observedWrites {
		if p == probe.Path {
			return
		}
	}
	h.observedWrites = append(h.observedWrites, probe.Path)
}

// TakeObservedWrites drains and returns the accumulated observed write/edit
// paths, resetting the accumulator (#126, AC2). The DAG executor calls this on
// task completion to build a StepLedgerEntry's files_touched.
func (h *StandardHarness) TakeObservedWrites() []string {
	h.observedWritesMu.Lock()
	defer h.observedWritesMu.Unlock()
	out := h.observedWrites
	h.observedWrites = nil
	return out
}

// ClearObservedWrites clears the observed-write accumulator (#126, AC2). The DAG
// executor calls this before each step so a task's files_touched reflect ONLY
// the writes that step issues.
func (h *StandardHarness) ClearObservedWrites() {
	h.observedWritesMu.Lock()
	defer h.observedWritesMu.Unlock()
	h.observedWrites = nil
}

// Finalize finalizes observability for a terminal outcome (mirrors the tail of
// runReAct / finalizePlanExecute). Non-terminal pauses are never finalized.
func (h *StandardHarness) Finalize(ctx context.Context, result RunResult) {
	switch result.Kind {
	case RunSuccess:
		h.finalizeObservability(ctx, result.SessionID, TerminalSuccess, "")
	case RunFailure:
		h.finalizeObservability(ctx, result.SessionID, TerminalFailure, haltReasonString(result.Reason))
	case RunEscalate:
		h.finalizeObservability(ctx, result.SessionID, TerminalEscalated, "")
	case RunWaitingForHuman, RunConsult:
		// Not terminal — do not flush.
	}
}

// Compile-time check: StandardHarness implements StrategyExecutor (#124).
var _ StrategyExecutor = (*StandardHarness)(nil)

// seedsIncomingSessionState reports whether a loop strategy seeds an incoming
// SessionState (issue #102, D7). Only ReAct and SelfVerifying do; Ralph and
// HillClimbing re-seed a fresh SessionState per context window / iteration and
// discard incoming state by design, so auto-load is skipped for them.
func seedsIncomingSessionState(kind LoopStrategyKind) bool {
	return kind == StrategyReAct || kind == StrategySelfVerifying
}

// autoPersistTerminal writes the terminal run state to the SessionStore when
// AutoPersistSessions is enabled (issue #102). One write per Run()/Resume(), at
// the same terminal seam as the observability flush.
//
// For Success/Failure it synthesizes a completed-run PausedState (D4): empty
// pending tool calls, empty approved results, no human request, no child state
// — carrying only the final SessionState so a later GetSession resumes
// losslessly. For WaitingForHuman/Escalate it persists the carried PausedState
// directly (D6 — the cross-process pause case). Storage errors are
// swallowed-and-logged (D8): a put failure must never lose the run nor surface
// as a HaltReason.
//
// When disabled (the default) it returns immediately WITHOUT touching the store
// — the off-by-default zero-I/O contract.
func (h *StandardHarness) autoPersistTerminal(ctx context.Context, result *RunResult) {
	if !h.config.AutoPersistSessions {
		return
	}
	var (
		sessionID SessionID
		state     *PausedState
	)
	switch result.Kind {
	case RunSuccess, RunFailure:
		sessionID = result.SessionID
		// Synthesize a completed-run PausedState: empty pending fields, no human
		// request, no child — it carries only the final history so a later
		// GetSession(..).SessionState resumes losslessly.
		state = &PausedState{
			SessionID:        sessionID,
			TaskID:           TaskID(string(sessionID)),
			TurnNumber:       result.Turns,
			SessionState:     result.SessionState,
			PendingToolCalls: nil,
			ApprovedResults:  nil,
			HumanRequest:     nil,
			Task:             NewTask("", sessionID, ReActStrategy(0)),
			BudgetUsed:       BudgetSnapshot{},
			ChildState:       nil,
		}
	case RunWaitingForHuman, RunEscalate, RunConsult:
		// Persist the carried pause state directly (D6). RunConsult (issue #114)
		// is non-terminal but carries a full PausedState — persist it like
		// WaitingForHuman so a cross-process host can later ResumeConsult it.
		if result.State == nil {
			return
		}
		sessionID = result.State.SessionID
		state = result.State
	default:
		return
	}
	// Swallow-and-log on error (D8): a storage hiccup must not lose the run.
	_ = h.config.effectiveSessionStore().PutSession(ctx, sessionID, state)
}

// nowRFC3339 returns the current UTC time as an RFC3339 string — the
// timestamp form every span carries (mirrors guideregistry.nowTimestamp and
// the observability backend's lexical timestamps).
func nowRFC3339() string { return time.Now().UTC().Format(time.RFC3339) }

// TerminalOutcome is the harness's 3-state terminal signal handed to the
// observability seam (issue #80). It mirrors the Rust `SessionOutcome` arms the
// harness loop produces: a clean success, a halt failure, or an escalation. An
// escalation is NOT a failure and NOT a partial — it is an intentional, clean
// termination that hands a structured HarnessSignal to the caller.
type TerminalOutcome uint8

const (
	// TerminalSuccess — the run completed successfully.
	TerminalSuccess TerminalOutcome = iota
	// TerminalFailure — the run halted on a HaltReason.
	TerminalFailure
	// TerminalEscalated — the run terminated via a tool escalation signal.
	TerminalEscalated
)

// finalizeObservability records the terminal outcome and flushes the session.
// Called at every terminal runReAct outcome (Success, any Failure, or an
// Escalation, issue #80) — never on a WaitingForHuman pause, which is not
// terminal. No-op when no observer is configured. Mirrors the Rust
// `finalize_observability` wrapper.
func (h *StandardHarness) finalizeObservability(ctx context.Context, sessionID SessionID, outcome TerminalOutcome, failureReason string) {
	if h.config.Observability == nil {
		return
	}
	h.config.Observability.SetSessionOutcome(sessionID, outcome, failureReason)
	h.config.Observability.FlushSession(ctx, sessionID)
}

// effectiveToolRegistry returns the harness-loop tool registry to use for a run
// keyed by sessionID. Mirrors Rust's effective_tool_registry.
//
// When catalogue tools were added via HarnessBuilder.Tool / Tools, the builder
// folded them into a *StandardToolRegistry held in CatalogueRegistry. This
// re-injects that registry's ToolContext with the run's SessionID + the
// configured storage seams (so every tool dispatch threads the run's storage)
// and returns it. Otherwise it returns the injected ToolRegistry seam unchanged.
//
// Unlike Rust — which builds a fresh RealToolRegistry bridge per run — Go's
// canonical registry IS the harness registry, so the per-run wiring is a
// SetToolContext call on the shared registry rather than a new bridge object.
//
// Issue 2 (per-node toolset scoping): toolset is the resolving leaf's ToolsetRef
// handle. When it is NON-EMPTY and a per-key catalogue was registered via
// HarnessBuilder.ToolsetTools, THAT catalogue is bridged (its ToolContext
// re-injected with the run's SessionID + storage), so the node dispatches ONLY
// its own tools — strict scoping. Otherwise (empty handle, or non-empty handle
// with no per-key catalogue) the existing global-catalogue / ToolRegistry seam
// fallback applies, so examples 01–11 that use .Tools() keep working
// byte-for-byte. Mirrors Rust effective_tool_registry.
func (h *StandardHarness) effectiveToolRegistry(sessionID SessionID, toolset ToolsetRef) ToolRegistry {
	// Strict per-node scoping: a non-empty handle with its own catalogue.
	if toolset != "" {
		if catalogue, ok := h.config.ToolsetCatalogues[string(toolset)]; ok && catalogue != nil {
			catalogue.SetToolContext(
				// #142: pin the stable project namespace so the catalogue task_list
				// tool keys its DURABLE blob by project_id (surviving Ralph window
				// resets), while ephemeral session state stays on sessionID.
				NewToolContextWithProject(sessionID, h.config.ProjectNamespace, h.config.ToolRunStore, h.config.ToolMemoryStore),
			)
			return catalogue
		}
	}
	// Fallback: empty handle (back-compat) or unregistered non-empty handle.
	if h.config.CatalogueRegistry == nil {
		return h.config.ToolRegistry
	}
	h.config.CatalogueRegistry.SetToolContext(
		NewToolContextWithProject(sessionID, h.config.ProjectNamespace, h.config.ToolRunStore, h.config.ToolMemoryStore),
	)
	return h.config.CatalogueRegistry
}

// registrySchemas flattens a tool registry's active schemas (full catalogue —
// nil phase) into the slim model-facing []ToolSchema the agent advertises.
// Mirrors Rust's ToolRegistry::schemas. Returns nil when the registry exposes
// no tools, so an empty registry advertises nothing (and the request hash stays
// byte-identical to the no-tools baseline).
// buildContextSources builds the per-turn ContextSources threaded into the
// structural ContextManager.Assemble seam (issue #115 / SC-26). Configured
// guides reach the model structurally through the seam, not as an ad-hoc
// User-message prepend. Skills (the manifest + active bodies, progressive
// disclosure) are appended as guides from the shared catalog, so loading a skill
// via load_skill makes its body sticky in the System block on the next turn.
//
// #163 / SC-26 follow-up: when a memory source is configured, the harness queries
// the provider each turn and injects the relevant items structurally (rendered
// into the same System block as guides). The query text defaults to the task
// instruction, so retrieved memory tracks the current work; a configured
// MemoryConfig.QueryText overrides it. A query error
// is swallowed here (empty memory) — memory is best-effort context, never a halt;
// the Assemble seam is synchronous and error-free, so the swallow lives here. No
// memory config leaves memory empty (byte-identical pre-#163 path).
//
// The composed static prompt is empty until the chunk-provider path is wired. An
// empty result renders to nothing, so a harness with no sources stays
// byte-identical.
func (h *StandardHarness) buildContextSources(ctx context.Context, toolSchemas []ToolSchema, taskInstruction string) ContextSources {
	guides := append([]Guide(nil), h.config.Guides...)
	if h.config.Skills != nil {
		guides = append(guides, h.config.Skills.ActiveGuides()...)
	}
	var memory []MemoryItem
	if mem := h.config.Memory; mem != nil && mem.Query != nil {
		// A configured query text overrides the task instruction, so retrieved
		// memory can track a fixed concern instead of the current task.
		queryText := taskInstruction
		if mem.QueryText != nil {
			queryText = *mem.QueryText
		}
		items, err := mem.Query(ctx, queryText, mem.Domain, mem.MinRelevance, mem.MaxItems)
		if err == nil {
			memory = items
		}
	}
	return ContextSources{
		Guides:      guides,
		Memory:      memory,
		ToolSchemas: toolSchemas,
	}
}

func registrySchemas(reg ToolRegistry) []ToolSchema {
	active := reg.ActiveSchemas(nil)
	if len(active) == 0 {
		return nil
	}
	out := make([]ToolSchema, len(active))
	for i, s := range active {
		out[i] = s.ToModelSchema()
	}
	return out
}

// enrichToolError renders a recoverable tool-error message annotated with the
// tool's parameter schema (when known) plus a fixed correctly-typed-JSON hint
// (issue #137; mirrors the Rust enrich_tool_error). Returns a recoverable
// ToolOutput.Error carrying the enriched text. schema may be nil — the schema
// section is then omitted but the hint is always appended.
//
// The schema JSON is rendered with object keys sorted lexicographically and
// recursively, with compact separators and NO HTML escaping, so the bytes are
// IDENTICAL to the Rust reference (serde_json without preserve_order serializes
// object keys via a BTreeMap → alphabetically sorted, and does not HTML-escape).
// json.Compact on the raw bytes would instead preserve insertion order and so
// diverge from Rust / TS / Python.
func enrichToolError(message string, schema *ToolSchema) ToolOutput {
	enriched := message
	if schema != nil && len(schema.InputSchema) > 0 {
		if rendered, ok := sortedCompactJSON(schema.InputSchema); ok {
			enriched += "\n\nExpected parameter schema: " + rendered
		}
	}
	enriched += "\n\nHint: provide arguments as correctly-typed JSON " +
		"(e.g. true/false as a bool, 42 as a number, [\"a\"] as an array) " +
		"rather than as quoted strings."
	return ToolOutput{Kind: ToolOutputError, Message: enriched, Recoverable: true}
}

// sortedCompactJSON re-serializes raw JSON with object keys sorted
// lexicographically/recursively, compact separators, and NO HTML escaping —
// byte-identical to serde_json's default (sorted BTreeMap, no <>& escaping).
// Go's encoding/json sorts map[string]any keys recursively; a json.Encoder with
// SetEscapeHTML(false) suppresses the <>& escaping serde_json never applies.
// Numbers are preserved verbatim via json.Number (UseNumber) so e.g. integer
// vs. float spelling round-trips unchanged. Returns false if raw is not valid
// JSON.
func sortedCompactJSON(raw json.RawMessage) (string, bool) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var tree any
	if err := dec.Decode(&tree); err != nil {
		return "", false
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(tree); err != nil {
		return "", false
	}
	// json.Encoder.Encode appends a trailing newline; strip it.
	return strings.TrimRight(buf.String(), "\n"), true
}

// runReAct drives the ReAct loop, then finalizes observability for terminal
// outcomes. A WaitingForHuman pause is not terminal, so it is never flushed
// here — the eventual Resume path reaches a terminal outcome and flushes then.
// Mirrors the Rust `run_react` thin wrapper over `run_react_inner`.
func (h *StandardHarness) runReAct(
	ctx context.Context,
	task Task,
	maxIterations uint32,
	session SessionState,
	budget BudgetSnapshot,
	onStream StreamSink,
	agent Agent,
) RunResult {
	// Issue 2: the legacy runReAct wrappers (PlanExecute/DAG execute steps,
	// single-shot run, the resume paths) carry no per-node toolset handle, so they
	// keep the global-catalogue fallback (empty handle) — byte-for-byte with
	// pre-Issue-2 behaviour.
	// Issue #139: the legacy wrapper does NOT enforce output schemas (the recursive
	// ReactConfig.Run is the single enforcement seam). nil/0 ⇒ byte-for-byte
	// pre-#139.
	// SC-10: the legacy runReAct wrapper carries no per-leaf system-prompt override
	// (only the recursive ReactConfig.Run does) ⇒ the window uses the global
	// config.SystemPrompt, byte-identical.
	result := h.runReActInner(ctx, task, maxIterations, session, budget, onStream, agent, ToolsetRef(""), nil, 0, nil)
	switch result.Kind {
	case RunSuccess:
		h.finalizeObservability(ctx, result.SessionID, TerminalSuccess, "")
	case RunFailure:
		h.finalizeObservability(ctx, result.SessionID, TerminalFailure, haltReasonString(result.Reason))
	case RunEscalate:
		// Escalation is a clean terminal outcome (issue #80) — finalize with
		// the dedicated Escalated outcome (NOT Failure, NOT Partial), in
		// contrast to WaitingForHuman which is not terminal and does not flush.
		h.finalizeObservability(ctx, result.SessionID, TerminalEscalated, "")
	case RunWaitingForHuman:
		// Not terminal — do not flush.
	}
	return result
}

// haltReasonString renders a HaltReason for the failure-outcome reason string,
// mirroring Rust's `format!("{reason:?}")`.
func haltReasonString(r HaltReason) string {
	switch r.Kind {
	case HaltBudgetExceeded:
		return fmt.Sprintf("budget exceeded: %s", r.LimitType)
	case HaltTerminationPolicyHalt:
		return fmt.Sprintf("termination policy halt: %s", r.Reason)
	case HaltMiddlewareHalt:
		return fmt.Sprintf("middleware halt at %s: %s", r.Hook, r.Reason)
	case HaltAgentError:
		if r.AgentError != nil {
			return fmt.Sprintf("agent error: %s", r.AgentError.Error())
		}
		return "agent error"
	case HaltSandboxViolation:
		if r.Violation != nil {
			return fmt.Sprintf("sandbox violation: %s", r.Violation.Error())
		}
		return "sandbox violation"
	case HaltUnrecoverableToolError:
		return fmt.Sprintf("unrecoverable tool error (%s): %s", r.Tool, r.Error)
	case HaltHumanHalted:
		return "human halted"
	case HaltStrategyNotYetImplemented:
		return fmt.Sprintf("strategy not yet implemented: %s", r.Strategy)
	case HaltEmptyPlan:
		return "empty plan"
	case HaltStepFailed:
		return fmt.Sprintf("step %d failed (%q): %s", r.TaskIndex, r.Task, r.Reason)
	case HaltPlanPhaseFailed:
		if r.PlanError != nil {
			return fmt.Sprintf("plan phase failed: %s", r.PlanError.Error())
		}
		return "plan phase failed"
	case HaltSelfVerifyExhausted:
		return fmt.Sprintf("self-verify exhausted after %d iterations: %s", r.Iterations, r.Reason)
	case HaltSelfVerifyMisconfigured:
		return fmt.Sprintf("self-verify misconfigured: %s", r.Reason)
	case HaltRalphCompletionUnmet:
		return fmt.Sprintf("ralph completion unmet after %d windows: %s", r.Iterations, r.Reason)
	case HaltStagnationLimitReached:
		return fmt.Sprintf("stagnation limit reached after %d non-improvements (best %.6f)", r.Iterations, r.BestMetric)
	case HaltHillClimbingMisconfigured:
		return fmt.Sprintf("hill-climbing misconfigured: %s", r.Reason)
	case HaltOutputSchemaViolation:
		return fmt.Sprintf("output schema violation after %d attempts: %s", r.Attempts, r.LastError)
	default:
		return string(r.Kind)
	}
}

// planPhaseOutcome is the internal result of a successful PlanExecute plan
// phase (issue #70). Carries the produced artifact plus the run accounting so
// the caller can build the terminal RunResult. Not part of the public surface —
// the artifact itself is observable via SessionState.Extras["plan_execute"].
type planPhaseOutcome struct {
	artifact PlanArtifact
	usage    AggregateUsage
	turns    uint32
}

// persistTaskList persists the parsed TaskList for the run (Q4). The DURABLE
// write goes through the RunStore seam under TaskListExtrasKey; the #71
// sandbox-filesystem path (.spore/task_list.json) is intentionally NOT used —
// one source of truth. The RunStore write is the single source of truth (#76
// removed the redundant SessionState.Extras mirror). Serialization / store
// failures are swallowed: a successful plan must not be lost to a storage hiccup
// (the default nil/no-op store never fails).
func (h *StandardHarness) persistTaskList(ctx context.Context, sessionID SessionID, taskList TaskList) {
	value, err := json.Marshal(taskList)
	if err != nil {
		return
	}
	// Durable write through the RunStore seam (optional). #142: keyed by the
	// stable project namespace (not the per-window sessionID, retained as the seam
	// param) so a Ralph window reset re-reads it.
	if h.config.RunStore != nil {
		_ = h.config.RunStore.Put(ctx, h.config.durableNamespace(sessionID), TaskListExtrasKey, json.RawMessage(value))
	}
}

// planDirective returns the planning directive seeded before the constrained
// plan turn (R1): the "respond with a single JSON plan object" instruction
// wrapped around the task. Seeded by the recursive PlanExecuteConfig.Run plan
// dispatch before c.Plan.Run (#124).
func planDirective(instruction string) string {
	return fmt.Sprintf(
		"Produce a step-by-step plan for the following task. Respond with a "+
			"single JSON object: {\"tasks\": [<ordered step strings>], "+
			"\"rationale\": <string>}.\n\nTask:\n%s",
		instruction,
	)
}

// captureAndPersistPlan captures + persists a PlanArtifact from the plan
// sub-strategy's output text (#124): R3 (parse), R11 (fire OnPlanCreated,
// mutable), R4 (persist to the RunStore under PlanExecuteExtrasKey). The model
// turn that produced planOutput ran elsewhere — the recursive c.Plan child via
// c.Plan.Run — so this carries no agent call. Returns the captured outcome, or a
// non-nil terminal failure to propagate.
//
// SC-28 — a free-text / markdown plan is NOT a hard failure.
// The strict canonical grammar (+ prose repair) runs first, so a JSON plan
// captures exactly as before — tasks/rationale come straight from the object.
// When BOTH fail the planner emitted prose rather than the JSON object; rather
// than aborting the whole run we capture it as a free-text artifact: the
// verbatim prose is preserved in Rationale, and the runnable Tasks are sourced
// from the durable task_list tool store via LoadTaskList — the ONE authoring
// path (#126 decision C) that the execute phase already prefers over the
// artifact, so JSON was never the only source of executable steps. Mirroring it
// here keeps the OnPlanCreated payload's Tasks populated for panel consumers
// (looper plan_tracker, cordyceps plan_announcer). A prose plan that authored no
// task_list yields empty Tasks — the execute phase then halts with EmptyPlan,
// exactly as a JSON {"tasks": []} would. The pure CapturePlanArtifact grammar
// stays strict and byte-identical across languages; only this harness driver is
// tolerant.
func (h *StandardHarness) captureAndPersistPlan(ctx context.Context, sessionID SessionID, planOutput string, usage AggregateUsage, turns uint32) (*planPhaseOutcome, *RunResult) {
	// R3: capture the artifact from the response text. Uses the prose-repair
	// fallback: a planner that wraps its plan JSON in prose still captures (the
	// strict canonical grammar runs first; repair only rescues a failure).
	artifact, err := CapturePlanArtifactWithRepair(planOutput)
	if err != nil {
		// SC-28: not JSON → capture as free-text. Tasks come from the task_list
		// tool store (LoadTaskList reads the durable, project-scoped list the plan
		// leaf authored via the tool); prose preserved verbatim.
		var tasks []string
		if list, ok := h.LoadTaskList(ctx, sessionID); ok {
			tasks = make([]string, 0, len(list.Tasks))
			for _, t := range list.Tasks {
				tasks = append(tasks, t.Description)
			}
		}
		artifact = PlanArtifact{
			Tasks:     tasks,
			Rationale: planOutput,
		}
	}

	// R11: fire OnPlanCreated synchronously; the hook may rewrite artifact in
	// place. The stored artifact reflects any mutation. Hook errors are non-fatal.
	if h.config.Hooks != nil {
		hctx := &HookContext{
			Event:     HookEventOnPlanCreated,
			SessionID: sessionID,
			Plan:      &artifact,
		}
		_, _ = h.config.Hooks.Fire(ctx, hctx)
	}

	// R4: persist the produced artifact to the RunStore seam under
	// PlanExecuteExtrasKey (#76 — the durable single source of truth). The Put
	// result is swallowed: a successfully-captured plan must not be lost to a
	// storage hiccup (the default nil/no-op store never fails).
	value, marshalErr := json.Marshal(artifact)
	if marshalErr != nil {
		return nil, &RunResult{
			Kind: RunFailure,
			Reason: HaltReason{
				Kind:      HaltPlanPhaseFailed,
				PlanError: newUnparseablePlan(fmt.Sprintf("failed to serialize plan artifact: %s", marshalErr)),
			},
			SessionID: sessionID,
			Usage:     usage,
			Turns:     turns,
		}
	}
	if h.config.RunStore != nil {
		// #142: the plan artifact is DURABLE — keyed by the stable project
		// namespace, not the per-window sessionID (retained as the seam param).
		_ = h.config.RunStore.Put(ctx, h.config.durableNamespace(sessionID), PlanExecuteExtrasKey, json.RawMessage(value))
	}

	return &planPhaseOutcome{artifact: artifact, usage: usage, turns: turns}, nil
}

// reconcileDeepResume marks every task already Completed on the DURABLE RunStore
// checkpoint as Completed in taskList so it is NOT re-run (A.6 deep-resume, Q2).
// Tasks are matched by id (the task list is regenerated deterministically from
// the same artifact). A read miss / error starts fresh.
func (h *StandardHarness) reconcileDeepResume(ctx context.Context, sessionID SessionID, taskList *TaskList) {
	if h.config.RunStore == nil {
		return
	}
	// #142: the durable checkpoint is keyed by the stable project namespace, not
	// the per-window sessionID (retained as the seam param).
	raw, found, err := h.config.RunStore.Get(ctx, h.config.durableNamespace(sessionID), TaskListExtrasKey)
	if err != nil || !found {
		return
	}
	var saved TaskList
	if json.Unmarshal(raw, &saved) != nil {
		return
	}
	for i := range taskList.Tasks {
		for j := range saved.Tasks {
			if saved.Tasks[j].ID == taskList.Tasks[i].ID && saved.Tasks[j].Status == TaskStatusCompleted {
				taskList.Tasks[i].Status = TaskStatusCompleted
				break
			}
		}
	}
}

// runReActInner is the workhorse loop for LoopStrategy ReAct.
// fireStopHooks fires registered Stop hooks (issue #69). It returns
// (reason, true) when the loop should continue — a hook blocked and the per-run
// MaxStopBlocks cap has not yet been hit — incrementing stopBlocks. It returns
// ("", false) to allow normal termination: no hook chain is configured, no hook
// blocked, the cap was reached, or a hook errored (a broken Stop hook must not
// loop the harness forever, so its error is treated as non-blocking).
//
// Firing order is registration order; a wrapped strategy verifier (registered
// as a Stop hook) fires in its registered position. The SelfVerifying verifier
// is expressed this way rather than as bespoke loop logic.
func (h *StandardHarness) fireStopHooks(
	ctx context.Context,
	sessionID SessionID,
	task *Task,
	turnNumber uint32,
	lastOutputText string,
	session *SessionState,
	stopBlocks *uint32,
) (string, bool) {
	if h.config.Hooks == nil {
		return "", false
	}
	instruction := task.Instruction
	lastOutput := TurnOutput{Text: lastOutputText, HadToolCalls: false}
	hctx := &HookContext{
		Event:           HookEventStop,
		SessionID:       sessionID,
		TurnNumber:      turnNumber,
		LastOutput:      &lastOutput,
		TaskInstruction: &instruction,
		SessionState:    session,
	}
	outcome, err := h.config.Hooks.Fire(ctx, hctx)
	if err != nil {
		return "", false
	}
	if outcome.Kind != FireBlock {
		return "", false
	}
	if *stopBlocks >= h.config.effectiveMaxStopBlocks() {
		// Cap reached — terminate anyway.
		return "", false
	}
	*stopBlocks++
	return outcome.Reason, true
}

// errorRun is the current run of consecutive recoverable tool errors for ONE
// tool name (issue #137). Tracked per tool name in a loop-local
// map[string]*errorRun inside runReActInner:
//
//   - On a recoverable ToolOutput.Error for tool T with args A: if the existing
//     run for T has args STRUCTURALLY equal to A, count is incremented; otherwise
//     the run is REPLACED with a fresh {args: A, count: 1} (covers both the first
//     error and the args-changed case).
//   - On ANY success for tool T: the entry is REMOVED (AC1 — success resets the
//     run regardless of args).
//   - At count == N (ErrorLoopThreshold): inject ONE corrective message (AC2).
//     injected guards against re-injection between N and 2*N for the same run.
//   - At count == 2*N: stop looping (AC3).
type errorRun struct {
	// args is the tool-call arguments shared by every error in this run.
	args json.RawMessage
	// count is how many consecutive identical-argument recoverable errors so far.
	count uint32
	// injected records whether the N-threshold corrective message has already
	// been injected for THIS run (so we inject exactly once between N and 2*N).
	injected bool
}

// sameToolArgs reports whether two tool-call argument blobs are structurally
// equal (issue #137). Compares the parsed JSON values rather than raw bytes so
// semantically identical args with different whitespace/key order still match.
func sameToolArgs(a, b json.RawMessage) bool {
	var va, vb any
	// Treat an unmarshal failure as a byte comparison fallback (both blobs are
	// the model's own output, so this is defensive only).
	if err := json.Unmarshal(a, &va); err != nil {
		return bytes.Equal(a, b)
	}
	if err := json.Unmarshal(b, &vb); err != nil {
		return bytes.Equal(a, b)
	}
	return reflect.DeepEqual(va, vb)
}

func (h *StandardHarness) runReActInner(
	ctx context.Context,
	task Task,
	maxIterations uint32,
	session SessionState,
	budget BudgetSnapshot,
	onStream StreamSink,
	// #124: the worker agent is RESOLVED by the caller (the recursing
	// ReactConfig.Run resolves task.LoopStrategy's leaf agent from the registry;
	// Ralph may override it per window). The leaf no longer reads config.Agent.
	agent Agent,
	// Issue 2: the leaf's RESOLVED toolset handle (mirrors agent). Empty ("") ⇒
	// global-catalogue fallback; a non-empty handle with its own per-key catalogue
	// ⇒ strict per-node scoping.
	toolset ToolsetRef,
	// Issue #139: the leaf's RESOLVED output schema (nil ⇒ no enforcement,
	// identical to pre-#139 behavior). When non-nil, the terminal is validated
	// against it (frozen validator subset) and a validation failure feeds the
	// frozen error back + retries up to outputSchemaMaxRetries extra turns WITHIN
	// budget; on exhaustion WITH budget remaining the window returns
	// HaltOutputSchemaViolation. The schema is also set on every turn's
	// ModelParams.OutputSchema so the Ollama `format` channel constrains decoding.
	outputSchema json.RawMessage,
	outputSchemaMaxRetries uint32,
	// SC-10: the leaf's per-node system-prompt OVERRIDE. nil ⇒ the global
	// config.SystemPrompt is used (byte-identical to pre-SC-10). Non-nil ⇒ it
	// REPLACES the global prompt for every turn of this window, so the leaf sees
	// ONLY its own prompt (the per-leaf prompt half of SC-10; the toolset half is
	// the toolset arg above).
	systemPrompt *string,
) (out RunResult) {
	// Issue #102 part 1: stamp the post-run conversation history onto the
	// terminal Success/Failure result. The loop mutates `session` in place
	// (appending assistant / tool-result turns); reading it in a defer captures
	// the final state at whichever return fired, so every Success/Failure exit
	// carries lossless history without threading SessionState through ~30
	// construction sites. WaitingForHuman/Escalate already carry session state
	// via their PausedState, so they are intentionally left untouched.
	defer func() {
		if out.Kind == RunSuccess || out.Kind == RunFailure {
			out.SessionState = session
		}
	}()
	sessionID := task.SessionID
	// Resolve the effective tool registry once per turn-loop window. Bridges the
	// per-node toolset catalogue (or the global catalogue when the handle is
	// empty) per-run (re-injects its ToolContext with this run's SessionID +
	// storage); otherwise returns the injected ToolRegistry seam.
	toolRegistry := h.effectiveToolRegistry(sessionID, toolset)
	startedAt := time.Now()
	usage := AggregateUsage{}

	// Adaptive prompt-based tool-calling fallback (#111). Reset the shared flag
	// at the start of this window so each Run begins on the native path. The
	// conversational preset wires this pointer (and the matching
	// AdaptiveToolCallModelInterface over the agent's model); it is nil for
	// non-conversational construction, which disables the escalation seam.
	if h.config.PromptToolCallFlag != nil {
		h.config.PromptToolCallFlag.Store(false)
	}

	// Monotonic per-run span counter for tool-call span ids, and the parent
	// span id of the most recent turn (parents this turn's tool-call spans).
	var spanSeq uint64
	var currentTurnSpanID string

	// Per-run Stop-hook block counter (issue #69). Resets each Run call — a
	// resume starts fresh. After MaxStopBlocks consecutive blocks the loop
	// terminates anyway.
	var stopBlocks uint32

	// Consecutive-recoverable-tool-error breaker state (issue #137), keyed by tool
	// name. Loop-local to this window: a fresh Run / re-driven Continue window
	// starts with a clean N/2N allowance (AC3 Continue reset). errorLoopN is N (a
	// nil config field defaults to 3; an explicit 0 disables); the hard stop is at
	// 2*N.
	errorLoopN := h.config.effectiveErrorLoopThreshold()
	errorRuns := map[string]*errorRun{}

	// Output-schema enforcement state (issue #139), loop-local to this window.
	// outputSchemaRetriesUsed counts the EXTRA retry turns spent on validation
	// feedback; the budget for them is outputSchemaMaxRetries (the N).
	// lastSchemaError holds the most recent frozen validator error so a final
	// exhaustion can report it. Both are inert when outputSchema is nil
	// (enforcement OFF / no schema).
	var outputSchemaRetriesUsed uint32
	var lastSchemaError string

	effectiveTurnCap := maxIterations
	if task.Budget.MaxTurns != nil && *task.Budget.MaxTurns < effectiveTurnCap {
		effectiveTurnCap = *task.Budget.MaxTurns
	}

	for {
		// Layer-1 budget gates before the turn.
		if budget.Turns >= effectiveTurnCap {
			return RunResult{
				Kind:      RunFailure,
				Reason:    HaltReason{Kind: HaltBudgetExceeded, LimitType: BudgetLimitTurns},
				SessionID: sessionID,
				Usage:     usage,
				Turns:     budget.Turns,
			}
		}
		if lt, over := budgetExceeded(task.Budget, budget, startedAt); over {
			return RunResult{
				Kind:      RunFailure,
				Reason:    HaltReason{Kind: HaltBudgetExceeded, LimitType: lt},
				SessionID: sessionID,
				Usage:     usage,
				Turns:     budget.Turns,
			}
		}

		// Middleware: BeforeTurn (rich chain, issue #158). The chain may mutate
		// `session` in place (priority-ordered fan-out); ContinueWithModification
		// is the modified-but-proceed signal. A non-nil error is coerced to a
		// defensive middleware_halt (the Rust/TS/Py chains have no error channel —
		// this preserves parity).
		if h.config.Middleware != nil {
			d, err := h.config.Middleware.FireBeforeTurn(ctx, &session, budget.Turns)
			switch {
			case err != nil:
				return RunResult{
					Kind: RunFailure,
					Reason: HaltReason{
						Kind:   HaltMiddlewareHalt,
						Hook:   HookBeforeTurn,
						Reason: "middleware error: " + err.Error(),
					},
					SessionID: sessionID, Usage: usage, Turns: budget.Turns,
				}
			case d.Kind == MiddlewareContinue || d.Kind == MiddlewareContinueWithModification:
				// proceed
			case d.Kind == MiddlewareHalt:
				return RunResult{
					Kind: RunFailure,
					Reason: HaltReason{
						Kind:   HaltMiddlewareHalt,
						Hook:   HookBeforeTurn,
						Reason: d.Reason,
					},
					SessionID: sessionID, Usage: usage, Turns: budget.Turns,
				}
			case d.Kind == MiddlewareSurfaceToHuman:
				req := HumanRequest{}
				if d.Request != nil {
					req = *d.Request
				}
				state := &PausedState{
					SessionID: sessionID, TaskID: task.ID, TurnNumber: budget.Turns,
					SessionState: session, PendingToolCalls: nil, ApprovedResults: nil,
					HumanRequest: &req, Task: task, BudgetUsed: budget, ChildState: nil,
					// #140: carry this leaf's toolset handle so resume routes
					// through its scoped catalogue.
					Toolset: toolset,
				}
				return RunResult{Kind: RunWaitingForHuman, State: state, Request: &req}
			default:
				// ForceAnotherTurn is valid only at BeforeCompletion; the standard
				// chain converts it to Halt here. Handle any out-of-place decision
				// defensively as a middleware_halt.
				return RunResult{
					Kind: RunFailure,
					Reason: HaltReason{
						Kind:   HaltMiddlewareHalt,
						Hook:   HookBeforeTurn,
						Reason: "illegal BeforeTurn decision: " + string(d.Kind),
					},
					SessionID: sessionID, Usage: usage, Turns: budget.Turns,
				}
			}
		}

		// Assemble + invoke agent for one turn. sources (issue #115 / SC-26)
		// carries the rich ContextSources into the structural assemble seam:
		// configured guides + active skills + merged memory (#163) + tool schemas
		// (composed prompt empty for now). An empty sources renders to nothing, so
		// the no-source path stays byte-identical.
		sources := h.buildContextSources(ctx, registrySchemas(toolRegistry), task.Instruction)
		c := h.config.ContextManager.Assemble(ctx, &session, &task, sources)
		// Deliver the registry's tool schemas to the model. Tool schemas are
		// owned by the ToolRegistry; the harness wires them into the assembled
		// context so the model knows what it can call. Only fill when the context
		// manager left Tools empty (the standard compaction adapter does), so a
		// context manager that deliberately sets a phase-specific tool subset is
		// preserved. Mirrors the Rust reference (harness.rs).
		if len(c.Tools) == 0 {
			c.Tools = registrySchemas(toolRegistry)
		}
		// Whether tools were advertised this turn — the precondition for the
		// adaptive prompt-based tool-calling prose-detection heuristic (#111).
		toolsAdvertised := len(c.Tools) > 0
		// Prepend the configured operating system prompt (issue #91), MERGING it
		// with any structural #115 context block the manager already placed as a
		// leading System message (guides/active skills/memory/composed prompt).
		// The system prompt always leads. When the manager produced NO System
		// message (the common case — empty sources, or a test double), this
		// inserts a fresh System message exactly as before, so the
		// no-structural-block path stays byte-identical. The HasPrefix guard keeps
		// a resumed/seeded session that already leads with the system prompt from
		// being given it twice. Empty SystemPrompt (the default) is a no-op.
		//
		// SC-10: a leaf's per-node systemPrompt override WINS over the global
		// config.SystemPrompt. When this window's leaf carries one (non-nil), it
		// is used EXCLUSIVELY (the global prompt does not leak in), so the plan
		// and execute phases of a PlanExecuteConfig can run under distinct
		// prompts. With no leaf override (nil) the global prompt applies exactly
		// as before — byte-identical.
		effectiveSystemPrompt := h.config.SystemPrompt
		if systemPrompt != nil {
			effectiveSystemPrompt = *systemPrompt
		}
		if effectiveSystemPrompt != "" {
			if len(c.Messages) > 0 && c.Messages[0].Role == RoleSystem &&
				c.Messages[0].Content.Type == ContentTypeText {
				text := c.Messages[0].Content.Text
				if !strings.HasPrefix(text, effectiveSystemPrompt) {
					c.Messages[0].Content = NewTextContent(effectiveSystemPrompt + "\n\n" + text)
				}
				// Already leads with the system prompt — leave it.
			} else if len(c.Messages) > 0 && c.Messages[0].Role == RoleSystem {
				// A non-text System block leads — leave it.
			} else {
				c.Messages = append([]Message{{
					Role:    RoleSystem,
					Content: NewTextContent(effectiveSystemPrompt),
				}}, c.Messages...)
			}
		}
		// Per-run model params win unconditionally (issue #93). The agent copies
		// Context.Params verbatim into the ModelRequest (IntoRequest /
		// IntoRequestStreaming), so this is the single seam that delivers the
		// configured params (e.g. structured tool calls) to every tool-requesting
		// ReAct / execute / streaming turn.
		c.Params = h.config.ModelParams
		// Issue #139: route the enforced output schema into this turn's
		// constrained-decoding channel (ModelParams.OutputSchema). Ollama honors it
		// via `format`; Anthropic/OpenAI ignore it (a no-op, like
		// StructuredToolCalls). nil leaves the params byte-identical to pre-#139.
		if outputSchema != nil {
			c.Params.OutputSchema = outputSchema
		}
		emit(onStream, HarnessStreamEvent{Kind: HarnessStreamTurnStart, Turn: budget.Turns + 1})
		turnStartedAt := nowRFC3339()
		turnClock := time.Now()
		result := h.runStreamingTurn(ctx, agent, c, onStream)
		budget.Turns++
		// Emit a turn span for every model call (issue #12). Fire-and-forget;
		// it never affects control flow. The span id is retained as the
		// parent for any tool-call spans dispatched this turn. EmitMiddleware/
		// EmitSensor/EmitContext are intentionally NOT wired here: middleware
		// spans are a separate forward-decl and there are no sensor/context
		// call sites in the ReAct loop; EmitPatch is handled in the patch
		// middleware. This mirrors the Rust reference's loop emission.
		turnSpanID := fmt.Sprintf("%s-turn-%d", sessionID, budget.Turns)
		if h.config.Observability != nil {
			var u TokenUsage
			if result.Usage != nil {
				u = *result.Usage
			}
			var (
				stopReason         StopReason
				toolCallsRequested uint32
				errMsg             string
			)
			switch result.Kind {
			case TurnFinalResponse:
				stopReason = StopEndTurn
			case TurnToolCallRequested:
				stopReason = StopToolUse
				toolCallsRequested = uint32(len(result.Calls))
			case TurnError:
				stopReason = StopEndTurn
				if result.Err != nil {
					errMsg = result.Err.Error()
				}
			default:
				stopReason = StopEndTurn
			}
			h.config.Observability.EmitTurn(
				turnSpanID,
				sessionID,
				task.ID,
				budget.Turns,
				turnStartedAt,
				uint64(time.Since(turnClock).Milliseconds()),
				u,
				h.config.Observability.CostFor(u),
				stopReason,
				toolCallsRequested,
				errMsg,
				result.Content,
				result.Calls,
				c.Messages,
			)
		}
		currentTurnSpanID = turnSpanID
		emit(onStream, HarnessStreamEvent{Kind: HarnessStreamTurnEnd, Turn: budget.Turns})

		switch result.Kind {
		case TurnFinalResponse:
			u := *result.Usage
			usage.AddTurn(u)
			budget.InputTokens += uint64(u.InputTokens)
			budget.OutputTokens += uint64(u.OutputTokens)

			// Adaptive prompt-based tool-calling escalation (#111). The model
			// answered in prose. If tools were advertised and the text shows
			// action intent (DetectProseResponse), flip the shared flag — which
			// activates the AdaptiveToolCallModelInterface for the rest of the
			// run — record the prose as the assistant's turn, nudge the model
			// with the tool-call format, and force another turn. One-shot: the
			// flag is only flipped while still unset, bounding this to exactly one
			// extra turn per window.
			if h.config.PromptToolCallFlag != nil && !h.config.PromptToolCallFlag.Load() {
				if prose, ok := DetectProseResponse(result.Content, toolsAdvertised); ok {
					h.config.PromptToolCallFlag.Store(true)
					if a, ok := h.config.ContextManager.(AssistantMessageAppender); ok {
						a.AppendAssistantMessage(ctx, &session, Message{Role: RoleAssistant, Content: NewTextContent(prose)})
					}
					h.config.ContextManager.AppendUserMessage(ctx, &session, PromptToolCallNudge)
					continue
				}
			}

			if h.config.Middleware != nil {
				d, err := h.config.Middleware.FireBeforeCompletion(ctx, result.Content, budget.Turns, &session)
				switch {
				case err != nil:
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookBeforeCompletion,
							Reason: "middleware error: " + err.Error(),
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				case d.Kind == MiddlewareContinue || d.Kind == MiddlewareContinueWithModification:
					// proceed to completion
				case d.Kind == MiddlewareForceAnotherTurn:
					// The chain concatenates every middleware's injection into one
					// ForceAnotherTurn (issue #11 / #158). Record the model's final
					// text, then the injection as a user message, and force another
					// turn instead of completing — the same channel the Stop-block
					// breaker uses, so the conversation stays well-formed (assistant
					// final text → user injection).
					if a, ok := h.config.ContextManager.(AssistantMessageAppender); ok {
						a.AppendAssistantMessage(ctx, &session, Message{Role: RoleAssistant, Content: NewTextContent(result.Content)})
					}
					h.config.ContextManager.AppendUserMessage(ctx, &session, d.Inject)
					continue
				case d.Kind == MiddlewareHalt:
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookBeforeCompletion,
							Reason: d.Reason,
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				case d.Kind == MiddlewareSurfaceToHuman:
					req := HumanRequest{}
					if d.Request != nil {
						req = *d.Request
					}
					state := &PausedState{
						SessionID: sessionID, TaskID: task.ID, TurnNumber: budget.Turns,
						SessionState: session, PendingToolCalls: nil, ApprovedResults: nil,
						HumanRequest: &req, Task: task, BudgetUsed: budget, ChildState: nil,
						// #140: carry this leaf's toolset handle so resume routes
						// through its scoped catalogue.
						Toolset: toolset,
					}
					return RunResult{Kind: RunWaitingForHuman, State: state, Request: &req}
				default:
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookBeforeCompletion,
							Reason: "illegal BeforeCompletion decision: " + string(d.Kind),
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				}
			}

			td := h.config.TerminationPolicy.Evaluate(ctx, &session, &budget)
			if td.Kind == TerminationHalt {
				return RunResult{
					Kind: RunFailure,
					Reason: HaltReason{
						Kind:   HaltTerminationPolicyHalt,
						Reason: td.Reason,
					},
					SessionID: sessionID, Usage: usage, Turns: budget.Turns,
				}
			}
			// Record the assistant's final text in history so a continued
			// session reflects what the agent said (multi-turn / S2 correctness).
			if a, ok := h.config.ContextManager.(AssistantMessageAppender); ok {
				a.AppendAssistantMessage(ctx, &session, Message{Role: RoleAssistant, Content: NewTextContent(result.Content)})
			}

			// Stop hook (issue #69). The strategy believes the task is done;
			// fire registered Stop hooks synchronously. If any blocks (and we
			// are under MaxStopBlocks), inject the reason via the same path
			// ForceAnotherTurn uses and continue the loop instead of
			// terminating.
			if reason, blocked := h.fireStopHooks(ctx, sessionID, &task, budget.Turns, result.Content, &session, &stopBlocks); blocked {
				h.config.ContextManager.AppendUserMessage(ctx, &session, reason)
				continue
			}

			// Output-schema enforcement gate (issue #139). Additive to and EARLIER
			// than the Success terminal: when a schema is active, validate this
			// terminal FinalResponse. On a validation FAILURE, feed the frozen error
			// back as a USER message and retry one more turn (up to
			// outputSchemaMaxRetries extra turns). The retried turn re-enters the loop
			// top, where the budget gate fires FIRST — so a retry that would exceed the
			// turn/budget backstop surfaces the existing HaltBudgetExceeded terminal,
			// NOT HaltOutputSchemaViolation (budget-cap-wins precedence). Only when the
			// N retries are exhausted WITH budget remaining does the window terminate
			// with HaltOutputSchemaViolation. A nil schema ⇒ no validation (migration
			// gate OFF / no Output), so the terminal is byte-identical to pre-#139.
			if outputSchema != nil {
				if verr := validateOutput(result.Content, outputSchema); verr != "" {
					lastSchemaError = verr
					if outputSchemaRetriesUsed < outputSchemaMaxRetries {
						// Grant one more turn: feed the validator error back as a USER
						// message (the assistant's invalid text is already in history
						// above) and loop. The budget gate at the loop top enforces the
						// budget-cap-wins precedence.
						outputSchemaRetriesUsed++
						emit(onStream, HarnessStreamEvent{
							Kind:    HarnessStreamOutputSchemaRetry,
							Attempt: outputSchemaRetriesUsed,
							Content: verr,
						})
						h.config.ContextManager.AppendUserMessage(ctx, &session, feedbackMessage(verr))
						continue
					}
					// Retries exhausted WITH budget remaining (the budget gate did not
					// fire) → the typed schema-violation terminal. Total attempts =
					// 1 + max_retries.
					attempts := outputSchemaMaxRetries + 1
					emit(onStream, HarnessStreamEvent{
						Kind:     HarnessStreamOutputSchemaViolation,
						Attempts: attempts,
						Content:  lastSchemaError,
					})
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:      HaltOutputSchemaViolation,
							Schema:    canonicalizeSchema(outputSchema),
							Attempts:  attempts,
							LastError: lastSchemaError,
						},
						SessionID: sessionID,
						Usage:     usage,
						Turns:     budget.Turns,
					}
				}
			}

			emit(onStream, HarnessStreamEvent{Kind: HarnessStreamFinalResponse, Content: result.Content})
			return RunResult{
				Kind:      RunSuccess,
				Output:    result.Content,
				SessionID: sessionID,
				Usage:     usage,
				Turns:     budget.Turns,
			}

		case TurnToolCallRequested:
			u := *result.Usage
			usage.AddTurn(u)
			budget.InputTokens += uint64(u.InputTokens)
			budget.OutputTokens += uint64(u.OutputTokens)

			calls := result.Calls

			// Always-halt short circuit (Layer 1).
			for _, c := range calls {
				if toolRegistry.IsAlwaysHalt(c.Name) {
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:  HaltUnrecoverableToolError,
							Tool:  c.Name,
							Error: "tool is annotated always_halt",
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				}
			}

			// Record the assistant's turn (the tool calls the model requested)
			// as soon as the calls are known — BEFORE the BeforeTool middleware
			// (which may pause via SurfaceToHuman) and before any tool result.
			// This keeps the conversation well-formed (assistant tool_use
			// precedes its tool result) on every path, including human-in-the-
			// loop resume, so the resume path never has to append it. The
			// recorded turn reflects the model's original request; a middleware
			// or human modification changes only what is dispatched.
			if a, ok := h.config.ContextManager.(AssistantMessageAppender); ok {
				for _, call := range calls {
					a.AppendAssistantMessage(ctx, &session, Message{Role: RoleAssistant, Content: NewToolCallContent(call)})
				}
			}

			// Middleware: BeforeTool (rich chain, issue #158 / SC-11). The chain
			// mutates the dispatched `calls` IN PLACE via a priority-ordered
			// fan-out; ContinueWithModification is the modified-but-proceed signal.
			// We pass a fresh copy of the calls so the assistant turn recorded just
			// above (which value-copied each call) keeps the model's ORIGINAL
			// request — only what is dispatched changes.
			if h.config.Middleware != nil {
				calls = append([]ToolCall(nil), calls...)
				d, err := h.config.Middleware.FireBeforeTool(ctx, &calls, budget.Turns)
				switch {
				case err != nil:
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookBeforeTool,
							Reason: "middleware error: " + err.Error(),
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				case d.Kind == MiddlewareContinue || d.Kind == MiddlewareContinueWithModification:
					// `calls` was mutated in place by the chain; dispatch it as-is.
				case d.Kind == MiddlewareHalt:
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookBeforeTool,
							Reason: d.Reason,
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				case d.Kind == MiddlewareSurfaceToHuman:
					req := HumanRequest{}
					if d.Request != nil {
						req = *d.Request
					}
					state := &PausedState{
						SessionID: sessionID, TaskID: task.ID, TurnNumber: budget.Turns,
						SessionState: session, PendingToolCalls: calls, ApprovedResults: nil,
						HumanRequest: &req, Task: task, BudgetUsed: budget, ChildState: nil,
						// #140: carry this leaf's toolset handle so resume routes
						// through its scoped catalogue.
						Toolset: toolset,
					}
					return RunResult{Kind: RunWaitingForHuman, State: state, Request: &req}
				default:
					// ForceAnotherTurn is valid only at BeforeCompletion; the
					// standard chain converts it to Halt here. Handle any
					// out-of-place decision defensively as a middleware_halt.
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookBeforeTool,
							Reason: "illegal BeforeTool decision: " + string(d.Kind),
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				}
			}

			var approved []HarnessToolResult
			// SC-9: the session.Messages index of each appended tool result,
			// recorded 1:1 with `approved`. The AfterTool middleware hook uses
			// these to re-render any result it rewrites in place (via the optional
			// ToolResultReplacer seam). Captured at append time, so the indices
			// survive the #137 corrective-message interleaving.
			var resultMsgIndices []int
			toolPause := false
			for i, call := range calls {
				// Sandbox validation.
				if v := h.config.Sandbox.Validate(ctx, call); v != nil {
					if v.IsAlwaysHalt() {
						return RunResult{
							Kind: RunFailure,
							Reason: HaltReason{
								Kind:      HaltSandboxViolation,
								Violation: v,
							},
							SessionID: sessionID, Usage: usage, Turns: budget.Turns,
						}
					}
					// Layer-2 recoverable: append as tool error.
					tr := HarnessToolResult{
						CallID: call.ID,
						Output: ToolOutput{
							Kind:        ToolOutputError,
							Message:     "sandbox: " + v.Error(),
							Recoverable: true,
						},
					}
					emit(onStream, HarnessStreamEvent{
						// Q5: carry the result content.
						Kind: HarnessStreamToolResult, CallID: call.ID, IsError: true,
						ResultContent: "sandbox: " + v.Error(),
					})
					h.config.ContextManager.AppendToolResult(ctx, &session, &tr)
					resultMsgIndices = append(resultMsgIndices, lastMessageIndex(&session))
					approved = append(approved, tr)
					continue
				}

				emit(onStream, HarnessStreamEvent{
					// Q5: carry the final tool-call arguments.
					Kind: HarnessStreamToolCall, CallID: call.ID, Name: call.Name, Args: call.Input,
				})
				toolStartedAt := nowRFC3339()
				toolClock := time.Now()
				// #126 AC2: record a harness-OBSERVED write/edit at the dispatch seam
				// for the call ACTUALLY dispatched, so a task's files_touched ledger is
				// built from real tool calls — never a model-self-reported field.
				h.observeWriteCall(call)
				output := dispatchAndUnwrap(ctx, toolRegistry, h.config.Sandbox, call)

				// Clarification pause (issue #81, Q4b): a tool (e.g.
				// ask_user_question) needs a human answer before it can produce a
				// result. UNLIKE the subagent WaitingForHuman path there is NO
				// ChildPausedState: the loop builds a PausedState directly with
				// HumanRequest set to Clarification. The CLARIFYING call itself is
				// preserved as the HEAD of PendingToolCalls (followed by the
				// remaining batch) so that, on resume, the human's answer is
				// injected as the tool RESULT for that pending call.
				if output.Kind == ToolOutputAwaitingClarification {
					pending := append([]ToolCall{call}, calls[i+1:]...)
					req := HumanRequest{
						Kind:     HumanReqClarification,
						Question: output.Question,
						Options:  output.Options,
					}
					state := &PausedState{
						SessionID: sessionID, TaskID: task.ID, TurnNumber: budget.Turns,
						SessionState: session, PendingToolCalls: pending,
						ApprovedResults: approved, HumanRequest: &req, Task: task,
						BudgetUsed: budget, ChildState: nil,
						// #140: carry this leaf's toolset handle so the preserved
						// clarifying call resumes against its scoped catalogue.
						Toolset: toolset,
					}
					return RunResult{Kind: RunWaitingForHuman, State: state, Request: &req}
				}

				// Consult pause (issue #114, R1/R10): a worker-side tool returns
				// ToolOutput.Consult (ChildState nil) to ask for mid-loop help.
				// Like the AwaitingClarification arm there is NO ChildPausedState
				// at this level — the loop builds a PausedState directly with
				// HumanRequest nil and preserves the CONSULTING call as the HEAD of
				// PendingToolCalls (followed by the remaining batch), so that on
				// ResumeConsult the helper's answer is injected as the tool RESULT
				// for that pending call. The consult is a control signal, NOT a
				// conversation turn: it is never appended to message history here
				// (R10).
				if output.Kind == ToolOutputConsult {
					pending := append([]ToolCall{call}, calls[i+1:]...)
					req := ConsultRequest{}
					if output.ConsultRequest != nil {
						req = *output.ConsultRequest
					}
					// Observability: lightweight consult-spawn event (issue #114).
					if h.config.Observability != nil {
						h.config.Observability.EmitConsultSpawned(
							fmt.Sprintf("%s-consult-spawn-%s", sessionID, call.ID),
							sessionID, task.ID, nowRFC3339(), req.Kind,
						)
					}
					state := &PausedState{
						SessionID: sessionID, TaskID: task.ID, TurnNumber: budget.Turns,
						SessionState: session, PendingToolCalls: pending,
						ApprovedResults: approved, HumanRequest: nil, Task: task,
						BudgetUsed: budget, ChildState: nil,
						// #140 (THE load-bearing path): carry this leaf's toolset
						// handle so ResumeConsult routes the preserved consulting
						// call through its scoped catalogue instead of the global
						// fallback.
						Toolset: toolset,
					}
					return RunResult{
						Kind:           RunConsult,
						ConsultRequest: &req,
						State:          state,
						SessionID:      sessionID,
						Usage:          usage,
						Turns:          budget.Turns,
					}
				}

				// SendMessage (issue #81): the send_message tool surfaces an
				// out-of-band message to the user. The loop emits a UserMessage
				// stream event rather than collapsing the content into a normal
				// tool result, then records the (minimal) success result so the
				// loop continues. Recognized by tool name.
				if call.Name == sendMessageToolName && output.Kind == ToolOutputSuccess {
					emit(onStream, HarnessStreamEvent{
						Kind: HarnessStreamUserMessage, Content: output.Content,
					})
				}

				// Pause propagation: WaitingForHuman from a subagent tool.
				if output.Kind == ToolOutputWaitingForHuman {
					remaining := append([]ToolCall(nil), calls[i+1:]...)
					req := HumanRequest{}
					if output.Request != nil {
						req = *output.Request
					}
					state := &PausedState{
						SessionID: sessionID, TaskID: task.ID, TurnNumber: budget.Turns,
						SessionState: session, PendingToolCalls: remaining,
						ApprovedResults: approved, HumanRequest: &req, Task: task,
						BudgetUsed: budget, ChildState: output.ChildState,
						// #140: the parent leaf's toolset handle (the child carries
						// its own inside ChildState).
						Toolset: toolset,
					}
					_ = toolPause
					return RunResult{Kind: RunWaitingForHuman, State: state, Request: &req}
				}

				// Escalation propagation (issue #80): a tool requests a
				// structural state change. The harness is a pure intermediary —
				// it does NOT act on the signal. It terminates cleanly,
				// preserves session state for a possible resume, and returns
				// RunResult.Escalate. The escalation is a control signal, NOT a
				// conversation turn: it is never appended to message history,
				// and the remaining tool calls in this batch are preserved into
				// PendingToolCalls (mirroring WaitingForHuman). The signal is NOT
				// stored in PausedState, so it is discarded on resume — the
				// harness never re-acts on it. HumanRequest is nil: an
				// escalation has no human request.
				if output.Kind == ToolOutputEscalate {
					remaining := append([]ToolCall(nil), calls[i+1:]...)
					signal := HarnessSignal{}
					if output.Signal != nil {
						signal = *output.Signal
					}
					state := &PausedState{
						SessionID: sessionID, TaskID: task.ID, TurnNumber: budget.Turns,
						SessionState: session, PendingToolCalls: remaining,
						ApprovedResults: approved, HumanRequest: nil, Task: task,
						BudgetUsed: budget, ChildState: nil,
						// #140: carry this leaf's toolset handle so resume routes
						// pending per-node calls through its catalogue.
						Toolset: toolset,
					}
					return RunResult{
						Kind:      RunEscalate,
						Signal:    &signal,
						State:     state,
						SessionID: sessionID,
						Usage:     usage,
						Turns:     budget.Turns,
					}
				}

				isError := output.Kind == ToolOutputError
				// Layer-2: unrecoverable tool error halts immediately.
				if output.Kind == ToolOutputError && !output.Recoverable {
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:  HaltUnrecoverableToolError,
							Tool:  call.Name,
							Error: output.Message,
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				}

				// ── Consecutive-recoverable-tool-error breaker (#137) ──
				// output here is either a Success or a RECOVERABLE Error (every other
				// variant early-returned above). On a success the tool's error run
				// resets (AC1); on a recoverable error we increment the identical-args
				// run (AC1 args-variant) and check the N / 2N thresholds. At N we stash
				// the corrective USER message here and append it AFTER the tool result
				// below (well-formed conversation: assistant tool_use -> tool result ->
				// corrective user message).
				var pendingCorrective string
				if output.Kind == ToolOutputError && output.Recoverable {
					run, ok := errorRuns[call.Name]
					if ok && sameToolArgs(run.args, call.Input) {
						// Same tool, structurally-identical args -> extend the run.
						run.count++
					} else {
						// First error, or the args changed -> fresh run.
						run = &errorRun{args: call.Input, count: 1}
						errorRuns[call.Name] = run
					}
					count := run.count
					twoN := errorLoopN * 2

					// 2N: HARD STOP (AC3). Do NOT append this last tool result or
					// continue — return a typed terminal that ReactConfig.Run routes
					// through the node's BudgetExhaustedBehavior WITHOUT burning the
					// rest of the budget. The breaker is disabled when N == 0.
					if errorLoopN > 0 && count >= twoN {
						// AC4: emit the "broken" pair (stream + obs).
						emit(onStream, HarnessStreamEvent{
							Kind: HarnessStreamToolErrorLoopBroken,
							Tool: call.Name, ConsecutiveErrors: count,
						})
						if h.config.Observability != nil {
							h.config.Observability.EmitToolErrorLoopBroken(
								fmt.Sprintf("%s-tool-error-loop-broken-%d", sessionID, spanSeq),
								sessionID, task.ID, nowRFC3339(), call.Name, count,
							)
						}
						return RunResult{
							Kind: RunFailure,
							Reason: HaltReason{
								Kind:              HaltToolErrorLoop,
								Tool:              call.Name,
								ConsecutiveErrors: count,
							},
							SessionID: sessionID, Usage: usage, Turns: budget.Turns,
						}
					}

					// N: inject ONE corrective message (AC2). Render the schema+hint
					// via enrichToolError (reused) and inject it as a USER-role message
					// — the SAME channel the Stop-block breaker uses — once per run (do
					// not re-inject between N and 2N).
					if errorLoopN > 0 && count >= errorLoopN && !run.injected {
						run.injected = true
						var schema *ToolSchema
						for _, s := range registrySchemas(toolRegistry) {
							if s.Name == call.Name {
								sc := s
								schema = &sc
								break
							}
						}
						// enrichToolError renders the bare message + schema + hint; the
						// bare message is this run's own error text.
						corrective := enrichToolError(output.Message, schema).Message
						// AC4: emit the "detected/warning" pair (stream + obs) BEFORE
						// the corrective message is appended.
						emit(onStream, HarnessStreamEvent{
							Kind: HarnessStreamToolErrorLoopDetected,
							Tool: call.Name, ConsecutiveErrors: count,
						})
						if h.config.Observability != nil {
							h.config.Observability.EmitToolErrorLoopDetected(
								fmt.Sprintf("%s-tool-error-loop-detected-%d", sessionID, spanSeq),
								sessionID, task.ID, nowRFC3339(), call.Name, count,
							)
						}
						// Append after the normal tool-result append below so the
						// conversation stays well-formed (assistant tool_use -> tool
						// result -> corrective user msg).
						pendingCorrective = corrective
					}
				} else {
					// AC1: ANY success for this tool resets its run.
					delete(errorRuns, call.Name)
				}

				// Tool-call span (issue #12), child of the current turn.
				if h.config.Observability != nil {
					var outputSize uint64
					switch output.Kind {
					case ToolOutputSuccess:
						outputSize = uint64(len(output.Content))
					case ToolOutputError:
						outputSize = uint64(len(output.Message))
					}
					var paramsSize uint64
					if call.Input != nil {
						paramsSize = uint64(len(call.Input))
					}
					var resultContent string
					switch output.Kind {
					case ToolOutputSuccess:
						resultContent = output.Content
					case ToolOutputError:
						resultContent = output.Message
					}
					toolSpanID := fmt.Sprintf("%s-tool-%d", sessionID, spanSeq)
					h.config.Observability.EmitToolCall(
						toolSpanID,
						currentTurnSpanID,
						sessionID,
						task.ID,
						call.Name,
						call.ID,
						toolStartedAt,
						uint64(time.Since(toolClock).Milliseconds()),
						paramsSize,
						outputSize,
						output.Truncated,
						isError,
						call.Input,
						resultContent,
					)
					spanSeq++
				}

				tr := HarnessToolResult{CallID: call.ID, Output: output}
				// Q5: carry the result content on the coarse tool_result event.
				var streamResultContent string
				switch output.Kind {
				case ToolOutputSuccess:
					streamResultContent = output.Content
				case ToolOutputError:
					streamResultContent = output.Message
				}
				emit(onStream, HarnessStreamEvent{
					Kind: HarnessStreamToolResult, CallID: call.ID, IsError: isError,
					ResultContent: streamResultContent,
				})
				h.config.ContextManager.AppendToolResult(ctx, &session, &tr)
				resultMsgIndices = append(resultMsgIndices, lastMessageIndex(&session))
				approved = append(approved, tr)

				// #137 (AC2): inject the N-threshold corrective USER message AFTER
				// the tool result so the conversation stays well-formed (assistant
				// tool_use -> tool result -> corrective user message). Same channel
				// the Stop-block breaker uses.
				if pendingCorrective != "" {
					h.config.ContextManager.AppendUserMessage(ctx, &session, pendingCorrective)
				}
			}

			// Middleware: AfterTool (rich chain, issue #158 / SC-9). The chain
			// receives the batch's results as a mutable model-side []ToolResult and
			// may rewrite any of them in place (priority-ordered, descending). On
			// ContinueWithModification, map the rewrites back onto the harness-side
			// `approved` results AND re-render the affected tool-result messages (via
			// the optional ToolResultReplacer seam) so the rewrite reaches the next
			// model turn — this is what lets an after-tool middleware turn a landed
			// write into a model-visible error (or vice versa) without the tool
			// itself having to invert its output.
			if h.config.Middleware != nil {
				results := harnessResultsToToolResults(approved)
				d, err := h.config.Middleware.FireAfterTool(ctx, calls, &results)
				switch {
				case err != nil:
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookAfterTool,
							Reason: "middleware error: " + err.Error(),
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				case d.Kind == MiddlewareContinue:
					// proceed
				case d.Kind == MiddlewareContinueWithModification:
					// Fold the (possibly rewritten) model-side results back into the
					// harness-side `approved`, then re-render the recorded session
					// messages 1:1 by the indices captured at append time.
					applyToolResultRewrites(approved, results)
					if replacer, ok := h.config.ContextManager.(ToolResultReplacer); ok {
						for i := range approved {
							if i >= len(resultMsgIndices) {
								break
							}
							replacer.ReplaceToolResult(ctx, &session, resultMsgIndices[i], &approved[i])
						}
					}
				case d.Kind == MiddlewareHalt:
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookAfterTool,
							Reason: d.Reason,
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				default:
					// SurfaceToHuman / ForceAnotherTurn are not valid at AfterTool;
					// the standard chain converts them to Halt. Handle any
					// out-of-place decision defensively as a middleware_halt.
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookAfterTool,
							Reason: "illegal AfterTool decision: " + string(d.Kind),
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				}
			}

			// Compaction (issue #46): after tool results are appended and the
			// AfterTool middleware has fired, before the loop restarts — matches
			// the concepts-doc loop diagram's "compact if should_compact()"
			// placement. Runs the verify→retry→warn loop; never halts the run.
			if h.config.ContextManager.ShouldCompact(&session) {
				h.runCompaction(ctx, &session, sessionID, task.ID, &spanSeq, &usage, agent)
			}

			// loop again
			continue

		case TurnError:
			if result.Usage != nil {
				usage.AddTurn(*result.Usage)
				budget.InputTokens += uint64(result.Usage.InputTokens)
				budget.OutputTokens += uint64(result.Usage.OutputTokens)
			}
			return RunResult{
				Kind: RunFailure,
				Reason: HaltReason{
					Kind:       HaltAgentError,
					AgentError: result.Err,
				},
				SessionID: sessionID, Usage: usage, Turns: budget.Turns,
			}
		default:
			return RunResult{
				Kind: RunFailure,
				Reason: HaltReason{
					Kind:       HaltAgentError,
					AgentError: NewEmptyResponseError(),
				},
				SessionID: sessionID, Usage: usage, Turns: budget.Turns,
			}
		}
	}
}

// runCompaction runs the post-compaction verify→retry→warn loop (issue
// #46/#29).
//
// It drives one compaction turn through the agent, verifies the summary, and
// either accepts it, retries with the missing items injected, or — after
// MaxCompactionAttempts (clamped to ≥1) — emits a warn event and accepts the
// summary anyway. A blocked compaction is worse than an imperfect one, so this
// method NEVER returns an error or halts the run; the worst case is an
// accepted-anyway summary plus one warn span.
//
// Token usage from compaction turns folds into the run-level AggregateUsage;
// each accepted summary is surfaced as a Compaction context span. The
// compaction_verification_failures metric is derived from the emitted warn
// span. Compaction is skipped entirely if the held ContextManager does not
// implement CompactingContextManager.
func (h *StandardHarness) runCompaction(
	ctx context.Context,
	session *SessionState,
	sessionID SessionID,
	taskID TaskID,
	spanSeq *uint64,
	usage *AggregateUsage,
	// #124: the compaction summary turn runs on the RESOLVED worker agent, not a
	// default config agent.
	agent Agent,
) {
	cm, ok := h.config.ContextManager.(CompactingContextManager)
	if !ok {
		// Manager does not support compaction — skip (Go equivalent of the
		// Rust default-bodied trait methods).
		return
	}
	turn, ok := cm.PrepareCompactionTurn(session)
	if !ok || turn == nil {
		// Nothing to compact (e.g. history shorter than preserve window).
		return
	}

	// Capture the pre-compaction token budget for span stamping (issue #57).
	// Managers that do not track tokens report (0, false); tokensBefore then
	// stays 0 and the span reports zero reclamation (the old behavior).
	var tokensBefore uint32
	if reader, ok := h.config.ContextManager.(TokenBudgetReader); ok {
		if v, present := reader.TokenBudgetUsed(session); present {
			tokensBefore = v
		}
	}

	maxAttempts := h.config.MaxCompactionAttempts
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	for attempt := uint32(1); ; attempt++ {
		// Run one compaction turn through the agent to produce a summary.
		result := agent.Turn(ctx, turn.Context)
		var summary string
		switch result.Kind {
		case TurnFinalResponse:
			if result.Usage != nil {
				usage.AddTurn(*result.Usage)
			}
			summary = result.Content
		case TurnToolCallRequested:
			// A compaction turn is expected to yield a summary, not a tool
			// call. Treat the (empty) response as the summary so verification
			// can run and the loop terminates predictably.
			if result.Usage != nil {
				usage.AddTurn(*result.Usage)
			}
			summary = ""
		default: // TurnError or unknown
			if result.Usage != nil {
				usage.AddTurn(*result.Usage)
			}
			summary = ""
		}

		// Verify. A nil verifier means no gate: treat every summary as passing.
		verification := CompactionVerificationResult{Passed: true}
		if h.config.CompactionVerifier != nil {
			verification = h.config.CompactionVerifier.Verify(summary, turn)
		}

		if verification.Passed {
			cm.ApplyCompaction(session, summary)
			h.emitCompactionSpan(session, sessionID, taskID, turn.MessagesRemoved, tokensBefore, spanSeq)
			return
		}

		if attempt < maxAttempts {
			// Inject the missing items and retry.
			cm.InjectMissingItems(&turn.Context, verification.MissingItems)
			continue
		}

		// Exhausted attempts: warn, then accept anyway.
		if h.config.Observability != nil {
			warnSpanID := fmt.Sprintf("%s-warn-%d", sessionID, *spanSeq)
			h.config.Observability.EmitCompactionVerificationFailed(
				warnSpanID,
				sessionID,
				taskID,
				nowRFC3339(),
				verification.MissingItems,
				true,
			)
			*spanSeq++
		}
		cm.ApplyCompaction(session, summary)
		h.emitCompactionSpan(session, sessionID, taskID, turn.MessagesRemoved, tokensBefore, spanSeq)
		return
	}
}

// emitCompactionSpan emits the Compaction context span for an accepted summary.
// It reads the post-compaction token budget from the manager (if it implements
// TokenBudgetReader) so the span carries the real tokens_after / tokens_reclaimed
// (issue #57). Managers that track no budget fall back to tokens_after ==
// tokens_before and zero reclamation (the old behavior).
func (h *StandardHarness) emitCompactionSpan(
	session *SessionState,
	sessionID SessionID,
	taskID TaskID,
	messagesRemoved uint32,
	tokensBefore uint32,
	spanSeq *uint64,
) {
	if h.config.Observability == nil {
		return
	}
	tokensAfter := tokensBefore
	if reader, ok := h.config.ContextManager.(TokenBudgetReader); ok {
		if v, present := reader.TokenBudgetUsed(session); present {
			tokensAfter = v
		}
	}
	var tokensReclaimed uint32
	if tokensBefore > tokensAfter {
		tokensReclaimed = tokensBefore - tokensAfter
	}
	spanID := fmt.Sprintf("%s-compaction-%d", sessionID, *spanSeq)
	h.config.Observability.EmitCompaction(
		spanID,
		sessionID,
		taskID,
		nowRFC3339(),
		messagesRemoved,
		tokensBefore,
		tokensAfter,
		tokensReclaimed,
	)
	*spanSeq++
}

// Resume continues a paused run after a human response.
//
// Issue #102: like Run, Resume is the thin auto-persist seam — it delegates the
// loop to resumeInner, then persists the terminal run state to the SessionStore
// when AutoPersistSessions is enabled (one write per Resume). When disabled it
// is byte-for-byte the prior behaviour.
func (h *StandardHarness) Resume(
	ctx context.Context,
	state PausedState,
	response HumanResponse,
	onStream StreamSink,
) RunResult {
	result := h.resumeInner(ctx, state, response, onStream)
	h.autoPersistTerminal(ctx, &result)
	return result
}

// ResumeConsult resumes a worker paused by RunResult.Consult (issue #114).
// Mirrors Resume: it dispatches to resumeConsultInner, then auto-persists the
// terminal run state when enabled.
func (h *StandardHarness) ResumeConsult(
	ctx context.Context,
	state PausedState,
	response ConsultResponse,
	onStream StreamSink,
) RunResult {
	result := h.resumeConsultInner(ctx, state, response, onStream)
	h.autoPersistTerminal(ctx, &result)
	return result
}

// resumeConsultInner is the consult resume seam (issue #114). Mirrors the
// resumeInner clarification branch: the ConsultResponse text is injected as the
// tool RESULT for the head pending (consult) call — NOT appended as a
// free-standing user message (R10) — then any remaining pending calls are
// dispatched and the ReAct loop resumes.
func (h *StandardHarness) resumeConsultInner(
	ctx context.Context,
	state PausedState,
	response ConsultResponse,
	onStream StreamSink,
) RunResult {
	session := state.SessionState
	task := state.Task
	budget := state.BudgetUsed
	pending := state.PendingToolCalls

	var (
		text     string
		answered bool
	)
	switch response.Kind {
	case ConsultRespAnswer:
		text, answered = response.Text, true
	case ConsultRespBudgetExhausted:
		text, answered = response.Message, false
	}

	// Observability: lightweight consult-resume event (issue #114). Recover the
	// consult kind from the head pending call's args, if present.
	if h.config.Observability != nil {
		kind := ""
		if len(pending) > 0 && len(pending[0].Input) > 0 {
			var probe struct {
				Kind string `json:"kind"`
			}
			if json.Unmarshal(pending[0].Input, &probe) == nil {
				kind = probe.Kind
			}
		}
		h.config.Observability.EmitConsultResumed(
			fmt.Sprintf("%s-consult-resume", state.SessionID),
			state.SessionID, task.ID, nowRFC3339(), kind, answered,
		)
	}

	// #140: resume routes the preserved consulting call (and any remaining
	// pending calls) through the pausing leaf's own toolset handle, restoring its
	// scoped per-node catalogue. An empty handle (the default) still falls back to
	// the global catalogue, so pre-#140 blobs behave unchanged.
	toolRegistry := h.effectiveToolRegistry(state.SessionID, state.Toolset)

	// Inject the consult answer as the RESULT of the head pending (consult)
	// call, then dispatch the remaining pending calls in the same batch.
	if len(pending) > 0 {
		consultCall := pending[0]
		tr := HarnessToolResult{
			CallID: consultCall.ID,
			Output: ToolOutput{Kind: ToolOutputSuccess, Content: text},
		}
		h.config.ContextManager.AppendToolResult(ctx, &session, &tr)
		for _, call := range pending[1:] {
			output := dispatchAndUnwrap(ctx, toolRegistry, h.config.Sandbox, call)
			rtr := HarnessToolResult{CallID: call.ID, Output: output}
			h.config.ContextManager.AppendToolResult(ctx, &session, &rtr)
		}
	}

	// #131: a consult that surfaced from inside a composed tree carries the FULL
	// strategy in task.LoopStrategy (each combinator's finish rewrote the pause's
	// task on the way up). Re-DRIVE that strategy rather than resuming only the
	// worker leaf: the PlanExecute walk resumes its in-progress task from the
	// injected worker session (resume seed), so the worker finishes mid-loop, its
	// SelfVerifying evaluator runs, the task is marked Completed, and the remaining
	// ready-set is walked. A BARE worker leaf (depth-1, e.g. a SubagentTool-mediated
	// consult) has no surrounding walk, so it keeps the original leaf-only resume
	// (back-compat).
	if task.LoopStrategy.Kind != StrategyReAct {
		// Top-level session starts fresh; the worker conversation is threaded into
		// the in-progress task via the resume seed.
		workerSession := session
		return h.driveStrategyWithResumeSeed(
			ctx,
			task,
			SessionState{},
			budget,
			onStream,
			nil,
			&workerSession,
		)
	}

	agent, agentFail := h.ResolveWorkerAgent(&task.LoopStrategy)
	if agentFail != nil {
		af := *agentFail
		af.SessionID = task.SessionID
		return af
	}
	return h.runReAct(ctx, task, task.LoopStrategy.MaxIterations(), session, budget, onStream, agent)
}

func (h *StandardHarness) resumeInner(
	ctx context.Context,
	state PausedState,
	response HumanResponse,
	onStream StreamSink,
) RunResult {
	session := state.SessionState
	task := state.Task
	budget := state.BudgetUsed
	pending := state.PendingToolCalls

	// #124: resolve the resumed task's worker agent from the registry once.
	agent, agentFail := h.ResolveWorkerAgent(&task.LoopStrategy)
	if agentFail != nil {
		af := *agentFail
		af.SessionID = task.SessionID
		return af
	}

	// Resolve the effective tool registry for this resumed session — bridges
	// catalogue tools the same way the turn loop does, so pending tool calls
	// dispatched during resume thread the run's storage + sandbox. #140: resume
	// now routes through the pausing leaf's own toolset handle, restoring its
	// scoped per-node catalogue. An empty handle (the default) still falls back to
	// the global catalogue, so pre-#140 blobs and root pauses behave unchanged.
	toolRegistry := h.effectiveToolRegistry(state.SessionID, state.Toolset)

	// Subagent depth-1: a child state, if present, would be resumed via the
	// caller-installed SubagentTool. Wiring lands with #4/#5. The harness
	// itself just round-trips the field and continues the parent loop.
	_ = state.ChildState

	// Clarification resume (issue #81, Q4b): if this pause came from
	// ToolOutput.AwaitingClarification, the human's answer is injected as the
	// tool RESULT for the clarifying call (the head of PendingToolCalls) — NOT
	// appended as a free-standing user message. Any remaining pending calls
	// after the clarifying one are then dispatched normally before the loop
	// resumes.
	if state.HumanRequest != nil && state.HumanRequest.Kind == HumanReqClarification {
		var answer string
		switch response.Kind {
		case HumanRespAnswer:
			answer = response.Text
		case HumanRespApproveWithFeedback:
			answer = response.Feedback
		}
		if response.Kind == HumanRespAnswer || response.Kind == HumanRespApproveWithFeedback {
			if len(pending) > 0 {
				clarifying := pending[0]
				tr := HarnessToolResult{
					CallID: clarifying.ID,
					Output: ToolOutput{Kind: ToolOutputSuccess, Content: answer},
				}
				h.config.ContextManager.AppendToolResult(ctx, &session, &tr)
				for _, call := range pending[1:] {
					output := dispatchAndUnwrap(ctx, toolRegistry, h.config.Sandbox, call)
					rtr := HarnessToolResult{CallID: call.ID, Output: output}
					h.config.ContextManager.AppendToolResult(ctx, &session, &rtr)
				}
				return h.runReAct(ctx, task, task.LoopStrategy.MaxIterations(), session, budget, onStream, agent)
			}
		}
	}

	// Budget-escalation resume (#130): if this pause came from
	// HumanRequest.BudgetExhausted, map the operator's EscalationAction BEFORE the
	// generic response handling (mirrors the Clarification branch). The node's
	// budget context is reconstructed from the request fields (StepsTaken carried
	// on the request — fork E), NOT a durable cross-process checkpoint (#129).
	// available_actions is ADVISORY (fork D): an out-of-set action is NOT
	// hard-rejected.
	if state.HumanRequest != nil && state.HumanRequest.Kind == HumanReqBudgetExhausted {
		stepsTaken := state.HumanRequest.StepsTaken
		switch {
		case response.Kind == HumanRespEscalate && response.Action.Kind == EscalationContinueWithBudget:
			// Grant `steps` ADDITIONAL allowance and re-enter the loop from the
			// restored checkpoint. The strategy tree is rebuilt with the node's
			// budget policy raised to steps_taken + steps so the restored scope has
			// room for `steps` more steps, and the resumed scope's ContinuesUsed is
			// seeded from the request (#129, AC2). Q3: continues_used rides the
			// REQUEST payload, NOT a new serialized BudgetContext/PausedState field.
			granted := saturatingAddU32(stepsTaken, response.Action.Steps)
			resumedTask := task
			grantTaskBudget(&resumedTask, granted)
			seed := &ResumeContinueSeed{
				Phase:         state.HumanRequest.Phase,
				ContinuesUsed: state.HumanRequest.ContinuesUsed,
			}
			// #138 AC2-b: for a COMPOSED task (PlanExecute etc.) thread the carried
			// worker session through the phase-agnostic resume seed and start the
			// TOP-LEVEL session EMPTY — exactly mirroring the consult resume. The
			// PlanExecute walk re-attaches the seed to the single InProgress task
			// (execute-phase exhaustion) via the InProgress->Pending->complete
			// machinery, or to the PLAN session (AC3 plan-phase exhaustion) — so the
			// stalled worker continues mid-loop instead of re-planning. A BARE leaf
			// has no surrounding walk, so it resumes from session_state directly (the
			// seed has nowhere to attach).
			topSession := SessionState{}
			var resumeSeed *SessionState
			if resumedTask.LoopStrategy.Kind == StrategyReAct {
				topSession = session
				resumeSeed = nil
			} else {
				ws := session
				resumeSeed = &ws
			}
			return h.driveStrategyWithResumeSeed(ctx, resumedTask, topSession, budget, onStream, seed, resumeSeed)
		case response.Kind == HumanRespEscalate && response.Action.Kind == EscalationSkip:
			// Skip: the node is marked skipped and the outer loop advances. For a
			// combinator (PlanExecute) re-entering the loop from the checkpoint
			// advances to the remaining ready tasks. For a leaf there is no sibling,
			// so a skip resolves to a clean (empty) Success carrying the captured
			// session.
			if task.LoopStrategy.Kind == StrategyPlanExecute {
				return h.driveStrategy(ctx, task, session, budget, onStream)
			}
			return RunResult{
				Kind:         RunSuccess,
				Output:       "",
				SessionID:    state.SessionID,
				Usage:        AggregateUsage{},
				Turns:        state.TurnNumber,
				SessionState: session,
			}
		default:
			// Fail (or any out-of-contract response to a budget pause): abort the
			// node and propagate BudgetExceeded; the partial is discarded (the Fail
			// resolution contract).
			return RunResult{
				Kind:         RunFailure,
				Reason:       HaltReason{Kind: HaltBudgetExceeded, LimitType: BudgetLimitTurns},
				SessionID:    state.SessionID,
				Usage:        AggregateUsage{},
				Turns:        state.TurnNumber,
				SessionState: SessionState{},
			}
		}
	}

	switch response.Kind {
	case HumanRespHalt:
		return RunResult{
			Kind:         RunFailure,
			Reason:       HaltReason{Kind: HaltHumanHalted},
			SessionID:    state.SessionID,
			Turns:        state.TurnNumber,
			SessionState: session,
		}
	case HumanRespDeny:
		for _, call := range pending {
			tr := HarnessToolResult{
				CallID: call.ID,
				Output: ToolOutput{
					Kind:        ToolOutputError,
					Message:     response.Reason,
					Recoverable: true,
				},
			}
			h.config.ContextManager.AppendToolResult(ctx, &session, &tr)
		}
	case HumanRespReject:
		h.config.ContextManager.AppendUserMessage(ctx, &session, response.Reason)
	case HumanRespAnswer:
		h.config.ContextManager.AppendUserMessage(ctx, &session, response.Text)
	case HumanRespApproveWithFeedback:
		h.config.ContextManager.AppendUserMessage(ctx, &session, response.Feedback)
	case HumanRespAllow:
		for _, call := range pending {
			output := dispatchAndUnwrap(ctx, toolRegistry, h.config.Sandbox, call)
			tr := HarnessToolResult{CallID: call.ID, Output: output}
			h.config.ContextManager.AppendToolResult(ctx, &session, &tr)
		}
	case HumanRespAllowWithModification:
		for _, call := range response.Calls {
			output := dispatchAndUnwrap(ctx, toolRegistry, h.config.Sandbox, call)
			tr := HarnessToolResult{CallID: call.ID, Output: output}
			h.config.ContextManager.AppendToolResult(ctx, &session, &tr)
		}
	case HumanRespEscalate:
		// An Escalate response delivered to a NON-budget pause is out of contract
		// (#130): the budget-escalation resume is handled by the dedicated
		// HumanReqBudgetExhausted branch above, which returns early. Reaching here
		// means the operator sent an EscalationAction for a ToolApproval / Review /
		// nil-request pause — halt cleanly rather than mis-resuming the loop.
		return RunResult{
			Kind:         RunFailure,
			Reason:       HaltReason{Kind: HaltHumanHalted},
			SessionID:    state.SessionID,
			Turns:        state.TurnNumber,
			SessionState: session,
		}
	}

	return h.runReAct(ctx, task, task.LoopStrategy.MaxIterations(), session, budget, onStream, agent)
}

// Compile-time interface check.
var _ Harness = (*StandardHarness)(nil)

// dispatchAndUnwrap calls the registry's Dispatch and folds a *DispatchError
// into a ToolOutput so the harness loop can keep operating on a uniform
// ToolOutput shape. Recoverability follows the spec's Layer-2 routing:
// UnregisteredTool is unrecoverable; SchemaValidationFailed and recoverable
// SandboxViolations stay recoverable; ToolExecutionFailed mirrors the
// originating recoverable flag (treated as unrecoverable by default since
// the spec routes it via middleware on failure).
func dispatchAndUnwrap(
	ctx context.Context,
	registry ToolRegistry,
	sandbox SandboxProvider,
	call ToolCall,
) ToolOutput {
	tr, err := registry.Dispatch(ctx, call, sandbox)
	if err == nil {
		return tr.Output
	}
	de, ok := err.(*DispatchError)
	if !ok {
		return ToolOutput{
			Kind:        ToolOutputError,
			Message:     err.Error(),
			Recoverable: false,
		}
	}
	switch de.Kind {
	case DispatchErrUnregisteredTool:
		return ToolOutput{
			Kind:        ToolOutputError,
			Message:     de.Error(),
			Recoverable: false,
		}
	case DispatchErrSchemaValidationFailed:
		return ToolOutput{
			Kind:        ToolOutputError,
			Message:     de.Error(),
			Recoverable: true,
		}
	case DispatchErrSandboxViolation:
		recoverable := true
		if de.Violation != nil && de.Violation.IsAlwaysHalt() {
			recoverable = false
		}
		return ToolOutput{
			Kind:        ToolOutputError,
			Message:     de.Error(),
			Recoverable: recoverable,
		}
	default:
		return ToolOutput{
			Kind:        ToolOutputError,
			Message:     de.Error(),
			Recoverable: false,
		}
	}
}

// ============================================================================
// Test-only stub implementations of the sibling interfaces.
// Exported so unit tests and the fixture-replay test can wire a harness.
// ============================================================================

// NoopContextManager is a minimal ContextManager that copies messages out of
// the session and appends tool results / user messages back into it.
type NoopContextManager struct{}

// Assemble returns a Context built from the current session messages. The
// #115 sources are ignored — the Noop manager renders no structural block.
func (NoopContextManager) Assemble(_ context.Context, s *SessionState, _ *Task, _ ContextSources) Context {
	msgs := make([]Message, len(s.Messages))
	copy(msgs, s.Messages)
	return Context{Messages: msgs}
}

// AppendToolResult appends a synthetic tool message to the session.
func (NoopContextManager) AppendToolResult(_ context.Context, s *SessionState, r *HarnessToolResult) {
	var text string
	switch r.Output.Kind {
	case ToolOutputSuccess:
		text = r.Output.Content
	case ToolOutputError:
		text = "[error] " + r.Output.Message
	case ToolOutputWaitingForHuman:
		text = "[waiting]"
	}
	s.Messages = append(s.Messages, Message{Role: RoleTool, Content: NewTextContent(text)})
}

// AppendUserMessage appends a user message.
func (NoopContextManager) AppendUserMessage(_ context.Context, s *SessionState, text string) {
	s.Messages = append(s.Messages, Message{Role: RoleUser, Content: NewTextContent(text)})
}

// ShouldCompact always returns false.
func (NoopContextManager) ShouldCompact(*SessionState) bool { return false }

// AllowAllSandbox is a SandboxProvider that approves every call. It embeds
// DefaultSandbox to pick up ExecuteCommand / HandleLargeOutput / ResolvePath
// implementations.
type AllowAllSandbox struct{ DefaultSandbox }

// Validate always returns nil (no violation).
func (AllowAllSandbox) Validate(context.Context, ToolCall) *SandboxViolation { return nil }

// AlwaysContinuePolicy is a TerminationPolicy that always Continues.
type AlwaysContinuePolicy struct{}

// Evaluate always returns Continue.
func (AlwaysContinuePolicy) Evaluate(context.Context, *SessionState, *BudgetSnapshot) TerminationDecision {
	return TerminationDecision{Kind: TerminationContinue}
}

// ScriptedToolRegistry is a test ToolRegistry that yields queued outputs.
type ScriptedToolRegistry struct {
	mu          sync.Mutex
	outputs     []ToolOutput
	alwaysHalt  map[string]struct{}
	alwaysError *ToolOutput
	schemas     []RegistryToolSchema
	CallCount   atomic.Int64
}

// NewScriptedToolRegistry constructs a ScriptedToolRegistry.
func NewScriptedToolRegistry() *ScriptedToolRegistry {
	return &ScriptedToolRegistry{alwaysHalt: map[string]struct{}{}}
}

// Push queues one output.
func (s *ScriptedToolRegistry) Push(o ToolOutput) *ScriptedToolRegistry {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.outputs = append(s.outputs, o)
	return s
}

// AlwaysRecoverableError makes every dispatch return the SAME recoverable error
// with message, regardless of the queued outputs (issue #137; mirrors the Rust
// ScriptedToolRegistry::always_recoverable_error). Useful for error-loop tests.
func (s *ScriptedToolRegistry) AlwaysRecoverableError(message string) *ScriptedToolRegistry {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.alwaysError = &ToolOutput{Kind: ToolOutputError, Message: message, Recoverable: true}
	return s
}

// WithSchema registers a tool schema advertised via ActiveSchemas (issue #137;
// mirrors the Rust ScriptedToolRegistry::with_schema). Lets the harness loop's
// schema lookup find a parameter schema to render in enrichToolError.
func (s *ScriptedToolRegistry) WithSchema(schema RegistryToolSchema) *ScriptedToolRegistry {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.schemas = append(s.schemas, schema)
	return s
}

// MarkAlwaysHalt flags tool names as always-halt.
func (s *ScriptedToolRegistry) MarkAlwaysHalt(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.alwaysHalt[name] = struct{}{}
}

// Dispatch returns the next queued output (defaulting to Success "ok"). When
// AlwaysRecoverableError was set, it returns that error for EVERY call.
func (s *ScriptedToolRegistry) Dispatch(_ context.Context, call ToolCall, _ SandboxProvider) (HarnessToolResult, error) {
	s.CallCount.Add(1)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.alwaysError != nil {
		return HarnessToolResult{CallID: call.ID, Output: *s.alwaysError}, nil
	}
	var out ToolOutput
	if len(s.outputs) == 0 {
		out = ToolOutput{Kind: ToolOutputSuccess, Content: "ok"}
	} else {
		out = s.outputs[0]
		s.outputs = s.outputs[1:]
	}
	return HarnessToolResult{CallID: call.ID, Output: out}, nil
}

// DispatchAll dispatches calls sequentially, preserving input order.
func (s *ScriptedToolRegistry) DispatchAll(ctx context.Context, calls []ToolCall, sandbox SandboxProvider) []DispatchOutcome {
	out := make([]DispatchOutcome, len(calls))
	for i, c := range calls {
		res, err := s.Dispatch(ctx, c, sandbox)
		if err != nil {
			out[i].Err = err.(*DispatchError)
		} else {
			out[i].Result = res
		}
	}
	return out
}

// Register is a no-op for the test stub.
func (s *ScriptedToolRegistry) Register(Tool, RegistryToolSchema) error { return nil }

// RegisterSet is a no-op for the test stub.
func (s *ScriptedToolRegistry) RegisterSet(ToolSet) error { return nil }

// ActiveSchemas returns the registered schemas (nil unless WithSchema was used).
func (s *ScriptedToolRegistry) ActiveSchemas(*TaskPhase) []RegistryToolSchema {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.schemas) == 0 {
		return nil
	}
	out := make([]RegistryToolSchema, len(s.schemas))
	copy(out, s.schemas)
	return out
}

// HasSubagentTools always reports false in the scripted stub.
func (s *ScriptedToolRegistry) HasSubagentTools() bool { return false }

// IsAlwaysHalt reports whether a tool name was marked.
func (s *ScriptedToolRegistry) IsAlwaysHalt(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.alwaysHalt[name]
	return ok
}

// ScriptedSandbox returns queued violations.
type ScriptedSandbox struct {
	DefaultSandbox
	mu       sync.Mutex
	outcomes []*SandboxViolation
}

// NewScriptedSandbox constructs a ScriptedSandbox.
func NewScriptedSandbox() *ScriptedSandbox { return &ScriptedSandbox{} }

// Push queues a violation (nil = approve).
func (s *ScriptedSandbox) Push(v *SandboxViolation) *ScriptedSandbox {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.outcomes = append(s.outcomes, v)
	return s
}

// Validate returns the next queued violation (default approve).
func (s *ScriptedSandbox) Validate(context.Context, ToolCall) *SandboxViolation {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.outcomes) == 0 {
		return nil
	}
	v := s.outcomes[0]
	s.outcomes = s.outcomes[1:]
	return v
}

// ScriptedTerminationPolicy returns queued decisions.
type ScriptedTerminationPolicy struct {
	mu        sync.Mutex
	decisions []TerminationDecision
}

// NewScriptedTerminationPolicy constructs a ScriptedTerminationPolicy.
func NewScriptedTerminationPolicy() *ScriptedTerminationPolicy {
	return &ScriptedTerminationPolicy{}
}

// Push queues a decision.
func (s *ScriptedTerminationPolicy) Push(d TerminationDecision) *ScriptedTerminationPolicy {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decisions = append(s.decisions, d)
	return s
}

// Evaluate returns the next queued decision (default Continue).
func (s *ScriptedTerminationPolicy) Evaluate(context.Context, *SessionState, *BudgetSnapshot) TerminationDecision {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.decisions) == 0 {
		return TerminationDecision{Kind: TerminationContinue}
	}
	d := s.decisions[0]
	s.decisions = s.decisions[1:]
	return d
}

// ScriptedMiddleware returns queued (hook, decision) pairs.
type ScriptedMiddleware struct {
	mu        sync.Mutex
	decisions []scriptedMW
}

type scriptedMW struct {
	hook HookPoint
	d    MiddlewareDecision
}

// NewScriptedMiddleware constructs a ScriptedMiddleware.
func NewScriptedMiddleware() *ScriptedMiddleware { return &ScriptedMiddleware{} }

// Push queues a (hook, decision) pair.
func (s *ScriptedMiddleware) Push(h HookPoint, d MiddlewareDecision) *ScriptedMiddleware {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decisions = append(s.decisions, scriptedMW{hook: h, d: d})
	return s
}

// next pops and returns the scripted decision for hook if it is at the front of
// the queue, else Continue. Decisions are returned RAW (no validateDecision), so
// a test can exercise the harness's defensive handling of an out-of-place
// decision — unlike the StandardMiddlewareChain.
func (s *ScriptedMiddleware) next(hook HookPoint) MiddlewareDecision {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.decisions) == 0 || s.decisions[0].hook != hook {
		return MiddlewareDecision{Kind: MiddlewareContinue}
	}
	d := s.decisions[0].d
	s.decisions = s.decisions[1:]
	return d
}

// Register is a no-op — the double scripts decisions directly.
func (s *ScriptedMiddleware) Register(_ Middleware) error { return nil }

// FireBeforeSession implements MiddlewareChain.
func (s *ScriptedMiddleware) FireBeforeSession(_ context.Context, _ *Task, _ SessionID) (MiddlewareDecision, error) {
	return s.next(HookBeforeSession), nil
}

// FireBeforeTurn implements MiddlewareChain.
func (s *ScriptedMiddleware) FireBeforeTurn(_ context.Context, _ *SessionState, _ uint32) (MiddlewareDecision, error) {
	return s.next(HookBeforeTurn), nil
}

// FireBeforeTool implements MiddlewareChain.
func (s *ScriptedMiddleware) FireBeforeTool(_ context.Context, _ *[]ToolCall, _ uint32) (MiddlewareDecision, error) {
	return s.next(HookBeforeTool), nil
}

// FireAfterTool implements MiddlewareChain.
func (s *ScriptedMiddleware) FireAfterTool(_ context.Context, _ []ToolCall, _ *[]ToolResult) (MiddlewareDecision, error) {
	return s.next(HookAfterTool), nil
}

// FireBeforeCompletion implements MiddlewareChain.
func (s *ScriptedMiddleware) FireBeforeCompletion(_ context.Context, _ string, _ uint32, _ *SessionState) (MiddlewareDecision, error) {
	return s.next(HookBeforeCompletion), nil
}

// FireAfterSession implements MiddlewareChain.
//
// After hooks ignore the decision; drain a scripted AfterSession entry.
func (s *ScriptedMiddleware) FireAfterSession(_ context.Context, _ *RunResult, _ SessionID) error {
	_ = s.next(HookAfterSession)
	return nil
}

// Compile-time interface check.
var _ MiddlewareChain = (*ScriptedMiddleware)(nil)

// MockAgent is a test Agent that yields queued TurnResults.
type MockAgent struct {
	id      AgentID
	mu      sync.Mutex
	results []TurnResult
}

// NewMockAgent constructs a MockAgent.
func NewMockAgent(id AgentID) *MockAgent { return &MockAgent{id: id} }

// Push queues a TurnResult.
func (m *MockAgent) Push(r TurnResult) *MockAgent {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.results = append(m.results, r)
	return m
}

// Turn returns the next queued TurnResult, or an EmptyResponse error.
func (m *MockAgent) Turn(context.Context, Context) TurnResult {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.results) == 0 {
		return NewTurnError(NewEmptyResponseError(), nil)
	}
	r := m.results[0]
	m.results = m.results[1:]
	return r
}

// ID returns the agent identifier.
func (m *MockAgent) ID() AgentID { return m.id }

var _ Agent = (*MockAgent)(nil)
