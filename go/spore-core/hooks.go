// hooks.go — issue #69 lifecycle hook system (Hook / HookChain).
//
// A general-purpose extension layer that lets external code observe and shape
// the harness at well-defined lifecycle moments. This is a NEW higher-level
// sibling of the middleware pipeline (middleware/middleware.go): middleware
// shapes the context block DURING assembly (a lower-level primitive); hooks
// fire at a higher level on already-assembled artifacts. The two layers are
// intentionally distinct and this file does not modify or subsume middleware.
//
// # Types
//   - HookEvent — the 17 lifecycle events, with classification predicates
//     IsPre, IsMutable, IsSyncOnly, IsAsyncOnly, CanBlock, CanDeny.
//   - HookContext — a tagged struct carrying the per-event payload; mutable
//     fields are pointers so pre-hooks can rewrite them in place.
//   - HookDecision — Continue / Block / Inject / Deny / Mutate, JSON-tagged on
//     "decision" with snake_case values (byte-identical to the shared fixture).
//   - HookSync — sync vs async registration mode.
//   - HookError — typed error with a discriminating Kind.
//   - Hook interface — Handle, Events, Name, SyncMode.
//   - HookChain interface — Register + Fire.
//   - StandardHookChain — in-memory reference impl: registration-order fan-out,
//     chained mutation through pre-hooks, sync aggregation, async
//     fire-and-forget.
//   - FunctionHook, CommandHook — the two v1 handler types.
//
// # The 17 events (mutation / blocking / sync classification)
//
//	| Event                | Pre/Post | Mutates              | Can block | Sync mode      |
//	|----------------------|----------|----------------------|-----------|----------------|
//	| PreTurn              | pre      | ContextBlock         | yes       | sync           |
//	| PostTurn             | post     | —                    | no        | sync or async  |
//	| PreToolUse           | pre      | ToolInput (or deny)  | yes       | sync           |
//	| PostToolUse          | post     | —                    | no        | sync or async  |
//	| PostToolUseFailure   | post     | —                    | no        | sync or async  |
//	| PostToolBatch        | post     | —                    | yes       | sync           |
//	| OnLoopStart          | pre      | TaskInstruction      | yes       | sync           |
//	| Stop                 | post     | —                    | yes       | sync only      |
//	| OnPause              | post     | —                    | no        | async only     |
//	| OnResume             | pre      | TaskInstruction      | no        | sync           |
//	| OnError              | post     | — (can suppress)     | yes       | sync or async  |
//	| OnPlanCreated        | post     | Plan                 | yes       | sync           |
//	| OnTaskAdvance        | pre      | Task                 | yes       | sync           |
//	| OnSubagentSpawn      | pre      | ChildTask (or deny)  | yes       | sync           |
//	| OnSubagentComplete   | post     | —                    | no        | sync or async  |
//	| PreCompact           | pre      | PreserveHints        | yes       | sync           |
//	| PostCompact          | post     | —                    | no        | async ok       |
//
// # Loop-wiring status
//
// Events whose loop machinery EXISTS are wired into the ReAct loop in
// harness.go. Stop is the live one in Go (it replaces the old
// ForceAnotherTurn / completion-check path): the harness fires registered Stop
// hooks synchronously when the loop believes it is done; a Block injects its
// reason into the next turn (same mechanism ForceAnotherTurn uses) and the loop
// continues, bounded by the per-run MaxStopBlocks cap.
//
// Events DEFINED-AND-UNIT-TESTED but NOT YET loop-wired (their strategy /
// subagent / pause machinery is deferred elsewhere): OnPause, OnResume,
// OnPlanCreated, OnTaskAdvance, OnSubagentSpawn, OnSubagentComplete. Each is
// exercised directly via StandardHookChain.Fire in the unit tests; once the
// owning strategy / subagent issues land, the fire calls move into that loop
// machinery.

package sporecore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"sync"
)

// ============================================================================
// Locally-defined payload types
//
// These artifacts are not yet modelled elsewhere in the package. ContextBlock
// reuses the agent's assembled Context (agent.go). CompactionPreserveHints
// mirrors the wire shape of contextmgr.CompactionPreserveHints — it cannot be
// imported here because the contextmgr subpackage imports this root package
// (importing it back is a compile-time cycle), the same reason CompactionTurn
// types PreserveHints as `any`. They are intentionally additive — when the
// owning strategy / subagent issues land, the canonical shapes replace these.
// ============================================================================

// ContextBlock is the per-turn input a PreTurn hook may rewrite. It aliases the
// agent's assembled Context.
type ContextBlock = Context

// TurnOutput is the output of a single completed turn, handed to PostTurn /
// Stop hooks.
type TurnOutput struct {
	// Text is the agent's final textual output for the turn (empty for tool
	// turns).
	Text string `json:"text"`
	// HadToolCalls reports whether the turn requested tool calls rather than a
	// final response.
	HadToolCalls bool `json:"had_tool_calls"`
}

// PlanArtifact is a composite-strategy plan, handed to OnPlanCreated.
type PlanArtifact struct {
	Tasks     []string `json:"tasks"`
	Rationale string   `json:"rationale"`
}

// MarshalJSON ensures Tasks serialises as [] rather than null.
func (p PlanArtifact) MarshalJSON() ([]byte, error) {
	type alias PlanArtifact
	a := alias(p)
	if a.Tasks == nil {
		a.Tasks = []string{}
	}
	return json.Marshal(a)
}

// ToolCallSummary is a one-line summary of a tool call in a batch, handed to
// PostToolBatch.
type ToolCallSummary struct {
	ToolName  string `json:"tool_name"`
	Succeeded bool   `json:"succeeded"`
}

// CompactionPreserveHints tell the compaction summariser what must survive.
// Mirrors contextmgr.CompactionPreserveHints (see package note above for why it
// is redefined here rather than imported). Handed to PreCompact hooks.
type CompactionPreserveHints struct {
	KeepArchitecturalDecisions bool `json:"keep_architectural_decisions"`
	KeepOpenProblems           bool `json:"keep_open_problems"`
	KeepCurrentTaskState       bool `json:"keep_current_task_state"`
	KeepRecentFileList         bool `json:"keep_recent_file_list"`
	KeepThinkingBlocks         bool `json:"keep_thinking_blocks"`
}

// ============================================================================
// HookEvent
// ============================================================================

// HookEvent identifies one of the 17 lifecycle moments at which a Hook fires.
// The string form is the snake_case wire name.
type HookEvent string

const (
	HookEventPreTurn            HookEvent = "pre_turn"
	HookEventPostTurn           HookEvent = "post_turn"
	HookEventPreToolUse         HookEvent = "pre_tool_use"
	HookEventPostToolUse        HookEvent = "post_tool_use"
	HookEventPostToolUseFailure HookEvent = "post_tool_use_failure"
	HookEventPostToolBatch      HookEvent = "post_tool_batch"
	HookEventOnLoopStart        HookEvent = "on_loop_start"
	HookEventStop               HookEvent = "stop"
	HookEventOnPause            HookEvent = "on_pause"
	HookEventOnResume           HookEvent = "on_resume"
	HookEventOnError            HookEvent = "on_error"
	HookEventOnPlanCreated      HookEvent = "on_plan_created"
	HookEventOnTaskAdvance      HookEvent = "on_task_advance"
	HookEventOnSubagentSpawn    HookEvent = "on_subagent_spawn"
	HookEventOnSubagentComplete HookEvent = "on_subagent_complete"
	HookEventPreCompact         HookEvent = "pre_compact"
	HookEventPostCompact        HookEvent = "post_compact"
)

// AllHookEvents lists all 17 events in catalogue order.
var AllHookEvents = [...]HookEvent{
	HookEventPreTurn,
	HookEventPostTurn,
	HookEventPreToolUse,
	HookEventPostToolUse,
	HookEventPostToolUseFailure,
	HookEventPostToolBatch,
	HookEventOnLoopStart,
	HookEventStop,
	HookEventOnPause,
	HookEventOnResume,
	HookEventOnError,
	HookEventOnPlanCreated,
	HookEventOnTaskAdvance,
	HookEventOnSubagentSpawn,
	HookEventOnSubagentComplete,
	HookEventPreCompact,
	HookEventPostCompact,
}

// IsPre reports whether this is a pre-event (fires before its action; may
// mutate its single mutable field).
func (e HookEvent) IsPre() bool {
	switch e {
	case HookEventPreTurn, HookEventPreToolUse, HookEventOnLoopStart,
		HookEventOnResume, HookEventOnTaskAdvance, HookEventOnSubagentSpawn,
		HookEventPreCompact:
		return true
	default:
		return false
	}
}

// IsMutable reports whether this event carries a mutable field a pre-hook may
// rewrite. Equivalent to IsPre — every pre-event is mutable.
func (e HookEvent) IsMutable() bool { return e.IsPre() }

// IsSyncOnly reports whether this event may only run synchronously. Stop is the
// divergence gate — it MUST block the loop, so it cannot be fire-and-forget;
// the pre-events are likewise sync because their mutation must complete before
// the action proceeds. PostToolBatch blocks the next model call.
func (e HookEvent) IsSyncOnly() bool {
	switch e {
	case HookEventStop, HookEventPreTurn, HookEventPreToolUse,
		HookEventPostToolBatch, HookEventOnLoopStart, HookEventOnResume,
		HookEventOnPlanCreated, HookEventOnTaskAdvance, HookEventOnSubagentSpawn,
		HookEventPreCompact:
		return true
	default:
		return false
	}
}

// IsAsyncOnly reports whether this event may only run asynchronously
// (fire-and-forget).
func (e HookEvent) IsAsyncOnly() bool {
	return e == HookEventOnPause || e == HookEventPostCompact
}

// CanBlock reports whether a hook on this event may return a Block decision.
func (e HookEvent) CanBlock() bool {
	switch e {
	case HookEventPreTurn, HookEventPostToolBatch, HookEventOnLoopStart,
		HookEventStop, HookEventOnError, HookEventOnPlanCreated,
		HookEventOnTaskAdvance:
		return true
	default:
		return false
	}
}

// CanDeny reports whether a hook on this event may return a Deny decision.
func (e HookEvent) CanDeny() bool {
	return e == HookEventPreToolUse || e == HookEventOnSubagentSpawn
}

// ============================================================================
// HookSync
// ============================================================================

// HookSync is whether a hook runs synchronously (blocking, result observed) or
// asynchronously (fire-and-forget).
type HookSync string

const (
	HookSyncSync  HookSync = "sync"
	HookSyncAsync HookSync = "async"
)

// ============================================================================
// HookDecision
// ============================================================================

// HookDecisionKind discriminates HookDecision variants.
type HookDecisionKind string

const (
	HookDecisionContinue HookDecisionKind = "continue"
	HookDecisionBlock    HookDecisionKind = "block"
	HookDecisionInject   HookDecisionKind = "inject"
	HookDecisionDeny     HookDecisionKind = "deny"
	HookDecisionMutate   HookDecisionKind = "mutate"
)

// HookDecision is the control a hook exerts when it fires. The wire format is
// tagged on "decision" with snake_case values, e.g.
// {"decision":"block","reason":"x"}. Use the constructor helpers
// (Continue/Block/Inject/Deny/Mutate) rather than building the struct directly.
type HookDecision struct {
	// Decision is the discriminator.
	Decision HookDecisionKind
	// Reason carries the text for Block and Deny.
	Reason string
	// Context carries the injected text for Inject.
	Context string
	// Data carries the replacement value for Mutate (raw JSON).
	Data json.RawMessage
}

// Continue builds a Continue decision (proceed; no change).
func Continue() HookDecision { return HookDecision{Decision: HookDecisionContinue} }

// Block builds a Block decision (can-block events only; injects reason into the
// next turn).
func Block(reason string) HookDecision {
	return HookDecision{Decision: HookDecisionBlock, Reason: reason}
}

// Inject builds an Inject decision (injects context into the next turn's
// context block).
func Inject(injectContext string) HookDecision {
	return HookDecision{Decision: HookDecisionInject, Context: injectContext}
}

// Deny builds a Deny decision (PreToolUse / OnSubagentSpawn only; rejects the
// action).
func Deny(reason string) HookDecision {
	return HookDecision{Decision: HookDecisionDeny, Reason: reason}
}

// Mutate builds a Mutate decision (pre-hooks only; replaces the mutable field
// with data). data must be valid JSON.
func Mutate(data json.RawMessage) HookDecision {
	return HookDecision{Decision: HookDecisionMutate, Data: data}
}

// MarshalJSON emits the serde-tagged form, byte-identical to the shared
// fixture: only the fields relevant to the variant appear.
func (d HookDecision) MarshalJSON() ([]byte, error) {
	switch d.Decision {
	case HookDecisionContinue:
		return json.Marshal(struct {
			Decision HookDecisionKind `json:"decision"`
		}{d.Decision})
	case HookDecisionBlock, HookDecisionDeny:
		return json.Marshal(struct {
			Decision HookDecisionKind `json:"decision"`
			Reason   string           `json:"reason"`
		}{d.Decision, d.Reason})
	case HookDecisionInject:
		return json.Marshal(struct {
			Decision HookDecisionKind `json:"decision"`
			Context  string           `json:"context"`
		}{d.Decision, d.Context})
	case HookDecisionMutate:
		data := d.Data
		if len(data) == 0 {
			data = json.RawMessage("null")
		}
		return json.Marshal(struct {
			Decision HookDecisionKind `json:"decision"`
			Data     json.RawMessage  `json:"data"`
		}{d.Decision, data})
	default:
		return nil, fmt.Errorf("HookDecision: unknown decision %q", d.Decision)
	}
}

// UnmarshalJSON decodes the serde-tagged form.
func (d *HookDecision) UnmarshalJSON(data []byte) error {
	var probe struct {
		Decision HookDecisionKind `json:"decision"`
		Reason   string           `json:"reason"`
		Context  string           `json:"context"`
		Data     json.RawMessage  `json:"data"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	switch probe.Decision {
	case HookDecisionContinue, HookDecisionBlock, HookDecisionInject,
		HookDecisionDeny, HookDecisionMutate:
	default:
		return fmt.Errorf("HookDecision: unknown decision %q", probe.Decision)
	}
	d.Decision = probe.Decision
	d.Reason = probe.Reason
	d.Context = probe.Context
	d.Data = probe.Data
	return nil
}

// ValidateFor checks that this decision is legal for event. Used at both
// register time (against a hook's declared events) and fire time.
func (d HookDecision) ValidateFor(event HookEvent) error {
	var ok bool
	switch d.Decision {
	case HookDecisionContinue, HookDecisionInject:
		ok = true
	case HookDecisionBlock:
		ok = event.CanBlock()
	case HookDecisionDeny:
		ok = event.CanDeny()
	case HookDecisionMutate:
		ok = event.IsMutable()
	default:
		ok = false
	}
	if ok {
		return nil
	}
	return &HookError{
		Kind:     HookErrIllegalDecision,
		Event:    event,
		Decision: d.Decision,
	}
}

// ============================================================================
// HookError
// ============================================================================

// HookErrorKind discriminates HookError variants.
type HookErrorKind string

const (
	HookErrIllegalDecision      HookErrorKind = "illegal_decision"
	HookErrSyncOnlyEvent        HookErrorKind = "sync_only_event"
	HookErrAsyncOnlyEvent       HookErrorKind = "async_only_event"
	HookErrCommandFailed        HookErrorKind = "command_failed"
	HookErrCommandOutputInvalid HookErrorKind = "command_output_invalid"
	HookErrHandlerFailed        HookErrorKind = "handler_failed"
)

// HookError is the typed error returned by hook registration and firing.
type HookError struct {
	Kind     HookErrorKind
	Event    HookEvent
	Decision HookDecisionKind
	Hook     string
	Command  string
	Code     int
	Stderr   string
	Detail   string
}

// Error implements error.
func (e *HookError) Error() string {
	switch e.Kind {
	case HookErrIllegalDecision:
		return fmt.Sprintf("hook %q decision is illegal for event %q", e.Decision, e.Event)
	case HookErrSyncOnlyEvent:
		return fmt.Sprintf("hook %q cannot register for sync-only event %q as async", e.Hook, e.Event)
	case HookErrAsyncOnlyEvent:
		return fmt.Sprintf("hook %q cannot register for async-only event %q as sync", e.Hook, e.Event)
	case HookErrCommandFailed:
		return fmt.Sprintf("command hook %q exited with status %d: %s", e.Command, e.Code, e.Stderr)
	case HookErrCommandOutputInvalid:
		return fmt.Sprintf("command hook %q produced invalid stdout: %s", e.Command, e.Detail)
	case HookErrHandlerFailed:
		return fmt.Sprintf("hook %q failed: %s", e.Hook, e.Detail)
	default:
		return fmt.Sprintf("hook error: %s", e.Kind)
	}
}

// ============================================================================
// HookContext — tagged per-event payload
// ============================================================================

// HookContext is the per-event payload a Hook receives. Event is the
// discriminator; only the fields belonging to that event are populated.
// Mutable fields are pointers so a pre-hook can rewrite them in place; the rest
// are read-only.
type HookContext struct {
	Event     HookEvent
	SessionID SessionID

	// PreTurn, PostTurn, PreToolUse, PostToolUse, PostToolUseFailure,
	// PostToolBatch, Stop, OnPause, OnError.
	TurnNumber uint32

	// PreTurn (mutable).
	ContextBlock *ContextBlock

	// PostTurn.
	Output *TurnOutput

	// PreToolUse, PostToolUse, PostToolUseFailure.
	ToolName string
	// PreToolUse (mutable), PostToolUse, PostToolUseFailure.
	ToolInput *json.RawMessage
	// PostToolUse.
	ToolResponse json.RawMessage
	// PostToolUse, PostToolUseFailure.
	DurationMs uint64
	// PostToolUseFailure, OnError.
	Error string

	// PostToolBatch.
	ToolCalls []ToolCallSummary

	// OnLoopStart, OnResume, OnSubagentSpawn (mutable). Pointer so a pre-hook
	// can rewrite the instruction / child task in place.
	TaskInstruction *string
	// OnLoopStart.
	Config *HarnessConfig

	// Stop.
	LastOutput   *TurnOutput
	SessionState *SessionState

	// OnResume.
	PausedState *PausedState

	// OnPlanCreated (mutable).
	Plan *PlanArtifact

	// OnTaskAdvance (mutable).
	Task       *Task
	TaskIndex  int
	TotalTasks int

	// OnSubagentSpawn. ChildTask aliases TaskInstruction (the mutable child
	// task string); Strategy is the spawn strategy name.
	Strategy string

	// OnSubagentComplete.
	ChildSessionID SessionID
	Result         json.RawMessage

	// PreCompact (mutable).
	PreserveHints *CompactionPreserveHints

	// PostCompact.
	CompactSummary string
}

// toPayload serialises this context to the JSON payload a command handler
// receives on stdin (the "context" field). Mutable fields are serialised by
// value. The field ordering matches the per-event context defined in the spec.
func (c *HookContext) toPayload() (json.RawMessage, error) {
	var m map[string]any
	switch c.Event {
	case HookEventPreTurn:
		m = map[string]any{
			"session_id":    c.SessionID,
			"turn_number":   c.TurnNumber,
			"context_block": c.ContextBlock,
		}
	case HookEventPostTurn:
		m = map[string]any{
			"session_id":  c.SessionID,
			"turn_number": c.TurnNumber,
			"output":      c.Output,
		}
	case HookEventPreToolUse:
		m = map[string]any{
			"session_id":  c.SessionID,
			"turn_number": c.TurnNumber,
			"tool_name":   c.ToolName,
			"tool_input":  c.toolInputValue(),
		}
	case HookEventPostToolUse:
		m = map[string]any{
			"session_id":    c.SessionID,
			"turn_number":   c.TurnNumber,
			"tool_name":     c.ToolName,
			"tool_input":    c.toolInputValue(),
			"tool_response": rawOrNull(c.ToolResponse),
			"duration_ms":   c.DurationMs,
		}
	case HookEventPostToolUseFailure:
		m = map[string]any{
			"session_id":  c.SessionID,
			"turn_number": c.TurnNumber,
			"tool_name":   c.ToolName,
			"tool_input":  c.toolInputValue(),
			"error":       c.Error,
			"duration_ms": c.DurationMs,
		}
	case HookEventPostToolBatch:
		calls := c.ToolCalls
		if calls == nil {
			calls = []ToolCallSummary{}
		}
		m = map[string]any{
			"session_id":  c.SessionID,
			"turn_number": c.TurnNumber,
			"tool_calls":  calls,
		}
	case HookEventOnLoopStart:
		m = map[string]any{
			"session_id":       c.SessionID,
			"task_instruction": derefString(c.TaskInstruction),
		}
	case HookEventStop:
		m = map[string]any{
			"session_id":       c.SessionID,
			"turn_number":      c.TurnNumber,
			"last_output":      c.LastOutput,
			"task_instruction": derefString(c.TaskInstruction),
			"session_state":    c.SessionState,
		}
	case HookEventOnPause:
		m = map[string]any{
			"session_id":  c.SessionID,
			"turn_number": c.TurnNumber,
		}
	case HookEventOnResume:
		m = map[string]any{
			"session_id":       c.SessionID,
			"task_instruction": derefString(c.TaskInstruction),
			"paused_state":     c.PausedState,
		}
	case HookEventOnError:
		m = map[string]any{
			"session_id":  c.SessionID,
			"turn_number": c.TurnNumber,
			"error":       c.Error,
		}
	case HookEventOnPlanCreated:
		m = map[string]any{
			"session_id": c.SessionID,
			"plan":       c.Plan,
		}
	case HookEventOnTaskAdvance:
		m = map[string]any{
			"session_id":  c.SessionID,
			"task":        c.Task,
			"task_index":  c.TaskIndex,
			"total_tasks": c.TotalTasks,
		}
	case HookEventOnSubagentSpawn:
		m = map[string]any{
			"session_id": c.SessionID,
			"child_task": derefString(c.TaskInstruction),
			"strategy":   c.Strategy,
		}
	case HookEventOnSubagentComplete:
		m = map[string]any{
			"session_id":       c.SessionID,
			"child_session_id": c.ChildSessionID,
			"result":           rawOrNull(c.Result),
		}
	case HookEventPreCompact:
		m = map[string]any{
			"session_id":     c.SessionID,
			"preserve_hints": c.PreserveHints,
		}
	case HookEventPostCompact:
		m = map[string]any{
			"session_id":      c.SessionID,
			"compact_summary": c.CompactSummary,
		}
	default:
		return nil, fmt.Errorf("HookContext: unknown event %q", c.Event)
	}
	return json.Marshal(m)
}

func (c *HookContext) toolInputValue() json.RawMessage {
	if c.ToolInput == nil {
		return json.RawMessage("null")
	}
	return rawOrNull(*c.ToolInput)
}

func rawOrNull(r json.RawMessage) json.RawMessage {
	if len(r) == 0 {
		return json.RawMessage("null")
	}
	return r
}

func derefString(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// applyMutation applies a Mutate decision's data to this context's mutable
// field. Errors if the field cannot be decoded into the target type, or this
// is not a mutable event.
func (c *HookContext) applyMutation(hookName string, data json.RawMessage) error {
	fail := func(detail string) error {
		return &HookError{Kind: HookErrHandlerFailed, Hook: hookName, Detail: detail}
	}
	switch c.Event {
	case HookEventPreTurn:
		var cb ContextBlock
		if err := json.Unmarshal(data, &cb); err != nil {
			return fail(err.Error())
		}
		if c.ContextBlock != nil {
			*c.ContextBlock = cb
		}
	case HookEventPreToolUse:
		if c.ToolInput != nil {
			*c.ToolInput = append(json.RawMessage(nil), data...)
		}
	case HookEventOnLoopStart, HookEventOnResume, HookEventOnSubagentSpawn:
		s, err := stringFromValue(data)
		if err != nil {
			return fail(err.Error())
		}
		if c.TaskInstruction != nil {
			*c.TaskInstruction = s
		}
	case HookEventOnPlanCreated:
		var p PlanArtifact
		if err := json.Unmarshal(data, &p); err != nil {
			return fail(err.Error())
		}
		if c.Plan != nil {
			*c.Plan = p
		}
	case HookEventOnTaskAdvance:
		var t Task
		if err := json.Unmarshal(data, &t); err != nil {
			return fail(err.Error())
		}
		if c.Task != nil {
			*c.Task = t
		}
	case HookEventPreCompact:
		var h CompactionPreserveHints
		if err := json.Unmarshal(data, &h); err != nil {
			return fail(err.Error())
		}
		if c.PreserveHints != nil {
			*c.PreserveHints = h
		}
	default:
		return &HookError{Kind: HookErrIllegalDecision, Event: c.Event, Decision: HookDecisionMutate}
	}
	return nil
}

// stringFromValue coerces a Mutate data value into a string (accepts a JSON
// string, or stringifies any other scalar/object as JSON text).
func stringFromValue(data json.RawMessage) (string, error) {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		return s, nil
	}
	return string(data), nil
}

// ============================================================================
// Hook
// ============================================================================

// Hook is a single lifecycle hook handler.
type Hook interface {
	// Handle handles one firing. The context borrows the live data; pre-hooks
	// may mutate the mutable field directly OR return a Mutate decision.
	Handle(ctx context.Context, hctx *HookContext) (HookDecision, error)
	// Events lists the events this hook subscribes to.
	Events() []HookEvent
	// Name is a stable name for diagnostics and error messages.
	Name() string
	// SyncMode reports whether this hook runs sync (blocking) or async
	// (fire-and-forget).
	SyncMode() HookSync
}

// ============================================================================
// HookChain
// ============================================================================

// FireOutcomeKind discriminates FireOutcome variants.
type FireOutcomeKind string

const (
	FireContinue FireOutcomeKind = "continue"
	FireBlock    FireOutcomeKind = "block"
	FireDeny     FireOutcomeKind = "deny"
	FireInject   FireOutcomeKind = "inject"
)

// FireOutcome is the aggregate result of firing an event chain, reported back
// to the harness loop.
type FireOutcome struct {
	Kind FireOutcomeKind
	// Reason is set for Block and Deny.
	Reason string
	// Context is the newline-joined injected text for Inject.
	Context string
}

// HookChain is the registry + dispatcher for Hooks. Implementations fan out to
// all hooks subscribed to an event in registration order.
type HookChain interface {
	// Register registers a hook. Rejects sync-only events registered async (and
	// vice versa). Registration order is firing order.
	Register(hook Hook) error
	// Fire fires the event identified by hctx.Event. The chain threads
	// mutations through hctx in place and returns the aggregate outcome (first
	// Block/Deny wins; Injects are newline-joined).
	Fire(ctx context.Context, hctx *HookContext) (FireOutcome, error)
}

// ============================================================================
// StandardHookChain
// ============================================================================

// StandardHookChain is the in-memory reference HookChain. It holds hooks behind
// a mutex and fans out in registration order.
type StandardHookChain struct {
	mu    sync.Mutex
	hooks []Hook
}

// NewStandardHookChain constructs an empty chain.
func NewStandardHookChain() *StandardHookChain { return &StandardHookChain{} }

// Register implements HookChain.
func (s *StandardHookChain) Register(hook Hook) error {
	mode := hook.SyncMode()
	for _, event := range hook.Events() {
		if event.IsSyncOnly() && mode == HookSyncAsync {
			return &HookError{Kind: HookErrSyncOnlyEvent, Hook: hook.Name(), Event: event}
		}
		if event.IsAsyncOnly() && mode == HookSyncSync {
			return &HookError{Kind: HookErrAsyncOnlyEvent, Hook: hook.Name(), Event: event}
		}
	}
	s.mu.Lock()
	s.hooks = append(s.hooks, hook)
	s.mu.Unlock()
	return nil
}

// hooksFor snapshots the registered hooks subscribed to event in registration
// order, releasing the lock before firing so handlers never re-enter under the
// lock.
func (s *StandardHookChain) hooksFor(event HookEvent) []Hook {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []Hook
	for _, h := range s.hooks {
		for _, e := range h.Events() {
			if e == event {
				out = append(out, h)
				break
			}
		}
	}
	return out
}

// Fire implements HookChain.
func (s *StandardHookChain) Fire(ctx context.Context, hctx *HookContext) (FireOutcome, error) {
	event := hctx.Event
	hooks := s.hooksFor(event)
	var injects []string

	for _, hook := range hooks {
		if hook.SyncMode() == HookSyncAsync {
			// Async hooks are fire-and-forget: spawned on a goroutine, never
			// awaited, and their result/failure is swallowed. We ship a detached
			// owned snapshot so the goroutine never touches the caller's live
			// context after Fire returns.
			payload, err := hctx.toPayload()
			if err != nil {
				// A malformed payload cannot reach an observability-only async
				// hook; skip it. The sync path below is unaffected.
				continue
			}
			h := hook
			go func() {
				detached := &HookContext{Event: event, SessionID: hctx.SessionID}
				_, _ = h.Handle(context.Background(), detached)
				_ = payload
			}()
			continue
		}

		decision, err := hook.Handle(ctx, hctx)
		if err != nil {
			return FireOutcome{}, err
		}
		if err := decision.ValidateFor(event); err != nil {
			return FireOutcome{}, err
		}

		switch decision.Decision {
		case HookDecisionContinue:
		case HookDecisionInject:
			injects = append(injects, decision.Context)
		case HookDecisionBlock:
			return FireOutcome{Kind: FireBlock, Reason: decision.Reason}, nil
		case HookDecisionDeny:
			return FireOutcome{Kind: FireDeny, Reason: decision.Reason}, nil
		case HookDecisionMutate:
			if err := hctx.applyMutation(hook.Name(), decision.Data); err != nil {
				return FireOutcome{}, err
			}
		}
	}

	if len(injects) == 0 {
		return FireOutcome{Kind: FireContinue}, nil
	}
	return FireOutcome{Kind: FireInject, Context: strings.Join(injects, "\n")}, nil
}

var _ HookChain = (*StandardHookChain)(nil)

// ============================================================================
// FunctionHook — inline closure handler
// ============================================================================

// HookFunc is the closure a FunctionHook runs. It is invoked synchronously
// inside Handle.
type HookFunc func(ctx context.Context, hctx *HookContext) (HookDecision, error)

// FunctionHook is a Hook backed by an inline closure. It is the primary handler
// type for harness builders.
type FunctionHook struct {
	name     string
	events   []HookEvent
	syncMode HookSync
	fn       HookFunc
}

// NewFunctionHook builds a sync FunctionHook.
func NewFunctionHook(name string, events []HookEvent, fn HookFunc) *FunctionHook {
	return &FunctionHook{name: name, events: events, syncMode: HookSyncSync, fn: fn}
}

// Async marks this function hook async (fire-and-forget). Only legal for events
// that are not sync-only; the chain enforces this at register time.
func (f *FunctionHook) Async() *FunctionHook {
	f.syncMode = HookSyncAsync
	return f
}

// Handle implements Hook.
func (f *FunctionHook) Handle(ctx context.Context, hctx *HookContext) (HookDecision, error) {
	return f.fn(ctx, hctx)
}

// Events implements Hook.
func (f *FunctionHook) Events() []HookEvent { return f.events }

// Name implements Hook.
func (f *FunctionHook) Name() string { return f.name }

// SyncMode implements Hook.
func (f *FunctionHook) SyncMode() HookSync { return f.syncMode }

var _ Hook = (*FunctionHook)(nil)

// ============================================================================
// CommandHook — shell command handler
// ============================================================================

// CommandHook is a Hook that shells out to an external command. stdin receives
// {"event":"<snake_case>","context":<payload>}; stdout is parsed as a tagged
// HookDecision. Nonzero exit → HookError{Kind: CommandFailed} (an explicit
// error, NOT an implicit block); malformed stdout → CommandOutputInvalid. There
// is no sandbox and no timeout in v1.
type CommandHook struct {
	name     string
	events   []HookEvent
	syncMode HookSync
	program  string
	args     []string
}

// NewCommandHook builds a sync CommandHook.
func NewCommandHook(name string, events []HookEvent, program string, args []string) *CommandHook {
	return &CommandHook{name: name, events: events, syncMode: HookSyncSync, program: program, args: args}
}

// Async marks this command hook async (fire-and-forget).
func (h *CommandHook) Async() *CommandHook {
	h.syncMode = HookSyncAsync
	return h
}

// Handle implements Hook.
func (h *CommandHook) Handle(ctx context.Context, hctx *HookContext) (HookDecision, error) {
	payloadCtx, err := hctx.toPayload()
	if err != nil {
		return HookDecision{}, &HookError{Kind: HookErrHandlerFailed, Hook: h.name, Detail: err.Error()}
	}
	stdin, err := json.Marshal(struct {
		Event   HookEvent       `json:"event"`
		Context json.RawMessage `json:"context"`
	}{hctx.Event, payloadCtx})
	if err != nil {
		return HookDecision{}, &HookError{Kind: HookErrHandlerFailed, Hook: h.name, Detail: err.Error()}
	}

	cmd := exec.CommandContext(ctx, h.program, h.args...)
	cmd.Stdin = strings.NewReader(string(stdin))
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	if runErr != nil {
		code := -1
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			code = exitErr.ExitCode()
		}
		return HookDecision{}, &HookError{
			Kind:    HookErrCommandFailed,
			Command: h.program,
			Code:    code,
			Stderr:  strings.TrimSpace(stderr.String()),
		}
	}

	var decision HookDecision
	if err := json.Unmarshal([]byte(strings.TrimSpace(stdout.String())), &decision); err != nil {
		return HookDecision{}, &HookError{
			Kind:    HookErrCommandOutputInvalid,
			Command: h.program,
			Detail:  err.Error(),
		}
	}
	return decision, nil
}

// Events implements Hook.
func (h *CommandHook) Events() []HookEvent { return h.events }

// Name implements Hook.
func (h *CommandHook) Name() string { return h.name }

// SyncMode implements Hook.
func (h *CommandHook) SyncMode() HookSync { return h.syncMode }

var _ Hook = (*CommandHook)(nil)
