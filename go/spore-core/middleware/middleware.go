// Package middleware — issue #11 `MiddlewareChain`: cross-cutting interception
// of the agent loop at six hook points.
//
// Middleware observes and optionally modifies the hot-path data threaded
// through the loop (task, session state, tool calls, tool results, final
// response). The chain is a registry plus a priority-ordered fan-out
// evaluator with short-circuit semantics on Halt / SurfaceToHuman.
//
// See `docs/harness-engineering-concepts.md` § "Middleware Chain" and the
// Rust reference at `rust/crates/spore-core/src/middleware.rs`.
//
// Rules enforced
//   - Before hooks (BeforeSession, BeforeTurn, BeforeTool, BeforeCompletion)
//     run sorted by priority ascending — lowest first.
//   - After hooks (AfterTool, AfterSession) run sorted by priority
//     descending — highest first (wrapping pattern).
//   - First Halt or SurfaceToHuman stops the chain; downstream middleware
//     does not run.
//   - ForceAnotherTurn is valid only on BeforeCompletion. All injections
//     from middleware that returned it are concatenated (newline-joined)
//     into a single decision; the chain continues running remaining
//     middleware so they may still Halt / SurfaceToHuman.
//   - ForceAnotherTurn returned from any other hook is an IllegalDecision
//     surfaced as Halt.
//   - SurfaceToHuman returned outside BeforeTool / BeforeCompletion is an
//     IllegalDecision surfaced as Halt.
//   - Middleware must not call ModelInterface or ToolRegistry. Not in
//     scope of any HookContext.
//   - Middleware must not hold per-session state on the receiver keyed by
//     SessionID — keep it in an external map and clear in AfterSession.
//     Not enforced; documented in each standard middleware.
//
// Standard middleware shipped in this package mirror the Rust reference
// subset: TracingMiddleware (priority math.MinInt32 — fires first on before,
// last on after), PatchToolCallsMiddleware (math.MinInt32+1 on BeforeTool —
// runs ahead of any other BeforeTool middleware), LoopDetectionMiddleware,
// PreCompletionChecklistMiddleware, TokenBudgetMiddleware.
package middleware

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// HookPoint
// ============================================================================

// HookPoint identifies one of the six hook points in the agent loop.
type HookPoint string

const (
	// HookBeforeSession fires once at the start of a session.
	HookBeforeSession HookPoint = "before_session"
	// HookBeforeTurn fires before each agent turn.
	HookBeforeTurn HookPoint = "before_turn"
	// HookBeforeTool fires before tool calls dispatch.
	HookBeforeTool HookPoint = "before_tool"
	// HookAfterTool fires after tool calls return.
	HookAfterTool HookPoint = "after_tool"
	// HookBeforeCompletion fires before the agent emits a final response.
	HookBeforeCompletion HookPoint = "before_completion"
	// HookAfterSession fires once at the end of a session.
	HookAfterSession HookPoint = "after_session"
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

// ============================================================================
// HookContext (tagged union — exactly one variant payload is meaningful per
// firing; see Point()).
// ============================================================================

// HookContext is the per-firing payload handed to Middleware.Handle.
//
// Mutable fields are passed by pointer where the spec allows modification
// (BeforeTurn session, BeforeTool calls, AfterTool results). The harness uses
// MiddlewareDecisionContinueWithModification as the observability signal that
// a mutation occurred.
type HookContext struct {
	Point HookPoint

	// BeforeSession
	Task      *sporecore.Task
	SessionID *sporecore.SessionID

	// BeforeTurn
	Session    *sporecore.SessionState
	TurnNumber uint32

	// BeforeTool — Calls is mutable (in-place modification permitted).
	Calls *[]sporecore.ToolCall

	// AfterTool — Results is mutable; Calls is read-only snapshot.
	CallsRO []sporecore.ToolCall
	Results *[]sporecore.ToolResult

	// BeforeCompletion
	Response     string
	SessionState *sporecore.SessionState

	// AfterSession
	Result *sporecore.RunResult
}

// ============================================================================
// MiddlewareDecision
// ============================================================================

// MiddlewareDecisionKind discriminates MiddlewareDecision variants.
type MiddlewareDecisionKind string

const (
	// DecisionContinue — proceed with no observable change.
	DecisionContinue MiddlewareDecisionKind = "continue"
	// DecisionContinueWithModification — proceed; this middleware mutated
	// the borrowed context. Semantically equivalent to Continue for chain
	// control flow.
	DecisionContinueWithModification MiddlewareDecisionKind = "continue_with_modification"
	// DecisionForceAnotherTurn — valid only on BeforeCompletion. The chain
	// concatenates injections from every middleware that returned this and
	// surfaces one combined decision.
	DecisionForceAnotherTurn MiddlewareDecisionKind = "force_another_turn"
	// DecisionHalt — stop the loop with reason.
	DecisionHalt MiddlewareDecisionKind = "halt"
	// DecisionSurfaceToHuman — valid only on BeforeTool / BeforeCompletion.
	// First occurrence wins; remaining middleware do not run.
	DecisionSurfaceToHuman MiddlewareDecisionKind = "surface_to_human"
)

// MiddlewareDecision is the tagged-union return value from Middleware.Handle.
type MiddlewareDecision struct {
	Kind MiddlewareDecisionKind `json:"kind"`
	// force_another_turn
	Inject string `json:"-"`
	// halt
	Reason string `json:"-"`
	// surface_to_human
	Request *sporecore.HumanRequest `json:"-"`
}

// DecisionContinueVal is the canonical Continue decision.
func DecisionContinueVal() MiddlewareDecision {
	return MiddlewareDecision{Kind: DecisionContinue}
}

// DecisionContinueWithModificationVal is the canonical
// ContinueWithModification decision.
func DecisionContinueWithModificationVal() MiddlewareDecision {
	return MiddlewareDecision{Kind: DecisionContinueWithModification}
}

// DecisionForceAnotherTurnVal builds a ForceAnotherTurn decision.
func DecisionForceAnotherTurnVal(inject string) MiddlewareDecision {
	return MiddlewareDecision{Kind: DecisionForceAnotherTurn, Inject: inject}
}

// DecisionHaltVal builds a Halt decision.
func DecisionHaltVal(reason string) MiddlewareDecision {
	return MiddlewareDecision{Kind: DecisionHalt, Reason: reason}
}

// DecisionSurfaceToHumanVal builds a SurfaceToHuman decision.
func DecisionSurfaceToHumanVal(req sporecore.HumanRequest) MiddlewareDecision {
	return MiddlewareDecision{Kind: DecisionSurfaceToHuman, Request: &req}
}

// MarshalJSON serialises as a flat tagged object matching the Rust shape.
func (d MiddlewareDecision) MarshalJSON() ([]byte, error) {
	switch d.Kind {
	case DecisionContinue, DecisionContinueWithModification:
		return json.Marshal(struct {
			Kind MiddlewareDecisionKind `json:"kind"`
		}{d.Kind})
	case DecisionForceAnotherTurn:
		return json.Marshal(struct {
			Kind   MiddlewareDecisionKind `json:"kind"`
			Inject string                 `json:"inject"`
		}{d.Kind, d.Inject})
	case DecisionHalt:
		return json.Marshal(struct {
			Kind   MiddlewareDecisionKind `json:"kind"`
			Reason string                 `json:"reason"`
		}{d.Kind, d.Reason})
	case DecisionSurfaceToHuman:
		return json.Marshal(struct {
			Kind    MiddlewareDecisionKind  `json:"kind"`
			Request *sporecore.HumanRequest `json:"request"`
		}{d.Kind, d.Request})
	default:
		return nil, fmt.Errorf("MiddlewareDecision: unknown kind %q", d.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (d *MiddlewareDecision) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind    MiddlewareDecisionKind  `json:"kind"`
		Inject  string                  `json:"inject"`
		Reason  string                  `json:"reason"`
		Request *sporecore.HumanRequest `json:"request"`
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

// ============================================================================
// Errors
// ============================================================================

// MiddlewareErrorKind discriminates MiddlewareError variants.
type MiddlewareErrorKind string

const (
	// ErrKindAlreadyRegistered — middleware with this name already exists.
	ErrKindAlreadyRegistered MiddlewareErrorKind = "already_registered"
	// ErrKindNoHooks — middleware declared an empty Hooks() list.
	ErrKindNoHooks MiddlewareErrorKind = "no_hooks"
	// ErrKindIllegalDecision — middleware returned a decision the hook
	// does not permit (e.g. SurfaceToHuman from BeforeTurn).
	ErrKindIllegalDecision MiddlewareErrorKind = "illegal_decision"
)

// MiddlewareError is the typed error returned by MiddlewareChain methods.
type MiddlewareError struct {
	Kind     MiddlewareErrorKind `json:"kind"`
	Name     string              `json:"name,omitempty"`
	Hook     HookPoint           `json:"hook,omitempty"`
	Decision string              `json:"decision,omitempty"`
}

// Error implements error.
func (e *MiddlewareError) Error() string {
	switch e.Kind {
	case ErrKindAlreadyRegistered:
		return fmt.Sprintf("middleware already registered: %q", e.Name)
	case ErrKindNoHooks:
		return fmt.Sprintf("middleware %q declared zero hooks", e.Name)
	case ErrKindIllegalDecision:
		return fmt.Sprintf("middleware %q returned %s from %s which does not allow it",
			e.Name, e.Decision, e.Hook)
	default:
		return fmt.Sprintf("middleware error: %s", e.Kind)
	}
}

// ============================================================================
// Interfaces
// ============================================================================

// Middleware is a single interceptor registered with a MiddlewareChain.
//
// Implementations MUST be safe to call concurrently. They MUST NOT call
// ModelInterface or ToolRegistry (design constraint — neither is in scope of
// any HookContext). They MUST NOT hold per-session state on the receiver
// keyed by SessionID — keep it in an external map and clear in
// AfterSession.
type Middleware interface {
	// Handle inspects (and optionally mutates) the hot-path data in ctx,
	// returning a decision.
	Handle(ctx context.Context, hctx HookContext) (MiddlewareDecision, error)
	// Hooks declares which HookPoints this middleware fires at. Must be
	// non-empty.
	Hooks() []HookPoint
	// Priority controls ordering. Default 0. Lower runs first on before
	// hooks; higher runs first on after hooks.
	Priority() int
	// Name is the unique registration identity.
	Name() string
}

// MiddlewareChain is the registry plus fan-out evaluator.
type MiddlewareChain interface {
	// Register validates and inserts a middleware. Duplicate names return
	// AlreadyRegistered; empty Hooks() list returns NoHooks.
	Register(m Middleware) error

	FireBeforeSession(ctx context.Context, task *sporecore.Task, sid sporecore.SessionID) (MiddlewareDecision, error)
	FireBeforeTurn(ctx context.Context, session *sporecore.SessionState, turn uint32) (MiddlewareDecision, error)
	FireBeforeTool(ctx context.Context, calls *[]sporecore.ToolCall, turn uint32) (MiddlewareDecision, error)
	FireAfterTool(ctx context.Context, calls []sporecore.ToolCall, results *[]sporecore.ToolResult) (MiddlewareDecision, error)
	FireBeforeCompletion(ctx context.Context, response string, turn uint32, state *sporecore.SessionState) (MiddlewareDecision, error)
	FireAfterSession(ctx context.Context, result *sporecore.RunResult, sid sporecore.SessionID) error
}

// ============================================================================
// StandardMiddlewareChain
// ============================================================================

type entry struct {
	name     string
	priority int
	hooks    []HookPoint
	mw       Middleware
}

// StandardMiddlewareChain is the in-memory reference implementation.
type StandardMiddlewareChain struct {
	mu          sync.Mutex
	middlewares []entry
}

// NewStandardMiddlewareChain returns an empty chain.
func NewStandardMiddlewareChain() *StandardMiddlewareChain {
	return &StandardMiddlewareChain{}
}

// Register implements MiddlewareChain.
func (c *StandardMiddlewareChain) Register(m Middleware) error {
	name := m.Name()
	hooks := m.Hooks()
	if len(hooks) == 0 {
		return &MiddlewareError{Kind: ErrKindNoHooks, Name: name}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, e := range c.middlewares {
		if e.name == name {
			return &MiddlewareError{Kind: ErrKindAlreadyRegistered, Name: name}
		}
	}
	c.middlewares = append(c.middlewares, entry{
		name:     name,
		priority: m.Priority(),
		hooks:    append([]HookPoint(nil), hooks...),
		mw:       m,
	})
	return nil
}

// eligible returns a fresh slice of entries that subscribe to the hook,
// sorted per the ordering rule. Caller must hold c.mu.
func (c *StandardMiddlewareChain) eligible(hook HookPoint) []entry {
	var out []entry
	for _, e := range c.middlewares {
		for _, h := range e.hooks {
			if h == hook {
				out = append(out, e)
				break
			}
		}
	}
	if hook.IsAfter() {
		sort.SliceStable(out, func(i, j int) bool {
			if out[i].priority != out[j].priority {
				return out[i].priority > out[j].priority
			}
			return out[i].name < out[j].name
		})
	} else {
		sort.SliceStable(out, func(i, j int) bool {
			if out[i].priority != out[j].priority {
				return out[i].priority < out[j].priority
			}
			return out[i].name < out[j].name
		})
	}
	return out
}

func (c *StandardMiddlewareChain) snapshot(hook HookPoint) []entry {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.eligible(hook)
}

func validateDecision(name string, hook HookPoint, d MiddlewareDecision) (MiddlewareDecision, error) {
	switch d.Kind {
	case DecisionSurfaceToHuman:
		if !hook.AllowsSurfaceToHuman() {
			return d, &MiddlewareError{
				Kind: ErrKindIllegalDecision, Name: name, Hook: hook, Decision: "SurfaceToHuman",
			}
		}
	case DecisionForceAnotherTurn:
		if !hook.AllowsForceAnotherTurn() {
			return d, &MiddlewareError{
				Kind: ErrKindIllegalDecision, Name: name, Hook: hook, Decision: "ForceAnotherTurn",
			}
		}
	}
	return d, nil
}

// FireBeforeSession implements MiddlewareChain.
func (c *StandardMiddlewareChain) FireBeforeSession(ctx context.Context, task *sporecore.Task, sid sporecore.SessionID) (MiddlewareDecision, error) {
	for _, e := range c.snapshot(HookBeforeSession) {
		d, err := e.mw.Handle(ctx, HookContext{
			Point: HookBeforeSession, Task: task, SessionID: &sid,
		})
		if err != nil {
			return DecisionHaltVal(err.Error()), err
		}
		d, verr := validateDecision(e.name, HookBeforeSession, d)
		if verr != nil {
			return DecisionHaltVal(verr.Error()), nil
		}
		switch d.Kind {
		case DecisionContinue, DecisionContinueWithModification:
			continue
		default:
			return d, nil
		}
	}
	return DecisionContinueVal(), nil
}

// FireBeforeTurn implements MiddlewareChain.
func (c *StandardMiddlewareChain) FireBeforeTurn(ctx context.Context, session *sporecore.SessionState, turn uint32) (MiddlewareDecision, error) {
	anyModified := false
	for _, e := range c.snapshot(HookBeforeTurn) {
		d, err := e.mw.Handle(ctx, HookContext{
			Point: HookBeforeTurn, Session: session, TurnNumber: turn,
		})
		if err != nil {
			return DecisionHaltVal(err.Error()), err
		}
		d, verr := validateDecision(e.name, HookBeforeTurn, d)
		if verr != nil {
			return DecisionHaltVal(verr.Error()), nil
		}
		switch d.Kind {
		case DecisionContinue:
			continue
		case DecisionContinueWithModification:
			anyModified = true
			continue
		default:
			return d, nil
		}
	}
	if anyModified {
		return DecisionContinueWithModificationVal(), nil
	}
	return DecisionContinueVal(), nil
}

// FireBeforeTool implements MiddlewareChain.
func (c *StandardMiddlewareChain) FireBeforeTool(ctx context.Context, calls *[]sporecore.ToolCall, turn uint32) (MiddlewareDecision, error) {
	anyModified := false
	for _, e := range c.snapshot(HookBeforeTool) {
		d, err := e.mw.Handle(ctx, HookContext{
			Point: HookBeforeTool, Calls: calls, TurnNumber: turn,
		})
		if err != nil {
			return DecisionHaltVal(err.Error()), err
		}
		d, verr := validateDecision(e.name, HookBeforeTool, d)
		if verr != nil {
			return DecisionHaltVal(verr.Error()), nil
		}
		switch d.Kind {
		case DecisionContinue:
			continue
		case DecisionContinueWithModification:
			anyModified = true
			continue
		default:
			return d, nil
		}
	}
	if anyModified {
		return DecisionContinueWithModificationVal(), nil
	}
	return DecisionContinueVal(), nil
}

// FireAfterTool implements MiddlewareChain.
func (c *StandardMiddlewareChain) FireAfterTool(ctx context.Context, calls []sporecore.ToolCall, results *[]sporecore.ToolResult) (MiddlewareDecision, error) {
	anyModified := false
	for _, e := range c.snapshot(HookAfterTool) {
		d, err := e.mw.Handle(ctx, HookContext{
			Point: HookAfterTool, CallsRO: calls, Results: results,
		})
		if err != nil {
			return DecisionHaltVal(err.Error()), err
		}
		d, verr := validateDecision(e.name, HookAfterTool, d)
		if verr != nil {
			return DecisionHaltVal(verr.Error()), nil
		}
		switch d.Kind {
		case DecisionContinue:
			continue
		case DecisionContinueWithModification:
			anyModified = true
			continue
		default:
			return d, nil
		}
	}
	if anyModified {
		return DecisionContinueWithModificationVal(), nil
	}
	return DecisionContinueVal(), nil
}

// FireBeforeCompletion implements MiddlewareChain.
func (c *StandardMiddlewareChain) FireBeforeCompletion(ctx context.Context, response string, turn uint32, state *sporecore.SessionState) (MiddlewareDecision, error) {
	var injections []string
	for _, e := range c.snapshot(HookBeforeCompletion) {
		d, err := e.mw.Handle(ctx, HookContext{
			Point: HookBeforeCompletion, Response: response, TurnNumber: turn, SessionState: state,
		})
		if err != nil {
			return DecisionHaltVal(err.Error()), err
		}
		d, verr := validateDecision(e.name, HookBeforeCompletion, d)
		if verr != nil {
			return DecisionHaltVal(verr.Error()), nil
		}
		switch d.Kind {
		case DecisionContinue, DecisionContinueWithModification:
			continue
		case DecisionForceAnotherTurn:
			injections = append(injections, d.Inject)
			// chain continues per spec
		case DecisionHalt, DecisionSurfaceToHuman:
			return d, nil
		}
	}
	if len(injections) > 0 {
		return DecisionForceAnotherTurnVal(strings.Join(injections, "\n")), nil
	}
	return DecisionContinueVal(), nil
}

// FireAfterSession implements MiddlewareChain.
//
// After-session hooks fire to completion regardless of their decisions —
// the session is already terminating, so Halt / SurfaceToHuman responses are
// not actionable. Errors from a middleware abort the remaining fan-out and
// are returned to the caller.
func (c *StandardMiddlewareChain) FireAfterSession(ctx context.Context, result *sporecore.RunResult, sid sporecore.SessionID) error {
	for _, e := range c.snapshot(HookAfterSession) {
		if _, err := e.mw.Handle(ctx, HookContext{
			Point: HookAfterSession, Result: result, SessionID: &sid,
		}); err != nil {
			return err
		}
	}
	return nil
}

// Compile-time interface check.
var _ MiddlewareChain = (*StandardMiddlewareChain)(nil)

// ============================================================================
// Standard middleware: TracingMiddleware
// ============================================================================

// TracingMiddleware records every firing. Priority math.MinInt32 — fires
// first on before hooks, last on after hooks.
type TracingMiddleware struct {
	name string
	mu   sync.Mutex
	log  []TracingEntry
}

// TracingEntry is one recorded firing.
type TracingEntry struct {
	Point HookPoint
	Turn  uint32
}

// NewTracingMiddleware returns a tracer named "tracing".
func NewTracingMiddleware() *TracingMiddleware {
	return &TracingMiddleware{name: "tracing"}
}

// NewTracingMiddlewareNamed returns a tracer with a custom name.
func NewTracingMiddlewareNamed(name string) *TracingMiddleware {
	return &TracingMiddleware{name: name}
}

// Entries returns a copy of the firing log.
func (t *TracingMiddleware) Entries() []TracingEntry {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]TracingEntry, len(t.log))
	copy(out, t.log)
	return out
}

// Handle implements Middleware.
func (t *TracingMiddleware) Handle(_ context.Context, hctx HookContext) (MiddlewareDecision, error) {
	t.mu.Lock()
	t.log = append(t.log, TracingEntry{Point: hctx.Point, Turn: hctx.TurnNumber})
	t.mu.Unlock()
	return DecisionContinueVal(), nil
}

// Hooks implements Middleware.
func (t *TracingMiddleware) Hooks() []HookPoint {
	return []HookPoint{
		HookBeforeSession, HookBeforeTurn, HookBeforeTool,
		HookAfterTool, HookBeforeCompletion, HookAfterSession,
	}
}

// Priority implements Middleware.
func (t *TracingMiddleware) Priority() int { return math.MinInt32 }

// Name implements Middleware.
func (t *TracingMiddleware) Name() string { return t.name }

// ============================================================================
// Standard middleware: PatchToolCallsMiddleware
// ============================================================================

// PatchKind classifies a tool-call patch on a PatchEvent. The string values
// match the observability.PatchType "kind" tags so the adapter can map them
// without a cross-package import (observability imports middleware, not the
// reverse).
type PatchKind string

const (
	// PatchKindMalformedJSON — JSON parse error, repair attempted.
	PatchKindMalformedJSON PatchKind = "malformed_json"
	// PatchKindDanglingToolCall — structurally incomplete call (e.g. empty
	// tool name) completed with defaults.
	PatchKindDanglingToolCall PatchKind = "dangling_tool_call"
	// PatchKindParameterCoercion — a parameter value was coerced.
	PatchKindParameterCoercion PatchKind = "parameter_coercion"
)

// PatchEvent is the observability-independent description of a single
// tool-call patch (issue #28). PatchToolCallsMiddleware hands one of these to
// its PatchEmitter on every patch; the emitter (typically the observability
// provider) stamps identity and records a warn-level PatchSpan.
//
// Carrying primitives here — rather than an observability.PatchSpan — keeps
// the dependency direction one-way: observability imports middleware, so this
// package must not import observability.
type PatchEvent struct {
	// SessionID and TaskID are the identity captured at BeforeSession.
	SessionID sporecore.SessionID
	TaskID    sporecore.TaskID
	// CallID and ToolName identify the patched call (ToolName is the patched
	// name actually dispatched).
	CallID   string
	ToolName string
	// Original is the parameters as the model sent them; Patched is what was
	// dispatched.
	Original json.RawMessage
	Patched  json.RawMessage
	// Kind classifies the patch; Reason/Error/Field/From/To carry the
	// variant-specific detail (only the fields relevant to Kind are set).
	Kind   PatchKind
	Reason string
	Error  string
	Field  string
	From   string
	To     string
}

// PatchEmitter is the consumer-side sink for PatchToolCallsMiddleware. The
// observability provider implements it (see observability.PatchEmitterAdapter)
// so a patch is recorded as a warn-level PatchSpan, but any backend may accept
// these events. EmitPatch is fire-and-forget — it must not block.
type PatchEmitter interface {
	EmitPatch(event PatchEvent)
}

// PatchToolCallsMiddleware renames any tool call whose name is empty or
// whitespace-only to a configurable fallback. Runs at math.MinInt32+1 on
// BeforeTool so it always precedes other BeforeTool middleware (per spec).
//
// Observability (issue #28): this middleware is an always-on, highest-priority
// action mutator. To keep it from silently rewriting calls, every patch emits
// a warn-level PatchEvent (classified as DanglingToolCall for the empty-name
// repair) to an injected PatchEmitter before the patched call proceeds. The
// shared HookContext.BeforeTool does not carry session/task identity, so this
// middleware captures it at BeforeSession into a guarded field and reads it at
// BeforeTool — the same external-identity pattern used by other middleware.
type PatchToolCallsMiddleware struct {
	name     string
	fallback string

	emitter PatchEmitter

	// identity is captured at BeforeSession, read at BeforeTool.
	mu       sync.Mutex
	identity *patchIdentity
}

type patchIdentity struct {
	sessionID sporecore.SessionID
	taskID    sporecore.TaskID
}

// NewPatchToolCallsMiddleware builds a patch middleware with the supplied
// fallback name. No observability emitter is wired — patching is silent.
func NewPatchToolCallsMiddleware(fallback string) *PatchToolCallsMiddleware {
	return &PatchToolCallsMiddleware{name: "patch-tool-calls", fallback: fallback}
}

// WithObservability injects the patch emitter. Patches emitted after this is
// set produce warn-level patch events (issue #28). Returns the receiver for
// chaining.
func (p *PatchToolCallsMiddleware) WithObservability(emitter PatchEmitter) *PatchToolCallsMiddleware {
	p.emitter = emitter
	return p
}

// Clear forgets the captured session identity. Tests use this to simulate
// session boundaries.
func (p *PatchToolCallsMiddleware) Clear() {
	p.mu.Lock()
	p.identity = nil
	p.mu.Unlock()
}

// emitPatchEvent stamps identity and forwards a patch event. A no-op when no
// emitter is wired (keeps NewPatchToolCallsMiddleware test-friendly).
func (p *PatchToolCallsMiddleware) emitPatchEvent(callID, toolName string, original, patched json.RawMessage, kind PatchKind, reason string) {
	if p.emitter == nil {
		return
	}
	p.mu.Lock()
	id := p.identity
	p.mu.Unlock()
	ev := PatchEvent{
		CallID:   callID,
		ToolName: toolName,
		Original: original,
		Patched:  patched,
		Kind:     kind,
		Reason:   reason,
	}
	if id != nil {
		ev.SessionID = id.sessionID
		ev.TaskID = id.taskID
	}
	p.emitter.EmitPatch(ev)
}

// Handle implements Middleware.
func (p *PatchToolCallsMiddleware) Handle(_ context.Context, hctx HookContext) (MiddlewareDecision, error) {
	switch hctx.Point {
	case HookBeforeSession:
		if hctx.Task != nil && hctx.SessionID != nil {
			p.mu.Lock()
			p.identity = &patchIdentity{sessionID: *hctx.SessionID, taskID: hctx.Task.ID}
			p.mu.Unlock()
		}
		return DecisionContinueVal(), nil
	case HookBeforeTool:
		if hctx.Calls == nil {
			return DecisionContinueVal(), nil
		}
		modified := false
		for i := range *hctx.Calls {
			call := &(*hctx.Calls)[i]
			if strings.TrimSpace(call.Name) == "" {
				// Capture the original parameters before mutating the name.
				original := call.Input
				call.Name = p.fallback
				modified = true
				// Classify the empty-name repair as a dangling tool call and
				// emit a warn-level patch event. The patched name is now on the
				// call; parameters are unchanged for this repair.
				p.emitPatchEvent(call.ID, call.Name, original, call.Input,
					PatchKindDanglingToolCall, "empty tool name")
			}
		}
		if modified {
			return DecisionContinueWithModificationVal(), nil
		}
		return DecisionContinueVal(), nil
	default:
		return DecisionContinueVal(), nil
	}
}

// Hooks implements Middleware.
func (p *PatchToolCallsMiddleware) Hooks() []HookPoint {
	return []HookPoint{HookBeforeSession, HookBeforeTool}
}

// Priority implements Middleware.
func (p *PatchToolCallsMiddleware) Priority() int { return math.MinInt32 + 1 }

// Name implements Middleware.
func (p *PatchToolCallsMiddleware) Name() string { return p.name }

// ============================================================================
// Standard middleware: LoopDetectionMiddleware
// ============================================================================

// LoopDetectionMiddleware tracks per-path edit counts for a given tool
// (e.g. "edit") and annotates tool result Content with a "[loop-detection]"
// warning once a path has been touched `threshold` times. Mutates results
// in place and signals via ContinueWithModification.
//
// Per-session state lives on this middleware via an external map keyed by
// path. Real production code keys by SessionID and clears in AfterSession;
// for the standalone reference impl callers may invoke Clear() between
// sessions.
type LoopDetectionMiddleware struct {
	name      string
	threshold uint32
	toolName  string
	mu        sync.Mutex
	counts    map[string]uint32
}

// NewLoopDetectionMiddleware builds a loop detector for the named tool.
func NewLoopDetectionMiddleware(toolName string, threshold uint32) *LoopDetectionMiddleware {
	return &LoopDetectionMiddleware{
		name:      "loop-detection",
		threshold: threshold,
		toolName:  toolName,
		counts:    make(map[string]uint32),
	}
}

// Clear resets the per-path counters. Tests use this to simulate session
// boundaries.
func (l *LoopDetectionMiddleware) Clear() {
	l.mu.Lock()
	l.counts = make(map[string]uint32)
	l.mu.Unlock()
}

// Handle implements Middleware.
func (l *LoopDetectionMiddleware) Handle(_ context.Context, hctx HookContext) (MiddlewareDecision, error) {
	if hctx.Point != HookAfterTool || hctx.Results == nil {
		return DecisionContinueVal(), nil
	}
	modified := false
	n := len(hctx.CallsRO)
	if n > len(*hctx.Results) {
		n = len(*hctx.Results)
	}
	for i := 0; i < n; i++ {
		call := hctx.CallsRO[i]
		if call.Name != l.toolName {
			continue
		}
		path := extractPath(call.Input)
		if path == "" {
			continue
		}
		l.mu.Lock()
		l.counts[path]++
		count := l.counts[path]
		l.mu.Unlock()
		if count >= l.threshold {
			r := &(*hctx.Results)[i]
			if r.IsError {
				continue
			}
			if !strings.Contains(r.Content, "[loop-detection]") {
				warning := fmt.Sprintf("[loop-detection] %s has been edited %d times — reconsider", path, count)
				if r.Content == "" {
					r.Content = warning
				} else {
					r.Content = r.Content + "\n\n" + warning
				}
				modified = true
			}
		}
	}
	if modified {
		return DecisionContinueWithModificationVal(), nil
	}
	return DecisionContinueVal(), nil
}

// extractPath pulls the "path" field from a tool-call input JSON object.
// Returns "" when absent / non-string / malformed.
func extractPath(input json.RawMessage) string {
	if len(input) == 0 {
		return ""
	}
	var probe struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(input, &probe); err != nil {
		return ""
	}
	return probe.Path
}

// Hooks implements Middleware.
func (l *LoopDetectionMiddleware) Hooks() []HookPoint { return []HookPoint{HookAfterTool} }

// Priority implements Middleware.
func (l *LoopDetectionMiddleware) Priority() int { return 0 }

// Name implements Middleware.
func (l *LoopDetectionMiddleware) Name() string { return l.name }

// ============================================================================
// Standard middleware: PreCompletionChecklistMiddleware
// ============================================================================

// PreCompletionChecklistMiddleware forces another turn at BeforeCompletion if
// the agent's response is missing any required substring.
type PreCompletionChecklistMiddleware struct {
	name     string
	required []string
}

// NewPreCompletionChecklistMiddleware builds a checklist middleware from the
// supplied required substrings.
func NewPreCompletionChecklistMiddleware(required []string) *PreCompletionChecklistMiddleware {
	return &PreCompletionChecklistMiddleware{
		name:     "pre-completion-checklist",
		required: append([]string(nil), required...),
	}
}

// Handle implements Middleware.
func (p *PreCompletionChecklistMiddleware) Handle(_ context.Context, hctx HookContext) (MiddlewareDecision, error) {
	if hctx.Point != HookBeforeCompletion {
		return DecisionContinueVal(), nil
	}
	var missing []string
	for _, s := range p.required {
		if !strings.Contains(hctx.Response, s) {
			missing = append(missing, s)
		}
	}
	if len(missing) == 0 {
		return DecisionContinueVal(), nil
	}
	return DecisionForceAnotherTurnVal(
		fmt.Sprintf("Verification incomplete. Required items not addressed: %s", strings.Join(missing, ", ")),
	), nil
}

// Hooks implements Middleware.
func (p *PreCompletionChecklistMiddleware) Hooks() []HookPoint {
	return []HookPoint{HookBeforeCompletion}
}

// Priority implements Middleware.
func (p *PreCompletionChecklistMiddleware) Priority() int { return 0 }

// Name implements Middleware.
func (p *PreCompletionChecklistMiddleware) Name() string { return p.name }

// ============================================================================
// Standard middleware: TokenBudgetMiddleware
// ============================================================================

// TokenBudgetMiddleware halts at BeforeTurn when cumulative token spend
// reaches the configured limit. Spend is tracked atomically so callers may
// Record() from any goroutine.
type TokenBudgetMiddleware struct {
	name  string
	limit uint64
	spent atomic.Uint64
}

// NewTokenBudgetMiddleware builds a budget middleware with the supplied
// token limit.
func NewTokenBudgetMiddleware(limit uint64) *TokenBudgetMiddleware {
	return &TokenBudgetMiddleware{name: "token-budget", limit: limit}
}

// Record adds tokens to the cumulative spend.
func (t *TokenBudgetMiddleware) Record(tokens uint64) {
	t.spent.Add(tokens)
}

// Spent returns the current cumulative spend.
func (t *TokenBudgetMiddleware) Spent() uint64 {
	return t.spent.Load()
}

// Handle implements Middleware.
func (t *TokenBudgetMiddleware) Handle(_ context.Context, _ HookContext) (MiddlewareDecision, error) {
	spent := t.spent.Load()
	if spent >= t.limit {
		return DecisionHaltVal(fmt.Sprintf("token budget exhausted: %d/%d", spent, t.limit)), nil
	}
	return DecisionContinueVal(), nil
}

// Hooks implements Middleware.
func (t *TokenBudgetMiddleware) Hooks() []HookPoint { return []HookPoint{HookBeforeTurn} }

// Priority implements Middleware.
func (t *TokenBudgetMiddleware) Priority() int { return 0 }

// Name implements Middleware.
func (t *TokenBudgetMiddleware) Name() string { return t.name }
