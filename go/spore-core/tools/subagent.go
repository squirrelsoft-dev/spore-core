// SubagentTool — wraps a child Harness and exposes it as a Tool.
//
// Per spec (issue #5), subagents cannot spawn their own subagents. The
// restriction is enforced at construction time by inspecting the child's
// ToolRegistry via HasSubagentTools.

package tools

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// ContextSharing — sealed interface with three implementing structs.
// ============================================================================

// ContextSharing describes how a subagent inherits (or does not inherit)
// context from its parent. Implementations: Isolated, SharedSession,
// SummaryHandoff. The unexported sealedContextSharing() method seals the
// interface — only the three types in this file can satisfy it.
type ContextSharing interface {
	sealedContextSharing()
	// Kind returns the snake_case tag used in JSON serialisation.
	Kind() string
}

// Isolated means the subagent runs in a fresh SessionId with no inherited
// state.
type Isolated struct{}

func (Isolated) sealedContextSharing() {}
func (Isolated) Kind() string          { return "isolated" }

// SharedSession means the subagent shares its parent's SessionId and
// therefore its episodic memory.
type SharedSession struct {
	SessionID sporecore.SessionID `json:"session_id"`
}

func (SharedSession) sealedContextSharing() {}
func (SharedSession) Kind() string          { return "shared_session" }

// SummaryHandoff means the subagent runs in a fresh session but with a
// short summary string injected as a synthetic extras entry.
type SummaryHandoff struct {
	Summary string `json:"summary"`
}

func (SummaryHandoff) sealedContextSharing() {}
func (SummaryHandoff) Kind() string          { return "summary_handoff" }

// MarshalContextSharing serialises a ContextSharing as a flat tagged object
// matching the Rust enum layout.
func MarshalContextSharing(c ContextSharing) ([]byte, error) {
	switch v := c.(type) {
	case Isolated:
		return json.Marshal(struct {
			Kind string `json:"kind"`
		}{v.Kind()})
	case SharedSession:
		return json.Marshal(struct {
			Kind      string              `json:"kind"`
			SessionID sporecore.SessionID `json:"session_id"`
		}{v.Kind(), v.SessionID})
	case SummaryHandoff:
		return json.Marshal(struct {
			Kind    string `json:"kind"`
			Summary string `json:"summary"`
		}{v.Kind(), v.Summary})
	default:
		return nil, fmt.Errorf("unknown ContextSharing variant %T", c)
	}
}

// UnmarshalContextSharing parses the flat tagged form back into a
// ContextSharing.
func UnmarshalContextSharing(data []byte) (ContextSharing, error) {
	var probe struct {
		Kind      string              `json:"kind"`
		SessionID sporecore.SessionID `json:"session_id"`
		Summary   string              `json:"summary"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, err
	}
	switch probe.Kind {
	case "isolated":
		return Isolated{}, nil
	case "shared_session":
		return SharedSession{SessionID: probe.SessionID}, nil
	case "summary_handoff":
		return SummaryHandoff{Summary: probe.Summary}, nil
	default:
		return nil, fmt.Errorf("unknown ContextSharing kind %q", probe.Kind)
	}
}

// ============================================================================
// BuildError — construction-time error.
// ============================================================================

// BuildError is returned by NewSubagentTool when the child harness is mis-
// configured (currently only when it contains a SubagentTool, which would
// violate the depth-1 rule).
type BuildError struct {
	Reason string
}

func (e *BuildError) Error() string {
	return fmt.Sprintf("invalid configuration: %s", e.Reason)
}

// ErrInvalidConfiguration is the sentinel returned from BuildError; useful
// with errors.Is.
var ErrInvalidConfiguration = errors.New("invalid configuration")

func (e *BuildError) Is(target error) bool { return target == ErrInvalidConfiguration }

// ============================================================================
// SubagentTool
// ============================================================================

// SubagentTool wraps a child sporecore.Harness as a Tool. The parent agent
// invokes it via ToolRegistry just like any other tool.
type SubagentTool struct {
	name           string
	description    string
	inputSchema    json.RawMessage
	timeout        time.Duration
	contextSharing ContextSharing
	harness        sporecore.Harness
	// consultHandlers is the per-kind consult-handler map (issue #114, seam A1).
	// Empty (the default) means consults are NOT mediated here — a child
	// RunResult.Consult degrades gracefully per R6 (no matching kind → Escalate).
	// Populated via WithConsultHandlers.
	consultHandlers map[string]sporecore.ConsultHandlerEntry
}

// NewSubagentTool constructs a SubagentTool. Returns *BuildError if the
// child registry already contains subagent tools (depth-1 rule).
func NewSubagentTool(
	name string,
	description string,
	inputSchema json.RawMessage,
	timeout time.Duration,
	contextSharing ContextSharing,
	harness sporecore.Harness,
	childRegistry sporecore.ToolRegistry,
) (*SubagentTool, error) {
	if childRegistry != nil && childRegistry.HasSubagentTools() {
		return nil, &BuildError{
			Reason: "child harness must not contain SubagentTool (depth-1 rule)",
		}
	}
	return &SubagentTool{
		name:           name,
		description:    description,
		inputSchema:    inputSchema,
		timeout:        timeout,
		contextSharing: contextSharing,
		harness:        harness,
	}, nil
}

// WithConsultHandlers installs the per-kind consult handlers (issue #114, seam
// A1) and returns the receiver for chaining. Typically the orchestrator passes
// a copy of its HarnessConfig.ConsultHandlers. With handlers installed, this
// tool MEDIATES a child consult internally (R2/R3) instead of letting it
// surface; without them, a child consult degrades to Escalate (R6).
func (s *SubagentTool) WithConsultHandlers(handlers map[string]sporecore.ConsultHandlerEntry) *SubagentTool {
	s.consultHandlers = handlers
	return s
}

// Name returns the registered tool name.
func (s *SubagentTool) Name() string { return s.name }

// IsSubagentTool always returns true — used by ToolRegistry.HasSubagentTools.
func (*SubagentTool) IsSubagentTool() bool { return true }

// MayProduceLargeOutput is false — subagent output is already shaped by the
// child harness.
func (*SubagentTool) MayProduceLargeOutput() bool { return false }

// Description returns the tool description (for schemas / introspection).
func (s *SubagentTool) Description() string { return s.description }

// InputSchema returns the registered JSON Schema for the tool's input.
func (s *SubagentTool) InputSchema() json.RawMessage { return s.inputSchema }

// Timeout returns the configured timeout.
func (s *SubagentTool) Timeout() time.Duration { return s.timeout }

// ContextSharing returns the configured context sharing strategy.
func (s *SubagentTool) ContextSharing() ContextSharing { return s.contextSharing }

// Schema returns a RegistryToolSchema suitable for registration.
func (s *SubagentTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        s.name,
		Description: s.description,
		Parameters:  s.inputSchema,
	}
}

// generateSessionID produces a fresh SessionID for Isolated / SummaryHandoff
// sharing modes. Uses crypto/rand for uniqueness.
func generateSessionID() sporecore.SessionID {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return sporecore.SessionID("sess_" + hex.EncodeToString(b[:]))
}

// Execute dispatches the child harness with the validated instruction.
// Failure modes:
//   - missing "instruction" param         → ToolOutput.Error (recoverable: true)
//   - child returns Success               → ToolOutput.Success
//   - child returns Failure               → ToolOutput.Error (recoverable: true)
//   - child returns WaitingForHuman       → ToolOutput.WaitingForHuman, with
//     ChildPausedState.ParentToolCallID
//     set to call.ID
//   - timeout (per s.Timeout)             → ToolOutput.Error (recoverable: true)
func (s *SubagentTool) Execute(
	ctx context.Context,
	call sporecore.ToolCall,
	_ sporecore.SandboxProvider,
	_ *sporecore.ToolContext,
) sporecore.ToolOutput {
	// Extract instruction without insisting on full param parse — keep
	// behaviour aligned with the Rust reference.
	var probe struct {
		Instruction string `json:"instruction"`
	}
	if err := json.Unmarshal(call.Input, &probe); err != nil || probe.Instruction == "" {
		return sporecore.ToolOutput{
			Kind:        sporecore.ToolOutputError,
			Message:     "invalid parameters: missing `instruction`",
			Recoverable: true,
		}
	}

	sessionID, seeded := s.resolveSessionContext()
	task := sporecore.NewTask(probe.Instruction, sessionID, sporecore.LoopStrategy{
		Kind:          sporecore.StrategyReAct,
		MaxIterations: 16,
	})

	opts := sporecore.HarnessRunOptions{Task: task}
	if seeded != nil {
		opts.SessionState = seeded
	}

	runCtx := ctx
	var cancel context.CancelFunc
	if s.timeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, s.timeout)
		defer cancel()
	}

	resultCh := make(chan sporecore.RunResult, 1)
	go func() { resultCh <- s.harness.Run(runCtx, opts) }()

	var result sporecore.RunResult
	select {
	case result = <-resultCh:
	case <-runCtx.Done():
		if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
			return sporecore.ToolOutput{
				Kind:        sporecore.ToolOutputError,
				Message:     fmt.Sprintf("subagent timed out after %ds", int(s.timeout.Seconds())),
				Recoverable: true,
			}
		}
		// Cancellation (parent ctx) — still drain.
		result = <-resultCh
	}

	// Per-kind consult counters (issue #114, R4). Each consult of a given kind
	// decrements its remaining budget; the (budget+1)th triggers the overflow
	// policy. Lives across the mediation loop below.
	consultCounts := map[string]uint32{}

	// A1 mediation loop: drive the full consult cycle internally. On a child
	// RunResult.Consult, mediate (route → run handler → resume) and continue
	// until the child reaches a terminal result.
	for {
		switch result.Kind {
		case sporecore.RunSuccess:
			return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: result.Output}
		case sporecore.RunFailure:
			return sporecore.ToolOutput{
				Kind:        sporecore.ToolOutputError,
				Message:     fmt.Sprintf("subagent failed: %s", reasonString(result.Reason)),
				Recoverable: true,
			}
		case sporecore.RunWaitingForHuman:
			child := childStateFromPaused(result.State, call.ID)
			return sporecore.ToolOutput{
				Kind:       sporecore.ToolOutputWaitingForHuman,
				ChildState: child,
				Request:    result.Request,
			}
		case sporecore.RunEscalate:
			// A subagent escalation (issue #80) propagates as a tool-side
			// escalation: the parent harness terminates cleanly and hands the
			// signal up to its own caller.
			return sporecore.ToolOutput{Kind: sporecore.ToolOutputEscalate, Signal: result.Signal}
		case sporecore.RunConsult:
			// Mid-loop consult (issue #114, R2): mediate it here — never bubble
			// it to the parent orchestrator's model.
			next, terminal, isTerminal := s.mediateConsult(runCtx, result, consultCounts, call.ID)
			if isTerminal {
				return terminal
			}
			result = next
			continue
		default:
			return sporecore.ToolOutput{
				Kind:        sporecore.ToolOutputError,
				Message:     fmt.Sprintf("subagent returned unknown run result kind %q", result.Kind),
				Recoverable: false,
			}
		}
	}
}

// mediateConsult mediates one child consult (issue #114, seam A1). Routes by
// kind, enforces the per-kind budget, runs the handler as the ORCHESTRATOR's
// direct child (R7), and resumes the worker — OR applies the overflow policy /
// graceful degradation. Returns (nextResult, "", false) to continue the
// mediation loop, or (zero, terminalOutput, true) to surface a terminal output.
func (s *SubagentTool) mediateConsult(
	ctx context.Context,
	result sporecore.RunResult,
	counts map[string]uint32,
	parentCallID string,
) (sporecore.RunResult, sporecore.ToolOutput, bool) {
	request := sporecore.ConsultRequest{}
	if result.ConsultRequest != nil {
		request = *result.ConsultRequest
	}
	state := sporecore.PausedState{}
	if result.State != nil {
		state = *result.State
	}

	// R6: no matching handler (empty map or unknown kind) → Escalate. Loud, not
	// silent. The parent harness terminates cleanly.
	entry, ok := s.consultHandlers[request.Kind]
	if !ok {
		return sporecore.RunResult{}, sporecore.ToolOutput{
			Kind: sporecore.ToolOutputEscalate,
			Signal: &sporecore.HarnessSignal{
				Kind:   sporecore.SignalAbort,
				Reason: fmt.Sprintf("no consult handler registered for kind %q", request.Kind),
			},
		}, true
	}

	// R4: per-kind budget. counts[kind] is the number of consults of this kind
	// ALREADY mediated. The handler runs while used < budget; the (budget+1)th
	// consult overflows.
	if counts[request.Kind] >= entry.Budget {
		// R5: overflow policy.
		switch entry.Overflow.Kind {
		case sporecore.ConsultOverflowSoftFail:
			// R5a: resume the worker with a BudgetExhausted response so it
			// finishes with what it has.
			resp := sporecore.NewConsultBudgetExhausted(fmt.Sprintf(
				"consult budget for kind %q exhausted; proceed without further help", request.Kind))
			next := s.harness.ResumeConsult(ctx, state, resp, nil)
			return next, sporecore.ToolOutput{}, false
		case sporecore.ConsultOverflowEscalateToHuman:
			// R5b: convert the over-budget consult into a human pause so the host
			// decides. The parent sees ToolOutput.WaitingForHuman.
			child := childStateFromPaused(&state, parentCallID)
			req := sporecore.HumanRequest{
				Kind: sporecore.HumanReqReview,
				Content: fmt.Sprintf(
					"consult budget for kind %q exhausted. situation: %s | question: %s",
					request.Kind, request.Situation, request.Question),
			}
			return sporecore.RunResult{}, sporecore.ToolOutput{
				Kind:       sporecore.ToolOutputWaitingForHuman,
				ChildState: child,
				Request:    &req,
			}, true
		default:
			return sporecore.RunResult{}, sporecore.ToolOutput{
				Kind:        sporecore.ToolOutputError,
				Message:     fmt.Sprintf("unknown consult overflow policy %q", entry.Overflow.Kind),
				Recoverable: false,
			}, true
		}
	}

	// R3/R7: run the handler harness as the orchestrator's direct child
	// (depth-1), WITHOUT the orchestrator model. The handler's instruction is
	// the consult request rendered to text.
	counts[request.Kind]++
	instruction := renderConsultInstruction(request)
	task := sporecore.NewTask(instruction, generateSessionID(), sporecore.LoopStrategy{
		Kind:          sporecore.StrategyReAct,
		MaxIterations: 16,
	})
	handlerResult := entry.Handler.Run(ctx, sporecore.HarnessRunOptions{Task: task})
	var answer string
	if handlerResult.Kind == sporecore.RunSuccess {
		answer = handlerResult.Output
	} else {
		// A handler that does not cleanly complete still must not stall the
		// worker — feed its failure back as the consult answer so the worker can
		// adapt. (The orchestrator model is never involved.)
		answer = fmt.Sprintf("consult handler did not complete cleanly: kind=%s", handlerResult.Kind)
	}
	next := s.harness.ResumeConsult(ctx, state, sporecore.NewConsultAnswer(answer), nil)
	return next, sporecore.ToolOutput{}, false
}

// renderConsultInstruction renders a ConsultRequest to a handler instruction
// string (issue #114).
func renderConsultInstruction(request sporecore.ConsultRequest) string {
	return fmt.Sprintf(
		"A worker agent is requesting help (kind: %s).\n\nSituation: %s\n\nAttempts so far: %d\n\nQuestion: %s",
		request.Kind, request.Situation, request.Attempts, request.Question,
	)
}

// resolveSessionContext picks the SessionID and optional seeded SessionState
// based on the configured ContextSharing variant.
func (s *SubagentTool) resolveSessionContext() (sporecore.SessionID, *sporecore.SessionState) {
	switch v := s.contextSharing.(type) {
	case Isolated, nil:
		_ = v
		return generateSessionID(), nil
	case SharedSession:
		return v.SessionID, nil
	case SummaryHandoff:
		state := sporecore.SessionState{
			Extras: map[string]any{
				"subagent_handoff_summary": v.Summary,
			},
		}
		return generateSessionID(), &state
	default:
		return generateSessionID(), nil
	}
}

// childStateFromPaused builds a ChildPausedState from a parent PausedState
// produced by the child harness, stamping the parent_tool_call_id.
func childStateFromPaused(p *sporecore.PausedState, parentCallID string) *sporecore.ChildPausedState {
	if p == nil {
		return &sporecore.ChildPausedState{ParentToolCallID: parentCallID}
	}
	return &sporecore.ChildPausedState{
		SessionID:        p.SessionID,
		TaskID:           p.TaskID,
		TurnNumber:       p.TurnNumber,
		SessionState:     p.SessionState,
		PendingToolCalls: p.PendingToolCalls,
		ApprovedResults:  p.ApprovedResults,
		HumanRequest:     p.HumanRequest,
		Task:             p.Task,
		BudgetUsed:       p.BudgetUsed,
		ParentToolCallID: parentCallID,
	}
}

func reasonString(r sporecore.HaltReason) string {
	if b, err := json.Marshal(r); err == nil {
		return string(b)
	}
	return string(r.Kind)
}

var _ sporecore.Tool = (*SubagentTool)(nil)
