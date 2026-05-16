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
	default:
		return sporecore.ToolOutput{
			Kind:        sporecore.ToolOutputError,
			Message:     fmt.Sprintf("subagent returned unknown run result kind %q", result.Kind),
			Recoverable: false,
		}
	}
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
