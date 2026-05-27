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
	"context"
	"encoding/json"
	"fmt"
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

// LoopStrategyKind discriminates LoopStrategy variants.
type LoopStrategyKind string

const (
	StrategyReAct         LoopStrategyKind = "re_act"
	StrategyPlanExecute   LoopStrategyKind = "plan_execute"
	StrategyRalph         LoopStrategyKind = "ralph"
	StrategySelfVerifying LoopStrategyKind = "self_verifying"
	StrategyHillClimbing  LoopStrategyKind = "hill_climbing"
)

// ModelConfig is a lightweight placeholder for an alternate planner model.
type ModelConfig struct {
	Provider string `json:"provider"`
	ModelID  string `json:"model_id"`
}

// LoopStrategy is a tagged-union. Only ReAct is fully executable in
// StandardHarness; the other variants return HaltReason
// StrategyNotYetImplemented per the Rust reference.
type LoopStrategy struct {
	Kind LoopStrategyKind `json:"kind"`
	// ReAct
	MaxIterations uint32 `json:"-"`
	// PlanExecute
	PlanModel *ModelConfig `json:"-"`
	// HillClimbing
	Direction             OptimizationDirection `json:"-"`
	MaxStagnation         *uint32               `json:"-"`
	RevertOnNoImprovement bool                  `json:"-"`
	MinImprovementDelta   *float64              `json:"-"`
}

// MarshalJSON serialises LoopStrategy as a flat tagged object.
func (s LoopStrategy) MarshalJSON() ([]byte, error) {
	switch s.Kind {
	case StrategyReAct:
		return json.Marshal(struct {
			Kind          LoopStrategyKind `json:"kind"`
			MaxIterations uint32           `json:"max_iterations"`
		}{s.Kind, s.MaxIterations})
	case StrategyPlanExecute:
		return json.Marshal(struct {
			Kind      LoopStrategyKind `json:"kind"`
			PlanModel *ModelConfig     `json:"plan_model"`
		}{s.Kind, s.PlanModel})
	case StrategyRalph, StrategySelfVerifying:
		return json.Marshal(struct {
			Kind LoopStrategyKind `json:"kind"`
		}{s.Kind})
	case StrategyHillClimbing:
		return json.Marshal(struct {
			Kind                  LoopStrategyKind      `json:"kind"`
			Direction             OptimizationDirection `json:"direction"`
			MaxStagnation         *uint32               `json:"max_stagnation"`
			RevertOnNoImprovement bool                  `json:"revert_on_no_improvement"`
			MinImprovementDelta   *float64              `json:"min_improvement_delta"`
		}{s.Kind, s.Direction, s.MaxStagnation, s.RevertOnNoImprovement, s.MinImprovementDelta})
	default:
		return nil, fmt.Errorf("LoopStrategy: unknown kind %q", s.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (s *LoopStrategy) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind                  LoopStrategyKind      `json:"kind"`
		MaxIterations         uint32                `json:"max_iterations"`
		PlanModel             *ModelConfig          `json:"plan_model"`
		Direction             OptimizationDirection `json:"direction"`
		MaxStagnation         *uint32               `json:"max_stagnation"`
		RevertOnNoImprovement bool                  `json:"revert_on_no_improvement"`
		MinImprovementDelta   *float64              `json:"min_improvement_delta"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	s.Kind = probe.Kind
	s.MaxIterations = probe.MaxIterations
	s.PlanModel = probe.PlanModel
	s.Direction = probe.Direction
	s.MaxStagnation = probe.MaxStagnation
	s.RevertOnNoImprovement = probe.RevertOnNoImprovement
	s.MinImprovementDelta = probe.MinImprovementDelta
	return nil
}

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

// HarnessStreamEventKind discriminates harness-level stream events.
type HarnessStreamEventKind string

const (
	HarnessStreamTurnStart     HarnessStreamEventKind = "turn_start"
	HarnessStreamTurnEnd       HarnessStreamEventKind = "turn_end"
	HarnessStreamToolCall      HarnessStreamEventKind = "tool_call"
	HarnessStreamToolResult    HarnessStreamEventKind = "tool_result"
	HarnessStreamFinalResponse HarnessStreamEventKind = "final_response"
	HarnessStreamBudgetWarning HarnessStreamEventKind = "budget_warning"
)

// HarnessStreamEvent is one event emitted while the loop runs.
type HarnessStreamEvent struct {
	Kind HarnessStreamEventKind `json:"kind"`
	// turn_start, turn_end
	Turn uint32 `json:"-"`
	// tool_call, tool_result
	CallID string `json:"-"`
	// tool_call
	Name string `json:"-"`
	// tool_result
	IsError bool `json:"-"`
	// final_response
	Content string `json:"-"`
	// budget_warning
	LimitType BudgetLimitType `json:"-"`
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
		return json.Marshal(struct {
			Kind   HarnessStreamEventKind `json:"kind"`
			CallID string                 `json:"call_id"`
			Name   string                 `json:"name"`
		}{e.Kind, e.CallID, e.Name})
	case HarnessStreamToolResult:
		return json.Marshal(struct {
			Kind    HarnessStreamEventKind `json:"kind"`
			CallID  string                 `json:"call_id"`
			IsError bool                   `json:"is_error"`
		}{e.Kind, e.CallID, e.IsError})
	case HarnessStreamFinalResponse:
		return json.Marshal(struct {
			Kind    HarnessStreamEventKind `json:"kind"`
			Content string                 `json:"content"`
		}{e.Kind, e.Content})
	case HarnessStreamBudgetWarning:
		return json.Marshal(struct {
			Kind      HarnessStreamEventKind `json:"kind"`
			LimitType BudgetLimitType        `json:"limit_type"`
		}{e.Kind, e.LimitType})
	default:
		return nil, fmt.Errorf("HarnessStreamEvent: unknown kind %q", e.Kind)
	}
}

// StreamSink consumes harness stream events. May be nil.
type StreamSink func(HarnessStreamEvent)

// ============================================================================
// Forward-declared sibling types (full surfaces in their owning issues)
// ============================================================================

// ToolOutputKind discriminates ToolOutput variants.
type ToolOutputKind string

const (
	ToolOutputSuccess         ToolOutputKind = "success"
	ToolOutputError           ToolOutputKind = "error"
	ToolOutputWaitingForHuman ToolOutputKind = "waiting_for_human"
)

// ToolOutput is the result of dispatching one tool call. Full shape lives
// in issue #4 (ToolRegistry) / #5 (Tool). The variants below cover what
// the harness loop needs to route.
type ToolOutput struct {
	Kind ToolOutputKind `json:"kind"`
	// success
	Content   string `json:"-"`
	Truncated bool   `json:"-"`
	// error
	Message     string `json:"-"`
	Recoverable bool   `json:"-"`
	// waiting_for_human
	ChildState *ChildPausedState `json:"-"`
	Request    *HumanRequest     `json:"-"`
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
	return nil
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

// IsolationMode is a sealed interface with concrete impls IsolationNone,
// IsolationWorkspaceScoped, IsolationBubblewrap, IsolationDocker. The
// unexported sealedIsolationMode() method seals the type so external
// implementations cannot satisfy it.
type IsolationMode interface {
	sealedIsolationMode()
	Kind() string
}

type IsolationNone struct{}
type IsolationWorkspaceScoped struct{}
type IsolationBubblewrap struct {
	Profile BwrapProfile `json:"profile"`
}
type IsolationDocker struct {
	Image   string        `json:"image"`
	Network NetworkPolicy `json:"network"`
}

func (IsolationNone) sealedIsolationMode()            {}
func (IsolationWorkspaceScoped) sealedIsolationMode() {}
func (IsolationBubblewrap) sealedIsolationMode()      {}
func (IsolationDocker) sealedIsolationMode()          {}

func (IsolationNone) Kind() string            { return "none" }
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
type HookPoint string

const (
	HookBeforeTurn       HookPoint = "before_turn"
	HookBeforeTool       HookPoint = "before_tool"
	HookAfterTool        HookPoint = "after_tool"
	HookBeforeCompletion HookPoint = "before_completion"
)

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
	Extras   map[string]any `json:"extras,omitempty"`
}

// MarshalJSON ensures Messages serialises as [] rather than null.
func (s SessionState) MarshalJSON() ([]byte, error) {
	type alias SessionState
	a := alias(s)
	if a.Messages == nil {
		a.Messages = []Message{}
	}
	return json.Marshal(a)
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

// ContextManager (#7) assembles per-turn context and appends results.
type ContextManager interface {
	Assemble(ctx context.Context, session *SessionState, task *Task) Context
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
type MiddlewareDecisionKind string

const (
	MiddlewareContinue                 MiddlewareDecisionKind = "continue"
	MiddlewareContinueWithModification MiddlewareDecisionKind = "continue_with_modification"
	MiddlewareHalt                     MiddlewareDecisionKind = "halt"
	MiddlewareSurfaceToHuman           MiddlewareDecisionKind = "surface_to_human"
)

// MiddlewareDecision is the output of MiddlewareChain.Fire.
type MiddlewareDecision struct {
	Kind    MiddlewareDecisionKind `json:"kind"`
	Calls   []ToolCall             `json:"-"`
	Reason  string                 `json:"-"`
	Request *HumanRequest          `json:"-"`
}

// MiddlewareChain (#11) fires lifecycle hooks. Stub surface.
type MiddlewareChain interface {
	Fire(ctx context.Context, hook HookPoint, session *SessionState) MiddlewareDecision
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
	)

	// SetSessionOutcome records the terminal outcome (success / failure).
	SetSessionOutcome(sessionID SessionID, success bool, failureReason string)

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

// HumanRequestKind discriminates HumanRequest variants.
type HumanRequestKind string

const (
	HumanReqToolApproval  HumanRequestKind = "tool_approval"
	HumanReqClarification HumanRequestKind = "clarification"
	HumanReqReview        HumanRequestKind = "review"
)

// HumanRequest is the question surfaced to the human-in-the-loop.
type HumanRequest struct {
	Kind HumanRequestKind `json:"kind"`
	// tool_approval
	Calls     []ToolCall `json:"-"`
	RiskLevel RiskLevel  `json:"-"`
	// clarification
	Question string `json:"-"`
	// review
	Content string `json:"-"`
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
		return json.Marshal(struct {
			Kind     HumanRequestKind `json:"kind"`
			Question string           `json:"question"`
		}{h.Kind, h.Question})
	case HumanReqReview:
		return json.Marshal(struct {
			Kind    HumanRequestKind `json:"kind"`
			Content string           `json:"content"`
		}{h.Kind, h.Content})
	default:
		return nil, fmt.Errorf("HumanRequest: unknown kind %q", h.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (h *HumanRequest) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind      HumanRequestKind `json:"kind"`
		Calls     []ToolCall       `json:"calls"`
		RiskLevel RiskLevel        `json:"risk_level"`
		Question  string           `json:"question"`
		Content   string           `json:"content"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	h.Kind = probe.Kind
	h.Calls = probe.Calls
	h.RiskLevel = probe.RiskLevel
	h.Question = probe.Question
	h.Content = probe.Content
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
)

// HumanResponse is the human's reply to a HumanRequest.
type HumanResponse struct {
	Kind     HumanResponseKind `json:"kind"`
	Calls    []ToolCall        `json:"-"`
	Reason   string            `json:"-"`
	Text     string            `json:"-"`
	Feedback string            `json:"-"`
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
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	h.Kind = probe.Kind
	h.Calls = probe.Calls
	h.Reason = probe.Reason
	h.Text = probe.Text
	h.Feedback = probe.Feedback
	return nil
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
	HumanRequest     HumanRequest        `json:"human_request"`
	Task             Task                `json:"task"`
	BudgetUsed       BudgetSnapshot      `json:"budget_used"`
	ChildState       *ChildPausedState   `json:"child_state,omitempty"`
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

// ChildPausedState is the paused state for a subagent. Deliberately has no
// child_state field — the type system enforces a one-level subagent depth.
type ChildPausedState struct {
	SessionID        SessionID           `json:"session_id"`
	TaskID           TaskID              `json:"task_id"`
	TurnNumber       uint32              `json:"turn_number"`
	SessionState     SessionState        `json:"session_state"`
	PendingToolCalls []ToolCall          `json:"pending_tool_calls"`
	ApprovedResults  []HarnessToolResult `json:"approved_results"`
	HumanRequest     HumanRequest        `json:"human_request"`
	Task             Task                `json:"task"`
	BudgetUsed       BudgetSnapshot      `json:"budget_used"`
	ParentToolCallID string              `json:"parent_tool_call_id"`
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
	HaltBudgetExceeded            HaltReasonKind = "budget_exceeded"
	HaltTerminationPolicyHalt     HaltReasonKind = "termination_policy_halt"
	HaltMiddlewareHalt            HaltReasonKind = "middleware_halt"
	HaltAgentError                HaltReasonKind = "agent_error"
	HaltSandboxViolation          HaltReasonKind = "sandbox_violation"
	HaltUnrecoverableToolError    HaltReasonKind = "unrecoverable_tool_error"
	HaltHumanHalted               HaltReasonKind = "human_halted"
	HaltStagnationLimitReached    HaltReasonKind = "stagnation_limit_reached"
	HaltStrategyNotYetImplemented HaltReasonKind = "strategy_not_yet_implemented"
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
	default:
		return nil, fmt.Errorf("HaltReason: unknown kind %q", h.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (h *HaltReason) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind       HaltReasonKind    `json:"kind"`
		LimitType  BudgetLimitType   `json:"limit_type"`
		Reason     string            `json:"reason"`
		Hook       HookPoint         `json:"hook"`
		Error      json.RawMessage   `json:"error"`
		Violation  *SandboxViolation `json:"violation"`
		Tool       string            `json:"tool"`
		Iterations uint32            `json:"iterations"`
		BestMetric float64           `json:"best_metric"`
		Strategy   string            `json:"strategy"`
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

	switch probe.Kind {
	case HaltAgentError:
		if len(probe.Error) > 0 && string(probe.Error) != "null" {
			ae := &AgentError{}
			if err := ae.UnmarshalJSON(probe.Error); err != nil {
				return err
			}
			h.AgentError = ae
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
	}
	return nil
}

// RunResultKind discriminates RunResult variants.
type RunResultKind string

const (
	RunSuccess         RunResultKind = "success"
	RunFailure         RunResultKind = "failure"
	RunWaitingForHuman RunResultKind = "waiting_for_human"
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
	// waiting_for_human
	State   *PausedState  `json:"-"`
	Request *HumanRequest `json:"-"`
}

// MarshalJSON serialises as a flat tagged object.
func (r RunResult) MarshalJSON() ([]byte, error) {
	switch r.Kind {
	case RunSuccess:
		return json.Marshal(struct {
			Kind      RunResultKind  `json:"kind"`
			Output    string         `json:"output"`
			SessionID SessionID      `json:"session_id"`
			Usage     AggregateUsage `json:"usage"`
			Turns     uint32         `json:"turns"`
		}{r.Kind, r.Output, r.SessionID, r.Usage, r.Turns})
	case RunFailure:
		return json.Marshal(struct {
			Kind      RunResultKind  `json:"kind"`
			Reason    HaltReason     `json:"reason"`
			SessionID SessionID      `json:"session_id"`
			Usage     AggregateUsage `json:"usage"`
			Turns     uint32         `json:"turns"`
		}{r.Kind, r.Reason, r.SessionID, r.Usage, r.Turns})
	case RunWaitingForHuman:
		return json.Marshal(struct {
			Kind    RunResultKind `json:"kind"`
			State   *PausedState  `json:"state"`
			Request *HumanRequest `json:"request"`
		}{r.Kind, r.State, r.Request})
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
		State     *PausedState   `json:"state"`
		Request   *HumanRequest  `json:"request"`
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
	r.State = probe.State
	r.Request = probe.Request
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
}

// ============================================================================
// StandardHarness — the canonical implementation
// ============================================================================

// HarnessConfig is the bag of components injected into StandardHarness.
// Middleware and Observability are optional; the rest are required.
type HarnessConfig struct {
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
}

// StandardHarness is the canonical Harness implementation.
type StandardHarness struct {
	config HarnessConfig
}

// NewStandardHarness constructs a StandardHarness.
func NewStandardHarness(c HarnessConfig) *StandardHarness {
	return &StandardHarness{config: c}
}

func emit(sink StreamSink, event HarnessStreamEvent) {
	if sink != nil {
		sink(event)
	}
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
func (h *StandardHarness) Run(ctx context.Context, options HarnessRunOptions) RunResult {
	task := options.Task
	var session SessionState
	if options.SessionState != nil {
		session = *options.SessionState
	}
	budget := BudgetSnapshot{}

	switch task.LoopStrategy.Kind {
	case StrategyReAct:
		// Seed the task instruction as the initial user message of this run.
		// The compaction adapter intentionally mirrors session.Messages and
		// ignores task on Assemble, so the harness must own delivering the
		// prompt. On a fresh run this turns an otherwise-empty conversation
		// into a real user turn; on multi-turn runs over a carried
		// SessionState each Run call appends its own follow-up instruction.
		// The resume path is intentionally excluded — its conversation already
		// exists, so it must not be re-seeded.
		h.config.ContextManager.AppendUserMessage(ctx, &session, task.Instruction)
		return h.runReAct(ctx, task, task.LoopStrategy.MaxIterations, session, budget, options.OnStream)
	case StrategyPlanExecute:
		return strategyNotImplemented(task, "plan_execute")
	case StrategyRalph:
		return strategyNotImplemented(task, "ralph")
	case StrategySelfVerifying:
		return strategyNotImplemented(task, "self_verifying")
	case StrategyHillClimbing:
		return strategyNotImplemented(task, "hill_climbing")
	default:
		return strategyNotImplemented(task, string(task.LoopStrategy.Kind))
	}
}

func strategyNotImplemented(task Task, strategy string) RunResult {
	return RunResult{
		Kind: RunFailure,
		Reason: HaltReason{
			Kind:     HaltStrategyNotYetImplemented,
			Strategy: strategy,
		},
		SessionID: task.SessionID,
	}
}

// nowRFC3339 returns the current UTC time as an RFC3339 string — the
// timestamp form every span carries (mirrors guideregistry.nowTimestamp and
// the observability backend's lexical timestamps).
func nowRFC3339() string { return time.Now().UTC().Format(time.RFC3339) }

// finalizeObservability records the terminal outcome and flushes the session.
// Called at every terminal runReAct outcome (Success or any Failure) — never
// on a WaitingForHuman pause, which is not terminal. No-op when no observer is
// configured. Mirrors the Rust `finalize_observability` wrapper.
func (h *StandardHarness) finalizeObservability(ctx context.Context, sessionID SessionID, success bool, failureReason string) {
	if h.config.Observability == nil {
		return
	}
	h.config.Observability.SetSessionOutcome(sessionID, success, failureReason)
	h.config.Observability.FlushSession(ctx, sessionID)
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
) RunResult {
	result := h.runReActInner(ctx, task, maxIterations, session, budget, onStream)
	switch result.Kind {
	case RunSuccess:
		h.finalizeObservability(ctx, result.SessionID, true, "")
	case RunFailure:
		h.finalizeObservability(ctx, result.SessionID, false, haltReasonString(result.Reason))
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
	default:
		return string(r.Kind)
	}
}

// runReActInner is the workhorse loop for LoopStrategy ReAct.
func (h *StandardHarness) runReActInner(
	ctx context.Context,
	task Task,
	maxIterations uint32,
	session SessionState,
	budget BudgetSnapshot,
	onStream StreamSink,
) RunResult {
	sessionID := task.SessionID
	startedAt := time.Now()
	usage := AggregateUsage{}

	// Monotonic per-run span counter for tool-call span ids, and the parent
	// span id of the most recent turn (parents this turn's tool-call spans).
	var spanSeq uint64
	var currentTurnSpanID string

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

		// Middleware: BeforeTurn.
		if h.config.Middleware != nil {
			d := h.config.Middleware.Fire(ctx, HookBeforeTurn, &session)
			switch d.Kind {
			case MiddlewareHalt:
				return RunResult{
					Kind: RunFailure,
					Reason: HaltReason{
						Kind:   HaltMiddlewareHalt,
						Hook:   HookBeforeTurn,
						Reason: d.Reason,
					},
					SessionID: sessionID, Usage: usage, Turns: budget.Turns,
				}
			case MiddlewareSurfaceToHuman:
				req := HumanRequest{}
				if d.Request != nil {
					req = *d.Request
				}
				state := &PausedState{
					SessionID: sessionID, TaskID: task.ID, TurnNumber: budget.Turns,
					SessionState: session, PendingToolCalls: nil, ApprovedResults: nil,
					HumanRequest: req, Task: task, BudgetUsed: budget, ChildState: nil,
				}
				return RunResult{Kind: RunWaitingForHuman, State: state, Request: &req}
			}
		}

		// Assemble + invoke agent for one turn.
		c := h.config.ContextManager.Assemble(ctx, &session, &task)
		emit(onStream, HarnessStreamEvent{Kind: HarnessStreamTurnStart, Turn: budget.Turns + 1})
		turnStartedAt := nowRFC3339()
		turnClock := time.Now()
		result := h.config.Agent.Turn(ctx, c)
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

			if h.config.Middleware != nil {
				d := h.config.Middleware.Fire(ctx, HookBeforeCompletion, &session)
				switch d.Kind {
				case MiddlewareHalt:
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookBeforeCompletion,
							Reason: d.Reason,
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				case MiddlewareSurfaceToHuman:
					req := HumanRequest{}
					if d.Request != nil {
						req = *d.Request
					}
					state := &PausedState{
						SessionID: sessionID, TaskID: task.ID, TurnNumber: budget.Turns,
						SessionState: session, PendingToolCalls: nil, ApprovedResults: nil,
						HumanRequest: req, Task: task, BudgetUsed: budget, ChildState: nil,
					}
					return RunResult{Kind: RunWaitingForHuman, State: state, Request: &req}
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
				if h.config.ToolRegistry.IsAlwaysHalt(c.Name) {
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

			// Middleware: BeforeTool.
			if h.config.Middleware != nil {
				d := h.config.Middleware.Fire(ctx, HookBeforeTool, &session)
				switch d.Kind {
				case MiddlewareContinueWithModification:
					calls = d.Calls
				case MiddlewareHalt:
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookBeforeTool,
							Reason: d.Reason,
						},
						SessionID: sessionID, Usage: usage, Turns: budget.Turns,
					}
				case MiddlewareSurfaceToHuman:
					req := HumanRequest{}
					if d.Request != nil {
						req = *d.Request
					}
					state := &PausedState{
						SessionID: sessionID, TaskID: task.ID, TurnNumber: budget.Turns,
						SessionState: session, PendingToolCalls: calls, ApprovedResults: nil,
						HumanRequest: req, Task: task, BudgetUsed: budget, ChildState: nil,
					}
					return RunResult{Kind: RunWaitingForHuman, State: state, Request: &req}
				}
			}

			var approved []HarnessToolResult
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
						Kind: HarnessStreamToolResult, CallID: call.ID, IsError: true,
					})
					h.config.ContextManager.AppendToolResult(ctx, &session, &tr)
					approved = append(approved, tr)
					continue
				}

				emit(onStream, HarnessStreamEvent{
					Kind: HarnessStreamToolCall, CallID: call.ID, Name: call.Name,
				})
				toolStartedAt := nowRFC3339()
				toolClock := time.Now()
				output := dispatchAndUnwrap(ctx, h.config.ToolRegistry, h.config.Sandbox, call)

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
						ApprovedResults: approved, HumanRequest: req, Task: task,
						BudgetUsed: budget, ChildState: output.ChildState,
					}
					_ = toolPause
					return RunResult{Kind: RunWaitingForHuman, State: state, Request: &req}
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
					)
					spanSeq++
				}

				tr := HarnessToolResult{CallID: call.ID, Output: output}
				emit(onStream, HarnessStreamEvent{
					Kind: HarnessStreamToolResult, CallID: call.ID, IsError: isError,
				})
				h.config.ContextManager.AppendToolResult(ctx, &session, &tr)
				approved = append(approved, tr)
			}

			// Middleware: AfterTool.
			if h.config.Middleware != nil {
				d := h.config.Middleware.Fire(ctx, HookAfterTool, &session)
				if d.Kind == MiddlewareHalt {
					return RunResult{
						Kind: RunFailure,
						Reason: HaltReason{
							Kind:   HaltMiddlewareHalt,
							Hook:   HookAfterTool,
							Reason: d.Reason,
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
				h.runCompaction(ctx, &session, sessionID, task.ID, &spanSeq, &usage)
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
		result := h.config.Agent.Turn(ctx, turn.Context)
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
func (h *StandardHarness) Resume(
	ctx context.Context,
	state PausedState,
	response HumanResponse,
	onStream StreamSink,
) RunResult {
	session := state.SessionState
	task := state.Task
	budget := state.BudgetUsed
	pending := state.PendingToolCalls

	// Subagent depth-1: a child state, if present, would be resumed via the
	// caller-installed SubagentTool. Wiring lands with #4/#5. The harness
	// itself just round-trips the field and continues the parent loop.
	_ = state.ChildState

	switch response.Kind {
	case HumanRespHalt:
		return RunResult{
			Kind:      RunFailure,
			Reason:    HaltReason{Kind: HaltHumanHalted},
			SessionID: state.SessionID,
			Turns:     state.TurnNumber,
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
			output := dispatchAndUnwrap(ctx, h.config.ToolRegistry, h.config.Sandbox, call)
			tr := HarnessToolResult{CallID: call.ID, Output: output}
			h.config.ContextManager.AppendToolResult(ctx, &session, &tr)
		}
	case HumanRespAllowWithModification:
		for _, call := range response.Calls {
			output := dispatchAndUnwrap(ctx, h.config.ToolRegistry, h.config.Sandbox, call)
			tr := HarnessToolResult{CallID: call.ID, Output: output}
			h.config.ContextManager.AppendToolResult(ctx, &session, &tr)
		}
	}

	maxIterations := uint32(^uint32(0))
	if task.LoopStrategy.Kind == StrategyReAct {
		maxIterations = task.LoopStrategy.MaxIterations
	}
	return h.runReAct(ctx, task, maxIterations, session, budget, onStream)
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

// Assemble returns a Context built from the current session messages.
func (NoopContextManager) Assemble(_ context.Context, s *SessionState, _ *Task) Context {
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
	mu         sync.Mutex
	outputs    []ToolOutput
	alwaysHalt map[string]struct{}
	CallCount  atomic.Int64
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

// MarkAlwaysHalt flags tool names as always-halt.
func (s *ScriptedToolRegistry) MarkAlwaysHalt(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.alwaysHalt[name] = struct{}{}
}

// Dispatch returns the next queued output (defaulting to Success "ok").
func (s *ScriptedToolRegistry) Dispatch(_ context.Context, call ToolCall, _ SandboxProvider) (HarnessToolResult, error) {
	s.CallCount.Add(1)
	s.mu.Lock()
	defer s.mu.Unlock()
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

// ActiveSchemas returns an empty slice.
func (s *ScriptedToolRegistry) ActiveSchemas(*TaskPhase) []RegistryToolSchema { return nil }

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

// Fire returns the queued decision for hook, or Continue.
func (s *ScriptedMiddleware) Fire(_ context.Context, hook HookPoint, _ *SessionState) MiddlewareDecision {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.decisions) == 0 {
		return MiddlewareDecision{Kind: MiddlewareContinue}
	}
	if s.decisions[0].hook != hook {
		return MiddlewareDecision{Kind: MiddlewareContinue}
	}
	d := s.decisions[0].d
	s.decisions = s.decisions[1:]
	return d
}

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
