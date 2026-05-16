// ToolRegistry — maintains available tools and dispatches tool calls.
//
// Implements issue #4. The registry holds the catalog of Tool
// implementations, validates their JSON schemas at registration time, and
// dispatches ToolCalls coming in from the agent — passing every tool a
// SandboxProvider so that no tool ever touches the environment directly.
//
// What this component does:
//   - Register tools with their schemas (validated up-front)
//   - Manage named ToolSet groupings keyed by TaskPhase
//   - Return the active schemas for a given phase (sorted by name)
//   - Dispatch a single call (sandbox-aware) or many calls (concurrent
//     where ToolAnnotations permit)
//   - Expose HasSubagentTools so SubagentTool can enforce depth-1
//
// Rules enforced here:
//
//  1. Tools are always dispatched via the registry — never directly.
//  2. Schemas validated at registration (name nonempty, parameters is
//     a JSON object with top-level "type").
//  3. Duplicate tool names → RegistrationError DuplicateName.
//  4. read_only && destructive → RegistrationError ConflictingAnnotations.
//  5. Tool name must match schema name.
//  6. An unregistered tool call → DispatchError UnregisteredTool.
//  7. Missing required field in input → DispatchError SchemaValidationFailed.
//  8. Sandbox violation → DispatchError SandboxViolation.
//  9. ActiveSchemas(phase) filters by ToolSet; falls back to all if no
//     set matches; always sorted by name (cache stability).
// 10. DispatchAll: read_only calls run concurrently, destructive /
//     open_world run sequentially. Output order matches input order.
// 11. HasSubagentTools returns true if any registered tool reports
//     IsSubagentTool() == true.
//
// Cross-language note: this struct mirrors rust/crates/spore-core/src/
// tool_registry.rs. The shared fixture
// fixtures/tool_registry/dispatch_scenarios.json is exercised in
// tool_registry_fixture_replay_test.go.

package sporecore

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
)

// ============================================================================
// ToolAnnotations & RegistryToolSchema
// ============================================================================

// ToolAnnotations are the behavioural flags attached to a registered tool.
// They drive the DispatchAll concurrency split and the auto-derived RiskLevel
// used by PermissionMiddleware (#11).
type ToolAnnotations struct {
	ReadOnly    bool `json:"read_only,omitempty"`
	Destructive bool `json:"destructive,omitempty"`
	Idempotent  bool `json:"idempotent,omitempty"`
	OpenWorld   bool `json:"open_world,omitempty"`
}

// RegistryToolSchema is the canonical, registry-side schema for a tool.
// Distinct from ToolSchema in model.go (which is the slimmer subset shipped
// to the LLM): this one carries the parameter schema and ToolAnnotations.
//
// Rust calls this `tool_registry::ToolSchema`; the Go variant is renamed to
// avoid a name collision with the model-side ToolSchema in the same package.
type RegistryToolSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	Annotations ToolAnnotations `json:"annotations"`
}

// ToModelSchema projects to the slimmer ToolSchema in model.go.
func (s RegistryToolSchema) ToModelSchema() ToolSchema {
	return ToolSchema{
		Name:        s.Name,
		Description: s.Description,
		InputSchema: s.Parameters,
	}
}

// ============================================================================
// TaskPhase & ToolSet
// ============================================================================

// TaskPhase tags ToolSets so ContextManager can swap the active tool list as
// the task progresses.
type TaskPhase string

const (
	PhaseInitialization TaskPhase = "initialization"
	PhasePlanning       TaskPhase = "planning"
	PhaseExecution      TaskPhase = "execution"
	PhaseVerification   TaskPhase = "verification"
	PhaseCleanup        TaskPhase = "cleanup"
)

// ToolSet is a named grouping of tools, optionally keyed by phase.
type ToolSet struct {
	Name  string     `json:"name"`
	Tools []string   `json:"tools"`
	Phase *TaskPhase `json:"phase,omitempty"`
}

// ============================================================================
// Errors
// ============================================================================

// RegistrationErrorKind discriminates RegistrationError variants.
type RegistrationErrorKind string

const (
	RegErrInvalidSchema          RegistrationErrorKind = "InvalidSchema"
	RegErrDuplicateName          RegistrationErrorKind = "DuplicateName"
	RegErrConflictingAnnotations RegistrationErrorKind = "ConflictingAnnotations"
)

// RegistrationError is the typed error returned by Register / RegisterSet.
type RegistrationError struct {
	Kind   RegistrationErrorKind `json:"kind"`
	Tool   string                `json:"tool"`
	Reason string                `json:"reason,omitempty"`
}

// Error implements error.
func (e *RegistrationError) Error() string {
	switch e.Kind {
	case RegErrInvalidSchema:
		return fmt.Sprintf("invalid schema for tool %q: %s", e.Tool, e.Reason)
	case RegErrDuplicateName:
		return fmt.Sprintf("tool %q already registered", e.Tool)
	case RegErrConflictingAnnotations:
		return fmt.Sprintf("conflicting annotations for tool %q: %s", e.Tool, e.Reason)
	default:
		return fmt.Sprintf("registration error: %s", e.Kind)
	}
}

// MarshalJSON serialises to match the Rust enum tag layout.
func (e RegistrationError) MarshalJSON() ([]byte, error) {
	switch e.Kind {
	case RegErrInvalidSchema, RegErrConflictingAnnotations:
		return json.Marshal(struct {
			Kind   RegistrationErrorKind `json:"kind"`
			Tool   string                `json:"tool"`
			Reason string                `json:"reason"`
		}{e.Kind, e.Tool, e.Reason})
	case RegErrDuplicateName:
		return json.Marshal(struct {
			Kind RegistrationErrorKind `json:"kind"`
			Tool string                `json:"tool"`
		}{e.Kind, e.Tool})
	default:
		return nil, fmt.Errorf("RegistrationError: unknown kind %q", e.Kind)
	}
}

// UnmarshalJSON decodes the tagged form.
func (e *RegistrationError) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind   RegistrationErrorKind `json:"kind"`
		Tool   string                `json:"tool"`
		Reason string                `json:"reason"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	e.Kind = probe.Kind
	e.Tool = probe.Tool
	e.Reason = probe.Reason
	return nil
}

// DispatchErrorKind discriminates DispatchError variants.
type DispatchErrorKind string

const (
	DispatchErrUnregisteredTool       DispatchErrorKind = "UnregisteredTool"
	DispatchErrSchemaValidationFailed DispatchErrorKind = "SchemaValidationFailed"
	DispatchErrSandboxViolation       DispatchErrorKind = "SandboxViolation"
	DispatchErrToolExecutionFailed    DispatchErrorKind = "ToolExecutionFailed"
)

// DispatchError is the typed error returned by Dispatch / DispatchAll.
type DispatchError struct {
	Kind      DispatchErrorKind `json:"kind"`
	Name      string            `json:"name,omitempty"`
	Tool      string            `json:"tool,omitempty"`
	Reason    string            `json:"reason,omitempty"`
	Violation *SandboxViolation `json:"violation,omitempty"`
	ErrorMsg  string            `json:"error,omitempty"`
}

// Error implements error.
func (e *DispatchError) Error() string {
	switch e.Kind {
	case DispatchErrUnregisteredTool:
		return fmt.Sprintf("unregistered tool: %s", e.Name)
	case DispatchErrSchemaValidationFailed:
		return fmt.Sprintf("schema validation failed for %s: %s", e.Tool, e.Reason)
	case DispatchErrSandboxViolation:
		if e.Violation != nil {
			return "sandbox violation: " + e.Violation.Error()
		}
		return "sandbox violation"
	case DispatchErrToolExecutionFailed:
		return fmt.Sprintf("tool %s failed: %s", e.Tool, e.ErrorMsg)
	default:
		return fmt.Sprintf("dispatch error: %s", e.Kind)
	}
}

// MarshalJSON serialises to match the Rust enum tag layout.
func (e DispatchError) MarshalJSON() ([]byte, error) {
	switch e.Kind {
	case DispatchErrUnregisteredTool:
		return json.Marshal(struct {
			Kind DispatchErrorKind `json:"kind"`
			Name string            `json:"name"`
		}{e.Kind, e.Name})
	case DispatchErrSchemaValidationFailed:
		return json.Marshal(struct {
			Kind   DispatchErrorKind `json:"kind"`
			Tool   string            `json:"tool"`
			Reason string            `json:"reason"`
		}{e.Kind, e.Tool, e.Reason})
	case DispatchErrSandboxViolation:
		return json.Marshal(struct {
			Kind      DispatchErrorKind `json:"kind"`
			Violation *SandboxViolation `json:"violation"`
		}{e.Kind, e.Violation})
	case DispatchErrToolExecutionFailed:
		return json.Marshal(struct {
			Kind  DispatchErrorKind `json:"kind"`
			Tool  string            `json:"tool"`
			Error string            `json:"error"`
		}{e.Kind, e.Tool, e.ErrorMsg})
	default:
		return nil, fmt.Errorf("DispatchError: unknown kind %q", e.Kind)
	}
}

// UnmarshalJSON decodes the tagged form.
func (e *DispatchError) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind      DispatchErrorKind `json:"kind"`
		Name      string            `json:"name"`
		Tool      string            `json:"tool"`
		Reason    string            `json:"reason"`
		Violation *SandboxViolation `json:"violation"`
		ErrorMsg  string            `json:"error"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	e.Kind = probe.Kind
	e.Name = probe.Name
	e.Tool = probe.Tool
	e.Reason = probe.Reason
	e.Violation = probe.Violation
	e.ErrorMsg = probe.ErrorMsg
	return nil
}

// ============================================================================
// Tool interface
// ============================================================================

// Tool is a single tool implementation. Tools are stateless and receive a
// SandboxProvider on every dispatch.
type Tool interface {
	// Name must match the registered RegistryToolSchema.Name.
	Name() string

	// IsSubagentTool returns true if this tool wraps a child Harness.
	// Used by ToolRegistry.HasSubagentTools to enforce the depth-1 rule
	// at construction time. Default false for non-subagent tools.
	IsSubagentTool() bool

	// MayProduceLargeOutput reports whether this tool can return output
	// large enough to warrant routing through SandboxProvider.HandleLargeOutput.
	MayProduceLargeOutput() bool

	// Execute runs the tool with validated input. The SandboxProvider is
	// the only path to the environment.
	Execute(ctx context.Context, call ToolCall, sandbox SandboxProvider) ToolOutput
}

// ============================================================================
// ToolRegistry interface
// ============================================================================

// ToolRegistry is the canonical issue-#4 interface. Implementations maintain
// the catalog and dispatch incoming tool calls.
type ToolRegistry interface {
	// Register a tool. The schema is validated at registration time.
	Register(tool Tool, schema RegistryToolSchema) error

	// RegisterSet registers a named ToolSet grouping.
	RegisterSet(set ToolSet) error

	// ActiveSchemas returns the schemas active in the given phase, sorted
	// by name. A nil phase returns every registered schema.
	ActiveSchemas(phase *TaskPhase) []RegistryToolSchema

	// Dispatch one tool call. Returns the HarnessToolResult on success
	// (its Output is the ToolOutput variant) or a *DispatchError.
	Dispatch(ctx context.Context, call ToolCall, sandbox SandboxProvider) (HarnessToolResult, error)

	// DispatchAll dispatches many calls. Read-only calls run concurrently,
	// destructive / open_world run sequentially. Output order matches the
	// input order. Each slot carries either a HarnessToolResult or a
	// *DispatchError via DispatchOutcome.
	DispatchAll(ctx context.Context, calls []ToolCall, sandbox SandboxProvider) []DispatchOutcome

	// HasSubagentTools reports whether any registered tool is a subagent
	// tool. SubagentTool construction uses this to enforce the depth-1
	// rule fail-fast.
	HasSubagentTools() bool

	// IsAlwaysHalt reports whether the named tool is annotated as a
	// Layer-1 always-halt tool. Kept for the harness loop's pre-dispatch
	// short circuit.
	IsAlwaysHalt(toolName string) bool
}

// DispatchOutcome is the per-slot result of DispatchAll. Exactly one of
// Result or Err is populated. (Go has no native Result type; this captures
// the same shape as Rust's Vec<Result<ToolResult, DispatchError>>.)
type DispatchOutcome struct {
	Result HarnessToolResult
	Err    *DispatchError
}

// ============================================================================
// StandardToolRegistry — canonical implementation
// ============================================================================

type registered struct {
	tool   Tool
	schema RegistryToolSchema
}

// StandardToolRegistry is the default in-memory registry. Concurrency-safe:
// register / lookup go through a RWMutex. The lock is held only briefly;
// the tool itself executes lock-free.
type StandardToolRegistry struct {
	mu    sync.RWMutex
	tools map[string]*registered
	sets  []ToolSet
}

// NewStandardToolRegistry constructs a StandardToolRegistry.
func NewStandardToolRegistry() *StandardToolRegistry {
	return &StandardToolRegistry{tools: map[string]*registered{}}
}

// Register validates schema and annotations, then stores the tool.
func (r *StandardToolRegistry) Register(tool Tool, schema RegistryToolSchema) error {
	if tool.Name() != schema.Name {
		return &RegistrationError{
			Kind: RegErrInvalidSchema,
			Tool: schema.Name,
			Reason: fmt.Sprintf("tool name %q does not match schema name %q",
				tool.Name(), schema.Name),
		}
	}
	if err := validateSchema(schema); err != nil {
		return err
	}
	if err := validateAnnotations(schema); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[schema.Name]; exists {
		return &RegistrationError{Kind: RegErrDuplicateName, Tool: schema.Name}
	}
	r.tools[schema.Name] = &registered{tool: tool, schema: schema}
	return nil
}

// RegisterSet stores a named ToolSet.
func (r *StandardToolRegistry) RegisterSet(set ToolSet) error {
	if set.Name == "" {
		return &RegistrationError{
			Kind:   RegErrInvalidSchema,
			Tool:   set.Name,
			Reason: "tool set name must not be empty",
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, s := range r.sets {
		if s.Name == set.Name {
			return &RegistrationError{Kind: RegErrDuplicateName, Tool: set.Name}
		}
	}
	r.sets = append(r.sets, set)
	return nil
}

// ActiveSchemas returns sorted schemas filtered by phase.
func (r *StandardToolRegistry) ActiveSchemas(phase *TaskPhase) []RegistryToolSchema {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var out []RegistryToolSchema
	if phase == nil {
		out = make([]RegistryToolSchema, 0, len(r.tools))
		for _, reg := range r.tools {
			out = append(out, reg.schema)
		}
	} else {
		// Union of matching ToolSets (matching phase OR no phase = always-active).
		var matching []ToolSet
		for _, s := range r.sets {
			if s.Phase == nil || *s.Phase == *phase {
				matching = append(matching, s)
			}
		}
		if len(matching) == 0 {
			// Fall back to the full catalog — zero sets must not silently mask
			// every tool.
			out = make([]RegistryToolSchema, 0, len(r.tools))
			for _, reg := range r.tools {
				out = append(out, reg.schema)
			}
		} else {
			seen := map[string]struct{}{}
			for _, s := range matching {
				for _, name := range s.Tools {
					if _, dup := seen[name]; dup {
						continue
					}
					seen[name] = struct{}{}
					if reg, ok := r.tools[name]; ok {
						out = append(out, reg.schema)
					}
				}
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Dispatch runs a single tool call with sandbox validation and per-call
// schema validation.
func (r *StandardToolRegistry) Dispatch(
	ctx context.Context,
	call ToolCall,
	sandbox SandboxProvider,
) (HarnessToolResult, error) {
	r.mu.RLock()
	reg, ok := r.tools[call.Name]
	r.mu.RUnlock()
	if !ok {
		return HarnessToolResult{}, &DispatchError{
			Kind: DispatchErrUnregisteredTool,
			Name: call.Name,
		}
	}

	// Sandbox validation. Layer-1 path-escape / network-violation surface
	// as a SandboxViolation DispatchError; the harness routes from there.
	if sandbox != nil {
		if v := sandbox.Validate(ctx, call); v != nil {
			vv := *v
			return HarnessToolResult{}, &DispatchError{
				Kind:      DispatchErrSandboxViolation,
				Violation: &vv,
			}
		}
	}

	if err := validateInput(reg.schema, call); err != nil {
		return HarnessToolResult{}, err
	}

	output := reg.tool.Execute(ctx, call, sandbox)
	return HarnessToolResult{CallID: call.ID, Output: output}, nil
}

// DispatchAll fans out read-only calls concurrently and runs the rest
// sequentially while preserving input order in the returned slice.
func (r *StandardToolRegistry) DispatchAll(
	ctx context.Context,
	calls []ToolCall,
	sandbox SandboxProvider,
) []DispatchOutcome {
	classify := make([]bool, len(calls)) // true = run concurrently
	r.mu.RLock()
	for i, c := range calls {
		if reg, ok := r.tools[c.Name]; ok {
			a := reg.schema.Annotations
			classify[i] = a.ReadOnly && !a.Destructive && !a.OpenWorld
		} else {
			// Unknown tools run sequentially so the UnregisteredTool error
			// surfaces deterministically alongside any other sequential failure.
			classify[i] = false
		}
	}
	r.mu.RUnlock()

	outcomes := make([]DispatchOutcome, len(calls))

	// Concurrent batch.
	var wg sync.WaitGroup
	for i, c := range classify {
		if !c {
			continue
		}
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			res, err := r.Dispatch(ctx, calls[idx], sandbox)
			if err != nil {
				outcomes[idx].Err = err.(*DispatchError)
			} else {
				outcomes[idx].Result = res
			}
		}(i)
	}
	wg.Wait()

	// Sequential batch.
	for i, c := range classify {
		if c {
			continue
		}
		res, err := r.Dispatch(ctx, calls[i], sandbox)
		if err != nil {
			outcomes[i].Err = err.(*DispatchError)
		} else {
			outcomes[i].Result = res
		}
	}
	return outcomes
}

// HasSubagentTools reports whether any tool flags itself as a subagent.
func (r *StandardToolRegistry) HasSubagentTools() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, reg := range r.tools {
		if reg.tool.IsSubagentTool() {
			return true
		}
	}
	return false
}

// IsAlwaysHalt reports whether the named tool is annotated as Layer-1
// always-halt. The registry treats no tool as always-halt by default; the
// harness can override on a per-tool basis via a wrapping registry.
func (r *StandardToolRegistry) IsAlwaysHalt(string) bool { return false }

// Compile-time interface check.
var _ ToolRegistry = (*StandardToolRegistry)(nil)

// ============================================================================
// Validation helpers
// ============================================================================

func validateSchema(schema RegistryToolSchema) error {
	if schema.Name == "" {
		return &RegistrationError{
			Kind:   RegErrInvalidSchema,
			Tool:   schema.Name,
			Reason: "name must not be empty",
		}
	}
	if len(schema.Parameters) == 0 {
		return &RegistrationError{
			Kind:   RegErrInvalidSchema,
			Tool:   schema.Name,
			Reason: "parameters must be a JSON object",
		}
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(schema.Parameters, &obj); err != nil {
		return &RegistrationError{
			Kind:   RegErrInvalidSchema,
			Tool:   schema.Name,
			Reason: "parameters must be a JSON object",
		}
	}
	if _, ok := obj["type"]; !ok {
		return &RegistrationError{
			Kind:   RegErrInvalidSchema,
			Tool:   schema.Name,
			Reason: "parameters must declare a top-level `type`",
		}
	}
	return nil
}

func validateAnnotations(schema RegistryToolSchema) error {
	a := schema.Annotations
	if a.ReadOnly && a.Destructive {
		return &RegistrationError{
			Kind:   RegErrConflictingAnnotations,
			Tool:   schema.Name,
			Reason: "read_only and destructive are mutually exclusive",
		}
	}
	return nil
}

// validateInput performs the best-effort per-call schema check: any
// `required` fields declared on the parameter schema must be present in the
// call's input object. Deeper JSON Schema validation can be plugged in later.
func validateInput(schema RegistryToolSchema, call ToolCall) error {
	var input map[string]json.RawMessage
	if len(call.Input) == 0 {
		return &DispatchError{
			Kind:   DispatchErrSchemaValidationFailed,
			Tool:   schema.Name,
			Reason: "input must be a JSON object",
		}
	}
	if err := json.Unmarshal(call.Input, &input); err != nil {
		return &DispatchError{
			Kind:   DispatchErrSchemaValidationFailed,
			Tool:   schema.Name,
			Reason: "input must be a JSON object",
		}
	}
	var params map[string]json.RawMessage
	if err := json.Unmarshal(schema.Parameters, &params); err != nil {
		// Schema was validated at registration; treat as no-op here.
		return nil
	}
	requiredRaw, ok := params["required"]
	if !ok {
		return nil
	}
	var required []string
	if err := json.Unmarshal(requiredRaw, &required); err != nil {
		return nil
	}
	for _, name := range required {
		if _, present := input[name]; !present {
			return &DispatchError{
				Kind:   DispatchErrSchemaValidationFailed,
				Tool:   schema.Name,
				Reason: fmt.Sprintf("missing required field `%s`", name),
			}
		}
	}
	return nil
}
