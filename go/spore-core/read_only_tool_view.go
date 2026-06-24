package sporecore

import "context"

// ReadonlyEvalToolNames is the read-only allow-list (SC-30) the SelfVerifying
// evaluate phase auto-restricts its reviewer catalogue to when no explicit
// eval_toolset is set. It is exactly the set of names returned by the tools
// catalogue's StandardTools.ReadonlySet() — kept here, in the core package, as a
// plain name list because core CANNOT import the tools package (that would close
// the import cycle the codebase already broke by moving StandardTool into core).
// A drift-guard test in the tools package asserts this slice equals the names of
// StandardTools.ReadonlySet(), so the two can never silently diverge.
var ReadonlyEvalToolNames = []string{
	"read_file",
	"list_dir",
	"grep_files",
	"grep",
	"find_files",
	"web_fetch",
	"web_search",
}

// readonlyEvalAllowSet returns ReadonlyEvalToolNames as a membership set.
func readonlyEvalAllowSet() map[string]struct{} {
	allow := make(map[string]struct{}, len(ReadonlyEvalToolNames))
	for _, n := range ReadonlyEvalToolNames {
		allow[n] = struct{}{}
	}
	return allow
}

// ReadOnlyToolView is a read-only VIEW over an inner harness-loop ToolRegistry
// (SC-30). It advertises (ActiveSchemas) and dispatches ONLY tools whose name is
// in allow — the INTERSECTION of the wrapped catalogue with a read-only
// allow-list (ReadonlyEvalToolNames, the names of StandardTools.ReadonlySet()).
// Used internally for the SelfVerifying evaluate phase so a reviewer cannot reach
// write / exec / side-effecting tools (web/MCP the read-only sandbox does not
// gate) even though the work phase could — WITHOUT the consumer registering a
// scoped read-only toolset. A non-allow-listed dispatch (which the model should
// never request, since it is never advertised) returns a RECOVERABLE
// ToolOutput error and never reaches the inner registry. A consumer that wants a
// different reviewer toolset sets an explicit eval_toolset handle, which bypasses
// this view entirely.
type ReadOnlyToolView struct {
	inner ToolRegistry
	allow map[string]struct{}
}

// NewReadOnlyToolView wraps inner, restricting it to the allow set.
func NewReadOnlyToolView(inner ToolRegistry, allow map[string]struct{}) *ReadOnlyToolView {
	return &ReadOnlyToolView{inner: inner, allow: allow}
}

// Dispatch delegates to the inner registry when the call's tool is allowed;
// otherwise it returns a recoverable ToolOutput error (no Go error) so the
// harness loop appends it as a tool result and lets the reviewer adapt, never
// reaching the inner registry.
func (v *ReadOnlyToolView) Dispatch(
	ctx context.Context,
	call ToolCall,
	sandbox SandboxProvider,
) (HarnessToolResult, error) {
	if _, ok := v.allow[call.Name]; ok {
		return v.inner.Dispatch(ctx, call, sandbox)
	}
	return HarnessToolResult{
		CallID: call.ID,
		Output: NewToolOutputError(
			"tool `" + call.Name + "` is not available in the read-only evaluate phase",
		),
	}, nil
}

// DispatchAll dispatches each call through Dispatch, preserving input order.
// (Unlike the StandardToolRegistry, the view does not split read-only vs
// destructive batches — every allowed tool here is already read-only.)
func (v *ReadOnlyToolView) DispatchAll(
	ctx context.Context,
	calls []ToolCall,
	sandbox SandboxProvider,
) []DispatchOutcome {
	outcomes := make([]DispatchOutcome, len(calls))
	for i, c := range calls {
		res, err := v.Dispatch(ctx, c, sandbox)
		if err != nil {
			if de, ok := err.(*DispatchError); ok {
				outcomes[i].Err = de
			} else {
				outcomes[i].Err = &DispatchError{Kind: DispatchErrToolExecutionFailed, Tool: c.Name, ErrorMsg: err.Error()}
			}
			continue
		}
		outcomes[i].Result = res
	}
	return outcomes
}

// ActiveSchemas returns the inner registry's active schemas filtered to the
// allow set (by Name), so the reviewer advertises only the read-only tools.
func (v *ReadOnlyToolView) ActiveSchemas(phase *TaskPhase) []RegistryToolSchema {
	inner := v.inner.ActiveSchemas(phase)
	filtered := make([]RegistryToolSchema, 0, len(inner))
	for _, s := range inner {
		if _, ok := v.allow[s.Name]; ok {
			filtered = append(filtered, s)
		}
	}
	return filtered
}

// Register delegates to the inner registry. The view is read-only in the sense
// that it FILTERS dispatch/advertising; delegating registration is harmless and
// simplest (the eval phase never registers through the view).
func (v *ReadOnlyToolView) Register(tool Tool, schema RegistryToolSchema) error {
	return v.inner.Register(tool, schema)
}

// RegisterSet delegates to the inner registry (see Register).
func (v *ReadOnlyToolView) RegisterSet(set ToolSet) error {
	return v.inner.RegisterSet(set)
}

// HasSubagentTools delegates to the inner registry.
func (v *ReadOnlyToolView) HasSubagentTools() bool {
	return v.inner.HasSubagentTools()
}

// IsAlwaysHalt delegates to the inner registry.
func (v *ReadOnlyToolView) IsAlwaysHalt(toolName string) bool {
	return v.inner.IsAlwaysHalt(toolName)
}

// Compile-time interface check.
var _ ToolRegistry = (*ReadOnlyToolView)(nil)
