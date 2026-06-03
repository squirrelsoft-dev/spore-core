// Package tools holds the two custom tools this example registers: remember
// (the write half) and recall (the read half). Each is a plain
// sporecore.Tool implementation paired with a schema constructor; main wraps
// each pair in a sporecore.StandardTool and hands it to the builder via
// .Tool(...). The harness wires the sandbox and a per-run *ToolContext (the
// storage seam) in automatically.
package tools

import (
	"context"
	"encoding/json"
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// FactPrefix namespaces every key so this example's facts live in their own
// region of the run store and cannot collide with reserved catalogue keys
// (the todo / task_list / memory tools use unprefixed keys).
const FactPrefix = "fact:"

// RememberName is the wire name advertised to the model. It MUST equal the
// Schema().Name and RememberTool.Name().
const RememberName = "remember"

// RememberTool is the WRITE half of the custom-tool pair: it persists a fact
// into the run store. It demonstrates the storage seam — *ToolContext is the
// only path to durable, per-run state. The sandbox parameter is part of the
// Tool.Execute signature but unused here: these tools never touch the
// filesystem, so they ignore it.
type RememberTool struct{}

// Name implements sporecore.Tool. It must match Schema().Name.
func (RememberTool) Name() string { return RememberName }

// IsSubagentTool implements sporecore.Tool. This tool wraps no child harness.
func (RememberTool) IsSubagentTool() bool { return false }

// MayProduceLargeOutput implements sporecore.Tool. The success message is tiny.
func (RememberTool) MayProduceLargeOutput() bool { return false }

// Schema is the registry-side schema. Name MUST equal Name(). It is
// intentionally NOT read-only: remember mutates shared persisted state.
func (RememberTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name: RememberName,
		Description: "Store a fact under a short key so it can be recalled later. " +
			"Use a stable, memorable key (e.g. 'habitat', 'lifespan').",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"key": {"type": "string"},
				"value": {"type": "string"}
			},
			"required": ["key", "value"]
		}`),
		// No ReadOnly: this is a mutation of shared state.
		Annotations: sporecore.ToolAnnotations{},
	}
}

// Execute persists call.Input["value"] under "fact:{key}" in the run store.
// Missing/non-string fields or a store failure are RECOVERABLE tool errors —
// they return a ToolOutput error variant (not a Go error), so the agent can
// adapt rather than halting the run.
func (RememberTool) Execute(ctx context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
	key, ok := stringField(call.Input, "key")
	if !ok {
		return sporecore.NewToolOutputError("remember: missing or non-string 'key'")
	}
	value, ok := stringField(call.Input, "value")
	if !ok {
		return sporecore.NewToolOutputError("remember: missing or non-string 'value'")
	}

	storeKey := FactPrefix + key
	// remember always stores the value as a JSON string; recall renders it back.
	encoded, err := json.Marshal(value)
	if err != nil {
		return sporecore.NewToolOutputError(fmt.Sprintf("remember: could not encode '%s': %v", key, err))
	}
	if err := toolCtx.Put(ctx, storeKey, encoded); err != nil {
		return sporecore.NewToolOutputError(fmt.Sprintf("remember: could not persist '%s': %v", key, err))
	}
	return sporecore.NewToolOutputSuccess(fmt.Sprintf("remembered %s", key))
}

// stringField extracts a required JSON string field from a tool call's raw
// input. It returns ok=false when the input is absent/malformed, the field is
// missing, or the field is present but not a JSON string — every "missing or
// non-string" case the spec treats as a recoverable argument error.
func stringField(input json.RawMessage, name string) (string, bool) {
	if len(input) == 0 {
		return "", false
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(input, &fields); err != nil {
		return "", false
	}
	raw, present := fields[name]
	if !present {
		return "", false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", false
	}
	return s, true
}

// Compile-time check that RememberTool satisfies the Tool interface.
var _ sporecore.Tool = RememberTool{}
