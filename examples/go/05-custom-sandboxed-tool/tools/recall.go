package tools

import (
	"context"
	"encoding/json"
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// RecallName is the wire name advertised to the model. It MUST equal the
// Schema().Name and RecallTool.Name().
const RecallName = "recall"

// RecallTool is the READ half of the custom-tool pair: it reads a fact back
// out of the run store. Unlike RememberTool it is annotated ReadOnly +
// Idempotent — it only reads shared state, so the harness may dispatch it
// concurrently with other read-only tools.
type RecallTool struct{}

// Name implements sporecore.Tool. It must match Schema().Name.
func (RecallTool) Name() string { return RecallName }

// IsSubagentTool implements sporecore.Tool. This tool wraps no child harness.
func (RecallTool) IsSubagentTool() bool { return false }

// MayProduceLargeOutput implements sporecore.Tool. Recalled facts stay short.
func (RecallTool) MayProduceLargeOutput() bool { return false }

// Schema is the registry-side schema. Name MUST equal Name(). A pure read of
// shared state is safe to mark ReadOnly + Idempotent.
func (RecallTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        RecallName,
		Description: "Recall a fact previously stored with `remember`, by its key.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"key": {"type": "string"}
			},
			"required": ["key"]
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true, Idempotent: true},
	}
}

// Execute reads the value stored under "fact:{key}" and returns it as plain
// text. A missing key is a RECOVERABLE error ("no fact stored under '{key}'"):
// the agent can adapt (try another key, or remember the fact first) rather than
// halting the run. Missing/non-string args are likewise recoverable.
func (RecallTool) Execute(ctx context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
	key, ok := stringField(call.Input, "key")
	if !ok {
		return sporecore.NewToolOutputError("recall: missing or non-string 'key'")
	}

	storeKey := FactPrefix + key
	value, found, err := toolCtx.Get(ctx, storeKey)
	if err != nil {
		return sporecore.NewToolOutputError(fmt.Sprintf("recall: could not read '%s': %v", key, err))
	}
	if !found {
		return sporecore.NewToolOutputError(fmt.Sprintf("no fact stored under '%s'", key))
	}
	return sporecore.NewToolOutputSuccess(valueToString(value))
}

// valueToString renders a stored value as plain text. remember always stores a
// JSON string, so decode that; fall back to the raw JSON for anything else.
func valueToString(value json.RawMessage) string {
	var s string
	if err := json.Unmarshal(value, &s); err == nil {
		return s
	}
	return string(value)
}

// Compile-time check that RecallTool satisfies the Tool interface.
var _ sporecore.Tool = RecallTool{}
