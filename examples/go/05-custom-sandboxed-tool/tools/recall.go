package tools

import (
	"context"
	"encoding/json"
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	coretools "github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
)

// RecallName is the wire name advertised to the model.
const RecallName = "recall"

// recallInput is the typed, validated input for recall.
type recallInput struct {
	// Key is the key a fact was previously stored under with remember.
	Key string `json:"key"`
}

// recallSchema is the explicit JSON schema advertised to the model.
var recallSchema = json.RawMessage(`{
	"type": "object",
	"properties": {
		"key": {"type": "string"}
	},
	"required": ["key"]
}`)

// RecallTool builds the READ half of the custom-tool pair with DefineTool: it
// reads a fact back out of the run store. Unlike RememberTool it is annotated
// ReadOnly + Idempotent — it only reads shared state, so the harness may
// dispatch it concurrently with other read-only tools.
//
// A missing key is a RECOVERABLE error ("no fact stored under '{key}'"): the
// agent can adapt (try another key, or remember the fact first) rather than
// halting the run. Missing/non-string args are likewise recoverable, caught by
// DefineTool as an "invalid parameters" error before Execute runs.
func RecallTool() sporecore.StandardTool {
	return coretools.DefineTool(
		RecallName,
		"Recall a fact previously stored with `remember`, by its key.",
		sporecore.ToolAnnotations{ReadOnly: true, Idempotent: true},
		recallSchema,
		func(ctx context.Context, in recallInput, _ sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
			storeKey := FactPrefix + in.Key
			value, found, err := toolCtx.Get(ctx, storeKey)
			if err != nil {
				return sporecore.NewToolOutputError(fmt.Sprintf("recall: could not read '%s': %v", in.Key, err))
			}
			if !found {
				return sporecore.NewToolOutputError(fmt.Sprintf("no fact stored under '%s'", in.Key))
			}
			return sporecore.NewToolOutputSuccess(valueToString(value))
		},
	)
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
