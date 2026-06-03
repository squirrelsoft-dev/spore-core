// Package tools holds the two custom tools this example registers: remember
// (the write half) and recall (the read half). Each is built with the
// DefineTool generic helper from the core tools package — the Go analog of
// Rust's `tool!` macro. DefineTool collapses the per-tool boilerplate (empty
// struct + Name/IsSubagentTool/MayProduceLargeOutput/Schema/Execute + the input
// unmarshal) into a single call: you supply a typed input struct, the explicit
// parameter schema, annotations, and a typed exec function, and it hands back a
// ready-to-register sporecore.StandardTool. main registers each via .Tool(...);
// the harness wires the sandbox and a per-run *ToolContext (the storage seam) in
// automatically.
//
// Why explicit schema (not derived)? Rust derives the JSON schema from the
// input type via schemars. Go's DefineTool takes the explicit-schema variant by
// design — the caller passes the parameter schema directly — so the core module
// stays dependency-free (no jsonschema lib). Reflection-based derivation is a
// possible later opt-in.
package tools

import (
	"context"
	"encoding/json"
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	coretools "github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
)

// FactPrefix namespaces every key so this example's facts live in their own
// region of the run store and cannot collide with reserved catalogue keys
// (the todo / task_list / memory tools use unprefixed keys).
const FactPrefix = "fact:"

// RememberName is the wire name advertised to the model.
const RememberName = "remember"

// rememberInput is the typed, validated input for remember. DefineTool
// unmarshals the model's raw arguments into this struct; a missing required
// field or a wrong-typed field becomes a recoverable "invalid parameters" error
// before Execute ever runs.
type rememberInput struct {
	// Key is a short, stable key to file the fact under (e.g. "habitat").
	Key string `json:"key"`
	// Value is the fact to remember.
	Value string `json:"value"`
}

// rememberSchema is the explicit JSON schema advertised to the model. It mirrors
// rememberInput; with the explicit-schema variant the two are kept in sync by
// hand (Rust derives this from the struct via schemars).
var rememberSchema = json.RawMessage(`{
	"type": "object",
	"properties": {
		"key": {"type": "string"},
		"value": {"type": "string"}
	},
	"required": ["key", "value"]
}`)

// RememberTool builds the WRITE half of the custom-tool pair with DefineTool: it
// persists a fact into the run store. It demonstrates the storage seam —
// *ToolContext is the only path to durable, per-run state. The sandbox parameter
// is part of the exec signature but unused here: these tools never touch the
// filesystem.
//
// Annotations are intentionally NOT ReadOnly: remember MUTATES shared persisted
// state (unlike recall, which is read-only + idempotent).
func RememberTool() sporecore.StandardTool {
	return coretools.DefineTool(
		RememberName,
		"Store a fact under a short key so it can be recalled later. "+
			"Use a stable, memorable key (e.g. 'habitat', 'lifespan').",
		sporecore.ToolAnnotations{}, // not ReadOnly — this mutates shared state
		rememberSchema,
		func(ctx context.Context, in rememberInput, _ sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
			storeKey := FactPrefix + in.Key
			// remember always stores the value as a JSON string; recall renders it back.
			encoded, err := json.Marshal(in.Value)
			if err != nil {
				return sporecore.NewToolOutputError(fmt.Sprintf("remember: could not encode '%s': %v", in.Key, err))
			}
			if err := toolCtx.Put(ctx, storeKey, encoded); err != nil {
				return sporecore.NewToolOutputError(fmt.Sprintf("remember: could not persist '%s': %v", in.Key, err))
			}
			return sporecore.NewToolOutputSuccess(fmt.Sprintf("remembered %s", in.Key))
		},
	)
}
