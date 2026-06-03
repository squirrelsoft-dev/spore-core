// Ergonomic custom-tool definition: DefineTool collapses the per-tool
// boilerplate of a hand-written sporecore.Tool implementation — the empty
// struct, the Name / IsSubagentTool / MayProduceLargeOutput methods, the
// Schema constructor, the input-JSON unmarshal + recoverable "invalid
// parameters" handling, and the `var _ Tool = ...` assertion — into a single
// generic adapter that returns a ready-to-register StandardTool.
//
// This is the Go analog of Rust's `tool!` macro
// (rust/crates/spore-core/src/macros.rs). The headline Rust feature is that the
// input type is the single source of truth and the JSON schema is *derived*
// from it (via schemars), so the advertised schema and the deserialization can
// never drift.
//
// Go takes the explicit-schema variant by design: the caller passes the
// parameter schema explicitly rather than pulling in a reflection-based
// JSON-schema dependency. This keeps the core module dependency-free (no
// jsonschema lib in go.mod). Reflection-based schema derivation from T is a
// possible later opt-in, but it is intentionally out of scope here.
//
// What DefineTool DOES mirror from the Rust macro:
//
//  1. It synthesizes the trivial Tool methods so the caller writes none of
//     them.
//  2. The caller provides a name, description, annotations, the explicit
//     parameter schema, and an exec function typed over the input T. The
//     adapter unmarshals the model's raw arguments into T; on failure it
//     returns a RECOVERABLE error ToolOutput whose message contains "invalid
//     parameters" (matching the Rust semantics) so a configured
//     ToolCallRepair can coerce the arguments and re-dispatch.
//  3. It returns a registrable StandardTool (Tool implementation + schema),
//     ready for HarnessBuilder.Tool(...).
//
// Example:
//
//	type calcInput struct {
//		Expression string `json:"expression"`
//	}
//	calc := tools.DefineTool(
//		"calculator",
//		"Evaluates a mathematical expression and returns the result",
//		sporecore.ToolAnnotations{},
//		json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
//		func(ctx context.Context, in calcInput, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
//			return sporecore.NewToolOutputSuccess("evaluated: " + in.Expression)
//		},
//	)
//	// calc plugs straight into HarnessBuilder.Tool(calc).

package tools

import (
	"context"
	"encoding/json"
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ToolExecFunc is the typed execution seam DefineTool wraps. It mirrors the
// sporecore.Tool.Execute signature (context, sandbox, *ToolContext) but takes
// the already-unmarshaled, typed input T in place of the raw ToolCall — the
// adapter does the JSON decode for you. Return a ToolOutput exactly as a
// hand-written Tool.Execute would: NewToolOutputSuccess / NewToolOutputError
// (recoverable) / NewToolOutputFatal.
type ToolExecFunc[T any] func(ctx context.Context, input T, sandbox sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput

// DefineTool bundles a typed exec function with its metadata and explicit
// parameter schema into a registrable StandardTool, synthesizing the trivial
// Tool methods so the caller writes none of them.
//
// The model's raw tool arguments are unmarshaled into T before execFn runs. If
// they fail to decode (a missing required field, a wrong-typed field, malformed
// JSON), DefineTool short-circuits with a RECOVERABLE error ToolOutput whose
// message contains "invalid parameters" — so a configured ToolCallRepair gets a
// chance to coerce the arguments and re-dispatch, rather than halting the run.
// This matches Rust's `tool!` macro semantics.
//
// schema is the explicit JSON-Schema object advertised to the model (it must be
// a JSON object with a top-level "type", as the registry validates at
// registration time). annotations carries the behavioural flags (ReadOnly /
// Destructive / Idempotent / OpenWorld) that drive concurrency and risk.
func DefineTool[T any](
	name string,
	description string,
	annotations sporecore.ToolAnnotations,
	schema json.RawMessage,
	execFn ToolExecFunc[T],
) StandardTool {
	impl := &definedTool[T]{name: name, required: requiredFields(schema), execFn: execFn}
	return StandardTool{
		Implementation: impl,
		Schema: sporecore.RegistryToolSchema{
			Name:        name,
			Description: description,
			Parameters:  schema,
			Annotations: annotations,
		},
	}
}

// definedTool is the synthesized sporecore.Tool implementation backing every
// DefineTool result. It is unexported: callers only ever see the StandardTool.
type definedTool[T any] struct {
	name string
	// required is the schema's top-level `required` field names, extracted once
	// at construction. Go's json.Unmarshal silently leaves a missing field at
	// its zero value (unlike Rust's serde, which errors on a missing required
	// field), so the adapter checks presence explicitly to reproduce Rust's
	// "missing required field → recoverable invalid parameters" semantics.
	required []string
	execFn   ToolExecFunc[T]
}

// requiredFields extracts the top-level `required` array from a JSON-Schema
// object. A malformed or absent `required` yields nil (no presence checks),
// matching the registry's own best-effort validateInput.
func requiredFields(schema json.RawMessage) []string {
	var params struct {
		Required []string `json:"required"`
	}
	if err := json.Unmarshal(schema, &params); err != nil {
		return nil
	}
	return params.Required
}

// Name implements sporecore.Tool. It returns the registered name, which must
// match the bundled schema's Name (the registry enforces this).
func (t *definedTool[T]) Name() string { return t.name }

// IsSubagentTool implements sporecore.Tool. A DefineTool tool wraps no child
// harness — subagent tools are built through the dedicated subagent path, not
// this convenience helper.
func (t *definedTool[T]) IsSubagentTool() bool { return false }

// MayProduceLargeOutput implements sporecore.Tool. DefineTool tools are not
// assumed to produce large output; a tool that streams large results should be
// hand-written so it can opt into SandboxProvider.HandleLargeOutput.
func (t *definedTool[T]) MayProduceLargeOutput() bool { return false }

// Execute implements sporecore.Tool. It unmarshals the call's raw input into T
// and dispatches to execFn, returning a recoverable "invalid parameters" error
// ToolOutput if the decode fails.
func (t *definedTool[T]) Execute(
	ctx context.Context,
	call sporecore.ToolCall,
	sandbox sporecore.SandboxProvider,
	toolCtx *sporecore.ToolContext,
) sporecore.ToolOutput {
	// An absent/empty body is treated as `{}` so a tool with no required fields
	// still runs; a tool WITH required fields fails the presence check below.
	raw := call.Input
	if len(raw) == 0 {
		raw = json.RawMessage("{}")
	}

	// Presence check for required fields FIRST. Go's json.Unmarshal silently
	// leaves a missing field at its zero value, so without this a missing
	// required field would slip through as the zero value rather than the
	// recoverable "invalid parameters" error Rust's serde produces. Decoding
	// into a generic object also surfaces malformed JSON as a recoverable error.
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(raw, &probe); err != nil {
		return sporecore.NewToolOutputError(
			fmt.Sprintf("invalid parameters for tool %q: %v", t.name, err),
		)
	}
	for _, field := range t.required {
		if _, present := probe[field]; !present {
			return sporecore.NewToolOutputError(
				fmt.Sprintf("invalid parameters for tool %q: missing required field %q", t.name, field),
			)
		}
	}

	// Now decode into the typed input. A present-but-wrong-typed field (e.g. a
	// numeric `key` where a string is expected) errors here — also recoverable.
	var input T
	if err := json.Unmarshal(raw, &input); err != nil {
		return sporecore.NewToolOutputError(
			fmt.Sprintf("invalid parameters for tool %q: %v", t.name, err),
		)
	}
	return t.execFn(ctx, input, sandbox, toolCtx)
}

// Compile-time check that the synthesized implementation satisfies Tool.
var _ sporecore.Tool = (*definedTool[struct{}])(nil)
