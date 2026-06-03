package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// echoInput is the typed input for the DefineTool test tool.
type echoInput struct {
	Message string `json:"message"`
	Shout   bool   `json:"shout"`
}

var echoSchema = json.RawMessage(`{
	"type": "object",
	"properties": {
		"message": {"type": "string"},
		"shout": {"type": "boolean"}
	},
	"required": ["message"]
}`)

func echoTool() StandardTool {
	return DefineTool(
		"echo",
		"Echoes the input message",
		sporecore.ToolAnnotations{},
		echoSchema,
		func(_ context.Context, in echoInput, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
			if in.Shout {
				return sporecore.NewToolOutputSuccess(strings.ToUpper(in.Message))
			}
			return sporecore.NewToolOutputSuccess(in.Message)
		},
	)
}

func echoCall(input string) sporecore.ToolCall {
	return sporecore.ToolCall{ID: "c1", Name: "echo", Input: json.RawMessage(input)}
}

// DefineTool synthesizes the Tool methods, carries the explicit schema, and
// runs the typed exec function.
func TestDefineToolBuildsRunnableTool(t *testing.T) {
	tool := echoTool()

	if tool.Schema.Name != "echo" {
		t.Errorf("schema name = %q, want echo", tool.Schema.Name)
	}
	if tool.Schema.Description != "Echoes the input message" {
		t.Errorf("schema description = %q", tool.Schema.Description)
	}
	if tool.Implementation.Name() != "echo" {
		t.Errorf("impl name = %q, want echo", tool.Implementation.Name())
	}
	// Synthesized defaults.
	if tool.Implementation.IsSubagentTool() {
		t.Error("DefineTool tool must not be a subagent tool")
	}
	if tool.Implementation.MayProduceLargeOutput() {
		t.Error("DefineTool tool must not flag large output")
	}
	// Explicit schema is carried through verbatim.
	var probe struct {
		Type       string                     `json:"type"`
		Properties map[string]json.RawMessage `json:"properties"`
	}
	if err := json.Unmarshal(tool.Schema.Parameters, &probe); err != nil {
		t.Fatalf("schema parameters not valid JSON: %v", err)
	}
	if probe.Type != "object" {
		t.Errorf("schema type = %q, want object", probe.Type)
	}
	if _, ok := probe.Properties["message"]; !ok {
		t.Error("schema missing property 'message'")
	}

	out := tool.Implementation.Execute(context.Background(), echoCall(`{"message":"hi","shout":true}`), nil, nil)
	if out.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("Kind = %v, want success (message=%q)", out.Kind, out.Message)
	}
	if out.Content != "HI" {
		t.Errorf("Content = %q, want HI", out.Content)
	}
}

// DefineTool preserves the annotations the caller passes.
func TestDefineToolPreservesAnnotations(t *testing.T) {
	tool := DefineTool(
		"lookup",
		"Reads shared state",
		sporecore.ToolAnnotations{ReadOnly: true, Idempotent: true},
		echoSchema,
		func(_ context.Context, in echoInput, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
			return sporecore.NewToolOutputSuccess(in.Message)
		},
	)
	a := tool.Schema.Annotations
	if !a.ReadOnly || !a.Idempotent {
		t.Errorf("annotations = %+v, want ReadOnly+Idempotent", a)
	}
	if a.Destructive {
		t.Error("annotations must not be destructive")
	}
}

// A missing required field is a recoverable "invalid parameters" error (Go's
// json.Unmarshal would silently zero it; DefineTool checks presence).
func TestDefineToolMissingRequiredFieldIsRecoverable(t *testing.T) {
	tool := echoTool()
	out := tool.Implementation.Execute(context.Background(), echoCall(`{"shout":true}`), nil, nil)
	if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
		t.Fatalf("Kind=%v Recoverable=%v, want recoverable error", out.Kind, out.Recoverable)
	}
	if !strings.Contains(out.Message, "invalid parameters") {
		t.Errorf("Message = %q, want it to contain 'invalid parameters'", out.Message)
	}
	if !strings.Contains(out.Message, "message") {
		t.Errorf("Message = %q, want it to name the missing field 'message'", out.Message)
	}
}

// A present-but-wrong-typed field is also a recoverable "invalid parameters"
// error (the typed decode fails).
func TestDefineToolWrongTypeIsRecoverable(t *testing.T) {
	tool := echoTool()
	out := tool.Implementation.Execute(context.Background(), echoCall(`{"message":123}`), nil, nil)
	if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
		t.Fatalf("Kind=%v Recoverable=%v, want recoverable error", out.Kind, out.Recoverable)
	}
	if !strings.Contains(out.Message, "invalid parameters") {
		t.Errorf("Message = %q, want it to contain 'invalid parameters'", out.Message)
	}
}

// Malformed JSON input is a recoverable "invalid parameters" error.
func TestDefineToolMalformedInputIsRecoverable(t *testing.T) {
	tool := echoTool()
	out := tool.Implementation.Execute(context.Background(), echoCall(`{not json`), nil, nil)
	if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
		t.Fatalf("Kind=%v Recoverable=%v, want recoverable error", out.Kind, out.Recoverable)
	}
	if !strings.Contains(out.Message, "invalid parameters") {
		t.Errorf("Message = %q, want it to contain 'invalid parameters'", out.Message)
	}
}

// A DefineTool tool registers and dispatches through the StandardToolRegistry
// end-to-end.
func TestDefineToolRegistersAndDispatches(t *testing.T) {
	tool := echoTool()
	reg := sporecore.NewStandardToolRegistry()
	if err := reg.Register(tool.Implementation, tool.Schema); err != nil {
		t.Fatalf("Register: %v", err)
	}
	res, err := reg.Dispatch(context.Background(), echoCall(`{"message":"yo"}`), nil)
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if res.Output.Kind != sporecore.ToolOutputSuccess || res.Output.Content != "yo" {
		t.Errorf("output = %+v, want success 'yo'", res.Output)
	}
}
