package tools

import (
	"context"
	"encoding/json"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func TestSendMessageEchoesContent(t *testing.T) {
	r := NewSendMessageTool().Execute(context.Background(),
		call("send_message", "c1", map[string]any{"content": "hi user"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "hi user" {
		t.Fatalf("expected success echo, got %+v", r)
	}
}

func TestSendMessageBadParamsRecoverable(t *testing.T) {
	r := NewSendMessageTool().Execute(context.Background(),
		sporecore.ToolCall{ID: "c1", Name: "send_message", Input: json.RawMessage(`{}`)}, sporecore.AllowAllSandbox{}, nil)
	// Missing required `content` unmarshals to "" in Go (no parse error); the
	// tool still succeeds with empty content. The registry's required-field
	// check (not the tool) rejects a truly missing field at dispatch. A
	// non-object body is the recoverable parse error.
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("empty object -> success with empty content, got %+v", r)
	}
	r2 := NewSendMessageTool().Execute(context.Background(),
		sporecore.ToolCall{ID: "c1", Name: "send_message", Input: json.RawMessage(`"not-an-object"`)}, sporecore.AllowAllSandbox{}, nil)
	if r2.Kind != sporecore.ToolOutputError || !r2.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r2)
	}
}

// Fixture replay: send_message_event.json. The tool-output half is asserted
// here; the StreamEvent half (UserMessage emission) is exercised by the
// harness-level test in the sporecore package.
func TestSendMessageEventFixtureReplay(t *testing.T) {
	type toolOut struct {
		Kind        string `json:"kind"`
		Content     string `json:"content"`
		Recoverable bool   `json:"recoverable"`
	}
	type smCase struct {
		Name               string          `json:"name"`
		Input              json.RawMessage `json:"input"`
		ExpectedToolOutput toolOut         `json:"expected_tool_output"`
	}
	data := readFixture(t, "send_message_event.json")
	var cases []smCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			out := NewSendMessageTool().Execute(context.Background(),
				sporecore.ToolCall{ID: "c1", Name: "send_message", Input: c.Input}, sporecore.AllowAllSandbox{}, nil)
			switch c.ExpectedToolOutput.Kind {
			case "success":
				if out.Kind != sporecore.ToolOutputSuccess {
					t.Fatalf("expected success, got %+v", out)
				}
				if out.Content != c.ExpectedToolOutput.Content {
					t.Fatalf("content: got %q want %q", out.Content, c.ExpectedToolOutput.Content)
				}
			case "error":
				// missing_content: the spec models a recoverable error. In Go a
				// missing field is caught at the registry's required-field gate,
				// not the tool. Drive the case through the registry to honor the
				// fixture's expectation.
				out := dispatchThroughRegistry(t, NewSendMessageTool(), c.Input)
				if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
					t.Fatalf("expected recoverable error, got %+v", out)
				}
			}
		})
	}
}

// dispatchThroughRegistry registers a tool in a StandardToolRegistry and
// dispatches one call, folding any DispatchError into a ToolOutput the way the
// harness loop does. Used where a fixture's expectation depends on the
// registry's required-field validation rather than the tool itself.
func dispatchThroughRegistry(t *testing.T, tool interface {
	sporecore.Tool
	schemaer
}, input json.RawMessage) sporecore.ToolOutput {
	t.Helper()
	reg := sporecore.NewStandardToolRegistry()
	if err := reg.Register(tool, tool.Schema()); err != nil {
		t.Fatal(err)
	}
	res, err := reg.Dispatch(context.Background(),
		sporecore.ToolCall{ID: "c1", Name: tool.Name(), Input: input}, sporecore.AllowAllSandbox{})
	if err != nil {
		de, ok := err.(*sporecore.DispatchError)
		if !ok {
			return sporecore.ToolOutput{Kind: sporecore.ToolOutputError, Message: err.Error(), Recoverable: false}
		}
		recoverable := de.Kind == sporecore.DispatchErrSchemaValidationFailed
		return sporecore.ToolOutput{Kind: sporecore.ToolOutputError, Message: de.Error(), Recoverable: recoverable}
	}
	return res.Output
}
