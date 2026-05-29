// SendMessage tool (#81, net-new Tier-1 tool).
//
// send_message surfaces an out-of-band message to the user. The TOOL itself is
// trivial: it echoes the content back as a ToolOutput.Success. The harness loop
// is what gives it meaning — it recognizes the send_message tool name, emits a
// HarnessStreamEvent of kind user_message with the content, and records a
// minimal success tool result so the loop continues (see the harness ReAct
// loop). The tool does NOT touch the sandbox or storage.

package tools

import (
	"context"
	"encoding/json"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// SendMessageToolName is the registered tool name. It must stay in sync with
// the harness-side sendMessageToolName constant (duplicated there to avoid the
// tools→sporecore→tools import cycle).
const SendMessageToolName = "send_message"

// SendMessageTool echoes a user-facing message back as success content; the
// harness loop emits the prominent UserMessage stream event.
type SendMessageTool struct{}

// NewSendMessageTool constructs a SendMessageTool.
func NewSendMessageTool() *SendMessageTool { return &SendMessageTool{} }

func (*SendMessageTool) Name() string                { return SendMessageToolName }
func (*SendMessageTool) IsSubagentTool() bool        { return false }
func (*SendMessageTool) MayProduceLargeOutput() bool { return false }

func (*SendMessageTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        SendMessageToolName,
		Description: "Send a message to the user",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {"content": {"type": "string"}},
			"required": ["content"]
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true},
	}
}

func (t *SendMessageTool) Execute(_ context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params SendMessageParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	// The content is returned verbatim; the harness loop reads it off this
	// Success and emits a UserMessage stream event.
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: params.Content}
}

var _ sporecore.Tool = (*SendMessageTool)(nil)
