// Tier-3 control tools (#81): tools that drive the harness via the escalation /
// clarification protocols rather than returning ordinary data.
//
//   - EnterPlanModeTool (enter_plan_mode) →
//     ToolOutput.Escalate{ HarnessSignal.EnterPlanMode{ Context } }.
//   - ExitPlanModeTool (exit_plan_mode) →
//     ToolOutput.Escalate{ HarnessSignal.ExitPlanMode{ Plan } }. The plan is a
//     structured tool param deserialized DIRECTLY into the existing
//     sporecore.PlanArtifact (issue #81, Q4a — no stub).
//   - AskUserQuestionTool (ask_user_question) →
//     ToolOutput.AwaitingClarification{ Question, Options } (issue #81, Q4b).
//     The harness loop pauses with HumanRequest.Clarification.
//   - AbortTool (abort) →
//     ToolOutput.Escalate{ HarnessSignal.Abort{ Reason } }.

package tools

import (
	"context"
	"encoding/json"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// hasField reports whether the JSON object in input contains the named key.
// Used to enforce required fields that Rust gets from serde but Go's
// json.Unmarshal silently defaults — keeping the recoverable-error parity.
func hasField(input []byte, name string) bool {
	var probe map[string]json.RawMessage
	if json.Unmarshal(input, &probe) != nil {
		return false
	}
	_, ok := probe[name]
	return ok
}

// ============================================================================
// EnterPlanMode
// ============================================================================

// EnterPlanModeToolName is the registered tool name.
const EnterPlanModeToolName = "enter_plan_mode"

// EnterPlanModeTool escalates a request to enter plan mode.
type EnterPlanModeTool struct{}

// NewEnterPlanModeTool constructs an EnterPlanModeTool.
func NewEnterPlanModeTool() *EnterPlanModeTool { return &EnterPlanModeTool{} }

func (*EnterPlanModeTool) Name() string                { return EnterPlanModeToolName }
func (*EnterPlanModeTool) IsSubagentTool() bool        { return false }
func (*EnterPlanModeTool) MayProduceLargeOutput() bool { return false }

func (*EnterPlanModeTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        EnterPlanModeToolName,
		Description: "Request entry into plan mode, seeding the planner with context",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {"context": {"type": "string"}}
		}`),
		Annotations: sporecore.ToolAnnotations{},
	}
}

func (t *EnterPlanModeTool) Execute(_ context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params EnterPlanModeParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	sig := sporecore.HarnessSignal{Kind: sporecore.SignalEnterPlanMode, Context: params.Context}
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputEscalate, Signal: &sig}
}

// ============================================================================
// ExitPlanMode
// ============================================================================

// ExitPlanModeToolName is the registered tool name.
const ExitPlanModeToolName = "exit_plan_mode"

// ExitPlanModeTool escalates the produced plan and requests exit from plan mode.
type ExitPlanModeTool struct{}

// NewExitPlanModeTool constructs an ExitPlanModeTool.
func NewExitPlanModeTool() *ExitPlanModeTool { return &ExitPlanModeTool{} }

func (*ExitPlanModeTool) Name() string                { return ExitPlanModeToolName }
func (*ExitPlanModeTool) IsSubagentTool() bool        { return false }
func (*ExitPlanModeTool) MayProduceLargeOutput() bool { return false }

func (*ExitPlanModeTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        ExitPlanModeToolName,
		Description: "Submit the produced plan and request exit from plan mode",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"plan": {
					"type": "object",
					"properties": {
						"tasks": {"type": "array", "items": {"type": "string"}},
						"rationale": {"type": "string"}
					},
					"required": ["tasks"]
				}
			},
			"required": ["plan"]
		}`),
		Annotations: sporecore.ToolAnnotations{},
	}
}

func (t *ExitPlanModeTool) Execute(_ context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params ExitPlanModeParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	plan := params.Plan
	sig := sporecore.HarnessSignal{Kind: sporecore.SignalExitPlanMode, Plan: &plan}
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputEscalate, Signal: &sig}
}

// ============================================================================
// AskUserQuestion
// ============================================================================

// AskUserQuestionToolName is the registered tool name.
const AskUserQuestionToolName = "ask_user_question"

// AskUserQuestionTool pauses the harness awaiting a human clarification.
type AskUserQuestionTool struct{}

// NewAskUserQuestionTool constructs an AskUserQuestionTool.
func NewAskUserQuestionTool() *AskUserQuestionTool { return &AskUserQuestionTool{} }

func (*AskUserQuestionTool) Name() string                { return AskUserQuestionToolName }
func (*AskUserQuestionTool) IsSubagentTool() bool        { return false }
func (*AskUserQuestionTool) MayProduceLargeOutput() bool { return false }

func (*AskUserQuestionTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        AskUserQuestionToolName,
		Description: "Ask the user a clarifying question (optionally with fixed choices)",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"question": {"type": "string"},
				"options": {"type": "array", "items": {"type": "string"}}
			},
			"required": ["question"]
		}`),
		Annotations: sporecore.ToolAnnotations{},
	}
}

func (t *AskUserQuestionTool) Execute(_ context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params AskUserQuestionParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	return sporecore.ToolOutput{
		Kind:     sporecore.ToolOutputAwaitingClarification,
		Question: params.Question,
		Options:  params.Options,
	}
}

// ============================================================================
// Abort
// ============================================================================

// AbortToolName is the registered tool name.
const AbortToolName = "abort"

// AbortTool escalates a graceful-abort signal with a reason.
type AbortTool struct{}

// NewAbortTool constructs an AbortTool.
func NewAbortTool() *AbortTool { return &AbortTool{} }

func (*AbortTool) Name() string                { return AbortToolName }
func (*AbortTool) IsSubagentTool() bool        { return false }
func (*AbortTool) MayProduceLargeOutput() bool { return false }

func (*AbortTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        AbortToolName,
		Description: "Request a graceful abort of the run with a reason",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {"reason": {"type": "string"}},
			"required": ["reason"]
		}`),
		Annotations: sporecore.ToolAnnotations{},
	}
}

func (t *AbortTool) Execute(_ context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params AbortParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	// reason is required (Rust serde rejects its absence; Go json.Unmarshal does
	// not, so validate explicitly to keep the recoverable-error parity).
	if !hasField(call.Input, "reason") {
		return InvalidParameters("missing required field `reason`").ToToolOutput()
	}
	sig := sporecore.HarnessSignal{Kind: sporecore.SignalAbort, Reason: params.Reason}
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputEscalate, Signal: &sig}
}

var (
	_ sporecore.Tool = (*EnterPlanModeTool)(nil)
	_ sporecore.Tool = (*ExitPlanModeTool)(nil)
	_ sporecore.Tool = (*AskUserQuestionTool)(nil)
	_ sporecore.Tool = (*AbortTool)(nil)
)
