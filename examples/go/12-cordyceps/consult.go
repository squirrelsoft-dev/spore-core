// The two consult tools the analysis worker calls to escalate mid-loop
// (issue #114). Both lower to sporecore.NewToolOutputConsult with a `kind` tag.
//
// In the pre-#131 example a SubagentTool mediated these consults. The #131
// declarative composition has NO SubagentTool seam, so the worker-leaf consult
// propagates all the way up to a top-level sporecore.RunResult.Consult and the
// HOST run loop mediates it instead — routing by `kind` to a helper harness with
// a per-kind budget + overflow policy (see main.go's mediateConsult). The seam
// moved; the #114 semantics are identical.
//
// Neither tool captures any host state — each simply renders its call input into
// a ConsultRequest and returns NewToolOutputConsult. The composed tree pauses
// (RunResult.Consult) and the host resumes it with the handler's answer (or a
// BudgetExhausted message).
package main

import (
	"context"
	"encoding/json"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	coretools "github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
)

// kindResearch routes the research consult ladder (→ research handler, web_search).
const kindResearch = "research"

// kindAdvice routes the advice consult ladder (→ advisor handler, cloud model).
const kindAdvice = "advice"

// consultInput is the shared input shape for both consult tools: the worker
// describes where it is stuck and the concrete question it wants answered.
// Attempts is advisory — the host enforces the per-kind budget independently.
type consultInput struct {
	// Situation is a free-form description of where you are stuck or uncertain.
	Situation string `json:"situation"`
	// Question is the concrete question you want answered.
	Question string `json:"question"`
	// Attempts is how many times you have already tried (advisory only).
	Attempts uint32 `json:"attempts"`
}

// consultSchema is the JSON schema advertised for both consult tools.
var consultSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "situation": {
      "type": "string",
      "description": "Free-form description of where you are stuck or uncertain."
    },
    "question": {
      "type": "string",
      "description": "The concrete question you want answered."
    },
    "attempts": {
      "type": "integer",
      "description": "How many times you have already tried (advisory only)."
    }
  },
  "required": ["situation", "question"]
}`)

// researchBestPracticesTool → kind="research". The host routes this to the
// research handler (web_search). Budget 5, overflow SoftFail: on exhaustion the
// worker resumes with BudgetExhausted and finishes on general knowledge. Looking
// up an idiom is normal, not distress, so it never reaches the human.
func researchBestPracticesTool() sporecore.StandardTool {
	return coretools.DefineTool(
		"research_best_practices",
		"Ask a research helper to web-search current best practices or idioms when you are "+
			"unsure whether a pattern is a real defect. Pass `situation` and a focused `question`. "+
			"Returns cited findings; use sparingly.",
		sporecore.ToolAnnotations{OpenWorld: true},
		consultSchema,
		func(_ context.Context, in consultInput, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
			return sporecore.NewToolOutputConsult(sporecore.ConsultRequest{
				Kind:      kindResearch,
				Situation: in.Situation,
				Attempts:  in.Attempts,
				Question:  in.Question,
			})
		},
	)
}

// consultAdvisorTool → kind="advice". The host routes this to the advisor (a
// near-frontier cloud model with read_file/grep). Budget 3, overflow
// EscalateToHuman: on exhaustion the host surfaces the three-choice ladder to the
// operator and resumes with their decision.
func consultAdvisorTool() sporecore.StandardTool {
	return coretools.DefineTool(
		"consult_advisor",
		"Ask a senior advisor agent (a stronger model that can read_file/grep the repo) when "+
			"you are stuck on whether a finding is real or how to rank its severity. Pass "+
			"`situation` and a concrete `question`. Reserve for genuine uncertainty — the advisor "+
			"budget is small.",
		sporecore.ToolAnnotations{OpenWorld: true},
		consultSchema,
		func(_ context.Context, in consultInput, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
			return sporecore.NewToolOutputConsult(sporecore.ConsultRequest{
				Kind:      kindAdvice,
				Situation: in.Situation,
				Attempts:  in.Attempts,
				Question:  in.Question,
			})
		},
	)
}
