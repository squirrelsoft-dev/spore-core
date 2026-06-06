// The custom tools the analysis worker uses to escalate mid-loop (issue #114)
// and to load a skill at runtime.
//
// The two consult tools lower to sporecore.NewToolOutputConsult with a `kind`
// tag; the analysis_worker SubagentTool mediates by kind (research →
// research_worker, advice → advisor) using the per-kind budgets + overflow
// policies installed via WithConsultHandlers. They close over no host state, so
// they are plain DefineTool functions.
//
// load_skill DOES close over the shared registry (to validate ids), but a Go
// closure captures it cleanly — no hand-written Tool impl is needed (unlike the
// Rust reference, where the tool! macro produces a zero-sized struct that cannot
// capture state). It appends the validated id to run_store["active_skills"] via
// the ToolContext, and the SkillInjectingContextManager re-injects that body
// every turn.
package main

import (
	"context"
	"encoding/json"
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
	coretools "github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
)

// kindResearch routes the research consult ladder (→ research_worker, web_search).
const kindResearch = "research"

// kindAdvice routes the advice consult ladder (→ advisor, cloud model).
const kindAdvice = "advice"

// consultInput is the shared input shape for both consult tools: the worker
// describes where it is stuck and the concrete question it wants answered.
// Attempts is advisory — the harness enforces the per-kind budget independently.
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

// researchBestPracticesTool → kind="research". Routed to the research worker
// (web_search). Budget 5, overflow SoftFail: on exhaustion the worker resumes
// with BudgetExhausted and finishes on general knowledge. Looking up an idiom is
// normal, not distress, so it never reaches the human.
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

// consultAdvisorTool → kind="advice". Routed to the advisor (near-frontier cloud
// model with read_file/grep). Budget 3, overflow EscalateToHuman: on exhaustion
// the consult converts to RunResult.WaitingForHuman and the REPL surfaces the
// three-choice ladder.
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

// loadSkillInput is the typed input for load_skill.
type loadSkillInput struct {
	// SkillID is the id (name) of the skill to activate, e.g. "audit".
	SkillID string `json:"skill_id"`
}

// loadSkillSchema is advertised to the model.
var loadSkillSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "skill_id": {
      "type": "string",
      "description": "The id (name) of the skill to activate, e.g. \"audit\"."
    }
  },
  "required": ["skill_id"]
}`)

// loadSkillTool builds the load_skill tool, closing over the shared registry so
// it can validate ids. On execute it (1) confirms the named skill exists in the
// registry (rejecting unknown ids recoverably so the model can pick a real one),
// (2) appends the id to run_store["active_skills"] (deduped) via the ToolContext,
// and (3) returns a short confirmation. The body is then injected every turn by
// SkillInjectingContextManager — no new ToolOutput variant, all storage-backed
// (issue #115 "flavor B").
func loadSkillTool(registry *guideregistry.StandardGuideRegistry) sporecore.StandardTool {
	return coretools.DefineTool(
		"load_skill",
		"Activate a skill by id so its full procedure stays in your context for the rest of "+
			"the session. Choose an id from the manifest of available skills.",
		// Writes a small per-run key; not read-only, but harmless.
		sporecore.ToolAnnotations{},
		loadSkillSchema,
		func(ctx context.Context, in loadSkillInput, _ sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
			skillID := strings.TrimSpace(in.SkillID)
			if skillID == "" {
				return sporecore.NewToolOutputError("invalid parameters: `skill_id` (string) is required")
			}

			// 1. Confirm the skill exists. Select returns only Active skill-type
			//    guides; a query with the id as the instruction surfaces it.
			//    Reject unknown ids recoverably.
			query := guideregistry.NewGuideQuery(skillID)
			query.GuideTypes = []guideregistry.GuideType{guideregistry.GuideTypeSkill}
			guides, err := registry.Select(ctx, query)
			known := false
			if err == nil {
				for _, g := range guides {
					if string(g.ID) == skillID {
						known = true
						break
					}
				}
			}
			if !known {
				return sporecore.NewToolOutputError(
					"unknown skill '" + skillID + "'. Pick one of the skills listed in the manifest.")
			}

			// 2. Append to run_store["active_skills"] (dedup). toolCtx.Get/Put are
			//    implicitly keyed by the run's SessionID.
			var active []string
			if value, found, getErr := toolCtx.Get(ctx, activeSkillsKey); getErr != nil {
				return sporecore.NewToolOutputError("load_skill: could not read active set: " + getErr.Error())
			} else if found {
				_ = json.Unmarshal(value, &active)
			}
			already := false
			for _, s := range active {
				if s == skillID {
					already = true
					break
				}
			}
			if !already {
				active = append(active, skillID)
			}
			encoded, _ := json.Marshal(active)
			if putErr := toolCtx.Put(ctx, activeSkillsKey, encoded); putErr != nil {
				return sporecore.NewToolOutputError("load_skill: could not persist active set: " + putErr.Error())
			}

			// 3. Confirm. The body is now injected every turn by the context
			//    manager, so the procedure is "active" from the next turn on.
			return sporecore.NewToolOutputSuccess(
				"Loaded skill '" + skillID + "'. Its procedure is now active — follow it.")
		},
	)
}
