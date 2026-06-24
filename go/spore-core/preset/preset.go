// Package preset holds the SC-8 builder presets — CodingAgent + HillClimber —
// the two working-consumer harness shapes extracted into one-call constructors
// (mirrors Rust's HarnessBuilder::{coding_agent,hill_climber},
// rust/crates/spore-core/src/harness.rs).
//
// Both build on observability.ConversationalBuilder and the Phase 1–2 knobs, so
// a consumer collapses to one call. They live in this dedicated package rather
// than alongside ConversationalBuilder because CodingAgent needs the coding tool
// catalogue from the tools package, and tools imports storage, which imports
// observability — so observability cannot import tools (import cycle). This
// package sits ABOVE observability + tools + metric and depends on all three,
// which is the cycle-free home that keeps both presets together with one public
// API surface.
package preset

import (
	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/metric"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
)

// CodingAgentSystemPrompt is the built-in system prompt for CodingAgent: a
// coding agent that ACTS through the workspace tools (rather than describing what
// it would do) and narrates each step to the user via send_message. Exposed so a
// consumer can extend it; override it wholesale with HarnessBuilder.SystemPrompt.
//
// COPIED VERBATIM from the Rust CODING_AGENT_SYSTEM_PROMPT for cross-language
// parity — do not edit without porting the same change to every language.
const CodingAgentSystemPrompt = "You are a coding agent working inside a sandboxed workspace directory. " +
	"Explore with list_dir, read_file, grep, and find_files; create and change files with " +
	"write_file and edit_file; run commands with bash. Use relative paths only. " +
	"Act using tools — do not just describe what you would do. When the task is done, " +
	"reply with a short summary of what you changed. " +
	"The user CANNOT see your reasoning or your tool calls — they only see the messages you " +
	"send with the `send_message` tool and your final reply. So before (or as) you act, " +
	"call `send_message` with one short sentence saying what you are about to do, in " +
	"PARALLEL with the tool that does the work, so narration never costs an extra round trip. " +
	"Keep each message to a single short sentence."

// PresetMaxAutoGrants is the default per-scope auto-continue cap for the
// autonomous presets (CodingAgent / HillClimber): grant up to this many extra
// step budgets at an Escalate point before the run gives up. Mirrors the
// hand-rolled drive loop the consumers used (the 12-cordyceps example's
// MAX_AUTO_CONTINUES). Override the whole policy via HarnessBuilder.EscalationMode.
const PresetMaxAutoGrants uint32 = 10

// PresetStepsPerGrant is the steps granted on each auto-continue for the
// autonomous presets (the 12-cordyceps example's CONTINUE_STEPS). See
// PresetMaxAutoGrants.
const PresetStepsPerGrant uint32 = 25

// presetAutoContinue is the AutoContinue escalation mode with the preset defaults
// (PresetMaxAutoGrants × PresetStepsPerGrant, no on-grant observer) — the
// "autonomous but capped" policy both autonomous presets share.
func presetAutoContinue() sporecore.EscalationMode {
	return sporecore.AutoContinueEscalation(PresetMaxAutoGrants, PresetStepsPerGrant, nil)
}

// CodingAgent assembles an autonomous coding agent over a workspace directory
// (SC-8) — the looper preset.
//
// It builds on observability.ConversationalBuilder and wires the bits a coding
// agent always needs: a read-write WorkspaceScopedSandbox rooted at workspace,
// the full tools.StandardTools{}.CodingSet() (read/write/edit/list/grep/find +
// bash + send_message + web/memory/task-list), the built-in
// CodingAgentSystemPrompt (act-with-tools + send_message narration), and
// AutoContinueEscalation (autonomous-but-capped — it keeps working through a
// spent step budget instead of pausing, so there is no consumer drive loop to
// hand-roll; SC-5).
//
// Window sizing (SC-4/SC-6): size the model's context window ONCE on the model
// before passing it in (e.g. ollama.New(..).WithContextWindow(n)); the preset's
// conversational context manager auto-derives its compaction budget from the
// provider's context window, so no manual ContextManager is needed.
//
// Returns a non-nil error if the workspace path can't be resolved (it must exist
// and canonicalize — the sandbox requirement); the returned builder is nil in
// that case (a typed *sporecore.BuildError, NOT a panic). The strategy is
// per-run: pass a ReAct / PlanExecute Task to Run.
//
// Example:
//
//	model := ollama.New("gemma4:e4b").WithContextWindow(256_000)
//	b, err := preset.CodingAgent(model, "/path/to/project")
//	if err != nil { return err }
//	h := b.Build()
func CodingAgent(model sporecore.ModelInterface, workspace string) (*observability.HarnessBuilder, error) {
	sandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: workspace})
	if err != nil {
		return nil, err
	}
	b := observability.ConversationalBuilder(model).
		Sandbox(sandbox).
		Tools(tools.StandardTools{}.CodingSet()...).
		SystemPrompt(CodingAgentSystemPrompt).
		EscalationMode(presetAutoContinue())
	return b, nil
}

// HillClimber assembles an autonomous hill-climbing agent (SC-8) — the cordyceps
// preset.
//
// It builds on observability.ConversationalBuilder and registers the scoring
// evaluator (required for the HillClimbing loop strategy) under the default ("")
// handle, plus AutoContinueEscalation (autonomous-but-capped; SC-5) so a spent
// per-iteration build budget keeps working instead of pausing.
//
// Unlike CodingAgent this does NOT install a sandbox or tools — hill-climbing
// workspaces vary (some climb a prose artifact, some climb files), and the build
// task's system prompt is task-specific. Add them with the existing
// Sandbox / Tools / SystemPrompt setters as the climb requires. Size the model's
// window on the model first (SC-4/SC-6), as in CodingAgent.
//
// The HillClimbing config (direction / max-stagnation / per-iteration budget)
// lives on the per-run Task's strategy; the iteration ceiling is the task's
// max_turns.
//
// Example:
//
//	model := ollama.New("gemma4:e4b").WithContextWindow(256_000)
//	// Add a workspace Sandbox(..) + Tools(..) for the climb as it requires.
//	h := preset.HillClimber(model, evaluator).Build()
func HillClimber(model sporecore.ModelInterface, evaluator metric.MetricEvaluator) *observability.HarnessBuilder {
	return observability.ConversationalBuilder(model).
		MetricEvaluator(metric.AsHarnessMetricEvaluator(evaluator)).
		EscalationMode(presetAutoContinue())
}
