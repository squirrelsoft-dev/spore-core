// ralph.go — the Ralph loop strategy (issue #58).
//
// Multi-context-window continuation loop. Each context window is a FRESH
// SessionState (no message carryover) re-seeded with the task instruction plus
// the reloaded .spore/ filesystem state, then run through a bounded inner ReAct
// sub-loop. After each window the strategy consults the SAME filesystem
// completion check the registered Stop hook reads (.spore/progress.json, B1):
//
//   - complete => Success at that iteration.
//   - incomplete => reset into the next window (a fresh SessionState), until the
//     MaxResets cap (B3, default 3) is exhausted, at which point the run halts
//     with RalphCompletionUnmet{Iterations, LastReason}.
//
// Budgets/usage fold across ALL windows; each window's turn budget is fresh (the
// reset discards the turn budget along with the SessionState). The filesystem is
// the load-bearing mechanism that makes multi-context-window work possible — v1
// reloads ONLY progress + feature_list, no git (B4, hermetic).
//
// Mirrors the Rust reference (rust/crates/spore-core/src/harness.rs run_ralph /
// ralph_completion_status / ralph_reload_context / RalphStopHook) but is written
// idiomatically for Go, following established patterns in this repo (the
// SelfVerifying strategy in self_verifying.go for placement, budget folding, and
// observability finalization; the Stop-hook seam in hooks.go).

package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// ============================================================================
// Ralph loop strategy support types (issue #58)
// ============================================================================

// Canonical .spore/ paths the Ralph strategy reads (issue #58, B2). These agree
// with termination.FeatureListCheck's default path.
const (
	ralphProgressRelPath    = ".spore/progress.json"
	ralphFeatureListRelPath = ".spore/feature_list.json"
)

// ralphProgress is the deserialized .spore/progress.json (issue #58, B1/B2). The
// handoff artifact between context windows: Complete is the primary completion
// signal, Remaining lists outstanding work so an incompletion reason can name
// what is left. Tolerant by default (zero value => incomplete) so a
// partially-written file deserializes to "incomplete" rather than erroring.
type ralphProgress struct {
	Complete  bool     `json:"complete"`
	Remaining []string `json:"remaining"`
}

// ralphFeatureEntry is one .spore/feature_list.json entry, mirroring
// termination.FeatureListCheck's schema so the two sources agree (issue #58, B2).
type ralphFeatureEntry struct {
	Name   string `json:"name"`
	Passes bool   `json:"passes"`
}

// ============================================================================
// Ralph Stop hook (issue #58, B1)
// ============================================================================

// newRalphStopHook builds the Stop lifecycle hook the Ralph loop strategy
// registers at construction (issue #58, B1). Drives multi-context-window
// continuation off .spore/progress.json: while tasks remain incomplete it
// returns Block (the reason describes what is left) so the harness loops into a
// new context window; when complete it returns Continue so the loop terminates
// with success.
//
// Registration is harmless for non-Ralph strategies: when .spore/progress.json
// is ABSENT the hook returns Continue and does not interfere with other runs. It
// only blocks when a progress file is PRESENT and reports incomplete tasks — the
// Ralph contract. The completion logic is the SAME ralphCompletionStatus the
// outer loop consults: one source of truth.
func newRalphStopHook(workspaceRoot string) *FunctionHook {
	return NewFunctionHook(
		"ralph_stop",
		[]HookEvent{HookEventStop},
		func(_ context.Context, hctx *HookContext) (HookDecision, error) {
			if hctx.Event != HookEventStop {
				return Continue(), nil
			}
			// Absent progress file => do not interfere with non-Ralph runs.
			if _, err := os.Stat(filepath.Join(workspaceRoot, ralphProgressRelPath)); err != nil {
				return Continue(), nil
			}
			if reason, ok := ralphCompletionStatus(workspaceRoot); ok {
				return Block(reason), nil
			}
			return Continue(), nil
		},
	)
}

// ============================================================================
// Ralph external completion check + filesystem reload (issue #58, B1/B4)
// ============================================================================

// ralphCompletionStatus is the Ralph external completion check (issue #58, B1).
// Reads the deterministic .spore/ files under workspaceRoot and reports whether
// the task is complete. Returns ("", false) when complete; (reason, true) when
// tasks remain (reason describes what is left). This is the SAME logic the
// registered Ralph Stop hook applies — one source of truth.
//
// Contract (both files are written by the strategy/initializer and fixture
// cleanly — B4, no git):
//   - .spore/progress.json: { "complete": bool, "remaining": [string] }.
//     complete:true with an empty remaining => progress satisfied.
//     Missing/unreadable/invalid => incomplete (so the agent learns to write it).
//   - .spore/feature_list.json: a JSON array of { "name", "passes" } (the
//     termination.FeatureListCheck schema). Any passes:false => incomplete. A
//     MISSING feature list is tolerated here (progress.json is the primary
//     signal); an invalid one is not.
func ralphCompletionStatus(workspaceRoot string) (string, bool) {
	raw, err := os.ReadFile(filepath.Join(workspaceRoot, ralphProgressRelPath))
	if err != nil {
		return ".spore/progress.json missing", true
	}
	var progress ralphProgress
	if err := json.Unmarshal(raw, &progress); err != nil {
		return ".spore/progress.json invalid JSON: " + err.Error(), true
	}
	if !progress.Complete {
		if len(progress.Remaining) == 0 {
			return "task not marked complete", true
		}
		return "remaining: " + strings.Join(progress.Remaining, ", "), true
	}
	if len(progress.Remaining) > 0 {
		return "remaining: " + strings.Join(progress.Remaining, ", "), true
	}

	// Progress says done — corroborate against the feature list when present.
	featureRaw, err := os.ReadFile(filepath.Join(workspaceRoot, ralphFeatureListRelPath))
	if err != nil {
		// Missing feature list is tolerated; progress.json is the primary signal.
		return "", false
	}
	var entries []ralphFeatureEntry
	if err := json.Unmarshal(featureRaw, &entries); err != nil {
		return ".spore/feature_list.json invalid JSON: " + err.Error(), true
	}
	var incomplete []string
	for _, e := range entries {
		if !e.Passes {
			incomplete = append(incomplete, e.Name)
		}
	}
	if len(incomplete) > 0 {
		return "incomplete features: " + strings.Join(incomplete, ", "), true
	}
	return "", false
}

// ralphReloadContext builds the filesystem-reload context block injected into
// each fresh context window (issue #58, B4). Returns the verbatim
// .spore/progress.json and .spore/feature_list.json contents (when present) so
// the re-seeded window knows what is already done and what remains. Returns
// ("", false) when neither file exists (nothing to reload).
func ralphReloadContext(workspaceRoot string) (string, bool) {
	var parts []string
	if raw, err := os.ReadFile(filepath.Join(workspaceRoot, ralphProgressRelPath)); err == nil {
		parts = append(parts, "Reloaded .spore/progress.json:\n"+strings.TrimSpace(string(raw)))
	}
	if raw, err := os.ReadFile(filepath.Join(workspaceRoot, ralphFeatureListRelPath)); err == nil {
		parts = append(parts, "Reloaded .spore/feature_list.json:\n"+strings.TrimSpace(string(raw)))
	}
	if len(parts) == 0 {
		return "", false
	}
	return strings.Join(parts, "\n\n"), true
}

// ============================================================================
// Ralph strategy driver
// ============================================================================

// runRalph drives the Ralph loop strategy (issue #58).
//
// Config fields read:
//   - config.MaxResets (B3, default 3 via effectiveMaxResets) — the OUTER loop
//     cap: the maximum number of context windows run before halting with
//     HaltRalphCompletionUnmet when tasks are still incomplete.
//   - config.Sandbox.WorkspaceRoot() — where the .spore/ state lives.
//
// Terminal HaltReason produced:
//   - HaltRalphCompletionUnmet{Iterations, Reason} — reached MaxResets context
//     windows but the filesystem completion check never reported done.
//
// Rules enforced (each maps to a test): R1 the outer loop resets the context
// window on incomplete; R2 a FRESH SessionState per window (no carryover); R3
// the filesystem reload injects reloaded .spore/ content into the fresh seed; R4
// incomplete,incomplete,complete => Success at iteration 3; R5 always-incomplete
// => exactly MaxResets windows => RalphCompletionUnmet; R6 budgets fold across
// ALL windows; R7 each reset is traceable (a distinct generated session id per
// window, finalized via observability).
func (h *StandardHarness) runRalph(
	ctx context.Context,
	task Task,
	budget BudgetSnapshot,
	_ StreamSink,
) RunResult {
	workspaceRoot := ""
	if h.config.Sandbox != nil {
		workspaceRoot = h.config.Sandbox.WorkspaceRoot()
	}
	maxResets := h.config.effectiveMaxResets()
	// Ralph's incoming budget snapshot is irrelevant — each context window is a
	// fresh start with its own per-window turn budget (the reset discards the
	// turn budget along with the SessionState). Token/turn accounting is
	// accumulated separately for terminal reporting (R6).
	_ = budget

	// Cumulative usage + turns across ALL context windows (R6).
	var totalUsage AggregateUsage
	var cumulativeTurns uint32
	// The most recent incompletion reason (for RalphCompletionUnmet).
	lastReason := ".spore/progress.json missing"
	// Session id of the most recent context window (terminal accounting).
	lastSessionID := task.SessionID

	// The OUTER loop: each iteration is ONE context window. maxResets caps the
	// number of windows (B3). Iteration 0 is the first window; a reset is the
	// transition into the next iteration (R1).
	for iteration := uint32(0); iteration < maxResets; iteration++ {
		// R7: a fresh, distinct session id per context window so each reset is
		// independently traceable.
		windowSessionID := task.SessionID
		if iteration > 0 {
			windowSessionID = NewSessionID()
		}
		lastSessionID = windowSessionID

		// R2: a FRESH SessionState per window — discard the prior one. No message
		// carryover; the window is re-seeded from scratch.
		var session SessionState

		// Seed the instruction (R2) then R3: reload the deterministic .spore/
		// state from the filesystem and inject it as context so the fresh window
		// knows what is already done / still outstanding.
		h.config.ContextManager.AppendUserMessage(ctx, &session, task.Instruction)
		if reload, ok := ralphReloadContext(workspaceRoot); ok {
			h.config.ContextManager.AppendUserMessage(ctx, &session, reload)
		}

		// The per-window bounded ReAct sub-loop. The registered Stop hook (B1)
		// fires inside it on each FinalResponse; this strategy's OUTER loop then
		// decides reset vs success off the same filesystem state.
		windowTask := Task{
			ID:           task.ID,
			Instruction:  task.Instruction,
			SessionID:    windowSessionID,
			Budget:       task.Budget,
			LoopStrategy: task.LoopStrategy,
		}
		windowCap := allTurns
		if task.Budget.MaxTurns != nil {
			windowCap = *task.Budget.MaxTurns
		}
		// FRESH per-window budget: the context-window reset resets the turn budget
		// too. Token fold is accumulated separately via totalUsage.
		carried := BudgetSnapshot{}
		windowResult := h.runReActInner(ctx, windowTask, windowCap, session, carried, nil)
		foldSelfVerifyUsage(&totalUsage, &carried, windowResult)
		cumulativeTurns += carried.Turns

		// A window that paused / escalated is propagated up unchanged.
		switch windowResult.Kind {
		case RunWaitingForHuman:
			// Not terminal — do not finalize.
			return windowResult
		case RunEscalate:
			h.finalizeObservability(ctx, windowResult.SessionID, TerminalEscalated, "")
			return windowResult
		}

		// External completion check (B1): consult the SAME filesystem state the
		// Stop hook reads. complete => Success; incomplete => reset into the next
		// window (R1) unless the cap is reached (R5).
		reason, incomplete := ralphCompletionStatus(workspaceRoot)
		if !incomplete {
			output := ""
			if windowResult.Kind == RunSuccess {
				output = windowResult.Output
			}
			result := RunResult{
				Kind:      RunSuccess,
				Output:    output,
				SessionID: windowSessionID,
				Usage:     totalUsage,
				Turns:     cumulativeTurns,
			}
			h.finalizeObservability(ctx, windowSessionID, TerminalSuccess, "")
			return result
		}
		lastReason = reason
	}

	// R5: ran out of context-window resets without completion.
	result := RunResult{
		Kind: RunFailure,
		Reason: HaltReason{
			Kind:       HaltRalphCompletionUnmet,
			Iterations: maxResets,
			Reason:     lastReason,
		},
		SessionID: lastSessionID,
		Usage:     totalUsage,
		Turns:     cumulativeTurns,
	}
	h.finalizeObservability(ctx, lastSessionID, TerminalFailure, haltReasonString(result.Reason))
	return result
}
