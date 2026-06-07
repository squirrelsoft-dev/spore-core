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
// Ralph strategy executor primitives (#124)
// ============================================================================
//
// The Ralph OUTER reset loop now lives in the recursive RalphConfig.Run
// (strategy.go), which dispatches c.Inner.Run per window. These leaf primitives
// supply the harness-side machinery the recursive body delegates to: the
// per-window FRESH session seed (instruction + .spore/ reload + optional VCS
// history), the external completion check, and the reset cap.

// RalphSeedSession builds a FRESH per-window SessionState re-seeded from the
// .spore/ filesystem checkpoint (issue #58, R2/R3). Seeds the instruction, then
// injects the reloaded .spore/progress.json + feature_list.json content, then —
// when a VcsProvider is wired — the recent VCS history block. No message
// carryover from any prior window.
func (h *StandardHarness) RalphSeedSession(ctx context.Context, instruction string) SessionState {
	workspaceRoot := ""
	if h.config.Sandbox != nil {
		workspaceRoot = h.config.Sandbox.WorkspaceRoot()
	}
	var session SessionState
	h.config.ContextManager.AppendUserMessage(ctx, &session, instruction)
	if reload, ok := ralphReloadContext(workspaceRoot); ok {
		h.config.ContextManager.AppendUserMessage(ctx, &session, reload)
	}
	if h.config.VcsProvider != nil {
		args := VcsLogArgs{MaxEntries: 20}
		if log, err := h.config.VcsProvider.Log(ctx, args); err == nil {
			if trimmed := strings.TrimSpace(log); trimmed != "" {
				block := "Recent VCS history:\n" + trimmed
				h.config.ContextManager.AppendUserMessage(ctx, &session, block)
			}
		}
	}
	return session
}

// RalphCompletionStatus is the Ralph external completion check (issue #58, B1):
// reads the .spore/ state under the sandbox workspace root and reports (reason,
// incomplete). incomplete=false means complete. Delegates to the package-level
// ralphCompletionStatus — the SAME logic the registered Stop hook reads.
func (h *StandardHarness) RalphCompletionStatus() (string, bool) {
	workspaceRoot := ""
	if h.config.Sandbox != nil {
		workspaceRoot = h.config.Sandbox.WorkspaceRoot()
	}
	return ralphCompletionStatus(workspaceRoot)
}

// RalphMaxResets returns the configured Ralph outer-loop reset cap (B3, default
// 3 via effectiveMaxResets).
func (h *StandardHarness) RalphMaxResets() uint32 {
	return h.config.effectiveMaxResets()
}
