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
	"strings"
)

// ============================================================================
// Ralph loop strategy support types (issue #58)
// ============================================================================

// Canonical labels the Ralph reload context block uses (issue #58, B2). After
// #142 the checkpoint MOVED off these filesystem paths onto the project-id
// RunStore (see RalphProgressKey / RalphFeatureListKey), but the human-readable
// reload labels keep the familiar .spore/ names so the re-seeded window's prompt
// reads the same.
const (
	ralphProgressRelPath    = ".spore/progress.json"
	ralphFeatureListRelPath = ".spore/feature_list.json"
)

// Durable RunStore keys for the Ralph checkpoint (issue #142, decision 3). The
// checkpoint moved off {workspace_root}/.spore/{progress,feature_list}.json onto
// the project-id store so it survives Ralph window resets (NewSessionID() per
// window) and process restarts. These string literals MIRROR
// storage.RalphProgressKey / storage.RalphFeatureListKey — the root sporecore
// package cannot import storage (a cycle), and the namespace-reuse seam means
// both sides key the same (project namespace, key) RunStore axis, so the literal
// strings must agree. (A storage-package test pins this agreement.)
const (
	RalphProgressKey    = "ralph_progress"
	RalphFeatureListKey = "ralph_feature_list"
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
// registers at construction (issue #58, B1; rewired in #142). Drives
// multi-context-window continuation off the DURABLE Ralph progress checkpoint
// (the project-id RunStore under RalphProgressKey, decision 3): while tasks
// remain incomplete it returns Block (the reason describes what is left) so the
// harness loops into a new context window; when complete it returns Continue so
// the loop terminates with success.
//
// Registration is harmless for non-Ralph strategies: when the progress
// checkpoint is ABSENT (or no RunStore is wired) the hook returns Continue and
// does not interfere with other runs. It only blocks when a progress checkpoint
// is PRESENT and reports incomplete tasks — the Ralph contract. The completion
// logic is the SAME ralphCompletionStatus the outer loop consults: one source of
// truth. project is the stable project namespace (#142) the checkpoint is keyed
// by; run is the durable RunStore (nil => inert).
func newRalphStopHook(run RunStore, project SessionID) *FunctionHook {
	return NewFunctionHook(
		"ralph_stop",
		[]HookEvent{HookEventStop},
		func(ctx context.Context, hctx *HookContext) (HookDecision, error) {
			if hctx.Event != HookEventStop {
				return Continue(), nil
			}
			// No store / absent progress checkpoint => do not interfere with
			// non-Ralph runs.
			if run == nil {
				return Continue(), nil
			}
			if _, found, err := run.Get(ctx, project, RalphProgressKey); err != nil || !found {
				return Continue(), nil
			}
			if reason, ok := ralphCompletionStatus(ctx, run, project); ok {
				return Block(reason), nil
			}
			return Continue(), nil
		},
	)
}

// ============================================================================
// Ralph external completion check + filesystem reload (issue #58, B1/B4)
// ============================================================================

// ralphCompletionStatus is the Ralph external completion check (issue #58, B1;
// rewired onto the project store in #142). Reads the DURABLE progress +
// feature-list checkpoints from run keyed by the stable project namespace and
// reports whether the task is complete. Returns ("", false) when complete;
// (reason, true) when tasks remain (reason describes what is left). This is the
// SAME logic the registered Ralph Stop hook applies — one source of truth. A nil
// run reads as "missing" (incomplete).
//
// Contract (both blobs are written by the strategy/initializer via
// WriteRalphProgress / WriteRalphFeatureList — B4, no git):
//   - RalphProgressKey: { "complete": bool, "remaining": [string] }.
//     complete:true with an empty remaining => progress satisfied.
//     Missing/unreadable/invalid => incomplete (so the agent learns to write it).
//   - RalphFeatureListKey: a JSON array of { "name", "passes" } (the
//     termination.FeatureListCheck schema). Any passes:false => incomplete. A
//     MISSING feature list is tolerated here (progress is the primary signal); an
//     invalid one is not.
func ralphCompletionStatus(ctx context.Context, run RunStore, project SessionID) (string, bool) {
	if run == nil {
		return ".spore/progress.json missing", true
	}
	raw, found, err := run.Get(ctx, project, RalphProgressKey)
	if err != nil || !found {
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
	featureRaw, found, err := run.Get(ctx, project, RalphFeatureListKey)
	if err != nil || !found {
		// Missing feature list is tolerated; progress is the primary signal.
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

// ralphReloadContext builds the checkpoint-reload context block injected into
// each fresh context window (issue #58, B4; rewired onto the project store in
// #142). Returns the verbatim progress + feature-list checkpoint contents (when
// present, under the .spore/ labels the prompt has always used) so the re-seeded
// window knows what is already done and what remains. Returns ("", false) when
// neither checkpoint exists (nothing to reload). A nil run reloads nothing.
func ralphReloadContext(ctx context.Context, run RunStore, project SessionID) (string, bool) {
	if run == nil {
		return "", false
	}
	var parts []string
	if raw, found, err := run.Get(ctx, project, RalphProgressKey); err == nil && found {
		parts = append(parts, "Reloaded "+ralphProgressRelPath+":\n"+strings.TrimSpace(string(raw)))
	}
	if raw, found, err := run.Get(ctx, project, RalphFeatureListKey); err == nil && found {
		parts = append(parts, "Reloaded "+ralphFeatureListRelPath+":\n"+strings.TrimSpace(string(raw)))
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
	var session SessionState
	h.config.ContextManager.AppendUserMessage(ctx, &session, instruction)
	// #142: reload the DURABLE checkpoint from the project store (keyed by the
	// stable project namespace), not the sandbox filesystem.
	if reload, ok := ralphReloadContext(ctx, h.config.RunStore, h.config.ProjectNamespace); ok {
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

// RalphCompletionStatus is the Ralph external completion check (issue #58, B1;
// rewired onto the project store in #142): reads the DURABLE checkpoint from the
// project store (keyed by the stable project namespace) and reports (reason,
// incomplete). incomplete=false means complete. Delegates to the package-level
// ralphCompletionStatus — the SAME logic the registered Stop hook reads. (Rust
// async-ified this when the checkpoint moved onto the store; Go's RunStore is
// sync, so this stays sync but threads ctx + the store + project namespace.)
func (h *StandardHarness) RalphCompletionStatus(ctx context.Context) (string, bool) {
	return ralphCompletionStatus(ctx, h.config.RunStore, h.config.ProjectNamespace)
}

// WriteRalphProgress writes the Ralph progress checkpoint to the DURABLE project
// store under RalphProgressKey (issue #142, decision 3 — the WRITE path the
// relocated checkpoint needs; nothing wrote progress.json before). body is the
// raw JSON ({ "complete": bool, "remaining": [string] }). A no-op when no
// RunStore is wired; serialization/store failures are swallowed (best-effort
// checkpoint).
func (h *StandardHarness) WriteRalphProgress(ctx context.Context, body json.RawMessage) {
	if h.config.RunStore == nil {
		return
	}
	_ = h.config.RunStore.Put(ctx, h.config.ProjectNamespace, RalphProgressKey, body)
}

// WriteRalphFeatureList writes the Ralph feature-list checkpoint to the DURABLE
// project store under RalphFeatureListKey (issue #142, decision 3). body is the
// raw JSON array of { "name", "passes" }. A no-op when no RunStore is wired.
func (h *StandardHarness) WriteRalphFeatureList(ctx context.Context, body json.RawMessage) {
	if h.config.RunStore == nil {
		return
	}
	_ = h.config.RunStore.Put(ctx, h.config.ProjectNamespace, RalphFeatureListKey, body)
}

// RalphMaxResets returns the configured Ralph outer-loop reset cap (B3, default
// 3 via effectiveMaxResets).
func (h *StandardHarness) RalphMaxResets() uint32 {
	return h.config.effectiveMaxResets()
}

// EscalationMode returns the configured HITL-vs-AFK escalation mode (#130),
// defaulting the zero value to EscalationSurfaceToHuman via
// EffectiveEscalationMode. Consulted at each ExhaustedResolutionEscalate site.
func (h *StandardHarness) EscalationMode() EscalationMode {
	return h.config.EffectiveEscalationMode()
}
