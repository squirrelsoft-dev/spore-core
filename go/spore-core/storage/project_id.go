// project_id.go — project-scoped durable storage namespace (#142).
//
// Where SessionID is regenerated per Ralph context window (NewSessionID()), a
// ProjectID derived from the workspace root stays constant across windows AND
// across process restarts — that stability is the whole point of #142: the
// task_list, plan artifact, Ralph checkpoint, and active-run slot persist under
// it so a window reset re-reads the prior window's work instead of re-planning
// from scratch.
//
// ## Rules enforced (#142)
//   - Derivation + collision policy. ProjectIDFromCanonicalPath is PURE and
//     infallible and delegates to the SAME algorithm as
//     WorkspaceIDFromCanonicalPath ({sanitized_basename}-{8hex}). The 8-hex
//     SHA-256 suffix of the FULL canonical path resolves the /a/b vs /a_b slug
//     collision (distinct canonical strings => distinct hashes => distinct ids);
//     pinned by fixtures/storage/project_id_derivation.json. The FS-touching
//     ProjectIDFromPath / ProjectIDFromCwd canonicalize FIRST (symlinks,
//     relative paths, macOS case-insensitivity).
//   - Namespace reuse, not interface widening. Durable artifacts key the
//     RunStore's existing (SessionID, key) axis via ProjectID.Namespace() — the
//     RunStore interface is UNCHANGED. Ephemeral session/conversation state stays
//     keyed by the per-window SessionID.
//   - Active-run lifecycle (caller-owned). StartOrResumeActiveRun decides
//     start-new vs resume by a deterministic match on the caller-supplied run
//     tag (NOT instruction-diffing, NOT auto-on-success); started_at is an
//     INJECTED Timestamp (deterministic in tests). CompleteActiveRun archives the
//     slot so the next start is fresh.
//
// Mirrors the Rust reference (storage.rs ProjectId / ActiveRun /
// start_or_resume_active_run / complete_active_run) but is written idiomatically
// for Go: the pure derivation reuses the existing WorkspaceID helpers, the
// FS-touching constructors return (ProjectID, error) per Go convention, and the
// active-run helpers take a context.Context as their leading argument.

package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// ============================================================================
// ProjectID
// ============================================================================

// ProjectID is a STABLE identifier for a project, used as the durable storage
// namespace (#142). Form: {sanitized_basename}-{8hex}, lowercased — IDENTICAL to
// WorkspaceID. The two share the pure derivation
// (WorkspaceIDFromCanonicalPath); a ProjectID differs only in carrying
// FS-touching constructors that canonicalize first.
//
// ## /a/b vs /a_b collision policy (RESOLVED)
// A naive "slashes -> underscores" slug would map both /a/b and /a_b to the same
// string. This derivation does NOT collide: it slugs ONLY the final basename and
// appends the first 8 hex of the SHA-256 of the FULL canonical path string. /a/b
// and /a_b have different canonical strings, hence different hashes, hence
// distinct ids (b-<h1> vs a-b-<h2>). The fixture
// fixtures/storage/project_id_derivation.json pins this distinct-id case.
//
// ## Namespace reuse
// The RunStore is keyed by (SessionID, key). Rather than widening that
// interface, a ProjectID is projected onto the same axis via Namespace(), which
// yields a SessionID-typed key whose string IS the derived project id. Durable
// call sites pass that key in place of the per-window session id, so the
// interface stays stable while the value keyed is the stable project namespace.
//
// Go initialism convention uses ID (uppercase), so the exported name is
// ProjectID.
type ProjectID string

// ProjectIDFromCanonicalPath derives a ProjectID from an already-OS-canonicalized
// path. PURE and infallible — it never touches the filesystem. This is the
// cross-language fixture anchor (fixtures/storage/project_id_derivation.json); it
// reuses the EXACT same algorithm as WorkspaceIDFromCanonicalPath
// (canonicalizePathString + sanitizeBasename + 8-hex-SHA-256), so the two
// derivations are byte-identical for the same input. Do NOT duplicate the
// algorithm here.
func ProjectIDFromCanonicalPath(path string) ProjectID {
	return ProjectID(WorkspaceIDFromCanonicalPath(path))
}

// ProjectIDFromPath derives a ProjectID from a path, canonicalizing the
// filesystem FIRST (resolves symlinks, relative components, and macOS
// case-insensitivity) before delegating to the pure ProjectIDFromCanonicalPath.
// Returns a non-nil error wrapping the underlying filepath/os failure if the
// path cannot be canonicalized (does not exist, a component is not a directory,
// a permission error, a broken symlink, ...).
func ProjectIDFromPath(path string) (ProjectID, error) {
	canonical, err := canonicalizeFilesystemPath(path)
	if err != nil {
		return "", fmt.Errorf("project id canonicalization failed: %w", err)
	}
	return ProjectIDFromCanonicalPath(canonical), nil
}

// ProjectIDFromCwd derives a ProjectID from the current working directory,
// canonicalizing FIRST. Convenience wrapper over ProjectIDFromPath for binaries
// that want the process cwd; the harness itself derives from the sandbox
// workspace root, NOT process cwd (decision 5).
func ProjectIDFromCwd() (ProjectID, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("project id canonicalization failed: %w", err)
	}
	return ProjectIDFromPath(cwd)
}

// canonicalizeFilesystemPath resolves a path to its OS-canonical form: symlinks
// are followed (filepath.EvalSymlinks, which also requires the path to exist)
// and the result is made absolute. This is the FS-touching step the pure
// derivation deliberately does NOT perform.
func canonicalizeFilesystemPath(path string) (string, error) {
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	abs, err := filepath.Abs(resolved)
	if err != nil {
		return "", err
	}
	return abs, nil
}

// Namespace projects this ProjectID onto the RunStore's (SessionID, key) string
// axis (the namespace-reuse seam, #142). The returned SessionID is NOT a real
// session — its string IS the derived project id — so durable RunStore Get/Put
// calls key by the stable project namespace without widening the interface.
// Ephemeral session-keyed state keeps using the real per-window SessionID.
func (p ProjectID) Namespace() SessionID {
	return SessionID(string(p))
}

// String returns the underlying derived id string.
func (p ProjectID) String() string { return string(p) }

// ============================================================================
// Durable run-store keys (#142)
// ============================================================================

const (
	// ActiveRunKey is the reserved RunStore key under the project namespace
	// holding the ActiveRun slot. The caller owns the lifecycle: start-new vs
	// resume is a deterministic match on the caller-supplied run tag, NOT
	// instruction-diffing and NOT auto-on-success.
	ActiveRunKey = "active_run"

	// RalphProgressKey is the reserved RunStore key under the project namespace
	// holding the Ralph progress checkpoint (#142, decision 3). The content
	// previously lived at {workspace_root}/.spore/progress.json — it now lives in
	// the project-id store so it survives Ralph window resets and process
	// restarts.
	RalphProgressKey = "ralph_progress"

	// RalphFeatureListKey is the reserved RunStore key under the project
	// namespace holding the Ralph feature-list checkpoint (#142, decision 3).
	// Mirrors the old {workspace_root}/.spore/feature_list.json.
	RalphFeatureListKey = "ralph_feature_list"
)

// ============================================================================
// Active-run lifecycle (#142, decision 2)
// ============================================================================

// ActiveRunStatus is the lifecycle status of the project's active run.
type ActiveRunStatus string

const (
	// ActiveRunStatusActive means the run is live: a window reset under the SAME
	// run tag resumes it.
	ActiveRunStatusActive ActiveRunStatus = "active"
	// ActiveRunStatusCompleted means the run was explicitly completed via
	// CompleteActiveRun; the slot is archived. A subsequent start under a NEW tag
	// (or even the same tag) begins fresh.
	ActiveRunStatusCompleted ActiveRunStatus = "completed"
)

// ActiveRun is the active-run slot persisted under ActiveRunKey in the project
// store.
//
// RunTag is caller-supplied and is the sole start-new-vs-resume discriminator
// (decision 2): a StartOrResumeActiveRun call whose tag matches a live slot
// RESUMES; a different tag (or an absent / completed slot) starts FRESH.
// StartedAt is an INJECTED timestamp (decision 2 — no wall-clock nondeterminism)
// so tests are deterministic. JSON tags are snake_case for cross-language wire
// parity with the Rust ActiveRun.
type ActiveRun struct {
	RunTag    string          `json:"run_tag"`
	StartedAt Timestamp       `json:"started_at"`
	Status    ActiveRunStatus `json:"status"`
}

// ActiveRunDecision is the outcome of StartOrResumeActiveRun: did the slot match
// (resume) or did the call mint a fresh active slot (start-new)?
type ActiveRunDecision string

const (
	// ActiveRunStartedNew means no live slot matched the tag — a fresh Active slot
	// was written.
	ActiveRunStartedNew ActiveRunDecision = "started_new"
	// ActiveRunResumed means a live slot under the SAME tag was found — the run
	// reattaches and the slot is left intact.
	ActiveRunResumed ActiveRunDecision = "resumed"
)

// LoadActiveRun reads the active-run slot for project, or (nil, nil) if absent or
// unparseable. A malformed slot is treated as "no live run" — the next start
// mints a fresh one rather than erroring.
func LoadActiveRun(ctx context.Context, run RunStore, project ProjectID) (*ActiveRun, error) {
	raw, found, err := run.Get(ctx, project.Namespace(), ActiveRunKey)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, nil
	}
	var slot ActiveRun
	if json.Unmarshal(raw, &slot) != nil {
		return nil, nil
	}
	return &slot, nil
}

// StartOrResumeActiveRun decides start-new vs resume for project on the
// caller-supplied runTag (decision 2). Deterministic: a live (Active) slot under
// the SAME tag => ActiveRunResumed (the slot is left intact); otherwise a fresh
// Active slot stamped with the injected startedAt is written and
// ActiveRunStartedNew is returned. startedAt is injected (not read from a clock)
// so the result is deterministic in tests.
func StartOrResumeActiveRun(ctx context.Context, run RunStore, project ProjectID, runTag string, startedAt Timestamp) (ActiveRunDecision, error) {
	existing, err := LoadActiveRun(ctx, run, project)
	if err != nil {
		return "", err
	}
	if existing != nil && existing.Status == ActiveRunStatusActive && existing.RunTag == runTag {
		return ActiveRunResumed, nil
	}
	fresh := ActiveRun{RunTag: runTag, StartedAt: startedAt, Status: ActiveRunStatusActive}
	value, err := json.Marshal(fresh)
	if err != nil {
		return "", fmt.Errorf("serialize active run: %w", err)
	}
	if err := run.Put(ctx, project.Namespace(), ActiveRunKey, value); err != nil {
		return "", err
	}
	return ActiveRunStartedNew, nil
}

// CompleteActiveRun marks the active run for project complete (decision 2):
// flips the slot's status to ActiveRunStatusCompleted so the next
// StartOrResumeActiveRun (even under the same tag) starts fresh. A no-op when
// there is no slot to complete.
func CompleteActiveRun(ctx context.Context, run RunStore, project ProjectID) error {
	existing, err := LoadActiveRun(ctx, run, project)
	if err != nil {
		return err
	}
	if existing == nil {
		return nil
	}
	existing.Status = ActiveRunStatusCompleted
	value, err := json.Marshal(existing)
	if err != nil {
		return fmt.Errorf("serialize active run: %w", err)
	}
	return run.Put(ctx, project.Namespace(), ActiveRunKey, value)
}
