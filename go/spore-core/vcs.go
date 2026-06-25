// vcs.go — the VcsProvider seam for the Ralph loop strategy (issue #58 v2).
//
// A thin, read-only VCS abstraction the Ralph strategy calls instead of shelling
// out to git directly, mirroring how SandboxProvider abstracts filesystem/shell
// access: define an interface, ship a real implementation (GitVcsProvider) and a
// deterministic fixture double (FixtureVcsProvider), and inject the chosen one at
// construction via HarnessConfig.VcsProvider.
//
// The v1 Ralph reload (commit d704e6b) re-seeded each fresh context window from
// .spore/progress.json + .spore/feature_list.json only; the spec's "reload git
// log" step was deferred (B4) because there was no hermetic, cross-language
// testable seam for VCS reads. This interface IS that seam.
//
// Ralph calls Log during its reload phase and injects the output into the next
// window's seed as a clearly delimited "Recent VCS history:" section. When NO
// provider is wired (HarnessConfig.VcsProvider == nil, the default) the git-log
// section is OMITTED and Ralph behaves exactly like v1 — the B4→nil decision.
//
// Mirrors the Rust reference (rust/crates/spore-core/src/harness.rs @ 55f45e8:
// VcsProvider / VcsLogArgs / VcsError / GitVcsProvider / FixtureVcsProvider) but
// is written idiomatically for Go, following the consumer-side interface pattern
// in go/CONVENTIONS.md and the SandboxProvider seam it wraps.

package sporecore

import (
	"context"
	"fmt"
	"strconv"
)

// ============================================================================
// VcsProvider interface (issue #58 v2)
// ============================================================================

// VcsProvider is the read-only VCS abstraction the Ralph loop strategy uses to
// reload git history between context windows (issue #58 v2, decision B4). It
// mirrors the SandboxProvider seam: a small interface with a real git-backed
// implementation (GitVcsProvider) and a deterministic test double
// (FixtureVcsProvider). Methods take a context.Context as their first argument,
// matching every other harness component (go/CONVENTIONS.md).
type VcsProvider interface {
	// Log returns the project's commit log, shaped by args. The returned string
	// is the verbatim VCS output (e.g. git log stdout); the caller does not parse
	// it — it is injected into the reloaded context block as-is.
	Log(ctx context.Context, args VcsLogArgs) (string, error)

	// Status returns the working-tree status (e.g. git status stdout), verbatim.
	Status(ctx context.Context) (string, error)

	// Revert discards the working tree's uncommitted changes, returning it to
	// the last known-good state (SC-14). Called by the HillClimbing loop to undo
	// a no-improvement iteration's edits. Go has no default interface methods, so
	// — unlike the Rust reference where Revert defaults to a no-op — every
	// implementer must define it; the no-op stand-in lives on FixtureVcsProvider
	// (a provider whose workspace is not under version control simply leaves the
	// tree as-is). GitVcsProvider runs git reset --hard HEAD through the sandbox.
	// Best-effort: the HillClimbing loop swallows any error and continues.
	Revert(ctx context.Context) error
}

// VcsLogArgs shapes a VcsProvider.Log read. Each field maps to a git log flag in
// GitVcsProvider. Optional fields use their zero value to mean "unset"
// (Go-idiomatic):
//   - MaxEntries → -n <N> (omitted when 0: full history),
//   - SinceRef   → <ref>.. (omitted when "": full history),
//   - Format     → --format=<fmt> (omitted when "": git's default formatting).
type VcsLogArgs struct {
	// MaxEntries caps the number of commits returned (git log -n <MaxEntries>).
	// Zero means unset — no -n flag is emitted.
	MaxEntries int
	// SinceRef restricts to commits reachable after this ref (git log <ref>..).
	// Empty means unset — the full history is returned (subject to MaxEntries).
	SinceRef string
	// Format is a custom git log --format=<Format> string. Empty means unset —
	// git's default formatting is used.
	Format string
}

// VcsError is the typed error a VcsProvider returns when a VCS command fails
// (issue #58 v2). It mirrors the per-component error-struct convention
// (go/CONVENTIONS.md): a struct implementing error, matchable via errors.As. A
// non-nil Sandbox field means the command was blocked or could not be spawned by
// the sandbox; otherwise Message carries the captured stderr of a non-zero exit.
type VcsError struct {
	// Message is the captured stderr of a command that exited non-zero. Empty
	// when the failure originated in the sandbox (see Sandbox).
	Message string
	// Sandbox is set when the VCS command was blocked or could not be spawned by
	// the sandbox; nil for a plain non-zero-exit command failure.
	Sandbox *SandboxViolation
}

// Error implements error.
func (e *VcsError) Error() string {
	if e.Sandbox != nil {
		return fmt.Sprintf("vcs command blocked by sandbox: %s", e.Sandbox.Error())
	}
	return fmt.Sprintf("vcs command failed: %s", e.Message)
}

// Unwrap exposes the underlying SandboxViolation (when any) to errors.Is/As.
func (e *VcsError) Unwrap() error {
	if e.Sandbox != nil {
		return e.Sandbox
	}
	return nil
}

// ============================================================================
// GitVcsProvider — real implementation (issue #58 v2)
// ============================================================================

// GitVcsProvider is the real VcsProvider that shells out to git THROUGH a
// SandboxProvider (issue #58 v2). It wraps the sandbox and calls
// ExecuteCommand("git", ...) — it never bypasses sandboxing to spawn git
// directly. The command line is built from VcsLogArgs (see that type for the
// flag mapping); Status runs git status. All commands run in WorkspaceRoot.
type GitVcsProvider struct {
	sandbox       SandboxProvider
	workspaceRoot string
}

// NewGitVcsProvider wraps sandbox, running git invocations in workspaceRoot.
func NewGitVcsProvider(sandbox SandboxProvider, workspaceRoot string) *GitVcsProvider {
	return &GitVcsProvider{sandbox: sandbox, workspaceRoot: workspaceRoot}
}

// gitLogArgs builds the git log argument vector from args (a function so the flag
// mapping can be asserted independently of process execution).
func gitLogArgs(args VcsLogArgs) []string {
	out := []string{"log", "-n", strconv.Itoa(args.MaxEntries)}
	if args.Format != "" {
		out = append(out, "--format="+args.Format)
	}
	if args.SinceRef != "" {
		out = append(out, args.SinceRef+"..")
	}
	return out
}

// Log runs git log (shaped by args) through the sandbox and returns its stdout.
func (g *GitVcsProvider) Log(ctx context.Context, args VcsLogArgs) (string, error) {
	return g.run(ctx, gitLogArgs(args))
}

// Status runs git status through the sandbox and returns its stdout.
func (g *GitVcsProvider) Status(ctx context.Context) (string, error) {
	return g.run(ctx, []string{"status"})
}

// Revert runs git reset --hard HEAD THROUGH the sandbox, discarding the working
// tree's uncommitted changes (SC-14). The behavior is relocated here from the
// HillClimbing loop's former hardcoded revert; it never spawns git directly.
func (g *GitVcsProvider) Revert(ctx context.Context) error {
	_, err := g.run(ctx, []string{"reset", "--hard", "HEAD"})
	return err
}

// run executes git with argv through the wrapped sandbox, mapping a sandbox
// violation or a non-zero exit to a *VcsError.
func (g *GitVcsProvider) run(ctx context.Context, argv []string) (string, error) {
	out, violation := g.sandbox.ExecuteCommand(ctx, "git", argv, g.workspaceRoot, 0)
	if violation != nil {
		return "", &VcsError{Sandbox: violation}
	}
	if out.ExitCode != 0 {
		return "", &VcsError{Message: out.Stderr}
	}
	return out.Stdout, nil
}

var _ VcsProvider = (*GitVcsProvider)(nil)

// ============================================================================
// FixtureVcsProvider — deterministic test double (issue #58 v2)
// ============================================================================

// FixtureVcsProvider is the deterministic VcsProvider double for tests and
// fixture replay (issue #58 v2). It returns pre-loaded strings VERBATIM with no
// process spawning, so multi-context-window Ralph continuation can be exercised
// hermetically. Log ignores its VcsLogArgs and yields LogOutput; Status yields
// StatusOutput. It lives alongside the production types (mirroring MockAgent /
// AllowAllSandbox in harness.go) so test files across the package can use it.
type FixtureVcsProvider struct {
	LogOutput    string
	StatusOutput string
}

// Log returns the seeded LogOutput verbatim, ignoring args.
func (f FixtureVcsProvider) Log(_ context.Context, _ VcsLogArgs) (string, error) {
	return f.LogOutput, nil
}

// Status returns the seeded StatusOutput verbatim.
func (f FixtureVcsProvider) Status(_ context.Context) (string, error) {
	return f.StatusOutput, nil
}

// Revert is a no-op (SC-14). Go lacks default interface methods, so this trivial
// method stands in for the Rust reference's default no-op revert: a fixture
// double has no working tree to reset, and a non-VCS workspace leaves the tree
// as-is. Always succeeds.
func (f FixtureVcsProvider) Revert(_ context.Context) error {
	return nil
}

var _ VcsProvider = FixtureVcsProvider{}
