package sporecore

import (
	"context"
	"reflect"
	"strings"
	"testing"
	"time"
)

// ============================================================================
// Test doubles
// ============================================================================

// capturingSandbox is an AllowAllSandbox that records the last ExecuteCommand
// invocation and returns a canned CommandOutput, so GitVcsProvider's argv
// construction can be asserted without spawning a real git process.
type capturingSandbox struct {
	AllowAllSandbox
	root string

	command string
	args    []string
	dir     string

	out CommandOutput
}

func (s *capturingSandbox) WorkspaceRoot() string { return s.root }

func (s *capturingSandbox) ExecuteCommand(
	_ context.Context,
	command string,
	args []string,
	workingDir string,
	_ time.Duration,
) (CommandOutput, *SandboxViolation) {
	s.command = command
	s.args = args
	s.dir = workingDir
	return s.out, nil
}

var _ SandboxProvider = (*capturingSandbox)(nil)

// ============================================================================
// (a) FixtureVcsProvider.Log returns the seeded string verbatim.
// ============================================================================

func TestFixtureVcsLogVerbatim(t *testing.T) {
	want := "cafe123 implement login\nbeef456 add login tests"
	provider := FixtureVcsProvider{LogOutput: want, StatusOutput: "clean"}
	got, err := provider.Log(context.Background(), VcsLogArgs{MaxEntries: 5, SinceRef: "HEAD~3", Format: "%h %s"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != want {
		t.Fatalf("Log returned %q, want verbatim %q", got, want)
	}
}

// ============================================================================
// (b) GitVcsProvider builds the correct git log command from VcsLogArgs.
// ============================================================================

func TestGitVcsLogCommandLine(t *testing.T) {
	sb := &capturingSandbox{root: "/work", out: CommandOutput{Stdout: "log output"}}
	git := NewGitVcsProvider(sb, "/work")
	args := VcsLogArgs{MaxEntries: 10, SinceRef: "v1.0", Format: "%h %s"}

	out, err := git.Log(context.Background(), args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "log output" {
		t.Fatalf("Log returned %q, want sandbox stdout", out)
	}
	if sb.command != "git" {
		t.Fatalf("command = %q, want git", sb.command)
	}
	if sb.dir != "/work" {
		t.Fatalf("working dir = %q, want /work", sb.dir)
	}
	want := []string{"log", "-n", "10", "--format=%h %s", "v1.0.."}
	if !reflect.DeepEqual(sb.args, want) {
		t.Fatalf("git log argv = %v, want %v", sb.args, want)
	}
}

// Minimal args (all optional fields unset) emit `git log -n 0`. The `-n` flag
// is emitted UNCONDITIONALLY, matching the Rust reference (and TypeScript and
// Python), so MaxEntries == 0 yields ["log", "-n", "0"] — not ["log"]. This is
// the cross-language consistency contract for the Ralph VcsProvider seam (#58).
func TestGitVcsLogCommandMinimalArgs(t *testing.T) {
	sb := &capturingSandbox{root: "/work", out: CommandOutput{Stdout: ""}}
	git := NewGitVcsProvider(sb, "/work")
	if _, err := git.Log(context.Background(), VcsLogArgs{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []string{"log", "-n", "0"}
	if !reflect.DeepEqual(sb.args, want) {
		t.Fatalf("minimal git log argv = %v, want %v", sb.args, want)
	}
}

// gitLogArgs emits `-n <MaxEntries>` unconditionally, matching the Rust
// reference exactly: MaxEntries == 0 (with no SinceRef/Format) produces
// ["log", "-n", "0"], NOT ["log"]. This pins the cross-language consistency
// contract at the argv-builder level (#58), locking the seam against regression.
func TestGitLogArgsMaxEntriesZeroUnconditional(t *testing.T) {
	got := gitLogArgs(VcsLogArgs{MaxEntries: 0})
	want := []string{"log", "-n", "0"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("gitLogArgs(MaxEntries: 0) = %v, want %v", got, want)
	}
}

// A non-zero exit maps to a *VcsError carrying stderr.
func TestGitVcsLogCommandFailed(t *testing.T) {
	sb := &capturingSandbox{root: "/work", out: CommandOutput{ExitCode: 128, Stderr: "not a git repo"}}
	git := NewGitVcsProvider(sb, "/work")
	_, err := git.Log(context.Background(), VcsLogArgs{})
	if err == nil {
		t.Fatal("expected error on non-zero exit")
	}
	if !strings.Contains(err.Error(), "not a git repo") {
		t.Fatalf("error should carry stderr, got %v", err)
	}
}

// ============================================================================
// (c) Ralph with a FixtureVcsProvider injects vcs_log into the reloaded context
//     across a reset.
// ============================================================================

func TestRalphInjectsVcsLogIntoReload(t *testing.T) {
	dir := t.TempDir()
	a := newRalphAgent(dir,
		ralphWindow{complete: false, remaining: []string{"task A"}},
		ralphWindow{complete: true},
	)
	cfg := ralphCfg(a, dir)
	cfg.MaxResets = 3
	cfg.VcsProvider = FixtureVcsProvider{LogOutput: "cafe123 implement login"}
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	texts := a.turnTexts()
	if len(texts) == 0 {
		t.Fatal("agent never ran")
	}
	if !strings.Contains(texts[0], "Recent VCS history:") ||
		!strings.Contains(texts[0], "cafe123 implement login") {
		t.Fatalf("first window seed missing injected VCS history: %q", texts[0])
	}
}

// ============================================================================
// (d) Ralph with a nil VcsProvider omits any git section (v1 unchanged).
// ============================================================================

func TestRalphNilVcsProviderOmitsGitSection(t *testing.T) {
	dir := t.TempDir()
	a := newRalphAgent(dir, ralphWindow{complete: true})
	cfg := ralphCfg(a, dir)
	cfg.MaxResets = 3
	if cfg.VcsProvider != nil {
		t.Fatal("VcsProvider should default to nil")
	}
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	for i, txt := range a.turnTexts() {
		if strings.Contains(txt, "Recent VCS history:") {
			t.Fatalf("turn %d seed must not contain a git section with nil provider: %q", i, txt)
		}
	}
}

// ============================================================================
// (e) Status round-trips through the provider.
// ============================================================================

func TestFixtureVcsStatusRoundTrips(t *testing.T) {
	provider := FixtureVcsProvider{StatusOutput: "On branch main\nnothing to commit"}
	got, err := provider.Status(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "On branch main\nnothing to commit" {
		t.Fatalf("Status returned %q", got)
	}
}

func TestGitVcsStatusRoundTrips(t *testing.T) {
	sb := &capturingSandbox{root: "/work", out: CommandOutput{Stdout: "clean tree"}}
	git := NewGitVcsProvider(sb, "/work")
	got, err := git.Status(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "clean tree" {
		t.Fatalf("Status returned %q, want sandbox stdout", got)
	}
	want := []string{"status"}
	if !reflect.DeepEqual(sb.args, want) {
		t.Fatalf("git status argv = %v, want %v", sb.args, want)
	}
}
