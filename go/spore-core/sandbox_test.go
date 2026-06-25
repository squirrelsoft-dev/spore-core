package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func cfg(root string) WorkspaceConfig {
	return WorkspaceConfig{Root: root}
}

func TestSandboxBuildFailsWhenRootMissing(t *testing.T) {
	_, err := NewWorkspaceScopedSandbox(WorkspaceConfig{Root: "/definitely/does/not/exist/spore-test-xyz"})
	if err == nil {
		t.Fatalf("expected error")
	}
	be, ok := err.(*BuildError)
	if !ok || be.Kind != BuildErrRootNotFound {
		t.Fatalf("expected RootNotFound, got %v", err)
	}
}

// TestSandboxDefaultIsolationIsWorkspaceScoped pins the safe-by-default
// behaviour of issue #34: a sandbox built without an explicit mode reports
// IsolationWorkspaceScoped — never the gated, no-enforcement IsolationNone.
func TestSandboxDefaultIsolationIsWorkspaceScoped(t *testing.T) {
	dir := t.TempDir()
	sb, err := NewWorkspaceScopedSandbox(cfg(dir))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if sb.IsolationMode().Kind() != "workspace_scoped" {
		t.Fatalf("default isolation kind = %q, want %q", sb.IsolationMode().Kind(), "workspace_scoped")
	}
	if _, ok := sb.IsolationMode().(IsolationWorkspaceScoped); !ok {
		t.Fatalf("default isolation type = %T, want IsolationWorkspaceScoped", sb.IsolationMode())
	}
}

// TestDefaultSandboxIsolationIsWorkspaceScoped pins the same safe default for
// the non-sandboxed DefaultSandbox stub (issue #34).
func TestDefaultSandboxIsolationIsWorkspaceScoped(t *testing.T) {
	var ds DefaultSandbox
	if _, ok := ds.IsolationMode().(IsolationWorkspaceScoped); !ok {
		t.Fatalf("DefaultSandbox isolation type = %T, want IsolationWorkspaceScoped", ds.IsolationMode())
	}
}

func TestSandboxWorkspaceRootIsCanonical(t *testing.T) {
	dir := t.TempDir()
	sb, err := NewWorkspaceScopedSandbox(cfg(dir))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	canonical, _ := filepath.EvalSymlinks(dir)
	if sb.WorkspaceRoot() != canonical {
		t.Fatalf("workspace root: got %q want %q", sb.WorkspaceRoot(), canonical)
	}
}

func TestSandboxEscapeViaDotDot(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewWorkspaceScopedSandbox(cfg(dir))
	_, v := sb.ResolvePath(context.Background(), "../etc/passwd", OperationRead)
	if v == nil || v.Kind != SandboxPathEscape {
		t.Fatalf("expected PathEscape, got %v", v)
	}
}

func TestSandboxEscapeViaAbsolutePath(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewWorkspaceScopedSandbox(cfg(dir))
	_, v := sb.ResolvePath(context.Background(), "/../../etc/passwd", OperationRead)
	if v == nil || v.Kind != SandboxPathEscape {
		t.Fatalf("expected PathEscape, got %v", v)
	}
}

func TestSandboxPathDeniedDenylist(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	if err := os.MkdirAll(filepath.Join(root, "secrets"), 0o755); err != nil {
		t.Fatalf("%v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "secrets", "k.txt"), []byte("shh"), 0o644); err != nil {
		t.Fatalf("%v", err)
	}
	c := cfg(root)
	c.DeniedPaths = []string{"secrets"}
	sb, _ := NewWorkspaceScopedSandbox(c)
	_, v := sb.ResolvePath(context.Background(), "secrets/k.txt", OperationRead)
	if v == nil || v.Kind != SandboxPathDenied {
		t.Fatalf("expected PathDenied, got %v", v)
	}
	if !strings.Contains(v.MatchedRule, "secrets") {
		t.Fatalf("matched_rule %q", v.MatchedRule)
	}
}

func TestSandboxPathDeniedAllowlistMiss(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	_ = os.MkdirAll(filepath.Join(root, "src"), 0o755)
	_ = os.WriteFile(filepath.Join(root, "src", "a.rs"), []byte(""), 0o644)
	_ = os.WriteFile(filepath.Join(root, "other.rs"), []byte(""), 0o644)
	c := cfg(root)
	c.AllowedPaths = []string{"src"}
	sb, _ := NewWorkspaceScopedSandbox(c)
	if _, v := sb.ResolvePath(context.Background(), "src/a.rs", OperationRead); v != nil {
		t.Fatalf("inside allowlist: got %v", v)
	}
	_, v := sb.ResolvePath(context.Background(), "other.rs", OperationRead)
	if v == nil || v.Kind != SandboxPathDenied || v.MatchedRule != "not in allowlist" {
		t.Fatalf("expected allowlist miss, got %v", v)
	}
}

func TestSandboxExtensionDenied(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	_ = os.WriteFile(filepath.Join(root, ".env"), []byte("SECRET=1"), 0o644)
	c := cfg(root)
	c.DeniedExtensions = []string{"env"}
	sb, _ := NewWorkspaceScopedSandbox(c)
	_, v := sb.ResolvePath(context.Background(), ".env", OperationRead)
	if v == nil || v.Kind != SandboxExtensionDenied {
		t.Fatalf("expected ExtensionDenied, got %v", v)
	}
}

func TestSandboxReadOnlyViolation(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	_ = os.WriteFile(filepath.Join(root, "a.txt"), []byte("x"), 0o644)
	c := cfg(root)
	c.ReadOnly = true
	sb, _ := NewWorkspaceScopedSandbox(c)
	if _, v := sb.ResolvePath(context.Background(), "a.txt", OperationRead); v != nil {
		t.Fatalf("read in read-only: got %v", v)
	}
	_, v := sb.ResolvePath(context.Background(), "a.txt", OperationWrite)
	if v == nil || v.Kind != SandboxReadOnly {
		t.Fatalf("expected ReadOnlyViolation, got %v", v)
	}
}

func TestSandboxFileSizeExceeded(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	_ = os.WriteFile(filepath.Join(root, "big.txt"), make([]byte, 1024), 0o644)
	c := cfg(root)
	c.MaxFileSize = 100
	sb, _ := NewWorkspaceScopedSandbox(c)
	_, v := sb.ResolvePath(context.Background(), "big.txt", OperationRead)
	if v == nil || v.Kind != SandboxFileSizeExceeded {
		t.Fatalf("expected FileSizeExceeded, got %v", v)
	}
	if v.Size != 1024 || v.Limit != 100 {
		t.Fatalf("got size=%d limit=%d", v.Size, v.Limit)
	}
}

func TestSandboxWriteToNonexistentFile(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	sb, _ := NewWorkspaceScopedSandbox(cfg(root))
	resolved, v := sb.ResolvePath(context.Background(), "new_file.txt", OperationWrite)
	if v != nil {
		t.Fatalf("write-new: got %v", v)
	}
	if filepath.Dir(resolved) != root {
		t.Fatalf("parent mismatch: %s vs %s", filepath.Dir(resolved), root)
	}
}

// Regression for #63: a Read of a not-yet-created file *inside* the workspace
// must resolve via its canonicalized parent (not be misclassified as
// PathEscape). The file is absent; resolution still succeeds so the actual
// read can surface a recoverable not-found.
func TestSandboxReadOfMissingInWorkspaceFileResolves(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	sb, _ := NewWorkspaceScopedSandbox(cfg(root))
	resolved, v := sb.ResolvePath(context.Background(), "output.txt", OperationRead)
	if v != nil {
		t.Fatalf("missing in-workspace read must resolve, got %v", v)
	}
	if filepath.Dir(resolved) != root {
		t.Fatalf("parent mismatch: %s vs %s", filepath.Dir(resolved), root)
	}
	if filepath.Base(resolved) != "output.txt" {
		t.Fatalf("leaf mismatch: %s", filepath.Base(resolved))
	}
	if _, err := os.Stat(resolved); !os.IsNotExist(err) {
		t.Fatalf("expected file to be absent, stat err=%v", err)
	}
}

// Regression for #63: parent dir exists, leaf file does not — still resolves
// for Read.
func TestSandboxReadOfMissingFileInSubdirResolves(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	if err := os.Mkdir(filepath.Join(root, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	sb, _ := NewWorkspaceScopedSandbox(cfg(root))
	resolved, v := sb.ResolvePath(context.Background(), "sub/missing.txt", OperationRead)
	if v != nil {
		t.Fatalf("missing in-subdir read must resolve, got %v", v)
	}
	if filepath.Dir(resolved) != filepath.Join(root, "sub") {
		t.Fatalf("parent mismatch: %s", filepath.Dir(resolved))
	}
	if _, err := os.Stat(resolved); !os.IsNotExist(err) {
		t.Fatalf("expected file to be absent, stat err=%v", err)
	}
}

// Regression for #63: a Read of a *non-existent* path that resolves outside
// the workspace root must still be a PathEscape, not a not-found.
func TestSandboxReadOfMissingFileOutsideRootStillPathEscape(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	sb, _ := NewWorkspaceScopedSandbox(cfg(root))
	_, v := sb.ResolvePath(context.Background(), "../nonexistent_passwd", OperationRead)
	if v == nil || v.Kind != SandboxPathEscape {
		t.Fatalf("expected PathEscape, got %v", v)
	}
}

// Read of an existing in-workspace file still resolves to its real path.
func TestSandboxReadOfExistingInWorkspaceFileResolves(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	if err := os.WriteFile(filepath.Join(root, "present.txt"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	sb, _ := NewWorkspaceScopedSandbox(cfg(root))
	resolved, v := sb.ResolvePath(context.Background(), "present.txt", OperationRead)
	if v != nil {
		t.Fatalf("existing read must resolve, got %v", v)
	}
	if resolved != filepath.Join(root, "present.txt") {
		t.Fatalf("resolved mismatch: %s", resolved)
	}
}

// ============================================================================
// SC-13 — read-everywhere / write-scoped (write_root)
// ============================================================================

func TestWriteRootAllowsReadEverywhereButScopesWrites(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	out := filepath.Join(root, "out")
	if err := os.Mkdir(out, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "secret.txt"), []byte("s"), 0o644); err != nil {
		t.Fatal(err)
	}
	c := cfg(root)
	c.WriteRoot = out
	sb, err := NewWorkspaceScopedSandbox(c)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	// Read-everywhere: a file under root but outside write_root resolves.
	if _, v := sb.ResolvePath(context.Background(), "secret.txt", OperationRead); v != nil {
		t.Fatalf("read under root must succeed, got %v", v)
	}
	// Write-scoped: writing that same file is a PathEscape.
	_, v := sb.ResolvePath(context.Background(), "secret.txt", OperationWrite)
	if v == nil || v.Kind != SandboxPathEscape {
		t.Fatalf("write outside write_root must be PathEscape, got %v", v)
	}
	// A write under write_root (path strings stay root-relative) is OK.
	if _, v := sb.ResolvePath(context.Background(), "out/result.txt", OperationWrite); v != nil {
		t.Fatalf("write under write_root must succeed, got %v", v)
	}
}

func TestNoWriteRootGatesWritesByRoot(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	sb, _ := NewWorkspaceScopedSandbox(cfg(root))
	// With write_root unset, writes resolve anywhere under root (legacy).
	if _, v := sb.ResolvePath(context.Background(), "a.txt", OperationWrite); v != nil {
		t.Fatalf("legacy write under root must succeed, got %v", v)
	}
}

func TestBuildFailsWhenWriteRootMissing(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	c := cfg(root)
	c.WriteRoot = filepath.Join(root, "does-not-exist")
	_, err := NewWorkspaceScopedSandbox(c)
	if err == nil {
		t.Fatalf("expected a build error for a missing write_root")
	}
	be, ok := err.(*BuildError)
	if !ok || be.Kind != BuildErrWriteRootNotFound {
		t.Fatalf("expected WriteRootNotFound, got %v", err)
	}
}

func TestSandboxViolationsRoundTripJSON(t *testing.T) {
	cases := []SandboxViolation{
		{Kind: SandboxPathEscape, Path: "/p"},
		{Kind: SandboxPathDenied, Path: "/p", MatchedRule: "r"},
		{Kind: SandboxExtensionDenied, Path: "/p.env", Extension: "env"},
		{Kind: SandboxReadOnly, Path: "/p"},
		{Kind: SandboxFileSizeExceeded, Path: "/p", Size: 1024, Limit: 100},
		{Kind: SandboxDisallowedCommand, Command: "rm"},
		{Kind: SandboxNetworkViolation, Host: "evil"},
		{Kind: SandboxExecSpawnFailed, Command: "no-such-bin", Message: "exec: \"no-such-bin\": executable file not found in $PATH"},
	}
	for _, v := range cases {
		raw, err := json.Marshal(v)
		if err != nil {
			t.Fatalf("marshal %v: %v", v, err)
		}
		var back SandboxViolation
		if err := json.Unmarshal(raw, &back); err != nil {
			t.Fatalf("unmarshal %s: %v", raw, err)
		}
		if back != v {
			t.Fatalf("round-trip:\n got  %+v\n want %+v\n raw  %s", back, v, raw)
		}
	}
}

func TestSandboxExecuteCommandEcho(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewWorkspaceScopedSandbox(cfg(dir))
	out, v := sb.ExecuteCommand(context.Background(), "echo", []string{"hello"}, "", 0)
	if v != nil {
		t.Fatalf("violation: %v", v)
	}
	if out.ExitCode != 0 || !strings.Contains(out.Stdout, "hello") {
		t.Fatalf("bad output: %+v", out)
	}
}

func TestSandboxExecuteCommandTimeout(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewWorkspaceScopedSandbox(cfg(dir))
	out, v := sb.ExecuteCommand(context.Background(), "sleep", []string{"5"}, "", 50*time.Millisecond)
	if v != nil {
		t.Fatalf("violation: %v", v)
	}
	if !out.TimedOut {
		t.Fatalf("expected timed out, got %+v", out)
	}
}

func TestSandboxExecuteCommandBubblewrapDisallowed(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewWorkspaceScopedSandboxWithMode(cfg(dir), IsolationBubblewrap{})
	_, v := sb.ExecuteCommand(context.Background(), "echo", nil, "", 0)
	if v == nil || v.Kind != SandboxDisallowedCommand {
		t.Fatalf("expected DisallowedCommand, got %v", v)
	}
}

// ============================================================================
// SC-15 — typed spawn failure
// ============================================================================

func TestSandboxExecuteCommandSpawnFailureIsTypedViolation(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewWorkspaceScopedSandbox(cfg(dir))
	_, v := sb.ExecuteCommand(context.Background(), "spore-definitely-no-such-binary-xyz", nil, "", 0)
	if v == nil {
		t.Fatalf("expected a SandboxViolation for a missing binary, got nil")
	}
	if v.Kind != SandboxExecSpawnFailed {
		t.Fatalf("expected SandboxExecSpawnFailed, got %v", v)
	}
	if v.Command != "spore-definitely-no-such-binary-xyz" {
		t.Fatalf("command = %q, want the passed command", v.Command)
	}
	// Layer-2: a spawn failure is always recoverable, never halt-eligible.
	if v.IsAlwaysHalt() {
		t.Fatalf("exec_spawn_failed must not be halt-eligible")
	}
}

// ============================================================================
// SC-12 — ExecConfig exec-hardening knobs
// ============================================================================

// execCfg returns a WorkspaceConfig rooted at root with ec wired in.
func execCfg(root string, ec *ExecConfig) WorkspaceConfig {
	c := cfg(root)
	c.ExecConfig = ec
	return c
}

func TestExecConfigDefaultTimeoutAppliesWhenCallPassesNone(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewWorkspaceScopedSandbox(execCfg(dir, &ExecConfig{DefaultTimeout: 50 * time.Millisecond}))
	// No per-call timeout (0) — the ExecConfig floor must still fire.
	out, v := sb.ExecuteCommand(context.Background(), "sleep", []string{"5"}, "", 0)
	if v != nil {
		t.Fatalf("violation: %v", v)
	}
	if !out.TimedOut {
		t.Fatalf("default_timeout did not apply: %+v", out)
	}
}

func TestExecConfigPerCallTimeoutOverridesDefault(t *testing.T) {
	dir := t.TempDir()
	// A generous default that must NOT veto the tight per-call timeout.
	sb, _ := NewWorkspaceScopedSandbox(execCfg(dir, &ExecConfig{DefaultTimeout: 30 * time.Second}))
	out, v := sb.ExecuteCommand(context.Background(), "sleep", []string{"5"}, "", 50*time.Millisecond)
	if v != nil {
		t.Fatalf("violation: %v", v)
	}
	if !out.TimedOut {
		t.Fatalf("per-call timeout should win over default: %+v", out)
	}
}

func TestExecConfigNonInteractiveEnvIsInjected(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewWorkspaceScopedSandbox(execCfg(dir, &ExecConfig{
		NonInteractiveEnv: map[string]string{"SPORE_SC12_ENV": "hardened"},
	}))
	out, v := sb.ExecuteCommand(context.Background(), "/bin/sh", []string{"-c", "echo $SPORE_SC12_ENV"}, "", 0)
	if v != nil {
		t.Fatalf("violation: %v", v)
	}
	if out.ExitCode != 0 {
		t.Fatalf("exit code %d, want 0", out.ExitCode)
	}
	if !strings.Contains(out.Stdout, "hardened") {
		t.Fatalf("env var not injected; stdout=%q", out.Stdout)
	}
}

func TestExecConfigCloseStdinYieldsEOF(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewWorkspaceScopedSandbox(execCfg(dir, &ExecConfig{CloseStdin: true}))
	// `cat` with no args reads stdin to EOF; with stdin closed it returns
	// immediately with exit 0 and empty output. A generous per-call timeout
	// guards against a hang if the knob regressed.
	out, v := sb.ExecuteCommand(context.Background(), "cat", nil, "", 5*time.Second)
	if v != nil {
		t.Fatalf("violation: %v", v)
	}
	if out.TimedOut {
		t.Fatalf("cat hung — stdin was not closed")
	}
	if out.ExitCode != 0 {
		t.Fatalf("exit code %d, want 0", out.ExitCode)
	}
	if out.Stdout != "" {
		t.Fatalf("expected empty stdout, got %q", out.Stdout)
	}
}

func TestExecConfigKillOnDropReapsChildOnTimeout(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	sb, _ := NewWorkspaceScopedSandbox(execCfg(root, &ExecConfig{KillOnDrop: true}))
	sentinel := filepath.Join(root, "kod_sentinel")
	// The shell sleeps, then would `touch` the sentinel. The 100ms timeout
	// cancels the context; exec.CommandContext kills the shell (Go's native
	// kill-on-drop) before it can run `touch`, so the sentinel never appears.
	script := "sleep 1; touch " + sentinel
	out, v := sb.ExecuteCommand(context.Background(), "/bin/sh", []string{"-c", script}, "", 100*time.Millisecond)
	if v != nil {
		t.Fatalf("violation: %v", v)
	}
	if !out.TimedOut {
		t.Fatalf("expected timed out, got %+v", out)
	}
	// Wait past when the un-killed shell would have created the sentinel.
	time.Sleep(1500 * time.Millisecond)
	if _, err := os.Stat(sentinel); err == nil {
		t.Fatalf("child was not reaped on drop — sentinel was created")
	}
}

func TestExecConfigUnsetIsLegacyBehavior(t *testing.T) {
	dir := t.TempDir()
	// Nil ExecConfig: no implicit timeout, inherited env, inherited stdin.
	sb, _ := NewWorkspaceScopedSandbox(cfg(dir))
	out, v := sb.ExecuteCommand(context.Background(), "echo", []string{"legacy"}, "", 0)
	if v != nil {
		t.Fatalf("violation: %v", v)
	}
	if out.ExitCode != 0 || !strings.Contains(out.Stdout, "legacy") {
		t.Fatalf("unexpected legacy output: %+v", out)
	}
}

func TestSandboxHandleLargeOutputBelowThreshold(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewWorkspaceScopedSandbox(cfg(dir))
	out := sb.HandleLargeOutput(context.Background(), "short content", "c1", 100, 100)
	if out.Truncated || out.FullRef != nil || out.Content != "short content" {
		t.Fatalf("unexpected: %+v", out)
	}
}

func TestSandboxHandleLargeOutputAboveThresholdOffloads(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewWorkspaceScopedSandbox(cfg(dir))
	content := strings.Repeat("x", 10000)
	out := sb.HandleLargeOutput(context.Background(), content, "call-1", 10, 10)
	if !out.Truncated {
		t.Fatalf("expected truncated")
	}
	if !strings.Contains(out.Content, "...[truncated]...") {
		t.Fatalf("missing truncated marker: %q", out.Content[:80])
	}
	if out.FullRef == nil {
		t.Fatalf("expected offloaded ref")
	}
	if out.FullRef.ByteLen != uint64(len(content)) {
		t.Fatalf("byte len: %d vs %d", out.FullRef.ByteLen, len(content))
	}
	if !strings.Contains(out.FullRef.Path, ".spore") {
		t.Fatalf("path %s does not contain .spore", out.FullRef.Path)
	}
	written, err := os.ReadFile(out.FullRef.Path)
	if err != nil {
		t.Fatalf("read offload: %v", err)
	}
	if len(written) != len(content) {
		t.Fatalf("len mismatch")
	}
}
