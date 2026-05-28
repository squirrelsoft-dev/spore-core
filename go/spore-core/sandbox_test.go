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

func TestSandboxBuildNoneIsolationWarns(t *testing.T) {
	dir := t.TempDir()
	sb, err := NewWorkspaceScopedSandboxWithMode(cfg(dir), IsolationNone{})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if _, ok := sb.IsolationMode().(IsolationNone); !ok {
		t.Fatalf("expected IsolationNone, got %T", sb.IsolationMode())
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

func TestSandboxViolationsRoundTripJSON(t *testing.T) {
	cases := []SandboxViolation{
		{Kind: SandboxPathEscape, Path: "/p"},
		{Kind: SandboxPathDenied, Path: "/p", MatchedRule: "r"},
		{Kind: SandboxExtensionDenied, Path: "/p.env", Extension: "env"},
		{Kind: SandboxReadOnly, Path: "/p"},
		{Kind: SandboxFileSizeExceeded, Path: "/p", Size: 1024, Limit: 100},
		{Kind: SandboxDisallowedCommand, Command: "rm"},
		{Kind: SandboxNetworkViolation, Host: "evil"},
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
