package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ---------------- ExecTool (shell-free) ----------------

func TestExecEchoRunsAndReturnsStdout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewExecTool().Execute(context.Background(),
		call("exec", "c1", map[string]any{"command": "echo", "args": []string{"hi"}}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("%+v", r)
	}
	if !strings.Contains(r.Content, "hi") {
		t.Fatalf("expected 'hi' in %q", r.Content)
	}
}

// exec must NOT interpret shell syntax: pipe/$(...)/redirect tokens are passed
// to echo as literal arguments, and no file is created.
func TestExecHasNoShellSemantics(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	dir := filepath.Join(os.TempDir(), fmt.Sprintf("spore-exec-noshell-%d", time.Now().UnixNano()))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	prev, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(prev)

	sb := sporecore.AllowAllSandbox{}
	r := NewExecTool().Execute(context.Background(),
		call("exec", "c1", map[string]any{"command": "echo", "args": []string{"a|b", "$(whoami)", ">out"}}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("%+v", r)
	}
	if !strings.Contains(r.Content, "a|b $(whoami) >out") {
		t.Fatalf("args must be literal, got %q", r.Content)
	}
	if _, err := os.Stat(filepath.Join(dir, "out")); err == nil {
		t.Fatal("no redirect: `out` must not be created")
	}
}

func TestExecNonzeroExitIsRecoverable(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewExecTool().Execute(context.Background(),
		call("exec", "c1", map[string]any{"command": "false"}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("%+v", r)
	}
}

func TestExecTimeoutRecoverable(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewExecTool().Execute(context.Background(),
		call("exec", "c1", map[string]any{"command": "sleep", "args": []string{"5"}, "timeout": 1}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
	if !strings.Contains(r.Message, "timed out") {
		t.Fatalf("expected timeout message, got %q", r.Message)
	}
}

func TestExecInvalidParams(t *testing.T) {
	sb := sporecore.AllowAllSandbox{}
	r := NewExecTool().Execute(context.Background(),
		sporecore.ToolCall{ID: "c1", Name: "exec", Input: json.RawMessage(`{}`)}, sb, nil)
	// SC-15: an empty command can't be spawned, so the sandbox returns a typed
	// SandboxExecSpawnFailed violation. The tool carries it to the harness as a
	// ToolOutputSandboxViolation (issue #150) — the harness applies its
	// SandboxViolationPolicy (recoverable feedback by default), rather than the
	// tool flattening the former fake exit_code -1 into a plain Error here.
	if r.Kind != sporecore.ToolOutputSandboxViolation {
		t.Fatalf("expected ToolOutputSandboxViolation, got %+v", r)
	}
	if r.Violation == nil || r.Violation.Kind != sporecore.SandboxExecSpawnFailed {
		t.Fatalf("expected SandboxExecSpawnFailed violation, got %+v", r.Violation)
	}
}

// ---------------- BashCommandTool (real shell) ----------------

func TestBashCommandSupportsPipeline(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewBashCommandTool().Execute(context.Background(),
		call("bash_command", "c1", map[string]any{"script": "printf 'hi' | tr a-z A-Z"}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("%+v", r)
	}
	if r.Content != "HI" {
		t.Fatalf("expected pipeline output 'HI', got %q", r.Content)
	}
}

func TestBashCommandSupportsRedirect(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	sb := sporecore.AllowAllSandbox{}
	tmp := filepath.Join(os.TempDir(), fmt.Sprintf("spore-bash-redirect-%d.txt", time.Now().UnixNano()))
	defer os.Remove(tmp)
	script := fmt.Sprintf("printf 'data' > %s", tmp)
	r := NewBashCommandTool().Execute(context.Background(),
		call("bash_command", "c1", map[string]any{"script": script}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("%+v", r)
	}
	got, err := os.ReadFile(tmp)
	if err != nil {
		t.Fatalf("redirect did not create file: %v", err)
	}
	if string(got) != "data" {
		t.Fatalf("redirect wrote %q, want %q", string(got), "data")
	}
}

func TestBashCommandNonzeroExitIsRecoverable(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewBashCommandTool().Execute(context.Background(),
		call("bash_command", "c1", map[string]any{"script": "exit 3"}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("%+v", r)
	}
}

func TestBashCommandTimeoutRecoverable(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewBashCommandTool().Execute(context.Background(),
		call("bash_command", "c1", map[string]any{"script": "sleep 5", "timeout": 1}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
	if !strings.Contains(r.Message, "timed out") {
		t.Fatalf("expected timeout message, got %q", r.Message)
	}
}

func TestBashCommandInvalidParams(t *testing.T) {
	sb := sporecore.AllowAllSandbox{}
	r := NewBashCommandTool().Execute(context.Background(),
		sporecore.ToolCall{ID: "c1", Name: "bash_command", Input: json.RawMessage(`{}`)}, sb, nil)
	// Missing script -> empty script via /bin/sh -c "" exits 0 with no output;
	// guard only that it does not panic and returns a defined output kind.
	if r.Kind != sporecore.ToolOutputSuccess && r.Kind != sporecore.ToolOutputError {
		t.Fatalf("%+v", r)
	}
}

func TestBashCommandLargeStderrIsTruncated(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	sb := sporecore.AllowAllSandbox{}
	// awk writes 10 KB to stderr and exits non-zero; verify elision in message.
	r := NewBashCommandTool().Execute(context.Background(),
		call("bash_command", "c1", map[string]any{
			"script": `awk 'BEGIN{for(i=0;i<10240;i++)printf "x" > "/dev/stderr"; exit 1}'`,
		}), sb, nil)
	if r.Kind != sporecore.ToolOutputError {
		t.Fatalf("expected error, got %+v", r)
	}
	if !strings.Contains(r.Message, "bytes elided") {
		t.Fatalf("expected elision marker in message, got %q", r.Message)
	}
	if len(r.Message) >= 10*1024 {
		t.Fatalf("message too long: %d bytes", len(r.Message))
	}
}

// ---------------- truncateForMessage unit tests ----------------

func TestTruncateForMessagePassthroughShort(t *testing.T) {
	s := "small error output"
	if got := truncateForMessage(s); got != s {
		t.Fatalf("expected passthrough, got %q", got)
	}
}

func TestTruncateForMessageElidesMiddleWhenLarge(t *testing.T) {
	long := strings.Repeat("x", 10*1024)
	result := truncateForMessage(long)
	if !strings.Contains(result, "bytes elided") {
		t.Fatalf("expected elision marker, got %q", result)
	}
	if len(result) >= 8*1024 {
		t.Fatalf("message too long: %d bytes", len(result))
	}
}
