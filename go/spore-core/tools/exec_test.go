package tools

import (
	"context"
	"encoding/json"
	"runtime"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func TestBashEchoRunsAndReturnsStdout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewBashCommandTool().Execute(context.Background(),
		call("bash_command", "c1", map[string]any{"command": "echo", "args": []string{"hi"}}), sb)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("%+v", r)
	}
	if !strings.Contains(r.Content, "hi") {
		t.Fatalf("expected 'hi' in %q", r.Content)
	}
}

func TestBashNonzeroExitIsRecoverable(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewBashCommandTool().Execute(context.Background(),
		call("bash_command", "c1", map[string]any{"command": "false"}), sb)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("%+v", r)
	}
}

func TestBashTimeoutRecoverable(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewBashCommandTool().Execute(context.Background(),
		call("bash_command", "c1", map[string]any{"command": "sleep", "args": []string{"5"}, "timeout": 1}), sb)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
	if !strings.Contains(r.Message, "timed out") {
		t.Fatalf("expected timeout message, got %q", r.Message)
	}
}

func TestBashInvalidParams(t *testing.T) {
	sb := sporecore.AllowAllSandbox{}
	r := NewBashCommandTool().Execute(context.Background(),
		sporecore.ToolCall{ID: "c1", Name: "bash_command", Input: json.RawMessage(`{}`)}, sb)
	// Empty command produces an exec error from the sandbox -> recoverable error.
	if r.Kind != sporecore.ToolOutputError {
		t.Fatalf("%+v", r)
	}
}
