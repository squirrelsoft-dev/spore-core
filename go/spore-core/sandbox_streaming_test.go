package sporecore

import (
	"context"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestSanitizeCommandOutputStripsAnsiAndCollapsesCR(t *testing.T) {
	got := sanitizeCommandOutput("\x1b[31mred\x1b[0m\nprogress 10%\rprogress 99%\rdone\n")
	want := "red\ndone\n"
	if got != want {
		t.Fatalf("sanitize=%q, want %q", got, want)
	}
}

func TestExecuteCommandSanitizesFinalAndProgressOutput(t *testing.T) {
	dir := t.TempDir()
	sb, err := NewWorkspaceScopedSandbox(cfg(dir))
	if err != nil {
		t.Fatal(err)
	}
	var previews []string
	out, v := sb.ExecuteCommandStreaming(context.Background(), "/bin/sh", []string{"-c", "printf '\\033[31mred\\033[0m\\n'; printf 'a\\rb\\n'; sleep 0.2"}, "", 0, func(p string) {
		previews = append(previews, p)
	})
	if v != nil {
		t.Fatalf("violation: %+v", v)
	}
	if out.Stdout != "red\nb\n" {
		t.Fatalf("stdout=%q, want sanitized red/b", out.Stdout)
	}
	for _, p := range previews {
		if strings.Contains(p, "\x1b") || strings.Contains(p, "\r") {
			t.Fatalf("preview not sanitized: %#v", previews)
		}
	}
}

func TestExecuteCommandLinuxCapturesBackgroundChildOutputAfterDirectExit(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux pipe ownership regression")
	}
	dir := t.TempDir()
	sb, err := NewWorkspaceScopedSandbox(cfg(dir))
	if err != nil {
		t.Fatal(err)
	}
	out, v := sb.ExecuteCommand(context.Background(), "/bin/sh", []string{"-c", "(sleep 0.1; echo late) & echo early"}, "", 0)
	if v != nil {
		t.Fatalf("violation: %+v", v)
	}
	stdout := strings.TrimSpace(out.Stdout)
	if !strings.Contains(stdout, "early") || !strings.Contains(stdout, "late") {
		t.Fatalf("stdout=%q, want early and late background-child output", out.Stdout)
	}
}

func TestExecuteCommandLinuxCapturesInjectedEnvOutput(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux pipe ownership regression")
	}
	dir := t.TempDir()
	cfg := cfg(dir)
	cfg.ExecConfig = &ExecConfig{NonInteractiveEnv: map[string]string{"SPORE_PIPE_ENV": "captured"}}
	sb, err := NewWorkspaceScopedSandbox(cfg)
	if err != nil {
		t.Fatal(err)
	}
	out, v := sb.ExecuteCommand(context.Background(), "/bin/sh", []string{"-c", "echo $SPORE_PIPE_ENV"}, "", 0)
	if v != nil {
		t.Fatalf("violation: %+v", v)
	}
	if strings.TrimSpace(out.Stdout) != "captured" {
		t.Fatalf("stdout=%q, want injected env output", out.Stdout)
	}
}

func TestExecuteCommandBackgroundPipeDoesNotHang(t *testing.T) {
	dir := t.TempDir()
	sb, err := NewWorkspaceScopedSandbox(cfg(dir))
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	out, v := sb.ExecuteCommand(context.Background(), "/bin/sh", []string{"-c", "sleep 1 & echo done"}, "", 0)
	if v != nil {
		t.Fatalf("violation: %+v", v)
	}
	if time.Since(start) > 2*time.Second {
		t.Fatalf("background pipe held command open for %s", time.Since(start))
	}
	if strings.TrimSpace(out.Stdout) != "done" {
		t.Fatalf("stdout=%q, want done", out.Stdout)
	}
}

func TestExecuteCommandStreamingEmitsTailPreviews(t *testing.T) {
	dir := t.TempDir()
	sb, err := NewWorkspaceScopedSandbox(cfg(dir))
	if err != nil {
		t.Fatal(err)
	}
	var previews []string
	out, v := sb.ExecuteCommandStreaming(
		context.Background(),
		"/bin/sh",
		[]string{"-c", "echo one; sleep 0.35; echo two"},
		"",
		0,
		func(p string) { previews = append(previews, p) },
	)
	if v != nil {
		t.Fatalf("violation: %+v", v)
	}
	if out.ExitCode != 0 {
		t.Fatalf("exit=%d stderr=%q", out.ExitCode, out.Stderr)
	}
	if !strings.Contains(out.Stdout, "one") || !strings.Contains(out.Stdout, "two") {
		t.Fatalf("final stdout = %q, want both one and two", out.Stdout)
	}
	if len(previews) == 0 {
		t.Fatal("no live previews emitted")
	}
	if !strings.Contains(previews[0], "one") {
		t.Fatalf("first preview missing early output: %#v", previews)
	}
	if strings.Contains(previews[0], "two") {
		t.Fatalf("first preview already has late output (not live): %#v", previews)
	}
}
