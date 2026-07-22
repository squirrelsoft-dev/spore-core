package sporecore

import (
	"context"
	"strings"
	"testing"
)

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
