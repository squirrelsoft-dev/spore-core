package tools

import (
	"context"
	"encoding/json"
	"os/exec"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func gitAvailable() bool {
	_, err := exec.LookPath("git")
	return err == nil
}

func TestGitStatusRuns(t *testing.T) {
	if !gitAvailable() {
		t.Skip("git not available")
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewGitStatusTool().Execute(context.Background(),
		call("git_status", "c1", map[string]any{}), sb)
	// Either Success (inside a repo) or Error (outside one) — both fine.
	if r.Kind != sporecore.ToolOutputSuccess && r.Kind != sporecore.ToolOutputError {
		t.Fatalf("%+v", r)
	}
}

func TestGitResetModeRoundtripsSnakeCase(t *testing.T) {
	b, _ := json.Marshal(GitResetHard)
	if string(b) != `"hard"` {
		t.Fatalf("got %s", b)
	}
	var m GitResetMode
	if err := json.Unmarshal([]byte(`"hard"`), &m); err != nil {
		t.Fatal(err)
	}
	if m != GitResetHard {
		t.Fatalf("got %v", m)
	}
}

func TestGitLogParamsDefaults(t *testing.T) {
	var p GitLogParams
	if err := json.Unmarshal([]byte(`{}`), &p); err != nil {
		t.Fatal(err)
	}
	if p.N != 20 || p.Format != "oneline" {
		t.Fatalf("defaults wrong: %+v", p)
	}
}
