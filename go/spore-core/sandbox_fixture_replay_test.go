package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type sandboxFsEntry struct {
	Dir  bool   `json:"dir,omitempty"`
	File string `json:"file,omitempty"`
}

type sandboxScenario struct {
	Name             string                    `json:"name"`
	Filesystem       map[string]sandboxFsEntry `json:"filesystem"`
	RawPath          string                    `json:"raw_path"`
	AllowedPaths     []string                  `json:"allowed_paths"`
	DeniedPaths      []string                  `json:"denied_paths"`
	DeniedExtensions []string                  `json:"denied_extensions"`
	ReadOnly         bool                      `json:"read_only"`
	MaxFileSize      uint64                    `json:"max_file_size"`
	Operation        string                    `json:"operation"`
	Expected         struct {
		Kind string `json:"kind"`
	} `json:"expected"`
}

func sandboxFixturesDir(t *testing.T) string {
	t.Helper()
	// Walk up to the repo root from this package directory.
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for i := 0; i < 8; i++ {
		candidate := filepath.Join(dir, "fixtures", "sandbox_violations")
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Skip("sandbox_violations fixtures not found")
	return ""
}

func parseOp(s string) Operation {
	switch s {
	case "read":
		return OperationRead
	case "write":
		return OperationWrite
	case "execute":
		return OperationExecute
	default:
		return OperationRead
	}
}

func TestSandboxViolationFixtures(t *testing.T) {
	dir := sandboxFixturesDir(t)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	ran := 0
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("read %s: %v", e.Name(), err)
		}
		var sc sandboxScenario
		if err := json.Unmarshal(raw, &sc); err != nil {
			t.Fatalf("parse %s: %v", e.Name(), err)
		}

		tmp := t.TempDir()
		root, err := filepath.EvalSymlinks(tmp)
		if err != nil {
			t.Fatalf("evalsymlinks: %v", err)
		}
		for rel, entry := range sc.Filesystem {
			target := filepath.Join(root, rel)
			if entry.Dir {
				if err := os.MkdirAll(target, 0o755); err != nil {
					t.Fatalf("mkdir %s: %v", target, err)
				}
			} else {
				if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
					t.Fatalf("mkdir parent %s: %v", target, err)
				}
				if err := os.WriteFile(target, []byte(entry.File), 0o644); err != nil {
					t.Fatalf("write %s: %v", target, err)
				}
			}
		}

		cfg := WorkspaceConfig{
			Root:             root,
			AllowedPaths:     sc.AllowedPaths,
			DeniedPaths:      sc.DeniedPaths,
			DeniedExtensions: sc.DeniedExtensions,
			ReadOnly:         sc.ReadOnly,
			MaxFileSize:      sc.MaxFileSize,
		}
		sb, err := NewWorkspaceScopedSandbox(cfg)
		if err != nil {
			t.Fatalf("%s: build: %v", sc.Name, err)
		}
		_, v := sb.ResolvePath(context.Background(), sc.RawPath, parseOp(sc.Operation))
		var actual string
		if v == nil {
			actual = "ok"
		} else {
			actual = string(v.Kind)
		}
		if actual != sc.Expected.Kind {
			t.Errorf("%s: expected kind=%s got kind=%s", sc.Name, sc.Expected.Kind, actual)
		}
		ran++
	}
	if ran == 0 {
		t.Fatalf("expected at least one fixture")
	}
}
