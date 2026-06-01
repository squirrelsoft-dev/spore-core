//go:build dangerous

package sporecore

import "testing"

// These tests exercise the gated safety footguns (issue #34). They compile and
// run ONLY under `go test -tags dangerous ./...`; the default suite never sees
// them because IsolationNone / ModeYolo do not exist without the build tag.

// TestDangerousIsolationNoneAvailable confirms IsolationNone is constructable
// under the dangerous build, satisfies the sealed IsolationMode interface, and
// reports the wire tag "none".
func TestDangerousIsolationNoneAvailable(t *testing.T) {
	var mode IsolationMode = IsolationNone{}
	if mode.Kind() != "none" {
		t.Fatalf("IsolationNone.Kind() = %q, want %q", mode.Kind(), "none")
	}
}

// TestDangerousSandboxBuildNoneIsolation confirms a WorkspaceScopedSandbox can
// be built with IsolationNone under the dangerous build and reports it back.
func TestDangerousSandboxBuildNoneIsolation(t *testing.T) {
	dir := t.TempDir()
	sb, err := NewWorkspaceScopedSandboxWithMode(WorkspaceConfig{Root: dir}, IsolationNone{})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if _, ok := sb.IsolationMode().(IsolationNone); !ok {
		t.Fatalf("expected IsolationNone, got %T", sb.IsolationMode())
	}
}

// TestDangerousModeYoloWireTag confirms the gated ModeYolo SwitchMode target
// keeps the "yolo" wire tag.
func TestDangerousModeYoloWireTag(t *testing.T) {
	if ModeYolo != Mode("yolo") {
		t.Fatalf("ModeYolo = %q, want %q", ModeYolo, "yolo")
	}
}
