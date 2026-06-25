package tools

import (
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func TestInvalidParametersIsRecoverable(t *testing.T) {
	e := InvalidParameters("missing path")
	out := e.ToToolOutput()
	if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
		t.Fatalf("%+v", out)
	}
}

func TestExecutionFailedPassesThroughFlag(t *testing.T) {
	out := ExecutionFailed("x", false).ToToolOutput()
	if out.Recoverable {
		t.Fatalf("expected not recoverable")
	}
}

func TestSandboxViolationCarriesTypedViolation(t *testing.T) {
	// issue #150: the conversion does NOT pre-decide recoverability — it carries
	// the typed violation through as ToolOutputSandboxViolation so the harness can
	// apply its SandboxViolationPolicy (recoverable by default; halt on opt-in).
	v := &sporecore.SandboxViolation{Kind: sporecore.SandboxPathEscape, Path: "/etc"}
	out := SandboxViolationError(v).ToToolOutput()
	if out.Kind != sporecore.ToolOutputSandboxViolation {
		t.Fatalf("expected ToolOutputSandboxViolation, got %q", out.Kind)
	}
	if out.Violation == nil || out.Violation.Kind != sporecore.SandboxPathEscape || out.Violation.Path != "/etc" {
		t.Fatalf("expected typed PathEscape violation carried through, got %+v", out.Violation)
	}
}

func TestTimeoutIsRecoverable(t *testing.T) {
	out := TimeoutError(5).ToToolOutput()
	if !out.Recoverable {
		t.Fatalf("expected recoverable")
	}
}
