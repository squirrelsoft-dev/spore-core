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

func TestSandboxViolationNotRecoverable(t *testing.T) {
	v := &sporecore.SandboxViolation{Kind: sporecore.SandboxPathEscape, Path: "/etc"}
	out := SandboxViolationError(v).ToToolOutput()
	if out.Recoverable {
		t.Fatalf("expected not recoverable")
	}
}

func TestTimeoutIsRecoverable(t *testing.T) {
	out := TimeoutError(5).ToToolOutput()
	if !out.Recoverable {
		t.Fatalf("expected recoverable")
	}
}
