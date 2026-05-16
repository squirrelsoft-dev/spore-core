// Canonical SandboxProvider implementation — issue #6.
//
// WorkspaceScopedSandbox enforces a workspace root with allow/deny lists,
// extension filters, a read-only mode, and per-file size limits. It runs
// subprocesses directly via os/exec.CommandContext and offloads large
// outputs to {workspace_root}/.spore/offload/{call_id}.txt.

package sporecore

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ============================================================================
// BuildError
// ============================================================================

// BuildErrorKind discriminates BuildError variants.
type BuildErrorKind string

const (
	BuildErrRootNotFound BuildErrorKind = "root_not_found"
	BuildErrRootIO       BuildErrorKind = "root_io"
)

// BuildError is the construction-time error type for WorkspaceScopedSandbox.
type BuildError struct {
	Kind BuildErrorKind
	Path string
	Err  error
}

func (e *BuildError) Error() string {
	switch e.Kind {
	case BuildErrRootNotFound:
		return fmt.Sprintf("workspace root does not exist: %s", e.Path)
	case BuildErrRootIO:
		return fmt.Sprintf("workspace root io error: %s: %s", e.Path, e.Err)
	default:
		return fmt.Sprintf("workspace build error: %s", e.Path)
	}
}

func (e *BuildError) Unwrap() error { return e.Err }

// ============================================================================
// WorkspaceScopedSandbox
// ============================================================================

// WorkspaceScopedSandbox is the canonical path-enforcing sandbox.
type WorkspaceScopedSandbox struct {
	config        WorkspaceConfig // .Root is the canonical (abs + symlink-resolved) root
	isolationMode IsolationMode
}

// NewWorkspaceScopedSandbox builds a sandbox with IsolationWorkspaceScoped.
func NewWorkspaceScopedSandbox(cfg WorkspaceConfig) (*WorkspaceScopedSandbox, error) {
	return NewWorkspaceScopedSandboxWithMode(cfg, IsolationWorkspaceScoped{})
}

// NewWorkspaceScopedSandboxWithMode builds a sandbox with an explicit
// isolation mode. IsolationNone emits a warning via log.Printf — it must
// never be enabled silently in production.
func NewWorkspaceScopedSandboxWithMode(cfg WorkspaceConfig, mode IsolationMode) (*WorkspaceScopedSandbox, error) {
	if cfg.Root == "" {
		return nil, &BuildError{Kind: BuildErrRootNotFound, Path: cfg.Root}
	}
	abs, err := filepath.Abs(cfg.Root)
	if err != nil {
		return nil, &BuildError{Kind: BuildErrRootIO, Path: cfg.Root, Err: err}
	}
	canonical, err := filepath.EvalSymlinks(abs)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, &BuildError{Kind: BuildErrRootNotFound, Path: cfg.Root}
		}
		return nil, &BuildError{Kind: BuildErrRootIO, Path: cfg.Root, Err: err}
	}
	info, err := os.Stat(canonical)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, &BuildError{Kind: BuildErrRootNotFound, Path: cfg.Root}
		}
		return nil, &BuildError{Kind: BuildErrRootIO, Path: cfg.Root, Err: err}
	}
	if !info.IsDir() {
		return nil, &BuildError{Kind: BuildErrRootIO, Path: cfg.Root, Err: fmt.Errorf("not a directory")}
	}
	cfg.Root = canonical

	if _, isNone := mode.(IsolationNone); isNone {
		log.Printf("spore-core: WorkspaceScopedSandbox constructed with IsolationNone — " +
			"trusted-dev use only; do not enable silently in production")
	}
	return &WorkspaceScopedSandbox{config: cfg, isolationMode: mode}, nil
}

// Config returns the (canonicalized) workspace config.
func (s *WorkspaceScopedSandbox) Config() WorkspaceConfig { return s.config }

// IsolationMode returns the active isolation mode.
func (s *WorkspaceScopedSandbox) IsolationMode() IsolationMode { return s.isolationMode }

// WorkspaceRoot returns the canonical workspace root.
func (s *WorkspaceScopedSandbox) WorkspaceRoot() string { return s.config.Root }

// Validate accepts every call — path resolution / policy enforcement happens
// in ResolvePath and ExecuteCommand.
func (s *WorkspaceScopedSandbox) Validate(context.Context, ToolCall) *SandboxViolation {
	return nil
}

// ============================================================================
// Path resolution — eight steps as specified.
// ============================================================================

// ResolvePath canonicalizes path against the workspace root and validates
// against allow/deny/extension/read-only policy for the given operation.
func (s *WorkspaceScopedSandbox) ResolvePath(_ context.Context, raw string, op Operation) (string, *SandboxViolation) {
	// 1. Join raw onto root, treating absolute paths as relative.
	rawPath := raw
	if filepath.IsAbs(rawPath) {
		// Strip a single leading separator so absolute callers don't
		// silently escape — `/foo` becomes `<root>/foo`.
		rawPath = strings.TrimPrefix(rawPath, string(filepath.Separator))
	}
	joined := filepath.Join(s.config.Root, rawPath)

	// 2. Canonicalize. For Write/Execute on a missing target, canonicalize
	// the parent and re-join the leaf.
	canonical, err := canonicalize(joined)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) && (op == OperationWrite || op == OperationExecute) {
			parent := filepath.Dir(joined)
			leaf := filepath.Base(joined)
			cp, parentErr := canonicalize(parent)
			if parentErr != nil {
				return "", &SandboxViolation{Kind: SandboxPathEscape, Path: raw}
			}
			canonical = filepath.Join(cp, leaf)
		} else {
			return "", &SandboxViolation{Kind: SandboxPathEscape, Path: raw}
		}
	}

	// 3. Boundary check.
	if !pathHasPrefix(canonical, s.config.Root) {
		return "", &SandboxViolation{Kind: SandboxPathEscape, Path: canonical}
	}

	// 4. Denylist (relative to root).
	for _, denied := range s.config.DeniedPaths {
		deniedAbs := filepath.Join(s.config.Root, denied)
		if pathHasPrefix(canonical, deniedAbs) {
			return "", &SandboxViolation{
				Kind:        SandboxPathDenied,
				Path:        canonical,
				MatchedRule: denied,
			}
		}
	}

	// 5. Allowlist (if non-empty).
	if len(s.config.AllowedPaths) > 0 {
		ok := false
		for _, allowed := range s.config.AllowedPaths {
			allowedAbs := filepath.Join(s.config.Root, allowed)
			if pathHasPrefix(canonical, allowedAbs) {
				ok = true
				break
			}
		}
		if !ok {
			return "", &SandboxViolation{
				Kind:        SandboxPathDenied,
				Path:        canonical,
				MatchedRule: "not in allowlist",
			}
		}
	}

	// 6. Denied extensions. Match either the path's extension or a dotfile
	// whose name (minus the leading dot) equals the denied extension.
	extFromExt := ""
	if e := filepath.Ext(canonical); e != "" {
		extFromExt = strings.ToLower(strings.TrimPrefix(e, "."))
	}
	base := filepath.Base(canonical)
	extFromDotfile := ""
	if strings.HasPrefix(base, ".") && !strings.Contains(base[1:], ".") {
		extFromDotfile = strings.ToLower(base[1:])
	}
	for _, deniedExt := range s.config.DeniedExtensions {
		trimmed := strings.ToLower(strings.TrimPrefix(deniedExt, "."))
		if trimmed == "" {
			continue
		}
		if extFromExt == trimmed || extFromDotfile == trimmed {
			return "", &SandboxViolation{
				Kind:      SandboxExtensionDenied,
				Path:      canonical,
				Extension: trimmed,
			}
		}
	}

	// 7. Read-only check.
	if s.config.ReadOnly && (op == OperationWrite || op == OperationExecute) {
		return "", &SandboxViolation{Kind: SandboxReadOnly, Path: canonical}
	}

	// 8. File-size cap (read only; writes are sized by content length upstream).
	if op == OperationRead && s.config.MaxFileSize > 0 {
		if info, err := os.Stat(canonical); err == nil && !info.IsDir() {
			if uint64(info.Size()) > s.config.MaxFileSize {
				return "", &SandboxViolation{
					Kind:  SandboxFileSizeExceeded,
					Path:  canonical,
					Size:  uint64(info.Size()),
					Limit: s.config.MaxFileSize,
				}
			}
		}
	}

	return canonical, nil
}

// canonicalize resolves a path to an absolute, symlink-free form. Returns
// the underlying os error (so callers can test for os.ErrNotExist) without
// wrapping.
func canonicalize(p string) (string, error) {
	abs, err := filepath.Abs(p)
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(abs)
}

// pathHasPrefix reports whether path is rooted at prefix. It is component-
// aware so /foo/bar is not considered a prefix of /foo/barbaz.
func pathHasPrefix(path, prefix string) bool {
	path = filepath.Clean(path)
	prefix = filepath.Clean(prefix)
	if path == prefix {
		return true
	}
	sep := string(filepath.Separator)
	if !strings.HasSuffix(prefix, sep) {
		prefix += sep
	}
	return strings.HasPrefix(path+sep, prefix)
}

// ============================================================================
// ExecuteCommand
// ============================================================================

// ExecuteCommand runs a subprocess inside the configured isolation mode.
// Bubblewrap/Docker modes return DisallowedCommand until those backends are
// wired up.
func (s *WorkspaceScopedSandbox) ExecuteCommand(
	ctx context.Context,
	command string,
	args []string,
	workingDir string,
	timeout time.Duration,
) (CommandOutput, *SandboxViolation) {
	switch s.isolationMode.(type) {
	case IsolationNone, IsolationWorkspaceScoped:
		// proceed directly
	case IsolationBubblewrap:
		return CommandOutput{}, &SandboxViolation{
			Kind:    SandboxDisallowedCommand,
			Command: fmt.Sprintf("bubblewrap isolation not implemented: %s", command),
		}
	case IsolationDocker:
		return CommandOutput{}, &SandboxViolation{
			Kind:    SandboxDisallowedCommand,
			Command: fmt.Sprintf("docker isolation not implemented: %s", command),
		}
	}

	runCtx := ctx
	var cancel context.CancelFunc
	if timeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	cmd := exec.CommandContext(runCtx, command, args...)
	if workingDir != "" {
		cmd.Dir = workingDir
	} else {
		cmd.Dir = s.config.Root
	}

	stdout, err := cmd.Output()
	stderr := ""
	if ee, ok := err.(*exec.ExitError); ok {
		stderr = string(ee.Stderr)
	}
	exitCode := 0
	if err != nil {
		if runCtx.Err() == context.DeadlineExceeded {
			return CommandOutput{
				Stdout:   "",
				Stderr:   fmt.Sprintf("command timed out after %s", timeout),
				ExitCode: -1,
				TimedOut: true,
			}, nil
		}
		if ee, ok := err.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			return CommandOutput{
				Stdout:   "",
				Stderr:   fmt.Sprintf("spawn failed: %s", err),
				ExitCode: -1,
			}, nil
		}
	}
	return CommandOutput{
		Stdout:   string(stdout),
		Stderr:   stderr,
		ExitCode: exitCode,
	}, nil
}

// ============================================================================
// HandleLargeOutput
// ============================================================================

// HandleLargeOutput head+tail-truncates content and offloads the original
// to {workspace_root}/.spore/offload/{call_id}.txt.
func (s *WorkspaceScopedSandbox) HandleLargeOutput(
	_ context.Context,
	content string,
	callID string,
	headTokens uint32,
	tailTokens uint32,
) TruncatedOutput {
	headChars := int(headTokens) * 4
	tailChars := int(tailTokens) * 4
	runes := []rune(content)
	originalSize := uint64(len(content))
	if len(runes) <= headChars+tailChars {
		return TruncatedOutput{Content: content, Truncated: false, OriginalSize: originalSize}
	}
	head := string(runes[:headChars])
	tail := string(runes[len(runes)-tailChars:])
	snippet := fmt.Sprintf("%s\n...[truncated]...\n%s", head, tail)

	// Offload the full content.
	var fullRef *FileRef
	offloadDir := filepath.Join(s.config.Root, ".spore", "offload")
	if err := os.MkdirAll(offloadDir, 0o755); err == nil {
		safeID := sanitizeCallID(callID)
		offloadPath := filepath.Join(offloadDir, safeID+".txt")
		if err := os.WriteFile(offloadPath, []byte(content), 0o644); err == nil {
			fullRef = &FileRef{Path: offloadPath, ByteLen: originalSize}
		}
	}

	return TruncatedOutput{
		Content:      snippet,
		Truncated:    true,
		FullRef:      fullRef,
		OriginalSize: originalSize,
	}
}

func sanitizeCallID(id string) string {
	var b strings.Builder
	b.Grow(len(id))
	for _, c := range id {
		switch {
		case c >= '0' && c <= '9', c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c == '-', c == '_':
			b.WriteRune(c)
		default:
			b.WriteRune('_')
		}
	}
	return b.String()
}

// Compile-time check.
var _ SandboxProvider = (*WorkspaceScopedSandbox)(nil)
