// Canonical SandboxProvider implementation — issue #6.
//
// WorkspaceScopedSandbox enforces a workspace root with allow/deny lists,
// extension filters, a read-only mode, and per-file size limits. It runs
// subprocesses directly via os/exec.CommandContext and offloads large
// outputs to {workspace_root}/.spore/offload/{call_id}.txt.

package sporecore

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ============================================================================
// BuildError
// ============================================================================

// BuildErrorKind discriminates BuildError variants.
type BuildErrorKind string

const (
	BuildErrRootNotFound      BuildErrorKind = "root_not_found"
	BuildErrRootIO            BuildErrorKind = "root_io"
	BuildErrWriteRootNotFound BuildErrorKind = "write_root_not_found"
	BuildErrWriteRootIO       BuildErrorKind = "write_root_io"
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
	case BuildErrWriteRootNotFound:
		return fmt.Sprintf("workspace write_root does not exist: %s", e.Path)
	case BuildErrWriteRootIO:
		return fmt.Sprintf("workspace write_root io error: %s: %s", e.Path, e.Err)
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

// warnIfDangerousIsolation is wired up by the `dangerous` build to emit a
// construction-time warning when a sandbox is built with IsolationNone. It is
// nil in the default build, where IsolationNone does not exist.
var warnIfDangerousIsolation func(mode IsolationMode)

// NewWorkspaceScopedSandboxWithMode builds a sandbox with an explicit
// isolation mode. Under the `dangerous` build tag, IsolationNone emits a
// warning via log.Printf — it must never be enabled silently in production.
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

	// SC-13: canonicalize the optional write_root the same way so the write
	// boundary check compares canonical paths. It must exist (typically a
	// subdirectory of root). The path strings callers pass stay root-relative;
	// write_root only narrows the boundary, it is not a second join base.
	if cfg.WriteRoot != "" {
		wrAbs, err := filepath.Abs(cfg.WriteRoot)
		if err != nil {
			return nil, &BuildError{Kind: BuildErrWriteRootIO, Path: cfg.WriteRoot, Err: err}
		}
		wrCanonical, err := filepath.EvalSymlinks(wrAbs)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil, &BuildError{Kind: BuildErrWriteRootNotFound, Path: cfg.WriteRoot}
			}
			return nil, &BuildError{Kind: BuildErrWriteRootIO, Path: cfg.WriteRoot, Err: err}
		}
		cfg.WriteRoot = wrCanonical
	}

	if warnIfDangerousIsolation != nil {
		warnIfDangerousIsolation(mode)
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

	// 2. Canonicalize. The target file may not yet exist — for *any*
	// operation, including Read — so canonicalize the parent and re-join the
	// leaf. Resolution is operation-agnostic on purpose: existence is
	// orthogonal to the boundary check. A missing in-workspace path still
	// resolves (via its canonicalized parent) and passes the boundary check;
	// the actual read then naturally returns NotFound, surfaced as a
	// recoverable error by the read tool rather than a PathEscape. A missing
	// path that resolves *outside* the root is still a PathEscape.
	canonical, err := canonicalize(joined)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
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

	// 3. Boundary check. Reads and execute must stay under the read Root;
	//    writes must stay under WriteRoot when set (SC-13: read-everywhere,
	//    write-scoped). The path string was already joined onto Root above —
	//    WriteRoot only narrows where a resolved write target may land, it is
	//    NOT a separate join base — so a write outside WriteRoot is a PathEscape
	//    even though it lives under Root.
	boundary := s.config.Root
	if op == OperationWrite && s.config.WriteRoot != "" {
		boundary = s.config.WriteRoot
	}
	if !pathHasPrefix(canonical, boundary) {
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
	default:
		// IsolationWorkspaceScoped (and, under the `dangerous` build tag,
		// IsolationNone) proceed directly.
	}

	// SC-12: apply exec-hardening knobs when configured. The per-call timeout
	// always wins; DefaultTimeout is the floor for callers that pass none (0).
	effectiveTimeout := timeout
	if effectiveTimeout == 0 && s.config.ExecConfig != nil {
		effectiveTimeout = s.config.ExecConfig.DefaultTimeout
	}

	runCtx := ctx
	var cancel context.CancelFunc
	if effectiveTimeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, effectiveTimeout)
		defer cancel()
	}
	// KillOnDrop is satisfied by construction: exec.CommandContext kills the
	// child when runCtx is cancelled (the timeout firing, or an outer cancel),
	// which is Go's native kill-on-drop. No extra wiring is needed.
	cmd := exec.CommandContext(runCtx, command, args...)
	if workingDir != "" {
		cmd.Dir = workingDir
	} else {
		cmd.Dir = s.config.Root
	}

	// SC-12: CloseStdin and NonInteractiveEnv are gated on ExecConfig so the
	// legacy (nil) path is byte-identical (inherited stdin, inherited env).
	if ec := s.config.ExecConfig; ec != nil {
		if ec.CloseStdin {
			// An empty reader yields immediate EOF, so input-blocked commands
			// fail fast instead of hanging on the inherited terminal.
			cmd.Stdin = bytes.NewReader(nil)
		}
		if len(ec.NonInteractiveEnv) > 0 {
			// Force vars onto the inherited environment in sorted key order
			// (deterministic), matching the Rust reference's BTreeMap.
			env := os.Environ()
			keys := make([]string, 0, len(ec.NonInteractiveEnv))
			for k := range ec.NonInteractiveEnv {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				env = append(env, k+"="+ec.NonInteractiveEnv[k])
			}
			cmd.Env = env
		}
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
				Stderr:   fmt.Sprintf("command timed out after %s", effectiveTimeout),
				ExitCode: -1,
				TimedOut: true,
			}, nil
		}
		if ee, ok := err.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			// SC-15: a failed spawn is a typed violation, not a fake
			// CommandOutput{ExitCode: -1}. Callers already handle the
			// *SandboxViolation arm. Timeout (handled above) keeps its
			// CommandOutput{ExitCode: -1, TimedOut: true} — a real run that
			// exceeded the clock, not a spawn failure.
			return CommandOutput{}, &SandboxViolation{
				Kind:    SandboxExecSpawnFailed,
				Command: command,
				Message: err.Error(),
			}
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
