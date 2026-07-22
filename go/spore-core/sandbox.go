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
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
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
		if !errors.Is(err, os.ErrNotExist) {
			return "", &SandboxViolation{Kind: SandboxPathEscape, Path: raw}
		}
		// Walk up to the nearest EXISTING ancestor, canonicalize it, and re-join
		// the missing components — so a target several levels inside a
		// not-yet-created subtree still resolves (write_file / create_dir then
		// materialize the directories). Mirrors the Rust reference's ancestor
		// walk. Symlinks can only live in the existing prefix (a missing path
		// contains none), so canonicalizing that prefix is as strong as
		// canonicalizing a direct parent. A `..`-terminated ancestor aborts
		// the walk: a lexical re-join across `..` cannot be trusted once the
		// directories exist.
		ancestor := joined
		var missing []string
		for {
			name := filepath.Base(ancestor)
			if name == string(filepath.Separator) || name == "." || name == "" {
				return "", &SandboxViolation{Kind: SandboxPathEscape, Path: raw}
			}
			missing = append(missing, name)
			parent := filepath.Dir(ancestor)
			if parent == ancestor {
				return "", &SandboxViolation{Kind: SandboxPathEscape, Path: raw}
			}
			cp, parentErr := canonicalize(parent)
			if parentErr == nil {
				// Re-join the missing components (in reverse — outermost first)
				// onto the canonicalized existing ancestor.
				canonical = cp
				for i := len(missing) - 1; i >= 0; i-- {
					canonical = filepath.Join(canonical, missing[i])
				}
				break
			}
			if !errors.Is(parentErr, os.ErrNotExist) {
				return "", &SandboxViolation{Kind: SandboxPathEscape, Path: raw}
			}
			ancestor = parent
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
	return s.ExecuteCommandStreaming(ctx, command, args, workingDir, timeout, nil)
}

const progressInterval = 100 * time.Millisecond
const progressTailBytes = 2048

type sharedOutputBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *sharedOutputBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *sharedOutputBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func progressPreview(stdout, stderr *sharedOutputBuffer) string {
	combined := sanitizeCommandOutput(stdout.String())
	if err := sanitizeCommandOutput(stderr.String()); err != "" {
		if combined != "" && !strings.HasSuffix(combined, "\n") {
			combined += "\n"
		}
		combined += err
	}
	trimmed := strings.TrimRight(combined, "\r\n\t ")
	if len(trimmed) <= progressTailBytes {
		return trimmed
	}
	start := len(trimmed) - progressTailBytes
	for start < len(trimmed) && !utf8.RuneStart(trimmed[start]) {
		start++
	}
	return trimmed[start:]
}

func drainCommandPipe(r io.Reader, buf *sharedOutputBuffer, done chan<- struct{}) {
	defer close(done)
	defer func() {
		if c, ok := r.(io.Closer); ok {
			_ = c.Close()
		}
	}()
	_, _ = io.Copy(buf, r)
}

func waitCommandDrains(done ...<-chan struct{}) {
	const drainGrace = 500 * time.Millisecond
	for _, ch := range done {
		if ch == nil {
			continue
		}
		select {
		case <-ch:
		case <-time.After(drainGrace):
		}
	}
}

func sanitizeCommandOutput(s string) string {
	return strings.Join(strings.Split(stripANSIAndControls(s), "\n"), "\n")
}

func stripANSIAndControls(s string) string {
	var out strings.Builder
	out.Grow(len(s))
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && size == 1 {
			i++
			continue
		}
		if r == '\x1b' {
			i += size
			if i >= len(s) {
				continue
			}
			next, nsize := utf8.DecodeRuneInString(s[i:])
			i += nsize
			switch next {
			case '[':
				for i < len(s) {
					r2, sz := utf8.DecodeRuneInString(s[i:])
					i += sz
					if r2 >= '\u0040' && r2 <= '\u007e' {
						break
					}
				}
			case ']':
				for i < len(s) {
					r2, sz := utf8.DecodeRuneInString(s[i:])
					i += sz
					if r2 == '\a' {
						break
					}
					if r2 == '\x1b' && i < len(s) {
						r3, sz3 := utf8.DecodeRuneInString(s[i:])
						if r3 == '\\' {
							i += sz3
						}
						break
					}
				}
			default:
				// Two-byte escape consumed.
			}
			continue
		}
		if r == '\n' || r == '\t' || r == '\r' || !isControlRune(r) {
			out.WriteRune(r)
		}
		i += size
	}
	lines := strings.Split(out.String(), "\n")
	for i, line := range lines {
		line = strings.TrimSuffix(line, "\r")
		if idx := strings.LastIndex(line, "\r"); idx >= 0 {
			line = line[idx+1:]
		}
		lines[i] = line
	}
	return strings.Join(lines, "\n")
}

func isControlRune(r rune) bool { return r >= 0 && r < 0x20 || r == 0x7f }

// ExecuteCommandStreaming runs a subprocess and, when progress is non-nil,
// emits sanitized tail previews while it is still running. Final stdout/stderr
// are identical to ExecuteCommand.
func (s *WorkspaceScopedSandbox) ExecuteCommandStreaming(
	ctx context.Context,
	command string,
	args []string,
	workingDir string,
	timeout time.Duration,
	progress CommandProgressSink,
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
	// Process-group reap parity (Rust `cmd.process_group(0)` + GroupKillGuard):
	// make the child a group leader so a cancel/timeout can kill the WHOLE
	// group, not just the direct child. Without this, a grandchild holding the
	// stdout pipe open (e.g. `sh -c "echo x; sleep 30"`) would block cmd.Output()
	// until it exits naturally — orphaning work on cancel/timeout.
	cmd := exec.CommandContext(runCtx, command, args...)
	setProcessGroup(cmd)
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

	// Drain stdout/stderr through parent-owned pipes so cmd.Wait only waits for
	// the direct child and cannot close the read side before our drains capture
	// short output. The child receives the write ends; the parent closes its write
	// descriptors immediately after Start, then bounded post-wait joins prevent a
	// background grandchild that inherited a pipe from holding the harness forever.
	stdoutBuf, stderrBuf := &sharedOutputBuffer{}, &sharedOutputBuffer{}
	stdoutR, stdoutW, err := os.Pipe()
	if err != nil {
		return CommandOutput{}, &SandboxViolation{Kind: SandboxExecSpawnFailed, Command: command, Message: err.Error()}
	}
	stderrR, stderrW, err := os.Pipe()
	if err != nil {
		_ = stdoutR.Close()
		_ = stdoutW.Close()
		return CommandOutput{}, &SandboxViolation{Kind: SandboxExecSpawnFailed, Command: command, Message: err.Error()}
	}
	cmd.Stdout = stdoutW
	cmd.Stderr = stderrW
	stdoutDone, stderrDone := make(chan struct{}), make(chan struct{})
	if err := cmd.Start(); err != nil {
		_ = stdoutR.Close()
		_ = stdoutW.Close()
		_ = stderrR.Close()
		_ = stderrW.Close()
		// SC-15: a failed spawn is a typed violation, not a fake
		// CommandOutput{ExitCode: -1}.
		return CommandOutput{}, &SandboxViolation{
			Kind:    SandboxExecSpawnFailed,
			Command: command,
			Message: err.Error(),
		}
	}
	_ = stdoutW.Close()
	_ = stderrW.Close()
	go drainCommandPipe(stdoutR, stdoutBuf, stdoutDone)
	go drainCommandPipe(stderrR, stderrBuf, stderrDone)
	pgid := processGroupID(cmd)
	waitCh := make(chan error, 1)
	go func() { waitCh <- cmd.Wait() }()

	var ticker *time.Ticker
	var tick <-chan time.Time
	if progress != nil {
		ticker = time.NewTicker(progressInterval)
		defer ticker.Stop()
		tick = ticker.C
	}
	lastPreview := ""

	for {
		select {
		case err := <-waitCh:
			// Normal exit. The group guard is disarmed (nothing to reap).
			waitCommandDrains(stdoutDone, stderrDone)
			exitCode := 0
			if err != nil {
				if ee, ok := err.(*exec.ExitError); ok {
					exitCode = ee.ExitCode()
				} else {
					// Should not happen after a successful Start, but stay safe.
					return CommandOutput{}, &SandboxViolation{
						Kind:    SandboxExecSpawnFailed,
						Command: command,
						Message: err.Error(),
					}
				}
			}
			return CommandOutput{
				Stdout:   sanitizeCommandOutput(stdoutBuf.String()),
				Stderr:   sanitizeCommandOutput(stderrBuf.String()),
				ExitCode: exitCode,
			}, nil
		case <-tick:
			preview := progressPreview(stdoutBuf, stderrBuf)
			if preview != "" && preview != lastPreview {
				progress(preview)
				lastPreview = preview
			}
		case <-runCtx.Done():
			// Cancel/timeout: kill the WHOLE process group (child + grandchildren),
			// then reap the child so it doesn't zombie.
			killProcessGroup(pgid)
			<-waitCh
			waitCommandDrains(stdoutDone, stderrDone)
			if runCtx.Err() == context.DeadlineExceeded {
				stderr := sanitizeCommandOutput(stderrBuf.String())
				if stderr != "" {
					stderr += "\n"
				}
				stderr += fmt.Sprintf("command timed out after %s", effectiveTimeout)
				return CommandOutput{
					Stdout:   sanitizeCommandOutput(stdoutBuf.String()),
					Stderr:   stderr,
					ExitCode: -1,
					TimedOut: true,
				}, nil
			}
			// Outer context cancelled (not a deadline). Surface as a signal-killed
			// CommandOutput — not TimedOut — so callers can distinguish cancel vs
			// timeout. The group has been reaped.
			return CommandOutput{
				Stdout:   sanitizeCommandOutput(stdoutBuf.String()),
				Stderr:   "command cancelled",
				ExitCode: -1,
			}, nil
		}
	}
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
