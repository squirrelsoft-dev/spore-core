//go:build windows

package sporecore

import "os/exec"

// setProcessGroup is a no-op on Windows: there is no POSIX process group, and
// exec.CommandContext's native kill-on-drop already terminates the child. The
// grandchild-reap parity gap is a Unix-only concern.
func setProcessGroup(_ *exec.Cmd) {}

// processGroupID returns 0 on Windows (no process group).
func processGroupID(_ *exec.Cmd) int { return 0 }

// killProcessGroup is a no-op on Windows.
func killProcessGroup(_ int) {}
