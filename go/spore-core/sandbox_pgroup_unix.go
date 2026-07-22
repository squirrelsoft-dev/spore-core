//go:build !windows

package sporecore

import (
	"os/exec"
	"syscall"
)

// setProcessGroup makes the child its own process-group leader (pgid == child
// pid), mirroring the Rust reference's `cmd.process_group(0)`. This lets a
// cancel/timeout kill the WHOLE group (the child plus any grandchildren it
// spawned) instead of orphaning grandchildren that hold the stdout pipe open.
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// processGroupID returns the process-group id of a started child. On Unix the
// group leader's pid == the pgid.
func processGroupID(cmd *exec.Cmd) int {
	if cmd.Process == nil {
		return 0
	}
	return cmd.Process.Pid
}

// killProcessGroup sends SIGKILL to the whole process group. A stale/invalid
// pgid returns ESRCH harmlessly. No-op when pgid <= 0.
func killProcessGroup(pgid int) {
	if pgid <= 0 {
		return
	}
	// Negative pid signals the whole process group.
	_ = syscall.Kill(-pgid, syscall.SIGKILL)
}
