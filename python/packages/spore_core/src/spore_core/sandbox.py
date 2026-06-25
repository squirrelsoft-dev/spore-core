"""Canonical :class:`SandboxProvider` implementation — issue #6.

:class:`WorkspaceScopedSandbox` is the default in-tree sandbox. It enforces a
workspace root with allow/deny lists, extension filters, a read-only mode,
and per-file size limits. It runs subprocesses directly via
:func:`asyncio.create_subprocess_exec` and offloads large outputs to
``{workspace_root}/.spore/offload/{call_id}.txt``.

Mirrors ``rust/crates/spore-core/src/sandbox.rs`` byte-for-byte where the wire
format crosses the language boundary.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from pathlib import Path

from .errors import SporeError
from .harness import (
    CommandOutput,
    FileRef,
    IsolationMode,
    IsolationModeBubblewrap,
    IsolationModeDocker,
    IsolationModeNone,
    IsolationModeWorkspaceScoped,
    Operation,
    SandboxDisallowedCommand,
    SandboxExecSpawnFailed,
    SandboxExtensionDenied,
    SandboxFileSizeExceeded,
    SandboxPathDenied,
    SandboxPathEscape,
    SandboxReadOnlyViolation,
    SandboxViolation,
    TruncatedOutput,
    WorkspaceConfig,
)
from .model import ToolCall

_logger = logging.getLogger(__name__)


class SandboxBuildError(SporeError):
    """Construction-time error for :class:`WorkspaceScopedSandbox`."""


class SandboxViolationException(SporeError):
    """Wraps a :class:`SandboxViolation` when raised as an exception.

    The canonical Python contract is to return violations as values; this
    type exists so internal helpers can short-circuit through ``try/except``
    when convenient.
    """

    def __init__(self, violation: SandboxViolation) -> None:
        super().__init__(f"sandbox violation: {violation.kind}")
        self.violation = violation


# ============================================================================
# WorkspaceScopedSandbox
# ============================================================================


class WorkspaceScopedSandbox:
    """Path-enforcing sandbox.

    Canonicalizes every resolved path against the workspace root and applies
    allow/deny + extension + read-only policies. Implements the
    :class:`SandboxProvider` protocol structurally — does not inherit.
    """

    def __init__(
        self,
        config: WorkspaceConfig,
        mode: IsolationMode | None = None,
    ) -> None:
        # Validate root exists, then canonicalize for stable comparisons.
        try:
            canonical = config.root.resolve(strict=True)
        except FileNotFoundError as e:
            raise SandboxBuildError(f"workspace root does not exist: {config.root}") from e
        except OSError as e:
            raise SandboxBuildError(f"workspace root io error: {config.root}: {e}") from e

        # SC-13: canonicalize the optional write_root the same way so the
        # boundary check compares canonical paths. It must already exist.
        canonical_write_root: Path | None = None
        if config.write_root is not None:
            try:
                canonical_write_root = config.write_root.resolve(strict=True)
            except FileNotFoundError as e:
                raise SandboxBuildError(
                    f"workspace write_root does not exist: {config.write_root}"
                ) from e
            except OSError as e:
                raise SandboxBuildError(
                    f"workspace write_root io error: {config.write_root}: {e}"
                ) from e

        self._config = config.model_copy(
            update={
                "root": canonical,
                "allowed_paths": [
                    self._normalize_under(canonical, p) for p in config.allowed_paths
                ],
                "denied_paths": [self._normalize_under(canonical, p) for p in config.denied_paths],
                "write_root": canonical_write_root,
            }
        )
        self._mode: IsolationMode = mode or IsolationModeWorkspaceScoped()

        if isinstance(self._mode, IsolationModeNone):
            msg = (
                "WorkspaceScopedSandbox constructed with IsolationMode=None — "
                "trusted-dev use only; do not enable silently in production"
            )
            warnings.warn(msg, stacklevel=2)
            _logger.warning(msg)

    @staticmethod
    def _normalize_under(root: Path, p: Path) -> Path:
        """Anchor a configured allow/deny path under the canonical root."""

        candidate = p if p.is_absolute() else root / p
        # Use resolve(strict=False) so we tolerate not-yet-existing entries.
        return candidate.resolve(strict=False)

    @property
    def config(self) -> WorkspaceConfig:
        return self._config

    # ---- protocol surface -------------------------------------------------

    def isolation_mode(self) -> IsolationMode:
        return self._mode

    def workspace_root(self) -> Path:
        return self._config.root

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        # Per-tool sandbox validation is performed at the call sites via
        # ``resolve_path``; this hook stays as a no-op pass-through. Tools
        # that don't go through ``resolve_path`` (e.g. ``bash_command``) are
        # gated by ``execute_command``.
        _ = call
        return None

    async def resolve_path(self, path: str, operation: Operation = "read") -> Path:
        result = self._resolve(path, operation)
        if isinstance(result, Path):
            return result
        raise SandboxViolationException(result)

    # ---- core resolution --------------------------------------------------

    def _resolve(self, raw: str, operation: Operation) -> Path | SandboxViolation:
        """Path-resolution core. Returns either a canonical :class:`Path`
        or a :class:`SandboxViolation` value.

        Follows the 8-step algorithm in the spec.
        """

        root = self._config.root

        # 1. Join root + raw, treating absolute paths as relative-to-root.
        raw_path = Path(raw)
        if raw_path.is_absolute():
            # Strip the leading separator(s).
            stripped = Path(*raw_path.parts[1:]) if len(raw_path.parts) > 1 else Path()
            joined = root / stripped
        else:
            joined = root / raw_path

        # 2. Canonicalize. The target file may not yet exist — for *any*
        #    operation, including Read — so canonicalize the parent and
        #    re-join the filename. Resolution is operation-agnostic on
        #    purpose: existence is orthogonal to the boundary check. A missing
        #    in-workspace path still resolves (via its canonicalized parent)
        #    and passes the boundary check; the actual read then naturally
        #    returns NotFound, surfaced as a recoverable error by the read
        #    tool rather than a PathEscape. A missing path that resolves
        #    *outside* the root is still a PathEscape (the boundary check in
        #    step 3 rejects it).
        canonical: Path
        try:
            canonical = joined.resolve(strict=True)
        except FileNotFoundError:
            parent = joined.parent
            try:
                parent_canon = parent.resolve(strict=True)
            except (FileNotFoundError, OSError):
                return SandboxPathEscape(path=raw)
            file_name = joined.name
            if not file_name:
                return SandboxPathEscape(path=raw)
            canonical = parent_canon / file_name
        except OSError:
            return SandboxPathEscape(path=raw)

        # 3. Boundary check. Reads and execute must stay under the read `root`;
        #    writes must stay under `write_root` when set (SC-13:
        #    read-everywhere, write-scoped). The path string was already joined
        #    onto `root` above — `write_root` only narrows where the resolved
        #    target is allowed to land, it is NOT a separate join base — so a
        #    write outside `write_root` is a PathEscape even though it lives
        #    under `root`.
        if operation == "write" and self._config.write_root is not None:
            boundary = self._config.write_root
        else:
            boundary = root
        try:
            canonical.relative_to(boundary)
        except ValueError:
            return SandboxPathEscape(path=str(canonical))

        # 4. Denylist.
        for denied in self._config.denied_paths:
            try:
                canonical.relative_to(denied)
            except ValueError:
                continue
            return SandboxPathDenied(path=str(canonical), matched_rule=str(denied))

        # 5. Allowlist (if non-empty).
        if self._config.allowed_paths:
            in_allowlist = False
            for allowed in self._config.allowed_paths:
                try:
                    canonical.relative_to(allowed)
                except ValueError:
                    continue
                in_allowlist = True
                break
            if not in_allowlist:
                return SandboxPathDenied(path=str(canonical), matched_rule="not in allowlist")

        # 6. Denied extensions. Match the path's real extension or a dotfile
        #    name (e.g. ``.env`` → "env").
        ext_real: str | None = None
        if canonical.suffix:
            ext_real = canonical.suffix.lstrip(".").lower()
        ext_dotfile: str | None = None
        name = canonical.name
        if name.startswith(".") and "." not in name[1:]:
            ext_dotfile = name[1:].lower()
        for denied_ext in self._config.denied_extensions:
            trimmed = denied_ext.lstrip(".").lower()
            if ext_real == trimmed or ext_dotfile == trimmed:
                return SandboxExtensionDenied(path=str(canonical), extension=trimmed)

        # 7. Read-only.
        if self._config.read_only and operation in ("write", "execute"):
            return SandboxReadOnlyViolation(path=str(canonical))

        # 8. File-size cap (read ops only).
        if operation == "read" and self._config.max_file_size > 0 and canonical.is_file():
            try:
                size = canonical.stat().st_size
            except OSError:
                size = None
            if size is not None and size > self._config.max_file_size:
                return SandboxFileSizeExceeded(
                    path=str(canonical), size=size, limit=self._config.max_file_size
                )

        return canonical

    # ---- command execution ------------------------------------------------

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        if isinstance(self._mode, IsolationModeBubblewrap):
            raise SandboxViolationException(
                SandboxDisallowedCommand(command=f"bubblewrap isolation not implemented: {command}")
            )
        if isinstance(self._mode, IsolationModeDocker):
            raise SandboxViolationException(
                SandboxDisallowedCommand(command=f"docker isolation not implemented: {command}")
            )

        cwd = str(working_dir) if working_dir is not None else str(self._config.root)

        # SC-12: apply exec-hardening knobs when configured. The per-call
        # `timeout` always wins; `default_timeout` is the floor for callers that
        # pass `None`. `None` exec_config keeps the legacy spawn byte-identical.
        ec = self._config.exec_config
        spawn_kwargs: dict[str, object] = {}
        effective_timeout = timeout
        kill_on_drop = False
        if ec is not None:
            if ec.close_stdin:
                spawn_kwargs["stdin"] = asyncio.subprocess.DEVNULL
            if ec.non_interactive_env:
                # `env=` REPLACES the environment, so merge the forced vars on top
                # of the inherited one. Applied in SORTED key order (deterministic,
                # matching Rust's BTreeMap).
                merged_env = dict(os.environ)
                for k in sorted(ec.non_interactive_env):
                    merged_env[k] = ec.non_interactive_env[k]
                spawn_kwargs["env"] = merged_env
            kill_on_drop = ec.kill_on_drop
            if timeout is None:
                effective_timeout = ec.default_timeout

        try:
            proc = await asyncio.create_subprocess_exec(
                command,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                **spawn_kwargs,  # type: ignore[arg-type]
            )
        except (FileNotFoundError, OSError) as e:
            # SC-15: a failed spawn is a typed violation, not a fake
            # exit_code: -1 success. Callers already handle the exception.
            raise SandboxViolationException(
                SandboxExecSpawnFailed(command=command, message=str(e))
            ) from e

        try:
            if effective_timeout is not None:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=effective_timeout
                )
            else:
                stdout_b, stderr_b = await proc.communicate()
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except (ProcessLookupError, asyncio.CancelledError):
                pass
            secs = int(effective_timeout) if effective_timeout is not None else 0
            return CommandOutput(
                stdout="",
                stderr=f"command timed out after {secs}s",
                exit_code=-1,
                timed_out=True,
                truncated=False,
            )
        except asyncio.CancelledError:
            # SC-12 kill_on_drop: the execute_command future was cancelled. Reap
            # the child instead of orphaning it, then re-raise so cancellation
            # still propagates. Without the knob the child is left running
            # (legacy leak-on-drop behavior).
            if kill_on_drop:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            raise

        return CommandOutput(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            exit_code=proc.returncode if proc.returncode is not None else -1,
            timed_out=False,
            truncated=False,
        )

    # ---- large output offload --------------------------------------------

    async def handle_large_output(
        self,
        content: str,
        call_id: str,
        head_tokens: int,
        tail_tokens: int,
    ) -> TruncatedOutput:
        head_chars = max(0, head_tokens) * 4
        tail_chars = max(0, tail_tokens) * 4
        total_chars = len(content)
        original_size = len(content.encode("utf-8"))

        if total_chars <= head_chars + tail_chars:
            return TruncatedOutput(
                content=content,
                truncated=False,
                full_ref=None,
                original_size=original_size,
            )

        head = content[:head_chars]
        tail = content[total_chars - tail_chars :]
        snippet = f"{head}\n...[truncated]...\n{tail}"

        offload_dir = self._config.root / ".spore" / "offload"
        full_ref: FileRef | None = None
        try:
            offload_dir.mkdir(parents=True, exist_ok=True)
            safe_id = _sanitize_call_id(call_id)
            offload_path = offload_dir / f"{safe_id}.txt"
            offload_path.write_text(content, encoding="utf-8")
            full_ref = FileRef(path=str(offload_path), byte_len=original_size)
        except OSError:
            full_ref = None

        return TruncatedOutput(
            content=snippet,
            truncated=True,
            full_ref=full_ref,
            original_size=original_size,
        )


def _sanitize_call_id(call_id: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in call_id)


__all__ = [
    "SandboxBuildError",
    "SandboxViolationException",
    "WorkspaceScopedSandbox",
]
