"""Tests for :class:`WorkspaceScopedSandbox` — issue #6."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from spore_core.harness import (
    IsolationModeBubblewrap,
    IsolationModeDocker,
    IsolationModeNone,
    IsolationModeWorkspaceScoped,
    NetworkPolicyNone,
    SandboxDisallowedCommand,
    SandboxExtensionDenied,
    SandboxFileSizeExceeded,
    SandboxPathDenied,
    SandboxPathEscape,
    SandboxReadOnlyViolation,
    WorkspaceConfig,
)
from spore_core.sandbox import (
    SandboxBuildError,
    SandboxViolationException,
    WorkspaceScopedSandbox,
)

FIXTURES_DIR = Path(__file__).resolve().parents[4] / "fixtures" / "sandbox_violations"


def _cfg(root: Path, **overrides: object) -> WorkspaceConfig:
    base: dict[str, object] = {
        "root": root,
        "allowed_paths": [],
        "denied_paths": [],
        "allowed_extensions": None,
        "denied_extensions": [],
        "read_only": False,
        "max_file_size": 0,
    }
    base.update(overrides)
    return WorkspaceConfig(**base)  # type: ignore[arg-type]


# ----- construction --------------------------------------------------------


def test_build_fails_when_root_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(SandboxBuildError):
        WorkspaceScopedSandbox(_cfg(missing))


def test_build_none_isolation_warns(tmp_path: Path) -> None:
    with pytest.warns(UserWarning):
        sb = WorkspaceScopedSandbox(_cfg(tmp_path), mode=IsolationModeNone())
    assert isinstance(sb.isolation_mode(), IsolationModeNone)


def test_workspace_root_canonicalized(tmp_path: Path) -> None:
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    assert sb.workspace_root() == tmp_path.resolve()


# ----- path resolution variants -------------------------------------------


async def test_escape_via_dotdot(tmp_path: Path) -> None:
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    with pytest.raises(SandboxViolationException) as ei:
        await sb.resolve_path("../etc/passwd", "read")
    assert isinstance(ei.value.violation, SandboxPathEscape)


async def test_escape_via_absolute_dotdot(tmp_path: Path) -> None:
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    with pytest.raises(SandboxViolationException) as ei:
        await sb.resolve_path("/../../etc/passwd", "read")
    assert isinstance(ei.value.violation, SandboxPathEscape)


async def test_path_denied_via_denylist(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "k.txt").write_text("x")
    sb = WorkspaceScopedSandbox(_cfg(tmp_path, denied_paths=[secrets]))
    with pytest.raises(SandboxViolationException) as ei:
        await sb.resolve_path("secrets/k.txt", "read")
    v = ei.value.violation
    assert isinstance(v, SandboxPathDenied)
    assert "secrets" in v.matched_rule


async def test_path_denied_via_allowlist_miss(tmp_path: Path) -> None:
    allowed = tmp_path / "src"
    allowed.mkdir()
    (allowed / "a.rs").write_text("x")
    (tmp_path / "other.rs").write_text("x")
    sb = WorkspaceScopedSandbox(_cfg(tmp_path, allowed_paths=[allowed]))
    # Inside allowlist — OK.
    await sb.resolve_path("src/a.rs", "read")
    # Outside allowlist — denied.
    with pytest.raises(SandboxViolationException) as ei:
        await sb.resolve_path("other.rs", "read")
    v = ei.value.violation
    assert isinstance(v, SandboxPathDenied)
    assert v.matched_rule == "not in allowlist"


async def test_extension_denied(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=1")
    sb = WorkspaceScopedSandbox(_cfg(tmp_path, denied_extensions=["env"]))
    with pytest.raises(SandboxViolationException) as ei:
        await sb.resolve_path(".env", "read")
    assert isinstance(ei.value.violation, SandboxExtensionDenied)


async def test_read_only_violation(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    sb = WorkspaceScopedSandbox(_cfg(tmp_path, read_only=True))
    await sb.resolve_path("a.txt", "read")
    with pytest.raises(SandboxViolationException) as ei:
        await sb.resolve_path("a.txt", "write")
    assert isinstance(ei.value.violation, SandboxReadOnlyViolation)


async def test_file_size_exceeded(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_bytes(b"x" * 1024)
    sb = WorkspaceScopedSandbox(_cfg(tmp_path, max_file_size=100))
    with pytest.raises(SandboxViolationException) as ei:
        await sb.resolve_path("big.txt", "read")
    v = ei.value.violation
    assert isinstance(v, SandboxFileSizeExceeded)
    assert v.size == 1024
    assert v.limit == 100


async def test_write_to_nonexistent_file_works(tmp_path: Path) -> None:
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    resolved = await sb.resolve_path("new_file.txt", "write")
    assert resolved.parent == tmp_path.resolve()


async def test_read_of_missing_in_workspace_file_resolves_not_path_escape(
    tmp_path: Path,
) -> None:
    # Regression for #63: a Read of a not-yet-created file *inside* the
    # workspace must resolve via its canonicalized parent (not be
    # misclassified as PathEscape). The file is absent; resolution still
    # succeeds so the actual read can surface a recoverable not-found.
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    resolved = await sb.resolve_path("output.txt", "read")
    assert resolved.parent == tmp_path.resolve()
    assert resolved.name == "output.txt"
    assert not resolved.exists()


async def test_read_of_missing_file_in_subdir_resolves(tmp_path: Path) -> None:
    # Parent dir exists, leaf file does not — still resolves for Read.
    (tmp_path / "sub").mkdir()
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    resolved = await sb.resolve_path("sub/missing.txt", "read")
    assert resolved.parent == (tmp_path / "sub").resolve()
    assert not resolved.exists()


async def test_read_of_missing_file_outside_root_still_path_escape(
    tmp_path: Path,
) -> None:
    # Regression for #63: a Read of a *non-existent* path that resolves
    # outside the workspace root must still be a PathEscape, not a not-found.
    # (`..` makes the canonicalized parent escape the root.)
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    with pytest.raises(SandboxViolationException) as ei:
        await sb.resolve_path("../nonexistent_passwd", "read")
    assert isinstance(ei.value.violation, SandboxPathEscape)


async def test_read_of_existing_in_workspace_file_resolves(tmp_path: Path) -> None:
    (tmp_path / "present.txt").write_text("hi")
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    resolved = await sb.resolve_path("present.txt", "read")
    assert resolved == (tmp_path / "present.txt").resolve()


# ----- execute_command ----------------------------------------------------


async def test_execute_command_echo(tmp_path: Path) -> None:
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    out = await sb.execute_command("echo", ["hello"], None, None)
    assert out.exit_code == 0
    assert "hello" in out.stdout
    assert out.timed_out is False


async def test_execute_command_timeout(tmp_path: Path) -> None:
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    out = await sb.execute_command("sleep", ["5"], None, 0.05)
    assert out.timed_out is True
    assert out.exit_code == -1


async def test_execute_command_bubblewrap_disallowed(tmp_path: Path) -> None:
    sb = WorkspaceScopedSandbox(_cfg(tmp_path), mode=IsolationModeBubblewrap())
    with pytest.raises(SandboxViolationException) as ei:
        await sb.execute_command("echo", ["hi"], None, None)
    assert isinstance(ei.value.violation, SandboxDisallowedCommand)


async def test_execute_command_docker_disallowed(tmp_path: Path) -> None:
    sb = WorkspaceScopedSandbox(
        _cfg(tmp_path),
        mode=IsolationModeDocker(image="alpine", network=NetworkPolicyNone()),
    )
    with pytest.raises(SandboxViolationException) as ei:
        await sb.execute_command("echo", ["hi"], None, None)
    assert isinstance(ei.value.violation, SandboxDisallowedCommand)


# ----- handle_large_output ------------------------------------------------


async def test_handle_large_output_below_threshold(tmp_path: Path) -> None:
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    out = await sb.handle_large_output("short", "c1", 100, 100)
    assert out.truncated is False
    assert out.full_ref is None
    assert out.content == "short"


async def test_handle_large_output_above_threshold_offloads(tmp_path: Path) -> None:
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    content = "x" * 10_000
    out = await sb.handle_large_output(content, "call-1", 10, 10)
    assert out.truncated is True
    assert "...[truncated]..." in out.content
    assert out.full_ref is not None
    assert out.full_ref.byte_len == len(content)
    assert ".spore" in out.full_ref.path
    assert Path(out.full_ref.path).read_text() == content


# ----- isolation mode -----------------------------------------------------


def test_default_isolation_mode_workspace_scoped(tmp_path: Path) -> None:
    sb = WorkspaceScopedSandbox(_cfg(tmp_path))
    assert isinstance(sb.isolation_mode(), IsolationModeWorkspaceScoped)


# ----- cross-language fixture replay --------------------------------------


def _materialize(root: Path, entries: dict[str, dict[str, object]]) -> None:
    for rel, entry in entries.items():
        target = root / rel
        if entry.get("dir"):
            target.mkdir(parents=True, exist_ok=True)
        elif "file" in entry:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(entry["file"]))


@pytest.mark.parametrize(
    "fixture_path",
    sorted(FIXTURES_DIR.glob("*.json")) if FIXTURES_DIR.is_dir() else [],
    ids=lambda p: p.stem,
)
async def test_sandbox_violation_fixture(fixture_path: Path, tmp_path: Path) -> None:
    sc = json.loads(fixture_path.read_text())
    root = tmp_path.resolve()
    _materialize(root, sc.get("filesystem", {}))
    cfg = WorkspaceConfig(
        root=root,
        allowed_paths=[root / p for p in sc.get("allowed_paths", [])],
        denied_paths=[root / p for p in sc.get("denied_paths", [])],
        allowed_extensions=None,
        denied_extensions=list(sc.get("denied_extensions", [])),
        read_only=bool(sc.get("read_only", False)),
        max_file_size=int(sc.get("max_file_size", 0)),
    )
    sb = WorkspaceScopedSandbox(cfg)
    operation = sc["operation"]
    expected_kind = sc["expected"]["kind"]

    try:
        await sb.resolve_path(sc["raw_path"], operation)
        actual_kind = "ok"
    except SandboxViolationException as e:
        actual_kind = e.violation.kind

    assert actual_kind == expected_kind, (
        f"fixture {sc.get('name')} expected={expected_kind} got={actual_kind}"
    )


def test_at_least_one_fixture_present() -> None:
    assert FIXTURES_DIR.is_dir(), f"fixtures dir missing: {FIXTURES_DIR}"
    assert any(FIXTURES_DIR.glob("*.json"))


# ----- platform notes -----------------------------------------------------


if sys.platform == "win32":  # pragma: no cover - guard
    pytest.skip("WorkspaceScopedSandbox tests assume POSIX paths", allow_module_level=True)
