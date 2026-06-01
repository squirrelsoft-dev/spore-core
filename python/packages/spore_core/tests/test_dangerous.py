"""Tests for the ``dangerous`` opt-in gate — issue #34.

Two footguns are gated out of the default ``spore_core`` surface and reachable
only via ``spore_core.dangerous``:

* ``YoloMode`` — full autonomy, no approval gates. Absent from the ``Mode``
  enum; ``Mode("yolo")`` / ``Mode.YOLO`` fail.
* ``IsolationModeNone`` — no path enforcement. Not exported from ``spore_core``
  or ``spore_core.harness``.

The dangerous-only fixture replay (``dangerous.json``) lives here, gated by the
``dangerous`` marker, so the default cross-language suite never loads it. Run
the gated path explicitly with ``uv run pytest -m dangerous``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import spore_core
import spore_core.harness as harness_mod
import spore_core.prompt_chunk_registry as pcr_mod
from spore_core.dangerous import IsolationModeNone, YoloMode
from spore_core.prompt_chunk_registry import (
    ApprovalPolicy,
    CacheBlock,
    ChunkId,
    ChunkSlot,
    Mode,
    ModeLike,
    PromptChunk,
    StandardPromptChunkRegistry,
)
from spore_core.tool_registry import TaskPhase


# ── The default surface cannot reach the footguns ───────────────────────────


def test_mode_enum_has_no_yolo_member() -> None:
    assert not hasattr(Mode, "YOLO")
    with pytest.raises(KeyError):
        Mode["YOLO"]
    with pytest.raises(ValueError):
        Mode("yolo")


def test_yolo_not_exported_from_default_surfaces() -> None:
    assert "YoloMode" not in spore_core.__all__
    assert not hasattr(spore_core, "YoloMode")
    assert "YoloMode" not in pcr_mod.__all__


def test_isolation_mode_none_not_exported_from_default_surfaces() -> None:
    assert "IsolationModeNone" not in spore_core.__all__
    assert not hasattr(spore_core, "IsolationModeNone")
    assert "IsolationModeNone" not in harness_mod.__all__


# ── The dangerous module IS the opt-in entry point ──────────────────────────


def test_dangerous_module_exposes_both_footguns() -> None:
    assert set(__import__("spore_core.dangerous", fromlist=["__all__"]).__all__) == {
        "YoloMode",
        "IsolationModeNone",
    }


@pytest.mark.dangerous
def test_yolo_mode_is_mode_like() -> None:
    assert isinstance(YoloMode.YOLO, ModeLike)
    assert YoloMode.YOLO.value == "yolo"


@pytest.mark.dangerous
def test_yolo_mode_behavior_matches_spec() -> None:
    yolo = YoloMode.YOLO
    assert yolo.approval_policy() is ApprovalPolicy.NONE
    assert yolo.default_tool_phase() is TaskPhase.EXECUTION
    chunk = yolo.prompt_chunk()
    assert chunk.id == "mode-yolo"
    assert chunk.slot is ChunkSlot.MODE
    assert chunk.cache_block is CacheBlock.STATIC
    assert "Mode: Yolo" in chunk.content


@pytest.mark.dangerous
def test_compose_accepts_yolo_mode() -> None:
    r = StandardPromptChunkRegistry()
    r.register(
        PromptChunk.new("role-test", "you are a test agent", ChunkSlot.ROLE, CacheBlock.STATIC)
    )
    composed = r.compose(ChunkId("role-test"), YoloMode.YOLO, [], [])
    mode_chunks = [c for c in composed.chunks if c.slot is ChunkSlot.MODE]
    assert len(mode_chunks) == 1
    assert mode_chunks[0].id == "mode-yolo"
    assert "Mode: Yolo" in composed.render()
    assert r.validate(composed) == []


@pytest.mark.dangerous
def test_isolation_mode_none_constructs_with_none_tag() -> None:
    none = IsolationModeNone()
    assert none.kind == "none"


# ── Default isolation mode is WorkspaceScoped (safe-by-default) ──────────────


def test_base_provider_default_isolation_is_workspace_scoped() -> None:
    provider = harness_mod.BaseSandboxProvider()
    mode = provider.isolation_mode()
    assert isinstance(mode, harness_mod.IsolationModeWorkspaceScoped)
    assert not isinstance(mode, IsolationModeNone)


# ── Dangerous-only fixture replay (gated) ───────────────────────────────────


_DANGEROUS_FIXTURE = (
    Path(__file__).resolve().parents[4] / "fixtures" / "prompt_chunk_registry" / "dangerous.json"
)

_SLOT_BY_NAME = {s.value: s for s in ChunkSlot}
_CACHE_BLOCK_BY_NAME = {b.value: b for b in CacheBlock}
# The dangerous fixture only uses the gated "yolo" mode.
_DANGEROUS_MODE_BY_TAG: dict[str, ModeLike] = {"yolo": YoloMode.YOLO}


@pytest.mark.dangerous
def test_fixture_replay_dangerous() -> None:
    payload = json.loads(_DANGEROUS_FIXTURE.read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert cases, "dangerous fixture must contain at least one case"

    for case in cases:
        name = case["name"]
        r = StandardPromptChunkRegistry()
        for c in case["register_inputs"]:
            r.register(
                PromptChunk.new(
                    c["id"],
                    c["content"],
                    _SLOT_BY_NAME[c["slot"]],
                    _CACHE_BLOCK_BY_NAME[c["cache_block"]],
                )
            )

        compose_spec = case["compose"]
        composed = r.compose(
            ChunkId(compose_spec["role"]),
            _DANGEROUS_MODE_BY_TAG[compose_spec["mode"]],
            [ChunkId(x) for x in compose_spec["capabilities"]],
            [ChunkId(x) for x in compose_spec["skills"]],
        )

        actual = [(c.slot.value, str(c.id)) for c in composed.chunks]
        expected = [(e["slot"], e["id"]) for e in case["expected_chunks"]]
        assert actual == expected, f"[{name}] composed chunks mismatch"

        assert r.validate(composed) == [], f"[{name}] validate should pass"

        rendered = composed.render()
        for needle in case["rendered_contains"]:
            assert needle in rendered, f"[{name}] rendered missing {needle!r}; got {rendered!r}"
