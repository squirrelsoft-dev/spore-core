"""Unit tests for :mod:`spore_core.prompt_chunk_registry` (issue #24).

Mirrors the Rust reference tests at
``rust/crates/spore-core/src/prompt_chunk_registry.rs``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spore_core.prompt_chunk_registry import (
    ApprovalPolicy,
    CacheBlock,
    ChunkId,
    ChunkSlot,
    ComposeFailed,
    ComposedPrompt,
    ConflictingCacheBlock,
    ConflictingModeChunks,
    DuplicateChunkId,
    InvalidSlot,
    MissingRequiredSlot,
    Mode,
    PerTurnChunkInStaticBlock,
    PromptChunk,
    StandardPromptChunkRegistry,
    standard_chunks,
)
from spore_core.tool_registry import TaskPhase


# ── Helpers ──────────────────────────────────────────────────────────────────


def _registry_with_role(chunk_id: str = "role-test") -> StandardPromptChunkRegistry:
    r = StandardPromptChunkRegistry()
    r.register(PromptChunk.new(chunk_id, "you are a test agent", ChunkSlot.ROLE, CacheBlock.STATIC))
    return r


# ── Rule: register rejects duplicate ids ────────────────────────────────────


def test_register_rejects_duplicate_id() -> None:
    r = StandardPromptChunkRegistry()
    r.register(PromptChunk.new("x", "hello", ChunkSlot.CAPABILITY, CacheBlock.STATIC))
    with pytest.raises(DuplicateChunkId):
        r.register(PromptChunk.new("x", "world", ChunkSlot.CAPABILITY, CacheBlock.STATIC))


# ── Rule: register rejects empty content ────────────────────────────────────


def test_register_rejects_empty_content() -> None:
    r = StandardPromptChunkRegistry()
    with pytest.raises(InvalidSlot):
        r.register(PromptChunk.new("x", "   ", ChunkSlot.CAPABILITY, CacheBlock.STATIC))


# ── Rule: Budget/Ephemeral slots must be PerTurn ────────────────────────────


def test_budget_slot_rejects_static_cache_block() -> None:
    r = StandardPromptChunkRegistry()
    with pytest.raises(ConflictingCacheBlock) as exc:
        r.register(PromptChunk.new("b", "budget warning", ChunkSlot.BUDGET, CacheBlock.STATIC))
    assert exc.value.slot is ChunkSlot.BUDGET
    assert exc.value.expected is CacheBlock.PER_TURN
    assert exc.value.actual is CacheBlock.STATIC


def test_ephemeral_slot_rejects_per_session_cache_block() -> None:
    r = StandardPromptChunkRegistry()
    with pytest.raises(ConflictingCacheBlock):
        r.register(PromptChunk.new("e", "ephemeral", ChunkSlot.EPHEMERAL, CacheBlock.PER_SESSION))


# ── Rule: Role/Mode slots must be Static ────────────────────────────────────


def test_role_slot_rejects_non_static_cache_block() -> None:
    r = StandardPromptChunkRegistry()
    with pytest.raises(ConflictingCacheBlock):
        r.register(PromptChunk.new("r", "role", ChunkSlot.ROLE, CacheBlock.PER_SESSION))


# ── Rule: compose requires the role chunk to exist ──────────────────────────


def test_compose_missing_role_returns_error() -> None:
    r = StandardPromptChunkRegistry()
    with pytest.raises(ComposeFailed) as exc:
        r.compose(ChunkId("missing"), Mode.SAFE_AUTO, [], [])
    assert any(
        isinstance(e, MissingRequiredSlot) and e.slot is ChunkSlot.ROLE for e in exc.value.errors
    )


# ── Rule: compose includes the mode chunk via Mode.prompt_chunk() ───────────


def test_compose_includes_mode_chunk_from_enum() -> None:
    r = _registry_with_role()
    composed = r.compose(ChunkId("role-test"), Mode.PLAN, [], [])
    mode_chunks = [c for c in composed.chunks if c.slot is ChunkSlot.MODE]
    assert len(mode_chunks) == 1
    assert mode_chunks[0].id == "mode-plan"


# ── Rule: compose orders chunks by slot ─────────────────────────────────────


def test_compose_orders_by_slot() -> None:
    r = _registry_with_role()
    r.register(PromptChunk.new("cap-1", "cap one", ChunkSlot.CAPABILITY, CacheBlock.STATIC))
    r.register(PromptChunk.new("skill-1", "skill one", ChunkSlot.SKILL, CacheBlock.STATIC))
    composed = r.compose(
        ChunkId("role-test"),
        Mode.AUTO_EDIT,
        [ChunkId("cap-1")],
        [ChunkId("skill-1")],
    )
    slots = [c.slot for c in composed.chunks]
    assert slots == [ChunkSlot.ROLE, ChunkSlot.MODE, ChunkSlot.CAPABILITY, ChunkSlot.SKILL]


# ── Rule: block_1_hash and block_2_hash reflect chunk contents ──────────────


def test_block_hashes_are_stable_for_identical_content() -> None:
    a = _registry_with_role().compose(ChunkId("role-test"), Mode.SAFE_AUTO, [], [])
    b = _registry_with_role().compose(ChunkId("role-test"), Mode.SAFE_AUTO, [], [])
    assert a.block_1_hash == b.block_1_hash
    assert a.block_2_hash == b.block_2_hash


def test_block_1_hash_changes_when_content_changes() -> None:
    a = _registry_with_role().compose(ChunkId("role-test"), Mode.SAFE_AUTO, [], [])
    r2 = StandardPromptChunkRegistry()
    r2.register(
        PromptChunk.new("role-test", "DIFFERENT ROLE CONTENT", ChunkSlot.ROLE, CacheBlock.STATIC)
    )
    b = r2.compose(ChunkId("role-test"), Mode.SAFE_AUTO, [], [])
    assert a.block_1_hash != b.block_1_hash


# ── Rule: validate flags PerTurn chunk in Static block ──────────────────────


def test_validate_flags_perturn_chunk_in_static_block() -> None:
    r = StandardPromptChunkRegistry()
    composed = ComposedPrompt(
        chunks=[
            PromptChunk.new("role-x", "x", ChunkSlot.ROLE, CacheBlock.STATIC),
            Mode.SAFE_AUTO.prompt_chunk(),
            # Budget chunk with Static cache block — simulates a bug.
            PromptChunk(
                id=ChunkId("bad-budget"),
                content="b",
                slot=ChunkSlot.BUDGET,
                cache_block=CacheBlock.STATIC,
                hash=0,
            ),
        ],
        block_1_hash=0,
        block_2_hash=0,
    )
    errors = r.validate(composed)
    assert any(isinstance(e, PerTurnChunkInStaticBlock) and e.id == "bad-budget" for e in errors)


# ── Rule: validate flags conflicting Mode chunks ────────────────────────────


def test_validate_flags_more_than_one_mode_chunk() -> None:
    r = StandardPromptChunkRegistry()
    composed = ComposedPrompt(
        chunks=[
            PromptChunk.new("role-x", "x", ChunkSlot.ROLE, CacheBlock.STATIC),
            Mode.SAFE_AUTO.prompt_chunk(),
            Mode.ALWAYS_ASK.prompt_chunk(),
        ],
        block_1_hash=0,
        block_2_hash=0,
    )
    errors = r.validate(composed)
    assert any(isinstance(e, ConflictingModeChunks) for e in errors)


# ── Rule: validate flags missing required slots ─────────────────────────────


def test_validate_flags_missing_role_slot() -> None:
    r = StandardPromptChunkRegistry()
    composed = ComposedPrompt(
        chunks=[Mode.SAFE_AUTO.prompt_chunk()],
        block_1_hash=0,
        block_2_hash=0,
    )
    errors = r.validate(composed)
    assert any(isinstance(e, MissingRequiredSlot) and e.slot is ChunkSlot.ROLE for e in errors)


# ── Rule: get returns registered chunks ─────────────────────────────────────


def test_get_returns_registered_chunk() -> None:
    r = _registry_with_role("role-x")
    c = r.get(ChunkId("role-x"))
    assert c is not None and c.id == "role-x"
    assert r.get(ChunkId("nope")) is None


# ── Mode helpers ────────────────────────────────────────────────────────────


def test_mode_approval_policy_matches_spec() -> None:
    assert Mode.ALWAYS_ASK.approval_policy() is ApprovalPolicy.ALWAYS_ASK
    assert Mode.AUTO_EDIT.approval_policy() is ApprovalPolicy.AUTO_EXPLAIN
    assert Mode.PLAN.approval_policy() is ApprovalPolicy.PLAN_ONLY
    assert Mode.SAFE_AUTO.approval_policy() is ApprovalPolicy.SAFE_AUTO


def test_mode_default_tool_phase() -> None:
    assert Mode.PLAN.default_tool_phase() is TaskPhase.PLANNING
    assert Mode.AUTO_EDIT.default_tool_phase() is TaskPhase.EXECUTION


# ── ComposedPrompt rendering ────────────────────────────────────────────────


def test_composed_prompt_render_joins_chunks_with_blank_line() -> None:
    r = _registry_with_role()
    composed = r.compose(ChunkId("role-test"), Mode.SAFE_AUTO, [], [])
    assert composed.rendered is None
    rendered = composed.render()
    assert "you are a test agent" in rendered
    assert "Mode: SafeAuto" in rendered
    assert composed.rendered is not None


# ── Standard library bootstraps cleanly ─────────────────────────────────────


def test_standard_chunks_register_cleanly() -> None:
    r = StandardPromptChunkRegistry()
    r.register_standard_chunks()
    assert r.get(ChunkId("role-coding-agent")) is not None
    assert r.get(ChunkId("capability-bash")) is not None
    assert r.get(ChunkId("skill-testing")) is not None


def test_standard_chunks_includes_all_safe_mode_ids() -> None:
    ids = {str(c.id) for c in standard_chunks()}
    for expected in (
        "mode-always-ask",
        "mode-auto-edit",
        "mode-plan",
        "mode-safe-auto",
    ):
        assert expected in ids
    # The Yolo footgun is gated out of the default library (issue #34).
    assert "mode-yolo" not in ids


# ── Compose with standard library smoke-test ────────────────────────────────


def test_compose_with_standard_chunks_produces_coding_agent_prompt() -> None:
    r = StandardPromptChunkRegistry()
    r.register_standard_chunks()
    composed = r.compose(
        ChunkId("role-coding-agent"),
        Mode.SAFE_AUTO,
        [ChunkId("capability-bash"), ChunkId("capability-filesystem"), ChunkId("capability-git")],
        [ChunkId("skill-testing"), ChunkId("skill-security-review")],
    )
    assert len(composed.chunks) == 1 + 1 + 3 + 2
    assert composed.chunks[0].slot is ChunkSlot.ROLE
    assert composed.chunks[1].slot is ChunkSlot.MODE
    assert composed.chunks[1].id == "mode-safe-auto"


# ── Fixture-replay: cross-language consistency ──────────────────────────────


_FIXTURE_PATH = (
    Path(__file__).resolve().parents[4] / "fixtures" / "prompt_chunk_registry" / "basic.json"
)

_SLOT_BY_NAME = {s.value: s for s in ChunkSlot}
_CACHE_BLOCK_BY_NAME = {b.value: b for b in CacheBlock}
_MODE_BY_NAME = {m.value: m for m in Mode}


def test_fixture_replay_basic() -> None:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert cases, "fixture must contain at least one case"

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
            _MODE_BY_NAME[compose_spec["mode"]],
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
