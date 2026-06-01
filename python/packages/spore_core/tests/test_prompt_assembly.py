"""Tests for the #79 prompt assembly engine (Python).

Mirrors the Rust reference test suite: rules R1–R19, R21 (R20 is the deferred
Remote provider, A6), plus fixture-replay tests against the shared ground-truth
fixtures in ``fixtures/prompt_assembly/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spore_core.context import ContextSources, SegmentStability
from spore_core.harness import SessionId, TaskId
from spore_core.hooks import HookEvent
from spore_core.prompt_assembly import (
    AssemblyContext,
    ChunkCondition,
    ChunkProvider,
    ChunkProviderError,
    CompositeChunkProvider,
    ContextSourcesBuilder,
    Custom,
    EmbeddedChunkProvider,
    InMemoryChunkProvider,
    PromptChunk,
    StorageScope,
    ToolAffinity,
    breakpoint_ids,
    chunks_to_segments,
)
from spore_core.prompt_chunk_registry import Mode
from spore_core.tool_registry import TaskPhase

_FIXTURE_DIR = Path(__file__).resolve().parents[4] / "fixtures" / "prompt_assembly"


def _ctx() -> AssemblyContext:
    return AssemblyContext.new(
        SessionId("s1"),
        TaskId("t1"),
        1,
        Mode.SAFE_AUTO,
        TaskPhase.EXECUTION,
    )


def _builder() -> ContextSourcesBuilder:
    return ContextSourcesBuilder()


# ── R1: Always always matches ───────────────────────────────────────────────
def test_r1_always_matches() -> None:
    assert _builder().evaluate(ChunkCondition.always(), _ctx())


# ── R2: WhenMode ─────────────────────────────────────────────────────────────
def test_r2_when_mode() -> None:
    b = _builder()
    c = _ctx()
    c.mode = Mode.PLAN
    assert b.evaluate(ChunkCondition.when_mode(Mode.PLAN), c)
    assert not b.evaluate(ChunkCondition.when_mode(Mode.SAFE_AUTO), c)


# ── R3: WhenToolActive ───────────────────────────────────────────────────────
def test_r3_when_tool_active() -> None:
    b = _builder()
    c = _ctx()
    c.active_tool_names.add("bash")
    assert b.evaluate(ChunkCondition.when_tool_active("bash"), c)
    assert not b.evaluate(ChunkCondition.when_tool_active("grep"), c)


# ── R4: WhenToolCapability ───────────────────────────────────────────────────
def test_r4_when_tool_capability() -> None:
    b = _builder()
    c = _ctx()
    c.active_capabilities.add(("bash", "sandbox"))
    assert b.evaluate(ChunkCondition.when_tool_capability("bash", "sandbox"), c)
    assert not b.evaluate(ChunkCondition.when_tool_capability("bash", "git"), c)


# ── R5: WhenPhase / WhenAgentType / WhenFeature ─────────────────────────────
def test_r5_when_phase_agent_feature() -> None:
    b = _builder()
    c = _ctx()
    c.phase = TaskPhase.PLANNING
    c.agent_type = "planner"
    c.features["beta"] = True
    c.features["alpha"] = False

    assert b.evaluate(ChunkCondition.when_phase(TaskPhase.PLANNING), c)
    assert not b.evaluate(ChunkCondition.when_phase(TaskPhase.EXECUTION), c)

    assert b.evaluate(ChunkCondition.when_agent_type("planner"), c)
    assert not b.evaluate(ChunkCondition.when_agent_type("coder"), c)

    # feature true iff present AND true
    assert b.evaluate(ChunkCondition.when_feature("beta"), c)
    assert not b.evaluate(ChunkCondition.when_feature("alpha"), c)
    assert not b.evaluate(ChunkCondition.when_feature("missing"), c)


# ── R6: OnTrigger ────────────────────────────────────────────────────────────
def test_r6_on_trigger() -> None:
    b = _builder()
    c = _ctx()
    cond = ChunkCondition.on_trigger(["deploy", "rollback"])
    assert not b.evaluate(cond, c)  # None message -> false
    c.incoming_message = "please deploy the service"
    assert b.evaluate(cond, c)
    c.incoming_message = "nothing relevant"
    assert not b.evaluate(cond, c)


# ── R7: OnEvent ──────────────────────────────────────────────────────────────
def test_r7_on_event() -> None:
    b = _builder()
    c = _ctx()
    cond = ChunkCondition.on_event(HookEvent.PRE_COMPACT)
    assert not b.evaluate(cond, c)
    c.pending_events.append(HookEvent.PRE_COMPACT)
    assert b.evaluate(cond, c)


# ── R8: All / Any / Not ──────────────────────────────────────────────────────
def test_r8_all_any_not() -> None:
    b = _builder()
    c = _ctx()
    c.mode = Mode.PLAN
    c.active_tool_names.add("bash")

    all_ok = ChunkCondition.all_of(
        [ChunkCondition.when_mode(Mode.PLAN), ChunkCondition.when_tool_active("bash")]
    )
    assert b.evaluate(all_ok, c)

    all_fail = ChunkCondition.all_of(
        [ChunkCondition.when_mode(Mode.PLAN), ChunkCondition.when_tool_active("grep")]
    )
    assert not b.evaluate(all_fail, c)

    any_ok = ChunkCondition.any_of(
        [ChunkCondition.when_tool_active("grep"), ChunkCondition.when_mode(Mode.PLAN)]
    )
    assert b.evaluate(any_ok, c)

    not_ok = ChunkCondition.not_(ChunkCondition.when_mode(Mode.SAFE_AUTO))
    assert b.evaluate(not_ok, c)


# ── R9: Custom evaluated against ctx ─────────────────────────────────────────
def test_r9_custom_evaluated_against_ctx() -> None:
    b = _builder()
    c = _ctx()
    c.turn_number = 5
    cond = ChunkCondition.custom(lambda ctx: ctx.turn_number > 3)
    assert b.evaluate(cond, c)
    c.turn_number = 1
    assert not b.evaluate(cond, c)


# ── R10: bucketed by stability ───────────────────────────────────────────────
def test_r10_bucketed_by_stability() -> None:
    b = ContextSourcesBuilder.with_chunks(
        [
            PromptChunk.new("s", "static").with_stability(SegmentStability.STATIC),
            PromptChunk.new("ps", "session").with_stability(SegmentStability.PER_SESSION),
            PromptChunk.new("pt", "turn").with_stability(SegmentStability.PER_TURN),
        ]
    )
    buckets = b.assemble(_ctx())
    assert [c.id for c in buckets.static_chunks] == ["s"]
    assert [c.id for c in buckets.per_session] == ["ps"]
    assert [c.id for c in buckets.per_turn] == ["pt"]


# ── R11: registration order preserved within bucket ─────────────────────────
def test_r11_registration_order_within_bucket() -> None:
    b = ContextSourcesBuilder.with_chunks(
        [PromptChunk.new("a", "a"), PromptChunk.new("b", "b"), PromptChunk.new("c", "c")]
    )
    buckets = b.assemble(_ctx())
    assert [c.id for c in buckets.static_chunks] == ["a", "b", "c"]


# ── R12: tool-affinity 4-way matrix ──────────────────────────────────────────
def test_r12_tool_affinity_matrix() -> None:
    chunk = PromptChunk.new("bash-git", "git guide").with_tool_affinity(
        ToolAffinity.with_capability("bash", "git")
    )
    b = ContextSourcesBuilder.with_chunks([chunk])

    # (1) tool inactive, cap inactive -> excluded
    c = _ctx()
    assert b.assemble(c).static_chunks == []

    # (2) tool active, cap inactive -> excluded
    c.active_tool_names.add("bash")
    assert b.assemble(c).static_chunks == []

    # (3) tool active, cap active -> included
    c.active_capabilities.add(("bash", "git"))
    assert len(b.assemble(c).static_chunks) == 1

    # (4) tool inactive but cap present -> excluded (tool gate first)
    c2 = _ctx()
    c2.active_capabilities.add(("bash", "git"))
    assert b.assemble(c2).static_chunks == []

    # capability None: included as soon as the tool is active
    chunk2 = PromptChunk.new("bash-any", "bash guide").with_tool_affinity(ToolAffinity.tool("bash"))
    b2 = ContextSourcesBuilder.with_chunks([chunk2])
    c3 = _ctx()
    assert b2.assemble(c3).static_chunks == []
    c3.active_tool_names.add("bash")
    assert len(b2.assemble(c3).static_chunks) == 1


# ── R13: OnTrigger matches pushed to PerTurn ─────────────────────────────────
def test_r13_trigger_match_routes_to_per_turn() -> None:
    chunk = (
        PromptChunk.new("playbook2", "rollback steps")
        .with_stability(SegmentStability.STATIC)
        .with_condition(ChunkCondition.on_trigger(["rollback"]))
        .with_triggers(["rollback"])
    )
    b = ContextSourcesBuilder.with_chunks([chunk])
    c = _ctx()
    assert b.assemble(c).static_chunks == []
    assert b.assemble(c).per_turn == []

    c.incoming_message = "we must rollback now"
    buckets = b.assemble(c)
    assert buckets.static_chunks == []
    assert [x.id for x in buckets.per_turn] == ["playbook2"]


# ── R14: OnEvent injected to PerTurn only when event pending ─────────────────
def test_r14_on_event_injected_only_when_pending() -> None:
    chunk = (
        PromptChunk.new("reminder", "system reminder")
        .with_stability(SegmentStability.PER_TURN)
        .with_condition(ChunkCondition.on_event(HookEvent.PRE_COMPACT))
    )
    b = ContextSourcesBuilder.with_chunks([chunk])

    c = _ctx()
    assert b.assemble(c).per_turn == []

    c.pending_events.append(HookEvent.PRE_COMPACT)
    buckets = b.assemble(c)
    assert [x.id for x in buckets.per_turn] == ["reminder"]


# ── R15: Block-1 hash stable across two builds of identical Static set ──────
def test_r15_block_1_hash_stable() -> None:
    def mk() -> ContextSourcesBuilder:
        return ContextSourcesBuilder.with_chunks(
            [
                PromptChunk.new("core", "identity rules"),
                PromptChunk.new("style", "be concise"),
            ]
        )

    b1, b2 = mk(), mk()
    cp1 = b1.compose_block_1(b1.assemble(_ctx()))
    cp2 = b2.compose_block_1(b2.assemble(_ctx()))
    assert cp1.block_1_hash == cp2.block_1_hash

    b3 = ContextSourcesBuilder.with_chunks(
        [PromptChunk.new("core", "DIFFERENT identity"), PromptChunk.new("style", "be concise")]
    )
    cp3 = b3.compose_block_1(b3.assemble(_ctx()))
    assert cp1.block_1_hash != cp3.block_1_hash


# ── R16: cache_breakpoint injects breakpoint after chunk ────────────────────
def test_r16_cache_breakpoint() -> None:
    b = ContextSourcesBuilder.with_chunks(
        [
            PromptChunk.new("a", "a"),
            PromptChunk.new("b", "b").with_cache_breakpoint(True),
            PromptChunk.new("c", "c"),
        ]
    )
    buckets = b.assemble(_ctx())
    assert breakpoint_ids(buckets) == ["b"]

    segs = chunks_to_segments(buckets.static_chunks)
    assert next(s for s in segs if s.name == "b").cache_breakpoint
    assert not next(s for s in segs if s.name == "a").cache_breakpoint


# ── R17: tool not active yields no description chunk ─────────────────────────
def test_r17_tool_not_active_no_description() -> None:
    chunk = PromptChunk.new("bash-desc", "Bash tool: run shell commands").with_tool_affinity(
        ToolAffinity.tool("bash")
    )
    b = ContextSourcesBuilder.with_chunks([chunk])
    assert b.assemble(_ctx()).static_chunks == []
    c = _ctx()
    c.active_tool_names.add("bash")
    assert len(b.assemble(c).static_chunks) == 1


# ── R18: EmbeddedChunkProvider invalidate no-op, load same ──────────────────
async def test_r18_embedded_provider() -> None:
    p = EmbeddedChunkProvider([PromptChunk.new("x", "y")])
    a = await p.load()
    p.invalidate()
    b = await p.load()
    assert [c.id for c in a] == [c.id for c in b]
    assert len(a) == 1


# ── R19: InMemoryChunkProvider returns registered; set replaces ─────────────
async def test_r19_in_memory_provider() -> None:
    p = InMemoryChunkProvider([PromptChunk.new("x", "y")])
    assert len(await p.load()) == 1
    p.set([PromptChunk.new("a", "1"), PromptChunk.new("b", "2")])
    after = await p.load()
    assert [c.id for c in after] == ["a", "b"]


# ── R21: CompositeChunkProvider merges + propagates invalidate ──────────────
class _CountingProvider:
    def __init__(self, chunks: list[PromptChunk]) -> None:
        self._chunks = chunks
        self.invalidated = 0

    async def load(self) -> list[PromptChunk]:
        return list(self._chunks)

    def invalidate(self) -> None:
        self.invalidated += 1


async def test_r21_composite_provider() -> None:
    p1 = _CountingProvider([PromptChunk.new("a", "1")])
    p2 = _CountingProvider([PromptChunk.new("b", "2"), PromptChunk.new("c", "3")])
    comp = CompositeChunkProvider().add(p1).add(p2)
    merged = await comp.load()
    assert [c.id for c in merged] == ["a", "b", "c"]  # add order preserved

    comp.invalidate()
    assert p1.invalidated == 1
    assert p2.invalidated == 1

    # Concrete providers satisfy the structural protocol.
    assert isinstance(p1, ChunkProvider)
    assert isinstance(comp, ChunkProvider)


# ── PartialEq / equality: Custom never equal (A3) ───────────────────────────
def test_custom_condition_never_equal() -> None:
    def f(_ctx: AssemblyContext) -> bool:
        return True

    a = ChunkCondition.custom(f)
    b = ChunkCondition.custom(f)
    assert a != b
    assert a != a  # never equal to anything, including itself

    # Non-custom variants still compare by value.
    assert ChunkCondition.always() == ChunkCondition.always()
    assert ChunkCondition.when_mode(Mode.SAFE_AUTO) == ChunkCondition.when_mode(Mode.SAFE_AUTO)


# ── Serialization: Custom skipped (A3) ──────────────────────────────────────
def test_custom_condition_serializes_to_none() -> None:
    cond = ChunkCondition.custom(lambda _c: True)
    assert cond.to_json() is None

    chunk = PromptChunk.new("x", "y").with_condition(Custom(lambda _c: True))
    wire = chunk.to_json()
    assert wire["condition"] is None
    back = PromptChunk.from_json(wire)
    assert back.condition == ChunkCondition.always()


def test_condition_round_trips_serializable_variants() -> None:
    cond = ChunkCondition.all_of(
        [
            ChunkCondition.when_mode(Mode.PLAN),
            ChunkCondition.any_of(
                [
                    ChunkCondition.when_tool_active("bash"),
                    ChunkCondition.not_(ChunkCondition.when_feature("beta")),
                ]
            ),
            ChunkCondition.on_event(HookEvent.PRE_TURN),
            ChunkCondition.on_trigger(["deploy"]),
            ChunkCondition.when_tool_capability("bash", "git"),
            ChunkCondition.when_phase(TaskPhase.PLANNING),
            ChunkCondition.when_agent_type("planner"),
        ]
    )
    s = json.dumps(cond.to_json())
    back = ChunkCondition.from_json(json.loads(s))
    assert cond == back


def test_custom_pruned_from_combinators_on_serialize() -> None:
    cond = ChunkCondition.all_of(
        [ChunkCondition.when_mode(Mode.SAFE_AUTO), ChunkCondition.custom(lambda _c: True)]
    )
    s = json.dumps(cond.to_json())
    back = ChunkCondition.from_json(json.loads(s))
    assert back == ChunkCondition.all_of([ChunkCondition.when_mode(Mode.SAFE_AUTO)])


# ── ChunkProviderError variants ──────────────────────────────────────────────
def test_chunk_provider_error_variants() -> None:
    e = ChunkProviderError.load_failed("remote", "timeout")
    assert e.kind == "load_failed"
    assert "remote" in str(e)
    assert "timeout" in str(e)

    p = ChunkProviderError.parse_error("bad json")
    assert p.kind == "parse_error"
    assert "bad json" in str(p)


# ── build_context_sources passes guides/memory/schemas through (A5) ─────────
def test_build_context_sources_threads_block_1_and_passthrough() -> None:
    b = ContextSourcesBuilder.with_chunks(
        [
            PromptChunk.new("core", "rules"),
            PromptChunk.new("ps", "ref").with_stability(SegmentStability.PER_SESSION),
        ]
    )
    sources, buckets = b.build_context_sources(_ctx(), [], [], [])
    assert isinstance(sources, ContextSources)
    assert len(sources.composed_prompt.chunks) == 1  # only Static in Block 1
    assert sources.composed_prompt.chunks[0].id == "core"
    assert [x.id for x in buckets.per_session] == ["ps"]
    assert sources.guides == []


# ── StorageScope serde snake_case ────────────────────────────────────────────
def test_storage_scope_values() -> None:
    assert StorageScope.USER.value == "user"
    assert StorageScope.PROJECT.value == "project"
    assert StorageScope.LOCAL.value == "local"
    assert StorageScope.default() is StorageScope.PROJECT


# ── agent_affinity gate ──────────────────────────────────────────────────────
def test_agent_affinity_gate() -> None:
    chunk = PromptChunk.new("planner-prompt", "you plan").with_agent_affinity("planner")
    b = ContextSourcesBuilder.with_chunks([chunk])
    assert b.assemble(_ctx()).static_chunks == []
    c = _ctx()
    c.agent_type = "planner"
    assert len(b.assemble(c).static_chunks) == 1
    c.agent_type = "coder"
    assert b.assemble(c).static_chunks == []


# ── HarnessBuilder.chunks / chunk_provider wiring ───────────────────────────
async def test_harness_builder_chunks_inline_resolves_to_in_memory() -> None:
    # Exercise the chunk plumbing without constructing the full (heavy) builder.
    from spore_core.harness import HarnessBuilder

    hb = HarnessBuilder.__new__(HarnessBuilder)
    hb._chunk_provider = None  # type: ignore[attr-defined]
    ret = hb.chunks([PromptChunk.new("a", "1"), PromptChunk.new("b", "2")])
    assert ret is hb  # fluent
    prov = hb._chunk_provider  # type: ignore[attr-defined]
    assert isinstance(prov, InMemoryChunkProvider)
    loaded = await prov.load()
    assert [c.id for c in loaded] == ["a", "b"]


async def test_harness_builder_chunk_provider_setter() -> None:
    from spore_core.harness import HarnessBuilder

    hb = HarnessBuilder.__new__(HarnessBuilder)
    hb._chunk_provider = None  # type: ignore[attr-defined]
    explicit = EmbeddedChunkProvider([PromptChunk.new("x", "y")])
    ret = hb.chunk_provider(explicit)
    assert ret is hb
    assert hb._chunk_provider is explicit  # type: ignore[attr-defined]
    assert len(await explicit.load()) == 1


# ── Fixture replay: condition_eval.json (R1–R8) ─────────────────────────────
def test_fixture_replay_condition_eval() -> None:
    raw = (_FIXTURE_DIR / "condition_eval.json").read_text(encoding="utf-8")
    suite = json.loads(raw)
    cases = suite["cases"]
    assert len(cases) >= 8, "expected >=8 cases (R1-R8)"
    b = _builder()
    for case in cases:
        cond = ChunkCondition.from_json(case["condition"])
        ctx = AssemblyContext.from_json(case["assembly_context"])
        got = b.evaluate(cond, ctx)
        assert got == case["expected"], f"case {case['name']!r} mismatch"


# ── Fixture replay: assembly_steps.json (R10–R17) ───────────────────────────
def test_fixture_replay_assembly_steps() -> None:
    raw = (_FIXTURE_DIR / "assembly_steps.json").read_text(encoding="utf-8")
    suite = json.loads(raw)
    cases = suite["cases"]
    assert cases
    for case in cases:
        chunks = [PromptChunk.from_json(rc) for rc in case["registered_chunks"]]
        b = ContextSourcesBuilder.with_chunks(chunks)
        ctx = AssemblyContext.from_json(case["assembly_context"])
        buckets = b.assemble(ctx)
        name = case["name"]
        assert [c.id for c in buckets.static_chunks] == case["expected_static"], (
            f"case {name!r} static mismatch"
        )
        assert [c.id for c in buckets.per_session] == case["expected_per_session"], (
            f"case {name!r} per_session mismatch"
        )
        assert [c.id for c in buckets.per_turn] == case["expected_per_turn"], (
            f"case {name!r} per_turn mismatch"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
