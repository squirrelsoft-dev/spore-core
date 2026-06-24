"""Unit tests for the harness-native skill subsystem (issue #115 / SC-26).

Mirrors the Rust reference's ``skills.rs`` tests: SKILL.md parse/tolerate/reject,
``SkillCatalog.activate`` (sticky + unknown rejected), and ``active_guides``
(manifest tier-1 then active-body tier-2, in order).
"""

from __future__ import annotations

from spore_core import (
    SkillCatalog,
    SkillEntry,
    parse_skill_doc,
)


# ---------------------------------------------------------------------------
# SKILL.md parse
# ---------------------------------------------------------------------------


def test_parses_frontmatter_name_description_body() -> None:
    doc = (
        "---\nname: security-review\ndescription: Review code for security issues.\n"
        "---\n\n# Procedure\n\nDo the thing.\n"
    )
    entry = parse_skill_doc(doc)
    assert entry is not None
    assert entry.name == "security-review"
    assert entry.description == "Review code for security issues."
    assert entry.body == "# Procedure\n\nDo the thing.\n"


def test_tolerates_optional_frontmatter_and_strips_quotes() -> None:
    doc = (
        "---\nname: \"pdf\"\ndescription: 'Handle PDFs.'\nlicense: Apache-2.0\n"
        "metadata:\n  author: me\n---\nbody\n"
    )
    entry = parse_skill_doc(doc)
    assert entry is not None
    assert entry.name == "pdf"
    assert entry.description == "Handle PDFs."
    assert entry.body == "body\n"


def test_rejects_missing_name_or_empty_body_or_no_frontmatter() -> None:
    assert parse_skill_doc("---\ndescription: no name\n---\nbody\n") is None
    assert parse_skill_doc("---\nname: x\ndescription: d\n---\n   \n") is None
    assert parse_skill_doc("no frontmatter at all") is None


# ---------------------------------------------------------------------------
# Catalog: active_guides + activate
# ---------------------------------------------------------------------------


def test_empty_catalog_yields_no_guides() -> None:
    cat = SkillCatalog.from_entries([])
    assert cat.is_empty()
    assert cat.active_guides() == []


def test_manifest_guide_then_active_body_guides() -> None:
    cat = SkillCatalog.from_entries(
        [
            SkillEntry(name="audit", description="Audit a module.", body="AUDIT BODY"),
            SkillEntry(name="style", description="Style guide.", body="STYLE BODY"),
        ]
    )

    # Before activation: only the manifest guide.
    guides = cat.active_guides()
    assert len(guides) == 1
    assert str(guides[0].id) == "AVAILABLE SKILLS"
    assert "- audit: Audit a module." in guides[0].content
    assert "- style: Style guide." in guides[0].content

    # Unknown activation is rejected; known activation is sticky.
    assert cat.activate("nope") is False
    assert cat.activate("audit") is True
    guides = cat.active_guides()
    assert len(guides) == 2
    assert str(guides[1].id) == "ACTIVE SKILL — audit"
    assert guides[1].content == "AUDIT BODY"

    # Idempotent / sticky.
    assert cat.activate("audit") is True
    assert len(cat.active_guides()) == 2
    assert cat.active() == ["audit"]


def test_clear_active_drops_bodies_but_keeps_manifest() -> None:
    cat = SkillCatalog.from_entries(
        [SkillEntry(name="audit", description="Audit a module.", body="AUDIT BODY")]
    )
    assert cat.activate("audit") is True
    assert len(cat.active_guides()) == 2
    cat.clear_active()
    guides = cat.active_guides()
    assert len(guides) == 1
    assert str(guides[0].id) == "AVAILABLE SKILLS"


def test_from_entries_dedups_last_wins_and_sorts() -> None:
    cat = SkillCatalog.from_entries(
        [
            SkillEntry(name="b", description="first b", body="B1"),
            SkillEntry(name="a", description="a", body="A"),
            SkillEntry(name="b", description="second b", body="B2"),
        ]
    )
    assert cat.names() == ["a", "b"]
    # Last duplicate wins.
    by_name = {e.name: e for e in cat.entries()}
    assert by_name["b"].body == "B2"


# ---------------------------------------------------------------------------
# load_skill tool: activates the shared set; rejects unknown / empty name
# ---------------------------------------------------------------------------


async def test_load_skill_tool_activates_shared_set() -> None:
    from spore_core import LOAD_SKILL
    from spore_core.harness import ToolOutputError, ToolOutputSuccess
    from spore_core.model import ToolCall

    cat = SkillCatalog.from_entries(
        [SkillEntry(name="audit", description="Audit.", body="AUDIT BODY")]
    )
    bundle = cat.load_skill_tool()
    assert bundle.schema.name == LOAD_SKILL
    assert bundle.implementation.name() == LOAD_SKILL
    tool = bundle.implementation

    # Empty name → recoverable error, no activation.
    out = await tool.execute(ToolCall(id="c1", name=LOAD_SKILL, input={}), None, None)  # type: ignore[arg-type]
    assert isinstance(out, ToolOutputError)
    assert cat.active() == []

    # Unknown name → recoverable error, no activation.
    out = await tool.execute(
        ToolCall(id="c2", name=LOAD_SKILL, input={"name": "nope"}),
        None,
        None,  # type: ignore[arg-type]
    )
    assert isinstance(out, ToolOutputError)
    assert cat.active() == []

    # Known name → success; the SHARED active set now reflects it.
    out = await tool.execute(
        ToolCall(id="c3", name=LOAD_SKILL, input={"name": "audit"}),
        None,
        None,  # type: ignore[arg-type]
    )
    assert isinstance(out, ToolOutputSuccess)
    assert cat.active() == ["audit"]
    assert any(str(g.id) == "ACTIVE SKILL — audit" for g in cat.active_guides())


def test_discover_reads_skill_dirs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bundled = tmp_path / "bundled"
    (bundled / "audit").mkdir(parents=True)
    (bundled / "audit" / "SKILL.md").write_text(
        "---\nname: audit\ndescription: Audit a module.\n---\nAUDIT BODY\n",
        encoding="utf-8",
    )
    cat = SkillCatalog.discover([bundled], tmp_path)
    assert cat.names() == ["audit"]
    assert cat.entries()[0].body == "AUDIT BODY\n"
