"""Smoke tests for the cordyceps example's harness-native skills (#115 / SC-26).

The skill subsystem now lives in :mod:`spore_core.skills` (the architect-side
``SkillInjectingContextManager`` shim was deleted in SC-26 slice 4). The parse /
catalog behaviour is unit-tested in core; here we only assert that the bundled
``audit`` SKILL.md next to this example discovers, parses, and seeds active —
the wiring ``main.build_skill_catalog`` relies on.
"""

from __future__ import annotations

from pathlib import Path

from spore_core import SkillCatalog, parse_skill_doc

_AUDIT_SKILL_MD = Path(__file__).parent / "skills" / "audit" / "SKILL.md"


def test_bundled_audit_skill_parses() -> None:
    entry = parse_skill_doc(_AUDIT_SKILL_MD.read_text(encoding="utf-8"))
    assert entry is not None
    assert entry.name == "audit"
    assert entry.description
    assert entry.body.strip()


def test_discover_finds_and_activates_bundled_audit() -> None:
    catalog = SkillCatalog.discover([Path(__file__).parent / "skills"], Path(__file__).parent)
    assert "audit" in catalog.names()

    # Before activation: only the manifest guide.
    guides = catalog.active_guides()
    assert any(str(g.id) == "AVAILABLE SKILLS" for g in guides)
    assert all(str(g.id) != "ACTIVE SKILL — audit" for g in guides)

    # main.build_skill_catalog seeds `audit` always-active: its body becomes a
    # sticky tier-2 guide.
    assert catalog.activate("audit") is True
    guides = catalog.active_guides()
    assert any(str(g.id) == "ACTIVE SKILL — audit" for g in guides)
