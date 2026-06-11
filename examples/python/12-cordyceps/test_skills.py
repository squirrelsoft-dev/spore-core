"""Unit tests for the architect-side skill machinery in :mod:`skills`.

Mirrors the Rust reference's ``skills.rs`` tests: the manifest is always
injected, an inactive skill's body is NOT, and activating a skill (as the
``load_skill`` tool does, by writing ``run_store["active_skills"]``) makes its
body appear on the next ``assemble``. Plus the frontmatter parser's two rules.
"""

from __future__ import annotations

from spore_core import (
    Context,
    HarnessToolResult,
    ReactConfig,
    SessionId,
    SessionState,
    Task,
    TextContent,
)
from spore_core.storage import InMemoryStorageProvider, StorageProvider

from skills import (
    ACTIVE_SKILLS_KEY,
    SkillEntry,
    SkillInjectingContextManager,
    parse_skill_doc,
)


class _PassThroughInner:
    """A minimal pass-through inner CM (the harness ``StandardCompactionAdapter``
    behaves the same way for ``assemble``: a pass-through of session messages)."""

    async def assemble(self, session: SessionState, task: Task) -> Context:
        _ = task
        return Context(messages=list(session.messages), tools=[])

    async def append_tool_result(self, session: SessionState, result: HarnessToolResult) -> None:
        _ = (session, result)

    async def append_user_message(self, session: SessionState, text: str) -> None:
        _ = (session, text)


def _manifest() -> list[SkillEntry]:
    return [
        SkillEntry(
            name="audit",
            description="Audit one Rust module for real, actionable defects.",
            body="GREP-FIRST PROCEDURE BODY",
        ),
        SkillEntry(name="other", description="Some other skill.", body="OTHER BODY"),
    ]


def _text_of(context: Context) -> str:
    parts: list[str] = []
    for m in context.messages:
        if isinstance(m.content, TextContent):
            parts.append(m.content.text)
    return "\n".join(parts)


async def test_manifest_always_injected_bodies_only_when_active() -> None:
    storage = StorageProvider.single(InMemoryStorageProvider())
    cm = SkillInjectingContextManager(_PassThroughInner(), storage.run(), _manifest())

    session = SessionState()
    task = Task.new("audit a module", SessionId("sess-1"), ReactConfig.per_loop(8))

    # No active skills yet: manifest present, NO body.
    ctx = await cm.assemble(session, task)
    body = _text_of(ctx)
    assert "AVAILABLE SKILLS" in body, "manifest must be injected"
    assert "audit: Audit one Rust module" in body
    assert "other: Some other skill" in body
    assert "GREP-FIRST PROCEDURE BODY" not in body, "inactive skill body must NOT be injected"

    # Activate `audit` (as the load_skill tool does) → body appears next turn.
    await storage.run().put(SessionId("sess-1"), ACTIVE_SKILLS_KEY, ["audit"])

    ctx = await cm.assemble(session, task)
    body = _text_of(ctx)
    assert "AVAILABLE SKILLS" in body, "manifest still present"
    assert "ACTIVE SKILL — audit" in body, "active skill body must be injected"
    assert "GREP-FIRST PROCEDURE BODY" in body
    assert "OTHER BODY" not in body, "only the active skill's body is injected"


def test_parses_frontmatter_name_and_description() -> None:
    doc = "---\nname: audit\ndescription: Audit one module.\n---\n\n# Body\nprocedure"
    entry = parse_skill_doc(doc)
    assert entry is not None
    assert entry.name == "audit"
    assert entry.description == "Audit one module."
    assert "procedure" in entry.body


def test_rejects_missing_name_or_empty_body() -> None:
    assert parse_skill_doc("---\ndescription: x\n---\nbody") is None
    assert parse_skill_doc("---\nname: audit\n---\n") is None
