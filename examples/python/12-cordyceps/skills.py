"""Architect-side skill injection (zero core-harness change).

After the #131 composition rewrite the ``load_skill`` worker-side tool was
dropped — the declarative tree exposes no per-node tool seam. The ``audit``
skill is KEPT, now riding the single GLOBAL
:class:`SkillInjectingContextManager` (the harness's ``context_manager``), seeded
ALWAYS-ACTIVE at startup. Its procedure reaches the model structurally every
turn, compaction-proof, with no ``load_skill`` round-trip.

Why this lives in the example, not the harness
----------------------------------------------

Issue #9 added ``GuideType.SKILL`` to the :class:`GuideRegistry`, and the rich
:class:`spore_core.context.StandardContextManager.assemble` knows how to inject
skills as a Block-3 segment. But the **live** harness loop does not call that
rich ``assemble`` — it calls :meth:`StandardCompactionAdapter.assemble`, a
pass-through of ``session.messages`` (see issue #115 / Known Deviation #8). So
today a skill can reach the model only as a tool-result message or via a custom
context manager — the route this example takes:

1. A :class:`SkillCatalog` scans ``.spore/skills/{name}/SKILL.md`` (project) then
   ``~/.spore/skills/{name}/SKILL.md`` (user) plus the bundled ``audit`` skill,
   parses YAML frontmatter ``{name, description}`` + markdown body, and keeps the
   manifest side-list of ``(name, description, body)`` the example owns (both the
   manifest text and the injected body come from this one list, so they agree).
2. :class:`SkillInjectingContextManager` wraps the standard compaction adapter
   and, in ``assemble``, prepends — **ephemerally**, never into
   ``session.messages`` — (a) the manifest of all skills, and (b) the full body
   of every id in ``run_store["active_skills"]``. Everything else delegates
   verbatim to the inner adapter.

Net effect: the manifest is present every turn; a loaded skill's body is
re-injected every turn until the session is cleared. Because the active set lives
in ``run_store`` (not the message history), it is compaction-proof. ``main.py``
seeds ``["audit"]`` ALWAYS-ACTIVE at startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from spore_core import (
    Context,
    GuideQuery,
    GuideType,
    HarnessToolResult,
    Message,
    Role,
    SessionState,
    Task,
    TextContent,
)
from spore_core.harness import ContextManager as HarnessContextManager
from spore_core.storage import RunStore

#: The run-store key under which the global context manager reads the active
#: skill id set (``main.py`` seeds ``["audit"]`` at startup).
ACTIVE_SKILLS_KEY = "active_skills"


@dataclass
class SkillEntry:
    """One parsed skill: its id (== frontmatter ``name``), the one-line
    description for the manifest, and the markdown body injected when active."""

    name: str
    description: str
    body: str


def parse_skill_doc(content: str) -> SkillEntry | None:
    """Parse a ``SKILL.md``: a ``---``-delimited YAML frontmatter block carrying
    ``name:`` and ``description:``, followed by the markdown body. Minimal,
    dependency-free parsing — the example owns this until #115's filesystem
    registry productionizes it. Returns ``None`` if there is no usable name or
    the body is empty."""
    trimmed = content.lstrip()
    if trimmed.startswith("---"):
        rest = trimmed[len("---") :]
        # Split the frontmatter block off at the closing ``---``.
        front, _, body = rest.partition("\n---")
        body = body.lstrip("\n")
        name = _yaml_scalar(front, "name")
        description = _yaml_scalar(front, "description") or ""
    else:
        name = None
        description = ""
        body = trimmed

    if name is None or not name.strip() or not body.strip():
        return None
    return SkillEntry(name=name.strip(), description=description.strip(), body=body)


def _yaml_scalar(front: str, key: str) -> str | None:
    """Pull a single ``key: value`` scalar out of a YAML frontmatter block.
    Strips surrounding quotes. Good enough for the ``{name, description}``
    contract; not a general YAML parser."""
    for line in front.splitlines():
        line = line.strip()
        if line.startswith(key):
            after = line[len(key) :].lstrip()
            if after.startswith(":"):
                value = after[1:].strip().strip('"').strip("'")
                return value
    return None


def _scan_skill_dir(directory: Path) -> list[SkillEntry]:
    """Scan one ``skills/`` directory: each ``{name}/SKILL.md`` is a candidate."""
    out: list[SkillEntry] = []
    try:
        children = sorted(directory.iterdir())
    except OSError:
        return out
    for child in children:
        skill_md = child / "SKILL.md"
        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError:
            continue
        entry = parse_skill_doc(content)
        if entry is not None:
            out.append(entry)
    return out


def _upsert(manifest: list[SkillEntry], entry: SkillEntry) -> None:
    """Insert-or-replace by ``name`` so later sources override earlier ones."""
    for i, existing in enumerate(manifest):
        if existing.name == entry.name:
            manifest[i] = entry
            return
    manifest.append(entry)


class SkillCatalog:
    """The example's skill catalog: the manifest side-list the example owns
    (``{name, description, body}``). After the #131 rewrite the catalog feeds
    only the GLOBAL :class:`SkillInjectingContextManager` (the ``load_skill`` tool
    + ``GuideRegistry`` seam were dropped — see the README's "what changed"
    note); the manifest text and the injected body always agree because both come
    from this one side-list."""

    def __init__(self, manifest: list[SkillEntry]) -> None:
        self._manifest = manifest

    @classmethod
    async def bootstrap(cls, project_root: Path, bundled_audit: str) -> SkillCatalog:
        """Scan the project + user skill directories plus the bundled ``audit``
        skill so the example is self-contained even with an empty
        ``.spore/skills/``. Project entries win over user entries (last-wins by
        name)."""
        manifest: list[SkillEntry] = []

        # 1. Bundled audit skill — always present.
        bundled = parse_skill_doc(bundled_audit)
        if bundled is not None:
            _upsert(manifest, bundled)

        # 2. Project skills: `.spore/skills/{name}/SKILL.md` relative to cwd.
        for entry in _scan_skill_dir(project_root / ".spore" / "skills"):
            _upsert(manifest, entry)

        # 3. User skills: `~/.spore/skills/{name}/SKILL.md`.
        home = os.environ.get("HOME")
        if home:
            for entry in _scan_skill_dir(Path(home) / ".spore" / "skills"):
                _upsert(manifest, entry)

        return cls(manifest)

    def manifest(self) -> list[SkillEntry]:
        """The manifest side-list — handed to the context manager so it can
        render ``name: description`` lines and resolve active bodies."""
        return list(self._manifest)


class SkillInjectingContextManager:
    """A harness :class:`ContextManager` that wraps the standard compaction
    adapter and injects the skill manifest + active skill bodies each turn. ALL
    non-``assemble`` methods delegate verbatim to the inner adapter — only
    ``assemble`` is overridden, and even there the base context is produced by
    the inner adapter first.

    Satisfies the harness ``ContextManager`` Protocol structurally; it does not
    inherit it (per the Python conventions)."""

    def __init__(
        self,
        inner: HarnessContextManager,
        run_store: RunStore,
        manifest: list[SkillEntry],
    ) -> None:
        self._inner = inner
        self._run_store = run_store
        self._manifest = manifest

    async def _active_skills(self, session_id: object) -> list[str]:
        """Read the active-skill id set from ``run_store["active_skills"]``.
        Absent / malformed ⇒ empty (the manifest is still injected)."""
        try:
            value = await self._run_store.get(session_id, ACTIVE_SKILLS_KEY)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 — a storage read failure degrades to no active skills
            return []
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str)]
        return []

    def _injected_messages(self, active: list[str]) -> list[Message]:
        """Render the leading injected messages: a manifest segment (always) plus
        one body segment per active skill (progressive disclosure). Returned as
        ``USER`` messages so the loop still inserts the operating system prompt
        ahead of them at position 0."""
        out: list[Message] = []

        manifest_lines = [
            "AVAILABLE SKILLS (an ACTIVE skill's full procedure stays in your context "
            "every turn):"
        ]
        for entry in self._manifest:
            manifest_lines.append(f"- {entry.name}: {entry.description}")
        out.append(
            Message(role=Role.USER, content=TextContent(text="\n".join(manifest_lines) + "\n"))
        )

        by_name = {e.name: e for e in self._manifest}
        for skill_id in active:
            entry = by_name.get(skill_id)
            if entry is not None:
                out.append(
                    Message(
                        role=Role.USER,
                        content=TextContent(text=f"ACTIVE SKILL — {entry.name}:\n\n{entry.body}"),
                    )
                )
        return out

    # ---- ContextManager Protocol: only ``assemble`` is overridden -------

    async def assemble(self, session: SessionState, task: Task) -> Context:
        context = await self._inner.assemble(session, task)
        active = await self._active_skills(task.session_id)
        injected = self._injected_messages(active)
        context.messages = injected + list(context.messages)
        return context

    async def append_tool_result(self, session: SessionState, result: HarnessToolResult) -> None:
        await self._inner.append_tool_result(session, result)

    async def append_assistant_message(self, session: SessionState, message: Message) -> None:
        await self._inner.append_assistant_message(session, message)

    async def append_user_message(self, session: SessionState, text: str) -> None:
        await self._inner.append_user_message(session, text)

    def should_compact(self, session: SessionState) -> bool:
        return self._inner.should_compact(session)

    def prepare_compaction_turn(self, session: SessionState):  # type: ignore[no-untyped-def]
        return self._inner.prepare_compaction_turn(session)

    def inject_missing_items(self, context: Context, missing: list[str]) -> None:
        inject = getattr(self._inner, "inject_missing_items", None)
        if inject is not None:
            inject(context, missing)

    def apply_compaction(self, session: SessionState, summary: str) -> None:
        self._inner.apply_compaction(session, summary)

    def token_budget_used(self, session: SessionState) -> int | None:
        return self._inner.token_budget_used(session)


# A skill id known to the registry can be confirmed without re-querying via a
# broad ``GuideQuery`` (the ``load_skill`` tool uses this to reject unknown ids).
def skill_query(skill_id: str) -> GuideQuery:
    """Build a :class:`GuideQuery` that surfaces the named skill (filtered to
    skill-type guides). Used by ``load_skill`` to validate an id."""
    return GuideQuery(task_instruction=skill_id, guide_types=[GuideType.SKILL])


__all__ = [
    "ACTIVE_SKILLS_KEY",
    "SkillCatalog",
    "SkillEntry",
    "SkillInjectingContextManager",
    "parse_skill_doc",
    "skill_query",
]
