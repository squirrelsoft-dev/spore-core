"""Issue #115 / SC-26 — **skills** baked into the harness.

Mirrors the Rust reference at ``rust/crates/spore-core/src/skills.rs``.

A *skill* is a directory with a ``SKILL.md``: YAML frontmatter (``name`` +
``description``, plus optional fields) followed by a markdown procedure body.
This module discovers them and exposes them to the agent with **progressive
disclosure** (the `Agent Skills spec <https://agentskills.io/specification>`_):

1. **Metadata (tier 1).** The ``name`` + ``description`` of every discovered
   skill is injected every turn as a compact manifest — cheap, always present,
   so the agent knows what exists.
2. **Instructions (tier 2).** A skill's full body is injected only once it is
   **active**, and then it stays injected every turn (sticky).

Unlike the pre-#115 architect-side shim (which injected skills as ad-hoc User
messages via a wrapping context manager), the catalog feeds the rich
:class:`~spore_core.context.ContextSources` seam:
:meth:`SkillCatalog.active_guides` returns the manifest + active bodies as
:class:`~spore_core.context.Guide`\\ s, which the harness places in the
structural System block. A skill becomes active when the agent calls the
:meth:`load_skill <SkillCatalog.load_skill_tool>` tool (or the host activates
it).

The active set is a shared :class:`set` held inside the :class:`SkillCatalog`,
so the ``load_skill`` tool and the per-turn :meth:`active_guides` read the SAME
set within a harness's lifetime (sticky for the session — the catalog is held
by both the tool and ``HarnessConfig``).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .context import Guide, GuideId
from .model import ToolCall
from .tool_registry import ToolAnnotations, ToolContext, ToolSchema

if TYPE_CHECKING:
    from .harness import SandboxProvider, ToolOutput

#: The registered name of the skill-activation tool.
LOAD_SKILL = "load_skill"


# ============================================================================
# Skill model + discovery
# ============================================================================


@dataclass
class SkillEntry:
    """One parsed skill. ``name`` is the identity (matches the skill's directory
    name per the spec); ``description`` is the one-line manifest entry (where the
    spec puts trigger keywords); ``body`` is the markdown procedure injected on
    load."""

    name: str
    description: str
    body: str


def parse_skill_doc(content: str) -> SkillEntry | None:
    """Parse a ``SKILL.md``: optional ``---``-delimited YAML frontmatter carrying
    at least ``name`` and ``description``, then the markdown body. Dependency-free
    and minimal — enough for the spec's required fields; optional fields are
    tolerated and ignored. Returns ``None`` if there is no usable ``name`` or the
    body is empty."""
    trimmed = content.lstrip()
    if trimmed.startswith("---"):
        rest = trimmed[len("---") :]
        idx = rest.find("\n---")
        if idx != -1:
            front = rest[:idx]
            body = rest[idx + 4 :].lstrip("\n\r")
        else:
            front = rest
            body = ""
    else:
        front = ""
        body = trimmed

    name = _yaml_scalar(front, "name")
    if name is None:
        return None
    name = name.strip()
    if not name or not body.strip():
        return None
    description = (_yaml_scalar(front, "description") or "").strip()
    return SkillEntry(name=name, description=description, body=body)


def _yaml_scalar(front: str, key: str) -> str | None:
    """Pull a single top-level ``key: value`` scalar from a YAML frontmatter
    block, stripping surrounding quotes. Not a general YAML parser — nested maps
    are skipped, since their indented children don't match a top-level ``key:``
    line."""
    for raw in front.splitlines():
        line = raw.strip()
        if not line.startswith(key):
            continue
        after = line[len(key) :].lstrip()
        if not after.startswith(":"):
            continue
        return _strip_quotes(after[1:].strip())
    return None


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def _scan_skill_dir(directory: Path) -> list[SkillEntry]:
    """Scan one ``skills/`` directory: each immediate ``<child>/SKILL.md`` is a
    candidate. A missing/unreadable directory or file is skipped silently."""
    out: list[SkillEntry] = []
    try:
        children = sorted(directory.iterdir())
    except OSError:
        return out
    for child in children:
        skill_md = child / "SKILL.md"
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError:
            continue
        entry = parse_skill_doc(text)
        if entry is not None:
            out.append(entry)
    return out


def _upsert(entries: list[SkillEntry], entry: SkillEntry) -> None:
    """Insert-or-replace by ``name``, so a later source overrides an earlier
    one."""
    for i, existing in enumerate(entries):
        if existing.name == entry.name:
            entries[i] = entry
            return
    entries.append(entry)


# ============================================================================
# Catalog
# ============================================================================


class SkillCatalog:
    """The discovered skill catalog plus the sticky active-skill set. Held by
    both the ``load_skill`` tool (which activates skills) and the harness (which
    reads :meth:`active_guides` each turn), so both share the same active set for
    the catalog's lifetime."""

    def __init__(self, entries: list[SkillEntry]) -> None:
        self._entries = entries
        self._active: set[str] = set()
        self._lock = threading.Lock()

    @classmethod
    def from_entries(cls, entries: list[SkillEntry] | None) -> SkillCatalog:
        """Build a catalog from already-parsed entries (sorted by name for stable
        output; later duplicates by name win)."""
        acc: list[SkillEntry] = []
        for e in entries or []:
            _upsert(acc, e)
        acc.sort(key=lambda e: e.name)
        return cls(acc)

    @classmethod
    def discover(
        cls,
        extra_dirs: list[Path] | None = None,
        workspace_root: Path | None = None,
    ) -> SkillCatalog:
        """Discover skills from ``extra_dirs`` (e.g. a host's bundled
        ``skills/``) plus the conventional ``<workspace>/.spore/skills`` and
        ``~/.spore/skills``, in that precedence order (last wins on a name
        clash). Discovery happens once."""
        dirs: list[Path] = list(extra_dirs or [])
        if workspace_root is not None:
            dirs.append(workspace_root / ".spore" / "skills")
        home = os.environ.get("HOME")
        if home:
            dirs.append(Path(home) / ".spore" / "skills")
        entries: list[SkillEntry] = []
        for directory in dirs:
            for entry in _scan_skill_dir(directory):
                _upsert(entries, entry)
        return cls.from_entries(entries)

    def names(self) -> list[str]:
        """Skill names, for host-driven activation (``/<name>``) and listing."""
        return [e.name for e in self._entries]

    def entries(self) -> list[SkillEntry]:
        """The full entries, for listing."""
        return list(self._entries)

    def is_empty(self) -> bool:
        return not self._entries

    def activate(self, name: str) -> bool:
        """Activate a skill by name so its full body is injected every turn.
        Returns ``False`` for an unknown name (no change)."""
        if not any(e.name == name for e in self._entries):
            return False
        with self._lock:
            self._active.add(name)
        return True

    def clear_active(self) -> None:
        """Deactivate every skill (e.g. on a conversation reset). The manifest
        stays; only the sticky bodies are dropped."""
        with self._lock:
            self._active.clear()

    def active(self) -> list[str]:
        """The currently active skill names (sorted)."""
        with self._lock:
            return sorted(self._active)

    def active_guides(self) -> list[Guide]:
        """The guides injected this turn (issue #115): a single manifest guide
        listing every available skill (tier 1), followed by one guide per active
        skill carrying its full body (tier 2). Empty when the catalog has no
        skills. The harness appends these to
        :attr:`~spore_core.context.ContextSources.guides` so they reach the model
        through the structural System block."""
        if not self._entries:
            return []
        out: list[Guide] = []

        manifest = (
            "Reusable procedures you can load on demand. When the user's request "
            "matches one's description, call load_skill(name) BEFORE acting, then "
            "follow the procedure it injects. Active skills' full bodies are "
            "included below and stay in context every turn:\n"
        )
        for e in self._entries:
            manifest += f"- {e.name}: {e.description}\n"
        out.append(Guide(id=GuideId("AVAILABLE SKILLS"), content=manifest))

        by_name = {e.name: e for e in self._entries}
        for name in self.active():
            e = by_name.get(name)
            if e is not None:
                out.append(Guide(id=GuideId(f"ACTIVE SKILL — {e.name}"), content=e.body))
        return out

    def load_skill_tool(self) -> _LoadSkillStandardTool:
        """Build the ``load_skill`` tool, sharing this catalog's active set. Add
        it to the harness with :meth:`HarnessBuilder.tool
        <spore_core.harness.HarnessBuilder.tool>` — or use :meth:`HarnessBuilder.skills
        <spore_core.harness.HarnessBuilder.skills>`, which registers both the
        catalog and this tool. Returns a ``StandardTool``-shaped bundle
        (``implementation`` + ``schema``) so the builder's
        ``drain_tools_into_registry`` can register it without
        :mod:`spore_core` importing :mod:`spore_tools`."""
        return _LoadSkillStandardTool(
            implementation=_LoadSkillTool(self),
            schema=ToolSchema(
                name=LOAD_SKILL,
                description=(
                    "Activate a skill by name so its full procedure stays in your "
                    "context for the rest of the session. Call this BEFORE acting when "
                    "the user's request matches a skill in the AVAILABLE SKILLS manifest."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The skill's name, exactly as listed in AVAILABLE SKILLS.",
                        }
                    },
                    "required": ["name"],
                },
                annotations=ToolAnnotations(),
            ),
        )


# ============================================================================
# load_skill tool
# ============================================================================


@dataclass
class _LoadSkillStandardTool:
    """A ``StandardTool``-shaped bundle (``implementation`` + ``schema``) so the
    harness builder's ``drain_tools_into_registry`` can destructure it exactly
    like a :class:`spore_tools.StandardTool` — without :mod:`spore_core` taking a
    dependency on :mod:`spore_tools` (which would invert the package
    dependency)."""

    implementation: _LoadSkillTool
    schema: ToolSchema


class _LoadSkillTool:
    """``load_skill(name)`` — activate a skill so its full procedure stays in
    context. Holds the shared :class:`SkillCatalog` (to mutate the active set and
    reject unknown ids recoverably). Satisfies the
    :class:`~spore_core.tool_registry.Tool` protocol structurally."""

    def __init__(self, catalog: SkillCatalog) -> None:
        self._catalog = catalog

    def name(self) -> str:
        return LOAD_SKILL

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        _ = (sandbox, ctx)
        from .harness import ToolOutputError, ToolOutputSuccess

        raw = call.input.get("name")
        name = raw.strip() if isinstance(raw, str) else ""
        if not name:
            return ToolOutputError.error("invalid parameters: `name` (string) is required")
        if not self._catalog.activate(name):
            return ToolOutputError.error(
                f"unknown skill '{name}'. Choose one listed in AVAILABLE SKILLS."
            )
        return ToolOutputSuccess.success(
            f"Loaded skill '{name}' — its full procedure is now in your context. Follow it."
        )


__all__ = [
    "LOAD_SKILL",
    "SkillCatalog",
    "SkillEntry",
    "parse_skill_doc",
]
