"""``load_skill(skill_id)`` — activate a skill for the rest of the session.

This tool closes over the shared :class:`SkillCatalog` / registry, so it is a
hand-written :class:`Tool` impl rather than a :func:`define_tool` tool (the
helper produces a tool that cannot capture the registry). On execute it:

1. confirms the named skill exists in the registry (rejects unknown ids,
   recoverably, so the model can pick a real one from the manifest);
2. reads ``run_store["active_skills"]`` → ``list[str]``, appends the id
   (deduped), and writes it back;
3. returns a short confirmation.

The active set is then re-injected every turn by
:class:`SkillInjectingContextManager` — no new ``ToolOutput`` variant, all
storage-backed (issue #115 "flavor B").
"""

from __future__ import annotations

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core import StandardGuideRegistry
from spore_core.model import ToolCall
from spore_core.tool_registry import (
    ToolAnnotations,
    ToolContext,
    ToolSchema,
)
from spore_tools import StandardTool

from skills import ACTIVE_SKILLS_KEY, skill_query

#: The registered name of the tool.
NAME = "load_skill"


class _LoadSkillTool:
    """``load_skill``, holding the shared registry so it can validate ids.
    Satisfies the :class:`Tool` Protocol structurally."""

    def __init__(self, registry: StandardGuideRegistry) -> None:
        self._registry = registry

    def name(self) -> str:
        return NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        _ = sandbox
        raw = call.input.get("skill_id") if isinstance(call.input, dict) else None
        if not isinstance(raw, str) or not raw.strip():
            return ToolOutputError.error("invalid parameters: `skill_id` (string) is required")
        skill_id = raw.strip()

        # 1. Confirm the skill exists. A broad query with the id as the
        #    instruction surfaces it (and ranks it first on term overlap). Reject
        #    unknown ids recoverably so the model can choose a real one.
        try:
            guides = await self._registry.select(skill_query(skill_id))
        except Exception as e:  # noqa: BLE001 — a registry error degrades to "unknown"
            return ToolOutputError.error(f"load_skill: could not query registry: {e}")
        if not any(str(g.id) == skill_id for g in guides):
            return ToolOutputError.error(
                f"unknown skill '{skill_id}'. Pick one of the skills listed in the manifest."
            )

        # 2. Append to run_store["active_skills"] (dedup).
        run_store = ctx.run_store
        session_id = ctx.session_id
        try:
            current = await run_store.get(session_id, ACTIVE_SKILLS_KEY)
        except Exception as e:  # noqa: BLE001 — a read failure is recoverable
            return ToolOutputError.error(f"load_skill: could not read active set: {e}")
        active: list[str] = (
            [v for v in current if isinstance(v, str)] if isinstance(current, list) else []
        )
        if skill_id not in active:
            active.append(skill_id)
        try:
            await run_store.put(session_id, ACTIVE_SKILLS_KEY, active)
        except Exception as e:  # noqa: BLE001 — a write failure is recoverable
            return ToolOutputError.error(f"load_skill: could not persist active set: {e}")

        # 3. Confirm. The body is now injected every turn by the context manager,
        #    so the procedure is "active" from the next turn on.
        return ToolOutputSuccess.success(
            f"Loaded skill '{skill_id}'. Its procedure is now active — follow it."
        )


def load_skill_tool(registry: StandardGuideRegistry) -> StandardTool:
    """Build the ``load_skill`` :class:`StandardTool`, closing over the shared
    registry."""
    schema = ToolSchema(
        name=NAME,
        description=(
            "Activate a skill by id so its full procedure stays in your context for the rest of "
            "the session. Choose an id from the manifest of available skills."
        ),
        parameters={
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": 'The id (name) of the skill to activate, e.g. "audit".',
                }
            },
            "required": ["skill_id"],
        },
        annotations=ToolAnnotations(),
    )
    return StandardTool(_LoadSkillTool(registry), schema)


__all__ = ["NAME", "load_skill_tool"]
