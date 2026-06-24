"""SC-30 drift guard: the hardcoded ``READONLY_EVAL_TOOL_NAMES`` constant in
``spore_core`` must stay in lock-step with the live ``StandardTools.readonly_set()``
in ``spore_tools``.

The constant is hardcoded in ``spore_core`` (rather than imported from
``spore_tools``) because ``spore_tools`` DEPENDS ON ``spore_core`` — importing it
at core's module load would risk an import cycle, and Go (the sibling port)
cannot lazy-import at all. This test lives in ``spore_tools``, which already
imports core, so both can be referenced; it fails loudly if either side drifts.
"""

from __future__ import annotations

from spore_core.tool_registry import READONLY_EVAL_TOOL_NAMES

from spore_tools.tools.catalogue import StandardTools


def test_readonly_eval_tool_names_match_readonly_set() -> None:
    live = {t.schema.name for t in StandardTools.readonly_set()}
    assert live == READONLY_EVAL_TOOL_NAMES
