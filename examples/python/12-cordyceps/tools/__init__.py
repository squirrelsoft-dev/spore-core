"""The example's custom tools. After the #131 composition rewrite the surviving
hand-written tools are ``send_user_message`` (observability) and the #114 consult
ladder (``research_best_practices``, ``consult_advisor``) — both part of
``exec-tools``. The consult ladder is PRESERVED: the worker leaf consult now
propagates to a top-level :class:`RunResultConsult` and the host run loop
mediates it (the seam moved from the old ``SubagentTool`` to the host loop; see
``consult.py`` and ``main.py``). The #115 ``load_skill`` tool was a worker-side
per-node seam the declarative tree does not expose, so it was dropped — the
``audit`` skill instead rides the GLOBAL ``SkillInjectingContextManager``.
"""

from .consult import (
    KIND_ADVICE,
    KIND_RESEARCH,
    consult_advisor_tool,
    research_best_practices_tool,
)
from .send_message import send_user_message_tool

__all__ = [
    "KIND_ADVICE",
    "KIND_RESEARCH",
    "consult_advisor_tool",
    "research_best_practices_tool",
    "send_user_message_tool",
]
