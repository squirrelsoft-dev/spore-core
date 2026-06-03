"""The two custom tools this example registers.

Each is defined with the :func:`~spore_tools.define_tool` helper: a typed
pydantic input model plus an async ``execute`` body. The helper derives the
advertised schema from the input model and returns a ready-to-register
:class:`~spore_tools.StandardTool`. ``main`` hands each to the builder via
``.tool(...)``; the harness wires the sandbox and a per-run
:class:`~spore_core.tool_registry.ToolContext` (the storage seam) in
automatically.
"""

from .recall import RecallInput, recall_tool
from .remember import FACT_PREFIX, RememberInput, remember_tool

__all__ = [
    "FACT_PREFIX",
    "RecallInput",
    "RememberInput",
    "recall_tool",
    "remember_tool",
]
