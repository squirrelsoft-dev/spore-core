"""The two custom tools this example registers.

Each is a plain object that satisfies the structural
:class:`spore_core.tool_registry.Tool` protocol (``name`` / ``is_subagent_tool``
/ ``may_produce_large_output`` / ``execute``) paired with a ``schema()``
constructor. ``main`` bundles each pair in a
:class:`spore_tools.StandardTool` and hands it to the builder via ``.tool(...)``.
The harness wires the sandbox and a per-run
:class:`~spore_core.tool_registry.ToolContext` (the storage seam) in
automatically.
"""

from .recall import RecallTool
from .remember import FACT_PREFIX, RememberTool

__all__ = ["FACT_PREFIX", "RecallTool", "RememberTool"]
