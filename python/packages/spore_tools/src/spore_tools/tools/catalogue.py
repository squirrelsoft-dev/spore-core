"""Standard Tool Catalogue (#81): the curated set of tools an architect drops
into a harness, plus ready-made presets.

Mirrors ``rust/crates/spore-core/src/tools/catalogue.rs``.

Types
-----
* :class:`StandardTool` — a tool implementation bundled with its
  :class:`~spore_core.tool_registry.ToolSchema` so the two can never be
  separated (issue #81, Q2). ``HarnessBuilder.tool()`` destructures it.
* :class:`StandardTools` — a namespace of one constructor per catalogue tool,
  each returning a :class:`StandardTool`, plus three presets:
  :meth:`StandardTools.readonly_set`, :meth:`StandardTools.coding_set`, and
  :meth:`StandardTools.full_set`.

Catalogue tools (constructor → registered name)
-----------------------------------------------
Tier 1 (sandbox / stateless):

* :meth:`read_file` → ``read_file`` (EXISTING #5 tool)
* :meth:`write_file` → ``write_file`` (EXISTING)
* :meth:`edit_file` → ``edit_file`` (NEW)
* :meth:`list_dir` → ``list_dir`` (EXISTING)
* :meth:`grep_files` → ``grep_files`` (EXISTING)
* :meth:`grep` → ``grep`` (NEW, output modes)
* :meth:`find_files` → ``find_files`` (EXISTING)
* :meth:`bash_command` → ``bash_command`` (EXISTING)
* :meth:`send_message` → ``send_message`` (NEW)
* :meth:`web_fetch` → ``web_fetch`` (NEW)
* :meth:`web_search` → ``web_search`` (NEW)

Tier 2 (storage via ``ToolContext``):

* :meth:`todo_write` → ``todo_write`` (NEW, RunStore key ``"todo"``)
* :meth:`task_list` → ``task_list`` (EXISTING #71)
* :meth:`memory` → ``memory`` (NEW #82, scope-aware MemoryStore seam #78)

Tier 3 (escalate / clarify):

* :meth:`enter_plan_mode` → ``enter_plan_mode`` (NEW)
* :meth:`exit_plan_mode` → ``exit_plan_mode`` (NEW)
* :meth:`ask_user_question` → ``ask_user_question`` (NEW)
* :meth:`abort` → ``abort`` (NEW)

Q5 — overlap with the EXISTING #5 catalogue (NO renames)
--------------------------------------------------------
The catalogue deliberately ships NET-NEW tools ALONGSIDE the existing #5 tools,
never renaming them. Where a preset needs functionality that an existing tool
already provides, the preset REUSES the existing tool by its existing name:

* ``read_file``, ``write_file``, ``list_dir``, ``find_files``, ``grep_files``,
  ``bash_command`` are the EXISTING tools (their fixtures —
  ``fixtures/tools/param_validation.json`` — stay byte-identical).
* ``edit_file`` and ``grep`` are NEW and live ALONGSIDE ``write_file`` /
  ``grep_files``; they do NOT replace them.

Because :class:`~spore_core.tool_registry.StandardToolRegistry.register` is now
a last-wins upsert (issue #81, Q1), registering a preset and then a custom tool
of the same name lets the architect override a standard tool.

Q3 — MemoryTool (now LANDED, #82)
---------------------------------
``MemoryTool`` (``memory``) was deferred from #81 (it depended on the scoped
``MemoryStore`` seam from #78). It now ships here as a Tier-2 storage tool,
included in :meth:`coding_set` / :meth:`full_set` alongside ``task_list`` /
``todo_write``.
"""

from __future__ import annotations

from dataclasses import dataclass

from spore_core.tool_registry import Tool, ToolSchema

from .control import (
    AbortTool,
    AskUserQuestionTool,
    EnterPlanModeTool,
    ExitPlanModeTool,
)
from .edit import EditFileTool
from .exec import BashCommandTool
from .fs import ListDirTool, ReadFileTool, WriteFileTool
from .memory import MemoryTool
from .message import SendMessageTool
from .search import FindFilesTool, GrepFilesTool, GrepTool
from .tasklist import TaskListTool
from .todo import TodoWriteTool
from .web import WebFetchTool, WebSearchTool


@dataclass
class StandardTool:
    """A catalogue tool: its :class:`~spore_core.tool_registry.Tool`
    implementation bundled with its
    :class:`~spore_core.tool_registry.ToolSchema` so they can never drift apart
    (issue #81, Q2). ``HarnessBuilder.tool()`` destructures it."""

    implementation: Tool
    schema: ToolSchema


class StandardTools:
    """Namespace of catalogue-tool constructors and presets. Never
    instantiated — methods are ``@staticmethod`` (mirrors the Rust unit-struct
    namespace)."""

    # ---- Tier 1 ---------------------------------------------------------

    @staticmethod
    def read_file() -> StandardTool:
        """``read_file`` — EXISTING #5 tool (Q5 overlap: reused, not renamed)."""
        return StandardTool(ReadFileTool(), ReadFileTool.schema())

    @staticmethod
    def write_file() -> StandardTool:
        """``write_file`` — EXISTING #5 tool (Q5 overlap: reused, not renamed)."""
        return StandardTool(WriteFileTool(), WriteFileTool.schema())

    @staticmethod
    def edit_file() -> StandardTool:
        """``edit_file`` — NEW unique-match in-place edit (alongside ``write_file``)."""
        return StandardTool(EditFileTool(), EditFileTool.schema())

    @staticmethod
    def list_dir() -> StandardTool:
        """``list_dir`` — EXISTING #5 tool (Q5 overlap: reused, not renamed)."""
        return StandardTool(ListDirTool(), ListDirTool.schema())

    @staticmethod
    def grep_files() -> StandardTool:
        """``grep_files`` — EXISTING #5 tool (Q5 overlap: reused, not renamed)."""
        return StandardTool(GrepFilesTool(), GrepFilesTool.schema())

    @staticmethod
    def grep() -> StandardTool:
        """``grep`` — NEW regex search with output modes (alongside ``grep_files``)."""
        return StandardTool(GrepTool(), GrepTool.schema())

    @staticmethod
    def find_files() -> StandardTool:
        """``find_files`` — EXISTING #5 tool (Q5 overlap: reused, not renamed)."""
        return StandardTool(FindFilesTool(), FindFilesTool.schema())

    @staticmethod
    def bash_command() -> StandardTool:
        """``bash_command`` — EXISTING #5 tool (Q5 overlap: reused, not renamed)."""
        return StandardTool(BashCommandTool(), BashCommandTool.schema())

    @staticmethod
    def send_message() -> StandardTool:
        """``send_message`` — NEW; surfaces a ``StreamUserMessage`` via the loop."""
        return StandardTool(SendMessageTool(), SendMessageTool.schema())

    @staticmethod
    def web_fetch() -> StandardTool:
        """``web_fetch`` — NEW; GET a URL."""
        return StandardTool(WebFetchTool(), WebFetchTool.schema())

    @staticmethod
    def web_search() -> StandardTool:
        """``web_search`` — NEW; structured search over a configurable HTTP
        backend. The default has NO backend (calls error until one is
        configured); construct a :class:`StandardTool` over
        :meth:`WebSearchTool.with_endpoint` to wire a real backend."""
        return StandardTool(WebSearchTool(), WebSearchTool.schema())

    @staticmethod
    def web_search_with_endpoint(endpoint: str) -> StandardTool:
        """``web_search`` wired to a concrete backend endpoint. The plain
        :meth:`web_search` preset ships with no backend and errors on every
        call until one is configured; use this when you have a search endpoint
        (e.g. a Brave/Tavily-compatible URL) to POST the query to."""
        return StandardTool(WebSearchTool.with_endpoint(endpoint), WebSearchTool.schema())

    # ---- Tier 2 ---------------------------------------------------------

    @staticmethod
    def todo_write() -> StandardTool:
        """``todo_write`` — NEW; persists the todo list via RunStore key ``"todo"``."""
        return StandardTool(TodoWriteTool(), TodoWriteTool.schema())

    @staticmethod
    def task_list() -> StandardTool:
        """``task_list`` — EXISTING #71 tool (Q5 overlap: reused, not renamed)."""
        return StandardTool(TaskListTool(), TaskListTool.schema())

    @staticmethod
    def memory() -> StandardTool:
        """``memory`` — NEW #82; scope-aware read/write over the
        :class:`~spore_core.storage.MemoryStore` seam (#78)."""
        return StandardTool(MemoryTool(), MemoryTool.schema())

    # ---- Tier 3 ---------------------------------------------------------

    @staticmethod
    def enter_plan_mode() -> StandardTool:
        """``enter_plan_mode`` — NEW; escalates ``HarnessSignalEnterPlanMode``."""
        return StandardTool(EnterPlanModeTool(), EnterPlanModeTool.schema())

    @staticmethod
    def exit_plan_mode() -> StandardTool:
        """``exit_plan_mode`` — NEW; escalates ``HarnessSignalExitPlanMode { plan }``."""
        return StandardTool(ExitPlanModeTool(), ExitPlanModeTool.schema())

    @staticmethod
    def ask_user_question() -> StandardTool:
        """``ask_user_question`` — NEW; returns ``ToolOutputAwaitingClarification``."""
        return StandardTool(AskUserQuestionTool(), AskUserQuestionTool.schema())

    @staticmethod
    def abort() -> StandardTool:
        """``abort`` — NEW; escalates ``HarnessSignalAbort { reason }``."""
        return StandardTool(AbortTool(), AbortTool.schema())

    # ---- Presets --------------------------------------------------------

    @staticmethod
    def readonly_set() -> list[StandardTool]:
        """Read-only investigation set: no mutating or escalating tools. Reuses
        the EXISTING read-only #5 tools by name (Q5 overlap) plus the NEW
        ``grep``."""
        return [
            StandardTools.read_file(),
            StandardTools.list_dir(),
            StandardTools.grep_files(),
            StandardTools.grep(),
            StandardTools.find_files(),
            StandardTools.web_fetch(),
            StandardTools.web_search(),
        ]

    @staticmethod
    def coding_set() -> list[StandardTool]:
        """Coding set: everything in :meth:`readonly_set` plus the mutating
        filesystem tools, shell, messaging, and the storage-backed todo/task
        tools. Reuses EXISTING tool names on overlap (Q5)."""
        return [
            StandardTools.read_file(),
            StandardTools.write_file(),
            StandardTools.edit_file(),
            StandardTools.list_dir(),
            StandardTools.grep_files(),
            StandardTools.grep(),
            StandardTools.find_files(),
            StandardTools.bash_command(),
            StandardTools.send_message(),
            StandardTools.web_fetch(),
            StandardTools.web_search(),
            StandardTools.todo_write(),
            StandardTools.task_list(),
            StandardTools.memory(),
        ]

    @staticmethod
    def full_set() -> list[StandardTool]:
        """Full set: the :meth:`coding_set` plus every Tier-3 control tool
        (plan / clarify / abort)."""
        return [
            *StandardTools.coding_set(),
            StandardTools.enter_plan_mode(),
            StandardTools.exit_plan_mode(),
            StandardTools.ask_user_question(),
            StandardTools.abort(),
        ]


__all__ = ["StandardTool", "StandardTools"]
