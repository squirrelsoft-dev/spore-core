"""Standard Tool implementations (issue #5).

Each submodule houses a family of tools that conform to the
:class:`spore_core.tool_registry.Tool` Protocol. Tools are stateless and
receive a :class:`SandboxProvider` on every dispatch.

Families:

* :mod:`fs`       — ReadFile, WriteFile, ListDir, DeleteFile, MoveFile
* :mod:`exec`     — Exec, BashCommand, RunTests
* :mod:`search`   — GrepFiles, FindFiles
* :mod:`git`      — GitLog, GitDiff, GitCommit, GitStatus, GitReset
* :mod:`http`     — HttpGet, HttpPost
* :mod:`subagent` — SubagentTool (wraps a child Harness)
"""

from ._common import DEFAULT_HEAD_TOKENS, DEFAULT_TAIL_TOKENS, LARGE_OUTPUT_THRESHOLD
from .catalogue import StandardTool, StandardTools
from .control import (
    AbortTool,
    AskUserQuestionTool,
    EnterPlanModeTool,
    ExitPlanModeTool,
)
from .edit import EditFileTool
from .error import (
    ExecutionFailed,
    InvalidParameters,
    SandboxViolationError,
    Timeout,
    ToolExecutionError,
)
from .exec import BashCommandTool, ExecTool, RunTestsTool
from .fs import (
    DeleteFileTool,
    ListDirTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)
from .git import (
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitResetMode,
    GitResetTool,
    GitStatusTool,
)
from .http import HttpGetTool, HttpPostTool
from .message import SendMessageTool
from .search import FindFilesTool, GrepFilesTool, GrepTool
from .subagent import (
    BuildError,
    ContextSharing,
    ContextSharingIsolated,
    ContextSharingSharedSession,
    ContextSharingSummaryHandoff,
    SubagentTool,
)
from .tasklist import TaskListTool
from .todo import TODO_STORE_KEY, TodoWriteTool
from .web import WebFetchTool, WebSearchTool

__all__ = [
    "AbortTool",
    "AskUserQuestionTool",
    "BashCommandTool",
    "BuildError",
    "ContextSharing",
    "ContextSharingIsolated",
    "ContextSharingSharedSession",
    "ContextSharingSummaryHandoff",
    "DEFAULT_HEAD_TOKENS",
    "DEFAULT_TAIL_TOKENS",
    "DeleteFileTool",
    "EditFileTool",
    "EnterPlanModeTool",
    "ExecTool",
    "ExecutionFailed",
    "ExitPlanModeTool",
    "FindFilesTool",
    "GitCommitTool",
    "GitDiffTool",
    "GitLogTool",
    "GitResetMode",
    "GitResetTool",
    "GitStatusTool",
    "GrepFilesTool",
    "GrepTool",
    "HttpGetTool",
    "HttpPostTool",
    "InvalidParameters",
    "LARGE_OUTPUT_THRESHOLD",
    "ListDirTool",
    "MoveFileTool",
    "ReadFileTool",
    "RunTestsTool",
    "SandboxViolationError",
    "SendMessageTool",
    "StandardTool",
    "StandardTools",
    "SubagentTool",
    "TODO_STORE_KEY",
    "TaskListTool",
    "Timeout",
    "TodoWriteTool",
    "ToolExecutionError",
    "WebFetchTool",
    "WebSearchTool",
    "WriteFileTool",
]
