"""Standard Tool implementations (issue #5).

Each submodule houses a family of tools that conform to the
:class:`spore_core.tool_registry.Tool` Protocol. Tools are stateless and
receive a :class:`SandboxProvider` on every dispatch.

Families:

* :mod:`fs`       — ReadFile, WriteFile, ListDir, DeleteFile, MoveFile
* :mod:`exec`     — BashCommand, RunTests
* :mod:`search`   — GrepFiles, FindFiles
* :mod:`git`      — GitLog, GitDiff, GitCommit, GitStatus, GitReset
* :mod:`http`     — HttpGet, HttpPost
* :mod:`subagent` — SubagentTool (wraps a child Harness)
"""

from ._common import DEFAULT_HEAD_TOKENS, DEFAULT_TAIL_TOKENS, LARGE_OUTPUT_THRESHOLD
from .error import (
    ExecutionFailed,
    InvalidParameters,
    SandboxViolationError,
    Timeout,
    ToolExecutionError,
)
from .exec import BashCommandTool, RunTestsTool
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
from .search import FindFilesTool, GrepFilesTool
from .subagent import (
    BuildError,
    ContextSharing,
    ContextSharingIsolated,
    ContextSharingSharedSession,
    ContextSharingSummaryHandoff,
    SubagentTool,
)

__all__ = [
    "BashCommandTool",
    "BuildError",
    "ContextSharing",
    "ContextSharingIsolated",
    "ContextSharingSharedSession",
    "ContextSharingSummaryHandoff",
    "DEFAULT_HEAD_TOKENS",
    "DEFAULT_TAIL_TOKENS",
    "DeleteFileTool",
    "ExecutionFailed",
    "FindFilesTool",
    "GitCommitTool",
    "GitDiffTool",
    "GitLogTool",
    "GitResetMode",
    "GitResetTool",
    "GitStatusTool",
    "GrepFilesTool",
    "HttpGetTool",
    "HttpPostTool",
    "InvalidParameters",
    "LARGE_OUTPUT_THRESHOLD",
    "ListDirTool",
    "MoveFileTool",
    "ReadFileTool",
    "RunTestsTool",
    "SandboxViolationError",
    "SubagentTool",
    "Timeout",
    "ToolExecutionError",
    "WriteFileTool",
]
