"""Per-tool pydantic parameter models and a parsing helper.

Mirrors ``rust/crates/spore-core/src/tools/params.rs`` — every standard tool
deserializes its ``ToolCall.input`` dict into one of these models. Validation
failures are mapped to :class:`InvalidParameters`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spore_core.hooks import PlanArtifact
from spore_core.model import ToolCall
from spore_core.tasklist import TaskStatus

from .error import InvalidParameters


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


T = TypeVar("T", bound=_Params)


def parse_params(model: type[T], call: ToolCall) -> T:
    """Parse ``call.input`` into ``model`` or raise :class:`InvalidParameters`."""

    try:
        return model.model_validate(call.input)
    except ValidationError as e:
        raise InvalidParameters(reason=str(e)) from e


# ---------- Filesystem ----------


class ReadFileParams(_Params):
    path: str
    offset: int = 1
    length: int = 0
    line_numbers: bool = False


class WriteFileParams(_Params):
    path: str
    content: str
    append: bool = False


class ListDirParams(_Params):
    path: str
    recursive: bool = False
    include_ignored: bool = False


class DeleteFileParams(_Params):
    path: str


class MoveFileParams(_Params):
    src: str
    dst: str


# ---------- Exec ----------


class ExecParams(_Params):
    """Parameters for the shell-free :class:`spore_tools.tools.exec.ExecTool`:
    a program name plus a verbatim argument vector. No shell is involved."""

    command: str
    args: list[str] = Field(default_factory=list)
    timeout: int | None = None


class ShellCommandParams(_Params):
    """Parameters for the real :class:`spore_tools.tools.exec.BashCommandTool`:
    a single shell ``script`` run via ``/bin/sh -c``, with an optional working
    directory."""

    script: str
    working_dir: str | None = None
    timeout: int | None = None


class RunTestsParams(_Params):
    command: str
    working_dir: str
    timeout: int | None = None


# ---------- Search ----------


class GrepFilesParams(_Params):
    pattern: str
    path: str
    recursive: bool = False


class GrepOutputMode(str, Enum):
    """Output mode for the net-new :class:`GrepTool` (#81)."""

    CONTENT = "content"
    FILES_WITH_MATCHES = "files_with_matches"
    COUNT = "count"


class GrepParams(_Params):
    """Parameters for the net-new :class:`GrepTool` (#81). ``output_mode``
    defaults to ``content`` (``path:line:text`` per match). ``context_lines``
    adds standard ``grep -C N`` context when > 0 (#133)."""

    pattern: str
    path: str
    recursive: bool = False
    output_mode: GrepOutputMode = GrepOutputMode.CONTENT
    context_lines: int = 0  # lines of context before/after each match (default 0)


class FindFilesParams(_Params):
    glob: str
    path: str


# ---------- EditFile (#81) ----------


class EditFileParams(_Params):
    path: str
    old_string: str
    new_string: str


# ---------- SendMessage (#81) ----------


class SendMessageParams(_Params):
    content: str


# ---------- Web (#81) ----------


class WebFetchParams(_Params):
    url: str


class WebSearchParams(_Params):
    query: str


# ---------- TodoWrite (#81) ----------


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TodoItem(_Params):
    content: str
    status: TodoStatus


class TodoWriteParams(_Params):
    todos: list[TodoItem]


# ---------- Tier-3 control tools (#81) ----------


class EnterPlanModeParams(_Params):
    """``enter_plan_mode`` — context is optional (defaults to empty)."""

    context: str = ""


class ExitPlanModeParams(_Params):
    """``exit_plan_mode`` — the agent-constructed plan deserializes DIRECTLY
    into the existing :class:`~spore_core.hooks.PlanArtifact` (issue #81, Q4a —
    no stub). ``PlanArtifact.rationale`` defaults to ``""``."""

    plan: PlanArtifact


class AskUserQuestionParams(_Params):
    question: str
    options: list[str] | None = None


class AbortParams(_Params):
    reason: str


# ---------- Git ----------


class GitResetMode(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    MIXED = "mixed"


class GitLogParams(_Params):
    n: int = 20
    format: str = "oneline"


class GitDiffParams(_Params):
    from_: str | None = Field(default=None, alias="from")
    to: str | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class GitCommitParams(_Params):
    message: str
    files: list[str] = Field(default_factory=list)


class GitStatusParams(_Params):
    pass


class GitResetParams(_Params):
    target: str
    mode: GitResetMode


# ---------- HTTP ----------


class HttpGetParams(_Params):
    url: str
    headers: dict[str, Any] | None = None


class HttpPostParams(_Params):
    url: str
    body: Any
    headers: dict[str, Any] | None = None


# ---------- Subagent ----------


class SubagentParams(_Params):
    instruction: str


# ---------- TaskList (#71) ----------


class AddTaskParams(_Params):
    """``action == "add_task"`` — append a new pending task.

    ``blockers`` (#118) are ids of tasks that must be ``completed`` before this
    one runs. Optional; defaults to empty. Validated by
    :meth:`~spore_core.tasklist.TaskList.add`.
    """

    description: str
    blockers: list[int] = Field(default_factory=list)


class UpdateTaskParams(_Params):
    """``action == "update_task"`` — change status and/or description by id."""

    id: int
    status: TaskStatus | None = None
    description: str | None = None


class CompleteTaskParams(_Params):
    """``action == "complete_task"`` — mark a task completed by id."""

    id: int


class ListTasksParams(_Params):
    """``action == "list_tasks"`` — return the current list (no fields)."""


#: Maps the ``action`` discriminator to its per-action parameter model. Mirrors
#: the Rust internally-tagged ``TaskListParams`` enum.
TASK_LIST_PARAMS_BY_ACTION: dict[str, type[_Params]] = {
    "add_task": AddTaskParams,
    "update_task": UpdateTaskParams,
    "complete_task": CompleteTaskParams,
    "list_tasks": ListTasksParams,
}


# ---------- Memory (#82) ----------


class MemoryWriteParams(_Params):
    """``operation == "write"`` — append one entry to ``scope``.

    ``scope`` is a free ``str`` here (not the :class:`StorageScope` enum) so that
    a ``"local"`` scope deserializes successfully and reaches the tool body,
    where it is rejected at runtime with a recoverable error (the advertised
    schema enum omits ``local``). ``metadata`` is optional and defaults to ``{}``
    (decision C); the :class:`~spore_core.storage.MemoryEntry` type is unchanged.
    """

    scope: str
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryReadParams(_Params):
    """``operation == "read"`` — read from ``scope`` (or the merged view).

    ``merged`` defaults to ``False``; ``limit`` defaults to ``50`` (decision B).
    ``scope`` is a free ``str`` for the same runtime-rejection reason as
    :class:`MemoryWriteParams`."""

    scope: str
    merged: bool = False
    limit: int = 50


#: Maps the ``operation`` discriminator to its per-operation parameter model.
#: Mirrors the Rust internally-tagged ``MemoryToolParams`` enum.
MEMORY_PARAMS_BY_OPERATION: dict[str, type[_Params]] = {
    "write": MemoryWriteParams,
    "read": MemoryReadParams,
}


def parse_memory_params(call: ToolCall) -> tuple[str, _Params]:
    """Parse the ``operation`` discriminator and dispatch to the matching model.

    Returns ``(operation, params)``. Raises :class:`InvalidParameters` if the
    input is not an object, ``operation`` is missing/unknown, or the
    per-operation fields fail validation. The ``operation`` field is stripped
    before per-operation validation so the ``extra="forbid"`` models do not
    reject it.
    """
    input_ = call.input
    if not isinstance(input_, dict):
        raise InvalidParameters(reason="input must be a JSON object")
    operation = input_.get("operation")
    if not isinstance(operation, str):
        raise InvalidParameters(reason="missing or non-string `operation`")
    model = MEMORY_PARAMS_BY_OPERATION.get(operation)
    if model is None:
        raise InvalidParameters(reason=f"unknown operation `{operation}`")
    fields = {k: v for k, v in input_.items() if k != "operation"}
    try:
        params = model.model_validate(fields)
    except ValidationError as e:
        raise InvalidParameters(reason=str(e)) from e
    return operation, params


def parse_task_list_params(call: ToolCall) -> tuple[str, _Params]:
    """Parse the ``action`` discriminator and dispatch to the matching model.

    Returns ``(action, params)``. Raises :class:`InvalidParameters` if the input
    is not an object, ``action`` is missing/unknown, or the per-action fields
    fail validation. The ``action`` field is stripped before per-action
    validation so the ``extra="forbid"`` models do not reject it.
    """
    input_ = call.input
    if not isinstance(input_, dict):
        raise InvalidParameters(reason="input must be a JSON object")
    action = input_.get("action")
    if not isinstance(action, str):
        raise InvalidParameters(reason="missing or non-string `action`")
    model = TASK_LIST_PARAMS_BY_ACTION.get(action)
    if model is None:
        raise InvalidParameters(reason=f"unknown action `{action}`")
    fields = {k: v for k, v in input_.items() if k != "action"}
    try:
        params = model.model_validate(fields)
    except ValidationError as e:
        raise InvalidParameters(reason=str(e)) from e
    return action, params


__all__ = [
    "AddTaskParams",
    "CompleteTaskParams",
    "ListTasksParams",
    "MEMORY_PARAMS_BY_OPERATION",
    "MemoryReadParams",
    "MemoryWriteParams",
    "TASK_LIST_PARAMS_BY_ACTION",
    "UpdateTaskParams",
    "parse_memory_params",
    "parse_task_list_params",
    "AbortParams",
    "AskUserQuestionParams",
    "DeleteFileParams",
    "EditFileParams",
    "EnterPlanModeParams",
    "ExitPlanModeParams",
    "FindFilesParams",
    "GitCommitParams",
    "GitDiffParams",
    "GitLogParams",
    "GitResetMode",
    "GitResetParams",
    "GitStatusParams",
    "ExecParams",
    "GrepFilesParams",
    "GrepOutputMode",
    "GrepParams",
    "HttpGetParams",
    "HttpPostParams",
    "ListDirParams",
    "MoveFileParams",
    "ReadFileParams",
    "RunTestsParams",
    "SendMessageParams",
    "ShellCommandParams",
    "SubagentParams",
    "TodoItem",
    "TodoStatus",
    "TodoWriteParams",
    "WebFetchParams",
    "WebSearchParams",
    "WriteFileParams",
    "parse_params",
]
