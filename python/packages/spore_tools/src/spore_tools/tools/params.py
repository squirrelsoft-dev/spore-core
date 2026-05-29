"""Per-tool pydantic parameter models and a parsing helper.

Mirrors ``rust/crates/spore-core/src/tools/params.rs`` — every standard tool
deserializes its ``ToolCall.input`` dict into one of these models. Validation
failures are mapped to :class:`InvalidParameters`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


class WriteFileParams(_Params):
    path: str
    content: str
    append: bool = False


class ListDirParams(_Params):
    path: str
    recursive: bool = False


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


class FindFilesParams(_Params):
    glob: str
    path: str


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
    """``action == "add_task"`` — append a new pending task."""

    description: str


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
    "TASK_LIST_PARAMS_BY_ACTION",
    "UpdateTaskParams",
    "parse_task_list_params",
    "DeleteFileParams",
    "FindFilesParams",
    "GitCommitParams",
    "GitDiffParams",
    "GitLogParams",
    "GitResetMode",
    "GitResetParams",
    "GitStatusParams",
    "ExecParams",
    "GrepFilesParams",
    "HttpGetParams",
    "HttpPostParams",
    "ListDirParams",
    "MoveFileParams",
    "ReadFileParams",
    "RunTestsParams",
    "ShellCommandParams",
    "SubagentParams",
    "WriteFileParams",
    "parse_params",
]
