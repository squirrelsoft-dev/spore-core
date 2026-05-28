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


__all__ = [
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
