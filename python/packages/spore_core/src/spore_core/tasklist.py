"""Issue #71 — Persisted task-list tool (PlanExecute, drives the execute loop).

Mirrors the Rust reference at ``rust/crates/spore-core/src/tasklist.rs``.

Decomposed out of #59. The accepted plan (#70) is parsed into a persisted task
list (#72), and the execute phase (#59) loops over the tasks until the list is
complete. This module owns the **task-list primitive** that holds and mutates
that list, plus its disk-persistence helpers. The single mutating tool lives in
:mod:`spore_tools.tools.tasklist`. It is consumed by #72 (which populates the
list) and #59 (whose execute loop drains it).

Types
-----
* :class:`TaskStatus` — ``pending | in_progress | completed | blocked``.
  Serializes to those exact snake_case strings. Byte-identical across all four
  languages.
* :class:`Task` — ``{id: int, description: str, status: TaskStatus}``. A flat
  record: NO hierarchy/subtasks, NO timestamps (byte-identity constraint), NO
  order field (order is positional in :attr:`TaskList.tasks`).
* :class:`TaskList` — ``{tasks: list[Task], next_id: int}``. The persisted
  collection. The default is empty with ``next_id == 1``.
* :class:`TaskListError` — domain error
  (:meth:`TaskListError.task_not_found`, :meth:`TaskListError.invalid_transition`).
  These map to a recoverable tool error at the tool boundary; the tool NEVER
  raises uncaught.

ID scheme
---------
Sequential ``int``, 1-based, assigned monotonically from
:attr:`TaskList.next_id`. Ids are never reused — ``next_id`` only ever grows,
and it is preserved across reload so a freshly-loaded list keeps minting fresh
ids.

Rules enforced
--------------
* R1  Ids are assigned 1, 2, 3, … monotonically from ``next_id``; never reused.
* R2  :meth:`TaskList.add` APPENDS to the end of ``tasks`` (positional order is
  stable).
* R3  Listing never mutates the list.
* R4  Unknown id on update/complete → :meth:`TaskListError.task_not_found`
  (recoverable at the tool boundary).
* R5  Status transitions follow the matrix in :func:`validate_transition`
  (DECISION 1, "permissive-except-terminal-Completed"). A rejected transition →
  :meth:`TaskListError.invalid_transition`.
* R6  ``completed`` is terminal: ANY transition OUT of ``completed`` is rejected
  (the idempotent ``completed → completed`` self-transition is allowed).
* R7  Self-transitions ``X → X`` are idempotent and always allowed.
* R8  Persistence is interim, through the FILESYSTEM via the
  :class:`SandboxProvider` read-modify-write at :data:`TASK_LIST_PATH`
  (DECISION 2). The tool does NOT touch ``SessionState.extras``; #59 owns the
  extras mirror.

Both design forks (transition matrix, state seam) were resolved before
implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import anyio

from .errors import SporeError
from .harness import SandboxProvider

__all__ = [
    "TASK_LIST_EXTRAS_KEY",
    "TASK_LIST_PATH",
    "Task",
    "TaskList",
    "TaskListError",
    "TaskStatus",
    "load_task_list",
    "store_task_list",
    "validate_transition",
]

#: Key under which the :class:`TaskList` is mirrored into ``SessionState.extras``
#: (serialized JSON) by the harness / #59. Mirrors ``PLAN_EXECUTE_EXTRAS_KEY``.
#: Stable across all four languages. NOTE: #71 itself does NOT write this key
#: (the ``Tool`` protocol has no ``SessionState`` access); it is the contract
#: the harness-side mirror uses.
TASK_LIST_EXTRAS_KEY = "task_list"

#: Canonical on-disk location of the persisted task list, relative to the
#: sandbox/workspace root. Resolved through :meth:`SandboxProvider.resolve_path`.
TASK_LIST_PATH = ".spore/task_list.json"


# ============================================================================
# Types
# ============================================================================


class TaskStatus(str, Enum):
    """Lifecycle status of a :class:`Task`. Serializes to its snake_case value."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """A single task: flat, no hierarchy, no timestamps, no order field (order is
    positional in :attr:`TaskList.tasks`)."""

    id: int
    description: str
    status: TaskStatus

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict with canonical field order ``id``,
        ``description``, ``status`` (matching the cross-language wire form)."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
        }

    @staticmethod
    def from_dict(value: dict[str, object]) -> Task:
        return Task(
            id=int(value["id"]),  # type: ignore[arg-type]
            description=str(value["description"]),
            status=TaskStatus(value["status"]),
        )


@dataclass
class TaskList:
    """The persisted collection of tasks plus the monotonic id counter.

    Serializes as ``{"tasks":[...],"next_id":N}``. The default (and every
    freshly-minted list) starts at ``next_id == 1``. A deserialized blob without
    ``next_id`` defaults to ``0`` (older/handwritten form), but the constructed
    default starts at ``1``.
    """

    tasks: list[Task] = field(default_factory=list)
    next_id: int = 1

    # ---- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Canonical dict form: ``tasks`` then ``next_id`` (field order is wire
        significant for byte-identity)."""
        return {
            "tasks": [t.to_dict() for t in self.tasks],
            "next_id": self.next_id,
        }

    def to_json(self) -> str:
        """Compact, byte-identical JSON: separators ``(",", ":")`` and the field
        order pinned by :meth:`to_dict`. Matches the Rust ``serde_json`` form."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @staticmethod
    def from_dict(value: dict[str, object]) -> TaskList:
        tasks_value = value.get("tasks", [])
        if not isinstance(tasks_value, list):
            raise ValueError("field `tasks` is not an array")
        tasks = [Task.from_dict(t) for t in tasks_value]
        # ``next_id`` is optional on the wire (older blob) → default 0.
        next_id = int(value.get("next_id", 0))  # type: ignore[arg-type]
        return TaskList(tasks=tasks, next_id=next_id)

    @staticmethod
    def from_json(text: str) -> TaskList:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("top-level JSON value is not an object")
        return TaskList.from_dict(value)

    # ---- mutation helpers (the seam #72 / #59 will call) ------------------

    def add(self, description: str) -> int:
        """Append a new ``pending`` task with the next sequential id and return
        that id. Increments :attr:`next_id`. R1, R2."""
        task_id = self.next_id
        self.tasks.append(Task(id=task_id, description=description, status=TaskStatus.PENDING))
        self.next_id += 1
        return task_id

    def update(
        self,
        task_id: int,
        status: TaskStatus | None = None,
        description: str | None = None,
    ) -> None:
        """Update a task's status and/or description.

        * Unknown id → raises :class:`TaskListError` (``task_not_found``).
        * ``status`` present → validated via :func:`validate_transition` then
          applied.
        * ``description`` present → set verbatim.
        * Both absent → no-op success.

        Status is validated BEFORE any field is written, so a rejected
        transition leaves the task untouched.
        """
        task = self._find(task_id)
        if status is not None:
            validate_transition(task_id, task.status, status)
            task.status = status
        if description is not None:
            task.description = description

    def complete(self, task_id: int) -> None:
        """Mark a task ``completed``, validating the transition first.

        * Unknown id → raises :class:`TaskListError` (``task_not_found``).
        * Already ``completed`` → idempotent success.
        """
        task = self._find(task_id)
        validate_transition(task_id, task.status, TaskStatus.COMPLETED)
        task.status = TaskStatus.COMPLETED

    def _find(self, task_id: int) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise TaskListError.task_not_found(task_id)


# ============================================================================
# Errors
# ============================================================================


class TaskListError(SporeError):
    """Errors raised by task-list mutations. Both variants are recoverable at the
    tool boundary.

    A single class with a ``kind`` discriminant
    (``"task_not_found"`` | ``"invalid_transition"``), mirroring the Rust
    ``TaskListError`` enum.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        id: int,
        from_status: TaskStatus | None = None,
        to_status: TaskStatus | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.id = id
        self.from_status = from_status
        self.to_status = to_status

    @staticmethod
    def task_not_found(task_id: int) -> TaskListError:
        return TaskListError(f"task not found: {task_id}", kind="task_not_found", id=task_id)

    @staticmethod
    def invalid_transition(
        task_id: int, from_status: TaskStatus, to_status: TaskStatus
    ) -> TaskListError:
        return TaskListError(
            f"invalid transition for task {task_id}: {from_status.value} -> {to_status.value}",
            kind="invalid_transition",
            id=task_id,
            from_status=from_status,
            to_status=to_status,
        )


# ============================================================================
# Transition matrix (DECISION 1)
# ============================================================================

# Allowed non-self transitions under DECISION 1
# ("permissive-except-terminal-Completed"). Self-transitions ``X → X`` are
# handled separately (always allowed). ANY transition OUT of ``completed`` is
# rejected (it is terminal).
_ALLOWED_TRANSITIONS: frozenset[tuple[TaskStatus, TaskStatus]] = frozenset(
    {
        (TaskStatus.PENDING, TaskStatus.IN_PROGRESS),
        (TaskStatus.PENDING, TaskStatus.COMPLETED),
        (TaskStatus.PENDING, TaskStatus.BLOCKED),
        (TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED),
        (TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED),
        (TaskStatus.BLOCKED, TaskStatus.IN_PROGRESS),
        (TaskStatus.BLOCKED, TaskStatus.COMPLETED),
    }
)


def validate_transition(task_id: int, from_status: TaskStatus, to_status: TaskStatus) -> None:
    """Validate a status transition under DECISION 1.

    Allowed:

    * any self-transition ``X → X`` (idempotent),
    * ``pending → in_progress | completed | blocked``,
    * ``in_progress → completed | blocked``,
    * ``blocked → in_progress | completed``.

    Rejected: ANY transition OUT of ``completed`` (it is terminal) — except the
    idempotent ``completed → completed``.

    Raises :class:`TaskListError` (``invalid_transition``) on a rejected
    transition. ``task_id`` only populates the error.
    """
    # Idempotent self-transition always allowed (incl. completed -> completed).
    if from_status == to_status:
        return
    if (from_status, to_status) in _ALLOWED_TRANSITIONS:
        return
    raise TaskListError.invalid_transition(task_id, from_status, to_status)


# ============================================================================
# Disk persistence (interim — DECISION 2, mirrors the plan.rs precedent)
# ============================================================================


async def load_task_list(sandbox: SandboxProvider) -> TaskList:
    """Load the persisted :class:`TaskList` from :data:`TASK_LIST_PATH` via the
    sandbox.

    An absent file (the expected first-run path) yields a fresh default list. A
    present-but-malformed file raises :class:`ValueError` so the caller (the
    tool boundary) can map it to a recoverable error rather than silently
    discarding state.
    """
    resolved = await sandbox.resolve_path(TASK_LIST_PATH, "read")

    def _read() -> str | None:
        try:
            return Path(resolved).read_text(encoding="utf-8")
        except OSError:
            # Absent file (or any read error) → fresh list.
            return None

    text = await anyio.to_thread.run_sync(_read)
    if text is None:
        return TaskList()
    return TaskList.from_json(text)


async def store_task_list(task_list: TaskList, sandbox: SandboxProvider) -> None:
    """Persist ``task_list`` to :data:`TASK_LIST_PATH` via the sandbox, creating
    the parent directory (``.spore/``) if needed. Serialization is the canonical
    compact form (field order ``tasks`` then ``next_id``)."""
    resolved = await sandbox.resolve_path(TASK_LIST_PATH, "write")
    payload = task_list.to_json()

    def _write() -> None:
        path = Path(resolved)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    await anyio.to_thread.run_sync(_write)
