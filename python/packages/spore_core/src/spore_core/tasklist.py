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
* :class:`Task` — ``{id: int, description: str, status: TaskStatus,
  blockers: list[int]}``. A flat record: NO hierarchy/subtasks, NO timestamps
  (byte-identity constraint), NO order field (order is positional in
  :attr:`TaskList.tasks`). ``blockers`` (#118) are ids of other tasks that must
  be ``completed`` before this task runs; it is the LAST wire field and defaults
  to empty so pre-#118 blobs still load. Empty blockers ALWAYS serialize as
  ``[]`` (never dropped), mirroring ``next_id``.
* :class:`TaskList` — ``{tasks: list[Task], next_id: int}``. The persisted
  collection. The default is empty with ``next_id == 1``.
* :class:`TaskListError` — domain error
  (:meth:`TaskListError.task_not_found`, :meth:`TaskListError.invalid_transition`,
  :meth:`TaskListError.invalid_blockers`). These map to a recoverable tool error
  at the tool boundary; the tool NEVER raises uncaught.
* :class:`BlockerRejection` — why a blocker set was rejected by
  :meth:`TaskList.add`: ``self_block`` | ``unknown_id`` | ``cycle``.

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
* R9  (#118) :meth:`TaskList.add` blockers are validated BEFORE any mutation; a
  reject leaves the list untouched (mirrors :meth:`TaskList.update`). Validation:
  self-block (a blocker == the about-to-be-assigned id) →
  :meth:`TaskListError.invalid_blockers` (``self_block``); unknown id (a blocker
  matching no existing task) → (``unknown_id`` with the offending ``blocker``);
  cycle (the new edges would close a directed cycle in the blockers graph,
  checked by :func:`would_create_cycle`) → (``cycle``). Empty blockers never
  reject.
* R8  Persistence is through the storage seam (#75): the standalone
  :class:`~spore_tools.tools.TaskListTool` persists via the
  :class:`~spore_core.storage.RunStore` on the ``ToolContext``, keyed by
  :class:`SessionId` under :data:`TASK_LIST_EXTRAS_KEY`. The retired interim
  sandbox path (``.spore/task_list.json``) is GONE; with the library's default
  no-op storage a standalone tool call persists nothing across processes (an
  accepted behavior change — no migration shim). #59's execute loop shares the
  same ``RunStore`` key.

Both design forks (transition matrix, state seam) were resolved before
implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from .errors import SporeError
from .hooks import PlanArtifact

__all__ = [
    "TASK_LIST_EXTRAS_KEY",
    "BlockerRejection",
    "Task",
    "TaskList",
    "TaskListError",
    "TaskStatus",
    "plan_artifact_to_task_list",
    "validate_transition",
    "would_create_cycle",
]

#: Key under which the :class:`TaskList` is persisted in the
#: :class:`~spore_core.storage.RunStore`, keyed by :class:`SessionId`. Both the
#: harness-side #59 execute loop and the standalone
#: :class:`~spore_tools.tools.TaskListTool` (#75) share this single key, so a
#: standalone tool call and a PlanExecute run on the same session intentionally
#: share one blob. Stable across all four languages. The JSON shape is the
#: canonical serialized :class:`TaskList` (``{"tasks":[...],"next_id":N}``).
TASK_LIST_EXTRAS_KEY = "task_list"


# ============================================================================
# Types
# ============================================================================


class TaskStatus(str, Enum):
    """Lifecycle status of a :class:`Task`. Serializes to its snake_case value."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    #: Waiting on a blocker. (#118) This means BOTH "waiting on a blocker that
    #: has not yet completed" AND "a blocker failed terminally" — the status is
    #: the same in either case; the distinction (if any) lives in the scheduler,
    #: not the schema.
    BLOCKED = "blocked"


@dataclass
class Task:
    """A single task: flat, no hierarchy, no timestamps, no order field (order is
    positional in :attr:`TaskList.tasks`).

    ``blockers`` (#118) are the ids of other tasks that must be
    :attr:`TaskStatus.COMPLETED` before this task runs. The canonical wire field
    order is ``id, description, status, blockers`` (blockers LAST). ``blockers``
    defaults to empty so a pre-#118 blob without the key still deserializes (to
    an empty list); empty blockers ALWAYS serialize as ``[]`` (never dropped),
    the same treatment as :attr:`TaskList.next_id`.
    """

    id: int
    description: str
    status: TaskStatus
    blockers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict with canonical field order ``id``,
        ``description``, ``status``, ``blockers`` (matching the cross-language
        wire form). Empty ``blockers`` always serialize as ``[]``."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "blockers": list(self.blockers),
        }

    @staticmethod
    def from_dict(value: dict[str, object]) -> Task:
        # ``blockers`` is optional on the wire (pre-#118 blob) → default empty.
        blockers_value = value.get("blockers", [])
        if not isinstance(blockers_value, list):
            raise ValueError("field `blockers` is not an array")
        return Task(
            id=int(value["id"]),  # type: ignore[arg-type]
            description=str(value["description"]),
            status=TaskStatus(value["status"]),
            blockers=[int(b) for b in blockers_value],
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

    def add(self, description: str, blockers: list[int] | None = None) -> int:
        """Append a new ``pending`` task with the next sequential id and return
        that id. Increments :attr:`next_id`. R1, R2.

        Fallible since #118: ``blockers`` are validated BEFORE any mutation, so a
        rejected blocker set leaves the list completely untouched (mirroring how
        :meth:`update` validates before writing). R9. Validation order:

        1. self-block — a blocker equal to the id about to be assigned
           (``next_id``) → :meth:`TaskListError.invalid_blockers` (``self_block``).
        2. unknown id — a blocker matching no existing task id → (``unknown_id``).
        3. cycle — the new edges would close a directed cycle, checked by
           :func:`would_create_cycle` → (``cycle``).

        Empty ``blockers`` always pass (and serialize as ``[]``).
        """
        blocker_ids = list(blockers) if blockers is not None else []
        task_id = self.next_id

        for blocker in blocker_ids:
            if blocker == task_id:
                raise TaskListError.invalid_blockers(task_id, BlockerRejection.self_block())
            if not any(t.id == blocker for t in self.tasks):
                raise TaskListError.invalid_blockers(task_id, BlockerRejection.unknown_id(blocker))

        if would_create_cycle(self.tasks, task_id, blocker_ids):
            raise TaskListError.invalid_blockers(task_id, BlockerRejection.cycle())

        self.tasks.append(
            Task(
                id=task_id,
                description=description,
                status=TaskStatus.PENDING,
                blockers=blocker_ids,
            )
        )
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


@dataclass(frozen=True)
class BlockerRejection:
    """Why an ``add_task`` blockers set was rejected (#118).

    A single value type with a ``reason`` discriminant
    (``"self_block"`` | ``"unknown_id"`` | ``"cycle"``), mirroring the Rust
    ``BlockerRejection`` enum (internally tagged on ``reason``). ``blocker`` is
    populated only for ``unknown_id`` (the offending id).
    """

    #: One of ``"self_block"`` | ``"unknown_id"`` | ``"cycle"``.
    reason: str
    #: The offending blocker id (``unknown_id`` only); ``None`` otherwise.
    blocker: int | None = None

    @staticmethod
    def self_block() -> BlockerRejection:
        """A blocker referenced the id about to be assigned to this very task."""
        return BlockerRejection(reason="self_block")

    @staticmethod
    def unknown_id(blocker: int) -> BlockerRejection:
        """A blocker referenced an id that matches no existing task."""
        return BlockerRejection(reason="unknown_id", blocker=blocker)

    @staticmethod
    def cycle() -> BlockerRejection:
        """The new blocker edges would close a directed cycle in the graph."""
        return BlockerRejection(reason="cycle")

    def to_dict(self) -> dict[str, object]:
        """Canonical dict form, internally tagged on ``reason``. ``unknown_id``
        also carries the offending ``blocker``."""
        if self.reason == "unknown_id":
            return {"reason": "unknown_id", "blocker": self.blocker}
        return {"reason": self.reason}

    def __str__(self) -> str:
        if self.reason == "self_block":
            return "a task cannot block itself"
        if self.reason == "unknown_id":
            return f"unknown blocker id: {self.blocker}"
        return "blocker edges would create a cycle"


class TaskListError(SporeError):
    """Errors raised by task-list mutations. Every variant is recoverable at the
    tool boundary.

    A single class with a ``kind`` discriminant
    (``"task_not_found"`` | ``"invalid_transition"`` | ``"invalid_blockers"``),
    mirroring the Rust ``TaskListError`` enum.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        id: int,
        from_status: TaskStatus | None = None,
        to_status: TaskStatus | None = None,
        reason: BlockerRejection | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.id = id
        self.from_status = from_status
        self.to_status = to_status
        self.reason = reason

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

    @staticmethod
    def invalid_blockers(task_id: int, reason: BlockerRejection) -> TaskListError:
        return TaskListError(
            f"invalid blockers for task {task_id}: {reason}",
            kind="invalid_blockers",
            id=task_id,
            reason=reason,
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
# Cycle detection (#118)
# ============================================================================


def would_create_cycle(tasks: list[Task], new_id: int, new_blockers: list[int]) -> bool:
    """Would adding a node ``new_id`` whose outgoing blocker edges are
    ``new_blockers`` close a directed cycle in the blockers graph of ``tasks``?

    The graph is ``task -> blocker`` (a task points at each id it is blocked by).
    A cycle exists if, starting from any of the new edges' targets, a directed
    path leads back to ``new_id``. Since a single append-only :meth:`TaskList.add`
    only references EARLIER real ids, this can never actually fire in normal use;
    the helper exists as a spec acceptance criterion (#118) and is unit-tested
    directly against a hand-built cyclic graph.
    """
    stack: list[int] = list(new_blockers)
    visited: set[int] = set()
    by_id = {t.id: t for t in tasks}
    while stack:
        node = stack.pop()
        if node == new_id:
            return True
        if node in visited:
            continue
        visited.add(node)
        task = by_id.get(node)
        if task is not None:
            stack.extend(task.blockers)
    return False


# ============================================================================
# Plan → TaskList parser (issue #72; the bridge between #70 and #59)
# ============================================================================


def plan_artifact_to_task_list(artifact: PlanArtifact) -> TaskList:
    """Parse an accepted :class:`PlanArtifact` (#70) into a fresh, ready-to-persist
    :class:`TaskList` (#71).

    This is the bridge between the plan phase and the execute loop: once a plan
    is produced and accepted, its steps become the task list that #59's execute
    loop drains.

    Types bridged
    -------------
    * Input: :class:`PlanArtifact` ``{tasks: list[str], rationale: str}``.
    * Output: :class:`TaskList` ``{tasks: list[Task], next_id: int}``.

    Rules enforced
    --------------
    * One :class:`Task` per plan step, in plan order (positional, via
      :meth:`TaskList.add`).
    * Every produced task is :attr:`TaskStatus.PENDING`.
    * Step descriptions are copied VERBATIM — no trim, no normalize, no filter
      (even ``"  spaced  "`` and ``""`` are kept).
    * Ids are assigned ``1..=n`` sequentially via the :attr:`TaskList.next_id`
      scheme; ``next_id`` ends at ``n + 1``.
    * An empty plan (``tasks: []``) yields the empty default ``TaskList``
      (``{tasks: [], next_id: 1}``). That is a valid EMPTY list, not an error
      and not "immediate completion"; the execute loop (#59) decides loop
      semantics.
    * ``rationale`` is DROPPED — neither :class:`Task` nor :class:`TaskList`
      carries it.

    Pure and total: ``PlanArtifact -> TaskList``, no async, no I/O, never raises.
    The same artifact always yields the same task list, so the mapping is
    byte-identical across all four languages.

    Always builds a fresh default :class:`TaskList`; it never merges into an
    existing list (replanning is out of scope — single parse per accepted plan).
    Wiring this into the plan-acceptance seam is DEFERRED to #59's execute loop;
    #72 ships only this pure function.
    """
    task_list = TaskList()  # next_id == 1
    for step in artifact.tasks:
        # verbatim; appends pending; bumps next_id. Empty blockers can never
        # reject, so `add` never raises here and the parser stays total. (#118)
        task_list.add(step)
    return task_list
