"""Unit tests for the #126 DAG scheduling helpers + Tier-2 step ledger on
:mod:`spore_core.tasklist`.

Pure, deterministic, no I/O — these mirror the Rust reference's tasklist tests
(``rust/crates/spore-core/src/tasklist.rs``) for ``next_ready``,
``transitive_blockers``, ``transitive_dependents``, ``has_cycle``, and the
bounded drop-oldest step ledger (decision B). The byte-identity ground truth for
the executor wiring lives in the fixture-replay tests.
"""

from __future__ import annotations

import warnings

from spore_core.tasklist import (
    STEP_LEDGER_ELISION_MARKER,
    STEP_LEDGER_MAX_ENTRIES,
    PlanArtifact,
    StepLedgerEntry,
    Task,
    TaskList,
    TaskStatus,
    plan_artifact_to_task_list,
    push_step_ledger,
    render_step_ledger,
)


def _list(*blockers_per_task: list[int]) -> TaskList:
    """Build a TaskList where task i (1-based) has the given blockers."""
    tl = TaskList()
    for i, blockers in enumerate(blockers_per_task, start=1):
        tl.add(f"task {i}", blockers)
    return tl


# ---------------------------------------------------------------------------
# next_ready — lowest-id pending task whose blockers are all completed.
# ---------------------------------------------------------------------------


def test_next_ready_picks_lowest_id_among_ready() -> None:
    # No blockers: every task is ready → lowest id wins (1).
    tl = _list([], [], [])
    assert tl.next_ready() == 1


def test_next_ready_respects_blockers() -> None:
    # 1, 2->1, 3->1, 4->2,3. Only 1 is ready initially.
    tl = _list([], [1], [1], [2, 3])
    assert tl.next_ready() == 1
    tl.complete(1)
    # 2 and 3 are now ready → lowest (2) wins (NOT positional — id tiebreak).
    assert tl.next_ready() == 2
    tl.complete(2)
    assert tl.next_ready() == 3
    tl.complete(3)
    assert tl.next_ready() == 4
    tl.complete(4)
    assert tl.next_ready() is None


def test_next_ready_skips_non_pending() -> None:
    tl = _list([], [])
    tl.update(1, TaskStatus.IN_PROGRESS)
    # 1 is in_progress (not pending) → 2 is the next ready.
    assert tl.next_ready() == 2


def test_next_ready_id_tiebreak_ignores_positional_order() -> None:
    # Hand-build a list where a higher positional index has a lower id is not
    # possible via add (ids are monotonic), but the tiebreak is over id: with
    # 1, 2 both ready, 1 wins regardless.
    tl = _list([], [])
    assert tl.next_ready() == 1


# ---------------------------------------------------------------------------
# transitive_blockers / transitive_dependents.
# ---------------------------------------------------------------------------


def test_transitive_blockers_closure_sorted_excludes_self() -> None:
    # 1, 2->1, 3->2, 4 (indep). transitive_blockers(3) = {1, 2}.
    tl = _list([], [1], [2], [])
    assert tl.transitive_blockers(3) == [1, 2]
    assert tl.transitive_blockers(1) == []
    assert tl.transitive_blockers(4) == []


def test_transitive_blockers_excludes_independent_branch() -> None:
    # Diamond: 1, 2->1, 3->1, 4->2,3. An independent 5 never appears.
    tl = _list([], [1], [1], [2, 3], [])
    assert tl.transitive_blockers(4) == [1, 2, 3]
    assert 5 not in tl.transitive_blockers(4)


def test_transitive_dependents_closure_sorted_excludes_self() -> None:
    # 1, 2->1, 3->2, 4 (indep). dependents(1) = {2, 3}; dependents(4) = {}.
    tl = _list([], [1], [2], [])
    assert tl.transitive_dependents(1) == [2, 3]
    assert tl.transitive_dependents(2) == [3]
    assert tl.transitive_dependents(3) == []
    assert tl.transitive_dependents(4) == []


# ---------------------------------------------------------------------------
# has_cycle — whole-graph detection (defense-in-depth at execute entry).
# ---------------------------------------------------------------------------


def test_has_cycle_false_for_acyclic_dag() -> None:
    tl = _list([], [1], [1], [2, 3])
    assert tl.has_cycle() is False


def test_has_cycle_true_for_handbuilt_cycle() -> None:
    # add() rejects cycles, so build a cyclic graph by hand: 1->2, 2->1.
    tl = TaskList(
        tasks=[
            Task(id=1, description="a", status=TaskStatus.PENDING, blockers=[2]),
            Task(id=2, description="b", status=TaskStatus.PENDING, blockers=[1]),
        ],
        next_id=3,
    )
    assert tl.has_cycle() is True


def test_has_cycle_ignores_unknown_blocker_edge() -> None:
    # A blocker pointing at a non-existent task is no edge → no cycle.
    tl = TaskList(
        tasks=[Task(id=1, description="a", status=TaskStatus.PENDING, blockers=[99])],
        next_id=2,
    )
    assert tl.has_cycle() is False


def test_has_cycle_true_for_self_loop() -> None:
    tl = TaskList(
        tasks=[Task(id=1, description="a", status=TaskStatus.PENDING, blockers=[1])],
        next_id=2,
    )
    assert tl.has_cycle() is True


# ---------------------------------------------------------------------------
# Step ledger — bounded drop-oldest (decision B), render, elision marker.
# ---------------------------------------------------------------------------


def test_push_step_ledger_keeps_most_recent_n_drop_oldest() -> None:
    ledger: list[StepLedgerEntry] = []
    dropped_any = False
    # Push 25 entries; the bound is 20 → the first 5 are dropped.
    for i in range(1, 26):
        dropped = push_step_ledger(ledger, StepLedgerEntry(task_id=i, summary=f"s{i}"))
        dropped_any = dropped_any or dropped
    assert len(ledger) == STEP_LEDGER_MAX_ENTRIES
    assert dropped_any is True
    # Exactly the 20 most-recent entries remain, completion order preserved.
    assert [e.task_id for e in ledger] == list(range(6, 26))


def test_push_step_ledger_returns_false_below_bound() -> None:
    ledger: list[StepLedgerEntry] = []
    assert push_step_ledger(ledger, StepLedgerEntry(task_id=1, summary="s")) is False
    assert len(ledger) == 1


def test_render_step_ledger_empty_is_none() -> None:
    assert render_step_ledger([], False) is None


def test_render_step_ledger_lines_and_files_suffix() -> None:
    ledger = [
        StepLedgerEntry(task_id=1, summary="did one"),
        StepLedgerEntry(task_id=2, summary="did two", files_touched=["a.py", "b.py"]),
    ]
    rendered = render_step_ledger(ledger, False)
    assert rendered == ("Progress ledger so far:\n#1 did one\n#2 did two [files: a.py, b.py]")


def test_render_step_ledger_elision_marker_when_elided() -> None:
    ledger = [StepLedgerEntry(task_id=1, summary="did one")]
    rendered = render_step_ledger(ledger, True)
    assert rendered is not None
    assert rendered.splitlines()[1] == STEP_LEDGER_ELISION_MARKER


# ---------------------------------------------------------------------------
# Decision C: the plan_artifact_to_task_list bridge is deprecated (still works).
# ---------------------------------------------------------------------------


def test_plan_artifact_bridge_is_deprecated_but_works() -> None:
    artifact = PlanArtifact(tasks=["one", "two"], rationale="r")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl = plan_artifact_to_task_list(artifact)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    # The bridge still produces a correct linear list (blockers all empty).
    assert [t.description for t in tl.tasks] == ["one", "two"]
    assert all(not t.blockers for t in tl.tasks)
