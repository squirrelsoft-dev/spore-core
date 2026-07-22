"""Public-API parity test — every Rust ``pub use`` re-export has a Python
counterpart importable from :mod:`spore_core` (parity plan §5.2).

These are the symbols that were present in their Python modules but NOT
re-exported from the package root, plus the ``observability_outbox`` surface.
The assertions pin the import surface so a future refactor cannot silently
drop a public name.
"""

from __future__ import annotations

import spore_core as sc


def test_observability_outbox_reexports() -> None:
    for name in (
        "OutboxConfig",
        "OutboxObservabilityProvider",
        "TraceLine",
        "ObservabilityError",
        "SessionNotFound",
    ):
        assert hasattr(sc, name), f"missing {name}"


def test_observability_extra_reexports() -> None:
    for name in (
        "SpanLevel",
        "PatchSpan",
        "PatchType",
        "ContentCaptureConfig",
        "TRUNCATION_MARKER",
        "truncate_field",
        "GenAiMessage",
        "GenAiRole",
        "SensorSpan",
    ):
        assert hasattr(sc, name), f"missing {name}"


def test_termination_check_reexports() -> None:
    for name in (
        "AlwaysComplete",
        "FeatureListCheck",
        "QuestionAnsweredCheck",
        "SqlResultCheck",
        "TestSuiteCheck",
    ):
        assert hasattr(sc, name), f"missing {name}"


def test_storage_scoped_memory_router_reexport() -> None:
    assert hasattr(sc, "ScopedMemoryRouter")


def test_step_ledger_reexports() -> None:
    for name in (
        "StepLedgerEntry",
        "push_step_ledger",
        "render_step_ledger",
        "STEP_LEDGER_ELISION_MARKER",
        "STEP_LEDGER_MAX_ENTRIES",
    ):
        assert hasattr(sc, name), f"missing {name}"


def test_truncate_field_behavior() -> None:
    # Mirrors Rust observability::truncate_field: returns (clipped, truncated?).
    out, clipped = sc.truncate_field("hello world", 5)
    assert out == "hello...[truncated]"
    assert clipped is True
    out, clipped = sc.truncate_field("short", 50)
    assert out == "short"
    assert clipped is False


def test_step_ledger_round_trip() -> None:
    entry = sc.StepLedgerEntry(task_id=1, summary="do thing", files_touched=["a.py"])
    ledger: list = []
    dropped_any = False
    for _ in range(sc.STEP_LEDGER_MAX_ENTRIES + 5):
        if sc.push_step_ledger(ledger, entry):
            dropped_any = True
    # Drop-oldest: never exceeds the cap; at least one drop happened.
    assert len(ledger) == sc.STEP_LEDGER_MAX_ENTRIES
    assert dropped_any is True
    rendered = sc.render_step_ledger(ledger, elided=True)
    assert rendered is not None
    assert sc.STEP_LEDGER_ELISION_MARKER in rendered
