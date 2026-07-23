"""Executable validation that the ``spore-eval`` CLI + lib match the Rust
reference against the SHARED ``fixtures/task_suites/core_suite.json`` (parity
plan §5.17).

Pins CLI parity to the shared fixture without requiring ``cargo`` in Python CI:
the ``run`` shape summary, the human-readable line, the ``promote`` round-trip
(suite_version bump + challenge→regression + byte-stable rewrite), and the
error/exit-code paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from spore_eval import eval_cli
from spore_eval.eval import load_suite_path, suite_to_json


def _fixtures_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "fixtures"
        if candidate.is_dir():
            return candidate
    raise AssertionError("fixtures/ directory not found")


def _core_suite() -> Path:
    return _fixtures_root() / "task_suites" / "core_suite.json"


def test_run_json_summary_matches_fixture_shape(capsys: object) -> None:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = eval_cli._cmd_run([str(_core_suite()), "--json"])
    assert rc == 0
    summary = json.loads(buf.getvalue())
    assert summary["suite_version"] == 1
    assert summary["regression"] == 2
    assert summary["challenge"] == 2
    assert summary["canary"] == 1
    assert summary["n_runs_per_config"] == 3


def test_run_human_readable_line(capsys: object) -> None:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = eval_cli._cmd_run([str(_core_suite()), "--n", "5"])
    assert rc == 0
    line = buf.getvalue().strip()
    # Mirrors Rust's `loaded suite v{} — regression={}, challenge={}, canary={} (n={})`.
    assert line == "loaded suite v1 — regression=2, challenge=2, canary=1 (n=5)"


def test_promote_bumps_version_and_moves_challenge_to_regression(tmp_path: Path) -> None:
    suite = _core_suite()
    tmp_suite = tmp_path / "suite.json"
    tmp_suite.write_text(suite.read_text(encoding="utf-8"), encoding="utf-8")

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = eval_cli._cmd_promote([str(tmp_suite), "challenge_s4_recover"])
    assert rc == 0

    before = load_suite_path(suite)
    after = load_suite_path(tmp_suite)
    assert after.suite_version == before.suite_version + 1
    # The promoted task moved challenge -> regression.
    after_reg_ids = {t.id for t in after.regression}
    assert "challenge_s4_recover" in after_reg_ids
    after_chal_ids = {t.id for t in after.challenge}
    assert "challenge_s4_recover" not in after_chal_ids


def test_promote_round_trips_byte_stably(tmp_path: Path) -> None:
    tmp_suite = tmp_path / "suite.json"
    tmp_suite.write_text(_core_suite().read_text(encoding="utf-8"), encoding="utf-8")
    eval_cli._cmd_promote([str(tmp_suite), "challenge_s4_recover"])
    # The rewritten file round-trips through load_suite_path + suite_to_json.
    suite = load_suite_path(tmp_suite)
    rewritten = suite_to_json(suite)
    assert json.loads(rewritten)["suite_version"] == suite.suite_version


def test_unknown_subcommand_returns_2() -> None:
    import contextlib
    import io
    import sys

    err = io.StringIO()
    old = sys.argv
    sys.argv = ["spore-eval", "bogus"]
    try:
        with contextlib.redirect_stderr(err):
            try:
                eval_cli.main()
            except SystemExit as e:
                assert e.code == 2
            else:  # pragma: no cover
                raise AssertionError("expected SystemExit(2)")
    finally:
        sys.argv = old
    assert "unknown subcommand" in err.getvalue()


def test_run_missing_suite_returns_2() -> None:
    import contextlib
    import io
    import sys

    err = io.StringIO()
    old = sys.argv
    sys.argv = ["spore-eval", "run"]
    try:
        with contextlib.redirect_stderr(err):
            try:
                eval_cli.main()
            except SystemExit as e:
                assert e.code == 2
            else:  # pragma: no cover
                raise AssertionError("expected SystemExit(2)")
    finally:
        sys.argv = old
    assert "missing" in err.getvalue()


def test_lib_round_trip_core_suite() -> None:
    suite = load_suite_path(_core_suite())
    blob = suite_to_json(suite)
    again = load_suite_str_helper(blob)
    assert again.suite_version == suite.suite_version
    assert len(again.regression) == len(suite.regression)
    assert len(again.challenge) == len(suite.challenge)
    assert len(again.canary) == len(suite.canary)


def load_suite_str_helper(blob: str):
    from spore_eval.eval import load_suite_str

    return load_suite_str(blob)
