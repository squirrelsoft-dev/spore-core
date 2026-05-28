"""``spore-eval`` CLI — mirrors the ``spore-e2e-agent`` pattern.

Subcommands:

``run <suite.json> [--candidate ID ...] [--n N] [--json]``
    Load a suite and print/serialize its shape. Candidate harness configs are
    not constructible from the CLI in the MVP — full runs are driven
    programmatically via :class:`spore_eval.eval.EvalHarness`; this command
    validates the suite and reports its shape.

``promote <suite.json> <task_id>``
    Manually promote a challenge task to regression, bump ``suite_version``,
    and rewrite the JSON in place (Rule 31). Auto-promotion is deferred.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .eval import (
    EvalError,
    load_suite_path,
    promote_challenge_task,
    suite_to_json,
)

_USAGE = (
    "spore-eval — EvalHarness CLI\n\n"
    "USAGE:\n"
    "  spore-eval run <suite.json> [--candidate ID ...] [--n N] [--json]\n"
    "  spore-eval promote <suite.json> <task_id>\n"
)


def _print_usage() -> None:
    print(_USAGE, file=sys.stderr)


def _cmd_run(args: list[str]) -> int:
    suite_path: str | None = None
    candidates: list[str] = []
    n: int | None = None
    as_json = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--candidate":
            i += 1
            if i >= len(args):
                print("--candidate needs an ID", file=sys.stderr)
                return 2
            candidates.append(args[i])
        elif arg == "--n":
            i += 1
            if i >= len(args):
                print("--n needs a number", file=sys.stderr)
                return 2
            try:
                n = int(args[i])
            except ValueError:
                print("--n must be an integer", file=sys.stderr)
                return 2
        elif arg == "--json":
            as_json = True
        elif not arg.startswith("--") and suite_path is None:
            suite_path = arg
        else:
            print(f"unexpected argument: {arg}", file=sys.stderr)
            return 2
        i += 1

    if suite_path is None:
        print("missing <suite.json>", file=sys.stderr)
        return 2
    try:
        suite = load_suite_path(suite_path)
    except EvalError as e:
        print(f"loading suite {suite_path}: {e}", file=sys.stderr)
        return 1

    n_runs = n if n is not None else 3
    if as_json:
        summary = {
            "suite_version": suite.suite_version,
            "regression": len(suite.regression),
            "challenge": len(suite.challenge),
            "canary": len(suite.canary),
            "n_runs_per_config": n_runs,
            "candidates": candidates,
            "note": (
                "candidate harness configs are wired programmatically; "
                "this CLI validates the suite and reports its shape"
            ),
        }
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"loaded suite v{suite.suite_version} — "
            f"regression={len(suite.regression)}, "
            f"challenge={len(suite.challenge)}, "
            f"canary={len(suite.canary)} (n={n_runs})"
        )
    return 0


def _cmd_promote(args: list[str]) -> int:
    if len(args) < 1:
        print("missing <suite.json>", file=sys.stderr)
        return 2
    if len(args) < 2:
        print("missing <task_id>", file=sys.stderr)
        return 2
    suite_path, task_id = args[0], args[1]
    try:
        suite = load_suite_path(suite_path)
        before = suite.suite_version
        promote_challenge_task(suite, task_id)
        Path(suite_path).write_text(suite_to_json(suite), encoding="utf-8")
    except EvalError as e:
        print(f"promoting {task_id}: {e}", file=sys.stderr)
        return 1
    print(
        f"promoted {task_id}: suite_version {before} -> {suite.suite_version} "
        "(challenge -> regression)"
    )
    return 0


def main() -> None:
    args = sys.argv[1:]
    if not args:
        _print_usage()
        raise SystemExit(2)
    cmd, rest = args[0], args[1:]
    if cmd == "run":
        raise SystemExit(_cmd_run(rest))
    if cmd == "promote":
        raise SystemExit(_cmd_promote(rest))
    if cmd in ("-h", "--help", "help"):
        _print_usage()
        raise SystemExit(0)
    _print_usage()
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
