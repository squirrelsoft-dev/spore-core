"""Manifest loading + manual promotion (Rules 6, 29, 31).

Mirrors ``rust/crates/spore-eval/src/manifest.rs``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .task import (
    ManifestParseError,
    MissingSuiteVersionError,
    TaskSuite,
)


def load_suite_str(body: str) -> TaskSuite:
    """Load a :class:`TaskSuite` from a JSON manifest string. Rejects a manifest
    without ``suite_version`` (Rule 6) with :class:`MissingSuiteVersionError`."""
    try:
        value = json.loads(body)
    except json.JSONDecodeError as e:
        raise ManifestParseError(f"manifest parse error: {e}") from e
    if not isinstance(value, dict) or "suite_version" not in value:
        raise MissingSuiteVersionError()
    try:
        return TaskSuite.model_validate(value)
    except ValidationError as e:
        raise ManifestParseError(f"manifest parse error: {e}") from e


def load_suite_path(path: Path | str) -> TaskSuite:
    """Load a :class:`TaskSuite` from a manifest file path."""
    return load_suite_str(Path(path).read_text(encoding="utf-8"))


def suite_to_json(suite: TaskSuite) -> str:
    """Serialize a :class:`TaskSuite` back to pretty JSON, omitting unset
    optional fields so the round-trip stays close to the source manifest."""
    return json.dumps(suite.model_dump(mode="json", exclude_none=True), indent=2)


def promote_challenge_task(suite: TaskSuite, task_id: str) -> None:
    """Manually promote a ``challenge`` task to ``regression``, bumping
    ``suite_version`` (Rule 31). Auto-promotion is deferred. Raises
    :class:`ManifestParseError` if ``task_id`` is not a challenge task."""
    pos = next((i for i, t in enumerate(suite.challenge) if t.id == task_id), None)
    if pos is None:
        raise ManifestParseError(f"challenge task {task_id!r} not found")
    task = suite.challenge.pop(pos)
    suite.regression.append(task)
    suite.suite_version += 1
