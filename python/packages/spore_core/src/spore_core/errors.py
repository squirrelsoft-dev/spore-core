"""Package-level error roots.

Every component-specific exception inherits from :class:`SporeError`. Errors
that signal a non-recoverable harness-loop halt (spec Layer 1) inherit from
:class:`AlwaysHaltError` as a marker.
"""

from __future__ import annotations


class SporeError(Exception):
    """Root of all spore-core exception types."""


class AlwaysHaltError(SporeError):
    """Marker base for errors the harness must treat as terminal."""


__all__ = ["AlwaysHaltError", "SporeError"]
