"""Dangerous, opt-in safety footguns — issue #34.

This submodule is the ONLY public path to the two named footguns that issue
#34 gates out of the default package surface:

* ``YoloMode`` — full autonomy, no approval gates. The :class:`spore_core.Mode`
  enum deliberately omits ``Yolo``; importing this module is the explicit,
  reviewable act of opting in. The wire tag stays ``"yolo"``.
* ``IsolationModeNone`` — no path enforcement. Re-exported here only; it is not
  bound by any default ``spore_core`` import.

Importing from ``spore_core`` (or any of its component modules) never exposes
either symbol. Reaching them requires::

    from spore_core.dangerous import YoloMode, IsolationModeNone

This mirrors the Rust reference's ``dangerous`` Cargo feature: in the default
build the variants do not exist, and using them is an explicit, deliberate
choice rather than an accident. These modes are intended for benchmarking and
local development only — do not enable them in production deployments.
"""

from __future__ import annotations

from enum import Enum

# IsolationModeNone is defined in harness.py for the wire discriminated union,
# but is not part of the default public surface. Re-export it here so the only
# way to *name* it is through the dangerous opt-in.
from .harness import IsolationModeNone
from .prompt_chunk_registry import (
    ApprovalPolicy,
    PromptChunk,
    _mode_approval_policy,
    _mode_default_tool_phase,
    _mode_prompt_chunk,
)
from .tool_registry import TaskPhase


class YoloMode(str, Enum):
    """The gated ``Yolo`` mode — full autonomy, no approval gates (issue #34).

    Structurally a :class:`spore_core.prompt_chunk_registry.ModeLike`, so it can
    be passed straight to :meth:`PromptChunkRegistry.compose`. Kept as a
    single-member enum (rather than a plain object) so its ``.value`` is the
    canonical wire tag ``"yolo"``, matching :class:`spore_core.Mode`.
    """

    YOLO = "yolo"

    def prompt_chunk(self) -> PromptChunk:
        """Standard prompt chunk for Yolo mode. Always Static, slot=Mode."""
        return _mode_prompt_chunk(self.value)

    def approval_policy(self) -> ApprovalPolicy:
        return _mode_approval_policy(self.value)

    def default_tool_phase(self) -> TaskPhase:
        return _mode_default_tool_phase(self.value)


__all__ = [
    "IsolationModeNone",
    "YoloMode",
]
