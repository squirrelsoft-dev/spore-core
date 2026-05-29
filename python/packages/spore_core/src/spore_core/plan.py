"""Issue #70 — Plan phase / plan artifact (PlanExecute, phase 1 of 2).

Mirrors the Rust reference at ``rust/crates/spore-core/src/plan.rs``.

This module owns the *capture* half of the PlanExecute plan phase: turning a
planner model's ``FinalResponse`` text into a structured :class:`PlanArtifact`.
The *phase driver* itself (``StandardHarness._run_plan_phase``) lives on the
harness because it needs the harness's turn machinery; this module supplies the
deterministic, total text→artifact step and the phase error type.

Public surface:

* :class:`PlanArtifact` — re-exported from :mod:`spore_core.hooks`; the
  existing, serializable contract (``{tasks: list[str], rationale: str}``) that
  is the payload of the ``OnPlanCreated`` hook. This issue REUSES it rather than
  defining a competing type. It is the contract consumed by #72 / #59.
* :class:`PlanPhaseError` — error type for the plan phase.
* :func:`capture_plan_artifact` — the model-text → :class:`PlanArtifact` capture
  function. Deterministic and total: never raises an unexpected error; malformed
  input raises :class:`PlanPhaseError` with ``kind == "unparseable_plan"``.

Resolved spec decisions (all four FINAL — match the Rust reference):

* **Q1 (model routing):** ``HarnessConfig.planner_agent`` plus a
  ``HarnessBuilder.planner_agent`` setter. When the strategy is ``PlanExecute``
  and ``planner_agent`` is set, the plan turn runs on it; otherwise on the
  default ``config.agent``. ``plan_model`` stays DESCRIPTIVE metadata only.
* **Q2 (HITL):** The plan phase ALWAYS runs to completion. It fires
  ``OnPlanCreated`` synchronously (the hook may rewrite the artifact); the
  stored artifact reflects any mutation. No pause / no ``WaitingForHuman``.
* **Q3 (capture grammar):** JSON-in-response. Trim ASCII whitespace; strip a
  single leading ```` ``` ````/```` ```json ```` fence line and a single
  trailing ```` ``` ```` fence if present; parse a JSON object with ``tasks``
  (required array of strings, kept verbatim, may be empty) and ``rationale``
  (optional string, default ``""``). Any failure → :class:`PlanPhaseError`
  (``unparseable_plan``).
* **Q4 (terminal RunResult):** After producing, firing ``OnPlanCreated``, and
  storing the artifact, the ``PlanExecute`` arm hands off to the execute loop
  (issue #59), which drains the parsed task list. A well-formed but empty plan
  halts with ``HaltReasonEmptyPlan``.
"""

from __future__ import annotations

import json
from typing import Literal

from .errors import SporeError
from .hooks import PlanArtifact

__all__ = [
    "PLAN_EXECUTE_EXTRAS_KEY",
    "PlanArtifact",
    "PlanPhaseError",
    "capture_plan_artifact",
]

#: Key under which the produced :class:`PlanArtifact` is stored in
#: ``SessionState.extras`` (serialized JSON object). Stable across all four
#: languages.
PLAN_EXECUTE_EXTRAS_KEY = "plan_execute"


class PlanPhaseError(SporeError):
    """Error raised by the plan phase.

    Mirrors the Rust ``PlanPhaseError`` enum: a single class with a ``kind``
    discriminant (``"unparseable_plan"`` | ``"planning_turn_failed"``).

    * ``unparseable_plan`` — the planner's response text could not be parsed
      into a :class:`PlanArtifact` under the Q3 grammar (not valid JSON, not a
      JSON object, or ``tasks`` absent / not an array / containing a non-string
      element).
    * ``planning_turn_failed`` — the plan turn errored or did not produce a
      ``FinalResponse`` (e.g. the planner requested a tool call — R2 — or the
      agent returned an error).
    """

    def __init__(
        self,
        message: str,
        *,
        kind: Literal["unparseable_plan", "planning_turn_failed"],
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message

    @staticmethod
    def unparseable_plan(message: str) -> PlanPhaseError:
        return PlanPhaseError(f"unparseable plan: {message}", kind="unparseable_plan")

    @staticmethod
    def planning_turn_failed(message: str) -> PlanPhaseError:
        return PlanPhaseError(f"planning turn failed: {message}", kind="planning_turn_failed")


# ASCII-whitespace set used for trimming. Matches `' '`, `'\t'`, `'\n'`, `'\r'`,
# and the form-feed / vertical-tab the JSON-adjacent grammar treats as
# whitespace — kept to the ASCII set so trimming is byte-identical
# cross-language (mirrors Rust's ``is_ascii_ws``).
_ASCII_WS = " \t\n\r\x0b\x0c"


def _strip_code_fence(trimmed: str) -> str:
    """Strip a single leading ```` ``` ````/```` ```json ```` fence line and a
    single trailing ```` ``` ```` fence, if the (already-trimmed) input opens
    with a triple-backtick fence. Returns the inner body, re-trimmed. If the
    input does not open with a fence it is returned unchanged. Mirrors Rust's
    ``strip_code_fence``."""
    if not trimmed.startswith("```"):
        return trimmed
    after_open = trimmed[3:]

    # Drop the rest of the opening fence line (the optional language tag) up to
    # and including the first newline. A fence with no newline at all has no
    # body to parse; let JSON parsing reject it downstream.
    nl = after_open.find("\n")
    body_start = after_open[nl + 1 :] if nl != -1 else after_open

    # Strip a single trailing closing fence if present, then re-trim.
    trailing_trimmed = body_start.rstrip(_ASCII_WS)
    body = trailing_trimmed[:-3] if trailing_trimmed.endswith("```") else body_start

    return body.strip(_ASCII_WS)


def capture_plan_artifact(final_text: str) -> PlanArtifact:
    """Capture a :class:`PlanArtifact` from a planner's ``FinalResponse`` text.

    This is the canonical Q3 grammar — it MUST be byte-identical across all four
    languages, so it is kept simple and total:

    1. Trim leading/trailing ASCII whitespace.
    2. If the trimmed text begins with a triple-backtick fence, strip a single
       leading fence line (the opening ```` ``` ```` plus any language tag up to
       and including the first newline) and a single trailing ```` ``` ````
       fence, then trim again.
    3. Parse the result as a JSON object with ``tasks`` (required array of JSON
       strings, kept verbatim; an empty array is allowed) and ``rationale``
       (optional string, default ``""``).

    Any deviation raises :class:`PlanPhaseError` (``unparseable_plan``).
    Deterministic and total: never raises an unexpected error.
    """
    trimmed = final_text.strip(_ASCII_WS)
    body = _strip_code_fence(trimmed)

    try:
        value = json.loads(body)
    except (json.JSONDecodeError, ValueError) as e:
        raise PlanPhaseError.unparseable_plan(f"invalid JSON: {e}") from e

    if not isinstance(value, dict):
        raise PlanPhaseError.unparseable_plan("top-level JSON value is not an object")

    if "tasks" not in value:
        raise PlanPhaseError.unparseable_plan("missing required field `tasks`")

    tasks_value = value["tasks"]
    # ``bool`` is a subclass of ``int`` in Python; ``list`` is the only accepted
    # type here so a stray bool/scalar is rejected like any non-array.
    if not isinstance(tasks_value, list):
        raise PlanPhaseError.unparseable_plan("field `tasks` is not an array")

    tasks: list[str] = []
    for i, element in enumerate(tasks_value):
        # Verbatim — do NOT trim or filter. ``bool`` is excluded explicitly
        # because it is a subclass of ``str``-incompatible ``int``.
        if isinstance(element, str):
            tasks.append(element)
        else:
            raise PlanPhaseError.unparseable_plan(f"element {i} of `tasks` is not a string")

    # ``rationale`` is optional; default "". If present it must be a string.
    if "rationale" not in value:
        rationale = ""
    else:
        rationale_value = value["rationale"]
        if isinstance(rationale_value, str):
            rationale = rationale_value
        else:
            raise PlanPhaseError.unparseable_plan("field `rationale` is not a string")

    return PlanArtifact(tasks=tasks, rationale=rationale)
