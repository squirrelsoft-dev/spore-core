"""Tool-call repair — deterministic coercion of weak-model tool arguments.

Mirrors ``rust/crates/spore-core/src/tool_call_repair.rs``. Weak models
(e.g. ``llama3.2``) frequently emit tool arguments with the wrong JSON *type*:
the string ``"false"`` where a ``bool`` is expected, ``"42"`` where a number is
expected, or ``"[\\"x\\"]"`` where a sequence is expected. The strict param
validation rejects these, the tool returns a recoverable
:class:`~spore_core.harness.ToolOutputError`, and weak models tend to loop on
the same mistake.

This module provides a *pure, deterministic* repair layer the harness applies
between dispatch attempts. Strictly additive: when no
:class:`ToolCallRepair` provider is wired into the harness, behaviour is
byte-identical to today.

Pieces:

* :func:`coerce_tool_args` — the pure coercion function. Given an input value
  and an optional :class:`~spore_core.model.ToolSchema`, returns a repaired
  dict when it changed something, else ``None``.
* :class:`ToolCallFailure` — classification shared by the repair seam and the
  prompt-based fallback.
* :func:`detect_prose_response` — conservative heuristic for prose responses.
* :class:`ToolCallRepair` / :class:`StandardToolCallRepair` — the repair
  strategy the harness calls on a recoverable dispatch error.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from .model import ToolCall, ToolSchema

__all__ = [
    "ACTION_PHRASES",
    "MalformedJson",
    "ProseResponse",
    "StandardToolCallRepair",
    "ToolCallFailure",
    "ToolCallRepair",
    "UnknownTool",
    "coerce_tool_args",
    "detect_prose_response",
]


# ============================================================================
# Failure classification
# ============================================================================


class ToolCallFailure:
    """How a model's tool-calling attempt failed (mirrors Rust's enum).

    Classification shared by the repair seam (malformed JSON, unknown tool) and
    the prompt-based fallback (prose responses). Construct one of the
    subclasses below.
    """


class MalformedJson(ToolCallFailure):
    """A tool call was emitted but its arguments were not valid JSON of the
    expected shape. Recoverable via :func:`coerce_tool_args`."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MalformedJson) and other.text == self.text

    def __repr__(self) -> str:
        return f"MalformedJson({self.text!r})"


class ProseResponse(ToolCallFailure):
    """Tools were advertised, but the model answered in prose instead of
    emitting a tool call. Recoverable by escalating to prompt-based tool
    calling on the next turn."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProseResponse) and other.text == self.text

    def __repr__(self) -> str:
        return f"ProseResponse({self.text!r})"


class UnknownTool(ToolCallFailure):
    """A tool call named a tool not present in the registry."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnknownTool) and other.name == self.name

    def __repr__(self) -> str:
        return f"UnknownTool({self.name!r})"


# ============================================================================
# coerce_tool_args
# ============================================================================


def _property_types(schema: ToolSchema | None) -> dict[str, str]:
    """Extract a ``field-name -> type`` map from a tool's ``input_schema``
    (``{"type":"object","properties":{"field":{"type":"boolean"}, ...}}``).
    A property whose ``type`` is an array (e.g. ``["string","null"]``)
    contributes no single token and is skipped."""
    if schema is None:
        return {}
    props = schema.input_schema.get("properties")
    if not isinstance(props, dict):
        return {}
    out: dict[str, str] = {}
    for name, prop in props.items():
        if isinstance(prop, dict):
            ty = prop.get("type")
            if isinstance(ty, str):
                out[name] = ty
    return out


def _strip_one_quote_pair(s: str) -> str:
    b = s.encode("utf-8")
    if len(b) >= 2:
        first, last = b[0], b[-1]
        if (first == 0x22 and last == 0x22) or (first == 0x27 and last == 0x27):
            return s[1 : len(s) - 1]
    return s


def _normalize_key(key: str, expected: dict[str, str]) -> str:
    if key in expected:
        return key
    stripped = _strip_one_quote_pair(key.strip())
    if stripped != key and (not expected or stripped in expected):
        return stripped
    return key


def _parse_number(s: str) -> Any:
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
    except ValueError:
        return None
    # Reject NaN/Inf — they have no JSON number representation (mirrors Rust's
    # ``serde_json::Number::from_f64`` returning ``None`` for non-finite).
    if not math.isfinite(f):
        return None
    return f


def _expects(expected: str | None, want: str) -> bool:
    return expected == want


def _coerce_scalar(value: Any, expected: str | None) -> Any:
    """Coerce a single field value given its (optional) expected JSON-schema
    type. Returns a repaired value only when a coercion was applied, else
    ``None``."""
    if not isinstance(value, str):
        return None
    trimmed = value.strip()

    # 1. Boolean tokens.
    if _expects(expected, "boolean") or expected is None:
        low = trimmed.lower()
        if low == "true":
            return True
        if low == "false":
            return False

    # 2. Numeric strings. Prefer integer when the token has no fractional part.
    if _expects(expected, "number") or _expects(expected, "integer") or expected is None:
        n = _parse_number(trimmed)
        if n is not None:
            return n

    # 3. JSON-encoded array / object.
    collection_expected = (
        _expects(expected, "array") or _expects(expected, "object") or expected is None
    )
    if collection_expected and (trimmed.startswith("[") or trimmed.startswith("{")):
        import json

        try:
            parsed = json.loads(trimmed)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, list | dict):
            type_ok = (
                (expected == "array" and isinstance(parsed, list))
                or (expected == "object" and isinstance(parsed, dict))
                or expected is None
            )
            if type_ok:
                return parsed
    return None


def coerce_tool_args(
    input_: dict[str, Any], schema: ToolSchema | None = None
) -> dict[str, Any] | None:
    """Attempt deterministic type coercion of a tool call's top-level argument
    object so string-encoded scalars/collections become the JSON types a weak
    model meant to emit.

    Coercions (driven by the JSON-schema's declared property types when a
    schema is provided; otherwise conservative heuristics):

    * ``"true"`` / ``"false"`` (case-insensitive) → ``bool``
    * numeric strings like ``"42"`` / ``"3.14"`` → number
    * a string whose trimmed value parses as a JSON array or object → that
      parsed array/object

    Recurses one level — into the top-level object's fields, which is where
    tool arguments live. Conservative: an already-valid value is never touched.

    Returns a repaired dict if any field changed, else ``None``.
    """
    if not isinstance(input_, dict):
        return None
    expected_types = _property_types(schema)
    changed = False
    out: dict[str, Any] = {}
    for key, value in input_.items():
        normalized = _normalize_key(str(key), expected_types)
        if normalized != key:
            changed = True
        expected = expected_types.get(normalized)
        repaired = _coerce_scalar(value, expected)
        if repaired is not None:
            changed = True
            out[normalized] = repaired
        else:
            out[normalized] = value
    if changed:
        return out
    return None


# ============================================================================
# detect_prose_response
# ============================================================================


ACTION_PHRASES: tuple[str, ...] = (
    "i'll use",
    "i will use",
    "i'll call",
    "i will call",
    "i'll run",
    "i will run",
    "let me use",
    "let me call",
    "let me run",
    "i need to use",
    "i need to call",
    "i should use",
    "i should call",
    "i'll invoke",
    "i will invoke",
    "using the",
    "i can use the",
    "i'm going to use",
    "i am going to use",
    "i'll list",
    "i will list",
    "let me list",
    "i'll inspect",
    "i will inspect",
    "let me inspect",
    "i'll read",
    "i will read",
    "let me read",
    "i'll examine",
    "i will examine",
    "let me examine",
    "i'll review",
    "i will review",
    "let me review",
    "i'll analyze",
    "i will analyze",
    "let me analyze",
    "i'll check",
    "i will check",
    "let me check",
    "i'll look at",
    "i will look at",
    "let me look",
    "i'll search",
    "i will search",
    "let me search",
    "i'll explore",
    "i will explore",
    "let me explore",
    "i'll start by",
    "i will start by",
    "let me start by",
    "let's start by",
)


def detect_prose_response(text: str, tools_advertised: bool) -> ToolCallFailure | None:
    """Conservative heuristic: did the model respond in prose when a tool call
    was the expected next step?

    Returns :class:`ProseResponse` only when **both**:

    1. tools were advertised this turn (``tools_advertised``), and
    2. the response text contains an explicit action-intent phrase suggesting
       the model *meant* to act.

    Bias toward false negatives: a missed prose response costs one extra turn,
    but a false positive activates prompt-based mode for a model that was
    simply giving a final answer.
    """
    if not tools_advertised:
        return None
    trimmed = text.strip()
    if not trimmed:
        return None
    lower = trimmed.lower()
    if any(p in lower for p in ACTION_PHRASES):
        return ProseResponse(trimmed)
    return None


# ============================================================================
# ToolCallRepair strategy
# ============================================================================


@runtime_checkable
class ToolCallRepair(Protocol):
    """Strategy for repairing a tool call that failed dispatch with a
    *recoverable* error. Implementations are pure and side-effect-free."""

    def repair(self, call: ToolCall, error: str, schema: ToolSchema | None) -> ToolCall | None: ...


class StandardToolCallRepair:
    """Default :class:`ToolCallRepair`: applies :func:`coerce_tool_args` to the
    call's input. Returns a repaired call when coercion changed something, else
    ``None``."""

    def repair(self, call: ToolCall, error: str, schema: ToolSchema | None) -> ToolCall | None:
        repaired_input = coerce_tool_args(call.input, schema)
        if repaired_input is None:
            return None
        return ToolCall(id=call.id, name=call.name, input=repaired_input)
