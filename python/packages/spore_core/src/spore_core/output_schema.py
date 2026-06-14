"""Output-schema delivery + enforcement (issue #139).

``ReactConfig.output`` was presence-validated by the
:class:`~spore_core.execution_registry.ExecutionRegistry` at startup but IGNORED
at runtime: the resolved schema was never delivered to the model and never
validated the terminal. This module is the hand-rolled validator + the frozen
literals that make delivery + enforcement deterministic and BYTE-IDENTICAL
across the four language ports (Rust is the reference at
``rust/crates/spore-core/src/output_schema.rs``).

Validator subset (matches the Ollama ``format`` channel): only ``type`` /
``required`` / ``properties`` / ``enum`` are honored — NO off-the-shelf
validator (they diverge across languages and would break byte-identical
fixtures). Anything else in the schema is ignored.

Evaluation order (first-match-wins — FROZEN):

1. Output does not parse as JSON → :data:`ERR_NOT_JSON`.
2. Root value's JSON type ≠ schema ``type`` → :func:`err_root_type`.
3. A ``required`` property absent (iterate ``required`` in ARRAY order) →
   :func:`err_missing_required`.
4. A present property's value type ≠ its subschema ``type`` →
   :func:`err_property_type`.
5. A present property's value ∉ its subschema ``enum`` →
   :func:`err_property_enum`.
6. Otherwise valid.

Determinism rules (parity-critical):

- ``properties`` are iterated in LEXICOGRAPHICALLY-SORTED key order for steps 4
  and 5 (NOT insertion order — Python's ``json`` does not sort by default).
- ``required`` is checked in its given ARRAY order; ``enum`` membership is
  order-independent but the ``{enum}`` rendering in the message is sorted.
- ``{enum}`` and ``{value}`` are rendered via
  :func:`~spore_core.model._canonicalize_json` (canonical compact key-sorted
  JSON), e.g. ``["error","ok"]``, ``"maybe"``. The enum SORT in the message is
  INTENTIONAL (determinism over author intent); the membership check itself is
  order-independent.
- JSON type names: ``object`` / ``array`` / ``string`` / ``number`` /
  ``integer`` / ``boolean`` / ``null``.
- ``integer`` is a SEMANTIC check: a number with no fractional part passes as
  ``integer`` (``42.0`` passes, ``42.5`` fails). Python ``bool`` is a subclass
  of ``int`` — a boolean must NOT validate as ``integer`` / ``number``.

These literals are HASH-LOAD-BEARING — the Rust/TS/Go ports reproduce these
exact bytes.
"""

from __future__ import annotations

import json
from typing import Any

from .model import _canonicalize_json


def feedback_message(error: str) -> str:
    """The validation-failure feedback appended as a USER-role message before a
    retry (#139, AC2). ``error`` is substituted with exactly one validator error
    string. Exact bytes (single spaces, no trailing newline) — HASH-LOAD-BEARING.
    """

    return (
        f"Your previous response did not match the required output schema. {error} "
        "Reply with only a JSON value that satisfies the schema."
    )


#: Validator error (step 1): the model's response did not parse as JSON.
ERR_NOT_JSON = "The response was not valid JSON."


def err_root_type(expected: str, actual: str) -> str:
    """Validator error (step 2): the root value's JSON type differs from the
    schema's ``type``."""

    return f'Expected type "{expected}" but found "{actual}".'


def err_missing_required(name: str) -> str:
    """Validator error (step 3): a ``required`` property is absent."""

    return f'Missing required property "{name}".'


def err_property_type(name: str, expected: str, actual: str) -> str:
    """Validator error (step 4): a present property's value type differs from its
    subschema ``type``."""

    return f'Property "{name}" should be type "{expected}" but found "{actual}".'


def err_property_enum(name: str, enum_json: str, value_json: str) -> str:
    """Validator error (step 5): a present property's value is not in its
    subschema ``enum``. ``enum_json`` and ``value_json`` are canonical compact
    key-sorted JSON."""

    return f'Property "{name}" must be one of {enum_json} but found {value_json}.'


def _json_type_name(v: Any) -> str:
    """The JSON type name of ``v`` (the JSON set; ``integer`` is reported as
    ``number`` here — the integer DISTINCTION is a schema-side semantic check,
    see :func:`_matches_type`). Python ``bool`` is a subclass of ``int`` and
    must be classified as ``boolean``, NOT ``number``."""

    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    # Unreachable for ``json.loads`` output; defensive.
    return "null"


def _matches_type(v: Any, expected: str) -> bool:
    """Whether ``v`` satisfies the schema type name ``expected``. ``integer`` is
    a SEMANTIC check: a JSON number with no fractional part passes (``42.0``
    passes, ``42.5`` fails); a ``number`` schema accepts any JSON number. A
    boolean is NEVER a number/integer (Python ``bool`` ⊂ ``int`` is handled
    explicitly)."""

    if expected == "object":
        return isinstance(v, dict)
    if expected == "array":
        return isinstance(v, list)
    if expected == "string":
        return isinstance(v, str)
    if expected == "boolean":
        return isinstance(v, bool)
    if expected == "null":
        return v is None
    if expected == "number":
        # A boolean is not a number (bool ⊂ int in Python).
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if expected == "integer":
        if isinstance(v, bool):
            return False
        if isinstance(v, int):
            return True
        if isinstance(v, float):
            # Integral iff no fractional part (42.0 passes, 42.5 fails).
            return v.is_integer()
        return False
    # Unknown/absent type name: treat as satisfied (the subset only constrains
    # the seven JSON type names; anything else is not enforced).
    return True


def _schema_type(schema: Any) -> str | None:
    """The schema's ``type`` keyword as a string, if present."""

    if isinstance(schema, dict):
        t = schema.get("type")
        if isinstance(t, str):
            return t
    return None


def validate_output(response: str, schema: Any) -> str | None:
    """Validate the model's terminal ``response`` text against ``schema`` (#139).

    Returns ``None`` on a match, or the FIRST (lowest-numbered) FROZEN validator
    error string in the evaluation order. The subset honored is ``type`` /
    ``required`` / ``properties`` / ``enum``; everything else is ignored.
    Iteration order is fixed (see the module docs) so the returned error is
    byte-identical across languages.
    """

    # Step 1: parse.
    try:
        value: Any = json.loads(response.strip())
    except ValueError:
        return ERR_NOT_JSON

    # Step 2: root type.
    expected = _schema_type(schema)
    if expected is not None and not _matches_type(value, expected):
        return err_root_type(expected, _json_type_name(value))

    obj = value if isinstance(value, dict) else None
    props = schema.get("properties") if isinstance(schema, dict) else None
    props = props if isinstance(props, dict) else None

    # Step 3: required (ARRAY order). Only meaningful for an object value.
    required = schema.get("required") if isinstance(schema, dict) else None
    if isinstance(required, list) and obj is not None:
        for r in required:
            if isinstance(r, str) and r not in obj:
                return err_missing_required(r)

    # Steps 4 + 5 iterate ``properties`` in LEXICOGRAPHICALLY-SORTED key order.
    if obj is not None and props is not None:
        keys = sorted(props.keys())

        # Step 4: present-property type.
        for key in keys:
            subschema = props[key]
            sub_expected = _schema_type(subschema)
            if key in obj and sub_expected is not None:
                present = obj[key]
                if not _matches_type(present, sub_expected):
                    return err_property_type(key, sub_expected, _json_type_name(present))

        # Step 5: present-property enum membership.
        for key in keys:
            subschema = props[key]
            enum_arr = subschema.get("enum") if isinstance(subschema, dict) else None
            if key in obj and isinstance(enum_arr, list):
                present = obj[key]
                if not any(_json_equal(present, e) for e in enum_arr):
                    # Render the enum SORTED (determinism over author intent) and
                    # the value canonically — both via ``_canonicalize_json``.
                    sorted_members = sorted(enum_arr, key=_canonicalize_json)
                    enum_json = "[" + ",".join(_canonicalize_json(e) for e in sorted_members) + "]"
                    value_json = _canonicalize_json(present)
                    return err_property_enum(key, enum_json, value_json)

    # Step 6: valid.
    return None


def _json_equal(a: Any, b: Any) -> bool:
    """JSON value equality for enum membership. Mirrors serde_json's ``==``: a
    boolean is never equal to a number even though Python ``True == 1``."""

    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


__all__ = [
    "ERR_NOT_JSON",
    "err_missing_required",
    "err_property_enum",
    "err_property_type",
    "err_root_type",
    "feedback_message",
    "validate_output",
]
