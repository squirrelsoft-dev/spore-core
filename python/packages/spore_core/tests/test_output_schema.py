"""Unit tests for the hand-rolled output-schema validator + frozen literals
(issue #139).

Mirrors the Rust reference tests in
``rust/crates/spore-core/src/output_schema.rs`` byte-for-byte (the literals are
HASH-LOAD-BEARING across all four language ports), plus the Python-specific
``bool``-is-not-``int`` cases (Python ``bool`` ⊂ ``int``).
"""

from __future__ import annotations

from spore_core.output_schema import (
    ERR_NOT_JSON,
    err_missing_required,
    err_property_enum,
    err_property_type,
    err_root_type,
    feedback_message,
    validate_output,
)

# ── Feedback + error literals (byte-exact, HASH-LOAD-BEARING) ───────────────


def test_feedback_message_exact_bytes() -> None:
    m = feedback_message("X.")
    assert m == (
        "Your previous response did not match the required output schema. X. "
        "Reply with only a JSON value that satisfies the schema."
    )


def test_error_literals_exact_bytes() -> None:
    assert ERR_NOT_JSON == "The response was not valid JSON."
    assert err_root_type("object", "array") == 'Expected type "object" but found "array".'
    assert err_missing_required("name") == 'Missing required property "name".'
    assert (
        err_property_type("age", "integer", "string")
        == 'Property "age" should be type "integer" but found "string".'
    )
    assert (
        err_property_enum("status", '["error","ok"]', '"maybe"')
        == 'Property "status" must be one of ["error","ok"] but found "maybe".'
    )


# ── Step 1: parse ──────────────────────────────────────────────────────────


def test_step1_not_json() -> None:
    schema = {"type": "object"}
    assert validate_output("not json at all", schema) == ERR_NOT_JSON


# ── Step 2: root type ──────────────────────────────────────────────────────


def test_step2_root_type_mismatch() -> None:
    schema = {"type": "object"}
    assert validate_output("[1,2,3]", schema) == err_root_type("object", "array")


def test_step2_root_type_match() -> None:
    schema = {"type": "array"}
    assert validate_output("[1,2,3]", schema) is None


# ── Step 3: required (array order) ─────────────────────────────────────────


def test_step3_missing_required_first_in_array_order() -> None:
    # ``required`` order is [b, a]; both absent → the FIRST in array order (b) is
    # reported, NOT the lexicographically-first (a).
    schema = {"type": "object", "required": ["b", "a"]}
    assert validate_output("{}", schema) == err_missing_required("b")


def test_step3_required_present() -> None:
    schema = {"type": "object", "required": ["a"]}
    assert validate_output('{"a":1}', schema) is None


# ── Step 4: present-property type (sorted key order) ───────────────────────


def test_step4_property_type_mismatch_sorted_order() -> None:
    # Both ``age`` and ``zip`` are wrong; sorted key order ⇒ ``age`` reported
    # first (NOT insertion order).
    schema = {
        "type": "object",
        "properties": {
            "zip": {"type": "number"},
            "age": {"type": "number"},
        },
    }
    assert validate_output('{"age":"x","zip":"y"}', schema) == err_property_type(
        "age", "number", "string"
    )


def test_step4_integer_accepts_whole_number_42_0() -> None:
    # 42.0 has no fractional part → passes ``integer``.
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert validate_output('{"n":42.0}', schema) is None
    assert validate_output('{"n":42}', schema) is None


def test_step4_integer_rejects_fractional_42_5() -> None:
    # 42.5 has a fractional part → fails ``integer``.
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert validate_output('{"n":42.5}', schema) == err_property_type("n", "integer", "number")


def test_step4_number_accepts_fractional() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "number"}}}
    assert validate_output('{"n":42.5}', schema) is None


def test_step4_bool_is_not_integer() -> None:
    # Python ``bool`` ⊂ ``int`` — a boolean must NOT validate as ``integer``.
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert validate_output('{"n":true}', schema) == err_property_type("n", "integer", "boolean")


def test_step4_bool_is_not_number() -> None:
    # A boolean must NOT validate as ``number`` either.
    schema = {"type": "object", "properties": {"n": {"type": "number"}}}
    assert validate_output('{"n":false}', schema) == err_property_type("n", "number", "boolean")


def test_step4_boolean_type_accepts_bool() -> None:
    schema = {"type": "object", "properties": {"b": {"type": "boolean"}}}
    assert validate_output('{"b":true}', schema) is None


def test_step2_integer_does_not_accept_bool_root() -> None:
    # Root-level: a bare ``true`` is a boolean, not an integer.
    schema = {"type": "integer"}
    assert validate_output("true", schema) == err_root_type("integer", "boolean")


# ── Step 5: present-property enum (sorted enum rendering) ──────────────────


def test_step5_enum_violation_renders_sorted_enum() -> None:
    # Author order is ["ok","error"]; the message renders SORTED (["error","ok"])
    # for determinism. Membership itself is order-free.
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok", "error"]}},
    }
    assert validate_output('{"status":"maybe"}', schema) == err_property_enum(
        "status", '["error","ok"]', '"maybe"'
    )


def test_step5_enum_member_passes_regardless_of_order() -> None:
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok", "error"]}},
    }
    assert validate_output('{"status":"ok"}', schema) is None
    assert validate_output('{"status":"error"}', schema) is None


def test_step5_enum_bool_value_not_equal_to_number() -> None:
    # serde_json's ``==`` (and our ``_json_equal``): ``true`` is NOT a member of
    # an enum [1, 0] even though Python ``True == 1``. The value renders
    # canonically (``true``); the enum renders sorted+canonical.
    schema = {"type": "object", "properties": {"x": {"enum": [1, 0]}}}
    assert validate_output('{"x":true}', schema) == err_property_enum("x", "[0,1]", "true")


# ── Step 6: valid ──────────────────────────────────────────────────────────


def test_step6_full_object_valid() -> None:
    schema = {
        "type": "object",
        "required": ["status", "count"],
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "count": {"type": "integer"},
        },
    }
    assert validate_output('{"status":"ok","count":3}', schema) is None


# ── Evaluation order: earlier rule wins ────────────────────────────────────


def test_order_root_type_beats_required() -> None:
    # A non-object value with a ``required`` list: step 2 fires before step 3.
    schema = {"type": "object", "required": ["a"]}
    assert validate_output('"a string"', schema) == err_root_type("object", "string")


def test_order_required_beats_property_type() -> None:
    # ``a`` absent (required) AND ``b`` present with wrong type: step 3 wins.
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"b": {"type": "number"}},
    }
    assert validate_output('{"b":"x"}', schema) == err_missing_required("a")


def test_order_property_type_beats_enum() -> None:
    # ``s`` present with wrong type AND would also violate enum: step 4 wins.
    schema = {
        "type": "object",
        "properties": {"s": {"type": "string", "enum": ["ok"]}},
    }
    assert validate_output('{"s":123}', schema) == err_property_type("s", "string", "number")
