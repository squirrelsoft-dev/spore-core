//! Output-schema delivery + enforcement (issue #139).
//!
//! `ReactConfig.output: Option<SchemaRef>` was presence-validated by the
//! [`ExecutionRegistry`](crate::ExecutionRegistry) at startup but IGNORED at
//! runtime: the resolved schema was never delivered to the model and never
//! validated the terminal. This module is the hand-rolled validator + the
//! frozen literals that make delivery + enforcement deterministic and
//! BYTE-IDENTICAL across the four language ports (Rust is the reference).
//!
//! ## Validator subset (matches the Ollama `format` channel at `ollama.rs`)
//!
//! Only these JSON-schema keywords are honored — NO off-the-shelf validator
//! (they diverge across languages and would break byte-identical fixtures):
//! `type` / `required` / `properties` / `enum`. Anything else in the schema is
//! ignored. [`validate_output`] returns `Ok(())` on a match or
//! `Err(String)` where the `String` is one of the FROZEN validator error
//! strings below.
//!
//! ## Evaluation order (first-match-wins — FROZEN)
//!
//! 1. Output does not parse as JSON → [`ERR_NOT_JSON`].
//! 2. Root value's JSON type ≠ schema `type` → [`err_root_type`].
//! 3. A `required` property absent (iterate `required` in ARRAY order) →
//!    [`err_missing_required`].
//! 4. A present property's value type ≠ its subschema `type` →
//!    [`err_property_type`].
//! 5. A present property's value ∉ its subschema `enum` →
//!    [`err_property_enum`].
//! 6. Otherwise valid.
//!
//! ## Determinism rules (parity-critical)
//!
//! - `properties` are iterated in LEXICOGRAPHICALLY-SORTED key order for steps
//!   4 and 5 (NOT JSON insertion order — serde_json / Go / Python / JS disagree
//!   on insertion order).
//! - `required` is checked in its given ARRAY order; `enum` membership is
//!   order-independent but the `{enum}` rendering in the message is sorted.
//! - `{enum}` and `{value}` are rendered via
//!   [`canonicalize_json`](crate::model::canonicalize_json) (canonical compact
//!   key-sorted JSON), e.g. `["error","ok"]`, `"maybe"`. The enum SORT in the
//!   message is INTENTIONAL (determinism over author intent); the membership
//!   check itself is order-independent.
//! - JSON type names: `object` / `array` / `string` / `number` / `integer` /
//!   `boolean` / `null`.
//! - `integer` is a SEMANTIC check, not a token check: a number with no
//!   fractional part passes as `integer` (`42.0` passes, `42.5` fails). It is
//!   NOT shortcut to "is the JSON token a number".
//!
//! These literals are HASH-LOAD-BEARING — the TS/Python/Go ports MUST reproduce
//! these exact bytes. No `// SPEC QUESTION:` markers remain.

use crate::model::canonicalize_json;
use serde_json::Value;

/// The validation-failure feedback appended as a USER-role message before a
/// retry (#139, AC2). `{error}` is substituted with exactly one validator error
/// string. Exact bytes (single spaces, no trailing newline) — HASH-LOAD-BEARING.
pub(crate) fn feedback_message(error: &str) -> String {
    format!(
        "Your previous response did not match the required output schema. {error} \
Reply with only a JSON value that satisfies the schema."
    )
}

/// Validator error (step 1): the model's response did not parse as JSON.
pub(crate) const ERR_NOT_JSON: &str = "The response was not valid JSON.";

/// Validator error (step 2): the root value's JSON type differs from the
/// schema's `type`.
pub(crate) fn err_root_type(expected: &str, actual: &str) -> String {
    format!("Expected type \"{expected}\" but found \"{actual}\".")
}

/// Validator error (step 3): a `required` property is absent.
pub(crate) fn err_missing_required(name: &str) -> String {
    format!("Missing required property \"{name}\".")
}

/// Validator error (step 4): a present property's value type differs from its
/// subschema `type`.
pub(crate) fn err_property_type(name: &str, expected: &str, actual: &str) -> String {
    format!("Property \"{name}\" should be type \"{expected}\" but found \"{actual}\".")
}

/// Validator error (step 5): a present property's value is not in its subschema
/// `enum`. `{enum}` and `{value}` are canonical compact key-sorted JSON.
pub(crate) fn err_property_enum(name: &str, enum_json: &str, value_json: &str) -> String {
    format!("Property \"{name}\" must be one of {enum_json} but found {value_json}.")
}

/// The JSON type name of `v` (the JSON set; `integer` is reported as `number`
/// here — the integer DISTINCTION is a schema-side semantic check, see
/// [`matches_type`]).
fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Whether `v` satisfies the schema type name `expected`. `integer` is a
/// SEMANTIC check: a JSON number with no fractional part passes (`42.0` passes,
/// `42.5` fails); a `number` schema accepts any JSON number (integer or not).
fn matches_type(v: &Value, expected: &str) -> bool {
    match expected {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        "number" => v.is_number(),
        "integer" => match v {
            // An i64/u64 is integral. An f64 is integral iff it has no
            // fractional part (42.0 passes, 42.5 fails). NOT a token check.
            Value::Number(n) => {
                if n.is_i64() || n.is_u64() {
                    true
                } else if let Some(f) = n.as_f64() {
                    f.fract() == 0.0
                } else {
                    false
                }
            }
            _ => false,
        },
        // Unknown/absent type name: treat as satisfied (the subset only
        // constrains the seven JSON type names; anything else is not enforced).
        _ => true,
    }
}

/// The schema's `type` keyword as a string, if present.
fn schema_type(schema: &Value) -> Option<&str> {
    schema.get("type").and_then(Value::as_str)
}

/// Validate the model's terminal `response` text against `schema` (#139).
///
/// Returns `Ok(())` on a match, or `Err(error)` where `error` is the FIRST
/// (lowest-numbered) FROZEN validator error in the evaluation order. The subset
/// honored is `type` / `required` / `properties` / `enum`; everything else in
/// the schema is ignored. Iteration order is fixed (see the module docs) so the
/// returned error is byte-identical across languages.
pub(crate) fn validate_output(response: &str, schema: &Value) -> Result<(), String> {
    // Step 1: parse.
    let value: Value = match serde_json::from_str(response.trim()) {
        Ok(v) => v,
        Err(_) => return Err(ERR_NOT_JSON.to_string()),
    };

    // Step 2: root type.
    if let Some(expected) = schema_type(schema) {
        if !matches_type(&value, expected) {
            return Err(err_root_type(expected, json_type_name(&value)));
        }
    }

    let obj = value.as_object();
    let props = schema.get("properties").and_then(Value::as_object);

    // Step 3: required (ARRAY order). Only meaningful for an object value;
    // a non-object value already passed step 2 (or had no `type`), so a
    // `required` list applies to its members only when it IS an object.
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        if let Some(obj) = obj {
            for r in required {
                if let Some(name) = r.as_str() {
                    if !obj.contains_key(name) {
                        return Err(err_missing_required(name));
                    }
                }
            }
        }
    }

    // Steps 4 + 5 iterate `properties` in LEXICOGRAPHICALLY-SORTED key order.
    if let (Some(obj), Some(props)) = (obj, props) {
        let mut keys: Vec<&String> = props.keys().collect();
        keys.sort();

        // Step 4: present-property type.
        for key in &keys {
            let subschema = &props[*key];
            if let (Some(present), Some(expected)) = (obj.get(*key), schema_type(subschema)) {
                if !matches_type(present, expected) {
                    return Err(err_property_type(key, expected, json_type_name(present)));
                }
            }
        }

        // Step 5: present-property enum membership.
        for key in &keys {
            let subschema = &props[*key];
            if let (Some(present), Some(enum_arr)) = (
                obj.get(*key),
                subschema.get("enum").and_then(Value::as_array),
            ) {
                let member = enum_arr.iter().any(|e| e == present);
                if !member {
                    // Render the enum SORTED (determinism over author intent)
                    // and the value canonically — both via canonicalize_json.
                    let mut sorted: Vec<&Value> = enum_arr.iter().collect();
                    sorted.sort_by_key(|a| canonicalize_json(a));
                    let enum_json = format!(
                        "[{}]",
                        sorted
                            .iter()
                            .map(|e| canonicalize_json(e))
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    let value_json = canonicalize_json(present);
                    return Err(err_property_enum(key, &enum_json, &value_json));
                }
            }
        }
    }

    // Step 6: valid.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── Feedback + error literals (byte-exact, HASH-LOAD-BEARING) ───────────

    #[test]
    fn feedback_message_exact_bytes() {
        let m = feedback_message("X.");
        assert_eq!(
            m,
            "Your previous response did not match the required output schema. X. \
Reply with only a JSON value that satisfies the schema."
        );
    }

    #[test]
    fn error_literals_exact_bytes() {
        assert_eq!(ERR_NOT_JSON, "The response was not valid JSON.");
        assert_eq!(
            err_root_type("object", "array"),
            "Expected type \"object\" but found \"array\"."
        );
        assert_eq!(
            err_missing_required("name"),
            "Missing required property \"name\"."
        );
        assert_eq!(
            err_property_type("age", "integer", "string"),
            "Property \"age\" should be type \"integer\" but found \"string\"."
        );
        assert_eq!(
            err_property_enum("status", "[\"error\",\"ok\"]", "\"maybe\""),
            "Property \"status\" must be one of [\"error\",\"ok\"] but found \"maybe\"."
        );
    }

    // ── Step 1: parse ──────────────────────────────────────────────────────

    #[test]
    fn step1_not_json() {
        let schema = json!({"type": "object"});
        assert_eq!(
            validate_output("not json at all", &schema),
            Err(ERR_NOT_JSON.to_string())
        );
    }

    // ── Step 2: root type ──────────────────────────────────────────────────

    #[test]
    fn step2_root_type_mismatch() {
        let schema = json!({"type": "object"});
        assert_eq!(
            validate_output("[1,2,3]", &schema),
            Err(err_root_type("object", "array"))
        );
    }

    #[test]
    fn step2_root_type_match() {
        let schema = json!({"type": "array"});
        assert!(validate_output("[1,2,3]", &schema).is_ok());
    }

    // ── Step 3: required (array order) ─────────────────────────────────────

    #[test]
    fn step3_missing_required_first_in_array_order() {
        // `required` order is [b, a]; both absent → the FIRST in array order
        // (`b`) is reported, NOT the lexicographically-first (`a`).
        let schema = json!({"type": "object", "required": ["b", "a"]});
        assert_eq!(
            validate_output("{}", &schema),
            Err(err_missing_required("b"))
        );
    }

    #[test]
    fn step3_required_present() {
        let schema = json!({"type": "object", "required": ["a"]});
        assert!(validate_output("{\"a\":1}", &schema).is_ok());
    }

    // ── Step 4: present-property type (sorted key order) ───────────────────

    #[test]
    fn step4_property_type_mismatch_sorted_order() {
        // Both `age` (number, but value is a string) and `zip` (number, but
        // value is a string) are wrong. Sorted key order ⇒ `age` reported first.
        let schema = json!({
            "type": "object",
            "properties": {
                "zip": {"type": "number"},
                "age": {"type": "number"}
            }
        });
        assert_eq!(
            validate_output("{\"age\":\"x\",\"zip\":\"y\"}", &schema),
            Err(err_property_type("age", "number", "string"))
        );
    }

    #[test]
    fn step4_integer_accepts_whole_number_42_0() {
        // 42.0 has no fractional part → passes `integer`.
        let schema = json!({"type": "object", "properties": {"n": {"type": "integer"}}});
        assert!(validate_output("{\"n\":42.0}", &schema).is_ok());
        assert!(validate_output("{\"n\":42}", &schema).is_ok());
    }

    #[test]
    fn step4_integer_rejects_fractional_42_5() {
        // 42.5 has a fractional part → fails `integer`.
        let schema = json!({"type": "object", "properties": {"n": {"type": "integer"}}});
        assert_eq!(
            validate_output("{\"n\":42.5}", &schema),
            Err(err_property_type("n", "integer", "number"))
        );
    }

    #[test]
    fn step4_number_accepts_fractional() {
        let schema = json!({"type": "object", "properties": {"n": {"type": "number"}}});
        assert!(validate_output("{\"n\":42.5}", &schema).is_ok());
    }

    // ── Step 5: present-property enum (sorted enum rendering) ──────────────

    #[test]
    fn step5_enum_violation_renders_sorted_enum() {
        // Author order is ["ok","error"]; the message renders SORTED
        // (["error","ok"]) for determinism. Membership itself is order-free.
        let schema = json!({
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ok", "error"]}}
        });
        assert_eq!(
            validate_output("{\"status\":\"maybe\"}", &schema),
            Err(err_property_enum(
                "status",
                "[\"error\",\"ok\"]",
                "\"maybe\""
            ))
        );
    }

    #[test]
    fn step5_enum_member_passes_regardless_of_order() {
        let schema = json!({
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ok", "error"]}}
        });
        assert!(validate_output("{\"status\":\"ok\"}", &schema).is_ok());
        assert!(validate_output("{\"status\":\"error\"}", &schema).is_ok());
    }

    // ── Step 6: valid ──────────────────────────────────────────────────────

    #[test]
    fn step6_full_object_valid() {
        let schema = json!({
            "type": "object",
            "required": ["status", "count"],
            "properties": {
                "status": {"type": "string", "enum": ["ok", "error"]},
                "count": {"type": "integer"}
            }
        });
        assert!(validate_output("{\"status\":\"ok\",\"count\":3}", &schema).is_ok());
    }

    // ── Evaluation order: earlier rule wins ────────────────────────────────

    #[test]
    fn order_root_type_beats_required() {
        // A non-object value with a `required` list: step 2 (root type) fires
        // before step 3 (required).
        let schema = json!({"type": "object", "required": ["a"]});
        assert_eq!(
            validate_output("\"a string\"", &schema),
            Err(err_root_type("object", "string"))
        );
    }

    #[test]
    fn order_required_beats_property_type() {
        // `a` absent (required) AND `b` present with wrong type: step 3 wins.
        let schema = json!({
            "type": "object",
            "required": ["a"],
            "properties": {"b": {"type": "number"}}
        });
        assert_eq!(
            validate_output("{\"b\":\"x\"}", &schema),
            Err(err_missing_required("a"))
        );
    }

    #[test]
    fn order_property_type_beats_enum() {
        // `s` present with wrong type AND would also violate enum: step 4 wins.
        let schema = json!({
            "type": "object",
            "properties": {"s": {"type": "string", "enum": ["ok"]}}
        });
        assert_eq!(
            validate_output("{\"s\":123}", &schema),
            Err(err_property_type("s", "string", "number"))
        );
    }
}
