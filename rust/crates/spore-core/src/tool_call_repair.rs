//! Tool-call repair — deterministic coercion of weak-model tool arguments.
//!
//! Weak models (e.g. `llama3.2`) frequently emit tool arguments with the wrong
//! JSON *type*: the string `"false"` where a `bool` is expected, `"42"` where a
//! number is expected, or `"[\"x\"]"` where a sequence is expected. The strict
//! `serde_json::from_value` in [`crate::tools::params::parse_params`] rejects
//! these, the tool returns a recoverable [`ToolOutput::Error`], and weak models
//! tend to loop on the same mistake.
//!
//! This module provides a *pure, deterministic* repair layer that the harness
//! can apply between dispatch attempts. It is strictly additive: when no
//! [`ToolCallRepair`] provider is wired into the harness, behaviour is
//! byte-identical to today.
//!
//! ## Pieces
//! - [`coerce_tool_args`] — the pure coercion function. Given an input value and
//!   an optional [`ToolSchema`], it returns `Some(repaired)` when it changed
//!   something, else `None`.
//! - [`ToolCallRepair`] — trait the harness calls when a recoverable error
//!   occurs. [`StandardToolCallRepair`] is the default, coercion-driven impl.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::model::{ToolCall, ToolSchema};

/// Attempt deterministic type coercion of a tool call's top-level argument
/// object so that string-encoded scalars/collections become the JSON types a
/// weak model meant to emit.
///
/// Coercions (driven by the JSON-schema's declared property types when a
/// [`ToolSchema`] is provided; otherwise applied as conservative heuristics):
/// - `"true"` / `"false"` (case-insensitive) → `bool`
/// - numeric strings like `"42"` / `"3.14"` → number
/// - a string whose *trimmed* value parses as a JSON array or object → that
///   parsed array/object
///
/// The function only recurses one level — into the top-level object's fields,
/// which is where tool arguments live. It is conservative: an already-valid
/// value is never touched, and a field is only coerced when the result
/// plausibly helps.
///
/// Returns `Some(repaired)` if any field changed, else `None`.
pub fn coerce_tool_args(input: &Value, schema: Option<&ToolSchema>) -> Option<Value> {
    // Tool args always live in a top-level object. Anything else, we leave alone.
    let obj = input.as_object()?;

    // Map of field-name → expected JSON-schema "type" token, when a schema is
    // available. Used to make coercion targeted instead of best-effort.
    let expected_types = schema.map(property_types).unwrap_or_default();

    let mut changed = false;
    let mut out = serde_json::Map::with_capacity(obj.len());

    for (key, value) in obj {
        let expected = expected_types.get(key.as_str()).map(String::as_str);
        match coerce_scalar(value, expected) {
            Some(repaired) => {
                changed = true;
                out.insert(key.clone(), repaired);
            }
            None => {
                out.insert(key.clone(), value.clone());
            }
        }
    }

    if changed {
        Some(Value::Object(out))
    } else {
        None
    }
}

/// Coerce a single field value given its (optional) expected JSON-schema type.
/// Returns `Some(new_value)` only when a coercion was applied.
fn coerce_scalar(value: &Value, expected: Option<&str>) -> Option<Value> {
    // Only string values are candidates for coercion. Already-typed values are
    // assumed correct and are never touched — this protects valid inputs.
    let s = value.as_str()?;
    let trimmed = s.trim();

    // 1. Boolean tokens. These exact tokens are unambiguous, so we coerce them
    //    whenever a bool is expected, or when no schema type is known.
    if expects(expected, "boolean") || expected.is_none() {
        match trimmed.to_ascii_lowercase().as_str() {
            "true" => return Some(Value::Bool(true)),
            "false" => return Some(Value::Bool(false)),
            _ => {}
        }
    }

    // 2. Numeric strings. Prefer integer when the token has no fractional part.
    if expects(expected, "number") || expects(expected, "integer") || expected.is_none() {
        if let Some(n) = parse_number(trimmed) {
            return Some(n);
        }
    }

    // 3. JSON-encoded array / object. A string whose trimmed content parses as
    //    a JSON array or object is very likely a mis-stringified collection.
    let collection_expected =
        expects(expected, "array") || expects(expected, "object") || expected.is_none();
    if collection_expected && (trimmed.starts_with('[') || trimmed.starts_with('{')) {
        if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
            if parsed.is_array() || parsed.is_object() {
                // Respect the declared type when we have one: don't turn an
                // array into an object slot or vice-versa.
                let type_ok = match expected {
                    Some("array") => parsed.is_array(),
                    Some("object") => parsed.is_object(),
                    _ => true,
                };
                if type_ok {
                    return Some(parsed);
                }
            }
        }
    }

    None
}

/// Parse a trimmed numeric string into a JSON number. Integers are preferred
/// over floats so `"42"` becomes `42` rather than `42.0`.
fn parse_number(s: &str) -> Option<Value> {
    if s.is_empty() {
        return None;
    }
    if let Ok(i) = s.parse::<i64>() {
        return Some(Value::Number(i.into()));
    }
    if let Ok(u) = s.parse::<u64>() {
        return Some(Value::Number(u.into()));
    }
    if let Ok(f) = s.parse::<f64>() {
        // Reject NaN/Inf — they have no JSON number representation.
        return serde_json::Number::from_f64(f).map(Value::Number);
    }
    None
}

/// `true` if the (optional) expected type token equals `want`.
fn expects(expected: Option<&str>, want: &str) -> bool {
    expected == Some(want)
}

/// Extract a `field-name → type` map from a tool's `input_schema`
/// (`{"type":"object","properties":{ "field": {"type":"boolean"}, ... }}`).
/// A property whose `type` is an array (e.g. `["string","null"]`) contributes
/// no single token and is skipped.
fn property_types(schema: &ToolSchema) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let Some(props) = schema
        .input_schema
        .get("properties")
        .and_then(Value::as_object)
    else {
        return map;
    };
    for (name, prop) in props {
        if let Some(ty) = prop.get("type").and_then(Value::as_str) {
            map.insert(name.clone(), ty.to_string());
        }
    }
    map
}

/// Strategy for repairing a tool call that failed dispatch with a *recoverable*
/// error. Implementations are pure and side-effect-free: they inspect the call
/// (and optionally its schema + the error text) and return a repaired
/// [`ToolCall`] to re-dispatch, or `None` to give up.
pub trait ToolCallRepair: Send + Sync {
    /// Given a call whose dispatch returned a recoverable error, return a
    /// repaired [`ToolCall`] to re-dispatch, or `None` to give up.
    fn repair(&self, call: &ToolCall, error: &str, schema: Option<&ToolSchema>)
        -> Option<ToolCall>;
}

/// Default [`ToolCallRepair`]: applies [`coerce_tool_args`] to the call's
/// input. Returns a repaired call when coercion changed something, else `None`.
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardToolCallRepair;

impl ToolCallRepair for StandardToolCallRepair {
    fn repair(
        &self,
        call: &ToolCall,
        _error: &str,
        schema: Option<&ToolSchema>,
    ) -> Option<ToolCall> {
        let repaired_input = coerce_tool_args(&call.input, schema)?;
        Some(ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            input: repaired_input,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema(props: Value) -> ToolSchema {
        ToolSchema {
            name: "t".into(),
            description: "t".into(),
            input_schema: json!({ "type": "object", "properties": props }),
        }
    }

    #[test]
    fn coerces_false_string_to_bool() {
        let input = json!({ "append": "false" });
        let s = schema(json!({ "append": { "type": "boolean" } }));
        let out = coerce_tool_args(&input, Some(&s)).expect("should repair");
        assert_eq!(out, json!({ "append": false }));
    }

    #[test]
    fn coerces_true_string_to_bool() {
        let input = json!({ "append": "TRUE" });
        let s = schema(json!({ "append": { "type": "boolean" } }));
        let out = coerce_tool_args(&input, Some(&s)).expect("should repair");
        assert_eq!(out, json!({ "append": true }));
    }

    #[test]
    fn coerces_numeric_string_to_integer() {
        let input = json!({ "n": "42" });
        let s = schema(json!({ "n": { "type": "integer" } }));
        let out = coerce_tool_args(&input, Some(&s)).expect("should repair");
        assert_eq!(out, json!({ "n": 42 }));
    }

    #[test]
    fn coerces_float_string_to_number() {
        let input = json!({ "x": "2.5" });
        let s = schema(json!({ "x": { "type": "number" } }));
        let out = coerce_tool_args(&input, Some(&s)).expect("should repair");
        assert_eq!(out, json!({ "x": 2.5 }));
    }

    #[test]
    fn coerces_json_array_string_to_array() {
        let input = json!({ "args": "[\"a\",\"b\"]" });
        let s = schema(json!({ "args": { "type": "array" } }));
        let out = coerce_tool_args(&input, Some(&s)).expect("should repair");
        assert_eq!(out, json!({ "args": ["a", "b"] }));
    }

    #[test]
    fn coerces_json_object_string_to_object() {
        let input = json!({ "headers": "{\"k\":\"v\"}" });
        let s = schema(json!({ "headers": { "type": "object" } }));
        let out = coerce_tool_args(&input, Some(&s)).expect("should repair");
        assert_eq!(out, json!({ "headers": { "k": "v" } }));
    }

    #[test]
    fn already_valid_bool_left_unchanged() {
        let input = json!({ "append": false });
        let s = schema(json!({ "append": { "type": "boolean" } }));
        assert!(coerce_tool_args(&input, Some(&s)).is_none());
    }

    #[test]
    fn already_valid_array_left_unchanged() {
        let input = json!({ "args": ["a", "b"] });
        let s = schema(json!({ "args": { "type": "array" } }));
        assert!(coerce_tool_args(&input, Some(&s)).is_none());
    }

    #[test]
    fn non_coercible_string_left_alone() {
        let input = json!({ "path": "/etc/hosts" });
        let s = schema(json!({ "path": { "type": "string" } }));
        assert!(coerce_tool_args(&input, Some(&s)).is_none());
    }

    #[test]
    fn string_field_does_not_consume_bool_token_when_schema_says_string() {
        // "false" is a legitimate string value when the schema declares string.
        let input = json!({ "q": "false" });
        let s = schema(json!({ "q": { "type": "string" } }));
        assert!(coerce_tool_args(&input, Some(&s)).is_none());
    }

    #[test]
    fn heuristic_without_schema_coerces_bool_and_number() {
        let input = json!({ "a": "true", "b": "7", "c": "hello" });
        let out = coerce_tool_args(&input, None).expect("should repair");
        assert_eq!(out, json!({ "a": true, "b": 7, "c": "hello" }));
    }

    #[test]
    fn non_object_input_returns_none() {
        assert!(coerce_tool_args(&json!("scalar"), None).is_none());
        assert!(coerce_tool_args(&json!(["a"]), None).is_none());
    }

    #[test]
    fn standard_repair_produces_repaired_call() {
        let repair = StandardToolCallRepair;
        let call = ToolCall {
            id: "c1".into(),
            name: "write".into(),
            input: json!({ "append": "false" }),
        };
        let s = schema(json!({ "append": { "type": "boolean" } }));
        let repaired = repair
            .repair(&call, "invalid parameters", Some(&s))
            .expect("repair");
        assert_eq!(repaired.id, "c1");
        assert_eq!(repaired.name, "write");
        assert_eq!(repaired.input, json!({ "append": false }));
    }

    #[test]
    fn standard_repair_gives_up_when_nothing_to_fix() {
        let repair = StandardToolCallRepair;
        let call = ToolCall {
            id: "c1".into(),
            name: "write".into(),
            input: json!({ "append": false }),
        };
        let s = schema(json!({ "append": { "type": "boolean" } }));
        assert!(repair.repair(&call, "boom", Some(&s)).is_none());
    }
}
