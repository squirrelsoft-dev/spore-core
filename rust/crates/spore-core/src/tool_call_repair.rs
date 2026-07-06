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
        // Repair a malformed argument KEY before resolving its type. Weak models
        // sometimes wrap the argument name in literal quotes (`"content"` instead
        // of `content`) — via hand-written `<input>` JSON or a `<parameter="K">`
        // tag — so the key misses the schema and the tool rejects the call for a
        // "missing required field". Adopt the quote-stripped form when it matches
        // a known property (or, with no schema, unconditionally — a quoted JSON
        // key is never intentional).
        let normalized = normalize_key(key, &expected_types);
        if normalized != *key {
            changed = true;
        }
        let expected = expected_types.get(normalized.as_str()).map(String::as_str);
        match coerce_scalar(value, expected) {
            Some(repaired) => {
                changed = true;
                out.insert(normalized, repaired);
            }
            None => {
                out.insert(normalized, value.clone());
            }
        }
    }

    if changed {
        Some(Value::Object(out))
    } else {
        None
    }
}

/// Normalize a possibly-malformed argument key. Weak models sometimes emit the
/// key wrapped in literal quotes (`"content"` / `'path'`), which makes it miss
/// the schema so the tool rejects the call for a missing required field. With a
/// schema, only rewrite a key that misses the declared properties but whose
/// quote-stripped form hits them; with no schema, strip one surrounding matched
/// quote pair unconditionally (a quoted JSON key is never intentional). A
/// stripped form that still isn't a known property is left untouched — we never
/// fabricate a schema match.
fn normalize_key(key: &str, expected: &BTreeMap<String, String>) -> String {
    if expected.contains_key(key) {
        return key.to_string();
    }
    let stripped = strip_one_quote_pair(key.trim());
    if stripped != key && (expected.is_empty() || expected.contains_key(stripped)) {
        return stripped.to_string();
    }
    key.to_string()
}

/// Strip exactly one pair of matching surrounding quotes (`"…"` or `'…'`) from a
/// string. Returns the input unchanged when it is not so wrapped. The quote
/// bytes are ASCII, so byte-slicing at the ends is UTF-8-safe.
fn strip_one_quote_pair(s: &str) -> &str {
    let b = s.as_bytes();
    if b.len() >= 2 {
        let (first, last) = (b[0], b[b.len() - 1]);
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &s[1..s.len() - 1];
        }
    }
    s
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

// ============================================================================
// Failure classification (issue: prompt-based tool-call fallback)
// ============================================================================

/// How a model's tool-calling attempt failed.
///
/// Classification type shared by the repair seam (malformed JSON, unknown tool)
/// and the prompt-based fallback (prose responses). The harness loop owns
/// *prose* detection — a prose response emits no tool call, so it never reaches
/// the dispatch-error [`ToolCallRepair`] seam — but the variant lives here so
/// all tool-call failure modes have one vocabulary for observability and
/// consistency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallFailure {
    /// A tool call was emitted but its arguments were not valid JSON of the
    /// expected shape. Carries the offending error/text. Recoverable via
    /// [`coerce_tool_args`].
    MalformedJson(String),
    /// Tools were advertised, but the model answered in prose instead of
    /// emitting a tool call. Carries the prose text. Recoverable by escalating
    /// to prompt-based tool calling on the next turn.
    ProseResponse(String),
    /// A tool call named a tool not present in the registry. Carries the name.
    UnknownTool(String),
}

/// Conservative heuristic: did the model respond in prose when a tool call was
/// the expected next step?
///
/// Returns `Some(ToolCallFailure::ProseResponse(..))` only when **both**:
/// 1. tools were advertised this turn (`tools_advertised`), and
/// 2. the response text contains an explicit action-intent phrase suggesting the
///    model *meant* to act (e.g. "I'll use the … tool", "let me call …").
///
/// The bias is deliberately toward false negatives: a missed prose response
/// costs one extra turn, but a false positive activates prompt-based mode for a
/// model that was simply giving a final answer. A bare final answer with no
/// action-intent language is therefore **not** classified as a prose response.
pub fn detect_prose_response(text: &str, tools_advertised: bool) -> Option<ToolCallFailure> {
    if !tools_advertised {
        return None;
    }
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Curated action-intent phrases. Lower-cased substring match — conservative
    // and cheap. Each phrase strongly implies "I am about to use a tool".
    const ACTION_PHRASES: &[&str] = &[
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
        // Research/inspection verbs an agent narrates instead of calling a tool
        // (e.g. Qwen: "I'll list the directory structure for you."). Each strongly
        // implies an action it should have performed via a tool.
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
    ];
    let lower = trimmed.to_ascii_lowercase();
    if ACTION_PHRASES.iter().any(|p| lower.contains(p)) {
        Some(ToolCallFailure::ProseResponse(trimmed.to_string()))
    } else {
        None
    }
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
    fn repairs_a_quoted_key_against_the_schema() {
        // The send_message regression: weak models wrap the arg name in quotes
        // (`"content"`), so the key misses the schema and the tool rejects the
        // call for a missing required field. The quote-stripped key is adopted.
        let s = schema(json!({ "content": { "type": "string" } }));
        let input = json!({ "\"content\"": "Adding slugify to util.rs." });
        let out = coerce_tool_args(&input, Some(&s)).expect("should repair the key");
        assert_eq!(out, json!({ "content": "Adding slugify to util.rs." }));
    }

    #[test]
    fn repairs_a_quoted_key_with_no_schema() {
        // No schema → a quoted JSON key is never intentional, so strip it.
        let input = json!({ "'path'": "src/util.rs" });
        let out = coerce_tool_args(&input, None).expect("should strip the quoted key");
        assert_eq!(out, json!({ "path": "src/util.rs" }));
    }

    #[test]
    fn repairs_a_quoted_key_and_coerces_its_value_together() {
        // Both defects at once (the set_plan shape): quoted key AND a
        // string-encoded array value.
        let s = schema(json!({ "tasks": { "type": "array" } }));
        let input = json!({ "\"tasks\"": "[\"a\",\"b\"]" });
        let out = coerce_tool_args(&input, Some(&s)).expect("should repair key + value");
        assert_eq!(out, json!({ "tasks": ["a", "b"] }));
    }

    #[test]
    fn does_not_invent_a_key_that_misses_the_schema() {
        // A quoted key whose stripped form still isn't a property is left as-is
        // (with an already-valid value → no change at all). We never fabricate a
        // schema match.
        let s = schema(json!({ "content": { "type": "string" } }));
        let input = json!({ "\"bogus\"": "x" });
        assert!(coerce_tool_args(&input, Some(&s)).is_none());
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

    // --- prose detection -----------------------------------------------------

    #[test]
    fn prose_detected_on_action_intent_with_tools() {
        let got = detect_prose_response("Sure, I'll use the calculator tool to add these.", true);
        assert!(matches!(got, Some(ToolCallFailure::ProseResponse(_))));
    }

    #[test]
    fn prose_detected_case_insensitive() {
        let got = detect_prose_response("LET ME CALL the search tool now.", true);
        assert!(matches!(got, Some(ToolCallFailure::ProseResponse(_))));
    }

    #[test]
    fn prose_detected_on_research_verb_narration() {
        // A model that narrates an inspection action instead of calling a tool —
        // the exact "I'll list the directory…" wording the agent must be nudged on.
        for text in [
            "I'll list the top-level directory structure for you.",
            "Let me inspect the key configuration files.",
            "I'll read the README and report findings.",
            "I'll start by exploring the source directory.",
        ] {
            assert!(
                matches!(
                    detect_prose_response(text, true),
                    Some(ToolCallFailure::ProseResponse(_))
                ),
                "should detect action-narration: {text:?}"
            );
        }
    }

    #[test]
    fn prose_not_detected_without_tools_advertised() {
        // Even with action-intent language, no tools advertised → no escalation.
        assert!(detect_prose_response("I'll use the calculator.", false).is_none());
    }

    #[test]
    fn prose_not_detected_for_plain_final_answer() {
        // Conservative: a final answer with no action-intent language must NOT
        // trip the heuristic (prefer false negatives).
        assert!(detect_prose_response("The answer is 42.", true).is_none());
    }

    #[test]
    fn prose_not_detected_for_empty_text() {
        assert!(detect_prose_response("   ", true).is_none());
    }
}
