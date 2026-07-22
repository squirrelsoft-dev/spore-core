// tool_call_repair.go — deterministic coercion of weak-model tool arguments.
//
// Weak models (e.g. llama3.2) frequently emit tool arguments with the wrong
// JSON *type*: the string "false" where a bool is expected, "42" where a number
// is expected, or "[\"x\"]" where a sequence is expected. The strict
// parseParams rejects these, the tool returns a recoverable ToolOutputError,
// and weak models tend to loop on the same mistake.
//
// This module provides a pure, deterministic repair layer the harness applies
// between dispatch attempts. It is strictly additive: when no ToolCallRepair
// provider is wired into the harness, behaviour is byte-identical.
//
// Mirrors the Rust reference (tool_call_repair.rs). DetectProseResponse lives
// in prompt_tool_call.go (already present); the ProseResponse failure variant
// is defined here so all tool-call failure modes share one vocabulary.

package sporecore

import (
	"encoding/json"
	"strconv"
	"strings"
)

// ToolCallFailureKind discriminates how a model's tool-calling attempt failed.
type ToolCallFailureKind string

const (
	// ToolCallFailureMalformedJson: a tool call was emitted but its arguments
	// were not valid JSON of the expected shape. Recoverable via CoerceToolArgs.
	ToolCallFailureMalformedJson ToolCallFailureKind = "malformed_json"
	// ToolCallFailureProseResponse: tools were advertised, but the model
	// answered in prose instead of emitting a tool call. Recoverable by
	// escalating to prompt-based tool calling on the next turn.
	ToolCallFailureProseResponse ToolCallFailureKind = "prose_response"
	// ToolCallFailureUnknownTool: a tool call named a tool not in the registry.
	ToolCallFailureUnknownTool ToolCallFailureKind = "unknown_tool"
)

// ToolCallFailure is the classification type shared by the repair seam and the
// prompt-based fallback.
type ToolCallFailure struct {
	Kind ToolCallFailureKind
	// Text is the offending error/text (malformed JSON text, prose text, or the
	// unknown tool name).
	Text string
}

// ToolCallRepair is the strategy for repairing a tool call that failed dispatch
// with a recoverable error. Implementations are pure and side-effect-free: they
// inspect the call (and optionally its schema + the error text) and return a
// repaired ToolCall to re-dispatch, or ok=false to give up.
type ToolCallRepair interface {
	// Repair returns a repaired ToolCall to re-dispatch, or ok=false to give up.
	Repair(call ToolCall, errMsg string, schema *ToolSchema) (ToolCall, bool)
}

// StandardToolCallRepair is the default ToolCallRepair: applies CoerceToolArgs
// to the call's input. Returns a repaired call when coercion changed something,
// else ok=false.
type StandardToolCallRepair struct{}

// Repair returns a repaired ToolCall when CoerceToolArgs changed something.
func (StandardToolCallRepair) Repair(call ToolCall, _ string, schema *ToolSchema) (ToolCall, bool) {
	repaired, ok := CoerceToolArgs(call.Input, schema)
	if !ok {
		return ToolCall{}, false
	}
	return ToolCall{ID: call.ID, Name: call.Name, Input: repaired}, true
}

// CoerceToolArgs attempts deterministic type coercion of a tool call's
// top-level argument object so that string-encoded scalars/collections become
// the JSON types a weak model meant to emit.
//
// Coercions (driven by the JSON-schema's declared property types when a schema
// is provided; otherwise applied as conservative heuristics):
//   - "true" / "false" (case-insensitive) → bool
//   - numeric strings like "42" / "3.14" → number
//   - a string whose trimmed value parses as a JSON array or object → that
//     parsed array/object
//
// It also repairs a malformed argument KEY before resolving its type: weak
// models sometimes wrap the argument name in literal quotes ("content" instead
// of content), so the key misses the schema and the tool rejects the call for a
// "missing required field". The quote-stripped form is adopted when it matches
// a known property (or, with no schema, unconditionally).
//
// The function only recurses one level — into the top-level object's fields,
// which is where tool arguments live. It is conservative: an already-valid
// value is never touched. Returns the repaired input and true when any field
// changed, else false.
func CoerceToolArgs(input json.RawMessage, schema *ToolSchema) (json.RawMessage, bool) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(input, &obj); err != nil {
		return nil, false // tool args always live in a top-level object; leave alone
	}
	expected := propertyTypes(schema)
	changed := false
	out := make(map[string]json.RawMessage, len(obj))
	for key, value := range obj {
		normalized := normalizeKey(key, expected)
		if normalized != key {
			changed = true
		}
		if repaired, ok := coerceScalar(value, expected[normalized]); ok {
			changed = true
			out[normalized] = repaired
			continue
		}
		out[normalized] = value
	}
	if !changed {
		return nil, false
	}
	rep, err := json.Marshal(out)
	if err != nil {
		return nil, false
	}
	return rep, true
}

// propertyTypes extracts a field-name → type map from a tool's input schema
// ({"type":"object","properties":{ "field": {"type":"boolean"}, ... }}). A
// property whose type is an array (e.g. ["string","null"]) contributes no
// single token and is skipped.
func propertyTypes(schema *ToolSchema) map[string]string {
	m := map[string]string{}
	if schema == nil || len(schema.InputSchema) == 0 {
		return m
	}
	var top struct {
		Properties map[string]struct {
			Type string `json:"type"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(schema.InputSchema, &top); err != nil {
		return m
	}
	for name, prop := range top.Properties {
		if prop.Type != "" {
			m[name] = prop.Type
		}
	}
	return m
}

// normalizeKey repairs a possibly-malformed argument key. With a schema, only
// rewrite a key that misses the declared properties but whose quote-stripped
// form hits them; with no schema, strip one surrounding matched quote pair
// unconditionally (a quoted JSON key is never intentional). A stripped form that
// still isn't a known property is left untouched — we never fabricate a match.
func normalizeKey(key string, expected map[string]string) string {
	if _, ok := expected[key]; ok {
		return key
	}
	stripped := stripOneQuotePair(strings.TrimSpace(key))
	if stripped != key && (len(expected) == 0 || expected[stripped] != "") {
		return stripped
	}
	return key
}

// stripOneQuotePair strips exactly one pair of matching surrounding quotes
// ("…" or '…') from s. Returns s unchanged when it is not so wrapped.
func stripOneQuotePair(s string) string {
	if len(s) >= 2 {
		first, last := s[0], s[len(s)-1]
		if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

// coerceScalar coerces a single field value given its (optional) expected
// JSON-schema type. Returns the repaired value and true only when a coercion
// was applied.
func coerceScalar(value json.RawMessage, expected string) (json.RawMessage, bool) {
	// Only string values are candidates for coercion. Already-typed values are
	// assumed correct and never touched — this protects valid inputs.
	var s string
	if err := json.Unmarshal(value, &s); err != nil {
		return nil, false
	}
	trimmed := strings.TrimSpace(s)

	// 1. Boolean tokens. Coerce whenever a bool is expected, or when no schema
	//    type is known.
	if expected == "boolean" || expected == "" {
		switch strings.ToLower(trimmed) {
		case "true":
			return json.RawMessage("true"), true
		case "false":
			return json.RawMessage("false"), true
		}
	}

	// 2. Numeric strings. Prefer integer when the token has no fractional part.
	if expected == "number" || expected == "integer" || expected == "" {
		if n, ok := parseNumber(trimmed); ok {
			return n, true
		}
	}

	// 3. JSON-encoded array / object. A string whose trimmed content parses as
	//    a JSON array or object is very likely a mis-stringified collection.
	collectionExpected := expected == "array" || expected == "object" || expected == ""
	if collectionExpected && (strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "{")) {
		var parsed any
		if err := json.Unmarshal([]byte(trimmed), &parsed); err == nil {
			typeOK := expected == "" ||
				(expected == "array" && isArray(parsed)) ||
				(expected == "object" && isObject(parsed))
			if typeOK {
				if rep, err := json.Marshal(parsed); err == nil {
					return rep, true
				}
			}
		}
	}

	return nil, false
}

func isArray(v any) bool { _, ok := v.([]any); return ok }
func isObject(v any) bool {
	_, ok := v.(map[string]any)
	return ok
}

// parseNumber parses a trimmed numeric string into a JSON number. Integers are
// preferred over floats so "42" becomes 42 rather than 42.0. NaN/Inf (no JSON
// representation) are rejected.
func parseNumber(s string) (json.RawMessage, bool) {
	if s == "" {
		return nil, false
	}
	if i, err := strconv.ParseInt(s, 10, 64); err == nil {
		return json.RawMessage(strconv.FormatInt(i, 10)), true
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		// Reject NaN/Inf — they have no JSON number representation.
		rep, err := json.Marshal(f)
		if err == nil {
			return rep, true
		}
	}
	return nil, false
}

// IsParamError reports whether a recoverable tool-error message is a PARAMETER
// problem (a malformed call) rather than a command/runtime failure (e.g. a
// non-zero cargo test exit). Only parameter-shaped errors benefit from the
// schema+hint enrichment; enriching a command failure misleads the model — and
// offline judges — into seeing a "schema violation" that did not happen.
// Mirrors the Rust reference's is_param_error.
func IsParamError(message string) bool {
	m := strings.ToLower(message)
	return strings.Contains(m, "invalid parameters") ||
		strings.Contains(m, "invalid type") ||
		strings.Contains(m, "missing required field") ||
		strings.Contains(m, "missing field") ||
		strings.Contains(m, "schema validation failed") ||
		strings.Contains(m, "unknown field") ||
		strings.Contains(m, "unknown variant")
}
