// Output-schema delivery + enforcement (issue #139).
//
// ReactConfig.Output (a *SchemaRef) was presence-validated by the
// ExecutionRegistry at startup but IGNORED at runtime: the resolved schema was
// never delivered to the model and never validated the terminal. This file is
// the hand-rolled validator + the frozen literals that make delivery +
// enforcement deterministic and BYTE-IDENTICAL across the four language ports
// (Rust is the reference: rust/crates/spore-core/src/output_schema.rs).
//
// # Validator subset (matches the Ollama `format` channel)
//
// Only these JSON-schema keywords are honored — NO off-the-shelf validator
// (they diverge across languages and would break byte-identical fixtures):
// `type` / `required` / `properties` / `enum`. Anything else in the schema is
// ignored. validateOutput returns "" on a match or one of the FROZEN validator
// error strings below.
//
// # Evaluation order (first-match-wins — FROZEN)
//
//  1. Output does not parse as JSON → errNotJSON.
//  2. Root value's JSON type ≠ schema `type` → errRootType.
//  3. A `required` property absent (iterate `required` in ARRAY order) →
//     errMissingRequired.
//  4. A present property's value type ≠ its subschema `type` → errPropertyType.
//  5. A present property's value ∉ its subschema `enum` → errPropertyEnum.
//  6. Otherwise valid.
//
// # Determinism rules (parity-critical)
//
//   - `properties` are iterated in LEXICOGRAPHICALLY-SORTED key order for steps
//     4 and 5 (Go map iteration is randomized — keys MUST be sorted explicitly;
//     this is the highest-risk parity bug).
//   - `required` is checked in its given ARRAY order; `enum` membership is
//     order-independent but the `{enum}` rendering in the message is SORTED.
//   - `{enum}` and `{value}` are rendered via canonicalizeJSON (canonical
//     compact key-sorted JSON), e.g. ["error","ok"], "maybe". The enum SORT in
//     the message is INTENTIONAL (determinism over author intent); the
//     membership check itself is order-independent.
//   - JSON type names: object / array / string / number / integer / boolean /
//     null.
//   - `integer` is a SEMANTIC check, not a token check: a number with no
//     fractional part passes as `integer` (42.0 passes, 42.5 fails). Go
//     encoding/json decodes all JSON numbers to float64 by default, so the value
//     tree is decoded with UseNumber() and the integral check inspects the
//     json.Number's fractional part — NOT the token type.
//
// These literals are HASH-LOAD-BEARING — the four ports MUST reproduce these
// exact bytes.

package sporecore

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

// feedbackMessage is the validation-failure feedback appended as a USER-role
// message before a retry (#139, AC2). The single argument is substituted with
// exactly one validator error string. Exact bytes (single spaces, no trailing
// newline) — HASH-LOAD-BEARING.
func feedbackMessage(validatorErr string) string {
	return "Your previous response did not match the required output schema. " +
		validatorErr +
		" Reply with only a JSON value that satisfies the schema."
}

// errNotJSON is the validator error (step 1): the model's response did not parse
// as JSON.
const errNotJSON = "The response was not valid JSON."

// errRootType is the validator error (step 2): the root value's JSON type
// differs from the schema's `type`.
func errRootType(expected, actual string) string {
	return fmt.Sprintf("Expected type %q but found %q.", expected, actual)
}

// errMissingRequired is the validator error (step 3): a `required` property is
// absent.
func errMissingRequired(name string) string {
	return fmt.Sprintf("Missing required property %q.", name)
}

// errPropertyType is the validator error (step 4): a present property's value
// type differs from its subschema `type`.
func errPropertyType(name, expected, actual string) string {
	return fmt.Sprintf("Property %q should be type %q but found %q.", name, expected, actual)
}

// errPropertyEnum is the validator error (step 5): a present property's value is
// not in its subschema `enum`. enumJSON and valueJSON are canonical compact
// key-sorted JSON.
func errPropertyEnum(name, enumJSON, valueJSON string) string {
	return fmt.Sprintf("Property %q must be one of %s but found %s.", name, enumJSON, valueJSON)
}

// jsonTypeName returns the JSON type name of v (the JSON set; an integer is
// reported as "number" here — the integer DISTINCTION is a schema-side semantic
// check, see matchesType). v is a value tree produced by encoding/json with
// UseNumber() (so numbers are json.Number, not float64).
func jsonTypeName(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case json.Number:
		return "number"
	case string:
		return "string"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	default:
		return "null"
	}
}

// isIntegral reports whether the json.Number n has no fractional part (42.0 is
// integral, 42.5 is not). It parses the number as a float and checks the
// fractional part — NOT the token form. Mirrors Rust's f.fract() == 0.0. A
// number that fails to parse is not integral (defensive; UseNumber values always
// parse).
func isIntegral(n json.Number) bool {
	f, err := n.Float64()
	if err != nil {
		return false
	}
	return f == math.Trunc(f)
}

// matchesType reports whether v satisfies the schema type name expected. v is a
// UseNumber() value tree. `integer` is a SEMANTIC check: a JSON number with no
// fractional part passes (42.0 passes, 42.5 fails); a `number` schema accepts
// any JSON number (integer or not). An unknown/absent type name is treated as
// satisfied (the subset only constrains the seven JSON type names).
func matchesType(v any, expected string) bool {
	switch expected {
	case "object":
		_, ok := v.(map[string]any)
		return ok
	case "array":
		_, ok := v.([]any)
		return ok
	case "string":
		_, ok := v.(string)
		return ok
	case "boolean":
		_, ok := v.(bool)
		return ok
	case "null":
		return v == nil
	case "number":
		_, ok := v.(json.Number)
		return ok
	case "integer":
		n, ok := v.(json.Number)
		if !ok {
			return false
		}
		return isIntegral(n)
	default:
		// Unknown/absent type name: the subset only constrains the seven JSON
		// type names; anything else is not enforced.
		return true
	}
}

// schemaType returns the schema's `type` keyword as a string, and whether it was
// present and a string. schema is a UseNumber() value tree.
func schemaType(schema any) (string, bool) {
	obj, ok := schema.(map[string]any)
	if !ok {
		return "", false
	}
	t, ok := obj["type"].(string)
	return t, ok
}

// decodeJSONTree parses raw JSON into a value tree using json.Number for numbers
// (so the `integer` semantic check can inspect the fractional part). Returns
// (nil, false) on a parse error.
func decodeJSONTree(raw []byte) (any, bool) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var tree any
	if err := dec.Decode(&tree); err != nil {
		return nil, false
	}
	return tree, true
}

// canonicalizeSchema returns the canonical compact key-sorted JSON encoding of a
// schema RawMessage (#139). The single cross-language source of truth for the
// schema bytes embedded in the directive seed AND reported on a violation, so
// they are identical across the four ports regardless of each language's map
// insertion order. A non-parseable schema (defensive) is returned verbatim as a
// trimmed string.
func canonicalizeSchema(schema json.RawMessage) string {
	tree, ok := decodeJSONTree([]byte(schema))
	if !ok {
		return strings.TrimSpace(string(schema))
	}
	return canonicalizeJSON(tree)
}

// validateOutput validates the model's terminal response text against schema
// (#139). schema is the resolved output schema (a json.RawMessage of canonical
// or author-order JSON).
//
// Returns "" on a match, or the FIRST (lowest-numbered) FROZEN validator error
// in the evaluation order. The subset honored is `type` / `required` /
// `properties` / `enum`; everything else in the schema is ignored. Iteration
// order is fixed (see the file docs) so the returned error is byte-identical
// across languages.
func validateOutput(response string, schema json.RawMessage) string {
	// Step 1: parse the response (trim surrounding whitespace, matching Rust's
	// serde_json::from_str(response.trim())).
	value, ok := decodeJSONTree([]byte(strings.TrimSpace(response)))
	if !ok {
		return errNotJSON
	}

	schemaTree, ok := decodeJSONTree([]byte(schema))
	if !ok {
		// A non-parseable schema constrains nothing (defensive; the registry
		// only stores valid JSON schemas).
		return ""
	}

	// Step 2: root type.
	if expected, present := schemaType(schemaTree); present {
		if !matchesType(value, expected) {
			return errRootType(expected, jsonTypeName(value))
		}
	}

	obj, _ := value.(map[string]any)
	schemaObj, _ := schemaTree.(map[string]any)

	// Step 3: required (ARRAY order). Only meaningful for an object value; a
	// non-object value already passed step 2 (or had no `type`), so a `required`
	// list applies to its members only when it IS an object.
	if schemaObj != nil {
		if required, ok := schemaObj["required"].([]any); ok && obj != nil {
			for _, r := range required {
				name, ok := r.(string)
				if !ok {
					continue
				}
				if _, present := obj[name]; !present {
					return errMissingRequired(name)
				}
			}
		}
	}

	// Steps 4 + 5 iterate `properties` in LEXICOGRAPHICALLY-SORTED key order.
	var props map[string]any
	if schemaObj != nil {
		props, _ = schemaObj["properties"].(map[string]any)
	}
	if obj != nil && props != nil {
		keys := make([]string, 0, len(props))
		for k := range props {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		// Step 4: present-property type.
		for _, key := range keys {
			present, has := obj[key]
			if !has {
				continue
			}
			expected, ok := schemaType(props[key])
			if !ok {
				continue
			}
			if !matchesType(present, expected) {
				return errPropertyType(key, expected, jsonTypeName(present))
			}
		}

		// Step 5: present-property enum membership.
		for _, key := range keys {
			present, has := obj[key]
			if !has {
				continue
			}
			sub, ok := props[key].(map[string]any)
			if !ok {
				continue
			}
			enumArr, ok := sub["enum"].([]any)
			if !ok {
				continue
			}
			member := false
			presentCanon := canonicalizeJSON(present)
			for _, e := range enumArr {
				if canonicalizeJSON(e) == presentCanon {
					member = true
					break
				}
			}
			if !member {
				// Render the enum SORTED (determinism over author intent) and the
				// value canonically — both via canonicalizeJSON.
				rendered := make([]string, len(enumArr))
				for i, e := range enumArr {
					rendered[i] = canonicalizeJSON(e)
				}
				sort.Strings(rendered)
				enumJSON := "[" + strings.Join(rendered, ",") + "]"
				return errPropertyEnum(key, enumJSON, presentCanon)
			}
		}
	}

	// Step 6: valid.
	return ""
}
