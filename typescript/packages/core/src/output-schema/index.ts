/**
 * Output-schema delivery + enforcement (issue #139).
 *
 * `ReactConfig.output` was presence-validated by the
 * {@link "../harness/execution-registry.js".ExecutionRegistry} at startup but
 * IGNORED at runtime: the resolved schema was never delivered to the model and
 * never validated the terminal. This module is the hand-rolled validator + the
 * frozen literals that make delivery + enforcement deterministic and
 * BYTE-IDENTICAL across the four language ports (Rust is the reference).
 *
 * ## Validator subset (matches the Ollama `format` channel)
 *
 * Only these JSON-schema keywords are honored — NO off-the-shelf validator
 * (they diverge across languages and would break byte-identical fixtures):
 * `type` / `required` / `properties` / `enum`. Anything else in the schema is
 * ignored. {@link validateOutput} returns `null` on a match or an `error`
 * string (one of the FROZEN validator error strings below).
 *
 * ## Evaluation order (first-match-wins — FROZEN)
 *
 * 1. Output does not parse as JSON → {@link ERR_NOT_JSON}.
 * 2. Root value's JSON type ≠ schema `type` → {@link errRootType}.
 * 3. A `required` property absent (iterate `required` in ARRAY order) →
 *    {@link errMissingRequired}.
 * 4. A present property's value type ≠ its subschema `type` →
 *    {@link errPropertyType}.
 * 5. A present property's value ∉ its subschema `enum` →
 *    {@link errPropertyEnum}.
 * 6. Otherwise valid.
 *
 * ## Determinism rules (parity-critical — TS does NOT auto-sort JSON keys)
 *
 * - `properties` are iterated in LEXICOGRAPHICALLY-SORTED key order for steps 4
 *   and 5 (NOT object insertion order — the languages disagree on it).
 * - `required` is checked in its given ARRAY order; `enum` membership is
 *   order-independent but the `{enum}` rendering in the message is sorted.
 * - `{enum}` and `{value}` are rendered via
 *   {@link "../model/index.js".canonicalizeJson} (canonical compact key-sorted
 *   JSON), e.g. `["error","ok"]`, `"maybe"`. The enum SORT in the message is
 *   INTENTIONAL (determinism over author intent); the membership check itself is
 *   order-independent.
 * - JSON type names: `object` / `array` / `string` / `number` / `integer` /
 *   `boolean` / `null`.
 * - `integer` is a SEMANTIC check, not a token check: a number with no
 *   fractional part passes as `integer` (`42.0` passes, `42.5` fails).
 *
 * These literals are HASH-LOAD-BEARING — the Rust/Python/Go ports reproduce
 * these exact bytes.
 */
import { canonicalizeJson } from "../model/index.js";

/**
 * The validation-failure feedback appended as a USER-role message before a retry
 * (#139, AC2). `{error}` is substituted with exactly one validator error string.
 * Exact bytes (single spaces, no trailing newline) — HASH-LOAD-BEARING.
 */
export function feedbackMessage(error: string): string {
  return (
    `Your previous response did not match the required output schema. ${error} ` +
    "Reply with only a JSON value that satisfies the schema."
  );
}

/** Validator error (step 1): the model's response did not parse as JSON. */
export const ERR_NOT_JSON = "The response was not valid JSON.";

/**
 * Validator error (step 2): the root value's JSON type differs from the schema's
 * `type`.
 */
export function errRootType(expected: string, actual: string): string {
  return `Expected type "${expected}" but found "${actual}".`;
}

/** Validator error (step 3): a `required` property is absent. */
export function errMissingRequired(name: string): string {
  return `Missing required property "${name}".`;
}

/**
 * Validator error (step 4): a present property's value type differs from its
 * subschema `type`.
 */
export function errPropertyType(name: string, expected: string, actual: string): string {
  return `Property "${name}" should be type "${expected}" but found "${actual}".`;
}

/**
 * Validator error (step 5): a present property's value is not in its subschema
 * `enum`. `enumJson` and `valueJson` are canonical compact key-sorted JSON.
 */
export function errPropertyEnum(name: string, enumJson: string, valueJson: string): string {
  return `Property "${name}" must be one of ${enumJson} but found ${valueJson}.`;
}

/** A parsed JSON value (the `serde_json::Value` analogue). */
type JsonValue = null | boolean | number | string | JsonValue[] | { [k: string]: JsonValue };

/**
 * The JSON type name of `v` (the JSON set; an integral number is still reported
 * as `number` here — the integer DISTINCTION is a schema-side semantic check,
 * see {@link matchesType}).
 */
function jsonTypeName(v: JsonValue): string {
  if (v === null) return "null";
  if (typeof v === "boolean") return "boolean";
  if (typeof v === "number") return "number";
  if (typeof v === "string") return "string";
  if (Array.isArray(v)) return "array";
  return "object";
}

/**
 * Whether `v` satisfies the schema type name `expected`. `integer` is a SEMANTIC
 * check: a JSON number with no fractional part passes (`42.0` passes, `42.5`
 * fails); a `number` schema accepts any JSON number (integer or not). Unknown
 * type names are treated as satisfied (the subset only constrains the seven JSON
 * type names).
 */
function matchesType(v: JsonValue, expected: string): boolean {
  switch (expected) {
    case "object":
      return typeof v === "object" && v !== null && !Array.isArray(v);
    case "array":
      return Array.isArray(v);
    case "string":
      return typeof v === "string";
    case "boolean":
      return typeof v === "boolean";
    case "null":
      return v === null;
    case "number":
      return typeof v === "number";
    case "integer":
      // A number with no fractional part is integral (42.0 passes, 42.5 fails).
      // NOT a token check; `JSON.parse` collapses `42.0` to `42` already, but a
      // genuine `42.5` is caught by `Number.isInteger`.
      return typeof v === "number" && Number.isInteger(v);
    default:
      return true;
  }
}

/** The schema's `type` keyword as a string, if present. */
function schemaType(schema: JsonValue): string | undefined {
  if (typeof schema === "object" && schema !== null && !Array.isArray(schema)) {
    const t = schema.type;
    if (typeof t === "string") return t;
  }
  return undefined;
}

/** Read an object subschema's `properties` map, if present. */
function schemaProperties(schema: JsonValue): { [k: string]: JsonValue } | undefined {
  if (typeof schema === "object" && schema !== null && !Array.isArray(schema)) {
    const p = schema.properties;
    if (typeof p === "object" && p !== null && !Array.isArray(p)) return p;
  }
  return undefined;
}

/** Read a schema's `required` array, if present. */
function schemaRequired(schema: JsonValue): JsonValue[] | undefined {
  if (typeof schema === "object" && schema !== null && !Array.isArray(schema)) {
    const r = schema.required;
    if (Array.isArray(r)) return r;
  }
  return undefined;
}

/** Read a subschema's `enum` array, if present. */
function schemaEnum(schema: JsonValue): JsonValue[] | undefined {
  if (typeof schema === "object" && schema !== null && !Array.isArray(schema)) {
    const e = schema.enum;
    if (Array.isArray(e)) return e;
  }
  return undefined;
}

/** Whether `a` and `b` are structurally equal JSON values (enum membership). */
function jsonEquals(a: JsonValue, b: JsonValue): boolean {
  return canonicalizeJson(a) === canonicalizeJson(b);
}

/**
 * Validate the model's terminal `response` text against `schema` (#139).
 *
 * Returns `null` on a match, or the FIRST (lowest-numbered) FROZEN validator
 * error in the evaluation order. The subset honored is
 * `type` / `required` / `properties` / `enum`; everything else in the schema is
 * ignored. Iteration order is fixed (see the module docs) so the returned error
 * is byte-identical across languages.
 */
export function validateOutput(response: string, schema: unknown): string | null {
  // Step 1: parse.
  let value: JsonValue;
  try {
    value = JSON.parse(response.trim()) as JsonValue;
  } catch {
    return ERR_NOT_JSON;
  }

  const schemaVal = schema as JsonValue;

  // Step 2: root type.
  const rootExpected = schemaType(schemaVal);
  if (rootExpected !== undefined && !matchesType(value, rootExpected)) {
    return errRootType(rootExpected, jsonTypeName(value));
  }

  const obj =
    typeof value === "object" && value !== null && !Array.isArray(value) ? value : undefined;
  const props = schemaProperties(schemaVal);

  // Step 3: required (ARRAY order). Only meaningful for an object value; a
  // non-object value already passed step 2 (or had no `type`).
  const required = schemaRequired(schemaVal);
  if (required !== undefined && obj !== undefined) {
    for (const r of required) {
      if (typeof r === "string" && !Object.prototype.hasOwnProperty.call(obj, r)) {
        return errMissingRequired(r);
      }
    }
  }

  // Steps 4 + 5 iterate `properties` in LEXICOGRAPHICALLY-SORTED key order.
  if (obj !== undefined && props !== undefined) {
    const keys = Object.keys(props).sort();

    // Step 4: present-property type.
    for (const key of keys) {
      const subschema = props[key] as JsonValue;
      const expected = schemaType(subschema);
      const present = Object.prototype.hasOwnProperty.call(obj, key)
        ? (obj[key] as JsonValue)
        : undefined;
      if (present !== undefined && expected !== undefined) {
        if (!matchesType(present, expected)) {
          return errPropertyType(key, expected, jsonTypeName(present));
        }
      }
    }

    // Step 5: present-property enum membership.
    for (const key of keys) {
      const subschema = props[key] as JsonValue;
      const enumArr = schemaEnum(subschema);
      const present = Object.prototype.hasOwnProperty.call(obj, key)
        ? (obj[key] as JsonValue)
        : undefined;
      if (present !== undefined && enumArr !== undefined) {
        const member = enumArr.some((e) => jsonEquals(e, present));
        if (!member) {
          // Render the enum SORTED (determinism over author intent) and the
          // value canonically — both via canonicalizeJson.
          const sorted = [...enumArr].sort((a, b) =>
            canonicalizeJson(a) < canonicalizeJson(b)
              ? -1
              : canonicalizeJson(a) > canonicalizeJson(b)
                ? 1
                : 0,
          );
          const enumJson = "[" + sorted.map((e) => canonicalizeJson(e)).join(",") + "]";
          const valueJson = canonicalizeJson(present);
          return errPropertyEnum(key, enumJson, valueJson);
        }
      }
    }
  }

  // Step 6: valid.
  return null;
}
