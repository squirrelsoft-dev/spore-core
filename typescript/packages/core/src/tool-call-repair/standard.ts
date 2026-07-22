/**
 * Tool-call repair — mirrors `rust/crates/spore-core/src/tool_call_repair.rs`.
 *
 * A tool call that fails dispatch with a *recoverable* error (typically
 * `invalid_parameters`) can be coerced and re-dispatched. {@link coerceToolArgs}
 * is the pure coercion; {@link StandardToolCallRepair} wraps it as a strategy.
 * {@link detectProseResponse} (re-exported from the prompt-tool-call module for
 * the unified {@link ToolCallFailure} vocabulary) classifies prose responses.
 */

import type { ToolCall, ToolSchema as ModelToolSchema } from "../model/schemas.js";
import type { ToolSchema as RegisteredToolSchema } from "../tool-registry/types.js";

/** A schema with a JSON-schema body under either `parameters` or `input_schema`. */
export type RepairSchema = ModelToolSchema | RegisteredToolSchema;

// ============================================================================
// Failure classification
// ============================================================================

/** How a model's tool-calling attempt failed. Mirrors Rust `ToolCallFailure`. */
export type ToolCallFailure =
  | { kind: "malformed_json"; text: string }
  | { kind: "prose_response"; text: string }
  | { kind: "unknown_tool"; name: string };

// ============================================================================
// coerce_tool_args
// ============================================================================

/**
 * Coerce a malformed tool-call `input` toward its expected shape. Returns the
 * repaired input only when a coercion was applied, else `null` (leave alone).
 *
 * - Repairs malformed argument KEYS (a key wrapped in literal quotes that
 *   misses the schema; the quote-stripped form is adopted when it matches a
 *   known property, or unconditionally when no schema is available).
 * - Coerces string SCALAR values: `"true"`/`"false"` → bool, numeric strings →
 *   number (integer preferred), and a string whose trimmed content parses as a
 *   JSON array/object → the parsed collection.
 *
 * Already-typed values are never touched — this protects valid inputs.
 */
export function coerceToolArgs(
  input: unknown,
  schema?: RepairSchema,
): unknown | null {
  if (typeof input !== "object" || input === null || Array.isArray(input)) return null;
  const obj = input as Record<string, unknown>;
  const expectedTypes = schema != null ? propertyTypes(schema) : new Map<string, string>();

  let changed = false;
  const out: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    const normalized = normalizeKey(key, expectedTypes);
    if (normalized !== key) changed = true;
    const expected = expectedTypes.get(normalized) ?? null;
    const repaired = coerceScalar(value, expected);
    if (repaired !== null) {
      changed = true;
      out[normalized] = repaired;
    } else {
      out[normalized] = value;
    }
  }

  return changed ? out : null;
}

/** Map of field-name → expected JSON-schema "type" token. Reads `parameters` (registry schema) OR `input_schema` (model schema). */
function propertyTypes(schema: RepairSchema): Map<string, string> {
  const map = new Map<string, string>();
  const params = (schema as { parameters?: unknown; input_schema?: unknown }).parameters ??
    (schema as { input_schema?: unknown }).input_schema;
  const props = (params as { properties?: Record<string, { type?: unknown }> } | undefined)?.properties;
  if (props == null || typeof props !== "object") return map;
  for (const [name, prop] of Object.entries(props)) {
    if (prop != null && typeof prop === "object" && typeof prop.type === "string") {
      map.set(name, prop.type);
    }
  }
  return map;
}

/** Normalize a possibly-malformed argument key (strip one quote pair). */
function normalizeKey(key: string, expected: Map<string, string>): string {
  if (expected.has(key)) return key;
  const stripped = stripOneQuotePair(key.trim());
  if (stripped !== key && (expected.size === 0 || expected.has(stripped))) {
    return stripped;
  }
  return key;
}

/** Strip exactly one pair of matching surrounding quotes (`"…"` or `'…'`). */
function stripOneQuotePair(s: string): string {
  if (s.length >= 2) {
    const first = s[0]!;
    const last = s[s.length - 1]!;
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return s.slice(1, -1);
    }
  }
  return s;
}

/** Coerce a single field value given its (optional) expected JSON-schema type. */
function coerceScalar(value: unknown, expected: string | null): unknown | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();

  // 1. Boolean tokens.
  if (expected === "boolean" || expected === null) {
    const lower = trimmed.toLowerCase();
    if (lower === "true") return true;
    if (lower === "false") return false;
  }

  // 2. Numeric strings (integer preferred).
  if (expected === "number" || expected === "integer" || expected === null) {
    const n = parseNumber(trimmed);
    if (n !== null) return n;
  }

  // 3. JSON-encoded array / object.
  const collectionExpected =
    expected === "array" || expected === "object" || expected === null;
  if (collectionExpected && (trimmed.startsWith("[") || trimmed.startsWith("{"))) {
    try {
      const parsed = JSON.parse(trimmed);
      if (typeof parsed === "object" && parsed !== null) {
        const typeOk =
          expected === "array" ? Array.isArray(parsed) : expected === "object" ? !Array.isArray(parsed) : true;
        if (typeOk) return parsed;
      }
    } catch {
      /* not JSON — leave alone */
    }
  }

  return null;
}

/** Parse a trimmed numeric string; integers preferred over floats. */
function parseNumber(s: string): number | null {
  if (s === "") return null;
  // Integer?
  if (/^-?\d+$/.test(s)) {
    const i = Number(s);
    if (Number.isSafeInteger(i)) return i;
  }
  const f = Number(s);
  if (Number.isFinite(f) && /^-?\d+(\.\d+)?([eE][-+]?\d+)?$/.test(s)) {
    return f;
  }
  return null;
}

// ============================================================================
// ToolCallRepair trait + StandardToolCallRepair
// ============================================================================

/**
 * Strategy for repairing a tool call that failed dispatch with a *recoverable*
 * error. Implementations are pure and side-effect-free: they inspect the call
 * (and optionally its schema + the error text) and return a repaired
 * {@link ToolCall} to re-dispatch, or `null` to give up.
 */
export interface ToolCallRepair {
  repair(call: ToolCall, error: string, schema?: RepairSchema): ToolCall | null;
}

/**
 * Default {@link ToolCallRepair}: applies {@link coerceToolArgs} to the call's
 * input. Returns a repaired call when coercion changed something, else `null`.
 */
export class StandardToolCallRepair implements ToolCallRepair {
  repair(call: ToolCall, _error: string, schema?: RepairSchema): ToolCall | null {
    const repairedInput = coerceToolArgs(call.input, schema);
    if (repairedInput === null) return null;
    return { id: call.id, name: call.name, input: repairedInput };
  }
}