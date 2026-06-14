/**
 * Cross-language request hashing (spore-core issues #37, #38).
 *
 * `requestHash(req)` returns a 16-character lowercase hex string that uniquely
 * identifies a `ModelRequest`. All four language implementations of
 * `RecordingModelInterface` and `ReplayModelInterface` must produce the same
 * hash for the same request. The cross-language consistency fixture lives at
 * `fixtures/model_hashing/cases.json`.
 *
 * Algorithm:
 *   1. Serialise the `ModelRequest` to canonical JSON (object keys sorted
 *      lexicographically by ASCII codepoint, no insignificant whitespace,
 *      standard JSON string escaping).
 *   2. SHA-256 the UTF-8 bytes.
 *   3. Hex-encode the first 8 bytes (16 hex characters representing the
 *      leading u64).
 *
 * The canonical shape mirrors what Rust's serde emits for `ModelRequest`:
 * `Option` fields on `ModelParams` always serialize (as `null` when unset),
 * tagged-union variants embed a `"type"` discriminator next to the inner
 * struct's fields, and `ToolResult.is_error` is always present.
 */
import { createHash } from "node:crypto";

import type { ModelRequest } from "./schemas.js";

/** Public entry point. Returns 16 lowercase hex characters. */
export function requestHash(request: ModelRequest): string {
  const canonical = canonicalize(toCanonicalValue(request));
  const digest = createHash("sha256").update(canonical, "utf8").digest();
  return digest.subarray(0, 8).toString("hex");
}

/**
 * Canonical compact key-sorted JSON encoding of `value` (object keys sorted
 * lexicographically by ASCII codepoint, no insignificant whitespace, standard
 * JSON string escaping). The single cross-language source of truth for
 * deterministic, byte-identical JSON — the TS analogue of Rust's
 * `canonicalize_json`. Reused by request hashing (#37/#38) AND output-schema
 * delivery + validator messages (#139), so the embedded schema and
 * `{enum}`/`{value}` renderings are byte-identical across Rust/TS/Python/Go
 * regardless of each language's map insertion order. `undefined` members are
 * dropped (JSON has no such concept), matching `serde_json::Value`.
 */
export function canonicalizeJson(value: unknown): string {
  return canonicalize(unknownToValue(value));
}

// ---------------------------------------------------------------------------
// Canonical-value construction
//
// We map the TypeScript `ModelRequest` (which uses optional/undefined for
// Rust's `Option<T>` fields) into a JSON tree that matches what Rust serde
// would produce. This way the generic canonicalizer below stays simple and
// the byte-for-byte agreement with Rust is explicit, not accidental.
// ---------------------------------------------------------------------------

type CanonValue = null | boolean | number | string | CanonValue[] | { [k: string]: CanonValue };

function toCanonicalValue(req: ModelRequest): CanonValue {
  return {
    messages: req.messages.map(messageValue),
    tools: req.tools.map(toolSchemaValue),
    params: paramsValue(req.params),
    stream: req.stream,
  };
}

function messageValue(m: ModelRequest["messages"][number]): CanonValue {
  return {
    role: m.role,
    content: contentValue(m.content),
  };
}

function contentValue(c: ModelRequest["messages"][number]["content"]): CanonValue {
  switch (c.type) {
    case "text":
      return { type: "text", text: c.text };
    case "tool_call":
      return {
        type: "tool_call",
        id: c.id,
        name: c.name,
        input: unknownToValue(c.input),
      };
    case "tool_result":
      return {
        type: "tool_result",
        tool_use_id: c.tool_use_id,
        content: c.content,
        is_error: c.is_error ?? false,
      };
    case "image":
      return { type: "image", media_type: c.media_type, data: c.data };
    default: {
      const _exhaustive: never = c;
      return _exhaustive;
    }
  }
}

function toolSchemaValue(t: ModelRequest["tools"][number]): CanonValue {
  return {
    name: t.name,
    description: t.description,
    input_schema: unknownToValue(t.input_schema),
  };
}

function paramsValue(p: ModelRequest["params"]): CanonValue {
  // Mirror Rust serde: every `Option` field is always emitted (`None` → null).
  const out: { [k: string]: CanonValue } = {
    temperature: p.temperature ?? null,
    max_tokens: p.max_tokens ?? null,
    reasoning_budget: p.reasoning_budget ?? null,
    top_p: p.top_p ?? null,
    stop_sequences: p.stop_sequences,
  };
  // `structured_tool_calls` mirrors Rust's `skip_serializing_if = "Not::not"`:
  // emitted ONLY when true, so the request hash stays byte-identical to the
  // other languages when the flag is off/absent.
  if (p.structured_tool_calls === true) {
    out.structured_tool_calls = true;
  }
  // `output_schema` mirrors Rust's `skip_serializing_if = "Option::is_none"`:
  // emitted ONLY when present (#139), so a request that carries an output schema
  // hashes identically across languages, while the common no-schema case stays
  // byte-for-byte unchanged.
  if (p.output_schema !== undefined) {
    out.output_schema = unknownToValue(p.output_schema);
  }
  return out;
}

/**
 * Convert a free-form `unknown` payload (e.g. `ToolCall.input`,
 * `ToolSchema.input_schema`) into a `CanonValue`. We deliberately accept the
 * same subset Rust's `serde_json::Value` accepts: null, bool, number, string,
 * array, object. `undefined` is normalised to `null` so that callers that
 * hand-craft requests in TS don't introduce divergence from the other langs.
 */
function unknownToValue(v: unknown): CanonValue {
  if (v === null || v === undefined) return null;
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v;
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.map(unknownToValue);
  if (typeof v === "object") {
    const out: { [k: string]: CanonValue } = {};
    for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
      if (val === undefined) continue; // serde_json::Value skips undefined; JSON has no such concept.
      out[k] = unknownToValue(val);
    }
    return out;
  }
  // Anything else (functions, bigint, symbol) is a developer bug; surface it.
  throw new TypeError(`requestHash: unsupported value of type ${typeof v}`);
}

// ---------------------------------------------------------------------------
// Generic canonicalizer
// ---------------------------------------------------------------------------

function canonicalize(v: CanonValue): string {
  if (v === null) return "null";
  if (v === true) return "true";
  if (v === false) return "false";
  if (typeof v === "number") {
    // Matches serde_json: integers render with no fractional part, floats use
    // shortest-round-trip. We don't currently hash float-bearing requests
    // (`fixtures/model_hashing/cases.json` deliberately avoids them); plain
    // `Number.toString()` is sufficient for integer payloads.
    return numberToCanonical(v);
  }
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(canonicalize).join(",") + "]";
  // Object: sort keys by ASCII codepoint, which is `Array.prototype.sort()`'s
  // default for strings.
  const keys = Object.keys(v).sort();
  const parts: string[] = [];
  for (const k of keys) {
    parts.push(JSON.stringify(k) + ":" + canonicalize(v[k] as CanonValue));
  }
  return "{" + parts.join(",") + "}";
}

function numberToCanonical(n: number): string {
  if (!Number.isFinite(n)) {
    throw new RangeError(`requestHash: non-finite numbers are not canonicalizable (${n})`);
  }
  // `Number.toString()` matches serde_json's default integer rendering and
  // produces shortest-round-trip output for floats.
  return n.toString();
}
