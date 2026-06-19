/**
 * Output-schema validator unit tests (spore-core issue #139).
 *
 * Mirrors the Rust `output_schema.rs` `#[cfg(test)]` module — same validator
 * subset (`type`/`required`/`properties`/`enum`), same FROZEN literals, same
 * first-match-wins evaluation order, same determinism rules
 * (lexicographic property iteration, array-order `required`, sorted `{enum}`
 * rendering, the `42.0` vs `42.5` SEMANTIC integer check).
 *
 * The error strings + feedback message are HASH-LOAD-BEARING — the four
 * language ports must reproduce these exact bytes.
 */

import { describe, expect, it } from "vitest";

import { ollamaBuildRequest, type ModelRequest } from "../src/index.js";
import {
  ERR_NOT_JSON,
  errMissingRequired,
  errPropertyEnum,
  errPropertyType,
  errRootType,
  feedbackMessage,
  validateOutput,
} from "../src/output-schema/index.js";

// ── Feedback + error literals (byte-exact, HASH-LOAD-BEARING) ─────────────────

describe("output-schema frozen literals (#139)", () => {
  it("feedback message has exact bytes", () => {
    expect(feedbackMessage("X.")).toBe(
      "Your previous response did not match the required output schema. X. " +
        "Reply with only a JSON value that satisfies the schema.",
    );
  });

  it("error literals have exact bytes", () => {
    expect(ERR_NOT_JSON).toBe("The response was not valid JSON.");
    expect(errRootType("object", "array")).toBe('Expected type "object" but found "array".');
    expect(errMissingRequired("name")).toBe('Missing required property "name".');
    expect(errPropertyType("age", "integer", "string")).toBe(
      'Property "age" should be type "integer" but found "string".',
    );
    expect(errPropertyEnum("status", '["error","ok"]', '"maybe"')).toBe(
      'Property "status" must be one of ["error","ok"] but found "maybe".',
    );
  });
});

// ── Step 1: parse ────────────────────────────────────────────────────────────

describe("output-schema validator (#139) — step 1 parse", () => {
  it("non-JSON → ERR_NOT_JSON", () => {
    expect(validateOutput("not json at all", { type: "object" })).toBe(ERR_NOT_JSON);
  });
});

// ── Step 2: root type ────────────────────────────────────────────────────────

describe("output-schema validator (#139) — step 2 root type", () => {
  it("root type mismatch", () => {
    expect(validateOutput("[1,2,3]", { type: "object" })).toBe(errRootType("object", "array"));
  });

  it("root type match", () => {
    expect(validateOutput("[1,2,3]", { type: "array" })).toBeNull();
  });
});

// ── Step 3: required (array order) ───────────────────────────────────────────

describe("output-schema validator (#139) — step 3 required", () => {
  it("missing required reported in ARRAY order, not lexicographic", () => {
    // `required` order is [b, a]; both absent → the FIRST in array order (`b`)
    // is reported, NOT the lexicographically-first (`a`).
    expect(validateOutput("{}", { type: "object", required: ["b", "a"] })).toBe(
      errMissingRequired("b"),
    );
  });

  it("required present passes", () => {
    expect(validateOutput('{"a":1}', { type: "object", required: ["a"] })).toBeNull();
  });
});

// ── Step 4: present-property type (sorted key order) ─────────────────────────

describe("output-schema validator (#139) — step 4 property type", () => {
  it("property type mismatch reported in SORTED key order", () => {
    // Both `age` and `zip` are wrong (string where number expected). Sorted key
    // order ⇒ `age` reported first (NOT insertion order `zip`).
    const schema = {
      type: "object",
      properties: { zip: { type: "number" }, age: { type: "number" } },
    };
    expect(validateOutput('{"age":"x","zip":"y"}', schema)).toBe(
      errPropertyType("age", "number", "string"),
    );
  });

  it("integer accepts a whole number 42.0 (semantic check)", () => {
    const schema = { type: "object", properties: { n: { type: "integer" } } };
    expect(validateOutput('{"n":42.0}', schema)).toBeNull();
    expect(validateOutput('{"n":42}', schema)).toBeNull();
  });

  it("integer rejects a fractional 42.5 (semantic check)", () => {
    const schema = { type: "object", properties: { n: { type: "integer" } } };
    expect(validateOutput('{"n":42.5}', schema)).toBe(errPropertyType("n", "integer", "number"));
  });

  it("number accepts a fractional value", () => {
    const schema = { type: "object", properties: { n: { type: "number" } } };
    expect(validateOutput('{"n":42.5}', schema)).toBeNull();
  });
});

// ── Step 5: present-property enum (sorted enum rendering) ─────────────────────

describe("output-schema validator (#139) — step 5 enum", () => {
  it("enum violation renders the enum SORTED regardless of author order", () => {
    // Author order is ["ok","error"]; the message renders SORTED (["error","ok"])
    // for determinism. Membership itself is order-free.
    const schema = {
      type: "object",
      properties: { status: { type: "string", enum: ["ok", "error"] } },
    };
    expect(validateOutput('{"status":"maybe"}', schema)).toBe(
      errPropertyEnum("status", '["error","ok"]', '"maybe"'),
    );
  });

  it("enum member passes regardless of author order", () => {
    const schema = {
      type: "object",
      properties: { status: { type: "string", enum: ["ok", "error"] } },
    };
    expect(validateOutput('{"status":"ok"}', schema)).toBeNull();
    expect(validateOutput('{"status":"error"}', schema)).toBeNull();
  });
});

// ── Step 6: valid ────────────────────────────────────────────────────────────

describe("output-schema validator (#139) — step 6 valid", () => {
  it("a full object passes", () => {
    const schema = {
      type: "object",
      required: ["status", "count"],
      properties: {
        status: { type: "string", enum: ["ok", "error"] },
        count: { type: "integer" },
      },
    };
    expect(validateOutput('{"status":"ok","count":3}', schema)).toBeNull();
  });
});

// ── Evaluation order: earlier rule wins ──────────────────────────────────────

describe("output-schema validator (#139) — first-match-wins order", () => {
  it("root type beats required", () => {
    expect(validateOutput('"a string"', { type: "object", required: ["a"] })).toBe(
      errRootType("object", "string"),
    );
  });

  it("required beats property type", () => {
    const schema = {
      type: "object",
      required: ["a"],
      properties: { b: { type: "number" } },
    };
    expect(validateOutput('{"b":"x"}', schema)).toBe(errMissingRequired("a"));
  });

  it("property type beats enum", () => {
    const schema = {
      type: "object",
      properties: { s: { type: "string", enum: ["ok"] } },
    };
    expect(validateOutput('{"s":123}', schema)).toBe(errPropertyType("s", "string", "number"));
  });
});

// ── AC1: Ollama `format` channel population + non-Ollama no-op ───────────────

function req(messages: ModelRequest["messages"], outputSchema?: unknown): ModelRequest {
  return {
    messages,
    tools: [],
    params: { stop_sequences: [], output_schema: outputSchema },
    stream: false,
  };
}

describe("output-schema delivery (#139) — AC1 Ollama format channel", () => {
  it("params.output_schema routes into the Ollama `format` channel verbatim (no tools)", () => {
    const schema = {
      type: "object",
      properties: { status: { type: "string", enum: ["ok", "error"] } },
      required: ["status"],
    };
    const body = ollamaBuildRequest(
      "llama3.2",
      null,
      null,
      req([{ role: "user", content: { type: "text", text: "answer" } }], schema),
      false,
    );
    expect(body.format).toEqual(schema);
  });

  it("absent output_schema leaves `format` unset — byte-identical to pre-#139", () => {
    const body = ollamaBuildRequest(
      "llama3.2",
      null,
      null,
      req([{ role: "user", content: { type: "text", text: "hi" } }]),
      false,
    );
    expect(body.format).toBeUndefined();
  });
});
