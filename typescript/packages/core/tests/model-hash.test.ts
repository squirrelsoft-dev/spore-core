import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { ModelRequestSchema, requestHash, type ModelRequest } from "../src/index.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(repoRoot, "fixtures/model_hashing/cases.json");

function req(text: string): ModelRequest {
  return ModelRequestSchema.parse({
    messages: [{ role: "user", content: { type: "text", text } }],
    tools: [],
    params: {},
    stream: false,
  });
}

describe("requestHash — basic properties", () => {
  it("is stable for equivalent requests", () => {
    expect(requestHash(req("hello world"))).toBe(requestHash(req("hello world")));
  });

  it("changes when the request changes", () => {
    expect(requestHash(req("hello"))).not.toBe(requestHash(req("hello!")));
  });

  it("returns 16 lowercase hex characters", () => {
    const h = requestHash(req("x"));
    expect(h).toHaveLength(16);
    expect(h).toMatch(/^[0-9a-f]{16}$/);
  });
});

describe("requestHash — cross-language fixture", () => {
  const raw = readFileSync(fixturePath, "utf-8");
  const suite = JSON.parse(raw) as {
    cases: { name: string; request: unknown; expected_hash: string }[];
  };

  for (const c of suite.cases) {
    it(`matches the expected hash for case '${c.name}'`, () => {
      const parsed = ModelRequestSchema.parse(c.request);
      const got = requestHash(parsed);
      expect(got).toBe(c.expected_hash);
    });
  }
});
