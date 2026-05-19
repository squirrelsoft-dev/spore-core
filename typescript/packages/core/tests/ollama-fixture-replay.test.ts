/**
 * Fixture replay for the Ollama basic-text exchange recorded in
 * `fixtures/model_responses/model_interface/ollama_basic_text.jsonl`.
 *
 * Round-trips the fixture through {@link ReplayModelInterface} so the
 * TypeScript and Rust implementations stay locked to the same wire shape.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { ReplayModelInterface, type ModelRequest, type ProviderInfo } from "../src/index.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(
  repoRoot,
  "fixtures/model_responses/model_interface/ollama_basic_text.jsonl",
);

const provider: ProviderInfo = {
  name: "ollama",
  model_id: "fixture",
  context_window: 128_000,
};

const emptyRequest: ModelRequest = {
  messages: [],
  tools: [],
  params: { stop_sequences: [] },
  stream: false,
};

describe("ReplayModelInterface — ollama_basic_text.jsonl", () => {
  it("round-trips the recorded text response", async () => {
    const jsonl = readFileSync(fixturePath, "utf-8");
    const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
    const r = await replay.call(emptyRequest);
    expect(r.stop_reason).toBe("end_turn");
    expect(r.usage.input_tokens).toBe(8);
    expect(r.usage.output_tokens).toBe(11);
    expect(r.content).toEqual([{ type: "text", text: "Hello! How can I help you today?" }]);
    expect(r.usage.cache_read_tokens).toBeNull();
    expect(r.usage.cache_write_tokens).toBeNull();
  });
});
