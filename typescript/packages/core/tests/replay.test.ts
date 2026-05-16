import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  ProviderError,
  ReplayModelInterface,
  type ModelRequest,
  type ProviderInfo,
  type StreamEvent,
} from "../src/index.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
// tests/ -> packages/core/ -> packages/ -> typescript/ -> repo root.
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(
  repoRoot,
  "fixtures/model_responses/model_interface/basic_text.jsonl",
);

const provider: ProviderInfo = {
  name: "anthropic",
  model_id: "fixture",
  context_window: 200_000,
};

const emptyRequest: ModelRequest = {
  messages: [],
  tools: [],
  params: { stop_sequences: [] },
  stream: false,
};

describe("ReplayModelInterface — fixture replay", () => {
  it("replays basic_text.jsonl in order, matching the Rust integration test", async () => {
    const jsonl = readFileSync(fixturePath, "utf-8");
    const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
    expect(replay.remaining()).toBe(3);

    const r1 = await replay.call(emptyRequest);
    expect(r1.stop_reason).toBe("end_turn");
    expect(r1.usage.input_tokens).toBe(8);
    expect(r1.usage.output_tokens).toBe(11);
    expect(r1.content).toEqual([
      { type: "text", text: "Hello! How can I help you today?" },
    ]);

    const r2 = await replay.call(emptyRequest);
    expect(r2.usage.input_tokens).toBe(10);
    expect(r2.usage.output_tokens).toBe(1);

    const r3 = await replay.call(emptyRequest);
    expect(r3.stop_reason).toBe("tool_use");
    const block = r3.content[0];
    expect(block?.type).toBe("tool_use");
    if (block?.type === "tool_use") {
      expect(block.name).toBe("echo");
      expect((block.input as { text: string }).text).toBe("hi");
    }
  });

  it("exhaustion surfaces as a typed ProviderError, never an unhandled throw", async () => {
    const replay = ReplayModelInterface.fromJsonl("", provider);
    await expect(replay.call(emptyRequest)).rejects.toBeInstanceOf(ProviderError);
    try {
      await replay.call(emptyRequest);
    } catch (e) {
      expect(e).toBeInstanceOf(ProviderError);
      expect((e as ProviderError).code).toBe(0);
    }
  });

  it("streaming yields message_start, per-block delta + stop, then message_stop with usage", async () => {
    const jsonl = readFileSync(fixturePath, "utf-8");
    const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
    const events: StreamEvent[] = [];
    for await (const ev of replay.callStreaming(emptyRequest)) events.push(ev);
    expect(events[0]).toEqual({ type: "message_start" });
    const last = events[events.length - 1];
    expect(last?.type).toBe("message_stop");
    if (last?.type === "message_stop") {
      expect(last.usage.input_tokens).toBe(8);
      expect(last.stop_reason).toBe("end_turn");
    }
    // Per-block delta + stop pair.
    expect(events.some((e) => e.type === "content_block_delta")).toBe(true);
    expect(events.some((e) => e.type === "content_block_stop")).toBe(true);
  });

  it("countTokens is deterministic ~chars/4", async () => {
    const replay = new ReplayModelInterface([], provider);
    const req: ModelRequest = {
      messages: [
        {
          role: "user",
          content: { type: "text", text: "a".repeat(40) },
        },
      ],
      tools: [],
      params: { stop_sequences: [] },
      stream: false,
    };
    expect(await replay.countTokens(req)).toBe(10);
  });
});
