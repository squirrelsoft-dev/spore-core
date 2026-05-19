import { mkdtempSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  MockModelInterface,
  ProviderError,
  RecordedExchangeSchema,
  RecordingModelInterface,
  ReplayModelInterface,
  requestHash,
  type ModelRequest,
  type ModelResponse,
  type ProviderInfo,
  type RecordedExchange,
} from "../src/index.js";

const provider: ProviderInfo = {
  name: "test",
  model_id: "test-1",
  context_window: 1000,
};

function reqText(s: string): ModelRequest {
  return {
    messages: [{ role: "user", content: { type: "text", text: s } }],
    tools: [],
    params: { stop_sequences: [] },
    stream: false,
  };
}

function respText(s: string): ModelResponse {
  return {
    content: [{ type: "text", text: s }],
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      cache_read_tokens: null,
      cache_write_tokens: null,
    },
    stop_reason: "end_turn",
  };
}

let workDir: string;
beforeEach(() => {
  workDir = mkdtempSync(join(tmpdir(), "spore-recording-"));
});
afterEach(() => {
  // Each test uses a fresh temp dir; we leave them to the OS to reap.
});

describe("ReplayModelInterface — mode auto-detection", () => {
  it("falls back to positional when no entries carry a request_hash", async () => {
    const exchanges: RecordedExchange[] = [
      {
        request: reqText("q1"),
        response: respText("r1"),
        provider: "fixture",
      },
    ];
    const r = new ReplayModelInterface(exchanges, provider);
    expect(r.mode()).toBe("positional");
    const got = await r.call(reqText("any"));
    expect(got).toEqual(respText("r1"));
  });

  it("picks hash_matched when every entry has a request_hash", async () => {
    const q1 = reqText("q1");
    const q2 = reqText("q2");
    const exchanges: RecordedExchange[] = [
      {
        request_hash: requestHash(q1),
        request: q1,
        response: respText("r1"),
        provider: "fixture",
      },
      {
        request_hash: requestHash(q2),
        request: q2,
        response: respText("r2"),
        provider: "fixture",
      },
    ];
    const r = new ReplayModelInterface(exchanges, provider);
    expect(r.mode()).toBe("hash_matched");
    // Out-of-order replay still maps each request to the right response.
    expect(await r.call(q2)).toEqual(respText("r2"));
    expect(await r.call(q1)).toEqual(respText("r1"));
    expect(await r.call(q2)).toEqual(respText("r2"));
  });

  it("falls back to positional when the exchange list is empty", () => {
    const r = new ReplayModelInterface([], provider);
    expect(r.mode()).toBe("positional");
  });

  it("hash_matched with no matching fixture surfaces a ProviderError", async () => {
    const q1 = reqText("q1");
    const r = new ReplayModelInterface(
      [
        {
          request_hash: requestHash(q1),
          request: q1,
          response: respText("r1"),
          provider: "fixture",
        },
      ],
      provider,
    );
    expect(r.mode()).toBe("hash_matched");
    await expect(r.call(reqText("unrecorded"))).rejects.toBeInstanceOf(ProviderError);
  });
});

describe("RecordingModelInterface", () => {
  it("`record` mode appends one JSONL line per call with a populated request_hash", async () => {
    const path = join(workDir, "recorded.jsonl");
    const inner = new MockModelInterface(provider)
      .pushResponse(respText("hello back"))
      .pushResponse(respText("hello again"));
    const rec = new RecordingModelInterface(inner, path, "record");
    await rec.call(reqText("hello"));
    await rec.call(reqText("hello2"));

    const raw = readFileSync(path, "utf-8");
    const lines = raw.split("\n").filter((l) => l.length > 0);
    expect(lines).toHaveLength(2);
    for (const line of lines) {
      const entry = RecordedExchangeSchema.parse(JSON.parse(line));
      expect(entry.request_hash).toBeTypeOf("string");
      expect(entry.request_hash).toHaveLength(16);
      expect(entry.provider).toBe("test");
      expect(entry.model_id).toBe("test-1");
      expect(entry.duration_ms).toBeTypeOf("number");
    }
  });

  it("`record_if_new` skips writing when the file already exists", async () => {
    const path = join(workDir, "existing.jsonl");
    writeFileSync(path, "preexisting line\n", "utf8");
    const inner = new MockModelInterface(provider).pushResponse(respText("ok"));
    const rec = new RecordingModelInterface(inner, path, "record_if_new");
    await rec.call(reqText("q"));
    expect(readFileSync(path, "utf-8")).toBe("preexisting line\n");
  });

  it("`record_if_new` writes when the file is missing", async () => {
    const path = join(workDir, "new.jsonl");
    const inner = new MockModelInterface(provider).pushResponse(respText("ok"));
    const rec = new RecordingModelInterface(inner, path, "record_if_new");
    await rec.call(reqText("q"));
    const lines = readFileSync(path, "utf-8")
      .split("\n")
      .filter((l) => l.length > 0);
    expect(lines).toHaveLength(1);
  });

  it("`passthrough` calls the inner model but writes nothing", async () => {
    const path = join(workDir, "nope.jsonl");
    const inner = new MockModelInterface(provider).pushResponse(respText("ok"));
    const rec = new RecordingModelInterface(inner, path, "passthrough");
    const r = await rec.call(reqText("q"));
    expect(r).toEqual(respText("ok"));
    expect(existsSync(path)).toBe(false);
  });

  it("provider/countTokens/callStreaming delegate to the inner model", async () => {
    const path = join(workDir, "delegate.jsonl");
    const inner = new MockModelInterface(provider).pushResponse(respText("hi")).pushTokenCount(7);
    const rec = new RecordingModelInterface(inner, path, "passthrough");
    expect(rec.provider()).toEqual(provider);
    expect(await rec.countTokens(reqText("any"))).toBe(7);
    const events = [];
    for await (const ev of rec.callStreaming(reqText("any"))) events.push(ev);
    expect(events[0]?.type).toBe("message_start");
    expect(events[events.length - 1]?.type).toBe("message_stop");
  });

  it("record-then-replay round-trip works in hash_matched mode", async () => {
    const path = join(workDir, "roundtrip.jsonl");
    const inner = new MockModelInterface(provider)
      .pushResponse(respText("answer1"))
      .pushResponse(respText("answer2"));
    const rec = new RecordingModelInterface(inner, path, "record");
    const q1 = reqText("question 1");
    const q2 = reqText("question 2");
    await rec.call(q1);
    await rec.call(q2);

    const jsonl = readFileSync(path, "utf-8");
    const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
    expect(replay.mode()).toBe("hash_matched");
    // Replay out of order to confirm hash matching works end-to-end.
    expect(await replay.call(q2)).toEqual(respText("answer2"));
    expect(await replay.call(q1)).toEqual(respText("answer1"));
  });
});
