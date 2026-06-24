/**
 * Unit + end-to-end mocked-HTTP tests for `OpenAIModelInterface`
 * (spore-core issue #40). Mirrors the Rust suite rule-for-rule.
 */

import * as http from "node:http";
import * as net from "node:net";
import type { AddressInfo } from "node:net";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  OpenAIModelInterface,
  ProviderError,
  RateLimited,
  StreamInterrupted,
  Timeout,
  openaiBackoffDelayMs,
  openaiBuildRequest,
  openaiParseResponse,
  openaiParseStopReason,
  openaiSseToEvents,
  type ContentBlock,
  type Message,
  type ModelRequest,
  type StreamEvent,
} from "../src/index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function user(text: string): Message {
  return { role: "user", content: { type: "text", text } };
}
function sys(text: string): Message {
  return { role: "system", content: { type: "text", text } };
}

function req(messages: Message[], extra: Partial<ModelRequest> = {}): ModelRequest {
  return {
    messages,
    tools: extra.tools ?? [],
    params: extra.params ?? { stop_sequences: [] },
    stream: extra.stream ?? false,
  };
}

interface MockCase {
  status: number;
  headers?: Record<string, string>;
  body?: string;
}

class MockServer {
  readonly server: http.Server;
  port = 0;
  requests: { url: string; headers: http.IncomingHttpHeaders; body: string }[] = [];
  private cases: MockCase[] = [];

  constructor() {
    this.server = http.createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", () => {
        const body = Buffer.concat(chunks).toString("utf-8");
        this.requests.push({ url: req.url ?? "", headers: req.headers, body });
        const idx = Math.min(this.requests.length - 1, this.cases.length - 1);
        const c = this.cases[idx] ?? { status: 200 };
        const headers = c.headers ?? {};
        res.writeHead(c.status, headers);
        res.end(c.body ?? "");
      });
    });
  }

  setCases(cases: MockCase[]): this {
    this.cases = cases;
    return this;
  }

  start(): Promise<void> {
    return new Promise((resolve) => {
      this.server.listen(0, "127.0.0.1", () => {
        this.port = (this.server.address() as AddressInfo).port;
        resolve();
      });
    });
  }

  stop(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server.close((e) => (e ? reject(e) : resolve()));
    });
  }

  baseUrl(): string {
    return `http://127.0.0.1:${this.port}`;
  }
}

function recordingSleep(): { fn: (ms: number) => Promise<void>; delays: number[] } {
  const delays: number[] = [];
  return {
    delays,
    fn: async (ms: number) => {
      delays.push(ms);
    },
  };
}

// ---------------------------------------------------------------------------
// buildRequest
// ---------------------------------------------------------------------------

describe("buildRequest", () => {
  it("keeps system messages in the messages array (not extracted)", () => {
    const body = openaiBuildRequest("gpt-4o", req([sys("be helpful"), user("hi")]), false);
    expect(body.messages).toHaveLength(2);
    expect(body.messages[0]!.role).toBe("system");
    expect(body.messages[1]!.role).toBe("user");
  });

  it("uses max_tokens for chat models", () => {
    const r = req([user("hi")]);
    r.params.max_tokens = 256;
    const body = openaiBuildRequest("gpt-4o", r, false);
    expect(body.max_tokens).toBe(256);
    expect(body.max_completion_tokens).toBeUndefined();
  });

  it("o-series uses max_completion_tokens and omits temperature", () => {
    const r = req([user("hi")]);
    r.params.max_tokens = 512;
    r.params.temperature = 0.7;
    const body = openaiBuildRequest("o3", r, false);
    expect(body.max_tokens).toBeUndefined();
    expect(body.max_completion_tokens).toBe(512);
    expect(body.temperature).toBeUndefined();
  });

  it("isReasoningModel identifies o-series correctly", () => {
    expect(OpenAIModelInterface.isReasoningModel("o4-mini")).toBe(true);
    expect(OpenAIModelInterface.isReasoningModel("o3")).toBe(true);
    expect(OpenAIModelInterface.isReasoningModel("o1-pro")).toBe(true);
    expect(OpenAIModelInterface.isReasoningModel("gpt-4o")).toBe(false);
  });

  it("maps an assistant tool_call to a tool_calls array with JSON-encoded arguments string", () => {
    const r = req([
      {
        role: "assistant",
        content: { type: "tool_call", id: "call-1", name: "fetch", input: { url: "x" } },
      },
    ]);
    const body = openaiBuildRequest("gpt-4o", r, false);
    const msg = body.messages[0]!;
    expect(msg.role).toBe("assistant");
    expect(msg.tool_calls).toBeDefined();
    expect(msg.tool_calls![0]!.id).toBe("call-1");
    expect(msg.tool_calls![0]!.type).toBe("function");
    // arguments must be a JSON-encoded STRING, not a nested object.
    expect(typeof msg.tool_calls![0]!.function.arguments).toBe("string");
    expect(JSON.parse(msg.tool_calls![0]!.function.arguments)).toEqual({ url: "x" });
  });

  it("maps a tool-role tool_result to role=tool with tool_call_id", () => {
    const r = req([
      {
        role: "tool",
        content: { type: "tool_result", tool_use_id: "call-1", content: "ok", is_error: false },
      },
    ]);
    const body = openaiBuildRequest("gpt-4o", r, false);
    expect(body.messages[0]!.role).toBe("tool");
    expect(body.messages[0]!.tool_call_id).toBe("call-1");
    expect(body.messages[0]!.content).toBe("ok");
  });

  it("streaming sets stream_options.include_usage", () => {
    const body = openaiBuildRequest("gpt-4o", req([user("hi")]), true);
    expect(body.stream).toBe(true);
    expect(body.stream_options?.include_usage).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// parseResponse
// ---------------------------------------------------------------------------

describe("parseResponse", () => {
  it("extracts text content and usage", () => {
    const r = openaiParseResponse({
      choices: [{ message: { content: "hi there" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 4, completion_tokens: 2 },
    });
    expect(r.content).toEqual([{ type: "text", text: "hi there" }]);
    expect(r.usage.input_tokens).toBe(4);
    expect(r.usage.output_tokens).toBe(2);
    expect(r.stop_reason).toBe("end_turn");
  });

  it("extracts tool_calls with JSON-parsed arguments", () => {
    const r = openaiParseResponse({
      choices: [
        {
          message: {
            tool_calls: [{ id: "c1", function: { name: "search", arguments: '{"q":"rust"}' } }],
          },
          finish_reason: "tool_calls",
        },
      ],
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
    expect(r.stop_reason).toBe("tool_use");
    const b = r.content[0] as Extract<ContentBlock, { type: "tool_use" }>;
    expect(b.type).toBe("tool_use");
    expect(b.id).toBe("c1");
    expect(b.name).toBe("search");
    expect(b.input).toEqual({ q: "rust" });
  });

  it("maps reasoning text to a thinking block before content text", () => {
    const r = openaiParseResponse({
      choices: [
        {
          message: { reasoning: "let me think...", content: "the answer is 4" },
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
    expect(r.content[0]?.type).toBe("thinking");
    expect(r.content[1]?.type).toBe("text");
  });

  it("reads cache_read from prompt_tokens_details and leaves cache_write null", () => {
    const r = openaiParseResponse({
      choices: [{ message: { content: "x" }, finish_reason: "stop" }],
      usage: {
        prompt_tokens: 100,
        completion_tokens: 2,
        prompt_tokens_details: { cached_tokens: 50 },
      },
    });
    expect(r.usage.cache_read_tokens).toBe(50);
    expect(r.usage.cache_write_tokens).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// stop reason
// ---------------------------------------------------------------------------

describe("parseStopReason", () => {
  it("maps every documented value", () => {
    expect(openaiParseStopReason("stop")).toBe("end_turn");
    expect(openaiParseStopReason("tool_calls")).toBe("tool_use");
    expect(openaiParseStopReason("function_call")).toBe("tool_use");
    expect(openaiParseStopReason("length")).toBe("max_tokens");
    expect(openaiParseStopReason(null)).toBe("end_turn");
    expect(openaiParseStopReason("???")).toBe("end_turn");
  });
});

// ---------------------------------------------------------------------------
// backoff
// ---------------------------------------------------------------------------

describe("backoffDelayMs", () => {
  it("grows then caps at 30s", () => {
    expect(openaiBackoffDelayMs(0)).toBe(500);
    expect(openaiBackoffDelayMs(1)).toBe(1000);
    expect(openaiBackoffDelayMs(2)).toBe(2000);
    expect(openaiBackoffDelayMs(20)).toBeLessThanOrEqual(30_000);
  });
});

// ---------------------------------------------------------------------------
// contextWindow / provider / toJSON
// ---------------------------------------------------------------------------

describe("contextWindow / provider()", () => {
  it("maps known model ids", () => {
    expect(OpenAIModelInterface.contextWindow("gpt-4o")).toBe(128_000);
    expect(OpenAIModelInterface.contextWindow("gpt-4o-mini")).toBe(128_000);
    expect(OpenAIModelInterface.contextWindow("o3")).toBe(200_000);
    expect(OpenAIModelInterface.contextWindow("o4-mini")).toBe(200_000);
    expect(OpenAIModelInterface.contextWindow("claude-x")).toBe(0);
  });

  it("provider() identity matches", () => {
    const c = new OpenAIModelInterface("k", "gpt-4o");
    const p = c.provider();
    expect(p.name).toBe("openai");
    expect(p.model_id).toBe("gpt-4o");
    expect(p.context_window).toBe(128_000);
  });

  it("toJSON redacts the API key", () => {
    const c = new OpenAIModelInterface("super-secret-key", "gpt-4o");
    const j = c.toJSON();
    expect(j.api_key).toBe("<redacted>");
    expect(JSON.stringify(j)).not.toContain("super-secret-key");
  });
});

// ---------------------------------------------------------------------------
// fromEnv
// ---------------------------------------------------------------------------

describe("fromEnv", () => {
  it("throws ProviderError when the variable is unset", () => {
    delete process.env.__SPORE_TEST_OPENAI_KEY_UNSET__;
    expect(() => OpenAIModelInterface.fromEnv("__SPORE_TEST_OPENAI_KEY_UNSET__", "gpt-4o")).toThrow(
      ProviderError,
    );
  });

  it("throws ProviderError when the variable is empty", () => {
    process.env.__SPORE_TEST_OPENAI_KEY_EMPTY__ = "   ";
    expect(() => OpenAIModelInterface.fromEnv("__SPORE_TEST_OPENAI_KEY_EMPTY__", "gpt-4o")).toThrow(
      ProviderError,
    );
    delete process.env.__SPORE_TEST_OPENAI_KEY_EMPTY__;
  });
});

// ---------------------------------------------------------------------------
// sseToEvents — driven from a synthesized ReadableStream
// ---------------------------------------------------------------------------

function streamFromString(s: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(s);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

describe("sseToEvents", () => {
  it("emits text deltas and terminates on [DONE] with accumulated usage", async () => {
    const sse =
      'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n' +
      'data: {"choices":[{"index":0,"delta":{"content":" world"}}]}\n\n' +
      'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5}}\n\n' +
      "data: [DONE]\n\n";
    const events: StreamEvent[] = [];
    for await (const ev of openaiSseToEvents(streamFromString(sse))) events.push(ev);

    expect(events[0]?.type).toBe("message_start");
    const textDeltas = events
      .filter(
        (e): e is Extract<StreamEvent, { type: "content_block_delta" }> =>
          e.type === "content_block_delta",
      )
      .map((e) => e.delta);
    expect(textDeltas).toEqual(["hello", " world"]);
    const last = events[events.length - 1];
    expect(last?.type).toBe("message_stop");
    if (last?.type === "message_stop") {
      expect(last.usage.input_tokens).toBe(3);
      expect(last.usage.output_tokens).toBe(5);
      expect(last.stop_reason).toBe("end_turn");
    }
  });

  it("accumulates tool-call argument deltas across chunks into a parseable JSON", async () => {
    const sse =
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"fetch","arguments":"{\\"u"}}]}}]}\n\n' +
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"rl\\":\\"x\\""}}]}}]}\n\n' +
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}\n\n' +
      'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n' +
      "data: [DONE]\n\n";

    const fragments: string[] = [];
    let start: { id: string; name: string } | null = null;
    let finalStop: StreamEvent["type"] | "" = "";
    let lastStopReason = "";
    for await (const ev of openaiSseToEvents(streamFromString(sse))) {
      if (ev.type === "tool_use_start") start = { id: ev.id, name: ev.name };
      if (ev.type === "tool_use_delta") fragments.push(ev.partial_json);
      if (ev.type === "message_stop") {
        finalStop = ev.type;
        lastStopReason = ev.stop_reason;
      }
    }
    // tool_use_start carries the id + name from the first chunk for the index.
    expect(start).toEqual({ id: "call-1", name: "fetch" });
    expect(finalStop).toBe("message_stop");
    expect(lastStopReason).toBe("tool_use");
    const joined = fragments.join("");
    expect(JSON.parse(joined)).toEqual({ url: "x" });
  });
});

// ---------------------------------------------------------------------------
// End-to-end against a node:http mock server
// ---------------------------------------------------------------------------

describe("call() against a mock server", () => {
  let server: MockServer;
  beforeAll(async () => {
    server = new MockServer();
    await server.start();
  });
  afterAll(async () => {
    await server.stop();
  });

  it("issues a POST with Bearer auth and parses the response", async () => {
    server.requests.length = 0;
    server.setCases([
      {
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          choices: [
            { message: { role: "assistant", content: "hello there" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 5, completion_tokens: 2 },
        }),
      },
    ]);
    const client = new OpenAIModelInterface("test-key", "gpt-4o", {
      baseUrl: server.baseUrl(),
    });
    const r = await client.call(req([user("hi")]));
    expect(r.content).toEqual([{ type: "text", text: "hello there" }]);
    expect(r.usage.input_tokens).toBe(5);
    const req0 = server.requests[0]!;
    expect(req0.url).toBe("/chat/completions");
    expect(req0.headers.authorization).toBe("Bearer test-key");
  });

  it("maps 429 with Retry-After to RateLimited(retry_after)", async () => {
    server.requests.length = 0;
    server.setCases([{ status: 429, headers: { "retry-after": "7" }, body: "" }]);
    const sleep = recordingSleep();
    const client = new OpenAIModelInterface("k", "gpt-4o", {
      baseUrl: server.baseUrl(),
      maxRetries: 0,
      sleep: sleep.fn,
    });
    let err: unknown;
    try {
      await client.call(req([user("hi")]));
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(RateLimited);
    expect((err as RateLimited).retryAfter).toBe(7);
  });

  it("maps 408 to Timeout", async () => {
    server.requests.length = 0;
    server.setCases([{ status: 408, body: "" }]);
    const client = new OpenAIModelInterface("k", "gpt-4o", {
      baseUrl: server.baseUrl(),
      maxRetries: 0,
    });
    await expect(client.call(req([user("hi")]))).rejects.toBeInstanceOf(Timeout);
  });

  it("maps 400 to ProviderError with the OpenAI-supplied message", async () => {
    server.requests.length = 0;
    server.setCases([
      {
        status: 400,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          error: { type: "invalid_request_error", message: "max_tokens must be > 0" },
        }),
      },
    ]);
    const client = new OpenAIModelInterface("k", "gpt-4o", {
      baseUrl: server.baseUrl(),
      maxRetries: 0,
    });
    let err: unknown;
    try {
      await client.call(req([user("hi")]));
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).code).toBe(400);
    expect((err as ProviderError).message).toContain("max_tokens");
  });

  it("retries 429 then succeeds", async () => {
    server.requests.length = 0;
    server.setCases([
      { status: 429, headers: { "retry-after": "0" }, body: "" },
      {
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          choices: [{ message: { content: "after retry" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
      },
    ]);
    const sleep = recordingSleep();
    const client = new OpenAIModelInterface("k", "gpt-4o", {
      baseUrl: server.baseUrl(),
      sleep: sleep.fn,
    });
    const r = await client.call(req([user("hi")]));
    expect(r.content).toEqual([{ type: "text", text: "after retry" }]);
    expect(server.requests.length).toBe(2);
    expect(sleep.delays).toEqual([0]);
  });

  it("callStreaming() returns parsed events end-to-end", async () => {
    server.requests.length = 0;
    const sse =
      'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n' +
      'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5}}\n\n' +
      "data: [DONE]\n\n";
    server.setCases([{ status: 200, headers: { "content-type": "text/event-stream" }, body: sse }]);
    const client = new OpenAIModelInterface("k", "gpt-4o", { baseUrl: server.baseUrl() });
    const events: StreamEvent[] = [];
    for await (const ev of client.callStreaming(req([user("hi")]))) events.push(ev);
    expect(events.map((e) => e.type)).toContain("message_stop");
    const last = events[events.length - 1];
    if (last?.type === "message_stop") {
      expect(last.usage.input_tokens).toBe(3);
      expect(last.usage.output_tokens).toBe(5);
    }
  });

  it("countTokens uses bytes/4 heuristic", async () => {
    const c = new OpenAIModelInterface("k", "gpt-4o");
    const r = req([user("a".repeat(40))]);
    expect(await c.countTokens(r)).toBe(10);
  });
});

// ---------------------------------------------------------------------------
// Mid-stream truncation — SC-3 typed StreamInterrupted
// ---------------------------------------------------------------------------

describe("callStreaming() mid-stream truncation", () => {
  it("surfaces a typed, retryable StreamInterrupted when the body is cut off", async () => {
    // A raw TCP server promises a 200-byte body (Content-Length) but sends only
    // a partial SSE line then drops the connection — so the client's body stream
    // errors mid-read after the 200 headers arrived. SC-3: a connection dropped
    // mid-stream surfaces as the typed, retryable StreamInterrupted variant — a
    // consumer drives its retry off retryable(), not a substring match.
    const server = net.createServer((sock) => {
      sock.once("data", () => {
        // 200 OK so callStreaming returns Ok (headers arrived), then promise 200
        // body bytes but deliver only a partial SSE line and drop the socket —
        // EOF before Content-Length errors the stream.
        sock.write(
          "HTTP/1.1 200 OK\r\n" +
            "content-type: text/event-stream\r\n" +
            "content-length: 200\r\n" +
            "\r\n" +
            "data: partial",
        );
        sock.end();
      });
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const addr = server.address() as AddressInfo;
    const baseUrl = `http://127.0.0.1:${addr.port}`;

    try {
      const client = new OpenAIModelInterface("k", "gpt-4o", { baseUrl });
      // Headers (200) arrive before the body is truncated, so callStreaming
      // itself does not throw — the stream errors mid-drain.
      let caught: unknown;
      try {
        for await (const _ev of client.callStreaming(req([user("hi")]))) {
          /* drain until the truncated body errors the stream */
        }
      } catch (e) {
        caught = e;
      }
      expect(caught).toBeInstanceOf(StreamInterrupted);
      expect((caught as StreamInterrupted).retryable()).toBe(true);
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((e) => (e ? reject(e) : resolve())),
      );
    }
  });
});

// ---------------------------------------------------------------------------
// Live-API tests — skipped by default
// ---------------------------------------------------------------------------

const LIVE = process.env.OPENAI_API_KEY != null && process.env.OPENAI_API_KEY !== "";

describe.skipIf(!LIVE)("OpenAIModelInterface — live API", () => {
  const modelId = process.env.OPENAI_TEST_MODEL ?? "gpt-4o-mini";

  it("call() returns a non-empty usage", async () => {
    const client = OpenAIModelInterface.fromEnv("OPENAI_API_KEY", modelId);
    const r = await client.call(req([user("Reply with the word 'pong'.")]));
    expect(r.usage.input_tokens).toBeGreaterThan(0);
    expect(r.usage.output_tokens).toBeGreaterThan(0);
  });

  it("callStreaming() emits a message_stop event", async () => {
    const client = OpenAIModelInterface.fromEnv("OPENAI_API_KEY", modelId);
    let sawStop = false;
    for await (const ev of client.callStreaming(req([user("Reply with the word 'pong'.")]))) {
      if (ev.type === "message_stop") sawStop = true;
    }
    expect(sawStop).toBe(true);
  });
});
