/**
 * Unit + end-to-end mocked-HTTP tests for `AnthropicModelInterface`
 * (spore-core issue #39). Mirrors the Rust suite rule-for-rule.
 */

import * as http from "node:http";
import type { AddressInfo } from "node:net";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  AnthropicModelInterface,
  ProviderError,
  RateLimited,
  Timeout,
  anthropicBackoffDelayMs,
  anthropicBuildRequest,
  anthropicParseResponse,
  anthropicParseSseEvent,
  anthropicParseStopReason,
  anthropicSseToEvents,
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

/**
 * Tiny configurable mock HTTP server.
 *
 * Each "case" can either be a complete response definition or a function
 * receiving the parsed request body. Requests are served in arrival order;
 * once the list is exhausted, the last case repeats. Useful for asserting
 * retry behaviour and per-call payload shape.
 */
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
        this.requests.push({
          url: req.url ?? "",
          headers: req.headers,
          body,
        });
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

// `sleep` stub that records delays and resolves immediately.
function recordingSleep(): {
  fn: (ms: number) => Promise<void>;
  delays: number[];
} {
  const delays: number[] = [];
  return {
    delays,
    fn: async (ms: number) => {
      delays.push(ms);
    },
  };
}

// ---------------------------------------------------------------------------
// build_request
// ---------------------------------------------------------------------------

describe("buildRequest", () => {
  it("extracts a single system message into top-level system field", () => {
    const body = anthropicBuildRequest(
      "claude-sonnet-4-6",
      req([sys("be helpful"), user("hi")]),
      false,
    );
    expect(body.system).toBe("be helpful");
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0]!.role).toBe("user");
  });

  it("joins multiple system messages with \\n\\n", () => {
    const body = anthropicBuildRequest(
      "claude-sonnet-4-6",
      req([sys("first"), sys("second"), user("hi")]),
      false,
    );
    expect(body.system).toBe("first\n\nsecond");
  });

  it("defaults max_tokens to 4096 when unset", () => {
    const body = anthropicBuildRequest("claude-sonnet-4-6", req([user("hi")]), false);
    expect(body.max_tokens).toBe(4096);
  });

  it("respects an explicit max_tokens", () => {
    const r = req([user("hi")]);
    r.params.max_tokens = 256;
    const body = anthropicBuildRequest("claude-sonnet-4-6", r, false);
    expect(body.max_tokens).toBe(256);
  });

  it("maps an assistant tool_call to a tool_use content block", () => {
    const r = req([
      {
        role: "assistant",
        content: { type: "tool_call", id: "call-1", name: "fetch", input: { url: "x" } },
      },
    ]);
    const body = anthropicBuildRequest("claude-sonnet-4-6", r, false);
    const wire = JSON.stringify(body);
    expect(wire).toContain('"type":"tool_use"');
    expect(wire).toContain('"id":"call-1"');
  });

  it("maps a tool-role tool_result to a user-role tool_result block", () => {
    const r = req([
      {
        role: "tool",
        content: { type: "tool_result", tool_use_id: "call-1", content: "ok", is_error: false },
      },
    ]);
    const body = anthropicBuildRequest("claude-sonnet-4-6", r, false);
    expect(body.messages[0]!.role).toBe("user");
    const wire = JSON.stringify(body.messages[0]!.content);
    expect(wire).toContain('"type":"tool_result"');
  });
});

// ---------------------------------------------------------------------------
// parse_response
// ---------------------------------------------------------------------------

describe("parseResponse", () => {
  it("extracts text content and usage", () => {
    const r = anthropicParseResponse({
      content: [{ type: "text", text: "hi there" }],
      stop_reason: "end_turn",
      usage: { input_tokens: 4, output_tokens: 2 },
    });
    expect(r.content).toEqual([{ type: "text", text: "hi there" }]);
    expect(r.usage.input_tokens).toBe(4);
    expect(r.usage.output_tokens).toBe(2);
    expect(r.stop_reason).toBe("end_turn");
  });

  it("extracts tool_use blocks", () => {
    const r = anthropicParseResponse({
      content: [{ type: "tool_use", id: "c1", name: "search", input: { q: "rust" } }],
      stop_reason: "tool_use",
      usage: { input_tokens: 1, output_tokens: 1 },
    });
    expect(r.stop_reason).toBe("tool_use");
    const b = r.content[0] as Extract<ContentBlock, { type: "tool_use" }>;
    expect(b.type).toBe("tool_use");
    expect(b.id).toBe("c1");
    expect(b.name).toBe("search");
  });

  it("extracts thinking blocks", () => {
    const r = anthropicParseResponse({
      content: [
        { type: "thinking", thinking: "let me reason..." },
        { type: "text", text: "answer" },
      ],
      stop_reason: "end_turn",
      usage: { input_tokens: 1, output_tokens: 1 },
    });
    expect(r.content[0]?.type).toBe("thinking");
    expect(r.content[1]?.type).toBe("text");
  });

  it("extracts cache usage fields", () => {
    const r = anthropicParseResponse({
      content: [{ type: "text", text: "x" }],
      stop_reason: "end_turn",
      usage: {
        input_tokens: 100,
        output_tokens: 2,
        cache_read_input_tokens: 50,
        cache_creation_input_tokens: 30,
      },
    });
    expect(r.usage.cache_read_tokens).toBe(50);
    expect(r.usage.cache_write_tokens).toBe(30);
  });
});

// ---------------------------------------------------------------------------
// stop reason
// ---------------------------------------------------------------------------

describe("parseStopReason", () => {
  it("maps every documented value", () => {
    expect(anthropicParseStopReason("end_turn")).toBe("end_turn");
    expect(anthropicParseStopReason("tool_use")).toBe("tool_use");
    expect(anthropicParseStopReason("max_tokens")).toBe("max_tokens");
    expect(anthropicParseStopReason("stop_sequence")).toBe("stop_sequence");
    expect(anthropicParseStopReason(null)).toBe("end_turn");
    expect(anthropicParseStopReason("???")).toBe("end_turn");
  });
});

// ---------------------------------------------------------------------------
// backoff
// ---------------------------------------------------------------------------

describe("backoffDelayMs", () => {
  it("grows then caps at 30s", () => {
    expect(anthropicBackoffDelayMs(0)).toBe(500);
    expect(anthropicBackoffDelayMs(1)).toBe(1000);
    expect(anthropicBackoffDelayMs(2)).toBe(2000);
    expect(anthropicBackoffDelayMs(3)).toBe(4000);
    expect(anthropicBackoffDelayMs(20)).toBeLessThanOrEqual(30_000);
  });
});

// ---------------------------------------------------------------------------
// context_window / provider
// ---------------------------------------------------------------------------

describe("contextWindow / provider()", () => {
  it("reports 200k for known and any claude-* id", () => {
    expect(AnthropicModelInterface.contextWindow("claude-sonnet-4-6")).toBe(200_000);
    expect(AnthropicModelInterface.contextWindow("claude-opus-4-7")).toBe(200_000);
    expect(AnthropicModelInterface.contextWindow("claude-imaginary-9")).toBe(200_000);
    expect(AnthropicModelInterface.contextWindow("gpt-4o")).toBe(0);
  });

  it("provider() identity matches", () => {
    const c = new AnthropicModelInterface("k", "claude-sonnet-4-6");
    const p = c.provider();
    expect(p.name).toBe("anthropic");
    expect(p.model_id).toBe("claude-sonnet-4-6");
    expect(p.context_window).toBe(200_000);
  });

  it("toJSON redacts the API key", () => {
    const c = new AnthropicModelInterface("super-secret-key", "claude-sonnet-4-6");
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
    delete process.env.__SPORE_TEST_ANTHROPIC_KEY_UNSET__;
    expect(() =>
      AnthropicModelInterface.fromEnv("__SPORE_TEST_ANTHROPIC_KEY_UNSET__", "claude-sonnet-4-6"),
    ).toThrow(ProviderError);
  });

  it("throws ProviderError when the variable is empty", () => {
    process.env.__SPORE_TEST_ANTHROPIC_KEY_EMPTY__ = "   ";
    expect(() =>
      AnthropicModelInterface.fromEnv("__SPORE_TEST_ANTHROPIC_KEY_EMPTY__", "claude-sonnet-4-6"),
    ).toThrow(ProviderError);
    delete process.env.__SPORE_TEST_ANTHROPIC_KEY_EMPTY__;
  });
});

// ---------------------------------------------------------------------------
// SSE parser
// ---------------------------------------------------------------------------

describe("parseSseEvent", () => {
  it("parses a basic event/data block", () => {
    const out = anthropicParseSseEvent('event: message_start\ndata: {"type":"message_start"}');
    expect(out).toEqual({ event: "message_start", data: '{"type":"message_start"}' });
  });

  it("joins multi-line data with \\n", () => {
    const out = anthropicParseSseEvent(
      'event: message_delta\ndata: {"first":1}\ndata: continuation',
    );
    expect(out).toEqual({ event: "message_delta", data: '{"first":1}\ncontinuation' });
  });

  it("returns null when no event line present", () => {
    expect(anthropicParseSseEvent("data: {}")).toBeNull();
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
  it("emits text deltas, content_block_stop, and a final MessageStop with accumulated usage", async () => {
    const sse =
      "event: message_start\n" +
      'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}\n' +
      "\n" +
      "event: content_block_delta\n" +
      'data: {"index":0,"delta":{"type":"text_delta","text":"hello"}}\n' +
      "\n" +
      "event: content_block_delta\n" +
      'data: {"index":0,"delta":{"type":"text_delta","text":" world"}}\n' +
      "\n" +
      "event: content_block_stop\n" +
      'data: {"index":0}\n' +
      "\n" +
      "event: message_delta\n" +
      'data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n' +
      "\n" +
      "event: message_stop\n" +
      'data: {"type":"message_stop"}\n' +
      "\n";

    const events: StreamEvent[] = [];
    for await (const ev of anthropicSseToEvents(streamFromString(sse))) events.push(ev);

    const kinds = events.map((e) => e.type);
    expect(kinds).toEqual([
      "message_start",
      "content_block_delta",
      "content_block_delta",
      "content_block_stop",
      "message_stop",
    ]);
    const last = events[events.length - 1];
    expect(last?.type).toBe("message_stop");
    if (last?.type === "message_stop") {
      expect(last.usage.input_tokens).toBe(3);
      expect(last.usage.output_tokens).toBe(5);
      expect(last.stop_reason).toBe("end_turn");
    }
  });

  it("emits thinking_delta and tool_use_delta from their respective sub-types", async () => {
    const sse =
      "event: content_block_delta\n" +
      'data: {"index":1,"delta":{"type":"thinking_delta","thinking":"hmm"}}\n' +
      "\n" +
      "event: content_block_delta\n" +
      'data: {"index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"k\\":1}"}}\n' +
      "\n" +
      "event: message_stop\n" +
      'data: {"type":"message_stop"}\n' +
      "\n";

    const events: StreamEvent[] = [];
    for await (const ev of anthropicSseToEvents(streamFromString(sse))) events.push(ev);

    expect(events[0]?.type).toBe("thinking_delta");
    expect(events[1]?.type).toBe("tool_use_delta");
    if (events[1]?.type === "tool_use_delta") {
      expect(events[1].partial_json).toBe('{"k":1}');
    }
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

  it("issues a POST with x-api-key and parses the response", async () => {
    server.requests.length = 0;
    server.setCases([
      {
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          id: "msg_x",
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "hello there" }],
          stop_reason: "end_turn",
          usage: { input_tokens: 5, output_tokens: 2 },
        }),
      },
    ]);
    const client = new AnthropicModelInterface("test-key", "claude-sonnet-4-6", {
      baseUrl: server.baseUrl(),
    });
    const r = await client.call(req([user("hi")]));
    expect(r.content).toEqual([{ type: "text", text: "hello there" }]);
    expect(r.usage.input_tokens).toBe(5);
    const req0 = server.requests[0]!;
    expect(req0.url).toBe("/v1/messages");
    expect(req0.headers["x-api-key"]).toBe("test-key");
    expect(req0.headers["anthropic-version"]).toBe("2023-06-01");
  });

  it("maps 429 with Retry-After to RateLimited(retry_after)", async () => {
    server.requests.length = 0;
    server.setCases([{ status: 429, headers: { "retry-after": "7" }, body: "" }]);
    const sleep = recordingSleep();
    const client = new AnthropicModelInterface("k", "claude-sonnet-4-6", {
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

  it("maps 529 to RateLimited(null)", async () => {
    server.requests.length = 0;
    server.setCases([{ status: 529, body: "" }]);
    const client = new AnthropicModelInterface("k", "claude-sonnet-4-6", {
      baseUrl: server.baseUrl(),
      maxRetries: 0,
    });
    let err: unknown;
    try {
      await client.call(req([user("hi")]));
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(RateLimited);
    expect((err as RateLimited).retryAfter).toBeNull();
  });

  it("maps 408 to Timeout and 504 to Timeout", async () => {
    server.requests.length = 0;
    server.setCases([{ status: 408, body: "" }]);
    const client = new AnthropicModelInterface("k", "claude-sonnet-4-6", {
      baseUrl: server.baseUrl(),
      maxRetries: 0,
    });
    await expect(client.call(req([user("hi")]))).rejects.toBeInstanceOf(Timeout);
    server.setCases([{ status: 504, body: "" }]);
    await expect(client.call(req([user("hi")]))).rejects.toBeInstanceOf(Timeout);
  });

  it("maps 400 to ProviderError with the Anthropic-supplied message", async () => {
    server.requests.length = 0;
    server.setCases([
      {
        status: 400,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          type: "error",
          error: { type: "invalid_request_error", message: "max_tokens must be > 0" },
        }),
      },
    ]);
    const client = new AnthropicModelInterface("k", "claude-sonnet-4-6", {
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

  it("retries 500 with exponential backoff then succeeds", async () => {
    server.requests.length = 0;
    server.setCases([
      { status: 500, body: "" },
      { status: 500, body: "" },
      {
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          content: [{ type: "text", text: "after retry" }],
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
      },
    ]);
    const sleep = recordingSleep();
    const client = new AnthropicModelInterface("k", "claude-sonnet-4-6", {
      baseUrl: server.baseUrl(),
      sleep: sleep.fn,
    });
    const r = await client.call(req([user("hi")]));
    expect(r.content).toEqual([{ type: "text", text: "after retry" }]);
    expect(server.requests.length).toBe(3);
    // First retry waited backoffDelayMs(0)=500ms, second backoffDelayMs(1)=1000ms.
    expect(sleep.delays).toEqual([500, 1000]);
  });

  it("honors Retry-After (seconds) on a retryable response", async () => {
    server.requests.length = 0;
    server.setCases([
      { status: 429, headers: { "retry-after": "2" }, body: "" },
      {
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          content: [{ type: "text", text: "ok" }],
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
      },
    ]);
    const sleep = recordingSleep();
    const client = new AnthropicModelInterface("k", "claude-sonnet-4-6", {
      baseUrl: server.baseUrl(),
      sleep: sleep.fn,
    });
    await client.call(req([user("hi")]));
    expect(sleep.delays).toEqual([2000]);
  });

  it("countTokens hits /v1/messages/count_tokens", async () => {
    server.requests.length = 0;
    server.setCases([
      {
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ input_tokens: 42 }),
      },
    ]);
    const client = new AnthropicModelInterface("k", "claude-sonnet-4-6", {
      baseUrl: server.baseUrl(),
    });
    const n = await client.countTokens(req([user("hi")]));
    expect(n).toBe(42);
    expect(server.requests[0]!.url).toBe("/v1/messages/count_tokens");
  });

  it("callStreaming() returns parsed events end-to-end", async () => {
    server.requests.length = 0;
    const sse =
      "event: message_start\n" +
      'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}\n' +
      "\n" +
      "event: content_block_delta\n" +
      'data: {"index":0,"delta":{"type":"text_delta","text":"hi"}}\n' +
      "\n" +
      "event: content_block_stop\n" +
      'data: {"index":0}\n' +
      "\n" +
      "event: message_delta\n" +
      'data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n' +
      "\n" +
      "event: message_stop\n" +
      'data: {"type":"message_stop"}\n' +
      "\n";
    server.setCases([
      {
        status: 200,
        headers: { "content-type": "text/event-stream" },
        body: sse,
      },
    ]);
    const client = new AnthropicModelInterface("k", "claude-sonnet-4-6", {
      baseUrl: server.baseUrl(),
    });
    const events: StreamEvent[] = [];
    for await (const ev of client.callStreaming(req([user("hi")]))) events.push(ev);
    expect(events.map((e) => e.type)).toEqual([
      "message_start",
      "content_block_delta",
      "content_block_stop",
      "message_stop",
    ]);
    const last = events[events.length - 1];
    if (last?.type === "message_stop") {
      expect(last.usage.input_tokens).toBe(3);
      expect(last.usage.output_tokens).toBe(5);
    }
  });
});

// ---------------------------------------------------------------------------
// Live-API tests — skipped by default
// ---------------------------------------------------------------------------

const LIVE = process.env.ANTHROPIC_API_KEY != null && process.env.ANTHROPIC_API_KEY !== "";

describe.skipIf(!LIVE)("AnthropicModelInterface — live API", () => {
  const modelId = process.env.ANTHROPIC_TEST_MODEL ?? "claude-sonnet-4-6";

  it("call() returns a non-empty usage", async () => {
    const client = AnthropicModelInterface.fromEnv("ANTHROPIC_API_KEY", modelId);
    const r = await client.call(req([user("Reply with the word 'pong'.")]));
    expect(r.usage.input_tokens).toBeGreaterThan(0);
    expect(r.usage.output_tokens).toBeGreaterThan(0);
  });

  it("countTokens() returns a positive count", async () => {
    const client = AnthropicModelInterface.fromEnv("ANTHROPIC_API_KEY", modelId);
    const n = await client.countTokens(req([user("count my tokens please")]));
    expect(n).toBeGreaterThan(0);
  });

  it("callStreaming() emits a MessageStop event", async () => {
    const client = AnthropicModelInterface.fromEnv("ANTHROPIC_API_KEY", modelId);
    let sawStop = false;
    for await (const ev of client.callStreaming(req([user("Reply with the word 'pong'.")]))) {
      if (ev.type === "message_stop") sawStop = true;
    }
    expect(sawStop).toBe(true);
  });
});
