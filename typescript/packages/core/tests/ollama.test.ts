/**
 * Unit + end-to-end mocked-HTTP tests for `OllamaModelInterface`
 * (spore-core issue #41). Mirrors the Rust suite rule-for-rule.
 */

import * as http from "node:http";
import type { AddressInfo } from "node:net";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  OllamaModelInterface,
  OLLAMA_DEFAULT_BASE_URL,
  OLLAMA_DEFAULT_KEEP_ALIVE,
  OLLAMA_DEFAULT_TIMEOUT_MS,
  ProviderError,
  Timeout,
  ollamaBuildRequest,
  ollamaNameMatches,
  ollamaNdjsonToEvents,
  ollamaParseResponse,
  ollamaParseStopReason,
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

function req(messages: Message[], extra: Partial<ModelRequest> = {}): ModelRequest {
  return {
    messages,
    tools: extra.tools ?? [],
    params: extra.params ?? { stop_sequences: [] },
    stream: extra.stream ?? false,
  };
}

interface MockHandler {
  (
    req: http.IncomingMessage,
    body: string,
  ): { status: number; headers?: Record<string, string>; body?: string };
}

class MockServer {
  readonly server: http.Server;
  port = 0;
  requests: { method: string; url: string; headers: http.IncomingHttpHeaders; body: string }[] = [];
  private routes: Map<string, MockHandler[]> = new Map();
  private callCounts: Map<string, number> = new Map();

  constructor() {
    this.server = http.createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", () => {
        const body = Buffer.concat(chunks).toString("utf-8");
        const key = `${req.method ?? "GET"} ${req.url ?? ""}`;
        this.requests.push({
          method: req.method ?? "GET",
          url: req.url ?? "",
          headers: req.headers,
          body,
        });
        const handlers = this.routes.get(key) ?? [];
        const count = this.callCounts.get(key) ?? 0;
        this.callCounts.set(key, count + 1);
        const handler = handlers[Math.min(count, handlers.length - 1)];
        if (handler == null) {
          res.writeHead(404, { "content-type": "text/plain" });
          res.end("no route");
          return;
        }
        const r = handler(req, body);
        res.writeHead(r.status, r.headers ?? {});
        res.end(r.body ?? "");
      });
    });
  }

  route(method: string, path: string, ...handlers: MockHandler[]): this {
    this.routes.set(`${method} ${path}`, handlers);
    return this;
  }

  callCount(method: string, path: string): number {
    return this.callCounts.get(`${method} ${path}`) ?? 0;
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

  reset(): void {
    this.routes.clear();
    this.callCounts.clear();
    this.requests.length = 0;
  }
}

function tagsOk(model: string): MockHandler {
  return () => ({
    status: 200,
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ models: [{ name: `${model}:latest` }] }),
  });
}

// ---------------------------------------------------------------------------
// constructors / defaults
// ---------------------------------------------------------------------------

describe("OllamaModelInterface defaults", () => {
  it("new uses localhost defaults", () => {
    const c = new OllamaModelInterface("llama3.2");
    const j = c.toJSON();
    expect(j.model_id).toBe("llama3.2");
    expect(j.base_url).toBe("http://localhost:11434");
    expect(j.timeout_ms).toBe(300_000);
    expect(j.keep_alive).toBe("5m");
  });

  it("withBaseUrl overrides", () => {
    const c = OllamaModelInterface.withBaseUrl("mistral", "http://remote:9999");
    expect(c.toJSON().base_url).toBe("http://remote:9999");
    expect(c.toJSON().model_id).toBe("mistral");
  });

  it("defaults match spec", () => {
    expect(OLLAMA_DEFAULT_BASE_URL).toBe("http://localhost:11434");
    expect(OLLAMA_DEFAULT_TIMEOUT_MS).toBe(300_000);
    expect(OLLAMA_DEFAULT_KEEP_ALIVE).toBe("5m");
  });
});

// ---------------------------------------------------------------------------
// buildRequest
// ---------------------------------------------------------------------------

describe("buildRequest", () => {
  it("serializes options and keep_alive", () => {
    const r = req([user("hi")]);
    r.params.max_tokens = 256;
    r.params.temperature = 0.7;
    r.params.top_p = 0.9;
    r.params.stop_sequences = ["END"];
    const body = ollamaBuildRequest("llama3.2", "10m", r, false);
    expect(body.keep_alive).toBe("10m");
    expect(body.options?.num_predict).toBe(256);
    expect(body.options?.temperature).toBe(0.7);
    expect(body.options?.top_p).toBe(0.9);
    expect(body.options?.stop).toEqual(["END"]);
    expect(body.stream).toBe(false);
  });

  it("omits options object when no sampling params set", () => {
    const body = ollamaBuildRequest("llama3.2", null, req([user("hi")]), false);
    expect(body.options).toBeUndefined();
    expect(body.keep_alive).toBeUndefined();
  });

  it("serializes tools", () => {
    const r = req([user("hi")]);
    r.tools.push({
      name: "search",
      description: "search the web",
      input_schema: { type: "object" },
    });
    const body = ollamaBuildRequest("llama3.2", null, r, false);
    const s = JSON.stringify(body);
    expect(s).toContain('"tools":[');
    expect(s).toContain('"name":"search"');
    expect(s).toContain('"type":"function"');
  });

  it("tool_call uses object arguments (NOT a JSON-encoded string)", () => {
    const r = req([
      {
        role: "assistant",
        content: { type: "tool_call", id: "call-0", name: "fetch", input: { url: "x" } },
      },
    ]);
    const body = ollamaBuildRequest("llama3.2", null, r, false);
    const m = body.messages[0]!;
    expect(typeof m.tool_calls![0]!.function.arguments).toBe("object");
    expect(m.tool_calls![0]!.function.arguments).toEqual({ url: "x" });
    const s = JSON.stringify(m);
    expect(s).toContain('"arguments":{"url":"x"}');
    expect(s).not.toContain('"arguments":"');
  });

  it("tool_result maps to role=tool with tool_call_id", () => {
    const r = req([
      {
        role: "tool",
        content: { type: "tool_result", tool_use_id: "call-0", content: "ok", is_error: false },
      },
    ]);
    const body = ollamaBuildRequest("llama3.2", null, r, false);
    const m = body.messages[0]!;
    expect(m.role).toBe("tool");
    expect(m.content).toBe("ok");
    expect(m.tool_call_id).toBe("call-0");
  });

  it("never emits a thinking key", () => {
    const body = ollamaBuildRequest("llama3.2", null, req([user("hi")]), false);
    const s = JSON.stringify(body);
    expect(s).not.toContain("thinking");
  });
});

// ---------------------------------------------------------------------------
// parseStopReason
// ---------------------------------------------------------------------------

describe("parseStopReason", () => {
  it("maps documented values", () => {
    expect(ollamaParseStopReason("stop")).toBe("end_turn");
    expect(ollamaParseStopReason("tool_calls")).toBe("tool_use");
    expect(ollamaParseStopReason("length")).toBe("max_tokens");
    expect(ollamaParseStopReason(null)).toBe("end_turn");
    expect(ollamaParseStopReason("???")).toBe("end_turn");
  });
});

// ---------------------------------------------------------------------------
// parseResponse
// ---------------------------------------------------------------------------

describe("parseResponse", () => {
  it("extracts text and usage", () => {
    const r = ollamaParseResponse({
      message: { role: "assistant", content: "hi" },
      done: true,
      done_reason: "stop",
      prompt_eval_count: 7,
      eval_count: 2,
    });
    expect(r.usage.input_tokens).toBe(7);
    expect(r.usage.output_tokens).toBe(2);
    expect(r.stop_reason).toBe("end_turn");
    expect(r.content).toEqual([{ type: "text", text: "hi" }]);
  });

  it("cache fields always null", () => {
    const r = ollamaParseResponse({
      message: { role: "assistant", content: "x" },
      done: true,
      prompt_eval_count: 1,
      eval_count: 1,
    });
    expect(r.usage.cache_read_tokens).toBeNull();
    expect(r.usage.cache_write_tokens).toBeNull();
  });

  it("synthesizes tool_call ids per index", () => {
    const r = ollamaParseResponse({
      message: {
        role: "assistant",
        tool_calls: [
          { function: { name: "fetch", arguments: { url: "x" } } },
          { function: { name: "search", arguments: { q: "y" } } },
        ],
      },
      done: true,
      done_reason: "tool_calls",
      prompt_eval_count: 1,
      eval_count: 1,
    });
    expect(r.stop_reason).toBe("tool_use");
    const a = r.content[0] as Extract<ContentBlock, { type: "tool_use" }>;
    expect(a.id).toBe("call-0");
    expect(a.name).toBe("fetch");
    expect(a.input).toEqual({ url: "x" });
    const b = r.content[1] as Extract<ContentBlock, { type: "tool_use" }>;
    expect(b.id).toBe("call-1");
  });
});

// ---------------------------------------------------------------------------
// context_window / provider
// ---------------------------------------------------------------------------

describe("contextWindow / provider()", () => {
  it("uses static table", () => {
    expect(OllamaModelInterface.contextWindow("llama3.2")).toBe(128_000);
    expect(OllamaModelInterface.contextWindow("llama3.2:3b")).toBe(128_000);
    expect(OllamaModelInterface.contextWindow("qwen2.5-coder-7b")).toBe(128_000);
    expect(OllamaModelInterface.contextWindow("mistral")).toBe(32_000);
    expect(OllamaModelInterface.contextWindow("gemma")).toBe(8_192);
    expect(OllamaModelInterface.contextWindow("unknown")).toBe(0);
  });

  it("provider() identity", () => {
    const c = new OllamaModelInterface("llama3.2");
    const p = c.provider();
    expect(p.name).toBe("ollama");
    expect(p.model_id).toBe("llama3.2");
    expect(p.context_window).toBe(128_000);
  });
});

// ---------------------------------------------------------------------------
// nameMatches
// ---------------------------------------------------------------------------

describe("nameMatches", () => {
  it("handles latest tag and bare name", () => {
    expect(ollamaNameMatches("llama3.2:latest", "llama3.2")).toBe(true);
    expect(ollamaNameMatches("llama3.2", "llama3.2")).toBe(true);
    expect(ollamaNameMatches("llama3.2:3b", "llama3.2")).toBe(true);
    expect(ollamaNameMatches("llama3.1", "llama3.2")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// ndjsonToEvents — driven from synthesized ReadableStreams
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

describe("ndjsonToEvents", () => {
  it("emits text deltas and terminates on done:true", async () => {
    const ndjson =
      '{"message":{"role":"assistant","content":"hello"},"done":false}\n' +
      '{"message":{"role":"assistant","content":" world"},"done":false}\n' +
      '{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":5}\n';
    const events: StreamEvent[] = [];
    for await (const ev of ollamaNdjsonToEvents(streamFromString(ndjson))) events.push(ev);
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

  it("parses multiple NDJSON lines per chunk", async () => {
    const ndjson =
      '{"message":{"role":"assistant","content":"ab"},"done":false}\n' +
      '{"message":{"role":"assistant","content":"cd"},"done":false}\n' +
      '{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}\n';
    const deltas: string[] = [];
    for await (const ev of ollamaNdjsonToEvents(streamFromString(ndjson))) {
      if (ev.type === "content_block_delta") deltas.push(ev.delta);
    }
    expect(deltas).toEqual(["ab", "cd"]);
  });

  it("done carries usage", async () => {
    const ndjson =
      '{"message":{"role":"assistant","content":"x"},"done":true,"done_reason":"stop","prompt_eval_count":42,"eval_count":7}\n';
    let usage: { input_tokens: number; output_tokens: number } | null = null;
    for await (const ev of ollamaNdjsonToEvents(streamFromString(ndjson))) {
      if (ev.type === "message_stop") {
        usage = { input_tokens: ev.usage.input_tokens, output_tokens: ev.usage.output_tokens };
      }
    }
    expect(usage).toEqual({ input_tokens: 42, output_tokens: 7 });
  });

  it("accumulates tool calls (full arguments object per chunk)", async () => {
    const ndjson =
      '{"message":{"role":"assistant","tool_calls":[{"function":{"name":"fetch","arguments":{"url":"x"}}}]},"done":false}\n' +
      '{"message":{"role":"assistant","content":""},"done":true,"done_reason":"tool_calls","prompt_eval_count":1,"eval_count":1}\n';
    const toolJsons: string[] = [];
    let finalStop = "";
    for await (const ev of ollamaNdjsonToEvents(streamFromString(ndjson))) {
      if (ev.type === "tool_use_delta") toolJsons.push(ev.partial_json);
      if (ev.type === "message_stop") finalStop = ev.stop_reason;
    }
    expect(toolJsons).toHaveLength(1);
    expect(JSON.parse(toolJsons[0]!)).toEqual({ url: "x" });
    expect(finalStop).toBe("tool_use");
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

  it("issues POST /api/chat and parses the response", async () => {
    server.reset();
    server.route("GET", "/api/tags", tagsOk("llama3.2"));
    server.route("POST", "/api/chat", () => ({
      status: 200,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        message: { role: "assistant", content: "hello there" },
        done: true,
        done_reason: "stop",
        prompt_eval_count: 5,
        eval_count: 2,
      }),
    }));
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    const r = await client.call(req([user("hi")]));
    expect(r.content).toEqual([{ type: "text", text: "hello there" }]);
    expect(r.usage.input_tokens).toBe(5);
    expect(r.usage.output_tokens).toBe(2);
    const chatReq = server.requests.find((q) => q.url === "/api/chat")!;
    expect(chatReq.headers["content-type"]).toBe("application/json");
  });

  it("connection refused yields a helpful 'Ollama not running' message", async () => {
    // Closed port → connect should fail immediately.
    const client = new OllamaModelInterface("llama3.2", {
      baseUrl: "http://127.0.0.1:1",
      timeoutMs: 2_000,
    });
    let err: unknown;
    try {
      await client.call(req([user("hi")]));
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).code).toBe(0);
    expect((err as ProviderError).message).toContain("Ollama not running");
  });

  it("connection error does NOT retry (fail-fast)", async () => {
    const client = new OllamaModelInterface("llama3.2", {
      baseUrl: "http://127.0.0.1:1",
      timeoutMs: 5_000,
    });
    const start = Date.now();
    try {
      await client.call(req([user("hi")]));
    } catch {
      /* expected */
    }
    const elapsed = Date.now() - start;
    expect(elapsed).toBeLessThan(500);
  });

  it("model not found suggests `ollama pull`", async () => {
    server.reset();
    server.route("GET", "/api/tags", () => ({
      status: 200,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ models: [{ name: "mistral:latest" }] }),
    }));
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    let err: unknown;
    try {
      await client.call(req([user("hi")]));
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).code).toBe(404);
    expect((err as ProviderError).message).toContain("ollama pull llama3.2");
  });

  it("/api/chat 404 maps to pull suggestion", async () => {
    server.reset();
    server.route("GET", "/api/tags", tagsOk("llama3.2"));
    server.route("POST", "/api/chat", () => ({
      status: 404,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ error: "model 'llama3.2' not found, try pulling it first" }),
    }));
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    let err: unknown;
    try {
      await client.call(req([user("hi")]));
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).code).toBe(404);
    expect((err as ProviderError).message).toContain("ollama pull llama3.2");
  });

  it("timeout maps to Timeout", async () => {
    server.reset();
    server.route("GET", "/api/tags", tagsOk("llama3.2"));
    // Chat handler never responds → request will hit our short client timeout.
    const stallSrv = http.createServer((_req, _res) => {
      /* hold the socket open without responding */
    });
    await new Promise<void>((resolve) => stallSrv.listen(0, "127.0.0.1", () => resolve()));
    const stallPort = (stallSrv.address() as AddressInfo).port;
    try {
      // Use a chat-stalling server but a tags-ok server is needed first; the
      // simplest setup: stand up the stall server alone and rely on the fact
      // that /api/tags will also stall, exercising the timeout path on the
      // probe call.
      const client = new OllamaModelInterface("llama3.2", {
        baseUrl: `http://127.0.0.1:${stallPort}`,
        timeoutMs: 150,
      });
      let err: unknown;
      try {
        await client.call(req([user("hi")]));
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(Timeout);
    } finally {
      await new Promise<void>((resolve) => stallSrv.close(() => resolve()));
    }
  });

  it("model check is cached after the first call", async () => {
    server.reset();
    server.route("GET", "/api/tags", tagsOk("llama3.2"));
    server.route("POST", "/api/chat", () => ({
      status: 200,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        message: { role: "assistant", content: "ok" },
        done: true,
        done_reason: "stop",
        prompt_eval_count: 1,
        eval_count: 1,
      }),
    }));
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    await client.call(req([user("a")]));
    await client.call(req([user("b")]));
    expect(server.callCount("GET", "/api/tags")).toBe(1);
    expect(server.callCount("POST", "/api/chat")).toBe(2);
  });

  it("callStreaming returns parsed NDJSON events end-to-end", async () => {
    server.reset();
    server.route("GET", "/api/tags", tagsOk("llama3.2"));
    const ndjson =
      '{"message":{"role":"assistant","content":"hi"},"done":false}\n' +
      '{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":5}\n';
    server.route("POST", "/api/chat", () => ({
      status: 200,
      headers: { "content-type": "application/x-ndjson" },
      body: ndjson,
    }));
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    const events: StreamEvent[] = [];
    for await (const ev of client.callStreaming(req([user("hi")]))) events.push(ev);
    expect(events.map((e) => e.type)).toContain("message_stop");
    const last = events[events.length - 1];
    if (last?.type === "message_stop") {
      expect(last.usage.input_tokens).toBe(3);
      expect(last.usage.output_tokens).toBe(5);
    }
  });
});

// ---------------------------------------------------------------------------
// count_tokens
// ---------------------------------------------------------------------------

describe("countTokens", () => {
  let server: MockServer;
  beforeAll(async () => {
    server = new MockServer();
    await server.start();
  });
  afterAll(async () => {
    await server.stop();
  });

  it("uses /api/embed when available", async () => {
    server.reset();
    server.route("POST", "/api/embed", () => ({
      status: 200,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt_eval_count: 123 }),
    }));
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    const n = await client.countTokens(req([user("hello world")]));
    expect(n).toBe(123);
  });

  it("falls back to bytes/4 when embed endpoint errors", async () => {
    server.reset();
    server.route("POST", "/api/embed", () => ({ status: 500, body: "" }));
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    // 40 chars + newline = 41 → floor(41/4) = 10
    const n = await client.countTokens(req([user("a".repeat(40))]));
    expect(n).toBe(10);
  });
});

// ---------------------------------------------------------------------------
// /api/show discovery + tool-capability guard
// ---------------------------------------------------------------------------

function chatOk(): MockHandler {
  return () => ({
    status: 200,
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      message: { role: "assistant", content: "ok" },
      done: true,
      done_reason: "stop",
      prompt_eval_count: 1,
      eval_count: 1,
    }),
  });
}

function showOk(body: unknown): MockHandler {
  return () => ({
    status: 200,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

function toolReq(): ModelRequest {
  const r = req([user("use a tool")]);
  r.tools.push({ name: "search", description: "search the web", input_schema: { type: "object" } });
  return r;
}

describe("/api/show discovery + tool guard", () => {
  let server: MockServer;
  beforeAll(async () => {
    server = new MockServer();
    await server.start();
  });
  afterAll(async () => {
    await server.stop();
  });

  it("provider() reflects discovered context window after the probe", async () => {
    server.reset();
    server.route("GET", "/api/tags", tagsOk("llama3.2"));
    server.route(
      "POST",
      "/api/show",
      showOk({ model_info: { "llama.context_length": 16_384 }, capabilities: ["tools"] }),
    );
    server.route("POST", "/api/chat", chatOk());
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    // Before the probe runs, provider() falls back to the static table.
    expect(client.provider().context_window).toBe(128_000);
    await client.call(req([user("hi")]));
    // After the probe, provider() reflects the discovered value.
    expect(client.provider().context_window).toBe(16_384);
  });

  it("falls back to the static table when /api/show 404s", async () => {
    server.reset();
    server.route("GET", "/api/tags", tagsOk("llama3.2"));
    server.route("POST", "/api/show", () => ({ status: 404, body: "" }));
    server.route("POST", "/api/chat", chatOk());
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    await client.call(req([user("hi")]));
    expect(client.provider().context_window).toBe(128_000);
  });

  it("falls back when context_length is missing from /api/show", async () => {
    server.reset();
    server.route("GET", "/api/tags", tagsOk("llama3.2"));
    server.route(
      "POST",
      "/api/show",
      showOk({ model_info: { "general.architecture": "llama" }, capabilities: ["tools"] }),
    );
    server.route("POST", "/api/chat", chatOk());
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    await client.call(req([user("hi")]));
    expect(client.provider().context_window).toBe(128_000);
  });

  it("rejects tool requests when the model lacks the 'tools' capability", async () => {
    server.reset();
    server.route("GET", "/api/tags", tagsOk("gemma"));
    // capabilities lacks "tools"; no /api/chat route — a call would 404.
    server.route(
      "POST",
      "/api/show",
      showOk({ model_info: { "gemma.context_length": 8_192 }, capabilities: ["completion"] }),
    );
    const client = OllamaModelInterface.withBaseUrl("gemma", server.baseUrl());
    let err: unknown;
    try {
      await client.call(toolReq());
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).code).toBe(0);
    expect((err as ProviderError).message).toContain("does not support tool calling");
    expect(server.callCount("POST", "/api/chat")).toBe(0);
  });

  it("proceeds normally when the model has the 'tools' capability", async () => {
    server.reset();
    server.route("GET", "/api/tags", tagsOk("llama3.2"));
    server.route(
      "POST",
      "/api/show",
      showOk({
        model_info: { "llama.context_length": 128_000 },
        capabilities: ["completion", "tools"],
      }),
    );
    server.route("POST", "/api/chat", chatOk());
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    const r = await client.call(toolReq());
    expect(r.content).toEqual([{ type: "text", text: "ok" }]);
  });

  it("fetches /api/show at most once across multiple calls", async () => {
    server.reset();
    server.route("GET", "/api/tags", tagsOk("llama3.2"));
    server.route(
      "POST",
      "/api/show",
      showOk({ model_info: { "llama.context_length": 32_000 }, capabilities: ["tools"] }),
    );
    server.route("POST", "/api/chat", chatOk());
    const client = OllamaModelInterface.withBaseUrl("llama3.2", server.baseUrl());
    await client.call(req([user("a")]));
    await client.call(req([user("b")]));
    expect(server.callCount("POST", "/api/show")).toBe(1);
    expect(server.callCount("POST", "/api/chat")).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// Live-API tests — skipped by default
// ---------------------------------------------------------------------------

const LIVE = process.env.OLLAMA_LIVE === "1";

describe.skipIf(!LIVE)("OllamaModelInterface — live API", () => {
  const modelId = process.env.OLLAMA_TEST_MODEL ?? "llama3.2";

  it("call() returns non-empty usage", async () => {
    const client = new OllamaModelInterface(modelId);
    const r = await client.call(req([user("Reply with the word 'pong'.")]));
    expect(r.usage.input_tokens).toBeGreaterThan(0);
    expect(r.usage.output_tokens).toBeGreaterThan(0);
  });

  it("callStreaming() emits a message_stop event", async () => {
    const client = new OllamaModelInterface(modelId);
    let sawStop = false;
    for await (const ev of client.callStreaming(req([user("Reply with the word 'pong'.")]))) {
      if (ev.type === "message_stop") sawStop = true;
    }
    expect(sawStop).toBe(true);
  });
});
