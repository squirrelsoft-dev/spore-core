/**
 * `web_search` auth-headers + GET/param-encoding tests (#108, TypeScript).
 *
 * Mirrors the Rust suite in `rust/crates/spore-core/src/tools/web.rs` in
 * scenario coverage, asserting OUTBOUND request shape against an ephemeral
 * `node:http` server (the repo's existing HTTP-mock approach — never the live
 * network). Each env-var test uses a UNIQUE name so parallel execution never
 * races on a shared one.
 */

import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";

import { harnessTesting, toolRegistry, type ToolCall } from "@spore/core";
import { afterEach, describe, expect, it } from "vitest";

import {
  applyWebFetchRange,
  UrlPolicy,
  validateFetchUrl,
  WebFetchTool,
  WebSearchConfigError,
  WebSearchTool,
} from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;
const ctx = toolRegistry.toolRegistryMock.testCtx();

function call(input: unknown): ToolCall {
  return { id: "c1", name: "web_search", input };
}

interface CapturedRequest {
  method?: string;
  url?: string;
  headers: IncomingMessage["headers"];
  body: string;
}

/** Start a one-shot server that captures the inbound request and replies with
 *  `responseBody`. Returns the base URL, a promise resolving to the captured
 *  request, and a close fn. */
async function startCapturingServer(responseBody: string): Promise<{
  url: string;
  captured: Promise<CapturedRequest>;
  close: () => Promise<void>;
}> {
  let resolveCaptured: (r: CapturedRequest) => void;
  const captured = new Promise<CapturedRequest>((res) => {
    resolveCaptured = res;
  });
  const server: Server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk: Buffer) => {
      body += chunk.toString("utf8");
    });
    req.on("end", () => {
      resolveCaptured({
        method: req.method,
        url: req.url,
        headers: req.headers,
        body,
      });
      res.statusCode = 200;
      res.end(responseBody);
    });
  });
  await new Promise<void>((res) => server.listen(0, "127.0.0.1", res));
  const addr = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${addr.port}`,
    captured,
    close: () => new Promise<void>((res) => server.close(() => res())),
  };
}

const setEnvNames: string[] = [];
function setEnv(name: string, value: string): string {
  process.env[name] = value;
  setEnvNames.push(name);
  return name;
}

afterEach(() => {
  for (const name of setEnvNames.splice(0)) {
    delete process.env[name];
  }
});

describe("web_search #108 — POST default unchanged", () => {
  it('withEndpoint still POSTs {"query": q} as JSON with only content-type', async () => {
    const { url, captured, close } = await startCapturingServer("res");
    try {
      const r = await WebSearchTool.withEndpoint(`${url}/search`).execute(
        call({ query: "rust" }),
        new AllowAllSandbox(),
        ctx,
      );
      const req = await captured;
      expect(req.method).toBe("POST");
      expect(req.url).toBe("/search");
      expect(JSON.parse(req.body)).toEqual({ query: "rust" });
      expect(req.headers["content-type"]).toContain("application/json");
      expect(r.kind).toBe("success");
      if (r.kind !== "success") throw new Error("unreachable");
      expect(r.content).toBe("res");
    } finally {
      await close();
    }
  });

  it('withConfig POST default keeps the {"query"} body shape', async () => {
    const { url, captured, close } = await startCapturingServer("res");
    try {
      const tool = WebSearchTool.withConfig({ endpoint: `${url}/search` });
      const r = await tool.execute(
        call({ query: "rust" }),
        new AllowAllSandbox(),
        ctx,
      );
      const req = await captured;
      expect(req.method).toBe("POST");
      expect(JSON.parse(req.body)).toEqual({ query: "rust" });
      expect(r.kind).toBe("success");
      if (r.kind !== "success") throw new Error("unreachable");
      expect(r.content).toBe("res");
    } finally {
      await close();
    }
  });
});

describe("web_search #108 — GET param encoding", () => {
  it("URL-encodes the query into the query string under queryParam (special chars)", async () => {
    const { url, captured, close } = await startCapturingServer("get-results");
    try {
      const tool = WebSearchTool.withConfig({
        endpoint: `${url}/search`,
        method: "GET",
        queryParam: "q",
      });
      const r = await tool.execute(
        call({ query: "rust & go" }),
        new AllowAllSandbox(),
        ctx,
      );
      const req = await captured;
      expect(req.method).toBe("GET");
      // No JSON body on GET.
      expect(req.body).toBe("");
      // Spaces and `&` are percent-encoded in the raw URL.
      expect(req.url).not.toContain(" ");
      const parsed = new URL(req.url ?? "", "http://x");
      expect(parsed.pathname).toBe("/search");
      expect(parsed.searchParams.get("q")).toBe("rust & go");
      expect(r.kind).toBe("success");
      if (r.kind !== "success") throw new Error("unreachable");
      expect(r.content).toBe("get-results");
    } finally {
      await close();
    }
  });

  it("preserves a pre-existing query string on the endpoint (SearXNG ?format=json)", async () => {
    const { url, captured, close } = await startCapturingServer("ok");
    try {
      const tool = WebSearchTool.withConfig({
        // Endpoint already carries `?format=json` (SearXNG JSON API).
        endpoint: `${url}/search?format=json`,
        method: "GET",
        queryParam: "q",
      });
      const r = await tool.execute(
        call({ query: "rust lang" }),
        new AllowAllSandbox(),
        ctx,
      );
      const req = await captured;
      expect(req.method).toBe("GET");
      const parsed = new URL(req.url ?? "", "http://x");
      expect(parsed.pathname).toBe("/search");
      // BOTH the pre-existing param and the added query param are present.
      expect(parsed.searchParams.get("format")).toBe("json");
      expect(parsed.searchParams.get("q")).toBe("rust lang");
      expect(r.kind).toBe("success");
    } finally {
      await close();
    }
  });
});

describe("web_search #108 — auth headers", () => {
  it("attaches the auth header on GET", async () => {
    const env = setEnv("__SPORE_TEST_BRAVE_KEY_GET__", "brave-secret");
    const { url, captured, close } = await startCapturingServer("ok");
    try {
      const tool = WebSearchTool.withConfig({
        endpoint: `${url}/search`,
        method: "GET",
        queryParam: "q",
        authHeaders: [["x-subscription-token", env]],
      });
      const r = await tool.execute(
        call({ query: "x" }),
        new AllowAllSandbox(),
        ctx,
      );
      const req = await captured;
      expect(req.headers["x-subscription-token"]).toBe("brave-secret");
      expect(r.kind).toBe("success");
    } finally {
      await close();
    }
  });

  it("attaches the auth header on POST", async () => {
    const env = setEnv("__SPORE_TEST_BRAVE_KEY_POST__", "brave-secret");
    const { url, captured, close } = await startCapturingServer("ok");
    try {
      const tool = WebSearchTool.withConfig({
        endpoint: `${url}/search`,
        method: "POST",
        authHeaders: [["x-subscription-token", env]],
      });
      const r = await tool.execute(
        call({ query: "x" }),
        new AllowAllSandbox(),
        ctx,
      );
      const req = await captured;
      expect(req.method).toBe("POST");
      expect(req.headers["x-subscription-token"]).toBe("brave-secret");
      expect(r.kind).toBe("success");
    } finally {
      await close();
    }
  });

  it("attaches multiple auth headers", async () => {
    const a = setEnv("__SPORE_TEST_MULTI_A__", "aaa");
    const b = setEnv("__SPORE_TEST_MULTI_B__", "bbb");
    const { url, captured, close } = await startCapturingServer("ok");
    try {
      const tool = WebSearchTool.withConfig({
        endpoint: `${url}/search`,
        method: "POST",
        authHeaders: [
          ["x-a", a],
          ["x-b", b],
        ],
      });
      const r = await tool.execute(
        call({ query: "x" }),
        new AllowAllSandbox(),
        ctx,
      );
      const req = await captured;
      expect(req.headers["x-a"]).toBe("aaa");
      expect(req.headers["x-b"]).toBe("bbb");
      expect(r.kind).toBe("success");
    } finally {
      await close();
    }
  });

  it("no auth configured → POST carries only content-type, no auth headers", async () => {
    const { url, captured, close } = await startCapturingServer("ok");
    try {
      const tool = WebSearchTool.withConfig({ endpoint: `${url}/search` });
      const r = await tool.execute(
        call({ query: "x" }),
        new AllowAllSandbox(),
        ctx,
      );
      const req = await captured;
      expect(req.headers["content-type"]).toContain("application/json");
      expect(req.headers["x-subscription-token"]).toBeUndefined();
      expect(req.headers.authorization).toBeUndefined();
      expect(r.kind).toBe("success");
    } finally {
      await close();
    }
  });
});

describe("web_search #108 — in-body auth (Tavily shape)", () => {
  it('POST body is {"api_key": <env>, "query": q}', async () => {
    const env = setEnv("__SPORE_TEST_TAVILY_KEY__", "tav-secret");
    const { url, captured, close } =
      await startCapturingServer("tavily-results");
    try {
      const tool = WebSearchTool.withConfig({
        endpoint: `${url}/search`,
        method: "POST",
        bodyAuthParams: [["api_key", env]],
      });
      const r = await tool.execute(
        call({ query: "rust" }),
        new AllowAllSandbox(),
        ctx,
      );
      const req = await captured;
      expect(JSON.parse(req.body)).toEqual({
        api_key: "tav-secret",
        query: "rust",
      });
      expect(r.kind).toBe("success");
      if (r.kind !== "success") throw new Error("unreachable");
      expect(r.content).toBe("tavily-results");
    } finally {
      await close();
    }
  });

  it("GET appends the body-auth param to the query string", async () => {
    const env = setEnv("__SPORE_TEST_GET_KEY__", "get-secret");
    const { url, captured, close } = await startCapturingServer("ok");
    try {
      const tool = WebSearchTool.withConfig({
        endpoint: `${url}/search`,
        method: "GET",
        queryParam: "q",
        bodyAuthParams: [["api_key", env]],
      });
      const r = await tool.execute(
        call({ query: "x" }),
        new AllowAllSandbox(),
        ctx,
      );
      const req = await captured;
      const parsed = new URL(req.url ?? "", "http://x");
      expect(parsed.searchParams.get("q")).toBe("x");
      expect(parsed.searchParams.get("api_key")).toBe("get-secret");
      expect(req.body).toBe("");
      expect(r.kind).toBe("success");
    } finally {
      await close();
    }
  });
});

describe("web_search #108 — construction-time env errors (no request)", () => {
  it("missing env var throws WebSearchConfigError at construction", () => {
    const env = "__SPORE_TEST_WEB_MISSING__";
    delete process.env[env];
    let thrown: unknown;
    try {
      WebSearchTool.withConfig({
        endpoint: "http://127.0.0.1:1/never",
        authHeaders: [["x-key", env]],
      });
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(WebSearchConfigError);
    if (!(thrown instanceof WebSearchConfigError))
      throw new Error("unreachable");
    expect(thrown.kind).toBe("env_var_not_set");
    expect(thrown.envVar).toBe(env);
  });

  it("empty (whitespace) env var throws WebSearchConfigError at construction", () => {
    const env = setEnv("__SPORE_TEST_WEB_EMPTY__", "   ");
    let thrown: unknown;
    try {
      WebSearchTool.withConfig({
        endpoint: "http://127.0.0.1:1/never",
        bodyAuthParams: [["api_key", env]],
      });
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(WebSearchConfigError);
    if (!(thrown instanceof WebSearchConfigError))
      throw new Error("unreachable");
    expect(thrown.kind).toBe("env_var_empty");
    expect(thrown.envVar).toBe(env);
  });
});

describe("web_search #108 — response body returned verbatim", () => {
  it("returns the GET body verbatim", async () => {
    const raw = '{"web":{"results":[{"title":"t"}]}}';
    const { url, close } = await startCapturingServer(raw);
    try {
      const tool = WebSearchTool.withConfig({
        endpoint: `${url}/search`,
        method: "GET",
        queryParam: "q",
      });
      const r = await tool.execute(
        call({ query: "t" }),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind).toBe("success");
      if (r.kind !== "success") throw new Error("unreachable");
      expect(r.content).toBe(raw);
    } finally {
      await close();
    }
  });

  it("returns the POST body verbatim", async () => {
    const raw = '{"results":[{"title":"t","url":"u"}]}';
    const { url, close } = await startCapturingServer(raw);
    try {
      const tool = WebSearchTool.withConfig({ endpoint: `${url}/search` });
      const r = await tool.execute(
        call({ query: "t" }),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind).toBe("success");
      if (r.kind !== "success") throw new Error("unreachable");
      expect(r.content).toBe(raw);
    } finally {
      await close();
    }
  });
});

describe("web_search #108 — unconfigured backend", () => {
  it("is an unchanged recoverable error", async () => {
    const r = await new WebSearchTool().execute(
      call({ query: "x" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
    expect(r.message).toBe("web_search backend not configured");
  });
});

// ============================================================================
// applyWebFetchRange unit tests (#135)
// ============================================================================

describe("applyWebFetchRange #135 — unit", () => {
  it("start_byte 0 returns body unchanged with no header", () => {
    const r = applyWebFetchRange("hello world", 0);
    expect(r.ok).toBe(true);
    if (!r.ok) throw new Error("unreachable");
    expect(r.value).toBe("hello world");
  });

  it("start_byte 0 on empty body returns empty string with no header", () => {
    const r = applyWebFetchRange("", 0);
    expect(r.ok).toBe(true);
    if (!r.ok) throw new Error("unreachable");
    expect(r.value).toBe("");
  });

  it("start_byte in the middle prepends header and slices body", () => {
    const r = applyWebFetchRange("hello world", 6);
    expect(r.ok).toBe(true);
    if (!r.ok) throw new Error("unreachable");
    expect(r.value).toBe("[starting at byte 6 of 11]\nworld");
  });

  it("start_byte at last byte returns last byte with header", () => {
    const r = applyWebFetchRange("hello", 4);
    expect(r.ok).toBe(true);
    if (!r.ok) throw new Error("unreachable");
    expect(r.value).toBe("[starting at byte 4 of 5]\no");
  });

  it("start_byte exactly at body length is a recoverable error", () => {
    const r = applyWebFetchRange("hello", 5);
    expect(r.ok).toBe(false);
    if (r.ok) throw new Error("unreachable");
    expect(r.error.kind).toBe("error");
    if (r.error.kind !== "error") throw new Error("unreachable");
    expect(r.error.recoverable).toBe(true);
    expect(r.error.message).toBe("start_byte 5 exceeds response length 5");
  });

  it("start_byte past end is a recoverable error", () => {
    const r = applyWebFetchRange("hello", 10);
    expect(r.ok).toBe(false);
    if (r.ok) throw new Error("unreachable");
    expect(r.error.kind).toBe("error");
    if (r.error.kind !== "error") throw new Error("unreachable");
    expect(r.error.recoverable).toBe(true);
    expect(r.error.message).toBe("start_byte 10 exceeds response length 5");
  });

  it("empty body with nonzero start_byte is a recoverable error", () => {
    const r = applyWebFetchRange("", 1);
    expect(r.ok).toBe(false);
    if (r.ok) throw new Error("unreachable");
    expect(r.error.kind).toBe("error");
    if (r.error.kind !== "error") throw new Error("unreachable");
    expect(r.error.recoverable).toBe(true);
    expect(r.error.message).toBe("start_byte 1 exceeds response length 0");
  });
});

// ============================================================================
// Fixture replay: web_fetch_range.json (#135)
// ============================================================================

describe("applyWebFetchRange #135 — fixture replay", () => {
  interface WebFetchRangeCase {
    name: string;
    body: string;
    start_byte: number;
    expected?: string;
    expected_error?: string;
  }

  it("replays all cases from fixtures/tools/web_fetch_range.json", async () => {
    const { readFileSync } = await import("node:fs");
    const { resolve, dirname } = await import("node:path");
    const { fileURLToPath } = await import("node:url");

    const __dirname = dirname(fileURLToPath(import.meta.url));
    const fixturePath = resolve(
      __dirname,
      "../../../../fixtures/tools/web_fetch_range.json",
    );
    const cases: WebFetchRangeCase[] = JSON.parse(
      readFileSync(fixturePath, "utf8"),
    );
    expect(cases.length).toBeGreaterThan(0);

    for (const c of cases) {
      const result = applyWebFetchRange(c.body, c.start_byte);
      if (c.expected !== undefined) {
        expect(result.ok).toBe(true);
        if (!result.ok) throw new Error(`case ${c.name}: expected ok`);
        expect(result.value).toBe(c.expected);
      } else if (c.expected_error !== undefined) {
        expect(result.ok).toBe(false);
        if (result.ok) throw new Error(`case ${c.name}: expected error`);
        expect(result.error.kind).toBe("error");
        if (result.error.kind !== "error")
          throw new Error(`case ${c.name}: unreachable`);
        expect(result.error.message).toBe(c.expected_error);
      } else {
        throw new Error(`case ${c.name}: missing expected or expected_error`);
      }
    }
  });
});

// ============================================================================
// WebFetchTool integration with start_byte via mock HTTP server (#135)
// ============================================================================

function webFetchCall(input: unknown): ToolCall {
  return { id: "c1", name: "web_fetch", input };
}

describe("WebFetchTool #135 — start_byte integration", () => {
  it("start_byte 0 returns body unchanged (no header)", async () => {
    const { url, close } = await startCapturingServer("hello world");
    try {
      const r = await new WebFetchTool().execute(
        webFetchCall({ url: `${url}/page`, start_byte: 0 }),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind).toBe("success");
      if (r.kind !== "success") throw new Error("unreachable");
      expect(r.content).toBe("hello world");
    } finally {
      await close();
    }
  });

  it("start_byte mid-body slices with header", async () => {
    const { url, close } = await startCapturingServer("hello world");
    try {
      const r = await new WebFetchTool().execute(
        webFetchCall({ url: `${url}/page`, start_byte: 6 }),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind).toBe("success");
      if (r.kind !== "success") throw new Error("unreachable");
      expect(r.content).toBe("[starting at byte 6 of 11]\nworld");
    } finally {
      await close();
    }
  });

  it("start_byte past end is a recoverable error", async () => {
    const { url, close } = await startCapturingServer("hello");
    try {
      const r = await new WebFetchTool().execute(
        webFetchCall({ url: `${url}/page`, start_byte: 99 }),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind).toBe("error");
      if (r.kind !== "error") throw new Error("unreachable");
      expect(r.recoverable).toBe(true);
      expect(r.message).toBe("start_byte 99 exceeds response length 5");
    } finally {
      await close();
    }
  });

  it("omitting start_byte defaults to 0 (byte-identical to old behavior)", async () => {
    const { url, close } = await startCapturingServer("body text here");
    try {
      const r = await new WebFetchTool().execute(
        webFetchCall({ url: `${url}/page` }),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind).toBe("success");
      if (r.kind !== "success") throw new Error("unreachable");
      expect(r.content).toBe("body text here");
    } finally {
      await close();
    }
  });
});

// ============================================================================
// SSRF guard: validateFetchUrl + opt-in UrlPolicy (#145)
// ============================================================================

describe("validateFetchUrl #145 — deny_private", () => {
  const policy = UrlPolicy.denyPrivate();
  const denied = [
    "http://169.254.169.254/latest/meta-data/",
    "http://localhost/",
    "file:///etc/passwd",
    "http://127.0.0.1/",
    "http://10.1.2.3/",
    "http://192.168.0.1/",
    "http://172.16.5.5/",
    "http://[::1]/",
    "ftp://example.com/x",
  ];
  for (const url of denied) {
    it(`denies ${url}`, () => {
      const r = validateFetchUrl(url, policy);
      expect(r.ok).toBe(false);
      if (r.ok) throw new Error("unreachable");
      expect(r.error.kind).toBe("error");
      if (r.error.kind !== "error") throw new Error("unreachable");
      expect(r.error.recoverable).toBe(true);
    });
  }

  const allowed = ["https://example.com/", "http://93.184.216.34/"];
  for (const url of allowed) {
    it(`allows ${url}`, () => {
      const r = validateFetchUrl(url, policy);
      expect(r.ok).toBe(true);
    });
  }

  it("denies an IPv4-mapped IPv6 metadata literal (::ffff:169.254.169.254)", () => {
    const r = validateFetchUrl("http://[::ffff:169.254.169.254]/", policy);
    expect(r.ok).toBe(false);
  });
});

describe("validateFetchUrl #145 — permissive (zero churn)", () => {
  const policy = UrlPolicy.permissive();
  const everything = [
    "http://169.254.169.254/latest/meta-data/",
    "http://localhost/",
    "file:///etc/passwd",
    "https://example.com/",
  ];
  for (const url of everything) {
    it(`allows ${url}`, () => {
      expect(validateFetchUrl(url, policy).ok).toBe(true);
    });
  }

  it("permissive is the default tool behavior (no parsing, allows anything)", () => {
    // A new tool with no .withUrlPolicy() call must not block anything: prove it
    // via the seam directly using a fresh permissive policy (the tool default).
    expect(
      validateFetchUrl("http://169.254.169.254/", UrlPolicy.permissive()).ok,
    ).toBe(true);
    // Even a totally unparseable string is allowed under permissive (no parse).
    expect(
      validateFetchUrl("not a url at all", UrlPolicy.permissive()).ok,
    ).toBe(true);
  });
});

describe("WebFetchTool #145 — opt-in deny_private blocks before any fetch", () => {
  it("returns a recoverable error for a metadata URL", async () => {
    const tool = new WebFetchTool().withUrlPolicy(UrlPolicy.denyPrivate());
    const r = await tool.execute(
      webFetchCall({ url: "http://169.254.169.254/" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it("never contacts the server when the (loopback) endpoint is denied", async () => {
    // The capturing server binds 127.0.0.1, which deny_private blocks; validate
    // that no request lands by racing the `captured` promise against a short
    // deadline. A hit would resolve `captured`; a clean block leaves it pending.
    const { url, captured, close } = await startCapturingServer("nope");
    try {
      const tool = new WebFetchTool().withUrlPolicy(UrlPolicy.denyPrivate());
      const r = await tool.execute(
        webFetchCall({ url: `${url}/page` }),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind).toBe("error");

      const sentinel = Symbol("never-hit");
      const raced = await Promise.race([
        captured.then(() => "hit" as const),
        new Promise<typeof sentinel>((res) =>
          setTimeout(() => res(sentinel), 50),
        ),
      ]);
      expect(raced).toBe(sentinel);
    } finally {
      await close();
    }
  });
});
