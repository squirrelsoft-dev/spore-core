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

import { WebSearchConfigError, WebSearchTool } from "../src/index.js";

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
