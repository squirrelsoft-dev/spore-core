/**
 * HTTP tool tests — uses an ephemeral `http.createServer` instance so we
 * don't pull in an HTTP mocking dependency.
 */

import { createServer, type Server } from "node:http";
import { AddressInfo } from "node:net";

import { harnessTesting, toolRegistry, type ToolCall } from "@spore/core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { HttpGetTool, HttpPostTool } from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;
// Storage seam (#75): these tools ignore ctx, but the signature requires one.
const ctx = toolRegistry.toolRegistryMock.testCtx();

let server: Server;
let baseUrl: string;

beforeAll(async () => {
  server = createServer((req, res) => {
    if (req.method === "GET" && req.url === "/hello") {
      res.writeHead(200, { "content-type": "text/plain" });
      res.end("world");
      return;
    }
    if (req.method === "POST" && req.url === "/echo") {
      let body = "";
      req.on("data", (chunk: Buffer) => {
        body += chunk.toString("utf8");
      });
      req.on("end", () => {
        res.writeHead(200, { "content-type": "text/plain" });
        res.end(`ok:${body}`);
      });
      return;
    }
    res.writeHead(404);
    res.end();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address() as AddressInfo;
  baseUrl = `http://127.0.0.1:${addr.port}`;
});

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

function call(name: string, input: unknown): ToolCall {
  return { id: "c1", name, input };
}

describe("HttpGetTool", () => {
  it("returns body", async () => {
    const sb = new AllowAllSandbox();
    const r = await new HttpGetTool().execute(
      call("http_get", { url: `${baseUrl}/hello` }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toBe("world");
  });

  it("invalid url is recoverable", async () => {
    const sb = new AllowAllSandbox();
    const r = await new HttpGetTool().execute(
      call("http_get", { url: "not-a-url://////" }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });
});

describe("HttpPostTool", () => {
  it("sends JSON body", async () => {
    const sb = new AllowAllSandbox();
    const r = await new HttpPostTool().execute(
      call("http_post", { url: `${baseUrl}/echo`, body: { x: 1 } }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toContain("ok:");
    expect(r.content).toContain('"x":1');
  });
});
