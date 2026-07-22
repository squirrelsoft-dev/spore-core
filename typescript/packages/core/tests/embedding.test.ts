import { describe, expect, it } from "vitest";

import { ProviderError, Timeout } from "../src/model/errors.js";
import { cosineSimilarity } from "../src/embedding/types.js";
import { OllamaEmbeddingModel, OpenAIEmbeddingModel } from "../src/embedding/standard.js";

/** A fetch impl that returns scripted responses per request, recording calls. */
function scriptedFetch(
  responses: Array<{ status?: number; body: unknown; headers?: Record<string, string> }>,
): { fetch: (url: string, init: RequestInit) => Promise<Response>; calls: Array<{ url: string; init: RequestInit }> } {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  let i = 0;
  const f = async (url: string, init: RequestInit): Promise<Response> => {
    calls.push({ url, init });
    const r = responses[i] ?? responses[responses.length - 1]!;
    i += 1;
    return new Response(JSON.stringify(r.body), {
      status: r.status ?? 200,
      headers: r.headers,
    });
  };
  return { fetch: f, calls };
}

describe("cosineSimilarity", () => {
  it("identical vectors → 1.0, orthogonal → 0.0", () => {
    expect(cosineSimilarity([1, 0, 0], [1, 0, 0])).toBeCloseTo(1.0, 6);
    expect(cosineSimilarity([1, 0, 0], [0, 1, 0])).toBeCloseTo(0.0, 6);
  });
  it("degenerate inputs → 0.0, never NaN", () => {
    expect(cosineSimilarity([1, 0, 0], [0, 0, 0])).toBe(0);
    expect(cosineSimilarity([1, 0], [1, 2, 3])).toBe(0);
  });
});

describe("OllamaEmbeddingModel", () => {
  it("round-trips /api/embed in input order", async () => {
    const { fetch, calls } = scriptedFetch([
      { body: { embeddings: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] } },
    ]);
    const model = new OllamaEmbeddingModel("nomic-embed-text", { fetchImpl: fetch });
    const vectors = await model.embed(["hello", "world"]);
    expect(vectors).toEqual([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]);
    expect(calls[0]!.url).toBe("http://localhost:11434/api/embed");
  });

  it("empty input skips the call", async () => {
    // No mock mounted: an unreachable base URL; empty input must not contact it.
    const fetch = async (): Promise<Response> => {
      throw new Error("should not be called");
    };
    const model = new OllamaEmbeddingModel("nomic-embed-text", {
      fetchImpl: fetch,
      baseUrl: "http://127.0.0.1:1",
    });
    expect(await model.embed([])).toEqual([]);
  });

  it("connect error → 'Ollama not running at <base>'", async () => {
    const fetch = async (): Promise<Response> => {
      const e = new Error("ECONNREFUSED");
      e.name = "Error";
      throw e;
    };
    const model = new OllamaEmbeddingModel("nomic-embed-text", {
      fetchImpl: fetch,
      baseUrl: "http://localhost:11434",
    });
    const err = await model.embed(["x"]).catch((e) => e);
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).providerMessage).toBe("Ollama not running at http://localhost:11434");
  });

  it("non-2xx → ProviderError(code, body[:500])", async () => {
    const { fetch } = scriptedFetch([{ status: 500, body: { error: "boom" } }]);
    const model = new OllamaEmbeddingModel("nomic-embed-text", { fetchImpl: fetch });
    const err = await model.embed(["x"]).catch((e) => e);
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).code).toBe(500);
  });

  it("probeDimensions returns the vector length", async () => {
    const { fetch } = scriptedFetch([{ body: { embeddings: [[0.1, 0.2, 0.3, 0.4]] } }]);
    const model = new OllamaEmbeddingModel("nomic-embed-text", { fetchImpl: fetch });
    expect(await model.probeDimensions()).toBe(4);
  });

  it("embedOne throws if no vectors returned", async () => {
    const { fetch } = scriptedFetch([{ body: { embeddings: [] } }]);
    const model = new OllamaEmbeddingModel("nomic-embed-text", { fetchImpl: fetch });
    const err = await model.embedOne("x").catch((e) => e);
    expect(err).toBeInstanceOf(ProviderError);
  });

  it("info uses Rust wire field model_id", () => {
    const model = new OllamaEmbeddingModel("nomic-embed-text", { dimensions: 768 });
    expect(model.info()).toEqual({ provider: "ollama", model_id: "nomic-embed-text", dimensions: 768 });
  });

  it("withTimeout aborts a never-responding fetch and maps to Timeout", async () => {
    const fetch = async (_url: string, init: RequestInit): Promise<Response> =>
      new Promise<Response>((_resolve, reject) => {
        const sig = init.signal as AbortSignal;
        sig.addEventListener("abort", () => reject(sig.reason), { once: true });
      });
    const model = new OllamaEmbeddingModel("nomic-embed-text", {
      fetchImpl: fetch,
      timeoutMs: 1,
    });
    const err = await model.embed(["x"]).catch((e) => e);
    expect(err).toBeInstanceOf(Timeout);
  });
});

describe("OpenAIEmbeddingModel", () => {
  it("authenticates with Bearer key and orders by index", async () => {
    const calls: Array<{ url: string; init: RequestInit }> = [];
    const fetch = async (url: string, init: RequestInit): Promise<Response> => {
      calls.push({ url, init });
      return new Response(
        JSON.stringify({
          data: [
            { embedding: [0.9, 0.9], index: 1 },
            { embedding: [0.1, 0.1], index: 0 },
          ],
        }),
        { status: 200 },
      );
    };
    const model = new OpenAIEmbeddingModel("test-key", "text-embedding-3-small", {
      fetchImpl: fetch,
    });
    const vectors = await model.embed(["a", "b"]);
    // Reordered by `index`, so input "a" (index 0) maps to [0.1, 0.1].
    expect(vectors).toEqual([[0.1, 0.1], [0.9, 0.9]]);
    expect(calls[0]!.url).toBe("https://api.openai.com/v1/embeddings");
    const headers = new Headers(calls[0]!.init.headers as Record<string, string>);
    expect(headers.get("authorization")).toBe("Bearer test-key");
  });

  it("default dimensions for text-embedding-3-small = 1536 and sends dimensions param", async () => {
    const calls: Array<{ init: RequestInit }> = [];
    const fetch = async (_url: string, init: RequestInit): Promise<Response> => {
      calls.push({ init });
      return new Response(JSON.stringify({ data: [{ embedding: [0.1], index: 0 }] }), {
        status: 200,
      });
    };
    const model = new OpenAIEmbeddingModel("k", "text-embedding-3-small", {
      fetchImpl: fetch,
    });
    expect(model.dimensions()).toBe(1536);
    await model.embed(["x"]);
    const body = JSON.parse(calls[0]!.init.body as string) as { dimensions?: number };
    expect(body.dimensions).toBe(1536);
  });

  it("ada-002 does NOT send the dimensions param", async () => {
    const calls: Array<{ init: RequestInit }> = [];
    const fetch = async (_url: string, init: RequestInit): Promise<Response> => {
      calls.push({ init });
      return new Response(JSON.stringify({ data: [{ embedding: [0.1], index: 0 }] }), {
        status: 200,
      });
    };
    const model = new OpenAIEmbeddingModel("k", "text-embedding-ada-002", {
      fetchImpl: fetch,
    });
    expect(model.dimensions()).toBe(1536);
    await model.embed(["x"]);
    const body = JSON.parse(calls[0]!.init.body as string) as { dimensions?: number };
    expect(body.dimensions).toBeUndefined();
  });

  it("400 {error:{message}} → ProviderError(400, msg) with maxRetries=0", async () => {
    const { fetch } = scriptedFetch([
      { status: 400, body: { error: { message: "bad model" } } },
    ]);
    const model = new OpenAIEmbeddingModel("k", "text-embedding-3-small", {
      fetchImpl: fetch,
      maxRetries: 0,
    });
    const err = await model.embed(["x"]).catch((e) => e);
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).code).toBe(400);
    expect((err as ProviderError).providerMessage).toBe("bad model");
  });

  it("429 then 200 succeeds on attempt 2 (no-sleep hook)", async () => {
    let i = 0;
    const fetch = async (): Promise<Response> => {
      i += 1;
      if (i === 1) return new Response("{}", { status: 429 });
      return new Response(JSON.stringify({ data: [{ embedding: [0.1], index: 0 }] }), {
        status: 200,
      });
    };
    const model = new OpenAIEmbeddingModel("k", "text-embedding-3-small", {
      fetchImpl: fetch,
      maxRetries: 1,
      sleep: async () => undefined,
    });
    const vectors = await model.embed(["x"]);
    expect(vectors).toEqual([[0.1]]);
    expect(i).toBe(2);
  });

  it("retry-after header is honored over computed backoff", async () => {
    let i = 0;
    let slept = 0;
    const fetch = async (): Promise<Response> => {
      i += 1;
      if (i === 1) return new Response("{}", { status: 503, headers: { "retry-after": "7" } });
      return new Response(JSON.stringify({ data: [{ embedding: [0.1], index: 0 }] }), {
        status: 200,
      });
    };
    const model = new OpenAIEmbeddingModel("k", "text-embedding-3-small", {
      fetchImpl: fetch,
      maxRetries: 1,
      sleep: async (ms) => {
        slept = ms;
      },
    });
    await model.embed(["x"]);
    expect(slept).toBe(7000);
  });

  it("default retry backoff sleep observes AbortSignal", async () => {
    let i = 0;
    const controller = new AbortController();
    const fetch = async (): Promise<Response> => {
      i += 1;
      return new Response("{}", { status: 429 });
    };
    const model = new OpenAIEmbeddingModel("k", "text-embedding-3-small", {
      fetchImpl: fetch,
      maxRetries: 1,
    });
    setTimeout(() => controller.abort(new Error("stop-now")), 20);
    const started = Date.now();
    const err = await model.embed(["x"], controller.signal).catch((e) => e);
    const elapsed = Date.now() - started;
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).providerMessage).toContain("request aborted");
    expect(elapsed).toBeLessThan(250);
    expect(i).toBe(1);
  });

  it("custom retry backoff sleep observes AbortSignal", async () => {
    let i = 0;
    const controller = new AbortController();
    const fetch = async (): Promise<Response> => {
      i += 1;
      return new Response("{}", { status: 429 });
    };
    const model = new OpenAIEmbeddingModel("k", "text-embedding-3-small", {
      fetchImpl: fetch,
      maxRetries: 1,
      sleep: async () => {
        controller.abort(new Error("stop"));
      },
    });
    const err = await model.embed(["x"], controller.signal).catch((e) => e);
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).providerMessage).toContain("request aborted");
    expect(i).toBe(1);
  });

  it("info uses Rust wire field model_id", () => {
    const model = new OpenAIEmbeddingModel("k", "text-embedding-3-small");
    expect(model.info()).toEqual({ provider: "openai", model_id: "text-embedding-3-small", dimensions: 1536 });
  });

  it("fromEnv rejects unset/empty env var", () => {
    const prev = process.env["SPORE_TEST_KEY"];
    delete process.env["SPORE_TEST_KEY"];
    expect(() => OpenAIEmbeddingModel.fromEnv("SPORE_TEST_KEY", "text-embedding-3-small")).toThrow(
      ProviderError,
    );
    process.env["SPORE_TEST_KEY"] = "   ";
    expect(() => OpenAIEmbeddingModel.fromEnv("SPORE_TEST_KEY", "text-embedding-3-small")).toThrow(
      ProviderError,
    );
    if (prev != null) process.env["SPORE_TEST_KEY"] = prev;
    else delete process.env["SPORE_TEST_KEY"];
  });

  it("empty input skips the call", async () => {
    const fetch = async (): Promise<Response> => {
      throw new Error("should not be called");
    };
    const model = new OpenAIEmbeddingModel("k", "text-embedding-3-small", {
      fetchImpl: fetch,
    });
    expect(await model.embed([])).toEqual([]);
  });
});