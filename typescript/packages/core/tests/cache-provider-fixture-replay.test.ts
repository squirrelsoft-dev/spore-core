/**
 * Cross-language fixture replay for {@link CacheProvider.parseCacheStats}
 * (spore-core issue #25). Loads the shared JSON fixture and asserts the same
 * outcomes the Rust suite asserts.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { cacheProvider } from "../src/index.js";
import type { ModelResponse } from "../src/index.js";

const { AnthropicCacheProvider, NullCacheProvider, OllamaCacheProvider, OpenAICacheProvider } =
  cacheProvider;

interface FixtureUsage {
  cache_read_tokens: number | null;
  cache_write_tokens: number | null;
}
interface FixtureExpected {
  is_some: boolean;
  cache_read_tokens?: number;
  cache_write_tokens?: number;
}
interface FixtureCase {
  name: string;
  provider: string;
  usage: FixtureUsage;
  expected: FixtureExpected;
}
interface FixtureFile {
  description?: string;
  cases: FixtureCase[];
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/cache_provider/parse_cache_stats.json");

function makeResponse(read: number | null, write: number | null): ModelResponse {
  return {
    content: [{ type: "text", text: "hi" }],
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      cache_read_tokens: read,
      cache_write_tokens: write,
    },
    stop_reason: "end_turn",
  };
}

function providerFor(name: string): cacheProvider.CacheProvider {
  switch (name) {
    case "anthropic":
      return new AnthropicCacheProvider();
    case "openai":
      return new OpenAICacheProvider();
    case "ollama":
      return new OllamaCacheProvider();
    case "null":
      return new NullCacheProvider();
    default:
      throw new Error(`unknown provider in fixture: ${name}`);
  }
}

describe("CacheProvider.parseCacheStats fixture replay", () => {
  const text = readFileSync(fixturePath, "utf8");
  const file = JSON.parse(text) as FixtureFile;

  for (const c of file.cases) {
    it(c.name, () => {
      const resp = makeResponse(c.usage.cache_read_tokens, c.usage.cache_write_tokens);
      const stats = providerFor(c.provider).parseCacheStats(resp);
      if (c.expected.is_some) {
        expect(stats).not.toBeNull();
        expect(stats!.cache_read_tokens).toBe(c.expected.cache_read_tokens ?? 0);
        expect(stats!.cache_write_tokens).toBe(c.expected.cache_write_tokens ?? 0);
      } else {
        expect(stats).toBeNull();
      }
    });
  }
});
