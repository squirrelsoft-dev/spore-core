/**
 * Cross-language fixture-replay for the configurable compaction window
 * (spore-core issue #141).
 *
 * Replays `fixtures/compaction_window/cases.json` — the SAME fixture the
 * Rust, Python, and Go suites consume. The four languages must produce
 * identical verdicts; never edit a fixture to make a failing implementation
 * pass (see `fixtures/README.md`).
 *
 * - `trigger_cases` — build a SessionState with the case's `window_limit` +
 *   `token_budget_used`, a CompactionConfig at the case's `threshold`, and
 *   assert `shouldCompact()` matches `expected_should_compact`. Proves
 *   `threshold × window_limit` respects the configured (often small) window.
 * - `resolver_cases` — stub a model whose `context_window` equals
 *   `model_context_window`, set `CompactionConfig.context_length` to
 *   `config_context_length` (`null` ⇒ unset), and assert
 *   `resolveContextLength()` equals `expected_resolved`. Exercises the
 *   config(>0) → model(>0) → 8000 fallback chain with no clamping.
 *
 * Mirrors `rust/.../tests` for issue #141.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { context, SessionId, TaskId } from "../src/index.js";
import type {
  ModelInterface,
  ModelRequest,
  ModelResponse,
  ProviderInfo,
  StreamEvent,
} from "../src/index.js";

const { StandardContextManager, NullCacheProvider, defaultCompactionConfig, newSessionState } =
  context;

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/compaction_window/cases.json");

interface TriggerCase {
  name: string;
  window_limit: number;
  token_budget_used: number;
  threshold: number;
  expected_should_compact: boolean;
}

interface ResolverCase {
  name: string;
  config_context_length: number | null;
  model_context_window: number;
  expected_resolved: number;
}

interface Cases {
  trigger_cases: TriggerCase[];
  resolver_cases: ResolverCase[];
}

const cases = JSON.parse(readFileSync(fixturePath, "utf8")) as Cases;

/** Minimal model whose `provider().context_window` is fixture-driven. */
class StubModel implements ModelInterface {
  constructor(private readonly contextWindow: number) {}
  async call(_req: ModelRequest): Promise<ModelResponse> {
    throw new Error("not used");
  }
  callStreaming(_req: ModelRequest): AsyncIterable<StreamEvent> {
    throw new Error("not used");
  }
  async countTokens(_req: ModelRequest): Promise<number> {
    return 0;
  }
  provider(): ProviderInfo {
    return { name: "stub", model_id: "stub", context_window: this.contextWindow };
  }
}

describe("compaction window fixture — trigger_cases (#141)", () => {
  for (const tc of cases.trigger_cases) {
    it(`${tc.name}: should_compact == ${tc.expected_should_compact}`, () => {
      const mgr = new StandardContextManager(new StubModel(200_000), new NullCacheProvider(), {
        ...defaultCompactionConfig(),
        threshold: tc.threshold,
      });
      const state = newSessionState(SessionId.of("s1"), TaskId.of("t1"), "task");
      state.window_limit = tc.window_limit;
      state.token_budget_used = tc.token_budget_used;
      expect(mgr.shouldCompact(state)).toBe(tc.expected_should_compact);
    });
  }
});

describe("compaction window fixture — resolver_cases (#141)", () => {
  for (const rc of cases.resolver_cases) {
    it(`${rc.name}: resolveContextLength() == ${rc.expected_resolved}`, () => {
      // null config_context_length ⇒ field left unset (absent).
      const mgr = new StandardContextManager(
        new StubModel(rc.model_context_window),
        new NullCacheProvider(),
        {
          ...defaultCompactionConfig(),
          context_length: rc.config_context_length,
        },
      );
      expect(mgr.resolveContextLength()).toBe(rc.expected_resolved);
    });
  }
});
