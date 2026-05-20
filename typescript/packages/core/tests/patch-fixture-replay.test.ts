/**
 * Fixture-replay tests for PatchToolCallsMiddleware observability
 * (spore-core issue #28).
 *
 * Loads `fixtures/patch/patch_events_basic.json`, runs each input call through
 * the middleware, and asserts the emitted patch events plus rolled-up metrics
 * match the fixture's expectations byte-for-byte with the Rust, Python, and Go
 * suites.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { middleware, observability, SessionId } from "../src/index.js";
import { newTask } from "../src/harness/types.js";
import type { ToolCall } from "../src/model/schemas.js";

const { StandardMiddlewareChain, PatchToolCallsMiddleware } = middleware;
const { InMemoryObservabilityProvider } = observability;

interface FixtureCall {
  id: string;
  name: string;
  input: unknown;
}

interface ExpectedPatch {
  call_id: string;
  tool_name: string;
  patch_type: string;
  original: unknown;
  patched: unknown;
}

interface PatchFixture {
  fallback_name: string;
  input_calls: FixtureCall[];
  expected_patches: ExpectedPatch[];
  expected_patch_count: number;
  expected_patches_by_tool: Record<string, number>;
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/patch/patch_events_basic.json");

describe("PatchToolCallsMiddleware fixture replay", () => {
  it("patch_events_basic matches the cross-language expected outcome", async () => {
    const raw = readFileSync(fixturePath, "utf8");
    const fix = JSON.parse(raw) as PatchFixture;

    const sid = SessionId.of("sess");
    const obs = new InMemoryObservabilityProvider();
    const mw = new PatchToolCallsMiddleware(fix.fallback_name, obs);

    // Drive identity capture, then the before_tool fire.
    const task = newTask("test task", sid, { kind: "re_act", max_iterations: 5 });
    await mw.handle({ kind: "before_session", task, session_id: sid });
    const calls: ToolCall[] = fix.input_calls.map((c) => ({
      id: c.id,
      name: c.name,
      input: c.input,
    }));
    await mw.handle({ kind: "before_tool", calls, turn_number: 1 });

    // Each expected patch event was recorded.
    const patches = obs.patchSpans(sid);
    expect(patches.length).toBe(fix.expected_patches.length);
    for (const exp of fix.expected_patches) {
      const found = patches.find((p) => p.call_id === exp.call_id);
      expect(found, `no patch span for call ${exp.call_id}`).toBeDefined();
      expect(found!.tool_name).toBe(exp.tool_name);
      expect(found!.original_parameters).toEqual(exp.original);
      expect(found!.patched_parameters).toEqual(exp.patched);
      expect(found!.patch_type.kind).toBe(exp.patch_type);
    }

    // Record an outcome so SessionMetrics materializes for a session that
    // emitted only patch spans (no turns). No tool-call spans were emitted in
    // this middleware-only replay, so patch_rate is 0 by the divide-by-zero
    // guard; the fixture asserts patch_count and patches_by_tool.
    obs.setSessionOutcome(sid, { kind: "success" });
    const m = await obs.getSessionMetrics(sid);
    expect(m!.patch_count).toBe(fix.expected_patch_count);
    expect(m!.patch_rate).toBe(0);
    for (const [tool, n] of Object.entries(fix.expected_patches_by_tool)) {
      expect(m!.patches_by_tool[tool]).toBe(n);
    }
  });

  it("the chain dispatches patch observability end-to-end", async () => {
    const raw = readFileSync(fixturePath, "utf8");
    const fix = JSON.parse(raw) as PatchFixture;

    const sid = SessionId.of("sess");
    const obs = new InMemoryObservabilityProvider();
    const chain = new StandardMiddlewareChain();
    chain.register(new PatchToolCallsMiddleware(fix.fallback_name, obs));

    const task = newTask("test task", sid, { kind: "re_act", max_iterations: 5 });
    await chain.fireBeforeSession(task, sid);
    const calls: ToolCall[] = fix.input_calls.map((c) => ({
      id: c.id,
      name: c.name,
      input: c.input,
    }));
    const d = await chain.fireBeforeTool(calls, 1);
    expect(d.kind).toBe("continue_with_modification");
    expect(obs.patchSpans(sid).length).toBe(fix.expected_patch_count);
  });
});
