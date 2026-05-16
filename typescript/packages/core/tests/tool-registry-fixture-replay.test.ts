/**
 * Fixture-replay tests for the canonical ToolRegistry (spore-core issue #4).
 *
 * Loads `fixtures/tool_registry/dispatch_scenarios.json` and asserts the same
 * verdicts as the Rust, Python, and Go suites. If a fixture fails in one
 * language, fix the implementation — never edit the fixture.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { toolRegistry } from "../src/index.js";

const {
  StandardToolRegistry,
  toolRegistryMock: { EchoTool, AllowAllSandbox },
} = toolRegistry;
type ToolSchema = toolRegistry.ToolSchema;
type ToolSet = toolRegistry.ToolSet;

interface DispatchScenario {
  name: string;
  register: ToolSchema[];
  sets?: ToolSet[];
  call: { id: string; name: string; input: unknown };
  expected: { kind: "ok"; call_id: string } | { kind: "err"; error: string };
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/tool_registry/dispatch_scenarios.json");

describe("ToolRegistry fixture replay", () => {
  const data = readFileSync(fixturePath, "utf8");
  const scenarios = JSON.parse(data) as DispatchScenario[];

  it("loads at least one scenario", () => {
    expect(scenarios.length).toBeGreaterThan(0);
  });

  for (const sc of scenarios) {
    it(`scenario: ${sc.name}`, async () => {
      const reg = new StandardToolRegistry();
      for (const s of sc.register) {
        const err = reg.register(new EchoTool(s.name), s);
        expect(err, `register ${s.name}`).toBeNull();
      }
      for (const set of sc.sets ?? []) {
        const err = reg.registerSet(set);
        expect(err, `register_set ${set.name}`).toBeNull();
      }
      const result = await reg.dispatch(
        { id: sc.call.id, name: sc.call.name, input: sc.call.input },
        new AllowAllSandbox(),
      );
      if (sc.expected.kind === "ok") {
        expect(result.ok).toBe(true);
        if (!result.ok) throw new Error("unreachable");
        expect(result.result.call_id).toBe(sc.expected.call_id);
      } else {
        expect(result.ok).toBe(false);
        if (result.ok) throw new Error("unreachable");
        expect(result.error.kind).toBe(sc.expected.error);
      }
    });
  }
});
