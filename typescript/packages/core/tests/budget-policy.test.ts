/**
 * BudgetPolicy + BudgetExhaustedBehavior (spore-core issue #117).
 *
 * Pure serializable budget vocabulary types. These tests assert:
 *   - every variant round-trips through parse/serialize,
 *   - the nested Continue→Continue→Fail behavior round-trips,
 *   - exact serialized bytes match the cross-language wire format,
 *   - unknown/missing `kind` is rejected (no silent `continue`),
 *   - byte-identity round-trip against the shared fixture.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  budgetExhaustedBehaviorFromJson,
  budgetExhaustedBehaviorToJson,
  budgetPolicyFromJson,
  budgetPolicyToJson,
  type BudgetExhaustedBehavior,
  type BudgetPolicy,
} from "../src/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/budget_policy/cases.json");

describe("BudgetPolicy", () => {
  const variants: BudgetPolicy[] = [
    { kind: "unlimited" },
    { kind: "total_steps", value: 100 },
    { kind: "per_loop", value: 10 },
    { kind: "per_attempt", value: 3 },
  ];

  it.each(variants)("round-trips %j", (policy) => {
    const json = JSON.stringify(budgetPolicyToJson(policy));
    const back = budgetPolicyFromJson(JSON.parse(json));
    expect(back).toEqual(policy);
  });

  it("serializes to exact bytes", () => {
    expect(JSON.stringify(budgetPolicyToJson({ kind: "unlimited" }))).toBe('{"kind":"unlimited"}');
    expect(JSON.stringify(budgetPolicyToJson({ kind: "total_steps", value: 100 }))).toBe(
      '{"kind":"total_steps","value":100}',
    );
    expect(JSON.stringify(budgetPolicyToJson({ kind: "per_loop", value: 10 }))).toBe(
      '{"kind":"per_loop","value":10}',
    );
    expect(JSON.stringify(budgetPolicyToJson({ kind: "per_attempt", value: 3 }))).toBe(
      '{"kind":"per_attempt","value":3}',
    );
  });

  it("rejects an unknown kind", () => {
    expect(() => budgetPolicyFromJson({ kind: "per_goal", value: 1 })).toThrow();
  });

  it("rejects a missing kind", () => {
    expect(() => budgetPolicyFromJson({ value: 1 })).toThrow();
  });

  it("rejects a non-integer value", () => {
    expect(() => budgetPolicyFromJson({ kind: "total_steps", value: 1.5 })).toThrow();
  });
});

describe("BudgetExhaustedBehavior", () => {
  const nested: BudgetExhaustedBehavior = {
    kind: "continue",
    max_continues: 1,
    on_exhausted: {
      kind: "continue",
      max_continues: 2,
      on_exhausted: { kind: "fail" },
    },
  };

  const variants: BudgetExhaustedBehavior[] = [
    { kind: "escalate" },
    { kind: "fail" },
    { kind: "continue", max_continues: 2, on_exhausted: { kind: "fail" } },
    nested,
  ];

  it.each(variants)("round-trips %j", (behavior) => {
    const json = JSON.stringify(budgetExhaustedBehaviorToJson(behavior));
    const back = budgetExhaustedBehaviorFromJson(JSON.parse(json));
    expect(back).toEqual(behavior);
  });

  it("round-trips the nested Continue→Continue→Fail case", () => {
    const json = JSON.stringify(budgetExhaustedBehaviorToJson(nested));
    expect(json).toBe(
      '{"kind":"continue","max_continues":1,"on_exhausted":' +
        '{"kind":"continue","max_continues":2,"on_exhausted":{"kind":"fail"}}}',
    );
    expect(budgetExhaustedBehaviorFromJson(JSON.parse(json))).toEqual(nested);
  });

  it("serializes to exact bytes", () => {
    expect(JSON.stringify(budgetExhaustedBehaviorToJson({ kind: "escalate" }))).toBe(
      '{"kind":"escalate"}',
    );
    expect(JSON.stringify(budgetExhaustedBehaviorToJson({ kind: "fail" }))).toBe('{"kind":"fail"}');
    expect(
      JSON.stringify(
        budgetExhaustedBehaviorToJson({
          kind: "continue",
          max_continues: 2,
          on_exhausted: { kind: "fail" },
        }),
      ),
    ).toBe('{"kind":"continue","max_continues":2,"on_exhausted":{"kind":"fail"}}');
  });

  it("rejects an unknown kind (no silent continue)", () => {
    expect(() => budgetExhaustedBehaviorFromJson({ kind: "retry" })).toThrow();
  });

  it("rejects a missing kind (no silent continue)", () => {
    expect(() => budgetExhaustedBehaviorFromJson({ max_continues: 1 })).toThrow();
  });

  it("rejects continue without max_continues (no default)", () => {
    expect(() =>
      budgetExhaustedBehaviorFromJson({ kind: "continue", on_exhausted: { kind: "fail" } }),
    ).toThrow();
  });

  it("rejects continue with an unknown nested on_exhausted kind", () => {
    expect(() =>
      budgetExhaustedBehaviorFromJson({
        kind: "continue",
        max_continues: 1,
        on_exhausted: { kind: "retry" },
      }),
    ).toThrow();
  });
});

interface FixtureFile {
  policies: unknown[];
  behaviors: unknown[];
}

describe("budget_policy fixture replay — byte identity", () => {
  const suite = JSON.parse(readFileSync(fixturePath, "utf-8")) as FixtureFile;

  it.each(suite.policies)("policy %j round-trips byte-identically", (raw) => {
    const policy = budgetPolicyFromJson(raw);
    expect(budgetPolicyToJson(policy)).toEqual(raw);
    expect(JSON.stringify(budgetPolicyToJson(policy))).toBe(JSON.stringify(raw));
  });

  it.each(suite.behaviors)("behavior %j round-trips byte-identically", (raw) => {
    const behavior = budgetExhaustedBehaviorFromJson(raw);
    expect(budgetExhaustedBehaviorToJson(behavior)).toEqual(raw);
    expect(JSON.stringify(budgetExhaustedBehaviorToJson(behavior))).toBe(JSON.stringify(raw));
  });
});
