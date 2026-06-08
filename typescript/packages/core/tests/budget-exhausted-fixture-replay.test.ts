/**
 * Fixture-replay test for `HumanRequest::BudgetExhausted` (spore-core issue
 * #130). Consumes the SHARED, cross-language ground-truth fixture
 * `fixtures/paused_states/budget_exhausted.json` (do NOT modify it):
 *   1. Deserialize the paused state's `human_request` via the zod schema and
 *      assert it FIELD-FOR-FIELD.
 *   2. Re-serialize byte-identically (parse → stringify deep-equals the fixture
 *      sub-object — the wire shape is byte-identical across the four languages).
 *
 * Must produce the same outcome as the Rust integration test
 * (`budget_exhausted_paused_state_fixture_replay`). Never edit the fixture to
 * make a failing implementation pass.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  BudgetPolicySchema,
  EscalationActionSchema,
  HumanRequestSchema,
  TaskSchema,
  type HumanRequest,
} from "../src/index.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(repoRoot, "fixtures/paused_states/budget_exhausted.json");

interface RawPaused {
  session_id: string;
  task_id: string;
  turn_number: number;
  session_state: { messages: unknown[]; extras: Record<string, unknown> };
  pending_tool_calls: unknown[];
  approved_results: unknown[];
  human_request: Record<string, unknown>;
  task: unknown;
  budget_used: Record<string, unknown>;
  child_state: unknown;
}

describe("budget_exhausted.json fixture replay", () => {
  const raw = JSON.parse(readFileSync(fixturePath, "utf-8")) as RawPaused;

  it("the top-level paused-state envelope matches", () => {
    expect(raw.session_id).toBe("sess-130");
    expect(raw.task_id).toBe("task-130");
    expect(raw.turn_number).toBe(6);
    expect(raw.pending_tool_calls).toEqual([]);
    expect(raw.approved_results).toEqual([]);
    expect(raw.child_state).toBeNull();
    expect(raw.budget_used).toMatchObject({ turns: 6, input_tokens: 0, output_tokens: 0 });
  });

  it("the human_request deserializes field-for-field as budget_exhausted", () => {
    const req: HumanRequest = HumanRequestSchema.parse(raw.human_request);
    expect(req.kind).toBe("budget_exhausted");
    if (req.kind !== "budget_exhausted") throw new Error("not budget_exhausted");
    expect(req.phase).toBe("plan_execute");
    expect(req.policy).toEqual({ kind: "total_steps", value: 6 });
    expect(req.steps_taken).toBe(6);
    expect(req.continues_used).toBe(1);
    expect(req.partial_output).toBe('{"node":"plan_execute","tasks":2,"ledger":[]}');
    expect(req.available_actions).toEqual([
      { kind: "continue_with_budget", steps: 6 },
      { kind: "skip" },
      { kind: "fail" },
    ]);
  });

  it("each available_action parses via the EscalationAction schema", () => {
    const actions = raw.human_request.available_actions as unknown[];
    expect(actions).toHaveLength(3);
    for (const a of actions) EscalationActionSchema.parse(a);
  });

  it("the embedded policy parses via the BudgetPolicy schema", () => {
    expect(BudgetPolicySchema.parse(raw.human_request.policy)).toEqual({
      kind: "total_steps",
      value: 6,
    });
  });

  it("the embedded task parses via the Task schema (PlanExecute tree)", () => {
    const task = TaskSchema.parse(raw.task);
    expect(task.id.asString()).toBe("task-130");
    expect(task.loop_strategy.kind).toBe("plan_execute");
  });

  it("the human_request re-serializes byte-identically (deep-equal the fixture)", () => {
    const req = HumanRequestSchema.parse(raw.human_request);
    // The parsed value re-stringifies to a structurally-identical wire object —
    // the #130 byte-identical cross-language guarantee.
    expect(JSON.parse(JSON.stringify(req))).toEqual(raw.human_request);
  });
});
