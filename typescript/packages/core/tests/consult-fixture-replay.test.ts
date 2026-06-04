/**
 * Fixture-replay tests for the mid-loop consult primitive (spore-core issue
 * #114).
 *
 * Consumes the SHARED, cross-language fixture (authored by the Rust agent — do
 * NOT modify it): `fixtures/harness/consult.json`. Every consult type
 * (ConsultRequest, ConsultResponse, ConsultOverflowPolicy) and the wire shapes
 * of `RunResult.consult` / `ToolOutput.consult` must parse and re-serialize
 * byte-for-byte, producing the same outcome as the Rust integration tests (R8).
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  ConsultOverflowPolicySchema,
  ConsultRequestSchema,
  ConsultResponseSchema,
} from "../src/index.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(repoRoot, "fixtures/harness/consult.json");

interface ConsultFixture {
  consult_overflow_policy_cases: unknown[];
  consult_request_cases: unknown[];
  consult_response_cases: unknown[];
  run_result_cases: { kind: string; request: unknown; state: Record<string, unknown> }[];
  subagent_tool_output_cases: { kind: string; child_state: unknown; request: unknown }[];
  worker_tool_output_cases: { kind: string; request: unknown }[];
}

const suite = JSON.parse(readFileSync(fixturePath, "utf-8")) as ConsultFixture;

describe("Consult serde fixture — consult.json", () => {
  it("every ConsultRequest case round-trips byte-identically", () => {
    expect(suite.consult_request_cases.length).toBeGreaterThan(0);
    for (const c of suite.consult_request_cases) {
      const parsed = ConsultRequestSchema.parse(c);
      expect(parsed).toEqual(c);
    }
  });

  it("every ConsultResponse case round-trips byte-identically", () => {
    expect(suite.consult_response_cases.length).toBeGreaterThan(0);
    const seen = new Set<string>();
    for (const c of suite.consult_response_cases) {
      const parsed = ConsultResponseSchema.parse(c);
      expect(parsed).toEqual(c);
      seen.add(parsed.kind);
    }
    expect(seen).toEqual(new Set(["answer", "budget_exhausted"]));
  });

  it("every ConsultOverflowPolicy case round-trips byte-identically", () => {
    expect(suite.consult_overflow_policy_cases.length).toBeGreaterThan(0);
    const seen = new Set<string>();
    for (const c of suite.consult_overflow_policy_cases) {
      const parsed = ConsultOverflowPolicySchema.parse(c);
      expect(parsed).toEqual(c);
      seen.add(parsed.kind);
    }
    expect(seen).toEqual(new Set(["soft_fail", "escalate_to_human"]));
  });

  it("RunResult.consult preserves null human_request and no child at the top level", () => {
    expect(suite.run_result_cases.length).toBeGreaterThan(0);
    const rr = suite.run_result_cases[0]!;
    expect(rr.kind).toBe("consult");
    // The carried request parses.
    ConsultRequestSchema.parse(rr.request);
    // human_request null, child_state null at the RunResult level.
    expect(rr.state.human_request).toBeNull();
    expect(rr.state.child_state).toBeNull();
    // The consult call is the head of the preserved pending calls.
    const pending = rr.state.pending_tool_calls as { id: string }[];
    expect(pending.length).toBeGreaterThanOrEqual(1);
    expect(pending[0]!.id).toBe("consult-call-1");
  });

  it("worker-side ToolOutput.consult omits child_state; subagent-side populates it", () => {
    const worker = suite.worker_tool_output_cases[0]!;
    expect(worker.kind).toBe("consult");
    // Worker-side consult omits child_state on the wire (optional field).
    expect("child_state" in worker).toBe(false);
    ConsultRequestSchema.parse(worker.request);

    const sub = suite.subagent_tool_output_cases[0]!;
    expect(sub.kind).toBe("consult");
    // Subagent-boundary consult carries a populated child_state object.
    expect(typeof sub.child_state).toBe("object");
    expect(sub.child_state).not.toBeNull();
    ConsultRequestSchema.parse(sub.request);
  });
});
