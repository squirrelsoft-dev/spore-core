/**
 * SC-8 (#157) — `HarnessBuilder` preset tests: `codingAgent` + `hillClimber`.
 *
 * Ports the three Rust regression tests in
 * `rust/crates/spore-core/src/harness.rs` (commit `6f39933`):
 *   - `coding_agent_preset_wires_sandbox_tools_prompt_and_autocontinue`
 *   - `coding_agent_preset_errors_on_missing_workspace`
 *   - `hill_climber_preset_registers_evaluator_and_autocontinue`
 *
 * The presets live in `@spore/tools` (not on `HarnessBuilder` in `@spore/core`)
 * because the concrete coding tool catalogue lives in `@spore/tools`, which
 * depends on `@spore/core`; see `packages/tools/src/presets.ts`.
 */

import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  metric,
  MockModelInterface,
  type MetricEvaluator,
  type MetricOutcome,
  type OptimizationDirection,
  type ProviderInfo,
  type SandboxProvider,
  type SessionStateSnapshot,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import {
  codingAgent,
  hillClimber,
  CODING_AGENT_SYSTEM_PROMPT,
  PRESET_MAX_AUTO_GRANTS,
  PRESET_STEPS_PER_GRANT,
} from "../src/index.js";

function presetMockModel(): MockModelInterface {
  const provider: ProviderInfo = {
    name: "test",
    model_id: "test-1",
    context_window: 8192,
  };
  return new MockModelInterface(provider);
}

/** Minimal `MetricEvaluator` test double (mirrors the Rust `LatencyEvaluator`
 *  stand-in's role — only its registration is asserted, never `evaluate`). */
class StubMetricEvaluator implements MetricEvaluator {
  async evaluate(
    _sandbox: SandboxProvider,
    _snapshot: SessionStateSnapshot,
    _signal?: AbortSignal,
  ): Promise<MetricOutcome> {
    return { kind: "ok", result: metric.newMetricResult(1) };
  }
  direction(): OptimizationDirection {
    return "maximize";
  }
  description(): string {
    return "stub";
  }
}

// SC-8: `codingAgent` wires the read-write workspace sandbox, the coding tool
// catalogue, the built-in coding system prompt, and auto_continue (the looper
// preset) — so a consumer collapses to one call.
describe("codingAgent preset", () => {
  it("wires sandbox, tools, prompt, and auto_continue", async () => {
    const tmp = await mkdtemp(join(tmpdir(), "spore-sc8-coding-"));
    const cfg = codingAgent(presetMockModel(), tmp).buildConfig();

    // Autonomous-but-capped escalation with the preset defaults (SC-5).
    expect(cfg.escalationMode).toBeDefined();
    expect(cfg.escalationMode?.kind).toBe("auto_continue");
    if (cfg.escalationMode?.kind !== "auto_continue")
      throw new Error("unreachable");
    expect(cfg.escalationMode.maxGrants).toBe(PRESET_MAX_AUTO_GRANTS);
    expect(cfg.escalationMode.stepsPerGrant).toBe(PRESET_STEPS_PER_GRANT);

    // The built-in coding system prompt is installed.
    expect(cfg.systemPrompt).toBe(CODING_AGENT_SYSTEM_PROMPT);

    // The workspace-scoped sandbox is rooted at the workspace (canonicalized).
    expect(cfg.sandbox.workspaceRoot?.()).toBeTruthy();
    expect(cfg.sandbox.isolationMode?.()).toEqual({ kind: "workspace_scoped" });

    // The coding catalogue is folded into the per-run catalogue registry (not the
    // harness-loop tool registry, which stays the conversational empty registry
    // until a run binds a ToolContext). Assert a representative sample of
    // codingSet().
    expect(cfg.catalogueRegistry).toBeDefined();
    const names = new Set(
      cfg.catalogueRegistry?.activeSchemas(null).map((s) => s.name),
    );
    for (const expected of [
      "read_file",
      "write_file",
      "edit_file",
      "bash_command",
      "send_message",
    ]) {
      expect(names, `coding_set must include ${expected}`).toContain(expected);
    }
  });

  it("errors on a missing workspace", () => {
    const missing = "/spore-sc8-does-not-exist-37a1/nope";
    // A workspace path that can't be resolved is a typed BuildError, not a crash.
    expect(() => codingAgent(presetMockModel(), missing)).toThrow();
    try {
      codingAgent(presetMockModel(), missing);
      throw new Error(
        "a missing workspace must surface a BuildError, not build",
      );
    } catch (e) {
      expect(e).toBeInstanceOf(Error);
      expect((e as { name?: string }).name).toBe("BuildError");
      expect((e as { kind?: string }).kind).toBe("root_not_found");
    }
  });
});

// SC-8: `hillClimber` registers the scoring evaluator (required for the
// hill_climbing strategy) under the default handle and defaults to auto_continue
// — the cordyceps preset.
describe("hillClimber preset", () => {
  it("registers the evaluator and defaults to auto_continue", () => {
    const evaluator = new StubMetricEvaluator();
    const cfg = hillClimber(presetMockModel(), evaluator).buildConfig();

    expect(cfg.registry.resolveMetricEvaluator("")).toBeDefined();
    expect(cfg.registry.resolveMetricEvaluator("")).toBe(evaluator);

    expect(cfg.escalationMode?.kind).toBe("auto_continue");
  });
});
