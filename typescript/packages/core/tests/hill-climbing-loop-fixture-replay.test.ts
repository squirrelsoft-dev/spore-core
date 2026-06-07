/**
 * Cross-language fixture-replay tests for the HillClimbing loop strategy
 * (issue #60).
 *
 * Loads `fixtures/metric_evaluator/hill_climbing_sequences.json` — the shared
 * ground-truth fixture written by the Rust phase — and replays EVERY scenario
 * through {@link StandardHarness} with a scripted {@link MetricEvaluator}. Each
 * scenario's terminal outcome (halt_reason / kept_iterations / revert_count /
 * best_metric) must match `expected`; where the scenario embeds `expected_tsv`,
 * the harness-written `.spore/results/{task_id}.tsv` must be BYTE-IDENTICAL.
 *
 * Same fixture, same outcome — see `/fixtures/README.md`.
 */

import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  AgentId,
  SessionId,
  StandardHarness,
  newTask,
  type Agent,
  type CommandOutput,
  type Context,
  type HarnessConfig,
  type LoopStrategy,
  type OptimizationDirection,
  type RunResult,
  type SandboxProvider,
  type SandboxViolation,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
} from "../src/index.js";
import type { MetricEvaluator, MetricOutcome } from "../src/metric/types.js";
import type { SessionStateSnapshot } from "../src/termination/types.js";
import {
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(repoRoot, "fixtures/metric_evaluator/hill_climbing_sequences.json");

interface Scenario {
  name: string;
  metric_sequence: (number | null)[];
  max_turns?: number;
  payload: {
    direction: OptimizationDirection;
    max_stagnation: number | null;
    revert_on_no_improvement: boolean;
    min_improvement_delta: number | null;
  };
  expected: {
    halt_reason: string;
    kept_iterations: number;
    revert_count: number;
    best_metric?: number;
  };
  expected_tsv?: string;
}

// --------------------------------------------------------------------------
// Doubles
// --------------------------------------------------------------------------

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

class AlwaysDoneAgent implements Agent {
  constructor(private readonly agentId: AgentId) {}
  id(): AgentId {
    return this.agentId;
  }
  async turn(_ctx: Context): Promise<TurnResult> {
    return { kind: "final_response", content: "done", usage: usage() };
  }
}

/** Replays the scripted metric sequence; null ⇒ crash. ZERO duration. */
class ScriptedMetricEvaluator implements MetricEvaluator {
  private calls = 0;
  constructor(
    private readonly sequence: (number | null)[],
    private readonly dir: OptimizationDirection,
  ) {}
  async evaluate(_sandbox: SandboxProvider, _state: SessionStateSnapshot): Promise<MetricOutcome> {
    const v = this.sequence[this.calls];
    this.calls += 1;
    if (v == null) {
      return { kind: "err", error: { kind: "crashed", log: "scripted crash" } };
    }
    return { kind: "ok", result: { value: v, raw_output: "", duration: 0, metadata: {} } };
  }
  direction(): OptimizationDirection {
    return this.dir;
  }
  description(): string {
    return "scripted metric";
  }
}

/** Records executeCommand calls (to count reverts) and reports a workspace. */
class RecordingSandbox implements SandboxProvider {
  readonly commands: { command: string; args: readonly string[] }[] = [];
  constructor(private readonly root: string) {}
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
  async executeCommand(
    command: string,
    args: readonly string[],
  ): Promise<CommandOutput | SandboxViolation> {
    this.commands.push({ command, args });
    return { stdout: "", stderr: "", exit_code: 0, timed_out: false, truncated: false };
  }
  workspaceRoot(): string {
    return this.root;
  }
  revertCount(): number {
    return this.commands.filter(
      (c) => c.command === "git" && c.args.join(" ") === "reset --hard HEAD",
    ).length;
  }
}

/** Map a terminal {@link RunResult} to the fixture's `halt_reason` vocabulary. */
function haltReasonTag(r: RunResult): string {
  if (r.kind !== "failure") return r.kind;
  switch (r.reason.kind) {
    case "stagnation_limit_reached":
      return "stagnation";
    case "budget_exceeded":
      return r.reason.limit_type === "turns" ? "budget_turns" : "budget";
    case "hill_climbing_misconfigured":
      return "misconfigured";
    default:
      return r.reason.kind;
  }
}

// --------------------------------------------------------------------------
// Replay
// --------------------------------------------------------------------------

const fixture = JSON.parse(readFileSync(fixturePath, "utf-8")) as { scenarios: Scenario[] };

describe("HillClimbing fixture replay (issue #60)", () => {
  it("loads all 7 scenarios", () => {
    expect(fixture.scenarios.length).toBe(7);
  });

  for (const sc of fixture.scenarios) {
    it(`scenario: ${sc.name}`, async () => {
      const root = mkdtempSync(join(tmpdir(), "spore-hc-fx-"));
      const sandbox = new RecordingSandbox(root);
      const evaluator = new ScriptedMetricEvaluator(sc.metric_sequence, sc.payload.direction);
      const agent = new AlwaysDoneAgent(AgentId.of("a"));

      const config: HarnessConfig = {
        toolRegistry: new ScriptedToolRegistry(),
        sandbox,
        contextManager: new NoopContextManager(),
        terminationPolicy: new AlwaysContinuePolicy(),
        modelParams: { stop_sequences: [] },
        registry: registryWith({ agent, metricEvaluator: evaluator }),
      };

      const strategy: LoopStrategy = {
        kind: "hill_climbing",
        inner: {
          kind: "react",
          budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
          agent: "",
          toolset: "",
          output: "",
        },
        direction: sc.payload.direction,
        // `null` ("unbounded" / "no delta") maps to behavior-preserving
        // concretes now that the fields are required (#119): MAX_SAFE_INTEGER
        // never trips stagnation, `0` matches the null-as-0.0 keep default.
        max_stagnation: sc.payload.max_stagnation ?? Number.MAX_SAFE_INTEGER,
        revert_on_no_improvement: sc.payload.revert_on_no_improvement,
        min_improvement_delta: sc.payload.min_improvement_delta ?? 0,
        evaluator: "",
      };
      const task = newTask("optimize", SessionId.of("hc-fx"), strategy, {
        max_turns: sc.max_turns ?? 100,
      });

      const h = new StandardHarness(config);
      const r = await h.run({ task });

      // halt_reason.
      expect(haltReasonTag(r)).toBe(sc.expected.halt_reason);

      // best_metric (where the scenario pins it).
      if (sc.expected.best_metric != null && r.kind === "failure") {
        if (r.reason.kind === "stagnation_limit_reached") {
          expect(r.reason.best_metric).toBe(sc.expected.best_metric);
        }
      }

      // revert_count.
      expect(sandbox.revertCount()).toBe(sc.expected.revert_count);

      // The harness always writes the TSV (workspace root is set). Read it to
      // both verify byte-identity (where embedded) and count kept iterations.
      const tsvPath = join(root, ".spore", "results", `${task.id.asString()}.tsv`);
      const tsv = readFileSync(tsvPath, "utf-8");

      // kept_iterations = number of data rows with status `kept`.
      const keptRows = tsv
        .trimEnd()
        .split("\n")
        .slice(1) // drop header
        .filter((line) => line.split("\t")[4] === "kept").length;
      expect(keptRows).toBe(sc.expected.kept_iterations);

      // byte-identical TSV where the scenario embeds it.
      if (sc.expected_tsv != null) {
        expect(tsv).toBe(sc.expected_tsv);
      }
    });
  }
});
