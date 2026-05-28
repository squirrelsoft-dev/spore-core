/**
 * {@link EvalHarness} — the runner (Rules 14-25) — plus its fluent builder and
 * the deferred {@link TraceAnalyzer} interface (Rule 30).
 */

import {
  StandardHarness,
  SessionId,
  TaskId,
  type HarnessConfig,
  type ObservabilityProvider,
  type RunResult,
  type Task,
} from "@spore/core";
import type { observability } from "@spore/core";

type SessionMetrics = observability.SessionMetrics;
type Span = observability.Span;

import {
  metricDirection,
  metricName,
  sampleFor,
  type EvalMetric,
  type RunSampleInputs,
} from "./metric-map.js";
import {
  classifyDirection,
  deriveRecommendation,
  type ComparisonReport,
  type MetricComparison,
} from "./report.js";
import {
  bootstrapCi,
  metricStatsFromSamples,
  welchTTest,
  DEFAULT_BOOTSTRAP_ITERATIONS,
  DEFAULT_BOOTSTRAP_SEED,
} from "./stats.js";
import { taskVerifier } from "./manifest.js";
import {
  allTasks,
  ConfigId,
  EvalError,
  type EvalTask,
  type TaskSuite,
} from "./task.js";
import { Workspace } from "./worktree.js";

// ============================================================================
// TraceAnalyzer (Rule 30 — interface only, no built-in impl ships)
// ============================================================================

/**
 * A proposed change to a {@link HarnessConfig} produced by a
 * {@link TraceAnalyzer}. Marker stub: the optimization loop (propose → run →
 * compare → open PR) is deferred (Rule 30).
 */
export interface HarnessConfigDiff {
  /** Free-form human-readable description of the proposed change. */
  description: string;
}

/**
 * Analyzes failure traces and proposes candidate config diffs (Rule 30).
 * Interface only — no built-in implementation ships in the MVP.
 */
export interface TraceAnalyzer {
  analyze(traces: Span[], signal?: AbortSignal): Promise<HarnessConfigDiff[]>;
}

// ============================================================================
// EvalHarness
// ============================================================================

const BASELINE_CONFIG_ID = "baseline";

/** Construction inputs for {@link EvalHarness}; assemble via {@link EvalHarnessBuilder}. */
export interface EvalHarnessOptions {
  taskSuite: TaskSuite;
  baselineConfig: HarnessConfig;
  candidateConfigs: [ConfigId, HarnessConfig][];
  nRunsPerConfig: number;
  metrics: EvalMetric[];
  observability: ObservabilityProvider;
  bootstrapIterations: number;
  primaryMetric: EvalMetric;
}

/** Per-config metric samples + the metadata the comparison needs. */
interface ConfigSamples {
  perMetric: number[][];
  traceLinks: string[];
  waitingForHuman: number;
}

/**
 * The evaluation harness: runs a task suite against a baseline and candidate
 * configs, aggregates metrics, and compares them.
 */
export class EvalHarness {
  readonly taskSuite: TaskSuite;
  readonly baselineConfig: HarnessConfig;
  readonly candidateConfigs: [ConfigId, HarnessConfig][];
  readonly nRunsPerConfig: number;
  readonly metrics: EvalMetric[];
  readonly observability: ObservabilityProvider;
  readonly bootstrapIterations: number;
  readonly primaryMetric: EvalMetric;

  constructor(opts: EvalHarnessOptions) {
    this.taskSuite = opts.taskSuite;
    this.baselineConfig = opts.baselineConfig;
    this.candidateConfigs = opts.candidateConfigs;
    this.nRunsPerConfig = opts.nRunsPerConfig;
    this.metrics = opts.metrics;
    this.observability = opts.observability;
    this.bootstrapIterations = opts.bootstrapIterations;
    this.primaryMetric = opts.primaryMetric;
  }

  /**
   * Run the full comparison (Rules 14-25). Produces one {@link ComparisonReport}
   * per candidate config.
   */
  async run(signal?: AbortSignal): Promise<ComparisonReport[]> {
    if (this.metrics.length === 0) {
      throw EvalError.missingMetrics("no metrics configured for comparison");
    }

    // Collect per-metric samples for the baseline once.
    const baseline = await this.runConfig(
      this.baselineConfig,
      BASELINE_CONFIG_ID,
      signal,
    );

    const reports: ComparisonReport[] = [];
    for (const [configId, config] of this.candidateConfigs) {
      const candidate = await this.runConfig(
        config,
        configId.asString(),
        signal,
      );
      reports.push(this.compare(baseline, candidate, configId));
    }
    return reports;
  }

  /**
   * Run every task `nRunsPerConfig` times for one config, collecting per-metric
   * samples and trace links for interesting runs.
   */
  private async runConfig(
    config: HarnessConfig,
    configId: string,
    signal?: AbortSignal,
  ): Promise<ConfigSamples> {
    const samples: ConfigSamples = {
      perMetric: this.metrics.map(() => []),
      traceLinks: [],
      waitingForHuman: 0,
    };
    for (const [, task] of allTasks(this.taskSuite)) {
      for (let runIdx = 0; runIdx < this.nRunsPerConfig; runIdx++) {
        await this.runOne(config, configId, task, runIdx, samples, signal);
      }
    }
    return samples;
  }

  /** Execute a single (config, task) run (Rules 2-3, 14-18, 25). */
  private async runOne(
    config: HarnessConfig,
    configId: string,
    task: EvalTask,
    runIdx: number,
    samples: ConfigSamples,
    signal?: AbortSignal,
  ): Promise<void> {
    // Rule 2: fresh workspace restored from the snapshot.
    const workspace = await Workspace.restore(task.workspace_snapshot);
    try {
      // Build a fresh harness from the config (Rule 15) and a unique session.
      const sessionId = SessionId.of(`${configId}-${task.id}-${runIdx}`);
      const harness = new StandardHarness(config);
      const maxTurns = task.expected_turns ? task.expected_turns[1] : 20;
      const coreTask: Task = {
        id: TaskId.of(`${configId}-${task.id}-${runIdx}`),
        instruction: task.instruction,
        session_id: sessionId,
        budget: { max_turns: maxTurns, max_wall_time: task.timeout },
        loop_strategy: { kind: "re_act", max_iterations: maxTurns },
      };

      // Rule 15 / Rule 4: run the harness, bounded by the per-task timeout. A
      // timeout yields a failed run rather than throwing.
      const runResult = await this.runWithTimeout(
        harness,
        coreTask,
        sessionId,
        task.timeout,
        signal,
      );

      // Rule 16: read metrics from observability (do not recompute).
      const sessionMetrics =
        (await this.observability.getSessionMetrics(sessionId)) ??
        emptySessionMetrics(sessionId, coreTask.id);
      const trace = await this.observability.getTrace(sessionId);

      // Run the verifier (Rules 7-13).
      const verifier = taskVerifier(task);
      const verification = await verifier.verify(
        task,
        runResult,
        workspace.path,
      );

      // Rule 18: WaitingForHuman counts as neither success nor failure; it is
      // reported separately and excluded from success-rate / score samples.
      const waiting = runResult.kind === "waiting_for_human";
      if (waiting) samples.waitingForHuman += 1;

      const inputs: RunSampleInputs = {
        verifierPassed: verification.passed,
        verifierScore: verification.score,
      };

      this.metrics.forEach((metric, i) => {
        // Skip success-rate / verification-score samples for WaitingForHuman
        // runs (Rule 18); resource metrics still count.
        if (
          waiting &&
          (metric.kind === "task_success_rate" ||
            metric.kind === "verification_score")
        ) {
          return;
        }
        samples.perMetric[i]!.push(
          sampleFor(metric, sessionMetrics, trace, inputs),
        );
      });

      // Rule 25: collect trace links for failed or non-passing runs.
      if (!verification.passed || runResult.kind === "failure") {
        samples.traceLinks.push(sessionId.asString());
      }
    } finally {
      // Rule 3: workspace torn down here regardless of outcome.
      await workspace.teardown();
    }
  }

  /** Run the harness with a wall-clock timeout (Rule 4). On timeout, returns a
   *  `failure` RunResult rather than throwing. */
  private async runWithTimeout(
    harness: StandardHarness,
    task: Task,
    sessionId: SessionId,
    timeoutSecs: number,
    signal?: AbortSignal,
  ): Promise<RunResult> {
    const timeoutMs = Math.max(1, timeoutSecs * 1000);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<RunResult>((resolve) => {
      timer = setTimeout(() => {
        resolve({
          kind: "failure",
          reason: { kind: "budget_exceeded", limit_type: "wall_time" },
          session_id: sessionId,
          usage: {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            cost_usd: 0,
          },
          turns: 0,
        });
      }, timeoutMs);
    });
    try {
      return await Promise.race([harness.run({ task, signal }), timeout]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  /** Compare baseline vs candidate samples (Rules 19-25). */
  private compare(
    baseline: ConfigSamples,
    candidate: ConfigSamples,
    configId: ConfigId,
  ): ComparisonReport {
    const comparisons: MetricComparison[] = [];
    this.metrics.forEach((metric, i) => {
      const base = baseline.perMetric[i]!;
      const cand = candidate.perMetric[i]!;
      const baseStats = metricStatsFromSamples(base); // Rule 19
      const candStats = metricStatsFromSamples(cand);
      const delta = candStats.mean - baseStats.mean;
      const welch = welchTTest(base, cand); // Rule 20
      const direction = classifyDirection(delta, metricDirection(metric), 1e-9); // Rule 22

      // Rule 21: bootstrap CI for metrics from non-deterministic verifiers.
      const cmp: MetricComparison = {
        metricName: metricName(metric),
        baseline: baseStats,
        candidate: candStats,
        delta,
        pValue: welch.pValue,
        direction,
      };
      if (this.metricIsNonDeterministic(metric)) {
        const ci = bootstrapCi(
          cand,
          this.bootstrapIterations,
          0.95,
          DEFAULT_BOOTSTRAP_SEED,
        );
        if (ci) cmp.ci = ci;
      }
      comparisons.push(cmp);
    });

    // Rules 23-24.
    const recommendation = deriveRecommendation(
      configId.asString(),
      comparisons,
      this.primaryMetric,
    );

    // Rule 25.
    const traceLinks = [...candidate.traceLinks, ...baseline.traceLinks];

    return {
      baselineConfigId: BASELINE_CONFIG_ID,
      candidateConfigId: configId.asString(),
      metrics: comparisons,
      recommendation,
      traceLinks,
    };
  }

  /**
   * Whether a metric should carry a bootstrap CI (Rule 21): metrics derived
   * from non-deterministic verifiers (any task whose verifier reports
   * `isDeterministic() === false`).
   */
  private metricIsNonDeterministic(metric: EvalMetric): boolean {
    const verifierDependent =
      metric.kind === "task_success_rate" ||
      metric.kind === "verification_score";
    if (!verifierDependent) return false;
    return allTasks(this.taskSuite).some(
      ([, t]) => !taskVerifier(t).isDeterministic(),
    );
  }
}

function emptySessionMetrics(
  sessionId: SessionId,
  taskId: TaskId,
): SessionMetrics {
  return {
    session_id: sessionId,
    task_id: taskId,
    total_turns: 0,
    total_input_tokens: 0,
    total_output_tokens: 0,
    total_cost_usd: 0,
    total_duration_ms: 0,
    tool_calls: 0,
    sensor_fires: 0,
    sensor_halts: 0,
    compactions: 0,
    outcome: { kind: "partial" },
    guides_used: [],
    patch_count: 0,
    patch_rate: 0,
    patches_by_tool: {},
    compaction_verification_failures: 0,
  };
}

// ============================================================================
// EvalHarnessBuilder
// ============================================================================

/** Fluent assembler for an {@link EvalHarness}, mirroring `HarnessBuilder`. */
export class EvalHarnessBuilder {
  private _candidateConfigs: [ConfigId, HarnessConfig][] = [];
  private _nRunsPerConfig = 3;
  private _metrics: EvalMetric[] = [{ kind: "task_success_rate" }];
  private _bootstrapIterations = DEFAULT_BOOTSTRAP_ITERATIONS;
  private _primaryMetric: EvalMetric = { kind: "task_success_rate" };

  /**
   * Start from the required pieces: a suite, the baseline config, and the
   * observability provider the runner reads metrics from.
   */
  constructor(
    private readonly taskSuite: TaskSuite,
    private readonly baselineConfig: HarnessConfig,
    private readonly observability: ObservabilityProvider,
  ) {}

  candidate(id: string, config: HarnessConfig): this {
    this._candidateConfigs.push([ConfigId.of(id), config]);
    return this;
  }

  nRunsPerConfig(n: number): this {
    this._nRunsPerConfig = n;
    return this;
  }

  metrics(metrics: EvalMetric[]): this {
    this._metrics = metrics;
    return this;
  }

  bootstrapIterations(n: number): this {
    this._bootstrapIterations = n;
    return this;
  }

  primaryMetric(metric: EvalMetric): this {
    this._primaryMetric = metric;
    return this;
  }

  build(): EvalHarness {
    return new EvalHarness({
      taskSuite: this.taskSuite,
      baselineConfig: this.baselineConfig,
      candidateConfigs: this._candidateConfigs,
      nRunsPerConfig: this._nRunsPerConfig,
      metrics: this._metrics,
      observability: this.observability,
      bootstrapIterations: this._bootstrapIterations,
      primaryMetric: this._primaryMetric,
    });
  }
}
