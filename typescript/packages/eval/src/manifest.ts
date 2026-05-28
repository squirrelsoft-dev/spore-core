/**
 * Manifest loading + manual promotion (Rules 6, 29, 31).
 */

import { readFile } from "node:fs/promises";

import { buildVerifier } from "./verifier.js";
import {
  DEFAULT_TASK_TIMEOUT_SECS,
  EvalError,
  type EvalTask,
  type TaskSuite,
} from "./task.js";

/**
 * Load a {@link TaskSuite} from a JSON manifest string. Rejects a manifest
 * without `suite_version` (Rule 6) with {@link EvalError} `missing_suite_version`.
 * Resolves each task's verifier from its spec.
 */
export function loadSuiteStr(json: string): TaskSuite {
  let value: unknown;
  try {
    value = JSON.parse(json);
  } catch (e) {
    throw EvalError.manifestParse((e as Error).message);
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw EvalError.manifestParse("manifest must be a JSON object");
  }
  const obj = value as Record<string, unknown>;
  // Check the raw JSON for the required field so a missing `suite_version` is a
  // precise error rather than a generic parse failure.
  if (!("suite_version" in obj)) {
    throw EvalError.missingSuiteVersion();
  }
  if (
    typeof obj.suite_version !== "number" ||
    !Number.isInteger(obj.suite_version)
  ) {
    throw EvalError.manifestParse("`suite_version` must be an integer");
  }

  const suite: TaskSuite = {
    suite_version: obj.suite_version,
    regression: parseTasks(obj.regression),
    challenge: parseTasks(obj.challenge),
    canary: parseTasks(obj.canary),
  };
  resolveVerifiers(suite);
  return suite;
}

/** Load a {@link TaskSuite} from a manifest file path. */
export async function loadSuitePath(path: string): Promise<TaskSuite> {
  let body: string;
  try {
    body = await readFile(path, "utf8");
  } catch (e) {
    throw EvalError.io(`failed to read ${path}: ${(e as Error).message}`);
  }
  return loadSuiteStr(body);
}

/** Resolve every task's verifier from its spec. Idempotent. */
export function resolveVerifiers(suite: TaskSuite): void {
  for (const t of [...suite.regression, ...suite.challenge, ...suite.canary]) {
    t.verifier = buildVerifier(t.verifier_spec);
  }
}

/** The resolved verifier for a task, building it on demand if not yet resolved. */
export function taskVerifier(task: EvalTask) {
  return task.verifier ?? buildVerifier(task.verifier_spec);
}

/**
 * Serialize a {@link TaskSuite} back to pretty JSON. Drops the transient
 * resolved `verifier` (rebuilt from `verifier_spec` on load).
 */
export function suiteToJson(suite: TaskSuite): string {
  const serializable = {
    suite_version: suite.suite_version,
    regression: suite.regression.map(serializeTask),
    challenge: suite.challenge.map(serializeTask),
    canary: suite.canary.map(serializeTask),
  };
  return JSON.stringify(serializable, null, 2);
}

/**
 * Manually promote a `challenge` task to `regression`, bumping `suite_version`
 * (Rule 31). Auto-promotion is deferred. Throws when `taskId` is not found
 * among the challenge tasks.
 */
export function promoteChallengeTask(suite: TaskSuite, taskId: string): void {
  const pos = suite.challenge.findIndex((t) => t.id === taskId);
  if (pos < 0) {
    throw EvalError.manifestParse(
      `challenge task ${JSON.stringify(taskId)} not found`,
    );
  }
  const [task] = suite.challenge.splice(pos, 1);
  suite.regression.push(task!);
  suite.suite_version += 1;
}

// ============================================================================
// Parsing helpers
// ============================================================================

function parseTasks(raw: unknown): EvalTask[] {
  if (raw == null) return [];
  if (!Array.isArray(raw)) {
    throw EvalError.manifestParse("task list must be an array");
  }
  return raw.map(parseTask);
}

function parseTask(raw: unknown): EvalTask {
  if (typeof raw !== "object" || raw === null) {
    throw EvalError.manifestParse("task must be an object");
  }
  const o = raw as Record<string, unknown>;
  if (typeof o.id !== "string")
    throw EvalError.manifestParse("task.id must be a string");
  if (typeof o.instruction !== "string") {
    throw EvalError.manifestParse(`task ${o.id}: instruction must be a string`);
  }
  if (o.workspace_snapshot == null) {
    throw EvalError.manifestParse(`task ${o.id}: missing workspace_snapshot`);
  }
  if (o.verifier_spec == null) {
    throw EvalError.manifestParse(`task ${o.id}: missing verifier_spec`);
  }
  return {
    id: o.id,
    instruction: o.instruction,
    workspace_snapshot: o.workspace_snapshot as EvalTask["workspace_snapshot"],
    verifier_spec: o.verifier_spec as EvalTask["verifier_spec"],
    expected_turns: (o.expected_turns as [number, number] | undefined) ?? null,
    expected_cost_usd: (o.expected_cost_usd as number | undefined) ?? null,
    tags: Array.isArray(o.tags) ? (o.tags as string[]) : [],
    timeout:
      typeof o.timeout === "number" ? o.timeout : DEFAULT_TASK_TIMEOUT_SECS,
    model_fixture: (o.model_fixture as string | undefined) ?? null,
  };
}

function serializeTask(task: EvalTask): Record<string, unknown> {
  const out: Record<string, unknown> = {
    id: task.id,
    instruction: task.instruction,
    workspace_snapshot: task.workspace_snapshot,
    verifier_spec: task.verifier_spec,
  };
  if (task.expected_turns != null) out.expected_turns = task.expected_turns;
  if (task.expected_cost_usd != null)
    out.expected_cost_usd = task.expected_cost_usd;
  out.tags = task.tags;
  out.timeout = task.timeout;
  if (task.model_fixture != null) out.model_fixture = task.model_fixture;
  return out;
}
