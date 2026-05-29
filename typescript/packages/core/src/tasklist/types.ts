/**
 * Persisted task-list primitive (spore-core issue #71 — PlanExecute, drives the
 * execute loop).
 *
 * Decomposed out of #59. The accepted plan (#70) is parsed into a persisted
 * task list (#72), and the execute phase (#59) loops over the tasks until the
 * list is complete. This module owns the **task-list primitive** that holds and
 * mutates that list, plus the disk-persistence helpers. The single mutating
 * tool (`task_list`) lives in `@spore/tools`. It is consumed by #72 (which
 * populates the list) and #59 (whose execute loop drains it).
 *
 * # Types
 * - {@link TaskStatus} — `pending | in_progress | completed | blocked`.
 *   Serializes to those snake_case strings. Byte-identical across all four
 *   languages.
 * - {@link Task} — `{ id, description, status }`. A flat record: NO hierarchy /
 *   subtasks, NO timestamps (byte-identity constraint), NO order field (order is
 *   positional in {@link TaskList.tasks}).
 * - {@link TaskList} — `{ tasks, next_id }`. The persisted collection.
 *   {@link defaultTaskList} is empty with `next_id === 1`.
 * - {@link TaskListError} — domain error ({@link TaskNotFoundError},
 *   {@link InvalidTransitionError}). These map to a recoverable tool error at
 *   the tool boundary; the helpers NEVER throw uncaught at that boundary.
 *
 * # ID scheme
 * Sequential, 1-based, assigned monotonically from {@link TaskList.next_id}.
 * Ids are never reused — `next_id` only ever grows, and it is preserved across
 * reload so a freshly-loaded list keeps minting fresh ids.
 *
 * # Rules enforced
 * - R1  Ids are assigned 1, 2, 3, … monotonically from `next_id`; never reused.
 * - R2  {@link addTask} APPENDS to the end of `tasks` (positional order stable).
 * - R3  Listing never mutates the list (the tool's `list_tasks` does not write).
 * - R4  Unknown id on update/complete → {@link TaskNotFoundError}.
 * - R5  Status transitions follow {@link validateTransition} (DECISION 1,
 *   "permissive-except-terminal-Completed"). A rejected transition →
 *   {@link InvalidTransitionError}.
 * - R6  `completed` is terminal: ANY transition OUT of `completed` is rejected
 *   (the idempotent `completed → completed` self-transition is allowed).
 * - R7  Self-transitions `X → X` are idempotent and always allowed.
 * - R8  Persistence is interim, through the FILESYSTEM via the
 *   {@link SandboxProvider} read-modify-write at {@link TASK_LIST_PATH}
 *   (DECISION 2). The tool does NOT touch `SessionState.extras`; #59 owns the
 *   extras mirror.
 *
 * Both design forks (transition matrix, state seam) were resolved before
 * implementation.
 */

import { promises as fs } from "node:fs";
import { dirname } from "node:path";

import { z } from "zod";

import type { Operation, SandboxProvider, SandboxViolation } from "../harness/types.js";

/**
 * Key under which the {@link TaskList} is mirrored into `SessionState.extras`
 * (serialized JSON) by the harness / #59. Mirrors `PLAN_EXECUTE_EXTRAS_KEY`.
 * Stable across all four languages. NOTE: #71 itself does NOT write this key
 * (the `Tool` interface has no `SessionState` access); it is the contract the
 * harness-side mirror uses.
 */
export const TASK_LIST_EXTRAS_KEY = "task_list";

/**
 * Canonical on-disk location of the persisted task list, relative to the
 * sandbox/workspace root. Resolved through {@link SandboxProvider.resolvePath}.
 */
export const TASK_LIST_PATH = ".spore/task_list.json";

// ============================================================================
// Types
// ============================================================================

/** Lifecycle status of a {@link Task}. Serializes to snake_case. */
export const TaskStatusSchema = z.enum(["pending", "in_progress", "completed", "blocked"]);
export type TaskStatus = z.infer<typeof TaskStatusSchema>;

/**
 * A single task: flat, no hierarchy, no timestamps, no order field (order is
 * positional in {@link TaskList.tasks}).
 */
export const TaskSchema = z.object({
  id: z.number().int().nonnegative(),
  description: z.string(),
  status: TaskStatusSchema,
});
export type Task = z.infer<typeof TaskSchema>;

/**
 * The persisted collection of tasks plus the monotonic id counter.
 *
 * Serializes as `{"tasks":[...],"next_id":N}`. `next_id` defaults to `0` when
 * absent so an older/handwritten blob without it still deserializes, but
 * {@link defaultTaskList} and every freshly-minted list start at `1`.
 */
export const TaskListSchema = z.object({
  tasks: z.array(TaskSchema),
  next_id: z.number().int().nonnegative().default(0),
});
export type TaskList = z.infer<typeof TaskListSchema>;

/** A fresh, empty list whose first minted id will be `1`. */
export function defaultTaskList(): TaskList {
  return { tasks: [], next_id: 1 };
}

// ============================================================================
// Serialization (byte-identical: field order `tasks` then `next_id`; per-task
// `id`, `description`, `status`). `JSON.stringify` preserves insertion order,
// so the objects are rebuilt in canonical key order rather than relying on the
// key order of an arbitrary input object.
// ============================================================================

/** Serialize `list` to the canonical compact JSON form. */
export function serializeTaskList(list: TaskList): string {
  const tasks = list.tasks.map((t) => ({
    id: t.id,
    description: t.description,
    status: t.status,
  }));
  return JSON.stringify({ tasks, next_id: list.next_id });
}

/** Parse the canonical JSON form into a validated {@link TaskList}. */
export function parseTaskList(text: string): TaskList {
  return TaskListSchema.parse(JSON.parse(text));
}

// ============================================================================
// Errors
// ============================================================================

/**
 * Errors raised by task-list mutations. Domain error classes with a discriminant
 * `kind` field (per CONVENTIONS.md). Both variants are recoverable at the tool
 * boundary. Wire-tagged on `kind` in snake_case to match the Rust
 * `TaskListError` enum.
 */
export type TaskListErrorKind =
  | { kind: "task_not_found"; id: number }
  | { kind: "invalid_transition"; id: number; from: TaskStatus; to: TaskStatus };

export class TaskListError extends Error {
  override readonly name = "TaskListError";
  readonly kind: TaskListErrorKind["kind"];
  readonly detail: TaskListErrorKind;

  constructor(detail: TaskListErrorKind) {
    super(taskListErrorMessage(detail));
    this.kind = detail.kind;
    this.detail = detail;
  }

  static taskNotFound(id: number): TaskListError {
    return new TaskListError({ kind: "task_not_found", id });
  }

  static invalidTransition(id: number, from: TaskStatus, to: TaskStatus): TaskListError {
    return new TaskListError({ kind: "invalid_transition", id, from, to });
  }
}

export function taskListErrorMessage(e: TaskListErrorKind): string {
  switch (e.kind) {
    case "task_not_found":
      return `task not found: ${e.id}`;
    case "invalid_transition":
      return `invalid transition for task ${e.id}: ${e.from} -> ${e.to}`;
  }
}

/** Total result of a mutation: success or a recoverable domain error. */
export type MutationResult = { ok: true } | { ok: false; error: TaskListError };

// ============================================================================
// Transition matrix (DECISION 1)
// ============================================================================

/**
 * Validate a status transition under DECISION 1
 * ("permissive-except-terminal-Completed").
 *
 * Allowed:
 * - any self-transition `X → X` (idempotent),
 * - `pending → in_progress | completed | blocked`,
 * - `in_progress → completed | blocked`,
 * - `blocked → in_progress | completed`.
 *
 * Rejected: ANY transition OUT of `completed` (it is terminal) — except the
 * idempotent `completed → completed`.
 *
 * The `id` is carried only to populate {@link InvalidTransitionError}; it is not
 * otherwise inspected. Returns a {@link MutationResult} and never throws.
 */
export function validateTransition(id: number, from: TaskStatus, to: TaskStatus): MutationResult {
  // Idempotent self-transition always allowed (incl. completed -> completed).
  if (from === to) return { ok: true };

  const allowed =
    (from === "pending" && to === "in_progress") ||
    (from === "pending" && to === "completed") ||
    (from === "pending" && to === "blocked") ||
    (from === "in_progress" && to === "completed") ||
    (from === "in_progress" && to === "blocked") ||
    (from === "blocked" && to === "in_progress") ||
    (from === "blocked" && to === "completed");

  return allowed
    ? { ok: true }
    : { ok: false, error: TaskListError.invalidTransition(id, from, to) };
}

// ============================================================================
// TaskList mutation helpers (the seam #72 / #59 call)
// ============================================================================

/**
 * Append a new `pending` task with the next sequential id and return that id.
 * Increments {@link TaskList.next_id}. R1, R2.
 */
export function addTask(list: TaskList, description: string): number {
  const id = list.next_id;
  list.tasks.push({ id, description, status: "pending" });
  list.next_id += 1;
  return id;
}

/**
 * Update a task's status and/or description.
 *
 * - Unknown id → {@link TaskNotFoundError}.
 * - `status` present → validated via {@link validateTransition} then applied.
 * - `description` present → set verbatim.
 * - Both absent → no-op success.
 *
 * Status is validated BEFORE any field is written, so a rejected transition
 * leaves the task untouched.
 */
export function updateTask(
  list: TaskList,
  id: number,
  status?: TaskStatus,
  description?: string,
): MutationResult {
  const task = list.tasks.find((t) => t.id === id);
  if (task === undefined) {
    return { ok: false, error: TaskListError.taskNotFound(id) };
  }
  if (status !== undefined) {
    const v = validateTransition(id, task.status, status);
    if (!v.ok) return v;
    task.status = status;
  }
  if (description !== undefined) {
    task.description = description;
  }
  return { ok: true };
}

/**
 * Mark a task `completed`, validating the transition first.
 *
 * - Unknown id → {@link TaskNotFoundError}.
 * - Already `completed` → idempotent success.
 */
export function completeTask(list: TaskList, id: number): MutationResult {
  const task = list.tasks.find((t) => t.id === id);
  if (task === undefined) {
    return { ok: false, error: TaskListError.taskNotFound(id) };
  }
  const v = validateTransition(id, task.status, "completed");
  if (!v.ok) return v;
  task.status = "completed";
  return { ok: true };
}

// ============================================================================
// Disk persistence (interim — DECISION 2, mirrors the plan-phase precedent)
// ============================================================================

/** Failure modes of {@link loadTaskList}. */
export type LoadError =
  | { kind: "sandbox"; violation: SandboxViolation }
  | { kind: "parse"; reason: string };

/** Failure modes of {@link storeTaskList}. */
export type StoreError =
  | { kind: "sandbox"; violation: SandboxViolation }
  | { kind: "serialize"; reason: string }
  | { kind: "io"; reason: string };

export type LoadResult = { ok: true; list: TaskList } | { ok: false; error: LoadError };

export type StoreResult = { ok: true } | { ok: false; error: StoreError };

function isSandboxViolation(v: unknown): v is SandboxViolation {
  return typeof v === "object" && v !== null && "kind" in v;
}

async function resolve(
  sandbox: SandboxProvider,
  op: Operation,
): Promise<string | SandboxViolation> {
  if (sandbox.resolvePath) return sandbox.resolvePath(TASK_LIST_PATH, op);
  // Identity fallback for sandboxes that don't override resolvePath.
  return TASK_LIST_PATH;
}

/**
 * Load the persisted {@link TaskList} from {@link TASK_LIST_PATH} via the
 * sandbox.
 *
 * An absent file (the expected first-run path) yields {@link defaultTaskList}.
 * A present-but-malformed file surfaces a `parse` error so the tool boundary can
 * map it to a recoverable error rather than silently discarding state.
 */
export async function loadTaskList(sandbox: SandboxProvider): Promise<LoadResult> {
  const resolved = await resolve(sandbox, "read");
  if (isSandboxViolation(resolved)) {
    return { ok: false, error: { kind: "sandbox", violation: resolved } };
  }
  let text: string;
  try {
    text = await fs.readFile(resolved, "utf8");
  } catch {
    // Absent (or unreadable) file → fresh list.
    return { ok: true, list: defaultTaskList() };
  }
  try {
    return { ok: true, list: parseTaskList(text) };
  } catch (e) {
    return {
      ok: false,
      error: {
        kind: "parse",
        reason: e instanceof Error ? e.message : String(e),
      },
    };
  }
}

/**
 * Persist `list` to {@link TASK_LIST_PATH} via the sandbox, creating the parent
 * directory (`.spore/`) if needed. Serialization is the canonical compact form
 * (field order `tasks` then `next_id`).
 */
export async function storeTaskList(
  list: TaskList,
  sandbox: SandboxProvider,
): Promise<StoreResult> {
  const resolved = await resolve(sandbox, "write");
  if (isSandboxViolation(resolved)) {
    return { ok: false, error: { kind: "sandbox", violation: resolved } };
  }
  let json: string;
  try {
    json = serializeTaskList(list);
  } catch (e) {
    return {
      ok: false,
      error: {
        kind: "serialize",
        reason: e instanceof Error ? e.message : String(e),
      },
    };
  }
  try {
    // Best-effort parent creation; ignore "already exists".
    await fs.mkdir(dirname(resolved), { recursive: true });
  } catch {
    // Ignore — a real write failure surfaces below.
  }
  try {
    await fs.writeFile(resolved, json, "utf8");
    return { ok: true };
  } catch (e) {
    return {
      ok: false,
      error: { kind: "io", reason: e instanceof Error ? e.message : String(e) },
    };
  }
}
