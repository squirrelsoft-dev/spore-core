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
 * - {@link Task} — `{ id, description, status, blockers }`. A flat record: NO
 *   hierarchy / subtasks, NO timestamps (byte-identity constraint), NO order
 *   field (order is positional in {@link TaskList.tasks}). `blockers` (#118) are
 *   ids of other tasks that must be `completed` before this task runs; it is the
 *   LAST wire field and defaults to `[]` when absent, so pre-#118 blobs still
 *   load. Empty blockers ALWAYS serialize as `[]`, mirroring `next_id`.
 * - {@link TaskList} — `{ tasks, next_id }`. The persisted collection.
 *   {@link defaultTaskList} is empty with `next_id === 1`.
 * - {@link TaskListError} — domain error ({@link TaskNotFoundError},
 *   {@link InvalidTransitionError}, {@link InvalidBlockersError}). These map to a
 *   recoverable tool error at the tool boundary; the helpers NEVER throw uncaught
 *   at that boundary.
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
 * - R9  (#118) {@link addTask} blockers are validated BEFORE any mutation; a
 *   reject leaves the list untouched (mirrors {@link updateTask}). Validation:
 *   self-block (a blocker == the about-to-be-assigned id) →
 *   {@link InvalidBlockersError} `{ self_block }`; unknown id (a blocker matching
 *   no existing task) → `{ unknown_id }`; cycle (the new edges would close a
 *   directed cycle in the blockers graph, checked by {@link wouldCreateCycle}) →
 *   `{ cycle }`. Empty blockers never reject.
 * - R8  Persistence is through the storage seam (#75): the standalone
 *   `task_list` tool (in `@spore/tools`) persists via the
 *   {@link "../storage/types.js".RunStore} on the `ToolContext`, keyed by
 *   `SessionId` under {@link TASK_LIST_EXTRAS_KEY}. The retired interim sandbox
 *   path (`.spore/task_list.json`) is GONE; with the library's default no-op
 *   storage a standalone tool call persists nothing across processes (an
 *   accepted behavior change — no migration shim). #59's execute loop shares the
 *   same `RunStore` key.
 *
 * Both design forks (transition matrix, state seam) were resolved before
 * implementation.
 */

import { z } from "zod";

import type { PlanArtifact } from "../plan/index.js";

/**
 * Key under which the {@link TaskList} is persisted in the
 * {@link "../storage/types.js".RunStore}, keyed by `SessionId`. Both the
 * harness-side #59 execute loop and the standalone `task_list` tool (#75) share
 * this single key, so a standalone tool call and a PlanExecute run on the same
 * session intentionally share one blob. Stable across all four languages. The
 * JSON shape is the canonical serialized {@link TaskList}
 * (`{"tasks":[...],"next_id":N}`).
 */
export const TASK_LIST_EXTRAS_KEY = "task_list";

// ============================================================================
// Types
// ============================================================================

/**
 * Lifecycle status of a {@link Task}. Serializes to snake_case.
 *
 * `blocked` (#118) means BOTH "waiting on a blocker that has not yet completed"
 * AND "a blocker failed terminally" — the status is the same in either case; the
 * distinction (if any) lives in the scheduler, not the schema.
 */
export const TaskStatusSchema = z.enum(["pending", "in_progress", "completed", "blocked"]);
export type TaskStatus = z.infer<typeof TaskStatusSchema>;

/**
 * A single task: flat, no hierarchy, no timestamps, no order field (order is
 * positional in {@link TaskList.tasks}).
 *
 * `blockers` (#118) are the ids of other tasks that must be `completed` before
 * this task runs. The canonical wire field order is `id, description, status,
 * blockers` (blockers LAST). `blockers` defaults to `[]` when absent so a
 * pre-#118 blob without the key still deserializes; empty blockers ALWAYS
 * serialize as `[]`, the same treatment as {@link TaskList.next_id}.
 */
export const TaskSchema = z.object({
  id: z.number().int().nonnegative(),
  description: z.string(),
  status: TaskStatusSchema,
  blockers: z.array(z.number().int().nonnegative()).default([]),
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
// `id`, `description`, `status`, `blockers`). `JSON.stringify` preserves
// insertion order, so the objects are rebuilt in canonical key order rather than
// relying on the key order of an arbitrary input object. Empty `blockers` always
// serialize as `[]` (never omitted), mirroring `next_id`.
// ============================================================================

/** Serialize `list` to the canonical compact JSON form. */
export function serializeTaskList(list: TaskList): string {
  const tasks = list.tasks.map((t) => ({
    id: t.id,
    description: t.description,
    status: t.status,
    blockers: t.blockers,
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
 * Why an `add_task` blockers set was rejected (#118). Tagged on `reason`
 * (snake_case), matching the Rust `BlockerRejection` enum wire shape:
 * - `self_block` — a blocker referenced the id about to be assigned to this task.
 * - `unknown_id` — a blocker referenced an id matching no existing task (carries
 *   the offending `blocker`).
 * - `cycle` — the new blocker edges would close a directed cycle in the graph.
 */
export type BlockerRejection =
  | { reason: "self_block" }
  | { reason: "unknown_id"; blocker: number }
  | { reason: "cycle" };

export function blockerRejectionMessage(r: BlockerRejection): string {
  switch (r.reason) {
    case "self_block":
      return "a task cannot block itself";
    case "unknown_id":
      return `unknown blocker id: ${r.blocker}`;
    case "cycle":
      return "blocker edges would create a cycle";
  }
}

/**
 * Errors raised by task-list mutations. Domain error classes with a discriminant
 * `kind` field (per CONVENTIONS.md). Every variant is recoverable at the tool
 * boundary. Wire-tagged on `kind` in snake_case to match the Rust
 * `TaskListError` enum.
 */
export type TaskListErrorKind =
  | { kind: "task_not_found"; id: number }
  | { kind: "invalid_transition"; id: number; from: TaskStatus; to: TaskStatus }
  | { kind: "invalid_blockers"; id: number; reason: BlockerRejection };

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

  static invalidBlockers(id: number, reason: BlockerRejection): TaskListError {
    return new TaskListError({ kind: "invalid_blockers", id, reason });
  }
}

export function taskListErrorMessage(e: TaskListErrorKind): string {
  switch (e.kind) {
    case "task_not_found":
      return `task not found: ${e.id}`;
    case "invalid_transition":
      return `invalid transition for task ${e.id}: ${e.from} -> ${e.to}`;
    case "invalid_blockers":
      return `invalid blockers for task ${e.id}: ${blockerRejectionMessage(e.reason)}`;
  }
}

/** Total result of a mutation: success or a recoverable domain error. */
export type MutationResult = { ok: true } | { ok: false; error: TaskListError };

/**
 * Result of {@link addTask}: the assigned id on success, or a recoverable domain
 * error (blocker validation, #118). Fallible since #118.
 */
export type AddResult = { ok: true; id: number } | { ok: false; error: TaskListError };

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
 *
 * Fallible since #118: `blockers` are validated BEFORE any mutation, so a
 * rejected blocker set leaves the list completely untouched (mirroring how
 * {@link updateTask} validates before writing). R9. Validation order:
 * 1. self-block — a blocker equal to the id about to be assigned (`next_id`) →
 *    {@link BlockerRejection} `self_block`.
 * 2. unknown id — a blocker matching no existing task id → `unknown_id`.
 * 3. cycle — the new edges would close a directed cycle, checked by
 *    {@link wouldCreateCycle} → `cycle`.
 *
 * Empty `blockers` always pass (and serialize as `[]`). `blockers` defaults to
 * `[]` when omitted, so existing callers (`addTask(list, "x")`) keep working.
 */
export function addTask(list: TaskList, description: string, blockers: number[] = []): AddResult {
  const id = list.next_id;

  for (const blocker of blockers) {
    if (blocker === id) {
      return { ok: false, error: TaskListError.invalidBlockers(id, { reason: "self_block" }) };
    }
    if (!list.tasks.some((t) => t.id === blocker)) {
      return {
        ok: false,
        error: TaskListError.invalidBlockers(id, { reason: "unknown_id", blocker }),
      };
    }
  }

  if (wouldCreateCycle(list, id, blockers)) {
    return { ok: false, error: TaskListError.invalidBlockers(id, { reason: "cycle" }) };
  }

  list.tasks.push({ id, description, status: "pending", blockers });
  list.next_id += 1;
  return { ok: true, id };
}

/**
 * Would adding a node `newId` whose outgoing blocker edges are `newBlockers`
 * close a directed cycle in the blockers graph?
 *
 * The graph is `task -> blocker` (a task points at each id it is blocked by). A
 * cycle exists if, starting from any of the new edges' targets, a directed path
 * leads back to `newId`. Since a single append-only {@link addTask} only
 * references EARLIER real ids, this can never actually fire today; the helper
 * exists as a spec acceptance criterion (#118) and is unit-tested directly
 * against a hand-built cyclic graph.
 */
export function wouldCreateCycle(list: TaskList, newId: number, newBlockers: number[]): boolean {
  const stack: number[] = [...newBlockers];
  const visited = new Set<number>();

  while (stack.length > 0) {
    const node = stack.pop() as number;
    if (node === newId) return true;
    if (visited.has(node)) continue;
    visited.add(node);
    const task = list.tasks.find((t) => t.id === node);
    if (task !== undefined) stack.push(...task.blockers);
  }
  return false;
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
// Plan → TaskList parser (issue #72; the bridge between #70 and #59)
// ============================================================================

/**
 * Parse an accepted {@link PlanArtifact} (#70) into a fresh, ready-to-persist
 * {@link TaskList} (#71). This is the bridge between the plan phase and the
 * execute loop: once a plan is produced and accepted, its steps become the task
 * list that #59's execute loop drains.
 *
 * # Types bridged
 * - Input: {@link PlanArtifact} `{ tasks: string[]; rationale: string }`.
 * - Output: {@link TaskList} `{ tasks: Task[]; next_id: number }`.
 *
 * # Rules enforced
 * - One {@link Task} per plan step, in plan order (positional, via
 *   {@link addTask}).
 * - Every produced task is `pending`.
 * - Step descriptions are copied VERBATIM — no trim, no normalize, no filter
 *   (matches #70's verbatim contract: even `"  spaced  "` and `""` are kept).
 * - Ids are assigned `1..=n` sequentially via the {@link TaskList.next_id}
 *   scheme; `next_id` ends at `n + 1`.
 * - An empty plan (`tasks: []`) yields {@link defaultTaskList} —
 *   `{ tasks: [], next_id: 1 }`. That is a valid EMPTY list, not an error and
 *   not "immediate completion"; the execute loop (#59) decides loop semantics.
 * - `rationale` is DROPPED — neither {@link Task} nor {@link TaskList} carries
 *   it.
 *
 * # Determinism
 * Pure and total: `PlanArtifact -> TaskList`, no async, no I/O, never throws.
 * The same artifact always yields the same task list, so the mapping is
 * byte-identical across all four languages.
 *
 * # Re-parsing / wiring
 * Always builds a fresh {@link defaultTaskList}; it never merges into an
 * existing list (replanning is out of scope — single parse per accepted plan).
 * Wiring this into the plan-acceptance seam is DEFERRED to #59's execute loop;
 * #72 ships only this pure function.
 */
export function planArtifactToTaskList(artifact: PlanArtifact): TaskList {
  const list = defaultTaskList(); // next_id === 1
  for (const step of artifact.tasks) {
    // verbatim; appends pending; bumps next_id. Empty blockers can never reject,
    // so addTask is always ok here and the parser stays total. (#118)
    addTask(list, step, []);
  }
  return list;
}
