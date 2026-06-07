/**
 * Public re-exports for the `tasklist` component (spore-core issue #71).
 *
 * The single mutating tool (`task_list`) lives in `@spore/tools`; this module
 * owns the types, the transition matrix, and the mutation helpers it drives.
 * Persistence is the storage seam (#75): the tool reads/writes the
 * {@link "../storage/types.js".RunStore} keyed by `SessionId` under
 * {@link TASK_LIST_EXTRAS_KEY}.
 */

export {
  TASK_LIST_EXTRAS_KEY,
  TaskStatusSchema,
  type TaskStatus,
  TaskSchema,
  type Task,
  TaskListSchema,
  type TaskList,
  defaultTaskList,
  serializeTaskList,
  parseTaskList,
  TaskListError,
  type TaskListErrorKind,
  type BlockerRejection,
  blockerRejectionMessage,
  taskListErrorMessage,
  type MutationResult,
  type AddResult,
  validateTransition,
  addTask,
  wouldCreateCycle,
  updateTask,
  completeTask,
  planArtifactToTaskList,
  STEP_LEDGER_MAX_ENTRIES,
  STEP_LEDGER_ELISION_MARKER,
  StepLedgerEntrySchema,
  type StepLedgerEntry,
  pushStepLedger,
  renderStepLedger,
  nextReady,
  transitiveBlockers,
  transitiveDependents,
  hasCycle,
} from "./types.js";
