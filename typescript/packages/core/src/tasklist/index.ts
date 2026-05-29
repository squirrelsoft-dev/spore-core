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
  taskListErrorMessage,
  type MutationResult,
  validateTransition,
  addTask,
  updateTask,
  completeTask,
  planArtifactToTaskList,
} from "./types.js";
