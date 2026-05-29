/**
 * Public re-exports for the `tasklist` component (spore-core issue #71).
 *
 * The single mutating tool (`task_list`) lives in `@spore/tools`; this module
 * owns the types, the transition matrix, the mutation helpers, and the
 * disk-persistence helpers it drives.
 */

export {
  TASK_LIST_EXTRAS_KEY,
  TASK_LIST_PATH,
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
  type LoadError,
  type StoreError,
  type LoadResult,
  type StoreResult,
  loadTaskList,
  storeTaskList,
} from "./types.js";
