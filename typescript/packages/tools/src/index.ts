/**
 * @spore/tools — standard Tool implementations (spore-core issue #5).
 *
 * Built on @spore/core: tools implement the canonical {@link Tool} interface
 * and run against a {@link SandboxProvider}. The package extends the
 * SandboxProvider interface with defaulted methods (`executeCommand`,
 * `handleLargeOutput`, `resolvePath`) — see `sandbox-defaults.ts`.
 */

export {
  type ToolExecutionError,
  toolExecutionErrorToOutput,
} from "./errors.js";

export {
  LARGE_OUTPUT_THRESHOLD,
  DEFAULT_HEAD_TOKENS,
  DEFAULT_TAIL_TOKENS,
  defaultExecuteCommand,
  defaultHandleLargeOutput,
  defaultResolvePath,
  finishWithPossibleTruncation,
  isSandboxViolation,
  sbExecuteCommand,
  sbHandleLargeOutput,
  sbResolvePath,
} from "./sandbox-defaults.js";

export * from "./params.js";

export {
  DeleteFileTool,
  ListDirTool,
  MoveFileTool,
  ReadFileTool,
  WriteFileTool,
} from "./fs.js";

export { BashCommandTool, ExecTool, RunTestsTool } from "./exec.js";
export { EditFileTool } from "./edit.js";
export { FindFilesTool, GrepFilesTool, GrepTool } from "./search.js";
export { SendMessageTool } from "./message.js";
export {
  applyWebFetchRange,
  UrlPolicy,
  validateFetchUrl,
  WebFetchTool,
  WebSearchTool,
  WebSearchConfigError,
  type WebSearchConfig,
  type SearchMethod,
} from "./web.js";
export { TODO_STORE_KEY, TodoWriteTool } from "./todo.js";
export { MEMORY_LOCAL_REJECTED_MESSAGE, MemoryTool } from "./memory.js";
export {
  AbortTool,
  AskUserQuestionTool,
  EnterPlanModeTool,
  ExitPlanModeTool,
} from "./control.js";
export { StandardTools } from "./catalogue.js";
export {
  GitCommitTool,
  GitDiffTool,
  GitLogTool,
  GitResetTool,
  GitStatusTool,
} from "./git.js";
export { HttpGetTool, HttpPostTool } from "./http.js";
export { TaskListTool } from "./tasklist.js";
export {
  type BuildError as SubagentBuildError,
  type ContextSharing,
  type SubagentToolConfig,
  SubagentTool,
  SubagentToolBuildError,
} from "./subagent.js";
export {
  buildRealToolRegistry,
  buildRichContextManager,
  buildScenario,
  CompleteOnFinalResponse,
  FailingTool,
  parseScenarioId,
  RealToolRegistry,
  type ScenarioId,
  scenarioPrompt,
  SchemaInjectingContextManager,
  seedCompactionState,
  toModelSchema,
} from "./scenarios.js";
