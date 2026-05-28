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
export { FindFilesTool, GrepFilesTool } from "./search.js";
export {
  GitCommitTool,
  GitDiffTool,
  GitLogTool,
  GitResetTool,
  GitStatusTool,
} from "./git.js";
export { HttpGetTool, HttpPostTool } from "./http.js";
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
