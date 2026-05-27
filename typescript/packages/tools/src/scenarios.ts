/**
 * End-to-end scenario assembly (issue #57).
 *
 * Reusable wiring shared by the `e2e-agent` CLI AND the hermetic scenario
 * tests, so a live run ({@link OllamaModelInterface} + {@link ModelAgent}) and
 * an offline run (mock agent + scripted tool registry) drive the *same* code
 * path. {@link buildScenario} is generic over the harness {@link Agent} +
 * {@link HarnessToolRegistry} interfaces, so the only difference between live
 * and mock mode is which agent/registry you inject.
 *
 * ## Architectural gaps closed here (mirroring the Rust reference)
 *
 * - {@link RealToolRegistry} bridges the two `ToolRegistry` interfaces: the
 *   harness loop calls `harness.ToolRegistry.dispatch(call)` (no sandbox arg),
 *   while the real tools live behind `toolRegistry.ToolRegistry.dispatch(call,
 *   sandbox)`. The bridge owns the inner {@link StandardToolRegistry} +
 *   {@link SandboxProvider} and forwards, mapping a {@link DispatchError} onto a
 *   recoverable {@link ToolOutput} error so the loop appends it and the agent
 *   adapts (S4 depends on this).
 * - {@link SchemaInjectingContextManager} decorates any harness context manager
 *   so the assembled context's `tools` is populated from the registry's tool
 *   schemas (sorted by name). The compaction adapter's `assemble` returns an
 *   empty tool list, so without this decorator the model never sees any tools
 *   and can never emit a tool call in live mode.
 * - {@link FailingTool} (`flaky_op`) always returns a *recoverable* error.
 * - {@link CompleteOnFinalResponse} is a non-test termination policy: it lets
 *   the loop succeed as soon as the agent produces a final response.
 */

import {
  type Agent,
  cacheProvider as coreCacheProvider,
  type Context,
  type ContextManager as HarnessContextManager,
  type CompactionTurn,
  context as coreContext,
  type ModelInterface,
  type ObservabilityProvider,
  type SandboxProvider,
  type SessionState as HarnessState,
  type SessionId,
  type TaskId,
  type Task,
  type TerminationDecision,
  type TerminationPolicy,
  type ToolCall,
  type ToolOutput,
  type ToolResultRecord,
  type ToolSchema as ModelToolSchema,
  type ToolRegistry as HarnessToolRegistry,
  toolRegistry,
  HarnessBuilder,
  type StandardHarness,
} from "@spore/core";

import { ListDirTool, ReadFileTool, WriteFileTool } from "./fs.js";
import { BashCommandTool } from "./exec.js";

type RegistryToolSchema = toolRegistry.ToolSchema;
type Tool = toolRegistry.Tool;

const { StandardContextManager } = coreContext;
const { dispatchErrorMessage } = toolRegistry;

// Re-export so callers can build a registry directly.
export { ReadFileTool, WriteFileTool, ListDirTool, BashCommandTool };

// ============================================================================
// Registry → model schema conversion
// ============================================================================

/** Project a registry {@link RegistryToolSchema} onto the model-facing
 *  {@link ModelToolSchema} (`parameters` → `input_schema`). */
export function toModelSchema(schema: RegistryToolSchema): ModelToolSchema {
  return {
    name: schema.name,
    description: schema.description,
    input_schema: schema.parameters,
  };
}

// ============================================================================
// RealToolRegistry — bridge between the two ToolRegistry interfaces
// ============================================================================

/**
 * Bridges the harness-loop {@link HarnessToolRegistry} onto the canonical
 * {@link toolRegistry.ToolRegistry} ({@link StandardToolRegistry}).
 *
 * A {@link DispatchError} becomes a **recoverable** error {@link ToolOutput} so
 * the loop appends it as a tool result rather than halting — S4 depends on
 * this. No bridged tool is marked always-halt.
 */
export class RealToolRegistry implements HarnessToolRegistry {
  private readonly _schemas: ModelToolSchema[];

  constructor(
    private readonly inner: toolRegistry.StandardToolRegistry,
    private readonly sandbox: SandboxProvider,
  ) {
    // Snapshot the model-facing schemas (sorted by name; activeSchemas already
    // sorts) once at construction; the catalog is fixed for a scenario run.
    this._schemas = inner.activeSchemas(null).map(toModelSchema);
  }

  /** The model-facing tool schemas, sorted by name. */
  modelSchemas(): ModelToolSchema[] {
    return this._schemas.slice();
  }

  async dispatch(call: ToolCall, signal?: AbortSignal): Promise<ToolOutput> {
    const outcome = await this.inner.dispatch(call, this.sandbox, signal);
    if (outcome.ok) return outcome.result.output;
    return {
      kind: "error",
      message: `dispatch failed: ${dispatchErrorMessage(outcome.error)}`,
      // Recoverable so the loop appends the error and lets the agent adapt.
      recoverable: true,
    };
  }

  isAlwaysHalt(_toolName: string): boolean {
    // No bridged tool is always-halt — S4 needs recoverable failure.
    return false;
  }

  schemas(): ModelToolSchema[] {
    return this._schemas.slice();
  }
}

// ============================================================================
// SchemaInjectingContextManager — fills assemble().tools from the registry
// ============================================================================

/**
 * Decorates a harness {@link HarnessContextManager}, delegating every seam
 * method to the inner manager but injecting the registry's tool schemas (sorted
 * by name) into the assembled context's `tools`. Without it the compaction
 * adapter surfaces no tools and the model can never emit a tool call in live
 * mode.
 */
export class SchemaInjectingContextManager implements HarnessContextManager {
  private readonly tools: ModelToolSchema[];

  constructor(
    private readonly inner: HarnessContextManager,
    tools: ModelToolSchema[],
  ) {
    this.tools = tools.slice().sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  }

  async assemble(session: HarnessState, task: Task, signal?: AbortSignal): Promise<Context> {
    const ctx = await this.inner.assemble(session, task, signal);
    ctx.tools = this.tools.slice();
    return ctx;
  }

  appendToolResult(session: HarnessState, result: ToolResultRecord): Promise<void> {
    return this.inner.appendToolResult(session, result);
  }

  appendUserMessage(session: HarnessState, text: string): Promise<void> {
    return this.inner.appendUserMessage(session, text);
  }

  shouldCompact(session: HarnessState): boolean {
    return this.inner.shouldCompact(session);
  }

  prepareCompactionTurn(session: HarnessState): CompactionTurn | undefined {
    return this.inner.prepareCompactionTurn?.(session);
  }

  injectMissingItems(context: Context, missing: string[]): void {
    this.inner.injectMissingItems?.(context, missing);
  }

  applyCompaction(session: HarnessState, summary: string): void {
    this.inner.applyCompaction?.(session, summary);
  }

  tokenBudgetUsed(session: HarnessState): number | undefined {
    return this.inner.tokenBudgetUsed?.(session);
  }
}

// ============================================================================
// FailingTool — deliberately-failing recoverable tool (S4)
// ============================================================================

/**
 * A tool that always fails with a *recoverable* error. Used by S4 to prove the
 * loop surfaces a tool error to the agent and lets it adapt rather than
 * crashing or hanging. Must NOT be always-halt.
 */
export class FailingTool implements Tool {
  static readonly NAME = "flaky_op";
  readonly name = FailingTool.NAME;

  static schema(): RegistryToolSchema {
    return {
      name: FailingTool.NAME,
      description: "A flaky operation that fails when it is called",
      parameters: {
        type: "object",
        properties: { reason: { type: "string" } },
      },
      annotations: {
        read_only: false,
        destructive: false,
        idempotent: true,
        open_world: false,
      },
    };
  }

  async execute(): Promise<ToolOutput> {
    return {
      kind: "error",
      message: "flaky_op is unavailable right now; try a different approach",
      recoverable: true,
    };
  }
}

// ============================================================================
// CompleteOnFinalResponse — non-test termination policy
// ============================================================================

/**
 * Termination policy that lets the loop complete as soon as the agent produces
 * a final response (always `continue`, which the harness interprets as "accept
 * the final response and succeed").
 */
export class CompleteOnFinalResponse implements TerminationPolicy {
  async evaluate(): Promise<TerminationDecision> {
    return { kind: "continue" };
  }
}

// ============================================================================
// Real tool registry construction
// ============================================================================

/**
 * Build a {@link StandardToolRegistry} populated with the real read/write/list/
 * bash tools plus the {@link FailingTool}. Shared by every scenario so the
 * agent always sees the same catalog. Registration errors are programming
 * errors (duplicate/invalid schema) — surfaced loudly via throw.
 */
export function buildRealToolRegistry(): toolRegistry.StandardToolRegistry {
  const registry = new toolRegistry.StandardToolRegistry();
  const reg = (tool: Tool, schema: RegistryToolSchema): void => {
    const err = registry.register(tool, schema);
    if (err) throw new Error(`register ${schema.name}: ${toolRegistry.registrationErrorMessage(err)}`);
  };
  reg(new ReadFileTool(), ReadFileTool.schema());
  reg(new WriteFileTool(), WriteFileTool.schema());
  reg(new ListDirTool(), ListDirTool.schema());
  reg(new BashCommandTool(), BashCommandTool.schema());
  reg(new FailingTool(), FailingTool.schema());
  return registry;
}

// ============================================================================
// Rich context-manager assembly (live compaction)
// ============================================================================

/**
 * Build a real compaction-capable context manager: a
 * {@link StandardContextManager} wrapped in the {@link StandardCompactionAdapter}
 * (`intoHarnessAdapter`). Generic over the model so live mode passes the Ollama
 * model and tests pass a mock.
 */
export function buildRichContextManager(
  model: ModelInterface,
  cache: coreCacheProvider.CacheProvider,
  config: coreContext.CompactionConfig,
): HarnessContextManager {
  return coreContext.intoHarnessAdapter(new StandardContextManager(model, cache, config));
}

/**
 * Seed a harness {@link HarnessState} with rich compaction state for S3: a
 * small window, a budget near the threshold, and a history longer than
 * `preserve_recent_n` so compaction fires mid-run. The session can then
 * compact, continue, and compact again because the token-accounting fix
 * decrements the budget on each compaction.
 */
export function seedCompactionState(
  session: HarnessState,
  taskInstruction: string,
  sessionId: SessionId,
  taskId: TaskId,
  windowLimit: number,
  tokenBudgetUsed: number,
  historyLen: number,
): void {
  const rich = coreContext.newSessionState(sessionId, taskId, taskInstruction);
  rich.window_limit = windowLimit;
  rich.token_budget_used = tokenBudgetUsed;
  rich.message_history = Array.from({ length: historyLen }, (_, i) => ({
    role: (i % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
    content: {
      type: "text" as const,
      text:
        `history message ${i}: progress notes on the payment service deploy with ` +
        `enough content to carry a meaningful token estimate for reclamation`,
    },
  }));
  coreContext.seedRichState(session, rich);
}

// ============================================================================
// Scenario builders
// ============================================================================

/** The scenario id, parsed from the CLI arg `s1`..`s4`. */
export type ScenarioId = "s1" | "s2" | "s3" | "s4";

/** Parse `s1`..`s4` (case-insensitive). */
export function parseScenarioId(s: string): ScenarioId | undefined {
  const v = s.trim().toLowerCase();
  return v === "s1" || v === "s2" || v === "s3" || v === "s4" ? v : undefined;
}

/** The default prompt that drives this scenario. */
export function scenarioPrompt(id: ScenarioId): string {
  switch (id) {
    case "s1":
      return (
        "Read the file input.txt, transform its contents to UPPERCASE, write the " +
        "result to output.txt, then read output.txt back to confirm it was written. " +
        "Use the read_file, write_file, and bash_command tools. When done, reply DONE."
      );
    case "s2":
      return (
        "Create a file notes.md containing a TODO list with one item: 'set up the " +
        "project'. Use write_file. Reply DONE when written."
      );
    case "s3":
      return (
        "Summarize the long conversation so far and continue working on the deploy of " +
        "the payment service. Reply DONE when finished."
      );
    case "s4":
      return (
        "Call the flaky_op tool. If it fails, do not give up: write a file " +
        "recovered.txt explaining that flaky_op failed and how you adapted, using " +
        "write_file. Reply DONE when finished."
      );
  }
}

/**
 * Assemble a {@link StandardHarness} for the given scenario from injected
 * components. Generic over the agent and tool registry so live mode and mock
 * mode share one code path.
 *
 * `toolSchemas` are injected into every assembled context (sorted by name) via
 * {@link SchemaInjectingContextManager}. Pass the registry's schemas in live
 * mode, or an empty array in mock mode. `observability`, when present, is
 * injected directly; `undefined` runs with no observability.
 */
export function buildScenario(args: {
  scenario: ScenarioId;
  agent: Agent;
  tools: HarnessToolRegistry;
  sandbox: SandboxProvider;
  contextManager: HarnessContextManager;
  terminationPolicy: TerminationPolicy;
  toolSchemas: ModelToolSchema[];
  observability?: ObservabilityProvider;
}): StandardHarness {
  const contextManager = new SchemaInjectingContextManager(args.contextManager, args.toolSchemas);
  let builder = new HarnessBuilder(
    args.agent,
    args.tools,
    args.sandbox,
    contextManager,
    args.terminationPolicy,
  );
  if (args.observability) builder = builder.observability(args.observability);
  return builder.build();
}
