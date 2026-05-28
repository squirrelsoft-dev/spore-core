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
  type Message,
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
import { BashCommandTool, ExecTool } from "./exec.js";

type RegistryToolSchema = toolRegistry.ToolSchema;
type Tool = toolRegistry.Tool;

const { StandardContextManager } = coreContext;
const { dispatchErrorMessage } = toolRegistry;

// Re-export so callers can build a registry directly.
export { ReadFileTool, WriteFileTool, ListDirTool, ExecTool, BashCommandTool };

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
 * Operational system prompt for the live agent. The compaction adapter's
 * `assemble` produces a context with **no system prompt** (it has no
 * `ContextSources` to render one), so without this the model receives only the
 * task as a user message and no guidance on how to behave. The three rules
 * target the failure modes observed with small local models: describing actions
 * instead of taking them, passing stringified arguments, and declaring success
 * without checking the result.
 */
export const AGENT_SYSTEM_PROMPT = `\
You are an autonomous agent that completes tasks by calling the provided tools. \
Follow these rules:

1. ACT, DON'T DESCRIBE. To make something happen, call the appropriate tool. \
Writing a shell command, code snippet, or file contents into your text reply \
does NOT run it — only a real tool call has any effect. When a task asks you to \
produce a file or a result, call the tool that performs the action and let the \
tool do the work; never paste the command, code, or expression you *would* run \
as if it were the finished result.

2. USE CORRECTLY-TYPED ARGUMENTS. Pass tool arguments as typed JSON: booleans \
as true/false (not "true"), numbers as 12 (not "12"), lists as ["a"] (not \
"[\\"a\\"]"). Quoted-string scalars where a bool/number/array is expected will \
be rejected.

3. VERIFY BEFORE FINISHING. Before replying DONE, confirm your work actually \
satisfies the request. If you wrote a file, read it back with read_file and \
check its contents are exactly what was asked. If they do not match, fix it and \
verify again. Only reply DONE once you have verified the result is correct.`;

/**
 * Decorates a harness {@link HarnessContextManager}, delegating every seam
 * method to the inner manager but injecting the registry's tool schemas (sorted
 * by name) into the assembled context's `tools` and prepending
 * {@link AGENT_SYSTEM_PROMPT}. The compaction adapter's `assemble` returns an
 * empty tool list and no system prompt, so without this decorator the model
 * never sees any tools (and can never emit a tool call) nor any operational
 * guidance in live mode.
 */
export class SchemaInjectingContextManager implements HarnessContextManager {
  private readonly tools: ModelToolSchema[];

  constructor(
    private readonly inner: HarnessContextManager,
    tools: ModelToolSchema[],
  ) {
    this.tools = tools
      .slice()
      .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  }

  async assemble(
    session: HarnessState,
    task: Task,
    signal?: AbortSignal,
  ): Promise<Context> {
    const ctx = await this.inner.assemble(session, task, signal);
    ctx.tools = this.tools.slice();
    // Prepend the operational system prompt. The adapter's assemble yields
    // none, so the model would otherwise get no guidance. Guard against
    // duplicates so a resumed/seeded session that already leads with a system
    // message isn't given two.
    const hasSystem = ctx.messages[0]?.role === "system";
    if (!hasSystem) {
      ctx.messages.unshift({
        role: "system",
        content: { type: "text", text: AGENT_SYSTEM_PROMPT },
      });
    }
    return ctx;
  }

  appendToolResult(
    session: HarnessState,
    result: ToolResultRecord,
  ): Promise<void> {
    return this.inner.appendToolResult(session, result);
  }

  appendAssistantMessage(
    session: HarnessState,
    message: Message,
  ): Promise<void> {
    // Delegate to the inner manager. Without this delegation the loop (which
    // holds this outer wrapper) would hit the optional method's absence and
    // silently skip recording the assistant turn — the fix would be dead.
    return (
      this.inner.appendAssistantMessage?.(session, message) ?? Promise.resolve()
    );
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
 * Build a {@link StandardToolRegistry} for `scenario`. The base catalog is
 * always `read_file`, `write_file`, `list_dir`, `exec`, and {@link FailingTool}
 * (`flaky_op`). The real shell tool `bash_command` is added ONLY for scenario
 * `s5` — S1/S2 measure reasoning + act-don't-describe, and a live model handed
 * a shell could shortcut S1 with `cat … | tr … > …` without demonstrating the
 * intended behavior. `exec` is safe everywhere because it cannot pipe or
 * redirect. Registration errors are programming errors (duplicate/invalid
 * schema) — surfaced loudly via throw.
 */
export function buildRealToolRegistry(
  scenario: ScenarioId,
): toolRegistry.StandardToolRegistry {
  const registry = new toolRegistry.StandardToolRegistry();
  const reg = (tool: Tool, schema: RegistryToolSchema): void => {
    const err = registry.register(tool, schema);
    if (err)
      throw new Error(
        `register ${schema.name}: ${toolRegistry.registrationErrorMessage(err)}`,
      );
  };
  reg(new ReadFileTool(), ReadFileTool.schema());
  reg(new WriteFileTool(), WriteFileTool.schema());
  reg(new ListDirTool(), ListDirTool.schema());
  reg(new ExecTool(), ExecTool.schema());
  reg(new FailingTool(), FailingTool.schema());
  if (scenario === "s5") {
    reg(new BashCommandTool(), BashCommandTool.schema());
  }
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
  return coreContext.intoHarnessAdapter(
    new StandardContextManager(model, cache, config),
  );
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

/** The scenario id, parsed from the CLI arg `s1`..`s5`. */
export type ScenarioId = "s1" | "s2" | "s3" | "s4" | "s5";

/** Parse `s1`..`s5` (case-insensitive). */
export function parseScenarioId(s: string): ScenarioId | undefined {
  const v = s.trim().toLowerCase();
  return v === "s1" || v === "s2" || v === "s3" || v === "s4" || v === "s5"
    ? v
    : undefined;
}

/** The default prompt that drives this scenario. */
export function scenarioPrompt(id: ScenarioId): string {
  switch (id) {
    case "s1":
      return (
        "Complete this task step by step, using the provided tools:\n" +
        "1. Call read_file to read the contents of input.txt. Use the exact " +
        "text it returns — do not invent or substitute any text.\n" +
        "2. Take that exact text and rewrite it with every lowercase letter " +
        "changed to its capital form, keeping all other characters, spaces, " +
        "and punctuation the same.\n" +
        "3. Call write_file with path 'output.txt' and content set to the " +
        "uppercased text from step 2 — the literal capital letters themselves. " +
        "The content must be the transformed words from input.txt, NOT a shell " +
        "command, NOT a $(...) expression, and NOT any code.\n" +
        "4. Call read_file on output.txt and check its contents equal the " +
        "uppercased text from step 2.\n" +
        "Reply DONE only once output.txt contains input.txt's contents in all " +
        "capital letters."
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
    case "s5":
      return (
        "Transform input.txt into output.txt with every lowercase letter " +
        "uppercased, using the shell.\n" +
        "1. Call bash_command with a real shell pipeline that reads input.txt, " +
        "uppercases it, and writes output.txt — e.g. " +
        "`cat input.txt | tr a-z A-Z > output.txt`. This is exactly what the " +
        "bash_command tool is for: it runs your script via /bin/sh -c, so pipes " +
        "(|) and redirects (>) work.\n" +
        "2. Call read_file on output.txt and check its contents are input.txt's " +
        "text in all capital letters.\n" +
        "Reply DONE only once output.txt contains the uppercased text."
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
  const contextManager = new SchemaInjectingContextManager(
    args.contextManager,
    args.toolSchemas,
  );
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
