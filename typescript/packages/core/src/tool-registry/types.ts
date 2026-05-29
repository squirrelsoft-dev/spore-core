/**
 * ToolRegistry — canonical types (spore-core issue #4).
 *
 * Mirrors `rust/crates/spore-core/src/tool_registry.rs`. Same field names
 * (snake_case on the wire), same enum variants, same validation semantics —
 * shared fixtures must produce identical outcomes.
 *
 * Cross-language note: `ToolCall` is reused from {@link "../model/schemas.js"}.
 * The spec field names map as `tool_name → name`, `parameters → input`.
 * `ToolOutput`, `ToolResult` shapes mirror the loop's forward declarations in
 * `../harness/types.ts`; the canonical `ToolResult` here is `{ call_id, output }`.
 */

import { z } from "zod";

import type { ToolCall } from "../model/schemas.js";
import type { SandboxProvider, SandboxViolation, ToolOutput } from "../harness/types.js";
import type { SessionId } from "../harness/types.js";
import type { RunStore } from "../storage/types.js";

// ============================================================================
// ToolContext — the storage seam handed to every tool (#75)
// ============================================================================

/**
 * The per-dispatch storage seam handed to every {@link Tool.execute} call,
 * alongside (but separate from) the {@link SandboxProvider}. It carries the
 * minimum a tool needs to persist durable state via the storage layer:
 *
 *   - `sessionId` — the run's {@link SessionId}, the key namespace for
 *     {@link RunStore}.
 *   - `runStore`  — the {@link RunStore} domain of the configured storage
 *     provider.
 *
 * It is a **class** (not a tuple/pair) so future fields can be added without
 * breaking the {@link Tool.execute} signature again. The {@link SandboxProvider}
 * is intentionally NOT folded in here — storage is additive; tools still receive
 * the sandbox as its own parameter (some tools need the filesystem sandbox and
 * no storage).
 */
export class ToolContext {
  /**
   * @param sessionId The session id keying this run's persisted state.
   * @param runStore  The run-store domain a tool persists durable state through.
   */
  constructor(
    readonly sessionId: SessionId,
    readonly runStore: RunStore,
  ) {}
}

// ============================================================================
// ToolAnnotations & ToolSchema
// ============================================================================

export const ToolAnnotationsSchema = z.object({
  read_only: z.boolean().default(false),
  destructive: z.boolean().default(false),
  idempotent: z.boolean().default(false),
  open_world: z.boolean().default(false),
});
export type ToolAnnotations = z.infer<typeof ToolAnnotationsSchema>;

export function defaultToolAnnotations(): ToolAnnotations {
  return {
    read_only: false,
    destructive: false,
    idempotent: false,
    open_world: false,
  };
}

/**
 * Canonical schema for a registered tool. Distinct from `model.ToolSchema`
 * (the slimmer subset shipped to the LLM) — this one carries
 * {@link ToolAnnotations} and is the registry-side type.
 */
export const ToolSchemaSchema = z.object({
  name: z.string(),
  description: z.string(),
  parameters: z.unknown(),
  annotations: ToolAnnotationsSchema.default(defaultToolAnnotations()),
});
export type ToolSchema = z.infer<typeof ToolSchemaSchema>;

// ============================================================================
// TaskPhase & ToolSet
// ============================================================================

export const TaskPhaseSchema = z.enum([
  "initialization",
  "planning",
  "execution",
  "verification",
  "cleanup",
]);
export type TaskPhase = z.infer<typeof TaskPhaseSchema>;

export const ToolSetSchema = z.object({
  name: z.string(),
  tools: z.array(z.string()),
  phase: TaskPhaseSchema.nullable().optional(),
});
export type ToolSet = z.infer<typeof ToolSetSchema>;

// ============================================================================
// ToolResult (canonical, distinct from harness's ToolResultRecord)
// ============================================================================

export interface ToolResult {
  call_id: string;
  output: ToolOutput;
}

// ============================================================================
// Errors — discriminant `kind` for spec parity with Rust's `#[serde(tag)]`.
// ============================================================================

export type RegistrationError =
  | { kind: "InvalidSchema"; tool: string; reason: string }
  | { kind: "DuplicateName"; tool: string }
  | { kind: "ConflictingAnnotations"; tool: string; reason: string };

/** Marker class so domain errors can be `throw`n where appropriate. */
export class RegistrationErrorException extends Error {
  override readonly name = "RegistrationErrorException";
  constructor(readonly error: RegistrationError) {
    super(registrationErrorMessage(error));
  }
}

export function registrationErrorMessage(e: RegistrationError): string {
  switch (e.kind) {
    case "InvalidSchema":
      return `invalid schema for tool ${e.tool}: ${e.reason}`;
    case "DuplicateName":
      return `tool ${e.tool} already registered`;
    case "ConflictingAnnotations":
      return `conflicting annotations for tool ${e.tool}: ${e.reason}`;
  }
}

export type DispatchError =
  | { kind: "UnregisteredTool"; name: string }
  | { kind: "SchemaValidationFailed"; tool: string; reason: string }
  | { kind: "SandboxViolation"; violation: SandboxViolation }
  | { kind: "ToolExecutionFailed"; tool: string; error: string };

export function dispatchErrorMessage(e: DispatchError): string {
  switch (e.kind) {
    case "UnregisteredTool":
      return `unregistered tool: ${e.name}`;
    case "SchemaValidationFailed":
      return `schema validation failed for ${e.tool}: ${e.reason}`;
    case "SandboxViolation":
      return `sandbox violation: ${e.violation.kind}`;
    case "ToolExecutionFailed":
      return `tool ${e.tool} failed: ${e.error}`;
  }
}

// ============================================================================
// Tool interface
// ============================================================================

/**
 * A single tool implementation. Tools are stateless and receive a
 * {@link SandboxProvider} (environment seam) and a {@link ToolContext} (storage
 * seam) on every dispatch.
 */
export interface Tool {
  /** Tool name — must match the registered {@link ToolSchema} `name`. */
  readonly name: string;

  /**
   * `true` for `SubagentTool`. Defaults to `false`. Used by
   * {@link ToolRegistry.hasSubagentTools} to enforce the depth-1 rule
   * at construction time, not at dispatch time.
   */
  readonly isSubagentTool?: boolean;

  /**
   * Execute the tool with validated input. The {@link SandboxProvider} is the
   * only path to the environment; the {@link ToolContext} is the only path to
   * durable storage ({@link RunStore}, keyed by the run's {@link SessionId}).
   * Most tools ignore `ctx`.
   */
  execute(
    call: ToolCall,
    sandbox: SandboxProvider,
    ctx: ToolContext,
    signal?: AbortSignal,
  ): Promise<ToolOutput>;

  /**
   * `true` if this tool may produce a large output that should be routed
   * through {@link SandboxProvider.handleLargeOutput}. Defaults to `false`.
   */
  mayProduceLargeOutput?(): boolean;
}

// ============================================================================
// ToolRegistry interface
// ============================================================================

/** Per-registry result for dispatch — either a {@link ToolResult} or {@link DispatchError}. */
export type DispatchOutcome =
  | { ok: true; result: ToolResult }
  | { ok: false; error: DispatchError };

/**
 * Canonical {@link ToolRegistry} interface. The concrete implementation is
 * {@link StandardToolRegistry}. The harness's forward-declared
 * `harness.ToolRegistry` is a narrower loop-only shape and will eventually be
 * replaced by this trait once issue #5/#6 land.
 */
export interface ToolRegistry {
  /** Register a tool — validates schema at registration time. */
  register(tool: Tool, schema: ToolSchema): RegistrationError | null;

  /** Register a named {@link ToolSet} grouping. */
  registerSet(set: ToolSet): RegistrationError | null;

  /** Return schemas for tools active in the given phase (always sorted by name). */
  activeSchemas(phase?: TaskPhase | null): ToolSchema[];

  /** Dispatch one tool call. `ctx` is the storage seam threaded to the tool. */
  dispatch(
    call: ToolCall,
    sandbox: SandboxProvider,
    ctx: ToolContext,
    signal?: AbortSignal,
  ): Promise<DispatchOutcome>;

  /** Dispatch multiple calls — concurrent where annotations permit. */
  dispatchAll(
    calls: ToolCall[],
    sandbox: SandboxProvider,
    ctx: ToolContext,
    signal?: AbortSignal,
  ): Promise<DispatchOutcome[]>;

  /** True if any registered tool has {@link Tool.isSubagentTool} = `true`. */
  hasSubagentTools(): boolean;
}
