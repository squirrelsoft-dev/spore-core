/**
 * RealToolRegistry — the bridge between the two `ToolRegistry` interfaces
 * (spore-core issue #91).
 *
 * This is **the production wiring** for running catalogue / {@link Tool}-based
 * tools inside the harness — not test scaffolding. It graduated out of the
 * scenarios module (`@spore/tools`) into this blessed surface so callers don't
 * import it from a scenarios-flavoured location; the old location re-exports it
 * for back-compat.
 */

import type { ToolCall, ToolSchema as ModelToolSchema } from "../model/schemas.js";
import type {
  SandboxProvider,
  SessionId,
  ToolOutput,
  ToolRegistry as HarnessToolRegistry,
} from "../harness/types.js";
import type { MemoryStore, ProjectId, RunStore } from "../storage/types.js";

import { ToolContext, type ToolSchema as RegistryToolSchema } from "./types.js";
import { dispatchErrorMessage } from "./types.js";
import type { StandardToolRegistry } from "./standard.js";

/** Project a registry {@link RegistryToolSchema} onto the model-facing
 *  {@link ModelToolSchema} (`parameters` → `input_schema`). */
export function toModelSchema(schema: RegistryToolSchema): ModelToolSchema {
  return {
    name: schema.name,
    description: schema.description,
    input_schema: schema.parameters,
  };
}

/**
 * Bridges the harness-loop {@link HarnessToolRegistry} onto the canonical
 * {@link "./types.js".ToolRegistry} (a populated {@link StandardToolRegistry}).
 *
 * This is **the production wiring** for running catalogue / {@link "./types.js".Tool}-based
 * tools inside the harness — not test scaffolding. The harness loop calls
 * `dispatch(call) -> ToolOutput` with no sandbox or storage; this bridge
 * forwards to the inner registry's `dispatch(call, sandbox, ctx)`, threading the
 * {@link SandboxProvider} and a per-run {@link ToolContext} (storage seam, #75)
 * it was constructed with. A {@link "./types.js".DispatchError} becomes a
 * **recoverable** error {@link ToolOutput} so the loop appends it as a tool
 * result and lets the agent adapt rather than halting — S4 depends on this. No
 * bridged tool is marked always-halt.
 *
 * It is built **once per run**: `sessionId`, `projectId`, `runStore`, and
 * `memoryStore` are injected at construction (the run's {@link SessionId} is only
 * known at `run()`-time) and used to build the {@link ToolContext} forwarded on
 * every dispatch. `projectId` (#142) is the STABLE durable namespace; `sessionId`
 * is the per-window ephemeral key. {@link "../harness/index.js".HarnessBuilder}
 * wires this automatically when catalogue tools are added via `.tool()` /
 * `.tools()`; construct it directly only when supplying your own
 * {@link StandardToolRegistry}.
 */
export class RealToolRegistry implements HarnessToolRegistry {
  private readonly _schemas: ModelToolSchema[];
  private readonly ctx: ToolContext;

  constructor(
    private readonly inner: StandardToolRegistry,
    private readonly sandbox: SandboxProvider,
    sessionId: SessionId,
    projectId: ProjectId,
    runStore: RunStore,
    memoryStore: MemoryStore,
  ) {
    // Snapshot the model-facing schemas (sorted by name; activeSchemas already
    // sorts) once at construction; the catalogue is fixed for a run.
    this._schemas = inner.activeSchemas(null).map(toModelSchema);
    // Build the storage seam once per run from the injected session + project +
    // stores.
    this.ctx = new ToolContext(sessionId, projectId, runStore, memoryStore);
  }

  /** The model-facing tool schemas, sorted by name. */
  modelSchemas(): ModelToolSchema[] {
    return this._schemas.slice();
  }

  /**
   * The {@link ToolContext} this bridge threads into every dispatch — exposes the
   * `sessionId`, `runStore`, and (#78) `memoryStore` seams it was wired with.
   * Lets callers verify the storage seams are live.
   */
  toolContext(): ToolContext {
    return this.ctx;
  }

  async dispatch(call: ToolCall, signal?: AbortSignal): Promise<ToolOutput> {
    const outcome = await this.inner.dispatch(call, this.sandbox, this.ctx, signal);
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
