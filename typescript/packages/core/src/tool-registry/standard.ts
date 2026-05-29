/**
 * StandardToolRegistry — canonical in-memory {@link ToolRegistry} (spore-core
 * issue #4). Mirrors `rust/crates/spore-core/src/tool_registry.rs` —
 * same rules, same dispatch ordering, same fixture outcomes.
 */

import type { ToolCall } from "../model/schemas.js";
import type { SandboxProvider, ToolOutput } from "../harness/types.js";

import {
  type DispatchError,
  type DispatchOutcome,
  type RegistrationError,
  type TaskPhase,
  type Tool,
  type ToolContext,
  type ToolRegistry,
  type ToolResult,
  type ToolSchema,
  type ToolSet,
} from "./types.js";

interface Registered {
  tool: Tool;
  schema: ToolSchema;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export class StandardToolRegistry implements ToolRegistry {
  private readonly tools = new Map<string, Registered>();
  private readonly sets: ToolSet[] = [];

  register(tool: Tool, schema: ToolSchema): RegistrationError | null {
    if (tool.name !== schema.name) {
      return {
        kind: "InvalidSchema",
        tool: schema.name,
        reason: `tool name \`${tool.name}\` does not match schema name \`${schema.name}\``,
      };
    }
    const schemaErr = validateSchema(schema);
    if (schemaErr) return schemaErr;
    const annoErr = validateAnnotations(schema);
    if (annoErr) return annoErr;
    if (this.tools.has(schema.name)) {
      return { kind: "DuplicateName", tool: schema.name };
    }
    this.tools.set(schema.name, { tool, schema });
    return null;
  }

  registerSet(set: ToolSet): RegistrationError | null {
    if (set.name.length === 0) {
      return {
        kind: "InvalidSchema",
        tool: set.name,
        reason: "tool set name must not be empty",
      };
    }
    if (this.sets.some((s) => s.name === set.name)) {
      return { kind: "DuplicateName", tool: set.name };
    }
    this.sets.push(set);
    return null;
  }

  activeSchemas(phase?: TaskPhase | null): ToolSchema[] {
    let out: ToolSchema[];
    if (phase == null) {
      out = Array.from(this.tools.values(), (r) => r.schema);
    } else {
      // Union of: sets matching this phase OR sets with no phase
      // (always-active). If no set matches, fall back to the full catalog —
      // registering zero sets must not silently mask every tool.
      const matching = this.sets.filter((s) => s.phase == null || s.phase === phase);
      if (matching.length === 0) {
        out = Array.from(this.tools.values(), (r) => r.schema);
      } else {
        const names = new Set<string>();
        for (const s of matching) for (const t of s.tools) names.add(t);
        out = [];
        for (const n of names) {
          const r = this.tools.get(n);
          if (r) out.push(r.schema);
        }
      }
    }
    // Cache-stability: schemas always sorted by name (see spec, cache rules).
    out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    return out;
  }

  async dispatch(
    call: ToolCall,
    sandbox: SandboxProvider,
    ctx: ToolContext,
    signal?: AbortSignal,
  ): Promise<DispatchOutcome> {
    const reg = this.tools.get(call.name);
    if (!reg) {
      return { ok: false, error: { kind: "UnregisteredTool", name: call.name } };
    }

    // Sandbox validation. PathEscape / NetworkViolation are Layer 1 —
    // the registry surfaces the violation as a DispatchError so the
    // harness can route it.
    const violation = await sandbox.validate(call, signal);
    if (violation != null) {
      return { ok: false, error: { kind: "SandboxViolation", violation } };
    }

    const inputErr = validateInput(reg.schema, call);
    if (inputErr) return { ok: false, error: inputErr };

    const output: ToolOutput = await reg.tool.execute(call, sandbox, ctx, signal);
    const result: ToolResult = { call_id: call.id, output };
    return { ok: true, result };
  }

  async dispatchAll(
    calls: ToolCall[],
    sandbox: SandboxProvider,
    ctx: ToolContext,
    signal?: AbortSignal,
  ): Promise<DispatchOutcome[]> {
    // Classify each call. Unknown tools are scheduled sequentially so their
    // error surfaces deterministically alongside any other sequential failures.
    const classifications: boolean[] = calls.map((call) => {
      const r = this.tools.get(call.name);
      if (!r) return false;
      const a = r.schema.annotations;
      return a.read_only && !a.destructive && !a.open_world;
    });

    const concurrentIdx: number[] = [];
    const sequentialIdx: number[] = [];
    classifications.forEach((c, i) => {
      if (c) concurrentIdx.push(i);
      else sequentialIdx.push(i);
    });

    const results: (DispatchOutcome | undefined)[] = new Array<DispatchOutcome | undefined>(
      calls.length,
    ).fill(undefined);

    if (concurrentIdx.length > 0) {
      const outs = await Promise.all(
        concurrentIdx.map((i) => this.dispatch(calls[i]!, sandbox, ctx, signal)),
      );
      concurrentIdx.forEach((slot, j) => {
        results[slot] = outs[j];
      });
    }

    for (const i of sequentialIdx) {
      results[i] = await this.dispatch(calls[i]!, sandbox, ctx, signal);
    }

    return results.map((r) => {
      if (!r) throw new Error("dispatchAll slot unfilled");
      return r;
    });
  }

  hasSubagentTools(): boolean {
    for (const r of this.tools.values()) {
      if (r.tool.isSubagentTool === true) return true;
    }
    return false;
  }
}

// ----------------------------------------------------------------------------
// Validation helpers
// ----------------------------------------------------------------------------

function validateSchema(schema: ToolSchema): RegistrationError | null {
  if (schema.name.length === 0) {
    return {
      kind: "InvalidSchema",
      tool: schema.name,
      reason: "name must not be empty",
    };
  }
  // Basic structural check: parameters must be a JSON object with a `type`
  // key. Full JSON Schema validation is intentionally not bundled here —
  // see validateInput for per-call enforcement.
  if (!isPlainObject(schema.parameters)) {
    return {
      kind: "InvalidSchema",
      tool: schema.name,
      reason: "parameters must be a JSON object",
    };
  }
  if (!("type" in schema.parameters)) {
    return {
      kind: "InvalidSchema",
      tool: schema.name,
      reason: "parameters must declare a top-level `type`",
    };
  }
  return null;
}

function validateAnnotations(schema: ToolSchema): RegistrationError | null {
  const a = schema.annotations;
  if (a.read_only && a.destructive) {
    return {
      kind: "ConflictingAnnotations",
      tool: schema.name,
      reason: "read_only and destructive are mutually exclusive",
    };
  }
  return null;
}

/**
 * Best-effort per-call schema validation. Checks that any `required` fields
 * declared on the parameter schema are present in the call's `input` object.
 * Deeper JSON Schema validation can be plugged in later.
 */
function validateInput(schema: ToolSchema, call: ToolCall): DispatchError | null {
  if (!isPlainObject(call.input)) {
    return {
      kind: "SchemaValidationFailed",
      tool: schema.name,
      reason: "input must be a JSON object",
    };
  }
  if (!isPlainObject(schema.parameters)) return null;
  const required = (schema.parameters as Record<string, unknown>).required;
  if (Array.isArray(required)) {
    for (const field of required) {
      if (typeof field === "string" && !(field in call.input)) {
        return {
          kind: "SchemaValidationFailed",
          tool: schema.name,
          reason: `missing required field \`${field}\``,
        };
      }
    }
  }
  return null;
}
