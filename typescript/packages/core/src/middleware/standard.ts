/**
 * StandardMiddlewareChain — in-memory reference implementation of
 * {@link MiddlewareChain} (spore-core issue #11).
 *
 * Mirrors `rust/crates/spore-core/src/middleware.rs#StandardMiddlewareChain`.
 * Priority ordering, ForceAnotherTurn concatenation, and first-wins
 * SurfaceToHuman semantics are all enforced here. Illegal decisions
 * (`force_another_turn` outside `before_completion`,
 * `surface_to_human` outside `before_tool` / `before_completion`) are
 * surfaced as `halt` so the loop has uniform handling.
 */

import type { HumanRequest, RunResult, SessionId, SessionState, Task } from "../harness/types.js";
import type { ToolCall, ToolResult } from "../model/schemas.js";

import {
  hookAllowsForceAnotherTurn,
  hookAllowsSurfaceToHuman,
  hookIsAfter,
  type HookPoint,
  type Middleware,
  type MiddlewareChain,
  type MiddlewareDecision,
  MiddlewareError,
} from "./types.js";

interface Entry {
  name: string;
  priority: number;
  hooks: HookPoint[];
  middleware: Middleware;
}

function priorityOf(m: Middleware): number {
  return m.priority?.() ?? 0;
}

export class StandardMiddlewareChain implements MiddlewareChain {
  private readonly entries: Entry[] = [];

  register(middleware: Middleware): void {
    const name = middleware.name();
    const hooks = middleware.hooks();
    if (hooks.length === 0) {
      throw MiddlewareError.noHooks(name);
    }
    if (this.entries.some((e) => e.name === name)) {
      throw MiddlewareError.alreadyRegistered(name);
    }
    this.entries.push({
      name,
      priority: priorityOf(middleware),
      hooks: hooks.slice(),
      middleware,
    });
  }

  private eligible(hook: HookPoint): Entry[] {
    const v = this.entries.filter((e) => e.hooks.includes(hook));
    if (hookIsAfter(hook)) {
      // Descending priority; tie-break by name ascending for determinism.
      v.sort((a, b) => {
        if (a.priority !== b.priority) return b.priority - a.priority;
        return a.name.localeCompare(b.name);
      });
    } else {
      v.sort((a, b) => {
        if (a.priority !== b.priority) return a.priority - b.priority;
        return a.name.localeCompare(b.name);
      });
    }
    return v;
  }

  async fireBeforeSession(
    task: Task,
    sessionId: SessionId,
    signal?: AbortSignal,
  ): Promise<MiddlewareDecision> {
    return this.fireSimple("before_session", (m) =>
      m.handle({ kind: "before_session", task, session_id: sessionId }, signal),
    );
  }

  async fireBeforeTurn(
    session: SessionState,
    turnNumber: number,
    signal?: AbortSignal,
  ): Promise<MiddlewareDecision> {
    return this.fireSimple("before_turn", (m) =>
      m.handle({ kind: "before_turn", session, turn_number: turnNumber }, signal),
    );
  }

  async fireBeforeTool(
    calls: ToolCall[],
    turnNumber: number,
    signal?: AbortSignal,
  ): Promise<MiddlewareDecision> {
    return this.fireSimple("before_tool", (m) =>
      m.handle({ kind: "before_tool", calls, turn_number: turnNumber }, signal),
    );
  }

  async fireAfterTool(
    calls: ToolCall[],
    results: ToolResult[],
    signal?: AbortSignal,
  ): Promise<MiddlewareDecision> {
    return this.fireSimple("after_tool", (m) =>
      m.handle({ kind: "after_tool", calls, results }, signal),
    );
  }

  async fireBeforeCompletion(
    response: string,
    turnNumber: number,
    state: SessionState,
    signal?: AbortSignal,
  ): Promise<MiddlewareDecision> {
    const entries = this.eligible("before_completion");
    const injections: string[] = [];
    let anyModified = false;
    for (const entry of entries) {
      const raw = await entry.middleware.handle(
        {
          kind: "before_completion",
          response,
          turn_number: turnNumber,
          session_state: state,
        },
        signal,
      );
      const validated = validateDecision(entry.name, "before_completion", raw);
      if (validated.kind === "halt_illegal") {
        return { kind: "halt", reason: validated.reason };
      }
      const decision = validated.decision;
      switch (decision.kind) {
        case "continue":
          continue;
        case "continue_with_modification":
          anyModified = true;
          continue;
        case "force_another_turn":
          injections.push(decision.inject);
          continue;
        case "halt":
        case "surface_to_human":
          return decision;
      }
    }
    if (injections.length > 0) {
      return { kind: "force_another_turn", inject: injections.join("\n") };
    }
    return anyModified ? { kind: "continue_with_modification" } : { kind: "continue" };
  }

  async fireAfterSession(
    result: RunResult,
    sessionId: SessionId,
    signal?: AbortSignal,
  ): Promise<void> {
    const entries = this.eligible("after_session");
    for (const entry of entries) {
      // After-session decisions are ignored — the session is terminating.
      await entry.middleware.handle(
        { kind: "after_session", result, session_id: sessionId },
        signal,
      );
    }
  }

  /** Linear walk for hooks that don't accumulate ForceAnotherTurn injections. */
  private async fireSimple(
    hook: HookPoint,
    invoke: (m: Middleware) => Promise<MiddlewareDecision>,
  ): Promise<MiddlewareDecision> {
    const entries = this.eligible(hook);
    let anyModified = false;
    for (const entry of entries) {
      const raw = await invoke(entry.middleware);
      const validated = validateDecision(entry.name, hook, raw);
      if (validated.kind === "halt_illegal") {
        return { kind: "halt", reason: validated.reason };
      }
      const decision = validated.decision;
      switch (decision.kind) {
        case "continue":
          continue;
        case "continue_with_modification":
          anyModified = true;
          continue;
        case "force_another_turn":
          // Shouldn't reach here — validateDecision turns into halt_illegal
          // for non-completion hooks. Treat as continue defensively.
          continue;
        case "halt":
        case "surface_to_human":
          return decision;
      }
    }
    return anyModified ? { kind: "continue_with_modification" } : { kind: "continue" };
  }
}

type Validated =
  | { kind: "ok"; decision: MiddlewareDecision }
  | { kind: "halt_illegal"; reason: string };

function validateDecision(name: string, hook: HookPoint, decision: MiddlewareDecision): Validated {
  if (decision.kind === "surface_to_human" && !hookAllowsSurfaceToHuman(hook)) {
    return {
      kind: "halt_illegal",
      reason: MiddlewareError.illegalDecision(name, hook, "SurfaceToHuman").message,
    };
  }
  if (decision.kind === "force_another_turn" && !hookAllowsForceAnotherTurn(hook)) {
    return {
      kind: "halt_illegal",
      reason: MiddlewareError.illegalDecision(name, hook, "ForceAnotherTurn").message,
    };
  }
  return { kind: "ok", decision };
}

// Re-export for callers that want HumanRequest at call sites.
export type { HumanRequest, RunResult };
