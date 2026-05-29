/**
 * Tier-3 control tools (#81): tools that drive the harness via the escalation /
 * clarification protocols rather than returning ordinary data.
 *
 * - {@link EnterPlanModeTool} (`enter_plan_mode`) → `escalate`
 *   `{ signal: { kind: "enter_plan_mode", context } }`.
 * - {@link ExitPlanModeTool} (`exit_plan_mode`) → `escalate`
 *   `{ signal: { kind: "exit_plan_mode", plan } }`. The plan is a structured
 *   tool param deserialized DIRECTLY into the existing
 *   {@link "@spore/core".PlanArtifact} (issue #81, Q4a — no stub).
 * - {@link AskUserQuestionTool} (`ask_user_question`) → `awaiting_clarification`
 *   `{ question, options }` (issue #81, Q4b). The harness loop pauses with a
 *   `clarification` `HumanRequest`.
 * - {@link AbortTool} (`abort`) → `escalate` `{ signal: { kind: "abort", reason } }`.
 */

import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import {
  AbortParamsSchema,
  AskUserQuestionParamsSchema,
  EnterPlanModeParamsSchema,
  ExitPlanModeParamsSchema,
  parseParams,
} from "./params.js";

const CONTROL_ANNOTATIONS = {
  read_only: false,
  destructive: false,
  idempotent: false,
  open_world: false,
} as const;

// ============================================================================
// EnterPlanMode
// ============================================================================

export class EnterPlanModeTool implements Tool {
  static readonly NAME = "enter_plan_mode";
  readonly name = EnterPlanModeTool.NAME;

  static schema(): ToolSchema {
    return {
      name: EnterPlanModeTool.NAME,
      description: "Request entry into plan mode, seeding the planner with context",
      parameters: {
        type: "object",
        properties: { context: { type: "string" } },
      },
      annotations: { ...CONTROL_ANNOTATIONS },
    };
  }

  async execute(
    call: ToolCall,
    _sandbox: SandboxProvider,
    _ctx: ToolContext,
  ): Promise<ToolOutput> {
    const p = parseParams(EnterPlanModeParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    return {
      kind: "escalate",
      signal: { kind: "enter_plan_mode", context: p.value.context ?? "" },
    };
  }
}

// ============================================================================
// ExitPlanMode
// ============================================================================

export class ExitPlanModeTool implements Tool {
  static readonly NAME = "exit_plan_mode";
  readonly name = ExitPlanModeTool.NAME;

  static schema(): ToolSchema {
    return {
      name: ExitPlanModeTool.NAME,
      description: "Submit the produced plan and request exit from plan mode",
      parameters: {
        type: "object",
        properties: {
          plan: {
            type: "object",
            properties: {
              tasks: { type: "array", items: { type: "string" } },
              rationale: { type: "string" },
            },
            required: ["tasks"],
          },
        },
        required: ["plan"],
      },
      annotations: { ...CONTROL_ANNOTATIONS },
    };
  }

  async execute(
    call: ToolCall,
    _sandbox: SandboxProvider,
    _ctx: ToolContext,
  ): Promise<ToolOutput> {
    const p = parseParams(ExitPlanModeParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    return {
      kind: "escalate",
      signal: {
        kind: "exit_plan_mode",
        plan: {
          tasks: p.value.plan.tasks,
          rationale: p.value.plan.rationale ?? "",
        },
      },
    };
  }
}

// ============================================================================
// AskUserQuestion
// ============================================================================

export class AskUserQuestionTool implements Tool {
  static readonly NAME = "ask_user_question";
  readonly name = AskUserQuestionTool.NAME;

  static schema(): ToolSchema {
    return {
      name: AskUserQuestionTool.NAME,
      description: "Ask the user a clarifying question (optionally with fixed choices)",
      parameters: {
        type: "object",
        properties: {
          question: { type: "string" },
          options: { type: "array", items: { type: "string" } },
        },
        required: ["question"],
      },
      annotations: { ...CONTROL_ANNOTATIONS },
    };
  }

  async execute(
    call: ToolCall,
    _sandbox: SandboxProvider,
    _ctx: ToolContext,
  ): Promise<ToolOutput> {
    const p = parseParams(AskUserQuestionParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    return {
      kind: "awaiting_clarification",
      question: p.value.question,
      options: p.value.options,
    };
  }
}

// ============================================================================
// Abort
// ============================================================================

export class AbortTool implements Tool {
  static readonly NAME = "abort";
  readonly name = AbortTool.NAME;

  static schema(): ToolSchema {
    return {
      name: AbortTool.NAME,
      description: "Request a graceful abort of the run with a reason",
      parameters: {
        type: "object",
        properties: { reason: { type: "string" } },
        required: ["reason"],
      },
      annotations: { ...CONTROL_ANNOTATIONS },
    };
  }

  async execute(
    call: ToolCall,
    _sandbox: SandboxProvider,
    _ctx: ToolContext,
  ): Promise<ToolOutput> {
    const p = parseParams(AbortParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    return { kind: "escalate", signal: { kind: "abort", reason: p.value.reason } };
  }
}
