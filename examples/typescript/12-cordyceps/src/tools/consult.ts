/**
 * The two consult tools the analysis worker calls to escalate mid-loop
 * (issue #114). Both lower to {@link toolOutput.consult} with a `kind` tag; the
 * `analysis_worker` {@link SubagentTool} mediates by `kind` (research →
 * research_worker, advice → advisor) using the per-kind budgets + overflow
 * policies installed via the subagent's `consultHandlers` map.
 *
 * Neither tool captures any host state — each simply renders its call input into
 * a {@link ConsultRequest} and returns {@link toolOutput.consult}. The worker
 * harness pauses (`RunResult` `consult`) and `SubagentTool` resumes it with the
 * handler's answer (or a `budget_exhausted` message). So these are plain
 * {@link toolRegistry.defineTool} tools — no closed-over state needed.
 */

import { toolOutput, toolRegistry } from "@spore/core";
import { z } from "zod";

type StandardTool = toolRegistry.StandardTool;

/** Routing key for the research consult ladder (→ research_worker, web_search). */
export const KIND_RESEARCH = "research";
/** Routing key for the advice consult ladder (→ advisor, cloud model). */
export const KIND_ADVICE = "advice";

/**
 * Shared input shape for both consult tools: the worker describes where it is
 * stuck and the concrete question it wants answered. `attempts` is advisory —
 * the harness enforces the per-kind budget independently.
 */
const ConsultInput = z.object({
  situation: z
    .string()
    .describe("Free-form description of where you are stuck or uncertain."),
  question: z.string().describe("The concrete question you want answered."),
  attempts: z
    .number()
    .int()
    .nonnegative()
    .default(0)
    .describe("How many times you have already tried (advisory only)."),
});

/**
 * `research_best_practices` → `kind="research"`. Routed to the research worker
 * (web_search). Budget 5, overflow `soft_fail`: on exhaustion the worker resumes
 * with `budget_exhausted` and finishes on general knowledge. Looking up an idiom
 * is normal, not distress, so it never reaches the human.
 */
export function researchBestPracticesTool(): StandardTool {
  return toolRegistry.defineTool({
    name: "research_best_practices",
    description:
      "Ask a research helper to web-search current best practices or idioms " +
      "when you are unsure whether a pattern is a real defect. Pass `situation` " +
      "and a focused `question`. Returns cited findings; use sparingly.",
    input: ConsultInput,
    execute: async (input) =>
      toolOutput.consult({
        kind: KIND_RESEARCH,
        situation: input.situation,
        attempts: input.attempts,
        question: input.question,
      }),
  });
}

/**
 * `consult_advisor` → `kind="advice"`. Routed to the advisor (near-frontier
 * cloud model with `read_file`/`grep`). Budget 3, overflow `escalate_to_human`:
 * on exhaustion the consult converts to `RunResult` `waiting_for_human` and the
 * REPL surfaces the three-choice ladder.
 */
export function consultAdvisorTool(): StandardTool {
  return toolRegistry.defineTool({
    name: "consult_advisor",
    description:
      "Ask a senior advisor agent (a stronger model that can read_file/grep " +
      "the repo) when you are stuck on whether a finding is real or how to rank " +
      "its severity. Pass `situation` and a concrete `question`. Reserve for " +
      "genuine uncertainty — the advisor budget is small.",
    input: ConsultInput,
    execute: async (input) =>
      toolOutput.consult({
        kind: KIND_ADVICE,
        situation: input.situation,
        attempts: input.attempts,
        question: input.question,
      }),
  });
}
