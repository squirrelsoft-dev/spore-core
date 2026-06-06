/**
 * `load_skill(skill_id)` — activate a skill for the rest of the session.
 *
 * This tool closes over the shared {@link guideRegistry.StandardGuideRegistry}
 * (via {@link toolRegistry.defineTool}'s closure body) so it can validate ids.
 * On execute it:
 *
 * 1. confirms the named skill exists in the registry (rejects unknown ids,
 *    recoverably, so the model can pick a real one from the manifest);
 * 2. reads `runStore["active_skills"]` → `string[]`, appends the id (deduped),
 *    and writes it back;
 * 3. returns a short confirmation.
 *
 * The active set is then re-injected every turn by
 * {@link SkillInjectingContextManager} — no new `ToolOutput` variant, all
 * storage-backed (issue #115 "flavor B").
 */

import {
  guideRegistry,
  toolOutput,
  toolRegistry,
  type storage,
} from "@spore/core";
import { z } from "zod";

import { ACTIVE_SKILLS_KEY } from "../skills.js";

type StandardTool = toolRegistry.StandardTool;
type StandardGuideRegistry = guideRegistry.StandardGuideRegistry;

const { newGuideQuery } = guideRegistry;

/** The registered name of the tool. */
export const LOAD_SKILL_NAME = "load_skill";

const LoadSkillInput = z.object({
  skill_id: z
    .string()
    .min(1)
    .describe('The id (name) of the skill to activate, e.g. "audit".'),
});

/** Build the `load_skill` tool, closing over the shared registry. */
export function loadSkillTool(registry: StandardGuideRegistry): StandardTool {
  return toolRegistry.defineTool({
    name: LOAD_SKILL_NAME,
    description:
      "Activate a skill by id so its full procedure stays in your context for " +
      "the rest of the session. Choose an id from the manifest of available " +
      "skills.",
    input: LoadSkillInput,
    execute: async (input, _sandbox, ctx) => {
      const skillId = input.skill_id.trim();

      // 1. Confirm the skill exists. `select` returns only active guides ranked
      //    by overlap; a broad query with the id surfaces it. Reject unknown ids
      //    recoverably so the model can choose a real one.
      let known = false;
      try {
        const guides = await registry.select(newGuideQuery(skillId));
        known = guides.some((g) => g.id.asString() === skillId);
      } catch {
        known = false;
      }
      if (!known) {
        return toolOutput.error(
          `unknown skill '${skillId}'. Pick one of the skills listed in the manifest.`,
        );
      }

      // 2. Append to runStore["active_skills"] (dedup).
      let active: string[];
      try {
        const value = await ctx.runStore.get(ctx.sessionId, ACTIVE_SKILLS_KEY);
        active = Array.isArray(value)
          ? value.filter((v): v is string => typeof v === "string")
          : [];
      } catch (e) {
        return toolOutput.error(
          `load_skill: could not read active set: ${errMessage(e)}`,
        );
      }
      if (!active.includes(skillId)) active.push(skillId);

      const stored: storage.JsonValue = active;
      try {
        await ctx.runStore.put(ctx.sessionId, ACTIVE_SKILLS_KEY, stored);
      } catch (e) {
        return toolOutput.error(
          `load_skill: could not persist active set: ${errMessage(e)}`,
        );
      }

      // 3. Confirm. The body is now injected every turn by the context manager.
      return toolOutput.success(
        `Loaded skill '${skillId}'. Its procedure is now active — follow it.`,
      );
    },
  });
}

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
