/**
 * SC-8 (#157) — `HarnessBuilder` presets: `codingAgent` + `hillClimber`.
 *
 * Mirrors the Rust `HarnessBuilder::{coding_agent, hill_climber}` reference
 * (commit `6f39933`, `rust/crates/spore-core/src/harness.rs`). Both compose
 * {@link HarnessBuilder.conversational} with the Phase 1–2 knobs so a consumer
 * collapses to one call (the plan's "presets carry the friendly behaviour", Q1).
 *
 * ## Why these live in `@spore/tools`, not on `HarnessBuilder` in `@spore/core`
 *
 * In Rust, `HarnessBuilder` and `StandardTools::coding_set()` live in the SAME
 * crate, so `coding_agent` calls the catalogue directly. In TypeScript the layering
 * is one-way: the concrete coding tool catalogue ({@link StandardTools.codingSet})
 * lives in `@spore/tools`, which DEPENDS ON `@spore/core` (where `HarnessBuilder`
 * lives). `@spore/core` therefore cannot reach the concrete tools without a
 * circular dependency. `@spore/tools` is the one layer that sees BOTH
 * {@link HarnessBuilder} and {@link StandardTools}, so the presets live here and
 * return a ready-to-`.build()` {@link HarnessBuilder} — same composition, same
 * defaults, same behaviour as the Rust constructors.
 */

import {
  autoContinue,
  HarnessBuilder,
  WorkspaceScopedSandbox,
  type EscalationMode,
  type MetricEvaluator,
  type ModelInterface,
  type WorkspaceConfig,
} from "@spore/core";

import { StandardTools } from "./catalogue.js";

/**
 * Built-in system prompt for {@link codingAgent}: a coding agent that ACTS
 * through the workspace tools (rather than describing what it would do) and
 * narrates each step to the user via `send_message`. Exposed so a consumer can
 * extend it; override it wholesale with {@link HarnessBuilder.systemPrompt}.
 *
 * COPIED VERBATIM from the Rust `HarnessBuilder::CODING_AGENT_SYSTEM_PROMPT`
 * constant for cross-language parity — the runtime string (Rust's `\`-continued
 * literal collapses each continuation's leading whitespace to a single space).
 */
export const CODING_AGENT_SYSTEM_PROMPT =
  "You are a coding agent working inside a sandboxed workspace directory. " +
  "Explore with list_dir, read_file, grep, and find_files; create and change files with " +
  "write_file and edit_file; run commands with bash. Use relative paths only. " +
  "Act using tools — do not just describe what you would do. When the task is done, " +
  "reply with a short summary of what you changed. " +
  "The user CANNOT see your reasoning or your tool calls — they only see the messages you " +
  "send with the `send_message` tool and your final reply. So before (or as) you act, " +
  "call `send_message` with one short sentence saying what you are about to do, in " +
  "PARALLEL with the tool that does the work, so narration never costs an extra round trip. " +
  "Keep each message to a single short sentence.";

/**
 * Default per-scope auto-continue cap for the autonomous presets
 * ({@link codingAgent} / {@link hillClimber}): grant up to this many extra step
 * budgets at an `escalate` point before the run gives up. Mirrors the hand-rolled
 * drive loop the consumers used (the `12-cordyceps` example's `MAX_AUTO_CONTINUES`).
 * Override the whole policy with {@link HarnessBuilder.escalationMode}.
 */
export const PRESET_MAX_AUTO_GRANTS = 10;

/**
 * Steps granted on each auto-continue for the autonomous presets (the
 * `12-cordyceps` example's `CONTINUE_STEPS`). See {@link PRESET_MAX_AUTO_GRANTS}.
 */
export const PRESET_STEPS_PER_GRANT = 25;

/**
 * `auto_continue` {@link EscalationMode} with the preset defaults
 * ({@link PRESET_MAX_AUTO_GRANTS} × {@link PRESET_STEPS_PER_GRANT}, no `onGrant`
 * observer). The "autonomous but capped" policy both autonomous presets share
 * (SC-5). Mirrors the Rust `HarnessBuilder::preset_auto_continue`.
 */
function presetAutoContinue(): EscalationMode {
  return autoContinue({
    maxGrants: PRESET_MAX_AUTO_GRANTS,
    stepsPerGrant: PRESET_STEPS_PER_GRANT,
  });
}

/**
 * Assemble an autonomous **coding agent** over a workspace directory (SC-8) —
 * the looper preset.
 *
 * Builds on {@link HarnessBuilder.conversational} and wires the bits a coding
 * agent always needs: a **read-write** {@link WorkspaceScopedSandbox} rooted at
 * `workspace`, the full {@link StandardTools.codingSet} (read/write/edit/list/
 * grep/find + `bash` + `send_message` + web/memory/task-list), the built-in
 * {@link CODING_AGENT_SYSTEM_PROMPT}, and an `auto_continue`
 * {@link EscalationMode} (autonomous-but-capped — it keeps working through a spent
 * step budget instead of pausing, so there is no consumer drive loop to
 * hand-roll; SC-5).
 *
 * **Window sizing (SC-4/SC-6).** Size the model's context window ONCE on the
 * model before passing it in (e.g. `OllamaModelInterface#withContextWindow`): the
 * preset's `conversational` context manager auto-derives its compaction budget
 * from `provider().context_window`, so one call sizes the window and no manual
 * {@link HarnessBuilder.contextManager} is needed.
 *
 * **THROWS** a {@link WorkspaceScopedSandbox} `BuildError` if `workspace` can't be
 * resolved (it must exist and canonicalize — the sandbox requirement), matching
 * the Rust `Result<Self, BuildError>` fallible constructor. The strategy is
 * per-run: pass a `react` / `plan_execute` `Task` to `run()`.
 *
 * Mirrors `HarnessBuilder::coding_agent` in `rust/crates/spore-core/src/harness.rs`.
 *
 * ```ts
 * const model = new OllamaModelInterface("gemma4:e4b").withContextWindow(256_000);
 * const harness = codingAgent(model, "/path/to/project").build();
 * ```
 */
export function codingAgent(
  model: ModelInterface,
  workspace: string,
): HarnessBuilder {
  // Read-write workspace-scoped sandbox over `workspace`. The default
  // WorkspaceConfig is read-write (`read_only` defaults to false) — the
  // equivalent of Rust's `WorkspaceConfig::scoped(workspace)`. Construction
  // throws a `BuildError` when the root can't be resolved/canonicalized.
  const config: WorkspaceConfig = { root: workspace };
  const sandbox = new WorkspaceScopedSandbox(config);
  return HarnessBuilder.conversational(model)
    .sandbox(sandbox)
    .tools(StandardTools.codingSet())
    .systemPrompt(CODING_AGENT_SYSTEM_PROMPT)
    .escalationMode(presetAutoContinue());
}

/**
 * Assemble an autonomous **hill-climbing agent** (SC-8) — the cordyceps preset.
 *
 * Builds on {@link HarnessBuilder.conversational} and registers the scoring
 * `evaluator` (required for the `hill_climbing` loop strategy) under the default
 * handle, plus an `auto_continue` {@link EscalationMode} (autonomous-but-capped;
 * SC-5) so a spent per-iteration build budget keeps working instead of pausing.
 *
 * Unlike {@link codingAgent} this does NOT install a sandbox or tools —
 * hill-climbing workspaces vary (some climb a prose artifact, some climb files),
 * and the build task's system prompt is task-specific. Add them with
 * {@link HarnessBuilder.sandbox} / {@link HarnessBuilder.tools} /
 * {@link HarnessBuilder.systemPrompt} as the climb requires. Size the model's
 * window on the model first (SC-4/SC-6), as in {@link codingAgent}.
 *
 * The `hill_climbing` config (direction / `max_stagnation` / per-iteration
 * budget) lives on the per-run `Task`'s strategy; the iteration ceiling is the
 * task's `max_turns`.
 *
 * Mirrors `HarnessBuilder::hill_climber` in `rust/crates/spore-core/src/harness.rs`.
 *
 * ```ts
 * const model = new OllamaModelInterface("gemma4:e4b").withContextWindow(256_000);
 * // Add a workspace `.sandbox(..)` + `.tools(..)` for the climb as it requires.
 * const harness = hillClimber(model, evaluator).build();
 * ```
 */
export function hillClimber(
  model: ModelInterface,
  evaluator: MetricEvaluator,
): HarnessBuilder {
  return HarnessBuilder.conversational(model)
    .metricEvaluator(evaluator)
    .escalationMode(presetAutoContinue());
}
