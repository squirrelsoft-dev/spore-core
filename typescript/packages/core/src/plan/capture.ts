/**
 * Plan-artifact capture grammar (spore-core issue #70, Q3).
 *
 * Turns a planner model's `final_response` text into a structured
 * {@link PlanArtifact}. The grammar MUST be byte-identical across all four
 * languages (Rust reference: `rust/crates/spore-core/src/plan.rs`), so it is
 * kept simple, deterministic, and TOTAL: it never throws on malformed input;
 * instead it returns a {@link PlanPhaseError} of kind `unparseable_plan`.
 */

import type { PlanArtifact } from "../hooks/index.js";

import { PlanPhaseError } from "./types.js";

/**
 * Total result of {@link capturePlanArtifact}. Mirrors Rust's
 * `Result<PlanArtifact, PlanPhaseError>` as an idiomatic TS discriminated union
 * so the function never has to throw (R9: deterministic & total).
 */
export type CaptureResult =
  | { ok: true; artifact: PlanArtifact }
  | { ok: false; error: PlanPhaseError };

/**
 * Capture a {@link PlanArtifact} from a planner's `final_response` text.
 *
 * The canonical Q3 grammar:
 *
 * 1. Trim leading/trailing ASCII whitespace.
 * 2. If the trimmed text begins with a triple-backtick fence, strip a single
 *    leading fence line (the opening ``` plus any language tag up to and
 *    including the first newline) and a single trailing ``` fence, then trim
 *    again.
 * 3. Parse the result as a JSON object with `tasks` (required array of JSON
 *    strings, kept verbatim; an empty array is allowed) and `rationale`
 *    (optional string, default `""`).
 *
 * Any deviation → an `unparseable_plan` {@link PlanPhaseError}. Never throws.
 */
export function capturePlanArtifact(finalText: string): CaptureResult {
  const trimmed = trimAsciiWs(finalText);
  const body = stripCodeFence(trimmed);

  let value: unknown;
  try {
    value = JSON.parse(body);
  } catch (e) {
    return err(`invalid JSON: ${e instanceof Error ? e.message : String(e)}`);
  }

  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return err("top-level JSON value is not an object");
  }
  const obj = value as Record<string, unknown>;

  if (!("tasks" in obj)) {
    return err("missing required field `tasks`");
  }
  const tasksValue = obj["tasks"];
  if (!Array.isArray(tasksValue)) {
    return err("field `tasks` is not an array");
  }

  const tasks: string[] = [];
  for (let i = 0; i < tasksValue.length; i++) {
    const element = tasksValue[i];
    if (typeof element !== "string") {
      return err(`element ${i} of \`tasks\` is not a string`);
    }
    // Verbatim — do NOT trim or filter.
    tasks.push(element);
  }

  // `rationale` is optional; default "". If present it must be a string.
  let rationale = "";
  if ("rationale" in obj && obj["rationale"] !== undefined) {
    const r = obj["rationale"];
    if (typeof r !== "string") {
      return err("field `rationale` is not a string");
    }
    rationale = r;
  }

  return { ok: true, artifact: { tasks, rationale } };
}

/**
 * Capture a {@link PlanArtifact}, falling back to a deterministic PROSE REPAIR
 * when the strict canonical grammar ({@link capturePlanArtifact}) fails.
 *
 * A planner sometimes emits its plan JSON wrapped in prose ("Here is the plan:
 * `{...}` — let me know…") instead of as a bare object, so the strict grammar
 * rejects it. This fallback extracts the FIRST balanced top-level JSON object
 * embedded in the text ({@link extractEmbeddedJsonObject}) and re-parses THAT
 * with the same canonical grammar. It is a pure, always-on fallback: it runs
 * ONLY after the strict path fails, so it can never change a plan the strict
 * grammar already accepts — it can only turn a hard failure into a success.
 * When no embedded object repairs cleanly, the ORIGINAL strict error is
 * returned (it is the more informative diagnostic).
 *
 * Like the strict grammar, this MUST be byte-identical across all four
 * languages — the embedded-object scan is a deterministic ASCII char walk.
 */
export function capturePlanArtifactWithRepair(finalText: string): CaptureResult {
  const strict = capturePlanArtifact(finalText);
  if (strict.ok) {
    return strict;
  }
  const candidate = extractEmbeddedJsonObject(finalText);
  if (candidate === undefined) {
    return strict;
  }
  // Re-parse the extracted object with the SAME canonical grammar; if it still
  // does not parse, surface the original strict error.
  const repaired = capturePlanArtifact(candidate);
  return repaired.ok ? repaired : strict;
}

/**
 * Extract the first balanced top-level JSON object (`{ … }`) embedded in `text`,
 * or `undefined` if there is no balanced object. Scans from the first `{`,
 * tracking brace depth while respecting JSON string literals (a `"` opens/closes
 * a string; a `\` escapes the next char inside one), and returns the slice up to
 * and including the matching `}`. Braces inside strings do not affect depth.
 *
 * The structural characters `{` `}` `"` `\` are all single-code-unit ASCII, so
 * walking JS string code units yields the same result as Rust's byte scan: the
 * scan branches only on these ASCII chars and slices only at `{`/`}` positions,
 * leaving any multi-byte content inside string literals intact. MUST be
 * byte-identical across all four languages.
 */
export function extractEmbeddedJsonObject(text: string): string | undefined {
  const start = text.indexOf("{");
  if (start < 0) {
    return undefined;
  }
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
    } else if (ch === '"') {
      inString = true;
    } else if (ch === "{") {
      depth += 1;
    } else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        return text.slice(start, i + 1);
      }
    }
  }
  return undefined;
}

function err(message: string): CaptureResult {
  return { ok: false, error: PlanPhaseError.unparseablePlan(message) };
}

/**
 * ASCII-whitespace set. Matches `' '`, `'\t'`, `'\n'`, `'\r'`, plus form-feed
 * and vertical-tab — kept to the ASCII set so trimming is byte-identical
 * cross-language (mirrors the Rust `is_ascii_ws` predicate). Deliberately NOT
 * `String.prototype.trim`, which also strips Unicode whitespace.
 */
const ASCII_WS = new Set([" ", "\t", "\n", "\r", "\u000b", "\u000c"]);

function isAsciiWs(ch: string | undefined): boolean {
  return ch !== undefined && ASCII_WS.has(ch);
}

function trimAsciiWs(s: string): string {
  let start = 0;
  let end = s.length;
  while (start < end && isAsciiWs(s[start])) start++;
  while (end > start && isAsciiWs(s[end - 1])) end--;
  return s.slice(start, end);
}

/**
 * Strip a single leading ``` / ```json fence line and a single trailing ```
 * fence, if the (already-trimmed) input opens with a triple-backtick fence.
 * Returns the inner body, re-trimmed. If the input does not open with a fence
 * it is returned unchanged. Mirrors the Rust `strip_code_fence`.
 */
export function stripCodeFence(trimmed: string): string {
  if (!trimmed.startsWith("```")) {
    return trimmed;
  }
  const afterOpen = trimmed.slice(3);

  // Drop the rest of the opening fence line (the optional language tag) up to
  // and including the first newline. A fence with no newline at all has no body
  // to parse; let JSON parsing reject it downstream.
  const nl = afterOpen.indexOf("\n");
  const bodyStart = nl >= 0 ? afterOpen.slice(nl + 1) : afterOpen;

  // Strip a single trailing closing fence if present, then re-trim.
  const bodyEndTrimmed = trimEndAsciiWs(bodyStart);
  const body = bodyEndTrimmed.endsWith("```")
    ? bodyEndTrimmed.slice(0, bodyEndTrimmed.length - 3)
    : bodyStart;

  return trimAsciiWs(body);
}

function trimEndAsciiWs(s: string): string {
  let end = s.length;
  while (end > 0 && isAsciiWs(s[end - 1])) end--;
  return s.slice(0, end);
}
