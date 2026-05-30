/**
 * Standard {@link CompletionCheck} implementations — spore-core issue #43.
 *
 * Mirrors `rust/crates/spore-core/src/termination.rs` (same issue) for
 * cross-language consistency:
 *
 *   - {@link AlwaysComplete}        — re-export of {@link NullCompletionCheck}.
 *   - {@link FeatureListCheck}      — reads `feature_list.json` under the
 *                                     snapshot's `workspace_root`.
 *   - {@link TestSuiteCheck}        — runs an external test command via an
 *                                     injected {@link SandboxProvider}.
 *   - {@link QuestionAnsweredCheck} — LLM-as-judge over the agent's last
 *                                     assistant text.
 *   - {@link SqlResultCheck}        — validates the most recent SQL tool
 *                                     result in `state.messages`.
 *
 * Reasons returned by each check are plain strings injected verbatim into
 * the next turn's context by the harness. Phrasing matches the Rust
 * reference so the shared `fixtures/completion_check/*.json` fixtures
 * round-trip identically across all four language targets.
 */

import { readFile } from "node:fs/promises";
import { isAbsolute, join } from "node:path";

import type { SandboxProvider } from "../harness/types.js";
import type { Content, Message, ModelRequest } from "../model/schemas.js";
import type { ModelInterface } from "../model/interface.js";

import { NullCompletionCheck } from "./types.js";
import type { CompletionCheck, SessionStateSnapshot } from "./types.js";

// ============================================================================
// AlwaysComplete — spec alias from issue #43
// ============================================================================

/**
 * Always-complete check. Returns `null` immediately — the task is
 * considered done the moment the agent claims done. Use for single-turn
 * tasks where the model's self-assessment is sufficient.
 *
 * Alias of {@link NullCompletionCheck}; the two have identical observable
 * behavior. Exported under both names for spec parity (issue #43) and
 * historical compatibility (issue #13).
 */
export const AlwaysComplete = NullCompletionCheck;
export type AlwaysComplete = NullCompletionCheck;

// ============================================================================
// FeatureListCheck
// ============================================================================

interface FeatureEntry {
  name: string;
  passes: boolean;
}

/**
 * Reads `.spore/feature_list.json` under the snapshot's `workspace_root` (or an
 * absolute path if `path` is absolute). Returns a reason listing
 * incomplete feature names if any entry has `passes: false`; returns
 * `null` when all entries pass.
 *
 * File schema: a JSON array of `{ "name": string, "passes": boolean }`.
 * Missing or unreadable file is treated as incomplete (so the agent
 * learns to create it).
 */
export class FeatureListCheck implements CompletionCheck {
  /**
   * Default location: `<workspace_root>/.spore/feature_list.json` (issue #58,
   * B2 — the canonical `.spore/`-prefixed path, agreeing with the Ralph
   * strategy's completion check).
   */
  constructor(readonly path: string = ".spore/feature_list.json") {}

  async check(state: SessionStateSnapshot): Promise<string | null> {
    const full = isAbsolute(this.path) ? this.path : join(state.workspace_root, this.path);
    let raw: string;
    try {
      raw = await readFile(full, "utf-8");
    } catch {
      return `${this.path} missing`;
    }
    let entries: FeatureEntry[];
    try {
      entries = JSON.parse(raw) as FeatureEntry[];
    } catch (e) {
      return `${this.path} invalid JSON: ${(e as Error).message}`;
    }
    const incomplete = entries.filter((e) => !e.passes).map((e) => e.name);
    if (incomplete.length === 0) return null;
    return `incomplete features: ${incomplete.join(", ")}`;
  }

  description(): string {
    return `feature list at ${this.path}`;
  }
}

// ============================================================================
// TestSuiteCheck
// ============================================================================

/**
 * Runs an external test command via an injected {@link SandboxProvider}.
 * Returns `null` if exit code is 0 and the command did not time out;
 * otherwise returns a reason containing the trailing portion of
 * stderr (or stdout, if stderr is empty) so the next turn knows what
 * failed.
 *
 * `command` is parsed shell-style: the first whitespace-separated token is
 * the program; the remainder become args. For more complex invocations,
 * callers should build a wrapper script and invoke it instead.
 *
 * The check requires `sandbox.executeCommand` to be defined; if it is
 * not, the check returns a reason explaining that. Construction is
 * intentionally awkward — production callers wire a real sandbox.
 */
export class TestSuiteCheck implements CompletionCheck {
  constructor(
    readonly command: string,
    readonly workingDir: string,
    /** Timeout, milliseconds. */
    readonly timeoutMs: number,
    readonly sandbox: SandboxProvider,
  ) {}

  async check(_state: SessionStateSnapshot, signal?: AbortSignal): Promise<string | null> {
    const parts = this.command.split(/\s+/).filter((p) => p.length > 0);
    if (parts.length === 0) return "empty test command";
    const [program, ...args] = parts;
    if (this.sandbox.executeCommand == null) {
      return "sandbox does not support executeCommand";
    }
    const result = await this.sandbox.executeCommand(
      program as string,
      args,
      this.workingDir,
      this.timeoutMs,
      signal,
    );
    if ("kind" in result) {
      // SandboxViolation
      return `sandbox refused test command: ${JSON.stringify(result)}`;
    }
    if (result.exit_code === 0 && !result.timed_out) return null;
    const stderrTail = tailLines(result.stderr, 20);
    const tail = stderrTail.trim().length === 0 ? tailLines(result.stdout, 20) : stderrTail;
    return `test suite failed (exit ${result.exit_code}, timed_out=${result.timed_out}):\n${tail}`;
  }

  description(): string {
    return `test suite: \`${this.command}\` in ${this.workingDir}`;
  }
}

function tailLines(s: string, n: number): string {
  const lines = s.split("\n");
  const start = Math.max(0, lines.length - n);
  return lines.slice(start).join("\n");
}

// ============================================================================
// QuestionAnsweredCheck
// ============================================================================

/**
 * LLM-as-judge: asks a judge {@link ModelInterface} whether the agent's
 * final assistant response actually answered the original question.
 *
 * The judge is invoked with a fixed system prompt instructing it to
 * begin its reply with either `ANSWERED: YES` or `ANSWERED: NO`. The
 * check returns `null` iff the first line, uppercased, starts with
 * `ANSWERED: YES`.
 *
 * Spec note: issue #43 lists `judge_model: ModelConfig`, but
 * `ModelConfig` is a thin descriptor — actually invoking a judge
 * requires a `ModelInterface`. TypeScript takes the interface directly
 * (no dyn-compatibility issue exists here, unlike Rust).
 */
export class QuestionAnsweredCheck implements CompletionCheck {
  rubric: string | null;

  constructor(
    readonly judge: ModelInterface,
    readonly originalQuestion: string,
    rubric: string | null = null,
  ) {
    this.rubric = rubric;
  }

  withRubric(rubric: string): this {
    this.rubric = rubric;
    return this;
  }

  async check(state: SessionStateSnapshot, signal?: AbortSignal): Promise<string | null> {
    const agentResponse = lastAssistantText(state.state.messages) ?? "<no agent response>";
    const rubricClause = this.rubric != null ? `\n\nRubric:\n${this.rubric}` : "";
    const userText =
      `Question:\n${this.originalQuestion}\n\nAgent's final response:\n${agentResponse}\n\n` +
      "Did the agent's response answer the question? Reply with the first line " +
      "`ANSWERED: YES` or `ANSWERED: NO`, then a brief reason on the next line." +
      rubricClause;

    const request: ModelRequest = {
      messages: [
        {
          role: "system",
          content: {
            type: "text",
            text:
              "You are an evaluation judge. Reply with `ANSWERED: YES` or " +
              "`ANSWERED: NO` on the first line, no other prefix.",
          },
        },
        { role: "user", content: { type: "text", text: userText } },
      ],
      tools: [],
      params: { stop_sequences: [] },
      stream: false,
    };

    let verdict = "";
    try {
      const resp = await this.judge.call(request, signal);
      for (const block of resp.content) {
        if (block.type === "text") {
          verdict = block.text;
          break;
        }
      }
    } catch (e) {
      return `judge model error: ${(e as Error).message}`;
    }
    const first = (verdict.split("\n")[0] ?? "").trim().toUpperCase();
    if (first.startsWith("ANSWERED: YES")) return null;
    return `judge says not answered: ${verdict}`;
  }

  description(): string {
    return `LLM-judge: did the response answer \`${this.originalQuestion}\``;
  }
}

function lastAssistantText(messages: readonly Message[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m == null) continue;
    if (m.role === "assistant" && m.content.type === "text") {
      return m.content.text;
    }
  }
  return null;
}

// ============================================================================
// SqlResultCheck
// ============================================================================

interface SqlResultPayload {
  columns?: string[];
  rows?: unknown[];
}

/**
 * Validates the most recent SQL tool result in the session. Scans
 * `state.messages` in reverse for the last `tool_result` whose matching
 * `tool_call` has `name === sqlToolName`, then parses the result content
 * as `{ "columns": string[], "rows": unknown[] }`.
 *
 * Returns `null` when the result satisfies all configured constraints.
 * Returns a reason if no matching SQL result was found, the payload did
 * not parse, the columns did not match `expectedColumns`, or the row
 * count was below `minRows` (default: 1).
 */
export class SqlResultCheck implements CompletionCheck {
  sqlToolName: string;
  expectedColumns: string[] | null;
  minRows: number | null;

  constructor(
    opts: {
      sqlToolName?: string;
      expectedColumns?: string[] | null;
      minRows?: number | null;
    } = {},
  ) {
    this.sqlToolName = opts.sqlToolName ?? "execute_sql";
    this.expectedColumns = opts.expectedColumns ?? null;
    this.minRows = opts.minRows ?? null;
  }

  withToolName(name: string): this {
    this.sqlToolName = name;
    return this;
  }

  withExpectedColumns(cols: string[]): this {
    this.expectedColumns = cols;
    return this;
  }

  withMinRows(n: number): this {
    this.minRows = n;
    return this;
  }

  async check(state: SessionStateSnapshot): Promise<string | null> {
    // Build id -> tool_name map from tool_calls so we can match tool_results
    // back to their originating tool.
    const idToName = new Map<string, string>();
    for (const m of state.state.messages) {
      const c: Content = m.content;
      if (c.type === "tool_call") idToName.set(c.id, c.name);
    }

    // Find the most recent tool_result belonging to sqlToolName.
    let resultContent: string | null = null;
    for (let i = state.state.messages.length - 1; i >= 0; i--) {
      const m = state.state.messages[i];
      if (m == null) continue;
      const c: Content = m.content;
      if (c.type === "tool_result") {
        const name = idToName.get(c.tool_use_id);
        if (name === this.sqlToolName) {
          resultContent = c.content;
          break;
        }
      }
    }
    if (resultContent == null) {
      return `no \`${this.sqlToolName}\` tool result found in session`;
    }

    let payload: SqlResultPayload;
    try {
      payload = JSON.parse(resultContent) as SqlResultPayload;
    } catch (e) {
      return `sql result is not JSON: ${(e as Error).message}`;
    }
    const columns = payload.columns ?? [];
    const rows = payload.rows ?? [];

    if (this.expectedColumns != null) {
      if (!arraysEqual(columns, this.expectedColumns)) {
        return `sql columns mismatch: expected ${JSON.stringify(
          this.expectedColumns,
        )}, got ${JSON.stringify(columns)}`;
      }
    }
    const min = this.minRows ?? 1;
    if (rows.length < min) {
      return `sql result has ${rows.length} rows, expected at least ${min}`;
    }
    return null;
  }

  description(): string {
    return `sql result check on tool \`${this.sqlToolName}\``;
  }
}

function arraysEqual(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
