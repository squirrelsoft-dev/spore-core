/**
 * Unit and fixture-replay tests for the standard
 * {@link termination.CompletionCheck} implementations (spore-core issue #43).
 *
 * Mirrors the Rust reference in `rust/crates/spore-core/src/termination.rs`
 * so the cross-language `fixtures/completion_check/sql_result.json` fixture
 * produces the same outcome on every target.
 */

import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  SessionId,
  TaskId,
  emptySessionState,
  type CommandOutput,
  type SandboxProvider,
  type SandboxViolation,
  type SessionState,
} from "../src/harness/types.js";
import type { ModelInterface } from "../src/model/interface.js";
import type {
  Message,
  ModelRequest,
  ModelResponse,
  ProviderInfo,
  StreamEvent,
} from "../src/model/schemas.js";
import { termination } from "../src/index.js";

const {
  AlwaysComplete,
  FeatureListCheck,
  NullCompletionCheck,
  QuestionAnsweredCheck,
  SqlResultCheck,
  TestSuiteCheck,
  newSessionStateSnapshot,
} = termination;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function snapshotIn(workspaceRoot: string) {
  return newSessionStateSnapshot(
    new SessionId("s1"),
    new TaskId("t1"),
    emptySessionState(),
    workspaceRoot,
  );
}

function snapshotWithState(state: SessionState) {
  return newSessionStateSnapshot(new SessionId("s1"), new TaskId("t1"), state, "");
}

// ---------------------------------------------------------------------------
// AlwaysComplete (alias of NullCompletionCheck)
// ---------------------------------------------------------------------------

describe("AlwaysComplete", () => {
  it("returns null and has the null description", async () => {
    const c = new AlwaysComplete();
    expect(await c.check(snapshotIn(""))).toBeNull();
    expect(c.description()).toBe("null (always complete)");
  });

  it("is an alias of NullCompletionCheck", () => {
    expect(AlwaysComplete).toBe(NullCompletionCheck);
    expect(new AlwaysComplete()).toBeInstanceOf(NullCompletionCheck);
  });
});

// ---------------------------------------------------------------------------
// FeatureListCheck
// ---------------------------------------------------------------------------

describe("FeatureListCheck", () => {
  // Default path is `.spore/feature_list.json` (issue #58, B2).
  function writeDefaultFeatureList(dir: string, body: string): void {
    mkdirSync(join(dir, ".spore"), { recursive: true });
    writeFileSync(join(dir, ".spore", "feature_list.json"), body);
  }

  it("returns null when all features pass", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-fl-"));
    writeDefaultFeatureList(
      dir,
      JSON.stringify([
        { name: "a", passes: true },
        { name: "b", passes: true },
      ]),
    );
    expect(await new FeatureListCheck().check(snapshotIn(dir))).toBeNull();
  });

  it("returns a reason listing failing features", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-fl-"));
    writeDefaultFeatureList(
      dir,
      JSON.stringify([
        { name: "a", passes: true },
        { name: "b", passes: false },
        { name: "c", passes: false },
      ]),
    );
    const r = await new FeatureListCheck().check(snapshotIn(dir));
    expect(r).not.toBeNull();
    expect(r).toContain("b");
    expect(r).toContain("c");
    expect(r).not.toContain("a, ");
  });

  it("returns a missing-file reason when the file is absent", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-fl-"));
    const r = await new FeatureListCheck().check(snapshotIn(dir));
    expect(r).not.toBeNull();
    expect(r).toContain("missing");
  });

  it("returns an invalid-JSON reason when the file is malformed", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-fl-"));
    writeDefaultFeatureList(dir, "not json");
    const r = await new FeatureListCheck().check(snapshotIn(dir));
    expect(r).not.toBeNull();
    expect(r).toContain("invalid JSON");
  });

  it("honors a custom relative path", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-fl-"));
    writeFileSync(join(dir, "custom.json"), JSON.stringify([{ name: "x", passes: true }]));
    expect(await new FeatureListCheck("custom.json").check(snapshotIn(dir))).toBeNull();
  });

  it("honors an absolute path independent of workspace_root", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-fl-"));
    const abs = join(dir, "abs.json");
    writeFileSync(abs, JSON.stringify([{ name: "x", passes: true }]));
    // workspace_root is empty; absolute path still resolves.
    expect(await new FeatureListCheck(abs).check(snapshotIn(""))).toBeNull();
  });

  it("description includes the path", () => {
    expect(new FeatureListCheck("custom.json").description()).toContain("custom.json");
  });
});

// ---------------------------------------------------------------------------
// TestSuiteCheck
// ---------------------------------------------------------------------------

class StubSandbox implements SandboxProvider {
  constructor(private readonly out: CommandOutput | SandboxViolation) {}
  async validate(): Promise<SandboxViolation | null> {
    return null;
  }
  workspaceRoot(): string {
    return "/";
  }
  async executeCommand(): Promise<CommandOutput | SandboxViolation> {
    return this.out;
  }
}

function stubSandbox(exit: number, stderr: string, stdout = ""): SandboxProvider {
  return new StubSandbox({
    stdout,
    stderr,
    exit_code: exit,
    timed_out: false,
    truncated: false,
  });
}

describe("TestSuiteCheck", () => {
  it("returns null on exit code 0", async () => {
    const c = new TestSuiteCheck("npm test", ".", 30_000, stubSandbox(0, ""));
    expect(await c.check(snapshotIn(""))).toBeNull();
  });

  it("returns a reason including stderr tail on failure", async () => {
    const c = new TestSuiteCheck(
      "npm test",
      ".",
      30_000,
      stubSandbox(1, "test foo ... FAILED\nassertion failed"),
    );
    const r = await c.check(snapshotIn(""));
    expect(r).not.toBeNull();
    expect(r).toContain("FAILED");
    expect(r).toContain("exit 1");
  });

  it("falls back to stdout tail when stderr is empty", async () => {
    const c = new TestSuiteCheck(
      "npm test",
      ".",
      30_000,
      stubSandbox(2, "", "out: failure summary"),
    );
    const r = await c.check(snapshotIn(""));
    expect(r).toContain("failure summary");
  });

  it("returns a reason for an empty command", async () => {
    const c = new TestSuiteCheck("   ", ".", 30_000, stubSandbox(0, ""));
    const r = await c.check(snapshotIn(""));
    expect(r).not.toBeNull();
    expect(r).toContain("empty");
  });

  it("returns a reason on sandbox violation", async () => {
    const violation: SandboxViolation = {
      kind: "disallowed_command",
      command: "rm",
    };
    const sandbox = new StubSandbox(violation);
    const c = new TestSuiteCheck("rm -rf /", ".", 30_000, sandbox);
    const r = await c.check(snapshotIn(""));
    expect(r).not.toBeNull();
    expect(r).toContain("sandbox");
  });

  it("description includes command and working dir", () => {
    const c = new TestSuiteCheck("npm test", "/work", 30_000, stubSandbox(0, ""));
    expect(c.description()).toContain("npm test");
    expect(c.description()).toContain("/work");
  });
});

// ---------------------------------------------------------------------------
// QuestionAnsweredCheck
// ---------------------------------------------------------------------------

class StubJudge implements ModelInterface {
  constructor(private readonly verdict: string) {}
  async call(_req: ModelRequest): Promise<ModelResponse> {
    return {
      content: [{ type: "text", text: this.verdict }],
      usage: { input_tokens: 0, output_tokens: 0 },
      stop_reason: "end_turn",
    };
  }
  async *callStreaming(): AsyncIterable<StreamEvent> {
    throw new Error("not used");
  }
  async countTokens(): Promise<number> {
    return 0;
  }
  provider(): ProviderInfo {
    return { name: "stub", model_id: "stub", context_window: 4096 };
  }
}

function snapWithAssistant(text: string) {
  const state = emptySessionState();
  const m: Message = { role: "assistant", content: { type: "text", text } };
  state.messages.push(m);
  return snapshotWithState(state);
}

describe("QuestionAnsweredCheck", () => {
  it("returns null when the judge says ANSWERED: YES", async () => {
    const c = new QuestionAnsweredCheck(
      new StubJudge("ANSWERED: YES\nLooks good."),
      "What is 2+2?",
    );
    expect(await c.check(snapWithAssistant("It is 4."))).toBeNull();
  });

  it("returns a reason when the judge says NO", async () => {
    const c = new QuestionAnsweredCheck(
      new StubJudge("ANSWERED: NO\nMissed the point."),
      "What is 2+2?",
    );
    const r = await c.check(snapWithAssistant("I don't know."));
    expect(r).not.toBeNull();
    expect(r).toContain("not answered");
  });

  it("returns a reason when the judge call throws", async () => {
    class FailingJudge extends StubJudge {
      override async call(): Promise<ModelResponse> {
        throw new Error("network down");
      }
    }
    const c = new QuestionAnsweredCheck(new FailingJudge(""), "q");
    const r = await c.check(snapWithAssistant("a"));
    expect(r).not.toBeNull();
    expect(r).toContain("judge model error");
  });

  it("supports a rubric without changing the YES path", async () => {
    const c = new QuestionAnsweredCheck(new StubJudge("ANSWERED: YES"), "q").withRubric(
      "Be strict about citations.",
    );
    expect(c.description()).toContain("q");
    expect(await c.check(snapWithAssistant("a"))).toBeNull();
  });

  it("falls back to <no agent response> when there is no assistant text", async () => {
    const c = new QuestionAnsweredCheck(new StubJudge("ANSWERED: YES"), "q");
    expect(await c.check(snapshotIn(""))).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// SqlResultCheck — unit
// ---------------------------------------------------------------------------

function snapWithSql(toolName: string, body: string) {
  const state = emptySessionState();
  state.messages.push({
    role: "assistant",
    content: { type: "tool_call", id: "call-1", name: toolName, input: { q: "select 1" } },
  });
  state.messages.push({
    role: "tool",
    content: {
      type: "tool_result",
      tool_use_id: "call-1",
      content: body,
      is_error: false,
    },
  });
  return snapshotWithState(state);
}

describe("SqlResultCheck — unit", () => {
  it("default passes when rows are present", async () => {
    const snap = snapWithSql(
      "execute_sql",
      JSON.stringify({
        columns: ["id", "name"],
        rows: [
          [1, "a"],
          [2, "b"],
        ],
      }),
    );
    expect(await new SqlResultCheck().check(snap)).toBeNull();
  });

  it("default fails when rows are empty (0 rows in message)", async () => {
    const snap = snapWithSql("execute_sql", JSON.stringify({ columns: ["id"], rows: [] }));
    const r = await new SqlResultCheck().check(snap);
    expect(r).not.toBeNull();
    expect(r).toContain("0 rows");
  });

  it("column mismatch reports columns mismatch", async () => {
    const snap = snapWithSql("execute_sql", JSON.stringify({ columns: ["id"], rows: [[1]] }));
    const r = await new SqlResultCheck().withExpectedColumns(["id", "name"]).check(snap);
    expect(r).toContain("columns mismatch");
  });

  it("min_rows is enforced", async () => {
    const snap = snapWithSql("execute_sql", JSON.stringify({ columns: ["id"], rows: [[1]] }));
    const r = await new SqlResultCheck().withMinRows(5).check(snap);
    expect(r).toContain("at least 5");
  });

  it("custom tool name", async () => {
    const snap = snapWithSql("run_query", JSON.stringify({ columns: [], rows: [[1]] }));
    expect(await new SqlResultCheck().withToolName("run_query").check(snap)).toBeNull();
  });

  it("no matching tool result fails", async () => {
    const snap = snapWithSql("other_tool", JSON.stringify({ columns: [], rows: [[1]] }));
    const r = await new SqlResultCheck().check(snap);
    expect(r).toContain("no `execute_sql`");
  });

  it("invalid JSON payload reports a parse error", async () => {
    const snap = snapWithSql("execute_sql", "not json");
    const r = await new SqlResultCheck().check(snap);
    expect(r).not.toBeNull();
    expect(r).toContain("not JSON");
  });

  it("description names the tool", () => {
    expect(new SqlResultCheck().description()).toContain("execute_sql");
  });
});

// ---------------------------------------------------------------------------
// SqlResultCheck — cross-language fixture replay
// ---------------------------------------------------------------------------

interface SqlFixtureCase {
  name: string;
  sql_tool_name: string;
  expected_columns?: string[];
  min_rows?: number;
  messages: Message[];
  expected: { kind: "complete" } | { kind: "incomplete"; contains: string };
}
interface SqlFixtureSuite {
  cases: SqlFixtureCase[];
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/completion_check/sql_result.json");

describe("SqlResultCheck — fixture replay", () => {
  const raw = readFileSync(fixturePath, "utf-8");
  const suite = JSON.parse(raw) as SqlFixtureSuite;

  for (const c of suite.cases) {
    it(c.name, async () => {
      const state = emptySessionState();
      state.messages = c.messages;
      const snap = newSessionStateSnapshot(new SessionId("fix"), new TaskId("fix"), state, "");
      let check = new SqlResultCheck().withToolName(c.sql_tool_name);
      if (c.expected_columns != null) check = check.withExpectedColumns(c.expected_columns);
      if (c.min_rows != null) check = check.withMinRows(c.min_rows);
      const got = await check.check(snap);
      if (c.expected.kind === "complete") {
        expect(got).toBeNull();
      } else {
        expect(got).not.toBeNull();
        expect(got).toContain(c.expected.contains);
      }
    });
  }
});
