/**
 * Fixture-replay tests for the shared tool fixtures in `fixtures/tools/`.
 * Mirrors the Rust suite in `rust/crates/spore-core/src/tools/mod.rs`.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  defaultHandleLargeOutput,
  DeleteFileParamsSchema,
  FindFilesParamsSchema,
  GitCommitParamsSchema,
  GitDiffParamsSchema,
  GitLogParamsSchema,
  GitResetParamsSchema,
  GitStatusParamsSchema,
  GrepFilesParamsSchema,
  HttpGetParamsSchema,
  HttpPostParamsSchema,
  ListDirParamsSchema,
  MoveFileParamsSchema,
  ReadFileParamsSchema,
  WriteFileParamsSchema,
} from "../src/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixturesRoot = resolve(here, "../../../../fixtures/tools");

interface ParamValidationScenario {
  tool: string;
  input: unknown;
  expected: "ok" | "invalid_parameters";
}

function paramParseOk(tool: string, input: unknown): boolean {
  switch (tool) {
    case "read_file":
      return ReadFileParamsSchema.safeParse(input).success;
    case "write_file":
      return WriteFileParamsSchema.safeParse(input).success;
    case "list_dir":
      return ListDirParamsSchema.safeParse(input).success;
    case "delete_file":
      return DeleteFileParamsSchema.safeParse(input).success;
    case "move_file":
      return MoveFileParamsSchema.safeParse(input).success;
    case "grep_files": {
      const r = GrepFilesParamsSchema.safeParse(input);
      if (!r.success) return false;
      try {
        new RegExp(r.data.pattern);
      } catch {
        return false;
      }
      return true;
    }
    case "find_files":
      return FindFilesParamsSchema.safeParse(input).success;
    case "git_status":
      return GitStatusParamsSchema.safeParse(input).success;
    case "git_log":
      return GitLogParamsSchema.safeParse(input).success;
    case "git_diff":
      return GitDiffParamsSchema.safeParse(input).success;
    case "git_commit":
      return GitCommitParamsSchema.safeParse(input).success;
    case "git_reset":
      return GitResetParamsSchema.safeParse(input).success;
    case "http_get":
      return HttpGetParamsSchema.safeParse(input).success;
    case "http_post":
      return HttpPostParamsSchema.safeParse(input).success;
    default:
      return true;
  }
}

describe("fixture: param_validation", () => {
  const data = readFileSync(
    resolve(fixturesRoot, "param_validation.json"),
    "utf8",
  );
  const scenarios = JSON.parse(data) as ParamValidationScenario[];

  it("loads at least one scenario", () => {
    expect(scenarios.length).toBeGreaterThan(0);
  });

  for (const sc of scenarios) {
    it(`${sc.tool} -> ${sc.expected}`, () => {
      const actual = paramParseOk(sc.tool, sc.input)
        ? "ok"
        : "invalid_parameters";
      expect(actual).toBe(sc.expected);
    });
  }
});

interface TruncationScenario {
  content_length: number;
  head_tokens: number;
  tail_tokens: number;
  expects_truncated: boolean;
}

describe("fixture: output_truncation", () => {
  const data = readFileSync(
    resolve(fixturesRoot, "output_truncation.json"),
    "utf8",
  );
  const scenarios = JSON.parse(data) as TruncationScenario[];

  for (const sc of scenarios) {
    it(`content_length=${sc.content_length} -> truncated=${sc.expects_truncated}`, async () => {
      const content = "x".repeat(sc.content_length);
      const out = await defaultHandleLargeOutput(
        content,
        "fx",
        sc.head_tokens,
        sc.tail_tokens,
      );
      expect(out.truncated).toBe(sc.expects_truncated);
    });
  }
});

interface SubagentScenario {
  name: string;
  child_run_result: {
    kind: "success" | "failure" | "waiting_for_human";
    output?: string;
  };
  parent_call_id: string;
  expected:
    | { kind: "success"; content: string }
    | { kind: "error"; recoverable: boolean }
    | { kind: "waiting_for_human"; parent_tool_call_id: string };
}

import {
  emptyAggregateUsage,
  emptyBudgetSnapshot,
  emptySessionState,
  harnessTesting,
  newTask,
  SessionId,
  TaskId,
  toolRegistry,
  type Harness,
  type HarnessRunOptions,
  type HumanResponse,
  type PausedState,
  type RunResult,
  type StreamSink,
} from "@spore/core";
import { SubagentTool } from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;
const { StandardToolRegistry } = toolRegistry;
// Storage seam (#75): SubagentTool ignores ctx, but the signature requires one.
const ctx = toolRegistry.toolRegistryMock.testCtx();

class ScriptedHarness implements Harness {
  constructor(private readonly r: RunResult) {}
  async run(_opts: HarnessRunOptions): Promise<RunResult> {
    return this.r;
  }
  async resume(
    _state: PausedState,
    _response: HumanResponse,
    _onStream?: StreamSink,
  ): Promise<RunResult> {
    return this.r;
  }
}

describe("fixture: subagent_scenarios", () => {
  const data = readFileSync(
    resolve(fixturesRoot, "subagent_scenarios.json"),
    "utf8",
  );
  const scenarios = JSON.parse(data) as SubagentScenario[];

  for (const sc of scenarios) {
    it(sc.name, async () => {
      let runResult: RunResult;
      const sid = SessionId.of("s");
      switch (sc.child_run_result.kind) {
        case "success":
          runResult = {
            kind: "success",
            output: sc.child_run_result.output ?? "done",
            session_id: sid,
            usage: emptyAggregateUsage(),
            turns: 1,
          };
          break;
        case "failure":
          runResult = {
            kind: "failure",
            reason: { kind: "human_halted" },
            session_id: sid,
            usage: emptyAggregateUsage(),
            turns: 1,
          };
          break;
        case "waiting_for_human":
          runResult = {
            kind: "waiting_for_human",
            state: {
              session_id: sid,
              task_id: TaskId.of("t"),
              turn_number: 1,
              session_state: emptySessionState(),
              pending_tool_calls: [],
              approved_results: [],
              human_request: { kind: "clarification", question: "?" },
              task: newTask("x", sid, {
                kind: "react",
                budget: { kind: "per_loop", value: 1 },
                agent: "",
                toolset: "",
              }),
              budget_used: emptyBudgetSnapshot(),
              child_state: null,
              toolset: "",
            },
            request: { kind: "clarification", question: "?" },
          };
          break;
      }

      const sub = SubagentTool.buildOrThrow({
        name: "subagent",
        description: "child",
        inputSchema: { type: "object" },
        timeoutMs: 5_000,
        contextSharing: { kind: "isolated" },
        harness: new ScriptedHarness(runResult),
        childRegistry: new StandardToolRegistry(),
      });

      const out = await sub.execute(
        {
          id: sc.parent_call_id,
          name: "subagent",
          input: { instruction: "x" },
        },
        new AllowAllSandbox(),
        ctx,
      );

      if (sc.expected.kind === "success") {
        expect(out.kind).toBe("success");
        if (out.kind !== "success") throw new Error("unreachable");
        expect(out.content).toBe(sc.expected.content);
      } else if (sc.expected.kind === "error") {
        expect(out.kind).toBe("error");
        if (out.kind !== "error") throw new Error("unreachable");
        expect(out.recoverable).toBe(sc.expected.recoverable);
      } else {
        expect(out.kind).toBe("waiting_for_human");
        if (out.kind !== "waiting_for_human") throw new Error("unreachable");
        expect(out.child_state.parent_tool_call_id).toBe(
          sc.expected.parent_tool_call_id,
        );
      }
    });
  }
});
