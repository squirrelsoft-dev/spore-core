/**
 * Unit + fixture-replay tests for the #81 Standard Tool Catalogue (TypeScript).
 *
 * Mirrors the Rust suites in `rust/crates/spore-core/src/tools/{edit,search,
 * message,control,todo,catalogue}.rs` and replays the shared fixtures in
 * `fixtures/tools/{edit_file_cases,grep_output_modes,send_message_event,
 * escalation_tools,todo_write}.json`.
 */

import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  harnessTesting,
  SessionId,
  storage,
  toolRegistry,
  type ToolCall,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import {
  AbortTool,
  AskUserQuestionTool,
  EditFileTool,
  EnterPlanModeTool,
  ExitPlanModeTool,
  GrepTool,
  SendMessageTool,
  StandardTools,
  TodoWriteTool,
  TODO_STORE_KEY,
  WebFetchTool,
  WebSearchTool,
} from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;
const { InMemoryStorageProvider, ProjectId } = storage;
const { ToolContext } = toolRegistry;

const ctx = toolRegistry.toolRegistryMock.testCtx();

const here = dirname(fileURLToPath(import.meta.url));
const fixturesRoot = resolve(here, "../../../../fixtures/tools");

function call(name: string, input: unknown): ToolCall {
  return { id: "c1", name, input };
}

async function tmp(): Promise<string> {
  return mkdtemp(join(tmpdir(), "spore-tools-cat-"));
}

function readFixture<T>(name: string): T {
  return JSON.parse(readFileSync(join(fixturesRoot, name), "utf8")) as T;
}

// ============================================================================
// edit_file
// ============================================================================

describe("EditFileTool", () => {
  it("replaces a unique occurrence", async () => {
    const dir = await tmp();
    const path = join(dir, "a.txt");
    await writeFile(path, "hello world\n");
    const r = await new EditFileTool().execute(
      call("edit_file", { path, old_string: "world", new_string: "there" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("success");
    expect(await readFile(path, "utf8")).toBe("hello there\n");
  });

  it("not-found is a recoverable error", async () => {
    const dir = await tmp();
    const path = join(dir, "a.txt");
    await writeFile(path, "hello\n");
    const r = await new EditFileTool().execute(
      call("edit_file", { path, old_string: "absent", new_string: "x" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
    expect(r.message).toContain("not found");
  });

  it("non-unique is a recoverable error", async () => {
    const dir = await tmp();
    const path = join(dir, "a.txt");
    await writeFile(path, "x x x\n");
    const r = await new EditFileTool().execute(
      call("edit_file", { path, old_string: "x", new_string: "y" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
    expect(r.message).toContain("not unique");
  });

  it("missing file is a recoverable error", async () => {
    const r = await new EditFileTool().execute(
      call("edit_file", {
        path: "/no/such/file",
        old_string: "a",
        new_string: "b",
      }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it("bad params is a recoverable error", async () => {
    const r = await new EditFileTool().execute(
      call("edit_file", { path: "/x" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it("schema is destructive, not read_only", () => {
    const s = EditFileTool.schema();
    expect(s.annotations.destructive).toBe(true);
    expect(s.annotations.read_only).toBe(false);
  });

  interface EditCase {
    name: string;
    initial_content: string;
    old_string: string;
    new_string: string;
    expected:
      | { kind: "success"; final_content: string }
      | { kind: "error"; recoverable: boolean; reason: string };
  }

  it("replays edit_file_cases.json", async () => {
    const cases = readFixture<EditCase[]>("edit_file_cases.json");
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const dir = await tmp();
      const path = join(dir, "f.txt");
      await writeFile(path, c.initial_content);
      const r = await new EditFileTool().execute(
        call("edit_file", {
          path,
          old_string: c.old_string,
          new_string: c.new_string,
        }),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind, c.name).toBe(c.expected.kind);
      if (c.expected.kind === "success") {
        expect(await readFile(path, "utf8"), c.name).toBe(
          c.expected.final_content,
        );
      } else {
        if (r.kind !== "error") throw new Error("unreachable");
        expect(r.recoverable, c.name).toBe(c.expected.recoverable);
        const token =
          c.expected.reason === "not_found" ? "not found" : "not unique";
        expect(r.message, c.name).toContain(token);
      }
    }
  });
});

// ============================================================================
// grep (output modes)
// ============================================================================

describe("GrepTool", () => {
  async function grepOut(
    dir: string,
    mode: string,
    pattern = "alpha",
  ): Promise<string> {
    const r = await new GrepTool().execute(
      call("grep", { pattern, path: dir, recursive: true, output_mode: mode }),
      new AllowAllSandbox(),
      ctx,
    );
    if (r.kind !== "success")
      throw new Error(`expected success, got ${r.kind}`);
    return r.content;
  }

  it("content mode", async () => {
    const dir = await tmp();
    await writeFile(join(dir, "a.txt"), "alpha\nbeta\nalpha2");
    const out = await grepOut(dir, "content");
    expect(out.split("\n").length).toBe(2);
    expect(out).toContain(":1:alpha");
    expect(out).toContain(":3:alpha2");
  });

  it("files_with_matches mode", async () => {
    const dir = await tmp();
    await writeFile(join(dir, "a.txt"), "alpha\nalpha");
    await writeFile(join(dir, "b.txt"), "nope");
    const out = await grepOut(dir, "files_with_matches");
    expect(out.split("\n").length).toBe(1);
    expect(out.endsWith("a.txt")).toBe(true);
  });

  it("count mode", async () => {
    const dir = await tmp();
    await writeFile(join(dir, "a.txt"), "alpha\nalpha\nx");
    const out = await grepOut(dir, "count");
    expect(out.split("\n").length).toBe(1);
    expect(out.endsWith(":2")).toBe(true);
  });

  it("defaults to content mode", async () => {
    const dir = await tmp();
    await writeFile(join(dir, "a.txt"), "alpha");
    const r = await new GrepTool().execute(
      call("grep", { pattern: "alpha", path: dir }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toContain(":1:alpha");
  });

  it("invalid regex is a recoverable error", async () => {
    const dir = await tmp();
    const r = await new GrepTool().execute(
      call("grep", { pattern: "(unclosed", path: dir }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  interface GrepCase {
    name: string;
    files: Record<string, string>;
    pattern: string;
    output_mode: string;
    expected_lines: number;
    expected_contains: string[];
  }

  it("replays grep_output_modes.json", async () => {
    const cases = readFixture<GrepCase[]>("grep_output_modes.json");
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const dir = await tmp();
      for (const [fname, content] of Object.entries(c.files)) {
        await writeFile(join(dir, fname), content);
      }
      const out = await grepOut(dir, c.output_mode, c.pattern);
      expect(out.split("\n").length, c.name).toBe(c.expected_lines);
      for (const sub of c.expected_contains) {
        expect(out, c.name).toContain(sub);
      }
    }
  });
});

// ============================================================================
// send_message
// ============================================================================

describe("SendMessageTool", () => {
  it("echoes content", async () => {
    const r = await new SendMessageTool().execute(
      call("send_message", { content: "hi user" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toBe("hi user");
  });

  it("missing content is a recoverable error", async () => {
    const r = await new SendMessageTool().execute(
      call("send_message", {}),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  interface SendCase {
    name: string;
    input: unknown;
    expected_tool_output:
      | { kind: "success"; content: string }
      | { kind: "error"; recoverable: boolean };
    expected_stream_event: { kind: "user_message"; content: string } | null;
  }

  it("replays send_message_event.json", async () => {
    const cases = readFixture<SendCase[]>("send_message_event.json");
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const r = await new SendMessageTool().execute(
        call("send_message", c.input),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind, c.name).toBe(c.expected_tool_output.kind);
      if (c.expected_tool_output.kind === "success") {
        if (r.kind !== "success") throw new Error("unreachable");
        expect(r.content, c.name).toBe(c.expected_tool_output.content);
        // The loop would emit a user_message event mirroring the content.
        expect(c.expected_stream_event).not.toBeNull();
        expect(c.expected_stream_event?.content, c.name).toBe(r.content);
      } else {
        if (r.kind !== "error") throw new Error("unreachable");
        expect(r.recoverable, c.name).toBe(c.expected_tool_output.recoverable);
        expect(c.expected_stream_event, c.name).toBeNull();
      }
    }
  });
});

// ============================================================================
// Tier 3 control tools
// ============================================================================

describe("Tier 3 control tools", () => {
  it("enter_plan_mode escalates", async () => {
    const r = await new EnterPlanModeTool().execute(
      call("enter_plan_mode", { context: "seed" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("escalate");
    if (r.kind !== "escalate") throw new Error("unreachable");
    expect(r.signal).toEqual({ kind: "enter_plan_mode", context: "seed" });
  });

  it("exit_plan_mode escalates with plan", async () => {
    const r = await new ExitPlanModeTool().execute(
      call("exit_plan_mode", {
        plan: { tasks: ["a", "b"], rationale: "because" },
      }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("escalate");
    if (r.kind !== "escalate") throw new Error("unreachable");
    expect(r.signal).toEqual({
      kind: "exit_plan_mode",
      plan: { tasks: ["a", "b"], rationale: "because" },
    });
  });

  it("exit_plan_mode rationale defaults to empty", async () => {
    const r = await new ExitPlanModeTool().execute(
      call("exit_plan_mode", { plan: { tasks: ["x"] } }),
      new AllowAllSandbox(),
      ctx,
    );
    if (r.kind !== "escalate") throw new Error("unreachable");
    if (r.signal.kind !== "exit_plan_mode") throw new Error("unreachable");
    expect(r.signal.plan).toEqual({ tasks: ["x"], rationale: "" });
  });

  it("ask_user_question awaits clarification with options", async () => {
    const r = await new AskUserQuestionTool().execute(
      call("ask_user_question", { question: "which?", options: ["a", "b"] }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("awaiting_clarification");
    if (r.kind !== "awaiting_clarification") throw new Error("unreachable");
    expect(r.question).toBe("which?");
    expect(r.options).toEqual(["a", "b"]);
  });

  it("ask_user_question options optional", async () => {
    const r = await new AskUserQuestionTool().execute(
      call("ask_user_question", { question: "free form?" }),
      new AllowAllSandbox(),
      ctx,
    );
    if (r.kind !== "awaiting_clarification") throw new Error("unreachable");
    expect(r.question).toBe("free form?");
    expect(r.options).toBeUndefined();
  });

  it("abort escalates", async () => {
    const r = await new AbortTool().execute(
      call("abort", { reason: "stop" }),
      new AllowAllSandbox(),
      ctx,
    );
    if (r.kind !== "escalate") throw new Error("unreachable");
    expect(r.signal).toEqual({ kind: "abort", reason: "stop" });
  });

  it("abort missing reason is a recoverable error", async () => {
    const r = await new AbortTool().execute(
      call("abort", {}),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  interface EscalationCase {
    name: string;
    tool: string;
    input: unknown;
    expected: {
      tool_output_kind: string;
      signal?: unknown;
      question?: string;
      options?: string[] | null;
    };
  }

  it("replays escalation_tools.json", async () => {
    const cases = readFixture<EscalationCase[]>("escalation_tools.json");
    expect(cases.length).toBeGreaterThan(0);
    const byName = (t: string) => {
      switch (t) {
        case "enter_plan_mode":
          return new EnterPlanModeTool();
        case "exit_plan_mode":
          return new ExitPlanModeTool();
        case "ask_user_question":
          return new AskUserQuestionTool();
        case "abort":
          return new AbortTool();
        default:
          throw new Error(`unknown tool ${t}`);
      }
    };
    for (const c of cases) {
      const r = await byName(c.tool).execute(
        call(c.tool, c.input),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind, c.name).toBe(c.expected.tool_output_kind);
      if (r.kind === "escalate") {
        expect(r.signal, c.name).toEqual(c.expected.signal);
      } else if (r.kind === "awaiting_clarification") {
        expect(r.question, c.name).toBe(c.expected.question);
        const expectedOptions =
          c.expected.options === null ? undefined : c.expected.options;
        expect(r.options, c.name).toEqual(expectedOptions);
      }
    }
  });
});

// ============================================================================
// todo_write
// ============================================================================

describe("TodoWriteTool", () => {
  function inMemoryCtx(): toolRegistry.ToolContext {
    return new ToolContext(
      SessionId.of("todo-session"),
      ProjectId.fromCanonicalPath("/test-project"),
      new InMemoryStorageProvider(),
      new InMemoryStorageProvider(),
    );
  }

  it("writes and persists under the todo key", async () => {
    const c = inMemoryCtx();
    const r = await new TodoWriteTool().execute(
      call("todo_write", {
        todos: [
          { content: "a", status: "pending" },
          { content: "b", status: "in_progress" },
        ],
      }),
      new AllowAllSandbox(),
      c,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    const got = JSON.parse(r.content) as unknown[];
    expect(got.length).toBe(2);
    const persisted = await c.runStore.get(c.sessionId, TODO_STORE_KEY);
    expect(persisted).toEqual([
      { content: "a", status: "pending" },
      { content: "b", status: "in_progress" },
    ]);
  });

  it("replaces the list wholesale", async () => {
    const c = inMemoryCtx();
    const tool = new TodoWriteTool();
    await tool.execute(
      call("todo_write", {
        todos: [
          { content: "old1", status: "pending" },
          { content: "old2", status: "pending" },
        ],
      }),
      new AllowAllSandbox(),
      c,
    );
    const r = await tool.execute(
      call("todo_write", { todos: [{ content: "new", status: "completed" }] }),
      new AllowAllSandbox(),
      c,
    );
    if (r.kind !== "success") throw new Error("unreachable");
    const got = JSON.parse(r.content) as { content: string }[];
    expect(got.length).toBe(1);
    expect(got[0]?.content).toBe("new");
  });

  it("bad params is a recoverable error", async () => {
    const r = await new TodoWriteTool().execute(
      call("todo_write", { todos: "not-an-array" }),
      new AllowAllSandbox(),
      inMemoryCtx(),
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it("schema is not read_only", () => {
    const s = TodoWriteTool.schema();
    expect(s.annotations.read_only).toBe(false);
    expect(s.annotations.destructive).toBe(false);
  });

  interface TodoCase {
    name: string;
    steps: { input: { todos: unknown[] }; expected_persisted: unknown[] }[];
  }

  it("replays todo_write.json", async () => {
    const cases = readFixture<TodoCase[]>("todo_write.json");
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const cx = inMemoryCtx();
      const tool = new TodoWriteTool();
      for (const step of c.steps) {
        await tool.execute(
          call("todo_write", step.input),
          new AllowAllSandbox(),
          cx,
        );
        const persisted = await cx.runStore.get(cx.sessionId, TODO_STORE_KEY);
        expect(persisted, `${c.name}`).toEqual(step.expected_persisted);
      }
    }
  });
});

// ============================================================================
// web tools (mock HTTP, never live network)
// ============================================================================

describe("web tools", () => {
  it("web_fetch returns the body", async () => {
    const { close, url } = await startServer((_req, res) => {
      res.statusCode = 200;
      res.end("page-body");
    });
    try {
      const r = await new WebFetchTool().execute(
        call("web_fetch", { url: `${url}/page` }),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind).toBe("success");
      if (r.kind !== "success") throw new Error("unreachable");
      expect(r.content).toBe("page-body");
    } finally {
      await close();
    }
  });

  it("web_search returns structured results from a mock backend", async () => {
    const results = '{"results":[{"title":"t","url":"u"}]}';
    const { close, url } = await startServer((req, res) => {
      expect(req.method).toBe("POST");
      res.statusCode = 200;
      res.end(results);
    });
    try {
      const r = await WebSearchTool.withEndpoint(`${url}/search`).execute(
        call("web_search", { query: "rust" }),
        new AllowAllSandbox(),
        ctx,
      );
      expect(r.kind).toBe("success");
      if (r.kind !== "success") throw new Error("unreachable");
      expect(r.content).toBe(results);
    } finally {
      await close();
    }
  });

  it("web_search without a backend is a recoverable error", async () => {
    const r = await new WebSearchTool().execute(
      call("web_search", { query: "x" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });
});

// ============================================================================
// StandardTools presets
// ============================================================================

describe("StandardTools presets", () => {
  const names = (set: { schema: { name: string } }[]) =>
    set.map((t) => t.schema.name);

  it("every constructor pairs a matching impl + schema", () => {
    for (const t of StandardTools.fullSet()) {
      expect(t.implementation.name).toBe(t.schema.name);
    }
  });

  it("readonly_set has no mutating or escalating tools", () => {
    const n = names(StandardTools.readonlySet());
    for (const forbidden of [
      "write_file",
      "edit_file",
      "bash_command",
      "todo_write",
      "enter_plan_mode",
      "exit_plan_mode",
      "ask_user_question",
      "abort",
    ]) {
      expect(n).not.toContain(forbidden);
    }
    expect(n).toContain("read_file");
    expect(n).toContain("grep");
  });

  it("coding_set reuses existing names on overlap (Q5)", () => {
    const n = names(StandardTools.codingSet());
    for (const existing of [
      "read_file",
      "write_file",
      "find_files",
      "grep_files",
      "bash_command",
    ]) {
      expect(n).toContain(existing);
    }
    expect(n).toContain("edit_file");
    expect(n).toContain("grep");
    expect(n).not.toContain("abort");
  });

  it("webSearchWithEndpoint yields a tool named web_search", () => {
    const t = StandardTools.webSearchWithEndpoint("http://localhost:9/search");
    expect(t.implementation.name).toBe("web_search");
    expect(t.schema.name).toBe("web_search");
  });

  it("full_set adds the Tier-3 control tools", () => {
    const n = names(StandardTools.fullSet());
    for (const t of [
      "enter_plan_mode",
      "exit_plan_mode",
      "ask_user_question",
      "abort",
    ]) {
      expect(n).toContain(t);
    }
  });
});

// ----------------------------------------------------------------------------
// Minimal local HTTP server helper (mock backend; never the live network).
// ----------------------------------------------------------------------------

import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import type { AddressInfo } from "node:net";

async function startServer(
  handler: (req: IncomingMessage, res: ServerResponse) => void,
): Promise<{ url: string; close: () => Promise<void> }> {
  const server = createServer(handler);
  await new Promise<void>((res) => server.listen(0, "127.0.0.1", res));
  const addr = server.address() as AddressInfo;
  const url = `http://127.0.0.1:${addr.port}`;
  return {
    url,
    close: () => new Promise<void>((res) => server.close(() => res())),
  };
}
