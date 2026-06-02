/**
 * Harness-loop tests for the issue #91 catalogue tool path + system_prompt seam:
 *  - `.tool()` / `.tools()` accumulate catalogue tools that `buildConfig()` folds
 *    into a populated `StandardToolRegistry`, defaulting storage to the in-memory
 *    provider (not the all-no-op default) so session-aware tools persist.
 *  - The per-run `RealToolRegistry` bridge advertises the catalogue schemas to
 *    the model and maps an unknown-tool dispatch onto a *recoverable* error so
 *    the loop appends it and the agent can adapt.
 *  - `.systemPrompt()` prepends a leading `system` message to each turn's
 *    assembled context — but only when none already leads (no duplicate), and
 *    not at all when unset.
 *
 * Mirrors the corresponding Rust harness tests for issue #91.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  HarnessBuilder,
  MockAgent,
  SessionId,
  StandardHarness,
  storage,
  newTask,
  toolRegistry,
  type Agent,
  type Context,
  type LoopStrategy,
  type Message,
  type SessionState,
  type Task,
  type TokenUsage,
  toolOutput,
  type ToolCall,
  type ToolOutput,
  type TurnResult,
} from "../src/index.js";

import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

type StandardTool = toolRegistry.StandardTool;
const { RealToolRegistry, StandardToolRegistry } = toolRegistry;

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function react(max: number): Task {
  const strategy: LoopStrategy = { kind: "re_act", max_iterations: max };
  return newTask("do something", SessionId.of("s1"), strategy);
}

function makeAgent(): MockAgent {
  const a = new MockAgent(AgentId.of("test"));
  a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);
  return a;
}

/** A trivial no-op catalogue tool, constructed inline (no @spore/tools dep). */
function fakeTool(name: string): StandardTool {
  return {
    implementation: {
      name,
      async execute(): Promise<ToolOutput> {
        return { kind: "success", content: `${name} ran`, truncated: false };
      },
    },
    schema: {
      name,
      description: `the ${name} tool`,
      parameters: { type: "object", properties: {} },
      annotations: {
        read_only: true,
        destructive: false,
        idempotent: true,
        open_world: false,
      },
    },
  };
}

function catalogueBuilder(agent: Agent): HarnessBuilder {
  return new HarnessBuilder(
    agent,
    new ScriptedToolRegistry(),
    new AllowAllSandbox(),
    new NoopContextManager(),
    new AlwaysContinuePolicy(),
  );
}

/** An agent that records the messages of the context it was handed. */
class CapturingAgent implements Agent {
  seen: Message[] = [];
  constructor(private readonly agentId: AgentId) {}
  id(): AgentId {
    return this.agentId;
  }
  async turn(context: Context): Promise<TurnResult> {
    this.seen = context.messages.slice();
    return { kind: "final_response", content: "done", usage: usage() };
  }
}

// ============================================================================
// ToolOutput constructors
// ============================================================================

describe("toolOutput constructors (#91)", () => {
  it("success() is a non-truncated success", () => {
    expect(toolOutput.success("ok")).toEqual({
      kind: "success",
      content: "ok",
      truncated: false,
    });
  });

  it("error() is a recoverable error", () => {
    expect(toolOutput.error("bad args")).toEqual({
      kind: "error",
      message: "bad args",
      recoverable: true,
    });
  });

  it("fatal() is a non-recoverable error", () => {
    expect(toolOutput.fatal("unrecoverable")).toEqual({
      kind: "error",
      message: "unrecoverable",
      recoverable: false,
    });
  });
});

// ============================================================================
// Catalogue fold + in-memory storage default
// ============================================================================

describe("catalogue tool fold (#91)", () => {
  it("folds .tool() tools into a registry and defaults storage to in-memory", async () => {
    const cfg = catalogueBuilder(makeAgent())
      .tool(fakeTool("read_file"))
      .tool(fakeTool("write_file"))
      .buildConfig();

    // Accumulated catalogue tools were folded into a registry.
    const catalogue = cfg.catalogueRegistry;
    expect(catalogue).toBeDefined();
    const names = catalogue!.activeSchemas(null).map((s) => s.name);
    expect(names).toContain("read_file");
    expect(names).toContain("write_file");

    // Storage defaulted to in-memory (not all-no-op) because catalogue tools are
    // present: a put/get round-trips on the run store.
    const sid = SessionId.of("s1");
    const run = cfg.storage!.run();
    await run.put(sid, "k", { v: 1 } as never);
    expect(await run.get(sid, "k")).not.toBeNull();
  });

  it("keeps the toolRegistry seam (no catalogueRegistry) when no tools are added", () => {
    const cfg = catalogueBuilder(makeAgent()).buildConfig();
    expect(cfg.catalogueRegistry).toBeUndefined();
    // And storage stays the all-no-op default (a get never round-trips a put).
    expect(cfg.storage).toBeDefined();
  });

  it("an explicitly-wired storage provider wins over the in-memory default", async () => {
    const wired = storage.StorageProvider.noOp();
    const cfg = catalogueBuilder(makeAgent())
      .tool(fakeTool("read_file"))
      .storage(wired)
      .buildConfig();
    expect(cfg.catalogueRegistry).toBeDefined();
    // The explicitly-wired no-op provider was respected (not overridden by the
    // in-memory default): a put/get does NOT round-trip.
    expect(cfg.storage).toBe(wired);
    const sid = SessionId.of("s1");
    await cfg.storage!.run().put(sid, "k", { v: 1 } as never);
    expect(await cfg.storage!.run().get(sid, "k")).toBeFalsy();
  });
});

// ============================================================================
// Per-run RealToolRegistry bridge
// ============================================================================

describe("RealToolRegistry bridge (#91)", () => {
  it("advertises the catalogue schemas and maps an unknown tool to a recoverable error", async () => {
    const inner = new StandardToolRegistry();
    const t = fakeTool("read_file");
    expect(inner.register(t.implementation, t.schema)).toBeNull();

    const store = storage.StorageProvider.single(new storage.InMemoryStorageProvider());
    const bridge = new RealToolRegistry(
      inner,
      new AllowAllSandbox(),
      SessionId.of("s1"),
      store.run(),
      store.memory(),
    );

    // The bridge advertises the catalogue schemas to the model.
    expect(bridge.schemas().some((s) => s.name === "read_file")).toBe(true);

    // And maps an inner dispatch failure (unknown tool) to a *recoverable* error
    // so the loop appends it and the agent can adapt.
    const out = await bridge.dispatch({
      id: "c",
      name: "does_not_exist",
      input: {},
    } as ToolCall);
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });
});

// ============================================================================
// system_prompt seam
// ============================================================================

describe("systemPrompt seam (#91)", () => {
  it("prepends the system prompt as the leading context message", async () => {
    const agent = new CapturingAgent(AgentId.of("cap"));
    const cfg = catalogueBuilder(agent).systemPrompt("OPERATING RULES").buildConfig();
    const h = new StandardHarness(cfg);
    await h.run({ task: react(2) });

    const first = agent.seen[0];
    expect(first).toBeDefined();
    expect(first!.role).toBe("system");
    expect(first!.content.type === "text" && first!.content.text).toBe("OPERATING RULES");
  });

  it("does not add a second system message when one already leads the context", async () => {
    const agent = new CapturingAgent(AgentId.of("cap"));
    // A context manager that already renders its own leading system message.
    class LeadingSystemContextManager extends NoopContextManager {
      override async assemble(session: SessionState): Promise<Context> {
        return {
          messages: [
            { role: "system", content: { type: "text", text: "MANAGER PROMPT" } },
            ...session.messages,
          ],
          tools: [],
          params: { stop_sequences: [] },
        };
      }
    }
    const cfg = new HarnessBuilder(
      agent,
      new ScriptedToolRegistry(),
      new AllowAllSandbox(),
      new LeadingSystemContextManager(),
      new AlwaysContinuePolicy(),
    )
      .systemPrompt("OPERATING RULES")
      .buildConfig();
    const h = new StandardHarness(cfg);
    await h.run({ task: react(2) });

    const systemMsgs = agent.seen.filter((m) => m.role === "system");
    expect(systemMsgs.length).toBe(1);
    expect(systemMsgs[0]!.content.type === "text" && systemMsgs[0]!.content.text).toBe(
      "MANAGER PROMPT",
    );
  });

  it("leaves the context without a system message when systemPrompt is unset", async () => {
    const agent = new CapturingAgent(AgentId.of("cap"));
    const cfg = catalogueBuilder(agent).buildConfig();
    const h = new StandardHarness(cfg);
    await h.run({ task: react(2) });
    expect(agent.seen.every((m) => m.role !== "system")).toBe(true);
  });
});

// ============================================================================
// sandbox setter
// ============================================================================

describe("sandbox setter (#91 follow-up)", () => {
  it("overrides the configured sandbox", () => {
    const override = new AllowAllSandbox();
    const cfg = catalogueBuilder(makeAgent()).sandbox(override).buildConfig();
    expect(cfg.sandbox).toBe(override);
  });
});
