/**
 * Unit tests for the canonical ContextManager (spore-core issue #7).
 *
 * Mirrors `rust/crates/spore-core/src/context.rs#tests` — same rules,
 * same verdicts, parallel structure.
 */

import { describe, expect, it, vi } from "vitest";

import { context, SessionId, TaskId } from "../src/index.js";
import type {
  ModelInterface,
  ModelRequest,
  ModelResponse,
  ProviderInfo,
  StreamEvent,
  ToolSchema,
} from "../src/index.js";
import type { SandboxProvider, ToolCall, SandboxViolation, TruncatedOutput } from "../src/index.js";

const {
  StandardContextManager,
  NullCacheProvider,
  GuideId,
  newSessionState,
  defaultCompactionConfig,
  ContextErrorException,
} = context;

type ContextSources = context.ContextSources;
type SessionState = context.SessionState;
type Guide = context.Guide;
type Context = context.Context;
type CacheProvider = context.CacheProvider;

// ── Test doubles ───────────────────────────────────────────────────────────

class FakeModel implements ModelInterface {
  tokens = 100;
  async call(_req: ModelRequest): Promise<ModelResponse> {
    return {
      content: [],
      stop_reason: "end_turn",
      usage: { input_tokens: 0, output_tokens: 0 },
    };
  }
  callStreaming(_req: ModelRequest): AsyncIterable<StreamEvent> {
    throw new Error("not implemented");
  }
  async countTokens(_req: ModelRequest): Promise<number> {
    return this.tokens;
  }
  provider(): ProviderInfo {
    return { name: "fake", model_id: "fake", context_window: 200_000 };
  }
}

class FailingModel implements ModelInterface {
  async call(_req: ModelRequest): Promise<ModelResponse> {
    throw new Error("nope");
  }
  callStreaming(_req: ModelRequest): AsyncIterable<StreamEvent> {
    throw new Error("not implemented");
  }
  async countTokens(_req: ModelRequest): Promise<number> {
    throw new Error("token boom");
  }
  provider(): ProviderInfo {
    return { name: "fail", model_id: "fail", context_window: 1 };
  }
}

class CountingCache implements CacheProvider {
  calls = 0;
  supportsCaching(): boolean {
    return true;
  }
  annotate(_ctx: Context): void {
    this.calls += 1;
  }
}

class PassthroughSandbox implements SandboxProvider {
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
  async handleLargeOutput(
    content: string,
    callId: string,
    headTokens: number,
    tailTokens: number,
  ): Promise<TruncatedOutput> {
    const headChars = headTokens * 4;
    const tailChars = tailTokens * 4;
    if (content.length <= headChars + tailChars) {
      return {
        content,
        truncated: false,
        full_ref: null,
        original_size: content.length,
      };
    }
    const head = content.slice(0, headChars);
    const tail = content.slice(-tailChars);
    return {
      content: `${head}\n…\n${tail}`,
      truncated: true,
      full_ref: { path: `/tmp/${callId}.txt`, size: content.length },
      original_size: content.length,
    };
  }
}

// ── Builders ───────────────────────────────────────────────────────────────

function sources(rendered: string, hash: number, schemas: ToolSchema[] = []): ContextSources {
  return {
    guides: [],
    memory: [],
    tool_schemas: schemas,
    composed_prompt: { rendered, block_1_hash: hash },
  };
}

function state(): SessionState {
  const s = newSessionState(SessionId.of("s1"), TaskId.of("t1"), "do the thing");
  s.window_limit = 1000;
  s.token_budget_used = 100;
  return s;
}

function mk(): context.ContextManager {
  return new StandardContextManager(
    new FakeModel(),
    new NullCacheProvider(),
    defaultCompactionConfig(),
  );
}

// ── Tests ──────────────────────────────────────────────────────────────────

describe("StandardContextManager.assemble", () => {
  // Rule: Assemble before every turn — token count comes from ModelInterface.
  it("returns context with token count from the model", async () => {
    const mgr = mk();
    const ctx = await mgr.assemble(state(), sources("BLOCK1", 0xab));
    expect(ctx.token_count).toBe(100);
    expect(ctx.window_limit).toBe(1000);
    expect(Math.abs(ctx.utilization - 0.1)).toBeLessThan(1e-6);
  });

  // Rule: Block 1 hash invariance — mismatch is an error.
  it("Block 1 hash mismatch is an error", async () => {
    const mgr = mk();
    await mgr.assemble(state(), sources("BLOCK1", 0xab));
    let caught: unknown = null;
    try {
      await mgr.assemble(state(), sources("BLOCK1", 0xcd));
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(ContextErrorException);
    const err = (caught as InstanceType<typeof ContextErrorException>).error;
    expect(err.kind).toBe("CacheHashMismatch");
    if (err.kind === "CacheHashMismatch") {
      expect(err.block).toBe("static");
    }
  });

  // Rule: session block hash change mid-session emits a warning.
  it("warns when session block changes mid-session", async () => {
    const mgr = mk();
    const s1 = state();
    s1.turn_number = 0;
    await mgr.assemble(s1, sources("BLOCK1", 0xab));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const s2 = state();
    s2.turn_number = 2;
    s2.task_instruction = "different task"; // changes session hash
    await mgr.assemble(s2, sources("BLOCK1", 0xab));
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  // Rule: Tool schemas sorted by name.
  it("sorts tool schemas by name", async () => {
    const mgr = mk();
    const schemas: ToolSchema[] = [
      { name: "zebra", description: "", input_schema: {} },
      { name: "apple", description: "", input_schema: {} },
      { name: "mango", description: "", input_schema: {} },
    ];
    const ctx = await mgr.assemble(state(), sources("BLOCK1", 0xab, schemas));
    expect(ctx.tool_schemas.map((s) => s.name)).toEqual(["apple", "mango", "zebra"]);
  });

  // Rule: pending skill injections appear in meta and Block 3.
  it("pending skill injections appear in meta and content", async () => {
    const mgr = mk();
    const s = state();
    s.pending_skill_injections.push({ id: GuideId.of("g1"), content: "do x" });
    const ctx = await mgr.assemble(s, sources("BLOCK1", 0xab));
    expect(ctx.meta.skills_injected.map((g) => g.asString())).toEqual(["g1"]);
    expect(ctx.system_prompt.content).toContain("do x");
  });

  // Rule: budget warning is in Block 3 only when active.
  it("includes budget warning only when active", async () => {
    const mgr = mk();
    const s = state();
    const off = await mgr.assemble(s, sources("BLOCK1", 0xab));
    expect(off.system_prompt.content).not.toContain("[BUDGET]");
    s.budget_warning_active = true;
    const on = await mgr.assemble(s, sources("BLOCK1", 0xab));
    expect(on.system_prompt.content).toContain("[BUDGET]");
  });

  // Rule: TokenCountFailed surfaces when ModelInterface fails.
  it("surfaces TokenCountFailed when count_tokens fails", async () => {
    const mgr = new StandardContextManager(
      new FailingModel(),
      new NullCacheProvider(),
      defaultCompactionConfig(),
    );
    let caught: unknown = null;
    try {
      await mgr.assemble(state(), sources("BLOCK1", 0xab));
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(ContextErrorException);
    expect((caught as InstanceType<typeof ContextErrorException>).error.kind).toBe(
      "TokenCountFailed",
    );
  });

  // Rule: CacheProvider.annotate is invoked at the end of assemble.
  it("invokes CacheProvider.annotate each assemble", async () => {
    const cache = new CountingCache();
    const mgr = new StandardContextManager(new FakeModel(), cache, defaultCompactionConfig());
    const s = state();
    const src = sources("BLOCK1", 0xab);
    await mgr.assemble(s, src);
    await mgr.assemble(s, src);
    expect(cache.calls).toBe(2);
  });

  // Cache-stability invariant: identical inputs ⇒ identical prefix bytes & hashes.
  it("produces a deterministic prefix across calls", async () => {
    const mgr = mk();
    const s = state();
    const src = sources("BLOCK1-content", 0x11);
    const a = await mgr.assemble(s, src);
    const b = await mgr.assemble(s, src);
    expect(a.system_prompt.content).toBe(b.system_prompt.content);
    expect(a.system_prompt.static_block_hash).toBe(b.system_prompt.static_block_hash);
    expect(a.system_prompt.session_block_hash).toBe(b.system_prompt.session_block_hash);
  });
});

describe("StandardContextManager.shouldCompact", () => {
  it("triggers at the default 80% threshold", () => {
    const mgr = mk();
    const s = state();
    s.window_limit = 1000;
    s.token_budget_used = 799;
    expect(mgr.shouldCompact(s)).toBe(false);
    s.token_budget_used = 800;
    expect(mgr.shouldCompact(s)).toBe(true);
    s.token_budget_used = 900;
    expect(mgr.shouldCompact(s)).toBe(true);
  });

  it("returns false when window_limit is 0", () => {
    const mgr = mk();
    const s = state();
    s.window_limit = 0;
    expect(mgr.shouldCompact(s)).toBe(false);
  });
});

describe("StandardContextManager compaction", () => {
  it("prepareCompaction keeps recent N and uses default preserve hints", () => {
    const mgr = mk();
    const s = state();
    for (let i = 0; i < 20; i += 1) {
      s.message_history.push({
        role: "assistant",
        content: { type: "text", text: `m${i}` },
      });
    }
    const req = mgr.prepareCompaction(s);
    expect(req.messages_to_compact.length).toBe(12);
    expect(req.preserve_hints.keep_thinking_blocks).toBe(true);
    expect(req.preserve_hints.keep_architectural_decisions).toBe(true);
    expect(req.preserve_hints.keep_open_problems).toBe(true);
  });

  it("applyCompaction replaces old slice with summary + recents", () => {
    const mgr = mk();
    const s = state();
    for (let i = 0; i < 20; i += 1) {
      s.message_history.push({
        role: "assistant",
        content: { type: "text", text: `m${i}` },
      });
    }
    s.token_budget_used = 800;
    mgr.applyCompaction(s, {
      summary_message: { role: "assistant", content: { type: "text", text: "summary" } },
      tokens_reclaimed: 500,
      messages_removed: 12,
    });
    expect(s.message_history.length).toBe(9); // 1 summary + 8 preserved recents
    expect(s.token_budget_used).toBe(300);
    const first = s.message_history[0]!.content;
    expect(first.type === "text" && first.text === "summary").toBe(true);
  });

  it("applyCompaction fails when history is shorter than preserve_recent_n", () => {
    const mgr = mk();
    const s = state();
    for (let i = 0; i < 4; i += 1) {
      s.message_history.push({
        role: "assistant",
        content: { type: "text", text: `m${i}` },
      });
    }
    let caught: unknown = null;
    try {
      mgr.applyCompaction(s, {
        summary_message: { role: "assistant", content: { type: "text", text: "x" } },
        tokens_reclaimed: 0,
        messages_removed: 0,
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(ContextErrorException);
    expect((caught as InstanceType<typeof ContextErrorException>).error.kind).toBe(
      "CompactionFailed",
    );
  });
});

describe("StandardContextManager.appendToolResult", () => {
  // Rule: head+tail truncate large content via SandboxProvider.handleLargeOutput.
  it("truncates large output via the sandbox", async () => {
    const mgr = new StandardContextManager(
      new FakeModel(),
      new NullCacheProvider(),
      defaultCompactionConfig(),
      { offloadThresholdBytes: 64 },
    );
    const s = state();
    const sb = new PassthroughSandbox();
    const big = "x".repeat(8 * 1024);
    await mgr.appendToolResult(
      s,
      { call_id: "c1", output: { kind: "success", content: big, truncated: false } },
      sb,
    );
    expect(s.message_history.length).toBe(1);
    const c = s.message_history[0]!.content;
    expect(c.type).toBe("text");
    if (c.type === "text") {
      expect(c.text).toContain("[truncated");
      expect(c.text.length).toBeLessThan(big.length);
    }
  });

  it("small output passes through untouched", async () => {
    const mgr = mk();
    const s = state();
    await mgr.appendToolResult(
      s,
      { call_id: "c1", output: { kind: "success", content: "hello", truncated: false } },
      new PassthroughSandbox(),
    );
    const c = s.message_history[0]!.content;
    expect(c.type === "text" && c.text === "hello").toBe(true);
  });

  it("error output is rendered with an [error] prefix", async () => {
    const mgr = mk();
    const s = state();
    await mgr.appendToolResult(
      s,
      { call_id: "c1", output: { kind: "error", message: "boom", recoverable: false } },
      new PassthroughSandbox(),
    );
    const c = s.message_history[0]!.content;
    expect(c.type === "text" && c.text === "[error] boom").toBe(true);
  });
});

describe("StandardContextManager.appendResponse", () => {
  // Rule: appends an assistant message to history.
  it("pushes an assistant message", () => {
    const mgr = mk();
    const s = state();
    mgr.appendResponse(s, "ack");
    expect(s.message_history.length).toBe(1);
    expect(s.message_history[0]!.role).toBe("assistant");
  });
});

describe("StandardContextManager.injectSkill", () => {
  // Rule: ephemeral — no history mutation, no cache invalidation.
  it("does not touch history or static hashes", async () => {
    const mgr = mk();
    const ctx = await mgr.assemble(state(), sources("BLOCK1", 0xab));
    const beforeStatic = ctx.system_prompt.static_block_hash;
    const beforeSession = ctx.system_prompt.session_block_hash;
    const beforeMessages = ctx.messages.length;
    mgr.injectSkill(ctx, { id: GuideId.of("rust-style"), content: "prefer iterators" });
    expect(ctx.system_prompt.static_block_hash).toBe(beforeStatic);
    expect(ctx.system_prompt.session_block_hash).toBe(beforeSession);
    expect(ctx.messages.length).toBe(beforeMessages);
    expect(ctx.system_prompt.content).toContain("[SKILL:rust-style]");
    expect(ctx.meta.skills_injected.length).toBe(1);
  });
});

describe("StandardContextManager.recordCacheResult", () => {
  // Rule: updates ContextMeta.cache_blocks.
  it("updates ContextMeta.cache_blocks", async () => {
    const mgr = mk();
    const ctx = await mgr.assemble(state(), sources("BLOCK1", 0xab));
    mgr.recordCacheResult(ctx, { static_hit: true, session_hit: false, history_hit: true });
    expect(ctx.meta.cache_blocks.static_hit).toBe(true);
    expect(ctx.meta.cache_blocks.session_hit).toBe(false);
    expect(ctx.meta.cache_blocks.history_hit).toBe(true);
  });
});
