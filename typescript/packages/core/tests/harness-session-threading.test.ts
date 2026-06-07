/**
 * Unit tests for opt-in conversation-history threading + auto-persist
 * (spore-core issue #102).
 *
 * Mirrors `rust/crates/spore-core/src/harness.rs#tests::session_threading` —
 * same rules, same verdicts, parallel structure. Covers:
 *   - off-by-default: ZERO session-store I/O + lossless Success messages;
 *   - Success.session_state lossless for a tool-using run;
 *   - auto-persist round-trip;
 *   - auto-load by session_id across two runs (in-memory store);
 *   - cross-process continuity (FileSystemStorageProvider under a tempdir);
 *   - explicit session_state WINS over auto-load (no get_session);
 *   - Failure also carries session_state.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  AgentId,
  HarnessConfig,
  MockAgent,
  SessionId,
  StandardHarness,
  newTask,
  runResultSessionState,
  storage,
  type Message,
  type PausedState,
  type SessionState,
  type Task,
  type TokenUsage,
  type ToolResultRecord,
  type TurnResult,
} from "../src/index.js";

import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

const { InMemoryStorageProvider, FileSystemStorageProvider, NoOpStorageProvider, StorageProvider } =
  storage;
type SessionStore = storage.SessionStore;

// ----------------------------------------------------------------------------
// Doubles
// ----------------------------------------------------------------------------

function makeAgent(): MockAgent {
  return new MockAgent(AgentId.of("test"));
}

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function fr(content: string): TurnResult {
  return { kind: "final_response", content, usage: usage() };
}

/**
 * A {@link ContextManager} that records every appended message (assistant turns
 * and tool results) into the live session so tests can assert the conversation
 * the loop built. Without `appendAssistantMessage` the harness silently drops
 * the assistant tool-call / final-text turns — so a lossless test MUST use a
 * recording manager (the Noop double does not record assistant turns).
 */
class RecordingContextManager {
  async assemble(session: SessionState, _task: Task) {
    return { messages: session.messages.slice(), tools: [], params: { stop_sequences: [] } };
  }
  async appendToolResult(session: SessionState, result: ToolResultRecord): Promise<void> {
    const content =
      result.output.kind === "success"
        ? result.output.content
        : result.output.kind === "error"
          ? result.output.message
          : "";
    session.messages.push({
      role: "tool",
      content: {
        type: "tool_result",
        tool_use_id: result.call_id,
        content,
        is_error: result.output.kind === "error",
      },
    });
  }
  async appendUserMessage(session: SessionState, text: string): Promise<void> {
    session.messages.push({ role: "user", content: { type: "text", text } });
  }
  async appendAssistantMessage(session: SessionState, message: Message): Promise<void> {
    session.messages.push(message);
  }
  shouldCompact(_session: SessionState): boolean {
    return false;
  }
}

/**
 * In-memory {@link SessionStore} that COUNTS every get/put so a test can assert
 * "zero session-store I/O when auto_persist is disabled". Delegates real storage
 * to a wrapped {@link InMemoryStorageProvider}.
 */
class CountingSessionStore implements SessionStore {
  readonly inner = new InMemoryStorageProvider();
  gets = 0;
  puts = 0;
  async getSession(id: SessionId): Promise<PausedState | undefined> {
    this.gets += 1;
    return this.inner.getSession(id);
  }
  async putSession(id: SessionId, state: PausedState): Promise<void> {
    this.puts += 1;
    return this.inner.putSession(id, state);
  }
  async deleteSession(id: SessionId): Promise<void> {
    return this.inner.deleteSession(id);
  }
  async listSessions(): Promise<SessionId[]> {
    return this.inner.listSessions();
  }
}

/** A {@link StorageProvider} whose session slot is `store` and whose other three
 *  slots are no-ops (so only session I/O is observed). */
function countingProvider(store: CountingSessionStore): storage.StorageProvider {
  const noop = new NoOpStorageProvider();
  return StorageProvider.of(store, noop, noop, noop);
}

function baseConfig(agent: MockAgent): HarnessConfig {
  return {
    registry: registryWith({ agent }),
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new RecordingContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
  };
}

/** A MockAgent that requests one tool call, then replies with final text. */
function toolThenFinalAgent(): MockAgent {
  const a = makeAgent();
  a.push({
    kind: "tool_call_requested",
    calls: [{ id: "c1", name: "x", input: {} }],
    usage: usage(),
  });
  a.push(fr("after-tool"));
  return a;
}

function toolRegistry(): ScriptedToolRegistry {
  return new ScriptedToolRegistry().push({ kind: "success", content: "tool-ok" });
}

function reactTask(instruction: string, sid: SessionId, max = 5): Task {
  return newTask(instruction, sid, {
    kind: "react",
    budget: { kind: "per_loop", value: max },
    agent: "",
    toolset: "",
  });
}

function textsOf(state: SessionState): string[] {
  return state.messages.flatMap((m) => (m.content.type === "text" ? [m.content.text] : []));
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

describe("Harness — session-state threading (#102)", () => {
  const tempDirs: string[] = [];
  afterEach(() => {
    for (const d of tempDirs.splice(0)) rmSync(d, { recursive: true, force: true });
  });

  it("off-by-default: ZERO session-store I/O and the message flow is identical", async () => {
    const store = new CountingSessionStore();
    const cfg = baseConfig(toolThenFinalAgent());
    cfg.toolRegistry = toolRegistry();
    cfg.storage = countingProvider(store);
    // autoPersistSessions defaults to false (absent on baseConfig).
    expect(cfg.autoPersistSessions).toBeUndefined();
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: reactTask("do something", SessionId.of("s1")) });

    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.output).toBe("after-tool");
      // The new field is populated even when persistence is off.
      const state = runResultSessionState(r);
      expect(state.messages.some((m) => m.content.type === "tool_call")).toBe(true);
    }
    expect(store.gets, "no getSession calls").toBe(0);
    expect(store.puts, "no putSession calls").toBe(0);
  });

  it("Success.session_state is LOSSLESS for a tool-using run", async () => {
    const cfg = baseConfig(toolThenFinalAgent());
    cfg.toolRegistry = toolRegistry();
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: reactTask("do something", SessionId.of("s1")) });

    expect(r.kind).toBe("success");
    if (r.kind !== "success") return;
    const msgs = runResultSessionState(r).messages;
    // user instruction
    expect(msgs.some((m) => m.role === "user")).toBe(true);
    // assistant tool-call turn
    expect(msgs.some((m) => m.content.type === "tool_call")).toBe(true);
    // tool-result turn
    expect(msgs.some((m) => m.role === "tool")).toBe(true);
    // final assistant text
    expect(
      msgs.some(
        (m) =>
          m.role === "assistant" && m.content.type === "text" && m.content.text === "after-tool",
      ),
    ).toBe(true);
  });

  it("auto-persist round-trip: getSession returns the final history with empty pending fields", async () => {
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const cfg = baseConfig(toolThenFinalAgent());
    cfg.toolRegistry = toolRegistry();
    cfg.storage = provider;
    cfg.autoPersistSessions = true;
    const h = new StandardHarness(cfg);
    const sid = SessionId.of("s1");
    await h.run({ task: reactTask("do something", sid) });

    const stored = await provider.session().getSession(sid);
    expect(stored, "session persisted").toBeDefined();
    expect(stored!.session_state.messages.some((m) => m.content.type === "tool_call")).toBe(true);
    expect(stored!.pending_tool_calls).toEqual([]);
    expect(stored!.human_request).toBeUndefined();
    expect(stored!.child_state).toBeNull();
  });

  it("auto-load by session_id across two runs (in-memory store) carries history forward", async () => {
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const sid = SessionId.of("shared");

    // Run 1: one final response, auto-persisted.
    {
      const a = makeAgent();
      a.push(fr("first"));
      const cfg = baseConfig(a);
      cfg.storage = provider;
      cfg.autoPersistSessions = true;
      await new StandardHarness(cfg).run({ task: reactTask("turn one", sid) });
    }

    // Run 2: same session_id, no explicit state. The loaded history carries forward.
    const a = makeAgent();
    a.push(fr("second"));
    const cfg = baseConfig(a);
    cfg.storage = provider;
    cfg.autoPersistSessions = true;
    const r = await new StandardHarness(cfg).run({ task: reactTask("turn two", sid) });

    expect(r.kind).toBe("success");
    if (r.kind !== "success") return;
    const texts = textsOf(runResultSessionState(r));
    expect(texts, "prior turn carried").toContain("first");
    expect(texts).toContain("turn one");
    expect(texts).toContain("turn two");
    expect(texts).toContain("second");
  });

  it("cross-process continuity: a fresh harness+provider over the same dir resumes by session_id", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-102-"));
    tempDirs.push(dir);
    const sid = SessionId.of("fs-session");

    // "Process 1": its own harness + provider instance.
    {
      const provider = StorageProvider.single(new FileSystemStorageProvider(dir));
      const a = makeAgent();
      a.push(fr("process-one"));
      const cfg = baseConfig(a);
      cfg.storage = provider;
      cfg.autoPersistSessions = true;
      await new StandardHarness(cfg).run({ task: reactTask("first process", sid) });
    }

    // "Process 2": brand-new provider over the SAME dir, brand-new harness.
    const provider = StorageProvider.single(new FileSystemStorageProvider(dir));
    const a = makeAgent();
    a.push(fr("process-two"));
    const cfg = baseConfig(a);
    cfg.storage = provider;
    cfg.autoPersistSessions = true;
    const r = await new StandardHarness(cfg).run({ task: reactTask("second process", sid) });

    expect(r.kind).toBe("success");
    if (r.kind !== "success") return;
    const texts = textsOf(runResultSessionState(r));
    expect(texts, "prior process history loaded").toContain("process-one");
    expect(texts).toContain("process-two");
  });

  it("explicit session_state WINS over auto-load (no getSession call)", async () => {
    const store = new CountingSessionStore();
    // Pre-seed the store with a DIFFERENT history under the session id.
    const sid = SessionId.of("s1");
    await store.inner.putSession(sid, {
      session_id: sid,
      task_id: newTask("", sid, {
        kind: "react",
        budget: { kind: "per_loop", value: 0 },
        agent: "",
        toolset: "",
      }).id,
      turn_number: 0,
      session_state: {
        messages: [{ role: "user", content: { type: "text", text: "STORED-history" } }],
        extras: {},
      },
      pending_tool_calls: [],
      approved_results: [],
      task: newTask("", sid, {
        kind: "react",
        budget: { kind: "per_loop", value: 0 },
        agent: "",
        toolset: "",
      }),
      budget_used: { turns: 0, input_tokens: 0, output_tokens: 0, cost_usd: 0 },
      child_state: null,
    });
    store.gets = 0;
    store.puts = 0; // reset the pre-seed write

    const a = makeAgent();
    a.push(fr("done"));
    const cfg = baseConfig(a);
    cfg.storage = countingProvider(store);
    cfg.autoPersistSessions = true;
    const h = new StandardHarness(cfg);

    const explicit: SessionState = {
      messages: [{ role: "user", content: { type: "text", text: "EXPLICIT-history" } }],
      extras: {},
    };
    const r = await h.run({ task: reactTask("turn", sid), session_state: explicit });

    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      const texts = textsOf(runResultSessionState(r));
      expect(texts).toContain("EXPLICIT-history");
      expect(texts, "load skipped").not.toContain("STORED-history");
    }
    expect(store.gets, "explicit state skips the auto-load getSession").toBe(0);
  });

  it("Failure ALSO carries session_state", async () => {
    // A tool annotated always_halt fails the run after the assistant tool-call
    // turn was recorded into session_state.
    const a = makeAgent();
    a.push({
      kind: "tool_call_requested",
      calls: [{ id: "c", name: "danger", input: {} }],
      usage: usage(),
    });
    const cfg = baseConfig(a);
    cfg.toolRegistry = new ScriptedToolRegistry().markAlwaysHalt("danger");
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: reactTask("do something", SessionId.of("s1")) });

    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      const msgs = runResultSessionState(r).messages;
      // The seeded user instruction is present in the failure state.
      expect(msgs.some((m) => m.role === "user" && m.content.type === "text")).toBe(true);
    }
  });
});
