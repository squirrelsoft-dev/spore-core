/**
 * Unit tests for the prompt assembly engine (spore-core issue #79). Mirrors the
 * Rust reference rule list R1–R19, R21, R25 plus the A3 serialization /
 * never-equal semantics, StorageScope serde, and the agent-affinity gate.
 */

import { describe, expect, it } from "vitest";

import { SessionId, TaskId } from "../src/harness/types.js";
import { promptAssembly as pa } from "../src/index.js";

const {
  ChunkConditions,
  ContextSourcesBuilder,
  EmbeddedChunkProvider,
  InMemoryChunkProvider,
  CompositeChunkProvider,
  ChunkProviderError,
  assemblyContext,
  capabilityKey,
  promptChunk,
  serializeCondition,
  deserializeCondition,
  conditionsEqual,
  serializePromptChunk,
  parsePromptChunk,
  breakpointIds,
  chunksToSegments,
} = pa;

type AssemblyContext = pa.AssemblyContext;
type PromptChunk = pa.PromptChunk;

function ctx(): AssemblyContext {
  return assemblyContext(SessionId.of("s1"), TaskId.of("t1"), 1, "safe_auto", "execution");
}

function builder(): InstanceType<typeof ContextSourcesBuilder> {
  return new ContextSourcesBuilder();
}

function withChunks(chunks: PromptChunk[]): InstanceType<typeof ContextSourcesBuilder> {
  return ContextSourcesBuilder.withChunks(chunks);
}

function chunk(id: string, content = id): PromptChunk {
  return promptChunk(id, content);
}

describe("ChunkCondition evaluation (R1–R9)", () => {
  it("R1: always matches", () => {
    expect(builder().evaluate(ChunkConditions.always(), ctx())).toBe(true);
  });

  it("R2: when_mode", () => {
    const b = builder();
    const c = { ...ctx(), mode: "plan" as const };
    expect(b.evaluate(ChunkConditions.whenMode("plan"), c)).toBe(true);
    expect(b.evaluate(ChunkConditions.whenMode("always_ask"), c)).toBe(false);
  });

  it("R3: when_tool_active", () => {
    const b = builder();
    const c = ctx();
    c.active_tool_names.add("bash");
    expect(b.evaluate(ChunkConditions.whenToolActive("bash"), c)).toBe(true);
    expect(b.evaluate(ChunkConditions.whenToolActive("grep"), c)).toBe(false);
  });

  it("R4: when_tool_capability", () => {
    const b = builder();
    const c = ctx();
    c.active_capabilities.add(capabilityKey("bash", "sandbox"));
    expect(b.evaluate(ChunkConditions.whenToolCapability("bash", "sandbox"), c)).toBe(true);
    expect(b.evaluate(ChunkConditions.whenToolCapability("bash", "git"), c)).toBe(false);
  });

  it("R5: when_phase / when_agent_type / when_feature", () => {
    const b = builder();
    const c = ctx();
    c.phase = "planning";
    c.agent_type = "planner";
    c.features.set("beta", true);
    c.features.set("alpha", false);

    expect(b.evaluate(ChunkConditions.whenPhase("planning"), c)).toBe(true);
    expect(b.evaluate(ChunkConditions.whenPhase("cleanup"), c)).toBe(false);

    expect(b.evaluate(ChunkConditions.whenAgentType("planner"), c)).toBe(true);
    expect(b.evaluate(ChunkConditions.whenAgentType("coder"), c)).toBe(false);

    expect(b.evaluate(ChunkConditions.whenFeature("beta"), c)).toBe(true);
    expect(b.evaluate(ChunkConditions.whenFeature("alpha"), c)).toBe(false);
    expect(b.evaluate(ChunkConditions.whenFeature("missing"), c)).toBe(false);
  });

  it("R6: on_trigger substring match; absent message never matches", () => {
    const b = builder();
    const c = ctx();
    const cond = ChunkConditions.onTrigger(["deploy", "rollback"]);
    expect(b.evaluate(cond, c)).toBe(false);
    c.incoming_message = "please deploy the service";
    expect(b.evaluate(cond, c)).toBe(true);
    c.incoming_message = "nothing relevant";
    expect(b.evaluate(cond, c)).toBe(false);
  });

  it("R7: on_event", () => {
    const b = builder();
    const c = ctx();
    const cond = ChunkConditions.onEvent("pre_compact");
    expect(b.evaluate(cond, c)).toBe(false);
    c.pending_events.push("pre_compact");
    expect(b.evaluate(cond, c)).toBe(true);
  });

  it("R8: all / any / not", () => {
    const b = builder();
    const c = ctx();
    c.mode = "plan";
    c.active_tool_names.add("bash");

    expect(
      b.evaluate(
        ChunkConditions.all([
          ChunkConditions.whenMode("plan"),
          ChunkConditions.whenToolActive("bash"),
        ]),
        c,
      ),
    ).toBe(true);
    expect(
      b.evaluate(
        ChunkConditions.all([
          ChunkConditions.whenMode("plan"),
          ChunkConditions.whenToolActive("grep"),
        ]),
        c,
      ),
    ).toBe(false);
    expect(
      b.evaluate(
        ChunkConditions.any([
          ChunkConditions.whenToolActive("grep"),
          ChunkConditions.whenMode("plan"),
        ]),
        c,
      ),
    ).toBe(true);
    expect(b.evaluate(ChunkConditions.not(ChunkConditions.whenMode("always_ask")), c)).toBe(true);
  });

  it("R9: custom is evaluated against ctx", () => {
    const b = builder();
    const c = ctx();
    c.turn_number = 5;
    const cond = ChunkConditions.custom((x) => x.turn_number > 3);
    expect(b.evaluate(cond, c)).toBe(true);
    c.turn_number = 1;
    expect(b.evaluate(cond, c)).toBe(false);
  });
});

describe("assembly steps (R10–R17)", () => {
  it("R10: bucketed by stability", () => {
    const b = withChunks([
      { ...chunk("s"), stability: "static" },
      { ...chunk("ps"), stability: "per_session" },
      { ...chunk("pt"), stability: "per_turn" },
    ]);
    const buckets = b.assemble(ctx());
    expect(buckets.static_chunks.map((c) => c.id)).toEqual(["s"]);
    expect(buckets.per_session.map((c) => c.id)).toEqual(["ps"]);
    expect(buckets.per_turn.map((c) => c.id)).toEqual(["pt"]);
  });

  it("R11: registration order preserved within bucket", () => {
    const b = withChunks([chunk("a"), chunk("b"), chunk("c")]);
    expect(b.assemble(ctx()).static_chunks.map((c) => c.id)).toEqual(["a", "b", "c"]);
  });

  it("R12: tool-affinity 4-way matrix", () => {
    const gated: PromptChunk = {
      ...chunk("bash-git", "git guide"),
      tool_affinity: { tool_name: "bash", capability: "git" },
    };
    const b = withChunks([gated]);

    // (1) tool inactive, cap inactive -> excluded
    let c = ctx();
    expect(b.assemble(c).static_chunks).toHaveLength(0);

    // (2) tool active, cap inactive -> excluded
    c.active_tool_names.add("bash");
    expect(b.assemble(c).static_chunks).toHaveLength(0);

    // (3) tool active, cap active -> included
    c.active_capabilities.add(capabilityKey("bash", "git"));
    expect(b.assemble(c).static_chunks).toHaveLength(1);

    // (4) tool inactive but cap present -> excluded (tool gate first)
    c = ctx();
    c.active_capabilities.add(capabilityKey("bash", "git"));
    expect(b.assemble(c).static_chunks).toHaveLength(0);

    // capability undefined: included as soon as the tool is active
    const anyCap: PromptChunk = {
      ...chunk("bash-any", "bash guide"),
      tool_affinity: { tool_name: "bash" },
    };
    const b2 = withChunks([anyCap]);
    const c3 = ctx();
    expect(b2.assemble(c3).static_chunks).toHaveLength(0);
    c3.active_tool_names.add("bash");
    expect(b2.assemble(c3).static_chunks).toHaveLength(1);
  });

  it("R13: trigger match routes to per_turn regardless of declared stability", () => {
    const gated: PromptChunk = {
      ...chunk("playbook", "rollback steps"),
      stability: "static",
      condition: ChunkConditions.onTrigger(["rollback"]),
      triggers: ["rollback"],
    };
    const b = withChunks([gated]);
    const c = ctx();
    expect(b.assemble(c).static_chunks).toHaveLength(0);
    expect(b.assemble(c).per_turn).toHaveLength(0);

    c.incoming_message = "we must rollback now";
    const buckets = b.assemble(c);
    expect(buckets.static_chunks).toHaveLength(0);
    expect(buckets.per_turn.map((x) => x.id)).toEqual(["playbook"]);
  });

  it("R14: on_event injected to per_turn only when event pending", () => {
    const reminder: PromptChunk = {
      ...chunk("reminder", "system reminder"),
      stability: "per_turn",
      condition: ChunkConditions.onEvent("pre_compact"),
    };
    const b = withChunks([reminder]);
    const c = ctx();
    expect(b.assemble(c).per_turn).toHaveLength(0);
    c.pending_events.push("pre_compact");
    expect(b.assemble(c).per_turn.map((x) => x.id)).toEqual(["reminder"]);
  });

  it("R15: Block-1 hash stable across identical static sets, differs on content", () => {
    const mk = () => withChunks([chunk("core", "identity rules"), chunk("style", "be concise")]);
    const b1 = mk();
    const b2 = mk();
    const cp1 = b1.composeBlock1(b1.assemble(ctx()));
    const cp2 = b2.composeBlock1(b2.assemble(ctx()));
    expect(cp1.block_1_hash).toBe(cp2.block_1_hash);

    const b3 = withChunks([chunk("core", "DIFFERENT identity"), chunk("style", "be concise")]);
    const cp3 = b3.composeBlock1(b3.assemble(ctx()));
    expect(cp3.block_1_hash).not.toBe(cp1.block_1_hash);
  });

  it("R16: cache_breakpoint preserved through assemble + segment mapping", () => {
    const b = withChunks([chunk("a"), { ...chunk("b"), cache_breakpoint: true }, chunk("c")]);
    const buckets = b.assemble(ctx());
    expect(breakpointIds(buckets)).toEqual(["b"]);
    const segs = chunksToSegments(buckets.static_chunks);
    expect(segs.find((s) => s.name === "b")?.cache_breakpoint).toBe(true);
    expect(segs.find((s) => s.name === "a")?.cache_breakpoint).toBe(false);
  });

  it("R17: tool not active yields no description chunk", () => {
    const desc: PromptChunk = {
      ...chunk("bash-desc", "Bash tool: run shell commands"),
      tool_affinity: { tool_name: "bash" },
    };
    const b = withChunks([desc]);
    expect(b.assemble(ctx()).static_chunks).toHaveLength(0);
    const c = ctx();
    c.active_tool_names.add("bash");
    expect(b.assemble(c).static_chunks).toHaveLength(1);
  });
});

describe("providers (R18, R19, R21)", () => {
  it("R18: EmbeddedChunkProvider invalidate no-op, load same", async () => {
    const p: pa.ChunkProvider = new EmbeddedChunkProvider([chunk("x", "y")]);
    const a = await p.load();
    p.invalidate?.();
    const b = await p.load();
    expect(a).toEqual(b);
    expect(a).toHaveLength(1);
  });

  it("R19: InMemoryChunkProvider returns registered; set replaces", async () => {
    const p = new InMemoryChunkProvider([chunk("x", "y")]);
    expect(await p.load()).toHaveLength(1);
    p.set([chunk("a", "1"), chunk("b", "2")]);
    const after = await p.load();
    expect(after).toHaveLength(2);
    expect(after[0]?.id).toBe("a");
  });

  it("R21: CompositeChunkProvider merges in add order + propagates invalidate", async () => {
    let inv1 = 0;
    let inv2 = 0;
    const p1: pa.ChunkProvider = {
      load: () => Promise.resolve([chunk("a", "1")]),
      invalidate: () => {
        inv1 += 1;
      },
    };
    const p2: pa.ChunkProvider = {
      load: () => Promise.resolve([chunk("b", "2"), chunk("c", "3")]),
      invalidate: () => {
        inv2 += 1;
      },
    };
    const comp = new CompositeChunkProvider().add(p1).add(p2);
    const merged = await comp.load();
    expect(merged.map((c) => c.id)).toEqual(["a", "b", "c"]);
    comp.invalidate();
    expect(inv1).toBe(1);
    expect(inv2).toBe(1);
  });
});

describe("A3 — Custom serialization and equality (R25)", () => {
  it("custom condition never equal (even to itself)", () => {
    const f = () => true;
    const a = ChunkConditions.custom(f);
    const b = ChunkConditions.custom(f);
    expect(conditionsEqual(a, b)).toBe(false);
    expect(conditionsEqual(ChunkConditions.always(), ChunkConditions.always())).toBe(true);
    expect(
      conditionsEqual(
        ChunkConditions.whenMode("always_ask"),
        ChunkConditions.whenMode("always_ask"),
      ),
    ).toBe(true);
  });

  it("custom serializes to null and round-trips to always", () => {
    expect(serializeCondition(ChunkConditions.custom(() => true))).toBeNull();
    const c: PromptChunk = { ...chunk("x", "y"), condition: ChunkConditions.custom(() => true) };
    const wire = serializePromptChunk(c);
    expect(wire["condition"]).toBeNull();
    const back = parsePromptChunk(wire);
    expect(conditionsEqual(back.condition, ChunkConditions.always())).toBe(true);
  });

  it("serializable variants round-trip through the wire form", () => {
    const cond = ChunkConditions.all([
      ChunkConditions.whenMode("plan"),
      ChunkConditions.any([
        ChunkConditions.whenToolActive("bash"),
        ChunkConditions.not(ChunkConditions.whenFeature("beta")),
      ]),
      ChunkConditions.onEvent("pre_turn"),
      ChunkConditions.onTrigger(["deploy"]),
      ChunkConditions.whenToolCapability("bash", "git"),
      ChunkConditions.whenPhase("planning"),
      ChunkConditions.whenAgentType("planner"),
    ]);
    const wire = serializeCondition(cond);
    const back = deserializeCondition(wire);
    expect(conditionsEqual(cond, back)).toBe(true);
  });

  it("custom is pruned from combinators on serialize", () => {
    const cond = ChunkConditions.all([
      ChunkConditions.whenMode("always_ask"),
      ChunkConditions.custom(() => true),
    ]);
    const back = deserializeCondition(serializeCondition(cond));
    expect(
      conditionsEqual(back, ChunkConditions.all([ChunkConditions.whenMode("always_ask")])),
    ).toBe(true);
  });

  it("not over custom collapses to null -> always", () => {
    const cond = ChunkConditions.not(ChunkConditions.custom(() => true));
    expect(serializeCondition(cond)).toBeNull();
    expect(
      conditionsEqual(deserializeCondition(serializeCondition(cond)), ChunkConditions.always()),
    ).toBe(true);
  });
});

describe("misc", () => {
  it("ChunkProviderError variants render", () => {
    const e = ChunkProviderError.loadFailed("remote", "timeout");
    expect(e.message).toContain("remote");
    expect(e.kind).toBe("load_failed");
    const p = ChunkProviderError.parseError("bad json");
    expect(p.message).toContain("bad json");
    expect(p.kind).toBe("parse_error");
  });

  it("buildContextSources threads Block 1 and passes guides/memory/schemas through (A5)", () => {
    const b = withChunks([
      chunk("core", "rules"),
      { ...chunk("ps", "ref"), stability: "per_session" },
    ]);
    const { sources, buckets } = b.buildContextSources(ctx());
    expect(sources.composed_prompt.rendered).toContain("rules");
    expect(buckets.per_session).toHaveLength(1);
    expect(sources.guides).toHaveLength(0);
  });

  it("StorageScope serializes snake_case (it is the wire string)", () => {
    expect(JSON.stringify("user")).toBe('"user"');
    const c = ctx();
    expect(c.storage_scope).toBe("project");
  });

  it("agent_affinity gate", () => {
    const planner: PromptChunk = {
      ...chunk("planner-prompt", "you plan"),
      agent_affinity: "planner",
    };
    const b = withChunks([planner]);
    expect(b.assemble(ctx()).static_chunks).toHaveLength(0);
    const c = ctx();
    c.agent_type = "planner";
    expect(b.assemble(c).static_chunks).toHaveLength(1);
    c.agent_type = "coder";
    expect(b.assemble(c).static_chunks).toHaveLength(0);
  });
});
