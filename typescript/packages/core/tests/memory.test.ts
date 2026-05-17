/**
 * Unit tests for the canonical MemoryProvider (spore-core issue #8).
 *
 * Mirrors `rust/crates/spore-core/src/memory.rs#tests` — same rules,
 * same verdicts, parallel structure.
 */

import { describe, expect, it } from "vitest";

import { memory, SessionId } from "../src/index.js";

const { StandardMemoryProvider, MemoryId, Timestamp, MemoryError } = memory;

type SemanticMemory = memory.SemanticMemory;
type EpisodicMemory = memory.EpisodicMemory;
type MemoryQuery = memory.MemoryQuery;

// ── Builders ────────────────────────────────────────────────────────────────

function ts(s: string): memory.Timestamp {
  return new Timestamp(s);
}

function sem(id: string, content: string): SemanticMemory {
  return {
    id: new MemoryId(id),
    content,
    source: { kind: "manual" },
    domain: null,
    version: 1,
    previous_versions: [],
    created_at: ts("2026-05-16T00:00:00Z"),
    updated_at: ts("2026-05-16T00:00:00Z"),
    status: { kind: "active" },
  };
}

function epi(id: string, session: string, content: string): EpisodicMemory {
  return {
    id: new MemoryId(id),
    session_id: new SessionId(session),
    content,
    created_at: ts("2026-05-16T00:00:00Z"),
    tags: [],
  };
}

function makeQuery(overrides: Partial<MemoryQuery> & { task_instruction: string }): MemoryQuery {
  return {
    task_instruction: overrides.task_instruction,
    domain: overrides.domain ?? null,
    session_id: overrides.session_id ?? null,
    min_relevance: overrides.min_relevance ?? 0.5,
    max_items: overrides.max_items ?? 10,
  };
}

describe("MemoryProvider — separate stores", () => {
  it("episodic and semantic use separate stores", async () => {
    const mp = new StandardMemoryProvider();
    await mp.storeEpisodic(epi("e1", "s1", "ran tests"));
    await mp.storeSemantic(sem("g1", "always run tests"), "reject");

    const eps = await mp.getEpisodic(new SessionId("s1"));
    expect(eps).toHaveLength(1);
    // Episodic id should not be retrievable as semantic.
    await expect(mp.getSemantic(new MemoryId("e1"))).rejects.toBeInstanceOf(MemoryError);
    // Semantic id should not appear under any session's episodics.
    const none = await mp.getEpisodic(new SessionId("g1"));
    expect(none).toEqual([]);
  });
});

describe("MemoryProvider — semantic replace", () => {
  it("replace archives previous and bumps version", async () => {
    const mp = new StandardMemoryProvider();
    await mp.storeSemantic(sem("g1", "v1 content"), "reject");

    const v2 = sem("g1", "v2 content");
    v2.version = 1; // caller version is ignored; provider bumps from existing.
    await mp.storeSemantic(v2, "replace");

    const current = await mp.getSemantic(new MemoryId("g1"));
    expect(current.content).toBe("v2 content");
    expect(current.version).toBe(2);
    expect(current.previous_versions).toHaveLength(1);

    const history = await mp.getVersionHistory(new MemoryId("g1"));
    expect(history).toHaveLength(2);
    expect(history[0]!.content).toBe("v2 content");
    expect(history[1]!.content).toBe("v1 content");
  });

  it("replace chains versions across multiple updates", async () => {
    const mp = new StandardMemoryProvider();
    await mp.storeSemantic(sem("g1", "v1"), "reject");
    await mp.storeSemantic(sem("g1", "v2"), "replace");
    await mp.storeSemantic(sem("g1", "v3"), "replace");
    const cur = await mp.getSemantic(new MemoryId("g1"));
    expect(cur.version).toBe(3);
    const history = await mp.getVersionHistory(new MemoryId("g1"));
    expect(history).toHaveLength(3);
  });
});

describe("MemoryProvider — reject", () => {
  it("reject on conflict errors and leaves original untouched", async () => {
    const mp = new StandardMemoryProvider();
    await mp.storeSemantic(sem("g1", "first"), "reject");
    await expect(mp.storeSemantic(sem("g1", "second"), "reject")).rejects.toMatchObject({
      kind: "merge_conflict",
    });
    const cur = await mp.getSemantic(new MemoryId("g1"));
    expect(cur.content).toBe("first");
  });
});

describe("MemoryProvider — append", () => {
  it("append concatenates without new version", async () => {
    const mp = new StandardMemoryProvider();
    await mp.storeSemantic(sem("g1", "a"), "reject");
    await mp.storeSemantic(sem("g1", "b"), "append");
    const cur = await mp.getSemantic(new MemoryId("g1"));
    expect(cur.content).toBe("ab");
    expect(cur.version).toBe(1);
    expect(cur.previous_versions).toEqual([]);
  });
});

describe("MemoryProvider — validation", () => {
  it("empty content fails validation", async () => {
    const mp = new StandardMemoryProvider();
    await expect(mp.storeSemantic(sem("g1", "   "), "reject")).rejects.toMatchObject({
      kind: "validation_failed",
    });
    await expect(mp.storeEpisodic(epi("e1", "s1", ""))).rejects.toMatchObject({
      kind: "validation_failed",
    });
  });
});

describe("MemoryProvider — meta-agent forced review", () => {
  it("meta-agent memories forced to pending_review", async () => {
    const mp = new StandardMemoryProvider();
    const m = sem("g1", "proposed skill");
    m.source = { kind: "meta_agent_proposed", approved_by: null };
    m.status = { kind: "active" }; // caller dishonestly sets Active
    await mp.storeSemantic(m, "reject");
    const stored = await mp.getSemantic(new MemoryId("g1"));
    expect(stored.status.kind).toBe("pending_review");
  });
});

describe("MemoryProvider — query", () => {
  it("scores, filters by min_relevance, and sorts descending", async () => {
    const mp = new StandardMemoryProvider();
    await mp.storeSemantic(sem("g1", "rust async tokio runtime"), "reject");
    await mp.storeSemantic(sem("g2", "python pytest fixtures"), "reject");
    await mp.storeSemantic(sem("g3", "unrelated cooking recipe"), "reject");

    const res = await mp.query(
      makeQuery({ task_instruction: "rust tokio async", min_relevance: 0.1, max_items: 10 }),
    );
    expect(res.length).toBeGreaterThan(0);
    expect(res[0]!.memory.id.asString()).toBe("g1");
    for (let i = 1; i < res.length; i += 1) {
      expect(res[i - 1]!.relevance_score).toBeGreaterThanOrEqual(res[i]!.relevance_score);
    }
  });

  it("excludes deprecated memories", async () => {
    const mp = new StandardMemoryProvider();
    await mp.storeSemantic(sem("g1", "rust tokio"), "reject");
    await mp.deprecate(new MemoryId("g1"), "obsolete");
    const res = await mp.query(
      makeQuery({ task_instruction: "rust tokio", min_relevance: 0.0, max_items: 10 }),
    );
    expect(res).toEqual([]);
  });

  it("respects min_relevance and max_items", async () => {
    const mp = new StandardMemoryProvider();
    for (let i = 0; i < 5; i += 1) {
      await mp.storeSemantic(sem(`g${i}`, `alpha beta gamma ${i}`), "reject");
    }
    const none = await mp.query(
      makeQuery({ task_instruction: "alpha beta gamma", min_relevance: 0.99, max_items: 10 }),
    );
    expect(none).toEqual([]);
    const capped = await mp.query(
      makeQuery({ task_instruction: "alpha beta gamma", min_relevance: 0.0, max_items: 2 }),
    );
    expect(capped).toHaveLength(2);
  });

  it("filters by domain", async () => {
    const mp = new StandardMemoryProvider();
    const a = sem("a", "shared content");
    a.domain = "rust";
    const b = sem("b", "shared content");
    b.domain = "python";
    await mp.storeSemantic(a, "reject");
    await mp.storeSemantic(b, "reject");
    const res = await mp.query(
      makeQuery({
        task_instruction: "shared content",
        domain: "rust",
        min_relevance: 0.0,
        max_items: 10,
      }),
    );
    expect(res).toHaveLength(1);
    expect(res[0]!.memory.id.asString()).toBe("a");
  });
});

describe("MemoryProvider — lifecycle", () => {
  it("deprecate sets status", async () => {
    const mp = new StandardMemoryProvider();
    await mp.storeSemantic(sem("g1", "x"), "reject");
    await mp.deprecate(new MemoryId("g1"), "no longer needed");
    const m = await mp.getSemantic(new MemoryId("g1"));
    expect(m.status.kind).toBe("deprecated");
    if (m.status.kind === "deprecated") {
      expect(m.status.reason).toBe("no longer needed");
    }
  });

  it("deprecate unknown id returns not_found", async () => {
    const mp = new StandardMemoryProvider();
    await expect(mp.deprecate(new MemoryId("nope"), "r")).rejects.toMatchObject({
      kind: "not_found",
    });
  });

  it("markPendingReview changes status", async () => {
    const mp = new StandardMemoryProvider();
    await mp.storeSemantic(sem("g1", "x"), "reject");
    await mp.markPendingReview(new MemoryId("g1"));
    const m = await mp.getSemantic(new MemoryId("g1"));
    expect(m.status.kind).toBe("pending_review");
  });
});

describe("MemoryProvider — NotFound / empty", () => {
  it("getSemantic unknown returns not_found", async () => {
    const mp = new StandardMemoryProvider();
    await expect(mp.getSemantic(new MemoryId("nope"))).rejects.toMatchObject({
      kind: "not_found",
    });
  });

  it("getEpisodic unknown session returns empty", async () => {
    const mp = new StandardMemoryProvider();
    const res = await mp.getEpisodic(new SessionId("none"));
    expect(res).toEqual([]);
  });
});

describe("MemoryProvider — episodic ordering", () => {
  it("preserves insertion order across multiple writes", async () => {
    const mp = new StandardMemoryProvider();
    for (let i = 0; i < 5; i += 1) {
      await mp.storeEpisodic(epi(`e${i}`, "s1", `event ${i}`));
    }
    const eps = await mp.getEpisodic(new SessionId("s1"));
    expect(eps).toHaveLength(5);
    eps.forEach((e, i) => {
      expect(e.id.asString()).toBe(`e${i}`);
    });
  });
});
