/**
 * Unit tests for the canonical GuideRegistry (spore-core issue #9).
 *
 * Mirrors `rust/crates/spore-core/src/guide_registry.rs#tests` one-for-one
 * (17 tests) — same rules, same verdicts, parallel structure.
 */

import { describe, expect, it } from "vitest";

import { guideRegistry, SessionId } from "../src/index.js";

const { StandardGuideRegistry, GuideId, Timestamp, GuideRegistryError, newGuideQuery } =
  guideRegistry;

type Guide = guideRegistry.Guide;
type GuideUsageRecord = guideRegistry.GuideUsageRecord;
type SessionOutcome = guideRegistry.SessionOutcome;

// ── Builders ────────────────────────────────────────────────────────────────

function ts(s: string): guideRegistry.Timestamp {
  return new Timestamp(s);
}

function makeGuide(id: string, content: string): Guide {
  return {
    id: new GuideId(id),
    name: id,
    content,
    guide_type: "skill",
    domain: null,
    source: { kind: "manual" },
    status: { kind: "active" },
    created_at: ts("2026-05-16T00:00:00Z"),
    last_used: null,
    version: 1,
  };
}

function usage(gid: string, sid: string, outcome: SessionOutcome): GuideUsageRecord {
  return {
    guide_id: new GuideId(gid),
    session_id: new SessionId(sid),
    task_domain: null,
    outcome,
    recorded_at: ts("2026-05-16T00:00:00Z"),
  };
}

// ── Tests ───────────────────────────────────────────────────────────────────

describe("GuideRegistry — register validates content", () => {
  it("register_empty_content_fails", async () => {
    const r = new StandardGuideRegistry();
    const g = makeGuide("g1", "   ");
    await expect(r.register(g)).rejects.toBeInstanceOf(GuideRegistryError);
    try {
      await r.register(g);
    } catch (e) {
      expect((e as guideRegistry.GuideRegistryError).kind).toBe("validation_failed");
    }
  });
});

describe("GuideRegistry — MetaAgentProposed forces PendingReview", () => {
  it("meta_agent_source_forces_pending_review", async () => {
    const r = new StandardGuideRegistry();
    const g = makeGuide("g1", "proposed");
    g.source = { kind: "meta_agent_proposed", proposed_at: ts("2026-05-16T01:00:00Z") };
    g.status = { kind: "active" }; // caller lies; provider corrects
    await r.register(g);

    const sel = await r.select(newGuideQuery("anything"));
    expect(sel.every((x) => x.id.value !== "g1")).toBe(true);

    const peeked = r._peekGuide(new GuideId("g1"));
    expect(peeked?.status.kind).toBe("pending_review");
    if (peeked?.status.kind === "pending_review") {
      expect(peeked.status.reason.kind).toBe("automated_proposal");
    }
  });
});

describe("GuideRegistry — select filters by status/domain/type", () => {
  it("select_filters_by_status_domain_and_type", async () => {
    const r = new StandardGuideRegistry();
    const a = makeGuide("a", "rust async tokio runtime");
    a.domain = "rust";
    const b = makeGuide("b", "pytest fixtures python");
    b.domain = "python";
    b.guide_type = "convention_doc";
    const c = makeGuide("c", "deprecated content");
    c.status = { kind: "deprecated", reason: "old", at: ts("2026-05-16T00:00:00Z") };

    await r.register(a);
    await r.register(b);
    // Inject c by registering then deprecating to avoid status validation.
    const cActive = makeGuide("c", "deprecated content");
    await r.register(cActive);
    await r.deprecate(new GuideId("c"), "old");

    const res = await r.select({
      task_instruction: "rust tokio",
      domain: "rust",
      phase: null,
      guide_types: ["skill"],
    });
    expect(res).toHaveLength(1);
    expect(res[0]!.id.value).toBe("a");
  });
});

describe("GuideRegistry — select sorted by relevance", () => {
  it("select_sorted_by_relevance", async () => {
    const r = new StandardGuideRegistry();
    await r.register(makeGuide("a", "alpha beta gamma delta"));
    await r.register(makeGuide("b", "zebra"));
    await r.register(makeGuide("c", "alpha beta"));
    const res = await r.select(newGuideQuery("alpha beta"));
    expect(res[0]!.id.value).toBe("c");
    expect(res[res.length - 1]!.id.value).toBe("b");
  });
});

describe("GuideRegistry — conflict detection at registration", () => {
  it("register_detects_conflict_in_same_domain", async () => {
    const r = new StandardGuideRegistry();
    const existing = makeGuide("a", "always run tests before commit");
    existing.domain = "rust";
    await r.register(existing);

    const conflicting = makeGuide("b", "always run tests before committing");
    conflicting.domain = "rust";
    try {
      await r.register(conflicting);
      throw new Error("expected ConflictDetected");
    } catch (e) {
      expect(e).toBeInstanceOf(GuideRegistryError);
      const err = e as guideRegistry.GuideRegistryError;
      expect(err.kind).toBe("conflict_detected");
      if (err.detail.kind === "conflict_detected") {
        expect(err.detail.conflict.guide_a.value).toBe("b");
        expect(err.detail.conflict.guide_b.value).toBe("a");
      }
    }
  });

  it("no_conflict_across_domains", async () => {
    const r = new StandardGuideRegistry();
    const a = makeGuide("a", "always run tests before commit");
    a.domain = "rust";
    await r.register(a);
    const b = makeGuide("b", "always run tests before commit");
    b.domain = "python";
    await r.register(b);
  });
});

describe("GuideRegistry — record_usage", () => {
  it("record_usage_requires_known_guide", async () => {
    const r = new StandardGuideRegistry();
    try {
      await r.recordUsage(usage("nope", "s1", { kind: "success" }));
      throw new Error("expected NotFound");
    } catch (e) {
      expect(e).toBeInstanceOf(GuideRegistryError);
      expect((e as guideRegistry.GuideRegistryError).kind).toBe("not_found");
    }
  });

  it("record_usage_updates_last_used", async () => {
    const r = new StandardGuideRegistry();
    await r.register(makeGuide("a", "x"));
    const u = usage("a", "s1", { kind: "success" });
    u.recorded_at = ts("2026-06-01T00:00:00Z");
    await r.recordUsage(u);

    const hist = await r.usageHistory(new GuideId("a"));
    expect(hist).toHaveLength(1);

    const peeked = r._peekGuide(new GuideId("a"));
    expect(peeked?.last_used?.value).toBe("2026-06-01T00:00:00Z");
  });
});

describe("GuideRegistry — deprecate", () => {
  it("deprecate_sets_status_and_404s_on_missing", async () => {
    const r = new StandardGuideRegistry();
    await r.register(makeGuide("a", "x"));
    await r.deprecate(new GuideId("a"), "obsolete");
    const peeked = r._peekGuide(new GuideId("a"));
    expect(peeked?.status.kind).toBe("deprecated");
    if (peeked?.status.kind === "deprecated") {
      expect(peeked.status.reason).toBe("obsolete");
    }
    try {
      await r.deprecate(new GuideId("nope"), "x");
      throw new Error("expected NotFound");
    } catch (e) {
      expect((e as guideRegistry.GuideRegistryError).kind).toBe("not_found");
    }
  });
});

describe("GuideRegistry — promote_to_active only from PendingReview", () => {
  it("promote_to_active_only_from_pending_review", async () => {
    const r = new StandardGuideRegistry();
    await r.register(makeGuide("a", "x"));
    try {
      await r.promoteToActive(new GuideId("a"));
      throw new Error("expected ValidationFailed");
    } catch (e) {
      expect((e as guideRegistry.GuideRegistryError).kind).toBe("validation_failed");
    }
    await r.markPendingReview(new GuideId("a"), { kind: "manual_flag", note: "x" });
    await r.promoteToActive(new GuideId("a"));
    const peeked = r._peekGuide(new GuideId("a"));
    expect(peeked?.status.kind).toBe("active");
  });
});

describe("GuideRegistry — analyze_performance", () => {
  it("analyze_performance_flags_high_failure_rate", async () => {
    const r = new StandardGuideRegistry();
    r.setNow(ts("2026-05-16T01:00:00Z"));
    await r.register(makeGuide("a", "x"));
    await r.register(makeGuide("b", "y"));
    for (let i = 0; i < 3; i++) {
      await r.recordUsage(usage("a", `s${i}`, { kind: "failure", reason: "boom" }));
    }
    for (let i = 0; i < 3; i++) {
      await r.recordUsage(usage("b", `sb${i}`, { kind: "success" }));
    }
    const signals = await r.analyzePerformance(86_400, 0.5, 100);
    const dep = signals.some(
      (s) => s.kind === "guide_deprecation_recommended" && s.guide_id.value === "a",
    );
    expect(dep).toBe(true);
  });

  it("analyze_performance_emits_skill_generation_for_repeated_pattern", async () => {
    const r = new StandardGuideRegistry();
    r.setNow(ts("2026-05-16T01:00:00Z"));
    await r.register(makeGuide("a", "x"));
    for (let i = 0; i < 4; i++) {
      await r.recordUsage(
        usage("a", `s${i}`, {
          kind: "failure",
          reason: "panic: index out of bounds",
        }),
      );
    }
    const signals = await r.analyzePerformance(86_400, 999.0, 3);
    const gen = signals.some(
      (s) =>
        s.kind === "skill_generation_needed" &&
        s.pattern === "panic: index out of bounds" &&
        s.session_ids.length === 4,
    );
    expect(gen).toBe(true);
  });

  it("analyze_performance_filters_by_window", async () => {
    const r = new StandardGuideRegistry();
    r.setNow(ts("2026-05-16T00:00:00Z"));
    await r.register(makeGuide("a", "x"));
    const old = usage("a", "s0", { kind: "failure", reason: "old-pattern" });
    old.recorded_at = ts("2020-01-01T00:00:00Z");
    await r.recordUsage(old);
    const signals = await r.analyzePerformance(3600, 0.0, 1);
    const anyPattern = signals.some((s) => s.kind === "skill_generation_needed");
    expect(anyPattern).toBe(false);
  });
});

describe("GuideRegistry — usage_history filters to one guide", () => {
  it("usage_history_filters_to_one_guide", async () => {
    const r = new StandardGuideRegistry();
    await r.register(makeGuide("a", "x"));
    await r.register(makeGuide("b", "y"));
    await r.recordUsage(usage("a", "s1", { kind: "success" }));
    await r.recordUsage(usage("b", "s2", { kind: "success" }));
    const h = await r.usageHistory(new GuideId("a"));
    expect(h).toHaveLength(1);
    expect(h[0]!.guide_id.value).toBe("a");
  });
});

describe("GuideRegistry — check_conflicts external API", () => {
  it("check_conflicts_does_not_flag_identical_content", async () => {
    const r = new StandardGuideRegistry();
    await r.register(makeGuide("a", "same exact content"));
    const conflicts = await r.checkConflicts("same exact content", null);
    expect(conflicts).toEqual([]);
  });
});

describe("GuideRegistry — select empty when no guides", () => {
  it("select_empty_when_no_guides", async () => {
    const r = new StandardGuideRegistry();
    const res = await r.select(newGuideQuery("anything"));
    expect(res).toEqual([]);
  });
});

describe("GuideRegistry — date arithmetic sanity", () => {
  it("rfc3339_round_trip", async () => {
    // Validate via analyzePerformance window math: a record exactly at `now`
    // is in-window; a record before `now - window` is not. This exercises the
    // same parse/format round-trip the Rust test asserts directly.
    const r = new StandardGuideRegistry();
    r.setNow(ts("2026-05-16T12:34:56Z"));
    await r.register(makeGuide("a", "x"));
    const u1 = usage("a", "s1", { kind: "failure", reason: "p1" });
    u1.recorded_at = ts("2026-05-16T12:34:56Z");
    await r.recordUsage(u1);
    const u2 = usage("a", "s2", { kind: "failure", reason: "p2" });
    u2.recorded_at = ts("2026-05-16T12:00:00Z");
    await r.recordUsage(u2);

    // 60-second window: only u1 should be in-window.
    const signals = await r.analyzePerformance(60, 999.0, 1);
    const patterns = signals
      .filter((s) => s.kind === "skill_generation_needed")
      .map((s) => (s as { pattern: string }).pattern);
    expect(patterns).toContain("p1");
    expect(patterns).not.toContain("p2");
  });
});
