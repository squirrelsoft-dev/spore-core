/**
 * Unit tests for {@link PromptChunkRegistry} (spore-core issue #24).
 *
 * Covers every rule in the spec: registration validation, compose ordering
 * and missing-slot errors, block-hash stability, validate() rules, Mode
 * helpers, ComposedPrompt rendering, and the standard chunk library.
 */

import { describe, expect, it } from "vitest";

import { promptChunkRegistry as pcr } from "../src/index.js";

const {
  ChunkId,
  StandardPromptChunkRegistry,
  hashContent,
  modeApprovalPolicy,
  modeDefaultToolPhase,
  modePromptChunk,
  promptChunk,
  recomputeBlockHashes,
  renderComposed,
  renderedStr,
  standardChunks,
} = pcr;

function registryWithRole(id: string): InstanceType<typeof StandardPromptChunkRegistry> {
  const r = new StandardPromptChunkRegistry();
  expect(r.register(promptChunk(id, "you are a test agent", "role", "static"))).toBeNull();
  return r;
}

// ── Rule: register rejects duplicate ids ───────────────────────────────────

describe("register", () => {
  it("rejects duplicate ids", () => {
    const r = new StandardPromptChunkRegistry();
    expect(r.register(promptChunk("x", "hello", "capability", "static"))).toBeNull();
    const err = r.register(promptChunk("x", "world", "capability", "static"));
    expect(err?.kind).toBe("duplicate_id");
  });

  it("rejects empty content", () => {
    const r = new StandardPromptChunkRegistry();
    const err = r.register(promptChunk("x", "   ", "capability", "static"));
    expect(err?.kind).toBe("invalid_slot");
  });

  it("rejects Budget slot with Static cache block", () => {
    const r = new StandardPromptChunkRegistry();
    const err = r.register(promptChunk("b", "budget warning", "budget", "static"));
    expect(err?.kind).toBe("conflicting_cache_block");
    if (err && err.kind === "conflicting_cache_block") {
      expect(err.slot).toBe("budget");
      expect(err.expected).toBe("per_turn");
      expect(err.actual).toBe("static");
    }
  });

  it("rejects Ephemeral slot with PerSession cache block", () => {
    const r = new StandardPromptChunkRegistry();
    const err = r.register(promptChunk("e", "ephemeral", "ephemeral", "per_session"));
    expect(err?.kind).toBe("conflicting_cache_block");
  });

  it("rejects Role slot with non-Static cache block", () => {
    const r = new StandardPromptChunkRegistry();
    const err = r.register(promptChunk("r", "role", "role", "per_session"));
    expect(err?.kind).toBe("conflicting_cache_block");
  });

  it("rejects Mode slot with non-Static cache block", () => {
    const r = new StandardPromptChunkRegistry();
    const err = r.register(promptChunk("m", "mode", "mode", "per_turn"));
    expect(err?.kind).toBe("conflicting_cache_block");
  });
});

// ── Rule: compose ───────────────────────────────────────────────────────────

describe("compose", () => {
  it("returns missing-role error when role chunk is not registered", () => {
    const r = new StandardPromptChunkRegistry();
    const res = r.compose(new ChunkId("missing"), "yolo", [], []);
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.errors.some((e) => e.kind === "missing_required_slot" && e.slot === "role")).toBe(
        true,
      );
    }
  });

  it("includes the mode chunk via modePromptChunk()", () => {
    const r = registryWithRole("role-test");
    const res = r.compose(new ChunkId("role-test"), "plan", [], []);
    expect(res.ok).toBe(true);
    if (res.ok) {
      const mode = res.composed.chunks.find((c) => c.slot === "mode");
      expect(mode?.id.value).toBe("mode-plan");
    }
  });

  it("orders chunks by slot (Role, Mode, Capability, Skill)", () => {
    const r = registryWithRole("role-test");
    expect(r.register(promptChunk("cap-1", "cap one", "capability", "static"))).toBeNull();
    expect(r.register(promptChunk("skill-1", "skill one", "skill", "static"))).toBeNull();
    const res = r.compose(
      new ChunkId("role-test"),
      "auto_edit",
      [new ChunkId("cap-1")],
      [new ChunkId("skill-1")],
    );
    expect(res.ok).toBe(true);
    if (res.ok) {
      expect(res.composed.chunks.map((c) => c.slot)).toEqual([
        "role",
        "mode",
        "capability",
        "skill",
      ]);
    }
  });

  it("returns missing-capability error when capability chunk is absent", () => {
    const r = registryWithRole("role-test");
    const res = r.compose(new ChunkId("role-test"), "yolo", [new ChunkId("nope")], []);
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(
        res.errors.some((e) => e.kind === "missing_required_slot" && e.slot === "capability"),
      ).toBe(true);
    }
  });
});

// ── Rule: block hashes ──────────────────────────────────────────────────────

describe("block hashes", () => {
  it("are stable for identical content", () => {
    const a = registryWithRole("role-test").compose(new ChunkId("role-test"), "yolo", [], []);
    const b = registryWithRole("role-test").compose(new ChunkId("role-test"), "yolo", [], []);
    expect(a.ok && b.ok).toBe(true);
    if (a.ok && b.ok) {
      expect(a.composed.block_1_hash).toBe(b.composed.block_1_hash);
      expect(a.composed.block_2_hash).toBe(b.composed.block_2_hash);
    }
  });

  it("block_1_hash changes when Static content changes", () => {
    const a = registryWithRole("role-test").compose(new ChunkId("role-test"), "yolo", [], []);
    const r2 = new StandardPromptChunkRegistry();
    r2.register(promptChunk("role-test", "DIFFERENT ROLE CONTENT", "role", "static"));
    const b = r2.compose(new ChunkId("role-test"), "yolo", [], []);
    expect(a.ok && b.ok).toBe(true);
    if (a.ok && b.ok) {
      expect(a.composed.block_1_hash).not.toBe(b.composed.block_1_hash);
    }
  });

  it("recomputeBlockHashes agrees with the cached hashes", () => {
    const res = registryWithRole("role-test").compose(new ChunkId("role-test"), "yolo", [], []);
    expect(res.ok).toBe(true);
    if (res.ok) {
      const [b1, b2] = recomputeBlockHashes(res.composed);
      expect(b1).toBe(res.composed.block_1_hash);
      expect(b2).toBe(res.composed.block_2_hash);
    }
  });

  it("hashContent is deterministic and stable", () => {
    expect(hashContent("hello")).toBe(hashContent("hello"));
    expect(hashContent("hello")).not.toBe(hashContent("Hello"));
  });
});

// ── Rule: validate flags malformed compositions ─────────────────────────────

describe("validate", () => {
  it("flags a PerTurn chunk wrongly placed in the Static block", () => {
    const r = new StandardPromptChunkRegistry();
    const composed = {
      chunks: [
        promptChunk("role-x", "x", "role", "static"),
        modePromptChunk("yolo"),
        // A Budget chunk with Static cache_block — simulates a bug.
        {
          id: new ChunkId("bad-budget"),
          content: "b",
          slot: "budget" as const,
          cache_block: "static" as const,
          hash: 0,
        },
      ],
      block_1_hash: 0,
      block_2_hash: 0,
      rendered: null,
    };
    const errs = r.validate(composed);
    expect(
      errs.some((e) => e.kind === "per_turn_chunk_in_static_block" && e.id.value === "bad-budget"),
    ).toBe(true);
  });

  it("flags more than one Mode chunk", () => {
    const r = new StandardPromptChunkRegistry();
    const composed = {
      chunks: [
        promptChunk("role-x", "x", "role", "static"),
        modePromptChunk("yolo"),
        modePromptChunk("always_ask"),
      ],
      block_1_hash: 0,
      block_2_hash: 0,
      rendered: null,
    };
    const errs = r.validate(composed);
    expect(errs.some((e) => e.kind === "conflicting_mode_chunks")).toBe(true);
  });

  it("flags missing Role slot", () => {
    const r = new StandardPromptChunkRegistry();
    const composed = {
      chunks: [modePromptChunk("yolo")],
      block_1_hash: 0,
      block_2_hash: 0,
      rendered: null,
    };
    const errs = r.validate(composed);
    expect(errs.some((e) => e.kind === "missing_required_slot" && e.slot === "role")).toBe(true);
  });

  it("flags missing Mode slot", () => {
    const r = new StandardPromptChunkRegistry();
    const composed = {
      chunks: [promptChunk("role-x", "x", "role", "static")],
      block_1_hash: 0,
      block_2_hash: 0,
      rendered: null,
    };
    const errs = r.validate(composed);
    expect(errs.some((e) => e.kind === "missing_required_slot" && e.slot === "mode")).toBe(true);
  });
});

// ── get / lookup ────────────────────────────────────────────────────────────

describe("get", () => {
  it("returns registered chunks and undefined for unknown ids", () => {
    const r = registryWithRole("role-x");
    expect(r.get(new ChunkId("role-x"))?.id.value).toBe("role-x");
    expect(r.get(new ChunkId("nope"))).toBeUndefined();
  });
});

// ── Mode helpers ────────────────────────────────────────────────────────────

describe("Mode helpers", () => {
  it("approvalPolicy matches the spec", () => {
    expect(modeApprovalPolicy("always_ask")).toBe("always_ask");
    expect(modeApprovalPolicy("auto_edit")).toBe("auto_explain");
    expect(modeApprovalPolicy("plan")).toBe("plan_only");
    expect(modeApprovalPolicy("safe_auto")).toBe("safe_auto");
    expect(modeApprovalPolicy("yolo")).toBe("none");
  });

  it("defaultToolPhase returns planning only for Plan mode", () => {
    expect(modeDefaultToolPhase("plan")).toBe("planning");
    expect(modeDefaultToolPhase("yolo")).toBe("execution");
    expect(modeDefaultToolPhase("safe_auto")).toBe("execution");
  });

  it("promptChunk ids match the cross-language fixture", () => {
    expect(modePromptChunk("always_ask").id.value).toBe("mode-always-ask");
    expect(modePromptChunk("auto_edit").id.value).toBe("mode-auto-edit");
    expect(modePromptChunk("plan").id.value).toBe("mode-plan");
    expect(modePromptChunk("safe_auto").id.value).toBe("mode-safe-auto");
    expect(modePromptChunk("yolo").id.value).toBe("mode-yolo");
  });
});

// ── ComposedPrompt rendering ────────────────────────────────────────────────

describe("renderComposed", () => {
  it("joins chunk contents with a blank line and caches the result", () => {
    const r = registryWithRole("role-test");
    const res = r.compose(new ChunkId("role-test"), "yolo", [], []);
    expect(res.ok).toBe(true);
    if (res.ok) {
      expect(res.composed.rendered).toBeNull();
      expect(renderedStr(res.composed)).toBe("");
      const rendered = renderComposed(res.composed);
      expect(rendered).toContain("you are a test agent");
      expect(rendered).toContain("Mode: Yolo");
      expect(res.composed.rendered).not.toBeNull();
      expect(renderedStr(res.composed)).toBe(rendered);
    }
  });
});

// ── Standard library bootstraps cleanly ─────────────────────────────────────

describe("standard chunk library", () => {
  it("registers cleanly and produces the canonical coding-agent composition", () => {
    const r = new StandardPromptChunkRegistry();
    expect(r.registerStandardChunks()).toBeNull();
    expect(r.get(new ChunkId("role-coding-agent"))).toBeDefined();
    expect(r.get(new ChunkId("capability-bash"))).toBeDefined();
    expect(r.get(new ChunkId("skill-testing"))).toBeDefined();

    const res = r.compose(
      new ChunkId("role-coding-agent"),
      "safe_auto",
      [
        new ChunkId("capability-bash"),
        new ChunkId("capability-filesystem"),
        new ChunkId("capability-git"),
      ],
      [new ChunkId("skill-testing"), new ChunkId("skill-security-review")],
    );
    expect(res.ok).toBe(true);
    if (res.ok) {
      expect(res.composed.chunks).toHaveLength(1 + 1 + 3 + 2);
      expect(res.composed.chunks[0]?.slot).toBe("role");
      expect(res.composed.chunks[1]?.slot).toBe("mode");
      expect(res.composed.chunks[1]?.id.value).toBe("mode-safe-auto");
    }
  });

  it("includes one chunk per Mode variant", () => {
    const chunks = standardChunks();
    const modeIds = chunks.filter((c) => c.slot === "mode").map((c) => c.id.value);
    expect(modeIds.sort()).toEqual(
      ["mode-always-ask", "mode-auto-edit", "mode-plan", "mode-safe-auto", "mode-yolo"].sort(),
    );
  });
});
