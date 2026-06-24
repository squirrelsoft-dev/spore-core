/**
 * Skills module — SKILL.md parse, discovery, and the SkillCatalog (spore-core
 * issue #115 / SC-26).
 *
 * Mirrors `rust/crates/spore-core/src/skills.rs#tests` — same rules, parallel
 * structure. A skill is a `SKILL.md` (YAML frontmatter `name`/`description` +
 * markdown body); the catalog exposes them with progressive disclosure (manifest
 * tier 1, active bodies tier 2) as guides, plus a `load_skill` tool that shares
 * the catalog's sticky active set.
 */

import { describe, expect, it } from "vitest";

import {
  skills as skillsNs,
  toolOutput,
  type SandboxProvider,
  type ToolCall,
} from "../src/index.js";
import type { ToolContext } from "../src/tool-registry/types.js";

const { SkillCatalog, parseSkillDoc, LOAD_SKILL } = skillsNs;

// `load_skill` ignores both the sandbox and the storage ctx — minimal stand-ins.
const SANDBOX = {} as SandboxProvider;
const CTX = {} as ToolContext;

function call(input: unknown): ToolCall {
  return { id: "c1", name: LOAD_SKILL, input };
}

describe("parseSkillDoc", () => {
  it("parses frontmatter name, description, and body", () => {
    const doc =
      "---\nname: security-review\ndescription: Review code for security issues.\n---\n\n# Procedure\n\nDo the thing.\n";
    const entry = parseSkillDoc(doc);
    expect(entry).toBeDefined();
    expect(entry?.name).toBe("security-review");
    expect(entry?.description).toBe("Review code for security issues.");
    expect(entry?.body).toBe("# Procedure\n\nDo the thing.\n");
  });

  it("tolerates optional frontmatter fields and strips quotes from scalars", () => {
    const doc =
      "---\nname: \"pdf\"\ndescription: 'Handle PDFs.'\nlicense: Apache-2.0\nmetadata:\n  author: me\n---\nbody\n";
    const entry = parseSkillDoc(doc);
    expect(entry?.name).toBe("pdf");
    expect(entry?.description).toBe("Handle PDFs.");
    expect(entry?.body).toBe("body\n");
  });

  it("rejects missing name, empty body, or no frontmatter", () => {
    expect(parseSkillDoc("---\ndescription: no name\n---\nbody\n")).toBeUndefined();
    expect(parseSkillDoc("---\nname: x\ndescription: d\n---\n   \n")).toBeUndefined();
    expect(parseSkillDoc("no frontmatter at all")).toBeUndefined();
  });
});

describe("SkillCatalog", () => {
  function twoSkills() {
    return SkillCatalog.fromEntries([
      { name: "audit", description: "Audit a module.", body: "AUDIT BODY" },
      { name: "style", description: "Style guide.", body: "STYLE BODY" },
    ]);
  }

  it("an empty catalog yields no guides", () => {
    const cat = SkillCatalog.fromEntries([]);
    expect(cat.isEmpty()).toBe(true);
    expect(cat.activeGuides()).toEqual([]);
  });

  it("before activation: only the manifest guide (sorted, with preamble)", () => {
    const cat = twoSkills();
    const guides = cat.activeGuides();
    expect(guides).toHaveLength(1);
    expect(guides[0]!.id.toString()).toBe("AVAILABLE SKILLS");
    expect(guides[0]!.content).toContain("- audit: Audit a module.");
    expect(guides[0]!.content).toContain("- style: Style guide.");
    expect(guides[0]!.content.startsWith("Reusable procedures you can load on demand.")).toBe(true);
    // Entries are sorted by name.
    expect(cat.names()).toEqual(["audit", "style"]);
  });

  it("activate rejects unknown names and is sticky + idempotent", () => {
    const cat = twoSkills();
    expect(cat.activate("nope")).toBe(false);
    expect(cat.active()).toEqual([]);

    expect(cat.activate("audit")).toBe(true);
    expect(cat.active()).toEqual(["audit"]);
    // Idempotent.
    expect(cat.activate("audit")).toBe(true);
    expect(cat.active()).toEqual(["audit"]);
  });

  it("after activate: manifest then the active-body guide, in order", () => {
    const cat = twoSkills();
    cat.activate("audit");
    const guides = cat.activeGuides();
    expect(guides).toHaveLength(2);
    expect(guides[0]!.id.toString()).toBe("AVAILABLE SKILLS");
    expect(guides[1]!.id.toString()).toBe("ACTIVE SKILL — audit");
    expect(guides[1]!.content).toBe("AUDIT BODY");
  });

  it("clearActive drops the sticky bodies but keeps the manifest", () => {
    const cat = twoSkills();
    cat.activate("audit");
    cat.activate("style");
    expect(cat.activeGuides()).toHaveLength(3);
    cat.clearActive();
    const guides = cat.activeGuides();
    expect(guides).toHaveLength(1);
    expect(guides[0]!.id.toString()).toBe("AVAILABLE SKILLS");
  });

  describe("loadSkillTool", () => {
    it("schema is `load_skill` with a required string `name`", () => {
      const tool = twoSkills().loadSkillTool();
      expect(tool.schema.name).toBe(LOAD_SKILL);
      const params = tool.schema.parameters as {
        properties: { name: { type: string } };
        required: string[];
      };
      expect(params.properties.name.type).toBe("string");
      expect(params.required).toEqual(["name"]);
    });

    it("activates the shared catalog and returns the verbatim success string", async () => {
      const cat = twoSkills();
      const tool = cat.loadSkillTool();
      const out = await tool.implementation.execute(call({ name: "audit" }), SANDBOX, CTX);
      expect(out).toEqual(
        toolOutput.success(
          "Loaded skill 'audit' — its full procedure is now in your context. Follow it.",
        ),
      );
      // The activation is visible through the SAME catalog instance.
      expect(cat.active()).toEqual(["audit"]);
    });

    it("rejects an unknown skill with the verbatim error string", async () => {
      const tool = twoSkills().loadSkillTool();
      const out = await tool.implementation.execute(call({ name: "nope" }), SANDBOX, CTX);
      expect(out).toEqual(
        toolOutput.error("unknown skill 'nope'. Choose one listed in AVAILABLE SKILLS."),
      );
    });

    it("rejects a missing/blank name with the verbatim error string", async () => {
      const tool = twoSkills().loadSkillTool();
      const out = await tool.implementation.execute(call({ name: "   " }), SANDBOX, CTX);
      expect(out).toEqual(toolOutput.error("invalid parameters: `name` (string) is required"));
    });
  });
});
