/**
 * Issue #115 / SC-26 — **skills** baked into the harness.
 *
 * A *skill* is a directory with a `SKILL.md`: YAML frontmatter (`name` +
 * `description`, plus optional fields) followed by a markdown procedure body.
 * This module discovers them and exposes them to the agent with **progressive
 * disclosure** (the [Agent Skills spec](https://agentskills.io/specification)):
 *
 * 1. **Metadata (tier 1).** The `name` + `description` of every discovered skill
 *    is injected every turn as a compact manifest — cheap, always present, so the
 *    agent knows what exists.
 * 2. **Instructions (tier 2).** A skill's full body is injected only once it is
 *    **active**, and then it stays injected every turn (sticky).
 *
 * Unlike the pre-#115 architect-side shim (which injected skills as ad-hoc User
 * messages via a wrapping context manager), the catalog feeds the rich
 * {@link "../context/types.js".ContextSources} seam: {@link SkillCatalog.activeGuides}
 * returns the manifest + active bodies as {@link "../context/types.js".Guide}s,
 * which the harness places in the structural System block. A skill becomes active
 * when the agent calls the {@link SkillCatalog.loadSkillTool} tool (or the host
 * activates it).
 *
 * The active set is a private `Set<string>` held inside the {@link SkillCatalog},
 * so the `load_skill` tool and the per-turn `activeGuides()` read the SAME set
 * within a harness's lifetime (the catalog is held by both the tool and
 * `HarnessConfig`). JS is single-threaded, so no lock is needed.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { GuideId, type Guide } from "../context/types.js";
import type { SandboxProvider, ToolOutput } from "../harness/types.js";
import { toolOutput } from "../harness/types.js";
import type { ToolCall } from "../model/schemas.js";
import {
  defaultToolAnnotations,
  type StandardTool,
  type Tool,
  type ToolContext,
} from "../tool-registry/types.js";

/** The registered name of the skill-activation tool. */
export const LOAD_SKILL = "load_skill";

// ============================================================================
// Skill model + discovery
// ============================================================================

/**
 * One parsed skill. `name` is the identity (matches the skill's directory name
 * per the spec); `description` is the one-line manifest entry (where the spec
 * puts trigger keywords); `body` is the markdown procedure injected on load.
 */
export interface SkillEntry {
  name: string;
  description: string;
  body: string;
}

/**
 * Parse a `SKILL.md`: optional `---`-delimited YAML frontmatter carrying at
 * least `name` and `description`, then the markdown body. Dependency-free and
 * minimal — enough for the spec's required fields; optional fields are tolerated
 * and ignored. Returns `undefined` if there is no usable `name` or the body is
 * empty.
 */
export function parseSkillDoc(content: string): SkillEntry | undefined {
  const trimmed = content.replace(/^\s+/, "");
  let name: string | undefined;
  let description = "";
  let body: string;

  if (trimmed.startsWith("---")) {
    const rest = trimmed.slice(3);
    const idx = rest.indexOf("\n---");
    const front = idx >= 0 ? rest.slice(0, idx) : rest;
    body = idx >= 0 ? rest.slice(idx + 4).replace(/^\n+/, "") : "";
    name = yamlScalar(front, "name");
    description = yamlScalar(front, "description") ?? "";
  } else {
    body = trimmed;
  }

  if (!name || name.trim() === "" || body.trim() === "") return undefined;
  return {
    name: name.trim(),
    description: description.trim(),
    body,
  };
}

/**
 * Pull a single top-level `key: value` scalar from a YAML frontmatter block,
 * stripping surrounding quotes. Not a general YAML parser — nested maps are
 * skipped, since their indented children don't match a top-level `key:` line.
 */
function yamlScalar(front: string, key: string): string | undefined {
  for (const raw of front.split("\n")) {
    const line = raw.trim();
    if (!line.startsWith(key)) continue;
    const afterKey = line.slice(key.length).trimStart();
    if (!afterKey.startsWith(":")) continue;
    const value = afterKey.slice(1).trim();
    return stripQuotes(value);
  }
  return undefined;
}

function stripQuotes(s: string): string {
  if (
    s.length >= 2 &&
    ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'")))
  ) {
    return s.slice(1, -1);
  }
  return s;
}

/**
 * Scan one `skills/` directory: each immediate `<child>/SKILL.md` is a
 * candidate. A missing/unreadable directory or file is skipped silently.
 */
function scanSkillDir(dir: string): SkillEntry[] {
  const out: SkillEntry[] = [];
  if (!existsSync(dir)) return out;
  let children: string[];
  try {
    children = readdirSync(dir);
  } catch {
    return out;
  }
  for (const child of children) {
    const skillMd = join(dir, child, "SKILL.md");
    if (!existsSync(skillMd)) continue;
    try {
      const entry = parseSkillDoc(readFileSync(skillMd, "utf8"));
      if (entry) out.push(entry);
    } catch {
      // unreadable file — skip it
    }
  }
  return out;
}

/** Insert-or-replace by `name`, so a later source overrides an earlier one. */
function upsert(entries: SkillEntry[], entry: SkillEntry): void {
  const slot = entries.findIndex((e) => e.name === entry.name);
  if (slot >= 0) entries[slot] = entry;
  else entries.push(entry);
}

// ============================================================================
// Catalog
// ============================================================================

/**
 * The fixed manifest preamble (tier 1). Copied verbatim from the Rust reference
 * (`rust/crates/spore-core/src/skills.rs`) for cross-language acceptance parity.
 */
const MANIFEST_PREAMBLE =
  "Reusable procedures you can load on demand. When the user's request matches " +
  "one's description, call load_skill(name) BEFORE acting, then follow the " +
  "procedure it injects. Active skills' full bodies are included below and stay " +
  "in context every turn:\n";

/** The `load_skill` tool description. Copied verbatim from the Rust reference. */
const LOAD_SKILL_DESCRIPTION =
  "Activate a skill by name so its full procedure stays in your context for the " +
  "rest of the session. Call this BEFORE acting when the user's request matches " +
  "a skill in the AVAILABLE SKILLS manifest.";

/**
 * The discovered skill catalog plus the sticky active-skill set. Held by both
 * the `load_skill` tool (which activates skills) and the harness (which reads
 * {@link activeGuides} each turn), so both share the same active set for the
 * catalog's lifetime.
 */
export class SkillCatalog {
  private readonly _entries: SkillEntry[];
  private readonly _active = new Set<string>();

  private constructor(entries: SkillEntry[]) {
    this._entries = entries;
  }

  /**
   * Build a catalog from already-parsed entries (sorted by name for stable
   * output; later duplicates by name win).
   */
  static fromEntries(entries: Iterable<SkillEntry>): SkillCatalog {
    const acc: SkillEntry[] = [];
    for (const e of entries) upsert(acc, e);
    acc.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    return new SkillCatalog(acc);
  }

  /**
   * Discover skills from `extraDirs` (e.g. a host's bundled `skills/`) plus the
   * conventional `<workspaceRoot>/.spore/skills` and `~/.spore/skills`, in that
   * precedence order (last wins on a name clash). Discovery happens once.
   */
  static discover(extraDirs: string[], workspaceRoot: string): SkillCatalog {
    const dirs: string[] = [...extraDirs];
    dirs.push(join(workspaceRoot, ".spore", "skills"));
    dirs.push(join(homedir(), ".spore", "skills"));
    const entries: SkillEntry[] = [];
    for (const dir of dirs) {
      for (const entry of scanSkillDir(dir)) {
        upsert(entries, entry);
      }
    }
    return SkillCatalog.fromEntries(entries);
  }

  /** Skill names, for host-driven activation and listing. */
  names(): string[] {
    return this._entries.map((e) => e.name);
  }

  /** The full entries, for listing. */
  entries(): SkillEntry[] {
    return this._entries.slice();
  }

  isEmpty(): boolean {
    return this._entries.length === 0;
  }

  /**
   * Activate a skill by name so its full body is injected every turn. Returns
   * `false` for an unknown name (no change). Sticky + idempotent.
   */
  activate(name: string): boolean {
    if (!this._entries.some((e) => e.name === name)) {
      return false;
    }
    this._active.add(name);
    return true;
  }

  /**
   * Deactivate every skill (e.g. on a conversation reset). The manifest stays;
   * only the sticky bodies are dropped.
   */
  clearActive(): void {
    this._active.clear();
  }

  /** The currently active skill names (sorted). */
  active(): string[] {
    return [...this._active].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  }

  /**
   * The guides injected this turn (issue #115): a single manifest guide listing
   * every available skill (tier 1), followed by one guide per active skill
   * carrying its full body (tier 2). Empty when the catalog has no skills. The
   * harness appends these to {@link "../context/types.js".ContextSources.guides}
   * so they reach the model through the structural System block.
   */
  activeGuides(): Guide[] {
    if (this._entries.length === 0) return [];
    const out: Guide[] = [];

    let manifest = MANIFEST_PREAMBLE;
    for (const e of this._entries) {
      manifest += `- ${e.name}: ${e.description}\n`;
    }
    out.push({ id: GuideId.of("AVAILABLE SKILLS"), content: manifest });

    for (const name of this.active()) {
      const e = this._entries.find((entry) => entry.name === name);
      if (e) {
        out.push({ id: GuideId.of(`ACTIVE SKILL — ${e.name}`), content: e.body });
      }
    }
    return out;
  }

  /**
   * Build the `load_skill` {@link StandardTool}, sharing this catalog's active
   * set. Add it to the harness with
   * {@link "../harness/standard.js".HarnessBuilder.tool} — or use
   * {@link "../harness/standard.js".HarnessBuilder.skills}, which registers both
   * the catalog and this tool.
   */
  loadSkillTool(): StandardTool {
    return {
      implementation: new LoadSkillTool(this),
      schema: {
        name: LOAD_SKILL,
        description: LOAD_SKILL_DESCRIPTION,
        parameters: {
          type: "object",
          properties: {
            name: {
              type: "string",
              description: "The skill's name, exactly as listed in AVAILABLE SKILLS.",
            },
          },
          required: ["name"],
        },
        annotations: defaultToolAnnotations(),
      },
    };
  }
}

// ============================================================================
// load_skill tool
// ============================================================================

/**
 * `load_skill(name)` — activate a skill so its full procedure stays in context.
 * Holds the shared {@link SkillCatalog} (to mutate the active set and reject
 * unknown ids recoverably).
 */
class LoadSkillTool implements Tool {
  readonly name = LOAD_SKILL;

  constructor(private readonly catalog: SkillCatalog) {}

  async execute(
    call: ToolCall,
    _sandbox: SandboxProvider,
    _ctx: ToolContext,
    _signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const input = call.input as Record<string, unknown> | undefined;
    const raw = input?.["name"];
    const name = typeof raw === "string" && raw.trim() !== "" ? raw.trim() : undefined;
    if (name === undefined) {
      return toolOutput.error("invalid parameters: `name` (string) is required");
    }
    if (!this.catalog.activate(name)) {
      return toolOutput.error(`unknown skill '${name}'. Choose one listed in AVAILABLE SKILLS.`);
    }
    return toolOutput.success(
      `Loaded skill '${name}' — its full procedure is now in your context. Follow it.`,
    );
  }
}
