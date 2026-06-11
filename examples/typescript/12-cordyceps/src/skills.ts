/**
 * Architect-side skill injection (zero core-harness change).
 *
 * ## Why this lives in the example, not the harness
 *
 * Issue #9 added the `skill` {@link guideRegistry.GuideType} to the
 * {@link guideRegistry.GuideRegistry}, and the rich
 * {@link context.StandardContextManager} knows how to inject skills as a Block-3
 * segment. But the **live** harness loop does not call that rich `assemble` — it
 * calls {@link context.StandardCompactionAdapter}'s `assemble`, a pass-through of
 * `session.messages` (see issue #115 / Known Deviation #8). So today skills reach
 * the model only as tool-result text, never as structural injection.
 *
 * ## #131 composition: the `audit` skill rides the GLOBAL context manager
 *
 * The pre-#131 example loaded skills on demand via a worker-side `load_skill`
 * tool. The declarative `LoopStrategy` tree exposes no such per-node seam, so
 * `load_skill` was dropped. Instead the `audit` skill is seeded ALWAYS-ACTIVE at
 * startup (`runStore["active_skills"] = ["audit"]`) and injected structurally by
 * the single GLOBAL {@link SkillInjectingContextManager} the harness runs as its
 * `context_manager`. The audit procedure reaches the model every turn,
 * compaction-proof, with no `load_skill` round-trip.
 *
 * This module wires the chain end-to-end **architect-side**:
 *
 * 1. A {@link SkillCatalog} scans `.spore/skills/{name}/SKILL.md` (project) then
 *    `~/.spore/skills/{name}/SKILL.md` (user), parses YAML frontmatter
 *    `{name, description}` + markdown body, and `register`s each as a `skill`
 *    {@link guideRegistry.Guide} in a {@link guideRegistry.StandardGuideRegistry}.
 *    It also keeps a manifest side-list of `(name, description)` because `Guide`
 *    has no `description` field — the example owns the manifest text.
 * 2. {@link SkillInjectingContextManager} wraps the standard compaction adapter
 *    and, in `assemble`, prepends — **ephemerally**, never into
 *    `session.messages` — (a) the manifest of all skills, and (b) the full body
 *    of every active skill. Everything else delegates verbatim to the inner
 *    adapter.
 *
 * Net effect: the manifest + the active `audit` body are present every turn.
 * Because the active set lives in `runStore` (not the message history), it is
 * compaction-proof.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import {
  guideRegistry,
  type Context,
  type Message,
  type SessionId,
  type SessionState,
  type Task,
  type ContextManager,
  type storage,
  type Content,
  type ToolResultRecord,
} from "@spore/core";

const { StandardGuideRegistry, GuideId, Timestamp } = guideRegistry;
type StandardGuideRegistry = guideRegistry.StandardGuideRegistry;
type Guide = guideRegistry.Guide;
type RunStore = storage.RunStore;

/** The run-store key under which `load_skill` and the context manager rendezvous
 *  on the active-skill id set. */
export const ACTIVE_SKILLS_KEY = "active_skills";

/** One parsed skill: its id (== frontmatter `name`), the one-line description for
 *  the manifest, and the markdown body that is injected when active. */
export interface SkillEntry {
  name: string;
  description: string;
  body: string;
}

/** Build a `skill`-type {@link guideRegistry.Guide} from a parsed entry. `Guide`
 *  is a plain object (there is no `Guide.skill()` factory in TS), so we fill the
 *  required identity/source/status fields directly — `manual` source, `active`
 *  status, version 0. */
function skillGuide(entry: SkillEntry): Guide {
  const now = Timestamp.of(new Date().toISOString());
  return {
    id: GuideId.of(entry.name),
    name: entry.name,
    content: entry.body,
    guide_type: "skill",
    domain: null,
    source: { kind: "manual" },
    status: { kind: "active" },
    created_at: now,
    last_used: null,
    version: 0,
  };
}

/**
 * The example's skill catalog: a {@link guideRegistry.StandardGuideRegistry}
 * (the real seam) plus the manifest side-list the example owns (because `Guide`
 * carries no description). Bodies are resolved from the side-list, not re-queried
 * from the registry, so the manifest text and the injected body always agree.
 */
export class SkillCatalog {
  private constructor(
    private readonly _registry: StandardGuideRegistry,
    private readonly _manifest: SkillEntry[],
  ) {}

  /**
   * Scan the project + user skill directories and register the bundled `audit`
   * skill so the example is self-contained even with an empty `.spore/skills/`.
   * Project entries win over user entries; the bundled `audit` body is also
   * registered directly so the example never depends on a seed file existing.
   */
  static async bootstrap(
    projectRoot: string,
    bundledAudit: string,
  ): Promise<SkillCatalog> {
    const registry = new StandardGuideRegistry();
    const manifest: SkillEntry[] = [];

    // 1. Bundled audit skill — always present, registered first so a
    //    project/user override of the same name supersedes it (last-wins in the
    //    manifest; the registry treats identical content as a no-op).
    const bundled = parseSkillDoc(bundledAudit);
    if (bundled) upsert(manifest, bundled);

    // 2. Project skills: `.spore/skills/{name}/SKILL.md` relative to cwd.
    for (const entry of scanSkillDir(join(projectRoot, ".spore", "skills"))) {
      upsert(manifest, entry);
    }

    // 3. User skills: `~/.spore/skills/{name}/SKILL.md`.
    for (const entry of scanSkillDir(join(homedir(), ".spore", "skills"))) {
      upsert(manifest, entry);
    }

    // Register every manifest entry as a skill-type guide. The registry rejects
    // empty content; parseSkillDoc already guarantees a body.
    for (const entry of manifest) {
      try {
        await registry.register(skillGuide(entry));
      } catch {
        // A duplicate-content conflict (identical bundled + seeded copy) is a
        // benign no-op for our purposes — the manifest already carries the entry.
      }
    }

    return new SkillCatalog(registry, manifest);
  }

  /** The shared registry — handed to the `load_skill` tool. */
  registry(): StandardGuideRegistry {
    return this._registry;
  }

  /** The manifest side-list — handed to the context manager so it can render
   *  `name: description` lines and resolve active bodies. */
  manifest(): SkillEntry[] {
    return this._manifest.slice();
  }
}

/** Insert-or-replace by `name` so later sources override earlier ones. */
function upsert(manifest: SkillEntry[], entry: SkillEntry): void {
  const slot = manifest.findIndex((e) => e.name === entry.name);
  if (slot >= 0) manifest[slot] = entry;
  else manifest.push(entry);
}

/** Scan one `skills/` directory: each `{name}/SKILL.md` is a candidate. */
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

/**
 * Parse a `SKILL.md`: a `---`-delimited YAML frontmatter block carrying `name:`
 * and `description:`, followed by the markdown body. Minimal, dependency-free
 * parsing — the example owns this until #115's `FileSystemGuideRegistry`
 * productionizes it. Returns `undefined` if there is no usable name or the body
 * is empty.
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

/** Pull a single `key: value` scalar out of a YAML frontmatter block. Strips
 *  surrounding quotes. Good enough for the `{name, description}` contract; not a
 *  general YAML parser. */
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
    (s.startsWith('"') && s.endsWith('"')) ||
    (s.startsWith("'") && s.endsWith("'"))
  ) {
    return s.slice(1, -1);
  }
  return s;
}

/**
 * A {@link ContextManager} (the harness-loop seam) that wraps the standard
 * compaction adapter and injects the skill manifest + active skill bodies each
 * turn. ALL non-`assemble` methods delegate verbatim to the inner adapter — only
 * `assemble` is overridden, and even there the base context is produced by the
 * inner adapter first.
 *
 * The optional `ContextManager` methods (`appendAssistantMessage`,
 * `prepareCompactionTurn`, `injectMissingItems`, `applyCompaction`,
 * `tokenBudgetUsed`) are forwarded only when the inner adapter implements them,
 * so the wrapper neither adds nor masks capabilities.
 */
export class SkillInjectingContextManager implements ContextManager {
  constructor(
    private readonly inner: ContextManager,
    private readonly runStore: RunStore,
    private readonly manifestEntries: SkillEntry[],
  ) {}

  async assemble(
    session: SessionState,
    task: Task,
    signal?: AbortSignal,
  ): Promise<Context> {
    const context = await this.inner.assemble(session, task, signal);
    const active = await this.activeSkills(task.session_id);
    const injected = this.injectedMessages(active);
    context.messages = [...injected, ...context.messages];
    return context;
  }

  appendToolResult(
    session: SessionState,
    result: ToolResultRecord,
  ): Promise<void> {
    return this.inner.appendToolResult(session, result);
  }

  appendUserMessage(session: SessionState, text: string): Promise<void> {
    return this.inner.appendUserMessage(session, text);
  }

  shouldCompact(session: SessionState): boolean {
    return this.inner.shouldCompact(session);
  }

  appendAssistantMessage(
    session: SessionState,
    message: Message,
  ): Promise<void> {
    return (
      this.inner.appendAssistantMessage?.(session, message) ?? Promise.resolve()
    );
  }

  prepareCompactionTurn(
    session: SessionState,
  ): ReturnType<NonNullable<ContextManager["prepareCompactionTurn"]>> {
    return this.inner.prepareCompactionTurn?.(session);
  }

  injectMissingItems(context: Context, missing: string[]): void {
    this.inner.injectMissingItems?.(context, missing);
  }

  applyCompaction(session: SessionState, summary: string): void {
    this.inner.applyCompaction?.(session, summary);
  }

  tokenBudgetUsed(session: SessionState): number | undefined {
    return this.inner.tokenBudgetUsed?.(session);
  }

  /** Read the active-skill id set from `runStore["active_skills"]`. Absent /
   *  malformed ⇒ empty (the manifest is still injected). */
  private async activeSkills(sessionId: SessionId): Promise<string[]> {
    try {
      const value = await this.runStore.get(sessionId, ACTIVE_SKILLS_KEY);
      if (Array.isArray(value)) {
        return value.filter((v): v is string => typeof v === "string");
      }
    } catch {
      // fall through to empty
    }
    return [];
  }

  /** Render the leading injected messages: a manifest segment (always) plus one
   *  body segment per active skill (progressive disclosure). Returned as `user`
   *  messages so the loop still inserts the operating system prompt ahead of them
   *  at position 0. */
  private injectedMessages(active: string[]): Message[] {
    const out: Message[] = [];

    let manifestText =
      "AVAILABLE SKILLS (active skills' full procedures are injected below and " +
      "stay in context every turn):\n";
    for (const entry of this.manifestEntries) {
      manifestText += `- ${entry.name}: ${entry.description}\n`;
    }
    out.push(textMessage(manifestText));

    for (const id of active) {
      const entry = this.manifestEntries.find((e) => e.name === id);
      if (entry) {
        out.push(textMessage(`ACTIVE SKILL — ${entry.name}:\n\n${entry.body}`));
      }
    }
    return out;
  }
}

function textMessage(text: string): Message {
  const content: Content = { type: "text", text };
  return { role: "user", content };
}
