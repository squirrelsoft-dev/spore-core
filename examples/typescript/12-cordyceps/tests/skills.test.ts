/**
 * Unit tests for the architect-side skill-loading machinery (`src/skills.ts`).
 *
 * The headline test — `manifest_always_injected_bodies_only_when_active` —
 * mirrors the Rust reference: it drives {@link SkillInjectingContextManager}
 * over an in-memory run store (no model, no Ollama) and asserts the progressive
 * disclosure contract:
 *
 *   - the manifest (`name: description` lines) is injected EVERY turn;
 *   - an inactive skill's BODY is NOT injected;
 *   - after `load_skill` writes the id into `runStore["active_skills"]`, that
 *     skill's body IS injected, while other skills' bodies are not.
 *
 * Plus frontmatter-parsing tests for the minimal YAML reader.
 */

import {
  SessionId,
  newTask,
  storage,
  type Context,
  type ContextManager,
  type Message,
  type SessionState,
  type Task,
  type ToolResultRecord,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import {
  ACTIVE_SKILLS_KEY,
  parseSkillDoc,
  SkillInjectingContextManager,
  type SkillEntry,
} from "../src/skills.js";

const { InMemoryStorageProvider, StorageProvider } = storage;

/** A minimal pass-through inner CM (the harness `StandardCompactionAdapter`
 *  behaves the same way for `assemble`: a pass-through of session messages). */
class PassThroughInner implements ContextManager {
  async assemble(session: SessionState, _task: Task): Promise<Context> {
    return {
      messages: session.messages.slice(),
      tools: [],
      params: { stop_sequences: [] },
    };
  }
  async appendToolResult(
    _session: SessionState,
    _result: ToolResultRecord,
  ): Promise<void> {}
  async appendUserMessage(
    _session: SessionState,
    _text: string,
  ): Promise<void> {}
  shouldCompact(_session: SessionState): boolean {
    return false;
  }
}

function manifest(): SkillEntry[] {
  return [
    {
      name: "audit",
      description: "Audit one Rust module for real, actionable defects.",
      body: "GREP-FIRST PROCEDURE BODY",
    },
    {
      name: "other",
      description: "Some other skill.",
      body: "OTHER BODY",
    },
  ];
}

function textOf(context: Context): string {
  return context.messages
    .map((m: Message) => (m.content.type === "text" ? m.content.text : ""))
    .join("\n");
}

describe("SkillInjectingContextManager", () => {
  it("manifest_always_injected_bodies_only_when_active", async () => {
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const cm = new SkillInjectingContextManager(
      new PassThroughInner(),
      provider.run(),
      manifest(),
    );

    const session: SessionState = { messages: [], extras: {} };
    const task = newTask("audit a module", new SessionId("sess-1"), {
      kind: "re_act",
      max_iterations: 8,
    });

    // No active skills yet: manifest present, NO body.
    let ctx = await cm.assemble(session, task);
    let body = textOf(ctx);
    expect(body).toContain("AVAILABLE SKILLS");
    expect(body).toContain("audit: Audit one Rust module");
    expect(body).toContain("other: Some other skill");
    expect(body).not.toContain("GREP-FIRST PROCEDURE BODY");

    // Activate `audit` (as the load_skill tool does) → body appears next turn.
    await provider
      .run()
      .put(new SessionId("sess-1"), ACTIVE_SKILLS_KEY, ["audit"]);

    ctx = await cm.assemble(session, task);
    body = textOf(ctx);
    expect(body).toContain("AVAILABLE SKILLS");
    expect(body).toContain("ACTIVE SKILL — audit");
    expect(body).toContain("GREP-FIRST PROCEDURE BODY");
    expect(body).not.toContain("OTHER BODY");
  });

  it("delegates non-assemble methods and prepends ephemerally", async () => {
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const cm = new SkillInjectingContextManager(
      new PassThroughInner(),
      provider.run(),
      manifest(),
    );
    // An existing user message must survive AFTER the injected manifest.
    const session: SessionState = {
      messages: [{ role: "user", content: { type: "text", text: "ORIGINAL" } }],
      extras: {},
    };
    const task = newTask("x", new SessionId("s"), {
      kind: "re_act",
      max_iterations: 8,
    });
    const ctx = await cm.assemble(session, task);
    // Manifest is first, original message is preserved (never mutated away).
    expect(ctx.messages[0]?.content).toMatchObject({ type: "text" });
    expect(textOf(ctx)).toContain("ORIGINAL");
    // session.messages itself was NOT mutated (ephemeral injection).
    expect(session.messages).toHaveLength(1);
    expect(cm.shouldCompact(session)).toBe(false);
  });
});

describe("parseSkillDoc", () => {
  it("parses frontmatter name and description", () => {
    const doc =
      "---\nname: audit\ndescription: Audit one module.\n---\n\n# Body\nprocedure";
    const entry = parseSkillDoc(doc);
    expect(entry).toBeDefined();
    expect(entry?.name).toBe("audit");
    expect(entry?.description).toBe("Audit one module.");
    expect(entry?.body).toContain("procedure");
  });

  it("rejects missing name or empty body", () => {
    expect(parseSkillDoc("---\ndescription: x\n---\nbody")).toBeUndefined();
    expect(parseSkillDoc("---\nname: audit\n---\n")).toBeUndefined();
  });

  it("strips surrounding quotes from scalars", () => {
    const doc = "---\nname: \"q-audit\"\ndescription: 'quoted'\n---\nbody";
    const entry = parseSkillDoc(doc);
    expect(entry?.name).toBe("q-audit");
    expect(entry?.description).toBe("quoted");
  });
});
