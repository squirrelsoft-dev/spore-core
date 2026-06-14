/**
 * Loop-replay integration tests for output-schema delivery + enforcement
 * (spore-core issue #139).
 *
 * Three recorded traces under `fixtures/model_responses/harness/`:
 *   - `output_schema_accept.jsonl` — turn-1 terminal already satisfies the
 *     schema ⇒ Success in ONE turn.
 *   - `output_schema_retry.jsonl` — turn-1 terminal is INVALID, the harness
 *     feeds the FROZEN feedback message back, turn-2 terminal is valid ⇒
 *     Success in TWO turns. The fixture's SECOND request carries the exact
 *     frozen feedback text (hash-load-bearing).
 *   - `output_schema_fail.jsonl` — every terminal is invalid; after
 *     `outputSchemaMaxRetries == 2` extra turns (3 attempts) WITH budget
 *     remaining ⇒ Failure { output_schema_violation }.
 *
 * Replay is POSITIONAL (no `request_hash`): the responses drive the flow in
 * order. The four languages must produce the same outcome + turn count — never
 * edit a fixture to make a failing implementation pass (see `fixtures/README.md`).
 * Mirrors `rust/.../tests/output_schema_fixture_replay.rs`.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyToolRegistry,
  ExecutionRegistry,
  ModelAgent,
  ReplayModelInterface,
  SessionId,
  StandardHarness,
  autonomous,
  newTask,
  type HarnessConfig,
  type LoopStrategy,
  type ProviderInfo,
  type Task,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");

function fixturePath(name: string): string {
  return resolve(repoRoot, "fixtures/model_responses/harness", name);
}

const provider: ProviderInfo = { name: "ollama", model_id: "fixture", context_window: 200_000 };

function outputSchema(): unknown {
  return {
    type: "object",
    required: ["status", "count"],
    properties: {
      status: { type: "string", enum: ["ok", "error"] },
      count: { type: "integer" },
    },
  };
}

/** Build a config with output-schema enforcement ON, the schema registered under
 *  the default empty key, and the named fixture wired as the replay model. */
function configFor(fixture: string, maxRetries: number): HarnessConfig {
  const jsonl = readFileSync(fixturePath(fixture), "utf-8");
  const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
  const agent = new ModelAgent(AgentId.of("fixture-agent"), replay);
  return {
    registry: ExecutionRegistry.builder()
      .agent("", agent)
      .toolset("", new EmptyToolRegistry())
      .schema("", outputSchema())
      .build(),
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    escalationMode: autonomous,
    enforceOutputSchemas: true,
    outputSchemaMaxRetries: maxRetries,
  };
}

/** A ReAct leaf carrying `output: ""` and a turn budget. */
function osTask(budget: number): Task {
  const strategy: LoopStrategy = {
    kind: "react",
    budget: { kind: "per_loop", value: budget },
    behavior: { kind: "fail" },
    agent: "",
    toolset: "",
    output: "",
  };
  return newTask("produce a status report", SessionId.of("output-schema-session"), strategy);
}

const FEEDBACK =
  "Your previous response did not match the required output schema. " +
  'Missing required property "status". Reply with only a JSON value that satisfies the schema.';

describe("Output-schema fixture replay (#139)", () => {
  it("accept: succeeds in one turn — output_schema_accept.jsonl", async () => {
    const r = await new StandardHarness(configFor("output_schema_accept.jsonl", 2)).run({
      task: osTask(10),
    });
    expect(r.kind).toBe("success");
    if (r.kind !== "success") return;
    expect(r.output).toBe('{"status":"ok","count":3}');
    expect(r.turns).toBe(1);
  });

  it("retry: feeds the frozen message then succeeds — output_schema_retry.jsonl", async () => {
    const r = await new StandardHarness(configFor("output_schema_retry.jsonl", 2)).run({
      task: osTask(10),
    });
    expect(r.kind).toBe("success");
    if (r.kind !== "success") return;
    expect(r.output).toBe('{"status":"ok","count":3}');
    expect(r.turns).toBe(2);
    const state = r.session_state ?? { messages: [], extras: {} };
    const fed = state.messages.some(
      (m) => m.role === "user" && m.content.type === "text" && m.content.text === FEEDBACK,
    );
    expect(fed).toBe(true);
  });

  it("fail: terminates with output_schema_violation — output_schema_fail.jsonl", async () => {
    const r = await new StandardHarness(configFor("output_schema_fail.jsonl", 2)).run({
      task: osTask(50),
    });
    expect(r.kind).toBe("failure");
    if (r.kind !== "failure") return;
    expect(r.reason.kind).toBe("output_schema_violation");
    if (r.reason.kind !== "output_schema_violation") return;
    expect(r.reason.attempts).toBe(3); // 1 + N == 1 + 2
    expect(r.turns).toBe(3); // exactly 1 + N turns; budget not exhausted
    expect(r.turns).toBeLessThan(50);
    expect(r.reason.last_error).toBe('Missing required property "status".');
    expect(r.session_id.asString()).toBe("output-schema-session");
  });
});
