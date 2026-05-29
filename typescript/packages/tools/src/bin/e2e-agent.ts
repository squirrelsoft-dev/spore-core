/**
 * End-to-end CLI agent harness + scenario suite (issue #57).
 *
 * One shared runnable entry that drives the *complete* harness loop against a
 * real local model (Ollama) through a `HarnessBuilder`-assembled harness with
 * real tools (read/write/list/bash + a deliberately-failing tool), the
 * `StandardCompactionAdapter`, and the durable-outbox observability provider.
 * The scenario is selected by a CLI arg.
 *
 * ## Scenarios
 *
 * - `s1` — multi-step / multi-tool: read input.txt → uppercase → write
 *   output.txt → read back + confirm.
 * - `s2` — multi-turn: run twice with the same `SessionId`, carrying session
 *   state across turns.
 * - `s3` — live compaction: a seeded small window + long history fires the
 *   compaction adapter mid-run; the token-accounting fix lets it compact,
 *   continue, and compact again.
 * - `s4` — tool failure + recovery: call `flaky_op` (recoverable error), then
 *   write recovered.txt explaining the adaptation.
 *
 * ## Run recipe (live, against a local model + observability stack)
 *
 * ```sh
 * # 1. Start Ollama and pull a tool-capable model.
 * ollama serve &              # or run the Ollama app
 * ollama pull llama3.2        # default model; passes the #41 capability guard
 *
 * # 2. (optional) Start the local observability stack and forward traces.
 * export SPORE_OTLP_ENDPOINT=http://localhost:4317
 *
 * # 3. Run a scenario. Prompt/model/endpoint/workspace come from args+env.
 * pnpm --filter @spore/tools e2e s1 --model llama3.2
 * pnpm --filter @spore/tools e2e s2
 * pnpm --filter @spore/tools e2e s3
 * pnpm --filter @spore/tools e2e s4
 *
 * # 4. Verify the grouped trace in Tempo (the run prints the trace_id):
 * curl -s http://localhost:3200/api/traces/<trace_id> | jq '.batches | length'
 * #    For S3, spot-check a Compaction span appears mid-trace.
 * ```
 *
 * Environment variables (all optional):
 * - `SPORE_OLLAMA_MODEL`     — default model id (overridden by `--model`).
 * - `SPORE_OLLAMA_BASE_URL`  — Ollama base url (default http://localhost:11434).
 * - `SPORE_OTLP_ENDPOINT`    — when set, forward spans to Tempo (issue #50).
 * - `SPORE_E2E_WORKSPACE`    — workspace root (default: a temp dir per run).
 *
 * ## Offline / hermetic mode
 *
 * `--mock` runs the same `buildScenario` builders against a scripted mock
 * agent, requiring no Ollama or network. The hermetic CI assertions live in
 * `packages/tools/tests/e2e-scenarios.test.ts`, which drive the same path.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  AgentId,
  type Agent,
  cacheProvider,
  context as coreContext,
  type HarnessRunOptions,
  type LoopStrategy,
  ModelAgent,
  newTask,
  observability as coreObs,
  OllamaModelInterface,
  OLLAMA_DEFAULT_BASE_URL,
  type RunResult,
  type SandboxProvider,
  SessionId,
  type SessionState,
  TaskId,
  type TerminationPolicy,
  type ToolSchema,
  type ToolRegistry as HarnessToolRegistry,
  type ContextManager as HarnessContextManager,
  emptySessionState,
  WorkspaceScopedSandbox,
  storage as coreStorage,
} from "@spore/core";

import {
  buildRealToolRegistry,
  buildRichContextManager,
  buildScenario,
  CompleteOnFinalResponse,
  parseScenarioId,
  RealToolRegistry,
  type ScenarioId,
  scenarioPrompt,
  seedCompactionState,
} from "../scenarios.js";

const { OutboxObservabilityProvider, outboxConfig } = coreObs;
const { OllamaCacheProvider } = cacheProvider;

function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

function reactStrategy(maxIterations: number): LoopStrategy {
  return { kind: "re_act", max_iterations: maxIterations };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const scenario = args[0] ? parseScenarioId(args[0]) : undefined;
  if (!scenario) {
    process.stderr.write(
      "usage: e2e-agent <s1|s2|s3|s4|s5> [--model <id>] [--mock]\n",
    );
    process.exit(2);
  }
  const mock = args.includes("--mock");
  const modelId =
    argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const sessionId = SessionId.of(`e2e-${scenario}-${stamp}`);

  const workspace =
    process.env.SPORE_E2E_WORKSPACE ??
    join(tmpdir(), `spore-e2e-${sessionId.asString()}`);
  mkdirSync(workspace, { recursive: true });
  prepareWorkspace(scenario, workspace);
  process.stdout.write(`workspace  : ${workspace}\n`);

  if (!mock && !process.env.SPORE_OTLP_ENDPOINT) {
    process.stderr.write(
      "note: SPORE_OTLP_ENDPOINT is unset — writing JSONL only (no Tempo forwarding).\n",
    );
  }

  const result = mock
    ? await runMock(scenario, sessionId)
    : await runLive(scenario, sessionId, modelId, workspace);

  if (result.kind === "success") {
    process.stdout.write(`result     : Success (${result.turns} turns)\n`);
    process.stdout.write(`output     : ${JSON.stringify(result.output)}\n`);
  } else {
    process.stderr.write(`result     : ${JSON.stringify(result)}\n`);
    process.exit(1);
  }
}

async function runLive(
  scenario: ScenarioId,
  sessionId: SessionId,
  modelId: string,
  workspace: string,
): Promise<RunResult> {
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const agent: Agent = new ModelAgent(AgentId.of("e2e-agent"), model);

  const registry = buildRealToolRegistry(scenario);
  const sandbox: SandboxProvider = new WorkspaceScopedSandbox({
    root: workspace,
    allowed_paths: [],
    denied_paths: [],
    denied_extensions: [],
    read_only: false,
    max_file_size: 0,
  });
  const bridge = new RealToolRegistry(
    registry,
    sandbox,
    sessionId,
    new coreStorage.InMemoryStorageProvider(),
    new coreStorage.InMemoryStorageProvider(),
  );
  const toolSchemas = bridge.modelSchemas();
  const tools: HarnessToolRegistry = bridge;

  const windowLimit = scenario === "s3" ? 200 : 128_000;
  const cfg: coreContext.CompactionConfig = {
    threshold: 0.8,
    preserve_recent_n: 2,
    head_tail_tokens: 64,
    offload_path: join(workspace, ".spore/offload"),
    max_compaction_attempts: 2,
  };
  const contextManager: HarnessContextManager = buildRichContextManager(
    model,
    new OllamaCacheProvider(),
    cfg,
  );

  const termination: TerminationPolicy = new CompleteOnFinalResponse();
  const obs = new OutboxObservabilityProvider(
    outboxConfig(join(workspace, ".spore"), { flushOnSessionEnd: true }),
  );
  process.stdout.write(`trace_id   : ${obs.traceIdFor(sessionId)}\n`);

  const harness = buildScenario({
    scenario,
    agent,
    tools,
    sandbox,
    contextManager,
    terminationPolicy: termination,
    toolSchemas,
    observability: obs,
  });

  return runScenario(scenario, harness, sessionId, windowLimit);
}

async function runScenario(
  scenario: ScenarioId,
  harness: { run(o: HarnessRunOptions): Promise<RunResult> },
  sessionId: SessionId,
  windowLimit: number,
): Promise<RunResult> {
  if (scenario === "s2") {
    const task1 = newTask(scenarioPrompt("s2"), sessionId, reactStrategy(8));
    const r1 = await harness.run({ task: task1 });
    if (r1.kind !== "success") {
      process.stderr.write(
        `S2 turn 1 did not succeed: ${JSON.stringify(r1)}\n`,
      );
      return r1;
    }
    const task2 = newTask(
      "Add a second TODO item to notes.md that references the first item you wrote. " +
        "Use write_file with append=true. Reply DONE when finished.",
      sessionId,
      reactStrategy(8),
    );
    return harness.run({ task: task2, session_state: emptySessionState() });
  }

  if (scenario === "s3") {
    const task = newTask(scenarioPrompt("s3"), sessionId, reactStrategy(8));
    const state: SessionState = emptySessionState();
    seedCompactionState(
      state,
      "deploy the payment service",
      sessionId,
      task.id as TaskId,
      windowLimit,
      Math.floor(windowLimit * 0.82),
      12,
    );
    return harness.run({ task, session_state: state });
  }

  const task = newTask(scenarioPrompt(scenario), sessionId, reactStrategy(8));
  return harness.run({ task });
}

function prepareWorkspace(scenario: ScenarioId, workspace: string): void {
  if (scenario === "s1" || scenario === "s5") {
    writeFileSync(
      join(workspace, "input.txt"),
      "hello from the spore harness end to end scenario\n",
    );
  }
}

// ---------------------------------------------------------------------------
// Offline / mock mode.
// ---------------------------------------------------------------------------

async function runMock(
  scenario: ScenarioId,
  sessionId: SessionId,
): Promise<RunResult> {
  const { MockAgent } = await import("@spore/core");
  const { harnessTesting } = await import("@spore/core");
  const agent = new MockAgent(AgentId.of("mock"));
  const usage = { input_tokens: 10, output_tokens: 5 };
  agent.push({
    kind: "tool_call_requested",
    calls: [{ id: "c1", name: "read_file", input: { path: "input.txt" } }],
    usage,
  });
  agent.push({ kind: "final_response", content: "DONE", usage });

  const tools = new harnessTesting.ScriptedToolRegistry();
  tools.push({ kind: "success", content: "contents", truncated: false });

  const harness = buildScenario({
    scenario,
    agent,
    tools,
    sandbox: new harnessTesting.AllowAllSandbox(),
    contextManager: new harnessTesting.NoopContextManager(),
    terminationPolicy: new harnessTesting.AlwaysContinuePolicy(),
    toolSchemas: [] as ToolSchema[],
  });
  const task = newTask(scenarioPrompt(scenario), sessionId, reactStrategy(5));
  return harness.run({ task });
}

void main();
