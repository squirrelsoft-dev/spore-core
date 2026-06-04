# 11 — multi-agent composition: agents as tools

> **Agents are composable. The harness doesn't care whether a "tool" dispatches
> to a function or to another agent.** An orchestrator delegates subtasks to
> isolated worker agents that it sees as ordinary tools — clean encapsulation,
> independent harness instances, no shared mutable state.

This example builds **three agents** and wires two of them into the third as
tools. The orchestrator receives a task — _"research a topic and produce a
polished report"_ — plans the work, delegates the research, delegates the
writing, and assembles the final `workspace/report.md`.

**Three agents, two handoffs, one output.**

| Agent                 | Tools                                              | Loop        | Returns                    |
| --------------------- | -------------------------------------------------- | ----------- | -------------------------- |
| **research worker**   | `web_search` (SearXNG, as in 06)                   | ReAct       | cited findings (text)      |
| **writing worker**    | _none_                                             | ReAct       | a polished markdown report |
| **orchestrator**      | `research_worker`, `writing_worker`, `write_file`  | PlanExecute | writes `report.md`         |

## The architectural insight: "agent as tool"

The reason this example exists is one idea: **a worker agent is just a tool from
the orchestrator's perspective.**

Each worker is a fully-built child `Harness` — its own model instance, its own
system prompt, its own tools, its own loop. We wrap that child in a
[`SubagentTool`], which implements the same `Tool` interface as `write_file` or
`web_search`. When the orchestrator emits a tool call for `research_worker`, the
`SubagentTool`:

1. reads a single `instruction` string from the call,
2. runs the child harness on a fresh `Task` with that instruction,
3. returns the child's final output string as the tool result.

The orchestrator cannot tell — and does not need to know — that the "tool" behind
`research_worker` is an entire agent with its own reasoning loop. That is the
composability thesis made concrete: **the harness treats a function-tool and an
agent-tool identically.**

In code, registering a worker looks exactly like registering `web_search` in
example 06 — build a `StandardTool` (`{ implementation, schema }`) from the
`SubagentTool` plus a schema, then `.tool(...)` it onto the orchestrator's
builder. The only difference is that the "implementation" boxed inside is a whole
agent:

```ts
const subagent = SubagentTool.buildOrThrow({
  name: "research_worker",
  description: "Delegate to the research agent …",
  inputSchema: instructionSchema(), // { instruction: string }
  timeoutMs: 180_000,
  contextSharing: { kind: "isolated" },
  harness: researchChild, // ← an entire agent
  childRegistry: new toolRegistry.StandardToolRegistry(), // depth-1 check only
});

orchestrator.tool({ implementation: subagent, schema: { name, description, … } });
```

## Independent agents, isolated context

Both workers are wrapped with `contextSharing: { kind: "isolated" }`:

- Each worker runs in a **brand-new session** with **no shared mutable state**
  with the orchestrator or with the other worker.
- Each agent has its **own model instance** (constructed fresh from the same
  model id — they do not share one object) and its **own context window**.
- The **only** thing that crosses an agent boundary is a single string: the
  worker's final output becomes the orchestrator's tool result, and the
  orchestrator passes the research findings into the writing worker as that
  worker's `instruction`.

This is _why_ you delegate to a subagent instead of inlining the work. The
research worker may burn a dozen internal turns issuing search queries and
sifting noisy JSON — but none of that noise enters the orchestrator's context.
The orchestrator stays small and on-topic; the messy work is encapsulated behind
a clean string hand-off.

### A visible consequence: the child's turns don't stream up

Because the workers are isolated, **the child's internal turns do not stream up
through the parent.** The orchestrator's stream only shows the `tool_call` to a
worker and the `tool_result` coming back. You will _not_ see the research
worker's individual `web_search` calls in this example's output — that is not a
missing feature, it _is_ the context isolation, made observable. (If you want to
watch a worker's internal ReAct loop, run example 06 directly: that is the same
research loop, un-encapsulated.)

## The strategy split: PlanExecute at the top, ReAct inside

The two layers use two different loop strategies, each fit to its level:

- The **orchestrator** runs `{ kind: "plan_execute" }`. It decomposes the job
  ("research → write → save") up front and executes the steps in order — natural
  for a coordinator (see example 08 for PlanExecute on its own).
- Each **worker** runs **ReAct** internally. This is hardcoded inside
  `SubagentTool`: a subagent always runs its child as `{ kind: "re_act" }`.
  Deliberate planning at the top; step-by-step tool use at the bottom.

## Reading the output: agent boundaries

The point of this example is _legibility_ — you should be able to read stdout and
see which agent is acting, what it received, and what it returned. Each worker
dispatch prints a boxed banner built from the orchestrator's `tool_call` /
`tool_result` stream events. (The `tool_result` event carries only a `call_id`,
so we track a `call_id → name` map to label the closing half of each banner.)

```text
┌─ orchestrator → research_worker
│  received: Research the history and core ideas of the Rust programming language…
└─ research_worker → orchestrator
   returned: Rust first appeared in 2010 as a Mozilla project… (https://…)
┌─ orchestrator → writing_worker
│  received: Rust first appeared in 2010… (the research findings, verbatim)
└─ writing_worker → orchestrator
   returned: # The Rust Programming Language\n\n## History …
  orchestrator → write_file({"path":"report.md", …})
```

`research_worker` / `writing_worker` lines are the two agent handoffs;
`write_file` is the orchestrator saving the result itself.

## Run it

```sh
ollama serve &
ollama pull llama3.2
pnpm install
export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"   # SearXNG JSON API

pnpm start                                       # default model + topic
pnpm start -- --topic "the history of the TCP/IP protocol suite"
pnpm start -- --model qwen2.5-coder:7b
```

### Configuration (`.env.example`)

| Variable / flag                  | Meaning                                                                                                                       |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `SPORE_WEB_SEARCH_ENDPOINT`      | **Required.** SearXNG JSON endpoint for the research worker's `web_search`. The example exits with code 2 if unset (as in 06). |
| `SPORE_OLLAMA_MODEL` / `--model` | Model id for **all three** agents (each gets its own instance). Default `llama3.2`.                                            |
| `SPORE_OLLAMA_BASE_URL`          | Ollama endpoint. Default `http://localhost:11434`.                                                                            |
| `--topic`                        | The (generic, timeless) subject to research. Has a stable default.                                                            |

### Expected output

- The configuration banner, then `orchestrator · plan/execute turn N` lines.
- A boxed `┌─ … └─` agent boundary for each worker hand-off (research, then
  writing), plus a `write_file` line.
- On success, `report.md now exists on disk: …/workspace/report.md`.

> This example needs a running model (Ollama) **and** a SearXNG backend. It is
> **not** part of the lint/typecheck gate, which is all CI checks.

## Where to look

- `src/main.ts` — the whole example. Start at `main`; the agent-as-tool wiring is
  in `buildResearchHarness`, `buildWritingHarness`, and `buildWorkerTool`.
- `typescript/packages/tools/src/subagent.ts` — `SubagentTool`: how a child
  `Harness` is exposed as a `Tool`, the `instruction` input, and the depth-1
  (no nested subagents) rule.
- `examples/typescript/06-web-research/` — the research worker's `web_search`
  loop, run standalone.
- `examples/typescript/08-plan-execute/` — `{ kind: "plan_execute" }` on its own.

[`SubagentTool`]: ../../../typescript/packages/tools/src/subagent.ts
