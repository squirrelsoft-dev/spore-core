# Architecture

> The foundational page. Every other doc links back here. Language-agnostic — no code.
> For the full normative spec, see [`harness-engineering-concepts.md`](../harness-engineering-concepts.md).

## The foundational equation

**Agent = Model + Harness**

The model does the reasoning. The **harness** is everything else: tools, sandbox, memory,
context management, sensors, guides, middleware, and the loop that drives them all. Most agent
failures are not model failures — they are configuration, context, and environment failures.
The harness is where reliability lives.

spore-core is a harness runtime. You bring a model; the library gives you a production-grade
loop around it.

## Two load-bearing ideas

**The agent is one turn.** The *agent* executes a single model call and returns a result — a
request to call tools, or a final response. It does not loop, manage context, execute tools, or
decide when to stop. The **harness** drives the loop. This separation is what makes loop
strategies, middleware, and termination policy composable: they all hang off the loop, not off
the model call.

**Stateless between runs.** The harness keeps no internal state between `run()` calls.
Everything it needs comes in via run options (the task, an optional starting `SessionState`);
everything it produces goes out via `RunResult`. The same harness object can therefore back a
CLI, a REST endpoint, a queue worker, or an embedded library with no code change. See
[deployment surfaces](#deployment-surfaces).

## The components

The harness is an *inversion-of-control container*. You inject pluggable components; the harness
wires them together and drives execution. Components don't know about each other — the harness
does. Five components are required; the rest have sensible defaults.

**Required:**

| Component | Role |
|-----------|------|
| **Model interface** | The boundary to the LLM. Normalizes provider differences; reports token usage on every call. Only the agent calls it. |
| **Agent** | Executes one turn: one model call → a tool-call request or a final response. |
| **Tool registry** | Holds the available tools, advertises their schemas to the model, dispatches calls. |
| **Sandbox** | The capability object every tool receives. The only path a tool has to the filesystem or process execution; enforces the workspace boundary. |
| **Context manager** | Assembles the context window before each turn; tracks tokens; compacts when near the limit. |
| **Termination policy** | After each turn, decides continue / halt-success / halt-failure / halt-budget. |

**Optional (defaulted):** middleware chain, observability provider, storage provider, hooks,
planner/evaluator agents, verifier, VCS provider, metric evaluator, cache provider, pricing
table. See the [harness-builder reference](../reference/harness-builder.md) for every knob and
its default.

You rarely construct all of these by hand. The builder's `conversational(model)` preset fills
in every required component so you can go from a model to a running harness in one line — see
[quickstart](../guides/quickstart.md). You then override individual defaults as your needs grow
([custom-harness](../guides/custom-harness.md)).

## The loop

Per turn, the harness:

1. Fires `before_turn` middleware.
2. Assembles the context (system prompt + guides + memory + tool schemas + history).
3. Calls the agent for **one** turn.
4. If the agent requested tools: checks always-halt violations, fires `before_tool`, validates
   each call against the sandbox, dispatches it through the registry, fires sensors and
   `after_tool`, appends results to the context, compacts if needed — then loops.
5. If the agent returned a final response: fires `before_completion` (which can force another
   turn), fires sensors, and asks the termination policy what to do.

Termination is evaluated against **external state**, not the model's say-so alone. Budget limits
are hard stops checked first, every turn. A turn that produces neither a tool call nor a final
response is an error condition, not a silent no-op.

## Identity model

Four nested levels of identity. **Session** is the primary handle you hold.

```
Project   (optional — groups sessions, owns semantic memory and guides)
  └── Session  (the persistent workspace / thread / conversation)
        └── Task  (one agentic run — instruction to completion)
              └── Turn  (one model call + the tool dispatches that flow from it)
```

A session persists across runs; a task is a single `run()`; a turn is one model call. Memory
scopes track this: episodic memory is scoped to the session ("what happened here"), semantic
memory to the project ("what we've learned across workspaces"). See
[memory](./memory.md).

## Caching is first-class

Provider prefix caching is designed in, not bolted on. The context window is assembled in three
blocks — **Static** (computed once at startup, a permanent cache hit), **Per-Session** (stable
within a session), and **Per-Turn** (never cached). Nothing in the first two blocks changes
after it is set, which is what makes the prefix byte-identical and therefore cacheable. Things
that quietly break the cache — timestamps in the static block, unsorted tool schemas,
non-deterministic guide rendering — are called out in the spec. You mostly get this for free
through the context manager and a cache provider matched to your model.

## Deployment surfaces

Because the harness is stateless between runs, the *same* `run()` / `resume()` interface backs:

- **CLI** — construct via builder, stream tokens to stdout, write paused state to disk on pause.
- **REST API** — async endpoint, SSE/WebSocket for streaming, a database or Redis for paused state.
- **Library** — embedded in a desktop app, editor extension, CI step, or serverless function.
- **Queue worker** — poll a job, construct the harness, publish the `RunResult` back.

The **host** owns session persistence, paused-state storage, stream delivery, queueing, auth,
and rate limiting. The **harness** owns loop execution, component orchestration, and producing
the `RunResult`.

## Where to go next

- The loop has several shapes — [loop-strategies](./loop-strategies.md).
- Give the agent capabilities — [tools](./tools.md) and the [building-a-tool guide](../guides/building-a-tool.md).
- Persist across turns and sessions — [memory](./memory.md), [conversation guide](../guides/conversation.md).
- See what the agent did — [observability](./observability.md).
- Compose agents — [multi-agent](./multi-agent.md).
