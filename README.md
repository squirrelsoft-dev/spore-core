# spore-core

> **An agentic harness runtime built from first principles.**

Spore is a language-agnostic harness for AI agents — the runtime container that takes a model and turns it into something reliable. It handles the agent loop, tool execution, sandbox isolation, context management, memory, sensors, guides, middleware, and the improvement flywheel that makes the harness get smarter over time.

The model does the reasoning. Spore handles everything else.

---

## The Problem

Most agent failures are not model failures. They are configuration, context, and environment failures. A model given the wrong tools, a bloated context window, no verification loop, and no cross-session memory will fail on tasks it could otherwise handle. The harness is where reliability lives.

```
Agent = Model + Harness
```

Spore is an implementation of that equation. It defines clear component boundaries, injects them via IoC, and drives them through a well-specified loop. Swap any component — swap the model, the memory backend, the sandbox, the loop strategy — without touching the rest.

---

## Not Just for Coding Agents

The harness engineering literature — and much of this project's documentation — uses coding agents as the primary example. That is because coding agents are where the discipline emerged and where the benchmarks are. It is not a constraint.

The harness primitives are domain-agnostic. What changes between a coding agent and a conversational agent is not the harness structure — it is the components you inject and the guides you load.

| Agent Type | Session | SandboxProvider | Tools | Sensors | Termination |
|---|---|---|---|---|---|
| **Coding agent** | Git workspace | WorkspaceScoped filesystem | bash, file read/write, git | Test runner, linter, type checker | Feature list complete, tests pass |
| **Conversational / RAG** | Chat thread | Read-only document scope | Document search, fetch | Citation grounding, answer completeness | Question answered with citations |
| **NL-to-SQL** | DB connection + user context | Database-scoped, read-only by default | Schema introspection, SQL execution | SQL safety (no unguarded DELETE/DROP), result sanity | Valid result set returned |
| **Research agent** | Research workspace | Workspace + web access | Web search, fetch, summarize, file write | Source credibility, claim grounding | Research brief complete |
| **Data analysis** | Notebook workspace | WorkspaceScoped | Code execution, data read, chart generation | Result sanity, statistical validity | Analysis complete with outputs |

The key insight: a RAG assistant's document scope is the same concept as a coding agent's filesystem scope — both are `SandboxProvider` implementations that enforce a capability boundary. A SQL safety sensor checking for unguarded `DELETE` statements is the same concept as a linter checking for code style violations — both are `SensorChain` implementations that provide feedback after tool execution. The conversation thread in a chat assistant is the same concept as the git workspace in a coding agent — both are the `SessionId`-scoped persistent container the user returns to.

The examples in this project use coding tasks because they are concrete, verifiable, and benchmark-able. The architecture applies everywhere agents need to act reliably.

---

## Design Principles

**The agent is one turn.** The agent executes one model call and returns a result (tool call requests or a final response). The harness drives the loop. This separation makes loop strategies, middleware, and termination policy fully composable.

**Inversion of control.** The harness is a runtime container. Components are injected at construction — model, sandbox, tools, memory, sensors, guides, middleware, termination policy, observability. Nothing is hardcoded.

**Stateless between runs.** `run()` takes options, returns a result. `resume()` takes a saved state and a human response, returns a result. No internal state between calls. Deploy as a CLI, REST API, library, queue worker, or subprocess without modification.

**Cache-aware by design.** The context window is assembled in three blocks — Static (permanent cache hit), PerSession (cached within a session), PerTurn (never cached). Provider prefix caching is a first-class concern, not an afterthought.

---

## Quick Start

> **Note**: The interfaces below represent the target API. Implementations are maintained at parity across all four languages — Rust, TypeScript, Python, and Go.

```rust
// Rust
let harness = HarnessBuilder::coding_agent(&workspace, model)
    .observability(OtelObservabilityProvider::new(endpoint))
    .build()?;

let result = harness.run(HarnessRunOptions {
    task: Task::simple("Fix the failing tests in src/auth.rs"),
    on_stream: Some(Box::new(|event| print_token(event))),
}).await;
```

```typescript
// TypeScript
const harness = HarnessBuilder.codingAgent(workspace, model)
    .observability(new OtelObservabilityProvider(endpoint))
    .build();

const result = await harness.run({
    task: Task.simple("Fix the failing tests in src/auth.rs"),
    onStream: (event) => printToken(event),
});
```

```python
# Python
harness = (
    HarnessBuilder.coding_agent(workspace, model)
    .observability(OtelObservabilityProvider(endpoint))
    .build()
)

result = await harness.run(HarnessRunOptions(
    task=Task.simple("Fix the failing tests in src/auth.rs"),
    on_stream=lambda event: print_token(event),
))
```

```go
// Go
harness, err := sporecore.CodingAgent(workspace, model).
    Observability(observability.NewOtelProvider(endpoint)).
    Build()

result := harness.Run(ctx, sporecore.RunOptions{
    Task:     sporecore.SimpleTask("Fix the failing tests in src/auth.rs"),
    OnStream: func(event sporecore.StreamEvent) { printToken(event) },
})
```

---

## Components

The harness wires together fifteen components. Every component is a trait/interface — bring your own implementation or use the reference implementations included in the library.

| Component | Purpose |
|---|---|
| **ModelInterface** | Boundary to the LLM. Normalizes providers, handles streaming, reports token usage. |
| **Agent** | Executes one turn — one model call, returns tool call requests or a final response. |
| **Harness** | Drives the loop. Wires everything together. Owns termination. |
| **ToolRegistry** | Registers tools, manages active ToolSets per task phase, dispatches calls. |
| **Tool** | Executes one action (file read, bash, SQL, HTTP). Stateless. Always receives a SandboxProvider. |
| **SandboxProvider** | Capability object enforcing the execution boundary. Path validation, command isolation. |
| **ContextManager** | Assembles the context window. Three-block cache structure. Compaction. Skill injection. |
| **PromptChunkRegistry** | Named, cacheable prompt chunks. Composes Block 1 once at startup — permanent cache hit. |
| **CacheProvider** | Provider-specific cache annotation (Anthropic, OpenAI, Ollama). Injected into ContextManager. |
| **MemoryProvider** | Episodic memory (session-scoped) and semantic memory (project-scoped). Versioned. |
| **GuideRegistry** | Feedforward artifacts — guides, skills, conventions. Lifecycle management and improvement flywheel. |
| **SensorChain** | Feedback controls — linters, test runners, LLM-as-judge. Post-tool and post-turn triggers. |
| **MiddlewareChain** | Hook-based interceptors at six points in the loop. Loop detection, HITL, PII redaction, cost control. |
| **TerminationPolicy** | Evaluates after every turn. Budget limits are hard stops. Model's self-assessment is one input, not the decision. |
| **ObservabilityProvider** | Structured spans for every harness operation. OTLP-compatible — works with Langfuse, Grafana, Datadog, Honeycomb out of the box. |

---

## Loop Strategies

The harness supports five loop strategies. The agent is the same in all cases — one turn. The strategy determines the outer structure.

| Strategy | Use Case |
|---|---|
| **ReAct** | Standard tool-calling loop. Thought/Action/Observation interleaved. |
| **PlanExecute** | Plan once (optionally with a different model), execute steps in a loop. |
| **Ralph** | Multi-context-window continuation. Intercepts exit, resets context, resumes from filesystem state. |
| **SelfVerifying** | Build loop + separate evaluator agent (read-only, fresh context, Default-FAIL contract). |
| **HillClimbing** | Iterative optimization. Establish baseline metric, propose changes, keep if improved, revert if not. Generalizes the [autoresearch](https://github.com/karpathy/autoresearch) pattern. |

---

## The Mode System

Mode is a first-class concept that drives three things at construction time — prompt chunk, approval policy, and active tool phase.

| Mode | Behavior | Approval Policy |
|---|---|---|
| `AlwaysAsk` | Describe plan, wait before any action | Require human for everything |
| `AutoEdit` | Edit freely, explain after | Auto-approve up to Medium risk |
| `Plan` | Plan only, no file edits during planning | Auto-approve reads only |
| `SafeAuto` | Autonomous with gates on destructive actions | Require human for High + Critical |
| `Yolo` | Full autonomy | Auto-approve everything |

---

## Human-in-the-Loop

The harness pauses asynchronously and returns a `WaitingForHuman` result. The caller owns `PausedState`. No blocking, no timeouts inside the harness.

```rust
match harness.run(options).await {
    RunResult::WaitingForHuman { state, request } => {
        // persist state however you want — database, Redis, filesystem
        db.save_paused_state(&state).await?;
        // surface request to the human via your UI
        ui.show_approval_request(&request);
    }
    // ...
}

// When the human responds:
let result = harness.resume(state, HumanResponse::Allow, None).await;
```

Three interaction types: **ToolApproval** (approve/deny/modify a tool call), **Clarification** (agent needs information), **Review** (agent wants sign-off before continuing).

---

## The Improvement Flywheel

Spore is designed to get better over time without changing the model.

```
Run → Trace → Analyze failure patterns → Propose harness changes
  ↑                                                          ↓
  └────── Human approves ← Statistical comparison ← Test candidates
```

1. Every session emits a structured trace via `ObservabilityProvider`
2. `GuideRegistry.analyze_performance()` identifies failure patterns across traces
3. A meta-agent (or human) proposes candidate changes to the harness configuration
4. The eval harness runs candidates against a task suite and produces a `ComparisonReport`
5. Human reviews and approves winners — they are promoted to Active and become the new baseline
6. Repeat

Automated proposals always start in `PendingReview`. Nothing is promoted without a review gate.

### What gets improved

This is not just prompt tuning. The flywheel targets the full harness configuration:

| What changes | Example |
|---|---|
| **Prompt chunk content** | Role description becomes more precise, mode instructions are tightened |
| **Guide content** | Schema annotations updated, domain conventions refined |
| **Middleware thresholds** | Loop detection fires after 3 file edits instead of 5 |
| **Sensor parameters** | Citation grounding threshold raised from 0.7 to 0.85 |
| **Tool schemas** | Parameter description clarified, reducing model misuse |
| **Active ToolSet per phase** | Browser tools only available during verification, not planning |
| **CompletionCheck logic** | Done condition tightened to require all tests passing, not just build success |
| **Approval policy** | SQL DELETE elevated from Medium to High risk after observed incidents |

Prompt chunks are the most visible artifact because they are human-readable text you can diff. But a middleware threshold going from 5 to 3, or a tool being removed from the planning phase, is equally a harness improvement — it changes what the agent can do and when, not just what it is told. The model never changes. The environment it operates in does.

---

## Observability Stack

A local, self-hosted observability stack (Grafana + Tempo + Loki + Alloy + Prometheus) ships with the repo under `observability/`. It receives OTLP traces from the harness, indexes the per-session trace JSONL, and provides Grafana dashboards for session outcomes, cost, cache hit rate, and sensor fire rate. All Grafana OSS — no vendor accounts, no API keys.

```bash
# Start the stack (Docker required)
docker compose -f observability/docker-compose.observability.yml up -d

# Point the harness at it
export SPORE_OTLP_ENDPOINT=http://localhost:4317

# Open Grafana (no login — anonymous admin)
open http://localhost:3000      # dashboards live under the "Spore" folder

# Stop the stack
docker compose -f observability/docker-compose.observability.yml down
```

When `SPORE_OTLP_ENDPOINT` is set, the observability provider forwards spans to Tempo over OTLP gRPC on port 4317. When unset (CI, unit tests), it falls back to the local trace JSONL only — the harness never depends on the stack running. Alloy tails `.spore/sessions/**/*.jsonl` and ships every span to Loki automatically; click any trace in Tempo to jump straight to its raw log lines.

The on-disk trace format is the source of truth and is documented in [`observability/TRACE_SCHEMA.md`](observability/TRACE_SCHEMA.md). Copy [`.env.observability.example`](.env.observability.example) to `.env` for the environment template.

> **Status:** the stack and dashboards are ready. The emitting side — the durable-outbox provider that writes the trace JSONL and forwards to OTLP (issue #33) — is in progress; until it lands, the stack starts cleanly but dashboards stay empty.

---

## Identity Model

```
Project   (optional — groups sessions, owns semantic memory)
  └── Session  (the workspace or conversation — primary caller handle)
        └── Task  (one agentic run — one call to harness.run())
              └── Turn  (one model call + all tool dispatches)
                    └── ToolDispatch  (one tool — (SessionId, TaskId, TurnNumber, DispatchIndex))
```

`SessionId` is the ThreadId equivalent — the thing the caller holds onto and comes back to. `TaskId` is internal to the harness run. The agent's internal todo list is not a harness concept — it is a planning artifact managed by the agent within a single Task.

---

## Multi-Agent

**Sequential** (v1): two calls to `harness.run()` with the same `SessionId`. Progress files and git history bridge them.

**SubagentTool** (v1): wrap a child `Harness` as a `Tool`. Parent agent calls it via `ToolRegistry`. Child runs to completion and returns a result string. Subagents cannot spawn their own subagents — enforced at construction time and in the type system.

**Parallel fan-out** (post-v1): `ParallelHarness` with a task queue and N concurrent instances. Filesystem + git for coordination.

---

## Deployment

The same harness interface deploys anywhere:

```
CLI           → thin wrapper around harness.run(), streams to stdout
REST API      → async task endpoint, SSE for streaming, DB for PausedState
Library       → embed in any application
Queue worker  → poll queue, run harness, publish RunResult
Subprocess    → TypeScript REST API shells out to Rust binary (recommended v1 polyglot setup)
```

---

## Project Status

All component interfaces, rules, identity models, and architectural decisions are fully specified, and the harness is implemented at **parity across all four languages**. The fifteen core components and the Mode/cache systems are landed; the remaining work is the non-ReAct loop strategies (#58–#61).

| Area | Status |
|---|---|
| Language-agnostic spec | ✅ Complete |
| Component interfaces | ✅ Complete (issues #1–#13) |
| Design decisions | ✅ Resolved (issues #14–#22) |
| PromptChunkRegistry + CacheProvider | ✅ Complete (#24, #25) |
| Eval harness design | 📋 Discussion (#26) |
| Rust implementation | ✅ Core complete |
| TypeScript implementation | ✅ Core complete |
| Python implementation | ✅ Core complete |
| Go implementation | ✅ Core complete |
| Loop strategies (ReAct landed; Ralph, PlanExecute, SelfVerifying, HillClimbing) | 🔜 In progress (#58–#61) |

---

## Documentation

- **[docs/harness-engineering-concepts.md](./docs/harness-engineering-concepts.md)** — the canonical language-agnostic specification. Component responsibilities, rules, type definitions, loop strategies, error propagation, cache architecture, identity model, and the improvement flywheel. Start here.
- **[GitHub Issues](https://github.com/squirrelsoft-dev/spore-core/issues)** — each component has a dedicated issue with full trait definitions and implementor notes. Discussion issues (#14–#26) capture design decisions with rationale.

---

## Background

Spore is informed by published work from Anthropic, LangChain, OpenAI, and the broader harness engineering community — particularly the concepts in:

- Böckeler, [Harness Engineering for Coding Agent Users](https://martinfowler.com/articles/harness-engineering.html) (Martin Fowler, April 2026)
- LangChain, [The Anatomy of an Agent Harness](https://blog.langchain.com/the-anatomy-of-an-agent-harness/)
- Anthropic, [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Karpathy, [autoresearch](https://github.com/karpathy/autoresearch)

The name comes from mycelium — the persistent underground network that connects, routes, and coordinates without a central brain. The harness is the mycelium. The agents are the fruiting bodies.

---

## License

[MIT](./LICENSE)
