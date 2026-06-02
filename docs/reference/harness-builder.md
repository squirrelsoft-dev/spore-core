# Reference: HarnessBuilder

> The complete builder surface — every method and the default it sets. Language-agnostic; method
> names are the shared API across all four languages. For the narrative, see
> [custom-harness](../guides/custom-harness.md); for Rust signatures,
> [rust/custom-harness](../rust/custom-harness.md).

The builder constructs a harness from five **required** components plus a set of **optional** ones
that default until you set them. Every default is overridable, and each setter touches one slot
only.

## Constructors

| Constructor | What it gives you |
|-------------|-------------------|
| **`new(agent, tool_registry, sandbox, context_manager, termination_policy)`** | A builder from the five required components. Optional components start at their defaults (below). |
| **`conversational(model)`** | The preset: defaults every required component from a model — a model-backed agent, an empty tool registry, a null sandbox, a standard context manager (null cache, default compaction), and respond-and-stop termination. The few-lines path. |

`build()` finalizes the builder into a harness; `build_config()` returns the configuration object
without constructing the harness.

## Required components (set by `new`, defaulted by `conversational`)

| Slot | `conversational` default | Override |
|------|--------------------------|----------|
| Agent | model-backed agent | pass to `new` |
| Tool registry | empty registry | `tool_registry(r)` |
| Sandbox | null sandbox (denies env access) | `sandbox(s)` |
| Context manager | standard manager, null cache, default compaction | pass to `new` |
| Termination policy | complete on first final response | pass to `new` |

## Tools

| Method | Effect | Default |
|--------|--------|---------|
| `tool(t)` | Add one catalogue tool | none |
| `tools(iter)` | Add many catalogue tools | none |
| `tool_registry(r)` | Replace the harness-loop tool registry wholesale | empty registry |
| `drain_tools_into_registry()` | Move the accumulated catalogue tools into a standard registry and return it | — |

## Prompt & context

| Method | Effect | Default |
|--------|--------|---------|
| `system_prompt(s)` | Set the system prompt | none |
| `chunks(v)` | Supply prompt chunks for deterministic, cache-stable system-prompt assembly | none |
| `chunk_provider(p)` | Supply a prompt-chunk provider | none |

## Storage & sessions

| Method | Effect | Default |
|--------|--------|---------|
| `storage(s)` | Attach the storage provider (the four domain stores) | all-no-op provider |
| `auto_persist_sessions(b)` | Load prior session state at run start and save post-run state, keyed by session id | `false` |

See [storage-seam](./storage-seam.md) and [conversation](../guides/conversation.md).

## Loop-strategy components

| Method | Effect | Default |
|--------|--------|---------|
| `planner_agent(a)` | The planner for the plan-execute strategy | none |
| `evaluator_agent(a)` | The fresh evaluator for the self-verifying strategy | none |
| `verifier(v)` | A verifier component | none |
| `metric_evaluator(e)` | The objective metric for hill-climbing | none |
| `vcs_provider(p)` | Version-control provider (revert/commit for hill-climbing, Ralph) | none |
| `max_resets(n)` | Max context-window resets (Ralph continuation) | `3` |
| `max_stop_blocks(n)` | Max times a stop attempt may be blocked before halting | `8` |

See [loop-strategies](../concepts/loop-strategies.md).

## Reliability & repair

| Method | Effect | Default |
|--------|--------|---------|
| `tool_call_repair(r)` | Repair malformed/dangling tool-call JSON | none |
| `max_repair_attempts(n)` | Repair attempts before giving up | `1` |
| `compaction_verifier(v)` | Verify a compaction preserved key terms | key-term verifier |
| `max_compaction_attempts(n)` | Compaction retries | `2` |
| `hooks(h)` | Lifecycle hook chain | none |

## Cross-cutting

| Method | Effect | Default |
|--------|--------|---------|
| `middleware(m)` | The middleware chain (permissions, budgets, directory map, loop detection, …) | none |
| `observability(o)` | The observability provider | none |
| `with_observability_outbox(dir)` | Durable JSONL trace outbox at `dir`, with OTLP forwarding | none |
| `pricing(table)` | Pricing table so usage reporting carries cost | default table |
| `content_capture(cfg)` | What message/tool content is captured into traces | default config |

See [observability](../concepts/observability.md).

## Finalizers

| Method | Returns |
|--------|---------|
| `build()` | The constructed harness, ready to `run()` / `resume()` |
| `build_config()` | The harness configuration object, without constructing the harness |

## The run surface

A built harness exposes:

- **`run(options)`** → a `RunResult`. `options` carries the task, an optional starting
  `SessionState`, and an optional stream sink.
- **`resume(paused_state, response)`** → a `RunResult`, continuing a run that returned
  `WaitingForHuman`.

`RunResult` variants: **Success** (output, session id, usage, turns, post-run session state),
**Failure** (typed halt reason, plus the same accounting and post-run state), **WaitingForHuman**
(a paused state to resume), and **Escalate** (a tool signalled the caller to take over; the paused
state lets you resume the original run). The post-run `SessionState` on Success/Failure is what
makes lossless multi-turn resume possible — see [conversation](../guides/conversation.md).
