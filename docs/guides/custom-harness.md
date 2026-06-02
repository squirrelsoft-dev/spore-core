# Custom harness

> Narrative — no code. Rust code: [rust/custom-harness](../rust/custom-harness.md). Full method
> list: [harness-builder reference](../reference/harness-builder.md).

The `conversational(model)` preset gets you running fast by choosing every component for you. A
**custom harness** is the same builder with some of those choices replaced. You override only
what you need; everything else keeps its default. This guide walks the progression from preset to
fully assembled.

## Two starting points

- **From the preset.** Start with `conversational(model)` and override individual components —
  add a tool registry, swap the sandbox, attach storage. This is the common path and how the
  examples grow from one to the next.
- **From scratch.** Construct the builder from the five required components directly — your own
  agent, tool registry, sandbox, context manager, and termination policy — when none of the
  presets' choices fit. Optional components still default until you set them.

## The overrides you'll reach for

**Tools.** Replace the empty registry with your own — either catalogue tools or a custom
registry. See [building-a-tool](./building-a-tool.md).

**Sandbox.** The preset uses a null sandbox that denies environment access. To let file and exec
tools touch a real directory, supply a workspace-scoped sandbox rooted at your workspace. Stricter
isolation modes (process- or container-level) exist for untrusted work.

**Context manager.** The preset's standard context manager handles assembly, token tracking, and
compaction with sensible defaults. Override it to change the compaction threshold, the cache
provider (match it to your model's caching — Anthropic, OpenAI, or none), or the system prompt
chunks.

**Termination policy.** The preset stops at the model's first final response. Real tasks want a
completion check — keep going until tests pass, a file exists, a feature list is satisfied — or a
different loop strategy entirely (plan-execute, self-verifying, hill-climbing). See
[loop-strategies](../concepts/loop-strategies.md).

**Storage.** Attach a storage provider and turn on auto-persist for conversations that survive
restarts. See [conversation](./conversation.md) and [storage-seam](../reference/storage-seam.md).

**Middleware.** Cross-cutting concerns hook into the loop: directory-map injection, token and
time budgets, loop detection, pre-completion checklists, and — importantly — a **permission**
policy that surfaces risky tool calls to a human. The harness **mode** (always-ask, auto-edit,
plan, safe-auto, yolo) selects a permission posture and a tool phase in one setting.

**Observability.** Attach a provider — or point the durable outbox at a directory — to get a
JSONL trace per session and OTLP forwarding to your telemetry stack. See
[observability](../concepts/observability.md).

**Budgets and pricing.** Set token/time budgets (hard stops the termination policy checks first)
and a pricing table so usage reporting carries cost.

## The principle

Every default is overridable, and overriding one doesn't disturb the others — the builder is
just a set of independent slots over the five required components. Start from the preset, change
the one thing your task needs, and grow from there. The
[harness-builder reference](../reference/harness-builder.md) lists every method and the exact
default it replaces.

## See it in action

The example progression *is* the custom-harness story: `01-hello-agent` is the bare preset,
`03-tool-use` overrides just the tool registry, and later examples layer on sandboxes, memory,
and loop strategies. Browse [`examples/rust/`](../../examples/rust).
