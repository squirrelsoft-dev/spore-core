# Multi-agent

> Language-agnostic — no code. Builds on [architecture](./architecture.md).

Because the harness is **stateless between runs** and an entire harness can be wrapped as a
**tool**, multiple agents compose without any special "multi-agent framework." There are three
patterns in v1.

## Sequential handoff

The simplest pattern: two calls to `run()` with the **same session id**. The second run inherits
the first run's conversation state and shares the same filesystem, so it picks up where the first
left off. This covers any handoff — an initializer agent that sets up a workspace followed by a
coding agent, a research pass followed by a writing pass, and so on.

You thread the post-run `SessionState` (or a fresh agent configuration over the same session)
from one run into the next. No coordination machinery required.

## Agent as a tool (subagent)

A child harness can be wrapped as a **tool** and registered with a parent agent's tool registry.
The parent calls it like any other tool; the child runs its own loop to completion (or to a
human-in-the-loop pause) and returns a result the parent observes. This is how an agent
delegates a self-contained subtask to a specialist configuration — a different model, a
different tool set, a read-only sandbox.

The child can share context with the parent in one of three ways:

- **Isolated** — a fresh session; the child starts clean.
- **Shared session** — the same session id; the child sees the parent's episodic memory.
- **Summary handoff** — a fresh session seeded with a summary the parent provides.

**One level deep.** A subagent cannot spawn its own subagents. This isn't a convention you have
to remember — it's enforced when the subagent tool is constructed, and encoded in the types
(the child's paused state has no slot for a grandchild). It keeps delegation trees shallow and
debuggable, and keeps human-in-the-loop pauses tractable: when a subagent pauses for approval,
the parent assembles a combined paused state with the child's nested inside, and surfaces one
pause to the caller.

## Filesystem coordination

Several harness instances can share a workspace root and coordinate through the filesystem, with
git handling collision resolution. No concurrent-harness abstraction is needed in v1 — the
filesystem *is* the coordination layer. This is the same idea that powers the
[Ralph loop strategy](./loop-strategies.md#ralph), scaled across agents instead of across context
windows.

## What's deferred

A first-class **parallel harness** — a task queue feeding N concurrent harness instances with a
result aggregator — is post-v1. The single-agent harness is stabilized first; the composition
patterns above cover the multi-agent needs that matter before then.

## Choosing

| You want… | Pattern |
|-----------|---------|
| One agent to pick up where another left off | **Sequential handoff** (shared session id) |
| An agent to delegate a self-contained subtask | **Agent as a tool** (subagent) |
| Several agents working one workspace | **Filesystem coordination** |
