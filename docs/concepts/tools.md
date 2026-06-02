# Tools

> Language-agnostic — no code. For the code, see [building-a-tool](../guides/building-a-tool.md)
> and its [Rust page](../rust/building-a-tool.md).

A **tool** is one specific action the agent can take against the real world — read a file, run a
command, search the web, do arithmetic. The model decides *when* to call a tool and with *what*
arguments; the harness decides *whether it's allowed* and runs it.

## The rules every tool obeys

- **Every tool receives a sandbox.** No tool touches the filesystem or runs a process directly —
  it goes through the [sandbox](./architecture.md#the-components), which enforces the workspace
  boundary. Pure-compute tools simply ignore it.
- **Invalid input returns a typed error, never a panic.** A bad argument becomes a recoverable
  error the agent can read and retry.
- **Tools are stateless.** No references to session state; no memory of prior calls.
- **A tool never calls another tool.** Composition happens at the harness level, not inside a
  tool.
- **Large outputs are truncated** (head + tail) with the full output offloaded through the
  sandbox, so a chatty command can't blow the context window.

A tool returns one of: **Success** (content, possibly truncated), **Error** (message, plus
whether it's recoverable), or — only from a subagent tool — **WaitingForHuman**.

## The three paths

There is more than one way to give an agent a tool, and they trade convenience against control.

### 1. Use a catalogue tool

spore-core ships a catalogue of ready-made tools: file read/write/edit, grep, exec, web
fetch/search, task-list and todo tools, an ask-user tool, plan-mode controls, and more. You pick
a set and hand it to the builder. These operate *through* the sandbox, so file tools need a
real workspace-scoped sandbox (not the default null one).

Use this when a standard capability already exists — which is most of the time. It's the
shortest path and the tools are already hardened.

### 2. Implement the tool trait

Write a type that implements the individual **tool** trait: it declares its schema and provides
the execute function. Register it alongside catalogue tools. You get per-tool control — your own
schema, your own validation, your own sandbox use — while still living inside the standard
registry, sandbox enforcement, truncation, and error routing.

Use this when you need a *new capability* that should behave like every other tool (sandboxed,
truncated, schema-validated).

### 3. Implement the tool registry directly

Skip individual tools and implement the harness-loop **tool registry** itself — a `schemas()`
method that advertises tools to the model and a `dispatch()` method that runs them. This is the
most direct path: no per-tool plumbing, just "here are my schemas, here's how to run a call."

Use this when your tools are pure functions (no sandbox needed) or when you want to back several
related tools with one piece of logic — for example a REPL of trivial compute tools. The
[tool-use example](../../examples/rust/03-tool-use) takes this path.

## Choosing

| Situation | Path |
|-----------|------|
| A standard capability exists | **Catalogue tool** |
| A new capability that should behave like the rest | **Implement the tool trait** |
| Pure-compute tools, or one backend for several tools | **Implement the registry** |

## Schemas and the model

Whatever path you take, each tool exposes a **schema** — a name, a description, and a JSON-Schema
description of its inputs. The registry advertises these to the model every turn, and they are
sorted by name so the prefix stays cache-stable (see
[architecture › caching](./architecture.md#caching-is-first-class)). Write descriptions for the
*model* to read: they're the model's only guide to when and how to call the tool.

## Approval and risk

Tools carry annotations (read-only, destructive, touches external systems) from which a risk
level is derived. Middleware — a permission policy tied to the harness **mode** — can route a
risky call to a human for approval before it runs, surfacing as a pause the caller resumes. This
is how "always ask," "auto-edit," "safe-auto," and "yolo" modes differ without changing any tool.
