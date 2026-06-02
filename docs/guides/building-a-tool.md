# Building a tool

> Narrative — no code. Rust code: [rust/building-a-tool](../rust/building-a-tool.md).
> Concept background: [tools](../concepts/tools.md).

You give an agent a new capability by giving it a **tool**. There are three paths, trading
convenience for control. This guide walks the decision and the shape of each; the
[tools concept page](../concepts/tools.md) has the rules every tool obeys.

## Path 1 — reach for a catalogue tool first

Before writing anything, check whether the capability already exists. spore-core ships a
catalogue: file read/write/edit, grep, exec, web fetch/search, task-list and todo tools, an
ask-user tool, plan-mode controls. You select a set and hand it to the builder.

The one thing to remember: catalogue file and exec tools operate **through the sandbox**, so the
default null sandbox won't let them touch a real directory. Swap in a workspace-scoped sandbox
and the file tools start working against that workspace. This is the
[custom-harness](./custom-harness.md) override pattern.

## Path 2 — implement the tool trait

When you need a genuinely new capability that should behave like every other tool — sandboxed,
schema-validated, output-truncated — implement the individual **tool trait**. You provide:

- a **schema**: the tool's name, a model-facing description, and a JSON-Schema for its inputs, and
- an **execute** function that takes the validated input (and the sandbox) and returns a typed
  success or error.

You register it alongside catalogue tools. It inherits sandbox enforcement, truncation, and the
standard error routing for free.

## Path 3 — implement the registry directly

When your tools are pure functions, or you want one piece of logic to back several related
tools, implement the harness-loop **tool registry** itself. It's two methods:

- `schemas()` — return the list of tool schemas the model should see, and
- `dispatch(call)` — given a tool call (name + JSON input), run it and return a typed output.

No per-tool plumbing, no sandbox needed for pure compute. This is the most direct path and the
one the [tool-use example](../../examples/rust/03-tool-use) takes for a small REPL of compute
tools (`calculator`, `get_current_time`, `reverse_string`).

## Writing a good schema

Whatever path you take, the schema is the model's only instruction manual:

- **Name** it for what it does, in the model's vocabulary.
- **Describe** when to use it and what each argument means — write for the model, not a human
  reader skimming docs.
- **Type** the inputs with JSON-Schema, marking required fields. Be tolerant where models are
  sloppy (e.g. accept a number that arrives as a JSON string).

Schemas are sorted by name before they reach the model so the cached prefix stays byte-stable.

## Returning results

Return **success** with content, or a **recoverable error** with a message the agent can read
and act on. A recoverable error is formatted back into the conversation as a failed tool result —
so a clear message ("division by zero", "missing field 'op'") is itself a hint that steers the
next turn. Don't panic on bad input; turn it into an error.

## See it in action

[`examples/rust/03-tool-use`](../../examples/rust/03-tool-use) implements the registry directly
and prints each Think → Act → Observe step so you can watch the loop work.

## Next steps

- Control *when* a risky tool runs → approval middleware and modes, in
  [custom-harness](./custom-harness.md).
- Let a whole agent be a tool → [multi-agent](../concepts/multi-agent.md).
