# Modes and approval policies

> Language-agnostic — no code. For the type-level spec see
> [`harness-engineering-concepts.md`](../harness-engineering-concepts.md) (PromptChunkRegistry,
> Human-in-the-Loop). For how a single tool surfaces approval, see [tools › approval and
> risk](./tools.md#approval-and-risk).

A **mode** is the single dial that decides how much the agent does on its own versus how much it
checks with a human first. You set it once, when you build the harness, and it stays fixed for
that harness's life. From that one choice the harness derives — deterministically — what the
system prompt tells the model, whether a given tool call needs approval, and which phase the loop
starts in. Modes and approval policies are two views of the same decision: the **mode** is the
name you pick; the **approval policy** is the enforcement it implies.

## The modes

spore-core ships four modes in the default build, plus one that only exists behind a dangerous
opt-in.

| Mode | Implied approval policy | Behavior |
|------|------------------------|----------|
| **AlwaysAsk** | `always_ask` | Describe the plan and wait for explicit approval before taking *any* action. |
| **AutoEdit** | `auto_explain` | Edit files freely; explain the changes after they're done. |
| **Plan** | `plan_only` | Produce a plan only — no file edits, no mutating tools. |
| **SafeAuto** | `safe_auto` | Auto-execute Low and Medium risk actions; High and Critical require approval. |
| **Yolo** *(dangerous)* | `none` | Full autonomy, no approval gates. Compiled out of the default build. |

The mode names and the policy names are deliberately distinct: a mode is the *intent* you select
(`AutoEdit`), and the policy is the *rule* the harness enforces (`auto_explain`). Every mode maps
to exactly one policy. The wire tags are stable snake_case (`always_ask`, `auto_edit`, …) and
byte-identical across Rust, TypeScript, Python, and Go so fixtures stay portable.

## What a mode drives

Picking a mode fixes three things at construction time. Nothing about them is decided later by
the model.

1. **The mode prompt chunk.** Each mode emits one short, *static* system-prompt chunk —
   `mode-always-ask`, `mode-safe-auto`, and so on — composed into the cached prefix of the prompt
   (Block 1). This is what tells the model, in words, how it's expected to behave. Because it's
   static it's a permanent cache hit; it never changes mid-run.

2. **The approval policy.** This is the runtime gate. At the point where the loop is about to run
   a tool call, middleware compares the call's **risk level** against the policy and decides
   whether the call may proceed or must pause for a human. This is the only place "approval"
   actually happens — the prompt chunk *describes* the contract; the policy *enforces* it.

3. **The default tool phase.** `Plan` starts the loop in a planning phase (where mutating tools
   are withheld); every other mode starts in an execution phase. This is why "plan mode" can't
   accidentally edit a file even if the model tries — the phase, not the model's goodwill, holds
   the line.

## Risk levels: how a call gets rated

The policy doesn't gate on tool *names* — it gates on **risk**, which is derived automatically
from each tool's annotations. Every tool declares whether it is read-only, whether it is
destructive, whether it is idempotent, and whether it touches external systems. From those the
harness computes one of four levels:

| Risk | Derived when… |
|------|---------------|
| **Low** | the tool is read-only |
| **Medium** | non-destructive, idempotent, and touches no external systems |
| **High** | destructive **or** touches external systems |
| **Critical** | destructive **and** touches external systems |

Because risk comes from annotations, the *same* tool behaves differently under different modes
without any change to the tool itself. A `write_file` rated High sails through under `AutoEdit`,
pauses for approval under `SafeAuto`, and is blocked outright under `Plan`. That's the whole point
of the design: one dial, no per-tool wiring.

## The approval pause, end to end

When the policy decides a call needs a human, the harness does not block a thread waiting for an
answer. It **pauses and returns control**:

1. The run ends early with a *waiting-for-human* result carrying a serializable paused state and a
   **tool-approval request** — the calls in question and their risk level.
2. The caller owns what happens next: show a prompt, enforce a timeout, ask a UI, log and
   auto-deny — whatever the application needs.
3. The caller **resumes** the paused state with a response: **allow**, **allow with
   modification** (hand back edited calls), or **deny** (with a reason the agent reads and reacts
   to).

Because the paused state is serializable, an approval can span a process restart, a web request /
response round-trip, or a human who answers an hour later. The harness is stateless across the
pause; the caller holds the state.

## Mode is permanent — and how to change it anyway

A harness does not switch modes in place. The mode is baked into the prompt prefix and the policy
at construction, and mutating it mid-run would invalidate the cache and the contract the model has
been operating under. When the agent itself decides it needs a different mode — e.g. it finishes
planning and wants to start editing — it raises a **switch-mode signal** that surfaces to the
caller as an escalation. The caller then builds a fresh harness for the new mode and continues.
The model requests; the application disposes.

## Modes vs. escalation and consult — not the same thing

It's easy to conflate the approval policy with the harness's other human-in-the-loop seams,
because they all "pause for a human." They are distinct mechanisms and they compose:

- **Approval policy** (this page) gates *tool execution* on *risk level*, keyed off the **mode**.
  The pause is a tool-approval request.
- **Escalation** pauses when the loop runs out of *budget* (a step or turn ceiling). The pause
  offers a ladder of actions — continue with more budget, skip, fail — and the escalation mode
  (e.g. surface-to-human) decides whether the operator sees it or the host auto-handles it.
- **Consult** lets a worker *ask for help* (research, advice) mid-task; the host mediates the
  request against a per-kind budget and an overflow policy (soft-fail vs. escalate-to-human).

A harness can use all three at once: SafeAuto-gated tool approval, budget escalation, and a
consult ladder are independent layers. When you read an example that "pauses for a human," check
*which* seam it's using — a budget pause or a consult ladder is **not** the mode/approval gate.

## Safe by default

The dangerous end of the dial is gated, not merely discouraged. `Mode::Yolo` and its
`none` policy are compiled out of the default build — reaching them requires an explicit dangerous
opt-in (a Cargo feature in Rust, a dedicated import elsewhere), so using full autonomy is a
build-time decision someone has to make on purpose rather than a runtime flag that can slip
through. The wire tag stays `"yolo"` for fixture parity, but the variant simply isn't there unless
you asked for it. (See issue #34.)

## Choosing a mode

| If you want… | Use |
|--------------|-----|
| To review and approve every action before it happens | **AlwaysAsk** |
| The agent to edit freely and tell you afterward | **AutoEdit** |
| A plan to review, with no edits until you say go | **Plan** |
| Autonomy for safe work, approval only for risky work | **SafeAuto** |
| No gates at all, and you've accepted the risk | **Yolo** *(dangerous build only)* |

## Where the examples stand

The mode → approval-policy → risk-gate path is a first-class part of the harness, but none of the
bundled `12-cordyceps` examples exercise it today. The Rust variant is an interactive coding REPL
that runs the full read/write toolset without an approval gate (its only pause is automatic budget
continuation). The TypeScript, Python, and Go variants are read-only audit harnesses that *do*
pause for humans — but through the **escalation** and **consult** seams described above, not the
mode/approval gate. If you want to see approval policies in action, you'll currently want to build
a harness with `SafeAuto` or `AlwaysAsk` yourself and drive the tool-approval pause/resume loop —
there isn't yet an example that does it for you.
