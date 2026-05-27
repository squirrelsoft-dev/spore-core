# PROJECT STATE
_Last updated: 2026-05-27 by /close_

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.

**#57 (a48fe84, PR #62) just landed — the harness now runs end-to-end.** A shared
e2e CLI in each language drives the *complete* ReAct loop against a real model
(Ollama) through a `HarnessBuilder`-assembled harness with real tools
(read/write/list/bash + a deliberately-failing `flaky_op`), the
`StandardCompactionAdapter`, and outbox observability. It's proven by a
**4-scenario hermetic suite** — S1 multi-step/multi-tool, S2 multi-turn (shared
session), S3 live compaction, S4 tool-failure + recovery — and was run live
against `llama3.2` (`Success`, real trace written). Rust: `cargo run -p spore-core
--example e2e_agent -- s1`; TS/Python/Go have parity CLIs. CI stays hermetic (mock
agent, no network); live runs are a documented manual recipe. The CLI prints
`session_id` / `trace_path` / `trace_id` per run.

Running it live surfaced and fixed bugs the hermetic suite couldn't catch (the
mock agent ignores the assembled context, so it never noticed the prompt wasn't
delivered):
- **Prompt delivery** — the harness never turned `task.instruction` into a user
  message, so the model received an empty conversation (`EmptyResponse`). Fixed
  in all four languages; seeded only on a fresh `run()`, not on resume.
- **Token accounting (former Known Deviation #2, now resolved)** —
  `tokens_reclaimed` is computed from dropped-message tokens, `token_budget_used`
  decrements live, and the Compaction span stamps real `tokens_after`/
  `tokens_reclaimed`. A session can compact → continue → compact again. All four.
- **OTLP attributes** — the forwarder now flattens per-span attributes (tokens,
  `stop_reason`, `tool_name`, compaction sizes, outcome) onto exported spans.
  Previously only 5 fixed envelope tags reached Tempo. All four.

Earlier work still in place: observability stack (#42), ObservabilityProvider +
OTLP→Tempo / JSONL→Loki parity (#49/#50/#33), `OllamaModelInterface` +
capability guard (#41), `StandardCompactionAdapter` wiring (#55), the
compaction verify→retry→warn loop (#46), and `StandardContextManager` /
verifiers (#29).

**main CI is green** across all four languages.

Known runnability limits: the harness is **ReAct-only** — the other four loop
strategies return `StrategyNotYetImplemented` (#58–#61). The observability stack
is **metrics-only**: it captures span structure + token/timing/outcome but **no
message content**, so it cannot show the actual conversation or tool-call
payloads — the standard agent-debugging view (tracked in #64). And reading a
not-yet-created in-workspace file is misclassified as a sandbox `PathEscape`,
making S1 model-nondeterministic (#63).

## Active Direction
With the harness now **runnable** end-to-end, the next bar is making a run
**debuggable** and **robust**. The headline gap is observability that shows the
actual conversation and tool usage (prompts, completions, tool args/results) —
the thing every LLM-observability tool provides and ours currently can't.
Decided approach: capture content via OpenTelemetry **GenAI semantic
conventions** (opt-in, truncated, off by default) and view it in an
**LLM-native, OTel-native, open-source backend (Arize Phoenix)**, keeping the
Tempo/Loki/Prometheus stack for system telemetry and keeping the backend
swappable over OTLP (#64). Alongside that, harden the pieces the live run
exposed (sandbox read semantics, #63). Loop-strategy breadth (#58–#61) remains a
later phase.

## Known Deviations
1. **Rust Agent dyn-compatibility** — Rust required a `BoxFut` workaround to make
   the `Agent` trait dyn-compatible, a generic-harness asymmetry versus the other
   three languages. Tracked in #45 (`scope: debt`).
2. **Only ReAct is executable** — the README and
   `docs/harness-engineering-concepts.md` advertise five loop strategies, but
   only **ReAct** runs. `PlanExecute`, `Ralph`, `SelfVerifying`, and
   `HillClimbing` return `HaltReason::StrategyNotYetImplemented` at
   `rust/crates/spore-core/src/harness.rs`. Tracked: #59 / #58 / #61 / #60
   (`scope: deferred`). #57's scenario suite is intentionally ReAct-only.
3. **Go outbox is not zero-dependency** — closing #50 added
   `go.opentelemetry.io/otel` + `otlptracegrpc` (v1.28.0) as blessed deps to
   `go/spore-core/go.mod` (accepted tradeoff, documented in `go/CONVENTIONS.md`).
   The durable JSONL path stays network-free.
4. **Observability captures no message content** — the trace pipeline records
   span structure + metrics (tokens, timing, outcomes) but no prompts,
   completions, or tool-call payloads, so it can't show the conversation. Tempo
   is also the wrong tool for rendering LLM events. Tracked in #64.
5. **Sandbox Read of a missing in-workspace file → `PathEscape`** — the
   not-exist fallback in `WorkspaceScopedSandbox` is gated to Write/Execute, so a
   Read of an absent (but in-bounds) path is misclassified as an escape. Makes S1
   brittle. Tracked in #63 (`scope: debt`).

_(Former Deviation: compaction `tokens_reclaimed = 0` — **resolved** in #57.)_

## Next Actions
[3-5 items max. Each references a GH issue # where possible.
This section is updated by /close after every PEE loop.]
1. **#64 — LLM-native agent tracing (conversation + tool-call content via GenAI
   conventions; view in Phoenix).** The north-star gap now that the harness runs:
   make a run debuggable by capturing message + tool-call content (opt-in,
   truncated) following OTel GenAI semantic conventions, add Arize Phoenix to the
   stack as the LLM-trace viewer, and keep the backend swappable over OTLP. Rust
   reference first, then TS/Python/Go parity. Resolve the OpenInference-vs-`gen_ai.*`
   rendering question in Rust before fanning out.
2. **#63 — Sandbox Read of a missing in-workspace file should return a recoverable
   NotFound, not `PathEscape`.** Small, contained robustness fix across all four
   languages; directly de-flakes S1 of the #57 suite. Good warm-up / parallel to #64.
3. **Loop-strategy breadth (#58–#61)** — implement the remaining four strategies
   (Ralph, PlanExecute, SelfVerifying, HillClimbing) so the harness matches its
   advertised capability. Larger, `scope: deferred`; pick up after the agent is
   debuggable.

_Note: #57 closed `status: complete` (a48fe84, PR #62 squash-merged to main) — the
harness runs end-to-end across all four languages, with the prompt-delivery,
token-accounting, and OTLP-attribute fixes that the live run surfaced. New issues
filed: #63 (sandbox debt) and #64 (LLM-native tracing, now the live north star).
Former Known Deviation #2 (compaction reclaims 0) is resolved and removed._
