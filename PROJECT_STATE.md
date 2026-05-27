# PROJECT STATE
_Last updated: 2026-05-27 by /close_

## Current State
spore-core is a language-agnostic agentic harness runtime being built component
by component across four targets: Rust (reference), TypeScript, Python, and Go.
Components land from GitHub issue specs and reach parity across the languages.

**#50 (06d470e) just landed:** Go's `OutboxObservabilityProvider` now forwards spans
to Tempo over OTLP gRPC via a real `otlpSdkForwarder` behind the previously no-op
`otlpForwarder` seam. This closes observability parity — **all four languages now
forward to Tempo** (Go previously appeared only in Loki via JSONL tailing). Trace/span
ids match the other three byte-for-byte (OTLP trace id from the 32-hex JSONL `trace_id`;
8-byte span id via SHA-256 of `SpanID`), so a session collapses into one grouped Tempo
trace. JSONL stays the durable, network-free path; OTLP is best-effort, 2s-bounded, and
never blocks/fails the loop. Decision recorded: `go.opentelemetry.io/otel` + `otlptracegrpc`
(v1.28.0, last Go-1.22-compatible line) are now **blessed deps** in `go/spore-core/go.mod`
— the outbox is no longer zero-dependency (accepted tradeoff, documented in `go/CONVENTIONS.md`).

Recent prior work: **#55 (PR #56)** the `StandardCompactionAdapter` across all four
languages connects the rich `StandardContextManager` (#29) to the harness compaction
seam (#46) — the harness now **actually compacts out of the box**. **#41 (PR #53)**
`OllamaModelInterface` + `OllamaCacheProvider` — the first real local model provider with
`/api/show` discovery and a tool-capability guard. **#46 (PR #54)** the post-compaction
verify→retry→warn loop wired into the runtime loop. Earlier: ObservabilityProvider wired
into the loop with a HarnessBuilder (#49), the local observability stack scaffolded
(#42 — Grafana + Tempo + Loki + Alloy + Prometheus), the OutboxObservabilityProvider
with OTLP forwarding (#33), and CompactionVerifier/KeyTermVerifier (#29).

**main CI is green** across all four languages (Python, Rust, TypeScript, Go).

The system is **still not runnable end-to-end** — no full agent run yet drives a
real model through the complete harness loop. But the pieces that were blocking it
are now in place: a real local model (#41) and a harness that genuinely compacts
(#55), with full observability parity (#50) for tracing any run. The remaining gap
is assembling a CLI/example that wires a real model + adapter through the full loop
and produces a result. The README still labels the project "in active design and
pre-implementation."

## Active Direction
Drive toward a **first end-to-end runnable harness**: a coding agent executing the
full harness loop against a real model, across all four languages. With #41 (real
local model) and #55 (the harness genuinely compacts) both done, the component
prerequisites are in place; the bar is now to **assemble and run** the loop —
wire a real model + context-manager adapter through the complete loop in a CLI or
example and produce a result — not to implement more components.

## Known Deviations
1. **Rust Agent dyn-compatibility** — Rust required a `BoxFut` workaround to make
   the `Agent` trait dyn-compatible, introducing a generic-harness asymmetry versus
   the other three languages. Tracked in #45 (labeled `scope: debt`).
2. **Compaction token accounting reclaims 0 at the seam** — the `StandardCompactionAdapter`
   (#55) builds its `CompactionResult` with `tokens_reclaimed = 0` (consistent across
   all four languages), so utilization stays above threshold after a single compaction.
   In practice the agent's next turn ends the run; the Go end-to-end test asserts
   compaction via the `Compactions` metric rather than a recomputed utilization. Real
   token reclamation accounting is not yet wired. No issue filed yet — file one if this
   blocks the end-to-end run.
3. **Only 1 of 5 loop strategies is implemented** — the README and
   `docs/harness-engineering-concepts.md` advertise five loop strategies, but only
   **ReAct** is executable. `PlanExecute`, `Ralph`, `SelfVerifying`, and `HillClimbing`
   are spec'd enum variants that return `HaltReason::StrategyNotYetImplemented` at
   `rust/crates/spore-core/src/harness.rs:1929-1960`. Now tracked: #59 (PlanExecute),
   #58 (Ralph), #61 (SelfVerifying), #60 (HillClimbing) — all `scope: deferred` as a
   post-#57 phase. #57's scenario suite is intentionally ReAct-only.
4. **Go outbox is no longer zero-dependency** — closing #50 (Option A) added
   `go.opentelemetry.io/otel` + `otlptracegrpc` (v1.28.0) as blessed deps to
   `go/spore-core/go.mod`, walking back the original zero-dep stance for the
   reliability-critical outbox. Accepted and documented in `go/CONVENTIONS.md`; the deps
   are pinned to the last Go-1.22-compatible otel line to hold the toolchain pin. Not a
   regression — a deliberate, recorded tradeoff. The durable JSONL path stays network-free.

## Next Actions
[3-5 items max. Each should reference a GH issue # where possible.
This section is updated by /close --issue after every PEE loop.]
1. **#57 — Assemble a first end-to-end run (Rust CLI + example scenario suite)** — wire a real model (#41 Ollama) + the `StandardCompactionAdapter` (#55) + observability (#49/#50) through the full harness loop in a runnable Rust CLI, and prove it isn't a one-trick pony via **four example scenarios**: S1 multi-step multi-tool task, S2 multi-turn conversation, S3 long session that fires compaction live, S4 tool failure + recovery. Then reach parity across TS/Python/Go. This is the north-star gap now that the component prerequisites are in place (real model, real compaction, full tracing parity). Decided scope: **CLI first, web agent later.** Rust is the reference; lands there first. (S3 is the scenario most likely to expose Known Deviation #2.)
2. **File the token-accounting issue (Known Deviation #2)** — the `StandardCompactionAdapter` reports `tokens_reclaimed = 0`, so utilization stays above threshold after a single compaction. No issue exists yet; file one if/when it blocks the end-to-end run.

_Note: #50 closed (06d470e, pushed to main) with `status: complete` — Go OTLP forwarder wired against the no-op seam; observability now reaches Tempo across all four languages. New Known Deviation #3 recorded: the Go outbox took on otel deps (Option A) and is no longer zero-dependency. The open board is now entirely `scope: deferred`/`scope: debt` (no queued component work) — the end-to-end assembly is the live north star._
