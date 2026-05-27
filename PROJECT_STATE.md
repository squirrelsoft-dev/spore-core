# PROJECT STATE
_Last updated: 2026-05-27 by /close_

## Current State
spore-core is a language-agnostic agentic harness runtime being built component
by component across four targets: Rust (reference), TypeScript, Python, and Go.
Components land from GitHub issue specs and reach parity across the languages.

**#55 (PR #56) just landed:** the `StandardCompactionAdapter` in all four languages
connects the rich `StandardContextManager` (#29) to the harness compaction seam
(#46). The harness now **actually compacts out of the box** once built with the
adapter — closing the gap where the loop seam was exercised only by test doubles.
The adapter is stateless: it serializes the rich `SessionState` into the harness
`SessionState.extras` under the reserved key `spore.compaction_adapter.rich_state`,
so it survives pause/resume. Cross-language parity verified (type name, reserved
key, and missing-items prompt all byte-identical; fixture-parity drives the real
adapter, not a test double).

Recent prior work: **#41 (PR #53)** `OllamaModelInterface` + `OllamaCacheProvider`
— the first real local model provider with `/api/show` discovery and a
tool-capability guard. **#46 (PR #54)** the post-compaction verify→retry→warn loop
wired into the runtime loop, with an opt-in `emit_warn` event and a
`compaction_verification_failures` metric. Earlier: ObservabilityProvider wired into
the loop with a HarnessBuilder (#49), the local observability stack scaffolded
(#42 — Grafana + Tempo + Loki + Alloy + Prometheus), the OutboxObservabilityProvider
with OTLP forwarding (#33), and CompactionVerifier/KeyTermVerifier (#29).

**main CI is green** across all four languages (Python, Rust, TypeScript, Go).

The system is **still not runnable end-to-end** — no full agent run yet drives a
real model through the complete harness loop. But the two pieces that were blocking
it are now in place: a real local model (#41) and a harness that genuinely compacts
(#55). The remaining gap is assembling a CLI/example that wires a real model +
adapter through the full loop and produces a result. The README still labels the
project "in active design and pre-implementation."

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

## Next Actions
[3-5 items max. Each should reference a GH issue # where possible.
This section is updated by /close --issue after every PEE loop.]
1. **Assemble a first end-to-end run** — wire a real model (#41 Ollama) + the `StandardCompactionAdapter` (#55) through the full harness loop in a CLI or runnable example, across all four languages, and produce a result. This is the north-star gap now that the component prerequisites are in place. No issue yet — likely needs one filed.
2. #50 — Wire the Go OTLP forwarder against the no-op seam (closes observability parity for Go).

_Note: #55 closed (PR #56 merged, 7df874c) — `StandardCompactionAdapter` landed across all four languages with cross-language parity verified; the harness now compacts out of the box. This resolved the former Known Deviation "compaction wired but not active." A new deviation surfaced: token accounting reclaims 0 at the seam (see Known Deviations #2)._
