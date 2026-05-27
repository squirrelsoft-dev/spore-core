# PROJECT STATE
_Last updated: 2026-05-26 by /close --session_

## Current State
spore-core is a language-agnostic agentic harness runtime being built component
by component across four targets: Rust (reference), TypeScript, Python, and Go.
Components are landing from GitHub issue specs and reaching parity across the
languages. As of today, PR #49 is **merged to main**: the ObservabilityProvider
is wired into the harness loop, a HarnessBuilder exists, and the local
observability stack (#42 — Grafana + Tempo + Loki + Alloy + Prometheus) is fully
scaffolded and closed. Earlier work landed the OutboxObservabilityProvider with
OTLP forwarding (#33) and CompactionVerifier/KeyTermVerifier (#29) across all four
languages.

The system is **still not runnable end-to-end** — there is no full agent run
driving a real model through the complete loop. Components exist and wire together,
but a working coding-agent run is still ahead, gated chiefly on a real local model
(#41). The README still labels the project "in active design and pre-implementation."

**main CI is currently red:** the `test` jobs for Go, Rust, and TypeScript are
failing (only Python passes). PR #49 was merged over these pre-existing failures
with an admin override. See Known Deviations #2.

## Active Direction
Drive toward a **first end-to-end runnable harness**: a coding agent executing the
full harness loop against a real model, across all four languages. Component work
continues, but the bar shifts from "implement the component" to "make the loop
actually run and produce a result."

## Known Deviations
1. **Rust Agent dyn-compatibility** — Rust required a `BoxFut` workaround to make
   the `Agent` trait dyn-compatible, introducing a generic-harness asymmetry versus
   the other three languages. Tracked in #45 (labeled `scope: debt`).
2. **main CI is red** — the Go, Rust, and TypeScript `test` jobs fail on main;
   only Python passes. PR #49 was merged with an admin override over these
   pre-existing failures (decision: address via a follow-up plan rather than block
   the merge). Tracked in #51.

## Next Actions
[3-5 items max. Each should reference a GH issue # where possible.
This section is updated by /close --issue after every PEE loop.]
1. #51 — Fix the failing CI on main (Go vet / Rust fmt / TypeScript eslint) — red main blocks confident future merges.
2. #41 — Implement OllamaModelInterface and OllamaCacheProvider (the real local model that directly unblocks a first end-to-end run).
3. #46 — Wire the post-compaction verify→retry→warn loop into the harness runtime loop (core loop reliability).
4. #50 — Wire the Go OTLP forwarder against the no-op seam (closes observability parity for Go).
