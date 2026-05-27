# PROJECT STATE
_Last updated: 2026-05-26 by /close --init_

## Current State
spore-core is a language-agnostic agentic harness runtime being built component
by component across four targets: Rust (reference), TypeScript, Python, and Go.
Individual components are landing from GitHub issue specs and are reaching parity
across the four languages — recent work wired the ObservabilityProvider into the
harness loop, added a HarnessBuilder, and implemented the OutboxObservabilityProvider
with OTLP forwarding (#33), plus CompactionVerifier/KeyTermVerifier (#29) and
Ollama model interfaces (#41).

The system is **not yet runnable end-to-end** — there is no full agent run driving
a real model through the complete loop. Components exist and wire together, but
integration into a working coding-agent run is still ahead. The README still labels
the project "in active design and pre-implementation."

## Active Direction
Drive toward a **first end-to-end runnable harness**: a coding agent executing the
full harness loop against a real model, across all four languages. Component work
continues, but the bar shifts from "implement the component" to "make the loop
actually run and produce a result."

## Known Deviations
1. **Rust Agent dyn-compatibility** — Rust required a `BoxFut` workaround to make
   the `Agent` trait dyn-compatible, introducing a generic-harness asymmetry versus
   the other three languages. Tracked in #45.

## Next Actions
[3-5 items max. Each should reference a GH issue # where possible.
This section is updated by /close --issue after every PEE loop.]
1. #46 — Wire the post-compaction verify→retry→warn loop into the harness runtime loop (core loop reliability).
2. #41 — Implement OllamaModelInterface and OllamaCacheProvider (a real local model to actually run the loop).
3. #50 — Wire the Go OTLP forwarder against the no-op seam (Tempo support; closes observability parity on the current branch).
