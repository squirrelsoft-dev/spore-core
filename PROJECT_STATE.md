# PROJECT STATE
_Last updated: 2026-05-26 by /close --session_

## Current State
spore-core is a language-agnostic agentic harness runtime being built component
by component across four targets: Rust (reference), TypeScript, Python, and Go.
Components land from GitHub issue specs and reach parity across the languages.

As of today, **#41 is merged** (PR #53): `OllamaModelInterface` +
`OllamaCacheProvider` exist in all four languages, with `/api/show` runtime
discovery (context window + tool capabilities) and a tool-capability guard. This
is the first real local model provider — no API key, runs on the user's hardware
— and it removes the model-provider gap that was blocking a first end-to-end run.
Earlier work wired the ObservabilityProvider into the harness loop with a
HarnessBuilder (#49), scaffolded the local observability stack (#42 — Grafana +
Tempo + Loki + Alloy + Prometheus), landed the OutboxObservabilityProvider with
OTLP forwarding (#33), and added CompactionVerifier/KeyTermVerifier (#29).

**main CI is green** across all four languages (Python, Rust, TypeScript, Go) —
the previously red `test` jobs were fixed by #52 and confirmed green on the #53
merge.

The system is **still not runnable end-to-end** — no full agent run yet drives a
real model through the complete harness loop. The pieces now exist and wire
together, and a real local model is in place, but the loop has not been exercised
to produce a result. The nearest reliability gap is #46 (post-compaction
verify→retry→warn loop). The README still labels the project "in active design
and pre-implementation."

## Active Direction
Drive toward a **first end-to-end runnable harness**: a coding agent executing the
full harness loop against a real model, across all four languages. With #41 done,
a real local model is available; the bar is now to make the loop actually run and
produce a result, not just to implement more components. The next reliability gap
on that path is #46.

## Known Deviations
1. **Rust Agent dyn-compatibility** — Rust required a `BoxFut` workaround to make
   the `Agent` trait dyn-compatible, introducing a generic-harness asymmetry versus
   the other three languages. Tracked in #45 (labeled `scope: debt`).

## Next Actions
[3-5 items max. Each should reference a GH issue # where possible.
This section is updated by /close --issue after every PEE loop.]
1. #46 — Wire the post-compaction verify→retry→warn loop into the harness runtime loop (core loop reliability; nearest gap on the end-to-end path).
2. #50 — Wire the Go OTLP forwarder against the no-op seam (closes observability parity for Go).
