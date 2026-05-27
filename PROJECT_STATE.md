# PROJECT STATE
_Last updated: 2026-05-27 by /close --session_

## Current State
spore-core is a language-agnostic agentic harness runtime being built component
by component across four targets: Rust (reference), TypeScript, Python, and Go.
Components land from GitHub issue specs and reach parity across the languages.

Two components landed this session. **#41 (PR #53):** `OllamaModelInterface` +
`OllamaCacheProvider` in all four languages — the first real local model provider
(no API key, runs on the user's hardware), with `/api/show` runtime discovery and
a tool-capability guard. This removes the model-provider gap that was blocking a
first end-to-end run. **#46 (PR #54):** the post-compaction verify→retry→warn loop
is now wired into the harness runtime loop in all four languages, proven against a
shared `compaction_loop` fixture, with a new opt-in `emit_warn` observability event
and a `compaction_verification_failures` metric. Earlier work wired the
ObservabilityProvider into the harness loop with a HarnessBuilder (#49), scaffolded
the local observability stack (#42 — Grafana + Tempo + Loki + Alloy + Prometheus),
landed the OutboxObservabilityProvider with OTLP forwarding (#33), and added
CompactionVerifier/KeyTermVerifier (#29).

**main CI is green** across all four languages (Python, Rust, TypeScript, Go).

The system is **still not runnable end-to-end** — no full agent run yet drives a
real model through the complete harness loop. A key caveat surfaced during #46:
although the compaction loop is "wired," **the harness does not actually compact
out of the box** — the loop seam is exercised only by test-double context managers,
and no production adapter connects the real `StandardContextManager` to it yet
(tracked in #55). So a long-running agent loop will not compact context until #55
lands. The README still labels the project "in active design and
pre-implementation."

## Active Direction
Drive toward a **first end-to-end runnable harness**: a coding agent executing the
full harness loop against a real model, across all four languages. With #41 done,
a real local model is available and the compaction loop is wired; the bar is now to
make the loop actually run and produce a result, not just to implement more
components. The next gap on that path is #55 (connect a real context manager so the
harness actually compacts).

## Known Deviations
1. **Rust Agent dyn-compatibility** — Rust required a `BoxFut` workaround to make
   the `Agent` trait dyn-compatible, introducing a generic-harness asymmetry versus
   the other three languages. Tracked in #45 (labeled `scope: debt`).
2. **Compaction wired but not active** — #46 wired the compaction verify→retry→warn
   loop into the harness, but the loop seam has no production context-manager adapter,
   so the harness does not compact out of the box. The original plan assumed #3's
   single-shot compaction step was already in the loop; it never was. Tracked in #55
   (`status: queued`); clears once #55 lands.

## Next Actions
[3-5 items max. Each should reference a GH issue # where possible.
This section is updated by /close --issue after every PEE loop.]
1. #55 — Wire StandardContextManager into the harness compaction seam (production adapter). #46 landed the compaction verify→retry→warn machinery but the harness does not compact out of the box yet; this connects a real manager — nearest gap on the end-to-end path.
2. #50 — Wire the Go OTLP forwarder against the no-op seam (closes observability parity for Go).

_Note: #46 closed (PR #54 merged, 4188c97) — compaction verify→retry→warn loop wired across all four languages and proven against a shared fixture; the production-adapter remainder was split into #55._
