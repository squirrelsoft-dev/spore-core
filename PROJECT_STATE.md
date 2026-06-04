# PROJECT STATE
_Last updated: 2026-06-04 by /close — closed #100 (example: 11-multi-agent — orchestrator and worker composition) `status: complete`. The **11-multi-agent** example now ships end-to-end across all four targets: an orchestrator agent delegates to a research worker and a writing worker (each a fully independent `Harness` instance, no shared mutable state — the "agent as tool" pattern), plans are printed in the example output, and the run assembles a single `report.md`. Building it on top of `08-plan-execute` (PlanExecute drives the orchestrator) surfaced three harness fixes that landed under the same issue across all four languages: a tunable-compaction `context_manager` / `HarnessBuilder::context_manager` setter; treating a clean `EndTurn` with no content as **completion** rather than `EmptyResponse`; and keeping `08-plan-execute` within small (128K-class) context windows by distilling `web_search` output and compacting earlier. Prereqs #102 (session-state round-trip) and #103 (streaming) were already closed. Commits through `2653de8`, all on `main`, pushed (`origin/main` == `2653de8`)._

_**Direction note for this loop:** the prior `/close` (2026-05-30) named correctness/safety gates (#34 → #31 → #30) as the active track, but **that track was never picked up** — every commit since (#98 docs, #99 10-hill-climbing, #100 11-multi-agent) has been building out the **examples suite**. Active Direction is rewritten below to reflect the real course; the correctness gates are reclassified as a parked track pending an explicit maintainer call._

## Current State
spore-core is a language-agnostic agentic harness runtime with a **complete core
capability surface**, now being demonstrated through a numbered **examples suite**
built across all four targets: Rust (reference), TypeScript, Python, Go.
**Everything is pushed — `origin/main` == local == `2653de8`.**

**Examples suite — 11 of 13 landed, all four languages each.** Present under
`examples/{rust,typescript,python,go}/`:
`01-hello-agent`, `02-conversational-repl`, `03-tool-use`, `04-filesystem-agent`,
`05-custom-sandboxed-tool`, `06-web-research`, `07-memory`, `08-plan-execute`,
`09-self-verifying`, `10-hill-climbing` (#99), `11-multi-agent` (#100, this loop).
Each example teaches one harness capability and runs against a local Ollama model;
the later web-search-dependent ones (06, 11) distill `web_search` output so they
stay inside small context windows. **Remaining example issues: #92** (observability
example — wire Phoenix/OTLP tracing, show structured trace output), **#101**
(`12-cordyceps` — fully autonomous task-completion capstone), **#109**
(`13-coding-agent` — batteries-not-included coding-agent CLI).

**Harness fixes shipped alongside the examples (all four languages):** a tunable
`context_manager` setter on `HarnessBuilder` so an example can tighten compaction;
clean `EndTurn`-with-no-content now means completion, not `EmptyResponse`
(removes a spurious error path on terminal turns); `08-plan-execute`
`web_search`-distillation + earlier compaction so the PlanExecute orchestrator
fits a 128K-class model.

**The harness core is DONE and was complete before the examples push began:**

- **All 5 of 5 advertised loop strategies run end-to-end across all four
  languages** — ReAct + PlanExecute + SelfVerifying (#61) + Ralph (#58, incl. its
  v2 `VcsProvider` seam) + HillClimbing (#60). No loop strategy is stubbed.
  HillClimbing writes a 6-decimal byte-identical TSV results log to
  `.spore/results/{task_id}.tsv`; Ralph is a multi-context-window continuation
  loop driven off a Stop hook reading `.spore/progress.json`, reloading state +
  optional `Recent VCS history:` through the `VcsProvider` seam.
- **Track A — tool/prompt architecture — DONE** (#79 prompt assembly engine, #80
  typed tool→caller escalation channel, #81 three-tier standard tool catalogue,
  #82 scope-aware `memory` tool).
- **Clean, fully-pluggable, scope-aware persistence seam** (#73/#75/#76/#78/#82):
  `StorageProvider` with per-domain + per-scope composite routing;
  `plan_execute`/`task_list`/`todo_write` persist via `RunStore`; the `memory`
  tool via the scoped `MemoryStore` (`StorageScope { User, Project, Local }`,
  merged-read single-sourced on the trait per #82 D2).
- **Runnable** (#57 — shared e2e CLI on Ollama, hermetic + live), **debuggable**
  (#64/#65 — GenAI content capture, Arize Phoenix trace viewer), with a working
  **evaluation/feedback loop** (#26 EvalHarness, #68 typed `Span` accessors).

**Parked (not the active track): correctness/safety debt + docs cleanup.** #32
(Block-2 cache-hash-mismatch halt) landed on the prior loop, but the rest of that
track was never started: **#34** (`Mode::Yolo`/`SandboxProvider::None` behind a
dangerous feature flag), **#31** (SharedSession subagent context read-only by
default), **#30** (memory distillation through the PendingReview gate), and docs
**#27/#35/#36**. All still `scope: deferred`.

## Active Direction
**Build out and harden the numbered examples suite across all four languages** —
each example a self-contained, runnable demonstration of one harness capability,
with four-language parity and small-context-window friendliness so they run on
local Ollama models. The suite is the current product surface: it's how the
finished harness core (5/5 loop strategies, full tool/prompt/persistence
architecture) gets shown and validated. **11 of 13 examples are in;** the
remaining work is the last two examples plus an observability example, and
hardening the `web_search` tool the research-style examples depend on.

A design session on **#101** (`12-cordyceps`) reshaped the capstone and spun out
a **new blocking harness issue, #114** (mid-loop consult primitive): a subagent
escalates mid-loop to a parent-spawned helper and resumes, orchestrator-mediated
and deterministic, reusing the `WaitingForHuman` pause/resume machinery (depth-1
respected — the helper is the orchestrator's child, not the worker's). #101 was
rewritten to consume it (ReAct + `task_list` orchestrator decomposing a per-crate/
per-module Rust audit; analysis worker loads an `audit` skill via the real #9
`GuideRegistry`→injection seam; two consult tools — `research_best_practices`
≤5 soft-fail, `consult_advisor` ≤3 → human; gemma4:e4b locally + minimax-m3:cloud
advisor; REPL approval → `gh` issue filing) and is now **`status: blocked` on #114**.

Immediate scope: **#114** (consult primitive) lands first, then **#101**
(`12-cordyceps` capstone) and **#109**
(`13-coding-agent`) complete the numbered suite; **#92** adds the observability
example. The `web_search` tool has two real gaps surfaced by the
research-dependent examples — **#108** (can't attach auth headers or query params,
which blocks several backends) and **#110** (response normalization across
Brave/Tavily/SearXNG shapes) — that should be closed to make those examples
robust across providers.

**Parked track (needs a maintainer call):** correctness/safety gates
#34 → #31 → #30 and docs #27/#35/#36 were named active on 2026-05-30 but never
started; the examples suite took priority instead. Pick these back up explicitly
when the suite is done, or confirm they stay parked. Larger new feature issues —
#113 (spore-lsp), #107 (PromptEngineeringAgent), #106 (MicroVMSandboxProvider) —
and the protocol-integration track (#83–87) remain unscheduled. Storage follow-ups
#77/#88/#89 stay `scope: deferred`.

## Known Deviations
1. **Go outbox is not zero-dependency** — closing #50 added
   `go.opentelemetry.io/otel` + `otlptracegrpc` (v1.28.0) as blessed deps
   (accepted tradeoff, documented in `go/CONVENTIONS.md`). The durable JSONL path
   stays network-free.
2. **`task_list` / `todo_write` tool default persistence is no-op, not a file**
   (`scope: debt`, minor) — #75 retired the `.spore/task_list.json` sandbox path;
   the standalone tools persist via `RunStore`. With the library default
   `no_op()` storage, a standalone invocation persists nothing across processes.
   Durable standalone use requires wiring a real `StorageProvider`. Accepted
   tradeoff; no migration shim.
3. **v1 memory keying limitation (#78 Q7), filed as #89** (`scope: deferred`,
   future phase) — `MemoryStore` is still `SessionId`-keyed, so #82's
   `MemoryTool` can only address the current session; durable session-independent
   cross-session addressing is the v2 feature. Documented in each language's
   `MemoryTool` module header. No SQL backend yet either (#77).
4. **Go-specific divergences (#80 + #81 + #79 + #78 + #82 + #61 + #58 + #60)**
   (`scope: debt`, minor, documented on the issues) — (a) local `Mode` newtype;
   (b) 3-state `TerminalOutcome`; (c) `StandardTool` in root `sporecore`
   type-aliased into `tools`; (d) `sendMessageToolName` duplicated in
   `harness.go`; (e) `abort` tool enforces required `reason` explicitly; (f)
   `promptassembly` owns a self-contained `HarnessBuilder`; (g)
   `ToolContext.MemoryStore` is an opaque marker interface asserted back to
   `storage.MemoryStore`; (h) #82 D2 single-source merge is an exported
   `storage.MergeMemories` (Go interfaces can't carry defaults); (i) #61
   `Verifier`/`EvaluatorAgent` config set directly on the struct with no builder
   setters (Go has no `PlannerAgent` setter either); (j) `role-evaluator` chunk
   is a `RoleEvaluatorChunk` constant in `sporecore` (can't import the registry);
   (k) the Ralph Stop hook is registered in `NewStandardHarness`, and
   `MaxResets`/the Ralph types / the v2 `VcsProvider` field are set directly on
   the config struct (no builder setter), same as (i). (l) #60 required a
   consumer-side `MetricEvaluator` seam interface in the root `sporecore` package
   (with an `AsHarnessMetricEvaluator` bridge in `metric`) to avoid an import cycle
   — same pattern as `Verifier`/`VcsProvider`; `MetricEvaluator` config + the
   `HillClimbing` fields set on the struct, no builder setter. All
   wire/behavior-identical to the other three (verified).
5. **Test-placement divergences (#78/#82)** (benign) — the #78 R9 registry-seam
   test lives in `@spore/tools` (TS) / the eval suite (Python); #82 reused TS
   `@spore/tools` `tool-context-memory-seam.test.ts` and keeps the Python
   catalogue test in `spore_tools`. Behavior identical.
6. **#79 cross-language divergences — both verified benign.** (a)
   `ContextSources.composed_prompt` carries the full `ComposedPrompt` in
   Rust/Python but a narrowed stub in TS/Go (outcomes identical). (b) The Block-1
   hash is not byte-identical (Rust SipHash vs. FNV-1a elsewhere) — the
   intentional #24 decision; #79 fixtures assert no hash values, only condition
   booleans and ordered bucket id lists.
7. **`Custom` condition is invisible in fixtures by design** (#79) —
   `ChunkCondition::Custom(predicate)` is first-class in the API but serializes
   to null/absent; architects using it knowingly opt that chunk out of the
   byte-identical cross-language contract.
8. **Cache-hash-mismatch halts are not live-wired (#32, depends on #7)** (`scope:
   deferred`, intentional) — BOTH Block-1 (Static) and Block-2 (PerSession)
   `CacheHashMismatch` halts live in `StandardContextManager::assemble` (fallible,
   the #7 canonical trait), but the live `StandardHarness` loop calls a separate,
   *infallible* placeholder `ContextManager` whose `assemble` returns `Context`
   not `Result`. So neither halt can fire end-to-end until the deferred #7
   ContextManager migration widens the live trait. #32 added the
   `HaltReason::ContextError` routing variant (all four) so the type is ready;
   wiring the live loop is #7's job. Go models this with a thin routing-level
   `sporecore.ContextError` (block as snake_case string) bridging to the rich
   `CacheBlock`-typed `contextmgr.ContextError`, to avoid an import cycle — same
   consumer-side-seam pattern as `MetricEvaluator`/`Verifier`/`VcsProvider`;
   wire-identical.

9. **Active direction shifted off the 2026-05-30 plan without a recorded
   decision** (process note, not a code deviation) — the correctness/safety track
   (#34/#31/#30) was named active but never started; the examples suite (#98–#100)
   was built instead. Reconciled this loop: Active Direction now reflects the
   examples suite; the correctness track is explicitly parked pending a maintainer
   call. No code impact.

_(Former Deviation — HillClimbing loop strategy stub, #60 (was Deviation #1: harness ran only 4/5 strategies) — **resolved** this loop; HillClimbing runs end-to-end across all four, harness now at 5/5, stub removed.)_
_(Former Deviation — push blocked on local SSH auth (13 unpushed commits) — **resolved**; SSH fixed, all pushed.)_
_(Former Deviation — Ralph git-log reload deferred to v2 (#58 B4) — **resolved** by the `VcsProvider` seam; Ralph now reloads all three spec'd sources.)_
_(Former Deviation — SelfVerifying strategy stub, #61 — **resolved** in #61.)_
_(Former Deviation — `MemoryTool` deferred/blocked — **resolved** in #82.)_
_(Former Deviation — storage scope + partitioning, #78 — **resolved** in #78.)_
_(Former Deviation — tool stuck on sandbox path — **resolved** in #75.)_
_(Former Deviation — extras persistence mirror unmigrated — **resolved** in #76.)_
_(Former Deviation — Rust Agent dyn-compatibility / `BoxFut` — **resolved** in #45.)_
_(Former Deviation — compaction `tokens_reclaimed = 0` — **resolved** in #57.)_
_(Former Deviation — observability captured no message content — **resolved** in #64.)_

## Next Actions
[3-5 items max, highest priority first. Each references a GH issue # where
possible. /next surfaces item 1 as "work this next."]
1. **#114 — mid-loop consult primitive (harness change, BLOCKS #101)** — new
   `RunResult::Consult` / `ToolOutput::Consult` / `ConsultResponse` resume path,
   orchestrator-mediated deterministic routing via a kind→handler map, per-kind
   budgets + overflow policy, depth-1 respected. Four-language parity (Rust
   reference, then TS/Python/Go), on the order of the #100 setter work but more
   invasive. Land before #101. Relabel `status: queued` → `queued`/grabbed.
2. **#101 — `12-cordyceps` example (autonomous task-completion capstone)** —
   **`status: blocked` on #114**; it's the consumer that validates the consult
   primitive. Design is fully pinned in the rewritten issue (ReAct + `task_list`
   per-crate/module Rust audit, `audit` skill via #9 seam, two consult tools,
   gemma4:e4b + minimax-m3:cloud advisor, REPL approval → `gh` filing). Grab once
   #114 lands; relabel `status: blocked` → grabbed.
3. **#109 — `13-coding-agent` example** — the final numbered example, a
   batteries-not-included coding-agent CLI; completes the 1–13 suite across all
   four languages.
4. **#108 + #110 — `web_search` hardening** — #108 (attach auth headers / query
   params; blocks several backends) and #110 (normalize cited sources across
   Brave/Tavily/SearXNG shapes). Both surfaced by the research-dependent examples
   (06, 11) and make them robust across providers. NOT a blocker for #101 —
   #101 defaults to a single local SearXNG backend, which is sufficient.
5. **#92 — observability example** — wire Phoenix/OTLP tracing and show structured
   trace output; demonstrates the already-shipped #64/#65 observability stack.
   (#101 deliberately skips observability; #92 owns the Phoenix wiring.)

_Maintainer decision — parked correctness/safety + docs track._ #34
(Yolo/None feature flag) → #31 (SharedSession read-only) → #30 (memory
PendingReview gate), then docs #27/#35/#36. Named active 2026-05-30, never
started, superseded in practice by the examples suite. Confirm: pick back up
after the suite, or keep parked. Other unscheduled work: #113 (spore-lsp), #107
(PromptEngineeringAgent), #106 (MicroVMSandboxProvider), protocol track #83–87,
deferred storage #77/#88/#89, #7 ContextManager migration (would live-wire the
#32 halts), #90 cache cost-spike.
