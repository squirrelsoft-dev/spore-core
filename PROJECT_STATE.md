# PROJECT STATE
_Last updated: 2026-05-29 by /close — closed #75 (Tool-trait storage seam) `status: complete`, retiring Deviation #3. Pushed the four #76 + four #75 commits to origin (tip now `ba0e7f9`, was `3fe68f3`). New untriaged tool/prompt-architecture track surfaced (#79/#80/#81); #75 unblocks #81 — direction decision flagged below._

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.

**#75 just landed — tools can now persist through the storage seam.** The `Tool`
trait gained a `ToolContext { session_id, run_store }` parameter on `execute`
(additive, after `sandbox`), construction-injected via the per-run registry
bridge so the **harness-loop dispatch signature is unchanged** in all four
languages. The standalone `task_list` tool is migrated off its
`.spore/task_list.json` sandbox path onto `RunStore` (shared `"task_list"` key,
keyed by `SessionId` — the same blob the PlanExecute loop uses); the sandbox path
and its load/store helpers are deleted everywhere. Rust ref `ed932ac`, TS
`ba0e7f9`, Python `35d609f`, Go `de97f80`. Cross-language verification: all four
suites green (Rust full workspace suite, TS 968, Python 847, Go all packages ok),
`fixtures/tasklist/operations.json` re-pointed to drive over an in-memory
`RunStore` and replays byte-identically; no divergences. **Behavior change:** with
the library default `no_op()` storage, a standalone `task_list` call now persists
**nothing across processes** (it previously wrote a file) — durable persistence
now requires wiring a real `StorageProvider`. No migration shim.

**The persistence layer is clean** (#73 + #76 + #75): `StorageProvider`
abstraction with per-domain composite routing; `plan_execute` and `task_list`
persist solely through `RunStore`; the standalone `task_list` tool now does too.
`SessionState.extras` is kept only for genuinely ephemeral scratch (`__rich_state`
for compaction, `subagent_handoff_summary` for subagents).

**The full PlanExecute chain is complete and on origin**: #45 (Agent
dyn-compatibility), #69 (lifecycle hooks — 17 events), #70 (one-shot plan phase →
`PlanArtifact`), #71 (`task_list` persisted tool), #72
(`plan_artifact_to_task_list` bridge), #73 (`StorageProvider`), #59 (PlanExecute
execute loop), #76 (extras→RunStore), #75 (Tool storage seam). The harness runs
**two of five** loop strategies (ReAct + PlanExecute) end-to-end.

Foundation in place: the harness is **runnable** (#57 — shared e2e CLI driving
the full ReAct loop against Ollama, 4-scenario hermetic suite, live against
`llama3.2`) and **debuggable** (#64/#65 — GenAI-convention content capture,
opt-in/truncated/env-guarded, Arize Phoenix LLM-trace viewer alongside
Tempo/Loki/Prometheus). It has a working **evaluation/feedback loop** (#26
EvalHarness — `TaskSuite` over fresh worktrees, native Welch's t-test +
seeded-bootstrap CIs, Adopt/Reject; #68 — typed `Span` accessors). Earlier work:
observability stack (#42/#49/#50/#33), `OllamaModelInterface` + capability guard
(#41), `StandardCompactionAdapter` + verify→retry→warn loop (#55/#46),
`StandardContextManager`/verifiers (#29), KeyTermVerifier (#47).

**origin/main is at `ba0e7f9`** — all work through #75 is pushed. CI green across
all four languages.

Known runnability limits: the harness runs **ReAct and PlanExecute** end-to-end.
`Ralph`, `SelfVerifying`, and `HillClimbing` (#58/#61/#60) still return the generic
`HaltReason::StrategyNotYetImplemented` — so the README and
`docs/harness-engineering-concepts.md` still overstate what runs (3 of 5
strategies remain stubs).

## Active Direction
The harness is **runnable** (#57), **debuggable** (#64/#65), has a working
**evaluation/feedback loop** (#26/#68), runs **two** of five advertised loop
strategies (ReAct + PlanExecute), and now has a **clean, fully-pluggable
persistence seam reaching all the way into tools** (#73 + #76 + #75). The bar
remains **capability breadth and correctness**.

Two capability tracks are now live and the next loop should pick between them:

1. **Tool/prompt architecture track (newly filed, just unblocked by #75).**
   #80 (tool escalation protocol — `ToolOutput::Escalate`/`HarnessSignal`), #81
   (standard tool catalogue — Tier 1 execution / Tier 2 session-aware via #75's
   `ToolContext` / Tier 3 harness-escalating via #80), and #79 (prompt assembly
   engine — `ChunkProvider`/`PromptChunk`/`AssemblyContext`). #75 directly
   unblocked #81's Tier 2; #80 unblocks #81's Tier 3. This track has momentum and
   fresh context.
2. **Remaining loop strategies.** **SelfVerifying (#61)** is likely-cheapest
   (buildable as a `Stop`-hook configuration on the #69 hook system),
   **Ralph (#58)** is the natural pair to PlanExecute's execute phase, and
   **HillClimbing (#60)** follows once the seams have a second consumer.

After whichever track: the accepted-debt correctness fixes (#30/#31/#32/#34) and
docs/spec cleanup (#27/#35/#36) so the docs stop overstating capability. Storage
stays first-class: scope + workspace partitioning (#78) and SQL backends (#77)
are filed, both deferred.

## Known Deviations
1. **Three of five loop strategies are still stubs** — the README and
   `docs/harness-engineering-concepts.md` advertise five loop strategies; **ReAct
   and PlanExecute run end-to-end**, but `Ralph`, `SelfVerifying`, and
   `HillClimbing` still return `HaltReason::StrategyNotYetImplemented` at
   `rust/crates/spore-core/src/harness.rs`. Tracked: #58 / #61 / #60
   (`scope: deferred`). #57's scenario suite is intentionally ReAct-only.
2. **Go outbox is not zero-dependency** — closing #50 added
   `go.opentelemetry.io/otel` + `otlptracegrpc` (v1.28.0) as blessed deps to
   `go/spore-core/go.mod` (accepted tradeoff, documented in `go/CONVENTIONS.md`).
   The durable JSONL path stays network-free.
3. **`task_list` tool default persistence is now no-op, not a file** (`scope: debt`,
   minor) — #75 retired the `.spore/task_list.json` sandbox path; the standalone
   tool persists via `RunStore`. With the library default `no_op()` storage, a
   standalone tool invocation now persists **nothing across processes** (it
   previously wrote a file). Durable standalone use requires wiring a real
   `StorageProvider` (e.g. `FileSystemStorageProvider`). Accepted tradeoff for
   retiring the sandbox path; no migration shim.
   _(The original Deviation #3 — tool can't reach the storage seam — is **resolved**
   by #75. This entry records the residual behavior change.)_
4. **#73 deferred follow-ups — both filed.** (a) persistence migration off
   `SessionState.extras` is **fully resolved** (#59 + #76 + #75). (b) SQL backends
   filed as **#77** (`scope: deferred`). Storage scope + workspace partitioning
   filed this milestone as **#78** (`scope: deferred`).

_(Former Deviation #3 — tool stuck on sandbox path — **resolved** in #75.)_
_(Former Deviation #4a — extras persistence mirror unmigrated — **resolved** in #76.)_
_(Former Deviation: #72/#73 commits unpushed — **resolved** in the #59 loop.)_
_(Former Deviation: Rust Agent dyn-compatibility / `BoxFut` — **resolved** in #45.)_
_(Former Deviation: compaction `tokens_reclaimed = 0` — **resolved** in #57.)_
_(Former Deviation: sandbox Read of a missing in-workspace file → `PathEscape` — **resolved** in #63.)_
_(Former Deviation: EvalHarness Rust-only Debug-string metric workaround — **resolved** in #68.)_
_(Former Deviation: observability captured no message content — **resolved** in #64.)_

## Next Actions
[3-5 items max. Each references a GH issue # where possible.
This section is updated by /close after every PEE loop.]
1. **DECIDE next track, then start it.** Two live options:
   (A) **Tool/prompt architecture** — start with **#80 (tool escalation protocol)**
   as the keystone (it unblocks #81's Tier 3 and builds on `ToolOutput`, with
   #75's `ToolContext` context still fresh), then **#81 (standard tool catalogue)**
   whose Tier 2 #75 just unblocked, then **#79 (prompt assembly engine)**.
   (B) **Loop strategies** — **#61 (SelfVerifying)** likely-cheapest, then
   **#58 (Ralph)**, then **#60 (HillClimbing)**. #79/#80/#81 are untriaged on the
   board (recommend `status: queued`); the loop strategies are `scope: deferred`.
   Pick A or B at the top of the next loop.
2. **Tool/prompt track (if A): #80 → #81 → #79.** Escalation protocol, then the
   tiered standard tool catalogue (Tier 2 rides #75's `ToolContext`, Tier 3 rides
   #80), then the prompt assembly engine extending #24.
3. **Loop strategies (if B): #61 → #58 → #60.** Reuse the PlanExecute seams
   (pluggable ReAct sub-loop executor, `task_list` drain, `OnTaskAdvance` hook,
   `RunStore` persistence).
4. **Correctness/safety debt + docs cleanup (#30/#31/#32/#34, #27/#35/#36)** —
   memory distillation gate (#30), read-only subagent context (#31), Block 2 hash
   mismatch halt (#32), dangerous-feature-flag gate (#34); README/spec
   clarifications (#27/#35) and E2B data-residency note (#36). Fold in so docs stop
   overstating capability.
5. **Storage future phases — #78 (scope + workspace partitioning), #77 (SQL
   backends)** — both `scope: deferred`; pick up once a consumer needs them.
