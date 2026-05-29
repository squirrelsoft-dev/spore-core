# PROJECT STATE
_Last updated: 2026-05-29 by /close — closed #75 (Tool-trait storage seam) `status: complete`, retiring Deviation #3. Pushed the four #76 + four #75 commits to origin (tip now `ba0e7f9`, was `3fe68f3`). Triaged the new tool/prompt-architecture track #79/#80/#81 as `status: queued` and **chose Track A (tool/prompt architecture) as the next focus** — #80 is "work this next."_

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

**Track A — tool/prompt architecture — is the chosen next focus** (decided this
loop). #75 just unblocked it: #80 (tool escalation protocol —
`ToolOutput::Escalate`/`HarnessSignal`), #81 (standard tool catalogue — Tier 1
execution / Tier 2 session-aware via #75's `ToolContext` / Tier 3
harness-escalating via #80), and #79 (prompt assembly engine —
`ChunkProvider`/`PromptChunk`/`AssemblyContext`). #75 directly unblocked #81's
Tier 2; #80 unblocks #81's Tier 3. All three are `status: queued`.

The remaining **loop strategies** are deferred behind track A: **SelfVerifying
(#61)** likely-cheapest (a `Stop`-hook configuration on the #69 hook system),
**Ralph (#58)** the natural pair to PlanExecute's execute phase, **HillClimbing
(#60)** once the seams have a second consumer.

After track A: the accepted-debt correctness fixes (#30/#31/#32/#34) and
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
1. **#80 — Tool escalation protocol** (`status: queued`). The keystone of the
   chosen track A: `HarnessSignal`, `ToolOutput::Escalate`, `RunResult::Escalate` —
   the typed channel by which a tool signals the harness to terminate cleanly and
   pass a structured signal to its caller (the harness never acts on it). Builds on
   `ToolOutput` with #75's `ToolContext` context still fresh; unblocks #81's Tier 3.
   Work this next.
2. **#81 — Standard tool catalogue** (`status: queued`). Tier 1 execution /
   Tier 2 session-aware (rides #75's `ToolContext`, already unblocked) / Tier 3
   harness-escalating (rides #80). Opt-in default tool implementations.
3. **#79 — Prompt assembly engine** (`status: queued`). `ChunkProvider`,
   `PromptChunk`, `AssemblyContext`, `ContextSourcesBuilder` — conditional/
   multi-backend chunk loading extending the #24 `PromptChunkRegistry`. Rounds out
   track A; can also proceed in parallel as it's largely independent of #80/#81.
4. **Loop strategies — #61 → #58 → #60** (`scope: deferred`, after track A).
   SelfVerifying, Ralph, HillClimbing; reuse the PlanExecute seams (pluggable ReAct
   sub-loop executor, `task_list` drain, `OnTaskAdvance` hook, `RunStore`).
5. **Correctness/safety debt + docs cleanup (#30/#31/#32/#34, #27/#35/#36)** —
   memory distillation gate (#30), read-only subagent context (#31), Block 2 hash
   mismatch halt (#32), dangerous-feature-flag gate (#34); README/spec
   clarifications (#27/#35) and E2B data-residency note (#36). Fold in so docs stop
   overstating capability.
6. **Storage future phases — #78 (scope + workspace partitioning), #77 (SQL
   backends)** — both `scope: deferred`; pick up once a consumer needs them.
