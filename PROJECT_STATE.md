# PROJECT STATE
_Last updated: 2026-05-29 by /close — closed #80 (Tool escalation protocol) `status: complete`. Merged the four #80 commits to `main` and pushed (tip now `09e328c`, was `a564729`). #80 was the keystone of Track A; it unblocks #81's Tier 3 (harness-escalating) tools, so **#81 is now "work this next."**_

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.

**#80 just landed — tools can now escalate structured signals to the harness's
caller.** A new `HarnessSignal` tagged union (`enter_plan_mode`,
`exit_plan_mode { plan: PlanArtifact }`, `switch_mode { mode: Mode }`,
`abort { reason }`) rides a new `ToolOutput::Escalate { signal }`. When a
dispatched tool returns `Escalate`, the harness loop terminates cleanly: it
**does not append to message history**, preserves the remaining batch tool calls
into `pending_tool_calls`, finalizes observability as the new
`SessionOutcome::Escalated`, and returns `RunResult::Escalate { signal, state,
session_id, usage, turns }`. **The harness never acts on the signal** — it is a
pure intermediary; the caller (CLI/UI/parent harness) owns the orchestration.
`resume()` discards the signal (it's not stored in `PausedState`) and continues
the original session. Rust `672014a`, TS `61f3e50`, Python `304ea71`, Go
`09e328c`. Cross-language verification: all four suites green (Rust full
workspace suite, TS 981, Python 859, Go all modules), wire format structurally
identical across all four. As part of #80, `PausedState.human_request` (and
`ChildPausedState.human_request`) became `Option<HumanRequest>` (`#[serde(default)]`,
emits `null`) so the same `PausedState` serves both the `WaitingForHuman` and
escalation paths.

**Track A is underway**: #80 (tool escalation protocol) is the first of three.
Remaining: #81 (standard tool catalogue — now fully unblocked, Tier 2 rides #75's
`ToolContext`, Tier 3 rides #80's `Escalate`) and #79 (prompt assembly engine,
largely independent).

**The persistence layer is clean** (#73 + #76 + #75): `StorageProvider`
abstraction with per-domain composite routing; `plan_execute` and `task_list`
persist solely through `RunStore`; the standalone `task_list` tool now does too.
`SessionState.extras` is kept only for genuinely ephemeral scratch (`__rich_state`
for compaction, `subagent_handoff_summary` for subagents).

**The full PlanExecute chain is complete and on origin**: #45 (Agent
dyn-compatibility), #69 (lifecycle hooks — 17 events), #70 (one-shot plan phase →
`PlanArtifact`), #71 (`task_list` persisted tool), #72
(`plan_artifact_to_task_list` bridge), #73 (`StorageProvider`), #59 (PlanExecute
execute loop), #76 (extras→RunStore), #75 (Tool storage seam), #80 (tool
escalation protocol). The harness runs **two of five** loop strategies (ReAct +
PlanExecute) end-to-end.

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

**origin/main is at `09e328c`** — all work through #80 is pushed. CI green across
all four languages.

Known runnability limits: the harness runs **ReAct and PlanExecute** end-to-end.
`Ralph`, `SelfVerifying`, and `HillClimbing` (#58/#61/#60) still return the generic
`HaltReason::StrategyNotYetImplemented` — so the README and
`docs/harness-engineering-concepts.md` still overstate what runs (3 of 5
strategies remain stubs).

## Active Direction
The harness is **runnable** (#57), **debuggable** (#64/#65), has a working
**evaluation/feedback loop** (#26/#68), runs **two** of five advertised loop
strategies (ReAct + PlanExecute), has a **clean, fully-pluggable persistence seam
reaching all the way into tools** (#73 + #76 + #75), and now has a **typed tool→
caller escalation channel** (#80). The bar remains **capability breadth and
correctness**.

**Track A — tool/prompt architecture — is the active focus.** #80 (escalation
protocol) just landed, the first of three. Remaining: **#81 (standard tool
catalogue)** — now fully unblocked: Tier 1 execution, Tier 2 session-aware via
#75's `ToolContext`, Tier 3 harness-escalating via #80's `ToolOutput::Escalate`
(its Tier-3 tools — `EnterPlanModeTool`, `ExitPlanModeTool`, etc. — are exactly
the things that *return* `Escalate`). And **#79 (prompt assembly engine —
`ChunkProvider`/`PromptChunk`/`AssemblyContext`)**, largely independent of #80/#81
and can proceed in parallel.

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
4. **#73 deferred follow-ups — both filed.** (a) persistence migration off
   `SessionState.extras` is **fully resolved** (#59 + #76 + #75). (b) SQL backends
   filed as **#77** (`scope: deferred`). Storage scope + workspace partitioning
   filed as **#78** (`scope: deferred`).
5. **Go-specific divergences from #80** (`scope: debt`, minor, documented on the
   issue) — to satisfy the shared escalation fixtures and avoid an import cycle,
   the Go target (a) defines a local `Mode` string newtype in the `sporecore`
   package rather than importing `promptchunkregistry.Mode` (identical bare-string
   wire form), and (b) widened `HarnessObserver.SetSessionOutcome` from `bool` to a
   3-state `TerminalOutcome` (Success/Failure/Escalated). #80 also **resolved** a
   pre-existing latent Go-only wire divergence: it dropped `omitempty` from Go's
   `SessionState.Extras` (now emits `{}`) and `PausedState.ChildState` (now `null`)
   to match the Rust (`#[serde(default)]`) and Python (`default_factory`) siblings.

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
1. **#81 — Standard tool catalogue** (`status: queued`, **work this next**). Now
   fully unblocked by #80. Tier 1 execution / Tier 2 session-aware (rides #75's
   `ToolContext`) / Tier 3 harness-escalating (rides #80's `ToolOutput::Escalate` —
   the Tier-3 tools `EnterPlanModeTool`/`ExitPlanModeTool`/etc. are exactly what
   *return* `Escalate`). Opt-in default tool implementations. The keystone consumer
   that proves both the #75 and #80 seams.
2. **#79 — Prompt assembly engine** (`status: queued`). `ChunkProvider`,
   `PromptChunk`, `AssemblyContext`, `ContextSourcesBuilder` — conditional/
   multi-backend chunk loading extending the #24 `PromptChunkRegistry`. Rounds out
   track A; largely independent of #80/#81 so can proceed in parallel with #81.
3. **Loop strategies — #61 → #58 → #60** (`scope: deferred`, after track A).
   SelfVerifying, Ralph, HillClimbing; reuse the PlanExecute seams (pluggable ReAct
   sub-loop executor, `task_list` drain, `OnTaskAdvance` hook, `RunStore`).
4. **Correctness/safety debt + docs cleanup (#30/#31/#32/#34, #27/#35/#36)** —
   memory distillation gate (#30), read-only subagent context (#31), Block 2 hash
   mismatch halt (#32), dangerous-feature-flag gate (#34); README/spec
   clarifications (#27/#35) and E2B data-residency note (#36). Fold in so docs stop
   overstating capability.
5. **Storage future phases — #78 (scope + workspace partitioning), #77 (SQL
   backends)** — both `scope: deferred`; pick up once a consumer needs them.
