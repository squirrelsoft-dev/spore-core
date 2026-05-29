# PROJECT STATE
_Last updated: 2026-05-29 by /close — closed #59 (PlanExecute execute loop) `status: complete`. **PlanExecute now runs end-to-end** across all four languages, completing the entire #45→#69→#70→#71→#72→#73→#59 chain. All commits (incl. the previously-unpushed #72/#73) are now on `origin/main` (tip `9b9acd6`)._

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.

**#59 (PlanExecute execute loop) just landed — the harness now runs TWO loop
strategies end-to-end (ReAct + PlanExecute).** Rust ref `384ca28`, TS `6374749`,
Python `0bd9562`, Go `9b9acd6`; merged to `main` and **pushed to origin** (tip
`9b9acd6`). This was the last piece of the PlanExecute chain: `run_plan_execute`
now runs the one-shot plan phase (#70) → parses the artifact into a `TaskList`
via `plan_artifact_to_task_list` (#72) → persists it → drains it through a real
execute phase, replacing the `ExecutePhaseNotImplemented` halt. The five spec
forks were resolved by the maintainer before coding and applied identically in
all four:
- **Q1 — execute step model:** each task gets its own bounded, **isolated,
  sequential** ReAct sub-loop (reusing `run_react_inner`); per-task turn cap is
  derived at each step start as `floor(remaining_turns / remaining_tasks)`
  floored at 1; the shared budget (turns/tokens/observability/compaction) is
  carried across all steps as the global hard stop.
- **Q2 — success output:** the last completed step's final-response text (not a
  concatenation, not the plan rationale).
- **Q3 — empty plan** (`tasks: []`): new `HaltReason::EmptyPlan` failure (no
  silent success).
- **Q4 — persistence migrated onto `StorageProvider`/`RunStore`:** the execute
  loop writes the durable task-list/plan copy through `RunStore` (`storage.run().put`)
  plus the `extras` mirror; it **no longer uses** the `.spore/task_list.json`
  sandbox path. (The standalone `task_list` *tool* keeps its own sandbox seam —
  unchanged.)
- **Q5 — per-task failure:** new `HaltReason::StepFailed { task_index, task,
  reason }` aborts the whole run (later tasks don't run); mid-execute budget
  exhaustion stays `BudgetExceeded`.

`HaltReason` across all four: added `EmptyPlan` + `StepFailed`, removed
`ExecutePhaseNotImplemented`. `OnTaskAdvance` (#69) fires once per task with
correct `task_index`/`total_tasks`. Shared fixture
`fixtures/model_responses/harness/plan_execute_loop.jsonl` (1 plan turn → 2 tasks,
then 2 execute completions) replays identically in all four (output
`"wrote the integration tests"`, 3 turns, both tasks completed). Cross-language
verification: all four suites green (Rust 56 incl. fixture-replay, TS 828 core,
Python 843, Go 875), no divergences, no spec gaps, no new issues spawned.

**The full PlanExecute prerequisite chain is now complete and on origin:** #45
(Agent dyn-compatibility), #69 (lifecycle hook system — 17 events, `Stop` +
`OnPlanCreated` + `OnTaskAdvance` now loop-wired), #70 (one-shot plan phase →
`PlanArtifact`), #71 (`task_list` persisted tool), #72 (`plan_artifact_to_task_list`
pure bridge), #73 (`StorageProvider` abstraction — four domain stores, NoOp/
InMemory/FileSystem/Composite providers, multi-endpoint OTLP fan-out, scope "1a").

Foundation in place: the harness is **runnable** (#57 — shared e2e CLI driving
the full ReAct loop against Ollama, proven by a 4-scenario hermetic suite, live
against `llama3.2`) and **debuggable** (#64/#65 — GenAI-convention content
capture, opt-in/truncated/env-guarded, Arize Phoenix LLM-trace viewer alongside
Tempo/Loki/Prometheus, assistant turns recorded). It has a working
**evaluation/feedback loop** (#26 EvalHarness — `TaskSuite` over fresh worktrees,
native Welch's t-test + seeded-bootstrap CIs, Adopt/Reject recommendation; #68 —
typed `Span` accessors, all four EvalHarnesses structurally identical). Earlier
work: observability stack (#42/#49/#50/#33), `OllamaModelInterface` + capability
guard (#41), `StandardCompactionAdapter` + verify→retry→warn loop (#55/#46),
`StandardContextManager`/verifiers (#29), and KeyTermVerifier honoring all five
preserve hints (#47).

**main CI is green** across all four languages.

Known runnability limits: the harness now runs **ReAct and PlanExecute**
end-to-end. `Ralph`, `SelfVerifying`, and `HillClimbing` (#58/#60/#61) still
return the generic `HaltReason::StrategyNotYetImplemented` — so the README and
`docs/harness-engineering-concepts.md` still overstate what runs (3 of 5
strategies remain stubs).

## Active Direction
The harness is **runnable** (#57) and **debuggable** (#64/#65), has a working
**evaluation/feedback loop** (#26/#68), and now runs **two** of its five
advertised loop strategies (ReAct + PlanExecute #59). The bar remains
**capability breadth and correctness**: deliver the remaining loop strategies
and close the queue of known correctness/safety gaps.

PlanExecute (#59) was built deep-first specifically to establish the shared
seams the other strategies reuse — the pluggable executor (ReAct sub-loop by
default, swappable for a future Ralph-style execute+verify), the `task_list`
drain, the `OnTaskAdvance` hook, and the `RunStore` persistence path. With those
seams live, **the remaining three strategies follow**: Ralph (#58), HillClimbing
(#60), and SelfVerifying (#61) — the last of which can now be built as a
`Stop`-hook configuration on the #69 hook system rather than a bespoke loop.
After the strategies come the accepted-debt correctness fixes (#30/#31/#32/#34)
and the docs/spec cleanup (#27/#35/#36) so the docs stop overstating capability.
Observability stays swappable over OTLP. Storage backends stay a first-class
pluggable seam (#73), with the remaining persistence migration + SQL backends as
follow-ups (Deviation #4).

## Known Deviations
1. **Three of five loop strategies are still stubs** — the README and
   `docs/harness-engineering-concepts.md` advertise five loop strategies; **ReAct
   and PlanExecute (#59) now run end-to-end**, but `Ralph`, `SelfVerifying`, and
   `HillClimbing` still return `HaltReason::StrategyNotYetImplemented` at
   `rust/crates/spore-core/src/harness.rs`. Tracked: #58 / #61 / #60
   (`scope: deferred`). #57's scenario suite is intentionally ReAct-only.
2. **Go outbox is not zero-dependency** — closing #50 added
   `go.opentelemetry.io/otel` + `otlptracegrpc` (v1.28.0) as blessed deps to
   `go/spore-core/go.mod` (accepted tradeoff, documented in `go/CONVENTIONS.md`).
   The durable JSONL path stays network-free.
3. **`task_list` *tool* persistence is still the interim sandbox path**
   (`scope: debt`) — the standalone `task_list` tool persists to
   `.spore/task_list.json` via `SandboxProvider` because the `Tool` trait has no
   `SessionState`/shared-store access. **The PlanExecute execute loop (#59) no
   longer uses this path** — it migrated to `RunStore` per Q4 (see resolution
   below) — but the tool itself, when invoked standalone, still writes to the
   sandbox file. Fully retiring the tool's sandbox path needs the `Tool` trait to
   reach the storage seam; folded into the remaining persistence-migration
   follow-up (Deviation #4a).
4. **#73 deferred follow-ups — persistence migration partially done; SQL backends
   not started** (`scope: deferred`) — #73 (scope "1a") shipped the
   `StorageProvider` abstraction only, deferring (a) migrating #70/#71 persistence
   onto `RunStore` + removing `SessionState.extras`, and (b)
   `Sqlite`/`Postgres` providers. **(a) is now partially resolved by #59**, which
   migrated the execute loop's task-list + plan persistence onto `RunStore`;
   **still outstanding:** removing `SessionState.extras` entirely (the `extras`
   mirror is still written for live readback) and migrating the remaining #70
   plan-phase sites + the standalone `task_list` tool (Deviation #3). **(b)** SQL
   providers remain a future phase. **These still need discrete follow-up issues
   filed** (not yet created — pending maintainer go-ahead); until then they live
   only here.

_(Former Deviation: #72/#73 commits unpushed to origin — **resolved** this loop;
all commits through `9b9acd6` are now on `origin/main`.)_
_(Former Deviation: Rust Agent dyn-compatibility / `BoxFut` workaround — **resolved**
in #45 (commit `afe5ff8`).)_
_(Former Deviation: compaction `tokens_reclaimed = 0` — **resolved** in #57.)_
_(Former Deviation: sandbox Read of a missing in-workspace file → `PathEscape` —
**resolved** in #63.)_
_(Former Deviation: EvalHarness Rust-only Debug-string metric workaround —
**resolved** in #68 via a typed `Span` accessor.)_
_(Former Deviation: observability captured no message content — **resolved** in #64
via opt-in GenAI-convention content capture + an Arize Phoenix viewer.)_

## Next Actions
[3-5 items max. Each references a GH issue # where possible.
This section is updated by /close after every PEE loop.]
1. **Next loop strategy — Ralph (#58) or SelfVerifying (#61)** — with the
   PlanExecute seams now live (pluggable ReAct sub-loop executor, `task_list`
   drain, `OnTaskAdvance` hook, `RunStore` persistence), the remaining strategies
   can reuse them. **SelfVerifying (#61)** is the likely-cheapest next step: it
   can be built as a `Stop`-hook configuration on the #69 hook system (separate
   evaluator, Default-FAIL) rather than a bespoke loop. **Ralph (#58)**
   (multi-context-window continuation) is the natural pair to PlanExecute's
   execute phase. Sequencing call to make at the top of the next loop.
2. **File the #73 deferred follow-ups** — now that #59 partially resolved the
   persistence migration (Deviation #4a), file discrete GH issues for the
   remainder: (a) remove `SessionState.extras` + migrate the remaining #70
   plan-phase and standalone `task_list`-tool sites onto `RunStore`, and (b)
   `Sqlite`/`Postgres` storage providers. Low effort, do opportunistically
   (pending maintainer go-ahead on issue creation).
3. **Remaining loop strategies (#60)** — HillClimbing (#60, iterative
   optimization) follows the first two strategies once the shared seams are
   exercised by more than one consumer.
4. **Correctness/safety debt fixes (#30, #31, #32, #34)** — memory distillation
   PendingReview gate (#30), read-only subagent context sharing (#31), halt on
   mid-session Block 2 hash mismatch (#32), and the dangerous-feature-flag gate
   for `Mode::Yolo`/`SandboxProvider::None` (#34). Contained `scope: deferred`
   fixes to pull in opportunistically alongside the loop-strategy work.
5. **Docs/spec cleanup (#27, #35, #36)** — README/spec clarifications (#27/#35)
   and the E2BSandboxProvider data-residency note (#36); fold in once the loop
   strategies land so the docs stop overstating capability (currently 3 of 5
   strategies are still stubs — Deviation #1).
