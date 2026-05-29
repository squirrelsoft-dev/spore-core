# PROJECT STATE
_Last updated: 2026-05-28 by /close — closed #69 (lifecycle hook system, four-language parity, merged to main); #70 remains the headline next item_

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.

**#57 (a48fe84, PR #62) just landed — the harness now runs end-to-end.** A shared
e2e CLI in each language drives the *complete* ReAct loop against a real model
(Ollama) through a `HarnessBuilder`-assembled harness with real tools
(read/write/list/bash + a deliberately-failing `flaky_op`), the
`StandardCompactionAdapter`, and outbox observability. It's proven by a
**4-scenario hermetic suite** — S1 multi-step/multi-tool, S2 multi-turn (shared
session), S3 live compaction, S4 tool-failure + recovery — and was run live
against `llama3.2` (`Success`, real trace written). Rust: `cargo run -p spore-core
--example e2e_agent -- s1`; TS/Python/Go have parity CLIs. CI stays hermetic (mock
agent, no network); live runs are a documented manual recipe. The CLI prints
`session_id` / `trace_path` / `trace_id` per run.

Running it live surfaced and fixed bugs the hermetic suite couldn't catch (the
mock agent ignores the assembled context, so it never noticed the prompt wasn't
delivered):
- **Prompt delivery** — the harness never turned `task.instruction` into a user
  message, so the model received an empty conversation (`EmptyResponse`). Fixed
  in all four languages; seeded only on a fresh `run()`, not on resume.
- **Token accounting (former Known Deviation #2, now resolved)** —
  `tokens_reclaimed` is computed from dropped-message tokens, `token_budget_used`
  decrements live, and the Compaction span stamps real `tokens_after`/
  `tokens_reclaimed`. A session can compact → continue → compact again. All four.
- **OTLP attributes** — the forwarder now flattens per-span attributes (tokens,
  `stop_reason`, `tool_name`, compaction sizes, outcome) onto exported spans.
  Previously only 5 fixed envelope tags reached Tempo. All four.

Earlier work still in place: observability stack (#42), ObservabilityProvider +
OTLP→Tempo / JSONL→Loki parity (#49/#50/#33), `OllamaModelInterface` +
capability guard (#41), `StandardCompactionAdapter` wiring (#55), the
compaction verify→retry→warn loop (#46), and `StandardContextManager` /
verifiers (#29).

**#47 (KeyTermVerifier honors all preserve hints) just landed** — `SessionState`
gained four structured fields (`open_problems`, `architectural_decisions`,
`recent_files` as lists, `reasoning_summary` as a string) at four-language parity,
and `KeyTermVerifier` now collects source terms from all five `CompactionPreserveHints`
in a pinned order (task_instruction → open_problems → architectural_decisions →
recent_files → reasoning_summary), each gated on its hint. Previously only
`keep_current_task_state` contributed terms and the other four hints were silent
no-ops. The #29 extraction rule (lowercase → split `[^a-z0-9]` → drop <4-char tokens →
first-occurrence dedupe → substring match) is unchanged and now applies uniformly to
every field; file paths tokenize via the same rule (`src/parser/mod.rs` → `parser`).
12 shared fixture cases in `fixtures/compaction_verifier/cases.json` replay identically
across all four. Spec pinned on the issue before implementation; the Python compaction
adapter's hand-rolled (de)serializer was extended for the new fields (commits 2d06fab,
fe5fa02, 2db4894, 32232a3).

**#26 (EvalHarness) just landed** — the evaluation harness / tight feedback loop is
implemented at parity across all four languages (`rust/crates/spore-eval`,
`typescript/packages/eval` (new `@spore/eval`), `python/packages/spore_eval`,
`go/spore-eval`). It runs a `TaskSuite` (regression/challenge/canary) of `EvalTask`s
against baseline vs candidate `HarnessConfig`s over fresh per-run git worktrees,
sources task-level signals from `ObservabilityProvider`, and produces a
`ComparisonReport` with native Welch's t-test + seeded-bootstrap CIs and an
Adopt/Reject/NeedsMoreRuns/Ambiguous `Recommendation`. The #26 design discussion's
five open questions were resolved into a numbered-rule spec (posted as a comment on
the issue). Stats use no external library (incomplete-beta + inline SplitMix64), so a
shared `fixtures/task_suites/welch_bootstrap.json` oracle replays byte-for-byte in all
four. `TraceAnalyzer` and challenge→regression auto-promotion are interface-only /
deferred; the nightly optimization-loop meta-agent is future work.

**#68 (typed `Span` accessor) just landed** — the Rust EvalHarness now reads
`TurnSpan`/`SensorSpan`/`MiddlewareSpan` fields through `as_turn()`/`as_sensor()`/
`as_middleware()` accessors on the `Span` trait instead of parsing
`format!("{s:?}")` Debug strings. This removes the last cross-language divergence in
the EvalHarness (former Known Deviation #6, now resolved) — all four implementations
are structurally identical. The refactor also surfaced and fixed a latent parity bug
the old Debug parser had hidden: `ContinueWithModification` was silently classified
as non-intervening in Rust (`"ContinueWithModification"` contains the `"Continue"`
substring) while Go/TS/Python all count it as an intervention. Rust was converged to
the three-language majority and a regression guard test was added in all four
languages (commits 10f00a8, 21c687f, cc09ec1, 387cf33, e826d4f).

**#64 (LLM-native agent tracing) + #65 (record assistant turns) landed** — a run is
now **debuggable**, not just runnable. Observability captures actual message and
tool-call content (prompts, completions, tool args/results) following OTel **GenAI
semantic conventions** — opt-in, truncated, env-guarded (`content_capture` from env),
off by default — with an **Arize Phoenix** backend added as the LLM-trace viewer
alongside the existing Tempo/Loki/Prometheus system telemetry, kept swappable over
OTLP. Turn spans carry the assembled input messages, and assistant turns (plus the
operational system prompt) are recorded in conversation history at parity across all
four languages (#64 commits ccdf645/d37b881/c660208/f13cdeb, 3a06e90,
7ea598e/9379854/9c7c921/8b68b09; #65 commits 1d07c45/497b8fb/e713e1d/00dbb77).

**#45 (Agent dyn-compatibility) is closed** — a `/implement 45` pass discovered the
work was already on `main` (commit `afe5ff8`, landed 2026-05-19) but the issue had
never been closed, so it kept surfacing as the PlanExecute chain's headline blocker.
The Rust `Agent` trait is `fn turn<'a>(&'a self, context: Context) -> BoxFut<'a,
TurnResult>` (no `trait_variant`/RPITIT), `StandardHarness` is non-generic, and
`HarnessConfig.agent` is `Arc<dyn Agent>` — fully consistent with every other
component trait. Option B was chosen; Option C (`trait_variant` dyn variant) ruled
out. Rust-only by nature (TS/Python/Go interfaces were already polymorphic). This
means **#70 is already unblocked** — real `plan_model` routing can use the
`Arc<dyn Agent>` seam today.

**#69 (lifecycle hook system) just landed** — a `Hook`/`HookChain` system at
four-language parity (Rust ref `ddbd8a4`, TS `b49c300`, Go `51bf7f4`, Python
`6423740` + parity fix `c593956`), merged to `main` via fast-forward. It defines all
**17 lifecycle events** (`PreTurn`/`PostTurn`, `PreToolUse`/`PostToolUse`/
`PostToolUseFailure`/`PostToolBatch`, `OnLoopStart`/`Stop`, `OnPause`/`OnResume`/
`OnError`, `OnPlanCreated`/`OnTaskAdvance`/`OnSubagentSpawn`/`OnSubagentComplete`,
`PreCompact`/`PostCompact`) with pinned per-event context schemas, a serde-tagged
`HookDecision` (`continue`/`block`/`inject`/`deny`/`mutate`), and `FunctionHook` +
`CommandHook` (stdin/stdout JSON) handlers. Hooks are a **higher-level layer above**
the existing `MiddlewareChain` (which stays the lower-level context-assembly
primitive), fire in registration order, and validate decision legality per event.
Only **`Stop` is wired into the live ReAct loop** — its `block{reason}` replaces the
old `ForceAnotherTurn`/`CompletionCheck`-returns-reason path and is bounded by a new
`max_stop_blocks` field on `HarnessConfig` (default 8, **per-run** counter that
resets each `run()`); `CompletionCheck` is kept for back-compat and marked deprecated
in Rust. The other 16 events are defined + unit-tested in isolation but not yet
loop-called — they activate when #59/#72 wire the two-phase loop and subagent
spawning. The issue was a conceptual design doc; six contract ambiguities were pinned
before implementation (recorded on the issue). One cross-language divergence surfaced
in verification and was fixed: Python originally left `HookError.kind` unset for
`illegal_decision`/`handler_failed` (`c593956`). Shared fixtures in `fixtures/hooks/`
(decision wire format, Stop block/cap, PreToolUse mutation/deny, command handler I/O)
replay identically across all four. **Note: #69 landed ahead of its place in the
PlanExecute chain (#70→#71→#72→#73→#69→#59)** — it was independent enough to build
early and it unblocks the SelfVerifying-as-Stop-hook approach for #61.

**main CI is green** across all four languages.

Known runnability limits: the harness is **ReAct-only** — the other four loop
strategies return `StrategyNotYetImplemented` (#58–#61). The sandbox
missing-file misclassification that made S1 nondeterministic was fixed in #63, the
previous "no message content" observability gap is closed by #64, and the lifecycle
hook seams (#69) now exist but only `Stop` is loop-wired so far.

## Active Direction
The harness is now **runnable** (#57) and **debuggable** (#64/#65) end-to-end, and
has a working **evaluation/feedback loop** (#26/#68). The next bar is **capability
breadth and correctness**: make the harness deliver the loop strategies it already
advertises, and close the queue of known correctness/safety gaps. The headline gap is
that the harness is **ReAct-only** — `PlanExecute`, `Ralph`, `SelfVerifying`, and
`HillClimbing` (#58–#61) still return `StrategyNotYetImplemented`, so the README and
`docs/harness-engineering-concepts.md` overstate what runs.

**PlanExecute (#59) is the first strategy being built, and a design pass decomposed it
into five separable concerns** rather than one large change. The build order is
**#70 → #71 → #72 → #73 → #59**, with the lifecycle hook system (**#69, now done**)
providing the `OnPlanCreated`/`OnTaskAdvance` seams the loop will fire. The plan phase
produces a plan artifact (#70); a persisted task-list tool (#71) holds the work; an
accepted plan is parsed into that task list (#72); a `StorageProvider` abstraction
(#73) generalizes the hybrid in-memory+on-disk persistence (spore-core is a
harness-building framework, so storage backends are a first-class seam); and #59
finally wires the two-phase loop together with a **pluggable executor** (ReAct by
default, swappable for a future Ralph-style execute+verify), firing the #69 hooks at
plan-created and task-advance. Two former chain heads are now cleared: #45 (Agent
dyn-compatibility, already on `main`) and #69 (hook system, landed this loop). After
PlanExecute lands this way, the remaining three strategies (#58/#60/#61) follow —
SelfVerifying (#61) becomes a `Stop`-hook configuration now that #69 exists — then the
accepted-debt correctness fixes (#30/#31/#32/#34). Observability backend stays
swappable over OTLP.

## Known Deviations
1. **Only ReAct is executable** — the README and
   `docs/harness-engineering-concepts.md` advertise five loop strategies, but
   only **ReAct** runs. `PlanExecute`, `Ralph`, `SelfVerifying`, and
   `HillClimbing` return `HaltReason::StrategyNotYetImplemented` at
   `rust/crates/spore-core/src/harness.rs`. Tracked: #59 / #58 / #61 / #60
   (`scope: deferred`). #57's scenario suite is intentionally ReAct-only.
2. **Go outbox is not zero-dependency** — closing #50 added
   `go.opentelemetry.io/otel` + `otlptracegrpc` (v1.28.0) as blessed deps to
   `go/spore-core/go.mod` (accepted tradeoff, documented in `go/CONVENTIONS.md`).
   The durable JSONL path stays network-free.

_(Former Deviation: Rust Agent dyn-compatibility / `BoxFut` workaround — **resolved**
in #45 (commit `afe5ff8`); `Agent` is dyn-compatible and `StandardHarness` is
non-generic, matching every other component trait.)_
_(Former Deviation: compaction `tokens_reclaimed = 0` — **resolved** in #57.)_
_(Former Deviation: sandbox Read of a missing in-workspace file → `PathEscape` —
**resolved** in #63.)_
_(Former Deviation: EvalHarness Rust-only Debug-string metric workaround —
**resolved** in #68 via a typed `Span` accessor; all four EvalHarnesses now
structurally identical.)_
_(Former Deviation: observability captured no message content — **resolved** in #64
via opt-in GenAI-convention content capture + an Arize Phoenix viewer.)_

## Next Actions
[3-5 items max. Each references a GH issue # where possible.
This section is updated by /close after every PEE loop.]
1. **PlanExecute build chain (#70 → #71 → #72 → #73 → #59)** — build PlanExecute via
   its decomposed prerequisites, in order. Next up is **#70** (plan phase: generate a
   plan artifact via a one-shot planning turn), unblocked by the `Arc<dyn Agent>` seam
   from the closed #45. Then #71 (task-list tool) → #72 (plan→task-list parse) → #73
   (StorageProvider) → #59 (wire the two-phase loop, firing the #69 hooks at
   plan-created/task-advance). The lifecycle hook system (#69) is **done** and dropped
   from this chain. Rust reference first, then TS/Python/Go parity at each step.
   PlanExecute goes deep first to establish the task-list/storage seams the other
   strategies will reuse.
2. **Remaining loop strategies (#58, #60, #61)** — Ralph (#58), HillClimbing (#60),
   SelfVerifying (#61) follow once PlanExecute lands and the shared seams (pluggable
   executor, task list) exist. SelfVerifying (#61) in particular can now be built as a
   `Stop`-hook configuration on the #69 hook system that just landed.
3. **Correctness/safety debt fixes (#30, #31, #32, #34)** — memory distillation
   PendingReview gate (#30), read-only subagent context sharing (#31), halt on
   mid-session Block 2 hash mismatch (#32), and the dangerous-feature-flag gate for
   `Mode::Yolo`/`SandboxProvider::None` (#34). Contained `scope: deferred` fixes to
   pull in opportunistically alongside the loop-strategy work.
4. **Docs/spec cleanup (#27, #35, #36)** — README/spec clarifications (#27/#35) and
   the E2BSandboxProvider data-residency note (#36); fold in once the loop strategies
   land so the docs stop overstating capability.

_Note: this `/close` loop closed **#69** (lifecycle hook system) `status: complete`.
A full `/implement 69` pass built it across all four languages (Rust ref `ddbd8a4`,
TS `b49c300`, Go `51bf7f4`, Python `6423740` + cross-language parity fix `c593956`),
cross-verified all four suites green, and fast-forward-merged the five commits to
`main`. The issue was a conceptual design doc, so the planning phase reported BLOCKED
on six contract ambiguities (middleware relationship, per-event context schemas,
`HookDecision` shape, `max_stop_blocks` placement, Stop-vs-`ForceAnotherTurn`,
Stop-vs-`Verifier`); all six were pinned by the maintainer before any code and
documented on the issue. Phase-4 verification caught one divergence (Python
`HookError.kind` unset for two categories), fixed in `c593956`. **#69 landed ahead of
its scheduled slot in the PlanExecute chain** (was #70→#71→#72→#73→#69→#59); it was
independent enough to build early, so the chain is now #70→#71→#72→#73→#59 with #69
done. No new issues spawned; no new spec gaps. No re-triage needed — the rest of the
open board (#70–#73 PlanExecute prerequisites, #58/#60/#61 strategies, #30–#36 debt)
is unaffected and its `scope: deferred` labels still hold. Active Direction unchanged
— capability breadth (loop strategies) remains the headline gap, PlanExecute first,
#70 the next concrete step._
