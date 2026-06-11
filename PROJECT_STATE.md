# PROJECT STATE
_Last updated: 2026-06-11 by /close (#137 **complete** — ReAct tool-error-loop breaker, all four languages, cross-language verifier PASS, fixture-backed). **Phase shift:** the Composable Execution refactor (#117–#131) has landed its full critical path and the `12-cordyceps` capstone is integrated (PR #136 merged). Running that composition live on small local models (gemma) surfaced a **harness-hardening cluster #137–#143** — all "verified in Rust"/"observed live" robustness gaps — which is now the active track. #137 is the first of that cluster done. ⚠️ **Local `main` is ahead of `origin/main` by 13 commits** (the cordyceps polish + the #137 series) — unpushed; the standing push gate requires maintainer OK (Deviation #10)._

_**Direction note:** Active direction is **hardening the composed cordyceps runtime for small-local-model reliability (cluster #137–#143)**. The linchpin is **#142** (project-scoped durable storage / stable `project_id`): Ralph mints a fresh `SessionId` per window, so the `task_list` is orphaned on every reset and the tree re-plans from scratch — **#138 depends on #142**, and the same task-survival failure feeds the error-grind loop that #137/#143 attack. The refactor (#117–#131) is landed; #131's capstone is integrated but the issue is **still formally open** (`status: queued`, last touched 2026-06-06) pending its own `/close`. Parallel-grabbable refactor finishers #121/#122/#127/#128 remain open and off the critical path. Use the `/implement` skill per issue (Rust reference + three parallel language agents + cross-language verifier)._

## Current State
spore-core is a language-agnostic agentic harness runtime with a **complete core
capability surface**, four targets — Rust (reference), TypeScript, Python, Go —
serialized formats byte-identical across all four. Local `main` is **13 commits
ahead of `origin/main`** (unpushed cordyceps polish + the #137 series; origin is at
the PR #136 merge `0954db1`).

**🎯 Active work: harden the composed `12-cordyceps` runtime for small local models
— cluster #137–#143.** Running the capstone composition live on gemma exposed a set
of robustness gaps, each verified in the Rust reference (and several observed live):

- **#137 — ReAct tool-error-loop breaker ✅ DONE this loop (`status: complete`).**
  The ReAct turn loop now tracks consecutive recoverable tool errors per tool name
  (any success resets); at N (default 3) identical-arg failures it injects ONE
  corrective message rendering the violated parameter schema; at 2N it stops and
  resolves via the node's `BudgetExhaustedBehavior` with a typed
  `HaltReason::ToolErrorLoop` (never `BudgetExceeded`, budget not burned); stream +
  observability events at both thresholds. Shared fixture
  `fixtures/model_responses/harness/tool_error_loop.jsonl`; all four languages pin the
  byte-identical AC2 schema string (key-sorted to match Rust's `serde_json` BTreeMap
  ordering; Go uses `SetEscapeHTML(false)`). Go also moved `ErrorLoopThreshold` to a
  `*uint32` sentinel (nil→3, explicit-0→disabled) for default parity. Commits Rust
  `832b1a2`/`5f47767`, TS `763d4cd`/`6c9df8a`, Py `10d02e5`/`dfa05c4`, Go
  `d5f6ae2`/`96ab7ff`.
- **#142 — project-scoped durable storage / stable `project_id` (linchpin, open).**
  `task_list` persists into `RunStore` keyed by `ctx.session_id()`, but Ralph mints a
  fresh `SessionId::generate()` per window, so each reset reads an empty list and
  re-plans; the prior window's list is orphaned. **#138 depends on this.**
- **#143 — `add_task` should return the assigned id (open).** The tool discards the
  `u32` id and returns the whole list, so small models must parse/predict ids to use
  `blockers`/`update_task`/`complete_task` — they get it wrong, feeding the error grind.
- **#141 — compaction window hardcoded `200_000` (open).** `SessionState::new`
  hardcodes `window_limit: 200_000`; a 128K/8K local model overruns real context long
  before the 0.80 trigger, so compaction never fires for the models that need it.
  `ModelProfile.context_window` exists but is never threaded in.
- **#139 — `ReactConfig.output` schemas are decorative (open).** Presence-validated
  by the registry but `ReactConfig::run` never reads it: the schema is never delivered
  to the model nor enforced on the terminal.
- **#140 — `PausedState` drops the leaf's toolset handle (open).** Both resume paths
  resolve tools with the empty global-catalogue fallback, so a node with a per-node
  toolset resumes against the wrong catalogue.
- **#138 — resume must seed the stalled worker + skip re-planning (open, depends
  #142).** `resume_inner`'s `ContinueWithBudget` arm re-enters and re-runs the PLAN
  phase; the "skip re-plan if task_list non-empty" fix can't fire until #142 makes the
  list survive Ralph window resets.

**Landed: Composable Execution refactor #117–#131 (all `status: complete` except the
still-open #131 capstone).** Delivered across all four languages, byte-identical where
serialized; per-issue detail lives on the GitHub issues. Summary of what the runtime
now does as a result:
- **#117 `BudgetPolicy` + `BudgetExhaustedBehavior`** value types (`Unlimited`/
  `TotalSteps`/`PerLoop`/`PerAttempt`; `Continue{max_continues,on_exhausted}`/`Escalate`/
  `Fail`), layered over the `BudgetLimits` global backstop.
- **#118 `Task.blockers` DAG schema** + `add_task` `blockers` param with validate-
  before-mutate (`self_block`/`unknown_id`/`cycle` → recoverable `invalid_blockers`).
- **#119 + #120 the strategy seam:** `LoopStrategy` is a closed recursive serde enum of
  config newtypes (`react`/`plan_execute`/`self_verifying`/`ralph`/`hill_climbing`),
  each owning its loop via the `RunStrategy` trait (one-line enum delegation, no central
  match); `StrategyRef::{BuiltIn,Custom}`; an `ExecutionRegistry` resolves `*Ref` handles
  + custom keys (`StrategyNotFound`/`UnresolvedHandle`).
- **#123 + #124 genuine composition:** typed runtime `StrategyOutcome`
  (`Complete`/`BudgetExhausted`/`Failed`, never serialized) + shared `ExecutionContext`;
  all five strategies genuinely recurse via `self.inner.run(cx)`; the monolithic
  `run_self_verifying`/`run_ralph`/`run_hill_climbing` loops and the legacy collaborator
  fields are **deleted** — all collaborators resolve through the registry.
- **#125 + #126 enforcement + scheduling:** per-node budget `charge()` with isolated,
  parent-inspectable `BudgetExhausted` (no auto-cascade; ReAct leaf propagates); ready-
  set DAG task walk, two-tier context (transitive-blocker outputs + bounded N=20 step
  ledger with harness-observed `files_touched`), failure cascade to transitive
  dependents only (`TasksBlockedByFailure`).
- **#130 + #129 HITL + Continue resume:** `EscalationMode{SurfaceToHuman,Autonomous}`
  consumed at every Escalate site → `HumanRequest::BudgetExhausted` +
  `EscalationAction{ContinueWithBudget,Skip,Fail}`; serialized `behavior` field on all
  five configs makes in-process `Continue` genuinely loop; resume seeds `continues_used`
  off the request payload; shared `PausedState::{serialize,load}_checkpoint`.

**`12-cordyceps` capstone (#101 + #131, all four languages):** a fully-autonomous
task-completion agent — ReAct orchestrator + `task_list` decomposes a per-module Rust
audit; an Isolated `analysis_worker` deep-dives one module and loads an `audit` skill at
runtime; the #114 consult ladder escalates a stuck worker to a sibling then a human;
heterogeneous models (local gemma + cloud advisor); within-run memory; REPL approval →
`gh issue create`. #131 re-expresses it as `Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]`.
Core integration merged via **PR #136**; #131 is **functionally landed but still formally
open** (run `/close 131` after confirming the success criteria).

**Skill loading is architect-side (the #101 design constraint):** the live loop builds
each turn's prompt via `StandardCompactionAdapter::assemble` (pass-through of
`session.messages`); the rich `StandardContextManager::assemble` that structurally injects
skills/chunks/merged-memory is **bypassed** pending the deferred **#7** ContextManager
migration (root cause of Deviation #8). So skills/chunks/memory reach the model only as
tool-result messages. #101 works around this with a `SkillCatalog` + `load_skill` tool +
a custom context manager; **#115** tracks baking this into the library.

**Harness core (done before the examples push):** all 5 loop strategies run end-to-end
in all four languages (ReAct/PlanExecute/SelfVerifying/Ralph/HillClimbing — none stubbed,
all via genuine recursive `RunStrategy` dispatch as of #124); mid-loop consult primitive
(#114, with the #116 HITL child-consult-resume gap); tool/prompt architecture (#79–82);
pluggable scope-aware persistence (#73/#75/#76/#78/#82); runnable (#57), debuggable
(#64/#65), evaluation loop (#26/#68).

**Examples suite — 12 of 13 landed, all four languages each** under
`examples/{rust,typescript,python,go}/`: `01-hello-agent` … `11-multi-agent`,
`12-cordyceps` (#101). Remaining: **#109** (`13-coding-agent`) and **#92**
(observability/Phoenix-OTLP). Parked behind the hardening cluster.

**Parked (not active): correctness/safety debt + docs.** #34 (Yolo/None feature flag),
#31 (SharedSession read-only), #30 (memory PendingReview gate), docs #27/#35/#36 — all
`scope: deferred`.

## Active Direction
**Harden the composed `12-cordyceps` runtime so `Ralph[PlanExecute[ReAct,
SelfVerifying[ReAct]]]` runs reliably on small local models — cluster #137–#143.** These
are the live-run robustness follow-ups the #131 capstone surfaced, each verified in the
Rust reference. Drive them with `/implement` (Rust reference → three parallel language
agents → cross-language verifier), byte-identical where serialized.

The **linchpin is #142** (project-scoped durable storage / stable `project_id`): the
fresh-`SessionId`-per-Ralph-window bug orphans the `task_list` and forces re-planning, which
is also the soil the error-grind (#137 done, #143) grows in. **#138 is blocked on #142.**
The other gaps are independent and parallel-grabbable: **#143** (return `add_task` id — cheap,
directly cuts error-grind), **#141** (model-configurable compaction window), **#139** (deliver +
enforce `ReactConfig.output`), **#140** (carry the toolset handle through resume).

**Also outstanding:** formally `/close 131` (confirm the capstone success criteria), and
push the 13-commit local backlog (maintainer OK required — Deviation #10). Refactor finishers
**#121** (`SubagentTool` strategy param), **#122** (`max_steps()`), **#127** (custom-strategy
tracer), **#128** (per-node observability span attrs) remain open, off the critical path.

**Parked behind the hardening cluster:** examples #109/#92 + `web_search` #108/#110; harness
gaps #115 (skill loading) and #116 (HITL child-consult resume — overlaps #130's resume seam,
may be cheaper to fold in); correctness/safety #34 → #31 → #30 and docs #27/#35/#36; larger
features #113 (spore-lsp), #107 (PromptEngineeringAgent), #106 (MicroVMSandboxProvider), the
protocol track #83–87, storage follow-ups #77/#88/#89. **#7** (ContextManager migration) would
live-wire the rich `assemble` (proper home for #115's injection + the #32 cache halts).

## Known Deviations
1. **Go outbox is not zero-dependency** — closing #50 added `go.opentelemetry.io/otel` +
   `otlptracegrpc` (v1.28.0) as blessed deps (documented in `go/CONVENTIONS.md`). The durable
   JSONL path stays network-free.
2. **`task_list` / `todo_write` default persistence is no-op, not a file** (`scope: debt`,
   minor) — #75 retired the sandbox path; standalone tools persist via `RunStore`, which is
   `no_op()` by default. Durable standalone use requires wiring a real `StorageProvider`. **#142
   is the issue that makes this real for the cordyceps Ralph loop.**
3. **v1 memory keying limitation (#78 Q7), filed as #89** (`scope: deferred`) — `MemoryStore`
   is `SessionId`-keyed; durable cross-session addressing is the v2 feature. No SQL backend yet
   (#77).
4. **Go-specific divergences** (`scope: debt`, minor, documented on the issues) — local `Mode`
   newtype; 3-state `TerminalOutcome`; type-aliased `StandardTool`; explicit `abort` `reason`;
   self-contained `promptassembly` builder; opaque `ToolContext.MemoryStore`; exported
   `storage.MergeMemories`; config-struct (not builder) setters; `RoleEvaluatorChunk` constant;
   consumer-side `MetricEvaluator`/`ContextError` seams to avoid import cycles; custom context
   manager embeds `*StandardCompactionAdapter`; `RunStrategy.Run` takes `ctx context.Context`
   first; Go keeps `Agent`/`ToolRegistry` as struct fields folded into the registry + an
   `IsEmpty()` validate gate. #137 adds the `ErrorLoopThreshold` `*uint32` sentinel + sibling
   `effectiveErrorLoopThreshold()` (the house idiom). All wire/behavior-identical.
5. **Test-placement divergences (#78/#82)** (benign) — registry-seam / catalogue tests live in
   language-idiomatic spots. Behavior identical.
6. **#79 cross-language divergences — both verified benign.** (a) narrowed `composed_prompt`
   stub in TS/Go; (b) Block-1 hash not byte-identical (Rust SipHash vs FNV-1a) — the intentional
   #24 decision; #79 fixtures assert no hash values.
7. **`Custom` condition is invisible in fixtures by design** (#79) — serializes to null/absent.
8. **The live harness loop does not call the rich `assemble`** (`scope: deferred`, intentional,
   depends on #7) — prompts are built via `StandardCompactionAdapter::assemble` (pass-through of
   `session.messages`); the rich `StandardContextManager::assemble` (skill/chunk/memory injection +
   Block-1/2 `CacheHashMismatch` halts) is bypassed. So skills/chunks/memory reach the model only
   as tool-result messages, and the #32 cache halts can't fire end-to-end. Live-wiring is #7's job.
9. **#114 HITL has no child-consult resume, filed as #116** (`status: queued`) — `EscalateToHuman`
   consult overflow surfaces `WaitingForHuman` at the parent with the worker's paused consult in
   `child_state`, but `resume`'s `child_state` branch is a **no-op** in all four cores. #101's three
   escalation choices are implemented host-side. **Overlaps #140 (toolset handle on resume) and the
   #138 resume-seeding work.**
10. **Local `main` push hygiene (standing reminder).** ⚠️ **Currently drifted: local `main` is
    13 commits ahead of `origin/main`** (origin at the PR #136 merge `0954db1`; unpushed = the
    cordyceps polish `f0d1d14`/`8a05076`/`be79cd8`/`faa7aae`/`b0065bf` + the #137 series
    `832b1a2`→`5f47767`). **Ask before pushing** — an agent-initiated push was denied in a prior
    session; push the backlog with maintainer OK to clear the drift.
11. **Rust-only `12-cordyceps` polish + a Rust-only core addition** (`scope: debt`, not yet
    mirrored) — `8bb7734` adds `SubagentTool::with_stream` to the core harness (optional child
    stream sink); `d65ae64` builds on it in the Rust example. **TS/Python/Go have neither the core
    seam nor the example polish.** Decide whether to mirror (file an issue) or keep as a Rust-ahead
    experiment.
12. **#119's `RunStrategy` is a hand-rolled `BoxFut`, not `#[trait_variant::make(Send)]`**
    (`scope: debt`, Rust-only, benign) — converted in #120 so the `custom: HashMap<String, Arc<dyn
    RunStrategy>>` map can exist. No-op on the wire. (The legacy-collaborator-field removal formerly
    tracked here is **RESOLVED by #124**.)
13. **#123 Go `SpanStack` holds `string`, not a typed `SpanId`** (`scope: debt`, intentional) —
    Go's `observability` package imports `sporecore`, so the reverse would be an import cycle; the
    scaffold types must live in `sporecore`. Safe — `SpanStack` is runtime-only, never serialized.
    Sub-note: `charge`'s fallible-result shape is idiomatic per language (Rust `Result`, TS tagged
    union, Go `*BudgetExhausted` nil-ok, Python raises); semantically identical.
14. **#125 review follow-ups — ADDRESSED** (hardening pass `ca0165b`/`ca89df6`/`c96ceed`/`9a9fb12`,
    all four, verifier PASS). Confirmed the `BudgetExhausted`/`partial_output` path is reachable
    end-to-end (no impl gap); made the leaf-cap test discriminating; added bounded-leaf F5/F6
    coverage; fixed a stale Rust doc; made the Continue arms explicit.
15. **#126 `PlanArtifact` type not formally deprecated, only the bridge function** (`scope: debt`,
    minor) — the bridge `plan_artifact_to_task_list` is marked deprecated in all four; the
    `PlanArtifact` **type** is not (still the live `OnPlanCreated` payload — attributing it would
    break the `-D warnings` gate). Documented in prose.
16. **#130 Go fixture comparison uses `jsonEqual`, not byte-equal** (`scope: debt`, benign,
    Go-only) — Go's `encoding/json` emits `0` vs serde's `0.0` for the whole-number `cost_usd`
    float; the established value-normalizing helper is used, same as the consult/escalation replay
    tests. Field order/structure still match the fixture exactly.
17. **#130 default `escalation_mode` is applied at different layers per language** (`scope: debt`,
    benign) — TS defaults the raw config to `autonomous` with the builder setting `surfaceToHuman`;
    Python/Go default the config itself to `surface_to_human`. Each preserves its own pre-#130
    legacy default. A maintainer may wish to harmonize separately.
18. **#129 benign per-language divergences** (`scope: debt`, benign) — (a) Python AC4 asserts
    context preservation by membership rather than message-count growth (Python's
    `NoopContextManager` doesn't append the resumed `FinalResponse`); (b) Go uses `jsonEqual` for
    the `cost_usd` float fixtures (as #16) and an idiomatic `ResumedBudgetContext` constructor name.
    No wire/behavior impact.

_(Former Deviations — HillClimbing/SelfVerifying/Ralph-git-log/MemoryTool/storage-scope/sandbox-
path/extras-mirror/Rust-dyn/compaction-tokens/observability-content stubs — all resolved in prior
loops.)_

## Next Actions
1. **#142 — project-scoped durable storage / stable `project_id` (linchpin, work this
   next).** Give the cordyceps Ralph loop a stable identity so the `task_list` survives window
   resets and process restarts instead of being orphaned under a fresh `SessionId` each window.
   Unblocks #138 and removes the re-planning waste that feeds the error grind. `/implement`.
2. **#143 + #141 + #139 + #140 — the parallel hardening gaps (grabbable now, no cross-deps).**
   #143 (return the `add_task` id — cheap, directly cuts the malformed-call grind #137 broke),
   #141 (thread `ModelProfile.context_window` into `SessionState.window_limit` so compaction fires
   for small models), #139 (deliver + enforce `ReactConfig.output` schemas), #140 (carry the
   pausing leaf's toolset handle through `PausedState`/resume). Each via `/implement`.
3. **#138 — resume seeding (after #142).** Make `ContinueWithBudget`/consult resume seed the
   stalled worker and skip re-running PLAN when the (now-surviving) task_list is non-empty.
4. **Close out #131 and push the backlog.** Run `/close 131` to confirm the capstone success
   criteria and reconcile it; push the 13-commit local `main` backlog with maintainer OK
   (Deviation #10).
5. **Refactor finishers (off critical path) + parked work.** #121/#122/#127/#128 whenever
   convenient; then the parked examples #109/#92, #115/#116, and correctness/safety #34→#31→#30 +
   docs — on an explicit maintainer call.
