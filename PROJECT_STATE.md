# PROJECT STATE
_Last updated: 2026-06-12 by /close (#143 **complete** — `add_task` returns the assigned task id, all four languages, on `main`; formally closed this loop. Implemented in a prior session, never closed — reconciled now. Earlier this same day #142 — project-scoped durable storage / stable `project_id` — was completed + merged + closed; it was the **linchpin** of the harness-hardening cluster #137–#143, so **#138 is now unblocked**.) ⚠️ **Local `main` is 24 commits ahead of `origin/main`** (origin at the PR #136 merge `0954db1`) — unpushed; standing push gate requires maintainer OK (Deviation #10)._

_**Direction note:** Active direction remains **hardening the composed `12-cordyceps` runtime for small-local-model reliability (cluster #137–#143)**. The cluster is now mostly done: **#137 ✅**, **#142 ✅**, **#143 ✅** (all closed). Remaining: **#138** (resume seeding — now unblocked by #142), and the independent parallel gaps **#139/#140/#141** (open, currently unlabeled — triage-labeling was scope-blocked under per-issue `/close`). The refactor (#117–#131) is landed; #131's capstone is integrated but the issue is **still formally open** (`status: queued`, last touched 2026-06-06) pending its own `/close 131`. Parallel-grabbable refactor finishers #121/#122/#127/#128 remain open and off the critical path. Use the `/implement` skill per issue (Rust reference → three parallel language agents → cross-language verifier)._

## Current State
spore-core is a language-agnostic agentic harness runtime with a **complete core
capability surface**, four targets — Rust (reference), TypeScript, Python, Go —
serialized formats byte-identical across all four. Local `main` is **24 commits
ahead of `origin/main`** (unpushed; origin at the PR #136 merge `0954db1`).

**🎯 Active work: harden the composed `12-cordyceps` runtime for small local models
— cluster #137–#143.** Running the capstone composition live on gemma exposed a set
of robustness gaps, each verified in the Rust reference (several observed live):

- **#137 — ReAct tool-error-loop breaker ✅ DONE (`status: complete`).** Per-tool
  consecutive-recoverable-error tracking; corrective schema injection at N (default 3);
  stop + `BudgetExhaustedBehavior` resolution with typed `HaltReason::ToolErrorLoop` at 2N
  (budget not burned); stream + observability at both thresholds. Shared fixture
  `tool_error_loop.jsonl`; byte-identical AC2 schema string in all four; Go `*uint32`
  sentinel for `ErrorLoopThreshold` default parity.
- **#142 — project-scoped durable storage / stable `project_id` ✅ DONE THIS LOOP
  (`status: complete`, CLOSED).** New `ProjectId` newtype derives a stable id from
  `sandbox.workspace_root()` (canonicalize-first → `{sanitized_basename}-{8hex}`, reusing
  the existing `WorkspaceId` algorithm; the 8-hex SHA-256 suffix resolves the `/a/b` vs
  `/a_b` slug collision). Threaded through `ToolContext` → tool registry →
  `HarnessConfig`/builder. The durable artifacts (`task_list`, plan, **Ralph checkpoint** —
  moved onto the store) are keyed by `project_id` **only** (not project_id+session) via
  namespace-reuse on the existing session-id axis (the `RunStore` trait was not widened);
  ephemeral session state (conversation, `active_skills`) stays session-keyed so it still
  resets per Ralph window. Active-run lifecycle (new / resume / **complete** via an explicit
  caller API + caller-supplied run tag). The `12-cordyceps` example now wires
  `FileSystemStorageProvider` via `CompositeStorageProvider` under central
  `~/.spore/projects/<project_id>/`. Two shared fixtures (`project_id_derivation.json`,
  `project_durable_survival.json`) replay byte-identically in all four; verifier
  independently recomputed all 7 pinned hashes. Commits Rust `6bcabb4`, TS `a037861`,
  Go `631290f`, Py `5b7804f`. **This makes the task_list survive Ralph window resets AND
  process restarts — the soil the error-grind grew in — and unblocks #138.**
- **#143 — `add_task` returns the assigned id ✅ DONE (`status: complete`, CLOSED this loop).**
  Implemented across all four languages in the prior session (Rust `a1d6053`, Py `b01f2d7`, Go
  `4c4b586`, TS `e508d23`, docs `5e206e1`) and on `main`; formally closed during this reconcile.
  Cuts the malformed-call grind: small models no longer parse/predict ids for
  `blockers`/`update_task`/`complete_task`.
- **#138 — resume must seed the stalled worker + skip re-planning (open, NOW UNBLOCKED).**
  `resume_inner`'s `ContinueWithBudget` arm re-enters and re-runs PLAN; the "skip re-plan if
  task_list non-empty" fix could not fire until #142 made the list survive window resets —
  **#142 is now landed, so #138 is workable.**
- **#141 — compaction window hardcoded `200_000` (open).** `SessionState::new` hardcodes
  `window_limit: 200_000`; `ModelProfile.context_window` exists but is never threaded in, so
  compaction never fires for the 128K/8K local models that need it.
- **#139 — `ReactConfig.output` schemas are decorative (open).** Presence-validated by the
  registry but `ReactConfig::run` never reads it: the schema is never delivered to the model
  nor enforced on the terminal.
- **#140 — `PausedState` drops the leaf's toolset handle (open).** Both resume paths resolve
  tools with the empty global-catalogue fallback, so a node with a per-node toolset resumes
  against the wrong catalogue.

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
open** (run `/close 131` after confirming the success criteria). As of #142 the example
persists durably under `~/.spore/projects/<project_id>/`.

**Skill loading is architect-side (the #101 design constraint):** the live loop builds
each turn's prompt via `StandardCompactionAdapter::assemble` (pass-through of
`session.messages`); the rich `StandardContextManager::assemble` that structurally injects
skills/chunks/merged-memory is **bypassed** pending the deferred **#7** ContextManager
migration (root cause of Deviation #8). So skills/chunks/memory reach the model only as
tool-result messages. #101 works around this with a `SkillCatalog` + `load_skill` tool +
a custom context manager; **#115** tracks baking this into the library.

**Harness core:** all 5 loop strategies run end-to-end in all four languages
(ReAct/PlanExecute/SelfVerifying/Ralph/HillClimbing — none stubbed, all via genuine
recursive `RunStrategy` dispatch as of #124); mid-loop consult primitive (#114, with the
#116 HITL child-consult-resume gap); tool/prompt architecture (#79–82); pluggable
scope-aware persistence (#73/#75/#76/#78/#82) — now with a stable `project_id` durable key
axis (#142); runnable (#57), debuggable (#64/#65), evaluation loop (#26/#68).

**Examples suite — 12 of 13 landed, all four languages each** under
`examples/{rust,typescript,python,go}/`: `01-hello-agent` … `11-multi-agent`,
`12-cordyceps` (#101). Remaining: **#109** (`13-coding-agent`) and **#92**
(observability/Phoenix-OTLP). Parked behind the hardening cluster.

**Parked (not active): correctness/safety debt + docs.** #34 (Yolo/None feature flag),
#31 (SharedSession read-only), #30 (memory PendingReview gate), docs #27/#35/#36 — all
`scope: deferred`.

## Active Direction
**Harden the composed `12-cordyceps` runtime so `Ralph[PlanExecute[ReAct,
SelfVerifying[ReAct]]]` runs reliably on small local models — cluster #137–#143.** With the
**linchpin #142 landed**, the task-survival failure that orphaned the `task_list` on every
Ralph window reset is fixed, and #138 is unblocked. The cluster is now mostly done
(#137 ✅, #142 ✅, #143 ✅ impl). Drive the remainder with `/implement` (Rust reference →
three parallel language agents → cross-language verifier), byte-identical where serialized.

**Work next: #138** (resume seeding) — make `ContinueWithBudget`/consult resume seed the
stalled worker and skip re-running PLAN when the now-surviving task_list is non-empty. Then
the independent, parallel-grabbable gaps: **#141** (thread `ModelProfile.context_window` into
`SessionState.window_limit`), **#139** (deliver + enforce `ReactConfig.output`), **#140**
(carry the pausing leaf's toolset handle through resume).

**Also outstanding (housekeeping):** run **`/close 131`** (confirm capstone success criteria +
reconcile — still formally open); push the 24-commit local `main` backlog (maintainer OK
required — Deviation #10). Refactor finishers **#121**
(`SubagentTool` strategy param), **#122** (`max_steps()`), **#127** (custom-strategy tracer),
**#128** (per-node observability span attrs) remain open, off the critical path.

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
   `no_op()` by default. **#142 (landed) makes this real for the cordyceps Ralph loop**: the
   example now wires `FileSystemStorageProvider` via `CompositeStorageProvider` under
   `~/.spore/projects/<project_id>/`. The library default is still no-op; durable standalone
   use still requires wiring a real `StorageProvider`.
3. **v1 memory keying limitation (#78 Q7), filed as #89** (`scope: deferred`) — `MemoryStore`
   is `SessionId`-keyed; durable cross-session addressing is the v2 feature. No SQL backend yet
   (#77). (Note: #142 added a separate `project_id` durable key axis for the run store, not for
   memory — memory keying is unchanged.)
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
    24 commits ahead of `origin/main`** (origin at the PR #136 merge `0954db1`). Unpushed = the
    cordyceps polish + the #137 series + the 2026-06-11 reconcile `dd984b0` + the **#143 series**
    (`a1d6053`→`5e206e1`) + the **#142 series** (`6bcabb4`→`5b7804f`) + the gitignore hygiene
    `41a9caf`. **Ask before pushing** — an agent-initiated push was denied in a prior session;
    push the backlog with maintainer OK to clear the drift.
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
19. **#142 benign per-language divergences** (`scope: debt`, benign, all documented + verifier-
    confirmed) — (a) TS `HarnessConfig.projectId` is **optional** (default resolved in the
    `StandardHarness` ctor) vs Rust's required field, avoiding churn across ~29 config literals;
    (b) Python `Path.resolve()` does **not** case-fold on macOS (unlike Rust `fs::canonicalize`), so
    the macOS-gated test asserts stdlib behavior (distinct-but-deterministic ids resolved by the hash
    suffix); `ProjectId`/`WorkspaceId` are `NewType` aliases ⇒ derivation is module functions, not
    methods; (c) Go `ProjectID` lives in the `storage` package and is projected onto the `SessionID`
    axis at the package boundary (storage→sporecore import cycle), `NewStandardHarness` does **not**
    auto-derive the namespace (the builder `.ProjectID(...)`/example does; empty namespace falls back
    to session id), and the Ralph progress/feature-list key literals are defined in both packages and
    pinned equal by `TestRalphKeyLiteralsAgreeAcrossPackages`. All wire/behavior-identical.

_(Former Deviations — HillClimbing/SelfVerifying/Ralph-git-log/MemoryTool/storage-scope/sandbox-
path/extras-mirror/Rust-dyn/compaction-tokens/observability-content stubs — all resolved in prior
loops.)_

## Next Actions
1. **#138 — resume seeding (now unblocked by #142, work this next).** Make
   `ContinueWithBudget`/consult resume seed the stalled worker and **skip re-running PLAN** when
   the (now-surviving) task_list is non-empty. The #142 durable key axis makes the list visible
   across Ralph windows, so the AC can finally fire. `/implement`.
2. **#141 + #139 + #140 — the parallel hardening gaps (grabbable now, no cross-deps).**
   #141 (thread `ModelProfile.context_window` into `SessionState.window_limit` so compaction fires
   for small models), #139 (deliver + enforce `ReactConfig.output` schemas), #140 (carry the
   pausing leaf's toolset handle through `PausedState`/resume). Each via `/implement`.
3. **Housekeeping (cheap, do soon).** Run **`/close 131`** (confirm the capstone success criteria
   + reconcile — still formally open). Apply `status: queued` to the now-unlabeled cluster issues
   #138/#139/#140/#141 (triage-labeling was scope-blocked under per-issue `/close`). Reconciliation
   only; no code.
4. **Push the 24-commit local `main` backlog** (maintainer OK required — Deviation #10) to clear
   the `origin/main` drift.
5. **Refactor finishers (off critical path) + parked work.** #121/#122/#127/#128 whenever
   convenient; then the parked examples #109/#92, #115/#116, and correctness/safety #34→#31→#30 +
   docs — on an explicit maintainer call.
