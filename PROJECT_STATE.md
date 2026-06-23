# PROJECT STATE

_2026-06-23 — **Consumer-friction plan ADOPTED** (`docs/consumer-friction-plan.md`, v2.0; signed off by spore-core + cordyceps + looper). It reframes the cross-language defaults as tuned for fixture-replay, not consumers, and sequences ~30 fixes (SC-#/ARK-#/LOC-#) to make spore-core ergonomic for the cordyceps hill-climber and looper coding-agent. **Phase 0 complete:** the #151 SelfVerifying eval-phase reviewer slice (`eval_agent`/`eval_toolset` + read-only eval + eval-phase middleware drop) was test-gated (1255 pass / 0 fail) and committed as **`d14341f`** — the agreed pin all subsequent PRs build on. Reconciliations: **D1** canonical middleware surface = the rich `middleware.rs` chain (verified: the only fixtured surface across all four languages) → wire the loop to it and delete the `harness.rs:5184` stub (collapses SC-9/SC-11/Q5); **D2** the `enforce_output_schemas` default lives **only** in `HarnessBuilder::new` (`harness.rs:6429`) — there is no `impl Default for HarnessConfig`, and `hooks.rs:1763` is a test-only builder → fix is a single `Default` impl both derive from. Decisions: Q1 no fixture re-baseline (friendliness via additive setters + presets SC-8), Q3 auto-synthesize, Q4 `EscalationMode::AutoContinue`, Q7 structural `ContextSources` (#115). **Next: bundled SC-1 (auto-synthesize structured-slot schema) + SC-30 (auto-derive read-only eval catalogue) PR against `d14341f`.**_

_Last updated: 2026-06-19 by /close (#149 **complete + CLOSED** — Ollama `num_ctx` ported from the Rust reference `3273386` to TS `ff5358a`, Python `5c9b613`, Go `4a544a7`; opt-in interface field (each language's `keep_alive` idiom, not the literal `with_num_ctx`), threaded into `options.num_ctx`, **omitted when unset** (existing fixtures replay byte-identical), `num_ctx` serialized **first** in `options` to match Rust serde field order; wire-level HTTP-mock test + unit tests in each; verifier PASS across all four. The three port commits are on **local `main` only — 3 ahead of `origin/main` (`64938ee`), NOT pushed**.)_

_**Direction note:** The project has shifted from "hardening cluster done, maintainer call" into an active **cross-language parity catch-up** phase. Rust raced ahead on `main` with a batch of Ollama + sandbox + bugfix work; TS/Python/Go are now catching up issue-by-issue via `/implement` (land Rust first → three parallel language agents → cross-language verifier). Open parity backlog: **#150** (recoverable sandbox violations — larger, breaking default change), **#148** (Ollama thinking/reasoning models), **#146** (web_fetch start_byte, Python/Go), **#147** (SelfVerifying evaluator budget, TS/Python), and **#144** (PlanExecute budget-resume — ports appear landed, verify + `/close 144`). #149 (num_ctx) and #145 (SSRF seam) are done/closed. This is the current north star until the backlog drains; refactor close-out (#131 + finishers) and parked examples (#109/#92) remain behind it. The hardening cluster #137–#143 + #139/#141 remain fully closed._

## Current State
spore-core is a language-agnostic agentic harness runtime with a **complete core
capability surface**, four targets — Rust (reference), TypeScript, Python, Go —
serialized formats byte-identical across all four. Local `main` is **3 commits ahead of
`origin/main`** (`origin/main` at `64938ee`); the three #149 `num_ctx` ports
(`ff5358a`/`5c9b613`/`4a544a7`) are committed locally but **not yet pushed** — ask the maintainer
before pushing (Deviation #10).

**Active phase — cross-language parity catch-up (#144–#150).** Since the 2026-06-14 reconcile, Rust
raced ahead on `main` with a wave of Ollama + sandbox + bugfix work, and TS/Python/Go are catching up
issue-by-issue via `/implement` (Rust reference → three parallel language agents → cross-language
verifier). Backlog status:
- **#149 — Ollama `num_ctx` ✅ DONE THIS LOOP (`status: complete`, CLOSED).** Opt-in interface field
  (`numCtx` ctor option / `num_ctx` kwarg / `SetNumCtx` setter — each language's `keep_alive` idiom, not
  the literal Rust `with_num_ctx`), threaded into `options.num_ctx`, **omitted when unset** (bare requests
  + existing fixtures stay byte-identical), `num_ctx` serialized **first** in `options` to match Rust serde
  field order (TS/Python carry explicit ordering tests); wire-level HTTP-mock test in each. Verifier PASS.
  Rust ref `3273386`; ports TS `ff5358a`, Python `5c9b613`, Go `4a544a7`.
- **#145 — SSRF URL-validation seam ✅ DONE (CLOSED).** Ported TS `986a15e`, Python `ca3d8a9`, Go
  `498c87f` (see memory `ssrf-seam-url-bracket-parity-gotchas`).
- **#144 — PlanExecute budget-resume fix → ports appear LANDED, issue still OPEN.** TS `2401fee`, Python
  `9b1d310`, Go `b56e852` + Rust docs `a25e6bf` are on `main`; verify the AC and run `/close 144`.
- **#150 — recoverable sandbox violations via `SandboxViolationPolicy` (port OWED).** Rust landed
  (`64938ee`); TS/Python/Go owed — the larger port, a **breaking default change** (path escapes / blocked
  commands become recoverable feedback). See issue #150 + the handoff for the typed-violation-to-harness
  shape (keep the loud compile error on exhaustive matches).
- **#148 — Ollama thinking/reasoning models (port OWED).** Rust landed (`a9c856b`); TS/Python/Go owed.
- **#146 — char-boundary-safe `web_fetch` `start_byte` slicing (Python/Go OWED).**
- **#147 — SelfVerifying must charge evaluator turns against budget (TS/Python OWED).**

Also Rust-ahead but **not yet filed as parity issues**: Ollama streaming `read_timeout` fix (`56361b2`)
and Ollama think-effort levels (`c486331`) — confirm whether these need their own TS/Py/Go parity issues
or fold into #148.

**🎯 The `12-cordyceps` hardening cluster #137–#143 is COMPLETE** (all five closed), and
the adjacent **#139** (output-schema enforcement) and **#141** (configurable compaction
window) are now done too — **every robustness gap is closed.** Running the capstone
composition live on gemma exposed these gaps, each verified in the Rust reference (several
observed live); all are now landed across all four languages:

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
- **#138 — resume seeds the stalled worker + skips re-planning ✅ DONE THIS LOOP
  (`status: complete`, CLOSED).** Three behavioral fixes, all four languages: **(AC1)** on
  PlanExecute re-entry with a persisted non-empty `task_list` (the #142 `project_id` durable
  axis lets it survive the Ralph window reset), skip the plan phase and go straight to the
  ready-set walk (`reconcile_completed_tasks` dedups completed tasks) instead of re-running PLAN
  unconditionally; **(AC2)** a budget-resume of an execute-phase exhaustion seeds the stalled
  worker by generalizing the #131 `consult_resume` seed to a phase-agnostic resume seed (was
  `None`), so the worker resumes its audit instead of re-exploring (the live gemma4:31b-cloud
  failure); **(AC3)** a plan-phase exhaustion resumes the planner's own session rather than
  cloning the paused worker's session into the planner's context. New shared fixture
  `cordyceps_budget_resume.jsonl` + updated `cordyceps_budget_exhausted.json` paused-state replay
  in all four. Tests verified green this loop (Rust 9 / Go 2 named / Py 11 / TS 29); the Go
  skip-replan test wires a **real in-memory `RunStore`**, not the no-op default, so the
  store-dependent guard is genuinely exercised. Commits Rust `99a16be`, Py `9133762`, Go
  `4827924`, TS `5ec555a`.
- **#141 — compaction window now model-configurable ✅ DONE THIS LOOP
  (`status: complete`, CLOSED).** `SessionState::new` no longer hardcodes `window_limit: 200_000`.
  `CompactionConfig` gained a `context_length: Option<u32>` field (serialized **absent** when unset
  — `skip_serializing_if`/`omitempty`/optional/`None`-excluded — so existing serialized configs stay
  byte-identical), and `StandardContextManager` gained a `resolve_context_length()` resolver with the
  fallback chain **config (`> 0`) → model `context_window` (`> 0`) → `DEFAULT_CONTEXT_LENGTH = 8000`**,
  applied through a manager-owned `seed_session()` helper (the real production seam — the harness
  round-trips the rich-state blob via `extras`, so callers/the manager seed it, not the loop). Trigger
  math (`should_compact`) is unchanged — once seeded with the resolved window, `threshold × window_limit`
  respects config automatically. **Maintainer-pinned spec changes vs. the original AC:** field renamed
  `window_limit` → `context_length`; explicit `0`/null/nil **falls through** (a zero window would
  silently disable compaction — the exact bug); **no clamping** of an oversized configured value; and
  the unknown-context fallback is **8000, not 200_000** (the `SessionState` constructor default dropped
  to 8k — conservative, fixes the gemma-8k / 128K overrun rather than preserving the dangerous default;
  provider `context_window` defaults like Claude/OpenAI 200_000 are untouched). New shared fixture
  `fixtures/compaction_window/cases.json` (5 `trigger_cases` + 6 `resolver_cases`) replays byte-
  identically in all four; existing `compaction_loop`/`compaction_verifier` fixtures untouched. The
  one verifier-caught divergence — TS Zod `.positive()` would have rejected an explicit `0` the other
  three accept — was fixed (`66aed39`, `.nonnegative()`). Verifier PASS. Commits Rust `1e3cf4c`, Go
  `67c8d39`, Py `6be78a2`, TS `32ed008` + `66aed39`.
- **#139 — `ReactConfig.output` schemas delivered + enforced ✅ DONE THIS LOOP
  (`status: complete`, CLOSED).** The schema was presence-validated by `ExecutionRegistry` at
  startup but `ReactConfig::run` never read it at runtime; that gap is closed. **(AC1)** when
  enforcement is on and a leaf has `output` set, the resolved (key-sorted) schema is appended to
  the directive seed AND routed into the model's structured-output channel (Ollama `format`;
  Anthropic/OpenAI no-op). **(AC2)** the terminal `FinalResponse` is validated by a **hand-rolled
  minimal validator** (subset `type`/`required`/`properties`/`enum`); on mismatch the frozen
  validation error is fed back as a user message and the leaf retries up to N extra turns (N =
  `output_schema_max_retries`, default 2; total 1+N), retries counting against budget. **(AC3)**
  after N failed retries with budget remaining → typed `HaltReason::OutputSchemaViolation`, distinct
  from budget exhaustion (budget cap wins precedence if hit first). **(AC4)** global
  `HarnessConfig.enforce_output_schemas`, **default OFF** — a documented migration gate; OFF keeps
  every existing replay fixture byte-for-byte green. Three shared fixtures
  `output_schema_{accept,retry,fail}.jsonl`. Parity-critical determinism: lexicographically-sorted
  property iteration, semantic `integer` check (42.0 passes / 42.5 fails), canonical key-sorted
  `{enum}`/`{value}` rendering — all frozen byte-identical across the four. Verifier PASS. Commits
  Rust `3790997`, TS `f412b95`, Go `0157749`, Py `15fbfc6`.
- **#140 — `PausedState` carries the leaf's toolset handle ✅ DONE THIS LOOP
  (`status: complete`, CLOSED).** `PausedState` + `ChildPausedState` gained an always-serialized,
  serde-default `toolset` field (last field, byte-parity-safe); all 7 leaf pause sites populate it
  and both resume paths thread it into `effective_tool_registry`, so a node with a per-node toolset
  now resumes against its scoped catalogue instead of the empty global fallback (the cordyceps
  Consult repro). Two extra embedded-paused-state fixtures (`harness/consult.json`,
  `harness/escalation_signals.json`) also needed the key. Load-bearing AC2b resume-routing test
  (+ negative control: empty handle → recoverable unknown-tool error) in all four. Commits Rust
  `9998a0c`, Py `d66afe3`, Go `3e177d3`, TS `d8f8123`.

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
**The `12-cordyceps` hardening cluster #137–#143 is COMPLETE** (#137 ✅, #138 ✅, #140 ✅,
#142 ✅, #143 ✅ — all closed), and the adjacent robustness gaps **#139** (output schemas) and
**#141** (compaction window) are **done and closed too.** `Ralph[PlanExecute[ReAct,
SelfVerifying[ReAct]]]` now survives Ralph window resets and process restarts with the `task_list`
durable (#142), resumes the stalled worker instead of re-planning (#138), routes resumed tool calls
through the leaf's scoped catalogue (#140), breaks tool-error grind loops (#137), enforces output
schemas behind a migration gate (#139), and **fires compaction correctly for small/unknown-window
models (#141)**. The composition's small-local-model reliability gaps that motivated the cluster are
fully addressed.

**The active direction is the cross-language parity catch-up backlog (#144–#150).** The maintainer
pushed a Rust-first wave of Ollama + sandbox + bugfix work; the job now is porting each to TS/Python/Go
via `/implement` (Rust reference → three parallel language agents → cross-language verifier),
byte-identical where serialized, until the backlog drains. Order, roughly: **#149 ✅ done → #150**
(recoverable sandbox violations — larger, breaking default change) **and #148** (Ollama thinking models)
for the Ollama/sandbox cluster; then the smaller bugfix ports **#146** (web_fetch start_byte) and **#147**
(SelfVerifying evaluator budget); **#144** ports appear landed — verify + `/close 144`. Behind the parity
backlog the prior maintainer-call candidates still stand: refactor close-out (`/close 131` + finishers
#121/#122/#127/#128), parked examples (#109 `13-coding-agent`, #92 observability) + `web_search`
#108/#110, then larger parked features (#113/#107/#106, protocol track #83–87) and correctness/safety
debt #34→#31→#30 + docs.

**Housekeeping:** local `main` is **3 commits ahead of `origin/main`** (`64938ee`) — the three #149
ports are unpushed; **ask the maintainer before pushing** (Deviation #10). `/close 131` (confirm the
capstone success criteria; no code) remains an outstanding cheap reconcile-only step.

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
   escalation choices are implemented host-side. **#140 (toolset handle on resume) is now landed —
   `ChildPausedState` carries the child's toolset and `child_state_from_paused` propagates it, so when
   #116 finally wires the `child_state` resume branch the scoped catalogue is already available.
   **#138 (now landed) generalized the resume seed to be phase-agnostic, so #116 can reuse that
   seam directly when it wires the `child_state` branch.**
10. **Local `main` push hygiene (standing reminder).** ⚠️ **3 AHEAD (2026-06-19, this loop):** local
    `main` is 3 commits ahead of `origin/main` (`64938ee`) — the three #149 `num_ctx` ports
    (`ff5358a`/`5c9b613`/`4a544a7`) are committed locally but **not pushed**. The Rust Ollama/sandbox wave
    up to `64938ee` IS on `origin`. The standing reminder holds: **ask before pushing** — an agent-initiated
    push was denied in an earlier session, so confirm maintainer OK before clearing this drift.
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
1. **Port #150 — recoverable sandbox violations via `SandboxViolationPolicy` (TS/Python/Go).** Rust
   landed (`64938ee`); the larger of the Ollama/sandbox parity ports and a **breaking default change**
   (path escapes / blocked commands become recoverable feedback). Drive via `/implement 150`; keep the
   typed-violation-to-harness shape and the loud compile error on exhaustive matches (see issue #150 + the
   handoff). Current top of the parity backlog.
2. **Port #148 — Ollama thinking/reasoning models (TS/Python/Go).** Rust landed (`a9c856b`); `/implement 148`.
   While here, confirm whether the Rust-ahead Ollama `read_timeout` fix (`56361b2`) and think-effort levels
   (`c486331`) need their own parity issues or fold into #148.
3. **Port the smaller bugfix-parity issues.** #146 (char-boundary-safe `web_fetch` `start_byte`, Python/Go)
   and #147 (SelfVerifying charge evaluator turns against budget, TS/Python). Both via `/implement`.
4. **Verify + `/close 144`, then push.** PlanExecute budget-resume ports appear landed
   (`2401fee`/`9b1d310`/`b56e852` + Rust docs `a25e6bf`); confirm the AC then close. Also push the 3
   unpushed #149 commits once the maintainer OKs (Deviation #10).
5. **Behind the parity backlog (maintainer call):** `/close 131` + refactor finishers #121/#122/#127/#128;
   parked examples #109/#92 + `web_search` #108/#110; larger features #113/#107/#106, protocol #83–87;
   debt #34→#31→#30 + docs. **#7** (ContextManager migration) would live-wire the rich `assemble`.

**Note:** the hardening cluster (#137–#143) + #139/#141 remain fully closed; the active work is the
**cross-language parity catch-up backlog (#144–#150)** — Rust-first features/bugfixes being ported to the
other three languages issue-by-issue.
