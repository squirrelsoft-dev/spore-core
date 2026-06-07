# PROJECT STATE
_Last updated: 2026-06-07 by /close (REOPENED #124 — ACs were wrong) — **The Composable Execution refactor (loop strategy / budget / task graph) is the top priority.** A PRD (`spore-core-composable-execution-prd.md`) was broken into **15 tracer-bullet issues #117–#131** (label `loop-strategy-refactor`) via `/to-issues`. The goal is to land #117→#131 to completion: make `LoopStrategy` a composable recursive enum where each variant owns its run loop via a `RunStrategy` trait (no central dispatch match), add a compositional per-node `BudgetPolicy`/`BudgetExhaustedBehavior` budget layer with typed `StrategyOutcome`, and make the task list an explicit blocker DAG with a ready-set walk + failure cascade. Capstone #131 re-expresses the `12-cordyceps` audit as `Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]`. **Five bricks landed (#117, #118, #119, #120, #123) all `status: complete`, all four languages. #124 is REOPENED as PARTIAL** — it shipped genuine recursion for only **2 of 5** strategies (ReAct leaf + PlanExecute); SelfVerifying/Ralph/HillClimbing still delegate to monolithic `StandardHarness::run_*` methods that hardcode a ReAct worker and ignore their `inner` child. The original ACs let this pass (parity-via-reuse); the real goal — implement the composable strategies AND delete the monolithic runs — is ~60% remaining. The examples suite (prior direction) is parked at 12 of 13 — #109/#92 remain but yield priority to the refactor. ✅ `origin/main` is level with local `main` (#124's landed commits pushed; the reconcile commit is local-only)._

_**Direction note:** Active direction is the **loop-strategy refactor (#117–#131)**. Critical path: 117 → 119 → 120 → 123 → 124 → {125, 126} → 130 → 131; **#117, #118, #119, #120, #123 are done; #124 is REOPENED (partial)** and is again the top critical-path item — finish converting SelfVerifying/Ralph/HillClimbing to genuine `self.inner.run(cx)` recursion, **delete** `StandardHarness::run_self_verifying`/`run_ralph`/`run_hill_climbing`, and (folded back in from #125) **remove the four legacy collaborator fields** — they're load-bearing precisely because those monolithic methods read them, so removal is coupled to this composition work, NOT to #125. Each combinator needs a regression test with a non-ReAct `inner` (the analogue of the PlanExecute fix's test). After #124 truly lands, #125 (budget enforcement) and #126 (ready-set walk) open up. #121/#122/#127/#128/#129 remain parallel-grabbable. Design decision baked into the issues (diverges from the PRD's literal sketch, per the maintainer): strategies own their loop via a `RunStrategy` trait with one-line enum delegation, and a `StrategyRef::{BuiltIn, Custom}` escape hatch keeps built-ins a closed serde enum while allowing registered opaque custom strategies (resolves PRD Open Q A-1). The examples suite (#109 `13-coding-agent`, #92 observability) and `web_search` hardening (#108/#110) are now parked behind the refactor. The two #101-spawned harness gaps (#115 skill loading, #116 HITL child-consult resume) and the correctness/safety gates (#34 → #31 → #30) + docs (#27/#35/#36) remain parked pending an explicit maintainer call._

## Current State
spore-core is a language-agnostic agentic harness runtime with a **complete core
capability surface**, demonstrated through a numbered **examples suite** built
across all four targets: Rust (reference), TypeScript, Python, Go. ✅ `origin/main`
is level with local `main` (this loop's #124 series — rust/ts/py/go — pushed).

**🎯 Active work: the Composable Execution refactor (#117–#131, label
`loop-strategy-refactor`) — five bricks landed; #124 reopened partial.** The `StandardHarness` hardwires
three things the PRD makes composable: (1) loop strategy is a fixed `run()`
dispatch match — becomes a recursive `LoopStrategy` enum where each variant's
config struct owns its loop via a `RunStrategy` trait, recursion is
`self.inner.run(cx)`, and the only `match` is one-line enum delegation **(done in
#124)**; (2) budget is a single global gate — becomes a per-node
`BudgetPolicy`/`BudgetExhaustedBehavior` with a typed, isolated `StrategyOutcome`
(`BudgetExhausted` never confused with `Failed`) **(scaffold #123; enforcement
#125)**; (3) the task list is an implicit linear chain — becomes an explicit
`Task.blockers` DAG with a ready-set walk, two-tier context, and failure cascade to
transitive dependents only **(schema #118; walk #126)**. A
`StrategyRef::{BuiltIn, Custom}` seam keeps built-ins a closed serde enum (for
resume/versioning/`max_steps()`) while allowing registered opaque custom strategies.

**#117 done (1 of 15, `status: complete`).** `BudgetPolicy`
(`Unlimited`/`TotalSteps`/`PerLoop`/`PerAttempt`) + `BudgetExhaustedBehavior`
(`Continue{max_continues, on_exhausted}`/`Escalate`/`Fail`) shipped as pure
serializable value types — **no executor wiring** (that lands in #119/#125), layered
on top of the unchanged `BudgetLimits` global backstop. Byte-identical `kind`-first
snake_case wire format across all four languages, replayed against a shared
`fixtures/budget_policy/cases.json` (incl. nested `Continue→Continue→Fail`).
Spec-prose ambiguities resolved by the implementer: named struct fields with
`value` (serde internally-tagged enums can't do tuple variants), `u32` width
(a step == a turn), and `max_continues` is a **required** field with no default (no
node silently defaults to `Continue`). A Go divergence was caught + fixed in
cross-language verify (`b60205`→`b9a46bc`: Go silently accepted `continue` without
`max_continues`).

**#118 done (2 of 15, `status: complete`).** `Task.blockers: Vec<u32>`
(`#[serde(default)]`, serialized last in canonical order, always `[]` when empty —
matching `next_id`'s always-serialized treatment) + an optional `blockers`
integer-array param on the `task_list` `add_task` tool. `add`/`addTask`/`Add` is
now fallible and validates **before** mutation (reject leaves the list untouched),
in order self-block → unknown-id → cycle, surfacing a recoverable `invalid_blockers`
error with a `reason` sub-enum (`self_block` / `unknown_id`{`blocker`} / `cycle`),
byte-identical across all four. A `would_create_cycle` DFS helper is in place
(unreachable through append-only `add_task` alone, but required by the acceptance
criteria and ready for the future `update`-blockers slice). **No ready-set
scheduling** (deferred to #126). New `fixtures/tasklist/deserialize.json` proves
serde-default backward-compat; existing tasklist fixtures gained `"blockers":[]`
byte-identically. Spec ambiguities resolved from the spec text (empty→`[]` per the
`next_id` analogy; single error variant w/ reason; cycle helper tested directly).

**#119 done (3 of 15, `status: complete`).** The strategy-node shapes + the
composition seam — types and trait only, **no per-variant run bodies yet**.
`LoopStrategy` is now a closed recursive serde enum of config newtypes —
`ReAct(ReactConfig)`, `PlanExecute(PlanExecuteConfig)`, `SelfVerifying(SelfVerifyingConfig)`,
`Ralph(RalphConfig)`, `HillClimbing(HillClimbingConfig)`, `#[serde(tag="kind",
rename_all="snake_case")]` with the leaf tag overridden to **`react`** (was
`re_act`). `ReactConfig` is the leaf (`budget: BudgetPolicy` — the renamed
`max_iterations`, semantically `PerLoop` — plus `agent`/`toolset` required `*Ref`
handles + optional `output`); the rest are combinators holding `Box<LoopStrategy>`
children under phase-named keys (`plan`/`execute`, `inner`). The `RunStrategy`
trait (`async fn run(&mut ExecutionContext) -> StrategyOutcome`) is defined and
implemented on the enum as **one-line delegation** — the only `match` site — with
each config's `run` body **stubbed to a `Pending` placeholder that never
panics/throws/raises** (real bodies land in #124). `StrategyRef::{BuiltIn(LoopStrategy),
Custom(String)}` is **adjacently tagged** (`{"kind":"built_in","value":…}` /
`{"kind":"custom","value":"key"}`) to dodge the `kind` collision with the nested
internally-tagged `LoopStrategy`. New per-node collaborator handles `AgentRef`/
`ToolsetRef`/`SchemaRef` serialize as bare strings (resolution lands in #120). The
cordyceps tree `Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]` round-trips
**byte-identically** across all four languages against shared fixtures
`fixtures/strategy/{cordyceps_tree,strategy_ref,paused_state,child_paused_state}.json`;
`PausedState`/`ChildPausedState` round-trip the new shape. Scope deliberately
deferred: `StrategyOutcome`/`ExecutionContext` are minimal placeholders (full shape
→ #123), `max_steps()` omitted (→ #122), the existing executor dispatch was adapted
to the new newtype shape (reads `max_iterations` from the `PerLoop` budget) but
**not** migrated into `run` bodies (→ #124). Spec ambiguities (config field sets,
combinator child names, `StrategyRef` tagging, `react` vs `re_act`) were surfaced
to the maintainer and pinned in prose before the parallel agents ran. Two benign
cross-language naming/repr notes, both pre-existing & orthogonal: Go kept the type
name `OptimizationDirection` (Rust renamed to `HillClimbingDirection`) — wire
identical (`direction`: `minimize`/`maximize`); Go/Python use semantic-JSON
equality for the float-bearing `PausedState` wrappers (`cost_usd` `0` vs `0.0`).

**#120 done (4 of 15, `status: complete`).** `ExecutionRegistry` — the runtime
resolver that turns #119's serializable `*Ref` handles + `StrategyRef::Custom` keys
into concrete collaborators. Exactly **five maps** (`agents`/`toolsets`/`schemas`/
`verifiers`/`custom`), **never serialized** (holds trait objects). Pure-lookup
resolvers (`resolve_agent/toolset/schema/verifier`, each `*Ref` type → one map;
`SchemaRef`→`schemas`), `resolve_strategy` (`BuiltIn`→tree / `Custom`→lookup-or-
recoverable-`StrategyNotFound`), `register_strategy`, and `validate(task)` (a
recursive tree-walk that returns the first unresolved handle as `UnresolvedHandle`).
New `HarnessError::{StrategyNotFound{key}, UnresolvedHandle{kind,key}}` (the latter's
`kind` serializes as JSON key **`handle_kind`** to dodge the `#[serde(tag="kind")]`
discriminant; tags are PascalCase, matching the new `fixtures/harness/registry_errors.json`)
+ `HaltReason::ConfigurationError`. `EscalationMode{SurfaceToHuman,Autonomous}` HITL-
vs-AFK config knob added to `HarnessConfig` (stored only; **#130 wires the behavior**;
not serialized into `Task`). `validate()` is wired into `run_inner` entry but **gated
on a populated registry** so legacy callers stay byte-identical. **Scope was Option B
(ADDITIVE), a maintainer call** (see Deviation #12): the four legacy collaborator
fields (`agent`/`verifier`/`planner_agent`/`evaluator_agent`) were **kept and doc'd
as superseded, NOT removed**; the executor consumption sites are untouched; the
registry coexists but is not yet consumed by run bodies. **Physical removal +
executor migration to registry resolution was originally tagged to #124 but did NOT
happen there — now un-owned (see Deviation #12).** Two cross-language artifacts
(documented on the issue): `RunStrategy` was converted off
`#[trait_variant::make(Send)]` (RPITIT, not dyn-compatible) to a hand-rolled `BoxFut`
shape so `Arc<dyn RunStrategy>` (the `custom` map) can exist — no-op on the wire,
only Rust affected. Planner reported BLOCKED with 6 spec questions; 5 resolved from
the dependency graph / pinned #119 design / sibling specs, the 6th (remove-now vs
additive) was the maintainer's Option-B call. Cross-language verifier PASS; per-
language gates green (rust 16 new tests, ts 21, py 22, go 19).

**#123 done (5 of 15, `status: complete`).** The typed `StrategyOutcome` strategies
return + the shared mutable `ExecutionContext` threaded through the recursive tree —
**scaffold only; enforcement is #125**. Replaces #119's `Pending`/empty placeholders.
`StrategyOutcome` = `Complete(String)` | `BudgetExhausted{policy, behavior, steps_taken,
continues_used, phase, partial_output}` | `Failed(HarnessError)` — runtime-only, NEVER
serialized; a child's `BudgetExhausted` is an inspectable value that does **not**
auto-propagate (distinguishable from `Failed`). `ExecutionContext<'a>` carries `registry:
&ExecutionRegistry` (borrowed, per spec — the `RunStrategy` trait signature + all 7 impls
were updated to thread the context lifetime; `Arc<dyn RunStrategy>` stays dyn-compatible)
plus `budgets: BudgetStack`, `usage: AggregateUsage`, `session: SessionState`, `spans:
SpanStack`, `stream: Option<StreamSink>` — one shared context for a whole nested tree.
`BudgetContext{policy, behavior, steps_taken, continues_used, phase}` (each node gets its
own; siblings don't share) with the only real behavior this slice: `charge(turns)` is
**pure step-allowance arithmetic** (debit, increment on success, return `BudgetExhausted`
with NO mutation on overflow; `Unlimited` never exhausts — does NOT walk the behavior
chain or consume continues, that's #125), `remaining()` (allowance − steps_taken; `None`
for `Unlimited`), `continues_remaining()` (`Continue` saturating; `Escalate`/`Fail`→0).
`BudgetStack`/`SpanStack` are runtime-only push/pop stacks threaded through recursion. The
stub `*Config::run` bodies now return `Complete("")` (the removed `Pending`). **No
fixtures** — all types are runtime-only and never serialized (nothing crosses the wire).
Four spec ambiguities the planner flagged were resolved from the scaffold framing + spec
text (borrowed registry per spec; charge pure-arithmetic; `continues_used` in-memory only
this slice — checkpoint persistence deferred to #129; typed `SpanId` element in three
languages). Cross-language verifier PASS; gates green (rust 9 new / 980 total, ts 9 /
1194, py 13 / 1234, go 10). One genuine Go-only blocker surfaced + resolved (Deviation
#13).

**#124 REOPENED — PARTIAL (`status: blocked`); ~60% remaining.** Originally closed
`complete`, then reopened on review: its six ACs were **wrong** — they permitted
parity-via-reuse, so only **2 of 5** strategies are genuinely composable. The brick's
real goal is to implement the composable strategies AND **delete the monolithic
harness run-loops** (`StandardHarness::run_self_verifying`/`run_ralph`/
`run_hill_climbing`) — those are exactly what prevents composition.

**What genuinely landed (commits stay on `main`):** the central dispatch `match` is
gone — `LoopStrategy::run` is a one-line enum→config delegation. `ReAct` (leaf) runs
via the executor `react_window` primitive (correct; no children). `PlanExecute`
**genuinely recurses** — `self.plan.run(cx)` then `self.execute.run(cx)` per task.
That PlanExecute recursion was itself a mid-#124 fix: the first cut delegated to a
harness primitive that hardcoded a flat ReAct sub-loop, silently dropping the
configured `execute` child (the `parity-shortcut-hid-fake-recursion` lesson); the fix
(`1a4f46e`/`1003163`/`99ec2f4`/`291f872`) moved orchestration into
`PlanExecuteConfig::run` and added `plan_execute_runs_non_react_execute_child_per_task`
(asserts a `SelfVerifying` execute child fires per task). A.5 output-contract
enforcement (bare ReAct without `output` schema rejected in a structured slot) and A.6
deep-resume (`deep_resume_skips_already_completed_task`) also landed.

**What is NOT done (the reopened scope):** `SelfVerifying`, `Ralph`, `HillClimbing`
each have an `inner: Box<LoopStrategy>` child but **ignore it**. Their `Config::run`
delegates through `StrategyExecutor::{self_verifying_loop,ralph_loop,hill_climbing_loop}`
to monolithic `StandardHarness::run_*` methods that **hardcode the worker as
`LoopStrategy::ReAct(ReactConfig::per_loop(..))`** and read collaborators from
*harness* config — the identical fake-recursion class as PlanExecute's, never caught
because no AC required genuine recursion for these three and no test exercises a
non-ReAct `inner`. So `Ralph[PlanExecute[..]]` today actually runs `Ralph[ReAct]`.
**Corrected ACs (on the issue):** delete the three monolithic `run_*` (+ their
`StrategyExecutor` facade); every combinator dispatches its child via
`self.inner.run(cx)`; a per-combinator regression test with a non-ReAct `inner`;
**remove the four legacy collaborator fields** (`agent`/`verifier`/`planner_agent`/
`evaluator_agent`) and resolve via `ExecutionRegistry` — folded **back from #125**
because the fields are load-bearing precisely in the `run_*` methods this brick
deletes (Deviation #12). The landed slice's gates are green (rust 976 lib, ts 1201,
py 1247, go 18 pkgs) but green proves only the 2 composable strategies + parity of
the 3 monolithic ones.

**Examples suite — 12 of 13 landed, all four languages each.** Present under
`examples/{rust,typescript,python,go}/`:
`01-hello-agent`, `02-conversational-repl`, `03-tool-use`, `04-filesystem-agent`,
`05-custom-sandboxed-tool`, `06-web-research`, `07-memory`, `08-plan-execute`,
`09-self-verifying`, `10-hill-climbing`, `11-multi-agent`, **`12-cordyceps`
(#101)**. Each teaches one harness capability against a local Ollama
model. **Remaining example issues: #109** (`13-coding-agent` — batteries-not-
included coding-agent CLI; the final numbered example) and **#92** (observability
example — Phoenix/OTLP tracing).

**`12-cordyceps` capstone shipped (#101, all four languages):** a
fully-autonomous task-completion agent. ReAct orchestrator + `task_list`
decomposes a per-crate/per-module Rust audit; an Isolated `analysis_worker`
deep-dives one module and loads an `audit` skill at runtime via the
`GuideRegistry`; the #114 consult ladder escalates a stuck/uncertain worker to a
sibling helper then a human; heterogeneous models (local `gemma4:e4b` orchestrator
+ workers, cloud `minimax-m3:cloud` advisor); within-run `memory`; REPL approval →
`gh issue create`. Read-only audit; only writes are `workspace/findings.md` +
opt-in issue filings. **Zero core-harness change** — entirely composition over
existing seams. Cross-language verifier PASS; per-language gates green.

**Skill loading is done architect-side (the key #101 design decision):** the
harness's *structural* skill-injection path is **not live-wired**. The live loop
builds each turn's prompt via `StandardCompactionAdapter::assemble`, a pass-through
of `session.messages`; the rich `StandardContextManager::assemble` — the only code
that injects `pending_skill_injections` (Block-3 skills), `chunk_provider` chunks,
and merged `MemoryStore` memory — is **bypassed**, pending the deferred #7
ContextManager migration (this is the same root cause as Known Deviation #8). So
in every live loop strategy, skills/chunks/merged-memory reach the model only as
**tool-result messages**, never via structural injection. #101 therefore loads
skills with a `SkillCatalog` (scans `.spore/skills/{name}/SKILL.md` project →
`~/.spore/skills/` user, registers each as a skill `Guide` in
`StandardGuideRegistry`), a `load_skill` tool (appends ids to
`run_store["active_skills"]`), and a custom `ContextManager` that wraps the
standard adapter and injects a manifest every turn + active-skill bodies on demand,
ephemerally (compaction-proof). **#115** tracks absorbing this pattern into the
library (+ a FileSystem/Composite `GuideRegistry`, sibling of #88).

**The harness core remains DONE** (complete before the examples push):

- **All 5 of 5 advertised loop strategies run end-to-end across all four
  languages** — ReAct + PlanExecute + SelfVerifying (#61) + Ralph (#58) +
  HillClimbing (#60). No loop strategy is stubbed. (As of #124's landed slice
  ReAct + PlanExecute run through genuine recursive `RunStrategy` dispatch;
  SelfVerifying/Ralph/HillClimbing still run via the monolithic harness loops —
  the reopened #124 converts them.)
- **Mid-loop consult primitive (#114)** — worker-side `ToolOutput::Consult` →
  `RunResult::Consult` → `SubagentTool` mediates internally (kind→handler map +
  per-kind budget + overflow `SoftFail`/`EscalateToHuman`). #101 is its first
  real consumer. ⚠️ HITL gap found: `resume` can't resume a *worker's in-flight
  consult* through the parent (#116).
- **Track A — tool/prompt architecture — DONE** (#79/#80/#81/#82). ⚠️ but the
  prompt-assembly `ChunkProvider` and the rich `assemble` are NOT called in the
  live loop (only the SelfVerifying `role-evaluator` chunk lookup is) — see #115.
- **Clean, fully-pluggable, scope-aware persistence seam** (#73/#75/#76/#78/#82).
- **Runnable** (#57), **debuggable** (#64/#65), with a working
  **evaluation/feedback loop** (#26/#68).

**Parked (not the active track): correctness/safety debt + docs cleanup.** #34
(Yolo/None feature flag), #31 (SharedSession read-only), #30 (memory PendingReview
gate), docs #27/#35/#36. All `scope: deferred`.

## Active Direction
**Land the Composable Execution refactor — drive issues #117 → #131 to
completion** (label `loop-strategy-refactor`). Make `LoopStrategy`, budget, and the
task list compositional and mutually consistent, applied byte-identically across all
four languages where serialized. Work the critical path:
**117 → 119 → 120 → 123 → 124 → {125, 126} → 130 → 131**; #117, #118, #119 (the
keystone strategy seam), #120 (the runtime resolver), and #123 (the `StrategyOutcome`
+ `ExecutionContext` runtime scaffold) are done, but **#124 is REOPENED (partial) and
is again the top critical-path item.** Finish it: convert SelfVerifying / Ralph /
HillClimbing from the monolithic `StandardHarness::run_*` loops (which hardcode a ReAct
worker and ignore their `inner` child) to genuine `self.inner.run(cx)` recursion,
**delete those `run_*` methods + their `StrategyExecutor` facade**, add a
per-combinator regression test with a non-ReAct `inner`, and **remove the four legacy
collaborator fields** (Deviation #12, folded back here from #125 — they're load-bearing
only in those `run_*` methods). Only then do **#125 (per-node budget enforcement)** and
**#126 (ready-set task walk)** open up. #121 (`SubagentTool` strategy param), #122
(`max_steps()`), #127 (custom-strategy tracer), #128 (observability), and #129
(`Continue` checkpoint) remain parallel-grabbable. Use the `/implement` skill per issue (Rust reference + three parallel language agents +
cross-language verifier). Honor the maintainer's design choice — `RunStrategy` trait +
`StrategyRef::Custom` escape hatch — over the PRD's literal recursive-`run_strategy`-
with-match sketch (A.4). The success bar is #131: the `12-cordyceps` audit runs
end-to-end as `Ralph[PlanExecute[ReAct-plan, SelfVerifying[ReAct-worker]]]`, a runaway
node bounded by its own `BudgetPolicy` without cascading, and a paused tree resumes by
re-resolving handles.

**Parked behind the refactor (prior active direction):** the numbered examples
suite is **12 of 13** in; #109 (`13-coding-agent`) + #92 (observability example)
finish it, and `web_search` hardening (#108/#110) follows — pick these back up
after the refactor lands, or on an explicit maintainer call. The two #101-spawned
harness gaps — **#115** (first-class skill loading; live-injection path bypassed)
and **#116** (#114 HITL child-consult resume) — stay `status: queued`; note #116
(child-consult resume) overlaps the refactor's resume/HITL work (#130) and may be
worth folding in. Correctness/safety gates #34 → #31 → #30 and docs #27/#35/#36
remain parked pending a maintainer call. Larger feature issues — #113 (spore-lsp),
#107 (PromptEngineeringAgent), #106 (MicroVMSandboxProvider) — and the
protocol-integration track (#83–87) remain unscheduled. Storage follow-ups
#77/#88/#89 stay `scope: deferred`. **#7** (ContextManager migration) would
live-wire the rich `assemble` — proper home for #115's skill injection and the #32
cache halts.

## Known Deviations
1. **Go outbox is not zero-dependency** — closing #50 added
   `go.opentelemetry.io/otel` + `otlptracegrpc` (v1.28.0) as blessed deps
   (accepted, documented in `go/CONVENTIONS.md`). The durable JSONL path stays
   network-free.
2. **`task_list` / `todo_write` tool default persistence is no-op, not a file**
   (`scope: debt`, minor) — #75 retired the sandbox path; standalone tools persist
   via `RunStore`, which is `no_op()` by default. Durable standalone use requires
   wiring a real `StorageProvider`. Accepted; no migration shim.
3. **v1 memory keying limitation (#78 Q7), filed as #89** (`scope: deferred`) —
   `MemoryStore` is `SessionId`-keyed; durable cross-session addressing is the v2
   feature. No SQL backend yet either (#77).
4. **Go-specific divergences** (`scope: debt`, minor, documented on the issues) —
   local `Mode` newtype; 3-state `TerminalOutcome`; type-aliased `StandardTool`;
   duplicated `sendMessageToolName`; explicit `abort` `reason`; self-contained
   `promptassembly` builder; opaque `ToolContext.MemoryStore` marker; exported
   `storage.MergeMemories`; config-struct (not builder) setters for
   `Verifier`/`EvaluatorAgent`/Ralph/`MetricEvaluator`/`HillClimbing`;
   `RoleEvaluatorChunk` constant; consumer-side `MetricEvaluator`/`ContextError`
   seams to avoid import cycles. All wire/behavior-identical to the other three.
   `12-cordyceps` adds one more idiomatic Go choice (benign): the custom context
   manager **embeds** `*StandardCompactionAdapter` (struct-embedding inherits the
   non-`Assemble` methods) rather than delegating each explicitly. #124 adds:
   `RunStrategy.Run` takes `ctx context.Context` first (no Context in structs).
5. **Test-placement divergences (#78/#82)** (benign) — registry-seam / catalogue
   tests live in language-idiomatic spots. Behavior identical.
6. **#79 cross-language divergences — both verified benign.** (a) narrowed
   `composed_prompt` stub in TS/Go; (b) Block-1 hash not byte-identical (Rust
   SipHash vs FNV-1a) — the intentional #24 decision; #79 fixtures assert no hash
   values.
7. **`Custom` condition is invisible in fixtures by design** (#79) — serializes to
   null/absent; architects opt that chunk out of the byte-identical contract.
8. **The live harness loop does not call the rich `assemble`** (`scope: deferred`,
   intentional, depends on #7) — the live `StandardHarness` loop builds prompts via
   `StandardCompactionAdapter::assemble` (a pass-through of `session.messages`).
   The rich `StandardContextManager::assemble` — which injects
   `pending_skill_injections` (Block-3 skills), `chunk_provider` chunks, merged
   `MemoryStore` memory, and runs the Block-1/Block-2 `CacheHashMismatch` halts —
   is bypassed. Consequences: (a) skills/chunks/memory reach the model only as
   tool-result messages (drove #101's architect-side skill loading; bake-in #115);
   (b) the #32 cache halts can't fire end-to-end. Live-wiring the rich `assemble`
   is the deferred **#7** migration's job. (`chunk_provider.load()` runs live only
   for the SelfVerifying `role-evaluator` lookup.)
9. **#114 HITL has no child-consult resume, filed as #116** (`status: queued`) —
   `EscalateToHuman` consult overflow surfaces as `RunResult::WaitingForHuman` at
   the parent with the worker's paused consult in `child_state`, but `resume`'s
   `child_state` branch is a **no-op** in all four cores. So a human can resume the
   orchestrator but not the worker's in-flight consult. #101's three escalation
   choices (+1 advisor / abort / free-form) are therefore implemented host-side
   ("+1" re-runs the advisor host-side). Documented in all four #101 READMEs+code.
10. **Local `main` push hygiene (standing reminder)** — each per-issue loop's series
    must be pushed promptly so `origin/main` doesn't drift behind. This loop's #124
    series (`9279df8..291f872`, rust/ts/py/go feat + the genuine-recursion fix) is
    pushed; `origin/main` is level with local `main`. Sub-note: the plan-execute
    scratch run-artifacts are covered by a `workspace/*` wildcard in
    `examples/rust/08-plan-execute/.gitignore` (preserving the tracked `.gitkeep` +
    canonical `Async_Runtime_Comparison_Report.md`).
11. **Rust-only `12-cordyceps` polish + a Rust-only core addition** (`scope: debt`,
    not yet mirrored) — `8bb7734` adds `SubagentTool::with_stream` to the **core
    harness** (`rust/crates/spore-core/src/tools/subagent.rs`): an optional child
    stream sink so a host can observe a subagent's internal `StreamEvent`s, not just
    its final answer (defaults to `None`, existing behavior unchanged). `d65ae64`
    builds on it in the Rust `12-cordyceps` example (a `send_user_message` narration
    tool, a persistent REPL loop, repo-root-anchored `findings.md`, worker
    child-stream sink). **TS/Python/Go have neither the core `with_stream` seam nor
    the example polish.** Not tracked by an issue yet — decide whether to mirror (file
    an issue) or keep as a Rust-reference-ahead experiment.
12. **The legacy-collaborator-field removal + registry-resolution migration — OWNED BY
    THE REOPENED #124** (`scope: debt`) — #120 was ADDITIVE (Option B): the four flat
    collaborator fields (`agent`/`verifier`/`planner_agent`/`evaluator_agent`) were kept
    (doc'd as superseded) because `ExecutionContext` (carrying the registry) didn't exist
    until #123, and `validate()` is gated on a populated registry so legacy callers stay
    byte-identical. The removal was tagged to #124, then briefly folded into #125, now
    **moved back to #124** — because the fields are load-bearing precisely in
    `StandardHarness::run_self_verifying`/`run_ralph`/`run_hill_climbing`, the monolithic
    methods the reopened #124 must delete to make those three combinators compose. (Verified:
    46 `planner_agent`/`evaluator_agent` refs remain in `rust/.../harness.rs`.) Once #124's
    three combinators recurse via `self.inner.run(cx)` and resolve collaborators through
    `ExecutionRegistry`, the flat fields fall out and `validate()` ungates to a single
    resolution path. Remove this deviation once #124 truly lands. Sub-note: #119's `RunStrategy` was declared with
    `#[trait_variant::make(Send)]` (RPITIT, not dyn-compatible) — #120 converted it to a
    hand-rolled `BoxFut` return shape so the `custom: HashMap<String, Arc<dyn
    RunStrategy>>` map can exist. No-op on the wire (the trait never serializes); only
    Rust affected. #123/#124 build on the `BoxFut` shape.
13. **#123 Go `SpanStack` holds `string`, not a typed `SpanId`** (`scope: debt`,
    intentional, documented on the issue + in `execution_scaffold.go`) — Rust/TS/Python
    model `SpanStack` as a stack of their typed `SpanId`; Go cannot, because Go's
    `observability` package **imports** the top-level `sporecore` package (re-exporting
    `SessionID` etc.), so `sporecore` importing `observability.SpanID` back would be a
    compile-time import cycle. The new scaffold types must live in `sporecore`. Plain
    `string` matches how the Go harness loop already represents span ids (`var
    currentTurnSpanID string`). **Safe** because `SpanStack` is runtime-only and never
    serialized — the element type never crosses the wire. Sub-note (benign): `charge`'s
    fallible-result shape is idiomatic per language — Rust `Result<(), BudgetExhausted>`,
    TS `{ok}|{ok,error}`, Go `*BudgetExhausted` (nil = ok), Python raises `BudgetExhausted`;
    semantically identical.

_(Former Deviations — HillClimbing/SelfVerifying/Ralph-git-log/MemoryTool/storage-
scope/sandbox-path/extras-mirror/Rust-dyn/compaction-tokens/observability-content
stubs — all resolved in prior loops.)_

## Next Actions
[3-5 items max, highest priority first. /next surfaces item 1 as "work this next."]
1. **#124 (REOPENED) — finish genuine composition.** Convert SelfVerifying / Ralph /
   HillClimbing to dispatch their `inner` child via `self.inner.run(cx)`; **delete**
   `StandardHarness::run_self_verifying`/`run_ralph`/`run_hill_climbing` and their
   `StrategyExecutor` facade; add a per-combinator regression test with a non-ReAct
   `inner` (e.g. `SelfVerifying[PlanExecute[..]]`, `Ralph[SelfVerifying[ReAct]]`); and
   **remove the four legacy collaborator fields**, resolving all collaborators through
   `ExecutionRegistry` (Deviation #12, folded back here from #125). Apply byte-identically
   across all four languages. Grab via `/implement`. This is the gate for the rest of the
   critical path.
2. **#125 — per-node budget enforcement + failure isolation (Composable Execution
   B.1–B.5).** Flesh out #123's pure-arithmetic `charge` into real enforcement: walk the
   `BudgetExhaustedBehavior` chain (`Continue`/`Escalate`/`Fail`), consume continues, and
   ensure a child's `BudgetExhausted` is isolated (never confused with `Failed`, never
   auto-cascades). **Blocked on #124's true completion** (needs the fully-composable
   executor). Grab via `/implement`.
3. **#126 — ready-set task walk + two-tier context + failure cascade (Part C
   executor).** Consume #118's `Task.blockers` DAG: a ready-set walk over unblocked
   tasks, two-tier (run vs task) context, and failure cascade to transitive
   dependents only. Opens up after #124; parallel-grabbable with #125.
4. **Refactor finishers (parallel-grabbable now)** — #121 (`SubagentTool` strategy
   param), #122 (`max_steps()`), #127 (custom-strategy tracer — exercises #120's
   `custom` map + `StrategyNotFound` end-to-end), #128 (observability span attrs),
   #129 (`Continue` checkpoint — owns #123's deferred `continues_used` persistence),
   then #130 (`HumanRequest::BudgetExhausted` HITL resume — consumes #120's
   `EscalationMode` knob), then **#131** (cordyceps capstone — the success bar).
5. **Decide on the Rust-only `SubagentTool::with_stream` (Deviation #11)** — file a
   mirror issue for TS/Python/Go or accept as a Rust-reference-ahead experiment.
   **Still parked:** examples #109 / #92 + `web_search` #108/#110; harness gaps #115
   (skill loading) and #116 (HITL child-consult resume — overlaps #130); correctness/
   safety #34 → #31 → #30 + docs #27/#35/#36.
