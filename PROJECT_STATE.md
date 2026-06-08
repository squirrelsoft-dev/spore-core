# PROJECT STATE
_Last updated: 2026-06-07 by /close (#130 **complete**, all four languages, cross-language verifier PASS; local `main` is **4 commits ahead** of `origin/main` — `f30caa2`→`06dbc69`→`1831e30`→`97b8775` — **not yet pushed**, awaiting maintainer OK per the standing push gate) — **The Composable Execution refactor (loop strategy / budget / task graph) is the top priority.** A PRD (`spore-core-composable-execution-prd.md`) was broken into **15 tracer-bullet issues #117–#131** (label `loop-strategy-refactor`) via `/to-issues`. The goal is to land #117→#131 to completion: make `LoopStrategy` a composable recursive enum where each variant owns its run loop via a `RunStrategy` trait (no central dispatch match), add a compositional per-node `BudgetPolicy`/`BudgetExhaustedBehavior` budget layer with typed `StrategyOutcome`, and make the task list an explicit blocker DAG with a ready-set walk + failure cascade. Capstone #131 re-expresses the `12-cordyceps` audit as `Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]`. **Nine bricks landed (#117, #118, #119, #120, #123, #124, #125, #126, #130) all `status: complete`, all four languages.** #130 (HITL budget-exhausted resume) wired the previously-stored-but-unconsumed #120 `EscalationMode` knob: an `Escalate`-behavior node that exhausts its budget now consults `escalation_mode()` — `SurfaceToHuman` pauses with a new resumable `HumanRequest::BudgetExhausted` (carrying `phase`/`policy`/`steps_taken`/`continues_used`/`partial_output`/`available_actions`), `Autonomous` keeps the prior propagate behavior. New `EscalationAction{ContinueWithBudget{steps}, Skip, Fail}` + `HumanResponse::Escalate{action}`; resume maps each action (Continue grants `steps_taken+steps` and re-enters from checkpoint, Skip advances PlanExecute's outer loop, Fail → `Failure{BudgetExceeded}`). New cross-language fixture `fixtures/paused_states/budget_exhausted.json`. With #130 done, **only the #131 capstone remains on the critical path** — #129/#127/#128/#121/#122 are parallel-grabbable (#131 is gated on #130 ✅ + #129 for the live Continue loop). The examples suite (prior direction) is parked at 12 of 13 — #109/#92 remain but yield priority to the refactor._

_**Direction note:** Active direction is the **loop-strategy refactor (#117–#131)**. Critical path: 117 → 119 → 120 → 123 → 124 → {125, 126} → 130 → 131; **#117, #118, #119, #120, #123, #124, #125, #126, #130 are all done** — the entire critical path **except the #131 capstone** is complete. **#131** (cordyceps composition end-to-end) is the success bar; it is gated on #130 (✅ done) and benefits from #129 (the live in-process `Continue` loop). #121/#122/#127/#128/#129 remain parallel-grabbable (#129 owns #125's deferred Continue wire-field + `continues_used` persistence; #130 deliberately reconstructs its `BudgetContext` from the request payload to stay out of #129's scope). Design decision baked into the issues (diverges from the PRD's literal sketch, per the maintainer): strategies own their loop via a `RunStrategy` trait with one-line enum delegation, and a `StrategyRef::{BuiltIn, Custom}` escape hatch keeps built-ins a closed serde enum while allowing registered opaque custom strategies (resolves PRD Open Q A-1). The examples suite (#109 `13-coding-agent`, #92 observability) and `web_search` hardening (#108/#110) are now parked behind the refactor. The two #101-spawned harness gaps (#115 skill loading, #116 HITL child-consult resume) and the correctness/safety gates (#34 → #31 → #30) + docs (#27/#35/#36) remain parked pending an explicit maintainer call. Note #116 (child-consult resume) overlaps #130's HITL resume seam and may now be cheaper to fold in._

## Current State
spore-core is a language-agnostic agentic harness runtime with a **complete core
capability surface**, demonstrated through a numbered **examples suite** built
across all four targets: Rust (reference), TypeScript, Python, Go. ⚠️ local `main`
is **4 commits ahead** of `origin/main` (this loop's #130 four-language series —
`f30caa2`/`06dbc69`/`1831e30`/`97b8775`) — **not yet pushed**, awaiting maintainer
OK per the standing push gate (Deviation #10).

**🎯 Active work: the Composable Execution refactor (#117–#131, label
`loop-strategy-refactor`) — nine bricks landed (#130 now complete; only the #131
capstone remains on the critical path).** The `StandardHarness` hardwires
three things the PRD makes composable: (1) loop strategy is a fixed `run()`
dispatch match — becomes a recursive `LoopStrategy` enum where each variant's
config struct owns its loop via a `RunStrategy` trait, recursion is
`self.inner.run(cx)`, and the only `match` is one-line enum delegation **(done in
#124)**; (2) budget is a single global gate — becomes a per-node
`BudgetPolicy`/`BudgetExhaustedBehavior` with a typed, isolated `StrategyOutcome`
(`BudgetExhausted` never confused with `Failed`) **(scaffold #123; enforcement
#125 — done: `charge()` is the real per-node gate, child exhaustion is isolated and
never auto-cascades, ReAct-leaf propagates per Open Q A-2; the `Continue` wire-field
+ `continues_used` persistence remain #129's job)**; (3) the task list is an implicit
linear chain — becomes an explicit `Task.blockers` DAG with a ready-set walk, two-tier
context, and failure cascade to transitive dependents only **(schema #118; walk
#126 — done)**. A `StrategyRef::{BuiltIn, Custom}` seam
keeps built-ins a closed serde enum (for resume/versioning/`max_steps()`) while
allowing registered opaque custom strategies.

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

**#124 done (6 of 15, `status: complete`) — TRULY complete after a reopen.** Was
closed `complete`, reopened on review (its six original ACs permitted parity-via-reuse,
so only **2 of 5** strategies genuinely composed), then finished this loop. **All five
strategies now genuinely compose.** The central dispatch `match` is gone —
`LoopStrategy::run` is a one-line enum→config delegation (AC1). `ReAct` (leaf) runs via
the executor `react_window` primitive, now resolving its worker agent from the registry
(`resolve_agent_ref`, threaded through `run_react_inner`/`react_window`). `PlanExecute`,
`SelfVerifying`, `Ralph`, `HillClimbing` all **genuinely recurse** into their children
via `self.inner.run(cx)` (PlanExecute: `self.plan.run(cx)` then `self.execute.run(cx)`
per task). The three monolithic `StandardHarness::run_self_verifying`/`run_ralph`/
`run_hill_climbing` loops + their `StrategyExecutor` facades (`self_verifying_loop`/
`ralph_loop`/`hill_climbing_loop`) are **DELETED** in all four languages.

**The reopen fix (commits `88eaafa`/`3906b03`/`4852985`/`59601cf` + README docs
`b2dc2d8`):** converted the three combinators to recurse mirroring PlanExecute;
deleted the monolithic loops + facades; **removed the four legacy collaborator fields**
(`agent`/`verifier`/`planner_agent`/`evaluator_agent`, plus `metric_evaluator`) and
resolve all collaborators via `ExecutionRegistry` — builders fold their defaults into
the registry under the empty key so public construction signatures stay stable;
`validate()` is **ungated** to a single resolution path. Three design forks were
resolved wire-stably (commented on the issue): **Q1** `SelfVerifying.evaluator`'s key
resolves against the verifier map + eval agent defaults to the inner worker (planner
concept dropped); **Q2** a sixth `metric_evaluators` registry map + `resolve_metric_
evaluator` (HillClimbing's `evaluator` string wire unchanged); **Q3** `RalphConfig.agent`
overrides the inner leaf's agent per window when set. **No fixture changes** (wire format
frozen — pure executor-side recomposition). Semantic shift: a missing verifier/metric
now surfaces as a startup `ConfigurationError`/`UnresolvedHandle` rather than a runtime
`*Misconfigured` halt; affected tests migrated.

**The lesson that caused (and closed) the reopen — the non-ReAct-`inner` tests:** the
original green proved only the 2 composable strategies + *parity* of the 3 monolithic
ones, because every test used a ReAct `inner` (which coincides with the hardcoded
worker). The fix adds a per-combinator regression test with a **non-ReAct `inner`** that
asserts a COUNT of an inner-combinator-only collaborator invocation (a hardcoded-ReAct
impl would record ZERO): `self_verifying_runs_non_react_inner_worker`
(SelfVerifying[PlanExecute[ReAct,ReAct]]), `ralph_runs_non_react_inner_per_window`
(Ralph[SelfVerifying[ReAct]]), `hill_climbing_runs_non_react_inner_per_iteration`
(HillClimbing[PlanExecute[ReAct,ReAct]]). Cross-language verifier PASS. Gates green:
rust 979 lib, ts 1437, py 1248, go all 18+ pkgs. A.5 output-contract + A.6 deep-resume
behaviors preserved. One benign documented Go divergence (Deviation #4): Go keeps
`Agent`/`ToolRegistry` as config-struct fields (its public constructor surface) folded
into the registry + an `IsEmpty()` validate gate for the nil-agent scaffold path —
behaviorally equivalent for all real harnesses.

**#125 done (7 of 15, `status: complete`).** Per-node budget enforcement + failure
isolation (B.1–B.5). Flesh out #123's pure-arithmetic scaffold into real enforcement:
`charge()` is now the **live** per-node gate and `StrategyOutcome::BudgetExhausted` a
real, isolated, parent-inspectable value (it was produced **nowhere** before — #124
routed all exhaustion through the legacy `task.budget.max_turns`/`BudgetExceeded→Failed`
path). New: `BudgetContext::{consume_continue, resolve_exhausted}` walking the
`BudgetExhaustedBehavior` chain (Continue grants a `steps_taken→0` reset while continues
remain, then falls through to `on_exhausted`; Escalate/Fail terminal), a runtime-only
`ExhaustedResolution` enum, `BudgetPolicy::allowance_value`, `ExecutionContext`
budget-scope helpers (push/pop/charge/resolve), a `promote_budget_exhausted` boundary +
four per-node `partial_output` extractors. All five `*Config::run` bodies wired: a
capped node stops **without killing siblings**, a child `BudgetExhausted` reaches the
parent as an inspectable outcome that **never auto-cascades** (parent's own
`BudgetContext` untouched), and the **ReAct leaf propagates** exhaustion — never
self-resolves (Open Q A-2). The dead per-task derivation
(`global_max_turns`/`remaining_tasks`/`per_task_turns`/`sub_loop_cap`/`step_cap`) is
removed from every PlanExecute body **and** its cfg-test helper in all four. Three
maintainer-resolved spec forks: Escalate→`Some(partial)` / Fail→`None`; `partial_output`
is a JSON string per node (ReAct=last FinalResponse; PlanExecute=tasklist+statuses+ledger;
SelfVerifying=worker+verdict; HillClimbing=best+score); `continues_used` persistence
**deferred to #129** (in-process Continue only, **no new fixtures**; #117 budget_policy
fixtures still pass byte-identically). `BudgetExhaustedBehavior` is deliberately **not**
wired onto config structs (a serialized wire change forbidden by fork #3 — belongs to
#129); live bodies push an in-process `Escalate` placeholder, and the Continue/Fail/
Escalate chain is exercised by direct primitive tests. Commits `763b53a` (rust, 15
tests) / `2644117`+`f3c9362` (py, 14) / `3e4a33e` (go, 14) / `f192fd0`+`5ccdb03` (ts,
19). Cross-language verifier PASS; two stale-helper divergences found+fixed during verify
(py `f3c9362`, ts `5ccdb03`). ✅ A post-merge adversarial-review hardening pass (Deviation
#14, `ca0165b`/`ca89df6`/`c96ceed`/`9a9fb12`, verifier PASS) confirmed the new
`BudgetExhausted`/`partial_output` path is reachable end-to-end in all four (no impl gap),
made the leaf-cap integration test discriminating, added bounded-leaf F5/F6 coverage,
fixed the stale Rust doc, and made the Continue arms explicit; only #129's wire-field
remains.

**#126 done (8 of 15, `status: complete`).** Ready-set task walk + two-tier context +
failure cascade (Part C executor) — the entire critical path through Part C is now
complete. Replaced PlanExecute's positional `for index in 0..total_tasks` loop, its
linear context folding, and its Q5 blanket abort with a DAG-aware executor honoring
`Task.blockers`: (1) **ready-set scheduler** — picks the lowest-id `Pending` task whose
blockers are all `Completed` (positional/id tiebreak), runs it, marks `Completed`,
repeats; v1 sequential (parallel deferred); cycles re-checked at execute entry
(`HaltReason::TaskGraphCycle`) on top of #118's `add_task` rejection. (2) **Two-tier
context** replacing linear folding: Tier-1 seeds a task's sub-strategy with ONLY its
**transitive blockers'** final outputs + scoped ledger (independent branches excluded —
no cross-branch pollution); Tier-2 is a global running `StepLedgerEntry {task_id,
summary, files_touched}` ledger injected into every step, **bounded N=20 drop-oldest**
(deterministic — no model summarization, byte-identical) with a static elision marker.
`files_touched` is **harness-observed** from write/edit tool-call dispatch
(`observe_write_call` records `write_file`/`edit_file` paths), NEVER self-reported.
(3) **Failure cascade** replacing the blanket abort: a terminal failure (unrecoverable
error OR budget resolving to `Fail`) blocks only its **transitive dependents**
(run-local `blocked_by_failure` set; no new `TaskStatus` variant); unrelated tasks keep
running; the run drains to `HaltReason::TasksBlockedByFailure {completed, blocked,
failed_task, reason}` (a `RunResult::Failure` reporting the partition — no first-failure
abort). A whole-run `BudgetExceeded` still hard-stops verbatim. (4) The
`plan_artifact_to_task_list` **bridge is deprecated** (kept working); the executor now
sources its `TaskList` from the persisted `task_list` tool store (the one authoring
path), linear-bridge fallback retained. New `StepLedgerEntry` + `STEP_LEDGER_MAX_ENTRIES`
+ DAG helpers (`next_ready`, `transitive_blockers`, `transitive_dependents`, `has_cycle`,
`push_step_ledger`, `render_step_ledger`) in `tasklist`. Five spec ambiguities resolved
with the maintainer up front (terminal=Failure-with-partition; ledger N=20 drop-oldest;
deprecate-not-remove bridge; Tier-1 = transitive outputs+ledger; blocked-reason in
run-scratch). 5 shared fixtures (`fixtures/model_responses/harness/plan_execute_dag_*`)
replay byte-identically — failure-cascade partition `failed_task=2, completed=[1,4],
blocked=[2,3]` identical across all four. Commits rust `a5d50be` (1092 tests) / go
`82cd3c3` / ts `e24236b` (1476) / py `924d4d4` (1292). Cross-language verifier PASS, no
divergences. One documented adaptation (Deviation #15): the `PlanArtifact` **type** was
NOT given a deprecation attribute (only the bridge **function** was) because it is still
the live `OnPlanCreated` hook payload — attributing it would cascade warnings through
live non-bridge code and break `-D warnings`.

**#130 done (9 of 15, `status: complete`).** HITL budget-exhausted resume (Part B,
B.6) — wires the #120 `EscalationMode{SurfaceToHuman,Autonomous}` knob that was stored
but **unconsumed** since #120. Today an `ExhaustedResolution::Escalate` node surfaced its
partial output and aborted (`StrategyOutcome::BudgetExhausted` → `RunResult::Failure
{BudgetExceeded}`); the knob was never read. #130 makes every Escalate resolution site
(PlanExecute ×2, SelfVerifying, HillClimbing, bare-leaf) consult `escalation_mode()`:
`Autonomous` keeps the existing propagate behavior; `SurfaceToHuman` builds a
`RunResult::WaitingForHuman` carrying a new `HumanRequest::BudgetExhausted {phase, policy,
steps_taken, continues_used, partial_output, available_actions}` (existing
`ToolApproval`/`Clarification`/`Review` variants UNCHANGED, `Eq` preserved) and installs
it via the existing `terminal_override` seam so it propagates verbatim. New
`EscalationAction {ContinueWithBudget{steps:u32}, Skip, Fail}` (kind-tagged snake_case)
+ `HumanResponse::Escalate{action}`; `resume_inner` gets a `BudgetExhausted`-request
branch before the generic handler: `ContinueWithBudget` reconstructs the node's
`BudgetContext` from the request payload and grants `steps_taken+steps` (re-enters from
checkpoint), `Skip` advances PlanExecute's outer loop (leaf → clean Success), `Fail` →
`Failure{BudgetExceeded}` (partial discarded). New `escalation_mode()` accessor on
`StrategyExecutor` + `StandardHarness`. Five spec forks resolved with the maintainer up
front: **A** named `ContinueWithBudget{steps:u32}` (serde internally-tagged can't do tuple
variants); **B** typed `HumanResponse::Escalate{action}` (no `Allow`/`Halt`/`Deny`
overload); **C** combinators offer `[ContinueWithBudget, Skip, Fail]`, a bare leaf omits
`Skip` (no outer loop); **D** `available_actions` is **advisory** for v1 (resume does not
hard-reject out-of-set); **E** #130 reconstructs `BudgetContext` from the request payload
to stay out of #129's durable-checkpoint scope. New cross-language fixture
`fixtures/paused_states/budget_exhausted.json` (new `paused_states/` subdir) replayed in
all four. Commits rust `f30caa2` (+15, 1107) / py `06dbc69` (+19, 1311) / ts `1831e30`
(+24, 1640) / go `97b8775` (+19). Cross-language verifier PASS; two benign divergences
documented (Deviations #16, #17).

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
  HillClimbing (#60). No loop strategy is stubbed. **As of #124 (complete) all
  five run through genuine recursive `RunStrategy` dispatch** — every combinator
  recurses into its child via `self.inner.run(cx)`; the monolithic harness loops
  are deleted.
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
keystone strategy seam), #120 (the runtime resolver), #123 (the `StrategyOutcome`
+ `ExecutionContext` runtime scaffold), **#124 (all five strategies genuinely
compose; monolithic loops deleted; legacy fields removed), #125 (per-node budget
enforcement), and #126 (ready-set task walk + two-tier context + failure cascade)
are done** — the whole path through Part C (the executor) is complete. The next
critical-path item is **#130 (HITL `HumanRequest::BudgetExhausted` + Escalate
resume)**, which consumes #125's `BudgetExhausted` + #120's `EscalationMode`, then the
**#131** capstone. #121 (`SubagentTool` strategy param), #122
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
   `RunStrategy.Run` takes `ctx context.Context` first (no Context in structs); and —
   because in Go the config-struct field *is* the public constructor surface — Go keeps
   `Agent`/`ToolRegistry` as struct fields folded into the registry under the empty key
   (vs Rust/TS/Python's retained builder `agent` arg) and retains an `IsEmpty()` gate at
   the validate site to skip validation for the nil-agent scaffold-only path the other
   three don't expose. Behaviorally equivalent for all real harnesses (validation always
   runs; missing handle → startup `ConfigurationError`).
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
    must be pushed promptly so `origin/main` doesn't drift behind. ⚠️ Current: local
    `main` is **4 commits ahead** of `origin/main` (the #130 four-language series
    `f30caa2`/`06dbc69`/`1831e30`/`97b8775`) — **not yet pushed**, awaiting maintainer OK.
    Note the standing push-approval gate: **ask before pushing** (an agent-initiated push
    was denied in a prior session).
    Sub-note: the plan-execute
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
12. **#119's `RunStrategy` is a hand-rolled `BoxFut`, not `#[trait_variant::make(Send)]`**
    (`scope: debt`, Rust-only, benign) — #119 declared `RunStrategy` with
    `#[trait_variant::make(Send)]` (RPITIT, not dyn-compatible); #120 converted it to a
    hand-rolled `BoxFut` return shape so the `custom: HashMap<String, Arc<dyn
    RunStrategy>>` map can exist. No-op on the wire (the trait never serializes); only
    Rust affected. #123/#124 build on the `BoxFut` shape. _(The legacy-collaborator-field
    removal + registry-resolution migration formerly tracked here is **RESOLVED by #124**:
    the four fields plus `metric_evaluator` are gone, all collaborators resolve via
    `ExecutionRegistry`, builders fold defaults under the empty key, and `validate()` is
    ungated to a single resolution path.)_
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

14. **#125 review follow-ups — ADDRESSED** (hardening pass `ca0165b`/`ca89df6`/`c96ceed`/
    `9a9fb12`, all four languages; cross-language verifier PASS). A post-merge adversarial
    review surfaced three items; the hardening pass resolved them: (c) **thin integration
    coverage** — the empirical question (was the headline behavior actually reachable
    end-to-end, or was the test degenerate like the one that got #124 reopened?) was
    answered: **the new `BudgetExhausted`/`partial_output` path IS reachable end-to-end in
    all four languages — no implementation gap.** The `react_leaf_cap_binding_propagates_partial`
    test is now discriminating (asserts the materialized session is exactly one assistant
    message == `react_partial_json("")` → `{"node":"react","last_final_response":""}`, byte-
    identical across all four; FAILS on pre-#125 code). Added the Ralph budget-exhausted-
    window→reset test (F5, **bounded** leaf — the path the old tests dodged) and strengthened
    the child-non-cascade test (F6, parent pre-charged to 4/5, survives child exhaustion with
    1 step intact). (b) **stale Rust doc header** fixed (described nonexistent `charge_step`/
    `run_with_budget`/`execute_budget` wrappers; corrected to the real per-body inline
    charge/resolve; TS/Py/Go had no analogous stale comment). (a) **defensive Continue** —
    the three combinator resolution sites now handle `ExhaustedResolution::Continue`
    EXPLICITLY (no silent `_`/`else`/`default` fall-through) with `#129` markers.
    **Remaining (genuinely #129's scope, not a deviation):** the in-process `Continue` *loop*
    is still not wired through a live run — every live `*Config::run` pushes an `Escalate`
    placeholder because the serialized `BudgetExhaustedBehavior` field is a wire change
    forbidden by fork #3; wiring the field + making `Continue` actually loop is **#129**.

15. **#126 `PlanArtifact` type not formally deprecated, only the bridge function**
    (`scope: debt`, minor, all four languages, documented on the issue) — decision C
    pinned "deprecate the `PlanArtifact` bridge." All four mark the bridge **function**
    (`plan_artifact_to_task_list`) deprecated (Rust `#[deprecated]`, TS `@deprecated`, Go
    `// Deprecated:`, Python `DeprecationWarning`) and rewire the executor to source its
    `TaskList` from the persisted `task_list` tool store. But the `PlanArtifact` **type**
    itself was deliberately **not** attributed, because it is still the live
    `OnPlanCreated` hook payload — attributing it would cascade deprecation warnings
    through legitimately-live non-bridge code and break the zero-warnings (`-D warnings`)
    gate. Its deprecation-as-an-authoring-source is documented in prose instead. Consistent
    judgment applied across all four (TS/Py/Go mirrored Rust's call). No wire/behavior
    impact; the bridge + its replay tests stay green.

16. **#130 Go fixture comparison uses `jsonEqual`, not byte-equal** (`scope: debt`,
    benign, Go-only, documented on the issue + `harness.go:599-604`) — the
    `fixtures/paused_states/budget_exhausted.json` replay test in Rust/TS/Python asserts
    byte-identical re-serialize; Go uses the codebase's established `jsonEqual`
    value-normalizing helper because Go's `encoding/json` emits `0` (vs serde's `0.0`) for
    the whole-number `cost_usd` float carried in `PausedState`. This is the same
    pre-existing stdlib-float-formatting convention already used by the consult/escalation/
    hooks fixture-replay tests — not a #130 wire-shape divergence. Field order/structure
    still match the fixture exactly.

17. **#130 default `escalation_mode` is applied at different layers per language**
    (`scope: debt`, benign, documented on the issue) — now that #130 *consumes* the
    `EscalationMode` knob, the default matters. TS defaults a *raw* `HarnessConfig
    .escalationMode` to `autonomous` and has the `HarnessBuilder` set `surfaceToHuman`;
    Python/Go default the config itself to `surfaceToHuman` (Go `EffectiveEscalationMode()`
    treats the zero value as `surface_to_human`). Each language preserves its own pre-#130
    legacy default — the only difference is whether the `surfaceToHuman` default lands at
    the raw-struct or builder layer. Consistent in spirit; pre-#130 budget tests pin
    `Autonomous` on their config helpers in all four to keep asserting propagate behavior.
    A maintainer may wish to harmonize the raw-config default separately.

_(Former Deviations — HillClimbing/SelfVerifying/Ralph-git-log/MemoryTool/storage-
scope/sandbox-path/extras-mirror/Rust-dyn/compaction-tokens/observability-content
stubs — all resolved in prior loops.)_

## Next Actions
[3-5 items max, highest priority first. /next surfaces item 1 as "work this next."]
0. **Push the #130 series to `origin/main`** (awaiting maintainer OK) — local `main` is
   4 commits ahead (`f30caa2`/`06dbc69`/`1831e30`/`97b8775`). Per the standing push gate,
   ask before pushing so `origin/main` doesn't keep drifting (Deviation #10).
1. **#129 — `Continue` cross-process checkpoint (B.4/B.7).** Now the highest-value
   refactor finisher: it owns #123's deferred `continues_used` persistence **and #125's
   deferred `BudgetExhaustedBehavior` wire-field** — the one piece that makes the
   in-process `Continue` loop actually loop through a live run, and a prerequisite for
   #131's "paused tree resumes" success criterion alongside #130. Grab via `/implement`.
2. **#131 — cordyceps composition end-to-end (the success bar).** Re-express the
   `12-cordyceps` audit as `Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]`: a runaway
   node bounded by its own `BudgetPolicy` without cascading, a failure cascading only to
   transitive dependents, and a paused tree resuming by re-resolving handles. Gated on
   #130 (✅ done) + #129 (live Continue loop). The last critical-path item.
3. **Other refactor finishers (parallel-grabbable now)** — #121 (`SubagentTool` strategy
   param), #122 (`max_steps()` advisory bound), #127 (custom-strategy tracer — exercises
   #120's `custom` map + `StrategyNotFound` end-to-end), #128 (observability span attrs).
4. **Decide on the Rust-only `SubagentTool::with_stream` (Deviation #11)** — file a
   mirror issue for TS/Python/Go or accept as a Rust-reference-ahead experiment.
   **Still parked:** examples #109 / #92 + `web_search` #108/#110; harness gaps #115
   (skill loading) and #116 (HITL child-consult resume — now overlaps #130's resume
   seam, may be cheaper to fold in); correctness/safety #34 → #31 → #30 + docs
   #27/#35/#36.
