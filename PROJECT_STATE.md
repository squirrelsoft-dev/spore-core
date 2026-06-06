# PROJECT STATE
_Last updated: 2026-06-06 by /close (closed #118) — **The Composable Execution refactor (loop strategy / budget / task graph) is the top priority.** A PRD (`spore-core-composable-execution-prd.md`) was broken into **15 tracer-bullet issues #117–#131** (label `loop-strategy-refactor`) via `/to-issues`. The goal is to land #117→#131 to completion: make `LoopStrategy` a composable recursive enum where each variant owns its run loop via a `RunStrategy` trait (no central dispatch match), add a compositional per-node `BudgetPolicy`/`BudgetExhaustedBehavior` budget layer with typed `StrategyOutcome`, and make the task list an explicit blocker DAG with a ready-set walk + failure cascade. Capstone #131 re-expresses the `12-cordyceps` audit as `Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]`. **Two bricks landed: #117 (`BudgetPolicy`/`BudgetExhaustedBehavior` value types) and #118 (`Task.blockers` schema + `add_task` blockers param) are both `status: complete`, all four languages, byte-identical.** The examples suite (prior direction) is parked at 12 of 13 — #109/#92 remain but yield priority to the refactor. ⚠️ **Local `main` is 13 commits ahead of `origin/main` (f63bf07) and unpushed** — the 5 #117 commits, 4 #118 commits, 2 Rust-only `12-cordyceps` polish commits (one of which, `SubagentTool::with_stream`, is a real core-harness addition not yet mirrored to TS/Python/Go), and 2 `/close` bookkeeping commits._

_**Direction note:** New active direction is the **loop-strategy refactor (#117–#131)**. Critical path: 117 → 119 → 120 → 123 → 124 → {125, 126} → 130 → 131; #118 is now done, #121/#122 remain parallel-grabbable immediately. Design decision baked into the issues (diverges from the PRD's literal sketch, per the maintainer): strategies own their loop via a `RunStrategy` trait with one-line enum delegation, and a `StrategyRef::{BuiltIn, Custom}` escape hatch keeps built-ins a closed serde enum while allowing registered opaque custom strategies (resolves PRD Open Q A-1). The examples suite (#109 `13-coding-agent`, #92 observability) and `web_search` hardening (#108/#110) are now parked behind the refactor. The two #101-spawned harness gaps (#115 skill loading, #116 HITL child-consult resume) and the correctness/safety gates (#34 → #31 → #30) + docs (#27/#35/#36) remain parked pending an explicit maintainer call._

## Current State
spore-core is a language-agnostic agentic harness runtime with a **complete core
capability surface**, demonstrated through a numbered **examples suite** built
across all four targets: Rust (reference), TypeScript, Python, Go. ⚠️ Local `main`
is **13 commits ahead of `origin/main` (f63bf07) and unpushed** (the #117 + #118
work, 2 Rust-only cordyceps commits, 2 `/close` bookkeeping commits).

**🎯 Active work: the Composable Execution refactor (#117–#131, label
`loop-strategy-refactor`) — two bricks landed.** The `StandardHarness` hardwires
three things the PRD makes composable: (1) loop strategy is a fixed `run()`
dispatch match — becomes a recursive `LoopStrategy` enum where each variant's
config struct owns its loop via a `RunStrategy` trait, recursion is
`self.inner.run(cx)`, and the only `match` is one-line enum delegation; (2) budget
is a single global gate — becomes a per-node `BudgetPolicy`/`BudgetExhaustedBehavior`
with a typed, isolated `StrategyOutcome` (`BudgetExhausted` never confused with
`Failed`); (3) the task list is an implicit linear chain — becomes an explicit
`Task.blockers` DAG with a ready-set walk, two-tier context, and failure cascade to
transitive dependents only. A `StrategyRef::{BuiltIn, Custom}` seam keeps built-ins
a closed serde enum (for resume/versioning/`max_steps()`) while allowing registered
opaque custom strategies.

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
**13 issues remain `status: queued`; none others started.**

**Examples suite — 12 of 13 landed, all four languages each.** Present under
`examples/{rust,typescript,python,go}/`:
`01-hello-agent`, `02-conversational-repl`, `03-tool-use`, `04-filesystem-agent`,
`05-custom-sandboxed-tool`, `06-web-research`, `07-memory`, `08-plan-execute`,
`09-self-verifying`, `10-hill-climbing`, `11-multi-agent`, **`12-cordyceps`
(#101, this loop)**. Each teaches one harness capability against a local Ollama
model. **Remaining example issues: #109** (`13-coding-agent` — batteries-not-
included coding-agent CLI; the final numbered example) and **#92** (observability
example — Phoenix/OTLP tracing).

**`12-cordyceps` capstone shipped (#101, this loop, all four languages):** a
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
  HillClimbing (#60). No loop strategy is stubbed.
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
completion** (label `loop-strategy-refactor`, all `status: queued`). Make
`LoopStrategy`, budget, and the task list compositional and mutually consistent,
applied byte-identically across all four languages where serialized. Work the
critical path: **117 → 119 → 120 → 123 → 124 → {125, 126} → 130 → 131**; #118
(task-graph schema) is **done**, and #121 (`SubagentTool` strategy param), #122
(`max_steps()`) remain parallel-grabbable immediately. Use the `/implement` skill per issue (Rust
reference + three parallel language agents + cross-language verifier). Honor the
maintainer's design choice — `RunStrategy` trait + `StrategyRef::Custom` escape
hatch — over the PRD's literal recursive-`run_strategy`-with-match sketch (A.4).
The success bar is #131: the `12-cordyceps` audit runs end-to-end as
`Ralph[PlanExecute[ReAct-plan, SelfVerifying[ReAct-worker]]]`, a runaway node
bounded by its own `BudgetPolicy` without cascading, and a paused tree resumes by
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
   non-`Assemble` methods) rather than delegating each explicitly.
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
10. **Local `main` is ahead of `origin/main` and unpushed** (`scope: debt`,
    transient) — `origin/main` is at `f63bf07`; local `main` is **13 commits ahead**:
    the five #117 commits (`0dd8b7a` rust, `a4eea8c` ts, `a3649d4` python, `771d66d`
    go + `b9a46bc` go fix), the four #118 commits (`7728a54` rust, `a68a337` python,
    `4ec76e2` go, `5d6c674` ts), two Rust-only `12-cordyceps` polish commits
    (`8bb7734`, `d65ae64`), and two `/close` bookkeeping commits (`60a9e83` project
    state, `cdf0c17` gitignore). **Needs a push.** The plan-execute scratch
    run-artifacts that were previously left untracked are now covered by a
    `workspace/*` wildcard in `examples/rust/08-plan-execute/.gitignore` (preserving
    the tracked `.gitkeep` + canonical `Async_Runtime_Comparison_Report.md`).
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

_(Former Deviations — HillClimbing/SelfVerifying/Ralph-git-log/MemoryTool/storage-
scope/sandbox-path/extras-mirror/Rust-dyn/compaction-tokens/observability-content
stubs — all resolved in prior loops. Former Deviation "#114 consult work unpushed"
— resolved; pushed at the start of this loop.)_

## Next Actions
[3-5 items max, highest priority first. /next surfaces item 1 as "work this next."]
1. **Push local `main` to `origin` (13 commits ahead, incl. #117 + #118).** Cheap,
   removes a growing divergence; two full cross-language features plus the Rust
   cordyceps polish are committed but unpushed. Do this before grabbing more work
   (see Deviation #10).
2. **#119 — recursive `LoopStrategy` + `RunStrategy` trait + `StrategyRef` (the
   keystone).** With #117 (budget value types) and #118 (task-graph schema) both
   landed, #119 is the next critical-path brick and unblocks most of the tree. Grab
   via `/implement`. #121 (`SubagentTool` strategy param) and #122 (`max_steps()`)
   are early parallel grabs alongside it.
3. **Critical path through the executor** — after #119: #120 (`ExecutionRegistry` +
   custom registry) → #123 (`StrategyOutcome` + `ExecutionContext`) → #124
   (per-variant `RunStrategy::run` impls — the big one). Then #125 (budget
   enforcement) and #126 (ready-set walk — consumes #118's `Task.blockers` schema)
   open up.
4. **Refactor finishers** — #127 (custom-strategy tracer), #128 (observability),
   #129 (`Continue` checkpoint), #130 (`HumanRequest::BudgetExhausted` HITL resume),
   then **#131** (cordyceps capstone — the success bar).
5. **Decide on the Rust-only `SubagentTool::with_stream` (Deviation #11)** — either
   file an issue to mirror it to TS/Python/Go (and the cordyceps example polish) or
   accept it as a Rust-reference-ahead experiment. **Still parked:** examples #109 /
   #92 + `web_search` #108/#110; harness gaps #115 (skill loading) and #116 (HITL
   child-consult resume — overlaps #130, consider folding in); correctness/safety
   #34 → #31 → #30 + docs #27/#35/#36.
