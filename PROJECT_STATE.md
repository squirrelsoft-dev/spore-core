# PROJECT STATE
_Last updated: 2026-06-06 by /close — closed #101 (`12-cordyceps` autonomous-agent capstone example) `status: complete`. Built across all four languages (Rust ref `4bb60a6`, TS `aa4cde0`, Python `408d003`, Go `ea755a8`), **zero core-harness change**, cross-language verifier PASS on 7 parity checks, gates green all four. The example loads its `audit` skill at runtime via the `GuideRegistry` **architect-side** (a `SkillCatalog` scanning `.spore/skills/` + a custom skill-injecting `ContextManager` wrapping `StandardCompactionAdapter` + a `load_skill` tool over `run_store`), because the harness's *structural* skill-injection path is NOT live-wired — the live loop's `assemble` is a pass-through; the rich `StandardContextManager::assemble` that would inject skills/chunks/memory is bypassed pending #7. Two harness gaps surfaced and filed: **#115** (bake skill loading into the harness) and **#116** (#114 HITL can't resume a worker's in-flight consult). ⚠️ **Local `main` (`ea755a8`) is 4 commits ahead of `origin/main` (`f1cc531`) — the #101 work is committed locally but NOT pushed.**_

_**Direction note:** Active Direction (examples suite) holds. The numbered suite is now **12 of 13** in. Remaining: #109 (`13-coding-agent`) completes the 1–13 suite, then #92 (observability example). `web_search` hardening (#108/#110) follows. The two #101-spawned harness gaps (#115 skill loading, #116 HITL child-consult resume) are real but not blocking the suite. The correctness/safety gates (#34 → #31 → #30) + docs (#27/#35/#36) remain parked pending an explicit maintainer call._

## Current State
spore-core is a language-agnostic agentic harness runtime with a **complete core
capability surface**, demonstrated through a numbered **examples suite** built
across all four targets: Rust (reference), TypeScript, Python, Go.
**⚠️ Local `main` (`ea755a8`) is 4 commits ahead of `origin/main` (`f1cc531`) — the
#101 capstone work is committed locally but NOT pushed.**

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
**Build out and harden the numbered examples suite across all four languages** —
each example a self-contained, runnable demonstration of one harness capability,
with four-language parity and small-context-window friendliness so they run on
local Ollama models. The suite is the current product surface. **12 of 13
examples are in;** the remaining work is the last example (#109 `13-coding-agent`)
plus the observability example (#92), and hardening the `web_search` tool the
research-style examples depend on (#108/#110). The two harness gaps the #101
capstone surfaced — **#115** (first-class skill loading; the live-injection path
is currently bypassed) and **#116** (#114 HITL child-consult resume) — are real
and worth scheduling, but neither blocks finishing the suite.

**Parked track (needs a maintainer call):** correctness/safety gates
#34 → #31 → #30 and docs #27/#35/#36 were named active on 2026-05-30 but never
started; the examples suite took priority. Pick these back up explicitly when the
suite is done, or confirm they stay parked. Larger new feature issues — #113
(spore-lsp), #107 (PromptEngineeringAgent), #106 (MicroVMSandboxProvider) — and
the protocol-integration track (#83–87) remain unscheduled. Storage follow-ups
#77/#88/#89 stay `scope: deferred`. **#7** (ContextManager migration) would
live-wire the rich `assemble` — and is the proper home for #115's skill injection
and the #32 cache halts.

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
10. **Local `main` is ahead of `origin/main` (unpushed)** (transient, action
    required) — the four #101 commits (`4bb60a6`, `aa4cde0`, `408d003`, `ea755a8`)
    are on local `main` but not pushed; `origin/main` is at `f1cc531` (4 behind).
    Push to clear. (The #114 commits from last loop ARE pushed.) Also still
    untracked: four scratch run-artifact `.txt`/`.md` files under
    `examples/rust/08-plan-execute/workspace/` — intentionally left out of source.

_(Former Deviations — HillClimbing/SelfVerifying/Ralph-git-log/MemoryTool/storage-
scope/sandbox-path/extras-mirror/Rust-dyn/compaction-tokens/observability-content
stubs — all resolved in prior loops. Former Deviation "#114 consult work unpushed"
— resolved; pushed at the start of this loop.)_

## Next Actions
[3-5 items max, highest priority first. /next surfaces item 1 as "work this next."]
1. **Push `main` to `origin`** — local `main` (`ea755a8`) is 4 commits ahead with
   the #101 capstone; `origin/main` is at `f1cc531`. `git push origin main`.
   (One-line task, do first.)
2. **#109 — `13-coding-agent` example** — the final numbered example, a
   batteries-not-included coding-agent CLI; **completes the 1–13 suite** across all
   four languages. The top feature item; grab next.
3. **#92 — observability example** — wire Phoenix/OTLP tracing and show structured
   trace output; demonstrates the already-shipped #64/#65 observability stack.
4. **#108 + #110 — `web_search` hardening** — #108 (attach auth headers / query
   params) and #110 (normalize cited sources across Brave/Tavily/SearXNG). Both
   surfaced by the research-dependent examples (06, 11); make them robust across
   providers.
5. **Harness gaps from #101 + parked-track decision** — **#115** (bake skill
   loading into the harness; the live-injection path is bypassed pending #7) and
   **#116** (#114 HITL child-consult resume) are both `status: queued`; schedule
   when the suite work allows. Separately, the parked correctness/safety track
   (#34 → #31 → #30) + docs (#27/#35/#36) still needs an explicit maintainer call:
   pick back up after the suite, or confirm parked.
