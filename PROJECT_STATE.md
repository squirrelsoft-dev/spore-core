# PROJECT STATE
_Last updated: 2026-05-30 by /close — re-closed #58 (Ralph loop strategy + v2 `VcsProvider` seam) `status: complete`. This loop landed the **v2 `VcsProvider` seam** that closes the original B4 git-log gap: Ralph now reloads all three spec'd filesystem sources (progress, feature list, **opt-in git log**). A `VcsProvider` trait/interface (`log`/`status`) with two impls — `GitVcsProvider` (shells `git` **through** `SandboxProvider::execute_command`, never a raw spawn) and `FixtureVcsProvider` (deterministic test double, returns seeded strings verbatim — this is what makes git-log reload hermetically fixture-replayable across four languages). Wired as `vcs_provider: Option<…>` on the Ralph config surface (alongside `max_resets`), default **none** → git context omitted, preserving v1 byte-for-byte; when set, the reload phase injects a delimited `Recent VCS history:` block. **PUSH BLOCKER RESOLVED** — SSH auth was fixed and all 13 backlog commits (#61 ×4, Ralph v1 ×4, VcsProvider ×4 + 1 Go fix) are pushed; `origin/main` == local == `bfeba21`. The harness runs **4 of 5 loop strategies**; only HillClimbing (#60) remains stubbed. Next: **#60** to reach 5/5._

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.
**Everything is pushed — `origin/main` == local == `bfeba21`** (the prior
multi-loop unpushed-commit backlog is cleared).

**The Ralph loop strategy (#58) is complete end-to-end across all four
languages, including its v2 `VcsProvider` seam.** Ralph is a
multi-context-window continuation loop: each outer iteration is one context
window — a FRESH `SessionState` (no message carryover) re-seeded with the
instruction + state reloaded from `.spore/`, then a bounded inner ReAct sub-loop.
The harness fires Stop hooks on `FinalResponse`; Ralph registers a Stop hook
that reads `.spore/progress.json` and returns `Block{reason}` while tasks remain
(intercepting the exit → resetting the window) or `Continue` when all complete
(terminating with success). Budgets/usage fold across all windows; each reset
gets a distinct session id. Terminal `HaltReason::RalphCompletionUnmet {
iterations, last_reason }` (peer of `SelfVerifyExhausted`) when `max_resets`
(default 3) exhausts with tasks incomplete. This takes the harness to **4 of 5
loop strategies** (ReAct + PlanExecute + SelfVerifying + Ralph).

v1 forks (pinned in Rust before fan-out): **B1** drive Ralph off the Stop hook
(#69), no new completion-check config surface, deprecated `CompletionCheck`
(landed by #43) left untouched; **B2** canonical `.spore/`-prefixed paths, with
`FeatureListCheck`'s default path updated to `.spore/feature_list.json` (one
source of truth); **B3** `max_resets` config field, `LoopStrategy::Ralph` stays
payload-free; **B4** git-log reload originally deferred — **now resolved by the
v2 seam this loop.**

**v2 `VcsProvider` seam (this loop)** — a thin VCS abstraction Ralph calls
instead of shelling out directly, mirroring how `SandboxProvider` abstracts
shell/filesystem access. `VcsProvider` trait (`log(VcsLogArgs)` + `status()`),
`VcsLogArgs { max_entries, since_ref?, format? }`, a `VcsError`, plus two impls:
`GitVcsProvider` (shells `git log`/`git status` **through**
`SandboxProvider::execute_command` — verified in all four, never a raw
`std::process`/`child_process`/`subprocess`/`os/exec`; argv mapping
`max_entries → -n N`, `format → --format=…`, `since_ref → <ref>..`) and
`FixtureVcsProvider` (deterministic test double, returns seeded strings verbatim,
no process spawning). Wired as `vcs_provider: Option<Arc<dyn VcsProvider>>` on the
Ralph config surface + builder setter, default `None` → git context omitted
(preserves v1 behavior byte-for-byte). When set, the reload phase injects a
delimited `Recent VCS history:` block into the next window's seed. Fixture
`fixtures/harness/ralph.json` gained an optional `vcs_log` field and a 7th case
`vcs_log_injected_across_reset`, exercised in all four replay suites; the
original 6 cases pass unchanged. Cross-language verification PASS; one latent Go
argv divergence (`-n` emitted conditionally at `max_entries==0`) was caught and
fixed (`bfeba21`) so all four match the Rust reference. Commits: Rust `55f45e8`,
TS `22ef61f`, Python `95dd1cc`, Go `6fcebae` + fix `bfeba21`. Test deltas: Rust
802 lib (+6), TS core 960 (+6), Python 996 (+7), Go all green incl. `-race` (+8
+contract test).

**#61 — the SelfVerifying loop strategy** runs end-to-end across all four. A
`run_self_verifying` method orchestrating a build ReAct sub-loop → a fresh
evaluator run (fresh `SessionId`, read-only sandbox, `role-evaluator` chunk) →
the existing `Verifier` (#44), reusing the failure-reason → user-message
injection path. Forks: D1 bespoke strategy; D2 `evaluator_agent` config field
(#70 defaulting) + `verifier` oracle; D3 `Verifier::max_iterations()` (default 3)
caps the round-trip; D4 peer halts `SelfVerifyExhausted`/`SelfVerifyMisconfigured`.
Reusable seam: the **`ReadOnlySandbox` decorator** (all four). Rust `c4f607a`,
TS `856736b`, Python `cc2bab0`, Go `f5b2f21`.

**#82 — the scope-aware `memory` tool** ships across all four, built on #78's
scoped `MemoryStore` + `ToolContext` memory seam. Single `operation`-discriminated
tool, `scope` explicit on both ops, `Local` rejected before any storage access.
Architectural outcome (D2): `get_memories_merged` promoted onto the `MemoryStore`
trait itself — one merge impl per language. **Known v1 limit (#78 Q7): memory
stays `SessionId`-keyed; session-independent cross-session addressing is filed as
#89.**

**#78 (the storage seam #82 built on)** shipped `StorageScope { User, Project,
Local }`, fixture-pinned `WorkspaceId` derivation, the scoped `MemoryStore`,
`(domain, scope) → backend` routing, user-scope workspace partitioning, and the
`ToolContext` memory-store field.

**Track A — tool/prompt architecture — is DONE** (#79 + #80 + #81 + #82): the
standard tool catalogue (#81, three tiers), the prompt assembly engine (#79), and
the typed tool→caller escalation channel (#80), all across four languages.
`RemoteChunkProvider` + scope-aware `FileSystemChunkProvider` deferred to **#88**.

**The persistence layer is clean** (#73 + #76 + #75 + #78 + #82):
`StorageProvider` with per-domain and per-scope composite routing;
`plan_execute`/`task_list`/`todo_write` persist through `RunStore`; the `memory`
tool through the scoped `MemoryStore` seam (merged-read single-sourced on the
trait, #82 D2).

Foundation: the harness is **runnable** (#57 — shared e2e CLI driving the ReAct
loop against Ollama, hermetic suite + live `llama3.2`) and **debuggable**
(#64/#65 — GenAI content capture, Arize Phoenix trace viewer). Working
**evaluation/feedback loop** (#26 EvalHarness; #68 typed `Span` accessors).
Earlier: observability stack (#42/#49/#50/#33), `OllamaModelInterface` +
capability guard (#41), `StandardCompactionAdapter` + verify→retry→warn loop
(#55/#46), `StandardContextManager`/verifiers (#29), KeyTermVerifier (#47).

Known runnability limit: the harness runs **ReAct, PlanExecute, SelfVerifying,
and Ralph** end-to-end (4 of 5). Only `HillClimbing` (#60) still returns the
generic `HaltReason::StrategyNotYetImplemented` — so the README and
`docs/harness-engineering-concepts.md` still overstate by one strategy.

## Active Direction
The harness is **runnable** (#57), **debuggable** (#64/#65), has a working
**evaluation/feedback loop** (#26/#68), runs **four** of five advertised loop
strategies (ReAct + PlanExecute + SelfVerifying + Ralph, #61/#58 — Ralph now
complete with full git-log reload), has a **clean, fully-pluggable, scope-aware
persistence seam** reaching into tools and exercised by a real `memory` tool (#73
+ #76 + #75 + #78 + #82), a **typed tool→caller escalation channel** (#80), a
**complete standard tool catalogue** (#81 + #82), and a **conditional prompt
assembly engine** (#79) — all across four languages. The bar remains **capability
breadth and correctness**.

**Track A — tool/prompt architecture — is CLOSED.** The **loop-strategy track**
(#61 → #58) is **one issue from done**:
1. **Finish the last loop strategy — #60 (HillClimbing).** With #58 fully
   complete, this is the **only remaining gap between advertised and actual
   *core* capability** (the docs claim five strategies; one is still a stub).
   Lands the harness at 5 of 5 and lets #27/#35 docs stop overstating. Reuse the
   seams #61/#58 exercised — the `run_react_inner` sub-loop executor, the
   failure-reason→user-message injection path, `RunStore`, the #69 Stop hooks, the
   #70 alternate-agent defaulting pattern, the `max_resets`-style config-driven
   outer cap, and (if it needs to read external state) the new `VcsProvider`-style
   provider-seam pattern. Resolve spec forks in Rust before fan-out. **Default
   recommendation — finish the track.**
2. **Protocol integrations (#83–87)** — MCP (#83), A2A (#84), ACP (#85), AG-UI
   (#86), A2UI (#87) remain unlabeled. A potential new interop/ecosystem track.
   **Needs a maintainer decision before triage/prioritization.**
3. **Correctness/safety debt + docs cleanup** — #30/#31/#32/#34 (safety gates)
   and #27/#35/#36 (docs stop overstating). #27 is now only one strategy from
   accurate (4/5). Lower-glamour but tightens the "correctness" half of the bar.

Storage remaining: SQL backends (#77, deferred), #88 deferred chunk providers,
#89 cross-session memory keying — all `scope: deferred`.

## Known Deviations
1. **One of five loop strategies is still a stub** — the README and
   `docs/harness-engineering-concepts.md` advertise five loop strategies;
   **ReAct, PlanExecute, SelfVerifying (#61), and Ralph (#58) run end-to-end**,
   but `HillClimbing` still returns `HaltReason::StrategyNotYetImplemented` at
   `rust/crates/spore-core/src/harness.rs`. Tracked: #60 (`scope: deferred`).
   #57's scenario suite is intentionally ReAct-only.
2. **Go outbox is not zero-dependency** — closing #50 added
   `go.opentelemetry.io/otel` + `otlptracegrpc` (v1.28.0) as blessed deps
   (accepted tradeoff, documented in `go/CONVENTIONS.md`). The durable JSONL path
   stays network-free.
3. **`task_list` / `todo_write` tool default persistence is no-op, not a file**
   (`scope: debt`, minor) — #75 retired the `.spore/task_list.json` sandbox path;
   the standalone tools persist via `RunStore`. With the library default
   `no_op()` storage, a standalone invocation persists nothing across processes.
   Durable standalone use requires wiring a real `StorageProvider`. Accepted
   tradeoff; no migration shim.
4. **v1 memory keying limitation (#78 Q7), filed as #89** (`scope: deferred`,
   future phase) — `MemoryStore` is still `SessionId`-keyed, so #82's
   `MemoryTool` can only address the current session; durable session-independent
   cross-session addressing is the v2 feature. Documented in each language's
   `MemoryTool` module header. No SQL backend yet either (#77).
5. **Go-specific divergences (#80 + #81 + #79 + #78 + #82 + #61 + #58)**
   (`scope: debt`, minor, documented on the issues) — (a) local `Mode` newtype;
   (b) 3-state `TerminalOutcome`; (c) `StandardTool` in root `sporecore`
   type-aliased into `tools`; (d) `sendMessageToolName` duplicated in
   `harness.go`; (e) `abort` tool enforces required `reason` explicitly; (f)
   `promptassembly` owns a self-contained `HarnessBuilder`; (g)
   `ToolContext.MemoryStore` is an opaque marker interface asserted back to
   `storage.MemoryStore`; (h) #82 D2 single-source merge is an exported
   `storage.MergeMemories` (Go interfaces can't carry defaults); (i) #61
   `Verifier`/`EvaluatorAgent` config set directly on the struct with no builder
   setters (Go has no `PlannerAgent` setter either); (j) `role-evaluator` chunk
   is a `RoleEvaluatorChunk` constant in `sporecore` (can't import the registry);
   (k) the Ralph Stop hook is registered in `NewStandardHarness`, and
   `MaxResets`/the Ralph types / the v2 `VcsProvider` field are set directly on
   the config struct (no builder setter), same as (i). All wire/behavior-identical
   to the other three (verified).
6. **Test-placement divergences (#78/#82)** (benign) — the #78 R9 registry-seam
   test lives in `@spore/tools` (TS) / the eval suite (Python); #82 reused TS
   `@spore/tools` `tool-context-memory-seam.test.ts` and keeps the Python
   catalogue test in `spore_tools`. Behavior identical.
7. **#79 cross-language divergences — both verified benign.** (a)
   `ContextSources.composed_prompt` carries the full `ComposedPrompt` in
   Rust/Python but a narrowed stub in TS/Go (outcomes identical). (b) The Block-1
   hash is not byte-identical (Rust SipHash vs. FNV-1a elsewhere) — the
   intentional #24 decision; #79 fixtures assert no hash values, only condition
   booleans and ordered bucket id lists.
8. **`Custom` condition is invisible in fixtures by design** (#79) —
   `ChunkCondition::Custom(predicate)` is first-class in the API but serializes
   to null/absent; architects using it knowingly opt that chunk out of the
   byte-identical cross-language contract.

_(Former Deviation — push blocked on local SSH auth (13 unpushed commits) — **resolved** this loop; SSH fixed, all pushed, `origin/main` == `bfeba21`.)_
_(Former Deviation — Ralph git-log reload deferred to v2 (#58 B4) — **resolved** this loop by the `VcsProvider` seam; Ralph now reloads all three spec'd sources.)_
_(Former Deviation — SelfVerifying strategy stub, #61 — **resolved** in #61.)_
_(Former Deviation — `MemoryTool` deferred/blocked — **resolved** in #82.)_
_(Former Deviation — storage scope + partitioning, #78 — **resolved** in #78.)_
_(Former Deviation — tool stuck on sandbox path — **resolved** in #75.)_
_(Former Deviation — extras persistence mirror unmigrated — **resolved** in #76.)_
_(Former Deviation — Rust Agent dyn-compatibility / `BoxFut` — **resolved** in #45.)_
_(Former Deviation — compaction `tokens_reclaimed = 0` — **resolved** in #57.)_
_(Former Deviation — observability captured no message content — **resolved** in #64.)_

## Next Actions
[3-5 items max, highest priority first. Each references a GH issue # where
possible. /next surfaces item 1 as "work this next."]
1. **#60 — HillClimbing loop strategy** (`scope: deferred`). The last strategy
   stub; lands the harness at 5 of 5 and closes the final advertised-vs-actual
   core gap. Reuse the seams #58/#61 exercised — `run_react_inner` sub-loop, the
   failure-reason→user-message injection path, `RunStore`, #69 Stop hooks, #70
   alternate-agent defaulting, a `max_resets`-style config-driven outer cap, and
   the new `VcsProvider`-style provider-seam pattern if it needs external state.
   Resolve spec forks in Rust before fan-out. **Work this next.**
2. **Docs + correctness debt** — #27 (README strategy count: now 4/5, finish #60
   then update to 5/5), #35 (`harness-engineering-concepts.md` drift), #36
   (observability docs); #30/#31/#32/#34 (safety gates). Tightens the
   "correctness" half of the bar once #60 lands.
3. **#83–87 — protocol integrations** (unlabeled) — MCP (#83), A2A (#84), ACP
   (#85), AG-UI (#86), A2UI (#87). Triage/label as a new interop track or
   `scope: deferred` — **needs a maintainer decision** before prioritization.
4. **Deferred storage/memory phases** — #77 (SQL backends), #88 (deferred chunk
   providers), #89 (cross-session memory keying). All `scope: deferred`.
