# PROJECT STATE
_Last updated: 2026-05-30 by /close — closed #60 (HillClimbing loop strategy) `status: complete`. This loop landed the **fifth and final loop strategy** across all four languages: the harness now runs **5 of 5** advertised strategies (ReAct + PlanExecute + SelfVerifying + Ralph + HillClimbing). #60 was the harness-wiring issue, not the trait-design one — the `MetricEvaluator` trait, `should_keep`, `MetricResult`/`MetricError`, `IterationStatus`, `ResultsEntry`, the four production evaluators, and `HaltReason::StagnationLimitReached` all already shipped under #23. #60 drives them from `StandardHarness::run`: baseline-first measurement (iteration 0, no agent turn) → agent turn → evaluate → keep/revert via payload-direction `should_keep`, plus a harness-written TSV results log at `.spore/results/{task_id}.tsv` and a misconfiguration guard. New public surface (mirrored ×4): `HaltReason::HillClimbingMisconfigured`, the `metric_evaluator` config field (injected like `verifier`/`vcs_provider`), the `hill_climbing_iteration` observability span. Seven spec ambiguities (git-reset seam, TSV float byte-identity at 6 decimals, TSV schema, direction source-of-truth, baseline semantics, misconfig halt, baseline-error) were pinned with the maintainer before fan-out. Cross-language verification PASS, byte-identical TSV confirmed, no divergences. Commits rust `5da525f`, python `e21c745`, ts `7d632bc`, go `d53e221` — all on `main`, pushed (`origin/main` == `d53e221`). **Both feature tracks (Track A tool/prompt + loop-strategy track) are now CLOSED.** Next: a maintainer fork — correctness/safety debt + docs cleanup vs. opening the protocol-integration track (#83–87)._

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.
**Everything is pushed — `origin/main` == local == `d53e221`.**

**The harness now runs all 5 of 5 advertised loop strategies end-to-end across
all four languages** (ReAct + PlanExecute + SelfVerifying + Ralph + HillClimbing).
The final one, **HillClimbing (#60)**, is an iterative-optimization loop:
iteration 0 establishes a baseline metric with no agent turn (`status: kept`),
then each iteration runs one agent turn → evaluates the metric → keeps or reverts
based on strict improvement (`should_keep`, payload `direction` authoritative).
`revert_on_no_improvement` discards the working tree via `git reset --hard HEAD`
through the sandbox seam (no commit-per-keep; harness never commits).
`max_stagnation` halts with `StagnationLimitReached` after N consecutive
non-improvements (counter resets on improvement); crash/timeout count as
non-improvements. The harness (not the agent) writes a TSV results log to
`.spore/results/{task_id}.tsv` — header + one row per iteration, 6-decimal float
formatting pinned for cross-language byte-identity, `metadata` excluded. Built on
#23's already-shipped `MetricEvaluator`/`should_keep`/`ResultsEntry` surface; #60
added only the harness wiring, the `metric_evaluator` config field, the
`HillClimbingMisconfigured` halt, and the `hill_climbing_iteration` span. The
`StrategyNotYetImplemented` stub is gone — **no loop strategy is stubbed anymore.**

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

Runnability: the harness runs **all five** advertised loop strategies (ReAct,
PlanExecute, SelfVerifying, Ralph, HillClimbing) end-to-end. The README and
`docs/harness-engineering-concepts.md` no longer overstate the *count* of working
strategies — but #27/#35 still track other spec-vs-code drift surfaced in review.

## Active Direction
The harness is **runnable** (#57), **debuggable** (#64/#65), has a working
**evaluation/feedback loop** (#26/#68), runs **all five** advertised loop
strategies (ReAct + PlanExecute + SelfVerifying + Ralph + HillClimbing,
#61/#58/#60), has a **clean, fully-pluggable, scope-aware persistence seam**
reaching into tools and exercised by a real `memory` tool (#73 + #76 + #75 + #78 +
#82), a **typed tool→caller escalation channel** (#80), a **complete standard tool
catalogue** (#81 + #82), and a **conditional prompt assembly engine** (#79) — all
across four languages. The bar remains **capability breadth and correctness**.

**Both feature tracks are now CLOSED** — Track A (tool/prompt architecture, #79 +
#80 + #81 + #82) and the loop-strategy track (#61 → #58 → #60). The harness has
its full advertised core capability surface. What remains is a genuine fork that
**needs a maintainer decision** before the next loop:

1. **Correctness/safety debt + docs cleanup** (the "correctness" half of the bar).
   Safety gates — #32 (Block-2 hash mismatch must *halt*, not just warn — a real
   correctness bug), #34 (`Mode::Yolo` / `SandboxProvider::None` behind a dangerous
   feature flag), #31 (SharedSession subagent context read-only by default), #30
   (memory distillation through the PendingReview gate). Docs — #27/#35/#36 (stop
   overstating / clarify spec; the strategy *count* is now accurate but other drift
   remains). Lower-glamour, all `scope: deferred` today, but tightens correctness
   now that breadth is done. **Default recommendation — this is concrete,
   in-repo, and needs no new architecture.**
2. **Open the protocol-integration track (#83–87)** — MCP (#83), A2A (#84), ACP
   (#85), AG-UI (#86), A2UI (#87), all still unlabeled. A new interop/ecosystem
   track that would broaden reach beyond the core runtime. **Needs a maintainer
   decision to commit to it and set ordering before triage.**

Storage remaining: SQL backends (#77, deferred), #88 deferred chunk providers,
#89 cross-session memory keying — all `scope: deferred`.

## Known Deviations
1. **Go outbox is not zero-dependency** — closing #50 added
   `go.opentelemetry.io/otel` + `otlptracegrpc` (v1.28.0) as blessed deps
   (accepted tradeoff, documented in `go/CONVENTIONS.md`). The durable JSONL path
   stays network-free.
2. **`task_list` / `todo_write` tool default persistence is no-op, not a file**
   (`scope: debt`, minor) — #75 retired the `.spore/task_list.json` sandbox path;
   the standalone tools persist via `RunStore`. With the library default
   `no_op()` storage, a standalone invocation persists nothing across processes.
   Durable standalone use requires wiring a real `StorageProvider`. Accepted
   tradeoff; no migration shim.
3. **v1 memory keying limitation (#78 Q7), filed as #89** (`scope: deferred`,
   future phase) — `MemoryStore` is still `SessionId`-keyed, so #82's
   `MemoryTool` can only address the current session; durable session-independent
   cross-session addressing is the v2 feature. Documented in each language's
   `MemoryTool` module header. No SQL backend yet either (#77).
4. **Go-specific divergences (#80 + #81 + #79 + #78 + #82 + #61 + #58 + #60)**
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
   the config struct (no builder setter), same as (i). (l) #60 required a
   consumer-side `MetricEvaluator` seam interface in the root `sporecore` package
   (with an `AsHarnessMetricEvaluator` bridge in `metric`) to avoid an import cycle
   — same pattern as `Verifier`/`VcsProvider`; `MetricEvaluator` config + the
   `HillClimbing` fields set on the struct, no builder setter. All
   wire/behavior-identical to the other three (verified).
5. **Test-placement divergences (#78/#82)** (benign) — the #78 R9 registry-seam
   test lives in `@spore/tools` (TS) / the eval suite (Python); #82 reused TS
   `@spore/tools` `tool-context-memory-seam.test.ts` and keeps the Python
   catalogue test in `spore_tools`. Behavior identical.
6. **#79 cross-language divergences — both verified benign.** (a)
   `ContextSources.composed_prompt` carries the full `ComposedPrompt` in
   Rust/Python but a narrowed stub in TS/Go (outcomes identical). (b) The Block-1
   hash is not byte-identical (Rust SipHash vs. FNV-1a elsewhere) — the
   intentional #24 decision; #79 fixtures assert no hash values, only condition
   booleans and ordered bucket id lists.
7. **`Custom` condition is invisible in fixtures by design** (#79) —
   `ChunkCondition::Custom(predicate)` is first-class in the API but serializes
   to null/absent; architects using it knowingly opt that chunk out of the
   byte-identical cross-language contract.

_(Former Deviation — HillClimbing loop strategy stub, #60 (was Deviation #1: harness ran only 4/5 strategies) — **resolved** this loop; HillClimbing runs end-to-end across all four, harness now at 5/5, stub removed.)_
_(Former Deviation — push blocked on local SSH auth (13 unpushed commits) — **resolved**; SSH fixed, all pushed.)_
_(Former Deviation — Ralph git-log reload deferred to v2 (#58 B4) — **resolved** by the `VcsProvider` seam; Ralph now reloads all three spec'd sources.)_
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
1. **MAINTAINER DECISION — pick the next track.** Both feature tracks are closed,
   so there is no single forced next issue. Choose: **(a) correctness/safety debt +
   docs** (concrete, in-repo, no new architecture — recommended) or **(b) open the
   protocol-integration track (#83–87)** (new interop scope). `/next` will surface
   this fork until a track is chosen.
2. **Correctness/safety gates** (track-a candidate) — **#32 first** (Block-2 hash
   mismatch must *halt* mid-session, not just warn — a genuine correctness bug),
   then #34 (`Mode::Yolo`/`SandboxProvider::None` behind a dangerous feature flag),
   #31 (SharedSession subagent context read-only by default), #30 (memory
   distillation through the PendingReview gate). All `scope: deferred` today; relabel
   `status: queued` if this track is chosen.
3. **Docs cleanup** (track-a candidate) — #27 (README: strategy count is now
   accurate at 5/5, but sweep remaining spec-vs-code drift), #35
   (`harness-engineering-concepts.md` drift), #36 (E2B data-residency/privacy doc).
4. **Protocol-integration track (#83–87)** (alternative to 2/3) — MCP (#83), A2A
   (#84), ACP (#85), AG-UI (#86), A2UI (#87), all unlabeled. Triage/label and set
   ordering **only if the maintainer commits to this track.**
5. **Deferred storage/memory phases** — #77 (SQL backends), #88 (deferred chunk
   providers), #89 (cross-session memory keying). All `scope: deferred`.
