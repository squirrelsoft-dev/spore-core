# PROJECT STATE
_Last updated: 2026-05-30 by /close — closed #58 (Ralph loop strategy) `status: complete`. **#58 brings the harness to 4 of 5 loop strategies running end-to-end** (ReAct + PlanExecute + SelfVerifying + Ralph) — only HillClimbing (#60) remains stubbed. Four spec forks were resolved & pinned in Rust before fan-out: **B1** Ralph drives off the **Stop hook (#69)** (NOT the deprecated `CompletionCheck`, which #43 already landed and #69 superseded) — a Stop hook reads `.spore/progress.json`, returns `Block{reason}` while tasks remain (loop resets the context window) and `Continue` when complete (loop terminates); **B2** canonical paths `.spore/progress.json` + `.spore/feature_list.json`, and the pre-`.spore/` `FeatureListCheck::new()` default was updated to `.spore/feature_list.json` so the two agree; **B3** new `max_resets` config field (default 3, outer-loop cap, independent of `max_turns`), `LoopStrategy::Ralph` stays payload-free per the #61 precedent; **B4** git-log reload deferred to v2 (hermetic fixture replay is non-negotiable) — v1 reloads only progress + feature_list. New terminal `HaltReason::RalphCompletionUnmet { iterations, last_reason }` (peer of `SelfVerifyExhausted`). **PUSH STILL BLOCKED** — local SSH auth is failing (`sign_and_send_pubkey: signing failed for RSA`), so `origin/main` is stale at `4f2f447` (post-#82); **the four #61 commits AND the four #58 commits are all local-only**. Fix the SSH agent and push before the next loop._

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.

**#58 just landed — the Ralph loop strategy runs end-to-end across all four
languages**, taking the harness to **4 of 5 loop strategies** (ReAct +
PlanExecute + SelfVerifying + Ralph). It was strategy-wiring on top of the
existing Stop-hook seam (#69): a `run_ralph` outer continuation loop replaced the
`StrategyNotYetImplemented` stub. Each outer iteration is **one context window** —
a FRESH `SessionState` (no message carryover) re-seeded with the instruction +
state reloaded from `.spore/`, then a bounded inner ReAct sub-loop. The harness
already fires Stop hooks on `FinalResponse`; Ralph registers a Stop hook that
reads `.spore/progress.json` and returns `Block{reason}` while tasks remain
(intercepting the exit → resetting the window) or `Continue` when all complete
(terminating with success). The hook is **inert** (returns `Continue`) when no
progress file exists, so ReAct/PlanExecute/SelfVerifying runs are unaffected.
Budgets/usage fold across all windows; each reset gets a distinct session id for
traceability. Terminal `HaltReason::RalphCompletionUnmet { iterations,
last_reason }` (peer of `SelfVerifyExhausted`) when `max_resets` exhausts with
tasks still incomplete.

Four spec forks were pinned in the Rust reference before fan-out — the issue's
stated scope ("define and implement the `CompletionCheck` trait") contradicted
the codebase, since `CompletionCheck` was already landed by #43 and is now
`#[deprecated]` in favor of the Stop hook (#69): **B1** drive Ralph off the Stop
hook, no new completion-check config surface, deprecated `CompletionCheck` left
untouched; **B2** canonical `.spore/`-prefixed paths, with `FeatureListCheck`'s
default path updated to `.spore/feature_list.json` in the same pass (one source
of truth); **B3** new `max_resets` config field (default 3, outer-loop cap,
independent of `max_turns`), `LoopStrategy::Ralph` stays payload-free per #61;
**B4** git-log reload deferred to v2 (a documented follow-up in the issue) — v1
reloads only the two deterministic files the strategy itself writes. New fixture
`fixtures/harness/ralph.json` (6 cases, byte-for-byte identical terminal kind +
iteration count all four). Cross-language verification PASS, no divergences (the
Go agent's flagged iterations-count caveat was confirmed a doc note, not a
behavioral difference — every implementation runs exactly one agent turn per
window). Rust `927cc57` (874 passed, +9), TS `0e1f407` (core 953, +14: 8 unit +
6 replay), Python `33f19e8` (989 passed, +12), Go `d704e6b` (+14: 13 unit +
6-case replay, vet/gofmt clean). The catalogue test now lists only `HillClimbing`
as not-yet-implemented in all four. **These four commits are on `main` locally
but NOT pushed** (see push blocker below).

**#61 (the prior loop) — the SelfVerifying loop strategy runs end-to-end across
all four languages.** A `run_self_verifying` method orchestrating a build ReAct
sub-loop → a fresh evaluator run (fresh `SessionId`, read-only sandbox,
`role-evaluator` chunk) → the existing `Verifier` (#44), reusing the
failure-reason → user-message injection path to resume the build loop on FAIL.
Forks pinned: D1 bespoke strategy (not a Stop-hook adapter — the evaluate phase
is a sibling *run*); D2 a new `evaluator_agent` config field following #70's
`planner_agent` defaulting + a `verifier` oracle field; D3
`Verifier::max_iterations()` (default 3) caps the build↔evaluate round-trip; D4
two peer `HaltReason` variants `SelfVerifyExhausted`/`SelfVerifyMisconfigured`.
New reusable seam: a **`ReadOnlySandbox` decorator** (all four) rejecting the
seven mutating tool names with a recoverable `ReadOnlyViolation`. Fixture
`fixtures/harness/self_verifying.json` (7 cases). Rust `c4f607a`, TS `856736b`,
Python `cc2bab0`, Go `f5b2f21` — **also local-only, NOT pushed.**

**#82 — the scope-aware `memory` tool ships across all four languages**, built on
#78's scoped `MemoryStore` API + `ToolContext` memory seam. A single
`operation`-discriminated `memory` tool, `scope` explicit on both ops, in the
`coding_set`/`full_set` presets. `Local` scope rejected on both ops before any
storage access. **Architectural outcome (D2): `get_memories_merged` was promoted
onto the `MemoryStore` trait/interface/protocol itself** — one merge
implementation per language (default trait method in Rust, shared helper
elsewhere, exported `MergeMemories` in Go). Rust `b7afbf5`, TS `5dd91c9`, Python
`625b090`, Go `829780e`. Fixture `fixtures/tools/memory.json`. **Known v1 limit
(#78 Q7): memory stays `SessionId`-keyed — session-independent cross-session
addressing is the v2 follow-up, filed as #89.**

**#78 (the storage seam #82 built on)** shipped `StorageScope { User, Project,
Local }`, a fixture-pinned `WorkspaceId` derivation, the scoped `MemoryStore`,
`(domain, scope) → backend` routing, user-scope workspace partitioning, and the
`ToolContext` memory-store field. (Rust `c0c2cbd`, TS `a3e2b1c`, Python
`f9acc84`, Go `078c5a7`.)

**Track A — tool/prompt architecture — is DONE** (#79 + #80 + #81 + #82). The
standard tool catalogue (#81, three tiers), the prompt assembly engine (#79,
`ChunkCondition`/`PromptChunk`/`ChunkProvider`/`ContextSourcesBuilder`), and the
typed tool→caller escalation channel (#80, `HarnessSignal` via
`ToolOutput::Escalate`) all landed end-to-end across four languages.
`RemoteChunkProvider` + scope-aware `FileSystemChunkProvider` deferred to **#88**.

**The persistence layer is clean** (#73 + #76 + #75 + #78 + #82):
`StorageProvider` abstraction with per-domain and per-scope composite routing;
`plan_execute`/`task_list`/`todo_write` persist through `RunStore`; the `memory`
tool reads/writes through the scoped `MemoryStore` seam, merged-read
single-sourced on the trait itself (#82 D2).

Foundation in place: the harness is **runnable** (#57 — shared e2e CLI driving
the ReAct loop against Ollama, hermetic suite + live against `llama3.2`) and
**debuggable** (#64/#65 — GenAI-convention content capture, Arize Phoenix
LLM-trace viewer). Working **evaluation/feedback loop** (#26 EvalHarness; #68
typed `Span` accessors). Earlier: observability stack (#42/#49/#50/#33),
`OllamaModelInterface` + capability guard (#41), `StandardCompactionAdapter` +
verify→retry→warn loop (#55/#46), `StandardContextManager`/verifiers (#29),
KeyTermVerifier (#47).

**origin/main is stale at `4f2f447`** (post-#82). **The four #61 commits AND the
four #58 commits — eight commits total — are local-only and NOT pushed**, because
local SSH auth to GitHub is failing (`sign_and_send_pubkey: signing failed for
RSA "/Users/sbeardsley/.ssh/sbeardsley"`). All four language suites pass locally.
Fix the SSH agent (`ssh-add`, or unlock the key) and push before the next loop.

Known runnability limits: the harness runs **ReAct, PlanExecute, SelfVerifying,
and Ralph** end-to-end (4 of 5). Only `HillClimbing` (#60) still returns the
generic `HaltReason::StrategyNotYetImplemented` — so the README and
`docs/harness-engineering-concepts.md` still overstate by one strategy (1 of 5
remains a stub, down from 2).

## Active Direction
The harness is **runnable** (#57), **debuggable** (#64/#65), has a working
**evaluation/feedback loop** (#26/#68), runs **four** of five advertised loop
strategies (ReAct + PlanExecute + SelfVerifying + Ralph, #61/#58), has a
**clean, fully-pluggable, scope-aware persistence seam** reaching into tools and
exercised by a real `memory` tool (#73 + #76 + #75 + #78 + #82), a **typed
tool→caller escalation channel** (#80), a **complete standard tool catalogue**
(#81 + #82), and a **conditional prompt assembly engine** (#79) — all across four
languages. The bar remains **capability breadth and correctness**.

**Track A — tool/prompt architecture — is CLOSED.** The **loop-strategy track**
(started by #61, continued by #58) is one issue from done:
1. **Finish the last loop strategy — #60 (HillClimbing).** With #58 done, this is
   the **only remaining gap between advertised and actual *core* capability** (the
   docs claim five strategies; one is still a stub). Lands the harness at 5 of 5
   and lets #27/#35 docs stop overstating. Reuse the seams #61/#58 exercised —
   the `run_react_inner` sub-loop executor, the failure-reason→user-message
   injection path, `RunStore`, the #69 Stop hooks, the #70 alternate-agent
   defaulting pattern, and the `max_resets`-style config-driven outer cap. Resolve
   spec forks in Rust before fan-out, same as #58/#61. **Default recommendation —
   finish the track.**
2. **Protocol integrations (#83–87)** — MCP (#83), A2A (#84), ACP (#85), AG-UI
   (#86), A2UI (#87) remain on the board unlabeled. A potential new
   interop/ecosystem track rather than core-capability depth. **Needs a maintainer
   decision before triage/prioritization.**
3. **Correctness/safety debt + docs cleanup** — #30/#31/#32/#34 (safety gates)
   and #27/#35/#36 (docs stop overstating). #27 is now only one strategy from
   accurate (4/5). Lower-glamour but tightens the "correctness" half of the bar.

Storage remaining: SQL backends (#77, deferred), #88 deferred chunk providers,
and #89 cross-session memory keying — all `scope: deferred`.

## Known Deviations
1. **Eight commits unpushed; push blocked on local SSH auth** (release hygiene,
   **active blocker**) — the four #61 commits
   (`c4f607a`/`856736b`/`cc2bab0`/`f5b2f21`) and the four #58 commits
   (`927cc57`/`0e1f407`/`33f19e8`/`d704e6b`) are on `main` locally; `origin/main`
   is stale at `4f2f447`. `git fetch`/`push` fail with `sign_and_send_pubkey:
   signing failed for RSA "/Users/sbeardsley/.ssh/sbeardsley" from agent`. The
   SSH agent can't sign with the key — re-add/unlock it (`ssh-add ~/.ssh/...`)
   then push. All four suites pass locally. **Push before the next loop so board,
   CI, and `main` agree.**
2. **One of five loop strategies is still a stub** (down from two) — the README
   and `docs/harness-engineering-concepts.md` advertise five loop strategies;
   **ReAct, PlanExecute, SelfVerifying (#61), and Ralph (#58) run end-to-end**,
   but `HillClimbing` still returns `HaltReason::StrategyNotYetImplemented` at
   `rust/crates/spore-core/src/harness.rs`. Tracked: #60 (`scope: deferred`).
   #57's scenario suite is intentionally ReAct-only. (#58 **resolved** — see
   Current State.)
3. **Ralph git-log reload deferred to v2 (#58 B4)** (`scope: deferred`, future
   phase) — the Ralph spec lists git log as a reload source between context
   windows, but reading real git history (or a fixture standing in for it) breaks
   hermetic cross-language fixture replay. v1 reloads only `.spore/progress.json`
   + `.spore/feature_list.json` (deterministic files the strategy writes). Add
   git history as a reload source once there's a sandbox VCS-read abstraction all
   four languages can implement identically and hermetically. Documented in the
   #58 close comment; no separate tracking issue filed yet.
4. **Go outbox is not zero-dependency** — closing #50 added
   `go.opentelemetry.io/otel` + `otlptracegrpc` (v1.28.0) as blessed deps
   (accepted tradeoff, documented in `go/CONVENTIONS.md`). The durable JSONL path
   stays network-free.
5. **`task_list` / `todo_write` tool default persistence is no-op, not a file**
   (`scope: debt`, minor) — #75 retired the `.spore/task_list.json` sandbox path;
   the standalone tools persist via `RunStore`. With the library default
   `no_op()` storage, a standalone invocation persists nothing across processes.
   Durable standalone use requires wiring a real `StorageProvider`. Accepted
   tradeoff; no migration shim.
6. **v1 memory keying limitation (#78 Q7), filed as #89** (`scope: deferred`,
   future phase) — `MemoryStore` is still `SessionId`-keyed, so #82's
   `MemoryTool` can only address the current session; durable session-independent
   cross-session addressing is the v2 feature. Documented in each language's
   `MemoryTool` module header. No SQL backend yet either (#77).
7. **Go-specific divergences (#80 + #81 + #79 + #78 + #82 + #61 + #58)**
   (`scope: debt`, minor, documented on the issues) — (a) local `Mode` newtype;
   (b) 3-state `TerminalOutcome`; (c) `StandardTool` in root `sporecore`
   type-aliased into `tools`; (d) `sendMessageToolName` duplicated in
   `harness.go`; (e) `abort` tool enforces required `reason` explicitly; (f)
   `promptassembly` owns a self-contained `HarnessBuilder`; (g)
   `ToolContext.MemoryStore` is an opaque marker interface asserted back to
   `storage.MemoryStore`; (h) #82 D2 single-source merge is an exported
   `storage.MergeMemories` (Go interfaces can't carry defaults); (i) #61 `Verifier`
   /`EvaluatorAgent` config set directly on the struct with no builder setters
   (Go has no `PlannerAgent` setter either); (j) `role-evaluator` chunk is a
   `RoleEvaluatorChunk` constant in `sporecore` (can't import the registry). New
   in #58: (k) the Ralph Stop hook is registered in `NewStandardHarness`,
   creating a `StandardHookChain` if none was configured; `MaxResets`/the Ralph
   types are set directly on the config struct (no builder setter), same as (i).
   All wire/behavior-identical to the other three (verified).
8. **Test-placement divergences (#78/#82)** (benign) — the #78 R9 registry-seam
   test lives in `@spore/tools` (TS) / the eval suite (Python); #82 reused TS
   `@spore/tools` `tool-context-memory-seam.test.ts` and keeps the Python
   catalogue test in `spore_tools`. Behavior identical.
9. **#79 cross-language divergences — both verified benign.** (a)
   `ContextSources.composed_prompt` carries the full `ComposedPrompt` in
   Rust/Python but a narrowed stub in TS/Go (outcomes identical). (b) The Block-1
   hash is not byte-identical (Rust SipHash vs. FNV-1a elsewhere) — the
   intentional #24 decision; #79 fixtures assert no hash values, only condition
   booleans and ordered bucket id lists.
10. **`Custom` condition is invisible in fixtures by design** (#79) —
    `ChunkCondition::Custom(predicate)` is first-class in the API but serializes
    to null/absent; architects using it knowingly opt that chunk out of the
    byte-identical cross-language contract.

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
1. **Push the eight unpushed commits — BLOCKED on local SSH auth.** The four #61
   and four #58 commits are on `main` locally only; `origin/main` is stale at
   `4f2f447`. `git push` fails (`sign_and_send_pubkey: signing failed for RSA`).
   Re-add/unlock the SSH key (`ssh-add ~/.ssh/sbeardsley`), then push so board,
   CI, and `main` agree. (No issue — release hygiene; **do this first**.)
2. **#60 — HillClimbing loop strategy** (`scope: deferred`). The last strategy
   stub; lands the harness at 5 of 5 and closes the final advertised-vs-actual
   core gap. Reuse the seams #58/#61 exercised — `run_react_inner` sub-loop, the
   failure-reason→user-message injection path, `RunStore`, #69 Stop hooks, #70
   alternate-agent defaulting, and a `max_resets`-style config-driven outer cap.
   Resolve spec forks in Rust before fan-out.
3. **Docs + correctness debt** — #27 (README strategy count: now 4/5, finish #60
   then update to 5/5), #35 (`harness-engineering-concepts.md` drift), #36
   (observability docs); #30/#31/#32/#34 (safety gates). Tightens the
   "correctness" half of the bar once #60 lands.
4. **#83–87 — protocol integrations** (unlabeled) — MCP (#83), A2A (#84), ACP
   (#85), AG-UI (#86), A2UI (#87). Triage/label as a new interop track or
   `scope: deferred` — **needs a maintainer decision** before prioritization.
5. **Deferred storage/memory phases** — #77 (SQL backends), #88 (deferred chunk
   providers), #89 (cross-session memory keying); plus optionally file the Ralph
   git-log v2 follow-up (Deviation #3) as its own issue. All `scope: deferred`.
