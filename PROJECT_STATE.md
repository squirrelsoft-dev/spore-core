# PROJECT STATE
_Last updated: 2026-05-30 by /close — closed #61 (SelfVerifying loop strategy) `status: complete`. All four #61 commits are on `main` **locally, not yet pushed** (`c4f607a` rust, `856736b` ts, `cc2bab0` python, `f5b2f21` go; `origin/main` still at `5dd91c9`). **#61 brings the harness to 3 of 5 loop strategies running end-to-end** (ReAct + PlanExecute + SelfVerifying) — it was strategy-wiring only on top of the existing `Verifier` family (#44), `planner_agent` defaulting (#70), Stop-hook seam (#69), `role-evaluator` chunk, and read-only sandbox flag. Four spec forks were resolved & pinned in Rust before fan-out (D1 bespoke strategy not Stop-hook adapter; D2 minimal `evaluator_agent` config w/ #70 defaulting; D3 `Verifier::max_iterations()`=3 caps round-trips; D4 two peer halts `SelfVerifyExhausted`/`SelfVerifyMisconfigured`). Notable new reusable seam: a **`ReadOnlySandbox` decorator** (all four languages) enforcing the read-only evaluate phase. The #78 Q7 v2 follow-up was **filed as #89** this prior loop. **Two strategy stubs remain** — #58 (Ralph), #60 (HillClimbing); finishing them closes the last advertised-vs-actual core gap (see Next Actions)._

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.

**#61 just landed — the SelfVerifying loop strategy runs end-to-end across all
four languages**, taking the harness to **3 of 5 loop strategies** (ReAct +
PlanExecute + SelfVerifying). It was strategy-wiring only: a `run_self_verifying`
method replaced the `StrategyNotYetImplemented` stub, orchestrating a build ReAct
sub-loop → a fresh evaluator run (fresh `SessionId`, read-only sandbox,
`role-evaluator` chunk) → the existing `Verifier` (#44), reusing the
failure-reason → user-message injection path to resume the build loop on FAIL.
Four spec forks were pinned in the Rust reference before fan-out: **D1** bespoke
strategy (not a Stop-hook adapter — the evaluate phase is a sibling *run* a Stop
hook can't express); **D2** one new `evaluator_agent` config field following the
#70 `planner_agent` defaulting contract (`None` → `config.agent`, documented on
the field), plus a `verifier` oracle field; **D3** `Verifier::max_iterations()`
(default 3) governs the build↔evaluate round-trip cap (not `max_stop_blocks`);
**D4** two **peer** `HaltReason` variants — `SelfVerifyExhausted { iterations,
last_reason }` (runtime limit) and `SelfVerifyMisconfigured { reason }`
(build-time bug, e.g. absent verifier; typed halt, never a panic). New reusable
seam: a **`ReadOnlySandbox` decorator** (all four) rejecting the seven mutating
tool names (`write_file`/`edit_file`/`delete_file`/`move_file`/`exec`/
`bash_command`/`run_tests`) with a recoverable `ReadOnlyViolation` and delegating
reads. New fixture `fixtures/harness/self_verifying.json` (7 cases, identical
outcomes all four). Cross-language verification PASS, no divergences;
`role-evaluator` chunk verified byte-identical across all four (incl. Go's
hardcoded constant, since Go can't import the registry). Rust `c4f607a` (787 lib
+ 12 new), TS `856736b` (+18), Python `cc2bab0` (978 passed, +14), Go `f5b2f21`
(+12 funcs/22 cases, `-race` clean). **These four commits are on `main` locally
but NOT pushed** — `origin/main` is still at `5dd91c9`.

**#82 (the prior loop) — the scope-aware `memory` tool ships across all four
languages**, built on #78's scoped `MemoryStore` API + `ToolContext` memory
seam. A single `memory` tool, `operation`-discriminated (`write`/`read`) like
`TaskListParams`, `scope` explicit on both ops, registered in the
`coding_set`/`full_set` presets (not readonly). Resolved design: `write` returns
the serialized just-written `MemoryEntry` (A); `read` `limit` defaults to 50 (B);
`metadata` is an optional `write` param defaulting to `{}` without growing
`MemoryEntry` (C); merged read is User∪Project, newest-first, no dedup, Local
excluded; single non-`read_only` tool so dispatch can't race the shared append
(E). `Local` scope is rejected on **both** ops before any storage access (exact
message `Local scope is not supported by MemoryTool — use User or Project.`,
nothing written); bad params + storage errors map to recoverable errors.
**Architectural outcome (decision D2): `get_memories_merged` was promoted onto
the `MemoryStore` trait/interface/protocol itself** — implemented once per
language (default trait method in Rust, shared helper elsewhere; Go via an
exported `MergeMemories` since interfaces can't carry defaults), and the prior
`StorageProvider`-level merge now delegates to it. Exactly one merge
implementation per language — D1 (tool re-implements the merge) was rejected to
avoid permanent cross-language divergence risk. Rust `b7afbf5` (14 + replay),
TS `5dd91c9` (21; core 922 / tools 153 / eval 46), Python `625b090` (964 passed,
8 skipped), Go `829780e` (14, incl. `-race`). New fixture
`fixtures/tools/memory.json` (merged-read step carries `"unordered": true` —
multiset compare, since tool-stamped timestamps can collide within a wall-clock
second); reuses `fixtures/storage/memory_scoped_merge.json` for strict
newest-first ordering through the tool. Cross-language verification PASS, no
divergences. **Known v1 limit (#78 Q7): memory stays `SessionId`-keyed — the
tool can only address the current session; session-independent cross-session
memory addressing is the v2 follow-up, now forced and **filed as #89**.**

**#78 (the storage seam #82 built on)** shipped `StorageScope { User, Project,
Local }`, a fixture-pinned `WorkspaceId` derivation
(`{sanitized_basename}-{8hex}`, SHA-256, host-independent), the scoped
`MemoryStore`, `(domain, scope) → backend` routing in
`CompositeStorageProvider` (`NoOp`-filled, never-null), user-scope workspace
partitioning at `{user_root}/projects/{workspace_id}`, and the `ToolContext`
memory-store field — the exact seam #82 consumed. (Rust `c0c2cbd`, TS `a3e2b1c`,
Python `f9acc84`, Go `078c5a7`.)

**Track A — tool/prompt architecture — is DONE, and the tool surface is now
complete with #82.** #80 (escalation protocol),
#81 (tool catalogue), and #79 (prompt assembly) all landed end-to-end across
four languages. #78 was a storage-debt item layered on top.

**The standard tool catalogue ships (#81)** across three tiers: Tier 1
pure-execution (`edit_file`, `grep`, `send_message`, `web_fetch`, `web_search`,
plus reused `read_file`/`write_file`/`find_files`/`grep_files`/`bash_command`),
Tier 2 session-aware (`todo_write`, `task_list`), Tier 3 harness-escalating
(`enter_plan_mode`/`exit_plan_mode`/`abort` → `Escalate`; `ask_user_question` →
`AwaitingClarification`). `StandardTools` presets (`readonly_set`/`coding_set`/
`full_set`), `HarnessBuilder::tool()`/`tools()`, last-wins registry upsert.

**The prompt assembly engine ships (#79)**: `ChunkCondition` (13 variants incl.
first-class `Custom(predicate)`), `PromptChunk`, `ToolAffinity`,
`AssemblyContext`, `ChunkProviderError`, the `ChunkProvider` trait
(`Embedded`/`InMemory`/`Composite`), and `ContextSourcesBuilder` (8-step
assembler). `HarnessBuilder` gained `chunk_provider()`/`chunks()`.
`RemoteChunkProvider` + scope-aware `FileSystemChunkProvider` were deferred to
**#88** (pin a constrained markdown-frontmatter fixture contract first).

**The persistence layer is clean** (#73 + #76 + #75 + #78 + now #82):
`StorageProvider` abstraction with per-domain **and per-scope** composite
routing; `plan_execute`, `task_list`, and `todo_write` persist solely through
`RunStore`; the `memory` tool reads/writes through the scoped `MemoryStore`
seam, with merged-read single-sourced on the `MemoryStore` trait itself (#82
D2). `SessionState.extras` keeps only genuinely ephemeral scratch.

**The escalation channel (#80)** returns structured `HarnessSignal`s via
`ToolOutput::Escalate`; the harness loop terminates cleanly to
`RunResult::Escalate`. `ask_user_question` extended the loop with a sibling
pause path (`AwaitingClarification` → `PausedState` with
`HumanRequest::Clarification`).

**The full PlanExecute chain is complete and on origin**: #45, #69, #70, #71,
#72, #73, #59, #76, #75, #80, #81, #79, #78, #82. The harness runs **two of
five** loop strategies (ReAct + PlanExecute) end-to-end.

Foundation in place: the harness is **runnable** (#57 — shared e2e CLI driving
the full ReAct loop against Ollama, 4-scenario hermetic suite, live against
`llama3.2`) and **debuggable** (#64/#65 — GenAI-convention content capture,
Arize Phoenix LLM-trace viewer alongside Tempo/Loki/Prometheus). Working
**evaluation/feedback loop** (#26 EvalHarness; #68 typed `Span` accessors).
Earlier: observability stack (#42/#49/#50/#33), `OllamaModelInterface` +
capability guard (#41), `StandardCompactionAdapter` + verify→retry→warn loop
(#55/#46), `StandardContextManager`/verifiers (#29), KeyTermVerifier (#47).

**origin/main is at `5dd91c9`** — work through #82 is pushed, but the **four #61
commits (`c4f607a`/`856736b`/`cc2bab0`/`f5b2f21`) are local-only and NOT yet
pushed**. All four language suites pass locally; push before the next loop.

Known runnability limits: the harness runs **ReAct, PlanExecute, and
SelfVerifying** end-to-end (3 of 5). `Ralph` and `HillClimbing` (#58/#60) still
return the generic `HaltReason::StrategyNotYetImplemented` — so the README and
`docs/harness-engineering-concepts.md` still overstate what runs (2 of 5
strategies remain stubs, down from 3).

## Active Direction
The harness is **runnable** (#57), **debuggable** (#64/#65), has a working
**evaluation/feedback loop** (#26/#68), runs **three** of five advertised loop
strategies (ReAct + PlanExecute + SelfVerifying, #61), has a **clean, fully-pluggable persistence seam
reaching all the way into tools, scope-aware, and now exercised by a real
`memory` tool** (#73 + #76 + #75 + #78 + #82), a **typed tool→caller escalation
channel** (#80), a **complete standard tool catalogue** (#81 + #82), and a
**conditional prompt assembly engine** (#79) — all across four languages. The bar
remains **capability breadth and correctness**.

**Track A — tool/prompt architecture — is CLOSED** (#79 + #80 + #81 + #82). #61
started the **loop-strategy track** — the recommended next bite after #82 — and
took the harness to 3 of 5 strategies. The active track is now finishing that:
1. **Finish the remaining loop strategies** — #58 (Ralph), then #60
   (HillClimbing). With #61 done, these two are the **only remaining gap between
   advertised and actual *core* capability** (the docs claim five strategies; two
   are still stubs). Each reuses the same seams #61/#69/#70/PlanExecute
   established. **Default recommendation — finish what #61 started.**
2. **Protocol integrations (#83–87)** — MCP (#83), A2A (#84), ACP (#85),
   AG-UI (#86), A2UI (#87) remain on the board unlabeled. A potential new track
   (interop/ecosystem) rather than core-capability depth. **Needs a maintainer
   decision before triage/prioritization.**
3. **Correctness/safety debt + docs cleanup** — #30/#31/#32/#34 (safety gates)
   and #27/#35/#36 (docs stop overstating). #27 specifically partially relieved
   by #61 (the count is now 3/5, not 2/5) but still overstates until #58/#60
   land. Lower-glamour but tightens the "correctness" half of the bar.

Storage remaining: SQL backends (#77, deferred) and the #88 deferred chunk
providers. The **#78 Q7 v2 follow-up — session-independent cross-session memory
keying — was filed as #89** (`scope: deferred`); interacts with #30.

## Known Deviations
1. **Two of five loop strategies are still stubs** (down from three) — the README
   and `docs/harness-engineering-concepts.md` advertise five loop strategies;
   **ReAct, PlanExecute, and SelfVerifying (#61) run end-to-end**, but `Ralph` and
   `HillClimbing` still return `HaltReason::StrategyNotYetImplemented` at
   `rust/crates/spore-core/src/harness.rs`. Tracked: #58 / #60
   (`scope: deferred`). #57's scenario suite is intentionally ReAct-only.
   (#61 **resolved** — see Current State.)
2. **Go outbox is not zero-dependency** — closing #50 added
   `go.opentelemetry.io/otel` + `otlptracegrpc` (v1.28.0) as blessed deps to
   `go/spore-core/go.mod` (accepted tradeoff, documented in `go/CONVENTIONS.md`).
   The durable JSONL path stays network-free.
3. **`task_list` / `todo_write` tool default persistence is no-op, not a file**
   (`scope: debt`, minor) — #75 retired the `.spore/task_list.json` sandbox path;
   the standalone tools persist via `RunStore`. With the library default
   `no_op()` storage, a standalone invocation persists **nothing across
   processes**. Durable standalone use requires wiring a real `StorageProvider`.
   Accepted tradeoff for retiring the sandbox path; no migration shim.
4. **v1 memory keying limitation (#78 Q7), now forced by #82** (`scope: deferred`,
   future phase) — `MemoryStore` is still `SessionId`-keyed, so #82's `MemoryTool`
   can only address the *current* session's memory; durable session-independent
   cross-session addressing is the v2 feature that makes memory useful across
   runs. Documented in each language's `MemoryTool` module header. **#82 forced
   the question — now filed as #89** (`scope: deferred`). No SQL backend yet
   either (#77, `scope: deferred`). _(Storage scope + workspace partitioning, formerly filed
   as #78, **resolved**; `MemoryTool` itself, #82, **resolved** — see Current
   State.)_
5. **Go-specific divergences (#80 + #81 + #79 + #78 + #82 + #61)** (`scope: debt`,
   minor, documented on the issues) — (a) the Go target defines a local `Mode`
   string newtype rather than importing it; (b) `HarnessObserver.SetSessionOutcome`
   is a 3-state `TerminalOutcome`; (c) `StandardTool` defined in root `sporecore`
   and type-aliased into `tools`; (d) `sendMessageToolName` duplicated in
   `harness.go`; (e) the `abort` tool enforces its required `reason` explicitly;
   (f) the `promptassembly` package owns a self-contained `HarnessBuilder` with
   the same contract (Go has no central builder); (g) `ToolContext.MemoryStore`
   is an opaque marker interface (`ToolMemoryStore interface{}`, asserted back to
   `storage.MemoryStore`) to dodge a package import cycle — mirrors the existing
   `ToolRunStore` seam. New in #82: (h) since Go interfaces can't carry default
   methods, the D2 single-source merge is an exported `storage.MergeMemories`
   helper that every `MemoryStore.GetMemoriesMerged` delegates to (Rust uses a
   default trait method, TS/Python a shared helper) — and `MemoryTool` performs
   the (g) `ToolMemoryStore`→`storage.MemoryStore` assertion to reach the store,
   exactly as it already consumes `ToolRunStore`. New in #61: (i) the `Verifier`
   and `EvaluatorAgent` config fields are set directly on the config struct
   (`cfg.Verifier = …`) with **no builder setters** — Go has no `PlannerAgent`
   setter either, so this matches the established pattern (the other three add
   `.verifier()`/`.evaluator_agent()` setters); and (j) the `role-evaluator`
   chunk is a `RoleEvaluatorChunk` constant in `sporecore` (verified
   byte-identical to the registry) because `sporecore` can't import
   `promptchunkregistry`. All wire/behavior-identical to the other three
   (verified).
6. **#78/#82 test-placement divergences** (benign, not spec violations) — the
   #78 R9 registry-seam test lives in `@spore/tools` (TS) and in the eval suite
   (Python) to avoid inverted package dependencies; Rust and Go keep theirs in
   the storage tests. #82 reused the existing TS `@spore/tools`
   `tool-context-memory-seam.test.ts` for its seam coverage and keeps the
   Python catalogue test in `spore_tools` (no inverted dependency). Behavior
   identical.
7. **#79 cross-language divergences — both verified benign.** (a)
   `ContextSources.composed_prompt` carries #24's full `ComposedPrompt` in
   Rust/Python but a narrowed `{rendered, block_1_hash}` stub in TS/Go (composed
   from the full type internally; outcomes identical). (b) The Block-1 hash is
   **not** byte-identical across languages (Rust SipHash vs. FNV-1a elsewhere) —
   the pre-existing, intentional #24 decision sanctioned in
   `fixtures/prompt_chunk_registry/basic.json`; #79's fixtures assert no hash
   values, only condition booleans and ordered bucket id lists (which ARE
   byte-identical), and R15 asserts intra-language hash stability.
8. **`Custom` condition is invisible in fixtures by design** (#79, not a defect)
   — `ChunkCondition::Custom(predicate)` is first-class in the public API (the
   primary escape hatch for non-serializable conditions) but serializes to
   null/absent and is pruned from `All`/`Any`/`Not` wire forms. Architects using
   `Custom` knowingly opt that chunk out of the byte-identical cross-language
   contract — a deliberate, supported choice.

_(Former Deviation — `MemoryTool` deferred/blocked — **resolved** in #82 this loop; shipped at parity across all four languages.)_
_(Former Deviation — storage scope + partitioning filed as #78 — **resolved** in #78.)_
_(Former Deviation — tool stuck on sandbox path — **resolved** in #75.)_
_(Former Deviation — extras persistence mirror unmigrated — **resolved** in #76.)_
_(Former Deviation — #72/#73 commits unpushed — **resolved** in the #59 loop.)_
_(Former Deviation — Rust Agent dyn-compatibility / `BoxFut` — **resolved** in #45.)_
_(Former Deviation — compaction `tokens_reclaimed = 0` — **resolved** in #57.)_
_(Former Deviation — sandbox Read of a missing in-workspace file → `PathEscape` — **resolved** in #63.)_
_(Former Deviation — EvalHarness Rust-only Debug-string metric workaround — **resolved** in #68.)_
_(Former Deviation — observability captured no message content — **resolved** in #64.)_

## Next Actions
[3-5 items max. Each references a GH issue # where possible.
This section is updated by /close after every PEE loop.]
1. **Push the #61 commits** — `c4f607a`/`856736b`/`cc2bab0`/`f5b2f21` are on
   `main` **locally only**; `origin/main` is still `5dd91c9`. All four suites pass
   locally. Push before starting the next strategy so the board, CI, and `main`
   agree. (No issue — release hygiene.)
2. **#58 — Ralph loop strategy** (`scope: deferred`). The next strategy after
   #61, and the recommended track: finishing #58 + #60 closes the last
   advertised-vs-actual core gap (2 of 5 still stubs). Reuse the seams #61
   exercised — pluggable ReAct sub-loop executor (`run_react_inner`), the
   failure-reason→user-message injection path, `RunStore`, the #69 hooks, and the
   #70 alternate-agent defaulting pattern. Resolve spec forks in Rust before
   fan-out, same as #61.
3. **#60 — HillClimbing loop strategy** (`scope: deferred`). The last strategy
   stub; lands the harness at 5 of 5 and lets #27/#35 docs stop overstating.
4. **Docs + correctness debt** — #27 (README strategy count: now 3/5, finish to
   5/5 then update), #35 (`harness-engineering-concepts.md` drift), #36
   (observability docs); #30/#31/#32/#34 (safety gates). Tightens the
   "correctness" half of the bar once the strategies land.
5. **#83–87 — protocol integrations** (unlabeled) + storage phases. MCP (#83),
   A2A (#84), ACP (#85), AG-UI (#86), A2UI (#87) — triage/label as a new
   interop track or `scope: deferred` (needs a maintainer decision). Plus #77
   (SQL backends), #88 (deferred chunk providers), #89 (cross-session memory
   keying), all `scope: deferred`.
