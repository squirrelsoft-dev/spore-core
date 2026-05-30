# PROJECT STATE
_Last updated: 2026-05-29 by /close — closed #82 (MemoryTool) `status: complete`. All four #82 commits are on `main` and pushed (`origin/main` now `5dd91c9`). **#82 closes the tool surface for the storage track** — the scope-aware `memory` tool ships at parity across all four languages on top of #78's seam. The notable architectural outcome: merged-read (`get_memories_merged`) was promoted onto the `MemoryStore` trait/interface/protocol itself (decision D2) so there is exactly one merge implementation per language. Memory remains `SessionId`-keyed — the #78 Q7 v2 cross-session-keying follow-up is now **forced** and needs filing (issue-create was permission-blocked this loop; drafted body ready). Track A is closed, #82 is done — the direction call (tool surface vs. loop strategies vs. protocol track) is **now genuinely open** with the obvious tool-surface increment consumed (see Next Actions)._

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.

**#82 just landed — the scope-aware `memory` tool ships across all four
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
memory addressing is the v2 follow-up, now forced and needing an issue (filing
was permission-blocked this loop).**

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

**origin/main is at `5dd91c9`** — all work through #82 is pushed. CI green
across all four languages.

Known runnability limits: the harness runs **ReAct and PlanExecute** end-to-end.
`Ralph`, `SelfVerifying`, and `HillClimbing` (#58/#61/#60) still return the generic
`HaltReason::StrategyNotYetImplemented` — so the README and
`docs/harness-engineering-concepts.md` still overstate what runs (3 of 5
strategies remain stubs).

## Active Direction
The harness is **runnable** (#57), **debuggable** (#64/#65), has a working
**evaluation/feedback loop** (#26/#68), runs **two** of five advertised loop
strategies (ReAct + PlanExecute), has a **clean, fully-pluggable persistence seam
reaching all the way into tools, scope-aware, and now exercised by a real
`memory` tool** (#73 + #76 + #75 + #78 + #82), a **typed tool→caller escalation
channel** (#80), a **complete standard tool catalogue** (#81 + #82), and a
**conditional prompt assembly engine** (#79) — all across four languages. The bar
remains **capability breadth and correctness**.

**Track A — tool/prompt architecture — is CLOSED, and the tool surface is now
complete** (#79 + #80 + #81 + #82). #82 consumed the obvious smallest increment,
so the **direction call is now genuinely open** between two remaining
core-capability tracks plus an interop track:
1. **Light up the remaining loop strategies** — #61 (SelfVerifying, likely
   cheapest — a `Stop`-hook config on the #69 hook system), then #58 (Ralph),
   then #60 (HillClimbing). This is what stops the docs overstating capability
   (3 of 5 strategies are still stubs) — the only remaining gap between
   advertised and actual *core* capability. **Default recommendation.**
2. **Protocol integrations (#83–87)** — MCP (#83), A2A (#84), ACP (#85),
   AG-UI (#86), A2UI (#87) remain on the board unlabeled. A potential new track
   (interop/ecosystem) rather than core-capability depth. **Needs a maintainer
   decision before triage/prioritization.**
3. **Correctness/safety debt + docs cleanup** — #30/#31/#32/#34 (safety gates)
   and #27/#35/#36 (docs stop overstating). Lower-glamour but tightens the
   "correctness" half of the bar.

Storage remaining: SQL backends (#77, deferred) and the #88 deferred chunk
providers. The **#78 Q7 v2 follow-up — session-independent cross-session memory
keying — is now forced by #82 and needs an issue filed** (issue-create was
permission-blocked this loop; drafted body is ready).

## Known Deviations
1. **Three of five loop strategies are still stubs** — the README and
   `docs/harness-engineering-concepts.md` advertise five loop strategies; **ReAct
   and PlanExecute run end-to-end**, but `Ralph`, `SelfVerifying`, and
   `HillClimbing` still return `HaltReason::StrategyNotYetImplemented` at
   `rust/crates/spore-core/src/harness.rs`. Tracked: #58 / #61 / #60
   (`scope: deferred`). #57's scenario suite is intentionally ReAct-only.
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
   runs. Documented in each language's `MemoryTool` module header. **#82 forces
   the question** — a v2 issue needs filing (drafted this loop; issue-create was
   permission-blocked, body held). No SQL backend yet either (#77,
   `scope: deferred`). _(Storage scope + workspace partitioning, formerly filed
   as #78, **resolved**; `MemoryTool` itself, #82, **resolved** — see Current
   State.)_
5. **Go-specific divergences (#80 + #81 + #79 + #78 + #82)** (`scope: debt`,
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
   exactly as it already consumes `ToolRunStore`. All wire/behavior-identical to
   the other three (verified).
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
1. **DIRECTION CALL — now genuinely open** (no issue; maintainer decision).
   Track A is closed and #82 consumed the obvious tool-surface increment, so
   there's no longer a default "smallest gap" item. Pick the next track:
   (a) **loop strategies #61→#58→#60** — the only remaining gap between
   advertised and actual *core* capability (3 of 5 strategies still stubs);
   **recommended**; (b) **protocol-integration track #83–87** — interop/ecosystem,
   still unlabeled, needs triage; (c) **correctness/safety debt + docs**
   (#30/#31/#32/#34, #27/#35/#36).
2. **File the #78 Q7 v2 follow-up** — session-independent cross-session memory
   keying, **now forced by #82** (the `memory` tool can only see the current
   session). Drafted this loop; `gh issue create` was permission-blocked, body
   held — file it (label `scope: deferred`) so the board reflects the forced
   question. Interacts with #30 (memory distillation PendingReview gate).
3. **Loop strategies — #61 → #58 → #60** (`scope: deferred`). SelfVerifying
   (likely cheapest — `Stop`-hook config on #69), Ralph, HillClimbing; reuse the
   PlanExecute seams (pluggable ReAct sub-loop executor, `task_list` drain,
   `OnTaskAdvance` hook, `RunStore`). The pick that stops the docs overstating.
4. **#83–87 — protocol integrations** (unlabeled). MCP (#83), A2A (#84),
   ACP (#85), AG-UI (#86), A2UI (#87). Triage and label once the direction call
   lands — a new track or `scope: deferred`.
5. **Correctness/safety debt + docs cleanup + storage phases** —
   #30/#31/#32/#34 (safety gates), #27/#35/#36 (docs stop overstating
   capability), #77 (SQL backends), #88 (deferred Remote + FileSystem chunk
   providers, once a constrained-frontmatter fixture contract is pinned).
