# PROJECT STATE
_Last updated: 2026-05-29 by /close — closed #79 (Prompt assembly engine) `status: complete`. Merged the four #79 commits to `main` and pushed (`origin/main` now `2e78af6`, was `9a300dd`). **#79 was the last open Track A item — Track A (tool/prompt architecture) is now fully closed.** #79 introduced the minimal `StorageScope` enum, which **unblocks #82 (MemoryTool)** — re-triaged from `scope: deferred` to **`status: queued`**. Spawned **#88** (deferred `RemoteChunkProvider` + scope-aware `FileSystemChunkProvider`). **New: a protocol-integration cluster (#83–87) appeared on the board unlabeled — needs a direction call (see Next Actions).**_

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.

**#79 just landed — the prompt assembly engine ships across all four
languages, and Track A is complete.** Architects can now drive conditional,
multi-source system-prompt assembly. The engine adds: `ChunkCondition` (13
variants — `Always`, `WhenMode`, `WhenToolActive`, `WhenToolCapability`,
`WhenPhase`, `WhenAgentType`, `WhenFeature`, `OnTrigger`, `OnEvent`, `All`,
`Any`, `Not`, and a first-class `Custom(predicate)`); a `PromptChunk` type
(distinct from #24's registry chunk) carrying `stability`/`condition`/`triggers`/
`tool_affinity`/`agent_affinity`/`cache_breakpoint`; `ToolAffinity`;
`AssemblyContext`; a minimal **`StorageScope { User, Project, Local }`** enum
(this module is its home); `ChunkProviderError`; the `ChunkProvider` trait with
`Embedded`/`InMemory`/`Composite` providers; and `ContextSourcesBuilder` — an
8-step assembler (evaluate conditions → bucket by stability → preserve
registration order → gate tool/agent affinity → route `OnTrigger` to PerTurn →
inject `OnEvent` chunks → compose Block-1 via #24's `ComposedPrompt` machinery →
produce `ContextSources`). `HarnessBuilder` gained `chunk_provider()`/`chunks()`
(default: empty `InMemoryChunkProvider`). Rust `11bede7` (50 assembly tests),
TS `5a2afbc` (35), Python `891b4d3` (32), Go `2e78af6` (33). Two shared fixtures
(`condition_eval.json`, `assembly_steps.json`) replay byte-identically in all
four. **`RemoteChunkProvider` and the scope-aware `FileSystemChunkProvider` were
deliberately deferred to #88** — network + YAML is where cross-language parity
gets expensive; #88 carries the hard prerequisite that the markdown frontmatter
be pinned as a constrained fixed key set (spec-first fixture contract, #29/#47
discipline) before any language starts.

**Track A — tool/prompt architecture — is DONE.** #80 (escalation protocol),
#81 (tool catalogue), and now #79 (prompt assembly) have all landed end-to-end
across four languages.

**The standard tool catalogue ships (#81)** across three tiers: Tier 1
pure-execution (`edit_file`, `grep`, `send_message`, `web_fetch`, `web_search`,
plus reused `read_file`/`write_file`/`find_files`/`grep_files`/`bash_command`),
Tier 2 session-aware (`todo_write`, `task_list`), Tier 3 harness-escalating
(`enter_plan_mode`/`exit_plan_mode`/`abort` → `Escalate`; `ask_user_question` →
`AwaitingClarification`). `StandardTools` presets (`readonly_set`/`coding_set`/
`full_set`), `HarnessBuilder::tool()`/`tools()`, last-wins registry upsert.

**The persistence layer is clean** (#73 + #76 + #75): `StorageProvider`
abstraction with per-domain composite routing; `plan_execute`, `task_list`, and
`todo_write` persist solely through `RunStore`. `SessionState.extras` keeps only
genuinely ephemeral scratch.

**The escalation channel (#80)** returns structured `HarnessSignal`s via
`ToolOutput::Escalate`; the harness loop terminates cleanly to
`RunResult::Escalate`. `ask_user_question` extended the loop with a sibling
pause path (`AwaitingClarification` → `PausedState` with
`HumanRequest::Clarification`).

**The full PlanExecute chain is complete and on origin**: #45, #69, #70, #71,
#72, #73, #59, #76, #75, #80, #81, #79. The harness runs **two of five** loop
strategies (ReAct + PlanExecute) end-to-end.

Foundation in place: the harness is **runnable** (#57 — shared e2e CLI driving
the full ReAct loop against Ollama, 4-scenario hermetic suite, live against
`llama3.2`) and **debuggable** (#64/#65 — GenAI-convention content capture,
Arize Phoenix LLM-trace viewer alongside Tempo/Loki/Prometheus). Working
**evaluation/feedback loop** (#26 EvalHarness; #68 typed `Span` accessors).
Earlier: observability stack (#42/#49/#50/#33), `OllamaModelInterface` +
capability guard (#41), `StandardCompactionAdapter` + verify→retry→warn loop
(#55/#46), `StandardContextManager`/verifiers (#29), KeyTermVerifier (#47).

**origin/main is at `2e78af6`** — all work through #79 is pushed. CI green
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
reaching all the way into tools** (#73 + #76 + #75), a **typed tool→caller
escalation channel** (#80), a **shippable standard tool catalogue** (#81), and
now a **conditional prompt assembly engine** (#79) — all across four languages.
The bar remains **capability breadth and correctness**.

**Track A — tool/prompt architecture — is CLOSED** (#79 + #80 + #81). The next
focus is a **direction call** between three candidate tracks:
1. **Finish the tool surface** — #82 (MemoryTool), now unblocked by #79's
   `StorageScope`; the remaining lift is a scoped `MemoryStore` seam on
   `ToolContext`.
2. **Light up the remaining loop strategies** — #61 (SelfVerifying, likely
   cheapest — a `Stop`-hook config on the #69 hook system), then #58 (Ralph),
   then #60 (HillClimbing). This is what stops the docs overstating capability
   (3 of 5 strategies are still stubs).
3. **Protocol integrations (NEW, #83–87)** — MCP (#83), A2A (#84), ACP (#85),
   AG-UI (#86), A2UI (#87) appeared on the board unlabeled. These represent a
   potential new track (interop/ecosystem) rather than core-capability depth.
   **Needs a maintainer decision before triage/prioritization.**

After the chosen track: the accepted-debt correctness fixes (#30/#31/#32/#34)
and docs/spec cleanup (#27/#35/#36) so the docs stop overstating capability.
Storage stays first-class: scope + workspace partitioning (#78) and SQL backends
(#77) are filed, both deferred — #78's scope work refines #79's minimal
`StorageScope` and is a softer dependency for the full #82 partition semantics.

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
4. **#73 deferred follow-ups — both filed.** (a) persistence migration off
   `SessionState.extras` is **fully resolved** (#59 + #76 + #75). (b) SQL backends
   filed as **#77** (`scope: deferred`). Storage scope + workspace partitioning
   filed as **#78** (`scope: deferred`).
5. **Go-specific divergences (#80 + #81 + #79)** (`scope: debt`, minor,
   documented on the issues) — (a) the Go target defines a local `Mode` string
   newtype rather than importing it; (b) `HarnessObserver.SetSessionOutcome` is a
   3-state `TerminalOutcome`; (c) `StandardTool` defined in root `sporecore` and
   type-aliased into `tools`; (d) `sendMessageToolName` duplicated in
   `harness.go`; (e) the `abort` tool enforces its required `reason` explicitly.
   New in #79: (f) Go has no central `HarnessBuilder`, so the `promptassembly`
   package owns a self-contained `HarnessBuilder` with the same contract
   (default empty `InMemoryChunkProvider`, `ChunkProvider()`/`Chunks()` setters).
   All are wire/behavior-identical to the other three (verified).
6. **#79 cross-language divergences — both verified benign.** (a)
   `ContextSources.composed_prompt` carries #24's full `ComposedPrompt` in
   Rust/Python but a narrowed `{rendered, block_1_hash}` stub in TS/Go (composed
   from the full type internally; outcomes identical). (b) The Block-1 hash is
   **not** byte-identical across languages (Rust SipHash vs. FNV-1a elsewhere) —
   this is the pre-existing, intentional #24 decision explicitly sanctioned in
   `fixtures/prompt_chunk_registry/basic.json`; #79's fixtures assert no hash
   values, only condition booleans and ordered bucket id lists (which ARE
   byte-identical), and R15 asserts intra-language hash stability.
7. **`Custom` condition is invisible in fixtures by design** (#79, not a defect)
   — `ChunkCondition::Custom(predicate)` is first-class in the public API (the
   primary escape hatch for non-serializable conditions) but serializes to
   null/absent and is pruned from `All`/`Any`/`Not` wire forms. Architects using
   `Custom` knowingly opt that chunk out of the byte-identical cross-language
   contract — a deliberate, supported choice.

_(Former Deviation #6 — `MemoryTool` deferred to #82 — still open but **unblocked**: #79 shipped the `StorageScope` it needed; now `status: queued`.)_
_(Former Deviation #3 — tool stuck on sandbox path — **resolved** in #75.)_
_(Former Deviation #4a — extras persistence mirror unmigrated — **resolved** in #76.)_
_(Former Deviation: #72/#73 commits unpushed — **resolved** in the #59 loop.)_
_(Former Deviation: Rust Agent dyn-compatibility / `BoxFut` — **resolved** in #45.)_
_(Former Deviation: compaction `tokens_reclaimed = 0` — **resolved** in #57.)_
_(Former Deviation: sandbox Read of a missing in-workspace file → `PathEscape` — **resolved** in #63.)_
_(Former Deviation: EvalHarness Rust-only Debug-string metric workaround — **resolved** in #68.)_
_(Former Deviation: observability captured no message content — **resolved** in #64.)_

## Next Actions
[3-5 items max. Each references a GH issue # where possible.
This section is updated by /close after every PEE loop.]
1. **DIRECTION CALL — pick the next track** (no issue; maintainer decision).
   Track A is closed. Choose between (a) **#82 MemoryTool** (now unblocked,
   finishes the tool surface), (b) **loop strategies #61→#58→#60** (stops the
   docs overstating capability), or (c) **protocol-integration track #83–87**
   (NEW — MCP/A2A/ACP/AG-UI/A2UI; currently unlabeled, needs triage). Default
   recommendation absent other input: **#82**, as the smallest next increment
   that closes a known gap.
2. **#82 — MemoryTool** (`status: queued`, unblocked by #79). Scope-aware
   (User/Project) episodic memory tool. Needs a scoped `MemoryStore` seam on
   `ToolContext` (a #75-level change across four languages); #78 partitioning is
   a softer dependency. The `StorageScope` type now exists (#79).
3. **#83–87 — protocol integrations** (NEW, unlabeled). MCP (#83), A2A (#84),
   ACP (#85), AG-UI (#86), A2UI (#87). Triage and label once the direction call
   in (1) lands — these may be a new track or `scope: deferred`.
4. **Loop strategies — #61 → #58 → #60** (`scope: deferred`). SelfVerifying,
   Ralph, HillClimbing; reuse the PlanExecute seams (pluggable ReAct sub-loop
   executor, `task_list` drain, `OnTaskAdvance` hook, `RunStore`).
5. **Correctness/safety debt + docs cleanup + storage phases** —
   #30/#31/#32/#34 (safety gates), #27/#35/#36 (docs stop overstating
   capability), #78 (scope + workspace partitioning), #77 (SQL backends).
   #88 (deferred Remote + FileSystem chunk providers) folds in here once a
   constrained-frontmatter fixture contract is pinned.
