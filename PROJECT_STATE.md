# PROJECT STATE
_Last updated: 2026-05-29 by /close — closed #81 (Standard tool catalogue) `status: complete`. Merged the four #81 commits to `main` and pushed (tip now `5a19240`, was `3e19cdc`). #81 was the keystone consumer proving the #75 (`ToolContext`) and #80 (`Escalate`) seams. MemoryTool was carved out to **#82** (`scope: deferred`, blocked on #79). With #80 + #81 done, **#79 (prompt assembly engine) is the last open Track A item — "work this next."**_

## Current State
spore-core is a language-agnostic agentic harness runtime built component by
component across four targets: Rust (reference), TypeScript, Python, and Go.

**#81 just landed — the standard tool catalogue ships across all four
languages.** Architects can now register pre-built tools instead of hand-rolling
them. The catalogue spans three tiers: **Tier 1** pure-execution
(`edit_file`, `grep` with `content`/`files_with_matches`/`count` output modes,
`send_message`, `web_fetch`, `web_search`; plus the already-shipped
`read_file`/`write_file`/`find_files`/`grep_files`/`bash_command` reused by name),
**Tier 2** session-aware (`todo_write` on `RunStore` key `"todo"`; `task_list`
already shipped via #71/#75), and **Tier 3** harness-escalating
(`enter_plan_mode`/`exit_plan_mode`/`abort` → `ToolOutput::Escalate{HarnessSignal}`;
`ask_user_question` → a **new `ToolOutput::AwaitingClarification` variant** with
its own loop pause/resume path). New: a `StandardTool` bundle (impl + schema),
a `StandardTools` namespace with `readonly_set`/`coding_set`/`full_set` presets,
`HarnessBuilder::tool()`/`tools()`, and registry `register()` is now a
**last-wins upsert** so an architect's same-name tool overrides a standard one.
`SendMessageTool` emits a new harness `StreamEvent::UserMessage`. Rust `2d9fcbb`
(796 tests), TS `962258e` (1022), Python `2a94655` (902), Go `5a19240` (all
modules). Cross-language verification: all four green, five new fixtures
(`edit_file_cases`, `grep_output_modes`, `send_message_event`, `escalation_tools`,
`todo_write`) replay identically; `param_validation.json` left byte-identical.
**`MemoryTool` was the one in-scope item deferred** — it needs a `StorageScope`
type and a scoped `MemoryStore` seam on `ToolContext` that don't exist yet, so it
was carved out to **#82** (blocked on #79).

**Track A is nearly done.** #80 (escalation protocol) and #81 (tool catalogue)
have both landed. **Only #79 (prompt assembly engine) remains** — largely
independent of #80/#81, it rounds out the tool/prompt architecture track.

**The persistence layer is clean** (#73 + #76 + #75): `StorageProvider`
abstraction with per-domain composite routing; `plan_execute` and `task_list`
persist solely through `RunStore`; the standalone `task_list` tool does too, and
now `todo_write` joins it (RunStore key `"todo"`). `SessionState.extras` is kept
only for genuinely ephemeral scratch (`__rich_state` for compaction,
`subagent_handoff_summary` for subagents).

**The escalation channel (#80) now has its proving consumer.** Tools return
structured `HarnessSignal`s via `ToolOutput::Escalate`; the harness loop
terminates cleanly and returns `RunResult::Escalate { signal, ... }` without
acting on the signal. #81's Tier-3 tools are exactly the things that *return*
`Escalate`, and `ask_user_question` extended the loop with a sibling pause path
(`AwaitingClarification` → `PausedState` with `HumanRequest::Clarification`, no
`ChildPausedState`; resume injects the answer as the clarifying call's result).

**The full PlanExecute chain is complete and on origin**: #45, #69, #70, #71,
#72, #73, #59, #76, #75, #80, #81. The harness runs **two of five** loop
strategies (ReAct + PlanExecute) end-to-end.

Foundation in place: the harness is **runnable** (#57 — shared e2e CLI driving
the full ReAct loop against Ollama, 4-scenario hermetic suite, live against
`llama3.2`) and **debuggable** (#64/#65 — GenAI-convention content capture,
opt-in/truncated/env-guarded, Arize Phoenix LLM-trace viewer alongside
Tempo/Loki/Prometheus). It has a working **evaluation/feedback loop** (#26
EvalHarness; #68 typed `Span` accessors). Earlier work: observability stack
(#42/#49/#50/#33), `OllamaModelInterface` + capability guard (#41),
`StandardCompactionAdapter` + verify→retry→warn loop (#55/#46),
`StandardContextManager`/verifiers (#29), KeyTermVerifier (#47).

**origin/main is at `5a19240`** — all work through #81 is pushed. CI green across
all four languages.

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
escalation channel** (#80), and now a **shippable standard tool catalogue across
all four languages** (#81). The bar remains **capability breadth and
correctness**.

**Track A — tool/prompt architecture — is almost closed.** #80 (escalation) and
#81 (tool catalogue) both landed. The last open item is **#79 (prompt assembly
engine — `ChunkProvider`/`PromptChunk`/`AssemblyContext`/`ContextSourcesBuilder`)**.
It also unblocks **#82 (MemoryTool)**, which needs the `StorageScope` concept #79
introduces. Finish #79 and Track A is done.

The remaining **loop strategies** are deferred behind track A: **SelfVerifying
(#61)** likely-cheapest (a `Stop`-hook configuration on the #69 hook system),
**Ralph (#58)** the natural pair to PlanExecute's execute phase, **HillClimbing
(#60)** once the seams have a second consumer.

After track A: the accepted-debt correctness fixes (#30/#31/#32/#34) and
docs/spec cleanup (#27/#35/#36) so the docs stop overstating capability. Storage
stays first-class: scope + workspace partitioning (#78) and SQL backends (#77)
are filed, both deferred — and #78's scope work is a prerequisite for #82.

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
   processes**. Durable standalone use requires wiring a real `StorageProvider`
   (e.g. `FileSystemStorageProvider`). Accepted tradeoff for retiring the sandbox
   path; no migration shim.
4. **#73 deferred follow-ups — both filed.** (a) persistence migration off
   `SessionState.extras` is **fully resolved** (#59 + #76 + #75). (b) SQL backends
   filed as **#77** (`scope: deferred`). Storage scope + workspace partitioning
   filed as **#78** (`scope: deferred`).
5. **Go-specific divergences (#80 + #81)** (`scope: debt`, minor, documented on
   the issues) — (a) the Go target defines a local `Mode` string newtype in the
   `sporecore` package rather than importing it (identical bare-string wire form,
   avoids an import cycle); (b) `HarnessObserver.SetSessionOutcome` is a 3-state
   `TerminalOutcome` (Success/Failure/Escalated). New in #81: (c) `StandardTool`
   is defined in the root `sporecore` package and type-aliased into `tools` (the
   `HarnessBuilder` lives in `observability`, which can't import `tools` without a
   cycle); (d) `sendMessageToolName` is a duplicated const in `harness.go` rather
   than imported from `tools` (same cycle reason, mirrors the #80 `Mode`
   pattern); (e) the `abort` tool enforces its required `reason` field explicitly
   because Go's `json.Unmarshal` — unlike serde — doesn't reject a missing
   required field. All five are wire/behavior-identical to the other three
   languages (verified in cross-language verification).
6. **`MemoryTool` deferred out of #81 → #82** (`scope: deferred`) — the one
   in-scope Tier-2 item not delivered. It requires a `StorageScope`
   (User/Project) type and a scoped `MemoryStore` seam on `ToolContext`, neither
   of which exists. Blocked on #79 (introduces `StorageScope`) and the #78
   scope/partitioning design. Everything else in #81's scope shipped.

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
1. **#79 — Prompt assembly engine** (`status: queued`, **work this next**).
   `ChunkProvider`, `PromptChunk`, `AssemblyContext`, `ContextSourcesBuilder` —
   conditional/multi-backend chunk loading extending the #24 `PromptChunkRegistry`.
   The last open Track A item. Also introduces the `StorageScope` concept that
   unblocks #82 (MemoryTool), and lets the #81 tools wire their description chunks
   via `tool_affinity`.
2. **#82 — MemoryTool** (`scope: deferred`, blocked on #79). Scope-aware
   (User/Project) episodic memory tool carved out of #81. Needs `StorageScope`
   (#79) + a scoped `MemoryStore` seam on `ToolContext` (a #75-level change across
   four languages) + the #78 partitioning design. Pick up once #79 lands.
3. **Loop strategies — #61 → #58 → #60** (`scope: deferred`, after track A).
   SelfVerifying, Ralph, HillClimbing; reuse the PlanExecute seams (pluggable ReAct
   sub-loop executor, `task_list` drain, `OnTaskAdvance` hook, `RunStore`).
4. **Correctness/safety debt + docs cleanup (#30/#31/#32/#34, #27/#35/#36)** —
   memory distillation gate (#30), read-only subagent context (#31), Block 2 hash
   mismatch halt (#32), dangerous-feature-flag gate (#34); README/spec
   clarifications (#27/#35) and E2B data-residency note (#36). Fold in so docs stop
   overstating capability.
5. **Storage future phases — #78 (scope + workspace partitioning), #77 (SQL
   backends)** — both `scope: deferred`; #78 is now also a prerequisite for #82.
