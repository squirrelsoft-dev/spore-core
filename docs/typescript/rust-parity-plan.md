# Rust → TypeScript Harness Parity Plan

> **Status: PLAN-FIRST CHECKPOINT (revised) — no implementation started.**
> Authoritative Rust revision recorded below. This document inventories the
> Rust harness and the current TypeScript implementation end to end, records a
> parity matrix, and specifies file-level steps. It is ready for captain
> approval. No implementation code has been written.
>
> **Revision note (captain scope decision incorporated):** the complete
> TypeScript `spore-core` + `spore-eval` surface is IN SCOPE, including the
> full in-core `VectorStore` / `EmbeddingModel` traits and their Ollama/OpenAI
> providers (§5 G-L, mocked-HTTP validation) and the three missing TypeScript
> doc pages (§5 G-M). The separate `spore-mcp` / `spore-sqlvec` packages and
> the feature-gated Petri backend are OUT OF SCOPE — documented as known gaps,
> not created (§9). Added §5.2 with explicit executable cancellation, timeout,
> retry, cleanup, and async/concurrency scenarios. Rust remains strictly
> read-only.

## 1. Authoritative revision & scope

- **Authoritative Rust revision:** `7058db115b4f05b6b7cde1a3f62ba345c202bcde`
  (recorded via `git rev-parse HEAD` on the disposable worktree at
  `/Users/sbeardsley/.treehouse/spore-core-a93132/3/spore-core`, branch
  `fm/parity-typescript`).
- **Authoritative surface:** `rust/crates/spore-core` (the harness runtime
  library), `rust/crates/spore-eval` (eval harness + CLI), and
  `rust/crates/spore-cli` (a `fn main() {}` stub — no behavior to mirror).
- **Read-only guard:** Rust source, Rust tests, `docs/rust/`, `examples/rust/`,
  and Rust configuration are treated as authoritative and read-only. No edits
  will be made under those paths. If parity appears to require a Rust change,
  the conflict will be reported and work will stop.
- **TypeScript edit surface:** `typescript/`, `docs/typescript/`,
  `examples/typescript/`, and TypeScript-specific tests/configuration only.

### Out-of-scope Rust subsystems (captain decision — known gaps, not created)

- `rust/crates/spore-mcp` — MCP client that surfaces a remote MCP server's
  tools as native `Tool`s via the `rmcp` SDK. A **separate crate** depending on
  `spore-core`; not part of the core harness library. No TypeScript package
  equivalent will be created.
- `rust/crates/spore-sqlvec` — sqlite-vec `VectorStore` backend. A **separate
  crate** depending on `spore-core`'s `VectorStore`/`EmbeddingModel` traits.
  No TypeScript package equivalent will be created.
- `rust/crates/spore-core/src/sandbox_petri.rs` — feature-gated Petri
  isolation backend (`#[cfg(feature = "petri")]`). No TS equivalent will be
  created.

Per the captain's scope decision (§9), these are documented as **known gaps**
in the final matrix (§10) and are NOT implemented in this parity pass. NOTE:
the `VectorStore` / `EmbeddingModel` *traits and their in-core Ollama/OpenAI
providers* are NOT out of scope — they live in `spore-core` itself and ARE
implemented (§5 G-L). Only the separate `@spore/sqlvec` *backend package*
(and `@spore/mcp`, Petri) are out of scope.

## 2. Methodology & evidence conventions

- Every capability row distinguishes:
  - **OBSERVED PARITY** — same public shape and behavior verified by reading
    both sides (file:line cited).
  - **INFERRED SIMILARITY** — looks similar on inspection but behavior not
    verified end to end (flagged for a regression test during implementation).
  - **GAP** — missing or observably different behavior (cited on both sides).
- Severity: **blocker** (core loop behavior absent/divergent), **major**
  (named feature absent), **minor** (helper/docs/ergonomics).
- Rust evidence is cited as `rust/...:line`; TypeScript as `typescript/...:line`.

## 3. Baseline (current TypeScript state)

- **Tests:** `pnpm -r test` — green. `@spore/core` 1502 passed | 7 skipped;
  `@spore/tools` 269 passed; `@spore/eval` 46 passed.
- **Lint:** `pnpm -r lint` — green (eslint, all packages).
- **Type-check:** `tsc --noEmit` (core `tsconfig.json` includes `tests/**/*`)
  reports **pre-existing errors** in test files only:
  `tests/tasklist.test.ts` (11× `TS2532 Object is possibly 'undefined'` under
  `noUncheckedIndexedAccess`) and `tests/tool-error-loop.test.ts:221`
  (`TS18048 'r.session_state' is possibly 'undefined'`). No `typecheck` script
  is defined in any `package.json`; `build` is `pnpm -r build` and core has no
  `build` script (source-only package). These test-only tsc errors are a
  pre-existing baseline and will be fixed as part of this work (they block a
  clean type-check gate).
- **Package layout:** pnpm workspace; `@spore/core` (source-only, ESM, zod),
  `@spore/tools`, `@spore/eval`, `@spore/test-utils`. Node 20 LTS, Vitest.
- **Examples:** `examples/typescript/01..12` mirror `examples/rust/01..12`
  one-for-one (hello-agent, conversational-repl, tool-use, filesystem-agent,
  custom-sandboxed-tool, web-research, memory, plan-execute, self-verifying,
  hill-climbing, multi-agent, cordyceps).
- **Docs:** `docs/typescript/` has only a stub `quickstart.md`. Rust has
  `quickstart.md`, `building-a-tool.md`, `conversation.md`, `custom-harness.md`.

## 4. Parity matrix

### 4.1 Harness run loop, strategies, budget, config

| # | Capability | Rust evidence | TS behavior | Status | Sev |
|---|---|---|---|---|---|
| L1 | `RunResult` variants (`success`/`failure`/`waiting_for_human`/`escalate`/`consult`) + `session_state` carry | `rust/crates/spore-core/src/harness/types.rs:1608-1700` | `typescript/packages/core/src/harness/types.ts:5243-5318` (+ `runResultSessionState` helper) | OBSERVED PARITY | — |
| L2 | `HaltReason` variants (budget/termination/middleware/agent/context/sandbox/unrecoverable/human/stagnation/strategy_not_yet_implemented/empty_plan/step_failed/plan_phase_failed/self_verify_*/ralph_*/hill_climbing_misconfigured/tasks_blocked_by_failure/task_graph_cycle/tool_error_loop/output_schema_violation/configuration_error) | `rust/.../harness/types.rs:1421-1606` | `typescript/.../harness/types.ts:5103-5241` | OBSERVED PARITY | — |
| L3 | `HarnessStreamEvent` coarse + delta variants (turn_start/end, tool_call, tool_result, final_response, budget_warning, user_message, text/reasoning/tool_args delta, block_start/stop, tool_call_start, tool_error_loop_detected/broken, output_schema_retry/violation) | `rust/.../harness/types.rs:60-410` | `typescript/.../harness/types.ts:4042-4121` | OBSERVED PARITY (variants) | — |
| L4 | **`NodeAttr` attribution on every stream event + `wrap_node_sink`** (multi-agent node routing) | `rust/.../harness/types.rs:426-460` (`NodeAttr`), `wrap_node_sink` in `types.rs:461+`, used in `react.rs:2125`, `plan_execute.rs:385`, `hill_climbing.rs:252`, `self_verifying.rs:246`, `ralph.rs:255` | **No `NodeAttr` type, no `node` field on any event, no `wrapNodeSink`.** `grep NodeAttr` in TS harness → 0 hits. | **GAP** | major |
| L5 | **`StreamEvent::ToolProgress`** (live tail-snapshot preview from a running tool) | `rust/.../harness/types.rs:166-178` | **No `tool_progress` variant** in `HarnessStreamEvent` (`typescript/.../harness/types.ts:4042-4121`) | **GAP** | major |
| L6 | `tool_result.revised` flag (AfterTool middleware rewrite signal) | `rust/.../harness/types.rs:148-160` | **No `revised` field** on TS `tool_result` (`typescript/.../harness/types.ts:4067-4072`) | **GAP** | major |
| L7 | `user_message.steered` flag (user-authored steering vs agent narration) | `rust/.../harness/types.rs:181-200` | **No `steered` field** on TS `user_message` (`typescript/.../harness/types.ts:4079-4081`) | **GAP** | major |
| L8 | `HarnessConfig` field set | `rust/.../harness/config.rs:100-205` | `typescript/.../harness/standard.ts:261-540` | INFERRED SIMILARITY (most fields present; see L9–L12) | — |
| L9 | **`steering_source: Option<SteeringSource>`** + mid-run drain at turn boundary (inject queued user messages, emit `UserMessage{steered:true}`, persist via ContextManager) | `rust/.../harness/config.rs:154`, `rust/.../harness/react.rs:343-360` | **No `steeringSource` field**, no drain logic. `grep steeringSource` TS → 0 hits. | **GAP** | major |
| L10 | **`tool_call_repair: Option<Arc<dyn ToolCallRepair>>` + `max_repair_attempts`** (coerce malformed tool args, retry within attempts, enrich error) | `rust/.../harness/config.rs:130-133`, `rust/.../harness/react.rs:27-35`, trait `rust/.../tool_call_repair.rs:332-345`, `coerce_tool_args` `:45`, `StandardToolCallRepair` `:342` | **No `ToolCallRepair` interface, no `StandardToolCallRepair`, no `coerceToolArgs`, no `maxRepairAttempts` config.** TS `define-tool.ts:14,139` only *references* "a configured ToolCallRepair" in comments. TS has `detectProseResponse` only. | **GAP** | major |
| L11 | **`tool_call_nudge: bool`** + behavioral nudge (inject a tool-calling nudge when a turn ends having called zero tools, capped at `MAX_TOOL_CALL_NUDGES=2`) | `rust/.../harness/config.rs:145`, `rust/.../harness/react.rs:328-335,741-810` | **No `toolCallNudge` config, no nudge logic, no `tools_called_this_window` tracking.** | **GAP** | major |
| L12 | **`plan_empty_max_retries: u32`** + plan-empty re-ask loop (re-ask with tools suppressed + `/no_think` nudge before halting `EmptyPlan`; fold discarded usage) | `rust/.../harness/config.rs:135-138`, `rust/.../harness/plan_execute.rs:499-670`, `StrategyExecutor::plan_empty_max_retries` `rust/.../harness/strategy_executor.rs:298-299` | **No `planEmptyMaxRetries` config, no re-ask loop.** TS `captureAndPersistPlan` (`typescript/.../harness/standard.ts:2556-2612`) is single-shot. | **GAP** | major |
| L13 | **`HarnessSignal::PlanRecorded`** (set_plan tool's terminal signal, consumed in PlanExecute plan phase) | `rust/.../harness/types.rs:552-555`, handled in `plan_execute.rs:560-575` | **No `plan_recorded` variant** in TS `HarnessSignal` (`typescript/.../harness/types.ts:4271-4276`: only enter_plan_mode/exit_plan_mode/switch_mode/abort) | **GAP** | major |
| L14 | **Narration refund**: `send_message`-only turns do not consume the productive-turn budget (refund one turn, capped at ~2× budget) | `rust/.../harness/react.rs:336-345` + gate logic | TS react loop (`typescript/.../harness/standard.ts:3140-3200`) has no narration refund; `send_message` success only emits `user_message` (`:4007-4009`) | **GAP** | major |
| L15 | Defaults: `error_loop_threshold=3`, `output_schema_max_retries=2`, `max_compaction_attempts=2`, `enforce_output_schemas=false`, `max_stop_blocks` default | `rust/.../harness/config.rs:521-551` | `typescript/.../harness/standard.ts:535,538,1910,1915,4351,909` | OBSERVED PARITY | — |
| L16 | ReAct / PlanExecute / SelfVerifying / HillClimbing / Ralph strategy shapes & combinators | `rust/.../harness/{react,plan_execute,self_verifying,hill_climbing,ralph}.rs` | `typescript/.../harness/standard.ts` (all five implemented) | INFERRED SIMILARITY — exercised by passing fixture-replay tests; full behavioral equivalence to be confirmed by representative E2E tests during implementation | — |
| L17 | Budget policy/enforcement/exhaustion, `BudgetExhaustedBehavior`, `ExhaustionCause` | `rust/.../harness/budget.rs:146-200` | `typescript/.../harness/types.ts:190,888`; `budget-policy.test.ts`/`budget-enforcement.test.ts` pass | OBSERVED PARITY | — |
| L18 | Escalation / paused-state / HITL / consult (`resume`, `resumeConsult`, consult handlers, overflow policy) | `rust/.../harness/types.rs:1108-1300` | `typescript/.../harness/types.ts:4821-4951`; `harness-escalation.test.ts`, `harness-consult.test.ts`, `budget-exhausted-hitl.test.ts` pass | OBSERVED PARITY | — |
| L19 | Compaction trigger/threshold + verifier retry | `rust/.../harness/compaction.rs`, `config.rs:111-117` | `typescript/.../harness/standard.ts:4320-4400`; `harness-compaction-loop.test.ts` passes | OBSERVED PARITY | — |
| L20 | `AbortSignal` cancellation threading through async public APIs | (Rust uses `CancellationToken`-style via `BoxFut`) | TS CONVENTIONS mandate `AbortSignal` on every async public API; threaded through run loop | OBSERVED PARITY (idiomatic difference) | — |

### 4.2 Tools & tool registry

| # | Capability | Rust evidence | TS behavior | Status | Sev |
|---|---|---|---|---|---|
| T1 | `Tool` trait (call/schema/annotations/progress), `ToolRegistry`, `RealToolRegistry`/`StandardToolRegistry`, `RegistrationError`/`DispatchError`, `ToolNamespace`/`ToolSet`/`ToolSearchConfig`/`ToolSearchRegistry`/`TaskPhase`, `ToolContext` | `rust/.../tool_registry.rs` | `typescript/.../tool-registry/*.ts` | OBSERVED PARITY (interface-level) | — |
| T2 | **`ToolProgressSink` / `ToolProgressUpdate`** (run-level progress sink → `StreamEvent::ToolProgress`), `ToolContext::command_progress` | `rust/.../tool_registry.rs:140-200` | **No `ToolProgressSink`/`ToolProgressUpdate`, no `commandProgress` on `ToolContext`.** | **GAP** | major |
| T3 | **`SetPlanTool` (`set_plan`)** — atomic plan authoring into the durable `task_list` store; escalates `PlanRecorded` | `rust/.../tools/set_plan.rs` (154 lines), catalogued `rust/.../tools/catalogue.rs:216` | **No `SetPlanTool`, no `set_plan` in `StandardTools`, no `set_plan` params.** `grep set_plan` TS → 0 hits. | **GAP** | major |
| T4 | **`CreateDirTool` (`create_dir`)** — mkdir -p | `rust/.../tools/fs.rs:276`, catalogued `catalogue.rs:131` | **No `CreateDirTool`, no `create_dir` params, no catalogue entry.** | **GAP** | major |
| T5 | `StandardTools` catalogue: `move_file`, `delete_file` constructors | `rust/.../tools/catalogue.rs:138,143` | TS impls exist (`MoveFileTool`/`DeleteFileTool` in `typescript/.../tools/src/fs.ts`) but **not exposed as `StandardTools` constructors** (`typescript/.../tools/src/catalogue.ts` has no `moveFile`/`deleteFile`) | **GAP** | major |
| T6 | `StandardTools.coding_set` membership (Rust includes `create_dir`, `move_file`, `delete_file`, `set_plan`) | `rust/.../tools/catalogue.rs:279-302` | TS `codingSet` (`typescript/.../tools/src/catalogue.ts:283-308`) **omits all four** | **GAP** | major |
| T7 | `readonly_set` / `full_set` membership | `rust/.../tools/catalogue.rs:264-335` | TS `readonlySet`/`fullSet` (`catalogue.ts:249-330`) | OBSERVED PARITY (modulo T6) | — |
| T8 | Tool impls: `read_file` (range #132), `write_file`, `edit_file`, `list_dir`, `grep`/`grep_files` (output modes, context lines #133), `find_files`, `bash_command`/`exec`/`run_tests`, `git_*`, `http_get`/`http_post`, `web_fetch`/`web_search`, `send_message`, `todo_write`, `task_list`, `memory`, `enter_plan_mode`/`exit_plan_mode`/`ask_user_question`/`abort`, `subagent` | `rust/.../tools/*.rs` | `typescript/.../tools/src/*.ts` | INFERRED SIMILARITY — covered by passing fixture-replay + per-tool tests; equivalence to be confirmed by representative E2E tests | — |
| T9 | `finish_with_possible_truncation` + `LARGE_OUTPUT_THRESHOLD` (64KiB) + `DEFAULT_HEAD/TAIL_TOKENS` (2000/2000) | `rust/.../tools/mod.rs:80-110` | TS truncation path present (`typescript/.../tools/src/fs.ts` etc.); `sandbox.test.ts` covers `handleLargeOutput` | INFERRED SIMILARITY | — |
| T10 | `subagent` `ContextSharing` / `BuildError` | `rust/.../tools/subagent.rs` | `typescript/.../tools/src/subagent.ts`; `subagent-consult.test.ts` passes | INFERRED SIMILARITY | — |

### 4.3 Sandbox

| # | Capability | Rust evidence | TS behavior | Status | Sev |
|---|---|---|---|---|---|
| S1 | `SandboxProvider` (`validate`, `execute_command`, `handle_large_output`, `resolve_path`, `isolation_mode`, `workspace_root`) | `rust/.../harness/providers.rs:113-260` | `typescript/.../harness/types.ts:4622-4660` | OBSERVED PARITY | — |
| S2 | **`execute_command_streaming` (with `CommandProgressSink`)** — streaming subprocess output, default delegates to `execute_command` | `rust/.../harness/providers.rs:111,177-200` | **No `executeCommandStreaming`, no `CommandProgressSink`** in TS `SandboxProvider` | **GAP** | major |
| S3 | `WorkspaceScopedSandbox`, `NullSandbox`, `ReadOnlySandbox`, `BwrapProfile`, `NetworkPolicy`, `IsolationMode`, `SandboxViolation`, `ExecConfig` (`kill_on_drop`) | `rust/.../sandbox.rs`, `providers.rs:306-360` | `typescript/.../sandbox/*.ts`, `harness/types.ts:4449-4552` | OBSERVED PARITY | — |
| S4 | `sandbox_petri` (Petri isolation backend, feature-gated) | `rust/.../sandbox_petri.rs` | **No TS equivalent** (feature-gated Rust-only backend) | **GAP** (out of scope — known gap, §9) | minor |

### 4.4 Model providers, streaming, recording/replay, prompt-tool-call, tool-call-repair

| # | Capability | Rust evidence | TS behavior | Status | Sev |
|---|---|---|---|---|---|
| M1 | `ModelInterface` (call/countTokens/stream), `ModelRequest/Response`, `Message`, `ContentBlock` (text/image/tool_use/tool_result), `ToolCall/ToolResult/ToolSchema`, `TokenUsage`, `StopReason`, `Role`, `ModelParams`, `ProviderInfo` | `rust/.../model.rs` | `typescript/.../model/{interface,types,schemas}.ts` | OBSERVED PARITY | — |
| M2 | `request_hash` determinism (canonical JSON) | `rust/.../model.rs` (`request_hash`) | `typescript/.../model/hash.ts`; `model-hash.test.ts` passes | OBSERVED PARITY | — |
| M3 | `enforce_budget` / `enforce_context_limit` | `rust/.../model.rs` | TS `model/index.ts` exports | INFERRED SIMILARITY | — |
| M4 | `RecordingModelInterface` + `RecordingMode` + `RecordedExchange`; `ReplayModelInterface` + `ReplayMode` | `rust/.../model.rs` | `typescript/.../model/{recording,replay}.ts`; `recording.test.ts`/`replay.test.ts` pass | OBSERVED PARITY | — |
| M5 | `Anthropic`/`OpenAI`/`Ollama` provider HTTP shape, auth, endpoint, request/response + tool-call mapping | `rust/.../{anthropic,openai,ollama,provider_http}.rs` | `typescript/.../model/{anthropic,openai,ollama}.ts`; provider tests + fixture replay pass | INFERRED SIMILARITY (TS uses Node `fetch`; no shared `provider_http` client) | — |
| M6 | `image.rs` helpers: `media_type_for_extension`, `is_image_path`, `image_content`, `load_image_content` | `rust/.../image.rs:16-50` | **No `image.ts` module / no image helpers** (the `image` ContentBlock *type* exists in `schemas.ts:46-48`) | **GAP** | minor |
| M7 | `PromptBasedToolCallModelInterface` + `AdaptiveToolCallModelInterface` + `inject_tool_prompt` + `parse_prose_response` | `rust/.../prompt_tool_call.rs` | `typescript/.../model/prompt-tool-call.ts` (all four present) | OBSERVED PARITY | — |
| M8 | **`ToolCallRepair` trait + `StandardToolCallRepair` + `coerce_tool_args` + `ToolCallFailure`** | `rust/.../tool_call_repair.rs:45,227,252,332-345` | **Absent in TS** (see L10) | **GAP** | major |
| M9 | `detect_prose_response` | `rust/.../tool_call_repair.rs:252` | `typescript/.../model/prompt-tool-call.ts:282` | OBSERVED PARITY | — |

### 4.5 Context, compaction, prompt assembly, prompt chunk registry, cache provider, output schema

| # | Capability | Rust evidence | TS behavior | Status | Sev |
|---|---|---|---|---|---|
| C1 | `ContextManager` + `StandardContextManager`; `Context`/`ContextMeta`/`ContextSources`/`CompactionConfig`/`CompactionRequest`/`CompactionResult`/`CompactionVerifier`/`KeyTermVerifier`/`CompactionVerificationResult`; `BreakpointInfo`/`CacheBlock*`/`SegmentStability`/`RenderedSystemPrompt`/`ComposedPrompt`/`PromptSegment`; `DEFAULT_CONTEXT_LENGTH` | `rust/.../context.rs` | `typescript/.../context/*.ts` | OBSERVED PARITY | — |
| C2 | `seed_rich_state` + `StandardCompactionAdapter` + `RICH_STATE_KEY` + `HarnessContextManagerExt` | `rust/.../compaction_adapter.rs` | `typescript/.../context/compaction-adapter.ts` | OBSERVED PARITY | — |
| C3 | `prompt_assembly` engine (`AssemblyBuckets`, `AssemblyContext`, `ChunkProvider`, `ChunkCondition`, `Composite/Embedded/InMemoryChunkProvider`, `ContextSourcesBuilder`, `ToolAffinity`, `breakpoint_ids`, `chunks_to_segments`) | `rust/.../prompt_assembly.rs` | `typescript/.../prompt-assembly/*.ts`; `prompt-assembly-fixture-replay.test.ts` (35 tests) passes | OBSERVED PARITY | — |
| C4 | `prompt_chunk_registry` (`standard_chunks`, `ApprovalPolicy`, `CacheBlock`, `ChunkId`, `ChunkSlot`, `Mode`, `PromptChunk`, `StandardPromptChunkRegistry`) | `rust/.../prompt_chunk_registry.rs` | `typescript/.../prompt-chunk-registry/*.ts` | OBSERVED PARITY | — |
| C5 | `cache_provider` (`auto_detect`, `Anthropic`/`OpenAI`/`Ollama`/`Null`, `CacheAnnotationResult`, `CacheStats`) | `rust/.../cache_provider.rs` | `typescript/.../cache-provider/*.ts`; `cache-provider.test.ts` passes | OBSERVED PARITY | — |
| C6 | `output_schema` enforcement (retry/violation) | `rust/.../output_schema.rs` + react loop | `typescript/.../output-schema/*.ts`; `harness-output-schema.test.ts` passes | OBSERVED PARITY | — |

### 4.6 Storage, observability, sensor, middleware, hooks, termination, verifier, metric, guide registry, memory, skills, plan, tasklist, execution registry

| # | Capability | Rust evidence | TS behavior | Status | Sev |
|---|---|---|---|---|---|
| X1 | `StorageProvider` + `FileSystem`/`InMemory`/`NoOp`/`Composite` + `ScopedMemoryRouter` + `SessionStore`/`MemoryStore`/`RunStore`/`ObservabilityStore` + `ProjectId`/`WorkspaceId` | `rust/.../storage.rs` | `typescript/.../storage/*.ts`; `storage.test.ts` passes | OBSERVED PARITY | — |
| X2 | Active-run lifecycle (`start_or_resume_active_run`, `complete_active_run`, `load_active_run`, `ActiveRun`/`ActiveRunDecision`/`ActiveRunStatus`) + `parse_otlp_endpoints` | `rust/.../storage.rs:446-560,1501` | `typescript/.../storage/{types,providers}.ts` | OBSERVED PARITY | — |
| X3 | `ObservabilityProvider` + span types + `GenAiMessage`/`GenAiRole` + `ContextSpan`/`ToolCallSpan`/`ToolResultContent`/`TurnSpan`/`MiddlewareSpan`/`PatchSpan`/`SensorSpan` + `truncate_field` + `ContentCaptureConfig` + `PricingTable` + `SessionMetrics` + `InMemoryObservabilityProvider` | `rust/.../observability.rs` | `typescript/.../observability/*.ts` | OBSERVED PARITY | — |
| X4 | `OutboxObservabilityProvider` + `OutboxConfig` + `TraceLine` | `rust/.../observability_outbox.rs` | `typescript/.../observability/outbox.ts` | OBSERVED PARITY | — |
| X5 | `Sensor`/`SensorChain`/`SensorConfig`/`SensorOutcome`/`SensorSignalThresholds`/`SensorStats` | `rust/.../sensor.rs` | `typescript/.../sensor/*.ts` | OBSERVED PARITY | — |
| X6 | `Middleware`/`MiddlewareChain`/`HookPoint`/`MiddlewareDecision` + `TokenBudget`/`LoopDetection`/`PatchToolCalls`/`PreCompletionChecklist`/`Tracing` middlewares | `rust/.../middleware.rs` | `typescript/.../middleware/*.ts` | OBSERVED PARITY | — |
| X7 | `Hook`/`HookChain`/`HookEvent`/`HookDecision`/`HookContext`/`CommandHook`/`FunctionHook`/`FireOutcome`/`PlanArtifact`/`TurnOutput`/`ToolCallSummary` | `rust/.../hooks.rs` | `typescript/.../hooks/*.ts` | OBSERVED PARITY | — |
| X8 | `TerminationPolicy` + checks (`AlwaysComplete`/`QuestionAnswered`/`TestSuite`/`SqlResult`/`FeatureList`/`Fixed`/`Null`) + `check_budget_default` + `SessionStateSnapshot` | `rust/.../termination.rs` | `typescript/.../termination/*.ts` | OBSERVED PARITY | — |
| X9 | `Verifier` + `CompositeVerifier`/`TestSuiteVerifier`/`EvaluatorResponseVerifier` | `rust/.../verifier.rs` | `typescript/.../verifier/*.ts` | OBSERVED PARITY | — |
| X10 | `Metric` (`should_keep`, `CommandMetricEvaluator`, `TestPassRateEvaluator`, `LatencyEvaluator`, `LlmJudgeEvaluator`, `JudgeModelConfig`, `IterationStatus`, `ResultsEntry`) | `rust/.../metric.rs` | `typescript/.../metric/*.ts` | OBSERVED PARITY | — |
| X11 | `GuideRegistry` + `StandardGuideRegistry` + `Guide`/`GuideQuery`/`GuideStatus`/`GuideSource`/`GuideType`/`GuideConflict`/`ImprovementSignal`/`PendingReason`/`SessionOutcome` | `rust/.../guide_registry.rs` | `typescript/.../guide-registry/*.ts` | OBSERVED PARITY | — |
| X12 | `MemoryProvider` + `Episodic`/`Semantic` + `MemoryQuery`/`MemorySource`/`MemoryStatus`/`MergeStrategy` + `StandardMemoryProvider` (in-memory HashMap, like Rust) | `rust/.../memory.rs` | `typescript/.../memory/*.ts` | OBSERVED PARITY | — |
| X13 | `skills` (`parse_skill_doc`, `SkillCatalog`, `SkillEntry`, `LOAD_SKILL`) | `rust/.../skills.rs` | `typescript/.../skills/*.ts` | OBSERVED PARITY | — |
| X14 | `plan` (`capture_plan_artifact`, `PlanPhaseError`, `PLAN_EXECUTE_EXTRAS_KEY`) | `rust/.../plan.rs` | `typescript/.../plan/*.ts` | OBSERVED PARITY | — |
| X15 | `tasklist` (`plan_artifact_to_task_list` [deprecated], `TaskList`, `TaskStatus`, `TASK_LIST_EXTRAS_KEY`, `push_step_ledger`, `render_step_ledger`, `StepLedgerEntry`, `STEP_LEDGER_*`) | `rust/.../tasklist.rs` | `typescript/.../tasklist/*.ts` | OBSERVED PARITY | — |
| X16 | `ExecutionRegistry`/`ExecutionRegistryBuilder`/`EscalationMode`/`AutoGrantInfo`/`StrategyResolution`/`GrantCallback` | `rust/.../execution_registry.rs` | `typescript/.../harness/execution-registry.ts` | OBSERVED PARITY | — |
| X17 | **`vector_store` (`VectorStore`, `VectorRecord`, `VectorHit`, `MetadataFilter`, `VectorStoreError`)** — full trait (add/search/delete/len/is_empty), upsert-by-id, dimension-mismatch rejection, AND-filter, in-memory reference impl | `rust/.../vector_store.rs` (345 lines, in spore-core) | **No TS module.** `grep VectorStore` TS → 0 hits. | **GAP** (IN SCOPE — §5 G-L) | major |
| X18 | **`embedding` (`cosine_similarity`, `EmbeddingInfo`, `EmbeddingModel`, `OllamaEmbeddingModel`, `OpenAIEmbeddingModel`)** — full provider behavior: Ollama `/api/embed` (no key, fail-fast on connect, `probe_dimensions`), OpenAI `${base}/embeddings` (bearer auth, shared retry loop + `{error:{message}}` envelope, `dimensions` param for `text-embedding-3-*`, sort-by-index, `from_env`, `default_dimensions`) | `rust/.../embedding.rs` (550 lines, in spore-core) | **No TS module.** TS `memory/standard.ts:10` notes embedding similarity is spec'd but unimplemented. | **GAP** (IN SCOPE — §5 G-L) | major |
| X19 | `text_util` (internal helpers) | `rust/.../text_util.rs` | TS uses inline equivalents (`truncateField` etc.) | INFERRED SIMILARITY | — |
| X20 | `sync_util` (`BoxFut` object-safe async alias) | `rust/.../sync_util.rs` | N/A (TS uses native `Promise`/`async`) | OBSERVED PARITY (idiomatic) | — |
| X21 | `macros.rs` (derive macros) | `rust/.../macros.rs` | N/A (TS has no proc-macro equivalent; zod schemas substitute) | OBSERVED PARITY (idiomatic) | — |

### 4.7 CLI, eval, examples, docs, project config

| # | Capability | Rust evidence | TS behavior | Status | Sev |
|---|---|---|---|---|---|
| E1 | `spore-cli` binary | `rust/crates/spore-cli/src/main.rs` = `fn main() {}` (stub) | No TS `spore` CLI binary (none needed — Rust stub has no behavior) | OBSERVED PARITY | — |
| E2 | `spore-eval` CLI (`run <suite> [--candidate ..] [--n N] [--json]`, `promote <suite> <task_id>`, `-h/--help/help`, exit codes) | `rust/crates/spore-eval/src/bin/spore-eval.rs:1-115` | `typescript/packages/eval/src/bin/spore-eval.ts:1-120` (mirrors subcommands/flags/exit codes) | OBSERVED PARITY | — |
| E3 | `EvalHarness` library API (`harness`/`task`/`manifest`/`report`/`stats`/`verifier`/`metric_map`/`worktree`) | `rust/crates/spore-eval/src/*.rs` | `typescript/packages/eval/src/*.ts`; `eval-harness.test.ts` (36) + `stats.test.ts` (10) pass | OBSERVED PARITY | — |
| E4 | `e2e-agent` tool binary | `rust/.../tools` (Rust exposes tools via catalogue, no separate e2e bin) | `typescript/packages/tools/src/bin/e2e-agent.ts` (TS-only convenience binary) | OBSERVED PARITY (TS additive) | — |
| E5 | Examples 01–12 | `examples/rust/01..12` | `examples/typescript/01..12` (one-to-one) | OBSERVED PARITY | — |
| E6 | Docs: `building-a-tool.md`, `conversation.md`, `custom-harness.md` | `docs/rust/{building-a-tool,conversation,custom-harness,quickstart}.md` | `docs/typescript/` has only stub `quickstart.md` | **GAP** (IN SCOPE — §5 G-M) | minor |
| E7 | Workspace/build config | Cargo workspace + feature flags (`petri`) | pnpm workspace, no feature flags | OBSERVED PARITY (idiomatic) | — |
| E8 | `spore-mcp` crate | `rust/crates/spore-mcp` (511 lines) | **No TS equivalent** | **GAP** (out of scope — known gap, §9) | major |
| E9 | `spore-sqlvec` crate | `rust/crates/spore-sqlvec` | **No TS equivalent** | **GAP** (out of scope — known gap, §9) | major |

## 5. Concrete gaps to close (implementation work, after plan approval)

Grouped by area. Each item lists: TS file-level steps, regression/E2E validation,
compatibility constraints, risk.

### G-A. Stream-event node attribution (L4) — major — **DONE**
- **Steps:** add `NodeAttr` type + optional `node?: NodeAttr` field to every
  `HarnessStreamEvent` variant in `typescript/.../harness/types.ts`; add
  `wrapNodeSink(parent, kind, agentId)` helper mirroring `wrap_node_sink`
  (set-once `kind`/`agent_id`, prepend `kind` to `path`, bump `depth`); install
  wrappers in the react leaf and each combinator (plan_execute, self_verifying,
  hill_climbing, ralph) where they build child harnesses in
  `typescript/.../harness/standard.ts`.
- **Validation:** new test asserting a multi-agent run stamps events with the
  full `path` and leaf `agent_id`; existing fixture-replay tests stay green
  (the field is optional, so old serialized events round-trip).
- **Compat:** optional field → back-compatible. Risk: low.

### G-B. `ToolProgress` stream event + live progress sink (L5, T2, S2) — major — **DONE**
- **Steps:** add `tool_progress` variant to `HarnessStreamEvent`; add
  `ToolProgressUpdate` + `ToolProgressSink` to `tool-registry/types.ts`; add
  optional `commandProgress(call)` on `ToolContext`; add `CommandProgressSink`
  type + optional `executeCommandStreaming(..., progress?)` to `SandboxProvider`
  in `harness/types.ts` (default delegates to `executeCommand`); wire the run
  loop to install a progress sink that emits `tool_progress` events; update
  `ExecTool`/`BashCommandTool` in `typescript/.../tools/src/exec.ts` to call
  `executeCommandStreaming` with `ctx.commandProgress(call)`; forward
  `executeCommandStreaming` through `ReadOnlySandbox`/`WorkspaceScopedSandbox`.
- **Validation:** new test: a long-running exec emits successive `tool_progress`
  previews (tail snapshots, replace-not-append) between `tool_call` and
  `tool_result`; default sandbox without streaming still completes.
- **Compat:** optional method on `SandboxProvider` + optional event field →
  back-compatible. Risk: medium (sandbox wrapper forwarding must be correct or
  streaming silently dies).

### G-C. `tool_result.revised` + `user_message.steered` fields (L6, L7) — major — **DONE**
- **Steps:** add optional `revised?: boolean` to `tool_result` and optional
  `steered?: boolean` to `user_message` in `HarnessStreamEvent`; emit
  `revised:true` when an AfterTool middleware rewrites a result (wire in the
  react tool-dispatch path); emit `steered:true` for steering-injected messages
  (depends on G-D), `steered:false` (or omitted) for `send_message` narration.
- **Validation:** test that an AfterTool `continue_with_modification` emits a
  revised `tool_result`; test that a steering injection emits `steered:true`.
- **Compat:** optional fields → back-compatible. Risk: low.

### G-D. Steering source (L9) — major — **DONE**
- **Steps:** add `SteeringSource` type (`() => string[]`) + optional
  `steeringSource?` to `HarnessConfig` and `HarnessBuilder.steeringSource(..)`;
  in the react loop, at the top of each turn iteration, drain
  `config.steeringSource()`, append each via `contextManager.appendUserMessage`,
  emit `user_message{steered:true}`.
- **Validation:** new test: a run with a steering source that queues a message
  mid-run injects it before the next model call, emits the steered event, and
  the message persists in the transcript.
- **Compat:** optional config → back-compatible. Risk: low.

### G-E. Tool-call repair seam (L10, M8) — major — **DONE**
- **Steps:** add `ToolCallRepair` interface + `StandardToolCallRepair` class +
  `coerceToolArgs` + `ToolCallFailure` in a new
  `typescript/.../src/tool-call-repair/` module (mirror
  `rust/.../tool_call_repair.rs`); add optional `toolCallRepair?` +
  `maxRepairAttempts?` to `HarnessConfig`/builder (default `1`); in the react
  tool-dispatch path, on `invalid_parameters`, attempt repair up to
  `maxRepairAttempts` before returning the error.
- **Validation:** new test: a tool call with coerced args succeeds after
  repair; a non-repairable call still returns `invalid_parameters`; the
  default (no repair configured) preserves today's behavior byte-for-byte.
- **Compat:** optional config → back-compatible. Risk: medium (must not change
  the no-repair path).

### G-F. Tool-call nudge (L11) — major — **DONE**
- **Steps:** add optional `toolCallNudge?` to `HarnessConfig`/builder; in the
  react loop track `toolsCalledThisWindow` and `toolCallNudgesUsed` (cap
  `MAX_TOOL_CALL_NUDGES=2`); when a turn ends having called zero tools and the
  flag is on, inject the behavioral nudge user message.
- **Validation:** new test: a model that answers without calling tools gets
  nudged up to the cap, then the loop proceeds; flag off → no nudge
  (byte-identical).
- **Compat:** optional config, default off → back-compatible. Risk: low.

### G-G. Plan-empty re-ask (L12) — major — **DONE**
- **Steps:** add optional `planEmptyMaxRetries?` to `HarnessConfig`/builder
  (default `0`); in `captureAndPersistPlan`/plan-phase path, on an empty
  captured plan, re-ask up to `planEmptyMaxRetries` with tools suppressed +
  corrective nudge, fold discarded usage, then halt `empty_plan` if still
  empty. Default `0` preserves single-shot behavior.
- **Validation:** new test: a planner that emits an empty plan twice then a
  real plan succeeds when `planEmptyMaxRetries=2`; default `0` halts
  `empty_plan` immediately (byte-identical to today).
- **Compat:** optional config, default `0` → back-compatible. Risk: medium.

### G-H. `set_plan` tool + `PlanRecorded` signal (L13, T3) — major — **DONE**
- **Steps:** add `plan_recorded` variant to `HarnessSignal`; add `SetPlanTool`
  in `typescript/.../tools/src/` (mirror `rust/.../tools/set_plan.rs`: writes
  the durable `task_list` store, escalates `PlanRecorded`); add `setPlan()`
  constructor to `StandardTools` and include it in `codingSet`/`fullSet`; in
  the plan phase, handle `RunResult::Escalate{signal:PlanRecorded}` by
  proceeding with empty `plan_output` (SC-28 fallback loads the durable list).
- **Validation:** new test: a planner using `set_plan` terminates the plan
  phase and the execute phase reads the durable list; the `PlanRecorded`
  signal is consumed (never ends the whole run).
- **Compat:** additive tool + new signal variant → back-compatible. Risk:
  medium (plan-phase control flow).

### G-I. `create_dir` tool + catalogue gaps (T4, T5, T6) — major — **DONE**
- **Steps:** add `CreateDirParamsSchema` to `params.ts` + `CreateDirTool` to
  `fs.ts` (mirror `rust/.../tools/fs.rs:276`); add `createDir()`, `moveFile()`,
  `deleteFile()` constructors to `StandardTools`; add `create_dir`, `move_file`,
  `delete_file`, `set_plan` to `codingSet` (and thus `fullSet`).
- **Validation:** new tests for `create_dir` (mkdir -p, idempotent, sandbox
  validation); catalogue test asserting `codingSet` membership matches Rust.
- **Compat:** additive tools → back-compatible. Risk: low.

### G-J. Narration refund (L14) — major — **DONE**
- **Steps:** in the react loop, track `narrationRefund`; when a turn's only
  tool call is `send_message` (success), refund one turn against the
  productive-turn cap (capped at `effectiveTurnCap` so a pure-narration loop
  still terminates).
- **Validation:** new test: a run whose turns are all `send_message` narration
  does not consume the productive budget up to ~2× cap, then terminates.
- **Compat:** behavior change only when `send_message` is in the toolset;
  preserves turn-count monotonicity. Risk: medium (budget accounting).

### G-K. Image helpers (M6) — minor — **DONE**
- **Steps:** add `typescript/.../model/image.ts` with `mediaTypeForExtension`,
  `isImagePath`, `imageContent`, `loadImageContent` (mirror
  `rust/.../image.rs`); export from `model/index.ts`.
- **Validation:** unit tests for extension → media-type mapping and
  `loadImageContent` round-trip.
- **Compat:** additive. Risk: low.

### G-L. VectorStore + EmbeddingModel traits + full Ollama/OpenAI providers (X17, X18) — major — **DONE**

Confirmed in scope by second-mate review. This is **not** a trait-shape stub:
the TS implementation must reproduce the full Rust provider behavior, validated
against mocked HTTP (the Rust side uses `wiremock`; the TS side will use a
mocked `fetch` / `MockServer` equivalent, consistent with the existing
`ollama.test.ts` / `openai.test.ts` provider tests).

**New modules:**
- `typescript/packages/core/src/embedding/{index.ts,types.ts,standard.ts}`
  - `EmbeddingModel` interface: `embed(inputs: string[]): Promise<number[][]>`,
    `embedOne(input): Promise<number[]>` (default → one-element `embed`,
    `ProviderError` if zero vectors returned), `dimensions(): number`,
    `info(): EmbeddingInfo`.
  - `EmbeddingInfo` (`provider`, `model_id`, `dimensions`).
  - `cosineSimilarity(a, b): number` — `[-1,1]`, `0.0` on length-mismatch OR
    zero-magnitude (never `NaN`).
  - `OllamaEmbeddingModel`: `DEFAULT_BASE_URL="http://localhost:11434"`,
    `DEFAULT_TIMEOUT=120s`, `new(modelId)`, `withBaseUrl`, `withDimensions`,
    `withTimeout`, `withHttpClient` (injectable for mocked fetch),
    `probeDimensions()` (embed `"dimension probe"`, return length, no
    mutation). `embed`: empty input → `[]` without contacting the server;
    POST `${base}/api/embed` `{model, input}`; map `is_timeout`→`Timeout`,
    `is_connect`→`ProviderError(0, "Ollama not running at <base>")`, else
    `Transport`; non-2xx → `ProviderError(code, first 500 chars of body)`;
    decode `{embeddings: number[][]}`.
  - `OpenAIEmbeddingModel`: `DEFAULT_BASE_URL="https://api.openai.com/v1"`,
    `DEFAULT_TIMEOUT=120s`, `DEFAULT_MAX_RETRIES=3`, `new(apiKey, modelId)`,
    `fromEnv(envVar, modelId)` (reject unset/empty with `ProviderError`),
    `withBaseUrl`, `withDimensions`, `withTimeout`, `withMaxRetries`,
    `withHttpClient`. `defaultDimensions`: `text-embedding-3-large`→3072,
    `text-embedding-3-small`|`text-embedding-ada-002`→1536, else 0.
    `supportsDimensionsParam` iff `modelId` starts with `text-embedding-3`.
    `embed`: empty input → `[]`; POST `${base}/embeddings` with
    `Authorization: Bearer <key>`; body `{model, input, dimensions?}` where
    `dimensions` is sent only when pinned AND `supportsDimensionsParam`;
    **reuse the existing OpenAI retry plumbing** (`sendWithRetry` /
    `backoffDelayMs` / `RETRYABLE_STATUSES` / `parseRetryAfter` in
    `model/openai.ts`) — or extract a shared `provider-http` helper first to
    avoid the current duplication (mirrors Rust's `provider_http::send_with_retry`);
    retryable set `408|425|429|500|502|503|504`, exponential backoff 0.5s→
    30s cap, honor `retry-after`; non-retryable → `ProviderError(code, msg)`
    from the `{error:{message}}` envelope (fallback first 500 chars); decode
    `{data:[{embedding,index}]}`, **sort by `index`** so a non-conforming
    compatible server cannot scramble input↔vector order.
  - `Debug`/redaction: `OpenAIEmbeddingModel` must not log the key (mirror
    Rust's `<redacted>`).
- `typescript/packages/core/src/vector-store/{index.ts,types.ts,standard.ts}`
  - `VectorRecord` (`id`, `vector: number[]`, `metadata: unknown`), `new(...)`.
  - `VectorHit` (`id`, `score`, `metadata`), higher = more similar.
  - `MetadataFilter`: `equals: [key, value][]`, `new()`, `eq(k,v)` builder,
    `isEmpty()`, `matches(metadata)` (AND of equalities).
  - `VectorStoreError` (discriminated): `dimension_mismatch {expected, actual}`,
    `backend {message}`.
  - `VectorStore` interface: `add(records)`, `search(query, topK, filter?)`,
    `delete(filter)` (empty filter = full clear, returns count), `len()`,
    `isEmpty()` (default `len()==0`). `add` rejects a record whose vector
    length ≠ store dimensionality with `dimension_mismatch`; upsert by `id`.
  - `InMemoryVectorStore` (reference backend, mirror Rust's test impl):
    `new(dimensions)`, cosine `search` sorted desc with `truncate(topK)`,
    filter applied as part of search (not post-truncation), `delete` via
    `retain(!filter.matches)`.
- Export both from `typescript/packages/core/src/index.ts` (`export * as vectorStore ...` /
  `export * as embedding ...`, matching the existing namespace-export style).

**Validation (mocked HTTP, not stubs):**
- `cosineSimilarity`: identical vectors → 1.0; orthogonal → 0.0; zero-magnitude
  → 0.0; length-mismatch → 0.0 (never `NaN`).
- Ollama: mocked `/api/embed` round-trip returns vectors in input order; empty
  input skips the call (no mock mounted, unreachable base URL); connect error
  → `ProviderError(0, "Ollama not running at <base>")`; non-2xx →
  `ProviderError(code, body[:500])`; `probeDimensions` returns the vector
  length.
- OpenAI: mocked `/embeddings` with `Authorization: Bearer <key>` header
  assertion; data returned out-of-order by `index` is reordered to input
  order; `text-embedding-3-small` default `dimensions=1536` and the request
  body includes `dimensions` only for `text-embedding-3-*`; `fromEnv` rejects
  unset/empty; 400 `{error:{message}}` → `ProviderError(400, msg)` with
  `maxRetries=0`; 429 retried then succeeds (with a no-sleep test hook);
  `retry-after` header honored.
- `InMemoryVectorStore`: add/search orders by similarity; filter scopes search;
  upsert does not duplicate; delete returns count; dimension mismatch rejected;
  `len`/`isEmpty` correct.
- Object-shape: a `VectorStore`/`EmbeddingModel` value is usable where the
  interface is expected (TS structural typing; mirrors Rust's `Box<dyn>`/
  `Arc<dyn>` object-safety test).

**Compat:** additive modules/exports → back-compatible. Risk: low–medium
(retry plumbing reuse must not change the chat providers' behavior).

### G-M. Docs (E6) — minor — **DONE**

Confirmed in scope by second-mate review.
- **Steps:** authored `docs/typescript/building-a-tool.md`,
  `conversation.md`, `custom-harness.md` mirroring the Rust pages' structure
  with TS code (imports from `@spore/core` / `@spore/tools`); cross-linked from
  `docs/typescript/quickstart.md` and removed its "stub" notice.
- **Validation:** doc links resolve; code samples reference the implemented API.
- **Compat:** docs-only. Risk: low.

### G-N. Pre-existing tsc errors (baseline) — minor — **DONE**
- **Steps:** fixed all pre-existing `tsc --noEmit` errors across `core`, `tools`,
  and `eval` packages: `noUncheckedIndexedAccess` array accesses in ~16 test
  files (non-null assertions), API drift in `compaction-adapter`/`harness-compaction-loop`
  (`WarnEvent` discriminated-union narrowing), `execution-registry` (`ModelInterface`
  test-double missing `callStreaming`/`provider`), `harness-catalogue-wiring`
  (`ToolRegistry` seam type — harness-side vs registry-side), `harness-conversational`
  (`ProviderInfo.context_window`), `harness.test` (`SandboxViolation.path_denied.matched_rule`,
  `ToolRegistry` import), `presets`/`presets.test` (`MetricEvaluator`/`MetricOutcome`/`SessionStateSnapshot`
  moved to `metric`/`termination` namespaces), `set-plan` (`taskListErrorMessage` arg
  `TaskListError.detail`), `memory` (`params.limit ?? 50` default), `eval/verifier`
  (`RunResult.consult` variant in `sessionIdOf`, `LlmJudgeVerifier.verify` workspace param).
- Added `typecheck` script (`tsc --noEmit`) to each `package.json`.
- **Validation:** `pnpm -r typecheck` clean (0 errors across core/tools/eval).
- **Compat:** test-only + minor src type fixes. Risk: low.

## 5.2 Executable lifecycle, cancellation, timeout, retry, cleanup & async/concurrency scenarios

These are concrete, executable test scenarios (not new gaps) that pin the
observable lifecycle behavior the harness must share with Rust. Each scenario
names the seam under test, the Rust reference behavior (file:line), the TS
location, and the assertion. They are added to the implementation's regression
suite. Rust is read-only evidence; the TS behavior is exercised against a
mocked model + sandbox so no real provider is contacted.

### 5.2.1 Cancellation (AbortSignal)

**Rust has no `AbortSignal` equivalent.** External cancellation in Rust is
*dropping the `run` future*: the caller stops polling and `tokio` cancels the
in-flight `await` points (the model HTTP request, the subprocess). No
`RunResult` is produced — the future is simply not awaited to completion.
`HaltReason::AgentError` (`rust/.../harness/types.rs:1432`) fires only when the
*model returns* a `TurnResult::Error` (`rust/.../harness/react.rs:1972-1984`),
NOT on external cancel. TypeScript cannot drop a `Promise`, so it threads
`AbortSignal` through every async boundary (CONVENTIONS.md) — the observable
parity target is therefore "aborting stops further turns and reaps in-flight
work," with the TS-specific terminal being a typed failure.

- **C1 — Mid-turn abort stops the loop and reaps in-flight work.** Abort
  `HarnessRunOptions.signal` while a model call is in flight. TS:
  `model.call(request, signal)` rejects with `ProviderError(0, "request aborted: …")`
  (`typescript/.../model/{anthropic,openai,ollama}.ts` AbortError mapping;
  `ProviderError extends ModelError` — `model/errors.ts:45`),
  `ModelAgent.turn` catches `err instanceof ModelError` and returns
  `TurnResult::Error` (`typescript/.../agent/model-agent.ts:167-172`), and the
  react loop routes that to `HaltReason::AgentError` → `RunResult.failure`
  (mirroring Rust's `TurnResult::Error`→`AgentError` routing at
  `rust/.../harness/react.rs:1972-1984`). **Assertion:** the run returns
  `RunResult.failure{reason:{kind:"agent_error"}}`; `turns` reflects the aborted
  turn; no further turns run; the stream emitted `turn_start` for the aborted
  turn but no `turn_end`. (This TS terminal is the idiomatic equivalent of
  Rust's dropped future — Rust itself produces no result here.)
- **C2 — Abort between turns stops the loop.** Abort the signal after a turn
  completes but before the next model call. **Assertion:** the loop does not
  start the next turn; the partial `usage`/`turns` are reported on the failure.
  (Rust equivalent: the caller drops the future at the turn boundary.)
- **C3 — Abort propagates to tool dispatch and reaps the child.** Abort during
  a long-running tool (exec). Rust: dropping the run future drops the
  `execute_command_streaming` future, which drops the tokio `Child` (killed on
  drop). TS: the tool's `execute(call, sandbox, ctx, signal)` threads the signal
  to `sbExecuteCommand` (`typescript/.../tools/src/exec.ts:117`); the sandbox's
  `AbortController` aborts the child (`workspace-sandbox.ts:421-451`).
  **Assertion:** the exec returns a recoverable `error` (timed_out/aborted) and
  the child is reaped (no orphan) — see 5.2.4 CL1.
- **C4 — Abort is non-destructive to session state (TS contract).** In TS the
  abort fires inside `await model.call(...)` BEFORE the assistant message is
  appended (the append happens after `classifyResponse`). **Assertion:** after
  C1, resuming with the carried `session_state` continues losslessly — the
  aborted turn produced no assistant message, so nothing is double-recorded; a
  resume run's transcript equals a single uninterrupted run's transcript for
  the same model responses. (Rust has no resume-after-drop path; this is a
  TS-specific contract pinned by test.)

### 5.2.2 Timeout

- **T1 — Per-exec timeout.** `exec`/`bash_command` with `timeout` (seconds).
  Rust: `rust/.../tools/exec.rs:104` → `execute_command_streaming(.., timeout,..)`;
  timeout → `CommandOutput{timed_out:true, exit_code:-1}` → recoverable `error`
  `"command timed out after Ns"`. TS: `typescript/.../tools/src/exec.ts:124-131`.
  **Assertion:** a sleeping command with `timeout=1` returns a recoverable error
  with the exact message shape; `stderr`/`stdout` captured up to the abort are
  not lost when the sandbox buffers them.
- **T2 — Model-call timeout.** A provider timeout (`OllamaEmbeddingModel`/
  `OpenAIEmbeddingModel`/chat providers' `withTimeout`) maps to `ModelError::Timeout`
  (Rust `rust/.../provider_http.rs:138-141`, `rust/.../embedding.rs` timeout
  branch). TS: `model/{anthropic,openai,ollama}.ts` map timeout/ETIMEDOUT to
  `Timeout`. **Assertion:** a mocked provider that never responds within the
  timeout returns `Timeout` (not `ProviderError`); on the embedding providers
  (G-L) a timeout with `maxRetries>0` is retried (5.2.3) and only the final
  exhaustion surfaces `Timeout`.
- **T3 — Budget timeout.** `BudgetLimits.time_ms` overrun halts with
  `budget_exceeded{limit_type:"time"}`. Rust `rust/.../harness/budget.rs`;
  TS `typescript/.../harness/standard.ts:3160` (`budgetExceeded`). **Assertion:**
  a run whose wall-clock exceeds `time_ms` between turn gates halts with the
  `time` limit type (checked at the turn gate, not mid-turn).

### 5.2.3 Retry

- **R1 — Hosted-provider HTTP retry.** A 429/503 response followed by a 200.
  Rust: `provider_http::send_with_retry` retries `408|425|429|500|502|503|504`
  up to `max_retries`, exponential backoff 0.5s→30s cap, honors `retry-after`
  (`rust/.../provider_http.rs:25-160`). TS: `sendWithRetry` in
  `model/{anthropic,openai}.ts` (same set/backoff/retry-after). **Assertion:**
  with a no-sleep test hook, a 429-then-200 succeeds on attempt 2 and the run
  continues; a 400 (non-retryable) is NOT retried and surfaces
  `ProviderError(400, …)`.
- **R2 — Retry exhaustion.** `maxRetries=0` + 429 → immediate `ProviderError`;
  `maxRetries=2` + three consecutive 503s → `ProviderError(503, …)` after the
  final retry. **Assertion:** the attempt count is bounded and the final status
  code/message is preserved.
- **R3 — Embedding provider retry (G-L).** `OpenAIEmbeddingModel` reuses the
  same retry loop; a 500-then-200 succeeds; `retry-after` header is honored over
  the computed backoff; Ollama has NO retry (fail-fast) — a connect error is a
  terminal `ProviderError(0, "Ollama not running at …")`.
- **R4 — Tool-call repair retry (G-E).** `maxRepairAttempts=1`: a malformed
  tool call whose args are coercible succeeds after one repair; a
  non-coercible call returns `invalid_parameters` after the budget is spent;
  default (no repair configured) → single-shot, byte-identical to today.
- **R5 — Plan-empty re-ask (G-G).** `planEmptyMaxRetries=2`: an empty-then-empty
  -then-real plan succeeds; default `0` → `empty_plan` halt immediately
  (byte-identical); discarded attempts' usage is folded into the terminal
  result's `usage`.
- **R6 — Output-schema retry (existing).** `outputSchemaMaxRetries=2`: a
  terminal that fails validation twice then passes succeeds; three failures →
  `output_schema_violation`; budget-cap-wins precedence (a retry that would
  exceed the turn budget surfaces `budget_exceeded` instead). Covered by
  `harness-output-schema.test.ts`; re-asserted for parity.

### 5.2.4 Cleanup

- **CL1 — `kill_on_drop` reaps the child on timeout/abort.** Rust:
  `ExecConfig.kill_on_drop` + the executor's wait loop. TS:
  `typescript/.../sandbox/workspace-sandbox.ts:432-451` (`controller.abort()`
  kills the child; with `kill_on_drop` set, an additional `SIGKILL` guarantees
  reaping). **Assertion:** a timed-out child whose sentinel is never created is
  reaped (no zombie/leak); the process table has no orphan after the run. This
  is already covered by `sandbox.test.ts` (SC-12) and is re-asserted.
- **CL2 — Sandbox wrapper cleanup forwarding.** `ReadOnlySandbox` and any
  combinator child harness must forward `executeCommandStreaming` (G-B) so a
  streamed exec under a read-only-wrapped sandbox still reaps on abort. **Assertion:**
  a streamed exec under `ReadOnlySandbox` emits `tool_progress` then reaps on
  abort.
- **CL3 — Run-finalization stream balance (modulo aborted turns).** On every
  terminal `RunResult` that the loop produces (success, failure from a
  non-abort cause, escalate, consult, waiting_for_human), the harness emits the
  closing `turn_end` for the in-flight turn when the turn completed normally.
  **Assertion:** for those terminals, `turn_start`/`turn_end` counts balance;
  for an aborted in-flight turn (C1) `turn_start` is emitted and `turn_end` is
  NOT — matching Rust, where a dropped future emits no closing event.
  Observability spans are fire-and-forget (`rust/.../harness/react.rs:611`);
  no assertion is made about span flush on the abort/drop path (Rust makes none
  either).
- **CL4 — `complete_active_run` is a standalone storage helper (NOT harness-
  auto-called).** Rust evidence: `complete_active_run` is defined in
  `rust/.../storage.rs:529` and exported from `lib.rs:185`, but is NOT called
  from any harness terminal path (`grep complete_active_run rust/.../harness/*`
  → 0 hits outside `storage.rs`). The harness terminal path calls
  `auto_persist_terminal` (`rust/.../harness/mod.rs:1768`), which persists the
  session, NOT `complete_active_run`. **Assertion (helper contract only):**
  `completeActiveRun` writes the `ActiveRun` slot to `completed` status and
  returns the prior slot; an empty/non-existent slot is handled per its
  documented contract. No assertion is made that the harness invokes it
  automatically — because it does not. (Callers compose it explicitly.)

### 5.2.5 Async / concurrency

- **A1 — The react loop dispatches tools sequentially (one `.await` per call).**
  Rust react loop: `for (i, call) in calls.iter().enumerate() { … .dispatch_with_repair(…).await … }`
  (`rust/.../harness/react.rs:1217-1283`) — a single sequential loop, NOT
  `join_all`. TS react loop mirrors this: `for (const call of …) { … await
  toolRegistry.dispatch(call, signal) … }` (`typescript/.../harness/standard.ts:2061-2065,2143-2147,2155-2159`).
  Both registries ALSO expose a `dispatch_all`/`dispatchAll` that partitions
  read-only tools to run concurrently (`rust/.../tool_registry.rs:389,617`;
  `typescript/.../tool-registry/standard.ts:150-194` via `Promise.all` for the
  `read_only && !destructive && !open_world` prefix) — but the react loop does
  NOT use it; it dispatches one call at a time. **Assertions:** (a) two
  `task_list`/`set_plan`/`todo_write` calls in one turn are applied in order and
  the persisted blob reflects the second call (sequential dispatch preserves
  durable-blob consistency); (b) `dispatchAll` partitions read-only tools
  concurrently and the rest sequentially, returning results in input order
  (parity with `rust/.../tool_registry.rs` `dispatch_all_preserves_input_order`
  test) — tested at the registry level, separate from the react loop.
- **A2 — Supported concurrency boundary (one harness instance = sequential).**
  Rust `StandardHarness::run(&self, …)` takes `&self` and the trait is
  `Send + Sync` (`rust/.../harness/types.rs:1861`), but the struct carries a
  shared `observed_writes: std::sync::Mutex<Vec<String>>` accumulator for the
  *currently-running* execute step, whose comment states "Steps run
  SEQUENTIALLY (v1 ready-set is sequential), so a single shared accumulator
  cleared per step is sufficient and deterministic"
  (`rust/.../harness/mod.rs:1208-1219`). There is NO Rust evidence that
  concurrent `run()` calls on ONE harness instance are supported or isolated —
  the shared accumulator would interleave. TS mirrors this: `StandardHarness`
  holds `private observedWrites: string[] = []` (`typescript/.../harness/standard.ts:599`)
  with no synchronization. **Supported boundary (documented + tested):** a
  single harness instance is used for SEQUENTIAL runs; concurrent runs require
  SEPARATE harness instances. **Assertions:** (a) two sequential `run()` calls
  on one instance with different `SessionId`s produce independent `usage`/`turns`
  and transcripts (per-run local `session_state`/`budget_used` in `run_inner`);
  (b) two runs on SEPARATE harness instances are fully isolated. No assertion
  is made about concurrent calls on a single shared instance — that is
  unsupported and left undefined, matching Rust.
- **A3 — `InMemoryVectorStore` is shared by design; per-instance isolation +
  shared-instance concurrency semantics.** The Rust reference `InMemoryVectorStore`
  holds `rows: Mutex<Vec<VectorRecord>>` shared across all callers of the
  instance (`rust/.../vector_store.rs:300-345`) — it is NOT session/project
  scoped. Scoping is the CALLER's responsibility via `MetadataFilter`
  (`.eq("session_id", …)` / `.eq("project_id", …)`); the store itself has no
  session/project notion. **Assertions:** (a) per-instance isolation — two
  separate `InMemoryVectorStore` instances do not observe each other's
  `add`/`delete`; (b) shared-instance concurrency — concurrent `add` then
  `search` on ONE shared instance are serialized by the mutex and a `search`
  sees all prior `add`s (higher = more similar, sorted desc, truncated to
  `top_k`); (c) scoping is opt-in via `MetadataFilter` — a `search` with a
  filter excludes non-matching records as part of the search, not as a
  post-truncation drop. No assertion is made that the store auto-scopes by
  session/project — because it does not.
- **A4 — Streaming sink is synchronous & inline (no isolation).** `StreamSink`
  is called inline from the run loop (`rust/.../harness/mod.rs:1384` `s(event)`
  with no `catch_unwind`; `StreamSink = Arc<dyn Fn(StreamEvent) + Send + Sync>`).
  TS `StreamSink` is a sync callback invoked inline
  (`typescript/.../harness/standard.ts` `emit`). **Parity assertion:** a sink
  that throws propagates the error out of `run()` (NOT swallowed into a warn) —
  matching Rust, which does not isolate sink panics. A documented non-throwing
  sink is the contract; the test pins that a throwing sink is NOT silently
  swallowed (the run rejects).
- **A5 — Combinator child runs do not reentrantly mutate parent state.** A
  PlanExecute/SelfVerifying/HillClimbing/Ralph child harness is built from a
  config clone and runs with a CLONED `budget_used` (`rust/.../harness/plan_execute.rs:379`
  `let budget_used = cx.scratch.run_budget.clone()`) and cloned `session_id`/
  `plan_session`, so the child cannot alias the parent's mutable scratch. TS
  builds child harnesses from a shallow-cloned config and passes `budgetUsed` by
  value (struct copy) per run (`typescript/.../harness/standard.ts:995,1115`).
  **Assertion:** after a combinator run, the parent's `budget_used` reflects
  only the explicitly folded child `usage`, not aliasing; the child's writes to
  its own `session_state` do not appear in the parent's transcript except via
  the documented threading (e.g. plan→execute read-files block).
- **A6 — `AbortSignal` is never retained beyond the call (TS contract).** The
  signal is passed DOWN through `run()` → `agent.turn` → `model.call`/
  `tool.execute` and not stored on long-lived config or the harness instance.
  (Rust has no `AbortSignal`; this is the TS idiomatic equivalent of
  future-drop cancellation, and the non-retention invariant keeps run N's abort
  from affecting run N+1.) **Assertion:** an aborted signal from run N does not
  cause run N+1 on the same instance to abort or short-circuit.

## 6. Compatibility constraints

- All new `HarnessConfig` fields, `HarnessStreamEvent` fields, `HarnessSignal`
  variants, `SandboxProvider` methods, and `StandardTools` constructors are
  **optional/additive** to preserve backward compatibility (mirrors Rust's
  `#[serde(default)]` / `Option<...>` discipline).
- Defaults for new config knobs match Rust (`planEmptyMaxRetries=0`,
  `maxRepairAttempts=1`, `toolCallNudge=false`, `enforceOutputSchemas=false`)
  so existing callers observe byte-identical behavior until they opt in.
- No changes to public type names or existing field spellings.
- ESM/zod/Vitest/Node 20 conventions (per `typescript/CONVENTIONS.md`) are
  preserved.

## 7. Validation strategy

- **Regression-first:** for every gap with an expressible behavior, add a
  failing-then-passing TS test before/with the implementation (TDD).
- **Fixture replay:** existing `*-fixture-replay.test.ts` suites are the
  cross-language ground truth — they must stay green. New gaps that change
  emitted bytes (steering, nudge, progress) are gated behind opt-in config so
  replay bytes are unchanged.
- **Representative E2E:** add end-to-end tests that exercise a real ReAct +
  PlanExecute run with the new seams (steering mid-run, set_plan plan phase,
  tool-call repair, live exec progress, narration refund), not only type
  assertions.
- **Gates before PR:** `pnpm -r test`, `pnpm -r lint`, `tsc --noEmit` (all
  packages), `pnpm -r build` where defined. Run examples' own test suites
  where present.

## 8. Risks & open compatibility notes

- **Multi-agent node attribution (G-A)** touches every combinator's child-run
  construction; incorrect wrapper forwarding could double-stamp or drop
  attribution. Mitigated by a dedicated multi-agent stamping test.
- **Live progress streaming (G-B)** requires every sandbox wrapper to forward
  `executeCommandStreaming` or streaming silently dies (Rust calls this out).
  Mitigated by a wrapper-forwarding test.
- **Plan-phase control-flow changes (G-H, G-G)** are the highest-risk area;
  mitigated by keeping defaults at the historical single-shot behavior and
  gating new behavior behind opt-in config + new signal handling that only
  fires when `set_plan` is offered.
- **No Rust edits:** if any gap is found to require a Rust-side change to
  achieve parity, work stops and the conflict is reported.

## 9. Decisions (resolved)

**Captain scope decision (incorporated):**

- **IN SCOPE — implement to full parity:** the complete TypeScript
  **`spore-core` + `spore-eval`** surface, including the full in-core
  `VectorStore` / `EmbeddingModel` traits and their Ollama/OpenAI providers
  (§5 G-L, with mocked-HTTP validation) and the three missing TypeScript doc
  pages (§5 G-M).
- **OUT OF SCOPE — document as known gaps, do NOT create:** the separate
  `spore-mcp` MCP-client package (E8), the separate `spore-sqlvec`
  sqlite-vec `VectorStore` backend package (E9), and the feature-gated Petri
  isolation backend (S4). These remain explicitly documented as known gaps in
  the final matrix (§10) with the rationale that they are separate Rust
  extension crates / an opt-in backend, not part of the core harness runtime.
  No TypeScript `@spore/mcp`, `@spore/sqlvec`, or Petri implementation will be
  authored in this parity pass.

The earlier open questions (D1 scope of extension crates, D2 Petri, D3 docs)
are closed by this decision. Rust remains strictly read-only throughout.

## 10. Final evidence matrix

All gaps in §5 are implemented and green, including the PR #169 independent-review
remediations for PlanExecute continuation/stream/retry parity, workspace command
progress, repair-on-resume bookkeeping, typed public errors, embedding wire shape,
and abort-aware async waits. Final gates (§7): `pnpm -r test` (core 1559 passed |
7 skipped, tools 278, eval 46), `pnpm -r lint` clean, `pnpm -r typecheck` clean
(0 `tsc --noEmit` errors across core/tools/eval), and `pnpm -r build` confirmed
there are no selected package build scripts. TypeScript examples 05-12 were
installed with frozen locks and their available `test`/`lint` scripts passed.

| Gap | Status | Tests | Notes |
| --- | --- | --- | --- |
| G-A NodeAttr attribution | **DONE** | `harness-parity-features.test.ts` (2), `plan-phase.test.ts` | `WithNode` wrapper on `HarnessStreamEvent`; react leaf + 4 combinators stamp `node`; PlanExecute child streams preserve nested paths like `["plan_execute","react"]`. Tests strip `node` before compare (mirrors Rust `*got_norm.node_mut() = None`). |
| G-B ToolProgress streaming | **DONE** | `harness-parity-features.test.ts` (1), `exec.test.ts` | `tool_progress` event variant; `ToolProgressSink` on `ToolContext`; `RealToolRegistry` threads `progressSink`; `ReadOnlySandbox` and `WorkspaceScopedSandbox` forward `executeCommandStreaming` tail snapshots. |
| G-C revised/steered fields | **DONE** | `harness-parity-features.test.ts` | `tool_result.revised?` + `user_message.steered?` on `HarnessStreamEvent`; `continue_with_modification` emits revised only when the stream-visible result changes; `send_message` emits `steered:false`. |
| G-D SteeringSource | **DONE** | `harness-parity-features.test.ts` | `steeringSource?` config + `HarnessBuilder.steeringSource()`; react loop drains steering at turn boundary, emits `user_message{steered:true}`. |
| G-E ToolCallRepair | **DONE** | `harness-parity-features.test.ts` | `StandardToolCallRepair` + `coerceToolArgs`; `dispatchWithRepair` wired into react dispatch plus resume/consult pending-call drains; repaired effective calls feed observed writes, spans, history, and enriched parameter errors. |
| G-F ToolCallNudge | **DONE** | `harness-parity-features.test.ts` | `toolCallNudge?` config; zero-tools-called nudge, bounded `MAX_TOOL_CALL_NUDGES=2`. |
| G-G planEmptyMaxRetries | **DONE** | `plan-phase.test.ts` (3) | `planEmptyMaxRetries?` config; plan-phase re-ask loop with corrective nudge, tools suppressed, forced `{tasks,rationale}` output schema, and folded discarded usage. |
| G-H SetPlanTool | **DONE** | `set-plan.test.ts` (7) | `SetPlanTool` writes durable task_list store; `plan_recorded` signal variant; catalogue + `codingSet` membership. |
| G-I CreateDirTool | **DONE** | `fs.test.ts` | `CreateDirTool` + `CreateDirParamsSchema`; `createDir`/`moveFile`/`deleteFile` in `StandardTools` + `codingSet`. |
| G-J Narration refund | **DONE** | `harness-parity-features.test.ts` | `narrationRefund` tracking + `budgetExceeded` `turnCapOverride`; send_message-only turns refund at window gate. |
| G-K Image helpers | **DONE** | `image.test.ts` (6) | `mediaTypeForExtension`, `isImagePath`, `imageContent`, cancellable `loadImageContent`, typed unsupported-extension errors. |
| G-L VectorStore/EmbeddingModel | **DONE** | `embedding.test.ts` + `vector-store.test.ts` (29) | `VectorStore`/`EmbeddingModel` traits + Ollama/OpenAI providers with mocked-HTTP retry, Rust `model_id` wire shape, structural metadata equality, typed vector errors, Ollama timeouts, and abort-aware retry waits. |
| G-M Docs | **DONE** | — | `building-a-tool.md`, `conversation.md`, `custom-harness.md` authored; `quickstart.md` cross-linked. |
| G-N tsc cleanup | **DONE** | — | 0 `tsc --noEmit` errors; `typecheck` script added to all 3 packages. |

### Known out-of-scope gaps (captain decision, §9)

| Gap | Rationale |
| --- | --- |
| `spore-mcp` (E8) | Separate Rust extension crate; not part of core harness runtime. |
| `spore-sqlvec` (E9) | Separate sqlite-vec `VectorStore` backend package; in-core traits (G-L) cover the parity seam. |
| Petri isolation backend (S4) | Feature-gated opt-in Rust backend; `WorkspaceScopedSandbox` is the TS default. |

---

done: all gaps implemented; PR #169 review findings fixed; final gates green