# Rust → TypeScript Harness Parity Plan

> **Status: PLAN-FIRST CHECKPOINT — no implementation started.**
> Authoritative Rust revision recorded below. This document inventories the
> Rust harness and the current TypeScript implementation end to end, records a
> parity matrix, and specifies file-level steps. It is ready for second-mate
> review. No implementation code has been written.

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

### Out-of-scope Rust crates (decision required)

- `rust/crates/spore-mcp` — MCP client that surfaces a remote MCP server's
  tools as native `Tool`s via the `rmcp` SDK. A **separate crate** depending on
  `spore-core`; not part of the core harness library. No TypeScript package
  equivalent exists today.
- `rust/crates/spore-sqlvec` — sqlite-vec `VectorStore` backend. A **separate
  crate** depending on `spore-core`'s `VectorStore`/`EmbeddingModel` traits.
  No TypeScript package equivalent exists today.

These are extension crates, not the harness itself. **Decision needed** (see
§9): whether "full observable behavioral parity" requires authoring TypeScript
`@spore/mcp` and `@spore/sqlvec` packages, or whether parity is scoped to the
`spore-core` + `spore-eval` surface (the harness runtime + eval). The plan
below assumes the **core + eval** scope and flags the trait-level prerequisites
(`VectorStore`, `EmbeddingModel`) that currently have no TS home.

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
| S4 | `sandbox_petri` (Petri isolation backend, feature-gated) | `rust/.../sandbox_petri.rs` | **No TS equivalent** (feature-gated Rust-only backend) | **GAP** (minor — opt-in backend; see §9) | minor |

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
| X17 | **`vector_store` (`VectorStore`, `VectorRecord`, `VectorHit`, `MetadataFilter`, `VectorStoreError`)** | `rust/.../vector_store.rs` (in spore-core) | **No TS module.** `grep VectorStore` TS → 0 hits. | **GAP** | major (trait prerequisite for sqlvec; see §9) |
| X18 | **`embedding` (`cosine_similarity`, `EmbeddingInfo`, `EmbeddingModel`, `OllamaEmbeddingModel`, `OpenAIEmbeddingModel`)** | `rust/.../embedding.rs` (in spore-core) | **No TS module.** TS `memory/standard.ts:10` notes embedding similarity is spec'd but unimplemented. | **GAP** | major (trait prerequisite for sqlvec; see §9) |
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
| E6 | Docs: `building-a-tool.md`, `conversation.md`, `custom-harness.md` | `docs/rust/{building-a-tool,conversation,custom-harness,quickstart}.md` | `docs/typescript/` has only stub `quickstart.md` | **GAP** | minor |
| E7 | Workspace/build config | Cargo workspace + feature flags (`petri`) | pnpm workspace, no feature flags | OBSERVED PARITY (idiomatic) | — |
| E8 | `spore-mcp` crate | `rust/crates/spore-mcp` (511 lines) | **No TS equivalent** | **GAP** (extension crate; see §9) | major |
| E9 | `spore-sqlvec` crate | `rust/crates/spore-sqlvec` | **No TS equivalent** | **GAP** (extension crate; see §9) | major |

## 5. Concrete gaps to close (implementation work, after plan approval)

Grouped by area. Each item lists: TS file-level steps, regression/E2E validation,
compatibility constraints, risk.

### G-A. Stream-event node attribution (L4) — major
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

### G-B. `ToolProgress` stream event + live progress sink (L5, T2, S2) — major
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

### G-C. `tool_result.revised` + `user_message.steered` fields (L6, L7) — major
- **Steps:** add optional `revised?: boolean` to `tool_result` and optional
  `steered?: boolean` to `user_message` in `HarnessStreamEvent`; emit
  `revised:true` when an AfterTool middleware rewrites a result (wire in the
  react tool-dispatch path); emit `steered:true` for steering-injected messages
  (depends on G-D), `steered:false` (or omitted) for `send_message` narration.
- **Validation:** test that an AfterTool `continue_with_modification` emits a
  revised `tool_result`; test that a steering injection emits `steered:true`.
- **Compat:** optional fields → back-compatible. Risk: low.

### G-D. Steering source (L9) — major
- **Steps:** add `SteeringSource` type (`() => string[]`) + optional
  `steeringSource?` to `HarnessConfig` and `HarnessBuilder.steeringSource(..)`;
  in the react loop, at the top of each turn iteration, drain
  `config.steeringSource()`, append each via `contextManager.appendUserMessage`,
  emit `user_message{steered:true}`.
- **Validation:** new test: a run with a steering source that queues a message
  mid-run injects it before the next model call, emits the steered event, and
  the message persists in the transcript.
- **Compat:** optional config → back-compatible. Risk: low.

### G-E. Tool-call repair seam (L10, M8) — major
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

### G-F. Tool-call nudge (L11) — major
- **Steps:** add optional `toolCallNudge?` to `HarnessConfig`/builder; in the
  react loop track `toolsCalledThisWindow` and `toolCallNudgesUsed` (cap
  `MAX_TOOL_CALL_NUDGES=2`); when a turn ends having called zero tools and the
  flag is on, inject the behavioral nudge user message.
- **Validation:** new test: a model that answers without calling tools gets
  nudged up to the cap, then the loop proceeds; flag off → no nudge
  (byte-identical).
- **Compat:** optional config, default off → back-compatible. Risk: low.

### G-G. Plan-empty re-ask (L12) — major
- **Steps:** add optional `planEmptyMaxRetries?` to `HarnessConfig`/builder
  (default `0`); in `captureAndPersistPlan`/plan-phase path, on an empty
  captured plan, re-ask up to `planEmptyMaxRetries` with tools suppressed +
  corrective nudge, fold discarded usage, then halt `empty_plan` if still
  empty. Default `0` preserves single-shot behavior.
- **Validation:** new test: a planner that emits an empty plan twice then a
  real plan succeeds when `planEmptyMaxRetries=2`; default `0` halts
  `empty_plan` immediately (byte-identical to today).
- **Compat:** optional config, default `0` → back-compatible. Risk: medium.

### G-H. `set_plan` tool + `PlanRecorded` signal (L13, T3) — major
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

### G-I. `create_dir` tool + catalogue gaps (T4, T5, T6) — major
- **Steps:** add `CreateDirParamsSchema` to `params.ts` + `CreateDirTool` to
  `fs.ts` (mirror `rust/.../tools/fs.rs:276`); add `createDir()`, `moveFile()`,
  `deleteFile()` constructors to `StandardTools`; add `create_dir`, `move_file`,
  `delete_file`, `set_plan` to `codingSet` (and thus `fullSet`).
- **Validation:** new tests for `create_dir` (mkdir -p, idempotent, sandbox
  validation); catalogue test asserting `codingSet` membership matches Rust.
- **Compat:** additive tools → back-compatible. Risk: low.

### G-J. Narration refund (L14) — major
- **Steps:** in the react loop, track `narrationRefund`; when a turn's only
  tool call is `send_message` (success), refund one turn against the
  productive-turn cap (capped at `effectiveTurnCap` so a pure-narration loop
  still terminates).
- **Validation:** new test: a run whose turns are all `send_message` narration
  does not consume the productive budget up to ~2× cap, then terminates.
- **Compat:** behavior change only when `send_message` is in the toolset;
  preserves turn-count monotonicity. Risk: medium (budget accounting).

### G-K. Image helpers (M6) — minor
- **Steps:** add `typescript/.../model/image.ts` with `mediaTypeForExtension`,
  `isImagePath`, `imageContent`, `loadImageContent` (mirror
  `rust/.../image.rs`); export from `model/index.ts`.
- **Validation:** unit tests for extension → media-type mapping and
  `loadImageContent` round-trip.
- **Compat:** additive. Risk: low.

### G-L. VectorStore + EmbeddingModel traits (X17, X18) — major (scope decision)
- **Steps (if in scope):** add `typescript/.../src/vector-store/` +
  `typescript/.../src/embedding/` modules mirroring the Rust traits
  (`VectorStore`, `VectorRecord`, `VectorHit`, `MetadataFilter`,
  `VectorStoreError`; `EmbeddingModel`, `EmbeddingInfo`, `cosineSimilarity`,
  `OllamaEmbeddingModel`, `OpenAIEmbeddingModel`); export from `index.ts`.
- **Validation:** unit tests for `cosineSimilarity` + trait shape; an in-memory
  `VectorStore` stub.
- **Compat:** additive. Risk: low. **Scope: see §9.**

### G-M. Docs (E6) — minor
- **Steps:** author `docs/typescript/building-a-tool.md`,
  `conversation.md`, `custom-harness.md` mirroring the Rust pages' structure
  with TS code.
- **Validation:** doc build/links. Risk: low.

### G-N. Pre-existing tsc errors (baseline) — minor
- **Steps:** fix the 12 `noUncheckedIndexedAccess` errors in
  `tests/tasklist.test.ts` and `tests/tool-error-loop.test.ts` (add `!` / null
  guards); add a `typecheck` script to each package `package.json`
  (`tsc --noEmit`) and wire it into CI-equivalent local validation.
- **Validation:** `tsc --noEmit` clean across packages.
- **Compat:** test-only. Risk: low.

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

## 9. Decisions required from second-mate / captain

1. **Scope of "full parity":** Is parity scoped to the **`spore-core` +
   `spore-eval`** harness runtime (this plan's default), or does it also
   require authoring TypeScript `@spore/mcp` and `@spore/sqlvec` packages
   (G-L, E8, E9)? The Rust `VectorStore`/`EmbeddingModel` traits live in
   `spore-core` itself — if in scope, G-L is required; if the extension crates
   are out of scope, G-L is deferred and the trait absence is documented as an
   explicit remaining gap.
2. **`sandbox_petri` (S4):** Rust-only feature-gated Petri isolation backend.
   Assume out of scope (opt-in backend, no TS equivalent) unless told
   otherwise.
3. **Docs scope (G-M):** confirm the three missing TS doc pages are in scope
   for this parity pass (vs. a separate docs task).

## 10. Intended final evidence

On completion, `docs/typescript/rust-parity-plan.md` will be updated to a final
matrix mapping every capability in §4 to: implementation commit, passing
test(s) name(s), and any explicitly remaining gap (with rationale). The final
gates in §7 will be green and the PR will reference this document.

---

needs-decision: plan ready for second-mate review at docs/typescript/rust-parity-plan.md; no implementation started