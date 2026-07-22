# Python ↔ Rust Behavioral Parity Plan — spore-core

> **Status: IMPLEMENTATION COMPLETE — reviewed; review findings fixed.**
> Authoritative Rust revision: `7058db115b4f05b6b7cde1a3f62ba345c202bcde`
> (`git rev-parse HEAD` at task start, branch `fm/parity-python`).
> Final evidence: 1732 passed, 8 skipped; `ruff check` + `ruff format --check`
> clean; 12 `examples/python/*` import + tests pass. See §12 for the
> capability→evidence matrix and the review/rereview-fix log.

## 1. Scope, authority, and read-only constraints

- The Rust implementation at the revision above is the **authoritative
  behavioral reference**. Rust source, tests, examples, configuration, and
  docs are treated **read-only** for this parity work; no edit is permitted
  under `rust/`, `docs/rust/`, `examples/rust/`, Rust-specific config, or
  Rust tests without explicit captain permission.
- If parity appears to require a Rust change, the exact conflict will be
  reported and work will stop. **No Rust conflict has been found during the
  inventory.** All gaps below are Python-side.
- Edits are confined to `python/`, `docs/python/`, `examples/python/`, and
  Python-specific tests/config. The single shared-file change owned by this
  task is the `AGENTS.md` contributor-guidance rule (see §10); no other
  shared top-level doc, CI, generated file, or shared fixture is touched.
- "Observable behavioral parity" means: identical results on the shared
  `fixtures/` corpus, identical public API surface and semantics, identical
  CLI/example behavior where present, and identical wire formats for
  serialized payloads that cross the language boundary. Naming may differ
  idiomatically (Rust `snake_case` fns vs Python `snake_case` fns; Rust
  `PascalCase` enums vs Python tagged-union dataclasses) provided the
  serialized `kind`/discriminator and field names match.

## 2. Methodology

1. Recorded `git rev-parse HEAD` (above).
2. Extracted Rust `pub use` re-exports from `rust/crates/spore-core/src/lib.rs`
   (391 symbols) and Python `__all__` from
   `python/packages/spore_core/src/spore_core/__init__.py` (700 symbols).
3. Diffed the two sets; for every Rust symbol absent from Python `__all__`,
   confirmed whether the symbol is (a) entirely missing, (b) present in a
   Python module but not re-exported, or (c) a naming/idiom difference.
4. Compared module-by-module: Rust `src/*.rs` + `src/harness/*.rs` against
   Python `src/spore_core/*.py`, focusing on public types, config fields,
   harness loop branches, stream-event fields, and fixture-replay behavior.
5. Ran the Python suite to establish a green baseline:
   `uv run pytest` → **1631 passed, 8 skipped**; `uv run ruff check .` →
   clean; `uv run ruff format --check .` → clean. Rust `cargo test
   --workspace --no-run` compiles (authoritative baseline confirmed).
6. Cross-checked the shared `fixtures/` corpus (29 fixture groups) against
   the Python tests that load them (43 test files reference `fixtures/`).

Legend for the matrix:
- ✅ **Observed parity** — same public symbol + same fixture/behavioral
  evidence on both sides.
- ⚠️ **Inferred similarity / partial** — surface looks equivalent but no
  shared fixture pins it, or a field/branch is missing.
- ❌ **Gap** — Rust behavior/symbol with no Python equivalent.

## 3. Baseline evidence

| Check | Command | Result |
|---|---|---|
| Rust HEAD | `git rev-parse HEAD` | `7058db1…` |
| Rust compiles | `cargo test --workspace --no-run` (in `rust/`) | OK |
| Python tests | `uv run pytest` (in `python/`) | 1631 passed, 8 skipped |
| Python lint | `uv run ruff check .` | All checks passed |
| Python format | `uv run ruff format --check .` | 158 files already formatted |

## 4. Module-by-module parity matrix

### 4.1 Core runtime (`harness`)

| Capability | Rust evidence | Python state | Verdict |
|---|---|---|---|
| ReAct loop | `harness/react.rs` (2252 LOC) | `harness.py` ReAct executor | ✅ observed (fixture `harness/react_loop.jsonl` replayed) |
| PlanExecute loop + DAG | `harness/plan_execute.rs`, `tests/plan_execute_*` | `test_harness_plan_execute*.py`, `test_harness_plan_execute_dag.py` | ✅ observed (shared DAG fixtures replayed) |
| Plan phase | `harness/…`, `tests/plan_phase_fixture_replay.rs` | `test_harness_plan_phase.py` | ✅ observed |
| Self-verifying | `harness/self_verifying.rs` | `test_harness_self_verifying*.py` | ✅ observed |
| Ralph | `harness/ralph.rs` | `test_harness_ralph.py` | ✅ observed |
| Hill-climbing | `harness/hill_climbing.rs` (`HillCandidate{current_sha,best_sha,…}`) | `harness.py` tracks `current_best` only; **no SHA tracking / VCS-aware revert** | ❌ gap (§5.3) |
| Cordyceps composition | `tests/cordyceps_composition_fixture_replay.rs` | `test_harness_cordyceps_composition.py` | ✅ observed |
| Budget policy/enforcement | `harness/budget.rs`, `budget_policy_tests.rs`, `budget_enforcement_tests.rs` | `test_budget_policy.py`, `test_budget_enforcement.py`, `test_harness_budget_escalation.py` | ✅ observed |
| Tool-error-loop breaker (#137) | `tests/tool_error_loop_fixture_replay.rs` | `test_harness_tool_error_loop*.py` | ✅ observed |
| Output-schema enforcement (#139) | `tests/output_schema_fixture_replay.rs` | `test_harness_output_schema*.py` | ✅ observed |
| Escalation protocol (#80) | `tests/escalation_*_fixture_replay.rs` | `test_harness_escalation.py` | ✅ observed |
| Mid-run steering (`steering_source`) | `harness/config.rs:11` `SteeringSource`, `harness/react.rs:343` injects queued user msgs + emits `UserMessage{steered:true}` | **no `steering_source` config, no injection, no `steered` field** | ❌ gap (§5.4) |
| Tool-call repair on dispatch | `harness/config.rs:130` `tool_call_repair: Option<Arc<dyn ToolCallRepair>>`, `harness/react.rs:27` applies repair | **no `tool_call_repair` config, no repair pass** | ❌ gap (§5.5) |
| `CommandProgressSink` (exec progress) | `harness/providers.rs:111` | absent | ❌ gap (§5.6) |
| Stream `NodeAttr` attribution | `harness/types.rs:426` `NodeAttr`, stamped on every `StreamEvent` variant via `wrap_node_sink` | **no `NodeAttr`, no `node` field on any harness stream event** | ❌ gap (§5.7) |
| `ToolProgress` stream event | `harness/types.rs` `StreamEvent::ToolProgress{call_id,name,preview}` (live tool output) | **no `StreamToolProgress` event** | ❌ gap (§5.8) |
| `TurnEnd.usage` per-turn tokens | `harness/types.rs` `TurnEnd{turn,usage,node}` | `StreamTurnEnd` has only `turn` | ❌ gap (§5.9) |
| `ToolResult.revised` flag | `harness/types.rs` `ToolResult{…,revised,node}` (AfterTool rewrite signal) | `StreamToolResult` has no `revised` | ❌ gap (§5.9) |
| `UserMessage.steered` flag | `harness/types.rs` `UserMessage{content,steered,node}` | `StreamUserMessage` has only `content` | ❌ gap (§5.9) |
| VcsProvider / GitVcsProvider | `harness/types.rs` | `harness.py` `GitVcsProvider`, `FixtureVcsProvider` | ✅ observed |
| SessionState / PausedState serialization | `harness/types.rs` | `test_harness_continue_checkpoint.py`, `test_harness_session_threading.py` | ✅ observed |
| Consult/subagent | `harness/…`, `tests/` | `test_harness_consult.py`, `test_harness_paused_toolset.py` | ✅ observed |

### 4.2 Model layer (`model`)

| Capability | Rust | Python | Verdict |
|---|---|---|---|
| `ModelInterface` (call, count_tokens, call_streaming) | `model.rs` | `model.py` | ✅ observed |
| Raw `StreamEvent` variants | `model.rs:224` (MessageStart, ContentBlockDelta, ThinkingDelta, ToolUseStart, ToolUseDelta, ContentBlockStop, MessageStop) | `model.py` identical 7 variants + fields | ✅ observed |
| Recording/Replay interfaces | `model.rs` | `test_model.py`, `test_anthropic.py` replay | ✅ observed |
| `enforce_budget` / `enforce_context_limit` / `request_hash` | `model.rs` | `model.py` + `test_model.py` | ✅ observed (shared `model_hashing` fixtures) |
| Providers: Anthropic, OpenAI, Ollama | `anthropic.rs`, `openai.rs`, `ollama.rs` | `anthropic.py`, `openai.py`, `ollama.py` + fixture replays | ✅ observed |

### 4.3 Agent (`agent`)

| Capability | Rust | Python | Verdict |
|---|---|---|---|
| `Agent`/`ModelAgent`/`TurnResult`/classification | `agent.rs`, `tests/agent_fixture_replay.rs` | `agent.py`, `test_agent.py` (fixture `agent/turn_classification.jsonl`) | ✅ observed |
| Streaming turn | `agent.rs` | `turn_streaming` in `agent.py` | ✅ observed |

### 4.4 Tool registry (`tool_registry`)

| Capability | Rust | Python | Verdict |
|---|---|---|---|
| `Tool` trait, `ToolRegistry`, `StandardToolRegistry`, `RealToolRegistry` | `tool_registry.rs` | `tool_registry.py` | ✅ observed |
| `ToolSet`, `ToolAnnotations`, `ToolContext`, `TaskPhase` | `tool_registry.rs` | re-exported | ✅ observed |
| `ToolProgressSink` / `ToolProgressUpdate` | `tool_registry.rs:149` | **absent** | ❌ gap (§5.10) |
| `ToolNamespace` | `tool_registry.rs:952` | **absent** | ❌ gap (§5.11) |
| `ToolSearchConfig` / `ToolSearchRegistry` (deferred-tool pattern) | `tool_registry.rs:992,1091` | **absent** | ❌ gap (§5.11) |
| `DispatchError`, `RegistrationError` | `tool_registry.rs` | re-exported | ✅ observed |

### 4.5 Tool-call repair (`tool_call_repair`) — **module absent**

| Capability | Rust | Python | Verdict |
|---|---|---|---|
| `coerce_tool_args(input, schema)` | `tool_call_repair.rs:45` | absent | ❌ gap (§5.5) |
| `detect_prose_response` (returns `ToolCallFailure`) | `tool_call_repair.rs:252` | Python has `prompt_tool_call.detect_prose_response` returning `str|None` (different contract) | ⚠️ partial (§5.5) |
| `ToolCallRepair` trait / `StandardToolCallRepair` | `tool_call_repair.rs:332,342` | absent | ❌ gap (§5.5) |
| `ToolCallFailure` enum | `tool_call_repair.rs:227` | absent | ❌ gap (§5.5) |

### 4.6 Context / compaction (`context`, `compaction_adapter`)

| Capability | Rust | Python | Verdict |
|---|---|---|---|
| `ContextManager` / `StandardContextManager` | `context.rs` | `context.py` | ✅ observed (`compaction_window`, `compaction_loop` fixtures) |
| `CompactionConfig/Request/Result/Verifier` | `context.rs` | re-exported | ✅ observed |
| `HarnessContextManagerExt` (rich-state ext trait) | `compaction_adapter.rs` | **absent** (Python has `into_harness_adapter` + `seed_rich_state` + `StandardCompactionAdapter`) | ⚠️ partial (§5.12) |
| `KeyTermVerifier`, `SegmentStability`, breakpoints | `context.rs` | re-exported | ✅ observed |

### 4.7 Prompt assembly / chunk registry (`prompt_assembly`, `prompt_chunk_registry`)

| Capability | Rust | Python | Verdict |
|---|---|---|---|
| `PromptChunkRegistry` / `StandardPromptChunkRegistry` / `standard_chunks` | `prompt_chunk_registry.rs` | re-exported; `prompt_chunk_registry` fixtures | ✅ observed |
| Assembly engine: `ChunkProvider`, `AssemblyContext`, `breakpoint_ids`, `chunks_to_segments` | `prompt_assembly.rs` | re-exported; `prompt_assembly` fixtures | ✅ observed |
| `CustomCondition` | `prompt_assembly.rs` | **absent** | ❌ gap (§5.13) |
| `ChunkCondition`, `ToolAffinity`, `StorageScope` | `prompt_assembly.rs` | re-exported | ✅ observed |

### 4.8 Observability (`observability`, `observability_outbox`)

| Capability | Rust | Python | Verdict |
|---|---|---|---|
| `ObservabilityProvider`, `InMemoryObservabilityProvider`, span types | `observability.rs` | `observability.py` | ✅ observed |
| `SpanLevel`, `PatchSpan`, `PatchType`, `ContentCaptureConfig`, `TRUNCATION_MARKER`, `truncate_field` | `observability.rs` | **present in `observability.py` but NOT re-exported from `__init__`** | ❌ re-export gap (§5.2) |
| `GenAiMessage`, `GenAiRole`, `ObservabilityError`, `SensorSpan` (exact name) | `observability.rs` | **present but NOT re-exported** | ❌ re-export gap (§5.2) |
| `OutboxConfig`, `OutboxObservabilityProvider`, `TraceLine` | `observability_outbox.rs` | **present in `observability_outbox.py` but NOT re-exported** | ❌ re-export gap (§5.2) |
| `OutboxObservabilityProvider` JSONL+OTLP behavior | `observability_outbox.rs` | `observability_outbox.py` (idiomatic port) | ⚠️ inferred (no shared fixture pins outbox bytes) |

### 4.9 Termination (`termination`)

| Capability | Rust | Python | Verdict |
|---|---|---|---|
| `TerminationPolicy`, `StandardTerminationPolicy`, `TerminationDecision` | `termination.rs` | re-exported | ✅ observed (`termination_policy` fixtures) |
| `BudgetValue` family, `check_budget_default` | `termination.rs` | re-exported | ✅ observed |
| `CompletionCheck`, `NullCompletionCheck`, `FixedCompletionCheck`, `SessionStateSnapshot` | `termination.rs` | re-exported | ✅ observed |
| `AlwaysComplete`, `FeatureListCheck`, `QuestionAnsweredCheck`, `SqlResultCheck`, `TestSuiteCheck` | `termination.rs` | **present in `termination.py` but NOT re-exported** | ❌ re-export gap (§5.2) |

### 4.10 Storage (`storage`)

| Capability | Rust | Python | Verdict |
|---|---|---|---|
| `StorageProvider`, `InMemory/FileSystem/Composite/NoOp`, `SessionStore`, `RunStore`, `MemoryStore` | `storage.rs` | re-exported | ✅ observed (`storage` fixtures) |
| `ScopedMemoryRouter` | `storage.rs` | **present in `storage.py` but NOT re-exported** | ❌ re-export gap (§5.2) |
| `ProjectId`, `WorkspaceId`, `ActiveRun`, `start_or_resume_active_run`, `complete_active_run` | `storage.rs` | re-exported | ✅ observed |
| `parse_otlp_endpoints` | `storage.rs` | re-exported | ✅ observed |

### 4.11 Memory / Guide / Sensor / Skills / Hooks / Middleware / Metric / Verifier / ExecutionRegistry / CacheProvider / Sandbox / Tasklist / Plan

All show ✅ observed parity via re-exports + dedicated test files + shared
fixtures (`memory`, `guide_registry` via guide tests, `sensor_chain`,
`hooks`, `middleware`, `metric_evaluator`, `verifier`, `budget_policy`,
`cache_provider`, `sandbox_violations`, `tasklist`, `plan_to_tasklist`).
Detailed per-symbol confirmation recorded in §6.

### 4.12 Entirely absent modules

| Rust module | LOC | Python | Verdict |
|---|---|---|---|
| `embedding.rs` (`cosine_similarity`, `EmbeddingInfo`, `EmbeddingModel`, `OllamaEmbeddingModel`, `OpenAIEmbeddingModel`) | 550 | **no `embedding.py`** | ❌ gap (§5.14) |
| `vector_store.rs` (`VectorStore`, `VectorRecord`, `VectorHit`, `VectorStoreError`, `MetadataFilter`) | 345 | **no `vector_store.py`** | ❌ gap (§5.14) |
| `image.rs` (image content helpers) | 93 | **no `image.py`** | ❌ gap (§5.15) |
| `tool_call_repair.rs` | 582 | **no `tool_call_repair.py`** | ❌ gap (§5.5) |
| `text_util.rs` / `sync_util.rs` / `macros.rs` / `provider_http.rs` | — | N/A (internal/Rust-specific) | ✅ not applicable |
| `sandbox_petri.rs` (feature-gated) | 1564 | N/A (Python uses subprocess sandbox) | ✅ not applicable |

### 4.13 Tools (`spore_tools` vs Rust `tools/`)

| Capability | Rust | Python `spore_tools` | Verdict |
|---|---|---|---|
| `fs`, `edit`, `exec`, `search`, `git`, `web`, `http`, `memory`, `tasklist`, `todo`, `control`, `message`, `subagent`, `catalogue`, `params`, `error` | `tools/*.rs` | `tools/*.py` | ✅ observed (`tools` fixtures) |
| `set_plan` tool (atomic plan authoring, #126) | `tools/set_plan.rs` | **absent** | ❌ gap (§5.16) |
| `define_tool` ergonomic helper | Rust `tool!` macro (`macros.rs`) | `tools/define.py` | ✅ observed (idiomatic port) |
| `StandardTools` aggregate / `AbortTool` / `AskUserQuestionTool` / `EnterPlanModeTool` / `ExitPlanModeTool` / `WebSearchTool` / `WebFetchTool` / `GrepTool` / `SendMessageTool` / `EditFileTool` / `TodoWriteTool` / `TaskListTool` | `tools/mod.rs` + `tools/*.rs` | built via `catalogue.py` + individual modules; not exported as named `StandardTools` aggregate from `spore_core` | ⚠️ partial (§5.16) |

### 4.14 CLI / examples

| Capability | Rust | Python | Verdict |
|---|---|---|---|
| Examples 01–12 | `examples/rust/0*` | `examples/python/0*` (each `import main` + per-example tests) | ✅ observed (CI examples gate) |
| `spore-eval` CLI (`run` / `promote`) | `rust/crates/spore-eval/src/bin/spore-eval.rs` (`run <suite> [--candidate … --n N --json]`, `promote <suite> <task_id>`) | `python/.../spore_eval/eval_cli.py` (identical subcommands + flags + exit codes) | ✅ observed — pinned by executable validation in §5.17 against the shared `fixtures/task_suites/core_suite.json` |
| `EvalHarness` lib API | `rust/crates/spore-eval/src/lib.rs` (`EvalHarness`, `load_suite_path`, `promote_challenge_task`, `suite_to_json`, metric/report/stats/task/verifier/worktree exports) | `python/.../spore_eval/eval/__init__.py` (mirrors every pub use) | ✅ observed — pinned by `test_eval_harness.py` + §5.17 |

### 4.15 Cancellation / timeout / retry / cleanup / async-concurrency

Evidence gathered read-only from Rust + Python. All items ✅ observed parity
unless noted.

| Scenario | Rust evidence | Python evidence | Verdict |
|---|---|---|---|
| **Cancellation — agent-initiated abort** | `HarnessSignal::Abort{reason}` → `RunResult::Escalate` (NOT `Failure`); `harness/tests.rs:2961` `abort_signal` + `escalate_abort_terminates_with_escalate_not_failure` (R8) | `HarnessSignalAbort` → `RunResultEscalate`; `test_harness_escalation.py` | ✅ observed |
| **Cancellation — in-flight tool work** | No internal cancellation token; steering injects only at turn boundaries AFTER tool execution (`react.rs:343`) so in-flight work is never cancelled | Python mirrors: `append_user_message` only at turn boundaries; no mid-turn cancel | ✅ observed |
| **Timeout — wall-time budget** | `BudgetValue::Duration`; `harness/mod.rs:1651` `started_at.elapsed() >= max`; `BudgetLimitType::WallTime` | `BudgetLimits.wall_time`; `budget_exceeded` uses `time.monotonic()` + `started_at`; `BudgetLimitTypeT = "wall_time"` | ✅ observed |
| **Timeout — provider HTTP** | `AnthropicModelInterface::DEFAULT_TIMEOUT = 120s`; per-provider timeout on the HTTP client | `anthropic.py` `DEFAULT_TIMEOUT`; `openai.py`/`ollama.py` timeouts | ✅ observed |
| **Retry — provider transient** | `provider_http::is_retryable_status` = `{408,425,429,500,502,503,504}`; `backoff_delay` = 0.5,1,2,4,…30s cap; Anthropic adds `529`; `ModelError::retryable()` covers Transport/StreamInterrupted/Timeout/RateLimited | `anthropic.py` `_RETRYABLE_STATUSES = {408,425,429,500,502,503,504,529}`; `_backoff_delay` = 0.5,1,2,4,…30s cap; `ModelError.retryable()` covers Transport/StreamInterrupted/TimeoutError/RateLimited | ✅ observed (byte-identical status set + backoff curve) |
| **Retry — output-schema enforcement** | `output_schema_max_retries` extra turns (#139); `StreamEvent::OutputSchemaRetry/Violation` | `output_schema_max_retries`; `StreamOutputSchemaRetry/Violation`; `test_harness_output_schema*.py` | ✅ observed |
| **Retry — tool-error-loop breaker** | `error_loop_threshold` N consecutive identical-arg errors → 1 corrective msg; `2*N` → halt (#137) | `test_harness_tool_error_loop*.py` | ✅ observed |
| **Retry — tool-call repair passes** | `tool_call_repair_max_passes` (default 1) bounds re-dispatch | **absent** — see §5.5 | ❌ gap (covered by §5.5) |
| **Cleanup — observability finalize** | `StrategyExecutor::finalize` → `finalize_observability(session_id, SessionOutcome*)` on terminal results; non-terminal pauses never finalized | `StrategyExecutor.finalize` → `_finalize_observability`; `test_observability.py` | ✅ observed |
| **Cleanup — budget scope pop** | `cx.pop_budget()` on every early return from a combinator | `BudgetStack` pop on early return | ✅ observed |
| **Async — structured concurrency** | `futures_util::future::join_all` for the concurrent (read-only) tool batch in `dispatch_many`; sequential `for` for the rest | `anyio.create_task_group()` for the concurrent read-only batch; sequential `for` for the rest (`tool_registry.py:500`) | ✅ observed (same partition rule: `read_only && !destructive && !open_world`) |
| **Async — single-call dispatch order** | `for call in calls { dispatch_with_repair(...) }` — sequential, order-preserving | `for call in ...: await dispatch(call)` — sequential, order-preserving | ✅ observed |
| **Async — locks** | `anyio.Lock` equivalents via `RwLock`/`Mutex` on registries | `anyio.Lock` in `RecordingModelInterface`/`ReplayModelInterface` (`model.py:586,711`) | ✅ observed |
| **Async — cancel-scope seam** | N/A (Rust uses `BoxFut` + budget gates; no public cancel token) | CONVENTIONS.md mandates `anyio.CancelScope` for public async APIs that accept one; harness run is driven by budget gates, matching Rust | ✅ observed (no public cancel-token divergence) |

**New regression tests to add** (executable validation for this sub-matrix):
- `test_harness_cancellation.py`: agent `HarnessSignalAbort` → `RunResultEscalate` (not `Failure`); steering injection does not cancel in-flight tool work.
- `test_harness_timeout.py`: `BudgetLimits.wall_time` trips `BudgetExhausted` mid-run via a scripted slow model.
- `test_provider_retry.py`: 429/503/529 + timeout → typed `RateLimited`/`TimeoutError` with the exact backoff curve (0.5,1,2,4,…30) and `retryable()` true; deterministic 400 → non-retryable.
- `test_tool_dispatch_concurrency.py`: a mixed batch (read-only + destructive) partitions correctly; read-only calls run concurrently and order is preserved.

## 5. Gaps — detailed, with file-level steps

Gaps are ordered by behavioral impact (wire-format / loop-branch first,
re-exports last). Each lists Rust evidence, current Python behavior, exact
file-level steps, regression/E2E validation, compatibility, risk, and
intended final evidence.

### 5.1 Stream-event wire format (`harness.py` + `harness/types.rs`)

- **Rust**: every `StreamEvent` variant carries `node: Option<NodeAttr>`
  (`#[serde(default, skip_serializing_if="Option::is_none")]`, so
  un-attributed events serialize byte-identically to pre-`NodeAttr` form).
  `TurnEnd` carries `usage: TokenUsage` (`#[serde(default)]`); `ToolResult`
  carries `revised: bool` (`#[serde(default)]`); `UserMessage` carries
  `steered: bool` (`#[serde(default)]`); `ToolProgress` is a live-output
  event.
- **Python**: `StreamTurnStart/End`, `StreamToolCall/Result`,
  `StreamFinalResponse`, `StreamBudgetWarning`, `StreamUserMessage`, and all
  delta/frame variants lack `node`; `StreamTurnEnd` lacks `usage`;
  `StreamToolResult` lacks `revised`; `StreamUserMessage` lacks `steered`;
  no `StreamToolProgress`.
- **Steps**: in `harness.py`, add `node: NodeAttr | None = None` (default
  `None`, excluded from `model_dump` when `None` to match
  `skip_serializing_if`) to every harness stream-event dataclass; add
  `usage: TokenUsage = Field(default_factory=TokenUsage)` to
  `StreamTurnEnd`; add `revised: bool = False` to `StreamToolResult`; add
  `steered: bool = False` to `StreamUserMessage`; add `StreamToolProgress`
  dataclass. Emit `usage` on `TurnEnd` from the model `MessageStop.usage`.
  Wire `revised` when an `AfterTool` middleware rewrites a result. Wire
  `steered` for `steering_source` injections (§5.4). Wire `ToolProgress`
  from a `CommandProgressSink`/tool progress sink (§5.6/§5.10).
- **Regression/E2E**: new `test_harness_stream_events.py` asserting (a)
  default-`None` `node` serializes identically to current bytes
  (back-compat), (b) `TurnEnd.usage` populated from a scripted model stream,
  (c) `ToolResult.revised=True` after a `ContinueWithModification`
  middleware rewrite, (d) `UserMessage.steered=True` from a steering
  injection, (e) `ToolProgress` emitted by a progress-sink tool.
- **Compatibility**: all new fields default to absent/false/empty, matching
  Rust's `#[serde(default)]` — backward compatible for existing consumers.
- **Risk**: low; additive fields. Must keep `node=None` out of serialized
  output to preserve byte-identical single-leaf streams.
- **Final evidence**: passing stream-event tests + a fixture-replay that
  round-trips a multi-agent attributed stream.

### 5.2 Missing `__init__` re-exports (pure export gaps; behavior already present)

- **Rust**: re-exports `SpanLevel`, `PatchSpan`, `PatchType`,
  `ContentCaptureConfig`, `TRUNCATION_MARKER`, `truncate_field`,
  `GenAiMessage`, `GenAiRole`, `ObservabilityError`, `SensorSpan`;
  `OutboxConfig`, `OutboxObservabilityProvider`, `TraceLine`; `AlwaysComplete`,
  `FeatureListCheck`, `QuestionAnsweredCheck`, `SqlResultCheck`,
  `TestSuiteCheck`; `ScopedMemoryRouter`; step-ledger
  `StepLedgerEntry`/`push_step_ledger`/`render_step_ledger`/
  `STEP_LEDGER_ELISION_MARKER`/`STEP_LEDGER_MAX_ENTRIES`.
- **Python**: all present in their respective modules but absent from
  `__init__.py` `__all__`.
- **Steps**: add the import + `__all__` entries in `__init__.py` only.
- **Regression/E2E**: `test_public_api.py` asserting each symbol is
  importable from `spore_core` and has the expected runtime type.
- **Compatibility**: purely additive exports.
- **Risk**: none.
- **Final evidence**: `test_public_api.py` green + `ruff` clean.

### 5.3 Hill-climbing VCS-aware revert (`harness.py` ↔ `harness/hill_climbing.rs`)

- **Rust**: `HillCandidate{current_sha, best_sha, current_best, best_score}`;
  revert targets `current_sha` (not `HEAD`) when a VCS provider is wired.
- **Python**: tracks `current_best` only; no SHA tracking; revert falls
  back to `HEAD`.
- **Steps**: add a `HillCandidate` dataclass in `harness.py`; thread
  `current_sha`/`best_sha` through the climb; on no-improvement, revert to
  `current_sha` via the configured `VcsProvider` when available.
- **Regression/E2E**: extend `test_harness_hill_climbing.py` with a
  `FixtureVcsProvider` scenario asserting revert targets `current_sha`, not
  `HEAD`, and that `best_sha` is applied at loop exit.
- **Compatibility**: additive state; serialized `PausedState` gains
  optional `current_sha`/`best_sha` (default `None`).
- **Risk**: medium — touches the climb state machine; needs careful
  back-compat for in-flight `PausedState`.
- **Final evidence**: passing hill-climbing VCS test.

### 5.4 Mid-run steering (`harness.py` ↔ `harness/config.rs` + `react.rs`)

- **Rust**: `HarnessConfig::steering_source: Option<SteeringSource>` (callback
  `Fn() -> Vec<String>`); the ReAct loop drains queued user messages at each
  turn boundary, appends them via `ContextManager::append_user_message`, and
  emits `StreamEvent::UserMessage{steered:true}`.
- **Python**: no `steering_source` config, no injection, no `steered`.
- **Steps**: add `steering_source: Callable[[], list[str]] | None` to
  `HarnessConfig`/`HarnessBuilder`; in the ReAct loop, before each model
  call, drain `steering_source()`, append each via
  `context_manager.append_user_message`, emit `StreamUserMessage(steered=True)`.
- **Regression/E2E**: new `test_harness_steering.py` with a queue-returning
  callback asserting messages are injected between turns and emitted with
  `steered=True`, and that in-flight tool work is not cancelled.
- **Compatibility**: additive optional config.
- **Risk**: low-medium; must inject after tool execution, not mid-turn.
- **Final evidence**: passing steering test.

### 5.5 Tool-call repair (`tool_call_repair.py` new + `harness.py`)

- **Rust**: `tool_call_repair.rs` (`coerce_tool_args`, `ToolCallFailure`,
  `detect_prose_response`→`ToolCallFailure`, `ToolCallRepair` trait,
  `StandardToolCallRepair`); `HarnessConfig::tool_call_repair` +
  `tool_call_repair_max_passes`; `react.rs` applies repair on dispatch.
- **Python**: `prompt_tool_call.detect_prose_response` exists with a
  different contract (`str|None`); no `coerce_tool_args`, no repair trait,
  no harness repair pass.
- **Steps**: new `python/.../spore_core/tool_call_repair.py` porting
  `coerce_tool_args` (schema-guided coercion), `ToolCallFailure`,
  `StandardToolCallRepair`, and a `detect_prose_response` returning
  `ToolCallFailure | None`; add `tool_call_repair` +
  `tool_call_repair_max_passes` to `HarnessConfig`/builder; apply repair in
  the ReAct dispatch path; re-export from `__init__`.
- **Regression/E2E**: new `test_tool_call_repair.py` mirroring Rust
  `harness/tests.rs::tool_call_repair_fixes_bad_bool_arg` (coerce a string
  `"true"`→`true`, re-dispatch succeeds) + prose-detection cases.
- **Compatibility**: additive optional config.
- **Risk**: medium — repair pass interacts with the dispatch loop; must
  bound passes to avoid infinite re-dispatch.
- **Final evidence**: passing repair tests + harness integration test.

### 5.6 `CommandProgressSink` (`harness.py` ↔ `harness/providers.rs`)

- **Rust**: `CommandProgressSink = Arc<dyn Fn(String)>`; exec tool emits
  live tail previews → `StreamEvent::ToolProgress`.
- **Python**: absent.
- **Steps**: add `CommandProgressSink` (callable) type + config wiring;
  emit `StreamToolProgress` from the exec tool when a sink is configured.
- **Regression/E2E**: `test_harness_stream_events.py` ToolProgress case.
- **Compatibility**: additive.
- **Risk**: low.
- **Final evidence**: passing ToolProgress test.

### 5.7 `NodeAttr` stream attribution (`harness.py` ↔ `harness/types.rs`)

- **Rust**: `NodeAttr{kind, agent_id, depth, path}` stamped on every event
  via `wrap_node_sink` so a multi-agent UI can route frames to the emitting
  node.
- **Python**: absent.
- **Steps**: add `NodeAttr` dataclass; add `node` field to every stream
  event (§5.1); stamp events in the combinator wrappers (plan_execute,
  self_verifying, ralph, hill_climbing) and react leaf.
- **Regression/E2E**: `test_harness_stream_events.py` multi-agent case
  asserting `node.kind`/`node.path`/`node.depth` on a cordyceps run.
- **Compatibility**: `node` defaults `None` and is omitted when `None`
  (byte-identical single-leaf streams).
- **Risk**: medium — touches every combinator's stream wrapping.
- **Final evidence**: passing multi-agent attribution test.

### 5.8 `ToolProgress` event — covered by §5.1/§5.6.

### 5.9 `TurnEnd.usage`, `ToolResult.revised`, `UserMessage.steered` — covered by §5.1.

### 5.10 `ToolProgressSink` / `ToolProgressUpdate` (`tool_registry.py` ↔ `tool_registry.rs`)

- **Rust**: `ToolProgressUpdate` + `ToolProgressSink` trait for tools to
  emit live progress.
- **Python**: absent.
- **Steps**: add `ToolProgressUpdate` dataclass + `ToolProgressSink`
  protocol in `tool_registry.py`; thread an optional sink through
  `ToolContext`; re-export.
- **Regression/E2E**: `test_tool_registry.py` progress-sink case.
- **Compatibility**: additive optional.
- **Risk**: low.
- **Final evidence**: passing progress-sink test.

### 5.11 `ToolNamespace` / `ToolSearchConfig` / `ToolSearchRegistry` (`tool_registry.py` ↔ `tool_registry.rs`)

- **Rust**: deferred-tool pattern for local models — core tools always
  advertised, the rest behind namespace headers + a `tool_search` meta-tool,
  LRU-loaded on demand; dispatch is never gated by exposure.
- **Python**: absent.
- **Steps**: new `ToolNamespace`, `ToolSearchConfig`, `ToolSearchRegistry`
  in `tool_registry.py` wrapping an inner `ToolRegistry`; synthesize
  namespace header schemas + the search meta-tool; LRU-load on
  `tool_search`/namespace open; delegate dispatch to the inner registry.
- **Regression/E2E**: new `test_tool_search_registry.py` asserting (a) core
  tools advertised, (b) deferred tool callable even when not advertised,
  (c) `tool_search` surfaces a deferred tool, (d) description-prose budget
  honored.
- **Compatibility**: additive.
- **Risk**: medium — non-trivial LRU + schema synthesis; needs careful
  parity with Rust's namespace-collision drop rule.
- **Final evidence**: passing tool-search tests.

### 5.12 `HarnessContextManagerExt` (`compaction_adapter.py` ↔ `compaction_adapter.rs`)

- **Rust**: ext trait adding rich-state seeding helpers to any
  `ContextManager`.
- **Python**: has `seed_rich_state` + `into_harness_adapter` +
  `StandardCompactionAdapter` but no ext-trait equivalent.
- **Steps**: add an idiomatic Python equivalent (a small mixin/protocol or
  free functions operating on a `ContextManager`) exposing the same helpers;
  re-export as `HarnessContextManagerExt` for API-name parity.
- **Regression/E2E**: extend `test_compaction_adapter.py`.
- **Compatibility**: additive.
- **Risk**: low.
- **Final evidence**: passing adapter test.

### 5.13 `CustomCondition` (`prompt_assembly.py` ↔ `prompt_assembly.rs`)

- **Rust**: `CustomCondition` chunk condition.
- **Python**: absent.
- **Steps**: add `CustomCondition` (callable condition) to
  `prompt_assembly.py`; integrate into `ChunkCondition`; re-export.
- **Regression/E2E**: extend `test_prompt_assembly.py`.
- **Compatibility**: additive.
- **Risk**: low.
- **Final evidence**: passing assembly test.

### 5.14 `embedding` + `vector_store` modules (new files) — full provider impls now

- **Rust**: `embedding.rs` (`cosine_similarity`, `EmbeddingInfo`, `EmbeddingModel`,
  `OllamaEmbeddingModel`, `OpenAIEmbeddingModel`); `vector_store.rs`
  (`VectorStore`, `VectorRecord`, `VectorHit`, `VectorStoreError`,
  `MetadataFilter`).
- **Python**: no `embedding.py`, no `vector_store.py`.
- **Scope decision (C1, resolved by plan review)**: implement **full**
  `OllamaEmbeddingModel` and `OpenAIEmbeddingModel` provider behavior now —
  not a follow-up. The pure-math core (`cosine_similarity`, `EmbeddingInfo`,
  `EmbeddingModel` protocol, `VectorStore`/`VectorRecord`/`VectorHit`/
  `MetadataFilter` in-memory) AND the HTTP-backed provider impls are all in
  scope for this PR.
- **Steps**: new `embedding.py` porting `cosine_similarity` (identical
  float math), `EmbeddingInfo`, the `EmbeddingModel` protocol, and
  `OllamaEmbeddingModel` + `OpenAIEmbeddingModel` (HTTP clients matching
  Rust's request/response shapes, retry/backoff via the shared
  `provider_http`-equivalent status set, `count_tokens`-style helpers where
  Rust has them); new `vector_store.py` porting `VectorStore`, `VectorRecord`,
  `VectorHit`, `VectorStoreError`, `MetadataFilter`; re-export both modules'
  public symbols from `__init__`.
- **Regression/E2E**: `test_embedding.py` (cosine-similarity known vectors;
  Ollama + OpenAI embedding HTTP shape via `respx` mocked endpoints, asserting
  request URL/headers/body match Rust's provider contract and the response
  parses to `EmbeddingInfo`); `test_vector_store.py` (upsert/search/hit
  ranking by cosine similarity, metadata-filter equality).
- **Compatibility**: additive new modules + `__init__` re-exports.
- **Risk**: medium — provider HTTP parity must match Rust request/response
  bytes; covered by `respx`-pinned tests.
- **Final evidence**: passing embedding + vector-store tests; symbols
  importable from `spore_core`.

### 5.15 `image` module (new file) — applicability resolved

- **Rust**: `image.rs` is a **public module** (`pub mod image;` in `lib.rs`)
  exposing `media_type_for_extension(ext) -> Option&str`,
  `is_image_path(path) -> bool`, `image_content(media_type, bytes) ->
  Content::Image`, and `load_image_content(path) -> io::Result<Content>`.
  It is the single producer of `Content::Image` for both user-attached images
  and a `view_image` tool. It is **not** re-exported at the crate root
  (callers reach it as `spore_core::image::*`); the Python parity target is a
  public `spore_core.image` submodule with the same four callables.
- **Python**: `model.py` already has `ImageContent`; no standalone `image.py`
  helpers.
- **Steps**: new `python/.../spore_core/image.py` porting all four functions
  (`media_type_for_extension` case-insensitive over png/jpg/jpeg/gif/webp,
  `is_image_path`, `image_content` base64-encoding into `ImageContent`,
  `load_image_content` reading the file with the same `InvalidInput`-style
  error for unsupported extensions); expose as `spore_core.image` (module
  path parity) and re-export the four callables from `__init__`.
- **Regression/E2E**: `test_image.py` mirroring Rust's three inline tests
  (`media_type_maps_known_extensions_case_insensitively`,
  `is_image_path_checks_extension`, `image_content_base64_encodes_bytes` —
  `b"hi"` → `"aGk="`) plus a `load_image_content` tmp-file round-trip and an
  unsupported-extension error case.
- **Compatibility**: additive.
- **Risk**: low — pure helper module.
- **Final evidence**: passing `test_image.py`; `from spore_core.image import
  load_image_content` works.

### 5.16 `set_plan` tool + `StandardTools` aggregate (`spore_tools`)

- **Rust**: `tools/set_plan.rs` (atomic plan authoring →
  `TASK_LIST_EXTRAS_KEY`); `StandardTools` aggregate exporting the standard
  tool set.
- **Python**: no `set_plan.py`; no named `StandardTools` aggregate exported
  from `spore_core` (tools are assembled via `catalogue.py`).
- **Steps**: add `spore_tools/tools/set_plan.py` mirroring `set_plan.rs`
  (writes the same `TASK_LIST_EXTRAS_KEY` blob); add a `StandardTools`
  aggregate/constructor for API-name parity.
- **Regression/E2E**: `test_set_plan.py` asserting the persisted blob is
  readable by the existing plan-capture fallback; `test_standard_tools.py`
  asserting the aggregate matches the catalogue.
- **Compatibility**: additive.
- **Risk**: low.
- **Final evidence**: passing set_plan + standard-tools tests.

### 5.17 Eval/CLI executable validation (replaces inferred parity)

- **Rust**: `spore-eval` CLI (`run <suite.json> [--candidate ID … --n N --json]`,
  `promote <suite.json> <task_id>`) over `rust/crates/spore-eval/src/lib.rs`
  (`load_suite_path`, `promote_challenge_task`, `suite_to_json`). Shared
  fixture: `fixtures/task_suites/core_suite.json`.
- **Python**: `spore_eval/eval_cli.py` mirrors the CLI; `spore_eval.eval`
  mirrors the lib API. Currently ⚠️ inferred — no test pins the CLI output to
  the shared fixture.
- **Steps**: add `python/packages/spore_eval/tests/test_eval_cli_fixture.py`
  that (a) invokes `eval_cli._cmd_run([core_suite_path, "--json"])` and
  asserts the JSON summary's `suite_version` + `regression`/`challenge`/
  `canary` counts equal the parsed `load_suite_path` shape; (b) invokes
  `eval_cli._cmd_run([core_suite_path])` and asserts the human-readable line
  matches the Rust `loaded suite v{} — regression={}, challenge={}, canary={}
  (n={})` format; (c) copies the suite to a tmp path, invokes
  `eval_cli._cmd_promote([tmp, task_id])`, and asserts `suite_version`
  bumped + the task moved challenge→regression + the rewritten JSON
  round-trips via `load_suite_path`; (d) asserts unknown-subcommand and
  missing-arg paths return exit code 2 with the Rust-equivalent stderr.
- **Regression/E2E**: the test above is the executable validation; it pins
  CLI parity to the shared `core_suite.json` without requiring `cargo` in
  Python CI. A companion `test_eval_lib_fixture.py` asserts
  `load_suite_path`/`suite_to_json` round-trip the fixture byte-stably.
- **Compatibility**: test-only.
- **Risk**: low.
- **Final evidence**: passing `test_eval_cli_fixture.py` +
  `test_eval_lib_fixture.py`.

## 6. Per-symbol confirmation for §4.11 (observed-parity modules)

For each module marked ✅ in §4.11, the Python re-export set was confirmed
to cover the Rust public surface (naming differences noted):

- `memory.py`: `MemoryProvider`, `StandardMemoryProvider`, `MemoryItem`,
  `MemoryQuery`, `MemorySource*`, `MemoryStatus*`, `MergeStrategy`,
  `EpisodicMemory`, `SemanticMemory`, `Timestamp`, `MemoryId` — ✅.
- `guide_registry.py`: `GuideRegistry`, `StandardGuideRegistry`, `Guide`,
  `GuideConflict`, `GuideStatus*`, `GuideSource*`, `ImprovementSignal*`,
  `PendingReason*`, `SessionOutcome*` — ✅.
- `sensor.py`: `SensorChain`, `StandardSensorChain`, `Sensor*`,
  `SensorTrigger*`, `SensorSignalFlag*` — ✅.
- `hooks.py`: `HookChain`, `StandardHookChain`, `Hook`, `CommandHook`,
  `FunctionHook`, `HookEvent`, `HookDecision`, `PlanArtifact`,
  `ToolCallSummary`, `TurnOutput`, all `*Context` event types — ✅.
- `middleware.py`: `MiddlewareChain`, `StandardMiddlewareChain`,
  `Middleware`, `MiddlewareDecision`, `LoopDetectionMiddleware`,
  `PatchToolCallsMiddleware`, `PreCompletionChecklistMiddleware`,
  `TokenBudgetMiddleware`, `TracingMiddleware`, `HookPoint` — ✅.
- `metric.py`: `MetricEvaluator`, `CommandMetricEvaluator`,
  `LlmJudgeEvaluator`, `LatencyEvaluator`, `TestPassRateEvaluator`,
  `MetricResult`, `IterationStatus`, `should_keep` — ✅.
- `verifier.py`: `Verifier`, `CompositeVerifier`, `TestSuiteVerifier`,
  `EvaluatorResponseVerifier`, `VerifierInput`, `VerifierVerdict*` — ✅.
- `execution_registry.py`: `ExecutionRegistry`, `ExecutionRegistryBuilder`,
  `EscalationMode*`, `StrategyResolution*`, `AutoGrantInfo`, `GrantCallback`
  — ✅.
- `cache_provider.py`: `CacheProvider`, `AnthropicCacheProvider`,
  `OpenAICacheProvider`, `OllamaCacheProvider`, `NullCacheProvider`,
  `auto_detect_cache_provider`, `CacheAnnotationResult`, `CacheStats` — ✅.
- `sandbox.py`: `WorkspaceScopedSandbox`, `SandboxBuildError`,
  `SandboxViolationException`; `harness.py` carries `SandboxProvider`,
  `NullSandbox`, `ReadOnlySandbox`, `IsolationMode*`, `NetworkPolicy*`,
  `BwrapProfile`, `WorkspaceConfig` — ✅.
- `tasklist.py`: `TaskList`, `TaskStatus`, `TaskListError`,
  `plan_artifact_to_task_list`, `validate_transition`, `would_create_cycle`,
  `TASK_LIST_EXTRAS_KEY` — ✅ (step-ledger re-export gap in §5.2).
- `plan.py`: `capture_plan_artifact`, `PlanPhaseError`,
  `PLAN_EXECUTE_EXTRAS_KEY` — ✅.
- `skills.py`: `parse_skill_doc`, `SkillCatalog`, `SkillEntry`, `LOAD_SKILL`
  — ✅.
- `prompt_tool_call.py`: `inject_tool_prompt`, `parse_prose_response`,
  `detect_prose_response` (different contract — §5.5),
  `AdaptiveToolCallModelInterface`, `PromptBasedToolCallModelInterface` — ✅.
- `output_schema.py`: output-schema enforcement — ✅ (fixture replayed).

## 7. Validation strategy

1. **Per-gap regression tests added before/with implementation** (each §5
   item lists its test file).
2. **Shared fixtures**: re-run every `fixtures/`-backed Python test; no
   fixture is edited. Add fixture-replay tests for any Rust fixture-replay
   test that lacks a Python counterpart (stream-event attribution,
   tool-call repair, hill-climbing VCS).
3. **Full suite**: `uv run pytest` (target: all green, no new skips);
   `uv run ruff check .`; `uv run ruff format --check .`.
4. **Examples gate**: every `examples/python/*/main.py` imports cleanly and
   per-example tests pass (mirrors CI `examples` job).
5. **Public-API test**: a new `test_public_api.py` asserting every Rust
   re-export has a Python counterpart importable from `spore_core`.

## 8. Compatibility constraints & decisions to escalate

- **C1 (resolved by plan review)**: implement **full** `OllamaEmbeddingModel`
  and `OpenAIEmbeddingModel` provider behavior now (§5.14), including the
  HTTP-backed provider impls — not a follow-up. No longer an open decision.
- **C2**: all new stream-event fields default to absent/empty to preserve
  byte-identical serialization for existing consumers (matches Rust
  `#[serde(default)]`). No escalation — proceeding with additive defaults.
- **C3**: `PausedState` gains optional `current_sha`/`best_sha` for
  hill-climbing (§5.3). Default `None` keeps existing serialized states
  loadable. No escalation.
- **C4**: no Rust change is required by any gap. Confirmed.
- **C5 (resolved by plan review)**: `image.rs` is a public Rust module; a
  Python `spore_core.image` submodule + `__init__` re-exports will be added
  (§5.15). No longer deferred.
- **C6 (resolved by plan review)**: eval/CLI parity is pinned by executable
  validation against `fixtures/task_suites/core_suite.json` (§5.17), not
  inferred. No longer an open item.

## 9. Risks

- **R1**: stream-event `node` attribution touches every combinator's stream
  wrapping — highest-blast-radius change; mitigated by default-`None` +
  omit-when-`None` serialization and a dedicated back-compat byte-equality
  test.
- **R2**: tool-call repair pass must be bounded (`max_passes`) to avoid
  infinite re-dispatch; Rust defaults to 1 — Python will match.
- **R3**: `ToolSearchRegistry` LRU + namespace-collision rules are subtle;
  ported behind focused tests.
- **R4**: large diff; will be split into per-gap commits for review.

## 10. `AGENTS.md` contributor-guidance change

This task owns the durable `AGENTS.md` rule (via the project memory
procedure): add a concise rule that **Rust is the authoritative behavioral
reference and Rust source/tests/examples/config/docs are read-only for
parity work absent explicit captain permission.** No other shared-file
change. The Go and TypeScript parity tasks are prohibited from editing
`AGENTS.md`.

## 11. Implementation order (after approval)

1. §5.2 re-exports + `test_public_api.py` (low-risk, unblocks API parity).
2. §5.1/§5.7/§5.9 stream-event wire format + `NodeAttr` + `test_harness_stream_events.py`.
3. §5.4 steering + §5.6 `CommandProgressSink` + `ToolProgress`.
4. §5.5 tool-call repair (new module + harness wiring).
5. §5.3 hill-climbing VCS revert.
6. §5.10/§5.11 tool-registry progress + search.
7. §5.12/§5.13 compaction ext + `CustomCondition`.
8. §5.14 embedding + vector_store (full provider impls per C1).
9. §5.15 `image` module (per C5).
10. §5.16 `set_plan` + `StandardTools`.
11. §5.17 eval/CLI executable validation.
12. §4.15 cancellation/timeout/retry/concurrency regression tests
    (`test_harness_cancellation.py`, `test_harness_timeout.py`,
    `test_provider_retry.py`, `test_tool_dispatch_concurrency.py`).
13. Final matrix update + full suite + PR.

## 12. Final parity matrix (implementation complete)

Rust authoritative revision: `7058db115b4f05b6b7cde1a3f62ba345c202bcde`.
Python suite: **1732 passed, 8 skipped**; `ruff check` clean; `ruff format --check` clean; all 12 `examples/python/*` import + per-example tests pass.

| Step | Capability | Implementation | Evidence |
|---|---|---|---|
| §5.2 | Missing `__init__` re-exports | `__init__.py` import + `__all__` additions, including `CustomCondition` alias | `test_public_api.py` (8) |
| §5.1/§5.7/§5.9 | Stream-event wire format (`NodeAttr`, `TurnEnd.usage`, `ToolResult.revised`, `UserMessage.steered`, `StreamToolProgress`) | `harness.py` `_StreamEvent` base + `NodeAttr` + `wrap_node_sink` wired in react leaf + 4 combinators; resume-path `StreamToolCall`/`StreamToolResult` emission | `test_harness_stream_events.py` (9); golden `test_harness_streaming.py` normalized |
| §5.4 | Mid-run `steering_source` | `HarnessConfig`/`HarnessBuilder.steering_source` + react-loop drain → `StreamUserMessage(steered=True)` | `test_harness_stream_events.py` steering test |
| §5.6 | `CommandProgressSink` + `ToolProgress` | `harness.py` `CommandProgressSink` + `SandboxProvider.execute_command_streaming`; `tool_registry.py` `ToolProgressUpdate`/`ToolProgressSink` + `ToolContext.command_progress`/`with_progress`; `_effective_tool_registry` bridges stream→`StreamToolProgress` | `test_harness_stream_events.py` ToolProgress test |
| §5.5 | Tool-call repair + prose/nudge recovery | new `tool_call_repair.py` (`coerce_tool_args`/`ToolCallFailure`/unified extended `detect_prose_response`/`StandardToolCallRepair`); runtime `prompt_tool_call.detect_prose_response` delegates to the same detector; `HarnessConfig.tool_call_repair`/`max_repair_attempts` + `_dispatch_with_repair`; `HarnessConfig.tool_call_nudge`/builder + structural execute-phase zero-tool nudge | `test_tool_call_repair.py` (17); `test_harness_plan_execute.py` nudge tests |
| §5.3 | Hill-climbing VCS-aware revert | `HillCandidate`; `VcsProvider.head/commit_checkpoint/reset_to`; `_hill_climbing_head_sha/commit_accepted/reset_to/apply_winner`; `hill_iteration` mutates `cand`; winner applied at exits; `best_sha` in partial JSON | `test_harness_hill_climbing.py` VCS-aware + winner tests |
| §5.10/§5.11 | Tool-registry progress + search | `ToolProgressSink`/`ToolProgressUpdate`; `ToolNamespace`/`ToolSearchConfig`/`ToolSearchRegistry` (LRU, budget eviction, namespace collision drop, dispatch-not-gated); harness wraps only catalogue-backed registries and preserves injected custom `tool_registry` unchanged | `test_tool_search_registry.py` (13) |
| §5.12 | `HarnessContextManagerExt` | `compaction_adapter.py` name-parity shim | re-export + `test_public_api.py` |
| §5.13 | `CustomCondition` | `prompt_assembly.CustomCondition = CustomPredicate`; root re-export | `test_public_api.py` alias test |
| §5.14 | `embedding` + `vector_store` (full provider impls) | new `embedding.py` (`cosine_similarity`/`EmbeddingModel`/`OllamaEmbeddingModel`/`OpenAIEmbeddingModel`); terminal OpenAI embedding 429s classify as `ProviderError`; new `vector_store.py` (`VectorStore`/`InMemoryVectorStore`/…) | `test_embedding.py` (22) |
| §5.15 | `image` module | new `image.py` (`media_type_for_extension`/`is_image_path`/`image_content`/`load_image_content`) | `test_image.py` (5) |
| §5.16 | `set_plan` + `StandardTools` | new `spore_tools/tools/set_plan.py` + `StandardTools.set_plan()` preset | `test_set_plan.py` (6) |
| §5.17 | Eval/CLI executable validation | `test_eval_cli_fixture.py` pins CLI to shared `core_suite.json` | `test_eval_cli_fixture.py` (7) |
| rereview | PlanExecute empty-plan retry | `HarnessConfig.plan_empty_max_retries`/builder + PlanExecute re-ask loop; retry suppresses tools via reserved `__none__` toolset and forces canonical plan schema; default `0` preserves hard-fail | `test_harness_plan_execute.py` (18) |
| §4.15 | Cancellation/timeout/retry/concurrency/cleanup/steering regression | `test_parity_lifecycle.py` (retryable set, backoff curve, `retryable()`, wall-time budget, `dispatch_all` concurrency, cleanup finalization success+failure, steering in-flight no-cancel) | `test_parity_lifecycle.py` (9) |
| §10 | `AGENTS.md` parity rule | `AGENTS.md` Rust-authoritative + read-only rule | committed |

### Remaining gaps (explicit)

None outstanding from the approved plan. All 13 implementation steps landed with
passing tests. The following were confirmed **not applicable** (Rust-internal or
feature-gated): `text_util`/`sync_util`/`macros`/`provider_http` (internal),
`sandbox_petri` (feature-gated), `BoxFut` (Rust async type).

### Review-fix log (independent review of PR #168)

Every actionable finding was fixed (the `CLAUDE.md` symlink finding was
retained per the intake — `fm-ensure-agents-md.sh` owns that guarded path):

- **High — NodeAttr wire compat**: `NodeAttr` now omits `agent_id` when `None`
  and `path` when empty via a `model_dump`/`model_dump_json` override + a
  `_StreamEvent` `@field_serializer("node")`; golden test
  `test_node_default_agent_omits_agent_id_and_empty_path`.
- **High — VectorStore API**: added `delete(filter) -> int`, `len() -> int`,
  `is_empty() -> bool` to the protocol + `InMemoryVectorStore`; tests for
  filtered delete, empty-filter full clear, `len`, `is_empty`.
- **Medium — ToolSearchRegistry harness integration**: `HarnessConfig.tool_search`
  + `HarnessBuilder.tool_search` + `_maybe_tool_search` wrapping in
  `_effective_tool_registry`; integration test asserting a harness run
  advertises only core + `tool_search` before a search.
- **Medium — OpenAI embedding error envelope**: `_openai_error_message`
  unwraps `{error:{message}}` (falls back to raw 500 chars); test
  `test_openai_embed_error_envelope_unwrapped`.
- **Medium — non-string `query`**: `_run_search` requires a `str` query;
  test `test_tool_search_rejects_non_string_query`.
- **Low — non-finite numeric coercion**: `_parse_number` guards
  `math.isfinite()`; tests for `NaN`/`inf`/`-inf`.
- **Low — lifecycle claims**: added executable cleanup-finalization tests
  (success + failure outcomes via `InMemoryObservabilityProvider`) and a
  steering in-flight-cancellation test (steering injects AFTER the tool
  result, never mid-turn) — `test_parity_lifecycle.py`.
- **Low — contradictory status text**: the document status is now a single
  coherent final-review state; stale plan-only lines removed.

### Rereview-fix log (second independent review of PR #168)

Every actionable finding in `data/parity-python-rereview/report.md` was fixed:

- **Medium — runtime prose detector**: `prompt_tool_call.detect_prose_response`
  now delegates to the extended `tool_call_repair` detector; regression covers
  `"I'll inspect ..."` on the runtime path.
- **Medium — `tool_call_nudge`**: added `HarnessConfig.tool_call_nudge` +
  `HarnessBuilder.tool_call_nudge`; ReAct tracks execute-phase zero-tool windows
  and applies Rust's bounded structural nudge. Regressions cover default-off and
  enabled behavior.
- **Medium — `plan_empty_max_retries`**: added config/builder/accessor and a
  PlanExecute retry loop. Retries suppress tools with reserved `__none__` and
  force the canonical plan JSON schema; discarded usage is folded into terminal
  accounting. Regressions cover `0` hard-fail and `>0` successful re-ask.
- **Medium — custom `tool_registry` + `tool_search`**: `_effective_tool_registry`
  now preserves injected custom registries unchanged when no catalogue registry
  exists; regression asserts no synthetic `tool_search` schema is advertised.
- **Medium — terminal OpenAI embedding 429**: terminal 429 after retry exhaustion
  now raises `ProviderError(code=429, ...)`, matching Rust's provider-error
  classification; retry-before-exhaustion behavior is unchanged.
- **Low — `CustomCondition`**: added `prompt_assembly.CustomCondition =
  CustomPredicate` and root re-export with public API coverage.
- **Low — matrix freshness**: §12 evidence and counts updated to the rereview
  state.

---

resolved: plan review decisions incorporated (C1 full embedding/vector_store provider impls now; C5 image.rs confirmed public module, concrete plan; C6 eval/CLI parity pinned by executable validation against fixtures/task_suites/core_suite.json; §4.15 cancellation/timeout/retry/cleanup + async/concurrency matrix added with evidence and regression tests)

resolved: independent review findings fixed (NodeAttr wire compat, VectorStore delete/len/is_empty, ToolSearchRegistry harness integration, OpenAI error envelope, non-string query, non-finite coercion, cleanup/steering lifecycle tests, status text coherence) [key=parity-python-review]

resolved: second independent rereview findings fixed (unified runtime prose detector, `tool_call_nudge`, `plan_empty_max_retries`, custom-registry `tool_search` fallback, terminal OpenAI embedding 429 classification, `CustomCondition` alias/export, final matrix refreshed) [key=parity-python-rereview]