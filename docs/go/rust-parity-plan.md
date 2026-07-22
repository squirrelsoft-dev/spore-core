# Rust → Go Behavioral Parity Plan (revised)

> **Status:** implementation complete on branch `fm/parity-go` for the approved
> bounded Go core+eval parity scope. Authoritative Rust revision recorded below.
> Rust source/tests/examples/config were read **read-only**; nothing under
> `rust/`, `docs/rust/`, `examples/rust/`, or Rust-specific configuration was
> modified.

**Authoritative Rust revision:** `7058db115b4f05b6b7cde1a3f62ba345c202bcde`
(recorded via `git rev-parse HEAD` on the disposable worktree's detached HEAD,
branch `fm/parity-go`).

## Captain scope decision (incorporated)

The captain has decided the scope for this task:

- **Complete Go core + eval parity**, *including* the full core
  `EmbeddingModel` + `VectorStore` providers and the missing Go documentation.
- **Document `spore-mcp`, `spore-sqlvec`, and the feature-gated `petri`
  microVM sandbox as known gaps.** Do **not** create those subsystems in Go.
- Rust remains strictly read-only; no Rust change is required by any gap below.

This revises the earlier Tier-2 escalation: MCP / SQL-vector / Petri are now
**out of scope as deliverables** and are recorded in §H as documented known
gaps. The core `embedding` and `vector_store` modules (§B1, §D1) are **in
scope** and will be ported with their corrected contracts.

## Scope & method

Bring `go/` into full *observable behavioral* parity with the authoritative
Rust `spore-core` (+ `spore-eval`, `spore-cli`). "Observable behavioral" means
public APIs, CLI behavior, configuration/defaults, agent/session lifecycle,
tool invocation, streaming/events, state/serialization, errors/exit behavior,
cancellation/timeout/retry/cleanup, concurrency, extension/provider seams,
examples, and documented usage — as exercised by tests and fixtures, not
byte-identical internal hashing or idiomatic shape.

Every capability was inventoried by reading the Rust source and its Go
counterpart. This **revised** audit re-walked the full Rust `HarnessConfig`
field set, the harness `StreamEvent` surface, `HarnessSignal`, `ToolOutput`,
`RunResult`, `HaltReason`, the `EmbeddingModel`/`VectorStore` traits, and the
docs tree — rather than assuming parity. Findings distinguish **OBSERVED
PARITY** (verified identical by reading both) from **INFERRED SIMILARITY**
(looks equivalent, not byte-checked) from **GAP** (Rust has it, Go does not,
with file:line evidence). Several initial "missing" claims were **disproven**
by direct grep and recorded as parity (§3) to avoid re-litigating.

## 1. Repository & module map

### Rust crates → Go
| Crate | Role | Go counterpart | Status |
|---|---|---|---|
| `spore-core` | harness runtime | `go/spore-core` | gaps §A–§G |
| `spore-cli` | thin CLI binary (stub) | `go/spore-cli` (stub) | OBSERVED PARITY |
| `spore-eval` | evaluation harness | `go/spore-eval` | OBSERVED PARITY |
| `spore-mcp` | MCP client → native Tools (quarantines `rmcp`) | — | **known gap §H (not created)** |
| `spore-sqlvec` | sqlite-vec `VectorStore` (quarantines `rusqlite`+C ext) | — | **known gap §H (not created)** |

### Rust `spore-core` modules → Go packages
| Rust module | Go location | Status |
|---|---|---|
| `agent` | `agent.go` | OBSERVED PARITY |
| `anthropic` | `anthropic/` | OBSERVED PARITY (extended sampling = §B3) |
| `cache_provider` | `contextmgr/cache_provider.go` | OBSERVED PARITY |
| `compaction_adapter` | `contextmgr/compaction_adapter.go` | OBSERVED PARITY |
| `context` | `contextmgr/` | OBSERVED PARITY |
| `embedding` | — | **GAP §B1 (in scope, corrected contract)** |
| `execution_registry` | `execution_registry.go` | OBSERVED PARITY |
| `guide_registry` | `guideregistry/` (+ `provider.go`) | OBSERVED PARITY |
| `harness/*` | `harness.go`, `executor.go`, `strategy.go`, `execution_scaffold.go`, `ralph.go`, `hill_climbing.go`, `self_verifying.go`, `plan.go`, `preset/` | OBSERVED PARITY on verified seams; **config/event gaps §E** |
| `hooks` | `hooks.go` | OBSERVED PARITY |
| `image` | — | **GAP §B2** |
| `macros` | n/a (Rust proc-macro) | intentional absence |
| `memory` | `memory/` (+ `provider.go`) | OBSERVED PARITY |
| `metric` | `metric/` | OBSERVED PARITY |
| `middleware` | `middleware/` | OBSERVED PARITY |
| `model` | `model.go` | OBSERVED PARITY (ModelParams fields = §B3) |
| `observability` + `observability_outbox` | `observability/` | OBSERVED PARITY |
| `ollama` | `ollama/` | **GAP §B4 (thinking/reasoning)** |
| `openai` | `openai/` | **GAP §B3 (extended sampling wire)** |
| `output_schema` | `output_schema.go` | OBSERVED PARITY |
| `plan` | `plan.go` | OBSERVED PARITY (SC-28 free-text capture present) |
| `prompt_assembly` | `promptassembly/` | OBSERVED PARITY (harness wiring = §F1) |
| `prompt_chunk_registry` | `promptchunkregistry/` | OBSERVED PARITY (hash differs by design — §3) |
| `prompt_tool_call` | `prompt_tool_call.go` | OBSERVED PARITY |
| `provider_http` (`pub(crate)`) | duplicated per-provider | INFERRED SIMILARITY (internal; not public) |
| `sandbox` | `sandbox.go`, `sandbox_defaults.go` | OBSERVED PARITY (streaming seam = §E2) |
| `sandbox_petri` (feature `petri`) | — | **known gap §H (not created)** |
| `sensor` | `sensor/` (+ `provider.go`) | OBSERVED PARITY |
| `skills` | `skills.go` | OBSERVED PARITY |
| `storage` (+ `storage/`) | `storage/` (+ `project_id.go`, `filesystem.go`) | OBSERVED PARITY |
| `tasklist` | `tasklist.go` | OBSERVED PARITY |
| `termination` | `termination/` (+ `completion_check.go`) | OBSERVED PARITY |
| `tool_call_repair` | — | **GAP §C1** |
| `tool_registry` | `tool_registry.go` | **GAP §C2 (ToolNamespace/ToolSearch) + §E2 (progress sink)** |
| `tools/*` | `tools/` | **GAP §A (set_plan/create_dir/catalogue)** |
| `vector_store` | — | **GAP §D1 (in scope, corrected contract)** |
| `verifier` | `verifier/` | OBSERVED PARITY |

## 2. Parity matrix — required axes

| Axis | Rust evidence | Go behavior | Status |
|---|---|---|---|
| Public APIs & CLI | `spore-cli` `fn main(){}`; `lib.rs` re-exports | `spore-cli` `func main(){}`; root `sporecore` exports | OBSERVED PARITY (CLI); API gaps §A–§G |
| Configuration / defaults | `HarnessConfig` fields + `HarnessBuilder` setters (`harness/config.rs`) | mirrored; **missing fields**: `tool_call_repair`+`max_repair_attempts` (§C1), `plan_empty_max_retries` (§E6), `tool_call_nudge` (§E5), `steering_source` (§E4), `tool_search` (§C2), `chunk_provider` (§F1) | GAP §C1/§C2/§E4–§E6/§F1 |
| Agent / session lifecycle | `agent.rs`; `SessionState`; `PausedState` | `agent.go` mirrors | OBSERVED PARITY |
| Tool invocation | `tool_registry.rs` `Tool`/`ToolContext`/dispatch + repair-on-dispatch + progress sink | mirrors repair (§C1) + progress (§E2) gaps | GAP §C1/§E2 |
| Streaming / events | `harness::StreamEvent` 18 variants, each `node: Option<NodeAttr>`; `ToolProgress`; `ToolResult.revised`; `UserMessage.steered`; `TurnEnd.usage` | 18 variants but **no `NodeAttr`**, no `ToolProgress`, no `revised`, no `steered`, no `TurnEnd.usage` | GAP §E1/§E2/§E3/§E4/§E9 |
| State / serialization | serde tagged; `skip_serializing_if` | tagged structs + `omitempty`; snake_case | OBSERVED PARITY (fixtures replay byte-identical) |
| Errors / exit | `ModelError` 7 variants + `retryable()`; `HaltReason` 23 variants; `HarnessSignal` 5 variants | `ModelError` 7 + `Retryable()`; `HaltReason` 22 (**missing `RepeatedToolCall`**); `HarnessSignal` 4 (**missing `PlanRecorded`**) | GAP §E7/§E10 |
| Cancellation / timeout / retry / cleanup | tokio cancel; provider retry; `GroupKillGuard` | `context.Context`; per-provider retry; `exec.CommandContext` kill | OBSERVED PARITY — **executable evidence §4** |
| Concurrency | `Send` + `Arc` + `Mutex` | interfaces + `sync.Mutex` | OBSERVED PARITY — **race evidence §4** |
| Extension / provider seams | `ModelInterface`/`ToolRegistry`/`SandboxProvider`/`MemoryProvider`/`VectorStore`/`EmbeddingModel`/`CacheProvider` | consumer-side interfaces; `Standard*` in `provider.go` | GAP §B1/§D1 (embedding/vector) |
| Examples | `examples/rust/01..12` | `examples/go/01..12` (1:1) | OBSERVED PARITY |
| Documented usage | `docs/rust/` 4 substantive pages | `docs/go/quickstart.md` stub only | GAP §G1 |

## 3. Intentional non-gaps (verified — do NOT "fix")

1. **`sandbox_petri` / `MicroVm`** — Rust gates petri behind feature `petri`;
   both Rust (without `petri`) and Go return `DisallowedCommand` for
   `MicroVm`. Now a **documented known gap (§H)** per captain scope — not created.
2. **`prompt_chunk_registry` block hash** — Rust `DefaultHasher` (SipHash) vs Go
   `fnv.New64a`. Fixture `fixtures/prompt_chunk_registry/basic.json` explicitly
   states cross-language byte equality is NOT required. Internal change-detection
   only. → intentional, fixture-sanctioned.
3. **`provider_http` shared retry** — Rust `pub(crate)`; Go duplicates per-
   provider. Behaviorally equivalent; not public. → INFERRED SIMILARITY.
4. **`StandardMemoryProvider`/`StandardSensorChain`/`StandardGuideRegistry`** —
   exist in Go `memory/provider.go`, `sensor/provider.go`,
   `guideregistry/provider.go`. Disproven "missing". → OBSERVED PARITY.
5. **`ProjectId`/`ActiveRun` lifecycle** — exist in `storage/project_id.go`.
   → OBSERVED PARITY.
6. **Termination domain checks** (`FeatureList`/`TestSuite`/`QuestionAnswered`/
   `SqlResult`) — exist in `termination/completion_check.go`. → OBSERVED PARITY.
7. **#150 recoverable sandbox violations** — Go has `ToolOutputSandboxViolation`
   + `SandboxViolationPolicy` + `IsAlwaysHalt()` (`harness.go:755–1123`).
   → OBSERVED PARITY.
8. **#146 `web_fetch` `start_byte`** — `applyWebFetchRange` (`tools/web.go:200`).
   → OBSERVED PARITY.
9. **#149 Ollama `num_ctx`** — Go `with_context_window` sets both. → OBSERVED PARITY.
10. **Truncation constants** `LARGE_OUTPUT_THRESHOLD`/`DEFAULT_HEAD_TOKENS`/
    `DEFAULT_TAIL_TOKENS` + `finish_with_possible_truncation` —
    `tools/tools.go:31–37,169`. → OBSERVED PARITY.
11. **`spore-eval`, `examples/go/01..12`, `spore-cli` stub** — full 1:1 parity.
    → OBSERVED PARITY.
12. **`content_capture` / `pricing`** — Rust top-level `HarnessConfig` fields;
    Go carries them on the `observability.HarnessObserver` (root package can't
    import `observability` — cycle). Idiom difference, behaviorally equivalent.
    → OBSERVED PARITY (idiom).

## 4. Lifecycle / cancellation / timeout / retry / cleanup / concurrency — executable evidence

The reviewer asked for *executable* evidence, not assertions. The following Go
tests were run on the current tree (branch `fm/parity-go`, baseline green) and
PASS, demonstrating each axis:

| Axis | Test | Result |
|---|---|---|
| Timeout (sandbox) | `go test -run TestSandboxExecuteCommandTimeout .` | PASS (0.05s) |
| Timeout (default applies) | `TestExecConfigDefaultTimeoutAppliesWhenCallPassesNone` | PASS |
| Timeout (per-call override) | `TestExecConfigPerCallTimeoutOverridesDefault` | PASS |
| Cleanup (kill-on-drop reaps child) | `TestExecConfigKillOnDropReapsChildOnTimeout` | PASS (2.52s) |
| Timeout (model surface) | `TestTimeoutSurface` | PASS |
| Timeout (hill-climbing counts as non-improvement) | `TestHillClimbingTimeoutCountsAsNonImprovement` | PASS |
| Retry (providers) | `go test -run 'Retry|Backoff|Retryable|Status' ./anthropic/ ./openai/ ./ollama/` | PASS (all three) |
| Concurrency (race detector) | `go test -race -run 'TestSandboxExecuteCommandTimeout|TestExecConfigKillOnDropReapsChildOnTimeout|TestHillClimbingTimeoutCountsAsNonImprovement' .` | PASS (race-clean) |

These confirm the existing Go cancellation/timeout/retry/cleanup/concurrency
seams are live and exercised. The gaps below are *additive* behaviors Rust has
that Go does not; none of them weaken this evidence. New gaps that touch these
axes (§E2 streaming sandbox, §E4 steering) will add dedicated tests when
implemented.

### 4.1 Explicit context-cancellation test (required — does not yet exist)

The table above covers **timeout** but not **context cancellation**. A grep
for `WithCancel`/`WithDeadline`/`context.Canceled`/`ctx, cancel` across all Go
tests returns **no matches** — there is currently **no executable test proving
a cancelled model/tool call stops and reaps work**. The cancellation seam
itself exists (`context.Context` is the first arg of `ModelInterface.Call`/
`CallStreaming`/`CountTokens` and `SandboxProvider.ExecuteCommand`; the latter
uses `exec.CommandContext`, which kills the child when the ctx is cancelled),
but it is not exercised by any test.

Per the reviewer's instruction, an **explicit context-cancellation executable
test** is required. It is recorded here as the **first post-approval
validation step** (TDD: written before any gap implementation, run against the
existing seam, expected to PASS immediately, then kept green). It is NOT run
in this plan-only checkpoint because the reviewer said "do not implement."

Required tests (to be added under `go/spore-core/`):

1. **`TestCancelledToolCallStopsAndReaps`** — drive a `WorkspaceScopedSandbox.
    ExecuteCommand` running a long-lived child (`sleep 30` or a shell loop)
    with a `context.WithCancel`; cancel the ctx from another goroutine after a
    short delay; assert: (a) `ExecuteCommand` returns promptly with a
    `context.Canceled`-ish error (or the sandbox's cancellation error shape),
    (b) the child process is **reaped** (no orphan remains — verify via
    `pgrep`/process listing or a pidfile the child writes and the test reads
    after return), (c) total wall time is far below 30s. This is the
    cancellation analogue of `TestExecConfigKillOnDropReapsChildOnTimeout`.
2. **`TestCancelledModelCallStops`** — a `ModelInterface` impl whose `Call`
    blocks on `<-ctx.Done()` and then returns `ctx.Err()`; call it via the
    harness react path (or directly) with a cancellable ctx; cancel; assert the
    call returns `context.Canceled` (mapped to the appropriate `ModelError`/
    `HaltReason` per the existing error mapping) promptly and the run stops
    without spinning further turns.
3. **`TestCancelledStreamingCallStops`** — a `CallStreaming` impl that emits
    one event then blocks on `ctx.Done()`; cancel; assert the channel closes
    with the cancellation error promptly (no goroutine leak — `goleak` or a
    `runtime.NumGoroutine` before/after check).

These prove cancellation (not just deadline) stops in-flight model/tool work
and reaps it. They will be the first tests committed post-approval and must
pass before any §A–§G gap work begins.

## 5. Gaps to close (detailed)

Each gap: Rust evidence, Go behavior/gap, file-level steps, regression/E2E
validation, compatibility, risks, intended final evidence.

### Tier 1 — Core behavioral ports (in scope, no new third-party dep)

#### §A1 — `set_plan` tool
- **Rust:** `tools/set_plan.rs` (130 lines): `SetPlanTool` writes an ordered
  step list to the durable `RunStore` blob (`TASK_LIST_EXTRAS_KEY`) that
  `task_list` authors; in `StandardTools::coding_set`/`full_set`
  (`catalogue.rs:107,279,304`). Its terminal signal is `HarnessSignal::PlanRecorded`
  (see §E7).
- **Go gap:** No `set_plan` tool, no `StandardTools.SetPlan`, absent from
  `CodingSet`/`FullSet` (`tools/catalogue.go:161–205`).
- **Steps:** new `tools/set_plan.go` (`SetPlanTool`: schema name `set_plan`,
  params `tasks` (required non-empty string array) + optional `rationale`;
  `Execute` builds a linear `TaskList`, persists under `TASK_LIST_EXTRAS_KEY`
  via `ctx.ProjectNamespace`, returns `Success("Recorded a N-step plan.")`,
  recoverable errors); `catalogue.go` adds `SetPlan()` + inserts into
  `CodingSet`/`FullSet` (after `TaskList`).
- **E2E:** port `set_plan.rs` schema test; seed a `RunStore`, call `Execute`,
  assert blob round-trips through `TaskList` load; catalogue name-set drift
  guard (exact 18-tool `CodingSet` / 22-tool `FullSet`).
- **Compat:** additive. **Risks:** low (use `ProjectNamespace`, #142).
- **Final evidence:** `go test ./tools/` green; name-set guard green.

#### §A2 — `create_dir` tool
- **Rust:** `tools/fs.rs` `CreateDirTool` (`mkdir -p`, `read_only=false`);
  `catalogue.rs:22,283,305`.
- **Go gap:** No `CreateDirTool` in `tools/fs.go`; no `StandardTools.CreateDir`.
- **Steps:** `tools/fs.go` `CreateDirTool` (param `path`; `os.MkdirAll 0o755`;
  path-escape → `ToolOutputSandboxViolation`); `catalogue.go` `CreateDir()` +
  add to `CodingSet`/`FullSet` (after `EditFile`).
- **E2E:** port `fs.rs` `create_dir` tests (nested dirs, path escape, read-only
  violation). **Compat:** additive. **Risks:** low.
- **Final evidence:** `go test ./tools/` green; name-set guard green.

#### §A3 — `delete_file` / `move_file` catalogue exposure
- **Rust:** `StandardTools::delete_file`/`move_file` (`catalogue.rs:34,39`);
  both in `coding_set`/`full_set`.
- **Go gap:** `DeleteFileTool`/`MoveFileTool` exist in `tools/fs.go` but are
  NOT exposed via `StandardTools` and NOT in `CodingSet`/`FullSet`.
- **Steps:** `catalogue.go` adds `DeleteFile()`/`MoveFile()` wrapping existing
  constructors; insert into `CodingSet`/`FullSet` matching Rust order (after
  `create_dir`, before `list_dir`).
- **E2E:** name-set drift guard. **Compat:** additive (tools already exist).
  **Risks:** low (match Rust order for deterministic schema advertisement).
- **Final evidence:** `go test ./tools/` green; name-set guard green.

#### §B1 — `embedding` module (CORRECTED contract, full provider behavior)
- **Rust:** `embedding.rs` (550 lines) + `provider_http.rs` (shared retry).
  The trait is **batch + single**:
  - `embed(inputs: Vec<String>) -> Result<Vec<Vec<f32>>, ModelError>` — one
    vector per input IN INPUT ORDER; **empty inputs returns empty Vec without
    contacting the provider**.
  - `embed_one(input: String) -> Result<Vec<f32>, ModelError>` — convenience,
    **defaults to a one-element `embed`** (pops, errors
    `ProviderError { code: 0, message: "embedding provider returned no vectors" }`
    if none returned).
  - `dimensions() -> usize` (`0` = unknown); `info() -> EmbeddingInfo`
    `{provider, model_id, dimensions}`. Blanket `Arc<T: ?Sized>` impl.
  - `cosine_similarity(a,b) -> f32` in `[-1,1]`; 0.0 on length mismatch or zero
    magnitude (never NaN).
- **`OllamaEmbeddingModel`** (full behavior):
  - POST `/api/embed` `{model, input: [..]}` (batch); response `{embeddings:
    [[f32],…]}`. **No API key.** **No retry** (fails fast).
  - Error mapping: timeout→`ModelError::Timeout`; connect error→
    `ProviderError { code: 0, message: "Ollama not running at {base_url}" }`;
    non-2xx→`ProviderError { code, message: <body, first 500 chars> }`;
    decode failure→`ProviderError { code: 0, message: "decode embed response
    failed: {e}" }`; other transport→`Transport { message: "HTTP transport
    error: {e}" }`.
  - `dimensions` starts `0` (Ollama doesn't advertise length). Builders:
    `with_base_url` (default `http://localhost:11434`), `with_dimensions`,
    `with_timeout` (default 120s), `with_http_client`. `probe_dimensions()`
    embeds `"dimension probe"` via `embed_one` and returns `vector.len()`
    (does NOT mutate self — pin with `with_dimensions`).
- **`OpenAIEmbeddingModel`** (full behavior — reuses `provider_http`):
  - POST `${base_url}/embeddings` (default `https://api.openai.com/v1`);
    request `{model, input: [..], dimensions?: usize}` (`dimensions` omitted
    via `skip_serializing_if` when `None`). Response `{data: [{embedding,
    index}],…}`.
  - **Bearer auth:** `Authorization: Bearer {api_key}` on every request.
  - **Retry/backoff/retry-after** (shared `provider_http::send_with_retry`):
    `max_retries` (default **3**, `with_max_retries`); retryable statuses
    `408|425|429|500|502|503|504` (`is_retryable_status`); exponential backoff
    `backoff_delay(attempt)` = `500ms * 2^attempt` capped at **30s**; honor a
    `retry-after` header (whole seconds) when present, else backoff; also retry
    on transport **timeout** up to `max_retries`, then `Timeout`; other
    transport error→`Transport { message }`.
  - **Error-envelope mapping:** non-retryable response → `error_parts(resp)` →
    `ProviderError { code, message }` where `message` is the typed
    `{"error":{"message":…}}` body field, falling back to the first 500 chars
    of the raw body.
  - **Response sort-by-index:** `parsed.data.sort_by_key(|d| d.index)` before
    returning — `index` is authoritative so a non-conforming compatible server
    can't scramble the input↔vector correspondence.
  - **Default model dimensions** (`default_dimensions`): `text-embedding-3-large`
    → 3072; `text-embedding-3-small` | `text-embedding-ada-002` → 1536; else 0.
  - **`supports_dimensions_param`** gated on `model_id.starts_with("text-embedding-3")`;
    `with_dimensions(n)` overrides `dimensions` AND, for v3 families, sends the
    request `dimensions` truncation param (older models like `ada-002` ignore
    it on the wire).
  - **`from_env(env_var, model_id)`** validation: `provider_http::key_from_env` —
    rejects an **unset** env var (`ProviderError "env var `{env_var}` not set"`)
    and an **empty/whitespace** value (`ProviderError "env var `{env_var}` is
    empty"`); trims before the empty check.
  - **Timeout/transport mapping:** request built with `.timeout(self.timeout)`
    each attempt; timeout→retry-then-`Timeout`; connect/other→`Transport`.
  - Builders: `with_base_url`, `with_dimensions`, `with_timeout` (default 120s),
    `with_max_retries` (default 3), `with_http_client`. `Debug` redacts
    `api_key` as `<redacted>`.
- **Go gap:** No embedding module. (The earlier plan's `Embed(ctx, text)
  (EmbeddingInfo, error)` was **wrong** — corrected above to batch `Embed` +
  `EmbedOne`.)
- **Go interface caveat (must plan):** Go interfaces **cannot supply a default
  `EmbedOne` body** (no default methods). Rust's trait gives `embed_one` a
  default that calls `embed`. In Go, `EmbedOne` must be a **required interface
  method**, and **both providers implement it through a shared helper**
  (`embedOne(ctx, m, input)` in `embedding.go`: calls `m.Embed(ctx, []string{input})`,
  pops the single vector, returns `ProviderError "embedding provider returned
  no vectors"` if empty). This keeps the two providers DRY and the contract
  identical.
- **Steps:** new `go/spore-core/embedding/embedding.go`:
  - `EmbeddingModel` interface: `Embed(ctx, []string) ([][]float32, error)`,
    `EmbedOne(ctx, string) ([]float32, error)` (**required**, both impls use the
    shared `embedOne` helper), `Dimensions() int`, `Info() EmbeddingInfo`.
  - `EmbeddingInfo{Provider, ModelID string; Dimensions int}`;
    `CosineSimilarity(a, b []float32) float32` (0.0 on mismatch/zero-magnitude).
  - `OllamaEmbeddingModel`: `/api/embed` batch, the exact `ModelError` mapping
    above, `WithBaseURL`/`WithDimensions`/`WithTimeout`/`WithHTTPClient`,
    `ProbeDimensions` (via `EmbedOne("dimension probe")`); no retry.
  - `OpenAIEmbeddingModel`: `/embeddings` with **bearer auth**, the shared
    retry/backoff/retry-after loop (port `send_with_retry`/`is_retryable_status`/
    `backoff_delay`/`retry_after`/`error_parts` — either a small internal
    `providerhttp` helper or inline, matching the existing per-provider retry
    idiom), `FromEnv` with the unset/empty validation, `supportsDimensionsParam`
    gate, `defaultDimensions` table (3072/1536/0), `WithDimensions` sends the
    request param, **sort-by-index**, `WithBaseURL`/`WithTimeout`/`WithMaxRetries`(default 3)/`WithHTTPClient`, `Debug` redacts the key.
- **E2E (mocked HTTP assertions — mirror the Rust `wiremock` tests):**
  1. `cosine_basics` — 1.0/0.0/0.0, never NaN.
  2. `ollama_embed_round_trip` — mock `/api/embed` 200 `{embeddings:[[0.1,0.2,0.3],[0.4,0.5,0.6]]}`;
     assert `Embed(["hello","world"])` returns exactly those vectors in order.
  3. `ollama_empty_input_skips_call` — no mock mounted; `Embed([])` returns
     empty without contacting the server (point at a dead `127.0.0.1:1`).
  4. `openai_embed_orders_by_index_and_authenticates` — mock `/embeddings` 200
     with `data` out of order (`index:1` before `index:0`); assert the
     `Authorization: Bearer test-key` header was sent AND the result is
     reordered by `index` (input "a"→index 0→`[0.1,0.1]`).
  5. `openai_maps_error_envelope` — mock `/embeddings` 400 `{"error":{"message":"bad model"}}`;
     `WithMaxRetries(0)`; assert `ProviderError{Code:400, Message:"bad model"}`.
  6. `openai_retries_on_429_then_succeeds` — mock 429 then 200; assert retry
     happens and the vectors return.
  7. `openai_from_env_unset_rejected` / `from_env_empty_rejected` — assert the
     two `ProviderError` messages.
  8. `openai_default_dimensions` — `text-embedding-3-large`→3072,
     `text-embedding-3-small`→1536, unknown→0; `supportsDimensionsParam` true
     only for `text-embedding-3-*`.
  9. `embed_one_shared_helper` — both providers' `EmbedOne` pop the single
     vector and return the "no vectors" `ProviderError` on an empty response.
- **Compat:** additive (new package). **Risks:** low (stdlib `net/http`); match
  Rust wire field order + the exact `ModelError` variant mapping so the
  cross-language `retryable()` classification stays identical.
- **Final evidence:** `go test ./embedding/` green; the 9 mocked-HTTP
  assertions above pass.

#### §B2 — `image` helpers
- **Rust:** `image.rs` (93 lines): `media_type_for_extension`,
  `is_image_path`, `image_content`, `load_image_file` (base64 `Content::Image`
  producer).
- **Go gap:** `NewImageContent(mediaType, data)` exists (`model.go:197`) but no
  media-type-by-extension helper or file loader.
- **Steps:** new `go/spore-core/image.go` (root package): `MediaTypeForExtension
  (ext string) (string, bool)` (png/jpeg/gif/webp/bmp, case-insensitive),
  `IsImagePath(p string) bool`, `LoadImageContent(path string) (Content, error)`.
- **E2E:** port `image.rs` tests. **Compat:** additive. **Risks:** low.
- **Final evidence:** `go test .` (root) green for image helpers.

#### §B3 — `ModelParams` extended sampling + OpenAI wire
- **Rust:** `model.rs:159–170` — `top_k: Option<u32>`, `min_p: Option<f32>`,
  `presence_penalty: Option<f32>`, `repetition_penalty: Option<f32>`.
  `openai.rs:214–224,426` emits them as request fields; reasoning models drop
  them (alongside temperature/top_p/stop).
- **Go gap:** `ModelParams` (`model.go:242`) lacks all four; OpenAI wire
  (`openai/openai.go:247–248`) emits only `temperature`/`top_p`.
- **Steps:** `model.go` add `TopK *uint32`, `MinP *float32`,
  `PresencePenalty *float32`, `RepetitionPenalty *float32` (`omitempty`, nil ⇒
  byte-identical); `openai/openai.go` emit all four when non-nil, drop for
  reasoning models (matching Rust's drop set).
- **E2E:** port `openai.rs` extended-sampling wire test; HTTP-mock asserting
  appear/drop; request-hash test asserting nil ⇒ byte-identical.
- **Compat:** additive. **Risks:** low (verify Anthropic/Ollama ignore them).
- **Final evidence:** `go test ./openai/` + `go test .` (model) green.

#### §B4 — Ollama thinking / reasoning (#148)
- **Rust:** `ollama.rs:48–55,129–156,387+` — `OllamaThink` (boolean or
  `low|medium|high|max` from `reasoning_effort`); `supports_thinking()` (gated
  on `/api/show` `capabilities` containing `"thinking"`); `gate_thinking`
  (drop `think` for unsupported, warn once via `AtomicBool`); request `think`
  flag + `message.thinking` → `ContentBlock::Thinking` (unary) /
  `StreamEvent::ThinkingDelta` (streaming); distinct `THINKING_INDEX` (0)
  disjoint from text (1) + tools (2+); `ContentBlockStop` between thinking and
  text.
- **Go gap:** `ollama/ollama.go:36` comment only — "Thinking blocks are
  silently omitted." No `think` field, no `thinking` response field, no
  `ThinkingDelta`/`ContentBlockStop` in `parseNDJSONStream`, no
  `supports_thinking`, no `reasoning_effort`→`think` mapping.
- **Steps:** `ollama/ollama.go` add `supportsThinking` (capability check),
  `think` in `wireRequest` (bool when `reasoning_effort` nil; level string when
  set), `thinking` in `wireResponseMessage`, parse → `ContentBlockThinking` +
  `StreamEventThinkingDelta`; emit `ContentBlockStop` between thinking and
  text; `THINKING_INDEX` disjoint; once-guard warning.
- **E2E:** port `ollama.rs` thinking tests (unary, streaming
  `ThinkingDelta`+`ContentBlockStop`, unsupported drops+warns once,
  `reasoning_effort`→level); NDJSON stream fixture test.
- **Compat:** additive when unset (existing fixtures byte-identical). Known
  backlog **#148**. **Risks:** medium — streaming index management must match
  Rust's disjoint `THINKING_INDEX`.
- **Final evidence:** `go test ./ollama/` green; existing Ollama fixture-replay
  byte-identical.

#### §C1 — `tool_call_repair` module + harness wiring
- **Rust:** `tool_call_repair.rs` (300 lines): `ToolCallRepair` trait,
  `StandardToolCallRepair`, `coerce_tool_args` (key normalization + scalar
  coercion per schema), `ToolCallFailure` (`MalformedJson`/`ProseResponse`/
  `UnknownTool`), `detect_prose_response` + `ACTION_PHRASES`. Wired:
  `HarnessConfig.tool_call_repair: Option<Arc<dyn ToolCallRepair>>` +
  `HarnessConfig.tool_call_repair: Option<Arc<dyn ToolCallRepair>>` +
  `max_repair_attempts: u32` (**default 1**, `config.rs:133,404,526`) +
  `HarnessBuilder::tool_call_repair(..)` (`config.rs:1254`) +
  `HarnessBuilder::max_repair_attempts(attempts)` (`config.rs:1261`, "Set the
  maximum number of repair-and-re-dispatch attempts per tool call. Defaults to
  `1`. Ignored when no repair provider is set."); react dispatch
  (`react.rs:7–86,28`) coerces + re-dispatches, decrementing
  `attempts_left = self.config.max_repair_attempts` per attempt, enriching the
  error (schema + hint) when repair is exhausted.
- **Go gap:** No `ToolCallRepair`/`CoerceToolArgs`/`ToolCallFailure`, no
  `HarnessConfig.ToolCallRepair`/`MaxRepairAttempts`, no `WithToolCallRepair`/
  `WithMaxRepairAttempts` setters, no repair-on-dispatch. Only comments in
  `tools/define.go:29,74`. `DetectProseResponse` is folded into
  `prompt_tool_call.go`.
- **Steps:** new `go/spore-core/tool_call_repair.go`: `ToolCallRepair`
  interface, `StandardToolCallRepair`, `CoerceToolArgs(input, schema)
  (json.RawMessage, bool)`, `ToolCallFailure` (`Kind` + `Message`); port
  `coerce_tool_args` key/scalar coercion; `HarnessConfig` add `ToolCallRepair`
  + `MaxRepairAttempts uint32` + `WithToolCallRepair(repair)` +
  `WithMaxRepairAttempts(attempts)`; react dispatch repair arm (decrement
  `attemptsLeft = config.MaxRepairAttempts` per attempt, enrich error on
  exhaustion); `StandardToolCallRepair` uses `DetectProseResponse` for the
  `ProseResponse` kind.
- **MaxRepairAttempts setter/default validation (required):**
  - Default **1** when `ToolCallRepair` is set and `MaxRepairAttempts` is the
    zero value — mirror Rust's builder default (`config.rs:526`). Use a
    pointer sentinel (`*uint32`) OR an `effectiveMaxRepairAttempts()` reader
    (nil ⇒ 1) so a default-constructed config gets 1 while an explicit 0 is
    honored, matching the house idiom used for `ErrorLoopThreshold`/
    `OutputSchemaMaxRetries`.
  - **Ignored when `ToolCallRepair` is nil** — the whole repair arm is skipped
    (byte-identical default); `MaxRepairAttempts` has no effect without a
    repair provider.
  - Setter `WithMaxRepairAttempts(n uint32)` returns the config (fluent).
  - E2E: a test that sets `ToolCallRepair` + `MaxRepairAttempts=2` and asserts
    the dispatch retries exactly twice before enriching; a test that sets
    `ToolCallRepair` with the default and asserts exactly one retry; a test
    that leaves `ToolCallRepair` nil and asserts `MaxRepairAttempts` is
    irrelevant (no repair, byte-identical).
- **E2E:** port `tool_call_repair.rs` tests; port `harness/tests.rs:2759`
  repair-on-dispatch test; the MaxRepairAttempts tests above; nil ⇒
  byte-identical test.
- **Compat:** additive (nil ⇒ byte-identical). **Risks:** medium — exact
  re-dispatch semantics (one retry, no double budget charge, no duplicate
  tool message).
- **Final evidence:** `go test .` (tool_call_repair) + `go test -run Repair .`
  green.

#### §C2 — `ToolNamespace` + `ToolSearchRegistry` / `ToolSearchConfig`
- **Rust:** `tool_registry.rs` — `ToolNamespace` (one logical group; "one MCP
  server → one namespace"), `ToolSearchRegistry`/`ToolSearchConfig` (deferred
  tool exposure: core set + `tool_search` meta-tool). `HarnessConfig.tool_search:
  Option<ToolSearchConfig>` + `HarnessBuilder::tool_search(..)`
  (`config.rs:358–362,498,1319–1325`).
- **Go gap:** `tool_registry.go` has no `ToolNamespace`/`ToolSearchRegistry`/
  `ToolSearchConfig`; no `HarnessConfig.ToolSearch`. (Go's `ProjectNamespace`
  is the *storage* namespace — different concept.)
- **Steps:** `tool_registry.go` add `ToolNamespace` (name + description + member
  names), `ToolSearchConfig` (core set, namespaces, description budget,
  search_top_k, custom search name) + stateful `ToolSearchRegistry` (core +
  namespace headers + search meta-tool + MRU loaded tools with LRU budget
  eviction); `HarnessConfig.ToolSearch *ToolSearchConfig` + `WithToolSearch`;
  wire advertisement path.
- **E2E:** port namespace + tool-search tests; harness test asserting
  `tool_search` advertises core/header/meta, a search/open loads deferred tools
  into subsequent schemas, and real dispatch delegates to the inner registry.
- **Compat:** additive (nil ⇒ byte-identical). **Risks:** medium — deferred
  exposure changes which tools the model sees; guard default path.
- **Final evidence:** `go test .` (tool_registry) green; harness tool-search
  test green.

#### §D1 — `vector_store` module (CORRECTED contract)
- **Rust:** `vector_store.rs` (345 lines). The trait is **batch + filter-delete
  returning count**:
  - `add(records: Vec<VectorRecord>) -> Result<(), VectorStoreError>` — batch
    upsert (matched by `id`); reject dimension mismatch with
    `DimensionMismatch { expected, actual }`.
  - `search(query: Vec<f32>, top_k: usize, filter: Option<MetadataFilter>) ->
    Result<Vec<VectorHit>, VectorStoreError>` — filter applied AS PART OF the
    search, not post-truncation.
  - `delete(filter: MetadataFilter) -> Result<u64, VectorStoreError>` — delete
    every record whose metadata matches `filter`, **RETURNS THE COUNT removed**;
    empty filter = full clear.
  - `len() -> Result<u64>`; `is_empty() -> Result<bool>` (defaults to
    `len()==0`).
  - `VectorRecord { id, vector, metadata: Value }` (re-add = upsert);
    `VectorHit { id, score, metadata }` (HIGHER = more similar);
    `MetadataFilter { equals: Vec<(String, Value)> }` (AND-ed equality, `eq()`
    builder, `matches()`); `VectorStoreError { DimensionMismatch, Backend }`
    (`#[non_exhaustive]`, serde tagged). Blanket `Arc` impl. In-memory cosine
    test impl is the trait's executable contract.
- **Go gap:** No `VectorStore`. (The earlier plan's `Add(ctx, VectorRecord)
  error` + `Delete(ctx, id) error` was **wrong** — corrected above to batch
  `Add` + filter `Delete` returning count.)
- **Steps:** new `go/spore-core/vectorstore/vectorstore.go`: `VectorStore`
  interface (`Add(ctx, []VectorRecord) error`, `Search(ctx, []float32, int,
  *MetadataFilter) ([]VectorHit, error)`, `Delete(ctx, MetadataFilter) (uint64,
  error)`, `Len(ctx) (uint64, error)`, `IsEmpty(ctx) (bool, error)` defaulting
  to `Len()==0`); `VectorRecord`, `VectorHit`, `MetadataFilter` (`Equals
  [][2]any` or equivalent + `Eq()` builder + `Matches()`), `VectorStoreError`
  (`DimensionMismatch`/`Backend`); in-memory cosine `InMemoryVectorStore` test
  impl.
- **E2E:** port `vector_store.rs` tests (batch add/upsert, search top-k,
  filter pushdown, delete-by-filter returns count, full clear on empty filter,
  dimension mismatch).
- **Compat:** additive (new package). **Risks:** low (pure stdlib). The
  sqlite-vec backend is §H (not created).
- **Final evidence:** `go test ./vectorstore/` green.

### Tier 1 (continued) — Harness event / config / lifecycle gaps (§E, §F)

#### §E1 — `NodeAttr` on all harness stream events
- **Rust:** `harness/types.rs:116–298` — every one of the 18 `StreamEvent`
  variants carries `node: Option<NodeAttr>` (`#[serde(default,
  skip_serializing_if = "Option::is_none")]` ⇒ un-attributed event
  byte-identical). `NodeAttr { kind, agent_id: Option<String>, depth: u32,
  path: Vec<String> }` (`types.rs:426`). `StreamEvent::node_mut()` +
  `wrap_node_sink(parent, kind, agent_id)` (`types.rs:301,440`) stamps each
  event as it flows up through node-wrapping sinks: leaf sets
  `kind`/`agent_id`/seeds `path`; each combinator wrapper PREPENDS its kind to
  `path` and bumps `depth`. Lets a multi-agent UI route a frame to the emitting
  node.
- **Go gap:** `HarnessStreamEvent` (`harness.go:376`) has NO `NodeAttr` field
  on any variant; no `NodeAttr` type; no `wrap_node_sink` equivalent. The
  multi-agent attribution seam is entirely absent.
- **Steps:** `harness.go` add `NodeAttr` struct (`Kind string`, `AgentID
  string` optional, `Depth uint32`, `Path []string`); add `Node *NodeAttr
  \`json:"-"\`` to `HarnessStreamEvent` + emit `node` in every `MarshalJSON`
  arm with `omitempty` (nil ⇒ byte-identical); add `NodeMut()`/`Node()` accessors;
  port `wrapNodeSink` stamping in the combinator/leaf event path
  (`executor.go`/`strategy.go`).
- **E2E:** port `harness/types.rs` NodeAttr tests (un-attributed ⇒ no `node`
  key; leaf stamps `kind`/`agent_id`; combinator prepends `path` + bumps
  `depth`); fixture-replay byte-identical when no nodes.
- **Compat:** additive (nil ⇒ byte-identical). **Risks:** medium — must stamp
  through every combinator wrapper without double-stamping `kind`/`agent_id`
  (innermost leaf wins, set-once).
- **Final evidence:** `go test -run NodeAttr .` green; fixture-replay
  byte-identical.

#### §E2 — `ToolProgress` + progress sinks + streaming sandbox seam
- **Rust:** `harness/types.rs:237` — `StreamEvent::ToolProgress { call_id,
  name, preview, node }` (live output from a RUNNING tool, ~100ms tail
  snapshot, REPLACES previous preview, never persisted). `tool_registry.rs:142–
  200` — `ToolContext.progress: Option<ToolProgressSink>`, `ToolProgressUpdate
  { tool, call_id, preview }`, `ToolProgressSink = Arc<dyn Fn(ToolProgressUpdate)>`,
  `ToolContext::with_progress(..)`, `ToolContext::command_progress(call) ->
  Option<CommandProgressSink>`. `harness/types.rs` `CommandProgressSink =
  Arc<dyn Fn(String)>`. `SandboxProvider::execute_command_streaming(command,
  args, working_dir, timeout, progress: Option<CommandProgressSink>)`
  (`sandbox.rs:620–626`); the default `execute_command` delegates to it with
  `None`. The react loop wires the per-call progress emitter
  (`react.rs:170–177`).
- **Go gap:** No `ToolProgress` stream variant; no `ToolProgressSink`/
  `ToolProgressUpdate`; no `ToolContext.WithProgress`/`CommandProgress`; no
  `CommandProgressSink` type; no `ExecuteCommandStreaming` on `SandboxProvider`.
- **Steps:** `harness.go` add `HarnessStreamToolProgress` variant + marshal
  (`call_id`/`name`/`preview` + `node`); `tool_registry.go` add
  `ToolProgressUpdate`, `ToolProgressSink func(ToolProgressUpdate)`,
  `ToolContext.Progress` + `WithProgress` + `CommandProgress(call)
  CommandProgressSink`; `harness.go` `CommandProgressSink func(string)`;
  `SandboxProvider` add `ExecuteCommandStreaming(ctx, command, args, workingDir,
  timeout, progress CommandProgressSink)` (default `ExecuteCommand` delegates
  with nil); react loop wires the per-call emitter.
- **E2E:** port `sandbox.rs:1269 execute_command_streaming_emits_tail_previews`
  (tail snapshots replace, not append); harness test asserting `ToolProgress`
  events fire between `ToolCall` and `ToolResult` and are not persisted.
- **Compat:** additive (nil progress ⇒ byte-identical). **Risks:** medium —
  ~100ms wait-loop tick must not block the executor; preview is sanitized tail.
- **Final evidence:** `go test -run ToolProgress .` + `go test -run Streaming
  ./...` green.

#### §E3 — `tool_result.revised`
- **Rust:** `harness/types.rs:191` — `StreamEvent::ToolResult { call_id,
  is_error, content, revised: bool, node }`. `revised=true` when this event
  REPLACES a prior `ToolResult` for the same `call_id` because an `AfterTool`
  middleware rewrote it in place (`ContinueWithModification`, SC-9); the model
  side is patched via `ContextManager::replace_tool_result`, this is the
  symmetric stream-side signal. `#[serde(default)]` ⇒ old consumers
  deserialize.
- **Go gap:** `HarnessStreamToolResult` marshal (`harness.go:432`) emits only
  `kind/call_id/is_error/content` — no `revised`. (Go HAS the `replace_tool_result`
  SC-9 seam in the adapter; it just doesn't signal it on the stream.)
- **Steps:** `harness.go` add `Revised bool \`json:"-"\`` to
  `HarnessStreamEvent`; emit `revised` in the `ToolResult` marshal arm with
  `omitempty` (false ⇒ byte-identical); set it `true` when the AfterTool
  middleware rewrites a result in place.
- **E2E:** port the Rust SC-9 stream-side test (AfterTool rewrite emits a
  second `ToolResult` with `revised=true` for the same `call_id`).
- **Compat:** additive (false ⇒ byte-identical). **Risks:** low.
- **Final evidence:** `go test -run Revised .` green.

#### §E4 — `user_message.steered` + `SteeringSource`
- **Rust:** `harness/types.rs:223` — `StreamEvent::UserMessage { content,
  steered: bool, node }`. `steered=true` for a USER-authored steering message
  injected at a turn boundary (rendered as user input); `false` for the agent's
  `send_message` narration (rendered as agent output). `harness/config.rs:146–
  154,1284–1290` — `HarnessConfig.steering_source: Option<SteeringSource>` (`type
  SteeringSource = Arc<dyn Fn() -> Vec<String> + Send + Sync>`), a non-blocking
  queue drain the ReAct loop polls at every turn boundary
  (`react.rs:343–377`); injects each returned string as a USER message + emits
  `UserMessage(steered=true)`. `None` ⇒ byte-identical.
- **Go gap:** `HarnessStreamUserMessage` marshal (`harness.go:439`) emits only
  `kind/content` — no `steered`. No `SteeringSource` config field, no
  `WithSteeringSource`, no turn-boundary polling/injection.
- **Steps:** `harness.go` add `Steered bool \`json:"-"\`` + emit `steered`
  (`omitempty`, false ⇒ byte-identical); add `SteeringSource func() []string`
  + `HarnessConfig.SteeringSource` + `WithSteeringSource`; react loop polls it
  at the turn boundary (after current turn's tool calls, before next model
  call), injects each string as a user message + emits
  `UserMessage(steered=true)`; `send_message` narration stays `steered=false`.
- **E2E:** port `harness/tests.rs:424 steering_message_injects_at_turn_boundary`
  (queued message injected at turn boundary, emitted as `UserMessage` with
  `steered=true`, never cancels in-flight work); nil ⇒ byte-identical.
- **Compat:** additive (nil/false ⇒ byte-identical). **Risks:** low — source
  must be non-blocking.
- **Final evidence:** `go test -run Steering .` green.

#### §E5 — tool-call nudge
- **Rust:** `harness/config.rs:139–145,1275–1280` — `HarnessConfig.tool_call_nudge:
  bool` (default false) + `HarnessBuilder::tool_call_nudge(on)`. Gates TWO
  distinct nudges in `react.rs:741–823`:
  1. **Structural** (line 759): `tool_call_nudge && execute_step &&
     tools_advertised && tools_called_this_window==0 && output_schema.is_none()`
     — combined with the phrase-based prose detection, flips the
     `prompt_tool_call_flag` and nudges (catches markdown command fences / bare
     narration / empty content that the phrase list misses).
  2. **Behavioral** (line 799): an EXECUTE-step terminal answer that called
     ZERO tools while tools were advertised → nudges to actually use a tool
     before reporting; bounded per window (`MAX_TOOL_CALL_NUDGES`).
  Both off (default) ⇒ byte-identical. Distinct from the #111 phrase-based
  `prompt_tool_call_flag` escalation (which Go already has).
- **Go gap:** No `ToolCallNudge` config field/setter; neither the structural
  nor the behavioral nudge branch exists. Go has only the #111 phrase-based
  path (`harness.go:5276`).
- **Steps:** `harness.go` add `ToolCallNudge bool` + `WithToolCallNudge`;
  react loop add the structural branch (gated on `ToolCallNudge`, flips
  `PromptToolCallFlag`) and the behavioral branch (execute-step zero-tools
  terminal, bounded per window); port `MAX_TOOL_CALL_NUDGES`.
- **E2E:** port the Rust nudge tests (structural flips flag on bare
  narration/fence; behavioral nudges zero-tool execute terminal; default off ⇒
  byte-identical).
- **Compat:** additive (false ⇒ byte-identical). **Risks:** low-medium —
  bounded per window to avoid loops.
- **Final evidence:** `go test -run Nudge .` green.

#### §E6 — plan-empty retries
- **Rust:** `harness/config.rs:134–138,1266–1271` —
  `HarnessConfig.plan_empty_max_retries: u32` (default 0 = halt on first empty
  plan; >0 = re-ask the planner that many times with the empty plan as message
  content before `EmptyPlan`) + `HarnessBuilder::plan_empty_max_retries(retries)`.
  `harness/mod.rs:316,822,1022` — a one-shot FORCED output schema for the
  plan-empty re-ask sub-run (the planner is offered NO tools).
- **Go gap:** `HaltEmptyPlan` exists (`harness.go:2603`) but no
  `PlanEmptyMaxRetries` config field/setter and no re-ask loop.
- **Steps:** `harness.go` add `PlanEmptyMaxRetries *uint32` (pointer sentinel,
  nil ⇒ 0 default, matching the house idiom) + `WithPlanEmptyMaxRetries`;
  plan-execute phase re-ask loop (re-run the planner with the empty plan as
  message content + a one-shot forced output schema + no tools, up to N times
  before `HaltEmptyPlan`).
- **E2E:** port the Rust plan-empty re-ask tests (0 ⇒ immediate `EmptyPlan`;
  N>0 ⇒ re-asks N times then `EmptyPlan`; a non-empty plan on a retry
  proceeds).
- **Compat:** additive (nil/0 ⇒ byte-identical). **Risks:** medium — re-ask
  must not double-charge budget incorrectly; one-shot forced schema + no-tools
  sub-run.
- **Final evidence:** `go test -run PlanEmpty .` green.

#### §E7 — `PlanRecorded` control flow
- **Rust:** `harness/types.rs:548–553` — `HarnessSignal::PlanRecorded` (5th
  variant): the planner recorded its plan via `set_plan` and is done planning
  → END the PlanExecute plan phase (plan-phase analogue of `plan_completed`),
  consumed inside PlanExecute, never ends the whole run. `harness/plan_execute.rs:587`
  — the plan phase matches `RunResult::Escalate { signal: PlanRecorded, .. }`,
  threads the reads it gathered, then proceeds with an EMPTY `plan_output`
  (the SC-28 fallback loads the runnable plan from the durable `task_list`
  store). Tied to §A1 (`set_plan` tool).
- **Go gap:** `HarnessSignal` has only 4 variants
  (`harness.go:645–648`): `EnterPlanMode`/`ExitPlanMode`/`SwitchMode`/`Abort` —
  no `PlanRecorded`. No plan-execute consumption of a `set_plan` terminal.
- **Steps:** `harness.go` add `SignalPlanRecorded HarnessSignalKind =
  "plan_recorded"`; `plan_execute` match arm threads reads + proceeds with
  empty `plan_output` (SC-28 fallback loads from `task_list` store). Depends on
  §A1.
- **E2E:** port the Rust `set_plan`→`PlanRecorded`→execute test (planner uses
  `set_plan`, plan phase ends, execute reads the durable task list).
- **Compat:** additive. **Risks:** low (gated on §A1).
- **Final evidence:** `go test -run PlanRecorded .` green.

#### §E8 — narration refund
- **Rust:** `harness/react.rs:334–340,377,589–601` — `send_message`-ONLY turns
  are pure narration (#81) and must NOT consume the productive-turn budget.
  `narration_refund: u32` accumulates; the turn cap becomes
  `effective_turn_cap.saturating_add(narration_refund)`; a `send_message`-only
  turn (terminal, only tool called was `send_message`) refunds one turn
  (`narration_refund += 1`), capped at `effective_turn_cap` so a pure-narration
  loop still terminates (~2× budget).
- **Go gap:** Go emits `HarnessStreamUserMessage` for `send_message`
  (`harness.go:5684`) but has NO narration refund — `send_message`-only turns
  consume the turn budget against the hard `effectiveTurnCap` gate
  (`harness.go:5050`).
- **Steps:** `harness.go` react loop add `narrationRefund` accumulator;
  `effectiveTurnCap` becomes `effectiveTurnCap + narrationRefund`; detect
  `send_message`-only terminal turns (only tool called was `send_message`) and
  refund one turn, capped at `effectiveTurnCap`.
- **E2E:** port the Rust narration-refund test (a `send_message`-only turn
  does not advance the productive-turn count; a pure-narration loop still
  terminates at ~2× budget).
- **Compat:** behavioral change (turn accounting) — but only widens the cap for
  narration, never narrows existing behavior; verify fixture-replay where
  `send_message` appears stays consistent. **Risks:** medium — must not let
  narration loop forever.
- **Final evidence:** `go test -run Narration .` green.

#### §E9 — `TurnEnd.usage`
- **Rust:** `harness/types.rs:124` + `react.rs:705` — `StreamEvent::TurnEnd {
  turn, usage: TokenUsage, node }` carries this turn's token usage so a live
  consumer updates the token chip PER-TURN (not only end-of-run). `#[serde(default)]`
  ⇒ old consumers deserialize.
- **Go gap:** `HarnessStreamTurnEnd` marshal (`harness.go:414`) emits only
  `kind/turn` — no `usage`.
- **Steps:** `harness.go` add `Usage TokenUsage \`json:"-"\`` (or reuse) +
  emit `usage` in the `TurnEnd` marshal arm with `omitempty`/default (zero ⇒
  byte-identical consideration — verify Rust's `#[serde(default)]` shape: the
  field IS emitted as `{}` for zero usage in Rust; match exactly); set it from
  the turn's `result.Usage`.
- **E2E:** port the Rust per-turn-usage test (TurnEnd carries the turn's
  usage); fixture-replay byte-check on TurnEnd events.
- **Compat:** additive but **wire-shape** — `usage` appears on `turn_end`;
  verify against Rust's exact emission (default TokenUsage serializes to a
  zero-fields object). **Risks:** low-medium — fixture byte-parity on
  `turn_end`.
- **Final evidence:** `go test -run TurnEnd .` green; fixture-replay
  byte-identical.

#### §E10 — `RepeatedToolCall` breaker (success twin) + `HaltRepeatedToolCall`
- **Rust:** `harness/config.rs:172–176` — the SAME `N`/`2N` `error_loop_threshold`
  also drives the repeated-identical-call breaker: the model calling one tool
  with identical args and SUCCEEDING `N` times in a row is nudged once, and at
  `2N` the run hard-stops with `HaltReason::RepeatedToolCall { tool, count }`
  (`types.rs:1583`, `react.rs:1710`). `0` disables both breakers.
- **Go gap:** Go implements the error breaker (`ToolErrorLoopDetected`/
  `Broken` + `HaltToolErrorLoop`) but has NO success twin: no
  `HaltRepeatedToolCall`, no consecutive-success streak tracking.
- **Steps:** `harness.go` add `HaltRepeatedToolCall HaltReasonKind =
  "repeated_tool_call"`; react loop track per-tool consecutive identical-arg
  SUCCESS streak (any different call/arg resets); at N nudge once, at 2N
  hard-stop with `HaltRepeatedToolCall`. Reuse `ErrorLoopThreshold` (already
  present).
- **E2E:** port `harness/tests.rs:12931 repeated_identical_successful_calls_hard_stop`
  (default N=3 → stop at 6 with `RepeatedToolCall`).
- **Compat:** additive (only fires on a no-progress loop not currently
  halted). **Risks:** low-medium — must not false-trip on legitimate repeated
  reads.
- **Final evidence:** `go test -run RepeatedToolCall .` green.

#### §F1 — `ChunkProvider` on `HarnessConfig` (#79 wiring)
- **Rust:** `harness/config.rs:228,434,538,1051–1063,1486` —
  `HarnessConfig.chunk_provider: Arc<dyn ChunkProvider>` +
  `HarnessBuilder::chunk_provider(..)`. `harness/self_verifying.rs:139` loads
  the role-evaluator chunk from `self.config.chunk_provider.load()`.
- **Go gap:** `HarnessConfig` has NO `ChunkProvider` field; `self_verifying.go:28–91`
  hardcodes `RoleEvaluatorChunk` as a constant because "HarnessConfig carries
  no chunk-provider field." Go consumers cannot plug conditional prompt chunks
  into the harness.
- **Steps:** `harness.go` add a structural chunk-provider adapter (root cannot
  import `promptassembly` without a cycle), default empty provider +
  `WithChunkProvider`; `self_verifying.go` prepends the role-evaluator chunk
  only when the provider supplies it (presence-only, no built-in fallback).
- **E2E:** port `harness/tests.rs:1459 default_chunk_provider_empty_and_chunks_setter_round_trips`
  + a self-verifying test that the chunk comes from the provider.
- **Compat:** additive (default empty ⇒ byte-identical). **Risks:** low —
  avoid the import cycle (root can't import `promptchunkregistry`; use
  `promptassembly.ChunkProvider`).
- **Final evidence:** `go test -run ChunkProvider .` green.

### Tier 1 (continued) — Documentation (§G)

#### §G1 — Go documentation
- **Rust:** `docs/rust/` has 4 substantive pages: `quickstart.md` (78 lines),
  `building-a-tool.md` (111), `conversation.md` (89), `custom-harness.md` (76)
  — each a Go-portable code walkthrough pointing at the matching
  `examples/rust/0N-*` example.
- **Go gap:** `docs/go/` has only `quickstart.md` (27 lines, a stub that
  points at the shared guide + Rust quickstart). The Go stub is **not**
  documented-usage parity.
- **Steps:** write Go equivalents under `docs/go/`:
  1. `quickstart.md` — a real Go walkthrough (not a stub): `OllamaModelInterface`,
     `observability.NewConversationalHarness`, `sporecore.SimpleTask`, run +
     match `Success`; points at `examples/go/01-hello-agent`.
  2. `building-a-tool.md` — catalogue tools (`StandardTools.CodingSet`) +
     custom `DefineTool`; points at `examples/go/03-tool-use`.
  3. `conversation.md` — thread `SessionState` across turns; points at
     `examples/go/02-conversational-repl`.
  4. `custom-harness.md` — `HarnessBuilder` override-one-slot chaining +
     `StorageProvider`/`InMemoryStorageProvider`; points at the
     `harness-builder` reference.
  Each mirrors its Rust counterpart's structure but uses Go idioms
  (`context.Context`, consumer-side interfaces, `DefineTool`).
- **E2E:** each doc's code snippet should compile against the current Go API
  (verify by extracting into a throwaway `_test.go` or by building the cited
  example).
- **Compat:** docs-only. **Risks:** low — keep snippets in sync with the API.
- **Final evidence:** `docs/go/{quickstart,building-a-tool,conversation,custom-harness}.md`
  present; cited examples build.

## 6. Documented known gaps (NOT created — captain scope)

Per the captain scope decision, the following Rust subsystems are **out of
scope as deliverables** and are recorded as known Go gaps. No Go code is
created for them.

### §H1 — `spore-mcp` (MCP client)
- **Rust:** `spore-mcp/src/lib.rs` (511 lines) — `McpManager`,
  `McpServerConfig` (`Stdio`/`StreamableHttp`), `McpTransport`, `McpError`,
  `tools_for`/`tools_for_namespaced` (returns `StandardTool` per remote tool),
  `probe`, `disconnect`. Quarantines the `rmcp` Rust MCP SDK.
- **Go:** not created. Document in `docs/go/quickstart.md` (or a
  `docs/go/known-gaps.md`) that MCP tool integration is a known gap; the
  `ToolNamespace` seam (§C2) is the future foundation.

### §H2 — `spore-sqlvec` (sqlite-vec `VectorStore` backend)
- **Rust:** `spore-sqlvec/src/lib.rs` (580 lines) — `SqliteVecStore` + builder,
  `vec0` virtual table, indexed-key metadata columns, KNN pushdown.
  Quarantines `rusqlite` + the `sqlite-vec` C extension.
- **Go:** not created. The core `VectorStore` trait + in-memory cosine impl
  (§D1) IS delivered; the sqlite-vec backend is a known gap. Document it.

### §H3 — `sandbox_petri` (feature-gated `petri` microVM sandbox)
- **Rust:** `sandbox_petri.rs` (1564 lines, feature `petri`) —
  `PetriSandboxProvider`, `reap_orphans`, `reap_session`. Both Rust (without
  `petri`) and Go return `DisallowedCommand` for `IsolationMode::MicroVm`.
- **Go:** not created. Document that microVM isolation is a known gap; the
  `WorkspaceScopedSandbox` (process/filesystem isolation) is the delivered
  sandbox.

A short `docs/go/known-gaps.md` (or a section in `docs/go/quickstart.md`) will
record §H1–§H3 so consumers know the boundary.

## 7. Implementation order (post-approval)

1. **§A1–§A3** (tools/catalogue) — smallest; unblocks the catalogue name-set
   drift guard. §A1 unblocks §E7.
2. **§B2** (image helpers) — tiny, pure stdlib.
3. **§B1** (embedding, corrected batch+EmbedOne) + **§D1** (vector_store,
   corrected batch+filter-delete-count) — pure stdlib, paired.
4. **§B3** (ModelParams extended sampling + OpenAI wire) — additive.
5. **§B4** (Ollama thinking, #148) — medium; verify fixture-replay
   byte-identical when unset.
6. **§C1** (tool_call_repair + harness wiring) — medium; only core-loop
   behavioral change besides §E.
7. **§C2** (ToolNamespace + ToolSearch) — medium.
8. **§E1** (NodeAttr) — medium; stamps through every combinator.
9. **§E2** (ToolProgress + streaming sandbox) — medium.
10. **§E3/§E4/§E9** (revised / steered / TurnEnd.usage) — small wire-shape
    additions; verify fixture byte-parity.
11. **§E5/§E6/§E8** (nudge / plan-empty retries / narration refund) —
    behavioral; TDD first.
12. **§E7** (PlanRecorded, after §A1) + **§E10** (RepeatedToolCall breaker).
13. **§F1** (ChunkProvider wiring).
14. **§G1** (Go docs) + **§H** (known-gaps doc) — last, against the finished
    API.

Each step: add/strengthen Go tests **before** implementation where a missing
behavior can be expressed as a regression (TDD); verify representative
end-to-end scenarios; run the full Go test/lint/format/build suite after each.

## 8. Validation plan

- **Unit:** `cd go/spore-core && go test ./...` (baseline green — recorded).
  `cd go/spore-eval && go test ./...`.
- **Race:** `go test -race ./...` (CI gate). §4 already shows a race-clean
  slice.
- **Lint/format:** `gofmt -l .` (empty); `go vet ./...`.
- **Build:** `go build ./...` for all three modules + `examples/go/` (via
  `go.work`).
- **Fixture replay:** every existing `*_fixture_replay_test.go` stays
  byte-identical (the cross-language fixtures are the parity oracle). New gaps
  get new fixture-replay tests where Rust has them.
- **Cross-language name-set guard:** new test asserting Go `CodingSet`/
  `FullSet`/`ReadonlySet` names exactly equal the Rust sets.
- **Lifecycle/concurrency:** §4 evidence re-run after changes; **§4.1
  context-cancellation tests added first** (post-approval, before any gap
  work); new §E2/§E4 tests added.
  tests added.

## 9. Compatibility constraints

- **Backward compatibility:** every Tier 1 gap is additive (new types/tools/
  fields with nil/zero/false defaults ⇒ byte-identical serialization when
  unset, mirroring Rust `skip_serializing_if` / Go `omitempty`). No existing
  public Go API signature changes.
- **Wire shape:** §E3/§E4/§E9 add fields to existing stream events — verify
  each matches Rust's exact `#[serde(default)]` emission (present-but-default
  vs omitted) so fixture-replay stays byte-identical. §E8 (narration refund)
  is a turn-accounting behavioral change that only widens the cap for
  narration — verify no fixture narrows.
- **No Rust edits:** nothing under `rust/`, `docs/rust/`, `examples/rust/`, or
  Rust config/tests is modified. No Rust change is required by any gap
  (confirmed during this audit).
- **No shared-file edits:** edits confined to `go/`, `docs/go/`,
  `examples/go/`, Go-specific tests/config. `AGENTS.md`, top-level docs, CI,
  generated files, shared fixtures untouched without firstmate coordination.

## 10. Risks & open questions

- **§E1 (NodeAttr)** — highest event-surface risk: must stamp through every
  combinator wrapper without double-stamping (innermost leaf wins, set-once).
- **§E2 (ToolProgress)** — ~100ms wait-loop tick must not block the executor.
- **§B4 (Ollama thinking)** — streaming index disjointness; fixture-replay
  byte-identical when unset.
- **§C1 (tool_call_repair)** — only core-loop dispatch change; one-retry
  semantics, no double budget charge.
- **§E8 (narration refund)** — behavioral turn-accounting change; verify
  fixtures.
- **§E9 (TurnEnd.usage)** — wire-shape; verify exact default emission.
- **§H (MCP/sqlvec/petri)** — out of scope per captain; documented only.
- **No evidenced Rust change required** by any in-scope gap.

---

needs-decision: revised plan-review checkpoint — §B1 expanded to full OpenAI embedding behavior (retry/backoff/retry-after, bearer auth, error-envelope mapping, sort-by-index, default dimensions, from-env validation, timeout/transport mapping, 9 mocked-HTTP assertions, Go interface default-EmbedOne caveat + shared helper); §4.1 adds explicit context-cancellation executable tests (cancelled model/tool call stops + reaps, not only timeout); §C1 adds MaxRepairAttempts setter/default validation (default 1, ignored without a repair provider, pointer-sentinel/effective-reader idiom); captain scope decision incorporated (core+eval+embedding+vector+docs in scope; MCP/sqlvec/petri documented as known gaps, not created); no implementation started

## 11. Final implementation matrix

| Item | Final Go status |
|---|---|
| §4.1 cancellation/process cleanup | Complete: cancellation tests and process-group reap parity added. |
| §A tools/catalogue | Complete: `set_plan`, `create_dir`, and catalogue exposure added. |
| §B1 embeddings | Complete: `embedding` module with batch `Embed`, `EmbedOne`, Ollama/OpenAI providers. |
| §B2 image helpers | Complete. |
| §B3 extended sampling | Complete: model params and OpenAI wire support. |
| §B4 Ollama thinking/reasoning | Complete. |
| §C1 tool-call repair | Complete: provider/config/default-attempt behavior and harness wiring. |
| §C2 tool search | Complete: full stateful `ToolNamespace`, `ToolSearchConfig`, `ToolSearchRegistry`, custom search name, `SearchTopK`, namespace opening, MRU loaded state, description-budget LRU eviction, and harness opt-in wiring. |
| §D1 vector store | Complete: `vectorstore` package and tests. |
| §E1 `NodeAttr` | Complete: event field, JSON emission, and nested stamping wrappers. |
| §E2 `ToolProgress` | Complete: stream event, progress sink, optional streaming sandbox seam. |
| §E3/§E4/§E7/§E9 | Complete: revised/steered/steering source, `SignalPlanRecorded`, turn-end usage. |
| §E5 tool-call nudge | Complete: default-off structural/behavioral execute-step nudge. |
| §E6 empty-plan retries | Complete: default-0 `PlanEmptyMaxRetries`, forced-schema/no-tools retry subtree, corrective nudge, inherited stream sink, and cumulative usage accounting for discarded empty attempts. |
| §E8 narration refund | Complete: ReAct window-gate behavior ported and tested. |
| §E10 repeated tool-call breaker | Complete: `HaltRepeatedToolCall`. |
| §F1 chunk provider | Complete: harness-level structural provider adapter and SelfVerifying presence-only role chunk override with malformed-provider fallback/no-panic behavior. |
| §G1 Go docs | Complete: quickstart, building tools, conversation/events, custom harness. |
| §H known gaps | Complete: MCP, sqlite-vec, and Petri documented as known gaps, not implemented. |

Final validation is recorded in the branch commit history and status file; the final PR notes include the latest full Go build/vet/test/race results.
