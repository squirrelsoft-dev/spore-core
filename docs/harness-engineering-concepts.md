# Harness Engineering Concepts

> This document is the canonical language-agnostic specification for spore-core. All component interfaces, rules, identity models, loop strategies, and architectural decisions are recorded here. GitHub issues reference this document; this document supersedes any earlier informal description.

---

## The Foundational Equation

**Agent = Model + Harness**

The model does the reasoning. The harness is everything else: tools, sandbox, memory, context management, sensors, guides, middleware, and the loop that drives them all. Most agent failures are not model failures — they are configuration, context, and environment failures. The harness is where reliability lives.

---

## Design Principles

**Inversion of Control**: The harness is a runtime container. It accepts pluggable components as dependencies, wires them together, and drives execution through a lifecycle of states and hooks. Components do not know about each other — the harness does.

**The Agent Is One Turn**: The agent executes one model call and returns a result. The harness drives the loop. This separation is load-bearing — it is what makes loop strategies, middleware, and termination policy composable.

**Stateless Between Runs**: The harness has no internal state between `run()` calls. Everything it needs comes in via `HarnessRunOptions` or `PausedState`. Everything it produces goes out via `RunResult`. This makes the harness deployable as a CLI, REST API, library, queue worker, or subprocess without modification.

**Stable Prefixes, Cacheable Blocks**: Provider prefix caching is a first-class concern. The context window is assembled in three blocks — Static (permanent cache hit), PerSession (cached within a session), PerTurn (never cached). Nothing in Block 1 or Block 2 changes after it is set.

---

## Identity Model

Four levels of identity. Session is the primary handle the caller holds.

```
Project   (optional — groups sessions, owns semantic memory and guides)
  └── Session  (persistent container — the workspace, thread, or conversation)
        └── Task  (one agentic run — from instruction to completion)
              └── Turn  (one model call + all tool dispatches that flow from it)
                    └── ToolDispatch  (one tool execution — identified by composite key)
```

```
ProjectId     String — optional, caller-assigned
SessionId     String — primary caller handle, ThreadId equivalent
                       caller-assigned or harness-generated
TaskId        String — one agentic run within a session
                       harness-generated, optionally tagged by caller
AgentId       String — agent configuration name, caller-assigned
                       e.g. "initializer-agent", "coding-agent"
TurnNumber    u32    — monotonic counter within a task, not a separate ID
DispatchIndex u32    — position of a tool call within a Turn, composite key only
```

Full address of any tool dispatch: `(SessionId, TaskId, TurnNumber, DispatchIndex)`

**Memory scoping**:
- EpisodicMemory → scoped to SessionId ("what happened in this workspace")
- SemanticMemory → scoped to ProjectId, falls back to global ("what we've learned across all workspaces")

**Observability rollup**: ToolDispatch → Turn → Task → Session → Project

---

## Loop Strategies

The harness drives the loop. The agent executes one turn. The loop strategy determines the outer structure.

### ReAct
Foundational pattern. Thought/Action/Observation interleaved in context. Loop until model returns a FinalResponse with no tool calls. Termination evaluated by TerminationPolicy.

### PlanExecute
Two phases. Planner model produces a plan artifact (once). Executor model implements steps from the plan (loop). Optionally uses a different model for planning vs execution.

### Ralph
Continuation loop. Intercepts the model's exit attempt, resets context window, reloads state from filesystem (progress file, git log, feature list), continues until external completion check passes. The filesystem is what makes multi-context-window work possible.

### SelfVerifying
Loop-within-a-loop. Build phase runs until agent claims done. Evaluate phase runs a **separate evaluator harness** with:
- Read-only sandbox — no Write or Edit tools
- Fresh session context — never shares a session with the build harness
- Explicit evaluator role chunk — "You are a fresh evaluator. You did not write the code you are reviewing."

This is the Default-FAIL contract. The evaluator cannot be biased by having watched the work happen. If the evaluator finds problems, findings are injected into the build context and the build loop continues.

### HillClimbing
Iterative optimization loop. Establishes a baseline metric, proposes changes, evaluates the metric after each iteration, keeps or reverts based on improvement. Generalizes the autoresearch pattern.

```
Termination conditions:
  max_stagnation: Option<u32>   — halt after N consecutive non-improvements
                                   None = run until externally stopped
  revert_on_no_improvement: bool — git reset if metric did not improve
  min_improvement_delta: f64    — minimum improvement to count as progress

Results log: written to .spore/results/{task_id}.tsv by the harness, not the agent
```

---

## Error Propagation

Three layers. Layer 1 cannot be overridden.

### Layer 1: Always-Halt (hardcoded)
```
SandboxViolation::PathEscape        — workspace root escape attempt
SandboxViolation::NetworkViolation  — disallowed network call
BudgetExceeded                      — any budget limit hit
ModelError::ContextLimitExceeded    — window full and compaction failed
```

### Layer 2: Middleware-Routable (with defaults)
```
SandboxViolation::PathDenied           → default: Recoverable
SandboxViolation::ReadOnlyViolation    → default: Recoverable
ToolExecutionError (recoverable: true) → default: Recoverable
ToolExecutionError (recoverable: false)→ default: Halt
DispatchError::UnregisteredTool        → default: Halt
DispatchError::SchemaValidationFailed  → default: Recoverable
SensorOutcome::Halt                    → default: SurfaceToHuman
MiddlewareDecision::Halt               → default: Halt
AgentError::EmptyResponse              → default: Retry once, then Halt
```

### Layer 3: Default Routing
Recoverable errors are formatted as failed tool results and appended to context. Error messages are deliberately informative — they tell the agent what went wrong and implicitly guide the correct approach.

```
ErrorRoute {
  Recoverable { agent_message: String },
  Halt,
  SurfaceToHuman,
}
```

Retried turns count against the turn budget.

---

## Cache Architecture

Providers use prefix caching. Byte-for-byte identity of the prefix is required for a cache hit. The context window is assembled in three blocks:

```
Block 1 — Static (computed ONCE at harness startup)
  Composed by PromptChunkRegistry from: role chunk, mode chunk,
  capability chunks, skill chunks, convention guides
  → permanent cache hit for the life of the harness instance
  → cache_provider inserts breakpoint after this block

Block 2 — Per-Session (stable within a session)
  task instruction, environment/directory map,
  prior session state, operational instructions
  → cache_provider inserts breakpoint after this block

Block 3 — Per-Turn (never cached)
  budget warnings, ephemeral skill injections
```

Message history: cache breakpoint placed after the last tool result, before the new user prompt.

**What breaks the cache**:
- Timestamps in Block 1 or Block 2
- Non-deterministic guide rendering (reformat at storage time, not assembly time)
- Unsorted tool schemas (always sort by name)
- Dynamic directory maps in Block 1 (belongs in Block 2)
- Whitespace normalization variance

---

## Human-in-the-Loop

HITL is the intersection of MiddlewareChain (`SurfaceToHuman` decision), RunResult (`WaitingForHuman` variant), and the `resume()` method on Harness.

**The harness is stateless between pause and resume.** The caller owns and persists `PausedState`. No timeout mechanism in the harness — that is the caller's concern.

Three interaction types: ToolApproval, Clarification, Review.

RiskLevel auto-derived from ToolAnnotations:
```
read_only: true                                     → Low
non-destructive, idempotent, no external systems    → Medium
destructive OR external systems                     → High
destructive AND external systems                    → Critical
```

**SubagentTool HITL**: when a subagent's PermissionMiddleware triggers `SurfaceToHuman`, the child harness returns `WaitingForHuman`. The `SubagentTool` surfaces this as `ToolOutput::WaitingForHuman`. The parent harness assembles a combined `PausedState` with a nested `ChildPausedState` and returns `WaitingForHuman` to the caller.

**Depth rule**: subagents cannot spawn their own subagents. `ChildPausedState` has no `child_state` field — the type system enforces one-level depth. `SubagentTool::new()` validates this at construction time.

---

## Multi-Agent Patterns (v1)

### Sequential Multi-Agent
Two calls to `harness.run()` with the same `SessionId`. The second call inherits session state via `MemoryProvider` and shared filesystem. Covers: initializer + coding agent, any handoff pattern.

### SubagentTool
A child `Harness` wrapped as a `Tool`. Parent agent calls it via `ToolRegistry`. Child runs to completion or HITL pause and returns a result. Construction validates that the child harness has no `SubagentTool` registered.

```
ContextSharing {
  Isolated,                              // fresh SessionId
  SharedSession { session_id },          // same session, shared episodic memory
  SummaryHandoff { summary },            // fresh session + injected summary
}
```

### Filesystem Coordination
Multiple harness instances sharing a workspace root. Git handles collision resolution. No concurrent harness abstraction required in v1.

### Post-v1: ParallelHarness
Task queue + N concurrent Harness instances + result aggregator. Deferred until single-agent harness is stable.

---

## Streaming

**UX streaming** (in scope for v1): `ModelInterface.call_streaming()` fires a `StreamEvent` callback for each token as it arrives. The harness loop operates on complete `TurnResult` values — streaming is a delivery mechanism for the UI, not an operational concern.

**Operational streaming** (deferred to post-v1): early tool dispatch, mid-stream interruption, token-by-token observability.

**Reasoning tokens**: `ThinkingDelta` is a distinct `StreamEvent` type. Thinking blocks (`ContentBlock::Thinking`) must be preserved in message history and passed back in subsequent requests. `CompactionPreserveHints.keep_thinking_blocks` defaults to `true` — never compact active reasoning blocks.

**Component boundary**: only `ModelInterface` and `Agent` ever touch `StreamEvent`. Everything else works on complete values.

---

## The .spore/ Directory Layout

```
{workspace_root}/
  .spore/
    progress.json              ← commit to git (handoff artifact)
    feature_list.json          ← commit to git (initializer writes this)
    sessions/
      {session_id}/
        state.json             ← gitignore (pause/resume state)
        trace.jsonl            ← gitignore (observability export)
    results/
      {task_id}.tsv            ← commit to git (HillClimbing improvement record)
    offload/
      {call_id}.txt            ← gitignore (large tool output)
    memory/
      episodic.db              ← gitignore (SQLite episodic memory)
      semantic.db              ← gitignore (SQLite semantic memory)
```

Progress file format (JSON, versioned):
```json
{
  "schema_version": "1.0",
  "session_id": "sess_abc123",
  "task_history": [
    {
      "task_id": "task_001",
      "completed_at": "2026-05-16T10:00:00Z",
      "summary": "Implemented login form. Committed abc123.",
      "outcome": "success"
    }
  ],
  "current_state": "Working on session management.",
  "next_suggested": "Complete SessionManager.store() method."
}
```

---

## The Harness Components

---

### Model Interface

**Purpose**: Boundary between the harness and the underlying LLM. Normalizes provider differences so nothing else in the harness knows or cares which model it is talking to.

**Responsibilities**:
- Send a context (messages + tools + parameters) to a model, return a response
- Handle streaming vs. non-streaming transparently
- Report token usage on every call
- Surface provider errors as typed harness errors
- Count tokens on request (for pre-call context size estimation)

**Rules**:
- Never called directly by tools, memory, or guides — only by the agent
- Token counts reported on every call, not optionally
- Provider-specific parameters injected by the harness, not hardcoded
- Token-budget overrun is a harness error, not a model error
- Provider-level transient errors (rate limits, timeouts) handled internally with exponential backoff — callers never see them

---

### Agent (One Turn)

**Purpose**: Execute a single turn — call the model with the current context, return either a tool call request or a final response.

**Responsibilities**:
- Accept a context and an optional stream handler, return a turn result
- Identify whether the model wants to call tools or is done

**Rules**:
- One turn = one model call, period
- Does not manage context — receives it fully assembled
- Does not execute tools — returns tool call requests to the harness
- Does not decide termination — returns a result type the harness evaluates
- Does not retry — surface errors upward
- Accumulates streaming events internally and returns a complete TurnResult

---

### Harness (Runtime / Loop)

**Purpose**: Drive the agent loop. Owns the execution lifecycle, wires all components together, decides when to keep going and when to stop.

**Responsibilities**:
- Assemble context before each turn (via ContextManager)
- Call the agent for one turn
- Dispatch tool calls to the ToolRegistry
- Evaluate the TerminationPolicy after each turn
- Fire middleware lifecycle hooks
- Track iteration count, token spend, and elapsed time
- Pause and resume for human-in-the-loop interactions

**Rules**:
- The harness owns the loop — nothing else does
- Termination evaluated against external state, not the model's self-assessment alone
- Budget overruns terminate with an explicit typed reason
- A turn that produces neither a tool call nor a final response is an error condition
- The harness never executes tools directly — dispatches to ToolRegistry
- Component injection happens at construction time
- Stateless between pause and resume — caller owns PausedState
- `WaitingForHuman` returns immediately — the harness does not block waiting for a human

**Loop execution order** (per turn):
```
1. middleware.fire_before_turn(context)
2. context_manager.assemble(session, guides, memory, tools)
3. agent.turn(context, on_stream) -> TurnResult
4. match TurnResult:
   ToolCallRequested(calls) =>
     Layer 1 check (always-halt violations)
     middleware.fire_before_tool(calls)
     for call in calls:
       sandbox.validate(call)
       tool_registry.dispatch(call, sandbox)
       if ToolOutput::WaitingForHuman: assemble PausedState, return WaitingForHuman
       sensor_chain.fire(PostTool, result)
     middleware.fire_after_tool(calls, results)
     context_manager.append_all(results)
     compact if should_compact()
     goto 1
   FinalResponse(response) =>
     middleware.fire_before_completion(response)
     sensor_chain.fire(PostTurn, response)
     termination_policy.evaluate(state)
5. observability.flush_turn(span)
```

---

### ToolRegistry

**Purpose**: Maintains available tools and dispatches tool calls from the agent.

**Responsibilities**:
- Register tools with their schemas
- Return the active ToolSet for a given task phase
- Dispatch tool calls to correct implementations
- Pass SandboxProvider to every tool on dispatch
- Return typed results or errors

**Rules**:
- Tools always dispatched via registry — never directly
- Active ToolSet can change between turns as TaskPhase advances
- Unregistered tool call = harness error
- Schemas validated at registration, not at call time
- No retry logic — that is middleware's job
- `dispatch_all` executes concurrently where `read_only: true`; sequential for `destructive` or `open_world`
- `has_subagent_tools()` method used by `SubagentTool::new()` to enforce no-nested-subagents rule

---

### Tool (Individual)

**Purpose**: Execute one specific action against the real world.

**Responsibilities**:
- Accept a SandboxProvider and validated parameters
- Execute the action
- Return a typed result or typed error

**Rules**:
- Every tool accepts a SandboxProvider — no tool touches the environment directly
- Invalid parameters return a typed error, never panic
- Tools are stateless — no references to session state
- A tool must not call other tools — composition happens at the harness level
- Large outputs truncated head+tail, full output offloaded via SandboxProvider

**ToolOutput variants**:
```
Success { content: String, truncated: bool }
Error { message: String, recoverable: bool }
WaitingForHuman { child_state: ChildPausedState, request: HumanRequest }
  — returned ONLY by SubagentTool when child harness pauses for HITL
```

**SubagentTool**: wraps a child Harness as a Tool. Subagents cannot spawn their own subagents — enforced at construction time via `has_subagent_tools()` check and encoded in the type system via `ChildPausedState` having no `child_state` field.

---

### SandboxProvider

**Purpose**: The capability object that enforces the execution boundary. All tools receive it.

**Responsibilities**:
- Validate and resolve file paths against the workspace root
- Execute shell commands within the defined isolation mode
- Enforce read/write/execute permissions
- Log all access attempts to the observability layer
- Truncate oversized outputs and offload to filesystem

**Rules**:
- Path resolution must canonicalize before any boundary check
- A path that escapes the workspace root is always a SandboxViolation
- Denylist evaluated after allowlist — a path matching both is denied
- Tools never know which isolation mode is active
- A SandboxViolation is a typed error, not a panic
- Never modifies tool input — validates or rejects

**Path resolution algorithm** (must be followed exactly):
```
1. Join root + raw_path (never trust raw_path as absolute)
2. Canonicalize (resolves .., symlinks)
3. Check starts_with(root) — if not: PathEscape violation
4. Check against denied_paths — if match: PathDenied violation
5. If allowlist non-empty, check allowed_paths — if no match: PathDenied violation
6. Check denied_extensions
7. If read_only and write operation: ReadOnlyViolation
8. Return canonicalized path
```

**Isolation modes**:
- `None` — trusted local development only, never production (emits warning at construction)
- `WorkspaceScoped` — path enforcement only, appropriate for most use cases
- `Bubblewrap` — Linux process isolation for shell commands, no daemon overhead
- `Docker` — full isolation including network

---

### ContextManager

**Purpose**: Assemble and maintain the context window. Decides what goes in, what stays in, and what gets summarized or evicted.

**Responsibilities**:
- Build context for each turn from PromptChunkRegistry, GuideRegistry, MemoryProvider, ToolRegistry
- Track token usage accurately
- Compact older context when approaching the window limit
- Offload tool results that exceed a size threshold
- Inject just-in-time skill context (ephemeral, Block 3 only)
- Annotate assembled context via CacheProvider

**Rules**:
- Context assembly happens before every turn — not once at session start
- Block 1 is computed ONCE at harness startup from PromptChunkRegistry — never recomputed
- Compaction fires at configurable threshold (default 80%), not at the limit
- Compaction preserves: decisions, open problems, task state, recent files, active thinking blocks
- Compaction discards: redundant tool outputs, superseded observations, dead-end reasoning
- Tool result truncation: keep first N and last N tokens, offload full result to filesystem
- ContextManager never calls the model directly — requests compaction through the agent
- Token counts come from ModelInterface, not estimated
- Tool schemas sorted by name before rendering — ensures deterministic ordering

**Cache responsibilities**:
- Block 1: rendered from `ComposedPrompt` (pre-computed at startup), memoized — never recomputed
- Block 2: assembled per-session, hash checked — log warning if it changes mid-session
- Block 3: assembled per-turn, never cached
- `record_cache_result()` updates ContextMeta and emits cache stats to ObservabilityProvider

---

### MemoryProvider

**Purpose**: Persist and retrieve knowledge across turns and sessions.

**Responsibilities**:
- Store episodic memory (what happened in this session/task)
- Store semantic memory (distilled rules, patterns, skills)
- Retrieve relevant memory given a query or task context
- Manage memory lifecycle (create, update, deprecate)

**Rules**:
- Episodic and semantic memory stored and retrieved separately
- Agent never writes to memory directly — harness mediates
- Writes validated — overwrites require explicit merge or replace decision
- Retrieved memory scored for relevance — low-relevance items not injected
- Memory is versioned — previous versions retained for regression analysis
- `MetaAgentProposed` memories start in PendingReview regardless of caller input

---

### PromptChunkRegistry

**Purpose**: Manage named, cacheable prompt chunks that compose the system prompt deterministically. Sits below GuideRegistry.

**Responsibilities**:
- Register named prompt chunks with slot and cache block metadata
- Compose chunks for a given agent configuration (role + mode + capabilities + skills)
- Validate compositions at harness construction time
- Provide the ComposedPrompt that ContextManager uses for Block 1

**Chunk slots**:
```
Role        — who the agent is (e.g. "role-coding-agent", "role-evaluator")
Mode        — how it behaves (one per Mode variant)
Capability  — what tools it has (one per tool or toolset)
Skill       — domain knowledge (from GuideRegistry)
Task        — what it's doing right now (PerSession)
Environment — where it is (PerSession)
PriorSession — what happened before (PerSession)
Budget      — resource constraints (PerTurn only)
Ephemeral   — just-in-time injection (PerTurn only)
```

**Rules**:
- Chunk content must be deterministic — no dynamic values, timestamps, or random elements
- `compose()` called at construction time, not at runtime — not on the hot path
- `ChunkSlot::Budget` and `ChunkSlot::Ephemeral` are always PerTurn — registering them as Static is a ChunkError
- If a chunk's content hash changes after startup, emit error-level log — cache is now invalid
- `ComposedPrompt.rendered` is memoized — invalidated only if a chunk hash changes

**Mode system**: `Mode` is a first-class concept driving three things at construction time:
```
Mode.AlwaysAsk  → "mode-always-ask" chunk + RequireHuman policy + ReadOnly tool phase
Mode.AutoEdit   → "mode-auto-edit" chunk + AutoApprove(Medium) policy + Execution phase
Mode.Plan       → "mode-plan" chunk + AutoApprove(Low) policy + Planning phase
Mode.SafeAuto   → "mode-safe-auto" chunk + RequireHuman(High+Critical) policy + Execution phase
Mode.Yolo       → "mode-yolo" chunk + AutoApprove(all) policy + Execution phase
```

---

### CacheProvider

**Purpose**: Provider-specific cache annotation and stats parsing. Keeps provider-specific cache logic out of ContextManager.

**Responsibilities**:
- Annotate a fully assembled context with provider-specific cache markers
- Parse cache usage stats from model responses
- Report provider identity for observability and auto-detection

**Rules**:
- Injected into ContextManager, not ModelInterface
- `annotate()` is a no-op if `supports_caching()` is false
- Auto-detected from the registered ModelInterface provider if not explicitly set
- `NullCacheProvider` is the default for testing — never interferes with unit tests

**Standard implementations**:
- `AnthropicCacheProvider` — explicit `cache_control: { type: "ephemeral" }` markers, max 4 breakpoints
- `OpenAICacheProvider` — automatic above 1024 tokens, `annotate()` is no-op
- `OllamaCacheProvider` — no caching support, all no-ops
- `NullCacheProvider` — testing default, all no-ops

---

### GuideRegistry

**Purpose**: Manage the lifecycle of all guides and skills — the feedforward artifacts that steer the agent before it acts.

**Responsibilities**:
- Store guides with metadata (source, domain, creation, usage history)
- Select relevant guides for a given task and domain at session start
- Track guide usage and outcome correlation
- Flag guides for review when performance degrades
- Accept new guides proposed by trace analysis or meta-agents

**Guide states**: Active, PendingReview, Deprecated, Stale — nothing is ever hard-deleted.

**Rules**:
- A guide is flagged for review if sessions using it fail at a significantly higher rate than sessions without it
- A failure pattern appearing in N or more traces within a time window triggers a skill-generation flag
- Automated proposals always start in PendingReview — human approval required to promote to Active
- A guide unused for T time periods is flagged as Stale, not auto-deprecated
- Conflicting guides detected at registration and flagged for resolution
- `MetaAgentProposed` source forces PendingReview regardless of caller input

---

### SensorChain

**Purpose**: Execute feedback controls after agent actions and evaluate output quality.

**Responsibilities**:
- Run registered sensors at defined trigger points (PostTool, PostTurn, PostSession, Continuous)
- Return sensor results to the harness for routing
- Track firing history for pattern detection
- Report results to observability layer

**Sensor outcomes**: Pass (continue), Warn (inject observation into next turn), Halt (stop and surface)

**Rules**:
- Computational sensors (linters, type checkers, test runners) run on every relevant trigger
- Inferential sensors (LLM-as-judge) run on configurable triggers — not every turn
- Sensors observe and report — they never modify agent output
- `fire()` returns all results — harness decides routing, chain does not short-circuit
- Warn results inject an observation string into the next turn's context
- Low-signal sensors (never fires after N sessions, or fires > X% of turns) flagged for review
- Sensor results feed GuideRegistry improvement loop

---

### Middleware Chain

**Purpose**: Intercept the agent loop at defined hook points for cross-cutting concerns.

**Hook points**: `before_session`, `before_turn`, `before_tool`, `after_tool`, `before_completion`, `after_session`

**MiddlewareDecision variants**:
```
Continue
ContinueWithModification
ForceAnotherTurn { inject: String }   — BeforeCompletion only
Halt { reason: String }
SurfaceToHuman { request: HumanRequest }  — valid on BeforeTool and BeforeCompletion
```

**Rules**:
- Before hooks: sorted by priority ascending (lowest runs first)
- After hooks: sorted by priority descending (wrapping pattern)
- First Halt or SurfaceToHuman stops the chain
- ForceAnotherTurn: all injections concatenated, chain continues
- Middleware must not hold session state between calls (use external map keyed by SessionId)
- Middleware must not call ModelInterface or ToolRegistry directly

**Standard implementations**:
- `DirectoryMapMiddleware` — injects workspace structure on session start and every N turns
- `TimeBudgetMiddleware` — injects budget warning when threshold crossed
- `LoopDetectionMiddleware` — tracks per-file edit counts, injects reconsideration after N edits
- `PreCompletionChecklistMiddleware` — forces verification pass before completion
- `PermissionMiddleware` — evaluates tool calls against ApprovalPolicy, returns SurfaceToHuman
- `PiiRedactionMiddleware` — scans tool inputs/outputs for PII, redacts before agent or logging
- `RateLimitMiddleware` — enforces per-session and per-minute token rate limits
- `TokenBudgetMiddleware` — checks cumulative spend, injects warning, Halt at limit
- `PatchToolCallsMiddleware` — fixes syntactically invalid or dangling tool call JSON (auto-registered at highest BeforeTool priority, always present)
- `TracingMiddleware` — emits structured spans for every hook point (lowest priority, always present)

---

### ObservabilityProvider

**Purpose**: Record everything the harness does in a queryable, structured form.

**Responsibilities**:
- Emit a span for every turn, tool call, sensor execution, and context operation
- Record token usage, latency, and cost per operation
- Attach session and task identifiers to every span
- Surface aggregated metrics

**Rules**:
- Every harness operation emits a span — nothing is exempt
- Spans are structured, not free-text — keys are defined
- Observability is a passive observer — never modifies harness behavior
- Trace data is the input to the improvement loop — if it isn't traced, it can't be improved
- Cost and token usage are first-class span fields, including cache read/write tokens and cost
- All `emit_*` methods are fire-and-forget — never block the harness loop
- **Reference implementation uses OTLP (OpenTelemetry Protocol)** — compatible with Langfuse, Grafana, Datadog, Honeycomb, Jaeger, New Relic on day one

---

### TerminationPolicy

**Purpose**: Evaluate after every turn whether the harness should continue, wait, or stop.

**Responsibilities**:
- Evaluate session state against defined completion criteria
- Return Continue, HaltSuccess, HaltFailure, or HaltBudgetExceeded
- Accept a pluggable CompletionCheck

**Rules**:
- Model's self-assessment is one input, not the decision
- Budget limits are hard stops — evaluated first, unconditionally, every turn
- HaltFailure must include a typed reason
- TerminationPolicy is domain-agnostic — all domain knowledge lives in the injected CompletionCheck
- HumanHalted is set by the harness directly when `HumanResponse::Halt` is received — TerminationPolicy is not consulted
- Budget tracking resumes from `PausedState.budget_used` on `resume()`

**Evaluation algorithm**:
```
evaluate(input):
  1. check_budget() — hard stop, always first
  2. if !agent_claims_done: return Continue
  3. if sensor_results.any(Halt): return HaltFailure(UnrecoverableSensorHalt)
  4. completion_check.check(session_state):
     None    → HaltSuccess
     Some(why) → Continue (harness injects `why` into next turn context)
```

---

## How They Wire Together

```
Harness.run(task):

  // Construction-time setup (once)
  chunk_registry.compose(role, mode, capabilities, skills) → ComposedPrompt

  // Session start
  guide_registry.select(task) → guides
  memory_provider.query(session) → prior_state
  middleware.fire_before_session(task)

  loop:
    middleware.fire_before_turn(context)
    context_manager.assemble(session, composed_prompt, guides, memory, tools)
    cache_provider.annotate(context)
    agent.turn(context, on_stream) → TurnResult

    if ToolCallRequested(calls):
      Layer 1 always-halt check
      middleware.fire_before_tool(calls)
      for call in calls:
        sandbox.validate(call)
        tool_registry.dispatch(call, sandbox) → ToolResult | WaitingForHuman
        sensor_chain.fire(PostTool, result)
      middleware.fire_after_tool(calls, results)
      context_manager.append_all(results)
      if should_compact(): compact via agent turn
      goto loop

    if FinalResponse(response):
      middleware.fire_before_completion(response)  ← can force another turn
      sensor_chain.fire(PostTurn, response)
      termination_policy.evaluate(state) → Continue | Halt*

  middleware.fire_after_session(result)
  memory_provider.store_episodic(session_observations)
  guide_registry.record_usage(guides_used, outcome)
  chunk_registry.record_usage(composition_used, outcome)
  observability.flush_session(session_id)
```

---

## The Improvement Flywheel

These rules govern how the system gets better over time and sit above any individual component:

1. A failure in one trace is an incident. A failure in N traces is a pattern.
2. A pattern without a guide is a gap. Create a candidate skill and route to pending review.
3. A guide that correlates with higher failure rates is a liability. Flag for review, not silent deprecation.
4. A guide that hasn't been used is either irrelevant or never loaded. Investigate before marking stale.
5. A sensor that never fires is not evidence of quality. It may be inadequate detection. Audit periodically.
6. A sensor that always fires is noise. Either the guide is wrong or the sensor threshold is wrong.
7. Episodic memory that appears across multiple sessions is a candidate for distillation into semantic memory.
8. No automated process promotes a guide to active without a review gate. Speed of learning is less important than correctness of learning.
9. Chunk compositions that correlate with higher failure rates are flagged the same way individual guides are.
10. The evaluation harness (EvalHarness) is the outer ring that drives the flywheel. Regression tasks must stay passing. Challenge tasks measure improvement. Canary tasks detect breakthroughs.

---

## Deployment Surfaces

The harness is stateless — the same `run()` / `resume()` interface works across:

- **CLI**: thin wrapper reads config, constructs harness via builder, streams tokens to stdout, writes PausedState to disk on WaitingForHuman
- **REST API**: async task endpoint, SSE or WebSocket for streaming, database/Redis for PausedState storage
- **Library**: embedded in any application — desktop, VS Code extension, CI pipeline, Lambda function
- **Queue worker**: polls job queue, constructs harness, publishes RunResult back to queue
- **TypeScript → Rust subprocess**: TypeScript REST API shells out to Rust harness binary, streams results back via stdout/pipe (recommended v1 deployment for polyglot setups — same pattern as Claude Agent SDK)

The host owns: SessionState persistence, PausedState storage, StreamEvent delivery, task queue management, auth, rate limiting, retry of failed runs.

The harness owns: agent loop execution, component orchestration, RunResult production.

---

## Ecosystem Strategy

**Path A now, architect for Path B later.**

v1 reference implementations use off-the-shelf tools:
```
ObservabilityProvider → OtelObservabilityProvider (OTLP)
MemoryProvider        → SqliteMemoryProvider, PostgresMemoryProvider
SandboxProvider       → DockerSandboxProvider, E2BSandboxProvider
GuideRegistry         → FilesystemGuideRegistry
```

The trait boundaries are already the right ecosystem surfaces for managed services. Learn from Path A adoption which managed service bets to make in Path B.
