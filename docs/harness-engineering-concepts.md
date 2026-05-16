## The Harness Components

------

### Model Interface

**Purpose**: The boundary between the harness and the underlying LLM. Normalizes provider differences so nothing else in the harness knows or cares which model it's talking to.

**Responsibilities**:

- Send a context (messages + tools + parameters) to a model, return a response
- Handle streaming vs. non-streaming transparently
- Report token usage back to the harness
- Surface provider errors as typed harness errors

**Rules**:

- Never called directly by tools, memory, or guides — only by the agent
- Token counts must be reported on every call, not optionally
- Provider-specific parameters (temperature, reasoning budget, etc.) are injected by the harness, not hardcoded
- A model call that exceeds a token budget is a harness error, not a model error

------

### Agent (One Turn)

**Purpose**: Execute a single turn — call the model with the current context, return either a tool call request or a final response. That's it.

**Responsibilities**:

- Accept a context and return a turn result
- Identify whether the model wants to call a tool or is done
- Not decide whether to call another turn — that's the harness

**Rules**:

- One turn = one model call, period
- Does not manage context — receives it fully assembled from the harness
- Does not execute tools — returns tool call requests to the harness for dispatch
- Does not decide termination — returns a result type that the harness evaluates
- Does not retry — if the model call fails, surface the error upward

------

### Harness (Runtime / Loop)

**Purpose**: Drive the agent loop. Owns the execution lifecycle, wires all components together, and decides when to keep going and when to stop.

**Responsibilities**:

- Assemble context before each turn (via ContextManager)
- Call the agent for one turn
- Dispatch tool calls to the ToolRegistry
- Evaluate the termination condition after each turn
- Fire lifecycle hooks at defined points
- Track iteration count, token spend, and elapsed time

**Rules**:

- The harness owns the loop — nothing else does
- Termination is evaluated against external state, not the model's self-assessment alone
- If max iterations or token budget is exceeded, terminate with an explicit reason, not silently
- A turn that produces neither a tool call nor a final response is an error condition
- The harness never executes tools directly — it dispatches to the ToolRegistry
- Component injection happens at construction time, not at runtime

------

### ToolRegistry

**Purpose**: Maintains the available tools and dispatches tool calls from the agent. The harness asks "what tools are available right now?" and "execute this tool call."

**Responsibilities**:

- Register tools with their schemas
- Return the active ToolSet for a given context (task phase, domain, etc.)
- Dispatch a tool call to the correct tool implementation
- Pass the SandboxProvider to every tool on dispatch
- Return typed results or typed errors

**Rules**:

- Tools are never called directly by the agent or harness — always dispatched through the registry
- The active ToolSet can change between turns — the registry decides what's available based on current state
- A tool call for an unregistered tool is a harness error, surfaced to the middleware chain
- Tool schemas must be validated at registration time, not at call time
- The registry does not retry failed tool calls — that's middleware's job

------

### Tool (Individual)

**Purpose**: Execute one specific action against the real world. File read, bash command, SQL query, document search, HTTP call.

**Responsibilities**:

- Accept a SandboxProvider and validated parameters
- Execute the action
- Return a typed result or typed error

**Rules**:

- Every tool must accept a SandboxProvider — no tool touches the environment directly
- A tool that receives invalid parameters returns a typed error, never panics
- Tools are stateless — they do not hold references to session state
- A tool must not call other tools — composition happens at the harness level
- Tool output should be the minimum necessary to inform the next turn — large outputs are truncated to head+tail, with the full output offloaded to the filesystem by the SandboxProvider

------

### SandboxProvider

**Purpose**: The capability object that enforces the execution boundary. All tools receive it and use it for every interaction with the environment.

**Responsibilities**:

- Validate and resolve file paths against the workspace root
- Execute shell commands within the defined isolation mode
- Enforce read/write/execute permissions
- Log all access attempts (allowed and denied) to the observability layer
- Truncate oversized outputs and offload to filesystem

**Rules**:

- Path resolution must canonicalize before any boundary check — no raw path operations
- A path that escapes the workspace root is a SandboxViolation, always, regardless of how it was constructed
- Denylist is evaluated after allowlist — a path matching both is denied
- Shell commands respect the isolation mode: None, WorkspaceScoped, Bubblewrap, or Docker — tools never know which
- A SandboxViolation is a typed error surfaced to the middleware chain, not a panic
- The SandboxProvider never modifies tool input — it validates or rejects

**Isolation modes and when to use them**:

- `None` — trusted local development only, never production
- `WorkspaceScoped` — path enforcement only, appropriate for most use cases
- `Bubblewrap` — adds process isolation for shell commands, Linux only, no daemon overhead
- `Docker` — full isolation including network, appropriate when reproducibility or network control is required

------

### ContextManager

**Purpose**: Assemble and maintain the context window. Decides what goes in, what stays in, and what gets summarized or evicted as the window fills.

**Responsibilities**:

- Build the initial context for a session (system prompt, guides, memory, tool schemas)
- Track token usage across the current context
- Compact older context when approaching the window limit
- Offload tool results that exceed a size threshold
- Inject just-in-time context when the agent requests a skill

**Rules**:

- Context assembly happens before every turn, not once at session start
- Compaction fires at a configurable threshold (default 80% of window limit), not at the limit — never let the window fill completely
- Compaction preserves: architectural decisions, open problems, current task state, recent file list
- Compaction discards: redundant tool outputs, superseded observations, intermediate reasoning that led nowhere
- Tool result truncation: keep first N and last N tokens, offload full result to filesystem with a retrieval reference
- The ContextManager never calls the model directly — if compaction requires a summarization call, it requests one through the agent
- Token counts come from the ModelInterface, not estimated

------

### MemoryProvider

**Purpose**: Persist and retrieve knowledge across turns and sessions. Abstracts over the storage mechanism so the harness doesn't care whether memory is files, a database, or a vector store.

**Responsibilities**:

- Store session-generated observations (episodic memory)
- Store distilled rules and patterns (semantic memory / skills)
- Retrieve relevant memory given a query or task context
- Manage memory lifecycle (creation, update, deprecation, deletion)

**Rules**:

- Episodic memory (what happened this session) and semantic memory (what we've learned generally) are stored and retrieved separately
- The agent never writes to memory directly — it produces observations that the harness passes to the MemoryProvider
- Memory writes are validated — a write that would overwrite an existing fact requires an explicit merge or replace decision, not a silent overwrite
- Memory retrieved for context injection is scored for relevance — low-relevance memory is not injected even if it exists
- Memory is versioned — previous versions of a guide or skill are retained, not deleted, so regressions can be identified

------

### GuideRegistry

**Purpose**: Manage the lifecycle of all guides and skills — the feedforward artifacts that steer the agent before it acts.

**Responsibilities**:

- Store guides with metadata (source, domain, creation, usage history)
- Select relevant guides for a given task and domain at session start
- Track guide usage and outcome correlation
- Flag guides for review when performance degrades
- Accept new guides proposed by trace analysis or meta-agents

**Rules**:

- A guide is active, deprecated, or pending review — nothing is ever hard-deleted
- A guide is flagged for review if sessions using it fail at a significantly higher rate than sessions without it
- A new failure pattern appearing in N or more traces within a time window triggers a flag for skill generation
- Guides proposed by automated processes (trace analysis, meta-agents) start in pending review state — a human or explicit approval step promotes them to active
- A guide that hasn't been used in T time periods is flagged as stale, not auto-deprecated
- Conflicting guides (guides that give contradictory instructions) are detected at registration time and flagged for resolution

------

### SensorChain

**Purpose**: Execute feedback controls after agent actions and evaluate output quality. The feedback half of the feedforward/feedback pair.

**Responsibilities**:

- Run registered sensors after defined trigger points (post-tool, post-turn, post-session, continuous)
- Return sensor results to the harness for routing (surface to agent, surface to human, halt)
- Track sensor firing history for pattern detection
- Report sensor results to the observability layer

**Rules**:

- Computational sensors (linters, type checkers, test runners) run on every relevant trigger — they are cheap enough
- Inferential sensors (LLM-as-judge, code review agents) run on configurable triggers — not every turn
- A sensor result has three possible outcomes: pass (continue), warn (continue with observation injected into context), halt (stop execution and surface to human)
- Sensors do not modify agent output — they observe and report
- A sensor that consistently fires without catching real failures is flagged as low signal — a sensor that never fires is flagged as potentially inadequate detection
- Sensor results feed the GuideRegistry — repeated sensor failures on the same pattern trigger skill generation

------

### Middleware Chain

**Purpose**: Intercept the agent loop at defined hook points to enforce cross-cutting concerns without modifying core component logic.

**Hook points**:

- `before_session` — runs once before the loop starts
- `before_turn` — runs before every model call
- `before_tool` — runs before every tool dispatch
- `after_tool` — runs after every tool result
- `before_completion` — runs when the agent attempts to terminate
- `after_session` — runs once after the loop ends

**Rules**:

- Middleware is ordered — earlier-registered middleware runs first on before hooks, last on after hooks (wrapping pattern)
- A middleware can halt execution by returning a halt result — downstream middleware in the chain does not run
- A middleware can modify context going into a hook point but not the harness state directly
- Middleware must not call the model or execute tools — those go through the harness
- The `before_completion` hook is the last line of defense before termination — it can force another turn by returning a continue result

------

### ObservabilityProvider

**Purpose**: Record everything the harness does in a queryable, structured form. The foundation for trace analysis, failure detection, and guide improvement.

**Responsibilities**:

- Emit a span for every turn, tool call, sensor execution, and context operation
- Record token usage, latency, and cost per operation
- Attach session and task identifiers to every span
- Surface aggregated metrics (failure rates, token burn, sensor firing frequency)

**Rules**:

- Every harness operation emits a span — nothing is exempt
- Spans are structured, not free-text — keys are defined, not arbitrary
- Observability is a passive observer — it never modifies harness behavior
- Trace data is the input to the improvement loop — if it isn't traced, it can't be improved
- Cost and token usage are first-class fields on every span, not derived later

------

### TerminationPolicy

**Purpose**: Evaluate after every turn whether the harness should continue, wait, or stop. Decouples the stop condition from the loop implementation.

**Responsibilities**:

- Evaluate the current session state against defined completion criteria
- Return continue, halt-success, halt-failure, or halt-budget-exceeded
- Accept pluggable completion checks (feature list complete, test suite passing, question answered, etc.)

**Rules**:

- The model's opinion that it is done is one input to the termination policy, not the decision
- Budget limits (max turns, max tokens, max time) are hard stops — they cannot be overridden by other conditions
- A halt-failure result must include a typed reason — silent failures are not permitted
- The completion check is injected at harness construction — the termination policy does not know the domain

------

## How They Wire Together

```
Harness.run(task):
  GuideRegistry   → select guides for this task
  ContextManager  → assemble initial context (guides + memory + tools)
  Middleware      → fire before_session hooks
  
  loop:
    Middleware      → fire before_turn hooks
    ContextManager  → assemble turn context
    Agent           → one turn → tool call | response
    
    if tool_call:
      Middleware      → fire before_tool hooks
      SandboxProvider → validate
      ToolRegistry    → dispatch
      SensorChain     → fire post_tool sensors
      Middleware      → fire after_tool hooks
      ContextManager  → append result (truncate if needed)
    
    if response:
      Middleware      → fire before_completion hooks  ← can force continue
      SensorChain     → fire post_turn sensors
      TerminationPolicy → continue | halt
  
  Middleware      → fire after_session hooks
  MemoryProvider  → store episodic observations
  GuideRegistry   → update usage/outcome metrics
  ObservabilityProvider → flush session trace
```

------

## The Improvement Flywheel Rules

These govern how the system gets better over time and sit above any individual component:

1. A failure that appears in one trace is an incident. A failure that appears in N traces is a pattern.
2. A pattern without a guide is a gap. Create a candidate skill and route to pending review.
3. A guide that correlates with higher failure rates is a liability. Flag for review, not silent deprecation.
4. A guide that hasn't been used is either irrelevant or never loaded. Investigate before marking stale.
5. A sensor that never fires is not evidence of quality. It may be inadequate detection. Audit periodically.
6. A sensor that always fires is noise. Either the guide is wrong or the sensor threshold is wrong.
7. Episodic memory that appears across multiple sessions is a candidate for distillation into semantic memory.
8. No automated process promotes a guide to active without a review gate. Speed of learning is less important than correctness of learning.