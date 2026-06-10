# Example 12 — cordyceps (the Composable Execution capstone)

The capstone of the Composable Execution refactor (#117–#131). It wires and runs
the motivating composition end-to-end:

```text
Ralph[ PlanExecute[ ReAct, SelfVerifying[ ReAct ] ] ]
```

```text
Ralph (continuation wrapper)        agent: ralph-agent
  resets the context window, resumes from durable task_list progress
  └─ PlanExecute
       ├─ plan:    ReAct            agent: planner   toolset: plan-tools   out: plan-schema   budget: PerLoop{4}
       │           explores the repo, builds a blocker-aware task graph via `task_list`
       └─ execute: SelfVerifying     evaluator: exec-evaluator (Default-FAIL)
            └─ worker: ReAct         agent: executor  toolset: exec-tools   out: worker-schema  budget: PerLoop{12}
                       audits ONE module per ready task, dependency-ordered
```

The whole tree shares ONE budget / usage / observability context. A single
runaway node is bounded by its own `BudgetPolicy` without cascading to unrelated
tasks. A paused tree resumes by re-resolving handles, with no reconfiguration.

## The big idea: the strategy is DATA

You do not write the loop. You describe it as a composed `LoopStrategy` tree —
here loaded VERBATIM from the canonical fixture
`fixtures/strategy/cordyceps_tree.json`:

```python
tree = TypeAdapter(LoopStrategy).validate_json(CORDYCEPS_TREE_JSON)
```

The tree carries only serializable string HANDLES (`planner`, `exec-tools`,
`worker-schema`, `exec-evaluator`, …). At run entry the harness resolves each
handle against an `ExecutionRegistry` you assemble once at startup:

```python
registry = (
    ExecutionRegistry.builder()
    .agent("planner", model_agent(...))
    .agent("executor", model_agent(...))
    .agent("ralph-agent", model_agent(...))
    .toolset("plan-tools", ...)
    .toolset("exec-tools", ...)
    .schema("plan-schema", ...)
    .schema("worker-schema", ...)
    .verifier("exec-evaluator", exec_evaluator())   # Default-FAIL
    .build()
)
```

`registry.validate(task)` walks the tree and raises on the FIRST unresolved
handle as a STARTUP error — before the first model turn. Live objects never enter
the serialized `Task`; only string handles do. That is what makes resume trivial:
a fresh registry re-resolves every handle with no Task reconfiguration.

## What you can read off the tree before running it (AC5)

The worst-case per-window turn count is computable statically:

```text
Ralph[PlanExecute[ReAct{4}, SelfVerifying[ReAct{12}]]]
     = 4 + (12 + 1) = 17           // SelfVerifying adds the single evaluator turn
```

```python
loop_strategy_max_steps(tree)  # == 17
```

`max_steps` is Option-monadic: an `Unlimited` budget anywhere in the tree
collapses the whole figure to `None` ("no finite advisory bound"). The example
prints `17` before each run.

## How the phases behave

- **Plan → ready-set (AC2).** The plan phase explores the repo and authors a
  blocker-aware task graph via the `task_list` tool. The execute phase walks that
  graph as a READY-SET: it repeatedly picks the lowest-id task whose blockers are
  all complete, audits it, and advances — dependency-ordered, with independent
  branches isolated.
- **Default-FAIL self-verification (AC3).** Each task's worker result is checked
  by the `exec-evaluator` (`EvaluatorResponseVerifier`, `max_iterations = 1`): a
  single read-only evaluator turn. Only an explicit `PASS` clears a task;
  indeterminate output ⇒ `Failed`. Default-FAIL is not configurable.
- **Bounded runaway, no cascade (AC4).** A node that exhausts its own
  `BudgetPolicy` resolves to a task FAILURE that blocks only its transitive
  dependents. Unrelated tasks keep scheduling; at drain the run reports
  `TasksBlockedByFailure { failed_task, completed, blocked }` with the partition.
- **Pause → resume re-resolving handles (AC6).** Under
  `EscalationModeSurfaceToHuman`, a budget-exhausted node pauses with a
  `HumanRequestBudgetExhausted` carrying its `available_actions`
  (`continue_with_budget` / `skip` / `fail`). The operator picks one; the harness
  resumes by RE-RESOLVING every handle from the registry — no reconfiguration.

## The #114 consult ladder — PRESERVED, with its mediation seam moved

The worker still escalates mid-audit through the two consult tools:

- `research_best_practices` → `kind="research"` → a web_search helper harness
  (budget 5, overflow **SoftFail**: on exhaustion the worker resumes with a
  "budget exhausted, proceed" message and finishes on general knowledge);
- `consult_advisor` → `kind="advice"` → a near-frontier cloud-model advisor with
  `read_file`/`grep` (budget 3, overflow **EscalateToHuman**: the host surfaces a
  three-choice ladder — run the advisor once more / proceed without help / type a
  free-form answer).

Both lower to `ToolOutputConsult`. The pre-#131 example had a `SubagentTool`
mediate these per-node. The declarative tree has NO `SubagentTool`, so **the seam
moved to the host run loop**: a worker-leaf consult propagates up through the
composed tree to a top-level `RunResultConsult`, and `main`'s `_mediate_consult`
routes it by `kind` to the helper harness (per-kind budget + overflow, host-owned
for the whole run), then calls `harness.resume_consult(...)`. Identical #114
semantics; the budget map just lives in the host instead of a tool.

**Resuming a consult continues the whole walk (#131 core change).** Because the
consult surfaces from a worker nested inside `PlanExecute`, resuming must do more
than restart that one leaf — the consulted task must still be self-verified and
the rest of the ready-set must still run. So each combinator rewrites the pause's
task to its own composed task on the way up (the pause ends up carrying the FULL
tree), and `resume_consult` re-drives the strategy: `PlanExecute` resumes its
in-progress task from the worker's own conversation (answer injected), the
SelfVerifying evaluator runs, the task is marked `Completed`, and the walk
proceeds to the remaining tasks.

The **#115 `load_skill`** worker-side tool WAS dropped — there is no per-node tool
seam in the declarative tree. The `audit` **skill is KEPT**, now riding the single
GLOBAL `context_manager` (`SkillInjectingContextManager`), seeded ALWAYS-ACTIVE at
startup: its procedure reaches the model structurally every turn, compaction-proof,
with no `load_skill` round-trip.

Per-node toolset scoping is now in place: each leaf dispatches ONLY its own
toolset's tools, wired per-toolset on the `HarnessBuilder` via
`.toolset_tools("plan-tools", ...)` / `.toolset_tools("exec-tools", ...)`. The
planner (`plan-tools`) cannot reach an exec-only tool (`read_file`) and the worker
(`exec-tools`) cannot reach a plan-only tool (`task_list`/`list_dir`) — the leaked
union is closed. The registry toolset HANDLES are validation-only presence entries
(they must resolve for `validate()`) and are never dispatched; the real dispatchable
catalogues live in `HarnessConfig.toolset_catalogues`. The audit is kept read-only by
a read-only sandbox + the absence of any write tool.

## Run it

```sh
ollama serve &
ollama pull gemma4:e4b
export SPORE_WEB_SEARCH_ENDPOINT=http://localhost:8888/search?format=json  # SearXNG (research consult)
uv run main.py   # press enter to accept the default audit prompt; Ctrl-D to quit
```

Configuration (see `.env.example`): `SPORE_OLLAMA_MODEL` (default `gemma4:e4b`),
`SPORE_OLLAMA_BASE_URL` (default `http://localhost:11434`),
`SPORE_ADVISOR_MODEL` (the `advice` consult handler's cloud model, default
`minimax-m3:cloud`), and `SPORE_WEB_SEARCH_ENDPOINT` (a SearXNG JSON endpoint
backing the `research` consult — REQUIRED, fail-fast like examples 06/11).

## Tests

- **Example** (no model): `test_skills.py` — the global skill injection
  (manifest always present, body only when active) and the frontmatter parser.
- **Core package** (deterministic recorded-model replay, in
  `python/packages/spore_core/tests/test_harness_cordyceps_composition.py`):
  `test_plan_builds_dag_execute_walks_readyset` (AC2),
  `test_self_verified_default_fail` (AC3), `test_cordyceps_runaway_bounded` (AC4),
  `test_cordyceps_max_steps_is_17_unlimited_is_none` (AC5),
  `test_resume_reresolves_handles` (AC6), and
  `test_worker_consult_surfaces_and_host_resumes` (the #114 ladder + #131 consult
  re-drive).

```sh
cd examples/python/12-cordyceps && uv run pytest
cd python/packages/spore_core && uv run pytest tests/test_harness_cordyceps_composition.py
```
