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

```rust
let tree: LoopStrategy = serde_json::from_str(CORDYCEPS_TREE_JSON)?;
```

The tree carries only serializable string HANDLES (`planner`, `exec-tools`,
`worker-schema`, `exec-evaluator`, …). At run entry the harness resolves each
handle against an `ExecutionRegistry` you assemble once at startup:

```rust
let registry = ExecutionRegistry::builder()
    .agent("planner", model_agent(...))
    .agent("executor", model_agent(...))
    .agent("ralph-agent", model_agent(...))
    .toolset("plan-tools", ...)
    .toolset("exec-tools", ...)
    .schema("plan-schema", ...)
    .schema("worker-schema", ...)
    .verifier("exec-evaluator", exec_evaluator())   // Default-FAIL
    .build();
```

`registry.validate(&task)` walks the tree and reports the FIRST unresolved
handle as a STARTUP error — before the first model turn. Trait objects never
enter the serialized `Task`; only string handles do. That is what makes resume
trivial: a fresh registry re-resolves every handle with no Task reconfiguration.

## What you can read off the tree before running it (AC5)

The worst-case per-window turn count is computable statically:

```text
Ralph[PlanExecute[ReAct{4}, SelfVerifying[ReAct{12}]]]
     = 4 + (12 + 1) = 17           // SelfVerifying adds the single evaluator turn
```

```rust
tree.max_steps()  // == Some(17)
```

`max_steps()` is Option-monadic: an `Unlimited` budget anywhere in the tree
collapses the whole figure to `None` ("no finite advisory bound"). The example
prints `Some(17)` before each run.

## How the phases behave

- **Plan → ready-set (AC2).** The plan phase explores the repo and authors a
  blocker-aware task graph via the `task_list` tool. The execute phase walks
  that graph as a READY-SET: it repeatedly picks the lowest-id task whose
  blockers are all complete, audits it, and advances — dependency-ordered, with
  independent branches isolated.
- **Default-FAIL self-verification (AC3).** Each task's worker result is checked
  by the `exec-evaluator` (`EvaluatorResponseVerifier`, `max_iterations = 1`): a
  single read-only evaluator turn. Only an explicit `PASS` clears a task;
  indeterminate output ⇒ `Failed`. Default-FAIL is not configurable.
- **Bounded runaway, no cascade (AC4).** A node that exhausts its own
  `BudgetPolicy` resolves to a task FAILURE that blocks only its transitive
  dependents. Unrelated tasks keep scheduling; at drain the run reports
  `TasksBlockedByFailure { failed_task, completed, blocked }` with the partition.
- **Pause → resume re-resolving handles (AC6).** Under
  `EscalationMode::SurfaceToHuman`, a budget-exhausted node pauses with a
  `HumanRequest::BudgetExhausted` carrying its `available_actions`
  (`continue_with_budget` / `skip` / `fail`). The operator picks one; the harness
  resumes by RE-RESOLVING every handle from the registry — no reconfiguration.

## What changed vs. the pre-#131 example (honest note)

The old depth-1 example was a hand-built `SubagentTool` orchestrator. Two of its
features were SubagentTool / per-node seams that the declarative tree does not
expose, so they were DROPPED:

- the **#114 consult ladder** (`research_best_practices` → `consult_advisor` →
  human). The composed tree has no per-node `ToolOutput::Consult` mediator, so
  there is nowhere to hang a per-kind consult handler.
- the **#115 `load_skill`** worker-side tool. There is no per-node tool seam in
  the declarative tree.

The `audit` **skill is KEPT** — but it now rides the single GLOBAL
`context_manager` (`SkillInjectingContextManager`), seeded ALWAYS-ACTIVE at
startup. Its procedure reaches the model structurally every turn,
compaction-proof, with no `load_skill` round-trip.

One more honest limitation: the harness dispatches every node's tool calls
through ONE global catalogue wired on the `HarnessBuilder` (the union of
`plan-tools` + `exec-tools`), not per-node — so the registry toolset HANDLES
must resolve for `validate()`, but tool scoping is not yet per-node. The audit
is kept read-only by a read-only sandbox + the absence of any write tool.

## Run it

```sh
ollama serve &
ollama pull gemma4:e4b
cargo run        # press enter to accept the default audit prompt; Ctrl-D to quit
```

Configuration (see `.env.example`): `SPORE_OLLAMA_MODEL` (default `gemma4:e4b`),
`SPORE_OLLAMA_BASE_URL` (default `http://localhost:11434`).

## Tests

- **Example crate** (no model): `tree_is_byte_identical` (the tree round-trips
  through serde and has the canonical keys/budgets/behaviors),
  `max_steps_is_17` (and `Unlimited ⇒ None`), `registry_validates`,
  `exec_evaluator_is_default_fail`, plus the kept skill-injection unit tests.
- **Core crate** (deterministic recorded-model replay, in
  `rust/crates/spore-core/tests/cordyceps_composition_fixture_replay.rs`):
  `plan_builds_dag_execute_walks_readyset` (AC2),
  `self_verified_default_fail` (AC3), `cordyceps_runaway_bounded` (AC4),
  `cordyceps_max_steps_is_17_unlimited_is_none` (AC5), and
  `resume_reresolves_handles` (AC6).

```sh
cd examples/rust/12-cordyceps && cargo test
cd rust && cargo test -p spore-core --test cordyceps_composition_fixture_replay
```
