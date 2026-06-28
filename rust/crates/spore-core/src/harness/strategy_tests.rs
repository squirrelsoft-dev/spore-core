// ============================================================================
// Composable Execution Part A (issue #119) — LoopStrategy / StrategyRef /
// *Ref handles / RunStrategy stubs round-trip + fixture replay.
// ============================================================================
use super::*;

/// The canonical cordyceps tree `Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]`.
/// Concrete values are the cross-language byte-identity ground truth; they
/// MUST match `fixtures/strategy/cordyceps_tree.json`.
fn cordyceps_tree() -> LoopStrategy {
    LoopStrategy::Ralph(RalphConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(LoopStrategy::PlanExecute(PlanExecuteConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            plan: Box::new(LoopStrategy::ReAct(ReactConfig {
                behavior: BudgetExhaustedBehavior::Escalate,
                budget: BudgetPolicy::PerLoop { value: 12 },
                agent: AgentRef("planner".to_string()),
                toolset: ToolsetRef("plan-tools".to_string()),
                // A.5 (#124, Q3): the structured plan slot declares an output
                // schema so it yields a typed task graph.
                output: Some(SchemaRef("plan-schema".to_string())),
                system_prompt: None,
            })),
            execute: Box::new(LoopStrategy::SelfVerifying(SelfVerifyingConfig {
                behavior: BudgetExhaustedBehavior::Escalate,
                inner: Box::new(LoopStrategy::ReAct(ReactConfig {
                    behavior: BudgetExhaustedBehavior::Escalate,
                    budget: BudgetPolicy::PerLoop { value: 12 },
                    agent: AgentRef("executor".to_string()),
                    toolset: ToolsetRef("exec-tools".to_string()),
                    // A.5: the structured worker slot declares an output schema.
                    output: Some(SchemaRef("worker-schema".to_string())),
                    system_prompt: None,
                })),
                evaluator: SchemaRef("exec-evaluator".to_string()),
                eval_agent: None,
                eval_toolset: None,
            })),
            plan_model: None,
        })),
        agent: AgentRef("ralph-agent".to_string()),
    })
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../../fixtures/{name}"))
}

/// Minify a pretty-printed JSON string WITHOUT reordering keys (unlike
/// re-serializing through `serde_json::Value`, which sorts map keys). The
/// fixtures are pretty-printed in struct-declaration order, so the minified
/// form is the cross-language byte-identity target our serializer must
/// reproduce. Strips insignificant whitespace while preserving string
/// contents verbatim.
fn minify_preserving_order(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_string = false;
    let mut escaped = false;
    for ch in s.chars() {
        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else if ch == '"' {
            in_string = true;
            out.push(ch);
        } else if !ch.is_whitespace() {
            out.push(ch);
        }
    }
    out
}

/// Assert a value's serialization is byte-identical to the order-preserving
/// minified fixture (the cross-language ground truth).
fn assert_fixture_byte_identical<T: Serialize>(value: &T, fixture: &str) {
    let raw = std::fs::read_to_string(fixture_path(fixture)).expect("fixture present");
    let canonical = minify_preserving_order(&raw);
    let ours = serde_json::to_string(value).unwrap();
    assert_eq!(ours, canonical, "{fixture} not byte-identical");
}

#[test]
fn each_variant_round_trips() {
    let variants = vec![
        LoopStrategy::ReAct(ReactConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            budget: BudgetPolicy::PerLoop { value: 8 },
            agent: AgentRef("a".to_string()),
            toolset: ToolsetRef("t".to_string()),
            output: Some(SchemaRef("out".to_string())),
            system_prompt: None,
        }),
        LoopStrategy::ReAct(ReactConfig::per_loop(3)),
        LoopStrategy::PlanExecute(PlanExecuteConfig::simple(None)),
        LoopStrategy::PlanExecute(PlanExecuteConfig::simple(Some(ModelConfig {
            provider: "anthropic".to_string(),
            model_id: "claude".to_string(),
        }))),
        LoopStrategy::SelfVerifying(SelfVerifyingConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            inner: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(2))),
            evaluator: SchemaRef("ev".to_string()),
            eval_agent: None,
            eval_toolset: None,
        }),
        LoopStrategy::Ralph(RalphConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            inner: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(1))),
            agent: AgentRef("r".to_string()),
        }),
        LoopStrategy::HillClimbing(HillClimbingConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            inner: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(5))),
            direction: HillClimbingDirection::Maximize,
            max_stagnation: 3,
            revert_on_no_improvement: true,
            min_improvement_delta: 0.25,
            evaluator: AgentRef("metric".to_string()),
        }),
    ];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: LoopStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back, "round-trip failed for {json}");
    }
}

#[test]
fn react_tag_is_react_not_re_act() {
    let json = serde_json::to_string(&LoopStrategy::ReAct(ReactConfig::per_loop(8))).unwrap();
    assert!(json.contains(r#""kind":"react""#), "got {json}");
    assert!(!json.contains("re_act"), "got {json}");
    // ReactConfig fields flatten next to the tag (no nesting).
    assert_eq!(
        json,
        r#"{"kind":"react","budget":{"kind":"per_loop","value":8},"behavior":{"kind":"escalate"},"agent":"","toolset":""}"#
    );
}

#[test]
fn handles_are_transparent_strings() {
    assert_eq!(
        serde_json::to_string(&AgentRef("x".to_string())).unwrap(),
        r#""x""#
    );
    assert_eq!(
        serde_json::to_string(&ToolsetRef("y".to_string())).unwrap(),
        r#""y""#
    );
    assert_eq!(
        serde_json::to_string(&SchemaRef("z".to_string())).unwrap(),
        r#""z""#
    );
    let a: AgentRef = serde_json::from_str(r#""x""#).unwrap();
    assert_eq!(a, AgentRef("x".to_string()));
}

#[test]
fn cordyceps_tree_round_trips() {
    let tree = cordyceps_tree();
    let json = serde_json::to_string(&tree).unwrap();
    let back: LoopStrategy = serde_json::from_str(&json).unwrap();
    assert_eq!(tree, back);
}

#[test]
fn strategy_ref_built_in_and_custom_round_trip() {
    let built_in = StrategyRef::BuiltIn(cordyceps_tree());
    let custom = StrategyRef::Custom("my-harness::DoubleVerify".to_string());
    for r in [&built_in, &custom] {
        let json = serde_json::to_string(r).unwrap();
        let back: StrategyRef = serde_json::from_str(&json).unwrap();
        assert_eq!(*r, back, "round-trip failed for {json}");
    }
    // Adjacent tagging: kind/value, no collision with nested LoopStrategy kind.
    let json = serde_json::to_string(&custom).unwrap();
    assert_eq!(
        json,
        r#"{"kind":"custom","value":"my-harness::DoubleVerify"}"#
    );
    let bi = serde_json::to_string(&built_in).unwrap();
    assert!(bi.starts_with(r#"{"kind":"built_in","value":{"kind":"ralph""#));
}

// #124: every per-variant `run` body, driven without a wired
// `StrategyExecutor` (the scaffold-only context), returns a TYPED `Failed`
// outcome — never a panic / unimplemented / todo (CONVENTIONS). The real
// end-to-end behavior (with an executor) is exercised by the recursive
// executor tests below and the strategy integration tests.
#[test]
fn run_without_executor_is_typed_failure_not_panic() {
    let strategies = vec![
        LoopStrategy::ReAct(ReactConfig::per_loop(1)),
        LoopStrategy::PlanExecute(PlanExecuteConfig::simple(None)),
        LoopStrategy::SelfVerifying(SelfVerifyingConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            inner: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(1))),
            evaluator: SchemaRef("e".to_string()),
            eval_agent: None,
            eval_toolset: None,
        }),
        LoopStrategy::Ralph(RalphConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            inner: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(1))),
            agent: AgentRef("r".to_string()),
        }),
        LoopStrategy::HillClimbing(HillClimbingConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            inner: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(1))),
            direction: HillClimbingDirection::Minimize,
            max_stagnation: 1,
            revert_on_no_improvement: false,
            min_improvement_delta: 0.0,
            evaluator: AgentRef("m".to_string()),
        }),
    ];
    let rt = tokio::runtime::Runtime::new().unwrap();
    let registry = ExecutionRegistry::empty();
    for s in &strategies {
        let mut cx = ExecutionContext::new(&registry);
        cx.scratch.task = Some(Task::new(
            "x",
            SessionId::generate(),
            LoopStrategy::ReAct(ReactConfig::per_loop(1)),
        ));
        let outcome = rt.block_on(s.run(&mut cx));
        assert!(
            matches!(outcome, StrategyOutcome::Failed(_)),
            "expected typed Failed (no executor wired), got {outcome:?}"
        );
    }
}

fn paused_state_with(task: Task) -> PausedState {
    PausedState {
        session_id: SessionId::new("sess-1"),
        task_id: TaskId::new("task-1"),
        turn_number: 3,
        session_state: SessionState::default(),
        pending_tool_calls: vec![],
        approved_results: vec![],
        human_request: None,
        task,
        budget_used: BudgetSnapshot::default(),
        child_state: None,
        // #140: Ralph/PlanExecute root, no scoped handle → empty default.
        toolset: ToolsetRef::default(),
    }
}

fn cordyceps_task() -> Task {
    Task {
        id: TaskId::new("task-1"),
        instruction: "compose".to_string(),
        session_id: SessionId::new("sess-1"),
        budget: BudgetLimits::default(),
        loop_strategy: cordyceps_tree(),
    }
}

#[test]
fn paused_state_round_trips_with_cordyceps_strategy() {
    let ps = paused_state_with(cordyceps_task());
    let json = serde_json::to_string(&ps).unwrap();
    let back: PausedState = serde_json::from_str(&json).unwrap();
    assert_eq!(ps, back);
}

#[test]
fn child_paused_state_round_trips_with_cordyceps_strategy() {
    let cps = ChildPausedState {
        session_id: SessionId::new("sess-1"),
        task_id: TaskId::new("task-1"),
        turn_number: 2,
        session_state: SessionState::default(),
        pending_tool_calls: vec![],
        approved_results: vec![],
        human_request: None,
        task: cordyceps_task(),
        budget_used: BudgetSnapshot::default(),
        parent_tool_call_id: "call-1".to_string(),
        // #140: no scoped handle in this fixture → empty default.
        toolset: ToolsetRef::default(),
    };
    let json = serde_json::to_string(&cps).unwrap();
    let back: ChildPausedState = serde_json::from_str(&json).unwrap();
    assert_eq!(cps, back);
}

// ── Fixture replay (byte-identity ground truth for all four languages) ───

#[test]
fn fixture_replay_cordyceps_tree() {
    assert_fixture_byte_identical(&cordyceps_tree(), "strategy/cordyceps_tree.json");
    // And the fixture deserializes back to the canonical tree.
    let raw = std::fs::read_to_string(fixture_path("strategy/cordyceps_tree.json")).unwrap();
    let parsed: LoopStrategy = serde_json::from_str(&raw).unwrap();
    assert_eq!(parsed, cordyceps_tree());
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StrategyRefSuite {
    built_in: StrategyRef,
    custom: StrategyRef,
}

#[test]
fn fixture_replay_strategy_ref() {
    let raw = std::fs::read_to_string(fixture_path("strategy/strategy_ref.json")).unwrap();
    let suite: StrategyRefSuite = serde_json::from_str(&raw).unwrap();
    assert_eq!(suite.built_in, StrategyRef::BuiltIn(cordyceps_tree()));
    assert_eq!(
        suite.custom,
        StrategyRef::Custom("my-harness::DoubleVerify".to_string())
    );
    // Whole-suite byte identity (field order: built_in, custom).
    assert_fixture_byte_identical(&suite, "strategy/strategy_ref.json");
}

#[test]
fn fixture_replay_paused_state() {
    let raw = std::fs::read_to_string(fixture_path("strategy/paused_state.json")).unwrap();
    let parsed: PausedState = serde_json::from_str(&raw).unwrap();
    assert_eq!(parsed.task.loop_strategy, cordyceps_tree());
    assert_fixture_byte_identical(&parsed, "strategy/paused_state.json");
}

#[test]
fn fixture_replay_child_paused_state() {
    let raw = std::fs::read_to_string(fixture_path("strategy/child_paused_state.json")).unwrap();
    let parsed: ChildPausedState = serde_json::from_str(&raw).unwrap();
    assert_eq!(parsed.task.loop_strategy, cordyceps_tree());
    assert_fixture_byte_identical(&parsed, "strategy/child_paused_state.json");
}

/// Generator: emits the exact PausedState / ChildPausedState JSON so the
/// committed fixtures stay byte-identical with the Rust serializer. Run with
/// `cargo test -p spore-core gen_paused_fixtures -- --ignored --nocapture`.
#[test]
#[ignore]
fn gen_paused_fixtures() {
    let ps = paused_state_with(cordyceps_task());
    println!("PAUSED_STATE={}", serde_json::to_string(&ps).unwrap());
    let cps = ChildPausedState {
        session_id: SessionId::new("sess-1"),
        task_id: TaskId::new("task-1"),
        turn_number: 2,
        session_state: SessionState::default(),
        pending_tool_calls: vec![],
        approved_results: vec![],
        human_request: None,
        task: cordyceps_task(),
        budget_used: BudgetSnapshot::default(),
        parent_tool_call_id: "call-1".to_string(),
        // #140: no scoped handle in this fixture → empty default.
        toolset: ToolsetRef::default(),
    };
    println!(
        "CHILD_PAUSED_STATE={}",
        serde_json::to_string(&cps).unwrap()
    );
}

// ------------------------------------------------------------------
// #122: max_steps() advisory worst-case turn bound.
// ------------------------------------------------------------------

/// Build a ReAct leaf carrying an arbitrary budget policy.
fn react_with(budget: BudgetPolicy) -> LoopStrategy {
    LoopStrategy::ReAct(ReactConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        budget,
        agent: AgentRef(String::new()),
        toolset: ToolsetRef(String::new()),
        output: None,
        system_prompt: None,
    })
}

fn self_verifying(inner: LoopStrategy) -> LoopStrategy {
    LoopStrategy::SelfVerifying(SelfVerifyingConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(inner),
        evaluator: SchemaRef("ev".to_string()),
        eval_agent: None,
        eval_toolset: None,
    })
}

fn plan_execute(plan: LoopStrategy, execute: LoopStrategy) -> LoopStrategy {
    LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        plan: Box::new(plan),
        execute: Box::new(execute),
        plan_model: None,
    })
}

fn ralph(inner: LoopStrategy) -> LoopStrategy {
    LoopStrategy::Ralph(RalphConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(inner),
        agent: AgentRef("r".to_string()),
    })
}

fn hill_climbing(inner: LoopStrategy, max_stagnation: u32) -> LoopStrategy {
    LoopStrategy::HillClimbing(HillClimbingConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(inner),
        direction: HillClimbingDirection::Maximize,
        max_stagnation,
        revert_on_no_improvement: true,
        min_improvement_delta: 0.0,
        evaluator: AgentRef("m".to_string()),
    })
}

#[test]
fn max_steps_react_leaf() {
    assert_eq!(
        react_with(BudgetPolicy::PerLoop { value: 4 }).max_steps(),
        Some(4)
    );
    assert_eq!(
        react_with(BudgetPolicy::TotalSteps { value: 7 }).max_steps(),
        Some(7)
    );
    assert_eq!(
        react_with(BudgetPolicy::PerAttempt { value: 5 }).max_steps(),
        Some(5)
    );
    assert_eq!(react_with(BudgetPolicy::Unlimited).max_steps(), None);
}

#[test]
fn max_steps_self_verifying_adds_one() {
    let s = self_verifying(react_with(BudgetPolicy::PerLoop { value: 12 }));
    assert_eq!(s.max_steps(), Some(13));
}

#[test]
fn max_steps_plan_execute_is_per_task_sum() {
    let s = plan_execute(
        react_with(BudgetPolicy::PerLoop { value: 4 }),
        react_with(BudgetPolicy::PerLoop { value: 6 }),
    );
    assert_eq!(s.max_steps(), Some(10));
}

#[test]
fn max_steps_hill_climbing() {
    // inner=5, max_stagnation=2 ⇒ 5 * (2 + 1) = 15.
    let s = hill_climbing(react_with(BudgetPolicy::PerLoop { value: 5 }), 2);
    assert_eq!(s.max_steps(), Some(15));
    // u32::MAX sentinel ⇒ unbounded windows ⇒ None.
    let unbounded = hill_climbing(react_with(BudgetPolicy::PerLoop { value: 5 }), u32::MAX);
    assert_eq!(unbounded.max_steps(), None);
}

#[test]
fn max_steps_ralph_is_per_window() {
    assert_eq!(
        ralph(react_with(BudgetPolicy::PerLoop { value: 9 })).max_steps(),
        Some(9)
    );
    // Canonical cordyceps subtree wrapped in Ralph ⇒ per-window 25.
    let s = ralph(plan_execute(
        react_with(BudgetPolicy::PerLoop { value: 12 }),
        self_verifying(react_with(BudgetPolicy::PerLoop { value: 12 })),
    ));
    assert_eq!(s.max_steps(), Some(25));
}

#[test]
fn max_steps_canonical_cordyceps_subtree() {
    // PlanExecute[ReAct{12}, SelfVerifying[ReAct{12}]] = 12 + (12 + 1) = 25.
    let subtree = plan_execute(
        react_with(BudgetPolicy::PerLoop { value: 12 }),
        self_verifying(react_with(BudgetPolicy::PerLoop { value: 12 })),
    );
    assert_eq!(subtree.max_steps(), Some(25));
}

#[test]
fn max_steps_cordyceps_fixture_is_25() {
    // The whole tree's root Ralph wraps PlanExecute, so per-window == 25.
    let tree = cordyceps_tree();
    assert_eq!(tree.max_steps(), Some(25));

    let raw = std::fs::read_to_string(fixture_path("strategy/cordyceps_tree.json"))
        .expect("fixture present");
    let deserialized: LoopStrategy = serde_json::from_str(&raw).unwrap();
    assert_eq!(deserialized.max_steps(), Some(25));
}

#[test]
fn max_steps_unlimited_anywhere_is_none() {
    // Plan leaf unlimited ⇒ None.
    assert_eq!(
        plan_execute(
            react_with(BudgetPolicy::Unlimited),
            self_verifying(react_with(BudgetPolicy::PerLoop { value: 12 })),
        )
        .max_steps(),
        None
    );
    // Execute's inner ReAct unlimited ⇒ None.
    assert_eq!(
        plan_execute(
            react_with(BudgetPolicy::PerLoop { value: 4 }),
            self_verifying(react_with(BudgetPolicy::Unlimited)),
        )
        .max_steps(),
        None
    );
    // HillClimbing-wrapped unlimited inner ⇒ None (propagates to root).
    assert_eq!(
        ralph(hill_climbing(react_with(BudgetPolicy::Unlimited), 2)).max_steps(),
        None
    );
}

#[test]
fn max_steps_strategy_ref() {
    assert_eq!(StrategyRef::Custom("x".to_string()).max_steps(), None);
    assert_eq!(
        StrategyRef::BuiltIn(react_with(BudgetPolicy::PerLoop { value: 4 })).max_steps(),
        Some(4)
    );
}

#[test]
fn max_steps_overflow_is_none() {
    // inner = u32::MAX/2, max_stagnation = 3 ⇒ (u32::MAX/2) * 4 overflows ⇒ None.
    let s = hill_climbing(
        react_with(BudgetPolicy::PerLoop {
            value: u32::MAX / 2,
        }),
        3,
    );
    assert_eq!(s.max_steps(), None);
}
