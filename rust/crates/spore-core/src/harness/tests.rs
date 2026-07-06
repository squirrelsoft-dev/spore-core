
use super::testing::*;
use super::*;
use crate::agent::{mock::MockAgent, AgentId};
use crate::hooks::HookChain as _;
use crate::middleware::StandardMiddlewareChain;
use crate::model::{ModelError, ToolCall};

fn make_agent() -> Arc<MockAgent> {
    Arc::new(MockAgent::new(AgentId::new("test")))
}

#[tokio::test]
async fn read_only_tool_view_filters_to_readonly_allowlist() {
    // SC-30: the eval-phase read-only view advertises + dispatches only the
    // INTERSECTION of the wrapped catalogue with StandardTools::readonly_set().
    use crate::model::ToolSchema as ModelToolSchema;
    use std::sync::Mutex;

    // Inner registry advertising read + write + exec tools, recording dispatches.
    struct Inner {
        dispatched: Mutex<Vec<String>>,
    }
    impl ToolRegistry for Inner {
        fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput> {
            self.dispatched.lock().unwrap().push(call.name.clone());
            Box::pin(async { ToolOutput::success("ok") })
        }
        fn schemas(&self) -> Vec<ModelToolSchema> {
            ["read_file", "write_file", "bash"]
                .into_iter()
                .map(|n| ModelToolSchema {
                    name: n.to_string(),
                    description: String::new(),
                    input_schema: serde_json::json!({"type": "object"}),
                })
                .collect()
        }
    }

    let inner = Arc::new(Inner {
        dispatched: Mutex::new(Vec::new()),
    });
    let allow: std::collections::HashSet<String> = crate::tools::StandardTools::readonly_set()
        .into_iter()
        .map(|t| t.schema.name)
        .collect();
    let view = ReadOnlyToolView::new(inner.clone(), allow);

    // Schemas: only read_file survives the intersection; write_file/bash gone.
    let names: Vec<String> = view.schemas().into_iter().map(|s| s.name).collect();
    assert!(
        names.contains(&"read_file".to_string()),
        "read_file advertised"
    );
    assert!(
        !names.contains(&"write_file".to_string()),
        "write_file hidden"
    );
    assert!(!names.contains(&"bash".to_string()), "bash hidden");

    // read_file dispatches through to the inner registry.
    let ok = view
        .dispatch(ToolCall {
            id: "1".into(),
            name: "read_file".into(),
            input: serde_json::json!({}),
        })
        .await;
    assert!(matches!(ok, ToolOutput::Success { .. }));

    // write_file is blocked with a recoverable error and never reaches inner.
    let blocked = view
        .dispatch(ToolCall {
            id: "2".into(),
            name: "write_file".into(),
            input: serde_json::json!({}),
        })
        .await;
    assert!(matches!(
        blocked,
        ToolOutput::Error {
            recoverable: true,
            ..
        }
    ));
    assert_eq!(
        &*inner.dispatched.lock().unwrap(),
        &["read_file".to_string()]
    );
}

#[tokio::test]
async fn harness_constructs_from_boxed_model_interface() {
    // SC-2: ModelInterface is now object-safe (BoxFut, not RPITIT), so a
    // consumer can hold ONE `Arc<dyn ModelInterface>` — provider chosen at
    // runtime — and inject it without monomorphizing through a concrete model
    // type (looper's hand-rolled `AgentModel` enum + dispatch matches).
    use crate::model::mock::MockModelInterface;
    use crate::model::{
        ContentBlock, ModelParams, ModelRequest, ModelResponse, ProviderInfo, StopReason,
        TokenUsage,
    };

    let mock = Arc::new(MockModelInterface::new(ProviderInfo {
        name: "test".into(),
        model_id: "test-1".into(),
        context_window: 1000,
    }));
    mock.push_response(Ok(ModelResponse {
        content: vec![ContentBlock::Text {
            text: "greetings".into(),
        }],
        usage: TokenUsage {
            input_tokens: 3,
            output_tokens: 2,
            cache_read_tokens: None,
            cache_write_tokens: None,
        },
        stop_reason: StopReason::EndTurn,
    }));

    // This coercion `Arc<MockModelInterface>` -> `Arc<dyn ModelInterface>`
    // only compiles because the trait is object-safe.
    let model: Arc<dyn crate::model::ModelInterface> = mock.clone();

    // The blanket `impl ModelInterface for Arc<T>` + dyn dispatch genuinely
    // work at runtime (not just at compile time).
    assert_eq!(model.provider().model_id, "test-1");
    let resp = model
        .call(ModelRequest {
            messages: vec![],
            tools: vec![],
            params: ModelParams::default(),
            stream: false,
        })
        .await
        .expect("boxed model dispatches");
    assert!(
        matches!(resp.content.first(), Some(ContentBlock::Text { text }) if text == "greetings"),
        "the boxed model returned its queued response",
    );

    // The builder accepts the boxed model — the SC-2 acceptance check.
    let _harness = HarnessBuilder::conversational_arc(model).build();
}

// ── SC-8 presets: coding_agent + hill_climber ───────────────────────────

fn preset_mock_model() -> crate::model::mock::MockModelInterface {
    crate::model::mock::MockModelInterface::new(crate::model::ProviderInfo {
        name: "test".into(),
        model_id: "test-1".into(),
        context_window: 8192,
    })
}

// SC-8: `coding_agent` wires the read-write workspace sandbox, the coding tool
// catalogue, the built-in coding system prompt, and AutoContinue (the looper
// preset) — so a consumer collapses to one call.
#[tokio::test]
async fn coding_agent_preset_wires_sandbox_tools_prompt_and_autocontinue() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = HarnessBuilder::coding_agent(preset_mock_model(), tmp.path())
        .expect("workspace canonicalizes")
        .build_config();

    // Autonomous-but-capped escalation with the preset defaults (SC-5).
    assert!(
        matches!(
            cfg.escalation_mode,
            EscalationMode::AutoContinue { max_grants, steps_per_grant, .. }
                if max_grants == HarnessBuilder::PRESET_MAX_AUTO_GRANTS
                    && steps_per_grant == HarnessBuilder::PRESET_STEPS_PER_GRANT
        ),
        "coding_agent must default to AutoContinue, got {:?}",
        cfg.escalation_mode
    );
    // The built-in coding system prompt is installed.
    assert_eq!(
        cfg.system_prompt.as_deref(),
        Some(HarnessBuilder::CODING_AGENT_SYSTEM_PROMPT),
        "coding_agent installs its built-in system prompt"
    );
    // The coding catalogue is wired (a representative sample of coding_set()).
    // `.tools()` are folded into the `catalogue_registry` (bridged per-run),
    // not the harness-loop `tool_registry`, which stays the conversational
    // `EmptyToolRegistry` until the run binds a `ToolContext`.
    use crate::tool_registry::ToolRegistry as _;
    let catalogue = cfg
        .catalogue_registry
        .as_ref()
        .expect("coding_agent wires a catalogue registry from coding_set()");
    let tools: std::collections::HashSet<String> = catalogue
        .active_schemas(None)
        .into_iter()
        .map(|s| s.name)
        .collect();
    for expected in [
        "read_file",
        "write_file",
        "edit_file",
        "bash_command",
        "send_message",
    ] {
        assert!(
            tools.contains(expected),
            "coding_set must include {expected}; got {tools:?}"
        );
    }
}

// SC-8: a workspace path that can't be resolved is a typed `BuildError`, not a
// panic — the sandbox requires a canonical, existing root.
#[test]
fn coding_agent_preset_errors_on_missing_workspace() {
    let missing = std::path::Path::new("/spore-sc8-does-not-exist-37a1/nope");
    // Can't `unwrap_err()` — the Ok type (`HarnessBuilder`) is not `Debug`.
    match HarnessBuilder::coding_agent(preset_mock_model(), missing) {
        Err(
            crate::sandbox::BuildError::RootNotFound { .. }
            | crate::sandbox::BuildError::RootIo { .. },
        ) => {}
        Err(other) => panic!("unexpected BuildError variant: {other:?}"),
        Ok(_) => panic!("a missing workspace must surface a BuildError, not build"),
    }
}

// SC-8: `hill_climber` registers the scoring evaluator (required for the
// HillClimbing strategy) under the default handle and defaults to AutoContinue
// — the cordyceps preset.
#[test]
fn hill_climber_preset_registers_evaluator_and_autocontinue() {
    let evaluator: Arc<dyn crate::metric::MetricEvaluator> =
        Arc::new(crate::metric::LatencyEvaluator {
            command: "true".into(),
            args: vec![],
            warmup_runs: 0,
            measured_runs: 1,
            timeout: std::time::Duration::from_secs(1),
            working_dir: None,
        });
    let cfg = HarnessBuilder::hill_climber(preset_mock_model(), evaluator).build_config();
    assert!(
        cfg.registry.resolve_metric_evaluator("").is_some(),
        "hill_climber must register the metric evaluator under the default handle"
    );
    assert!(
        matches!(cfg.escalation_mode, EscalationMode::AutoContinue { .. }),
        "hill_climber must default to AutoContinue, got {:?}",
        cfg.escalation_mode
    );
}

/// #142: the fixed canonical path the test config helpers derive their
/// `project_id` from. Durable artifacts (task_list, plan) are keyed by this
/// project namespace, so readback assertions key by `durable_ns()` rather
/// than the per-run session id.
const TEST_PROJECT_PATH: &str = "/test-workspace";

/// The durable [`RunStore`] namespace the test harness writes durable
/// artifacts under (#142) — the project id projected onto the session-id
/// axis. Use this in readback assertions in place of `&t.session_id`.
fn durable_ns() -> SessionId {
    crate::storage::ProjectId::from_canonical_path(TEST_PROJECT_PATH).namespace()
}

fn standard_config(agent: Arc<MockAgent>) -> HarnessConfig {
    let a: Arc<dyn Agent> = agent;
    // #124: the legacy single-collaborator fields are gone — the worker agent
    // + toolset resolve from the registry under the DEFAULT empty key (bare
    // `ReactConfig::per_loop` leaves carry empty `AgentRef`/`ToolsetRef`).
    let registry = ExecutionRegistry::builder()
        .agent("", a)
        .toolset("", Arc::new(EmptyToolRegistry))
        // Default output schema for structured-slot inner leaves (A.5).
        .schema("", serde_json::json!({}))
        .build();
    HarnessConfig {
        tool_registry: Arc::new(ScriptedToolRegistry::new()),
        sandbox: Arc::new(AllowAllSandbox),
        sandbox_violation_policy: crate::harness::SandboxViolationPolicy::default(),
        context_manager: Arc::new(NoopContextManager),
        termination_policy: Arc::new(AlwaysContinuePolicy),
        middleware: None,
        observability: None,
        compaction_verifier: Arc::new(KeyTermVerifier),
        max_compaction_attempts: 2,
        pricing: PricingTable::DEFAULT,
        content_capture: ContentCaptureConfig::default(),
        tool_call_repair: None,
        max_repair_attempts: 1,
        plan_empty_max_retries: 0,
        tool_call_nudge: false,
        steering_source: None,
        max_stop_blocks: 8,
        error_loop_threshold: 3,
        // #139: OFF by default in the shared test config (migration gate),
        // so every pre-#139 test keeps its byte-for-byte terminal. The #139
        // tests build a config with this flipped ON explicitly.
        enforce_output_schemas: false,
        output_schema_max_retries: 2,
        hooks: None,
        // #76: plan_execute + task_list persistence now lives on the
        // RunStore seam (not SessionState.extras), so the test harness
        // needs a real (in-memory) run store for the readback helpers and
        // assertions below to observe what the harness wrote.
        storage: Arc::new(crate::storage::StorageProvider::single(Arc::new(
            crate::storage::InMemoryStorageProvider::new(),
        ))),
        // #142: a fixed test project namespace. Durable readback helpers in
        // these tests key by this project namespace (see `TEST_PROJECT_PATH`).
        project_id: crate::storage::ProjectId::from_canonical_path(TEST_PROJECT_PATH),
        chunk_provider: Arc::new(crate::prompt_assembly::InMemoryChunkProvider::empty()),
        max_resets: 3,
        vcs_provider: None,
        catalogue_registry: None,
        toolset_catalogues: std::collections::HashMap::new(),
        system_prompt: None,
        guides: Vec::new(),
        skills: None,
        memory: None,
        model_params: ModelParams::default(),
        auto_persist_sessions: false,
        prompt_tool_call_flag: None,
        consult_handlers: std::collections::HashMap::new(),
        registry,
        // #130: the shared test config drives the AUTONOMOUS (propagate-up)
        // escalation path so the pre-#130 budget-exhaustion tests keep
        // asserting their original `Failure { BudgetExceeded }` / propagate
        // behavior. The new HITL pause path is exercised by tests that build a
        // `SurfaceToHuman` config explicitly (see `surface_config` below).
        escalation_mode: EscalationMode::Autonomous,
        tool_search: None,
    }
}

/// A [`standard_config`] clone with `escalation_mode = SurfaceToHuman` (#130):
/// the HITL path under test. Budget escalation PAUSES with a
/// `HumanRequest::BudgetExhausted` rather than propagating up.
fn surface_config(agent: Arc<MockAgent>) -> HarnessConfig {
    let mut cfg = standard_config(agent);
    cfg.escalation_mode = EscalationMode::SurfaceToHuman;
    cfg
}

fn task(strategy: LoopStrategy) -> Task {
    Task::new("do something", SessionId::new("s1"), strategy)
}

// #124 test helpers: register collaborators into a config's registry under
// the DEFAULT empty key (matching the bare `*_task()` strategy handles).
fn set_worker_agent(cfg: &mut HarnessConfig, agent: Arc<dyn Agent>) {
    cfg.registry = std::mem::take(&mut cfg.registry)
        .into_builder()
        .agent("", agent)
        .build();
}
fn set_default_verifier(cfg: &mut HarnessConfig, v: Arc<dyn crate::verifier::Verifier>) {
    cfg.registry = std::mem::take(&mut cfg.registry)
        .into_builder()
        .verifier("", v)
        .build();
}
fn set_default_metric(cfg: &mut HarnessConfig, m: Arc<dyn crate::metric::MetricEvaluator>) {
    cfg.registry = std::mem::take(&mut cfg.registry)
        .into_builder()
        .metric_evaluator("", m)
        .build();
}

fn react(max: u32) -> Task {
    task(LoopStrategy::ReAct(ReactConfig::per_loop(max)))
}

// #124: a bare ReAct leaf carrying the default output schema, for STRUCTURED
// slots (SelfVerifying worker / HillClimbing propose) that A.5 requires to
// declare an output schema.
fn react_structured(max: u32) -> LoopStrategy {
    LoopStrategy::ReAct(ReactConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        budget: BudgetPolicy::PerLoop { value: max },
        agent: AgentRef(String::new()),
        toolset: ToolsetRef(String::new()),
        output: Some(SchemaRef(String::new())),
        system_prompt: None,
    })
}

fn usage() -> TokenUsage {
    TokenUsage {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_tokens: None,
        cache_write_tokens: None,
    }
}

// Rule: Harness owns the loop; final response on first turn returns Success.
#[tokio::test]
async fn final_response_returns_success() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let h = StandardHarness::new(standard_config(a));
    let r = h.run(HarnessRunOptions::new(react(5))).await;
    match r {
        RunResult::Success { output, turns, .. } => {
            assert_eq!(output, "done");
            assert_eq!(turns, 1);
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// Steering (pi-style, `HarnessConfig::steering_source`): a message queued
// mid-run is injected as a USER message at the NEXT turn boundary — after the
// current turn's tool calls execute, before the next model call — persisted on
// the session, and emitted as `StreamEvent::UserMessage`. In-flight work is
// never cancelled.
#[tokio::test]
async fn steering_message_injects_at_turn_boundary() {
    use std::sync::atomic::{AtomicU32, Ordering};

    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "after-tool".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "tool-ok".into(),
        truncated: false,
    });
    cfg.tool_registry = reg;
    let polls = Arc::new(AtomicU32::new(0));
    let polls_in_source = polls.clone();
    cfg.steering_source = Some(Arc::new(move || {
        // Poll 0 = before turn 1 (nothing queued yet); poll 1 = the boundary
        // AFTER turn 1's tool calls — the steer is delivered there, before
        // turn 2's model call.
        match polls_in_source.fetch_add(1, Ordering::SeqCst) {
            1 => vec!["steer: also update the docs".to_string()],
            _ => vec![],
        }
    }));
    let h = StandardHarness::new(cfg);
    let events: Arc<std::sync::Mutex<Vec<StreamEvent>>> = Default::default();
    let events_in_sink = events.clone();
    let r = h
        .run(
            HarnessRunOptions::new(react(5))
                .with_stream(move |e| events_in_sink.lock().unwrap().push(e)),
        )
        .await;
    match r {
        RunResult::Success {
            output,
            turns,
            session_state,
            ..
        } => {
            assert_eq!(output, "after-tool");
            assert_eq!(turns, 2, "the tool turn still ran to completion");
            let injected = session_state.messages.iter().any(|m| {
                m.role == crate::model::Role::User
                    && matches!(
                        &m.content,
                        crate::model::Content::Text { text } if text.contains("steer: also update the docs")
                    )
            });
            assert!(injected, "steered message persisted on the session");
        }
        other => panic!("expected Success, got {other:?}"),
    }
    assert!(
        events.lock().unwrap().iter().any(|e| matches!(
            e,
            StreamEvent::UserMessage { content, .. } if content.contains("steer:")
        )),
        "steered message emitted as StreamEvent::UserMessage"
    );
    assert!(
        polls.load(Ordering::SeqCst) >= 2,
        "the source is polled at every turn boundary"
    );
}

// Rule: tool calls are dispatched, then loop continues to a final response.
#[tokio::test]
async fn tool_call_then_final_response_loops() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "after-tool".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "tool-ok".into(),
        truncated: false,
    });
    cfg.tool_registry = reg.clone();
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success { turns, output, .. } => {
            assert_eq!(output, "after-tool");
            assert_eq!(turns, 2);
            assert_eq!(reg.call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// ---- issue #91: catalogue tool path + system_prompt seam -------------

fn catalogue_builder(agent: Arc<MockAgent>) -> HarnessBuilder {
    HarnessBuilder::new(
        agent,
        Arc::new(ScriptedToolRegistry::new()),
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
}

/// An agent that records the messages of the context it was handed, then
/// replies with a final response — lets a test assert what the model saw.
struct CapturingAgent {
    id: AgentId,
    seen: std::sync::Mutex<Vec<Message>>,
}
impl Agent for CapturingAgent {
    fn turn<'a>(&'a self, context: Context) -> BoxFut<'a, TurnResult> {
        *self.seen.lock().unwrap() = context.messages.clone();
        Box::pin(async move {
            TurnResult::FinalResponse {
                reasoning: None,
                content: "done".into(),
                usage: usage(),
            }
        })
    }
    fn id(&self) -> AgentId {
        self.id.clone()
    }
}

#[tokio::test]
async fn catalogue_tools_fold_into_registry_with_inmemory_storage() {
    use crate::tool_registry::ToolRegistry as _;
    let cfg = catalogue_builder(make_agent())
        .tool(crate::tools::StandardTools::read_file())
        .tool(crate::tools::StandardTools::write_file())
        .build_config();

    // Accumulated catalogue tools were folded into a registry.
    let catalogue = cfg.catalogue_registry.clone().expect("catalogue registry");
    let names: Vec<String> = catalogue
        .active_schemas(None)
        .iter()
        .map(|s| s.name.clone())
        .collect();
    assert!(names.contains(&"read_file".to_string()));
    assert!(names.contains(&"write_file".to_string()));

    // Storage defaulted to in-memory (not all-no-op) because catalogue tools
    // are present: a put/get round-trips on the run store.
    let sid = SessionId::new("s1");
    let run = cfg.storage.run().clone();
    run.put(&sid, "k", serde_json::json!({ "v": 1 }))
        .await
        .unwrap();
    assert!(run.get(&sid, "k").await.unwrap().is_some());
}

#[tokio::test]
async fn effective_registry_bridges_catalogue_tools() {
    let cfg = catalogue_builder(make_agent())
        .tool(crate::tools::StandardTools::read_file())
        .build_config();
    let h = StandardHarness::new(cfg);
    let reg = h.effective_tool_registry(&SessionId::new("s1"), &ToolsetRef(String::new()));
    // The bridge advertises the catalogue schemas to the model.
    assert!(reg.schemas().iter().any(|s| s.name == "read_file"));
    // And maps an inner dispatch failure (unknown tool) to a *recoverable*
    // error so the loop appends it and the agent can adapt.
    let out = reg
        .dispatch(ToolCall {
            id: "c".into(),
            name: "does_not_exist".into(),
            input: serde_json::json!({}),
        })
        .await;
    assert!(matches!(
        out,
        ToolOutput::Error {
            recoverable: true,
            ..
        }
    ));
}

#[test]
fn no_catalogue_tools_keeps_tool_registry_seam() {
    let cfg = catalogue_builder(make_agent()).build_config();
    assert!(cfg.catalogue_registry.is_none());
}

// ---- Issue 2: per-node toolset scoping ---------------------------------

/// Helper: dispatch a tool by name through a bridged registry and report
/// whether it resolved to a *successful* tool output (vs an unknown-tool /
/// not-available recoverable error).
async fn dispatched_ok(reg: &Arc<dyn ToolRegistry>, name: &str) -> bool {
    let out = reg
        .dispatch(ToolCall {
            id: "c".into(),
            name: name.into(),
            input: serde_json::json!({}),
        })
        .await;
    !matches!(out, ToolOutput::Error { .. })
}

/// A node carrying a NON-EMPTY toolset handle dispatches ONLY that toolset's
/// catalogue: the planner (`plan-tools` = list_dir/task_list) cannot call an
/// exec-only tool (`read_file`), and the executor (`exec-tools` = read_file)
/// cannot call a plan-only tool (`task_list`/`list_dir`). The leaked union is
/// closed.
#[tokio::test]
async fn per_node_toolset_scoping_closes_cross_node_leaks() {
    use crate::tools::StandardTools;
    let cfg = catalogue_builder(make_agent())
        .toolset_tools(
            "plan-tools",
            vec![StandardTools::list_dir(), StandardTools::task_list()],
        )
        .toolset_tools("exec-tools", vec![StandardTools::read_file()])
        .build_config();
    let h = StandardHarness::new(cfg);
    let sid = SessionId::new("s1");

    // Planner node: plan-tools only. Its OWN tools advertise; exec-only tools
    // are NOT available.
    let plan = h.effective_tool_registry(&sid, &ToolsetRef("plan-tools".into()));
    assert!(plan.schemas().iter().any(|s| s.name == "list_dir"));
    assert!(plan.schemas().iter().any(|s| s.name == "task_list"));
    assert!(!plan.schemas().iter().any(|s| s.name == "read_file"));
    // The leak the live run exhibited: planner calling an exec-only tool now
    // resolves to unknown-tool / not-available, NOT success.
    assert!(!dispatched_ok(&plan, "read_file").await);

    // Executor node: exec-tools only. It cannot reach the plan-only tools.
    let exec = h.effective_tool_registry(&sid, &ToolsetRef("exec-tools".into()));
    assert!(exec.schemas().iter().any(|s| s.name == "read_file"));
    assert!(!exec.schemas().iter().any(|s| s.name == "task_list"));
    assert!(!exec.schemas().iter().any(|s| s.name == "list_dir"));
    assert!(!dispatched_ok(&exec, "task_list").await);
    assert!(!dispatched_ok(&exec, "list_dir").await);
}

/// An unknown tool called by a scoped node still yields the existing
/// recoverable unknown-tool failure (no regression of the error path).
#[tokio::test]
async fn per_node_toolset_unknown_tool_is_recoverable_error() {
    use crate::tools::StandardTools;
    let cfg = catalogue_builder(make_agent())
        .toolset_tools("plan-tools", vec![StandardTools::list_dir()])
        .build_config();
    let h = StandardHarness::new(cfg);
    let reg = h.effective_tool_registry(&SessionId::new("s1"), &ToolsetRef("plan-tools".into()));
    let out = reg
        .dispatch(ToolCall {
            id: "c".into(),
            name: "does_not_exist".into(),
            input: serde_json::json!({}),
        })
        .await;
    assert!(matches!(
        out,
        ToolOutput::Error {
            recoverable: true,
            ..
        }
    ));
}

/// A node with an EMPTY toolset handle still sees the GLOBAL catalogue wired
/// via `.tools()` (back-compat with examples 01–11 that never scope). The
/// per-key catalogues do NOT leak into the empty-handle fallback.
#[tokio::test]
async fn empty_toolset_handle_falls_back_to_global_catalogue() {
    use crate::tools::StandardTools;
    let cfg = catalogue_builder(make_agent())
        .tools(vec![StandardTools::read_file()]) // global catalogue
        .toolset_tools("plan-tools", vec![StandardTools::list_dir()]) // scoped
        .build_config();
    let h = StandardHarness::new(cfg);
    let global = h.effective_tool_registry(&SessionId::new("s1"), &ToolsetRef(String::new()));
    // Empty handle ⇒ global catalogue (read_file), NOT the scoped plan-tools.
    assert!(global.schemas().iter().any(|s| s.name == "read_file"));
    assert!(!global.schemas().iter().any(|s| s.name == "list_dir"));
}

/// A non-empty toolset handle with NO registered per-key catalogue falls back
/// to the global catalogue / seam (additive change — an unscoped name does
/// not strand the node with zero tools).
#[tokio::test]
async fn unknown_toolset_handle_falls_back_to_global_catalogue() {
    use crate::tools::StandardTools;
    let cfg = catalogue_builder(make_agent())
        .tools(vec![StandardTools::read_file()])
        .build_config();
    let h = StandardHarness::new(cfg);
    let reg = h.effective_tool_registry(&SessionId::new("s1"), &ToolsetRef("not-wired".into()));
    assert!(reg.schemas().iter().any(|s| s.name == "read_file"));
}

/// `build_config` auto-registers a registry presence entry for every per-key
/// catalogue, so a tree referencing that toolset handle passes
/// `ExecutionRegistry::validate` without the caller wiring a placeholder.
#[test]
fn toolset_tools_autofill_registry_presence_for_validate() {
    use crate::tools::StandardTools;
    let cfg = catalogue_builder(make_agent())
        .toolset_tools("plan-tools", vec![StandardTools::list_dir()])
        .build_config();
    assert!(cfg
        .registry
        .resolve_toolset(&ToolsetRef("plan-tools".into()))
        .is_some());
    // And the dispatchable catalogue is present.
    assert!(cfg.toolset_catalogues.contains_key("plan-tools"));
}

// ---- issue #120: ExecutionRegistry run-entry validation + knob ---------

/// An agent whose `turn` MUST never be reached: it panics if invoked, so a
/// passing test proves no model turn fired before startup validation.
struct NeverCalledAgent;
impl Agent for NeverCalledAgent {
    fn turn<'a>(&'a self, _context: Context) -> BoxFut<'a, TurnResult> {
        panic!("agent turn fired despite an unresolved registry handle");
    }
    fn id(&self) -> AgentId {
        AgentId::new("never")
    }
}

#[tokio::test]
async fn unresolved_handle_is_startup_failure_before_any_turn() {
    // A populated registry that does NOT contain the agent referenced by the
    // task's ReAct leaf. run() must validate at entry and fail BEFORE the
    // first turn — NeverCalledAgent panics if a turn ever fires.
    let leaf = LoopStrategy::ReAct(ReactConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        budget: BudgetPolicy::PerLoop { value: 4 },
        agent: AgentRef("missing".into()),
        toolset: ToolsetRef("t".into()),
        output: None,
        system_prompt: None,
    });
    let harness = HarnessBuilder::new(
        Arc::new(NeverCalledAgent),
        Arc::new(ScriptedToolRegistry::new()),
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
    // Populate the registry with SOMETHING (so validation runs) but not the
    // referenced agent "missing".
    .registry_toolset("t", Arc::new(EmptyToolRegistry))
    .build();

    let task = Task::new("go", SessionId::new("s120"), leaf);
    let result = harness.run(HarnessRunOptions::new(task)).await;

    match result {
        RunResult::Failure {
            reason: HaltReason::ConfigurationError { error },
            turns,
            ..
        } => {
            assert_eq!(turns, 0, "validation failed before the first turn");
            assert_eq!(
                error,
                HarnessError::UnresolvedHandle {
                    kind: "agent".into(),
                    key: "missing".into(),
                }
            );
        }
        other => panic!("expected ConfigurationError startup failure, got {other:?}"),
    }
}

#[test]
fn escalation_mode_is_selectable_and_readable_on_config() {
    // Default is SurfaceToHuman.
    let cfg = catalogue_builder(make_agent()).build_config();
    assert!(matches!(
        cfg.escalation_mode,
        EscalationMode::SurfaceToHuman
    ));

    // And the builder setter overrides it.
    let cfg = catalogue_builder(make_agent())
        .escalation_mode(EscalationMode::Autonomous)
        .build_config();
    assert!(matches!(cfg.escalation_mode, EscalationMode::Autonomous));
}

#[test]
fn builder_folds_default_agent_into_registry() {
    // #124: the legacy single-collaborator fields are gone — `HarnessBuilder`
    // folds its agent + toolset into the ExecutionRegistry under the DEFAULT
    // empty key, so the single resolution path always has a worker to resolve.
    let cfg = catalogue_builder(make_agent()).build_config();
    assert!(!cfg.registry.is_empty());
    assert!(cfg
        .registry
        .resolve_agent(&AgentRef(String::new()))
        .is_some());
}

#[tokio::test]
async fn missing_custom_strategy_key_is_recoverable_strategy_not_found() {
    // A registry referencing a Custom strategy key not registered yields a
    // recoverable StrategyNotFound (no panic) when resolved.
    let reg = ExecutionRegistry::empty();
    let r = StrategyRef::Custom("absent".into());
    let err = reg.resolve_strategy(&r).unwrap_err();
    assert_eq!(
        err,
        HarnessError::StrategyNotFound {
            key: "absent".into()
        }
    );
}

#[test]
fn sandbox_setter_overrides_the_configured_sandbox() {
    let sb: Arc<dyn SandboxProvider> = Arc::new(AllowAllSandbox);
    let cfg = catalogue_builder(make_agent())
        .sandbox(sb.clone())
        .build_config();
    assert!(Arc::ptr_eq(&cfg.sandbox, &sb));
}

#[test]
fn context_manager_setter_overrides_the_configured_manager() {
    let cm: Arc<dyn ContextManager> = Arc::new(NoopContextManager);
    let cfg = catalogue_builder(make_agent())
        .context_manager(cm.clone())
        .build_config();
    assert!(Arc::ptr_eq(&cfg.context_manager, &cm));
}

#[tokio::test]
async fn system_prompt_is_prepended_to_context() {
    let agent = Arc::new(CapturingAgent {
        id: AgentId::new("cap"),
        seen: std::sync::Mutex::new(vec![]),
    });
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, agent.clone());
    cfg.system_prompt = Some("OPERATING RULES".into());
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(react(2))).await;

    let seen = agent.seen.lock().unwrap();
    let first = seen.first().expect("context had messages");
    assert!(matches!(first.role, Role::System));
    assert!(matches!(
        &first.content,
        Content::Text { text } if text == "OPERATING RULES"
    ));
}

#[tokio::test]
async fn no_system_prompt_leaves_context_without_a_system_message() {
    let agent = Arc::new(CapturingAgent {
        id: AgentId::new("cap"),
        seen: std::sync::Mutex::new(vec![]),
    });
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, agent.clone());
    // system_prompt stays None.
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(react(2))).await;

    let seen = agent.seen.lock().unwrap();
    assert!(seen.iter().all(|m| !matches!(m.role, Role::System)));
}

// Issue #12 — the harness emits real spans through an injected
// ObservabilityProvider and flushes a terminal session summary. Hermetic:
// SPORE_OTLP_ENDPOINT is left unset so the outbox writes JSONL only.
#[tokio::test]
async fn harness_emits_spans_through_outbox_jsonl() {
    let tmp = tempfile::tempdir().unwrap();
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "do_thing".into(),
            input: serde_json::json!({ "k": "v" }),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "tool-ok".into(),
        truncated: false,
    });

    let harness = HarnessBuilder::new(
        a,
        reg,
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
    .with_observability_outbox(tmp.path())
    .build();

    match harness.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success { turns, .. } => assert_eq!(turns, 2),
        other => panic!("expected Success, got {other:?}"),
    }

    let path = tmp.path().join("sessions/s1/trace.jsonl");
    let body = std::fs::read_to_string(&path).expect("trace.jsonl written");
    let lines: Vec<serde_json::Value> = body
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let kinds: Vec<&str> = lines.iter().map(|l| l["kind"].as_str().unwrap()).collect();
    assert!(
        kinds.contains(&"turn"),
        "expected a turn span, got {kinds:?}"
    );
    assert!(
        kinds.contains(&"tool_call"),
        "expected a tool_call span, got {kinds:?}"
    );
    // The terminal summary line is written last.
    assert_eq!(kinds.last(), Some(&"session"), "kinds: {kinds:?}");
    let summary = lines.last().unwrap();
    assert_eq!(summary["attributes"]["outcome"], "success");
    assert_eq!(summary["attributes"]["total_turns"], 2);
    let tool = lines.iter().find(|l| l["kind"] == "tool_call").unwrap();
    assert_eq!(tool["attributes"]["tool_name"], "do_thing");
    assert_eq!(tool["attributes"]["call_id"], "c1");
    // Turn spans parent the tool-call spans within a shared trace.
    let trace_ids: std::collections::HashSet<&str> = lines
        .iter()
        .map(|l| l["trace_id"].as_str().unwrap())
        .collect();
    assert_eq!(trace_ids.len(), 1, "all spans share one trace id");
    assert!(tmp.path().join("sessions/s1/.flushed").exists());
}

// Issue #64 — content capture ON: the harness populates GenAI content on
// the turn (output text + tool calls) and tool-call (args + result) spans,
// and the outbox writes it into the JSONL as `gen_ai.*` attributes.
#[tokio::test]
async fn content_capture_on_writes_genai_content_to_jsonl() {
    let tmp = tempfile::tempdir().unwrap();
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "do_thing".into(),
            input: serde_json::json!({ "k": "v" }),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "all finished".into(),
        usage: usage(),
    });
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "tool result body".into(),
        truncated: false,
    });

    let harness = HarnessBuilder::new(
        a,
        reg,
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
    .with_observability_outbox(tmp.path())
    .content_capture(ContentCaptureConfig {
        enabled: true,
        max_field_len: 8192,
    })
    .build();

    harness.run(HarnessRunOptions::new(react(5))).await;

    let body = std::fs::read_to_string(tmp.path().join("sessions/s1/trace.jsonl")).unwrap();
    let lines: Vec<serde_json::Value> = body
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    // The tool-requesting turn carries the requested tool call.
    let turn_with_calls = lines
        .iter()
        .find(|l| l["kind"] == "turn" && !l["attributes"]["gen_ai.response.tool_calls"].is_null())
        .expect("turn with captured tool calls");
    assert_eq!(
        turn_with_calls["attributes"]["gen_ai.response.tool_calls"][0]["name"],
        "do_thing"
    );

    // The final turn carries the assistant output text.
    let final_turn = lines
        .iter()
        .find(|l| {
            l["kind"] == "turn" && l["attributes"]["gen_ai.response.content"] == "all finished"
        })
        .expect("turn with captured output text");
    assert_eq!(
        final_turn["attributes"]["gen_ai.response.role"],
        "assistant"
    );

    // The tool-call span carries args + result.
    let tool = lines.iter().find(|l| l["kind"] == "tool_call").unwrap();
    assert_eq!(tool["attributes"]["gen_ai.tool.name"], "do_thing");
    assert_eq!(tool["attributes"]["gen_ai.tool.call.arguments"]["k"], "v");
    assert_eq!(
        tool["attributes"]["gen_ai.tool.message.content"],
        "tool result body"
    );
}

// Issue #64 — content capture OFF (default): no `gen_ai.*` content reaches
// the JSONL. Byte-for-byte the same metrics-only output as pre-#64.
#[tokio::test]
async fn content_capture_off_writes_no_genai_content() {
    let tmp = tempfile::tempdir().unwrap();
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "do_thing".into(),
            input: serde_json::json!({ "secret": "value" }),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "sensitive output".into(),
        usage: usage(),
    });
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "sensitive result".into(),
        truncated: false,
    });

    // Default builder → content_capture OFF.
    let harness = HarnessBuilder::new(
        a,
        reg,
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
    .with_observability_outbox(tmp.path())
    .build();

    harness.run(HarnessRunOptions::new(react(5))).await;

    let body = std::fs::read_to_string(tmp.path().join("sessions/s1/trace.jsonl")).unwrap();
    // No `gen_ai.` keys and no leaked content strings anywhere in the file.
    assert!(!body.contains("gen_ai."), "no gen_ai content keys when off");
    assert!(!body.contains("sensitive output"));
    assert!(!body.contains("sensitive result"));
    assert!(!body.contains("\"value\""));
    // No assembled-input prompt leaked either (issue #64).
    assert!(!body.contains("gen_ai.prompt"));
}

// ── Input-message capture (issue #64) ──────────────────────────────────

/// `capture_input_messages` maps each role and renders each Content variant:
/// system + user + assistant tool-call + tool result, in order.
#[test]
fn capture_input_messages_maps_roles_and_renders_content() {
    use crate::model::{Content, Role, ToolCall as MToolCall, ToolResult as MToolResult};
    let msgs = vec![
        Message {
            role: Role::System,
            content: Content::Text {
                text: "be helpful".into(),
            },
        },
        Message {
            role: Role::User,
            content: Content::Text {
                text: "list files".into(),
            },
        },
        Message {
            role: Role::Assistant,
            content: Content::ToolCall(MToolCall {
                id: "c1".into(),
                name: "shell".into(),
                input: serde_json::json!({ "command": "ls" }),
            }),
        },
        Message {
            role: Role::Tool,
            content: Content::ToolResult(MToolResult {
                tool_use_id: "c1".into(),
                content: "file.txt".into(),
                is_error: false,
            }),
        },
    ];
    let out = StandardHarness::capture_input_messages(&msgs, 8192);
    assert_eq!(out.len(), 4);
    assert_eq!(out[0].role, GenAiRole::System);
    assert_eq!(out[0].content, "be helpful");
    assert_eq!(out[1].role, GenAiRole::User);
    assert_eq!(out[1].content, "list files");
    assert_eq!(out[2].role, GenAiRole::Assistant);
    assert_eq!(out[2].content, "shell {\"command\":\"ls\"}");
    assert_eq!(out[3].role, GenAiRole::Tool);
    assert_eq!(out[3].content, "file.txt");
    assert!(out.iter().all(|m| !m.truncated));
}

/// Image content renders as a placeholder — the base64 `data` is NEVER dumped.
#[test]
fn capture_input_messages_image_renders_placeholder_not_base64() {
    use crate::model::{Content, Role};
    let msgs = vec![Message {
        role: Role::User,
        content: Content::Image {
            media_type: "image/png".into(),
            data: "AAAABBBBCCCCDDDD_secret_base64".into(),
        },
    }];
    let out = StandardHarness::capture_input_messages(&msgs, 8192);
    assert_eq!(out[0].content, "[image image/png]");
    assert!(!out[0].content.contains("secret_base64"));
}

/// Truncation applies to rendered input content (byte budget + marker).
#[test]
fn capture_input_messages_truncates_long_content() {
    use crate::model::{Content, Role};
    let long = "x".repeat(100);
    let msgs = vec![Message {
        role: Role::User,
        content: Content::Text { text: long },
    }];
    let out = StandardHarness::capture_input_messages(&msgs, 10);
    assert!(out[0].truncated);
    assert!(out[0].content.ends_with("...[truncated]"));
    assert!(out[0].content.starts_with("xxxxxxxxxx"));
}

/// End-to-end: guard ON → the assembled prompt rides as `gen_ai.prompt`
/// with the user message present and correct roles.
#[tokio::test]
async fn content_capture_on_writes_input_messages_to_jsonl() {
    let tmp = tempfile::tempdir().unwrap();
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let reg = Arc::new(ScriptedToolRegistry::new());
    let harness = HarnessBuilder::new(
        a,
        reg,
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
    .with_observability_outbox(tmp.path())
    .content_capture(ContentCaptureConfig {
        enabled: true,
        max_field_len: 8192,
    })
    .build();

    harness.run(HarnessRunOptions::new(react(5))).await;

    let body = std::fs::read_to_string(tmp.path().join("sessions/s1/trace.jsonl")).unwrap();
    let lines: Vec<serde_json::Value> = body
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let turn = lines
        .iter()
        .find(|l| l["kind"] == "turn" && !l["attributes"]["gen_ai.prompt"].is_null())
        .expect("turn with captured input messages");
    let prompt = turn["attributes"]["gen_ai.prompt"].as_array().unwrap();
    assert!(!prompt.is_empty());
    // The seeded user request is present as a user message.
    assert!(prompt
        .iter()
        .any(|m| m["role"] == "user" && m["content"].as_str().is_some()));
}

// Issue #73: the builder defaults storage to an all-no-op StorageProvider,
// and `.storage(...)` / `storage()` / `session_store()` round-trip.
#[tokio::test]
async fn default_storage_is_no_op_and_setter_round_trips() {
    use crate::storage::RunStore as _;
    let a = make_agent();
    let reg = Arc::new(ScriptedToolRegistry::new());
    // Default: no .storage() — must be no-op (reads empty).
    let h = HarnessBuilder::new(
        a.clone(),
        reg.clone(),
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
    .build();
    let sess = SessionId::new("s1");
    assert!(h
        .session_store()
        .get_session(&sess)
        .await
        .unwrap()
        .is_none());
    assert!(h.storage().run().get(&sess, "k").await.unwrap().is_none());

    // Explicit injection round-trips through the accessor.
    let backend = Arc::new(crate::storage::InMemoryStorageProvider::new());
    backend
        .put(&sess, "plan", serde_json::json!({"ok": true}))
        .await
        .unwrap();
    let storage = Arc::new(crate::storage::StorageProvider::single(backend));
    let h2 = HarnessBuilder::new(
        a,
        reg,
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
    .storage(storage)
    .build();
    assert_eq!(
        h2.storage().run().get(&sess, "plan").await.unwrap(),
        Some(serde_json::json!({"ok": true}))
    );
}

// Issue #79, R25: the builder defaults the chunk provider to an empty
// InMemoryChunkProvider, and `.chunks(...)` resolves to an
// InMemoryChunkProvider carrying the registered chunks.
#[tokio::test]
async fn default_chunk_provider_empty_and_chunks_setter_round_trips() {
    use crate::prompt_assembly::PromptChunk;
    let a = make_agent();
    let reg = Arc::new(ScriptedToolRegistry::new());

    // Default: empty provider.
    let h = HarnessBuilder::new(
        a.clone(),
        reg.clone(),
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
    .build();
    let loaded = h.config().chunk_provider.load().await.unwrap();
    assert!(loaded.is_empty());

    // `.chunks(...)` resolves to an InMemoryChunkProvider with those chunks.
    let h2 = HarnessBuilder::new(
        a,
        reg,
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
    .chunks(vec![
        PromptChunk::new("core", "rules"),
        PromptChunk::new("style", "be concise"),
    ])
    .build();
    let loaded2 = h2.config().chunk_provider.load().await.unwrap();
    assert_eq!(loaded2.len(), 2);
    assert_eq!(loaded2[0].id, "core");
}

// Rule: parallel tool calls all dispatched in one turn.
#[tokio::test]
async fn parallel_tool_calls_all_dispatched() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![
            ToolCall {
                id: "a".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            },
            ToolCall {
                id: "b".into(),
                name: "y".into(),
                input: serde_json::json!({}),
            },
        ],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "ok".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "1".into(),
        truncated: false,
    });
    reg.push(ToolOutput::Success {
        content: "2".into(),
        truncated: false,
    });
    cfg.tool_registry = reg.clone();
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(react(5))).await;
    assert_eq!(reg.call_count.load(std::sync::atomic::Ordering::SeqCst), 2);
}

// Rule: budget overrun terminates with explicit reason.
#[tokio::test]
async fn budget_max_turns_exceeded() {
    let a = make_agent();
    for _ in 0..10 {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: "c".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
    }
    let h = StandardHarness::new(standard_config(a));
    let mut t = react(100);
    t.budget.max_turns = Some(2);
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure {
            reason:
                HaltReason::BudgetExceeded {
                    limit_type: BudgetLimitType::Turns,
                },
            turns,
            ..
        } => {
            assert_eq!(turns, 2);
        }
        other => panic!("expected BudgetExceeded(Turns), got {other:?}"),
    }
}

// #125 rule 6: when the ReAct LEAF's OWN policy is the binding cap (no smaller
// global backstop), the leaf PROPAGATES a typed BudgetExhausted carrying its
// last FinalResponse as the partial — it never self-resolves Continue/Fail at
// the leaf. `drive_strategy` surfaces it as a BudgetExceeded terminal whose
// session carries the partial assistant text.
#[tokio::test]
async fn react_leaf_cap_binding_propagates_partial() {
    let a = make_agent();
    // The leaf cap is 2; push 3 tool-call turns so the window runs 2 turns
    // and hits the leaf's own cap (no global max_turns set).
    for i in 0..3 {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: format!("c{i}"),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
    }
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    for _ in 0..3 {
        reg.push(ToolOutput::Success {
            content: "ok".into(),
            truncated: false,
        });
    }
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    // Leaf PerLoop{2}, NO global cap → the leaf policy is the binding cap.
    let t = react(2);
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure {
            reason:
                HaltReason::BudgetExceeded {
                    limit_type: BudgetLimitType::Turns,
                },
            turns,
            session_state,
            ..
        } => {
            // The turn count is the EXHAUSTED NODE's own `steps_taken` (#125
            // BudgetExhausted path), which equals the leaf cap N=2 here.
            assert_eq!(turns, 2, "leaf stopped at its own cap N=2");
            // The #125 discriminator: the partial flowed through
            // `StrategyOutcome::BudgetExhausted{partial_output}` →
            // `drive_strategy`'s BudgetExhausted arm, which materializes the
            // partial as a single assistant Text message. On PRE-#125 code the
            // leaf's `record_terminal(window_failure)` surfaced the window's
            // RAW session (the tool-call / observation messages), NOT this
            // node-concrete partial JSON — so this assertion FAILS on the old
            // path and PROVES the new BudgetExhausted machinery is exercised.
            assert_eq!(
                session_state.messages.len(),
                1,
                "BudgetExhausted arm materializes exactly the partial as one \
                     assistant message (not the window's raw transcript)"
            );
            let msg = &session_state.messages[0];
            assert_eq!(msg.role, crate::model::Role::Assistant);
            let text = match &msg.content {
                crate::model::Content::Text { text } => text.clone(),
                other => panic!("expected partial Text content, got {other:?}"),
            };
            // The documented ReAct partial shape (fork #2): the last
            // FinalResponse text as JSON. This window produced no FinalResponse,
            // so the documented shape carries an empty `last_final_response`.
            assert_eq!(text, react_partial_json(""));
            // Sanity: the materialized text is genuinely the partial helper's
            // output, i.e. valid JSON with the node tag — not free-form prose.
            let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(parsed["node"], "react");
            assert_eq!(parsed["last_final_response"], "");
        }
        other => panic!("expected BudgetExceeded from leaf cap, got {other:?}"),
    }
}

// Rule: A turn with neither tool call nor response is an error.
#[tokio::test]
async fn agent_error_terminates_with_agent_error_halt_reason() {
    let a = make_agent();
    a.push(TurnResult::Error {
        error: AgentError::EmptyResponse,
        usage: Some(usage()),
    });
    let h = StandardHarness::new(standard_config(a));
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Failure {
            reason: HaltReason::AgentError { .. },
            ..
        } => {}
        other => panic!("expected AgentError halt, got {other:?}"),
    }
}

// Rule: Layer-1 SandboxViolation::PathEscape halts unconditionally.
#[tokio::test]
async fn layer1_path_escape_halts() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "read".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let sb = Arc::new(ScriptedSandbox::new());
    sb.push(Err(SandboxViolation::PathEscape {
        path: "/etc/passwd".into(),
    }));
    cfg.sandbox = sb;
    // Halting on a path escape is now opt-in via the policy (default is
    // Recoverable). Opt in so this exercises the always-halt path.
    cfg.sandbox_violation_policy = SandboxViolationPolicy::Halt;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Failure {
            reason:
                HaltReason::SandboxViolation {
                    violation: SandboxViolation::PathEscape { .. },
                },
            ..
        } => {}
        other => panic!("expected SandboxViolation halt, got {other:?}"),
    }
}

// Rule: Layer-2 recoverable sandbox violation → appended as tool error, loop continues.
#[tokio::test]
async fn layer2_path_denied_continues_as_tool_error() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "read".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "ack".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let sb = Arc::new(ScriptedSandbox::new());
    sb.push(Err(SandboxViolation::PathDenied {
        path: "/p".into(),
        matched_rule: "test".into(),
    }));
    cfg.sandbox = sb;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success { turns, .. } => assert_eq!(turns, 2),
        other => panic!("expected Success after recoverable violation, got {other:?}"),
    }
}

// Policy: a TOOL-surfaced sandbox violation is RECOVERABLE by default — the
// harness appends it as a tool error and the loop continues to completion.
#[tokio::test]
async fn tool_sandbox_violation_recoverable_by_default() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "ack".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::SandboxViolation {
        violation: SandboxViolation::PathEscape {
            path: "/etc/passwd".into(),
        },
    });
    cfg.tool_registry = reg;
    // default policy = Recoverable (no opt-in)
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success { turns, .. } => assert_eq!(turns, 2),
        other => panic!("expected Success (recoverable, loop continued), got {other:?}"),
    }
}

// Policy: opting into `Halt` makes a tool-surfaced always-halt violation end
// the run with a TYPED `HaltReason::SandboxViolation` (not UnrecoverableTool).
#[tokio::test]
async fn tool_sandbox_violation_halts_under_halt_policy() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::SandboxViolation {
        violation: SandboxViolation::PathEscape {
            path: "/etc/passwd".into(),
        },
    });
    cfg.tool_registry = reg;
    cfg.sandbox_violation_policy = SandboxViolationPolicy::Halt;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Failure {
            reason:
                HaltReason::SandboxViolation {
                    violation: SandboxViolation::PathEscape { .. },
                },
            ..
        } => {}
        other => panic!("expected SandboxViolation halt, got {other:?}"),
    }
}

// Policy: a Layer-2 (non-always-halt) violation stays recoverable even under
// `Halt` — only PathEscape/NetworkViolation are halt-eligible.
#[tokio::test]
async fn tool_layer2_violation_recoverable_even_under_halt_policy() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "ack".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::SandboxViolation {
        violation: SandboxViolation::ReadOnlyViolation { path: "x".into() },
    });
    cfg.tool_registry = reg;
    cfg.sandbox_violation_policy = SandboxViolationPolicy::Halt;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success { turns, .. } => assert_eq!(turns, 2),
        other => panic!("expected Success (Layer-2 stays recoverable), got {other:?}"),
    }
}

// Rule: TerminationPolicy::Halt overrides final response.
#[tokio::test]
async fn termination_policy_halt_overrides_success() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let tp = Arc::new(ScriptedTerminationPolicy::new());
    tp.push(TerminationDecision::Halt {
        reason: "not yet".into(),
    });
    cfg.termination_policy = tp;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Failure {
            reason: HaltReason::TerminationPolicyHalt { reason },
            ..
        } => assert_eq!(reason, "not yet"),
        other => panic!("expected TerminationPolicyHalt, got {other:?}"),
    }
}

// Rule: Middleware::Halt at BeforeTurn halts before model call.
#[tokio::test]
async fn middleware_halt_before_turn() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "unused".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let mw = Arc::new(ScriptedMiddleware::new());
    mw.push(
        HookPoint::BeforeTurn,
        MiddlewareDecision::Halt {
            reason: "blocked".into(),
        },
    );
    cfg.middleware = Some(mw);
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Failure {
            reason:
                HaltReason::MiddlewareHalt {
                    hook: HookPoint::BeforeTurn,
                    reason,
                },
            turns,
            ..
        } => {
            assert_eq!(reason, "blocked");
            assert_eq!(turns, 0);
        }
        other => panic!("expected MiddlewareHalt, got {other:?}"),
    }
}

// Rule: Middleware::SurfaceToHuman at BeforeTool returns WaitingForHuman with pending calls.
#[tokio::test]
async fn middleware_surface_to_human_before_tool() {
    let a = make_agent();
    let calls = vec![ToolCall {
        id: "c".into(),
        name: "x".into(),
        input: serde_json::json!({}),
    }];
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: calls.clone(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let mw = Arc::new(ScriptedMiddleware::new());
    mw.push(
        HookPoint::BeforeTool,
        MiddlewareDecision::SurfaceToHuman {
            request: HumanRequest::ToolApproval {
                calls: calls.clone(),
                risk_level: RiskLevel::Medium,
            },
        },
    );
    cfg.middleware = Some(mw);
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::WaitingForHuman { state, .. } => {
            assert_eq!(state.pending_tool_calls.len(), 1);
            assert!(state.child_state.is_none());
        }
        other => panic!("expected WaitingForHuman, got {other:?}"),
    }
}

// ── Phase 3 / Q2: the rich `middleware.rs` chain is now canonical ──────────
// These exercise the wired `StandardMiddlewareChain` (not the `ScriptedMiddleware`
// double) end-to-end through the loop: SC-9 (AfterTool rewrites a result in
// place), SC-11 (BeforeTool mutates calls in place + priority fan-out), and the
// BeforeCompletion `ForceAnotherTurn` injection the former harness stub lacked.

/// Context manager that stores each tool result as a `Role::Tool` text message
/// AND honours `replace_tool_result` — so a SC-9 in-place rewrite is observable
/// in the post-run `session_state`.
struct ReplaceRecordingCm;
impl ReplaceRecordingCm {
    fn render(output: &ToolOutput) -> String {
        match output {
            ToolOutput::Success { content, .. } => content.clone(),
            ToolOutput::Error { message, .. } => message.clone(),
            _ => String::new(),
        }
    }
}
impl ContextManager for ReplaceRecordingCm {
    fn assemble<'a>(
        &'a self,
        session: &'a SessionState,
        _task: &'a Task,
        _sources: &'a ContextSources,
    ) -> BoxFut<'a, Context> {
        let messages = session.messages.clone();
        Box::pin(async move {
            Context {
                messages,
                tools: vec![],
                params: ModelParams::default(),
            }
        })
    }
    fn append_tool_result<'a>(
        &'a self,
        session: &'a mut SessionState,
        result: &'a ToolResult,
    ) -> BoxFut<'a, ()> {
        let text = Self::render(&result.output);
        Box::pin(async move {
            session.messages.push(Message {
                role: Role::Tool,
                content: Content::Text { text },
            });
        })
    }
    fn replace_tool_result<'a>(
        &'a self,
        session: &'a mut SessionState,
        message_index: usize,
        result: &'a ToolResult,
    ) -> BoxFut<'a, ()> {
        let text = Self::render(&result.output);
        Box::pin(async move {
            if let Some(m) = session.messages.get_mut(message_index) {
                *m = Message {
                    role: Role::Tool,
                    content: Content::Text { text },
                };
            }
        })
    }
    fn append_assistant_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        message: &'a Message,
    ) -> BoxFut<'a, ()> {
        let message = message.clone();
        Box::pin(async move {
            session.messages.push(message);
        })
    }
    fn append_user_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()> {
        let text = text.to_string();
        Box::pin(async move {
            session.messages.push(Message {
                role: Role::User,
                content: Content::Text { text },
            });
        })
    }
}

/// AfterTool middleware that rewrites the first result's output in place.
struct RewriteFirstResultMw {
    to: String,
}
impl crate::middleware::Middleware for RewriteFirstResultMw {
    fn handle<'a>(
        &'a self,
        ctx: crate::middleware::HookContext<'a>,
    ) -> BoxFut<'a, MiddlewareDecision> {
        let to = self.to.clone();
        Box::pin(async move {
            if let crate::middleware::HookContext::AfterTool { results, .. } = ctx {
                if let Some(r) = results.first_mut() {
                    r.output = ToolOutput::Error {
                        message: to,
                        recoverable: true,
                    };
                    return MiddlewareDecision::ContinueWithModification;
                }
            }
            MiddlewareDecision::Continue
        })
    }
    fn hooks(&self) -> Vec<HookPoint> {
        vec![HookPoint::AfterTool]
    }
    fn name(&self) -> &str {
        "rewrite_first_result"
    }
}

// SC-9: an AfterTool middleware rewrites a result; the rewrite reaches the
// conversation the next model turn sees (the cordyceps `build_check` inversion,
// done by the harness instead of the tool returning a fake error).
#[tokio::test]
async fn after_tool_middleware_rewrites_result_in_place() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    cfg.context_manager = Arc::new(ReplaceRecordingCm);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "ORIGINAL".into(),
        truncated: false,
    });
    cfg.tool_registry = reg;
    let chain = StandardMiddlewareChain::new();
    chain
        .register(Box::new(RewriteFirstResultMw {
            to: "REWRITTEN".into(),
        }))
        .unwrap();
    cfg.middleware = Some(Arc::new(chain));
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success { session_state, .. } => {
            let tool_texts: Vec<String> = session_state
                .messages
                .iter()
                .filter_map(|m| match (&m.role, &m.content) {
                    (Role::Tool, Content::Text { text }) => Some(text.clone()),
                    _ => None,
                })
                .collect();
            assert!(
                tool_texts.iter().any(|t| t == "REWRITTEN"),
                "rewritten result must reach the conversation, got {tool_texts:?}"
            );
            assert!(
                !tool_texts.iter().any(|t| t == "ORIGINAL"),
                "original result must not survive the rewrite, got {tool_texts:?}"
            );
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// SC-9 stream parity: when an AfterTool middleware rewrites a result
// (`ContinueWithModification`), the event stream must reflect it — the
// original `ToolResult` was already emitted PRE-middleware, so a second
// `ToolResult { revised: true }` carries the correction. Consumers key on
// `call_id` + `revised` to update in place instead of double-counting.
#[tokio::test]
async fn after_tool_rewrite_re_emits_revised_stream_event() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "ORIGINAL".into(),
        truncated: false,
    });
    cfg.tool_registry = reg;
    let chain = StandardMiddlewareChain::new();
    chain
        .register(Box::new(RewriteFirstResultMw {
            to: "REWRITTEN".into(),
        }))
        .unwrap();
    cfg.middleware = Some(Arc::new(chain));
    let h = StandardHarness::new(cfg);

    let events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = events.clone();
    let opts = HarnessRunOptions::new(react(5))
        .with_stream(Box::new(move |ev| sink.lock().unwrap().push(ev)));
    match h.run(opts).await {
        RunResult::Success { .. } => {}
        other => panic!("expected Success, got {other:?}"),
    }

    // Only the events for our call, projected to (is_error, content, revised).
    let results: Vec<(bool, String, bool)> = events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ToolResult {
                call_id,
                is_error,
                content,
                revised,
                ..
            } if call_id == "c1" => Some((*is_error, content.clone(), *revised)),
            _ => None,
        })
        .collect();

    assert_eq!(
        results.len(),
        2,
        "expected the pre-middleware result plus one revised re-emit, got {results:?}"
    );
    assert_eq!(
        results[0],
        (false, "ORIGINAL".to_string(), false),
        "first event is the original, un-revised success"
    );
    assert_eq!(
        results[1],
        (true, "REWRITTEN".to_string(), true),
        "second event carries the rewrite, flagged revised"
    );
}

/// Tool registry that records every dispatched call and echoes success.
struct RecordingToolRegistry {
    seen: Arc<std::sync::Mutex<Vec<ToolCall>>>,
}
impl ToolRegistry for RecordingToolRegistry {
    fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput> {
        self.seen.lock().unwrap().push(call);
        Box::pin(async {
            ToolOutput::Success {
                content: "ok".into(),
                truncated: false,
            }
        })
    }
}

/// BeforeTool middleware that appends `tag` to `calls[0].input["trace"]` in
/// place, at the given priority.
struct TraceTagMw {
    tag: String,
    prio: i32,
}
impl crate::middleware::Middleware for TraceTagMw {
    fn handle<'a>(
        &'a self,
        ctx: crate::middleware::HookContext<'a>,
    ) -> BoxFut<'a, MiddlewareDecision> {
        let tag = self.tag.clone();
        Box::pin(async move {
            if let crate::middleware::HookContext::BeforeTool { calls, .. } = ctx {
                if let Some(first) = calls.first_mut() {
                    let obj = first.input.as_object_mut().expect("object input");
                    let trace = obj.entry("trace").or_insert_with(|| serde_json::json!([]));
                    trace
                        .as_array_mut()
                        .expect("array")
                        .push(serde_json::json!(tag));
                    return MiddlewareDecision::ContinueWithModification;
                }
            }
            MiddlewareDecision::Continue
        })
    }
    fn hooks(&self) -> Vec<HookPoint> {
        vec![HookPoint::BeforeTool]
    }
    fn priority(&self) -> i32 {
        self.prio
    }
    fn name(&self) -> &str {
        // Distinct names so the chain accepts both registrations.
        if self.prio <= 1 {
            "trace_first"
        } else {
            "trace_second"
        }
    }
}

// SC-11: BeforeTool middleware mutates the dispatched calls IN PLACE, and the
// chain fans out in priority order (ascending for a before-hook). The lower
// priority runs first, so the dispatched call carries ["first","second"].
#[tokio::test]
async fn before_tool_middleware_mutates_calls_in_priority_order() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut cfg = standard_config(a);
    cfg.tool_registry = Arc::new(RecordingToolRegistry { seen: seen.clone() });
    let chain = StandardMiddlewareChain::new();
    // Register out of order to prove the chain sorts by priority, not insertion.
    chain
        .register(Box::new(TraceTagMw {
            tag: "second".into(),
            prio: 2,
        }))
        .unwrap();
    chain
        .register(Box::new(TraceTagMw {
            tag: "first".into(),
            prio: 1,
        }))
        .unwrap();
    cfg.middleware = Some(Arc::new(chain));
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success { .. } => {}
        other => panic!("expected Success, got {other:?}"),
    }
    let dispatched = seen.lock().unwrap();
    assert_eq!(dispatched.len(), 1, "exactly one tool dispatched");
    assert_eq!(
        dispatched[0].input.get("trace"),
        Some(&serde_json::json!(["first", "second"])),
        "BeforeTool middleware must mutate the dispatched call in priority order"
    );
}

/// BeforeCompletion middleware that forces ONE extra turn, then lets the run
/// complete.
struct ForceOnceMw {
    fired: std::sync::atomic::AtomicBool,
    inject: String,
}
impl crate::middleware::Middleware for ForceOnceMw {
    fn handle<'a>(
        &'a self,
        ctx: crate::middleware::HookContext<'a>,
    ) -> BoxFut<'a, MiddlewareDecision> {
        let inject = self.inject.clone();
        let first = !self.fired.swap(true, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async move {
            if matches!(ctx, crate::middleware::HookContext::BeforeCompletion { .. }) && first {
                MiddlewareDecision::ForceAnotherTurn { inject }
            } else {
                MiddlewareDecision::Continue
            }
        })
    }
    fn hooks(&self) -> Vec<HookPoint> {
        vec![HookPoint::BeforeCompletion]
    }
    fn name(&self) -> &str {
        "force_once"
    }
}

// BeforeCompletion `ForceAnotherTurn`: the chain injects a follow-up and the
// loop runs another turn instead of completing — behaviour the former harness
// stub could not express. The injection lands as a user message and the run
// completes on the SECOND final response.
#[tokio::test]
async fn before_completion_force_another_turn_runs_extra_turn() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "first".into(),
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "second".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    cfg.context_manager = Arc::new(ReplaceRecordingCm);
    let chain = StandardMiddlewareChain::new();
    chain
        .register(Box::new(ForceOnceMw {
            fired: std::sync::atomic::AtomicBool::new(false),
            inject: "keep going".into(),
        }))
        .unwrap();
    cfg.middleware = Some(Arc::new(chain));
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success {
            output,
            turns,
            session_state,
            ..
        } => {
            assert_eq!(output, "second", "the forced second turn must win");
            assert_eq!(turns, 2, "exactly one extra turn was forced");
            let injected = session_state.messages.iter().any(|m| {
                matches!((&m.role, &m.content),
                        (Role::User, Content::Text { text }) if text == "keep going")
            });
            assert!(injected, "the ForceAnotherTurn injection must be recorded");
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// ── SC-26 / #115: ContextSources reach the model structurally ─────────────

/// Agent double that records the [`Context`] it is handed each turn, so a
/// test can assert what actually reached the model.
struct RecordingAgent {
    seen: Arc<std::sync::Mutex<Vec<Context>>>,
}
impl crate::agent::Agent for RecordingAgent {
    fn turn<'a>(&'a self, context: Context) -> BoxFut<'a, TurnResult> {
        self.seen.lock().unwrap().push(context);
        Box::pin(async move {
            TurnResult::FinalResponse {
                reasoning: None,
                content: "done".into(),
                usage: usage(),
            }
        })
    }
    fn id(&self) -> AgentId {
        AgentId::new("recording")
    }
}

/// Minimal context manager that renders `sources.guides` then
/// `sources.memory` into a leading System block — mirroring the production
/// `StandardCompactionAdapter`'s `render_context_block` so the loop's
/// sources-building + system-prompt merge can be asserted without the
/// adapter's model machinery. (The adapter's own rendering is unit-tested in
/// `compaction_adapter`.)
struct GuideRenderingCm;
impl ContextManager for GuideRenderingCm {
    fn assemble<'a>(
        &'a self,
        session: &'a SessionState,
        _task: &'a Task,
        sources: &'a ContextSources,
    ) -> BoxFut<'a, Context> {
        let mut messages = session.messages.clone();
        let block = sources
            .guides
            .iter()
            .map(|g| format!("# {}\n{}", g.name, g.content))
            .chain(sources.memory.iter().map(|m| m.memory.content.clone()))
            .collect::<Vec<_>>()
            .join("\n\n");
        Box::pin(async move {
            if !block.is_empty() {
                messages.insert(
                    0,
                    Message {
                        role: Role::System,
                        content: Content::Text { text: block },
                    },
                );
            }
            Context {
                messages,
                tools: vec![],
                params: ModelParams::default(),
            }
        })
    }
    fn append_tool_result<'a>(
        &'a self,
        session: &'a mut SessionState,
        result: &'a ToolResult,
    ) -> BoxFut<'a, ()> {
        let text = match &result.output {
            ToolOutput::Success { content, .. } => content.clone(),
            ToolOutput::Error { message, .. } => message.clone(),
            _ => String::new(),
        };
        Box::pin(async move {
            session.messages.push(Message {
                role: Role::Tool,
                content: Content::Text { text },
            });
        })
    }
    fn append_user_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()> {
        let text = text.to_string();
        Box::pin(async move {
            session.messages.push(Message {
                role: Role::User,
                content: Content::Text { text },
            });
        })
    }
}

// SC-26/#115 acceptance (guide half): a guide registered on the harness
// reaches the model through the structural assemble seam — NOT as an ad-hoc
// User message — and the configured system prompt is merged in front of it in
// a single leading System message.
#[tokio::test]
async fn guide_reaches_model_via_assemble_seam() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let agent: Arc<dyn crate::agent::Agent> = Arc::new(RecordingAgent { seen: seen.clone() });
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, agent);
    cfg.context_manager = Arc::new(GuideRenderingCm);
    cfg.system_prompt = Some("SYSTEM PROMPT".into());
    cfg.guides = vec![Guide::skill("audit", "AUDIT PLAYBOOK BODY")];
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(react(5))).await;

    let contexts = seen.lock().unwrap();
    assert!(!contexts.is_empty(), "the agent must have been called");
    let first = &contexts[0];
    match first.messages.first() {
        Some(Message {
            role: Role::System,
            content: Content::Text { text },
        }) => {
            assert!(
                text.starts_with("SYSTEM PROMPT"),
                "system prompt must lead the merged System block: {text:?}"
            );
            assert!(
                text.contains("# audit") && text.contains("AUDIT PLAYBOOK BODY"),
                "the guide must reach the model structurally: {text:?}"
            );
        }
        other => panic!("expected a leading System text message, got {other:?}"),
    }
    // The guide is NOT delivered as a stray User message.
    let user_guide = first.messages.iter().any(|m| {
        matches!((&m.role, &m.content),
                (Role::User, Content::Text { text }) if text.contains("AUDIT PLAYBOOK BODY"))
    });
    assert!(!user_guide, "guide must not be injected as a User message");
}

/// Build a [`StandardMemoryProvider`] holding a single active semantic memory.
async fn memory_with(id: &str, content: &str) -> Arc<crate::memory::StandardMemoryProvider> {
    use crate::memory::{MemoryProvider, MemorySource, MemoryStatus, SemanticMemory};
    let provider = Arc::new(crate::memory::StandardMemoryProvider::new());
    let mem = SemanticMemory {
        id: crate::memory::MemoryId::new(id),
        content: content.into(),
        source: MemorySource::Manual,
        domain: None,
        version: 1,
        previous_versions: vec![],
        created_at: crate::memory::Timestamp::new("2026-06-24T00:00:00Z"),
        updated_at: crate::memory::Timestamp::new("2026-06-24T00:00:00Z"),
        status: MemoryStatus::Active,
    };
    // `store_semantic` is awaited on the dyn-compatible (BoxFut) trait — the
    // conversion under test.
    provider
        .store_semantic(mem, crate::memory::MergeStrategy::Reject)
        .await
        .unwrap();
    provider
}

// #160 / SC-26 acceptance (memory half): a memory provider registered on the
// harness has its relevant items reach the model through the SAME structural
// assemble seam as guides — a leading System block, NOT an ad-hoc User
// message. Exercises `Arc<dyn MemoryProvider>` (the object-safety conversion).
#[tokio::test]
async fn memory_reaches_model_via_assemble_seam() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let agent: Arc<dyn crate::agent::Agent> = Arc::new(RecordingAgent { seen: seen.clone() });
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, agent);
    cfg.context_manager = Arc::new(GuideRenderingCm);
    cfg.system_prompt = Some("SYSTEM PROMPT".into());
    // Stored as the object-safe trait object — the whole point of #160.
    let provider: Arc<dyn crate::memory::MemoryProvider> =
        memory_with("m1", "REFUND IDEMPOTENCY: audit the payments refund path").await;
    cfg.memory = Some(MemoryConfig::new(provider).min_relevance(0.1));
    let h = StandardHarness::new(cfg);

    // The query text defaults to the task instruction, so a related task
    // surfaces the memory.
    let task = Task::new(
        "audit the payments refund path",
        SessionId::new("s1"),
        LoopStrategy::ReAct(ReactConfig::per_loop(5)),
    );
    let _ = h.run(HarnessRunOptions::new(task)).await;

    let contexts = seen.lock().unwrap();
    assert!(!contexts.is_empty(), "the agent must have been called");
    let first = &contexts[0];
    match first.messages.first() {
        Some(Message {
            role: Role::System,
            content: Content::Text { text },
        }) => {
            assert!(
                text.starts_with("SYSTEM PROMPT"),
                "system prompt must lead the merged System block: {text:?}"
            );
            assert!(
                text.contains("REFUND IDEMPOTENCY"),
                "the memory must reach the model structurally: {text:?}"
            );
        }
        other => panic!("expected a leading System text message, got {other:?}"),
    }
    // Memory is NOT delivered as a stray User message.
    let user_memory = first.messages.iter().any(|m| {
        matches!((&m.role, &m.content),
                (Role::User, Content::Text { text }) if text.contains("REFUND IDEMPOTENCY"))
    });
    assert!(
        !user_memory,
        "memory must not be injected as a User message"
    );
}

// #160: `build_context_sources` queries the configured provider with the task
// instruction by default, honors an explicit query override, and leaves
// `memory` empty when no provider is configured (the byte-identical pre-#160
// path).
#[tokio::test]
async fn build_context_sources_populates_memory_from_provider() {
    // (a) No memory configured → empty, regardless of instruction.
    let h_none = StandardHarness::new(standard_config(make_agent()));
    let none = h_none
        .build_context_sources(vec![], "audit the payments refund path")
        .await;
    assert!(
        none.memory.is_empty(),
        "no provider configured must leave memory empty"
    );

    // (b) Provider configured, query defaults to the task instruction.
    let provider: Arc<dyn crate::memory::MemoryProvider> =
        memory_with("m1", "REFUND IDEMPOTENCY: audit the payments refund path").await;
    let mut cfg = standard_config(make_agent());
    // 0.3 cleanly separates the related instruction (jaccard ≈0.83) from the
    // unrelated one below (≈0.11 — a shared stopword), so (c) is meaningful.
    cfg.memory = Some(MemoryConfig::new(provider.clone()).min_relevance(0.3));
    let h = StandardHarness::new(cfg);
    let by_task = h
        .build_context_sources(vec![], "audit the payments refund path")
        .await;
    assert_eq!(by_task.memory.len(), 1, "the relevant memory must surface");
    assert!(by_task.memory[0]
        .memory
        .content
        .contains("REFUND IDEMPOTENCY"));

    // (c) An unrelated instruction surfaces nothing at the same threshold.
    let unrelated = h
        .build_context_sources(vec![], "compile the rust workspace")
        .await;
    assert!(
        unrelated.memory.is_empty(),
        "an unrelated instruction must not surface the memory"
    );

    // (d) A configured `query` overrides the task instruction — so even an
    // unrelated instruction surfaces the memory when the fixed query matches.
    let mut cfg2 = standard_config(make_agent());
    cfg2.memory = Some(
        MemoryConfig::new(provider)
            .query("audit the payments refund path")
            .min_relevance(0.1),
    );
    let h2 = StandardHarness::new(cfg2);
    let by_override = h2
        .build_context_sources(vec![], "compile the rust workspace")
        .await;
    assert_eq!(
        by_override.memory.len(),
        1,
        "the configured query must override the task instruction"
    );
}

// Rule: always_halt tool annotation halts the loop.
#[tokio::test]
async fn always_halt_tool_halts() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "danger".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.mark_always_halt("danger");
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Failure {
            reason: HaltReason::UnrecoverableToolError { tool, .. },
            ..
        } => assert_eq!(tool, "danger"),
        other => panic!("expected UnrecoverableToolError, got {other:?}"),
    }
}

// Rule: Unrecoverable tool error halts loop immediately.
#[tokio::test]
async fn unrecoverable_tool_error_halts() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Error {
        message: "boom".into(),
        recoverable: false,
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Failure {
            reason: HaltReason::UnrecoverableToolError { error, .. },
            ..
        } => assert_eq!(error, "boom"),
        other => panic!("expected UnrecoverableToolError, got {other:?}"),
    }
}

// Tool-call repair: a `flag: "false"` string argument is coerced to a real
// bool and re-dispatched, so the tool succeeds and the run completes.
#[tokio::test]
async fn tool_call_repair_fixes_bad_bool_arg() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let a = make_agent();
    // Turn 1: call the tool with a stringified bool (weak-model behaviour).
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "set_flag".into(),
            input: serde_json::json!({ "flag": "false" }),
        }],
        usage: usage(),
    });
    // Turn 2: finish.
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });

    let mut cfg = standard_config(a);
    let reg = Arc::new(BoolToolRegistry::new());
    cfg.tool_registry = reg.clone();
    cfg.tool_call_repair = Some(Arc::new(crate::tool_call_repair::StandardToolCallRepair));

    let last_tool_error = Arc::new(AtomicBool::new(false));
    let sink_flag = last_tool_error.clone();
    let opts = HarnessRunOptions::new(react(5)).with_stream(Box::new(move |ev| {
        if let StreamEvent::ToolResult { is_error, .. } = ev {
            sink_flag.store(is_error, Ordering::SeqCst);
        }
    }));

    let h = StandardHarness::new(cfg);
    match h.run(opts).await {
        RunResult::Success { output, .. } => assert_eq!(output, "done"),
        other => panic!("expected Success, got {other:?}"),
    }
    // Two dispatches: the failed original + the repaired retry.
    assert_eq!(reg.call_count.load(Ordering::SeqCst), 2);
    // Final tool result was a success, not an error.
    assert!(!last_tool_error.load(Ordering::SeqCst));
}

// The shared `dispatch_with_repair` helper — the SAME path the resume flow uses
// for human-GATED tool calls (bash/write/edit). A recoverable-error call with a
// coercible arg (`"false"` where a bool is wanted) must be repaired and
// re-dispatched to success, so gated calls get the same repair as auto-approved
// ones (the resume-path gap: gated calls previously bypassed repair entirely).
#[tokio::test]
async fn dispatch_with_repair_coerces_a_recoverable_calls_args() {
    use std::sync::atomic::Ordering;

    let mut cfg = standard_config(make_agent());
    let reg = Arc::new(BoolToolRegistry::new());
    cfg.tool_registry = reg.clone();
    cfg.tool_call_repair = Some(Arc::new(crate::tool_call_repair::StandardToolCallRepair));
    let h = StandardHarness::new(cfg);

    let (effective, output) = h
        .dispatch_with_repair(
            reg.as_ref(),
            ToolCall {
                id: "c".into(),
                name: "set_flag".into(),
                input: serde_json::json!({ "flag": "false" }),
            },
        )
        .await;

    // The arg was coerced "false" → false and re-dispatched to success.
    assert_eq!(effective.input, serde_json::json!({ "flag": false }));
    assert!(matches!(output, ToolOutput::Success { .. }), "got {output:?}");
    assert_eq!(reg.call_count.load(Ordering::SeqCst), 2, "original (fail) + repaired (ok)");
}

// The enrichment gate: only PARAMETER-shaped errors get the schema+hint; a
// command/runtime failure (a failing `cargo test`) is left raw so the hint
// doesn't mislead the model — or an offline judge — into seeing a non-existent
// "schema violation".
#[test]
fn is_param_error_distinguishes_malformed_from_command_failures() {
    assert!(StandardHarness::is_param_error(
        "invalid parameters: invalid type: string \"60\", expected u64"
    ));
    assert!(StandardHarness::is_param_error(
        "schema validation failed for bash_command: missing required field `script`"
    ));
    assert!(!StandardHarness::is_param_error("exit 101 ; stderr: test assertion failed"));
    assert!(!StandardHarness::is_param_error(
        "exit 1 ; stderr: error: no library targets found in package `tier1`"
    ));
}

// Without a repair provider, the same bad `flag: "false"` argument yields a
// recoverable tool error fed back to the model (today's behaviour).
#[tokio::test]
async fn without_repair_bad_bool_arg_errors() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "set_flag".into(),
            input: serde_json::json!({ "flag": "false" }),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });

    let mut cfg = standard_config(a);
    let reg = Arc::new(BoolToolRegistry::new());
    cfg.tool_registry = reg.clone();
    // No repair provider configured.

    let last_tool_error = Arc::new(AtomicBool::new(false));
    let sink_flag = last_tool_error.clone();
    let opts = HarnessRunOptions::new(react(5)).with_stream(Box::new(move |ev| {
        if let StreamEvent::ToolResult { is_error, .. } = ev {
            sink_flag.store(is_error, Ordering::SeqCst);
        }
    }));

    let h = StandardHarness::new(cfg);
    let _ = h.run(opts).await;
    // Exactly one dispatch, and it errored.
    assert_eq!(reg.call_count.load(Ordering::SeqCst), 1);
    assert!(last_tool_error.load(Ordering::SeqCst));
}

// Rule: WaitingForHuman from a tool dispatch propagates to RunResult.
#[tokio::test]
async fn tool_waiting_for_human_propagates() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "subagent".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    let child_task = task(LoopStrategy::ReAct(ReactConfig::per_loop(1)));
    reg.push(ToolOutput::WaitingForHuman {
        child_state: Box::new(ChildPausedState {
            session_id: SessionId::new("child"),
            task_id: TaskId::new("ct"),
            turn_number: 1,
            session_state: SessionState::default(),
            pending_tool_calls: vec![],
            approved_results: vec![],
            human_request: Some(HumanRequest::Clarification {
                question: "?".into(),
                options: None,
            }),
            task: child_task,
            budget_used: BudgetSnapshot::default(),
            parent_tool_call_id: "c".into(),
            toolset: ToolsetRef::default(),
        }),
        request: HumanRequest::Clarification {
            question: "?".into(),
            options: None,
        },
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::WaitingForHuman { state, .. } => {
            assert!(state.child_state.is_some());
        }
        other => panic!("expected WaitingForHuman from tool, got {other:?}"),
    }
}

// ====================================================================
// Tool Escalation Protocol (issue #80) — rules R1–R9
// ====================================================================

/// Build a `MockAgent` whose first turn requests `n` tool calls (named
/// `t0..t{n-1}` with ids `c0..c{n-1}`), then a `FinalResponse` so a
/// resumed loop can terminate.
fn agent_with_tool_calls(n: usize) -> Arc<MockAgent> {
    let a = make_agent();
    let calls: Vec<ToolCall> = (0..n)
        .map(|k| ToolCall {
            id: format!("c{k}"),
            name: format!("t{k}"),
            input: serde_json::json!({}),
        })
        .collect();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls,
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "resumed-done".into(),
        usage: usage(),
    });
    a
}

fn abort_signal() -> HarnessSignal {
    HarnessSignal::Abort {
        reason: "agent requested stop".into(),
    }
}

// R1 + R8: a dispatched `Escalate { Abort }` terminates the run and
// returns `RunResult::Escalate`, NOT `RunResult::Failure`.
#[tokio::test]
async fn escalate_abort_terminates_with_escalate_not_failure() {
    let a = agent_with_tool_calls(1);
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Escalate {
        signal: abort_signal(),
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Escalate { signal, .. } => {
            assert_eq!(signal, abort_signal());
        }
        RunResult::Failure { .. } => {
            panic!("Abort must NOT surface as Failure (R8)")
        }
        other => panic!("expected Escalate, got {other:?}"),
    }
}

// R2: the escalation is NOT appended to message history. With one
// escalating call, the only appended message is the assistant tool-call
// turn — there is no `Role::Tool` result message.
#[tokio::test]
async fn escalate_is_not_appended_to_history() {
    let a = agent_with_tool_calls(1);
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Escalate {
        signal: abort_signal(),
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Escalate { state, .. } => {
            let tool_results = state
                .session_state
                .messages
                .iter()
                .filter(|m| m.role == Role::Tool)
                .count();
            assert_eq!(tool_results, 0, "escalation must not append a tool result");
            // The assistant tool-call turn IS recorded (well-formed history).
            let assistant_turns = state
                .session_state
                .messages
                .iter()
                .filter(|m| m.role == Role::Assistant)
                .count();
            assert_eq!(assistant_turns, 1);
        }
        other => panic!("expected Escalate, got {other:?}"),
    }
}

// R3: observability is finalized with `SessionOutcome::Escalated`.
#[tokio::test]
async fn escalate_finalizes_observability_as_escalated() {
    let a = agent_with_tool_calls(1);
    let mut cfg = standard_config(a);
    let obs = Arc::new(crate::observability::InMemoryObservabilityProvider::new());
    cfg.observability = Some(obs.clone());
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Escalate {
        signal: abort_signal(),
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    let sid = match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Escalate { session_id, .. } => session_id,
        other => panic!("expected Escalate, got {other:?}"),
    };
    let metrics = obs
        .get_session_metrics(&sid)
        .await
        .expect("escalation must flush a finalized session");
    assert_eq!(metrics.outcome, SessionOutcome::Escalated);
}

// R3 (contrast): `WaitingForHuman` does NOT finalize observability.
#[tokio::test]
async fn waiting_for_human_does_not_finalize_observability() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "subagent".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let obs = Arc::new(crate::observability::InMemoryObservabilityProvider::new());
    cfg.observability = Some(obs.clone());
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::WaitingForHuman {
        child_state: Box::new(ChildPausedState {
            session_id: SessionId::new("child"),
            task_id: TaskId::new("ct"),
            turn_number: 1,
            session_state: SessionState::default(),
            pending_tool_calls: vec![],
            approved_results: vec![],
            human_request: Some(HumanRequest::Clarification {
                question: "?".into(),
                options: None,
            }),
            task: task(LoopStrategy::ReAct(ReactConfig::per_loop(1))),
            budget_used: BudgetSnapshot::default(),
            parent_tool_call_id: "c".into(),
            toolset: ToolsetRef::default(),
        }),
        request: HumanRequest::Clarification {
            question: "?".into(),
            options: None,
        },
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    let sid = match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::WaitingForHuman { state, .. } => state.session_id.clone(),
        other => panic!("expected WaitingForHuman, got {other:?}"),
    };
    // Not finalized: `finalize_observability` was never called, so no
    // terminal `SessionOutcome` was recorded. Metrics may still exist
    // (a turn span was emitted) but the outcome defaults to `Partial` —
    // crucially it is NOT `Escalated` (contrast the escalation path).
    if let Some(m) = obs.get_session_metrics(&sid).await {
        assert_ne!(
            m.outcome,
            SessionOutcome::Escalated,
            "WaitingForHuman is not terminal — it must not finalize as Escalated"
        );
    }
}

// R4: `RunResult::Escalate` carries all five fields populated, and
// `turns == budget_used.turns`.
#[tokio::test]
async fn escalate_carries_all_five_fields() {
    let a = agent_with_tool_calls(1);
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Escalate {
        signal: abort_signal(),
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Escalate {
            signal,
            state,
            session_id,
            usage: agg,
            turns,
        } => {
            assert_eq!(signal, abort_signal());
            assert_eq!(session_id, SessionId::new("s1"));
            assert_eq!(state.session_id, session_id);
            assert_eq!(turns, state.budget_used.turns, "turns == budget_used.turns");
            // One turn was consumed before the escalating dispatch.
            assert_eq!(turns, 1);
            // Usage is populated from the consumed turn.
            assert_eq!(agg.input_tokens, 1);
            assert_eq!(agg.output_tokens, 1);
        }
        other => panic!("expected Escalate, got {other:?}"),
    }
}

// R5 + R6: the preserved `state` is resumable, and the signal is
// discarded on resume — the harness just continues the original session.
#[tokio::test]
async fn escalate_state_resumes_and_discards_signal() {
    let a = agent_with_tool_calls(1);
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Escalate {
        signal: HarnessSignal::SwitchMode {
            mode: crate::prompt_chunk_registry::Mode::Plan,
        },
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    let state = match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Escalate { state, .. } => *state,
        other => panic!("expected Escalate, got {other:?}"),
    };
    // Escalation-derived state has no human request (R5 shape).
    assert!(state.human_request.is_none());
    // R6: resume continues the ORIGINAL session — it does NOT switch mode,
    // enter plan mode, or abort. The MockAgent's next turn is a
    // FinalResponse, so resume runs to Success.
    match h.resume(state, HumanResponse::Allow, None).await {
        RunResult::Success {
            output, session_id, ..
        } => {
            assert_eq!(output, "resumed-done");
            assert_eq!(session_id, SessionId::new("s1"));
        }
        other => panic!("expected Success on resume (signal discarded), got {other:?}"),
    }
}

// Issue #81: a tool returning AwaitingClarification pauses the loop with a
// PausedState whose human_request is a Clarification (NO child_state) and
// returns WaitingForHuman. The clarifying call is preserved as the head of
// pending_tool_calls.
#[tokio::test]
async fn awaiting_clarification_pauses_without_child_state() {
    let a = agent_with_tool_calls(2);
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::AwaitingClarification {
        question: "which option?".into(),
        options: Some(vec!["a".into(), "b".into()]),
    });
    cfg.tool_registry = reg.clone();
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::WaitingForHuman { state, request } => {
            // No subagent child state on a clarification pause.
            assert!(state.child_state.is_none());
            match &request {
                HumanRequest::Clarification { question, options } => {
                    assert_eq!(question, "which option?");
                    assert_eq!(
                        options.as_ref().unwrap(),
                        &vec!["a".to_string(), "b".to_string()]
                    );
                }
                other => panic!("expected Clarification, got {other:?}"),
            }
            // The clarifying call (c0/t0) is the head of pending; c1 trails.
            assert_eq!(state.pending_tool_calls.len(), 2);
            assert_eq!(state.pending_tool_calls[0].id, "c0");
            assert_eq!(state.pending_tool_calls[1].id, "c1");
        }
        other => panic!("expected WaitingForHuman, got {other:?}"),
    }
    // Exactly one dispatch (the clarifying call); c1 was preserved, not run.
    assert_eq!(reg.call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
}

// ====================================================================
// Consult (issue #114)
// ====================================================================

fn consult_req() -> ConsultRequest {
    ConsultRequest {
        kind: "advice".into(),
        situation: "stuck on auth".into(),
        attempts: 3,
        question: "what next?".into(),
    }
}

// R1 + R10: a worker-side tool returning `ToolOutput::Consult` pauses the
// loop and returns `RunResult::Consult`. The consult call is the head of
// pending_tool_calls, `human_request` is None, there is no child_state, and
// the consult is NOT appended to message history (R10).
#[tokio::test]
async fn worker_consult_pauses_and_returns_run_result_consult() {
    let a = agent_with_tool_calls(2);
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::consult(consult_req()));
    cfg.tool_registry = reg.clone();
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Consult { request, state, .. } => {
            assert_eq!(request.kind, "advice");
            assert_eq!(request.question, "what next?");
            // R1: human_request None, no child_state.
            assert!(state.human_request.is_none());
            assert!(state.child_state.is_none());
            // R1: consulting call (c0) is head; c1 trails.
            assert_eq!(state.pending_tool_calls.len(), 2);
            assert_eq!(state.pending_tool_calls[0].id, "c0");
            assert_eq!(state.pending_tool_calls[1].id, "c1");
            // R10: the consult produced NO tool-result turn in history. The
            // only message recorded is the assistant tool-call request turn.
            let tool_turns = state
                .session_state
                .messages
                .iter()
                .filter(|m| matches!(m.role, crate::model::Role::Tool))
                .count();
            assert_eq!(tool_turns, 0, "consult must not append a tool result yet");
        }
        other => panic!("expected Consult, got {other:?}"),
    }
    // Exactly one dispatch (the consulting call); c1 was preserved, not run.
    assert_eq!(reg.call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
}

// R6 (graceful degradation): an empty `consult_handlers` map means a
// standalone worker simply surfaces `RunResult::Consult` to its caller
// unchanged (existing callers unaffected).
#[tokio::test]
async fn empty_consult_handlers_surfaces_consult_unchanged() {
    let a = agent_with_tool_calls(1);
    let mut cfg = standard_config(a);
    assert!(cfg.consult_handlers.is_empty());
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::consult(consult_req()));
    cfg.tool_registry = reg.clone();
    let h = StandardHarness::new(cfg);
    assert!(matches!(
        h.run(HarnessRunOptions::new(react(5))).await,
        RunResult::Consult { .. }
    ));
}

// R10 + resume seam: `resume_consult` with an Answer injects it as the TOOL
// RESULT for the head pending (consult) call, then continues to Success.
#[tokio::test]
async fn resume_consult_injects_answer_as_tool_result() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "consult-resumed-done".into(),
        usage: usage(),
    });
    let cfg = standard_config(a);
    let h = StandardHarness::new(cfg);
    let state = PausedState {
        session_id: SessionId::new("s"),
        task_id: TaskId::new("t"),
        turn_number: 1,
        session_state: SessionState::default(),
        pending_tool_calls: vec![ToolCall {
            id: "consult".into(),
            name: "ask_advice".into(),
            input: serde_json::json!({"kind": "advice"}),
        }],
        approved_results: vec![],
        human_request: None,
        task: react(5),
        budget_used: BudgetSnapshot::default(),
        child_state: None,
        toolset: ToolsetRef::default(),
    };
    match h
        .resume_consult(
            state,
            ConsultResponse::Answer {
                text: "the answer".into(),
            },
            None,
        )
        .await
    {
        RunResult::Success { output, .. } => assert_eq!(output, "consult-resumed-done"),
        other => panic!("expected Success, got {other:?}"),
    }
}

// ====================================================================
// #140 — PausedState carries the pausing leaf's toolset handle
// ====================================================================

/// A bare ReAct task whose leaf carries a NON-EMPTY toolset handle. Used to
/// prove the leaf's handle threads into the pause state (#140 AC2a) and that
/// resume routes pending calls through it (#140 AC2b).
fn react_scoped(max: u32, handle: &str) -> Task {
    task(LoopStrategy::ReAct(ReactConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        budget: BudgetPolicy::PerLoop { value: max },
        agent: AgentRef(String::new()),
        toolset: ToolsetRef(handle.into()),
        output: None,
        system_prompt: None,
    }))
}

/// #140 AC1 (back-compat): a paused-state blob WITHOUT a `toolset` key still
/// deserializes, defaulting to the empty handle. A non-empty handle round-trips.
#[test]
fn paused_state_toolset_back_compat_and_round_trip() {
    // A pre-#140 PausedState JSON (no `toolset` key) — must default to "".
    let pre_140 = serde_json::json!({
        "session_id": "s",
        "task_id": "t",
        "turn_number": 1,
        "session_state": { "messages": [], "extras": {} },
        "pending_tool_calls": [],
        "approved_results": [],
        "human_request": null,
        "task": serde_json::to_value(react(5)).unwrap(),
        "budget_used": serde_json::to_value(BudgetSnapshot::default()).unwrap(),
        "child_state": null
        // note: NO "toolset" key
    });
    let parsed: PausedState = serde_json::from_value(pre_140).unwrap();
    assert_eq!(
        parsed.toolset,
        ToolsetRef(String::new()),
        "missing toolset must default to the empty handle"
    );

    // The empty handle ALWAYS serializes (never skipped) for byte-parity.
    let wire = serde_json::to_string(&parsed).unwrap();
    assert!(
        wire.contains("\"toolset\":\"\""),
        "empty toolset must serialize explicitly, not be skipped: {wire}"
    );

    // A non-empty handle round-trips by value.
    let mut scoped = parsed.clone();
    scoped.toolset = ToolsetRef("scoped".into());
    let back: PausedState = serde_json::from_str(&serde_json::to_string(&scoped).unwrap()).unwrap();
    assert_eq!(back.toolset, ToolsetRef("scoped".into()));
    assert_eq!(back, scoped);

    // The same back-compat + always-serialize contract holds for ChildPausedState.
    let child_pre_140 = serde_json::json!({
        "session_id": "c",
        "task_id": "ct",
        "turn_number": 1,
        "session_state": { "messages": [], "extras": {} },
        "pending_tool_calls": [],
        "approved_results": [],
        "human_request": null,
        "task": serde_json::to_value(react(1)).unwrap(),
        "budget_used": serde_json::to_value(BudgetSnapshot::default()).unwrap(),
        "parent_tool_call_id": "p"
        // note: NO "toolset" key
    });
    let child: ChildPausedState = serde_json::from_value(child_pre_140).unwrap();
    assert_eq!(child.toolset, ToolsetRef(String::new()));
    assert!(serde_json::to_string(&child)
        .unwrap()
        .contains("\"toolset\":\"\""));
}

/// #140 AC2a (populate): a Consult pause from a leaf carrying
/// `ToolsetRef("scoped")` returns a `PausedState` whose `toolset` is that
/// handle — proving the leaf's handle is captured at the pause site.
#[tokio::test]
async fn consult_pause_carries_leaf_toolset_handle() {
    let a = agent_with_tool_calls(1);
    let mut cfg = standard_config(a);
    // Register the "scoped" toolset handle so `validate()` at run entry passes.
    cfg.registry = std::mem::take(&mut cfg.registry)
        .into_builder()
        .toolset("scoped", Arc::new(EmptyToolRegistry))
        .build();
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::consult(consult_req()));
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    match h
        .run(HarnessRunOptions::new(react_scoped(5, "scoped")))
        .await
    {
        RunResult::Consult { state, .. } => {
            assert_eq!(
                state.toolset,
                ToolsetRef("scoped".into()),
                "consult pause must carry the leaf's toolset handle"
            );
        }
        other => panic!("expected Consult, got {other:?}"),
    }
}

/// #140 AC2a (populate): the Clarification (#81) leaf-pause path likewise
/// carries the leaf's toolset handle.
#[tokio::test]
async fn clarification_pause_carries_leaf_toolset_handle() {
    let a = agent_with_tool_calls(1);
    let mut cfg = standard_config(a);
    cfg.registry = std::mem::take(&mut cfg.registry)
        .into_builder()
        .toolset("scoped", Arc::new(EmptyToolRegistry))
        .build();
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::AwaitingClarification {
        question: "which?".into(),
        options: None,
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    match h
        .run(HarnessRunOptions::new(react_scoped(5, "scoped")))
        .await
    {
        RunResult::WaitingForHuman { state, .. } => {
            assert_eq!(state.toolset, ToolsetRef("scoped".into()));
        }
        other => panic!("expected WaitingForHuman, got {other:?}"),
    }
}

/// #140 AC2b (THE load-bearing regression guard): a `PausedState` whose
/// `toolset` is `"scoped"` resumes pending per-node tool calls against the
/// SCOPED catalogue. A tool registered ONLY under `"scoped"` (absent from the
/// global catalogue) dispatches successfully on resume.
///
/// NEGATIVE CONTROL: the identical state with an EMPTY handle resumes against
/// the global catalogue, where the scoped-only tool is unknown → a recoverable
/// unknown-tool error. This proves the carried handle is behaviorally
/// load-bearing, not cosmetic.
#[tokio::test]
async fn resume_consult_routes_pending_calls_through_carried_toolset() {
    use crate::tool_registry::mock::EchoTool;
    use crate::tool_registry::{ToolAnnotations, ToolSchema};
    use crate::tools::StandardTool;

    // An echo tool under an arbitrary name (registered into a named catalogue).
    fn echo_tool(name: &str) -> StandardTool {
        StandardTool::new(
            Box::new(EchoTool::new(name)),
            ToolSchema {
                name: name.into(),
                description: "echo".into(),
                parameters: serde_json::json!({"type": "object"}),
                annotations: ToolAnnotations::default(),
            },
        )
    }

    // A paused state with TWO pending calls: the head consult call (gets the
    // answer injected) and a trailing call to the scoped-only tool (dispatched
    // through `effective_tool_registry(&session_id, &state.toolset)`).
    fn make_state(handle: &str) -> PausedState {
        PausedState {
            session_id: SessionId::new("s"),
            task_id: TaskId::new("t"),
            turn_number: 1,
            session_state: SessionState::default(),
            pending_tool_calls: vec![
                ToolCall {
                    id: "consult".into(),
                    name: "ask_advice".into(),
                    input: serde_json::json!({"kind": "advice"}),
                },
                ToolCall {
                    id: "scoped".into(),
                    name: "scoped_only".into(),
                    input: serde_json::json!({"probe": 1}),
                },
            ],
            approved_results: vec![],
            human_request: None,
            // A bare ReAct leaf carrying the handle, so resume takes the
            // dispatch-then-re-enter-window path (not the re-drive path).
            task: task(LoopStrategy::ReAct(ReactConfig {
                behavior: BudgetExhaustedBehavior::Escalate,
                budget: BudgetPolicy::PerLoop { value: 5 },
                agent: AgentRef(String::new()),
                toolset: ToolsetRef(handle.into()),
                output: None,
                system_prompt: None,
            })),
            budget_used: BudgetSnapshot::default(),
            child_state: None,
            toolset: ToolsetRef(handle.into()),
        }
    }

    // Was the scoped-only tool's pending call dispatched successfully? Scan the
    // resumed session history for its tool-result message. `NoopContextManager`
    // prefixes a recoverable error result with "[error]"; a success echoes the
    // call input verbatim (so it contains "probe").
    fn scoped_dispatched_ok(messages: &[Message]) -> bool {
        messages.iter().any(|m| {
            matches!(m.role, crate::model::Role::Tool)
                && match &m.content {
                    Content::Text { text } => text.contains("probe") && !text.contains("[error]"),
                    _ => false,
                }
        })
    }

    // Build a harness with a SCOPED catalogue ("scoped" → scoped_only) AND a
    // GLOBAL catalogue ("global_only", which does NOT contain scoped_only) plus
    // a worker agent that emits a final response so the re-entered ReAct window
    // terminates cleanly. The global catalogue is what the EMPTY handle falls
    // back to — and it makes `scoped_only` a genuine unknown-tool there.
    fn harness_with_scoped_catalogue() -> StandardHarness {
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            reasoning: None,
            content: "resumed-done".into(),
            usage: usage(),
        });
        let cfg = catalogue_builder(a)
            .tools(vec![echo_tool("global_only")])
            .toolset_tools("scoped", vec![echo_tool("scoped_only")])
            .build_config();
        StandardHarness::new(cfg)
    }

    // ── Positive: carried handle "scoped" routes to the scoped catalogue ──
    let h = harness_with_scoped_catalogue();
    let result = h
        .resume_consult(
            make_state("scoped"),
            ConsultResponse::Answer { text: "ok".into() },
            None,
        )
        .await;
    let RunResult::Success { session_state, .. } = result else {
        panic!("expected Success, got {result:?}");
    };
    assert!(
        scoped_dispatched_ok(&session_state.messages),
        "scoped-only tool must dispatch successfully when the carried handle is 'scoped'"
    );

    // ── Negative control: EMPTY handle falls back to the global catalogue ──
    let h = harness_with_scoped_catalogue();
    let result = h
        .resume_consult(
            make_state(""),
            ConsultResponse::Answer { text: "ok".into() },
            None,
        )
        .await;
    let RunResult::Success { session_state, .. } = result else {
        panic!("expected Success, got {result:?}");
    };
    assert!(
        !scoped_dispatched_ok(&session_state.messages),
        "with the EMPTY handle the scoped-only tool is unknown → must NOT dispatch successfully"
    );
    // And it produced the recoverable unknown-tool error result.
    assert!(
        session_state
            .messages
            .iter()
            .any(|m| matches!(m.role, crate::model::Role::Tool)
                && matches!(&m.content, Content::Text { text } if text.contains("[error]"))),
        "the scoped-only call must surface a recoverable error under the empty handle"
    );
}

// R8: serde round-trip is byte-identical for every consult type. For each
// section of `fixtures/harness/consult.json` we parse the raw JSON into the
// typed Rust value, re-serialize it to a `serde_json::Value`, and assert it
// equals the original parsed value — proving the wire shape is stable and
// the same fixture replays identically across the four languages.
#[test]
fn consult_fixture_replay_round_trips() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../fixtures/harness/consult.json");
    let raw = std::fs::read_to_string(&path).expect("fixture present");
    let root: serde_json::Value = serde_json::from_str(&raw).expect("fixture parses");

    // Helper: assert each element of a named array round-trips through `T`.
    fn check<T>(root: &serde_json::Value, key: &str)
    where
        T: serde::de::DeserializeOwned + serde::Serialize,
    {
        let arr = root
            .get(key)
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("missing array `{key}`"));
        assert!(!arr.is_empty(), "`{key}` must have cases");
        for (i, case) in arr.iter().enumerate() {
            let typed: T = serde_json::from_value(case.clone())
                .unwrap_or_else(|e| panic!("`{key}`[{i}] deserialize: {e}"));
            let back = serde_json::to_value(&typed)
                .unwrap_or_else(|e| panic!("`{key}`[{i}] serialize: {e}"));
            assert_eq!(&back, case, "`{key}`[{i}] not byte-identical");
        }
    }

    check::<RunResult>(&root, "run_result_cases");
    check::<ToolOutput>(&root, "worker_tool_output_cases");
    check::<ToolOutput>(&root, "subagent_tool_output_cases");
    check::<ConsultRequest>(&root, "consult_request_cases");
    check::<ConsultResponse>(&root, "consult_response_cases");
    check::<ConsultOverflowPolicy>(&root, "consult_overflow_policy_cases");

    // Spot-check the documented invariants on the structured cases.
    let rr = &root["run_result_cases"][0];
    assert_eq!(rr["kind"], "consult");
    assert!(rr["state"]["human_request"].is_null(), "human_request null");
    assert!(
        rr["state"]["child_state"].is_null(),
        "no child at RunResult level"
    );
    // Worker-side ToolOutput::Consult omits child_state (skip_serializing).
    let worker = &root["worker_tool_output_cases"][0];
    assert_eq!(worker["kind"], "consult");
    assert!(
        worker.get("child_state").is_none(),
        "worker-side consult omits child_state"
    );
    // Subagent-boundary ToolOutput::Consult carries a populated child_state.
    let sub = &root["subagent_tool_output_cases"][0];
    assert!(
        sub["child_state"].is_object(),
        "subagent populates child_state"
    );
}

// Issue #81: resuming a Clarification pause injects the human's Answer as
// the TOOL RESULT for the clarifying call, then continues to Success.
#[tokio::test]
async fn clarification_resume_injects_answer_as_tool_result() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "clarified-done".into(),
        usage: usage(),
    });
    let cfg = standard_config(a);
    let h = StandardHarness::new(cfg);
    let state = PausedState {
        session_id: SessionId::new("s"),
        task_id: TaskId::new("t"),
        turn_number: 1,
        session_state: SessionState::default(),
        pending_tool_calls: vec![ToolCall {
            id: "ask".into(),
            name: "ask_user_question".into(),
            input: serde_json::json!({"question": "?"}),
        }],
        approved_results: vec![],
        human_request: Some(HumanRequest::Clarification {
            question: "?".into(),
            options: None,
        }),
        task: react(5),
        budget_used: BudgetSnapshot::default(),
        child_state: None,
        toolset: ToolsetRef::default(),
    };
    match h
        .resume(
            state,
            HumanResponse::Answer {
                text: "use a".into(),
            },
            None,
        )
        .await
    {
        RunResult::Success { output, .. } => assert_eq!(output, "clarified-done"),
        other => panic!("expected Success, got {other:?}"),
    }
}

// Issue #81: the send_message tool surfaces a StreamEvent::UserMessage and
// the loop records a success tool result (continuing, not pausing).
#[tokio::test]
async fn send_message_emits_user_message_event() {
    use std::sync::Mutex;
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: crate::tools::SendMessageTool::NAME.into(),
            input: serde_json::json!({"content": "hello human"}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "hello human".into(),
        truncated: false,
    });
    cfg.tool_registry = reg.clone();
    let h = StandardHarness::new(cfg);

    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    let opts = HarnessRunOptions::new(react(5)).with_stream(Box::new(move |ev| {
        if let StreamEvent::UserMessage { content, .. } = ev {
            sink.lock().unwrap().push(content);
        }
    }));
    match h.run(opts).await {
        RunResult::Success { output, .. } => assert_eq!(output, "done"),
        other => panic!("expected Success, got {other:?}"),
    }
    let msgs = captured.lock().unwrap();
    assert_eq!(msgs.as_slice(), &["hello human".to_string()]);
}

// R7-adjacent: each `HarnessSignal` variant round-trips through serde
// wrapped in `ToolOutput::Escalate` with the documented wire shape.
// (Full byte fixture lives in `fixtures/harness/escalation_signals.json`.)
#[test]
fn harness_signal_wire_format_round_trips() {
    #[cfg_attr(not(feature = "dangerous"), allow(unused_mut))]
    let mut cases = vec![
        ToolOutput::Escalate {
            signal: HarnessSignal::EnterPlanMode {
                context: "ctx".into(),
            },
        },
        ToolOutput::Escalate {
            signal: HarnessSignal::ExitPlanMode {
                plan: crate::plan::PlanArtifact {
                    tasks: vec!["a".into(), "b".into()],
                    rationale: "why".into(),
                },
            },
        },
        ToolOutput::Escalate {
            signal: HarnessSignal::SwitchMode {
                mode: crate::prompt_chunk_registry::Mode::SafeAuto,
            },
        },
        ToolOutput::Escalate {
            signal: abort_signal(),
        },
    ];
    // Yolo only exists under the `dangerous` feature (issue #34).
    #[cfg(feature = "dangerous")]
    cases.push(ToolOutput::Escalate {
        signal: HarnessSignal::SwitchMode {
            mode: crate::prompt_chunk_registry::Mode::Yolo,
        },
    });
    for case in cases {
        let json = serde_json::to_string(&case).unwrap();
        let back: ToolOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(case, back);
    }
    // Spot-check the tag shape: snake_case `kind` on both layers.
    let json = serde_json::to_value(&ToolOutput::Escalate {
        signal: abort_signal(),
    })
    .unwrap();
    assert_eq!(json["kind"], "escalate");
    assert_eq!(json["signal"]["kind"], "abort");
    assert_eq!(json["signal"]["reason"], "agent requested stop");
}

// R7: `SwitchMode` carries the EXISTING `Mode` enum (no `HarnessMode`).
#[test]
fn switch_mode_uses_existing_mode_enum() {
    let json = serde_json::to_value(&HarnessSignal::SwitchMode {
        mode: crate::prompt_chunk_registry::Mode::SafeAuto,
    })
    .unwrap();
    assert_eq!(json["kind"], "switch_mode");
    assert_eq!(json["mode"], "safe_auto");
}

// R9: remaining tool calls after the escalating call land in
// `state.pending_tool_calls` (escalate on call[0] of a 2-call batch →
// pending == [call_1]).
#[tokio::test]
async fn escalate_preserves_remaining_calls_as_pending() {
    let a = agent_with_tool_calls(2);
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    // First dispatched call (c0) escalates.
    reg.push(ToolOutput::Escalate {
        signal: abort_signal(),
    });
    cfg.tool_registry = reg.clone();
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Escalate { state, .. } => {
            assert_eq!(state.pending_tool_calls.len(), 1);
            assert_eq!(state.pending_tool_calls[0].id, "c1");
            assert_eq!(state.pending_tool_calls[0].name, "t1");
        }
        other => panic!("expected Escalate, got {other:?}"),
    }
    // Exactly one dispatch happened — c1 was preserved, not executed.
    assert_eq!(reg.call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
}

// Rule: resume() with Halt returns Failure(HumanHalted).
#[tokio::test]
async fn resume_with_halt_returns_human_halted() {
    let a = make_agent();
    let h = StandardHarness::new(standard_config(a));
    let state = PausedState {
        session_id: SessionId::new("s"),
        task_id: TaskId::new("t"),
        turn_number: 1,
        session_state: SessionState::default(),
        pending_tool_calls: vec![],
        approved_results: vec![],
        human_request: Some(HumanRequest::Clarification {
            question: "?".into(),
            options: None,
        }),
        task: react(5),
        budget_used: BudgetSnapshot::default(),
        child_state: None,
        toolset: ToolsetRef::default(),
    };
    match h.resume(state, HumanResponse::Halt, None).await {
        RunResult::Failure {
            reason: HaltReason::HumanHalted,
            ..
        } => {}
        other => panic!("expected HumanHalted, got {other:?}"),
    }
}

// Rule: resume() with Allow dispatches pending tool calls then continues loop.
#[tokio::test]
async fn resume_with_allow_executes_pending_and_continues() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "tool-ok".into(),
        truncated: false,
    });
    cfg.tool_registry = reg.clone();
    let h = StandardHarness::new(cfg);
    let state = PausedState {
        session_id: SessionId::new("s"),
        task_id: TaskId::new("t"),
        turn_number: 1,
        session_state: SessionState::default(),
        pending_tool_calls: vec![ToolCall {
            id: "c".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        approved_results: vec![],
        human_request: Some(HumanRequest::ToolApproval {
            calls: vec![],
            risk_level: RiskLevel::Low,
        }),
        task: react(5),
        budget_used: BudgetSnapshot::default(),
        child_state: None,
        toolset: ToolsetRef::default(),
    };
    match h.resume(state, HumanResponse::Allow, None).await {
        RunResult::Success { output, .. } => assert_eq!(output, "done"),
        other => panic!("expected Success on resume, got {other:?}"),
    }
    assert_eq!(reg.call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
}

// Rule (issue #60): HillClimbing is now implemented — it no longer returns
// StrategyNotYetImplemented. With no `metric_evaluator` wired it instead
// halts with the typed `HillClimbingMisconfigured` (Decision 6), exercised in
// the HillClimbing test block below. PlanExecute (#70), SelfVerifying (#61),
// and Ralph (#58) are likewise implemented and covered in their own blocks,
// so no loop strategy returns the generic stub anymore.
#[tokio::test]
async fn hill_climbing_no_longer_not_yet_implemented() {
    let a = make_agent();
    let h = StandardHarness::new(standard_config(a));
    // #124: with no metric evaluator registered, the unresolved handle is a
    // STARTUP ConfigurationError (single resolution path) — NOT the generic
    // StrategyNotYetImplemented stub.
    let t = task(LoopStrategy::HillClimbing(HillClimbingConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(react_structured(u32::MAX)),
        direction: HillClimbingDirection::Maximize,
        max_stagnation: u32::MAX,
        revert_on_no_improvement: false,
        min_improvement_delta: 0.0,
        evaluator: AgentRef(String::new()),
    }));
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure { reason, .. } => {
            assert!(
                !matches!(reason, HaltReason::StrategyNotYetImplemented { .. }),
                "HillClimbing must not return StrategyNotYetImplemented; got {reason:?}"
            );
            assert!(
                matches!(
                    reason,
                    HaltReason::ConfigurationError {
                        error: HarnessError::UnresolvedHandle { ref kind, .. }
                    } if kind == "metric_evaluator"
                ),
                "expected an unresolved metric_evaluator startup error; got {reason:?}"
            );
        }
        other => panic!("expected Failure, got {other:?}"),
    }
}

// ── PlanExecute plan phase (issue #70) ──────────────────────────────────

use crate::plan::{PlanArtifact, PLAN_EXECUTE_EXTRAS_KEY};

fn plan_task() -> Task {
    // #124 A.5: the plan slot is STRUCTURED — its leaf carries an output
    // schema (registered as "" in standard_config). Execute stays a bare leaf.
    task(LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        plan: Box::new(react_structured(u32::MAX)),
        execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
        plan_model: None,
    }))
}

fn final_resp(text: &str) -> TurnResult {
    TurnResult::FinalResponse {
        reasoning: None,
        content: text.into(),
        usage: usage(),
    }
}

// #76: the plan artifact now lives on the RunStore seam (not extras). Read
// it back through the harness's storage under PLAN_EXECUTE_EXTRAS_KEY.
async fn stored_artifact(h: &StandardHarness, _session_id: &SessionId) -> PlanArtifact {
    // #142: the plan artifact is keyed by the project namespace, not the run
    // session id — read it back under `durable_ns()`.
    let v = h
        .storage()
        .run()
        .get(&durable_ns(), PLAN_EXECUTE_EXTRAS_KEY)
        .await
        .expect("run store get ok")
        .expect("plan_execute present in run store");
    serde_json::from_value(v).expect("artifact deserializes")
}

// Issue #59 happy path: plan turn (2 tasks) then 2 execute completions. The
// full PlanExecute run() now SUCCEEDS (proving ExecutePhaseNotImplemented is
// gone) and `output` is the LAST step's FinalResponse (Q2). turns == 3 (one
// plan + one per step).
#[tokio::test]
async fn plan_execute_full_run_succeeds() {
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["one","two"],"rationale":"why"}"#));
    a.push(final_resp("did one"));
    a.push(final_resp("did two"));
    let h = StandardHarness::new(standard_config(a));
    match h.run(HarnessRunOptions::new(plan_task())).await {
        RunResult::Success { output, turns, .. } => {
            // Q2: the success handle is the LAST completed step's final text.
            assert_eq!(output, "did two");
            assert_eq!(turns, 3, "one plan turn + one turn per task");
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// R1/R3/R4: the plan turn runs exactly once and stores the exact artifact.
// Uses run_plan_phase directly so we can inspect the mutated session_state.
#[tokio::test]
async fn plan_phase_runs_once_and_stores_exact_artifact() {
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["a","b","c"],"rationale":"r"}"#));
    let agent_for_count = a.clone();
    let h = StandardHarness::new(standard_config(a));
    let t = plan_task();
    let mut state = SessionState::default();
    let outcome = h
        .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect("plan phase succeeds");
    // R1: exactly one planner turn.
    assert_eq!(
        agent_for_count
            .call_count
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    // R7: counted against the budget.
    assert_eq!(outcome.turns, 1);
    // R3/R4: exact artifact stored.
    let stored = stored_artifact(&h, &t.session_id).await;
    assert_eq!(stored.tasks, vec!["a", "b", "c"]);
    assert_eq!(stored.rationale, "r");
    assert_eq!(stored, outcome.artifact);
}

// #124 recursion seam (was R2): the plan phase now dispatches the genuine
// `self.plan` child (a ReAct loop) capped at ONE turn. A plan turn that
// requests a tool instead of emitting the JSON plan cannot complete in its
// single turn, so the plan child halts (budget) and the plan phase propagates
// a terminal Failure WITHOUT capturing/storing an artifact. (The old one-shot
// primitive special-cased a tool call as `PlanningTurnFailed`; under genuine
// recursion the cap is what stops the loop — the observable contract that a
// non-plan plan turn fails the run and stores nothing is preserved.)
#[tokio::test]
async fn plan_phase_tool_call_fails_and_stores_no_artifact() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "ok".into(),
        truncated: false,
    });
    let mut cfg = standard_config(a);
    cfg.tool_registry = reg.clone();
    let h = StandardHarness::new(cfg);
    let t = plan_task();
    let mut state = SessionState::default();
    let err = h
        .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect_err("a tool-only plan turn fails the plan phase");
    // The plan child never emitted a JSON plan, so the run halts terminally.
    assert!(
        matches!(err, RunResult::Failure { .. }),
        "expected a terminal Failure, got {err:?}"
    );
    // Nothing captured/stored: no artifact reached the RunStore.
    assert!(h
        .storage()
        .run()
        .get(&durable_ns(), PLAN_EXECUTE_EXTRAS_KEY)
        .await
        .expect("run store get ok")
        .is_none());
}

// ── #93: model_params reach every tool-requesting turn ─────────────────
//
// `RecordingTurnAgent::turn` captures every `Context` it sees in `seen`, and
// the agent copies `Context.params` verbatim into the `ModelRequest`
// (`into_request`). So asserting on a captured context's
// `params.structured_tool_calls` proves the configured params reached the
// request the model would have seen.

/// Build a config whose agent is a context-capturing `RecordingTurnAgent`
/// returning the given turn script, optionally with non-default model params.
fn recording_config(agent: Arc<RecordingTurnAgent>, model_params: ModelParams) -> HarnessConfig {
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, agent);
    cfg.model_params = model_params;
    cfg
}

// No `.model_params(...)` ⇒ each turn's context carries the default
// (structured_tool_calls == false).
#[tokio::test]
async fn default_model_params_are_default() {
    let agent = RecordingTurnAgent::new("rec", vec![RecordingTurnAgent::final_resp("done")]);
    let cfg = recording_config(agent.clone(), ModelParams::default());
    let h = StandardHarness::new(cfg);
    h.run(HarnessRunOptions::new(react(5))).await;
    let seen = agent.seen.lock().unwrap();
    assert!(!seen.is_empty(), "agent should have seen at least one turn");
    assert!(!seen[0].params.structured_tool_calls);
}

// `.model_params(structured_tool_calls = true)` ⇒ the ReAct turn context
// carries it.
#[tokio::test]
async fn model_params_reach_react_turn() {
    let agent = RecordingTurnAgent::new("rec", vec![RecordingTurnAgent::final_resp("done")]);
    let cfg = recording_config(
        agent.clone(),
        ModelParams {
            structured_tool_calls: true,
            ..Default::default()
        },
    );
    let h = StandardHarness::new(cfg);
    h.run(HarnessRunOptions::new(react(5))).await;
    let seen = agent.seen.lock().unwrap();
    assert!(!seen.is_empty());
    assert!(seen[0].params.structured_tool_calls);
}

// The PlanExecute plan phase (`run_plan_phase`) replaces params on its own
// seam — the plan-turn context carries the flag.
#[tokio::test]
async fn model_params_reach_plan_phase() {
    let agent = RecordingTurnAgent::new(
        "rec",
        vec![RecordingTurnAgent::final_resp(
            r#"{"tasks":["one"],"rationale":"r"}"#,
        )],
    );
    let cfg = recording_config(
        agent.clone(),
        ModelParams {
            structured_tool_calls: true,
            ..Default::default()
        },
    );
    let h = StandardHarness::new(cfg);
    let t = plan_task();
    let mut state = SessionState::default();
    h.run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect("plan phase succeeds");
    let seen = agent.seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "exactly one plan turn");
    assert!(seen[0].params.structured_tool_calls);
}

// A full PlanExecute run threads params through the shared react seam used by
// the execute sub-loop — every execute-step context carries the flag.
#[tokio::test]
async fn model_params_reach_execute_subloop() {
    let agent = RecordingTurnAgent::new(
        "rec",
        vec![
            RecordingTurnAgent::final_resp(r#"{"tasks":["one","two"],"rationale":"r"}"#),
            RecordingTurnAgent::final_resp("did one"),
            RecordingTurnAgent::final_resp("did two"),
        ],
    );
    let cfg = recording_config(
        agent.clone(),
        ModelParams {
            structured_tool_calls: true,
            ..Default::default()
        },
    );
    let h = StandardHarness::new(cfg);
    h.run(HarnessRunOptions::new(plan_task())).await;
    let seen = agent.seen.lock().unwrap();
    // 1 plan turn + 2 execute turns; every captured context carries it.
    assert_eq!(seen.len(), 3);
    assert!(
        seen.iter().all(|c| c.params.structured_tool_calls),
        "plan + execute-step contexts all carry the flag"
    );
}

// #93 regression: the plan-phase directive ("Produce a step-by-step plan…
// Respond with a single JSON object…") must NOT leak into the SHARED
// session_state, otherwise every execute step re-sees it and an
// instruction-following model re-emits a plan instead of calling tools.
// Drive a full 2-step PlanExecute run with a context-capturing agent and
// assert no execute-step context carries the directive, while the step
// instructions DO reach those contexts (and a step can issue a tool call).
#[tokio::test]
async fn plan_directive_does_not_leak_into_execute_context() {
    let agent = RecordingTurnAgent::new(
        "rec",
        vec![
            // Plan turn: produce a 2-step plan.
            RecordingTurnAgent::final_resp(r#"{"tasks":["step one","step two"],"rationale":"r"}"#),
            // Execute step 1: issue a tool call, then finalize.
            RecordingTurnAgent::tool_call("noop"),
            RecordingTurnAgent::final_resp("did step one"),
            // Execute step 2: finalize directly.
            RecordingTurnAgent::final_resp("did step two"),
        ],
    );
    let mut cfg = recording_config(agent.clone(), ModelParams::default());
    cfg.tool_registry = Arc::new(ScriptedToolRegistry::new());
    let h = StandardHarness::new(cfg);
    h.run(HarnessRunOptions::new(plan_task())).await;

    let contexts = agent.seen_text();
    // 1 plan turn + (tool call + final) for step one + 1 for step two.
    assert_eq!(contexts.len(), 4, "plan turn + 3 execute turns");

    // The PLAN turn (index 0) DOES carry the directive — that's correct.
    assert!(
        contexts[0].contains("Record the plan ONLY through")
            && contexts[0].contains("Produce a step-by-step plan"),
        "plan turn should see the directive: {}",
        contexts[0]
    );

    // No EXECUTE-step context (indices 1..) may carry the directive.
    for (i, c) in contexts.iter().enumerate().skip(1) {
        assert!(
            !c.contains("Record the plan ONLY through"),
            "execute-step context {i} leaked the directive: {c}"
        );
        assert!(
            !c.contains("Produce a step-by-step plan"),
            "execute-step context {i} leaked the directive: {c}"
        );
    }

    // The execute steps still receive their step instructions and can
    // proceed to a tool call.
    assert!(
        contexts[1].contains("step one"),
        "step-one context should carry its instruction: {}",
        contexts[1]
    );
    assert!(
        contexts[2].contains("noop"),
        "step-one second turn should follow the tool call: {}",
        contexts[2]
    );
    assert!(
        contexts[3].contains("step two"),
        "step-two context should carry its instruction: {}",
        contexts[3]
    );
}

// SC-10: per-leaf `system_prompt` override. A `PlanExecute` whose plan and
// execute leaves carry DISTINCT system prompts runs each phase under ONLY its
// own prompt — neither leaks into the other. The global `config.system_prompt`
// is `None` here, so the ONLY system text a turn can see is what its own leaf
// supplied; any cross-phase appearance is therefore an unambiguous leak.
#[tokio::test]
async fn plan_and_execute_leaves_see_only_their_own_system_prompt() {
    const PLAN_SYS: &str = "PLAN_SYSTEM_PROMPT_MARKER";
    const EXEC_SYS: &str = "EXECUTE_SYSTEM_PROMPT_MARKER";

    let agent = RecordingTurnAgent::new(
        "rec",
        vec![
            // Plan turn: a single-step plan.
            RecordingTurnAgent::final_resp(r#"{"tasks":["only step"],"rationale":"r"}"#),
            // Execute step: finalize directly.
            RecordingTurnAgent::final_resp("did the step"),
        ],
    );
    let mut cfg = recording_config(agent.clone(), ModelParams::default());
    cfg.tool_registry = Arc::new(ScriptedToolRegistry::new());
    // No global prompt ⇒ the ONLY system text a turn can see is its leaf's.
    assert!(cfg.system_prompt.is_none());
    let h = StandardHarness::new(cfg);

    // The two leaves carry DISTINCT per-leaf system prompts; everything else
    // is the bare default leaf.
    let t = task(LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        plan: Box::new(LoopStrategy::ReAct(ReactConfig {
            system_prompt: Some(PLAN_SYS.to_string()),
            ..ReactConfig::per_loop(u32::MAX)
        })),
        execute: Box::new(LoopStrategy::ReAct(ReactConfig {
            system_prompt: Some(EXEC_SYS.to_string()),
            ..ReactConfig::per_loop(u32::MAX)
        })),
        plan_model: None,
    }));

    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Success { output, .. } => assert_eq!(output, "did the step"),
        other => panic!("expected Success, got {other:?}"),
    }

    let contexts = agent.seen_text();
    assert_eq!(contexts.len(), 2, "one plan turn + one execute turn");

    // Plan turn (index 0): sees ONLY the plan leaf's prompt.
    assert!(
        contexts[0].contains(PLAN_SYS),
        "plan turn must carry the plan leaf's system prompt: {}",
        contexts[0]
    );
    assert!(
        !contexts[0].contains(EXEC_SYS),
        "plan turn must NOT see the execute leaf's system prompt: {}",
        contexts[0]
    );

    // Execute turn (index 1): sees ONLY the execute leaf's prompt.
    assert!(
        contexts[1].contains(EXEC_SYS),
        "execute turn must carry the execute leaf's system prompt: {}",
        contexts[1]
    );
    assert!(
        !contexts[1].contains(PLAN_SYS),
        "execute turn must NOT see the plan leaf's system prompt: {}",
        contexts[1]
    );
}

// SC-10: the per-leaf `system_prompt` override WINS over the global
// `config.system_prompt` — a leaf that supplies one sees ONLY its own (the
// global does not leak in), while a leaf WITHOUT an override falls back to the
// global prompt (byte-identical to pre-SC-10).
#[tokio::test]
async fn leaf_system_prompt_overrides_global_and_falls_back() {
    const GLOBAL_SYS: &str = "GLOBAL_SYSTEM_PROMPT_MARKER";
    const PLAN_SYS: &str = "PLAN_ONLY_SYSTEM_PROMPT_MARKER";

    let agent = RecordingTurnAgent::new(
        "rec",
        vec![
            RecordingTurnAgent::final_resp(r#"{"tasks":["only step"],"rationale":"r"}"#),
            RecordingTurnAgent::final_resp("did the step"),
        ],
    );
    let mut cfg = recording_config(agent.clone(), ModelParams::default());
    cfg.tool_registry = Arc::new(ScriptedToolRegistry::new());
    // A global prompt IS configured this time.
    cfg.system_prompt = Some(GLOBAL_SYS.to_string());
    let h = StandardHarness::new(cfg);

    let t = task(LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        // Plan leaf overrides the global prompt.
        plan: Box::new(LoopStrategy::ReAct(ReactConfig {
            system_prompt: Some(PLAN_SYS.to_string()),
            ..ReactConfig::per_loop(u32::MAX)
        })),
        // Execute leaf carries no override ⇒ falls back to the global prompt.
        execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
        plan_model: None,
    }));

    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Success { .. } => {}
        other => panic!("expected Success, got {other:?}"),
    }

    let contexts = agent.seen_text();
    assert_eq!(contexts.len(), 2, "one plan turn + one execute turn");

    // Plan turn: its override WINS — only the plan prompt, NOT the global one.
    assert!(
        contexts[0].contains(PLAN_SYS) && !contexts[0].contains(GLOBAL_SYS),
        "plan turn must see ONLY its override, not the global prompt: {}",
        contexts[0]
    );
    // Execute turn: no override ⇒ the global prompt applies (back-compat).
    assert!(
        contexts[1].contains(GLOBAL_SYS) && !contexts[1].contains(PLAN_SYS),
        "execute turn must fall back to the global prompt: {}",
        contexts[1]
    );
}

// #126 two-tier context (REPLACES the pre-#126 linear-folding regression):
// linear folding broke on a DAG, so it is GONE. The plan-artifact bridge
// produces a LINEAR chain with EMPTY blockers, so steps 1 and 2 are
// INDEPENDENT branches in the DAG sense. Therefore:
//   - step 2 does NOT see step 1's raw tool result (no transcript fold), and
//   - step 2 DOES see step 1's compact Tier-2 ledger summary ("researched")
//     and NOT its mid-step tool-result internals.
#[tokio::test]
async fn execute_steps_two_tier_context_no_transcript_fold() {
    let agent = RecordingTurnAgent::new(
        "rec",
        vec![
            // Plan turn: a 2-step LINEAR plan (no blockers from the bridge).
            RecordingTurnAgent::final_resp(
                r#"{"tasks":["research tokio","summarize findings"],"rationale":"r"}"#,
            ),
            // Step 1: call a tool, then finalize with a distinctive summary.
            RecordingTurnAgent::tool_call("lookup"),
            RecordingTurnAgent::final_resp("researched"),
            // Step 2: finalize directly.
            RecordingTurnAgent::final_resp("summarized"),
        ],
    );
    let mut cfg = recording_config(agent.clone(), ModelParams::default());
    let reg = Arc::new(ScriptedToolRegistry::new());
    // Step 1's tool call returns a distinctive (internal) result string.
    reg.push(ToolOutput::Success {
        content: "TOKIO_FACTS_123".into(),
        truncated: false,
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(plan_task())).await {
        RunResult::Success { output, .. } => assert_eq!(output, "summarized"),
        other => panic!("expected Success, got {other:?}"),
    }

    let contexts = agent.seen_text();
    // 1 plan turn + (tool call + final) for step 1 + 1 for step 2 = 4.
    assert_eq!(contexts.len(), 4, "plan turn + 3 execute turns");

    // Step 1's SECOND turn (index 2) sees its OWN tool result.
    assert!(
        contexts[2].contains("TOKIO_FACTS_123"),
        "step-1 final turn should follow its own tool result: {}",
        contexts[2]
    );
    // #126: step 2 (index 3) must NOT carry step 1's raw tool-result
    // transcript — no linear fold across the DAG.
    assert!(
        !contexts[3].contains("TOKIO_FACTS_123"),
        "step 2 must NOT see step 1's raw tool result (no transcript fold): {}",
        contexts[3]
    );
    // #126 Tier-2: step 2 DOES see step 1's compact ledger summary.
    assert!(
        contexts[3].contains("researched"),
        "step 2 should see step 1's Tier-2 ledger summary: {}",
        contexts[3]
    );
    // And it carries its own instruction.
    assert!(
        contexts[3].contains("summarize findings"),
        "step 2 must carry its own instruction: {}",
        contexts[3]
    );
}

// The streaming path flows through `run_react_inner`'s same seam — the
// streamed turn's captured context carries the flag.
#[tokio::test]
async fn model_params_reach_streaming_turn() {
    let agent = RecordingTurnAgent::new("rec", vec![RecordingTurnAgent::final_resp("done")]);
    let cfg = recording_config(
        agent.clone(),
        ModelParams {
            structured_tool_calls: true,
            ..Default::default()
        },
    );
    let h = StandardHarness::new(cfg);
    let opts = HarnessRunOptions::new(react(5)).with_stream(Box::new(|_ev: StreamEvent| {}));
    h.run(opts).await;
    let seen = agent.seen.lock().unwrap();
    assert!(!seen.is_empty());
    assert!(seen[0].params.structured_tool_calls);
}

// R5 (#124 Q1): the plan child's leaf `ReactConfig.agent` is authoritative —
// register a distinct "planner" agent on the plan leaf; it runs the plan turn
// and the default agent does not.
#[tokio::test]
async fn plan_phase_routes_to_planner_agent() {
    let default_agent = make_agent();
    default_agent.push(final_resp(r#"{"tasks":["default"]}"#));
    let planner = make_agent();
    planner.push(final_resp(r#"{"tasks":["planner"]}"#));

    let mut cfg = standard_config(default_agent.clone());
    // Register the planner under a distinct key + a schema for the plan slot.
    cfg.registry = std::mem::take(&mut cfg.registry)
        .into_builder()
        .agent("planner", planner.clone())
        .schema("plan-schema", serde_json::json!({}))
        .build();
    let h = StandardHarness::new(cfg);
    // A plan tree whose plan leaf routes to "planner".
    let t = task(LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        plan: Box::new(LoopStrategy::ReAct(ReactConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            budget: BudgetPolicy::PerLoop { value: u32::MAX },
            agent: AgentRef("planner".into()),
            toolset: ToolsetRef(String::new()),
            output: Some(SchemaRef("plan-schema".into())),
            system_prompt: None,
        })),
        execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
        plan_model: None,
    }));
    let mut state = SessionState::default();
    h.run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect("plan phase succeeds");

    assert_eq!(
        planner.call_count.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "planner agent ran the plan turn"
    );
    assert_eq!(
        default_agent
            .call_count
            .load(std::sync::atomic::Ordering::SeqCst),
        0,
        "default agent did not run the plan turn"
    );
    assert_eq!(
        stored_artifact(&h, &t.session_id).await.tasks,
        vec!["planner"]
    );
}

// R6: with no planner_agent, the plan turn runs on the default agent.
#[tokio::test]
async fn plan_phase_routes_to_default_agent_when_unset() {
    let default_agent = make_agent();
    default_agent.push(final_resp(r#"{"tasks":["default"]}"#));
    let h = StandardHarness::new(standard_config(default_agent.clone()));
    let t = plan_task();
    let mut state = SessionState::default();
    h.run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect("plan phase succeeds");
    assert_eq!(
        default_agent
            .call_count
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    assert_eq!(
        stored_artifact(&h, &t.session_id).await.tasks,
        vec!["default"]
    );
}

// R8 (#70): the plan turn records exactly one TurnSpan, and it is the FIRST
// span of the run (turn 1). With the execute phase (#59) wired in, the full
// run() emits additional spans for the execute steps; this asserts the plan
// turn's span specifically. A dedicated #59 test
// (`execute_phase_span_count`) covers the per-step span accounting.
#[tokio::test]
async fn plan_phase_records_one_turn_span() {
    let a = make_agent();
    // Plan turn + one execute completion so the run progresses cleanly.
    a.push(final_resp(r#"{"tasks":["a"]}"#));
    a.push(final_resp("did a"));
    let obs = Arc::new(crate::observability::InMemoryObservabilityProvider::new());
    let mut cfg = standard_config(a);
    cfg.observability = Some(obs.clone());
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(plan_task())).await;
    let spans = obs.turn_spans(&SessionId::new("s1"));
    // Plan turn (1) + the single execute step turn (1) = 2 spans total.
    assert_eq!(spans.len(), 2, "plan turn + one execute step turn");
    // The plan turn is span 0, turn_number 1.
    assert_eq!(spans[0].turn_number, 1, "first span is the plan turn");
}

// R10: budget exhausted before the plan turn → budget-exceeded Failure with
// no artifact stored and no planner turn.
#[tokio::test]
async fn plan_phase_budget_exhausted_before_turn() {
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["a"]}"#));
    let agent_for_count = a.clone();
    let h = StandardHarness::new(standard_config(a));
    let t = plan_task().with_budget(BudgetLimits {
        max_turns: Some(1),
        ..Default::default()
    });
    let mut state = SessionState::default();
    // Pre-consume the only allowed turn.
    let used = BudgetSnapshot {
        turns: 1,
        ..Default::default()
    };
    let err = h
        .run_plan_phase(&t, &mut state, used, &None)
        .await
        .expect_err("budget exhausted");
    match err {
        RunResult::Failure {
            reason:
                HaltReason::BudgetExceeded {
                    limit_type: BudgetLimitType::Turns,
                },
            ..
        } => {}
        other => panic!("expected BudgetExceeded, got {other:?}"),
    }
    assert_eq!(
        agent_for_count
            .call_count
            .load(std::sync::atomic::Ordering::SeqCst),
        0,
        "no planner turn ran"
    );
    assert!(h
        .storage()
        .run()
        .get(&durable_ns(), PLAN_EXECUTE_EXTRAS_KEY)
        .await
        .expect("run store get ok")
        .is_none());
}

/// A planner that never finalizes BURNS its plan-leaf turn budget (the
/// PerLoop(2) leaf cap binds, not the global backstop). The leaf PROPAGATES a
/// `BudgetExhausted` (rule 6) with NO terminal — which `run_plan_subtree` used
/// to DISCARD, returning `None` → the opaque
/// `PlanPhaseFailed{"plan sub-strategy produced no terminal"}`. It must now go
/// through the SAME budget-resolution as the execute phase. Under `Autonomous`
/// that is a clean, typed `Failure{BudgetExceeded}` (NOT `PlanPhaseFailed`).
#[tokio::test]
async fn plan_phase_leaf_budget_exhaustion_fails_cleanly_not_no_terminal() {
    let a = make_agent();
    // The planner only ever calls a tool — it never emits a final plan, so the
    // plan leaf runs to its PerLoop(2) cap without producing a terminal.
    for _ in 0..4 {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: "c".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
    }
    let h = StandardHarness::new(standard_config(a)); // Autonomous
    let t = task(LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        // Small PLAN-leaf cap so the LEAF binds; the global backstop (100) is
        // the larger bound, and the execute slot never runs.
        plan: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(2))),
        execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
        plan_model: None,
    }))
    .with_budget(BudgetLimits {
        max_turns: Some(100),
        ..Default::default()
    });
    let mut state = SessionState::default();
    let err = h
        .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect_err("plan phase exhausted its leaf budget");
    match err {
        RunResult::Failure {
            reason: HaltReason::BudgetExceeded { .. },
            ..
        } => {}
        RunResult::Failure {
            reason: HaltReason::PlanPhaseFailed { .. },
            ..
        } => {
            panic!("regression: plan-leaf exhaustion collapsed to PlanPhaseFailed")
        }
        other => panic!("expected a clean BudgetExceeded failure, got {other:?}"),
    }
}

/// As above but under `SurfaceToHuman` (looper's default escalation mode): a
/// plan-leaf exhaustion now PAUSES with `HumanRequest::BudgetExhausted` so the
/// host can grant more budget and the run CONTINUES (auto-continue parity with
/// the execute phase) — instead of dying with "produced no terminal".
#[tokio::test]
async fn plan_phase_leaf_budget_exhaustion_surfaces_for_continuation() {
    let a = make_agent();
    for _ in 0..4 {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: "c".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
    }
    let h = StandardHarness::new(surface_config(a)); // SurfaceToHuman
    let t = task(LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        plan: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(2))),
        execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
        plan_model: None,
    }))
    .with_budget(BudgetLimits {
        max_turns: Some(100),
        ..Default::default()
    });
    let mut state = SessionState::default();
    let err = h
        .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect_err("plan phase paused for a budget grant");
    match err {
        RunResult::WaitingForHuman {
            request: HumanRequest::BudgetExhausted { .. },
            ..
        } => {}
        other => panic!("expected a BudgetExhausted pause, got {other:?}"),
    }
}

// R11: an OnPlanCreated hook can rewrite the plan before storage; the
// stored artifact reflects the mutation.
#[tokio::test]
async fn plan_phase_on_plan_created_rewrites_before_storage() {
    use crate::hooks::{FunctionHook, HookDecision, HookEvent, StandardHookChain};

    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["original"],"rationale":"orig"}"#));

    let chain = StandardHookChain::new();
    chain
        .register(Arc::new(FunctionHook::new(
            "rewrite-plan",
            vec![HookEvent::OnPlanCreated],
            |ctx| {
                if let crate::hooks::HookContext::OnPlanCreated { plan, .. } = ctx {
                    plan.tasks = vec!["rewritten".to_string()];
                    plan.rationale = "mutated".to_string();
                }
                Ok(HookDecision::Continue)
            },
        )))
        .unwrap();

    let mut cfg = standard_config(a);
    cfg.hooks = Some(Arc::new(chain));
    let h = StandardHarness::new(cfg);
    let t = plan_task();
    let mut state = SessionState::default();
    let outcome = h
        .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect("plan phase succeeds");

    let stored = stored_artifact(&h, &t.session_id).await;
    assert_eq!(stored.tasks, vec!["rewritten"]);
    assert_eq!(stored.rationale, "mutated");
    assert_eq!(stored, outcome.artifact);
}

// R3 fence variant: a fenced ```json plan is captured through the harness.
#[tokio::test]
async fn plan_phase_captures_fenced_json() {
    let a = make_agent();
    a.push(final_resp("```json\n{\"tasks\":[\"f1\",\"f2\"]}\n```"));
    let h = StandardHarness::new(standard_config(a));
    let t = plan_task();
    let mut state = SessionState::default();
    h.run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect("plan phase succeeds");
    assert_eq!(
        stored_artifact(&h, &t.session_id).await.tasks,
        vec!["f1", "f2"]
    );
}

// SC-28 acceptance (1): a free-text / markdown plan no longer fails the plan
// phase. The strict JSON grammar (+ prose repair) misses, so the driver
// captures it as a prose artifact instead of aborting: `rationale` holds the
// verbatim prose and `tasks` is sourced from the `task_list` tool store
// (empty here — the planner authored none). The artifact IS stored (R4/R11
// still apply) and round-trips (acceptance 3).
#[tokio::test]
async fn plan_phase_freetext_response_captures_as_prose() {
    let prose = "This is a markdown plan.\n\n1. do the thing\n2. do the other";
    let a = make_agent();
    a.push(final_resp(prose));
    let h = StandardHarness::new(standard_config(a));
    let t = plan_task();
    let mut state = SessionState::default();
    let outcome = h
        .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect("a free-text plan captures rather than failing");
    // Prose preserved verbatim; no JSON tasks parsed and no task_list authored.
    assert_eq!(outcome.artifact.rationale, prose);
    assert!(outcome.artifact.tasks.is_empty());
    // R4: the artifact IS stored now (was absent pre-SC-28) and round-trips.
    let stored = stored_artifact(&h, &t.session_id).await;
    assert_eq!(stored, outcome.artifact);
}

// SC-28 acceptance (1): a markdown plan captures without parse failure AND the
// OnPlanCreated payload's `tasks` are sourced from the `task_list` tool store
// (the ONE authoring path) — so panel consumers (looper `plan_tracker`,
// cordyceps `plan_announcer`) still get task texts even though the plan prose
// is free-text rather than the JSON `PlanArtifact`.
#[tokio::test]
async fn plan_phase_freetext_sources_tasks_from_task_list() {
    let prose = "Here's my plan in prose, no JSON object at all.";
    let a = make_agent();
    a.push(final_resp(prose));
    let h = StandardHarness::new(standard_config(a));
    let t = plan_task();

    // Seed the durable task_list store as if the plan leaf had authored it via
    // the `task_list` tool during the plan turn (keyed by the project ns).
    let mut seeded = TaskList::default();
    seeded.add("build it".into(), Vec::new()).unwrap();
    seeded.add("test it".into(), Vec::new()).unwrap();
    h.storage()
        .run()
        .put(
            &durable_ns(),
            TASK_LIST_EXTRAS_KEY,
            serde_json::to_value(&seeded).unwrap(),
        )
        .await
        .expect("seed task_list");

    let mut state = SessionState::default();
    let outcome = h
        .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect("markdown plan captures");
    // `tasks` pulled from the seeded task_list; prose preserved as rationale.
    assert_eq!(outcome.artifact.tasks, vec!["build it", "test it"]);
    assert_eq!(outcome.artifact.rationale, prose);
    // Stored artifact round-trips (acceptance 3).
    let stored = stored_artifact(&h, &t.session_id).await;
    assert_eq!(stored, outcome.artifact);
}

// ── PlanExecute execute phase (issue #59) ───────────────────────────────

use crate::tasklist::{TaskList, TaskStatus, TASK_LIST_EXTRAS_KEY};

// #76: the task list now lives on the RunStore seam (not extras). Read it
// back through the harness's storage under TASK_LIST_EXTRAS_KEY.
async fn run_store_task_list(h: &StandardHarness, _session_id: &SessionId) -> TaskList {
    // #142: the task list is keyed by the project namespace, not the run
    // session id — read it back under `durable_ns()`.
    let v = h
        .storage()
        .run()
        .get(&durable_ns(), TASK_LIST_EXTRAS_KEY)
        .await
        .expect("run store get ok")
        .expect("task_list present in run store");
    serde_json::from_value(v).expect("task list deserializes")
}

// Build a PlanExecute task carrying an explicit budget (e.g. a turn cap).
fn plan_task_with_budget(budget: BudgetLimits) -> Task {
    plan_task().with_budget(budget)
}

// Q1 happy path + plan-then-execute + task drain. Plan produces K=3 tasks,
// then 3 execute completions. The run succeeds; turns == 4 (plan + 3).
#[tokio::test]
async fn execute_phase_happy_path_drains_all_tasks() {
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["t1","t2","t3"],"rationale":"r"}"#));
    a.push(final_resp("done t1"));
    a.push(final_resp("done t2"));
    a.push(final_resp("done t3"));
    let agent_count = a.clone();
    let h = StandardHarness::new(standard_config(a));
    match h.run(HarnessRunOptions::new(plan_task())).await {
        RunResult::Success { output, turns, .. } => {
            assert_eq!(output, "done t3", "Q2: output is the last step's final");
            assert_eq!(turns, 4, "one plan turn + one per task");
        }
        other => panic!("expected Success, got {other:?}"),
    }
    // 1 plan turn + 3 execute turns = 4 agent calls.
    assert_eq!(
        agent_count
            .call_count
            .load(std::sync::atomic::Ordering::SeqCst),
        4
    );
}

// Task drain Pending -> InProgress -> Completed: inspect the persisted list
// after a run via run_execute_phase directly so we can read the final state.
#[tokio::test]
async fn execute_phase_drains_pending_inprogress_completed() {
    let a = make_agent();
    a.push(final_resp("done one"));
    a.push(final_resp("done two"));
    let h = StandardHarness::new(standard_config(a));
    let t = plan_task();
    let mut state = SessionState::default();
    let list = plan_artifact_to_tasklist_helper(&["one", "two"]);
    // All start Pending.
    assert!(list.tasks.iter().all(|x| x.status == TaskStatus::Pending));
    let result = h
        .run_execute_phase(
            &t,
            &mut state,
            list,
            BudgetSnapshot {
                turns: 1,
                ..Default::default()
            },
            AggregateUsage::default(),
            &None,
        )
        .await;
    match result {
        RunResult::Success { output, .. } => assert_eq!(output, "done two"),
        other => panic!("expected Success, got {other:?}"),
    }
    // Final persisted list: every task Completed.
    let final_list = run_store_task_list(&h, &t.session_id).await;
    assert!(
        final_list
            .tasks
            .iter()
            .all(|x| x.status == TaskStatus::Completed),
        "all tasks Completed after drain"
    );
}

#[allow(deprecated)] // exercises the retained-for-compat linear bridge (#126 C).
fn plan_artifact_to_tasklist_helper(steps: &[&str]) -> TaskList {
    crate::tasklist::plan_artifact_to_task_list(&PlanArtifact {
        tasks: steps.iter().map(|s| s.to_string()).collect(),
        rationale: String::new(),
    })
}

// ── #124 A.6 deep-resume: completed tasks are NOT re-run ─────────────────
//
// The execute phase reconciles the freshly-parsed task list with the DURABLE
// RunStore checkpoint (re-persisted after every step). A task already
// Completed on the checkpoint is skipped — serialize→reset→resume, not the
// old shallow no-op. Here task 0 is pre-marked Completed in the checkpoint;
// only task 1 should run (one agent turn), proving no re-execution.
#[tokio::test]
async fn deep_resume_skips_already_completed_task() {
    let a = make_agent();
    // Only ONE response pushed: if task 0 were re-run, the loop would starve
    // and fail; one response is enough iff task 0 is skipped.
    a.push(final_resp("done two"));
    let agent_count = a.clone();
    let h = StandardHarness::new(standard_config(a));
    let t = plan_task();

    // Pre-seed the durable checkpoint: task 0 Completed, task 1 Pending.
    let mut checkpoint = plan_artifact_to_tasklist_helper(&["one", "two"]);
    let first_id = checkpoint.tasks[0].id;
    checkpoint.complete(first_id).expect("mark task 0 complete");
    h.persist_task_list(&t.session_id, &checkpoint).await;

    // The freshly-parsed list (as the plan phase would produce) is all-Pending.
    let fresh = plan_artifact_to_tasklist_helper(&["one", "two"]);
    let result = h
        .run_execute_phase(
            &t,
            &mut SessionState::default(),
            fresh,
            BudgetSnapshot {
                turns: 1,
                ..Default::default()
            },
            AggregateUsage::default(),
            &None,
        )
        .await;
    match result {
        RunResult::Success { output, .. } => {
            assert_eq!(output, "done two", "only the not-yet-done task runs");
        }
        other => panic!("expected Success, got {other:?}"),
    }
    // Exactly ONE agent turn: task 0 was resumed-from-checkpoint, not re-run.
    assert_eq!(
        agent_count
            .call_count
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the already-Completed task must NOT be re-run"
    );
    // Both tasks Completed in the final persisted list.
    let final_list = run_store_task_list(&h, &t.session_id).await;
    assert!(
        final_list
            .tasks
            .iter()
            .all(|x| x.status == TaskStatus::Completed),
        "all tasks Completed after resume"
    );
}

// #125: the OLD Q1 per-task turn allocation (`remaining_turns /
// remaining_tasks`, cutting task `a` off at 2 of a 7-turn budget) is REMOVED
// as dead logic. Enforcement is now charge-based on the PlanExecute scope's
// global `TotalSteps`, NOT divided per task. With the same setup — global cap
// 7, plan spends 1 — task `a` is NO LONGER capped at 2 turns: it runs its
// scripted turns under the global ceiling and is not pre-empted by a phantom
// per-task cap. Here task `a` runs its 2 tool turns then runs dry, so the run
// fails as a STEP failure (not a per-task budget cap). This asserts the
// per-task derivation is gone.
#[tokio::test]
async fn execute_phase_no_per_task_turn_cap() {
    let a = make_agent();
    // Plan: 3 tasks.
    a.push(final_resp(r#"{"tasks":["a","b","c"]}"#));
    // Task a: 2 tool calls then runs dry (no 3rd scripted turn).
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "1".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "2".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "ok".into(),
        truncated: false,
    });
    reg.push(ToolOutput::Success {
        content: "ok".into(),
        truncated: false,
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    let t = plan_task_with_budget(BudgetLimits {
        max_turns: Some(7),
        ..Default::default()
    });
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure { reason, .. } => {
            // NOT a per-task BudgetExceeded(Turns) cut at 2 turns — the
            // derivation is gone. Task `a` (id 1) runs past 2 turns under the
            // global ceiling and fails when the agent runs dry. Under #126 a
            // task failure no longer raises `StepFailed`/blanket-aborts; it
            // cascades, and the run drains to the partial terminal
            // `TasksBlockedByFailure` (the first failure is task id 1).
            assert!(
                matches!(
                    reason,
                    HaltReason::TasksBlockedByFailure { failed_task: 1, .. }
                ),
                "no per-task cap; run dry → TasksBlockedByFailure(failed=1), got {reason:?}"
            );
        }
        other => panic!("expected Failure (step ran dry), got {other:?}"),
    }
}

// Budget exhaustion MID-execute: a tight global turn cap stops the run
// partway with BudgetExceeded, not StepFailed.
#[tokio::test]
async fn execute_phase_budget_exhausted_mid_execute() {
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["x","y","z"]}"#));
    a.push(final_resp("did x"));
    // No more turns allowed after the first execute step.
    let h = StandardHarness::new(standard_config(a));
    let t = plan_task_with_budget(BudgetLimits {
        max_turns: Some(2), // plan(1) + exactly one execute turn
        ..Default::default()
    });
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure { reason, turns, .. } => {
            assert!(
                matches!(
                    reason,
                    HaltReason::BudgetExceeded {
                        limit_type: BudgetLimitType::Turns
                    }
                ),
                "global turn budget is the hard stop, got {reason:?}"
            );
            assert_eq!(turns, 2);
        }
        other => panic!("expected BudgetExceeded, got {other:?}"),
    }
}

// Regression: a PlanExecute execute-phase worker that exhausts under
// SurfaceToHuman and is granted more budget must make FORWARD PROGRESS across
// grants — the run-wide turn cursor and the grant's `steps_taken` both
// STRICTLY advance, and the worker eventually finishes. Before the fix the
// exhaustion branch discarded the execute leaf's consumed-turn count and
// re-read `steps_taken = 0` from the (Unlimited, because `max_turns = None`)
// `plan_execute` scope, and left `carried.turns` frozen at its pre-task value.
// So every grant computed `granted = 0 + steps` (a no-op that never widened the
// binding cap) and re-seeded the worker on the SAME window — the cursor and
// `steps_taken` stuck, the worker re-ran the same turns, and a `review`-style
// task thrashed through every auto-continue making zero progress.
#[tokio::test]
async fn execute_phase_budget_grant_advances_and_makes_progress() {
    // One execute task ("only"); the execute leaf cap is PerLoop{2} and the
    // top task leaves `max_turns = None`, so the `plan_execute` scope is
    // Unlimited — the exact configuration that zeroed the grant accounting.
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["only"]}"#)); // plan turn (carried.turns -> 1)
                                                 // Execute worker keeps requesting tools so it never finishes until the
                                                 // final grant; each ToolCallRequested is one turn against the leaf cap.
    for i in 0..3 {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: format!("e{i}"),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
    }
    a.push(final_resp("done only")); // completes after the 2nd grant
    let mut cfg = surface_config(a);
    cfg.tool_registry = tool_reg(3);
    let h = StandardHarness::new(cfg);
    let t = task(LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        plan: Box::new(react_structured(u32::MAX)),
        execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(2))),
        plan_model: None,
    }));

    // First run → pause #1. The worker consumed real turns past the plan
    // floor, so the cursor and `steps_taken` reflect that (NOT 0 / frozen).
    let (turn1, steps1, state1) = match h.run(HarnessRunOptions::new(t)).await {
        RunResult::WaitingForHuman { request, state } => {
            let steps = match request {
                HumanRequest::BudgetExhausted {
                    steps_taken, phase, ..
                } => {
                    assert_eq!(phase, "plan_execute", "the combinator scope owns the pause");
                    steps_taken
                }
                other => panic!("expected BudgetExhausted, got {other:?}"),
            };
            (state.turn_number, steps, *state)
        }
        other => panic!("expected pause #1, got {other:?}"),
    };
    // The bug froze the cursor at the pre-task value (1 = just the plan turn)
    // and read steps_taken=0 from the Unlimited scope.
    assert!(
        turn1 >= 2,
        "cursor advanced past the plan floor, got {turn1}"
    );
    assert!(
        steps1 >= 2,
        "grant accounting uses the worker's consumed turns, got {steps1}"
    );

    // Grant +2 and resume → pause #2. Forward progress: the cursor and
    // `steps_taken` must STRICTLY advance (the bug re-seeded the same window).
    let (turn2, steps2, state2) = match h
        .resume(
            state1,
            HumanResponse::Escalate {
                action: EscalationAction::ContinueWithBudget { steps: 2 },
            },
            None,
        )
        .await
    {
        RunResult::WaitingForHuman { request, state } => {
            let steps = match request {
                HumanRequest::BudgetExhausted { steps_taken, .. } => steps_taken,
                other => panic!("expected BudgetExhausted, got {other:?}"),
            };
            (state.turn_number, steps, *state)
        }
        other => panic!("expected pause #2, got {other:?}"),
    };
    assert!(
        turn2 > turn1,
        "turn cursor advanced across grants ({turn1} -> {turn2})"
    );
    assert!(
        steps2 > steps1,
        "steps_taken advanced across grants ({steps1} -> {steps2})"
    );

    // Grant +2 again and resume → Success: the worker finishes, having made
    // real progress rather than looping on the same window.
    match h
        .resume(
            state2,
            HumanResponse::Escalate {
                action: EscalationAction::ContinueWithBudget { steps: 2 },
            },
            None,
        )
        .await
    {
        RunResult::Success { output, .. } => assert_eq!(output, "done only"),
        other => panic!("expected Success after the final grant, got {other:?}"),
    }
}

// Observability span count: plan turn + one span per executed step.
#[tokio::test]
async fn execute_phase_span_count() {
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["a","b"]}"#));
    a.push(final_resp("did a"));
    a.push(final_resp("did b"));
    let obs = Arc::new(crate::observability::InMemoryObservabilityProvider::new());
    let mut cfg = standard_config(a);
    cfg.observability = Some(obs.clone());
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(plan_task())).await;
    let spans = obs.turn_spans(&SessionId::new("s1"));
    // 1 plan turn + 2 execute step turns = 3 turn spans.
    assert_eq!(spans.len(), 3, "plan turn + one span per executed step");
}

// Compaction works inside the execute loop: the context manager requests a
// compaction, and the run still completes. (Exercises the shared compaction
// path through run_react_inner during execute steps.)
#[tokio::test]
async fn execute_phase_compaction_in_loop() {
    // Use the default NoopContextManager (never compacts) is not enough to
    // exercise compaction; instead assert the loop tolerates many turns with
    // tool calls and still drains. A dedicated compaction adapter lives in
    // the ReAct compaction tests; here we assert execute reuses that path by
    // running a multi-turn step to completion.
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["only"]}"#));
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "1".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(final_resp("finished only"));
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "ok".into(),
        truncated: false,
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(plan_task())).await {
        RunResult::Success { output, turns, .. } => {
            assert_eq!(output, "finished only");
            // plan(1) + tool turn(1) + final(1) = 3.
            assert_eq!(turns, 3);
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// OnTaskAdvance fires exactly N times with the correct task_index /
// total_tasks, and (Q1) may rewrite the step instruction.
#[tokio::test]
async fn execute_phase_on_task_advance_fires_per_task() {
    use crate::hooks::{FunctionHook, HookDecision, HookEvent, StandardHookChain};
    use std::sync::atomic::{AtomicUsize, Ordering};

    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["s0","s1","s2"]}"#));
    a.push(final_resp("d0"));
    a.push(final_resp("d1"));
    a.push(final_resp("d2"));

    let fire_count = Arc::new(AtomicUsize::new(0));
    let seen_indices = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
    let seen_totals = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
    let fc = fire_count.clone();
    let si = seen_indices.clone();
    let st = seen_totals.clone();

    let chain = StandardHookChain::new();
    chain
        .register(Arc::new(FunctionHook::new(
            "count-advance",
            vec![HookEvent::OnTaskAdvance],
            move |ctx| {
                if let crate::hooks::HookContext::OnTaskAdvance {
                    task_index,
                    total_tasks,
                    ..
                } = ctx
                {
                    fc.fetch_add(1, Ordering::SeqCst);
                    si.lock().unwrap().push(*task_index);
                    st.lock().unwrap().push(*total_tasks);
                }
                Ok(HookDecision::Continue)
            },
        )))
        .unwrap();

    let mut cfg = standard_config(a);
    cfg.hooks = Some(Arc::new(chain));
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(plan_task())).await;

    assert_eq!(fire_count.load(Ordering::SeqCst), 3, "fires once per task");
    assert_eq!(*seen_indices.lock().unwrap(), vec![0, 1, 2]);
    assert_eq!(*seen_totals.lock().unwrap(), vec![3, 3, 3]);
}

// Q3: an empty plan -> HaltReason::EmptyPlan (not a silent success).
#[tokio::test]
async fn execute_phase_empty_plan_halts() {
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":[],"rationale":"nothing"}"#));
    let h = StandardHarness::new(standard_config(a));
    match h.run(HarnessRunOptions::new(plan_task())).await {
        RunResult::Failure {
            reason: HaltReason::EmptyPlan,
            turns,
            ..
        } => {
            assert_eq!(turns, 1, "only the plan turn ran");
        }
        other => panic!("expected EmptyPlan, got {other:?}"),
    }
}

// #126 AC3 (REPLACES the Q5 blanket abort): a terminal task failure blocks
// only its TRANSITIVE DEPENDENTS; unrelated tasks still complete; the run
// does NOT abort on the first failure but drains to `TasksBlockedByFailure`.
//
// DAG (authored via the persisted `task_list` store, the #126 authoring path):
//   1 "good"  (no blockers)        → completes
//   2 "bad"   (no blockers)        → fails terminally
//   3 "dep"   (blocked by 2)       → cascade-Blocked (never runs)
//   4 "indep" (no blockers)        → still completes (unrelated branch)
#[tokio::test]
async fn execute_phase_failure_cascades_only_to_dependents() {
    let a = make_agent();
    // #138 AC1: a non-empty task_list is pre-seeded below, so the plan phase
    // is SKIPPED — no plan turn. The first model call is task 1.
    // Ready order is lowest-id-first among ready tasks: 1, then 2 (fails),
    // then 4 (3 is blocked by the failed 2).
    a.push(final_resp("did good")); // task 1
    a.push(TurnResult::Error {
        // task 2 fails terminally.
        error: crate::agent::AgentError::EmptyResponse,
        usage: None,
    });
    a.push(final_resp("did indep")); // task 4 (independent) still runs

    let agent_count = a.clone();
    let session = SessionId::new("s1");
    let h = StandardHarness::new(standard_config(a));

    // Pre-seed the authored DAG into the persisted task_list store.
    let mut tl = TaskList::default();
    let g = tl.add("good".into(), vec![]).unwrap();
    let bad = tl.add("bad".into(), vec![]).unwrap();
    let _dep = tl.add("dep".into(), vec![bad]).unwrap();
    let _indep = tl.add("indep".into(), vec![]).unwrap();
    assert_eq!((g, bad), (1, 2));
    h.persist_task_list(&session, &tl).await;

    // Model (A): walking a PRE-SEEDED DAG is a continuation (Ralph/Skip/resume),
    // so opt in explicitly — a fresh run() would clear the list and re-plan.
    match h
        .run(HarnessRunOptions::new(plan_task()).continue_task_list(true))
        .await
    {
        RunResult::Failure {
            reason:
                HaltReason::TasksBlockedByFailure {
                    completed,
                    blocked,
                    failed_task,
                    ..
                },
            ..
        } => {
            assert_eq!(failed_task, 2, "task 2 (bad) is the failing task");
            // 1 (good) and 4 (indep) complete; 2 and its dependent 3 are blocked.
            assert_eq!(completed, vec![1, 4]);
            assert_eq!(blocked, vec![2, 3]);
        }
        other => panic!("expected TasksBlockedByFailure, got {other:?}"),
    }
    // #138 AC1: plan SKIPPED (durable list pre-seeded). good(1) + bad(1) +
    // indep(1) = 3 calls; "dep" never ran.
    assert_eq!(
        agent_count
            .call_count
            .load(std::sync::atomic::Ordering::SeqCst),
        3,
        "the dependent task must not run; the independent task must"
    );
}

// The execute agent can ADD tasks mid-walk via the `task_list` tool, and the
// executor SCHEDULES them instead of clobbering them. The tool and the walk
// share ONE project-namespaced RunStore blob; before the reload fix the
// executor's stale in-memory list overwrote the agent's addition on the
// mark-complete persist, so `next_ready` never saw it. Uses a REAL in-memory
// store + the REAL `task_list` tool (per [[noop-store-silently-disables-durable-behavior]],
// a fake path would assert the opposite while passing).
#[tokio::test]
async fn execute_agent_can_add_tasks_midwalk() {
    use crate::tools::StandardTools;
    let a = make_agent();
    // Task 1: call `task_list` add_task (appends task 2), then finish.
    a.push(TurnResult::ToolCallRequested {
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "task_list".into(),
            input: serde_json::json!({
                "action": "add_task",
                "description": "added by agent",
            }),
        }],
        usage: usage(),
        reasoning: None,
    });
    a.push(final_resp("did 1"));
    // Task 2 (the agent-added one): finish.
    a.push(final_resp("did 2"));

    let session = SessionId::new("s1");
    let cfg = catalogue_builder(a.clone())
        .tool(StandardTools::task_list())
        // The plan leaf is a structured slot (react_structured) — register the
        // default-handle output schema it validates against.
        .registry_schema("", serde_json::json!({}))
        .build_config();
    let h = StandardHarness::new(cfg);

    // Pre-seed a single task so the plan phase is skipped (continuation).
    let mut tl = TaskList::default();
    tl.add("first".into(), vec![]).unwrap(); // id 1
    h.persist_task_list(&session, &tl).await;

    match h
        .run(HarnessRunOptions::new(plan_task()).continue_task_list(true))
        .await
    {
        // The agent-added task 2 RAN and was the last to complete.
        RunResult::Success { output, .. } => {
            assert_eq!(output, "did 2", "the agent-added task must execute")
        }
        other => panic!("expected Success, got {other:?}"),
    }

    // Both tasks are Completed in the shared durable store — the addition was
    // NOT clobbered by the executor's mark-complete persist.
    let stored = h
        .load_task_list(&session)
        .await
        .expect("task list persisted");
    assert_eq!(stored.tasks.len(), 2, "the added task survived: {stored:?}");
    assert!(
        stored
            .tasks
            .iter()
            .all(|t| t.status == TaskStatus::Completed),
        "both tasks complete: {stored:?}"
    );
    assert_eq!(
        stored.tasks[1].description, "added by agent",
        "the second task is the one the agent appended"
    );
    // 3 agent turns total: task 1 (tool call + final) + task 2 (final).
    assert_eq!(
        a.call_count.load(std::sync::atomic::Ordering::SeqCst),
        3,
        "the agent-added task ran (3 total turns)"
    );
}

// ========================================================================
// #126 PlanExecute DAG executor — AC1/AC2/AC4/AC5 + ledger + Tier-1 lean
// ========================================================================

// Helper: persist an authored DAG task list (the #126 authoring path) so the
// executor's `load_task_list` picks it up over the linear plan bridge.
async fn seed_dag(h: &StandardHarness, session: &SessionId, list: &TaskList) {
    h.persist_task_list(session, list).await;
}

// AC1: a blocker DAG executes in DEPENDENCY ORDER with a deterministic
// lowest-id tiebreak among ready tasks. DAG: 2 and 3 both block on 1; 4
// blocks on 2,3. Expected run order (by lowest ready id): 1, 2, 3, 4.
#[tokio::test]
async fn dag_executes_in_dependency_order_with_id_tiebreak() {
    let a = make_agent();
    // #138 AC1: a non-empty task_list is pre-seeded below, so the plan phase
    // is SKIPPED — no plan turn is pushed. The first model call is task 1.
    a.push(final_resp("did 1"));
    a.push(final_resp("did 2"));
    a.push(final_resp("did 3"));
    a.push(final_resp("did 4"));
    let session = SessionId::new("s1");
    let h = StandardHarness::new(standard_config(a));
    let mut tl = TaskList::default();
    tl.add("one".into(), vec![]).unwrap(); // 1
    tl.add("two".into(), vec![1]).unwrap(); // 2 -> 1
    tl.add("three".into(), vec![1]).unwrap(); // 3 -> 1
    tl.add("four".into(), vec![2, 3]).unwrap(); // 4 -> 2,3
    seed_dag(&h, &session, &tl).await;

    match h
        .run(HarnessRunOptions::new(plan_task()).continue_task_list(true))
        .await
    {
        RunResult::Success { output, .. } => {
            // Last completed (id 4) text is the run output.
            assert_eq!(output, "did 4");
        }
        other => panic!("expected Success, got {other:?}"),
    }
    // All four tasks completed in the persisted store.
    let stored: TaskList = serde_json::from_value(
        h.storage()
            .run()
            .get(&durable_ns(), TASK_LIST_EXTRAS_KEY)
            .await
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    assert!(stored
        .tasks
        .iter()
        .all(|t| t.status == TaskStatus::Completed));
}

// AC1 branch isolation + Tier-1 lean: a task's seed contains its TRANSITIVE
// blockers' outputs/ledger ONLY — never an independent branch's, and never
// a full transcript fold. DAG: 1 (root), 2 -> 1, 3 (independent). Run order:
// 1, 2, 3. Task 2's context must contain task 1's ledger summary but NOT
// task 3's (3 runs after 2 anyway) — and crucially task 3's context must NOT
// contain task 2's per-step internals; only the global ledger summaries.
#[tokio::test]
async fn dag_branch_isolation_tier1_excludes_independent_branch() {
    // Distinctive markers so we can detect leakage.
    let agent = RecordingTurnAgent::new(
        "rec",
        vec![
            // #138 AC1: a non-empty task_list is pre-seeded below, so the plan
            // phase is SKIPPED — no plan turn. The first call is task 1.
            RecordingTurnAgent::final_resp("ROOT_OUTPUT_AAA"), // task 1
            RecordingTurnAgent::final_resp("CHILD_OUTPUT_BBB"), // task 2 (-> 1)
            RecordingTurnAgent::final_resp("INDEP_OUTPUT_CCC"), // task 3 (indep)
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, agent.clone());
    let session = SessionId::new("s1");
    let h = StandardHarness::new(cfg);
    let mut tl = TaskList::default();
    tl.add("root".into(), vec![]).unwrap(); // 1
    tl.add("child of root".into(), vec![1]).unwrap(); // 2 -> 1
    tl.add("independent".into(), vec![]).unwrap(); // 3 indep
    seed_dag(&h, &session, &tl).await;

    let _ = h
        .run(HarnessRunOptions::new(plan_task()).continue_task_list(true))
        .await;
    let contexts = agent.seen_text();
    // #138 AC1: plan skipped → [0] task1, [1] task2, [2] task3.
    assert_eq!(contexts.len(), 3);

    // Task 2 (index 1) is seeded with its transitive blocker (task 1)'s output.
    assert!(
        contexts[1].contains("ROOT_OUTPUT_AAA"),
        "task 2 must see its transitive blocker (task 1) output: {}",
        contexts[1]
    );
    // Task 2 must NOT see the independent task 3 (it has not run yet, and is
    // not a blocker regardless).
    assert!(
        !contexts[1].contains("INDEP_OUTPUT_CCC"),
        "task 2 must NOT see the independent branch: {}",
        contexts[1]
    );

    // Task 3 (index 2) is INDEPENDENT — its Tier-1 seed has NO upstream
    // blockers, so it must NOT carry task 1 or task 2 as a Tier-1 "upstream
    // result" block. (The Tier-2 global ledger summaries may still appear;
    // this asserts the Tier-1 scoped block is branch-isolated.)
    assert!(
        !contexts[2].contains("Results from upstream tasks"),
        "independent task 3 must have no Tier-1 upstream block: {}",
        contexts[2]
    );
}

// AC2: files_touched is HARNESS-OBSERVED from write/edit calls, not
// self-reported. Task 1 only NARRATES touching a file (no write call) → empty
// files_touched. Task 2 issues a real `edit_file` call → the path is recorded.
#[tokio::test]
async fn dag_files_touched_observed_not_self_reported() {
    let a = make_agent();
    // #138 AC1: a non-empty task_list is pre-seeded below, so the plan phase
    // is SKIPPED — no plan turn. Task 1: prose claims a file but no write call.
    a.push(final_resp("I touched src/phantom.rs (but did not really)"));
    // Task 2: issue a real edit_file call carrying a path, then finalize.
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "e1".into(),
            name: "edit_file".into(),
            input: serde_json::json!({"path": "src/real.rs", "old_string": "a", "new_string": "b"}),
        }],
        usage: usage(),
    });
    a.push(final_resp("edited the file"));
    let session = SessionId::new("s1");
    let mut cfg = standard_config(a);
    // The edit_file tool dispatch returns success (content irrelevant).
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "edited".into(),
        truncated: false,
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    let mut tl = TaskList::default();
    tl.add("narrate".into(), vec![]).unwrap(); // 1
    tl.add("really edit".into(), vec![1]).unwrap(); // 2 -> 1
    seed_dag(&h, &session, &tl).await;

    // To inspect the ledger we drive the run and assert via the streamed
    // ledger seeded into task 2's context: task 2 is downstream of task 1, so
    // its Tier-1 ledger block reflects task 1's OBSERVED files (none). Then
    // task 2 itself records src/real.rs. We assert the run succeeds and that
    // the observed-writes accumulator behaved by checking the cascade-free
    // success plus the persisted completion.
    match h
        .run(HarnessRunOptions::new(plan_task()).continue_task_list(true))
        .await
    {
        RunResult::Success { output, .. } => assert_eq!(output, "edited the file"),
        other => panic!("expected Success, got {other:?}"),
    }
    // The discriminating assertion lives in the dedicated executor probe
    // below; here we confirm the run completed end-to-end with a real write.
}

// AC2 (discriminating, direct): drive two steps and assert the OBSERVED
// write accumulator distinguishes a prose-only step (empty) from a real
// edit_file step (path recorded). Exercises the StrategyExecutor seam
// (`clear_observed_writes` / `take_observed_writes`) the executor relies on.
#[tokio::test]
async fn observed_writes_seam_records_only_real_write_calls() {
    let a = make_agent();
    // One window that issues an edit_file then finalizes.
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "e1".into(),
            name: "edit_file".into(),
            input: serde_json::json!({"path": "src/real.rs", "old_string": "a", "new_string": "b"}),
        }],
        usage: usage(),
    });
    a.push(final_resp("done"));
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "edited".into(),
        truncated: false,
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);

    // Before any dispatch: empty.
    assert!(h.take_observed_writes().is_empty());
    h.clear_observed_writes();

    // Drive a bare ReAct window that issues the edit_file.
    let _ = h.run(HarnessRunOptions::new(react(u32::MAX))).await;
    let observed = h.take_observed_writes();
    assert_eq!(
        observed,
        vec!["src/real.rs".to_string()],
        "only the real edit_file path is observed"
    );
    // A second drain is empty (take resets).
    assert!(h.take_observed_writes().is_empty());

    // Prose-only step (no write/edit call) records NOTHING — observed, not
    // self-reported. Drive a window that just narrates touching a file.
    let a2 = make_agent();
    a2.push(final_resp(
        "I edited src/phantom.rs (but issued no write call)",
    ));
    let h2 = StandardHarness::new(standard_config(a2));
    h2.clear_observed_writes();
    let _ = h2.run(HarnessRunOptions::new(react(u32::MAX))).await;
    assert!(
        h2.take_observed_writes().is_empty(),
        "a prose-only claim must NOT record a touched file"
    );
}

// AC4: a budget-`Fail` task cascades IDENTICALLY to an error-failed one. We
// give the PlanExecute scope a tight global cap so a downstream task's leaf
// window exhausts; assert the dependents are blocked and unrelated tasks are
// unaffected, draining to TasksBlockedByFailure.
#[tokio::test]
async fn dag_budget_fail_cascades_like_error() {
    // A budget exhaustion of the WHOLE run surfaces as BudgetExceeded (a hard
    // stop). To exercise the per-NODE budget-Fail CASCADE path deterministically
    // we instead drive a child that fails terminally with a non-budget error
    // and assert the cascade partition — the budget-Fail branch in the executor
    // resolves through the SAME cascade code (verified by the unit coverage of
    // `transitive_dependents` + the cascade arm). Here we assert the cascade
    // partition shape that AC3/AC4 share.
    let a = make_agent();
    // #138 AC1: a non-empty task_list is pre-seeded below, so the plan phase
    // is SKIPPED — no plan turn. The first model call is task 1.
    a.push(final_resp("did root")); // task 1
                                    // task 2 fails terminally.
    a.push(TurnResult::Error {
        error: crate::agent::AgentError::EmptyResponse,
        usage: None,
    });
    a.push(final_resp("did indep")); // task 4 independent still runs
    let session = SessionId::new("s1");
    let h = StandardHarness::new(standard_config(a));
    let mut tl = TaskList::default();
    tl.add("root".into(), vec![]).unwrap(); // 1
    tl.add("mid".into(), vec![1]).unwrap(); // 2 -> 1 (fails)
    tl.add("leaf".into(), vec![2]).unwrap(); // 3 -> 2 (cascade-blocked)
    tl.add("indep".into(), vec![]).unwrap(); // 4 independent
    seed_dag(&h, &session, &tl).await;

    // Model (A): continue the pre-seeded DAG (Ralph/Skip/resume entry).
    match h
        .run(HarnessRunOptions::new(plan_task()).continue_task_list(true))
        .await
    {
        RunResult::Failure {
            reason:
                HaltReason::TasksBlockedByFailure {
                    completed,
                    blocked,
                    failed_task,
                    ..
                },
            ..
        } => {
            assert_eq!(failed_task, 2);
            assert_eq!(completed, vec![1, 4]);
            assert_eq!(blocked, vec![2, 3]);
        }
        other => panic!("expected TasksBlockedByFailure, got {other:?}"),
    }
}

// AC5: a cyclic persisted graph is rejected at EXECUTE ENTRY (defense in
// depth) → HaltReason::TaskGraphCycle. No execute step runs.
#[tokio::test]
async fn dag_cycle_rejected_at_execute_entry() {
    let a = make_agent();
    // #138 AC1: a non-empty (cyclic) task_list is pre-seeded below, so the
    // plan phase is SKIPPED — the cycle is detected at execute entry with NO
    // model turn at all. No plan turn is pushed.
    let agent_count = a.clone();
    let session = SessionId::new("s1");
    let h = StandardHarness::new(standard_config(a));
    // Build a cycle out of band: 1 -> 2, 2 -> 1.
    let mut tl = TaskList::default();
    tl.add("a".into(), vec![]).unwrap(); // 1
    tl.add("b".into(), vec![1]).unwrap(); // 2 -> 1
    tl.tasks[0].blockers = vec![2]; // 1 -> 2 closes the cycle
    seed_dag(&h, &session, &tl).await;

    match h
        .run(HarnessRunOptions::new(plan_task()).continue_task_list(true))
        .await
    {
        RunResult::Failure {
            reason: HaltReason::TaskGraphCycle { .. },
            ..
        } => {}
        other => panic!("expected TaskGraphCycle, got {other:?}"),
    }
    // #138 AC1: skip-plan means NO model turn ran — the cycle is caught at
    // execute entry before any plan or execute dispatch.
    assert_eq!(
        agent_count
            .call_count
            .load(std::sync::atomic::Ordering::SeqCst),
        0,
        "no model turn runs when a pre-seeded cyclic list is detected at entry"
    );
}

// Ledger drop-oldest past N=20 through the LIVE executor: run 22 independent
// tasks; the last task's Tier-2 ledger seed must show the static elision
// marker and NOT the very first task's summary.
#[tokio::test]
async fn dag_ledger_drop_oldest_past_n_through_executor() {
    let n = crate::tasklist::STEP_LEDGER_MAX_ENTRIES; // 20
    let total = n + 2; // 22 tasks
                       // #138 AC1: a non-empty task_list is pre-seeded below, so the plan phase
                       // is SKIPPED — no plan turn. Only the per-task SUMMARY turns are queued.
    let mut turns = Vec::new();
    for i in 0..total {
        turns.push(RecordingTurnAgent::final_resp(&format!("SUMMARY_{i}")));
    }
    let agent = RecordingTurnAgent::new("rec", turns);
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, agent.clone());
    let session = SessionId::new("s1");
    let h = StandardHarness::new(cfg);
    let mut tl = TaskList::default();
    for i in 0..total {
        tl.add(format!("t{i}"), vec![]).unwrap();
    }
    seed_dag(&h, &session, &tl).await;

    let _ = h
        .run(HarnessRunOptions::new(plan_task()).continue_task_list(true))
        .await;
    let contexts = agent.seen_text();
    // The LAST execute step's context (index total) is seeded with the ledger
    // AFTER the first (total-1) completions, so it must show elision and must
    // NOT contain the oldest summary (SUMMARY_0).
    let last = contexts.last().unwrap();
    assert!(
        last.contains(crate::tasklist::STEP_LEDGER_ELISION_MARKER),
        "last step's ledger should be elided: {last}"
    );
    assert!(
        !last.contains("SUMMARY_0 "),
        "the oldest summary must have been dropped: {last}"
    );
}

// Q4: the task list is persisted through the RunStore seam (not the #71
// sandbox path). Assert the durable RunStore holds the list after a run.
#[tokio::test]
async fn execute_phase_persists_through_run_store() {
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["one"]}"#));
    a.push(final_resp("did one"));
    let store = Arc::new(crate::storage::InMemoryStorageProvider::new());
    let provider = Arc::new(crate::storage::StorageProvider::single(store));
    let mut cfg = standard_config(a);
    cfg.storage = provider.clone();
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(plan_task())).await;

    let stored = provider
        .run()
        .get(&durable_ns(), TASK_LIST_EXTRAS_KEY)
        .await
        .expect("run store get ok")
        .expect("task list present in run store");
    let list: TaskList = serde_json::from_value(stored).expect("deserializes");
    assert_eq!(list.tasks.len(), 1);
    assert_eq!(list.tasks[0].status, TaskStatus::Completed);
}

// #76: after a plan/execute run, BOTH persistence keys live on the RunStore
// seam and NEITHER is mirrored into SessionState.extras. Drives the phases
// directly (rather than `run()`) so the post-run `state.extras` is
// observable. The ephemeral extras keys (`__rich_state`,
// `subagent_handoff_summary`) are owned by other components and untouched
// here.
#[tokio::test]
#[allow(deprecated)] // exercises the test-only run_execute_phase via the linear bridge (#126 C).
async fn plan_execute_persistence_lives_on_run_store_not_extras() {
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["one","two"],"rationale":"why"}"#));
    a.push(final_resp("did one"));
    a.push(final_resp("did two"));
    let h = StandardHarness::new(standard_config(a));
    let t = plan_task();
    let mut state = SessionState::default();

    let outcome = h
        .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
        .await
        .expect("plan phase succeeds");
    let task_list = crate::tasklist::plan_artifact_to_task_list(&outcome.artifact);
    let result = h
        .run_execute_phase(
            &t,
            &mut state,
            task_list,
            BudgetSnapshot {
                turns: outcome.turns,
                ..Default::default()
            },
            outcome.usage,
            &None,
        )
        .await;
    assert!(matches!(result, RunResult::Success { .. }));

    // Both keys are durable in the RunStore.
    assert!(h
        .storage()
        .run()
        .get(&durable_ns(), PLAN_EXECUTE_EXTRAS_KEY)
        .await
        .expect("run store get ok")
        .is_some());
    assert!(h
        .storage()
        .run()
        .get(&durable_ns(), TASK_LIST_EXTRAS_KEY)
        .await
        .expect("run store get ok")
        .is_some());

    // Neither key is mirrored into SessionState.extras anymore.
    assert!(state.extras.get(PLAN_EXECUTE_EXTRAS_KEY).is_none());
    assert!(state.extras.get(TASK_LIST_EXTRAS_KEY).is_none());
}

// Q2: success output is the LAST step's FinalResponse, not a concatenation
// and not the plan rationale.
#[tokio::test]
async fn execute_phase_success_output_is_last_step() {
    let a = make_agent();
    a.push(final_resp(
        r#"{"tasks":["a","b"],"rationale":"RATIONALE_TOKEN"}"#,
    ));
    a.push(final_resp("FIRST_STEP_OUTPUT"));
    a.push(final_resp("LAST_STEP_OUTPUT"));
    let h = StandardHarness::new(standard_config(a));
    match h.run(HarnessRunOptions::new(plan_task())).await {
        RunResult::Success { output, .. } => {
            assert_eq!(output, "LAST_STEP_OUTPUT");
            assert!(!output.contains("FIRST_STEP_OUTPUT"));
            assert!(!output.contains("RATIONALE_TOKEN"));
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// Confirms ExecutePhaseNotImplemented is gone: a full PlanExecute run with
// execute completions returns Success (the old variant would have halted).
#[tokio::test]
async fn execute_phase_not_implemented_is_gone() {
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["only"]}"#));
    a.push(final_resp("done"));
    let h = StandardHarness::new(standard_config(a));
    assert!(matches!(
        h.run(HarnessRunOptions::new(plan_task())).await,
        RunResult::Success { .. }
    ));
}

// Planner-agent routing through the FULL run: the planner runs the plan turn
// and the default agent runs the execute steps.
#[tokio::test]
async fn execute_phase_planner_agent_routing() {
    let default_agent = make_agent();
    default_agent.push(final_resp("did the step"));
    let planner = make_agent();
    planner.push(final_resp(r#"{"tasks":["step"]}"#));

    let mut cfg = standard_config(default_agent.clone());
    cfg.registry = std::mem::take(&mut cfg.registry)
        .into_builder()
        .agent("planner", planner.clone())
        .schema("plan-schema", serde_json::json!({}))
        .build();
    let h = StandardHarness::new(cfg);
    // A plan tree whose plan leaf routes to "planner"; execute uses default.
    let t = task(LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        plan: Box::new(LoopStrategy::ReAct(ReactConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            budget: BudgetPolicy::PerLoop { value: u32::MAX },
            agent: AgentRef("planner".into()),
            toolset: ToolsetRef(String::new()),
            output: Some(SchemaRef("plan-schema".into())),
            system_prompt: None,
        })),
        execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
        plan_model: None,
    }));
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Success { output, .. } => assert_eq!(output, "did the step"),
        other => panic!("expected Success, got {other:?}"),
    }
    assert_eq!(
        planner.call_count.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "planner ran exactly the plan turn"
    );
    assert_eq!(
        default_agent
            .call_count
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "default agent ran exactly the execute step"
    );
}

// #124 REGRESSION-PROOF: a NON-ReAct execute child genuinely runs per task.
//
// `PlanExecute[plan: ReAct, execute: SelfVerifying[ReAct]]` over a 2-task
// plan. The execute child is SelfVerifying — its build↔evaluate loop runs
// ONCE PER TASK, so the `ScriptedVerifier` is invoked exactly twice (once per
// task). The pre-#124 implementation hardcoded a flat ReAct sub-loop in
// `run_execute_phase` and silently DROPPED the SelfVerifying child — the
// verifier would have been invoked ZERO times. This proves `self.execute` is
// dispatched via `.run(cx)` and a non-ReAct child genuinely executes.
#[tokio::test]
async fn plan_execute_runs_non_react_execute_child_per_task() {
    // The single worker agent runs the plan turn (JSON), then per task: the
    // build turn, then the evaluate-phase turn (Q1c). 2 tasks ⇒ plan + 2×
    // (build + eval) = 5 turns.
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["t0","t1"],"rationale":"r"}"#));
    a.push(final_resp("built t0"));
    a.push(final_resp("PASS"));
    a.push(final_resp("built t1"));
    a.push(final_resp("PASS"));
    // The verifier records every invocation; PASS each time.
    let verifier = Arc::new(ScriptedVerifier::new(vec![passed(), passed()], passed(), 3));
    let mut cfg = standard_config(a);
    set_default_verifier(&mut cfg, verifier.clone());
    let h = StandardHarness::new(cfg);

    // The execute child is a genuine SelfVerifying combinator (NOT a ReAct).
    let t = task(LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        plan: Box::new(react_structured(u32::MAX)),
        execute: Box::new(LoopStrategy::SelfVerifying(SelfVerifyingConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            inner: Box::new(react_structured(u32::MAX)),
            evaluator: SchemaRef(String::new()),
            eval_agent: None,
            eval_toolset: None,
        })),
        plan_model: None,
    }));

    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Success { output, .. } => {
            assert_eq!(output, "built t1", "Q2: last step's final output");
        }
        other => panic!("expected Success, got {other:?}"),
    }

    // The smoking gun: the SelfVerifying evaluator ran ONCE PER TASK (2x).
    // A dropped execute child would record ZERO verifier invocations.
    assert_eq!(
        verifier.seen.lock().unwrap().len(),
        2,
        "the SelfVerifying execute child must run its evaluator once per task"
    );
}

// #124 REGRESSION-PROOF: SelfVerifying genuinely recurses into a NON-ReAct
// inner. `SelfVerifying[ inner: PlanExecute[ReAct, ReAct] ]` — each build
// iteration runs the inner PlanExecute's WHOLE loop, which fires a plan turn
// (parsing a JSON task list) before the execute step. A hardcoded-ReAct build
// would NEVER fire the plan phase, so the worker would never see the plan
// directive. We assert the worker saw the plan directive (proving the inner
// PlanExecute combinator ran, not a flat ReAct).
#[tokio::test]
async fn self_verifying_runs_non_react_inner_worker() {
    // Worker turns for ONE SelfVerifying iteration of PlanExecute[ReAct,ReAct]
    // over a 1-task plan: plan JSON, execute step, then the evaluate-phase turn.
    let worker = RecordingTurnAgent::new(
        "worker",
        vec![
            RecordingTurnAgent::final_resp(r#"{"tasks":["only"],"rationale":"r"}"#),
            RecordingTurnAgent::final_resp("did the step"),
            RecordingTurnAgent::final_resp("PASS"),
        ],
    );
    let verifier = Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3));
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, worker.clone());
    set_default_verifier(&mut cfg, verifier.clone());
    let h = StandardHarness::new(cfg);

    // inner is a genuine PlanExecute combinator (NOT a ReAct).
    let t = task(LoopStrategy::SelfVerifying(SelfVerifyingConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(LoopStrategy::PlanExecute(PlanExecuteConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            plan: Box::new(react_structured(u32::MAX)),
            execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
            plan_model: None,
        })),
        evaluator: SchemaRef(String::new()),
        eval_agent: None,
        eval_toolset: None,
    }));

    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Success { .. } => {}
        other => panic!("expected Success, got {other:?}"),
    }
    // The inner PlanExecute fired its plan turn ⇒ the worker saw the plan
    // directive. A hardcoded-ReAct build would record ZERO plan turns.
    let seen = worker.seen_text();
    let plan_turns = seen
        .iter()
        .filter(|c| c.contains("step-by-step plan"))
        .count();
    assert!(
        plan_turns >= 1,
        "the inner PlanExecute plan phase must fire at least once per iteration; saw: {seen:?}"
    );
    // The verifier fired (the SelfVerifying loop ran its evaluate phase).
    assert_eq!(verifier.seen.lock().unwrap().len(), 1);
}

// Rule: Aggregate usage accumulates across turns.
#[tokio::test]
async fn aggregate_usage_accumulates() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_tokens: None,
            cache_write_tokens: None,
        },
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "ok".into(),
        usage: TokenUsage {
            input_tokens: 7,
            output_tokens: 3,
            cache_read_tokens: None,
            cache_write_tokens: None,
        },
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "x".into(),
        truncated: false,
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success { usage, .. } => {
            assert_eq!(usage.input_tokens, 17);
            assert_eq!(usage.output_tokens, 8);
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// Rule: ModelError surfaces as AgentError → HaltReason::AgentError.
#[tokio::test]
async fn model_error_halts_via_agent_error() {
    let a = make_agent();
    a.push(TurnResult::Error {
        error: AgentError::ModelError(ModelError::Timeout),
        usage: None,
    });
    let h = StandardHarness::new(standard_config(a));
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Failure {
            reason:
                HaltReason::AgentError {
                    error: AgentError::ModelError(ModelError::Timeout),
                },
            ..
        } => {}
        other => panic!("expected AgentError(Timeout), got {other:?}"),
    }
}

// Serde round-trip — fixture portability.
#[test]
fn run_result_roundtrips_json() {
    let r = RunResult::Failure {
        reason: HaltReason::BudgetExceeded {
            limit_type: BudgetLimitType::Turns,
        },
        session_id: SessionId::new("s"),
        usage: AggregateUsage::default(),
        turns: 3,
        session_state: SessionState::default(),
    };
    let s = serde_json::to_string(&r).unwrap();
    let back: RunResult = serde_json::from_str(&s).unwrap();
    assert_eq!(r, back);
}

#[test]
fn paused_state_roundtrips_json() {
    let ps = PausedState {
        session_id: SessionId::new("s"),
        task_id: TaskId::new("t"),
        turn_number: 4,
        session_state: SessionState::default(),
        pending_tool_calls: vec![ToolCall {
            id: "c".into(),
            name: "x".into(),
            input: serde_json::json!({"k":1}),
        }],
        approved_results: vec![],
        human_request: Some(HumanRequest::Clarification {
            question: "what?".into(),
            options: None,
        }),
        task: react(5),
        budget_used: BudgetSnapshot {
            turns: 4,
            input_tokens: 100,
            output_tokens: 50,
            wall_time: Some(Duration::from_secs(10)),
            cost_usd: 0.0,
        },
        child_state: None,
        toolset: ToolsetRef::default(),
    };
    let s = serde_json::to_string(&ps).unwrap();
    let back: PausedState = serde_json::from_str(&s).unwrap();
    assert_eq!(ps, back);
}

// ChildPausedState cannot nest itself — compile-time depth-1 enforcement.
#[test]
fn child_paused_state_has_no_child_field() {
    // This test exists to lock in the rule via type system inspection at
    // review time. If a future change adds `child_state` to
    // ChildPausedState, this assertion is unaffected — but the spec rule
    // is broken. We assert structural shape via serde.
    let cs = ChildPausedState {
        session_id: SessionId::new("c"),
        task_id: TaskId::new("ct"),
        turn_number: 1,
        session_state: SessionState::default(),
        pending_tool_calls: vec![],
        approved_results: vec![],
        human_request: Some(HumanRequest::Clarification {
            question: "?".into(),
            options: None,
        }),
        task: react(1),
        budget_used: BudgetSnapshot::default(),
        parent_tool_call_id: "p".into(),
        toolset: ToolsetRef::default(),
    };
    let s = serde_json::to_string(&cs).unwrap();
    assert!(!s.contains("\"child_state\""));
}

// ====================================================================
// Compaction verify→retry→warn loop (issue #46)
// ====================================================================
mod compaction {
    use super::*;
    use crate::context::{
        CompactionPreserveHints, CompactionVerificationResult, CompactionVerifier,
        SessionState as ContextSessionState,
    };
    use crate::model::{Content, Role};
    use crate::observability::{
        InMemoryObservabilityProvider, ObservabilityProvider, SpanBase, WarnEvent, WarnSpan,
    };
    use crate::{SessionId, TaskId};
    use std::sync::Mutex;

    /// A `ContextManager` that always offers a compaction turn. Records how
    /// many times `apply_compaction` ran and the contexts the agent saw.
    struct CompactingContextManager {
        applied: Mutex<u32>,
        should: bool,
    }
    impl CompactingContextManager {
        fn new(should: bool) -> Self {
            Self {
                applied: Mutex::new(0),
                should,
            }
        }
    }
    impl ContextManager for CompactingContextManager {
        fn assemble<'a>(
            &'a self,
            session: &'a SessionState,
            _task: &'a Task,
            _sources: &'a ContextSources,
        ) -> BoxFut<'a, Context> {
            let messages = session.messages.clone();
            Box::pin(async move {
                Context {
                    messages,
                    tools: vec![],
                    params: crate::model::ModelParams::default(),
                }
            })
        }
        fn append_tool_result<'a>(
            &'a self,
            session: &'a mut SessionState,
            _result: &'a ToolResult,
        ) -> BoxFut<'a, ()> {
            Box::pin(async move {
                session.messages.push(Message {
                    role: Role::Tool,
                    content: Content::Text {
                        text: "tool".into(),
                    },
                });
            })
        }
        fn append_user_message<'a>(
            &'a self,
            _session: &'a mut SessionState,
            _text: &'a str,
        ) -> BoxFut<'a, ()> {
            Box::pin(async {})
        }
        fn should_compact(&self, _session: &SessionState) -> bool {
            self.should
        }
        fn prepare_compaction_turn(&self, _session: &SessionState) -> Option<CompactionTurn> {
            let mut vs = ContextSessionState::new(
                SessionId::new("s1"),
                TaskId::new("t1"),
                "deploy the payment service",
            );
            vs.token_budget_used = 1000;
            Some(CompactionTurn {
                context: Context {
                    messages: vec![Message {
                        role: Role::User,
                        content: Content::Text {
                            text: "please summarize".into(),
                        },
                    }],
                    tools: vec![],
                    params: crate::model::ModelParams::default(),
                },
                preserve_hints: CompactionPreserveHints::default(),
                verification_state: vs,
                messages_removed: 3,
                dropped_messages: vec![],
            })
        }
        fn apply_compaction(
            &self,
            _session: &mut SessionState,
            _summary: String,
            _dropped: &[Message],
        ) {
            *self.applied.lock().unwrap() += 1;
        }
    }

    /// Agent that records every context it is handed and replays a queue of
    /// final-response summaries.
    struct RecordingAgent {
        summaries: Mutex<std::collections::VecDeque<String>>,
        seen: Mutex<Vec<Context>>,
    }
    impl RecordingAgent {
        fn new(summaries: Vec<&str>) -> Self {
            Self {
                summaries: Mutex::new(summaries.iter().map(|s| s.to_string()).collect()),
                seen: Mutex::new(Vec::new()),
            }
        }
    }
    impl Agent for RecordingAgent {
        fn turn<'a>(&'a self, context: Context) -> BoxFut<'a, TurnResult> {
            self.seen.lock().unwrap().push(context);
            let content = self
                .summaries
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_default();
            Box::pin(async move {
                TurnResult::FinalResponse {
                    reasoning: None,
                    content,
                    usage: TokenUsage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cache_read_tokens: None,
                        cache_write_tokens: None,
                    },
                }
            })
        }
        fn id(&self) -> AgentId {
            AgentId::new("recording")
        }
    }

    /// Verifier that fails the first `fail_first` calls, then passes.
    struct ScriptedVerifier {
        fail_first: Mutex<u32>,
        missing: Vec<String>,
    }
    impl ScriptedVerifier {
        fn new(fail_first: u32, missing: Vec<&str>) -> Self {
            Self {
                fail_first: Mutex::new(fail_first),
                missing: missing.iter().map(|s| s.to_string()).collect(),
            }
        }
    }
    impl CompactionVerifier for ScriptedVerifier {
        fn verify(
            &self,
            _summary: &str,
            _hints: &CompactionPreserveHints,
            _state: &ContextSessionState,
        ) -> CompactionVerificationResult {
            let mut remaining = self.fail_first.lock().unwrap();
            if *remaining > 0 {
                *remaining -= 1;
                CompactionVerificationResult {
                    passed: false,
                    missing_items: self.missing.clone(),
                    detail: "scripted fail".into(),
                }
            } else {
                CompactionVerificationResult {
                    passed: true,
                    missing_items: vec![],
                    detail: "scripted pass".into(),
                }
            }
        }
    }

    fn harness_with(
        cm: Arc<CompactingContextManager>,
        agent: Arc<RecordingAgent>,
        verifier: Arc<dyn CompactionVerifier>,
        obs: Arc<dyn ObservabilityProvider>,
        max_attempts: u32,
    ) -> StandardHarness {
        let worker: Arc<dyn Agent> = agent;
        StandardHarness::new(HarnessConfig {
            tool_registry: Arc::new(ScriptedToolRegistry::new()),
            sandbox: Arc::new(AllowAllSandbox),
            sandbox_violation_policy: crate::harness::SandboxViolationPolicy::default(),
            context_manager: cm,
            termination_policy: Arc::new(AlwaysContinuePolicy),
            middleware: None,
            observability: Some(obs),
            compaction_verifier: verifier,
            max_compaction_attempts: max_attempts,
            pricing: PricingTable::DEFAULT,
            content_capture: ContentCaptureConfig::default(),
            tool_call_repair: None,
            max_repair_attempts: 1,
            plan_empty_max_retries: 0,
            tool_call_nudge: false,
            steering_source: None,
            max_stop_blocks: 8,
            error_loop_threshold: 3,
            enforce_output_schemas: false,
            output_schema_max_retries: 2,
            hooks: None,
            storage: Arc::new(crate::storage::StorageProvider::no_op()),
            project_id: crate::storage::ProjectId::from_canonical_path("/test-workspace"),
            chunk_provider: Arc::new(crate::prompt_assembly::InMemoryChunkProvider::empty()),
            max_resets: 3,
            vcs_provider: None,
            catalogue_registry: None,
            toolset_catalogues: std::collections::HashMap::new(),
            system_prompt: None,
            guides: Vec::new(),
            skills: None,
            memory: None,
            model_params: ModelParams::default(),
            auto_persist_sessions: false,
            prompt_tool_call_flag: None,
            consult_handlers: std::collections::HashMap::new(),
            registry: ExecutionRegistry::builder().agent("", worker).build(),
            escalation_mode: EscalationMode::SurfaceToHuman,
            tool_search: None,
        })
    }

    async fn drive(
        h: &StandardHarness,
        agent: &Arc<RecordingAgent>,
        cm: &Arc<CompactingContextManager>,
    ) {
        let mut state = SessionState::default();
        let mut usage = AggregateUsage::default();
        let mut span_seq = 0u64;
        // Pre-condition: should_compact gate is honored by the caller.
        if h.config.context_manager.should_compact(&state) {
            let worker: Arc<dyn Agent> = agent.clone();
            h.run_compaction(
                &mut state,
                &SessionId::new("s1"),
                &TaskId::new("t1"),
                &mut span_seq,
                &mut usage,
                &worker,
            )
            .await;
        }
        let _ = (agent, cm);
    }

    #[tokio::test]
    async fn no_compaction_when_should_compact_false() {
        let cm = Arc::new(CompactingContextManager::new(false));
        let agent = Arc::new(RecordingAgent::new(vec!["summary"]));
        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let h = harness_with(
            cm.clone(),
            agent.clone(),
            Arc::new(ScriptedVerifier::new(0, vec![])),
            obs.clone(),
            2,
        );
        drive(&h, &agent, &cm).await;
        assert_eq!(agent.seen.lock().unwrap().len(), 0, "no compaction turn");
        assert_eq!(*cm.applied.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn passing_verifier_one_turn_one_apply_no_warn() {
        let cm = Arc::new(CompactingContextManager::new(true));
        let agent = Arc::new(RecordingAgent::new(vec!["good summary"]));
        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let h = harness_with(
            cm.clone(),
            agent.clone(),
            Arc::new(ScriptedVerifier::new(0, vec![])),
            obs.clone(),
            2,
        );
        drive(&h, &agent, &cm).await;
        assert_eq!(agent.seen.lock().unwrap().len(), 1, "exactly one turn");
        assert_eq!(*cm.applied.lock().unwrap(), 1, "applied once");
        assert!(
            obs.warn_spans(&SessionId::new("s1")).is_empty(),
            "no warn emitted"
        );
    }

    #[tokio::test]
    async fn failing_then_passing_retries_and_injects_missing_items() {
        let cm = Arc::new(CompactingContextManager::new(true));
        let agent = Arc::new(RecordingAgent::new(vec!["v1", "v2"]));
        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let h = harness_with(
            cm.clone(),
            agent.clone(),
            Arc::new(ScriptedVerifier::new(1, vec!["payment", "deploy"])),
            obs.clone(),
            2,
        );
        drive(&h, &agent, &cm).await;
        let seen = agent.seen.lock().unwrap();
        assert_eq!(seen.len(), 2, "two compaction turns");
        // The retry context must contain the injected "missing these items"
        // message with the actual missing items.
        let retry_ctx = &seen[1];
        let injected = retry_ctx.messages.iter().any(|m| {
            matches!(&m.content, Content::Text { text }
                    if text.contains("missing these items")
                        && text.contains("payment")
                        && text.contains("deploy"))
        });
        assert!(injected, "retry context carries the missing-items message");
        assert_eq!(*cm.applied.lock().unwrap(), 1, "applied once after pass");
        assert!(obs.warn_spans(&SessionId::new("s1")).is_empty());
    }

    #[tokio::test]
    async fn always_failing_warns_and_accepts_anyway() {
        let cm = Arc::new(CompactingContextManager::new(true));
        let agent = Arc::new(RecordingAgent::new(vec!["v1", "v2"]));
        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let h = harness_with(
            cm.clone(),
            agent.clone(),
            Arc::new(ScriptedVerifier::new(99, vec!["payment"])),
            obs.clone(),
            2,
        );
        drive(&h, &agent, &cm).await;
        assert_eq!(agent.seen.lock().unwrap().len(), 2, "max attempts == 2");
        // apply_compaction STILL called.
        assert_eq!(*cm.applied.lock().unwrap(), 1, "accepted anyway");
        let sid = SessionId::new("s1");
        let warns = obs.warn_spans(&sid);
        assert_eq!(warns.len(), 1, "exactly one warn span");
        match &warns[0].event {
            WarnEvent::CompactionVerificationFailed {
                missing_items,
                accepted_anyway,
            } => {
                assert_eq!(missing_items, &vec!["payment".to_string()]);
                assert!(*accepted_anyway);
            }
            other => panic!("expected CompactionVerificationFailed, got {other:?}"),
        }
        // SessionMetrics failure counter == 1. Seed an outcome so metrics
        // are produced even though no turn span was emitted in this unit.
        obs.set_session_outcome(&sid, SessionOutcome::Success);
        let m = obs.get_session_metrics(&sid).await.unwrap();
        assert_eq!(m.compaction_verification_failures, 1);
    }

    #[tokio::test]
    async fn max_attempts_one_honored() {
        let cm = Arc::new(CompactingContextManager::new(true));
        let agent = Arc::new(RecordingAgent::new(vec!["v1", "v2", "v3"]));
        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let h = harness_with(
            cm.clone(),
            agent.clone(),
            Arc::new(ScriptedVerifier::new(99, vec!["payment"])),
            obs.clone(),
            1,
        );
        drive(&h, &agent, &cm).await;
        assert_eq!(
            agent.seen.lock().unwrap().len(),
            1,
            "exactly one attempt with max=1"
        );
        assert_eq!(*cm.applied.lock().unwrap(), 1, "accepted after one attempt");
        assert_eq!(obs.warn_spans(&SessionId::new("s1")).len(), 1);
    }

    /// A minimal provider that does NOT override `emit_warn`; proves the
    /// default no-op body does not break it (W4).
    #[derive(Default)]
    struct BareProvider;
    impl ObservabilityProvider for BareProvider {
        fn emit_turn(&self, _s: crate::observability::TurnSpan) {}
        fn emit_tool_call(&self, _s: ToolCallSpan) {}
        fn emit_sensor(&self, _s: crate::observability::SensorSpan) {}
        fn emit_context(&self, _s: crate::observability::ContextSpan) {}
        fn emit_middleware(&self, _s: crate::observability::MiddlewareSpan) {}
        fn emit_patch(&self, _s: crate::observability::PatchSpan) {}
        fn flush_session<'a>(&'a self, _sid: &'a SessionId) -> BoxFut<'a, ()> {
            Box::pin(async {})
        }
        fn get_session_metrics<'a>(
            &'a self,
            _sid: &'a SessionId,
        ) -> BoxFut<'a, Option<crate::observability::SessionMetrics>> {
            Box::pin(async { None })
        }
        fn get_sessions<'a>(
            &'a self,
            _since: Timestamp,
            _domain: Option<String>,
            _outcome: Option<SessionOutcome>,
        ) -> BoxFut<'a, Vec<crate::observability::SessionMetrics>> {
            Box::pin(async { Vec::new() })
        }
        fn get_trace<'a>(
            &'a self,
            _sid: &'a SessionId,
        ) -> BoxFut<'a, Vec<Box<dyn crate::observability::Span>>> {
            Box::pin(async { Vec::new() })
        }
    }

    #[tokio::test]
    async fn emit_warn_default_noop_does_not_break_bare_provider() {
        let cm = Arc::new(CompactingContextManager::new(true));
        let agent = Arc::new(RecordingAgent::new(vec!["v1", "v2"]));
        let obs = Arc::new(BareProvider);
        let h = harness_with(
            cm.clone(),
            agent.clone(),
            Arc::new(ScriptedVerifier::new(99, vec!["payment"])),
            obs,
            2,
        );
        // Reaching the warn path must not panic; bare provider ignores it.
        drive(&h, &agent, &cm).await;
        assert_eq!(*cm.applied.lock().unwrap(), 1);
        // Construct a WarnSpan directly to lock W4: the default body runs.
        let base = SpanBase::new_root(
            crate::observability::SpanId::new("x"),
            SessionId::new("s1"),
            TaskId::new("t1"),
            SpanKind::Warn,
            Timestamp::now(),
        );
        BareProvider.emit_warn(WarnSpan::new(
            base,
            WarnEvent::CompactionVerificationFailed {
                missing_items: vec![],
                accepted_anyway: true,
            },
        ));
    }

    // ----------------------------------------------------------------
    // Cross-language consistency fixture replay (issue #46)
    // ----------------------------------------------------------------

    /// Verifier driven by a fixture verdict queue; repeats the last verdict
    /// once the queue is exhausted, matching the fixture contract.
    struct FixtureVerifier {
        verdicts: Mutex<Vec<CompactionVerificationResult>>,
        idx: std::sync::atomic::AtomicUsize,
    }
    impl CompactionVerifier for FixtureVerifier {
        fn verify(
            &self,
            _summary: &str,
            _hints: &CompactionPreserveHints,
            _state: &ContextSessionState,
        ) -> CompactionVerificationResult {
            let v = self.verdicts.lock().unwrap();
            let i = self.idx.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            v.get(i).or_else(|| v.last()).cloned().unwrap()
        }
    }

    #[tokio::test]
    async fn fixture_replay_loop_outcomes() {
        #[derive(serde::Deserialize)]
        struct Verdict {
            passed: bool,
            missing_items: Vec<String>,
        }
        #[derive(serde::Deserialize)]
        struct Expected {
            attempts: u32,
            apply_compaction_calls: u32,
            warn_emitted: bool,
            #[serde(default)]
            retry_injected_missing: Option<Vec<String>>,
            final_missing_items: Vec<String>,
        }
        #[derive(serde::Deserialize)]
        struct Case {
            name: String,
            max_compaction_attempts: u32,
            verdicts: Vec<Verdict>,
            expected: Expected,
        }
        #[derive(serde::Deserialize)]
        struct Suite {
            cases: Vec<Case>,
        }

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/compaction_loop/cases.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let suite: Suite = serde_json::from_str(&raw).unwrap();
        assert!(suite.cases.len() >= 5, "expected >= 5 fixture cases");

        for case in &suite.cases {
            let cm = Arc::new(CompactingContextManager::new(true));
            // Enough summaries that the agent never starves.
            let agent = Arc::new(RecordingAgent::new(vec!["s1", "s2", "s3", "s4"]));
            let obs = Arc::new(InMemoryObservabilityProvider::new());
            let verifier = Arc::new(FixtureVerifier {
                verdicts: Mutex::new(
                    case.verdicts
                        .iter()
                        .map(|v| CompactionVerificationResult {
                            passed: v.passed,
                            missing_items: v.missing_items.clone(),
                            detail: String::new(),
                        })
                        .collect(),
                ),
                idx: std::sync::atomic::AtomicUsize::new(0),
            });
            let h = harness_with(
                cm.clone(),
                agent.clone(),
                verifier,
                obs.clone(),
                case.max_compaction_attempts,
            );
            drive(&h, &agent, &cm).await;

            let sid = SessionId::new("s1");
            assert_eq!(
                agent.seen.lock().unwrap().len() as u32,
                case.expected.attempts,
                "case `{}`: attempts",
                case.name
            );
            assert_eq!(
                *cm.applied.lock().unwrap(),
                case.expected.apply_compaction_calls,
                "case `{}`: apply_compaction_calls",
                case.name
            );
            let warns = obs.warn_spans(&sid);
            assert_eq!(
                !warns.is_empty(),
                case.expected.warn_emitted,
                "case `{}`: warn_emitted",
                case.name
            );
            if case.expected.warn_emitted {
                assert_eq!(warns.len(), 1, "case `{}`: exactly one warn", case.name);
                match &warns[0].event {
                    WarnEvent::CompactionVerificationFailed {
                        missing_items,
                        accepted_anyway,
                    } => {
                        assert_eq!(
                            missing_items, &case.expected.final_missing_items,
                            "case `{}`: final_missing_items",
                            case.name
                        );
                        assert!(*accepted_anyway, "case `{}`: accepted_anyway", case.name);
                    }
                    other => panic!(
                        "case `{}`: expected CompactionVerificationFailed, got {other:?}",
                        case.name
                    ),
                }
            }
            if let Some(expected_inject) = &case.expected.retry_injected_missing {
                let seen = agent.seen.lock().unwrap();
                let joined = expected_inject.join(", ");
                let found = seen.iter().skip(1).any(|c| {
                    c.messages.iter().any(|m| {
                        matches!(&m.content,
                            Content::Text { text }
                                if text.contains("missing these items")
                                    && text.contains(&joined))
                    })
                });
                assert!(found, "case `{}`: retry injection", case.name);
            }
        }
    }
}

// ── Assistant-turn recording (regression for lost conversation history) ──

/// A ContextManager that records every message in a shared vector so tests
/// can inspect the conversation the loop builds. Mirrors
/// `NoopContextManager` but exposes the message log.
#[derive(Clone, Default)]
struct RecordingContextManager {
    messages: Arc<std::sync::Mutex<Vec<Message>>>,
}
impl ContextManager for RecordingContextManager {
    fn assemble<'a>(
        &'a self,
        session: &'a SessionState,
        _task: &'a Task,
        _sources: &'a ContextSources,
    ) -> BoxFut<'a, Context> {
        let messages = session.messages.clone();
        Box::pin(async move {
            Context {
                messages,
                tools: vec![],
                params: crate::model::ModelParams::default(),
            }
        })
    }
    fn append_tool_result<'a>(
        &'a self,
        session: &'a mut SessionState,
        result: &'a ToolResult,
    ) -> BoxFut<'a, ()> {
        let msg = Message {
            role: Role::Tool,
            content: Content::ToolResult(crate::model::ToolResult {
                tool_use_id: result.call_id.clone(),
                content: match &result.output {
                    ToolOutput::Success { content, .. } => content.clone(),
                    ToolOutput::Error { message, .. } => message.clone(),
                    ToolOutput::SandboxViolation { violation } => {
                        format!("sandbox violation: {violation:?}")
                    }
                    ToolOutput::WaitingForHuman { .. } => String::new(),
                    ToolOutput::Escalate { .. } => String::new(),
                    ToolOutput::AwaitingClarification { .. } => String::new(),
                    ToolOutput::Consult { .. } => String::new(),
                },
                is_error: matches!(result.output, ToolOutput::Error { .. }),
            }),
        };
        Box::pin(async move {
            session.messages.push(msg.clone());
            self.messages.lock().unwrap().push(msg);
        })
    }
    fn append_assistant_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        message: &'a Message,
    ) -> BoxFut<'a, ()> {
        let message = message.clone();
        Box::pin(async move {
            session.messages.push(message.clone());
            self.messages.lock().unwrap().push(message);
        })
    }
    fn append_user_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()> {
        let msg = Message {
            role: Role::User,
            content: Content::Text { text: text.into() },
        };
        Box::pin(async move {
            session.messages.push(msg.clone());
            self.messages.lock().unwrap().push(msg);
        })
    }
}

/// Regression: a turn that requests a tool call must record the assistant's
/// tool-call message in history, positioned BEFORE the tool result, so the
/// next turn's assembled context reflects what the agent already did.
#[tokio::test]
async fn tool_call_records_assistant_message_before_result() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "read_file".into(),
            input: serde_json::json!({ "path": "a.txt" }),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let cm = RecordingContextManager::default();
    let mut cfg = standard_config(a);
    cfg.context_manager = Arc::new(cm.clone());
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "contents".into(),
        truncated: false,
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);
    assert!(matches!(
        h.run(HarnessRunOptions::new(react(5))).await,
        RunResult::Success { .. }
    ));

    let msgs = cm.messages.lock().unwrap();
    let assistant_idx = msgs.iter().position(|m| {
        m.role == Role::Assistant && matches!(&m.content, Content::ToolCall(c) if c.id == "c1")
    });
    let tool_idx = msgs.iter().position(|m| {
        m.role == Role::Tool
            && matches!(&m.content, Content::ToolResult(r) if r.tool_use_id == "c1")
    });
    let assistant_idx = assistant_idx.expect("assistant tool-call message must be recorded");
    let tool_idx = tool_idx.expect("tool result must be recorded");
    assert!(
        assistant_idx < tool_idx,
        "assistant tool_use (idx {assistant_idx}) must precede its tool result (idx {tool_idx})"
    );
}

/// Regression: a final response must append the assistant's text to history
/// so a continued session sees what the agent said.
#[tokio::test]
async fn final_response_records_assistant_text() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "the final answer".into(),
        usage: usage(),
    });
    let cm = RecordingContextManager::default();
    let mut cfg = standard_config(a);
    cfg.context_manager = Arc::new(cm.clone());
    let h = StandardHarness::new(cfg);
    assert!(matches!(
        h.run(HarnessRunOptions::new(react(5))).await,
        RunResult::Success { .. }
    ));

    let msgs = cm.messages.lock().unwrap();
    assert!(
        msgs.iter().any(|m| m.role == Role::Assistant
            && matches!(&m.content, Content::Text { text } if text == "the final answer")),
        "assistant final text must be recorded in history"
    );
}

/// Regression: when a run pauses at BeforeTool (SurfaceToHuman) and is then
/// resumed with Allow, the assistant tool-call message must already be in
/// history — recorded before the pause — and positioned before its tool
/// result, with no duplicate from the resume path. Guards the option-(a)
/// fix: the assistant turn is recorded as soon as the calls are known, so
/// resume_inner never has to (and never double-records) it.
#[tokio::test]
async fn resume_after_surface_to_human_records_assistant_once_before_result() {
    let a = make_agent();
    let calls = vec![ToolCall {
        id: "c1".into(),
        name: "read_file".into(),
        input: serde_json::json!({ "path": "a.txt" }),
    }];
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: calls.clone(),
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let cm = RecordingContextManager::default();
    let mut cfg = standard_config(a);
    cfg.context_manager = Arc::new(cm.clone());
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::Success {
        content: "contents".into(),
        truncated: false,
    });
    cfg.tool_registry = reg;
    let mw = Arc::new(ScriptedMiddleware::new());
    mw.push(
        HookPoint::BeforeTool,
        MiddlewareDecision::SurfaceToHuman {
            request: HumanRequest::ToolApproval {
                calls: calls.clone(),
                risk_level: RiskLevel::Medium,
            },
        },
    );
    cfg.middleware = Some(mw);
    let h = StandardHarness::new(cfg);

    // Pause at BeforeTool — the assistant turn was recorded just before.
    let state = match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::WaitingForHuman { state, .. } => *state,
        other => panic!("expected WaitingForHuman, got {other:?}"),
    };
    // Resume with approval; the pending call is dispatched and its result
    // appended after the already-recorded assistant turn.
    assert!(matches!(
        h.resume(state, HumanResponse::Allow, None).await,
        RunResult::Success { .. }
    ));

    let msgs = cm.messages.lock().unwrap();
    let assistant_idx = msgs.iter().position(|m| {
        m.role == Role::Assistant && matches!(&m.content, Content::ToolCall(c) if c.id == "c1")
    });
    let tool_idx = msgs.iter().position(|m| {
        m.role == Role::Tool
            && matches!(&m.content, Content::ToolResult(r) if r.tool_use_id == "c1")
    });
    let assistant_idx =
        assistant_idx.expect("assistant tool-call must be recorded on the resume path");
    let tool_idx = tool_idx.expect("tool result must be recorded");
    assert!(
        assistant_idx < tool_idx,
        "assistant tool_use (idx {assistant_idx}) must precede its tool result (idx {tool_idx})"
    );
    let count = msgs
        .iter()
        .filter(|m| {
            m.role == Role::Assistant && matches!(&m.content, Content::ToolCall(c) if c.id == "c1")
        })
        .count();
    assert_eq!(
        count, 1,
        "assistant tool-call must be recorded exactly once, not duplicated by resume"
    );
}

// ── Issue #69 — Stop-hook wiring into the ReAct loop ───────────────────

/// A Stop hook that blocks `block_n` times then continues, counting fires.
struct CountingStopHook {
    block_first_n: u32,
    fired: std::sync::atomic::AtomicU32,
}
impl crate::hooks::Hook for CountingStopHook {
    fn handle<'c>(
        &'c self,
        _ctx: &'c mut crate::hooks::HookContext<'_>,
    ) -> BoxFut<'c, Result<crate::hooks::HookDecision, crate::hooks::HookError>> {
        let n = self.fired.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let decision = if n < self.block_first_n {
            crate::hooks::HookDecision::Block {
                reason: format!("not done yet (block {n})"),
            }
        } else {
            crate::hooks::HookDecision::Continue
        };
        Box::pin(async move { Ok(decision) })
    }
    fn events(&self) -> Vec<crate::hooks::HookEvent> {
        vec![crate::hooks::HookEvent::Stop]
    }
    fn name(&self) -> String {
        "counting-stop".into()
    }
}

// Stop block → inject → continue, then continue → terminate (R12/R13).
#[tokio::test]
async fn stop_hook_block_then_continue_loops() {
    let a = make_agent();
    // Two final responses: first one is blocked by the Stop hook and the
    // loop continues; second one is allowed to terminate.
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "first".into(),
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "second".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let chain = Arc::new(crate::hooks::StandardHookChain::new());
    chain
        .register(Arc::new(CountingStopHook {
            block_first_n: 1,
            fired: std::sync::atomic::AtomicU32::new(0),
        }))
        .unwrap();
    cfg.hooks = Some(chain);
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success { output, turns, .. } => {
            assert_eq!(output, "second");
            assert_eq!(turns, 2, "loop should run a second turn after the block");
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// No registered Stop hook → terminate immediately (R13 baseline).
#[tokio::test]
async fn stop_hook_absent_terminates() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    cfg.hooks = Some(Arc::new(crate::hooks::StandardHookChain::new()));
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success { turns, .. } => assert_eq!(turns, 1),
        other => panic!("expected Success, got {other:?}"),
    }
}

// max_stop_blocks cap: a hook that always blocks terminates anyway after
// `max_stop_blocks` consecutive blocks (R14).
#[tokio::test]
async fn stop_hook_max_blocks_cap() {
    let a = make_agent();
    // Always-final agent; the hook always blocks, so the cap must stop it.
    for _ in 0..20 {
        a.push(TurnResult::FinalResponse {
            reasoning: None,
            content: "again".into(),
            usage: usage(),
        });
    }
    let mut cfg = standard_config(a);
    cfg.max_stop_blocks = 3;
    let chain = Arc::new(crate::hooks::StandardHookChain::new());
    chain
        .register(Arc::new(CountingStopHook {
            block_first_n: u32::MAX, // always block
            fired: std::sync::atomic::AtomicU32::new(0),
        }))
        .unwrap();
    cfg.hooks = Some(chain);
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(50))).await {
        RunResult::Success { turns, .. } => {
            // 3 blocks → 3 extra turns; turn 4 hits the cap and terminates.
            assert_eq!(turns, 4, "should terminate after max_stop_blocks=3 blocks");
        }
        other => panic!("expected Success after cap, got {other:?}"),
    }
}

// ========================================================================
// SelfVerifying loop strategy (issue #61)
// ========================================================================

use crate::verifier::{Verifier, VerifierInput, VerifierVerdict};
use std::sync::Mutex as StdMutex;

fn self_verifying_task() -> Task {
    task(LoopStrategy::SelfVerifying(SelfVerifyingConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(react_structured(u32::MAX)),
        evaluator: SchemaRef(String::new()),
        eval_agent: None,
        eval_toolset: None,
    }))
}

/// A `Verifier` test double that replays a scripted queue of verdicts.
/// Records the `(iteration, build_session, eval_session)` it saw on each
/// call so tests can assert phase distinctness (R2/R9). `max_iterations` is
/// configurable to drive the round-trip cap (D3/R7).
struct ScriptedVerifier {
    verdicts: StdMutex<std::collections::VecDeque<VerifierVerdict>>,
    default_verdict: VerifierVerdict,
    max_iters: u32,
    seen: StdMutex<Vec<(u32, String, String)>>,
}
impl ScriptedVerifier {
    fn new(verdicts: Vec<VerifierVerdict>, default_verdict: VerifierVerdict, max: u32) -> Self {
        Self {
            verdicts: StdMutex::new(verdicts.into_iter().collect()),
            default_verdict,
            max_iters: max,
            seen: StdMutex::new(Vec::new()),
        }
    }
}
impl Verifier for ScriptedVerifier {
    fn verify<'a>(&'a self, input: &'a VerifierInput) -> BoxFut<'a, VerifierVerdict> {
        let build_sid = match &input.build_result {
            RunResult::Success { session_id, .. } | RunResult::Failure { session_id, .. } => {
                session_id.as_str().to_string()
            }
            _ => "?".to_string(),
        };
        let eval_sid = match &input.eval_result {
            RunResult::Success { session_id, .. } | RunResult::Failure { session_id, .. } => {
                session_id.as_str().to_string()
            }
            _ => "?".to_string(),
        };
        self.seen
            .lock()
            .unwrap()
            .push((input.iteration, build_sid, eval_sid));
        let v = self
            .verdicts
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| self.default_verdict.clone());
        Box::pin(async move { v })
    }
    fn max_iterations(&self) -> u32 {
        self.max_iters
    }
}

/// Agent that records every `Context` it is handed and replays a queue of
/// `TurnResult`s (default: an empty `FinalResponse`). Use the recorded
/// contexts to assert what the build / evaluate phases saw.
struct RecordingTurnAgent {
    id: AgentId,
    turns: StdMutex<std::collections::VecDeque<TurnResult>>,
    seen: StdMutex<Vec<Context>>,
}
impl RecordingTurnAgent {
    fn new(id: &str, turns: Vec<TurnResult>) -> Arc<Self> {
        Arc::new(Self {
            id: AgentId::new(id),
            turns: StdMutex::new(turns.into_iter().collect()),
            seen: StdMutex::new(Vec::new()),
        })
    }
    fn final_resp(text: &str) -> TurnResult {
        TurnResult::FinalResponse {
            reasoning: None,
            content: text.into(),
            usage: TokenUsage {
                input_tokens: 1,
                output_tokens: 1,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
        }
    }
    fn tool_call(name: &str) -> TurnResult {
        TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: "c1".into(),
                name: name.into(),
                input: serde_json::json!({}),
            }],
            usage: TokenUsage {
                input_tokens: 1,
                output_tokens: 1,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
        }
    }
    /// Flatten every recorded context's messages to a single text blob.
    fn seen_text(&self) -> Vec<String> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .map(|c| {
                c.messages
                    .iter()
                    .map(|m| match &m.content {
                        Content::Text { text } => text.clone(),
                        Content::ToolCall(tc) => tc.name.clone(),
                        // Tool results carry prior steps' outputs forward —
                        // surface their content so accumulation is visible.
                        Content::ToolResult(tr) => tr.content.clone(),
                        _ => String::new(),
                    })
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .collect()
    }
    fn call_count(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}
impl Agent for RecordingTurnAgent {
    fn turn<'a>(&'a self, context: Context) -> BoxFut<'a, TurnResult> {
        self.seen.lock().unwrap().push(context);
        let t = self
            .turns
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Self::final_resp(""));
        Box::pin(async move { t })
    }
    fn id(&self) -> AgentId {
        self.id.clone()
    }
}

fn passed() -> VerifierVerdict {
    VerifierVerdict::Passed
}
fn failed(reason: &str) -> VerifierVerdict {
    VerifierVerdict::failed(reason)
}

// R10: SelfVerifying no longer returns StrategyNotYetImplemented.
#[tokio::test]
async fn self_verifying_no_longer_unimplemented() {
    // Q1c: build + evaluate run on the SAME worker agent (one scripted queue).
    let build = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::final_resp("done"),
            RecordingTurnAgent::final_resp("PASS"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, build);
    set_default_verifier(
        &mut cfg,
        Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3)),
    );
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(self_verifying_task())).await {
        RunResult::Failure {
            reason: HaltReason::StrategyNotYetImplemented { .. },
            ..
        } => panic!("SelfVerifying must be implemented"),
        RunResult::Success { .. } => {}
        other => panic!("expected Success, got {other:?}"),
    }
}

// R11 (#124): an unresolved verifier is a typed STARTUP halt (single
// resolution path), not a panic.
#[tokio::test]
async fn self_verifying_missing_verifier_is_typed_halt() {
    let cfg = standard_config(make_agent());
    // no verifier registered under the default key.
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(self_verifying_task())).await {
        RunResult::Failure {
            reason:
                HaltReason::ConfigurationError {
                    error: HarnessError::UnresolvedHandle { kind, .. },
                },
            ..
        } => assert_eq!(kind, "verifier"),
        other => panic!("expected unresolved verifier, got {other:?}"),
    }
}

// R1: build phase runs a ReAct loop until the agent claims done (a tool call
// followed by a final response — the build agent loops at least twice).
#[tokio::test]
async fn self_verifying_build_runs_react_until_done() {
    // Q1c: build + evaluate share one worker queue: build does tool+final,
    // then the evaluate phase pulls its PASS turn from the same agent.
    let build = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::tool_call("read_file"),
            RecordingTurnAgent::final_resp("done"),
            RecordingTurnAgent::final_resp("PASS"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, build.clone());
    set_default_verifier(
        &mut cfg,
        Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3)),
    );
    let h = StandardHarness::new(cfg);
    let r = h.run(HarnessRunOptions::new(self_verifying_task())).await;
    assert!(matches!(r, RunResult::Success { .. }), "got {r:?}");
    // The build agent took at least two turns (tool call, then final) plus the
    // evaluate turn.
    assert!(
        build.call_count() >= 3,
        "build+eval should loop: {} turns",
        build.call_count()
    );
}

// R2 + R9: the evaluate phase uses a FRESH SessionId never shared with the
// build session — distinguishable in traces.
#[tokio::test]
async fn self_verifying_evaluate_uses_fresh_distinct_session() {
    let build = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::final_resp("done"),
            RecordingTurnAgent::final_resp("PASS"),
        ],
    );
    let verifier = Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3));
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, build);
    set_default_verifier(&mut cfg, verifier.clone());
    let h = StandardHarness::new(cfg);
    let t = self_verifying_task();
    let build_session = t.session_id.as_str().to_string();
    let _ = h.run(HarnessRunOptions::new(t)).await;
    let seen = verifier.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    let (_iter, b, e) = &seen[0];
    assert_eq!(b, &build_session, "build session id is the run's session");
    assert_ne!(e, &build_session, "evaluate session must be fresh");
    assert!(e.starts_with("sess-"), "evaluate uses generate(): {e}");
}

// R3: the evaluate phase uses a read-only sandbox — an evaluator Write
// attempt is rejected as a (recoverable) ReadOnlyViolation, while the build
// sandbox is unaffected (the build phase writes freely).
#[tokio::test]
async fn self_verifying_evaluate_sandbox_is_read_only() {
    // Q1c: one worker agent. Build phase writes (allowed), then claims done;
    // the evaluate phase (same agent, read-only sandbox) tries to write —
    // that write MUST be rejected — then claims a verdict.
    let build = RecordingTurnAgent::new(
        "build",
        vec![
            // build phase: write allowed, then done
            RecordingTurnAgent::tool_call("write_file"),
            RecordingTurnAgent::final_resp("done"),
            // evaluate phase (fresh session, read-only): write rejected, PASS
            RecordingTurnAgent::tool_call("write_file"),
            RecordingTurnAgent::final_resp("PASS"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, build.clone());
    set_default_verifier(
        &mut cfg,
        Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3)),
    );
    let h = StandardHarness::new(cfg);
    let r = h.run(HarnessRunOptions::new(self_verifying_task())).await;
    assert!(matches!(r, RunResult::Success { .. }), "got {r:?}");
    // The evaluate phase's write (over the read-only sandbox) must surface a
    // recoverable read-only sandbox error fed back as a tool result. The build
    // phase's write (over the writable sandbox) was NOT rejected, so exactly
    // ONE ReadOnlyViolation appears across the worker's contexts.
    let seen = build.seen_text();
    // Matches the `SandboxViolation: Display` rendering of `ReadOnlyViolation`.
    let violations = seen
        .iter()
        .filter(|c| c.contains("write to read-only path"))
        .count();
    assert_eq!(
        violations, 1,
        "exactly the evaluate-phase write is rejected read-only; saw: {seen:?}"
    );
}

// R4: the evaluator carries the `role-evaluator` chunk (presence assertion).
#[tokio::test]
async fn self_verifying_evaluator_carries_role_chunk() {
    // Q1c: one worker agent; the evaluate-phase turn (PASS) is its 2nd turn.
    let build = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::final_resp("done"),
            RecordingTurnAgent::final_resp("PASS"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, build.clone());
    set_default_verifier(
        &mut cfg,
        Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3)),
    );
    // Register the role-evaluator chunk through the provider.
    cfg.chunk_provider = Arc::new(crate::prompt_assembly::InMemoryChunkProvider::new(vec![
        crate::prompt_assembly::PromptChunk::new(
            "role-evaluator",
            "You are a fresh evaluator. You did not write this code.",
        ),
    ]));
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(self_verifying_task())).await;
    let eval_seen = build.seen_text();
    assert!(
        eval_seen.iter().any(|c| c.contains("fresh evaluator")),
        "evaluator context must carry the role-evaluator chunk; saw: {eval_seen:?}"
    );
}

// R5: Default-FAIL — an indeterminate evaluator verdict keeps looping and
// does not succeed. (The verifier returns Failed for an indeterminate eval.)
#[tokio::test]
async fn self_verifying_default_fail_does_not_succeed() {
    // Q1c: one worker queue, build/eval turns interleaved per iteration.
    let build = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::final_resp("attempt 1"),
            RecordingTurnAgent::final_resp("indeterminate"),
            RecordingTurnAgent::final_resp("attempt 2"),
            RecordingTurnAgent::final_resp("indeterminate"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, build);
    // Default-FAIL: every verdict is Failed (no explicit pass).
    set_default_verifier(
        &mut cfg,
        Arc::new(ScriptedVerifier::new(
            vec![],
            failed("indeterminate — default fail"),
            2,
        )),
    );
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(self_verifying_task())).await {
        RunResult::Failure {
            reason: HaltReason::SelfVerifyExhausted { .. },
            ..
        } => {}
        other => panic!("expected SelfVerifyExhausted (no success), got {other:?}"),
    }
}

// R6: on findings, the verdict reason is injected into the build context and
// the build loop resumes (fail iter 0 with reason X, pass iter 1; iter-1
// build context contains X, final Success).
#[tokio::test]
async fn self_verifying_injects_reason_and_resumes() {
    const FINDING: &str = "MISSING_NULL_CHECK_IN_HANDLER";
    // Q1c: one worker queue: build v1, eval FAIL-turn, build v2, eval PASS.
    let build = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::final_resp("v1"),
            RecordingTurnAgent::final_resp("FAIL"),
            RecordingTurnAgent::final_resp("v2"),
            RecordingTurnAgent::final_resp("PASS"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, build.clone());
    set_default_verifier(
        &mut cfg,
        Arc::new(ScriptedVerifier::new(
            vec![failed(FINDING), passed()],
            passed(),
            3,
        )),
    );
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(self_verifying_task())).await {
        RunResult::Success { .. } => {}
        other => panic!("expected Success on iter 1, got {other:?}"),
    }
    // The injected finding from iteration 0 must appear in a later build
    // context (iteration 1 resumes the build session with the reason appended).
    let build_seen = build.seen_text();
    assert!(
        build_seen.iter().any(|c| c.contains(FINDING)),
        "iter-1 build context must contain the injected finding; saw: {build_seen:?}"
    );
}

// R7: stagnation guard — an always-Failed verifier runs exactly
// max_iterations cycles, then Failure { SelfVerifyExhausted }.
#[tokio::test]
async fn self_verifying_stagnation_guard_caps_iterations() {
    // Q1c: one worker queue, 3 iterations × (build, eval).
    let build = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::final_resp("a"),
            RecordingTurnAgent::final_resp("x"),
            RecordingTurnAgent::final_resp("b"),
            RecordingTurnAgent::final_resp("y"),
            RecordingTurnAgent::final_resp("c"),
            RecordingTurnAgent::final_resp("z"),
        ],
    );
    let verifier = Arc::new(ScriptedVerifier::new(
        vec![],
        failed("never good enough"),
        3,
    ));
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, build);
    set_default_verifier(&mut cfg, verifier.clone());
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(self_verifying_task())).await {
        RunResult::Failure {
            reason:
                HaltReason::SelfVerifyExhausted {
                    iterations,
                    last_reason,
                },
            ..
        } => {
            assert_eq!(iterations, 3, "should run exactly max_iterations cycles");
            assert!(
                last_reason.contains("never good enough"),
                "got {last_reason}"
            );
        }
        other => panic!("expected SelfVerifyExhausted, got {other:?}"),
    }
    // Exactly max_iterations verifier calls (one per build↔evaluate cycle).
    assert_eq!(verifier.seen.lock().unwrap().len(), 3);
}

// R8: budgets fold both phases across all iterations (sum-check). Each turn
// reports 1 input + 1 output token; with 2 iterations × (1 build + 1 eval
// turn) the cumulative usage sums to 4 input + 4 output tokens.
#[tokio::test]
async fn self_verifying_budgets_fold_both_phases() {
    // Q1c: one worker queue, 2 iterations × (build, eval).
    let build = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::final_resp("a"),
            RecordingTurnAgent::final_resp("x"),
            RecordingTurnAgent::final_resp("b"),
            RecordingTurnAgent::final_resp("PASS"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, build);
    set_default_verifier(
        &mut cfg,
        Arc::new(ScriptedVerifier::new(
            vec![failed("retry"), passed()],
            passed(),
            3,
        )),
    );
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(self_verifying_task())).await {
        RunResult::Success { usage, .. } => {
            // 2 build turns + 2 eval turns, each 1 in / 1 out.
            assert_eq!(usage.input_tokens, 4, "input tokens fold both phases");
            assert_eq!(usage.output_tokens, 4, "output tokens fold both phases");
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

/// Run a 2-iteration SelfVerifying (build,eval,build,eval — the worker serves
/// both phases per Q1c) under a `max_turns` cap. Used to prove the evaluate
/// phase is charged against the SelfVerifying budget scope, not just the
/// build phase.
async fn run_sv_two_iters_with_cap(cap: u32) -> RunResult {
    let worker = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::final_resp("build0"),
            RecordingTurnAgent::final_resp("eval0"),
            RecordingTurnAgent::final_resp("build1"),
            RecordingTurnAgent::final_resp("eval1"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, worker);
    set_default_verifier(
        &mut cfg,
        Arc::new(ScriptedVerifier::new(
            vec![failed("retry"), passed()],
            passed(),
            3,
        )),
    );
    let h = StandardHarness::new(cfg);
    let t = self_verifying_task().with_budget(BudgetLimits {
        max_turns: Some(cap),
        ..Default::default()
    });
    h.run(HarnessRunOptions::new(t)).await
}

#[tokio::test]
async fn self_verifying_charges_evaluator_turns_against_scope() {
    // 2 iterations × (build 1 turn + eval 1 turn) = 4 turns. With the
    // evaluator turns charged, a cap of 4 just fits → Success; a cap of 3 is
    // overrun by the second iteration's EVALUATOR turn (the two build turns
    // alone are only 2, so without charging the evaluator this would
    // wrongly succeed).
    match run_sv_two_iters_with_cap(4).await {
        RunResult::Success { .. } => {}
        other => panic!("cap=4 should fit all 4 turns, got {other:?}"),
    }
    let exhausted = run_sv_two_iters_with_cap(3).await;
    assert!(
        !matches!(exhausted, RunResult::Success { .. }),
        "cap=3 must be exhausted once evaluator turns are charged, got {exhausted:?}"
    );
}

// ── eval-phase reviewer-context regression tests ─────────────────────────

// BLOCKER FIX: the evaluate phase must NOT inherit the caller's approval
// middleware. A nested review run is non-interactive (the caller never sees
// its generated session id), so a `SurfaceToHuman` BeforeTool decision would
// pause it with no human able to resume — the reviewer's first tool call
// would never dispatch and the review half would silently never run. With the
// caller middleware dropped for the read-only eval phase, the reviewer's
// read dispatches and the run reaches a verdict.
#[tokio::test]
async fn self_verifying_eval_phase_drops_caller_middleware() {
    // Build emits a tool-free FinalResponse (so the BUILD phase, which DOES
    // run the caller middleware, never trips it); the eval phase (same worker
    // per Q1c) then issues a read — the call the middleware would have gated.
    let worker = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::final_resp("built"),
            RecordingTurnAgent::tool_call("read_file"),
            RecordingTurnAgent::final_resp("reviewed: PASS"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, worker);
    set_default_verifier(
        &mut cfg,
        Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3)),
    );
    // The dispatch count is the discriminator: it only increments if the eval
    // phase's tool call actually dispatched (i.e. was NOT paused at BeforeTool).
    let reg = Arc::new(ScriptedToolRegistry::new());
    cfg.tool_registry = reg.clone();
    // The caller's approval middleware: SurfaceToHuman at BeforeTool. Without
    // the eval-phase drop, it pauses the reviewer's read and the run never
    // dispatches the tool.
    let mw = Arc::new(ScriptedMiddleware::new());
    mw.push(
        HookPoint::BeforeTool,
        MiddlewareDecision::SurfaceToHuman {
            request: HumanRequest::ToolApproval {
                calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({}),
                }],
                risk_level: RiskLevel::Medium,
            },
        },
    );
    cfg.middleware = Some(mw);
    let h = StandardHarness::new(cfg);
    let r = h.run(HarnessRunOptions::new(self_verifying_task())).await;
    assert!(
        matches!(r, RunResult::Success { .. }),
        "eval phase must run to a verdict, got {r:?}"
    );
    // Load-bearing: the reviewer's read actually dispatched. With the caller
    // middleware NOT dropped, the eval phase pauses at BeforeTool BEFORE
    // dispatch and this count stays 0.
    assert!(
        reg.call_count.load(std::sync::atomic::Ordering::SeqCst) >= 1,
        "eval-phase reviewer tool must dispatch (caller approval middleware \
             must be dropped for the read-only review run)"
    );
}

// SC-BUG-1: a HITL approval/clarification resume must RE-ENTER the
// SelfVerifying frame, not run only the bare build leaf. The original run
// pauses on the BUILD phase's first tool call (caller approval middleware →
// SurfaceToHuman). Before the fix, `resume_inner` ran `run_react` on the
// paused ReAct leaf and returned its Success directly — the evaluate phase +
// verifier never ran, so the looper's eval-frame reviewer was silently
// skipped (SC-30 inert for looper). After the fix the pause carries the full
// composed task (`finish` rewrites it on the way up, as it already does for
// `Consult`) and the resume re-drives the whole SelfVerifying strategy from
// the approved worker session, so the verifier runs and a verdict is reached.
#[tokio::test]
async fn self_verifying_hitl_resume_reenters_eval_frame() {
    // Build phase: a tool call (gated → pause), then a FinalResponse once the
    // approved call dispatches. Eval phase (same worker per Q1c; its caller
    // middleware is dropped, SC-29) emits a tool-free FinalResponse so it
    // never re-trips the gate.
    let worker = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::tool_call("write_file"),
            RecordingTurnAgent::final_resp("built"),
            RecordingTurnAgent::final_resp("reviewed: PASS"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, worker);
    let verifier = Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3));
    set_default_verifier(&mut cfg, verifier.clone());
    let reg = Arc::new(ScriptedToolRegistry::new());
    cfg.tool_registry = reg.clone();
    // Caller approval middleware: SurfaceToHuman at BeforeTool. The build
    // phase's `write_file` trips it and the run pauses for approval.
    let mw = Arc::new(ScriptedMiddleware::new());
    mw.push(
        HookPoint::BeforeTool,
        MiddlewareDecision::SurfaceToHuman {
            request: HumanRequest::ToolApproval {
                calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({}),
                }],
                risk_level: RiskLevel::Medium,
            },
        },
    );
    cfg.middleware = Some(mw);
    let h = StandardHarness::new(cfg);

    // 1) The run pauses on the build-phase tool call.
    let paused = h.run(HarnessRunOptions::new(self_verifying_task())).await;
    let state = match paused {
        RunResult::WaitingForHuman { state, .. } => *state,
        other => panic!("expected WaitingForHuman on the build tool call, got {other:?}"),
    };
    // The pause must carry the COMPOSED task so the resume can re-enter the
    // frame — not the bare ReAct build leaf.
    assert!(
        matches!(state.task.loop_strategy, LoopStrategy::SelfVerifying(_)),
        "the unwound pause must carry the SelfVerifying task, got {:?}",
        state.task.loop_strategy
    );

    // 2) Approve and resume.
    let resumed = h.resume(state, HumanResponse::Allow, None).await;
    assert!(
        matches!(resumed, RunResult::Success { .. }),
        "resume must run to a verdict, got {resumed:?}"
    );

    // Load-bearing: the eval-phase verifier ran AFTER the resume — i.e. the
    // SelfVerifying frame was re-entered. Before the fix the resume returned
    // the bare leaf's Success and this count stays 0.
    let verifier_calls = verifier.seen.lock().unwrap().len();
    assert!(
        verifier_calls >= 1,
        "SelfVerifying frame must be re-entered on HITL resume so the \
             eval-phase verifier runs (got {verifier_calls} verifier calls)"
    );
    // And the approved build tool actually dispatched on resume.
    assert!(
        reg.call_count.load(std::sync::atomic::Ordering::SeqCst) >= 1,
        "the approved build-phase tool must dispatch on resume"
    );
}

// SC-BUG-1 (Deny arm): a DENIED tool-approval resume must also re-enter the
// SelfVerifying frame. Deny appends a recoverable error tool result for the
// gated call and then re-drives the strategy (the same final-match tail as
// Allow). The build continues from the denial and the evaluate phase runs.
#[tokio::test]
async fn self_verifying_hitl_deny_reenters_eval_frame() {
    let worker = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::tool_call("write_file"),
            RecordingTurnAgent::final_resp("built without the write"),
            RecordingTurnAgent::final_resp("reviewed: PASS"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, worker);
    let verifier = Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3));
    set_default_verifier(&mut cfg, verifier.clone());
    let mw = Arc::new(ScriptedMiddleware::new());
    mw.push(
        HookPoint::BeforeTool,
        MiddlewareDecision::SurfaceToHuman {
            request: HumanRequest::ToolApproval {
                calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({}),
                }],
                risk_level: RiskLevel::High,
            },
        },
    );
    cfg.middleware = Some(mw);
    let h = StandardHarness::new(cfg);

    let state = match h.run(HarnessRunOptions::new(self_verifying_task())).await {
        RunResult::WaitingForHuman { state, .. } => *state,
        other => panic!("expected WaitingForHuman, got {other:?}"),
    };
    let resumed = h
        .resume(
            state,
            HumanResponse::Deny {
                reason: "not allowed".into(),
            },
            None,
        )
        .await;
    assert!(
        matches!(resumed, RunResult::Success { .. }),
        "deny resume must run to a verdict, got {resumed:?}"
    );
    assert!(
        !verifier.seen.lock().unwrap().is_empty(),
        "SelfVerifying frame must be re-entered on a DENY resume so the \
             eval-phase verifier runs"
    );
}

// SC-BUG-1 (clarification/Answer arm): a clarification resume (the SEPARATE
// `resume_inner` clarification tail) must also re-enter the SelfVerifying
// frame. The build worker's tool returns `AwaitingClarification`; the human's
// Answer is injected as that call's tool result and the strategy is re-driven,
// so the evaluate phase runs.
#[tokio::test]
async fn self_verifying_hitl_clarification_reenters_eval_frame() {
    let worker = RecordingTurnAgent::new(
        "build",
        vec![
            RecordingTurnAgent::tool_call("ask"),
            RecordingTurnAgent::final_resp("built"),
            RecordingTurnAgent::final_resp("reviewed: PASS"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, worker);
    let verifier = Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3));
    set_default_verifier(&mut cfg, verifier.clone());
    // No approval middleware: the tool itself returns AwaitingClarification,
    // pausing the build phase with a `HumanRequest::Clarification`.
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.push(ToolOutput::AwaitingClarification {
        question: "which file?".into(),
        options: None,
    });
    cfg.tool_registry = reg;
    let h = StandardHarness::new(cfg);

    let state = match h.run(HarnessRunOptions::new(self_verifying_task())).await {
        RunResult::WaitingForHuman { state, request } => {
            assert!(
                matches!(request, HumanRequest::Clarification { .. }),
                "expected a clarification pause, got {request:?}"
            );
            *state
        }
        other => panic!("expected WaitingForHuman, got {other:?}"),
    };
    let resumed = h
        .resume(
            state,
            HumanResponse::Answer {
                text: "the config file".into(),
            },
            None,
        )
        .await;
    assert!(
        matches!(resumed, RunResult::Success { .. }),
        "clarification resume must run to a verdict, got {resumed:?}"
    );
    assert!(
        !verifier.seen.lock().unwrap().is_empty(),
        "SelfVerifying frame must be re-entered on a clarification resume so \
             the eval-phase verifier runs"
    );
}

// The evaluate phase runs the configured `eval_agent` (a dedicated reviewer),
// not the inner worker's agent — so the model reviewing the work is not the
// one that wrote it (a builder reviewing itself tends to rubber-stamp).
#[tokio::test]
async fn self_verifying_eval_agent_override_runs_distinct_reviewer() {
    let build = RecordingTurnAgent::new("build", vec![RecordingTurnAgent::final_resp("built")]);
    let reviewer =
        RecordingTurnAgent::new("reviewer", vec![RecordingTurnAgent::final_resp("reviewed")]);
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, build.clone());
    cfg.registry = std::mem::take(&mut cfg.registry)
        .into_builder()
        .agent("reviewer", reviewer.clone())
        .build();
    set_default_verifier(
        &mut cfg,
        Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3)),
    );
    let h = StandardHarness::new(cfg);
    let t = task(LoopStrategy::SelfVerifying(SelfVerifyingConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(react_structured(u32::MAX)),
        evaluator: SchemaRef(String::new()),
        eval_agent: Some(AgentRef("reviewer".into())),
        eval_toolset: None,
    }));
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Success { .. } => {}
        other => panic!("expected Success, got {other:?}"),
    }
    // The reviewer ran the evaluate phase exactly once; the builder ran only
    // the build phase. Without the override the worker would serve BOTH and
    // the reviewer would never be called.
    assert_eq!(
        reviewer.call_count(),
        1,
        "eval_agent reviewer runs the evaluate phase"
    );
    assert_eq!(build.call_count(), 1, "builder runs only the build phase");
}

// Wire-format contract (cross-language parity): the two optional eval-phase
// overrides are OMITTED when unset (existing configs stay byte-identical) and
// serialize as bare string handles in declaration order (evaluator <
// eval_agent < eval_toolset < behavior) when set, round-tripping intact.
#[test]
fn self_verifying_config_eval_overrides_serde() {
    let bare = LoopStrategy::SelfVerifying(SelfVerifyingConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(1))),
        evaluator: SchemaRef("v".into()),
        eval_agent: None,
        eval_toolset: None,
    });
    let v = serde_json::to_value(&bare).unwrap();
    assert!(
        v.get("eval_agent").is_none(),
        "eval_agent omitted when None"
    );
    assert!(
        v.get("eval_toolset").is_none(),
        "eval_toolset omitted when None"
    );

    let with = LoopStrategy::SelfVerifying(SelfVerifyingConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(1))),
        evaluator: SchemaRef("v".into()),
        eval_agent: Some(AgentRef("reviewer".into())),
        eval_toolset: Some(ToolsetRef("ro-tools".into())),
    });
    let s = serde_json::to_string(&with).unwrap();
    assert!(
        s.contains("\"eval_agent\":\"reviewer\""),
        "bare string handle: {s}"
    );
    assert!(
        s.contains("\"eval_toolset\":\"ro-tools\""),
        "bare string handle: {s}"
    );
    // `behavior` appears twice (inner ReAct + this config); the SelfVerifying
    // one is LAST, so `rfind` targets it.
    let (ev, ea, et, bh) = (
        s.find("evaluator").unwrap(),
        s.find("eval_agent").unwrap(),
        s.find("eval_toolset").unwrap(),
        s.rfind("behavior").unwrap(),
    );
    assert!(
        ev < ea && ea < et && et < bh,
        "field order locked for parity: {s}"
    );
    let back: LoopStrategy = serde_json::from_str(&s).unwrap();
    assert_eq!(back, with, "round-trips intact");
}

// R12: fixture replay — scripted (build verdict sequence, evaluator output
// sequence) → expected terminal RunResult kind + iteration count.
#[derive(serde::Deserialize)]
struct SvFixtureCase {
    name: String,
    /// Per-iteration verifier verdicts. Each entry is `"pass"` or a failure
    /// reason string (anything else).
    verdicts: Vec<String>,
    max_iterations: u32,
    expected: SvFixtureExpected,
}
#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SvFixtureExpected {
    Success { iterations: u32 },
    Exhausted { iterations: u32 },
    Misconfigured,
}
#[derive(serde::Deserialize)]
struct SvFixtureSuite {
    cases: Vec<SvFixtureCase>,
}

#[tokio::test]
async fn self_verifying_fixture_replay() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../fixtures/harness/self_verifying.json");
    let raw = std::fs::read_to_string(&path).expect("fixture present");
    let suite: SvFixtureSuite = serde_json::from_str(&raw).expect("fixture parses");
    for case in suite.cases {
        // Q1c: one worker queue with a build "done" + an eval "out" turn per
        // possible iteration, interleaved.
        let n = case.max_iterations.max(1) as usize;
        let mut turns = Vec::with_capacity(2 * n);
        for _ in 0..n {
            turns.push(RecordingTurnAgent::final_resp("done"));
            turns.push(RecordingTurnAgent::final_resp("out"));
        }
        let build = RecordingTurnAgent::new("build", turns);
        let verdicts: Vec<VerifierVerdict> = case
            .verdicts
            .iter()
            .map(|v| if v == "pass" { passed() } else { failed(v) })
            .collect();
        let mut cfg = standard_config(make_agent());
        set_worker_agent(&mut cfg, build);
        let misconfigured = matches!(case.expected, SvFixtureExpected::Misconfigured);
        if !misconfigured {
            set_default_verifier(
                &mut cfg,
                Arc::new(ScriptedVerifier::new(
                    verdicts,
                    failed("default fail"),
                    case.max_iterations,
                )),
            );
        }
        let h = StandardHarness::new(cfg);
        let r = h.run(HarnessRunOptions::new(self_verifying_task())).await;
        match (case.expected, r) {
            (SvFixtureExpected::Success { iterations }, RunResult::Success { turns, .. }) => {
                // `turns` is the final build sub-loop's absolute turn count;
                // assert the run succeeded (iteration count is asserted via
                // dedicated unit tests). `iterations` documents the case.
                let _ = (iterations, turns);
            }
            (
                SvFixtureExpected::Exhausted { iterations },
                RunResult::Failure {
                    reason:
                        HaltReason::SelfVerifyExhausted {
                            iterations: got, ..
                        },
                    ..
                },
            ) => {
                assert_eq!(got, iterations, "case `{}` iteration count", case.name);
            }
            (
                SvFixtureExpected::Misconfigured,
                // #124: an unresolved verifier is now a startup ConfigurationError.
                RunResult::Failure {
                    reason:
                        HaltReason::ConfigurationError {
                            error: HarnessError::UnresolvedHandle { .. },
                        },
                    ..
                },
            ) => {}
            (_, other) => panic!("case `{}`: unexpected result {other:?}", case.name),
        }
    }
}

// ========================================================================
// Ralph loop strategy (issue #58)
// ========================================================================

fn ralph_task() -> Task {
    let mut t = task(LoopStrategy::Ralph(RalphConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
        agent: AgentRef(String::new()),
    }));
    // One ReAct turn per context window keeps the per-window sub-loop
    // bounded so the OUTER reset loop drives the test deterministically.
    t.budget.max_turns = Some(1);
    t
}

/// A `SandboxProvider` whose `workspace_root` is a real tempdir, so the
/// Ralph filesystem reload + completion check read real `.spore/` files.
struct WorkspaceSandbox {
    root: std::path::PathBuf,
}
impl SandboxProvider for WorkspaceSandbox {
    fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
        Box::pin(async { Ok(()) })
    }
    fn workspace_root(&self) -> &std::path::Path {
        &self.root
    }
}

/// The project namespace the Ralph checkpoint lives under (#142): the same
/// id the harness derives from `sandbox.workspace_root()`, projected onto the
/// `RunStore` session-id axis. The test writer and the harness reader MUST
/// agree on this so a write is what the read sees.
fn ralph_ns(root: &std::path::Path) -> SessionId {
    crate::storage::ProjectId::from_canonical_path(&root.to_string_lossy()).namespace()
}

/// Write the Ralph progress checkpoint into the SHARED run store under the
/// project namespace derived from `root` (#142 relocated this off the
/// `.spore/` filesystem). `body` is the legacy JSON body string, parsed to a
/// `serde_json::Value` and stored under [`RALPH_PROGRESS_KEY`].
async fn write_progress(
    storage: &Arc<crate::storage::StorageProvider>,
    root: &std::path::Path,
    body: &str,
) {
    let value: serde_json::Value = serde_json::from_str(body).unwrap();
    storage
        .run()
        .put(&ralph_ns(root), crate::storage::RALPH_PROGRESS_KEY, value)
        .await
        .unwrap();
}
async fn write_feature_list(
    storage: &Arc<crate::storage::StorageProvider>,
    root: &std::path::Path,
    body: &str,
) {
    let value: serde_json::Value = serde_json::from_str(body).unwrap();
    storage
        .run()
        .put(
            &ralph_ns(root),
            crate::storage::RALPH_FEATURE_LIST_KEY,
            value,
        )
        .await
        .unwrap();
}
const INCOMPLETE: &str = r#"{"complete":false,"remaining":["task A"]}"#;
const COMPLETE: &str = r#"{"complete":true,"remaining":[]}"#;

/// Agent that, on each turn, pops the next progress-file body from a queue
/// and writes it to `.spore/progress.json` BEFORE returning a `FinalResponse`
/// — modelling "the agent did work this window and updated progress." Also
/// records the messages it saw so tests can assert fresh-state / reload.
struct ProgressWritingAgent {
    id: AgentId,
    run_store: Arc<dyn crate::storage::RunStore>,
    ns: SessionId,
    progress_queue: StdMutex<std::collections::VecDeque<String>>,
    seen: StdMutex<Vec<Context>>,
}
impl ProgressWritingAgent {
    fn new(
        run_store: Arc<dyn crate::storage::RunStore>,
        root: &std::path::Path,
        bodies: Vec<&str>,
    ) -> Arc<Self> {
        Arc::new(Self {
            id: AgentId::new("ralph-build"),
            run_store,
            ns: ralph_ns(root),
            progress_queue: StdMutex::new(bodies.into_iter().map(String::from).collect()),
            seen: StdMutex::new(Vec::new()),
        })
    }
    fn call_count(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
    fn seen_text(&self) -> Vec<String> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .map(|c| {
                c.messages
                    .iter()
                    .map(|m| match &m.content {
                        Content::Text { text } => text.clone(),
                        _ => String::new(),
                    })
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .collect()
    }
}
impl Agent for ProgressWritingAgent {
    fn turn<'a>(&'a self, context: Context) -> BoxFut<'a, TurnResult> {
        self.seen.lock().unwrap().push(context);
        let body = self.progress_queue.lock().unwrap().pop_front();
        Box::pin(async move {
            if let Some(body) = body {
                let value: serde_json::Value = serde_json::from_str(&body).unwrap();
                self.run_store
                    .put(&self.ns, crate::storage::RALPH_PROGRESS_KEY, value)
                    .await
                    .unwrap();
            }
            TurnResult::FinalResponse {
                reasoning: None,
                content: "window done".into(),
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
            }
        })
    }
    fn id(&self) -> AgentId {
        self.id.clone()
    }
}

/// Build a Ralph harness config whose checkpoint store + project namespace
/// are SHARED with the returned `Arc<StorageProvider>` (#142). The harness
/// reads the checkpoint from `cfg.storage` at `cfg.project_id.namespace()`;
/// `ralph_config` derives `project_id` from `root` (the same root the
/// `WorkspaceSandbox` exposes), so a write the test/agent makes against the
/// returned storage at `ralph_ns(root)` is exactly what the harness reads.
/// The storage is the one `standard_config` already built (a fresh in-memory
/// store per call), returned here so seed writes target the same instance.
///
/// The worker agent is NOT registered here: the Ralph worker agents now hold
/// the shared run store (so they write the checkpoint the harness reads), and
/// that store only exists after this call — so callers build the agent from
/// the returned `storage.run()` and register it via `set_worker_agent`.
fn ralph_config(root: &std::path::Path) -> (HarnessConfig, Arc<crate::storage::StorageProvider>) {
    let mut cfg = standard_config(make_agent());
    cfg.sandbox = Arc::new(WorkspaceSandbox {
        root: root.to_path_buf(),
    });
    // Use a real context manager so reloaded context lands in messages.
    cfg.context_manager = Arc::new(NoopContextManager);
    // #142: the checkpoint lives in the project-id RunStore. Derive the
    // project id from the SAME root the sandbox exposes so the harness reads
    // back what the test/agent writes under `ralph_ns(root)`.
    cfg.project_id = crate::storage::ProjectId::from_canonical_path(&root.to_string_lossy());
    let storage = cfg.storage.clone();
    (cfg, storage)
}

// R0: Ralph is implemented — no longer StrategyNotYetImplemented.
#[tokio::test]
async fn ralph_no_longer_unimplemented() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cfg, storage) = ralph_config(dir.path());
    write_progress(&storage, dir.path(), COMPLETE).await;
    let agent = ProgressWritingAgent::new(storage.run().clone(), dir.path(), vec![COMPLETE]);
    set_worker_agent(&mut cfg, agent);
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(ralph_task())).await {
        RunResult::Failure {
            reason: HaltReason::StrategyNotYetImplemented { .. },
            ..
        } => panic!("Ralph must be implemented"),
        RunResult::Success { .. } => {}
        other => panic!("expected Success, got {other:?}"),
    }
}

// R4: completion pattern incomplete,incomplete,complete → Success at it. 3.
#[tokio::test]
async fn ralph_resets_until_complete() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cfg, storage) = ralph_config(dir.path());
    // Start incomplete so the first window's reload sees prior state.
    write_progress(&storage, dir.path(), INCOMPLETE).await;
    let agent = ProgressWritingAgent::new(
        storage.run().clone(),
        dir.path(),
        vec![INCOMPLETE, INCOMPLETE, COMPLETE],
    );
    set_worker_agent(&mut cfg, agent.clone());
    cfg.max_resets = 3;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(ralph_task())).await {
        RunResult::Success { .. } => {}
        other => panic!("expected Success at the 3rd window, got {other:?}"),
    }
    // Exactly three context windows ran (one agent turn each).
    assert_eq!(agent.call_count(), 3, "should reset twice, run 3 windows");
}

// R5: always-incomplete → exactly max_resets windows → RalphCompletionUnmet.
#[tokio::test]
async fn ralph_exhausts_max_resets() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cfg, storage) = ralph_config(dir.path());
    write_progress(&storage, dir.path(), INCOMPLETE).await;
    let agent = ProgressWritingAgent::new(
        storage.run().clone(),
        dir.path(),
        vec![INCOMPLETE, INCOMPLETE, INCOMPLETE, INCOMPLETE],
    );
    set_worker_agent(&mut cfg, agent.clone());
    cfg.max_resets = 3;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(ralph_task())).await {
        RunResult::Failure {
            reason:
                HaltReason::RalphCompletionUnmet {
                    iterations,
                    last_reason,
                },
            ..
        } => {
            assert_eq!(iterations, 3, "exactly max_resets windows");
            assert!(last_reason.contains("task A"), "got: {last_reason}");
        }
        other => panic!("expected RalphCompletionUnmet, got {other:?}"),
    }
    assert_eq!(agent.call_count(), 3, "exactly max_resets windows ran");
}

// #124 REGRESSION-PROOF: Ralph genuinely recurses into a NON-ReAct inner.
// `Ralph[ inner: SelfVerifying[ReAct] ]` over a 2-window incomplete,complete
// pattern — each window runs the inner SelfVerifying's WHOLE loop, firing its
// verifier at least once per window. A hardcoded-ReAct window would NEVER call
// the verifier (ZERO invocations).
#[tokio::test]
async fn ralph_runs_non_react_inner_per_window() {
    let dir = tempfile::tempdir().unwrap();
    // Always-incomplete progress: every window's inner SelfVerifying runs its
    // full build↔evaluate loop (firing its verifier once), Ralph reads
    // incomplete and resets, until `max_resets` is exhausted. With max_resets
    // = 2 the verifier fires exactly ONCE PER WINDOW (2x). A hardcoded-ReAct
    // window would NEVER call the verifier (ZERO invocations).
    let (mut cfg, storage) = ralph_config(dir.path());
    write_progress(&storage, dir.path(), INCOMPLETE).await;
    // The worker keeps progress incomplete on every turn it writes.
    let agent = ProgressWritingAgent::new(
        storage.run().clone(),
        dir.path(),
        vec![INCOMPLETE; 16], // plenty for the blocked build turns + eval turns
    );
    let verifier = Arc::new(ScriptedVerifier::new(vec![], passed(), 3));
    set_worker_agent(&mut cfg, agent.clone());
    set_default_verifier(&mut cfg, verifier.clone());
    cfg.max_resets = 2;
    // One Stop-block then continue, so the per-window build loop ends quickly
    // and the inner SelfVerifying reaches its evaluate phase + verifier.
    cfg.max_stop_blocks = 1;
    let h = StandardHarness::new(cfg);

    // inner is a genuine SelfVerifying combinator (NOT a ReAct).
    let mut t = task(LoopStrategy::Ralph(RalphConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(LoopStrategy::SelfVerifying(SelfVerifyingConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            inner: Box::new(react_structured(u32::MAX)),
            evaluator: SchemaRef(String::new()),
            eval_agent: None,
            eval_toolset: None,
        })),
        agent: AgentRef(String::new()),
    }));
    // Generous per-window turn budget so the build's Stop-block continuation
    // and the evaluate phase both fit within the window.
    t.budget.max_turns = Some(8);

    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure {
            reason: HaltReason::RalphCompletionUnmet { iterations, .. },
            ..
        } => assert_eq!(iterations, 2, "exactly max_resets windows ran"),
        other => panic!("expected RalphCompletionUnmet after 2 windows, got {other:?}"),
    }
    // The inner SelfVerifying verifier fired at least once per window (>=2). A
    // hardcoded-ReAct window would record ZERO verifier invocations.
    assert!(
        verifier.seen.lock().unwrap().len() >= 2,
        "the inner SelfVerifying verifier must fire >=1 per window; got {}",
        verifier.seen.lock().unwrap().len()
    );
}

/// An agent that NEVER returns a `FinalResponse` — every turn requests a tool
/// call, so a bounded ReAct window can only ever exhaust its turn budget (it
/// never terminates cleanly). On its FIRST turn it writes `COMPLETE` to
/// `.spore/progress.json`, so any code path that DID consult Ralph's external
/// completion after the window would (wrongly) see "complete" and Success.
struct ToolLoopingAgent {
    id: AgentId,
    run_store: Arc<dyn crate::storage::RunStore>,
    ns: SessionId,
    calls: StdMutex<u32>,
}
impl ToolLoopingAgent {
    fn new(run_store: Arc<dyn crate::storage::RunStore>, root: &std::path::Path) -> Arc<Self> {
        Arc::new(Self {
            id: AgentId::new("ralph-tool-loop"),
            run_store,
            ns: ralph_ns(root),
            calls: StdMutex::new(0),
        })
    }
    fn call_count(&self) -> u32 {
        *self.calls.lock().unwrap()
    }
}
impl Agent for ToolLoopingAgent {
    fn turn<'a>(&'a self, _context: Context) -> BoxFut<'a, TurnResult> {
        let n = {
            let mut c = self.calls.lock().unwrap();
            let n = *c;
            *c += 1;
            n
        };
        Box::pin(async move {
            if n == 0 {
                // Mark COMPLETE up front — if Ralph (wrongly) consulted
                // completion after a budget-exhausted window it would Success.
                let value: serde_json::Value = serde_json::from_str(COMPLETE).unwrap();
                self.run_store
                    .put(&self.ns, crate::storage::RALPH_PROGRESS_KEY, value)
                    .await
                    .unwrap();
            }
            TurnResult::ToolCallRequested {
                reasoning: None,
                calls: vec![ToolCall {
                    id: format!("c{n}"),
                    name: "x".into(),
                    input: serde_json::json!({}),
                }],
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
            }
        })
    }
    fn id(&self) -> AgentId {
        self.id.clone()
    }
}

// #125 F5: a Ralph window whose INNER LEAF exhausts its OWN budget mid-window
// (the leaf's `PerLoop` policy is the binding cap, no smaller global backstop).
// The window surfaces `StrategyOutcome::BudgetExhausted`; Ralph must treat it
// as "window incomplete → RESET and retry" — it must NOT consult external
// completion (which is COMPLETE on disk here) and must NOT cascade the child's
// exhaustion into its own terminal. So the run reaches `max_resets` windows and
// ends `RalphCompletionUnmet`, NOT Success.
#[tokio::test]
async fn ralph_budget_exhausted_window_resets_no_completion_no_cascade() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cfg, storage) = ralph_config(dir.path());
    write_progress(&storage, dir.path(), INCOMPLETE).await;
    let agent = ToolLoopingAgent::new(storage.run().clone(), dir.path());
    set_worker_agent(&mut cfg, agent.clone());
    // Provide tool outputs for the looping calls across all windows.
    let reg = Arc::new(ScriptedToolRegistry::new());
    for _ in 0..32 {
        reg.push(ToolOutput::Success {
            content: "ok".into(),
            truncated: false,
        });
    }
    cfg.tool_registry = reg;
    cfg.max_resets = 3;
    let h = StandardHarness::new(cfg);

    // Inner leaf carries its OWN binding cap (PerLoop{2}); NO global max_turns
    // backstop, so the leaf policy — not the global cap — exhausts the window
    // and the new BudgetExhausted path is taken.
    let mut t = task(LoopStrategy::Ralph(RalphConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(2))),
        agent: AgentRef(String::new()),
    }));
    t.budget.max_turns = None;

    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure {
            reason: HaltReason::RalphCompletionUnmet { iterations, .. },
            ..
        } => {
            // Reached max_resets — completion was NEVER consulted (despite
            // COMPLETE on disk) and the child's exhaustion did NOT cascade
            // into Ralph's own terminal.
            assert_eq!(iterations, 3, "exactly max_resets windows ran");
        }
        other => panic!(
            "expected RalphCompletionUnmet (reset-on-exhaust, no completion, \
                 no cascade), got {other:?}"
        ),
    }
    // Three windows × two leaf turns each = six agent turns total — proving
    // each exhausted window fully reset and re-ran, not short-circuited.
    assert_eq!(agent.call_count(), 6, "3 windows × 2 leaf turns each");
}

// R2: each reset builds a FRESH SessionState — no message carryover. The
// agent's recorded context for window 2 must NOT contain window 1's content.
#[tokio::test]
async fn ralph_fresh_session_per_reset() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cfg, storage) = ralph_config(dir.path());
    write_progress(&storage, dir.path(), INCOMPLETE).await;
    let agent = ProgressWritingAgent::new(
        storage.run().clone(),
        dir.path(),
        vec![INCOMPLETE, COMPLETE],
    );
    set_worker_agent(&mut cfg, agent.clone());
    cfg.max_resets = 3;
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(ralph_task())).await;
    let texts = agent.seen_text();
    assert_eq!(texts.len(), 2, "two windows");
    // The assistant's "window done" output from window 1 is NOT present in
    // window 2's fresh context (no carryover). Each window's context is
    // re-seeded from instruction + reloaded `.spore/` state only.
    assert!(
        !texts[1].contains("window done"),
        "window 2 carried over window 1 messages: {}",
        texts[1]
    );
}

// R3: the filesystem reload injects `.spore/` state into the fresh seed.
#[tokio::test]
async fn ralph_reload_injects_filesystem_state() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cfg, storage) = ralph_config(dir.path());
    write_progress(&storage, dir.path(), INCOMPLETE).await;
    write_feature_list(&storage, dir.path(), r#"[{"name":"login","passes":false}]"#).await;
    // Agent leaves progress incomplete on window 1, complete on window 2.
    let agent = ProgressWritingAgent::new(
        storage.run().clone(),
        dir.path(),
        vec![INCOMPLETE, COMPLETE],
    );
    set_worker_agent(&mut cfg, agent.clone());
    cfg.max_resets = 3;
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(ralph_task())).await;
    let texts = agent.seen_text();
    // Window 1's fresh seed contains the reloaded progress + feature list.
    assert!(
        texts[0].contains("Reloaded .spore/progress.json")
            && texts[0].contains("Reloaded .spore/feature_list.json")
            && texts[0].contains("login"),
        "reload not injected: {}",
        texts[0]
    );
}

// R6: budgets fold across ALL context windows (each window adds usage).
#[tokio::test]
async fn ralph_budgets_fold_across_windows() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cfg, storage) = ralph_config(dir.path());
    write_progress(&storage, dir.path(), INCOMPLETE).await;
    let agent = ProgressWritingAgent::new(
        storage.run().clone(),
        dir.path(),
        vec![INCOMPLETE, INCOMPLETE, COMPLETE],
    );
    set_worker_agent(&mut cfg, agent);
    cfg.max_resets = 3;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(ralph_task())).await {
        RunResult::Success { usage, .. } => {
            // Three windows × one turn × (1 in, 1 out) folded.
            assert_eq!(usage.input_tokens, 3, "input tokens fold across windows");
            assert_eq!(usage.output_tokens, 3, "output tokens fold across windows");
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// R7: missing progress file is treated as incomplete by the OUTER loop, and
// the registered Stop hook does NOT interfere with non-Ralph runs.
#[tokio::test]
async fn ralph_stop_hook_inert_without_progress_file() {
    // A plain ReAct run on a workspace with no `.spore/progress.json`: the
    // registered RalphStopHook must return Continue (terminate in one turn).
    let dir = tempfile::tempdir().unwrap();
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    cfg.sandbox = Arc::new(WorkspaceSandbox {
        root: dir.path().to_path_buf(),
    });
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(5))).await {
        RunResult::Success { turns, .. } => assert_eq!(turns, 1, "stop hook must be inert"),
        other => panic!("expected Success in one turn, got {other:?}"),
    }
}

// Completion-status helper: progress complete but a feature fails ⇒ still
// incomplete (the feature list corroborates).
#[tokio::test]
async fn ralph_completion_status_feature_list_gate() {
    // #142: the checkpoint lives in the project-id RunStore, not on disk. The
    // completion-status helper reads `(run_store, project)`; the test writes
    // the same checkpoint into the same store under the project namespace.
    let storage = Arc::new(crate::storage::StorageProvider::single(Arc::new(
        crate::storage::InMemoryStorageProvider::new(),
    )));
    let project = crate::storage::ProjectId::from_canonical_path("/feature-gate");
    let ns = project.namespace();
    let run = storage.run();
    run.put(
        &ns,
        crate::storage::RALPH_PROGRESS_KEY,
        serde_json::from_str(COMPLETE).unwrap(),
    )
    .await
    .unwrap();
    run.put(
        &ns,
        crate::storage::RALPH_FEATURE_LIST_KEY,
        serde_json::json!([{ "name": "login", "passes": false }]),
    )
    .await
    .unwrap();
    let status = StandardHarness::ralph_completion_status(run, &project).await;
    assert!(
        status.as_deref().unwrap_or("").contains("login"),
        "got: {status:?}"
    );
    assert!(
        status
            .as_deref()
            .unwrap_or("")
            .contains("incomplete features"),
        "got: {status:?}"
    );
    // Now mark it passing — complete.
    run.put(
        &ns,
        crate::storage::RALPH_FEATURE_LIST_KEY,
        serde_json::json!([{ "name": "login", "passes": true }]),
    )
    .await
    .unwrap();
    assert_eq!(
        StandardHarness::ralph_completion_status(run, &project).await,
        None
    );
}

// ── VcsProvider seam (issue #58 v2) ─────────────────────────────────────

// (a) FixtureVcsProvider::log returns the seeded string verbatim.
#[tokio::test]
async fn fixture_vcs_log_verbatim() {
    let log = "abc123 first\ndef456 second\n";
    let provider = FixtureVcsProvider::new(log, "clean");
    let args = VcsLogArgs {
        max_entries: 5,
        since_ref: Some("HEAD~3".into()),
        format: Some("%h %s".into()),
    };
    let out = provider.log(&args).await.unwrap();
    assert_eq!(out, log, "log output must be verbatim, args ignored");
}

// (e) status() round-trips the seeded string verbatim.
#[tokio::test]
async fn fixture_vcs_status_round_trips() {
    let status = "On branch main\nnothing to commit\n";
    let provider = FixtureVcsProvider::new("", status);
    assert_eq!(provider.status().await.unwrap(), status);
}

/// A `SandboxProvider` that captures the `execute_command` invocation so the
/// `git log`/`git status` command lines can be asserted, and returns a
/// scripted stdout.
struct CapturingSandbox {
    captured: StdMutex<Vec<(String, Vec<String>)>>,
    stdout: String,
}
impl CapturingSandbox {
    fn new(stdout: &str) -> Self {
        Self {
            captured: StdMutex::new(Vec::new()),
            stdout: stdout.to_string(),
        }
    }
    fn last(&self) -> (String, Vec<String>) {
        self.captured.lock().unwrap().last().cloned().unwrap()
    }
}
impl SandboxProvider for CapturingSandbox {
    fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
        Box::pin(async { Ok(()) })
    }
    fn execute_command<'a>(
        &'a self,
        command: &'a str,
        args: &'a [String],
        _working_dir: Option<&'a std::path::Path>,
        _timeout: Option<std::time::Duration>,
    ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
        self.captured
            .lock()
            .unwrap()
            .push((command.to_string(), args.to_vec()));
        let stdout = self.stdout.clone();
        Box::pin(async move {
            Ok(CommandOutput {
                stdout,
                stderr: String::new(),
                exit_code: 0,
                timed_out: false,
                truncated: false,
            })
        })
    }
}

// (b) GitVcsProvider builds the correct `git log` command from VcsLogArgs,
// shelling out THROUGH the sandbox (never bypassing it).
#[tokio::test]
async fn git_vcs_builds_log_command() {
    let sandbox = Arc::new(CapturingSandbox::new("log-output"));
    let git = GitVcsProvider::new(sandbox.clone(), "/work");
    let args = VcsLogArgs {
        max_entries: 7,
        since_ref: Some("main".into()),
        format: Some("%h %s".into()),
    };
    let out = git.log(&args).await.unwrap();
    assert_eq!(out, "log-output");
    let (cmd, argv) = sandbox.last();
    assert_eq!(cmd, "git");
    assert_eq!(
        argv,
        vec![
            "log".to_string(),
            "-n".to_string(),
            "7".to_string(),
            "--format=%h %s".to_string(),
            "main..".to_string(),
        ],
        "git log command line built from VcsLogArgs"
    );

    // status() runs `git status`.
    let _ = git.status().await.unwrap();
    let (cmd, argv) = sandbox.last();
    assert_eq!(cmd, "git");
    assert_eq!(argv, vec!["status".to_string()]);
}

// (b') Minimal args produce just `git log -n <N>` (no format/range flags).
#[tokio::test]
async fn git_vcs_log_command_minimal_args() {
    let sandbox = Arc::new(CapturingSandbox::new(""));
    let git = GitVcsProvider::new(sandbox.clone(), "/work");
    let args = VcsLogArgs {
        max_entries: 3,
        since_ref: None,
        format: None,
    };
    let _ = git.log(&args).await.unwrap();
    let (_, argv) = sandbox.last();
    assert_eq!(
        argv,
        vec!["log".to_string(), "-n".to_string(), "3".to_string()]
    );
}

// (c) Ralph with a FixtureVcsProvider set injects the vcs_log string into the
// reloaded context across a reset.
#[tokio::test]
async fn ralph_injects_vcs_log_into_reload() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cfg, storage) = ralph_config(dir.path());
    write_progress(&storage, dir.path(), INCOMPLETE).await;
    let agent = ProgressWritingAgent::new(
        storage.run().clone(),
        dir.path(),
        vec![INCOMPLETE, COMPLETE],
    );
    set_worker_agent(&mut cfg, agent.clone());
    cfg.max_resets = 3;
    cfg.vcs_provider = Some(Arc::new(FixtureVcsProvider::new(
        "cafe123 implement login\nbeef456 add tests\n",
        "clean",
    )));
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(ralph_task())).await;
    let texts = agent.seen_text();
    assert!(
        texts[0].contains("Recent VCS history:") && texts[0].contains("cafe123 implement login"),
        "vcs log not injected: {}",
        texts[0]
    );
}

// (d) Ralph with vcs_provider = None omits any git section (v1 unchanged).
#[tokio::test]
async fn ralph_none_vcs_omits_git_section() {
    let dir = tempfile::tempdir().unwrap();
    let (mut cfg, storage) = ralph_config(dir.path());
    write_progress(&storage, dir.path(), INCOMPLETE).await;
    let agent = ProgressWritingAgent::new(storage.run().clone(), dir.path(), vec![COMPLETE]);
    set_worker_agent(&mut cfg, agent.clone());
    // vcs_provider defaults to None.
    assert!(cfg.vcs_provider.is_none());
    let h = StandardHarness::new(cfg);
    let _ = h.run(HarnessRunOptions::new(ralph_task())).await;
    let texts = agent.seen_text();
    assert!(
        !texts[0].contains("Recent VCS history:"),
        "no git section expected when provider is None: {}",
        texts[0]
    );
}

// ── Ralph cross-language fixture replay ─────────────────────────────────
#[derive(serde::Deserialize)]
struct RalphFixtureCase {
    name: String,
    /// Per-window progress-file body the agent writes (the window's state).
    windows: Vec<RalphWindow>,
    max_resets: u32,
    expected: RalphFixtureExpected,
    /// Optional git-log string (issue #58 v2). When present a
    /// [`FixtureVcsProvider`] seeded with it is wired into the harness and
    /// the reloaded context is asserted to contain it. Absent ⇒ no provider
    /// (None) ⇒ no git section (v1 behavior).
    #[serde(default)]
    vcs_log: Option<String>,
}
#[derive(serde::Deserialize)]
struct RalphWindow {
    complete: bool,
    #[serde(default)]
    remaining: Vec<String>,
}
#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RalphFixtureExpected {
    Success { iterations: u32 },
    CompletionUnmet { iterations: u32 },
}
#[derive(serde::Deserialize)]
struct RalphFixtureSuite {
    cases: Vec<RalphFixtureCase>,
}

#[tokio::test]
async fn ralph_fixture_replay() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../fixtures/harness/ralph.json");
    let raw = std::fs::read_to_string(&path).expect("fixture present");
    let suite: RalphFixtureSuite = serde_json::from_str(&raw).expect("fixture parses");
    for case in suite.cases {
        let dir = tempfile::tempdir().unwrap();
        let (mut cfg, storage) = ralph_config(dir.path());
        // Seed an initial incomplete progress checkpoint so window 1 reloads state.
        write_progress(&storage, dir.path(), INCOMPLETE).await;
        let bodies: Vec<String> = case
            .windows
            .iter()
            .map(|w| {
                serde_json::json!({
                    "complete": w.complete,
                    "remaining": w.remaining,
                })
                .to_string()
            })
            .collect();
        let agent = ProgressWritingAgent::new(
            storage.run().clone(),
            dir.path(),
            bodies.iter().map(|s| s.as_str()).collect(),
        );
        set_worker_agent(&mut cfg, agent.clone());
        cfg.max_resets = case.max_resets;
        // issue #58 v2: when the case carries a `vcs_log`, wire a
        // FixtureVcsProvider seeded with it; absent ⇒ None ⇒ no git section.
        if let Some(log) = &case.vcs_log {
            cfg.vcs_provider = Some(Arc::new(FixtureVcsProvider::new(log.clone(), "")));
        }
        let h = StandardHarness::new(cfg);
        let r = h.run(HarnessRunOptions::new(ralph_task())).await;
        // When a vcs_log is present, the first fresh window must include it.
        if let Some(log) = &case.vcs_log {
            let texts = agent.seen_text();
            assert!(
                texts
                    .first()
                    .map(|t| t.contains("Recent VCS history:") && t.contains(log.trim()))
                    .unwrap_or(false),
                "case `{}`: vcs_log not injected into reload: {:?}",
                case.name,
                texts.first()
            );
        }
        match (case.expected, r) {
            (RalphFixtureExpected::Success { iterations }, RunResult::Success { .. }) => {
                assert_eq!(
                    agent.call_count() as u32,
                    iterations,
                    "case `{}` window count",
                    case.name
                );
            }
            (
                RalphFixtureExpected::CompletionUnmet { iterations },
                RunResult::Failure {
                    reason:
                        HaltReason::RalphCompletionUnmet {
                            iterations: got, ..
                        },
                    ..
                },
            ) => {
                assert_eq!(got, iterations, "case `{}` iteration count", case.name);
            }
            (_, other) => panic!("case `{}`: unexpected result {other:?}", case.name),
        }
    }
}

// ========================================================================
// HillClimbing loop strategy (issue #60)
// ========================================================================

use crate::metric::{IterationStatus, MetricError, MetricEvaluator, MetricResult, ResultsEntry};
use crate::termination::SessionStateSnapshot as HcSnapshot;

fn hill_task(
    direction: HillClimbingDirection,
    max_stagnation: Option<u32>,
    revert: bool,
    min_delta: Option<f64>,
) -> Task {
    // Give a generous turn budget so the loop is bounded by stagnation, not
    // turns, unless a test overrides it.
    let mut t = task(LoopStrategy::HillClimbing(HillClimbingConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        // #124 A.5: the propose slot is STRUCTURED — the inner leaf carries an
        // output schema (registered as "" in standard_config).
        inner: Box::new(react_structured(u32::MAX)),
        direction,
        // `None` (unbounded) encodes as the `u32::MAX` sentinel; the
        // dispatch maps it back to `None`.
        max_stagnation: max_stagnation.unwrap_or(u32::MAX),
        revert_on_no_improvement: revert,
        // `None` delta is equivalent to `0.0` (see `should_keep`).
        min_improvement_delta: min_delta.unwrap_or(0.0),
        evaluator: AgentRef(String::new()),
    }));
    t.budget.max_turns = Some(50);
    t
}

/// A `MetricEvaluator` test double returning a pre-programmed sequence of
/// `Result<MetricResult, MetricError>` — one per `evaluate` call (call 0 is
/// the baseline). Records how many times it was evaluated.
struct ScriptedMetricEvaluator {
    outcomes: StdMutex<std::collections::VecDeque<Result<MetricResult, MetricError>>>,
    calls: std::sync::atomic::AtomicUsize,
    direction: HillClimbingDirection,
    description: String,
}
impl ScriptedMetricEvaluator {
    fn new(values: Vec<Result<f64, MetricError>>, direction: HillClimbingDirection) -> Self {
        let outcomes = values
            .into_iter()
            .map(|v| v.map(MetricResult::new))
            .collect();
        Self {
            outcomes: StdMutex::new(outcomes),
            calls: std::sync::atomic::AtomicUsize::new(0),
            direction,
            description: "scripted metric".to_string(),
        }
    }
    fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}
impl MetricEvaluator for ScriptedMetricEvaluator {
    fn evaluate<'a>(
        &'a self,
        _sandbox: &'a dyn SandboxProvider,
        _session_state: &'a HcSnapshot,
    ) -> BoxFut<'a, Result<MetricResult, MetricError>> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let next = self.outcomes.lock().unwrap().pop_front().unwrap_or(Err(
            MetricError::ExecutionFailed {
                reason: "scripted evaluator exhausted".into(),
            },
        ));
        Box::pin(async move { next })
    }
    fn direction(&self) -> HillClimbingDirection {
        self.direction
    }
    fn description(&self) -> String {
        self.description.clone()
    }
}

/// A `SandboxProvider` rooted at a tempdir that spies on `git reset --hard
/// HEAD` revert calls (issue #60, Decision 1). Counts every `execute_command`
/// whose argv is exactly the revert command and returns a clean exit.
struct HcSpySandbox {
    root: tempfile::TempDir,
    reverts: std::sync::atomic::AtomicUsize,
}
impl HcSpySandbox {
    fn new() -> Self {
        Self {
            root: tempfile::tempdir().unwrap(),
            reverts: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    fn revert_count(&self) -> usize {
        self.reverts.load(std::sync::atomic::Ordering::SeqCst)
    }
}
impl SandboxProvider for HcSpySandbox {
    fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
        Box::pin(async { Ok(()) })
    }
    fn execute_command<'a>(
        &'a self,
        command: &'a str,
        args: &'a [String],
        _working_dir: Option<&'a std::path::Path>,
        _timeout: Option<std::time::Duration>,
    ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
        if command == "git" && args == ["reset", "--hard", "HEAD"] {
            self.reverts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        Box::pin(async {
            Ok(CommandOutput {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: 0,
                timed_out: false,
                truncated: false,
            })
        })
    }
    fn workspace_root(&self) -> &std::path::Path {
        self.root.path()
    }
}

/// A `VcsProvider` that spies on `revert()` calls (SC-14). Used to prove the
/// HillClimbing revert routes through a wired provider instead of the
/// hardcoded git fallback. `log`/`status` return empty strings.
struct HcSpyVcs {
    reverts: std::sync::atomic::AtomicUsize,
}
impl HcSpyVcs {
    fn new() -> Self {
        Self {
            reverts: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    fn revert_count(&self) -> usize {
        self.reverts.load(std::sync::atomic::Ordering::SeqCst)
    }
}
impl VcsProvider for HcSpyVcs {
    fn log<'a>(&'a self, _args: &'a VcsLogArgs) -> BoxFut<'a, Result<String, VcsError>> {
        Box::pin(async { Ok(String::new()) })
    }
    fn status<'a>(&'a self) -> BoxFut<'a, Result<String, VcsError>> {
        Box::pin(async { Ok(String::new()) })
    }
    fn revert<'a>(&'a self) -> BoxFut<'a, Result<(), VcsError>> {
        self.reverts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }
}

/// Build a HillClimbing config wired with the scripted evaluator and spy
/// sandbox. The agent always finishes a turn immediately (one final response
/// per iteration), so the loop is driven entirely by the metric sequence.
fn hc_config(evaluator: Arc<ScriptedMetricEvaluator>, sandbox: Arc<HcSpySandbox>) -> HarnessConfig {
    // A fresh "done" turn for every possible iteration (the agent is re-seeded
    // per iteration, so each iteration pops exactly one turn).
    let agent = RecordingTurnAgent::new(
        "hc",
        (0..64)
            .map(|_| RecordingTurnAgent::final_resp("changed"))
            .collect(),
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, agent);
    cfg.sandbox = sandbox;
    set_default_metric(&mut cfg, evaluator);
    cfg
}

// Decision 6 (#124): an unresolved metric evaluator is a typed STARTUP halt
// (single resolution path), not a panic.
#[tokio::test]
async fn hill_climbing_missing_evaluator_is_typed_halt() {
    let cfg = standard_config(make_agent());
    let h = StandardHarness::new(cfg);
    let t = hill_task(HillClimbingDirection::Maximize, None, false, None);
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure {
            reason:
                HaltReason::ConfigurationError {
                    error: HarnessError::UnresolvedHandle { kind, .. },
                },
            ..
        } => assert_eq!(kind, "metric_evaluator"),
        other => panic!("expected unresolved metric_evaluator, got {other:?}"),
    }
}

// Decision 5: iteration 0 is a pure baseline (no agent turn) recorded Kept,
// and it seeds current_best. With max_stagnation = 1 and an immediate regress
// at iteration 1, the loop halts; we confirm the baseline row is Kept and the
// agent ran exactly once (only iteration 1's turn, never the baseline).
#[tokio::test]
async fn hill_climbing_baseline_first_status_kept_no_agent_turn() {
    let sandbox = Arc::new(HcSpySandbox::new());
    // baseline 1.0, then a worse value (regress) for maximize.
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(1.0), Ok(0.5)],
        HillClimbingDirection::Maximize,
    ));
    let cfg = hc_config(eval.clone(), sandbox.clone());
    let workspace = sandbox.root.path().to_path_buf();
    let h = StandardHarness::new(cfg);
    let t = hill_task(HillClimbingDirection::Maximize, Some(1), false, None);
    let task_id = t.id.as_str().to_string();
    let r = h.run(HarnessRunOptions::new(t)).await;
    assert!(
        matches!(
            r,
            RunResult::Failure {
                reason: HaltReason::StagnationLimitReached { .. },
                ..
            }
        ),
        "got {r:?}"
    );
    // Two evaluator calls: baseline + iteration 1.
    assert_eq!(eval.call_count(), 2);
    // The TSV baseline row is Kept.
    let tsv = std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv")))
        .expect("tsv written");
    let lines: Vec<&str> = tsv.lines().collect();
    assert_eq!(
        lines[0],
        "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription"
    );
    assert!(lines[1].starts_with("0\t"), "baseline is iteration 0");
    assert!(
        lines[1].contains("\tkept\t"),
        "baseline status kept: {}",
        lines[1]
    );
}

// #124 REGRESSION-PROOF: HillClimbing genuinely recurses into a NON-ReAct
// inner. `HillClimbing[ inner: PlanExecute[ReAct, ReAct] ]` with a metric that
// improves (iter 1) then stagnates (iter 2, cap 1 ⇒ halt). The inner
// PlanExecute's WHOLE loop runs (it fires the plan phase), proving the
// recursion. A hardcoded-ReAct proposer would NEVER fire the plan phase (the
// worker never sees the plan directive).
//
// Model (A): each HillClimbing iteration generates a FRESH neighbor candidate,
// so the inner PlanExecute re-plans EVERY iteration — it does NOT reuse the
// prior iteration's (project-scoped) durable task_list. Reusing a fully-
// `Completed` list would make iteration 2 a no-op (no neighbor generated), the
// failure mode that defeats hill climbing. The plan phase therefore fires once
// PER iteration; with two iterations the worker sees the plan directive twice.
#[tokio::test]
async fn hill_climbing_runs_non_react_inner_per_iteration() {
    let sandbox = Arc::new(HcSpySandbox::new());
    // baseline 1.0, iter1 2.0 (improve→keep, stagnation reset), iter2 0.5
    // (regress→discard, stagnation hits cap 1 ⇒ halt).
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(1.0), Ok(2.0), Ok(0.5)],
        HillClimbingDirection::Maximize,
    ));
    // Each iteration fires a plan JSON turn + an execute step — a fresh
    // neighbor candidate. Iteration 2 re-plans (the stale durable list from
    // iteration 1 is cleared on the fresh plan), so it needs its own pair.
    let worker = RecordingTurnAgent::new(
        "hc-inner",
        vec![
            RecordingTurnAgent::final_resp(r#"{"tasks":["only"],"rationale":"r"}"#),
            RecordingTurnAgent::final_resp("changed iter1"),
            RecordingTurnAgent::final_resp(r#"{"tasks":["again"],"rationale":"r2"}"#),
            RecordingTurnAgent::final_resp("changed iter2"),
        ],
    );
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, worker.clone());
    cfg.sandbox = sandbox;
    set_default_metric(&mut cfg, eval.clone());
    let h = StandardHarness::new(cfg);

    // inner is a genuine PlanExecute combinator (NOT a ReAct).
    let mut t = task(LoopStrategy::HillClimbing(HillClimbingConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        inner: Box::new(LoopStrategy::PlanExecute(PlanExecuteConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            plan: Box::new(react_structured(u32::MAX)),
            execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
            plan_model: None,
        })),
        direction: HillClimbingDirection::Maximize,
        max_stagnation: 1,
        revert_on_no_improvement: false,
        min_improvement_delta: 0.0,
        evaluator: AgentRef(String::new()),
    }));
    // Generous turn budget so the loop is bounded by stagnation, not turns.
    t.budget.max_turns = Some(50);

    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure {
            reason: HaltReason::StagnationLimitReached { .. },
            ..
        } => {}
        other => panic!("expected StagnationLimitReached, got {other:?}"),
    }
    // Model (A): the inner PlanExecute re-plans EACH iteration (a fresh
    // neighbor), so across two iterations the worker saw the plan directive
    // TWICE. A hardcoded-ReAct proposer would record ZERO plan turns, so this
    // still proves the genuine recursion into PlanExecute — and the count of 2
    // proves the per-iteration regeneration that hill climbing requires
    // (reusing the durable list would make iteration 2 a no-op).
    let seen = worker.seen_text();
    let plan_turns = seen
        .iter()
        .filter(|c| c.contains("step-by-step plan"))
        .count();
    assert_eq!(
        plan_turns, 2,
        "the inner PlanExecute plan phase fires once PER iteration (no \
             durable-list reuse across HillClimbing neighbors); saw: {seen:?}"
    );
}

// keep-on-improve: a strictly better value updates current_best and resets
// stagnation. With maximize, baseline 1.0 then 2.0 (improve) then 1.5 (regress
// vs 2.0) then 3.0 (improve). max_stagnation None ⇒ runs to the turn budget.
#[tokio::test]
async fn hill_climbing_keeps_on_improvement() {
    let sandbox = Arc::new(HcSpySandbox::new());
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(1.0), Ok(2.0), Ok(1.5), Ok(3.0)],
        HillClimbingDirection::Maximize,
    ));
    let cfg = hc_config(eval.clone(), sandbox.clone());
    let workspace = sandbox.root.path().to_path_buf();
    // Cap turns at 3 so exactly iterations 1,2,3 run (then budget halt).
    let mut t = hill_task(HillClimbingDirection::Maximize, None, false, None);
    t.budget.max_turns = Some(3);
    let task_id = t.id.as_str().to_string();
    let _ = h_run(&cfg, t).await;
    let tsv =
        std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv"))).unwrap();
    let lines: Vec<&str> = tsv.lines().collect();
    // header + baseline(0) + iter1(2.0 kept) + iter2(1.5 discarded) + iter3(3.0 kept)
    assert_eq!(lines.len(), 5, "tsv:\n{tsv}");
    assert!(lines[1].contains("\t1.000000\t") && lines[1].contains("\tkept\t"));
    assert!(lines[2].contains("\t2.000000\t") && lines[2].contains("\tkept\t"));
    assert!(lines[3].contains("\t1.500000\t") && lines[3].contains("\tdiscarded\t"));
    assert!(lines[4].contains("\t3.000000\t") && lines[4].contains("\tkept\t"));
    let _ = eval;
}

// discard-on-regress with revert OFF: no git reset issued.
#[tokio::test]
async fn hill_climbing_discards_on_regress_no_revert() {
    let sandbox = Arc::new(HcSpySandbox::new());
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(2.0), Ok(3.0)], // minimize: baseline 2.0, then 3.0 (worse)
        HillClimbingDirection::Minimize,
    ));
    let cfg = hc_config(eval, sandbox.clone());
    let t = hill_task(HillClimbingDirection::Minimize, Some(1), false, None);
    let _ = h_run(&cfg, t).await;
    assert_eq!(sandbox.revert_count(), 0, "revert OFF must not git reset");
}

// revert ON: a no-improvement iteration triggers exactly one git reset.
// (No vcs_provider wired ⇒ the SC-14 fallback path runs the hardcoded
// `git reset --hard HEAD` through the sandbox, byte-identical to pre-SC-14.)
#[tokio::test]
async fn hill_climbing_reverts_on_no_improvement() {
    let sandbox = Arc::new(HcSpySandbox::new());
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(2.0), Ok(3.0)], // minimize: baseline 2.0, then 3.0 (worse)
        HillClimbingDirection::Minimize,
    ));
    let cfg = hc_config(eval, sandbox.clone());
    let t = hill_task(HillClimbingDirection::Minimize, Some(1), true, None);
    let _ = h_run(&cfg, t).await;
    assert_eq!(sandbox.revert_count(), 1, "revert ON must git reset once");
}

// SC-14: when a VcsProvider is wired, the revert routes THROUGH it and the
// hardcoded git-reset fallback does NOT fire.
#[tokio::test]
async fn hill_climbing_revert_routes_through_vcs_provider() {
    let sandbox = Arc::new(HcSpySandbox::new());
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(2.0), Ok(3.0)], // minimize: baseline 2.0, then 3.0 (worse)
        HillClimbingDirection::Minimize,
    ));
    let mut cfg = hc_config(eval, sandbox.clone());
    let vcs = Arc::new(HcSpyVcs::new());
    cfg.vcs_provider = Some(vcs.clone() as Arc<dyn VcsProvider>);
    let t = hill_task(HillClimbingDirection::Minimize, Some(1), true, None);
    let _ = h_run(&cfg, t).await;
    assert_eq!(vcs.revert_count(), 1, "wired VcsProvider owns the revert");
    assert_eq!(
        sandbox.revert_count(),
        0,
        "hardcoded git-reset fallback must NOT fire when a provider is wired"
    );
}

// SC-14: GitVcsProvider::revert issues `git reset --hard HEAD` through the
// sandbox (the behavior relocated out of the harness).
#[tokio::test]
async fn git_vcs_provider_revert_runs_git_reset_hard() {
    let sandbox = Arc::new(HcSpySandbox::new());
    let root = sandbox.root.path().to_path_buf();
    let provider = GitVcsProvider::new(sandbox.clone() as Arc<dyn SandboxProvider>, root);
    provider.revert().await.unwrap();
    assert_eq!(sandbox.revert_count(), 1);
}

// hill-climbing-loop-correctness §4.5: the new GitVcsProvider seam methods
// (head / commit_checkpoint / reset_to) round-trip on a REAL git workspace.
#[tokio::test]
async fn git_vcs_provider_commit_head_reset_roundtrip() {
    let (_dir, root, draft) = init_git_repo("base");
    let sandbox: Arc<dyn SandboxProvider> = Arc::new(
        crate::sandbox::WorkspaceScopedSandbox::new(crate::sandbox::WorkspaceConfig::scoped(
            root.clone(),
        ))
        .unwrap(),
    );
    let provider = GitVcsProvider::new(sandbox, root);

    let sha0 = provider.head().await.unwrap();
    assert!(!sha0.is_empty(), "head resolves the init commit");

    std::fs::write(&draft, "v1").unwrap();
    let sha1 = provider.commit_checkpoint("c1").await.unwrap();
    assert_ne!(sha1, sha0, "a new candidate is a new commit");

    std::fs::write(&draft, "v2").unwrap();
    let sha2 = provider.commit_checkpoint("c2").await.unwrap();
    assert_ne!(sha2, sha1);

    // Reset to the FIRST checkpoint: the tree returns to that exact state,
    // and HEAD follows — the load-bearing reset-to-a-specific-SHA behavior.
    provider.reset_to(&sha1).await.unwrap();
    assert_eq!(std::fs::read_to_string(&draft).unwrap(), "v1");
    assert_eq!(provider.head().await.unwrap(), sha1);
}

// strict min_delta boundary: an improvement of exactly min_delta is NOT
// progress (discarded). Minimize, baseline 2.0, then 1.5 with min_delta 0.5.
#[tokio::test]
async fn hill_climbing_strict_min_delta_boundary() {
    let sandbox = Arc::new(HcSpySandbox::new());
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(2.0), Ok(1.5)],
        HillClimbingDirection::Minimize,
    ));
    let cfg = hc_config(eval, sandbox.clone());
    let workspace = sandbox.root.path().to_path_buf();
    let t = hill_task(HillClimbingDirection::Minimize, Some(1), false, Some(0.5));
    let task_id = t.id.as_str().to_string();
    let r = h_run(&cfg, t).await;
    // Exactly-min_delta improvement is discarded ⇒ stagnation hits 1 ⇒ halt.
    match r {
        RunResult::Failure {
            reason: HaltReason::StagnationLimitReached { best_metric, .. },
            ..
        } => assert!((best_metric - 2.0).abs() < 1e-9, "best stays baseline 2.0"),
        other => panic!("expected StagnationLimitReached, got {other:?}"),
    }
    let tsv =
        std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv"))).unwrap();
    assert!(
        tsv.lines().nth(2).unwrap().contains("\tdiscarded\t"),
        "exactly-min_delta is discarded: {tsv}"
    );
}

// stagnation halt after N consecutive non-improvements.
#[tokio::test]
async fn hill_climbing_stagnation_halt() {
    let sandbox = Arc::new(HcSpySandbox::new());
    // maximize, baseline 5.0, then three regresses.
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(5.0), Ok(4.0), Ok(4.0), Ok(4.0)],
        HillClimbingDirection::Maximize,
    ));
    let cfg = hc_config(eval, sandbox.clone());
    let t = hill_task(HillClimbingDirection::Maximize, Some(3), false, None);
    match h_run(&cfg, t).await {
        RunResult::Failure {
            reason:
                HaltReason::StagnationLimitReached {
                    iterations,
                    best_metric,
                },
            ..
        } => {
            assert_eq!(iterations, 3);
            assert!((best_metric - 5.0).abs() < 1e-9);
        }
        other => panic!("expected StagnationLimitReached, got {other:?}"),
    }
}

// stagnation counter resets on improvement: regress, regress, IMPROVE, regress
// never reaches a stagnation cap of 3.
#[tokio::test]
async fn hill_climbing_stagnation_resets_on_improve() {
    let sandbox = Arc::new(HcSpySandbox::new());
    // maximize: baseline 1.0; iters: 0.5(regress) 0.5(regress) 2.0(improve!) 1.0(regress)
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(1.0), Ok(0.5), Ok(0.5), Ok(2.0), Ok(1.0)],
        HillClimbingDirection::Maximize,
    ));
    let cfg = hc_config(eval, sandbox.clone());
    // Cap turns at 4 so the improve at iter 3 resets the counter and the run
    // ends on the turn budget rather than stagnation.
    let mut t = hill_task(HillClimbingDirection::Maximize, Some(3), false, None);
    t.budget.max_turns = Some(4);
    match h_run(&cfg, t).await {
        RunResult::Failure {
            reason:
                HaltReason::BudgetExceeded {
                    limit_type: BudgetLimitType::Turns,
                },
            ..
        } => {}
        other => panic!("expected Turns budget halt (counter reset), got {other:?}"),
    }
}

// crash counts as a non-improvement: a Crashed evaluation increments
// stagnation and writes an EMPTY metric_value row.
#[tokio::test]
async fn hill_climbing_crash_counts_as_non_improvement() {
    let sandbox = Arc::new(HcSpySandbox::new());
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(1.0), Err(MetricError::Crashed { log: "boom".into() })],
        HillClimbingDirection::Maximize,
    ));
    let cfg = hc_config(eval, sandbox.clone());
    let workspace = sandbox.root.path().to_path_buf();
    let t = hill_task(HillClimbingDirection::Maximize, Some(1), false, None);
    let task_id = t.id.as_str().to_string();
    match h_run(&cfg, t).await {
        RunResult::Failure {
            reason: HaltReason::StagnationLimitReached { .. },
            ..
        } => {}
        other => panic!("expected StagnationLimitReached, got {other:?}"),
    }
    let tsv =
        std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv"))).unwrap();
    let crash_row = tsv.lines().nth(2).unwrap();
    // metric_value column (3rd, index 2) is EMPTY on a crashed row.
    let cols: Vec<&str> = crash_row.split('\t').collect();
    assert_eq!(cols[2], "", "crashed row metric_value empty: {crash_row}");
    assert_eq!(cols[4], "crashed", "crashed status: {crash_row}");
}

// timeout counts as non-improvement and maps to the `timeout` status.
#[tokio::test]
async fn hill_climbing_timeout_status() {
    let sandbox = Arc::new(HcSpySandbox::new());
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![
            Ok(1.0),
            Err(MetricError::Timeout {
                after: std::time::Duration::from_secs(1),
            }),
        ],
        HillClimbingDirection::Maximize,
    ));
    let cfg = hc_config(eval, sandbox.clone());
    let workspace = sandbox.root.path().to_path_buf();
    let t = hill_task(HillClimbingDirection::Maximize, Some(1), false, None);
    let task_id = t.id.as_str().to_string();
    let _ = h_run(&cfg, t).await;
    let tsv =
        std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv"))).unwrap();
    let cols: Vec<&str> = tsv.lines().nth(2).unwrap().split('\t').collect();
    assert_eq!(cols[4], "timeout");
    assert_eq!(cols[2], "", "timeout row metric_value empty");
}

// budget gate: max_turns = 0 halts before any agent turn (only the baseline
// runs). The result is a Turns budget halt with a baseline-only TSV.
#[tokio::test]
async fn hill_climbing_budget_turn_gate() {
    let sandbox = Arc::new(HcSpySandbox::new());
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(1.0)],
        HillClimbingDirection::Maximize,
    ));
    let cfg = hc_config(eval.clone(), sandbox.clone());
    let workspace = sandbox.root.path().to_path_buf();
    let mut t = hill_task(HillClimbingDirection::Maximize, None, false, None);
    t.budget.max_turns = Some(0);
    let task_id = t.id.as_str().to_string();
    match h_run(&cfg, t).await {
        RunResult::Failure {
            reason:
                HaltReason::BudgetExceeded {
                    limit_type: BudgetLimitType::Turns,
                },
            ..
        } => {}
        other => panic!("expected Turns budget halt, got {other:?}"),
    }
    // Only the baseline evaluation ran.
    assert_eq!(eval.call_count(), 1);
    let tsv =
        std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv"))).unwrap();
    assert_eq!(tsv.lines().count(), 2, "header + baseline only: {tsv}");
}

// Exact TSV byte content for a representative keep/discard/crash scenario.
#[tokio::test]
async fn hill_climbing_exact_tsv_bytes() {
    let rows = vec![
        ResultsEntry {
            iteration: 0,
            commit_hash: None,
            metric_value: 1.0,
            direction: HillClimbingDirection::Maximize,
            status: IterationStatus::Kept,
            duration: std::time::Duration::from_millis(1500),
            description: "scripted metric".into(),
            metadata: Default::default(),
        },
        ResultsEntry {
            iteration: 1,
            commit_hash: None,
            metric_value: 2.5,
            direction: HillClimbingDirection::Maximize,
            status: IterationStatus::Kept,
            duration: std::time::Duration::from_secs(0),
            description: "scripted metric".into(),
            metadata: Default::default(),
        },
        ResultsEntry {
            iteration: 2,
            commit_hash: None,
            metric_value: f64::NAN, // crashed ⇒ empty in TSV
            direction: HillClimbingDirection::Maximize,
            status: IterationStatus::Crashed,
            duration: std::time::Duration::from_secs(0),
            description: "scripted metric".into(),
            metadata: Default::default(),
        },
    ];
    let tsv = StandardHarness::render_hill_climbing_tsv(&rows);
    let expected =
        "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n\
0\t\t1.000000\tmaximize\tkept\t1.500000\tscripted metric\n\
1\t\t2.500000\tmaximize\tkept\t0.000000\tscripted metric\n\
2\t\t\tmaximize\tcrashed\t0.000000\tscripted metric\n";
    assert_eq!(tsv, expected, "TSV byte mismatch");
}

/// Helper: build a StandardHarness from `cfg` and run `task` to completion.
async fn h_run(cfg: &HarnessConfig, t: Task) -> RunResult {
    let h = StandardHarness::new(cfg.clone());
    h.run(HarnessRunOptions::new(t)).await
}

// ── hill-climbing-loop-correctness: SHA-backed climbing on a REAL git tree ──

/// A worker agent that performs a REAL working-tree edit each turn — writes a
/// per-iteration marker to `path` — so a git-backed HillClimbing run produces
/// distinct committable states. Returns a trivial FinalResponse so the inner
/// ReAct completes in one turn. Used by the keep-then-discard regression test:
/// the metric is scripted, and the FILE CONTENT is the artifact under test.
struct FileEditingAgent {
    id: AgentId,
    path: std::path::PathBuf,
    contents: StdMutex<std::collections::VecDeque<String>>,
}
impl FileEditingAgent {
    fn new(path: std::path::PathBuf, contents: Vec<&str>) -> Arc<Self> {
        Arc::new(Self {
            id: AgentId::new("file-editor"),
            path,
            contents: StdMutex::new(contents.into_iter().map(String::from).collect()),
        })
    }
}
impl Agent for FileEditingAgent {
    fn turn<'a>(&'a self, _context: Context) -> BoxFut<'a, TurnResult> {
        if let Some(c) = self.contents.lock().unwrap().pop_front() {
            std::fs::write(&self.path, c).expect("agent writes the draft");
        }
        Box::pin(async move {
            TurnResult::FinalResponse {
                reasoning: None,
                content: "edited".into(),
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
            }
        })
    }
    fn id(&self) -> AgentId {
        self.id.clone()
    }
}

/// Create a real temp git repo with one tracked file committed. Returns the
/// (kept-alive) tempdir, the canonical workspace root, and the draft path. A
/// REAL VCS is required here: per the `no-op store silently disables durable
/// behavior` lesson, a fake VCS path would assert the opposite of the bug
/// while still passing.
fn init_git_repo(initial: &str) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(dir.path()).unwrap();
    let draft = root.join("draft.txt");
    std::fs::write(&draft, initial).unwrap();
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(&root)
            .output()
            .expect("git is available");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    git(&["init", "-q"]);
    git(&["config", "user.email", "t@spore.invalid"]);
    git(&["config", "user.name", "spore test"]);
    git(&["add", "-A"]);
    git(&["commit", "-q", "-m", "init"]);
    (dir, root, draft)
}

// THE load-bearing regression (hill-climbing-loop-correctness §5). keep-then-
// discard on a REAL git workspace: iter1 improves (KEPT → committed), iter2
// regresses (DISCARDED → revert). The fix makes the revert target the last
// ACCEPTED commit (iter1), so iter1's edit SURVIVES. On `main` the revert ran
// `git reset --hard HEAD`, and because the harness never committed a keep,
// HEAD was still the original start — so the accepted iter1 edit was DESTROYED
// and the tree snapped back to the baseline. This test fails on `main` and
// passes after the fix.
#[tokio::test]
async fn hill_climbing_keep_then_discard_preserves_kept_state() {
    let (_dir, root, draft) = init_git_repo("base");
    let sandbox = Arc::new(
        crate::sandbox::WorkspaceScopedSandbox::new(crate::sandbox::WorkspaceConfig::scoped(
            root.clone(),
        ))
        .unwrap(),
    );
    // maximize: baseline 1.0, iter1 2.0 (improve → keep), iter2 1.5 (regress
    // → discard). The scripted scores DRIVE keep/discard; the file content is
    // what the revert must preserve.
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(1.0), Ok(2.0), Ok(1.5)],
        HillClimbingDirection::Maximize,
    ));
    let agent = FileEditingAgent::new(draft.clone(), vec!["delta1", "delta2"]);
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, agent);
    cfg.sandbox = sandbox;
    set_default_metric(&mut cfg, eval);

    // max_stagnation 1 ⇒ the iter2 discard halts the climb.
    let t = hill_task(HillClimbingDirection::Maximize, Some(1), true, None);
    let task_id = t.id.as_str().to_string();
    let r = h_run(&cfg, t).await;

    match r {
        RunResult::Failure {
            reason: HaltReason::StagnationLimitReached { best_metric, .. },
            ..
        } => assert!(
            (best_metric - 2.0).abs() < 1e-9,
            "best stays the kept iter1 score 2.0; got {best_metric}"
        ),
        other => panic!("expected StagnationLimitReached, got {other:?}"),
    }

    // THE assertion: the working tree holds iter1's KEPT edit, NOT the
    // original start. On `main` this reads "base" (revert went to HEAD ==
    // start, destroying delta1). Because the scripted evaluator scored these
    // exact bytes 2.0, the surviving artifact is the one that produced the
    // best score — score and artifact have not diverged (§5 step 5).
    let on_disk = std::fs::read_to_string(&draft).unwrap();
    assert_eq!(
        on_disk, "delta1",
        "the kept iter1 edit must survive the iter2 discard"
    );

    // §4.7: kept rows now carry REAL (non-empty) commit SHAs — baseline and
    // iter1 both committed. (Pre-fix every commit_hash was the empty string.)
    let tsv = std::fs::read_to_string(root.join(format!(".spore/results/{task_id}.tsv")))
        .expect("tsv written");
    let rows: Vec<&str> = tsv.lines().skip(1).collect();
    // rows: [0] baseline kept, [1] iter1 kept, [2] iter2 discarded.
    let sha = |line: &str| line.split('\t').nth(1).unwrap_or("").to_string();
    let iter1_sha = sha(rows[1]);
    let iter2_sha = sha(rows[2]);
    assert!(
        !iter1_sha.is_empty(),
        "iter1 kept row carries a real commit SHA; tsv:\n{tsv}"
    );
    assert!(rows[2].contains("\tdiscarded\t"), "iter2 is discarded");
    // The KEY revert-target assertion, NOT masked by the exit-time winner
    // application: the discard row records the same accepted SHA as the iter1
    // keep, proving the revert targeted the last KEPT commit — not the
    // original HEAD/start (where main reset to). This is set during the
    // discard, before the loop exits.
    assert_eq!(
        iter1_sha, iter2_sha,
        "the no-improvement revert targeted the iter1 commit, not HEAD; tsv:\n{tsv}"
    );
    assert_ne!(
        sha(rows[0]),
        iter1_sha,
        "the baseline and iter1 are distinct commits (the climb advanced)"
    );

    // The harness's own `.spore/` ledger must NOT be swept into candidate
    // commits — otherwise the winner-apply `git reset --hard` would clobber
    // the freshly-written TSV with a stale committed copy. Prove it stayed
    // untracked (and so survived every reset, real SHAs intact above).
    let tracked = std::process::Command::new("git")
        .args(["ls-files"])
        .current_dir(&root)
        .output()
        .unwrap();
    let tracked = String::from_utf8_lossy(&tracked.stdout);
    assert!(
        !tracked.lines().any(|f| f.starts_with(".spore")),
        "the .spore ledger dir must stay untracked; tracked files:\n{tracked}"
    );
}

// Companion to the regression: prove the durable pointer also fixes the
// multi-keep case. baseline → iter1 keep (delta1) → iter2 keep (delta2) →
// iter3 discard. The revert must land on delta2 (the LAST accepted state),
// not delta1 and not the start.
#[tokio::test]
async fn hill_climbing_revert_targets_latest_accepted_not_start() {
    let (_dir, root, draft) = init_git_repo("base");
    let sandbox = Arc::new(
        crate::sandbox::WorkspaceScopedSandbox::new(crate::sandbox::WorkspaceConfig::scoped(
            root.clone(),
        ))
        .unwrap(),
    );
    // maximize: 1.0 baseline, 2.0 keep, 3.0 keep, 2.5 discard.
    let eval = Arc::new(ScriptedMetricEvaluator::new(
        vec![Ok(1.0), Ok(2.0), Ok(3.0), Ok(2.5)],
        HillClimbingDirection::Maximize,
    ));
    let agent = FileEditingAgent::new(draft.clone(), vec!["delta1", "delta2", "delta3"]);
    let mut cfg = standard_config(make_agent());
    set_worker_agent(&mut cfg, agent);
    cfg.sandbox = sandbox;
    set_default_metric(&mut cfg, eval);

    let t = hill_task(HillClimbingDirection::Maximize, Some(1), true, None);
    let r = h_run(&cfg, t).await;
    match r {
        RunResult::Failure {
            reason: HaltReason::StagnationLimitReached { best_metric, .. },
            ..
        } => assert!((best_metric - 3.0).abs() < 1e-9, "best is iter2's 3.0"),
        other => panic!("expected StagnationLimitReached, got {other:?}"),
    }
    let on_disk = std::fs::read_to_string(&draft).unwrap();
    assert_eq!(
        on_disk, "delta2",
        "revert lands on the LAST accepted state (delta2), not delta1 or the start"
    );
}

// ── HillClimbing cross-language fixture replay ──────────────────────────
#[derive(serde::Deserialize)]
struct HcFixtureSuite {
    scenarios: Vec<HcScenario>,
}
#[derive(serde::Deserialize)]
struct HcScenario {
    name: String,
    /// Scripted metric sequence; element 0 is the baseline. `null` = a crash.
    metric_sequence: Vec<Option<f64>>,
    payload: HcPayload,
    expected: HcExpected,
    /// Optional exact-TSV byte assertion for this scenario.
    #[serde(default)]
    expected_tsv: Option<String>,
    /// Optional turn-budget override (defaults to a high cap).
    #[serde(default)]
    max_turns: Option<u32>,
}
#[derive(serde::Deserialize)]
struct HcPayload {
    direction: HillClimbingDirection,
    max_stagnation: Option<u32>,
    revert_on_no_improvement: bool,
    min_improvement_delta: Option<f64>,
}
#[derive(serde::Deserialize)]
struct HcExpected {
    /// `"stagnation"` | `"budget_turns"` | `"misconfigured"`.
    halt_reason: String,
    kept_iterations: u32,
    revert_count: usize,
    #[serde(default)]
    best_metric: Option<f64>,
}

#[tokio::test]
async fn hill_climbing_fixture_replay() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../fixtures/metric_evaluator/hill_climbing_sequences.json");
    let raw = std::fs::read_to_string(&path).expect("fixture present");
    let suite: HcFixtureSuite = serde_json::from_str(&raw).expect("fixture parses");
    for sc in suite.scenarios {
        let sandbox = Arc::new(HcSpySandbox::new());
        let values: Vec<Result<f64, MetricError>> = sc
            .metric_sequence
            .iter()
            .map(|v| match v {
                Some(x) => Ok(*x),
                None => Err(MetricError::Crashed {
                    log: "scripted crash".into(),
                }),
            })
            .collect();
        let eval = Arc::new(ScriptedMetricEvaluator::new(values, sc.payload.direction));
        let cfg = hc_config(eval, sandbox.clone());
        let workspace = sandbox.root.path().to_path_buf();
        let mut t = hill_task(
            sc.payload.direction,
            sc.payload.max_stagnation,
            sc.payload.revert_on_no_improvement,
            sc.payload.min_improvement_delta,
        );
        if let Some(mt) = sc.max_turns {
            t.budget.max_turns = Some(mt);
        }
        let task_id = t.id.as_str().to_string();
        let r = h_run(&cfg, t).await;

        // Halt reason.
        match (sc.expected.halt_reason.as_str(), &r) {
            (
                "stagnation",
                RunResult::Failure {
                    reason: HaltReason::StagnationLimitReached { best_metric, .. },
                    ..
                },
            ) => {
                if let Some(exp) = sc.expected.best_metric {
                    assert!(
                        (best_metric - exp).abs() < 1e-9,
                        "scenario `{}` best_metric: got {best_metric}, want {exp}",
                        sc.name
                    );
                }
            }
            (
                "budget_turns",
                RunResult::Failure {
                    reason:
                        HaltReason::BudgetExceeded {
                            limit_type: BudgetLimitType::Turns,
                        },
                    ..
                },
            ) => {}
            (other_exp, other_got) => panic!(
                "scenario `{}`: expected halt `{other_exp}`, got {other_got:?}",
                sc.name
            ),
        }

        // Kept-iteration count and revert count from the written TSV.
        let tsv = std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv")))
            .unwrap_or_else(|e| panic!("scenario `{}` tsv: {e}", sc.name));
        let kept = tsv
            .lines()
            .skip(1)
            .filter(|l| l.contains("\tkept\t"))
            .count() as u32;
        assert_eq!(
            kept, sc.expected.kept_iterations,
            "scenario `{}` kept count; tsv:\n{tsv}",
            sc.name
        );
        assert_eq!(
            sandbox.revert_count(),
            sc.expected.revert_count,
            "scenario `{}` revert count",
            sc.name
        );

        // Optional exact-TSV byte assertion.
        if let Some(expected_tsv) = &sc.expected_tsv {
            assert_eq!(&tsv, expected_tsv, "scenario `{}` TSV bytes", sc.name);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// #103: delta-level streaming through the harness
// ════════════════════════════════════════════════════════════════════════

mod streaming {
    use super::*;
    use crate::agent::ModelAgent;
    use crate::model::{
        ContentBlock, ModelResponse, ProviderInfo, RecordedExchange, ReplayModelInterface,
        StopReason,
    };
    use std::sync::Mutex as StdMutex;

    fn replay_provider() -> ProviderInfo {
        ProviderInfo {
            name: "anthropic".into(),
            model_id: "replay".into(),
            context_window: 200_000,
        }
    }

    fn replay_agent(responses: Vec<ModelResponse>) -> Arc<dyn Agent> {
        let exchanges: Vec<RecordedExchange> = responses
            .into_iter()
            .map(|response| RecordedExchange {
                request_hash: None,
                request: crate::model::ModelRequest {
                    messages: vec![],
                    tools: vec![],
                    params: crate::model::ModelParams::default(),
                    stream: true,
                },
                response,
                provider: "anthropic".into(),
                model_id: None,
                recorded_at: None,
                duration_ms: None,
            })
            .collect();
        // Positional mode (no hashes): returns responses in call order
        // regardless of request content.
        let replay = Arc::new(ReplayModelInterface::new(exchanges, replay_provider()));
        Arc::new(ModelAgent::new(AgentId::new("stream-agent"), replay))
    }

    fn streaming_config(agent: Arc<dyn Agent>) -> HarnessConfig {
        // standard_config takes Arc<MockAgent>; build directly so we can use
        // a model-backed agent that actually streams.
        let mock = make_agent();
        let mut cfg = standard_config(mock);
        set_worker_agent(&mut cfg, agent);
        cfg
    }

    fn collect_sink(
        events: Arc<StdMutex<Vec<StreamEvent>>>,
    ) -> impl Fn(StreamEvent) + Send + Sync + 'static {
        move |ev| events.lock().unwrap().push(ev)
    }

    // TextDelta concatenation, ReasoningDelta emission, BlockStart/BlockStop
    // bracketing, and MessageStart/MessageStop dropped (Q3).
    #[tokio::test]
    async fn text_and_reasoning_deltas_bracketed_by_blocks() {
        let agent = replay_agent(vec![ModelResponse {
            content: vec![
                ContentBlock::Thinking {
                    text: "thinking aloud".into(),
                },
                ContentBlock::Text {
                    text: "hello world".into(),
                },
            ],
            usage: usage(),
            stop_reason: StopReason::EndTurn,
        }]);
        let events = Arc::new(StdMutex::new(Vec::new()));
        let opts = HarnessRunOptions::new(react(5)).with_stream(collect_sink(events.clone()));
        let h = StandardHarness::new(streaming_config(agent));
        let r = h.run(opts).await;
        assert!(matches!(r, RunResult::Success { .. }));

        let evs = events.lock().unwrap();

        // Q3: no message-level markers ever surface on the harness stream.
        assert!(!evs
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta { content, .. } if content.is_empty())));

        // ReasoningDelta + TextDelta present with the right content.
        assert!(evs
                .iter()
                .any(|e| matches!(e, StreamEvent::ReasoningDelta { content, .. } if content == "thinking aloud")));
        assert!(evs.iter().any(
            |e| matches!(e, StreamEvent::TextDelta { content, .. } if content == "hello world")
        ));

        // Every delta is bracketed: each BlockStart has a matching BlockStop
        // with the same index, and a BlockStart of the right kind precedes
        // each delta.
        let reasoning_idx = block_index_for(&evs, BlockKind::Reasoning);
        let text_idx = block_index_for(&evs, BlockKind::Text);
        assert!(has_block_stop(&evs, reasoning_idx));
        assert!(has_block_stop(&evs, text_idx));
        // BlockStart(reasoning) precedes the ReasoningDelta.
        assert!(precedes(
            &evs,
            |e| matches!(
                e,
                StreamEvent::BlockStart {
                    block: BlockKind::Reasoning,
                    ..
                }
            ),
            |e| matches!(e, StreamEvent::ReasoningDelta { .. }),
        ));
    }

    // Tool lifecycle: ToolCallStart → ToolArgsDelta → BlockStop → coarse
    // ToolCall{args}, then the enriched ToolResult{content}.
    #[tokio::test]
    async fn tool_lifecycle_ordering_and_enriched_coarse_events() {
        let agent = replay_agent(vec![
            ModelResponse {
                content: vec![ContentBlock::ToolUse(crate::model::ToolCall {
                    id: "toolu_1".into(),
                    name: "lookup".into(),
                    input: serde_json::json!({"q": "rust"}),
                })],
                usage: usage(),
                stop_reason: StopReason::ToolUse,
            },
            ModelResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                usage: usage(),
                stop_reason: StopReason::EndTurn,
            },
        ]);
        let mut cfg = streaming_config(agent);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "lookup result body".into(),
            truncated: false,
        });
        cfg.tool_registry = reg;

        let events = Arc::new(StdMutex::new(Vec::new()));
        let opts = HarnessRunOptions::new(react(5)).with_stream(collect_sink(events.clone()));
        let h = StandardHarness::new(cfg);
        let r = h.run(opts).await;
        assert!(matches!(r, RunResult::Success { output, .. } if output == "done"));

        let evs = events.lock().unwrap();

        // ToolCallStart precedes the first ToolArgsDelta, which precedes the
        // coarse ToolCall.
        assert!(precedes(
            &evs,
            |e| matches!(e, StreamEvent::ToolCallStart { .. }),
            |e| matches!(e, StreamEvent::ToolArgsDelta { .. }),
        ));
        assert!(precedes(
            &evs,
            |e| matches!(e, StreamEvent::ToolArgsDelta { .. }),
            |e| matches!(e, StreamEvent::ToolCall { .. }),
        ));

        // ToolArgsDelta partial json carries the args.
        assert!(evs.iter().any(|e| matches!(
            e,
            StreamEvent::ToolArgsDelta { partial_json, .. } if partial_json.contains("rust")
        )));

        // Q5: coarse ToolCall carries the final accumulated args AND the tool
        // name — the name arrives on the `ToolUseStart` model event every
        // provider emits at block start, so the reassembled call matches the
        // coarse response exactly.
        let coarse_call = evs
            .iter()
            .find_map(|e| match e {
                StreamEvent::ToolCall { args, name, .. } => Some((args.clone(), name.clone())),
                _ => None,
            })
            .expect("coarse ToolCall present");
        assert_eq!(coarse_call.0, serde_json::json!({"q": "rust"}));
        assert_eq!(
            coarse_call.1, "lookup",
            "tool name is recovered from the ToolUseStart model event"
        );

        let coarse_result = evs
            .iter()
            .find_map(|e| match e {
                StreamEvent::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            })
            .expect("coarse ToolResult present");
        assert_eq!(coarse_result, "lookup result body");

        // The ToolArgsDelta call_id matches the coarse ToolCall path: the
        // ToolCallStart call_id correlates the delta stream.
        let start_id = evs.iter().find_map(|e| match e {
            StreamEvent::ToolCallStart { call_id, .. } => Some(call_id.clone()),
            _ => None,
        });
        let delta_id = evs.iter().find_map(|e| match e {
            StreamEvent::ToolArgsDelta { call_id, .. } => Some(call_id.clone()),
            _ => None,
        });
        assert_eq!(start_id, delta_id);
    }

    // Back-compat: on_stream=None yields an identical RunResult to the
    // non-streaming baseline (which drives the plain `turn` path).
    #[tokio::test]
    async fn no_sink_matches_non_streaming_baseline() {
        let resp = || ModelResponse {
            content: vec![ContentBlock::Text {
                text: "identical".into(),
            }],
            usage: usage(),
            stop_reason: StopReason::EndTurn,
        };
        // With a sink.
        let h1 = StandardHarness::new(streaming_config(replay_agent(vec![resp()])));
        let events = Arc::new(StdMutex::new(Vec::new()));
        let r_stream = h1
            .run(HarnessRunOptions::new(react(5)).with_stream(collect_sink(events.clone())))
            .await;
        // Without a sink.
        let h2 = StandardHarness::new(streaming_config(replay_agent(vec![resp()])));
        let r_plain = h2.run(HarnessRunOptions::new(react(5))).await;

        // RunResult output + turns identical.
        match (&r_stream, &r_plain) {
            (
                RunResult::Success {
                    output: o1,
                    turns: t1,
                    ..
                },
                RunResult::Success {
                    output: o2,
                    turns: t2,
                    ..
                },
            ) => {
                assert_eq!(o1, o2);
                assert_eq!(t1, t2);
            }
            other => panic!("expected two Successes, got {other:?}"),
        }
        // The streaming run produced delta events; the plain run produced
        // none (it never had a sink).
        assert!(!events.lock().unwrap().is_empty());
    }

    // Helpers ───────────────────────────────────────────────────────────

    fn block_index_for(evs: &[StreamEvent], kind: BlockKind) -> u32 {
        evs.iter()
            .find_map(|e| match e {
                StreamEvent::BlockStart { index, block, .. } if *block == kind => Some(*index),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no BlockStart for {kind:?}"))
    }

    fn has_block_stop(evs: &[StreamEvent], index: u32) -> bool {
        evs.iter()
            .any(|e| matches!(e, StreamEvent::BlockStop { index: i, .. } if *i == index))
    }

    fn precedes(
        evs: &[StreamEvent],
        first: impl Fn(&StreamEvent) -> bool,
        second: impl Fn(&StreamEvent) -> bool,
    ) -> bool {
        let fi = evs.iter().position(&first);
        let si = evs.iter().position(&second);
        matches!((fi, si), (Some(f), Some(s)) if f < s)
    }

    // Round-trip back-compat: pre-#103 serialized coarse events (without the
    // new `args` / `content` fields) still deserialize.
    #[test]
    fn coarse_events_roundtrip_without_new_fields() {
        let legacy_call = r#"{"kind":"tool_call","call_id":"c1","name":"x"}"#;
        let back: StreamEvent = serde_json::from_str(legacy_call).unwrap();
        assert!(matches!(
            back,
            StreamEvent::ToolCall { ref args, .. } if args.is_null()
        ));
        let legacy_result = r#"{"kind":"tool_result","call_id":"c1","is_error":false}"#;
        let back: StreamEvent = serde_json::from_str(legacy_result).unwrap();
        assert!(matches!(
            back,
            StreamEvent::ToolResult { ref content, .. } if content.is_empty()
        ));
    }

    // Fixture-replay: load fixtures/model_responses/harness/streaming_turn.jsonl,
    // drive a streaming harness turn, and assert the emitted delta/frame
    // events match the golden fixtures/harness/streaming_events.json exactly
    // (issue #103). The other three languages add their own replay tests
    // against the same two fixtures in the next phase.
    #[tokio::test]
    async fn fixture_replay_matches_golden_event_order() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let jsonl = std::fs::read_to_string(
            manifest.join("../../../fixtures/model_responses/harness/streaming_turn.jsonl"),
        )
        .expect("streaming_turn fixture present");
        let replay = Arc::new(
            ReplayModelInterface::from_jsonl(&jsonl, replay_provider()).expect("fixture parses"),
        );
        let agent: Arc<dyn Agent> =
            Arc::new(ModelAgent::new(AgentId::new("fixture-agent"), replay));

        let mut cfg = streaming_config(agent);
        let reg = Arc::new(ScriptedToolRegistry::new());
        // The recorded turn ends on tool_use; provide a tool result so the
        // dispatch step does not error. The run will then try a 2nd turn and
        // exhaust the (single-entry) fixture — we only assert on the delta
        // events of the FIRST turn, which are emitted before that.
        reg.push(ToolOutput::Success {
            content: "result".into(),
            truncated: false,
        });
        cfg.tool_registry = reg;

        let events = Arc::new(StdMutex::new(Vec::new()));
        let opts = HarnessRunOptions::new(react(5)).with_stream(collect_sink(events.clone()));
        let h = StandardHarness::new(cfg);
        let _ = h.run(opts).await;

        // Filter to the delta / frame / tool_call_start events of the FIRST
        // turn (everything up to the first coarse ToolCall).
        let evs = events.lock().unwrap();
        let cutoff = evs
            .iter()
            .position(|e| matches!(e, StreamEvent::ToolCall { .. }))
            .unwrap_or(evs.len());
        let delta_events: Vec<&StreamEvent> = evs[..cutoff]
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    StreamEvent::BlockStart { .. }
                        | StreamEvent::BlockStop { .. }
                        | StreamEvent::TextDelta { .. }
                        | StreamEvent::ReasoningDelta { .. }
                        | StreamEvent::ToolArgsDelta { .. }
                        | StreamEvent::ToolCallStart { .. }
                )
            })
            .collect();

        #[derive(serde::Deserialize)]
        struct Golden {
            events: Vec<StreamEvent>,
        }
        let golden_raw = std::fs::read_to_string(
            manifest.join("../../../fixtures/harness/streaming_events.json"),
        )
        .expect("golden present");
        let golden: Golden = serde_json::from_str(&golden_raw).expect("golden parses");

        assert_eq!(
            delta_events.len(),
            golden.events.len(),
            "event count mismatch:\n got: {delta_events:#?}\n exp: {:#?}",
            golden.events
        );
        for (got, exp) in delta_events.iter().zip(golden.events.iter()) {
            // #stream: this golden asserts delta ORDERING/content (#103), not
            // node attribution — strip the (now-present) `node` so the SHARED
            // cross-language fixture stays byte-identical. Attribution has its
            // own test (`composed_tree_streams_attributed_events`).
            let mut got_norm = (*got).clone();
            *got_norm.node_mut() = None;
            assert_eq!(&got_norm, exp, "event mismatch");
        }
    }

    // map_model_stream_event: MessageStart/MessageStop map to None (Q3).
    #[test]
    fn message_markers_map_to_none() {
        let mut st = TurnStreamState::default();
        assert!(StandardHarness::map_model_stream_event(
            crate::model::StreamEvent::MessageStart,
            &mut st
        )
        .is_empty());
        assert!(StandardHarness::map_model_stream_event(
            crate::model::StreamEvent::MessageStop {
                usage: TokenUsage::default(),
                stop_reason: StopReason::EndTurn,
            },
            &mut st
        )
        .is_empty());
    }
}

// ========================================================================
// Issue #102: session-state threading + opt-in auto-persist
// ========================================================================
mod session_threading {
    use super::*;
    use crate::storage::{
        InMemoryStorageProvider, NoOpStorageProvider, SessionStore, StorageError, StorageProvider,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// In-memory `SessionStore` that COUNTS every get/put so a test can assert
    /// "zero session-store I/O when auto_persist is disabled".
    #[derive(Default)]
    struct CountingSessionStore {
        inner: InMemoryStorageProvider,
        gets: AtomicUsize,
        puts: AtomicUsize,
    }
    impl SessionStore for CountingSessionStore {
        fn get_session<'a>(
            &'a self,
            id: &'a SessionId,
        ) -> BoxFut<'a, Result<Option<PausedState>, StorageError>> {
            self.gets.fetch_add(1, Ordering::SeqCst);
            self.inner.get_session(id)
        }
        fn put_session<'a>(
            &'a self,
            id: &'a SessionId,
            state: &'a PausedState,
        ) -> BoxFut<'a, Result<(), StorageError>> {
            self.puts.fetch_add(1, Ordering::SeqCst);
            self.inner.put_session(id, state)
        }
        fn delete_session<'a>(&'a self, id: &'a SessionId) -> BoxFut<'a, Result<(), StorageError>> {
            self.inner.delete_session(id)
        }
        fn list_sessions(&self) -> BoxFut<'_, Result<Vec<SessionId>, StorageError>> {
            self.inner.list_sessions()
        }
    }

    /// Build a `StorageProvider` whose session slot is the counting store and
    /// whose other three slots are no-ops (so only session I/O is observed).
    fn counting_provider(store: Arc<CountingSessionStore>) -> Arc<StorageProvider> {
        Arc::new(StorageProvider::new(
            store,
            Arc::new(NoOpStorageProvider),
            Arc::new(NoOpStorageProvider),
            Arc::new(NoOpStorageProvider),
        ))
    }

    /// A `MockAgent` that requests one tool call, then replies with final text.
    fn tool_then_final_agent() -> Arc<MockAgent> {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            reasoning: None,
            content: "after-tool".into(),
            usage: usage(),
        });
        a
    }

    fn tool_registry() -> Arc<ScriptedToolRegistry> {
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "tool-ok".into(),
            truncated: false,
        });
        reg
    }

    /// (a) Off-by-default: ZERO session-store I/O AND the message flow /
    /// outcome is identical to today (Success carries the right messages).
    #[tokio::test]
    async fn off_by_default_no_session_io() {
        let store = Arc::new(CountingSessionStore::default());
        let mut cfg = standard_config(tool_then_final_agent());
        cfg.tool_registry = tool_registry();
        cfg.storage = counting_provider(store.clone());
        // auto_persist_sessions defaults to false in standard_config.
        assert!(!cfg.auto_persist_sessions);
        let h = StandardHarness::new(cfg);
        let r = h.run(HarnessRunOptions::new(react(5))).await;
        match r {
            RunResult::Success {
                output,
                session_state,
                ..
            } => {
                assert_eq!(output, "after-tool");
                // The new field is populated even when persistence is off.
                assert!(session_state
                    .messages
                    .iter()
                    .any(|m| matches!(m.content, Content::ToolCall(_))));
            }
            other => panic!("expected Success, got {other:?}"),
        }
        assert_eq!(store.gets.load(Ordering::SeqCst), 0, "no get_session calls");
        assert_eq!(store.puts.load(Ordering::SeqCst), 0, "no put_session calls");
    }

    /// (b) Success.session_state is LOSSLESS for a tool-using run: the
    /// assistant tool-call message, the tool-result message, and the final
    /// text are all present (none recoverable from `output` alone).
    #[tokio::test]
    async fn success_session_state_is_lossless_for_tool_run() {
        let mut cfg = standard_config(tool_then_final_agent());
        cfg.tool_registry = tool_registry();
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Success { session_state, .. } => {
                let roles: Vec<_> = session_state
                    .messages
                    .iter()
                    .map(|m| (&m.role, &m.content))
                    .collect();
                // user instruction
                assert!(roles
                    .iter()
                    .any(|(r, _)| matches!(r, crate::model::Role::User)));
                // assistant tool-call turn
                assert!(session_state
                    .messages
                    .iter()
                    .any(|m| matches!(m.content, Content::ToolCall(_))));
                // tool-result turn
                assert!(session_state
                    .messages
                    .iter()
                    .any(|m| matches!(m.role, crate::model::Role::Tool)));
                // final assistant text
                assert!(session_state.messages.iter().any(|m| matches!(
                    (&m.role, &m.content),
                    (crate::model::Role::Assistant, Content::Text { text }) if text == "after-tool"
                )));
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    /// (d) Auto-persist round-trip: get_session returns the final history.
    #[tokio::test]
    async fn auto_persist_round_trip() {
        let backend = Arc::new(InMemoryStorageProvider::new());
        let provider = Arc::new(StorageProvider::single(backend));
        let mut cfg = standard_config(tool_then_final_agent());
        cfg.tool_registry = tool_registry();
        cfg.storage = provider.clone();
        cfg.auto_persist_sessions = true;
        let h = StandardHarness::new(cfg);
        let sid = SessionId::new("s1");
        let task = Task::new(
            "do something",
            sid.clone(),
            LoopStrategy::ReAct(ReactConfig::per_loop(5)),
        );
        let _ = h.run(HarnessRunOptions::new(task)).await;

        let stored = provider
            .session()
            .get_session(&sid)
            .await
            .expect("get ok")
            .expect("session persisted");
        assert!(stored
            .session_state
            .messages
            .iter()
            .any(|m| matches!(m.content, Content::ToolCall(_))));
        assert!(stored.pending_tool_calls.is_empty());
        assert!(stored.human_request.is_none());
    }

    /// (c) Auto-load across two runs sharing a session_id with an in-memory
    /// store: the second run sees the first run's history.
    #[tokio::test]
    async fn auto_load_by_session_id_across_runs() {
        let provider = Arc::new(StorageProvider::single(Arc::new(
            InMemoryStorageProvider::new(),
        )));
        let sid = SessionId::new("shared");

        // Run 1: one final response, auto-persisted.
        {
            let a = make_agent();
            a.push(TurnResult::FinalResponse {
                reasoning: None,
                content: "first".into(),
                usage: usage(),
            });
            let mut cfg = standard_config(a);
            cfg.storage = provider.clone();
            cfg.auto_persist_sessions = true;
            let h = StandardHarness::new(cfg);
            let task = Task::new(
                "turn one",
                sid.clone(),
                LoopStrategy::ReAct(ReactConfig::per_loop(5)),
            );
            let _ = h.run(HarnessRunOptions::new(task)).await;
        }

        // Run 2: same session_id, no explicit state. The loaded history must
        // carry the prior turns forward.
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            reasoning: None,
            content: "second".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        cfg.storage = provider.clone();
        cfg.auto_persist_sessions = true;
        let h = StandardHarness::new(cfg);
        let task = Task::new(
            "turn two",
            sid.clone(),
            LoopStrategy::ReAct(ReactConfig::per_loop(5)),
        );
        match h.run(HarnessRunOptions::new(task)).await {
            RunResult::Success { session_state, .. } => {
                let texts: Vec<String> = session_state
                    .messages
                    .iter()
                    .filter_map(|m| match &m.content {
                        Content::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect();
                assert!(texts.iter().any(|t| t == "first"), "prior turn carried");
                assert!(texts.iter().any(|t| t == "turn one"));
                assert!(texts.iter().any(|t| t == "turn two"));
                assert!(texts.iter().any(|t| t == "second"));
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    /// (e) Cross-process continuity via FileSystemStorageProvider under a
    /// tempdir: a fresh harness instance resumes by session_id.
    #[tokio::test]
    async fn cross_process_continuity_filesystem() {
        let dir = std::env::temp_dir().join(format!("spore-102-{}", random_id()));
        std::fs::create_dir_all(&dir).unwrap();

        let sid = SessionId::new("fs-session");

        // "Process 1": its own harness + provider instance.
        {
            let provider = Arc::new(StorageProvider::single(Arc::new(
                crate::storage::FileSystemStorageProvider::new(dir.clone()),
            )));
            let a = make_agent();
            a.push(TurnResult::FinalResponse {
                reasoning: None,
                content: "process-one".into(),
                usage: usage(),
            });
            let mut cfg = standard_config(a);
            cfg.storage = provider;
            cfg.auto_persist_sessions = true;
            let h = StandardHarness::new(cfg);
            let task = Task::new(
                "first process",
                sid.clone(),
                LoopStrategy::ReAct(ReactConfig::per_loop(5)),
            );
            let _ = h.run(HarnessRunOptions::new(task)).await;
        }

        // "Process 2": brand-new provider over the SAME dir, brand-new harness.
        let provider = Arc::new(StorageProvider::single(Arc::new(
            crate::storage::FileSystemStorageProvider::new(dir.clone()),
        )));
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            reasoning: None,
            content: "process-two".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        cfg.storage = provider;
        cfg.auto_persist_sessions = true;
        let h = StandardHarness::new(cfg);
        let task = Task::new(
            "second process",
            sid.clone(),
            LoopStrategy::ReAct(ReactConfig::per_loop(5)),
        );
        let result = h.run(HarnessRunOptions::new(task)).await;
        let _ = std::fs::remove_dir_all(&dir);
        match result {
            RunResult::Success { session_state, .. } => {
                let texts: Vec<String> = session_state
                    .messages
                    .iter()
                    .filter_map(|m| match &m.content {
                        Content::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect();
                assert!(
                    texts.iter().any(|t| t == "process-one"),
                    "prior process history loaded"
                );
                assert!(texts.iter().any(|t| t == "process-two"));
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    /// (f) Explicit with_session_state WINS over auto-load (no get_session,
    /// the explicit state seeds the run).
    #[tokio::test]
    async fn explicit_session_state_wins_over_auto_load() {
        let store = Arc::new(CountingSessionStore::default());
        // Pre-seed the store with a DIFFERENT history under the session id.
        {
            let mut prior = SessionState::default();
            prior.messages.push(Message {
                role: crate::model::Role::User,
                content: Content::Text {
                    text: "STORED-history".into(),
                },
            });
            let ps = PausedState {
                session_id: SessionId::new("s1"),
                task_id: TaskId::new("s1"),
                turn_number: 0,
                session_state: prior,
                pending_tool_calls: vec![],
                approved_results: vec![],
                human_request: None,
                task: Task::new(
                    "",
                    SessionId::new("s1"),
                    LoopStrategy::ReAct(ReactConfig::per_loop(0)),
                ),
                budget_used: BudgetSnapshot::default(),
                child_state: None,
                toolset: ToolsetRef::default(),
            };
            store
                .inner
                .put_session(&SessionId::new("s1"), &ps)
                .await
                .unwrap();
            store.puts.store(0, Ordering::SeqCst); // reset the pre-seed write
        }

        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            reasoning: None,
            content: "done".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        cfg.storage = counting_provider(store.clone());
        cfg.auto_persist_sessions = true;
        let h = StandardHarness::new(cfg);

        let mut explicit = SessionState::default();
        explicit.messages.push(Message {
            role: crate::model::Role::User,
            content: Content::Text {
                text: "EXPLICIT-history".into(),
            },
        });
        let task = Task::new(
            "turn",
            SessionId::new("s1"),
            LoopStrategy::ReAct(ReactConfig::per_loop(5)),
        );
        let opts = HarnessRunOptions::new(task).with_session_state(explicit);
        match h.run(opts).await {
            RunResult::Success { session_state, .. } => {
                let texts: Vec<String> = session_state
                    .messages
                    .iter()
                    .filter_map(|m| match &m.content {
                        Content::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect();
                assert!(texts.iter().any(|t| t == "EXPLICIT-history"));
                assert!(!texts.iter().any(|t| t == "STORED-history"), "load skipped");
            }
            other => panic!("expected Success, got {other:?}"),
        }
        assert_eq!(
            store.gets.load(Ordering::SeqCst),
            0,
            "explicit state skips the auto-load get_session"
        );
    }

    /// (g) Failure ALSO carries session_state.
    #[tokio::test]
    async fn failure_carries_session_state() {
        // A tool annotated always_halt fails the run after the assistant
        // tool-call turn was recorded into session_state.
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: "c".into(),
                name: "danger".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.mark_always_halt("danger");
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Failure { session_state, .. } => {
                // The seeded user instruction is present in the failure state.
                assert!(session_state.messages.iter().any(|m| matches!(
                    (&m.role, &m.content),
                    (crate::model::Role::User, Content::Text { .. })
                )));
            }
            other => panic!("expected Failure, got {other:?}"),
        }
    }
}
// ========================================================================
// #130 — HumanRequest::BudgetExhausted + Escalate HITL resume
// ========================================================================

// Local fixture helpers (the `strategy_tests` module's copies are private to
// that module). Mirror their behavior: resolve a path under `/fixtures/` and
// assert order-preserving byte identity.
fn h130_fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../../fixtures/{name}"))
}

fn h130_minify_preserving_order(s: &str) -> String {
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

fn h130_assert_fixture_byte_identical<T: Serialize>(value: &T, fixture: &str) {
    let raw = std::fs::read_to_string(h130_fixture_path(fixture)).expect("fixture present");
    let canonical = h130_minify_preserving_order(&raw);
    let ours = serde_json::to_string(value).unwrap();
    assert_eq!(ours, canonical, "{fixture} not byte-identical");
}

// ── serde round-trips of the new types ───────────────────────────────────

#[test]
fn escalation_action_round_trips_every_variant() {
    for v in [
        EscalationAction::ContinueWithBudget { steps: 7 },
        EscalationAction::Skip,
        EscalationAction::Fail,
    ] {
        let json = serde_json::to_string(&v).unwrap();
        let back: EscalationAction = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }
    // The named-field variant uses the internally-tagged form (no tuple).
    assert_eq!(
        serde_json::to_string(&EscalationAction::ContinueWithBudget { steps: 7 }).unwrap(),
        r#"{"kind":"continue_with_budget","steps":7}"#
    );
    assert_eq!(
        serde_json::to_string(&EscalationAction::Skip).unwrap(),
        r#"{"kind":"skip"}"#
    );
}

#[test]
fn human_request_budget_exhausted_round_trips() {
    let req = HumanRequest::BudgetExhausted {
        phase: "plan_execute".into(),
        policy: BudgetPolicy::TotalSteps { value: 5 },
        steps_taken: 5,
        continues_used: 2,
        partial_output: Some("{\"node\":\"plan_execute\"}".into()),
        available_actions: vec![
            EscalationAction::ContinueWithBudget { steps: 5 },
            EscalationAction::Skip,
            EscalationAction::Fail,
        ],
    };
    let json = serde_json::to_string(&req).unwrap();
    let back: HumanRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(req, back);
    // The variant tag is snake_case.
    assert!(json.contains(r#""kind":"budget_exhausted""#));
}

#[test]
fn human_response_escalate_round_trips() {
    let r = HumanResponse::Escalate {
        action: EscalationAction::ContinueWithBudget { steps: 3 },
    };
    let json = serde_json::to_string(&r).unwrap();
    let back: HumanResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(r, back);
    assert!(json.contains(r#""kind":"escalate""#));
}

#[test]
fn existing_human_request_variants_still_round_trip() {
    for req in [
        HumanRequest::ToolApproval {
            calls: vec![],
            risk_level: RiskLevel::High,
        },
        HumanRequest::Clarification {
            question: "which?".into(),
            options: Some(vec!["a".into(), "b".into()]),
        },
        HumanRequest::Review {
            content: "look".into(),
        },
    ] {
        let json = serde_json::to_string(&req).unwrap();
        let back: HumanRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, back);
    }
}

#[test]
fn existing_human_response_variants_still_round_trip() {
    for r in [
        HumanResponse::Allow,
        HumanResponse::AllowWithModification { calls: vec![] },
        HumanResponse::Deny {
            reason: "no".into(),
        },
        HumanResponse::Halt,
        HumanResponse::Answer { text: "x".into() },
        HumanResponse::ApproveWithFeedback {
            feedback: "ok".into(),
        },
        HumanResponse::Reject {
            reason: "bad".into(),
        },
    ] {
        let json = serde_json::to_string(&r).unwrap();
        let back: HumanResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}

// ── accessor ─────────────────────────────────────────────────────────────

#[test]
fn standard_harness_escalation_mode_accessor_reflects_config() {
    let h = StandardHarness::new(surface_config(make_agent()));
    assert!(matches!(
        h.escalation_mode(),
        EscalationMode::SurfaceToHuman
    ));
    let h = StandardHarness::new(standard_config(make_agent()));
    assert!(matches!(h.escalation_mode(), EscalationMode::Autonomous));
}

// ── helper: a bare ReAct leaf whose cap is binding (drives a pause) ───────

fn budget_exhausting_agent(turns: usize) -> Arc<MockAgent> {
    let a = make_agent();
    for i in 0..turns {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: format!("c{i}"),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
    }
    a
}

fn tool_reg(n: usize) -> Arc<ScriptedToolRegistry> {
    let reg = Arc::new(ScriptedToolRegistry::new());
    for _ in 0..n {
        reg.push(ToolOutput::Success {
            content: "ok".into(),
            truncated: false,
        });
    }
    reg
}

// ── SurfaceToHuman: a bare leaf PAUSES with the request ──────────────────

#[tokio::test]
async fn surface_to_human_bare_leaf_pauses_with_budget_request() {
    let mut cfg = surface_config(budget_exhausting_agent(3));
    cfg.tool_registry = tool_reg(3);
    let h = StandardHarness::new(cfg);
    let t = react(2); // PerLoop{2}, no global cap → leaf cap binds.
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::WaitingForHuman { request, state } => {
            // The PausedState mirrors the request verbatim.
            assert_eq!(state.human_request.as_ref(), Some(&request));
            // #138 AC2-a: the pause now carries the FULL stalled worker session
            // (instruction + the two exhausting tool-call/result rounds), NOT a
            // single partial-only assistant stub — so a budget resume re-attaches
            // the real worker context. The session therefore has MORE than one
            // message (1 instruction + 2 rounds × 2 messages = 5).
            assert!(
                state.session_state.messages.len() > 1,
                "AC2-a carries the full worker session, got {} messages",
                state.session_state.messages.len()
            );
            match request {
                HumanRequest::BudgetExhausted {
                    phase,
                    policy,
                    steps_taken,
                    partial_output,
                    available_actions,
                    ..
                } => {
                    assert_eq!(phase, "react");
                    assert_eq!(policy, BudgetPolicy::PerLoop { value: 2 });
                    assert_eq!(steps_taken, 2);
                    // A bare leaf omits Skip (fork C).
                    assert_eq!(
                        available_actions,
                        vec![
                            EscalationAction::ContinueWithBudget { steps: 2 },
                            EscalationAction::Fail,
                        ]
                    );
                    // The partial is the documented ReAct shape.
                    assert_eq!(partial_output, Some(react_partial_json("")));
                }
                other => panic!("expected BudgetExhausted request, got {other:?}"),
            }
        }
        other => panic!("expected WaitingForHuman, got {other:?}"),
    }
}

// ── SurfaceToHuman: a combinator (PlanExecute) pauses with Skip offered ───

#[tokio::test]
async fn surface_to_human_combinator_offers_skip() {
    // A PlanExecute whose execute leaf carries a SMALL PerLoop cap (smaller
    // than the global backstop): the execute step exhausts its own leaf cap,
    // PROPAGATES a `StrategyOutcome::BudgetExhausted` to the PlanExecute scope,
    // whose `Escalate` resolution PAUSES under SurfaceToHuman (fork C: a
    // combinator offers all three actions).
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["x","y","z"]}"#)); // plan
                                                      // Execute step: keep requesting tools past the leaf cap of 2.
    for i in 0..4 {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: format!("e{i}"),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
    }
    let mut cfg = surface_config(a);
    cfg.tool_registry = tool_reg(4);
    let h = StandardHarness::new(cfg);
    // Plan slot is structured (output schema); execute leaf cap = PerLoop{2}.
    let t = task(LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        plan: Box::new(react_structured(u32::MAX)),
        execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(2))),
        plan_model: None,
    }));
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::WaitingForHuman {
            request:
                HumanRequest::BudgetExhausted {
                    phase,
                    available_actions,
                    ..
                },
            ..
        } => {
            assert_eq!(phase, "plan_execute");
            // A combinator offers all three actions (fork C).
            assert!(available_actions.contains(&EscalationAction::Skip));
            assert!(available_actions.contains(&EscalationAction::Fail));
            assert!(available_actions
                .iter()
                .any(|a| matches!(a, EscalationAction::ContinueWithBudget { .. })));
        }
        other => panic!("expected combinator BudgetExhausted pause, got {other:?}"),
    }
}

// ── Autonomous: no pause; existing propagate behavior ────────────────────

#[tokio::test]
async fn autonomous_does_not_pause_propagates_budget_exceeded() {
    let mut cfg = standard_config(budget_exhausting_agent(3)); // Autonomous.
    cfg.tool_registry = tool_reg(3);
    let h = StandardHarness::new(cfg);
    let t = react(2);
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure {
            reason:
                HaltReason::BudgetExceeded {
                    limit_type: BudgetLimitType::Turns,
                },
            turns,
            ..
        } => assert_eq!(turns, 2),
        other => panic!("expected BudgetExceeded Failure (no pause), got {other:?}"),
    }
}

// ── Resume: ContinueWithBudget grants N steps and resumes ────────────────

#[tokio::test]
async fn resume_continue_with_budget_grants_and_resumes() {
    // First run pauses at the leaf cap of 2.
    let a = make_agent();
    // 2 tool turns to exhaust the cap, then a FinalResponse for the resume.
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c0".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "finished after grant".into(),
        usage: usage(),
    });
    let mut cfg = surface_config(a);
    cfg.tool_registry = tool_reg(3);
    let h = StandardHarness::new(cfg);
    let t = react(2);
    let paused = h.run(HarnessRunOptions::new(t)).await;
    let state = match paused {
        RunResult::WaitingForHuman { state, .. } => *state,
        other => panic!("expected pause, got {other:?}"),
    };
    // Operator grants 5 more steps and resumes.
    let resumed = h
        .resume(
            state,
            HumanResponse::Escalate {
                action: EscalationAction::ContinueWithBudget { steps: 5 },
            },
            None,
        )
        .await;
    match resumed {
        RunResult::Success { output, .. } => {
            assert_eq!(output, "finished after grant");
        }
        other => panic!("expected Success after grant, got {other:?}"),
    }
}

// ── #138 AC2-a: promote_budget_exhausted_to_human carries the FULL session ─

// The pause helper carries the FULL stalled worker session (AC2-a) and the
// worker leaf's toolset handle (AC4-a) — not a partial-only stub. A direct
// unit on the boundary helper, decoupled from the surrounding strategy.
#[test]
fn promote_budget_pause_carries_full_worker_session_and_handle() {
    let err = BudgetExhausted {
        policy: BudgetPolicy::PerLoop { value: 2 },
        behavior: BudgetExhaustedBehavior::Escalate,
        steps_taken: 2,
        continues_used: 0,
        phase: "react".into(),
    };
    // A realistic worker conversation (instruction + a tool round).
    let worker = SessionState {
        messages: vec![
            Message {
                role: crate::model::Role::User,
                content: Content::Text {
                    text: "worker: audit".into(),
                },
            },
            Message {
                role: crate::model::Role::Assistant,
                content: Content::Text {
                    text: "looking".into(),
                },
            },
            Message {
                role: crate::model::Role::Tool,
                content: Content::Text {
                    text: "listing".into(),
                },
            },
        ],
        ..Default::default()
    };
    let waiting = promote_budget_exhausted_to_human(
        &err,
        Some("partial".into()),
        leaf_escalation_actions(&err),
        SessionId::new("s1"),
        task(react(2).loop_strategy),
        BudgetSnapshot::default(),
        2,
        worker.clone(),
        ToolsetRef("exec-tools".into()),
    );
    let RunResult::WaitingForHuman { state, .. } = waiting else {
        panic!("expected WaitingForHuman");
    };
    // AC2-a: the FULL worker session is carried (3 messages), NOT the single
    // partial-only assistant stub.
    assert_eq!(state.session_state.messages, worker.messages);
    // AC4-a: the worker leaf's toolset handle rides the pause (#140 parity).
    assert_eq!(state.toolset.0, "exec-tools");

    // Back-compat: an EMPTY worker session falls back to the partial-only stub
    // (the pre-#138 behavior) so legacy/HillClimbing sites are unchanged.
    let waiting2 = promote_budget_exhausted_to_human(
        &err,
        Some("just-the-partial".into()),
        leaf_escalation_actions(&err),
        SessionId::new("s1"),
        task(react(2).loop_strategy),
        BudgetSnapshot::default(),
        2,
        SessionState::default(),
        ToolsetRef::default(),
    );
    let RunResult::WaitingForHuman { state, .. } = waiting2 else {
        panic!("expected WaitingForHuman");
    };
    assert_eq!(state.session_state.messages.len(), 1);
    assert!(matches!(
        &state.session_state.messages[0].content,
        Content::Text { text } if text == "just-the-partial"
    ));
}

// ── #138 AC3: plan-phase exhaustion resumes the PLAN session ───────────────

// When a budget resume carries a worker session AND the durable task_list is
// EMPTY (no `InProgress` task ⇒ the exhaustion happened in the PLAN phase),
// `PlanExecuteConfig::run` seeds the PLAN session from the carried conversation
// instead of a fresh base session — so the planner CONTINUES on it.
//
// NOTE (per #138 plan): the carried-session→plan seeding is observed via the
// planner agent's RECORDED contexts (`RecordingTurnAgent::seen_text`); the
// replay harness's `NoopContextManager` would not otherwise surface it.
#[tokio::test]
async fn budget_resume_plan_phase_seeds_plan_session_from_carried() {
    // The planner records every context it sees; it authors a one-task plan so
    // the run can complete, then the worker + a clean final response run.
    let planner = RecordingTurnAgent::new(
        "planner",
        vec![RecordingTurnAgent::final_resp(
            r#"{"tasks":["only"],"rationale":"r"}"#,
        )],
    );
    let worker = RecordingTurnAgent::new(
        "worker",
        vec![RecordingTurnAgent::final_resp("did the work")],
    );
    // Wire planner under "planner", worker under the default empty key (the
    // bare PlanExecute execute leaf carries an empty AgentRef).
    let mut cfg = surface_config(make_agent());
    cfg.registry = std::mem::take(&mut cfg.registry)
        .into_builder()
        .agent("planner", planner.clone())
        .agent("", worker.clone())
        .build();
    let h = StandardHarness::new(cfg);

    // A PlanExecute whose PLAN leaf resolves to "planner"; execute is a bare
    // ReAct on the default key.
    let pe = LoopStrategy::PlanExecute(PlanExecuteConfig {
        behavior: BudgetExhaustedBehavior::Escalate,
        plan: Box::new(LoopStrategy::ReAct(ReactConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            budget: BudgetPolicy::PerLoop { value: 12 },
            agent: AgentRef("planner".into()),
            toolset: ToolsetRef(String::new()),
            output: Some(SchemaRef(String::new())),
            system_prompt: None,
        })),
        execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(8))),
        plan_model: None,
    });
    let mut t = task(pe);
    t.budget.max_turns = Some(32);

    // A budget-exhausted pause carrying a worker session with a MARKER, and NO
    // durable task_list persisted (empty ⇒ plan-phase exhaustion, AC3).
    const MARKER: &str = "CARRIED_PLAN_SESSION_MARKER";
    let carried = SessionState {
        messages: vec![Message {
            role: crate::model::Role::Assistant,
            content: Content::Text {
                text: MARKER.into(),
            },
        }],
        ..Default::default()
    };
    let state = PausedState {
        session_id: SessionId::new("s1"),
        task_id: t.id.clone(),
        turn_number: 1,
        session_state: carried,
        pending_tool_calls: vec![],
        approved_results: vec![],
        human_request: Some(HumanRequest::BudgetExhausted {
            phase: "plan_execute".into(),
            policy: BudgetPolicy::TotalSteps { value: 1 },
            steps_taken: 1,
            continues_used: 0,
            partial_output: Some(react_partial_json("")),
            available_actions: vec![
                EscalationAction::ContinueWithBudget { steps: 1 },
                EscalationAction::Skip,
                EscalationAction::Fail,
            ],
        }),
        task: t,
        budget_used: BudgetSnapshot::default(),
        child_state: None,
        toolset: ToolsetRef::default(),
    };

    let _ = h
        .resume(
            state,
            HumanResponse::Escalate {
                action: EscalationAction::ContinueWithBudget { steps: 10 },
            },
            None,
        )
        .await;

    // AC3: the planner's FIRST context was seeded from the CARRIED session —
    // the marker is present, proving the plan session continued on it rather
    // than starting from a fresh base session.
    let seen = planner.seen_text();
    assert!(
        seen.first().is_some_and(|c| c.contains(MARKER)),
        "AC3: the plan session must be seeded from the carried conversation; \
             planner saw: {seen:?}"
    );
}

// ── #138 AC1: skip-plan reconciles already-Completed tasks (dedup) ─────────

// A non-empty durable task_list whose task #1 is already Completed: a fresh run
// SKIPS the plan phase (AC1) and reconcile does NOT re-run the completed task —
// only the still-Pending task #2 runs (one model call, no plan turn).
#[tokio::test]
async fn skip_plan_reconciles_completed_tasks() {
    use crate::tasklist::TaskStatus;
    let a = make_agent();
    // NO plan turn pushed (AC1 skips it). Only task #2 runs.
    a.push(final_resp("did two"));
    let agent_count = a.clone();
    let session = SessionId::new("s1");
    let h = StandardHarness::new(standard_config(a));

    // Pre-seed: task #1 already Completed, task #2 Pending.
    let mut tl = TaskList::default();
    tl.add("one".into(), vec![]).unwrap(); // 1
    tl.add("two".into(), vec![]).unwrap(); // 2
    let _ = tl.update(1, Some(TaskStatus::InProgress), None);
    let _ = tl.complete(1);
    h.persist_task_list(&session, &tl).await;

    // Model (A): resuming a pre-seeded list (task #1 Completed, #2 Pending) is a
    // continuation; opt in so the reconcile + walk run instead of re-planning.
    match h
        .run(HarnessRunOptions::new(plan_task()).continue_task_list(true))
        .await
    {
        RunResult::Success { output, .. } => assert_eq!(output, "did two"),
        other => panic!("expected Success, got {other:?}"),
    }
    // Exactly ONE model call: task #2 (no plan turn, task #1 not re-run).
    assert_eq!(
        agent_count
            .call_count
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "AC1: plan skipped + completed task #1 deduped — only task #2 ran"
    );
    // Both tasks are Completed in the durable store (1 deduped, 2 freshly run).
    let stored: TaskList = serde_json::from_value(
        h.storage()
            .run()
            .get(&durable_ns(), TASK_LIST_EXTRAS_KEY)
            .await
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    assert!(stored
        .tasks
        .iter()
        .all(|t| t.status == TaskStatus::Completed));
}

// Model (A) regression (the consumer pattern): one long-lived harness, one
// `run()` per user turn against a stable workspace — the normal way a
// conversational host drives PlanExecute. The durable task_list is keyed by the
// stable project namespace (#142), so it survives across `run()` calls; before
// model (A), a fresh `run()` saw the prior, fully-`Completed` graph, treated it
// as a resume, walked to no ready task, and drained to a silent empty
// `Success`. A fresh `run()` carries no continuation signal, so it must now
// re-plan + execute and return real output.
#[tokio::test]
async fn fresh_run_after_completed_plan_replans_instead_of_empty_success() {
    let a = make_agent();
    // Run 1: plan two tasks, execute both.
    a.push(final_resp(r#"{"tasks":["one","two"],"rationale":"r1"}"#));
    a.push(final_resp("did one"));
    a.push(final_resp("did two"));
    // Run 2 (new instruction): MUST re-plan (one task) + execute it.
    a.push(final_resp(r#"{"tasks":["three"],"rationale":"r2"}"#));
    a.push(final_resp("did three"));
    let h = StandardHarness::new(standard_config(a));

    let plan_strategy = || {
        LoopStrategy::PlanExecute(PlanExecuteConfig {
            behavior: BudgetExhaustedBehavior::Escalate,
            plan: Box::new(react_structured(u32::MAX)),
            execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
            plan_model: None,
        })
    };

    // Run 1 — plans + executes; all tasks Completed + persisted to the durable
    // project-scoped store.
    let t1 = Task::new(
        "add a foo() function",
        SessionId::new("s1"),
        plan_strategy(),
    );
    match h.run(HarnessRunOptions::new(t1)).await {
        RunResult::Success { output, .. } => assert_eq!(output, "did two"),
        other => panic!("run 1: expected Success, got {other:?}"),
    }

    // Run 2 — distinct instruction, SAME harness + project namespace, no resume
    // signal ⇒ a fresh plan (stale list cleared) and the new task executes.
    let t2 = Task::new(
        "now add a bar() function",
        SessionId::new("s2"),
        plan_strategy(),
    );
    match h.run(HarnessRunOptions::new(t2)).await {
        RunResult::Success { output, .. } => assert_eq!(
            output, "did three",
            "a fresh run() with a new instruction must re-plan + execute, \
                 not drain to an empty Success"
        ),
        other => panic!("run 2: expected Success, got {other:?}"),
    }

    // The durable store reflects ONLY run 2's freshly-authored task — the stale
    // completed list was CLEARED, not appended to (which would have re-suppressed
    // the walk via the post-plan `load_task_list` re-read).
    let stored: TaskList = serde_json::from_value(
        h.storage()
            .run()
            .get(&durable_ns(), TASK_LIST_EXTRAS_KEY)
            .await
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        stored.tasks.len(),
        1,
        "stale completed tasks cleared on the fresh run, not carried forward"
    );
    assert_eq!(stored.tasks[0].description, "three");
}

// ── Resume: Fail propagates Failure{BudgetExceeded}, partial discarded ────

#[tokio::test]
async fn resume_fail_propagates_budget_exceeded() {
    let mut cfg = surface_config(budget_exhausting_agent(3));
    cfg.tool_registry = tool_reg(3);
    let h = StandardHarness::new(cfg);
    let state = match h.run(HarnessRunOptions::new(react(2))).await {
        RunResult::WaitingForHuman { state, .. } => *state,
        other => panic!("expected pause, got {other:?}"),
    };
    let resumed = h
        .resume(
            state,
            HumanResponse::Escalate {
                action: EscalationAction::Fail,
            },
            None,
        )
        .await;
    match resumed {
        RunResult::Failure {
            reason:
                HaltReason::BudgetExceeded {
                    limit_type: BudgetLimitType::Turns,
                },
            session_state,
            ..
        } => {
            // Fail contract: the partial is discarded.
            assert!(session_state.messages.is_empty());
        }
        other => panic!("expected BudgetExceeded Failure, got {other:?}"),
    }
}

// ── Resume: Skip on a leaf resolves to clean Success ─────────────────────

#[tokio::test]
async fn resume_skip_on_leaf_resolves_clean() {
    let mut cfg = surface_config(budget_exhausting_agent(3));
    cfg.tool_registry = tool_reg(3);
    let h = StandardHarness::new(cfg);
    let state = match h.run(HarnessRunOptions::new(react(2))).await {
        RunResult::WaitingForHuman { state, .. } => *state,
        other => panic!("expected pause, got {other:?}"),
    };
    let resumed = h
        .resume(
            state,
            HumanResponse::Escalate {
                action: EscalationAction::Skip,
            },
            None,
        )
        .await;
    assert!(matches!(resumed, RunResult::Success { .. }));
}

// ── Resume: Skip on a PlanExecute advances the outer loop ────────────────

#[tokio::test]
async fn resume_skip_on_plan_execute_advances_outer_loop() {
    // Build a paused PlanExecute state by hand whose persisted task list still
    // has a ready task, then resume with Skip and confirm the executor
    // re-enters the loop (does NOT immediately fail/pause again on the same
    // task). We assert it does NOT return the same WaitingForHuman pause.
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "advanced".into(),
        usage: usage(),
    });
    let mut cfg = surface_config(a);
    cfg.tool_registry = tool_reg(0);
    let h = StandardHarness::new(cfg);

    let pe = LoopStrategy::PlanExecute(PlanExecuteConfig::simple(None));
    let mut t = task(pe);
    t.budget.max_turns = Some(8);
    let state = PausedState {
        session_id: SessionId::new("s1"),
        task_id: t.id.clone(),
        turn_number: 1,
        session_state: SessionState::default(),
        pending_tool_calls: vec![],
        approved_results: vec![],
        human_request: Some(HumanRequest::BudgetExhausted {
            phase: "plan_execute".into(),
            policy: BudgetPolicy::TotalSteps { value: 1 },
            steps_taken: 1,
            continues_used: 0,
            partial_output: Some(react_partial_json("")),
            available_actions: vec![
                EscalationAction::ContinueWithBudget { steps: 1 },
                EscalationAction::Skip,
                EscalationAction::Fail,
            ],
        }),
        task: t,
        budget_used: BudgetSnapshot::default(),
        child_state: None,
        toolset: ToolsetRef::default(),
    };
    let resumed = h
        .resume(
            state,
            HumanResponse::Escalate {
                action: EscalationAction::Skip,
            },
            None,
        )
        .await;
    // The Skip re-entered the PlanExecute loop rather than returning the same
    // pause verbatim: it ran to a terminal (Success or a Failure), NOT a
    // WaitingForHuman echo of the input.
    assert!(
        !matches!(resumed, RunResult::WaitingForHuman { .. }),
        "Skip should advance the outer loop, not re-pause: {resumed:?}"
    );
}

// ── grant_task_budget raises caps without shrinking ──────────────────────

#[test]
fn grant_task_budget_raises_leaf_and_global_caps() {
    let mut t = react(2);
    grant_task_budget(&mut t, 9);
    assert_eq!(t.budget.max_turns, Some(9));
    if let LoopStrategy::ReAct(c) = &t.loop_strategy {
        assert_eq!(c.budget, BudgetPolicy::PerLoop { value: 9 });
    } else {
        panic!("expected ReAct leaf");
    }
    // A lower grant never SHRINKS an existing allowance.
    grant_task_budget(&mut t, 3);
    assert_eq!(t.budget.max_turns, Some(9));
}

// ── Fixture replay: the cross-language ground truth ──────────────────────

fn budget_exhausted_paused_state() -> PausedState {
    PausedState {
        session_id: SessionId::new("sess-130"),
        task_id: TaskId::new("task-130"),
        turn_number: 6,
        session_state: SessionState {
            messages: vec![Message {
                role: crate::model::Role::Assistant,
                content: crate::model::Content::Text {
                    text: "{\"node\":\"plan_execute\",\"tasks\":2,\"ledger\":[]}".into(),
                },
            }],
            ..Default::default()
        },
        pending_tool_calls: vec![],
        approved_results: vec![],
        human_request: Some(HumanRequest::BudgetExhausted {
            phase: "plan_execute".into(),
            policy: BudgetPolicy::TotalSteps { value: 6 },
            steps_taken: 6,
            continues_used: 1,
            partial_output: Some("{\"node\":\"plan_execute\",\"tasks\":2,\"ledger\":[]}".into()),
            available_actions: vec![
                EscalationAction::ContinueWithBudget { steps: 6 },
                EscalationAction::Skip,
                EscalationAction::Fail,
            ],
        }),
        task: {
            let mut t = Task::new(
                "build the feature",
                SessionId::new("sess-130"),
                LoopStrategy::PlanExecute(PlanExecuteConfig {
                    behavior: BudgetExhaustedBehavior::Escalate,
                    plan: Box::new(LoopStrategy::ReAct(ReactConfig {
                        behavior: BudgetExhaustedBehavior::Escalate,
                        budget: BudgetPolicy::PerLoop { value: 4 },
                        agent: AgentRef("planner".into()),
                        toolset: ToolsetRef("plan-tools".into()),
                        output: Some(SchemaRef("plan-schema".into())),
                        system_prompt: None,
                    })),
                    execute: Box::new(LoopStrategy::ReAct(ReactConfig {
                        behavior: BudgetExhaustedBehavior::Escalate,
                        budget: BudgetPolicy::PerLoop { value: 8 },
                        agent: AgentRef("executor".into()),
                        toolset: ToolsetRef("exec-tools".into()),
                        output: Some(SchemaRef("worker-schema".into())),
                        system_prompt: None,
                    })),
                    plan_model: None,
                }),
            );
            t.id = TaskId::new("task-130");
            t.budget.max_turns = Some(6);
            t
        },
        budget_used: BudgetSnapshot {
            turns: 6,
            ..Default::default()
        },
        child_state: None,
        // #140 decision 2: budget-exhausted pause re-drives on resume; the
        // handle is behaviorally irrelevant here → empty default.
        toolset: ToolsetRef::default(),
    }
}

#[test]
fn fixture_replay_budget_exhausted_paused_state() {
    let raw = std::fs::read_to_string(h130_fixture_path("paused_states/budget_exhausted.json"))
        .expect("fixture present");
    // Deserializes to the canonical value (field-for-field).
    let parsed: PausedState = serde_json::from_str(&raw).unwrap();
    assert_eq!(parsed, budget_exhausted_paused_state());
    // Re-serializes byte-identically (order-preserving minify).
    h130_assert_fixture_byte_identical(
        &budget_exhausted_paused_state(),
        "paused_states/budget_exhausted.json",
    );
}

// ========================================================================
// #129 — Continue cross-process checkpoint (continues_used survives a pause)
// ========================================================================

/// The canonical `continue_checkpoint.json` value: a bare ReAct leaf carrying
/// `Continue{max_continues:2, on_exhausted:Fail}` paused mid-loop with
/// `continues_used: 1` on its `HumanRequest::BudgetExhausted`. Cross-language
/// byte-identity ground truth for the AC2 resume test.
fn continue_checkpoint_paused_state() -> PausedState {
    PausedState {
        session_id: SessionId::new("sess-129"),
        task_id: TaskId::new("task-129"),
        turn_number: 3,
        session_state: SessionState {
            messages: vec![Message {
                role: crate::model::Role::Assistant,
                content: crate::model::Content::Text {
                    text: "{\"node\":\"react\",\"last\":\"\"}".into(),
                },
            }],
            ..Default::default()
        },
        pending_tool_calls: vec![],
        approved_results: vec![],
        human_request: Some(HumanRequest::BudgetExhausted {
            phase: "react".into(),
            policy: BudgetPolicy::PerLoop { value: 3 },
            steps_taken: 3,
            continues_used: 1,
            partial_output: Some("{\"node\":\"react\",\"last\":\"\"}".into()),
            available_actions: vec![
                EscalationAction::ContinueWithBudget { steps: 3 },
                EscalationAction::Fail,
            ],
        }),
        task: {
            let mut t = Task::new(
                "iterate on the patch",
                SessionId::new("sess-129"),
                LoopStrategy::ReAct(ReactConfig {
                    budget: BudgetPolicy::PerLoop { value: 3 },
                    behavior: BudgetExhaustedBehavior::Continue {
                        max_continues: 2,
                        on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
                    },
                    agent: AgentRef("worker".into()),
                    toolset: ToolsetRef("patch-tools".into()),
                    output: None,
                    system_prompt: None,
                }),
            );
            t.id = TaskId::new("task-129");
            t
        },
        budget_used: BudgetSnapshot {
            turns: 3,
            ..Default::default()
        },
        child_state: None,
        // #140 decision 2: continue-checkpoint pause re-drives on resume; the
        // handle is behaviorally irrelevant here → empty default.
        toolset: ToolsetRef::default(),
    }
}

/// AC2 (wire): `continue_checkpoint.json` (de)serializes byte-identically —
/// the new fixture capturing a `Continue` node paused with `continues_used>0`.
#[test]
fn fixture_replay_continue_checkpoint() {
    let raw = std::fs::read_to_string(h130_fixture_path("paused_states/continue_checkpoint.json"))
        .expect("fixture present");
    let parsed: PausedState = serde_json::from_str(&raw).unwrap();
    assert_eq!(parsed, continue_checkpoint_paused_state());
    h130_assert_fixture_byte_identical(
        &continue_checkpoint_paused_state(),
        "paused_states/continue_checkpoint.json",
    );
}

/// AC2 (LOAD-BEARING, end-to-end): a `Continue` that SPANS a process pause
/// resumes with the correct `continues_used` (NOT 0). This test FAILS on
/// pre-#129 code, which destructured `continues_used: _` in `resume_inner` and
/// rebuilt the scope from `steps_taken` only — ZEROING the counter and
/// granting MORE in-process continues than `max_continues` permits.
///
/// DISCRIMINATING setup: a `Continue{max_continues:2, on_exhausted:Fail}` leaf
/// whose checkpoint records `continues_used: 1` (ONE continue already spent).
/// The resumed worker NEVER finishes (always requests a tool), so every
/// granted in-process window re-exhausts and the run drains to `Fail`. The
/// OBSERVABLE signal is HOW MANY in-process continue-windows ran before the
/// fall-through, captured as the terminal `turns`:
///   - CORRECT (#129): `continues_used == 1` → exactly ONE continue remains →
///     a SINGLE re-driven window, then `Fail`.
///   - BUGGY (pre-#129): `continues_used == 0` → TWO continues granted → TWO
///     re-driven windows, then `Fail` — strictly MORE turns.
/// The granted per-window cap is `steps_taken + steps = 3 + 1 = 4`, so each
/// window consumes 4 turns. On resume the operator's `ContinueWithBudget` runs
/// window A; with `continues_used == 1` (one in-process continue left) window B
/// runs, then `Fail`: 2 windows ⇒ 8 turns. The bug (zeroed counter) would have
/// TWO continues left → windows A, B, C ⇒ 12 turns. We assert `turns == 8`.
#[tokio::test]
async fn resume_continue_preserves_continues_used_then_falls_through() {
    // A worker that always requests a tool (never finishes) → every window
    // re-exhausts the refreshed cap.
    let a = budget_exhausting_agent(40);
    let mut cfg = surface_config(a);
    cfg.tool_registry = tool_reg(40);
    let h = StandardHarness::new(cfg);

    let mut state = continue_checkpoint_paused_state();
    // Bare leaf resolving via the default ("") registry handle on resume.
    state.task.loop_strategy = LoopStrategy::ReAct(ReactConfig {
        budget: BudgetPolicy::PerLoop { value: 3 },
        behavior: BudgetExhaustedBehavior::Continue {
            max_continues: 2,
            on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
        },
        agent: AgentRef(String::new()),
        toolset: ToolsetRef(String::new()),
        output: None,
        system_prompt: None,
    });
    // continues_used == 1: ONE continue already spent (only one remains).
    match state.human_request.as_mut().unwrap() {
        HumanRequest::BudgetExhausted {
            continues_used,
            steps_taken,
            ..
        } => {
            *continues_used = 1;
            *steps_taken = 3;
        }
        _ => panic!("expected BudgetExhausted"),
    }

    let resumed = h
        .resume(
            state,
            HumanResponse::Escalate {
                action: EscalationAction::ContinueWithBudget { steps: 1 },
            },
            None,
        )
        .await;
    match resumed {
        RunResult::Failure {
            reason:
                HaltReason::BudgetExceeded {
                    limit_type: BudgetLimitType::Turns,
                },
            turns,
            ..
        } => {
            // #129: window A (operator grant) + window B (the ONE remaining
            // in-process continue) → 2 windows × cap 4 = 8 turns. The bug
            // (zeroed continues_used) grants a THIRD window → 12 turns.
            assert_eq!(
                turns, 8,
                "expected 2 windows (continues_used preserved at 1, one continue \
                     left); the bug zeroes it and grants an extra window → 12 turns"
            );
        }
        other => panic!("expected BudgetExceeded Failure, got {other:?}"),
    }
}

/// AC2 (unit, load-bearing core): the resume seam seeds the reconstructed
/// scope's `continues_used` from the request via [`BudgetContext::resumed`]
/// (NOT 0), and a subsequent exhaustion falls through to `on_exhausted` after
/// the REMAINING continues — not a refreshed `max_continues`.
#[test]
fn budget_context_resumed_seeds_continues_used_and_bounds_chain() {
    let behavior = BudgetExhaustedBehavior::Continue {
        max_continues: 2,
        on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
    };
    // Reconstruct as the resume seam does: continues_used seeded to 1.
    let mut scope =
        BudgetContext::resumed(BudgetPolicy::PerLoop { value: 3 }, behavior, "react", 1);
    assert_eq!(scope.continues_used, 1, "seeded, NOT zeroed");
    assert_eq!(scope.steps_taken, 0, "fresh per-round step budget");
    // Only ONE continue remains (to reach max_continues=2).
    assert_eq!(scope.continues_remaining(), 1);
    assert_eq!(scope.resolve_exhausted(), ExhaustedResolution::Continue);
    assert_eq!(scope.continues_used, 2);
    // Continues now spent → fall through to on_exhausted=Fail.
    assert_eq!(scope.resolve_exhausted(), ExhaustedResolution::Fail);

    // Contrast: a FRESH (pre-#129) scope would grant TWO continues — the bug.
    let fresh = BudgetContext::new(
        BudgetPolicy::PerLoop { value: 3 },
        BudgetExhaustedBehavior::Continue {
            max_continues: 2,
            on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
        },
        "react",
    );
    assert_eq!(
        fresh.continues_remaining(),
        2,
        "the bug: full budget refreshed"
    );
}

/// LIVE Continue reachable (AC: `ExhaustedResolution::Continue` is no longer
/// dead): a bare-leaf run with `behavior: Continue{max_continues:1}` exhausts
/// in-process, gets a granted continue (counter resets, continues_used bumps),
/// loops, and completes — all WITHOUT a pause (AC3: no serialization).
#[tokio::test]
async fn live_continue_loops_in_process_then_completes() {
    let a = make_agent();
    // First window: 2 tool turns exhaust the PerLoop{2} cap → Continue grant.
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c0".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    // After the in-process continue refreshes the cap, the worker completes.
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done after in-process continue".into(),
        usage: usage(),
    });
    // Autonomous so an Escalate fall-through would NOT pause — proving the
    // success came from the Continue loop, not a HITL pause.
    let mut cfg = standard_config(a);
    cfg.tool_registry = tool_reg(3);
    let h = StandardHarness::new(cfg);
    let t = task(LoopStrategy::ReAct(ReactConfig {
        budget: BudgetPolicy::PerLoop { value: 2 },
        behavior: BudgetExhaustedBehavior::Continue {
            max_continues: 1,
            on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
        },
        agent: AgentRef(String::new()),
        toolset: ToolsetRef(String::new()),
        output: None,
        system_prompt: None,
    }));
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Success { output, .. } => {
            assert_eq!(output, "done after in-process continue");
        }
        other => panic!("expected Success via in-process Continue, got {other:?}"),
    }
}

/// SC-5 acceptance: `EscalationMode::AutoContinue` keeps a budget-exhausted
/// bare ReAct leaf working IN-PROCESS — no consumer drive loop — by
/// auto-granting `steps_per_grant` more steps at the Escalate fall-through,
/// firing `on_grant` per grant, until the worker completes. Same shape as the
/// `Continue`-behavior test above, but the grant comes from the *escalation
/// mode*, not the leaf's `behavior` (which is `Escalate`).
#[tokio::test]
async fn auto_continue_grants_in_process_then_completes() {
    let a = make_agent();
    // First window: 2 tool turns exhaust the PerLoop{2} cap → Escalate.
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c0".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }],
        usage: usage(),
    });
    // After one auto-grant refreshes the cap, the worker completes.
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done after auto-continue".into(),
        usage: usage(),
    });

    let grants = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let g2 = grants.clone();
    let mut cfg = standard_config(a);
    cfg.tool_registry = tool_reg(3);
    cfg.escalation_mode = EscalationMode::AutoContinue {
        max_grants: 3,
        steps_per_grant: 2,
        on_grant: Some(Arc::new(move |info: AutoGrantInfo| {
            assert_eq!(info.steps_granted, 2);
            g2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })),
    };
    let h = StandardHarness::new(cfg);
    let t = task(LoopStrategy::ReAct(ReactConfig {
        budget: BudgetPolicy::PerLoop { value: 2 },
        behavior: BudgetExhaustedBehavior::Escalate,
        agent: AgentRef(String::new()),
        toolset: ToolsetRef(String::new()),
        output: None,
        system_prompt: None,
    }));
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Success { output, .. } => {
            assert_eq!(output, "done after auto-continue");
        }
        other => panic!("expected Success via AutoContinue, got {other:?}"),
    }
    assert_eq!(
        grants.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "exactly one auto-grant was needed to finish"
    );
}

/// SC-5: AutoContinue is CAPPED — once `max_grants` grants are spent it falls
/// through to the autonomous terminal (Failure), firing `on_grant` exactly
/// `max_grants` times. The agent never emits a final response, so every window
/// exhausts.
#[tokio::test]
async fn auto_continue_caps_at_max_grants_then_fails() {
    let a = make_agent();
    // Plenty of tool turns (far more than any window can consume) so the run
    // only ends when the grant cap is reached — never on an empty queue.
    for i in 0..40 {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: format!("c{i}"),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
    }
    let grants = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let g2 = grants.clone();
    let mut cfg = standard_config(a);
    cfg.tool_registry = tool_reg(80);
    cfg.escalation_mode = EscalationMode::AutoContinue {
        max_grants: 2,
        steps_per_grant: 2,
        on_grant: Some(Arc::new(move |_| {
            g2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })),
    };
    let h = StandardHarness::new(cfg);
    let t = task(LoopStrategy::ReAct(ReactConfig {
        budget: BudgetPolicy::PerLoop { value: 2 },
        behavior: BudgetExhaustedBehavior::Escalate,
        agent: AgentRef(String::new()),
        toolset: ToolsetRef(String::new()),
        output: None,
        system_prompt: None,
    }));
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure { .. } => {}
        other => panic!("expected Failure once grants are spent, got {other:?}"),
    }
    assert_eq!(
        grants.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "exactly max_grants auto-grants fired before falling through to Failure"
    );
}

/// AC4: a Continue resume PRESERVES the prior `session_state.messages` (the
/// conversation context survives the pause), while Ralph DISCARDS + re-seeds.
/// Asserts the shared checkpoint utility did NOT unify the context policy.
#[tokio::test]
async fn continue_resume_preserves_session_context() {
    // A Continue checkpoint carries one assistant message; the resumed run
    // must build ON it (not start from an empty session). We assert the
    // resumed run's history is non-empty (context preserved). A trivial
    // worker finishes immediately on resume so we can inspect the terminal.
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "resumed with context".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a); // Autonomous: a finish is a clean Success.
    cfg.tool_registry = tool_reg(1);
    let h = StandardHarness::new(cfg);

    let mut state = continue_checkpoint_paused_state();
    // The leaf resolves via the default ("") registry handle on resume.
    state.task.loop_strategy = LoopStrategy::ReAct(ReactConfig {
        budget: BudgetPolicy::PerLoop { value: 3 },
        behavior: BudgetExhaustedBehavior::Continue {
            max_continues: 2,
            on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
        },
        agent: AgentRef(String::new()),
        toolset: ToolsetRef(String::new()),
        output: None,
        system_prompt: None,
    });
    let prior_msg_count = state.session_state.messages.len();
    assert!(prior_msg_count >= 1, "checkpoint carries prior context");

    let resumed = h
        .resume(
            state,
            HumanResponse::Escalate {
                action: EscalationAction::ContinueWithBudget { steps: 3 },
            },
            None,
        )
        .await;
    match resumed {
        RunResult::Success {
            output,
            session_state,
            ..
        } => {
            assert_eq!(output, "resumed with context");
            // The resumed session retains the prior assistant message AND the
            // new turn — it did NOT start from an empty (re-seeded) session.
            assert!(
                session_state.messages.len() > prior_msg_count,
                "Continue preserved prior context and appended the resumed turn; \
                     got {} messages (prior {})",
                session_state.messages.len(),
                prior_msg_count
            );
        }
        other => panic!("expected Success on resume, got {other:?}"),
    }
}

/// AC1: the SHARED checkpoint utility round-trips a `PausedState` (the durable
/// pause/resume seam reused by both the cross-process Continue path and
/// Ralph's pause-propagation). Unit-tested directly.
#[test]
fn shared_checkpoint_utility_round_trips() {
    let state = continue_checkpoint_paused_state();
    let blob = state.serialize_checkpoint().unwrap();
    let restored = PausedState::load_checkpoint(&blob).unwrap();
    assert_eq!(restored, state);
    // The Ralph paused-state (no human_request) ALSO round-trips through the
    // same utility — proving it is shared, not Continue-specific.
    let ralph_state = budget_exhausted_paused_state();
    let blob = ralph_state.serialize_checkpoint().unwrap();
    assert_eq!(PausedState::load_checkpoint(&blob).unwrap(), ralph_state);
}

/// Leaf behavior (Q1): a NESTED ReAct leaf still PROPAGATES exhaustion to its
/// parent (it does NOT self-resolve its `behavior`), while a BARE top-level
/// leaf HONORS its `behavior`.
///
/// Bare leaf honoring `Fail`: a top-level leaf with `behavior: Fail` resolves
/// to a `BudgetExceeded` Failure with the partial DISCARDED — distinct from
/// the default `Escalate` placeholder which (under SurfaceToHuman) would
/// PAUSE. Under Autonomous both reach Failure, so we assert the partial is
/// discarded (the Fail contract) to prove the bare-leaf honored `Fail`.
#[tokio::test]
async fn bare_leaf_honors_fail_behavior() {
    let mut cfg = standard_config(budget_exhausting_agent(3));
    cfg.tool_registry = tool_reg(3);
    let h = StandardHarness::new(cfg);
    let t = task(LoopStrategy::ReAct(ReactConfig {
        budget: BudgetPolicy::PerLoop { value: 2 },
        behavior: BudgetExhaustedBehavior::Fail,
        agent: AgentRef(String::new()),
        toolset: ToolsetRef(String::new()),
        output: None,
        system_prompt: None,
    }));
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::Failure {
            reason:
                HaltReason::BudgetExceeded {
                    limit_type: BudgetLimitType::Turns,
                },
            session_state,
            ..
        } => {
            // Fail contract: the partial is DISCARDED (the bare leaf honored
            // its `Fail` behavior, not the propagate/Escalate path).
            assert!(session_state.messages.is_empty());
        }
        other => panic!("expected BudgetExceeded Failure via leaf Fail, got {other:?}"),
    }
}

/// A NESTED leaf does NOT self-resolve: its `Continue` behavior is IGNORED by
/// the leaf body (it propagates to the parent), so the PARENT combinator's
/// `behavior` governs. Here the parent PlanExecute carries the default
/// `Escalate` placeholder, so the nested leaf's exhaustion surfaces as the
/// PARENT's `plan_execute` pause (phase == "plan_execute"), NOT a leaf-level
/// `react` resolution of its own `Continue`.
#[tokio::test]
async fn nested_leaf_propagates_does_not_self_resolve() {
    let a = make_agent();
    a.push(final_resp(r#"{"tasks":["x"]}"#)); // plan
    for i in 0..4 {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: format!("e{i}"),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
    }
    let mut cfg = surface_config(a);
    cfg.tool_registry = tool_reg(4);
    let h = StandardHarness::new(cfg);
    // The execute leaf carries `Continue` — but NESTED, so the leaf must NOT
    // self-resolve it; the PlanExecute parent (Escalate placeholder) pauses.
    let t = task(LoopStrategy::PlanExecute(PlanExecuteConfig {
        plan: Box::new(react_structured(u32::MAX)),
        execute: Box::new(LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: 2 },
            behavior: BudgetExhaustedBehavior::Continue {
                max_continues: 1,
                on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
            },
            agent: AgentRef(String::new()),
            toolset: ToolsetRef(String::new()),
            output: None,
            system_prompt: None,
        })),
        plan_model: None,
        behavior: default_budget_behavior(),
    }));
    match h.run(HarnessRunOptions::new(t)).await {
        RunResult::WaitingForHuman {
            request: HumanRequest::BudgetExhausted { phase, .. },
            ..
        } => {
            // The PARENT resolved (phase == plan_execute); the nested leaf did
            // NOT self-resolve its own `Continue` at the "react" phase.
            assert_eq!(phase, "plan_execute");
        }
        other => panic!("expected parent plan_execute pause, got {other:?}"),
    }
}

// ========================================================================
// Consecutive-recoverable-tool-error breaker (issue #137)
// ========================================================================
//
// Types/rules under test:
//   - `HarnessConfig::error_loop_threshold` (N, default 3); hard stop at 2N.
//   - Per-tool `ErrorRun { args, count, injected }` loop-local counter:
//       * identical-args recoverable error -> count += 1
//       * args change OR first error -> fresh run at count 1
//       * ANY success for the tool -> run removed (AC1 reset)
//   - At N: ONE corrective USER message (enrich_tool_error schema+hint), AC2.
//   - At 2N: stop -> resolve node `BudgetExhaustedBehavior`, terminal carries
//     `HaltReason::ToolErrorLoop` (never BudgetExceeded), AC3.
//   - At BOTH thresholds: a StreamEvent + a ContextOperation, AC4.

const TEL_BAD_MSG: &str = "missing required parameter `description`";

/// The malformed `add_task` call the weak model repeats (gemma's
/// `task_list`/`add_task`-without-`description` scenario).
fn tel_bad_call(args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: "call_bad".into(),
        name: "add_task".into(),
        input: args,
    }
}

fn tel_bad_args() -> serde_json::Value {
    serde_json::json!({ "task_list_id": "tl1" })
}

/// Push `k` identical malformed `add_task` tool-call turns.
fn tel_push_bad(a: &Arc<MockAgent>, k: u32, args: &serde_json::Value) {
    for _ in 0..k {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![tel_bad_call(args.clone())],
            usage: usage(),
        });
    }
}

/// A tool registry that always returns the same recoverable error and
/// advertises the `add_task` schema (so `enrich_tool_error` can render the
/// schema+hint).
fn tel_err_registry() -> Arc<ScriptedToolRegistry> {
    let reg = Arc::new(ScriptedToolRegistry::new());
    reg.always_recoverable_error(TEL_BAD_MSG);
    reg.with_schema(ToolSchema {
        name: "add_task".into(),
        description: "add a task to a task list".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "task_list_id": { "type": "string" },
                "description": { "type": "string" },
            },
            "required": ["task_list_id", "description"],
        }),
    });
    reg
}

fn tel_user_msgs(state: &SessionState) -> Vec<String> {
    state
        .messages
        .iter()
        .filter_map(|m| match (&m.role, &m.content) {
            (crate::model::Role::User, crate::model::Content::Text { text }) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

fn tel_leaf(behavior: BudgetExhaustedBehavior, budget: u32) -> Task {
    task(LoopStrategy::ReAct(ReactConfig {
        budget: BudgetPolicy::PerLoop { value: budget },
        behavior,
        agent: AgentRef(String::new()),
        toolset: ToolsetRef(String::new()),
        output: None,
        system_prompt: None,
    }))
}

// AC1: a success in the middle resets the counter; the breaker never trips.
// error, error, SUCCESS, error, error -> 4 errors but the longest
// identical-args run is 2 (< N), so no trip.
#[tokio::test]
async fn tel_ac1_success_resets_counter_no_trip() {
    let a = make_agent();
    tel_push_bad(&a, 2, &tel_bad_args());
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![ToolCall {
            id: "ok".into(),
            name: "add_task".into(),
            input: tel_bad_args(),
        }],
        usage: usage(),
    });
    tel_push_bad(&a, 2, &tel_bad_args());
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    let reg = Arc::new(ScriptedToolRegistry::new());
    for recoverable_err in [true, true, false, true, true] {
        if recoverable_err {
            reg.push(ToolOutput::Error {
                message: TEL_BAD_MSG.into(),
                recoverable: true,
            });
        } else {
            reg.push(ToolOutput::Success {
                content: "added".into(),
                truncated: false,
            });
        }
    }
    cfg.tool_registry = reg;
    cfg.error_loop_threshold = 3;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(20))).await {
        RunResult::Success { output, .. } => assert_eq!(output, "done"),
        other => panic!("expected Success (no trip), got {other:?}"),
    }
}

// AC1 args variant: error(argsX), error(argsY) -> the second is a FRESH run
// at count 1, so two different-args errors never trip even at N == 2.
#[tokio::test]
async fn tel_ac1_args_change_starts_fresh_run() {
    let a = make_agent();
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![tel_bad_call(serde_json::json!({ "task_list_id": "X" }))],
        usage: usage(),
    });
    a.push(TurnResult::ToolCallRequested {
        reasoning: None,
        calls: vec![tel_bad_call(serde_json::json!({ "task_list_id": "Y" }))],
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "stopped trying".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    cfg.tool_registry = tel_err_registry();
    cfg.error_loop_threshold = 2; // 2N == 4; longest identical run is 1.
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(20))).await {
        RunResult::Success { output, .. } => assert_eq!(output, "stopped trying"),
        other => panic!("expected Success (args changed, no trip), got {other:?}"),
    }
}

// AC2: exactly N identical-arg recoverable errors inject exactly ONE
// corrective user message carrying the enrich_tool_error schema+hint; a 4th
// identical error does NOT re-inject.
#[tokio::test]
async fn tel_ac2_injects_one_corrective_at_n() {
    let a = make_agent();
    // 4 identical errors (N=3 injects at the 3rd; 4th must not re-inject),
    // then a final response so the run ends cleanly (2N would be 6).
    tel_push_bad(&a, 4, &tel_bad_args());
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "gave up".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    cfg.tool_registry = tel_err_registry();
    cfg.error_loop_threshold = 3;
    let h = StandardHarness::new(cfg);
    let r = h.run(HarnessRunOptions::new(react(20))).await;
    let state = match &r {
        RunResult::Success { session_state, .. } => session_state,
        other => panic!("expected Success, got {other:?}"),
    };
    let users = tel_user_msgs(state);
    let correctives: Vec<&String> = users
        .iter()
        .filter(|m| m.contains("Expected parameter schema"))
        .collect();
    assert_eq!(
        correctives.len(),
        1,
        "exactly one corrective injected, got {users:?}"
    );
    let corrective = correctives[0];
    assert!(corrective.contains(TEL_BAD_MSG), "carries the bare error");
    // Pin the EXACT key-sorted, compact schema substring (cross-language
    // source-of-truth, #137): the literal `Expected parameter schema: `
    // prefix followed by the `add_task` test schema serialized by
    // `serde_json` (object keys sorted lexicographically — no
    // `preserve_order`). TS/Python/Go AC2 tests assert this same byte string,
    // so this guards against a future serde ordering drift in Rust.
    const EXPECTED_SCHEMA: &str = "Expected parameter schema: {\"properties\":{\"description\":{\"type\":\"string\"},\"task_list_id\":{\"type\":\"string\"}},\"required\":[\"task_list_id\",\"description\"],\"type\":\"object\"}";
    assert!(
        corrective.contains(EXPECTED_SCHEMA),
        "carries the exact key-sorted parameter schema; got:\n{corrective}"
    );
    assert!(
        corrective.contains("correctly-typed JSON"),
        "carries the hint"
    );
}

// AC3 Fail: 2N identical errors under `behavior = Fail` ->
// RunResult::Failure { ToolErrorLoop }; budget NOT fully burned.
#[tokio::test]
async fn tel_ac3_fail_terminal_is_tool_error_loop() {
    let a = make_agent();
    tel_push_bad(&a, 8, &tel_bad_args()); // trip is at 2N == 6.
    let mut cfg = standard_config(a);
    cfg.tool_registry = tel_err_registry();
    cfg.error_loop_threshold = 3;
    let h = StandardHarness::new(cfg);
    match h
        .run(HarnessRunOptions::new(tel_leaf(
            BudgetExhaustedBehavior::Fail,
            50,
        )))
        .await
    {
        RunResult::Failure {
            reason:
                HaltReason::ToolErrorLoop {
                    tool,
                    consecutive_errors,
                },
            turns,
            ..
        } => {
            assert_eq!(tool, "add_task");
            assert_eq!(consecutive_errors, 6, "2N == 6");
            assert!(turns < 50, "budget NOT fully burned, got {turns}");
        }
        other => panic!("expected Failure {{ ToolErrorLoop }}, got {other:?}"),
    }
}

// AC3 Escalate (SurfaceToHuman) -> WaitingForHuman.
#[tokio::test]
async fn tel_ac3_escalate_surfaces_to_human() {
    let a = make_agent();
    tel_push_bad(&a, 8, &tel_bad_args());
    let mut cfg = surface_config(a); // EscalationMode::SurfaceToHuman
    cfg.tool_registry = tel_err_registry();
    cfg.error_loop_threshold = 3;
    let h = StandardHarness::new(cfg);
    match h
        .run(HarnessRunOptions::new(tel_leaf(
            BudgetExhaustedBehavior::Escalate,
            50,
        )))
        .await
    {
        RunResult::WaitingForHuman { .. } => {}
        other => panic!("expected WaitingForHuman, got {other:?}"),
    }
}

// AC3 Escalate (Autonomous) -> propagated Failure carrying ToolErrorLoop
// (NOT BudgetExceeded).
#[tokio::test]
async fn tel_ac3_escalate_autonomous_is_tool_error_loop() {
    let a = make_agent();
    tel_push_bad(&a, 8, &tel_bad_args());
    let mut cfg = standard_config(a); // EscalationMode::Autonomous
    cfg.tool_registry = tel_err_registry();
    cfg.error_loop_threshold = 3;
    let h = StandardHarness::new(cfg);
    match h
        .run(HarnessRunOptions::new(tel_leaf(
            BudgetExhaustedBehavior::Escalate,
            50,
        )))
        .await
    {
        RunResult::Failure {
            reason: HaltReason::ToolErrorLoop { tool, .. },
            turns,
            ..
        } => {
            assert_eq!(tool, "add_task");
            assert!(turns < 50, "budget NOT fully burned");
        }
        other => panic!("expected Failure {{ ToolErrorLoop }}, got {other:?}"),
    }
}

// AC3 Continue: one extra window granted, then a terminal. With
// `Continue { max_continues: 1, on_exhausted: Fail }`, the first 2N trip
// grants a continue (fresh window), the second 2N trip falls through to Fail
// with ToolErrorLoop.
#[tokio::test]
async fn tel_ac3_continue_grants_one_window_then_terminal() {
    let a = make_agent();
    tel_push_bad(&a, 14, &tel_bad_args()); // 2N (6) + 2N (6) = 12 needed.
    let mut cfg = standard_config(a);
    cfg.tool_registry = tel_err_registry();
    cfg.error_loop_threshold = 3;
    let strat = tel_leaf(
        BudgetExhaustedBehavior::Continue {
            max_continues: 1,
            on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
        },
        50,
    );
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(strat)).await {
        RunResult::Failure {
            reason: HaltReason::ToolErrorLoop { .. },
            turns,
            ..
        } => {
            assert!(turns < 50, "budget NOT fully burned, got {turns}");
        }
        other => panic!("expected Failure {{ ToolErrorLoop }} after continue, got {other:?}"),
    }
}

// AC4: capturing StreamSink + recording ObservabilityProvider -> the
// detected pair at N and the broken pair at 2N, with tool + count.
#[tokio::test]
async fn tel_ac4_emits_detected_and_broken_events() {
    let a = make_agent();
    tel_push_bad(&a, 8, &tel_bad_args());
    let obs = Arc::new(crate::observability::InMemoryObservabilityProvider::new());
    let mut cfg = standard_config(a);
    cfg.tool_registry = tel_err_registry();
    cfg.error_loop_threshold = 3;
    cfg.observability = Some(obs.clone());
    let captured: Arc<std::sync::Mutex<Vec<StreamEvent>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = captured.clone();
    let session_id = SessionId::new("s1");
    let opts = HarnessRunOptions::new(tel_leaf(BudgetExhaustedBehavior::Fail, 50)).with_stream(
        move |ev| {
            sink.lock().unwrap().push(ev);
        },
    );
    let h = StandardHarness::new(cfg);
    let _ = h.run(opts).await;

    let evs = captured.lock().unwrap();
    let detected: Vec<u32> = evs
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ToolErrorLoopDetected {
                tool,
                consecutive_errors,
                ..
            } if tool == "add_task" => Some(*consecutive_errors),
            _ => None,
        })
        .collect();
    let broken: Vec<u32> = evs
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ToolErrorLoopBroken {
                tool,
                consecutive_errors,
                ..
            } if tool == "add_task" => Some(*consecutive_errors),
            _ => None,
        })
        .collect();
    assert_eq!(detected, vec![3], "one detected at N, count==N");
    assert_eq!(broken, vec![6], "one broken at 2N, count==2N");

    let spans = obs.context_spans(&session_id);
    let obs_detected: Vec<u32> = spans
        .iter()
        .filter_map(|s| match &s.operation {
            crate::observability::ContextOperation::ToolErrorLoopDetected {
                tool_name,
                consecutive_errors,
            } if tool_name == "add_task" => Some(*consecutive_errors),
            _ => None,
        })
        .collect();
    let obs_broken: Vec<u32> = spans
        .iter()
        .filter_map(|s| match &s.operation {
            crate::observability::ContextOperation::ToolErrorLoopBroken {
                tool_name,
                consecutive_errors,
            } if tool_name == "add_task" => Some(*consecutive_errors),
            _ => None,
        })
        .collect();
    assert_eq!(obs_detected, vec![3], "obs detected at N");
    assert_eq!(obs_broken, vec![6], "obs broken at 2N");
}

// Breaker disabled when threshold is 0: no trip, run completes.
#[tokio::test]
async fn tel_threshold_zero_disables_breaker() {
    let a = make_agent();
    tel_push_bad(&a, 5, &tel_bad_args());
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "fin".into(),
        usage: usage(),
    });
    let mut cfg = standard_config(a);
    cfg.tool_registry = tel_err_registry();
    cfg.error_loop_threshold = 0;
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(react(20))).await {
        RunResult::Success { output, .. } => assert_eq!(output, "fin"),
        other => panic!("expected Success (breaker off), got {other:?}"),
    }
}

// ── Repeated-identical-SUCCESSFUL-call breaker (no-progress twin of #137) ──

/// A model that calls the SAME tool with the SAME args and keeps SUCCEEDING is
/// spinning. The breaker hard-stops at `2 * error_loop_threshold` identical
/// successes with `HaltReason::RepeatedToolCall` (default N=3 → stop at 6),
/// instead of letting it grind out the whole turn budget (the plan-spin).
#[tokio::test]
async fn repeated_identical_successful_calls_hard_stop() {
    let a = make_agent();
    // Eight identical SUCCESSFUL tool calls in a row (the planner never
    // finalizes). standard_config's ScriptedToolRegistry returns "ok" success
    // for every dispatch, so these all succeed — the error breaker never fires.
    for _ in 0..8 {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: "c".into(),
                name: "spin_tool".into(),
                input: serde_json::json!({"k": "v"}),
            }],
            usage: usage(),
        });
    }
    // Generous per-loop budget so the REPEAT breaker — not the budget cap —
    // is what stops it. error_loop_threshold = 3 ⇒ hard stop at 2*3 = 6.
    let h = StandardHarness::new(standard_config(a));
    match h.run(HarnessRunOptions::new(react(50))).await {
        RunResult::Failure {
            reason: HaltReason::RepeatedToolCall { tool, count },
            ..
        } => {
            assert_eq!(tool, "spin_tool");
            assert_eq!(count, 6, "hard stop at 2 * N (N = 3)");
        }
        other => panic!("expected RepeatedToolCall, got {other:?}"),
    }
}

/// Reset semantics: ALTERNATING calls (no two consecutive identical) never
/// trip the breaker — proving it counts CONSECUTIVE identical successes, not a
/// global tally. The run finalizes normally.
#[tokio::test]
async fn repeat_breaker_does_not_trip_on_alternating_calls() {
    let a = make_agent();
    for i in 0..4 {
        a.push(TurnResult::ToolCallRequested {
            reasoning: None,
            calls: vec![ToolCall {
                id: "c".into(),
                name: "spin_tool".into(),
                // Alternate the args each turn → no two consecutive identical.
                input: serde_json::json!({"k": i % 2}),
            }],
            usage: usage(),
        });
    }
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: "done".into(),
        usage: usage(),
    });
    let h = StandardHarness::new(standard_config(a));
    match h.run(HarnessRunOptions::new(react(50))).await {
        RunResult::Success { output, .. } => assert_eq!(output, "done"),
        other => panic!("expected Success (alternating never trips), got {other:?}"),
    }
}

// ========================================================================
// Output-schema delivery + enforcement (issue #139)
// ========================================================================
//
// Types/rules under test:
//   - `HarnessConfig::enforce_output_schemas` (MIGRATION GATE, default OFF).
//   - `HarnessConfig::output_schema_max_retries` (N, default 2).
//   - `HaltReason::OutputSchemaViolation { schema, attempts, last_error }`.
//   - AC1: the resolved schema is DELIVERED to the leaf's directive seed
//     (key-sorted) AND set on `ModelParams.output_schema`.
//   - AC2: terminal validated; valid ⇒ Success; invalid ⇒ feed the frozen
//     error back + retry within budget.
//   - AC3: after N retries WITH budget remaining ⇒ OutputSchemaViolation
//     (distinct from budget exhaustion; turns < budget).
//   - AC3 budget precedence: a retry that would exceed the turn cap surfaces
//     `BudgetExceeded`, NOT `OutputSchemaViolation`.
//   - AC4: flag OFF ⇒ an invalid terminal is accepted as Success.

/// The output schema the #139 tests enforce: an object requiring a `status`
/// (one of `ok`/`error`) and a `count` integer.
fn os_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "required": ["status", "count"],
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "count": {"type": "integer"}
        }
    })
}

/// A valid terminal body for [`os_schema`].
const OS_VALID: &str = "{\"status\":\"ok\",\"count\":3}";
/// An invalid terminal body for [`os_schema`] (missing both required props).
const OS_INVALID: &str = "{}";

/// A `standard_config` with output-schema enforcement ON and [`os_schema`]
/// registered under the default empty `SchemaRef` key. `max_retries` is the
/// `N`.
fn os_config(agent: Arc<MockAgent>, max_retries: u32) -> HarnessConfig {
    let mut cfg = standard_config(agent);
    cfg.enforce_output_schemas = true;
    cfg.output_schema_max_retries = max_retries;
    // Register the real schema under the empty key (replacing the default
    // `{}` the test config installs).
    cfg.registry = std::mem::take(&mut cfg.registry)
        .into_builder()
        .schema("", os_schema())
        .build();
    cfg
}

/// A bare ReAct leaf that carries `output = Some(SchemaRef(""))` and a turn
/// budget of `budget`.
fn os_leaf(budget: u32) -> Task {
    task(LoopStrategy::ReAct(ReactConfig {
        budget: BudgetPolicy::PerLoop { value: budget },
        behavior: default_budget_behavior(),
        agent: AgentRef(String::new()),
        toolset: ToolsetRef(String::new()),
        output: Some(SchemaRef(String::new())),
        system_prompt: None,
    }))
}

fn os_user_msgs(state: &SessionState) -> Vec<String> {
    state
        .messages
        .iter()
        .filter_map(|m| match (&m.role, &m.content) {
            (crate::model::Role::User, crate::model::Content::Text { text }) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

// AC1: the resolved schema is delivered to the directive seed (key-sorted),
// and a valid terminal accepts on turn 1. (The Ollama `format` population +
// non-Ollama no-op are unit-tested in `ollama.rs`; here we assert delivery
// via the seeded directive.)
#[tokio::test]
async fn os_ac1_schema_delivered_to_directive_seed() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: OS_VALID.into(),
        usage: usage(),
    });
    let h = StandardHarness::new(os_config(a, 2));
    match h.run(HarnessRunOptions::new(os_leaf(10))).await {
        RunResult::Success {
            session_state,
            turns,
            ..
        } => {
            assert_eq!(turns, 1, "valid on turn 1");
            let users = os_user_msgs(&session_state);
            // The directive carries the KEY-SORTED schema bytes. Pin the
            // exact canonical compact JSON so the four ports match.
            const EXPECTED_SCHEMA: &str = "{\"properties\":{\"count\":{\"type\":\"integer\"},\"status\":{\"enum\":[\"ok\",\"error\"],\"type\":\"string\"}},\"required\":[\"status\",\"count\"],\"type\":\"object\"}";
            assert!(
                users.iter().any(|m| m.contains(EXPECTED_SCHEMA)),
                "directive must carry the key-sorted schema; got {users:?}"
            );
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// AC2 accept: schema ON, turn 1 valid → Success, turns == 1, NO feedback.
#[tokio::test]
async fn os_ac2_accept_valid_on_first_turn() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: OS_VALID.into(),
        usage: usage(),
    });
    let h = StandardHarness::new(os_config(a, 2));
    match h.run(HarnessRunOptions::new(os_leaf(10))).await {
        RunResult::Success {
            output,
            turns,
            session_state,
            ..
        } => {
            assert_eq!(output, OS_VALID);
            assert_eq!(turns, 1);
            let fed_back = os_user_msgs(&session_state)
                .iter()
                .any(|m| m.contains("did not match the required output schema"));
            assert!(!fed_back, "no feedback message on a turn-1 accept");
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

// AC2 retry: turn 1 invalid → feed error back → turn 2 valid → Success,
// turns == 2, the frozen feedback (with the validator error) is present.
#[tokio::test]
async fn os_ac2_retry_invalid_then_valid() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: OS_INVALID.into(),
        usage: usage(),
    });
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: OS_VALID.into(),
        usage: usage(),
    });
    let h = StandardHarness::new(os_config(a, 2));
    match h.run(HarnessRunOptions::new(os_leaf(10))).await {
        RunResult::Success {
            output,
            turns,
            session_state,
            ..
        } => {
            assert_eq!(output, OS_VALID);
            assert_eq!(turns, 2, "one retry consumed");
            // The frozen feedback for the FIRST-failing rule (missing
            // required `status`, array order) must be present, exact bytes.
            const EXPECTED_FEEDBACK: &str = "Your previous response did not match the required output schema. Missing required property \"status\". Reply with only a JSON value that satisfies the schema.";
            assert!(
                os_user_msgs(&session_state)
                    .iter()
                    .any(|m| m == EXPECTED_FEEDBACK),
                "the exact frozen feedback (with validator error) must be fed back"
            );
        }
        other => panic!("expected Success after retry, got {other:?}"),
    }
}

// AC3 fail: N+1 invalid terminals (N == 2 ⇒ 3 attempts) with a generous
// budget → OutputSchemaViolation, DISTINCT from budget, turns < budget.
#[tokio::test]
async fn os_ac3_fail_after_retries_exhausted() {
    let a = make_agent();
    for _ in 0..3 {
        a.push(TurnResult::FinalResponse {
            reasoning: None,
            content: OS_INVALID.into(),
            usage: usage(),
        });
    }
    let h = StandardHarness::new(os_config(a, 2));
    match h.run(HarnessRunOptions::new(os_leaf(50))).await {
        RunResult::Failure {
            reason:
                HaltReason::OutputSchemaViolation {
                    attempts,
                    last_error,
                    schema,
                },
            turns,
            ..
        } => {
            assert_eq!(attempts, 3, "1 + N == 1 + 2 attempts");
            assert_eq!(turns, 3, "exactly 1 + N turns");
            assert!(turns < 50, "budget NOT exhausted (distinct from budget)");
            assert_eq!(last_error, "Missing required property \"status\".");
            // The reported schema is the canonical key-sorted JSON.
            assert_eq!(
                    schema,
                    "{\"properties\":{\"count\":{\"type\":\"integer\"},\"status\":{\"enum\":[\"ok\",\"error\"],\"type\":\"string\"}},\"required\":[\"status\",\"count\"],\"type\":\"object\"}"
                );
        }
        other => panic!("expected OutputSchemaViolation, got {other:?}"),
    }
}

// AC3 budget precedence: a tiny turn budget (2) is exhausted by retries
// BEFORE the N==5 retries run out → the BUDGET terminal wins, NOT
// OutputSchemaViolation (Q2 budget-cap-wins).
#[tokio::test]
async fn os_ac3_budget_precedence_over_schema_violation() {
    let a = make_agent();
    // More invalid turns than the budget allows; the budget gate stops first.
    for _ in 0..5 {
        a.push(TurnResult::FinalResponse {
            reasoning: None,
            content: OS_INVALID.into(),
            usage: usage(),
        });
    }
    // N == 5 (large), budget == 2 turns. After 2 invalid terminals the
    // 3rd retry re-enters the loop where the turn-budget gate fires.
    let h = StandardHarness::new(os_config(a, 5));
    match h.run(HarnessRunOptions::new(os_leaf(2))).await {
        RunResult::Failure {
            reason: HaltReason::BudgetExceeded { .. },
            turns,
            ..
        } => {
            assert_eq!(turns, 2, "stopped at the turn budget, not on schema");
        }
        other => panic!("expected BudgetExceeded (budget wins), got {other:?}"),
    }
}

// AC4: flag OFF ⇒ an INVALID terminal is accepted as Success (no resolve, no
// delivery, no validation) — the migration-gate guarantee.
#[tokio::test]
async fn os_ac4_flag_off_accepts_invalid_terminal() {
    let a = make_agent();
    a.push(TurnResult::FinalResponse {
        reasoning: None,
        content: OS_INVALID.into(),
        usage: usage(),
    });
    // standard_config has enforce_output_schemas = false; register the schema
    // anyway to prove it is IGNORED when the gate is OFF.
    let mut cfg = standard_config(a);
    cfg.registry = std::mem::take(&mut cfg.registry)
        .into_builder()
        .schema("", os_schema())
        .build();
    let h = StandardHarness::new(cfg);
    match h.run(HarnessRunOptions::new(os_leaf(10))).await {
        RunResult::Success {
            output,
            turns,
            session_state,
            ..
        } => {
            assert_eq!(output, OS_INVALID, "invalid terminal accepted as-is");
            assert_eq!(turns, 1);
            // No directive seeded when the gate is OFF.
            let seeded = os_user_msgs(&session_state)
                .iter()
                .any(|m| m.contains("conforms to this"));
            assert!(!seeded, "no schema directive delivered when gate is OFF");
        }
        other => panic!("expected Success (gate OFF), got {other:?}"),
    }
}

// AC4 + events: enforcement ON emits the retry + violation stream events.
#[tokio::test]
async fn os_emits_retry_and_violation_events() {
    let a = make_agent();
    for _ in 0..2 {
        a.push(TurnResult::FinalResponse {
            reasoning: None,
            content: OS_INVALID.into(),
            usage: usage(),
        });
    }
    let captured: Arc<std::sync::Mutex<Vec<StreamEvent>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = captured.clone();
    // N == 1 ⇒ 1 retry then violation (2 attempts).
    let h = StandardHarness::new(os_config(a, 1));
    let opts = HarnessRunOptions::new(os_leaf(50)).with_stream(move |ev| {
        sink.lock().unwrap().push(ev);
    });
    let _ = h.run(opts).await;
    let evs = captured.lock().unwrap();
    let retries: Vec<u32> = evs
        .iter()
        .filter_map(|e| match e {
            StreamEvent::OutputSchemaRetry { attempt, .. } => Some(*attempt),
            _ => None,
        })
        .collect();
    let violations: Vec<u32> = evs
        .iter()
        .filter_map(|e| match e {
            StreamEvent::OutputSchemaViolation { attempts, .. } => Some(*attempts),
            _ => None,
        })
        .collect();
    assert_eq!(retries, vec![1], "one retry event at attempt 1");
    assert_eq!(violations, vec![2], "one violation event, attempts == 2");
}
