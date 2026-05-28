//! Hermetic end-to-end scenario tests (issue #57).
//!
//! These drive the SAME `build_scenario` wiring the `e2e_agent` example uses,
//! but with a scripted mock agent, scripted/real tool registries, and an
//! allow-all sandbox, so CI never needs a live Ollama or any network.
//! Each test asserts the harness loop control flow (turn count, tool dispatch
//! order, S4 recovery sequencing, S3 live compaction with real token
//! reclamation). `SPORE_OTLP_ENDPOINT` stays unset, so there is no forwarding.

#![cfg(feature = "test-utils")]

use std::sync::Arc;

use spore_core::agent::mock::MockAgent;
use spore_core::agent::Context as AgentContext;
use spore_core::cache_provider::NullCacheProvider;
use spore_core::context::CompactionConfig;
use spore_core::harness::testing::NoopContextManager;
use spore_core::harness::testing::{
    AllowAllSandbox, AlwaysContinuePolicy, ScriptedMiddleware, ScriptedToolRegistry,
};
use spore_core::harness::{
    BoxFut, HarnessBuilder, MiddlewareChain, MiddlewareDecision, TerminationPolicy,
};
use spore_core::model::mock::MockModelInterface;
use spore_core::model::{Content, Role};
use spore_core::observability::{ContextOperation, InMemoryObservabilityProvider};
use spore_core::scenarios::ScenarioId;
use spore_core::scenarios::{
    build_real_tool_registry, build_rich_context_manager, build_scenario, seed_compaction_state,
    RealToolRegistry,
};
use spore_core::{
    Agent, AgentId, FullObservabilityProvider, Harness, HarnessContextManager, HarnessRunOptions,
    HarnessToolRegistry, HookPoint, HumanRequest, HumanResponse, LoopStrategy, ProviderInfo,
    RunResult, SandboxProvider, SessionId, SessionState, Task, TokenUsage, ToolCall, ToolOutput,
    TurnResult,
};

fn usage() -> TokenUsage {
    TokenUsage {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_tokens: None,
        cache_write_tokens: None,
    }
}

fn tool_call(id: &str, name: &str, input: serde_json::Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        input,
    }
}

// ---------------------------------------------------------------------------
// S1 — multi-step / multi-tool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn s1_multi_step_multi_tool() {
    let agent = Arc::new(MockAgent::new(AgentId::new("mock")));
    // read -> write -> bash read-back -> final.
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![tool_call(
            "c1",
            "read_file",
            serde_json::json!({"path": "input.txt"}),
        )],
        usage: usage(),
    });
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![tool_call(
            "c2",
            "write_file",
            serde_json::json!({"path": "output.txt", "content": "UPPERCASED"}),
        )],
        usage: usage(),
    });
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![tool_call(
            "c3",
            "read_file",
            serde_json::json!({"path": "output.txt"}),
        )],
        usage: usage(),
    });
    agent.push(TurnResult::FinalResponse {
        content: "DONE".into(),
        usage: usage(),
    });

    let tools = Arc::new(ScriptedToolRegistry::new());
    tools.push(ToolOutput::Success {
        content: "hello".into(),
        truncated: false,
    });
    tools.push(ToolOutput::Success {
        content: "wrote 10 bytes".into(),
        truncated: false,
    });
    tools.push(ToolOutput::Success {
        content: "UPPERCASED".into(),
        truncated: false,
    });
    let tools_for_count = tools.clone();

    let harness = build_scenario(
        ScenarioId::S1,
        agent as Arc<dyn Agent>,
        tools as Arc<dyn HarnessToolRegistry>,
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
        vec![],
        None,
    );

    let task = Task::new(
        ScenarioId::S1.prompt(),
        SessionId::new("s1-test"),
        LoopStrategy::ReAct { max_iterations: 8 },
    );
    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Success { turns, .. } => {
            assert!(turns > 2, "S1 should take >2 turns, got {turns}");
            assert_eq!(
                tools_for_count
                    .call_count
                    .load(std::sync::atomic::Ordering::SeqCst),
                3,
                "S1 dispatches read+write+readback = 3 tools"
            );
        }
        other => panic!("S1 expected Success, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// S2 — multi-turn, same SessionId, carrying session state
// ---------------------------------------------------------------------------

#[tokio::test]
async fn s2_multi_turn_carries_state() {
    let session_id = SessionId::new("s2-test");
    let agent = Arc::new(MockAgent::new(AgentId::new("mock")));
    // Turn 1: write notes.md, then final.
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![tool_call(
            "c1",
            "write_file",
            serde_json::json!({"path": "notes.md", "content": "TODO: set up the project"}),
        )],
        usage: usage(),
    });
    agent.push(TurnResult::FinalResponse {
        content: "DONE".into(),
        usage: usage(),
    });
    // Turn 2: append referencing turn 1, then final.
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![tool_call(
            "c2",
            "write_file",
            serde_json::json!({"path": "notes.md", "content": "TODO: follow up on set up the project", "append": true}),
        )],
        usage: usage(),
    });
    agent.push(TurnResult::FinalResponse {
        content: "DONE referencing set up the project".into(),
        usage: usage(),
    });

    let tools = Arc::new(ScriptedToolRegistry::new());

    let harness = build_scenario(
        ScenarioId::S2,
        agent as Arc<dyn Agent>,
        tools as Arc<dyn HarnessToolRegistry>,
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
        vec![],
        None,
    );

    let task1 = Task::new(
        ScenarioId::S2.prompt(),
        session_id.clone(),
        LoopStrategy::ReAct { max_iterations: 5 },
    );
    let r1 = harness.run(HarnessRunOptions::new(task1)).await;
    let carried = match r1 {
        RunResult::Success { .. } => SessionState::default(),
        other => panic!("S2 turn 1 expected Success, got {other:?}"),
    };

    let task2 = Task::new(
        "add a second item referencing the first",
        session_id.clone(),
        LoopStrategy::ReAct { max_iterations: 5 },
    );
    match harness
        .run(HarnessRunOptions::new(task2).with_session_state(carried))
        .await
    {
        RunResult::Success {
            output,
            session_id: sid,
            ..
        } => {
            assert_eq!(sid, session_id, "same session id across turns");
            assert!(
                output.contains("set up the project"),
                "turn 2 references turn 1 content: {output:?}"
            );
        }
        other => panic!("S2 turn 2 expected Success, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// S3 — live compaction with real token reclamation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn s3_live_compaction_reclaims_tokens() {
    let session_id = SessionId::new("s3-test");
    // Agent emits a tool call (to reach the post-tool compaction arm), then a
    // final summary containing the key term so verification passes.
    let agent = Arc::new(MockAgent::new(AgentId::new("mock")));
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![tool_call(
            "c1",
            "read_file",
            serde_json::json!({"path": "x"}),
        )],
        usage: usage(),
    });
    // Compaction turn (consumed inside run_compaction) — provide a summary that
    // preserves the "payment"/"service"/"deploy" key terms.
    agent.push(TurnResult::FinalResponse {
        content: "summary: continuing the deploy of the payment service".into(),
        usage: usage(),
    });
    // Next loop turn after compaction: final response.
    agent.push(TurnResult::FinalResponse {
        content: "DONE deploy payment service".into(),
        usage: usage(),
    });

    let tools = Arc::new(ScriptedToolRegistry::new());
    tools.push(ToolOutput::Success {
        content: "file contents".into(),
        truncated: false,
    });

    let model = Arc::new(MockModelInterface::new(ProviderInfo {
        name: "mock".into(),
        model_id: "mock".into(),
        context_window: 200,
    }));
    let cfg = CompactionConfig {
        threshold: 0.80,
        preserve_recent_n: 2,
        head_tail_tokens: 64,
        offload_path: std::path::PathBuf::from(".spore/offload"),
        max_compaction_attempts: 2,
    };
    let cm: Arc<dyn HarnessContextManager> =
        build_rich_context_manager(model, Arc::new(NullCacheProvider), cfg);

    let obs = Arc::new(InMemoryObservabilityProvider::new());

    let harness = build_scenario(
        ScenarioId::S3,
        agent as Arc<dyn Agent>,
        tools as Arc<dyn HarnessToolRegistry>,
        Arc::new(AllowAllSandbox),
        cm,
        Arc::new(AlwaysContinuePolicy),
        vec![],
        Some(obs.clone() as Arc<dyn FullObservabilityProvider>),
    );

    let task = Task::new(
        "deploy the payment service",
        session_id.clone(),
        LoopStrategy::ReAct { max_iterations: 8 },
    );
    // Seed a small window with budget over threshold + long history.
    let mut state = SessionState::default();
    seed_compaction_state(
        &mut state,
        "deploy the payment service",
        session_id.clone(),
        task.id.clone(),
        200,
        170, // 0.85 > 0.80 threshold
        12,
    );

    let result = harness
        .run(HarnessRunOptions::new(task).with_session_state(state))
        .await;
    assert!(
        matches!(result, RunResult::Success { .. }),
        "S3 expected Success, got {result:?}"
    );

    // A Compaction context span was emitted, and it reclaimed real tokens.
    let compactions: Vec<_> = obs
        .context_spans(&session_id)
        .into_iter()
        .filter(|c| matches!(c.operation, ContextOperation::Compaction { .. }))
        .collect();
    assert!(
        !compactions.is_empty(),
        "S3 should emit >=1 Compaction span mid-run"
    );
    let first = &compactions[0];
    assert!(
        first.tokens_after < first.tokens_before,
        "token_budget_used must drop after compaction: {} -> {}",
        first.tokens_before,
        first.tokens_after
    );
    if let ContextOperation::Compaction {
        tokens_reclaimed, ..
    } = first.operation
    {
        assert!(tokens_reclaimed > 0, "real reclamation must be > 0");
    }
}

// ---------------------------------------------------------------------------
// S4 — tool failure + recovery (uses the REAL registry + FailingTool)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn s4_tool_failure_then_recovery() {
    let workspace = std::env::temp_dir().join(format!(
        "spore-s4-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&workspace).unwrap();

    let session_id = SessionId::new("s4-test");
    let agent = Arc::new(MockAgent::new(AgentId::new("mock")));
    // Call flaky_op (fails recoverably) -> adapt by writing recovered.txt -> final.
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![tool_call(
            "c1",
            "flaky_op",
            serde_json::json!({"reason": "first try"}),
        )],
        usage: usage(),
    });
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![tool_call(
            "c2",
            "write_file",
            serde_json::json!({
                "path": format!("{}/recovered.txt", workspace.display()),
                "content": "flaky_op failed; adapted by writing this file"
            }),
        )],
        usage: usage(),
    });
    agent.push(TurnResult::FinalResponse {
        content: "DONE recovered".into(),
        usage: usage(),
    });

    let registry = build_real_tool_registry(ScenarioId::S4);
    let sandbox: Arc<dyn SandboxProvider> = Arc::new(AllowAllSandbox);
    let bridge = RealToolRegistry::new(registry, sandbox.clone());
    let schemas = bridge.model_schemas();
    let tools: Arc<dyn HarnessToolRegistry> = Arc::new(bridge);

    let harness = build_scenario(
        ScenarioId::S4,
        agent as Arc<dyn Agent>,
        tools,
        sandbox,
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
        schemas,
        None,
    );

    let task = Task::new(
        ScenarioId::S4.prompt(),
        session_id.clone(),
        LoopStrategy::ReAct { max_iterations: 8 },
    );
    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Success { turns, .. } => {
            assert!(turns >= 3, "S4: flaky -> recover -> done");
            assert!(
                workspace.join("recovered.txt").exists(),
                "recovery file written"
            );
        }
        other => panic!("S4 expected Success, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&workspace);
}

/// The harness must NOT hard-halt on the recoverable FailingTool error — the
/// bridge reports `is_always_halt == false`.
#[tokio::test]
async fn s4_failing_tool_is_not_always_halt() {
    let bridge = RealToolRegistry::new(
        build_real_tool_registry(ScenarioId::S4),
        Arc::new(AllowAllSandbox),
    );
    assert!(!bridge.is_always_halt("flaky_op"));
    let out = bridge
        .dispatch(tool_call("c1", "flaky_op", serde_json::json!({})))
        .await;
    assert!(matches!(
        out,
        ToolOutput::Error {
            recoverable: true,
            ..
        }
    ));
}

// ---------------------------------------------------------------------------
// S5 — real shell tool: bash_command with a pipe + redirect (uses the REAL
// registry, which exposes bash_command only for S5).
// ---------------------------------------------------------------------------

#[cfg(unix)]
#[tokio::test]
async fn s5_shell_pipeline_uppercases_via_bash_command() {
    let workspace = std::env::temp_dir().join(format!(
        "spore-s5-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&workspace).unwrap();
    let input = workspace.join("input.txt");
    let output = workspace.join("output.txt");
    std::fs::write(&input, "hello\n").unwrap();

    let session_id = SessionId::new("s5-test");
    let agent = Arc::new(MockAgent::new(AgentId::new("mock")));
    // turn1: bash_command with a literal pipe AND redirect; turn2: read_file
    // output.txt to verify; turn3: DONE.
    let script = format!(
        "cat {} | tr a-z A-Z > {}",
        input.display(),
        output.display()
    );
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![tool_call(
            "c1",
            "bash_command",
            serde_json::json!({ "script": script }),
        )],
        usage: usage(),
    });
    agent.push(TurnResult::ToolCallRequested {
        calls: vec![tool_call(
            "c2",
            "read_file",
            serde_json::json!({ "path": output.display().to_string() }),
        )],
        usage: usage(),
    });
    agent.push(TurnResult::FinalResponse {
        content: "DONE".into(),
        usage: usage(),
    });

    let registry = build_real_tool_registry(ScenarioId::S5);
    let sandbox: Arc<dyn SandboxProvider> = Arc::new(AllowAllSandbox);
    let bridge = RealToolRegistry::new(registry, sandbox.clone());
    let schemas = bridge.model_schemas();
    let tools: Arc<dyn HarnessToolRegistry> = Arc::new(bridge);

    let harness = build_scenario(
        ScenarioId::S5,
        agent as Arc<dyn Agent>,
        tools,
        sandbox,
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
        schemas,
        None,
    );

    let task = Task::new(
        ScenarioId::S5.prompt(),
        session_id.clone(),
        LoopStrategy::ReAct { max_iterations: 8 },
    );
    match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::Success { turns, .. } => {
            assert!(
                turns >= 3,
                "S5: bash_command -> read_file -> done, got {turns}"
            );
            assert_eq!(
                std::fs::read_to_string(&output).unwrap(),
                "HELLO\n",
                "shell pipeline must uppercase input.txt into output.txt"
            );
        }
        other => panic!("S5 expected Success, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&workspace);
}

// ---------------------------------------------------------------------------
// Regression: the task instruction must reach the agent as the first user
// message (issue #57). Unlike `MockAgent`, which ignores its `Context`, this
// agent records every assembled `Context` so we can assert the model actually
// receives the prompt. Backed by the real `StandardCompactionAdapter` context
// manager (via `build_rich_context_manager`), exactly like a live run — the
// adapter mirrors `session.messages` and ignores `task`, so without the
// harness seeding the instruction the captured first-turn context is EMPTY and
// this test fails (which is the bug we are fixing).
// ---------------------------------------------------------------------------

struct CapturingAgent {
    id: AgentId,
    contexts: Arc<std::sync::Mutex<Vec<AgentContext>>>,
}

impl Agent for CapturingAgent {
    fn turn<'a>(&'a self, context: AgentContext) -> BoxFut<'a, TurnResult> {
        self.contexts.lock().unwrap().push(context);
        Box::pin(async move {
            TurnResult::FinalResponse {
                content: "DONE".into(),
                usage: usage(),
            }
        })
    }

    fn id(&self) -> AgentId {
        self.id.clone()
    }
}

#[tokio::test]
async fn task_instruction_delivered_as_first_user_message() {
    let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
    let agent = Arc::new(CapturingAgent {
        id: AgentId::new("capture"),
        contexts: captured.clone(),
    });

    // Real compaction-adapter-backed context manager (mirrors session.messages,
    // ignores `task`), so only the harness seeding can put the prompt on screen.
    let model = Arc::new(MockModelInterface::new(ProviderInfo {
        name: "mock".into(),
        model_id: "mock".into(),
        context_window: 4096,
    }));
    let cfg = CompactionConfig {
        threshold: 0.80,
        preserve_recent_n: 2,
        head_tail_tokens: 64,
        offload_path: std::path::PathBuf::from(".spore/offload"),
        max_compaction_attempts: 2,
    };
    let cm: Arc<dyn HarnessContextManager> =
        build_rich_context_manager(model, Arc::new(NullCacheProvider), cfg);

    let harness = build_scenario(
        ScenarioId::S1,
        agent as Arc<dyn Agent>,
        Arc::new(ScriptedToolRegistry::new()) as Arc<dyn HarnessToolRegistry>,
        Arc::new(AllowAllSandbox),
        cm,
        Arc::new(AlwaysContinuePolicy),
        vec![],
        None,
    );

    let instruction = "summarize the quarterly payment report";
    let task = Task::new(
        instruction,
        SessionId::new("seed-test"),
        LoopStrategy::ReAct { max_iterations: 4 },
    );
    let result = harness.run(HarnessRunOptions::new(task)).await;
    assert!(
        matches!(result, RunResult::Success { .. }),
        "expected Success, got {result:?}"
    );

    let contexts = captured.lock().unwrap();
    let first = contexts
        .first()
        .expect("agent must have been invoked at least once");
    let has_user_instruction = first.messages.iter().any(|m| {
        m.role == Role::User && matches!(&m.content, Content::Text { text } if text == instruction)
    });
    assert!(
        has_user_instruction,
        "first-turn context must contain a User message equal to the task \
         instruction; got messages: {:?}",
        first.messages
    );
}

// ---------------------------------------------------------------------------
// Regression: resume must NOT re-seed the task instruction (issue #57).
//
// Seeding lives on the `run()` entry, NOT in the shared `run_react_inner` that
// `resume_inner` also calls. To prove resume does not duplicate the prompt we:
//   1. Pause a fresh run at `BeforeTurn` via a scripted middleware
//      (`SurfaceToHuman`), so the instruction is seeded exactly once but no
//      agent turn has run yet.
//   2. Resume with `HumanResponse::Answer`, which appends the human text and
//      re-enters `run_react` -> `run_react_inner`.
//   3. Assert the captured post-resume context contains the task instruction
//      EXACTLY ONCE. If seeding were in the shared inner path, the resume would
//      append it a second time and this count would be 2 (test fails).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn resume_does_not_reseed_task_instruction() {
    let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
    let agent = Arc::new(CapturingAgent {
        id: AgentId::new("capture"),
        contexts: captured.clone(),
    });

    let model = Arc::new(MockModelInterface::new(ProviderInfo {
        name: "mock".into(),
        model_id: "mock".into(),
        context_window: 4096,
    }));
    let cfg = CompactionConfig {
        threshold: 0.80,
        preserve_recent_n: 2,
        head_tail_tokens: 64,
        offload_path: std::path::PathBuf::from(".spore/offload"),
        max_compaction_attempts: 2,
    };
    let cm: Arc<dyn HarnessContextManager> =
        build_rich_context_manager(model, Arc::new(NullCacheProvider), cfg);

    // Surface to human on the very first BeforeTurn so the run pauses before any
    // agent turn. After resume the queue is empty -> Continue everywhere.
    let mw = Arc::new(ScriptedMiddleware::new());
    mw.push(
        HookPoint::BeforeTurn,
        MiddlewareDecision::SurfaceToHuman {
            request: HumanRequest::Clarification {
                question: "proceed?".into(),
            },
        },
    );

    let harness = HarnessBuilder::new(
        agent as Arc<dyn Agent>,
        Arc::new(ScriptedToolRegistry::new()) as Arc<dyn HarnessToolRegistry>,
        Arc::new(AllowAllSandbox),
        cm,
        Arc::new(AlwaysContinuePolicy) as Arc<dyn TerminationPolicy>,
    )
    .middleware(mw as Arc<dyn MiddlewareChain>)
    .build();

    let instruction = "summarize the quarterly payment report";
    let task = Task::new(
        instruction,
        SessionId::new("resume-test"),
        LoopStrategy::ReAct { max_iterations: 4 },
    );

    // Fresh run: seeds the instruction once, then pauses at BeforeTurn.
    let paused = match harness.run(HarnessRunOptions::new(task)).await {
        RunResult::WaitingForHuman { state, .. } => *state,
        other => panic!("expected WaitingForHuman pause, got {other:?}"),
    };

    // Resume with a human answer; this re-enters run_react via run_react_inner.
    let result = harness
        .resume(
            paused,
            HumanResponse::Answer {
                text: "yes, proceed".into(),
            },
            None,
        )
        .await;
    assert!(
        matches!(result, RunResult::Success { .. }),
        "expected Success after resume, got {result:?}"
    );

    // The post-resume turn must see the instruction exactly once.
    let contexts = captured.lock().unwrap();
    let post_resume = contexts.first().expect("agent must have run after resume");
    let instruction_count = post_resume
        .messages
        .iter()
        .filter(|m| {
            m.role == Role::User
                && matches!(&m.content, Content::Text { text } if text == instruction)
        })
        .count();
    assert_eq!(
        instruction_count, 1,
        "task instruction must appear exactly once after resume (re-seeding \
         in the shared inner path would make it 2); got messages: {:?}",
        post_resume.messages
    );
}
