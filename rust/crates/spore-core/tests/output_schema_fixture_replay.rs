//! Loop-replay integration tests for output-schema delivery + enforcement
//! (issue #139).
//!
//! Three recorded traces under `fixtures/model_responses/harness/`:
//!   - `output_schema_accept.jsonl` — turn-1 terminal already satisfies the
//!     schema ⇒ `RunResult::Success` in ONE turn.
//!   - `output_schema_retry.jsonl` — turn-1 terminal is INVALID, the harness
//!     feeds the FROZEN feedback message back, turn-2 terminal is valid ⇒
//!     `Success` in TWO turns. The fixture's SECOND request carries the exact
//!     frozen feedback message text (hash-load-bearing).
//!   - `output_schema_fail.jsonl` — every terminal is invalid; after
//!     `output_schema_max_retries == 2` extra turns (3 attempts) WITH budget
//!     remaining ⇒ `RunResult::Failure { OutputSchemaViolation }`.
//!
//! Replay is POSITIONAL (no `request_hash` in these fixtures): the responses
//! drive the flow in order; the request bodies document the conversation the
//! harness actually builds (the retry/fail requests embed the frozen feedback
//! so the cross-language ports can byte-compare it). The four languages must
//! produce the same outcome + turn count — never edit a fixture to make a
//! failing implementation pass (see `fixtures/README.md`).

use std::sync::Arc;

use spore_core::harness::testing::{
    AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager, ScriptedToolRegistry,
};
use spore_core::{
    Agent, AgentId, AgentRef, BudgetExhaustedBehavior, BudgetPolicy, HaltReason, Harness,
    HarnessConfig, HarnessRunOptions, LoopStrategy, ModelAgent, ProviderInfo, ReactConfig,
    ReplayModelInterface, RunResult, SchemaRef, SessionId, StandardHarness, Task, ToolsetRef,
};

fn fixture_path(name: &str) -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("../../..")
        .join("fixtures/model_responses/harness")
        .join(name)
}

fn output_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "required": ["status", "count"],
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "count": {"type": "integer"}
        }
    })
}

/// Build a `StandardHarness` with output-schema enforcement ON, the
/// `output_schema` registered under the default empty `SchemaRef`, and the named
/// fixture wired as the replay model.
fn harness_for(fixture: &str, max_retries: u32) -> StandardHarness {
    let jsonl = std::fs::read_to_string(fixture_path(fixture)).expect("fixture readable");
    let replay = Arc::new(
        ReplayModelInterface::from_jsonl(
            &jsonl,
            ProviderInfo {
                name: "ollama".into(),
                model_id: "fixture".into(),
                context_window: 200_000,
            },
        )
        .expect("fixture parses"),
    );
    let agent: Arc<ModelAgent<ReplayModelInterface>> =
        Arc::new(ModelAgent::new(AgentId::new("fixture-agent"), replay));

    let tool_registry = Arc::new(ScriptedToolRegistry::new());

    let config = HarnessConfig {
        tool_registry: tool_registry.clone(),
        sandbox: Arc::new(AllowAllSandbox),
        sandbox_violation_policy: spore_core::harness::SandboxViolationPolicy::default(),
        context_manager: Arc::new(NoopContextManager),
        termination_policy: Arc::new(AlwaysContinuePolicy),
        toolset_catalogues: Default::default(),
        middleware: None,
        observability: None,
        compaction_verifier: Arc::new(spore_core::KeyTermVerifier),
        max_compaction_attempts: 2,
        pricing: spore_core::PricingTable::DEFAULT,
        content_capture: spore_core::ContentCaptureConfig::default(),
        tool_call_repair: None,
        max_repair_attempts: 1,
        max_stop_blocks: 8,
        error_loop_threshold: 3,
        // #139: enforcement ON; N == `max_retries`.
        enforce_output_schemas: true,
        output_schema_max_retries: max_retries,
        hooks: None,
        storage: Arc::new(spore_core::StorageProvider::no_op()),
        project_id: spore_core::ProjectId::from_canonical_path("/test-workspace"),
        chunk_provider: Arc::new(spore_core::prompt_assembly::InMemoryChunkProvider::empty()),
        max_resets: 3,
        vcs_provider: None,
        catalogue_registry: None,
        system_prompt: None,
        guides: Vec::new(),
        skills: None,
        model_params: spore_core::ModelParams::default(),
        auto_persist_sessions: false,
        prompt_tool_call_flag: None,
        consult_handlers: std::collections::HashMap::new(),
        registry: spore_core::ExecutionRegistry::builder()
            .agent("", agent as Arc<dyn Agent>)
            .toolset("", tool_registry.clone())
            .schema("", output_schema())
            .build(),
        escalation_mode: spore_core::EscalationMode::Autonomous,
    };
    StandardHarness::new(config)
}

/// A ReAct leaf carrying `output = Some(SchemaRef(""))` and a turn budget.
fn os_task(budget: u32) -> Task {
    Task::new(
        "produce a status report",
        SessionId::new("output-schema-session"),
        LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: budget },
            behavior: BudgetExhaustedBehavior::Fail,
            agent: AgentRef(String::new()),
            toolset: ToolsetRef(String::new()),
            output: Some(SchemaRef(String::new())),
        }),
    )
}

const FEEDBACK: &str = "Your previous response did not match the required output schema. \
Missing required property \"status\". Reply with only a JSON value that satisfies the schema.";

#[tokio::test]
async fn output_schema_accept_succeeds_in_one_turn() {
    let h = harness_for("output_schema_accept.jsonl", 2);
    match h.run(HarnessRunOptions::new(os_task(10))).await {
        RunResult::Success { output, turns, .. } => {
            assert_eq!(output, "{\"status\":\"ok\",\"count\":3}");
            assert_eq!(turns, 1, "valid terminal accepted on turn 1");
        }
        other => panic!("expected Success, got {other:?}"),
    }
}

#[tokio::test]
async fn output_schema_retry_feeds_frozen_message_then_succeeds() {
    let h = harness_for("output_schema_retry.jsonl", 2);
    match h.run(HarnessRunOptions::new(os_task(10))).await {
        RunResult::Success {
            output,
            turns,
            session_state,
            ..
        } => {
            assert_eq!(output, "{\"status\":\"ok\",\"count\":3}");
            assert_eq!(turns, 2, "one retry consumed");
            // The harness fed the EXACT frozen feedback (with the validator
            // error) back as a user message — the same bytes the fixture's
            // second request embeds.
            let fed = session_state.messages.iter().any(|m| {
                matches!(
                    (&m.role, &m.content),
                    (spore_core::Role::User, spore_core::Content::Text { text }) if text == FEEDBACK
                )
            });
            assert!(fed, "the exact frozen feedback message must be fed back");
        }
        other => panic!("expected Success after retry, got {other:?}"),
    }
}

#[tokio::test]
async fn output_schema_fail_terminates_with_violation() {
    let h = harness_for("output_schema_fail.jsonl", 2);
    match h.run(HarnessRunOptions::new(os_task(50))).await {
        RunResult::Failure {
            reason:
                HaltReason::OutputSchemaViolation {
                    attempts,
                    last_error,
                    ..
                },
            turns,
            session_id,
            ..
        } => {
            assert_eq!(attempts, 3, "1 + N == 1 + 2");
            assert_eq!(turns, 3, "exactly 1 + N turns; budget not exhausted");
            assert!(turns < 50, "distinct from budget exhaustion");
            assert_eq!(last_error, "Missing required property \"status\".");
            assert_eq!(session_id, SessionId::new("output-schema-session"));
        }
        other => panic!("expected OutputSchemaViolation, got {other:?}"),
    }
}
