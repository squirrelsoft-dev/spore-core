//! Fixture-replay integration test for the lifecycle hook system (issue #69).
//!
//! Loads the JSON fixtures under `fixtures/hooks/` and asserts the reference
//! Rust implementation reaches the expected outcome for each case. The same
//! fixtures are replayed by the TypeScript, Python, and Go suites; a divergence
//! means the implementation is wrong, not the fixture.
//!
//! Covered fixtures:
//!   - `hook_decision_wire.json`   — HookDecision serde round-trips.
//!   - `pre_tool_use_mutation.json`— PreToolUse mutation chain + deny.
//!   - `stop_block_basic.json`     — Stop block/continue + max_stop_blocks cap.
//!   - `command_handler_io.json`   — CommandHook stdin/stdout contract.

use std::sync::Arc;

use serde_json::Value;
use spore_core::{
    CommandHook, FireOutcome, FunctionHook, HookChain, HookContext, HookDecision, HookError,
    HookEvent, SessionId, StandardHookChain,
};

fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("fixtures/hooks")
}

fn load(name: &str) -> Value {
    let path = fixtures_dir().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture {} unreadable: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

// ── hook_decision_wire.json ────────────────────────────────────────────────
#[test]
fn hook_decision_wire_roundtrips() {
    let fixture = load("hook_decision_wire.json");
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let json = &case["json"];
        let decoded: HookDecision = serde_json::from_value(json.clone())
            .unwrap_or_else(|e| panic!("{name}: decode failed: {e}"));
        let reencoded = serde_json::to_value(&decoded).unwrap();
        assert_eq!(&reencoded, json, "{name}: re-encoded wire mismatch");
    }
}

// ── pre_tool_use_mutation.json ──────────────────────────────────────────────
#[tokio::test]
async fn pre_tool_use_mutation_replay() {
    let fixture = load("pre_tool_use_mutation.json");
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap().to_string();
        let chain = StandardHookChain::new();
        for (i, d) in case["hook_decisions"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let decision: HookDecision = serde_json::from_value(d.clone()).unwrap();
            chain
                .register(Arc::new(FunctionHook::new(
                    format!("{name}-{i}"),
                    vec![HookEvent::PreToolUse],
                    move |_ctx| Ok(decision.clone()),
                )))
                .unwrap();
        }
        let sid = SessionId::new("sess-1");
        let mut input = case["tool_input"].clone();
        let tool_name = case["tool_name"].as_str().unwrap().to_string();
        let mut ctx = HookContext::PreToolUse {
            session_id: &sid,
            turn_number: 1,
            tool_name: &tool_name,
            tool_input: &mut input,
        };
        let outcome = chain.fire(&mut ctx).await.unwrap();
        let expected = &case["expected"];
        match expected["outcome"].as_str().unwrap() {
            "continue" => {
                assert_eq!(outcome, FireOutcome::Continue, "{name}: expected continue");
                assert_eq!(input, expected["tool_input"], "{name}: tool_input mismatch");
            }
            "deny" => {
                assert_eq!(
                    outcome,
                    FireOutcome::Deny {
                        reason: expected["reason"].as_str().unwrap().to_string()
                    },
                    "{name}: expected deny"
                );
            }
            other => panic!("{name}: unknown outcome {other}"),
        }
    }
}

// ── stop_block_basic.json ───────────────────────────────────────────────────
//
// Models the per-run Stop-block counter the harness applies (R12/R13/R14): for
// each Stop decision in `hook_decisions`, a `block` while `blocks <
// max_stop_blocks` consumes a block and continues; a `continue` terminates; and
// reaching the cap terminates anyway.
#[tokio::test]
async fn stop_block_basic_replay() {
    let fixture = load("stop_block_basic.json");
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap().to_string();
        let max = case["max_stop_blocks"].as_u64().unwrap() as u32;
        let mut blocks: u32 = 0;
        let mut terminated_by = "continue";

        for d in case["hook_decisions"].as_array().unwrap() {
            // Fire a single Stop hook returning this decision.
            let chain = StandardHookChain::new();
            let decision: HookDecision = serde_json::from_value(d.clone()).unwrap();
            chain
                .register(Arc::new(FunctionHook::new(
                    "stop",
                    vec![HookEvent::Stop],
                    move |_ctx| Ok(decision.clone()),
                )))
                .unwrap();
            let sid = SessionId::new("sess-1");
            let out = spore_core::TurnOutput::default();
            let state = spore_core::ContextSessionState::new(
                sid.clone(),
                spore_core::TaskId::new("t"),
                "x",
            );
            let mut ctx = HookContext::Stop {
                session_id: &sid,
                turn_number: blocks + 1,
                last_output: &out,
                task_instruction: "x",
                session_state: &state,
            };
            match chain.fire(&mut ctx).await.unwrap() {
                FireOutcome::Block { .. } => {
                    if blocks >= max {
                        terminated_by = "cap";
                        break;
                    }
                    blocks += 1;
                }
                _ => {
                    terminated_by = "continue";
                    break;
                }
            }
        }

        let expected = &case["expected"];
        assert_eq!(
            blocks,
            expected["blocks"].as_u64().unwrap() as u32,
            "{name}: block count mismatch"
        );
        assert_eq!(
            terminated_by,
            expected["terminated_by"].as_str().unwrap(),
            "{name}: termination cause mismatch"
        );
    }
}

// ── command_handler_io.json ─────────────────────────────────────────────────
//
// Builds a tiny shell handler per case that echoes the fixture `stdout` (or
// exits with `exit_code`), fires it, and asserts the parsed decision / error.
#[cfg(unix)]
#[tokio::test]
async fn command_handler_io_replay() {
    use std::os::unix::fs::PermissionsExt;

    let fixture = load("command_handler_io.json");
    let dir = std::env::temp_dir().join(format!("hooks-fixture-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    for (idx, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let name = case["name"].as_str().unwrap().to_string();
        let stdout = case["stdout"].as_str().unwrap_or("");
        let exit_code = case["exit_code"].as_i64().unwrap_or(0);

        let script = dir.join(format!("h{idx}.sh"));
        // Read & discard stdin, optionally echo stdout, then exit with the code.
        let body = format!(
            "#!/bin/sh\ncat >/dev/null\nprintf '%s' '{}'\nexit {}\n",
            stdout.replace('\'', "'\\''"),
            exit_code
        );
        std::fs::write(&script, body).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();

        let event = match case["event"].as_str().unwrap() {
            "stop" => HookEvent::Stop,
            "pre_tool_use" => HookEvent::PreToolUse,
            other => panic!("{name}: unsupported event {other}"),
        };
        let chain = StandardHookChain::new();
        chain
            .register(Arc::new(CommandHook::new(
                "cmd",
                vec![event],
                "sh",
                vec![script.to_string_lossy().to_string()],
            )))
            .unwrap();

        // Build a context matching the event; values are not asserted here (the
        // stdin shape is pinned by `expected_stdin` for cross-language parity).
        let sid = SessionId::new("sess-1");
        let result = match event {
            HookEvent::Stop => {
                let out = spore_core::TurnOutput::default();
                let state = spore_core::ContextSessionState::new(
                    sid.clone(),
                    spore_core::TaskId::new("t"),
                    "make the tests pass",
                );
                let mut ctx = HookContext::Stop {
                    session_id: &sid,
                    turn_number: 3,
                    last_output: &out,
                    task_instruction: "make the tests pass",
                    session_state: &state,
                };
                chain.fire(&mut ctx).await
            }
            HookEvent::PreToolUse => {
                let mut input = serde_json::json!({"path": "/etc/passwd"});
                let mut ctx = HookContext::PreToolUse {
                    session_id: &sid,
                    turn_number: 1,
                    tool_name: "read_file",
                    tool_input: &mut input,
                };
                chain.fire(&mut ctx).await
            }
            _ => unreachable!(),
        };

        if let Some(expected_err) = case.get("expected_error").and_then(|v| v.as_str()) {
            let err = result.expect_err(&format!("{name}: expected an error"));
            match expected_err {
                "command_failed" => {
                    assert!(matches!(err, HookError::CommandFailed { .. }), "{name}")
                }
                "command_output_invalid" => {
                    assert!(
                        matches!(err, HookError::CommandOutputInvalid { .. }),
                        "{name}"
                    )
                }
                other => panic!("{name}: unknown expected_error {other}"),
            }
        } else {
            let outcome = result.unwrap_or_else(|e| panic!("{name}: unexpected error {e}"));
            let expected: HookDecision =
                serde_json::from_value(case["expected_decision"].clone()).unwrap();
            // Map the parsed decision to the FireOutcome the chain reports.
            let expected_outcome = match expected {
                HookDecision::Block { reason } => FireOutcome::Block { reason },
                HookDecision::Deny { reason } => FireOutcome::Deny { reason },
                HookDecision::Inject { context } => FireOutcome::Inject { context },
                HookDecision::Continue | HookDecision::Mutate { .. } => FireOutcome::Continue,
            };
            assert_eq!(outcome, expected_outcome, "{name}: decision mismatch");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
