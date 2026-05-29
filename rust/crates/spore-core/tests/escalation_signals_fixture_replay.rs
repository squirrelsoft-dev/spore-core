//! Serde round-trip fixture replay for the Tool Escalation Protocol (issue #80).
//!
//! Loads `fixtures/harness/escalation_signals.json` — the shared,
//! cross-language wire-format fixture — and asserts that every case
//! deserializes into the Rust type and re-serializes to a STRUCTURALLY
//! IDENTICAL JSON value. This locks the byte-identical wire guarantee across
//! the four language implementations:
//!   - `tool_output_cases`: each is a `ToolOutput::Escalate` wrapping one of
//!     the four `HarnessSignal` variants (EnterPlanMode, ExitPlanMode,
//!     SwitchMode, Abort).
//!   - `run_result_cases`: `RunResult::Escalate` carrying a full PausedState,
//!     for at least `ExitPlanMode` and `Abort`.
//!
//! Never edit the fixture to make a failing implementation pass — fix the
//! implementation (see `fixtures/README.md`).

use serde_json::Value;
use spore_core::{RunResult, ToolOutput};

fn fixture() -> Value {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest
        .join("../../..")
        .join("fixtures/harness/escalation_signals.json");
    let raw =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).expect("fixture is valid JSON")
}

#[test]
fn tool_output_escalate_cases_round_trip() {
    let fixture = fixture();
    let cases = fixture["tool_output_cases"]
        .as_array()
        .expect("tool_output_cases array");
    assert_eq!(cases.len(), 4, "all four HarnessSignal variants present");

    // Each variant tag must appear exactly once.
    let mut seen = std::collections::BTreeSet::new();
    for case in cases {
        assert_eq!(case["kind"], "escalate", "outer ToolOutput tag");
        seen.insert(case["signal"]["kind"].as_str().unwrap().to_string());

        let typed: ToolOutput =
            serde_json::from_value(case.clone()).expect("deserialize ToolOutput::Escalate");
        let reser = serde_json::to_value(&typed).expect("re-serialize");
        assert_eq!(&reser, case, "structural round-trip identity for {case}");
    }
    let expected: std::collections::BTreeSet<String> =
        ["enter_plan_mode", "exit_plan_mode", "switch_mode", "abort"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    assert_eq!(seen, expected, "every HarnessSignal variant covered");
}

#[test]
fn run_result_escalate_cases_round_trip() {
    let fixture = fixture();
    let cases = fixture["run_result_cases"]
        .as_array()
        .expect("run_result_cases array");
    assert!(
        cases.len() >= 2,
        "at least ExitPlanMode and Abort wrapped in RunResult::Escalate"
    );

    let mut signal_kinds = std::collections::BTreeSet::new();
    for case in cases {
        assert_eq!(case["kind"], "escalate", "outer RunResult tag");
        // Escalation-derived state carries no human request.
        assert!(
            case["state"]["human_request"].is_null(),
            "escalation PausedState.human_request must be null"
        );
        signal_kinds.insert(case["signal"]["kind"].as_str().unwrap().to_string());

        let typed: RunResult =
            serde_json::from_value(case.clone()).expect("deserialize RunResult::Escalate");
        let reser = serde_json::to_value(&typed).expect("re-serialize");
        assert_eq!(&reser, case, "structural round-trip identity for {case}");

        // The five fields are all present and typed.
        match typed {
            RunResult::Escalate {
                state,
                turns,
                usage,
                ..
            } => {
                assert!(state.human_request.is_none());
                assert_eq!(turns, state.budget_used.turns);
                assert_eq!(usage.input_tokens, state.budget_used.input_tokens);
            }
            other => panic!("expected RunResult::Escalate, got {other:?}"),
        }
    }
    assert!(signal_kinds.contains("exit_plan_mode"));
    assert!(signal_kinds.contains("abort"));
}
