// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// BudgetPolicy + BudgetExhaustedBehavior tests (issue #117)
// ============================================================================

use super::*;

fn nested() -> BudgetExhaustedBehavior {
    // Continue{1, Continue{2, Fail}}
    BudgetExhaustedBehavior::Continue {
        max_continues: 1,
        on_exhausted: Box::new(BudgetExhaustedBehavior::Continue {
            max_continues: 2,
            on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
        }),
    }
}

#[test]
fn budget_policy_round_trips_every_variant() {
    let cases = vec![
        BudgetPolicy::Unlimited,
        BudgetPolicy::TotalSteps { value: 100 },
        BudgetPolicy::PerLoop { value: 10 },
        BudgetPolicy::PerAttempt { value: 3 },
    ];
    for c in cases {
        let json = serde_json::to_string(&c).unwrap();
        let back: BudgetPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}

#[test]
fn budget_behavior_round_trips_every_variant_including_nested() {
    let cases = vec![
        BudgetExhaustedBehavior::Escalate,
        BudgetExhaustedBehavior::Fail,
        BudgetExhaustedBehavior::Continue {
            max_continues: 0,
            on_exhausted: Box::new(BudgetExhaustedBehavior::Escalate),
        },
        nested(),
    ];
    for c in cases {
        let json = serde_json::to_string(&c).unwrap();
        let back: BudgetExhaustedBehavior = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}

#[test]
fn budget_policy_exact_serialized_bytes() {
    assert_eq!(
        serde_json::to_string(&BudgetPolicy::Unlimited).unwrap(),
        r#"{"kind":"unlimited"}"#
    );
    assert_eq!(
        serde_json::to_string(&BudgetPolicy::TotalSteps { value: 100 }).unwrap(),
        r#"{"kind":"total_steps","value":100}"#
    );
    assert_eq!(
        serde_json::to_string(&BudgetPolicy::PerLoop { value: 10 }).unwrap(),
        r#"{"kind":"per_loop","value":10}"#
    );
    assert_eq!(
        serde_json::to_string(&BudgetPolicy::PerAttempt { value: 3 }).unwrap(),
        r#"{"kind":"per_attempt","value":3}"#
    );
}

#[test]
fn budget_behavior_exact_serialized_bytes() {
    assert_eq!(
        serde_json::to_string(&BudgetExhaustedBehavior::Escalate).unwrap(),
        r#"{"kind":"escalate"}"#
    );
    assert_eq!(
        serde_json::to_string(&BudgetExhaustedBehavior::Fail).unwrap(),
        r#"{"kind":"fail"}"#
    );
    assert_eq!(
        serde_json::to_string(&BudgetExhaustedBehavior::Continue {
            max_continues: 2,
            on_exhausted: Box::new(BudgetExhaustedBehavior::Fail),
        })
        .unwrap(),
        r#"{"kind":"continue","max_continues":2,"on_exhausted":{"kind":"fail"}}"#
    );
    assert_eq!(
        serde_json::to_string(&nested()).unwrap(),
        r#"{"kind":"continue","max_continues":1,"on_exhausted":{"kind":"continue","max_continues":2,"on_exhausted":{"kind":"fail"}}}"#
    );
}

#[test]
fn budget_policy_unknown_kind_errors() {
    assert!(serde_json::from_str::<BudgetPolicy>(r#"{"kind":"per_goal","value":1}"#).is_err());
}

#[test]
fn budget_policy_missing_kind_errors() {
    assert!(serde_json::from_str::<BudgetPolicy>(r#"{"value":1}"#).is_err());
}

#[test]
fn budget_behavior_unknown_kind_errors() {
    assert!(serde_json::from_str::<BudgetExhaustedBehavior>(r#"{"kind":"retry"}"#).is_err());
}

#[test]
fn budget_behavior_missing_kind_errors() {
    assert!(serde_json::from_str::<BudgetExhaustedBehavior>(r#"{"max_continues":1}"#).is_err());
}

#[test]
fn budget_limits_no_regression() {
    // BudgetLimits is untouched: default is all-None and round-trips.
    let l = BudgetLimits::default();
    let json = serde_json::to_string(&l).unwrap();
    let back: BudgetLimits = serde_json::from_str(&json).unwrap();
    assert_eq!(l, back);
    assert!(l.max_turns.is_none());
    let populated = BudgetLimits {
        max_turns: Some(5),
        ..Default::default()
    };
    let back2: BudgetLimits =
        serde_json::from_str(&serde_json::to_string(&populated).unwrap()).unwrap();
    assert_eq!(populated, back2);
}

#[derive(serde::Deserialize)]
struct BudgetFixtureSuite {
    policies: Vec<BudgetPolicy>,
    behaviors: Vec<BudgetExhaustedBehavior>,
}

#[test]
fn fixture_replay_budget_policy() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../fixtures/budget_policy/cases.json");
    let raw = std::fs::read_to_string(&path).expect("fixture present");
    // Deserialize each case, re-serialize canonically, assert byte-equality
    // against a canonical re-serialization of the parsed JSON value.
    let suite: BudgetFixtureSuite = serde_json::from_str(&raw).unwrap();
    let root: serde_json::Value = serde_json::from_str(&raw).unwrap();

    for (i, policy) in suite.policies.iter().enumerate() {
        let ours = serde_json::to_string(policy).unwrap();
        let canonical = serde_json::to_string(&root["policies"][i]).unwrap();
        assert_eq!(ours, canonical, "policy[{i}] not byte-identical");
    }
    for (i, behavior) in suite.behaviors.iter().enumerate() {
        let ours = serde_json::to_string(behavior).unwrap();
        let canonical = serde_json::to_string(&root["behaviors"][i]).unwrap();
        assert_eq!(ours, canonical, "behavior[{i}] not byte-identical");
    }
}
