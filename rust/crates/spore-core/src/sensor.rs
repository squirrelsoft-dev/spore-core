//! Issue #10 — `SensorChain`: post-action feedback controls and output quality
//! evaluation.
//!
//! Sensors observe the agent's actions (tool calls, tool results, agent
//! responses) at defined trigger points and emit `SensorResult`s. The chain
//! is a registry plus a fan-out evaluator: it runs every sensor registered
//! for a trigger and returns all results without short-circuiting. The
//! harness decides routing (Warn → inject observation; Halt → stop).
//!
//! See `docs/harness-engineering-concepts.md` § "SensorChain" for rules this
//! module enforces.
//!
//! ## Rules enforced
//!   - `fire` runs every sensor whose `config.triggers` contains the trigger
//!     and returns all results — the chain never short-circuits.
//!   - `Computational` sensors run on every matching trigger.
//!   - `Inferential` sensors are gated by `run_every_n_turns` (modulo the
//!     `SensorInput::turn_number`) and `run_on_phases` (if set, the input's
//!     `phase` must match).
//!   - `stats` aggregates fire history; `fire_rate` = `total_fires` /
//!     `sessions_observed` clamped to `[0.0, 1.0]`.
//!   - `signal_quality_report` flags:
//!       - `NeverFired` — sensor observed ≥ `min_sessions` distinct sessions
//!         (across all sensors firing) yet never fired itself.
//!       - `AlwaysFiring` — sensor's fire-rate > its
//!         `low_signal_threshold.always_fired_rate`.
//!   - `Sensor::evaluate` is permitted side-effect-free observation only —
//!     the trait does not expose any mutator on the input.
//!   - Trigger matching for `PostTool`: a sensor configured with
//!     `PostTool{tool_name: ""}` (empty) matches any tool. A non-empty
//!     `tool_name` matches only exact equality.
//!
//! ## Implementor notes
//!   - `SensorInput` carries `session_id`, `turn_number`, and `phase` so the
//!     chain can gate inferential sensors and track per-session firing
//!     history without leaning on `SessionState` internals.
//!   - The standard impl is in-memory. Production deployments wire in a
//!     durable history store if cross-process stats are needed.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::harness::{BoxFut, SessionId, SessionState};
use crate::memory::Timestamp;
use crate::model::{ToolCall, ToolResult};
use crate::tool_registry::TaskPhase;

// ============================================================================
// Identity
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SensorId(pub String);

impl SensorId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// ============================================================================
// Enums
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SensorTrigger {
    PostTool { tool_name: String },
    PostTurn,
    PostSession,
    Continuous,
    OnToolError,
    OnCompaction,
}

impl SensorTrigger {
    /// True if this configured trigger matches the trigger that fired.
    /// `PostTool { tool_name: "" }` is a wildcard matching any `PostTool`.
    pub fn matches(&self, fired: &SensorTrigger) -> bool {
        match (self, fired) {
            (
                SensorTrigger::PostTool { tool_name: a },
                SensorTrigger::PostTool { tool_name: b },
            ) => a.is_empty() || a == b,
            (a, b) => a == b,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorKind {
    Computational,
    Inferential,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorOutcome {
    Pass,
    Warn,
    Halt,
}

// ============================================================================
// Records
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorInput {
    pub session_id: SessionId,
    #[serde(default)]
    pub turn_number: Option<u32>,
    #[serde(default)]
    pub phase: Option<TaskPhase>,
    #[serde(default)]
    pub tool_call: Option<ToolCall>,
    #[serde(default)]
    pub tool_result: Option<ToolResult>,
    #[serde(default)]
    pub agent_response: Option<String>,
    pub session_state: SessionState,
}

impl SensorInput {
    pub fn new(session_id: SessionId, session_state: SessionState) -> Self {
        Self {
            session_id,
            turn_number: None,
            phase: None,
            tool_call: None,
            tool_result: None,
            agent_response: None,
            session_state,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorResult {
    pub sensor_id: SensorId,
    pub outcome: SensorOutcome,
    #[serde(default)]
    pub observation: Option<String>,
    pub detail: String,
    pub fired_at: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorSignalThresholds {
    pub never_fired_after_n_sessions: u32,
    pub always_fired_rate: f32,
}

impl Default for SensorSignalThresholds {
    fn default() -> Self {
        Self {
            never_fired_after_n_sessions: 10,
            always_fired_rate: 0.9,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorConfig {
    pub id: SensorId,
    pub name: String,
    pub kind: SensorKind,
    pub triggers: Vec<SensorTrigger>,
    #[serde(default)]
    pub run_every_n_turns: Option<u32>,
    #[serde(default)]
    pub run_on_phases: Option<Vec<TaskPhase>>,
    #[serde(default)]
    pub low_signal_threshold: SensorSignalThresholds,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorStats {
    pub sensor_id: SensorId,
    pub total_fires: u32,
    pub warn_count: u32,
    pub halt_count: u32,
    pub pass_count: u32,
    pub fire_rate: f32,
    #[serde(default)]
    pub last_fired: Option<Timestamp>,
    pub low_signal_flag: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SensorSignalFlag {
    NeverFired {
        sensor_id: SensorId,
        sessions_observed: u32,
    },
    AlwaysFiring {
        sensor_id: SensorId,
        fire_rate: f32,
    },
}

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SensorError {
    #[error("sensor already registered: {0:?}")]
    AlreadyRegistered(SensorId),
    #[error("validation failed: {reason}")]
    ValidationFailed { reason: String },
}

// ============================================================================
// Traits
// ============================================================================

/// A single sensor. Object-safe via [`BoxFut`].
pub trait Sensor: Send + Sync {
    fn evaluate<'a>(&'a self, input: &'a SensorInput) -> BoxFut<'a, SensorResult>;
    fn config(&self) -> SensorConfig;
}

/// Registry + fan-out evaluator. Object-safe via [`BoxFut`].
pub trait SensorChain: Send + Sync {
    fn register(&self, sensor: Box<dyn Sensor>) -> Result<(), SensorError>;
    fn fire<'a>(
        &'a self,
        trigger: SensorTrigger,
        input: &'a SensorInput,
    ) -> BoxFut<'a, Vec<SensorResult>>;
    fn stats<'a>(&'a self, since: Option<Timestamp>) -> BoxFut<'a, Vec<SensorStats>>;
    fn signal_quality_report<'a>(&'a self, min_sessions: u32) -> BoxFut<'a, Vec<SensorSignalFlag>>;
}

// ============================================================================
// Standard implementation
// ============================================================================

/// Reference `SensorChain`. In-memory; sufficient for tests, short-lived
/// processes, and as a building block under a durable wrapper.
#[derive(Default)]
pub struct StandardSensorChain {
    inner: Mutex<Store>,
}

#[derive(Default)]
struct Store {
    sensors: Vec<SensorEntry>,
    /// Append-only history. Holds (result, session_id) so we can compute
    /// `sessions_observed` per-sensor for the signal-quality report.
    history: Vec<HistoryRecord>,
    /// All session ids the chain has been asked to fire against, regardless
    /// of whether any sensor fired for them. Used as the denominator for
    /// "never-fired" detection.
    sessions_seen: HashSet<SessionId>,
    /// Optional pinned "now" for deterministic tests.
    now_override: Option<Timestamp>,
}

struct SensorEntry {
    config: SensorConfig,
    sensor: Arc<dyn Sensor>,
}

#[derive(Clone)]
struct HistoryRecord {
    sensor_id: SensorId,
    #[allow(dead_code)]
    session_id: SessionId,
    outcome: SensorOutcome,
    fired_at: Timestamp,
}

impl StandardSensorChain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pin the "now" timestamp recorded on result history. Tests use this for
    /// determinism; otherwise the system clock is consulted.
    pub fn set_now(&self, now: Timestamp) {
        self.inner.lock().unwrap().now_override = Some(now);
    }
}

fn now_rfc3339() -> Timestamp {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Timestamp::new(format_rfc3339_secs(secs))
}

fn format_rfc3339_secs(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let hh = (rem / 3600) as u32;
    let mm = ((rem % 3600) / 60) as u32;
    let ss = (rem % 60) as u32;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y0 = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y0 + 1 } else { y0 };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Decide whether an inferential sensor should run on this input, given its
/// gating fields. Computational sensors are unconditional.
fn inferential_gate_open(cfg: &SensorConfig, input: &SensorInput) -> bool {
    if cfg.kind == SensorKind::Computational {
        return true;
    }
    if let Some(allowed) = &cfg.run_on_phases {
        match (&input.phase, allowed.is_empty()) {
            (_, true) => {} // empty list = no phase restriction
            (Some(p), _) if allowed.contains(p) => {}
            _ => return false,
        }
    }
    if let Some(n) = cfg.run_every_n_turns {
        if n == 0 {
            return false;
        }
        // Fire when turn_number is a multiple of n. Missing turn_number means
        // we can't gate — default to firing so caller pays the cost rather
        // than silently dropping inferential evidence.
        if let Some(t) = input.turn_number {
            if t % n != 0 {
                return false;
            }
        }
    }
    true
}

impl StandardSensorChain {
    fn register_impl(&self, sensor: Box<dyn Sensor>) -> Result<(), SensorError> {
        let cfg = sensor.config();
        if cfg.triggers.is_empty() {
            return Err(SensorError::ValidationFailed {
                reason: "sensor must declare at least one trigger".into(),
            });
        }
        let mut store = self.inner.lock().unwrap();
        if store.sensors.iter().any(|e| e.config.id == cfg.id) {
            return Err(SensorError::AlreadyRegistered(cfg.id));
        }
        store.sensors.push(SensorEntry {
            config: cfg,
            sensor: Arc::from(sensor),
        });
        Ok(())
    }

    async fn fire_impl(&self, trigger: SensorTrigger, input: &SensorInput) -> Vec<SensorResult> {
        // Snapshot eligible sensors (cloned `Arc`s) under the lock, then run
        // them outside the lock so `evaluate` cannot deadlock by re-entering
        // the chain.
        let candidates: Vec<(SensorId, Arc<dyn Sensor>)> = {
            let mut store = self.inner.lock().unwrap();
            store.sessions_seen.insert(input.session_id.clone());
            store
                .sensors
                .iter()
                .filter(|e| e.config.triggers.iter().any(|t| t.matches(&trigger)))
                .filter(|e| inferential_gate_open(&e.config, input))
                .map(|e| (e.config.id.clone(), e.sensor.clone()))
                .collect()
        };

        let mut results = Vec::with_capacity(candidates.len());
        for (sensor_id, sensor) in candidates {
            let result = sensor.evaluate(input).await;
            let mut store = self.inner.lock().unwrap();
            store.history.push(HistoryRecord {
                sensor_id: sensor_id.clone(),
                session_id: input.session_id.clone(),
                outcome: result.outcome,
                fired_at: result.fired_at.clone(),
            });
            results.push(result);
        }

        let _ = now_rfc3339; // keep symbol live for impls that want it
        results
    }

    fn stats_impl(&self, since: Option<Timestamp>) -> Vec<SensorStats> {
        #[derive(Default)]
        struct Agg {
            total: u32,
            pass: u32,
            warn: u32,
            halt: u32,
            last: Option<Timestamp>,
        }
        let store = self.inner.lock().unwrap();
        let mut by_sensor: HashMap<SensorId, Agg> = HashMap::new();

        for entry in &store.sensors {
            by_sensor.insert(entry.config.id.clone(), Agg::default());
        }
        for rec in &store.history {
            if let Some(cutoff) = &since {
                if rec.fired_at.as_str() < cutoff.as_str() {
                    continue;
                }
            }
            let e = by_sensor.entry(rec.sensor_id.clone()).or_default();
            e.total += 1;
            match rec.outcome {
                SensorOutcome::Pass => e.pass += 1,
                SensorOutcome::Warn => e.warn += 1,
                SensorOutcome::Halt => e.halt += 1,
            }
            e.last = Some(rec.fired_at.clone());
        }

        let sessions_total = store.sessions_seen.len() as f32;
        let mut out = Vec::with_capacity(by_sensor.len());
        for (id, agg) in by_sensor {
            let Agg {
                total,
                pass,
                warn,
                halt,
                last,
            } = agg;
            let cfg = store
                .sensors
                .iter()
                .find(|e| e.config.id == id)
                .map(|e| e.config.clone());
            let fire_rate = if sessions_total == 0.0 {
                0.0
            } else {
                (total as f32 / sessions_total).clamp(0.0, 1.0)
            };
            let low_signal_flag = if let Some(c) = &cfg {
                fire_rate > c.low_signal_threshold.always_fired_rate
                    || (total == 0
                        && store.sessions_seen.len() as u32
                            >= c.low_signal_threshold.never_fired_after_n_sessions)
            } else {
                false
            };
            out.push(SensorStats {
                sensor_id: id,
                total_fires: total,
                warn_count: warn,
                halt_count: halt,
                pass_count: pass,
                fire_rate,
                last_fired: last,
                low_signal_flag,
            });
        }
        out.sort_by(|a, b| a.sensor_id.0.cmp(&b.sensor_id.0));
        out
    }

    fn signal_quality_report_impl(&self, min_sessions: u32) -> Vec<SensorSignalFlag> {
        let store = self.inner.lock().unwrap();
        let sessions_observed = store.sessions_seen.len() as u32;
        let mut out = Vec::new();

        if sessions_observed < min_sessions {
            return out;
        }

        for entry in &store.sensors {
            let fires: Vec<&HistoryRecord> = store
                .history
                .iter()
                .filter(|r| r.sensor_id == entry.config.id)
                .collect();
            let total = fires.len() as u32;
            let fire_rate = if sessions_observed == 0 {
                0.0
            } else {
                (total as f32 / sessions_observed as f32).clamp(0.0, 1.0)
            };

            if total == 0
                && sessions_observed
                    >= entry
                        .config
                        .low_signal_threshold
                        .never_fired_after_n_sessions
            {
                out.push(SensorSignalFlag::NeverFired {
                    sensor_id: entry.config.id.clone(),
                    sessions_observed,
                });
            } else if fire_rate > entry.config.low_signal_threshold.always_fired_rate {
                out.push(SensorSignalFlag::AlwaysFiring {
                    sensor_id: entry.config.id.clone(),
                    fire_rate,
                });
            }
        }
        out.sort_by(|a, b| {
            let key = |f: &SensorSignalFlag| match f {
                SensorSignalFlag::NeverFired { sensor_id, .. } => ("a", sensor_id.0.clone()),
                SensorSignalFlag::AlwaysFiring { sensor_id, .. } => ("b", sensor_id.0.clone()),
            };
            key(a).cmp(&key(b))
        });
        out
    }
}

impl SensorChain for StandardSensorChain {
    fn register(&self, sensor: Box<dyn Sensor>) -> Result<(), SensorError> {
        self.register_impl(sensor)
    }

    fn fire<'a>(
        &'a self,
        trigger: SensorTrigger,
        input: &'a SensorInput,
    ) -> BoxFut<'a, Vec<SensorResult>> {
        Box::pin(async move { self.fire_impl(trigger, input).await })
    }

    fn stats<'a>(&'a self, since: Option<Timestamp>) -> BoxFut<'a, Vec<SensorStats>> {
        Box::pin(async move { self.stats_impl(since) })
    }

    fn signal_quality_report<'a>(&'a self, min_sessions: u32) -> BoxFut<'a, Vec<SensorSignalFlag>> {
        Box::pin(async move { self.signal_quality_report_impl(min_sessions) })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn ts(s: &str) -> Timestamp {
        Timestamp::new(s)
    }

    // ── Programmable test sensor ────────────────────────────────────────────

    struct StubSensor {
        cfg: SensorConfig,
        outcome: SensorOutcome,
    }

    impl Sensor for StubSensor {
        fn evaluate<'a>(&'a self, _input: &'a SensorInput) -> BoxFut<'a, SensorResult> {
            Box::pin(async move {
                SensorResult {
                    sensor_id: self.cfg.id.clone(),
                    outcome: self.outcome,
                    observation: if self.outcome == SensorOutcome::Warn {
                        Some("warn-obs".into())
                    } else {
                        None
                    },
                    detail: format!("{:?}", self.outcome),
                    fired_at: ts("2026-05-16T00:00:00Z"),
                }
            })
        }
        fn config(&self) -> SensorConfig {
            self.cfg.clone()
        }
    }

    fn computational(
        id: &str,
        triggers: Vec<SensorTrigger>,
        outcome: SensorOutcome,
    ) -> Box<StubSensor> {
        Box::new(StubSensor {
            cfg: SensorConfig {
                id: SensorId::new(id),
                name: id.into(),
                kind: SensorKind::Computational,
                triggers,
                run_every_n_turns: None,
                run_on_phases: None,
                low_signal_threshold: SensorSignalThresholds::default(),
            },
            outcome,
        })
    }

    fn inferential(
        id: &str,
        triggers: Vec<SensorTrigger>,
        outcome: SensorOutcome,
        every_n: Option<u32>,
        phases: Option<Vec<TaskPhase>>,
    ) -> Box<StubSensor> {
        Box::new(StubSensor {
            cfg: SensorConfig {
                id: SensorId::new(id),
                name: id.into(),
                kind: SensorKind::Inferential,
                triggers,
                run_every_n_turns: every_n,
                run_on_phases: phases,
                low_signal_threshold: SensorSignalThresholds::default(),
            },
            outcome,
        })
    }

    fn input(sid: &str) -> SensorInput {
        SensorInput::new(SessionId::new(sid), SessionState::default())
    }

    // ── Rule: register validates triggers ───────────────────────────────────

    #[tokio::test]
    async fn register_rejects_empty_triggers() {
        let chain = StandardSensorChain::new();
        let s = computational("s1", vec![], SensorOutcome::Pass);
        let err = chain.register(s).unwrap_err();
        assert!(matches!(err, SensorError::ValidationFailed { .. }));
    }

    #[tokio::test]
    async fn register_rejects_duplicate_ids() {
        let chain = StandardSensorChain::new();
        chain
            .register(computational(
                "s1",
                vec![SensorTrigger::PostTurn],
                SensorOutcome::Pass,
            ))
            .unwrap();
        let err = chain
            .register(computational(
                "s1",
                vec![SensorTrigger::PostTurn],
                SensorOutcome::Pass,
            ))
            .unwrap_err();
        assert!(matches!(err, SensorError::AlreadyRegistered(_)));
    }

    // ── Rule: fire runs every matching sensor, returns all results ──────────

    #[tokio::test]
    async fn fire_runs_all_matching_sensors_no_short_circuit() {
        let chain = StandardSensorChain::new();
        chain
            .register(computational(
                "pass",
                vec![SensorTrigger::PostTurn],
                SensorOutcome::Pass,
            ))
            .unwrap();
        chain
            .register(computational(
                "warn",
                vec![SensorTrigger::PostTurn],
                SensorOutcome::Warn,
            ))
            .unwrap();
        chain
            .register(computational(
                "halt",
                vec![SensorTrigger::PostTurn],
                SensorOutcome::Halt,
            ))
            .unwrap();
        let results = chain.fire(SensorTrigger::PostTurn, &input("s1")).await;
        assert_eq!(results.len(), 3);
        let outcomes: HashSet<_> = results.iter().map(|r| r.outcome).collect();
        assert!(outcomes.contains(&SensorOutcome::Pass));
        assert!(outcomes.contains(&SensorOutcome::Warn));
        assert!(outcomes.contains(&SensorOutcome::Halt));
    }

    // ── Rule: triggers filter ───────────────────────────────────────────────

    #[tokio::test]
    async fn fire_ignores_sensors_without_matching_trigger() {
        let chain = StandardSensorChain::new();
        chain
            .register(computational(
                "post-turn",
                vec![SensorTrigger::PostTurn],
                SensorOutcome::Pass,
            ))
            .unwrap();
        chain
            .register(computational(
                "post-session",
                vec![SensorTrigger::PostSession],
                SensorOutcome::Pass,
            ))
            .unwrap();
        let results = chain.fire(SensorTrigger::PostTurn, &input("s1")).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].sensor_id.0, "post-turn");
    }

    // ── Rule: PostTool wildcard matches any tool, named matches exact ───────

    #[tokio::test]
    async fn post_tool_wildcard_and_named_matching() {
        let chain = StandardSensorChain::new();
        chain
            .register(computational(
                "any",
                vec![SensorTrigger::PostTool {
                    tool_name: "".into(),
                }],
                SensorOutcome::Pass,
            ))
            .unwrap();
        chain
            .register(computational(
                "bash-only",
                vec![SensorTrigger::PostTool {
                    tool_name: "bash".into(),
                }],
                SensorOutcome::Pass,
            ))
            .unwrap();
        let r1 = chain
            .fire(
                SensorTrigger::PostTool {
                    tool_name: "bash".into(),
                },
                &input("s1"),
            )
            .await;
        assert_eq!(r1.len(), 2);
        let r2 = chain
            .fire(
                SensorTrigger::PostTool {
                    tool_name: "edit".into(),
                },
                &input("s2"),
            )
            .await;
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].sensor_id.0, "any");
    }

    // ── Rule: Inferential gated by run_every_n_turns ────────────────────────

    #[tokio::test]
    async fn inferential_run_every_n_turns_gating() {
        let chain = StandardSensorChain::new();
        chain
            .register(inferential(
                "judge",
                vec![SensorTrigger::PostTurn],
                SensorOutcome::Warn,
                Some(3),
                None,
            ))
            .unwrap();
        // Turn 1, 2 → skip. Turn 3 → fire. Turn 6 → fire.
        let mut input = input("s1");
        input.turn_number = Some(1);
        assert!(chain.fire(SensorTrigger::PostTurn, &input).await.is_empty());
        input.turn_number = Some(2);
        assert!(chain.fire(SensorTrigger::PostTurn, &input).await.is_empty());
        input.turn_number = Some(3);
        assert_eq!(chain.fire(SensorTrigger::PostTurn, &input).await.len(), 1);
        input.turn_number = Some(6);
        assert_eq!(chain.fire(SensorTrigger::PostTurn, &input).await.len(), 1);
    }

    // ── Rule: Inferential gated by run_on_phases ────────────────────────────

    #[tokio::test]
    async fn inferential_run_on_phases_gating() {
        let chain = StandardSensorChain::new();
        chain
            .register(inferential(
                "judge",
                vec![SensorTrigger::PostTurn],
                SensorOutcome::Pass,
                None,
                Some(vec![TaskPhase::Execution]),
            ))
            .unwrap();
        let mut input = input("s1");
        input.phase = Some(TaskPhase::Planning);
        assert!(chain.fire(SensorTrigger::PostTurn, &input).await.is_empty());
        input.phase = Some(TaskPhase::Execution);
        assert_eq!(chain.fire(SensorTrigger::PostTurn, &input).await.len(), 1);
    }

    // ── Rule: Computational sensors always fire on matching trigger ─────────

    #[tokio::test]
    async fn computational_ignores_turn_gating() {
        let chain = StandardSensorChain::new();
        // Even with `run_every_n_turns` set, computational should fire every time.
        let mut cfg = computational("c", vec![SensorTrigger::PostTurn], SensorOutcome::Pass);
        cfg.cfg.run_every_n_turns = Some(99);
        chain.register(cfg).unwrap();
        let mut i = input("s1");
        i.turn_number = Some(1);
        assert_eq!(chain.fire(SensorTrigger::PostTurn, &i).await.len(), 1);
    }

    // ── Rule: stats aggregates per-sensor outcomes ──────────────────────────

    #[tokio::test]
    async fn stats_aggregates_outcomes_and_fire_rate() {
        let chain = StandardSensorChain::new();
        chain
            .register(computational(
                "warner",
                vec![SensorTrigger::PostTurn],
                SensorOutcome::Warn,
            ))
            .unwrap();
        for i in 0..4 {
            chain
                .fire(SensorTrigger::PostTurn, &input(&format!("s{i}")))
                .await;
        }
        let stats = chain.stats(None).await;
        assert_eq!(stats.len(), 1);
        let s = &stats[0];
        assert_eq!(s.total_fires, 4);
        assert_eq!(s.warn_count, 4);
        assert_eq!(s.halt_count, 0);
        assert_eq!(s.pass_count, 0);
        assert!((s.fire_rate - 1.0).abs() < 1e-6);
    }

    // ── Rule: signal_quality_report flags AlwaysFiring ──────────────────────

    #[tokio::test]
    async fn signal_quality_flags_always_firing() {
        let chain = StandardSensorChain::new();
        let mut cfg = computational("noisy", vec![SensorTrigger::PostTurn], SensorOutcome::Warn);
        cfg.cfg.low_signal_threshold = SensorSignalThresholds {
            never_fired_after_n_sessions: 100,
            always_fired_rate: 0.5,
        };
        chain.register(cfg).unwrap();
        for i in 0..5 {
            chain
                .fire(SensorTrigger::PostTurn, &input(&format!("s{i}")))
                .await;
        }
        let flags = chain.signal_quality_report(5).await;
        assert!(flags.iter().any(|f| matches!(
            f,
            SensorSignalFlag::AlwaysFiring { sensor_id, .. } if sensor_id.0 == "noisy"
        )));
    }

    // ── Rule: signal_quality_report flags NeverFired ────────────────────────

    #[tokio::test]
    async fn signal_quality_flags_never_fired() {
        let chain = StandardSensorChain::new();
        // Sensor only fires on PostSession.
        let mut cfg = computational(
            "quiet",
            vec![SensorTrigger::PostSession],
            SensorOutcome::Pass,
        );
        cfg.cfg.low_signal_threshold = SensorSignalThresholds {
            never_fired_after_n_sessions: 3,
            always_fired_rate: 0.9,
        };
        chain.register(cfg).unwrap();
        // Fire PostTurn many times against many sessions, so chain sees them
        // but our sensor never fires.
        for i in 0..5 {
            chain
                .fire(SensorTrigger::PostTurn, &input(&format!("s{i}")))
                .await;
        }
        let flags = chain.signal_quality_report(3).await;
        assert!(flags.iter().any(|f| matches!(
            f,
            SensorSignalFlag::NeverFired { sensor_id, sessions_observed }
                if sensor_id.0 == "quiet" && *sessions_observed >= 3
        )));
    }

    // ── Rule: signal_quality_report respects min_sessions floor ─────────────

    #[tokio::test]
    async fn signal_quality_respects_min_sessions() {
        let chain = StandardSensorChain::new();
        chain
            .register(computational(
                "quiet",
                vec![SensorTrigger::PostSession],
                SensorOutcome::Pass,
            ))
            .unwrap();
        chain.fire(SensorTrigger::PostTurn, &input("s1")).await;
        // min_sessions=10 → not enough data, no flags.
        let flags = chain.signal_quality_report(10).await;
        assert!(flags.is_empty());
    }

    // ── Edge: history is appended in-order ──────────────────────────────────

    #[tokio::test]
    async fn history_is_recorded_in_fire_order() {
        let chain = StandardSensorChain::new();
        chain
            .register(computational(
                "s1",
                vec![SensorTrigger::PostTurn],
                SensorOutcome::Pass,
            ))
            .unwrap();
        chain.fire(SensorTrigger::PostTurn, &input("a")).await;
        chain.fire(SensorTrigger::PostTurn, &input("b")).await;
        let stats = chain.stats(None).await;
        assert_eq!(stats[0].total_fires, 2);
    }

    // ── Send-safety smoke ───────────────────────────────────────────────────

    #[tokio::test]
    async fn chain_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        let chain: Arc<dyn SensorChain> = Arc::new(StandardSensorChain::new());
        assert_send_sync(&chain);
    }

    // ── Fixture replay ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn fixture_replay_signal_quality() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/sensor_chain/signal_quality_basic.json");
        let raw = std::fs::read_to_string(&path).expect("fixture present");
        let case: FixtureCase = serde_json::from_str(&raw).unwrap();

        let chain = StandardSensorChain::new();
        for s in case.sensors {
            let cfg = SensorConfig {
                id: SensorId::new(&s.id),
                name: s.id.clone(),
                kind: s.kind,
                triggers: s.triggers,
                run_every_n_turns: None,
                run_on_phases: None,
                low_signal_threshold: s.thresholds,
            };
            chain
                .register(Box::new(StubSensor {
                    cfg,
                    outcome: s.outcome,
                }))
                .unwrap();
        }
        for ev in case.events {
            chain.fire(ev.trigger, &input(&ev.session_id)).await;
        }
        let flags = chain.signal_quality_report(case.min_sessions).await;
        let want_never: HashSet<String> = case.expected.never_fired.iter().cloned().collect();
        let want_always: HashSet<String> = case.expected.always_firing.iter().cloned().collect();
        let mut got_never = HashSet::new();
        let mut got_always = HashSet::new();
        for f in flags {
            match f {
                SensorSignalFlag::NeverFired { sensor_id, .. } => {
                    got_never.insert(sensor_id.0);
                }
                SensorSignalFlag::AlwaysFiring { sensor_id, .. } => {
                    got_always.insert(sensor_id.0);
                }
            }
        }
        assert_eq!(got_never, want_never);
        assert_eq!(got_always, want_always);
    }

    #[derive(Deserialize)]
    struct FixtureCase {
        sensors: Vec<FixtureSensor>,
        events: Vec<FixtureEvent>,
        min_sessions: u32,
        expected: FixtureExpected,
    }
    #[derive(Deserialize)]
    struct FixtureSensor {
        id: String,
        kind: SensorKind,
        triggers: Vec<SensorTrigger>,
        outcome: SensorOutcome,
        thresholds: SensorSignalThresholds,
    }
    #[derive(Deserialize)]
    struct FixtureEvent {
        trigger: SensorTrigger,
        session_id: String,
    }
    #[derive(Deserialize)]
    struct FixtureExpected {
        never_fired: Vec<String>,
        always_firing: Vec<String>,
    }
}
