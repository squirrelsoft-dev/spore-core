//! ExecutionRegistry — runtime resolution of serializable strategy handles
//! (Composable Execution A.3, issue #120; part of the #117–#131 refactor).
//!
//! # Types
//! - [`ExecutionRegistry`] — five `HashMap`s of `Arc<dyn _>` collaborators keyed
//!   by string: `agents`, `toolsets`, `schemas`, `verifiers`, `custom` (custom
//!   strategies). Trait objects never serialize, so the registry derives
//!   `Clone` (Arc maps clone cheaply) but NOT `Serialize`/`Deserialize`.
//! - [`ExecutionRegistryBuilder`] — fluent assembler mirroring `HarnessBuilder`.
//! - [`StrategyResolution`] — the result of resolving a [`StrategyRef`]: either a
//!   borrowed built-in [`LoopStrategy`] or a borrowed custom `Arc<dyn RunStrategy>`.
//! - [`EscalationMode`] — the HITL-vs-AFK config knob (PRD goal #7).
//!
//! # Methods
//! - `resolve_agent`/`resolve_toolset`/`resolve_schema`/`resolve_verifier` —
//!   read-only, non-async pure lookups. Each `*Ref` type maps to exactly ONE map
//!   ([`SchemaRef`] → `schemas`).
//! - `resolve_strategy` — `Result<StrategyResolution, HarnessError>`; a missing
//!   `Custom` key returns the recoverable [`HarnessError::StrategyNotFound`].
//! - `register_strategy` — register a custom `Arc<dyn RunStrategy>` at startup.
//! - `validate` — walks a [`Task`]'s strategy tree, returning the FIRST
//!   unresolved handle as [`HarnessError::UnresolvedHandle`] (or
//!   [`HarnessError::StrategyNotFound`] for a missing custom key). Called at the
//!   entry of `StandardHarness::run` so an unresolved handle is a STARTUP error,
//!   before the first turn.
//!
//! # Rules enforced
//! - Unresolved handle (missing agent/toolset/schema) → startup error before the
//!   first turn ([`HarnessError::UnresolvedHandle`]).
//! - A missing `StrategyRef::Custom` key → recoverable
//!   [`HarnessError::StrategyNotFound`], never a panic.
//! - Resume re-resolves every handle from the registry with no reconfiguration:
//!   trait objects never enter the serialized `Task`, only string handles do.
//! - `register_strategy` makes a custom strategy resolvable by key.
//! - HITL-vs-AFK escalation is selectable via [`EscalationMode`] on
//!   `HarnessConfig`, not hardcoded.
//!
//! # Resolutions applied (do not re-litigate — pinned in #120)
//! - **Scope = ADDITIVE (Option B).** This slice ADDS the registry +
//!   `escalation_mode` to `HarnessConfig`; it does NOT remove the four
//!   single-collaborator fields (`agent`, `verifier`, `planner_agent`,
//!   `evaluator_agent`) nor touch the executor consumption sites. Those four
//!   carry a `Deprecated:` doc comment documenting the migration path; physical
//!   removal + executor migration to registry resolution lands in #124. The
//!   registry coexists with the deprecated fields this slice and is not yet read
//!   by the run bodies (that's #123/#124).
//! - The registry has exactly FIVE maps (no sixth).
//! - [`EscalationMode`] has NO `Default` impl (mirrors the budget-types
//!   discipline); the builder picks an explicit default (`SurfaceToHuman`).
//! - [`EscalationMode`] is STORED only this slice (#130 consumes it) and is NOT
//!   part of the serialized `Task` → no fixture for it.

use std::collections::HashMap;
use std::sync::Arc;

use crate::agent::Agent;
use crate::harness::{
    AgentRef, HarnessError, HillClimbingConfig, LoopStrategy, PlanExecuteConfig, RalphConfig,
    ReactConfig, RunStrategy, SchemaRef, SelfVerifyingConfig, StrategyRef, Task, ToolRegistry,
    ToolsetRef,
};
use crate::verifier::Verifier;

/// HITL-vs-AFK escalation knob (PRD goal #7: local vs. prod differ only by
/// config). Selects whether budget escalation surfaces to a human or proceeds
/// autonomously. Stored on `HarnessConfig` this slice; consumed in #130.
///
/// No `Default` impl by design — mirrors the budget-types discipline
/// (`BudgetExhaustedBehavior` has none). The `HarnessBuilder` picks an explicit
/// default ([`EscalationMode::SurfaceToHuman`]).
///
/// Has serde derives for symmetry with the other harness enums, but it is NOT
/// placed on the serialized `Task`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EscalationMode {
    /// Budget escalation pauses and surfaces to a human (HITL).
    SurfaceToHuman,
    /// Budget escalation proceeds autonomously (AFK / prod).
    Autonomous,
}

/// The result of resolving a [`StrategyRef`] against an [`ExecutionRegistry`]:
/// either the borrowed built-in [`LoopStrategy`] tree or the borrowed custom
/// `Arc<dyn RunStrategy>` looked up in [`ExecutionRegistry::custom`].
pub enum StrategyResolution<'a> {
    /// `StrategyRef::BuiltIn(ls)` resolves to the borrowed built-in tree.
    BuiltIn(&'a LoopStrategy),
    /// `StrategyRef::Custom(key)` resolves to the borrowed custom strategy.
    Custom(&'a Arc<dyn RunStrategy>),
}

impl std::fmt::Debug for StrategyResolution<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // `Arc<dyn RunStrategy>` isn't Debug; report the variant + tree only.
            StrategyResolution::BuiltIn(ls) => f.debug_tuple("BuiltIn").field(ls).finish(),
            StrategyResolution::Custom(_) => f.write_str("Custom(<dyn RunStrategy>)"),
        }
    }
}

/// Runtime resolver mapping serializable string handles (and
/// `StrategyRef::Custom` keys) to concrete `Arc<dyn _>` collaborators. See the
/// module docs for the full type/method/rule documentation.
///
/// Trait objects never serialize, so this type is NOT `Serialize`/`Deserialize`;
/// it derives `Clone` (the Arc maps clone cheaply). Build one with
/// [`ExecutionRegistry::builder`] or [`ExecutionRegistry::empty`].
#[derive(Clone, Default)]
pub struct ExecutionRegistry {
    agents: HashMap<String, Arc<dyn Agent>>,
    toolsets: HashMap<String, Arc<dyn ToolRegistry>>,
    schemas: HashMap<String, serde_json::Value>,
    verifiers: HashMap<String, Arc<dyn Verifier>>,
    custom: HashMap<String, Arc<dyn RunStrategy>>,
}

impl std::fmt::Debug for ExecutionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Trait objects aren't Debug; report key sets only.
        f.debug_struct("ExecutionRegistry")
            .field("agents", &self.agents.keys().collect::<Vec<_>>())
            .field("toolsets", &self.toolsets.keys().collect::<Vec<_>>())
            .field("schemas", &self.schemas.keys().collect::<Vec<_>>())
            .field("verifiers", &self.verifiers.keys().collect::<Vec<_>>())
            .field("custom", &self.custom.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ExecutionRegistry {
    /// An empty registry (no entries in any of the five maps).
    pub fn empty() -> Self {
        Self::default()
    }

    /// True when no entries exist in any of the five maps. Lets the harness skip
    /// startup validation for callers that never wire a registry (Option B
    /// additive scope — they still use the deprecated single-collaborator
    /// fields).
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
            && self.toolsets.is_empty()
            && self.schemas.is_empty()
            && self.verifiers.is_empty()
            && self.custom.is_empty()
    }

    /// Start a fluent [`ExecutionRegistryBuilder`].
    pub fn builder() -> ExecutionRegistryBuilder {
        ExecutionRegistryBuilder::default()
    }

    /// Consume this registry into a builder preserving all existing entries, so
    /// a caller (e.g. `HarnessBuilder`'s per-key convenience setters) can add
    /// more before re-[`build`](ExecutionRegistryBuilder::build)ing.
    pub fn into_builder(self) -> ExecutionRegistryBuilder {
        ExecutionRegistryBuilder { registry: self }
    }

    /// Resolve an [`AgentRef`] to its registered agent, or `None` if absent.
    pub fn resolve_agent(&self, r: &AgentRef) -> Option<&Arc<dyn Agent>> {
        self.agents.get(&r.0)
    }

    /// Resolve a [`ToolsetRef`] to its registered toolset, or `None` if absent.
    pub fn resolve_toolset(&self, r: &ToolsetRef) -> Option<&Arc<dyn ToolRegistry>> {
        self.toolsets.get(&r.0)
    }

    /// Resolve a [`SchemaRef`] to its registered JSON schema, or `None` if
    /// absent. ([`SchemaRef`] maps to the `schemas` map.)
    pub fn resolve_schema(&self, r: &SchemaRef) -> Option<&serde_json::Value> {
        self.schemas.get(&r.0)
    }

    /// Resolve a verifier key to its registered verifier, or `None` if absent.
    pub fn resolve_verifier(&self, key: &str) -> Option<&Arc<dyn Verifier>> {
        self.verifiers.get(key)
    }

    /// Resolve a [`StrategyRef`]: a `BuiltIn(ls)` borrows the built-in tree; a
    /// `Custom(key)` looks up [`custom`](Self::custom) and returns
    /// [`HarnessError::StrategyNotFound`] (recoverable) when the key is absent.
    pub fn resolve_strategy<'a>(
        &'a self,
        r: &'a StrategyRef,
    ) -> Result<StrategyResolution<'a>, HarnessError> {
        match r {
            StrategyRef::BuiltIn(ls) => Ok(StrategyResolution::BuiltIn(ls)),
            StrategyRef::Custom(key) => self
                .custom
                .get(key)
                .map(StrategyResolution::Custom)
                .ok_or_else(|| HarnessError::StrategyNotFound { key: key.clone() }),
        }
    }

    /// Register (or replace, last-wins) a custom strategy under `key`.
    pub fn register_strategy(&mut self, key: impl Into<String>, s: Arc<dyn RunStrategy>) {
        self.custom.insert(key.into(), s);
    }

    /// Validate that every handle referenced by `task.loop_strategy` resolves
    /// against this registry. Walks the strategy tree and returns the FIRST
    /// unresolved handle as [`HarnessError::UnresolvedHandle`] (or
    /// [`HarnessError::StrategyNotFound`] for a missing custom key). Returns
    /// `Ok(())` when the whole tree resolves. Called at the entry of
    /// `StandardHarness::run` so an unresolved handle is a startup error.
    pub fn validate(&self, task: &Task) -> Result<(), HarnessError> {
        self.walk_strategy(&task.loop_strategy)
    }

    /// Recursive tree-walk over a [`LoopStrategy`], checking every child handle.
    fn walk_strategy(&self, ls: &LoopStrategy) -> Result<(), HarnessError> {
        match ls {
            LoopStrategy::ReAct(ReactConfig {
                agent,
                toolset,
                output,
                ..
            }) => {
                self.check_agent(agent)?;
                self.check_toolset(toolset)?;
                if let Some(schema) = output {
                    self.check_schema(schema)?;
                }
                Ok(())
            }
            LoopStrategy::PlanExecute(PlanExecuteConfig { plan, execute, .. }) => {
                // A.5 (#124, Q3): the `plan` slot is STRUCTURED — it must yield a
                // task graph. A bare `ReAct` there needs an output schema.
                Self::check_structured_slot(plan, "plan")?;
                self.walk_strategy(plan)?;
                self.walk_strategy(execute)?;
                Ok(())
            }
            LoopStrategy::SelfVerifying(SelfVerifyingConfig { inner, evaluator }) => {
                // A.5: the `inner` (worker) slot is STRUCTURED — its result must be
                // evaluable. A bare `ReAct` worker needs an output schema.
                Self::check_structured_slot(inner, "worker")?;
                self.walk_strategy(inner)?;
                // The evaluator is a SchemaRef (the evaluator schema handle).
                self.check_schema(evaluator)?;
                Ok(())
            }
            LoopStrategy::Ralph(RalphConfig { inner, agent }) => {
                self.walk_strategy(inner)?;
                self.check_agent(agent)?;
                Ok(())
            }
            LoopStrategy::HillClimbing(HillClimbingConfig {
                inner, evaluator, ..
            }) => {
                // A.5: the `inner` (propose) slot is STRUCTURED — it must yield a
                // candidate. A bare `ReAct` proposer needs an output schema.
                Self::check_structured_slot(inner, "propose")?;
                self.walk_strategy(inner)?;
                // The evaluator is an AgentRef (the metric-evaluator agent).
                self.check_agent(evaluator)?;
                Ok(())
            }
        }
    }

    /// A.5 output-contract enforcement (#124, Q3): a bare `ReAct` feeding a
    /// STRUCTURED slot (`plan` ⇒ task graph, `propose` ⇒ candidate, `worker` ⇒
    /// evaluable result) MUST declare `ReAct.output = Some(SchemaRef)`. A combinator
    /// child carries its own contract, so this check applies only to the leaf.
    /// Returns [`HarnessError::InvalidConfiguration`] naming the offending slot.
    fn check_structured_slot(slot: &LoopStrategy, slot_name: &str) -> Result<(), HarnessError> {
        if let LoopStrategy::ReAct(ReactConfig { output: None, .. }) = slot {
            return Err(HarnessError::InvalidConfiguration(format!(
                "a bare ReAct in the structured `{slot_name}` slot requires \
                 `output = Some(schema)` so the slot yields a typed result"
            )));
        }
        Ok(())
    }

    fn check_agent(&self, r: &AgentRef) -> Result<(), HarnessError> {
        if self.agents.contains_key(&r.0) {
            Ok(())
        } else {
            Err(HarnessError::UnresolvedHandle {
                kind: "agent".to_string(),
                key: r.0.clone(),
            })
        }
    }

    fn check_toolset(&self, r: &ToolsetRef) -> Result<(), HarnessError> {
        if self.toolsets.contains_key(&r.0) {
            Ok(())
        } else {
            Err(HarnessError::UnresolvedHandle {
                kind: "toolset".to_string(),
                key: r.0.clone(),
            })
        }
    }

    fn check_schema(&self, r: &SchemaRef) -> Result<(), HarnessError> {
        if self.schemas.contains_key(&r.0) {
            Ok(())
        } else {
            Err(HarnessError::UnresolvedHandle {
                kind: "schema".to_string(),
                key: r.0.clone(),
            })
        }
    }
}

/// Fluent assembler for an [`ExecutionRegistry`], mirroring `HarnessBuilder`.
#[derive(Clone, Default)]
pub struct ExecutionRegistryBuilder {
    registry: ExecutionRegistry,
}

impl ExecutionRegistryBuilder {
    /// Register an agent under `key`.
    pub fn agent(mut self, key: impl Into<String>, agent: Arc<dyn Agent>) -> Self {
        self.registry.agents.insert(key.into(), agent);
        self
    }

    /// Register a toolset under `key`.
    pub fn toolset(mut self, key: impl Into<String>, toolset: Arc<dyn ToolRegistry>) -> Self {
        self.registry.toolsets.insert(key.into(), toolset);
        self
    }

    /// Register a JSON schema under `key`.
    pub fn schema(mut self, key: impl Into<String>, schema: serde_json::Value) -> Self {
        self.registry.schemas.insert(key.into(), schema);
        self
    }

    /// Register a verifier under `key`.
    pub fn verifier(mut self, key: impl Into<String>, verifier: Arc<dyn Verifier>) -> Self {
        self.registry.verifiers.insert(key.into(), verifier);
        self
    }

    /// Register a custom strategy under `key`.
    pub fn register_strategy(
        mut self,
        key: impl Into<String>,
        strategy: Arc<dyn RunStrategy>,
    ) -> Self {
        self.registry.custom.insert(key.into(), strategy);
        self
    }

    /// Finish and return the assembled [`ExecutionRegistry`].
    pub fn build(self) -> ExecutionRegistry {
        self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, AgentId, Context, TurnResult};
    use crate::harness::{
        AgentRef, BoxFut, BudgetPolicy, EmptyToolRegistry, ExecutionContext, LoopStrategy,
        ReactConfig, SchemaRef, SessionId, StrategyOutcome, Task, ToolsetRef,
    };
    use crate::verifier::{Verifier, VerifierInput, VerifierVerdict};

    // ---- Test-only stubs ---------------------------------------------------

    #[derive(Debug)]
    struct StubAgent;

    impl Agent for StubAgent {
        fn turn<'a>(&'a self, _context: Context) -> BoxFut<'a, TurnResult> {
            Box::pin(async { unreachable!("validate() must fail before any agent turn") })
        }
        fn id(&self) -> AgentId {
            AgentId::new("stub")
        }
    }

    #[derive(Debug)]
    struct StubVerifier;

    impl Verifier for StubVerifier {
        fn verify<'a>(&'a self, _input: &'a VerifierInput) -> BoxFut<'a, VerifierVerdict> {
            Box::pin(async { unreachable!("verifier not invoked in registry tests") })
        }
    }

    #[derive(Debug)]
    struct StubStrategy;

    impl RunStrategy for StubStrategy {
        fn run<'a>(&'a self, _cx: &'a mut ExecutionContext<'_>) -> BoxFut<'a, StrategyOutcome> {
            Box::pin(async { StrategyOutcome::Complete(String::new()) })
        }
    }

    fn react_leaf(agent: &str, toolset: &str) -> LoopStrategy {
        LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: 4 },
            agent: AgentRef(agent.to_string()),
            toolset: ToolsetRef(toolset.to_string()),
            output: None,
        })
    }

    fn fully_wired_registry() -> ExecutionRegistry {
        ExecutionRegistry::builder()
            .agent("a1", Arc::new(StubAgent))
            .toolset("t1", Arc::new(EmptyToolRegistry))
            .schema("s1", serde_json::json!({"type": "object"}))
            .verifier("v1", Arc::new(StubVerifier))
            .build()
    }

    // ---- resolve_* happy path + miss --------------------------------------

    #[test]
    fn resolve_each_happy_and_miss() {
        let reg = fully_wired_registry();

        assert!(reg.resolve_agent(&AgentRef("a1".into())).is_some());
        assert!(reg.resolve_agent(&AgentRef("nope".into())).is_none());

        assert!(reg.resolve_toolset(&ToolsetRef("t1".into())).is_some());
        assert!(reg.resolve_toolset(&ToolsetRef("nope".into())).is_none());

        assert!(reg.resolve_schema(&SchemaRef("s1".into())).is_some());
        assert!(reg.resolve_schema(&SchemaRef("nope".into())).is_none());

        assert!(reg.resolve_verifier("v1").is_some());
        assert!(reg.resolve_verifier("nope").is_none());
    }

    // ---- register_strategy + resolve_strategy(Custom) ----------------------

    #[test]
    fn register_then_resolve_custom_strategy() {
        let mut reg = ExecutionRegistry::empty();
        reg.register_strategy("mine::Custom", Arc::new(StubStrategy));

        let r = StrategyRef::Custom("mine::Custom".to_string());
        match reg.resolve_strategy(&r) {
            Ok(StrategyResolution::Custom(_)) => {}
            other => panic!("expected Custom resolution, got {other:?}"),
        }
    }

    #[test]
    fn resolve_builtin_strategy_borrows_tree() {
        let reg = ExecutionRegistry::empty();
        let r = StrategyRef::BuiltIn(react_leaf("a1", "t1"));
        match reg.resolve_strategy(&r) {
            Ok(StrategyResolution::BuiltIn(LoopStrategy::ReAct(_))) => {}
            other => panic!("expected BuiltIn react, got {other:?}"),
        }
    }

    // ---- missing custom key → recoverable StrategyNotFound, no panic -------

    #[test]
    fn missing_custom_key_is_recoverable_strategy_not_found() {
        let reg = ExecutionRegistry::empty();
        let r = StrategyRef::Custom("absent".to_string());
        let err = reg.resolve_strategy(&r).unwrap_err();
        assert_eq!(
            err,
            HarnessError::StrategyNotFound {
                key: "absent".to_string()
            }
        );
        // Returned, never panicked — reaching here proves it.
    }

    // ---- validate() unresolved handle → UnresolvedHandle -------------------

    #[test]
    fn validate_unresolved_agent_handle() {
        let reg = ExecutionRegistry::empty();
        let task = Task::new(
            "do it",
            SessionId::generate(),
            react_leaf("missing-agent", "t1"),
        );
        let err = reg.validate(&task).unwrap_err();
        assert_eq!(
            err,
            HarnessError::UnresolvedHandle {
                kind: "agent".to_string(),
                key: "missing-agent".to_string(),
            }
        );
    }

    #[test]
    fn validate_unresolved_toolset_handle() {
        let reg = ExecutionRegistry::builder()
            .agent("a1", Arc::new(StubAgent))
            .build();
        let task = Task::new(
            "do it",
            SessionId::generate(),
            react_leaf("a1", "missing-tools"),
        );
        let err = reg.validate(&task).unwrap_err();
        assert_eq!(
            err,
            HarnessError::UnresolvedHandle {
                kind: "toolset".to_string(),
                key: "missing-tools".to_string(),
            }
        );
    }

    #[test]
    fn validate_unresolved_schema_handle() {
        let reg = ExecutionRegistry::builder()
            .agent("a1", Arc::new(StubAgent))
            .toolset("t1", Arc::new(EmptyToolRegistry))
            .build();
        let leaf = LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: 4 },
            agent: AgentRef("a1".into()),
            toolset: ToolsetRef("t1".into()),
            output: Some(SchemaRef("missing-schema".into())),
        });
        let task = Task::new("do it", SessionId::generate(), leaf);
        let err = reg.validate(&task).unwrap_err();
        assert_eq!(
            err,
            HarnessError::UnresolvedHandle {
                kind: "schema".to_string(),
                key: "missing-schema".to_string(),
            }
        );
    }

    #[test]
    fn validate_happy_path_react() {
        let reg = fully_wired_registry();
        let leaf = LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: 4 },
            agent: AgentRef("a1".into()),
            toolset: ToolsetRef("t1".into()),
            output: Some(SchemaRef("s1".into())),
        });
        let task = Task::new("ok", SessionId::generate(), leaf);
        assert!(reg.validate(&task).is_ok());
    }

    // ---- A.5 output-contract enforcement (#124, Q3) ------------------------

    #[test]
    fn structured_slot_rejects_bare_react_without_output_schema() {
        // A PlanExecute whose `plan` slot is a bare ReAct with no output schema
        // violates the A.5 contract: the structured plan slot must yield a typed
        // task graph. Validation rejects it BEFORE any handle resolution.
        let reg = ExecutionRegistry::builder()
            .agent("a1", Arc::new(StubAgent))
            .toolset("t1", Arc::new(EmptyToolRegistry))
            .build();
        let tree = LoopStrategy::PlanExecute(PlanExecuteConfig {
            plan: Box::new(react_leaf("a1", "t1")), // output: None
            execute: Box::new(react_leaf("a1", "t1")),
            plan_model: None,
        });
        let task = Task::new("contract", SessionId::generate(), tree);
        match reg.validate(&task) {
            Err(HarnessError::InvalidConfiguration(msg)) => {
                assert!(msg.contains("plan"), "error should name the slot: {msg}");
            }
            other => {
                panic!("expected InvalidConfiguration for bare-ReAct plan slot, got {other:?}")
            }
        }
    }

    #[test]
    fn structured_slot_accepts_react_with_output_schema() {
        let reg = ExecutionRegistry::builder()
            .agent("a1", Arc::new(StubAgent))
            .toolset("t1", Arc::new(EmptyToolRegistry))
            .schema("plan-schema", serde_json::json!({}))
            .build();
        let plan = LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: 4 },
            agent: AgentRef("a1".into()),
            toolset: ToolsetRef("t1".into()),
            output: Some(SchemaRef("plan-schema".into())),
        });
        let tree = LoopStrategy::PlanExecute(PlanExecuteConfig {
            plan: Box::new(plan),
            execute: Box::new(react_leaf("a1", "t1")),
            plan_model: None,
        });
        let task = Task::new("contract", SessionId::generate(), tree);
        assert!(reg.validate(&task).is_ok());
    }

    #[test]
    fn structured_slot_accepts_combinator_child() {
        // A non-leaf child in a structured slot carries its own contract; the
        // bare-ReAct check applies only to a leaf, so a PlanExecute plan slot
        // holding (say) another PlanExecute is accepted.
        let reg = ExecutionRegistry::builder()
            .agent("a1", Arc::new(StubAgent))
            .toolset("t1", Arc::new(EmptyToolRegistry))
            .schema("worker-schema", serde_json::json!({}))
            .schema("eval-schema", serde_json::json!({}))
            .build();
        let worker = LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: 4 },
            agent: AgentRef("a1".into()),
            toolset: ToolsetRef("t1".into()),
            output: Some(SchemaRef("worker-schema".into())),
        });
        let inner_sv = LoopStrategy::SelfVerifying(SelfVerifyingConfig {
            inner: Box::new(worker),
            evaluator: SchemaRef("eval-schema".into()),
        });
        let tree = LoopStrategy::PlanExecute(PlanExecuteConfig {
            plan: Box::new(inner_sv),
            execute: Box::new(react_leaf("a1", "t1")),
            plan_model: None,
        });
        let task = Task::new("contract", SessionId::generate(), tree);
        assert!(reg.validate(&task).is_ok());
    }

    // ---- tree-walk over the nested cordyceps fixture tree ------------------

    fn cordyceps_tree() -> LoopStrategy {
        let json = include_str!("../../../../fixtures/strategy/cordyceps_tree.json");
        serde_json::from_str(json).expect("cordyceps_tree.json parses as LoopStrategy")
    }

    #[test]
    fn validate_tree_walk_reports_first_unresolved_in_nested_tree() {
        // The cordyceps tree references agents planner/executor/ralph-agent,
        // toolsets plan-tools/exec-tools, schema exec-evaluator. An empty
        // registry must report the FIRST unresolved handle (depth-first: the
        // ralph inner -> plan_execute -> plan react -> agent "planner").
        let reg = ExecutionRegistry::empty();
        let task = Task::new("nested", SessionId::generate(), cordyceps_tree());
        let err = reg.validate(&task).unwrap_err();
        assert_eq!(
            err,
            HarnessError::UnresolvedHandle {
                kind: "agent".to_string(),
                key: "planner".to_string(),
            }
        );
    }

    #[test]
    fn validate_tree_walk_passes_when_fully_wired() {
        let reg = ExecutionRegistry::builder()
            .agent("planner", Arc::new(StubAgent))
            .agent("executor", Arc::new(StubAgent))
            .agent("ralph-agent", Arc::new(StubAgent))
            .toolset("plan-tools", Arc::new(EmptyToolRegistry))
            .toolset("exec-tools", Arc::new(EmptyToolRegistry))
            .schema("exec-evaluator", serde_json::json!({}))
            .schema("plan-schema", serde_json::json!({}))
            .schema("worker-schema", serde_json::json!({}))
            .build();
        let task = Task::new("nested", SessionId::generate(), cordyceps_tree());
        assert!(reg.validate(&task).is_ok());
    }

    // ---- resume: round-trip a Task through serde, re-resolve all -----------

    #[test]
    fn resume_reresolves_all_handles_after_serde_roundtrip() {
        // Build a Task, serialize it (trait objects never enter the wire), then
        // deserialize and re-resolve every handle against a freshly-built
        // registry — no reconfiguration of the Task required.
        let leaf = LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: 4 },
            agent: AgentRef("a1".into()),
            toolset: ToolsetRef("t1".into()),
            output: Some(SchemaRef("s1".into())),
        });
        let task = Task::new("resume me", SessionId::generate(), leaf);

        let wire = serde_json::to_string(&task).expect("Task serializes");
        let restored: Task = serde_json::from_str(&wire).expect("Task deserializes");

        // Fresh registry built independently (as on resume) re-resolves all.
        let reg = fully_wired_registry();
        assert!(reg.validate(&restored).is_ok());

        if let LoopStrategy::ReAct(c) = &restored.loop_strategy {
            assert!(reg.resolve_agent(&c.agent).is_some());
            assert!(reg.resolve_toolset(&c.toolset).is_some());
            assert!(reg.resolve_schema(c.output.as_ref().unwrap()).is_some());
        } else {
            panic!("expected ReAct leaf");
        }
    }

    // ---- fixture replay: new HarnessError variants round-trip --------------

    #[test]
    fn registry_errors_fixture_round_trips_byte_identical() {
        let raw = include_str!("../../../../fixtures/harness/registry_errors.json");
        let doc: serde_json::Value = serde_json::from_str(raw).expect("fixture parses");

        // StrategyNotFound
        let snf: HarnessError =
            serde_json::from_value(doc["strategy_not_found"].clone()).expect("StrategyNotFound");
        assert_eq!(
            snf,
            HarnessError::StrategyNotFound {
                key: "my-harness::DoubleVerify".into()
            }
        );
        assert_eq!(
            serde_json::to_value(&snf).unwrap(),
            doc["strategy_not_found"]
        );

        // UnresolvedHandle (Rust field `kind` serializes as `handle_kind`).
        let uh: HarnessError =
            serde_json::from_value(doc["unresolved_handle"].clone()).expect("UnresolvedHandle");
        assert_eq!(
            uh,
            HarnessError::UnresolvedHandle {
                kind: "agent".into(),
                key: "planner".into(),
            }
        );
        assert_eq!(serde_json::to_value(&uh).unwrap(), doc["unresolved_handle"]);
    }

    // ---- builder fluent style + last-wins ----------------------------------

    #[test]
    fn builder_last_wins_on_duplicate_key() {
        let reg = ExecutionRegistry::builder()
            .schema("s", serde_json::json!({"v": 1}))
            .schema("s", serde_json::json!({"v": 2}))
            .build();
        assert_eq!(
            reg.resolve_schema(&SchemaRef("s".into())),
            Some(&serde_json::json!({"v": 2}))
        );
    }
}
