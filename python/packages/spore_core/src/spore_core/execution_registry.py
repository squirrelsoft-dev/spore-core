"""ExecutionRegistry — runtime resolution of serializable strategy handles
(Composable Execution A.3, issue #120; part of the #117–#131 refactor).

The registry maps the serializable string handles carried in a :class:`Task`'s
strategy tree (``AgentRef`` / ``ToolsetRef`` / ``SchemaRef``) and
``StrategyRef.Custom`` keys to concrete runtime collaborators. Trait objects
never enter the serialized ``Task`` — only string handles do — so on resume the
tree is reconstructed and every handle is **re-resolved** against a freshly
built registry, with no reconfiguration.

Types
-----
- :class:`ExecutionRegistry` — six maps keyed by string: ``agents``,
  ``toolsets``, ``schemas``, ``verifiers``, ``metric_evaluators`` (#124, Q2),
  ``custom`` (custom strategies). NOT serialized.
- :class:`ExecutionRegistryBuilder` — fluent assembler mirroring
  ``HarnessBuilder``.
- :class:`StrategyResolution` — the result of resolving a ``StrategyRef``:
  either the built-in :data:`LoopStrategy` tree or a custom ``RunStrategy``.
- :class:`EscalationMode` — the HITL-vs-AFK config knob (PRD goal #7).

Rules enforced (pinned in #120 — do not re-litigate)
---------------------------------------------------
- The registry has exactly FIVE maps (no sixth).
- An unresolved handle (missing agent/toolset/schema) → startup error before the
  first turn (:class:`~spore_core.harness.HarnessErrorUnresolvedHandle`).
- A missing ``StrategyRef.Custom`` key → recoverable
  :class:`~spore_core.harness.HarnessErrorStrategyNotFound`, never a crash.
- ``register_strategy`` makes a custom strategy resolvable by key.
- :class:`EscalationMode` has NO baked-in default (mirrors the budget-types
  discipline); ``HarnessBuilder`` picks an explicit default
  (:class:`EscalationModeSurfaceToHuman`). Stored only this slice (#130
  consumes it) and NOT part of the serialized ``Task``.
- Scope is ADDITIVE (Option B): the registry coexists with the deprecated
  single-collaborator fields on ``HarnessConfig`` and is not yet read by the
  run bodies (that lands in #123/#124).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .agent import Agent
from .harness import (
    AgentRef,
    HarnessError,
    HarnessErrorException,
    HarnessErrorInvalidConfiguration,
    HarnessErrorStrategyNotFound,
    HarnessErrorUnresolvedHandle,
    HillClimbingConfig,
    LoopStrategy,
    PlanExecuteConfig,
    RalphConfig,
    ReactConfig,
    RunStrategy,
    SchemaRef,
    SelfVerifyingConfig,
    StrategyRef,
    StrategyRefBuiltIn,
    Task,
    ToolsetRef,
    _Model,
)
from .verifier import Verifier


# ── EscalationMode ───────────────────────────────────────────────────────────
#
# HITL-vs-AFK escalation knob (PRD goal #7: local vs. prod differ only by
# config). Selects what budget escalation does: surface to a human, fail
# autonomously, or keep working under an auto-granted cap (SC-5). Tagged on
# ``kind`` (snake_case), matching Rust's serde shape:
#   {"kind": "surface_to_human"} / {"kind": "autonomous"} / {"kind": "auto_continue"}
#
# NO default is baked into the type (mirrors the budget-types discipline); the
# ``HarnessBuilder`` picks an explicit default (``SurfaceToHuman``). It is NOT
# placed on the serialized ``Task`` (no fixture) — the mode is never serialized,
# so the ``AutoContinue.on_grant`` runtime callback has no wire impact.


@dataclass
class AutoGrantInfo:
    """Details handed to an :class:`EscalationModeAutoContinue` ``on_grant``
    callback each time the harness auto-grants more budget at an escalation
    point (SC-5).
    """

    #: 1-based index of this grant within the exhausted scope (``1..=max_grants``).
    grant_number: int
    #: Steps granted this round (the mode's ``steps_per_grant``).
    steps_granted: int
    #: The budget scope phase that exhausted (e.g. ``"react"``, ``"plan_execute"``).
    phase: str


#: Callback fired once per :class:`EscalationModeAutoContinue` grant (SC-5).
#: Runtime only — never serialized. Lets a consumer (e.g. looper's governor)
#: observe each auto-grant without owning the keep-working loop itself.
GrantCallback = Callable[[AutoGrantInfo], None]


class EscalationModeSurfaceToHuman(_Model):
    """Budget escalation pauses and surfaces to a human (HITL)."""

    kind: Literal["surface_to_human"] = "surface_to_human"


class EscalationModeAutonomous(_Model):
    """Budget escalation fails the run autonomously (AFK / prod): the partial
    is propagated and the run stops."""

    kind: Literal["autonomous"] = "autonomous"


@dataclass
class EscalationModeAutoContinue:
    """ "Autonomous but capped" (SC-5): at an escalation point, auto-grant
    ``steps_per_grant`` more steps and KEEP WORKING in-process, up to
    ``max_grants`` times, firing ``on_grant`` per grant. Once the grants are
    spent it falls through to the same terminal as
    :class:`EscalationModeAutonomous`. This is the
    keep-working-to-completion-but-cap-at-N policy consumers otherwise hand-roll
    around the harness.

    A PLAIN dataclass (not a pydantic ``_Model``): it carries a runtime-only
    :data:`GrantCallback`, which a strict ``extra="forbid"`` model with no
    arbitrary types cannot hold. The mode is never serialized, so ``on_grant``
    has no wire impact.
    """

    #: Maximum number of auto-grants per exhausted scope before falling through
    #: to the autonomous terminal. ``0`` behaves like
    #: :class:`EscalationModeAutonomous` (no grants).
    max_grants: int
    #: Steps granted each time. A grant raises the exhausted scope's cap to
    #: ``steps_taken + steps_per_grant`` so the loop gets exactly this many more
    #: steps after the exhaustion point.
    steps_per_grant: int
    #: Optional per-grant observer (runtime only; never serialized).
    on_grant: GrantCallback | None = None
    #: Discriminator literal, for symmetry with the pydantic variants.
    kind: Literal["auto_continue"] = "auto_continue"


EscalationMode = (
    EscalationModeSurfaceToHuman | EscalationModeAutonomous | EscalationModeAutoContinue
)


# ── StrategyResolution ───────────────────────────────────────────────────────


@dataclass
class StrategyResolutionBuiltIn:
    """``StrategyRef.BuiltIn(ls)`` resolves to the built-in strategy tree."""

    strategy: LoopStrategy


@dataclass
class StrategyResolutionCustom:
    """``StrategyRef.Custom(key)`` resolves to the registered custom strategy."""

    strategy: RunStrategy


StrategyResolution = StrategyResolutionBuiltIn | StrategyResolutionCustom


# ── ExecutionRegistry ────────────────────────────────────────────────────────


@dataclass
class ExecutionRegistry:
    """Runtime resolver mapping serializable string handles (and
    ``StrategyRef.Custom`` keys) to concrete collaborators.

    Six maps; NOT serialized. Build one with :meth:`builder` or :meth:`empty`.
    The default constructor yields an empty registry.
    """

    agents: dict[str, Agent] = field(default_factory=dict)
    toolsets: dict[str, Any] = field(default_factory=dict)
    schemas: dict[str, Any] = field(default_factory=dict)
    verifiers: dict[str, Verifier] = field(default_factory=dict)
    # Sixth map (#124, Q2): HillClimbing metric evaluators, keyed by the same
    # string ``HillClimbingConfig.evaluator`` carries on the wire. Runtime-only
    # (never serialized) like the other maps; keeping it distinct from ``agents``
    # preserves the metric-evaluator wire string while resolving it to a
    # ``MetricEvaluator`` rather than an ``Agent``.
    metric_evaluators: dict[str, Any] = field(default_factory=dict)
    custom: dict[str, RunStrategy] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> ExecutionRegistry:
        """An empty registry (no entries in any of the six maps)."""
        return cls()

    def is_empty(self) -> bool:
        """True when no entries exist in any of the six maps."""
        return not (
            self.agents
            or self.toolsets
            or self.schemas
            or self.verifiers
            or self.metric_evaluators
            or self.custom
        )

    @classmethod
    def builder(cls) -> ExecutionRegistryBuilder:
        """Start a fluent :class:`ExecutionRegistryBuilder`."""
        return ExecutionRegistryBuilder()

    def into_builder(self) -> ExecutionRegistryBuilder:
        """Return a builder preserving all existing entries, so a caller can add
        more before re-:meth:`~ExecutionRegistryBuilder.build`ing. Shares the
        underlying maps (no deep copy)."""
        return ExecutionRegistryBuilder(registry=self)

    # ── resolution (pure lookups, each Ref type → one map) ───────────────────

    def resolve_agent(self, ref: AgentRef) -> Agent | None:
        """Resolve an ``AgentRef`` to its registered agent, or ``None``."""
        return self.agents.get(ref)

    def resolve_toolset(self, ref: ToolsetRef) -> Any | None:
        """Resolve a ``ToolsetRef`` to its registered toolset, or ``None``."""
        return self.toolsets.get(ref)

    def resolve_schema(self, ref: SchemaRef) -> Any | None:
        """Resolve a ``SchemaRef`` to its registered JSON schema, or ``None``.
        (``SchemaRef`` maps to the ``schemas`` map.)"""
        return self.schemas.get(ref)

    def resolve_verifier(self, key: str) -> Verifier | None:
        """Resolve a verifier key to its registered verifier, or ``None``."""
        return self.verifiers.get(key)

    def resolve_metric_evaluator(self, key: str) -> Any | None:
        """Resolve a metric-evaluator key (the string ``HillClimbingConfig.evaluator``
        carries, #124 Q2) to its registered ``MetricEvaluator``, or ``None`` if
        absent. The wire string is identical to the legacy ``AgentRef``; only the
        resolution target differs (the sixth ``metric_evaluators`` map)."""
        return self.metric_evaluators.get(key)

    def resolve_strategy(self, ref: StrategyRef) -> StrategyResolution:
        """Resolve a ``StrategyRef``: a ``BuiltIn(ls)`` returns the built-in
        tree; a ``Custom(key)`` looks up :attr:`custom` and raises the
        recoverable :class:`~spore_core.harness.HarnessErrorException`
        (``StrategyNotFound``) when the key is absent."""
        if isinstance(ref, StrategyRefBuiltIn):
            return StrategyResolutionBuiltIn(strategy=ref.value)
        # StrategyRefCustom
        strategy = self.custom.get(ref.value)
        if strategy is None:
            raise HarnessErrorException(HarnessErrorStrategyNotFound(key=ref.value))
        return StrategyResolutionCustom(strategy=strategy)

    def register_strategy(self, key: str, strategy: RunStrategy) -> None:
        """Register (or replace, last-wins) a custom strategy under ``key``."""
        self.custom[key] = strategy

    # ── validation (startup tree-walk) ───────────────────────────────────────

    def validate(self, task: Task) -> None:
        """Validate that every handle referenced by ``task.loop_strategy``
        resolves against this registry. Walks the strategy tree and raises the
        FIRST unresolved handle as a
        :class:`~spore_core.harness.HarnessErrorException` wrapping an
        ``UnresolvedHandle`` (or ``StrategyNotFound`` for a missing custom key).
        Returns ``None`` when the whole tree resolves. Called at the entry of
        :meth:`StandardHarness.run` so an unresolved handle is a startup error,
        before the first turn."""
        self._walk_strategy(task.loop_strategy)

    def _walk_strategy(self, ls: LoopStrategy) -> None:
        if isinstance(ls, ReactConfig):
            self._check_agent(ls.agent)
            self._check_toolset(ls.toolset)
            if ls.output is not None:
                self._check_schema(ls.output)
        elif isinstance(ls, PlanExecuteConfig):
            # A.5 (#124, Q3): the ``plan`` slot is STRUCTURED — it must yield a
            # task graph. A bare ``ReAct`` there needs an output schema.
            self._check_structured_slot(ls.plan, "plan")
            self._walk_strategy(ls.plan)
            self._walk_strategy(ls.execute)
        elif isinstance(ls, SelfVerifyingConfig):
            # A.5: the ``inner`` (worker) slot is STRUCTURED — its result must be
            # evaluable. A bare ``ReAct`` worker needs an output schema.
            self._check_structured_slot(ls.inner, "worker")
            self._walk_strategy(ls.inner)
            # #124 Q1: the evaluator's wire string (a ``SchemaRef``) is the
            # VERIFIER registry key — resolved against the ``verifiers`` map.
            self._check_verifier(ls.evaluator)
            # The optional eval-phase overrides resolve against the ``agents`` /
            # ``toolsets`` maps when set; ``None`` ⇒ the worker-agent / global-
            # catalogue fallback (no validation needed, mirrors Ralph's agent).
            if ls.eval_agent is not None:
                self._check_agent(ls.eval_agent)
            if ls.eval_toolset is not None:
                self._check_toolset(ls.eval_toolset)
        elif isinstance(ls, RalphConfig):
            self._walk_strategy(ls.inner)
            self._check_agent(ls.agent)
        elif isinstance(ls, HillClimbingConfig):
            # A.5: the ``inner`` (propose) slot is STRUCTURED — it must yield a
            # candidate. A bare ``ReAct`` proposer needs an output schema.
            self._check_structured_slot(ls.inner, "propose")
            self._walk_strategy(ls.inner)
            # #124 Q2: the evaluator's wire string is resolved against the sixth
            # ``metric_evaluators`` map (not ``agents``).
            self._check_metric_evaluator(ls.evaluator)
        else:  # pragma: no cover — closed union; exhaustive above
            raise AssertionError(f"unknown loop strategy: {ls!r}")

    @staticmethod
    def _check_structured_slot(slot: LoopStrategy, slot_name: str) -> None:
        """A.5 output-contract enforcement (#124, Q3): a bare ``ReAct`` feeding a
        STRUCTURED slot (``plan`` ⇒ task graph, ``propose`` ⇒ candidate,
        ``worker`` ⇒ evaluable result) MUST declare ``ReAct.output`` set. A
        combinator child carries its own contract, so this check applies only to a
        leaf. Raises :class:`~spore_core.harness.HarnessErrorException` wrapping an
        :class:`~spore_core.harness.HarnessErrorInvalidConfiguration` naming the
        offending slot."""
        if isinstance(slot, ReactConfig) and slot.output is None:
            raise HarnessErrorException(
                HarnessErrorInvalidConfiguration(
                    reason=(
                        f"a bare ReAct in the structured `{slot_name}` slot requires "
                        "`output = Some(schema)` so the slot yields a typed result"
                    )
                )
            )

    def _check_agent(self, ref: AgentRef) -> None:
        if ref not in self.agents:
            raise HarnessErrorException(HarnessErrorUnresolvedHandle(handle_kind="agent", key=ref))

    def _check_toolset(self, ref: ToolsetRef) -> None:
        if ref not in self.toolsets:
            raise HarnessErrorException(
                HarnessErrorUnresolvedHandle(handle_kind="toolset", key=ref)
            )

    def _check_schema(self, ref: SchemaRef) -> None:
        if ref not in self.schemas:
            raise HarnessErrorException(HarnessErrorUnresolvedHandle(handle_kind="schema", key=ref))

    def _check_verifier(self, ref: SchemaRef) -> None:
        """#124 Q1: a SelfVerifying ``evaluator`` (a ``SchemaRef`` on the wire)
        resolves against the ``verifiers`` map."""
        if ref not in self.verifiers:
            raise HarnessErrorException(
                HarnessErrorUnresolvedHandle(handle_kind="verifier", key=ref)
            )

    def _check_metric_evaluator(self, ref: AgentRef) -> None:
        """#124 Q2: a HillClimbing ``evaluator`` (an ``AgentRef`` on the wire)
        resolves against the sixth ``metric_evaluators`` map."""
        if ref not in self.metric_evaluators:
            raise HarnessErrorException(
                HarnessErrorUnresolvedHandle(handle_kind="metric_evaluator", key=ref)
            )


# ── ExecutionRegistryBuilder ─────────────────────────────────────────────────


@dataclass
class ExecutionRegistryBuilder:
    """Fluent assembler for an :class:`ExecutionRegistry`, mirroring
    ``HarnessBuilder``. Each setter returns ``self`` for chaining."""

    registry: ExecutionRegistry = field(default_factory=ExecutionRegistry)

    def agent(self, key: str, agent: Agent) -> ExecutionRegistryBuilder:
        """Register an agent under ``key``."""
        self.registry.agents[key] = agent
        return self

    def toolset(self, key: str, toolset: Any) -> ExecutionRegistryBuilder:
        """Register a toolset under ``key``."""
        self.registry.toolsets[key] = toolset
        return self

    def schema(self, key: str, schema: Any) -> ExecutionRegistryBuilder:
        """Register a JSON schema under ``key``."""
        self.registry.schemas[key] = schema
        return self

    def verifier(self, key: str, verifier: Verifier) -> ExecutionRegistryBuilder:
        """Register a verifier under ``key``."""
        self.registry.verifiers[key] = verifier
        return self

    def metric_evaluator(self, key: str, evaluator: Any) -> ExecutionRegistryBuilder:
        """Register a metric evaluator under ``key`` (#124, Q2 — the sixth map)."""
        self.registry.metric_evaluators[key] = evaluator
        return self

    def register_strategy(self, key: str, strategy: RunStrategy) -> ExecutionRegistryBuilder:
        """Register a custom strategy under ``key``."""
        self.registry.custom[key] = strategy
        return self

    def fill_default_agent(self, agent: Agent) -> ExecutionRegistryBuilder:
        """#124 migration seam: register ``agent`` under the DEFAULT empty-string
        key ONLY if that key is not already wired. ``HarnessConfig`` folds its
        single agent here so bare ``ReactConfig.per_loop`` leaves (empty
        ``AgentRef``) resolve to it. An explicitly-registered ``""`` agent wins."""
        self.registry.agents.setdefault("", agent)
        return self

    def fill_default_toolset(self, toolset: Any) -> ExecutionRegistryBuilder:
        """#124: as :meth:`fill_default_agent`, for the default toolset (the
        config's ``tool_registry``) under the empty key."""
        self.registry.toolsets.setdefault("", toolset)
        return self

    def fill_toolset(self, key: str, toolset: Any) -> ExecutionRegistryBuilder:
        """Issue 2 (per-node toolset scoping): register ``toolset`` under ``key``
        ONLY if that key is not already wired. ``HarnessConfig`` calls this for
        every per-key catalogue accumulated via
        :meth:`HarnessBuilder.toolset_tools` so a leaf referencing that handle
        passes ``ExecutionRegistry.validate`` WITHOUT the caller wiring a
        placeholder. The registry VALUE is never dispatched (dispatch goes through
        ``HarnessConfig.toolset_catalogues``), so a presence entry suffices; an
        explicitly-registered toolset under the same key wins. Mirrors
        :meth:`fill_default_toolset` (and Rust's
        ``ExecutionRegistryBuilder::fill_toolset``)."""
        self.registry.toolsets.setdefault(key, toolset)
        return self

    def fill_default_schema(self, schema: Any) -> ExecutionRegistryBuilder:
        """#124: as :meth:`fill_default_agent`, for a default output schema under
        the empty key — so a bare structured-slot leaf (``output=""``) resolves
        under the single resolution path without each caller wiring a schema."""
        self.registry.schemas.setdefault("", schema)
        return self

    def fill_default_verifier(self, verifier: Verifier) -> ExecutionRegistryBuilder:
        """#124: as :meth:`fill_default_agent`, for a default SelfVerifying
        verifier under the empty key."""
        self.registry.verifiers.setdefault("", verifier)
        return self

    def fill_default_metric_evaluator(self, evaluator: Any) -> ExecutionRegistryBuilder:
        """#124: as :meth:`fill_default_agent`, for a default HillClimbing metric
        evaluator under the empty key."""
        self.registry.metric_evaluators.setdefault("", evaluator)
        return self

    def build(self) -> ExecutionRegistry:
        """Finish and return the assembled :class:`ExecutionRegistry`."""
        return self.registry


__all__ = [
    "AutoGrantInfo",
    "EscalationMode",
    "EscalationModeAutoContinue",
    "EscalationModeAutonomous",
    "EscalationModeSurfaceToHuman",
    "GrantCallback",
    "ExecutionRegistry",
    "ExecutionRegistryBuilder",
    "StrategyResolution",
    "StrategyResolutionBuiltIn",
    "StrategyResolutionCustom",
    "HarnessError",
    "HarnessErrorException",
    "HarnessErrorInvalidConfiguration",
    "HarnessErrorStrategyNotFound",
    "HarnessErrorUnresolvedHandle",
]
