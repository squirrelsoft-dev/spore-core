"""Harness — the agent runtime loop (issue #3).

Mirrors the Rust reference at ``rust/crates/spore-core/src/harness.rs``.
The harness owns execution lifecycle and wires all components together.
It is stateless between :meth:`Harness.run` calls; everything the harness
needs comes in via :class:`HarnessRunOptions` or :class:`PausedState`, and
everything it produces goes out via :class:`RunResult`.

``dangerous`` gate (issue #34): ``IsolationModeNone`` (no path enforcement) is
a named safety footgun. It is not part of the default public surface — it is
not exported from this module or ``spore_core``, and is reachable only via
``from spore_core.dangerous import IsolationModeNone``. Consequently the
default :meth:`SandboxProvider.isolation_mode` body returns
``IsolationModeWorkspaceScoped`` (safe-by-default), never ``None``. The wire
tag for the gated mode stays ``"none"``.

What this component does:

* Assemble context (via :class:`ContextManager`) before each turn
* Call the agent for one turn
* Dispatch tool calls to :class:`ToolRegistry`
* Evaluate :class:`TerminationPolicy` after each turn
* Fire middleware lifecycle hooks
* Track iterations, token spend, elapsed time
* Pause and resume for human-in-the-loop interactions

What this component does NOT do:

* Touch the filesystem, execute commands, or call the model directly
* Persist :class:`PausedState` — the caller owns persistence
* Implement individual tools, sandbox policy, or context assembly

Rules enforced here:

1. Harness owns the loop — the agent only executes one turn at a time.
2. Termination is evaluated against external state via
   :class:`TerminationPolicy`.
3. Any budget overrun terminates the loop with an explicit
   :class:`HaltReason`.
4. A turn that yields neither a tool call nor a final response is an error
   (surfaced via :class:`AgentError`).
5. All components are injected at construction; the harness never builds
   them itself.
6. Stateless between pause and resume — the caller owns
   :class:`PausedState`.
7. :class:`WaitingForHuman` returns immediately; no internal timeout.
8. ``approved_results`` prevents double-execution on resume.
9. Subagents cannot spawn their own subagents — :class:`ChildPausedState`
   has no ``child_state`` field (depth-1 enforcement).

Many of the sibling component traits (``ToolRegistry``, ``SandboxProvider``,
``ContextManager``, …) ship in their own component issues (#4–#13). Until
those land, this module defines minimal forward-declared :class:`Protocol`
stubs of the trait surface the loop actually consumes. When a sibling
issue lands its canonical definition will replace the stub here.
"""

from __future__ import annotations

import asyncio
import json
import time
import warnings
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    NewType,
    Protocol,
    runtime_checkable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
)

from .agent import (
    Agent,
    AgentError,
    AgentStreamSink,
    Context,
    FinalResponse,
    ToolCallRequested,
    TurnError,
    TurnResult,
    turn_streaming as _agent_turn_streaming,
)
from .errors import SporeError
from .model import (
    ContentBlockDelta,
    ImageContent,
    Message,
    MessageStart,
    MessageStop,
    ModelInterface,
    ModelParams,
    Role,
    StopReason,
    StreamEvent as ModelStreamEvent,
    TextContent,
    ThinkingDelta,
    TokenUsage,
    ToolCall,
    ToolCallContent as MsgToolCallContent,
    ToolResultContent as MsgToolResultContent,
    ToolSchema,
    ToolUseDelta,
    ToolUseStart,
    _canonicalize_json,
)
from .model import (
    ContentBlockStop as ModelContentBlockStop,
)
from .output_schema import (
    feedback_message as output_schema_feedback_message,
    validate_output,
)
from .prompt_tool_call import (
    AdaptiveToolCallModelInterface,
    PromptToolCallFlag,
    detect_prose_response,
)

if TYPE_CHECKING:
    from .context import (
        CompactionPreserveHints,
        CompactionVerifier,
        ContextErrorModel,
        ContextSources,
        Guide,
    )
    from .context import (
        SessionState as ContextSessionState,
    )
    from .skills import SkillCatalog
    from .memory import MemoryProvider
    from .execution_registry import EscalationMode, ExecutionRegistry
    from .hooks import HookChain
    from .middleware import MiddlewareChain
    from .tasklist import TaskList
    from .tool_registry import StandardToolRegistry

# ============================================================================
# Identity newtypes
# ============================================================================

SessionId = NewType("SessionId", str)
TaskId = NewType("TaskId", str)

_id_counter: int = 0


def _random_id() -> str:
    global _id_counter
    _id_counter += 1
    return f"{_id_counter:016x}"


def new_session_id(s: str | None = None) -> SessionId:
    return SessionId(s if s is not None else f"sess-{_random_id()}")


def new_task_id(s: str | None = None) -> TaskId:
    return TaskId(s if s is not None else f"task-{_random_id()}")


# ============================================================================
# Pydantic base
# ============================================================================


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# Budget tracking
# ============================================================================


class BudgetLimits(_Model):
    max_turns: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    # Wire format: seconds (int) to match Rust's `duration_secs_opt`.
    max_wall_time: int | None = None
    max_cost_usd: float | None = None


BudgetLimitTypeT = Literal["turns", "input_tokens", "output_tokens", "wall_time", "cost_usd"]


class BudgetSnapshot(_Model):
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time: int | None = None
    cost_usd: float = 0.0


class AggregateUsage(_Model):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def add_turn(self, u: TokenUsage) -> None:
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cache_read_tokens += u.cache_read_tokens or 0
        self.cache_write_tokens += u.cache_write_tokens or 0


# ============================================================================
# Composable-execution budget vocabulary (issue #117)
# ============================================================================
#
# Pure, serializable value types from the Composable Execution PRD (Part B) —
# no executor wiring. Later slices thread them through the strategy tree. They
# layer *on top of* ``BudgetLimits`` (the global turns/tokens/wall/cost
# backstop), which is unchanged.
#
# ``BudgetPolicy`` is a per-scope step allowance, where a step is one model turn
# (matches ``BudgetSnapshot.turns``). ``PerGoal`` is intentionally excluded in
# v1.
#
# ``BudgetExhaustedBehavior`` says what to do when a policy's allowance is spent.
# No node silently defaults to ``Continue`` — ``Continue`` carries a required
# ``max_continues`` and a recursively nested ``on_exhausted`` fall-through.
#
# Both are internally tagged on ``kind`` (snake_case), matching the established
# tagged-union pattern in this module and byte-identical with the Rust / TS / Go
# definitions (see ``fixtures/budget_policy/cases.json``).


class BudgetPolicyUnlimited(_Model):
    kind: Literal["unlimited"] = "unlimited"


class BudgetPolicyTotalSteps(_Model):
    kind: Literal["total_steps"] = "total_steps"
    value: int


class BudgetPolicyPerLoop(_Model):
    kind: Literal["per_loop"] = "per_loop"
    value: int


class BudgetPolicyPerAttempt(_Model):
    kind: Literal["per_attempt"] = "per_attempt"
    value: int


BudgetPolicy = Annotated[
    BudgetPolicyUnlimited | BudgetPolicyTotalSteps | BudgetPolicyPerLoop | BudgetPolicyPerAttempt,
    Field(discriminator="kind"),
]


def budget_allowance_value(policy: BudgetPolicy) -> int | None:
    """The per-scope step allowance carried by ``policy`` (``None`` for
    ``Unlimited``). Shared by :meth:`BudgetContext._allowance` and the leaf
    cap-binding check in :func:`_run_react_config` (#125). Mirrors Rust's
    ``BudgetPolicy::allowance_value``."""
    if isinstance(policy, BudgetPolicyUnlimited):
        return None
    return policy.value


class BudgetExhaustedContinue(_Model):
    kind: Literal["continue"] = "continue"
    max_continues: int
    # Recursively nested fall-through behavior once ``max_continues`` extra
    # rounds are spent. ``max_continues == 0`` means immediate fall-through.
    on_exhausted: BudgetExhaustedBehavior


class BudgetExhaustedEscalate(_Model):
    kind: Literal["escalate"] = "escalate"


class BudgetExhaustedFail(_Model):
    kind: Literal["fail"] = "fail"


BudgetExhaustedBehavior = Annotated[
    BudgetExhaustedContinue | BudgetExhaustedEscalate | BudgetExhaustedFail,
    Field(discriminator="kind"),
]

# Resolve the forward reference to ``BudgetExhaustedBehavior`` used by
# ``BudgetExhaustedContinue.on_exhausted`` (recursive tagged union).
BudgetExhaustedContinue.model_rebuild()


def default_budget_behavior() -> BudgetExhaustedEscalate:
    """The default :class:`BudgetExhaustedBehavior` for a config node's serialized
    ``behavior`` field (#129): ``Escalate``. Two roles:

    - as the pydantic ``default_factory`` so a strategy tree serialized BEFORE #129
      (no ``behavior`` key) still deserializes to the historical placeholder,
      preserving backward-compat reads;
    - the value the migration shims (``ReactConfig.per_loop`` / ``*.simple``)
      stamp so a bare leaf keeps its pre-#129 propagate-to-parent contract by
      default.

    The field is NOT skip-serialized: it ALWAYS serializes (uniform wire shape
    across all five config structs, Q1), so the cross-language fixtures carry an
    explicit ``"behavior":{"kind":"escalate"}`` on every node. Mirrors Rust's
    ``default_budget_behavior``."""
    return BudgetExhaustedEscalate()


# ============================================================================
# Task + loop strategy (tagged union on ``kind``)
# ============================================================================


OptimizationDirection = Literal["minimize", "maximize"]
# Wire alias kept for the old name; the strategy direction literal is unchanged
# (snake_case ``"minimize"`` / ``"maximize"``). Renamed in #119 to live with the
# other strategy-config types (Rust ``HillClimbingDirection``).
HillClimbingDirection = OptimizationDirection


class ModelConfig(_Model):
    provider: str
    model_id: str


# ============================================================================
# Composable Execution Part A (issue #119): recursive LoopStrategy config
# newtypes + per-node collaborator handles + StrategyRef + RunStrategy.
# ============================================================================
#
# ``LoopStrategy`` is a closed, recursive discriminated union of config
# newtypes. ``ReactConfig`` is the leaf; the rest are combinators holding
# ``LoopStrategy`` children. Wire format is byte-identical with Rust / TS / Go
# (see ``fixtures/strategy/``).
#
# Handle types (``AgentRef``, ``ToolsetRef``, ``SchemaRef``) are bare strings on
# the wire (transparent newtypes), modeled here as ``str`` type aliases.
#
# The ``RunStrategy`` composition seam is the Python runtime-polymorphism idiom:
# a ``Protocol`` with ``async def run(...)`` and a single ``match`` site on
# ``LoopStrategy`` that delegates to each config's ``run``. Per-config bodies are
# STUBS returning a pending ``StrategyOutcome`` (never raise); real bodies land
# in #124. ``ExecutionContext`` / ``StrategyOutcome`` are minimal placeholders
# (full shape owned by #123).


# Per-node collaborator handles. Bare strings on the wire (transparent); the
# registry slice (#120) resolves them to concrete agents/toolsets/schemas.
AgentRef = str
ToolsetRef = str
SchemaRef = str


class ReactConfig(_Model):
    """Leaf ReAct node config. ``budget`` is the renamed ``max_iterations``
    (semantically ``PerLoop(n)``). Serializes flat next to ``"kind":"react"``."""

    kind: Literal["react"] = "react"
    budget: BudgetPolicy
    # #129: what this node does when its ``budget`` is spent. CANONICAL POSITION
    # on ``ReactConfig``: IMMEDIATELY after ``budget``. A leaf honors its
    # ``behavior`` ONLY at the top-level/bare-leaf resolution site
    # (:meth:`_drive_strategy`); in the normal NESTED case the leaf still
    # PROPAGATES exhaustion to its parent (#125 rule 6 — a nested leaf never
    # self-resolves). Always serialized (Q1).
    behavior: BudgetExhaustedBehavior = Field(default_factory=default_budget_behavior)
    agent: AgentRef
    toolset: ToolsetRef
    # OMITTED from JSON when None (matches Rust ``skip_serializing_if``).
    output: SchemaRef | None = None
    # SC-10 (#161): per-leaf operating system-prompt OVERRIDE. ``None`` (the
    # default) preserves today's behaviour — the leaf's turn window uses the
    # global :attr:`HarnessConfig.system_prompt`. When set, THIS prompt REPLACES
    # the global one for every turn of this leaf's window (the leaf sees ONLY its
    # own prompt; nothing leaks from the configured global prompt). This is the
    # per-leaf prompt half of SC-10; the per-leaf TOOLSET half is the existing
    # :attr:`toolset` handle. In a :class:`PlanExecuteConfig` this lets the plan
    # and execute phases run under DISTINCT system prompts. CANONICAL POSITION:
    # LAST field (after ``output``, alongside ``toolset``). OMITTED from the JSON
    # wire when None (matches Rust ``skip_serializing_if``), so existing
    # strategy-config fixtures serialize byte-identically.
    system_prompt: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if self.output is None:
            data.pop("output", None)
        if self.system_prompt is None:
            data.pop("system_prompt", None)
        return data

    @classmethod
    def per_loop(cls, value: int) -> ReactConfig:
        """A bare ReAct leaf with a ``PerLoop { value }`` budget and empty agent
        / toolset handles (resolution lands with #120). Migration shim for the
        old ``ReAct { max_iterations }`` shape."""
        return cls(budget=BudgetPolicyPerLoop(value=value), agent="", toolset="")

    def max_iterations(self) -> int:
        """The ``max_iterations`` value extracted from a ``PerLoop`` budget; any
        other budget shape yields ``2**31 - 1`` (matching the legacy executor's
        unbounded fall-through)."""
        if isinstance(self.budget, BudgetPolicyPerLoop):
            return self.budget.value
        return 2**31 - 1


class PlanExecuteConfig(_Model):
    """PlanExecute combinator: a ``plan`` sub-strategy feeds an ``execute``
    sub-strategy. ``plan_model`` stays optional/omittable."""

    kind: Literal["plan_execute"] = "plan_execute"
    plan: LoopStrategy
    execute: LoopStrategy
    plan_model: ModelConfig | None = None
    # #129: what this combinator does when its execute-phase budget is spent.
    # CANONICAL POSITION on a combinator: the LAST field. Always serialized (Q1).
    behavior: BudgetExhaustedBehavior = Field(default_factory=default_budget_behavior)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if self.plan_model is None:
            data.pop("plan_model", None)
        return data

    @classmethod
    def simple(cls, plan_model: ModelConfig | None = None) -> PlanExecuteConfig:
        """A PlanExecute whose plan and execute phases are both bare ReAct leaves
        (migration shim for the old ``PlanExecute { plan_model }`` shape). #124:
        the ``plan`` slot is STRUCTURED (A.5), so its leaf declares the default
        ``output`` schema handle (empty key) to satisfy the output contract."""
        return cls(
            plan=ReactConfig(
                budget=BudgetPolicyPerLoop(value=2**31 - 1),
                agent="",
                toolset="",
                output="",
            ),
            execute=ReactConfig.per_loop(2**31 - 1),
            plan_model=plan_model,
        )


class SelfVerifyingConfig(_Model):
    """SelfVerifying combinator: run ``inner``, then judge it against
    ``evaluator``."""

    kind: Literal["self_verifying"] = "self_verifying"
    inner: LoopStrategy
    evaluator: SchemaRef
    # Optional dedicated reviewer agent for the evaluate phase. None ⇒ the
    # evaluate phase reuses the inner worker's resolved agent (Q1c default). Set
    # this to a SEPARATE agent so the reviewer is not the same model that wrote
    # the code (a builder reviewing its own work tends to rubber-stamp). OMITTED
    # from the JSON wire when None (existing configs stay byte-identical).
    eval_agent: AgentRef | None = None
    # Optional read-only/inspection toolset for the evaluate phase. None ⇒ the
    # empty handle (global-catalogue fallback), byte-identical to prior behavior.
    # The evaluate phase already runs on a read-only sandbox; scoping the toolset
    # to read-only tools additionally prevents the reviewer from reaching
    # non-filesystem side-effecting tools (web/MCP) the sandbox does not gate.
    # OMITTED from the JSON wire when None.
    eval_toolset: ToolsetRef | None = None
    # #129: what this combinator does when its build↔evaluate budget is spent.
    # CANONICAL POSITION on a combinator: the LAST field. Always serialized (Q1).
    behavior: BudgetExhaustedBehavior = Field(default_factory=default_budget_behavior)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if self.eval_agent is None:
            data.pop("eval_agent", None)
        if self.eval_toolset is None:
            data.pop("eval_toolset", None)
        return data

    @classmethod
    def simple(cls) -> SelfVerifyingConfig:
        """Migration shim for the old empty ``SelfVerifying`` shape: a bare ReAct
        inner (worker) leaf and an empty evaluator handle. #124: the ``inner``
        slot is STRUCTURED (A.5), so its leaf declares the default ``output``
        schema handle (empty key) to satisfy the output contract."""
        return cls(
            inner=ReactConfig(
                budget=BudgetPolicyPerLoop(value=2**31 - 1),
                agent="",
                toolset="",
                output="",
            ),
            evaluator="",
        )


class RalphConfig(_Model):
    """Ralph combinator: re-run ``inner`` under a fixed ``agent`` across
    context-window resets."""

    kind: Literal["ralph"] = "ralph"
    inner: LoopStrategy
    agent: AgentRef
    # #129: what Ralph does when its own scope is spent. CANONICAL POSITION on a
    # combinator: the LAST field. Always serialized (Q1). NOTE: Ralph's window
    # recovery (reset + retry) is independent of this field — it governs Ralph's
    # OWN budget scope, not the per-window child exhaustion (which Ralph already
    # absorbs as "window incomplete" and retries).
    behavior: BudgetExhaustedBehavior = Field(default_factory=default_budget_behavior)

    @classmethod
    def simple(cls) -> RalphConfig:
        """Migration shim for the old empty ``Ralph`` shape: a bare ReAct inner
        leaf and an empty agent handle (resolution lands with #120)."""
        return cls(inner=ReactConfig.per_loop(2**31 - 1), agent="")


class HillClimbingConfig(_Model):
    """HillClimbing combinator: iterate ``inner``, keeping/reverting per the
    metric ``evaluator`` and ``direction``. ``max_stagnation`` and
    ``min_improvement_delta`` are now required (#119)."""

    kind: Literal["hill_climbing"] = "hill_climbing"
    inner: LoopStrategy
    direction: HillClimbingDirection
    max_stagnation: int
    revert_on_no_improvement: bool
    min_improvement_delta: float
    evaluator: AgentRef
    # #129: what this combinator does when its optimization-loop budget is spent.
    # CANONICAL POSITION on a combinator: the LAST field. Always serialized (Q1).
    behavior: BudgetExhaustedBehavior = Field(default_factory=default_budget_behavior)


LoopStrategy = Annotated[
    ReactConfig | PlanExecuteConfig | SelfVerifyingConfig | RalphConfig | HillClimbingConfig,
    Field(discriminator="kind"),
]

# Resolve the forward references to ``LoopStrategy`` used by the recursive
# combinator children (``plan``/``execute``/``inner``).
PlanExecuteConfig.model_rebuild()
SelfVerifyingConfig.model_rebuild()
RalphConfig.model_rebuild()
HillClimbingConfig.model_rebuild()


# ── StrategyRef: adjacently-tagged BuiltIn(LoopStrategy) | Custom(str) ───────
#
# Adjacently tagged on ``kind``/``value`` to avoid a tag collision with the
# nested ``LoopStrategy``'s own ``kind``:
#   - {"kind":"built_in","value":{"kind":"react",...}}
#   - {"kind":"custom","value":"my-harness::DoubleVerify"}


class StrategyRefBuiltIn(_Model):
    kind: Literal["built_in"] = "built_in"
    value: LoopStrategy


class StrategyRefCustom(_Model):
    kind: Literal["custom"] = "custom"
    value: str


StrategyRef = Annotated[
    StrategyRefBuiltIn | StrategyRefCustom,
    Field(discriminator="kind"),
]


# ── max_steps: advisory worst-case turn bound (#122) ─────────────────────────
#
# Python ``LoopStrategy`` / ``StrategyRef`` are discriminated-union aliases, not
# classes, so Rust's ``impl ... { fn max_steps }`` maps to module-level
# dispatchers — the same idiom as :func:`run_strategy` / :func:`budget_allowance_value`.

# The unbounded-windows sentinel for ``HillClimbingConfig.max_stagnation``: a
# value of ``2**31 - 1`` (Python's i32-compatible stand-in for Rust's
# ``u32::MAX``) means "no stagnation cap" and collapses the bound to ``None``.
# This is a SEMANTIC unbounded rule, distinct from arithmetic overflow below.
_MAX_STAGNATION_UNBOUNDED = 2**31 - 1

# The u32 representable ceiling (``4294967295``). Python ints are unbounded, so
# to stay cross-language-identical with Rust's ``checked_add`` / ``checked_mul``
# (which yield ``None`` on overflow) we guard explicitly: any add/multiply whose
# result exceeds this ceiling yields ``None`` ("no finite advisory bound").
_U32_MAX = 2**32 - 1


def _checked_add(a: int, b: int) -> int | None:
    """``a + b``, or ``None`` if the sum exceeds the u32 ceiling. Mirrors Rust's
    ``u32::checked_add``."""
    total = a + b
    return total if total <= _U32_MAX else None


def _checked_mul(a: int, b: int) -> int | None:
    """``a * b``, or ``None`` if the product exceeds the u32 ceiling. Mirrors
    Rust's ``u32::checked_mul``."""
    product = a * b
    return product if product <= _U32_MAX else None


def loop_strategy_max_steps(strategy: LoopStrategy) -> int | None:
    """Advisory worst-case **turn** count for a fully-bounded strategy tree,
    computed before a run (#122).

    This is a pre-run advisory figure, **not** an enforcement mechanism — the
    per-node budget ceiling is the safety mechanism. The bound is derived
    additively/multiplicatively down the tree and is Option-monadic: any
    ``Unlimited`` node anywhere collapses the whole figure to ``None`` ("no
    finite advisory bound"). It is a runtime-only computation, **never
    serialized**.

    Per-variant rules:

    - ``ReactConfig`` ⇒ :func:`budget_allowance_value` of its ``budget``
      (``Unlimited`` ⇒ ``None``; ``PerLoop`` / ``TotalSteps`` / ``PerAttempt``
      ⇒ their ``value``).
    - ``SelfVerifyingConfig`` ⇒ ``inner + 1`` — the single read-only evaluator
      turn is exactly one extra turn.
    - ``PlanExecuteConfig`` ⇒ ``plan + execute``. **PER-TASK bound**: a full
      run's total is ``plan + execute_per_task × task_count``, where
      ``task_count`` is data-dependent (the plan phase builds the task graph at
      runtime), so it is intentionally NOT part of this static figure.
    - ``RalphConfig`` ⇒ ``inner`` — **PER-WINDOW bound**, mirroring
      PlanExecute's per-task treatment. A full run's total is
      ``per_window × max_windows``, where ``max_windows`` derives from
      ``HarnessConfig.max_resets`` (default 3) at runtime and is intentionally
      NOT part of this static figure.
    - ``HillClimbingConfig`` ⇒ ``inner × (max_stagnation + 1)`` — the ``+1`` is
      the one productive pass; ``max_stagnation`` non-improving passes follow.
      The ``max_stagnation == 2**31 - 1`` sentinel means unbounded windows ⇒
      ``None`` (a semantic unbounded rule, distinct from arithmetic overflow).

    Arithmetic is u32-checked; an unrepresentable bound (overflow past
    ``4294967295``) yields ``None``. Mirrors Rust's ``LoopStrategy::max_steps``.
    """
    match strategy:
        case ReactConfig():
            return budget_allowance_value(strategy.budget)
        case SelfVerifyingConfig():
            inner = loop_strategy_max_steps(strategy.inner)
            return None if inner is None else _checked_add(inner, 1)
        case PlanExecuteConfig():
            plan = loop_strategy_max_steps(strategy.plan)
            execute = loop_strategy_max_steps(strategy.execute)
            if plan is None or execute is None:
                return None
            return _checked_add(plan, execute)
        case RalphConfig():
            return loop_strategy_max_steps(strategy.inner)
        case HillClimbingConfig():
            if strategy.max_stagnation == _MAX_STAGNATION_UNBOUNDED:
                return None
            inner = loop_strategy_max_steps(strategy.inner)
            if inner is None:
                return None
            passes = _checked_add(strategy.max_stagnation, 1)
            if passes is None:
                return None
            return _checked_mul(inner, passes)


def strategy_ref_max_steps(ref: StrategyRef) -> int | None:
    """Advisory worst-case turn bound for a :data:`StrategyRef` (#122).

    ``Custom`` is opaque to the framework (it cannot be introspected), so it
    yields ``None``; ``BuiltIn`` delegates to :func:`loop_strategy_max_steps`.
    Mirrors Rust's ``StrategyRef::max_steps``.
    """
    match ref:
        case StrategyRefCustom():
            return None
        case StrategyRefBuiltIn():
            return loop_strategy_max_steps(ref.value)


# ============================================================================
# Composable Execution runtime scaffold (issue #123): StrategyOutcome +
# ExecutionContext / BudgetContext / BudgetStack / SpanStack.
# ============================================================================
#
# SCAFFOLD ONLY. This slice establishes the typed strategy outcome and the
# shared, mutable runtime context that threads through a nested strategy tree.
# ``BudgetContext.charge`` here is PURE ARITHMETIC against a per-scope step
# allowance — the behavior-chain walk, continue-consumption, and persistence
# are the later budget-enforcement slice (#124+).
#
# Rules established this slice:
#   - A child's :class:`StrategyOutcomeBudgetExhausted` is an INSPECTABLE value
#     the parent reads; it does NOT auto-propagate as a failure.
#   - ``charge`` is pure arithmetic: it debits ``turns`` steps; on success
#     increments ``steps_taken`` and returns ``None``; on overflow raises
#     :class:`BudgetExhausted` from current state WITHOUT mutating. It does NOT
#     walk the behavior chain or consume continues. ``BudgetPolicyUnlimited``
#     never exhausts.
#   - Each :class:`BudgetContext` represents ONE scope; the allowance is the
#     policy's own ``value`` (Unlimited = no cap → ``None``).
#   - All runtime types here are NEVER serialized — they are plain dataclasses
#     (NOT pydantic ``_Model``), holding live objects.
#
# Resolved spec ambiguities (DECIDED — see issue #123 plan):
#   1. :class:`ExecutionContext` holds the :class:`ExecutionRegistry` object;
#      ``run_strategy`` threads the context through. Typed under
#      ``TYPE_CHECKING`` to avoid the registry↔harness import cycle.
#   2. ``charge`` is pure arithmetic; :class:`BudgetExhausted` is a dedicated
#      exception raised by ``charge`` (recoverable-failure-as-raise inside the
#      harness boundary). :class:`StrategyOutcomeBudgetExhausted` mirrors its
#      fields and adds ``partial_output``. ``Output`` maps to ``str``.
#   3. ``continues_used`` is an IN-MEMORY field ONLY in this slice; checkpoint
#      persistence is DEFERRED to the enforcement slice.
#   4. :class:`SpanStack` reuses :data:`~spore_core.observability.SpanId`.


@dataclass
class StrategyOutcomeComplete:
    """The strategy completed and produced its final output. ``Output`` maps to
    ``str`` in this codebase (mirrors :class:`RunResultSuccess`)."""

    output: str


@dataclass
class ExhaustionCauseBudget:
    """The scope's step allowance ran out (the pre-#137 behavior). Resolves to
    :class:`HaltReasonBudgetExceeded` on a ``Fail`` / ``Escalate``→``Fail``
    terminal. Mirrors Rust's ``ExhaustionCause::Budget`` (the default)."""


@dataclass
class ExhaustionCauseToolErrorLoop:
    """The ReAct consecutive-recoverable-tool-error breaker hard-stopped at
    ``2 * error_loop_threshold`` identical-argument errors for ``tool`` (#137).
    Resolves to :class:`HaltReasonToolErrorLoop` on a ``Fail`` /
    ``Escalate``→``Fail`` terminal — NEVER ``BudgetExceeded``. Mirrors Rust's
    ``ExhaustionCause::ToolErrorLoop { tool, consecutive_errors }``."""

    tool: str
    consecutive_errors: int


# Why a :class:`StrategyOutcomeBudgetExhausted` was raised (issue #137). The
# budget-exhaustion resolution site (:meth:`StandardHarness._drive_strategy`)
# routes BOTH causes through the node's ``BudgetExhaustedBehavior``, but stamps a
# DIFFERENT terminal :data:`HaltReason` so a caller can tell a genuine budget
# exhaustion from an error-grinding circuit-break. Runtime-only (NOT serialized).
ExhaustionCause = ExhaustionCauseBudget | ExhaustionCauseToolErrorLoop


@dataclass
class ErrorRun:
    """The current run of consecutive recoverable tool errors for ONE tool name
    (issue #137). Tracked per tool name in a loop-local ``dict[str, ErrorRun]``
    inside :meth:`StandardHarness._run_react_inner`:

    - On a recoverable :class:`ToolOutputError` for tool ``T`` with args ``A``:
      if the existing run for ``T`` has ``args`` STRUCTURALLY equal to ``A``,
      ``count`` is incremented; otherwise the run is REPLACED with a fresh
      ``ErrorRun(args=A, count=1)`` (covers the first error and the args-changed
      case).
    - On ANY success for tool ``T``: the entry is REMOVED (AC1 — success resets
      the run regardless of args).
    - At ``count == N`` (``error_loop_threshold``): inject ONE corrective message
      (AC2). ``injected`` is set so the message is NOT re-injected between ``N``
      and ``2 * N`` for the same run.
    - At ``count == 2 * N``: stop looping (AC3).

    Mirrors Rust's loop-local ``ErrorRun { args, count, injected }``."""

    args: dict[str, Any]
    count: int
    injected: bool = False


@dataclass
class StrategyOutcomeBudgetExhausted:
    """The strategy's budget scope ran out of allowance. Mirrors the
    :class:`BudgetExhausted` charge-error fields and adds ``partial_output`` —
    any output produced before exhaustion. Inspectable by a parent, NOT
    auto-propagating: a parent reads it to decide whether to grant a continue or
    escalate.

    ``cause`` (#137) discriminates a genuine budget exhaustion from a ReAct
    tool-error-loop break so the single resolution site picks the right terminal
    :data:`HaltReason`. Defaults to :class:`ExhaustionCauseBudget` for every
    pre-#137 path."""

    policy: BudgetPolicy
    behavior: BudgetExhaustedBehavior
    steps_taken: int
    continues_used: int
    phase: str
    partial_output: str | None = None
    cause: ExhaustionCause = field(default_factory=ExhaustionCauseBudget)


@dataclass
class StrategyOutcomeFailed:
    """The strategy halted with a :data:`HarnessError`. Distinguishable from
    :class:`StrategyOutcomeBudgetExhausted` by callers (``isinstance``)."""

    error: HarnessError


# The typed result a strategy node returns. Runtime-only (NOT serialized): a
# strategy outcome is an in-process value, never persisted. A child's
# ``BudgetExhausted`` is a value the parent INSPECTS; it does NOT auto-propagate.
StrategyOutcome = StrategyOutcomeComplete | StrategyOutcomeBudgetExhausted | StrategyOutcomeFailed


class BudgetExhausted(SporeError):
    """Raised by :meth:`BudgetContext.charge` when a debit would exceed the
    scope's step allowance. Captures the budget state at the moment of
    exhaustion. Runtime-only (NOT serialized).

    Recoverable-failure-as-raise inside the harness boundary: a strategy
    promotes a caught :class:`BudgetExhausted` to a
    :class:`StrategyOutcomeBudgetExhausted` (adding ``partial_output``) at the
    strategy boundary."""

    def __init__(
        self,
        policy: BudgetPolicy,
        behavior: BudgetExhaustedBehavior,
        steps_taken: int,
        continues_used: int,
        phase: str,
    ) -> None:
        self.policy = policy
        self.behavior = behavior
        self.steps_taken = steps_taken
        self.continues_used = continues_used
        self.phase = phase
        super().__init__(f"budget exhausted at phase {phase!r}: {steps_taken} steps taken")


class ExhaustedResolution(str, Enum):
    """The runtime-only resolution of a :class:`BudgetExhaustedBehavior` chain at
    the moment of exhaustion (#125). NOT serialized — purely a control-flow signal
    returned by :meth:`BudgetContext.resolve_exhausted`.

    - ``CONTINUE`` — the scope was granted an in-process continue (counter reset,
      ``continues_used`` bumped); the caller loops again.
    - ``FAIL`` — terminate; ``partial_output = None`` (discarded by contract).
    - ``ESCALATE`` — hand off to the parent; ``partial_output = Some(..)`` carries
      the node-concrete partial.
    """

    CONTINUE = "continue"
    FAIL = "fail"
    ESCALATE = "escalate"


@dataclass
class BudgetContext:
    """One budget scope in the strategy tree. Each recursion node gets its OWN
    ``BudgetContext``; siblings do NOT share. Runtime-only (NOT serialized).

    The per-scope step allowance is the policy's own ``value``: ``TotalSteps`` /
    ``PerLoop`` / ``PerAttempt`` all expose ``value`` as the cap for this scope;
    ``Unlimited`` is uncapped (:meth:`remaining` → ``None``).

    ``continues_used`` is an in-memory field ONLY in this slice; its checkpoint
    persistence is deferred to the enforcement slice."""

    policy: BudgetPolicy
    behavior: BudgetExhaustedBehavior
    phase: str
    steps_taken: int = 0
    continues_used: int = 0
    #: In-process auto-grants spent at this scope's escalation point under
    #: :class:`EscalationModeAutoContinue` (SC-5). Bounded by the mode's
    #: ``max_grants``. Runtime-only, like ``continues_used``; never serialized.
    auto_grants_used: int = 0

    def _allowance(self) -> int | None:
        """The per-scope step allowance (``None`` for ``Unlimited``)."""
        return budget_allowance_value(self.policy)

    def charge(self, turns: int) -> None:
        """Debit ``turns`` steps against the scope allowance (pure arithmetic).
        On success increments ``steps_taken``. If the debit would exceed the
        allowance, raises :class:`BudgetExhausted` from current state WITHOUT
        mutating. Does NOT walk the behavior chain or consume continues.
        ``Unlimited`` never exhausts."""
        allowance = self._allowance()
        if allowance is not None and self.steps_taken + turns > allowance:
            raise BudgetExhausted(
                policy=self.policy,
                behavior=self.behavior,
                steps_taken=self.steps_taken,
                continues_used=self.continues_used,
                phase=self.phase,
            )
        self.steps_taken += turns

    def remaining(self) -> int | None:
        """Steps left in this scope (``allowance - steps_taken``, saturating ≥
        0). ``None`` for ``Unlimited`` (no cap)."""
        allowance = self._allowance()
        if allowance is None:
            return None
        return max(allowance - self.steps_taken, 0)

    def continues_remaining(self) -> int:
        """Continues left before fall-through. For a ``Continue`` behavior this
        is ``max_continues - continues_used`` (saturating ≥ 0); for
        ``Escalate`` / ``Fail`` there are no continues, so ``0``."""
        if isinstance(self.behavior, BudgetExhaustedContinue):
            return max(self.behavior.max_continues - self.continues_used, 0)
        return 0

    def consume_continue(self) -> None:
        """Grant one in-process continue (#125): bump ``continues_used`` and RESET
        ``steps_taken`` to 0 so the scope's step allowance refreshes for the next
        round. A purely in-memory reset — the session / messages are untouched
        (the loop keeps the same conversation; only the per-scope step counter
        rewinds). The ``continues_used`` persistence across a serialized
        checkpoint is DEFERRED to #129. Mirrors Rust's ``consume_continue``."""
        self.continues_used += 1
        self.steps_taken = 0

    def grant_auto_continue(self, steps_per_grant: int) -> None:
        """Grant one in-process AUTO-continue at this scope's escalation point
        (SC-5, :class:`EscalationModeAutoContinue`): bump ``auto_grants_used``
        and raise the scope's cap to ``steps_taken + steps_per_grant`` so the
        loop gets exactly ``steps_per_grant`` more steps after the exhaustion
        point (mirrors a human ``ContinueWithBudget { steps }`` grant, applied
        automatically). ``Unlimited`` never reaches here. Unlike
        :meth:`consume_continue` this does NOT rewind ``steps_taken`` — the cap
        moves up instead, so the grant is a strict, additive
        ``steps_per_grant``. Mirrors Rust's ``grant_auto_continue``."""
        self.auto_grants_used += 1
        granted = self.steps_taken + steps_per_grant
        self.policy = _grant_budget_policy(self.policy, granted)

    def auto_grants_remaining(self, max_grants: int) -> int:
        """Auto-grants still available before falling through to the autonomous
        terminal, given the mode's ``max_grants`` (SC-5). ``0`` once spent.
        Mirrors Rust's ``auto_grants_remaining``."""
        return max(max_grants - self.auto_grants_used, 0)

    @classmethod
    def resumed(
        cls,
        policy: BudgetPolicy,
        behavior: BudgetExhaustedBehavior,
        phase: str,
        continues_used: int,
    ) -> BudgetContext:
        """Reconstruct a RESUMED scope (#129) whose ``continues_used`` is seeded
        from a cross-process checkpoint — the sole field of :class:`BudgetContext`
        that must survive a process pause. ``steps_taken`` starts at 0 (the resumed
        run re-enters the loop with a fresh per-round step budget; the checkpoint
        only carries how many continues were ALREADY spent so a ``Continue``
        spanning the pause cannot exceed ``max_continues``). Runtime-only —
        ``continues_used`` is read off the
        :class:`HumanRequestBudgetExhausted` payload (Q3: NOT a new serialized
        :class:`BudgetContext` / :class:`PausedState` field). Mirrors Rust's
        ``BudgetContext::resumed``."""
        return cls(
            policy=policy,
            behavior=behavior,
            phase=phase,
            steps_taken=0,
            continues_used=continues_used,
        )

    def resolve_exhausted(self) -> ExhaustedResolution:
        """Resolve this scope's :class:`BudgetExhaustedBehavior` at the moment of
        exhaustion (#125), walking the on-exhausted fall-through chain:

        - ``Fail``     → :attr:`ExhaustedResolution.FAIL`.
        - ``Escalate`` → :attr:`ExhaustedResolution.ESCALATE`.
        - ``Continue {max_continues, on_exhausted}`` →
            - if :meth:`continues_remaining` > 0: :meth:`consume_continue` (reset
              counter, bump ``continues_used``) and return
              :attr:`ExhaustedResolution.CONTINUE`;
            - otherwise the continues are spent: ADOPT the boxed ``on_exhausted``
              behavior as this scope's behavior and recurse into it (the
              fall-through), so a ``Continue {on_exhausted: Escalate}`` whose
              continues are spent resolves to ``Escalate``.

        Mutates ``self``: on a granted continue the counter resets; on
        fall-through ``self.behavior`` is replaced by the nested behavior so
        subsequent resolutions see the post-fall-through behavior. Mirrors Rust's
        ``resolve_exhausted``."""
        behavior = self.behavior
        if isinstance(behavior, BudgetExhaustedFail):
            return ExhaustedResolution.FAIL
        if isinstance(behavior, BudgetExhaustedEscalate):
            return ExhaustedResolution.ESCALATE
        # Continue: grant a reset if any remain, else fall through.
        if self.continues_remaining() > 0:
            self.consume_continue()
            return ExhaustedResolution.CONTINUE
        self.behavior = behavior.on_exhausted
        return self.resolve_exhausted()


@dataclass
class BudgetStack:
    """Runtime push/pop stack of :class:`BudgetContext` scopes — one node per
    recursion frame, pushed on descent and popped on ascent. Runtime-only (NOT
    serialized). Siblings get DISTINCT contexts and do not share state."""

    stack: list[BudgetContext] = field(default_factory=list)

    def push(self, cx: BudgetContext) -> None:
        """Push a new scope onto the stack."""
        self.stack.append(cx)

    def pop(self) -> BudgetContext | None:
        """Pop the current scope, returning it (``None`` if empty)."""
        return self.stack.pop() if self.stack else None

    def current(self) -> BudgetContext | None:
        """The current (innermost) scope, or ``None`` if empty. The scope is
        mutable in place (a dataclass), so callers ``charge`` it directly."""
        return self.stack[-1] if self.stack else None

    def depth(self) -> int:
        """The current stack depth (recursion frames active)."""
        return len(self.stack)


@dataclass
class SpanStack:
    """Runtime push/pop stack of :data:`~spore_core.observability.SpanId`.
    Runtime-only (NOT serialized)."""

    stack: list[SpanId] = field(default_factory=list)

    def push(self, span_id: SpanId) -> None:
        """Push a span id onto the stack."""
        self.stack.append(span_id)

    def pop(self) -> SpanId | None:
        """Pop the current span id, returning it (``None`` if empty)."""
        return self.stack.pop() if self.stack else None

    def depth(self) -> int:
        """The current stack depth."""
        return len(self.stack)


@dataclass
class RunScratch:
    """Per-run mutable orchestration state threaded through the recursive
    strategy tree (#124). Runtime-only (NOT serialized).

    The combinator bodies set up the per-phase sub-``task`` here before recursing
    and restore the parent afterwards; the leaf (:func:`_run_react_config`) reads
    it to drive the ReAct window. ``on_stream`` lives here (not re-threaded per
    call) because it is moved out of the context only at the harness entry.

    ``terminal_override`` carries either a non-terminal pause
    (``WaitingForHuman`` / ``Consult`` / ``Escalate``) or a fully-formed terminal
    that must propagate up the recursion VERBATIM as a :class:`RunResult` rather
    than collapse into a :class:`StrategyOutcome` — preserving the typed
    :class:`HaltReason` and accounting through the recursive executor."""

    task: Task | None = None
    # ``SessionState`` is defined later in this module; defer the factory.
    run_session: SessionState = field(default_factory=lambda: SessionState())
    run_budget: BudgetSnapshot = field(default_factory=lambda: BudgetSnapshot())
    terminal_override: RunResult | None = None
    # Cross-process Continue checkpoint seed (#129): ``(phase, continues_used)``
    # carried from a resumed :class:`HumanRequestBudgetExhausted`. The FIRST
    # :meth:`ExecutionContext.push_budget` whose ``phase`` matches seeds the
    # reconstructed scope's ``continues_used`` (via :meth:`BudgetContext.resumed`)
    # and CLEARS this seed — so a ``Continue`` spanning a process pause resumes
    # with the correct continue count (AC2). Runtime-only; the value rides the
    # request payload, NOT a serialized :class:`BudgetContext` / :class:`PausedState`
    # field (Q3). ``None`` on a fresh run and after the seed is consumed (an
    # in-process Continue never sets it → AC3: no serialization on the in-process
    # path). Mirrors Rust's ``RunScratch::resume_continues``.
    resume_continues: tuple[str, int] | None = None
    # Phase-agnostic resume seed (#131 consult re-drive + #138 budget resume): the
    # stalled worker conversation carried from a resumed pause. For a CONSULT
    # resume the consult answer is already injected as the pending call's tool
    # result; for a BUDGET resume (#138) it is the worker's full post-exhaustion
    # session. When set, a :class:`PlanExecuteConfig` walk resumes its single
    # ``InProgress`` task from THIS session (instead of a fresh instruction-seeded
    # session) so the stalled worker continues mid-loop, its SelfVerifying
    # evaluator still runs, and the ready-set walk proceeds.
    #
    # #138 AC3 (plan-phase exhaustion): when the durable task_list has NO
    # ``InProgress`` task, the exhaustion happened in the PLAN phase before any
    # task was authored — the walk seeds the PLAN session from this carried
    # conversation instead of cloning a fresh base session.
    #
    # The FIRST PlanExecute walk consumes and CLEARS it. Runtime-only — the
    # session itself rides the serialized ``PausedState.session_state``, so a
    # cross-process resume reconstructs this seed in
    # :meth:`StandardHarness.resume_consult` (consult) or :meth:`resume`'s
    # ``ContinueWithBudget`` arm (#138 budget). ``None`` on a fresh run / after the
    # seed is consumed. Mirrors Rust's ``RunScratch::resume_seed``.
    resume_seed: SessionState | None = None


@dataclass
class ExecutionContext:
    """The one shared, mutable runtime context threaded through a whole nested
    strategy tree (issue #123/#124). Holds the :class:`ExecutionRegistry` for the
    duration of the run. Runtime-only — NOT serialized (a plain dataclass
    holding live objects, incl. an optional non-serializable ``stream`` sink).

    ``registry`` is typed under ``TYPE_CHECKING`` only: the
    :mod:`~spore_core.execution_registry` module imports :class:`RunStrategy`
    from here, so a runtime import would cycle.

    ## Recursive executor wiring (#124)

    The per-variant ``run`` bodies are LOOP OWNERS: a combinator recurses by
    calling ``run_strategy(self.inner, cx)`` (or ``self.plan`` / ``self.execute``
    as applicable). The model-touching primitives stay on the harness behind the
    :class:`StrategyExecutor` Protocol, reachable through :attr:`executor`. The
    per-run orchestration state (``task`` / ``run_session`` / ``run_budget`` /
    ``terminal_override``) lives in :attr:`scratch` so it threads across recursion
    through this one shared context."""

    registry: ExecutionRegistry
    budgets: BudgetStack = field(default_factory=BudgetStack)
    usage: AggregateUsage = field(default_factory=AggregateUsage)
    # ``SessionState`` is defined later in this module, so the factory is
    # deferred via a lambda (resolved at construction time, never at import).
    session: SessionState = field(default_factory=lambda: SessionState())
    spans: SpanStack = field(default_factory=SpanStack)
    stream: StreamSink | None = None
    # The harness primitives the per-variant run bodies delegate to (#124).
    # ``None`` only for the scaffold/unit fixtures that exercise the runtime
    # context without a real harness (the recursion stub tests).
    executor: StrategyExecutor | None = None
    scratch: RunScratch = field(default_factory=RunScratch)

    def _current_task(self) -> Task:
        """The current per-run task. The harness always sets it before driving a
        strategy; cloned (a deep copy) because the recursive bodies mutate the
        context while reading the task."""
        if self.scratch.task is None:
            raise AssertionError(
                "ExecutionContext.scratch.task must be set before running a strategy"
            )
        return self.scratch.task.model_copy(deep=True)

    def _require_executor(self) -> StrategyExecutor | StrategyOutcomeFailed:
        """The executor primitives, or a TYPED failure outcome when absent (the
        scaffold-only contexts). Real harness runs always wire one — never a
        raise (CONVENTIONS: a missing executor is a typed ``Failed``, not a
        crash)."""
        if self.executor is None:
            return StrategyOutcomeFailed(
                error=HarnessErrorInvalidConfiguration(
                    reason="ExecutionContext has no StrategyExecutor wired"
                )
            )
        return self.executor

    def _record_terminal(self, result: RunResult) -> StrategyOutcome:
        """Record a terminal/pause :class:`RunResult` from a whole-loop primitive
        (ReAct / SelfVerifying / Ralph / HillClimbing): carry the post-run session
        into the scratch (so a parent resumes losslessly) and stash the FULL
        result in ``terminal_override`` so the harness entry returns it VERBATIM —
        preserving the strategy's typed :class:`HaltReason` and accounting.
        Returns the matchable :class:`StrategyOutcome` for a wrapping combinator.

        Usage is NOT folded into ``self.usage`` here: the primitive's RunResult
        already carries the cumulative usage for its subtree (returned verbatim as
        the override), so folding would double-count."""
        if isinstance(result, RunResultSuccess | RunResultFailure):
            self.scratch.run_session = result.session_state
        outcome = _outcome_from_run_result(result)
        self.scratch.terminal_override = result
        return outcome

    def _take_child_override(self) -> RunResult | None:
        """Take the FULL terminal :class:`RunResult` a child strategy stashed into
        ``terminal_override`` when it returned from ``run_strategy(child, cx)``
        (#124). A combinator that recurses per-phase / per-task calls this
        immediately after each child dispatch to fold the child's usage / turns /
        session back into the shared execute context. Clearing the override is
        REQUIRED: the combinator builds its OWN terminal once the whole loop
        finishes (via :meth:`_finish`), and a stale child override would otherwise
        propagate verbatim and mask it."""
        result = self.scratch.terminal_override
        self.scratch.terminal_override = None
        return result

    async def _finish(
        self,
        executor: StrategyExecutor,
        parent_task: Task,
        result: RunResult,
    ) -> StrategyOutcome:
        """A combinator's terminal seam: finalize observability for ``result``,
        restore the parent ``task`` into scratch, stash ``result`` as the override
        so the harness entry returns it VERBATIM, and return the matching
        outcome."""
        await executor.finalize(result)
        # #131: a ``Consult`` propagated from a worker leaf carries the LEAF task,
        # so a host ``resume_consult`` would resume only that leaf and lose the
        # surrounding walk. As the pause unwinds through each combinator's
        # ``_finish``, rewrite its ``state.task`` to the combinator's OWN composed
        # task; by the top it carries the FULL tree, so ``resume_consult``
        # re-drives the whole strategy (the in-progress task resumes from
        # ``resume_seed``).
        if isinstance(result, RunResultConsult):
            new_state = result.state.model_copy()
            new_state.task = parent_task
            result = result.model_copy(update={"state": new_state})
        # SC-BUG-1: a HITL pause (``RunResultWaitingForHuman``) propagated from a
        # worker leaf ALSO carries the LEAF task (a leaf records its terminal via
        # ``_record_terminal``, which never rewrites it). Without the same rewrite,
        # a host ``resume`` runs only that bare leaf and loses the surrounding
        # frame — the SelfVerifying evaluate phase / PlanExecute walk never re-runs
        # (so under ``AlwaysAsk`` the verify gate silently degrades to a plain
        # executor). Rewrite it on the way up exactly like ``Consult``, so
        # ``_resume_inner`` re-drives the whole composed strategy from the approved
        # worker session.
        elif isinstance(result, RunResultWaitingForHuman):
            new_state = result.state.model_copy()
            new_state.task = parent_task
            result = result.model_copy(update={"state": new_state})
        self.scratch.task = parent_task
        if isinstance(result, RunResultSuccess | RunResultFailure):
            self.scratch.run_session = result.session_state
        outcome = _outcome_from_run_result(result)
        self.scratch.terminal_override = result
        return outcome

    def push_budget(
        self,
        policy: BudgetPolicy,
        behavior: BudgetExhaustedBehavior,
        phase: str,
    ) -> int:
        """Push a fresh per-node :class:`BudgetContext` scope for ``policy`` /
        ``behavior`` / ``phase`` onto ``self.budgets`` (#125). Each node —
        including a sibling — gets its OWN scope (``steps_taken = 0``), so a node
        capped at N never spends a sibling's allowance (rule 1) and a child's
        exhaustion never touches the parent scope (rule 4/7). Returns the depth
        AFTER the push. Mirrors Rust's ``push_budget``.

        #129 (AC2): if a resumed ``Continue`` checkpoint seed is waiting for THIS
        ``phase``, reconstruct the scope with its prior ``continues_used``
        (consuming the seed once) via :meth:`BudgetContext.resumed` instead of
        zeroing it. The root resumed node pushes first, and the request's ``phase``
        names that node, so the FIRST matching push restores the count. Any other
        push (or a fresh run) is unaffected."""
        seed = self.scratch.resume_continues
        if seed is not None and seed[0] == phase:
            self.scratch.resume_continues = None
            scope = BudgetContext.resumed(policy, behavior, phase, seed[1])
        else:
            scope = BudgetContext(policy=policy, behavior=behavior, phase=phase)
        self.budgets.push(scope)
        return self.budgets.depth()

    def pop_budget(self) -> BudgetContext | None:
        """Pop the current per-node budget scope (#125). Always paired with
        :meth:`push_budget` so the stack returns to its parent baseline on
        ascent. Mirrors Rust's ``pop_budget``."""
        return self.budgets.pop()

    def charge_current(self, turns: int) -> BudgetExhausted | None:
        """Charge ``turns`` steps against the CURRENT (innermost) budget scope
        (#125): the real enforcement point. Returns ``None`` when within
        allowance; returns the :class:`BudgetExhausted` (the charge error,
        capturing the budget state at exhaustion) when the debit would exceed the
        allowance. A node with no pushed scope (the scaffold contexts) never
        exhausts — charging is a no-op ``None``.

        NOTE (Python #123 idiomatic divergence): ``BudgetContext.charge`` RAISES
        :class:`BudgetExhausted`; this helper catches it and returns it as a value
        so the recursive run bodies branch on it without try/except scattered
        through the loop. Mirrors Rust's ``charge_current`` (which returns
        ``Result``)."""
        scope = self.budgets.current()
        if scope is None:
            return None
        try:
            scope.charge(turns)
        except BudgetExhausted as err:
            return err
        return None

    def resolve_current(self) -> ExhaustedResolution:
        """Resolve the current scope's exhaustion behavior (#125). Walks the chain
        (Continue grants a reset; spent continues fall through). A node with no
        pushed scope resolves to ``FAIL`` (defensive — should not happen in a
        wired run). Mirrors Rust's ``resolve_current``."""
        scope = self.budgets.current()
        if scope is None:
            return ExhaustedResolution.FAIL
        return scope.resolve_exhausted()

    def try_auto_continue(self, mode: EscalationMode) -> bool:
        """SC-5: attempt one :class:`EscalationModeAutoContinue` auto-grant at
        the CURRENT scope. Returns ``True`` when the mode is ``AutoContinue``,
        grants remain (``auto_grants_used < max_grants``), and the scope was
        refreshed with ``steps_per_grant`` more steps — the caller should then
        ``continue`` its loop IN-PROCESS (the scope is still on the stack).
        Fires ``on_grant`` for the grant. Returns ``False`` for any other mode,
        when grants are spent, or when there is no current scope — the caller
        then falls through to its existing pause (``SurfaceToHuman``) / abort
        (``Autonomous``) handling. Mirrors Rust's ``try_auto_continue``."""
        from .execution_registry import AutoGrantInfo, EscalationModeAutoContinue

        if not isinstance(mode, EscalationModeAutoContinue):
            return False
        scope = self.budgets.current()
        if scope is None:
            return False
        if scope.auto_grants_remaining(mode.max_grants) == 0:
            return False
        scope.grant_auto_continue(mode.steps_per_grant)
        if mode.on_grant is not None:
            mode.on_grant(
                AutoGrantInfo(
                    grant_number=scope.auto_grants_used,
                    steps_granted=mode.steps_per_grant,
                    phase=scope.phase,
                )
            )
        return True


@runtime_checkable
class StrategyExecutor(Protocol):
    """The harness-side LEAF primitives the per-variant ``run`` bodies delegate to
    (#124). Implemented by :class:`StandardHarness`. This is the seam that lets the
    recursive config bodies own their loops while the model-touching machinery
    (ReAct turn-loop window, SelfVerifying evaluate phase, HillClimbing metric
    machinery, Ralph ``.spore/`` checks) stays where it is tested.

    For PlanExecute (#124) recursion is GENUINE: :func:`_run_plan_execute_config`
    OWNS the plan→execute orchestration and dispatches its children via
    ``run_strategy(self.plan, cx)`` / ``run_strategy(self.execute, cx)`` once per
    task. The harness keeps only the PlanExecute leaf helpers — the plan directive,
    the plan-subtree dispatch, the artifact capture/persist, the deep-resume
    reconcile, and the ``OnTaskAdvance`` fire — none of which touch the per-task
    model loop.

    Each whole-loop primitive returns a terminal :class:`RunResult` for its phase;
    the config bodies translate the terminal into a :class:`StrategyOutcome` (or
    recurse)."""

    async def react_window(
        self,
        task: Task,
        max_iterations: int,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
        agent: Agent,
        toolset: ToolsetRef = "",
        output_schema: dict[str, Any] | None = None,
        output_schema_max_retries: int = 0,
        system_prompt: str | None = None,
    ) -> RunResult:
        """Run ONE bounded ReAct turn-loop window (the leaf primitive) on the
        resolved worker ``agent`` (#124). Issue 2: ``toolset`` is the leaf's
        RESOLVED handle — empty (``""``) ⇒ global-catalogue fallback; a non-empty
        handle with its own catalogue ⇒ strict per-node scoping.

        Issue #139: ``output_schema`` is the leaf's RESOLVED output schema
        (``None`` ⇒ no delivery, no validation). When set, the window VALIDATES
        its terminal against it, retries up to ``output_schema_max_retries`` extra
        turns, and sets :attr:`ModelParams.output_schema` on every turn so the
        Ollama ``format`` channel constrains decoding.

        SC-10 (#161): ``system_prompt`` is the leaf's per-node system-prompt
        OVERRIDE. ``None`` ⇒ the window uses the global ``config.system_prompt``
        (byte-identical to pre-SC-10). When set, it REPLACES the global prompt for
        every turn of this window, so the leaf sees ONLY its own prompt. Mirrors
        ``toolset``."""
        ...

    def resolve_agent_ref(self, ref: str, session_id: SessionId) -> Agent | RunResultFailure:
        """Resolve an ``AgentRef`` to its registered agent (#124). The leaf and the
        combinators resolve their worker agent through this so a missing handle is
        a typed terminal ``Failure`` (carrying an ``UnresolvedHandle``) rather than
        a raise."""
        ...

    async def evaluate_phase(
        self,
        task: Task,
        eval_agent: Agent,
        eval_toolset: ToolsetRef,
        carried: BudgetSnapshot,
        total_usage: AggregateUsage,
    ) -> RunResult:
        """Run the SelfVerifying evaluate phase (#124): a fresh evaluator RUN over
        a read-only sandbox in a never-shared session, on ``eval_agent``, scoped to
        ``eval_toolset`` (empty handle ⇒ global-catalogue fallback). Folds the
        run's usage into ``total_usage`` / ``carried``; returns its terminal."""
        ...

    async def append_user_message(self, session_state: SessionState, text: str) -> None:
        """Append ``text`` as a user message on ``session_state`` (Default-FAIL)."""
        ...

    def workspace_root(self) -> Path:
        """The configured sandbox workspace root (#124, for ``VerifierInput``)."""
        ...

    async def ralph_seed_session(self, instruction: str) -> SessionState:
        """Build the per-window Ralph seed session (#124): a fresh
        :class:`SessionState` seeded with ``instruction``, the ``.spore/`` reload
        context (R3), and the optional VCS history block."""
        ...

    async def ralph_completion_status(self) -> str | None:
        """Ralph external completion check (#124): ``None`` ⇒ complete; ``str``
        reason ⇒ tasks remain. ASYNC since #142: the Ralph checkpoint moved off
        the ``.spore/`` filesystem onto the durable project-id
        :class:`~spore_core.storage.RunStore`."""
        ...

    def ralph_max_resets(self) -> int:
        """The Ralph outer-loop reset cap (``config.max_resets``, #124)."""
        ...

    def resolve_metric_evaluator(self, key: str, session_id: SessionId) -> Any | RunResultFailure:
        """Resolve the HillClimbing metric evaluator for ``key`` (#124, Q2), or a
        typed misconfiguration ``Failure`` when absent."""
        ...

    async def hill_baseline(
        self,
        evaluator: Any,
        session_id: SessionId,
        task_id: TaskId,
        direction: OptimizationDirection,
        rows: list[Any],
        span_seq: list[int],
        total_usage: AggregateUsage,
        turns: int,
    ) -> float | RunResultFailure:
        """HillClimbing iteration-0 baseline (#124): evaluate the metric (no agent
        turn), record the row + span, and return the baseline value — or a
        ``Failure`` on a baseline-evaluation failure (already records the failed
        row + writes the TSV)."""
        ...

    async def hill_iteration(
        self,
        evaluator: Any,
        session_id: SessionId,
        task_id: TaskId,
        iteration: int,
        direction: OptimizationDirection,
        revert_on_no_improvement: bool,
        min_improvement_delta: float | None,
        current_best: float,
        rows: list[Any],
        span_seq: list[int],
    ) -> tuple[float, bool]:
        """HillClimbing per-iteration metric eval + keep/revert decision (#124):
        the agent turn already ran (recursively); this evaluates the metric,
        applies ``should_keep``, optionally reverts, records the row + span, and
        returns ``(current_best, non_improvement)``."""
        ...

    async def hill_write_tsv(self, task_id: TaskId, rows: list[Any]) -> None:
        """Write the HillClimbing results TSV (#124, leaf primitive)."""
        ...

    def budget_exceeded(
        self, budget: BudgetLimits, used: BudgetSnapshot, started_at: float
    ) -> BudgetLimitTypeT | None:
        """The wall-time/cost/token budget gate (#124, HillClimbing)."""
        ...

    def plan_directive(self, instruction: str) -> str:
        """The planning directive seeded before the plan sub-strategy runs (R1) —
        the "respond with a single JSON plan" instruction wrapped around the
        task. Seeded by the recursive PlanExecute ``run`` body before it dispatches
        ``self.plan`` (#124)."""
        ...

    async def seed_user_message(self, session_state: SessionState, text: str) -> None:
        """Append ``text`` as a user message onto ``session_state`` through the
        :class:`ContextManager` seam (#124). Used by the recursive PlanExecute body
        to seed the plan directive / per-step instruction before dispatching a
        child sub-strategy."""
        ...

    async def run_plan_subtree(
        self,
        plan: LoopStrategy,
        plan_task: Task,
        plan_session: SessionState,
        budget_used: BudgetSnapshot,
    ) -> RunResult | None:
        """Dispatch the plan sub-strategy ``plan`` for ``plan_task`` over
        ``plan_session``, returning its terminal :class:`RunResult` (#124).
        Genuinely recursive — the child's ``run`` drives its whole loop. Routes the
        configured ``planner_agent`` (R5/R6) by running the child against an
        agent-swapped child harness when one is set; otherwise the default agent
        runs the plan turns. ``None`` ⇒ the child produced no terminal."""
        ...

    async def capture_plan_artifact(
        self,
        session_id: SessionId,
        plan_output: str,
        usage: AggregateUsage,
        turns: int,
    ) -> _PlanPhaseOutcome | RunResult:
        """Capture + persist a :class:`PlanArtifact` from the plan sub-strategy's
        final output text (#124). The leaf model work (the planner turns) runs
        through ``run_plan_subtree``; this primitive owns only the harness-side
        artifact machinery: parse the response (R3), fire ``OnPlanCreated`` (R11),
        and persist to the RunStore under ``PLAN_EXECUTE_EXTRAS_KEY`` (R4). Returns
        the captured outcome or a terminal failure to propagate."""
        ...

    async def reconcile_completed_tasks(self, session_id: SessionId, task_list: object) -> None:
        """Reconcile a freshly-parsed task list against the DURABLE RunStore
        checkpoint (A.6 deep-resume): any task already ``Completed`` on the
        checkpoint is marked ``Completed`` in ``task_list`` so it is NOT re-run."""
        ...

    async def fire_task_advance(
        self,
        session_id: SessionId,
        step_task: Task,
        task_index: int,
        total_tasks: int,
    ) -> Task:
        """Fire the ``OnTaskAdvance`` hook (pre, mutable) for an execute step. The
        hook may rewrite ``step_task.instruction``; the (possibly mutated) task is
        returned and is what the execute sub-strategy then runs."""
        ...

    async def persist_task_list(self, session_id: SessionId, task_list: object) -> None:
        """Persist a parsed task list through the RunStore seam."""
        ...

    async def load_task_list(self, session_id: SessionId) -> object | None:
        """Load the persisted :class:`~spore_core.tasklist.TaskList` from the
        RunStore ``task_list`` store (#126, decision C) — the ONE authoring path
        that can carry real ``blockers``. Returns the parsed list, or ``None``
        when nothing was persisted (or the blob is unparseable)."""
        ...

    def take_observed_writes(self) -> list[str]:
        """Drain + return the harness-OBSERVED write/edit file paths recorded for
        the currently-running execute step (#126 AC2), clearing the accumulator.
        Never a model-self-reported field."""
        ...

    def clear_observed_writes(self) -> None:
        """Clear the harness-observed write/edit accumulator so the next execute
        step's ``files_touched`` reflects ONLY its own writes (#126 AC2)."""
        ...

    async def finalize(self, result: RunResult) -> None:
        """Finalize observability for a terminal outcome (no-op for pauses)."""
        ...

    def escalation_mode(self) -> EscalationMode:
        """The configured budget-escalation mode (#130). The per-config bodies
        and the strategy driver consult this at each
        :attr:`ExhaustedResolution.ESCALATE` site: ``SurfaceToHuman`` pauses with
        a :class:`HumanRequestBudgetExhausted`; ``Autonomous`` keeps the existing
        propagate behavior."""
        ...

    def enforce_output_schemas(self) -> bool:
        """The output-schema enforcement MIGRATION GATE (issue #139). ``False``
        (the default) means :func:`_run_react_config` never
        resolves/delivers/validates a leaf's ``output`` schema — byte-identical to
        pre-#139. Read through this accessor (mirrors :meth:`escalation_mode`)."""
        ...

    def output_schema_max_retries(self) -> int:
        """The number of EXTRA terminal-validation retry turns ``N`` granted under
        output-schema enforcement (issue #139; total attempts = ``1 + N``). Read
        by :func:`_run_react_config` to thread into the window. Default ``2``."""
        ...


@runtime_checkable
class RunStrategy(Protocol):
    """The runtime composition seam: every strategy node knows how to run itself
    given an :class:`ExecutionContext`. The ONLY dispatch is one ``match`` on
    :func:`run_strategy` delegating to each config's per-variant ``run`` body
    (#124 — the central dispatch ``match`` in ``_run_inner`` is GONE). Each
    per-config body OWNS its loop: a combinator recurses via
    ``run_strategy(self.inner, cx)`` (or ``self.plan`` / ``self.execute``), and
    the leaf (:func:`_run_react_config`) drives one bounded ReAct window through
    the :class:`StrategyExecutor` primitive. Without a wired executor (the
    scaffold-only contexts) every body returns a TYPED
    :class:`StrategyOutcomeFailed` — never raises (CONVENTIONS)."""

    async def run(self, cx: ExecutionContext) -> StrategyOutcome: ...


def _outcome_from_run_result(result: RunResult) -> StrategyOutcome:
    """Translate a terminal :class:`RunResult` into a :class:`StrategyOutcome`
    (#124, Q5): ``Success → Complete(output)``, every non-success terminal →
    ``Failed``. A ``BudgetExceeded`` failure maps to ``Failed`` here (the
    enforcement-slice ``StrategyOutcomeBudgetExhausted`` value is produced by
    ``BudgetContext.charge`` at the boundary; full HITL-through-recursion is
    #130). The pause variants are handled separately (the override path) and
    degrade to a typed failure only if they ever reach this mapping."""
    if isinstance(result, RunResultSuccess):
        return StrategyOutcomeComplete(output=result.output)
    if isinstance(result, RunResultFailure):
        return StrategyOutcomeFailed(
            error=HarnessErrorInvalidConfiguration(reason=repr(result.reason))
        )
    # WaitingForHuman / Consult / Escalate must travel the override path.
    return StrategyOutcomeFailed(
        error=HarnessErrorInvalidConfiguration(
            reason="non-terminal outcome reached strategy boundary"
        )
    )


# ============================================================================
# Per-node budget enforcement + failure isolation (issue #125)
# ============================================================================
#
# This slice makes :meth:`BudgetContext.charge` the REAL per-node enforcement
# point and :class:`StrategyOutcomeBudgetExhausted` a real, isolated,
# parent-inspectable value. #123 built ``BudgetContext`` / ``BudgetStack`` /
# ``charge`` as pure-arithmetic scaffold (dead outside tests); #124 wired
# enforcement through the LEGACY path (``task.budget.max_turns``) and never
# produced a ``BudgetExhausted`` outcome. #125 replaces that with charge-based
# enforcement at each ``*Config.run`` and a typed, non-cascading exhaustion.
#
# Resolved spec forks (DECIDED by the maintainer — do NOT re-litigate):
#   - ``Escalate`` carries ``partial_output = <node JSON>``; ``Fail`` carries
#     ``partial_output = None``.
#   - ``partial_output`` is a JSON-serialized string of the structured per-node
#     partial (shapes per the helpers below).
#   - ``continues_used`` persistence stays DEFERRED to #129 — in-process Continue
#     ONLY, no serialization, no new fixtures.
#
# DESIGN (mirrors the Rust impl): the ``BudgetExhaustedBehavior`` field was never
# wired onto any config struct (a serialized wire change forbidden by fork #3).
# So ``charge`` / ``resolve_exhausted`` provide the full behavior-chain mechanism,
# and the live ``*Config.run`` bodies push scopes with an in-process ``Escalate``
# placeholder behavior; the Continue/Fail/Escalate chains are exercised by unit
# tests driving the enforcement primitives directly. This is the in-process-only
# contract #125 was scoped to.


def _last_final_response_text(result: RunResult) -> str | None:
    """The last FinalResponse text from a ReAct window terminal (#125, fork #2):
    the ``Success.output``, or for a ``Failure`` the last assistant text message
    on its post-run session state (the partial captured before exhaustion).
    ``None`` for non-terminal pauses. Mirrors Rust's
    ``last_final_response_text``."""
    if isinstance(result, RunResultSuccess):
        return result.output
    if isinstance(result, RunResultFailure):
        for message in reversed(result.session_state.messages):
            if message.role == Role.ASSISTANT and isinstance(message.content, TextContent):
                return message.content.text
        return None
    return None


def _react_partial_json(last_final_response: str) -> str:
    """ReAct partial: the window's last FinalResponse text (#125, fork #2)."""
    return json.dumps(
        {"node": "react", "last_final_response": last_final_response},
        separators=(",", ":"),
    )


def _plan_execute_partial_json(task_list: "TaskList") -> str:
    """PlanExecute partial: the task list + per-task statuses + ledger (#125, fork
    #2). ``ledger`` is the per-task ``(id, description, status)`` rows."""
    ledger = [
        {
            "id": str(t.id),
            "description": t.description,
            "status": t.status.value,
        }
        for t in task_list.tasks
    ]
    return json.dumps(
        {"node": "plan_execute", "tasks": len(task_list.tasks), "ledger": ledger},
        separators=(",", ":"),
    )


def _self_verifying_partial_json(last_worker_output: str, last_verdict: str) -> str:
    """SelfVerifying partial: the last worker result summary + the last verdict
    reason (#125, fork #2)."""
    return json.dumps(
        {
            "node": "self_verifying",
            "last_worker_result": last_worker_output,
            "last_verdict": last_verdict,
        },
        separators=(",", ":"),
    )


def _hill_climbing_partial_json(best_score: float) -> str:
    """HillClimbing partial: the best candidate value + its score (#125, fork
    #2)."""
    return json.dumps(
        {"node": "hill_climbing", "best_candidate": best_score, "score": best_score},
        separators=(",", ":"),
    )


def _promote_budget_exhausted(
    err: BudgetExhausted, partial_output: str | None
) -> StrategyOutcomeBudgetExhausted:
    """Promote a charge-time :class:`BudgetExhausted` to a
    :class:`StrategyOutcomeBudgetExhausted` (#125 promotion boundary), attaching
    ``partial_output``. Per fork #1: an ``Escalate``-resolved exhaustion carries
    a partial; a ``Fail``-resolved one carries ``None``. The caller supplies the
    node-concrete partial JSON; this helper threads the budget state from the
    charge error. Mirrors Rust's ``promote_budget_exhausted``."""
    return StrategyOutcomeBudgetExhausted(
        policy=err.policy,
        behavior=err.behavior,
        steps_taken=err.steps_taken,
        continues_used=err.continues_used,
        phase=err.phase,
        partial_output=partial_output,
    )


def _promote_tool_error_loop(
    leaf_budget: BudgetPolicy,
    leaf_behavior: BudgetExhaustedBehavior,
    steps_taken: int,
    tool: str,
    consecutive_errors: int,
    partial_output: str | None,
) -> StrategyOutcomeBudgetExhausted:
    """Promote a ReAct tool-error-loop hard-stop (issue #137) to a
    :class:`StrategyOutcomeBudgetExhausted` carrying
    :class:`ExhaustionCauseToolErrorLoop` so it flows through the SAME single
    budget-exhaustion resolution site (:meth:`StandardHarness._drive_strategy`)
    as a real budget exhaustion, but the ``Fail`` / ``Escalate``→``Fail``
    terminals report :class:`HaltReasonToolErrorLoop` instead of
    ``BudgetExceeded``. The leaf's CONFIGURED ``behavior`` is carried so the
    resolution site can honor ``Fail`` / ``Escalate`` / ``Continue`` (a granted
    ``Continue`` re-drives the window with a fresh per-tool error allowance, since
    the counter is loop-local to ``_run_react_inner``). ``steps_taken`` is the
    window's turn count at the break, so the terminal's ``turns`` reflects work
    actually done — the breaker does NOT burn the rest of the budget. Mirrors
    Rust's ``promote_tool_error_loop``."""
    return StrategyOutcomeBudgetExhausted(
        policy=leaf_budget,
        behavior=leaf_behavior,
        steps_taken=steps_taken,
        continues_used=0,
        phase="react",
        partial_output=partial_output,
        cause=ExhaustionCauseToolErrorLoop(tool=tool, consecutive_errors=consecutive_errors),
    )


# ----- #130: Escalate HITL pause / resume wiring --------------------------
#
# The ``escalation_mode`` config knob (#120, previously UNCONSUMED) is read at
# every ``ExhaustedResolution.ESCALATE`` site. Under ``Autonomous`` the existing
# propagate behavior is unchanged; under ``SurfaceToHuman`` the node PAUSES with
# a :class:`HumanRequestBudgetExhausted` request via the ``terminal_override``
# seam instead of propagating up. ``available_actions`` is ADVISORY (fork D):
# resume does NOT hard-reject an out-of-set action.


def _combinator_escalation_actions(err: BudgetExhausted) -> list[EscalationAction]:
    """The advisory ``available_actions`` a COMBINATOR (PlanExecute / Ralph /
    SelfVerifying / HillClimbing) offers on a budget-exhaustion pause (fork C):
    ``[ContinueWithBudget, Skip, Fail]`` — a combinator CAN skip the node and let
    its outer loop advance. ``ContinueWithBudget`` seeds ``steps`` from the spent
    allowance (``steps_taken``) as a sensible default grant."""
    steps = err.steps_taken
    return [
        EscalationActionContinueWithBudget(steps=steps),
        EscalationActionSkip(),
        EscalationActionFail(),
    ]


def _leaf_escalation_actions(err: BudgetExhausted) -> list[EscalationAction]:
    """The advisory ``available_actions`` a BARE LEAF offers on a
    budget-exhaustion pause (fork C): ``[ContinueWithBudget, Fail]`` — a leaf has
    no sibling to ``Skip`` to, so ``Skip`` is OMITTED."""
    steps = err.steps_taken
    return [
        EscalationActionContinueWithBudget(steps=steps),
        EscalationActionFail(),
    ]


def _promote_budget_exhausted_to_human(
    err: BudgetExhausted,
    partial_output: str | None,
    available_actions: list[EscalationAction],
    session_id: SessionId,
    task: Task,
    budget_used: BudgetSnapshot,
    turn_number: int,
    worker_session: SessionState | None = None,
    toolset: ToolsetRef = "",
) -> RunResult:
    """Promote a charge-time :class:`BudgetExhausted` to a
    :class:`RunResultWaitingForHuman` (#130 HITL pause boundary). Built ONLY when
    a node's ``Escalate`` resolution is consulted under
    :class:`EscalationModeSurfaceToHuman`; carries the node's ``partial_output``
    and the advisory ``available_actions``. The request also stashes
    ``steps_taken`` / ``continues_used`` so ``resume`` can reconstruct the node's
    budget context from the request alone (fork E).

    #138 AC2-a: the paused ``session_state`` is the FULL stalled worker session
    (``worker_session``) — parallel to how the consult pause preserves the worker
    conversation — so a budget RESUME re-attaches it as the in-progress task's
    seed and the worker continues with its real context instead of a partial-only
    stub. When ``worker_session`` is empty / ``None`` (a bare-leaf propagate site
    with no richer conversation, or a legacy caller), it falls back to the single
    ``partial_output`` assistant message so the pre-#138 behavior is intact.
    #138 AC4-a: ``toolset`` carries the stalled worker leaf's handle (e.g.
    ``"exec-tools"``), consistent with #140, instead of staying ``""``. Mirrors
    Rust's ``promote_budget_exhausted_to_human``."""
    # #138 AC2-a: prefer the FULL worker conversation; fall back to the
    # partial-only stub when the carried session is empty (back-compat).
    if worker_session is not None and worker_session.messages:
        session_state = worker_session
    else:
        messages = (
            [Message(role=Role.ASSISTANT, content=TextContent(text=partial_output))]
            if partial_output is not None
            else []
        )
        session_state = SessionState(messages=messages)
    request = HumanRequestBudgetExhausted(
        phase=err.phase,
        policy=err.policy,
        steps_taken=err.steps_taken,
        continues_used=err.continues_used,
        partial_output=partial_output,
        available_actions=available_actions,
    )
    state = PausedState(
        session_id=session_id,
        task_id=task.id,
        turn_number=turn_number,
        session_state=session_state,
        pending_tool_calls=[],
        approved_results=[],
        human_request=request,
        task=task,
        budget_used=budget_used,
        child_state=None,
        # #138 AC4-a: carry the stalled worker leaf's toolset handle (#140 parity)
        # so a resume routes the re-driven worker through its scoped catalogue.
        toolset=toolset,
    )
    return RunResultWaitingForHuman(state=state, request=request)


def _grant_budget_policy(policy: BudgetPolicy, granted: int) -> BudgetPolicy:
    """Return ``policy`` with its step cap raised to at least ``granted`` (#130
    ``ContinueWithBudget``). ``Unlimited`` is untouched; a capped policy is raised
    only when its current ``value`` is below ``granted``. Mirrors Rust's
    ``grant_budget_policy``."""
    if isinstance(policy, BudgetPolicyUnlimited):
        return policy
    if policy.value < granted:
        return policy.model_copy(update={"value": granted})
    return policy


def _grant_strategy_budget(ls: LoopStrategy, granted: int) -> LoopStrategy:
    """Recurse a :class:`LoopStrategy` tree raising every ReAct leaf's ``budget``
    cap to at least ``granted`` (#130). The combinator nodes carry no inline
    policy (they derive it from ``task.budget.max_turns``, raised by
    :func:`_grant_task_budget`), so this only touches the leaves. Mirrors Rust's
    ``grant_strategy_budget``."""
    if isinstance(ls, ReactConfig):
        return ls.model_copy(update={"budget": _grant_budget_policy(ls.budget, granted)})
    if isinstance(ls, PlanExecuteConfig):
        return ls.model_copy(
            update={
                "plan": _grant_strategy_budget(ls.plan, granted),
                "execute": _grant_strategy_budget(ls.execute, granted),
            }
        )
    if isinstance(ls, SelfVerifyingConfig):
        return ls.model_copy(update={"inner": _grant_strategy_budget(ls.inner, granted)})
    if isinstance(ls, RalphConfig):
        return ls.model_copy(update={"inner": _grant_strategy_budget(ls.inner, granted)})
    if isinstance(ls, HillClimbingConfig):
        return ls.model_copy(update={"inner": _grant_strategy_budget(ls.inner, granted)})
    return ls


def _grant_task_budget(task: Task, granted: int) -> Task:
    """Return ``task`` with its strategy tree's budget caps raised to ``granted``
    (#130 ``ContinueWithBudget``). The ReAct leaf caps live on each node's own
    ``budget`` policy; the combinator nodes derive their cap from
    ``task.budget.max_turns``, so BOTH are raised. Fork E: ``granted`` is
    ``request.steps_taken + steps``, so the restored scope has room for ``steps``
    more steps after the checkpoint. Mirrors Rust's ``grant_task_budget``."""
    new_budget = task.budget
    if task.budget.max_turns is None or task.budget.max_turns < granted:
        new_budget = task.budget.model_copy(update={"max_turns": granted})
    return task.model_copy(
        update={
            "budget": new_budget,
            "loop_strategy": _grant_strategy_budget(task.loop_strategy, granted),
        }
    )


async def _run_react_config(self: ReactConfig, cx: ExecutionContext) -> StrategyOutcome:
    """The leaf: a bounded ReAct turn-loop window. Reads the per-run scratch
    (``task`` / ``run_session`` / ``run_budget``) and drives one ReAct window
    through the executor primitive, threading the result back into the scratch."""
    executor = cx._require_executor()
    if isinstance(executor, StrategyOutcomeFailed):
        return executor
    task = cx._current_task()
    max_iterations = self.max_iterations()
    session_state = cx.scratch.run_session
    cx.scratch.run_session = SessionState()
    budget_used = cx.scratch.run_budget.model_copy(deep=True)
    # #124: resolve the worker agent from the registry by THIS leaf's handle
    # (genuine recursion — no ``config.agent``). A missing handle is a typed
    # terminal failure.
    agent = executor.resolve_agent_ref(self.agent, task.session_id)
    if isinstance(agent, RunResultFailure):
        await executor.finalize(agent)
        return cx._record_terminal(agent)
    # Output-schema delivery + enforcement (issue #139). MIGRATION GATE: only when
    # ``enforce_output_schemas`` is ON AND this leaf carries ``output`` set do we
    # resolve the schema, DELIVER it (directive seed + the constrained-decoding
    # channel), and pass it (plus the retry budget ``N``) into the window for
    # terminal validation. When the gate is OFF or there is no ``output``,
    # ``output_schema`` is ``None`` and the window behaves byte-identically to
    # pre-#139 (no resolve, no delivery, no validation). The schema is
    # canonicalized to compact key-sorted JSON so its delivered/reported bytes are
    # identical across the four language ports.
    output_schema: dict[str, Any] | None = None
    if executor.enforce_output_schemas() and self.output is not None:
        resolved = cx.registry.resolve_schema(self.output)
        if isinstance(resolved, dict):
            output_schema = resolved
    output_schema_max_retries = executor.output_schema_max_retries()
    if output_schema is not None:
        # AC1: append the resolved schema to the leaf's directive/system context
        # as a USER message, key-sorted via ``_canonicalize_json`` so the seeded
        # bytes are identical across languages.
        directive = (
            "Your final response must be a JSON value that conforms to this "
            f"JSON schema: {_canonicalize_json(output_schema)}"
        )
        await executor.seed_user_message(session_state, directive)
    # #125/#129: push this leaf's OWN budget scope carrying its CONFIGURED
    # ``behavior``. The leaf still never RESOLVES it in the nested case (rule 6: it
    # PROPAGATES a ``BudgetExhausted`` to its parent, which owns the single
    # recovery site). Carrying the real ``behavior`` only means the propagated
    # error reports it, so the TOP-LEVEL/bare-leaf resolution site
    # (:meth:`_drive_strategy`) can honor it (Q1 — a bare leaf self-resolves, a
    # nested leaf does not).
    cx.push_budget(self.budget, self.behavior, "react")
    # The leaf takes the run's stream sink for the window. Combinators that
    # recurse per-phase suppress it (they take it before recursing).
    on_stream = cx.stream
    cx.stream = None
    result = await executor.react_window(
        task,
        max_iterations,
        session_state,
        budget_used,
        on_stream,
        agent,
        # Issue 2: thread THIS leaf's toolset handle down so the window dispatches
        # the per-node scoped catalogue (empty handle ⇒ global-catalogue
        # fallback). Mirrors ``agent``.
        self.toolset,
        # Issue #139: thread the resolved output schema (or ``None``) and the
        # retry budget so the window validates the terminal.
        output_schema,
        output_schema_max_retries,
        # SC-10 (#161): thread THIS leaf's per-node system-prompt override
        # (``None`` ⇒ the window keeps using the global prompt). Mirrors
        # ``self.toolset``.
        self.system_prompt,
    )
    await executor.finalize(result)

    # #125: charge the window's turns against this leaf's OWN scope. The leaf
    # POLICY (``self.budget``) — not the global backstop — is the per-node
    # enforcement point. When the LEAF cap is the binding constraint (the window
    # consumed >= the leaf policy value) the leaf is exhausted and PROPAGATES a
    # typed ``BudgetExhausted`` to its parent (rule 6 — the leaf never
    # self-resolves). When the smaller GLOBAL backstop trips first, the legacy
    # ``BudgetExceeded`` terminal is recorded VERBATIM.
    if isinstance(result, RunResultSuccess | RunResultFailure):
        window_turns = result.turns
    else:
        window_turns = 0
    # #137: the window hit the consecutive-tool-error breaker's 2N hard stop.
    # PROPAGATE it through the SAME single budget-exhaustion resolution site (so
    # the node's ``behavior`` governs Fail/Escalate/Continue), but carry the
    # ``ToolErrorLoop`` cause so the terminal reports
    # :class:`HaltReasonToolErrorLoop`, never ``BudgetExceeded``. The window's
    # turns are still CHARGED against the leaf scope (accurate accounting), but
    # the breaker stopped EARLY — the remaining budget is NOT burned. Detection
    # is independent of ``leaf_cap_binding``.
    if isinstance(result, RunResultFailure) and isinstance(result.reason, HaltReasonToolErrorLoop):
        tool = result.reason.tool
        consecutive_errors = result.reason.consecutive_errors
        last_final = _last_final_response_text(result) or ""
        cx.scratch.run_session = result.session_state
        cx.charge_current(window_turns)
        cx.pop_budget()
        return _promote_tool_error_loop(
            self.budget,
            self.behavior,
            window_turns,
            tool,
            consecutive_errors,
            _react_partial_json(last_final),
        )
    window_hit_budget = isinstance(result, RunResultFailure) and isinstance(
        result.reason, HaltReasonBudgetExceeded
    )
    leaf_cap = budget_allowance_value(self.budget)
    leaf_cap_binding = window_hit_budget and leaf_cap is not None and window_turns >= leaf_cap
    charge_err = cx.charge_current(window_turns)
    if leaf_cap_binding or charge_err is not None:
        last_final = _last_final_response_text(result) or ""
        # Carry the post-run session so a parent resumes losslessly.
        if isinstance(result, RunResultSuccess | RunResultFailure):
            cx.scratch.run_session = result.session_state
        err = charge_err
        if err is None:
            # The window itself hit the cap; synthesize the charge error from the
            # current scope state.
            scope = cx.budgets.current()
            if scope is not None:
                err = BudgetExhausted(
                    policy=scope.policy,
                    behavior=scope.behavior,
                    steps_taken=scope.steps_taken,
                    continues_used=scope.continues_used,
                    phase=scope.phase,
                )
            else:
                err = BudgetExhausted(
                    policy=self.budget,
                    behavior=self.behavior,
                    steps_taken=window_turns,
                    continues_used=0,
                    phase="react",
                )
        cx.pop_budget()
        # Rule 6: the leaf PROPAGATES — partial carries the last FinalResponse
        # (Escalate semantics, fork #1/#2).
        return _promote_budget_exhausted(err, _react_partial_json(last_final))
    cx.pop_budget()
    return cx._record_terminal(result)


async def _run_plan_execute_config(
    self: PlanExecuteConfig, cx: ExecutionContext
) -> StrategyOutcome:
    """Plan→execute (#124). GENUINELY recursive: the plan phase dispatches
    ``self.plan`` (seeding the planning directive + a one-turn budget first) and
    the execute phase dispatches ``self.execute`` ONCE PER TASK. The child
    strategy's full loop runs for each phase — a non-ReAct execute child
    (SelfVerifying / HillClimbing) really executes per task, not a hardcoded flat
    ReAct (the defeated-the-point bug this fixes).

    This config body OWNS the orchestration: per-task turn/budget allocation
    (Q1), the ``OnTaskAdvance`` hook (pre, mutable), seeding each step instruction
    as a user message, A.6 deep-resume against the durable RunStore checkpoint,
    task-list persistence after each transition (Q4), and cumulative usage /
    last-output / last-state carry. The harness keeps only LEAF primitives: the
    constrained-plan capture/persist machinery, the deep-resume reconcile, and the
    ``OnTaskAdvance`` fire — none of which touch the per-task model loop.

    #126 — what this body now does
    -----------------------------
    * **Task list source (decision C):** after the plan phase persists the
      captured artifact (so the legacy plan→tasklist replay path is intact), the
      executor LOADS its runnable :class:`TaskList` from the persisted
      ``task_list`` tool store (:meth:`load_task_list`) — the ONE authoring path
      that can carry real ``blockers``. The plan artifact only seeds an initial
      linear list when nothing was authored via the tool.
    * **Cycle re-check (AC5):** :meth:`TaskList.has_cycle` is re-checked at
      execute entry (defense in depth) → :class:`HaltReasonTaskGraphCycle`.
    * **Ready-set walk (AC1):** repeatedly pick the LOWEST-id ``pending`` task
      whose blockers are all ``completed`` (:meth:`TaskList.next_ready`), run it,
      mark ``completed``, repeat — honoring the blocker DAG, deterministic id
      tiebreak. v1 runs ready tasks SEQUENTIALLY.
    * **Two-tier context (AC1 isolation):** each step is seeded with Tier-1 (its
      TRANSITIVE blockers' final outputs + their ledger rows, decision D) plus
      Tier-2 (the global running ledger, every step). Independent branches never
      appear in a task's seed.
    * **Ledger (decision B):** on completion a
      :class:`~spore_core.tasklist.StepLedgerEntry` is appended whose
      ``files_touched`` is HARNESS-OBSERVED from write/edit tool calls
      (:meth:`take_observed_writes`, AC2) — never self-reported. The ledger is
      bounded at :data:`STEP_LEDGER_MAX_ENTRIES` via drop-oldest.
    * **Failure cascade (decision A/E, AC3/AC4):** a terminal task failure
      (unrecoverable error OR a ``BudgetExhausted`` resolving to ``Fail``) marks
      its transitive dependents ``blocked`` (tracked in the run-local
      ``blocked_by_failure`` set) and KEEPS scheduling unrelated tasks; at drain
      the run returns :class:`HaltReasonTasksBlockedByFailure` with the full
      completed/blocked partition. A run where every task completes is Success.
    """
    from .tasklist import (
        StepLedgerEntry,
        TaskList,
        TaskStatus,
        plan_artifact_to_task_list,
        push_step_ledger,
        render_step_ledger,
    )

    executor = cx._require_executor()
    if isinstance(executor, StrategyOutcomeFailed):
        return executor
    task = cx._current_task()
    session_id = task.session_id
    # The incoming shared execute session (``[user: task.instruction]``).
    base_session = cx.scratch.run_session
    cx.scratch.run_session = SessionState()
    budget_used = cx.scratch.run_budget.model_copy(deep=True)
    # PlanExecute suppresses the run's stream sink for its phases; restore it
    # before returning so the parent-visible step boundaries are re-emitted.
    on_stream = cx.stream
    cx.stream = None

    # #131/#138: the phase-agnostic resume seed (the stalled worker conversation
    # carried from a resumed consult or budget pause). Taken BEFORE the plan phase
    # so AC3 (plan-phase exhaustion) can seed the plan session from it, and AC2
    # (execute-phase exhaustion) can re-attach it to the single ``InProgress`` task.
    resume_seed_session = cx.scratch.resume_seed
    cx.scratch.resume_seed = None

    # #138 AC1 (skip re-planning): probe the DURABLE task_list BEFORE the plan
    # phase. If a non-empty list already exists (e.g. a prior window authored it,
    # #142 makes it survive Ralph's per-window session reset), the plan phase is
    # REDUNDANT — go straight to reconcile + the ready-set walk. This is the core
    # fix: a budget/consult re-entry no longer burns its grant re-planning a graph
    # that is already durable. The plan artifact may be ABSENT in this case (AC1-a
    # artifact-optional) — nothing downstream requires it once a task_list exists.
    preexisting = await executor.load_task_list(session_id)
    skip_plan = isinstance(preexisting, TaskList) and bool(preexisting.tasks)

    if skip_plan:
        # ── AC1: skip-plan path. The list is the durable source of truth. ───────
        assert isinstance(preexisting, TaskList)
        task_list = preexisting

        # AC5 (defense in depth): re-check the durable graph for cycles.
        if task_list.has_cycle():
            cx.stream = on_stream
            result: RunResult = RunResultFailure(
                reason=HaltReasonTaskGraphCycle(
                    reason="persisted task graph contains a directed cycle",
                ),
                session_id=session_id,
                usage=AggregateUsage(),
                turns=budget_used.turns,
                session_state=SessionState(),
            )
            return await cx._finish(executor, task, result)

        # A.6 deep-resume: reconcile already-Completed tasks so they are NOT
        # re-run (handles the dedup the plan path would have done).
        await executor.reconcile_completed_tasks(session_id, task_list)
        await executor.persist_task_list(session_id, task_list)

        # No plan turn ran: usage starts empty and the shared budget is carried
        # unchanged (turns stay at the incoming ``budget_used.turns``).
        total_usage = AggregateUsage()
        carried = budget_used.model_copy(deep=True)
    else:
        # ── AC3 / fresh-plan path: run the plan phase. ──────────────────────────
        #
        # Seed the planning directive onto a CLONE of the base session so the
        # shared execute context stays ``[user: task.instruction]`` (#93 — a
        # leaked directive would make every execute step re-emit a plan). The plan
        # phase runs under the plan sub-strategy's OWN declared budget (e.g. a
        # ReAct ``PerLoop{4}``); the global ``max_turns`` is only the outer
        # backstop.
        #
        # #138 AC3 (plan-phase exhaustion, infer-from-task_list option iii): when a
        # resume seed is present AND the durable list has no ``InProgress`` task,
        # the exhaustion happened in the PLAN phase before authoring any task
        # (AC3-b: the list is still empty). Seed the PLAN session from the carried
        # conversation so the planner CONTINUES on it instead of starting fresh —
        # and consume the seed here so the execute-phase re-attach below does not
        # also fire.
        plan_resume = resume_seed_session is not None and (
            not isinstance(preexisting, TaskList)
            or all(t.status != TaskStatus.IN_PROGRESS for t in preexisting.tasks)
        )
        directive = executor.plan_directive(task.instruction)
        if plan_resume and resume_seed_session is not None:
            plan_session = resume_seed_session
            resume_seed_session = None
        else:
            plan_session = base_session.model_copy(deep=True)
        await executor.seed_user_message(plan_session, directive)
        plan_task = Task(
            id=task.id,
            instruction=directive,
            session_id=session_id,
            budget=task.budget.model_copy(deep=True),
            loop_strategy=self.plan,
        )
        plan_result = await executor.run_plan_subtree(
            self.plan, plan_task, plan_session, budget_used.model_copy(deep=True)
        )
        if isinstance(plan_result, RunResultSuccess):
            plan_output, plan_usage, plan_turns = (
                plan_result.output,
                plan_result.usage,
                plan_result.turns,
            )
        elif plan_result is None:
            cx.stream = on_stream
            result = RunResultFailure(
                reason=HaltReasonPlanPhaseFailed(
                    error=PlanPhaseErrorPayload(
                        kind="planning_turn_failed",
                        message="plan sub-strategy produced no terminal",
                    ),
                ),
                session_id=session_id,
                usage=AggregateUsage(),
                turns=budget_used.turns,
                session_state=SessionState(),
            )
            return await cx._finish(executor, task, result)
        else:
            # A non-success plan terminal (budget / agent error / pause)
            # propagates verbatim — the run never reaches execute.
            cx.stream = on_stream
            return await cx._finish(executor, task, plan_result)

        # Capture + persist the artifact from the plan child's output (R3/R4/R11) —
        # the harness-side machinery, no model turn.
        outcome = await executor.capture_plan_artifact(
            session_id, plan_output, plan_usage, plan_turns
        )
        if not isinstance(outcome, _PlanPhaseOutcome):
            cx.stream = on_stream
            return await cx._finish(executor, task, outcome)

        # #126 decision C: the runnable task list comes from the persisted
        # ``task_list`` tool store (the ONE authoring path — it can carry real
        # blockers). Fall back to the linear plan-artifact bridge only when nothing
        # was authored via the tool (back-compat with the #59/#124 plan-only path
        # and its replay fixtures).
        persisted = await executor.load_task_list(session_id)
        if isinstance(persisted, TaskList) and persisted.tasks:
            task_list = persisted
        else:
            # The fallback intentionally uses the deprecated linear bridge
            # (decision C keeps it working for the plan-only back-compat path);
            # silence its DeprecationWarning here (mirrors Rust's
            # ``#[allow(deprecated)]``).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                task_list = plan_artifact_to_task_list(outcome.artifact)
        if not task_list.tasks:
            cx.stream = on_stream
            result = RunResultFailure(
                reason=HaltReasonEmptyPlan(),
                session_id=session_id,
                usage=outcome.usage,
                turns=outcome.turns,
                session_state=SessionState(),
            )
            return await cx._finish(executor, task, result)

        # #126 AC5: re-check the WHOLE graph for cycles at execute entry (defense in
        # depth — ``add_task`` already rejects cycles, but the persisted store
        # could be cyclic out of band). No task runs.
        if task_list.has_cycle():
            cx.stream = on_stream
            result = RunResultFailure(
                reason=HaltReasonTaskGraphCycle(
                    reason="persisted task graph contains a directed cycle",
                ),
                session_id=session_id,
                usage=outcome.usage,
                turns=outcome.turns,
                session_state=SessionState(),
            )
            return await cx._finish(executor, task, result)
        await executor.persist_task_list(session_id, task_list)

        # Carry the shared budget past the plan phase.
        carried = budget_used.model_copy(deep=True)
        carried.turns = outcome.turns
        carried.input_tokens += outcome.usage.input_tokens
        carried.output_tokens += outcome.usage.output_tokens

        # A.6 deep-resume (Q2): reconcile against the durable checkpoint so
        # already-Completed tasks are not re-run.
        await executor.reconcile_completed_tasks(session_id, task_list)

        total_usage = outcome.usage.model_copy(deep=True)

    # ── Phase 2: ready-set DAG walk (#126). ─────────────────────────────────
    #
    # Each step is seeded from a FRESH copy of ``base_session`` (NOT a
    # forward-folded shared transcript — that breaks on a DAG) plus its Tier-1
    # scoped context and the Tier-2 global ledger (#93 — the plan directive never
    # leaks because the plan child ran over a clone).

    # #131 consult / #138 budget re-drive: a resumed pause carries the stalled
    # worker conversation in ``resume_seed_session``. The stalled task is the
    # single ``InProgress`` task in the durable list (PlanExecute marks a task
    # InProgress before running it). Reset it to ``Pending`` so ``next_ready``
    # re-schedules it, and remember its id so its step uses the carried session
    # instead of a fresh one — the worker then continues mid-loop and its
    # evaluator still runs. (When the seed was already consumed by the AC3
    # plan-resume path above, this is a no-op.)
    consult_resume_session = resume_seed_session
    consult_resume_task: int | None = None
    if consult_resume_session is not None:
        in_progress = next(
            (t.id for t in task_list.tasks if t.status == TaskStatus.IN_PROGRESS),
            None,
        )
        if in_progress is not None:
            # Reset the status field DIRECTLY (bypassing ``update``'s forward-only
            # transition guard, which rejects InProgress->Pending): the consult
            # resume legitimately re-schedules the in-flight task.
            for t in task_list.tasks:
                if t.id == in_progress:
                    t.status = TaskStatus.PENDING
            consult_resume_task = in_progress
        else:
            # No in-progress task to resume (out-of-contract); drop the seed so
            # the walk proceeds normally rather than stalling.
            consult_resume_session = None

    last_output = ""
    last_state = SessionState()

    # #126 Tier-1/Tier-2 + cascade run-local state.
    # - ``final_outputs``: each completed task's ``Success.output`` (Tier-1).
    # - ``ledger``: the Tier-2 global running ledger (bounded, drop-oldest).
    # - ``ledger_elided``: sticky flag set the first time entries are dropped.
    # - ``blocked_by_failure``: tasks cascade-Blocked by a terminal failure
    #   (decision E — run-local scratch, NOT a TaskStatus variant).
    # - ``first_failure``: the first terminal failure that triggered a cascade
    #   (decision A), as ``(task_id, reason)``.
    final_outputs: dict[int, str] = {}
    ledger: list[StepLedgerEntry] = []
    ledger_elided = False
    blocked_by_failure: set[int] = set()
    first_failure: tuple[int, str] | None = None

    # #125: PlanExecute owns a budget scope for its execute phase. Its POLICY is
    # the task's global turn ceiling (``TotalSteps``). Behavior is ``Escalate``
    # (in-process placeholder; the serialized behavior field is #129).
    if task.budget.max_turns is not None:
        plan_policy: BudgetPolicy = BudgetPolicyTotalSteps(value=task.budget.max_turns)
    else:
        plan_policy = BudgetPolicyUnlimited()
    cx.push_budget(plan_policy, self.behavior, "plan_execute")

    # Total positional count for the OnTaskAdvance hook (stable).
    total_tasks = len(task_list.tasks)

    while True:
        task_id = task_list.next_ready()
        if task_id is None:
            break
        index = next(i for i, t in enumerate(task_list.tasks) if t.id == task_id)
        instruction = task_list.tasks[index].description

        # Mark InProgress and re-persist (Q4).
        task_list.update(task_id, TaskStatus.IN_PROGRESS)
        await executor.persist_task_list(session_id, task_list)

        # Fire OnTaskAdvance (pre, mutable). The hook may rewrite the step
        # instruction; the (possibly mutated) instruction seeds the execute child.
        step_task = Task(
            id=task.id,
            instruction=instruction,
            session_id=session_id,
            budget=task.budget.model_copy(deep=True),
            loop_strategy=self.execute,
        )
        step_task = await executor.fire_task_advance(session_id, step_task, index, total_tasks)

        # #131 consult re-drive: if THIS is the task that was consulting, resume
        # its worker from the carried conversation (answer already injected)
        # instead of a fresh instruction-seeded session — the worker continues
        # mid-loop and its evaluator still runs. Otherwise build the normal
        # fresh, Tier-1/Tier-2-seeded step session.
        if consult_resume_task is not None and task_id == consult_resume_task:
            step_session = (
                consult_resume_session
                if consult_resume_session is not None
                else base_session.model_copy(deep=True)
            )
            consult_resume_session = None
        else:
            # #126 Tier-1 scoped context: seed this step from a FRESH copy of the
            # base session plus, for THIS task's transitive blockers only, their
            # final outputs + their ledger rows. Independent branches never appear
            # (AC1).
            step_session = base_session.model_copy(deep=True)
            blockers = task_list.transitive_blockers(task_id)
            if blockers:
                blocker_set = set(blockers)
                tier1_lines = [
                    f"#{b} result: {final_outputs[b]}" for b in blockers if b in final_outputs
                ]
                if tier1_lines:
                    await executor.seed_user_message(
                        step_session, "Results from upstream tasks:\n" + "\n".join(tier1_lines)
                    )
                scoped = [e for e in ledger if e.task_id in blocker_set]
                scoped_block = render_step_ledger(scoped, False)
                if scoped_block is not None:
                    await executor.seed_user_message(step_session, scoped_block)

            # #126 Tier-2: inject the FULL global running ledger into EVERY step
            # (with the static elision marker once entries were dropped).
            ledger_block = render_step_ledger(ledger, ledger_elided)
            if ledger_block is not None:
                await executor.seed_user_message(step_session, ledger_block)

            # Finally seed this step's own instruction.
            await executor.seed_user_message(step_session, step_task.instruction)

        # #126 AC2: clear the observed-write accumulator so this task's
        # files_touched reflect ONLY the writes this step issues.
        executor.clear_observed_writes()

        cx.scratch.task = step_task
        cx.scratch.run_session = step_session
        cx.scratch.run_budget = carried.model_copy(deep=True)
        # #125: absolute turn count BEFORE this step, so the success path charges
        # only the DELTA against the PlanExecute scope.
        carried_before = carried.turns
        step_outcome = await run_strategy(self.execute, cx)

        # ── BudgetExhausted (#125 rule 4/7) — resolve THIS scope. ────────────
        if isinstance(step_outcome, StrategyOutcomeBudgetExhausted):
            # The execute LEAF already counted the worker's true consumed turns
            # (ABSOLUTE — including the incoming ``carried_before`` floor) and
            # carried them in the outcome's ``steps_taken``. Use THAT for the grant
            # accounting, NOT the parent ``plan_execute`` scope's ``steps_taken``:
            # that scope is ``Unlimited`` when the Task has ``max_turns = None`` (so
            # its ``steps_taken`` is 0), which made the resume grant
            # ``granted = 0 + steps`` a no-op and stalled the worker on the same
            # window every grant. Everything ELSE on ``err`` stays the parent
            # scope's: ``phase`` must remain "plan_execute" so the resume-seed
            # ``continues_used`` restore matches in ``_grant_task_budget``, and
            # ``continues_used`` is the combinator's (``max_continues``) count, not
            # the leaf's.
            leaf_steps = step_outcome.steps_taken
            scope = cx.budgets.current()
            assert scope is not None, "plan_execute scope pushed above"
            err = BudgetExhausted(
                policy=scope.policy,
                behavior=scope.behavior,
                steps_taken=leaf_steps,
                continues_used=scope.continues_used,
                phase=scope.phase,
            )
            resolution = cx.resolve_current()
            # #129: a granted ``Continue`` loops IN-PROCESS — the scope's
            # :meth:`resolve_current` already reset ``steps_taken`` and bumped
            # ``continues_used``. Reset this task to ``Pending`` and re-enter the
            # ready-set walk so it runs again under the refreshed scope allowance
            # (NO serialization — AC3). ``max_continues`` bounds the loop: once
            # continues are spent, the chain falls through to ``Fail``/``Escalate``.
            if resolution == ExhaustedResolution.CONTINUE:
                task_list.update(task_id, TaskStatus.PENDING)
                await executor.persist_task_list(session_id, task_list)
                continue
            # #126 AC4: a budget-``Fail`` task cascades IDENTICALLY to an
            # error-failed one — block its transitive dependents and keep
            # scheduling unrelated tasks.
            if resolution == ExhaustedResolution.FAIL:
                task_list.update(task_id, TaskStatus.BLOCKED)
                for dep in task_list.transitive_dependents(task_id):
                    task_list.update(dep, TaskStatus.BLOCKED)
                    blocked_by_failure.add(dep)
                blocked_by_failure.add(task_id)
                await executor.persist_task_list(session_id, task_list)
                if first_failure is None:
                    first_failure = (task_id, f"budget exhausted (Fail): {scope.policy!r}")
                continue
            # Escalate: under ``Autonomous`` surface the partial and abort the run.
            # Under ``SurfaceToHuman`` (#130) the node PAUSES with a
            # ``BudgetExhausted`` request via the ``terminal_override`` seam.
            from .execution_registry import EscalationModeSurfaceToHuman

            # SC-5: AutoContinue auto-grants this scope in-process (bounded by
            # ``max_grants``, fires ``on_grant``) and re-walks the ready set —
            # mirroring the ``Continue`` arm above. Once grants are spent it
            # falls through to the surface/abort handling below.
            if cx.try_auto_continue(executor.escalation_mode()):
                task_list.update(task_id, TaskStatus.PENDING)
                await executor.persist_task_list(session_id, task_list)
                continue
            surface = isinstance(executor.escalation_mode(), EscalationModeSurfaceToHuman)
            # #138 AC2-b: under SurfaceToHuman the task is NOT permanently failed —
            # it pauses awaiting a budget grant. Leave it ``InProgress`` (the
            # consult path's invariant) so the resume's seed re-attaches via the
            # SAME InProgress→Pending→complete machinery. Under Autonomous the run
            # aborts, so the task is ``Blocked`` as before.
            task_list.update(
                task_id,
                TaskStatus.IN_PROGRESS if surface else TaskStatus.BLOCKED,
            )
            await executor.persist_task_list(session_id, task_list)
            partial = _plan_execute_partial_json(task_list)
            # #138 AC2-a: carry the FULL stalled worker session (the execute child
            # left it in scratch) into the pause so a budget RESUME re-attaches it
            # as the in-progress task's seed — parallel to the consult pause —
            # instead of discarding it for a partial-only stub.
            worker_session = cx.scratch.run_session
            cx.scratch.run_session = SessionState()
            worker_toolset = _worker_toolset_of(self.execute)
            cx.pop_budget()
            cx.scratch.task = task
            cx.stream = on_stream
            # Advance the run-wide turn cursor to the worker's consumed turns
            # BEFORE building the pause (mirrors the Success branch's
            # ``carried.turns = sub_result.turns``). Otherwise the cursor stays
            # frozen at ``carried_before``, so the paused ``budget_used`` /
            # ``turn_number`` and the displayed ``TurnStart`` turn regress, and the
            # resumed window re-seeds its floor to the same value and re-runs the
            # same turns. ``leaf_steps`` is absolute (>= carried_before).
            carried.turns = leaf_steps

            if surface:
                waiting = _promote_budget_exhausted_to_human(
                    err,
                    partial,
                    _combinator_escalation_actions(err),
                    session_id,
                    task,
                    carried.model_copy(deep=True),
                    carried.turns,
                    worker_session,
                    worker_toolset,
                )
                return cx._record_terminal(waiting)
            return _promote_budget_exhausted(err, partial)

        sub_result = cx._take_child_override()

        if isinstance(sub_result, RunResultSuccess):
            carried.turns = sub_result.turns
            last_state = sub_result.session_state
            carried.input_tokens += sub_result.usage.input_tokens
            carried.output_tokens += sub_result.usage.output_tokens
            total_usage.input_tokens += sub_result.usage.input_tokens
            total_usage.output_tokens += sub_result.usage.output_tokens
            total_usage.cache_read_tokens += sub_result.usage.cache_read_tokens
            total_usage.cache_write_tokens += sub_result.usage.cache_write_tokens
            total_usage.cost_usd += sub_result.usage.cost_usd
            last_output = sub_result.output

            task_list.complete(task_id)
            await executor.persist_task_list(session_id, task_list)

            # #126: record this task's final output (Tier-1) and append a ledger
            # entry whose files_touched is HARNESS-OBSERVED (AC2).
            final_outputs[task_id] = sub_result.output
            files_touched = executor.take_observed_writes()
            entry = StepLedgerEntry(
                task_id=task_id, summary=sub_result.output, files_touched=files_touched
            )
            if push_step_ledger(ledger, entry):
                ledger_elided = True

            executor._emit(on_stream, StreamFinalResponse(content=last_output))

            # #125: charge this step's turns against the PlanExecute scope.
            charge_err = cx.charge_current(max(0, sub_result.turns - carried_before))
            if charge_err is not None:
                resolution = cx.resolve_current()
                # #129: a granted ``Continue`` refreshes the scope and keeps
                # scheduling the remaining ready tasks IN-PROCESS (this step already
                # completed). Do NOT pop the scope. NO serialization (AC3).
                if resolution == ExhaustedResolution.CONTINUE:
                    continue
                # SC-5: AutoContinue auto-grants in-process at this Escalate
                # point (scope still pushed) before surfacing.
                if resolution == ExhaustedResolution.ESCALATE and cx.try_auto_continue(
                    executor.escalation_mode()
                ):
                    continue
                partial = _plan_execute_partial_json(task_list)
                cx.pop_budget()
                cx.scratch.task = task
                cx.stream = on_stream
                if resolution == ExhaustedResolution.FAIL:
                    return _promote_budget_exhausted(charge_err, None)
                if resolution == ExhaustedResolution.ESCALATE:
                    from .execution_registry import EscalationModeSurfaceToHuman

                    if isinstance(executor.escalation_mode(), EscalationModeSurfaceToHuman):
                        # #138 AC2-a: this step already COMPLETED; carry its
                        # post-run session (``last_state``) so a budget resume
                        # re-attaches the real worker context, not a partial stub.
                        waiting = _promote_budget_exhausted_to_human(
                            charge_err,
                            partial,
                            _combinator_escalation_actions(charge_err),
                            session_id,
                            task,
                            carried.model_copy(deep=True),
                            carried.turns,
                            last_state,
                            _worker_toolset_of(self.execute),
                        )
                        return cx._record_terminal(waiting)
                    return _promote_budget_exhausted(charge_err, partial)
                raise AssertionError(f"unhandled resolution {resolution!r}")

        elif isinstance(sub_result, RunResultFailure):
            total_usage.input_tokens += sub_result.usage.input_tokens
            total_usage.output_tokens += sub_result.usage.output_tokens
            total_usage.cache_read_tokens += sub_result.usage.cache_read_tokens
            total_usage.cache_write_tokens += sub_result.usage.cache_write_tokens
            total_usage.cost_usd += sub_result.usage.cost_usd

            # A GLOBAL turn-budget hard stop (#117 backstop) surfaces as a
            # ``BudgetExceeded`` Failure from the leaf. That is a WHOLE-RUN hard
            # stop, NOT a single-task terminal failure — it aborts the run verbatim
            # (preserving the pre-#126 mid-execute budget behavior). It is distinct
            # from a per-NODE ``BudgetExhausted`` resolving to ``Fail``, which DOES
            # cascade (AC4, handled above).
            if isinstance(sub_result.reason, HaltReasonBudgetExceeded):
                task_list.update(task_id, TaskStatus.BLOCKED)
                await executor.persist_task_list(session_id, task_list)
                cx.pop_budget()
                cx.stream = on_stream
                result = RunResultFailure(
                    reason=sub_result.reason,
                    session_id=session_id,
                    usage=total_usage,
                    turns=sub_result.turns,
                    session_state=last_state,
                )
                return await cx._finish(executor, task, result)

            # #126 AC3: a terminal task FAILURE cascade-blocks its transitive
            # dependents and KEEPS scheduling unrelated tasks (replaces the Q5
            # blanket abort).
            carried.turns = sub_result.turns
            task_list.update(task_id, TaskStatus.BLOCKED)
            for dep in task_list.transitive_dependents(task_id):
                task_list.update(dep, TaskStatus.BLOCKED)
                blocked_by_failure.add(dep)
            blocked_by_failure.add(task_id)
            await executor.persist_task_list(session_id, task_list)
            if first_failure is None:
                first_failure = (task_id, repr(sub_result.reason))
            continue

        elif sub_result is None:
            # No terminal from the child: treat as a terminal failure of this task
            # and cascade (same as a Failure).
            task_list.update(task_id, TaskStatus.BLOCKED)
            for dep in task_list.transitive_dependents(task_id):
                task_list.update(dep, TaskStatus.BLOCKED)
                blocked_by_failure.add(dep)
            blocked_by_failure.add(task_id)
            await executor.persist_task_list(session_id, task_list)
            if first_failure is None:
                first_failure = (task_id, "execute sub-strategy produced no terminal")
            continue

        else:
            # A pause / consult / escalate propagates the whole run verbatim.
            cx.pop_budget()
            cx.stream = on_stream
            return await cx._finish(executor, task, sub_result)

    cx.pop_budget()
    cx.stream = on_stream

    # ── Drain (#126, decision A). ───────────────────────────────────────────
    #
    # A run where a terminal failure cascade-blocked any task returns a PARTIAL
    # terminal ``Failure`` reporting the full partition. A run where every task
    # completed returns ``Success`` (output = last step's text).
    if first_failure is not None:
        failed_task, fail_reason = first_failure
        completed = sorted(t.id for t in task_list.tasks if t.status == TaskStatus.COMPLETED)
        blocked = sorted(blocked_by_failure)
        result = RunResultFailure(
            reason=HaltReasonTasksBlockedByFailure(
                completed=completed,
                blocked=blocked,
                failed_task=failed_task,
                reason=fail_reason,
            ),
            session_id=session_id,
            usage=total_usage,
            turns=carried.turns,
            session_state=last_state,
        )
        return await cx._finish(executor, task, result)

    result = RunResultSuccess(
        output=last_output,
        session_id=session_id,
        usage=total_usage,
        turns=carried.turns,
        session_state=last_state,
    )
    return await cx._finish(executor, task, result)


async def _run_self_verifying_config(
    self: SelfVerifyingConfig, cx: ExecutionContext
) -> StrategyOutcome:
    """SelfVerifying (#124): GENUINELY recursive build↔evaluate loop. Each
    iteration dispatches ``run_strategy(self.inner, cx)`` for the build phase (a
    non-ReAct inner — e.g. PlanExecute — really runs its whole loop per
    iteration), then runs a fresh evaluate phase on the inner worker's resolved
    agent (Q1c) and consults the verifier resolved from ``self.evaluator``'s key
    (Q1a). Passed ⇒ Success; Failed ⇒ append the reason (Default-FAIL) and loop;
    exhausted ⇒ ``SelfVerifyExhausted``."""
    from .verifier import VerifierInput, VerifierVerdictPassed

    executor = cx._require_executor()
    if isinstance(executor, StrategyOutcomeFailed):
        return executor
    task = cx._current_task()
    build_session_id = task.session_id
    # SC-BUG-1: a HITL resume re-drives the whole SelfVerifying strategy with the
    # stalled build (worker) conversation carried in the phase-agnostic resume
    # seed. Consume it as the FIRST build iteration's session so the build phase
    # CONTINUES the approved/answered worker turn (which already carries the
    # original instruction + the dispatched tool result) instead of restarting
    # from an empty top-level session — then the evaluate phase + verifier run,
    # reaching the looper's eval-frame reviewer. On a fresh run the seed is
    # ``None`` and this is the incoming ``run_session``, byte-identical to before.
    # When SelfVerifying is NESTED under a PlanExecute walk, that outer walk takes
    # the seed BEFORE recursing, so this sees ``None`` and the nested behavior is
    # unchanged.
    session_state = cx.scratch.resume_seed or cx.scratch.run_session
    cx.scratch.resume_seed = None
    carried = cx.scratch.run_budget.model_copy(deep=True)
    # Suppress the run's stream sink for the recursive child phases.
    on_stream = cx.stream
    cx.stream = None

    # Q1a: resolve the verifier from ``evaluator``'s key (NO wire change).
    verifier = cx.registry.resolve_verifier(self.evaluator)
    if verifier is None:
        cx.stream = on_stream
        result: RunResult = RunResultFailure(
            reason=HaltReasonSelfVerifyMisconfigured(
                reason=(
                    f"SelfVerifying requires a verifier registered under key {self.evaluator!r}"
                )
            ),
            session_id=build_session_id,
            usage=AggregateUsage(),
            turns=0,
            session_state=SessionState(),
        )
        return await cx._finish(executor, task, result)
    # Q1c: the evaluate-phase agent defaults to the inner worker's agent; an
    # explicit ``eval_agent`` override (a dedicated reviewer, distinct from the
    # builder) takes precedence.
    eval_agent_ref = (
        self.eval_agent if self.eval_agent is not None else _worker_agent_key_of(self.inner)
    )
    eval_agent = executor.resolve_agent_ref(eval_agent_ref, build_session_id)
    if isinstance(eval_agent, RunResultFailure):
        cx.stream = on_stream
        return await cx._finish(executor, task, eval_agent)
    # The evaluate phase runs against ``eval_toolset`` (a read-only/inspection
    # catalogue) when set; None ⇒ the empty handle (global-catalogue fallback),
    # byte-identical to pre-override behavior.
    eval_toolset = self.eval_toolset if self.eval_toolset is not None else ""

    max_iterations = verifier.max_iterations()
    total_usage = AggregateUsage()
    last_reason = ""
    last_worker_output = ""

    # #125: SelfVerifying owns a budget scope for its build↔evaluate loop. POLICY
    # is the task's global turn ceiling (``TotalSteps``); behavior is ``Escalate``
    # (in-process placeholder; serialized behavior is #129).
    if task.budget.max_turns is not None:
        sv_policy: BudgetPolicy = BudgetPolicyTotalSteps(value=task.budget.max_turns)
    else:
        sv_policy = BudgetPolicyUnlimited()
    cx.push_budget(sv_policy, self.behavior, "self_verifying")

    for iteration in range(max_iterations):
        # ── Build phase: recurse ``run_strategy(self.inner, cx)``.
        build_task = Task(
            id=task.id,
            instruction=task.instruction,
            session_id=build_session_id,
            budget=task.budget,
            loop_strategy=self.inner,
        )
        cx.scratch.task = build_task
        cx.scratch.run_session = session_state.model_copy(deep=True)
        cx.scratch.run_budget = carried.model_copy(deep=True)
        carried_before = carried.turns
        build_outcome = await run_strategy(self.inner, cx)
        # #125 rule 4/7: a child's BudgetExhausted reaches THIS parent as a
        # StrategyOutcome, never auto-cascaded. SelfVerifying surfaces its own
        # typed BudgetExhausted (partial = last worker result + last verdict)
        # without charging the child's exhaustion against its own scope.
        if isinstance(build_outcome, StrategyOutcomeBudgetExhausted):
            partial = _self_verifying_partial_json(last_worker_output, last_reason)
            scope = cx.budgets.current()
            assert scope is not None, "self_verifying scope pushed above"
            err = BudgetExhausted(
                policy=scope.policy,
                behavior=scope.behavior,
                steps_taken=scope.steps_taken,
                continues_used=scope.continues_used,
                phase=scope.phase,
            )
            cx.pop_budget()
            cx.scratch.task = task
            cx.stream = on_stream
            return _promote_budget_exhausted(err, partial)
        build_result = cx._take_child_override()
        if build_result is None:
            build_result = RunResultFailure(
                reason=HaltReasonSelfVerifyMisconfigured(
                    reason="build sub-strategy produced no terminal"
                ),
                session_id=build_session_id,
                usage=AggregateUsage(),
                turns=carried.turns,
                session_state=session_state,
            )
        _fold_usage(total_usage, carried, build_result)

        # A paused / escalated build propagates verbatim.
        if isinstance(
            build_result, RunResultWaitingForHuman | RunResultConsult | RunResultEscalate
        ):
            cx.pop_budget()
            cx.stream = on_stream
            return await cx._finish(executor, task, build_result)
        # Capture the build's output for the partial (last worker result).
        if isinstance(build_result, RunResultSuccess):
            last_worker_output = build_result.output
        # Carry the build's post-run session forward for the next round.
        if isinstance(build_result, RunResultSuccess | RunResultFailure):
            session_state = build_result.session_state

        # #125: charge this iteration's build turns against the SelfVerifying
        # scope. If the global cap is spent, the node surfaces its OWN typed
        # BudgetExhausted (partial = last worker result + last verdict).
        charge_err = cx.charge_current(max(0, carried.turns - carried_before))
        if charge_err is not None:
            resolution = cx.resolve_current()
            # #129: a granted ``Continue`` resets the scope and RE-RUNS the
            # build↔evaluate iteration IN-PROCESS (do NOT pop the scope; the loop
            # continues under the refreshed allowance). NO serialization (AC3).
            # ``max_continues`` bounds the loop.
            if resolution == ExhaustedResolution.CONTINUE:
                continue
            # SC-5: AutoContinue auto-grants in-process at this Escalate point
            # (scope still pushed) before surfacing/aborting.
            if resolution == ExhaustedResolution.ESCALATE and cx.try_auto_continue(
                executor.escalation_mode()
            ):
                continue
            partial = _self_verifying_partial_json(last_worker_output, last_reason)
            cx.pop_budget()
            cx.scratch.task = task
            cx.stream = on_stream
            if resolution == ExhaustedResolution.FAIL:
                return _promote_budget_exhausted(charge_err, None)
            # #129: the in-process ``Continue`` is handled above; this arm is
            # ``Escalate``-only now (kept for the surface/propagate shape).
            if resolution == ExhaustedResolution.ESCALATE:
                from .execution_registry import EscalationModeSurfaceToHuman

                if isinstance(executor.escalation_mode(), EscalationModeSurfaceToHuman):
                    # #138 AC2-a: carry the FULL build (worker) session — the
                    # SelfVerifying loop carries it forward in ``session_state``
                    # after each iteration — so a budget resume re-attaches the
                    # real worker context.
                    waiting = _promote_budget_exhausted_to_human(
                        charge_err,
                        partial,
                        _combinator_escalation_actions(charge_err),
                        build_session_id,
                        task,
                        carried.model_copy(deep=True),
                        carried.turns,
                        session_state,
                        _worker_toolset_of(self.inner),
                    )
                    return cx._record_terminal(waiting)
                return _promote_budget_exhausted(charge_err, partial)
            # Defensive: ``ExhaustedResolution`` is exhausted above.
            raise AssertionError(f"unhandled resolution {resolution!r}")

        # ── Evaluate phase: a fresh evaluator run on ``eval_agent``.
        eval_result = await executor.evaluate_phase(
            task, eval_agent, eval_toolset, carried, total_usage
        )

        # #147: charge the evaluator's OWN turns against the SelfVerifying scope,
        # mirroring the build charge above. ``evaluate_phase`` runs the evaluator
        # in an isolated harness from a fresh ``BudgetSnapshot``, so
        # ``eval_result.turns`` is the evaluator's standalone turn count — and
        # ``_fold_usage`` folds it via ``max`` (NOT a sum), so it never reaches
        # ``carried.turns`` once build turns dominate. Charging the build delta
        # alone therefore missed the evaluate phase entirely; charge it explicitly
        # here so both phases count against the budget.
        if isinstance(eval_result, RunResultSuccess | RunResultFailure | RunResultEscalate):
            eval_turns = eval_result.turns
        else:
            eval_turns = 0
        eval_charge_err = cx.charge_current(eval_turns)
        if eval_charge_err is not None:
            resolution = cx.resolve_current()
            # A granted ``Continue`` resets the scope and re-runs the
            # build↔evaluate iteration in-process (mirrors the build path).
            if resolution == ExhaustedResolution.CONTINUE:
                continue
            # SC-5: AutoContinue auto-grants in-process at this Escalate point
            # (scope still pushed) before surfacing/aborting.
            if resolution == ExhaustedResolution.ESCALATE and cx.try_auto_continue(
                executor.escalation_mode()
            ):
                continue
            partial = _self_verifying_partial_json(last_worker_output, last_reason)
            cx.pop_budget()
            cx.scratch.task = task
            cx.stream = on_stream
            if resolution == ExhaustedResolution.FAIL:
                return _promote_budget_exhausted(eval_charge_err, None)
            if resolution == ExhaustedResolution.ESCALATE:
                from .execution_registry import EscalationModeSurfaceToHuman

                if isinstance(executor.escalation_mode(), EscalationModeSurfaceToHuman):
                    waiting = _promote_budget_exhausted_to_human(
                        eval_charge_err,
                        partial,
                        _combinator_escalation_actions(eval_charge_err),
                        build_session_id,
                        task,
                        carried.model_copy(deep=True),
                        carried.turns,
                        session_state,
                        _worker_toolset_of(self.inner),
                    )
                    return cx._record_terminal(waiting)
                return _promote_budget_exhausted(eval_charge_err, partial)
            # Defensive: ``ExhaustedResolution`` is exhausted above.
            raise AssertionError(f"unhandled resolution {resolution!r}")

        verdict = await verifier.verify(
            VerifierInput(
                build_result=build_result,
                eval_result=eval_result,
                workspace=executor.workspace_root(),
                iteration=iteration,
            )
        )
        if isinstance(verdict, VerifierVerdictPassed):
            if isinstance(build_result, RunResultSuccess):
                output, turns = build_result.output, build_result.turns
                final_state = build_result.session_state
            else:
                output, turns = "", carried.turns
                final_state = session_state
            cx.pop_budget()
            cx.stream = on_stream
            result = RunResultSuccess(
                output=output,
                session_id=build_session_id,
                usage=total_usage,
                turns=turns,
                session_state=final_state,
            )
            return await cx._finish(executor, task, result)
        last_reason = verdict.reason
        await executor.append_user_message(session_state, verdict.reason)

    cx.pop_budget()
    cx.stream = on_stream
    result = RunResultFailure(
        reason=HaltReasonSelfVerifyExhausted(
            iterations=max_iterations,
            last_reason=last_reason,
        ),
        session_id=build_session_id,
        usage=total_usage,
        turns=carried.turns,
        session_state=session_state,
    )
    return await cx._finish(executor, task, result)


async def _run_ralph_config(self: RalphConfig, cx: ExecutionContext) -> StrategyOutcome:
    """Ralph (#124): GENUINELY recursive continuation wrapper. Each context window
    seeds a FRESH session from the ``.spore/`` checkpoint, then recurses
    ``run_strategy(self.inner, cx)`` (a non-ReAct inner — e.g. SelfVerifying —
    really runs its whole loop per window). Ralph is an OUTER LOOP that re-runs
    ``inner`` as the architect declared it — it does NOT replace nodes the inner
    tree already assigned. When ``self.agent`` is set it only FILLS a worker leaf
    that left its own agent handle EMPTY (the bare-leaf Ralph convenience); an
    explicit leaf agent is authoritative and never overwritten.
    ``ralph_completion_status`` drives the OUTER reset loop;
    exhaustion ⇒ ``RalphCompletionUnmet``. Ralph discards the incoming session
    state by design (each window is a fresh start re-seeded from the filesystem
    checkpoint)."""
    executor = cx._require_executor()
    if isinstance(executor, StrategyOutcomeFailed):
        return executor
    task = cx._current_task()
    on_stream = cx.stream
    cx.stream = None
    cx.scratch.run_session = SessionState()
    max_resets = max(executor.ralph_max_resets(), 1)

    # Ralph fills — never replaces. When ``self.agent`` is set it supplies a
    # worker agent ONLY where the inner leaf left its handle empty; an
    # explicitly-declared leaf agent (the architect's node) is authoritative.
    inner_for_window = (
        self.inner if not self.agent else _fill_empty_worker_agent(self.inner, self.agent)
    )

    total_usage = AggregateUsage()
    cumulative_turns = 0
    last_reason = ".spore/progress.json missing"
    last_session_id = task.session_id

    for iteration in range(max_resets):
        window_session_id = task.session_id if iteration == 0 else new_session_id()
        last_session_id = window_session_id

        # R2/R3: a FRESH session seeded from the ``.spore/`` checkpoint.
        session_state = await executor.ralph_seed_session(task.instruction)

        window_task = Task(
            id=task.id,
            instruction=task.instruction,
            session_id=window_session_id,
            budget=task.budget,
            loop_strategy=inner_for_window,
        )
        cx.scratch.task = window_task
        cx.scratch.run_session = session_state
        # FRESH per-window budget (the reset discards the turn budget).
        cx.scratch.run_budget = BudgetSnapshot()
        window_outcome = await run_strategy(inner_for_window, cx)
        # #125 rule 4/7: a window child's BudgetExhausted reaches Ralph as a
        # StrategyOutcome, never auto-cascaded. Ralph's recovery semantics: a
        # budget-exhausted window is treated as "window incomplete" — RESET the
        # context window and retry (next outer iteration). After ``max_resets``
        # this falls through to ``RalphCompletionUnmet``. Ralph's own scope is
        # unaffected.
        if isinstance(window_outcome, StrategyOutcomeBudgetExhausted):
            partial = window_outcome.partial_output or "<no partial>"
            last_reason = f"window {iteration + 1} budget-exhausted: {partial}"
            continue
        window_result = cx._take_child_override()
        if window_result is None:
            window_result = RunResultFailure(
                reason=HaltReasonRalphCompletionUnmet(
                    iterations=iteration + 1,
                    last_reason="window sub-strategy produced no terminal",
                ),
                session_id=window_session_id,
                usage=AggregateUsage(),
                turns=0,
                session_state=SessionState(),
            )
        window_budget = BudgetSnapshot()
        _fold_usage(total_usage, window_budget, window_result)
        cumulative_turns += window_budget.turns

        # A paused / escalated window propagates verbatim.
        if isinstance(
            window_result, RunResultWaitingForHuman | RunResultConsult | RunResultEscalate
        ):
            cx.stream = on_stream
            return await cx._finish(executor, task, window_result)

        reason = await executor.ralph_completion_status()
        if reason is None:
            if isinstance(window_result, RunResultSuccess):
                output = window_result.output
                final_state = window_result.session_state
            else:
                output = ""
                final_state = SessionState()
            cx.stream = on_stream
            result: RunResult = RunResultSuccess(
                output=output,
                session_id=window_session_id,
                usage=total_usage,
                turns=cumulative_turns,
                session_state=final_state,
            )
            return await cx._finish(executor, task, result)
        last_reason = reason

    cx.stream = on_stream
    result = RunResultFailure(
        reason=HaltReasonRalphCompletionUnmet(
            iterations=max_resets,
            last_reason=last_reason,
        ),
        session_id=last_session_id,
        usage=total_usage,
        turns=cumulative_turns,
        session_state=SessionState(),
    )
    return await cx._finish(executor, task, result)


async def _run_hill_climbing_config(
    self: HillClimbingConfig, cx: ExecutionContext
) -> StrategyOutcome:
    """HillClimbing (#124): GENUINELY recursive optimization loop. Iteration 0 is
    a pure baseline (no agent turn). Iterations 1.. recurse
    ``run_strategy(self.inner, cx)`` to propose a change (a non-ReAct inner —
    e.g. PlanExecute — really runs its whole loop per iteration), then evaluate
    the metric (resolved via ``resolve_metric_evaluator``, Q2) and keep/revert.
    Bounded by ``max_stagnation`` and the turn budget. Discards the incoming
    session state (each iteration seeds its own fresh window)."""
    executor = cx._require_executor()
    if isinstance(executor, StrategyOutcomeFailed):
        return executor
    task = cx._current_task()
    session_id = task.session_id
    task_id = task.id
    on_stream = cx.stream
    cx.stream = None
    carried = cx.scratch.run_budget.model_copy(deep=True)
    cx.scratch.run_session = SessionState()
    direction = self.direction
    revert = self.revert_on_no_improvement
    min_delta = self.min_improvement_delta
    # ``2**31 - 1`` sentinel ⇒ no stagnation cap.
    max_stagnation = self.max_stagnation if self.max_stagnation != 2**31 - 1 else None

    # Q2: resolve the metric evaluator from ``evaluator``'s key.
    evaluator = executor.resolve_metric_evaluator(self.evaluator, session_id)
    if isinstance(evaluator, RunResultFailure):
        cx.stream = on_stream
        return await cx._finish(executor, task, evaluator)

    total_usage = AggregateUsage()
    rows: list[Any] = []
    span_seq = [0]

    # #125: HillClimbing owns a budget scope for its optimization loop. POLICY is
    # the task's global turn ceiling (``TotalSteps``); this REPLACES the ad-hoc
    # ``turn_cap`` / ``carried.turns >= turn_cap`` gate that #124 used. Behavior
    # is ``Escalate`` (in-process placeholder; #129).
    if task.budget.max_turns is not None:
        hc_policy: BudgetPolicy = BudgetPolicyTotalSteps(value=task.budget.max_turns)
    else:
        hc_policy = BudgetPolicyUnlimited()
    cx.push_budget(hc_policy, self.behavior, "hill_climbing")

    # ── Iteration 0: pure baseline (no agent turn).
    baseline = await executor.hill_baseline(
        evaluator,
        session_id,
        task_id,
        direction,
        rows,
        span_seq,
        total_usage,
        carried.turns,
    )
    if isinstance(baseline, RunResultFailure):
        cx.pop_budget()
        cx.stream = on_stream
        return await cx._finish(executor, task, baseline)
    current_best = baseline

    stagnation = 0
    iteration = 1
    started_at = time.monotonic()

    while True:
        # #125: charge-based budget gate before the iteration's agent turn. A
        # spent ``TotalSteps`` cap surfaces this node's OWN typed BudgetExhausted
        # (partial = best candidate + score), resolving its behavior — replacing
        # the legacy ``BudgetExceeded`` Failure.
        charge_err = cx.charge_current(1)
        if charge_err is not None:
            resolution = cx.resolve_current()
            # #129: a granted ``Continue`` resets the scope and KEEPS ITERATING the
            # climb IN-PROCESS (do NOT pop; the refreshed allowance lets the next
            # charge pass). NO serialization (AC3). ``max_continues`` bounds the
            # loop.
            if resolution == ExhaustedResolution.CONTINUE:
                continue
            # SC-5: AutoContinue auto-grants in-process at this Escalate point
            # (scope still pushed) before surfacing/aborting.
            if resolution == ExhaustedResolution.ESCALATE and cx.try_auto_continue(
                executor.escalation_mode()
            ):
                continue
            await executor.hill_write_tsv(task_id, rows)
            partial = _hill_climbing_partial_json(current_best)
            cx.pop_budget()
            cx.scratch.task = task
            cx.stream = on_stream
            if resolution == ExhaustedResolution.FAIL:
                return _promote_budget_exhausted(charge_err, None)
            # #129: the in-process ``Continue`` is handled above; this arm is
            # ``Escalate``-only now (kept for the surface/propagate shape).
            if resolution == ExhaustedResolution.ESCALATE:
                from .execution_registry import EscalationModeSurfaceToHuman

                if isinstance(executor.escalation_mode(), EscalationModeSurfaceToHuman):
                    # #138 AC2-a: HillClimbing iterates on a METRIC, not a worker
                    # conversation, so there is no richer session to carry — pass
                    # the empty default to fall back to the partial-only stub
                    # (pre-#138 behavior). The worker leaf's toolset handle is
                    # still carried (#140 parity, AC4-a).
                    waiting = _promote_budget_exhausted_to_human(
                        charge_err,
                        partial,
                        _combinator_escalation_actions(charge_err),
                        session_id,
                        task,
                        carried.model_copy(deep=True),
                        carried.turns,
                        SessionState(),
                        _worker_toolset_of(self.inner),
                    )
                    return cx._record_terminal(waiting)
                return _promote_budget_exhausted(charge_err, partial)
            # Defensive: ``ExhaustedResolution`` is exhausted above.
            raise AssertionError(f"unhandled resolution {resolution!r}")
        limit_type = executor.budget_exceeded(task.budget, carried, started_at)
        if limit_type is not None:
            await executor.hill_write_tsv(task_id, rows)
            cx.pop_budget()
            cx.stream = on_stream
            result: RunResult = RunResultFailure(
                reason=HaltReasonBudgetExceeded(limit_type=limit_type),
                session_id=session_id,
                usage=total_usage,
                turns=carried.turns,
                session_state=SessionState(),
            )
            return await cx._finish(executor, task, result)

        # ── One agent turn proposes a change: recurse ``self.inner``.
        iter_task = Task(
            id=task.id,
            instruction=task.instruction,
            session_id=session_id,
            budget=task.budget,
            loop_strategy=self.inner,
        )
        cx.scratch.task = iter_task
        iter_state = SessionState()
        await executor.append_user_message(iter_state, task.instruction)
        cx.scratch.run_session = iter_state
        cx.scratch.run_budget = carried.model_copy(deep=True)
        iter_outcome = await run_strategy(self.inner, cx)
        # #125 rule 4/7: a child's BudgetExhausted reaches HillClimbing as a
        # StrategyOutcome, never auto-cascaded. Surface this node's own typed
        # BudgetExhausted (partial = best candidate + score).
        if isinstance(iter_outcome, StrategyOutcomeBudgetExhausted):
            await executor.hill_write_tsv(task_id, rows)
            partial = _hill_climbing_partial_json(current_best)
            scope = cx.budgets.current()
            assert scope is not None, "hill_climbing scope pushed above"
            err = BudgetExhausted(
                policy=scope.policy,
                behavior=scope.behavior,
                steps_taken=scope.steps_taken,
                continues_used=scope.continues_used,
                phase=scope.phase,
            )
            cx.pop_budget()
            cx.scratch.task = task
            cx.stream = on_stream
            return _promote_budget_exhausted(err, partial)
        turn_result = cx._take_child_override()
        if turn_result is None:
            turn_result = RunResultFailure(
                reason=HaltReasonBudgetExceeded(limit_type="turns"),
                session_id=session_id,
                usage=AggregateUsage(),
                turns=carried.turns,
                session_state=SessionState(),
            )
        _fold_usage(total_usage, carried, turn_result)

        # A paused / escalated turn propagates verbatim.
        if isinstance(turn_result, RunResultWaitingForHuman | RunResultConsult | RunResultEscalate):
            await executor.hill_write_tsv(task_id, rows)
            cx.pop_budget()
            cx.stream = on_stream
            return await cx._finish(executor, task, turn_result)

        # ── Evaluate the metric + keep/revert decision.
        current_best, non_improvement = await executor.hill_iteration(
            evaluator,
            session_id,
            task_id,
            iteration,
            direction,
            revert,
            min_delta,
            current_best,
            rows,
            span_seq,
        )
        stagnation = stagnation + 1 if non_improvement else 0

        if max_stagnation is not None and stagnation >= max_stagnation:
            await executor.hill_write_tsv(task_id, rows)
            cx.pop_budget()
            cx.stream = on_stream
            result = RunResultFailure(
                reason=HaltReasonStagnationLimitReached(
                    iterations=stagnation,
                    best_metric=current_best,
                ),
                session_id=session_id,
                usage=total_usage,
                turns=carried.turns,
                session_state=SessionState(),
            )
            return await cx._finish(executor, task, result)

        iteration += 1


def _fold_usage(total_usage: AggregateUsage, carried: BudgetSnapshot, r: RunResult) -> None:
    """Fold a sub-run's token usage / turn count into the cumulative
    ``total_usage`` and the shared ``carried`` budget snapshot (#124, R8).
    ``carried.turns`` becomes the running MAX of the sub-runs' absolute turn
    counts. A non-terminal pause (``WaitingForHuman`` / ``Consult``) carries no
    fold. Mirrors Rust's standalone ``fold_usage``."""
    if isinstance(r, RunResultWaitingForHuman | RunResultConsult):
        return
    usage = r.usage
    turns = r.turns
    total_usage.input_tokens += usage.input_tokens
    total_usage.output_tokens += usage.output_tokens
    total_usage.cache_read_tokens += usage.cache_read_tokens
    total_usage.cache_write_tokens += usage.cache_write_tokens
    total_usage.cost_usd += usage.cost_usd
    carried.input_tokens += usage.input_tokens
    carried.output_tokens += usage.output_tokens
    carried.turns = max(carried.turns, turns)


def _worker_agent_key_of(ls: LoopStrategy) -> str:
    """Descend a :data:`LoopStrategy` tree to the worker leaf's agent key (#124).
    The worker is the agent on the LEAF reached by descending the recursion: a
    ``ReAct`` leaf's ``agent``; a combinator descends into its primary worker
    child (``inner`` / ``execute``). A ``Ralph`` with a non-empty ``agent``
    override resolves THAT (Q3). Mirrors Rust's ``worker_agent_key_of``."""
    if isinstance(ls, ReactConfig):
        return ls.agent
    if isinstance(ls, PlanExecuteConfig):
        return _worker_agent_key_of(ls.execute)
    if isinstance(ls, SelfVerifyingConfig):
        return _worker_agent_key_of(ls.inner)
    if isinstance(ls, RalphConfig):
        return ls.agent if ls.agent else _worker_agent_key_of(ls.inner)
    if isinstance(ls, HillClimbingConfig):
        return _worker_agent_key_of(ls.inner)
    raise AssertionError(f"unknown loop strategy: {ls!r}")  # pragma: no cover


def _worker_toolset_of(ls: LoopStrategy) -> ToolsetRef:
    """Descend a :data:`LoopStrategy` tree to the worker (execute) leaf's toolset
    handle (#138 AC4-a). Mirrors :func:`_worker_agent_key_of`: a combinator
    descends into its EXECUTE child (PlanExecute) / inner (SelfVerifying / Ralph /
    HillClimbing) so a budget-exhausted pause records the same handle #140 would
    have on the leaf's own pause (e.g. ``"exec-tools"``), not the empty default.
    Mirrors Rust's ``worker_toolset_of``."""
    if isinstance(ls, ReactConfig):
        return ls.toolset
    if isinstance(ls, PlanExecuteConfig):
        return _worker_toolset_of(ls.execute)
    if isinstance(ls, SelfVerifyingConfig):
        return _worker_toolset_of(ls.inner)
    if isinstance(ls, RalphConfig):
        return _worker_toolset_of(ls.inner)
    if isinstance(ls, HillClimbingConfig):
        return _worker_toolset_of(ls.inner)
    raise AssertionError(f"unknown loop strategy: {ls!r}")  # pragma: no cover


def _fill_empty_worker_agent(ls: LoopStrategy, agent: str) -> LoopStrategy:
    """Return a copy of ``ls`` with the worker leaf's agent handle filled with
    ``agent`` ONLY where the leaf left it empty — Ralph's bare-leaf convenience.
    An explicitly-declared leaf agent is authoritative and is never replaced (the
    architect's node wins). Descends the worker child chain to the innermost leaf.
    Mirrors Rust's ``fill_empty_worker_agent``."""
    if isinstance(ls, ReactConfig):
        if not ls.agent:
            return ls.model_copy(update={"agent": agent})
        return ls
    if isinstance(ls, PlanExecuteConfig):
        return ls.model_copy(update={"execute": _fill_empty_worker_agent(ls.execute, agent)})
    if isinstance(ls, SelfVerifyingConfig):
        return ls.model_copy(update={"inner": _fill_empty_worker_agent(ls.inner, agent)})
    if isinstance(ls, RalphConfig):
        return ls.model_copy(update={"inner": _fill_empty_worker_agent(ls.inner, agent)})
    if isinstance(ls, HillClimbingConfig):
        return ls.model_copy(update={"inner": _fill_empty_worker_agent(ls.inner, agent)})
    raise AssertionError(f"unknown loop strategy: {ls!r}")  # pragma: no cover


async def run_strategy(strategy: LoopStrategy, cx: ExecutionContext) -> StrategyOutcome:
    """The ONLY ``match`` site in the strategy system — one-line delegation per
    arm to the inner config's per-variant ``run`` body (#124)."""
    match strategy:
        case ReactConfig():
            return await _run_react_config(strategy, cx)
        case PlanExecuteConfig():
            return await _run_plan_execute_config(strategy, cx)
        case SelfVerifyingConfig():
            return await _run_self_verifying_config(strategy, cx)
        case RalphConfig():
            return await _run_ralph_config(strategy, cx)
        case HillClimbingConfig():
            return await _run_hill_climbing_config(strategy, cx)


class Task(_Model):
    id: TaskId
    instruction: str
    session_id: SessionId
    budget: BudgetLimits = Field(default_factory=BudgetLimits)
    loop_strategy: LoopStrategy

    @classmethod
    def new(
        cls,
        instruction: str,
        session_id: SessionId,
        loop_strategy: LoopStrategy,
        *,
        budget: BudgetLimits | None = None,
    ) -> Task:
        return cls(
            id=new_task_id(),
            instruction=instruction,
            session_id=session_id,
            budget=budget or BudgetLimits(),
            loop_strategy=loop_strategy,
        )

    @classmethod
    def simple(cls, instruction: str) -> Task:
        """A one-shot task from just an instruction: a fresh :class:`SessionId`
        and a default ``ReAct`` loop (``max_iterations=8``). Use :meth:`new`
        when you need to control the session id (e.g. multi-turn) or the loop
        strategy. Mirrors Rust's ``Task::simple``."""
        return cls.new(
            instruction,
            new_session_id(),
            ReactConfig.per_loop(8),
        )


# ============================================================================
# Stream events
# ============================================================================


class BlockKind(str, Enum):
    """The kind of content block a :class:`StreamBlockStart` opens (issue #103,
    Q2: a single generic frame marker carrying a ``BlockKind`` rather than
    typed-per-kind markers)."""

    TEXT = "text"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"


class StreamTurnStart(_Model):
    kind: Literal["turn_start"] = "turn_start"
    turn: int


class StreamTurnEnd(_Model):
    kind: Literal["turn_end"] = "turn_end"
    turn: int


class StreamToolCall(_Model):
    kind: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    # Final, fully-accumulated tool-call arguments (issue #103, Q5). Defaults
    # to ``{}`` so pre-#103 serialized events round-trip (back-compat).
    args: dict[str, Any] = Field(default_factory=dict)


class StreamToolResult(_Model):
    kind: Literal["tool_result"] = "tool_result"
    call_id: str
    is_error: bool
    # The tool result content (issue #103, Q5). Defaults to ``""`` so pre-#103
    # serialized events round-trip (back-compat).
    content: str = ""


class StreamFinalResponse(_Model):
    kind: Literal["final_response"] = "final_response"
    content: str


class StreamBudgetWarning(_Model):
    kind: Literal["budget_warning"] = "budget_warning"
    limit_type: BudgetLimitTypeT


class StreamUserMessage(_Model):
    """Out-of-band, prominent message to the user (issue #81). Emitted by the
    loop when the ``send_message`` tool runs, INSTEAD of collapsing the content
    into a normal tool result. The harness only emits the event — rendering it
    prominently is the architect's UI concern."""

    kind: Literal["user_message"] = "user_message"
    content: str


# ── Delta-level streaming (issue #103) ──────────────────────────────────────
#
# The harness maps each raw :class:`spore_core.model.StreamEvent` produced by
# the agent through :meth:`StandardHarness._map_model_stream_event` into zero or
# more of the delta/frame variants below, alongside the coarse lifecycle events.
# Resolution notes for the spec ambiguities:
#
# * Q2: frame markers are the generic :class:`StreamBlockStart` /
#   :class:`StreamBlockStop` carrying a :class:`BlockKind`.
# * Q3: ``model`` ``MessageStart`` / ``MessageStop`` are DROPPED at the harness
#   boundary (mapped to nothing). ``TurnStart`` / ``TurnEnd`` already cover
#   message lifecycle.
# * Q5: the coarse :class:`StreamToolCall` also carries the final ``args`` and
#   :class:`StreamToolResult` the result ``content`` (both defaulted).
#
# Tool lifecycle ordering per call:
# :class:`StreamToolCallStart` -> :class:`StreamToolArgsDelta`* ->
# (:class:`StreamBlockStop`) -> coarse :class:`StreamToolCall`.
#
# Tool name + id are recovered from ``model.StreamEvent`` ``ToolUseStart``, which
# every provider emits at the tool block's start frame (Anthropic
# ``content_block_start``, Ollama / OpenAI's first ``tool_calls`` chunk) before
# the ``ToolUseDelta`` argument fragments arrive. The harness records them and
# emits the real id + name on ``StreamToolCallStart``. The ``ToolUseDelta`` path
# keeps a fallback (stable per-index ``call_{index}`` id + empty name) only for
# a stream that somehow omitted the start frame.


class StreamTextDelta(_Model):
    """Streamed text fragment (``model`` ``ContentBlockDelta``)."""

    kind: Literal["text_delta"] = "text_delta"
    content: str


class StreamReasoningDelta(_Model):
    """Streamed reasoning/thinking fragment (``model`` ``ThinkingDelta``). Q4."""

    kind: Literal["reasoning_delta"] = "reasoning_delta"
    content: str


class StreamToolArgsDelta(_Model):
    """Streamed tool-argument JSON fragment (``model`` ``ToolUseDelta``),
    correlated to a ``call_id`` via the open-block index."""

    kind: Literal["tool_args_delta"] = "tool_args_delta"
    call_id: str
    partial_json: str


class StreamBlockStart(_Model):
    """A content block opened (Q2). Emitted the first time a delta for an index
    is seen."""

    kind: Literal["block_start"] = "block_start"
    index: int
    block: BlockKind


class StreamBlockStop(_Model):
    """A content block closed (``model`` ``ContentBlockStop``). Q2."""

    kind: Literal["block_stop"] = "block_stop"
    index: int


class StreamToolCallStart(_Model):
    """A tool-use block opened (issue #103). Emitted so consumers can correlate
    the subsequent :class:`StreamToolArgsDelta` fragments and the final coarse
    :class:`StreamToolCall` by ``call_id``. The ``name`` and ``call_id`` are the
    real values recovered from the ``model.StreamEvent`` ``ToolUseStart`` frame
    every provider emits at the tool block's start."""

    kind: Literal["tool_call_start"] = "tool_call_start"
    index: int
    call_id: str
    name: str


class StreamToolErrorLoopDetected(_Model):
    """The consecutive-recoverable-tool-error breaker DETECTED a loop (issue
    #137): ``tool`` hit ``error_loop_threshold`` (``N``) consecutive
    identical-argument recoverable errors and ONE corrective message was
    injected. A warning — the loop continues. Carries the tool name and the
    consecutive-error count (``= N``). Mirrors Rust's
    ``StreamEvent::ToolErrorLoopDetected``."""

    kind: Literal["tool_error_loop_detected"] = "tool_error_loop_detected"
    tool: str
    consecutive_errors: int


class StreamToolErrorLoopBroken(_Model):
    """The consecutive-recoverable-tool-error breaker TRIPPED (issue #137):
    ``tool`` reached ``2 * error_loop_threshold`` (``2 * N``) consecutive
    identical-argument recoverable errors and the loop STOPPED to resolve the
    node's ``BudgetExhaustedBehavior``. Carries the tool name and the count
    (``= 2*N``). Mirrors Rust's ``StreamEvent::ToolErrorLoopBroken``."""

    kind: Literal["tool_error_loop_broken"] = "tool_error_loop_broken"
    tool: str
    consecutive_errors: int


class StreamOutputSchemaRetry(_Model):
    """Output-schema enforcement (issue #139) fed a validation error back and
    RETRIED: the terminal ``FinalResponse`` failed validation against the leaf's
    ``output`` schema and a retry turn was granted (within budget). A warning —
    the loop continues. Carries the extra-retry count so far (``= attempt``) and
    the frozen validator error fed back. Mirrors Rust's
    ``StreamEvent::OutputSchemaRetry``."""

    kind: Literal["output_schema_retry"] = "output_schema_retry"
    attempt: int
    error: str


class StreamOutputSchemaViolation(_Model):
    """Output-schema enforcement (issue #139) EXHAUSTED its retries: the terminal
    still failed validation after ``output_schema_max_retries`` extra turns (with
    budget remaining), so the run terminates with
    :class:`HaltReasonOutputSchemaViolation`. Carries the total attempt count
    (``= 1 + max_retries``) and the final frozen validator error. Mirrors Rust's
    ``StreamEvent::OutputSchemaViolation``."""

    kind: Literal["output_schema_violation"] = "output_schema_violation"
    attempts: int
    error: str


HarnessStreamEvent = Annotated[
    StreamTurnStart
    | StreamTurnEnd
    | StreamToolCall
    | StreamToolResult
    | StreamFinalResponse
    | StreamBudgetWarning
    | StreamUserMessage
    | StreamTextDelta
    | StreamReasoningDelta
    | StreamToolArgsDelta
    | StreamBlockStart
    | StreamBlockStop
    | StreamToolCallStart
    | StreamToolErrorLoopDetected
    | StreamToolErrorLoopBroken
    | StreamOutputSchemaRetry
    | StreamOutputSchemaViolation,
    Field(discriminator="kind"),
]

StreamSink = Callable[[HarnessStreamEvent], None]


class TurnStreamState:
    """Per-turn state threaded through
    :meth:`StandardHarness._map_model_stream_event` (issue #103). Tracks which
    block indices are open and their kind so ``ToolUseDelta`` events can be
    correlated to a ``call_id``, and so each block's ``BlockStart`` is emitted
    exactly once."""

    def __init__(self) -> None:
        self.open_blocks: dict[int, BlockKind] = {}
        self.tool_calls: dict[int, str] = {}

    @staticmethod
    def call_id_for(index: int) -> str:
        # Matches the id synthesized by the agent's streaming accumulator so
        # the coarse ``StreamToolCall`` correlates.
        return f"call_{index}"


# ============================================================================
# Forward-declared sibling types
# ============================================================================


class SessionState(_Model):
    """Opaque session state round-tripped across pause/resume.

    The harness does not interpret its contents; #7 (ContextManager) and
    #8 (MemoryProvider) own the schema.
    """

    messages: list[Message] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)


# ----- SandboxViolation (issue #6) -----------------------------------------


class SandboxPathEscape(_Model):
    kind: Literal["path_escape"] = "path_escape"
    path: str


class SandboxNetworkViolation(_Model):
    kind: Literal["network_violation"] = "network_violation"
    host: str


class SandboxPathDenied(_Model):
    kind: Literal["path_denied"] = "path_denied"
    path: str
    matched_rule: str = ""


class SandboxReadOnlyViolation(_Model):
    kind: Literal["read_only_violation"] = "read_only_violation"
    path: str


class SandboxExtensionDenied(_Model):
    kind: Literal["extension_denied"] = "extension_denied"
    path: str
    extension: str


class SandboxFileSizeExceeded(_Model):
    kind: Literal["file_size_exceeded"] = "file_size_exceeded"
    path: str
    size: int
    limit: int


class SandboxDisallowedCommand(_Model):
    kind: Literal["disallowed_command"] = "disallowed_command"
    command: str


SandboxViolation = Annotated[
    SandboxPathEscape
    | SandboxNetworkViolation
    | SandboxPathDenied
    | SandboxReadOnlyViolation
    | SandboxExtensionDenied
    | SandboxFileSizeExceeded
    | SandboxDisallowedCommand,
    Field(discriminator="kind"),
]


def sandbox_violation_is_always_halt(v: SandboxViolation) -> bool:
    """Layer-1 violations cannot be overridden — they always halt."""
    return isinstance(v, SandboxPathEscape | SandboxNetworkViolation)


# ----- HookPoint (issue #11) ----------------------------------------------
# Re-exported from :mod:`spore_core.middleware`. We use a string alias here
# because :class:`HaltReasonMiddlewareHalt` and :class:`ScriptedMiddleware`
# round-trip these values as strings on the wire.

HookPoint = Literal[
    "before_session",
    "before_turn",
    "before_tool",
    "after_tool",
    "before_completion",
    "after_session",
]


# ----- TerminationDecision (issue #13) ------------------------------------


class TerminationContinue(_Model):
    kind: Literal["continue"] = "continue"


class TerminationHalt(_Model):
    kind: Literal["halt"] = "halt"
    reason: str


TerminationDecision = Annotated[
    TerminationContinue | TerminationHalt,
    Field(discriminator="kind"),
]


# ----- Human-in-the-loop --------------------------------------------------


RiskLevel = Literal["low", "medium", "high", "critical"]


# ----- Escalation actions (issue #130) ------------------------------------
#
# ``EscalationAction`` is the operator's choice on a
# :class:`HumanRequestBudgetExhausted` pause: grant more budget, skip the node,
# or fail. Internally tagged on ``kind`` (snake_case), byte-identical with the
# Rust / TS / Go definitions. ``ContinueWithBudget`` carries a NAMED ``steps``
# field (not positional) — internally-tagged unions cannot carry a tuple payload
# (mirrors :class:`BudgetPolicyTotalSteps` ``{ value }``).


class EscalationActionContinueWithBudget(_Model):
    kind: Literal["continue_with_budget"] = "continue_with_budget"
    steps: int


class EscalationActionSkip(_Model):
    kind: Literal["skip"] = "skip"


class EscalationActionFail(_Model):
    kind: Literal["fail"] = "fail"


EscalationAction = Annotated[
    EscalationActionContinueWithBudget | EscalationActionSkip | EscalationActionFail,
    Field(discriminator="kind"),
]


class HumanRequestToolApproval(_Model):
    kind: Literal["tool_approval"] = "tool_approval"
    calls: list[ToolCall]
    risk_level: RiskLevel


class HumanRequestClarification(_Model):
    kind: Literal["clarification"] = "clarification"
    question: str
    # Issue #81 (Q4b): an ``ask_user_question`` clarification may carry a fixed
    # set of multiple-choice options. Optional / defaulted so older
    # ``Clarification`` blobs (no ``options`` field) still deserialize —
    # back-compat with the bare pre-#81 shape.
    options: list[str] | None = None


class HumanRequestReview(_Model):
    kind: Literal["review"] = "review"
    content: str


class HumanRequestBudgetExhausted(_Model):
    """A node's budget scope resolved to ``Escalate`` under
    :class:`EscalationModeSurfaceToHuman` (#130): the run pauses and surfaces the
    exhaustion to the operator. Carries the node's ``phase``, its ``policy``, the
    ``steps_taken`` / ``continues_used`` counters (so ``resume`` can reconstruct
    the node's budget context — fork E), any ``partial_output`` produced before
    exhaustion, and the ADVISORY ``available_actions`` the author offers (fork
    C/D). The operator answers with :class:`HumanResponseEscalate`.

    ``partial_output`` is serialized as ``null`` when absent (mirrors the Rust
    ``Option<String>`` with no skip-serializing) — NOT omitted."""

    kind: Literal["budget_exhausted"] = "budget_exhausted"
    phase: str
    policy: BudgetPolicy
    steps_taken: int
    continues_used: int
    partial_output: str | None = None
    available_actions: list[EscalationAction] = Field(default_factory=list)


HumanRequest = Annotated[
    HumanRequestToolApproval
    | HumanRequestClarification
    | HumanRequestReview
    | HumanRequestBudgetExhausted,
    Field(discriminator="kind"),
]


class HumanResponseAllow(_Model):
    kind: Literal["allow"] = "allow"


class HumanResponseAllowWithModification(_Model):
    kind: Literal["allow_with_modification"] = "allow_with_modification"
    calls: list[ToolCall]


class HumanResponseDeny(_Model):
    kind: Literal["deny"] = "deny"
    reason: str


class HumanResponseHalt(_Model):
    kind: Literal["halt"] = "halt"


class HumanResponseAnswer(_Model):
    kind: Literal["answer"] = "answer"
    text: str


class HumanResponseApproveWithFeedback(_Model):
    kind: Literal["approve_with_feedback"] = "approve_with_feedback"
    feedback: str


class HumanResponseReject(_Model):
    kind: Literal["reject"] = "reject"
    reason: str


class HumanResponseEscalate(_Model):
    """The operator's resolution of a :class:`HumanRequestBudgetExhausted` pause
    (#130): the chosen :class:`EscalationAction`. Distinct from
    ``Allow`` / ``Halt`` / ``Deny`` so the budget-escalation resume path is
    unambiguous."""

    kind: Literal["escalate"] = "escalate"
    action: EscalationAction


HumanResponse = Annotated[
    HumanResponseAllow
    | HumanResponseAllowWithModification
    | HumanResponseDeny
    | HumanResponseHalt
    | HumanResponseAnswer
    | HumanResponseApproveWithFeedback
    | HumanResponseReject
    | HumanResponseEscalate,
    Field(discriminator="kind"),
]


# ----- ToolOutput / ToolResult (issue #4/#5) ------------------------------


class ToolOutputSuccess(_Model):
    """A successful tool result.

    ``truncated`` is ``True`` only when the tool itself clipped its output to
    fit an inline budget (large outputs routed through
    :meth:`SandboxProvider.handle_large_output` set this). Plain tool authors
    should leave it ``False`` — use :meth:`success`.
    """

    kind: Literal["success"] = "success"
    content: str
    truncated: bool = False

    @classmethod
    def success(cls, content: str) -> ToolOutputSuccess:
        """A successful, non-truncated result. The common case for a tool that
        returns its full output — saves spelling out ``truncated=False``."""
        return cls(content=content, truncated=False)


class ToolOutputError(_Model):
    """A failed tool result.

    ``recoverable`` is ``True`` if the agent may sensibly retry or adapt: the
    loop appends the error as a tool result and continues. ``False`` halts the
    run. Most tool failures are recoverable — prefer :meth:`error`; reach for
    :meth:`fatal` only when continuing is pointless.
    """

    kind: Literal["error"] = "error"
    message: str
    recoverable: bool = True

    @classmethod
    def error(cls, message: str) -> ToolOutputError:
        """A **recoverable** error: the harness loop appends it as a tool result
        and lets the agent adapt or retry. This is the right default for almost
        every tool failure (bad arguments, missing file, transient I/O)."""
        return cls(message=message, recoverable=True)

    @classmethod
    def fatal(cls, message: str) -> ToolOutputError:
        """A **fatal** error: continuing is pointless, so the run halts. Reserve
        for genuinely unrecoverable conditions; prefer :meth:`error` when the
        agent could reasonably do something different next turn."""
        return cls(message=message, recoverable=False)


class ToolOutputWaitingForHuman(_Model):
    kind: Literal["waiting_for_human"] = "waiting_for_human"
    child_state: ChildPausedState
    request: HumanRequest


# ----- HarnessSignal (issue #80, Tool Escalation Protocol) ----------------
# The typed channel by which a tool signals the harness to terminate cleanly
# and pass a structured signal up to its caller. The harness never acts on the
# signal itself — it is a pure intermediary. Tagged union over ``kind`` with
# snake_case discriminators (``enter_plan_mode``, ``exit_plan_mode``,
# ``switch_mode``, ``abort``) byte-identical across the four languages.
#
# ``ExitPlanMode`` carries a :class:`PlanArtifact` (defined in
# :mod:`spore_core.hooks`) and ``SwitchMode`` carries a :class:`Mode` (defined
# in :mod:`spore_core.prompt_chunk_registry`). Both of those modules import from
# this module, so the two types are pulled in via the deferred import block at
# the bottom and these models are rebuilt there (forward references).


class HarnessSignalEnterPlanMode(_Model):
    """Agent requests entry into plan mode. Carries the context the agent has
    accumulated so far as a seed for the planning harness."""

    kind: Literal["enter_plan_mode"] = "enter_plan_mode"
    context: str


class HarnessSignalExitPlanMode(_Model):
    """Planning agent has produced a plan and requests exit from plan mode.
    Carries the plan artifact for human approval. The planning agent's terminal
    signal."""

    kind: Literal["exit_plan_mode"] = "exit_plan_mode"
    plan: PlanArtifact


class HarnessSignalSwitchMode(_Model):
    """Agent requests a mode switch. The caller instantiates the appropriate
    harness for the new mode."""

    kind: Literal["switch_mode"] = "switch_mode"
    mode: Mode


class HarnessSignalAbort(_Model):
    """Agent requests a graceful, intentional stop with a reason surfaced to the
    user. Distinct from :class:`HaltReasonAgentError` — it surfaces as
    :class:`RunResultEscalate`, NOT :class:`RunResultFailure`."""

    kind: Literal["abort"] = "abort"
    reason: str


HarnessSignal = Annotated[
    HarnessSignalEnterPlanMode
    | HarnessSignalExitPlanMode
    | HarnessSignalSwitchMode
    | HarnessSignalAbort,
    Field(discriminator="kind"),
]


class ToolOutputEscalate(_Model):
    """Tool requests a structural state change from the harness's parent. The
    harness loop recognizes this variant, terminates the current run cleanly,
    and passes the signal up to the caller via :class:`RunResultEscalate`."""

    kind: Literal["escalate"] = "escalate"
    signal: HarnessSignal


class ToolOutputAwaitingClarification(_Model):
    """Tool needs a human answer before it can produce a result (issue #81,
    Q4b). UNLIKE :class:`ToolOutputWaitingForHuman` (the subagent-shaped pause
    that carries a :class:`ChildPausedState`), this variant carries NO child
    state: the loop pauses by building a :class:`PausedState` directly with
    ``human_request`` set to :class:`HumanRequestClarification`. On resume the
    human's answer is injected as the clarifying call's tool result."""

    kind: Literal["awaiting_clarification"] = "awaiting_clarification"
    question: str
    options: list[str] | None = None


# ----- Mid-loop consult primitive (issue #114) ----------------------------
# This is the doc-hub for the consult feature on the Python side. See the Rust
# reference (`rust/crates/spore-core/src/harness.rs`) for the authoritative
# narrative; the type / interface / rule contract is identical here.
#
# Worker side:
#   - A worker-side tool returns :class:`ToolOutputConsult` (built via
#     :meth:`ToolOutputConsult.consult`, ``child_state=None``) to ask for
#     mid-loop help. The loop pauses, preserving the consult call as the HEAD of
#     ``pending_tool_calls`` (``human_request=None``), and returns
#     :class:`RunResultConsult` (R1, R10) — a sibling of
#     :class:`RunResultWaitingForHuman`. The consult is NEVER appended to message
#     history until the resume injects the answer as its tool result (R10).
#
# Resume side:
#   - :meth:`StandardHarness.resume_consult` injects the
#     :class:`ConsultResponse` text as the tool RESULT of the head pending
#     (consult) call, dispatches the remaining batch, and resumes the ReAct loop.
#
# Orchestrator mediation (seam A1) lives in ``spore_tools.tools.subagent``:
#   - ``SubagentTool`` drives the full run -> Consult -> route-by-kind ->
#     budget-check -> run-handler -> resume loop INTERNALLY, returning a final
#     success to the parent (R2/R3). The parent model never sees the consult.
#     Depth-1 is preserved: the handler is the orchestrator's direct child (R7).


class ConsultRequest(_Model):
    """The worker's free-form ask when it pauses mid-loop to consult a
    parent-spawned helper (issue #114). ``kind`` selects the handler;
    ``situation`` / ``attempts`` / ``question`` carry the free-form context the
    handler needs. All fields are REQUIRED — there are deliberately no defaults,
    so a malformed / partial request fails to deserialize rather than silently
    defaulting (matches the Rust ``ConsultRequest``)."""

    kind: str
    situation: str
    attempts: int
    question: str


class ConsultResponseAnswer(_Model):
    """The handler produced an answer; ``text`` is injected as the tool RESULT
    for the pending consult call on resume."""

    kind: Literal["answer"] = "answer"
    text: str


class ConsultResponseBudgetExhausted(_Model):
    """The per-kind budget is exhausted under a ``SoftFail`` overflow policy: the
    worker is resumed with this message and finishes with what it has."""

    kind: Literal["budget_exhausted"] = "budget_exhausted"
    message: str


ConsultResponse = Annotated[
    ConsultResponseAnswer | ConsultResponseBudgetExhausted,
    Field(discriminator="kind"),
]


class ConsultOverflowPolicySoftFail(_Model):
    """Resume the worker with :class:`ConsultResponseBudgetExhausted` so it
    finishes without further help."""

    kind: Literal["soft_fail"] = "soft_fail"


class ConsultOverflowPolicyEscalateToHuman(_Model):
    """Convert the over-budget consult into a :class:`RunResultWaitingForHuman`
    (surfaced from ``SubagentTool`` as :class:`ToolOutputWaitingForHuman`) so the
    host decides."""

    kind: Literal["escalate_to_human"] = "escalate_to_human"


ConsultOverflowPolicy = Annotated[
    ConsultOverflowPolicySoftFail | ConsultOverflowPolicyEscalateToHuman,
    Field(discriminator="kind"),
]


@dataclass
class ConsultHandlerEntry:
    """A registered consult handler: the helper harness to run, the per-kind
    budget (max consults of this kind before overflow), and the overflow policy
    (issue #114). Held by ``kind`` in :attr:`HarnessConfig.consult_handlers`.

    The ``handler`` is run by ``SubagentTool`` as the ORCHESTRATOR's direct child
    (depth-1, R7), never nested under the worker. A plain dataclass — it holds a
    live :class:`Harness`, so it is never serialized."""

    handler: Harness
    budget: int
    overflow: ConsultOverflowPolicy


class ToolOutputConsult(_Model):
    """Mid-loop consult signal (issue #114). A worker-side tool returns it with
    ``child_state=None`` (use :meth:`consult`); the worker harness pauses and
    returns :class:`RunResultConsult` with the consult call preserved as the head
    of ``pending_tool_calls`` (``human_request=None``). At the subagent boundary
    ``SubagentTool`` may populate ``child_state`` — but under the A1 mediation
    seam it consumes the signal itself rather than bubbling it, so a parent
    orchestrator never observes this variant on the happy path. Mirrors
    :class:`ToolOutputWaitingForHuman` (one variant, optional child state)."""

    kind: Literal["consult"] = "consult"
    child_state: ChildPausedState | None = None
    request: ConsultRequest

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Mirror Rust's ``#[serde(default, skip_serializing_if = "Option::is_none")]``
        # on ``child_state``: a worker-side consult (``child_state=None``) OMITS
        # the field entirely so the wire form is byte-identical to the fixture's
        # ``worker_tool_output_cases``. The subagent boundary populates it.
        data = handler(self)
        if self.child_state is None:
            data.pop("child_state", None)
        return data

    @classmethod
    def consult(cls, request: ConsultRequest) -> ToolOutputConsult:
        """A worker-side consult signal: the tool asks for mid-loop help.
        ``child_state`` is ``None`` — the harness loop builds the
        :class:`RunResultConsult` pause; only ``SubagentTool`` populates
        ``child_state`` at the boundary."""
        return cls(child_state=None, request=request)


ToolOutput = Annotated[
    ToolOutputSuccess
    | ToolOutputError
    | ToolOutputWaitingForHuman
    | ToolOutputEscalate
    | ToolOutputAwaitingClarification
    | ToolOutputConsult,
    Field(discriminator="kind"),
]

#: The registered name of the catalogue ``send_message`` tool (issue #81). The
#: harness loop recognizes this name and emits a :class:`StreamUserMessage`
#: event rather than collapsing the tool result. The implementation lives in
#: ``spore_tools`` (``SendMessageTool``); the harness only needs the name.
SEND_MESSAGE_TOOL_NAME = "send_message"


class HarnessToolResult(_Model):
    """Result of dispatching a tool call (harness-side).

    Distinct from :class:`spore_core.model.ToolResult` which is the wire
    content block appended to messages.
    """

    call_id: str
    output: ToolOutput


# ----- MiddlewareDecision (issue #11) -------------------------------------
# The canonical types live in :mod:`spore_core.middleware`. They are
# imported below (at the bottom of this module, after PausedState resolves
# its forward references) and re-exported for ergonomic
# ``from spore_core.harness import MiddlewareHalt`` style imports used by
# the existing harness tests.


# ----- Component protocols (forward declarations) -------------------------


@runtime_checkable
class ToolRegistry(Protocol):
    """Issue #4 — dispatches tool calls."""

    async def dispatch(self, call: ToolCall) -> ToolOutput: ...

    def is_always_halt(self, tool_name: str) -> bool: ...

    def schemas(self) -> list[ToolSchema]: ...


class CommandOutput(_Model):
    """Output of a subprocess executed through the sandbox."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    truncated: bool = False


class FileRef(_Model):
    """Reference to a file holding offloaded tool output."""

    path: str
    byte_len: int


class TruncatedOutput(_Model):
    """Head+tail-truncated output. ``full_ref`` is set when the sandbox
    offloads the original content to a backing file."""

    content: str
    truncated: bool = False
    full_ref: FileRef | None = None
    original_size: int = 0


# ----- Operation / IsolationMode / NetworkPolicy (issue #6) ---------------


Operation = Literal["read", "write", "execute"]


class BwrapProfile(_Model):
    """Bubblewrap profile descriptor. Opaque placeholder in v1."""


class NetworkPolicyNone(_Model):
    kind: Literal["none"] = "none"


class NetworkPolicyAllowlist(_Model):
    kind: Literal["allowlist"] = "allowlist"
    hosts: list[str] = Field(default_factory=list)


class NetworkPolicyFull(_Model):
    kind: Literal["full"] = "full"


NetworkPolicy = Annotated[
    NetworkPolicyNone | NetworkPolicyAllowlist | NetworkPolicyFull,
    Field(discriminator="kind"),
]


class IsolationModeNone(_Model):
    kind: Literal["none"] = "none"


class IsolationModeWorkspaceScoped(_Model):
    kind: Literal["workspace_scoped"] = "workspace_scoped"


class IsolationModeBubblewrap(_Model):
    kind: Literal["bubblewrap"] = "bubblewrap"
    profile: BwrapProfile = Field(default_factory=BwrapProfile)


class IsolationModeDocker(_Model):
    kind: Literal["docker"] = "docker"
    image: str
    network: NetworkPolicy


IsolationMode = Annotated[
    IsolationModeNone
    | IsolationModeWorkspaceScoped
    | IsolationModeBubblewrap
    | IsolationModeDocker,
    Field(discriminator="kind"),
]


class WorkspaceConfig(_Model):
    """Configuration injected at harness construction time.

    Mirrors the Rust ``WorkspaceConfig`` struct. ``allowed_paths`` /
    ``denied_paths`` are relative-or-absolute filesystem paths;
    :class:`WorkspaceScopedSandbox` normalizes them at construction time.
    """

    root: Path
    allowed_paths: list[Path] = Field(default_factory=list)
    denied_paths: list[Path] = Field(default_factory=list)
    allowed_extensions: list[str] | None = None
    denied_extensions: list[str] = Field(default_factory=list)
    read_only: bool = False
    max_file_size: int = 0


@runtime_checkable
class SandboxProvider(Protocol):
    """Issue #6 — validates tool calls against sandbox policy.

    ``resolve_path`` takes an :data:`Operation` so the sandbox can apply
    read-only policy and handle missing-file Write/Execute canonicalization.
    """

    async def validate(self, call: ToolCall) -> SandboxViolation | None: ...

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput: ...

    async def handle_large_output(
        self,
        content: str,
        call_id: str,
        head_tokens: int,
        tail_tokens: int,
    ) -> TruncatedOutput: ...

    async def resolve_path(self, path: str, operation: Operation = "read") -> Path: ...

    def isolation_mode(self) -> IsolationMode: ...

    def workspace_root(self) -> Path: ...


# ----- VcsProvider seam (issue #58 v2) — git-log reload for Ralph ----------


@dataclass
class VcsLogArgs:
    """Parameters shaping a :meth:`VcsProvider.log` read. Each field maps to a
    ``git log`` flag in :class:`GitVcsProvider`:

    - ``max_entries`` → ``-n <N>`` (cap the number of commits returned),
    - ``since_ref``   → ``<ref>..`` (only commits AFTER ``<ref>``),
    - ``format``      → ``--format=<fmt>`` (custom pretty format).

    Mirrors Rust's ``VcsLogArgs``.
    """

    max_entries: int
    since_ref: str | None = None
    format: str | None = None


class VcsError(SporeError):
    """Error raised by a :class:`VcsProvider`. One exception class per
    component, inheriting from the package root :class:`SporeError` — mirrors
    Rust's ``VcsError`` (``CommandFailed`` / ``Sandbox`` variants). The
    ``message`` carries the captured stderr or the sandbox-block detail."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@runtime_checkable
class VcsProvider(Protocol):
    """Read-only VCS abstraction the ``Ralph`` loop strategy uses to reload git
    history between context windows (issue #58 v2, decision B4).

    Mirrors how :class:`SandboxProvider` abstracts filesystem/shell access:
    a Protocol, a real implementation (:class:`GitVcsProvider`), and a
    deterministic fixture double (:class:`FixtureVcsProvider`), injected at
    construction via :meth:`HarnessBuilder.vcs_provider`. ``Ralph`` calls
    :meth:`log` during its reload phase and injects the output into the next
    window's seed as a clearly delimited "Recent VCS history:" section. When NO
    provider is wired (``vcs_provider is None``, the default) the git-log
    section is OMITTED and Ralph behaves exactly like v1. Raises
    :class:`VcsError` on failure. Mirrors Rust's ``VcsProvider`` trait.
    """

    async def log(self, args: VcsLogArgs) -> str:
        """Return the project's commit log, shaped by ``args``, verbatim."""
        ...

    async def status(self) -> str:
        """Return the working-tree status (``git status`` stdout), verbatim."""
        ...


class GitVcsProvider:
    """Real :class:`VcsProvider` that shells out to ``git`` THROUGH a
    :class:`SandboxProvider` (issue #58 v2). It wraps the sandbox and calls
    :meth:`SandboxProvider.execute_command` — it never bypasses sandboxing to
    spawn ``git`` directly. The command line is built from :class:`VcsLogArgs`
    (see that type for the flag mapping); :meth:`status` runs ``git status``.
    All commands run in ``workspace_root``. Mirrors Rust's ``GitVcsProvider``.
    """

    def __init__(self, sandbox: SandboxProvider, workspace_root: str | Path) -> None:
        self._sandbox = sandbox
        self._workspace_root = Path(workspace_root)

    @staticmethod
    def _log_args(args: VcsLogArgs) -> list[str]:
        """Build the ``git log`` argument vector from ``args`` (static so the
        flag mapping is testable independently of process execution)."""
        out = ["log", "-n", str(args.max_entries)]
        if args.format is not None:
            out.append(f"--format={args.format}")
        if args.since_ref is not None:
            out.append(f"{args.since_ref}..")
        return out

    async def _run(self, argv: list[str]) -> str:
        from .sandbox import SandboxViolationException

        try:
            out = await self._sandbox.execute_command("git", argv, self._workspace_root)
        except SandboxViolationException as exc:
            raise VcsError(f"vcs command blocked by sandbox: {exc.violation!r}") from exc
        if out.exit_code != 0:
            raise VcsError(out.stderr)
        return out.stdout

    async def log(self, args: VcsLogArgs) -> str:
        return await self._run(self._log_args(args))

    async def status(self) -> str:
        return await self._run(["status"])


class BaseSandboxProvider:
    """Concrete base class providing default implementations of
    ``execute_command``, ``handle_large_output``, and ``resolve_path``.

    **NOT production-safe** — these defaults spawn processes directly with
    no sandboxing, return paths as-is, and truncate inline without
    offloading. Issue #6 will replace this with a real sandbox.

    Subclasses must still implement :meth:`validate`.
    """

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        try:
            proc = await asyncio.create_subprocess_exec(
                command,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(working_dir) if working_dir is not None else None,
            )
        except (FileNotFoundError, OSError) as e:
            return CommandOutput(
                stdout="",
                stderr=f"spawn failed: {e}",
                exit_code=-1,
                timed_out=False,
            )
        try:
            if timeout is not None:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            else:
                stdout_b, stderr_b = await proc.communicate()
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except (ProcessLookupError, asyncio.CancelledError):
                pass
            secs = int(timeout) if timeout is not None else 0
            return CommandOutput(
                stdout="",
                stderr=f"command timed out after {secs}s",
                exit_code=-1,
                timed_out=True,
            )
        return CommandOutput(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            exit_code=proc.returncode if proc.returncode is not None else -1,
            timed_out=False,
        )

    async def handle_large_output(
        self,
        content: str,
        call_id: str,
        head_tokens: int,
        tail_tokens: int,
    ) -> TruncatedOutput:
        head_chars = max(0, head_tokens) * 4
        tail_chars = max(0, tail_tokens) * 4
        total = len(content)
        original = len(content.encode("utf-8"))
        if total <= head_chars + tail_chars:
            return TruncatedOutput(
                content=content, truncated=False, full_ref=None, original_size=original
            )
        head = content[:head_chars]
        tail = content[total - tail_chars :]
        elided = total - head_chars - tail_chars
        summary = f"{head}\n... [{elided} chars elided] ...\n{tail}"
        return TruncatedOutput(
            content=summary, truncated=True, full_ref=None, original_size=original
        )

    async def resolve_path(self, path: str, operation: Operation = "read") -> Path:
        return Path(path)

    def isolation_mode(self) -> IsolationMode:
        # Safe-by-default (issue #34): the default isolation mode is
        # WorkspaceScoped, never None. No-isolation requires the explicit
        # dangerous opt-in (``from spore_core.dangerous import
        # IsolationModeNone``).
        return IsolationModeWorkspaceScoped()

    def workspace_root(self) -> Path:
        return Path("/")


class ReadOnlySandbox:
    """Read-only :class:`SandboxProvider` decorator (issue #61, R3).

    Wraps an inner provider and blocks the standard mutating tools by name —
    any :class:`ToolCall` whose ``name`` is in :attr:`DEFAULT_WRITE_TOOLS` is
    rejected at :meth:`validate` with :class:`SandboxReadOnlyViolation`; every
    other call (the read tools) is delegated to the inner provider. Subprocess
    execution is forbidden outright (commands may have arbitrary write side
    effects), and ``resolve_path`` rejects Write/Execute operations.

    ``ReadOnlyViolation`` is a Layer-2 (recoverable) violation, so in the harness
    loop a blocked write surfaces to the evaluator agent as a recoverable tool
    error — it does NOT halt the evaluate run. Mirrors Rust's ``ReadOnlySandbox``
    decorator in ``harness.rs``."""

    #: Standard-catalogue tool names that MUTATE the workspace and are therefore
    #: blocked by a read-only sandbox.
    DEFAULT_WRITE_TOOLS: frozenset[str] = frozenset(
        {
            "write_file",
            "edit_file",
            "delete_file",
            "move_file",
            "exec",
            "bash_command",
            "run_tests",
        }
    )

    def __init__(
        self,
        inner: SandboxProvider,
        write_tools: Iterable[str] | None = None,
    ) -> None:
        self._inner = inner
        self._write_tools: frozenset[str] = (
            frozenset(write_tools) if write_tools is not None else self.DEFAULT_WRITE_TOOLS
        )

    def _is_write(self, tool_name: str) -> bool:
        return tool_name in self._write_tools

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        if self._is_write(call.name):
            return SandboxReadOnlyViolation(path=call.name)
        return await self._inner.validate(call)

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        # A read-only sandbox forbids subprocess execution outright. Deferred
        # import: :mod:`spore_core.sandbox` imports from this module.
        from .sandbox import SandboxViolationException

        raise SandboxViolationException(SandboxReadOnlyViolation(path=command))

    async def handle_large_output(
        self,
        content: str,
        call_id: str,
        head_tokens: int,
        tail_tokens: int,
    ) -> TruncatedOutput:
        return await self._inner.handle_large_output(content, call_id, head_tokens, tail_tokens)

    async def resolve_path(self, path: str, operation: Operation = "read") -> Path:
        if operation in ("write", "execute"):
            from .sandbox import SandboxViolationException

            raise SandboxViolationException(SandboxReadOnlyViolation(path=path))
        return await self._inner.resolve_path(path, operation)

    def isolation_mode(self) -> IsolationMode:
        return self._inner.isolation_mode()

    def workspace_root(self) -> Path:
        return self._inner.workspace_root()


@dataclass
class CompactionTurn:
    """Inputs the harness compaction loop (issue #46) needs to run one
    compaction turn and verify its result.

    The harness loop operates on the opaque :class:`SessionState`; the rich
    compaction/verification API
    (:class:`spore_core.context.ContextManager`,
    :class:`spore_core.context.CompactionVerifier`) operates on
    :class:`spore_core.context.SessionState`. This struct is the bridge: a
    :class:`ContextManager` that supports compaction projects everything the
    loop needs into one value, so the loop never has to know which concrete
    state type its manager uses internally.

    ``context`` is fed straight to :meth:`Agent.turn` to produce the summary;
    ``preserve_hints`` and ``verification_state`` are passed to
    :meth:`CompactionVerifier.verify`. On a verification failure the loop
    re-runs the turn with :meth:`ContextManager.inject_missing_items` applied
    to ``context``.
    """

    context: Context
    preserve_hints: CompactionPreserveHints
    verification_state: ContextSessionState
    messages_removed: int


@runtime_checkable
class ContextManager(Protocol):
    """Issue #7 — assembles per-turn context.

    Issue #46 adds the optional compaction-loop surface
    (:meth:`prepare_compaction_turn`, :meth:`inject_missing_items`,
    :meth:`apply_compaction`). All three have defaults so managers that do not
    compact (the default :meth:`should_compact` returns ``False``) need not
    implement them.
    """

    async def assemble(self, session: SessionState, task: Task, sources: ContextSources) -> Context:
        """Build one turn's model input. ``sources`` (issue #115 / SC-26) carries
        the rich :class:`~spore_core.context.ContextSources` — guides, merged
        memory, tool schemas, and the composed static prompt — so a manager can
        place them in structural slots instead of the caller injecting them
        ad-hoc as User messages. Managers that do not consume sources may ignore
        the argument (the pre-#115 behaviour); the production
        :class:`~spore_core.compaction_adapter.StandardCompactionAdapter` renders
        them into a leading System block."""
        ...

    async def append_tool_result(
        self, session: SessionState, result: HarnessToolResult
    ) -> None: ...

    async def replace_tool_result(
        self, session: SessionState, message_index: int, result: HarnessToolResult
    ) -> None:
        """Replace the tool-result message previously appended at
        ``message_index`` with a fresh rendering of ``result``. The harness loop
        calls this from the ``AfterTool`` middleware hook (issue #11 / SC-9) when
        a middleware rewrote a result in place, so the rewrite reaches the next
        model turn. ``message_index`` is the position in ``session.messages``
        recorded right after the original :meth:`append_tool_result`. Default:
        no-op — a manager that does not store tool results as standalone messages
        need not act (the rewrite simply does not propagate, the pre-#11
        behaviour). Structural (non-inheriting) managers do not pick up this body,
        so the harness loop probes for the method via ``getattr`` before calling
        it (see ``_run_react``)."""
        _ = (session, message_index, result)

    async def append_assistant_message(self, session: SessionState, message: Message) -> None:
        """Append the assistant's turn (model output: text and/or the tool calls
        it requested) to the conversation so the next :meth:`assemble` reflects
        what the agent already did. Without this the model loses track of its own
        actions and repeats them. Default: no-op — but structural (non-inheriting)
        managers do not pick up this body, so the harness loop probes for the
        method via ``getattr`` before calling it (see ``_run_react_inner``)."""
        _ = (session, message)

    async def append_user_message(self, session: SessionState, text: str) -> None: ...

    def should_compact(self, session: SessionState) -> bool:
        """Whether the session is over its compaction threshold. Default:
        ``False`` — compaction stays opt-in behind this gate."""
        _ = session
        return False

    def prepare_compaction_turn(self, session: SessionState) -> CompactionTurn | None:
        """Build the inputs for one compaction turn (issue #46). Returns
        ``None`` when there is nothing to compact (e.g. history shorter than
        the preserve window), in which case the harness skips compaction
        entirely. Default: ``None`` — managers that never compact need not
        override this."""
        _ = session
        return None

    def inject_missing_items(self, context: Context, missing: list[str]) -> None:
        """Mutate a compaction :class:`Context` in place to request a revised
        summary on retry (issue #46). The harness calls this with the items the
        prior summary failed to preserve. Default: append the standard
        "missing these items … please revise" instruction as a user message."""
        context.messages.append(
            Message(
                role=Role.USER,
                content=TextContent(
                    text=(
                        f"Your summary is missing these items: {', '.join(missing)}. Please revise."
                    )
                ),
            )
        )

    def apply_compaction(self, session: SessionState, summary: str) -> None:
        """Accept a verified (or accepted-anyway) summary into the session,
        replacing the compacted span (issue #46). Default: no-op — only
        compaction-capable managers implement it."""
        _ = (session, summary)

    def token_budget_used(self, session: SessionState) -> int | None:
        """Post-compaction budget seam (#57 Known Deviation #2). The harness
        reads this *after* applying a compaction so the emitted ``Compaction``
        span can stamp the real ``tokens_after`` / ``tokens_reclaimed`` instead
        of reporting zero reclamation. Default: ``None`` — managers that do not
        track a token budget leave the span's pre-compaction estimate in
        place."""
        _ = session
        return None


def _default_inject_missing_items(context: Context, missing: list[str]) -> None:
    """Module-level twin of :meth:`ContextManager.inject_missing_items`'s
    default body. Structural (non-inheriting) managers do not pick up Protocol
    method defaults, so the harness loop falls back to this when a manager does
    not override ``inject_missing_items`` (issue #46)."""
    context.messages.append(
        Message(
            role=Role.USER,
            content=TextContent(
                text=(f"Your summary is missing these items: {', '.join(missing)}. Please revise.")
            ),
        )
    )


@runtime_checkable
class TerminationPolicy(Protocol):
    """Issue #13 — evaluated after each turn."""

    async def evaluate(
        self, session: SessionState, budget_used: BudgetSnapshot
    ) -> TerminationDecision: ...


class CompleteOnFinalResponse:
    """Termination policy that lets the loop complete as soon as the agent
    produces a final response.

    Always returns :class:`TerminationContinue`, which the harness interprets
    as "accept the final response and succeed". This is the default policy
    wired by :meth:`HarnessBuilder.conversational` — a tool-less chat agent
    halts naturally on its first final response, with no extra completion
    criteria to satisfy. Mirrors Rust's ``CompleteOnFinalResponse``.
    """

    async def evaluate(
        self, session: SessionState, budget_used: BudgetSnapshot
    ) -> TerminationDecision:
        return TerminationContinue()


class EmptyToolRegistry:
    """Harness-loop :class:`ToolRegistry` with no tools.

    :meth:`schemas` advertises nothing, :meth:`is_always_halt` is always
    ``False``, and :meth:`dispatch` returns a recoverable error for any call
    (there is nothing to run). This is the registry wired by
    :meth:`HarnessBuilder.conversational` for a tool-less chat agent. Mirrors
    Rust's ``EmptyToolRegistry``.
    """

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        return ToolOutputError.error(f"unknown tool: {call.name}")

    def is_always_halt(self, tool_name: str) -> bool:
        return False

    def schemas(self) -> list[ToolSchema]:
        return []


# Issue #11 / Phase 3 (Q2): the harness loop now wires the canonical, spec-rich
# :class:`spore_core.middleware.MiddlewareChain` directly through its per-hook
# ``fire_before_*`` / ``fire_after_*`` surface (the former harness-local
# ``fire(hook, session)`` stub Protocol was deleted). ``HarnessMiddlewareChain``
# survives as a deprecated alias to the rich ``MiddlewareChain`` (an external
# re-export depends on the name); it is bound at the bottom of this module,
# alongside the deferred middleware import, since the rich chain imports types
# from here.


# Issue #12 — ``ObservabilityProvider`` is no longer a per-turn no-op stub
# here. The canonical Protocol lives in :mod:`spore_core.observability`; the
# harness loop emits real spans (turn spans, child tool-call spans) through it
# and flushes on terminal outcomes. It is imported at the bottom of this module
# (after this module's own types are defined) to avoid a circular import, since
# :mod:`spore_core.observability` imports :class:`SessionId` / :class:`TaskId`
# from here. See the deferred import block alongside the middleware import.


# ============================================================================
# PausedState / ChildPausedState
# ============================================================================


class PausedState(_Model):
    session_id: SessionId
    task_id: TaskId
    turn_number: int
    session_state: SessionState
    pending_tool_calls: list[ToolCall] = Field(default_factory=list)
    approved_results: list[HarnessToolResult] = Field(default_factory=list)
    # Optional (issue #80): a ``WaitingForHuman`` pause always sets this; an
    # escalation pause sets it to ``None``. Omitted from serialization when
    # ``None`` to match the Rust ``#[serde(default)]`` wire behavior — old
    # ``WaitingForHuman`` blobs (field present) still deserialize.
    human_request: HumanRequest | None = None
    task: Task
    budget_used: BudgetSnapshot
    child_state: ChildPausedState | None = None
    # The toolset handle of the leaf that paused (#140). Resume routes pending
    # per-node tool calls through this handle's scoped catalogue via
    # :meth:`StandardHarness._effective_tool_registry`; an empty handle (the
    # default) falls back to the global catalogue. The pydantic default ``""``
    # keeps pre-#140 paused-state blobs (no ``toolset`` key) deserializing — they
    # restore as ``""``. The field ALWAYS serializes (even when empty, as
    # ``"toolset":""``) for cross-language byte-parity; declared LAST so it
    # serializes after ``child_state`` to byte-match the shared fixtures.
    toolset: ToolsetRef = ""

    def serialize_checkpoint(self) -> str:
        """The SHARED durable checkpoint round-trip (#129, AC1). Both cross-process
        ``Continue`` (a :class:`HumanRequestBudgetExhausted` pause whose request
        carries ``continues_used``) and Ralph's pause-propagation hand the SAME
        :class:`PausedState` to the caller for persistence; this is the one seam
        they share. It is JUST the :class:`PausedState` serialize/deserialize — NOT
        a unification of their CONTEXT policies (Q2): a ``Continue`` resumes
        preserving ``session_state.messages``; Ralph re-seeds a fresh window from
        its filesystem ``.spore/progress.json`` checkpoint, which stays
        Ralph-specific.

        ``serialize_checkpoint`` produces the durable blob the caller persists;
        :meth:`load_checkpoint` restores it on resume. Mirrors Rust's
        ``PausedState::serialize_checkpoint``."""
        return self.model_dump_json()

    @classmethod
    def load_checkpoint(cls, blob: str) -> PausedState:
        """Restore a :class:`PausedState` from a durable checkpoint blob (#129,
        AC1). The resume side of :meth:`serialize_checkpoint`. Mirrors Rust's
        ``PausedState::load_checkpoint``."""
        return cls.model_validate_json(blob)


class ChildPausedState(_Model):
    """Child paused state. **Deliberately has no ``child_state`` field** —
    subagents cannot spawn their own subagents (spec depth-1 rule)."""

    session_id: SessionId
    task_id: TaskId
    turn_number: int
    session_state: SessionState
    pending_tool_calls: list[ToolCall] = Field(default_factory=list)
    approved_results: list[HarnessToolResult] = Field(default_factory=list)
    # Optional (issue #80): mirrors :class:`PausedState.human_request`.
    human_request: HumanRequest | None = None
    task: Task
    budget_used: BudgetSnapshot
    parent_tool_call_id: str
    # The toolset handle of the child leaf that paused (#140); same semantics and
    # serialization contract as :attr:`PausedState.toolset`. ALWAYS serializes
    # (``"toolset":""`` when empty); the pydantic default keeps pre-#140 child
    # blobs deserializing. Declared LAST to byte-match the shared fixtures.
    toolset: ToolsetRef = ""


# Forward refs are resolved at the BOTTOM of the module (after the deferred
# import block) rather than here: ``ToolOutputWaitingForHuman`` and
# ``PausedState`` both transitively reach ``ToolOutputEscalate`` →
# ``HarnessSignal`` → :class:`PlanArtifact` / :class:`Mode` (issue #80), and
# those two types live in modules that import from this one — they are only
# importable once ``HarnessConfig`` is defined. See the ``model_rebuild`` block
# after the ``from .hooks import PlanArtifact`` / ``from .prompt_chunk_registry
# import Mode`` imports.


# ============================================================================
# HarnessError — serializable error union (issue #120)
# ============================================================================
#
# Mirrors Rust's ``#[serde(tag = "kind")]`` ``HarnessError`` enum. The ``kind``
# discriminant uses the Rust variant names verbatim (PascalCase — the Rust enum
# has no ``rename_all``), so the wire tags are ``"StrategyNotFound"`` /
# ``"UnresolvedHandle"`` (see ``fixtures/harness/registry_errors.json``).
#
# ``UnresolvedHandle``'s handle-category field is named ``kind`` in Rust but
# serializes as ``handle_kind`` (``#[serde(rename = "handle_kind")]``) so it
# does not collide with the enum's ``kind`` discriminant tag. We model the
# Python attribute as ``handle_kind`` directly to keep the wire layout byte
# identical to the fixture.


class HarnessErrorInvalidConfiguration(_Model):
    """A strategy tree (or runtime context) was misconfigured — e.g. a bare
    ``ReAct`` feeding a STRUCTURED slot without an output schema (A.5), or an
    :class:`ExecutionContext` with no wired :class:`StrategyExecutor`. Mirrors
    Rust's ``HarnessError::InvalidConfiguration(String)``. Runtime-surfaced (it
    rides a :class:`StrategyOutcomeFailed`); the ``message`` field is the
    human-readable reason. Not part of the #120 startup-handle fixture set."""

    kind: Literal["InvalidConfiguration"] = "InvalidConfiguration"
    reason: str

    def message(self) -> str:
        return f"invalid configuration: {self.reason}"


class HarnessErrorStrategyNotFound(_Model):
    """A ``StrategyRef::Custom(key)`` referenced a custom strategy absent from
    the :class:`~spore_core.execution_registry.ExecutionRegistry`'s ``custom``
    map. RECOVERABLE — returned/raised, never a crash (same pattern as a missing
    ``AgentRef``). Mirrors Rust's ``HarnessError::StrategyNotFound``."""

    kind: Literal["StrategyNotFound"] = "StrategyNotFound"
    key: str

    def message(self) -> str:
        return f"custom strategy not found: {self.key}"


class HarnessErrorUnresolvedHandle(_Model):
    """A serializable handle (``AgentRef``/``ToolsetRef``/``SchemaRef``)
    referenced an entry absent from the
    :class:`~spore_core.execution_registry.ExecutionRegistry`. The STARTUP-
    validation error: surfaced before the first turn. Mirrors Rust's
    ``HarnessError::UnresolvedHandle`` — the handle category serializes as
    ``handle_kind`` to avoid colliding with the ``kind`` discriminant tag."""

    kind: Literal["UnresolvedHandle"] = "UnresolvedHandle"
    handle_kind: str
    key: str

    def message(self) -> str:
        return f"unresolved {self.handle_kind} handle: {self.key}"


HarnessError = Annotated[
    HarnessErrorInvalidConfiguration | HarnessErrorStrategyNotFound | HarnessErrorUnresolvedHandle,
    Field(discriminator="kind"),
]


class HarnessErrorException(SporeError):
    """Raised form of a :data:`HarnessError` wire variant. Carries the
    serializable :data:`HarnessError` value so callers can ``isinstance`` /
    re-serialize it. The registry raises this for startup-validation failures
    and missing custom-strategy keys (issue #120)."""

    def __init__(self, error: HarnessError) -> None:
        self.error = error
        super().__init__(error.message())


# ============================================================================
# HaltReason / RunResult
# ============================================================================


class HaltReasonBudgetExceeded(_Model):
    kind: Literal["budget_exceeded"] = "budget_exceeded"
    limit_type: BudgetLimitTypeT


class HaltReasonTerminationPolicyHalt(_Model):
    kind: Literal["termination_policy_halt"] = "termination_policy_halt"
    reason: str


class HaltReasonMiddlewareHalt(_Model):
    kind: Literal["middleware_halt"] = "middleware_halt"
    hook: HookPoint
    reason: str


class HaltReasonAgentError(_Model):
    kind: Literal["agent_error"] = "agent_error"
    error: AgentError


class HaltReasonContextError(_Model):
    """A :class:`spore_core.context.ContextError` surfaced by the
    ``ContextManager`` during assembly halts the run — e.g. a cache-hash
    mismatch, where both Block 1 (``static``) and, as of #32, Block 2
    (``per_session``) halt mid-session. This is the routing TYPE; mirrors
    :class:`HaltReasonAgentError`. The live :class:`StandardHarness` loop does
    not yet trigger it because its placeholder ``ContextManager.assemble`` is
    infallible pending the #7 migration. Mirrors Rust's
    ``HaltReason::ContextError { error: ContextError }``."""

    kind: Literal["context_error"] = "context_error"
    error: ContextErrorModel


class HaltReasonSandboxViolation(_Model):
    kind: Literal["sandbox_violation"] = "sandbox_violation"
    violation: SandboxViolation


class HaltReasonUnrecoverableToolError(_Model):
    kind: Literal["unrecoverable_tool_error"] = "unrecoverable_tool_error"
    tool: str
    error: str


class HaltReasonToolErrorLoop(_Model):
    """The ReAct consecutive-recoverable-tool-error breaker hard-stopped (issue
    #137): ``tool`` produced ``2 * error_loop_threshold`` consecutive
    identical-argument recoverable errors. A typed terminal distinguishing an
    error-grinding circuit-break from genuine budget exhaustion
    (:class:`HaltReasonBudgetExceeded`) — the ``Fail`` / ``Escalate``→``Fail``
    resolution carries THIS reason, never ``BudgetExceeded``. Mirrors Rust's
    ``HaltReason::ToolErrorLoop { tool, consecutive_errors }``."""

    kind: Literal["tool_error_loop"] = "tool_error_loop"
    tool: str
    consecutive_errors: int


class HaltReasonHumanHalted(_Model):
    kind: Literal["human_halted"] = "human_halted"


class HaltReasonStagnationLimitReached(_Model):
    kind: Literal["stagnation_limit_reached"] = "stagnation_limit_reached"
    iterations: int
    best_metric: float


class HaltReasonStrategyNotYetImplemented(_Model):
    """Returned by :class:`StandardHarness` for non-ReAct strategies whose
    concrete trait dependencies ship with later component issues."""

    kind: Literal["strategy_not_yet_implemented"] = "strategy_not_yet_implemented"
    strategy: str


class HaltReasonEmptyPlan(_Model):
    """Returned by :class:`StandardHarness` for the ``PlanExecute`` strategy
    (issue #59, Q3). The plan phase produced a well-formed artifact whose task
    list is empty (``tasks: []``); the execute loop has nothing to drive, so the
    run fails rather than silently succeeding. Mirrors Rust's
    ``HaltReason::EmptyPlan`` and the Go/TypeScript equivalents."""

    kind: Literal["empty_plan"] = "empty_plan"


class HaltReasonStepFailed(_Model):
    """Returned by :class:`StandardHarness` for the ``PlanExecute`` strategy
    (issue #59, Q5). A per-task execute sub-loop errored or returned a
    blocked/failed outcome; this aborts the whole run (a plan is a dependency
    chain, so later tasks do not run). Carries the failing step's index, its
    description, and a debug rendering of the underlying terminal reason.
    Mirrors Rust's ``HaltReason::StepFailed { task_index, task, reason }``."""

    kind: Literal["step_failed"] = "step_failed"
    task_index: int
    task: str
    reason: str


class PlanPhaseErrorPayload(_Model):
    """Serialized form of a :class:`spore_core.plan.PlanPhaseError` as it appears
    nested under :class:`HaltReasonPlanPhaseFailed`. Wire shape matches Rust's
    ``PlanPhaseError`` and Go's ``PlanPhaseError``:
    ``{"kind": "unparseable_plan"|"planning_turn_failed", "message": "..."}``."""

    kind: Literal["unparseable_plan", "planning_turn_failed"]
    message: str


class HaltReasonPlanPhaseFailed(_Model):
    """The PlanExecute plan phase (issue #70) failed before producing an
    artifact — the planning turn failed (R2: a tool call, or an agent error) or
    the response text could not be parsed under the Q3 grammar.

    Carries the underlying :class:`PlanPhaseError` NESTED under an ``error`` key,
    matching the 3-language majority (Rust's
    ``HaltReason::PlanPhaseFailed { error: PlanPhaseError }`` and Go's
    ``HaltPlanPhaseFailed`` carrying ``*PlanPhaseError`` under ``json:"error"``,
    and the TypeScript nested ``error``)."""

    kind: Literal["plan_phase_failed"] = "plan_phase_failed"
    error: PlanPhaseErrorPayload


class HaltReasonSelfVerifyExhausted(_Model):
    """Returned by :class:`StandardHarness` for the ``SelfVerifying`` strategy
    (issue #61, D4) when the build↔evaluate loop ran out of the verifier's
    ``max_iterations`` round-trips without an explicit ``Passed`` verdict — a
    RUNTIME limit (the Default-FAIL stagnation guard). Carries the number of
    round-trips run and the last failure reason the verifier gave. PEER to
    :class:`HaltReasonSelfVerifyMisconfigured` (NOT a sub-case of it). Mirrors
    Rust's ``HaltReason::SelfVerifyExhausted { iterations, last_reason }``."""

    kind: Literal["self_verify_exhausted"] = "self_verify_exhausted"
    iterations: int
    last_reason: str


class HaltReasonSelfVerifyMisconfigured(_Model):
    """Returned by :class:`StandardHarness` for the ``SelfVerifying`` strategy
    (issue #61, D4) when the strategy cannot run because it is misconfigured —
    e.g. ``config.verifier`` is ``None``. A BUILD-TIME wiring bug, surfaced as a
    typed halt, NOT a raise. PEER to :class:`HaltReasonSelfVerifyExhausted`.
    Mirrors Rust's ``HaltReason::SelfVerifyMisconfigured { reason }``."""

    kind: Literal["self_verify_misconfigured"] = "self_verify_misconfigured"
    reason: str


class HaltReasonHillClimbingMisconfigured(_Model):
    """Returned by :class:`StandardHarness` for the ``HillClimbing`` strategy
    (issue #60) when the strategy cannot run because it is misconfigured — e.g.
    ``config.metric_evaluator`` is ``None`` (Decision 6), or the iteration-0
    baseline evaluation itself errored so there is no current best to climb from
    (Decision 7). A typed halt, NOT a raise. PEER to
    :class:`HaltReasonSelfVerifyMisconfigured`. Mirrors Rust's
    ``HaltReason::HillClimbingMisconfigured { reason }``."""

    kind: Literal["hill_climbing_misconfigured"] = "hill_climbing_misconfigured"
    reason: str


class HaltReasonRalphCompletionUnmet(_Model):
    """Returned by :class:`StandardHarness` for the ``Ralph`` strategy (issue
    #58) when the multi-context-window continuation loop reached its
    ``max_resets`` cap with tasks still incomplete (the Ralph analogue of
    :class:`HaltReasonSelfVerifyExhausted`). A RUNTIME limit — the work was
    attempted across ``iterations`` context windows but the filesystem-backed
    completion check (the registered ``Stop`` hook reading
    ``.spore/progress.json``) never reported done. Carries the number of
    context-window resets performed and the last incompletion reason. Mirrors
    Rust's ``HaltReason::RalphCompletionUnmet { iterations, last_reason }``."""

    kind: Literal["ralph_completion_unmet"] = "ralph_completion_unmet"
    iterations: int
    last_reason: str


class HaltReasonConfigurationError(_Model):
    """Returned by :class:`StandardHarness` when
    :meth:`~spore_core.execution_registry.ExecutionRegistry.validate` fails at
    run entry: a handle referenced by the task's strategy tree is unresolved
    against the configured registry, or a ``StrategyRef.Custom`` key is missing.
    A STARTUP error surfaced before the first turn (issue #120). Carries the
    underlying :data:`HarnessError`. Mirrors Rust's
    ``HaltReason::ConfigurationError { error }``."""

    kind: Literal["configuration_error"] = "configuration_error"
    error: HarnessError


class HaltReasonTasksBlockedByFailure(_Model):
    """Returned by the PlanExecute DAG executor (#126, decision A) for a PARTIAL
    run: a task failed terminally (unrecoverable error or ``BudgetExhausted``
    resolving to ``Fail``) and its transitive dependents were cascade-``blocked``,
    while unrelated branches still completed. The run as a whole is a
    :class:`RunResultFailure`, but the full partition is reported: which tasks
    ``completed``, which were ``blocked`` by the cascade, the ``failed_task`` that
    triggered it, and the human-readable ``reason``. (A run where EVERY task
    completes is a :class:`RunResultSuccess`, as before.) Mirrors Rust's
    ``HaltReason::TasksBlockedByFailure { completed, blocked, failed_task, reason }``.
    """

    kind: Literal["tasks_blocked_by_failure"] = "tasks_blocked_by_failure"
    completed: list[int]
    blocked: list[int]
    failed_task: int
    reason: str


class HaltReasonTaskGraphCycle(_Model):
    """Returned by the PlanExecute DAG executor (#126) when the persisted task
    graph contains a directed cycle, re-checked at EXECUTE ENTRY as
    defense-in-depth (``add_task`` already rejects cycles, but the ``task_list``
    tool path could in principle persist a cyclic graph out of band). No task is
    run. Carries a human-readable description. Mirrors Rust's
    ``HaltReason::TaskGraphCycle { reason }``."""

    kind: Literal["task_graph_cycle"] = "task_graph_cycle"
    reason: str


class HaltReasonOutputSchemaViolation(_Model):
    """Returned by :class:`StandardHarness` (issue #139) when output-schema
    enforcement is ON (:attr:`HarnessConfig.enforce_output_schemas`) for a
    ``ReactConfig`` leaf carrying ``output`` set, and the leaf's terminal
    ``FinalResponse`` STILL failed validation after the configured
    ``output_schema_max_retries`` extra turns were exhausted WITH budget
    remaining. DISTINCT from :class:`HaltReasonBudgetExceeded`: a budget/turn cap
    that a retry would exceed surfaces the budget terminal instead
    (budget-cap-wins precedence) — ``OutputSchemaViolation`` fires ONLY on a
    genuine validation exhaustion. Carries the resolved ``schema`` (canonical
    compact key-sorted JSON), the total number of ``attempts``
    (``1 + max_retries``), and the ``last_error`` (the frozen validator error
    string from the final attempt). Mirrors Rust's
    ``HaltReason::OutputSchemaViolation``."""

    kind: Literal["output_schema_violation"] = "output_schema_violation"
    schema: str
    attempts: int
    last_error: str


HaltReason = Annotated[
    HaltReasonBudgetExceeded
    | HaltReasonTerminationPolicyHalt
    | HaltReasonMiddlewareHalt
    | HaltReasonAgentError
    | HaltReasonContextError
    | HaltReasonSandboxViolation
    | HaltReasonUnrecoverableToolError
    | HaltReasonToolErrorLoop
    | HaltReasonHumanHalted
    | HaltReasonStagnationLimitReached
    | HaltReasonStrategyNotYetImplemented
    | HaltReasonEmptyPlan
    | HaltReasonStepFailed
    | HaltReasonPlanPhaseFailed
    | HaltReasonSelfVerifyExhausted
    | HaltReasonSelfVerifyMisconfigured
    | HaltReasonHillClimbingMisconfigured
    | HaltReasonRalphCompletionUnmet
    | HaltReasonConfigurationError
    | HaltReasonTasksBlockedByFailure
    | HaltReasonTaskGraphCycle
    | HaltReasonOutputSchemaViolation,
    Field(discriminator="kind"),
]


class RunResultSuccess(_Model):
    kind: Literal["success"] = "success"
    output: str
    session_id: SessionId
    usage: AggregateUsage
    turns: int
    # Post-run conversation history (issue #102). Lets a caller resume
    # losslessly via ``HarnessRunOptions(session_state=..)`` without
    # reconstructing tool-call / tool-result turns from ``output``. Defaulted so
    # old serialized blobs (pre-#102, no field) still deserialize — the Python
    # analogue of Rust's ``#[serde(default)]``.
    session_state: SessionState = Field(default_factory=SessionState)


class RunResultFailure(_Model):
    kind: Literal["failure"] = "failure"
    reason: HaltReason
    session_id: SessionId
    usage: AggregateUsage
    turns: int
    # Post-run conversation history (issue #102). Carried on failure too so a
    # caller can inspect what the loop produced before halting. Defaulted for
    # back-compat with pre-#102 serialized blobs (Rust ``#[serde(default)]``).
    session_state: SessionState = Field(default_factory=SessionState)


class RunResultWaitingForHuman(_Model):
    kind: Literal["waiting_for_human"] = "waiting_for_human"
    state: PausedState
    request: HumanRequest


class RunResultEscalate(_Model):
    """The harness terminated cleanly because a tool returned
    :class:`ToolOutputEscalate` (issue #80). Carries the signal plus the
    preserved :class:`PausedState` (with ``human_request = None``) so the caller
    can handle the signal and decide whether to resume the original harness,
    instantiate a new one, or present UI to the user. The signal is NOT stored
    in the ``state`` — it is discarded on resume; the harness never re-acts on
    it."""

    kind: Literal["escalate"] = "escalate"
    signal: HarnessSignal
    state: PausedState
    session_id: SessionId
    usage: AggregateUsage
    turns: int


class RunResultConsult(_Model):
    """Worker paused mid-loop to consult a parent-spawned helper (issue #114).
    Sibling of :class:`RunResultWaitingForHuman`, but it stops at the
    ORCHESTRATOR (via ``SubagentTool``'s A1 mediation), not the human. The
    ``state`` preserves the full :class:`PausedState` with ``human_request=None``
    and the consult call as the head of ``pending_tool_calls``, so
    ``harness.resume_consult(state, response, ..)`` continues the worker. With no
    consult handlers registered, a standalone worker simply returns this
    unchanged to its caller (R6 graceful degradation)."""

    kind: Literal["consult"] = "consult"
    request: ConsultRequest
    state: PausedState
    session_id: SessionId
    usage: AggregateUsage
    turns: int


RunResult = Annotated[
    RunResultSuccess
    | RunResultFailure
    | RunResultWaitingForHuman
    | RunResultEscalate
    | RunResultConsult,
    Field(discriminator="kind"),
]


# Import canonical middleware decision types. This import is deliberately
# placed after the harness's own types are defined so
# :mod:`spore_core.middleware` can import :class:`HumanRequest`,
# :class:`RunResult`, :class:`Task`, :class:`SessionState`, etc., from
# this module without circularity.
from .middleware import (  # noqa: E402
    Middleware,
    MiddlewareChain,
    MiddlewareContinue,
    MiddlewareContinueWithModification,
    MiddlewareDecision,
    MiddlewareForceAnotherTurn,
    MiddlewareHalt,
    MiddlewareSurfaceToHuman,
)

# Deprecated alias: the harness-local middleware stub was deleted in Phase 3 (Q2);
# the canonical rich ``MiddlewareChain`` is now the type the loop wires. Kept so
# the ``spore_core`` package re-export of ``HarnessMiddlewareChain`` keeps working.
HarnessMiddlewareChain = MiddlewareChain

# Canonical observability surface (issue #12). Imported here, after this
# module's identity types are defined, because :mod:`spore_core.observability`
# imports :class:`SessionId` / :class:`TaskId` from this module — a top-level
# import would be circular. The harness loop emits real spans through this
# provider and flushes on terminal outcomes (mirrors the Rust ``run_react``
# wrapper). The durable-outbox provider is used by
# :meth:`HarnessBuilder.with_observability_outbox`.
from .guide_registry import (  # noqa: E402
    SessionOutcome,
    SessionOutcomeEscalated,
    SessionOutcomeFailure,
    SessionOutcomeSuccess,
)
from .memory import MemoryError, MemoryItem, MemoryQuery  # noqa: E402
from .memory import now as _now  # noqa: E402
from .observability import (  # noqa: E402
    ContentCaptureConfig,
    ContextOperationCompaction,
    ContextOperationConsultResumed,
    ContextOperationConsultSpawned,
    ContextOperationOutputSchemaRetry,
    ContextOperationOutputSchemaViolation,
    ContextOperationToolErrorLoopBroken,
    ContextOperationToolErrorLoopDetected,
    ContextSpan,
    GenAiMessage,
    GenAiRole,
    ObservabilityProvider,
    PricingTable,
    SpanBase,
    SpanId,
    SpanKind,
    SpanStatusError,
    SpanStatusOk,
    ToolCallContent,
    ToolCallSpan,
    ToolResultContent,
    TurnSpan,
    WarnEventCompactionVerificationFailed,
    WarnEventHillClimbingIteration,
    WarnSpan,
    new_span_id,
    truncate_field,
)
from .observability_outbox import (  # noqa: E402
    OutboxConfig,
    OutboxObservabilityProvider,
)
from .storage import (  # noqa: E402
    RALPH_FEATURE_LIST_KEY,
    RALPH_PROGRESS_KEY,
    ProjectId,
    RunStore,
    SessionStore,
    StorageError,
    StorageProvider,
    project_namespace,
)


def _capture_tool_call_args(call: ToolCall, max_len: int) -> ToolCallContent:
    """Capture a requested tool call as :class:`ToolCallContent`, truncating its
    arguments to ``max_len`` UTF-8 bytes (issue #64). The arguments are measured
    by their canonical JSON serialization; when over budget they are clipped and
    stored as a JSON string carrying the truncation marker (a structured value
    cannot be clipped in place), with ``arguments_truncated=True``. Mirrors the
    Rust reference ``capture_tool_call_args``."""
    serialized = json.dumps(call.input, separators=(",", ":"))
    clipped, truncated = truncate_field(serialized, max_len)
    arguments: Any = clipped if truncated else call.input
    return ToolCallContent(
        name=call.name,
        arguments=arguments,
        arguments_truncated=truncated,
    )


_ROLE_TO_GENAI: dict[Role, GenAiRole] = {
    Role.SYSTEM: GenAiRole.SYSTEM,
    Role.USER: GenAiRole.USER,
    Role.ASSISTANT: GenAiRole.ASSISTANT,
    Role.TOOL: GenAiRole.TOOL,
}


def _capture_input_messages(messages: list[Message], max_len: int) -> list[GenAiMessage]:
    """Snapshot the assembled INPUT messages (the full prompt the model saw)
    into :class:`GenAiMessage`s for LLM-native tracing (issue #64). Each
    message's :class:`Role` maps to the conventional :class:`GenAiRole`; the
    :class:`Content` is rendered to a plain string and truncated to ``max_len``
    UTF-8 bytes:

    * ``TextContent``       → the text verbatim
    * ``ToolResultContent`` → its ``content`` body (role stays ``Tool``)
    * ``ToolCallContent``   → ``"<name> <compact-json-args>"`` (assistant)
    * ``ImageContent``      → ``"[image <media_type>]"`` — NEVER the base64 data

    System-first, then history order is preserved because the assembled
    ``messages`` already lead with the :class:`Role.SYSTEM` prompt. Mirrors the
    Rust reference ``capture_input_messages``."""
    out: list[GenAiMessage] = []
    for m in messages:
        content = m.content
        if isinstance(content, TextContent):
            rendered = content.text
        elif isinstance(content, MsgToolResultContent):
            rendered = content.content
        elif isinstance(content, MsgToolCallContent):
            rendered = f"{content.name} {json.dumps(content.input, separators=(',', ':'))}"
        elif isinstance(content, ImageContent):
            # NEVER dump the base64 ``data`` — placeholder only.
            rendered = f"[image {content.media_type}]"
        else:  # pragma: no cover - exhaustive over the Content union
            rendered = ""
        clipped, truncated = truncate_field(rendered, max_len)
        out.append(
            GenAiMessage(
                role=_ROLE_TO_GENAI[m.role],
                content=clipped,
                truncated=truncated,
            )
        )
    return out


# ============================================================================
# HarnessRunOptions
# ============================================================================


class HarnessRunOptions:
    """Options for :meth:`Harness.run`. Not a pydantic model because
    ``on_stream`` is a callable."""

    def __init__(
        self,
        task: Task,
        *,
        on_stream: StreamSink | None = None,
        session_state: SessionState | None = None,
    ) -> None:
        self.task = task
        self.on_stream = on_stream
        self.session_state = session_state


# ============================================================================
# Harness protocol
# ============================================================================


@runtime_checkable
class Harness(Protocol):
    """Drives the agent loop."""

    async def run(self, options: HarnessRunOptions) -> RunResult: ...

    async def resume(
        self,
        state: PausedState,
        response: HumanResponse,
        on_stream: StreamSink | None = None,
    ) -> RunResult: ...

    async def resume_consult(
        self,
        state: PausedState,
        response: ConsultResponse,
        on_stream: StreamSink | None = None,
    ) -> RunResult:
        """Resume a worker paused by :class:`RunResultConsult` (issue #114). The
        :class:`ConsultResponse` is injected as the tool RESULT of the head
        pending consult call, then the loop continues. Default-implemented to
        ``NotImplementedError`` so callers that never use consults need not
        provide it (mirrors the Rust default trait method)."""
        raise NotImplementedError("resume_consult is not implemented for this harness")


# ============================================================================
# HarnessConfig + StandardHarness
# ============================================================================


@dataclass
class MemoryConfig:
    """A configured memory source for the live loop (issue #160 / SC-26
    follow-up).

    Bundles a :class:`~spore_core.memory.MemoryProvider` (held structurally /
    by Protocol — Python dispatch is already dynamic, so the object-safety
    conversion the Rust reference needed is a no-op here) with the per-turn
    query policy. When present on :attr:`HarnessConfig.memory`, the harness
    queries the provider each turn in
    :meth:`StandardHarness._build_context_sources` and injects the returned
    items into :attr:`~spore_core.context.ContextSources.memory`, which the
    production ``StandardCompactionAdapter`` renders into the leading
    structural System block — alongside guides + skills, with no consumer-side
    wrapper. ``None`` (the default) leaves
    :attr:`~spore_core.context.ContextSources.memory` empty, byte-identical to
    the pre-#160 behaviour.

    :param provider: The provider queried each turn.
    :param query: Fixed query text. ``None`` (the default) uses the current
        task's ``instruction``, so retrieved memory tracks what the agent is
        working on; a configured ``query`` overrides it.
    :param domain: Optional domain filter passed to
        :attr:`~spore_core.memory.MemoryQuery.domain`.
    :param min_relevance: Minimum relevance score; items below it are dropped.
        Defaults to ``0.5`` (matching :class:`~spore_core.memory.MemoryQuery`).
    :param max_items: Maximum number of items injected per turn. Defaults to
        ``10``.
    """

    provider: MemoryProvider
    query: str | None = None
    domain: str | None = None
    min_relevance: float = 0.5
    max_items: int = 10


class HarnessConfig:
    """Components injected at construction. Mirrors ``HarnessConfig`` in
    the spec. Optional components default to no-op stubs once the loop is
    actually exercised; only ``agent``, ``tool_registry``, ``sandbox``,
    ``context_manager``, ``termination_policy`` are required by the
    ReAct path."""

    def __init__(
        self,
        *,
        agent: Agent,
        tool_registry: ToolRegistry,
        sandbox: SandboxProvider,
        context_manager: ContextManager,
        termination_policy: TerminationPolicy,
        middleware: MiddlewareChain | None = None,
        observability: ObservabilityProvider | None = None,
        compaction_verifier: CompactionVerifier | None = None,
        max_compaction_attempts: int = 2,
        pricing: PricingTable | None = None,
        content_capture: ContentCaptureConfig | None = None,
        max_stop_blocks: int = 8,
        error_loop_threshold: int = 3,
        enforce_output_schemas: bool = False,
        output_schema_max_retries: int = 2,
        max_resets: int = 3,
        vcs_provider: VcsProvider | None = None,
        hooks: HookChain | None = None,
        planner_agent: Agent | None = None,
        verifier: Any | None = None,
        evaluator_agent: Agent | None = None,
        storage: StorageProvider | None = None,
        project_id: ProjectId | None = None,
        chunk_provider: Any | None = None,
        metric_evaluator: Any | None = None,
        catalogue_registry: StandardToolRegistry | None = None,
        toolset_catalogues: dict[str, StandardToolRegistry] | None = None,
        system_prompt: str | None = None,
        guides: list[Guide] | None = None,
        skills: SkillCatalog | None = None,
        memory: MemoryConfig | None = None,
        model_params: ModelParams | None = None,
        auto_persist_sessions: bool = False,
        prompt_tool_call_flag: PromptToolCallFlag | None = None,
        consult_handlers: dict[str, ConsultHandlerEntry] | None = None,
        registry: ExecutionRegistry | None = None,
        escalation_mode: EscalationMode | None = None,
    ) -> None:
        # #124: the legacy single-collaborator fields (``agent`` / ``verifier`` /
        # ``planner_agent`` / ``evaluator_agent`` / ``metric_evaluator``) are GONE
        # as live fields — all collaborator resolution now goes through
        # :class:`ExecutionRegistry`. The constructor still ACCEPTS them (public
        # signature stays stable) and folds them into ``self.registry`` under the
        # DEFAULT empty-string handle so bare ``ReactConfig.per_loop`` leaves
        # (empty ``AgentRef`` / ``ToolsetRef``) and the default
        # ``SelfVerifyingConfig.evaluator`` / ``HillClimbingConfig.evaluator``
        # (empty key) resolve to them. ``planner_agent`` is DROPPED (Q1: the plan
        # child's leaf ``ReactConfig.agent`` is authoritative); ``evaluator_agent``
        # is DROPPED (Q1c: the evaluate-phase agent defaults to the inner worker's
        # resolved agent). The folding happens after ``self.registry`` is set up
        # below. ``_agent`` is retained ONLY as the value to fold (never read as a
        # live collaborator).
        self._fold_agent = agent
        self._fold_verifier = verifier
        self._fold_metric_evaluator = metric_evaluator
        # ``planner_agent`` / ``evaluator_agent`` are accepted-but-ignored (#124,
        # Q1): the resolution path no longer routes a separate planner / evaluator
        # agent. They are kept in the signature only for source compatibility.
        _ = planner_agent
        _ = evaluator_agent
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.context_manager = context_manager
        self.termination_policy = termination_policy
        self.middleware = middleware
        self.observability = observability
        # Lifecycle hook chain (issue #69). The harness fires registered Stop
        # hooks when a loop strategy believes it is done; a Stop ``block``
        # injects a reason and continues the loop. ``None`` means no hooks.
        self.hooks: HookChain | None = hooks
        # Maximum consecutive Stop-hook blocks honored per run before the loop
        # terminates anyway (issue #69, R14). Per-run counter; resume starts
        # fresh. Default 8, matching Claude Code's behavior.
        self.max_stop_blocks = max_stop_blocks
        # Consecutive-recoverable-tool-error breaker threshold ``N`` (issue #137).
        # In the ReAct turn-loop, ``N`` consecutive identical-argument recoverable
        # errors from one tool inject ONE corrective message (AC2); ``2 * N``
        # hard-stops the loop and resolves the node's ``BudgetExhaustedBehavior``
        # with :class:`HaltReasonToolErrorLoop` (never ``BudgetExceeded``),
        # WITHOUT burning the remaining budget (AC3). Defaults to ``3``; ``0``
        # disables the breaker. Mirrors Rust's ``HarnessConfig::error_loop_threshold``.
        self.error_loop_threshold = error_loop_threshold
        # MIGRATION GATE (issue #139) — NOT a permanent feature flag. When
        # ``True``, a :class:`ReactConfig` leaf carrying ``output`` set has its
        # resolved output schema DELIVERED to the model (directive seed +
        # :attr:`ModelParams.output_schema` constrained-decoding channel) and its
        # terminal ``FinalResponse`` VALIDATED against that schema; a validation
        # failure feeds the error back and retries up to
        # ``output_schema_max_retries`` extra turns (within budget), then
        # terminates with :class:`HaltReasonOutputSchemaViolation`. Enforcement is
        # UNIFORM. Default ``False`` (OFF) keeps every existing replay fixture
        # byte-for-byte green: when OFF, ``ReactConfig.run`` behaves EXACTLY as
        # before — no resolve, no delivery, no validation. Mirrors Rust's
        # ``HarnessConfig::enforce_output_schemas``.
        self.enforce_output_schemas = enforce_output_schemas
        # The ``N`` extra terminal-validation retry turns granted when
        # ``enforce_output_schemas`` is ON and a terminal fails output-schema
        # validation (issue #139). Total attempts = ``1 + N``. Retried turns
        # COUNT against the turn budget; a retry that would exceed the budget
        # surfaces the budget terminal instead of
        # :class:`HaltReasonOutputSchemaViolation` (budget-cap-wins precedence).
        # Defaults to ``2``. Mirrors Rust's
        # ``HarnessConfig::output_schema_max_retries``.
        self.output_schema_max_retries = output_schema_max_retries
        # Ralph outer-loop reset cap (issue #58, B3). The maximum number of
        # context windows the ``Ralph`` strategy runs before halting with
        # :class:`HaltReasonRalphCompletionUnmet` when tasks are still
        # incomplete. Independent of ``budget.max_turns`` (which bounds the
        # per-window ReAct sub-loop). Default ``3``.
        self.max_resets = max_resets
        # VcsProvider seam (issue #58 v2, decision B4). When set, the ``Ralph``
        # reload phase ALSO calls ``vcs_provider.log(args)`` and injects the
        # output into each fresh context window's seed as a delimited
        # "Recent VCS history:" section, exactly the way the reloaded
        # ``.spore/`` progress/feature-list content is injected. When ``None``
        # (the default) the git-log section is OMITTED and Ralph behaves
        # byte-for-byte like v1. Mirrors Rust's ``HarnessConfig::vcs_provider``.
        self.vcs_provider: VcsProvider | None = vcs_provider
        # Post-compaction verifier (issue #29/#46). The harness runs it after
        # each compaction turn and retries up to ``max_compaction_attempts``
        # before accepting a failing summary. Defaults to ``KeyTermVerifier``.
        if compaction_verifier is None:
            from .context import KeyTermVerifier

            compaction_verifier = KeyTermVerifier()
        self.compaction_verifier: CompactionVerifier = compaction_verifier
        # Maximum compaction-summary attempts before accepting a failing summary
        # anyway (issue #46). Defaults to ``2`` (mirrors ``CompactionConfig``).
        self.max_compaction_attempts = max_compaction_attempts
        # Token → USD pricing used to stamp ``cost_usd`` on emitted turn spans.
        # Defaults to :attr:`PricingTable.DEFAULT` (zero cost) when unset.
        self.pricing: PricingTable = pricing if pricing is not None else PricingTable.DEFAULT
        # LLM-native content capture config (issue #64). Defaults to OFF. When
        # disabled the harness populates none of the ``gen_ai.*`` content fields,
        # so the durable JSONL stays byte-identical to the pre-#64 output.
        self.content_capture: ContentCaptureConfig = (
            content_capture if content_capture is not None else ContentCaptureConfig()
        )
        # Pluggable persistence layer (issue #73). Defaults to an all-no-op
        # provider so existing callers/tests are unaffected; v1 is expose-only —
        # the run/resume loop does NOT read/write sessions internally. Callers
        # reach the four domain stores via :meth:`StandardHarness.storage` /
        # :meth:`StandardHarness.session_store`.
        self.storage: StorageProvider = storage if storage is not None else StorageProvider.no_op()
        # The STABLE project namespace for DURABLE artifacts (issue #142). Where
        # the per-window :class:`SessionId` is regenerated on every Ralph
        # context-window reset (``new_session_id()``), this
        # :class:`~spore_core.storage.ProjectId` stays constant across windows AND
        # process restarts — which is what lets the ``task_list``, plan artifact,
        # and Ralph checkpoint persist across a window reset instead of being
        # orphaned under a regenerated session. Defaults (when unset, e.g. via
        # :meth:`HarnessBuilder.build_config`) to a project id derived from
        # ``sandbox.workspace_root()`` (decision 5 — NOT process cwd). Durable
        # ``RunStore`` call sites key by ``project_namespace(project_id)``
        # (namespace-reuse on the existing ``session_id`` axis); ephemeral
        # session/conversation state stays keyed by the per-window
        # :class:`SessionId`. Mirrors Rust's ``HarnessConfig::project_id``.
        if project_id is None:
            from .storage import project_id_from_canonical_path, project_id_from_path

            workspace_root = self.sandbox.workspace_root()
            try:
                project_id = project_id_from_path(workspace_root)
            except StorageError:
                # The workspace root may not exist on disk yet (e.g. a test stub);
                # fall back to the PURE derivation over the workspace_root string
                # so a project_id is ALWAYS present and deterministic.
                project_id = project_id_from_canonical_path(str(workspace_root))
        self.project_id: ProjectId = project_id
        # Pluggable chunk source for the #79 prompt assembly engine. Defaults to
        # an empty in-memory provider. Typed ``Any`` to avoid importing
        # ``prompt_assembly`` at module load.
        if chunk_provider is None:
            from .prompt_assembly import InMemoryChunkProvider

            chunk_provider = InMemoryChunkProvider.empty()
        self.chunk_provider: Any = chunk_provider
        # Catalogue tools accumulated via :meth:`HarnessBuilder.tool` /
        # ``tools`` (issue #81), drained into a populated
        # :class:`StandardToolRegistry` at :meth:`HarnessBuilder.build_config`.
        # When not ``None`` the run loop bridges it per-run via
        # :class:`~spore_core.tool_registry.RealToolRegistry` — threading the
        # run's :class:`SessionId`, sandbox, and storage into every tool
        # dispatch — and uses that instead of :attr:`tool_registry` (which stays
        # the harness-loop seam for custom slim registries). ``None`` (the
        # default) preserves the ``tool_registry``-only path unchanged.
        self.catalogue_registry: StandardToolRegistry | None = catalogue_registry
        # Per-toolset catalogues (Issue 2: per-node toolset scoping), keyed by the
        # non-empty ``ToolsetRef`` handle a leaf carries. Populated by
        # :meth:`HarnessBuilder.toolset_tools` — each value is a populated
        # :class:`StandardToolRegistry` folded from that key's catalogue tools. At
        # dispatch the run loop bridges the matching catalogue per-run via
        # :class:`~spore_core.tool_registry.RealToolRegistry` (same
        # sandbox/session/storage wiring as the global :attr:`catalogue_registry`),
        # so a node with a non-empty ``toolset`` handle dispatches ONLY its own
        # tools. A leaf with an EMPTY handle (``""``) — or a non-empty handle with
        # no entry here — falls back to the global catalogue / ``tool_registry``
        # seam (back-compat with examples 01–11 that use ``.tools()``). Empty (the
        # default) preserves today's behaviour. Mirrors Rust's
        # ``HarnessConfig::toolset_catalogues``.
        self.toolset_catalogues: dict[str, StandardToolRegistry] = (
            dict(toolset_catalogues) if toolset_catalogues is not None else {}
        )
        # Operating system prompt prepended to each turn's assembled context
        # when the context manager renders none (issue #91). See
        # :meth:`HarnessBuilder.system_prompt`. ``None`` (the default) preserves
        # today's behaviour.
        self.system_prompt: str | None = system_prompt
        # Guides (skills/playbooks/domain knowledge) injected structurally into
        # every turn's assembled context via the rich ``ContextSources`` seam
        # (issue #115 / SC-26 / #9). The harness clones these into
        # ``ContextSources.guides`` each turn; the production
        # ``StandardCompactionAdapter`` renders them into the leading System block,
        # not as ad-hoc User messages. Empty (the default) preserves today's
        # behaviour byte-for-byte. See :meth:`HarnessBuilder.guide`.
        self.guides: list[Guide] = list(guides) if guides is not None else []
        # Optional skill catalog (issue #115 / SC-26). When set, the harness
        # injects its manifest + active skill bodies into ``ContextSources.guides``
        # each turn (progressive disclosure) and the ``load_skill`` tool activates
        # skills against its shared active set. ``None`` (the default) means no
        # skills. See :meth:`HarnessBuilder.skills`.
        self.skills: SkillCatalog | None = skills
        # Optional memory source (issue #160 / SC-26 follow-up). When set, the
        # harness queries the provider each turn and injects the returned items
        # into ``ContextSources.memory``, rendered into the structural System
        # block alongside guides + skills. ``None`` (the default) leaves memory
        # empty, byte-identical to the pre-#160 behaviour. See
        # :meth:`HarnessBuilder.memory` and :class:`MemoryConfig`.
        self.memory: MemoryConfig | None = memory
        # Authoritative per-run model sampling/decoding parameters (issue #93).
        # The harness replaces each turn's ``Context.params`` with this value
        # UNCONDITIONALLY (builder params win) right before the request is built,
        # so the configured params reach every agent turn that requests tools.
        # See :meth:`HarnessBuilder.model_params`. Defaults to ``ModelParams()``.
        self.model_params: ModelParams = model_params if model_params is not None else ModelParams()
        # Opt-in conversation-history threading via the SessionStore (issue
        # #102). OFF by default: when ``False`` the run/resume loop performs
        # ZERO session-store I/O and behaves byte-for-byte like today. When
        # ``True`` the harness auto-loads the prior :class:`SessionState` for the
        # run's ``session_id`` at the start of a fresh ``run()`` (ReAct /
        # SelfVerifying only — Ralph/HillClimbing discard incoming state by
        # design; explicit ``session_state`` always wins) and auto-persists the
        # post-run state back to the store at the terminal seam. Mirrors Rust's
        # ``HarnessConfig::auto_persist_sessions``.
        self.auto_persist_sessions: bool = auto_persist_sessions
        # Shared escalation flag for adaptive prompt-based tool calling (#111).
        # ``Some`` only on the :meth:`HarnessBuilder.conversational` path, where
        # the agent's model is wrapped in an
        # :class:`~spore_core.prompt_tool_call.AdaptiveToolCallModelInterface`
        # holding the same holder. ``None`` (every other construction) disables
        # adaptive escalation. The run loop resets it at each window start and
        # flips it when a prose response is detected where a tool call was
        # expected. Mirrors Rust's ``HarnessConfig::prompt_tool_call_flag``.
        self.prompt_tool_call_flag: PromptToolCallFlag | None = prompt_tool_call_flag
        # Per-kind consult handlers (issue #114), keyed by ``ConsultRequest.kind``.
        # Empty (the default) means consults are NOT mediated: a worker that
        # pauses with :class:`RunResultConsult` surfaces it unchanged to its
        # caller (R6 graceful degradation). Populated via
        # :meth:`HarnessBuilder.consult_handler`. ``SubagentTool`` consumes this
        # map (built from the orchestrator's config) to mediate child consults.
        self.consult_handlers: dict[str, ConsultHandlerEntry] = (
            dict(consult_handlers) if consult_handlers is not None else {}
        )
        # Runtime resolver for the serializable strategy handles (#120/#124). The
        # registry resolves per-node ``AgentRef`` / ``ToolsetRef`` / ``SchemaRef``
        # handles, the SelfVerifying verifier key, the HillClimbing metric
        # evaluator key (the sixth map, Q2), and ``StrategyRef.Custom`` keys;
        # :meth:`StandardHarness.run` calls ``registry.validate(task)`` at entry —
        # the single resolution path — so an unresolved handle is a startup error
        # before the first turn. #124: the builder's single-collaborator inputs
        # (``agent`` / ``tool_registry`` / ``verifier`` / ``metric_evaluator``)
        # are FOLDED into this registry under the DEFAULT empty-string key, so the
        # single resolution path always has a worker to resolve. Explicitly
        # registered handles win (``setdefault`` semantics).
        from .execution_registry import ExecutionRegistry as _ExecutionRegistry

        if registry is None:
            registry = _ExecutionRegistry.empty()
        reg_builder = registry.into_builder()
        reg_builder = reg_builder.fill_default_agent(self._fold_agent)
        reg_builder = reg_builder.fill_default_toolset(tool_registry)
        # Issue 2: a leaf carrying a non-empty ``toolset`` handle must RESOLVE
        # against the registry (``ExecutionRegistry.validate`` runs
        # ``check_toolset`` at run entry). For every per-key catalogue wired via
        # ``.toolset_tools``, ensure the registry has a presence entry under that
        # handle so ``validate()`` passes WITHOUT the caller manually registering
        # a placeholder. The registry VALUE is never dispatched (dispatch goes
        # through ``self.toolset_catalogues``), so a no-op ``EmptyToolRegistry`` is
        # sufficient; an explicitly-registered toolset under the same key wins.
        for key in self.toolset_catalogues:
            reg_builder = reg_builder.fill_toolset(key, EmptyToolRegistry())
        reg_builder = reg_builder.fill_default_schema({})
        if self._fold_verifier is not None:
            reg_builder = reg_builder.fill_default_verifier(self._fold_verifier)
        if self._fold_metric_evaluator is not None:
            reg_builder = reg_builder.fill_default_metric_evaluator(self._fold_metric_evaluator)
        self.registry: ExecutionRegistry = reg_builder.build()
        # HITL-vs-AFK escalation knob (#120, PRD goal #7). Selects whether budget
        # escalation surfaces to a human or proceeds autonomously. STORED only
        # this slice (#130 consumes it); NOT part of the serialized ``Task``.
        # Defaults to ``SurfaceToHuman`` (the type itself carries no default).
        if escalation_mode is None:
            from .execution_registry import EscalationModeSurfaceToHuman

            escalation_mode = EscalationModeSurfaceToHuman()
        self.escalation_mode: EscalationMode = escalation_mode

    def with_sandbox(self, sandbox: SandboxProvider) -> HarnessConfig:
        """A full copy of this config with only ``sandbox`` swapped (#124). Every
        other component (incl. the ExecutionRegistry) is shared by reference so the
        child run's spans land in the same trace stream and the configured handles
        resolve identically. Mirrors Rust's ``self.config.clone()`` +
        ``eval_config.sandbox = read_only_sandbox`` in ``run_evaluate_phase``."""
        return HarnessConfig(
            agent=self._fold_agent,
            tool_registry=self.tool_registry,
            sandbox=sandbox,
            context_manager=self.context_manager,
            termination_policy=self.termination_policy,
            middleware=self.middleware,
            observability=self.observability,
            compaction_verifier=self.compaction_verifier,
            max_compaction_attempts=self.max_compaction_attempts,
            pricing=self.pricing,
            content_capture=self.content_capture,
            max_stop_blocks=self.max_stop_blocks,
            error_loop_threshold=self.error_loop_threshold,
            enforce_output_schemas=self.enforce_output_schemas,
            output_schema_max_retries=self.output_schema_max_retries,
            max_resets=self.max_resets,
            vcs_provider=self.vcs_provider,
            hooks=self.hooks,
            storage=self.storage,
            # #142: the child run keys durable artifacts by the SAME stable
            # project namespace as the parent (a swapped sandbox must not change
            # the durable namespace).
            project_id=self.project_id,
            chunk_provider=self.chunk_provider,
            catalogue_registry=self.catalogue_registry,
            toolset_catalogues=self.toolset_catalogues,
            system_prompt=self.system_prompt,
            guides=self.guides,
            skills=self.skills,
            memory=self.memory,
            model_params=self.model_params,
            auto_persist_sessions=self.auto_persist_sessions,
            prompt_tool_call_flag=self.prompt_tool_call_flag,
            consult_handlers=self.consult_handlers,
            registry=self.registry,
            escalation_mode=self.escalation_mode,
        )


# ── SC-8 builder-preset shared defaults ──────────────────────────────────────
#
# Module constants for the autonomous presets
# (:meth:`HarnessBuilder.coding_agent` / :meth:`HarnessBuilder.hill_climber`),
# so a consumer can reference or extend them. Mirrors the Rust associated
# constants ``HarnessBuilder::{CODING_AGENT_SYSTEM_PROMPT, PRESET_MAX_AUTO_GRANTS,
# PRESET_STEPS_PER_GRANT}``; re-exposed as class attributes on
# :class:`HarnessBuilder` for parity with that access path.

#: Built-in system prompt for :meth:`HarnessBuilder.coding_agent`: a coding
#: agent that ACTS through the workspace tools (rather than describing what it
#: would do) and narrates each step to the user via ``send_message``. Exposed so
#: a consumer can extend it; override it wholesale with
#: :meth:`HarnessBuilder.system_prompt`. The text is COPIED VERBATIM from the
#: Rust ``CODING_AGENT_SYSTEM_PROMPT`` for cross-language parity.
CODING_AGENT_SYSTEM_PROMPT = (
    "You are a coding agent working inside a sandboxed workspace directory. "
    "Explore with list_dir, read_file, grep, and find_files; create and change files with "
    "write_file and edit_file; run commands with bash. Use relative paths only. "
    "Act using tools — do not just describe what you would do. When the task is done, "
    "reply with a short summary of what you changed. "
    "The user CANNOT see your reasoning or your tool calls — they only see the messages you "
    "send with the `send_message` tool and your final reply. So before (or as) you act, "
    "call `send_message` with one short sentence saying what you are about to do, in "
    "PARALLEL with the tool that does the work, so narration never costs an extra round trip. "
    "Keep each message to a single short sentence."
)

#: Default per-scope auto-continue cap for the autonomous presets
#: (:meth:`HarnessBuilder.coding_agent` / :meth:`HarnessBuilder.hill_climber`):
#: grant up to this many extra step budgets at an ``Escalate`` point before the
#: run gives up. Mirrors the hand-rolled drive loop the consumers used (the
#: ``12-cordyceps`` example's ``MAX_AUTO_CONTINUES``). Override the whole policy
#: with :meth:`HarnessBuilder.escalation_mode`.
PRESET_MAX_AUTO_GRANTS = 10
#: Steps granted on each auto-continue for the autonomous presets (the
#: ``12-cordyceps`` example's ``CONTINUE_STEPS``). See
#: :data:`PRESET_MAX_AUTO_GRANTS`.
PRESET_STEPS_PER_GRANT = 25


class HarnessBuilder:
    """Fluent assembler for a :class:`HarnessConfig` / :class:`StandardHarness`.

    Mirrors the Rust ``HarnessBuilder``. The harness follows strict inversion
    of control: every component is injected. The builder takes the five
    required components up front and exposes fluent setters for the optional
    ones (middleware, observability, pricing), including the durable outbox via
    :meth:`with_observability_outbox`.
    """

    # SC-8: the autonomous-preset shared defaults, re-exposed as class
    # attributes for parity with the Rust associated-constant access path
    # (``HarnessBuilder::CODING_AGENT_SYSTEM_PROMPT`` etc.). The module-level
    # constants are the source of truth.
    CODING_AGENT_SYSTEM_PROMPT: ClassVar[str] = CODING_AGENT_SYSTEM_PROMPT
    PRESET_MAX_AUTO_GRANTS: ClassVar[int] = PRESET_MAX_AUTO_GRANTS
    PRESET_STEPS_PER_GRANT: ClassVar[int] = PRESET_STEPS_PER_GRANT

    def __init__(
        self,
        agent: Agent,
        tool_registry: ToolRegistry,
        sandbox: SandboxProvider,
        context_manager: ContextManager,
        termination_policy: TerminationPolicy,
    ) -> None:
        self._agent = agent
        self._tool_registry = tool_registry
        self._sandbox = sandbox
        self._context_manager = context_manager
        self._termination_policy = termination_policy
        self._middleware: MiddlewareChain | None = None
        self._observability: ObservabilityProvider | None = None
        self._compaction_verifier: CompactionVerifier | None = None
        self._max_compaction_attempts: int = 2
        self._pricing: PricingTable = PricingTable.DEFAULT
        self._content_capture: ContentCaptureConfig | None = None
        self._max_stop_blocks: int = 8
        self._error_loop_threshold: int = 3
        # MIGRATION GATE (issue #139): deliver + enforce ``ReactConfig.output``
        # schemas. Defaults to ``False`` (OFF). See
        # :attr:`HarnessConfig.enforce_output_schemas`.
        self._enforce_output_schemas: bool = False
        # Extra output-schema validation retry turns ``N`` (issue #139). Defaults
        # to ``2``. See :attr:`HarnessConfig.output_schema_max_retries`.
        self._output_schema_max_retries: int = 2
        self._max_resets: int = 3
        self._vcs_provider: VcsProvider | None = None
        self._hooks: HookChain | None = None
        self._planner_agent: Agent | None = None
        self._verifier: Any | None = None
        self._evaluator_agent: Agent | None = None
        self._metric_evaluator: Any | None = None
        self._storage: StorageProvider | None = None
        # The STABLE durable-storage project namespace (issue #142). ``None`` (the
        # default) resolves at :meth:`build_config` to a
        # :class:`~spore_core.storage.ProjectId` derived from
        # ``sandbox.workspace_root()`` (decision 5 — NOT process cwd),
        # canonicalizing the path first; an explicit :meth:`project_id` always
        # wins. See :attr:`HarnessConfig.project_id`.
        self._project_id: ProjectId | None = None
        # Opt-in session-state threading + auto-persist (issue #102). OFF by
        # default — see :meth:`auto_persist_sessions`.
        self._auto_persist_sessions: bool = False
        # Standard catalogue tools accumulated via :meth:`tool` / :meth:`tools`
        # (issue #81). Each is a ``StandardTool``-shaped object exposing
        # ``implementation`` (a :class:`Tool`) and ``schema`` (a
        # :class:`ToolSchema`). They are drained into a populated
        # :class:`StandardToolRegistry` by :meth:`drain_tools_into_registry`,
        # applying last-wins upsert. Typed structurally (the concrete
        # ``StandardTool`` lives in ``spore_tools`` and must not be imported
        # here — that would invert the package dependency).
        self._standard_tools: list[Any] = []
        # Per-toolset catalogue tools accumulated via :meth:`toolset_tools`
        # (Issue 2: per-node toolset scoping), keyed by the non-empty
        # ``ToolsetRef`` handle. At :meth:`build_config` each key's tools are
        # folded into its own populated :class:`StandardToolRegistry` (last-wins
        # upsert) and stored in ``HarnessConfig.toolset_catalogues``. Empty (the
        # default) keeps the global-only ``.tools()`` path byte-for-byte. Mirrors
        # Rust's ``HarnessBuilder::toolset_tools``.
        self._toolset_tools: dict[str, list[Any]] = {}
        # Optional operating system prompt prepended to each turn's assembled
        # context (issue #91) when the context manager renders none. ``None``
        # (the default) preserves today's behaviour. See :meth:`system_prompt`.
        self._system_prompt: str | None = None
        # Guides injected structurally into every turn via ``ContextSources``
        # (issue #115 / #9). Empty (the default) preserves today's behaviour. See
        # :meth:`guide`.
        self._guides: list[Guide] = []
        # Optional skill catalog (issue #115 / SC-26). See :meth:`skills`.
        self._skills: SkillCatalog | None = None
        # Optional memory source (issue #160). ``None`` (the default) leaves
        # memory empty. See :meth:`memory`.
        self._memory: MemoryConfig | None = None
        # Authoritative per-run model sampling/decoding parameters (issue #93).
        # Defaults to ``ModelParams()``. See :meth:`model_params`.
        self._model_params: ModelParams = ModelParams()
        # Pluggable chunk source for the #79 prompt assembly engine. Defaults to
        # an empty in-memory provider so existing callers are unaffected. Typed
        # ``Any`` to avoid importing ``prompt_assembly`` at module load (that
        # module imports ``SessionId``/``TaskId`` from here).
        self._chunk_provider: Any | None = None
        # Shared adaptive prompt-based tool-calling escalation flag (#111). Set
        # only by :meth:`conversational`, which wraps the agent's model in an
        # :class:`AdaptiveToolCallModelInterface` holding the same holder. ``None``
        # on the plain constructor leaves adaptive escalation off.
        self._prompt_tool_call_flag: PromptToolCallFlag | None = None
        # Per-kind consult handlers (issue #114). Empty by default — consults
        # degrade gracefully (R6). Populated via :meth:`consult_handler`.
        self._consult_handlers: dict[str, ConsultHandlerEntry] = {}
        # ExecutionRegistry (#120). Defaults to empty so legacy callers stay
        # byte-identical (Option B). Replaced wholesale via :meth:`registry` or
        # incrementally via the per-key convenience setters
        # (:meth:`register_agent` / :meth:`register_toolset` /
        # :meth:`register_schema` / :meth:`register_verifier` /
        # :meth:`register_strategy`).
        from .execution_registry import ExecutionRegistry as _ExecutionRegistry

        self._registry: ExecutionRegistry = _ExecutionRegistry.empty()
        # HITL-vs-AFK escalation knob (#120). The builder picks the explicit
        # default (``SurfaceToHuman``); the type itself carries no default. See
        # :meth:`escalation_mode`.
        from .execution_registry import EscalationModeSurfaceToHuman as _SurfaceToHuman

        self._escalation_mode: EscalationMode = _SurfaceToHuman()

    @classmethod
    def conversational(cls, model: ModelInterface) -> HarnessBuilder:
        """Assemble a minimal conversational harness from a model — no tools, no
        filesystem.

        This is the few-lines path: it defaults every required component so you
        can go from a model to a running harness in one call. The defaults are a
        :class:`~spore_core.agent.ModelAgent` over ``model``, an
        :class:`EmptyToolRegistry`, a :class:`NullSandbox` (permits tool-call
        validation and applies no path/process isolation — fine for a tool-less
        agent), a
        :class:`~spore_core.context.StandardContextManager` with a null cache
        provider and default compaction, and :class:`CompleteOnFinalResponse`
        termination (the model's first final response is the result).

        Every default is overridable: add catalogue tools with :meth:`tool` /
        :meth:`tools`, swap the sandbox with :meth:`sandbox`, supply your own
        harness-loop tool registry with :meth:`tool_registry`, or construct the
        builder via ``__init__`` directly. Mirrors Rust's
        ``HarnessBuilder::conversational``.

        Example::

            harness = HarnessBuilder.conversational(
                OllamaModelInterface("llama3.2")
            ).build()
            result = await harness.run(
                HarnessRunOptions(Task.simple("Reply with a friendly greeting."))
            )
        """
        from .agent import AgentId, ModelAgent
        from .cache_provider import NullCacheProvider
        from .compaction_adapter import into_harness_adapter
        from .context import CompactionConfig, StandardContextManager

        # Adaptive prompt-based tool-calling fallback (#111). Wrap the agent's
        # model in an AdaptiveToolCallModelInterface gated on a shared flag the
        # run loop flips when it detects a prose response where a tool call was
        # expected. The context manager keeps the RAW model — injection/parsing
        # only happen on the agent's tool-requesting turns.
        prompt_tool_call_flag = PromptToolCallFlag(value=False)
        wrapped_model = AdaptiveToolCallModelInterface(model, prompt_tool_call_flag)

        agent = ModelAgent(AgentId("agent"), wrapped_model)
        tool_registry = EmptyToolRegistry()
        sandbox = NullSandbox()
        context_manager = into_harness_adapter(
            StandardContextManager(
                model,
                NullCacheProvider(),
                CompactionConfig(),
            )
        )
        termination_policy = CompleteOnFinalResponse()
        builder = cls(
            agent=agent,
            tool_registry=tool_registry,
            sandbox=sandbox,
            context_manager=context_manager,
            termination_policy=termination_policy,
        )
        builder._prompt_tool_call_flag = prompt_tool_call_flag
        return builder

    @staticmethod
    def _preset_auto_continue() -> EscalationMode:
        """``AutoContinue`` with the preset defaults
        (:data:`PRESET_MAX_AUTO_GRANTS` × :data:`PRESET_STEPS_PER_GRANT`, no
        ``on_grant`` observer) — the "autonomous but capped" policy both
        autonomous presets share (SC-8). Mirrors Rust's
        ``HarnessBuilder::preset_auto_continue``."""
        from .execution_registry import EscalationModeAutoContinue

        return EscalationModeAutoContinue(
            max_grants=PRESET_MAX_AUTO_GRANTS,
            steps_per_grant=PRESET_STEPS_PER_GRANT,
        )

    @classmethod
    def coding_agent(cls, model: ModelInterface, workspace: str | Path) -> HarnessBuilder:
        """Assemble an autonomous **coding agent** over a workspace directory
        (SC-8) — the looper preset.

        Builds on :meth:`conversational` and wires the bits a coding agent
        always needs: a **read-write**
        :class:`~spore_core.sandbox.WorkspaceScopedSandbox` rooted at
        ``workspace``, the full ``StandardTools.coding_set()``
        (read/write/edit/list/grep/find + ``bash`` + ``send_message`` +
        web/memory/task-list), the built-in :data:`CODING_AGENT_SYSTEM_PROMPT`,
        and :class:`~spore_core.execution_registry.EscalationModeAutoContinue`
        (autonomous-but-capped — it keeps working through a spent step budget
        instead of pausing, so there is no consumer drive loop to hand-roll;
        SC-5).

        **Window sizing (SC-4/SC-6).** Size the model's context window ONCE on
        the model before passing it in (e.g.
        ``OllamaModelInterface.with_context_window``): the preset's
        ``conversational`` context manager auto-derives its compaction budget
        from ``provider().context_window`` — so one call sizes both halves and
        no manual :meth:`context_manager` is needed.

        Raises :class:`~spore_core.sandbox.SandboxBuildError` if the workspace
        path can't be resolved (it must exist and canonicalize — the sandbox
        requirement). This is the fallible-constructor convention matching
        Rust's ``Result<Self, BuildError>``: an unresolvable workspace surfaces a
        typed error, never a bare crash. The strategy is per-run: pass a
        ``ReAct`` / ``PlanExecute`` :class:`Task` to :meth:`StandardHarness.run`;
        the empty agent/toolset handles on its leaves resolve to this preset's
        defaults. Mirrors Rust's ``HarnessBuilder::coding_agent``.

        Example::

            model = OllamaModelInterface("gemma4:e4b").with_context_window(256_000)
            harness = HarnessBuilder.coding_agent(model, "/path/to/project").build()
        """
        # ``coding_set()`` lives in the ``spore_tools`` package, which depends on
        # ``spore_core`` — so importing it at module scope would be a cycle. A
        # deferred import inside the preset breaks the cycle: ``spore_tools`` is a
        # workspace sibling, importable at call time.
        from spore_tools import StandardTools

        from .sandbox import WorkspaceScopedSandbox

        # A read-write scoped sandbox over ``workspace`` (``read_only`` defaults
        # to ``False``); construction canonicalizes the root and raises
        # ``SandboxBuildError`` if it can't resolve.
        sandbox = WorkspaceScopedSandbox(WorkspaceConfig(root=Path(workspace)))
        return (
            cls.conversational(model)
            .sandbox(sandbox)
            .tools(StandardTools.coding_set())
            .system_prompt(CODING_AGENT_SYSTEM_PROMPT)
            .escalation_mode(cls._preset_auto_continue())
        )

    @classmethod
    def hill_climber(cls, model: ModelInterface, evaluator: Any) -> HarnessBuilder:
        """Assemble an autonomous **hill-climbing agent** (SC-8) — the cordyceps
        preset.

        Builds on :meth:`conversational` and registers the scoring ``evaluator``
        (required for the
        :class:`~spore_core.execution_registry.HillClimbingConfig` loop strategy)
        under the default ("") handle, plus
        :class:`~spore_core.execution_registry.EscalationModeAutoContinue`
        (autonomous-but-capped; SC-5) so a spent per-iteration build budget keeps
        working instead of pausing.

        Unlike :meth:`coding_agent` this does NOT install a sandbox or tools —
        hill-climbing workspaces vary (some climb a prose artifact, some climb
        files), and the build task's system prompt is task-specific. Add them
        with :meth:`sandbox` / :meth:`tools` / :meth:`system_prompt` as the climb
        requires. Size the model's window on the model first (SC-4/SC-6), as in
        :meth:`coding_agent`. Non-fallible. Mirrors Rust's
        ``HarnessBuilder::hill_climber``.

        Example::

            model = OllamaModelInterface("gemma4:e4b").with_context_window(256_000)
            # Add a workspace ``.sandbox(..)`` + ``.tools(..)`` for the climb.
            harness = HarnessBuilder.hill_climber(model, evaluator).build()
        """
        return (
            cls.conversational(model)
            .metric_evaluator(evaluator)
            .escalation_mode(cls._preset_auto_continue())
        )

    def chunk_provider(self, provider: Any) -> HarnessBuilder:
        """Set the chunk provider for the #79 prompt assembly engine. Defaults
        to an empty :class:`~spore_core.prompt_assembly.InMemoryChunkProvider`
        when unset. Mirrors Rust's ``HarnessBuilder::chunk_provider``."""
        self._chunk_provider = provider
        return self

    def chunks(self, chunks: Iterable[Any]) -> HarnessBuilder:
        """Convenience: register chunks inline without constructing a provider
        (issue #79). Resolves to an
        :class:`~spore_core.prompt_assembly.InMemoryChunkProvider` internally.
        Mirrors Rust's ``HarnessBuilder::chunks``."""
        from .prompt_assembly import InMemoryChunkProvider

        self._chunk_provider = InMemoryChunkProvider(list(chunks))
        return self

    def tool(self, tool: Any) -> HarnessBuilder:
        """Accumulate a single ``StandardTool`` (issue #81, Q1/Q2) — an object
        bundling ``implementation`` + ``schema``. The bundle is destructured
        when the registry is built via :meth:`drain_tools_into_registry`.
        Registration applies LAST-WINS upsert: a later ``.tool()`` with the same
        name overrides an earlier one (e.g. a custom tool after a preset)."""
        self._standard_tools.append(tool)
        return self

    def tools(self, tools: Iterable[Any]) -> HarnessBuilder:
        """Accumulate many ``StandardTool``s at once (e.g. a preset like
        ``StandardTools.coding_set()``). Order is preserved, so last-wins upsert
        still applies across the batch."""
        self._standard_tools.extend(tools)
        return self

    def toolset_tools(self, key: str, tools: Iterable[Any]) -> HarnessBuilder:
        """Register catalogue tools SCOPED to a single named toolset handle
        (Issue 2: per-node toolset scoping). Mirrors :meth:`tools`, but instead of
        folding into the ONE global catalogue these tools are accumulated into a
        per-``key`` bucket (additive across calls) and, at
        :meth:`build_config`, folded into a per-key
        :class:`StandardToolRegistry` (last-wins upsert) stored in
        ``HarnessConfig.toolset_catalogues``.

        At dispatch, a leaf whose ``toolset`` handle equals ``key`` resolves ONLY
        this catalogue (bridged per-run via
        :class:`~spore_core.tool_registry.RealToolRegistry` with the run's
        sandbox/session/storage), so it cannot reach another node's tools. A leaf
        with an EMPTY (``""``) handle, or a non-empty handle with no entry here,
        falls back to the global :meth:`tools` catalogue / ``tool_registry`` seam
        (back-compat).

        ``key`` should be a NON-EMPTY handle string; the empty handle is reserved
        for the global-catalogue fallback. Mirrors Rust's
        ``HarnessBuilder::toolset_tools``."""
        self._toolset_tools.setdefault(key, []).extend(tools)
        return self

    def tool_registry(self, tool_registry: ToolRegistry) -> HarnessBuilder:
        """Override the harness-loop tool registry (issue #4 seam).

        Use this to supply your own :class:`ToolRegistry` implementation — e.g. a
        set of custom tools — on top of a preset like :meth:`conversational`::

            harness = (
                HarnessBuilder.conversational(model)
                .tool_registry(LocalTools())
                .build()
            )

        The registry's :meth:`~ToolRegistry.schemas` are delivered to the model
        automatically each turn, and :meth:`~ToolRegistry.dispatch` is called
        when the model requests a tool. Mirrors Rust's
        ``HarnessBuilder::tool_registry``."""
        self._tool_registry = tool_registry
        return self

    def context_manager(self, context_manager: ContextManager) -> HarnessBuilder:
        """Override the :class:`ContextManager` that assembles per-turn context
        and drives compaction.

        :meth:`conversational` installs a
        :class:`~spore_core.context.StandardContextManager` with
        ``CompactionConfig()`` defaults (compaction at 80% of a 200K window);
        supply your own (e.g. a lower ``threshold``) to make compaction fire
        earlier for models with a smaller context window::

            harness = (
                HarnessBuilder.conversational(model)
                .context_manager(my_low_threshold_manager)
                .build()
            )

        Mirrors Rust's ``HarnessBuilder::context_manager``."""
        self._context_manager = context_manager
        return self

    def sandbox(self, sandbox: SandboxProvider) -> HarnessBuilder:
        """Override the :class:`SandboxProvider` — the only path tools have to
        the environment (filesystem, process exec).

        :meth:`conversational` defaults to a null sandbox that denies
        environment access — fine for pure-compute tools, but catalogue file
        tools (``read_file`` / ``write_file`` / ``list_dir``) operate *through*
        the sandbox, so an agent that touches a real directory needs a
        workspace-scoped sandbox here::

            harness = (
                HarnessBuilder.conversational(model)
                .sandbox(workspace)
                .tools(StandardTools.coding_set())
                .build()
            )
        """
        self._sandbox = sandbox
        return self

    def system_prompt(self, system_prompt: str) -> HarnessBuilder:
        """Set an operating system prompt prepended to each turn's assembled
        context (issue #91).

        The standard compaction context manager renders no system prompt, so
        without this the model receives only the task as a user message and no
        guidance on how to behave. When set, the run loop inserts this text as a
        leading :class:`~spore_core.model.Role.SYSTEM` message each turn — but
        only when the assembled context does not already start with one, so a
        context manager that renders its own system prompt is preserved.
        ``None`` (the default) preserves today's behaviour."""
        self._system_prompt = system_prompt
        return self

    def guide(self, guide: Guide) -> HarnessBuilder:
        """Register a :class:`~spore_core.context.Guide` (skill/playbook/domain
        knowledge) injected structurally into every turn's assembled context via
        the rich ``ContextSources`` seam (issue #115 / SC-26 / #9). Unlike an
        ad-hoc User-message prepend, the guide is rendered into the leading System
        block by the production context manager. Call repeatedly to register
        several; order is preserved. Mirrors Rust's ``HarnessBuilder::guide``."""
        self._guides.append(guide)
        return self

    def guides(self, guides: Iterable[Guide]) -> HarnessBuilder:
        """Register several :class:`~spore_core.context.Guide`s at once (issue
        #115 / SC-26). Appends to any already registered via :meth:`guide`.
        Mirrors Rust's ``HarnessBuilder::guides``."""
        self._guides.extend(guides)
        return self

    def skills(self, catalog: SkillCatalog) -> HarnessBuilder:
        """Register a :class:`~spore_core.skills.SkillCatalog` (issue #115 /
        SC-26). This both (a) injects the catalog's manifest + active skill bodies
        into every turn's structural context (progressive disclosure) and (b)
        registers the ``load_skill`` tool, sharing the catalog's active set, so
        the model can activate a skill on demand. Replaces the architect-side
        skill-injecting context-manager shim. Mirrors Rust's
        ``HarnessBuilder::skills``."""
        self._standard_tools.append(catalog.load_skill_tool())
        self._skills = catalog
        return self

    def memory(self, memory: MemoryConfig) -> HarnessBuilder:
        """Wire a memory source (issue #160 / SC-26 follow-up). The harness
        queries the provider each turn and injects the relevant memories into
        the structural System block — alongside guides + skills via the rich
        ``ContextSources`` seam — with no consumer-side context-manager shim.

        Pass a :class:`MemoryConfig` to control the query policy, e.g.
        ``MemoryConfig(provider, min_relevance=0.6, max_items=5)``. Without an
        explicit ``query`` text the current task instruction is used, so
        retrieved memory tracks what the agent is working on. Not set (the
        default) leaves memory empty, byte-identical to the pre-#160 behaviour.
        Mirrors Rust's ``HarnessBuilder::memory``."""
        self._memory = memory
        return self

    def model_params(self, params: ModelParams) -> HarnessBuilder:
        """Set the authoritative model sampling/decoding parameters for the
        whole run (issue #93).

        These params are authoritative: the harness replaces each turn's
        ``Context.params`` with this value UNCONDITIONALLY (builder params win)
        right before the request is built, so the configured params reach every
        agent turn that requests tools — the ReAct loop, the PlanExecute plan
        phase, the execute sub-loop, and the streaming path alike. (The internal
        compaction/summarization turn is intentionally left on defaults; it
        requests no tools, so decoding params are a no-op there.)

        Enabling :attr:`~spore_core.model.ModelParams.structured_tool_calls`
        trades interleaved reasoning for one schema-constrained tool call per
        turn — useful for small local models that otherwise emit malformed tool
        calls. See :attr:`~spore_core.model.ModelParams.structured_tool_calls`
        for the full behaviour contract. Defaults to ``ModelParams()``."""
        self._model_params = params
        return self

    def registry(self, registry: ExecutionRegistry) -> HarnessBuilder:
        """Replace the whole :class:`ExecutionRegistry` (issue #120). The
        registry resolves the serializable strategy handles at run entry; an
        unresolved handle becomes a startup error. Mirrors Rust's
        ``HarnessBuilder::registry``."""
        self._registry = registry
        return self

    def register_agent(self, key: str, agent: Agent) -> HarnessBuilder:
        """Register a named agent in the :class:`ExecutionRegistry` (issue
        #120). Convenience over :meth:`registry`."""
        self._registry.agents[key] = agent
        return self

    def register_toolset(self, key: str, toolset: ToolRegistry) -> HarnessBuilder:
        """Register a named toolset in the :class:`ExecutionRegistry` (#120)."""
        self._registry.toolsets[key] = toolset
        return self

    def register_schema(self, key: str, schema: Any) -> HarnessBuilder:
        """Register a named JSON schema in the :class:`ExecutionRegistry` (#120)."""
        self._registry.schemas[key] = schema
        return self

    def register_verifier(self, key: str, verifier: Any) -> HarnessBuilder:
        """Register a named verifier in the :class:`ExecutionRegistry` (#120)."""
        self._registry.verifiers[key] = verifier
        return self

    def register_strategy(self, key: str, strategy: RunStrategy) -> HarnessBuilder:
        """Register a custom strategy in the :class:`ExecutionRegistry` under
        ``key`` (issue #120). Resolvable later via
        ``registry.resolve_strategy(StrategyRefCustom(value=key))``."""
        self._registry.custom[key] = strategy
        return self

    def escalation_mode(self, mode: EscalationMode) -> HarnessBuilder:
        """Select the HITL-vs-AFK budget-escalation behaviour (issue #120, PRD
        goal #7). Defaults to ``SurfaceToHuman``. Stored only this slice (#130
        consumes it). Mirrors Rust's ``HarnessBuilder::escalation_mode``."""
        self._escalation_mode = mode
        return self

    def drain_tools_into_registry(self) -> StandardToolRegistry:
        """Drain the accumulated catalogue tools into a populated
        :class:`StandardToolRegistry`, registering each with last-wins upsert
        (issue #81, Q1/Q2). Consumes the accumulated set. Returns an empty
        registry if no catalogue tools were added. Mirrors Rust's
        ``HarnessBuilder::drain_tools_into_registry``."""
        from .tool_registry import StandardToolRegistry as _Registry

        reg = _Registry()
        for t in self._standard_tools:
            reg.register(t.implementation, t.schema)
        self._standard_tools = []
        return reg

    def storage(self, storage: StorageProvider) -> HarnessBuilder:
        """Inject a :class:`StorageProvider` (issue #73). Defaults to an
        all-no-op provider when unset. v1 is expose-only — the harness loop does
        not read/write sessions internally; callers reach the four domain stores
        via :meth:`StandardHarness.storage` / :meth:`StandardHarness.session_store`."""
        self._storage = storage
        return self

    def project_id(self, project_id: ProjectId) -> HarnessBuilder:
        """Override the STABLE durable-storage project namespace (issue #142).

        When unset, :meth:`build_config` derives it from
        ``sandbox.workspace_root()`` (decision 5), canonicalizing the path first.
        Set this explicitly when the workspace root is not a stable on-disk path
        (e.g. a fixture-replay tempdir, or to pin a known project across
        processes). See :attr:`HarnessConfig.project_id`. Mirrors Rust's
        ``HarnessBuilder::project_id``."""
        self._project_id = project_id
        return self

    def auto_persist_sessions(self, enabled: bool) -> HarnessBuilder:
        """Opt into conversation-history threading via the :class:`SessionStore`
        (issue #102). OFF by default.

        When enabled the harness, for ReAct / SelfVerifying runs, **auto-loads**
        the prior :class:`SessionState` for the run's ``session_id`` from the
        store at the start of a fresh :meth:`StandardHarness.run` (so a caller
        can resume by id without threading messages by hand), and **auto-persists**
        the post-run state back to the store at the terminal seam (one write per
        ``run()`` / ``resume()``). An explicit
        ``HarnessRunOptions(session_state=..)`` always wins over the auto-load.
        Ralph / HillClimbing discard incoming session state by design, so they
        are NOT auto-loaded. Storage errors are swallowed-and-logged: a load
        failure starts fresh, a persist failure is ignored — never surfaced as a
        halt. Pair with a real (in-memory / filesystem) :meth:`storage` provider;
        without one the default all-no-op store makes this a no-op. Mirrors
        Rust's ``HarnessBuilder::auto_persist_sessions``."""
        self._auto_persist_sessions = enabled
        return self

    def planner_agent(self, planner_agent: Agent) -> HarnessBuilder:
        """Inject an alternate agent for the PlanExecute plan phase (issue #70,
        Q1). When set and the loop strategy is ``PlanExecute``, the one-shot
        plan turn runs on this agent instead of the default agent."""
        self._planner_agent = planner_agent
        return self

    def consult_handler(
        self,
        kind: str,
        handler: Harness,
        budget: int,
        overflow: ConsultOverflowPolicy,
    ) -> HarnessBuilder:
        """Register a per-kind consult handler (issue #114). Analogous to
        :meth:`planner_agent`. When a worker pauses with a
        :class:`ConsultRequest` whose ``kind`` matches ``kind``, the orchestrator
        (via ``SubagentTool``) runs ``handler`` on the request and resumes the
        worker. Up to ``budget`` consults of this kind are mediated before
        ``overflow`` applies. Without any registered handler, consults degrade
        gracefully (R6): a standalone worker surfaces :class:`RunResultConsult`
        unchanged. Empty by default."""
        self._consult_handlers[kind] = ConsultHandlerEntry(
            handler=handler, budget=budget, overflow=overflow
        )
        return self

    def verifier(self, verifier: Any) -> HarnessBuilder:
        """Inject the SelfVerifying oracle (issue #61, D2). REQUIRED for the
        ``SelfVerifying`` strategy: without it the run halts with
        :class:`HaltReasonSelfVerifyMisconfigured` (D4). Its ``max_iterations()``
        caps the build↔evaluate round-trips (D3). Mirrors Rust's
        ``HarnessBuilder::verifier``."""
        self._verifier = verifier
        return self

    def evaluator_agent(self, evaluator_agent: Agent) -> HarnessBuilder:
        """Inject an alternate agent for the SelfVerifying evaluate phase (issue
        #61, D2). Mirrors :meth:`planner_agent`: when set and the loop strategy
        is ``SelfVerifying``, the evaluate phase runs on this agent instead of
        the default agent. Mirrors Rust's ``HarnessBuilder::evaluator_agent``."""
        self._evaluator_agent = evaluator_agent
        return self

    def metric_evaluator(self, evaluator: Any) -> HarnessBuilder:
        """Inject the HillClimbing scoring strategy (issue #60). REQUIRED for the
        ``HillClimbing`` strategy: without it the run halts with
        :class:`HaltReasonHillClimbingMisconfigured` (Decision 6) — a typed halt,
        never a raise. The harness calls ``evaluate`` once per iteration (with
        iteration 0 the pure baseline) and routes the result through
        :func:`~spore_core.metric.should_keep`. Mirrors Rust's
        ``HarnessBuilder::metric_evaluator``."""
        self._metric_evaluator = evaluator
        return self

    def hooks(self, hooks: HookChain) -> HarnessBuilder:
        """Inject a lifecycle hook chain (issue #69). The harness fires its
        registered ``Stop`` hooks when a loop strategy believes it is done."""
        self._hooks = hooks
        return self

    def max_stop_blocks(self, max_blocks: int) -> HarnessBuilder:
        """Set the maximum consecutive Stop-hook blocks honored per run before
        the loop terminates anyway (issue #69). Defaults to ``8``."""
        self._max_stop_blocks = max_blocks
        return self

    def error_loop_threshold(self, n: int) -> HarnessBuilder:
        """Set the consecutive-recoverable-tool-error breaker threshold ``N``
        (issue #137). ``N`` consecutive identical-argument recoverable errors
        from one tool inject ONE corrective message; ``2 * N`` hard-stops the
        loop with :class:`HaltReasonToolErrorLoop` (the ``2x`` multiplier is
        fixed). Defaults to ``3``; ``0`` disables the breaker. Mirrors Rust's
        ``HarnessBuilder::error_loop_threshold``."""
        self._error_loop_threshold = n
        return self

    def enforce_output_schemas(self, enforce: bool) -> HarnessBuilder:
        """Turn ON output-schema delivery + enforcement for ``ReactConfig`` leaves
        carrying ``output`` set (issue #139 — MIGRATION GATE). Defaults to
        ``False`` (OFF), which keeps existing replay fixtures byte-identical. See
        :attr:`HarnessConfig.enforce_output_schemas`."""
        self._enforce_output_schemas = enforce
        return self

    def output_schema_max_retries(self, n: int) -> HarnessBuilder:
        """Set the number of EXTRA terminal-validation retry turns ``N`` granted
        when :meth:`enforce_output_schemas` is ON (issue #139). Total attempts =
        ``1 + N``. Defaults to ``2``. See
        :attr:`HarnessConfig.output_schema_max_retries`."""
        self._output_schema_max_retries = n
        return self

    def max_resets(self, max_resets: int) -> HarnessBuilder:
        """Set the Ralph outer-loop reset cap (issue #58, B3) — the maximum
        number of context windows the ``Ralph`` strategy runs before halting
        with :class:`HaltReasonRalphCompletionUnmet`. Independent of
        ``budget.max_turns``. Defaults to ``3``. Mirrors Rust's
        ``HarnessBuilder::max_resets``."""
        self._max_resets = max_resets
        return self

    def vcs_provider(self, provider: VcsProvider) -> HarnessBuilder:
        """Inject a :class:`VcsProvider` for the ``Ralph`` loop strategy (issue
        #58 v2). When set, Ralph's reload phase calls :meth:`VcsProvider.log`
        and injects a delimited "Recent VCS history:" section into each fresh
        context window's seed. Unset (the default) ⇒ no git section is injected
        and Ralph behaves exactly like v1. Mirrors Rust's
        ``HarnessBuilder::vcs_provider``."""
        self._vcs_provider = provider
        return self

    def compaction_verifier(self, verifier: CompactionVerifier) -> HarnessBuilder:
        """Inject a post-compaction verifier (issue #46). Defaults to
        ``KeyTermVerifier``."""
        self._compaction_verifier = verifier
        return self

    def max_compaction_attempts(self, attempts: int) -> HarnessBuilder:
        """Set the maximum number of compaction-summary attempts before
        accepting a failing summary anyway (issue #46). Defaults to ``2``."""
        self._max_compaction_attempts = attempts
        return self

    def middleware(self, middleware: MiddlewareChain) -> HarnessBuilder:
        """Inject a middleware chain."""
        self._middleware = middleware
        return self

    def observability(self, observability: ObservabilityProvider) -> HarnessBuilder:
        """Inject an observability provider. The harness loop emits real spans
        through it (turn spans, tool-call spans) and flushes on terminal
        outcomes."""
        self._observability = observability
        return self

    def with_observability_outbox(self, root: str | Path) -> HarnessBuilder:
        """Construct and inject a durable-outbox observability provider rooted
        at ``root`` (typically the ``.spore`` directory). Honors the
        ``SPORE_OTLP_ENDPOINT`` env var for OTLP forwarding (issue #33)."""
        provider = OutboxObservabilityProvider(OutboxConfig(root=Path(root)))
        return self.observability(provider)

    def pricing(self, pricing: PricingTable) -> HarnessBuilder:
        """Set the token → USD pricing table used to stamp ``cost_usd`` on
        turn spans."""
        self._pricing = pricing
        return self

    def content_capture(self, content_capture: ContentCaptureConfig) -> HarnessBuilder:
        """Set the LLM-native content-capture config (issue #64). OFF by
        default. Use :meth:`ContentCaptureConfig.from_env` to honor
        ``SPORE_TRACE_CONTENT`` / ``SPORE_TRACE_CONTENT_MAX_LEN``."""
        self._content_capture = content_capture
        return self

    def build_config(self) -> HarnessConfig:
        """Assemble the :class:`HarnessConfig` without wrapping it in a
        harness."""
        # Fold catalogue tools accumulated via ``.tool()`` / ``.tools()`` into a
        # populated :class:`StandardToolRegistry` (last-wins upsert). The run
        # loop bridges it per-run — ``build()`` can't, because the
        # :class:`ToolContext` is keyed by the run's :class:`SessionId`, unknown
        # until ``run()``.
        catalogue_registry: StandardToolRegistry | None = (
            self.drain_tools_into_registry() if self._standard_tools else None
        )
        # Issue 2: fold each per-toolset bucket into its own populated
        # :class:`StandardToolRegistry` (last-wins upsert), keyed by the toolset
        # handle. Bridged per-run at dispatch — same as the global catalogue.
        from .tool_registry import StandardToolRegistry as _Registry

        toolset_catalogues: dict[str, StandardToolRegistry] = {}
        for key, tools in self._toolset_tools.items():
            reg = _Registry()
            for t in tools:
                reg.register(t.implementation, t.schema)
            toolset_catalogues[key] = reg
        self._toolset_tools = {}
        # When catalogue tools are present and the caller wired no storage,
        # default to an in-memory provider (not the all-no-op default) so that
        # session-aware tools (todo_write, memory, task_list) actually persist
        # within the run. Pure tools (read_file/write_file via the sandbox) are
        # unaffected either way.
        storage = self._storage
        if storage is None and (catalogue_registry is not None or toolset_catalogues):
            from .storage import InMemoryStorageProvider

            storage = StorageProvider.single(InMemoryStorageProvider())
        return HarnessConfig(
            agent=self._agent,
            tool_registry=self._tool_registry,
            sandbox=self._sandbox,
            context_manager=self._context_manager,
            termination_policy=self._termination_policy,
            middleware=self._middleware,
            observability=self._observability,
            compaction_verifier=self._compaction_verifier,
            max_compaction_attempts=self._max_compaction_attempts,
            pricing=self._pricing,
            content_capture=self._content_capture,
            max_stop_blocks=self._max_stop_blocks,
            error_loop_threshold=self._error_loop_threshold,
            enforce_output_schemas=self._enforce_output_schemas,
            output_schema_max_retries=self._output_schema_max_retries,
            max_resets=self._max_resets,
            vcs_provider=self._vcs_provider,
            hooks=self._hooks,
            verifier=self._verifier,
            storage=storage,
            # #142: an explicit project_id wins; ``None`` lets HarnessConfig derive
            # it from ``sandbox.workspace_root()`` (decision 5).
            project_id=self._project_id,
            chunk_provider=self._chunk_provider,
            metric_evaluator=self._metric_evaluator,
            catalogue_registry=catalogue_registry,
            toolset_catalogues=toolset_catalogues,
            system_prompt=self._system_prompt,
            guides=self._guides,
            skills=self._skills,
            memory=self._memory,
            model_params=self._model_params,
            auto_persist_sessions=self._auto_persist_sessions,
            prompt_tool_call_flag=self._prompt_tool_call_flag,
            consult_handlers=self._consult_handlers,
            registry=self._registry,
            escalation_mode=self._escalation_mode,
        )

    def build(self) -> StandardHarness:
        """Assemble a ready-to-run :class:`StandardHarness`."""
        return StandardHarness(self.build_config())


# ============================================================================
# Ralph loop strategy (issue #58) — filesystem-backed completion contract
# ============================================================================


class RalphProgress(_Model):
    """The Ralph progress checkpoint (issue #58, B2/B4; relocated by #142). The
    agent writes this each context window to record what it has finished and
    what remains. ``complete: true`` with an empty ``remaining`` ⇒ progress
    satisfied. #142: the checkpoint moved off ``.spore/progress.json`` onto the
    durable project-id RunStore. Mirrors Rust's ``RalphProgress``."""

    complete: bool = False
    remaining: list[str] = Field(default_factory=list)


class RalphFeatureEntry(_Model):
    """One entry of the Ralph feature-list checkpoint — the
    :class:`~spore_core.termination.FeatureListCheck` schema (issue #58, B2; #142
    relocated it onto the project-id RunStore). Any ``passes: false`` ⇒
    incomplete. Mirrors Rust's ``RalphFeatureEntry``."""

    name: str
    passes: bool = False


async def _ralph_completion_status(run_store: RunStore, project: ProjectId) -> str | None:
    """Ralph external completion check (issue #58, B1; #142 — async, store-backed).
    Reads the Ralph checkpoint from the durable project-id :class:`RunStore`
    (NOT the ``.spore/`` filesystem) and reports whether the task is complete.
    Returns ``None`` when complete, the failure reason when tasks remain. This is
    the SAME logic the registered :class:`RalphStopHook` applies — one source of
    truth. Mirrors Rust's ``ralph_completion_status``.

    The reason strings stay byte-identical to the legacy ``.spore/``-path
    messages so the surfaced :class:`HaltReasonRalphCompletionUnmet` text is
    stable across the storage relocation.

    Contract (#142, decision 3 — the checkpoint MOVED off the ``.spore/``
    filesystem onto the durable project-id store):

    * :data:`~spore_core.storage.RALPH_PROGRESS_KEY` →
      ``{"complete": bool, "remaining": [str]}``. ``complete: true`` with an
      empty ``remaining`` ⇒ progress satisfied. Absent / unreadable / invalid ⇒
      incomplete (so the agent learns to write it).
    * :data:`~spore_core.storage.RALPH_FEATURE_LIST_KEY` → a JSON array of
      ``{"name", "passes"}``. Any ``passes: false`` ⇒ incomplete. An ABSENT
      feature list is tolerated (progress is the primary signal); an invalid one
      is not.
    """
    ns = project_namespace(project)
    try:
        progress_value = await run_store.get(ns, RALPH_PROGRESS_KEY)
    except StorageError:
        progress_value = None
    if progress_value is None:
        # Absent OR a storage error ⇒ incomplete (the agent must write it).
        return ".spore/progress.json missing"
    try:
        progress = RalphProgress.model_validate(progress_value)
    except ValueError as e:
        return f".spore/progress.json invalid JSON: {e}"
    if not progress.complete:
        if not progress.remaining:
            return "task not marked complete"
        return f"remaining: {', '.join(progress.remaining)}"
    if progress.remaining:
        return f"remaining: {', '.join(progress.remaining)}"

    # Progress says done — corroborate against the feature list when present.
    try:
        feature_value = await run_store.get(ns, RALPH_FEATURE_LIST_KEY)
    except StorageError:
        feature_value = None
    if feature_value is None:
        return None
    try:
        entries = TypeAdapter(list[RalphFeatureEntry]).validate_python(feature_value)
    except ValueError as e:
        return f".spore/feature_list.json invalid JSON: {e}"
    incomplete = [e.name for e in entries if not e.passes]
    if incomplete:
        return f"incomplete features: {', '.join(incomplete)}"
    return None


async def _ralph_reload_context(run_store: RunStore, project: ProjectId) -> str | None:
    """Build the reload context block injected into each fresh Ralph context
    window (issue #58, R3/B4; #142 — async, store-backed). Reads the checkpoint
    from the durable project-id :class:`RunStore` (not the ``.spore/``
    filesystem) and returns the verbatim progress + feature-list JSON (when
    present) so the re-seeded window knows what is done and what remains. Returns
    ``None`` when neither checkpoint key is set. The "Reloaded .spore/…" prefix
    is retained so the seeded prompt text is byte-stable across the relocation.
    Mirrors Rust's ``ralph_reload_context``."""
    ns = project_namespace(project)
    parts: list[str] = []
    try:
        progress_value = await run_store.get(ns, RALPH_PROGRESS_KEY)
    except StorageError:
        progress_value = None
    if progress_value is not None:
        raw = json.dumps(progress_value, separators=(",", ":"))
        parts.append(f"Reloaded .spore/progress.json:\n{raw.strip()}")
    try:
        feature_value = await run_store.get(ns, RALPH_FEATURE_LIST_KEY)
    except StorageError:
        feature_value = None
    if feature_value is not None:
        raw = json.dumps(feature_value, separators=(",", ":"))
        parts.append(f"Reloaded .spore/feature_list.json:\n{raw.strip()}")
    if not parts:
        return None
    return "\n\n".join(parts)


class RalphStopHook:
    """``Stop`` hook driving Ralph's multi-context-window continuation (issue
    #58, B1). At each ``FinalResponse`` it reads the Ralph checkpoint from the
    durable project-id :class:`RunStore` (#142, decision 3 — moved off
    ``.spore/progress.json``): incomplete tasks ⇒ :class:`HookBlock` (the loop
    continues), all complete ⇒ :class:`HookContinue` (the loop terminates).

    Registration is harmless for non-Ralph strategies: when the progress
    checkpoint key is UNSET the hook returns ``Continue`` and does not interfere
    with ReAct / PlanExecute / SelfVerifying runs. It only blocks when the
    progress checkpoint is PRESENT and reports incomplete tasks — the Ralph
    contract. Mirrors Rust's ``RalphStopHook``."""

    def __init__(self, run_store: RunStore, project_id: ProjectId) -> None:
        self._run_store = run_store
        self._project_id = project_id

    async def handle(self, ctx: Any) -> Any:
        from .hooks import HookBlock, HookContinue, StopContext

        # Only act on ``Stop``; any other event is a no-op ``Continue``.
        if not isinstance(ctx, StopContext):
            return HookContinue()
        # Absent checkpoint ⇒ do not interfere with non-Ralph runs. The checkpoint
        # now lives in the project-id RunStore (#142); when the progress key is
        # UNSET this is not a Ralph run, so ``Continue``.
        try:
            present = (
                await self._run_store.get(project_namespace(self._project_id), RALPH_PROGRESS_KEY)
                is not None
            )
        except StorageError:
            present = False
        if not present:
            return HookContinue()
        reason = await _ralph_completion_status(self._run_store, self._project_id)
        if reason is None:
            return HookContinue()
        return HookBlock(reason=reason)

    def events(self) -> list[Any]:
        from .hooks import HookEvent

        return [HookEvent.STOP]

    def name(self) -> str:
        return "ralph-stop"

    def sync_mode(self) -> Any:
        from .hooks import HookSync

        return HookSync.SYNC


@dataclass
class _PlanPhaseOutcome:
    """Internal result of a successful PlanExecute plan phase (issue #70).

    Carries the produced (and possibly hook-mutated) :class:`PlanArtifact` plus
    the run accounting so the ``PlanExecute`` arm can build its terminal
    :class:`RunResult`. Private to the harness."""

    artifact: Any  # PlanArtifact — typed Any to avoid a top-level hooks import
    usage: AggregateUsage
    turns: int


class StandardHarness:
    """Canonical :class:`Harness` implementation.

    Implements the ReAct loop fully and the PlanExecute plan phase (issue #70,
    phase 1 of 2); the remaining :class:`LoopStrategy` variants return
    :class:`HaltReasonStrategyNotYetImplemented` per the Rust reference.
    """

    def __init__(self, config: HarnessConfig) -> None:
        # Ralph completion mechanism (issue #58, B1): register a ``Stop`` hook
        # that drives multi-context-window continuation off
        # ``.spore/progress.json``. Registration is harmless for non-Ralph runs
        # — the hook only BLOCKS when a progress file is PRESENT and reports
        # incomplete tasks; when the file is absent (the common case for ReAct /
        # other strategies) it returns ``Continue``, so existing strategies are
        # unaffected. Mirrors Rust's ``StandardHarness::new``.
        from .hooks import StandardHookChain

        chain = config.hooks if config.hooks is not None else StandardHookChain()
        # Best-effort: a duplicate/invalid registration must never raise out of
        # the constructor. The hook subscribes only to the can-block ``Stop``
        # event, so registration cannot be rejected for sync/async mismatch.
        # #142: the checkpoint moved off ``.spore/progress.json`` onto the durable
        # project-id RunStore — the hook reads it from ``config.storage.run()`` at
        # ``config.project_id``.
        try:
            chain.register(RalphStopHook(config.storage.run(), config.project_id))
        except Exception:  # noqa: BLE001 — construction must not fail on a hook
            pass
        config.hooks = chain
        self._config = config
        # #126: harness-observed write/edit file paths for the CURRENTLY-running
        # execute step. The tool-dispatch seam pushes the ``path`` of every
        # ``write_file`` / ``edit_file`` call here as it dispatches; the DAG
        # executor drains it (:meth:`take_observed_writes`) on task completion to
        # build the observed ``files_touched`` for that task's
        # :class:`~spore_core.tasklist.StepLedgerEntry` — never model-self-reported.
        # Steps run SEQUENTIALLY (v1 ready-set is sequential), so a single shared
        # accumulator cleared per step is sufficient and deterministic.
        self._observed_writes: list[str] = []

    def _observe_write_call(self, call: ToolCall) -> None:
        """#126: record a harness-OBSERVED write/edit at the tool-dispatch seam.
        Only ``write_file`` / ``edit_file`` calls with a string ``path`` are
        recorded; de-duplicated against what is already accumulated for the
        current step. Called from the ReAct loop for the call ACTUALLY dispatched,
        so the path comes from the real tool call — never a model-self-reported
        field."""
        if call.name not in ("write_file", "edit_file"):
            return
        path = call.input.get("path")
        if isinstance(path, str) and path not in self._observed_writes:
            self._observed_writes.append(path)

    async def _build_context_sources(
        self,
        config: HarnessConfig,
        tool_schemas: list[ToolSchema],
        task_instruction: str,
    ) -> ContextSources:
        """Build the per-turn :class:`~spore_core.context.ContextSources` threaded
        into the structural ``ContextManager.assemble`` seam (issue #115 / SC-26).

        Configured guides (the guide source + active skills) reach the model
        structurally through the assemble seam, not as an ad-hoc User-message
        prepend. Skills (the manifest + active bodies, progressive disclosure) are
        appended as guides from the shared catalog, so loading a skill via
        ``load_skill`` makes its body sticky in the System block on the next turn.
        Memory (#160 / SC-26 follow-up) is queried from the configured provider
        each turn and injected structurally into the same System block; the
        composed static prompt is empty until the chunk-provider path is wired.
        An empty result renders to nothing, so a harness with no sources stays
        byte-identical to the pre-#115 pass-through."""
        from .context import ComposedPrompt, ContextSources

        guides: list[Guide] = list(config.guides)
        catalog = config.skills
        if catalog is not None:
            guides.extend(catalog.active_guides())
        # #160 / SC-26 follow-up: query the configured memory provider and inject
        # the relevant items structurally (rendered into the same System block as
        # guides). The query text defaults to the task instruction, so retrieved
        # memory tracks the current work; a configured ``query`` overrides it. A
        # query error is swallowed (empty memory) — memory is best-effort context,
        # never a halt. ``None`` leaves memory empty (byte-identical pre-#160 path).
        memory: list[MemoryItem] = []
        mem = config.memory
        if mem is not None:
            query = MemoryQuery(
                task_instruction=(mem.query if mem.query is not None else task_instruction),
                domain=mem.domain,
                session_id=None,
                min_relevance=mem.min_relevance,
                max_items=mem.max_items,
            )
            try:
                memory = await mem.provider.query(query)
            except MemoryError:
                memory = []
        return ContextSources(
            guides=guides,
            memory=memory,
            tool_schemas=tool_schemas,
            composed_prompt=ComposedPrompt(rendered="", block_1_hash=0),
        )

    def escalation_mode(self) -> EscalationMode:
        """The configured budget-escalation mode (#130, :class:`StrategyExecutor`
        accessor): returns ``self._config.escalation_mode``. Consulted at each
        ``ExhaustedResolution.ESCALATE`` site to decide between the autonomous
        propagate and the HITL pause."""
        return self._config.escalation_mode

    def enforce_output_schemas(self) -> bool:
        """The output-schema enforcement MIGRATION GATE (issue #139,
        :class:`StrategyExecutor` accessor): returns
        ``self._config.enforce_output_schemas``."""
        return self._config.enforce_output_schemas

    def output_schema_max_retries(self) -> int:
        """The number of EXTRA terminal-validation retry turns ``N`` (issue #139,
        :class:`StrategyExecutor` accessor): returns
        ``self._config.output_schema_max_retries``."""
        return self._config.output_schema_max_retries

    def storage(self) -> StorageProvider:
        """The injected :class:`StorageProvider` (issue #73). Defaults to an
        all-no-op provider when ``.storage(...)`` was never set."""
        return self._config.storage

    def project_id(self) -> ProjectId:
        """The STABLE durable-storage project namespace (issue #142). Durable
        artifacts (the ``task_list``, plan, Ralph checkpoint) are keyed by
        ``project_namespace(harness.project_id())``; defaults to a project id
        derived from ``sandbox.workspace_root()`` (decision 5)."""
        return self._config.project_id

    def session_store(self) -> SessionStore:
        """Convenience accessor for the storage layer's :class:`SessionStore`
        (issue #73, expose-only)."""
        return self._config.storage.session()

    def _effective_tool_registry(
        self, session_id: SessionId, toolset: ToolsetRef = ""
    ) -> ToolRegistry:
        """The harness-loop tool registry to use for a run keyed by
        ``session_id`` (issue #91).

        When catalogue tools were added via :meth:`HarnessBuilder.tool` /
        ``tools``, this bridges the folded :class:`StandardToolRegistry` through
        :class:`~spore_core.tool_registry.RealToolRegistry` — built fresh per run
        so the run's :class:`SessionId` + storage thread into every tool
        dispatch. Otherwise it returns the injected
        :attr:`HarnessConfig.tool_registry` seam unchanged.

        Issue 2 (per-node toolset scoping): ``toolset`` is the resolving leaf's
        ``ToolsetRef`` handle. When it is NON-EMPTY and a per-key catalogue was
        registered via :meth:`HarnessBuilder.toolset_tools`, THAT catalogue is
        bridged (so the node dispatches ONLY its own tools — strict scoping).
        Otherwise (empty handle, or non-empty handle with no per-key catalogue)
        the existing global-catalogue / ``tool_registry`` seam fallback applies, so
        examples 01–11 that use ``.tools()`` keep working byte-for-byte. Mirrors
        Rust's ``StandardHarness::effective_tool_registry``."""
        from .tool_registry import RealToolRegistry

        # Strict per-node scoping: a non-empty handle with its own catalogue.
        if toolset:
            scoped = self._config.toolset_catalogues.get(toolset)
            if scoped is not None:
                return RealToolRegistry(
                    scoped,
                    self._config.sandbox,
                    session_id,
                    self._config.project_id,
                    self._config.storage.run(),
                    self._config.storage.memory(),
                )
        # Fallback: empty handle (back-compat) or unregistered non-empty handle.
        catalogue = self._config.catalogue_registry
        if catalogue is None:
            return self._config.tool_registry
        return RealToolRegistry(
            catalogue,
            self._config.sandbox,
            session_id,
            self._config.project_id,
            self._config.storage.run(),
            self._config.storage.memory(),
        )

    # ---- helpers ----------------------------------------------------

    @staticmethod
    def _emit(stream: StreamSink | None, event: HarnessStreamEvent) -> None:
        if stream is not None:
            stream(event)

    @staticmethod
    def _tool_output_text(output: ToolOutput) -> str:
        """Extract the human-readable content from a tool output for the
        enriched coarse :class:`StreamToolResult` (issue #103, Q5)."""

        if isinstance(output, ToolOutputSuccess):
            return output.content
        if isinstance(output, ToolOutputError):
            return output.message
        return ""

    @staticmethod
    def _enrich_tool_error(message: str, schema: ToolSchema | None) -> ToolOutputError:
        """Render a recoverable tool-error ``message`` with the tool's parameter
        schema and a typing hint (issue #137, AC2 reuse). The breaker injects the
        result's text as the corrective USER message at the ``N`` threshold.
        Mirrors Rust's ``enrich_tool_error`` byte-for-byte: Rust's ``serde_json``
        (no ``preserve_order``) serializes objects via a ``BTreeMap``, sorting
        object keys alphabetically and recursively, so the schema JSON must be
        key-sorted here too (``sort_keys=True``) for the corrective message the
        model receives to be byte-identical across languages — the repo's
        canonical form is also key-sorted (``canonicalize_json``)."""
        enriched = message
        if schema is not None:
            schema_json = json.dumps(schema.input_schema, sort_keys=True, separators=(",", ":"))
            enriched += "\n\nExpected parameter schema: " + schema_json
        enriched += (
            "\n\nHint: provide arguments as correctly-typed JSON "
            '(e.g. true/false as a bool, 42 as a number, ["a"] as an array) '
            "rather than as quoted strings."
        )
        return ToolOutputError(message=enriched, recoverable=True)

    @classmethod
    async def _drive_turn(
        cls,
        agent: Agent,
        context: Context,
        on_stream: StreamSink | None,
    ) -> TurnResult:
        """Run one user-facing turn (issue #103). When a stream sink is
        attached, drive the turn through ``agent.turn_streaming`` with an
        adapter that maps each raw ``model`` ``StreamEvent`` to harness
        ``StreamEvent``s via :meth:`_map_model_stream_event`, threading a fresh
        :class:`TurnStreamState`. With no sink, falls back to plain ``turn``."""

        if on_stream is None:
            return await agent.turn(context)

        state = TurnStreamState()

        def adapter(event: ModelStreamEvent) -> None:
            for mapped in cls._map_model_stream_event(event, state):
                on_stream(mapped)

        sink: AgentStreamSink = adapter
        return await _agent_turn_streaming(agent, context, sink)

    @staticmethod
    def _map_model_stream_event(
        event: ModelStreamEvent, state: TurnStreamState
    ) -> list[HarnessStreamEvent]:
        """Map one raw :class:`spore_core.model.StreamEvent` to zero or more
        harness :class:`HarnessStreamEvent`s (issue #103), threading
        :class:`TurnStreamState` so blocks and tool calls are correlated.

        Rules:

        * Q2: a block's ``BlockStart`` is emitted exactly once, the first time
          a delta for that index is observed; ``ContentBlockStop`` maps to
          ``BlockStop``.
        * Q3: ``MessageStart`` / ``MessageStop`` map to nothing (dropped).
        * A tool-use block additionally emits ``ToolCallStart`` on open, then
          each fragment as ``ToolArgsDelta`` keyed by the derived ``call_id``.
        """

        if isinstance(event, (MessageStart, MessageStop)):
            return []  # Q3: dropped at the harness boundary.

        out: list[HarnessStreamEvent] = []
        if isinstance(event, ContentBlockDelta):
            if event.index not in state.open_blocks:
                state.open_blocks[event.index] = BlockKind.TEXT
                out.append(StreamBlockStart(index=event.index, block=BlockKind.TEXT))
            out.append(StreamTextDelta(content=event.delta))
            return out
        if isinstance(event, ThinkingDelta):
            if event.index not in state.open_blocks:
                state.open_blocks[event.index] = BlockKind.REASONING
                out.append(StreamBlockStart(index=event.index, block=BlockKind.REASONING))
            out.append(StreamReasoningDelta(content=event.delta))
            return out
        if isinstance(event, ToolUseStart):
            if event.index not in state.open_blocks:
                state.open_blocks[event.index] = BlockKind.TOOL_USE
                # Use the real call id from the model; consumers correlate
                # subsequent ToolArgsDelta by it.
                state.tool_calls[event.index] = event.id
                out.append(StreamBlockStart(index=event.index, block=BlockKind.TOOL_USE))
                out.append(
                    StreamToolCallStart(index=event.index, call_id=event.id, name=event.name)
                )
            return out
        if isinstance(event, ToolUseDelta):
            if event.index not in state.open_blocks:
                # Fallback: if a stream omitted ToolUseStart, open the block
                # here with a synthesized id and empty name so args still
                # surface.
                state.open_blocks[event.index] = BlockKind.TOOL_USE
                call_id = TurnStreamState.call_id_for(event.index)
                state.tool_calls[event.index] = call_id
                out.append(StreamBlockStart(index=event.index, block=BlockKind.TOOL_USE))
                out.append(StreamToolCallStart(index=event.index, call_id=call_id, name=""))
            call_id = state.tool_calls.get(event.index, TurnStreamState.call_id_for(event.index))
            out.append(StreamToolArgsDelta(call_id=call_id, partial_json=event.partial_json))
            return out
        if isinstance(event, ModelContentBlockStop):
            state.open_blocks.pop(event.index, None)
            return [StreamBlockStop(index=event.index)]
        return []

    @staticmethod
    def _budget_exceeded(
        budget: BudgetLimits, used: BudgetSnapshot, started_at: float
    ) -> BudgetLimitTypeT | None:
        if budget.max_turns is not None and used.turns >= budget.max_turns:
            return "turns"
        if budget.max_input_tokens is not None and used.input_tokens > budget.max_input_tokens:
            return "input_tokens"
        if budget.max_output_tokens is not None and used.output_tokens > budget.max_output_tokens:
            return "output_tokens"
        if (
            budget.max_wall_time is not None
            and (time.monotonic() - started_at) >= budget.max_wall_time
        ):
            return "wall_time"
        if budget.max_cost_usd is not None and used.cost_usd > budget.max_cost_usd:
            return "cost_usd"
        return None

    def _fail(
        self,
        reason: HaltReason,
        session_id: SessionId,
        usage: AggregateUsage,
        turns: int,
        session_state: SessionState | None = None,
    ) -> RunResultFailure:
        # ``session_state`` carries the post-run history on failure (issue #102).
        # ReAct / SelfVerifying / PlanExecute pass the live state; strategies
        # that re-seed a fresh window (Ralph / HillClimbing) leave it unset and
        # the failure carries an empty state, mirroring the Rust reference's
        # ``SessionState::default()``.
        return RunResultFailure(
            reason=reason,
            session_id=session_id,
            usage=usage,
            turns=turns,
            session_state=session_state if session_state is not None else SessionState(),
        )

    async def _fire_stop_hooks(
        self,
        session_id: SessionId,
        task: Task,
        turn_number: int,
        last_output_text: str,
        stop_blocks: int,
    ) -> str | None:
        """Fire registered ``Stop`` hooks (issue #69, R12-R14).

        Returns the block ``reason`` to inject when the loop should continue (a
        hook blocked and the per-run ``max_stop_blocks`` cap has not yet been
        hit). Returns ``None`` to allow normal termination — no hook chain, no
        block, the cap was reached, or a hook errored (a broken hook must not
        loop forever, so its error is treated as a non-blocking outcome).
        """
        config = self._config
        chain = config.hooks
        if chain is None:
            return None

        from .context import SessionState as ContextSessionState
        from .hooks import FireOutcome, StopContext, TurnOutput

        rich_state = ContextSessionState(
            session_id=session_id,
            task_id=task.id,
            task_instruction=task.instruction,
        )
        ctx = StopContext(
            session_id=session_id,
            turn_number=turn_number,
            last_output=TurnOutput(text=last_output_text, had_tool_calls=False),
            task_instruction=task.instruction,
            session_state=rich_state,
        )
        try:
            outcome = await chain.fire(ctx)
        except Exception:  # noqa: BLE001 — a broken Stop hook must not loop forever
            return None

        if isinstance(outcome, FireOutcome) and outcome.kind == "block":
            if stop_blocks >= config.max_stop_blocks:
                return None  # R14: cap reached — terminate anyway.
            return outcome.reason
        # Continue / Inject / Deny → allow normal termination.
        return None

    # ---- public API -------------------------------------------------

    async def run(self, options: HarnessRunOptions) -> RunResult:
        """Drive one run, then persist its terminal state (issue #102).

        Thin wrapper around :meth:`_run_inner`: it runs the loop and then calls
        :meth:`_auto_persist_terminal`, which is a no-op unless
        ``auto_persist_sessions`` is enabled. Mirrors Rust's ``run`` wrapper."""
        result = await self._run_inner(options)
        await self._auto_persist_terminal(result)
        return result

    async def _auto_persist_terminal(self, result: RunResult) -> None:
        """Issue #102 auto-persist seam: write the terminal run state to the
        :class:`SessionStore` when ``auto_persist_sessions`` is enabled.

        One write per ``run()`` / ``resume()``, at the same terminal seam as the
        observability flush. For Success / Failure a :class:`PausedState` is
        synthesized carrying the final ``session_state`` with empty pending
        fields (D4); for WaitingForHuman / Escalate the carried
        :class:`PausedState` is persisted (D6 — the cross-process pause case).
        Storage errors are swallowed-and-logged (D8): a put failure must never
        lose the run nor surface as a :class:`HaltReason`. When disabled (the
        default) this returns immediately WITHOUT touching the store — the
        off-by-default zero-I/O contract. Mirrors Rust's
        ``auto_persist_terminal``."""
        if not self._config.auto_persist_sessions:
            return
        if isinstance(result, RunResultSuccess | RunResultFailure):
            # Synthesize a completed-run PausedState: empty pending fields, no
            # human request, no child — it carries only the final history so a
            # later get_session(..).session_state resumes losslessly (D4).
            session_id = result.session_id
            state = PausedState(
                session_id=session_id,
                task_id=TaskId(str(session_id)),
                turn_number=result.turns,
                session_state=result.session_state,
                pending_tool_calls=[],
                approved_results=[],
                human_request=None,
                task=Task.new("", session_id, ReactConfig.per_loop(0)),
                budget_used=BudgetSnapshot(),
                child_state=None,
            )
        elif isinstance(result, RunResultWaitingForHuman | RunResultEscalate | RunResultConsult):
            # Persist the carried pause state directly (D6). Consult (issue #114)
            # is non-terminal but carries a resumable state, like WaitingForHuman,
            # so a cross-process host can later ``resume_consult`` it.
            state = result.state
            session_id = state.session_id
        else:  # pragma: no cover — RunResult is a closed union.
            return
        try:
            await self._config.storage.session().put_session(session_id, state)
        except Exception:  # noqa: BLE001 — swallow-and-log (D8): never lose the run
            pass

    async def _run_inner(self, options: HarnessRunOptions) -> RunResult:
        task = options.task
        on_stream = options.on_stream
        budget_used = BudgetSnapshot()

        strategy = task.loop_strategy

        # #124 startup validation: every serializable handle in the task's
        # strategy tree must resolve against the configured ExecutionRegistry,
        # BEFORE the first turn. The legacy single-collaborator fields are gone —
        # resolution is now the SINGLE path, so validation ALWAYS runs (the
        # ``is_empty()`` gate is dropped). An unresolved handle is a startup error.
        try:
            self._config.registry.validate(task)
        except HarnessErrorException as exc:
            return RunResultFailure(
                reason=HaltReasonConfigurationError(error=exc.error),
                session_id=task.session_id,
                usage=AggregateUsage(),
                turns=0,
                session_state=SessionState(),
            )

        # Issue #102 auto-load: when enabled AND no explicit session_state was
        # provided AND the strategy seeds incoming state (ReAct / SelfVerifying —
        # Ralph/HillClimbing discard it by design, D7), load the prior session
        # for this ``session_id`` from the SessionStore so a caller can resume by
        # id. Explicit ``session_state`` always wins (D5). A load failure starts
        # fresh (D8, swallow-and-log).
        if options.session_state is not None:
            session_state = options.session_state
        elif self._config.auto_persist_sessions and isinstance(
            strategy, ReactConfig | SelfVerifyingConfig
        ):
            session_state = await self._auto_load_session(task.session_id)
        else:
            session_state = SessionState()
        # #124: the central dispatch ``match`` is GONE — the only ``match`` left
        # is the enum→config delegation inside :func:`run_strategy`. The harness
        # entry collapses to ``run_strategy(task.loop_strategy, cx)`` via
        # :meth:`_drive_strategy`. Strategies that BUILD ON incoming state (ReAct
        # / PlanExecute / SelfVerifying) get the instruction seeded here on the
        # FRESH run (the compaction adapter ignores ``task``, so the harness owns
        # prompt delivery); Ralph / HillClimbing re-seed a fresh window internally
        # (D7), so their incoming state is discarded by the config body.
        if isinstance(strategy, ReactConfig | PlanExecuteConfig | SelfVerifyingConfig):
            await self._config.context_manager.append_user_message(session_state, task.instruction)
        return await self._drive_strategy(task, session_state, budget_used, on_stream)

    async def _auto_load_session(self, session_id: SessionId) -> SessionState:
        """Load the prior :class:`SessionState` for ``session_id`` from the
        :class:`SessionStore` (issue #102 auto-load). Returns the stored history
        when present, a fresh :class:`SessionState` when absent OR on any storage
        error (swallow-and-log, D8 — never surface a load failure as a halt)."""
        try:
            prior = await self._config.storage.session().get_session(session_id)
        except Exception:  # noqa: BLE001 — swallow-and-log (D8): start fresh
            return SessionState()
        return prior.session_state if prior is not None else SessionState()

    async def resume(
        self,
        state: PausedState,
        response: HumanResponse,
        on_stream: StreamSink | None = None,
    ) -> RunResult:
        """Resume a paused run, then persist its terminal state (issue #102).

        Thin wrapper around :meth:`_resume_inner` mirroring :meth:`run`: it
        resumes and then calls :meth:`_auto_persist_terminal` (a no-op unless
        ``auto_persist_sessions`` is enabled). Mirrors Rust's ``resume``
        wrapper."""
        result = await self._resume_inner(state, response, on_stream)
        await self._auto_persist_terminal(result)
        return result

    async def _resume_inner(
        self,
        state: PausedState,
        response: HumanResponse,
        on_stream: StreamSink | None = None,
    ) -> RunResult:
        session_state = state.session_state
        pending = state.pending_tool_calls
        task = state.task
        budget_used = state.budget_used
        session_id = state.session_id
        # Resolve the effective tool registry for this resumed session — bridges
        # catalogue tools the same way the turn loop does, so pending tool calls
        # dispatched during resume thread the run's storage + sandbox (issue #91).
        # #140: resume now routes through the pausing leaf's own toolset handle,
        # restoring its scoped per-node catalogue. An empty handle (the default)
        # still falls back to the global catalogue, so pre-#140 blobs and root
        # pauses behave unchanged. The budget-escalation branch below returns early
        # via ``_drive_strategy``, which re-resolves per-leaf toolsets during the
        # re-drive — so this registry is only used by the Clarification /
        # direct-resume paths whose pending calls need the carried handle.
        tool_registry = self._effective_tool_registry(session_id, state.toolset)

        # Clarification resume (issue #81, Q4b): if this pause came from
        # :class:`ToolOutputAwaitingClarification`, the human's answer is
        # injected as the tool RESULT for the clarifying call (the HEAD of
        # ``pending_tool_calls``) — NOT appended as a free-standing user message.
        # Any remaining pending calls from the same batch are then dispatched
        # normally before the loop resumes.
        if isinstance(state.human_request, HumanRequestClarification) and isinstance(
            response, HumanResponseAnswer | HumanResponseApproveWithFeedback
        ):
            answer = (
                response.text if isinstance(response, HumanResponseAnswer) else response.feedback
            )
            if pending:
                clarifying_call, *rest = pending
                tr = HarnessToolResult(
                    call_id=clarifying_call.id,
                    output=ToolOutputSuccess(content=answer, truncated=False),
                )
                await self._config.context_manager.append_tool_result(session_state, tr)
                for call in rest:
                    output = await tool_registry.dispatch(call)
                    tr = HarnessToolResult(call_id=call.id, output=output)
                    await self._config.context_manager.append_tool_result(session_state, tr)
            # SC-BUG-1: a clarification that surfaced from inside a composed tree
            # carries the FULL strategy in ``task.loop_strategy`` (each
            # combinator's ``_finish`` rewrote the pause's task on the way up).
            # Re-DRIVE that strategy from the answered worker session — exactly as
            # the consult resume does — so the SelfVerifying evaluate phase /
            # PlanExecute walk runs instead of only the bare worker leaf. A BARE
            # ReAct leaf has no surrounding frame, so it keeps the leaf-only resume
            # below (back-compat).
            if not isinstance(task.loop_strategy, ReactConfig):
                return await self._drive_strategy(
                    task,
                    # Top-level session starts fresh; the answered worker
                    # conversation threads in as the resume seed.
                    SessionState(),
                    budget_used,
                    on_stream,
                    None,
                    session_state,
                )
            max_iterations = (
                task.loop_strategy.max_iterations()
                if isinstance(task.loop_strategy, ReactConfig)
                else 2**31 - 1
            )
            agent = self._resolve_worker_agent(task.loop_strategy)
            if isinstance(agent, RunResultFailure):
                return agent.model_copy(update={"session_id": session_id})
            return await self._run_react(
                task, max_iterations, session_state, budget_used, on_stream, agent
            )

        # #130: a ``BudgetExhausted`` pause resumes via the operator's chosen
        # ``EscalationAction`` (carried on a typed :class:`HumanResponseEscalate`).
        # This branch runs BEFORE the generic ToolApproval / Clarification / Review
        # response handling below. ``available_actions`` is ADVISORY (fork D): an
        # out-of-set action is NOT hard-rejected here.
        if isinstance(state.human_request, HumanRequestBudgetExhausted):
            req = state.human_request
            steps_taken = req.steps_taken
            # #129 (AC2): ``continues_used`` is the SOLE :class:`BudgetContext`
            # field that rides a process pause — it is seeded back into the rebuilt
            # scope so a ``Continue`` spanning the pause resumes with the correct
            # continue count (it cannot exceed ``max_continues``). Q3:
            # ``continues_used`` rides the REQUEST payload, NOT a new serialized
            # :class:`BudgetContext` / :class:`PausedState` field.
            resume_continues = (req.phase, req.continues_used)
            # ``ContinueWithBudget(steps)``: grant ``steps_taken + steps`` total
            # allowance and re-enter the loop from the restored checkpoint. The
            # strategy tree is rebuilt with the node's budget caps raised so the
            # restored scope has room for ``steps`` more steps (fork E), and the
            # resumed scope's ``continues_used`` is seeded from the request (#129,
            # AC2).
            if isinstance(response, HumanResponseEscalate) and isinstance(
                response.action, EscalationActionContinueWithBudget
            ):
                granted = steps_taken + response.action.steps
                resumed_task = _grant_task_budget(task, granted)
                # #138 AC2-b: for a COMPOSED task (PlanExecute etc.) thread the
                # carried worker session through the phase-agnostic resume seed and
                # start the TOP-LEVEL session EMPTY — exactly mirroring
                # ``_resume_consult_inner``. The PlanExecute walk re-attaches the
                # seed to the single ``InProgress`` task (execute-phase exhaustion)
                # via the InProgress→Pending→complete machinery, or to the PLAN
                # session (AC3 plan-phase exhaustion) — so the stalled worker
                # continues mid-loop instead of re-planning. A BARE leaf has no
                # surrounding walk, so it resumes from ``session_state`` directly
                # (the seed has nowhere to attach).
                if isinstance(resumed_task.loop_strategy, ReactConfig):
                    top_session, resume_seed = session_state, None
                else:
                    top_session, resume_seed = SessionState(), session_state
                return await self._drive_strategy(
                    resumed_task,
                    top_session,
                    budget_used,
                    on_stream,
                    resume_continues,
                    resume_seed,
                )
            # ``Skip``: the node is marked skipped and the outer loop advances. For
            # a combinator (PlanExecute) the per-task partial already recorded the
            # blocked task and persisted the list; re-entering the loop from the
            # checkpoint advances to the remaining ready tasks. For a bare leaf
            # there is no sibling, so a skip resolves to a clean (empty) Success
            # carrying whatever partial history was captured.
            if isinstance(response, HumanResponseEscalate) and isinstance(
                response.action, EscalationActionSkip
            ):
                if isinstance(task.loop_strategy, PlanExecuteConfig):
                    return await self._drive_strategy(task, session_state, budget_used, on_stream)
                return RunResultSuccess(
                    output="",
                    session_id=session_id,
                    usage=AggregateUsage(),
                    turns=state.turn_number,
                    session_state=session_state,
                )
            # ``Fail`` (and any non-``Escalate`` response — out of contract, treated
            # conservatively as ``Fail``): abort the node and propagate
            # ``BudgetExceeded``; the partial is discarded (the ``Fail`` contract).
            return self._fail(
                HaltReasonBudgetExceeded(limit_type="turns"),
                session_id,
                AggregateUsage(),
                state.turn_number,
                SessionState(),
            )

        # Subagent depth: a full child.resume() dispatch lives with
        # SubagentTool (#5); for now we surface a placeholder and continue
        # the parent loop — mirrors the Rust reference behavior.
        if state.child_state is not None:
            pass

        if isinstance(response, HumanResponseHalt):
            return self._fail(
                HaltReasonHumanHalted(),
                session_id,
                AggregateUsage(),
                state.turn_number,
                session_state,
            )

        # An ``Escalate`` response delivered to a NON-budget pause is out of
        # contract (#130): the budget-escalation resume is handled by the
        # dedicated ``HumanRequestBudgetExhausted`` branch ABOVE, which returns
        # early. Reaching here means the operator sent an ``EscalationAction`` for
        # a ToolApproval / Review / ``None`` pause — halt cleanly rather than
        # mis-resuming the loop.
        if isinstance(response, HumanResponseEscalate):
            return self._fail(
                HaltReasonHumanHalted(),
                session_id,
                AggregateUsage(),
                state.turn_number,
                session_state,
            )

        if isinstance(response, HumanResponseDeny):
            for call in pending:
                tr = HarnessToolResult(
                    call_id=call.id,
                    output=ToolOutputError(message=response.reason, recoverable=True),
                )
                await self._config.context_manager.append_tool_result(session_state, tr)

        elif isinstance(response, HumanResponseReject):
            await self._config.context_manager.append_user_message(session_state, response.reason)

        elif isinstance(response, HumanResponseAnswer):
            await self._config.context_manager.append_user_message(session_state, response.text)

        elif isinstance(response, HumanResponseApproveWithFeedback):
            await self._config.context_manager.append_user_message(session_state, response.feedback)

        elif isinstance(response, HumanResponseAllow):
            for call in pending:
                output = await tool_registry.dispatch(call)
                tr = HarnessToolResult(call_id=call.id, output=output)
                await self._config.context_manager.append_tool_result(session_state, tr)

        elif isinstance(response, HumanResponseAllowWithModification):
            for call in response.calls:
                output = await tool_registry.dispatch(call)
                tr = HarnessToolResult(call_id=call.id, output=output)
                await self._config.context_manager.append_tool_result(session_state, tr)

        # SC-BUG-1: an Allow / Deny / Answer / Reject resume that surfaced from
        # inside a composed tree carries the FULL strategy in
        # ``task.loop_strategy`` (each combinator's ``_finish`` rewrote the pause's
        # task on the way up). The human response has already been applied to
        # ``session_state`` above (the approved calls dispatched, or the
        # deny/answer message appended). Re-DRIVE the whole strategy from that
        # mutated worker session — mirroring the consult resume — so the
        # SelfVerifying evaluate phase (the looper's eval-frame reviewer) /
        # PlanExecute walk re-runs instead of degrading to a plain executor. A BARE
        # ReAct leaf keeps the leaf-only resume below.
        if not isinstance(task.loop_strategy, ReactConfig):
            return await self._drive_strategy(
                task,
                SessionState(),
                budget_used,
                on_stream,
                None,
                session_state,
            )

        # Resume the ReAct loop from where we paused.
        max_iterations = (
            task.loop_strategy.max_iterations()
            if isinstance(task.loop_strategy, ReactConfig)
            else 2**31 - 1
        )
        agent = self._resolve_worker_agent(task.loop_strategy)
        if isinstance(agent, RunResultFailure):
            return agent.model_copy(update={"session_id": session_id})
        return await self._run_react(
            task, max_iterations, session_state, budget_used, on_stream, agent
        )

    async def resume_consult(
        self,
        state: PausedState,
        response: ConsultResponse,
        on_stream: StreamSink | None = None,
    ) -> RunResult:
        """Resume a worker paused by :class:`RunResultConsult` (issue #114), then
        persist its terminal state (issue #102). Thin wrapper around
        :meth:`_resume_consult_inner` mirroring :meth:`resume`."""
        result = await self._resume_consult_inner(state, response, on_stream)
        await self._auto_persist_terminal(result)
        return result

    async def _resume_consult_inner(
        self,
        state: PausedState,
        response: ConsultResponse,
        on_stream: StreamSink | None = None,
    ) -> RunResult:
        """Consult resume seam (issue #114). Mirrors the clarification resume
        branch: the :class:`ConsultResponse` text is injected as the tool RESULT
        for the head pending (consult) call — NOT appended as a free-standing
        user message (R10) — then any remaining pending calls are dispatched and
        the ReAct loop resumes."""
        session_state = state.session_state
        pending = state.pending_tool_calls
        task = state.task
        budget_used = state.budget_used
        session_id = state.session_id
        # #140: resume routes the preserved consulting call (and any remaining
        # pending calls) through the pausing leaf's own toolset handle, restoring
        # its scoped per-node catalogue. An empty handle (the default) still falls
        # back to the global catalogue, so pre-#140 blobs behave unchanged.
        tool_registry = self._effective_tool_registry(session_id, state.toolset)

        if isinstance(response, ConsultResponseAnswer):
            text, answered = response.text, True
        else:  # ConsultResponseBudgetExhausted
            text, answered = response.message, False

        # Observability: lightweight consult-resume event.
        if self._config.observability is not None:
            # Recover the consult ``kind`` from the head pending call's args, if
            # present, else fall back to a generic label.
            kind = ""
            if pending and isinstance(pending[0].input, dict):
                k = pending[0].input.get("kind")
                if isinstance(k, str):
                    kind = k
            base = SpanBase.new_root(
                new_span_id(f"{session_id}-consult-resume"),
                session_id,
                task.id,
                SpanKind.CONTEXT_ASSEMBLY,
                _now(),
            )
            self._config.observability.emit_context(
                ContextSpan(
                    base=base,
                    operation=ContextOperationConsultResumed(consult_kind=kind, answered=answered),
                    tokens_before=0,
                    tokens_after=0,
                    utilization_before=0.0,
                    utilization_after=0.0,
                )
            )

        # Inject the consult answer as the RESULT of the head pending (consult)
        # call, then dispatch the remaining pending calls in the same batch.
        if pending:
            consult_call, *rest = pending
            tr = HarnessToolResult(
                call_id=consult_call.id,
                output=ToolOutputSuccess(content=text, truncated=False),
            )
            await self._config.context_manager.append_tool_result(session_state, tr)
            for call in rest:
                output = await tool_registry.dispatch(call)
                tr = HarnessToolResult(call_id=call.id, output=output)
                await self._config.context_manager.append_tool_result(session_state, tr)

        # #131: a consult that surfaced from inside a composed tree carries the
        # FULL strategy in ``task.loop_strategy`` (each combinator's ``_finish``
        # rewrote the pause's task on the way up). Re-DRIVE that strategy rather
        # than resuming only the worker leaf: the PlanExecute walk resumes its
        # in-progress task from the injected worker session (``resume_seed``
        # seed), so the worker finishes mid-loop, its SelfVerifying evaluator
        # runs, the task is marked Completed, and the remaining ready-set is
        # walked. A BARE worker leaf (depth-1, e.g. a SubagentTool-mediated
        # consult) has no surrounding walk, so it keeps the original leaf-only
        # resume (back-compat).
        if not isinstance(task.loop_strategy, ReactConfig):
            return await self._drive_strategy(
                task,
                # Top-level session starts fresh; the worker conversation is
                # threaded into the in-progress task via the consult seed.
                SessionState(),
                budget_used,
                on_stream,
                None,
                session_state,
            )

        max_iterations = (
            task.loop_strategy.max_iterations()
            if isinstance(task.loop_strategy, ReactConfig)
            else 2**31 - 1
        )
        agent = self._resolve_worker_agent(task.loop_strategy)
        if isinstance(agent, RunResultFailure):
            return agent.model_copy(update={"session_id": session_id})
        return await self._run_react(
            task, max_iterations, session_state, budget_used, on_stream, agent
        )

    # ---- observability finalization ---------------------------------

    async def _finalize_observability(self, session_id: SessionId, outcome: SessionOutcome) -> None:
        """Record the terminal outcome and flush the observability session.
        Called at every terminal ``_run_react`` outcome (success or any halt)
        — never on a ``WaitingForHuman`` pause, which is not terminal. No-op
        when no provider is configured. Mirrors Rust's
        ``finalize_observability``."""
        obs = self._config.observability
        if obs is not None:
            obs.set_session_outcome(session_id, outcome)
            await obs.flush_session(session_id)

    # ---- StrategyExecutor protocol (#124) ---------------------------
    #
    # The harness-side primitives the recursive per-variant ``run`` bodies
    # delegate to. Each wraps an existing, tested orchestration method so behavior
    # stays at parity — the only structural change is that the per-variant bodies
    # now own their loops and the central dispatch ``match`` is gone.

    async def react_window(
        self,
        task: Task,
        max_iterations: int,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
        agent: Agent,
        toolset: ToolsetRef = "",
        output_schema: dict[str, Any] | None = None,
        output_schema_max_retries: int = 0,
        system_prompt: str | None = None,
    ) -> RunResult:
        """:class:`StrategyExecutor` primitive: one bounded ReAct turn-loop window
        on the resolved worker ``agent`` (delegates to :meth:`_run_react_inner`).
        Does NOT finalize observability — the leaf ``run`` body does. Issue 2:
        ``toolset`` is the leaf's RESOLVED handle, threaded down alongside
        ``agent``. Issue #139: ``output_schema`` / ``output_schema_max_retries``
        drive terminal-output validation + retry (``None`` ⇒ no enforcement).
        SC-10 (#161): ``system_prompt`` is the leaf's per-node prompt override,
        threaded down alongside ``toolset`` (``None`` ⇒ global prompt)."""
        return await self._run_react_inner(
            task,
            max_iterations,
            session_state,
            budget_used,
            on_stream,
            agent,
            toolset,
            output_schema,
            output_schema_max_retries,
            system_prompt,
        )

    def resolve_agent_ref(self, ref: str, session_id: SessionId) -> Agent | RunResultFailure:
        """:class:`StrategyExecutor` primitive: resolve an ``AgentRef`` to its
        registered agent (#124), or a typed ``UnresolvedHandle`` ``Failure``."""
        agent = self._config.registry.resolve_agent(ref)
        if agent is None:
            return RunResultFailure(
                reason=HaltReasonConfigurationError(
                    error=HarnessErrorUnresolvedHandle(handle_kind="agent", key=ref)
                ),
                session_id=session_id,
                usage=AggregateUsage(),
                turns=0,
                session_state=SessionState(),
            )
        return agent

    async def evaluate_phase(
        self,
        task: Task,
        eval_agent: Agent,
        eval_toolset: ToolsetRef,
        carried: BudgetSnapshot,
        total_usage: AggregateUsage,
    ) -> RunResult:
        """:class:`StrategyExecutor` primitive: the SelfVerifying evaluate phase
        (delegates to :meth:`_run_evaluate_phase`)."""
        return await self._run_evaluate_phase(task, eval_agent, eval_toolset, carried, total_usage)

    async def append_user_message(self, session_state: SessionState, text: str) -> None:
        """:class:`StrategyExecutor` primitive: append ``text`` as a user message
        onto ``session_state`` through the :class:`ContextManager` seam (#124)."""
        await self._config.context_manager.append_user_message(session_state, text)

    def workspace_root(self) -> Path:
        """:class:`StrategyExecutor` primitive: the configured sandbox workspace
        root (#124, for ``VerifierInput``)."""
        return self._config.sandbox.workspace_root()

    async def ralph_seed_session(self, instruction: str) -> SessionState:
        """:class:`StrategyExecutor` primitive: build the per-window Ralph seed
        session (delegates to :meth:`_ralph_seed_session`)."""
        return await self._ralph_seed_session(instruction)

    async def ralph_completion_status(self) -> str | None:
        """:class:`StrategyExecutor` primitive: Ralph external completion check
        (delegates to the module-level :func:`_ralph_completion_status`). #142:
        reads the checkpoint from the durable project-id RunStore (async)."""
        return await _ralph_completion_status(self._config.storage.run(), self._config.project_id)

    def ralph_max_resets(self) -> int:
        """:class:`StrategyExecutor` primitive: the Ralph outer-loop reset cap."""
        return self._config.max_resets

    def resolve_metric_evaluator(self, key: str, session_id: SessionId) -> Any | RunResultFailure:
        """:class:`StrategyExecutor` primitive: resolve the HillClimbing metric
        evaluator for ``key`` (#124, Q2), or a typed misconfiguration ``Failure``."""
        evaluator = self._config.registry.resolve_metric_evaluator(key)
        if evaluator is None:
            return RunResultFailure(
                reason=HaltReasonHillClimbingMisconfigured(
                    reason=(
                        f"HillClimbing requires a metric evaluator registered under key {key!r}"
                    )
                ),
                session_id=session_id,
                usage=AggregateUsage(),
                turns=0,
                session_state=SessionState(),
            )
        return evaluator

    def plan_directive(self, instruction: str) -> str:
        """:class:`StrategyExecutor` primitive: the planning directive seeded
        before the plan sub-strategy runs (R1)."""
        return (
            "Produce a step-by-step plan for the following task. Respond with a "
            'single JSON object: {"tasks": [<ordered step strings>], '
            '"rationale": <string>}.\n\nTask:\n' + instruction
        )

    async def seed_user_message(self, session_state: SessionState, text: str) -> None:
        """:class:`StrategyExecutor` primitive: append ``text`` as a user message
        onto ``session_state`` through the :class:`ContextManager` seam (#124)."""
        await self._config.context_manager.append_user_message(session_state, text)

    async def run_plan_subtree(
        self,
        plan: LoopStrategy,
        plan_task: Task,
        plan_session: SessionState,
        budget_used: BudgetSnapshot,
    ) -> RunResult | None:
        """:class:`StrategyExecutor` primitive: dispatch the plan sub-strategy
        ``plan`` for ``plan_task`` over ``plan_session`` (#124).

        #124 Q1: the planner concept is DROPPED — the plan child's leaf
        ``ReactConfig.agent`` is authoritative and resolved by the recursing leaf
        itself. The child's ``run`` drives the WHOLE plan loop (genuine recursion).
        Returns the child's verbatim terminal, or ``None`` when it produced no
        terminal."""
        cx = ExecutionContext(registry=self._config.registry)
        cx.executor = self
        cx.scratch.run_session = plan_session
        cx.scratch.run_budget = budget_used
        cx.scratch.task = plan_task
        await run_strategy(plan, cx)
        return cx.scratch.terminal_override

    async def capture_plan_artifact(
        self,
        session_id: SessionId,
        plan_output: str,
        usage: AggregateUsage,
        turns: int,
    ) -> _PlanPhaseOutcome | RunResult:
        """:class:`StrategyExecutor` primitive: capture + persist a
        :class:`PlanArtifact` from the plan sub-strategy's final output text
        (delegates to :meth:`_capture_and_persist_plan`)."""
        return await self._capture_and_persist_plan(session_id, plan_output, usage, turns)

    async def reconcile_completed_tasks(self, session_id: SessionId, task_list: object) -> None:
        """:class:`StrategyExecutor` primitive: A.6 deep-resume reconcile
        (delegates to :meth:`_reconcile_deep_resume`)."""
        await self._reconcile_deep_resume(session_id, task_list)

    async def fire_task_advance(
        self,
        session_id: SessionId,
        step_task: Task,
        task_index: int,
        total_tasks: int,
    ) -> Task:
        """:class:`StrategyExecutor` primitive: fire the ``OnTaskAdvance`` hook
        (pre, mutable) for an execute step, returning the (possibly mutated)
        step task."""
        from .hooks import OnTaskAdvanceContext

        if self._config.hooks is not None:
            ctx = OnTaskAdvanceContext(
                session_id=session_id,
                task=step_task,
                task_index=task_index,
                total_tasks=total_tasks,
            )
            try:
                await self._config.hooks.fire(ctx)
            except Exception:  # noqa: BLE001 — a broken hook must not abort the run
                pass
            # The chain threads mutations through ``ctx.task`` in place.
            step_task = ctx.task
        return step_task

    async def persist_task_list(self, session_id: SessionId, task_list: object) -> None:
        """:class:`StrategyExecutor` primitive: persist a parsed task list
        (delegates to :meth:`_persist_task_list`)."""
        await self._persist_task_list(session_id, task_list)

    async def load_task_list(self, session_id: SessionId) -> object | None:
        """:class:`StrategyExecutor` primitive: load the persisted
        :class:`~spore_core.tasklist.TaskList` from the RunStore ``task_list``
        store (#126, decision C). Returns the parsed list, or ``None`` when
        nothing was persisted (or the blob is unparseable). Mirrors Rust's
        ``load_task_list``.

        #142: the task_list is DURABLE — read it from the STABLE project
        namespace, NOT the per-window ``session_id`` (so the execute phase of a
        fresh Ralph window sees the prior window's list). The ephemeral
        ``session_id`` is retained for the seam but unused for this durable read."""
        from .tasklist import TASK_LIST_EXTRAS_KEY, TaskList

        _ = session_id  # #142: ephemeral key — durable read keys by project ns.
        durable_ns = project_namespace(self._config.project_id)
        try:
            value = await self._config.storage.run().get(durable_ns, TASK_LIST_EXTRAS_KEY)
        except Exception:  # noqa: BLE001 — a load failure starts fresh, never aborts
            return None
        if value is None:
            return None
        try:
            return TaskList.from_dict(value)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 — an unparseable checkpoint is ignored
            return None

    def take_observed_writes(self) -> list[str]:
        """:class:`StrategyExecutor` primitive: drain + return the harness-observed
        write/edit paths for the current step (#126 AC2), clearing the
        accumulator. Mirrors Rust's ``take_observed_writes``."""
        observed = self._observed_writes
        self._observed_writes = []
        return observed

    def clear_observed_writes(self) -> None:
        """:class:`StrategyExecutor` primitive: clear the harness-observed
        write/edit accumulator (#126 AC2). Mirrors Rust's
        ``clear_observed_writes``."""
        self._observed_writes = []

    async def finalize(self, result: RunResult) -> None:
        """:class:`StrategyExecutor` primitive: finalize observability for a
        terminal outcome (no-op for non-terminal pauses). Mirrors the tail of the
        per-strategy ``_finalize_*`` helpers — routes Success / Failure / Escalate
        to the matching :class:`SessionOutcome`."""
        if isinstance(result, RunResultSuccess):
            await self._finalize_observability(result.session_id, SessionOutcomeSuccess())
        elif isinstance(result, RunResultFailure):
            await self._finalize_observability(
                result.session_id,
                SessionOutcomeFailure(reason=result.reason.kind),
            )
        elif isinstance(result, RunResultEscalate):
            await self._finalize_observability(result.session_id, SessionOutcomeEscalated())
        # RunResultWaitingForHuman / RunResultConsult: not terminal, do not finalize.

    async def _drive_strategy(
        self,
        task: Task,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
        resume_continues: tuple[str, int] | None = None,
        resume_seed: SessionState | None = None,
    ) -> RunResult:
        """The recursive-executor entry (#124): build the shared
        :class:`ExecutionContext`, seed the per-run scratch (task / session /
        budget), drive ``run_strategy(task.loop_strategy, cx)``, and translate the
        outcome back into a terminal :class:`RunResult` (Q5). A non-terminal pause
        / escalate stashed in ``scratch.terminal_override`` propagates VERBATIM (it
        never collapses into a :class:`StrategyOutcome`).

        #129: ``_drive_strategy`` is the BARE-LEAF resolution site (a bare leaf
        never self-resolves inside its own body — rule 6 — it PROPAGATES a typed
        :class:`StrategyOutcomeBudgetExhausted` here, the single recovery site for
        a top-level leaf). When the leaf's CONFIGURED ``behavior`` resolves to
        ``Continue``, grant the leaf one more round IN-PROCESS (bump
        ``continues_used``, refresh the step cap) and re-drive WITHOUT any
        serialization (AC3) — looping until the behavior resolves to
        ``Fail``/``Escalate`` or the strategy completes. ``resume_continues =
        (phase, continues_used)`` seeds the FIRST matching budget scope's
        ``continues_used`` (via :meth:`BudgetContext.resumed`) so a ``Continue``
        that spanned a process pause resumes with the correct continue count (AC2);
        ``None`` is the fresh-run path.

        ``resume_seed = session`` (#131 consult + #138 budget) seeds the FIRST
        PlanExecute walk from ``session`` (the stalled worker conversation). For a
        consult resume the consult answer is already injected; for a budget resume
        (#138) it is the worker's full post-exhaustion session. The walk
        re-attaches it to the single ``InProgress`` task (execute-phase
        exhaustion) or, when the durable task_list has no ``InProgress`` task, to
        the PLAN session (#138 AC3 plan-phase exhaustion). ``None`` on every fresh
        / non-resume path."""
        session_id = task.session_id
        # The Continue resolution state threaded across in-process rounds: the
        # (possibly fallen-through) behavior + how many continues have been spent.
        behavior_for_resolution: tuple[BudgetExhaustedBehavior, int] | None = None
        # SC-5: auto-grants spent at the BARE-LEAF Escalate site under
        # :class:`EscalationModeAutoContinue`. The scope is rebuilt each round
        # here (unlike the in-loop combinator scopes, which carry their own
        # ``auto_grants_used``), so the counter lives across rounds in this
        # local.
        auto_grants_used: int = 0

        while True:
            cx = ExecutionContext(registry=self._config.registry)
            cx.executor = self
            cx.stream = on_stream
            cx.scratch.run_session = session_state
            cx.scratch.run_budget = budget_used
            cx.scratch.task = task.model_copy(deep=True)
            # #129 (AC2): the cross-process checkpoint seed is consumed by the
            # FIRST round only (the resumed node's scope); later in-process rounds
            # carry their continue count via ``behavior_for_resolution``.
            cx.scratch.resume_continues = resume_continues
            resume_continues = None
            # #131/#138: the resume seed is consumed by the FIRST round's
            # PlanExecute walk only; later in-process rounds run normally.
            cx.scratch.resume_seed = resume_seed
            resume_seed = None

            outcome = await run_strategy(task.loop_strategy, cx)

            # A pause / escalate (or any verbatim terminal) propagates unchanged.
            if cx.scratch.terminal_override is not None:
                return cx.scratch.terminal_override
            if isinstance(outcome, StrategyOutcomeComplete):
                return RunResultSuccess(
                    output=outcome.output,
                    session_id=session_id,
                    usage=cx.usage,
                    turns=cx.scratch.run_budget.turns,
                    session_state=cx.scratch.run_session,
                )
            if isinstance(outcome, StrategyOutcomeFailed):
                return RunResultFailure(
                    reason=HaltReasonConfigurationError(error=outcome.error),
                    session_id=session_id,
                    usage=cx.usage,
                    turns=cx.scratch.run_budget.turns,
                    session_state=cx.scratch.run_session,
                )
            # StrategyOutcomeBudgetExhausted (#125/#129): a BARE LEAF exhaustion
            # propagated here carrying its CONFIGURED ``behavior`` (Q1 — the
            # bare-leaf resolution site honors it; the leaf body never did).
            # Resolve it ONCE: a ``Continue`` with continues left re-drives
            # in-process; a spent ``Continue`` falls through to ``Fail``/
            # ``Escalate``. Carry the resolution chain's state across rounds so
            # ``max_continues`` is respected. (A combinator under ``SurfaceToHuman``
            # never reaches this arm — it sets ``terminal_override``, returned
            # above.)
            #
            # #125: the exhausted node's own ``steps_taken`` is the turn count it
            # reached (the scratch budget is not written back on the propagate
            # path). Fall back to the scratch turns if it is somehow 0.
            turns = outcome.steps_taken if outcome.steps_taken > 0 else cx.scratch.run_budget.turns
            # #137: the terminal :data:`HaltReason` for a ``Fail`` /
            # ``Escalate``→``Fail`` resolution depends on the CAUSE — a genuine
            # budget exhaustion reports ``BudgetExceeded``; an error-loop break
            # reports ``ToolErrorLoop``. (A granted ``Continue`` re-drives the
            # window, whose loop-local error counter starts fresh.)
            terminal_reason: HaltReason
            if isinstance(outcome.cause, ExhaustionCauseToolErrorLoop):
                terminal_reason = HaltReasonToolErrorLoop(
                    tool=outcome.cause.tool,
                    consecutive_errors=outcome.cause.consecutive_errors,
                )
            else:
                terminal_reason = HaltReasonBudgetExceeded(limit_type="turns")
            # Reconstruct the resolution scope: the FIRST round uses the leaf's
            # propagated behavior + continues_used; later in-process rounds reuse
            # the threaded (possibly fallen-through) state.
            if behavior_for_resolution is not None:
                resolve_behavior, resolve_continues = behavior_for_resolution
            else:
                resolve_behavior, resolve_continues = outcome.behavior, outcome.continues_used
            scope = BudgetContext.resumed(
                outcome.policy, resolve_behavior, outcome.phase, resolve_continues
            )
            resolution = scope.resolve_exhausted()

            if resolution == ExhaustedResolution.CONTINUE:
                # In-process continue (AC3: NO serialization). Refresh the leaf's
                # step cap and re-enter the loop carrying the post-run session so
                # the conversation context survives (Continue PRESERVES context —
                # AC4). The granted cap is ``steps_taken + policy.value`` so the
                # leaf gets a fresh window after the checkpoint.
                allowance = budget_allowance_value(outcome.policy)
                granted = outcome.steps_taken + (allowance if allowance is not None else 1)
                task = _grant_task_budget(task, granted)
                session_state = cx.scratch.run_session
                budget_used = cx.scratch.run_budget.model_copy(deep=True)
                on_stream = cx.stream
                # Thread the resolution chain's post-continue state so a subsequent
                # exhaustion sees the bumped ``continues_used``.
                behavior_for_resolution = (scope.behavior, scope.continues_used)
                continue

            from .execution_registry import (
                AutoGrantInfo,
                EscalationModeAutoContinue,
                EscalationModeSurfaceToHuman,
            )

            if resolution == ExhaustedResolution.ESCALATE:
                # SC-5: AutoContinue auto-grants ``steps_per_grant`` more steps
                # and re-drives IN-PROCESS (mirroring the ``Continue`` arm
                # above), up to ``max_grants`` times, firing ``on_grant`` per
                # grant. This is the bare-leaf / top-level
                # keep-working-but-capped site — where consumers otherwise
                # hand-roll a drive loop. Once the grants are spent it falls
                # through to the surface/abort handling below (the
                # ``Autonomous`` terminal).
                mode = self._config.escalation_mode
                if (
                    isinstance(mode, EscalationModeAutoContinue)
                    and auto_grants_used < mode.max_grants
                ):
                    auto_grants_used += 1
                    if mode.on_grant is not None:
                        mode.on_grant(
                            AutoGrantInfo(
                                grant_number=auto_grants_used,
                                steps_granted=mode.steps_per_grant,
                                phase=outcome.phase,
                            )
                        )
                    granted = outcome.steps_taken + mode.steps_per_grant
                    task = _grant_task_budget(task, granted)
                    session_state = cx.scratch.run_session
                    budget_used = cx.scratch.run_budget.model_copy(deep=True)
                    on_stream = cx.stream
                    continue
                # #130: a BARE LEAF ``Escalate`` under ``SurfaceToHuman`` PAUSES
                # with a ``BudgetExhausted`` request offering ``[ContinueWithBudget,
                # Fail]`` (fork C — no sibling to ``Skip`` to).
                if isinstance(self._config.escalation_mode, EscalationModeSurfaceToHuman):
                    err = BudgetExhausted(
                        policy=outcome.policy,
                        behavior=scope.behavior,
                        steps_taken=outcome.steps_taken,
                        continues_used=scope.continues_used,
                        phase=outcome.phase,
                    )
                    # #138 AC2-a: carry the FULL stalled leaf session (it
                    # propagated into scratch with its conversation) and the leaf's
                    # own toolset handle (#140/AC4-a).
                    worker_session = cx.scratch.run_session
                    cx.scratch.run_session = SessionState()
                    worker_toolset = _worker_toolset_of(task.loop_strategy)
                    return _promote_budget_exhausted_to_human(
                        err,
                        outcome.partial_output,
                        _leaf_escalation_actions(err),
                        session_id,
                        task,
                        cx.scratch.run_budget.model_copy(deep=True),
                        turns,
                        worker_session,
                        worker_toolset,
                    )
                # Under ``Autonomous``: surface the partial (Escalate carries it).
                messages = (
                    [Message(role=Role.ASSISTANT, content=TextContent(text=outcome.partial_output))]
                    if outcome.partial_output is not None
                    else []
                )
                return RunResultFailure(
                    # #137: ToolErrorLoop vs BudgetExceeded per cause.
                    reason=terminal_reason,
                    session_id=session_id,
                    usage=cx.usage,
                    turns=turns,
                    session_state=SessionState(messages=messages),
                )

            # ExhaustedResolution.FAIL: the partial is DISCARDED by contract.
            return RunResultFailure(
                # #137: ToolErrorLoop vs BudgetExceeded per cause.
                reason=terminal_reason,
                session_id=session_id,
                usage=cx.usage,
                turns=turns,
                session_state=SessionState(),
            )

    async def _run_evaluate_phase(
        self,
        task: Task,
        eval_agent: Agent,
        eval_toolset: ToolsetRef,
        carried: BudgetSnapshot,
        total_usage: AggregateUsage,
    ) -> RunResult:
        """Run the SelfVerifying evaluate phase (issue #61, #124): a fresh
        evaluator RUN over a read-only sandbox in a never-shared session, on the
        passed ``eval_agent`` (Q1c — the inner worker's resolved agent), scoped to
        ``eval_toolset`` (empty handle ⇒ global-catalogue fallback).

        Builds a child :class:`StandardHarness` from a copy of ``self._config``
        with the ``sandbox`` wrapped in a :class:`ReadOnlySandbox` (R3). The
        evaluator runs a fresh ReAct loop seeded with the ``role-evaluator`` chunk
        (R4, presence-only) plus a review directive, in a freshly generated session
        (R2/R9). Folds the evaluate run's usage into ``total_usage`` / ``carried``
        (R8) and returns its terminal :class:`RunResult`. Mirrors Rust's
        ``run_evaluate_phase``."""
        config = self._config

        # R3: derive a read-only sandbox internally from the build sandbox.
        read_only_sandbox = ReadOnlySandbox(config.sandbox)

        # R2/R9: fresh, never-shared session id for the evaluate run.
        eval_session_id = new_session_id()

        # R4 (presence-only): prepend the role-evaluator chunk content (if the
        # configured provider supplies it) to the review directive.
        role_chunk = await self._role_evaluator_chunk()
        review = (
            "Review the work produced for the following task and report whether "
            "it is correct. You did NOT write this code; default to FAIL unless "
            f"you can confirm it is right.\n\nTask:\n{task.instruction}"
        )
        directive = f"{role_chunk}\n\n{review}" if role_chunk is not None else review

        eval_task = Task(
            id=new_task_id(),
            instruction=directive,
            session_id=eval_session_id,
            budget=task.budget,
            loop_strategy=ReactConfig.per_loop(
                task.budget.max_turns if task.budget.max_turns is not None else 2**31 - 1
            ),
        )

        # Child harness: copy the config, swap the sandbox to read-only. The copy
        # shares the same observability / storage seams (incl. the registry) so the
        # evaluate run's spans land in the SAME trace stream (distinguished by its
        # distinct session id). The evaluate agent is passed to ``_run_react``
        # directly (#124 — no ``config.agent`` swap).
        eval_config = config.with_sandbox(read_only_sandbox)
        # The evaluate phase is a NON-INTERACTIVE nested review run: the caller
        # never sees its (freshly generated) session id, so an approval/HITL
        # middleware that resolves ``SurfaceToHuman`` would pause this run with no
        # human able to resume it — the reviewer's first ``read_file`` would yield
        # ``WaitingForHuman``, which the verifier reads as a misconfiguration, and
        # the review half would silently never run. The read-only sandbox already
        # enforces the only safety property a reviewer needs (no writes, no command
        # execution), so the caller's approval middleware is both redundant and
        # harmful here — drop it for the evaluate phase ONLY (cleared on this
        # eval-specific clone, not in ``with_sandbox`` globally). Observability /
        # pricing / storage seams stay (they are separate fields). Pair with an
        # ``eval_toolset`` scoped to read-only tools so non-filesystem
        # side-effecting tools the sandbox does not gate stay out of reach.
        eval_config.middleware = None
        # SC-30: when the eval phase falls back to the GLOBAL catalogue (empty
        # ``eval_toolset``), auto-derive a read-only VIEW of it — advertise +
        # dispatch only the intersection with ``StandardTools.readonly_set()`` (the
        # hardcoded ``READONLY_EVAL_TOOL_NAMES``), so the reviewer cannot reach
        # write / exec / side-effecting tools (incl. web/MCP the read-only sandbox
        # does not gate) WITHOUT the consumer registering a scoped read-only
        # toolset. A non-empty ``eval_toolset`` is an explicit opt-in and is left
        # untouched (it resolves via its own catalogue).
        if eval_toolset == "" and eval_config.catalogue_registry is not None:
            from .tool_registry import (
                READONLY_EVAL_TOOL_NAMES,
                ReadOnlyToolView,
                RealToolRegistry,
            )

            inner = RealToolRegistry(
                eval_config.catalogue_registry,
                eval_config.sandbox,
                eval_session_id,
                eval_config.project_id,
                eval_config.storage.run(),
                eval_config.storage.memory(),
            )
            # ``catalogue_registry`` is now cleared, so the empty-handle path of
            # ``_effective_tool_registry`` returns this filtered ``tool_registry``
            # for the reviewer's dispatch + schema advertising.
            eval_config.tool_registry = ReadOnlyToolView(inner, set(READONLY_EVAL_TOOL_NAMES))
            eval_config.catalogue_registry = None
        eval_harness = StandardHarness(eval_config)

        eval_state = SessionState()
        await eval_config.context_manager.append_user_message(eval_state, directive)

        cap = task.budget.max_turns if task.budget.max_turns is not None else 2**31 - 1
        eval_result = await eval_harness._run_react(
            eval_task, cap, eval_state, BudgetSnapshot(), None, eval_agent, eval_toolset
        )

        _fold_usage(total_usage, carried, eval_result)
        return eval_result

    async def _role_evaluator_chunk(self) -> str | None:
        """Look up the ``role-evaluator`` chunk content from the configured chunk
        provider (R4, presence-only). Returns ``None`` if the provider has no such
        chunk or fails to load. Mirrors Rust's ``role_evaluator_chunk``."""
        provider = self._config.chunk_provider
        if provider is None:
            return None
        try:
            chunks = await provider.load()
        except Exception:  # noqa: BLE001 — a broken provider must not abort the run
            return None
        for chunk in chunks:
            if chunk.id == "role-evaluator":
                return chunk.content
        return None

    async def _hill_climbing_revert(self) -> None:
        """Revert the working tree to current HEAD for a no-improvement iteration
        (issue #60, Decision 1). Runs ``git reset --hard HEAD`` THROUGH the
        sandbox; the harness NEVER spawns git directly. A sandbox rejection /
        non-zero exit is best-effort: the loop continues (the next agent turn
        re-derives state). Mirrors Rust's ``hill_climbing_revert``."""
        try:
            await self._config.sandbox.execute_command("git", ["reset", "--hard", "HEAD"])
        except Exception:  # noqa: BLE001 — revert is best-effort; never abort the loop
            pass

    @staticmethod
    def _hill_climbing_commit_hash() -> str | None:
        """Resolve the ``commit_hash`` recorded on a HillClimbing TSV row (issue
        #60, Decision 1). The harness never commits, so this is the EMPTY string
        (``None`` serialized as empty in the TSV) unless a ``VcsProvider`` is
        wired to supply a hash. v1 has no per-keep commit, so we always return
        ``None``. Mirrors Rust's ``hill_climbing_commit_hash``."""
        return None

    def _emit_hill_climbing_span(
        self,
        session_id: SessionId,
        task_id: TaskId,
        span_seq: int,
        iteration: int,
        metric_value: float | None,
        delta: float | None,
        status: str,
        reverted: bool,
    ) -> None:
        """Emit one fire-and-forget per-iteration observability span for a
        HillClimbing run (issue #60). No-op when no provider is configured.
        Mirrors Rust's ``emit_hill_climbing_span``."""
        obs = self._config.observability
        if obs is None:
            return
        emit_warn = getattr(obs, "emit_warn", None)
        if emit_warn is None:
            return
        base = SpanBase.new_root(
            new_span_id(f"{session_id}-hill-{span_seq}"),
            session_id,
            task_id,
            SpanKind.WARN,
            _now(),
        )
        emit_warn(
            WarnSpan(
                base=base,
                event=WarnEventHillClimbingIteration(
                    iteration=iteration,
                    metric_value=metric_value,
                    delta=delta,
                    status=status,
                    reverted=reverted,
                ),
            )
        )

    async def _write_hill_climbing_tsv(
        self,
        workspace_root: Path,
        task_id: TaskId,
        rows: list[Any],
    ) -> None:
        """Serialize the HillClimbing results log and write it to
        ``{workspace_root}/.spore/results/{task_id}.tsv`` (issue #60, Decisions
        2/3). Best-effort: a filesystem error is swallowed (the run outcome is
        authoritative, the TSV is a diagnostic artifact). Mirrors Rust's
        ``write_hill_climbing_tsv``."""
        body = self._render_hill_climbing_tsv(rows)
        try:
            dir_path = workspace_root / ".spore" / "results"
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / f"{task_id}.tsv").write_text(body, encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _render_hill_climbing_tsv(rows: list[Any]) -> str:
        """Render the HillClimbing results-log TSV body (issue #60, Decisions
        2/3). Pure function over the rows so the exact byte content is
        unit-testable and cross-language-comparable. Tab-separated, REQUIRED
        header, one row per iteration in ascending order. Floats use exactly 6
        decimal places for cross-language byte-identity. ``metric_value`` is the
        empty string on crashed/timeout rows; ``commit_hash`` is empty when no
        VCS. ``metadata`` is excluded. Trailing newline after every row.
        Mirrors Rust's ``render_hill_climbing_tsv``."""
        out = [
            "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription"
        ]
        for r in rows:
            # Decision 3: metric_value is EMPTY on crashed/timeout rows.
            if r.status in ("crashed", "timeout"):
                metric_value = ""
            else:
                metric_value = f"{r.metric_value:.6f}"
            commit_hash = r.commit_hash if r.commit_hash is not None else ""
            duration_secs = f"{r.duration:.6f}"
            out.append(
                f"{r.iteration}\t{commit_hash}\t{metric_value}\t{r.direction}\t"
                f"{r.status}\t{duration_secs}\t{r.description}"
            )
        return "\n".join(out) + "\n"

    # ---- #124 leaf primitives (Ralph seed + HillClimbing scoring) ----------

    async def _ralph_seed_session(self, instruction: str) -> SessionState:
        """Build the per-window Ralph seed session (#124): a fresh
        :class:`SessionState` seeded with ``instruction``, the reload context
        (R3), and the optional VCS history block — exactly the legacy
        ``_run_ralph`` window setup, minus the model loop (which now recurses).
        #142: the reload context now comes from the durable project-id RunStore
        checkpoint, not the ``.spore/`` filesystem. Mirrors Rust's
        ``ralph_seed_session``."""
        session_state = SessionState()
        await self._config.context_manager.append_user_message(session_state, instruction)
        reload = await _ralph_reload_context(self._config.storage.run(), self._config.project_id)
        if reload is not None:
            await self._config.context_manager.append_user_message(session_state, reload)
        # R3 (issue #58 v2): inject git history when a ``VcsProvider`` is wired.
        vcs = self._config.vcs_provider
        if vcs is not None:
            args = VcsLogArgs(max_entries=20)
            try:
                log = await vcs.log(args)
            except VcsError:
                log = ""
            trimmed = log.strip()
            if trimmed:
                block = f"Recent VCS history:\n{trimmed}"
                await self._config.context_manager.append_user_message(session_state, block)
        return session_state

    async def write_ralph_progress(self, complete: bool, remaining: list[str]) -> None:
        """Write the Ralph progress checkpoint to the durable project-id
        :class:`~spore_core.storage.RunStore` (#142, decision 3 — the WRITE path
        the relocated checkpoint needs; nothing wrote ``progress.json`` before).
        ``complete: true`` + empty ``remaining`` ⇒
        :func:`_ralph_completion_status` reports done. Mirrors Rust's
        ``write_ralph_progress``."""
        progress = RalphProgress(complete=complete, remaining=list(remaining))
        await self._config.storage.run().put(
            project_namespace(self._config.project_id),
            RALPH_PROGRESS_KEY,
            progress.model_dump(mode="json"),
        )

    async def write_ralph_feature_list(self, features: list[tuple[str, bool]]) -> None:
        """Write the Ralph feature-list checkpoint to the durable project-id
        :class:`~spore_core.storage.RunStore` (#142, decision 3). Each entry is
        ``{"name", "passes"}``; any ``passes: false`` keeps the run incomplete.
        Mirrors Rust's ``write_ralph_feature_list``."""
        entries = [{"name": name, "passes": passes} for (name, passes) in features]
        await self._config.storage.run().put(
            project_namespace(self._config.project_id),
            RALPH_FEATURE_LIST_KEY,
            entries,
        )

    def budget_exceeded(
        self, budget: BudgetLimits, used: BudgetSnapshot, started_at: float
    ) -> BudgetLimitTypeT | None:
        """:class:`StrategyExecutor` primitive: the wall-time / cost / token budget
        gate (#124, HillClimbing). Delegates to :meth:`_budget_exceeded`."""
        return self._budget_exceeded(budget, used, started_at)

    async def hill_baseline(
        self,
        evaluator: Any,
        session_id: SessionId,
        task_id: TaskId,
        direction: OptimizationDirection,
        rows: list[Any],
        span_seq: list[int],
        total_usage: AggregateUsage,
        turns: int,
    ) -> float | RunResultFailure:
        """HillClimbing iteration-0 baseline (#124): evaluate the metric (no agent
        turn), record the row + span, and return the baseline value — or a
        ``Failure`` on a baseline-evaluation failure (already records the failed
        row + writes the TSV). Mirrors Rust's ``hill_baseline``."""
        from .metric import MetricResult, ResultsEntry, iteration_status_from_error
        from .termination import SessionStateSnapshot

        workspace_root = self._config.sandbox.workspace_root()
        description = evaluator.description()
        snapshot = SessionStateSnapshot(
            session_id=session_id,
            task_id=task_id,
            state=SessionState(),
            workspace_root=workspace_root,
        )
        baseline = await evaluator.evaluate(self._config.sandbox, snapshot)
        if isinstance(baseline, MetricResult):
            rows.append(
                ResultsEntry(
                    iteration=0,
                    commit_hash=self._hill_climbing_commit_hash(),
                    metric_value=baseline.value,
                    direction=direction,
                    status="kept",
                    duration=baseline.duration,
                    description=description,
                )
            )
            self._emit_hill_climbing_span(
                session_id, task_id, span_seq[0], 0, baseline.value, None, "kept", False
            )
            span_seq[0] += 1
            return baseline.value
        # A baseline that cannot even be measured is a misconfiguration of the
        # experiment — record the failed row, write the TSV, and halt.
        status = iteration_status_from_error(baseline)
        rows.append(
            ResultsEntry(
                iteration=0,
                commit_hash=self._hill_climbing_commit_hash(),
                metric_value=float("nan"),
                direction=direction,
                status=status,
                duration=0.0,
                description=description,
            )
        )
        self._emit_hill_climbing_span(
            session_id, task_id, span_seq[0], 0, None, None, status, False
        )
        span_seq[0] += 1
        await self._write_hill_climbing_tsv(workspace_root, task_id, rows)
        return RunResultFailure(
            reason=HaltReasonHillClimbingMisconfigured(
                reason=f"baseline evaluation failed: {baseline.kind}"
            ),
            session_id=session_id,
            usage=total_usage,
            turns=turns,
            session_state=SessionState(),
        )

    async def hill_iteration(
        self,
        evaluator: Any,
        session_id: SessionId,
        task_id: TaskId,
        iteration: int,
        direction: OptimizationDirection,
        revert_on_no_improvement: bool,
        min_improvement_delta: float | None,
        current_best: float,
        rows: list[Any],
        span_seq: list[int],
    ) -> tuple[float, bool]:
        """HillClimbing per-iteration metric eval + keep/revert decision (#124):
        the agent turn already ran (recursively); this evaluates the metric,
        applies ``should_keep``, optionally reverts, records the row + span, and
        returns ``(current_best, non_improvement)``. Mirrors Rust's
        ``hill_iteration``."""
        from .metric import MetricResult, ResultsEntry, iteration_status_from_error, should_keep
        from .termination import SessionStateSnapshot

        workspace_root = self._config.sandbox.workspace_root()
        description = evaluator.description()
        snapshot = SessionStateSnapshot(
            session_id=session_id,
            task_id=task_id,
            state=SessionState(),
            workspace_root=workspace_root,
        )
        eval_result = await evaluator.evaluate(self._config.sandbox, snapshot)
        if isinstance(eval_result, MetricResult):
            value = eval_result.value
            kept = should_keep(value, current_best, direction, min_improvement_delta)
            delta = (current_best - value) if direction == "minimize" else (value - current_best)
            if kept:
                rows.append(
                    ResultsEntry(
                        iteration=iteration,
                        commit_hash=self._hill_climbing_commit_hash(),
                        metric_value=value,
                        direction=direction,
                        status="kept",
                        duration=eval_result.duration,
                        description=description,
                    )
                )
                self._emit_hill_climbing_span(
                    session_id, task_id, span_seq[0], iteration, value, delta, "kept", False
                )
                span_seq[0] += 1
                return value, False
            reverted = revert_on_no_improvement
            if reverted:
                await self._hill_climbing_revert()
            rows.append(
                ResultsEntry(
                    iteration=iteration,
                    commit_hash=self._hill_climbing_commit_hash(),
                    metric_value=value,
                    direction=direction,
                    status="discarded",
                    duration=eval_result.duration,
                    description=description,
                )
            )
            self._emit_hill_climbing_span(
                session_id, task_id, span_seq[0], iteration, value, delta, "discarded", reverted
            )
            span_seq[0] += 1
            return current_best, True
        # Crash/timeout/etc.: counts as a non-improvement.
        status = iteration_status_from_error(eval_result)
        reverted = revert_on_no_improvement
        if reverted:
            await self._hill_climbing_revert()
        rows.append(
            ResultsEntry(
                iteration=iteration,
                commit_hash=self._hill_climbing_commit_hash(),
                metric_value=float("nan"),
                direction=direction,
                status=status,
                duration=0.0,
                description=description,
            )
        )
        self._emit_hill_climbing_span(
            session_id, task_id, span_seq[0], iteration, None, None, status, reverted
        )
        span_seq[0] += 1
        return current_best, True

    async def hill_write_tsv(self, task_id: TaskId, rows: list[Any]) -> None:
        """:class:`StrategyExecutor` primitive: write the HillClimbing results TSV
        (#124). Delegates to :meth:`_write_hill_climbing_tsv`."""
        await self._write_hill_climbing_tsv(self._config.sandbox.workspace_root(), task_id, rows)

    async def _persist_task_list(
        self,
        session_id: SessionId,
        task_list: object,
    ) -> None:
        """Persist the parsed :class:`TaskList` for the run (Q4).

        The write goes through the :class:`RunStore` seam under
        ``TASK_LIST_EXTRAS_KEY``; the #71 sandbox-filesystem path
        (``.spore/task_list.json``) is intentionally NOT used — one source of
        truth. The RunStore write is the single source of truth (#76 removed the
        redundant ``SessionState.extras`` mirror). Storage failures are
        swallowed: a successful plan must not be lost to a storage hiccup (the
        default no-op provider never fails). Mirrors Rust's ``persist_task_list``.

        #142: the task_list is DURABLE — key it by the STABLE project namespace
        (``project_namespace(project_id)``), NOT the per-window ``session_id`` the
        Ralph wrapper regenerates each context window. The incoming ``session_id``
        remains the seam's ephemeral key and is unused for this durable write.
        """
        from .tasklist import TASK_LIST_EXTRAS_KEY, TaskList

        assert isinstance(task_list, TaskList)
        _ = session_id  # #142: ephemeral key — durable write keys by project ns.
        durable_ns = project_namespace(self._config.project_id)
        value = task_list.to_dict()
        try:
            await self._config.storage.run().put(durable_ns, TASK_LIST_EXTRAS_KEY, value)
        except Exception:  # noqa: BLE001 — a storage hiccup must not lose the plan
            pass

    async def _reconcile_deep_resume(
        self,
        session_id: SessionId,
        task_list: object,
    ) -> None:
        """A.6 deep-resume reconcile (#124, Q2): mark every task already
        ``Completed`` on the DURABLE RunStore checkpoint as ``Completed`` in the
        freshly-parsed ``task_list`` so it is NOT re-run. Tasks are matched by
        ``id`` (the task list is regenerated deterministically from the same
        artifact). Extracted from the legacy execute phase so the recursive
        PlanExecute body owns the per-task loop while the durable-state reconcile
        stays here as a leaf primitive. Mirrors Rust's ``reconcile_deep_resume``.

        #142: the durable checkpoint is keyed by the STABLE project namespace,
        NOT the per-window ``session_id`` — so deep-resume reads the SAME list a
        prior Ralph window persisted. The ephemeral ``session_id`` is unused for
        this durable read.
        """
        from .tasklist import TASK_LIST_EXTRAS_KEY, TaskList, TaskStatus

        assert isinstance(task_list, TaskList)
        _ = session_id  # #142: ephemeral key — durable read keys by project ns.
        durable_ns = project_namespace(self._config.project_id)
        try:
            saved_value = await self._config.storage.run().get(durable_ns, TASK_LIST_EXTRAS_KEY)
        except Exception:  # noqa: BLE001 — a load failure starts fresh, never aborts
            saved_value = None
        if saved_value is None:
            return
        try:
            saved = TaskList.from_dict(saved_value)
        except Exception:  # noqa: BLE001 — an unparseable checkpoint is ignored
            return
        completed_ids = {s.id for s in saved.tasks if s.status == TaskStatus.COMPLETED}
        for t in task_list.tasks:
            if t.id in completed_ids:
                t.status = TaskStatus.COMPLETED

    async def _capture_and_persist_plan(
        self,
        session_id: SessionId,
        plan_output: str,
        usage: AggregateUsage,
        turns: int,
    ) -> _PlanPhaseOutcome | RunResultFailure:
        """Capture + persist a :class:`PlanArtifact` from the plan sub-strategy's
        output text (#124): R3 (parse), R11 (fire ``OnPlanCreated``, mutable), R4
        (persist to the RunStore under ``PLAN_EXECUTE_EXTRAS_KEY``). The model
        turns that produced ``plan_output`` ran elsewhere — the recursive
        ``self.plan`` child — so this carries no agent call. Returns the captured
        outcome + accounting or a terminal failure to propagate. Mirrors Rust's
        ``capture_and_persist_plan``.

        SC-28 — a free-text / markdown plan is NOT a hard failure
        --------------------------------------------------------
        The strict canonical grammar (+ prose repair) runs first, so a JSON plan
        captures exactly as before — ``tasks``/``rationale`` come straight from
        the object. When BOTH fail the planner emitted prose rather than the JSON
        object; rather than aborting the whole run we capture it as a free-text
        artifact: the verbatim prose is preserved in ``rationale``, and the
        runnable ``tasks`` are sourced from the durable ``task_list`` tool store —
        the ONE authoring path (#126 decision C) that the execute phase already
        prefers over the artifact, so JSON was never the only source of executable
        steps. Mirroring it here keeps the ``OnPlanCreated`` payload's ``tasks``
        populated for panel consumers (looper ``plan_tracker``, cordyceps
        ``plan_announcer``). A prose plan that authored no ``task_list`` yields
        empty ``tasks`` — the execute phase then halts with ``EmptyPlan``, exactly
        as a JSON ``{"tasks": []}`` would. The pure
        :func:`~spore_core.plan.capture_plan_artifact` grammar stays strict and
        byte-identical across languages; only this harness driver is tolerant."""
        from .plan import (
            PLAN_EXECUTE_EXTRAS_KEY,
            PlanPhaseError,
            capture_plan_artifact_with_repair,
        )

        # R3: capture the artifact from the response text (the strict canonical
        # grammar runs first; repair only rescues a failure).
        try:
            artifact = capture_plan_artifact_with_repair(plan_output)
        except PlanPhaseError:
            # SC-28: not JSON → capture as free-text. ``tasks`` from the task_list
            # tool store (load_task_list reads the durable, project-scoped list the
            # plan leaf authored via the tool); prose preserved verbatim.
            from .tasklist import TaskList

            task_list = await self.load_task_list(session_id)
            tasks = (
                [t.description for t in task_list.tasks] if isinstance(task_list, TaskList) else []
            )
            artifact = PlanArtifact(tasks=tasks, rationale=plan_output)

        # R11: fire OnPlanCreated synchronously; the hook may rewrite the
        # artifact. The stored artifact reflects any mutation. Errors are
        # non-fatal: an observability/handler error must not lose a
        # successfully-captured plan, so the (possibly mutated) artifact is
        # still stored.
        if self._config.hooks is not None:
            from .hooks import OnPlanCreatedContext

            ctx = OnPlanCreatedContext(session_id=session_id, plan=artifact)
            try:
                await self._config.hooks.fire(ctx)
            except Exception:  # noqa: BLE001 — a broken hook must not lose the plan
                pass
            # The chain threads mutations through ``ctx.plan`` in place.
            artifact = ctx.plan

        # R4: persist the produced artifact to the RunStore seam under
        # PLAN_EXECUTE_EXTRAS_KEY (#76 — the durable single source of truth; no
        # longer mirrored into SessionState.extras). #142: the plan artifact is
        # DURABLE — key it by the STABLE project namespace, NOT the per-window
        # ``session_id`` (so a Ralph window reset re-reads the prior window's
        # plan). Storage failures are swallowed (matching the task-list persist):
        # a successfully-captured plan must not be lost to a storage hiccup (the
        # default no-op provider never fails).
        try:
            await self._config.storage.run().put(
                project_namespace(self._config.project_id),
                PLAN_EXECUTE_EXTRAS_KEY,
                artifact.model_dump(mode="json"),
            )
        except Exception:  # noqa: BLE001 — a storage hiccup must not lose the plan
            pass

        return _PlanPhaseOutcome(artifact=artifact, usage=usage, turns=turns)

    # ---- PlanExecute phase test helpers (#124) ----------------------
    #
    # The genuine plan / execute orchestration lives ONLY in
    # :func:`_run_plan_execute_config` (where it can dispatch the child
    # ``self.plan`` / ``self.execute`` strategies via ``run_strategy``). These two
    # helpers reproduce the minimal driver around a SINGLE phase using the same
    # leaf primitives + scratch wiring, so the granular plan/execute regression
    # tests exercise the real recursive dispatch rather than a stale parallel copy.

    async def _run_plan_phase(
        self,
        task: Task,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
    ) -> _PlanPhaseOutcome | RunResult:
        """Test helper (#124): drive the genuine recursive plan phase for ``task``
        (whose ``loop_strategy`` must be a :class:`PlanExecuteConfig`) — seed the
        directive, dispatch ``self.plan`` under the plan sub-strategy's OWN
        declared budget (the global ``max_turns`` is only the outer backstop) via
        :meth:`run_plan_subtree`, then capture + persist the artifact via
        :meth:`_capture_and_persist_plan`. Mirrors the plan half of
        :func:`_run_plan_execute_config` and Rust's cfg-test ``run_plan_phase`` —
        it must NOT clamp the planner to a single turn or it becomes a stale
        parallel copy of production. On success returns a :class:`_PlanPhaseOutcome`;
        on any failure returns the terminal :class:`RunResult` to propagate."""
        assert isinstance(task.loop_strategy, PlanExecuteConfig)
        plan_strategy = task.loop_strategy.plan
        session_id = task.session_id

        directive = self.plan_directive(task.instruction)
        plan_session = session_state.model_copy(deep=True)
        await self.seed_user_message(plan_session, directive)
        # The plan phase runs under the plan sub-strategy's OWN declared budget;
        # the global ``max_turns`` is only the outer backstop. (Previously clamped
        # to ``turns + 1`` — a single turn — which starved multi-turn ``task_list``
        # authoring and diverged this test seam from production.)
        plan_task = Task(
            id=task.id,
            instruction=directive,
            session_id=session_id,
            budget=task.budget.model_copy(deep=True),
            loop_strategy=plan_strategy,
        )
        plan_result = await self.run_plan_subtree(
            plan_strategy, plan_task, plan_session, budget_used.model_copy(deep=True)
        )
        if isinstance(plan_result, RunResultSuccess):
            return await self._capture_and_persist_plan(
                session_id, plan_result.output, plan_result.usage, plan_result.turns
            )
        if plan_result is None:
            return self._fail(
                HaltReasonPlanPhaseFailed(
                    error=PlanPhaseErrorPayload(
                        kind="planning_turn_failed",
                        message="plan sub-strategy produced no terminal",
                    ),
                ),
                session_id,
                AggregateUsage(),
                budget_used.turns,
            )
        return plan_result

    async def _run_execute_phase(
        self,
        task: Task,
        session_state: SessionState,
        task_list: object,
        carried: BudgetSnapshot,
        plan_usage: AggregateUsage,
        on_stream: StreamSink | None,
    ) -> RunResult:
        """Test helper (#124): drive the genuine recursive execute phase for
        ``task`` (whose ``loop_strategy`` must be a :class:`PlanExecuteConfig`),
        draining ``task_list`` by dispatching ``self.execute`` per task via
        ``run_strategy``. Mirrors the execute half of
        :func:`_run_plan_execute_config` and Rust's cfg-test ``run_execute_phase``.
        """
        from .tasklist import TaskList, TaskStatus

        assert isinstance(task_list, TaskList)
        assert isinstance(task.loop_strategy, PlanExecuteConfig)
        execute_strategy = task.loop_strategy.execute
        session_id = task.session_id

        await self._reconcile_deep_resume(session_id, task_list)

        total_tasks = len(task_list.tasks)
        total_usage = plan_usage.model_copy(deep=True)
        last_output = ""
        last_state = SessionState()

        for index in range(total_tasks):
            task_id = task_list.tasks[index].id
            instruction = task_list.tasks[index].description

            if task_list.tasks[index].status == TaskStatus.COMPLETED:
                last_output = instruction
                continue

            # #125: the per-task ``remaining_turns / remaining_tasks / step_cap``
            # derivation is REMOVED (dead) — enforcement is now ``charge``-based on
            # the PlanExecute scope. This helper mirrors the live body's structure.
            task_list.update(task_id, TaskStatus.IN_PROGRESS)
            await self._persist_task_list(session_id, task_list)

            step_task = Task(
                id=task.id,
                instruction=instruction,
                session_id=session_id,
                budget=task.budget.model_copy(deep=True),
                loop_strategy=execute_strategy,
            )
            step_task = await self.fire_task_advance(session_id, step_task, index, total_tasks)

            await self.seed_user_message(session_state, step_task.instruction)

            cx = ExecutionContext(registry=self._config.registry)
            cx.executor = self
            cx.scratch.run_session = session_state
            cx.scratch.run_budget = carried.model_copy(deep=True)
            cx.scratch.task = step_task
            await run_strategy(execute_strategy, cx)
            sub_result = cx.scratch.terminal_override

            if isinstance(sub_result, RunResultSuccess):
                carried.turns = sub_result.turns
                session_state = sub_result.session_state
                last_state = session_state
                carried.input_tokens += sub_result.usage.input_tokens
                carried.output_tokens += sub_result.usage.output_tokens
                total_usage.input_tokens += sub_result.usage.input_tokens
                total_usage.output_tokens += sub_result.usage.output_tokens
                total_usage.cache_read_tokens += sub_result.usage.cache_read_tokens
                total_usage.cache_write_tokens += sub_result.usage.cache_write_tokens
                total_usage.cost_usd += sub_result.usage.cost_usd
                last_output = sub_result.output

                task_list.complete(task_id)
                await self._persist_task_list(session_id, task_list)
                self._emit(on_stream, StreamFinalResponse(content=last_output))

            elif isinstance(sub_result, RunResultFailure):
                total_usage.input_tokens += sub_result.usage.input_tokens
                total_usage.output_tokens += sub_result.usage.output_tokens
                total_usage.cache_read_tokens += sub_result.usage.cache_read_tokens
                total_usage.cache_write_tokens += sub_result.usage.cache_write_tokens
                total_usage.cost_usd += sub_result.usage.cost_usd

                task_list.update(task_id, TaskStatus.BLOCKED)
                await self._persist_task_list(session_id, task_list)

                if isinstance(sub_result.reason, HaltReasonBudgetExceeded):
                    terminal_reason: HaltReason = sub_result.reason
                else:
                    terminal_reason = HaltReasonStepFailed(
                        task_index=index,
                        task=task_list.tasks[index].description,
                        reason=repr(sub_result.reason),
                    )
                return self._fail(
                    terminal_reason,
                    session_id,
                    total_usage,
                    sub_result.turns,
                    last_state,
                )

            elif sub_result is None:
                return self._fail(
                    HaltReasonStepFailed(
                        task_index=index,
                        task=task_list.tasks[index].description,
                        reason="execute sub-strategy produced no terminal",
                    ),
                    session_id,
                    total_usage,
                    carried.turns,
                    last_state,
                )
            else:
                # A pause / consult / escalate propagates the whole run unchanged.
                return sub_result

        return RunResultSuccess(
            output=last_output,
            session_id=session_id,
            usage=total_usage,
            turns=carried.turns,
            session_state=last_state,
        )

    # ---- ReAct loop -------------------------------------------------

    def _resolve_worker_agent(self, ls: LoopStrategy) -> Agent | RunResultFailure:
        """Resolve the worker agent for a :data:`LoopStrategy` tree from the
        :class:`ExecutionRegistry` (#124). Mirrors Rust's ``resolve_worker_agent``:
        the worker is the agent on the LEAF reached by descending the recursion.
        Returns the resolved agent, or a typed ``UnresolvedHandle`` ``Failure``."""
        key = _worker_agent_key_of(ls)
        agent = self._config.registry.resolve_agent(key)
        if agent is None:
            return RunResultFailure(
                reason=HaltReasonConfigurationError(
                    error=HarnessErrorUnresolvedHandle(handle_kind="agent", key=key)
                ),
                session_id=SessionId(""),
                usage=AggregateUsage(),
                turns=0,
                session_state=SessionState(),
            )
        return agent

    async def _run_react(
        self,
        task: Task,
        max_iterations: int,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
        agent: Agent,
        toolset: ToolsetRef = "",
    ) -> RunResult:
        """Drive the ReAct loop, then finalize observability for terminal
        outcomes. A ``WaitingForHuman`` pause is not terminal, so it is never
        flushed here — the eventual ``resume`` reaches a terminal outcome and
        flushes then. Mirrors Rust's ``run_react`` wrapper. The worker ``agent``
        is RESOLVED by the caller (#124). ``toolset`` is the RESOLVED toolset
        handle (mirrors ``agent``); the evaluate phase passes its ``eval_toolset``
        so the reviewer is scoped to read-only tools. The empty handle (the
        default) ⇒ global-catalogue fallback, as the other ``_run_react`` callers
        pass."""
        result = await self._run_react_inner(
            task, max_iterations, session_state, budget_used, on_stream, agent, toolset
        )
        if isinstance(result, RunResultSuccess):
            await self._finalize_observability(result.session_id, SessionOutcomeSuccess())
        elif isinstance(result, RunResultFailure):
            await self._finalize_observability(
                result.session_id,
                SessionOutcomeFailure(reason=result.reason.kind),
            )
        elif isinstance(result, RunResultEscalate):
            # An escalation is a clean, intentional terminal outcome (issue #80)
            # — finalize with the dedicated ``Escalated`` outcome, NOT
            # ``Partial``. Contrast with ``WaitingForHuman``, which is a pause,
            # not terminal, and is never finalized here.
            await self._finalize_observability(result.session_id, SessionOutcomeEscalated())
        # RunResultWaitingForHuman: not terminal, do not finalize.
        return result

    async def _run_react_inner(
        self,
        task: Task,
        max_iterations: int,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: StreamSink | None,
        # #124: the worker agent is RESOLVED by the caller (the recursing
        # ``_run_react_config`` resolves ``self.agent`` from the registry; Ralph
        # may override it per window). The leaf no longer reads ``config.agent``.
        agent: Agent,
        # Issue 2: the leaf's RESOLVED ``toolset`` handle (mirrors ``agent``).
        # Empty (``""``) ⇒ global-catalogue fallback; a non-empty handle with its
        # own per-key catalogue ⇒ strict per-node scoping. Legacy/resume paths
        # (no per-node toolset) thread the EMPTY handle → global fallback.
        toolset: ToolsetRef = "",
        # Issue #139: the leaf's RESOLVED output schema (``None`` ⇒ no
        # enforcement, identical to pre-#139). When set, the terminal is validated
        # against it (frozen validator subset) and a validation failure feeds the
        # error back + retries up to ``output_schema_max_retries`` extra turns
        # WITHIN budget; on exhaustion WITH budget remaining the window returns
        # :class:`HaltReasonOutputSchemaViolation`. The schema is also set on every
        # turn's ``ModelParams.output_schema`` so the Ollama ``format`` channel
        # constrains decoding (Anthropic/OpenAI ignore it).
        output_schema: dict[str, Any] | None = None,
        output_schema_max_retries: int = 0,
        # SC-10 (#161): the leaf's RESOLVED per-node system-prompt OVERRIDE.
        # ``None`` ⇒ the global ``config.system_prompt`` is used (byte-identical to
        # pre-SC-10). When set, it REPLACES the global prompt for every turn of
        # this window, so the leaf sees ONLY its own prompt (the per-leaf prompt
        # half of SC-10; the toolset half is the ``toolset`` arg above).
        system_prompt: str | None = None,
    ) -> RunResult:
        session_id = task.session_id
        # Resolve the effective tool registry once per turn-loop window (all
        # strategies funnel through here). Bridges the per-node toolset catalogue
        # (or the global catalogue when the handle is empty) per-run.
        tool_registry = self._effective_tool_registry(session_id, toolset)
        # Reset the adaptive prompt-based-tool-calling escalation flag at the
        # start of this turn-loop window so detection is scoped to the window and
        # does not leak across run() calls (the flag is shared with the model
        # wrapper for the harness's lifetime). No-op unless a conversational
        # harness installed the adaptive wrapper (#111).
        if self._config.prompt_tool_call_flag is not None:
            self._config.prompt_tool_call_flag.value = False
        started_at = time.monotonic()
        usage = AggregateUsage()
        # Monotonic per-run span counter for tool-call span ids, and the most
        # recent turn span base — the parent for that turn's tool-call spans.
        span_seq = 0
        current_turn_base: SpanBase | None = None
        # Per-run Stop-hook block counter (issue #69, R14). Resets on every
        # run() — resume starts fresh. After ``max_stop_blocks`` consecutive
        # blocks the loop terminates anyway.
        stop_blocks = 0
        # Consecutive-recoverable-tool-error breaker state (issue #137), keyed by
        # tool name. Loop-local to this window: a fresh run()/re-driven
        # ``Continue`` window starts with a clean N/2N allowance (AC3 Continue
        # reset). ``N`` is ``error_loop_threshold``; the hard stop is at ``2 * N``.
        error_loop_n = self._config.error_loop_threshold
        error_runs: dict[str, ErrorRun] = {}
        # Output-schema enforcement state (issue #139), loop-local to this window.
        # ``output_schema_retries_used`` counts the EXTRA retry turns spent on
        # validation feedback; the budget for them is ``output_schema_max_retries``
        # (the ``N``). ``last_schema_error`` holds the most recent frozen validator
        # error so a final exhaustion can report it. Both are inert when
        # ``output_schema`` is ``None`` (enforcement OFF / no schema).
        output_schema_retries_used = 0
        last_schema_error = ""
        if task.budget.max_turns is not None:
            effective_turn_cap = min(task.budget.max_turns, max_iterations)
        else:
            effective_turn_cap = max_iterations

        config = self._config

        while True:
            # Layer-1 budget gates before the turn.
            if budget_used.turns >= effective_turn_cap:
                return self._fail(
                    HaltReasonBudgetExceeded(limit_type="turns"),
                    session_id,
                    usage,
                    budget_used.turns,
                    session_state,
                )
            limit_type = self._budget_exceeded(task.budget, budget_used, started_at)
            if limit_type is not None:
                return self._fail(
                    HaltReasonBudgetExceeded(limit_type=limit_type),
                    session_id,
                    usage,
                    budget_used.turns,
                    session_state,
                )

            # Middleware: BeforeTurn (rich chain, issue #11). The chain may mutate
            # ``session_state`` in place (priority-ordered fan-out);
            # ``ContinueWithModification`` is the modified-but-proceed signal.
            if config.middleware is not None:
                decision = await config.middleware.fire_before_turn(
                    session_state, budget_used.turns
                )
                if isinstance(decision, MiddlewareContinue | MiddlewareContinueWithModification):
                    pass
                elif isinstance(decision, MiddlewareHalt):
                    return self._fail(
                        HaltReasonMiddlewareHalt(hook="before_turn", reason=decision.reason),
                        session_id,
                        usage,
                        budget_used.turns,
                        session_state,
                    )
                elif isinstance(decision, MiddlewareSurfaceToHuman):
                    paused = PausedState(
                        session_id=session_id,
                        task_id=task.id,
                        turn_number=budget_used.turns,
                        session_state=session_state,
                        pending_tool_calls=[],
                        approved_results=[],
                        human_request=decision.request,
                        task=task,
                        budget_used=budget_used,
                        child_state=None,
                        # #140: carry this leaf's toolset handle so resume routes
                        # through its scoped catalogue.
                        toolset=toolset,
                    )
                    return RunResultWaitingForHuman(state=paused, request=decision.request)
                else:
                    # ``ForceAnotherTurn`` is valid only at ``BeforeCompletion``; the
                    # StandardMiddlewareChain converts it to ``Halt`` here. Handle it
                    # defensively as a halt for any custom chain that emits it.
                    inject = getattr(decision, "inject", "")
                    return self._fail(
                        HaltReasonMiddlewareHalt(
                            hook="before_turn",
                            reason=f"ForceAnotherTurn is not valid at BeforeTurn: {inject}",
                        ),
                        session_id,
                        usage,
                        budget_used.turns,
                        session_state,
                    )

            # Assemble + invoke agent for one turn. ``sources`` (issue #115 /
            # SC-26) carries the rich ContextSources into the structural assemble
            # seam: tool schemas, configured guides, and active skills. An empty
            # ``sources`` renders to nothing, so the no-source path stays
            # byte-identical.
            sources = await self._build_context_sources(
                config, tool_registry.schemas(), task.instruction
            )
            context = await config.context_manager.assemble(session_state, task, sources)
            # Fill the model-facing tool list from the effective registry only
            # when the context manager rendered none (the compaction adapter
            # does), so a context manager that deliberately sets a phase-specific
            # tool subset is preserved.
            if not context.tools:
                context.tools = tool_registry.schemas()
            # Whether tools were advertised to the model this turn — a
            # precondition for classifying a prose final response as a missed
            # tool call (adaptive prompt-based escalation, #111).
            tools_advertised = bool(context.tools)
            # Prepend the configured operating system prompt (issue #91), MERGING
            # it with any structural #115 context block the manager already placed
            # as a leading System message (guides/memory/composed prompt). The
            # system prompt always leads. When the manager produced NO System
            # message (the common case — empty sources, or a test double), this
            # inserts a fresh System message exactly as before, so the
            # no-structural-block path stays byte-identical. The ``startswith``
            # guard keeps a resumed/seeded session that already leads with the
            # system prompt from being given it twice.
            #
            # SC-10 (#161): the leaf's per-node ``system_prompt`` override WINS
            # over the global ``config.system_prompt``. When this window's leaf
            # carries one it is used EXCLUSIVELY (the global prompt does not leak
            # in), so the plan and execute phases of a ``PlanExecuteConfig`` can
            # run under distinct prompts. With no leaf override (``None``) the
            # global prompt applies exactly as before — byte-identical. The check
            # is None-based (mirrors Rust's ``leaf.or(global)``), NOT falsy: an
            # explicit empty-string leaf override still REPLACES the global.
            effective_system_prompt = (
                system_prompt if system_prompt is not None else config.system_prompt
            )
            if effective_system_prompt is not None:
                first = context.messages[0] if context.messages else None
                if (
                    first is not None
                    and first.role == Role.SYSTEM
                    and isinstance(first.content, TextContent)
                ):
                    if not first.content.text.startswith(effective_system_prompt):
                        first.content = TextContent(
                            text=f"{effective_system_prompt}\n\n{first.content.text}"
                        )
                    # else: already leads with the system prompt — leave it.
                elif first is not None and first.role == Role.SYSTEM:
                    # A non-text System block — leave it.
                    pass
                else:
                    context.messages.insert(
                        0,
                        Message(
                            role=Role.SYSTEM,
                            content=TextContent(text=effective_system_prompt),
                        ),
                    )
            # Per-run model params win unconditionally (issue #93). The agent
            # copies ``Context.params`` verbatim into the ``ModelRequest``, so
            # this is the single seam that delivers configured params (e.g.
            # structured tool calls) to every tool-requesting ReAct/execute/
            # streaming turn.
            context.params = config.model_params
            # Issue #139: route the enforced output schema into this turn's
            # constrained-decoding channel (``ModelParams.output_schema``). Ollama
            # honors it via ``format``; Anthropic/OpenAI ignore it (no-op, like
            # ``structured_tool_calls``). ``None`` leaves the params byte-identical.
            # Copy before mutating so the shared ``config.model_params`` is never
            # touched.
            if output_schema is not None:
                context.params = config.model_params.model_copy(
                    update={"output_schema": output_schema}
                )
            self._emit(on_stream, StreamTurnStart(turn=budget_used.turns + 1))
            turn_started_at = _now()
            turn_clock = time.monotonic()
            # LLM-native content capture (issue #64): snapshot the assembled
            # INPUT messages (the full prompt the model saw) BEFORE the agent
            # turn call. Zero work when the guard is off.
            input_messages: list[GenAiMessage] | None = None
            if config.content_capture.enabled:
                input_messages = _capture_input_messages(
                    context.messages, config.content_capture.max_field_len
                )
            result: TurnResult = await self._drive_turn(agent, context, on_stream)
            budget_used.turns += 1

            # Emit a turn span for every model call (issue #12). Fire-and-forget;
            # it never affects control flow. The span base is retained as the
            # parent for any tool-call spans dispatched this turn.
            zero = TokenUsage()
            if isinstance(result, ToolCallRequested | FinalResponse):
                u = result.usage
            elif isinstance(result, TurnError):
                u = result.usage or zero
            else:
                u = zero
            if isinstance(result, FinalResponse):
                stop_reason, tool_calls_requested = StopReason.END_TURN, 0
            elif isinstance(result, ToolCallRequested):
                stop_reason, tool_calls_requested = StopReason.TOOL_USE, len(result.calls)
            else:
                stop_reason, tool_calls_requested = StopReason.END_TURN, 0
            turn_base = SpanBase.new_root(
                new_span_id(f"{session_id}-turn-{budget_used.turns}"),
                session_id,
                task.id,
                SpanKind.TURN,
                turn_started_at,
            )
            turn_status: SpanStatusOk | SpanStatusError
            if isinstance(result, TurnError):
                turn_status = SpanStatusError(message=result.error.kind)
            else:
                turn_status = SpanStatusOk()
            turn_base.finish(
                _now(),
                turn_status,
                int((time.monotonic() - turn_clock) * 1000),
            )
            if config.observability is not None:
                # LLM-native content capture (issue #64): output text + requested
                # tool calls, ONLY when the guard is enabled. Decision 4: the turn
                # span carries output + tool calls; no assembled input history.
                cc = config.content_capture
                output_text: GenAiMessage | None = None
                turn_tool_calls: list[ToolCallContent] | None = None
                if cc.enabled:
                    if isinstance(result, FinalResponse):
                        content, content_truncated = truncate_field(
                            result.content, cc.max_field_len
                        )
                        output_text = GenAiMessage(
                            role=GenAiRole.ASSISTANT,
                            content=content,
                            truncated=content_truncated,
                        )
                    elif isinstance(result, ToolCallRequested):
                        turn_tool_calls = [
                            _capture_tool_call_args(c, cc.max_field_len) for c in result.calls
                        ]
                config.observability.emit_turn(
                    TurnSpan(
                        base=turn_base,
                        turn_number=budget_used.turns,
                        input_tokens=u.input_tokens,
                        output_tokens=u.output_tokens,
                        cache_read_tokens=u.cache_read_tokens,
                        cache_write_tokens=u.cache_write_tokens,
                        cost_usd=config.pricing.cost_for(
                            u.input_tokens,
                            u.output_tokens,
                            u.cache_read_tokens,
                            u.cache_write_tokens,
                        ),
                        stop_reason=stop_reason,
                        tool_calls_requested=tool_calls_requested,
                        output_text=output_text,
                        tool_calls=turn_tool_calls,
                        input_messages=input_messages,
                    )
                )
            span_seq += 1
            current_turn_base = turn_base

            self._emit(on_stream, StreamTurnEnd(turn=budget_used.turns))

            # ---- FinalResponse --------------------------------------
            if isinstance(result, FinalResponse):
                usage.add_turn(result.usage)
                budget_used.input_tokens += result.usage.input_tokens
                budget_used.output_tokens += result.usage.output_tokens

                # Adaptive prompt-based tool-calling escalation (#111). When
                # tools were advertised but the model answered in prose with
                # action-intent language (it *meant* to act), set the session
                # flag so the wrapped model switches to prompt-based tool calling,
                # nudge the model, and force another turn instead of completing.
                # Guarded on the flag being unset so it fires at most once per
                # window (bounded — one extra turn) and only on the conversational
                # adaptive path.
                flag = config.prompt_tool_call_flag
                if (
                    flag is not None
                    and flag.value is False
                    and detect_prose_response(result.content, tools_advertised) is not None
                ):
                    flag.value = True
                    # Record the model's prose, then a corrective nudge, so the
                    # next turn has coherent context.
                    appender = getattr(config.context_manager, "append_assistant_message", None)
                    if appender is not None:
                        await appender(
                            session_state,
                            Message(
                                role=Role.ASSISTANT,
                                content=TextContent(text=result.content),
                            ),
                        )
                    nudge = (
                        "You described an action but did not call a tool. Use the "
                        "provided tool-call format to actually invoke the tool."
                    )
                    await config.context_manager.append_user_message(session_state, nudge)
                    continue

                if config.middleware is not None:
                    decision = await config.middleware.fire_before_completion(
                        result.content, budget_used.turns, session_state
                    )
                    if isinstance(
                        decision, MiddlewareContinue | MiddlewareContinueWithModification
                    ):
                        pass
                    elif isinstance(decision, MiddlewareForceAnotherTurn):
                        # The chain concatenates every middleware's injection into
                        # one ``ForceAnotherTurn`` (issue #11). Record the model's
                        # final text, then the injection as a user message, and force
                        # another turn instead of completing — the same channel the
                        # Stop-block breaker uses, so the conversation stays
                        # well-formed (assistant final text → user injection).
                        # Structural managers do not inherit the Protocol default, so
                        # probe ``append_assistant_message`` via ``getattr``.
                        appender = getattr(config.context_manager, "append_assistant_message", None)
                        if appender is not None:
                            await appender(
                                session_state,
                                Message(
                                    role=Role.ASSISTANT,
                                    content=TextContent(text=result.content),
                                ),
                            )
                        await config.context_manager.append_user_message(
                            session_state, decision.inject
                        )
                        continue
                    elif isinstance(decision, MiddlewareHalt):
                        return self._fail(
                            HaltReasonMiddlewareHalt(
                                hook="before_completion", reason=decision.reason
                            ),
                            session_id,
                            usage,
                            budget_used.turns,
                            session_state,
                        )
                    elif isinstance(decision, MiddlewareSurfaceToHuman):
                        paused = PausedState(
                            session_id=session_id,
                            task_id=task.id,
                            turn_number=budget_used.turns,
                            session_state=session_state,
                            pending_tool_calls=[],
                            approved_results=[],
                            human_request=decision.request,
                            task=task,
                            budget_used=budget_used,
                            child_state=None,
                            # #140: carry this leaf's toolset handle so resume
                            # routes through its scoped catalogue.
                            toolset=toolset,
                        )
                        return RunResultWaitingForHuman(state=paused, request=decision.request)

                tdecision = await config.termination_policy.evaluate(session_state, budget_used)
                if isinstance(tdecision, TerminationHalt):
                    return self._fail(
                        HaltReasonTerminationPolicyHalt(reason=tdecision.reason),
                        session_id,
                        usage,
                        budget_used.turns,
                        session_state,
                    )

                # Record the assistant's final text in history so a continued
                # session reflects what the agent said. Structural managers do
                # not inherit the Protocol default, so probe via ``getattr``.
                appender = getattr(config.context_manager, "append_assistant_message", None)
                if appender is not None:
                    await appender(
                        session_state,
                        Message(role=Role.ASSISTANT, content=TextContent(text=result.content)),
                    )

                # Stop hook (issue #69). The strategy believes the task is done;
                # fire registered Stop hooks synchronously. If any blocks (and
                # we are under ``max_stop_blocks``), inject the reason as a user
                # message and continue the loop instead of terminating.
                stop_reason = await self._fire_stop_hooks(
                    session_id,
                    task,
                    budget_used.turns,
                    result.content,
                    stop_blocks,
                )
                if stop_reason is not None:
                    stop_blocks += 1
                    await config.context_manager.append_user_message(session_state, stop_reason)
                    continue

                # Output-schema enforcement gate (issue #139). Additive to and
                # EARLIER than the Success terminal: when a schema is active,
                # validate this terminal ``FinalResponse``. On a validation
                # FAILURE, feed the frozen error back as a USER message and retry
                # one more turn (up to ``output_schema_max_retries`` extra turns).
                # The retried turn re-enters the loop top, where the budget gate
                # fires FIRST — so a retry that would exceed the turn/budget
                # backstop surfaces the existing ``BudgetExceeded`` terminal, NOT
                # ``OutputSchemaViolation`` (budget-cap-wins). Only when the N
                # retries are exhausted WITH budget remaining does the window
                # terminate with ``OutputSchemaViolation``. ``None`` schema ⇒ no
                # validation (migration gate OFF / no ``output``), so the terminal
                # is byte-identical to pre-#139.
                if output_schema is not None:
                    verr = validate_output(result.content, output_schema)
                    if verr is not None:
                        last_schema_error = verr
                        if output_schema_retries_used < output_schema_max_retries:
                            # Grant one more turn: feed the validator error back as
                            # a USER message (the assistant's invalid text is
                            # already in history above) and loop. The budget gate at
                            # the loop top enforces the budget-cap-wins precedence.
                            output_schema_retries_used += 1
                            self._emit(
                                on_stream,
                                StreamOutputSchemaRetry(
                                    attempt=output_schema_retries_used,
                                    error=verr,
                                ),
                            )
                            if config.observability is not None:
                                base = SpanBase.new_root(
                                    new_span_id(f"{session_id}-output-schema-retry-{span_seq}"),
                                    session_id,
                                    task.id,
                                    SpanKind.CONTEXT_ASSEMBLY,
                                    _now(),
                                )
                                config.observability.emit_context(
                                    ContextSpan(
                                        base=base,
                                        operation=ContextOperationOutputSchemaRetry(
                                            attempt=output_schema_retries_used,
                                            error=verr,
                                        ),
                                        tokens_before=0,
                                        tokens_after=0,
                                        utilization_before=0.0,
                                        utilization_after=0.0,
                                    )
                                )
                            feedback = output_schema_feedback_message(verr)
                            await config.context_manager.append_user_message(
                                session_state, feedback
                            )
                            continue
                        # Retries exhausted WITH budget remaining (the budget gate
                        # did not fire) → the typed schema-violation terminal
                        # (AC3). Total attempts = 1 + max_retries.
                        attempts = output_schema_max_retries + 1
                        self._emit(
                            on_stream,
                            StreamOutputSchemaViolation(
                                attempts=attempts,
                                error=last_schema_error,
                            ),
                        )
                        if config.observability is not None:
                            base = SpanBase.new_root(
                                new_span_id(f"{session_id}-output-schema-violation-{span_seq}"),
                                session_id,
                                task.id,
                                SpanKind.CONTEXT_ASSEMBLY,
                                _now(),
                            )
                            config.observability.emit_context(
                                ContextSpan(
                                    base=base,
                                    operation=ContextOperationOutputSchemaViolation(
                                        attempts=attempts,
                                        error=last_schema_error,
                                    ),
                                    tokens_before=0,
                                    tokens_after=0,
                                    utilization_before=0.0,
                                    utilization_after=0.0,
                                )
                            )
                        return self._fail(
                            HaltReasonOutputSchemaViolation(
                                schema=_canonicalize_json(output_schema),
                                attempts=attempts,
                                last_error=last_schema_error,
                            ),
                            session_id,
                            usage,
                            budget_used.turns,
                            session_state,
                        )

                self._emit(on_stream, StreamFinalResponse(content=result.content))
                return RunResultSuccess(
                    output=result.content,
                    session_id=session_id,
                    usage=usage,
                    turns=budget_used.turns,
                    session_state=session_state,
                )

            # ---- ToolCallRequested ----------------------------------
            if isinstance(result, ToolCallRequested):
                usage.add_turn(result.usage)
                budget_used.input_tokens += result.usage.input_tokens
                budget_used.output_tokens += result.usage.output_tokens

                # Always-halt short-circuit.
                for c in result.calls:
                    if tool_registry.is_always_halt(c.name):
                        return self._fail(
                            HaltReasonUnrecoverableToolError(
                                tool=c.name,
                                error="tool is annotated always_halt",
                            ),
                            session_id,
                            usage,
                            budget_used.turns,
                            session_state,
                        )

                # Record the assistant's turn (the tool calls the model
                # requested) as soon as the calls are known — BEFORE the
                # BeforeTool middleware (which may pause via SurfaceToHuman) and
                # before any tool result. This keeps the conversation well-formed
                # (assistant tool_use precedes its tool result) on every path,
                # including human-in-the-loop resume, so the resume path never
                # has to append it. The recorded turn reflects the model's
                # original request; a middleware or human modification changes
                # only what is dispatched. Structural managers do not inherit the
                # Protocol default, so probe via ``getattr``.
                appender = getattr(config.context_manager, "append_assistant_message", None)
                if appender is not None:
                    for call in result.calls:
                        await appender(
                            session_state,
                            Message(
                                role=Role.ASSISTANT,
                                content=MsgToolCallContent(
                                    id=call.id, name=call.name, input=call.input
                                ),
                            ),
                        )

                # Middleware: BeforeTool (rich chain, issue #11 / SC-11). The chain
                # mutates ``calls`` IN PLACE via a priority-ordered fan-out;
                # ``ContinueWithModification`` is the modified-but-proceed signal. The
                # assistant turn recorded just above keeps the model's ORIGINAL
                # request (``result.calls``) — only what is dispatched (this copied
                # list) changes.
                calls = list(result.calls)
                if config.middleware is not None:
                    decision = await config.middleware.fire_before_tool(calls, budget_used.turns)
                    if isinstance(
                        decision, MiddlewareContinue | MiddlewareContinueWithModification
                    ):
                        pass
                    elif isinstance(decision, MiddlewareHalt):
                        return self._fail(
                            HaltReasonMiddlewareHalt(hook="before_tool", reason=decision.reason),
                            session_id,
                            usage,
                            budget_used.turns,
                            session_state,
                        )
                    elif isinstance(decision, MiddlewareSurfaceToHuman):
                        paused = PausedState(
                            session_id=session_id,
                            task_id=task.id,
                            turn_number=budget_used.turns,
                            session_state=session_state,
                            pending_tool_calls=calls,
                            approved_results=[],
                            human_request=decision.request,
                            task=task,
                            budget_used=budget_used,
                            child_state=None,
                            # #140: carry this leaf's toolset handle so resume
                            # routes through its scoped catalogue.
                            toolset=toolset,
                        )
                        return RunResultWaitingForHuman(state=paused, request=decision.request)
                    else:
                        # ``ForceAnotherTurn`` is valid only at ``BeforeCompletion``;
                        # the StandardMiddlewareChain converts it to ``Halt`` here.
                        # Defensive for a custom chain that emits it.
                        inject = getattr(decision, "inject", "")
                        return self._fail(
                            HaltReasonMiddlewareHalt(
                                hook="before_tool",
                                reason=f"ForceAnotherTurn is not valid at BeforeTool: {inject}",
                            ),
                            session_id,
                            usage,
                            budget_used.turns,
                            session_state,
                        )

                approved_results: list[HarnessToolResult] = []
                # SC-9: the ``session_state.messages`` index of each appended tool
                # result, recorded 1:1 with ``approved_results``. The AfterTool
                # middleware hook uses these to re-render any result it rewrites in
                # place (via ``ContextManager.replace_tool_result``). Captured at
                # append time so it survives the #137 corrective-message interleaving.
                result_msg_indices: list[int] = []
                for i, call in enumerate(calls):
                    # Sandbox validation.
                    violation = await config.sandbox.validate(call)
                    if violation is not None:
                        if sandbox_violation_is_always_halt(violation):
                            return self._fail(
                                HaltReasonSandboxViolation(violation=violation),
                                session_id,
                                usage,
                                budget_used.turns,
                                session_state,
                            )
                        # Layer-2 default: recoverable — append as tool error.
                        tr = HarnessToolResult(
                            call_id=call.id,
                            output=ToolOutputError(
                                message=f"sandbox: {violation.kind}",
                                recoverable=True,
                            ),
                        )
                        self._emit(
                            on_stream,
                            StreamToolResult(
                                call_id=call.id,
                                is_error=True,
                                content=f"sandbox: {violation.kind}",
                            ),
                        )
                        await config.context_manager.append_tool_result(session_state, tr)
                        result_msg_indices.append(max(len(session_state.messages) - 1, 0))
                        approved_results.append(tr)
                        continue

                    self._emit(
                        on_stream,
                        StreamToolCall(call_id=call.id, name=call.name, args=call.input),
                    )
                    tool_started_at = _now()
                    tool_clock = time.monotonic()
                    output = await tool_registry.dispatch(call)

                    # #126 (AC2): record harness-OBSERVED write/edit paths from
                    # the call ACTUALLY dispatched. This is the single
                    # file-observation point the PlanExecute DAG executor reads via
                    # :meth:`take_observed_writes` to build a task's ledger
                    # ``files_touched`` — never a model-self-reported field.
                    self._observe_write_call(call)

                    # Subagent pause propagation.
                    if isinstance(output, ToolOutputWaitingForHuman):
                        remaining = calls[i + 1 :]
                        paused = PausedState(
                            session_id=session_id,
                            task_id=task.id,
                            turn_number=budget_used.turns,
                            session_state=session_state,
                            pending_tool_calls=remaining,
                            approved_results=approved_results,
                            human_request=output.request,
                            task=task,
                            budget_used=budget_used,
                            child_state=output.child_state,
                            # #140: the parent leaf's toolset handle (the child
                            # carries its own inside ``child_state``).
                            toolset=toolset,
                        )
                        return RunResultWaitingForHuman(state=paused, request=output.request)

                    # Escalation propagation (issue #80): a tool requests a
                    # structural state change. The harness is a pure
                    # intermediary — it does NOT act on the signal. It
                    # terminates cleanly, preserves session state for a possible
                    # resume, and returns :class:`RunResultEscalate`. The
                    # escalation is a control signal, NOT a conversation turn: it
                    # is never appended to message history (we ``return`` before
                    # the ``append_tool_result`` below), and the remaining tool
                    # calls in this batch are preserved into
                    # ``pending_tool_calls`` (mirroring WaitingForHuman). The
                    # signal is NOT stored in ``PausedState`` (``human_request``
                    # is ``None``), so it is discarded on resume — the harness
                    # never re-acts on it.
                    if isinstance(output, ToolOutputEscalate):
                        remaining = calls[i + 1 :]
                        turns = budget_used.turns
                        paused = PausedState(
                            session_id=session_id,
                            task_id=task.id,
                            turn_number=budget_used.turns,
                            session_state=session_state,
                            pending_tool_calls=remaining,
                            approved_results=approved_results,
                            human_request=None,
                            task=task,
                            budget_used=budget_used,
                            child_state=None,
                            # #140: carry this leaf's toolset handle so resume
                            # routes pending per-node calls through its catalogue.
                            toolset=toolset,
                        )
                        return RunResultEscalate(
                            signal=output.signal,
                            state=paused,
                            session_id=session_id,
                            usage=usage,
                            turns=turns,
                        )

                    # Clarification pause (issue #81, Q4b): a tool (e.g.
                    # ``ask_user_question``) needs a human answer before it can
                    # produce a result. UNLIKE the subagent ``WaitingForHuman``
                    # path, there is NO ``ChildPausedState``: the loop builds a
                    # :class:`PausedState` directly with ``human_request`` set to
                    # :class:`HumanRequestClarification`. The CLARIFYING call
                    # itself is preserved as the HEAD of ``pending_tool_calls``
                    # (followed by the remaining batch) so that, on resume, the
                    # human's answer is injected as the tool RESULT for that call.
                    if isinstance(output, ToolOutputAwaitingClarification):
                        pending = [call, *calls[i + 1 :]]
                        request = HumanRequestClarification(
                            question=output.question, options=output.options
                        )
                        paused = PausedState(
                            session_id=session_id,
                            task_id=task.id,
                            turn_number=budget_used.turns,
                            session_state=session_state,
                            pending_tool_calls=pending,
                            approved_results=approved_results,
                            human_request=request,
                            task=task,
                            budget_used=budget_used,
                            child_state=None,
                            # #140: carry this leaf's toolset handle so the
                            # preserved clarifying call resumes against its scoped
                            # catalogue.
                            toolset=toolset,
                        )
                        return RunResultWaitingForHuman(state=paused, request=request)

                    # Consult pause (issue #114, R1/R10): a worker-side tool
                    # returns :class:`ToolOutputConsult` (``child_state=None``) to
                    # ask for mid-loop help. Like the clarification arm there is
                    # NO ``ChildPausedState`` at this level — the loop builds a
                    # :class:`PausedState` directly with ``human_request=None`` and
                    # preserves the CONSULTING call as the HEAD of
                    # ``pending_tool_calls`` (followed by the remaining batch), so
                    # that on ``resume_consult`` the helper's answer is injected as
                    # the tool RESULT for that pending call. The consult is a
                    # control signal, NOT a conversation turn: it is never appended
                    # to message history here (R10) — we ``return`` before any
                    # ``append_tool_result``.
                    if isinstance(output, ToolOutputConsult):
                        # Observability: lightweight consult-spawn event alongside
                        # ``SkillInjected``.
                        if config.observability is not None:
                            base = SpanBase.new_root(
                                new_span_id(f"{session_id}-consult-spawn-{span_seq}"),
                                session_id,
                                task.id,
                                SpanKind.CONTEXT_ASSEMBLY,
                                _now(),
                            )
                            config.observability.emit_context(
                                ContextSpan(
                                    base=base,
                                    operation=ContextOperationConsultSpawned(
                                        consult_kind=output.request.kind
                                    ),
                                    tokens_before=0,
                                    tokens_after=0,
                                    utilization_before=0.0,
                                    utilization_after=0.0,
                                )
                            )
                        pending = [call, *calls[i + 1 :]]
                        turns = budget_used.turns
                        paused = PausedState(
                            session_id=session_id,
                            task_id=task.id,
                            turn_number=budget_used.turns,
                            session_state=session_state,
                            pending_tool_calls=pending,
                            approved_results=approved_results,
                            human_request=None,
                            task=task,
                            budget_used=budget_used,
                            child_state=None,
                            # #140 (THE load-bearing path): carry this leaf's
                            # toolset handle so ``resume_consult`` routes the
                            # preserved consulting call through its scoped
                            # catalogue instead of the global fallback.
                            toolset=toolset,
                        )
                        return RunResultConsult(
                            request=output.request,
                            state=paused,
                            session_id=session_id,
                            usage=usage,
                            turns=turns,
                        )

                    # SendMessage (issue #81): the ``send_message`` tool surfaces
                    # an out-of-band message to the user. The loop emits a
                    # :class:`StreamUserMessage` rather than collapsing the
                    # content into a normal tool result, then records a minimal
                    # success result so the loop continues.
                    if call.name == SEND_MESSAGE_TOOL_NAME and isinstance(
                        output, ToolOutputSuccess
                    ):
                        self._emit(on_stream, StreamUserMessage(content=output.content))

                    is_error = isinstance(output, ToolOutputError)
                    # Layer-2: unrecoverable tool error halts immediately.
                    if isinstance(output, ToolOutputError) and not output.recoverable:
                        return self._fail(
                            HaltReasonUnrecoverableToolError(tool=call.name, error=output.message),
                            session_id,
                            usage,
                            budget_used.turns,
                            session_state,
                        )

                    # ── Consecutive-recoverable-tool-error breaker (#137) ──────
                    # ``output`` here is either a Success or a RECOVERABLE Error
                    # (every other variant early-returned above). On a success the
                    # tool's error run resets (AC1); on a recoverable error we
                    # increment the identical-args run (AC1 args-variant) and check
                    # the N / 2N thresholds. At N we stash the corrective USER
                    # message here and append it AFTER the tool result below
                    # (well-formed conversation: assistant tool_use → tool result
                    # → corrective user message).
                    pending_corrective: str | None = None
                    if isinstance(output, ToolOutputError) and output.recoverable:
                        run = error_runs.get(call.name)
                        if run is not None and run.args == call.input:
                            # Same tool, structurally-identical args → extend.
                            run.count += 1
                        else:
                            # First error, or the args changed → fresh run.
                            run = ErrorRun(args=call.input, count=1)
                            error_runs[call.name] = run
                        count = run.count
                        two_n = error_loop_n * 2

                        # 2N: HARD STOP (AC3). Do NOT append this last tool result
                        # or continue — return a typed terminal that
                        # ``_run_react_config`` routes through the node's
                        # ``BudgetExhaustedBehavior`` WITHOUT burning the rest of
                        # the budget. The breaker is disabled when N == 0.
                        if error_loop_n > 0 and count >= two_n:
                            # AC4: emit the "broken" pair (stream + obs).
                            self._emit(
                                on_stream,
                                StreamToolErrorLoopBroken(tool=call.name, consecutive_errors=count),
                            )
                            if config.observability is not None:
                                base = SpanBase.new_root(
                                    new_span_id(f"{session_id}-tool-error-loop-broken-{span_seq}"),
                                    session_id,
                                    task.id,
                                    SpanKind.CONTEXT_ASSEMBLY,
                                    _now(),
                                )
                                config.observability.emit_context(
                                    ContextSpan(
                                        base=base,
                                        operation=ContextOperationToolErrorLoopBroken(
                                            tool_name=call.name,
                                            consecutive_errors=count,
                                        ),
                                        tokens_before=0,
                                        tokens_after=0,
                                        utilization_before=0.0,
                                        utilization_after=0.0,
                                    )
                                )
                            return self._fail(
                                HaltReasonToolErrorLoop(tool=call.name, consecutive_errors=count),
                                session_id,
                                usage,
                                budget_used.turns,
                                session_state,
                            )

                        # N: inject ONE corrective message (AC2). Render the
                        # schema+hint via ``_enrich_tool_error`` (reused) and inject
                        # it as a USER-role message — the SAME channel the Stop-block
                        # breaker uses — once per run (do not re-inject between N and
                        # 2N).
                        if error_loop_n > 0 and count >= error_loop_n and not run.injected:
                            run.injected = True
                            schema = next(
                                (s for s in tool_registry.schemas() if s.name == call.name),
                                None,
                            )
                            corrective = self._enrich_tool_error(output.message, schema).message
                            # AC4: emit the "detected/warning" pair (stream + obs)
                            # BEFORE the corrective message is appended.
                            self._emit(
                                on_stream,
                                StreamToolErrorLoopDetected(
                                    tool=call.name, consecutive_errors=count
                                ),
                            )
                            if config.observability is not None:
                                base = SpanBase.new_root(
                                    new_span_id(
                                        f"{session_id}-tool-error-loop-detected-{span_seq}"
                                    ),
                                    session_id,
                                    task.id,
                                    SpanKind.CONTEXT_ASSEMBLY,
                                    _now(),
                                )
                                config.observability.emit_context(
                                    ContextSpan(
                                        base=base,
                                        operation=ContextOperationToolErrorLoopDetected(
                                            tool_name=call.name,
                                            consecutive_errors=count,
                                        ),
                                        tokens_before=0,
                                        tokens_after=0,
                                        utilization_before=0.0,
                                        utilization_after=0.0,
                                    )
                                )
                            # Append AFTER the normal tool-result append below so
                            # the conversation stays well-formed. Stash it.
                            pending_corrective = corrective
                    else:
                        # AC1: ANY success for this tool resets its run.
                        error_runs.pop(call.name, None)

                    # Tool-call span (issue #12), child of the current turn.
                    if config.observability is not None:
                        if isinstance(output, ToolOutputSuccess):
                            output_size_bytes = len(output.content)
                            out_truncated = output.truncated
                        elif isinstance(output, ToolOutputError):
                            output_size_bytes = len(output.message)
                            out_truncated = False
                        else:
                            output_size_bytes = 0
                            out_truncated = False
                        # LLM-native content capture (issue #64): tool args + tool
                        # result, ONLY when the guard is enabled.
                        cc = config.content_capture
                        tool_args_content: ToolCallContent | None = None
                        tool_result_content: ToolResultContent | None = None
                        if cc.enabled:
                            tool_args_content = _capture_tool_call_args(call, cc.max_field_len)
                            if isinstance(output, ToolOutputSuccess):
                                rc, rt = truncate_field(output.content, cc.max_field_len)
                                tool_result_content = ToolResultContent(content=rc, truncated=rt)
                            elif isinstance(output, ToolOutputError):
                                rc, rt = truncate_field(output.message, cc.max_field_len)
                                tool_result_content = ToolResultContent(content=rc, truncated=rt)
                        span_id = new_span_id(f"{session_id}-tool-{span_seq}")
                        if current_turn_base is not None:
                            tool_base = SpanBase.new_child(
                                span_id, current_turn_base, SpanKind.TOOL_CALL, tool_started_at
                            )
                        else:
                            tool_base = SpanBase.new_root(
                                span_id,
                                session_id,
                                task.id,
                                SpanKind.TOOL_CALL,
                                tool_started_at,
                            )
                        tool_status: SpanStatusOk | SpanStatusError = (
                            SpanStatusError(message="tool returned a recoverable error")
                            if is_error
                            else SpanStatusOk()
                        )
                        tool_base.finish(
                            _now(),
                            tool_status,
                            int((time.monotonic() - tool_clock) * 1000),
                        )
                        config.observability.emit_tool_call(
                            ToolCallSpan(
                                base=tool_base,
                                tool_name=call.name,
                                call_id=call.id,
                                parameters_size_bytes=len(
                                    json.dumps(call.input, separators=(",", ":"))
                                ),
                                output_size_bytes=output_size_bytes,
                                truncated=out_truncated,
                                sandbox_mode="",
                                sandbox_violations=[],
                                arguments=tool_args_content,
                                result=tool_result_content,
                            )
                        )
                        span_seq += 1

                    tr = HarnessToolResult(call_id=call.id, output=output)
                    self._emit(
                        on_stream,
                        StreamToolResult(
                            call_id=call.id,
                            is_error=is_error,
                            content=self._tool_output_text(output),
                        ),
                    )
                    await config.context_manager.append_tool_result(session_state, tr)
                    # SC-9: record this result's message index BEFORE the #137
                    # corrective user message may be appended, so the index points
                    # at the tool-result message (re-sync, not defer-append).
                    result_msg_indices.append(max(len(session_state.messages) - 1, 0))
                    approved_results.append(tr)

                    # #137 (AC2): inject the ONE corrective user message at the N
                    # threshold, AFTER this call's tool result is recorded — same
                    # ``append_user_message`` channel the Stop-block breaker uses.
                    if pending_corrective is not None:
                        await config.context_manager.append_user_message(
                            session_state, pending_corrective
                        )

                # Middleware: AfterTool (rich chain, issue #11 / SC-9). The chain
                # receives the batch's results as a mutable list and may rewrite any
                # of them in place (priority-ordered, descending). On
                # ``ContinueWithModification``, re-render the affected tool-result
                # messages so the rewrite reaches the next model turn — this is what
                # lets an after-tool middleware turn a landed write into a
                # model-visible error (or vice versa) without the tool itself having
                # to invert its output (the cordyceps ``build_check`` inversion, done
                # by the harness).
                if config.middleware is not None:
                    decision = await config.middleware.fire_after_tool(calls, approved_results)
                    if isinstance(decision, MiddlewareContinue):
                        pass
                    elif isinstance(decision, MiddlewareContinueWithModification):
                        # Rewrite the SESSION MESSAGE HISTORY (not ``approved_results``,
                        # which resume ignores). Index captured at append time, so the
                        # re-sync survives #137 interleaving. Structural managers do not
                        # inherit the Protocol default, so probe via ``getattr``.
                        replacer = getattr(config.context_manager, "replace_tool_result", None)
                        if replacer is not None:
                            for res, idx in zip(approved_results, result_msg_indices, strict=False):
                                await replacer(session_state, idx, res)
                    elif isinstance(decision, MiddlewareHalt):
                        return self._fail(
                            HaltReasonMiddlewareHalt(hook="after_tool", reason=decision.reason),
                            session_id,
                            usage,
                            budget_used.turns,
                            session_state,
                        )
                    else:
                        # ``SurfaceToHuman`` / ``ForceAnotherTurn`` are not valid at
                        # ``AfterTool``; the StandardMiddlewareChain converts them to
                        # ``Halt``. Defensive for a custom chain that emits one.
                        return self._fail(
                            HaltReasonMiddlewareHalt(
                                hook="after_tool",
                                reason=f"illegal AfterTool decision: {decision.kind}",
                            ),
                            session_id,
                            usage,
                            budget_used.turns,
                            session_state,
                        )

                # Compaction (issue #46): after tool results are appended, before
                # the loop restarts — matches the concepts-doc loop diagram's
                # "compact if should_compact()" placement. Runs the
                # verify→retry→warn loop; never halts the run.
                if config.context_manager.should_compact(session_state):
                    span_seq = await self._run_compaction(
                        session_state,
                        session_id,
                        task.id,
                        span_seq,
                        usage,
                        agent,
                    )

                continue

            # ---- TurnError ------------------------------------------
            if isinstance(result, TurnError):
                if result.usage is not None:
                    usage.add_turn(result.usage)
                    budget_used.input_tokens += result.usage.input_tokens
                    budget_used.output_tokens += result.usage.output_tokens
                return self._fail(
                    HaltReasonAgentError(error=result.error),
                    session_id,
                    usage,
                    budget_used.turns,
                    session_state,
                )

            raise AssertionError(f"unhandled TurnResult variant: {result!r}")

    # ---- compaction loop (issue #46/#29) ----------------------------

    async def _run_compaction(
        self,
        session_state: SessionState,
        session_id: SessionId,
        task_id: TaskId,
        span_seq: int,
        usage: AggregateUsage,
        # #124: the compaction summary turn runs on the SAME resolved worker
        # agent driving the ReAct window (no ``config.agent``).
        agent: Agent,
    ) -> int:
        """Run the post-compaction verify→retry→warn loop (issue #46/#29).

        Drives one compaction turn through the agent, verifies the summary, and
        either accepts it, retries with the missing items injected, or — after
        ``max_compaction_attempts`` — emits a warn event and accepts the summary
        anyway. A blocked compaction is worse than an imperfect one, so this
        method NEVER raises or halts the run; the worst case is an
        accepted-anyway summary plus one warn span.

        Token usage from compaction turns folds into the run-level
        :class:`AggregateUsage`; each accepted summary is surfaced as a
        ``Compaction`` :class:`ContextSpan`. The
        ``compaction_verification_failures`` metric is derived from the emitted
        :class:`WarnSpan`. Returns the advanced ``span_seq``.
        """
        config = self._config
        turn = config.context_manager.prepare_compaction_turn(session_state)
        if turn is None:
            # Nothing to compact (e.g. history shorter than preserve window).
            return span_seq
        tokens_before = turn.verification_state.token_budget_used
        max_attempts = max(1, config.max_compaction_attempts)
        attempt = 0

        while True:
            attempt += 1
            # Run one compaction turn through the agent to produce a summary.
            result = await agent.turn(turn.context)
            if isinstance(result, FinalResponse):
                usage.add_turn(result.usage)
                summary = result.content
            elif isinstance(result, ToolCallRequested):
                # A compaction turn is expected to yield a summary, not a tool
                # call. Treat the (empty) response as the summary so
                # verification can run and the loop terminates predictably.
                usage.add_turn(result.usage)
                summary = ""
            else:  # TurnError
                if result.usage is not None:
                    usage.add_turn(result.usage)
                summary = ""

            verification = config.compaction_verifier.verify(
                summary, turn.preserve_hints, turn.verification_state
            )

            if verification.passed:
                return self._accept_compaction(
                    session_state,
                    summary,
                    turn.messages_removed,
                    tokens_before,
                    session_id,
                    task_id,
                    span_seq,
                )

            if attempt < max_attempts:
                # Inject the missing items and retry. Use the manager's override
                # if it provides one; otherwise the standard default body.
                inject = getattr(config.context_manager, "inject_missing_items", None)
                if inject is not None:
                    inject(turn.context, verification.missing_items)
                else:
                    _default_inject_missing_items(turn.context, verification.missing_items)
                continue

            # Exhausted attempts: warn, then accept anyway.
            if config.observability is not None:
                base = SpanBase.new_root(
                    new_span_id(f"{session_id}-warn-{span_seq}"),
                    session_id,
                    task_id,
                    SpanKind.WARN,
                    _now(),
                )
                warn_span = WarnSpan(
                    base=base,
                    event=WarnEventCompactionVerificationFailed(
                        missing_items=list(verification.missing_items),
                        accepted_anyway=True,
                    ),
                )
                # ``emit_warn`` carries a Protocol default no-op (issue #46), but
                # structural providers predating #46 may not define it at all —
                # fall back to the default no-op so they keep working (W4).
                emit_warn = getattr(config.observability, "emit_warn", None)
                if emit_warn is not None:
                    emit_warn(warn_span)
                span_seq += 1
            return self._accept_compaction(
                session_state,
                summary,
                turn.messages_removed,
                tokens_before,
                session_id,
                task_id,
                span_seq,
            )

    def _accept_compaction(
        self,
        session_state: SessionState,
        summary: str,
        messages_removed: int,
        tokens_before: int,
        session_id: SessionId,
        task_id: TaskId,
        span_seq: int,
    ) -> int:
        """Apply an accepted summary and emit the ``Compaction`` context span.
        Returns the advanced ``span_seq``."""
        config = self._config
        config.context_manager.apply_compaction(session_state, summary)

        # Real token accounting (#57 Known Deviation #2): read the
        # post-compaction budget back through the optional ``token_budget_used``
        # seam so the span reports what was actually reclaimed instead of zero.
        # Structural managers do not inherit the Protocol default, so probe via
        # ``getattr`` and fall back to the pre-compaction estimate when absent.
        tokens_after = tokens_before
        budget_seam = getattr(config.context_manager, "token_budget_used", None)
        if budget_seam is not None:
            after = budget_seam(session_state)
            if after is not None:
                tokens_after = after
        tokens_reclaimed = max(0, tokens_before - tokens_after)

        if config.observability is not None:
            base = SpanBase.new_root(
                new_span_id(f"{session_id}-compaction-{span_seq}"),
                session_id,
                task_id,
                SpanKind.COMPACTION,
                _now(),
            )
            util_before = 0.0
            util_after = 0.0
            config.observability.emit_context(
                ContextSpan(
                    base=base,
                    operation=ContextOperationCompaction(
                        messages_removed=messages_removed,
                        tokens_reclaimed=tokens_reclaimed,
                    ),
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    utilization_before=util_before,
                    utilization_after=util_after,
                )
            )
            span_seq += 1
        return span_seq


# ============================================================================
# Test-only stub implementations of the sibling component traits.
# Used by the unit tests and the fixture-replay test in this package.
# ============================================================================


class NoopContextManager:
    """Default context manager: passes session messages through unchanged
    and appends tool results / user messages as plain text. Sufficient for
    ReAct unit tests; the canonical impl lands in #7."""

    async def assemble(self, session: SessionState, task: Task, sources: ContextSources) -> Context:
        _ = (task, sources)
        return Context(
            messages=list(session.messages),
            tools=[],
            params=ModelParams(),
        )

    async def append_tool_result(self, session: SessionState, result: HarnessToolResult) -> None:
        # No-op: harness unit tests do not assert on message-shape; #7
        # owns the canonical wire shape.
        return None

    async def append_user_message(self, session: SessionState, text: str) -> None:
        return None

    def should_compact(self, session: SessionState) -> bool:
        return False


class NullSandbox(BaseSandboxProvider):
    """A sandbox that permits every tool-call validation and applies no path or
    process isolation. It is the right starting point for a tool-less or
    pure-compute agent — one where no tool is ever dispatched and the
    environment boundary is never exercised.

    This is the sandbox wired by :meth:`HarnessBuilder.conversational`. Agents
    that actually touch the filesystem or shell must use a real sandbox such as
    :class:`~spore_core.sandbox.WorkspaceScopedSandbox`. Mirrors Rust's
    ``NullSandbox``."""

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None


class AllowAllSandbox(BaseSandboxProvider):
    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None


class FixtureVcsProvider:
    """Deterministic :class:`VcsProvider` double for tests and fixture replay
    (issue #58 v2). Returns pre-loaded strings VERBATIM with no process
    spawning, so multi-context-window Ralph continuation can be exercised
    hermetically. :meth:`log` ignores its :class:`VcsLogArgs` and yields
    ``log_output``; :meth:`status` yields ``status_output``. Mirrors Rust's
    ``FixtureVcsProvider``."""

    def __init__(self, log_output: str, status_output: str) -> None:
        self._log_output = log_output
        self._status_output = status_output

    async def log(self, args: VcsLogArgs) -> str:
        return self._log_output

    async def status(self) -> str:
        return self._status_output


class ScriptedSandbox(BaseSandboxProvider):
    """Pop-front queue of validation outcomes (None for allow)."""

    def __init__(self) -> None:
        self._outcomes: list[SandboxViolation | None] = []

    def push(self, outcome: SandboxViolation | None) -> ScriptedSandbox:
        self._outcomes.append(outcome)
        return self

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        if not self._outcomes:
            return None
        return self._outcomes.pop(0)


class ScriptedToolRegistry:
    def __init__(self) -> None:
        self._outputs: list[ToolOutput] = []
        self._always_halt: set[str] = set()
        self.call_count: int = 0

    def push(self, output: ToolOutput) -> ScriptedToolRegistry:
        self._outputs.append(output)
        return self

    def mark_always_halt(self, name: str) -> None:
        self._always_halt.add(name)

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        self.call_count += 1
        if not self._outputs:
            return ToolOutputSuccess(content="ok")
        return self._outputs.pop(0)

    def is_always_halt(self, tool_name: str) -> bool:
        return tool_name in self._always_halt

    def schemas(self) -> list[ToolSchema]:
        return []


class AlwaysContinuePolicy:
    async def evaluate(
        self, session: SessionState, budget_used: BudgetSnapshot
    ) -> TerminationDecision:
        return TerminationContinue()


class ScriptedTerminationPolicy:
    def __init__(self) -> None:
        self._decisions: list[TerminationDecision] = []

    def push(self, d: TerminationDecision) -> ScriptedTerminationPolicy:
        self._decisions.append(d)
        return self

    async def evaluate(
        self, session: SessionState, budget_used: BudgetSnapshot
    ) -> TerminationDecision:
        if not self._decisions:
            return TerminationContinue()
        return self._decisions.pop(0)


class ScriptedMiddleware:
    """Scripted :class:`MiddlewareChain` test double (rich surface, issue #11).

    Decisions are queued per hook via :meth:`push`; each ``fire_*`` method pops
    the front entry if it targets that hook, else returns ``Continue`` without
    consuming the entry. Unlike :class:`StandardMiddlewareChain`, scripted
    decisions are returned RAW (no ``_validate_decision``), so a test can exercise
    the harness's defensive handling of an out-of-place decision. The
    ``.push(hook, decision)`` API is stable.
    """

    def __init__(self) -> None:
        self._decisions: list[tuple[HookPoint, MiddlewareDecision]] = []

    def push(self, hook: HookPoint, decision: MiddlewareDecision) -> ScriptedMiddleware:
        self._decisions.append((hook, decision))
        return self

    def _next_for(self, hook: HookPoint) -> MiddlewareDecision:
        if not self._decisions:
            return MiddlewareContinue()
        front_hook, _ = self._decisions[0]
        if front_hook != hook:
            return MiddlewareContinue()
        return self._decisions.pop(0)[1]

    async def register(self, middleware: Middleware) -> None:
        # The double scripts decisions directly; registration is a no-op.
        _ = middleware

    async def fire_before_session(self, task: Task, session_id: SessionId) -> MiddlewareDecision:
        _ = (task, session_id)
        return self._next_for("before_session")

    async def fire_before_turn(self, session: SessionState, turn_number: int) -> MiddlewareDecision:
        _ = (session, turn_number)
        return self._next_for("before_turn")

    async def fire_before_tool(self, calls: list[ToolCall], turn_number: int) -> MiddlewareDecision:
        _ = (calls, turn_number)
        return self._next_for("before_tool")

    async def fire_after_tool(
        self, calls: list[ToolCall], results: list[HarnessToolResult]
    ) -> MiddlewareDecision:
        _ = (calls, results)
        return self._next_for("after_tool")

    async def fire_before_completion(
        self, response: str, turn_number: int, state: SessionState
    ) -> MiddlewareDecision:
        _ = (response, turn_number, state)
        return self._next_for("before_completion")

    async def fire_after_session(self, result: RunResult, session_id: SessionId) -> None:
        # After hooks ignore the decision; drain a scripted AfterSession entry.
        _ = (result, session_id)
        self._next_for("after_session")


# ============================================================================
# Deferred forward-ref resolution (issue #80)
# ============================================================================
# ``PlanArtifact`` (from :mod:`spore_core.hooks`) and ``Mode`` (from
# :mod:`spore_core.prompt_chunk_registry`) live in modules that import
# ``HarnessConfig`` / ``PausedState`` from THIS module, so they can only be
# imported after every class above is defined — hence at the very bottom. The
# escalation types reach them directly, and ``ToolOutputWaitingForHuman`` /
# ``ChildPausedState`` / ``PausedState`` reach them transitively through the
# ``ToolOutput`` union, so all of these models are rebuilt here.
from .hooks import PlanArtifact  # noqa: E402
from .prompt_chunk_registry import Mode  # noqa: E402

# ``ContextErrorModel`` lives in :mod:`spore_core.context`, which imports this
# module at its top — a top-level import here would be circular. It is imported
# after this module is otherwise defined so ``HaltReasonContextError``'s
# forward-ref ``error`` field can be resolved by the rebuild below.
from .context import ContextErrorModel  # noqa: E402, F401

HaltReasonContextError.model_rebuild()
HarnessSignalExitPlanMode.model_rebuild()
HarnessSignalSwitchMode.model_rebuild()
ToolOutputEscalate.model_rebuild()
ToolOutputWaitingForHuman.model_rebuild()
ToolOutputConsult.model_rebuild()
RunResultEscalate.model_rebuild()
RunResultConsult.model_rebuild()
ChildPausedState.model_rebuild()
PausedState.model_rebuild()


# ============================================================================
# Internal helper exports
# ============================================================================


# Type aliases re-exported as the canonical names so downstream code can
# write ``isinstance(decision, MiddlewareSurfaceToHuman)`` etc.
__all__ = [
    "AggregateUsage",
    "AllowAllSandbox",
    "AlwaysContinuePolicy",
    "BaseSandboxProvider",
    "CommandOutput",
    "CompleteOnFinalResponse",
    "EmptyToolRegistry",
    "FileRef",
    "TruncatedOutput",
    "BudgetLimits",
    "BudgetLimitTypeT",
    "BudgetSnapshot",
    "ChildPausedState",
    "ConsultHandlerEntry",
    "ConsultOverflowPolicy",
    "ConsultOverflowPolicyEscalateToHuman",
    "ConsultOverflowPolicySoftFail",
    "ConsultRequest",
    "ConsultResponse",
    "ConsultResponseAnswer",
    "ConsultResponseBudgetExhausted",
    "ContextManager",
    "HaltReason",
    "HaltReasonAgentError",
    "HaltReasonBudgetExceeded",
    "HaltReasonContextError",
    "HaltReasonEmptyPlan",
    "HaltReasonHumanHalted",
    "HaltReasonMiddlewareHalt",
    "HaltReasonSandboxViolation",
    "HaltReasonStagnationLimitReached",
    "HaltReasonStepFailed",
    "HaltReasonSelfVerifyExhausted",
    "HaltReasonSelfVerifyMisconfigured",
    "HaltReasonHillClimbingMisconfigured",
    "HaltReasonRalphCompletionUnmet",
    "HaltReasonStrategyNotYetImplemented",
    "HaltReasonTaskGraphCycle",
    "HaltReasonTasksBlockedByFailure",
    "HaltReasonTerminationPolicyHalt",
    "HaltReasonToolErrorLoop",
    "HaltReasonUnrecoverableToolError",
    "Harness",
    "HarnessBuilder",
    "HarnessConfig",
    "HarnessRunOptions",
    "HarnessSignal",
    "HarnessSignalAbort",
    "HarnessSignalEnterPlanMode",
    "HarnessSignalExitPlanMode",
    "HarnessSignalSwitchMode",
    "HarnessStreamEvent",
    "HarnessToolResult",
    "HookPoint",
    "EscalationAction",
    "EscalationActionContinueWithBudget",
    "EscalationActionFail",
    "EscalationActionSkip",
    "HumanRequest",
    "HumanRequestBudgetExhausted",
    "HumanRequestClarification",
    "HumanRequestReview",
    "HumanRequestToolApproval",
    "HumanResponse",
    "HumanResponseAllow",
    "HumanResponseAllowWithModification",
    "HumanResponseAnswer",
    "HumanResponseApproveWithFeedback",
    "HumanResponseDeny",
    "HumanResponseEscalate",
    "HumanResponseHalt",
    "HumanResponseReject",
    "AgentRef",
    "BudgetContext",
    "BudgetExhausted",
    "BudgetStack",
    "ErrorRun",
    "ExecutionContext",
    "ExhaustedResolution",
    "ExhaustionCause",
    "ExhaustionCauseBudget",
    "ExhaustionCauseToolErrorLoop",
    "HillClimbingConfig",
    "HillClimbingDirection",
    "LoopStrategy",
    "PlanExecuteConfig",
    "RalphConfig",
    "ReactConfig",
    "RunScratch",
    "RunStrategy",
    "SchemaRef",
    "SelfVerifyingConfig",
    "SpanStack",
    "StrategyExecutor",
    "StrategyOutcome",
    "StrategyOutcomeBudgetExhausted",
    "StrategyOutcomeComplete",
    "StrategyOutcomeFailed",
    "StrategyRef",
    "StrategyRefBuiltIn",
    "StrategyRefCustom",
    "ToolsetRef",
    "loop_strategy_max_steps",
    "strategy_ref_max_steps",
    "run_strategy",
    "HarnessMiddlewareChain",
    "MiddlewareContinue",
    "MiddlewareContinueWithModification",
    "MiddlewareDecision",
    "MiddlewareForceAnotherTurn",
    "MiddlewareHalt",
    "MiddlewareSurfaceToHuman",
    "ModelConfig",
    "NoopContextManager",
    "OptimizationDirection",
    "PausedState",
    "RalphFeatureEntry",
    "RalphProgress",
    "RalphStopHook",
    "ReadOnlySandbox",
    "RiskLevel",
    "RunResult",
    "RunResultConsult",
    "RunResultEscalate",
    "RunResultFailure",
    "RunResultSuccess",
    "RunResultWaitingForHuman",
    "BwrapProfile",
    "IsolationMode",
    "IsolationModeBubblewrap",
    "IsolationModeDocker",
    # IsolationModeNone is intentionally NOT exported (issue #34). It is the
    # no-path-enforcement footgun, reachable only via the dangerous opt-in:
    # ``from spore_core.dangerous import IsolationModeNone``. The class stays
    # defined for the wire discriminated union; only its name is gated.
    "IsolationModeWorkspaceScoped",
    "NetworkPolicy",
    "NetworkPolicyAllowlist",
    "NetworkPolicyFull",
    "NetworkPolicyNone",
    "Operation",
    "SandboxDisallowedCommand",
    "SandboxExtensionDenied",
    "SandboxFileSizeExceeded",
    "SandboxNetworkViolation",
    "SandboxPathDenied",
    "SandboxPathEscape",
    "SandboxProvider",
    "SandboxReadOnlyViolation",
    "SandboxViolation",
    "WorkspaceConfig",
    "ScriptedMiddleware",
    "ScriptedSandbox",
    "ScriptedTerminationPolicy",
    "ScriptedToolRegistry",
    "SessionId",
    "SessionState",
    "StandardHarness",
    "BlockKind",
    "StreamBlockStart",
    "StreamBlockStop",
    "StreamBudgetWarning",
    "StreamFinalResponse",
    "StreamReasoningDelta",
    "StreamSink",
    "StreamTextDelta",
    "StreamToolArgsDelta",
    "StreamToolCall",
    "StreamToolCallStart",
    "StreamToolErrorLoopBroken",
    "StreamToolErrorLoopDetected",
    "StreamToolResult",
    "StreamTurnEnd",
    "StreamTurnStart",
    "StreamUserMessage",
    "TurnStreamState",
    "SEND_MESSAGE_TOOL_NAME",
    "Task",
    "TaskId",
    "TerminationContinue",
    "TerminationDecision",
    "TerminationHalt",
    "TerminationPolicy",
    "ToolOutput",
    "ToolOutputAwaitingClarification",
    "ToolOutputConsult",
    "ToolOutputError",
    "ToolOutputEscalate",
    "ToolOutputSuccess",
    "ToolOutputWaitingForHuman",
    "ToolRegistry",
    "VcsError",
    "VcsLogArgs",
    "VcsProvider",
    "GitVcsProvider",
    "FixtureVcsProvider",
    "new_session_id",
    "new_task_id",
    "sandbox_violation_is_always_halt",
]

# Avoid unused-import warnings for `Awaitable` (kept for IDE hover usefulness).
_: Awaitable[None] | None = None
