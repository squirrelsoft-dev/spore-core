"""End-to-end scenario assembly (issue #57).

Reusable wiring shared by the :mod:`spore_eval.e2e_agent` CLI AND the hermetic
integration tests, so a live run (``OllamaModelInterface`` + ``ModelAgent``) and
an offline run (``MockAgent`` + ``ScriptedToolRegistry``) drive the *same* code
path. :func:`build_scenario` is generic over the injected agent + harness tool
registry, so the only difference between live and mock mode is which
agent/registry you inject.

Architectural gaps closed here (mirroring the Rust reference,
``rust/crates/spore-core/src/scenarios.rs``):

* :class:`RealToolRegistry` bridges the two ``ToolRegistry`` surfaces: the
  harness loop calls :meth:`spore_core.harness.ToolRegistry.dispatch`
  (``ToolCall -> ToolOutput``, no sandbox arg), while the real tools live behind
  :meth:`spore_core.tool_registry.ToolRegistry.dispatch` (takes a sandbox). The
  bridge owns the inner ``StandardToolRegistry`` + a ``SandboxProvider`` and
  forwards, mapping a ``DispatchError`` onto a *recoverable* ``ToolOutputError``.
* :class:`SchemaInjectingContextManager` decorates any harness context manager
  so ``assemble().tools`` is populated from the registry's tool schemas (sorted
  by name for cache stability). Without it the compaction adapter surfaces no
  tools and the model can never emit a tool call.
* :class:`FailingTool` (``flaky_op``) always returns a *recoverable* error so
  the harness appends it as a tool result and the agent can adapt (scenario S4).
* :class:`CompleteOnFinalResponse` is a non-test termination policy: it lets the
  loop succeed as soon as the agent produces a final response.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from spore_core.cache_provider import CacheProvider
from spore_core.compaction_adapter import into_harness_adapter, seed_rich_state
from spore_core.context import CompactionConfig, StandardContextManager
from spore_core.context import SessionState as RichSessionState
from spore_core.harness import (
    Agent,
    BudgetSnapshot,
    CompactionTurn,
    ContextManager as HarnessContextManager,
    HarnessBuilder,
    HarnessToolResult,
    ObservabilityProvider,
    SandboxProvider,
    SessionId,
    SessionState as HarnessState,
    StandardHarness,
    Task,
    TaskId,
    TerminationContinue,
    TerminationDecision,
    ToolOutput,
    ToolOutputError,
)
from spore_core.model import (
    Message,
    ModelInterface,
    Role,
    TextContent,
    ToolCall,
    ToolSchema,
)
from spore_core.storage import MemoryStore, RunStore
from spore_core.tool_registry import (
    DispatchError,
    StandardToolRegistry,
    ToolAnnotations,
    ToolContext,
)
from spore_core.tool_registry import ToolSchema as RegistrySchema
from spore_core.agent import Context as AgentContext
from spore_tools.tools import (
    BashCommandTool,
    ExecTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)

# ============================================================================
# RealToolRegistry — bridge between the two ToolRegistry surfaces
# ============================================================================


class RealToolRegistry:
    """Bridges the harness-loop ``ToolRegistry`` onto the canonical
    :class:`spore_core.tool_registry.StandardToolRegistry`.

    The harness calls ``dispatch(ToolCall) -> ToolOutput`` with no sandbox (the
    sandbox is validated separately by the loop). This bridge forwards to the
    inner registry's ``dispatch(call, sandbox, ctx)`` and maps the result. A
    ``DispatchError`` becomes a *recoverable* ``ToolOutputError`` so the loop
    appends it as a tool result rather than halting — S4 depends on this.

    Storage seam (#75, #78)
    -----------------------
    Per the construction-injection decision, the bridge is given the run's
    :class:`SessionId`, a :class:`RunStore`, and (#78) a :class:`MemoryStore` at
    construction time (it is already built per-run). On each dispatch it forwards
    a :class:`ToolContext` built from those injected fields into the inner
    registry. This keeps the harness-loop ``dispatch(call)`` signature unchanged
    while threading storage to tools.
    """

    def __init__(
        self,
        inner: StandardToolRegistry,
        sandbox: SandboxProvider,
        session_id: SessionId,
        run_store: RunStore,
        memory_store: MemoryStore,
    ) -> None:
        self._inner = inner
        self._sandbox = sandbox
        self._ctx = ToolContext(
            session_id=session_id,
            run_store=run_store,
            memory_store=memory_store,
        )
        # Snapshot the model-facing schemas (sorted by name) once at
        # construction; the catalog is fixed for a scenario run.
        self._schemas: list[ToolSchema] = sorted(
            (s.to_model_schema() for s in inner.active_schemas(None)),
            key=lambda s: s.name,
        )

    def model_schemas(self) -> list[ToolSchema]:
        """The model-facing tool schemas, sorted by name."""
        return list(self._schemas)

    def tool_context(self) -> ToolContext:
        """The :class:`ToolContext` this bridge threads into every dispatch —
        exposing the ``session_id``, ``run_store`` and (#78) ``memory_store``
        seams it was wired with."""
        return self._ctx

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        try:
            result = await self._inner.dispatch(call, self._sandbox, self._ctx)
        except DispatchError as err:
            return ToolOutputError(message=f"dispatch failed: {err}", recoverable=True)
        return result.output

    def is_always_halt(self, tool_name: str) -> bool:
        # No bridged tool is always-halt — S4 needs recoverable failure.
        _ = tool_name
        return False

    def schemas(self) -> list[ToolSchema]:
        return list(self._schemas)


# ============================================================================
# SchemaInjectingContextManager — fills assemble().tools from the registry
# ============================================================================


#: Operational system prompt for the live agent. The compaction adapter's
#: ``assemble`` produces a context with **no system prompt** (it has no
#: ``ContextSources`` to render one), so without this the model receives only
#: the task as a user message and no guidance on how to behave. The three rules
#: target the failure modes observed with small local models: describing actions
#: instead of taking them, passing stringified arguments, and declaring success
#: without checking the result.
AGENT_SYSTEM_PROMPT = (
    "You are an autonomous agent that completes tasks by calling the provided "
    "tools. Follow these rules:\n"
    "\n"
    "1. ACT, DON'T DESCRIBE. To make something happen, call the appropriate "
    "tool. Writing a shell command, code snippet, or file contents into your "
    "text reply does NOT run it — only a real tool call has any effect. When a "
    "task asks you to produce a file or a result, call the tool that performs "
    "the action and let the tool do the work; never paste the command, code, or "
    "expression you *would* run as if it were the finished result.\n"
    "\n"
    "2. USE CORRECTLY-TYPED ARGUMENTS. Pass tool arguments as typed JSON: "
    'booleans as true/false (not "true"), numbers as 12 (not "12"), lists as '
    '["a"] (not "[\\"a\\"]"). Quoted-string scalars where a bool/number/array '
    "is expected will be rejected.\n"
    "\n"
    "3. VERIFY BEFORE FINISHING. Before replying DONE, confirm your work "
    "actually satisfies the request. If you wrote a file, read it back with "
    "read_file and check its contents are exactly what was asked. If they do "
    "not match, fix it and verify again. Only reply DONE once you have verified "
    "the result is correct."
)


class SchemaInjectingContextManager:
    """Decorates a harness :class:`ContextManager`, delegating every seam
    method to the inner manager but injecting the registry's tool schemas into
    ``assemble().tools`` and prepending :data:`AGENT_SYSTEM_PROMPT`. The
    compaction adapter's ``assemble`` returns an empty tool list and no system
    prompt, so without this decorator the model never sees any tools (and can
    never emit a tool call) nor any operational guidance in live mode.
    """

    def __init__(self, inner: HarnessContextManager, tools: list[ToolSchema]) -> None:
        self._inner = inner
        self._tools = sorted(tools, key=lambda s: s.name)

    async def assemble(self, session: HarnessState, task: Task) -> AgentContext:
        ctx = await self._inner.assemble(session, task)
        ctx.tools = list(self._tools)
        # Prepend the operational system prompt. The adapter's assemble yields
        # none, so the model would otherwise get no guidance. Guard against
        # duplicates so a resumed/seeded session that already leads with a System
        # message isn't given two.
        if not (ctx.messages and ctx.messages[0].role == Role.SYSTEM):
            ctx.messages.insert(
                0,
                Message(role=Role.SYSTEM, content=TextContent(text=AGENT_SYSTEM_PROMPT)),
            )
        return ctx

    async def append_tool_result(self, session: HarnessState, result: HarnessToolResult) -> None:
        await self._inner.append_tool_result(session, result)

    async def append_assistant_message(self, session: HarnessState, message: Message) -> None:
        # DELEGATE to the inner manager. The harness loop calls this outer
        # decorator; without forwarding it the no-op Protocol default would run
        # and the assistant-turn-recording fix would be dead. Probe via
        # ``getattr`` mirroring this class's other optional-capability forwarding,
        # since structural inner managers may not define the method.
        appender = getattr(self._inner, "append_assistant_message", None)
        if appender is not None:
            await appender(session, message)

    async def append_user_message(self, session: HarnessState, text: str) -> None:
        await self._inner.append_user_message(session, text)

    def should_compact(self, session: HarnessState) -> bool:
        return self._inner.should_compact(session)

    def prepare_compaction_turn(self, session: HarnessState) -> CompactionTurn | None:
        return self._inner.prepare_compaction_turn(session)

    def inject_missing_items(self, context: AgentContext, missing: list[str]) -> None:
        inject = getattr(self._inner, "inject_missing_items", None)
        if inject is not None:
            inject(context, missing)
        else:
            context.messages.append(
                Message(
                    role=Role.USER,
                    content=TextContent(
                        text=(
                            f"Your summary is missing these items: {', '.join(missing)}. "
                            "Please revise."
                        )
                    ),
                )
            )

    def apply_compaction(self, session: HarnessState, summary: str) -> None:
        self._inner.apply_compaction(session, summary)

    def token_budget_used(self, session: HarnessState) -> int | None:
        seam = getattr(self._inner, "token_budget_used", None)
        if seam is None:
            return None
        return seam(session)


# ============================================================================
# FailingTool — deliberately-failing recoverable tool (S4)
# ============================================================================


class FailingTool:
    """A tool that always fails with a *recoverable* error. Used by scenario S4
    to prove the loop surfaces a tool error to the agent and lets it adapt
    rather than crashing or hanging. Must NOT be ``is_always_halt``."""

    NAME = "flaky_op"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> RegistrySchema:
        return RegistrySchema(
            name=cls.NAME,
            description="A flaky operation that fails the first time it is called",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
            },
            annotations=ToolAnnotations(idempotent=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        _ = (call, sandbox, ctx)
        return ToolOutputError(
            message="flaky_op is unavailable right now; try a different approach",
            recoverable=True,
        )


# ============================================================================
# CompleteOnFinalResponse — non-test termination policy
# ============================================================================


class CompleteOnFinalResponse:
    """Termination policy that lets the loop complete as soon as the agent
    produces a final response (always ``Continue``, which the harness
    interprets as "accept the final response and succeed"). Lives on the live
    path with no test-only dependency."""

    async def evaluate(
        self, session: HarnessState, budget_used: BudgetSnapshot
    ) -> TerminationDecision:
        _ = (session, budget_used)
        return TerminationContinue()


# ============================================================================
# Real tool registry construction
# ============================================================================


def build_real_tool_registry(scenario: ScenarioId) -> StandardToolRegistry:
    """Build a :class:`StandardToolRegistry` for ``scenario``. The base catalog
    is always ``read_file``, ``write_file``, ``list_dir``, ``exec``, and the
    :class:`FailingTool` (``flaky_op``). The real shell tool ``bash_command`` is
    added ONLY for :attr:`ScenarioId.S5` — S1/S2 measure reasoning +
    act-don't-describe, and a live model handed a shell could shortcut S1 with
    ``cat … | tr … > …`` without demonstrating the intended behavior. ``exec``
    is safe everywhere because it cannot pipe or redirect. Registration errors
    here are programming errors (duplicate/invalid schema) and propagate."""
    registry = StandardToolRegistry()
    registry.register(ReadFileTool(), ReadFileTool.schema())
    registry.register(WriteFileTool(), WriteFileTool.schema())
    registry.register(ListDirTool(), ListDirTool.schema())
    registry.register(ExecTool(), ExecTool.schema())
    registry.register(FailingTool(), FailingTool.schema())
    if scenario is ScenarioId.S5:
        registry.register(BashCommandTool(), BashCommandTool.schema())
    return registry


# ============================================================================
# Rich context-manager assembly (live compaction)
# ============================================================================


def build_rich_context_manager(
    model: ModelInterface,
    cache: CacheProvider,
    config: CompactionConfig,
) -> HarnessContextManager:
    """Build a real compaction-capable context manager: a
    :class:`StandardContextManager` wrapped in the
    :class:`StandardCompactionAdapter` (via :func:`into_harness_adapter`).
    Generic over the model so live mode passes the Ollama model and tests pass
    a stub."""
    return into_harness_adapter(StandardContextManager(model, cache, config))


def seed_compaction_state(
    session: HarnessState,
    task_instruction: str,
    session_id: SessionId,
    task_id: TaskId,
    window_limit: int,
    token_budget_used: int,
    history_len: int,
) -> None:
    """Seed a harness :class:`SessionState` with rich compaction state for the
    S3 scenario: a small window, a budget near the threshold, and a history
    longer than ``preserve_recent_n`` so compaction fires mid-run. The session
    can then compact, continue, and compact again (healthy multi-compaction)
    because the token-accounting fix decrements the budget on each
    compaction."""
    rich = RichSessionState(
        session_id=session_id,
        task_id=task_id,
        task_instruction=task_instruction,
    )
    rich.window_limit = window_limit
    rich.token_budget_used = token_budget_used
    rich.message_history = [
        Message(
            role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
            content=TextContent(
                text=(
                    f"history message {i}: progress notes on the payment service deploy with "
                    "enough content to carry a meaningful token estimate for reclamation"
                )
            ),
        )
        for i in range(history_len)
    ]
    seed_rich_state(session, rich)


# ============================================================================
# Scenario ids + prompts
# ============================================================================


class ScenarioId(str, Enum):
    """The scenario id, parsed from the CLI arg ``s1``..``s5``."""

    S1 = "s1"
    S2 = "s2"
    S3 = "s3"
    S4 = "s4"
    S5 = "s5"

    @classmethod
    def parse(cls, s: str) -> ScenarioId | None:
        try:
            return cls(s.strip().lower())
        except ValueError:
            return None

    def prompt(self) -> str:
        return _PROMPTS[self]


_PROMPTS: dict[ScenarioId, str] = {
    ScenarioId.S1: (
        "Complete this task step by step, using the provided tools:\n"
        "1. Call read_file to read the contents of input.txt. Use the exact text "
        "it returns — do not invent or substitute any text.\n"
        "2. Take that exact text and rewrite it with every lowercase letter "
        "changed to its capital form, keeping all other characters, spaces, and "
        "punctuation the same.\n"
        "3. Call write_file with path 'output.txt' and content set to the "
        "uppercased text from step 2 — the literal capital letters themselves. "
        "The content must be the transformed words from input.txt, NOT a shell "
        "command, NOT a $(...) expression, and NOT any code.\n"
        "4. Call read_file on output.txt and check its contents equal the "
        "uppercased text from step 2.\n"
        "Reply DONE only once output.txt contains input.txt's contents in all "
        "capital letters."
    ),
    ScenarioId.S2: (
        "Create a file notes.md containing a TODO list with one item: 'set up the "
        "project'. Use write_file. Reply DONE when written."
    ),
    ScenarioId.S3: (
        "Summarize the long conversation so far and continue working on the deploy of "
        "the payment service. Reply DONE when finished."
    ),
    ScenarioId.S4: (
        "Call the flaky_op tool. If it fails, do not give up: write a file "
        "recovered.txt explaining that flaky_op failed and how you adapted, using "
        "write_file. Reply DONE when finished."
    ),
    ScenarioId.S5: (
        "Transform input.txt into output.txt with every lowercase letter "
        "uppercased, using the shell.\n"
        "1. Call bash_command with a real shell pipeline that reads input.txt, "
        "uppercases it, and writes output.txt — e.g. "
        "`cat input.txt | tr a-z A-Z > output.txt`. This is exactly what the "
        "bash_command tool is for: it runs your script via /bin/sh -c, so pipes "
        "(|) and redirects (>) work.\n"
        "2. Call read_file on output.txt and check its contents are input.txt's "
        "text in all capital letters.\n"
        "Reply DONE only once output.txt contains the uppercased text."
    ),
}


# ============================================================================
# Scenario builder (generic over agent + tool registry)
# ============================================================================


@dataclass
class ScenarioComponents:
    """The injected components a scenario harness is assembled from. Live mode
    fills these from Ollama + the real registry; mock mode from a scripted
    agent + scripted registry, so both share :func:`build_scenario`."""

    agent: Agent
    tools: object  # harness ToolRegistry (structural)
    sandbox: SandboxProvider
    context_manager: HarnessContextManager
    termination_policy: object  # harness TerminationPolicy (structural)
    tool_schemas: list[ToolSchema]
    observability: ObservabilityProvider | None = None


def build_scenario(scenario: ScenarioId, components: ScenarioComponents) -> StandardHarness:
    """Assemble a :class:`StandardHarness` for the given scenario from injected
    components. Generic over the agent and tool registry so live mode
    (``OllamaModelInterface``/``ModelAgent`` + :class:`RealToolRegistry`) and
    mock mode (``MockAgent`` + ``ScriptedToolRegistry``) share one code path.

    ``tool_schemas`` are injected into every assembled context (sorted by name)
    via :class:`SchemaInjectingContextManager`. Pass the registry's schemas in
    live mode, or an empty list in mock mode where the scripted agent does not
    need them.
    """
    _ = scenario
    context_manager: HarnessContextManager = SchemaInjectingContextManager(
        components.context_manager, components.tool_schemas
    )
    builder = HarnessBuilder(
        components.agent,
        components.tools,
        components.sandbox,
        context_manager,
        components.termination_policy,
    )
    if components.observability is not None:
        builder = builder.observability(components.observability)
    return builder.build()


__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "CompleteOnFinalResponse",
    "FailingTool",
    "RealToolRegistry",
    "ScenarioComponents",
    "ScenarioId",
    "SchemaInjectingContextManager",
    "build_real_tool_registry",
    "build_rich_context_manager",
    "build_scenario",
    "seed_compaction_state",
]
