"""ToolRegistry — maintains available tools and dispatches tool calls (issue #4).

Python implementation mirroring ``rust/crates/spore-core/src/tool_registry.rs``.
The registry holds the catalog of :class:`Tool` implementations, validates
their JSON schemas at registration time, and dispatches :class:`ToolCall`s
coming in from the agent — passing every tool a :class:`SandboxProvider`
so that no tool ever touches the environment directly.

What this component does:

* Register tools with their schemas (validated up-front)
* Manage named :class:`ToolSet` groupings keyed by :class:`TaskPhase`
* Return active schemas for a given phase (sorted by name for cache-stability)
* Dispatch a single call (sandbox-aware) or many calls (concurrent where
  :class:`ToolAnnotations` permit)
* Expose :meth:`ToolRegistry.has_subagent_tools` so ``SubagentTool``
  (issue #5) can enforce the depth-1 rule at construction time

Storage seam — :class:`ToolContext` (#75)
-----------------------------------------
Tools receive a :class:`ToolContext` on every dispatch, *in addition to* the
:class:`SandboxProvider`. :class:`ToolContext` is the storage seam:
``{session_id: SessionId, run_store: RunStore}``. The new
:meth:`Tool.execute` signature is ``execute(call, sandbox, ctx)`` — ``ctx`` is
added *after* the sandbox (storage is additive; the sandbox is NOT folded in).
:meth:`StandardToolRegistry.dispatch` / :meth:`dispatch_all` thread the
:class:`ToolContext` through to every tool. The harness-side :class:`RealToolRegistry`
bridge (defined below) is constructed per-run with the :class:`SessionId` +
:class:`RunStore` + :class:`MemoryStore` and builds the :class:`ToolContext`
itself before forwarding; the harness-loop ``dispatch(call)`` signature is
UNCHANGED.
:class:`ToolContext` is a dataclass so future fields are non-breaking.

What this component does NOT do:

* Retry recoverable failures (middleware concern — issue #11)
* Maintain conversation state, budgets, or termination policy
* Interpret :class:`ToolOutputWaitingForHuman` — the registry returns it
  verbatim; the harness loop assembles the combined :class:`PausedState`

Rules enforced here (mirror Rust reference byte-for-byte):

1. Tools are always dispatched via the registry — never directly.
2. Schemas are validated at registration (basic structural check on the
   JSON Schema document: nonempty name + ``parameters`` dict with a
   top-level ``"type"`` key).
3. Duplicate tool names → LAST-WINS upsert: the later ``register`` overwrites
   the earlier one (issue #81, Q1). ``register_set`` still rejects duplicate
   SET names with :class:`RegistrationError` (kind=DuplicateName).
4. ``ToolAnnotations(read_only=True, destructive=True)`` is contradictory →
   :class:`RegistrationError` (kind=ConflictingAnnotations).
5. Active :class:`ToolSet` can change between turns (selected by
   :class:`TaskPhase`).
6. Unregistered tool call → :class:`DispatchError` (kind=UnregisteredTool).
7. Missing ``required`` field in input → :class:`DispatchError`
   (kind=SchemaValidationFailed).
8. ``dispatch_all``:

   * Calls whose tools are ``read_only=True`` (and neither destructive nor
     open-world) execute concurrently.
   * ``destructive`` and ``open_world`` calls execute sequentially.
   * Result order matches input order.

9. :meth:`ToolRegistry.has_subagent_tools` reads each tool's
   ``is_subagent_tool`` flag so the rule can be checked at construction
   time, not at dispatch time.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import anyio

from .errors import SporeError
from .harness import (
    HarnessToolResult as ToolResult,
)
from .harness import (
    BaseSandboxProvider,
    CommandProgressSink,
    SandboxProvider,
    SandboxViolation,
    SessionId,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from .model import ToolCall
from .model import ToolSchema as ModelToolSchema

if TYPE_CHECKING:
    from .harness import ToolRegistry as HarnessLoopToolRegistry
    from .storage import MemoryStore, ProjectId, RunStore

# ============================================================================
# SC-30 — read-only eval-phase allow-list
# ============================================================================

#: SC-30: the read-only tool names auto-derived for the SelfVerifying evaluate
#: phase when ``eval_toolset`` is empty. These are exactly the names produced by
#: ``StandardTools.readonly_set()`` in ``spore_tools``. The constant is hardcoded
#: HERE (in ``spore_core``) rather than imported from ``spore_tools`` because
#: ``spore_tools`` DEPENDS ON ``spore_core`` — importing it at module load would
#: risk an import cycle. A drift-guard test (in ``spore_tools``, which already
#: imports core) asserts this set equals the live ``readonly_set()`` names so the
#: two never silently diverge.
READONLY_EVAL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read_file",
        "list_dir",
        "grep_files",
        "grep",
        "find_files",
        "web_fetch",
        "web_search",
    }
)


# ============================================================================
# ToolContext — the storage seam handed to every tool (#75)
# ============================================================================


@dataclass(frozen=True)
class ToolContext:
    """The per-dispatch storage seam handed to every :meth:`Tool.execute` call,
    alongside (but separate from) the :class:`SandboxProvider`.

    It carries the minimum a tool needs to persist durable state via the
    storage layer:

    * ``session_id`` — the run's :class:`SessionId`, the key namespace for
      EPHEMERAL state that SHOULD reset per Ralph window (conversation/session
      state, ``active_skills``).
    * ``project_id`` — the stable :class:`~spore_core.storage.ProjectId` (#142),
      the key namespace for DURABLE artifacts (the ``task_list``, plan artifact,
      …) that must survive a Ralph window reset (fresh :class:`SessionId` per
      window) and process restarts. Durable tools key the :class:`RunStore` by
      ``project_namespace(ctx.project_id)`` instead of ``session_id`` so the
      prior window's list is re-read, not orphaned.
    * ``run_store`` — the :class:`RunStore` domain of the configured storage
      provider.
    * ``memory_store`` — the :class:`MemoryStore` domain (#78). Scope-aware: the
      tool passes a :class:`StorageScope` on every call. For a composite
      provider this is the scope-routing memory slot; for the never-null
      contract it is at worst a :class:`NoOpStorageProvider`. ``MemoryTool``
      (#82) picks up this already-threaded seam.
    * ``progress`` — optional run-level live-progress sink (pi-style bash
      streaming). Set by the harness bridge
      (:meth:`RealToolRegistry.with_progress`) when the run has a stream sink;
      tools derive a per-call emitter via :meth:`command_progress`. ``None``
      (default) means no consumer is listening — tools skip streaming entirely.

    It is a **dataclass** (not a tuple) so future fields can be added without
    breaking the trait signature again. The :class:`SandboxProvider` is
    intentionally NOT folded in here — storage is additive; tools still receive
    the sandbox as its own parameter (some tools need the filesystem sandbox and
    no storage).
    """

    session_id: SessionId
    project_id: ProjectId
    run_store: RunStore
    memory_store: MemoryStore
    progress: ToolProgressSink | None = None

    def command_progress(self, call: ToolCall) -> CommandProgressSink | None:
        """Derive a per-call :class:`CommandProgressSink` that forwards tail
        snapshots to the run's :class:`ToolProgressSink`, labeled with this
        call's tool name + id. Returns ``None`` when no run sink is wired so
        tools skip streaming. Mirrors Rust's ``ToolContext::command_progress``.
        """
        sink = self.progress
        if sink is None:
            return None
        tool = call.name
        call_id = call.id

        def _emit(preview: str) -> None:
            sink(ToolProgressUpdate(tool=tool, call_id=call_id, preview=preview))

        return _emit


@dataclass
class ToolProgressUpdate:
    """One live-progress update from a RUNNING tool, labeled with the call it
    belongs to. The harness bridge maps it onto
    :class:`spore_core.harness.StreamToolProgress` for UIs. Mirrors Rust's
    ``ToolProgressUpdate``.
    """

    tool: str
    call_id: str
    preview: str


#: Run-level sink for :class:`ToolProgressUpdate`-es. Must not block (called
#: from the executor's ~100ms wait-loop tick). Mirrors Rust's
#: ``ToolProgressSink``.
ToolProgressSink = Callable[[ToolProgressUpdate], None]


# ============================================================================
# ToolAnnotations & ToolSchema (registry-side, richer than model.ToolSchema)
# ============================================================================


@dataclass(frozen=True)
class ToolAnnotations:
    """Behavioural annotations attached to a registered tool.

    Drives the :meth:`ToolRegistry.dispatch_all` concurrency split and the
    auto-derived ``RiskLevel`` used by ``PermissionMiddleware`` (issue #11).
    """

    read_only: bool = False
    destructive: bool = False
    idempotent: bool = False
    open_world: bool = False


@dataclass
class ToolSchema:
    """Canonical schema for a registered tool.

    Distinct from :class:`spore_core.model.ToolSchema` (the minimal subset
    shipped to the LLM) — this one carries :class:`ToolAnnotations` and is
    the registry-side type.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    annotations: ToolAnnotations = field(default_factory=ToolAnnotations)

    def to_model_schema(self) -> ModelToolSchema:
        """Project to the slimmer :class:`spore_core.model.ToolSchema`
        used in :class:`ModelRequest`."""

        return ModelToolSchema(
            name=self.name,
            description=self.description,
            input_schema=dict(self.parameters),
        )


# ============================================================================
# TaskPhase & ToolSet
# ============================================================================


class TaskPhase(str, Enum):
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    CLEANUP = "cleanup"


@dataclass
class ToolSet:
    """A named grouping of tools. ``phase`` is ``None`` if always active."""

    name: str
    tools: list[str] = field(default_factory=list)
    phase: TaskPhase | None = None


# ============================================================================
# Errors
# ============================================================================


class RegistrationError(SporeError):
    """Registration-time failure. Discriminated by :attr:`kind`."""

    kind: ClassVar[str] = "RegistrationError"

    def __init__(self, kind: str, tool: str, reason: str | None = None) -> None:
        self.kind = kind
        self.tool = tool
        self.reason = reason or ""
        msg = f"{kind}: tool={tool}"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg)

    @classmethod
    def invalid_schema(cls, tool: str, reason: str) -> RegistrationError:
        return cls("InvalidSchema", tool, reason)

    @classmethod
    def duplicate_name(cls, tool: str) -> RegistrationError:
        return cls("DuplicateName", tool)

    @classmethod
    def conflicting_annotations(cls, tool: str, reason: str) -> RegistrationError:
        return cls("ConflictingAnnotations", tool, reason)


class DispatchError(SporeError):
    """Dispatch-time failure. Discriminated by :attr:`kind`."""

    kind: ClassVar[str] = "DispatchError"

    def __init__(
        self,
        kind: str,
        *,
        tool: str | None = None,
        reason: str | None = None,
        violation: SandboxViolation | None = None,
    ) -> None:
        self.kind = kind
        self.tool = tool
        self.reason = reason
        self.violation = violation
        parts: list[str] = [kind]
        if tool is not None:
            parts.append(f"tool={tool}")
        if reason is not None:
            parts.append(reason)
        if violation is not None:
            parts.append(f"violation={violation.kind}")
        super().__init__(": ".join(parts))

    @classmethod
    def unregistered_tool(cls, name: str) -> DispatchError:
        return cls("UnregisteredTool", tool=name)

    @classmethod
    def schema_validation_failed(cls, tool: str, reason: str) -> DispatchError:
        return cls("SchemaValidationFailed", tool=tool, reason=reason)

    @classmethod
    def sandbox_violation(cls, violation: SandboxViolation) -> DispatchError:
        return cls("SandboxViolation", violation=violation)

    @classmethod
    def tool_execution_failed(cls, tool: str, error: str) -> DispatchError:
        return cls("ToolExecutionFailed", tool=tool, reason=error)


# ============================================================================
# Tool protocol
# ============================================================================


@runtime_checkable
class Tool(Protocol):
    """A single tool implementation.

    Tools are stateless and receive a :class:`SandboxProvider` (environment
    seam) and a :class:`ToolContext` (storage seam) on every dispatch. The
    protocol is structural — concrete impls do not inherit.
    """

    def name(self) -> str: ...

    def is_subagent_tool(self) -> bool: ...

    def may_produce_large_output(self) -> bool: ...

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput: ...


# ============================================================================
# ToolRegistry protocol
# ============================================================================


@runtime_checkable
class ToolRegistry(Protocol):
    """Canonical registry protocol."""

    def register(self, tool: Tool, schema: ToolSchema) -> None: ...

    def register_set(self, set_: ToolSet) -> None: ...

    def active_schemas(self, phase: TaskPhase | None) -> list[ToolSchema]: ...

    async def dispatch(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolResult: ...

    async def dispatch_all(
        self, calls: list[ToolCall], sandbox: SandboxProvider, ctx: ToolContext
    ) -> list[ToolResult | DispatchError]: ...

    def has_subagent_tools(self) -> bool: ...


# ============================================================================
# StandardToolRegistry — canonical implementation
# ============================================================================


@dataclass
class _Registered:
    tool: Tool
    schema: ToolSchema


class StandardToolRegistry:
    """Default in-memory registry."""

    def __init__(self) -> None:
        self._tools: dict[str, _Registered] = {}
        self._sets: list[ToolSet] = []

    # ---- validation helpers ----------------------------------------------

    @staticmethod
    def _validate_schema(schema: ToolSchema) -> None:
        if not schema.name:
            raise RegistrationError.invalid_schema(schema.name, "name must not be empty")
        if not isinstance(schema.parameters, dict):
            raise RegistrationError.invalid_schema(schema.name, "parameters must be a JSON object")
        if "type" not in schema.parameters:
            raise RegistrationError.invalid_schema(
                schema.name, "parameters must declare a top-level `type`"
            )

    @staticmethod
    def _validate_annotations(schema: ToolSchema) -> None:
        a = schema.annotations
        if a.read_only and a.destructive:
            raise RegistrationError.conflicting_annotations(
                schema.name,
                "read_only and destructive are mutually exclusive",
            )

    @staticmethod
    def _validate_input(schema: ToolSchema, call: ToolCall) -> None:
        if not isinstance(call.input, dict):
            raise DispatchError.schema_validation_failed(schema.name, "input must be a JSON object")
        params = schema.parameters
        if not isinstance(params, dict):
            return
        required = params.get("required")
        if isinstance(required, list):
            for field_name in required:
                if isinstance(field_name, str) and field_name not in call.input:
                    raise DispatchError.schema_validation_failed(
                        schema.name, f"missing required field `{field_name}`"
                    )

    # ---- ToolRegistry surface --------------------------------------------

    def register(self, tool: Tool, schema: ToolSchema) -> None:
        if tool.name() != schema.name:
            raise RegistrationError.invalid_schema(
                schema.name,
                f"tool name `{tool.name()}` does not match schema name `{schema.name}`",
            )
        self._validate_schema(schema)
        self._validate_annotations(schema)
        # Last-wins upsert (issue #81, Q1): registering a tool with the same
        # name as an existing one OVERWRITES it. This is what lets an architect
        # override a standard catalogue tool by registering their own after a
        # preset (e.g. ``StandardTools.coding_set()``). ``register_set`` keeps
        # the duplicate-name error for intra-set duplicates.
        self._tools[schema.name] = _Registered(tool=tool, schema=schema)

    def register_set(self, set_: ToolSet) -> None:
        if not set_.name:
            raise RegistrationError.invalid_schema(set_.name, "tool set name must not be empty")
        if any(s.name == set_.name for s in self._sets):
            raise RegistrationError.duplicate_name(set_.name)
        self._sets.append(set_)

    def active_schemas(self, phase: TaskPhase | None) -> list[ToolSchema]:
        if phase is None:
            schemas = [r.schema for r in self._tools.values()]
        else:
            matching = [s for s in self._sets if s.phase is None or s.phase == phase]
            if not matching:
                # Fallback: no set registered → expose all tools (do not
                # silently mask every tool just because no set exists).
                schemas = [r.schema for r in self._tools.values()]
            else:
                names: set[str] = set()
                for s in matching:
                    for t in s.tools:
                        names.add(t)
                schemas = [self._tools[n].schema for n in sorted(names) if n in self._tools]
        schemas.sort(key=lambda s: s.name)
        return schemas

    async def dispatch(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolResult:
        entry = self._tools.get(call.name)
        if entry is None:
            raise DispatchError.unregistered_tool(call.name)

        # Sandbox first (Layer-1 violations propagate to the harness).
        violation = await sandbox.validate(call)
        if violation is not None:
            raise DispatchError.sandbox_violation(violation)

        self._validate_input(entry.schema, call)

        output = await entry.tool.execute(call, sandbox, ctx)
        return ToolResult(call_id=call.id, output=output)

    async def dispatch_all(
        self, calls: list[ToolCall], sandbox: SandboxProvider, ctx: ToolContext
    ) -> list[ToolResult | DispatchError]:
        # Classify each call. Unknown tools schedule sequentially so their
        # error surfaces deterministically alongside other sequential
        # failures.
        concurrent_flags: list[bool] = []
        for call in calls:
            entry = self._tools.get(call.name)
            if entry is None:
                concurrent_flags.append(False)
                continue
            a = entry.schema.annotations
            concurrent_flags.append(a.read_only and not a.destructive and not a.open_world)

        results: list[ToolResult | DispatchError | None] = [None] * len(calls)

        async def run_one(idx: int) -> None:
            try:
                results[idx] = await self.dispatch(calls[idx], sandbox, ctx)
            except DispatchError as e:
                results[idx] = e

        # Concurrent batch: dispatch read-only calls under a task group.
        concurrent_idx = [i for i, c in enumerate(concurrent_flags) if c]
        sequential_idx = [i for i, c in enumerate(concurrent_flags) if not c]

        if concurrent_idx:
            async with anyio.create_task_group() as tg:
                for i in concurrent_idx:
                    tg.start_soon(run_one, i)

        # Sequential batch — preserves caller-visible order alongside the
        # already-filled concurrent slots.
        for i in sequential_idx:
            await run_one(i)

        # All slots filled by construction.
        final: list[ToolResult | DispatchError] = []
        for r in results:
            assert r is not None  # noqa: S101 — invariant: every slot filled
            final.append(r)
        return final

    def has_subagent_tools(self) -> bool:
        return any(r.tool.is_subagent_tool() for r in self._tools.values())


# ============================================================================
# RealToolRegistry — the bridge between the two ToolRegistry surfaces
# ============================================================================


class RealToolRegistry:
    """Bridges the harness-loop :class:`spore_core.harness.ToolRegistry` onto
    this canonical :class:`ToolRegistry` (a populated
    :class:`StandardToolRegistry`).

    This is **the production wiring** for running catalogue / :class:`Tool`-based
    tools inside the harness — not test scaffolding. The harness loop calls
    ``dispatch(ToolCall) -> ToolOutput`` with no sandbox or storage; this bridge
    forwards to the inner registry's ``dispatch(call, sandbox, ctx)``, threading
    the :class:`SandboxProvider` and a per-run :class:`ToolContext` (storage
    seam, #75) it was constructed with. A :class:`DispatchError` becomes a
    **recoverable** :class:`ToolOutputError` so the loop appends it as a tool
    result and lets the agent adapt rather than halting.

    It is built **once per run**: ``session_id``, ``project_id`` (#142),
    ``run_store``, and ``memory_store`` are injected at construction (the run's
    :class:`SessionId` is only known at ``run()``-time) and used to build the
    :class:`ToolContext` forwarded on every dispatch. ``project_id`` is the
    STABLE durable namespace; ``session_id`` is the per-window ephemeral key.
    :class:`~spore_core.harness.HarnessBuilder` wires this automatically when
    catalogue tools are added via ``.tool()`` / ``.tools()``; construct it
    directly only when supplying your own :class:`StandardToolRegistry`.
    """

    def __init__(
        self,
        inner: StandardToolRegistry,
        sandbox: SandboxProvider,
        session_id: SessionId,
        project_id: ProjectId,
        run_store: RunStore,
        memory_store: MemoryStore,
    ) -> None:
        self._inner = inner
        self._sandbox = sandbox
        self._ctx = ToolContext(
            session_id=session_id,
            project_id=project_id,
            run_store=run_store,
            memory_store=memory_store,
        )
        # Snapshot the model-facing schemas (sorted by name) once at
        # construction; the catalogue is fixed for a run.
        self._schemas: list[ModelToolSchema] = sorted(
            (s.to_model_schema() for s in inner.active_schemas(None)),
            key=lambda s: s.name,
        )

    def model_schemas(self) -> list[ModelToolSchema]:
        """The model-facing tool schemas, sorted by name."""
        return list(self._schemas)

    def tool_context(self) -> ToolContext:
        """The :class:`ToolContext` this bridge threads into every dispatch —
        exposing the ``session_id``, ``run_store`` and (#78) ``memory_store``
        seams it was wired with. Lets callers verify the storage seams are
        live."""
        return self._ctx

    def with_progress(self, sink: ToolProgressSink | None) -> RealToolRegistry:
        """Wire a run-level :class:`ToolProgressSink` into the threaded
        :class:`ToolContext` so tools can emit live progress. Mirrors Rust's
        ``RealToolRegistry::with_progress``."""
        self._ctx = replace(self._ctx, progress=sink)
        return self

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        try:
            result = await self._inner.dispatch(call, self._sandbox, self._ctx)
        except DispatchError as err:
            # Recoverable so the loop appends the error and lets the agent
            # adapt — S4 depends on this.
            return ToolOutputError(message=f"dispatch failed: {err}", recoverable=True)
        return result.output

    def is_always_halt(self, tool_name: str) -> bool:
        # No bridged tool is always-halt — S4 needs recoverable failure.
        _ = tool_name
        return False

    def schemas(self) -> list[ModelToolSchema]:
        return list(self._schemas)


# ============================================================================
# ToolSearchRegistry — deferred-tool pattern for local models (issue #looper)
# ============================================================================


@dataclass
class ToolNamespace:
    """A namespace grouping: a cheap header advertised to the model; calling
    it loads the member tools. Mirrors Rust's ``ToolNamespace``."""

    name: str
    description: str
    tools: list[str]

    @staticmethod
    def of(name: str, description: str, tools: list[str]) -> ToolNamespace:
        return ToolNamespace(name=name, description=description, tools=list(tools))


@dataclass
class ToolSearchConfig:
    """Configuration for :class:`ToolSearchRegistry` (mirrors Rust's
    ``ToolSearchConfig``): which tools are always advertised, the deferred-tool
    namespaces, the description-prose byte budget, and how many tools a single
    ``tool_search`` call surfaces."""

    core: list[str]
    namespaces: list[ToolNamespace] = field(default_factory=list)
    description_budget: int = 4096
    search_top_k: int = 5
    search_tool_name: str = "tool_search"

    @staticmethod
    def new(core: list[str]) -> ToolSearchConfig:
        return ToolSearchConfig(core=list(core))

    def namespace(self, ns: ToolNamespace) -> ToolSearchConfig:
        self.namespaces.append(ns)
        return self

    def with_description_budget(self, bytes_: int) -> ToolSearchConfig:
        self.description_budget = bytes_
        return self

    def with_search_top_k(self, k: int) -> ToolSearchConfig:
        self.search_top_k = k
        return self

    def with_search_tool_name(self, name: str) -> ToolSearchConfig:
        self.search_tool_name = name
        return self


class ToolSearchRegistry:
    """A harness-loop :class:`~spore_core.harness.ToolRegistry` wrapper that
    advertises only a small, description-bounded ACTIVE set — the configured
    core tools plus a ``tool_search`` meta-tool (and namespace headers) — and
    loads the rest of the catalogue on demand. Mirrors Rust's
    ``ToolSearchRegistry``.

    ``schemas()`` is re-read by the harness every turn, so the wrapper
    grows/shrinks the advertised set turn-to-turn. ``dispatch`` of any
    non-search, non-namespace tool delegates straight to the inner registry, so
    a model that calls a catalogue tool that is not currently advertised STILL
    succeeds — exposure is a hint, never a hard gate.
    """

    def __init__(self, inner: HarnessLoopToolRegistry, config: ToolSearchConfig) -> None:
        self._inner = inner
        self._config = config
        catalogue = inner.schemas()
        self._catalogue = catalogue
        names = {s.name for s in catalogue}

        # Core names that actually exist in the catalogue, catalogue order, deduped.
        core: list[str] = []
        for s in catalogue:
            if s.name in config.core and s.name not in core:
                core.append(s.name)
        self._core = core
        core_set = set(core)

        # Validate namespaces: drop a name colliding with a catalogue tool or an
        # earlier namespace; keep only members that exist and are non-core.
        namespaces: list[ToolNamespace] = []
        ns_names: set[str] = set()
        for ns in config.namespaces:
            if ns.name in names or ns.name in ns_names:
                continue
            tools: list[str] = []
            for t in ns.tools:
                if t in names and t not in core_set and t not in tools:
                    tools.append(t)
            if not tools:
                continue
            namespaces.append(ToolNamespace(name=ns.name, description=ns.description, tools=tools))
            ns_names.add(ns.name)
        self._namespaces = namespaces
        self._namespace_headers = [self._header_schema(ns) for ns in namespaces]
        self._search_schema = self._build_search_schema(config.search_tool_name)
        self._loaded: deque[str] = deque()

    @staticmethod
    def _header_schema(ns: ToolNamespace) -> ModelToolSchema:
        return ModelToolSchema(
            name=ns.name,
            description=f"{ns.description} (call to load this group's tools)",
            input_schema={"type": "object", "properties": {}},
        )

    @staticmethod
    def _build_search_schema(name: str) -> ModelToolSchema:
        return ModelToolSchema(
            name=name,
            description=(
                "Load more tools by keyword when you need a capability you don't "
                "currently have (e.g. run command, web search, task list)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords for the capability you need.",
                    }
                },
                "required": ["query"],
            },
        )

    def _is_core(self, name: str) -> bool:
        return name in self._core

    def _touch(self, name: str) -> None:
        if name == self._config.search_tool_name or self._is_core(name):
            return
        if not any(s.name == name for s in self._catalogue):
            return
        try:
            self._loaded.remove(name)
        except ValueError:
            pass
        self._loaded.appendleft(name)

    def _search(self, query: str) -> list[str]:
        import re

        terms = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) >= 2]
        if not terms:
            return []
        scored: list[tuple[int, str]] = []
        for s in self._catalogue:
            if s.name == self._config.search_tool_name or self._is_core(s.name):
                continue
            name_l = s.name.lower()
            desc_l = s.description.lower()
            score = 0
            for t in terms:
                if t in name_l:
                    score += 3
                if t in desc_l:
                    score += 1
            if score > 0:
                scored.append((score, s.name))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [n for _, n in scored[: self._config.search_top_k]]

    def _load_and_list(self, names: list[str], header: str) -> ToolOutput:
        # Push in reverse so the first-listed ends at the LRU front.
        for name in reversed(names):
            self._touch(name)
        by_name = {s.name: s for s in self._catalogue}
        lines = [header]
        for name in names:
            s = by_name.get(name)
            if s is not None:
                lines.append(f"- {s.name}: {s.description}")
                lines.append(f"  parameters: {s.input_schema}")
        return ToolOutputSuccess(content="\n".join(lines))

    def _run_search(self, input_: dict[str, Any]) -> ToolOutput:
        query = str(input_.get("query", "")).strip()
        if not query:
            return ToolOutputError(
                message=(
                    f"{self._config.search_tool_name}: provide a non-empty `query` "
                    "string describing the capability you need."
                ),
                recoverable=True,
            )
        hits = self._search(query)
        if not hits:
            return ToolOutputSuccess(
                content=(
                    f'No tools matched "{query}". Try different keywords — the tools '
                    "you already have are in your current tool list."
                )
            )
        return self._load_and_list(
            hits,
            f'Loaded {len(hits)} tool(s) for "{query}". You can now call them directly:',
        )

    def _open_namespace(self, name: str) -> ToolOutput:
        ns = next((n for n in self._namespaces if n.name == name), None)
        if ns is None:
            return ToolOutputError(
                message=f"namespace `{name}` is not registered", recoverable=True
            )
        members = ns.tools[: self._config.search_top_k]
        if not members:
            return ToolOutputSuccess(content=f'Namespace "{name}" has no available tools.')
        return self._load_and_list(
            members,
            f'Opened namespace "{name}" — loaded {len(members)} tool(s). '
            "You can now call them directly:",
        )

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        if call.name == self._config.search_tool_name:
            return self._run_search(call.input)
        if any(n.name == call.name for n in self._namespaces):
            return self._open_namespace(call.name)
        self._touch(call.name)
        return await self._inner.dispatch(call)

    def is_always_halt(self, tool_name: str) -> bool:
        if tool_name == self._config.search_tool_name or any(
            n.name == tool_name for n in self._namespaces
        ):
            return False
        return self._inner.is_always_halt(tool_name)

    def schemas(self) -> list[ModelToolSchema]:
        by_name = {s.name: s for s in self._catalogue}
        out: list[ModelToolSchema] = []
        used = 0
        for name in self._core:
            s = by_name.get(name)
            if s is not None:
                used += len(s.description)
                out.append(s.model_copy(deep=True))
        for header in self._namespace_headers:
            used += len(header.description)
            out.append(header.model_copy(deep=True))
        used += len(self._search_schema.description)
        out.append(self._search_schema.model_copy(deep=True))
        kept: deque[str] = deque()
        for name in list(self._loaded):
            if self._is_core(name):
                continue
            s = by_name.get(name)
            if s is None:
                continue
            nxt = used + len(s.description)
            if nxt > self._config.description_budget:
                break
            used = nxt
            kept.append(name)
            out.append(s.model_copy(deep=True))
        self._loaded = kept
        out.sort(key=lambda s: s.name)
        return out

    def advertised_description_bytes(self) -> int:
        return sum(len(s.description) for s in self.schemas())


# ============================================================================
# ReadOnlyToolView — SC-30 read-only view over a harness-loop ToolRegistry
# ============================================================================

# ReadOnlyToolView — SC-30 read-only view over a harness-loop ToolRegistry
# ============================================================================


class ReadOnlyToolView:
    """SC-30: a read-only VIEW over an inner harness-loop
    :class:`~spore_core.harness.ToolRegistry`.

    It advertises (:meth:`schemas`) and dispatches ONLY tools whose name is in
    ``allow`` — the INTERSECTION of the wrapped catalogue with a read-only
    allow-list (:data:`READONLY_EVAL_TOOL_NAMES`, the
    ``StandardTools.readonly_set()`` names). Used internally for the SelfVerifying
    evaluate phase so a reviewer cannot reach write / exec / side-effecting tools
    (web/MCP the read-only sandbox does not gate) even though the work phase could
    — WITHOUT the consumer registering a scoped read-only toolset. A
    non-allow-listed dispatch (which the model should never request, since it is
    never advertised) returns a recoverable :class:`~spore_core.harness.ToolOutputError`.
    A consumer that wants a different reviewer toolset sets an explicit
    ``eval_toolset`` handle, which bypasses this view entirely.

    Satisfies the harness-loop ``ToolRegistry`` Protocol structurally (it does not
    inherit it); mirrors Rust's ``ReadOnlyToolView``.
    """

    def __init__(self, inner: HarnessLoopToolRegistry, allow: set[str]) -> None:
        self._inner = inner
        self._allow = set(allow)

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        if call.name in self._allow:
            return await self._inner.dispatch(call)
        return ToolOutputError(
            message=f"tool `{call.name}` is not available in the read-only evaluate phase",
            recoverable=True,
        )

    def is_always_halt(self, tool_name: str) -> bool:
        return self._inner.is_always_halt(tool_name)

    def schemas(self) -> list[ModelToolSchema]:
        return [s for s in self._inner.schemas() if s.name in self._allow]


# ============================================================================
# Mock tools (test utility)
# ============================================================================


class EchoTool:
    """Echo tool — returns its input as JSON-string content."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.calls: int = 0

    def name(self) -> str:
        return self._name

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        import json as _json

        self.calls += 1
        return ToolOutputSuccess(
            content=_json.dumps(call.input, separators=(",", ":"), sort_keys=False),
            truncated=False,
        )


class FailingTool:
    """Failing tool — returns a recoverable :class:`ToolOutputError`."""

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        from .harness import ToolOutputError

        return ToolOutputError(message="boom", recoverable=True)


class SubagentMock:
    """Subagent-flagged tool — ``is_subagent_tool`` returns ``True``."""

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name

    def is_subagent_tool(self) -> bool:
        return True

    def may_produce_large_output(self) -> bool:
        return False

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        return ToolOutputSuccess(content="subagent done", truncated=False)


def make_test_ctx() -> ToolContext:
    """Build a throwaway :class:`ToolContext` for tests: a fresh in-memory run
    store, a fixed test session id, and a fixed test project id (#142). Mirrors
    the Rust ``mock::test_ctx`` (named ``make_test_ctx`` here so pytest does not
    collect it as a test)."""
    from .storage import InMemoryStorageProvider, project_id_from_canonical_path

    backend = InMemoryStorageProvider()
    return ToolContext(
        session_id=SessionId("test-session"),
        project_id=project_id_from_canonical_path("/test-project"),
        run_store=backend,
        memory_store=backend,
    )


class AllowAllSandbox(BaseSandboxProvider):
    """Permissive sandbox stub — accepts everything."""

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None


class DenyAllSandbox(BaseSandboxProvider):
    """Denying sandbox stub — rejects everything with ``PathEscape``."""

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        from .harness import SandboxPathEscape

        return SandboxPathEscape(path="denied")


# ============================================================================
# Awaitable export so ``Awaitable`` is used (keeps ruff happy across editors).
# ============================================================================

_: Awaitable[None] | None = None


__all__ = [
    "READONLY_EVAL_TOOL_NAMES",
    "AllowAllSandbox",
    "DenyAllSandbox",
    "DispatchError",
    "EchoTool",
    "FailingTool",
    "ReadOnlyToolView",
    "RealToolRegistry",
    "RegistrationError",
    "StandardToolRegistry",
    "SubagentMock",
    "TaskPhase",
    "Tool",
    "ToolAnnotations",
    "ToolContext",
    "ToolNamespace",
    "ToolProgressSink",
    "ToolProgressUpdate",
    "ToolRegistry",
    "ToolSearchConfig",
    "ToolSearchRegistry",
    "ToolResult",
    "ToolSchema",
    "ToolSet",
    "make_test_ctx",
]
