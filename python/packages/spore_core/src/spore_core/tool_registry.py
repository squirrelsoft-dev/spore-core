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
3. Duplicate tool names → :class:`RegistrationError` (kind=DuplicateName).
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

from collections.abc import Awaitable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Protocol, runtime_checkable

import anyio

from .errors import SporeError
from .harness import (
    HarnessToolResult as ToolResult,
)
from .harness import (
    BaseSandboxProvider,
    SandboxProvider,
    SandboxViolation,
    ToolOutput,
    ToolOutputSuccess,
)
from .model import ToolCall
from .model import ToolSchema as ModelToolSchema

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

    Tools are stateless and receive a :class:`SandboxProvider` on every
    dispatch. The protocol is structural — concrete impls do not inherit.
    """

    def name(self) -> str: ...

    def is_subagent_tool(self) -> bool: ...

    def may_produce_large_output(self) -> bool: ...

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput: ...


# ============================================================================
# ToolRegistry protocol
# ============================================================================


@runtime_checkable
class ToolRegistry(Protocol):
    """Canonical registry protocol."""

    def register(self, tool: Tool, schema: ToolSchema) -> None: ...

    def register_set(self, set_: ToolSet) -> None: ...

    def active_schemas(self, phase: TaskPhase | None) -> list[ToolSchema]: ...

    async def dispatch(self, call: ToolCall, sandbox: SandboxProvider) -> ToolResult: ...

    async def dispatch_all(
        self, calls: list[ToolCall], sandbox: SandboxProvider
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
        if schema.name in self._tools:
            raise RegistrationError.duplicate_name(schema.name)
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

    async def dispatch(self, call: ToolCall, sandbox: SandboxProvider) -> ToolResult:
        entry = self._tools.get(call.name)
        if entry is None:
            raise DispatchError.unregistered_tool(call.name)

        # Sandbox first (Layer-1 violations propagate to the harness).
        violation = await sandbox.validate(call)
        if violation is not None:
            raise DispatchError.sandbox_violation(violation)

        self._validate_input(entry.schema, call)

        output = await entry.tool.execute(call, sandbox)
        return ToolResult(call_id=call.id, output=output)

    async def dispatch_all(
        self, calls: list[ToolCall], sandbox: SandboxProvider
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
                results[idx] = await self.dispatch(calls[idx], sandbox)
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

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
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

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
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

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        return ToolOutputSuccess(content="subagent done", truncated=False)


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
    "AllowAllSandbox",
    "DenyAllSandbox",
    "DispatchError",
    "EchoTool",
    "FailingTool",
    "RegistrationError",
    "StandardToolRegistry",
    "SubagentMock",
    "TaskPhase",
    "Tool",
    "ToolAnnotations",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    "ToolSet",
]
