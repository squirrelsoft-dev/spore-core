"""Memory tool (#82, storage seam #78): the scope-aware read/write tool over the
persisted episodic :class:`~spore_core.storage.MemoryStore`.

Mirrors ``rust/crates/spore-core/src/tools/memory.rs`` (behaviour, not structure).

One tool, :class:`MemoryTool` (``NAME = "memory"``), dispatched on an
``operation`` discriminator (``write``, ``read``). It is the agent-facing surface
over the scope-aware memory seam shipped in #78.

Types
-----
* :class:`~spore_core.storage.MemoryEntry` — ``{ role, content, timestamp,
  metadata }``. UNCHANGED by #82 — ``metadata`` already exists; the tool exposes
  it as an optional ``write`` param (decision C).
* :class:`~spore_core.prompt_assembly.StorageScope` — ``{ user, project, local
  }``. ``local`` is rejected at runtime on BOTH ops (see Rules).

Operations
----------
* ``write`` — ``{operation, scope, role, content, metadata?}``. Appends one
  :class:`~spore_core.storage.MemoryEntry` to ``scope``; on success returns the
  serialized just-written entry (decision A).
* ``read`` — ``{operation, scope, merged?=False, limit?=50}``. A scoped read
  returns the most-recent ``limit`` entries of ``scope`` newest-first; a merged
  read returns the User ∪ Project merge via the single
  :meth:`~spore_core.storage.MemoryStore.get_memories_merged` (decision D2).

Rules enforced
--------------
* **R1 write→read roundtrip / R2 write success content (decision A).** A
  ``write`` appends one entry; ``write`` returns the serialized just-written
  entry as success content; a subsequent same-scope ``read`` returns it.
* **R3 read default limit (decision B).** ``limit`` defaults to ``50``,
  overridable; ``read`` returns the most-recent ``limit`` entries newest-first.
* **R4 metadata on write (decision C).** ``metadata`` is optional, defaults to
  ``{}``, and is stored verbatim. :class:`MemoryEntry` is NOT changed.
* **R5 scope isolation.** A non-merged ``read`` of one scope never sees the
  other scope's entries.
* **R6 merged read (decision D2).** ``read`` with ``merged: true`` returns the
  User ∪ Project merge via the single merge method.
* **R7 Local rejected on BOTH ops.** ``local`` scope → recoverable
  :class:`~spore_core.harness.ToolOutputError` with the EXACT message
  ``"Local scope is not supported by MemoryTool — use User or Project."``,
  checked BEFORE any storage access (nothing is written).
* **R8 bad params recoverable.** Bad input maps to a recoverable error.
* **R9 storage error recoverable.** A :class:`~spore_core.storage.StorageError`
  from append/get maps to a recoverable error.
* **R10 read does not write.** A ``read`` performs no append.

Annotations (decision E)
------------------------
NOT annotated ``read_only``. A ``read_only`` tool would be run CONCURRENTLY by
``dispatch_all`` and could race the shared append; like :class:`TaskListTool`
this tool uses default annotations (all false) so the registry dispatches it
sequentially.

Known v1 limitation (#78 Q7)
----------------------------
Memory is :class:`~spore_core.harness.SessionId`-keyed for v1: the tool always
uses ``ctx.session_id`` and offers NO cross-session addressing param. v2 should
add session-independent memory keying — do not introduce it here.
"""

from __future__ import annotations

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.memory import now
from spore_core.model import ToolCall
from spore_core.prompt_assembly import StorageScope
from spore_core.storage import MemoryEntry, StorageError
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

from .error import ToolExecutionError
from .params import MemoryReadParams, MemoryWriteParams, parse_memory_params

#: Exact error message returned when a ``local``-scoped op is attempted.
LOCAL_REJECTED_MESSAGE = "Local scope is not supported by MemoryTool — use User or Project."


def _resolve_scope(raw: str) -> StorageScope | None:
    """Map the free-string ``scope`` param to a :class:`StorageScope`, or
    ``None`` if it is not a recognized scope value. ``local`` resolves (so the
    runtime rejection in :meth:`MemoryTool.execute` fires); a wholly unknown
    string resolves to ``None`` → a recoverable bad-params error."""
    try:
        return StorageScope(raw)
    except ValueError:
        return None


class MemoryTool:
    """Read or write scope-aware episodic memory for this session."""

    NAME = "memory"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        # Properties kept sorted/stable for cache stability. ``scope`` advertises
        # only user/project — ``local`` is rejected at runtime but intentionally
        # omitted from the advertised enum.
        return ToolSchema(
            name=cls.NAME,
            description="Read or write scope-aware episodic memory for this session",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "limit": {"type": "integer"},
                    "merged": {"type": "boolean"},
                    "metadata": {"type": "object"},
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write"],
                    },
                    "role": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "enum": ["project", "user"],
                    },
                },
                "required": ["operation", "scope"],
            },
            # Intentionally NOT read_only: the shared append must dispatch
            # sequentially. See module docs (decision E).
            annotations=ToolAnnotations(),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        session_id = ctx.session_id
        memory_store = ctx.memory_store

        # 1. Parse params (bad input → recoverable). R8.
        try:
            operation, params = parse_memory_params(call)
        except ToolExecutionError as e:
            return e.to_tool_output()

        # 2. Resolve scope. An unknown scope string is a recoverable bad-params
        #    error; ``local`` resolves so it can be rejected at runtime (R7).
        scope = _resolve_scope(params.scope)
        if scope is None:
            return ToolOutputError(
                message=f"invalid parameters: unknown scope `{params.scope}`",
                recoverable=True,
            )

        # R7: reject Local on BOTH ops BEFORE touching storage; nothing written.
        if scope == StorageScope.LOCAL:
            return ToolOutputError(message=LOCAL_REJECTED_MESSAGE, recoverable=True)

        if isinstance(params, MemoryWriteParams):
            return await self._write(memory_store, session_id, scope, params)
        assert isinstance(params, MemoryReadParams)  # noqa: S101 — invariant
        assert operation == "read"  # noqa: S101 — invariant
        return await self._read(memory_store, session_id, scope, params)

    @staticmethod
    async def _write(
        memory_store: object,
        session_id: object,
        scope: StorageScope,
        params: MemoryWriteParams,
    ) -> ToolOutput:
        # R4: metadata stored verbatim, default {}. The tool stamps "now".
        entry = MemoryEntry(
            role=params.role,
            content=params.content,
            timestamp=now(),
            metadata=params.metadata,
        )
        try:
            await memory_store.append_memory(scope, session_id, entry)  # type: ignore[attr-defined]
        except StorageError as e:
            return ToolOutputError(message=f"could not append memory: {e}", recoverable=True)
        # R2 (decision A): success content = the serialized just-written entry.
        return ToolOutputSuccess(content=entry.model_dump_json(), truncated=False)

    @staticmethod
    async def _read(
        memory_store: object,
        session_id: object,
        scope: StorageScope,
        params: MemoryReadParams,
    ) -> ToolOutput:
        # R6 (decision D2): merged read drives the single merge method. Otherwise
        # a scoped read (R5 isolation). R10: neither path writes.
        try:
            if params.merged:
                entries = await memory_store.get_memories_merged(  # type: ignore[attr-defined]
                    session_id, params.limit
                )
            else:
                entries = await memory_store.get_memories(  # type: ignore[attr-defined]
                    scope, session_id, params.limit
                )
        except StorageError as e:
            return ToolOutputError(message=f"could not read memory: {e}", recoverable=True)
        payload = "[" + ",".join(e.model_dump_json() for e in entries) + "]"
        return ToolOutputSuccess(content=payload, truncated=False)


__all__ = ["LOCAL_REJECTED_MESSAGE", "MemoryTool"]
