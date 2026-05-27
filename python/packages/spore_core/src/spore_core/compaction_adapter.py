"""Compaction adapter — bridges the rich :class:`StandardContextManager`
onto the harness-loop compaction seam.

Implements issue #55. Issue #46 wired the verify→retry→warn *machinery* into
the harness loop and proved it with test-double context managers. The rich
:class:`spore_core.context.StandardContextManager` from #29 implements
compaction against the *rich* :class:`spore_core.context.SessionState` /
:class:`spore_core.context.CompactionResult` API and was never reachable from
the loop seam. This module is the production bridge.

Adapter type
------------
:class:`StandardCompactionAdapter` wraps a
:class:`spore_core.context.StandardContextManager` and satisfies the
harness-side :class:`spore_core.harness.ContextManager` Protocol structurally,
so a :class:`spore_core.harness.HarnessBuilder` can be built with a rich
manager that *actually compacts*.

Seam methods satisfied
-----------------------
* ``assemble`` — minimal pass-through (NOT load-bearing for compaction; builds
  an :class:`spore_core.agent.Context` from the session messages, mirroring the
  loop's test doubles).
* ``append_tool_result`` / ``append_user_message`` — minimal: append to
  ``harness.SessionState.messages``.
* ``should_compact`` — reconstruct rich state from ``session.extras``, delegate
  to rich ``should_compact``.
* ``prepare_compaction_turn`` — reconstruct rich state → rich
  ``prepare_compaction``; ``None`` when there is nothing to compact, else
  project hints + verification state + count.
* ``inject_missing_items`` — *not* defined here: the harness loop falls back to
  its module-level default (the fixture asserts that exact prompt).
* ``apply_compaction`` — reconstruct rich state, delegate to rich
  ``apply_compaction``, log+swallow any error (the loop must never halt on
  compaction), write the mutated rich state back into the session.

Rules enforced
--------------
1. STATELESS bridge — the adapter holds no session state. Rich
   :class:`spore_core.context.SessionState` is serialized into
   ``harness.SessionState.extras`` under :data:`RICH_STATE_KEY` on every
   mutating seam call and re-read on every read. No instance attribute carries
   session state.
2. Compaction never halts the loop — ``apply_compaction`` swallows the rich
   error (logged), and a malformed/absent rich-state blob degrades to a safe
   default (no compaction) rather than raising.
3. The summary is wrapped as a :class:`~spore_core.model.Message` with role
   ``ASSISTANT`` for the rich :class:`CompactionResult` so the rich manager
   prepends it as the summary turn.
"""

from __future__ import annotations

import logging
from typing import Any

from .agent import Context as AgentContext
from .context import (
    CompactionResult,
    Guide,
    GuideId,
    SessionState as RichSessionState,
    StandardContextManager,
)
from .harness import (
    CompactionTurn,
    ContextManager as HarnessContextManager,
    HarnessToolResult,
    SessionState as HarnessState,
    Task,
    ToolOutputError,
    ToolOutputSuccess,
)
from .model import (
    ImageContent,
    Message,
    ModelParams,
    Role,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from .tool_registry import TaskPhase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token estimation (Known Deviation #2 fix)
# ---------------------------------------------------------------------------


def estimate_message_tokens(message: Message) -> int:
    """Rough token estimate for a single message: the byte length of its
    textual content divided by four (the same chars/4 proxy the rich
    :class:`StandardContextManager` uses for cache-marker placement). Used by
    the adapter to compute real ``tokens_reclaimed`` from the messages a
    compaction drops, since the synchronous harness seam cannot call the async
    ``count_tokens``.

    Returns at least ``1`` token for any non-empty message so a dropped message
    is never accounted as zero reclamation.
    """
    content = message.content
    if isinstance(content, TextContent):
        nbytes = len(content.text.encode("utf-8"))
    elif isinstance(content, ToolCallContent):
        import json as _json

        nbytes = len(content.name.encode("utf-8")) + len(
            _json.dumps(content.input, separators=(",", ":")).encode("utf-8")
        )
    elif isinstance(content, ToolResultContent):
        nbytes = len(content.content.encode("utf-8"))
    elif isinstance(content, ImageContent):
        nbytes = len(content.data.encode("utf-8"))
    else:  # pragma: no cover — exhaustive over the Content union
        nbytes = 0
    estimate = nbytes // 4
    if estimate == 0 and nbytes > 0:
        return 1
    return estimate


def estimate_tokens(messages: list[Message]) -> int:
    """Sum :func:`estimate_message_tokens` over a list of messages."""
    return sum(estimate_message_tokens(m) for m in messages)


#: Reserved key under ``harness.SessionState.extras`` holding the serialized
#: rich :class:`spore_core.context.SessionState`. The adapter is the only
#: writer/reader.
RICH_STATE_KEY = "spore.compaction_adapter.rich_state"


# ---------------------------------------------------------------------------
# Rich-state serialization (stateless bridge round-trip)
# ---------------------------------------------------------------------------


def _rich_to_dict(rich: RichSessionState) -> dict[str, Any]:
    """Serialize a rich :class:`SessionState` into a JSON-safe dict."""
    return {
        "session_id": str(rich.session_id),
        "task_id": str(rich.task_id),
        "task_instruction": rich.task_instruction,
        "turn_number": rich.turn_number,
        "environment": rich.environment,
        "prior_state": rich.prior_state,
        "operational_instructions": rich.operational_instructions,
        "active_phase": rich.active_phase.value,
        "message_history": [m.model_dump(mode="json") for m in rich.message_history],
        "token_budget_used": rich.token_budget_used,
        "window_limit": rich.window_limit,
        "guides_loaded": [str(g) for g in rich.guides_loaded],
        "pending_skill_injections": [
            {"id": str(g.id), "content": g.content} for g in rich.pending_skill_injections
        ],
        "budget_warning_active": rich.budget_warning_active,
    }


def _rich_from_dict(data: dict[str, Any]) -> RichSessionState:
    """Reconstruct a rich :class:`SessionState` from a serialized dict.

    Raises ``KeyError`` / ``ValueError`` / pydantic ``ValidationError`` on a
    malformed blob; callers treat any such failure as "nothing to compact".
    """
    from .harness import SessionId, TaskId

    state = RichSessionState(
        session_id=SessionId(data["session_id"]),
        task_id=TaskId(data["task_id"]),
        task_instruction=data["task_instruction"],
        turn_number=data.get("turn_number", 0),
        environment=data.get("environment", ""),
        prior_state=data.get("prior_state", ""),
        operational_instructions=data.get("operational_instructions", ""),
        active_phase=TaskPhase(data.get("active_phase", TaskPhase.EXECUTION.value)),
        message_history=[Message.model_validate(m) for m in data.get("message_history", [])],
        token_budget_used=data.get("token_budget_used", 0),
        window_limit=data.get("window_limit", 200_000),
        guides_loaded=[GuideId(g) for g in data.get("guides_loaded", [])],
        pending_skill_injections=[
            Guide(id=GuideId(g["id"]), content=g["content"])
            for g in data.get("pending_skill_injections", [])
        ],
        budget_warning_active=data.get("budget_warning_active", False),
    )
    return state


def seed_rich_state(session: HarnessState, rich: RichSessionState) -> None:
    """Project a rich session state into the harness session before the first
    turn. Callers that drive the harness with :class:`StandardCompactionAdapter`
    use this to seed ``extras`` (and ``messages``) so ``should_compact`` and
    ``prepare_compaction_turn`` have rich state to read."""
    session.messages = list(rich.message_history)
    session.extras[RICH_STATE_KEY] = _rich_to_dict(rich)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class StandardCompactionAdapter:
    """Stateless bridge from the rich :class:`StandardContextManager` onto the
    harness-loop compaction seam (:class:`spore_core.harness.ContextManager`).

    Construct via :class:`StandardCompactionAdapter` directly or
    :func:`into_harness_adapter`, then inject the result into a
    :class:`spore_core.harness.HarnessBuilder` / :class:`HarnessConfig`.
    """

    def __init__(self, inner: StandardContextManager) -> None:
        self._inner = inner

    # ---- rich-state helpers (stateless) -----------------------------

    @staticmethod
    def _read_rich_state(session: HarnessState) -> RichSessionState | None:
        """Reconstruct the rich session state from ``extras``. Returns ``None``
        when no rich state has been seeded yet or the blob is malformed —
        callers treat that as "nothing to compact" so the loop is never
        blocked."""
        value = session.extras.get(RICH_STATE_KEY)
        if not isinstance(value, dict):
            return None
        try:
            return _rich_from_dict(value)
        except Exception:  # noqa: BLE001 — any decode failure degrades safely
            logger.warning("spore.compaction: malformed rich state blob; skipping compaction")
            return None

    @staticmethod
    def _write_rich_state(session: HarnessState, rich: RichSessionState) -> None:
        """Serialize the rich session state back into ``extras`` and project its
        ``message_history`` onto the harness-side ``messages``."""
        session.messages = list(rich.message_history)
        session.extras[RICH_STATE_KEY] = _rich_to_dict(rich)

    # ---- minimal (non-load-bearing) seam methods --------------------

    async def assemble(self, session: HarnessState, task: Task) -> AgentContext:
        # NOT load-bearing for compaction. The rich ``assemble`` requires
        # ``ContextSources`` the seam does not supply, so produce a minimal
        # context straight from the session messages (mirrors the loop's
        # test-double managers).
        _ = task
        return AgentContext(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: HarnessState, result: HarnessToolResult) -> None:
        output = result.output
        if isinstance(output, ToolOutputSuccess):
            text = output.content
        elif isinstance(output, ToolOutputError):
            text = output.message
        else:
            text = ""
        session.messages.append(Message(role=Role.TOOL, content=TextContent(text=text)))

    async def append_user_message(self, session: HarnessState, text: str) -> None:
        session.messages.append(Message(role=Role.USER, content=TextContent(text=text)))

    # ---- compaction seam --------------------------------------------

    def should_compact(self, session: HarnessState) -> bool:
        rich = self._read_rich_state(session)
        if rich is None:
            return False
        return self._inner.should_compact(rich)

    def prepare_compaction_turn(self, session: HarnessState) -> CompactionTurn | None:
        rich = self._read_rich_state(session)
        if rich is None:
            return None
        request = self._inner.prepare_compaction(rich)
        if not request.messages_to_compact:
            return None
        messages_removed = len(request.messages_to_compact)

        # Build the summarization context: the messages to compact, followed by
        # the summarization instruction. The harness loop's default
        # ``inject_missing_items`` appends the retry instruction on a
        # verification failure.
        messages = list(request.messages_to_compact)
        messages.append(
            Message(
                role=Role.USER,
                content=TextContent(
                    text=(
                        "Summarize the conversation above, preserving the items "
                        "in the preservation hints."
                    )
                ),
            )
        )

        return CompactionTurn(
            context=AgentContext(messages=messages, tools=[], params=ModelParams()),
            preserve_hints=request.preserve_hints,
            verification_state=rich,
            messages_removed=messages_removed,
        )

    # ``inject_missing_items`` is intentionally NOT defined: the harness loop's
    # module-level default produces the exact "Your summary is missing these
    # items: …" prompt the ``compaction_loop`` fixture asserts.

    def apply_compaction(self, session: HarnessState, summary: str) -> None:
        rich = self._read_rich_state(session)
        if rich is None:
            # No rich state to apply against — degrade safely; never raise.
            return
        request = self._inner.prepare_compaction(rich)
        dropped = request.messages_to_compact
        messages_removed = len(dropped)

        summary_message = Message(role=Role.ASSISTANT, content=TextContent(text=summary))

        # Real token accounting (Known Deviation #2 fix): reclaim the tokens of
        # the messages we drop, net of the summary that replaces them, and clamp
        # to the live budget so ``token_budget_used`` never underflows. The rich
        # ``apply_compaction`` (context.py) decrements ``token_budget_used`` by
        # this amount, so utilization actually falls below threshold after a
        # compaction and a long session can compact repeatedly.
        dropped_tokens = estimate_tokens(dropped)
        summary_tokens = estimate_message_tokens(summary_message)
        net_reclaimed = max(0, dropped_tokens - summary_tokens)
        tokens_reclaimed = min(net_reclaimed, rich.token_budget_used)

        result = CompactionResult(
            summary_message=summary_message,
            tokens_reclaimed=tokens_reclaimed,
            messages_removed=messages_removed,
        )
        try:
            self._inner.apply_compaction(rich, result)
        except Exception as err:  # noqa: BLE001 — compaction must never halt loop
            logger.warning(
                "spore.compaction: rich apply_compaction failed, leaving session unchanged: %s",
                err,
            )
            return
        self._write_rich_state(session, rich)

    def token_budget_used(self, session: HarnessState) -> int | None:
        """Post-compaction budget seam (Known Deviation #2 fix). The harness
        reads this after applying a compaction to stamp the real
        ``tokens_after`` / ``tokens_reclaimed`` on the emitted ``Compaction``
        span. Returns ``None`` when no rich state has been seeded."""
        rich = self._read_rich_state(session)
        if rich is None:
            return None
        return rich.token_budget_used


def into_harness_adapter(inner: StandardContextManager) -> HarnessContextManager:
    """Ergonomic constructor: wrap a rich :class:`StandardContextManager` as the
    harness-seam adapter for injection into a
    :class:`spore_core.harness.HarnessBuilder` / :class:`HarnessConfig`."""
    return StandardCompactionAdapter(inner)


__all__ = [
    "RICH_STATE_KEY",
    "StandardCompactionAdapter",
    "estimate_message_tokens",
    "estimate_tokens",
    "into_harness_adapter",
    "seed_rich_state",
]
