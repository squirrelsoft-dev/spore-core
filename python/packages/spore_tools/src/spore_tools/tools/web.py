"""Web tools (#81, net-new Tier-1 tools): ``web_fetch``, ``web_search``.

Mirrors ``rust/crates/spore-core/src/tools/web.rs``.

Both use :mod:`httpx` for async HTTP. They are ``open_world`` (external
effects) and ``read_only``.

* :class:`WebFetchTool` (``web_fetch``) — GET a URL, return the body text.
* :class:`WebSearchTool` (``web_search``) — POST the query to a configurable
  search endpoint and return the structured JSON results verbatim. There is no
  live web-search backend in spore-core; the endpoint is INJECTED at
  construction so tests drive it against a mock HTTP server (NEVER the live
  network). The default endpoint is ``None``, which yields a recoverable error
  until a real backend is configured.
"""

from __future__ import annotations

import httpx

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

from ._common import finish_with_possible_truncation
from .error import ToolExecutionError
from .params import WebFetchParams, WebSearchParams, parse_params


class WebFetchTool:
    NAME = "web_fetch"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return True

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Fetch the contents of a URL",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            annotations=ToolAnnotations(read_only=True, open_world=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(WebFetchParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(params.url)
                body = resp.text
        except httpx.HTTPError as e:
            return ToolOutputError(message=f"web fetch failed: {e}", recoverable=True)
        except (UnicodeDecodeError, ValueError) as e:
            return ToolOutputError(message=f"web fetch body read failed: {e}", recoverable=True)
        return await finish_with_possible_truncation(body, call.id, sandbox)


class WebSearchTool:
    """Web search tool. The search backend endpoint is INJECTED so tests run
    against a mock HTTP server. With no endpoint configured (the default), every
    call is a recoverable error."""

    NAME = "web_search"

    def __init__(self, endpoint: str | None = None) -> None:
        self._endpoint = endpoint

    @classmethod
    def with_endpoint(cls, endpoint: str) -> WebSearchTool:
        """Construct with a search endpoint (the query is POSTed to it as JSON
        ``{"query": ...}``; the response body is returned verbatim)."""
        return cls(endpoint=endpoint)

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return True

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Search the web and return structured results",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            annotations=ToolAnnotations(read_only=True, open_world=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(WebSearchParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        if self._endpoint is None:
            return ToolOutputError(message="web_search backend not configured", recoverable=True)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(self._endpoint, json={"query": params.query})
                body = resp.text
        except httpx.HTTPError as e:
            return ToolOutputError(message=f"web search failed: {e}", recoverable=True)
        except (UnicodeDecodeError, ValueError) as e:
            return ToolOutputError(message=f"web search body read failed: {e}", recoverable=True)
        return await finish_with_possible_truncation(body, call.id, sandbox)


__all__ = ["WebFetchTool", "WebSearchTool"]
