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

``web_search`` configurability (#108)
--------------------------------------

:meth:`WebSearchTool.with_config` accepts a :class:`WebSearchConfig` so the tool
can talk to real search APIs (Brave, Tavily, …) that need GET-style param
encoding and/or auth secrets. The original ``WebSearchTool(endpoint=...)`` /
:meth:`WebSearchTool.with_endpoint` constructors and their behavior are FROZEN
and unchanged: ``with_endpoint`` still POSTs ``{"query": <q>}`` as a JSON body
and returns the response body verbatim.

New public types:

* :class:`SearchMethod` — ``GET`` | ``POST`` (default ``POST``).
* :class:`WebSearchConfig` — endpoint + method + ``query_param`` + two auth
  lists (``auth_headers``, ``body_auth_params``). Stores env-var NAMES, not
  values.
* :class:`WebSearchConfigError` — construction-time error raised by
  ``with_config`` when a referenced env var is unset or empty.

Env sourcing mirrors :meth:`OpenAIModelInterface.from_env`: the caller supplies
the env-var NAME; the value is resolved at CONSTRUCTION time; an unset OR empty
env var raises :class:`WebSearchConfigError` — a request is NEVER sent with a
missing or empty secret. Resolved secrets are held only in a private, non-
serializable structure and never stored on the serializable config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

import httpx

from spore_core.errors import SporeError
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


class SearchMethod(str, Enum):
    """HTTP method used to dispatch a web search. An enum (NOT a bool) so the
    wire shape is explicit and extensible. Default is :attr:`POST`."""

    GET = "get"
    POST = "post"


@dataclass
class WebSearchConfig:
    """Construction-time configuration for :meth:`WebSearchTool.with_config`.

    Env-var NAMES (not values) are stored here; values are resolved when
    ``with_config`` is called.

    Fields:

    * ``endpoint`` — the search backend URL.
    * ``method`` — :class:`SearchMethod`, default :attr:`SearchMethod.POST`.
    * ``query_param`` — the field/param name the query is keyed under (default
      ``"query"``; Brave uses ``"q"``).
    * ``auth_headers`` — ``(header_name, env_var)`` pairs. Each env var is
      resolved at construction time and attached as an HTTP header (by
      ``header_name``) on every request, for both GET and POST.
    * ``body_auth_params`` — ``(field_name, env_var)`` pairs. Each env var is
      resolved at construction time and injected as a secret request parameter.
      For POST it goes into the JSON body alongside the query (e.g. Tavily's
      ``{"api_key": ..., "query": ...}``). For GET it is appended to the URL
      query string alongside the query param.
    """

    endpoint: str
    method: SearchMethod = SearchMethod.POST
    auth_headers: list[tuple[str, str]] = field(default_factory=list)
    query_param: str = "query"
    body_auth_params: list[tuple[str, str]] = field(default_factory=list)


class WebSearchConfigError(SporeError):
    """Raised by :meth:`WebSearchTool.with_config` when a referenced env var is
    unset or empty. Mirrors the ``from_env`` precedent: a request is never sent
    with a missing/empty secret."""


class EnvVarNotSet(WebSearchConfigError):
    """The referenced environment variable is not set."""

    def __init__(self, env_var: str) -> None:
        self.env_var = env_var
        super().__init__(f"env var `{env_var}` not set")


class EnvVarEmpty(WebSearchConfigError):
    """The referenced environment variable is set but empty/whitespace."""

    def __init__(self, env_var: str) -> None:
        self.env_var = env_var
        super().__init__(f"env var `{env_var}` is empty")


def _resolve_env(env_var: str) -> str:
    """Resolve an env var by NAME at construction time. Unset or empty → raise."""
    value = os.environ.get(env_var)
    if value is None:
        raise EnvVarNotSet(env_var)
    if not value.strip():
        raise EnvVarEmpty(env_var)
    return value


@dataclass
class _ResolvedBackend:
    """Resolved (header_name, secret_value) and (field_name, secret_value)
    pairs. Private + non-serializable; ``repr`` redacts the secret values so
    they never leak into logs or traces."""

    endpoint: str
    method: SearchMethod
    query_param: str
    auth_headers: list[tuple[str, str]]
    body_auth_params: list[tuple[str, str]]

    def __repr__(self) -> str:
        redacted_headers = [(name, "<redacted>") for name, _ in self.auth_headers]
        redacted_body = [(name, "<redacted>") for name, _ in self.body_auth_params]
        return (
            f"_ResolvedBackend(endpoint={self.endpoint!r}, method={self.method!r}, "
            f"query_param={self.query_param!r}, auth_headers={redacted_headers!r}, "
            f"body_auth_params={redacted_body!r})"
        )


class WebSearchTool:
    """Web search tool. The search backend endpoint is INJECTED so tests run
    against a mock HTTP server. With no endpoint configured (the default), every
    call is a recoverable error."""

    NAME = "web_search"

    def __init__(self, endpoint: str | None = None) -> None:
        if endpoint is None:
            self._backend: _ResolvedBackend | None = None
        else:
            self._backend = _ResolvedBackend(
                endpoint=endpoint,
                method=SearchMethod.POST,
                query_param="query",
                auth_headers=[],
                body_auth_params=[],
            )

    @classmethod
    def with_endpoint(cls, endpoint: str) -> WebSearchTool:
        """Construct with a search endpoint (the query is POSTed to it as JSON
        ``{"query": ...}``; the response body is returned verbatim).

        FROZEN behavior — kept compatible with the original tool."""
        return cls(endpoint=endpoint)

    @classmethod
    def with_config(cls, config: WebSearchConfig) -> WebSearchTool:
        """Construct from a :class:`WebSearchConfig`, resolving every referenced
        env var at construction time. Raises :class:`WebSearchConfigError` if any
        auth env var is unset or empty — no request is ever attempted in that
        case."""
        auth_headers = [(name, _resolve_env(env_var)) for name, env_var in config.auth_headers]
        body_auth_params = [
            (name, _resolve_env(env_var)) for name, env_var in config.body_auth_params
        ]
        tool = cls.__new__(cls)
        tool._backend = _ResolvedBackend(
            endpoint=config.endpoint,
            method=config.method,
            query_param=config.query_param,
            auth_headers=auth_headers,
            body_auth_params=body_auth_params,
        )
        return tool

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
        backend = self._backend
        if backend is None:
            return ToolOutputError(message="web_search backend not configured", recoverable=True)
        headers = {name: value for name, value in backend.auth_headers}
        try:
            async with httpx.AsyncClient() as client:
                if backend.method is SearchMethod.GET:
                    # Query + body-auth params are URL-encoded into the query
                    # string; httpx encodes spaces, ``&``, etc.
                    query_params: dict[str, str] = {backend.query_param: params.query}
                    for field_name, value in backend.body_auth_params:
                        query_params[field_name] = value
                    resp = await client.get(
                        backend.endpoint, params=query_params, headers=headers or None
                    )
                else:
                    # Query + body-auth params go into the JSON body (Tavily
                    # shape: {"api_key": ..., "query": ...}).
                    json_body: dict[str, str] = {backend.query_param: params.query}
                    for field_name, value in backend.body_auth_params:
                        json_body[field_name] = value
                    resp = await client.post(
                        backend.endpoint, json=json_body, headers=headers or None
                    )
                body = resp.text
        except httpx.HTTPError as e:
            return ToolOutputError(message=f"web search failed: {e}", recoverable=True)
        except (UnicodeDecodeError, ValueError) as e:
            return ToolOutputError(message=f"web search body read failed: {e}", recoverable=True)
        return await finish_with_possible_truncation(body, call.id, sandbox)


__all__ = [
    "EnvVarEmpty",
    "EnvVarNotSet",
    "SearchMethod",
    "WebFetchTool",
    "WebSearchConfig",
    "WebSearchConfigError",
    "WebSearchTool",
]
