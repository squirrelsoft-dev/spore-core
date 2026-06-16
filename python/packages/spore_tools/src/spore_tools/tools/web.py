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

import ipaddress
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


# ── SSRF guard — outbound URL policy (validate_fetch_url seam) ────────────────


@dataclass(frozen=True)
class UrlPolicy:
    """Policy consulted before any outbound ``web_fetch`` / ``web_search`` request.

    **Permissive by default** — :meth:`UrlPolicy.permissive` (the default) allows
    every URL with no parsing at all, so existing wiring, tests, and examples are
    unaffected. A deployment that wants SSRF protection opts in with
    :meth:`UrlPolicy.deny_private` (wired via the web-tool builders, see
    :meth:`WebFetchTool.with_url_policy` / :meth:`WebSearchTool.with_url_policy`).

    ``deny_private`` rejects non-``http(s)`` schemes and hosts given as an IP
    literal in a loopback / link-local / private (RFC-1918) / cloud-metadata
    (``169.254.169.254``) range, plus the ``localhost`` hostname (and
    ``*.localhost``). **Limitation:** non-``localhost`` hostnames are NOT
    DNS-resolved here, so a hostname that resolves to a private address is not
    caught by this seam — deployments that need resolution-time protection must
    enforce it at the network layer.
    """

    # The boolean is stored as a private field so the ``deny_private`` name is
    # free for the constructor classmethod below (a public field named
    # ``deny_private`` would collide with / be shadowed by the classmethod).
    _deny_private: bool = False

    @classmethod
    def permissive(cls) -> UrlPolicy:
        """Allow every URL (the default). No SSRF restrictions."""
        return cls(_deny_private=False)

    @classmethod
    def deny_private(cls) -> UrlPolicy:
        """Reject non-``http(s)`` schemes and private/loopback/link-local/metadata
        IP-literal hosts (and the ``localhost`` hostname)."""
        return cls(_deny_private=True)


def _is_blocked_v4(v4: ipaddress.IPv4Address) -> bool:
    """Match the Rust SSRF range set exactly.

    Python's ``IPv4Address.is_private`` is both broader and narrower than the
    Rust predicate (e.g. it does NOT flag ``255.255.255.255``), so assemble the
    blocked set explicitly rather than relying on a single property.
    """
    return (
        v4.is_loopback  # 127.0.0.0/8
        or v4 in ipaddress.ip_network("10.0.0.0/8")  # RFC-1918
        or v4 in ipaddress.ip_network("172.16.0.0/12")  # RFC-1918
        or v4 in ipaddress.ip_network("192.168.0.0/16")  # RFC-1918
        or v4.is_link_local  # 169.254.0.0/16 — includes 169.254.169.254
        or v4.is_unspecified  # 0.0.0.0
        or v4 == ipaddress.IPv4Address("255.255.255.255")  # broadcast
        or v4 in ipaddress.ip_network("192.0.2.0/24")  # documentation
        or v4 in ipaddress.ip_network("198.51.100.0/24")  # documentation
        or v4 in ipaddress.ip_network("203.0.113.0/24")  # documentation
    )


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv4Address):
        return _is_blocked_v4(ip)
    # IPv4-mapped (``::ffff:a.b.c.d``) → unwrap and apply the v4 rules so e.g.
    # ``::ffff:169.254.169.254`` is blocked.
    mapped = ip.ipv4_mapped
    if mapped is not None:
        return _is_blocked_v4(mapped)
    return (
        ip.is_loopback  # ::1
        or ip.is_unspecified  # ::
        or ip in ipaddress.ip_network("fe80::/10")  # link-local
        or ip in ipaddress.ip_network("fc00::/7")  # unique-local
    )


def validate_fetch_url(url: str, policy: UrlPolicy) -> None | ToolOutputError:
    """Validate ``url`` against ``policy`` before an outbound request.

    Returns ``None`` when the URL is allowed, or a recoverable
    :class:`ToolOutputError` when ``deny_private`` rejects it. The permissive
    policy performs no parsing and always allows.
    """
    if not policy._deny_private:
        return None
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, ValueError) as e:
        return ToolOutputError(message=f"invalid URL: {e}", recoverable=True)
    if parsed.scheme not in ("http", "https"):
        return ToolOutputError(
            message=f"scheme '{parsed.scheme}' not allowed by URL policy",
            recoverable=True,
        )
    host = parsed.host
    if not host:
        return ToolOutputError(message="URL has no host", recoverable=True)
    lowered = host.lower()
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return ToolOutputError(message=f"host '{host}' is loopback", recoverable=True)
    # ``httpx.URL.host`` is already unbracketed for IPv6 literals. A non-IP host
    # (DNS name) raises ``ValueError`` here → allowed without resolution (the
    # documented limitation: a name resolving to a private address is not caught).
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if _is_blocked_ip(ip):
        return ToolOutputError(
            message=f"host '{host}' is in a blocked address range",
            recoverable=True,
        )
    return None


def _apply_web_fetch_range(body: str, start_byte: int) -> str | ToolOutputError:
    """Apply ``start_byte`` slicing to a fetched response body.

    - ``start_byte == 0`` → return ``body`` unchanged (no header).
    - ``0 < start_byte < len(body_bytes)`` → prepend
      ``[starting at byte N of total]\\n`` and return the slice from
      ``start_byte``.
    - ``start_byte >= len(body_bytes)`` (non-empty body) → recoverable error.
    - Empty body + ``start_byte > 0`` → recoverable error.

    Byte arithmetic uses UTF-8 encoding, identical to the Rust reference.
    """
    if start_byte == 0:
        return body
    body_bytes = body.encode("utf-8")
    total = len(body_bytes)
    if start_byte >= total:
        return ToolOutputError(
            message=f"start_byte {start_byte} exceeds response length {total}",
            recoverable=True,
        )
    sliced = body_bytes[start_byte:].decode("utf-8", errors="replace")
    return f"[starting at byte {start_byte} of {total}]\n{sliced}"


class WebFetchTool:
    NAME = "web_fetch"

    def __init__(self, policy: UrlPolicy | None = None) -> None:
        self._policy = policy or UrlPolicy.permissive()

    def with_url_policy(self, policy: UrlPolicy) -> WebFetchTool:
        """Opt into an SSRF policy (e.g. :meth:`UrlPolicy.deny_private`). Returns
        ``self`` so it chains off the constructor."""
        self._policy = policy
        return self

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
                "properties": {
                    "url": {"type": "string"},
                    "start_byte": {
                        "type": "integer",
                        "description": (
                            "Byte offset into the response body to start reading from. "
                            "Default 0 (no offset, output identical to a plain fetch). "
                            "Use to page through responses larger than the 64 KB "
                            "truncation window."
                        ),
                        "default": 0,
                    },
                },
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
        denied = validate_fetch_url(params.url, self._policy)
        if isinstance(denied, ToolOutputError):
            return denied
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(params.url)
                body = resp.text
        except httpx.HTTPError as e:
            return ToolOutputError(message=f"web fetch failed: {e}", recoverable=True)
        except (UnicodeDecodeError, ValueError) as e:
            return ToolOutputError(message=f"web fetch body read failed: {e}", recoverable=True)
        result = _apply_web_fetch_range(body, params.start_byte)
        if isinstance(result, ToolOutputError):
            return result
        return await finish_with_possible_truncation(result, call.id, sandbox)


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
        self._policy = UrlPolicy.permissive()
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

    def with_url_policy(self, policy: UrlPolicy) -> WebSearchTool:
        """Opt into an SSRF policy (e.g. :meth:`UrlPolicy.deny_private`) applied
        to the configured backend endpoint. Returns ``self`` so it chains off the
        constructor."""
        self._policy = policy
        return self

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
        # ``__new__`` bypasses ``__init__``, so set ``_policy`` explicitly or
        # ``execute`` would ``AttributeError``.
        tool._policy = UrlPolicy.permissive()
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
        # Validate the CONFIGURED endpoint (raw string, pre-merge) — not the URL
        # with the query/auth params appended.
        denied = validate_fetch_url(backend.endpoint, self._policy)
        if isinstance(denied, ToolOutputError):
            return denied
        headers = {name: value for name, value in backend.auth_headers}
        try:
            async with httpx.AsyncClient() as client:
                if backend.method is SearchMethod.GET:
                    # Query + body-auth params are URL-encoded into the query
                    # string; httpx encodes spaces, ``&``, etc. Any query string
                    # already present on the endpoint URL (e.g. SearXNG's
                    # ``?format=json``) is PRESERVED — httpx's ``params=`` would
                    # otherwise REPLACE the existing query string, so we merge
                    # onto the endpoint URL explicitly.
                    query_params: dict[str, str] = {backend.query_param: params.query}
                    for field_name, value in backend.body_auth_params:
                        query_params[field_name] = value
                    url = httpx.URL(backend.endpoint).copy_merge_params(query_params)
                    resp = await client.get(url, headers=headers or None)
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
    "UrlPolicy",
    "WebFetchTool",
    "WebSearchConfig",
    "WebSearchConfigError",
    "WebSearchTool",
    "_apply_web_fetch_range",
    "validate_fetch_url",
]
