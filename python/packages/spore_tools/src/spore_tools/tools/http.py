"""HTTP tools: HttpGet, HttpPost. Uses :mod:`httpx` for async HTTP."""

from __future__ import annotations

import httpx

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import ToolAnnotations, ToolSchema

from ._common import finish_with_possible_truncation
from .error import ToolExecutionError
from .params import HttpGetParams, HttpPostParams, parse_params


def _string_headers(h: dict[str, object] | None) -> dict[str, str]:
    if not h:
        return {}
    return {k: v for k, v in h.items() if isinstance(v, str)}


class HttpGetTool:
    NAME = "http_get"

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
            description="Perform an HTTP GET",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "headers": {"type": "object"},
                },
                "required": ["url"],
            },
            annotations=ToolAnnotations(read_only=True, open_world=True),
        )

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        try:
            params = parse_params(HttpGetParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(params.url, headers=_string_headers(params.headers))
                body = resp.text
        except httpx.HTTPError as e:
            return ToolOutputError(message=f"http get failed: {e}", recoverable=True)
        except (UnicodeDecodeError, ValueError) as e:
            return ToolOutputError(message=f"http body read failed: {e}", recoverable=True)
        return await finish_with_possible_truncation(body, call.id, sandbox)


class HttpPostTool:
    NAME = "http_post"

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
            description="Perform an HTTP POST with a JSON body",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "body": {},
                    "headers": {"type": "object"},
                },
                "required": ["url", "body"],
            },
            annotations=ToolAnnotations(open_world=True),
        )

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        try:
            params = parse_params(HttpPostParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        headers = _string_headers(params.headers)
        # httpx sets content-type when ``json=`` is used; preserve caller's
        # explicit content-type if provided.
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(params.url, json=params.body, headers=headers)
                body = resp.text
        except httpx.HTTPError as e:
            return ToolOutputError(message=f"http post failed: {e}", recoverable=True)
        except (UnicodeDecodeError, ValueError) as e:
            return ToolOutputError(message=f"http body read failed: {e}", recoverable=True)
        return await finish_with_possible_truncation(body, call.id, sandbox)


__all__ = ["HttpGetTool", "HttpPostTool"]
