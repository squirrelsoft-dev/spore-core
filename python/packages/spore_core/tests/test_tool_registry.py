"""Tests for :mod:`spore_core.tool_registry` (issue #4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from spore_core.harness import (
    SandboxPathEscape,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import (
    AllowAllSandbox,
    DenyAllSandbox,
    DispatchError,
    EchoTool,
    FailingTool,
    RegistrationError,
    StandardToolRegistry,
    SubagentMock,
    TaskPhase,
    ToolAnnotations,
    ToolSchema,
    ToolSet,
    make_test_ctx,
)

_CTX = make_test_ctx()


def _schema(name: str, annotations: ToolAnnotations | None = None) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {}},
        annotations=annotations or ToolAnnotations(),
    )


def _schema_with_required(name: str, required: list[str]) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=name,
        parameters={
            "type": "object",
            "properties": {},
            "required": required,
        },
        annotations=ToolAnnotations(read_only=True),
    )


def _call(name: str, id_: str, input_: dict[str, Any]) -> ToolCall:
    return ToolCall(id=id_, name=name, input=input_)


# ----- Rule 1: tools dispatched via registry --------------------------------


async def test_registered_tool_dispatches_through_registry() -> None:
    reg = StandardToolRegistry()
    reg.register(EchoTool("echo"), _schema("echo", ToolAnnotations(read_only=True)))
    result = await reg.dispatch(_call("echo", "c1", {"x": 1}), AllowAllSandbox(), _CTX)
    assert result.call_id == "c1"
    assert isinstance(result.output, ToolOutputSuccess)
    assert result.output.content == '{"x":1}'
    assert result.output.truncated is False


# ----- Rule 3: duplicate registration ---------------------------------------


async def test_duplicate_registration_errors() -> None:
    reg = StandardToolRegistry()
    reg.register(EchoTool("echo"), _schema("echo"))
    with pytest.raises(RegistrationError) as excinfo:
        reg.register(EchoTool("echo"), _schema("echo"))
    assert excinfo.value.kind == "DuplicateName"


# ----- Rule 2: schema validated at registration ------------------------------


async def test_invalid_schema_rejected_at_registration() -> None:
    reg = StandardToolRegistry()
    bad = ToolSchema(
        name="x",
        description="x",
        parameters={"properties": {}},  # missing top-level `type`
        annotations=ToolAnnotations(),
    )
    with pytest.raises(RegistrationError) as excinfo:
        reg.register(EchoTool("x"), bad)
    assert excinfo.value.kind == "InvalidSchema"


async def test_empty_schema_name_rejected() -> None:
    reg = StandardToolRegistry()
    with pytest.raises(RegistrationError):
        reg.register(EchoTool(""), _schema(""))


# ----- Rule 4: conflicting annotations rejected -----------------------------


async def test_read_only_plus_destructive_rejected() -> None:
    reg = StandardToolRegistry()
    with pytest.raises(RegistrationError) as excinfo:
        reg.register(
            EchoTool("rm"),
            _schema("rm", ToolAnnotations(read_only=True, destructive=True)),
        )
    assert excinfo.value.kind == "ConflictingAnnotations"


# ----- Rule: tool/schema name mismatch ---------------------------------------


async def test_tool_and_schema_name_must_match() -> None:
    reg = StandardToolRegistry()
    with pytest.raises(RegistrationError) as excinfo:
        reg.register(EchoTool("a"), _schema("b"))
    assert excinfo.value.kind == "InvalidSchema"


# ----- Rule 6: unregistered tool dispatch -----------------------------------


async def test_unregistered_tool_call_errors() -> None:
    reg = StandardToolRegistry()
    with pytest.raises(DispatchError) as excinfo:
        await reg.dispatch(_call("missing", "c1", {}), AllowAllSandbox(), _CTX)
    assert excinfo.value.kind == "UnregisteredTool"


# ----- Rule 7: schema validation failure on missing required field ----------


async def test_missing_required_field_errors() -> None:
    reg = StandardToolRegistry()
    reg.register(EchoTool("read"), _schema_with_required("read", ["path"]))
    with pytest.raises(DispatchError) as excinfo:
        await reg.dispatch(_call("read", "c1", {}), AllowAllSandbox(), _CTX)
    assert excinfo.value.kind == "SchemaValidationFailed"
    assert excinfo.value.tool == "read"
    assert "path" in (excinfo.value.reason or "")


async def test_required_field_present_succeeds() -> None:
    reg = StandardToolRegistry()
    reg.register(EchoTool("read"), _schema_with_required("read", ["path"]))
    result = await reg.dispatch(_call("read", "c1", {"path": "/x"}), AllowAllSandbox(), _CTX)
    assert result.call_id == "c1"


# ----- Sandbox violation surfaces as DispatchError --------------------------


async def test_sandbox_violation_surfaces_as_dispatch_error() -> None:
    reg = StandardToolRegistry()
    reg.register(EchoTool("echo"), _schema("echo", ToolAnnotations(read_only=True)))
    with pytest.raises(DispatchError) as excinfo:
        await reg.dispatch(_call("echo", "c1", {}), DenyAllSandbox(), _CTX)
    assert excinfo.value.kind == "SandboxViolation"
    assert isinstance(excinfo.value.violation, SandboxPathEscape)


# ----- Tool returning recoverable error wraps cleanly -----------------------


async def test_tool_error_returned_as_tool_output() -> None:
    reg = StandardToolRegistry()
    reg.register(FailingTool("fail"), _schema("fail"))
    result = await reg.dispatch(_call("fail", "c1", {}), AllowAllSandbox(), _CTX)
    assert isinstance(result.output, ToolOutputError)
    assert result.output.message == "boom"
    assert result.output.recoverable is True


# ----- Rule 8: dispatch_all preserves input order ---------------------------


async def test_dispatch_all_preserves_input_order() -> None:
    reg = StandardToolRegistry()
    reg.register(EchoTool("r"), _schema("r", ToolAnnotations(read_only=True)))
    reg.register(EchoTool("d"), _schema("d", ToolAnnotations(destructive=True)))
    calls = [
        _call("d", "1", {"v": "a"}),
        _call("r", "2", {"v": "b"}),
        _call("d", "3", {"v": "c"}),
        _call("r", "4", {"v": "d"}),
    ]
    results = await reg.dispatch_all(calls, AllowAllSandbox(), _CTX)
    ids = [r.call_id for r in results if not isinstance(r, DispatchError)]
    assert ids == ["1", "2", "3", "4"]


async def test_dispatch_all_surfaces_individual_errors() -> None:
    reg = StandardToolRegistry()
    reg.register(EchoTool("ok"), _schema("ok", ToolAnnotations(read_only=True)))
    results = await reg.dispatch_all(
        [_call("ok", "1", {}), _call("missing", "2", {})],
        AllowAllSandbox(),
        _CTX,
    )
    assert not isinstance(results[0], DispatchError)
    assert isinstance(results[1], DispatchError)
    assert results[1].kind == "UnregisteredTool"


async def test_dispatch_all_concurrent_read_only_completes() -> None:
    """All-read-only batches go through the task group concurrent path."""
    reg = StandardToolRegistry()
    reg.register(EchoTool("a"), _schema("a", ToolAnnotations(read_only=True)))
    reg.register(EchoTool("b"), _schema("b", ToolAnnotations(read_only=True)))
    results = await reg.dispatch_all(
        [_call("a", "1", {}), _call("b", "2", {})],
        AllowAllSandbox(),
        _CTX,
    )
    ids = [r.call_id for r in results if not isinstance(r, DispatchError)]
    assert ids == ["1", "2"]


# ----- Rule 10: has_subagent_tools ------------------------------------------


async def test_has_subagent_tools_reflects_registration() -> None:
    reg = StandardToolRegistry()
    assert reg.has_subagent_tools() is False
    reg.register(EchoTool("echo"), _schema("echo"))
    assert reg.has_subagent_tools() is False
    reg.register(SubagentMock("subagent"), _schema("subagent"))
    assert reg.has_subagent_tools() is True


# ----- Rule 5/8: active_schemas filtered by phase and sorted -----------------


def test_active_schemas_filtered_by_phase_and_sorted() -> None:
    reg = StandardToolRegistry()
    for n in ("zeta", "alpha", "beta"):
        reg.register(EchoTool(n), _schema(n))
    reg.register_set(ToolSet(name="plan", tools=["alpha", "zeta"], phase=TaskPhase.PLANNING))
    reg.register_set(ToolSet(name="always", tools=["beta"], phase=None))

    plan = reg.active_schemas(TaskPhase.PLANNING)
    assert [s.name for s in plan] == ["alpha", "beta", "zeta"]

    exec_phase = reg.active_schemas(TaskPhase.EXECUTION)
    assert [s.name for s in exec_phase] == ["beta"]


def test_active_schemas_no_sets_returns_all() -> None:
    reg = StandardToolRegistry()
    reg.register(EchoTool("a"), _schema("a"))
    reg.register(EchoTool("b"), _schema("b"))
    assert [s.name for s in reg.active_schemas(None)] == ["a", "b"]
    # With no sets, phase-filtered call falls back to all.
    assert len(reg.active_schemas(TaskPhase.EXECUTION)) == 2


def test_register_set_duplicate_rejected() -> None:
    reg = StandardToolRegistry()
    reg.register_set(ToolSet(name="x", tools=[], phase=None))
    with pytest.raises(RegistrationError) as excinfo:
        reg.register_set(ToolSet(name="x", tools=[], phase=None))
    assert excinfo.value.kind == "DuplicateName"


def test_register_set_empty_name_rejected() -> None:
    reg = StandardToolRegistry()
    with pytest.raises(RegistrationError):
        reg.register_set(ToolSet(name="", tools=[], phase=None))


# ----- ToolSchema → model.ToolSchema projection ------------------------------


def test_to_model_schema_drops_annotations() -> None:
    s = _schema("x", ToolAnnotations(read_only=True))
    m = s.to_model_schema()
    assert m.name == "x"
    assert m.description == "x tool"
    assert m.input_schema == {"type": "object", "properties": {}}


# ----- Fixture replay -------------------------------------------------------


def _fixture_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "fixtures"
        / "tool_registry"
        / "dispatch_scenarios.json"
    )


async def test_fixture_replay_dispatch_scenarios() -> None:
    path = _fixture_path()
    scenarios = json.loads(path.read_text())
    assert isinstance(scenarios, list) and scenarios, "expected >=1 scenario"

    sandbox = AllowAllSandbox()
    for sc in scenarios:
        reg = StandardToolRegistry()
        for s in sc["register"]:
            ann = s.get("annotations", {}) or {}
            schema = ToolSchema(
                name=s["name"],
                description=s["description"],
                parameters=s["parameters"],
                annotations=ToolAnnotations(
                    read_only=bool(ann.get("read_only", False)),
                    destructive=bool(ann.get("destructive", False)),
                    idempotent=bool(ann.get("idempotent", False)),
                    open_world=bool(ann.get("open_world", False)),
                ),
            )
            reg.register(EchoTool(s["name"]), schema)
        for st in sc.get("sets", []) or []:
            phase = st.get("phase")
            reg.register_set(
                ToolSet(
                    name=st["name"],
                    tools=list(st.get("tools", [])),
                    phase=TaskPhase(phase) if phase else None,
                )
            )
        call = ToolCall(id=sc["call"]["id"], name=sc["call"]["name"], input=sc["call"]["input"])
        expected = sc["expected"]
        if expected["kind"] == "ok":
            result = await reg.dispatch(call, sandbox, _CTX)
            assert result.call_id == expected["call_id"], f"scenario {sc['name']}"
        elif expected["kind"] == "err":
            with pytest.raises(DispatchError) as excinfo:
                await reg.dispatch(call, sandbox, _CTX)
            assert excinfo.value.kind == expected["error"], f"scenario {sc['name']}"
        else:
            raise AssertionError(f"unknown expected kind: {expected['kind']!r}")
