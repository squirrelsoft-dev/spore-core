"""Tests for the SC-8 builder presets (parity with Rust's
``HarnessBuilder::{coding_agent, hill_climber}`` in
``rust/crates/spore-core/src/harness.rs``).

``coding_agent`` is the looper preset: a read-write workspace-scoped sandbox,
the full coding tool catalogue, the built-in coding system prompt, and
``AutoContinue { 10, 25 }`` — so a consumer collapses to one call. An
unresolvable workspace surfaces a typed ``SandboxBuildError``, not a crash.
``hill_climber`` is the cordyceps preset: it registers the scoring evaluator
under the default handle and defaults to ``AutoContinue``, leaving sandbox /
tools / system prompt to the caller.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spore_core import (
    CODING_AGENT_SYSTEM_PROMPT,
    PRESET_MAX_AUTO_GRANTS,
    PRESET_STEPS_PER_GRANT,
    HarnessBuilder,
    ProviderInfo,
    SandboxBuildError,
    WorkspaceScopedSandbox,
)
from spore_core.execution_registry import EscalationModeAutoContinue
from spore_core.harness import OptimizationDirection
from spore_core.metric import MetricResult
from spore_core.model import MockModelInterface


def _model() -> MockModelInterface:
    """A scripted model with a sized context window — enough to build a config
    without a live provider (mirrors Rust's ``preset_mock_model``)."""
    return MockModelInterface(ProviderInfo(name="test", model_id="test-1", context_window=8192))


class _StubEvaluator:
    """Minimal :class:`~spore_core.metric.MetricEvaluator`-shaped double — just
    enough to be registered + resolved under the default handle."""

    async def evaluate(self, sandbox: object, session_state: object) -> MetricResult:
        return MetricResult(value=1.0, raw_output="", duration=0.0)

    def direction(self) -> OptimizationDirection:
        return "maximize"

    def description(self) -> str:
        return "stub"


# ---------------------------------------------------------------------------
# coding_agent — the looper preset
# ---------------------------------------------------------------------------


def test_coding_agent_preset_wires_sandbox_tools_prompt_and_autocontinue(
    tmp_path: Path,
) -> None:
    """``coding_agent`` wires the read-write workspace sandbox, the coding tool
    catalogue, the built-in coding system prompt, and ``AutoContinue {10, 25}``
    (the looper preset) — so a consumer collapses to one call. Mirrors Rust's
    ``coding_agent_preset_wires_sandbox_tools_prompt_and_autocontinue``."""
    cfg = HarnessBuilder.coding_agent(_model(), tmp_path).build_config()

    # A read-write workspace-scoped sandbox rooted at the workspace.
    assert isinstance(cfg.sandbox, WorkspaceScopedSandbox)
    assert cfg.sandbox.workspace_root() == tmp_path.resolve()
    assert cfg.sandbox.config.read_only is False

    # Autonomous-but-capped escalation with the preset defaults (SC-5).
    assert isinstance(cfg.escalation_mode, EscalationModeAutoContinue)
    assert cfg.escalation_mode.max_grants == PRESET_MAX_AUTO_GRANTS
    assert cfg.escalation_mode.steps_per_grant == PRESET_STEPS_PER_GRANT

    # The built-in coding system prompt is installed.
    assert cfg.system_prompt == CODING_AGENT_SYSTEM_PROMPT

    # The coding catalogue is folded into the catalogue registry (bridged
    # per-run), not the harness-loop ``tool_registry``. Assert a representative
    # sample of ``coding_set()``.
    assert cfg.catalogue_registry is not None
    names = {s.name for s in cfg.catalogue_registry.active_schemas(None)}
    for expected in (
        "read_file",
        "write_file",
        "edit_file",
        "bash_command",
        "send_message",
    ):
        assert expected in names, f"coding_set must include {expected}; got {names}"


def test_coding_agent_preset_errors_on_missing_workspace() -> None:
    """A workspace path that can't be resolved surfaces a typed
    ``SandboxBuildError`` — the sandbox requires a canonical, existing root, and
    the fallible constructor surfaces an error rather than crashing. Mirrors
    Rust's ``coding_agent_preset_errors_on_missing_workspace``."""
    missing = "/spore-sc8-does-not-exist-37a1/nope"
    with pytest.raises(SandboxBuildError):
        HarnessBuilder.coding_agent(_model(), missing)


# ---------------------------------------------------------------------------
# hill_climber — the cordyceps preset
# ---------------------------------------------------------------------------


def test_hill_climber_preset_registers_evaluator_and_autocontinue() -> None:
    """``hill_climber`` registers the scoring evaluator (required for the
    HillClimbing strategy) under the default handle and defaults to
    ``AutoContinue`` — the cordyceps preset. Mirrors Rust's
    ``hill_climber_preset_registers_evaluator_and_autocontinue``."""
    evaluator = _StubEvaluator()
    cfg = HarnessBuilder.hill_climber(_model(), evaluator).build_config()

    # The evaluator resolves under the DEFAULT empty handle (the config folds it
    # there), so a bare ``HillClimbingConfig.evaluator`` ("") finds it.
    assert cfg.registry.resolve_metric_evaluator("") is evaluator

    # Autonomous-but-capped escalation by default.
    assert isinstance(cfg.escalation_mode, EscalationModeAutoContinue)
    assert cfg.escalation_mode.max_grants == PRESET_MAX_AUTO_GRANTS
    assert cfg.escalation_mode.steps_per_grant == PRESET_STEPS_PER_GRANT
