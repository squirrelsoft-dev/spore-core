"""Task-suite types for the EvalHarness (issue #26).

Mirrors the Rust reference at ``rust/crates/spore-eval/src/task.rs``.

Ships: :class:`EvalError` (one exception per component, rooted at
:class:`spore_core.errors.SporeError`), :class:`VerificationResult`
(Rule 7/8), the :data:`WorkspaceSnapshot` tagged union (Resolution 2), the
:data:`VerifierSpec` tagged union, :class:`EvalTask`, and :class:`TaskSuite`
(Rules 1, 5, 6).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from spore_core.errors import SporeError

# ============================================================================
# Errors (CONVENTIONS.md: one exception per component, rooted at SporeError)
# ============================================================================


class EvalError(SporeError):
    """Base error for the EvalHarness. Recoverable per-run failures are
    returned as values (failed :class:`RunResult`); this is raised only for
    programmer/manifest errors (Rules 6, 8) and infrastructure failures."""


class MissingSuiteVersionError(EvalError):
    """A manifest was loaded without the required ``suite_version`` (Rule 6)."""

    def __init__(self) -> None:
        super().__init__("manifest is missing required field `suite_version`")


class ManifestParseError(EvalError):
    """A manifest failed to parse."""


class VerifyError(EvalError):
    """A verifier failed in a way that is not a normal "task failed" outcome
    (e.g. an out-of-range score, Rule 8)."""


class WorktreeError(EvalError):
    """Restoring or tearing down a workspace/worktree failed (Rules 2-3)."""


class MissingMetricsError(EvalError):
    """An :class:`EvalHarness` was built or run without the metrics it needs."""


# ============================================================================
# Pydantic base
# ============================================================================


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# VerificationResult (Rule 7/8)
# ============================================================================


class VerificationResult(_Model):
    """The outcome of a :class:`TaskVerifier` (Rule 7): a pass/fail flag, a
    ``score`` clamped to ``[0.0, 1.0]``, a human-readable ``detail``, and
    granular ``signals``."""

    passed: bool
    score: float
    detail: str = ""
    signals: dict[str, float] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def _score_in_range(cls, v: float) -> float:
        # Rule 8: out-of-range score is a verifier error.
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score {v} out of range [0.0, 1.0]")
        return v

    @classmethod
    def new(cls, passed: bool, score: float, detail: str = "") -> VerificationResult:
        """Build a result, erroring (Rule 8, via :class:`VerifyError`) on an
        out-of-range score."""
        if not (0.0 <= score <= 1.0):
            raise VerifyError(f"score {score} out of range [0.0, 1.0]")
        return cls(passed=passed, score=score, detail=detail)

    @classmethod
    def clamped(cls, passed: bool, score: float, detail: str = "") -> VerificationResult:
        """Build a result, clamping any out-of-range ``score`` into
        ``[0.0, 1.0]`` instead of erroring."""
        return cls(passed=passed, score=max(0.0, min(1.0, score)), detail=detail)

    def with_signal(self, key: str, value: float) -> VerificationResult:
        self.signals[key] = value
        return self


# ============================================================================
# WorkspaceSnapshot (Resolution 2) — tagged union on ``kind``
# ============================================================================


class WorkspaceSnapshotFiles(_Model):
    """Canonical hermetic form: a map of relative path → file contents."""

    kind: Literal["files"] = "files"
    files: dict[str, str] = Field(default_factory=dict)


class WorkspaceSnapshotGitRef(_Model):
    """A real git snapshot: a repo URL/path and a ref to check out."""

    kind: Literal["git_ref"] = "git_ref"
    repo: str
    reference: str


class WorkspaceSnapshotEmpty(_Model):
    """An empty workspace."""

    kind: Literal["empty"] = "empty"


WorkspaceSnapshot = Annotated[
    WorkspaceSnapshotFiles | WorkspaceSnapshotGitRef | WorkspaceSnapshotEmpty,
    Field(discriminator="kind"),
]


# ============================================================================
# MetricDirection (serializable mirror of OptimizationDirection)
# ============================================================================

MetricDirection = Literal["minimize", "maximize"]


# ============================================================================
# VerifierSpec — serializable; resolved to a TaskVerifier
# ============================================================================


class VerifierSpecTestSuite(_Model):
    """Run a command in the workspace; score = pass rate (Rule 10)."""

    kind: Literal["test_suite"] = "test_suite"
    command: str
    args: list[str] = Field(default_factory=list)
    timeout_secs: int | None = None


class CompositeChildSpec(_Model):
    """One child of a :class:`VerifierSpecComposite` with its weight and
    required-ness."""

    spec: VerifierSpec
    weight: float
    required: bool = False


class VerifierSpecComposite(_Model):
    """Combine children by weight; ``required`` children must all pass
    (Rule 11)."""

    kind: Literal["composite"] = "composite"
    children: list[CompositeChildSpec] = Field(default_factory=list)


class VerifierSpecMetricEvaluator(_Model):
    """Adapt a metric evaluator, normalizing its value to a score (Rule 12)."""

    kind: Literal["metric_evaluator"] = "metric_evaluator"
    descriptor: str = ""
    direction: MetricDirection = "maximize"
    min: float | None = None
    max: float | None = None
    threshold: float | None = None


class VerifierSpecLlmJudge(_Model):
    """An LLM-judge verifier; non-deterministic (Rule 13)."""

    kind: Literal["llm_judge"] = "llm_judge"
    rubric: str
    score_range: tuple[float, float]


class VerifierSpecAlwaysPass(_Model):
    """Test scaffolding: always passes with score 1.0."""

    kind: Literal["always_pass"] = "always_pass"


class VerifierSpecAlwaysFail(_Model):
    """Test scaffolding: always fails with score 0.0."""

    kind: Literal["always_fail"] = "always_fail"


VerifierSpec = Annotated[
    VerifierSpecTestSuite
    | VerifierSpecComposite
    | VerifierSpecMetricEvaluator
    | VerifierSpecLlmJudge
    | VerifierSpecAlwaysPass
    | VerifierSpecAlwaysFail,
    Field(discriminator="kind"),
]

# Resolve the forward reference in CompositeChildSpec now that VerifierSpec exists.
CompositeChildSpec.model_rebuild()


# ============================================================================
# TaskCategory + EvalTask + TaskSuite
# ============================================================================

TaskCategory = Literal["regression", "challenge", "canary"]

_DEFAULT_TIMEOUT_SECS = 300


class EvalTask(_Model):
    """One evaluation task. ``timeout`` is serialized as whole seconds
    (Rule 4)."""

    id: str
    instruction: str
    workspace_snapshot: WorkspaceSnapshot
    verifier_spec: VerifierSpec
    expected_turns: tuple[int, int] | None = None
    expected_cost_usd: float | None = None
    tags: list[str] = Field(default_factory=list)
    timeout: int = _DEFAULT_TIMEOUT_SECS
    model_fixture: str | None = None


class TaskSuite(_Model):
    """A versioned task suite holding three disjoint task lists (Rule 1).

    ``suite_version`` is required (Rule 6); the loader rejects a manifest
    without it (see :mod:`spore_eval.eval.manifest`)."""

    suite_version: int
    regression: list[EvalTask] = Field(default_factory=list)
    challenge: list[EvalTask] = Field(default_factory=list)
    canary: list[EvalTask] = Field(default_factory=list)

    def all_tasks(self) -> list[tuple[TaskCategory, EvalTask]]:
        """All tasks across the three categories, tagged with their category."""
        out: list[tuple[TaskCategory, EvalTask]] = []
        out.extend(("regression", t) for t in self.regression)
        out.extend(("challenge", t) for t in self.challenge)
        out.extend(("canary", t) for t in self.canary)
        return out
