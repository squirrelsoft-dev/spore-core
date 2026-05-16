# Shared Fixtures

This directory holds language-agnostic test data used by **all four**
spore-core implementations (Rust, TypeScript, Python, Go).

## Purpose

Cross-language consistency. The same byte-level fixture is loaded by
each language's test suite and the assertions must produce the same
result. If a fixture passes in three languages and fails in one, the
implementation in that one language is wrong — **never edit the fixture
to make a failing implementation pass.**

> If a fixture fails in one language, fix the implementation, not the fixture.

## Layout

```
fixtures/
  model_responses/      JSONL: recorded (request, response) pairs
  sandbox_violations/   SandboxProvider path-validation cases
  task_suites/          EvalHarness task definitions
```

### `model_responses/` — JSONL

One JSON object per line. Each object is a `(ModelRequest, ModelResponse)`
pair captured against a real provider by `RecordingModelInterface`
(see issue #1, ModelInterface). Schema:

```jsonc
{
  "request": { /* ModelRequest */ },
  "response": { /* ModelResponse */ },
  "provider": "anthropic|openai|ollama",
  "recorded_at": "2026-05-16T00:00:00Z"
}
```

Used by `ReplayModelInterface` in every language. Replay matches requests
deterministically (canonical JSON; tool schemas sorted by name — see the
spec section "What breaks the cache").

### `sandbox_violations/` — JSON

One file per scenario, exercising the SandboxProvider path-resolution
algorithm (issue #6). Schema:

```jsonc
{
  "name": "escape_via_symlink",
  "workspace_root": "/work",
  "raw_path": "/work/link/../../etc/passwd",
  "filesystem": { /* virtual fs layout */ },
  "allowed_paths": [],
  "denied_paths": [],
  "read_only": false,
  "operation": "read",
  "expected": { "kind": "PathEscape" }
}
```

Each language loads these and asserts the SandboxProvider reaches the
same verdict for every case.

### `task_suites/` — JSON

EvalHarness task definitions (see the Improvement Flywheel section of
the spec). Three classes:

- **Regression** — must stay passing across versions.
- **Challenge** — measure improvement over time.
- **Canary** — detect breakthroughs.

Schema:

```jsonc
{
  "suite": "regression",
  "task_id": "regression_001",
  "instruction": "...",
  "setup": { /* workspace fixture */ },
  "model_fixture": "model_responses/regression_001.jsonl",
  "expected_outcome": { /* metric or terminal state */ }
}
```

## Adding a new fixture

1. Pick the language you're most comfortable in and record against a real
   model via `RecordingModelInterface` (issue #1).
2. Commit the resulting JSONL/JSON file to the appropriate subdirectory.
3. Add a replay test in **all four** language test suites that exercises
   the same fixture and asserts the same outcome.
4. Run the full CI matrix locally before opening the PR — if any language
   diverges, fix the implementation.

A PR that adds a fixture without replay tests in all four languages will
be rejected.
