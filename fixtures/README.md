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

EvalHarness task definitions (issue #26). A suite manifest is a single
JSON object carrying a required `suite_version` (loaders reject a manifest
without it) and three disjoint task lists:

- **regression** — must stay passing across versions.
- **challenge** — measure improvement over time.
- **canary** — detect breakthroughs.

Suite schema:

```jsonc
{
  "suite_version": 1,                    // REQUIRED — loaders reject if absent
  "regression": [ /* EvalTask */ ],
  "challenge":  [ /* EvalTask */ ],
  "canary":     [ /* EvalTask */ ]
}
```

Each `EvalTask`:

```jsonc
{
  "id": "regression_s1_uppercase",
  "instruction": "...",
  "workspace_snapshot": {                // restored fresh per run, then torn down
    "kind": "files",                     // "files" | "git_ref" | "empty"
    "files": { "input.txt": "hello\n" }  // canonical hermetic form
  },
  "verifier_spec": {                     // resolved to a TaskVerifier
    "kind": "test_suite",                // test_suite | composite | metric_evaluator
                                         //  | llm_judge | always_pass | always_fail
    "command": "sh",
    "args": ["-c", "grep -q HELLO output.txt"],
    "timeout_secs": 30
  },
  "expected_turns": [2, 8],              // optional (min, max)
  "expected_cost_usd": 0.01,            // optional
  "tags": ["s1"],                        // free-form
  "timeout": 60,                         // per-run timeout, seconds
  "model_fixture": "model_responses/s1.jsonl"  // optional, for recorded replay
}
```

`workspace_snapshot.kind`:
- `files` — a `{ path: contents }` map written into a fresh tempdir. This is
  the canonical hermetic form the shipped suites use (no real git repo needed
  for cross-language replay).
- `git_ref` — `{ "repo": "...", "reference": "..." }`; restored via
  `git worktree add` for real snapshots.
- `empty` — a bare workspace.

> Note: the pre-#26 draft schema used `setup` / `expected_outcome` per task.
> Those are superseded by `workspace_snapshot` + `verifier_spec`. The
> `welch_bootstrap.json` manifest is a statistics oracle (hand-computed Welch
> t/p and seeded bootstrap CI bounds), not a task suite.

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
