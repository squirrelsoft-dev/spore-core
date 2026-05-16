# Python Conventions — spore-core

Read alongside `/docs/harness-engineering-concepts.md` before implementing
any component (issues #1–#13).

## Key decisions (already made — do not relitigate)

- **Python**: 3.12+. Type hints on every public function and method.
- **Package manager**: `uv` workspace. Never `pip`/`poetry`/`pipenv`
  invoked directly — always `uv run` / `uv add`.
- **Async**: `asyncio` with `anyio` as the structured-concurrency layer.
  Use `anyio.CancelScope` / `anyio.create_task_group` rather than raw
  `asyncio.gather`. Public async APIs accept an optional `anyio` cancel
  scope where relevant.
- **Structural typing**: `typing.Protocol` for all harness interfaces —
  not abstract base classes. Concrete impls do not inherit the Protocol;
  they satisfy it structurally.
- **Validation**: `pydantic` v2 for serialized payloads (ModelRequest,
  ModelResponse, PausedState, fixture rows). Plain dataclasses for
  internal value types.
- **Lint**: `ruff` for lint + format. No `black`/`isort`/`flake8`.

## Protocol pattern

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ModelInterface(Protocol):
    async def call(self, req: ModelRequest) -> ModelResponse: ...
    async def count_tokens(self, req: ModelRequest) -> int: ...
```

Components are injected as Protocol-typed parameters. No registration,
no metaclass magic.

## Error handling

- One exception class per component, all inheriting from a package root
  `SporeError`.
- Always-halt errors (spec Layer 1) inherit from a marker class
  `AlwaysHaltError(SporeError)`.
- Recoverable failures are returned as values (e.g. `ToolOutput.Error`),
  not raised, when crossing the agent/harness boundary.
- Never bare-`except` — always name the exception class.

## Naming

- Modules: `snake_case`. One module per harness component.
- Classes: `UpperCamelCase`. Protocol names match the spec exactly
  (`ModelInterface`, `ToolRegistry`).
- Functions, variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- IDs: `NewType` aliases — `SessionId = NewType("SessionId", str)`.

## Layout

```
python/
  pyproject.toml          ← uv workspace root
  packages/
    spore_core/
      pyproject.toml
      src/spore_core/     ← one module per component
      tests/
    spore_tools/          ← standard Tool implementations
    spore_eval/           ← evaluation harness
```

`src/` layout is mandatory — never put package code at the top level.

## Running tests

- Full suite:        `uv run pytest`
- Per package:       `uv run --package spore-core pytest`
- Unit only:         `uv run pytest -m "not integration"`
- Single by name:    `uv run pytest -k "<substring>"`
- Single file:       `uv run pytest packages/spore_core/tests/test_x.py`

Async tests use `pytest-asyncio` in auto mode (configured in the root
`pyproject.toml`).

## Lint / format

- `uv run ruff check .` — CI gate
- `uv run ruff format --check .` — CI gate

## Adding a dependency

`uv add --package <package-name> <dep>`. Pin to a minor version. Devtools go
to the workspace root with `uv add --dev <dep>`.

## Cross-language consistency

Public types (`ModelResponse`, `ToolOutput`, `PausedState`) must match
the Rust, TypeScript, and Go definitions. Pydantic field names use
`snake_case` and rely on alias generators only when wire compatibility
demands it. See `/fixtures/README.md`.
