# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Cross-language parity authority

Rust is the authoritative behavioral reference for spore-core. The Rust
implementation under `rust/`, its tests, examples (`examples/rust/`),
configuration, and docs (`docs/rust/`) are **read-only** for parity work — do
not modify them to make a non-Rust port pass. If parity appears to require a
Rust change, report the exact conflict and stop; do not edit Rust absent
explicit captain permission. When a shared `fixtures/` case passes in Rust and
fails in another language, fix the port, never the fixture.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
