---
name: implement
description: >
  Implement a spore-core component across all four language targets (Rust,
  TypeScript, Python, Go) from a GitHub issue spec. Use this skill whenever
  the user says "implement issue #N", "implement <ComponentName>", "build the
  <X> component", or anything that involves translating a spore-core spec issue
  into working code. Also use it when the user wants to implement a single
  language target ("implement the Rust version of issue #5"). The skill handles
  reading the spec, planning, implementing Rust first as the reference, then
  spawning parallel subagents for the remaining languages, running shared
  fixtures, and verifying cross-language consistency.
---

# Skill: implement

Implement a spore-core component from a GitHub issue spec across all four
language targets. Rust is always implemented first and serves as the reference
implementation for the other three languages.

## Usage

```
implement <issue-number>
implement <issue-number> --lang rust        # Rust only
implement <issue-number> --lang typescript  # single language
```

---

## Before You Write Anything

Read these in order. Do not skip any of them.

1. **The spec issue** — `https://github.com/squirrelsoft-dev/spore-core/issues/<N>`
2. **`docs/harness-engineering-concepts.md`** — full system context, component
   relationships, rules that cross component boundaries
3. **Any related issues** referenced in the spec issue — fetch and read them
4. **`rust/CONVENTIONS.md`** — Rust-specific patterns and decisions
5. **`fixtures/README.md`** — shared fixture format and how all four
   implementations use them

If any of these files don't exist yet, note it and proceed — the monorepo
scaffold may not be complete.

---

## Phase 1 — Read and Plan

After reading the spec:

1. Identify all types, traits, rules, and implementor notes in the issue
2. Map each rule from the spec to a specific test case — if a rule cannot be
   tested, flag it
3. Identify any ambiguities. **Do not guess — stop and surface them.**
   An ambiguity is anything where two reasonable engineers would implement
   differently. Flag it as a comment before writing code.
4. Write a brief plan as a doc comment at the top of the implementation file:
   - Types to define
   - Trait methods and what each does
   - Rules from the spec and how each is enforced
   - Any ambiguities flagged with `// SPEC QUESTION: ...`

If there are unresolved ambiguities after flagging, stop here. Surface them
before proceeding to Phase 2.

---

## Phase 2 — Rust Implementation

Rust is always first. Its type system surfaces spec ambiguity fastest and
most verbosely. The Rust implementation becomes the reference for all other
languages.

### Implementation steps

1. Create `rust/crates/spore-core/src/<component>.rs`
2. Implement:
   - All types from the spec (structs, enums, error types)
   - The trait definition
   - Any standard implementations specified (e.g. `WorkspaceScopedSandbox`,
     `NullCacheProvider`, `StandardTerminationPolicy`)
   - Mock implementation under `#[cfg(feature = "test-utils")]` if the spec
     calls for one
3. Update `rust/crates/spore-core/src/lib.rs` to export the new module
4. Write unit tests inline in `#[cfg(test)]` covering:
   - Every rule stated in the spec (one test per rule minimum)
   - Every error variant (verify it is returned under the right conditions)
   - Edge cases from the implementor notes
   - Happy path for every trait method

### Verification gates — do not proceed until all pass

```bash
cd rust
cargo build          # zero errors
cargo clippy         # zero warnings
cargo test           # all tests pass
cargo fmt --check    # no formatting issues
```

Fix all issues before proceeding. Do not move to Phase 3 with a failing build.

### Fixtures

Check `fixtures/` for existing fixture files relevant to this component.

If none exist, create the first fixture:
- Model-touching components: create a `fixtures/model_responses/<component>/`
  JSONL file with at minimum one recorded `(ModelRequest, ModelResponse)` pair
- Sandbox components: create `fixtures/sandbox_violations/<component>/`
  test cases covering every `SandboxViolation` variant
- Add a fixture-replay test in the Rust test suite that reads the fixture file
  and asserts the expected outcome

Fixture format: see `fixtures/README.md`

### Commit

```
feat(rust): implement <ComponentName> (#<issue-number>)
```

---

## Phase 3 — Parallel Language Implementations

Only proceed after Phase 2 is complete and all tests pass.

Spawn one subagent per remaining language. Each subagent receives:

- The original issue text
- The Rust implementation just completed
- The language-specific `CONVENTIONS.md`
- The shared fixtures in `fixtures/<component>/`
- The instructions below

### Subagent instructions (provide verbatim to each)

> You are implementing `<ComponentName>` for `<language>` in the spore-core
> monorepo.
>
> Read these before writing any code:
> 1. The spec issue (provided above)
> 2. `<language>/CONVENTIONS.md`
> 3. The Rust reference implementation (provided above) — use it to resolve
>    spec ambiguity, not to copy its structure
>
> Rules:
> - Implement the same types, the same interface, the same rules as the spec
> - Write idiomatically for `<language>` — do not transliterate Rust
> - Write unit tests covering every rule in the spec
> - Add fixture-replay tests using the shared fixtures in `fixtures/<component>/`
> - Run the full test suite — all tests must pass before committing
> - Run linter and formatter
> - Commit: `feat(<language>): implement <ComponentName> (#<issue-number>)`
> - Report back: DONE with test count, or BLOCKED with a specific question
>
> If you are blocked on a genuine spec ambiguity, stop and report BLOCKED.
> Do not make a judgment call — the Rust implementation is the reference.

### Language targets and test commands

| Language | Test | Lint | Format |
|---|---|---|---|
| TypeScript | `pnpm test` | `pnpm lint` | `pnpm format --check` |
| Python | `uv run pytest` | `uv run ruff check` | `uv run ruff format --check` |
| Go | `go test ./...` | `go vet ./...` | `gofmt -l .` |

If a subagent reports BLOCKED:
1. Read the specific question
2. Resolve it using the spec and Rust implementation
3. Update the spec issue with the resolution as a comment
4. Re-run that subagent

---

## Phase 4 — Consistency Verification

After all subagents report DONE, verify cross-language consistency.

### Run all four test suites

```bash
cd rust && cargo test
cd typescript && pnpm test
cd python && uv run pytest
cd go && go test ./...
```

All must pass before continuing.

### Run all fixture tests

Fixture tests must pass in all four languages against the same fixture files.
A fixture test that passes in three languages and fails in one means that
implementation has diverged — fix the implementation, not the fixture.

**Fixtures are ground truth. Never modify a fixture to make a test pass.**

### Consistency checks

After all tests pass, verify:

- All four implementations export the same public type names (mapped to
  language naming conventions — `SandboxViolation` in Rust,
  `SandboxViolation` in TypeScript, `SandboxViolation` in Python,
  `SandboxViolation` in Go)
- All four implementations enforce the same rules from the spec
- All four fixture tests produce the same outcomes
- Error variants match — every error variant in the spec exists in all four

### If a test fails in one language but passes in others

The failing implementation has diverged from the spec. Fix it.
Do not modify the fixture or the passing implementations.

### If a test fails in all four languages

The fixture or spec may be wrong. Stop, analyze, and flag it as a spec issue.

---

## Phase 5 — Update the Issue

After all tests pass and consistency is verified:

1. Post a comment on the GitHub issue with the summary (see Output format below)
2. Note any spec ambiguities discovered and how they were resolved
3. Note any rules that were impossible or awkward to implement in a specific
   language — document the divergence explicitly
4. If any fixes were needed for consistency: commit with
   `fix: cross-language consistency for <ComponentName> (#<issue-number>)`

---

## Output

When complete, report:

```
✅ <ComponentName> (#<issue-number>)

Rust:       cargo test — N tests passed
TypeScript: pnpm test  — N tests passed
Python:     pytest     — N tests passed
Go:         go test    — N tests passed

Fixtures:   fixtures/<component>/ — N fixture files, all pass in all 4 languages

Divergences: <none | description of any intentional language-specific differences>
Spec notes:  <any ambiguities discovered and how they were resolved>
Gaps found:  <any spec issues that surfaced during implementation>
```

---

## Key Rules

- **Phase 2 before Phase 3 always.** Rust surfaces spec ambiguity before four
  parallel subagents each handle ambiguity differently.
- **Do not transliterate Rust.** Idiomatic code in each language is the goal.
  The Rust implementation is a spec reference, not a translation template.
- **Fixtures are ground truth.** Modifying a fixture to make a test pass
  invalidates the consistency guarantee.
- **BLOCKED beats wrong.** A subagent that stops and asks is better than one
  that silently makes a judgment call.
- **Zero warnings policy.** All four implementations must pass their linters
  with zero warnings before the skill is complete.
- **Spec gaps are issues.** If implementation reveals a gap or ambiguity in
  the spec, open a GitHub issue for it. Do not paper over it in code.
