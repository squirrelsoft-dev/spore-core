---
name: implement
description: >
  Implement a spore-core component across all four language targets (Rust,
  TypeScript, Python, Go) from a GitHub issue spec. Use this skill whenever
  the user says "implement issue #N", "implement <ComponentName>", "build the
  <X> component", or anything that involves translating a spore-core spec issue
  into working code. Also use it when the user wants to implement a single
  language target ("implement the Rust version of issue #5"). The skill
  orchestrates subagents end-to-end: a research/planning agent, a Rust
  reference-implementation agent, three parallel language agents, and a
  cross-language verifier agent. The main thread coordinates — it does not
  write implementation code itself.
---

# Skill: implement

Implement a spore-core component from a GitHub issue spec across all four
language targets. The main thread is an **orchestrator**: it spawns subagents
for every phase that involves reading, planning, writing code, or verifying.
The orchestrator only synthesizes results between phases and surfaces blockers
to the user.

Rust is always implemented first and serves as the reference for the other
three languages.

## Usage

```
implement <issue-number>
implement <issue-number> --lang rust        # Rust only — still uses subagents
implement <issue-number> --lang typescript  # single non-Rust language
```

---

## Orchestration model

The main thread runs five phases. **Every phase delegates the actual work to
one or more subagents.** The main thread:

- Spawns subagents with self-contained prompts (each agent starts with no
  context from the conversation)
- Waits for structured results (DONE / BLOCKED with specific question)
- Resolves cross-phase decisions and forwards context between agents
- Never writes implementation code directly — if you find yourself reaching
  for Edit/Write on a `.rs`/`.ts`/`.py`/`.go` file, stop and spawn an agent

Independent subagents in the same phase MUST be launched in a single message
(parallel tool calls) — never sequentially.

---

## Phase 1 — Research and Plan (1 subagent)

Spawn a `general-purpose` agent (or `Plan` if available) to read the spec and
produce a plan. Prompt verbatim:

> You are the planning agent for spore-core component `<ComponentName>`,
> GitHub issue #<N> at
> `https://github.com/squirrelsoft-dev/spore-core/issues/<N>`.
>
> Read in order, then return a plan:
> 1. The spec issue (fetch with `gh issue view <N>`)
> 2. `docs/harness-engineering-concepts.md`
> 3. Any related issues referenced in the spec — fetch each
> 4. `rust/CONVENTIONS.md`
> 5. `fixtures/README.md`
>
> If any file is missing, note it and continue.
>
> Return a structured report:
> - **Types**: every struct/enum/error to define
> - **Trait/interface**: method signatures and what each does
> - **Rules**: every rule from the spec, mapped to a concrete test case;
>   flag rules that cannot be tested
> - **Standard impls**: any concrete implementations the spec calls for
> - **Fixtures needed**: which fixture files this component requires
> - **Ambiguities**: anything where two reasonable engineers would diverge.
>   Mark each `SPEC QUESTION: …`. Do not guess.
>
> If unresolved ambiguities remain, report BLOCKED with the questions.
> Otherwise report DONE with the plan.

If the agent reports BLOCKED, surface the questions to the user and stop.
Do not proceed until ambiguities are resolved.

---

## Phase 2 — Rust Reference Implementation (1 subagent)

Spawn a coding agent (`general-purpose`, or `executor` if available — prefer
`model=opus` for non-trivial components). Prompt verbatim, with the Phase 1
plan inlined:

> You are implementing `<ComponentName>` in Rust for spore-core. This is the
> reference implementation — the other three languages will derive from it.
>
> Plan from Phase 1:
> <PASTE PHASE 1 REPORT>
>
> Spec: `https://github.com/squirrelsoft-dev/spore-core/issues/<N>` (fetch
> with `gh issue view <N>`).
>
> Also read: `rust/CONVENTIONS.md`, `fixtures/README.md`.
>
> Steps:
> 1. Create `rust/crates/spore-core/src/<component>.rs` with all types, the
>    trait, any standard impls, and a `#[cfg(feature = "test-utils")]` mock
>    if the spec calls for one.
> 2. Update `rust/crates/spore-core/src/lib.rs` to export the module.
> 3. Write inline `#[cfg(test)]` tests covering every rule, every error
>    variant, edge cases from implementor notes, and happy paths.
> 4. Create or extend fixtures under `fixtures/<component>/` per the plan.
>    Add a fixture-replay test that reads each fixture and asserts the
>    expected outcome.
> 5. Run all gates — every one must pass:
>    ```
>    cd rust && cargo build && cargo clippy -- -D warnings && cargo test && cargo fmt --check
>    ```
> 6. Commit: `feat(rust): implement <ComponentName> (#<N>)`
>
> Top-of-file doc comment must list: types, trait methods, rules enforced,
> and any `// SPEC QUESTION:` markers (there should be none if Phase 1 was
> clean).
>
> Report DONE with: test count, fixture file paths, commit SHA.
> Report BLOCKED with a specific question if you hit a genuine spec gap.

If BLOCKED, resolve with the user, then re-spawn with the resolution
appended to the prompt. Do not proceed to Phase 3 until DONE.

If the user passed `--lang rust`, stop here.

---

## Phase 3 — Parallel Language Implementations (3 subagents in parallel)

Spawn three subagents **in a single message** — TypeScript, Python, Go. Each
gets the same template, with `<language>` and tooling substituted:

> You are implementing `<ComponentName>` for `<language>` in the spore-core
> monorepo.
>
> Read before writing code:
> 1. The spec — fetch with `gh issue view <N>`
> 2. `<language>/CONVENTIONS.md`
> 3. The Rust reference at `rust/crates/spore-core/src/<component>.rs` — use
>    it to resolve spec ambiguity, NOT to copy structure
> 4. Shared fixtures at `fixtures/<component>/`
>
> Rules:
> - Same types, same interface, same rules as the spec
> - Idiomatic `<language>` — do not transliterate Rust
> - Unit tests for every rule
> - Fixture-replay tests against `fixtures/<component>/`
> - Full test suite, linter, formatter all clean before committing
> - Commit: `feat(<language>): implement <ComponentName> (#<N>)`
>
> Test/lint/format commands:
>
> | Language | Test | Lint | Format |
> |---|---|---|---|
> | TypeScript | `pnpm test` | `pnpm lint` | `pnpm format --check` |
> | Python | `uv run pytest` | `uv run ruff check` | `uv run ruff format --check` |
> | Go | `go test ./...` | `go vet ./...` | `gofmt -l .` |
>
> Report DONE with test count and commit SHA, or BLOCKED with a specific
> question. Do NOT make judgment calls on spec ambiguity — BLOCKED beats
> wrong.

If any subagent reports BLOCKED:
1. Read the question
2. Resolve from the spec and Rust implementation
3. Comment the resolution on the GitHub issue
4. Re-spawn just that one subagent with the resolution appended

If the user passed `--lang typescript|python|go`, spawn only that one.

---

## Phase 4 — Cross-Language Verification (1 subagent)

Spawn a verifier agent (`general-purpose`, read-heavy — `Explore` is fine if
no edits expected). Prompt verbatim:

> You are verifying cross-language consistency for `<ComponentName>` (#<N>)
> in spore-core.
>
> Run each test suite and capture pass/fail:
> ```
> cd rust && cargo test
> cd typescript && pnpm test
> cd python && uv run pytest
> cd go && go test ./...
> ```
>
> Then verify:
> - Public type names match across all four (mapped to each language's naming
>   conventions)
> - Every spec rule is enforced in all four
> - Fixture tests produce identical outcomes in all four
> - Every error variant in the spec exists in all four
>
> Rules:
> - Fixtures are ground truth. NEVER modify a fixture to make a test pass.
> - If a test fails in one language but passes in others, the failing
>   implementation diverged — report the divergence with file/line.
> - If a test fails in all four, the fixture or spec may be wrong — report
>   it as a spec gap.
>
> Do NOT fix anything yourself. Report:
> - PASS with per-language test counts, or
> - DIVERGED with specific files and what differs, or
> - SPEC GAP with the failing fixture and analysis.

If DIVERGED: spawn the appropriate language agent from Phase 3 with the
divergence as a fix prompt. Commit fixes as
`fix: cross-language consistency for <ComponentName> (#<N>)`. Then re-run
Phase 4.

If SPEC GAP: surface to the user and open a GitHub issue.

---

## Phase 5 — Issue Update (main thread, or 1 small subagent)

After Phase 4 reports PASS, post a summary comment on the GitHub issue using
the Output format below. This is small enough to run directly with
`gh issue comment <N> --body "…"`, but may be delegated to a subagent if the
write-up requires synthesizing significant context.

Document:
- Any spec ambiguities discovered and how they were resolved
- Any rules awkward in a specific language — document the divergence
- Any spec gaps that surfaced (link the new issues)

---

## Output

When complete, report:

```
✅ <ComponentName> (#<N>)

Rust:       cargo test — N tests passed
TypeScript: pnpm test  — N tests passed
Python:     pytest     — N tests passed
Go:         go test    — N tests passed

Fixtures:   fixtures/<component>/ — N files, all pass in all 4 languages

Divergences: <none | intentional language-specific differences>
Spec notes:  <ambiguities discovered and resolutions>
Gaps found:  <new spec issues opened>
```

---

## Key Rules

- **The orchestrator does not write code.** Every implementation edit goes
  through a subagent. The main thread reads reports and routes work.
- **Phase 2 before Phase 3 always.** Rust surfaces ambiguity before three
  parallel agents each resolve it differently.
- **Phase 3 agents launch in a single message.** Parallel tool calls only —
  never sequential.
- **Do not transliterate Rust.** Idiomatic per-language code. The Rust impl
  is a spec reference, not a translation template.
- **Fixtures are ground truth.** Modifying a fixture to pass a test
  invalidates the consistency guarantee.
- **BLOCKED beats wrong.** A subagent that stops and asks is better than one
  that silently makes a judgment call.
- **Zero warnings policy.** All four implementations pass their linters with
  zero warnings before the skill is complete.
- **Spec gaps are issues.** File a GitHub issue — do not paper over in code.
