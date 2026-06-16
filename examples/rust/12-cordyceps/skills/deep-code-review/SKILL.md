---
name: deep-code-review
description: Perform a thorough, whole-project code review for any language or framework — map the codebase, scan for risky patterns, deep-read the important files, and produce a per-file findings report (file, line, severity, and why it matters). Load this whenever the user asks to review, audit, assess, or critique a codebase, project, module, or directory; to find bugs, code smells, security issues, SOLID or structure problems, or tech debt; or says things like "review this code", "do a code review", "audit the project", "look over this codebase", or "what's wrong with this code". This is for reviewing existing code in place — not for reviewing a single diff or pull request.
---

# Deep code review

You are doing a **read-only, whole-project code review**: find and explain problems,
don't fix them. Make no edits, and run no commands that change files (formatters,
codegen, `git` writes). Your *final reply* is the report (Phase 4); everything before
it is investigation.

## Plan of attack

The review is four phases — **Map → Static scan → Deep read → Report**. They are your
task list, in order. Get two things right before diving in:

- **Planning vs. doing.** If your harness asks you to produce a plan first, make these
  four phases the plan — roughly one task each — and emit it right away. A quick look
  to detect the stack (a single `list_dir`) is fine while planning, but do **not** run
  the scans or read source files yet — that work belongs to the execute step. Trying
  to perform the review while planning will exhaust the planning budget before you ever
  produce a plan.
- **Scope.** Review one coherent project or package at a time. If you're pointed at a
  large or multi-language monorepo (several of the manifests below, tens of thousands
  of lines), don't try to map all of it at once — pick the single most relevant package
  or subtree, or ask which to focus on. A review that tries to read everything will
  stall before it produces any findings.

## Phase 1 — Map the codebase

**1. Detect the stack.** Find the manifest files to learn the language(s) and layout,
then target the right source extensions and size thresholds:

| Manifest                          | Language | Sources                     | Size smell |
| --------------------------------- | -------- | --------------------------- | ---------- |
| `Cargo.toml`                      | Rust     | `*.rs`                      | > 400      |
| `package.json` / `tsconfig.json`  | TS / JS  | `*.ts *.tsx *.js *.jsx`     | > 300      |
| `pyproject.toml` / `requirements.txt` | Python | `*.py`                    | > 400      |
| `go.mod`                          | Go       | `*.go`                      | > 500      |
| `*.csproj`                        | C#       | `*.cs`                      | > 400      |
| `pom.xml` / `build.gradle`        | Java     | `*.java`                    | > 400      |
| `Gemfile`                         | Ruby     | `*.rb`                      | > 300      |
| `CMakeLists.txt` / `Makefile`     | C / C++  | `*.c *.cpp *.h *.hpp`       | > 500      |

A repo may use more than one. If you find none, infer the language from the dominant
file extension and use a ~400-line smell.

**2. Rank by size.** List source files biggest-first, e.g.

```
find . -name '*.rs' -not -path '*/target/*' | xargs wc -l | sort -rn
```

(swap the extension and the ignored build dir for the stack). Complexity and smells
concentrate in the biggest files: every file over its size smell is a structural
finding on its own, and the top files are your deep-read priority.

**3. See the shape.** List the directory tree, ignoring build/vendor dirs (`target`,
`node_modules`, `.venv`, `dist`, `build`, `bin`, `obj`, `vendor`). Understand module
and package boundaries before reading code — that tells you what *should* depend on
what, so you can spot it when the code doesn't.

## Phase 2 — Static pattern scan

Grep the whole tree for risk patterns *before* reading files — it's cheap and it
points you at the lines worth reading. Translate each to the stack's idiom; these are
starting points, not the full list:

- **Panic / crash paths** — `unwrap(` `expect(` `panic!` `unreachable!` (Rust);
  `throw` `process.exit` non-null `!` (TS); `raise` `sys.exit` `assert ` (Python);
  `panic(` `os.Exit` (Go); `throw ` (C#/Java).
- **Unsafe / deprecated** — `unsafe` (Rust); `any` `@ts-ignore` `eval(` (TS);
  `eval(` `exec(` `pickle` (Python); `unsafe.Pointer` (Go); `@deprecated`.
- **Swallowed errors** — empty `catch {}`, `let _ =`, `.ok();`, `except: pass`,
  `if err != nil {}` with no handling, `_ = err`.
- **Markers** — `TODO` `FIXME` `HACK` `XXX` `unimplemented` `todo!` `NotImplemented`.
- **Hardcoded secrets / magic values** — `password` `secret` `api_key` `token`,
  `AKIA…`, `BEGIN PRIVATE KEY`, long hex/base64 literals, hardcoded URLs/IPs.
- **Dead code** — `#[allow(dead_code)]`, `eslint-disable`, large commented-out blocks.

Record every hit with its file and line — these seed Phases 3 and 4.

## Phase 3 — Deep read

Read the **full contents** of every file that (a) had Phase-2 hits or (b) is over its
size smell. For each, evaluate against the lenses below — and only record what you can
point at in the code:

- **Idioms** — does it follow current best practice for this language/version, or
  fight the language?
- **SOLID & structure** — single responsibility (is one module/class doing too much?),
  sane boundaries, dependency direction. Flag god objects and tangled coupling.
- **Code smells** — long functions, deep nesting, duplicated logic, inappropriate
  intimacy between modules, primitive obsession.
- **Error handling** — are failures surfaced, or swallowed / papered over? Are results
  actually checked?
- **Security** — injection (SQL/command/path), missing input validation, insecure
  defaults, over-broad permissions, secrets in code, unsafe deserialization.
- **Bugs & gotchas** — off-by-one, race conditions, nullability/encoding assumptions,
  resource leaks (unclosed files/handles/locks), integer overflow.

Grep-first and read narrowly when you only need surrounding context; read the whole
file when it's flagged. On a large codebase, work the highest-risk and largest files
first; if you can't read them all within budget, review the top ones and say what you
skipped rather than padding the report. Don't fabricate — if a file is clean, say
nothing about it.

## Phase 4 — Findings report

Your final reply is a **markdown report**. Lead with a one-line count by severity and
the stack you found, then group findings by file, most severe file first. Every
finding names a concrete **file**, **line**, **severity**, and *why it matters* — no
generic advice ("add tests"), no style nits a formatter would catch.

Severity:
**critical** (data loss, security hole, or a crash on normal input) ·
**major** (likely bug, serious smell, broken contract) ·
**minor** (real but low-impact) ·
**info** (worth knowing, not urgent).

Use this shape:

```markdown
# Code review — <project>

**12 findings:** 1 critical · 4 major · 5 minor · 2 info.
Stack: Rust (Cargo) — 38 source files, 2 over the 400-line smell.

## src/auth/session.rs
- **critical · L142** — session token compared with `==`, not a constant-time
  check; enables a timing attack on session validation. Compare in constant time.
- **major · L88** — `unwrap()` on `parse()` of an attacker-controlled header; a
  malformed header panics the request handler (DoS).

## src/db/pool.rs
- **minor · L57** — connection acquired but not released on the early-return at L61;
  leaks under load.
```

If the project is genuinely clean, say so plainly rather than inventing findings.
