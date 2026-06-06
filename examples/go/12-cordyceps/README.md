# Example 12 — cordyceps: a fully autonomous task-completion agent (capstone)

> **You give it a task. It does not stop until the job is done — and when a
> worker gets stuck or uncertain, it asks for help (a sibling helper, then a
> human) rather than giving up.**

This is the capstone of the suite and the "very basic" seed of the
[`cordyceps`](https://github.com/squirrelsoft-dev/cordyceps) project: a
Hermes-style autonomous agent. The concrete task here is a **read-only audit of
this repo's Rust code**. The orchestrator decomposes the work itself
(crate → module), drives a per-module deep-dive to a finished `findings.md`,
presents the top 5, and offers to file them as GitHub issues.

It composes everything the suite has built — subagents-as-tools (11), custom
sandboxed tools (05), `web_search` (06), `memory` (07), `task_list` — and adds
two new capabilities: **runtime skill loading** and **a generalized
consult/escalation ladder** (the consumer of #114).

## Topology (depth-1)

```
orchestrator  (ReAct, gemma4:e4b)
  tools: list_dir, grep, task_list, memory, write_file, bash_command,
         analysis_worker (SubagentTool, + consult handlers)
  ├── analysis_worker  (ReAct, gemma, Isolated) — deep-dive audits ONE module
  │     tools: read_file, grep, research_best_practices, consult_advisor, load_skill
  ├── research_worker  (ReAct, gemma, Isolated) — web_search   [consult handler: kind=research]
  └── advisor          (ReAct, minimax-m3:cloud, Isolated) — read_file, grep  [consult handler: kind=advice]
```

| Agent | Model | Tools | Role |
| --- | --- | --- | --- |
| **orchestrator** | `gemma4:e4b` | `list_dir`, `grep`, `task_list`, `memory`, `write_file`, `bash_command`, `analysis_worker` | enumerate → dispatch → accumulate → top-5 → file |
| **analysis_worker** | `gemma4:e4b` | `read_file`, `grep`, `research_best_practices`, `consult_advisor`, `load_skill` | audit ONE module, emit JSON findings |
| **research_worker** | `gemma4:e4b` | `web_search` | `kind=research` consult handler |
| **advisor** | `minimax-m3:cloud` | `read_file`, `grep` | `kind=advice` consult handler |

## Why each decision

### ReAct, not PlanExecute (the orchestrator)

The orchestrator runs `StrategyReAct`, not `StrategyPlanExecute`. The PlanExecute
plan turn is **tool-free** — it cannot call `list_dir` to discover which crates
and modules actually exist. This audit must *enumerate the repo dynamically*
(crates under `rust/crates/`, then `src/*.rs` per crate) before it knows what the
tasks are. ReAct + the `task_list` tool lets the orchestrator interleave
discovery and task creation: list a crate, add a task per module, dispatch, and
repeat. (Example 11 shows the PlanExecute case, where the three steps are known
up front.)

### Per-module scoping (one worker per module)

Each `analysis_worker` audits exactly **one** module and runs `Isolated`: a fresh
session with no shared mutable state. The worker may burn a dozen turns grepping
and reading narrow line ranges, but only its final JSON findings cross back into
the orchestrator's context. This keeps each worker inside its own context window
and keeps the orchestrator lean across a long, many-module audit. Findings
accumulate in the `memory` tool, which survives compaction — so a long run does
not lose earlier modules' findings.

### Heterogeneous models

The orchestrator and workers run a small **local** model (`gemma4:e4b`); the
advisor runs a **near-frontier cloud** model (`minimax-m3:cloud`). Cheap, fast
local inference does the bulk enumeration and grep-first auditing; the expensive
model is reserved for the hard calls (*is this a real defect? what severity?*),
and only when a worker escalates to it. All four ride the **same** Ollama
endpoint — only the model id differs (`ollama.WithBaseURL`).

### Depth-1 → orchestrator-mediated consult

Subagents cannot spawn subagents (the depth-1 rule, enforced at construction). So
when the `analysis_worker` needs help, it does **not** spawn a helper itself. It
pauses mid-loop with a `RunConsult`, and the `analysis_worker`'s `SubagentTool` —
sitting at the orchestrator boundary — **mediates** the consult: it routes by
`kind` to the right handler (which is the *orchestrator's* direct child,
depth-1), runs it via `ResumeConsult`, and resumes the worker with the answer.
The orchestrator's model is never involved in a consult; mediation is
deterministic.

## The escalation ladder in practice (consumer of #114)

The analysis worker audits on its general knowledge, but escalates mid-loop via
two custom tools, each lowering to `sporecore.NewToolOutputConsult(ConsultRequest{ Kind, … })`:

| Tool | `kind` | Routed to | Budget | Overflow |
| --- | --- | --- | --- | --- |
| `research_best_practices` | `research` | research_worker (web_search) | **5** | `SoftFail` |
| `consult_advisor` | `advice` | advisor (cloud model) | **3** | `EscalateToHuman` |

Both handlers are installed on the `analysis_worker` `SubagentTool` via
`WithConsultHandlers` (not on the builder — mediation fires at the tool).

1. **Best-practices research** (`research`, budget 5). Looking up an idiom is
   *normal*, not distress, so it never reaches the human. On the 6th consult the
   `SoftFail` policy resumes the worker with `ConsultRespBudgetExhausted` and it
   finishes on general knowledge.
2. **Advisor** (`advice`, budget 3). For *is-this-real / how-bad-is-this*
   questions. On the 4th consult, `EscalateToHuman` converts the over-budget
   consult into `RunWaitingForHuman`, which bubbles up to the REPL with three
   choices:
   - **[1] +1 advisor turn** — re-run the advisor once and feed its answer back
     as guidance;
   - **[2] abort subagent & chat** — halt the stuck worker, return to the
     orchestrator;
   - **[3] free-form** — you answer the worker's question yourself (you play
     advisor for one turn).

### Honest note on the human-escalation mechanics

The worker's paused consult lives inside the orchestrator's `PausedState` child
state, and the harness does **not yet wire a child-consult resume through the
parent** (it lands with the #5/#115 follow-up). So in this example every human
choice resumes the **orchestrator** with your decision injected as guidance, and
the specific module's in-flight worker audit is dropped. **"+1 advisor turn"**
re-runs the advisor handler **host-side** and injects its answer as that guidance
— the closest we can get to a "budget bump" without a core primitive (there is no
harness budget-bump). This is a faithful demonstration of the *ladder shape*; the
lossless child-resume is a tracked follow-up, not something this example hacks
into core.

## Skills: loaded at runtime, architect-side

The `analysis_worker` doesn't hard-code the audit procedure — it **loads** an
`audit` skill at runtime. This is the suite's first end-to-end skill-loading
example, and it is wired **architect-side, with zero core-harness change**.

### Why architect-side (the teaching note)

The skill-loading chain the issue originally envisioned —
`GuideRegistry → pending_skill_injections` (the rich
`contextmgr.StandardContextManager`'s Block-3 segment) — **isn't live-wired in
the harness loop yet**. The live loop assembles each turn via
`StandardCompactionAdapter.Assemble`, a pass-through of `session.Messages`; the
rich `Assemble` that would inject `pending_skill_injections` (plus chunks and
merged memory) is bypassed pending the deferred #7 ContextManager migration
(cf. Known Deviation #8). So today a skill can reach the model only as a
tool-result message or via a **custom context manager**.

This example takes the custom-context-manager route and documents it honestly,
because it is exactly the pattern **#115** will absorb into the library
(`HarnessBuilder.GuideRegistry(..)` + a standard `load_skill` tool + sticky
injection). The companion productionization — a scope-aware
`FileSystemGuideRegistry`/`CompositeGuideRegistry` over `.spore/skills/` (the
direct sibling of #88's `FileSystemChunkProvider`) — is also tracked in #115;
this example inlines the filesystem scan.

### How it works here (three pieces, in `skills.go` + `tools.go`)

1. **`SkillCatalog`** (`skills.go`) scans `.spore/skills/{name}/SKILL.md`
   (project, relative to cwd) then `~/.spore/skills/{name}/SKILL.md` (user),
   parses YAML frontmatter `{name, description}` + markdown body, and registers
   each as a `GuideTypeSkill` guide in a `guideregistry.StandardGuideRegistry`. It
   also keeps a **manifest side-list** of `(name, description)` — the example owns
   the model-facing manifest text. The bundled `skills/audit/SKILL.md` is embedded
   with `//go:embed` (the idiomatic equivalent of Rust's `include_str!`) and
   always registered, so the example is self-contained even with an empty
   `.spore/skills/`; on first run it also **seeds** `.spore/skills/audit/SKILL.md`
   from that bundled copy so you can see the filesystem-registry shape.
2. **`load_skill` tool** (`tools.go`, closes over the registry): confirms the
   named skill exists, then appends its id to `run_store["active_skills"]`
   (deduped) via the tool context. No new `ToolOutput` variant — this is the
   storage-backed flavor (#115 "flavor B").
3. **`SkillInjectingContextManager`** (`skills.go`) **embeds** the standard
   compaction adapter and, in `Assemble` only, prepends — **ephemerally**, never
   into `session.Messages` — (a) a **manifest** of every skill
   (`name: description`, progressive disclosure), and (b) the **full body** of
   every id in `run_store["active_skills"]`. Every other method
   (`AppendToolResult`, `AppendUserMessage`, `ShouldCompact`, and the optional
   `CompactingContextManager` / `AssistantMessageAppender` / `TokenBudgetReader`
   seams the harness type-asserts for) is **inherited verbatim** from the embedded
   adapter via Go struct embedding.

The net effect: the manifest is present **every turn** (the model can choose to
load a skill by description); a loaded skill's body is **re-injected every turn**
until the session is cleared. Because the active set lives in `run_store` (not the
message history), it is **compaction-proof** — a long audit cannot "forget" the
loaded procedure. A unit test in `skills_test.go`
(`TestManifestAlwaysInjectedBodiesOnlyWhenActive`) asserts exactly this: manifest
always present, body present only after `load_skill`.

### Go wiring notes (the documented divergences)

- **The `ContextManager` interface is small.** Go's `sporecore.ContextManager` is
  four methods (`Assemble`, `AppendToolResult`, `AppendUserMessage`,
  `ShouldCompact`); the heavier compaction / assistant-append / token-budget
  surfaces are **optional seams the loop type-asserts for** (Go's equivalent of
  Rust's default-bodied trait methods). Embedding `*StandardCompactionAdapter`
  satisfies all of them at once, so this example overrides only `Assemble` and
  inherits the rest — the cleanest "delegate everything else verbatim".
- **Consumer-side seams, set on the builder, not the config struct.** The
  `observability.ConversationalBuilder` exposes `.ContextManager(..)`,
  `.Storage(runStore, memStore)`, and `.Sandbox(..)` fluent setters; the shared
  `storage.NewInMemoryStorageProvider()`'s `.Run()` store satisfies the
  consumer-side `ToolRunStore` interface structurally (no import cycle on the
  storage package).

## The `audit` skill

`skills/audit/SKILL.md` is a **terse procedure** with grep-first / read-narrow
discipline (grep the module for risk patterns → `read_file` only the narrow line
ranges → consult when unsure) plus a **hard output schema**: the worker's final
answer must be a JSON array of `{ file, line, severity, description }`. That hard
schema is what lets the orchestrator reliably parse per-module findings into a
top-5. The discipline keeps each module audit inside the worker's context window.

## REPL flow

1. Opens with the **pre-filled audit prompt** (from the issue). Press **enter**
   to accept it verbatim, or type your own. (Input is read with a
   `bufio.Scanner`, the suite's REPL convention.)
2. The orchestrator runs the audit, streaming agent boundaries
   (`┌─ orchestrator → analysis_worker … └─`, mirroring example 11).
3. On success it writes `workspace/findings.md`, prints the **top 5**, and asks
   **y/N** to file them as GitHub issues. On `y`, it drives `gh issue create` per
   finding via `bash_command` (no `gh` skill — the model drives it from general
   knowledge).

The audit is **READ-ONLY**: the only writes are `workspace/findings.md` and the
(approved) issues.

## Configuration

| Flag / env | Default | Purpose |
| --- | --- | --- |
| `--model` / `SPORE_OLLAMA_MODEL` | `gemma4:e4b` | orchestrator + workers |
| `--advisor-model` / `SPORE_ADVISOR_MODEL` | `minimax-m3:cloud` | the advisor (cloud) |
| `--search-url` / `SPORE_WEB_SEARCH_ENDPOINT` | _(required)_ `http://localhost:8888/search?format=json` | research worker's `web_search` |
| `SPORE_OLLAMA_BASE_URL` | `http://localhost:11434` | shared Ollama endpoint |

**Run gemma4:e4b or better.** Small models (e.g. llama3.2 3B) *narrate* tool use
instead of emitting tool calls, which would masquerade as a different failure
than the consult ladder is meant to handle.

## Prerequisites

- Ollama running, with `gemma4:e4b` pulled and an Ollama **cloud** account for
  the default advisor model (or override `--advisor-model` to a local model).
- A local **SearXNG** with the JSON format enabled at the default search URL.
- `gh` CLI authenticated (only for the opt-in issue-filing step).

## Run it

```sh
ollama serve &
ollama pull gemma4:e4b
export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
go run .
# press enter at the prompt to accept the default audit
```

## Where cordyceps goes next (out of scope here)

- **First-class skill loading in the harness** — #115 absorbs the architect-side
  pattern above (manifest + `load_skill` + sticky injection) into `HarnessBuilder`,
  plus a scope-aware `FileSystemGuideRegistry`/`CompositeGuideRegistry` (sibling
  of #88) and retiring the dead Block-3 path (reconciling with #7).
- **Lossless child-consult resume** through the parent (the human-escalation
  mechanics note above) — the #5/#115 follow-up.
- **Observability / Phoenix** (#92) — spans emit, but no Phoenix wiring here.
- **Cross-run persistent memory** (#89) — this example does within-run memory only.
- **`web_search` multi-provider hardening** (#108/#110) — one SearXNG backend is
  enough here.
- **Conversational human escalation** (park the worker while the orchestrator
  chats) — the human path here is single-instruction.
- **Skill creation** — this example only *consumes* a skill; writing a new one is
  a real cordyceps-project feature.
