# spore-core consumer-friction plan (unified)

**Version:** v2.0 — **ADOPTED baseline.** All three teams approved (spore-core · cordyceps · looper); Q1–Q8 resolved.
**Status:** Adopted. Supersedes the cross-team `upstream-issues.md` tracker (which does not live in this repo).

> **Pinned SHA: `d14341f`** (`feat(rust): SelfVerifying eval-phase dedicated reviewer + read-only toolset (#151)`).
> All line references below were re-verified against this commit during Phase 0. The prior pin `18cda89` and
> the dirty-tree references in earlier revisions are superseded. Re-verify before each PR.

---

## Phase 0 — COMPLETE (2026-06-23)

The three reconciliation tasks are done. Two of them corrected the plan; record the corrections here.

- **D1 — canonical middleware surface: the rich `middleware.rs` chain.** Verified empirically: `fixtures/middleware/checklist_basic.json` is replayed by a test that calls the **rich chain's** `fire_before_completion(...)` (`middleware.rs:1370`), and all four languages (rust/ts/python/go) carry a `middleware*fixture*` replay test against that surface. The wired `harness.rs:5184` `MiddlewareChain::fire(hook, session)` is a **self-described stub** with **no fixture coverage of its own**. Decision: **wire the loop to the rich `middleware.rs` chain and delete the stub**, additively (no-middleware path stays byte-identical). This collapses **SC-9, SC-11, and Q5** into one change — the rich chain already carries `fire_before_tool(calls: &mut …)` and `fire_after_tool(results: &mut …)` plus priority-ordered fan-out. Re-file the cross-team ISSUE 1 & 2 against the `middleware.rs` surface.

- **D2 — `enforce_output_schemas` default: corrected.** There is **no `impl Default for HarnessConfig`**; every construction site builds it field-by-field. The **production default lives solely in `HarnessBuilder::new` (`harness.rs:6429`)**, re-emitted in `build_config` (`harness.rs:7093`). The `hooks.rs:1763` site cited earlier is a **test-only `HarnessConfig` builder** (MockAgent/ScriptedToolRegistry/AllowAllSandbox), not a `Default` impl. So the drift risk is **builder ↔ test-helper** (a test config can silently misrepresent the shipped default), not two production defaults. **Fix (accepted): add a single `impl Default for HarnessConfig`** that `new()` and the test helpers both derive from (`..HarnessConfig::default()`), giving one source of truth. Small, behaviour-neutral; land it as a standalone or with Phase 1.

- **D3 — re-pin: DONE.** The #151 SelfVerifying eval-phase reviewer slice (uncommitted on the working tree) was test-gated (`cargo test -p spore-core` → 1255 passed / 0 failed) and committed as **`d14341f`**, which is the pin. All #151-adjacent references (SC-29, SC-30, SC-BUG-1) now resolve against a real commit.

---

## 1. The core diagnosis

spore-core's defaults are tuned to keep `*_fixture_replay.rs` byte-identical, not to serve a new consumer. Nearly every one of `HarnessConfig`'s ~30 fields was added by a separate issue and each defaults to "preserves today's behaviour byte-for-byte." That is the right instinct for a maintainer guarding a replay suite. It is the wrong starting point for cordyceps and looper, which have no fixtures to protect — so onboarding a real consumer means discovering and flipping a dozen OFF-by-default gates one at a time.

The result: cordyceps carries ~700 lines of `main.rs` that is mostly workarounds, and looper carries 250+ lines of pure adapter/forwarding/ceremony boilerplate. Neither team wrote bad code. They wrote apologies for defaults that assume there are no consumers. The three reviews describe one problem, not three.

## 2. Two operating principles

**A. Convergence is the priority signal.** Where the same wall shows up in multiple independent consumers, it is one wrong decision, not N bugs. Items are ranked first by how many teams hit them, then by line-count leverage.

**B. "Documented = wired" is a release gate.** spore-core repeatedly advertises seams it does not wire (`HookContext::PostToolUse` documented "wired" but never constructed; `CONVENTIONS.md` claims `Arc<dyn ModelInterface>` injection the trait can't support; the "exactly FIVE maps" doc that documents a sixth). **Every item ships with an acceptance check that proves the documented behaviour fires from a consumer.** No item is "done" until a consumer-level test confirms it.

## 3. Decisions (all resolved)

- **Q1 / SC-7 → no re-baseline.** `fixtures/` is a **4-language byte-parity contract** (114 files replayed by Rust/TS/Py/Go). Friendliness ships via additive setters/variants + presets (SC-8), not default flips. **Phase 2 is un-gated** and ships alongside Phase 1.
- **Q2 / D1 → the rich `middleware.rs` chain is canonical.** Wire the loop to it, delete the `harness.rs:5184` stub. Collapses SC-9 + SC-11 + Q5. (Phase 0 confirmed: the rich chain is the only fixtured surface.)
- **Q3 / SC-1 → (b) auto-synthesize.** (a) re-arms the ceremony the instant enforcement flips on; (b) removes it unconditionally. Same shape as SC-30 (auto-derive) — one coherent "no startup ceremony" story.
- **Q4 / SC-5 → `EscalationMode::AutoContinue { max_grants, steps_per_grant, on_grant: Option<Callback> }`.** A serializable enum variant over a trait object (which would force a registry handle — the SC-1 ceremony again).
- **Q5 / SC-11 → moot, resolved by Q2.** The rich chain's priority fan-out is the richer-context option; no separate `DenyOverridesChain`.
- **Q6 / D2 → single `impl Default for HarnessConfig`.** See Phase 0 D2 (corrected).
- **Q7 / SC-26 → structural `ContextSources` threading** through the live-loop `assemble` (the #115 design). Not a system-prompt prepend, not permanent User-message injection.
- **Q8 / verified-review roadmap → moot for Rust.** The #151 reviewer slice is committed at `d14341f` (SC-29 always-drop, SC-30 mechanism). Remaining work is the TS/Py/Go port of #151.

---

## 4. Phased plan

### Phase 1 — No fixture risk (do regardless of the re-baseline decision)

**SC-1 — Empty-schema ceremony. Hit by ALL THREE teams.**
- *Symptom:* every consumer registers a do-nothing schema to pass startup. cordyceps `PLAN_SCHEMA_KEY` (`main.rs:137-143,271,416`); looper two `*_SCHEMA_KEY` consts + `SchemaRef` stamps (`harness.rs:150-154`, `mode.rs:124,161`).
- *Root cause:* `check_structured_slot` (`execution_registry.rs:311`) rejects a bare ReAct in a plan/worker/propose slot unless it declares `output = Some(SchemaRef)` — unconditionally — but `enforce_output_schemas` defaults `false`, so the schema is never used.
- *Fix — DECIDED (b) auto-synthesize:* when enforcement is off, auto-synthesize an empty schema for structured slots. Same class as SC-30; bundle them.
- *Fixture impact:* none (schema never enforced when off).
- *Acceptance:* a PlanExecute harness with `enforce_output_schemas = false` and a bare ReAct plan leaf, **no schema registered**, passes startup. cordyceps deletes `PLAN_SCHEMA_KEY`; looper deletes both `*_SCHEMA_KEY` consts.

**SC-2 — `ModelInterface` object-safety. Highest line-count win for looper.**
- *Symptom:* looper can't hold `Arc<dyn ModelInterface>`, so it monomorphizes via an `AgentModel` enum (`config.rs:151-155`) + three dispatch matches.
- *Root cause:* `ModelInterface` uses bare `async fn` (RPITIT, dyn-incompatible — `model.rs:299`); `conversational<M>` takes by value (`harness.rs:6477`). `CONVENTIONS.md:29` claims `Arc<dyn ModelInterface>` injection the trait can't support.
- *Fix:* make the trait object-safe (boxed-future methods + blanket impl, matching the crate's existing `BoxFut` idiom — not `#[async_trait]`) and add `conversational(Arc<dyn ModelInterface>)`.
- *Fixture impact:* none (additive constructor). **But not small** — the signature change ripples to every impl (anthropic/openai/ollama/replay/recording/adaptive), the eval crate, and test doubles. Phase 1 (fixture-safe), not a "quick" PR.
- *Acceptance:* looper holds `Arc<dyn ModelInterface>`, deletes the enum + three arms (~40–50 lines), re-mints the model once; `CONVENTIONS.md:29` becomes true.

**SC-3 — Typed model errors. Hit by cordyceps.**
- *Symptom:* cordyceps substring-matches error text to tell a retryable stream drop from a permanent failure (`main.rs:1371-1376`).
- *Root cause:* Ollama folds transport drops, stream-decode failures, and deterministic errors all into `ProviderError { code: 0, message }`.
- *Fix:* add a typed `ModelError::Transport` / `StreamInterrupted` variant or a `retryable()` method.
- *Acceptance:* cordyceps replaces all substring matches with a typed check; a test forces a stream interruption and asserts the variant.

### Phase 2 — Provider/model knobs (additive; un-gated by Q1, ships alongside Phase 1)

> **LANDED (Rust) at `f1c0beb`** (2026-06-23). All four items below are implemented + tested in the Rust reference; TS/Py/Go parity tracked in **#155**. Per-item notes inline.

**SC-4 — Compaction window silently collapses to 8K. Hit by cordyceps + any non-tiny model. — LANDED `f1c0beb`.**
- *Done (Rust):* Ollama `with_context_window(n)` sets BOTH `num_ctx` (what the model loads) AND the reported `provider().context_window`; the compaction budget auto-derives via the existing #141 `resolve_context_length` chain (no second setter). Anthropic/OpenAI `with_context_window(n)` set the reported window only (no `num_ctx`). Stale "80% of 200K" builder doc corrected.
- *Root cause:* `DEFAULT_CONTEXT_LENGTH = 8_000` (`context.rs:227`); `resolve_context_length` (`context.rs:613`) → config → `provider().context_window` → 8000; Ollama prefix table (`ollama.rs:232`: `gemma*`→8192, unknown→0), best-effort `/api/show`. Stale doc `harness.rs:6541` ("80% of 200K").
- *Fix:* one `.context_window(n)` builder setter fanning out to both the compaction budget and the model's `num_ctx`; fix the stale doc. Also closes looper's `num_ctx` drift.
- *Fixture impact:* the **setter** is additive; only flipping the raw 8K default touches `fixtures/compaction_window`/`compaction_loop` — and per Q1 we don't flip it (presets carry the friendly value).
- *Acceptance:* `.context_window(256_000)` once → both compaction budget and `num_ctx` reflect it, model built once, 200K conversation doesn't compact prematurely. cordyceps deletes `main.rs:144-260`.

**SC-5 — No "autonomous but capped" escalation mode. Hit by cordyceps + looper. — LANDED `f1c0beb`.**
- *Done (Rust):* `EscalationMode::AutoContinue { max_grants, steps_per_grant, on_grant: Option<Callback> }` consulted at every `Escalate` site (5 nested via `ExecutionContext::try_auto_continue` + `BudgetContext::grant_auto_continue`; the top-level bare-leaf `drive_strategy_with_resume_seed` site via a loop-local counter). Auto-grants in-process up to `max_grants`, fires `on_grant` per grant, then falls through to the `Autonomous` terminal. `on_grant` (`Arc<dyn Fn>`, serde-skipped) drops `Copy`/`Eq` from `EscalationMode` (hand-rolled `Debug`); never serialized in fixtures.
- *Symptom:* keep-working-to-completion-but-cap-at-N exists nowhere. cordyceps hand-rolls `drive()` (`main.rs:567-609`); looper hand-rolls it in `governor.rs:567-590` (`MAX_AUTO_CONTINUES = 10`).
- *Root cause:* `EscalationMode` is binary (`execution_registry.rs:77`): `SurfaceToHuman` | `Autonomous` (propagate up = give up). `ContinueWithBudget`/`resume` machinery exists but isn't wired as a policy.
- *Fix — DECIDED (Q4):* `EscalationMode::AutoContinue { max_grants, steps_per_grant, on_grant: Option<Callback> }`. Default stays `SurfaceToHuman` (no fixture impact).
- *Acceptance:* `AutoContinue { max_grants: 5, .. }` continues without consumer loop code, firing `on_grant` per grant. cordyceps deletes `drive()`; looper deletes the governor arm, keeps the callback.

**SC-6 — Per-provider window tables have no override. Enables SC-4. — LANDED `f1c0beb`.**
- *Done (Rust):* per-provider `context_window_override: Option<u32>` + `with_context_window(n)` setter (a single builder value, not a `HashMap<prefix>` — each interface wraps one model id). `provider().context_window` prefers the override; built-in table stays default (no fixture impact). Ollama precedence: override > `/api/show` discovery > static.
- *Root cause:* hard-coded match arms (`anthropic.rs:132`, `openai.rs:116`, `ollama.rs:232`), unknown → 0 → 8K fallback.

**SC-27 — OpenAI compat is id-heuristic-only; no `with_compat`. Hit by looper (Gap A). — LANDED `f1c0beb`.**
- *Done (Rust):* `OpenAICompat { reasoning_model, developer_role, supports_reasoning_effort }` + `with_compat(..)`, OR'd OVER the id heuristic. Sets `max_completion_tokens` + drops sampling params (reasoning_model), routes system→`developer` role (developer_role), and emits a `reasoning_effort` field `low|medium|high|max` (supports_reasoning_effort, gated on reasoning-treated). Default compat keeps recognized o-series byte-identical.
- *Root cause:* `is_reasoning_model` is a hard-coded `o1/o3/o4` prefix match (`openai.rs:129`); no vehicle to declare developer-role / reasoning-effort for an unrecognized (e.g. local) model. looper's parsed `Compat` (`config.rs:104-114,398`) is dead plumbing.
- *Acceptance:* `with_compat { supports_reasoning_effort: true, .. }` on an unrecognized model → request carries reasoning-effort/developer-role; looper's `Compat` becomes live.

**SC-7 — Fixture re-baseline. RESOLVED (Q1): no re-baseline.** Friendly behaviour via additive setters + presets (SC-8), not default flips. The 8K default (SC-4) is the only candidate flip and is revisited only inside SC-23's deliberate, fixture-affecting deprecation batch.

### Phase 2.5 — Presets (the force multiplier)

**SC-8 — Build presets from the working consumers.** cordyceps **is** `HarnessBuilder::hill_climber(model, evaluator)`; looper **is** `HarnessBuilder::coding_agent(model, workspace)`. Once Phases 1–2 land, extract each preset; each consumer collapses to ~40 lines. These presets are what carry the friendly behaviour (Q1).

### Phase 3 — Wire the dead seams

**Anchor decision (Q2):** adopt the rich `middleware.rs` chain as canonical — wire the loop to it, delete the `harness.rs:5184` stub, additively (no-middleware path byte-identical). This **subsumes SC-9 + SC-11** (mutable `calls`/`results` + priority fan-out are there by construction) and moots Q5. Real loop surgery (replace the one `fire(hook, session)` call site with per-hook `fire_before_*/after_*` calls). Phase 0 confirmed the rich chain is the fixtured surface.

**SC-9 — `AfterTool` can't rewrite a result (collapses into Q2). Hit by cordyceps.** `build_check.rs` returns a landed write as `ToolOutput::error` (`build_check.rs:217`) to force model reaction. `PostToolUse` is documented wired (`hooks.rs:89`) but never constructed; `AfterTool` fires immutable + Halt-only (`harness.rs:9082`). The rich chain's `fire_after_tool(results: &mut …)` gives rewriting by construction. *Acceptance:* an after-tool middleware rewrites a result; cordyceps removes the inversion.

**SC-10 — No per-phase prompt/toolset in PlanExecute. Hit by cordyceps.** Plan + execute run under one `HarnessConfig.system_prompt`; plan format hard-coded in `plan_directive`. *Fix:* per-leaf `system_prompt`/toolset override, or let a leaf's output schema drive a phase-specific directive. *Acceptance:* distinct plan/execute prompts, each phase sees only its own.

**SC-28 — Plan phase forces a JSON `PlanArtifact`; relax to free-text/markdown. Hit by looper (Gap B).**
- *Root cause:* `capture_plan_artifact` parses `PlanArtifact { tasks, rationale }` (`plan.rs:106`); a markdown plan fails the parse. Executor now seeds from the `task_list` tool (`harness.rs:1713`), so JSON isn't the only source.
- *Fix (upstream):* relax the captured prose to free-text/markdown **but keep a structured `tasks: Vec<String>` in the `OnPlanCreated` payload** (sourced from `task_list`).
- *Contract impact (wider than a parser change):* `PlanArtifact` (`hooks.rs:147`) is also the `OnPlanCreated` payload (`hooks.rs:467`) and the `PLAN_EXECUTE_EXTRAS_KEY` stored shape.
- *Hook consumers that must keep working:* looper `plan_tracker` seed (`repl.rs:50-55`); cordyceps `plan_announcer` (`main.rs:1169-1173`, reads `plan.tasks`). The relaxation must stay backward-compatible — a JSON plan must still populate `plan.tasks`.
- *Design note:* `OnTaskAdvance` carries only a `task_index` (flips already-seeded items Done/Active/Pending, `repl.rs:68-78`); panel **texts** come from `plan.tasks`. A free-string artifact with no `tasks` leaves the panel textless.
- *Acceptance:* (1) markdown plan captures without parse failure; (2) a JSON plan still populates `plan.tasks`; (3) stored `PLAN_EXECUTE_EXTRAS_KEY` still deserializes; (4) looper removes only suppression + re-emission code, seed intact.

**SC-11 — `fire` gets only `&SessionState` (collapses into Q2). Hit by looper.** looper hand-rolls `pending_calls` + `risk_level` (`policy.rs:287-366`). The rich chain's `fire_before_tool(calls: &mut …)` + priority fan-out resolve it; **Q5 moot** (no `DenyOverridesChain`). *Acceptance:* looper's `decide` consumes harness-supplied calls + risk; derivations deleted.

**SC-26 — Rich `ContextSources` not wired into the live loop (#115). Surfaced via cordyceps (skills); affects guides/memory/prompt-chunks.** `SkillInjectingContextManager` (`skills.rs:269`) forwards 9 methods + injects skills as User messages (`skills.rs:244`). Root cause: the harness-loop seam `ContextManager::assemble(session, task)` (`harness.rs:5069`) carries **no `ContextSources`**, so rich `assemble(state, &ContextSources)` (`context.rs:538`) is bypassed (#115). *Fix:* wire `ContextSources` through the live-loop `assemble` (structural slots), not a prepend/User-message. *Acceptance:* a consumer registers skills **and** a guide/memory source; all reach the model via the rich path, no wrapper. cordyceps deletes `SkillInjectingContextManager`.

#### Gap D — #151 reviewer slice (committed at `d14341f`)

**SC-29 — Eval phase must not inherit `BeforeTool` approval middleware. DONE (#151).** `run_evaluate_phase` sets `eval_config.middleware = None` (drops HITL middleware to avoid an `AlwaysAsk` deadlock — no human to resume a non-interactive nested review). **Always-drop, not a knob** (recommended; a knob is surface for no benefit). Remaining: TS/Py/Go port. *Acceptance:* under `AlwaysAsk` the eval phase reads without pausing; parity test across languages.

**SC-30 — Read-only eval toolset. Mechanism DONE (#151); convenience remaining. Phase 1.** `SelfVerifyingConfig.eval_agent`/`eval_toolset` exist; eval phase runs on `ReadOnlySandbox` + threads `eval_toolset`; registry validates the handles.
- *Correction:* a `read_only_eval()` that stamps `eval_toolset = Some(ToolsetRef("readonly"))` re-triggers the SC-1 ceremony (`validate()` requires the handle registered). The "~5-line config helper" billing is wrong.
- *Fix — (b) auto-derive:* when `eval_toolset` is empty, the harness auto-derives a read-only view of the global catalogue for the eval phase (no registration). `StandardTools::readonly_set()` exists (`tools/catalogue.rs:230`). Adds: not advertising write tools (avoids error round-trips) + blocking non-FS side-effecting tools (web/MCP) the sandbox doesn't gate.
- *Looper inertness:* SC-30 lands green but was **inert for looper until SC-BUG-1** — the reviewer (and looper's gates-based `/verify`) run in the eval frame, which the HITL-resume discarded. **SC-BUG-1 is the real unlock and is now LANDED (`8d1d679`)**, so a HITL-resumed run re-enters the SelfVerifying frame and reaches the eval-phase reviewer.
- *Acceptance:* SelfVerifying with no `eval_toolset` → eval phase exposes only the read-only catalogue, **no registration required at startup**.

### Phase 4 — Sandbox / exec knobs

- **SC-12 — exec-hardening knob (looper, ~100 lines).** `WorkspaceScopedSandbox::execute_command` inherits stdin, no timeout, no non-interactive env, no `kill_on_drop`. *Fix:* `ExecConfig { default_timeout, close_stdin, non_interactive_env, kill_on_drop }` on `WorkspaceConfig`. looper deletes `DefaultSandbox`.
- **SC-13 — read-everywhere/write-scoped (looper, ~100 lines).** `WorkspaceConfig` gates reads/writes identically (`sandbox.rs:4-6`). *Fix:* `write_root: Option<PathBuf>` distinct from the read root. looper deletes `PlanSandbox`.
- **SC-14 — hard-coded `git reset --hard HEAD` revert (cordyceps).** HillClimbing revert (`harness.rs:9571-9576`) assumes a git repo, only the working tree. *Fix:* pluggable VCS provider; at minimum document the coupling.
- **SC-15 — sandbox returns `Ok { exit_code: -1 }` on spawn failure (cordyceps).** *Fix:* return `Err`. **Not purely additive** — flipping `-1`→`Err` changes control flow for callers branching on `-1`; check the replay suite before landing.
- **SC-16 — reasoning silently no-ops on non-thinking models (cordyceps).** `think` dropped with no signal (`ollama.rs:53,142`). *Fix:* surface a typed/warning signal when reasoning is requested but unsupported.

### Phase 5 — Grouping & cleanup (lowest priority)

- **SC-17** — group `HarnessConfig`'s ~30 fields into sub-configs (`OutputSchemaPolicy`, `RepairPolicy`, `RalphPolicy`, `LimitsConfig`).
- **SC-18** — one `TruncationPolicy` (64 KiB tool-output + 2K/2K `tools/mod.rs:56`; 8 KiB exec stderr `tools/exec.rs:352`; 32 KiB offload `context.rs:600`; 2000-byte composite reason `verifier.rs:309`; 8 KiB content capture `observability.rs:215`). Also: optional structured-results path for `TestPassRateEvaluator` (avoid regex-scrape of cargo stdout).
- **SC-19** — `RetryConfig` (anthropic 3/500ms→30s `anthropic.rs:439`; Ollama 30s connect / 300s read).
- **SC-20** — expose `reasoning_budget` 2048, `max_tokens` 4096 (`anthropic.rs:314`), guide-conflict Jaccard 0.6 (`guide_registry.rs:298`), step-ledger cap 20 (`tasklist.rs:174`).
- **SC-21** — `ExecutionRegistry` handle indirection taxes the in-process single-agent case; doc self-contradicts ("FIVE maps" `:47` vs the sixth `:117`). Add a typed convenience layer / let `LoopStrategy` carry `Arc`s for the non-resumable case; fix the doc.
- **SC-22** — `StrategyExecutor` is a 25-method grab-bag; `StrategyRef::Custom` leaks the whole surface. Document the real custom-strategy contract; consider a narrower trait.
- **SC-23** — retire migration cruft (`#[deprecated]` re-exports; the `enforce_output_schemas` "MIGRATION GATE" field). Tied to SC-7.
- **SC-24** — split `harness.rs` (24K lines) along seams (`strategies/`, `budget`, `builder`, `escalation`, streaming). Mechanical.
- **SC-25 — CLOSED (no upstream bug).** `walk_strategy` is strictly per-variant; `check_metric_evaluator` fires only in the HillClimbing arm (`execution_registry.rs:300`). spore-core does not validate inactive-strategy handles. Consumer-side: gate registration on the active `--strategy`.

## 5. The one correctness bug

**SC-BUG-1 — HITL approval/clarification resume runs the bare leaf, skipping SelfVerifying/PlanExecute frames. HIGH. looper. LANDED (Rust) `8d1d679`; parity #156.** `resume_inner`'s Allow/Deny/Answer tails end at `self.run_react(task, …)` on the paused leaf (`harness.rs:11567/11390`); only the budget path threads through `drive_strategy_with_resume_seed`. Under `AlwaysAsk`, the verify loop silently degrades to a plain executor. **#151 raises the stakes:** a HITL-approved resume never re-enters the SelfVerifying frame, so it never reaches the now-functional reviewer (SC-29's middleware drop does NOT fix this — different path). **Hard prerequisite for an AFK verifying agent; land it with the #151 work.** *Acceptance:* under `AlwaysAsk`, a resumed Allow/Deny/Answer re-enters the SelfVerifying frame and reaches the eval-phase reviewer.

> **LANDED `8d1d679`.** Fixed by mirroring the #131 consult-resume pattern, three parts: (1) `ExecutionContext::finish` now rewrites a propagating `WaitingForHuman`'s `state.task` to the combinator's composed task on the way up — exactly as it already does for `Consult` — so the top-level pause carries the full tree; (2) `resume_inner`'s clarification tail AND the final Allow/Deny/Answer/Reject tail route a composed task through `drive_strategy_with_resume_seed` (fresh top session + the mutated worker session as the resume seed), keeping a bare ReAct leaf on the original `run_react` path; (3) `SelfVerifyingConfig::run` consumes `scratch.resume_seed` as its first build iteration's session so the resumed worker continues and the evaluate phase + verifier run. PlanExecute already consumes the seed when it is the outer frame (nested SelfVerifying sees `None`, unchanged) and is re-driven by the same call the #131 consult path exercises. Additive — a fresh run leaves the seed `None` (byte-identical); `WaitingForHuman`/`EscalationMode` are not serialized, so no wire/fixture impact. Tests: `self_verifying_hitl_{resume,deny,clarification}_reenters_eval_frame` (verifier consulted 0× before the fix, ≥1× after). Full suite green (1235 lib pass / 0 fail), clippy clean.

## 6. agent-repl-kit (looper-only library)

- **ARK-1** — `SLASH_COMMANDS` hardcoded const; add `with_slash_commands(...)` (the menu currently lies: advertises unimplemented, omits routed).
- **ARK-2** — mascot `Success`/`Error` poses are dead code (`app.rs:248` forces Idle); add a terminal-state hook or `with_mascot_auto(false)`.
- **ARK-3** — no `ToolKind::Generic`; non-shell tools render as fake `Bash` (`repl.rs:257`). Add `Generic { label, summary }`.
- **ARK-4** — re-export `style::fg`/`fg_bold` (`mascot.rs:17-25` reimplements them).

## 7. Looper-local cleanups

- **LOC-1** — wire parsed `Compat` into the model once SC-27's `with_compat` exists (keep the struct; removal only if SC-27 rejected).
- **LOC-2** — with SC-28 preserving `tasks`, delete only the `FinalResponse` suppression (`repl.rs:167-173`, `main.rs:210-212`) + `governor::finish` re-emission (`governor.rs:438-441`); seed survives. Lands with SC-28.
- **LOC-3** — unify three arg-extractors into one `str_arg` helper. Minor.
- **LOC-4** — fold away the `--verify` headless flag (`main.rs:37`).
- **LOC-5** — decide JSON-vs-TOML config split.
- **LOC-6** — confirm `.looper/.looper/policy.json` is a test artifact, not a live path-join.
- Hardening (cordyceps): pull SC-3 retry substrings into named constants; add a "forward all of these" comment over the `SkillInjectingContextManager` 9-method delegation.

## 8. Mapping to the original three reviews

| Unified ID | spore-core | cordyceps | looper |
|---|---|---|---|
| SC-1 | Tier 1 #3 | Tier 1 #4 | #5 |
| SC-2 | — | — | #1 |
| SC-3 | Tier 2 | Tier 1 #3 | — |
| SC-4 | Tier 1 #1 | compaction comment | num_ctx drift |
| SC-5 | Tier 1 #2 | drive loop | governor.rs:567-590 |
| SC-6 | Tier 2 | — | — |
| SC-8 | "do first" #4 | — | — |
| SC-9 | — | Tier 1 #2 | — |
| SC-10 | — | Tier 1 #1 | partial (SC-28) |
| SC-11 | — | — | #1/#2 (D1) |
| SC-12 | — | — | #2 |
| SC-13 | — | — | #3 |
| SC-14 | — | Tier 2 #7 | — |
| SC-15 | — | Tier 2 #8 | — |
| SC-16 | Tier 2 | Tier 2 #9 | — |
| SC-17–25 | Tier 2/3 | — | — |
| SC-26 | reframed | Tier 2 #6 | — |
| SC-27 | confirmed | — | Gap A |
| SC-28 | contract caveat | `plan_announcer` consumer | Gap B |
| SC-29 | DONE #151 | — | Gap D-i |
| SC-30 | mechanism DONE | — | Gap D-ii |
| SC-BUG-1 | — | — | #4 |
| ARK-1–4 | — | — | #8/#9 |
| LOC-1/2 | — | — | #6/#7 |

## 9. Execution order

1. **Phase 0** (DONE) — D1/D2 reconciled, #151 committed at `d14341f` (the pin).
2. **Phase 1** — **SC-1 + SC-30 bundled** (same "handle must resolve or it's ceremony" class: auto-synthesize the schema + auto-derive the read-only eval catalogue, one PR), **SC-2** (fixture-safe but not quick), **SC-3**. Plus the D2 single-`Default` fix.
3. **SC-BUG-1** — with the #151 work; it's the resume path the reviewer depends on (SC-30 inert for looper until then). **LANDED (Rust) `8d1d679`; parity #156.**
4. **Phase 2** (SC-4/5/6/27) — additive, ships alongside Phase 1 (un-gated by Q1). **LANDED (Rust) `f1c0beb`; parity #155.**
5. **Phase 2.5** (SC-8) — presets, once Phase 1 lands.
6. **Phase 3** — Q2 canonical-chain adoption (subsumes SC-9 + SC-11) + SC-10/26/28. **Phase 4/5** as capacity allows.
7. **agent-repl-kit** (ARK) + **looper-local** (LOC) — alongside, mostly independent.
8. **SC-29** — confirm always-drop, port to TS/Py/Go (#151).

Cross-language: land Rust first (+ live example), then file the TS/Py/Go parity issue — do not port all four at once.

## 10. Sign-off

| Team | Decision | Notes |
|---|---|---|
| spore-core | **Approve** | Q1/Q2 decided; SC-30 auto-derive rescope; volunteered the bundled SC-1+SC-30 PR. |
| cordyceps | **Approve** | SC-26 broadening + SC-25 closure confirmed; `plan_announcer` folded into SC-28. |
| looper | **Approve** | SC-29/30 rescope + inertness note; SC-27/28 match intent; Q3→(b), Q5→richer `fire`. |

**Adopted — unanimous.** Phase 0 complete (pin `d14341f`). Next: the bundled SC-1 + SC-30 PR against the pin.
