# PROJECT STATE

_2026-06-24 — **SC-26 / #115 guides + skills LANDED (local `main`, NOT pushed) — structural `ContextSources` into the live loop, in 4 slices.** The Phase 3 "structural one": skills/guides now reach the model through the live-loop `assemble` seam (a leading System block) instead of the example-side `SkillInjectingContextManager` shim's ad-hoc User-message injection — the deferred #7/#115 migration, done for the guide + skill halves (memory deferred, see below). **Slice 1 (`a061b6d`):** thread a `sources: &ContextSources` param through the harness `ContextManager::assemble` (signature ripple to all 6 impls + the one loop call site); the harness builds a per-turn `ContextSources`; every impl ignores it for now → byte-identical (1242 lib pass, unchanged). Added `Default` to `ComposedPrompt`/`ContextSources`. **Slice 2 (`18ed309`, #9):** `HarnessConfig.guides: Vec<Guide>` + `HarnessBuilder::guide/guides`; `build_context_sources` populates `sources.guides`; `StandardCompactionAdapter::assemble` renders composed-prompt + guides + memory into a LEADING System block (new `render_context_block`); the loop's system-prompt handling now MERGES the configured prompt INTO that block (system prompt first, `starts_with` guard) instead of skipping when a System message exists — empty sources → no block → loop inserts the system prompt exactly as before (byte-identical; all fixture-replay green). **Slice 3 (`00b6106`):** new `crate::skills` module — `SkillEntry` + SKILL.md parsing + filesystem discovery (`.spore/skills` + `~/.spore/skills` + caller dirs) + `SkillCatalog` (entries + shared sticky active-set); `active_guides()` returns the manifest (tier 1) + each active body (tier 2) as `Guide`s; `load_skill_tool()` builds the `load_skill` `StandardTool` sharing the active set; `HarnessConfig.skills: Option<Arc<SkillCatalog>>` + `HarnessBuilder::skills(catalog)` (registers catalog + tool); `build_context_sources` appends `active_guides()` so loading a skill makes its body sticky in the System block next turn. **Slice 4 (`2a9b62b`):** migrated `12-cordyceps` onto the harness-native skills and **DELETED `examples/.../src/skills.rs` (516 lines) incl. the `SkillInjectingContextManager` shim** — `main.rs` now does `SkillCatalog::discover([bundled], ws)` + `.skills(catalog)`, dropping the second model handle + base adapter + the `.context_manager(..)` override (skills ride the preset's `StandardCompactionAdapter` via the seam); the REPL shares the `Arc<SkillCatalog>` for `/skills` / `/<name>` (`activate`) / clear-on-reset (`clear_active`). **Live-verified against Ollama `qwen3-coder-next:cloud`:** given "review calc.py", the agent called `load_skill("deep-code-review")` on turn 4, the procedure went sticky, and it produced the skill's per-file findings report — with NO shim. **Additive throughout** — every new branch guards on non-empty sources, so the no-source path is byte-identical (no fixture/wire change; the only fixtured middleware/observability surfaces were untouched). Full suite green (**1249 lib pass / 0 fail**, integration + 9 doctests; spore-eval 55), clippy clean (2 pre-existing ollama warnings). **Memory DEFERRED** — wiring `Arc<dyn MemoryProvider>` needs an SC-2-class object-safety conversion first (the trait is `#[trait_variant::make]`/RPITIT, not `dyn`-compatible); `ContextSources.memory` stays empty and the render path already accepts it, so wiring is additive later. Tracked: **#160** (memory follow-up). TS/Py/Go parity (guides + skills): **#159**. Local `main` now **17 commits ahead of `origin/main`, NOT pushed** — ask the maintainer before pushing (Deviation #10). Next within Phase 3: **SC-10** (per-phase prompt/toolset in PlanExecute), **SC-28** (relax plan to free-text/markdown); or the #160 memory follow-up; or port the parity backlog (#158/#159 + #154–#157)._

_2026-06-24 — **Phase 3 anchor LANDED (local `main`, NOT pushed) — `ca41f8f` (Q2 canonical middleware chain; subsumes SC-9 + SC-11).** The consumer-friction plan's Phase 3 opens with the Q2 anchor decision: adopt the rich `middleware.rs` `MiddlewareChain` as canonical, wire it into the ReAct loop, and **delete the harness-local stubs** (`MiddlewareChain`/`MiddlewareDecision`/the 4-variant `HookPoint` at `harness.rs`). The public `HookPoint`/`MiddlewareChain`/`MiddlewareDecision` now re-export the rich `middleware::` types (the former `Full*`/`MiddlewareHookPoint` aliases collapsed away; `HookContext` stays aliased `MiddlewareHookContext` to avoid colliding with the #69 hook `HookContext`). **Additive** — every new branch is guarded by `middleware.is_some()`, so the no-middleware path is byte-identical (no fixture/wire impact; the only fixtured middleware surfaces — `fixtures/middleware/checklist_basic.json` replayed by `middleware.rs`, and `observability/trace_line_middleware.json` — were already on the rich types). This **subsumes two consumer-friction items by construction:** **SC-9 (AfterTool can rewrite a result)** — `fire_after_tool(&calls, &mut results)` hands the batch's results MUTABLY; on `ContinueWithModification` the loop re-renders the affected conversation messages via a new `ContextManager::replace_tool_result` seam (default no-op; overridden in `StandardCompactionAdapter`), so the rewrite reaches the next model turn — the cordyceps `build_check` inversion done by the HARNESS, not the tool returning a fake error. Each appended result is tracked 1:1 with its `session_state.messages` index so the re-sync survives the #137 corrective-message interleaving (results live in `session_state.messages`, not `approved_results`, which `resume_inner` ignores — so re-sync, not defer-append, is the correct mechanism). **SC-11 (BeforeTool mutates dispatched calls)** — `fire_before_tool(&mut calls, _)` mutates in place with priority-ordered fan-out (the assistant turn recorded just before keeps the model's ORIGINAL request; only what's dispatched changes). Also delivers the **BeforeCompletion `ForceAnotherTurn`** injection the stub lacked: records the model's final text + the injection as a user message and re-enters the loop instead of completing (the same channel the Stop-block breaker uses). Each rich decision is mapped at its hook; out-of-place decisions are handled defensively (the `StandardMiddlewareChain` already converts illegal ones to `Halt`). **Deferred (separate follow-ups, noted not regressions):** the loop still does NOT fire `BeforeSession`/`AfterSession` (the stub never did either — needs a single-exit refactor of `run_react`'s ~15 return points), and it does not emit `MiddlewareSpan`s (unchanged). New tests: `after_tool_middleware_rewrites_result_in_place` (SC-9), `before_tool_middleware_mutates_calls_in_priority_order` (SC-11, two middlewares registered out of order → dispatched call carries `["first","second"]`), `before_completion_force_another_turn_runs_extra_turn` (turns==2, injection recorded), `replace_tool_result_rerenders_the_recorded_message` (adapter unit). The `ScriptedMiddleware` test double was ported to the rich 6-method trait; existing middleware tests (halt-at-BeforeTurn, SurfaceToHuman-at-BeforeTool, eval-phase middleware drop, resume-after-surface) compile unchanged (the rich `SurfaceToHuman`/`Halt` variants share the stub's shape). Full workspace suite green (**1242 lib pass / 0 fail**, 9 doctests; spore-eval 55), clippy clean (2 pre-existing `ollama.rs` test warnings unrelated); both in-repo examples build. Local `main` now **12 commits ahead of `origin/main`, NOT pushed** — ask the maintainer before pushing (Deviation #10). TS/Py/Go parity tracked in **#158**. Next within Phase 3: **SC-26/#115** (rich `ContextSources` into the live loop — the structural one, would retire ex12's `SkillInjectingContextManager` shim), **SC-10** (per-phase prompt/toolset in PlanExecute), **SC-28** (relax plan phase to free-text/markdown)._

_2026-06-24 — **SC-8 example migration DONE (local `main`, NOT pushed) — the "collapses to ~40 lines" proof.** The deferred SC-8 follow-up: both in-repo Rust examples now BUILD FROM the presets that were extracted from them, closing the loop. **`10-hill-climbing`:** `HarnessBuilder::hill_climber(build_model, evaluator)` folds in the metric evaluator (HillClimbing requires one) + `EscalationMode::AutoContinue`; the caller still adds the workspace-specific bits the preset deliberately omits (read-write sandbox, `write_file`/`read_file`, system prompt, propose-schema registry, observability sink). **`12-cordyceps`:** `HarnessBuilder::coding_agent(model, workspace_root)?` folds in the read-WRITE sandbox + `coding_set()` + built-in coding prompt + `AutoContinue` — which let the example **DELETE its hand-rolled `drive()`/`MAX_AUTO_CONTINUES`/`CONTINUE_STEPS` budget-grant loop, the manual `WorkspaceScopedSandbox` wiring, and 5 now-unused imports** (`drive` collapses to a direct `run_abortable(harness.run(..))` — `AutoContinue` grants in-process so `run()` returns terminal directly; the cap now surfaces as `Failure`, durable task list resumes on a follow-up prompt). It also collapses the two-call window sizing (`with_num_ctx` + explicit `CompactionConfig.context_length`) to a single `with_context_window` (SC-4) — `CompactionConfig::default()` auto-derives the budget from `provider().context_window` (#141). Skills stay example-side (the live loop bypasses the rich `assemble`, #115): the `load_skill` tool + `SkillInjectingContextManager` are layered on top of the preset (override `system_prompt` for the skills + plan-JSON contract, override `context_manager` for injection). A new `auto-cont` status line demonstrates the exposed `PRESET_MAX_AUTO_GRANTS`/`PRESET_STEPS_PER_GRANT` consts. **Additive, examples-only — no core change, no fixture/wire impact.** Both green: build + clippy clean, ex10 1 test / ex12 6 tests pass. **Live-verified against Ollama `qwen3-coder:480b-cloud`** (all local models are tool-capable cloud variants): ex10 wrote + scored (24/30, kept) + halted on the budget terminal — no stray pause/crash; ex12's plan→execute REPL emitted a 5-task plan, wrote+edited+ran `greeting.py` (`send_message` narration + skills manifest + plan-announcer hook all fired), and returned `Success` in 13 turns with NO consumer drive loop. READMEs + module docs updated to the preset assembly; ex10 README run-instructions now recommend a tool-capable model (the `llama3.2` default never emits tool calls). ex10's `Cargo.lock` synced to spore-core's current dep tree (pre-existing staleness surfaced by the build). Local `main` now **10 commits ahead of `origin/main`, NOT pushed** — ask the maintainer before pushing (Deviation #10). Next: **Phase 3** (Q2 canonical-chain + SC-10/26/28; SC-26/#115 would also fold skills injection INTO the harness, retiring this example-side shim); or port the parity backlog (#154/#155/#156/#157) via `/implement`._

_2026-06-23 — **Phase 2.5 SC-8 presets LANDED (local `main`, NOT pushed) — `6f39933`.** The two working-consumer harness shapes, extracted into `HarnessBuilder` presets so a consumer collapses to ~40 lines (the plan's "presets carry the friendly behaviour", Q1). Both compose `conversational(model)` + the Phase 1–2 knobs. **`coding_agent(model, workspace) -> Result<Self, BuildError>` (looper):** a read-WRITE `WorkspaceScopedSandbox` rooted at `workspace`, the full `StandardTools::coding_set()`, the built-in `CODING_AGENT_SYSTEM_PROMPT` (act-with-tools + `send_message` narration), and `EscalationMode::AutoContinue` (SC-5, autonomous-but-capped — replaces the consumer's hand-rolled drive loop). Returns `Err` if the workspace can't canonicalize. **`hill_climber(model, evaluator) -> Self` (cordyceps):** registers the scoring `MetricEvaluator` (required for HillClimbing) under the default handle + `AutoContinue`; leaves sandbox/tools/system-prompt to the caller (climbs vary — some climb a prose artifact, some climb files; matches the plan signature). Shared defaults exposed as `HarnessBuilder::{CODING_AGENT_SYSTEM_PROMPT, PRESET_MAX_AUTO_GRANTS=10, PRESET_STEPS_PER_GRANT=25}` (the `12-cordyceps` example's MAX_AUTO_CONTINUES × CONTINUE_STEPS), overridable via `escalation_mode(..)`. **Window sizing rides SC-4/SC-6:** size the model's window once with `with_context_window(n)` before passing it in, and the preset's `conversational` context manager auto-derives the compaction budget from `provider().context_window` — no manual `context_manager`. The strategy stays per-run on the `Task` (ReAct/PlanExecute/HillClimbing — a few lines). **Additive** — new constructors only, no existing-path change, no fixture impact. New tests: `coding_agent_preset_{wires_sandbox_tools_prompt_and_autocontinue, errors_on_missing_workspace}`, `hill_climber_preset_registers_evaluator_and_autocontinue`, + a `no_run` doctest each. Full workspace suite green (1238 lib pass / 0 fail, 9 doctests); clippy clean (also tightened two SC-BUG-1 test assertions `.len() >= 1` → `!is_empty()` that slipped into `8d1d679`; the 2 pre-existing ollama.rs test warnings remain, unrelated). TS/Py/Go parity tracked in **#157** (depends on #155 for the window-sizing half). **Follow-up:** migrate the in-repo `10-hill-climbing` / `12-cordyceps` examples to USE the presets — the "collapses to ~40 lines" proof — a separate live-model-verified change. Local `main` is now **7 commits ahead of `origin/main`, NOT pushed** (`c5f7a01`, `2fc0fc0`, `f1c0beb`, `6732e85`, `8d1d679`, `af788ca`, `6f39933`) — ask the maintainer before pushing (Deviation #10). Next: **Phase 3** (Q2 canonical-chain adoption — subsumes SC-9 + SC-11 — plus SC-10/26/28); or migrate the two examples onto the presets; or port the parity backlog (#154/#155/#156/#157) via `/implement`._

_2026-06-23 — **SC-BUG-1 LANDED (local `main`, NOT pushed) — `8d1d679`.** The one correctness bug in the consumer-friction plan (§5), ranked ABOVE Phase 2 — **the looper verify unlock**. A HITL pause (tool-approval / clarification / deny) raised inside a `SelfVerifying` or `PlanExecute` frame is generated by the worker ReAct leaf, so the `PausedState` carried the LEAF task; `resume_inner`'s Allow/Deny/Answer/clarification tails ended at `run_react` on that bare leaf, re-driving only the executor and SKIPPING the surrounding frame. Under `AlwaysAsk` the SelfVerifying verify gate silently degraded to a plain executor and a resumed run never re-entered the build↔evaluate loop — so it never reached the #151 eval-phase reviewer (SC-30 was inert for looper until now). **Fix mirrors the #131 consult-resume pattern, three parts:** (1) `ExecutionContext::finish` now rewrites a propagating `WaitingForHuman`'s `state.task` to the combinator's OWN composed task on the way up — exactly as it already did for `Consult` — so the top-level pause carries the full strategy tree; (2) `resume_inner`'s clarification tail AND the final Allow/Deny/Answer/Reject match tail route a COMPOSED task (`loop_strategy != ReAct`) through `drive_strategy_with_resume_seed` (fresh top session + the mutated worker session as the resume seed), keeping a bare ReAct leaf on the original `run_react` path (back-compat); (3) `SelfVerifyingConfig::run` consumes `scratch.resume_seed` as its FIRST build iteration's session, so the resumed worker conversation continues (build finishes) and then the evaluate phase + verifier run. PlanExecute already consumes the seed when it is the OUTER frame (so a nested SelfVerifying sees `None`, unchanged) and is re-driven by the same call the #131 consult path already exercises; only combinator frames rewrite the task (the leaf's `record_terminal` does not), so a top-level bare-leaf pause is unaffected. **Additive** — a fresh run leaves the seed `None` (byte-identical); `WaitingForHuman`/`PausedState.task`/`EscalationMode` are not serialized on the recorded wire, so no fixture impact. New tests: `self_verifying_hitl_{resume,deny,clarification}_reenters_eval_frame` — each pauses on the build phase, resumes (Allow / Deny / clarification Answer), and asserts the eval-phase verifier ran (0 verifier calls before the fix, ≥1 after). Full workspace suite green (1235 lib pass / 0 fail); clippy clean (2 pre-existing ollama.rs test warnings unrelated). TS/Py/Go parity tracked in **#156**. Local `main` is now **5 commits ahead of `origin/main`, NOT pushed** (`c5f7a01`, `2fc0fc0`, `f1c0beb`, `6732e85`, `8d1d679`) — ask the maintainer before pushing (Deviation #10). Next: **Phase 2.5 (SC-8 presets)** — extract `HarnessBuilder::hill_climber(model, evaluator)` (cordyceps) + `coding_agent(model, workspace)` (looper) presets now that Phases 1–2 + SC-BUG-1 give the friendly knobs + working verify-resume; or **port #156** (and #154/#155) via `/implement`; then Phase 3._

_2026-06-23 — **Phase 2 SC-4/5/6/27 LANDED (local `main`, NOT pushed) — `f1c0beb`.** The consumer-friction plan's Phase 2 (provider/model knobs; all additive, un-gated by Q1, ships alongside Phase 1). **SC-6 (per-provider window override):** each provider (ollama/anthropic/openai) gained a `context_window_override` field + `with_context_window(n)` setter; `provider().context_window` prefers it over the static id table (Ollama precedence: override > `/api/show` discovery > static). **SC-4 (one-call window sizing):** Ollama's `with_context_window(n)` ALSO sets `num_ctx`, so one call makes the model load at AND report window `n`; the compaction budget auto-derives via the existing #141 `resolve_context_length` chain (no second setter) — closes looper's `num_ctx` drift (the documented discovery/enforcement split). Stale "80% of 200K" builder doc corrected. **SC-27 (OpenAI `with_compat`):** `OpenAICompat { reasoning_model, developer_role, supports_reasoning_effort }` is OR'd OVER the `o1/o3/o4` `is_reasoning_model` heuristic — a local/renamed reasoning model gets `max_completion_tokens`, the `developer` role, and a `reasoning_effort` request field (`low|medium|high|max`, mirroring Ollama's `think` levels); default compat keeps recognized o-series byte-identical. **SC-5 (`EscalationMode::AutoContinue`):** new third variant `AutoContinue { max_grants, steps_per_grant, on_grant: Option<Callback> }` = "autonomous but capped" — at every `Escalate` resolution site the harness auto-grants `steps_per_grant` more steps and keeps working IN-PROCESS (per-scope `auto_grants_used` + `BudgetContext::grant_auto_continue` additive cap-raise; `ExecutionContext::try_auto_continue` at the 5 nested combinator/leaf sites; a loop-local counter at the top-level bare-leaf `drive_strategy_with_resume_seed` site), firing `on_grant` per grant up to `max_grants`, then falls through to the existing `Autonomous` terminal. Default stays `SurfaceToHuman`. This is the keep-working-but-cap-at-N policy cordyceps `drive()` / looper governor hand-roll. The `on_grant` callback (`Arc<dyn Fn>`, serde-skipped) makes `EscalationMode` no longer `Copy`/`Eq` (hand-rolled `Debug`); it is never serialized in fixtures, so no wire impact. New tests: provider window override + num_ctx fan-out + compaction-budget resolution; compat reasoning/developer-role/reasoning-effort; AutoContinue grant mechanics (`BudgetContext` + `ExecutionContext` units) + full-run completes-via-grant + caps-at-max-grants-then-fails. Full workspace suite green (1232 lib pass / 0 fail); clippy clean (2 pre-existing ollama test warnings unrelated). TS/Py/Go parity tracked in **#155**. Next: **Phase 2.5 (SC-8 presets)** once Phase 1 lands, then Phase 3 (Q2 canonical-chain adoption + SC-10/26/28); **SC-BUG-1 with the #151 follow-up** (HITL-resume re-enters SelfVerifying/PlanExecute frames — the looper unlock) remains the ranked-above-Phase-2 correctness item not yet done._

_2026-06-23 — **Phase 1 SC-2 + SC-3 LANDED (local `main`, NOT pushed) — `c5f7a01`.** The rest of Phase 1 after SC-1/SC-30, the "alongside" pair. **SC-2 (ModelInterface object-safety):** converted `ModelInterface` from `#[trait_variant::make(Send)]` (RPITIT, not `dyn`-compatible) to the house `BoxFut` idiom (the `RunStrategy`/`ToolRegistry` precedent — Deviation #12), rippling through all 14 impls (anthropic/openai/ollama/replay/recording/adaptive + prompt-based wrapper + 7 test doubles; the 5 stream-building impls annotate the `Ok::<ModelStream,_>` tail since the unsize-coercion site moved inside `Box::pin(async move {})`). Added a blanket `impl ModelInterface for Arc<T: ?Sized>` + `HarnessBuilder::conversational_arc(Arc<dyn ModelInterface>)` (Approach A — blanket impl, generic path retained, no `?Sized` ripple into the wrappers; the A-vs-B internal-cleanliness choice is Rust-only and does not affect the cross-language contract). `CONVENTIONS.md` + stale `ModelAgent`/wrapper doc comments updated — the "components injected as `Arc<dyn ModelInterface>`" claim is now true. **SC-3 (typed retryable model errors):** added `ModelError::Transport` + `ModelError::StreamInterrupted` variants (enum already `#[non_exhaustive]`) + a `retryable()` predicate (`Transport | StreamInterrupted | Timeout | RateLimited` ⇒ true); reclassified the transport-error + stream-chunk-error sites in ollama/anthropic/openai (deterministic encode/decode/capability/not-found stay `ProviderError`). No fixture impact (errors never on the recorded happy-path wire; plan Q1). New tests: boxed-model harness construction + dispatch, `retryable()` classification, serde tag-shape (`{"kind":"Transport"|"StreamInterrupted","message":..}` — cross-language ground truth), and a raw-`std::net::TcpListener` mid-stream truncation asserting `StreamInterrupted` + `retryable()`. Full workspace suite green; clippy clean (same 2 pre-existing ollama test warnings, now at 1583/1588). TS/Py/Go parity tracked in **#154**. Next: **SC-BUG-1 with the #151 follow-up** (HITL-resume re-enters SelfVerifying/PlanExecute frames — the actual looper unlock), then **Phase 2** (SC-4/5/6/27, all additive, un-gated) + presets (**SC-8**)._

_2026-06-23 — **Consumer-friction plan ADOPTED** (`docs/consumer-friction-plan.md`, v2.0; signed off by spore-core + cordyceps + looper). It reframes the cross-language defaults as tuned for fixture-replay, not consumers, and sequences ~30 fixes (SC-#/ARK-#/LOC-#) to make spore-core ergonomic for the cordyceps hill-climber and looper coding-agent. **Phase 0 complete:** the #151 SelfVerifying eval-phase reviewer slice (`eval_agent`/`eval_toolset` + read-only eval + eval-phase middleware drop) was test-gated (1255 pass / 0 fail) and committed as **`d14341f`** — the agreed pin all subsequent PRs build on. Reconciliations: **D1** canonical middleware surface = the rich `middleware.rs` chain (verified: the only fixtured surface across all four languages) → wire the loop to it and delete the `harness.rs:5184` stub (collapses SC-9/SC-11/Q5); **D2** the `enforce_output_schemas` default lives **only** in `HarnessBuilder::new` (`harness.rs:6429`) — there is no `impl Default for HarnessConfig`, and `hooks.rs:1763` is a test-only builder → fix is a single `Default` impl both derive from. Decisions: Q1 no fixture re-baseline (friendliness via additive setters + presets SC-8), Q3 auto-synthesize, Q4 `EscalationMode::AutoContinue`, Q7 structural `ContextSources` (#115). **Phase 1 SC-1 + SC-30 LANDED + PUSHED:** SC-1 `37596d5` (removed `check_structured_slot` — structured slots may omit `output`), SC-30 `8be060b` (internal `ReadOnlyToolView` auto-derives a read-only eval catalogue = catalogue ∩ `readonly_set()` when `eval_toolset` empty; decision A intersection), example proof `e063cfc` (12-cordyceps drops `PLAN_SCHEMA_KEY`). Full suite green, clippy-clean (2 pre-existing ollama.rs:1573/1578 test warnings unrelated). **`origin/main` is now current at `e063cfc`** (Deviation #10's "NOT pushed" backlog — #149 ports, #150's `64938ee`, #151, Phase-0 — was pushed in the same `git push`, maintainer-authorized). TS/Py/Go parity tracked in **#153**. Next: SC-2/SC-3 (Phase 1), then Phase 2 (additive) + presets (SC-8); SC-BUG-1 with the #151 follow-up._

_Last updated: 2026-06-19 by /close (#149 **complete + CLOSED** — Ollama `num_ctx` ported from the Rust reference `3273386` to TS `ff5358a`, Python `5c9b613`, Go `4a544a7`; opt-in interface field (each language's `keep_alive` idiom, not the literal `with_num_ctx`), threaded into `options.num_ctx`, **omitted when unset** (existing fixtures replay byte-identical), `num_ctx` serialized **first** in `options` to match Rust serde field order; wire-level HTTP-mock test + unit tests in each; verifier PASS across all four. The three port commits are on **local `main` only — 3 ahead of `origin/main` (`64938ee`), NOT pushed**.)_

_**Direction note:** The project has shifted from "hardening cluster done, maintainer call" into an active **cross-language parity catch-up** phase. Rust raced ahead on `main` with a batch of Ollama + sandbox + bugfix work; TS/Python/Go are now catching up issue-by-issue via `/implement` (land Rust first → three parallel language agents → cross-language verifier). Open parity backlog: **#150** (recoverable sandbox violations — larger, breaking default change), **#148** (Ollama thinking/reasoning models), **#146** (web_fetch start_byte, Python/Go), **#147** (SelfVerifying evaluator budget, TS/Python), and **#144** (PlanExecute budget-resume — ports appear landed, verify + `/close 144`). #149 (num_ctx) and #145 (SSRF seam) are done/closed. This is the current north star until the backlog drains; refactor close-out (#131 + finishers) and parked examples (#109/#92) remain behind it. The hardening cluster #137–#143 + #139/#141 remain fully closed._

## Current State
spore-core is a language-agnostic agentic harness runtime with a **complete core
capability surface**, four targets — Rust (reference), TypeScript, Python, Go —
serialized formats byte-identical across all four. Local `main` is **3 commits ahead of
`origin/main`** (`origin/main` at `64938ee`); the three #149 `num_ctx` ports
(`ff5358a`/`5c9b613`/`4a544a7`) are committed locally but **not yet pushed** — ask the maintainer
before pushing (Deviation #10).

**Active phase — cross-language parity catch-up (#144–#150).** Since the 2026-06-14 reconcile, Rust
raced ahead on `main` with a wave of Ollama + sandbox + bugfix work, and TS/Python/Go are catching up
issue-by-issue via `/implement` (Rust reference → three parallel language agents → cross-language
verifier). Backlog status:
- **#149 — Ollama `num_ctx` ✅ DONE THIS LOOP (`status: complete`, CLOSED).** Opt-in interface field
  (`numCtx` ctor option / `num_ctx` kwarg / `SetNumCtx` setter — each language's `keep_alive` idiom, not
  the literal Rust `with_num_ctx`), threaded into `options.num_ctx`, **omitted when unset** (bare requests
  + existing fixtures stay byte-identical), `num_ctx` serialized **first** in `options` to match Rust serde
  field order (TS/Python carry explicit ordering tests); wire-level HTTP-mock test in each. Verifier PASS.
  Rust ref `3273386`; ports TS `ff5358a`, Python `5c9b613`, Go `4a544a7`.
- **#145 — SSRF URL-validation seam ✅ DONE (CLOSED).** Ported TS `986a15e`, Python `ca3d8a9`, Go
  `498c87f` (see memory `ssrf-seam-url-bracket-parity-gotchas`).
- **#144 — PlanExecute budget-resume fix → ports appear LANDED, issue still OPEN.** TS `2401fee`, Python
  `9b1d310`, Go `b56e852` + Rust docs `a25e6bf` are on `main`; verify the AC and run `/close 144`.
- **#150 — recoverable sandbox violations via `SandboxViolationPolicy` (port OWED).** Rust landed
  (`64938ee`); TS/Python/Go owed — the larger port, a **breaking default change** (path escapes / blocked
  commands become recoverable feedback). See issue #150 + the handoff for the typed-violation-to-harness
  shape (keep the loud compile error on exhaustive matches).
- **#148 — Ollama thinking/reasoning models (port OWED).** Rust landed (`a9c856b`); TS/Python/Go owed.
- **#146 — char-boundary-safe `web_fetch` `start_byte` slicing (Python/Go OWED).**
- **#147 — SelfVerifying must charge evaluator turns against budget (TS/Python OWED).**

Also Rust-ahead but **not yet filed as parity issues**: Ollama streaming `read_timeout` fix (`56361b2`)
and Ollama think-effort levels (`c486331`) — confirm whether these need their own TS/Py/Go parity issues
or fold into #148.

**🎯 The `12-cordyceps` hardening cluster #137–#143 is COMPLETE** (all five closed), and
the adjacent **#139** (output-schema enforcement) and **#141** (configurable compaction
window) are now done too — **every robustness gap is closed.** Running the capstone
composition live on gemma exposed these gaps, each verified in the Rust reference (several
observed live); all are now landed across all four languages:

- **#137 — ReAct tool-error-loop breaker ✅ DONE (`status: complete`).** Per-tool
  consecutive-recoverable-error tracking; corrective schema injection at N (default 3);
  stop + `BudgetExhaustedBehavior` resolution with typed `HaltReason::ToolErrorLoop` at 2N
  (budget not burned); stream + observability at both thresholds. Shared fixture
  `tool_error_loop.jsonl`; byte-identical AC2 schema string in all four; Go `*uint32`
  sentinel for `ErrorLoopThreshold` default parity.
- **#142 — project-scoped durable storage / stable `project_id` ✅ DONE THIS LOOP
  (`status: complete`, CLOSED).** New `ProjectId` newtype derives a stable id from
  `sandbox.workspace_root()` (canonicalize-first → `{sanitized_basename}-{8hex}`, reusing
  the existing `WorkspaceId` algorithm; the 8-hex SHA-256 suffix resolves the `/a/b` vs
  `/a_b` slug collision). Threaded through `ToolContext` → tool registry →
  `HarnessConfig`/builder. The durable artifacts (`task_list`, plan, **Ralph checkpoint** —
  moved onto the store) are keyed by `project_id` **only** (not project_id+session) via
  namespace-reuse on the existing session-id axis (the `RunStore` trait was not widened);
  ephemeral session state (conversation, `active_skills`) stays session-keyed so it still
  resets per Ralph window. Active-run lifecycle (new / resume / **complete** via an explicit
  caller API + caller-supplied run tag). The `12-cordyceps` example now wires
  `FileSystemStorageProvider` via `CompositeStorageProvider` under central
  `~/.spore/projects/<project_id>/`. Two shared fixtures (`project_id_derivation.json`,
  `project_durable_survival.json`) replay byte-identically in all four; verifier
  independently recomputed all 7 pinned hashes. Commits Rust `6bcabb4`, TS `a037861`,
  Go `631290f`, Py `5b7804f`. **This makes the task_list survive Ralph window resets AND
  process restarts — the soil the error-grind grew in — and unblocks #138.**
- **#143 — `add_task` returns the assigned id ✅ DONE (`status: complete`, CLOSED this loop).**
  Implemented across all four languages in the prior session (Rust `a1d6053`, Py `b01f2d7`, Go
  `4c4b586`, TS `e508d23`, docs `5e206e1`) and on `main`; formally closed during this reconcile.
  Cuts the malformed-call grind: small models no longer parse/predict ids for
  `blockers`/`update_task`/`complete_task`.
- **#138 — resume seeds the stalled worker + skips re-planning ✅ DONE THIS LOOP
  (`status: complete`, CLOSED).** Three behavioral fixes, all four languages: **(AC1)** on
  PlanExecute re-entry with a persisted non-empty `task_list` (the #142 `project_id` durable
  axis lets it survive the Ralph window reset), skip the plan phase and go straight to the
  ready-set walk (`reconcile_completed_tasks` dedups completed tasks) instead of re-running PLAN
  unconditionally; **(AC2)** a budget-resume of an execute-phase exhaustion seeds the stalled
  worker by generalizing the #131 `consult_resume` seed to a phase-agnostic resume seed (was
  `None`), so the worker resumes its audit instead of re-exploring (the live gemma4:31b-cloud
  failure); **(AC3)** a plan-phase exhaustion resumes the planner's own session rather than
  cloning the paused worker's session into the planner's context. New shared fixture
  `cordyceps_budget_resume.jsonl` + updated `cordyceps_budget_exhausted.json` paused-state replay
  in all four. Tests verified green this loop (Rust 9 / Go 2 named / Py 11 / TS 29); the Go
  skip-replan test wires a **real in-memory `RunStore`**, not the no-op default, so the
  store-dependent guard is genuinely exercised. Commits Rust `99a16be`, Py `9133762`, Go
  `4827924`, TS `5ec555a`.
- **#141 — compaction window now model-configurable ✅ DONE THIS LOOP
  (`status: complete`, CLOSED).** `SessionState::new` no longer hardcodes `window_limit: 200_000`.
  `CompactionConfig` gained a `context_length: Option<u32>` field (serialized **absent** when unset
  — `skip_serializing_if`/`omitempty`/optional/`None`-excluded — so existing serialized configs stay
  byte-identical), and `StandardContextManager` gained a `resolve_context_length()` resolver with the
  fallback chain **config (`> 0`) → model `context_window` (`> 0`) → `DEFAULT_CONTEXT_LENGTH = 8000`**,
  applied through a manager-owned `seed_session()` helper (the real production seam — the harness
  round-trips the rich-state blob via `extras`, so callers/the manager seed it, not the loop). Trigger
  math (`should_compact`) is unchanged — once seeded with the resolved window, `threshold × window_limit`
  respects config automatically. **Maintainer-pinned spec changes vs. the original AC:** field renamed
  `window_limit` → `context_length`; explicit `0`/null/nil **falls through** (a zero window would
  silently disable compaction — the exact bug); **no clamping** of an oversized configured value; and
  the unknown-context fallback is **8000, not 200_000** (the `SessionState` constructor default dropped
  to 8k — conservative, fixes the gemma-8k / 128K overrun rather than preserving the dangerous default;
  provider `context_window` defaults like Claude/OpenAI 200_000 are untouched). New shared fixture
  `fixtures/compaction_window/cases.json` (5 `trigger_cases` + 6 `resolver_cases`) replays byte-
  identically in all four; existing `compaction_loop`/`compaction_verifier` fixtures untouched. The
  one verifier-caught divergence — TS Zod `.positive()` would have rejected an explicit `0` the other
  three accept — was fixed (`66aed39`, `.nonnegative()`). Verifier PASS. Commits Rust `1e3cf4c`, Go
  `67c8d39`, Py `6be78a2`, TS `32ed008` + `66aed39`.
- **#139 — `ReactConfig.output` schemas delivered + enforced ✅ DONE THIS LOOP
  (`status: complete`, CLOSED).** The schema was presence-validated by `ExecutionRegistry` at
  startup but `ReactConfig::run` never read it at runtime; that gap is closed. **(AC1)** when
  enforcement is on and a leaf has `output` set, the resolved (key-sorted) schema is appended to
  the directive seed AND routed into the model's structured-output channel (Ollama `format`;
  Anthropic/OpenAI no-op). **(AC2)** the terminal `FinalResponse` is validated by a **hand-rolled
  minimal validator** (subset `type`/`required`/`properties`/`enum`); on mismatch the frozen
  validation error is fed back as a user message and the leaf retries up to N extra turns (N =
  `output_schema_max_retries`, default 2; total 1+N), retries counting against budget. **(AC3)**
  after N failed retries with budget remaining → typed `HaltReason::OutputSchemaViolation`, distinct
  from budget exhaustion (budget cap wins precedence if hit first). **(AC4)** global
  `HarnessConfig.enforce_output_schemas`, **default OFF** — a documented migration gate; OFF keeps
  every existing replay fixture byte-for-byte green. Three shared fixtures
  `output_schema_{accept,retry,fail}.jsonl`. Parity-critical determinism: lexicographically-sorted
  property iteration, semantic `integer` check (42.0 passes / 42.5 fails), canonical key-sorted
  `{enum}`/`{value}` rendering — all frozen byte-identical across the four. Verifier PASS. Commits
  Rust `3790997`, TS `f412b95`, Go `0157749`, Py `15fbfc6`.
- **#140 — `PausedState` carries the leaf's toolset handle ✅ DONE THIS LOOP
  (`status: complete`, CLOSED).** `PausedState` + `ChildPausedState` gained an always-serialized,
  serde-default `toolset` field (last field, byte-parity-safe); all 7 leaf pause sites populate it
  and both resume paths thread it into `effective_tool_registry`, so a node with a per-node toolset
  now resumes against its scoped catalogue instead of the empty global fallback (the cordyceps
  Consult repro). Two extra embedded-paused-state fixtures (`harness/consult.json`,
  `harness/escalation_signals.json`) also needed the key. Load-bearing AC2b resume-routing test
  (+ negative control: empty handle → recoverable unknown-tool error) in all four. Commits Rust
  `9998a0c`, Py `d66afe3`, Go `3e177d3`, TS `d8f8123`.

**Landed: Composable Execution refactor #117–#131 (all `status: complete` except the
still-open #131 capstone).** Delivered across all four languages, byte-identical where
serialized; per-issue detail lives on the GitHub issues. Summary of what the runtime
now does as a result:
- **#117 `BudgetPolicy` + `BudgetExhaustedBehavior`** value types (`Unlimited`/
  `TotalSteps`/`PerLoop`/`PerAttempt`; `Continue{max_continues,on_exhausted}`/`Escalate`/
  `Fail`), layered over the `BudgetLimits` global backstop.
- **#118 `Task.blockers` DAG schema** + `add_task` `blockers` param with validate-
  before-mutate (`self_block`/`unknown_id`/`cycle` → recoverable `invalid_blockers`).
- **#119 + #120 the strategy seam:** `LoopStrategy` is a closed recursive serde enum of
  config newtypes (`react`/`plan_execute`/`self_verifying`/`ralph`/`hill_climbing`),
  each owning its loop via the `RunStrategy` trait (one-line enum delegation, no central
  match); `StrategyRef::{BuiltIn,Custom}`; an `ExecutionRegistry` resolves `*Ref` handles
  + custom keys (`StrategyNotFound`/`UnresolvedHandle`).
- **#123 + #124 genuine composition:** typed runtime `StrategyOutcome`
  (`Complete`/`BudgetExhausted`/`Failed`, never serialized) + shared `ExecutionContext`;
  all five strategies genuinely recurse via `self.inner.run(cx)`; the monolithic
  `run_self_verifying`/`run_ralph`/`run_hill_climbing` loops and the legacy collaborator
  fields are **deleted** — all collaborators resolve through the registry.
- **#125 + #126 enforcement + scheduling:** per-node budget `charge()` with isolated,
  parent-inspectable `BudgetExhausted` (no auto-cascade; ReAct leaf propagates); ready-
  set DAG task walk, two-tier context (transitive-blocker outputs + bounded N=20 step
  ledger with harness-observed `files_touched`), failure cascade to transitive
  dependents only (`TasksBlockedByFailure`).
- **#130 + #129 HITL + Continue resume:** `EscalationMode{SurfaceToHuman,Autonomous}`
  consumed at every Escalate site → `HumanRequest::BudgetExhausted` +
  `EscalationAction{ContinueWithBudget,Skip,Fail}`; serialized `behavior` field on all
  five configs makes in-process `Continue` genuinely loop; resume seeds `continues_used`
  off the request payload; shared `PausedState::{serialize,load}_checkpoint`.

**`12-cordyceps` capstone (#101 + #131, all four languages):** a fully-autonomous
task-completion agent — ReAct orchestrator + `task_list` decomposes a per-module Rust
audit; an Isolated `analysis_worker` deep-dives one module and loads an `audit` skill at
runtime; the #114 consult ladder escalates a stuck worker to a sibling then a human;
heterogeneous models (local gemma + cloud advisor); within-run memory; REPL approval →
`gh issue create`. #131 re-expresses it as `Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]`.
Core integration merged via **PR #136**; #131 is **functionally landed but still formally
open** (run `/close 131` after confirming the success criteria). As of #142 the example
persists durably under `~/.spore/projects/<project_id>/`.

**Skill loading is architect-side (the #101 design constraint):** the live loop builds
each turn's prompt via `StandardCompactionAdapter::assemble` (pass-through of
`session.messages`); the rich `StandardContextManager::assemble` that structurally injects
skills/chunks/merged-memory is **bypassed** pending the deferred **#7** ContextManager
migration (root cause of Deviation #8). So skills/chunks/memory reach the model only as
tool-result messages. #101 works around this with a `SkillCatalog` + `load_skill` tool +
a custom context manager; **#115** tracks baking this into the library.

**Harness core:** all 5 loop strategies run end-to-end in all four languages
(ReAct/PlanExecute/SelfVerifying/Ralph/HillClimbing — none stubbed, all via genuine
recursive `RunStrategy` dispatch as of #124); mid-loop consult primitive (#114, with the
#116 HITL child-consult-resume gap); tool/prompt architecture (#79–82); pluggable
scope-aware persistence (#73/#75/#76/#78/#82) — now with a stable `project_id` durable key
axis (#142); runnable (#57), debuggable (#64/#65), evaluation loop (#26/#68).

**Examples suite — 12 of 13 landed, all four languages each** under
`examples/{rust,typescript,python,go}/`: `01-hello-agent` … `11-multi-agent`,
`12-cordyceps` (#101). Remaining: **#109** (`13-coding-agent`) and **#92**
(observability/Phoenix-OTLP). Parked behind the hardening cluster.

**Parked (not active): correctness/safety debt + docs.** #34 (Yolo/None feature flag),
#31 (SharedSession read-only), #30 (memory PendingReview gate), docs #27/#35/#36 — all
`scope: deferred`.

## Active Direction
**The `12-cordyceps` hardening cluster #137–#143 is COMPLETE** (#137 ✅, #138 ✅, #140 ✅,
#142 ✅, #143 ✅ — all closed), and the adjacent robustness gaps **#139** (output schemas) and
**#141** (compaction window) are **done and closed too.** `Ralph[PlanExecute[ReAct,
SelfVerifying[ReAct]]]` now survives Ralph window resets and process restarts with the `task_list`
durable (#142), resumes the stalled worker instead of re-planning (#138), routes resumed tool calls
through the leaf's scoped catalogue (#140), breaks tool-error grind loops (#137), enforces output
schemas behind a migration gate (#139), and **fires compaction correctly for small/unknown-window
models (#141)**. The composition's small-local-model reliability gaps that motivated the cluster are
fully addressed.

**The active direction is the cross-language parity catch-up backlog (#144–#150).** The maintainer
pushed a Rust-first wave of Ollama + sandbox + bugfix work; the job now is porting each to TS/Python/Go
via `/implement` (Rust reference → three parallel language agents → cross-language verifier),
byte-identical where serialized, until the backlog drains. Order, roughly: **#149 ✅ done → #150**
(recoverable sandbox violations — larger, breaking default change) **and #148** (Ollama thinking models)
for the Ollama/sandbox cluster; then the smaller bugfix ports **#146** (web_fetch start_byte) and **#147**
(SelfVerifying evaluator budget); **#144** ports appear landed — verify + `/close 144`. Behind the parity
backlog the prior maintainer-call candidates still stand: refactor close-out (`/close 131` + finishers
#121/#122/#127/#128), parked examples (#109 `13-coding-agent`, #92 observability) + `web_search`
#108/#110, then larger parked features (#113/#107/#106, protocol track #83–87) and correctness/safety
debt #34→#31→#30 + docs.

**Housekeeping:** local `main` is **3 commits ahead of `origin/main`** (`64938ee`) — the three #149
ports are unpushed; **ask the maintainer before pushing** (Deviation #10). `/close 131` (confirm the
capstone success criteria; no code) remains an outstanding cheap reconcile-only step.

**Parked behind the hardening cluster:** examples #109/#92 + `web_search` #108/#110; harness
gaps #115 (skill loading) and #116 (HITL child-consult resume — overlaps #130's resume seam,
may be cheaper to fold in); correctness/safety #34 → #31 → #30 and docs #27/#35/#36; larger
features #113 (spore-lsp), #107 (PromptEngineeringAgent), #106 (MicroVMSandboxProvider), the
protocol track #83–87, storage follow-ups #77/#88/#89. **#7** (ContextManager migration) would
live-wire the rich `assemble` (proper home for #115's injection + the #32 cache halts).

## Known Deviations
1. **Go outbox is not zero-dependency** — closing #50 added `go.opentelemetry.io/otel` +
   `otlptracegrpc` (v1.28.0) as blessed deps (documented in `go/CONVENTIONS.md`). The durable
   JSONL path stays network-free.
2. **`task_list` / `todo_write` default persistence is no-op, not a file** (`scope: debt`,
   minor) — #75 retired the sandbox path; standalone tools persist via `RunStore`, which is
   `no_op()` by default. **#142 (landed) makes this real for the cordyceps Ralph loop**: the
   example now wires `FileSystemStorageProvider` via `CompositeStorageProvider` under
   `~/.spore/projects/<project_id>/`. The library default is still no-op; durable standalone
   use still requires wiring a real `StorageProvider`.
3. **v1 memory keying limitation (#78 Q7), filed as #89** (`scope: deferred`) — `MemoryStore`
   is `SessionId`-keyed; durable cross-session addressing is the v2 feature. No SQL backend yet
   (#77). (Note: #142 added a separate `project_id` durable key axis for the run store, not for
   memory — memory keying is unchanged.)
4. **Go-specific divergences** (`scope: debt`, minor, documented on the issues) — local `Mode`
   newtype; 3-state `TerminalOutcome`; type-aliased `StandardTool`; explicit `abort` `reason`;
   self-contained `promptassembly` builder; opaque `ToolContext.MemoryStore`; exported
   `storage.MergeMemories`; config-struct (not builder) setters; `RoleEvaluatorChunk` constant;
   consumer-side `MetricEvaluator`/`ContextError` seams to avoid import cycles; custom context
   manager embeds `*StandardCompactionAdapter`; `RunStrategy.Run` takes `ctx context.Context`
   first; Go keeps `Agent`/`ToolRegistry` as struct fields folded into the registry + an
   `IsEmpty()` validate gate. #137 adds the `ErrorLoopThreshold` `*uint32` sentinel + sibling
   `effectiveErrorLoopThreshold()` (the house idiom). All wire/behavior-identical.
5. **Test-placement divergences (#78/#82)** (benign) — registry-seam / catalogue tests live in
   language-idiomatic spots. Behavior identical.
6. **#79 cross-language divergences — both verified benign.** (a) narrowed `composed_prompt`
   stub in TS/Go; (b) Block-1 hash not byte-identical (Rust SipHash vs FNV-1a) — the intentional
   #24 decision; #79 fixtures assert no hash values.
7. **`Custom` condition is invisible in fixtures by design** (#79) — serializes to null/absent.
8. **The live harness loop does not call the rich `assemble`** (`scope: deferred`, intentional,
   depends on #7) — prompts are built via `StandardCompactionAdapter::assemble` (pass-through of
   `session.messages`); the rich `StandardContextManager::assemble` (skill/chunk/memory injection +
   Block-1/2 `CacheHashMismatch` halts) is bypassed. So skills/chunks/memory reach the model only
   as tool-result messages, and the #32 cache halts can't fire end-to-end. Live-wiring is #7's job.
9. **#114 HITL has no child-consult resume, filed as #116** (`status: queued`) — `EscalateToHuman`
   consult overflow surfaces `WaitingForHuman` at the parent with the worker's paused consult in
   `child_state`, but `resume`'s `child_state` branch is a **no-op** in all four cores. #101's three
   escalation choices are implemented host-side. **#140 (toolset handle on resume) is now landed —
   `ChildPausedState` carries the child's toolset and `child_state_from_paused` propagates it, so when
   #116 finally wires the `child_state` resume branch the scoped catalogue is already available.
   **#138 (now landed) generalized the resume seed to be phase-agnostic, so #116 can reuse that
   seam directly when it wires the `child_state` branch.**
10. **Local `main` push hygiene (standing reminder).** ⚠️ **3 AHEAD (2026-06-19, this loop):** local
    `main` is 3 commits ahead of `origin/main` (`64938ee`) — the three #149 `num_ctx` ports
    (`ff5358a`/`5c9b613`/`4a544a7`) are committed locally but **not pushed**. The Rust Ollama/sandbox wave
    up to `64938ee` IS on `origin`. The standing reminder holds: **ask before pushing** — an agent-initiated
    push was denied in an earlier session, so confirm maintainer OK before clearing this drift.
11. **Rust-only `12-cordyceps` polish + a Rust-only core addition** (`scope: debt`, not yet
    mirrored) — `8bb7734` adds `SubagentTool::with_stream` to the core harness (optional child
    stream sink); `d65ae64` builds on it in the Rust example. **TS/Python/Go have neither the core
    seam nor the example polish.** Decide whether to mirror (file an issue) or keep as a Rust-ahead
    experiment.
12. **#119's `RunStrategy` is a hand-rolled `BoxFut`, not `#[trait_variant::make(Send)]`**
    (`scope: debt`, Rust-only, benign) — converted in #120 so the `custom: HashMap<String, Arc<dyn
    RunStrategy>>` map can exist. No-op on the wire. (The legacy-collaborator-field removal formerly
    tracked here is **RESOLVED by #124**.)
13. **#123 Go `SpanStack` holds `string`, not a typed `SpanId`** (`scope: debt`, intentional) —
    Go's `observability` package imports `sporecore`, so the reverse would be an import cycle; the
    scaffold types must live in `sporecore`. Safe — `SpanStack` is runtime-only, never serialized.
    Sub-note: `charge`'s fallible-result shape is idiomatic per language (Rust `Result`, TS tagged
    union, Go `*BudgetExhausted` nil-ok, Python raises); semantically identical.
14. **#125 review follow-ups — ADDRESSED** (hardening pass `ca0165b`/`ca89df6`/`c96ceed`/`9a9fb12`,
    all four, verifier PASS). Confirmed the `BudgetExhausted`/`partial_output` path is reachable
    end-to-end (no impl gap); made the leaf-cap test discriminating; added bounded-leaf F5/F6
    coverage; fixed a stale Rust doc; made the Continue arms explicit.
15. **#126 `PlanArtifact` type not formally deprecated, only the bridge function** (`scope: debt`,
    minor) — the bridge `plan_artifact_to_task_list` is marked deprecated in all four; the
    `PlanArtifact` **type** is not (still the live `OnPlanCreated` payload — attributing it would
    break the `-D warnings` gate). Documented in prose.
16. **#130 Go fixture comparison uses `jsonEqual`, not byte-equal** (`scope: debt`, benign,
    Go-only) — Go's `encoding/json` emits `0` vs serde's `0.0` for the whole-number `cost_usd`
    float; the established value-normalizing helper is used, same as the consult/escalation replay
    tests. Field order/structure still match the fixture exactly.
17. **#130 default `escalation_mode` is applied at different layers per language** (`scope: debt`,
    benign) — TS defaults the raw config to `autonomous` with the builder setting `surfaceToHuman`;
    Python/Go default the config itself to `surface_to_human`. Each preserves its own pre-#130
    legacy default. A maintainer may wish to harmonize separately.
18. **#129 benign per-language divergences** (`scope: debt`, benign) — (a) Python AC4 asserts
    context preservation by membership rather than message-count growth (Python's
    `NoopContextManager` doesn't append the resumed `FinalResponse`); (b) Go uses `jsonEqual` for
    the `cost_usd` float fixtures (as #16) and an idiomatic `ResumedBudgetContext` constructor name.
    No wire/behavior impact.
19. **#142 benign per-language divergences** (`scope: debt`, benign, all documented + verifier-
    confirmed) — (a) TS `HarnessConfig.projectId` is **optional** (default resolved in the
    `StandardHarness` ctor) vs Rust's required field, avoiding churn across ~29 config literals;
    (b) Python `Path.resolve()` does **not** case-fold on macOS (unlike Rust `fs::canonicalize`), so
    the macOS-gated test asserts stdlib behavior (distinct-but-deterministic ids resolved by the hash
    suffix); `ProjectId`/`WorkspaceId` are `NewType` aliases ⇒ derivation is module functions, not
    methods; (c) Go `ProjectID` lives in the `storage` package and is projected onto the `SessionID`
    axis at the package boundary (storage→sporecore import cycle), `NewStandardHarness` does **not**
    auto-derive the namespace (the builder `.ProjectID(...)`/example does; empty namespace falls back
    to session id), and the Ralph progress/feature-list key literals are defined in both packages and
    pinned equal by `TestRalphKeyLiteralsAgreeAcrossPackages`. All wire/behavior-identical.

_(Former Deviations — HillClimbing/SelfVerifying/Ralph-git-log/MemoryTool/storage-scope/sandbox-
path/extras-mirror/Rust-dyn/compaction-tokens/observability-content stubs — all resolved in prior
loops.)_

## Next Actions
1. **Port #150 — recoverable sandbox violations via `SandboxViolationPolicy` (TS/Python/Go).** Rust
   landed (`64938ee`); the larger of the Ollama/sandbox parity ports and a **breaking default change**
   (path escapes / blocked commands become recoverable feedback). Drive via `/implement 150`; keep the
   typed-violation-to-harness shape and the loud compile error on exhaustive matches (see issue #150 + the
   handoff). Current top of the parity backlog.
2. **Port #148 — Ollama thinking/reasoning models (TS/Python/Go).** Rust landed (`a9c856b`); `/implement 148`.
   While here, confirm whether the Rust-ahead Ollama `read_timeout` fix (`56361b2`) and think-effort levels
   (`c486331`) need their own parity issues or fold into #148.
3. **Port the smaller bugfix-parity issues.** #146 (char-boundary-safe `web_fetch` `start_byte`, Python/Go)
   and #147 (SelfVerifying charge evaluator turns against budget, TS/Python). Both via `/implement`.
4. **Verify + `/close 144`, then push.** PlanExecute budget-resume ports appear landed
   (`2401fee`/`9b1d310`/`b56e852` + Rust docs `a25e6bf`); confirm the AC then close. Also push the 3
   unpushed #149 commits once the maintainer OKs (Deviation #10).
5. **Behind the parity backlog (maintainer call):** `/close 131` + refactor finishers #121/#122/#127/#128;
   parked examples #109/#92 + `web_search` #108/#110; larger features #113/#107/#106, protocol #83–87;
   debt #34→#31→#30 + docs. **#7** (ContextManager migration) would live-wire the rich `assemble`.

**Note:** the hardening cluster (#137–#143) + #139/#141 remain fully closed; the active work is the
**cross-language parity catch-up backlog (#144–#150)** — Rust-first features/bugfixes being ported to the
other three languages issue-by-issue.
