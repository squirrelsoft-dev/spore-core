//! spore-core example 12 — **cordyceps**: a basic plan→execute coding agent.
//!
//! This started as [`04-filesystem-agent`](../../04-filesystem-agent) and grows it
//! into a coding REPL. The differences from 04:
//!
//! 1. it is built from the [`HarnessBuilder::coding_agent`] PRESET (SC-8), which in
//!    one call wires the things a coding agent always needs: a **read-write**
//!    workspace sandbox, the full [`StandardTools::coding_set`]
//!    (read/write/edit/list/grep/find + `bash`), a built-in coding system prompt,
//!    and `EscalationMode::AutoContinue` — so it can actually change code, and a
//!    spent step budget keeps working in-process rather than pausing;
//! 2. the single `harness.run(...)` is wrapped in a **REPL**: build the harness
//!    once, then read a task per line, threading the conversation across turns; and
//! 3. each turn runs the **PlanExecute** strategy instead of a bare `ReAct` loop —
//!    a plan phase turns your prompt into a task list, then an execute phase works
//!    that list to completion (see [`plan_execute_strategy`]).
//!
//! The preset builds on the same `conversational(model)` core, with the same
//! stream-printed `think · turn N` / `act` / `obs` trace — the strategy and the
//! autonomous escalation are what changed. The example then layers its own extras
//! (skills, a richer prompt, the plan-announcer hook) on top of the preset.
//!
//! ## What it shows
//!
//! - **PlanExecute, not bare ReAct.** Each turn's [`Task`] carries a
//!   [`LoopStrategy::PlanExecute`]: a `plan` sub-loop emits a JSON
//!   `{"tasks":[…]}` plan (printed via an `OnPlanCreated` hook — see
//!   [`plan_announcer`]), then an `execute` sub-loop runs each task as its own
//!   ReAct loop, in dependency order, until the whole list is done. The task list
//!   is DURABLE and project-scoped, so a turn that runs out of budget mid-list
//!   doesn't re-plan — a later turn resumes the unfinished tasks. See
//!   [`plan_execute_strategy`].
//! - **A REPL over one harness, one conversation.** The harness is built once and
//!   reused; each line you type is a new [`Task`] on a STABLE [`SessionId`]. We
//!   carry the prior turn's [`SessionState`] forward — `RunResult::Success`
//!   returns the full post-run history losslessly (issue #102), and
//!   [`HarnessRunOptions::with_session_state`] feeds it into the next run, where
//!   the new prompt is appended on top. So the agent remembers the dialogue, not
//!   just what's on disk. (Type `clear` to reset the conversation; the
//!   conversational `ContextManager` compacts it when the window fills.)
//! - **Auto-continue on a spent budget — in the harness, not the consumer.** A
//!   node's step budget is finite, so a long task can spend it mid-flight. Because
//!   the `coding_agent` preset sets `EscalationMode::AutoContinue` (SC-5), the
//!   harness then grants more budget IN-PROCESS and keeps working — up to
//!   [`HarnessBuilder::PRESET_MAX_AUTO_GRANTS`] grants of
//!   [`HarnessBuilder::PRESET_STEPS_PER_GRANT`] steps — re-seeding the stalled
//!   worker so no work is lost. There is no consumer-side drive/resume loop to
//!   hand-roll: `harness.run(..)` returns a terminal result directly. (Past the
//!   cap it ends with `Failure`; the durable task list still holds the rest, so a
//!   follow-up prompt resumes it.)
//! - **A real coding sandbox.** The preset's `WorkspaceScopedSandbox` is scoped
//!   to the workspace ROOT — by default the directory you launched from, so running
//!   at your project root lets the agent work on that project. Unlike 04 it is NOT
//!   read-only, so `write_file` / `edit_file` / `bash` can change files there.
//!   Override the root with `--workspace <path>` or `SPORE_WORKSPACE`.
//! - **Live narration via `send_message`.** `coding_set()` includes the
//!   `send_message` tool, which surfaces an out-of-band line to the user. The
//!   system prompt tells the agent the user only sees these messages plus the
//!   final answer, so it should narrate each step in one short sentence — called
//!   in parallel with the tool doing the work. The harness turns each call into a
//!   [`HarnessStreamEvent::UserMessage`] we print as a `💬` line.
//! - **Skills (progressive disclosure).** Drop a `SKILL.md` under `skills/<name>/`
//!   (or `.spore/skills/`), and the agent sees a cheap manifest (name +
//!   description) of it every turn, pulling the full body into context only when
//!   it calls the `load_skill` tool — or when you load it yourself with `/<name>`.
//!   It follows the [Agent Skills spec](https://agentskills.io/specification),
//!   now productionized in the harness (#115 / SC-26): `HarnessBuilder::skills`
//!   takes a `spore_core::SkillCatalog`, registers `load_skill`, and injects the
//!   manifest + active bodies STRUCTURALLY via the rich `ContextSources` seam — no
//!   example-side context-manager wrapper.
//! - **Esc-to-abort, without losing context.** A run executes with the terminal
//!   in raw mode and a background key watcher; pressing Esc drops the
//!   `harness.run(..)` future, cancelling the in-flight turn at its next await
//!   point, and drops back to the REPL (see [`run_abortable`]). A dropped future never
//!   returns its `session_state`, so the turn's progress would be lost — instead
//!   we mirror the turn from the stream as it happens (each [`HarnessStreamEvent`]
//!   carries the `call_id` that pairs a result to its call) and, on abort, splice
//!   that partial transcript onto the prior history. So "continue" still works.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull gemma4:e4b
//! # operates on the current directory by default:
//! cargo run --manifest-path examples/rust/12-cordyceps/Cargo.toml
//! # or point it somewhere explicit:
//! cargo run --manifest-path examples/rust/12-cordyceps/Cargo.toml -- --workspace /path/to/project
//! ```

use std::io::Write;
use std::sync::{Arc, Mutex};

use spore_core::{
    AgentRef, BudgetExhaustedBehavior, BudgetPolicy, Content, FunctionHook, Harness, HarnessBuilder,
    HarnessRunOptions, HarnessStreamEvent, HookChain, HookContext, HookDecision, HookEvent,
    LoopStrategy, Message, OllamaModelInterface, PlanExecuteConfig, ReactConfig, Role, RunResult,
    SessionId, SessionState, StandardHookChain, Task, ToolCall, ToolResult, ToolsetRef,
};

const SYSTEM_PROMPT: &str = "You are a coding agent working inside a sandboxed workspace directory. \
     Explore with list_dir, read_file, grep, and find_files; create and change files with \
     write_file and edit_file; run commands with bash. Use `.` and relative paths only. \
     Act using tools — do not just describe what you would do. (The one exception: when you are \
     asked to PRODUCE A PLAN, reply with the requested JSON plan object directly, with no tool \
     calls in that turn.) When the task is done, reply with a short summary of what you changed. \
     \
     The user CANNOT see your reasoning or your tool calls — they only see the messages you \
     send with the `send_message` tool and your final reply. So keep the user in the loop: \
     before (or as) you act, call `send_message` with one short sentence saying what you are \
     about to do, e.g. \"Reading the Cargo.toml to find the entry point.\" Call `send_message` \
     in PARALLEL with the tool that does the work — emit both in the same turn — so narration \
     never costs an extra round trip. Keep each message to a single short sentence. \
     \
     You may have SKILLS available — reusable, named procedures listed under AVAILABLE SKILLS \
     in your context (each as `name: description`). When the user's request matches a skill's \
     description, call the `load_skill` tool with that skill's name BEFORE you start, then \
     follow the full procedure it injects. You can load more than one.";

/// Per-loop ReAct step budget for EACH execute-phase task (04 used 8; a coding
/// task wants more room to explore, edit, and verify). The plan phase runs under
/// its own, smaller budget (`PLAN_STEPS`).
const MAX_STEPS: u32 = 25;

/// Per-loop ReAct step budget for the PLAN phase — a few turns for the planner to
/// look around (read_file / grep / list_dir) before it emits its JSON plan.
const PLAN_STEPS: u32 = 12;

/// Compaction window, in tokens — the size the harness believes the model's
/// context is, and the budget it compacts against. gemma4's real window is 256K,
/// but the harness's #141 resolver only falls back to a static table that maps
/// every `gemma*` id to 8_192 (and Ollama's `/api/show` discovery is best-effort
/// and timing-dependent). So we size it explicitly to use the model's real
/// headroom instead of compacting ~30× too early. Override for a smaller model
/// with `--context-window <tokens>` / `SPORE_CONTEXT_WINDOW` — the value is used
/// as-is and is NOT clamped to the model's true window, so don't set it larger
/// than the model can actually hold. We apply it with ONE call —
/// [`OllamaModelInterface::with_context_window`] (SC-4): that sets Ollama's
/// `num_ctx` (sizing the KV cache, so longer prompts aren't silently truncated)
/// AND the window reported by `provider()`, which the compaction budget
/// auto-derives — so a larger window costs proportionally more memory at
/// model-load time.
const DEFAULT_CONTEXT_WINDOW: u32 = 256_000;

/// Fraction of the window at which the harness compacts. `should_compact` fires
/// when `tokens_used / window >= threshold`, so 0.80 means compact at 80% of the
/// window (e.g. ~204_800 of 256_000), leaving headroom for the turn that trips
/// it. This is `CompactionConfig`'s own default; we name it for the status line.
const COMPACT_THRESHOLD: f32 = 0.80;

// ANSI styling for the REPL trace. The `send_message` narration is the group
// SECTION HEADER — bright white and flush left, so it stands out as the one line
// the user is meant to read. The think / act / obs detail under it is dim and
// indented so the mechanical trace recedes and doesn't distract.
const HEADER: &str = "\x1b[1;97m"; // bold bright white
const MUTED: &str = "\x1b[90m"; // gray (bright black)
const ERR: &str = "\x1b[31m"; // red — tool errors still want to be noticed
const RESET: &str = "\x1b[0m";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // A tool-capable model is required — a small model that only narrates tool
    // use (e.g. llama3.2 3B) will never act. Default to gemma4:e4b or better.
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "gemma4:e4b".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());
    let context_window: u32 = arg_value(&args, "--context-window")
        .or_else(|| std::env::var("SPORE_CONTEXT_WINDOW").ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CONTEXT_WINDOW);

    // The agent operates inside a writable workspace root. By DEFAULT this is the
    // directory you launched from (the current working directory) — so running
    // from your project root points the agent at that project. Override with
    // `--workspace <path>` or `SPORE_WORKSPACE`. The sandbox requires a canonical,
    // existing root, so we create it if missing and canonicalize it.
    let workspace_root = match arg_value(&args, "--workspace")
        .or_else(|| std::env::var("SPORE_WORKSPACE").ok())
    {
        Some(p) => std::path::PathBuf::from(p),
        None => std::env::current_dir()?,
    };
    std::fs::create_dir_all(&workspace_root)?;
    let workspace_root = std::fs::canonicalize(&workspace_root)?;

    // --- Skills (the Agent Skills spec, now productionized in the harness) ----
    // Discover `SKILL.md` files (bundled with the example + `.spore/skills` in the
    // workspace + `~/.spore/skills`). `HarnessBuilder::skills` (below) registers the
    // catalog AND the `load_skill` tool, and the harness injects the manifest (every
    // turn) + each ACTIVE skill's full body (sticky) into the model's context
    // STRUCTURALLY via the rich `ContextSources` seam (#115 / SC-26) — no example-side
    // context-manager wrapper, no second model handle. A skill goes active when the
    // agent calls `load_skill` (it should, when a request matches a skill's
    // description) or when you load it yourself with `/<name>`. The `Arc<SkillCatalog>`
    // is shared between the harness and this REPL, so both see the same active set.
    let catalog = spore_core::SkillCatalog::discover(
        &[std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("skills")],
        &workspace_root,
    );

    // Build the harness via the `coding_agent` PRESET (SC-8). The preset wires the
    // bits a coding agent always needs — a read-WRITE `WorkspaceScopedSandbox`
    // rooted at `workspace_root`, the full coding tool catalogue
    // (`StandardTools::coding_set`), a built-in coding system prompt, and
    // `EscalationMode::AutoContinue` so a spent step budget keeps working IN-PROCESS
    // instead of pausing (no hand-rolled drive loop; SC-5). Sizing the model's
    // window with `with_context_window` (as above) is the only window knob.
    //
    // On top of the preset we layer this example's extras: the skill catalog +
    // `load_skill` tool (via `.skills`, which also injects the manifest/active
    // bodies structurally — no context-manager override needed anymore, #115), a
    // RICHER system prompt (it overrides the built-in to add the skills contract
    // AND the plan-phase "reply with JSON only" exception PlanExecute needs), and
    // the plan-announcer hook. Built once and reused for every turn.
    let model = OllamaModelInterface::with_base_url(&model_id, base_url)
        .with_context_window(context_window);
    let harness = HarnessBuilder::coding_agent(model, workspace_root.clone())?
        .skills(catalog.clone())
        .system_prompt(SYSTEM_PROMPT)
        // Surface the plan to the user the moment it's captured (OnPlanCreated).
        .hooks(plan_announcer())
        .build();

    println!("spore-core — plan→execute coding agent");
    println!("model     : {model_id}");
    println!("strategy  : plan (≤{PLAN_STEPS} steps) → execute (≤{MAX_STEPS} steps/task)");
    println!(
        "auto-cont : up to {} grants × {} steps in-process when a step budget is spent (preset)",
        HarnessBuilder::PRESET_MAX_AUTO_GRANTS,
        HarnessBuilder::PRESET_STEPS_PER_GRANT,
    );
    println!(
        "context   : {context_window} tokens (num_ctx sent to Ollama; compact at {:.0}% → {} tokens)",
        COMPACT_THRESHOLD * 100.0,
        (context_window as f32 * COMPACT_THRESHOLD) as u32,
    );
    println!("workspace : {}", workspace_root.display());
    println!(
        "tools     : read_file, write_file, edit_file, list_dir, grep, find_files, bash, load_skill, …"
    );
    println!(
        "skills    : {} discovered — load with /<name>, or the agent loads via load_skill (/skills to list)",
        catalog.entries().len()
    );
    println!("Type a coding task and press enter. Esc aborts a running task; Ctrl-D quits.\n");

    // One conversation for the whole REPL. We keep a stable SessionId and carry
    // the prior turn's SessionState forward: `RunResult::Success` returns the
    // post-run history losslessly (issue #102 — user turns, assistant tool-call
    // turns, tool results, final text), and `with_session_state` feeds it into
    // the next run, where the new prompt is appended on top. So the agent now
    // remembers what was said earlier, not just what's on disk. (The
    // conversational ContextManager compacts the history when the window fills.)
    let session_id = SessionId::generate();
    let mut history: Option<SessionState> = None;

    while let Some(line) = read_prompt() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // `clear` wipes the in-memory CONVERSATION back to a clean slate — and the
        // active-skill set with it, since both are conversation-scoped context. The
        // workspace on disk AND the durable (project-scoped) task list are left
        // intact — so the agent keeps resuming any unfinished plan; `clear` only
        // forgets the dialogue.
        if trimmed.eq_ignore_ascii_case("clear") {
            history = None;
            catalog.clear_active();
            println!("(conversation cleared)\n");
            continue;
        }

        // Slash commands resolve to the task text we actually run (if any).
        // `/skills` lists the catalog; `/<name>` loads a skill yourself (the
        // host-driven path); `/<name> <task>` loads it then runs <task> in one go.
        let prompt: String = if let Some(rest) = trimmed.strip_prefix('/') {
            let (cmd, inline) = rest
                .split_once(char::is_whitespace)
                .map(|(c, r)| (c, r.trim()))
                .unwrap_or((rest, ""));
            if cmd.eq_ignore_ascii_case("skills") {
                print_skills(&catalog);
                continue;
            }
            if catalog.activate(cmd) {
                println!("✓ loaded skill '{cmd}' — active for this conversation.\n");
                if inline.is_empty() {
                    continue; // just loaded; wait for the next prompt
                }
                inline.to_string()
            } else {
                eprintln!("unknown command '/{cmd}'. Try /skills to list available skills.\n");
                continue;
            }
        } else {
            trimmed.to_string()
        };
        // Each REPL turn appends to the SAME conversation and runs as a
        // PlanExecute task: the planner turns your prompt into a JSON task list,
        // then each task runs its own ReAct loop in dependency order. The list is
        // DURABLE and project-scoped, so if a turn runs out of budget mid-list, a
        // later turn resumes the unfinished tasks instead of re-planning. Files the
        // agent wrote earlier are still on disk AND the dialogue carries forward,
        // so it can build on both.
        let task = Task::new(prompt.clone(), session_id.clone(), plan_execute_strategy());

        // Mirror this turn's conversation as it streams. On a clean finish we use
        // the harness's own lossless `session_state`; but an Esc-aborted run is
        // dropped before it can return one, so we reconstruct the partial turn
        // from the stream (`call_id` ties each result to its call) and splice it
        // onto the prior history — otherwise the aborted work would be forgotten.
        let turn_msgs: Arc<Mutex<Vec<Message>>> = Arc::new(Mutex::new(Vec::new()));

        // The stream sink prints each turn (Think) and each tool call + result
        // (Act / Observe) AND mirrors the turn into `turn_msgs` for the abort path.
        // It's built as a shareable `Arc` so the SAME sink feeds both the initial
        // run and any budget-grant resumes (see `drive`). Lines END WITH `\r\n`,
        // not `\n`: the run executes with the terminal in raw mode (so a bare Esc
        // can abort it), and raw mode turns off the kernel's `\n`→`\r\n`
        // translation — without the `\r` the trace would stair-step to the right.
        // The stray `\r` is harmless when raw mode isn't active (the non-TTY
        // fallback in `run_abortable`).
        let mirror = turn_msgs.clone();
        let sink: Arc<dyn Fn(HarnessStreamEvent) + Send + Sync> =
            Arc::new(move |event: HarnessStreamEvent| match event {
                // The agent's running narration via `send_message` is the section
                // header: bright white and flush left. This — plus the final
                // answer — is all the user is really meant to read. (Not recorded:
                // the send_message tool already appears as a tool call + result.)
                HarnessStreamEvent::UserMessage { content, .. } => {
                    print!("{HEADER}💬 {content}{RESET}\r\n");
                }
                // Everything else is muted, indented detail beneath that header.
                HarnessStreamEvent::TurnStart { turn, .. } => {
                    print!("{MUTED}   think · turn {turn}{RESET}\r\n");
                }
                HarnessStreamEvent::ToolCall {
                    call_id,
                    name,
                    args,
                    ..
                } => {
                    print!("{MUTED}   act → {name}({args}){RESET}\r\n");
                    mirror.lock().unwrap().push(Message {
                        role: Role::Assistant,
                        content: Content::ToolCall(ToolCall {
                            id: call_id,
                            name,
                            input: args,
                        }),
                    });
                }
                HarnessStreamEvent::ToolResult {
                    call_id,
                    is_error,
                    content,
                    ..
                } => {
                    let (color, tag) = if is_error {
                        (ERR, "obs(err)")
                    } else {
                        (MUTED, "obs")
                    };
                    print!("{color}   {tag} → {}{RESET}\r\n", truncate(&content, 200));
                    mirror.lock().unwrap().push(Message {
                        role: Role::Tool,
                        content: Content::ToolResult(ToolResult {
                            tool_use_id: call_id,
                            content,
                            is_error,
                        }),
                    });
                }
                _ => {}
            });
        // Assign `on_stream` directly (it's a public `Option<StreamSink>`) rather
        // than `with_stream`, which wants a bare `Fn`. One run carries the whole
        // turn now: `EscalationMode::AutoContinue` (from the `coding_agent` preset)
        // grants more budget IN-PROCESS when a step budget is spent, so there are no
        // consumer-side resumes to feed — the same sink streams every grant.
        let mut options = HarnessRunOptions::new(task);
        options.on_stream = Some(sink);
        // Carry the running conversation into this turn (no-op on the first).
        // CLONE rather than take: an aborted run never hands back a post-run
        // state, so keeping `history` intact lets us rebuild from it below.
        if let Some(state) = &history {
            options = options.with_session_state(state.clone());
        }

        // Run the turn to a terminal result, Esc-abortable throughout. The preset's
        // AutoContinue works a spent step budget to completion in-process (capped at
        // PRESET_MAX_AUTO_GRANTS), so the old hand-rolled budget-grant drive loop is
        // gone. `None` ⇒ the user aborted with Esc.
        match run_abortable(harness.run(options)).await {
            None => {
                // Reconstruct the aborted turn so "continue" still has context:
                // prior history + this turn's user prompt + the tool calls/results
                // that ran before the abort. (The harness would have appended the
                // user prompt itself; we mirror that since its state was dropped.)
                let mut partial = std::mem::take(&mut *turn_msgs.lock().unwrap());
                // If Esc landed mid-tool we may have a tool CALL with no result;
                // a dangling tool_use makes the next request malformed, so drop it.
                while matches!(
                    partial.last(),
                    Some(Message {
                        content: Content::ToolCall(_),
                        ..
                    })
                ) {
                    partial.pop();
                }
                if !partial.is_empty() {
                    let mut state = history.take().unwrap_or_default();
                    state.messages.push(Message {
                        role: Role::User,
                        content: Content::Text { text: prompt },
                    });
                    state.messages.extend(partial);
                    history = Some(state);
                }
                eprintln!("\n(aborted — back to the prompt)\n");
            }
            Some(RunResult::Success {
                output,
                turns,
                session_state,
                ..
            }) => {
                history = Some(session_state); // remember it for the next turn
                println!("\nanswer ({turns} turn(s)): {output}\n");
            }
            Some(RunResult::Failure {
                reason,
                session_state,
                ..
            }) => {
                // A budget-exhausted Failure here means AutoContinue hit its grant
                // cap (PRESET_MAX_AUTO_GRANTS) before the plan finished. Keep the
                // partial history — the durable, project-scoped task list still holds
                // the remaining work, so another prompt resumes it.
                history = Some(session_state);
                eprintln!(
                    "\nrun did not finish: {reason:?}\n  send another prompt to keep going \
                     (or `clear` to reset).\n"
                );
            }
            Some(RunResult::WaitingForHuman { state, .. }) => {
                // With AutoContinue a spent step budget no longer pauses here (the
                // harness grants more in-process, then ends with Failure above), so a
                // pause here is an unexpected human request. Keep the conversation so
                // a follow-up prompt can continue; the durable task list survives.
                history = Some(state.session_state.clone());
                eprintln!("\n⏸ run paused awaiting input — send another prompt to continue.\n");
            }
            Some(RunResult::Consult { state, .. }) | Some(RunResult::Escalate { state, .. }) => {
                // Not expected in this single-agent example, but handle it cleanly
                // rather than dumping the paused state.
                history = Some(state.session_state.clone());
                eprintln!("\n⏸ run paused (consult/escalate) — send another prompt to continue.\n");
            }
        }
    }

    println!("\nbye.");
    Ok(())
}

/// The strategy each REPL turn runs: **PlanExecute** — a plan phase produces a
/// JSON task list, then an execute phase runs each task as its own ReAct loop, in
/// dependency order, until the whole list is `Completed`.
///
/// - **plan** — a ReAct sub-loop (≤ [`PLAN_STEPS`]) that may look around with the
///   read tools, then emits the `{"tasks":[…],"rationale":…}` plan. The harness
///   seeds the "respond with a single JSON plan" directive itself; we only supply
///   the slot. The leaf carries NO `output` schema: SC-1 lets a structured slot
///   omit it (an absent schema is treated as accept-all), so no registry stamp is
///   needed just to pass startup validation — the plan phase's own "respond with a
///   single JSON plan" directive drives the format.
/// - **execute** — a bare ReAct leaf (≤ [`MAX_STEPS`] per task). The executor
///   walks the durable task list, running this loop once per ready task.
///
/// Both leaves carry empty agent/toolset handles, so they resolve to the
/// conversational harness's default agent + `coding_set()` toolset. `Escalate` is
/// the same budget-exhausted behavior `ReactConfig::per_loop` already uses.
fn plan_execute_strategy() -> LoopStrategy {
    LoopStrategy::PlanExecute(PlanExecuteConfig {
        plan: Box::new(LoopStrategy::ReAct(ReactConfig {
            budget: BudgetPolicy::PerLoop { value: PLAN_STEPS },
            behavior: BudgetExhaustedBehavior::Escalate,
            agent: AgentRef(String::new()),
            toolset: ToolsetRef(String::new()),
            output: None,
            system_prompt: None,
        })),
        execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(MAX_STEPS))),
        plan_model: None,
        behavior: BudgetExhaustedBehavior::Escalate,
    })
}

/// A hook chain that prints the plan the moment it's captured (the `OnPlanCreated`
/// lifecycle event), so the user sees the task list before the execute phase
/// starts grinding through it. Returned as `Arc<dyn HookChain>` for
/// [`HarnessBuilder::hooks`].
///
/// Lines end with `\r\n` for the same reason the stream trace does — the run is in
/// raw mode while this fires (see [`run_abortable`]).
fn plan_announcer() -> Arc<dyn HookChain> {
    let chain = StandardHookChain::new();
    let _ = chain.register(Arc::new(FunctionHook::new(
        "print-plan",
        vec![HookEvent::OnPlanCreated],
        |ctx| {
            if let HookContext::OnPlanCreated { plan, .. } = ctx {
                print!("{HEADER}📋 plan ({} task(s)):{RESET}\r\n", plan.tasks.len());
                for (i, step) in plan.tasks.iter().enumerate() {
                    print!("{MUTED}   {}. {step}{RESET}\r\n", i + 1);
                }
            }
            Ok(HookDecision::Continue)
        },
    )));
    Arc::new(chain)
}

/// Run one terminal-producing future (`harness.run` or `harness.resume`) with
/// **Esc-to-abort** armed. Returns `Some(result)` if it finished on its own, or
/// `None` if the user pressed Esc (the future is dropped, cancelling the in-flight
/// turn at its next await point).
///
/// How it works: put the terminal in raw mode so a single Esc keypress is
/// readable without an Enter, then `select!` the future against a background
/// watcher that blocks on key events. If Esc wins, `fut` is dropped — which
/// cancels the in-flight turn — and we return `None`. Raw mode is always restored
/// before returning. If raw mode can't be enabled (e.g. stdin isn't a TTY), we
/// just await the future without the watcher.
async fn run_abortable<F>(fut: F) -> Option<RunResult>
where
    F: std::future::Future<Output = RunResult>,
{
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    if crossterm::terminal::enable_raw_mode().is_err() {
        return Some(fut.await);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let mut watcher = {
        let stop = stop.clone();
        tokio::task::spawn_blocking(move || watch_for_escape(&stop))
    };

    tokio::pin!(fut);
    let result = tokio::select! {
        r = &mut fut => {
            // The run finished first — tell the watcher to stop and join it so it
            // releases stdin before the REPL reads the next prompt.
            stop.store(true, Ordering::Relaxed);
            let _ = (&mut watcher).await;
            Some(r)
        }
        _ = &mut watcher => {
            // Esc was pressed. Dropping `fut` (the other select branch) cancels
            // the turn. Prior history is untouched.
            None
        }
    };

    let _ = crossterm::terminal::disable_raw_mode();
    result
}

/// Block on a dedicated thread watching for a single Esc keypress. Returns when
/// Esc is seen, or when `stop` is set (the run finished on its own). Transient
/// poll errors are ignored so a hiccup never spuriously aborts a healthy run.
fn watch_for_escape(stop: &std::sync::atomic::AtomicBool) {
    use crossterm::event::{poll, read, Event, KeyCode};
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    let tick = Duration::from_millis(80);
    while !stop.load(Ordering::Relaxed) {
        match poll(tick) {
            Ok(true) => {
                if let Ok(Event::Key(key)) = read() {
                    if key.code == KeyCode::Esc {
                        return;
                    }
                }
            }
            Ok(false) => {} // timed out — re-check `stop` and poll again
            Err(_) => std::thread::sleep(tick),
        }
    }
}

/// List the discovered skills and which are currently active — the `/skills`
/// command. `●` marks an active skill (its full body is in context every turn);
/// `○` marks one the agent (or you, via `/<name>`) can still load.
fn print_skills(catalog: &spore_core::SkillCatalog) {
    if catalog.is_empty() {
        println!(
            "no skills found. Add one at skills/<name>/SKILL.md (next to this example) or \
             .spore/skills/<name>/SKILL.md in your workspace, then restart.\n"
        );
        return;
    }
    let active = catalog.active();
    println!("skills:");
    for e in catalog.entries() {
        let mark = if active.contains(&e.name) {
            "● active  "
        } else {
            "○ loadable"
        };
        println!("  {mark}  {} — {}", e.name, e.description);
    }
    println!(
        "\nLoad one yourself with /<name>, or just describe your task and the agent loads \
         what it needs.\n"
    );
}

/// Read one task line from the REPL. `Some(line)` to run; `None` on EOF (Ctrl-D),
/// which quits.
fn read_prompt() -> Option<String> {
    print!("code> ");
    let _ = std::io::stdout().flush();
    let mut buf = String::new();
    match std::io::stdin().read_line(&mut buf) {
        Ok(0) => None, // EOF
        Ok(_) => Some(buf.trim_end_matches(['\n', '\r']).to_string()),
        Err(_) => None,
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Keep observe lines readable — file contents can be long.
fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max {
        s
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}
