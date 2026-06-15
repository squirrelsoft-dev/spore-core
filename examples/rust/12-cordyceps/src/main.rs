//! spore-core example 12 — **cordyceps**: a basic ReAct coding agent.
//!
//! This is [`04-filesystem-agent`](../../04-filesystem-agent) with two changes:
//!
//! 1. the workspace sandbox is **read-write** and the agent gets the full
//!    [`StandardTools::coding_set`] (read/write/edit/list/grep/find + `bash`), so
//!    it can actually change code — not just summarize files; and
//! 2. the single `harness.run(...)` is wrapped in a **REPL**: build the harness
//!    once, then read a task per line, threading the conversation across turns.
//!
//! Everything else is 04 verbatim — the same `conversational(model)` builder, the
//! same `ReAct` loop, the same stream-printed `think · turn N` / `act` / `obs`
//! trace. The thesis is the same as 04's: **the harness doesn't change** — here
//! we only widen the toolset and drive it in a loop.
//!
//! ## What it shows
//!
//! - **A REPL over one harness, one conversation.** The harness is built once and
//!   reused; each line you type is a new [`Task`] on a STABLE [`SessionId`]. We
//!   carry the prior turn's [`SessionState`] forward — `RunResult::Success`
//!   returns the full post-run history losslessly (issue #102), and
//!   [`HarnessRunOptions::with_session_state`] feeds it into the next run, where
//!   the new prompt is appended on top. So the agent remembers the dialogue, not
//!   just what's on disk. (Type `clear` to reset the conversation; the
//!   conversational `ContextManager` compacts it when the window fills.)
//! - **A real coding sandbox.** Catalogue file tools go through a
//!   [`WorkspaceScopedSandbox`] scoped to the workspace ROOT — by default the
//!   directory you launched from, so running at your project root lets the agent
//!   work on that project. Unlike 04 it is NOT read-only, so `write_file` /
//!   `edit_file` / `bash` can change files there. Override the root with
//!   `--workspace <path>` or `SPORE_WORKSPACE`.
//! - **Live narration via `send_message`.** `coding_set()` includes the
//!   `send_message` tool, which surfaces an out-of-band line to the user. The
//!   system prompt tells the agent the user only sees these messages plus the
//!   final answer, so it should narrate each step in one short sentence — called
//!   in parallel with the tool doing the work. The harness turns each call into a
//!   [`HarnessStreamEvent::UserMessage`] we print as a `💬` line.
//! - **Esc-to-abort, without losing context.** A run executes with the terminal
//!   in raw mode and a background key watcher; pressing Esc drops the
//!   `harness.run(..)` future, cancelling the in-flight turn at its next await
//!   point, and drops back to the REPL (see [`run_turn`]). A dropped future never
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
    CompactionConfig, Content, Harness, HarnessBuilder, HarnessContextManagerExt,
    HarnessRunOptions, HarnessStreamEvent, LoopStrategy, Message, NullCacheProvider,
    OllamaModelInterface, ReactConfig, Role, RunResult, SessionId, SessionState,
    StandardContextManager, StandardTools, Task, ToolCall, ToolResult, WorkspaceConfig,
    WorkspaceScopedSandbox,
};

const SYSTEM_PROMPT: &str = "You are a coding agent working inside a sandboxed workspace directory. \
     Explore with list_dir, read_file, grep, and find_files; create and change files with \
     write_file and edit_file; run commands with bash. Use `.` and relative paths only. \
     Act using tools — do not just describe what you would do. When the task is done, reply with \
     a short summary of what you changed. \
     \
     The user CANNOT see your reasoning or your tool calls — they only see the messages you \
     send with the `send_message` tool and your final reply. So keep the user in the loop: \
     before (or as) you act, call `send_message` with one short sentence saying what you are \
     about to do, e.g. \"Reading the Cargo.toml to find the entry point.\" Call `send_message` \
     in PARALLEL with the tool that does the work — emit both in the same turn — so narration \
     never costs an extra round trip. Keep each message to a single short sentence.";

/// Per-loop ReAct step budget for one REPL turn (04 used 8; a coding task wants
/// more room to explore, edit, and verify).
const MAX_STEPS: u32 = 25;

/// Compaction window, in tokens — the size the harness believes the model's
/// context is, and the budget it compacts against. gemma4's real window is 256K,
/// but the harness's #141 resolver only falls back to a static table that maps
/// every `gemma*` id to 8_192 (and Ollama's `/api/show` discovery is best-effort
/// and timing-dependent). So we set it explicitly to use the model's real
/// headroom instead of compacting ~30× too early. Override for a smaller model
/// with `--context-window <tokens>` / `SPORE_CONTEXT_WINDOW` — the value is used
/// as-is and is NOT clamped to the model's true window, so don't set it larger
/// than the model can actually hold.
const DEFAULT_CONTEXT_WINDOW: u32 = 256_000;

/// Fraction of the window at which the harness compacts. `should_compact` fires
/// when `tokens_used / window >= threshold`, so 0.80 means compact at 80% of the
/// window (e.g. ~204_800 of 256_000), leaving headroom for the turn that trips
/// it. This is `CompactionConfig`'s own default; we name it for clarity.
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

    // The SAME conversational ReAct harness as 04 — the differences are a
    // read-WRITE sandbox, the full coding catalogue, and a context window sized
    // for the model (below). Built once and reused for every REPL turn.
    let sandbox = WorkspaceScopedSandbox::new(WorkspaceConfig::scoped(workspace_root.clone()))?;

    // `conversational` installs a context manager whose compaction window
    // resolves to the gemma static fallback (8K). Override it with one configured
    // for the model's real window so the persisted conversation isn't compacted
    // prematurely. The context manager needs its own model handle (it uses it
    // only for compaction summarization), so build a second cheap instance —
    // `OllamaModelInterface` is config-only and isn't `Clone`.
    // `context_length` is the model's TOTAL window; compaction fires earlier, at
    // `threshold × window` (should_compact: used/window >= threshold), leaving
    // headroom for the turn that crosses the line. 0.80 is the default — set it
    // explicitly here so the 80% trigger is visible, not buried in a default.
    let context_manager = Arc::new(StandardContextManager::new(
        Arc::new(OllamaModelInterface::with_base_url(&model_id, base_url.clone())),
        Arc::new(NullCacheProvider),
        CompactionConfig {
            context_length: Some(context_window),
            threshold: COMPACT_THRESHOLD,
            ..Default::default()
        },
    ))
    .into_harness_adapter();

    let model = OllamaModelInterface::with_base_url(&model_id, base_url);
    let harness = HarnessBuilder::conversational(model)
        .sandbox(Arc::new(sandbox))
        .tools(StandardTools::coding_set())
        .system_prompt(SYSTEM_PROMPT)
        .context_manager(context_manager)
        .build();

    println!("spore-core — basic ReAct coding agent");
    println!("model     : {model_id}");
    println!(
        "context   : {context_window} tokens (compact at {:.0}% → {} tokens)",
        COMPACT_THRESHOLD * 100.0,
        (context_window as f32 * COMPACT_THRESHOLD) as u32,
    );
    println!("workspace : {}", workspace_root.display());
    println!("tools     : read_file, write_file, edit_file, list_dir, grep, find_files, bash, …");
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

    while let Some(prompt) = read_prompt() {
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            continue;
        }
        // `clear` wipes the conversation back to a clean slate (the workspace on
        // disk is untouched) — handy now that history persists across turns.
        if trimmed.eq_ignore_ascii_case("clear") {
            history = None;
            println!("(conversation cleared)\n");
            continue;
        }
        // Each REPL turn appends to the SAME conversation. Files the agent wrote
        // on a previous turn are still on disk AND the dialogue carries forward,
        // so it can build on both.
        let task = Task::new(
            prompt.clone(),
            session_id.clone(),
            LoopStrategy::ReAct(ReactConfig::per_loop(MAX_STEPS)),
        );

        // Mirror this turn's conversation as it streams. On a clean finish we use
        // the harness's own lossless `session_state`; but an Esc-aborted run is
        // dropped before it can return one, so we reconstruct the partial turn
        // from the stream (`call_id` ties each result to its call) and splice it
        // onto the prior history — otherwise the aborted work would be forgotten.
        let turn_msgs: Arc<Mutex<Vec<Message>>> = Arc::new(Mutex::new(Vec::new()));

        // Print each turn (Think) and each catalogue tool call + result (Act /
        // Observe). Lines END WITH `\r\n`, not `\n`: the run executes with the
        // terminal in raw mode (so a bare Esc can abort it), and raw mode turns
        // off the kernel's `\n`→`\r\n` translation — without the `\r` the trace
        // would stair-step to the right. The stray `\r` is harmless when raw mode
        // isn't active (the non-TTY fallback in `run_turn`).
        let sink = turn_msgs.clone();
        let mut options = HarnessRunOptions::new(task).with_stream(Box::new(
            move |event: HarnessStreamEvent| match event {
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
                    sink.lock().unwrap().push(Message {
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
                    sink.lock().unwrap().push(Message {
                        role: Role::Tool,
                        content: Content::ToolResult(ToolResult {
                            tool_use_id: call_id,
                            content,
                            is_error,
                        }),
                    });
                }
                _ => {}
            },
        ));
        // Carry the running conversation into this turn (no-op on the first).
        // CLONE rather than take: an aborted run never hands back a post-run
        // state, so keeping `history` intact lets us rebuild from it below.
        if let Some(state) = &history {
            options = options.with_session_state(state.clone());
        }

        // `run_turn` runs with Esc-to-abort armed and returns `None` if aborted.
        match run_turn(&harness, options).await {
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
                // Keep the partial history so a follow-up turn can continue.
                history = Some(session_state);
                eprintln!("\nrun did not succeed: {reason:?}\n");
            }
            Some(other) => {
                eprintln!("\nrun did not succeed: {other:?}\n");
            }
        }
    }

    println!("\nbye.");
    Ok(())
}

/// Run one task with **Esc-to-abort** armed. Returns `Some(result)` if the run
/// finished on its own, or `None` if the user pressed Esc (the run is cancelled
/// and we fall back to the REPL).
///
/// How it works: put the terminal in raw mode so a single Esc keypress is
/// readable without an Enter, then `select!` the harness run against a background
/// watcher that blocks on key events. If Esc wins, the `harness.run(..)` future
/// is dropped — which cancels the in-flight turn at its next await point — and we
/// return `None`. Raw mode is always restored before returning. If raw mode
/// can't be enabled (e.g. stdin isn't a TTY), we just run without the watcher.
async fn run_turn(harness: &dyn Harness, options: HarnessRunOptions) -> Option<RunResult> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    if crossterm::terminal::enable_raw_mode().is_err() {
        return Some(harness.run(options).await);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let mut watcher = {
        let stop = stop.clone();
        tokio::task::spawn_blocking(move || watch_for_escape(&stop))
    };

    let result = tokio::select! {
        r = harness.run(options) => {
            // The run finished first — tell the watcher to stop and join it so it
            // releases stdin before the REPL reads the next prompt.
            stop.store(true, Ordering::Relaxed);
            let _ = (&mut watcher).await;
            Some(r)
        }
        _ = &mut watcher => {
            // Esc was pressed. Dropping the `harness.run` future (it's the other
            // select branch) cancels the turn. Prior history is untouched.
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
