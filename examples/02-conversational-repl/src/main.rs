//! spore-core example 02 — conversational REPL.
//!
//! Takes example 01 one step further: an interactive chat loop where the agent
//! remembers what you said earlier in the session. Same `conversational(model)`
//! harness as 01 — the new idea here is *conversation continuity across runs*.
//!
//! ## How memory works
//!
//! The harness is stateless between `run()` calls: each call takes an optional
//! starting [`SessionState`] (the message history) and drives one task to a
//! final response. As of issue #102, `RunResult::Success` now hands the
//! post-run [`SessionState`] back, so the caller resumes the conversation
//! LOSSLESSLY — no reconstruction. After each turn we feed the returned
//! `session_state` straight into the next run via
//! [`HarnessRunOptions::with_session_state`]. The harness appends the new user
//! line on top of that history before calling the model, so the model sees the
//! whole conversation and can refer back to it.
//!
//! This works for tool-using agents too: the returned `session_state` carries
//! the tool-call and tool-result messages the loop produced, which the old
//! "reconstruct history from `output`" trick could not recover. (The previous
//! version of this example documented that as an honest limitation — issue #102
//! retired it.)
//!
//! Prefer it hands-free? Wire a `SessionStore` and call
//! `HarnessBuilder::auto_persist_sessions(true)`: the harness then auto-loads
//! and auto-persists by `session_id`, so you reuse the id instead of threading
//! state at all (great for a web service that resumes across restarts).
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull llama3.2
//! cargo run                 # then chat; /exit or Ctrl-D to quit
//! ```

use std::io::{self, BufRead, Write};

use spore_core::{
    Harness, HarnessBuilder, HarnessRunOptions, LoopStrategy, OllamaModelInterface, RunResult,
    SessionId, SessionState, Task,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());

    // Build the harness once; reuse it for every turn.
    let model = OllamaModelInterface::with_base_url(&model_id, base_url);
    let harness = HarnessBuilder::conversational(model).build();

    // One session id for the whole REPL, and the conversation state we thread
    // back in on each turn. Each run hands the post-run `SessionState` back
    // (issue #102), so we just carry it forward — lossless, no reconstruction.
    let session_id = SessionId::generate();
    let mut state = SessionState::default();
    let mut turns_exchanged = 0usize;

    println!("conversational REPL — model {model_id}. Type a message; /exit or Ctrl-D to quit.");
    let stdin = io::stdin();
    loop {
        print!("you> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break; // EOF (Ctrl-D)
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        if line == "/exit" || line == "/quit" {
            break;
        }

        // Thread the running state into this turn. The harness appends `line`
        // as the new user message before calling the model.
        let task = Task::new(
            line.clone(),
            session_id.clone(),
            LoopStrategy::ReAct { max_iterations: 4 },
        );
        let options = HarnessRunOptions::new(task).with_session_state(state.clone());

        match harness.run(options).await {
            RunResult::Success {
                output,
                session_state,
                ..
            } => {
                println!("bot> {output}");
                // Carry the post-run state forward losslessly (issue #102):
                // it already contains this turn's user + assistant messages
                // (and any tool messages a tool-using agent would produce).
                state = session_state;
                turns_exchanged += 1;
            }
            other => {
                eprintln!("bot> [run did not succeed: {other:?}]");
            }
        }
    }

    println!(
        "bye ({turns_exchanged} turn(s); {} message(s) in history)",
        state.messages.len()
    );
    Ok(())
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
