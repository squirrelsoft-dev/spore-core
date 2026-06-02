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
//! final response. It does NOT persist history for you, and — worth knowing —
//! `RunResult::Success` does not hand the post-run state back. The caller owns
//! the conversation.
//!
//! For a no-tools conversational agent that's easy: the assistant's turn is
//! exactly the `output` string. So after each turn we append the user's line
//! and the agent's reply to a local `Vec<Message>` and thread it back into the
//! next run via [`HarnessRunOptions::with_session_state`]. The harness appends
//! the new user line on top of that history before calling the model, so the
//! model sees the whole conversation and can refer back to it.
//!
//! (A tool-using agent would also produce tool-call/tool-result messages that
//! aren't recoverable from `output` alone — that's a limitation of this
//! reconstruction trick, and motivation for `RunResult::Success` to return the
//! final `SessionState` someday. For plain chat, reconstruction is exact.)
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
    Content, Harness, HarnessBuilder, HarnessRunOptions, LoopStrategy, Message,
    OllamaModelInterface, Role, RunResult, SessionId, SessionState, Task,
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

    // One session id for the whole REPL, and the conversation history we thread
    // back in on each turn. The harness never sees this as "its" state — we own
    // it.
    let session_id = SessionId::generate();
    let mut history: Vec<Message> = Vec::new();

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

        // Thread the running history into this turn. The harness appends `line`
        // as the new user message before calling the model.
        let task = Task::new(
            line.clone(),
            session_id.clone(),
            LoopStrategy::ReAct { max_iterations: 4 },
        );
        let options = HarnessRunOptions::new(task).with_session_state(SessionState {
            messages: history.clone(),
            extras: Default::default(),
        });

        match harness.run(options).await {
            RunResult::Success { output, .. } => {
                println!("bot> {output}");
                // Record both halves of the exchange so the next turn remembers.
                history.push(text_message(Role::User, line));
                history.push(text_message(Role::Assistant, output));
            }
            other => {
                eprintln!("bot> [run did not succeed: {other:?}]");
            }
        }
    }

    println!("bye ({} message(s) exchanged)", history.len());
    Ok(())
}

fn text_message(role: Role, text: String) -> Message {
    Message {
        role,
        content: Content::Text { text },
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
