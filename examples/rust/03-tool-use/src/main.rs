//! spore-core example 03 — ReAct with local tools.
//!
//! The agent now *acts*: it thinks, calls a tool, observes the result, and
//! loops until it can answer. The tools here are deliberately trivial —
//! `calculator`, `get_current_time`, `reverse_string` — because the star of
//! this example is the **Think → Act → Observe** loop, not the tools.
//!
//! ## What it shows
//!
//! - Implementing the harness-loop [`ToolRegistry`] directly: `schemas()`
//!   advertises the tools to the model, `dispatch()` runs them. No filesystem,
//!   no sandbox needed — these are pure functions, so the
//!   [`conversational`](spore_core::HarnessBuilder::conversational) defaults
//!   (incl. `NullSandbox`) are fine; we only override the tool registry.
//! - The loop itself: the program prints each turn (Think) and each tool call +
//!   result (Act / Observe), so you can watch the agent work.
//!
//! ## Run it
//!
//! ```sh
//! ollama serve &
//! ollama pull llama3.2
//! cargo run
//! cargo run -- --prompt "reverse the word 'mycelium' and multiply 6 by 7"
//! ```

use std::sync::Arc;

use spore_core::harness::BoxFut;
use spore_core::{
    Harness, HarnessBuilder, HarnessRunOptions, HarnessStreamEvent, HarnessToolRegistry,
    LoopStrategy, OllamaModelInterface, ReactConfig, RunResult, SessionId, Task, ToolCall,
    ToolOutput, ToolSchema,
};

/// Three trivial, pure-compute tools, exposed through the harness-loop tool
/// registry. `schemas()` is what the model sees; `dispatch()` is what runs.
struct LocalTools;

impl HarnessToolRegistry for LocalTools {
    fn schemas(&self) -> Vec<ToolSchema> {
        vec![
            schema(
                "calculator",
                "Compute a binary arithmetic operation. 'op' is one of + - * /.",
                serde_json::json!({
                    "a": { "type": "number" },
                    "b": { "type": "number" },
                    "op": { "type": "string", "enum": ["+", "-", "*", "/"] }
                }),
                &["a", "b", "op"],
            ),
            schema(
                "get_current_time",
                "Return the current time of day as HH:MM:SS UTC. Takes no arguments.",
                serde_json::json!({}),
                &[],
            ),
            schema(
                "reverse_string",
                "Reverse the characters in a string.",
                serde_json::json!({ "text": { "type": "string" } }),
                &["text"],
            ),
        ]
    }

    fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let result = match call.name.as_str() {
                "calculator" => calculator(&call.input),
                "get_current_time" => Ok(current_time()),
                "reverse_string" => reverse_string(&call.input),
                other => Err(format!("unknown tool: {other}")),
            };
            match result {
                // Print the Act + Observe step so the loop is visible.
                Ok(content) => {
                    println!("    act    → {}({}) = {content}", call.name, call.input);
                    ToolOutput::Success {
                        content,
                        truncated: false,
                    }
                }
                Err(message) => {
                    println!("    act    → {}({}) failed: {message}", call.name, call.input);
                    ToolOutput::Error {
                        message,
                        recoverable: true,
                    }
                }
            }
        })
    }
}

fn calculator(input: &serde_json::Value) -> Result<String, String> {
    let a = number(input, "a")?;
    let b = number(input, "b")?;
    let op = input
        .get("op")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing string 'op'".to_string())?;
    let value = match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" if b == 0.0 => return Err("division by zero".to_string()),
        "/" => a / b,
        other => return Err(format!("unknown op '{other}' (use + - * /)")),
    };
    Ok(format!("{value}"))
}

fn number(input: &serde_json::Value, key: &str) -> Result<f64, String> {
    let value = input
        .get(key)
        .ok_or_else(|| format!("missing number '{key}'"))?;
    // Models often pass numbers as JSON strings ("144"); accept either.
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
        .ok_or_else(|| format!("'{key}' is not a number: {value}"))
}

fn current_time() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!(
        "{:02}:{:02}:{:02} UTC",
        (secs / 3600) % 24,
        (secs / 60) % 60,
        secs % 60
    )
}

fn reverse_string(input: &serde_json::Value) -> Result<String, String> {
    let text = input
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing string 'text'".to_string())?;
    Ok(text.chars().rev().collect())
}

fn schema(name: &str, description: &str, properties: serde_json::Value, required: &[&str]) -> ToolSchema {
    ToolSchema {
        name: name.to_string(),
        description: description.to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
        }),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_id = arg_value(&args, "--model")
        .or_else(|| std::env::var("SPORE_OLLAMA_MODEL").ok())
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = std::env::var("SPORE_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| OllamaModelInterface::DEFAULT_BASE_URL.to_string());
    let prompt = arg_value(&args, "--prompt").unwrap_or_else(|| {
        "Use your tools to answer: what is 144 divided by 12, what is the current time, \
         and what is 'harness' reversed?"
            .to_string()
    });

    // Same conversational harness as 01/02 — we only swap in our tool registry.
    let model = OllamaModelInterface::with_base_url(&model_id, base_url);
    let harness = HarnessBuilder::conversational(model)
        .tool_registry(Arc::new(LocalTools))
        .build();

    let task = Task::new(
        prompt.clone(),
        SessionId::generate(),
        LoopStrategy::ReAct(ReactConfig::per_loop(6)),
    );
    // Print each turn so the "Think" steps are visible alongside the tool calls.
    let options = HarnessRunOptions::new(task).with_stream(Box::new(|event: HarnessStreamEvent| {
        if let HarnessStreamEvent::TurnStart { turn, .. } = event {
            println!("think  · turn {turn}");
        }
    }));

    println!("model  : {model_id}");
    println!("prompt : {prompt}\n");
    match harness.run(options).await {
        RunResult::Success { output, turns, .. } => {
            println!("\nanswer ({turns} turn(s)): {output}");
            Ok(())
        }
        other => {
            eprintln!("\nrun did not succeed: {other:?}");
            std::process::exit(1);
        }
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}
