//! `spore-eval` CLI — mirrors the spore-e2e-agent pattern. anyhow is allowed
//! ONLY here (CONVENTIONS.md).
//!
//! Subcommands:
//!   run <suite.json> [--candidate ID ...] [--n N] [--json]
//!       Load a suite and print/serialize the comparison reports. (Candidate
//!       harness configs are not constructible from the CLI in the MVP — the
//!       command validates the suite and prints its shape; full runs are driven
//!       programmatically via `EvalHarness`.)
//!   promote <suite.json> <task_id>
//!       Manually promote a challenge task to regression, bump suite_version,
//!       rewrite the JSON in place (Rule 31). Auto-promotion is deferred.

use anyhow::{bail, Context, Result};
use spore_eval::{load_suite_path, promote_challenge_task, suite_to_json};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        print_usage();
        bail!("no subcommand given");
    };
    match cmd.as_str() {
        "run" => cmd_run(&args[1..]),
        "promote" => cmd_promote(&args[1..]),
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        other => {
            print_usage();
            bail!("unknown subcommand: {other}");
        }
    }
}

fn print_usage() {
    eprintln!(
        "spore-eval — EvalHarness CLI\n\n\
         USAGE:\n\
         \x20 spore-eval run <suite.json> [--candidate ID ...] [--n N] [--json]\n\
         \x20 spore-eval promote <suite.json> <task_id>\n"
    );
}

fn cmd_run(args: &[String]) -> Result<()> {
    let mut suite_path: Option<String> = None;
    let mut candidates: Vec<String> = Vec::new();
    let mut n: Option<u32> = None;
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--candidate" => {
                i += 1;
                let v = args.get(i).context("--candidate needs an ID")?;
                candidates.push(v.clone());
            }
            "--n" => {
                i += 1;
                let v = args.get(i).context("--n needs a number")?;
                n = Some(v.parse().context("--n must be a u32")?);
            }
            "--json" => json = true,
            other if !other.starts_with("--") && suite_path.is_none() => {
                suite_path = Some(other.to_string());
            }
            other => bail!("unexpected argument: {other}"),
        }
        i += 1;
    }
    let suite_path = suite_path.context("missing <suite.json>")?;
    let suite = load_suite_path(std::path::Path::new(&suite_path))
        .with_context(|| format!("loading suite {suite_path}"))?;

    let summary = serde_json::json!({
        "suite_version": suite.suite_version,
        "regression": suite.regression.len(),
        "challenge": suite.challenge.len(),
        "canary": suite.canary.len(),
        "n_runs_per_config": n.unwrap_or(3),
        "candidates": candidates,
        "note": "candidate harness configs are wired programmatically; \
                 this CLI validates the suite and reports its shape",
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "loaded suite v{} — regression={}, challenge={}, canary={} (n={})",
            suite.suite_version,
            suite.regression.len(),
            suite.challenge.len(),
            suite.canary.len(),
            n.unwrap_or(3),
        );
    }
    Ok(())
}

fn cmd_promote(args: &[String]) -> Result<()> {
    let suite_path = args.first().context("missing <suite.json>")?;
    let task_id = args.get(1).context("missing <task_id>")?;
    let mut suite = load_suite_path(std::path::Path::new(suite_path))
        .with_context(|| format!("loading suite {suite_path}"))?;
    let before = suite.suite_version;
    promote_challenge_task(&mut suite, task_id).with_context(|| format!("promoting {task_id}"))?;
    let json = suite_to_json(&suite)?;
    std::fs::write(suite_path, json).with_context(|| format!("writing {suite_path}"))?;
    println!(
        "promoted {task_id}: suite_version {before} -> {} (challenge -> regression)",
        suite.suite_version
    );
    Ok(())
}
