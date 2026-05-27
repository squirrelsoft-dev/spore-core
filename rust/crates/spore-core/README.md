# spore-core (Rust)

Harness runtime library. Implements issues #1–#13 from the canonical spec at
`/docs/harness-engineering-concepts.md`.

See `../../CONVENTIONS.md` for Rust project conventions.

## End-to-end agent scenarios (issue #57)

`examples/e2e_agent.rs` is one shared CLI harness binary that drives the
complete loop through four proof scenarios against a real local model:

- `s1` multi-step/multi-tool, `s2` multi-turn, `s3` live compaction,
  `s4` tool failure + recovery.

Live run recipe (Ollama + the local observability stack in `observability/`):

```sh
# 1. Start Ollama and pull a tool-capable model (default: llama3.2).
ollama serve &
ollama pull llama3.2

# 2. (optional) Forward traces to the local Tempo/Loki/Grafana stack.
export SPORE_OTLP_ENDPOINT=http://localhost:4317

# 3. Run a scenario (prompt/model/endpoint/workspace come from args+env).
cargo run -p spore-core --example e2e_agent -- s1 --model llama3.2
cargo run -p spore-core --example e2e_agent -- s2
cargo run -p spore-core --example e2e_agent -- s3
cargo run -p spore-core --example e2e_agent -- s4

# 4. Verify the grouped trace in Tempo (the run prints the trace_id):
curl -s http://localhost:3200/api/traces/<trace_id> | jq '.batches | length'
#    For s3, spot-check that a Compaction span appears mid-trace.
```

Offline / hermetic mode (no Ollama, no network) reuses the same scenario
builders against a scripted mock model:

```sh
cargo run -p spore-core --features test-utils --example e2e_agent -- s1 --mock
```

CI assertions for all four scenarios live in `tests/e2e_scenarios.rs` and run
under `cargo test --features test-utils` — they never need a live model.
