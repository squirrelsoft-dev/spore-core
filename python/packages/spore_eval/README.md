# spore_eval (Python)

Evaluation harness. Runs the shared task suites from `/fixtures/task_suites/`
against a `spore_core` Harness.

## End-to-end CLI agent + scenario suite (issue #57)

`spore_eval.e2e_agent` is one shared runnable entry point that drives the
**complete** harness loop through a `HarnessBuilder`-assembled harness with real
tools (read/write/list/bash + a deliberately-failing `flaky_op`), the
`StandardCompactionAdapter`, and the durable-outbox observability provider. The
scenario is selected by a CLI arg (`s1`..`s4`). `spore_eval.scenarios` holds the
shared builders consumed by **both** this CLI and the hermetic tests, so live
and offline runs drive one code path.

### Scenarios (the proofs)

- **S1 — multi-step / multi-tool:** read `input.txt` → uppercase → write
  `output.txt` → read back and confirm.
- **S2 — multi-turn:** two `run()` calls sharing one `SessionId`; turn 2
  references turn 1.
- **S3 — live compaction:** a seeded small window + long history fires the
  compaction adapter mid-run; the token-accounting fix lets the budget drop and
  the session compact, continue, and compact again.
- **S4 — tool failure + recovery:** call `flaky_op` (recoverable error), then
  write `recovered.txt` explaining the adaptation.

### Manual run recipe (live, against a local model + observability stack)

```sh
# 1. Start Ollama and pull a tool-capable model.
ollama serve &              # or run the Ollama app
ollama pull llama3.2        # default model; passes the #41 capability guard

# 2. (optional) Start the local observability stack and forward traces.
#    The compose stack lives at observability/docker-compose.observability.yml
#    (Tempo + Loki + Grafana + Alloy).
docker compose -f observability/docker-compose.observability.yml up -d
export SPORE_OTLP_ENDPOINT=http://localhost:4317

# 3. Run each scenario. Prompt/model/endpoint/workspace come from args + env.
uv run --package spore-eval spore-e2e-agent s1 --model llama3.2
uv run --package spore-eval spore-e2e-agent s2
uv run --package spore-eval spore-e2e-agent s3
uv run --package spore-eval spore-e2e-agent s4

# 4. Verify the grouped trace in Tempo (the run prints the session id):
curl -s http://localhost:3200/api/traces/<trace_id> | jq '.batches | length'
#    For S3, spot-check a Compaction span appears mid-trace with a non-zero
#    tokens_reclaimed and tokens_after < tokens_before.
```

Environment variables (all optional):

- `SPORE_OLLAMA_MODEL`    — default model id (overridden by `--model`).
- `SPORE_OLLAMA_BASE_URL` — Ollama base url (default `http://localhost:11434`).
- `SPORE_OTLP_ENDPOINT`   — when set, forward spans to Tempo (issue #50).
- `SPORE_E2E_WORKSPACE`   — workspace root (default: a temp dir per run).

### Offline / hermetic mode

`--mock` runs the same scenario builders against a scripted `MockAgent`,
requiring no Ollama or network:

```sh
uv run --package spore-eval spore-e2e-agent s1 --mock
```

The hermetic CI assertions live in `tests/test_e2e_scenarios.py`, which drive
the same `build_scenario` path with mock components (one test per scenario, plus
shared-builder unit checks). They never require a live Ollama or any network.
