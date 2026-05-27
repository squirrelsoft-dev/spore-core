# Trace JSONL Schema (canonical contract)

This file pins the **on-disk trace format** that the harness writes and that the
observability stack consumes. It is the shared contract between three things:

1. The emitter — the durable-outbox observability provider (issue #33) writes
   one line per span.
2. **Loki / Alloy** — Alloy tails `.spore/sessions/**/*.jsonl` and ships every
   line to Loki, where the Grafana dashboards query it with LogQL.
3. **Tempo** — the same spans are forwarded over OTLP gRPC (`:4317`); the
   `trace_id` field below is what links a Loki log line back to its Tempo trace
   (see the Loki `derivedFields` config in
   `grafana/datasources/datasources.yaml`).

> The local JSONL file is the **source of truth** (issue #33). OTLP is a
> best-effort forwarding mechanism. The dashboards therefore query Loki, not
> Prometheus — no harness `/metrics` endpoint is required.

## File location

```
{outbox_root}/sessions/{session_id}/trace.jsonl        # active file
{outbox_root}/sessions/{session_id}/trace-001.jsonl    # rotated segment(s)
{outbox_root}/sessions/{session_id}/.flushed           # written by flush_session
```

`OutboxConfig.root` points at the outbox root directory (`.spore` by
convention). Per session the provider derives
`{root}/sessions/{session_id}/trace.jsonl`. Append-only. One JSON object per
line (JSONL).

**Rotation.** When the active `trace.jsonl` exceeds `max_size_bytes` (default
50 MB) it is renamed to `trace-{NNN}.jsonl` (zero-padded, increasing) and a
fresh `trace.jsonl` is opened. Rotated segments keep the `.jsonl` suffix so
Alloy's `**/*.jsonl` glob continues to ship them to Loki.

**Flush marker.** `flush_session()` writes the trailing `session` summary line
(when `flush_on_session_end`), then creates a sibling `.flushed` marker file.
`list_unflushed_sessions()` returns session dirs that have a `trace.jsonl` but
no `.flushed` marker. The marker is not a `.jsonl` file, so Alloy ignores it.

**Ownership.** A single provider instance owns the open file handle for a given
session; concurrent appends from multiple processes to the same session file
are out of scope.

## Line format

Every line is one span. Common envelope fields are top-level; span-kind-specific
fields live under `attributes`. This keeps LogQL queries uniform: `| json`
flattens `attributes.cost_usd` to the label `attributes_cost_usd`.

```jsonc
{
  "trace_id":        "0af7651916cd43dd8448eb211c80319c", // 32-hex; generated once per session.
                                                         // IDENTICAL in JSONL and OTLP — this is the
                                                         // only id the Loki→Tempo derived field joins on.
  "span_id":         "b7ad6b7169203331",                 // the harness SpanId string, verbatim.
                                                         // NOT required to be 16-hex. The OTLP exporter
                                                         // derives an 8-byte OTLP span id from it by
                                                         // hashing; the JSONL keeps the readable string.
  "parent_span_id":  "0000000000000000",                 // harness SpanId string, or null for root
  "session_id":      "sess_01H...",
  "task_id":         "task_01H...",
  "kind":            "turn",         // see "kind" values below
  "level":           "info",         // "info" | "warn"  (patch spans are always "warn")
  "timestamp":       "2026-05-26T18:00:00.123Z", // RFC 3339, = span ended_at
  "started_at":      "2026-05-26T17:59:58.000Z", // RFC 3339
  "duration_ms":     2123,
  "status":          "ok",           // "ok" | "error" | "halted" (bare strings, mapped from SpanStatus)
  "status_detail":   null,           // SpanStatus Error.message / Halted.reason; null when status == ok
  "attributes":      { /* per-kind, see below */ }
}
```

> **`attributes` values are the verbatim serde serialization of the span payload
> fields — they are NOT renamed, flattened, or simplified.** Scalar fields stay
> scalar; structured fields (`trigger`, `operation`, `decision`, `patch_type`)
> stay as their existing tagged objects (e.g. `"trigger":{"kind":"post_tool",...}`).
> The Grafana dashboards only bind to the scalar attributes.

### `kind` values

`session`, `turn`, `tool_call`, `sensor_evaluation`, `context_assembly`,
`compaction`, `middleware_hook`, `guide_selection`, `memory_query`,
`memory_write`, `patch`.

### `attributes` by kind

| kind | attributes |
|---|---|
| `session` | `outcome` (`success`/`failure`/`partial` — the bare-string serialization of `SessionOutcome`), `total_turns`, `total_cost_usd`, `sensor_fires`, `sensor_halts`, `patch_count` |
| `turn` | `turn_number`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `cost_usd`, `stop_reason`, `tool_calls_requested` |
| `tool_call` | `tool_name`, `call_id`, `parameters_size_bytes`, `output_size_bytes`, `truncated`, `sandbox_mode`, `sandbox_violations` |
| `sensor_evaluation` | `sensor_id`, `sensor_kind`, `trigger`, `outcome`, `fired` |
| `context_assembly` / `compaction` | `operation`, `tokens_before`, `tokens_after`, `utilization_before`, `utilization_after` |
| `middleware_hook` | `hook`, `decision` |
| `patch` | `tool_name`, `call_id`, `patch_type`, `original_parameters`, `patched_parameters` |

These names are the snake_case serializations of the span structs already
defined in `spore-core` (`observability.rs` / equivalents). The emitter MUST NOT
rename them — the dashboards depend on these exact keys.

## LLM-native content capture (issue #64) — opt-in `gen_ai.*` content

By default the trace is **metrics/structure only**: it carries no prompts, model
output, tool arguments, or tool results. Issue #64 adds **opt-in** capture of
conversation/tool-call content following the OpenTelemetry **GenAI semantic
conventions**. It is gated and OFF by default because content can be large and
may contain secrets/PII.

### Env guard + truncation

| env var | default | effect |
|---|---|---|
| `SPORE_TRACE_CONTENT` | `false` (OFF) | `1`/`true`/`yes`/`on` (case-insensitive) enables content capture. Anything else, or unset, leaves it OFF. |
| `SPORE_TRACE_CONTENT_MAX_LEN` | `8192` | Max **UTF-8 bytes** of any single captured field. Over-budget fields are clipped at a valid UTF-8 char boundary (a multibyte char is never split — it backs off to the previous boundary) and the exact ASCII marker `...[truncated]` is appended. |

**With the guard OFF the durable JSONL is byte-identical to the pre-#64 output** —
none of the keys below appear (they serialize with `skip_serializing_if`).

### Added `attributes` keys (only present when capture is ON)

| kind | additional `gen_ai.*` attributes |
|---|---|
| `turn` | `gen_ai.response.role` (`assistant`), `gen_ai.response.content` (model output text), `gen_ai.response.content_truncated` (bool); and, when the turn requested tools, `gen_ai.response.tool_calls` — an array of `{ name, arguments, arguments_truncated }`. Per the maintainer decision, the turn span carries the model **output + requested tool calls only**; the assembled input-message history is NOT plumbed here. |
| `tool_call` | `gen_ai.tool.name`, `gen_ai.tool.call.arguments` (the tool-call arguments — a JSON value, or a clipped JSON string carrying the marker when truncated), `gen_ai.tool.call.arguments_truncated` (bool), `gen_ai.tool.message.content` (the tool result body), `gen_ai.tool.message.content_truncated` (bool). |

### OTLP span events

In addition to the attributes above, the OTLP forwarder emits one **span event
per message** using the conventional event names — `gen_ai.system.message`,
`gen_ai.user.message`, `gen_ai.assistant.message`, `gen_ai.tool.message` — each
carrying `gen_ai.message.role` plus the message content (and, for tool-call
requests, `gen_ai.tool.name` / `gen_ai.tool.call.arguments`). This is what an
LLM-native, OTel-native backend (e.g. Arize Phoenix) renders as the readable
conversation. The convention is vendor-neutral: the same OTLP stream is portable
across Phoenix / Langfuse / LangSmith without code changes.

### Routing

Content rides the existing single OTLP target configured by
`SPORE_OTLP_ENDPOINT`. There is no fan-out / multi-forwarder — Phoenix is simply
another OTLP endpoint you point that variable at.

## Example lines

```json
{"trace_id":"0af7651916cd43dd8448eb211c80319c","span_id":"b7ad6b7169203331","parent_span_id":null,"session_id":"sess_a","task_id":"task_a","kind":"turn","level":"info","timestamp":"2026-05-26T18:00:02.1Z","started_at":"2026-05-26T18:00:00.0Z","duration_ms":2100,"status":"ok","status_detail":null,"attributes":{"turn_number":1,"input_tokens":1820,"output_tokens":140,"cache_read_tokens":1600,"cache_write_tokens":0,"cost_usd":0.0123,"stop_reason":"tool_use","tool_calls_requested":1}}
{"trace_id":"0af7651916cd43dd8448eb211c80319c","span_id":"f1a2","parent_span_id":"b7ad6b7169203331","session_id":"sess_a","task_id":"task_a","kind":"sensor_evaluation","level":"info","timestamp":"2026-05-26T18:00:02.3Z","started_at":"2026-05-26T18:00:02.2Z","duration_ms":40,"status":"ok","status_detail":null,"attributes":{"sensor_id":"test-runner","sensor_kind":"test","trigger":"after_tool","outcome":"pass","fired":true}}
{"trace_id":"0af7651916cd43dd8448eb211c80319c","span_id":"root","parent_span_id":null,"session_id":"sess_a","task_id":"task_a","kind":"session","level":"info","timestamp":"2026-05-26T18:01:00.0Z","started_at":"2026-05-26T18:00:00.0Z","duration_ms":60000,"status":"ok","status_detail":null,"attributes":{"outcome":"success","total_turns":4,"total_cost_usd":0.0456,"sensor_fires":3,"sensor_halts":0,"patch_count":0}}
```
