# Observability

> Language-agnostic — no code. The full story lives in the spec's
> [ObservabilityProvider section](../harness-engineering-concepts.md#observabilityprovider).

If it isn't traced, it can't be improved. spore-core treats observability as a first-class
output of every run, not an afterthought.

## What gets recorded

The **observability provider** emits a structured span for every meaningful operation: each
turn, each tool call, each sensor execution, each context operation. Spans are structured — keys
are defined, not free text — and every span carries session and task identifiers so it rolls up
cleanly:

```
ToolDispatch → Turn → Task → Session → Project
```

Token usage, latency, and cost are **first-class span fields**, including cache read/write tokens
and their cost. The provider is a passive observer: it never changes what the harness does, and
every emit is fire-and-forget so tracing never blocks the loop.

## The outbox

The reference durable provider is an **outbox**: every emit writes exactly one JSONL line to a
per-session trace file on disk. That gives you a complete, append-only record of a run that
survives a crash, independent of any network. Each session's spans land in its own
`trace.jsonl` under the `.spore/` directory.

The outbox is the safety net — it captures the trace locally first, then forwards.

## OTLP — your tools, day one

When an OTLP endpoint is configured (via the `SPORE_OTLP_ENDPOINT` environment variable), the
provider forwards spans over **OpenTelemetry Protocol**, best-effort and non-blocking. Because
it's standard OTLP, the traces land in whatever you already run — Langfuse, Grafana, Datadog,
Honeycomb, Jaeger, New Relic — with no spore-specific integration. The session id becomes the
trace id, so a run is one trace end to end.

If the endpoint is unset, nothing is forwarded and the local outbox still captures everything.

## Phoenix and local trace viewing

For local development, point the OTLP endpoint at a self-hosted collector such as
[Arize Phoenix](https://github.com/Arize-ai/phoenix) to get a UI over your traces — turns, tool
calls, token spend, and latency — without standing up production telemetry. Because the wire
format is plain OTLP, any OpenTelemetry-compatible viewer works the same way; Phoenix is just a
convenient, free, local one.

## The improvement loop

Trace data is the input to making the system better over time. The spec's improvement flywheel
turns on it: a failure in one trace is an incident, the same failure across many traces is a
pattern, and a pattern without a guide is a gap to close. None of that is possible without
complete, structured traces — which is why every harness operation emits one.
