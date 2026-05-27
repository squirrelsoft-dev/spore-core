# Go Conventions — spore-core

Read alongside `/docs/harness-engineering-concepts.md` before implementing
any component (issues #1–#13).

## Key decisions (already made — do not relitigate)

- **Go**: 1.22. Workspace mode via `go.work`. Module path prefix:
  `github.com/squirrelsoft-dev/spore-core/go/<module>`.
- **Cancellation**: standard `context.Context` as the first argument of
  every async-style public function. Never store a `Context` in a struct.
- **Interfaces**: all harness components are Go interfaces defined in the
  consuming package (consumer-side interfaces). Concrete impls live in
  sibling subpackages (`model/anthropic`, `model/openai`, etc.).
- **Errors**: `errors` package with `fmt.Errorf("...: %w", err)` for wrapping.
  Sentinel errors (`var ErrPathEscape = errors.New(...)`) for always-halt
  conditions (spec Layer 1). Typed error structs implementing `error` for
  recoverable cases. Match with `errors.Is` / `errors.As`.
- **No third-party DI / web framework** in `spore-core`. Standard library
  plus minimal blessed deps only.

## Interface pattern

```go
type ModelInterface interface {
    Call(ctx context.Context, req ModelRequest) (ModelResponse, error)
    CountTokens(ctx context.Context, req ModelRequest) (int, error)
}
```

Components are passed by interface value. Implementations are concrete
structs in subpackages and never need to declare they implement the
interface — the compile-time check `var _ ModelInterface = (*Anthropic)(nil)`
goes at the bottom of each impl file.

## Error handling

- Sentinel errors for Layer 1 (always-halt): `ErrPathEscape`,
  `ErrNetworkViolation`, `ErrBudgetExceeded`, `ErrContextLimitExceeded`.
- Typed error structs for component errors: `type ToolError struct{ ... }`
  implementing `Error() string`.
- Wrap with `%w` when crossing layers — never lose the underlying error.
- `ToolOutput` is a union-style struct (variant tag + payload) for
  `Success | Error | WaitingForHuman`; recoverable tool failures do not
  return a Go `error`, they return a `ToolOutput` with the error variant.

## Naming

- Package names: short, lowercase, no underscores — e.g. `sporecore`,
  `eval`, `sandbox`.
- Exported types: `UpperCamelCase` matching the spec exactly
  (`ModelInterface`, `ToolRegistry`).
- Unexported: `lowerCamelCase`.
- IDs: defined types over `string` — `type SessionID string`. Note Go
  convention is `ID` (uppercase) not `Id`.
- File names: `snake_case.go`. One harness component per file.

## Layout

```
go/
  go.work
  spore-core/    ← module: harness runtime, one file per component
  spore-cli/     ← module: thin CLI binary
  spore-eval/    ← module: evaluation harness
```

Inside `spore-core/`, top-level package `sporecore` defines all
interfaces. Concrete impls live in subpackages (`sandbox/`, `model/`,
`memory/sqlite/`, ...).

## Running tests

- Full suite:        `go test ./...` (from each module dir or via `go.work`)
- Per module:        `cd spore-core && go test ./...`
- Unit only:         `go test -short ./...`
- Single by name:    `go test -run '<RegexpName>' ./...`
- With race detector:`go test -race ./...`

CI runs with the race detector enabled.

## Lint / format

- `gofmt -l .` — CI fails if output is non-empty.
- `go vet ./...` — CI gate.
- Optional locally: `golangci-lint run`.

## Adding a dependency

`go get <module>@<version>` from within the relevant module directory.
Pin to a minor version. Run `go mod tidy` before committing.

### Blessed dependencies

`spore-core` favors the standard library plus a minimal set of blessed deps.
The currently blessed third-party deps are the OpenTelemetry SDK + OTLP gRPC
exporter, used solely by the observability outbox's OTLP forwarder:

- `go.opentelemetry.io/otel`
- `go.opentelemetry.io/otel/sdk`
- `go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc`
- `go.opentelemetry.io/otel/trace`

**Decision (issue #50, Option A): the core outbox is no longer zero-dep.**
Issue #49 wired `ObservabilityProvider` into the harness loop across all four
languages and explicitly *deferred* Go OTLP, leaving `newForwarder` as a no-op
seam so the outbox stayed zero-dependency. #50 closes that parity gap: Rust,
TypeScript, and Python already forward end-to-end traces to Tempo over OTLP
gRPC, so Go must too. Rather than hide the SDK behind a build tag or an optional
sub-module, we added the otel SDK + `otlptracegrpc` directly to
`go/spore-core/go.mod` (matching how Python/Rust take the dependency). The
accepted tradeoff: the reliability-critical outbox now pulls grpc/protobuf and
their transitive deps. This is bounded because the OTLP surface is isolated in
`observability/otlp.go` behind the internal `otlpForwarder` interface, and the
durable JSONL path remains fully network-free — it appends the line *before*
the best-effort OTLP forward, never blocks on it, and never fails the harness
loop on OTLP errors. When `SPORE_OTLP_ENDPOINT` is unset/empty the forwarder is
the no-op `nullForwarder` and no OTLP code runs.

## Cross-language consistency

Public types (`ModelResponse`, `ToolOutput`, `PausedState`) must match
the Rust, TypeScript, and Python definitions. JSON tag names use
`snake_case` for wire compatibility. See `/fixtures/README.md`.
