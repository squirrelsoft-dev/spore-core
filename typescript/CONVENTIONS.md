# TypeScript Conventions — spore-core

Read alongside `/docs/harness-engineering-concepts.md` before implementing
any component (issues #1–#13).

## Key decisions (already made — do not relitigate)

- **TypeScript**: `strict: true` everywhere. No implicit `any`, no
  `// @ts-ignore` without an inline justification.
- **Modules**: ESM only. `"type": "module"` in every `package.json`.
  No CommonJS interop except where a third-party dep forces it.
- **Runtime validation**: `zod` schemas for every external boundary —
  tool schemas, model responses, fixture loading, PausedState (de)serialization.
  Static types are derived from zod via `z.infer<...>`.
- **Package manager**: `pnpm` workspaces (`pnpm@9`). Never npm/yarn.
- **Async**: native `async`/`await`. Use `AbortSignal` for cancellation —
  pass it down through every async public API.
- **Node**: 20 LTS. Vitest for tests.

## Interface pattern

Components are TypeScript `interface`s. Concrete implementations are
classes implementing one or more interfaces. Inject via constructor
parameters typed as the interface.

```ts
export interface ModelInterface {
  call(req: ModelRequest, signal?: AbortSignal): Promise<ModelResponse>;
  countTokens(req: ModelRequest): Promise<number>;
}
```

## Error handling

- Domain errors are classes extending `Error` with a `name` set to the
  class name and a discriminant `kind` field for `switch` exhaustiveness.
- Always-halt errors (spec Layer 1) extend a marker class `AlwaysHaltError`.
- Tools return a `ToolOutput` union (`Success | Error | WaitingForHuman`)
  rather than throwing for recoverable cases.
- Never throw plain `Error`s across package boundaries.

## Naming

- Files: `kebab-case.ts`.
- Types and classes: `UpperCamelCase`. Interface names match the spec
  (`ModelInterface`, not `IModel`).
- Functions and variables: `camelCase`.
- Constants: `UPPER_SNAKE_CASE` only for genuine module-level constants.
- Zod schemas: same name as the type plus `Schema` (`ModelRequestSchema`).

## Layout

```
typescript/
  package.json
  pnpm-workspace.yaml
  packages/
    core/        ← @spore/core, one module per component
    tools/       ← @spore/tools, standard Tool impls
    test-utils/  ← @spore/test-utils, recording/replay helpers
```

Inside `packages/core/src/`, one folder per harness component, with an
`index.ts` re-export at the package root.

## Running tests

- Full suite:        `pnpm test`
- Per package:       `pnpm --filter @spore/core test`
- Unit only:         `pnpm --filter @spore/core test -- --run`
- Single by name:    `pnpm --filter @spore/core test -- -t "<name pattern>"`
- Watch mode:        `pnpm --filter @spore/core test -- --watch`

## Lint / format

- `pnpm lint` (eslint) — CI gate.
- Formatter is `prettier` invoked through eslint.

## Adding a dependency

`pnpm --filter <package> add <dep>`. Devtools go to the workspace root via
`pnpm add -Dw <dep>`. Pin to a minor version.

## Cross-language consistency

Public types (`ModelResponse`, `ToolOutput`, `PausedState`, etc.) must match
the Rust, Python, and Go definitions. Same fixture, same outcome — see
`/fixtures/README.md`.
