# Rust Conventions — spore-core

Read alongside `/docs/harness-engineering-concepts.md` before implementing
any component (issues #1–#13).

## Key decisions (already made — do not relitigate)

- **Async**: native async traits (Rust 1.75+). No `async-trait` crate.
- **Runtime**: `tokio` (multi-threaded). Components must not assume a
  current-thread runtime.
- **Trait objects**: `Arc<dyn Trait + Send + Sync>` everywhere a component
  is injected. The harness owns clones; trait objects are cheap to share.
- **Errors**: `thiserror` for library error enums. `anyhow` only inside
  the CLI binary, never in `spore-core`.
- **Edition**: 2021. MSRV is current stable.
- **Serialization**: `serde` + `serde_json`. JSON for fixtures and on-disk
  state under `.spore/`.

## Trait / interface pattern

Component traits return a boxed `Send` future (`BoxFut`) rather than using
`async fn` in traits. `async fn`/RPITIT (`#[trait_variant::make(Send)]`) is **not**
`dyn`-compatible, and every injected component must be holdable as `Arc<dyn Trait>`
(above) — so the dyn-safe traits hand-roll the future type:

```rust
pub type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub trait ModelInterface: Send + Sync {
    fn call<'a>(&'a self, req: ModelRequest) -> BoxFut<'a, Result<ModelResponse, ModelError>>;
    fn count_tokens<'a>(&'a self, req: &'a ModelRequest) -> BoxFut<'a, Result<u32, ModelError>>;
}
```

Components are injected as `Arc<dyn ModelInterface>` (a blanket
`impl ModelInterface for Arc<T: ?Sized>` makes the boxed model a first-class
`ModelInterface`; build one with `HarnessBuilder::conversational_arc`). Concrete
impls live in sibling modules (`anthropic.rs`, `openai.rs`, `ollama.rs`,
`model.rs`).

## Error handling

- One error enum per component, derived with `thiserror::Error`.
- Public error enums are `#[non_exhaustive]`.
- Always-halt vs recoverable distinction lives in the error variant — see
  the spec's Layer 1 / Layer 2 / Layer 3 routing.
- No `panic!`, `unwrap()`, or `expect()` in `spore-core` outside `#[cfg(test)]`.

## Naming

- Modules: `snake_case`.
- Types: `UpperCamelCase`. Trait names match the spec (`ModelInterface`, not
  `Model`). Builder types end in `Builder`.
- Errors: `<Component>Error`.
- IDs: newtype wrappers (`SessionId(String)`, `TaskId(String)`).

## Layout

```
rust/
  Cargo.toml          ← workspace
  crates/
    spore-core/       ← library; one module per component
    spore-cli/        ← thin binary
    spore-eval/       ← evaluation harness
```

Inside `spore-core/src/`, one module per harness component. Public items
re-exported from `lib.rs`. Internal helpers stay private to the module.

## Running tests

- Full suite:        `cargo test --workspace`
- Unit only:         `cargo test --lib --workspace`
- Single by name:    `cargo test --workspace <name_substring>`
- One crate:         `cargo test -p spore-core`
- Doctests:          `cargo test --doc --workspace`

Replay tests against shared fixtures (`/fixtures/model_responses/`) run as
part of the unit suite and must pass byte-for-byte.

## Lint / format

- `cargo fmt --all --check` — CI gate
- `cargo clippy --workspace --all-targets -- -D warnings`

## Adding a dependency

`cargo add <crate> -p <member>`. Pin to a minor version. Justify any
dependency that pulls in a sync runtime or `lazy_static`/`once_cell`
(prefer `std::sync::OnceLock`).

## Cross-language consistency

When you change a public-facing type (e.g. `ModelResponse`, `ToolOutput`,
`PausedState`), mirror the change in the TypeScript, Python, and Go packages
in the same PR. Same fixture, same outcome — see `/fixtures/README.md`.
