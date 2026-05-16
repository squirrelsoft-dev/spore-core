# spore-cli (Rust)

Thin CLI wrapper around `spore-core`. Reads config, constructs a Harness via
the builder, streams tokens to stdout, persists `PausedState` to disk on
`WaitingForHuman`. See the Deployment Surfaces section of the spec.
