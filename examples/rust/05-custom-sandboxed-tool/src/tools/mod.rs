//! The two custom tools this example registers.
//!
//! Each is defined with the [`tool!`](spore_core::tool) macro: a typed input
//! struct deriving `Deserialize` + `JsonSchema`, plus an async `execute`
//! closure. The macro derives the advertised schema from the input struct and
//! returns a ready-to-register [`StandardTool`](spore_core::StandardTool). `main`
//! hands each to the builder via `.tool(...)`; the harness wires the sandbox and
//! a per-run `ToolContext` (the storage seam) in automatically.

pub mod recall;
pub mod remember;
