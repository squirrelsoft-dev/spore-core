//! The two custom tools this example registers.
//!
//! Each is a plain `impl Tool` (see [`spore_core::Tool`]) paired with a
//! `schema()` constructor. `main` wraps each pair in `StandardTool::new` and
//! hands it to the builder via `.tool(...)`. The harness wires the sandbox and a
//! per-run `ToolContext` (the storage seam) in automatically.

pub mod recall;
pub mod remember;
