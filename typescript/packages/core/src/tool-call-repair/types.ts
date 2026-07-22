/**
 * Tool-call repair types — mirrors `rust/crates/spore-core/src/tool_call_repair.rs`.
 *
 * {@link detectProseResponse} is re-exported from the prompt-tool-call module
 * (where it already lives) so all tool-call failure modes share one vocabulary.
 */

export { detectProseResponse } from "../model/prompt-tool-call.js";

/**
 * How a model's tool-calling attempt failed. Mirrors Rust `ToolCallFailure`.
 */
export type ToolCallFailure =
  | { kind: "malformed_json"; text: string }
  | { kind: "prose_response"; text: string }
  | { kind: "unknown_tool"; name: string };