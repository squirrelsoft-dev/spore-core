/**
 * Tool-call repair — mirrors `rust/crates/spore-core/src/tool_call_repair.rs`.
 */

export type { ToolCallFailure } from "./types.js";
export { detectProseResponse } from "./types.js";
export { coerceToolArgs, StandardToolCallRepair, type ToolCallRepair, type RepairSchema } from "./standard.js";