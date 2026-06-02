/**
 * Zod schemas mirroring the wire shape of every public type in this module.
 *
 * Wire format invariants (must match Rust byte-for-byte):
 *   - `snake_case` field names
 *   - `Content` / `ContentBlock` / `StreamEvent` are tagged unions with a
 *     `"type"` discriminator
 *   - enum variants are `snake_case` (`end_turn`, `tool_use`, ...)
 *
 * Static types are derived from these schemas via `z.infer` and re-exported
 * from the package root.
 */

import { z } from "zod";

export const RoleSchema = z.enum(["system", "user", "assistant", "tool"]);

export const ToolCallSchema = z.object({
  id: z.string(),
  name: z.string(),
  // Free-form JSON; Rust uses serde_json::Value.
  input: z.unknown(),
});

export const ToolResultSchema = z.object({
  tool_use_id: z.string(),
  content: z.string(),
  is_error: z.boolean().default(false),
});

export const ContentSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("text"), text: z.string() }),
  z.object({
    type: z.literal("tool_call"),
    id: z.string(),
    name: z.string(),
    input: z.unknown(),
  }),
  z.object({
    type: z.literal("tool_result"),
    tool_use_id: z.string(),
    content: z.string(),
    is_error: z.boolean().default(false),
  }),
  z.object({
    type: z.literal("image"),
    media_type: z.string(),
    data: z.string(),
  }),
]);

export const MessageSchema = z.object({
  role: RoleSchema,
  content: ContentSchema,
});

export const ToolSchemaSchema = z.object({
  name: z.string(),
  description: z.string(),
  input_schema: z.unknown(),
});

export const ModelParamsSchema = z.object({
  temperature: z.number().nullable().optional(),
  max_tokens: z.number().int().nonnegative().nullable().optional(),
  reasoning_budget: z.number().int().nonnegative().nullable().optional(),
  top_p: z.number().nullable().optional(),
  stop_sequences: z.array(z.string()).default([]),
});

export const ModelRequestSchema = z.object({
  messages: z.array(MessageSchema),
  tools: z.array(ToolSchemaSchema).default([]),
  params: ModelParamsSchema.default({ stop_sequences: [] }),
  stream: z.boolean().default(false),
});

export const StopReasonSchema = z.enum(["tool_use", "end_turn", "max_tokens", "stop_sequence"]);

export const ContentBlockSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("text"), text: z.string() }),
  z.object({ type: z.literal("thinking"), text: z.string() }),
  z.object({
    type: z.literal("tool_use"),
    id: z.string(),
    name: z.string(),
    input: z.unknown(),
  }),
]);

export const TokenUsageSchema = z.object({
  input_tokens: z.number().int().nonnegative(),
  output_tokens: z.number().int().nonnegative(),
  cache_read_tokens: z.number().int().nonnegative().nullable().optional(),
  cache_write_tokens: z.number().int().nonnegative().nullable().optional(),
});

export const ModelResponseSchema = z.object({
  content: z.array(ContentBlockSchema),
  usage: TokenUsageSchema,
  stop_reason: StopReasonSchema,
});

export const StreamEventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("message_start") }),
  z.object({
    type: z.literal("content_block_delta"),
    index: z.number().int().nonnegative(),
    delta: z.string(),
  }),
  z.object({
    type: z.literal("thinking_delta"),
    index: z.number().int().nonnegative(),
    delta: z.string(),
  }),
  // Start of a tool-use block. Carries the tool `name` and call `id` — both
  // arrive on the provider's block-start frame (Anthropic `content_block_start`,
  // Ollama / OpenAI's first `tool_calls` chunk) and would otherwise be lost,
  // since `tool_use_delta` carries only argument JSON. The streaming
  // accumulator uses this to reconstruct the tool call faithfully.
  z.object({
    type: z.literal("tool_use_start"),
    index: z.number().int().nonnegative(),
    id: z.string(),
    name: z.string(),
  }),
  z.object({
    type: z.literal("tool_use_delta"),
    index: z.number().int().nonnegative(),
    partial_json: z.string(),
  }),
  z.object({
    type: z.literal("content_block_stop"),
    index: z.number().int().nonnegative(),
  }),
  z.object({
    type: z.literal("message_stop"),
    usage: TokenUsageSchema,
    stop_reason: StopReasonSchema,
  }),
]);

export const ProviderInfoSchema = z.object({
  name: z.string(),
  model_id: z.string(),
  context_window: z.number().int().nonnegative(),
});

export const RecordedExchangeSchema = z.object({
  /**
   * Stable hash of `request` (see `requestHash`). Optional so pre-#37
   * positional fixtures continue to deserialize.
   */
  request_hash: z.string().nullable().optional(),
  request: ModelRequestSchema,
  response: ModelResponseSchema,
  provider: z.string(),
  model_id: z.string().nullable().optional(),
  recorded_at: z.string().nullable().optional(),
  duration_ms: z.number().int().nonnegative().nullable().optional(),
});

// Static types derived from the schemas above.
export type Role = z.infer<typeof RoleSchema>;
export type ToolCall = z.infer<typeof ToolCallSchema>;
export type ToolResult = z.infer<typeof ToolResultSchema>;
export type Content = z.infer<typeof ContentSchema>;
export type Message = z.infer<typeof MessageSchema>;
export type ToolSchema = z.infer<typeof ToolSchemaSchema>;
export type ModelParams = z.infer<typeof ModelParamsSchema>;
export type ModelRequest = z.infer<typeof ModelRequestSchema>;
export type StopReason = z.infer<typeof StopReasonSchema>;
export type ContentBlock = z.infer<typeof ContentBlockSchema>;
export type TokenUsage = z.infer<typeof TokenUsageSchema>;
export type ModelResponse = z.infer<typeof ModelResponseSchema>;
export type StreamEvent = z.infer<typeof StreamEventSchema>;
export type ProviderInfo = z.infer<typeof ProviderInfoSchema>;
export type RecordedExchange = z.infer<typeof RecordedExchangeSchema>;
