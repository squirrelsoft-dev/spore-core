/**
 * Public re-exports for the `model` component (spore-core issue #1).
 */

export * from "./errors.js";
export * from "./interface.js";
export * from "./schemas.js";
export { requestHash, canonicalizeJson } from "./hash.js";
export { ReplayModelInterface, type ReplayMode } from "./replay.js";
export { RecordingModelInterface, type RecordingMode } from "./recording.js";
export { MockModelInterface } from "./mock.js";
export {
  PromptBasedToolCallModelInterface,
  AdaptiveToolCallModelInterface,
  buildToolPrompt,
  injectToolPrompt,
  parseProseResponse,
  detectProseResponse,
  newSharedFlag,
  PROMPT_TOOL_CALL_NUDGE,
  type SharedFlag,
} from "./prompt-tool-call.js";
export {
  AnthropicModelInterface,
  ANTHROPIC_VERSION,
  DEFAULT_BASE_URL as ANTHROPIC_DEFAULT_BASE_URL,
  DEFAULT_TIMEOUT_MS as ANTHROPIC_DEFAULT_TIMEOUT_MS,
  DEFAULT_MAX_RETRIES as ANTHROPIC_DEFAULT_MAX_RETRIES,
  backoffDelayMs as anthropicBackoffDelayMs,
  buildRequest as anthropicBuildRequest,
  parseResponse as anthropicParseResponse,
  parseStopReason as anthropicParseStopReason,
  parseSseEvent as anthropicParseSseEvent,
  sseToEvents as anthropicSseToEvents,
  type AnthropicModelInterfaceOptions,
} from "./anthropic.js";
export {
  OllamaModelInterface,
  DEFAULT_BASE_URL as OLLAMA_DEFAULT_BASE_URL,
  DEFAULT_TIMEOUT_MS as OLLAMA_DEFAULT_TIMEOUT_MS,
  DEFAULT_KEEP_ALIVE as OLLAMA_DEFAULT_KEEP_ALIVE,
  buildRequest as ollamaBuildRequest,
  parseResponse as ollamaParseResponse,
  parseStructuredContent as ollamaParseStructuredContent,
  parseStopReason as ollamaParseStopReason,
  ndjsonToEvents as ollamaNdjsonToEvents,
  nameMatches as ollamaNameMatches,
  type OllamaModelInterfaceOptions,
} from "./ollama.js";
export {
  OpenAIModelInterface,
  DEFAULT_BASE_URL as OPENAI_DEFAULT_BASE_URL,
  DEFAULT_TIMEOUT_MS as OPENAI_DEFAULT_TIMEOUT_MS,
  DEFAULT_MAX_RETRIES as OPENAI_DEFAULT_MAX_RETRIES,
  backoffDelayMs as openaiBackoffDelayMs,
  buildRequest as openaiBuildRequest,
  parseResponse as openaiParseResponse,
  parseStopReason as openaiParseStopReason,
  sseToEvents as openaiSseToEvents,
  type OpenAIModelInterfaceOptions,
} from "./openai.js";
