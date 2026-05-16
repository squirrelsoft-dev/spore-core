/**
 * `ModelInterface` — boundary between the harness and the underlying LLM.
 *
 * Implements spore-core issue #1. The harness only ever talks to a model
 * through this interface; provider-specific concerns (Anthropic, OpenAI,
 * Ollama, replay) live behind concrete implementations.
 *
 * Rules enforced (mirrored from the Rust reference):
 *   1. `TokenUsage` is reported on every successful call (`call` and the
 *      final `MessageStop` summary of `callStreaming`). Not optional.
 *   2. `ContextLimitExceeded` is reported by the implementation *before*
 *      the provider is contacted whenever `countTokens` exceeds the
 *      context window. The shared `enforceContextLimit` helper does the
 *      check.
 *   3. `BudgetExceeded` is a harness-side check against
 *      `ModelRequest.params.max_tokens`, surfaced as a typed error.
 *   4. Provider-specific retry/backoff lives in the implementation.
 */

import type {
  ModelRequest,
  ModelResponse,
  ProviderInfo,
  StreamEvent,
} from "./schemas.js";

export interface ModelInterface {
  /** One blocking model call. `TokenUsage` must be populated on success. */
  call(request: ModelRequest, signal?: AbortSignal): Promise<ModelResponse>;

  /**
   * Streaming variant. Yields `StreamEvent`s as they arrive; the final
   * `MessageStop` event carries the accumulated `TokenUsage`.
   */
  callStreaming(
    request: ModelRequest,
    signal?: AbortSignal,
  ): AsyncIterable<StreamEvent>;

  /**
   * Pre-call token count for context-size estimation. Used by the harness
   * to detect `ContextLimitExceeded` before contacting the provider.
   */
  countTokens(request: ModelRequest, signal?: AbortSignal): Promise<number>;

  /** Provider identity for tracing and routing decisions. */
  provider(): ProviderInfo;
}
