/**
 * Public re-exports for the canonical {@link ObservabilityProvider}
 * (spore-core issue #12). Re-exported under the `observability` namespace at
 * the package root for symmetry with `memory`, `context`, `guideRegistry`,
 * `sensor`, and `middleware`.
 */

export * from "./types.js";
export { InMemoryObservabilityProvider } from "./in-memory.js";
export {
  OutboxObservabilityProvider,
  SessionNotFoundError,
  TraceLine,
  attributesToOtelAttributes,
  emitGenaiEvents,
  outboxConfig,
} from "./outbox.js";
export type { GenAiSpanEvent } from "./outbox.js";
export type { OutboxConfig } from "./outbox.js";
export type { OtlpForwarder } from "./outbox.js";
