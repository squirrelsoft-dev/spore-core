/**
 * @spore/core — harness runtime library.
 *
 * Implements the language-agnostic spec in
 * `docs/harness-engineering-concepts.md`.
 *
 * Components:
 *   #1  ModelInterface
 *   #2  Agent (one turn)
 *   #3  Harness runtime loop
 *   #4  ToolRegistry
 *   #5  Tool trait and base implementations
 *   #6  SandboxProvider
 *   #7  ContextManager
 *   #8  MemoryProvider
 *   #9  GuideRegistry
 *   #10 SensorChain
 *   #11 Middleware Chain
 *   #12 ObservabilityProvider
 *   #13 TerminationPolicy
 */
export * from "./model/index.js";
export * from "./agent/index.js";
export * from "./harness/index.js";
export * as toolRegistry from "./tool-registry/index.js";
export * from "./sandbox/index.js";
export * as context from "./context/index.js";
export * as memory from "./memory/index.js";
export * as guideRegistry from "./guide-registry/index.js";
export * as sensor from "./sensor/index.js";
export * as middleware from "./middleware/index.js";
export * as observability from "./observability/index.js";
