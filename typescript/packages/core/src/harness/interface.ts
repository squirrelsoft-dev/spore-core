/**
 * `Harness` — drives the agent runtime loop (spore-core issue #3).
 *
 * Owns the execution lifecycle and wires sibling components together. It is
 * stateless between `run` calls; everything the harness needs comes in via
 * {@link HarnessRunOptions} or {@link PausedState}, and everything it produces
 * goes out via {@link RunResult}.
 */

import type {
  HarnessRunOptions,
  HumanResponse,
  PausedState,
  RunResult,
  StreamSink,
} from "./types.js";

export interface Harness {
  run(options: HarnessRunOptions): Promise<RunResult>;

  resume(
    state: PausedState,
    response: HumanResponse,
    onStream?: StreamSink,
    signal?: AbortSignal,
  ): Promise<RunResult>;
}
