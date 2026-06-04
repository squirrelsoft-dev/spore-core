/**
 * `Harness` — drives the agent runtime loop (spore-core issue #3).
 *
 * Owns the execution lifecycle and wires sibling components together. It is
 * stateless between `run` calls; everything the harness needs comes in via
 * {@link HarnessRunOptions} or {@link PausedState}, and everything it produces
 * goes out via {@link RunResult}.
 */

import type {
  ConsultResponse,
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

  /**
   * Resume a worker paused by {@link RunResult} `consult` (issue #114). The
   * resume seam parallel to {@link resume}: it injects the
   * {@link ConsultResponse} as the tool RESULT of the head pending consult call,
   * dispatches any remaining pending calls, and resumes the loop.
   *
   * OPTIONAL so harnesses that never participate in consults (lightweight test
   * doubles, non-standard harnesses) need not implement it — the TS analogue of
   * the Rust trait's default impl. {@link "./standard.js".StandardHarness}
   * implements it with the real behaviour; the {@link "@spore/tools".SubagentTool}
   * mediator treats its absence as a non-resumable child.
   */
  resumeConsult?(
    state: PausedState,
    response: ConsultResponse,
    onStream?: StreamSink,
    signal?: AbortSignal,
  ): Promise<RunResult>;
}
