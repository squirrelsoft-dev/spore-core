/**
 * `MockAgent` — programmable mock for unit tests. Each `turn` pops the next
 * queued {@link TurnResult}. Falls back to `EmptyResponse` if the queue is
 * empty, mirroring the Rust reference.
 */

import { EmptyResponse } from "./errors.js";
import type { Agent } from "./interface.js";
import type { AgentId, Context, TurnResult } from "./types.js";

export class MockAgent implements Agent {
  private readonly results: TurnResult[] = [];
  callCount = 0;

  constructor(private readonly agentId: AgentId) {}

  push(result: TurnResult): this {
    this.results.push(result);
    return this;
  }

  id(): AgentId {
    return this.agentId;
  }

  async turn(_context: Context, _signal?: AbortSignal): Promise<TurnResult> {
    this.callCount += 1;
    const next = this.results.shift();
    if (next == null) {
      return { kind: "error", error: new EmptyResponse(), usage: null };
    }
    return next;
  }
}
