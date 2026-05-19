/**
 * `RecordingModelInterface` — transparent wrapper that appends each
 * `(request, response)` pair to a JSONL fixture file as a `RecordedExchange`
 * carrying a stable `requestHash` (spore-core issue #38).
 *
 * Three modes:
 *   - `record`: append every pair to `outputPath`.
 *   - `record_if_new`: append only if `outputPath` does not yet exist (checked
 *     at write time, not construction). Useful for "record once, then replay
 *     forever" workflows.
 *   - `passthrough`: call the inner model but never write.
 *
 * Streaming calls pass straight through — the spec only requires the blocking
 * `call()` pair to be recorded.
 */

import { appendFile, mkdir, stat } from "node:fs/promises";
import { dirname } from "node:path";

import { ProviderError } from "./errors.js";
import { requestHash } from "./hash.js";
import type { ModelInterface } from "./interface.js";
import type {
  ModelRequest,
  ModelResponse,
  ProviderInfo,
  RecordedExchange,
  StreamEvent,
} from "./schemas.js";

export type RecordingMode = "record" | "record_if_new" | "passthrough";

export class RecordingModelInterface implements ModelInterface {
  /** Serialises concurrent recording writes so we don't double-write. */
  private writeChain: Promise<void> = Promise.resolve();

  constructor(
    private readonly inner: ModelInterface,
    private readonly outputPath: string,
    private readonly modeValue: RecordingMode,
  ) {}

  outputPathValue(): string {
    return this.outputPath;
  }

  mode(): RecordingMode {
    return this.modeValue;
  }

  provider(): ProviderInfo {
    return this.inner.provider();
  }

  async call(request: ModelRequest, signal?: AbortSignal): Promise<ModelResponse> {
    const start = Date.now();
    const response = await this.inner.call(request, signal);
    const durationMs = Date.now() - start;
    try {
      await this.record(request, response, durationMs);
    } catch (err) {
      throw new ProviderError(0, `recorder write failed: ${(err as Error).message ?? String(err)}`);
    }
    return response;
  }

  async *callStreaming(request: ModelRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    // Spec: streaming recording is out of scope. Pass through.
    yield* this.inner.callStreaming(request, signal);
  }

  async countTokens(request: ModelRequest, signal?: AbortSignal): Promise<number> {
    return this.inner.countTokens(request, signal);
  }

  private async record(
    request: ModelRequest,
    response: ModelResponse,
    durationMs: number,
  ): Promise<void> {
    // Chain all writes so concurrent callers serialise and `record_if_new`
    // is observed atomically (otherwise two parallel callers would both see
    // the file missing and both write).
    const prev = this.writeChain;
    this.writeChain = prev.then(async () => {
      const shouldWrite = await this.shouldWrite();
      if (!shouldWrite) return;
      const providerInfo = this.inner.provider();
      const entry: RecordedExchange = {
        request_hash: requestHash(request),
        request,
        response,
        provider: providerInfo.name,
        model_id: providerInfo.model_id,
        duration_ms: durationMs,
      };
      const dir = dirname(this.outputPath);
      if (dir && dir !== ".") {
        await mkdir(dir, { recursive: true });
      }
      await appendFile(this.outputPath, serializeEntry(entry) + "\n", "utf8");
    });
    return this.writeChain;
  }

  private async shouldWrite(): Promise<boolean> {
    switch (this.modeValue) {
      case "record":
        return true;
      case "passthrough":
        return false;
      case "record_if_new":
        return !(await pathExists(this.outputPath));
      default: {
        const _exhaustive: never = this.modeValue;
        return _exhaustive;
      }
    }
  }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw err;
  }
}

/**
 * Serialise a `RecordedExchange` to JSON, omitting optional fields that are
 * `null`/`undefined` so the file stays byte-portable with Rust's
 * `#[serde(skip_serializing_if = "Option::is_none")]` semantics.
 */
function serializeEntry(entry: RecordedExchange): string {
  const out: Record<string, unknown> = {};
  if (entry.request_hash != null) out.request_hash = entry.request_hash;
  out.request = entry.request;
  out.response = entry.response;
  out.provider = entry.provider;
  if (entry.model_id != null) out.model_id = entry.model_id;
  if (entry.recorded_at != null) out.recorded_at = entry.recorded_at;
  if (entry.duration_ms != null) out.duration_ms = entry.duration_ms;
  return JSON.stringify(out);
}
