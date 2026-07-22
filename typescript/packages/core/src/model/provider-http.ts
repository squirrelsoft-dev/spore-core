/**
 * Shared HTTP retry plumbing for hosted providers — the TS analogue of Rust's
 * `provider_http` module. Reused by the embedding providers
 * ({@link "../embedding/standard.js"}). The existing chat providers
 * (`anthropic.ts`, `openai.ts`) keep their own copies untouched to avoid any
 * behavior change; this helper is additive.
 *
 * - Retryable status set: `408 | 425 | 429 | 500 | 502 | 503 | 504`
 *   (mirrors `rust/.../provider_http.rs::is_retryable_status`).
 * - Exponential backoff with a 30s ceiling: 0.5s, 1s, 2s, 4s, …
 *   (mirrors `backoff_delay`).
 * - Honors a `retry-after` header (whole seconds) over the computed backoff.
 */

import { ProviderError, Timeout, Transport, type ModelError } from "./errors.js";

/** Retryable HTTP status codes shared by the hosted providers. */
export const RETRYABLE_STATUSES: ReadonlySet<number> = new Set([
  408, 425, 429, 500, 502, 503, 504,
]);

/**
 * Exponential backoff with a 30s ceiling: 500ms, 1s, 2s, 4s, … Pure function.
 * Mirrors Rust `backoff_delay` and `model/openai.ts::backoffDelayMs`.
 */
export function backoffDelayMs(attempt: number): number {
  const shift = Math.min(Math.max(attempt, 0), 6);
  const base = 500 * (1 << shift);
  return Math.min(base, 30_000);
}

/** Parse a `retry-after` header (whole seconds) into ms. `null` if absent/invalid. */
export function parseRetryAfterMs(header: string | null): number | null {
  if (header == null) return null;
  const n = parseInt(header.trim(), 10);
  if (!Number.isFinite(n) || n < 0) return null;
  return n * 1000;
}

/**
 * Read an API key from `envVar`, rejecting an unset or empty value with a
 * `ProviderError`. Mirrors Rust `key_from_env`.
 */
export function keyFromEnv(envVar: string): string {
  const key = process.env[envVar];
  if (key == null) {
    throw new ProviderError(0, `env var \`${envVar}\` not set`);
  }
  if (key.trim() === "") {
    throw new ProviderError(0, `env var \`${envVar}\` is empty`);
  }
  return key;
}

/** A function that builds a `fetch` request for one attempt (re-invoked per retry). */
export type RequestBuilder = (signal: AbortSignal) => Promise<Response>;

/** Sleep hook — tests override to skip waits. Defaults to `setTimeout`. */
export type SleepFn = (ms: number, signal?: AbortSignal) => Promise<void>;

const defaultSleep: SleepFn = (ms, signal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new ProviderError(0, `request aborted: ${abortMessage(signal.reason)}`));
      return;
    }
    const cleanup = (): void => signal?.removeEventListener("abort", onAbort);
    const onAbort = (): void => {
      clearTimeout(timer);
      cleanup();
      reject(new ProviderError(0, `request aborted: ${abortMessage(signal?.reason)}`));
    };
    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });

/**
 * Retrying HTTP sender. `requestBuilder` is re-invoked each attempt (so a
 * consumed body can be rebuilt); `retryable` decides per-status whether to
 * retry; `mapError` converts a terminal non-retryable response into a
 * `ModelError`. Mirrors Rust `provider_http::send_with_retry`.
 */
export async function sendWithRetry(
  requestBuilder: RequestBuilder,
  maxRetries: number,
  timeoutMs: number,
  retryable: (code: number) => boolean,
  mapError: (resp: Response) => Promise<ModelError>,
  signal?: AbortSignal,
  sleep: SleepFn = defaultSleep,
): Promise<Response> {
  let attempt = 0;
  while (true) {
    let resp: Response;
    try {
      resp = await fetchWithTimeout(requestBuilder, timeoutMs, signal);
    } catch (e) {
      // Timeout: retry while budget remains, else surface Timeout.
      if (isTimeoutError(e) && attempt < maxRetries) {
        await sleepWithAbort(sleep, backoffDelayMs(attempt), signal);
        attempt += 1;
        continue;
      }
      if (isTimeoutError(e)) throw new Timeout();
      if (isAbortError(e)) {
        const err = e as Error;
        throw new ProviderError(0, `request aborted: ${err.message}`);
      }
      throw new Transport(`HTTP transport error: ${formatError(e)}`);
    }
    if (resp.ok) return resp;
    const code = resp.status;
    if (retryable(code) && attempt < maxRetries) {
      const delayMs = parseRetryAfterMs(resp.headers.get("retry-after")) ?? backoffDelayMs(attempt);
      try {
        await resp.text();
      } catch {
        /* ignore */
      }
      await sleepWithAbort(sleep, delayMs, signal);
      attempt += 1;
      continue;
    }
    throw await mapError(resp);
  }
}

async function sleepWithAbort(
  sleep: SleepFn,
  ms: number,
  signal: AbortSignal | undefined,
): Promise<void> {
  if (signal?.aborted) throw new ProviderError(0, `request aborted: ${abortMessage(signal.reason)}`);
  await sleep(ms, signal);
  if (signal?.aborted) throw new ProviderError(0, `request aborted: ${abortMessage(signal.reason)}`);
}

function abortMessage(reason: unknown): string {
  if (reason instanceof Error) return reason.message;
  if (typeof reason === "string") return reason;
  return "aborted";
}

/** Fetch with a timeout via `AbortController`, composing with an optional user signal. */
async function fetchWithTimeout(
  requestBuilder: RequestBuilder,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new TimeoutSentinel()), timeoutMs);
  const onUserAbort = (): void => controller.abort(signal?.reason);
  if (signal != null) {
    if (signal.aborted) controller.abort(signal.reason);
    else signal.addEventListener("abort", onUserAbort, { once: true });
  }
  try {
    // The builder composes the controller signal with any user signal via
    // the `onUserAbort` listener above; passing `controller.signal` lets fetch
    // abort on timeout OR user abort.
    return await requestBuilder(controller.signal);
  } finally {
    clearTimeout(timer);
    if (signal != null) signal.removeEventListener("abort", onUserAbort);
  }
}

class TimeoutSentinel extends Error {
  constructor() {
    super("provider-http-timeout");
    this.name = "TimeoutSentinel";
  }
}

function isTimeoutError(e: unknown): boolean {
  if (e instanceof Error) {
    if (e.name === "TimeoutSentinel") return true;
    if (e instanceof Timeout) return true;
    if (/timeout|ETIMEDOUT/i.test(e.message)) return true;
    // An AbortError whose cause is the timeout sentinel is a timeout.
    const cause = (e as { cause?: unknown }).cause;
    if (cause instanceof TimeoutSentinel) return true;
  }
  return false;
}

function isAbortError(e: unknown): boolean {
  return e instanceof Error && e.name === "AbortError";
}

function formatError(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

/**
 * Consume a non-success response into `{ code, message }`: pulls the message
 * from the typed `{error:{message}}` body, falling back to the first 500 chars
 * of the raw body. Mirrors Rust `error_parts`.
 */
export async function errorParts(resp: Response): Promise<{ code: number; message: string }> {
  const code = resp.status;
  let bodyText = "";
  try {
    bodyText = await resp.text();
  } catch {
    /* ignore */
  }
  let message = bodyText.slice(0, 500);
  try {
    const parsed = JSON.parse(bodyText) as { error?: { message?: string } };
    if (parsed?.error?.message != null) message = parsed.error.message;
  } catch {
    /* keep truncated body */
  }
  return { code, message };
}