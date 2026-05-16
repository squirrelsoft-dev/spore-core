/**
 * Typed harness errors for {@link ModelInterface}.
 *
 * Each error is a class extending `Error` with a discriminant `kind` field so
 * call-sites can exhaustively `switch` on `err.kind`. Mirrors the Rust
 * `ModelError` enum and serialises with the same `{ "kind": ... }` wire shape.
 */

export type ModelErrorKind =
  | "provider_error"
  | "rate_limited"
  | "context_limit_exceeded"
  | "budget_exceeded"
  | "timeout";

export abstract class ModelError extends Error {
  abstract readonly kind: ModelErrorKind;

  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }

  /** JSON wire shape — matches Rust `#[serde(tag = "kind")]`. */
  abstract toJSON(): Record<string, unknown>;
}

export class ProviderError extends ModelError {
  readonly kind = "provider_error" as const;
  constructor(
    readonly code: number,
    message: string,
  ) {
    super(`provider error ${code}: ${message}`);
  }
  toJSON() {
    return { kind: this.kind, code: this.code, message: this.providerMessage };
  }
  get providerMessage(): string {
    // Strip the "provider error N: " prefix we added for Error.message.
    return this.message.replace(/^provider error \d+: /, "");
  }
}

export class RateLimited extends ModelError {
  readonly kind = "rate_limited" as const;
  /** Seconds. `null` mirrors Rust `Option<Duration>` (None). */
  readonly retryAfter: number | null;
  constructor(retryAfter?: number | null) {
    super(`rate limited (retry_after=${retryAfter ?? "none"})`);
    this.retryAfter = retryAfter ?? null;
  }
  toJSON() {
    return { kind: this.kind, retry_after: this.retryAfter };
  }
}

export class ContextLimitExceeded extends ModelError {
  readonly kind = "context_limit_exceeded" as const;
  constructor(
    readonly limit: number,
    readonly actual: number,
  ) {
    super(`context limit exceeded: ${actual} tokens > limit ${limit}`);
  }
  toJSON() {
    return { kind: this.kind, limit: this.limit, actual: this.actual };
  }
}

export class BudgetExceeded extends ModelError {
  readonly kind = "budget_exceeded" as const;
  constructor(
    readonly budget: number,
    readonly used: number,
  ) {
    super(`budget exceeded: ${used} > budget ${budget}`);
  }
  toJSON() {
    return { kind: this.kind, budget: this.budget, used: this.used };
  }
}

export class Timeout extends ModelError {
  readonly kind = "timeout" as const;
  constructor() {
    super("model call timed out");
  }
  toJSON() {
    return { kind: this.kind };
  }
}

/**
 * Pre-call context-window check. Returns `null` on success, a typed error
 * otherwise. We return rather than throw so call-sites can decide how to
 * surface the error and stay friendly to async pipelines.
 */
export function enforceContextLimit(
  actual: number,
  limit: number,
): ContextLimitExceeded | null {
  if (actual > limit) return new ContextLimitExceeded(limit, actual);
  return null;
}

/**
 * Post-call budget check against the harness-injected
 * `ModelParams.max_tokens`. Returns `null` on success, a typed error
 * otherwise. `budget` undefined means "no budget configured".
 */
export function enforceBudget(
  used: number,
  budget?: number | null,
): BudgetExceeded | null {
  if (budget != null && used > budget) return new BudgetExceeded(budget, used);
  return null;
}
