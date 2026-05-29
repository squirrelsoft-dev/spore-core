/**
 * Storage domain errors (spore-core issue #73).
 *
 * Follows `typescript/CONVENTIONS.md`: domain errors are classes extending
 * `Error` with `name` set to the class name and a discriminant `kind` field for
 * `switch` exhaustiveness. Mirrors the Rust `StorageError` variants
 * (`Io`, `Serialization`, `NotFound`, `Backend`) as a small class hierarchy
 * under a shared {@link StorageError} base.
 */

/** Base class for every storage error. Carries a discriminant `kind`. */
export abstract class StorageError extends Error {
  abstract readonly kind: "io" | "serialization" | "not_found" | "backend";
}

/** An I/O failure from a filesystem-backed store. The underlying error is
 *  attached via the standard {@link Error.cause} option. */
export class StorageIoError extends StorageError {
  readonly kind = "io" as const;
  constructor(message: string, cause?: unknown) {
    super(`storage I/O error: ${message}`, { cause });
    this.name = "StorageIoError";
  }
}

/** A (de)serialization failure. The underlying error is attached via the
 *  standard {@link Error.cause} option. */
export class StorageSerializationError extends StorageError {
  readonly kind = "serialization" as const;
  constructor(message: string, cause?: unknown) {
    super(`storage serialization error: ${message}`, { cause });
    this.name = "StorageSerializationError";
  }
}

/** A keyed lookup found nothing where the caller required a value. */
export class StorageNotFoundError extends StorageError {
  readonly kind = "not_found" as const;
  constructor(
    readonly domain: string,
    readonly key: string,
  ) {
    super(`storage not found: domain=${domain} key=${key}`);
    this.name = "StorageNotFoundError";
  }
}

/** A backend-specific failure that does not map to the variants above. */
export class StorageBackendError extends StorageError {
  readonly kind = "backend" as const;
  constructor(message: string) {
    super(`storage backend error: ${message}`);
    this.name = "StorageBackendError";
  }
}
