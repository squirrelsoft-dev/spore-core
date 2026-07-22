/**
 * Multimodal image input — read an image file into a base64 `Content::Image`
 * block the vision adapters forward to the model.
 *
 * Mirrors `rust/crates/spore-core/src/image.rs`. The single producer of image
 * content for both entry paths: a user attaching an image to a prompt, and a
 * `view_image` tool the agent calls. The media type is derived from the file
 * extension (the model APIs key off it).
 */

import { Buffer } from "node:buffer";
import { readFile } from "node:fs/promises";
import { extname } from "node:path";

import type { Content } from "./schemas.js";

export type ImageContentErrorKind =
  | { kind: "unsupported_extension"; extension: string; expected: readonly string[] };

/** Typed public image helper error (no plain package-boundary Error). */
export class ImageContentError extends Error {
  override readonly name = "ImageContentError";

  constructor(readonly detail: ImageContentErrorKind) {
    super(imageContentErrorMessage(detail));
  }
}

export function imageContentErrorMessage(detail: ImageContentErrorKind): string {
  switch (detail.kind) {
    case "unsupported_extension":
      return `unsupported image extension: ${detail.extension === "" ? '\"\"' : detail.extension} (expected ${detail.expected.join("/")})`;
  }
}

const SUPPORTED_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "webp"] as const;

/**
 * The image media type for a file extension, or `null` if it is not a format
 * the vision adapters accept. Case-insensitive.
 */
export function mediaTypeForExtension(ext: string): string | null {
  switch (ext.toLowerCase()) {
    case "png":
      return "image/png";
    case "jpg":
    case "jpeg":
      return "image/jpeg";
    case "gif":
      return "image/gif";
    case "webp":
      return "image/webp";
    default:
      return null;
  }
}

/** `true` if `path` ends in a recognized image extension. */
export function isImagePath(path: string): boolean {
  const ext = extname(path);
  if (ext === "") return false;
  // extname includes the leading dot; strip it.
  return mediaTypeForExtension(ext.slice(1)) != null;
}

/**
 * Base64-encode raw image `bytes` of `mediaType` into a `Content` image block.
 * Pure (no I/O) so callers that already hold bytes can reuse the same encoding
 * the file-path helper uses.
 */
export function imageContent(mediaType: string, bytes: Uint8Array): Content {
  return { type: "image", media_type: mediaType, data: toBase64(bytes) };
}

/**
 * Read an image file at `path` into a `Content` image block. Rejects if the
 * extension is unrecognized or the file cannot be read.
 */
export async function loadImageContent(path: string, signal?: AbortSignal): Promise<Content> {
  const ext = extname(path);
  const mediaType = mediaTypeForExtension(ext.slice(1));
  if (mediaType == null) {
    throw new ImageContentError({
      kind: "unsupported_extension",
      extension: ext,
      expected: SUPPORTED_IMAGE_EXTENSIONS,
    });
  }
  const bytes = await readFile(path, { signal });
  return imageContent(mediaType, bytes);
}

/** Base64-encode bytes (Node `Buffer`). */
function toBase64(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString("base64");
}