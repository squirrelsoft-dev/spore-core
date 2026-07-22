import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  ImageContentError,
  imageContent,
  isImagePath,
  loadImageContent,
  mediaTypeForExtension,
} from "../src/model/image.js";

describe("image helpers", () => {
  it("mediaTypeForExtension is case-insensitive over png/jpg/jpeg/gif/webp", () => {
    expect(mediaTypeForExtension("png")).toBe("image/png");
    expect(mediaTypeForExtension("JPG")).toBe("image/jpeg");
    expect(mediaTypeForExtension("jpeg")).toBe("image/jpeg");
    expect(mediaTypeForExtension("gif")).toBe("image/gif");
    expect(mediaTypeForExtension("webp")).toBe("image/webp");
    expect(mediaTypeForExtension("bmp")).toBeNull();
    expect(mediaTypeForExtension("")).toBeNull();
  });

  it("isImagePath recognizes recognized extensions", () => {
    expect(isImagePath("a/b/c.PNG")).toBe(true);
    expect(isImagePath("photo.jpeg")).toBe(true);
    expect(isImagePath("noext")).toBe(false);
    expect(isImagePath("file.txt")).toBe(false);
  });

  it("imageContent base64-encodes bytes into an image ContentBlock", () => {
    const block = imageContent("image/png", new Uint8Array([1, 2, 3, 255]));
    expect(block.type).toBe("image");
    expect(block).toMatchObject({ type: "image", media_type: "image/png" });
    // base64 of [1,2,3,255]
    expect((block as { data: string }).data).toBe("AQID/w==");
  });

  it("loadImageContent reads + encodes a real file", async () => {
    const dir = await mkdtemp(join(tmpdir(), "spore-img-"));
    try {
      const p = join(dir, "x.png");
      await writeFile(p, new Uint8Array([1, 2, 3, 255]));
      const block = await loadImageContent(p);
      expect(block).toMatchObject({ type: "image", media_type: "image/png" });
      expect((block as { data: string }).data).toBe("AQID/w==");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("loadImageContent rejects an unrecognized extension with typed error", async () => {
    const err = await loadImageContent("file.bmp").catch((e) => e);
    expect(err).toBeInstanceOf(ImageContentError);
    expect((err as ImageContentError).kind).toBe("unsupported_extension");
    expect((err as ImageContentError).detail.kind).toBe("unsupported_extension");
  });

  it("loadImageContent wraps file read failures in a typed error", async () => {
    const err = await loadImageContent("/definitely/not/found.png").catch((e) => e);
    expect(err).toBeInstanceOf(ImageContentError);
    expect((err as ImageContentError).kind).toBe("read_failed");
    expect((err as ImageContentError).detail.kind).toBe("read_failed");
  });

  it("loadImageContent passes AbortSignal to the file read and wraps aborts", async () => {
    const dir = await mkdtemp(join(tmpdir(), "spore-img-"));
    try {
      const p = join(dir, "x.png");
      await writeFile(p, new Uint8Array([1, 2, 3]));
      const controller = new AbortController();
      controller.abort();
      const err = await loadImageContent(p, controller.signal).catch((e) => e);
      expect(err).toBeInstanceOf(ImageContentError);
      expect((err as ImageContentError).kind).toBe("aborted");
      expect((err as ImageContentError).detail.kind).toBe("aborted");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});