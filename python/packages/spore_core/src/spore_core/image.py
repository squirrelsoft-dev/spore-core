"""Multimodal image input — read an image file into a base64
:class:`~spore_core.model.ImageContent` block the vision adapters forward to
the model.

Mirrors ``rust/crates/spore-core/src/image.rs``. This is the single producer of
image content for both entry paths: a user attaching an image to a prompt, and
a ``view_image`` tool the agent calls. The media type is derived from the file
extension (the model APIs key off it).
"""

from __future__ import annotations

import base64
from pathlib import Path

from .model import ImageContent

__all__ = [
    "image_content",
    "is_image_path",
    "load_image_content",
    "media_type_for_extension",
]

_KNOWN: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


def media_type_for_extension(ext: str) -> str | None:
    """The image media type for a file extension, or ``None`` if it is not a
    format the vision adapters accept. Case-insensitive."""
    return _KNOWN.get(ext.lower())


def is_image_path(path: Path) -> bool:
    """``True`` if ``path`` ends in a recognized image extension."""
    ext = path.suffix.lstrip(".")
    return media_type_for_extension(ext) is not None


def image_content(media_type: str, bytes_: bytes) -> ImageContent:
    """Base64-encode raw image ``bytes`` of ``media_type`` into an
    :class:`~spore_core.model.ImageContent`. Pure (no I/O) so callers that
    already hold bytes (e.g. a sandbox read) can reuse the same encoding the
    file path helper uses."""
    return ImageContent(media_type=media_type, data=base64.b64encode(bytes_).decode("ascii"))


def load_image_content(path: Path) -> ImageContent:
    """Read an image file at ``path`` into an :class:`~spore_core.model.ImageContent`
    block. Errors with a ``ValueError`` if the extension is unrecognized or the
    file cannot be read."""
    ext = path.suffix.lstrip(".")
    media_type = media_type_for_extension(ext)
    if media_type is None:
        raise ValueError(f"unsupported image extension: {ext!r} (expected png/jpg/jpeg/gif/webp)")
    data = path.read_bytes()
    return image_content(media_type, data)
