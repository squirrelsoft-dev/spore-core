"""Tests for the image helpers (parity plan §5.15).

Mirrors the inline tests in ``rust/crates/spore-core/src/image.rs``.
"""

from __future__ import annotations

from pathlib import Path

from spore_core import (
    ImageContent,
    image_content,
    is_image_path,
    load_image_content,
    media_type_for_extension,
)


def test_media_type_maps_known_extensions_case_insensitively() -> None:
    assert media_type_for_extension("png") == "image/png"
    assert media_type_for_extension("JPG") == "image/jpeg"
    assert media_type_for_extension("jpeg") == "image/jpeg"
    assert media_type_for_extension("WebP") == "image/webp"
    assert media_type_for_extension("gif") == "image/gif"
    assert media_type_for_extension("txt") is None
    assert media_type_for_extension("") is None


def test_is_image_path_checks_extension() -> None:
    assert is_image_path(Path("a/b/shot.png")) is True
    assert is_image_path(Path("UP.JPEG")) is True
    assert is_image_path(Path("notes.md")) is False
    assert is_image_path(Path("noext")) is False


def test_image_content_base64_encodes_bytes() -> None:
    # "hi" -> base64 "aGk=" (a known vector), wrapped as ImageContent.
    c = image_content("image/png", b"hi")
    assert isinstance(c, ImageContent)
    assert c.media_type == "image/png"
    assert c.data == "aGk="


def test_load_image_content_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "shot.png"
    p.write_bytes(b"hi")
    c = load_image_content(p)
    assert c.media_type == "image/png"
    assert c.data == "aGk="


def test_load_image_content_unsupported_extension(tmp_path: Path) -> None:
    p = tmp_path / "notes.txt"
    p.write_bytes(b"hi")
    try:
        load_image_content(p)
    except ValueError as e:
        assert "unsupported image extension" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unsupported extension")
