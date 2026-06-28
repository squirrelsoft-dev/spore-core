//! Multimodal image input — read an image file into a base64 [`Content::Image`]
//! block the vision adapters forward to the model.
//!
//! This is the single producer of image content for both entry paths: a user
//! attaching an image to a prompt, and a `view_image` tool the agent calls. The
//! media type is derived from the file extension (the model APIs key off it).

use std::path::Path;

use base64::Engine;

use crate::model::Content;

/// The image media type for a file extension, or `None` if it is not a format
/// the vision adapters accept. Case-insensitive.
pub fn media_type_for_extension(ext: &str) -> Option<&'static str> {
    match ext.to_ascii_lowercase().as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

/// `true` if `path` ends in a recognized image extension.
pub fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .and_then(media_type_for_extension)
        .is_some()
}

/// Base64-encode raw image `bytes` of `media_type` into a [`Content::Image`].
/// Pure (no I/O) so callers that already hold bytes (e.g. a sandbox read) can
/// reuse the same encoding the file path helper uses.
pub fn image_content(media_type: impl Into<String>, bytes: &[u8]) -> Content {
    Content::Image {
        media_type: media_type.into(),
        data: base64::engine::general_purpose::STANDARD.encode(bytes),
    }
}

/// Read an image file at `path` into a [`Content::Image`] block. Errors if the
/// extension is unrecognized or the file cannot be read.
pub fn load_image_content(path: &Path) -> std::io::Result<Content> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let media_type = media_type_for_extension(ext).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unsupported image extension: {ext:?} (expected png/jpg/jpeg/gif/webp)"),
        )
    })?;
    let bytes = std::fs::read(path)?;
    Ok(image_content(media_type, &bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_type_maps_known_extensions_case_insensitively() {
        assert_eq!(media_type_for_extension("png"), Some("image/png"));
        assert_eq!(media_type_for_extension("JPG"), Some("image/jpeg"));
        assert_eq!(media_type_for_extension("jpeg"), Some("image/jpeg"));
        assert_eq!(media_type_for_extension("WebP"), Some("image/webp"));
        assert_eq!(media_type_for_extension("gif"), Some("image/gif"));
        assert_eq!(media_type_for_extension("txt"), None);
        assert_eq!(media_type_for_extension(""), None);
    }

    #[test]
    fn is_image_path_checks_extension() {
        assert!(is_image_path(Path::new("a/b/shot.png")));
        assert!(is_image_path(Path::new("UP.JPEG")));
        assert!(!is_image_path(Path::new("notes.md")));
        assert!(!is_image_path(Path::new("noext")));
    }

    #[test]
    fn image_content_base64_encodes_bytes() {
        // "hi" -> base64 "aGk=" (a known vector), wrapped as a Content::Image.
        let c = image_content("image/png", b"hi");
        match c {
            Content::Image { media_type, data } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(data, "aGk=");
            }
            _ => panic!("expected Content::Image"),
        }
    }
}
