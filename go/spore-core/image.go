// Multimodal image input — read an image file into a base64 Content (image)
// block the vision adapters forward to the model.
//
// This is the single producer of image content for both entry paths: a user
// attaching an image to a prompt, and a view_image tool the agent calls. The
// media type is derived from the file extension (the model APIs key off it).
//
// Mirrors the Rust reference (image.rs).

package sporecore

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// MediaTypeForExtension returns the image media type for a file extension, or
// ("", false) if it is not a format the vision adapters accept. Case-insensitive.
// Recognized: png, jpg/jpeg, gif, webp.
func MediaTypeForExtension(ext string) (string, bool) {
	switch strings.ToLower(ext) {
	case "png":
		return "image/png", true
	case "jpg", "jpeg":
		return "image/jpeg", true
	case "gif":
		return "image/gif", true
	case "webp":
		return "image/webp", true
	default:
		return "", false
	}
}

// IsImagePath reports whether path ends in a recognized image extension.
func IsImagePath(path string) bool {
	ext := strings.TrimPrefix(filepath.Ext(path), ".")
	_, ok := MediaTypeForExtension(ext)
	return ok
}

// ImageContent base64-encodes raw image bytes of mediaType into a Content
// (image) block. Pure (no I/O) so callers that already hold bytes (e.g. a
// sandbox read) can reuse the same encoding the file path helper uses.
func ImageContent(mediaType string, bytes []byte) Content {
	return NewImageContent(mediaType, base64.StdEncoding.EncodeToString(bytes))
}

// LoadImageContent reads an image file at path into a Content (image) block.
// Returns an error if the extension is unrecognized or the file cannot be read.
func LoadImageContent(path string) (Content, error) {
	ext := strings.TrimPrefix(filepath.Ext(path), ".")
	mediaType, ok := MediaTypeForExtension(ext)
	if !ok {
		return Content{}, fmt.Errorf("unsupported image extension: %q (expected png/jpg/jpeg/gif/webp)", ext)
	}
	bytes, err := os.ReadFile(path)
	if err != nil {
		return Content{}, err
	}
	return ImageContent(mediaType, bytes), nil
}
