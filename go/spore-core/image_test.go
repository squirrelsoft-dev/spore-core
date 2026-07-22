package sporecore

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMediaTypeMapsKnownExtensionsCaseInsensitively(t *testing.T) {
	cases := []struct {
		ext    string
		want   string
		wantOK bool
	}{
		{"png", "image/png", true},
		{"JPG", "image/jpeg", true},
		{"jpeg", "image/jpeg", true},
		{"WebP", "image/webp", true},
		{"gif", "image/gif", true},
		{"txt", "", false},
		{"", "", false},
	}
	for _, c := range cases {
		got, ok := MediaTypeForExtension(c.ext)
		if got != c.want || ok != c.wantOK {
			t.Errorf("MediaTypeForExtension(%q) = (%q,%v), want (%q,%v)", c.ext, got, ok, c.want, c.wantOK)
		}
	}
}

func TestIsImagePathChecksExtension(t *testing.T) {
	if !IsImagePath("a/b/shot.png") {
		t.Error("a/b/shot.png should be an image path")
	}
	if !IsImagePath("UP.JPEG") {
		t.Error("UP.JPEG should be an image path")
	}
	if IsImagePath("notes.md") {
		t.Error("notes.md should not be an image path")
	}
	if IsImagePath("noext") {
		t.Error("noext should not be an image path")
	}
}

func TestImageContentBase64EncodesBytes(t *testing.T) {
	// "hi" -> base64 "aGk=".
	c := ImageContent("image/png", []byte("hi"))
	if c.Type != ContentTypeImage {
		t.Fatalf("type = %v, want image", c.Type)
	}
	if c.MediaType != "image/png" {
		t.Fatalf("media_type = %q, want image/png", c.MediaType)
	}
	if c.Data != "aGk=" {
		t.Fatalf("data = %q, want aGk=", c.Data)
	}
}

func TestLoadImageContentRoundTrips(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "shot.png")
	if err := os.WriteFile(p, []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := LoadImageContent(p)
	if err != nil {
		t.Fatalf("LoadImageContent: %v", err)
	}
	if c.Type != ContentTypeImage || c.MediaType != "image/png" || c.Data != "aGk=" {
		t.Fatalf("unexpected content: %+v", c)
	}
}

func TestLoadImageContentRejectsUnsupportedExtension(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "notes.txt")
	_ = os.WriteFile(p, []byte("hi"), 0o644)
	if _, err := LoadImageContent(p); err == nil {
		t.Fatal("expected an error for an unsupported extension")
	}
}
