package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRecordingAppendsRequestResponsePair(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "recorded.jsonl")
	inner := NewMockModel(testProvider())
	inner.PushResponse(respText("hello back"))
	inner.PushResponse(respText("hello again"))
	r := NewRecordingModel(inner, path, RecordingModeRecord)
	ctx := context.Background()
	if _, err := r.Call(ctx, reqText("hello")); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Call(ctx, reqText("hello2")); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("lines = %d, want 2: %q", len(lines), raw)
	}
	for i, line := range lines {
		var entry RecordedExchange
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			t.Fatalf("line %d unmarshal: %v", i, err)
		}
		if entry.RequestHash == "" {
			t.Errorf("line %d: request_hash must be populated", i)
		}
		if entry.Provider != "test" {
			t.Errorf("line %d: provider = %q, want test", i, entry.Provider)
		}
		if entry.ModelID != "test-1" {
			t.Errorf("line %d: model_id = %q, want test-1", i, entry.ModelID)
		}
	}
}

func TestRecordingRecordIfNewSkipsWhenFileExists(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "existing.jsonl")
	if err := os.WriteFile(path, []byte("preexisting line\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	inner := NewMockModel(testProvider())
	inner.PushResponse(respText("ok"))
	r := NewRecordingModel(inner, path, RecordingModeRecordIfNew)
	if _, err := r.Call(context.Background(), reqText("q")); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != "preexisting line\n" {
		t.Fatalf("RecordIfNew must not touch existing file, got %q", raw)
	}
}

func TestRecordingRecordIfNewWritesWhenFileMissing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "fresh.jsonl")
	inner := NewMockModel(testProvider())
	inner.PushResponse(respText("ok"))
	r := NewRecordingModel(inner, path, RecordingModeRecordIfNew)
	if _, err := r.Call(context.Background(), reqText("q")); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) == 0 {
		t.Fatal("RecordIfNew should write when file is missing")
	}
}

func TestRecordingPassthroughDoesNotWrite(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nope.jsonl")
	inner := NewMockModel(testProvider())
	inner.PushResponse(respText("ok"))
	r := NewRecordingModel(inner, path, RecordingModePassthrough)
	if _, err := r.Call(context.Background(), reqText("q")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("Passthrough must not create the file, stat err=%v", err)
	}
}

func TestRecordingThenReplayRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "roundtrip.jsonl")
	inner := NewMockModel(testProvider())
	inner.PushResponse(respText("answer1"))
	inner.PushResponse(respText("answer2"))
	recorder := NewRecordingModel(inner, path, RecordingModeRecord)
	ctx := context.Background()
	q1 := reqText("question 1")
	q2 := reqText("question 2")
	if _, err := recorder.Call(ctx, q1); err != nil {
		t.Fatal(err)
	}
	if _, err := recorder.Call(ctx, q2); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := ParseReplayJSONL(string(raw), testProvider())
	if err != nil {
		t.Fatal(err)
	}
	if replay.Mode() != ReplayModeHashMatched {
		t.Fatalf("replay mode = %v, want HashMatched", replay.Mode())
	}
	// Out-of-order roundtrip.
	g2, err := replay.Call(ctx, q2)
	if err != nil || g2.Content[0].Text != "answer2" {
		t.Fatalf("q2 roundtrip: %+v %v", g2, err)
	}
	g1, err := replay.Call(ctx, q1)
	if err != nil || g1.Content[0].Text != "answer1" {
		t.Fatalf("q1 roundtrip: %+v %v", g1, err)
	}
}
