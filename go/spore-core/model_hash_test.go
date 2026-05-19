package sporecore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func reqText(s string) ModelRequest {
	return ModelRequest{
		Messages: []Message{{Role: RoleUser, Content: NewTextContent(s)}},
		Tools:    []ToolSchema{},
		Params:   ModelParams{},
		Stream:   false,
	}
}

func respText(s string) ModelResponse {
	return ModelResponse{
		Content:    []ContentBlock{NewTextBlock(s)},
		Usage:      TokenUsage{},
		StopReason: StopEndTurn,
	}
}

func TestRequestHashIsStable(t *testing.T) {
	a := reqText("hello world")
	b := reqText("hello world")
	if RequestHash(a) != RequestHash(b) {
		t.Fatalf("hash should be stable across equal requests")
	}
}

func TestRequestHashChangesWithMessages(t *testing.T) {
	if RequestHash(reqText("hello")) == RequestHash(reqText("hello!")) {
		t.Fatalf("hash should differ when messages differ")
	}
}

func TestRequestHashFormat(t *testing.T) {
	h := RequestHash(reqText("x"))
	if len(h) != 16 {
		t.Fatalf("hash length = %d, want 16", len(h))
	}
	for _, c := range h {
		isHex := (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')
		if !isHex {
			t.Fatalf("hash %q contains non-hex/non-lowercase character %q", h, c)
		}
	}
}

func TestReplayAutoDetectsPositionalWhenNoHashes(t *testing.T) {
	ex := []RecordedExchange{{
		Request:  reqText("q1"),
		Response: respText("r1"),
		Provider: "fixture",
	}}
	r := NewReplayModel(ex, testProvider())
	if r.Mode() != ReplayModePositional {
		t.Fatalf("Mode = %v, want Positional", r.Mode())
	}
	got, err := r.Call(context.Background(), reqText("anything"))
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got.Content[0].Text != "r1" {
		t.Fatalf("got text = %q, want %q", got.Content[0].Text, "r1")
	}
}

func TestReplayAutoDetectsHashMatchedWhenAllHaveHashes(t *testing.T) {
	q1 := reqText("q1")
	q2 := reqText("q2")
	ex := []RecordedExchange{
		{RequestHash: RequestHash(q1), Request: q1, Response: respText("r1"), Provider: "fixture"},
		{RequestHash: RequestHash(q2), Request: q2, Response: respText("r2"), Provider: "fixture"},
	}
	r := NewReplayModel(ex, testProvider())
	if r.Mode() != ReplayModeHashMatched {
		t.Fatalf("Mode = %v, want HashMatched", r.Mode())
	}
	ctx := context.Background()
	// Out-of-order calls return the right response.
	g2, err := r.Call(ctx, q2)
	if err != nil || g2.Content[0].Text != "r2" {
		t.Fatalf("q2: got %+v err=%v", g2, err)
	}
	g1, err := r.Call(ctx, q1)
	if err != nil || g1.Content[0].Text != "r1" {
		t.Fatalf("q1: got %+v err=%v", g1, err)
	}
	g2b, err := r.Call(ctx, q2)
	if err != nil || g2b.Content[0].Text != "r2" {
		t.Fatalf("q2 (repeat): got %+v err=%v", g2b, err)
	}
}

func TestReplayHashMatchedNoMatchReturnsProviderError(t *testing.T) {
	q1 := reqText("q1")
	ex := []RecordedExchange{
		{RequestHash: RequestHash(q1), Request: q1, Response: respText("r1"), Provider: "fixture"},
	}
	r := NewReplayModel(ex, testProvider())
	_, err := r.Call(context.Background(), reqText("unrecorded"))
	if err == nil {
		t.Fatalf("expected error")
	}
	var me *ModelError
	if !errors.As(err, &me) || me.Kind != ModelErrProviderError {
		t.Fatalf("expected ProviderError, got %v", err)
	}
	if !strings.Contains(me.Message, "no matching fixture") {
		t.Fatalf("expected 'no matching fixture' in message, got %q", me.Message)
	}
}

func TestFixtureReplayRequestHashStability(t *testing.T) {
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	path := filepath.Join(dir, "..", "..", "fixtures", "model_hashing", "cases.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var suite struct {
		Cases []struct {
			Name         string       `json:"name"`
			Request      ModelRequest `json:"request"`
			ExpectedHash string       `json:"expected_hash"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &suite); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(suite.Cases) == 0 {
		t.Fatal("no cases in fixture")
	}
	for _, c := range suite.Cases {
		got := RequestHash(c.Request)
		if got != c.ExpectedHash {
			t.Errorf("case %q: got %s, want %s", c.Name, got, c.ExpectedHash)
		}
	}
}
