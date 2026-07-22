package openai

import (
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// TestExtendedSamplingEmittedForNonReasoning confirms the four extended sampling
// fields (top_k, min_p, presence_penalty, repetition_penalty) reach the wire for
// a non-reasoning model when set.
func TestExtendedSamplingEmittedForNonReasoning(t *testing.T) {
	r := req(userMsg("hi"))
	topK := uint32(40)
	minP := float32(0.05)
	pres := float32(0.6)
	rep := float32(1.1)
	r.Params.TopK = &topK
	r.Params.MinP = &minP
	r.Params.PresencePenalty = &pres
	r.Params.RepetitionPenalty = &rep
	body := buildRequest("gpt-4o", r, false, OpenAICompat{})
	if body.TopK == nil || *body.TopK != 40 {
		t.Fatalf("top_k = %v, want 40", body.TopK)
	}
	if body.MinP == nil || *body.MinP != 0.05 {
		t.Fatalf("min_p = %v, want 0.05", body.MinP)
	}
	if body.PresencePenalty == nil || *body.PresencePenalty != 0.6 {
		t.Fatalf("presence_penalty = %v, want 0.6", body.PresencePenalty)
	}
	if body.RepetitionPenalty == nil || *body.RepetitionPenalty != 1.1 {
		t.Fatalf("repetition_penalty = %v, want 1.1", body.RepetitionPenalty)
	}
}

// TestExtendedSamplingDroppedForReasoning confirms the four fields are dropped
// for an o-series reasoning model (which reject sampling-control params),
// exactly like temperature/top_p/stop.
func TestExtendedSamplingDroppedForReasoning(t *testing.T) {
	r := req(userMsg("hi"))
	topK := uint32(40)
	minP := float32(0.05)
	pres := float32(0.6)
	rep := float32(1.1)
	temp := float32(0.7)
	r.Params.TopK = &topK
	r.Params.MinP = &minP
	r.Params.PresencePenalty = &pres
	r.Params.RepetitionPenalty = &rep
	r.Params.Temperature = &temp
	body := buildRequest("o3", r, false, OpenAICompat{})
	if body.TopK != nil || body.MinP != nil || body.PresencePenalty != nil || body.RepetitionPenalty != nil {
		t.Fatalf("extended sampling must be nil for reasoning model: top_k=%v min_p=%v presence=%v repetition=%v",
			body.TopK, body.MinP, body.PresencePenalty, body.RepetitionPenalty)
	}
	if body.Temperature != nil {
		t.Fatalf("temperature must be nil for reasoning model: %v", body.Temperature)
	}
}

// TestExtendedSamplingAbsentByDefault confirms nil fields keep the wire
// byte-identical (omitempty drops them).
func TestExtendedSamplingAbsentByDefault(t *testing.T) {
	r := req(userMsg("hi"))
	body := buildRequest("gpt-4o", r, false, OpenAICompat{})
	if body.TopK != nil || body.MinP != nil || body.PresencePenalty != nil || body.RepetitionPenalty != nil {
		t.Fatalf("extended sampling must be absent by default")
	}
}

var _ = sporecore.RoleUser // keep the import used if future tests reference it
