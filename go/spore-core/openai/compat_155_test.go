package openai

import (
	"encoding/json"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// SC-6 / SC-27 — OpenAI context-window override + with_compat (#155)
//
// Mirrors rust/crates/spore-core/src/openai.rs (f1c0beb). Every OpenAICompat
// field is OR'd over the o1/o3/o4 id heuristic; default (all false) keeps the
// recognized o-series byte-identical. Additive — no fixture re-baseline.
// ============================================================================

// SC-6: an unrecognized id reports 0; the override pins it.
func TestWithContextWindowOverridesReportedWindow(t *testing.T) {
	bare := New("k", "local-llama")
	if bare.Provider().ContextWindow != 0 {
		t.Fatalf("bare unrecognized window = %d, want 0", bare.Provider().ContextWindow)
	}
	pinned := New("k", "local-llama").WithContextWindow(32_768)
	if pinned.Provider().ContextWindow != 32_768 {
		t.Fatalf("override window = %d, want 32768", pinned.Provider().ContextWindow)
	}
}

// SC-27: with_compat.ReasoningModel beats the id heuristic. An unrecognized id
// is NOT reasoning by the heuristic, so by default it gets chat shaping
// (max_tokens, temperature kept). Declaring it reasoning flips the shaping even
// though the id is unknown.
func TestCompatReasoningModelBeatsIDHeuristic(t *testing.T) {
	r := req(userMsg("hi"))
	mt := uint32(512)
	temp := float32(0.7)
	r.Params.MaxTokens = &mt
	r.Params.Temperature = &temp

	chat := buildRequest("local-reasoner", r, false, OpenAICompat{})
	if chat.MaxTokens == nil || *chat.MaxTokens != 512 {
		t.Fatalf("chat max_tokens = %v, want 512", chat.MaxTokens)
	}
	if chat.MaxCompletionTokens != nil {
		t.Fatalf("chat max_completion_tokens = %v, want nil", chat.MaxCompletionTokens)
	}
	if chat.Temperature == nil || *chat.Temperature != 0.7 {
		t.Fatalf("chat temperature = %v, want 0.7", chat.Temperature)
	}

	reasoning := buildRequest("local-reasoner", r, false, OpenAICompat{ReasoningModel: true})
	if reasoning.MaxTokens != nil {
		t.Fatalf("reasoning max_tokens = %v, want nil", reasoning.MaxTokens)
	}
	if reasoning.MaxCompletionTokens == nil || *reasoning.MaxCompletionTokens != 512 {
		t.Fatalf("reasoning max_completion_tokens = %v, want 512", reasoning.MaxCompletionTokens)
	}
	if reasoning.Temperature != nil {
		t.Fatalf("reasoning temperature = %v, want nil", reasoning.Temperature)
	}
}

// SC-27: ReasoningModel drops temperature AND top_p AND stop (reasoning models
// reject all three). Chat models keep them.
func TestCompatReasoningModelDropsSamplingParams(t *testing.T) {
	r := req(userMsg("hi"))
	tp := float32(0.9)
	r.Params.TopP = &tp
	r.Params.StopSequences = []string{"STOP"}

	reasoning := buildRequest("local-reasoner", r, false, OpenAICompat{ReasoningModel: true})
	if reasoning.TopP != nil {
		t.Fatalf("reasoning top_p = %v, want nil", reasoning.TopP)
	}
	if len(reasoning.Stop) != 0 {
		t.Fatalf("reasoning stop = %v, want empty", reasoning.Stop)
	}

	chat := buildRequest("gpt-4o", r, false, OpenAICompat{})
	if chat.TopP == nil || *chat.TopP != 0.9 {
		t.Fatalf("chat top_p = %v, want 0.9", chat.TopP)
	}
	if len(chat.Stop) != 1 || chat.Stop[0] != "STOP" {
		t.Fatalf("chat stop = %v, want [STOP]", chat.Stop)
	}
}

// SC-27: DeveloperRole routes the system message to the "developer" role.
func TestCompatDeveloperRoleRoutesSystemMessage(t *testing.T) {
	r := req(sysMsg("be terse"), userMsg("hi"))

	plain := buildRequest("local-reasoner", r, false, OpenAICompat{})
	if plain.Messages[0].Role != "system" {
		t.Fatalf("default system role = %q, want system", plain.Messages[0].Role)
	}

	dev := buildRequest("local-reasoner", r, false, OpenAICompat{DeveloperRole: true})
	if dev.Messages[0].Role != "developer" {
		t.Fatalf("developer_role system role = %q, want developer", dev.Messages[0].Role)
	}
	if dev.Messages[1].Role != "user" {
		t.Fatalf("user message role = %q, want user (untouched)", dev.Messages[1].Role)
	}
}

// SC-27 acceptance: an unrecognized model with reasoning + effort support
// carries reasoning_effort AND the developer role on the wire. Opt-in is
// required (no SupportsReasoningEffort → absent), and it's gated on the model
// being reasoning at all.
func TestCompatEmitsReasoningEffortForReasoningModel(t *testing.T) {
	r := req(sysMsg("x"), userMsg("hi"))
	high := sporecore.ReasoningEffortHigh
	r.Params.ReasoningEffort = &high

	full := OpenAICompat{ReasoningModel: true, DeveloperRole: true, SupportsReasoningEffort: true}
	body := buildRequest("local-reasoner", r, false, full)
	if body.ReasoningEffort == nil || *body.ReasoningEffort != "high" {
		t.Fatalf("reasoning_effort = %v, want high", body.ReasoningEffort)
	}
	if body.Messages[0].Role != "developer" {
		t.Fatalf("system role = %q, want developer", body.Messages[0].Role)
	}

	// Opt-in is required: without SupportsReasoningEffort, the field is absent.
	noEffort := buildRequest("local-reasoner", r, false, OpenAICompat{ReasoningModel: true})
	if noEffort.ReasoningEffort != nil {
		t.Fatalf("reasoning_effort = %v, want nil (no opt-in)", noEffort.ReasoningEffort)
	}

	// Gated on the model being reasoning at all (effort alone on a chat model
	// does nothing).
	effortOnly := buildRequest("gpt-4o", r, false, OpenAICompat{SupportsReasoningEffort: true})
	if effortOnly.ReasoningEffort != nil {
		t.Fatalf("reasoning_effort = %v, want nil (chat model)", effortOnly.ReasoningEffort)
	}
}

// SC-27: reasoning_effort is serialized on the wire when set, and absent by
// default (default-compat requests stay byte-identical).
func TestCompatReasoningEffortSerializedOnWire(t *testing.T) {
	r := req(userMsg("hi"))
	medium := sporecore.ReasoningEffortMedium
	r.Params.ReasoningEffort = &medium

	compat := OpenAICompat{ReasoningModel: true, SupportsReasoningEffort: true}
	body := buildRequest("local-reasoner", r, false, compat)
	out, _ := json.Marshal(body)
	if !strings.Contains(string(out), `"reasoning_effort":"medium"`) {
		t.Fatalf("reasoning_effort not on wire: %s", out)
	}

	// A bare (default-compat) request must NOT carry the field.
	bare := buildRequest("gpt-4o", r, false, OpenAICompat{})
	bareOut, _ := json.Marshal(bare)
	if strings.Contains(string(bareOut), "reasoning_effort") {
		t.Fatalf("reasoning_effort must be absent by default: %s", bareOut)
	}
}
