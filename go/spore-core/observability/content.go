// LLM-native content capture (issue #64).
//
// Opt-in capture of conversation/tool-call content following the OpenTelemetry
// GenAI semantic conventions. OFF by default. Plumbed into the existing span
// payloads (no interface-signature changes) and serialized into the durable
// JSONL only when present, so the content-OFF path stays byte-identical to the
// pre-#64 metrics-only output.
//
// This is the Go port of the Rust reference
// (rust/crates/spore-core/src/observability.rs). It mirrors Rust semantics, not
// structure. Cross-language ground truth lives in fixtures/observability/.
//
// Resolved maintainer decisions (issue #64):
//  1. Canonical convention is pure OTel gen_ai.* events (no OpenInference).
//  2. Routing is the single configurable SPORE_OTLP_ENDPOINT (no fan-out).
//  3. Truncation default is 8192 UTF-8 bytes, marker "...[truncated]", clipped
//     at a UTF-8 char boundary; override SPORE_TRACE_CONTENT_MAX_LEN; guard
//     SPORE_TRACE_CONTENT (default OFF).
//  4. Capture is output text + tool calls on the turn span, tool args + result
//     on the tool-call span only (no assembled input-message history).
package observability

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"
)

// TruncationMarker is the exact ASCII marker appended to any captured field
// that was clipped by TruncateField. Cross-language ground truth — never change
// the bytes.
const TruncationMarker = "...[truncated]"

// DefaultContentMaxLen is the default content-field cap, in UTF-8 bytes
// (maintainer decision 3).
const DefaultContentMaxLen = 8192

// ContentCaptureConfig is the opt-in guard + truncation limit for LLM-native
// content capture (issue #64).
//
// Content capture is OFF by default. When Enabled is false the harness populates
// none of the gen_ai.* content fields, so the durable JSONL stays byte-identical
// to the pre-#64 metrics-only output.
type ContentCaptureConfig struct {
	// Enabled controls whether to capture message / tool-call content at all.
	// Default false.
	Enabled bool
	// MaxFieldLen is the maximum UTF-8 byte length of any single captured field
	// before TruncateField clips it. Default DefaultContentMaxLen (8192).
	MaxFieldLen int
}

// DefaultContentCaptureConfig returns the OFF-by-default config with the 8192
// byte cap.
func DefaultContentCaptureConfig() ContentCaptureConfig {
	return ContentCaptureConfig{Enabled: false, MaxFieldLen: DefaultContentMaxLen}
}

// ContentCaptureConfigFromEnv reads the config from the environment:
//   - SPORE_TRACE_CONTENT — "1"/"true"/"yes"/"on" (case-insensitive) enables
//     capture; anything else (or unset) leaves it OFF.
//   - SPORE_TRACE_CONTENT_MAX_LEN — parsed as an int; falls back to the
//     8192-byte default when unset, unparseable, or non-positive.
func ContentCaptureConfigFromEnv() ContentCaptureConfig {
	cfg := DefaultContentCaptureConfig()
	switch strings.ToLower(strings.TrimSpace(os.Getenv("SPORE_TRACE_CONTENT"))) {
	case "1", "true", "yes", "on":
		cfg.Enabled = true
	}
	if v := strings.TrimSpace(os.Getenv("SPORE_TRACE_CONTENT_MAX_LEN")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.MaxFieldLen = n
		}
	}
	return cfg
}

// GenAiRole is the role of a captured message. The bare-string values map onto
// the conventional GenAI span-event names (gen_ai.<role>.message).
type GenAiRole string

const (
	// GenAiRoleSystem is the system message role.
	GenAiRoleSystem GenAiRole = "system"
	// GenAiRoleUser is the user message role.
	GenAiRoleUser GenAiRole = "user"
	// GenAiRoleAssistant is the assistant (model output) message role.
	GenAiRoleAssistant GenAiRole = "assistant"
	// GenAiRoleTool is the tool result message role.
	GenAiRoleTool GenAiRole = "tool"
)

// EventName returns the conventional OTel GenAI span-event name for this role,
// e.g. "gen_ai.assistant.message".
func (r GenAiRole) EventName() string {
	switch r {
	case GenAiRoleSystem:
		return "gen_ai.system.message"
	case GenAiRoleUser:
		return "gen_ai.user.message"
	case GenAiRoleAssistant:
		return "gen_ai.assistant.message"
	case GenAiRoleTool:
		return "gen_ai.tool.message"
	default:
		return "gen_ai." + string(r) + ".message"
	}
}

// GenAiMessage is one captured conversation message (issue #64).
type GenAiMessage struct {
	Role GenAiRole `json:"role"`
	// Content is the message text.
	Content string `json:"content"`
	// Truncated is true when Content was clipped by TruncateField.
	Truncated bool `json:"truncated"`
}

// ToolCallContent is a requested tool call captured on a TurnSpan (issue #64).
type ToolCallContent struct {
	Name string `json:"name"`
	// Arguments is the tool-call arguments. When clipped, the arguments are
	// stored as a JSON string value carrying the truncation marker (a JSON value
	// cannot be clipped in place), and ArgumentsTruncated is true.
	Arguments json.RawMessage `json:"arguments"`
	// ArgumentsTruncated is true when Arguments was clipped.
	ArgumentsTruncated bool `json:"arguments_truncated"`
}

// ToolResultContent is a tool result body captured on a ToolCallSpan (issue
// #64).
type ToolResultContent struct {
	Content string `json:"content"`
	// Truncated is true when Content was clipped by TruncateField.
	Truncated bool `json:"truncated"`
}

// TruncateField clips s to at most max UTF-8 bytes, appending TruncationMarker
// when (and only when) a clip occurred. Returns (clipped, wasTruncated).
//
// Pure and deterministic — this is the cross-language ground truth:
//   - Measurement is in UTF-8 bytes, not runes.
//   - When len(s) <= max, returns s unchanged with false.
//   - Otherwise clips to the largest valid UTF-8 char boundary <= max (never
//     splitting a multibyte char — backs off to the previous boundary) and
//     appends the marker. The marker is appended AFTER the byte budget, so the
//     returned string may exceed max bytes by the marker's length; the budget
//     bounds the captured payload, not the marker.
func TruncateField(s string, max int) (string, bool) {
	if max < 0 {
		max = 0
	}
	if len(s) <= max {
		return s, false
	}
	// Back off to the largest char boundary <= max. utf8.RuneStart reports
	// whether the byte at index begins a rune; s[max] is the first byte NOT
	// kept, so we shrink the boundary until s[boundary] starts a rune (i.e. the
	// kept prefix ends on a boundary).
	boundary := max
	for boundary > 0 && !utf8.RuneStart(s[boundary]) {
		boundary--
	}
	return s[:boundary] + TruncationMarker, true
}
