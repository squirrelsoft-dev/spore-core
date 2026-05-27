package observability

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ── TruncateField: within budget, byte-boundary clip+mark, multibyte back-off ─

func TestTruncateFieldWithinBudget(t *testing.T) {
	s := "hello world"
	got, truncated := TruncateField(s, 8192)
	if truncated {
		t.Fatalf("within-budget input must not truncate")
	}
	if got != s {
		t.Fatalf("within-budget input must be returned unchanged: got %q", got)
	}
}

func TestTruncateFieldExactBudgetNotTruncated(t *testing.T) {
	s := "abcde"
	got, truncated := TruncateField(s, len(s))
	if truncated || got != s {
		t.Fatalf("len(s) == max must not truncate: got %q truncated=%v", got, truncated)
	}
}

func TestTruncateFieldClipsAndMarksAsciiBoundary(t *testing.T) {
	s := "the quick brown fox jumps over the lazy dog"
	max := 9 // "the quick"
	got, truncated := TruncateField(s, max)
	if !truncated {
		t.Fatalf("over-budget input must truncate")
	}
	want := "the quick" + TruncationMarker
	if got != want {
		t.Fatalf("clip mismatch: got %q want %q", got, want)
	}
	if !strings.HasSuffix(got, TruncationMarker) {
		t.Fatalf("clipped output must carry the marker")
	}
}

func TestTruncateFieldMultibyteBackoff(t *testing.T) {
	// Each "é" is 2 UTF-8 bytes. "aé" = bytes [a(1), é(2)] => 3 bytes total.
	// Clipping at max=2 lands mid-é (byte 2 is the é continuation byte, not a
	// rune start), so we must back off to byte 1 ("a"), never splitting the char.
	s := "aéb" // bytes: 'a'(1) + 'é'(2) + 'b'(1) = 4
	got, truncated := TruncateField(s, 2)
	if !truncated {
		t.Fatalf("over-budget multibyte input must truncate")
	}
	want := "a" + TruncationMarker
	if got != want {
		t.Fatalf("multibyte back-off mismatch: got %q want %q", got, want)
	}
	// The kept prefix must be valid UTF-8 (no split rune).
	if !json.Valid([]byte(`"` + strings.TrimSuffix(got, TruncationMarker) + `"`)) {
		t.Fatalf("kept prefix is not valid UTF-8: %q", got)
	}
}

func TestTruncateFieldBoundaryLandsExactlyOnRuneStart(t *testing.T) {
	// "ab" + "é"(2 bytes). max=2 lands exactly on the start of é (a rune
	// boundary), so the prefix is "ab" with no back-off.
	s := "abé"
	got, truncated := TruncateField(s, 2)
	if !truncated {
		t.Fatalf("must truncate")
	}
	if got != "ab"+TruncationMarker {
		t.Fatalf("boundary clip mismatch: got %q", got)
	}
}

// ── Config: default OFF + env parsing ────────────────────────────────────────

func TestDefaultContentCaptureConfigOff(t *testing.T) {
	c := DefaultContentCaptureConfig()
	if c.Enabled {
		t.Fatalf("default config must be OFF")
	}
	if c.MaxFieldLen != DefaultContentMaxLen {
		t.Fatalf("default max len must be %d, got %d", DefaultContentMaxLen, c.MaxFieldLen)
	}
}

func TestContentCaptureConfigFromEnvDefaultOff(t *testing.T) {
	t.Setenv("SPORE_TRACE_CONTENT", "")
	t.Setenv("SPORE_TRACE_CONTENT_MAX_LEN", "")
	c := ContentCaptureConfigFromEnv()
	if c.Enabled {
		t.Fatalf("unset SPORE_TRACE_CONTENT must leave capture OFF")
	}
	if c.MaxFieldLen != DefaultContentMaxLen {
		t.Fatalf("unset max len must fall back to %d", DefaultContentMaxLen)
	}
}

func TestContentCaptureConfigFromEnvEnableVariants(t *testing.T) {
	for _, v := range []string{"1", "true", "TRUE", "Yes", "on", " on "} {
		t.Run(v, func(t *testing.T) {
			t.Setenv("SPORE_TRACE_CONTENT", v)
			if !ContentCaptureConfigFromEnv().Enabled {
				t.Fatalf("%q must enable capture", v)
			}
		})
	}
	for _, v := range []string{"0", "false", "no", "off", "garbage"} {
		t.Run("off_"+v, func(t *testing.T) {
			t.Setenv("SPORE_TRACE_CONTENT", v)
			if ContentCaptureConfigFromEnv().Enabled {
				t.Fatalf("%q must leave capture OFF", v)
			}
		})
	}
}

func TestContentCaptureConfigFromEnvMaxLenOverride(t *testing.T) {
	t.Setenv("SPORE_TRACE_CONTENT", "1")
	t.Setenv("SPORE_TRACE_CONTENT_MAX_LEN", "16")
	c := ContentCaptureConfigFromEnv()
	if c.MaxFieldLen != 16 {
		t.Fatalf("override max len must be 16, got %d", c.MaxFieldLen)
	}
	// Unparseable / non-positive falls back to default.
	t.Setenv("SPORE_TRACE_CONTENT_MAX_LEN", "nope")
	if ContentCaptureConfigFromEnv().MaxFieldLen != DefaultContentMaxLen {
		t.Fatalf("unparseable max len must fall back to default")
	}
	t.Setenv("SPORE_TRACE_CONTENT_MAX_LEN", "0")
	if ContentCaptureConfigFromEnv().MaxFieldLen != DefaultContentMaxLen {
		t.Fatalf("non-positive max len must fall back to default")
	}
}

// ── Role → event-name mapping ────────────────────────────────────────────────

func TestGenAiRoleEventName(t *testing.T) {
	cases := map[GenAiRole]string{
		GenAiRoleSystem:    "gen_ai.system.message",
		GenAiRoleUser:      "gen_ai.user.message",
		GenAiRoleAssistant: "gen_ai.assistant.message",
		GenAiRoleTool:      "gen_ai.tool.message",
	}
	for role, want := range cases {
		if got := role.EventName(); got != want {
			t.Errorf("role %q event name: got %q want %q", role, got, want)
		}
	}
}

// ── Content on/off serialization + gen_ai.* attributes on/off ────────────────

func contentTurnSpan() TurnSpan {
	cr, cw := uint32(0), uint32(0)
	return TurnSpan{
		Base: SpanBase{
			SpanID:    "sp1",
			SessionID: "s1",
			TaskID:    "t1",
			Kind:      SpanKindTurn,
			Status:    NewStatusOk(),
		},
		TurnNumber:       1,
		InputTokens:      10,
		OutputTokens:     5,
		CacheReadTokens:  &cr,
		CacheWriteTokens: &cw,
		StopReason:       "end_turn",
	}
}

func attrsOf(t *testing.T, line TraceLine) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(line.Attributes, &m); err != nil {
		t.Fatalf("decode attributes: %v", err)
	}
	return m
}

func TestTurnContentOffOmitsGenAiKeys(t *testing.T) {
	span := contentTurnSpan() // no OutputText / ToolCalls
	line := TraceLineFromTurn(span, "tid")
	attrs := attrsOf(t, line)
	for k := range attrs {
		if strings.HasPrefix(k, "gen_ai.") {
			t.Fatalf("content-OFF turn must carry no gen_ai.* key, found %q", k)
		}
	}
	// And the span struct itself serializes without the optional keys.
	raw, _ := json.Marshal(span)
	var sm map[string]any
	_ = json.Unmarshal(raw, &sm)
	if _, ok := sm["output_text"]; ok {
		t.Fatalf("nil OutputText must be omitted from span JSON")
	}
	if _, ok := sm["tool_calls"]; ok {
		t.Fatalf("nil ToolCalls must be omitted from span JSON")
	}
}

func TestTurnContentOnEmitsGenAiAttributes(t *testing.T) {
	span := contentTurnSpan()
	span.OutputText = &GenAiMessage{Role: GenAiRoleAssistant, Content: "hi there", Truncated: false}
	span.ToolCalls = []ToolCallContent{
		{Name: "shell", Arguments: json.RawMessage(`{"command":"ls"}`), ArgumentsTruncated: false},
	}
	attrs := attrsOf(t, TraceLineFromTurn(span, "tid"))
	if attrs["gen_ai.response.role"] != "assistant" {
		t.Errorf("gen_ai.response.role: got %v", attrs["gen_ai.response.role"])
	}
	if attrs["gen_ai.response.content"] != "hi there" {
		t.Errorf("gen_ai.response.content: got %v", attrs["gen_ai.response.content"])
	}
	if attrs["gen_ai.response.content_truncated"] != false {
		t.Errorf("gen_ai.response.content_truncated: got %v", attrs["gen_ai.response.content_truncated"])
	}
	calls, ok := attrs["gen_ai.response.tool_calls"].([]any)
	if !ok || len(calls) != 1 {
		t.Fatalf("gen_ai.response.tool_calls: got %v", attrs["gen_ai.response.tool_calls"])
	}
	// Metrics keys must still be present (content rides ALONGSIDE metrics).
	if attrs["input_tokens"] == nil || attrs["turn_number"] == nil {
		t.Fatalf("metrics keys must remain alongside gen_ai.* keys")
	}
}

func TestToolCallContentOnEmitsGenAiAttributes(t *testing.T) {
	span := ToolCallSpan{
		Base:        SpanBase{SpanID: "tc1", SessionID: "s1", TaskID: "t1", Kind: SpanKindToolCall, Status: NewStatusOk()},
		ToolName:    "shell",
		CallID:      "c1",
		SandboxMode: "workspace_scoped",
		Arguments:   &ToolCallContent{Name: "shell", Arguments: json.RawMessage(`{"command":"ls"}`)},
		Result:      &ToolResultContent{Content: "total 0", Truncated: false},
	}
	attrs := attrsOf(t, TraceLineFromToolCall(span, "tid"))
	if attrs["gen_ai.tool.name"] != "shell" {
		t.Errorf("gen_ai.tool.name: got %v", attrs["gen_ai.tool.name"])
	}
	if attrs["gen_ai.tool.message.content"] != "total 0" {
		t.Errorf("gen_ai.tool.message.content: got %v", attrs["gen_ai.tool.message.content"])
	}
	args, ok := attrs["gen_ai.tool.call.arguments"].(map[string]any)
	if !ok || args["command"] != "ls" {
		t.Fatalf("gen_ai.tool.call.arguments must be the raw JSON value: got %v", attrs["gen_ai.tool.call.arguments"])
	}
}

func TestToolCallContentOffOmitsGenAiKeys(t *testing.T) {
	span := ToolCallSpan{
		Base:        SpanBase{SpanID: "tc1", SessionID: "s1", TaskID: "t1", Kind: SpanKindToolCall, Status: NewStatusOk()},
		ToolName:    "shell",
		CallID:      "c1",
		SandboxMode: "workspace_scoped",
	}
	attrs := attrsOf(t, TraceLineFromToolCall(span, "tid"))
	for k := range attrs {
		if strings.HasPrefix(k, "gen_ai.") {
			t.Fatalf("content-OFF tool_call must carry no gen_ai.* key, found %q", k)
		}
	}
}

// ── genai events per message (OTLP) ──────────────────────────────────────────

func TestGenAiEventsPerMessage(t *testing.T) {
	// A turn line with output text + one tool call → 2 assistant events.
	turn := contentTurnSpan()
	turn.OutputText = &GenAiMessage{Role: GenAiRoleAssistant, Content: "thinking"}
	turn.ToolCalls = []ToolCallContent{{Name: "shell", Arguments: json.RawMessage(`{"command":"ls"}`)}}
	events := genAiEvents(TraceLineFromTurn(turn, "tid"))
	if len(events) != 2 {
		t.Fatalf("turn with output + 1 tool call must yield 2 events, got %d", len(events))
	}
	for _, e := range events {
		if e.name != GenAiRoleAssistant.EventName() {
			t.Errorf("turn events must use the assistant event name, got %q", e.name)
		}
	}

	// A tool_call line with a result → 1 tool event.
	tc := ToolCallSpan{
		Base:     SpanBase{SpanID: "tc1", SessionID: "s1", TaskID: "t1", Kind: SpanKindToolCall, Status: NewStatusOk()},
		ToolName: "shell", CallID: "c1",
		Arguments: &ToolCallContent{Name: "shell", Arguments: json.RawMessage(`{}`)},
		Result:    &ToolResultContent{Content: "ok"},
	}
	tcEvents := genAiEvents(TraceLineFromToolCall(tc, "tid"))
	if len(tcEvents) != 1 {
		t.Fatalf("tool_call with result must yield 1 event, got %d", len(tcEvents))
	}
	if tcEvents[0].name != GenAiRoleTool.EventName() {
		t.Errorf("tool result event name: got %q", tcEvents[0].name)
	}
}

func TestGenAiEventsEmptyWhenContentOff(t *testing.T) {
	if events := genAiEvents(TraceLineFromTurn(contentTurnSpan(), "tid")); len(events) != 0 {
		t.Fatalf("content-OFF turn must produce no gen_ai events, got %d", len(events))
	}
}

// ── Adapter gating + truncation (issue #64) ──────────────────────────────────

func TestAdapterContentOffEmitsNoContent(t *testing.T) {
	provider := NewInMemoryObservabilityProvider()
	obs := NewHarnessObserverWithContent(provider, DefaultPricing(), DefaultContentCaptureConfig())
	obs.EmitTurn("sp1", "s1", "t1", 1, "ts", 1, sporecore.TokenUsage{}, 0, "end_turn", 0, "",
		"the model said this", []sporecore.ToolCall{{ID: "c1", Name: "shell", Input: json.RawMessage(`{"command":"ls"}`)}})
	trace, _ := provider.GetTrace(nil, "s1")
	if len(trace) != 1 {
		t.Fatalf("expected 1 span, got %d", len(trace))
	}
	turn := trace[0].(TurnSpan)
	if turn.OutputText != nil || turn.ToolCalls != nil {
		t.Fatalf("content-OFF adapter must not populate content fields")
	}
}

func TestAdapterContentOnPopulatesAndTruncates(t *testing.T) {
	provider := NewInMemoryObservabilityProvider()
	obs := NewHarnessObserverWithContent(provider, DefaultPricing(),
		ContentCaptureConfig{Enabled: true, MaxFieldLen: 5})
	obs.EmitTurn("sp1", "s1", "t1", 1, "ts", 1, sporecore.TokenUsage{}, 0, "end_turn", 0, "",
		"abcdefghij", nil)
	turn := mustGetTurn(t, provider, "s1")
	if turn.OutputText == nil {
		t.Fatalf("content-ON adapter must populate OutputText")
	}
	if turn.OutputText.Role != GenAiRoleAssistant {
		t.Errorf("output role must be assistant, got %q", turn.OutputText.Role)
	}
	if !turn.OutputText.Truncated {
		t.Errorf("over-budget output must be marked truncated")
	}
	if turn.OutputText.Content != "abcde"+TruncationMarker {
		t.Errorf("output must be clipped to budget: got %q", turn.OutputText.Content)
	}
}

func TestAdapterToolArgumentsTruncationStoredAsString(t *testing.T) {
	provider := NewInMemoryObservabilityProvider()
	obs := NewHarnessObserverWithContent(provider, DefaultPricing(),
		ContentCaptureConfig{Enabled: true, MaxFieldLen: 4})
	args := json.RawMessage(`{"command":"a very long command line"}`)
	obs.EmitToolCall("tc1", "sp1", "s1", "t1", "shell", "c1", "ts", 1, 0, 0, false, false, args, "result body")
	tc := mustGetToolCall(t, provider, "s1")
	if tc.Arguments == nil || !tc.Arguments.ArgumentsTruncated {
		t.Fatalf("over-budget args must be marked truncated")
	}
	// Truncated args are stored as a JSON string value carrying the marker.
	var s string
	if err := json.Unmarshal(tc.Arguments.Arguments, &s); err != nil {
		t.Fatalf("truncated arguments must be a JSON string value: %v", err)
	}
	if !strings.HasSuffix(s, TruncationMarker) {
		t.Fatalf("truncated arguments string must carry marker: got %q", s)
	}
}

func TestAdapterToolArgumentsUntruncatedStaysRawValue(t *testing.T) {
	provider := NewInMemoryObservabilityProvider()
	obs := NewHarnessObserverWithContent(provider, DefaultPricing(),
		ContentCaptureConfig{Enabled: true, MaxFieldLen: 8192})
	args := json.RawMessage(`{"command":"ls"}`)
	obs.EmitToolCall("tc1", "sp1", "s1", "t1", "shell", "c1", "ts", 1, 0, 0, false, false, args, "ok")
	tc := mustGetToolCall(t, provider, "s1")
	if tc.Arguments.ArgumentsTruncated {
		t.Fatalf("within-budget args must not be truncated")
	}
	var obj map[string]any
	if err := json.Unmarshal(tc.Arguments.Arguments, &obj); err != nil {
		t.Fatalf("untruncated args must stay a JSON object value: %v", err)
	}
	if obj["command"] != "ls" {
		t.Fatalf("args value mismatch: %v", obj)
	}
	if tc.Result == nil || tc.Result.Content != "ok" {
		t.Fatalf("result body must be captured: %+v", tc.Result)
	}
}

func mustGetTurn(t *testing.T, p *InMemoryObservabilityProvider, sid SessionID) TurnSpan {
	t.Helper()
	tr, _ := p.GetTrace(nil, sid)
	for _, s := range tr {
		if turn, ok := s.(TurnSpan); ok {
			return turn
		}
	}
	t.Fatalf("no turn span for %q", sid)
	return TurnSpan{}
}

func mustGetToolCall(t *testing.T, p *InMemoryObservabilityProvider, sid SessionID) ToolCallSpan {
	t.Helper()
	tr, _ := p.GetTrace(nil, sid)
	for _, s := range tr {
		if tc, ok := s.(ToolCallSpan); ok {
			return tc
		}
	}
	t.Fatalf("no tool_call span for %q", sid)
	return ToolCallSpan{}
}

// Sanity: the optional content fields round-trip through TurnSpan JSON when set.
func TestTurnSpanContentRoundTrip(t *testing.T) {
	span := contentTurnSpan()
	span.OutputText = &GenAiMessage{Role: GenAiRoleAssistant, Content: "hi", Truncated: true}
	raw, err := json.Marshal(span)
	if err != nil {
		t.Fatal(err)
	}
	var back TurnSpan
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(span.OutputText, back.OutputText) {
		t.Fatalf("OutputText round-trip mismatch: %+v vs %+v", span.OutputText, back.OutputText)
	}
}
