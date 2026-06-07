package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"sync"
	"sync/atomic"
	"testing"
)

// Issue #102: opt-in conversation-history threading + session-store auto-persist.
//
// These tests live in package sporecore and use a test-local SessionStore (the
// consumer-side sporecore.SessionStore interface) so they do not import the
// storage package (storage imports sporecore — that would be a cycle). The
// storage-backed round-trip / auto-load / cross-process tests live alongside the
// real providers in storage/session_threading_test.go.

// These tests reuse the *recordingContextManager defined in harness_test.go: it
// implements ContextManager AND AssistantMessageAppender, so the loop records
// the assistant tool-call + final-text turns into the session (NoopContextManager
// does not implement AssistantMessageAppender, so it would never record the
// assistant tool-call message the lossless assertions check for).

// countingSessionStore wraps an in-memory map and COUNTS every Get/Put so a test
// can assert "zero session-store I/O when auto-persist is disabled".
type countingSessionStore struct {
	mu       sync.Mutex
	sessions map[SessionID]PausedState
	gets     atomic.Int64
	puts     atomic.Int64
}

func newCountingSessionStore() *countingSessionStore {
	return &countingSessionStore{sessions: make(map[SessionID]PausedState)}
}

func (c *countingSessionStore) GetSession(_ context.Context, id SessionID) (*PausedState, bool, error) {
	c.gets.Add(1)
	c.mu.Lock()
	defer c.mu.Unlock()
	s, ok := c.sessions[id]
	if !ok {
		return nil, false, nil
	}
	cp := s
	return &cp, true, nil
}

func (c *countingSessionStore) PutSession(_ context.Context, id SessionID, state *PausedState) error {
	c.puts.Add(1)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sessions[id] = *state
	return nil
}

// seed inserts a prior state WITHOUT bumping the put counter (test setup).
func (c *countingSessionStore) seed(id SessionID, state PausedState) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sessions[id] = state
}

var _ SessionStore = (*countingSessionStore)(nil)

// toolThenFinalConfig builds a config whose agent requests one tool call then
// replies with final text, with a recording context manager.
func toolThenFinalConfig() (HarnessConfig, *ScriptedToolRegistry) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	a.Push(NewFinalResponse("after-tool", turnUsage()))
	cfg := standardCfg(a)
	cfg.ContextManager = &recordingContextManager{}
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "tool-ok"})
	cfg.ToolRegistry = reg
	return cfg, reg
}

func hasToolCallMsg(s SessionState) bool {
	for _, m := range s.Messages {
		if m.Content.Type == ContentTypeToolCall {
			return true
		}
	}
	return false
}

func hasRole(s SessionState, role Role) bool {
	for _, m := range s.Messages {
		if m.Role == role {
			return true
		}
	}
	return false
}

func hasText(s SessionState, want string) bool {
	for _, m := range s.Messages {
		if m.Content.Type == ContentTypeText && m.Content.Text == want {
			return true
		}
	}
	return false
}

// (a) Off-by-default: ZERO session-store I/O AND the message flow / outcome is
// identical to today (Success carries the right messages).
func TestSessionThreading_OffByDefaultNoSessionIO(t *testing.T) {
	cfg, _ := toolThenFinalConfig()
	store := newCountingSessionStore()
	cfg.SessionStore = store
	// AutoPersistSessions defaults to false.
	if cfg.AutoPersistSessions {
		t.Fatal("expected AutoPersistSessions to default to false")
	}
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess || r.Output != "after-tool" {
		t.Fatalf("got %+v", r)
	}
	// The new field is populated even when persistence is off.
	if !hasToolCallMsg(r.SessionState) {
		t.Fatal("expected SessionState to carry the assistant tool-call message")
	}
	if g := store.gets.Load(); g != 0 {
		t.Fatalf("expected 0 GetSession calls, got %d", g)
	}
	if p := store.puts.Load(); p != 0 {
		t.Fatalf("expected 0 PutSession calls, got %d", p)
	}
}

// (b) Success.SessionState is LOSSLESS for a tool-using run: the user
// instruction, the assistant tool-call message, the tool-result message, and the
// final assistant text are all present (none recoverable from Output alone).
func TestSessionThreading_SuccessSessionStateLosslessForToolRun(t *testing.T) {
	cfg, _ := toolThenFinalConfig()
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	s := r.SessionState
	if !hasRole(s, RoleUser) {
		t.Fatal("missing user instruction turn")
	}
	if !hasToolCallMsg(s) {
		t.Fatal("missing assistant tool-call turn")
	}
	if !hasRole(s, RoleTool) {
		t.Fatal("missing tool-result turn")
	}
	if !hasText(s, "after-tool") {
		t.Fatal("missing final assistant text turn")
	}
}

// (f) Explicit HarnessRunOptions.SessionState WINS over auto-load: no
// GetSession is called, and the explicit state seeds the run.
func TestSessionThreading_ExplicitSessionStateWinsOverAutoLoad(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := standardCfg(a)
	cfg.ContextManager = &recordingContextManager{}
	store := newCountingSessionStore()
	// Pre-seed a DIFFERENT history under the session id (without bumping puts).
	store.seed(SessionID("s1"), PausedState{
		SessionID:    SessionID("s1"),
		SessionState: SessionState{Messages: []Message{{Role: RoleUser, Content: NewTextContent("STORED-history")}}},
	})
	cfg.SessionStore = store
	cfg.AutoPersistSessions = true
	h := NewStandardHarness(cfg)

	explicit := SessionState{Messages: []Message{{Role: RoleUser, Content: NewTextContent("EXPLICIT-history")}}}
	opts := NewHarnessRunOptions(reactTask(5))
	opts.SessionState = &explicit
	r := h.Run(context.Background(), opts)
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if !hasText(r.SessionState, "EXPLICIT-history") {
		t.Fatal("explicit history should seed the run")
	}
	if hasText(r.SessionState, "STORED-history") {
		t.Fatal("stored history must NOT be loaded when explicit state is provided")
	}
	if g := store.gets.Load(); g != 0 {
		t.Fatalf("explicit state must skip the auto-load GetSession; got %d gets", g)
	}
}

// (g) Failure ALSO carries SessionState.
func TestSessionThreading_FailureCarriesSessionState(t *testing.T) {
	// A tool annotated always_halt fails the run after the assistant tool-call
	// turn was recorded into the session.
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c", Name: "danger", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	cfg.ContextManager = &recordingContextManager{}
	reg := NewScriptedToolRegistry()
	reg.MarkAlwaysHalt("danger")
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunFailure {
		t.Fatalf("expected Failure, got %+v", r)
	}
	// The always-halt short-circuit fires before the assistant tool-call turn is
	// recorded (matching the Rust reference), so the seeded user instruction is
	// the load-bearing evidence that Failure carries the post-run session state.
	if !hasRole(r.SessionState, RoleUser) {
		t.Fatal("Failure.SessionState must carry the user instruction turn")
	}
}

// Auto-persist round-trip against the test-local in-memory store: GetSession
// returns the final history, synthesized into a completed-run PausedState with
// empty pending fields and no human request (D4).
func TestSessionThreading_AutoPersistRoundTrip(t *testing.T) {
	cfg, _ := toolThenFinalConfig()
	store := newCountingSessionStore()
	cfg.SessionStore = store
	cfg.AutoPersistSessions = true
	h := NewStandardHarness(cfg)
	sid := SessionID("s1")
	task := NewTask("do something", sid, ReActStrategy(5))
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	stored, found, err := store.GetSession(context.Background(), sid)
	if err != nil || !found || stored == nil {
		t.Fatalf("session not persisted: found=%v err=%v", found, err)
	}
	if !hasToolCallMsg(stored.SessionState) {
		t.Fatal("persisted SessionState must carry the assistant tool-call turn")
	}
	if len(stored.PendingToolCalls) != 0 {
		t.Fatalf("synthesized PausedState must have empty pending tool calls, got %d", len(stored.PendingToolCalls))
	}
	if stored.HumanRequest != nil {
		t.Fatal("synthesized PausedState must have no human request")
	}
	if stored.ChildState != nil {
		t.Fatal("synthesized PausedState must have no child state")
	}
}

// Auto-load across two runs sharing a session_id (test-local in-memory store):
// the second run sees the first run's history, threaded forward.
func TestSessionThreading_AutoLoadBySessionIDAcrossRuns(t *testing.T) {
	store := newCountingSessionStore()
	sid := SessionID("shared")

	// Run 1: one final response, auto-persisted.
	{
		a := NewMockAgent("t")
		a.Push(NewFinalResponse("first", turnUsage()))
		cfg := standardCfg(a)
		cfg.ContextManager = &recordingContextManager{}
		cfg.SessionStore = store
		cfg.AutoPersistSessions = true
		h := NewStandardHarness(cfg)
		task := NewTask("turn one", sid, ReActStrategy(5))
		_ = h.Run(context.Background(), NewHarnessRunOptions(task))
	}

	// Run 2: same session_id, no explicit state. The loaded history must carry
	// the prior turns forward.
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("second", turnUsage()))
	cfg := standardCfg(a)
	cfg.ContextManager = &recordingContextManager{}
	cfg.SessionStore = store
	cfg.AutoPersistSessions = true
	h := NewStandardHarness(cfg)
	task := NewTask("turn two", sid, ReActStrategy(5))
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	for _, want := range []string{"first", "turn one", "turn two", "second"} {
		if !hasText(r.SessionState, want) {
			t.Fatalf("expected carried-forward history to contain %q; messages=%+v", want, r.SessionState.Messages)
		}
	}
}

// Ralph / HillClimbing discard incoming session state by design (D7): auto-load
// is skipped for them — GetSession is never called even with auto-persist on.
func TestSessionThreading_RalphDoesNotAutoLoad(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := standardCfg(a)
	cfg.ContextManager = &recordingContextManager{}
	store := newCountingSessionStore()
	store.seed(SessionID("r1"), PausedState{SessionID: SessionID("r1")})
	cfg.SessionStore = store
	cfg.AutoPersistSessions = true
	h := NewStandardHarness(cfg)
	task := NewTask("ralph task", SessionID("r1"), RalphStrategy(RalphConfig{Inner: PtrStrategy(ReActStrategy(^uint32(0))), Agent: AgentRef("ralph-agent")}))
	_ = h.Run(context.Background(), NewHarnessRunOptions(task))
	if g := store.gets.Load(); g != 0 {
		t.Fatalf("Ralph must NOT auto-load (D7); got %d GetSession calls", g)
	}
}

// RunResult JSON round-trips the new session_state field, and an old blob that
// predates the field still decodes (tolerant of absence — the Go analog of
// Rust's #[serde(default)]).
func TestSessionThreading_RunResultSessionStateSerdeTolerant(t *testing.T) {
	orig := RunResult{
		Kind:      RunSuccess,
		Output:    "out",
		SessionID: SessionID("s1"),
		Turns:     2,
		SessionState: SessionState{
			Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}},
		},
	}
	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back RunResult
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !hasText(back.SessionState, "hi") {
		t.Fatalf("round-trip lost session_state: %s", b)
	}

	// Old blob WITHOUT session_state still parses to the zero value.
	old := `{"kind":"success","output":"out","session_id":"s1","usage":{},"turns":2}`
	var legacy RunResult
	if err := json.Unmarshal([]byte(old), &legacy); err != nil {
		t.Fatalf("legacy blob must still parse: %v", err)
	}
	if len(legacy.SessionState.Messages) != 0 {
		t.Fatalf("legacy blob must decode to empty session_state, got %+v", legacy.SessionState)
	}
}

// Fixture-replay lossless path against the SHARED
// fixtures/model_responses/harness/react_loop.jsonl: driving the same fixture
// the cross-language harness replay uses, but with a recording context manager,
// the Success.SessionState carries the full tool-using conversation (user
// instruction, assistant tool-call turn, tool-result turn, final assistant
// text) — none recoverable from Output alone (issue #102 part 1). Mirrors the
// outcome asserted in TestHarnessReActLoopDispatchesToolThenCompletes.
func TestSessionThreading_FixtureReplayLosslessSessionState(t *testing.T) {
	raw, err := os.ReadFile(harnessFixturePath(t))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	replay, err := ParseReplayJSONL(string(raw), ProviderInfo{
		Name: "anthropic", ModelID: "fixture", ContextWindow: 200_000,
	})
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	agent := NewModelAgent(AgentID("fixture-agent"), replay)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "127.0.0.1 localhost"})

	cfg := HarnessConfig{
		Agent:             agent,
		ToolRegistry:      reg,
		Sandbox:           AllowAllSandbox{},
		ContextManager:    &recordingContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
	}
	h := NewStandardHarness(cfg)
	task := NewTask(
		"read /etc/hosts then summarize",
		SessionID("fixture-session"),
		ReActStrategy(5),
	)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if r.Output != "127.0.0.1 localhost" || r.Turns != 2 {
		t.Fatalf("unexpected replay outcome: output=%q turns=%d", r.Output, r.Turns)
	}
	s := r.SessionState
	if !hasRole(s, RoleUser) {
		t.Fatal("replay session state missing user instruction turn")
	}
	if !hasToolCallMsg(s) {
		t.Fatal("replay session state missing assistant tool-call turn")
	}
	if !hasRole(s, RoleTool) {
		t.Fatal("replay session state missing tool-result turn")
	}
	if !hasText(s, "127.0.0.1 localhost") {
		t.Fatal("replay session state missing final assistant text turn")
	}
}
