package storage_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/storage"
)

// Issue #102 cross-process continuity: a SessionStore-backed harness persists
// the post-run SessionState by session_id, and a brand-new harness instance over
// the SAME backend resumes by id losslessly. These tests live in the storage
// package (which depends on sporecore) so they can exercise the REAL
// InMemory/FileSystem providers — package sporecore can't import storage (cycle),
// so its own session-threading tests use a test-local store instead.

// recordingCM records the conversation the loop produces. It implements the
// exported sporecore.ContextManager AND sporecore.AssistantMessageAppender so the
// loop records the assistant + tool turns into the session (the Go analog of the
// Rust reference's real context manager).
type recordingCM struct{}

func (recordingCM) Assemble(_ context.Context, s *sporecore.SessionState, _ *sporecore.Task) sporecore.Context {
	msgs := make([]sporecore.Message, len(s.Messages))
	copy(msgs, s.Messages)
	return sporecore.Context{Messages: msgs}
}

func (recordingCM) AppendToolResult(_ context.Context, s *sporecore.SessionState, r *sporecore.HarnessToolResult) {
	s.Messages = append(s.Messages, sporecore.Message{
		Role:    sporecore.RoleTool,
		Content: sporecore.NewTextContent(r.Output.Content),
	})
}

func (recordingCM) AppendUserMessage(_ context.Context, s *sporecore.SessionState, text string) {
	s.Messages = append(s.Messages, sporecore.Message{Role: sporecore.RoleUser, Content: sporecore.NewTextContent(text)})
}

func (recordingCM) ShouldCompact(*sporecore.SessionState) bool { return false }

func (recordingCM) AppendAssistantMessage(_ context.Context, s *sporecore.SessionState, m sporecore.Message) {
	s.Messages = append(s.Messages, m)
}

func turnUsage() sporecore.TokenUsage {
	return sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1}
}

// finalAgent returns a MockAgent that replies once with the given text.
func finalAgent(text string) *sporecore.MockAgent {
	a := sporecore.NewMockAgent("t")
	a.Push(sporecore.NewFinalResponse(text, turnUsage()))
	return a
}

func cfgWithStore(agent sporecore.Agent, store sporecore.SessionStore) sporecore.HarnessConfig {
	return sporecore.HarnessConfig{
		Agent:               agent,
		ToolRegistry:        sporecore.NewScriptedToolRegistry(),
		Sandbox:             sporecore.AllowAllSandbox{},
		ContextManager:      recordingCM{},
		TerminationPolicy:   sporecore.AlwaysContinuePolicy{},
		SessionStore:        store,
		AutoPersistSessions: true,
	}
}

func sessionTexts(s sporecore.SessionState) []string {
	var out []string
	for _, m := range s.Messages {
		if m.Content.Type == sporecore.ContentTypeText {
			out = append(out, m.Content.Text)
		}
	}
	return out
}

func containsStr(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

// Auto-persist round-trip against the REAL InMemory provider's Session() store.
func TestSessionThreading_AutoPersistRoundTrip_InMemory(t *testing.T) {
	provider := storage.SingleStorageProvider(storage.NewInMemoryStorageProvider())
	sid := sporecore.SessionID("s1")
	h := sporecore.NewStandardHarness(cfgWithStore(finalAgent("done"), provider.Session()))
	task := sporecore.NewTask("do something", sid, sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, MaxIterations: 5})
	r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
	if r.Kind != sporecore.RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	stored, found, err := provider.Session().GetSession(context.Background(), sid)
	if err != nil || !found || stored == nil {
		t.Fatalf("session not persisted: found=%v err=%v", found, err)
	}
	if !containsStr(sessionTexts(stored.SessionState), "done") {
		t.Fatalf("persisted history missing final text; got %+v", stored.SessionState.Messages)
	}
	if len(stored.PendingToolCalls) != 0 || stored.HumanRequest != nil {
		t.Fatalf("synthesized completed-run PausedState must be empty-pending / no-request: %+v", stored)
	}
}

// Cross-process continuity via the REAL FileSystemStorageProvider under a
// tempdir: "process 1" persists by session_id, then a brand-new provider +
// harness over the SAME dir resumes by id and sees the prior history.
func TestSessionThreading_CrossProcessContinuity_FileSystem(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "spore-102")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	sid := sporecore.SessionID("fs-session")

	// "Process 1": its own provider + harness instance.
	{
		provider := storage.SingleStorageProvider(storage.NewFileSystemStorageProvider(dir))
		h := sporecore.NewStandardHarness(cfgWithStore(finalAgent("process-one"), provider.Session()))
		task := sporecore.NewTask("first process", sid, sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, MaxIterations: 5})
		if r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(task)); r.Kind != sporecore.RunSuccess {
			t.Fatalf("process 1 expected Success, got %+v", r)
		}
	}

	// "Process 2": brand-new provider over the SAME dir, brand-new harness.
	provider := storage.SingleStorageProvider(storage.NewFileSystemStorageProvider(dir))
	h := sporecore.NewStandardHarness(cfgWithStore(finalAgent("process-two"), provider.Session()))
	task := sporecore.NewTask("second process", sid, sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, MaxIterations: 5})
	r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
	if r.Kind != sporecore.RunSuccess {
		t.Fatalf("process 2 expected Success, got %+v", r)
	}
	texts := sessionTexts(r.SessionState)
	if !containsStr(texts, "process-one") {
		t.Fatalf("prior process history not loaded across the filesystem; texts=%v", texts)
	}
	if !containsStr(texts, "process-two") {
		t.Fatalf("current process turn missing; texts=%v", texts)
	}
}
