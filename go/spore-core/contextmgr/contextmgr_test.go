package contextmgr

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ── Test doubles ───────────────────────────────────────────────────────────

type fakeModel struct {
	tokens uint32
}

func (f *fakeModel) Call(_ context.Context, _ sporecore.ModelRequest) (sporecore.ModelResponse, error) {
	return sporecore.ModelResponse{}, nil
}
func (f *fakeModel) CallStreaming(_ context.Context, _ sporecore.ModelRequest) (<-chan sporecore.StreamEventOrErr, error) {
	return nil, nil
}
func (f *fakeModel) CountTokens(_ context.Context, _ sporecore.ModelRequest) (uint32, error) {
	return f.tokens, nil
}
func (f *fakeModel) Provider() sporecore.ProviderInfo {
	return sporecore.ProviderInfo{Name: "fake", ModelID: "fake", ContextWindow: 200000}
}

type failingModel struct{}

func (failingModel) Call(_ context.Context, _ sporecore.ModelRequest) (sporecore.ModelResponse, error) {
	return sporecore.ModelResponse{}, nil
}
func (failingModel) CallStreaming(_ context.Context, _ sporecore.ModelRequest) (<-chan sporecore.StreamEventOrErr, error) {
	return nil, nil
}
func (failingModel) CountTokens(_ context.Context, _ sporecore.ModelRequest) (uint32, error) {
	return 0, errors.New("boom")
}
func (failingModel) Provider() sporecore.ProviderInfo {
	return sporecore.ProviderInfo{Name: "f", ModelID: "f", ContextWindow: 1}
}

type countingCache struct {
	calls atomic.Int64
}

func (c *countingCache) SupportsCaching() bool { return true }
func (c *countingCache) Annotate(_ *Context) CacheAnnotationResult {
	c.calls.Add(1)
	return CacheAnnotationResult{}
}
func (c *countingCache) ParseCacheStats(_ *sporecore.ModelResponse) (CacheStats, bool) {
	return CacheStats{}, false
}
func (c *countingCache) ProviderName() string { return "counting" }

// passthroughSandbox truncates via head+tail with a small marker; it does
// not offload to disk.
type passthroughSandbox struct{}

func (passthroughSandbox) Validate(context.Context, sporecore.ToolCall) *sporecore.SandboxViolation {
	return nil
}
func (passthroughSandbox) ExecuteCommand(_ context.Context, _ string, _ []string, _ string, _ time.Duration) (sporecore.CommandOutput, *sporecore.SandboxViolation) {
	return sporecore.CommandOutput{}, nil
}
func (passthroughSandbox) HandleLargeOutput(_ context.Context, content string, _ string, headTokens, tailTokens uint32) sporecore.TruncatedOutput {
	head := int(headTokens) * 4
	tail := int(tailTokens) * 4
	if len(content) <= head+tail {
		return sporecore.TruncatedOutput{Content: content, Truncated: false, OriginalSize: uint64(len(content))}
	}
	snippet := content[:head] + "\n...[truncated]...\n" + content[len(content)-tail:]
	return sporecore.TruncatedOutput{
		Content:      snippet,
		Truncated:    true,
		OriginalSize: uint64(len(content)),
	}
}
func (passthroughSandbox) ResolvePath(_ context.Context, p string, _ sporecore.Operation) (string, *sporecore.SandboxViolation) {
	return p, nil
}
func (passthroughSandbox) IsolationMode() sporecore.IsolationMode { return sporecore.IsolationNone{} }
func (passthroughSandbox) WorkspaceRoot() string                  { return "/" }

// ── Helpers ────────────────────────────────────────────────────────────────

func newState() *SessionState {
	s := NewSessionState("s1", "t1", "do the thing")
	s.WindowLimit = 1000
	s.TokenBudgetUsed = 100
	return &s
}

func newSources(rendered string, hash uint64, schemas []sporecore.ToolSchema) *ContextSources {
	return &ContextSources{
		ToolSchemas: schemas,
		ComposedPrompt: ComposedPrompt{
			Rendered:   rendered,
			Block1Hash: hash,
		},
	}
}

func newMgr() *StandardContextManager {
	return NewStandardContextManager(
		&fakeModel{tokens: 100},
		NullCacheProvider{},
		DefaultCompactionConfig(),
	)
}

// ── Rule: Assemble before every turn — produces Context using model tokens
//          and computed utilization. ──────────────────────────────────────

func TestAssemble_TokenCountFromModel(t *testing.T) {
	mgr := newMgr()
	st := newState()
	c, err := mgr.Assemble(context.Background(), st, newSources("BLOCK1", 0xAB, nil))
	if err != nil {
		t.Fatalf("assemble: %v", err)
	}
	if c.TokenCount != 100 {
		t.Errorf("tokenCount=%d want 100", c.TokenCount)
	}
	if c.WindowLimit != 1000 {
		t.Errorf("windowLimit=%d want 1000", c.WindowLimit)
	}
	if c.Utilization < 0.099 || c.Utilization > 0.101 {
		t.Errorf("utilization=%f want ~0.1", c.Utilization)
	}
}

// ── Rule: Block 1 hash invariance — second Assemble with a different
//          Block1Hash returns CacheHashMismatch. ─────────────────────────

func TestBlock1HashMismatchIsError(t *testing.T) {
	mgr := newMgr()
	st := newState()
	if _, err := mgr.Assemble(context.Background(), st, newSources("BLOCK1", 0xAB, nil)); err != nil {
		t.Fatalf("first assemble: %v", err)
	}
	_, err := mgr.Assemble(context.Background(), st, newSources("BLOCK1", 0xCD, nil))
	if err == nil {
		t.Fatalf("expected error on second assemble")
	}
	var ce *ContextError
	if !errors.As(err, &ce) {
		t.Fatalf("expected *ContextError, got %T", err)
	}
	if ce.Kind != ErrKindCacheHashMismatch || ce.Block != "static" {
		t.Errorf("got %+v", ce)
	}
}

// ── Rule: Tool schemas sorted by name. ─────────────────────────────────────

func TestToolSchemasSortedByName(t *testing.T) {
	mgr := newMgr()
	st := newState()
	schemas := []sporecore.ToolSchema{
		{Name: "zebra"}, {Name: "apple"}, {Name: "mango"},
	}
	c, err := mgr.Assemble(context.Background(), st, newSources("BLOCK1", 0xAB, schemas))
	if err != nil {
		t.Fatal(err)
	}
	got := []string{c.ToolSchemas[0].Name, c.ToolSchemas[1].Name, c.ToolSchemas[2].Name}
	want := []string{"apple", "mango", "zebra"}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("sort: got %v want %v", got, want)
			break
		}
	}
}

// ── Rule: ShouldCompact at threshold (default 80%). ───────────────────────

func TestShouldCompactAtThreshold(t *testing.T) {
	mgr := newMgr()
	st := newState()
	st.WindowLimit = 1000
	cases := []struct {
		used uint32
		want bool
	}{
		{799, false},
		{800, true},
		{900, true},
	}
	for _, c := range cases {
		st.TokenBudgetUsed = c.used
		if got := mgr.ShouldCompact(st); got != c.want {
			t.Errorf("used=%d got=%v want=%v", c.used, got, c.want)
		}
	}
}

func TestShouldCompactZeroWindow(t *testing.T) {
	mgr := newMgr()
	st := newState()
	st.WindowLimit = 0
	if mgr.ShouldCompact(st) {
		t.Errorf("expected false for zero window")
	}
}

// ── Rule: PrepareCompaction keeps recent N and uses default hints. ────────

func TestPrepareCompactionKeepsRecentN(t *testing.T) {
	mgr := newMgr()
	st := newState()
	for i := 0; i < 20; i++ {
		st.MessageHistory = append(st.MessageHistory, sporecore.Message{
			Role: sporecore.RoleAssistant, Content: sporecore.NewTextContent("m"),
		})
	}
	req, err := mgr.PrepareCompaction(st)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.MessagesToCompact) != 12 {
		t.Errorf("compact len=%d want 12", len(req.MessagesToCompact))
	}
	if !req.PreserveHints.KeepThinkingBlocks {
		t.Error("KeepThinkingBlocks should default true")
	}
	if !req.PreserveHints.KeepArchitecturalDecisions || !req.PreserveHints.KeepOpenProblems {
		t.Error("preserve hints should default true")
	}
}

// ── Rule: ApplyCompaction replaces old slice with summary + recents and
//          decrements TokenBudgetUsed by tokens_reclaimed. ────────────────

func TestApplyCompactionReplacesOldWithSummary(t *testing.T) {
	mgr := newMgr()
	st := newState()
	for i := 0; i < 20; i++ {
		st.MessageHistory = append(st.MessageHistory, sporecore.Message{
			Role: sporecore.RoleAssistant, Content: sporecore.NewTextContent("m"),
		})
	}
	st.TokenBudgetUsed = 800
	err := mgr.ApplyCompaction(st, CompactionResult{
		SummaryMessage:  sporecore.Message{Role: sporecore.RoleAssistant, Content: sporecore.NewTextContent("summary")},
		TokensReclaimed: 500,
		MessagesRemoved: 12,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(st.MessageHistory) != 9 {
		t.Errorf("len=%d want 9", len(st.MessageHistory))
	}
	if st.TokenBudgetUsed != 300 {
		t.Errorf("budget=%d want 300", st.TokenBudgetUsed)
	}
	if st.MessageHistory[0].Content.Text != "summary" {
		t.Errorf("summary not at index 0: %q", st.MessageHistory[0].Content.Text)
	}
}

func TestApplyCompactionFailsWhenTooShort(t *testing.T) {
	mgr := newMgr()
	st := newState()
	for i := 0; i < 4; i++ {
		st.MessageHistory = append(st.MessageHistory, sporecore.Message{
			Role: sporecore.RoleAssistant, Content: sporecore.NewTextContent("m"),
		})
	}
	err := mgr.ApplyCompaction(st, CompactionResult{
		SummaryMessage: sporecore.Message{Role: sporecore.RoleAssistant, Content: sporecore.NewTextContent("x")},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var ce *ContextError
	if !errors.As(err, &ce) || ce.Kind != ErrKindCompactionFailed {
		t.Errorf("wrong error: %v", err)
	}
}

// ── Rule: AppendToolResult head+tail truncates large output via sandbox. ──

func TestAppendToolResultTruncatesLargeOutput(t *testing.T) {
	mgr := newMgr().WithOffloadThreshold(64)
	st := newState()
	big := strings.Repeat("x", 8*1024)
	res := &sporecore.HarnessToolResult{
		CallID: "c1",
		Output: sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: big},
	}
	if err := mgr.AppendToolResult(context.Background(), st, res, passthroughSandbox{}); err != nil {
		t.Fatal(err)
	}
	if len(st.MessageHistory) != 1 {
		t.Fatalf("len=%d want 1", len(st.MessageHistory))
	}
	text := st.MessageHistory[0].Content.Text
	if !strings.Contains(text, "[truncated") {
		t.Errorf("expected truncation marker, got %q", text[:min(80, len(text))])
	}
	if len(text) >= len(big) {
		t.Errorf("truncation did not shrink: %d >= %d", len(text), len(big))
	}
}

func TestAppendToolResultSmallOutputPassesThrough(t *testing.T) {
	mgr := newMgr()
	st := newState()
	res := &sporecore.HarnessToolResult{
		CallID: "c1",
		Output: sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "hello"},
	}
	if err := mgr.AppendToolResult(context.Background(), st, res, passthroughSandbox{}); err != nil {
		t.Fatal(err)
	}
	if st.MessageHistory[0].Content.Text != "hello" {
		t.Errorf("expected pass-through, got %q", st.MessageHistory[0].Content.Text)
	}
}

// ── Rule: AppendResponse appends an Assistant message. ────────────────────

func TestAppendResponse(t *testing.T) {
	mgr := newMgr()
	st := newState()
	mgr.AppendResponse(st, "ack")
	if len(st.MessageHistory) != 1 || st.MessageHistory[0].Role != sporecore.RoleAssistant {
		t.Errorf("got %+v", st.MessageHistory)
	}
}

// ── Rule: InjectSkill ephemeral — no history mutation, no cache
//          invalidation (Block 1/2 hashes unchanged). ────────────────────

func TestInjectSkillIsEphemeral(t *testing.T) {
	mgr := newMgr()
	st := newState()
	c, err := mgr.Assemble(context.Background(), st, newSources("BLOCK1", 0xAB, nil))
	if err != nil {
		t.Fatal(err)
	}
	beforeStatic := c.SystemPrompt.StaticBlockHash
	beforeSession := c.SystemPrompt.SessionBlockHash
	beforeMsgs := len(c.Messages)
	if err := mgr.InjectSkill(c, &Guide{ID: "rust-style", Content: "prefer iterators"}); err != nil {
		t.Fatal(err)
	}
	if c.SystemPrompt.StaticBlockHash != beforeStatic {
		t.Error("static hash changed")
	}
	if c.SystemPrompt.SessionBlockHash != beforeSession {
		t.Error("session hash changed")
	}
	if len(c.Messages) != beforeMsgs {
		t.Error("message history changed")
	}
	if !strings.Contains(c.SystemPrompt.Content, "[SKILL:rust-style]") {
		t.Error("skill not injected into system prompt")
	}
	if len(c.Meta.SkillsInjected) != 1 {
		t.Errorf("skills_injected=%v", c.Meta.SkillsInjected)
	}
}

// ── Rule: RecordCacheResult updates ContextMeta.CacheBlocks. ──────────────

func TestRecordCacheResult(t *testing.T) {
	mgr := newMgr()
	st := newState()
	c, err := mgr.Assemble(context.Background(), st, newSources("BLOCK1", 0xAB, nil))
	if err != nil {
		t.Fatal(err)
	}
	tr, fa := true, false
	mgr.RecordCacheResult(c, CacheBlockHits{StaticHit: &tr, SessionHit: &fa, HistoryHit: &tr})
	if c.Meta.CacheBlocks.StaticHit == nil || !*c.Meta.CacheBlocks.StaticHit {
		t.Error("static_hit not recorded")
	}
	if c.Meta.CacheBlocks.SessionHit == nil || *c.Meta.CacheBlocks.SessionHit {
		t.Error("session_hit not recorded")
	}
	if c.Meta.CacheBlocks.HistoryHit == nil || !*c.Meta.CacheBlocks.HistoryHit {
		t.Error("history_hit not recorded")
	}
}

// ── Rule: CacheProvider.Annotate is invoked at end of Assemble. ───────────

func TestCacheProviderAnnotateIsCalled(t *testing.T) {
	cache := &countingCache{}
	mgr := NewStandardContextManager(&fakeModel{tokens: 10}, cache, DefaultCompactionConfig())
	st := newState()
	srcs := newSources("BLOCK1", 0xAB, nil)
	if _, err := mgr.Assemble(context.Background(), st, srcs); err != nil {
		t.Fatal(err)
	}
	if _, err := mgr.Assemble(context.Background(), st, srcs); err != nil {
		t.Fatal(err)
	}
	if got := cache.calls.Load(); got != 2 {
		t.Errorf("annotate calls=%d want 2", got)
	}
}

// ── Rule: Pending skill injections become Block-3 segments and are
//          reflected in ContextMeta.SkillsInjected. ──────────────────────

func TestPendingSkillInjectionsAppearInMeta(t *testing.T) {
	mgr := newMgr()
	st := newState()
	st.PendingSkillInjections = append(st.PendingSkillInjections, Guide{ID: "g1", Content: "do x"})
	c, err := mgr.Assemble(context.Background(), st, newSources("BLOCK1", 0xAB, nil))
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Meta.SkillsInjected) != 1 || c.Meta.SkillsInjected[0] != "g1" {
		t.Errorf("skills_injected=%v", c.Meta.SkillsInjected)
	}
	if !strings.Contains(c.SystemPrompt.Content, "do x") {
		t.Error("skill content missing from system prompt")
	}
}

// ── Rule: Budget warning lives in Block 3 only when active. ───────────────

func TestBudgetWarningOnlyWhenActive(t *testing.T) {
	mgr := newMgr()
	st := newState()
	cOff, err := mgr.Assemble(context.Background(), st, newSources("BLOCK1", 0xAB, nil))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(cOff.SystemPrompt.Content, "[BUDGET]") {
		t.Error("budget warning should not appear when inactive")
	}
	st.BudgetWarningActive = true
	cOn, err := mgr.Assemble(context.Background(), st, newSources("BLOCK1", 0xAB, nil))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(cOn.SystemPrompt.Content, "[BUDGET]") {
		t.Error("budget warning should appear when active")
	}
}

// ── Rule: TokenCountFailed surfaces when ModelInterface fails. ────────────

func TestTokenCountFailureSurfaces(t *testing.T) {
	mgr := NewStandardContextManager(failingModel{}, NullCacheProvider{}, DefaultCompactionConfig())
	st := newState()
	_, err := mgr.Assemble(context.Background(), st, newSources("BLOCK1", 0xAB, nil))
	if err == nil {
		t.Fatal("expected error")
	}
	var ce *ContextError
	if !errors.As(err, &ce) || ce.Kind != ErrKindTokenCountFailed {
		t.Errorf("wrong error: %v", err)
	}
}

// ── Cache-stability invariant: same inputs ⇒ identical rendered prefix
//   and identical hashes (the contract the prefix cache relies on). ──────

func TestDeterministicPrefixAcrossCalls(t *testing.T) {
	mgr := newMgr()
	st := newState()
	srcs := newSources("BLOCK1-content", 0x11, nil)
	a, err := mgr.Assemble(context.Background(), st, srcs)
	if err != nil {
		t.Fatal(err)
	}
	b, err := mgr.Assemble(context.Background(), st, srcs)
	if err != nil {
		t.Fatal(err)
	}
	if a.SystemPrompt.Content != b.SystemPrompt.Content {
		t.Error("rendered content differs across calls")
	}
	if a.SystemPrompt.StaticBlockHash != b.SystemPrompt.StaticBlockHash {
		t.Error("static hash differs across calls")
	}
	if a.SystemPrompt.SessionBlockHash != b.SystemPrompt.SessionBlockHash {
		t.Error("session hash differs across calls")
	}
}

// ── Coverage: ToRequest projects to a sporecore.ModelRequest with a
//   leading system message. ─────────────────────────────────────────────

func TestContextToRequest(t *testing.T) {
	mgr := newMgr()
	st := newState()
	c, err := mgr.Assemble(context.Background(), st, newSources("BLOCK1", 0xAB, nil))
	if err != nil {
		t.Fatal(err)
	}
	req := c.ToRequest(sporecore.ModelParams{})
	if len(req.Messages) < 1 || req.Messages[0].Role != sporecore.RoleSystem {
		t.Errorf("first message is not system: %+v", req.Messages)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
