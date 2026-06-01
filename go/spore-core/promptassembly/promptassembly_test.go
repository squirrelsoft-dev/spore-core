package promptassembly

import (
	"context"
	"encoding/json"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/promptchunkregistry"
)

func newCtx() AssemblyContext {
	return NewAssemblyContext(
		sporecore.SessionID("s1"),
		sporecore.TaskID("t1"),
		1,
		promptchunkregistry.ModeSafeAuto,
		sporecore.PhaseExecution,
	)
}

func ids(chunks []PromptChunk) []string {
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, c.ID)
	}
	return out
}

func eqIDs(t *testing.T, got []string, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("id mismatch: got %v want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("id mismatch at %d: got %v want %v", i, got, want)
		}
	}
}

func strptr(s string) *string { return &s }

// ── R1: Always always matches ──────────────────────────────────────────────
func TestR1AlwaysMatches(t *testing.T) {
	b := NewContextSourcesBuilder()
	c := newCtx()
	if !b.Evaluate(Always(), &c) {
		t.Fatal("Always should match")
	}
}

// ── R2: WhenMode ────────────────────────────────────────────────────────────
func TestR2WhenMode(t *testing.T) {
	b := NewContextSourcesBuilder()
	c := newCtx()
	c.Mode = promptchunkregistry.ModePlan
	if !b.Evaluate(WhenMode(promptchunkregistry.ModePlan), &c) {
		t.Fatal("expected plan match")
	}
	if b.Evaluate(WhenMode(promptchunkregistry.ModeAutoEdit), &c) {
		t.Fatal("expected yolo no-match")
	}
}

// ── R3: WhenToolActive ──────────────────────────────────────────────────────
func TestR3WhenToolActive(t *testing.T) {
	b := NewContextSourcesBuilder()
	c := newCtx()
	c.AddTool("bash")
	if !b.Evaluate(WhenToolActive("bash"), &c) {
		t.Fatal("expected bash active")
	}
	if b.Evaluate(WhenToolActive("grep"), &c) {
		t.Fatal("expected grep inactive")
	}
}

// ── R4: WhenToolCapability ──────────────────────────────────────────────────
func TestR4WhenToolCapability(t *testing.T) {
	b := NewContextSourcesBuilder()
	c := newCtx()
	c.AddCapability("bash", "sandbox")
	if !b.Evaluate(WhenToolCapability("bash", "sandbox"), &c) {
		t.Fatal("expected (bash,sandbox) match")
	}
	if b.Evaluate(WhenToolCapability("bash", "git"), &c) {
		t.Fatal("expected (bash,git) no-match")
	}
}

// ── R5: WhenPhase / WhenAgentType / WhenFeature ────────────────────────────
func TestR5PhaseAgentFeature(t *testing.T) {
	b := NewContextSourcesBuilder()
	c := newCtx()
	c.Phase = sporecore.PhasePlanning
	c.AgentType = "planner"
	c.Features["beta"] = true
	c.Features["alpha"] = false

	if !b.Evaluate(WhenPhase(sporecore.PhasePlanning), &c) {
		t.Fatal("expected planning phase match")
	}
	if b.Evaluate(WhenPhase(sporecore.PhaseCleanup), &c) {
		t.Fatal("expected cleanup no-match")
	}
	if !b.Evaluate(WhenAgentType("planner"), &c) {
		t.Fatal("expected planner agent match")
	}
	if b.Evaluate(WhenAgentType("coder"), &c) {
		t.Fatal("expected coder no-match")
	}
	if !b.Evaluate(WhenFeature("beta"), &c) {
		t.Fatal("expected beta=true match")
	}
	if b.Evaluate(WhenFeature("alpha"), &c) {
		t.Fatal("expected alpha=false no-match")
	}
	if b.Evaluate(WhenFeature("missing"), &c) {
		t.Fatal("expected missing feature no-match")
	}
}

// ── R6: OnTrigger ───────────────────────────────────────────────────────────
func TestR6OnTrigger(t *testing.T) {
	b := NewContextSourcesBuilder()
	c := newCtx()
	cond := OnTrigger([]string{"deploy", "rollback"})
	if b.Evaluate(cond, &c) {
		t.Fatal("nil message should not match")
	}
	c.IncomingMessage = strptr("please deploy the service")
	if !b.Evaluate(cond, &c) {
		t.Fatal("expected substring match")
	}
	c.IncomingMessage = strptr("nothing relevant")
	if b.Evaluate(cond, &c) {
		t.Fatal("expected no match")
	}
}

// ── R7: OnEvent ─────────────────────────────────────────────────────────────
func TestR7OnEvent(t *testing.T) {
	b := NewContextSourcesBuilder()
	c := newCtx()
	cond := OnEvent(sporecore.HookEventPreCompact)
	if b.Evaluate(cond, &c) {
		t.Fatal("no pending events should not match")
	}
	c.PendingEvents = append(c.PendingEvents, sporecore.HookEventPreCompact)
	if !b.Evaluate(cond, &c) {
		t.Fatal("expected pending event match")
	}
}

// ── R8: All / Any / Not ─────────────────────────────────────────────────────
func TestR8AllAnyNot(t *testing.T) {
	b := NewContextSourcesBuilder()
	c := newCtx()
	c.Mode = promptchunkregistry.ModePlan
	c.AddTool("bash")

	all := All(WhenMode(promptchunkregistry.ModePlan), WhenToolActive("bash"))
	if !b.Evaluate(all, &c) {
		t.Fatal("expected All true")
	}
	allFail := All(WhenMode(promptchunkregistry.ModePlan), WhenToolActive("grep"))
	if b.Evaluate(allFail, &c) {
		t.Fatal("expected All false")
	}
	any := Any(WhenToolActive("grep"), WhenMode(promptchunkregistry.ModePlan))
	if !b.Evaluate(any, &c) {
		t.Fatal("expected Any true")
	}
	not := Not(WhenMode(promptchunkregistry.ModeAutoEdit))
	if !b.Evaluate(not, &c) {
		t.Fatal("expected Not true")
	}
}

// ── R9: Custom ──────────────────────────────────────────────────────────────
func TestR9Custom(t *testing.T) {
	b := NewContextSourcesBuilder()
	c := newCtx()
	c.TurnNumber = 5
	cond := Custom(func(ctx *AssemblyContext) bool { return ctx.TurnNumber > 3 })
	if !b.Evaluate(cond, &c) {
		t.Fatal("expected custom true at turn 5")
	}
	c.TurnNumber = 1
	if b.Evaluate(cond, &c) {
		t.Fatal("expected custom false at turn 1")
	}
}

// ── R10: bucketed by stability ─────────────────────────────────────────────
func TestR10BucketedByStability(t *testing.T) {
	b := NewContextSourcesBuilderWithChunks([]PromptChunk{
		NewPromptChunk("s", "static").WithStability(contextmgr.StabilityStatic),
		NewPromptChunk("ps", "session").WithStability(contextmgr.StabilityPerSession),
		NewPromptChunk("pt", "turn").WithStability(contextmgr.StabilityPerTurn),
	})
	c := newCtx()
	buckets := b.Assemble(&c)
	eqIDs(t, ids(buckets.Static), []string{"s"})
	eqIDs(t, ids(buckets.PerSession), []string{"ps"})
	eqIDs(t, ids(buckets.PerTurn), []string{"pt"})
}

// ── R11: registration order preserved within bucket ────────────────────────
func TestR11RegistrationOrder(t *testing.T) {
	b := NewContextSourcesBuilderWithChunks([]PromptChunk{
		NewPromptChunk("a", "a"),
		NewPromptChunk("b", "b"),
		NewPromptChunk("c", "c"),
	})
	c := newCtx()
	buckets := b.Assemble(&c)
	eqIDs(t, ids(buckets.Static), []string{"a", "b", "c"})
}

// ── R12: tool-affinity 4-way matrix ────────────────────────────────────────
func TestR12ToolAffinityMatrix(t *testing.T) {
	chunk := NewPromptChunk("bash-git", "git guide").
		WithToolAffinity(NewToolCapabilityAffinity("bash", "git"))
	b := NewContextSourcesBuilderWithChunks([]PromptChunk{chunk})

	// (1) tool inactive, cap inactive -> excluded
	c := newCtx()
	if len(b.Assemble(&c).Static) != 0 {
		t.Fatal("(1) expected excluded")
	}
	// (2) tool active, cap inactive -> excluded (cap required)
	c.AddTool("bash")
	if len(b.Assemble(&c).Static) != 0 {
		t.Fatal("(2) expected excluded")
	}
	// (3) tool active, cap active -> included
	c.AddCapability("bash", "git")
	if len(b.Assemble(&c).Static) != 1 {
		t.Fatal("(3) expected included")
	}
	// (4) tool inactive but cap present -> excluded (tool gate first)
	c2 := newCtx()
	c2.AddCapability("bash", "git")
	if len(b.Assemble(&c2).Static) != 0 {
		t.Fatal("(4) expected excluded")
	}

	// Capability empty: included as soon as the tool is active.
	chunk2 := NewPromptChunk("bash-any", "bash guide").
		WithToolAffinity(NewToolAffinity("bash"))
	b2 := NewContextSourcesBuilderWithChunks([]PromptChunk{chunk2})
	c3 := newCtx()
	if len(b2.Assemble(&c3).Static) != 0 {
		t.Fatal("tool-only: expected excluded when inactive")
	}
	c3.AddTool("bash")
	if len(b2.Assemble(&c3).Static) != 1 {
		t.Fatal("tool-only: expected included when active")
	}
}

// ── R13: OnTrigger matches pushed to PerTurn ───────────────────────────────
func TestR13TriggerRoutesToPerTurn(t *testing.T) {
	chunk := NewPromptChunk("playbook", "rollback steps").
		WithStability(contextmgr.StabilityStatic).
		WithCondition(OnTrigger([]string{"rollback"})).
		WithTriggers([]string{"rollback"})
	b := NewContextSourcesBuilderWithChunks([]PromptChunk{chunk})

	c := newCtx()
	if len(b.Assemble(&c).Static) != 0 || len(b.Assemble(&c).PerTurn) != 0 {
		t.Fatal("no message -> nothing included")
	}
	c.IncomingMessage = strptr("we must rollback now")
	buckets := b.Assemble(&c)
	if len(buckets.Static) != 0 {
		t.Fatal("expected static empty (trigger routes to per_turn)")
	}
	eqIDs(t, ids(buckets.PerTurn), []string{"playbook"})
}

// ── R14: OnEvent injected to PerTurn only when pending ──────────────────────
func TestR14OnEventInjectedOnlyWhenPending(t *testing.T) {
	chunk := NewPromptChunk("reminder", "system reminder").
		WithStability(contextmgr.StabilityPerTurn).
		WithCondition(OnEvent(sporecore.HookEventPreCompact))
	b := NewContextSourcesBuilderWithChunks([]PromptChunk{chunk})

	c := newCtx()
	if len(b.Assemble(&c).PerTurn) != 0 {
		t.Fatal("no pending event -> excluded")
	}
	c.PendingEvents = append(c.PendingEvents, sporecore.HookEventPreCompact)
	buckets := b.Assemble(&c)
	eqIDs(t, ids(buckets.PerTurn), []string{"reminder"})
}

// ── R15: Block-1 hash stable across two builds of identical Static set ──────
func TestR15Block1HashStable(t *testing.T) {
	mk := func() *ContextSourcesBuilder {
		return NewContextSourcesBuilderWithChunks([]PromptChunk{
			NewPromptChunk("core", "identity rules"),
			NewPromptChunk("style", "be concise"),
		})
	}
	c := newCtx()
	b1 := mk()
	b2 := mk()
	cp1 := b1.ComposeBlock1(b1.Assemble(&c))
	cp2 := b2.ComposeBlock1(b2.Assemble(&c))
	if cp1.Block1Hash != cp2.Block1Hash {
		t.Fatal("expected stable block-1 hash")
	}

	b3 := NewContextSourcesBuilderWithChunks([]PromptChunk{
		NewPromptChunk("core", "DIFFERENT identity"),
		NewPromptChunk("style", "be concise"),
	})
	cp3 := b3.ComposeBlock1(b3.Assemble(&c))
	if cp1.Block1Hash == cp3.Block1Hash {
		t.Fatal("expected different hash for different content")
	}
}

// ── R16: cache_breakpoint preserved ─────────────────────────────────────────
func TestR16CacheBreakpoint(t *testing.T) {
	b := NewContextSourcesBuilderWithChunks([]PromptChunk{
		NewPromptChunk("a", "a"),
		NewPromptChunk("b", "b").WithCacheBreakpoint(true),
		NewPromptChunk("c", "c"),
	})
	c := newCtx()
	buckets := b.Assemble(&c)
	eqIDs(t, BreakpointIDs(buckets), []string{"b"})

	segs := ChunksToSegments(buckets.Static)
	for _, s := range segs {
		if s.Name == "b" && !s.CacheBreakpoint {
			t.Fatal("b segment should carry breakpoint")
		}
		if s.Name == "a" && s.CacheBreakpoint {
			t.Fatal("a segment should not carry breakpoint")
		}
	}
}

// ── R17: tool not active yields no description chunk ────────────────────────
func TestR17ToolInactiveNoDescription(t *testing.T) {
	chunk := NewPromptChunk("bash-desc", "Bash tool: run shell commands").
		WithToolAffinity(NewToolAffinity("bash"))
	b := NewContextSourcesBuilderWithChunks([]PromptChunk{chunk})
	c := newCtx()
	if len(b.Assemble(&c).Static) != 0 {
		t.Fatal("inactive tool -> excluded")
	}
	c.AddTool("bash")
	if len(b.Assemble(&c).Static) != 1 {
		t.Fatal("active tool -> included")
	}
}

// ── R18: EmbeddedChunkProvider invalidate no-op, load same ──────────────────
func TestR18EmbeddedProvider(t *testing.T) {
	p := NewEmbeddedChunkProvider([]PromptChunk{NewPromptChunk("x", "y")})
	a, err := p.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	p.Invalidate()
	b, err := p.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 1 || len(b) != 1 || a[0].ID != b[0].ID {
		t.Fatal("embedded load should be stable")
	}
}

// ── R19: InMemoryChunkProvider returns registered; set replaces ─────────────
func TestR19InMemoryProvider(t *testing.T) {
	p := NewInMemoryChunkProvider([]PromptChunk{NewPromptChunk("x", "y")})
	got, err := p.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatal("expected 1 chunk")
	}
	p.Set([]PromptChunk{NewPromptChunk("a", "1"), NewPromptChunk("b", "2")})
	after, err := p.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != 2 || after[0].ID != "a" {
		t.Fatalf("expected replaced set, got %v", ids(after))
	}
}

// ── R21: CompositeChunkProvider merges + propagates invalidate ──────────────
type countingProvider struct {
	invalidated int
	chunks      []PromptChunk
}

func (c *countingProvider) Load(_ context.Context) ([]PromptChunk, error) {
	return c.chunks, nil
}
func (c *countingProvider) Invalidate() { c.invalidated++ }

func TestR21CompositeProvider(t *testing.T) {
	p1 := &countingProvider{chunks: []PromptChunk{NewPromptChunk("a", "1")}}
	p2 := &countingProvider{chunks: []PromptChunk{NewPromptChunk("b", "2"), NewPromptChunk("c", "3")}}
	comp := NewCompositeChunkProvider().Add(p1).Add(p2)
	merged, err := comp.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	eqIDs(t, ids(merged), []string{"a", "b", "c"})

	comp.Invalidate()
	if p1.invalidated != 1 || p2.invalidated != 1 {
		t.Fatalf("expected invalidate propagated, got %d/%d", p1.invalidated, p2.invalidated)
	}
}

// ── R25: HarnessBuilder defaults + overrides ────────────────────────────────
func TestR25HarnessBuilder(t *testing.T) {
	// Default: empty InMemoryChunkProvider.
	hb := NewHarnessBuilder()
	got, err := hb.Provider().Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatal("default provider should be empty")
	}
	if _, ok := hb.Provider().(*InMemoryChunkProvider); !ok {
		t.Fatal("default provider should be InMemoryChunkProvider")
	}

	// Chunks override.
	hb.Chunks([]PromptChunk{NewPromptChunk("x", "y")})
	got, _ = hb.Provider().Load(context.Background())
	if len(got) != 1 {
		t.Fatal("Chunks should register inline")
	}

	// ChunkProvider override.
	emb := NewEmbeddedChunkProvider([]PromptChunk{NewPromptChunk("e", "e")})
	hb.ChunkProvider(emb)
	if hb.Provider() != emb {
		t.Fatal("ChunkProvider should set the provider directly")
	}
}

// ── PartialEq/Equal: Custom never equal (A3) ───────────────────────────────
func TestCustomConditionNeverEqual(t *testing.T) {
	f := func(*AssemblyContext) bool { return true }
	a := Custom(f)
	b := Custom(f)
	if a.Equal(b) {
		t.Fatal("Custom should never be equal")
	}
	if !Always().Equal(Always()) {
		t.Fatal("Always should equal Always")
	}
	if !WhenMode(promptchunkregistry.ModeAutoEdit).Equal(WhenMode(promptchunkregistry.ModeAutoEdit)) {
		t.Fatal("WhenMode should compare by value")
	}
}

// ── Serialization: Custom skipped (A3) ─────────────────────────────────────
func TestCustomConditionSerializesToNull(t *testing.T) {
	cond := Custom(func(*AssemblyContext) bool { return true })
	data, err := json.Marshal(cond)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "null" {
		t.Fatalf("expected null, got %s", data)
	}

	// A PromptChunk carrying Custom round-trips back to Always.
	chunk := NewPromptChunk("x", "y").WithCondition(Custom(func(*AssemblyContext) bool { return true }))
	s, err := json.Marshal(chunk)
	if err != nil {
		t.Fatal(err)
	}
	var back PromptChunk
	if err := json.Unmarshal(s, &back); err != nil {
		t.Fatal(err)
	}
	if back.Condition.Kind() != KindAlways {
		t.Fatalf("expected Always after round-trip, got %s", back.Condition.Kind())
	}
}

// ── Serialization: serializable variants round-trip ────────────────────────
func TestConditionRoundTripsSerializableVariants(t *testing.T) {
	cond := All(
		WhenMode(promptchunkregistry.ModePlan),
		Any(
			WhenToolActive("bash"),
			Not(WhenFeature("beta")),
		),
		OnEvent(sporecore.HookEventPreTurn),
		OnTrigger([]string{"deploy"}),
		WhenToolCapability("bash", "git"),
		WhenPhase(sporecore.PhasePlanning),
		WhenAgentType("planner"),
	)
	s, err := json.Marshal(cond)
	if err != nil {
		t.Fatal(err)
	}
	var back ChunkCondition
	if err := json.Unmarshal(s, &back); err != nil {
		t.Fatal(err)
	}
	if !cond.Equal(back) {
		t.Fatalf("round-trip mismatch:\n%s", s)
	}
}

// ── Serialization: Custom pruned from combinators ──────────────────────────
func TestCustomPrunedFromCombinators(t *testing.T) {
	cond := All(
		WhenMode(promptchunkregistry.ModeAutoEdit),
		Custom(func(*AssemblyContext) bool { return true }),
	)
	s, err := json.Marshal(cond)
	if err != nil {
		t.Fatal(err)
	}
	var back ChunkCondition
	if err := json.Unmarshal(s, &back); err != nil {
		t.Fatal(err)
	}
	want := All(WhenMode(promptchunkregistry.ModeAutoEdit))
	if !back.Equal(want) {
		t.Fatalf("expected Custom pruned, got %s", s)
	}
}

// ── ChunkProviderError variants ─────────────────────────────────────────────
func TestChunkProviderErrorVariants(t *testing.T) {
	e := NewLoadFailedError("remote", "timeout")
	if got := e.Error(); got == "" || !contains(got, "remote") || !contains(got, "timeout") {
		t.Fatalf("unexpected error string: %q", got)
	}
	s, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var back ChunkProviderError
	if err := json.Unmarshal(s, &back); err != nil {
		t.Fatal(err)
	}
	if back.Kind != ChunkProviderErrorLoadFailed || back.Provider != "remote" || back.Detail != "timeout" {
		t.Fatalf("round-trip mismatch: %+v", back)
	}

	p := NewParseError("bad json")
	if !contains(p.Error(), "bad json") {
		t.Fatalf("unexpected parse error: %q", p.Error())
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// ── BuildContextSources passes guides/memory/schemas through (A5) ──────────
func TestBuildContextSourcesPassthrough(t *testing.T) {
	b := NewContextSourcesBuilderWithChunks([]PromptChunk{
		NewPromptChunk("core", "rules"),
		NewPromptChunk("ps", "ref").WithStability(contextmgr.StabilityPerSession),
	})
	c := newCtx()
	guides := []contextmgr.Guide{{ID: "g1", Content: "guide"}}
	sources, buckets := b.BuildContextSources(&c, guides, nil, nil)
	if sources.ComposedPrompt.Rendered != "rules" {
		t.Fatalf("expected only Static in block 1, got %q", sources.ComposedPrompt.Rendered)
	}
	if sources.ComposedPrompt.Block1Hash == 0 {
		t.Fatal("expected non-zero block-1 hash")
	}
	if len(buckets.PerSession) != 1 {
		t.Fatal("expected one per-session chunk")
	}
	if len(sources.Guides) != 1 || sources.Guides[0].ID != "g1" {
		t.Fatal("guides should pass through verbatim")
	}
}

// ── StorageScope wire form ──────────────────────────────────────────────────
func TestStorageScopeWire(t *testing.T) {
	cases := map[StorageScope]string{
		StorageScopeUser:    `"user"`,
		StorageScopeProject: `"project"`,
		StorageScopeLocal:   `"local"`,
	}
	for scope, want := range cases {
		got, err := json.Marshal(scope)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != want {
			t.Fatalf("scope %v: got %s want %s", scope, got, want)
		}
	}
}

// ── agent_affinity gate ─────────────────────────────────────────────────────
func TestAgentAffinityGate(t *testing.T) {
	chunk := NewPromptChunk("planner-prompt", "you plan").WithAgentAffinity("planner")
	b := NewContextSourcesBuilderWithChunks([]PromptChunk{chunk})
	c := newCtx()
	if len(b.Assemble(&c).Static) != 0 {
		t.Fatal("no agent type -> excluded")
	}
	c.AgentType = "planner"
	if len(b.Assemble(&c).Static) != 1 {
		t.Fatal("matching agent -> included")
	}
	c.AgentType = "coder"
	if len(b.Assemble(&c).Static) != 0 {
		t.Fatal("non-matching agent -> excluded")
	}
}

// ── AssemblyContext wire round-trip ─────────────────────────────────────────
func TestAssemblyContextRoundTrip(t *testing.T) {
	c := newCtx()
	c.AddTool("bash")
	c.AddCapability("bash", "git")
	c.AgentType = "planner"
	c.IncomingMessage = strptr("hello")
	c.PendingEvents = []sporecore.HookEvent{sporecore.HookEventPreTurn}
	c.Features["beta"] = true

	data, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	var back AssemblyContext
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if !back.hasTool("bash") || !back.hasCapability("bash", "git") {
		t.Fatal("tool/capability sets should round-trip")
	}
	if back.AgentType != "planner" || back.StorageScope != StorageScopeProject {
		t.Fatalf("scalar fields mismatch: %+v", back)
	}
	if back.IncomingMessage == nil || *back.IncomingMessage != "hello" {
		t.Fatal("incoming message should round-trip")
	}
}

// ── default condition when absent in JSON (A3 / #81 discipline) ─────────────
func TestPromptChunkAbsentConditionDefaultsAlways(t *testing.T) {
	var chunk PromptChunk
	if err := json.Unmarshal([]byte(`{"id":"x","content":"y","stability":"static"}`), &chunk); err != nil {
		t.Fatal(err)
	}
	if chunk.Condition.Kind() != KindAlways {
		t.Fatalf("absent condition should default to Always, got %s", chunk.Condition.Kind())
	}
}

var _ = promptchunkregistry.ModeSafeAuto
