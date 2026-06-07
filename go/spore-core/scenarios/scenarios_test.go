// Hermetic end-to-end scenario tests (issue #57).
//
// These drive the SAME BuildScenario wiring the e2e-agent command uses, but with
// a scripted MockAgent, scripted/real tool registries, and an allow-all sandbox,
// so CI never needs a live Ollama or any network. Each test asserts the harness
// loop control flow (turn count, tool dispatch order, S4 recovery sequencing, S3
// live compaction with real token reclamation). SPORE_OTLP_ENDPOINT stays unset,
// so there is no forwarding.

package scenarios_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
	obs "github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/scenarios"
)

func usage() sporecore.TokenUsage {
	return sporecore.TokenUsage{InputTokens: 10, OutputTokens: 5}
}

func toolCall(id, name string, input string) sporecore.ToolCall {
	return sporecore.ToolCall{ID: id, Name: name, Input: json.RawMessage(input)}
}

func reactStrategy(maxIter uint32) sporecore.LoopStrategy {
	return sporecore.ReActStrategy(maxIter)
}

// ---------------------------------------------------------------------------
// S1 — multi-step / multi-tool
// ---------------------------------------------------------------------------

func TestS1MultiStepMultiTool(t *testing.T) {
	agent := sporecore.NewMockAgent("mock")
	agent.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{toolCall("c1", "read_file", `{"path":"input.txt"}`)}, usage()))
	agent.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{toolCall("c2", "write_file", `{"path":"output.txt","content":"UPPERCASED"}`)}, usage()))
	agent.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{toolCall("c3", "read_file", `{"path":"output.txt"}`)}, usage()))
	agent.Push(sporecore.NewFinalResponse("DONE", usage()))

	tools := sporecore.NewScriptedToolRegistry()
	tools.Push(sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "hello"})
	tools.Push(sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "wrote 10 bytes"})
	tools.Push(sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "UPPERCASED"})

	h := scenarios.BuildScenario(
		agent, tools, sporecore.AllowAllSandbox{}, sporecore.NoopContextManager{},
		sporecore.AlwaysContinuePolicy{}, nil, nil, 2, nil,
	)

	task := sporecore.NewTask(scenarios.S1.Prompt(), "s1-test", reactStrategy(8))
	res := h.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
	if res.Kind != sporecore.RunSuccess {
		t.Fatalf("S1 expected Success, got %v (%s)", res.Kind, res.Reason.Kind)
	}
	if res.Turns <= 2 {
		t.Fatalf("S1 should take >2 turns, got %d", res.Turns)
	}
	if got := tools.CallCount.Load(); got != 3 {
		t.Fatalf("S1 dispatches read+write+readback = 3 tools, got %d", got)
	}
}

// ---------------------------------------------------------------------------
// S2 — multi-turn, same SessionID, carrying session state
// ---------------------------------------------------------------------------

func TestS2MultiTurnCarriesState(t *testing.T) {
	sessionID := sporecore.SessionID("s2-test")
	agent := sporecore.NewMockAgent("mock")
	// Turn 1: write notes.md, then final.
	agent.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{toolCall("c1", "write_file", `{"path":"notes.md","content":"TODO: set up the project"}`)}, usage()))
	agent.Push(sporecore.NewFinalResponse("DONE", usage()))
	// Turn 2: append referencing turn 1, then final.
	agent.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{toolCall("c2", "write_file", `{"path":"notes.md","content":"TODO: follow up on set up the project","append":true}`)}, usage()))
	agent.Push(sporecore.NewFinalResponse("DONE referencing set up the project", usage()))

	tools := sporecore.NewScriptedToolRegistry()

	h := scenarios.BuildScenario(
		agent, tools, sporecore.AllowAllSandbox{}, sporecore.NoopContextManager{},
		sporecore.AlwaysContinuePolicy{}, nil, nil, 2, nil,
	)

	task1 := sporecore.NewTask(scenarios.S2.Prompt(), sessionID, reactStrategy(5))
	r1 := h.Run(context.Background(), sporecore.NewHarnessRunOptions(task1))
	if r1.Kind != sporecore.RunSuccess {
		t.Fatalf("S2 turn 1 expected Success, got %v", r1.Kind)
	}
	// Carry session state into turn 2 (same SessionID).
	carried := sporecore.SessionState{}

	task2 := sporecore.NewTask("add a second item referencing the first", sessionID, reactStrategy(5))
	opts := sporecore.NewHarnessRunOptions(task2)
	opts.SessionState = &carried
	r2 := h.Run(context.Background(), opts)
	if r2.Kind != sporecore.RunSuccess {
		t.Fatalf("S2 turn 2 expected Success, got %v", r2.Kind)
	}
	if r2.SessionID != sessionID {
		t.Fatalf("S2 session id = %q, want %q (same across turns)", r2.SessionID, sessionID)
	}
	if !contains(r2.Output, "set up the project") {
		t.Fatalf("S2 turn 2 should reference turn 1 content, got %q", r2.Output)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (func() bool {
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	})()
}

// ---------------------------------------------------------------------------
// S3 — live compaction with real token reclamation
// ---------------------------------------------------------------------------

func TestS3LiveCompactionReclaimsTokens(t *testing.T) {
	sessionID := sporecore.SessionID("s3-test")
	// Agent emits a tool call (to reach the post-tool compaction arm), then a
	// final summary containing the key terms so verification passes, then a final
	// response after compaction.
	agent := sporecore.NewMockAgent("mock")
	agent.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{toolCall("c1", "read_file", `{"path":"x"}`)}, usage()))
	agent.Push(sporecore.NewFinalResponse("summary: continuing the deploy of the payment service", usage()))
	agent.Push(sporecore.NewFinalResponse("DONE deploy payment service", usage()))

	tools := sporecore.NewScriptedToolRegistry()
	tools.Push(sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "file contents"})

	model := sporecore.NewMockModel(sporecore.ProviderInfo{Name: "mock", ModelID: "mock", ContextWindow: 200})
	cfg := contextmgr.CompactionConfig{Threshold: 0.80, PreserveRecentN: 2, HeadTailTokens: 64, OffloadPath: ".spore/offload", MaxCompactionAttempts: 2}
	rich := contextmgr.NewStandardContextManager(model, nil, cfg)
	cm := contextmgr.NewStandardCompactionAdapter(rich)

	provider := obs.NewInMemoryObservabilityProvider()
	observer := obs.NewHarnessObserver(provider, obs.DefaultPricing())

	h := scenarios.BuildScenario(
		agent, tools, sporecore.AllowAllSandbox{}, cm,
		sporecore.AlwaysContinuePolicy{}, nil, contextmgr.NewKeyTermVerifier(), 2, observer,
	)

	task := sporecore.NewTask("deploy the payment service", sessionID, reactStrategy(8))
	var state sporecore.SessionState
	// Seed a small window with budget over threshold (0.85 > 0.80) + long history.
	scenarios.SeedCompactionState(&state, "deploy the payment service", sessionID, task.ID, 200, 170, 12)
	opts := sporecore.NewHarnessRunOptions(task)
	opts.SessionState = &state
	res := h.Run(context.Background(), opts)
	if res.Kind != sporecore.RunSuccess {
		t.Fatalf("S3 expected Success, got %v (%s)", res.Kind, res.Reason.Kind)
	}

	// A Compaction context span was emitted, and it reclaimed real tokens with a
	// budget that dropped.
	trace, err := provider.GetTrace(context.Background(), sessionID)
	if err != nil {
		t.Fatal(err)
	}
	var compactions []obs.ContextSpan
	for _, span := range trace {
		if cs, ok := span.(obs.ContextSpan); ok && cs.Base.Kind == obs.SpanKindCompaction {
			compactions = append(compactions, cs)
		}
	}
	if len(compactions) == 0 {
		t.Fatal("S3 should emit >=1 Compaction span mid-run")
	}
	first := compactions[0]
	if first.TokensAfter >= first.TokensBefore {
		t.Fatalf("token_budget_used must drop after compaction: %d -> %d", first.TokensBefore, first.TokensAfter)
	}
	if first.Operation.TokensReclaimed == 0 {
		t.Fatal("real reclamation must be > 0")
	}
}

// ---------------------------------------------------------------------------
// S4 — tool failure + recovery (uses the REAL registry + FailingTool)
// ---------------------------------------------------------------------------

func TestS4ToolFailureThenRecovery(t *testing.T) {
	workspace := filepath.Join(os.TempDir(), fmt.Sprintf("spore-s4-%d", time.Now().UnixNano()))
	if err := os.MkdirAll(workspace, 0o755); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(workspace)

	sessionID := sporecore.SessionID("s4-test")
	recoveredPath := filepath.Join(workspace, "recovered.txt")

	agent := sporecore.NewMockAgent("mock")
	// Call flaky_op (fails recoverably) -> adapt by writing recovered.txt -> final.
	agent.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{toolCall("c1", "flaky_op", `{"reason":"first try"}`)}, usage()))
	agent.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{toolCall("c2", "write_file",
		fmt.Sprintf(`{"path":%q,"content":"flaky_op failed; adapted by writing this file"}`, recoveredPath))}, usage()))
	agent.Push(sporecore.NewFinalResponse("DONE recovered", usage()))

	registry := scenarios.BuildRealToolRegistry(scenarios.S4, sporecore.SessionID("test-session"), nil, nil)
	sandbox := sporecore.AllowAllSandbox{}
	schemas := scenarios.ModelSchemas(registry)

	h := scenarios.BuildScenario(
		agent, registry, sandbox, sporecore.NoopContextManager{},
		sporecore.AlwaysContinuePolicy{}, schemas, nil, 2, nil,
	)

	task := sporecore.NewTask(scenarios.S4.Prompt(), sessionID, reactStrategy(8))
	res := h.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
	if res.Kind != sporecore.RunSuccess {
		t.Fatalf("S4 expected Success, got %v (%s)", res.Kind, res.Reason.Kind)
	}
	if res.Turns < 3 {
		t.Fatalf("S4: flaky -> recover -> done expects >=3 turns, got %d", res.Turns)
	}
	if _, err := os.Stat(recoveredPath); err != nil {
		t.Fatalf("recovery file not written: %v", err)
	}
}

// The harness must NOT hard-halt on the recoverable FailingTool error — flaky_op
// is not annotated always-halt, and its output is a recoverable error.
func TestS4FailingToolIsRecoverableNotAlwaysHalt(t *testing.T) {
	registry := scenarios.BuildRealToolRegistry(scenarios.S4, sporecore.SessionID("test-session"), nil, nil)
	if registry.IsAlwaysHalt("flaky_op") {
		t.Fatal("flaky_op must not be always-halt")
	}
	out := registry
	res, err := out.Dispatch(context.Background(), toolCall("c1", "flaky_op", `{}`), sporecore.AllowAllSandbox{})
	if err != nil {
		t.Fatalf("dispatch error: %v", err)
	}
	if res.Output.Kind != sporecore.ToolOutputError || !res.Output.Recoverable {
		t.Fatalf("flaky_op output = %v recoverable=%v, want recoverable error", res.Output.Kind, res.Output.Recoverable)
	}
}

// ---------------------------------------------------------------------------
// Regression: the task instruction must reach the agent as the first user
// message (issue #57). Unlike MockAgent, which ignores its Context, this agent
// records every assembled Context so we can assert the model actually receives
// the prompt. Backed by the real StandardCompactionAdapter context manager
// (the same wiring a live run uses) — the adapter mirrors session.Messages and
// ignores task on Assemble, so without the harness seeding the instruction the
// captured first-turn context is EMPTY and this test fails (the bug we fix).
// ---------------------------------------------------------------------------

type capturingAgent struct {
	id       sporecore.AgentID
	contexts []sporecore.Context
}

func (a *capturingAgent) Turn(_ context.Context, c sporecore.Context) sporecore.TurnResult {
	a.contexts = append(a.contexts, c)
	return sporecore.NewFinalResponse("DONE", usage())
}

func (a *capturingAgent) ID() sporecore.AgentID { return a.id }

func TestTaskInstructionDeliveredAsFirstUserMessage(t *testing.T) {
	agent := &capturingAgent{id: "capture"}

	// Real compaction-adapter-backed context manager (mirrors session.Messages,
	// ignores task), so only the harness seeding can put the prompt on screen.
	model := sporecore.NewMockModel(sporecore.ProviderInfo{Name: "mock", ModelID: "mock", ContextWindow: 4096})
	cfg := contextmgr.CompactionConfig{Threshold: 0.80, PreserveRecentN: 2, HeadTailTokens: 64, OffloadPath: ".spore/offload", MaxCompactionAttempts: 2}
	rich := contextmgr.NewStandardContextManager(model, nil, cfg)
	cm := contextmgr.NewStandardCompactionAdapter(rich)

	h := scenarios.BuildScenario(
		agent, sporecore.NewScriptedToolRegistry(), sporecore.AllowAllSandbox{}, cm,
		sporecore.AlwaysContinuePolicy{}, nil, nil, 2, nil,
	)

	const instruction = "summarize the quarterly payment report"
	task := sporecore.NewTask(instruction, "seed-test", reactStrategy(4))
	res := h.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
	if res.Kind != sporecore.RunSuccess {
		t.Fatalf("expected Success, got %v (%s)", res.Kind, res.Reason.Kind)
	}

	if len(agent.contexts) == 0 {
		t.Fatal("agent must have been invoked at least once")
	}
	first := agent.contexts[0]
	hasUserInstruction := false
	for _, m := range first.Messages {
		if m.Role == sporecore.RoleUser && m.Content.Type == sporecore.ContentTypeText && m.Content.Text == instruction {
			hasUserInstruction = true
			break
		}
	}
	if !hasUserInstruction {
		t.Fatalf("first-turn context must contain a User message equal to the task instruction; got messages: %+v", first.Messages)
	}
}

// ParseScenarioID round-trips s1..s5 and rejects junk.
func TestParseScenarioID(t *testing.T) {
	for _, in := range []string{"s1", "S2", " s3 ", "S4", "s5"} {
		if _, ok := scenarios.ParseScenarioID(in); !ok {
			t.Fatalf("ParseScenarioID(%q) failed", in)
		}
	}
	if _, ok := scenarios.ParseScenarioID("nope"); ok {
		t.Fatal("ParseScenarioID should reject junk")
	}
}

// ---------------------------------------------------------------------------
// Per-scenario tool catalog: S1/S2 expose exec but NOT bash_command; S5 both.
// ---------------------------------------------------------------------------

func schemaNames(scenario scenarios.ScenarioID) []string {
	reg := scenarios.BuildRealToolRegistry(scenario, sporecore.SessionID("test-session"), nil, nil)
	schemas := scenarios.ModelSchemas(reg)
	names := make([]string, 0, len(schemas))
	for _, s := range schemas {
		names = append(names, s.Name)
	}
	return names
}

func hasName(names []string, want string) bool {
	for _, n := range names {
		if n == want {
			return true
		}
	}
	return false
}

func TestCatalogS1HasExecNotBashCommand(t *testing.T) {
	names := schemaNames(scenarios.S1)
	if !hasName(names, "exec") {
		t.Fatalf("S1 catalog must contain exec, got %v", names)
	}
	if hasName(names, "bash_command") {
		t.Fatalf("S1 catalog must NOT contain bash_command, got %v", names)
	}
}

func TestCatalogS2LacksBashCommand(t *testing.T) {
	names := schemaNames(scenarios.S2)
	if !hasName(names, "exec") {
		t.Fatalf("S2 catalog must contain exec, got %v", names)
	}
	if hasName(names, "bash_command") {
		t.Fatalf("S2 catalog must NOT contain bash_command, got %v", names)
	}
}

func TestCatalogS5HasBashCommand(t *testing.T) {
	names := schemaNames(scenarios.S5)
	if !hasName(names, "exec") {
		t.Fatalf("S5 catalog must contain exec, got %v", names)
	}
	if !hasName(names, "bash_command") {
		t.Fatalf("S5 catalog must contain bash_command, got %v", names)
	}
}

// ---------------------------------------------------------------------------
// S5 — real shell: bash_command pipeline (cat | tr > out) then read_file verify
// ---------------------------------------------------------------------------

func TestS5BashCommandShellPipeline(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix only")
	}
	workspace := filepath.Join(os.TempDir(), fmt.Sprintf("spore-s5-%d", time.Now().UnixNano()))
	if err := os.MkdirAll(workspace, 0o755); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(workspace)

	inPath := filepath.Join(workspace, "input.txt")
	outPath := filepath.Join(workspace, "output.txt")
	if err := os.WriteFile(inPath, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}

	sessionID := sporecore.SessionID("s5-test")
	script := fmt.Sprintf("cat %s | tr a-z A-Z > %s", inPath, outPath)

	agent := sporecore.NewMockAgent("mock")
	// Turn 1: real shell pipeline (literal pipe + redirect) via bash_command.
	agent.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{
		toolCall("c1", "bash_command", fmt.Sprintf(`{"script":%q}`, script))}, usage()))
	// Turn 2: read back output.txt to verify.
	agent.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{
		toolCall("c2", "read_file", fmt.Sprintf(`{"path":%q}`, outPath))}, usage()))
	// Turn 3: final.
	agent.Push(sporecore.NewFinalResponse("DONE", usage()))

	registry := scenarios.BuildRealToolRegistry(scenarios.S5, sporecore.SessionID("test-session"), nil, nil)
	sandbox := sporecore.AllowAllSandbox{}
	schemas := scenarios.ModelSchemas(registry)

	h := scenarios.BuildScenario(
		agent, registry, sandbox, sporecore.NoopContextManager{},
		sporecore.AlwaysContinuePolicy{}, schemas, nil, 2, nil,
	)

	task := sporecore.NewTask(scenarios.S5.Prompt(), sessionID, reactStrategy(8))
	res := h.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
	if res.Kind != sporecore.RunSuccess {
		t.Fatalf("S5 expected Success, got %v (%s)", res.Kind, res.Reason.Kind)
	}
	if res.Turns < 3 {
		t.Fatalf("S5: shell -> read -> done expects >=3 turns, got %d", res.Turns)
	}
	got, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("output.txt not written by shell pipeline: %v", err)
	}
	if string(got) != "HELLO" {
		t.Fatalf("S5 output = %q, want %q", string(got), "HELLO")
	}
}
