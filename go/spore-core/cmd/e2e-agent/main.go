// Command e2e-agent is the end-to-end CLI agent + scenario suite (issue #57).
//
// One shared binary that drives the *complete* harness loop against a real local
// model (Ollama) through a HarnessConfig-assembled StandardHarness with real
// tools (read/write/list/bash + a deliberately-failing tool), the
// StandardCompactionAdapter, and the durable-outbox observability provider. The
// scenario is selected by a CLI arg.
//
// # Scenarios
//
//   - s1 — multi-step / multi-tool: read input.txt -> uppercase -> write
//     output.txt -> read back + confirm.
//   - s2 — multi-turn: run twice with the same SessionID, carrying session state
//     across turns.
//   - s3 — live compaction: a seeded small window + long history fires the
//     compaction adapter mid-run; the #57 token-accounting fix lets it compact,
//     continue, and compact again.
//   - s4 — tool failure + recovery: call flaky_op (recoverable error), then write
//     recovered.txt explaining the adaptation.
//
// # Run recipe (live, against a local model + observability stack)
//
//	# 1. Start Ollama and pull a tool-capable model.
//	ollama serve &              # or run the Ollama app
//	ollama pull llama3.2        # default model; passes the #41 capability guard
//
//	# 2. (optional) Start the local observability stack and forward traces.
//	#    See observability/ for the compose stack (Tempo + Loki + Grafana).
//	export SPORE_OTLP_ENDPOINT=http://localhost:4317
//
//	# 3. Run a scenario. Prompt/model/endpoint/workspace come from args+env.
//	go run ./cmd/e2e-agent s1 --model llama3.2
//	go run ./cmd/e2e-agent s2
//	go run ./cmd/e2e-agent s3
//	go run ./cmd/e2e-agent s4
//
//	# 4. Verify the grouped trace in Tempo (the run prints the trace_id):
//	curl -s http://localhost:3200/api/traces/<trace_id> | jq '.batches | length'
//	#    For s3, spot-check a Compaction span appears mid-trace.
//
// Environment variables (all optional):
//   - SPORE_OLLAMA_MODEL     — default model id (overridden by --model).
//   - SPORE_OLLAMA_BASE_URL  — Ollama base url (default http://localhost:11434).
//   - SPORE_OTLP_ENDPOINT    — when set, forward spans to Tempo (issue #50).
//   - SPORE_E2E_WORKSPACE    — workspace root (default: a temp dir per run).
//
// # Offline / hermetic mode
//
// --mock runs the same scenario builders against a scripted MockAgent, requiring
// no Ollama or network. The hermetic CI assertions live in scenarios/
// scenarios_test.go, which drive the same BuildScenario path with mock
// components.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/scenarios"
)

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		usage()
		os.Exit(2)
	}
	scenario, ok := scenarios.ParseScenarioID(args[0])
	if !ok {
		usage()
		os.Exit(2)
	}
	mock := hasFlag(args, "--mock")
	modelID := argValue(args, "--model")
	if modelID == "" {
		modelID = os.Getenv("SPORE_OLLAMA_MODEL")
	}
	if modelID == "" {
		modelID = "llama3.2"
	}

	// Per-run session id so repeated runs don't collide in Tempo.
	stamp := strings.NewReplacer(":", "-", ".", "-").Replace(time.Now().UTC().Format("2006-01-02T15:04:05.000Z07:00"))
	sessionID := sporecore.SessionID(fmt.Sprintf("e2e-%s-%s", scenario, stamp))

	// Workspace: a real directory the file tools operate inside.
	workspace := os.Getenv("SPORE_E2E_WORKSPACE")
	if workspace == "" {
		workspace = filepath.Join(os.TempDir(), "spore-e2e-"+string(sessionID))
	}
	if err := os.MkdirAll(workspace, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "create workspace: %v\n", err)
		os.Exit(1)
	}
	prepareWorkspace(scenario, workspace)
	fmt.Printf("workspace  : %s\n", workspace)

	endpoint := os.Getenv("SPORE_OTLP_ENDPOINT")
	if strings.TrimSpace(endpoint) == "" {
		fmt.Fprintln(os.Stderr, "note: SPORE_OTLP_ENDPOINT is unset — writing JSONL only (no Tempo forwarding).")
	}

	var res sporecore.RunResult
	if mock {
		res = runMock(scenario, sessionID)
	} else {
		res = runLive(scenario, sessionID, modelID, workspace)
	}

	switch res.Kind {
	case sporecore.RunSuccess:
		fmt.Printf("result     : Success (%d turns)\n", res.Turns)
		fmt.Printf("output     : %q\n", res.Output)
	default:
		fmt.Fprintf(os.Stderr, "result     : %v (%s)\n", res.Kind, res.Reason.Kind)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: e2e-agent <s1|s2|s3|s4|s5> [--model <id>] [--mock]")
}

// runLive runs the scenario against a live Ollama model.
func runLive(scenario scenarios.ScenarioID, sessionID sporecore.SessionID, modelID, workspace string) sporecore.RunResult {
	baseURL := os.Getenv("SPORE_OLLAMA_BASE_URL")
	if baseURL == "" {
		baseURL = ollama.DefaultBaseURL
	}
	model := ollama.WithBaseURL(modelID, baseURL)
	agent := sporecore.NewModelAgent("e2e-agent", model)

	// No RunStore is configured for the e2e binary, so the standalone
	// task_list tool runs with the no-op storage default (persists nothing
	// across processes — the retired .spore sandbox path is gone, #75).
	registry := scenarios.BuildRealToolRegistry(scenario, sessionID, nil)
	sandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: workspace})
	if err != nil {
		fmt.Fprintf(os.Stderr, "build sandbox: %v\n", err)
		os.Exit(1)
	}
	schemas := scenarios.ModelSchemas(registry)

	// Compaction-capable context manager (small window for S3).
	windowLimit := uint32(128_000)
	if scenario == scenarios.S3 {
		windowLimit = 200
	}
	cfg := contextmgr.CompactionConfig{
		Threshold:             0.80,
		PreserveRecentN:       2,
		HeadTailTokens:        64,
		OffloadPath:           filepath.Join(workspace, ".spore", "offload"),
		MaxCompactionAttempts: 2,
	}
	rich := contextmgr.NewStandardContextManager(model, nil, cfg)
	cm := contextmgr.NewStandardCompactionAdapter(rich)

	// Durable-outbox observability (honors SPORE_OTLP_ENDPOINT like the smoke).
	root := ".spore"
	if wd, err := os.Getwd(); err == nil {
		root = filepath.Join(wd, ".spore")
	}
	provider := observability.NewOutboxObservabilityProvider(observability.NewOutboxConfig(root))
	observer := observability.NewHarnessObserver(provider, observability.DefaultPricing())

	h := scenarios.BuildScenario(
		agent, registry, sandbox, cm,
		sporecore.AlwaysContinuePolicy{}, schemas,
		contextmgr.NewKeyTermVerifier(), cfg.MaxCompactionAttempts, observer,
	)

	return runScenario(scenario, h, sessionID, windowLimit)
}

// runScenario drives the scenario, including the S2 multi-turn and S3 seed.
func runScenario(scenario scenarios.ScenarioID, h sporecore.Harness, sessionID sporecore.SessionID, windowLimit uint32) sporecore.RunResult {
	ctx := context.Background()
	strategy := sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, MaxIterations: 8}

	switch scenario {
	case scenarios.S2:
		// Multi-turn: run twice with the same SessionID, carrying state.
		task1 := sporecore.NewTask(scenario.Prompt(), sessionID, strategy)
		r1 := h.Run(ctx, sporecore.NewHarnessRunOptions(task1))
		if r1.Kind != sporecore.RunSuccess {
			fmt.Fprintf(os.Stderr, "S2 turn 1 did not succeed: %v\n", r1.Kind)
			return r1
		}
		state := sporecore.SessionState{}
		task2 := sporecore.NewTask(
			"Add a second TODO item to notes.md that references the first item you wrote. "+
				"Use write_file with append=true. Reply DONE when finished.",
			sessionID, strategy)
		opts := sporecore.NewHarnessRunOptions(task2)
		opts.SessionState = &state
		return h.Run(ctx, opts)
	case scenarios.S3:
		task := sporecore.NewTask(scenario.Prompt(), sessionID, strategy)
		var state sporecore.SessionState
		scenarios.SeedCompactionState(&state, "deploy the payment service", sessionID, task.ID,
			windowLimit, uint32(float32(windowLimit)*0.82), 12)
		opts := sporecore.NewHarnessRunOptions(task)
		opts.SessionState = &state
		return h.Run(ctx, opts)
	default:
		task := sporecore.NewTask(scenario.Prompt(), sessionID, strategy)
		return h.Run(ctx, sporecore.NewHarnessRunOptions(task))
	}
}

// prepareWorkspace seeds scenario-specific workspace files.
func prepareWorkspace(scenario scenarios.ScenarioID, workspace string) {
	if scenario == scenarios.S1 || scenario == scenarios.S5 {
		_ = os.WriteFile(filepath.Join(workspace, "input.txt"),
			[]byte("hello from the spore harness end to end scenario\n"), 0o644)
	}
}

// runMock runs the same scenario builders against a scripted MockAgent, requiring
// no Ollama or network. Mirrors the hermetic test wiring so --mock exercises the
// same BuildScenario path offline.
func runMock(scenario scenarios.ScenarioID, sessionID sporecore.SessionID) sporecore.RunResult {
	agent := sporecore.NewMockAgent("mock")
	u := sporecore.TokenUsage{InputTokens: 10, OutputTokens: 5}
	agent.Push(sporecore.NewToolCallRequested([]sporecore.ToolCall{
		{ID: "c1", Name: "read_file", Input: json.RawMessage(`{"path":"input.txt"}`)},
	}, u))
	agent.Push(sporecore.NewFinalResponse("DONE", u))

	tools := sporecore.NewScriptedToolRegistry()
	tools.Push(sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "contents"})

	h := scenarios.BuildScenario(
		agent, tools, sporecore.AllowAllSandbox{}, sporecore.NoopContextManager{},
		sporecore.AlwaysContinuePolicy{}, nil, nil, 2, nil,
	)
	task := sporecore.NewTask(scenario.Prompt(), sessionID, sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, MaxIterations: 5})
	return h.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
}

func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

func argValue(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}
