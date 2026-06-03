// spore-core example 08 — multi-step goal decomposition with PlanExecute.
//
// This is the first example to swap the LOOP STRATEGY. Everything else — the
// ConversationalBuilder, the WorkspaceScopedSandbox, and the tool set
// (web_search + write_file + read_file, identical to 06) — is held constant. The
// ONLY substantive change is one line on the Task:
//
//	// 06 — react step-by-step:
//	LoopStrategy{Kind: StrategyReAct, MaxIterations: 10}
//	// 08 — decompose the goal first, then execute each subtask:
//	LoopStrategy{Kind: StrategyPlanExecute}
//
// With PlanExecute, the harness runs one constrained planner turn FIRST: the
// model must return strict JSON {"tasks": [...], "rationale": ...}. That plan is
// captured into a PlanArtifact, surfaced, then each subtask is run in a bounded
// sub-loop. The turn budget is divided across subtasks (per-task cap =
// remaining_turns / remaining_tasks), so we set a generous MaxTurns.
//
// # Surfacing the plan — via lifecycle HOOKS, not stream events
//
// There are no plan/subtask STREAM events; the plan is visible through the hook
// chain. We register a PlanExecuteReporter (a Hook) on two events:
//
//   - OnPlanCreated fires post-capture / pre-execute — we print a "── plan ──"
//     banner: the rationale, then the numbered tasks (HookContext.Plan).
//   - OnTaskAdvance fires before each subtask — we print "[i/N] <instruction>"
//     using HookContext.Task.Instruction, HookContext.TaskIndex (0-based), and
//     HookContext.TotalTasks.
//
// # The Go hooks-wiring asymmetry
//
// Unlike the Rust/TS/Python builders, the Go observability HarnessBuilder has no
// .Hooks() setter. The Go path is: build the config with BuildConfig(), set
// cfg.Hooks to the chain that carries the reporter, then NewStandardHarness(cfg).
// NewStandardHarness only defaults Hooks when nil and auto-registers the Ralph
// stop hook onto whatever chain is set, so registering the reporter on our chain
// first is all that is needed.
//
// # Tools (all from the built-in catalogue — no custom impl Tool)
//
//   - web_search — built via tools.NewWebSearchToolFromConfig with a GET config
//     (Method=GET, QueryParam="q"). The tool issues GET <endpoint>?q=<query>,
//     preserving any query string already on the endpoint (e.g. ?format=json), and
//     returns the response body to the agent verbatim. The endpoint comes from
//     SPORE_WEB_SEARCH_ENDPOINT — point it at a SearXNG JSON API.
//   - write_file — StandardTools{}.WriteFile(). The agent writes the synthesized,
//     cited comparison to async-comparison.md.
//   - read_file — StandardTools{}.ReadFile(). Lets the agent re-read what it wrote.
//
// # The agent writes the file, not main
//
// Just like 04 and 06, the file write happens INSIDE the loop via the catalogue
// write_file tool, backed by the WorkspaceScopedSandbox. main never writes
// async-comparison.md itself — the agent does, and the sandbox keeps it inside
// workspace/.
//
// # Run it
//
//	ollama serve &
//	ollama pull llama3.2
//	export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"   # SearXNG JSON API
//	go run .
//	go run . --prompt "Compare three Rust web frameworks (axum, actix-web, rocket) on performance, ergonomics, and ecosystem; cite sources and save to async-comparison.md."
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
)

const systemPrompt = "You are a planning research agent. Decompose the goal into clear " +
	"subtasks. For each subtask, use web_search to find current information, then synthesize a " +
	"clear, cited comparison and save the final document with write_file. Act using tools — do " +
	"not answer from memory alone."

// planMaxTurns is a generous turn budget. PlanExecute divides the budget across
// subtasks (per-task cap = remaining_turns / remaining_tasks), so a stingy cap
// starves later steps — e.g. an 8-step plan at 64 turns gives each subtask ~8.
const planMaxTurns uint32 = 64

// planExecuteReporter is a lifecycle hook that prints the PlanExecute plan and
// each subtask as it runs.
//
// OnPlanCreated fires once, after the planner turn captures the plan and before
// any subtask executes — the money moment for PlanExecute. OnTaskAdvance fires
// before each subtask. Both are sync events. This hook only observes; it always
// returns Continue.
type planExecuteReporter struct{}

func (planExecuteReporter) Handle(_ context.Context, hctx *sporecore.HookContext) (sporecore.HookDecision, error) {
	switch hctx.Event {
	case sporecore.HookEventOnPlanCreated:
		fmt.Println("\n── plan ──")
		if hctx.Plan != nil {
			if r := strings.TrimSpace(hctx.Plan.Rationale); r != "" {
				fmt.Printf("rationale: %s\n", r)
			}
			for i, t := range hctx.Plan.Tasks {
				fmt.Printf("  %d. %s\n", i+1, t)
			}
		}
		fmt.Println("──────────")
	case sporecore.HookEventOnTaskAdvance:
		instruction := ""
		if hctx.Task != nil {
			instruction = hctx.Task.Instruction
		}
		fmt.Printf("[%d/%d] %s\n", hctx.TaskIndex+1, hctx.TotalTasks, instruction)
	}
	return sporecore.Continue(), nil
}

func (planExecuteReporter) Events() []sporecore.HookEvent {
	return []sporecore.HookEvent{
		sporecore.HookEventOnPlanCreated,
		sporecore.HookEventOnTaskAdvance,
	}
}

func (planExecuteReporter) Name() string { return "plan-execute-reporter" }

func (planExecuteReporter) SyncMode() sporecore.HookSync { return sporecore.HookSyncSync }

var _ sporecore.Hook = planExecuteReporter{}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "\n%s\n", err)
		os.Exit(1)
	}
}

func run() error {
	model := flagValue("--model")
	if model == "" {
		if env := os.Getenv("SPORE_OLLAMA_MODEL"); env != "" {
			model = env
		} else {
			model = "llama3.2"
		}
	}
	baseURL := os.Getenv("SPORE_OLLAMA_BASE_URL")
	if baseURL == "" {
		baseURL = ollama.DefaultBaseURL
	}

	// The search backend endpoint. web_search issues GET <endpoint>?q=<query> and
	// returns the JSON body to the agent. There is no live backend in spore-core,
	// so you must supply one — a self-hosted SearXNG JSON API works out of the box
	// (?format=json). The GET path preserves any query string already on the
	// endpoint, so SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
	// becomes GET /search?format=json&q=<query>. See README.
	endpoint := strings.TrimSpace(os.Getenv("SPORE_WEB_SEARCH_ENDPOINT"))
	if endpoint == "" {
		fmt.Fprintln(os.Stderr, "SPORE_WEB_SEARCH_ENDPOINT is not set.\n"+
			"Set it to a SearXNG JSON API endpoint, e.g. "+
			"\"http://localhost:8888/search?format=json\".\n"+
			"See .env.example and the README.")
		os.Exit(2)
	}

	// Build the web_search tool with a GET config pointed at the SearXNG JSON API:
	// the query is sent as ?q=<query>, preserving any existing query string on the
	// endpoint. No auth is needed for a local SearXNG instance.
	webSearch, err := tools.NewWebSearchToolFromConfig(tools.WebSearchConfig{
		Endpoint:       endpoint,
		Method:         tools.SearchMethodGet,
		QueryParam:     "q",
		AuthHeaders:    []tools.AuthHeader{},
		BodyAuthParams: []tools.BodyAuthParam{},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to construct web_search tool: %s\n", err)
		os.Exit(1)
	}

	// The agent operates inside this example's workspace/ directory. Resolve it
	// relative to this source file so `go run .` works from anywhere, create it,
	// and make it absolute — the sandbox requires a canonical, existing root.
	workspaceRoot, err := workspaceDir()
	if err != nil {
		return err
	}

	prompt := flagValue("--prompt")
	if prompt == "" {
		// A multi-step goal that benefits from upfront decomposition: search each
		// runtime, synthesize a comparison, then write the file.
		prompt = "Research the Rust async ecosystem, write a comparison of tokio vs async-std " +
			"vs smol covering performance, ecosystem maturity, and use cases, and save it to " +
			"async-comparison.md."
	}

	// Same conversational harness + WorkspaceScopedSandbox + tool set as 06. The
	// ONLY substantive change vs 06 is the LoopStrategy on the Task below.
	mi := ollama.WithBaseURL(model, baseURL)
	sandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: workspaceRoot})
	if err != nil {
		return fmt.Errorf("failed to create sandbox: %w", err)
	}

	// Go hooks-wiring asymmetry: the builder has no .Hooks() setter. Build the
	// config, register the reporter on a chain, then set cfg.Hooks before
	// NewStandardHarness. The chain is how the plan becomes visible — there are no
	// plan/subtask stream events.
	cfg := observability.ConversationalBuilder(mi).
		Sandbox(sandbox).
		Tool(tools.StandardTool{Implementation: webSearch, Schema: webSearch.Schema()}). // ← external API (SearXNG GET/JSON)
		Tool(tools.StandardTools{}.WriteFile()).                                         // ← writes async-comparison.md
		Tool(tools.StandardTools{}.ReadFile()).
		SystemPrompt(systemPrompt).
		BuildConfig()

	chain := sporecore.NewStandardHookChain()
	if err := chain.Register(planExecuteReporter{}); err != nil {
		return fmt.Errorf("failed to register plan reporter: %w", err)
	}
	cfg.Hooks = chain
	harness := sporecore.NewStandardHarness(cfg)

	// THE ONE-LINE SWAP. 06 used LoopStrategy{Kind: StrategyReAct, MaxIterations:
	// 10}; here we decompose first via PlanExecute. The turn budget is divided
	// across subtasks, so we give it generous headroom.
	maxTurns := planMaxTurns
	task := sporecore.NewTask(prompt, sporecore.NewSessionID(), sporecore.LoopStrategy{
		Kind: sporecore.StrategyPlanExecute,
	}).WithBudget(sporecore.BudgetLimits{MaxTurns: &maxTurns})

	// Print each turn (Think) and each tool call + result (Act / Observe). This is
	// most useful for the plan-phase turn; Go suppresses the subtask sub-loop
	// stream, so the hooks above are the portable view of execution.
	options := sporecore.NewHarnessRunOptions(task)
	options.OnStream = func(ev sporecore.HarnessStreamEvent) {
		switch ev.Kind {
		case sporecore.HarnessStreamTurnStart:
			fmt.Printf("think  · turn %d\n", ev.Turn)
		case sporecore.HarnessStreamToolCall:
			fmt.Printf("    act    → %s(%s)\n", ev.Name, string(ev.Args))
		case sporecore.HarnessStreamToolResult:
			tag := "obs "
			if ev.IsError {
				tag = "obs(err)"
			}
			fmt.Printf("    %s→ %s\n", tag, truncate(ev.ResultContent, 200))
		}
	}

	fmt.Printf("model    : %s\n", model)
	fmt.Printf("endpoint : %s\n", endpoint)
	fmt.Printf("workspace: %s\n", workspaceRoot)
	fmt.Printf("strategy : PlanExecute (06 used ReAct)\n")
	fmt.Printf("prompt   : %s\n\n", prompt)

	result := harness.Run(context.Background(), options)
	switch result.Kind {
	case sporecore.RunSuccess:
		fmt.Printf("\nanswer (%d turn(s)): %s\n", result.Turns, result.Output)
		doc := filepath.Join(workspaceRoot, "async-comparison.md")
		if _, statErr := os.Stat(doc); statErr == nil {
			fmt.Printf("\nasync-comparison.md now exists on disk: %s\n", doc)
		}
		return nil
	default:
		return fmt.Errorf("run did not succeed: %s %+v", result.Kind, result.Reason)
	}
}

// workspaceDir returns the absolute path to this example's workspace/ dir,
// creating it if needed. It resolves relative to this source file (so `go run .`
// works from anywhere), falling back to the current working directory.
func workspaceDir() (string, error) {
	dir := "workspace"
	if _, file, _, ok := runtime.Caller(0); ok {
		dir = filepath.Join(filepath.Dir(file), "workspace")
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", fmt.Errorf("failed to resolve workspace dir: %w", err)
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return "", fmt.Errorf("failed to create workspace dir %s: %w", abs, err)
	}
	return abs, nil
}

// flagValue returns the value following the given flag in os.Args, or "".
func flagValue(flag string) string {
	args := os.Args[1:]
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

// truncate keeps observe lines readable — search results can be long.
func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}
