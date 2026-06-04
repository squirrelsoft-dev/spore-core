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
// # Staying within small (~128K) context windows
//
// Under PlanExecute, verbose tool output is retained across every plan step, so
// a few searches can overflow a model with a ~128K window (e.g. gemma4:e4b,
// 131072 tokens). Two measures keep this example running cleanly on such models:
//
//   - It DISTILLS web_search results: a conciseWebSearch wrapper trims the
//     verbatim SearXNG JSON (25-40K tokens/call) down to the top 6 results with
//     only title / url / content, so context stays small.
//   - It LOWERS the compaction threshold to 0.45 (compaction at ≈90K tokens
//     instead of the default ≈160K), installed via ContextManager(...), so
//     compaction fires before a 128K-window model overflows.
//
// # Tools (web_search is wrapped to distill its output; the rest are catalogue)
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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
)

const systemPrompt = "You are a research-and-writing agent. Your ONLY capabilities are: " +
	"web_search (find current information online), read_file, and write_file (save your work to " +
	"the workspace). You have NO shell or terminal — you cannot install software, set up projects " +
	"or environments, run/compile/build code, or execute commands. Decompose the goal into " +
	"subtasks that are each achievable with web_search and writing alone; never plan setup, " +
	"installation, or build steps. For each subtask, use web_search to gather current information, " +
	"then synthesize a clear, cited comparison and save the final document with write_file. Act " +
	"using tools — do not answer from memory alone."

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

// conciseWebSearch is a thin wrapper around the built-in web_search tool that
// distills its output.
//
// WHY: the core web_search tool returns the SearXNG JSON body VERBATIM (by
// frozen spec — normalization is out of scope for the core tool). Each search
// yields ~25-30 results, each carrying the full content plus a dozen noise
// fields (thumbnail, engine, score, parsed_url, …) — roughly 25-40K tokens per
// call. Under PlanExecute those dumps are retained across every plan step, so
// three searches alone can overflow a ~128K-window model. This wrapper keeps
// only the top results and the fields the agent actually reads, so the
// conversation context stays small. The model still sees an identical
// web_search tool (same name + schema); only the result is trimmed.
type conciseWebSearch struct {
	inner *tools.WebSearchTool
}

func (w conciseWebSearch) Name() string                { return w.inner.Name() }
func (w conciseWebSearch) IsSubagentTool() bool        { return w.inner.IsSubagentTool() }
func (w conciseWebSearch) MayProduceLargeOutput() bool { return true }

func (w conciseWebSearch) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
	out := w.inner.Execute(ctx, call, sandbox, toolCtx)
	// Distill only successful output; errors and every non-success variant pass
	// through untouched.
	if out.Kind == sporecore.ToolOutputSuccess {
		out.Content = distillSearchResults(out.Content)
	}
	return out
}

var _ sporecore.Tool = conciseWebSearch{}

// distillSearchResults keeps only the top 6 results, and for each only title /
// url / content (content clipped to ~500 chars). It drops all other fields and
// top-level keys (answers, infoboxes, suggestions, unresponsive_engines, …) and
// re-serializes as compact {"results":[...]}. Defensive: if the body is not
// JSON or has no results array, the original string is returned unchanged — we
// never error just because the shape was unexpected.
func distillSearchResults(content string) string {
	var body struct {
		Results []json.RawMessage `json:"results"`
	}
	if err := json.Unmarshal([]byte(content), &body); err != nil || body.Results == nil {
		return content
	}

	type distilled struct {
		Title   string `json:"title"`
		URL     string `json:"url"`
		Content string `json:"content"`
	}
	out := struct {
		Results []distilled `json:"results"`
	}{Results: []distilled{}}

	for i, raw := range body.Results {
		if i >= 6 {
			break
		}
		var r struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		}
		_ = json.Unmarshal(raw, &r) // best-effort: missing fields stay empty
		out.Results = append(out.Results, distilled{
			Title:   r.Title,
			URL:     r.URL,
			Content: clip(r.Content, 500),
		})
	}

	encoded, err := json.Marshal(out)
	if err != nil {
		return content
	}
	return string(encoded)
}

// clip truncates s to at most n runes (not bytes), so multibyte content is not
// split mid-character.
func clip(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}

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

	// Native Ollama tool calling is the default. Pass --structured to opt into
	// constrained-decoding (structured tool calls) for small local models.
	structured := hasFlag("--structured")

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

	// Wrap the raw WebSearchTool in conciseWebSearch so verbose SearXNG JSON is
	// distilled before it enters the conversation (see the struct doc above).
	// Same name + schema, so the model sees an identical web_search tool.
	concise := conciseWebSearch{inner: webSearch}

	// Lower the compaction threshold so it fires at ~0.45 * 200K ≈ 90K tokens,
	// BEFORE a ~128K-window model (e.g. gemma4:e4b) overflows. ConversationalBuilder
	// installs a StandardContextManager with the default compaction config
	// (compaction at 80% of a 200K window ≈ 160K), which is too late here. The
	// context manager gets its OWN raw model instance for summarization turns.
	compaction := contextmgr.DefaultCompactionConfig()
	compaction.Threshold = 0.45
	contextManager := contextmgr.NewStandardCompactionAdapter(
		contextmgr.NewStandardContextManager(
			ollama.WithBaseURL(model, baseURL),
			contextmgr.NullCacheProvider{},
			compaction,
		),
	)

	// Go hooks-wiring asymmetry: the builder has no .Hooks() setter. Build the
	// config, register the reporter on a chain, then set cfg.Hooks before
	// NewStandardHarness. The chain is how the plan becomes visible — there are no
	// plan/subtask stream events.
	cfg := observability.ConversationalBuilder(mi).
		Sandbox(sandbox).
		ContextManager(contextManager).                                                // ← compaction fires ~90K, not ~160K (small windows)
		Tool(tools.StandardTool{Implementation: concise, Schema: webSearch.Schema()}). // ← external API (SearXNG GET/JSON), distilled
		Tool(tools.StandardTools{}.WriteFile()).                                       // ← writes async-comparison.md
		Tool(tools.StandardTools{}.ReadFile()).
		SystemPrompt(systemPrompt).
		// Native Ollama tool calling by default — it exposes the real typed tool
		// schema and works for tool-capable / cloud models (e.g. gemma4:31b-cloud).
		// Pass --structured to enable constrained-decoding (structured tool calls)
		// for small local models (e.g. llama3.2) that need it to emit clean calls.
		WithModelParams(sporecore.ModelParams{StructuredToolCalls: structured}).
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

// hasFlag reports whether the given boolean flag appears in os.Args.
func hasFlag(name string) bool {
	for _, a := range os.Args[1:] {
		if a == name {
			return true
		}
	}
	return false
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
