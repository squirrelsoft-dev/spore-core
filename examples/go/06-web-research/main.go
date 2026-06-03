// spore-core example 06 — web research with an external API tool.
//
// This is the first example whose tools reach *outside the process* to a
// third-party HTTP service. The whole point is that this changes nothing about
// the harness: an external API is just another tool. It drops into the same
// ConversationalBuilder, the same ReAct loop, and the same
// WorkspaceScopedSandbox you saw in 04.
//
// # The tools (all from the built-in catalogue — no custom impl Tool)
//
//   - web_search — built via tools.NewWebSearchToolFromConfig with a GET config
//     (Method=GET, QueryParam="q"). The tool issues GET <endpoint>?q=<query>,
//     preserving any query string already on the endpoint (e.g. ?format=json), and
//     returns the response body to the agent verbatim. The endpoint comes from
//     SPORE_WEB_SEARCH_ENDPOINT — point it at a SearXNG JSON API (see the README +
//     .env.example).
//   - write_file — StandardTools{}.WriteFile(). The agent writes its synthesized,
//     cited answer to answer.md.
//   - read_file — StandardTools{}.ReadFile(). Lets the agent re-read what it wrote
//     (e.g. to verify or revise the answer).
//
// # The only change from 04
//
// 04 registered the local filesystem catalogue (.Tools(CodingSet()...)); 06 swaps
// in an external web_search tool and keeps two file tools so the agent can persist
// its answer. Same conversational harness, same ReAct loop, same
// WorkspaceScopedSandbox scoped to this example's workspace/ dir so write_file
// cannot escape it. 04 wrote SUMMARY.md; 06 writes answer.md.
//
// # The agent writes the answer, not main
//
// Just like 04, the file write happens INSIDE the ReAct loop via the catalogue
// write_file tool, backed by the WorkspaceScopedSandbox. main never writes
// answer.md itself — the agent does, and the sandbox keeps it from escaping
// workspace/.
//
// # Run it
//
//	ollama serve &
//	ollama pull llama3.2
//	export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"   # SearXNG JSON API
//	go run .
//	go run . --prompt "What are the current options for running WebAssembly outside the browser? Cite sources and write answer.md."
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

const systemPrompt = "You are a web-research agent. Use web_search to find current " +
	"information, synthesize what you learn into a clear answer, and ALWAYS cite the sources " +
	"you used. Write the final answer to answer.md using write_file. Act using tools — do not " +
	"answer from memory alone."

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
		return fmt.Errorf("SPORE_WEB_SEARCH_ENDPOINT is not set.\n" +
			"Set it to a SearXNG JSON API endpoint, e.g. " +
			"\"http://localhost:8888/search?format=json\".\n" +
			"See .env.example and the README.")
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
		return fmt.Errorf("failed to construct web_search tool: %w", err)
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
		// A TIMELESS research question: the answer evolves over time but the
		// question stays interesting and is not tied to a single news event.
		prompt = "What is the current recommended way to install Rust on macOS, and what are the " +
			"main alternatives? Search the web, synthesize the options, cite your sources, and " +
			"write the answer to answer.md."
	}

	// Same conversational harness + WorkspaceScopedSandbox as 04. The ONLY
	// substantive change is the tool set: web_search (external API) composes with
	// write_file / read_file in one builder chain. .Tool() and .Tools() push into
	// the same registry with last-wins upsert by name.
	mi := ollama.WithBaseURL(model, baseURL)
	sandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: workspaceRoot})
	if err != nil {
		return fmt.Errorf("failed to create sandbox: %w", err)
	}
	harness := observability.ConversationalBuilder(mi).
		Sandbox(sandbox).
		Tool(tools.StandardTool{Implementation: webSearch, Schema: webSearch.Schema()}). // ← external API (SearXNG GET/JSON)
		Tool(tools.StandardTools{}.WriteFile()).                                         // ← writes answer.md
		Tool(tools.StandardTools{}.ReadFile()).
		SystemPrompt(systemPrompt).
		Build()

	task := sporecore.NewTask(prompt, sporecore.NewSessionID(), sporecore.LoopStrategy{
		Kind:          sporecore.StrategyReAct,
		MaxIterations: 10,
	})

	// Print each turn (Think) and each tool call + result (Act / Observe). The
	// search queries and result snippets show up here because web_search dispatches
	// through the harness like any other catalogue tool.
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
	fmt.Printf("prompt   : %s\n\n", prompt)

	result := harness.Run(context.Background(), options)
	switch result.Kind {
	case sporecore.RunSuccess:
		fmt.Printf("\nanswer (%d turn(s)): %s\n", result.Turns, result.Output)
		answer := filepath.Join(workspaceRoot, "answer.md")
		if _, statErr := os.Stat(answer); statErr == nil {
			fmt.Printf("\nanswer.md now exists on disk: %s\n", answer)
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
