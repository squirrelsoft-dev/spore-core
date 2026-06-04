// spore-core example 11 — multi-agent composition: agents as tools.
//
// # The thesis: agents are composable
//
// The harness does not care whether a "tool" dispatches to a function or to
// *another agent*. This example builds THREE agents and wires two of them into
// the third as ordinary tools.
//
// # The three agents
//
//   - research worker — a conversational harness with exactly one tool,
//     web_search (the SearXNG-backed tool from example 06). Given an instruction
//     string, it searches the web and returns raw, cited findings as its output
//     string.
//   - writing worker — a conversational harness with NO tools. Given the research
//     findings as its instruction, it formats them into a polished markdown
//     report and returns that markdown as its output string. It never touches the
//     network — it only shapes prose.
//   - orchestrator — a conversational harness whose "tools" are the two workers
//     (wrapped as tools.SubagentTool) plus write_file. It plans the job, calls
//     research_worker, hands that output to writing_worker, then writes the final
//     markdown to workspace/report.md.
//
// Three agents, two handoffs (research → writing, writing → report.md), one
// output.
//
// # The agent-as-tool mechanism
//
// Each worker is a fully-built child sporecore.Harness wrapped in a
// tools.SubagentTool. SubagentTool implements sporecore.Tool: when the
// orchestrator emits a tool call for research_worker, the tool reads a single
// `instruction` string from the call, runs the child harness on a fresh Task,
// and returns the child's final output string as the tool result. The
// orchestrator cannot tell — and does not need to know — that the "tool" behind
// research_worker is an entire agent with its own loop, its own model, and its
// own web-search tool.
//
// We register each worker on the orchestrator's builder the same way example 06
// registers web_search: build a sporecore.StandardTool from the SubagentTool plus
// a RegistryToolSchema advertising the { instruction: string } input, then
// .Tool(...) it.
//
// # Why this keeps the orchestrator's context clean
//
// Both workers use tools.Isolated context sharing: each runs in a brand-new
// session with NO shared mutable state with the orchestrator or with each other.
// The research worker may burn a dozen internal turns issuing search queries and
// sifting noisy JSON — but the ONLY thing that crosses back into the
// orchestrator's context is the worker's final output string. The orchestrator
// never sees the worker's intermediate turns, failed searches, or raw result
// blobs. The noisy work is encapsulated; the orchestrator's context stays small
// and on-topic. This is the whole reason to delegate to a subagent rather than
// inline the work.
//
// A direct, visible consequence: the child's internal turns do NOT stream up
// through the parent. The orchestrator's stream only shows the ToolCall to
// research_worker and the ToolResult coming back — which is exactly the agent
// boundary we print. The invisibility of the child's turns is not a limitation;
// it IS the context isolation, made observable.
//
// # The strategy split: PlanExecute at the top, ReAct inside
//
// The orchestrator runs LoopStrategy{Kind: StrategyPlanExecute}: it decomposes
// the job ("research, then write, then save") into subtasks up front and executes
// them in order — natural for a coordinator. Each worker, by contrast, runs ReAct
// internally. (The ReAct loop is hardcoded inside SubagentTool; a subagent always
// runs its child as ReAct.) So the two layers use two different loop strategies,
// each fit to its level: deliberate planning at the orchestrator, step-by-step
// tool use inside the workers.
//
// # Agent boundaries in stdout
//
// The point of this example is legibility: you should be able to read stdout and
// see which agent is acting, what it received, and what it returned. The
// orchestrator's stream fires a ToolCall{Name, Args} and a ToolResult for each
// worker dispatch — we turn those into a boxed banner:
//
//	┌─ orchestrator → research_worker
//	│  received: <instruction>
//	└─ research_worker → orchestrator
//	   returned: <truncated findings>
//
// Go wiring note: the ToolResult stream event carries only a CallID (no tool
// name), so we remember which CallID belonged to which tool when the ToolCall
// fires, then look it up on the result to label the closing half of the boundary.
//
// # Run it
//
//	ollama serve &
//	ollama pull llama3.2
//	export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"   # SearXNG JSON API
//	go run .
//	go run . --topic "the history of the TCP/IP protocol suite"
//	go run . --model qwen2.5-coder:7b
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
)

// workerTimeout is the per-worker wall-clock cap. A worker can burn many internal
// ReAct turns; this bounds how long the orchestrator will wait on any single
// delegation.
const workerTimeout = 180 * time.Second

// maxOrchestratorTurns is the orchestrator's turn budget. PlanExecute divides the
// budget across subtasks, and each worker dispatch may itself be slow, so give it
// generous headroom.
const maxOrchestratorTurns uint32 = 32

const researchPrompt = "You are a research worker. Use the web_search tool to gather " +
	"current, factual information on the topic you are given. Issue focused queries, read the " +
	"results, and return a concise set of findings as plain text — key facts, figures, and " +
	"definitions — each followed by the source URL it came from. Do NOT format a report; just " +
	"return the raw, cited findings. Act using web_search — do not answer from memory alone."

const writingPrompt = "You are a writing worker. You will be given a set of raw, cited " +
	"research findings. Turn them into a polished markdown report: a top-level `# ` title, a " +
	"short intro, well-organized `## ` sections, and a `## Sources` list preserving the URLs " +
	"from the findings. Return ONLY the markdown of the report — no preamble, no commentary. " +
	"You have no tools; produce the report directly as your final answer."

const orchestratorPrompt = "You are an orchestrator. You coordinate two worker agents, " +
	"each exposed to you as a tool. Your plan is always the same three steps: (1) call " +
	"`research_worker` with an `instruction` describing the topic to research; (2) call " +
	"`writing_worker` with an `instruction` that is the EXACT findings text returned by the " +
	"research worker, asking it to format a polished markdown report; (3) call `write_file` to " +
	"save the writing worker's markdown verbatim to `report.md`. Do the research and writing " +
	"by delegating to the workers — never do it yourself — and always finish by writing report.md."

// instructionSchema is the single-parameter input schema every worker tool
// advertises: the orchestrator passes one `instruction` string, which
// SubagentTool forwards to the child harness as its task. Matches the schema
// SubagentTool reads.
func instructionSchema() json.RawMessage {
	return json.RawMessage(`{
  "type": "object",
  "properties": {
    "instruction": {
      "type": "string",
      "description": "The full instruction / task for the worker agent."
    }
  },
  "required": ["instruction"]
}`)
}

// buildWebSearch builds the SearXNG-backed web_search catalogue tool (identical
// wiring to example 06). Only the research worker gets this.
func buildWebSearch(endpoint string) (sporecore.StandardTool, error) {
	tool, err := tools.NewWebSearchToolFromConfig(tools.WebSearchConfig{
		Endpoint:       endpoint,
		Method:         tools.SearchMethodGet,
		QueryParam:     "q",
		AuthHeaders:    []tools.AuthHeader{},
		BodyAuthParams: []tools.BodyAuthParam{},
	})
	if err != nil {
		return sporecore.StandardTool{}, fmt.Errorf("failed to construct web_search tool: %w", err)
	}
	return tools.StandardTool{Implementation: tool, Schema: tool.Schema()}, nil
}

// buildResearchHarness builds the research worker: a child harness whose only
// tool is web_search. Each agent gets its OWN fresh model instance — the workers
// are genuinely independent and do not share a model object with the
// orchestrator.
func buildResearchHarness(model, baseURL, endpoint string) (sporecore.Harness, error) {
	webSearch, err := buildWebSearch(endpoint)
	if err != nil {
		return nil, err
	}
	mi := ollama.WithBaseURL(model, baseURL)
	harness := observability.ConversationalBuilder(mi).
		Tool(webSearch).
		SystemPrompt(researchPrompt).
		Build()
	return harness, nil
}

// buildWritingHarness builds the writing worker: a child harness with NO tools —
// it formats prose and returns the report as its final answer.
func buildWritingHarness(model, baseURL string) sporecore.Harness {
	mi := ollama.WithBaseURL(model, baseURL)
	return observability.ConversationalBuilder(mi).
		SystemPrompt(writingPrompt).
		Build()
}

// buildWorkerTool wraps a child harness as a SubagentTool and bundles it into a
// StandardTool the orchestrator can register — exactly how example 06 wraps
// web_search, only the "implementation" here is an entire agent.
func buildWorkerTool(name, description string, child sporecore.Harness) (sporecore.StandardTool, error) {
	// childRegistry is used ONLY for the depth-1 HasSubagentTools() check. The
	// workers have no subagent tools of their own, so a fresh empty registry passes
	// trivially. The child's REAL tools were wired on its builder above.
	emptyChildRegistry := sporecore.NewStandardToolRegistry()
	subagent, err := tools.NewSubagentTool(
		name,
		description,
		instructionSchema(),
		workerTimeout,
		tools.Isolated{},
		child,
		emptyChildRegistry,
	)
	if err != nil {
		return sporecore.StandardTool{}, fmt.Errorf("failed to build subagent tool %q: %w", name, err)
	}

	return tools.StandardTool{
		Implementation: subagent,
		Schema: sporecore.RegistryToolSchema{
			Name:        name,
			Description: description,
			Parameters:  instructionSchema(),
			// OpenWorld: a subagent reaches outside the process (it runs a whole
			// agent, and the research worker hits the network), so it is not a closed,
			// read-only computation.
			Annotations: sporecore.ToolAnnotations{OpenWorld: true},
		},
	}, nil
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

	// Same required search backend as example 06 — only the research worker uses
	// it, but the orchestrator cannot do its job without it, so we fail fast here
	// with the same exit code 2 example 06 uses.
	endpoint := strings.TrimSpace(os.Getenv("SPORE_WEB_SEARCH_ENDPOINT"))
	if endpoint == "" {
		fmt.Fprintln(os.Stderr, "SPORE_WEB_SEARCH_ENDPOINT is not set.\n"+
			"Set it to a SearXNG JSON endpoint, e.g. "+
			"\"http://localhost:8888/search?format=json\" — the query is appended as "+
			"`&q=<query>`.\n"+
			"See .env.example and the README.")
		os.Exit(2)
	}

	// The orchestrator operates inside this example's workspace/ directory; it
	// writes the final report there. Resolve it relative to this source file so
	// `go run .` works from anywhere, create it, and make it absolute — the sandbox
	// requires a canonical, existing root.
	workspaceRoot, err := workspaceDir()
	if err != nil {
		return err
	}

	topic := flagValue("--topic")
	if topic == "" {
		// A TIMELESS, encyclopedic subject so web-search results stay stable and
		// useful across runs (per the issue: keep the topic generic). Matches the
		// Rust reference so all four cores demonstrate the same thing.
		topic = "the history and core ideas of the Rust programming language"
	}
	prompt := fmt.Sprintf(
		"Research %s and produce a polished markdown report saved to report.md. "+
			"Delegate the research to research_worker and the writing to writing_worker.",
		topic,
	)

	// ---- Build the two workers, then wrap them as orchestrator tools --------
	researchChild, err := buildResearchHarness(model, baseURL, endpoint)
	if err != nil {
		return err
	}
	writingChild := buildWritingHarness(model, baseURL)

	researchTool, err := buildWorkerTool(
		"research_worker",
		"Delegate to the research agent: pass an `instruction` describing a topic; it "+
			"web-searches and returns concise, cited findings as text.",
		researchChild,
	)
	if err != nil {
		return err
	}
	writingTool, err := buildWorkerTool(
		"writing_worker",
		"Delegate to the writing agent: pass an `instruction` containing research findings; "+
			"it returns a polished markdown report.",
		writingChild,
	)
	if err != nil {
		return err
	}

	// ---- Build the orchestrator: workers-as-tools + write_file --------------
	orchestratorModel := ollama.WithBaseURL(model, baseURL)
	sandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: workspaceRoot})
	if err != nil {
		return fmt.Errorf("failed to create sandbox: %w", err)
	}
	orchestrator := observability.ConversationalBuilder(orchestratorModel).
		Sandbox(sandbox).
		Tool(researchTool).                      // ← agent-as-tool (research)
		Tool(writingTool).                       // ← agent-as-tool (writing)
		Tool(tools.StandardTools{}.WriteFile()). // ← orchestrator saves report.md itself
		SystemPrompt(orchestratorPrompt).
		Build()

	// The orchestrator plans the three steps up front via PlanExecute, then
	// executes them. The turn budget is divided across subtasks, so give it
	// generous headroom — each worker dispatch may itself be slow.
	maxTurns := maxOrchestratorTurns
	task := sporecore.NewTask(prompt, sporecore.NewSessionID(), sporecore.LoopStrategy{
		Kind: sporecore.StrategyPlanExecute,
	}).WithBudget(sporecore.BudgetLimits{MaxTurns: &maxTurns})

	// The orchestrator's stream is where the agent boundaries become visible. A
	// ToolCall to a worker IS the "→ worker" boundary; the matching ToolResult IS
	// the "← worker" boundary. The child's own internal turns do NOT appear here —
	// that invisibility is the context isolation, made observable (see the package
	// docs).
	//
	// The ToolResult event carries only CallID (no tool name), so we remember which
	// CallID belonged to which tool when the ToolCall fires, then look it up on the
	// result to label the closing half of the boundary.
	callNames := map[string]string{}
	options := sporecore.NewHarnessRunOptions(task)
	options.OnStream = func(ev sporecore.HarnessStreamEvent) {
		switch ev.Kind {
		case sporecore.HarnessStreamTurnStart:
			fmt.Printf("orchestrator · plan/execute turn %d\n", ev.Turn)
		case sporecore.HarnessStreamToolCall:
			callNames[ev.CallID] = ev.Name
			if isWorker(ev.Name) {
				fmt.Printf("┌─ orchestrator → %s\n", ev.Name)
				fmt.Printf("│  received: %s\n", truncate(instructionArg(ev.Args), 200))
			} else {
				fmt.Printf("  orchestrator → %s(%s)\n", ev.Name, truncate(string(ev.Args), 160))
			}
		case sporecore.HarnessStreamToolResult:
			name, ok := callNames[ev.CallID]
			if !ok {
				name = "<tool>"
			}
			delete(callNames, ev.CallID)
			if isWorker(name) {
				tag := "returned"
				if ev.IsError {
					tag = "FAILED"
				}
				fmt.Printf("└─ %s → orchestrator\n", name)
				fmt.Printf("   %s: %s\n", tag, truncate(ev.ResultContent, 280))
			} else {
				tag := "ok"
				if ev.IsError {
					tag = "err"
				}
				fmt.Printf("  %s → orchestrator [%s]: %s\n", name, tag, truncate(ev.ResultContent, 160))
			}
		}
	}

	fmt.Printf("model       : %s\n", model)
	fmt.Printf("endpoint    : %s\n", endpoint)
	fmt.Printf("workspace   : %s\n", workspaceRoot)
	fmt.Printf("strategy    : orchestrator=PlanExecute, workers=ReAct (isolated)\n")
	fmt.Printf("agents      : orchestrator → [research_worker, writing_worker]\n")
	fmt.Printf("topic       : %s\n\n", topic)

	result := orchestrator.Run(context.Background(), options)
	switch result.Kind {
	case sporecore.RunSuccess:
		fmt.Printf("\norchestrator done (%d turn(s)): %s\n", result.Turns, truncate(result.Output, 280))
		report := filepath.Join(workspaceRoot, "report.md")
		if _, statErr := os.Stat(report); statErr == nil {
			fmt.Printf("\nreport.md now exists on disk: %s\n", report)
		} else {
			fmt.Fprintf(os.Stderr, "\nwarning: orchestrator finished but report.md was not written.\n")
		}
		return nil
	default:
		return fmt.Errorf("run did not succeed: %s %+v", result.Kind, result.Reason)
	}
}

// isWorker reports whether a tool name maps to one of the two worker agents (vs.
// write_file).
func isWorker(name string) bool {
	return name == "research_worker" || name == "writing_worker"
}

// instructionArg pulls the `instruction` field out of a tool-call args blob for
// the boundary banner, falling back to a placeholder when absent.
func instructionArg(args json.RawMessage) string {
	var probe struct {
		Instruction string `json:"instruction"`
	}
	if err := json.Unmarshal(args, &probe); err == nil && probe.Instruction != "" {
		return probe.Instruction
	}
	return "<no instruction>"
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

// truncate keeps boundary lines readable — findings and reports can be long.
func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}
