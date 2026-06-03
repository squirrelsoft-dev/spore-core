// spore-core example 07 — the storage seam, via a MarkdownMemoryProvider.
//
// # What it demonstrates
//
// The harness is STATELESS: all durable state lives behind the storage seam.
// Memory is just one domain of that seam — a storage.MemoryStore you implement.
// The simplest useful implementation is a human-readable markdown file. This
// example ships MarkdownMemoryProvider (in memory_provider.go), composes it into
// a storage.StorageProvider (NoOp for the other three domains), and runs the SAME
// agent twice against it:
//
//   - --phase store  — the agent is given facts about a fictional "Project
//     Ironwood" and writes each as a memory via the built-in memory tool. The
//     process exits leaving a readable memory.md on disk.
//   - --phase recall — a fresh process loads memory.md through the same provider
//     and answers questions that restate NONE of the facts. The agent recalls
//     them from memory via the memory tool.
//
// # The seam
//
// main never calls AppendMemory/GetMemories directly. It hands the composed
// provider's Memory() store to the harness builder's Storage seam; the harness
// threads it into the built-in memory tool's ToolContext per run. The agent
// drives all reads/writes from inside the ReAct loop. Swap the provider (e.g. the
// built-in JSONL FileSystemStorageProvider) and nothing else changes — that is
// the point of the seam.
//
// # Pinned session id (critical)
//
// Memory is keyed by SessionID; the memory tool always uses the run's SessionID.
// Both phases therefore pin the SAME id — session() = SessionID("project-ironwood"),
// NOT a generated one. With a generated id Run 2 would key a different session and
// read nothing back.
//
// # Scope
//
// All facts use storage.StorageScopeProject (the memory tool rejects Local). The
// prompts instruct the agent to use scope "project" consistently so the recall
// read hits the same scope the store writes wrote.
//
// # Run it
//
//	ollama serve &
//	ollama pull llama3.2
//	go run . --phase store     # writes memory.md
//	cat memory.md              # inspect the human-readable artifact
//	go run . --phase recall    # answers from memory.md alone
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

// session returns the pinned session id shared by BOTH phases. Memory is
// session-keyed, so store and recall MUST agree on this id or recall reads
// nothing. This is the single most important line in the example.
func session() sporecore.SessionID { return sporecore.SessionID("project-ironwood") }

const storeSystemPrompt = "You are a memory-keeping agent. You will be given a briefing of " +
	"facts. For EACH distinct fact, call the `memory` tool with operation \"write\", scope " +
	"\"project\", role \"assistant\", and the fact text as `content`. Write the facts verbatim " +
	"and one at a time. Do not summarize or merge facts. When every fact has been written, reply " +
	"with a short confirmation of how many you stored."

const recallSystemPrompt = "You are a recall agent. Everything you know about Project Ironwood " +
	"lives in memory — nothing is in this prompt. FIRST call the `memory` tool with operation " +
	"\"read\", scope \"project\" to load what you remember. THEN answer the user's questions using " +
	"only the recalled memories. Cite the relevant remembered fact when you answer. Do not invent " +
	"facts that are not in memory."

const recallQuestions = "Answer these about Project Ironwood, using only your memory:\n" +
	"1. How many engineers are on the team, and who leads it?\n" +
	"2. What database was chosen as the system of record, and why over the alternative?\n" +
	"3. What are the two hard constraints?\n" +
	"4. What is the known single point of failure?"

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

	// memory.md lives next to this example's sources so `go run .` works from
	// anywhere and the artifact is easy to find and inspect between phases.
	memoryPath, err := memoryFilePath()
	if err != nil {
		return err
	}

	// Default (no --phase): run store, then point the user at recall and exit.
	phase := flagValue("--phase")
	defaultPhase := phase == ""
	if defaultPhase {
		phase = "store"
	}

	mi := ollama.WithBaseURL(model, baseURL)

	fmt.Printf("model      : %s\n", model)
	fmt.Printf("memory.md  : %s\n", memoryPath)
	fmt.Printf("session id : %s  (pinned — shared by both phases)\n", session())
	fmt.Printf("phase      : %s\n\n", phase)

	switch phase {
	case "store":
		// Read the briefing and feed it to the agent. The agent writes each fact
		// via the memory tool; main never writes memory itself.
		briefing, err := readNextTo("project-ironwood.md")
		if err != nil {
			return err
		}
		taskPrompt := "Here is the Project Ironwood briefing. Store each fact to memory.\n\n" + briefing
		output, err := runPhase(mi, memoryPath, storeSystemPrompt, taskPrompt)
		if err != nil {
			return err
		}
		fmt.Printf("\nstored. agent said: %s\n", output)
		if _, statErr := os.Stat(memoryPath); statErr == nil {
			fmt.Printf("\nmemory.md now exists on disk: %s\n", memoryPath)
			fmt.Println("inspect it, then run:  go run . --phase recall")
		}
		if defaultPhase {
			fmt.Println("\n(no --phase given, so we ran `store`. Now run `go run . --phase recall`.)")
		}
		return nil

	case "recall":
		if _, statErr := os.Stat(memoryPath); statErr != nil {
			return fmt.Errorf("memory.md does not exist yet at %s.\n"+
				"Run `go run . --phase store` first.", memoryPath)
		}
		output, err := runPhase(mi, memoryPath, recallSystemPrompt, recallQuestions)
		if err != nil {
			return err
		}
		fmt.Printf("\nanswers from memory:\n%s\n", output)
		return nil

	default:
		return fmt.Errorf("unknown --phase %q. Use `store` or `recall`", phase)
	}
}

// runPhase builds a harness over the markdown memory provider + the built-in
// memory tool, pins the shared session id, runs one task, and streams the loop.
func runPhase(mi sporecore.ModelInterface, memoryPath, systemPrompt, taskPrompt string) (string, error) {
	// Compose the real markdown MemoryStore with NoOp for the other three storage
	// domains. This is the entire integration: the harness threads the memory
	// store into the memory tool's context per run. Storage(runStore, memStore)
	// takes the run-domain and memory-domain stores; here only memory matters, so
	// the run store is nil (defaults to an in-memory store for catalogue tools).
	storage := NewMarkdownMemoryProvider(memoryPath).IntoStorageProvider()

	harness := observability.ConversationalBuilder(mi).
		Storage(nil, storage.Memory()).       // ← the seam: the memory-domain store
		Tool(tools.StandardTools{}.Memory()). // ← the built-in memory read/write tool
		SystemPrompt(systemPrompt).
		// Structured mode helps small Ollama models emit clean tool calls (one per
		// turn, no interleaved reasoning, so the "think" line is just a turn marker).
		WithModelParams(sporecore.ModelParams{StructuredToolCalls: true}).
		Build()

	// PIN the session id — both phases pass the same one so recall reads what
	// store wrote.
	task := sporecore.NewTask(taskPrompt, session(), sporecore.LoopStrategy{
		Kind:          sporecore.StrategyReAct,
		MaxIterations: 20,
	})

	options := sporecore.NewHarnessRunOptions(task)
	options.OnStream = func(ev sporecore.HarnessStreamEvent) {
		switch ev.Kind {
		case sporecore.HarnessStreamTurnStart:
			fmt.Printf("think  · turn %d\n", ev.Turn)
		case sporecore.HarnessStreamToolCall:
			fmt.Printf("    act    → %s(%s)\n", ev.Name, truncate(string(ev.Args), 160))
		case sporecore.HarnessStreamToolResult:
			tag := "obs "
			if ev.IsError {
				tag = "obs(err)"
			}
			fmt.Printf("    %s→ %s\n", tag, truncate(ev.ResultContent, 160))
		}
	}

	result := harness.Run(context.Background(), options)
	switch result.Kind {
	case sporecore.RunSuccess:
		return result.Output, nil
	default:
		return "", fmt.Errorf("run did not succeed: %s %+v", result.Kind, result.Reason)
	}
}

// memoryFilePath returns the absolute path to memory.md next to this source file
// (so `go run .` works from anywhere), falling back to the working directory.
func memoryFilePath() (string, error) {
	dir := "."
	if _, file, _, ok := runtime.Caller(0); ok {
		dir = filepath.Dir(file)
	}
	abs, err := filepath.Abs(filepath.Join(dir, "memory.md"))
	if err != nil {
		return "", fmt.Errorf("resolve memory.md path: %w", err)
	}
	return abs, nil
}

// readNextTo reads a file located next to this source file (e.g. the briefing).
func readNextTo(name string) (string, error) {
	dir := "."
	if _, file, _, ok := runtime.Caller(0); ok {
		dir = filepath.Dir(file)
	}
	data, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return "", fmt.Errorf("read %s: %w", name, err)
	}
	return string(data), nil
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

// truncate keeps stream lines readable — memory reads return a JSON array of
// entries.
func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}
