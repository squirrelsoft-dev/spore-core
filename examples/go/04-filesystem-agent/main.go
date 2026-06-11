// spore-core example 04 — ReAct with the built-in catalogue file tools.
//
// This is 03-tool-use with one substantive change. In 03 the agent's tools were
// hand-rolled: we built a StandardToolRegistry ourselves and registered each
// Tool by hand. Here we register spore-core's *built-in* catalogue instead — a
// single builder line:
//
//	.Tools(tools.StandardTools{}.CodingSet()...)   // read_file, write_file, list_dir, …
//
// Everything else is the same: the same conversational builder, the same ReAct
// loop, the same stream-printed `think · turn N` / tool-call output. The thesis
// of this example is exactly that: the harness doesn't change — only the
// registration path does.
//
// # What it shows
//
//   - Catalogue registration. .Tools(tools.StandardTools{}.CodingSet()...)
//     advertises and dispatches read_file / write_file / list_dir (and friends)
//     with no bespoke code.
//   - A real sandbox. Catalogue file tools go through a sandbox, so unlike 03's
//     pure-compute tools (which were happy with the default NullSandbox) this
//     example wires a WorkspaceScopedSandbox scoped to sample-files/.
//   - A side effect that outlives the process. The agent writes SUMMARY.md into
//     sample-files/. It is still there after the program exits — the first
//     example that leaves something behind on disk.
//
// # Run it
//
//	ollama serve &
//	ollama pull llama3.2
//	go run .
//	go run . --prompt "List the files and tell me which one mentions nutrients."
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

const systemPrompt = "You are a file-summarizing agent. Use list_dir to find files, " +
	"read_file to read each, and write_file to create SUMMARY.md. " +
	"Act using tools — do not just describe."

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

	// The agent operates inside the shipped sample-files/ directory. Resolve it
	// relative to this source file so `go run .` works from anywhere, and make it
	// absolute — the sandbox requires a canonical, existing root.
	workspaceRoot, err := sampleFilesDir()
	if err != nil {
		return err
	}

	prompt := flagValue("--prompt")
	if prompt == "" {
		prompt = "There are several .txt files in this directory. Use list_dir to find them, " +
			"read_file to read each one, then write a SUMMARY.md containing a one-sentence " +
			"summary of every file. Use write_file to create it."
	}

	// Same conversational harness as 03 — the ONLY substantive change is that we
	// register the built-in catalogue (.Tools(...)) over a real sandbox instead
	// of hand-building a StandardToolRegistry.
	mi := ollama.WithBaseURL(model, baseURL)
	sandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: workspaceRoot})
	if err != nil {
		return fmt.Errorf("failed to create sandbox: %w", err)
	}
	harness := observability.ConversationalBuilder(mi).
		Sandbox(sandbox).
		Tools(tools.StandardTools{}.CodingSet()...).
		SystemPrompt(systemPrompt).
		Build()

	task := sporecore.NewTask(prompt, sporecore.NewSessionID(), sporecore.ReActStrategy(8))

	// Print each turn (Think) and each catalogue tool call + result (Act /
	// Observe). Because the catalogue dispatches internally, the Act/Observe
	// lines come from harness STREAM events, not from inside a hand-rolled
	// dispatch like 03.
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

	fmt.Printf("model  : %s\n", model)
	fmt.Printf("dir    : %s\n", workspaceRoot)
	fmt.Printf("prompt : %s\n\n", prompt)

	result := harness.Run(context.Background(), options)
	switch result.Kind {
	case sporecore.RunSuccess:
		fmt.Printf("\nanswer (%d turn(s)): %s\n", result.Turns, result.Output)
		summary := filepath.Join(workspaceRoot, "SUMMARY.md")
		if _, statErr := os.Stat(summary); statErr == nil {
			fmt.Printf("\nSUMMARY.md now exists on disk: %s\n", summary)
		}
		return nil
	default:
		return fmt.Errorf("run did not succeed: %s %+v", result.Kind, result.Reason)
	}
}

// sampleFilesDir returns the absolute path to the shipped sample-files/ dir.
// It resolves relative to this source file (so `go run .` works from anywhere),
// falling back to the current working directory.
func sampleFilesDir() (string, error) {
	if _, file, _, ok := runtime.Caller(0); ok {
		dir := filepath.Join(filepath.Dir(file), "sample-files")
		if abs, err := filepath.Abs(dir); err == nil {
			if _, statErr := os.Stat(abs); statErr == nil {
				return abs, nil
			}
		}
	}
	abs, err := filepath.Abs("sample-files")
	if err != nil {
		return "", fmt.Errorf("failed to resolve sample-files dir: %w", err)
	}
	if _, statErr := os.Stat(abs); statErr != nil {
		return "", fmt.Errorf("sample-files dir not found at %s: %w", abs, statErr)
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

// truncate keeps observe lines readable — file contents can be long.
func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}
