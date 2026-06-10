// spore-core example 05 — a custom tool you write yourself.
//
// Examples 03 and 04 showed the two *built-in* tool paths: hand-rolling the
// harness-loop ToolRegistry (03) and registering the shipped catalogue with
// .Tools(tools.StandardTools{}.CodingSet()...) (04). This example shows the
// third and most important path — bringing your own tool — using the ergonomic
// DefineTool helper (the Go analog of Rust's `tool!` macro).
//
// # The two custom tools
//
// Both live in the local tools package, built with tools.DefineTool: a typed
// input struct, an explicit parameter schema, annotations, and a typed exec
// function.
//
//   - remember(key, value) — persists a fact into the run store. It MUTATES
//     shared state, so it is not ReadOnly.
//   - recall(key) — reads a fact back out. It only reads, so it is ReadOnly +
//     Idempotent.
//
// # The pattern: DefineTool(...) → .Tool()
//
//  1. Call tools.DefineTool[T](name, description, annotations, schema, execFn).
//     It synthesizes the sporecore.Tool methods, unmarshals the model's
//     arguments into T (a bad decode becomes a recoverable "invalid parameters"
//     error so ToolCallRepair can retry), and bundles the impl with its schema
//     into a sporecore.StandardTool so the two can never drift.
//  2. Register each with .Tool(...). The builder wires the sandbox and a per-run
//     *ToolContext (the storage seam) automatically — the harness doesn't
//     change, only what you register does.
//
// Unlike Rust, the schema is passed EXPLICITLY (by design): the core module
// stays dependency-free rather than pulling in a reflection-based JSON-schema
// library. Deriving the schema from T is a possible later opt-in.
//
// Two builder differences from 04: there is no .Tools(...) catalogue, and no
// explicit .Sandbox(...) / .Storage(...). Build() defaults the run store to an
// in-memory provider whenever .Tool() tools are present, so the run store works
// for free.
//
// # Run it
//
//	ollama serve &
//	ollama pull llama3.2
//	go run .
//	go run . --prompt "Research mycelium. Remember a few facts, then recall and summarize them."
package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"

	"github.com/squirrelsoft-dev/spore-core/examples/go/05-custom-sandboxed-tool/tools"
)

const systemPrompt = "You are a research agent with a memory. Research the topic the user gives " +
	"you across several turns. As you discover each fact, call `remember` to store it under a " +
	"short, stable key (e.g. 'habitat', 'diet'). Keep track of the keys you use. When you have " +
	"gathered enough facts, call `recall` on each key you remembered, then write a final summary " +
	"built ONLY from the recalled facts. Act using tools — do not just describe."

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

	prompt := flagValue("--prompt")
	if prompt == "" {
		prompt = "Research the common octopus. Remember a few key facts (habitat, diet, lifespan, " +
			"intelligence), then recall them and write a short summary."
	}

	// Same conversational harness as 03 / 04 — the substantive change is that we
	// register two tools WE wrote (.Tool(...)) instead of a catalogue preset. No
	// .Sandbox(...) (these tools ignore it) and no .Storage(...) (Build() defaults
	// to an in-memory run store when .Tool() tools are present).
	mi := ollama.WithBaseURL(model, baseURL)
	harness := observability.ConversationalBuilder(mi).
		Tool(tools.RememberTool()).
		Tool(tools.RecallTool()).
		SystemPrompt(systemPrompt).
		Build()

	task := sporecore.NewTask(prompt, sporecore.NewSessionID(), sporecore.ReActStrategy(12))

	// Print each turn (Think) and each tool call + result (Act / Observe) from
	// harness STREAM events — the builder dispatches our tools internally, just
	// as it does the catalogue in 04.
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
	fmt.Printf("tools  : remember, recall\n")
	fmt.Printf("prompt : %s\n\n", prompt)

	result := harness.Run(context.Background(), options)
	switch result.Kind {
	case sporecore.RunSuccess:
		fmt.Printf("\nsummary (%d turn(s)): %s\n", result.Turns, result.Output)
		return nil
	default:
		return fmt.Errorf("run did not succeed: %s %+v", result.Kind, result.Reason)
	}
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

// truncate keeps observe lines readable — recalled facts can be long.
func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}
