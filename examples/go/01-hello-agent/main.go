// spore-core example 01 — hello agent.
//
// The smallest real thing you can build with spore-core: turn a model into a
// running agent and ask it to say hello. No tools, no filesystem, no
// multi-turn state.
//
// observability.NewConversationalHarness(model) defaults every required
// component (a model-backed agent, an empty tool registry, a null sandbox, a
// standard context manager, and respond-and-stop termination), so the whole
// thing is a few lines. Later examples override individual defaults — add
// tools, change the loop strategy — via the conversational builder.
//
// # Run it
//
//	ollama serve &            # start a local model server
//	ollama pull llama3.2      # pull the default model
//	go run .                  # or: go run . --model <id>
//
// SPORE_OLLAMA_MODEL / SPORE_OLLAMA_BASE_URL override the model id and the
// Ollama endpoint (default http://localhost:11434).
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
)

func main() {
	// Model id + endpoint come from flags/env so you can swap models without a
	// recompile.
	modelFlag := flag.String("model", "", "Ollama model id (default llama3.2 or $SPORE_OLLAMA_MODEL)")
	flag.Parse()

	modelID := *modelFlag
	if modelID == "" {
		if env := os.Getenv("SPORE_OLLAMA_MODEL"); env != "" {
			modelID = env
		} else {
			modelID = "llama3.2"
		}
	}
	baseURL := os.Getenv("SPORE_OLLAMA_BASE_URL")
	if baseURL == "" {
		baseURL = ollama.DefaultBaseURL
	}

	// A model, a harness, a task — that's the whole setup.
	model := ollama.WithBaseURL(modelID, baseURL)
	harness := observability.NewConversationalHarness(model)
	task := sporecore.SimpleTask("Reply with a friendly one-line greeting.")

	fmt.Printf("model      : %s\n", modelID)
	result := harness.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
	switch result.Kind {
	case sporecore.RunSuccess:
		fmt.Printf("result     : Success (%d turn(s))\n", result.Turns)
		fmt.Printf("greeting   : %s\n", result.Output)
	default:
		fmt.Fprintf(os.Stderr, "result     : %s (%+v)\n", result.Kind, result.Reason)
		os.Exit(1)
	}
}
