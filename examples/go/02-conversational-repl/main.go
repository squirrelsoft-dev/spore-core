// spore-core example 02 — conversational REPL.
//
// Takes example 01 one step further: an interactive chat loop where the agent
// remembers what you said earlier in the session. Same conversational harness
// as 01 — the new idea here is *conversation continuity across runs*.
//
// # How memory works
//
// The harness is stateless between Run() calls: each call takes an optional
// starting SessionState (the message history) and drives one task to a final
// response. As of issue #102, RunResult on success now hands the post-run
// SessionState back, so the caller resumes the conversation LOSSLESSLY — no
// reconstruction. After each turn we feed the returned SessionState straight
// into the next run via HarnessRunOptions.SessionState. The harness appends the
// new user line on top of that history before calling the model, so the model
// sees the whole conversation and can refer back to it.
//
// This works for tool-using agents too: the returned SessionState carries the
// tool-call and tool-result messages the loop produced, which a "reconstruct
// history from output" trick could not recover.
//
// Prefer it hands-free? Wire a SessionStore and set AutoPersistSessions: the
// harness then auto-loads and auto-persists by SessionID, so you reuse the id
// instead of threading state at all (great for a web service that resumes
// across restarts).
//
// # Run it
//
//	ollama serve &
//	ollama pull llama3.2
//	go run .                  # then chat; /exit or Ctrl-D to quit
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
)

func main() {
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

	// Build the harness once; reuse it for every turn.
	model := ollama.WithBaseURL(modelID, baseURL)
	harness := observability.NewConversationalHarness(model)

	// One session id for the whole REPL, and the conversation state we thread
	// back in on each turn. Each run hands the post-run SessionState back
	// (issue #102), so we just carry it forward — lossless, no reconstruction.
	sessionID := sporecore.NewSessionID()
	state := sporecore.SessionState{}
	turnsExchanged := 0

	ctx := context.Background()
	fmt.Printf("conversational REPL — model %s. Type a message; /exit or Ctrl-D to quit.\n", modelID)
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("you> ")
		if !scanner.Scan() {
			fmt.Println()
			break // EOF (Ctrl-D)
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if line == "/exit" || line == "/quit" {
			break
		}

		// Thread the running state into this turn. The harness appends `line`
		// as the new user message before calling the model.
		task := sporecore.NewTask(line, sessionID, sporecore.ReActStrategy(4))
		threaded := state
		options := sporecore.NewHarnessRunOptions(task)
		options.SessionState = &threaded

		result := harness.Run(ctx, options)
		switch result.Kind {
		case sporecore.RunSuccess:
			fmt.Printf("bot> %s\n", result.Output)
			// Carry the post-run state forward losslessly (issue #102): it
			// already contains this turn's user + assistant messages (and any
			// tool messages a tool-using agent would produce).
			state = result.SessionState
			turnsExchanged++
		default:
			fmt.Fprintf(os.Stderr, "bot> [run did not succeed: %s %+v]\n", result.Kind, result.Reason)
		}
	}

	fmt.Printf("bye (%d turn(s); %d message(s) in history)\n", turnsExchanged, len(state.Messages))
}
