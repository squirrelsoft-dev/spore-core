// spore-core example 03 — ReAct with local tools.
//
// The agent now *acts*: it thinks, calls a tool, observes the result, and loops
// until it can answer. The tools here are deliberately trivial — calculator,
// get_current_time, reverse_string — because the star of this example is the
// **Think → Act → Observe** loop, not the tools.
//
// # What it shows
//
//   - Registering custom tools on a StandardToolRegistry (which IS the
//     harness-loop ToolRegistry in Go): each tool implements the sporecore.Tool
//     interface, advertised to the model via its RegistryToolSchema and run via
//     Execute. No filesystem, no sandbox needed — these are pure functions, so
//     the conversational defaults (incl. NullSandbox) are fine; we only override
//     the tool registry on the harness config.
//   - The loop itself: the program prints each turn (Think, via the stream sink)
//     and each tool call + result (Act / Observe, from inside Execute), so you can
//     watch the agent work.
//
// # Run it
//
//	ollama serve &
//	ollama pull llama3.2
//	go run .
//	go run . --prompt "reverse the word 'mycelium' and multiply 6 by 7"
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
)

// ----------------------------------------------------------------------------
// Three trivial, pure-compute tools. Each implements sporecore.Tool; schema()
// is what the model sees, Execute is what runs.
// ----------------------------------------------------------------------------

// calculatorTool computes a binary arithmetic operation.
type calculatorTool struct{}

func (calculatorTool) Name() string                { return "calculator" }
func (calculatorTool) IsSubagentTool() bool        { return false }
func (calculatorTool) MayProduceLargeOutput() bool { return false }

func (calculatorTool) Execute(_ context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	return runTool(call, func(input map[string]any) (string, error) {
		a, err := number(input, "a")
		if err != nil {
			return "", err
		}
		b, err := number(input, "b")
		if err != nil {
			return "", err
		}
		op, _ := input["op"].(string)
		var value float64
		switch op {
		case "+":
			value = a + b
		case "-":
			value = a - b
		case "*":
			value = a * b
		case "/":
			if b == 0 {
				return "", fmt.Errorf("division by zero")
			}
			value = a / b
		default:
			return "", fmt.Errorf("unknown op %q (use + - * /)", op)
		}
		return strconv.FormatFloat(value, 'f', -1, 64), nil
	})
}

// timeTool returns the current time of day as HH:MM:SS UTC.
type timeTool struct{}

func (timeTool) Name() string                { return "get_current_time" }
func (timeTool) IsSubagentTool() bool        { return false }
func (timeTool) MayProduceLargeOutput() bool { return false }

func (timeTool) Execute(_ context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	return runTool(call, func(map[string]any) (string, error) {
		return time.Now().UTC().Format("15:04:05") + " UTC", nil
	})
}

// reverseTool reverses the characters in a string.
type reverseTool struct{}

func (reverseTool) Name() string                { return "reverse_string" }
func (reverseTool) IsSubagentTool() bool        { return false }
func (reverseTool) MayProduceLargeOutput() bool { return false }

func (reverseTool) Execute(_ context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	return runTool(call, func(input map[string]any) (string, error) {
		text, ok := input["text"].(string)
		if !ok {
			return "", fmt.Errorf("missing string 'text'")
		}
		runes := []rune(text)
		for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
			runes[i], runes[j] = runes[j], runes[i]
		}
		return string(runes), nil
	})
}

// runTool decodes the call input, runs fn, prints the Act/Observe step, and maps
// the result onto a ToolOutput — shared by all three tools so the loop stays
// visible.
func runTool(call sporecore.ToolCall, fn func(map[string]any) (string, error)) sporecore.ToolOutput {
	var input map[string]any
	if len(call.Input) > 0 {
		_ = json.Unmarshal(call.Input, &input)
	}
	content, err := fn(input)
	if err != nil {
		fmt.Printf("    act    → %s(%s) failed: %s\n", call.Name, string(call.Input), err)
		return sporecore.NewToolOutputError(err.Error())
	}
	fmt.Printf("    act    → %s(%s) = %s\n", call.Name, string(call.Input), content)
	return sporecore.NewToolOutputSuccess(content)
}

// number reads a numeric field, accepting JSON numbers OR strings ("144"),
// since local models often pass numbers as strings.
func number(input map[string]any, key string) (float64, error) {
	v, ok := input[key]
	if !ok {
		return 0, fmt.Errorf("missing number %q", key)
	}
	switch n := v.(type) {
	case float64:
		return n, nil
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(n), 64)
		if err != nil {
			return 0, fmt.Errorf("%q is not a number: %v", key, v)
		}
		return f, nil
	default:
		return 0, fmt.Errorf("%q is not a number: %v", key, v)
	}
}

func schema(name, description, properties string, required []string) sporecore.RegistryToolSchema {
	req, _ := json.Marshal(required)
	params := fmt.Sprintf(`{"type":"object","properties":%s,"required":%s}`, properties, string(req))
	return sporecore.RegistryToolSchema{
		Name:        name,
		Description: description,
		Parameters:  json.RawMessage(params),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true},
	}
}

func main() {
	modelFlag := flag.String("model", "", "Ollama model id (default llama3.2 or $SPORE_OLLAMA_MODEL)")
	promptFlag := flag.String("prompt", "", "the task prompt for the agent")
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
	prompt := *promptFlag
	if prompt == "" {
		prompt = "Use your tools to answer: what is 144 divided by 12, what is the current time, " +
			"and what is 'harness' reversed?"
	}

	// Build the same conversational config as 01/02, then swap in our tool
	// registry. StandardToolRegistry IS the harness-loop ToolRegistry in Go, so a
	// populated one plugs straight into the config.
	model := ollama.WithBaseURL(modelID, baseURL)
	registry := sporecore.NewStandardToolRegistry()
	mustRegister(registry, calculatorTool{}, schema(
		"calculator",
		"Compute a binary arithmetic operation. 'op' is one of + - * /.",
		`{"a":{"type":"number"},"b":{"type":"number"},"op":{"type":"string","enum":["+","-","*","/"]}}`,
		[]string{"a", "b", "op"},
	))
	mustRegister(registry, timeTool{}, schema(
		"get_current_time",
		"Return the current time of day as HH:MM:SS UTC. Takes no arguments.",
		`{}`,
		[]string{},
	))
	mustRegister(registry, reverseTool{}, schema(
		"reverse_string",
		"Reverse the characters in a string.",
		`{"text":{"type":"string"}}`,
		[]string{"text"},
	))

	cfg := observability.ConversationalBuilder(model).BuildConfig()
	cfg.ToolRegistry = registry
	harness := sporecore.NewStandardHarness(cfg)

	task := sporecore.NewTask(prompt, sporecore.NewSessionID(), sporecore.LoopStrategy{
		Kind:          sporecore.StrategyReAct,
		MaxIterations: 6,
	})
	// Print each turn so the "Think" steps are visible alongside the tool calls.
	options := sporecore.NewHarnessRunOptions(task)
	toolName := map[string]string{}
	toolArgs := map[string]string{}
	options.OnStream = func(ev sporecore.HarnessStreamEvent) {
		switch ev.Kind {
		case sporecore.HarnessStreamTurnStart:
			fmt.Printf("think  · turn %d\n", ev.Turn)
		case sporecore.HarnessStreamToolCallStart:
			// Streamed tool-call start now carries the real name + id (#103 fix).
			toolName[ev.CallID] = ev.Name
			fmt.Printf("tool   · %s (call_id=%s)\n", ev.Name, ev.CallID)
		case sporecore.HarnessStreamToolArgsDelta:
			toolArgs[ev.CallID] += ev.PartialJSON
			fmt.Printf("call   · %s(%s)\n", toolName[ev.CallID], toolArgs[ev.CallID])
		}
	}

	fmt.Printf("model  : %s\n", modelID)
	fmt.Printf("prompt : %s\n\n", prompt)
	result := harness.Run(context.Background(), options)
	switch result.Kind {
	case sporecore.RunSuccess:
		fmt.Printf("\nanswer (%d turn(s)): %s\n", result.Turns, result.Output)
	default:
		fmt.Fprintf(os.Stderr, "\nrun did not succeed: %s %+v\n", result.Kind, result.Reason)
		os.Exit(1)
	}
}

func mustRegister(r *sporecore.StandardToolRegistry, tool sporecore.Tool, s sporecore.RegistryToolSchema) {
	if err := r.Register(tool, s); err != nil {
		fmt.Fprintf(os.Stderr, "failed to register %s: %v\n", s.Name, err)
		os.Exit(1)
	}
}
