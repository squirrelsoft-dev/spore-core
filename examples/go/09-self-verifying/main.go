// spore-core example 09 — the SelfVerifying loop strategy.
//
// # What this example demonstrates
//
// Quality loops are a harness concern, not application logic. The agent drafts a
// Go function, a fresh evaluator run critiques that draft against an explicit
// spec, and a Verifier turns the critique into a verdict. If the verdict is
// FAIL, the reason is injected back into the build context and the loop revises.
// This repeats until the verifier returns Passed or max-iterations is exhausted.
// You write NO loop code — you wire a composed strategy and a Verifier, and the
// harness runs the verify -> revise cycle for you. Post-#119 the strategy is a
// tree: SelfVerifying(inner: ReAct{output: "worker-schema"}, evaluator: ""). The
// `inner` (worker) slot is STRUCTURED — a bare ReAct there MUST declare an
// `output` schema (here `worker-schema`) so its result is evaluable
// (ExecutionRegistry.Validate enforces this at run entry). The `evaluator` stays
// the EMPTY handle (""), which resolves to the registry verifier registered under
// "". Post-#124 there is NO evaluator-agent setter — the evaluate phase defaults
// to the inner worker's resolved agent.
//
// # The task the agent-under-test must solve
//
// Write a Go function `ParseIntList(string) ([]int, error)` that parses a
// comma-separated list of integers. The verifier checks FIVE criteria, each
// explicitly:
//
//  1. SIGNATURE: `func ParseIntList(input string) ([]int, error)` returning a
//     typed/sentinel error (e.g. a custom ParseIntListError) on bad input.
//  2. EDGE CASES: empty/whitespace-only input -> empty slice, nil error;
//     whitespace around each number tolerated (" 1, 2 ,3 " -> [1 2 3]); a
//     non-integer token -> non-nil error, NEVER a panic.
//  3. DOC COMMENTS: the function has a doc comment describing what it does.
//  4. NO PANICS: no panic() and no must-style helpers anywhere in the code.
//  5. USAGE EXAMPLE: at least one usage example in the doc comment (an
//     Example-style snippet showing a call and its result).
//
// # How the draft reaches the evaluator — and why we need a file tool
//
// Reading the strategy source (runSelfVerifying / runSelfVerifyEvaluatePhase in
// self_verifying.go) settles the tool question. The evaluate phase builds a
// FRESH evaluator run whose context is seeded ONLY with a directive containing
// the task.Instruction plus a read-only sandbox. The build agent's draft text is
// NOT auto-injected into the evaluator's context. So for the evaluator to
// actually read the draft, the draft must live on disk where the (read-only)
// evaluator can read it.
//
// Therefore this example wires exactly the minimal file tool set:
//   - write_file — the BUILD agent saves its draft to workspace/parse_int_list.go.
//   - read_file  — the EVALUATOR reads that file back (its write_file is blocked
//     by the internally-derived ReadOnlySandbox).
//
// No web_search, no shell, nothing else. The loop is the point.
//
// # The observability seam — reportingVerifier
//
// Sub-loop streaming is suppressed by design (the build and evaluate sub-runs run
// with a nil sink, exactly like PlanExecute). The ONE reliable seam to watch the
// verify -> revise cycle is the Verifier itself: the harness calls
// Verify(SelfVerifyInput) once per iteration, and SelfVerifyInput carries the
// DRAFT (BuildResult output), the CRITIQUE (EvalResult output), and the 0-indexed
// Iteration. So we wrap the bridged EvaluatorResponseVerifier in a small
// reportingVerifier that prints, each iteration: a 1-based header with the
// configured max, the draft, the critique, and the verdict — then delegates the
// actual pass/fail decision to the inner verifier.
//
// EvaluatorResponseVerifier matches the evaluator's text against a PASS pattern
// and a FAIL: <reason> pattern; if NEITHER matches it returns FAIL by contract
// (default-to-FAIL is baked into the verifier and reinforced by the harness's
// evaluator directive — "you did NOT write this code; default to FAIL unless you
// can confirm it is right").
//
// # Run it
//
//	ollama serve &
//	ollama pull llama3.2
//	go run .                       # default model llama3.2, 3 iterations
//	go run . --max-iterations 5
//	go run . --model qwen2.5-coder:7b
//
// See the README for the honest rough-edges section: SelfVerifying against a
// small local model is genuinely flaky (the evaluator may mis-judge, the loop
// may exhaust without passing). A larger hosted model gives a cleaner demo.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/verifier"
)

// draftFile is the filename the build agent writes and the evaluator reads back.
const draftFile = "parse_int_list.go"

// taskPrompt is the spec the agent must satisfy. It is the Task instruction, so
// the BUILD agent sees it directly, and — because the evaluate phase embeds the
// task.Instruction in the evaluator's directive — the EVALUATOR sees the exact
// same five criteria. One source of truth for both roles.
const taskPrompt = "Write a Go function named `ParseIntList` and save it to the file " +
	"`parse_int_list.go` using the write_file tool. It must satisfy ALL of the following, " +
	"which you will be graded on criterion-by-criterion:\n" +
	"\n" +
	"1. SIGNATURE: `func ParseIntList(input string) ([]int, error)`. On bad input it must " +
	"return a non-nil error of a custom type you define in the same file (e.g. a " +
	"`ParseIntListError` struct or a sentinel error), never a bare errors.New string only.\n" +
	"2. EDGE CASES: empty or whitespace-only input returns an empty slice and a nil error; " +
	"whitespace around each number is tolerated (e.g. \" 1, 2 ,3 \" parses to [1 2 3]); a " +
	"non-integer token returns a non-nil error and NEVER panics.\n" +
	"3. DOC COMMENTS: the function has a `//` doc comment describing what it does.\n" +
	"4. NO PANICS: no `panic(...)` call anywhere, and no must-style helper that panics on error.\n" +
	"5. USAGE EXAMPLE: include at least one usage example in the doc comment showing a call " +
	"to ParseIntList and its result.\n" +
	"\n" +
	"Write ONLY the file contents — a valid Go source file beginning with `package main`. " +
	"Save it with write_file, then report that you are done."

// systemPrompt is shared by the build agent and the evaluator agent (the harness
// system prompt is shared across both phases). It is deliberately role-neutral:
// the build/evaluate framing is supplied per-phase (the build agent gets the spec
// as its task; the evaluator gets the harness's built-in review directive plus
// the same spec). It reinforces the file-tool contract and the evaluator's
// default-to-FAIL posture.
const systemPrompt = "You work on Go code. Your only tools are write_file (save a file to " +
	"the workspace) and read_file (read a file back). You have no shell and cannot run or " +
	"compile code.\n" +
	"\n" +
	"When ASKED TO WRITE code: write the file with write_file, then say you are done.\n" +
	"\n" +
	"When ASKED TO REVIEW code: first read_file the file under review. Then check the work " +
	"against EACH numbered criterion in the task, one at a time. You did NOT write this code " +
	"— default to FAIL unless you can positively confirm every criterion holds. Respond with " +
	"EXACTLY ONE verdict line as the LAST line of your reply:\n" +
	"  - `PASS` if and only if every criterion holds, or\n" +
	"  - `FAIL: <which criteria failed and why>` otherwise.\n" +
	"Never emit PASS when unsure."

// buildMaxTurns is a generous per-build/evaluate sub-run turn budget so a small
// model can take a few tool calls before claiming done.
const buildMaxTurns uint32 = 12

// workerSchema is the worker (build-phase) output contract (`worker-schema`).
// Post-#119, SelfVerifying's `inner` (worker) slot is STRUCTURED: a bare ReAct
// there must declare an `output` schema so its result is EVALUABLE
// (ExecutionRegistry.Validate enforces this via checkStructuredSlot). The contract
// is {file: string, summary?: string} with file required.
func workerSchema() json.RawMessage {
	return json.RawMessage(`{
  "type": "object",
  "properties": {
    "file": { "type": "string", "description": "Path of the file the worker wrote." },
    "summary": { "type": "string", "description": "Short description of what was written." }
  },
  "required": ["file"]
}`)
}

// buildRegistry assembles the ExecutionRegistry the composed strategy's handles
// resolve against. The worker leaf carries the `worker-schema` output handle; the
// SelfVerifying `evaluator` is the EMPTY handle (""), so the verifier is
// registered under "" (the default-fill key the #124 single-collaborator seam
// uses). The empty agent/toolset handles are default-filled by NewStandardHarness
// from the harness's own model + tools. Old flat shape (the unit
// LoopStrategy{Kind: StrategySelfVerifying}) had no registry at all.
func buildRegistry(harnessVerifier sporecore.Verifier) sporecore.ExecutionRegistry {
	return sporecore.NewExecutionRegistryBuilder().
		Schema("worker-schema", workerSchema()).
		Verifier("", harnessVerifier).
		Build()
}

// selfVerifyingStrategy is the post-#119 composed strategy:
// SelfVerifying(inner: ReAct{worker-schema}, evaluator: ""). The worker leaf
// carries the `worker-schema` output contract (required for the structured
// `worker` slot) and a PerLoop{maxIterations} budget. The `evaluator` is the EMPTY
// handle (""), which resolves to the registry verifier registered under "".
// Behavior is Escalate (the ReactPerLoop shim default). Old flat shape was the
// unit LoopStrategy{Kind: StrategySelfVerifying}.
func selfVerifyingStrategy(maxIterations uint32) sporecore.LoopStrategy {
	worker := sporecore.ReactPerLoop(maxIterations)
	schema := sporecore.SchemaRef("worker-schema")
	worker.Output = &schema
	return sporecore.SelfVerifyingStrategy(sporecore.SelfVerifyingConfig{
		Inner:     sporecore.PtrStrategy(sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, ReActCfg: &worker}),
		Evaluator: sporecore.SchemaRef(""),
		Behavior:  sporecore.BudgetExhaustedBehavior{Kind: sporecore.BehaviorEscalate},
	})
}

// reportingVerifier is a Verifier decorator: it prints the verify -> revise cycle
// to stdout, then delegates the actual verdict to an inner verifier.
//
// This is the one reliable observability seam for SelfVerifying — the build and
// evaluate sub-runs are streamed with a suppressed sink, so the verifier call is
// where the draft + critique + verdict become visible. Per iteration it prints: a
// 1-based header with the configured max, the DRAFT (BuildResult output), the
// CRITIQUE (EvalResult output), and the VERDICT. It forwards MaxIterations() from
// the inner verifier so the harness reads the same cap off the outer verifier.
type reportingVerifier struct {
	inner         sporecore.Verifier
	maxIterations uint32
}

func newReportingVerifier(inner sporecore.Verifier, maxIterations uint32) *reportingVerifier {
	return &reportingVerifier{inner: inner, maxIterations: maxIterations}
}

func (v *reportingVerifier) Verify(ctx context.Context, input sporecore.SelfVerifyInput) sporecore.SelfVerifyVerdict {
	// Iteration is 0-indexed on the wire; display it 1-based.
	n := input.Iteration + 1
	fmt.Printf("\n══════════════ iteration %d/%d ══════════════\n", n, v.maxIterations)

	fmt.Println("\n── draft (what the agent wrote) ──")
	fmt.Println(runResultOutput(input.BuildResult))

	fmt.Println("\n── evaluation (the critique) ──")
	fmt.Println(runResultOutput(input.EvalResult))

	// Delegate the actual decision to the inner verifier.
	verdict := v.inner.Verify(ctx, input)

	fmt.Println("\n── verdict ──")
	switch verdict.Kind {
	case sporecore.SelfVerifyPassed:
		fmt.Println("PASS — criteria satisfied; loop halts.")
	default:
		fmt.Printf("FAIL — %s\n", verdict.Reason)
		if n < v.maxIterations {
			fmt.Println("(reason injected into next build turn; revising…)")
		} else {
			fmt.Println("(no iterations left; loop will exhaust)")
		}
	}
	fmt.Println("════════════════════════════════════════════════")
	return verdict
}

func (v *reportingVerifier) MaxIterations() uint32 { return v.maxIterations }

var _ sporecore.Verifier = (*reportingVerifier)(nil)

// runResultOutput reduces a RunResult to printable text: the success output, or a
// short description of why the run did not complete.
func runResultOutput(r sporecore.RunResult) string {
	switch r.Kind {
	case sporecore.RunSuccess:
		return r.Output
	case sporecore.RunFailure:
		return fmt.Sprintf("<run did not complete: %s>", r.Reason.Kind)
	default:
		return fmt.Sprintf("<run did not complete: %s>", r.Kind)
	}
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

	// Max iterations: CLI flag wins, then env var, then default 3.
	maxIterations := uint32(3)
	if v, ok := parseMaxIterations(flagValue("--max-iterations")); ok {
		maxIterations = v
	} else if v, ok := parseMaxIterations(os.Getenv("SPORE_MAX_ITERATIONS")); ok {
		maxIterations = v
	}

	prompt := flagValue("--prompt")
	if prompt == "" {
		prompt = taskPrompt
	}

	// The agents operate inside this example's workspace/ directory. Resolve it
	// relative to this source file so `go run .` works from anywhere, create it,
	// and make it absolute — the sandbox requires a canonical, existing root.
	workspaceRoot, err := workspaceDir()
	if err != nil {
		return err
	}

	// Post-#119/#124: the .evaluator_agent(..) single-collaborator seam is GONE.
	// The SelfVerifying evaluate phase defaults to the inner worker's resolved
	// agent — the harness still runs the evaluate phase as a FRESH read-only turn
	// (write_file blocked, read_file works — exactly what a reviewer needs), so no
	// separate evaluator agent is wired. The judging seam is the Verifier below
	// (the SelfVerifying `evaluator` handle resolves to it via the empty-key fill).

	// The verifier: pattern-match the evaluator's text. PASS (anchored,
	// case-insensitive, multiline) -> Passed; FAIL: <reason> -> Failed(reason);
	// neither -> Failed by contract (default-to-FAIL). Built from the verifier
	// package, bridged into the harness seam via AsHarnessVerifier, then wrapped
	// in reportingVerifier so the cycle is visible. The harness reads
	// MaxIterations() off the OUTER (reporting) verifier, which forwards the same
	// cap the inner verifier was constructed with.
	inner, err := verifier.NewEvaluatorResponseVerifier(
		`(?im)^\s*PASS\s*$`,
		`(?im)FAIL:\s*.+`,
		maxIterations,
	)
	if err != nil {
		return fmt.Errorf("failed to construct verifier: %w", err)
	}
	harnessVerifier := newReportingVerifier(verifier.AsHarnessVerifier(inner), maxIterations)

	// Build harness: conversational preset, workspace sandbox, the minimal file
	// tool set (write_file for the builder + read_file for the evaluator), shared
	// system prompt, and the ExecutionRegistry carrying the `worker-schema` output
	// contract + the verifier under the empty `evaluator` key. Go wiring asymmetry:
	// the builder has no Registry setter, so set cfg.Registry on the built config
	// before NewStandardHarness (mirrors 08's cfg.Hooks pattern).
	mi := ollama.WithBaseURL(model, baseURL)
	sandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: workspaceRoot})
	if err != nil {
		return fmt.Errorf("failed to create sandbox: %w", err)
	}

	cfg := observability.ConversationalBuilder(mi).
		Sandbox(sandbox).
		Tool(tools.StandardTools{}.WriteFile()). // ← builder writes the draft
		Tool(tools.StandardTools{}.ReadFile()).  // ← evaluator reads it back
		SystemPrompt(systemPrompt).
		BuildConfig()
	// Post-#119: the worker leaf's `worker-schema` output handle and the
	// empty-key evaluator verifier resolve against this registry; the empty
	// agent/toolset handles default-fill from the harness's own model + tools.
	cfg.Registry = buildRegistry(harnessVerifier)
	harness := sporecore.NewStandardHarness(cfg)

	// THE STRATEGY. There is no loop code below — the harness runs the
	// verify -> revise cycle via the composed SelfVerifying(inner: ReAct, evaluator)
	// tree. The worker leaf carries the `worker-schema` output contract (required
	// for the structured `worker` slot) and a PerLoop{maxIterations} budget; a
	// generous global turn budget lets a small model take a few tool calls per
	// build/evaluate sub-run before claiming done.
	maxTurns := buildMaxTurns
	task := sporecore.NewTask(prompt, sporecore.NewSessionID(), selfVerifyingStrategy(maxIterations)).
		WithBudget(sporecore.BudgetLimits{MaxTurns: &maxTurns})

	fmt.Printf("model         : %s\n", model)
	fmt.Printf("base url      : %s\n", baseURL)
	fmt.Printf("workspace     : %s\n", workspaceRoot)
	fmt.Printf("strategy      : SelfVerifying (draft → critique → revise)\n")
	fmt.Printf("max iterations: %d\n", maxIterations)
	fmt.Printf("verifier      : EvaluatorResponseVerifier (PASS / FAIL:) wrapped in reportingVerifier\n")
	fmt.Printf("\nThe agent will draft `ParseIntList`, an evaluator will critique it against the\n")
	fmt.Printf("five spec criteria, and the loop revises until PASS or %d iterations elapse.\n\n", maxIterations)

	draftPath := filepath.Join(workspaceRoot, draftFile)
	result := harness.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
	switch {
	case result.Kind == sporecore.RunSuccess:
		fmt.Printf("\n✓ PASSED — the evaluator accepted the draft (after at most %d iteration(s), %d build turn(s) total).\n",
			maxIterations, result.Turns)
		if code, readErr := os.ReadFile(draftPath); readErr == nil {
			fmt.Printf("\n── final function (%s) ──\n%s\n", draftPath, string(code))
		}
		return nil
	case result.Kind == sporecore.RunFailure && result.Reason.Kind == sporecore.HaltSelfVerifyExhausted:
		fmt.Printf("\n✗ EXHAUSTED — %d iteration(s) elapsed without a PASS.\n", result.Reason.Iterations)
		fmt.Printf("last failure reason: %s\n", result.Reason.Reason)
		if code, readErr := os.ReadFile(draftPath); readErr == nil {
			fmt.Printf("\n── last draft on disk (%s) ──\n%s\n", draftPath, string(code))
		}
		fmt.Printf("\nThis is an expected rough edge with small local models — see the README. " +
			"Try a larger model or raise --max-iterations.\n")
		os.Exit(1)
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

// parseMaxIterations parses a positive uint32 from s; ok is false for empty,
// unparseable, or non-positive input (so the caller falls back to the default).
func parseMaxIterations(s string) (uint32, bool) {
	if s == "" {
		return 0, false
	}
	n, err := strconv.ParseUint(s, 10, 32)
	if err != nil || n == 0 {
		return 0, false
	}
	return uint32(n), true
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
