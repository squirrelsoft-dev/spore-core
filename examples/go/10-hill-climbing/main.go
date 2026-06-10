// spore-core example 10 — the HillClimbing loop strategy.
//
// # What this example demonstrates
//
// Iterative refinement under a scoring oracle is a harness concern, not
// application logic. The agent edits ONE file in place (workspace/README.md)
// across iterations. After every iteration a custom MetricEvaluator reads that
// file and asks a SEPARATE judge model to score it on three dimensions —
// Clarity, Completeness, Example quality (0–10 each) — returning the total/30
// normalized to [0,1]. The harness applies its keep/revert rule
// (hillClimbShouldKeep): a strictly-better score is KEPT; anything else is
// DISCARDED and (because RevertOnNoImprovement is on) the workspace is
// `git reset --hard`-ed back to the best-so-far. The loop halts on STAGNATION
// (MaxStagnation consecutive non-improvements) or BUDGET (MaxTurns). You write
// NO loop code — you wire a composed strategy, a metric evaluator, and an
// observability sink, and the harness runs the climb. Post-#119 the strategy is a
// tree: HillClimbing(inner: ReAct{output: "propose-schema"}, evaluator: ""). The
// `inner` (propose) slot is STRUCTURED — a bare ReAct there MUST declare an
// `output` schema (here `propose-schema`) so each iteration yields a scorable
// candidate (ExecutionRegistry.Validate enforces this). MaxStagnation /
// MinImprovementDelta are now BARE values (was Option/pointer-wrapped pre-#119),
// and the `evaluator` is the EMPTY handle (""), which resolves to the registry
// metric evaluator registered under "".
//
// # The contrast with example 09 (SelfVerifying) — the teaching point
//
// 09 has a BINARY exit condition: a Verifier returns PASS and the loop succeeds,
// or it exhausts and fails. HillClimbing has NO PASS. It is an optimization
// loop: there is only best-so-far. It does not know it is "done" — it only knows
// it has stopped improving. The terminal outcome is therefore a
// HaltStagnationLimitReached or HaltBudgetExceeded, NOT a success/fail verdict on
// quality. The README frames this contrast in a table.
//
// # SPEC NOTE — why this diverges from issue #99's original framing (Option A)
//
// The original issue asked the agent to "climb until total ≥ 25/30 or max
// iterations". Planning (#99 spec-resolution comment) established that framing
// does NOT match the real HillClimbing strategy in spore-core:
//   - There is no score-threshold success condition. The loop keeps/reverts on
//     RELATIVE improvement and halts on stagnation/budget — it never compares the
//     metric against an absolute target.
//   - MaxIterations is not a HillClimbing parameter; iterations are bounded by
//     BudgetLimits.MaxTurns. The MaxIterations constant maps there.
//   - The shipped metric.LlmJudgeEvaluator scores a FIXED construction-time
//     string, so it cannot see the evolving draft. This example therefore ships a
//     small example-local MetricEvaluator (readmeQualityEvaluator) that reads
//     workspace/README.md through the sandbox each iteration before scoring.
//
// Resolution = Option A (reframe to real semantics, no core change):
//   - ScoreThreshold (25/30) is kept as a DISPLAY annotation only. When a draft's
//     total crosses it, the printed line is marked `★ crossed target threshold`.
//     It does NOT terminate the loop — HillClimbing has no score-threshold exit.
//   - The per-iteration print is split across two seams, mirroring how the
//     harness actually exposes the run:
//   - the evaluator prints the draft + 3 sub-scores + total (it is the only
//     place that sees the rubric breakdown), and
//   - a custom ObservabilityProvider handling the HillClimbingIteration warn
//     event prints the kept/discarded/reverted decision (iteration, metric
//     value, delta) — the harness emits exactly one such event per iteration.
//
// There are no unresolved spec-question markers: every divergence above is
// resolved against the source and the #99 resolution comment.
//
// # Go wiring notes (the asymmetry with the Rust reference)
//
//   - The harness-seam MetricEvaluator interface lives in the ROOT sporecore
//     package and its signature differs from the metric package's: Evaluate(ctx,
//     sandbox, sessionID, taskID, state) (*HillClimbMetricResult,
//     *HillClimbMetricError) plus Description(). This example implements that
//     root-package consumer-side interface directly (no metric package needed).
//   - The per-iteration warn event is delivered through the HarnessObserver seam.
//     The observability.HarnessBuilder bridges an observability.ObservabilityProvider
//     into that seam and type-asserts it to WarnEmitter before forwarding the
//     HillClimbingIteration warn. So reportingObservability embeds an
//     InMemoryObservabilityProvider (which satisfies the full provider interface
//     AND WarnEmitter) and overrides EmitWarn to print the decision.
//   - Post-#119/#124, the single-collaborator cfg.MetricEvaluator field is GONE.
//     The metric evaluator resolves from the ExecutionRegistry under the empty
//     `evaluator` key. Like 09, the builder has no Registry setter, so we build the
//     config with observability.ConversationalBuilder(...).BuildConfig() and set
//     cfg.Registry on the built config before NewStandardHarness (the same
//     cfg-field asymmetry 09 uses for its registry).
//
// # Constants (see their doc comments below)
//   - MaxIterations  — maps to BudgetLimits.MaxTurns (default 6).
//   - MaxStagnation  — consecutive non-improvements before halt (2).
//   - PerIterBudget  — per-iteration build budget for the propose leaf (6).
//   - ScoreThreshold — DISPLAY annotation only (25). Never terminates.
//   - dimensionMax / totalMax — 10 per dimension, 30 total.
//
// # The seams this example wires
//   - readmeQualityEvaluator   — implements sporecore.MetricEvaluator; reads the
//     file via the SandboxProvider, runs a fresh judge-model call, prints the
//     rubric.
//   - reportingObservability   — implements observability.ObservabilityProvider;
//     wraps an in-memory provider and prints each HillClimbingIteration decision.
//
// # Run it
//
//	ollama serve &
//	ollama pull llama3.2
//	go run .                       # default model llama3.2, 6-iteration budget
//	go run . --max-iterations 8    # widen the budget
//	go run . --model qwen2.5-coder:7b
//
// See the README for the honest rough-edges section.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
)

// ============================================================================
// Constants — all clearly named, with the spec semantics in their doc comments.
// ============================================================================

// MaxIterations is the climbing-iteration ceiling. Maps to BudgetLimits.MaxTurns
// — this is the BUDGET, not a success target. The loop may halt EARLIER on
// stagnation. There is no "reached the goal" outcome; HillClimbing always halts
// on budget or stagnation. Six gives a small local model room for a few edits.
const MaxIterations uint32 = 6

// MaxStagnation is the number of consecutive non-improvements tolerated before
// the loop halts with HaltStagnationLimitReached. The stagnation counter resets
// to 0 on any kept (strictly-improving) iteration. Maps to MaxStagnation.
const MaxStagnation uint32 = 2

// PerIterBudget is the per-iteration build budget for the propose leaf. Post-#119,
// HillClimbing's `inner` is a composed ReAct(ReactPerLoop(PerIterBudget)) — this
// bounds the turns ONE proposal iteration may take (the build sub-run), while
// BudgetLimits.MaxTurns bounds the NUMBER of iterations. Six gives a small local
// model room to read the prior draft and write an improved one in a single pass.
const PerIterBudget uint32 = 6

// ScoreThreshold is a DISPLAY ANNOTATION ONLY. When a draft's total score (0–30)
// reaches this, the evaluator marks the printed line `★ crossed target
// threshold`. SPEC NOTE: this does NOT terminate the loop — HillClimbing has no
// score-threshold exit.
const ScoreThreshold uint32 = 25

// dimensionMax is the max score per rubric dimension (Clarity, Completeness,
// Example quality).
const dimensionMax uint32 = 10

// totalMax is the max total across the three dimensions (3 * dimensionMax).
const totalMax uint32 = 3 * dimensionMax

// draftFile is the file under refinement, relative to the workspace root. The
// build agent edits it in place; the evaluator reads it back through the sandbox.
const draftFile = "README.md"

// taskPrompt is what the build agent is asked to do each iteration. It edits ONE
// file in place — the climb is over successive revisions of the same README.
const taskPrompt = "You are writing the README.md for a fictional Rust crate called `ironwood`, " +
	"a small library for parsing and validating semantic-version strings. Use the write_file " +
	"tool to save your README to `README.md`. If a `README.md` already exists, first read_file " +
	"it, then improve it and write it back.\n" +
	"\n" +
	"A great README for this crate has THREE qualities, each scored 0–10 by a reviewer:\n" +
	"  1. CLARITY: a crisp one-line summary, then prose a newcomer can follow.\n" +
	"  2. COMPLETENESS: what the crate does, how to add it to Cargo.toml, the main API surface, " +
	"and an error/edge-cases note.\n" +
	"  3. EXAMPLE QUALITY: at least one fenced ```rust code block showing a real call and its " +
	"expected result.\n" +
	"\n" +
	"Write the BEST README you can, then report that you are done."

// systemPrompt is shared by the build agent. (The judge model is prompted
// separately by the evaluator; it does not share this prompt.) It reinforces the
// minimal file-tool contract.
const systemPrompt = "You write developer documentation in Markdown. Your only tools are " +
	"write_file (save a file to the workspace) and read_file (read a file back). You have no " +
	"shell and cannot run code. When asked to write or improve the README: read any existing " +
	"file first, write the improved Markdown with write_file, then say you are done."

// judgeRubric is the rubric handed to the judge model. Kept separate from the
// build prompt so the judge scores independently of how the writer was
// instructed.
const judgeRubric = "You are a strict technical-documentation reviewer. Score the README below " +
	"on THREE dimensions, each an integer from 0 to 10:\n" +
	"  - CLARITY: is there a crisp one-line summary and prose a newcomer can follow?\n" +
	"  - COMPLETENESS: does it cover what the crate does, how to add it to Cargo.toml, the main " +
	"API, and an error/edge-cases note?\n" +
	"  - EXAMPLE_QUALITY: is there at least one fenced ```rust block with a real call and " +
	"expected result?\n" +
	"\n" +
	"Reply with EXACTLY these three lines and nothing else:\n" +
	"clarity: <0-10>\n" +
	"completeness: <0-10>\n" +
	"example_quality: <0-10>"

// ============================================================================
// readmeQualityEvaluator — the example-local MetricEvaluator.
// ============================================================================

// readmeQualityEvaluator scores workspace/README.md by reading it through the
// SandboxProvider then making a SEPARATE judge-model call that returns three
// sub-scores. The value reported to the harness is total / totalMax, normalized
// to [0,1], with direction Maximize (set on the strategy payload).
//
// SPEC NOTE: this replaces the shipped metric.LlmJudgeEvaluator, which scores a
// fixed construction-time string and so cannot observe the evolving draft.
type readmeQualityEvaluator struct {
	judge sporecore.ModelInterface
}

func newReadmeQualityEvaluator(judge sporecore.ModelInterface) *readmeQualityEvaluator {
	return &readmeQualityEvaluator{judge: judge}
}

// Evaluate implements sporecore.MetricEvaluator. On success it returns a
// *HillClimbMetricResult and a nil error; a missing draft (e.g. the baseline
// before the agent has written anything) scores 0 rather than erroring, and a
// malformed judge reply reads as a poor score, never a crash.
func (e *readmeQualityEvaluator) Evaluate(
	ctx context.Context,
	sandbox sporecore.SandboxProvider,
	_ sporecore.SessionID,
	_ sporecore.TaskID,
	_ sporecore.SessionState,
) (*sporecore.HillClimbMetricResult, *sporecore.HillClimbMetricError) {
	start := time.Now()

	// Read the current draft through the sandbox root, exactly as the core
	// evaluators do. A missing/empty draft scores 0 rather than erroring.
	draftPath := filepath.Join(sandbox.WorkspaceRoot(), draftFile)
	draftBytes, _ := os.ReadFile(draftPath) //nolint:errcheck // missing file => empty draft => score 0
	draft := string(draftBytes)

	var clarity, completeness, example, total uint32
	if strings.TrimSpace(draft) == "" {
		fmt.Printf("\n── evaluator: no draft on disk yet (baseline) — total 0/%d ──\n", totalMax)
	} else {
		prompt := judgeRubric + "\n\n----- README under review -----\n" + draft
		req := sporecore.ModelRequest{
			Messages: []sporecore.Message{{
				Role:    sporecore.RoleUser,
				Content: sporecore.NewTextContent(prompt),
			}},
			Tools:  nil,
			Params: sporecore.ModelParams{},
			Stream: false,
		}
		resp, err := e.judge.Call(ctx, req)
		if err != nil {
			return nil, &sporecore.HillClimbMetricError{
				Status:  sporecore.HillClimbCrashed,
				Message: fmt.Sprintf("judge model call failed: %v", err),
			}
		}
		text := responseText(resp)

		clarity = parseDimension(text, "clarity")
		completeness = parseDimension(text, "completeness")
		example = parseDimension(text, "example_quality")
		total = clarity + completeness + example

		fmt.Printf("\n── evaluator: scored draft (%d bytes) ──\n", len(draft))
		fmt.Println(draft)
		fmt.Printf("  clarity        : %d/%d\n", clarity, dimensionMax)
		fmt.Printf("  completeness   : %d/%d\n", completeness, dimensionMax)
		fmt.Printf("  example quality: %d/%d\n", example, dimensionMax)
	}

	// SPEC NOTE: the threshold is DISPLAY-ONLY. We annotate the line; we do NOT
	// halt the loop here. The harness halts on stagnation/budget.
	crossed := ""
	if total >= ScoreThreshold {
		crossed = "  ★ crossed target threshold"
	}
	fmt.Printf("  TOTAL          : %d/%d%s\n", total, totalMax, crossed)

	value := float64(total) / float64(totalMax)
	return &sporecore.HillClimbMetricResult{
		Value:    value,
		Duration: time.Since(start),
	}, nil
}

// Description implements sporecore.MetricEvaluator. Recorded in the results-log
// description column.
func (e *readmeQualityEvaluator) Description() string {
	return fmt.Sprintf("ironwood README quality (clarity+completeness+example, /%d)", totalMax)
}

var _ sporecore.MetricEvaluator = (*readmeQualityEvaluator)(nil)

// proposeSchema is the propose-phase output contract (`propose-schema`).
// Post-#119, HillClimbing's `inner` (propose) slot is STRUCTURED: a bare ReAct
// there must declare an `output` schema so each iteration yields a scorable
// candidate (ExecutionRegistry.Validate enforces this via checkStructuredSlot).
// The contract is {file: string, summary?: string} with file required.
func proposeSchema() json.RawMessage {
	return json.RawMessage(`{
  "type": "object",
  "properties": {
    "file": { "type": "string", "description": "Path of the draft the iteration wrote." },
    "summary": { "type": "string", "description": "Short description of the change made." }
  },
  "required": ["file"]
}`)
}

// buildRegistry assembles the ExecutionRegistry the composed strategy's handles
// resolve against. The propose leaf carries the `propose-schema` output handle;
// the HillClimbing `evaluator` is the EMPTY handle (""), so the metric evaluator
// is registered under "" (the default-fill key the #124 single-collaborator seam
// uses). The empty agent/toolset handles are default-filled by NewStandardHarness
// from the harness's own model + tools. Old flat shape (the unit
// LoopStrategy{Kind: StrategyHillClimbing, ...}) had no registry at all.
func buildRegistry(evaluator sporecore.MetricEvaluator) sporecore.ExecutionRegistry {
	return sporecore.NewExecutionRegistryBuilder().
		Schema("propose-schema", proposeSchema()).
		MetricEvaluator("", evaluator).
		Build()
}

// hillClimbingStrategy is the post-#119 composed strategy:
// HillClimbing(inner: ReAct{propose-schema}, evaluator: ""). The propose leaf
// carries the `propose-schema` output contract (required for the structured
// `propose` slot) and a PerLoop{perIterBudget} budget. MaxStagnation /
// MinImprovementDelta are now BARE values (was Option/pointer-wrapped pre-#119);
// MinImprovementDelta 0.0 means any strict improvement counts. The `evaluator` is
// the EMPTY handle (""), which resolves to the registry metric evaluator
// registered under "". Old flat shape was
// LoopStrategy{Kind: StrategyHillClimbing, Direction, MaxStagnation: Some(_), ...}.
func hillClimbingStrategy(perIterBudget uint32) sporecore.LoopStrategy {
	propose := sporecore.ReactPerLoop(perIterBudget)
	schema := sporecore.SchemaRef("propose-schema")
	propose.Output = &schema
	return sporecore.HillClimbingStrategy(sporecore.HillClimbingConfig{
		Inner:                 sporecore.PtrStrategy(sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, ReActCfg: &propose}),
		Direction:             sporecore.OptimizationMaximize,
		MaxStagnation:         MaxStagnation,
		RevertOnNoImprovement: true,
		MinImprovementDelta:   0,
		Evaluator:             sporecore.AgentRef(""),
		Behavior:              sporecore.BudgetExhaustedBehavior{Kind: sporecore.BehaviorEscalate},
	})
}

// parseDimension parses a `name: <int>` line, clamped to [0, dimensionMax]. A
// missing or unparseable line scores 0 — a malformed judge reply must not crash
// the run (it just reads as a poor score, which the loop treats as a normal
// outcome).
func parseDimension(text, name string) uint32 {
	prefix := name + ":"
	for _, line := range strings.Split(text, "\n") {
		lower := strings.ToLower(strings.TrimSpace(line))
		rest, ok := strings.CutPrefix(lower, prefix)
		if !ok {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			continue
		}
		n, err := strconv.ParseUint(fields[0], 10, 32)
		if err != nil {
			continue
		}
		if uint32(n) > dimensionMax {
			return dimensionMax
		}
		return uint32(n)
	}
	return 0
}

// responseText flattens a ModelResponse's text blocks into a single string.
func responseText(resp sporecore.ModelResponse) string {
	var parts []string
	for _, b := range resp.Content {
		if b.Type == sporecore.ContentBlockTypeText {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// ============================================================================
// reportingObservability — prints each HillClimbingIteration decision.
// ============================================================================

// reportingObservability is an observability.ObservabilityProvider that
// delegates everything to an embedded InMemoryObservabilityProvider but
// additionally PRINTS each HillClimbingIteration warn event. This is the seam
// the harness uses to report the per-iteration keep/revert decision — the
// evaluator prints the scores, this prints what the loop DID with them.
//
// Embedding the in-memory provider promotes its full ObservabilityProvider
// method set AND its EmitWarn (so the type satisfies WarnEmitter, which the
// harness adapter requires before it forwards the warn). We shadow EmitWarn to
// intercept the HillClimbingIteration event, then delegate.
type reportingObservability struct {
	*observability.InMemoryObservabilityProvider
	maxIterations uint32
}

func newReportingObservability(maxIterations uint32) *reportingObservability {
	return &reportingObservability{
		InMemoryObservabilityProvider: observability.NewInMemoryObservabilityProvider(),
		maxIterations:                 maxIterations,
	}
}

// EmitWarn intercepts the HillClimbingIteration warn to print the per-iteration
// decision, then delegates to the embedded provider so the span is still stored.
func (o *reportingObservability) EmitWarn(span observability.WarnSpan) {
	if span.Event.Kind == observability.WarnKindHillClimbingIteration {
		ev := span.Event
		// iteration is 0-based on the wire (0 = baseline). Display 1-based.
		n := ev.Iteration + 1
		value := "n/a"
		if ev.MetricValue != nil {
			value = fmt.Sprintf("%.3f", *ev.MetricValue)
		}
		deltaStr := "—"
		if ev.Delta != nil {
			deltaStr = fmt.Sprintf("%+.3f", *ev.Delta)
		}
		revertedNote := ""
		if ev.Reverted {
			revertedNote = " (workspace git-reset to best-so-far)"
		}
		fmt.Printf("\n══ iteration %d/%d — %s ══  metric=%s (Δ %s)%s\n",
			n, o.maxIterations, ev.Status, value, deltaStr, revertedNote)
	}
	o.InMemoryObservabilityProvider.EmitWarn(span)
}

var _ observability.ObservabilityProvider = (*reportingObservability)(nil)
var _ observability.WarnEmitter = (*reportingObservability)(nil)

// ============================================================================
// main
// ============================================================================

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

	// Iteration budget: CLI flag wins, then env var, then MaxIterations.
	maxIterations := MaxIterations
	if v, ok := parseMaxIterations(flagValue("--max-iterations")); ok {
		maxIterations = v
	} else if v, ok := parseMaxIterations(os.Getenv("SPORE_MAX_ITERATIONS")); ok {
		maxIterations = v
	}

	prompt := flagValue("--prompt")
	if prompt == "" {
		prompt = taskPrompt
	}

	// The agent edits this example's workspace/ in place. Resolve it relative to
	// this source file so `go run .` works from anywhere, create it, and make it
	// absolute — the sandbox requires a canonical, existing root.
	workspaceRoot, err := workspaceDir()
	if err != nil {
		return err
	}

	// git-init the workspace so RevertOnNoImprovement's `git reset --hard` has a
	// clean baseline to return to. Idempotent: skip if already a repo.
	if err := initGitWorkspace(workspaceRoot); err != nil {
		return err
	}

	// Two model instances on the same Ollama endpoint: one drives the build agent
	// (writing the README), one is the judge the evaluator calls to score it.
	buildModel := ollama.WithBaseURL(model, baseURL)
	judgeModel := ollama.WithBaseURL(model, baseURL)
	evaluator := newReadmeQualityEvaluator(judgeModel)
	obs := newReportingObservability(maxIterations)

	sandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: workspaceRoot})
	if err != nil {
		return fmt.Errorf("failed to create sandbox: %w", err)
	}

	// Build harness: conversational preset, workspace sandbox, the minimal file
	// tool set (write_file + read_file), shared system prompt, the observability
	// sink, and the ExecutionRegistry carrying the `propose-schema` output contract
	// + the metric evaluator under the empty `evaluator` key. Go wiring asymmetry
	// (mirrors 09): the builder has no Registry setter, so set cfg.Registry on the
	// built config before NewStandardHarness. HillClimbing REQUIRES a metric
	// evaluator (a missing one halts with HaltHillClimbingMisconfigured, not a
	// panic).
	cfg := observability.ConversationalBuilder(buildModel).
		Sandbox(sandbox).
		Tool(tools.StandardTools{}.WriteFile()). // ← builder writes the draft
		Tool(tools.StandardTools{}.ReadFile()).  // ← evaluator/agent reads it back
		SystemPrompt(systemPrompt).
		Observability(obs).
		BuildConfig()
	// Post-#119: the propose leaf's `propose-schema` output handle and the
	// empty-key evaluator metric evaluator resolve against this registry; the
	// empty agent/toolset handles default-fill from the harness's own model + tools.
	cfg.Registry = buildRegistry(evaluator)
	harness := sporecore.NewStandardHarness(cfg)

	// THE STRATEGY. No loop code below — the harness runs the climb via the composed
	// HillClimbing(inner: ReAct{propose-schema}, evaluator) tree. MaxTurns bounds
	// the NUMBER OF ITERATIONS (the budget ceiling); MaxStagnation can halt sooner;
	// the propose leaf's PerLoop{PerIterBudget} bounds ONE proposal's build sub-run.
	// SPEC NOTE: there is no score-threshold field — by design.
	maxTurns := maxIterations
	task := sporecore.NewTask(prompt, sporecore.NewSessionID(), hillClimbingStrategy(PerIterBudget)).
		WithBudget(sporecore.BudgetLimits{MaxTurns: &maxTurns})

	fmt.Printf("model         : %s\n", model)
	fmt.Printf("base url      : %s\n", baseURL)
	fmt.Printf("workspace     : %s\n", workspaceRoot)
	fmt.Printf("strategy      : HillClimbing (score → keep/revert → climb)\n")
	fmt.Printf("direction     : Maximize (higher README score is better)\n")
	fmt.Printf("max iterations: %d (budget ceiling — NOT a success target)\n", maxIterations)
	fmt.Printf("max stagnation: %d (halt after this many non-improvements)\n", MaxStagnation)
	fmt.Printf("threshold     : %d/%d — DISPLAY ONLY (★ marks it; never halts)\n", ScoreThreshold, totalMax)
	fmt.Printf("\nThe agent will draft and refine `%s`; each iteration a judge model scores it on\n", draftFile)
	fmt.Printf("three dimensions, and the loop keeps the best — reverting the rest — until it\n")
	fmt.Printf("stops improving (stagnation) or the budget is spent. There is no PASS.\n\n")

	draftPath := filepath.Join(workspaceRoot, draftFile)
	result := harness.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
	switch {
	case result.Kind == sporecore.RunFailure && result.Reason.Kind == sporecore.HaltStagnationLimitReached:
		reportBest(result.Reason.BestMetric, draftPath)
		fmt.Printf("\n■ HALTED ON STAGNATION — %d consecutive non-improving iteration(s).\n", result.Reason.Iterations)
		fmt.Printf("This is the NORMAL terminal outcome for HillClimbing: it stopped because it\n")
		fmt.Printf("could not improve, not because it hit a target. The file on disk is best-so-far.\n")
		return nil
	case result.Kind == sporecore.RunFailure && result.Reason.Kind == sporecore.HaltBudgetExceeded:
		reportBest(math.NaN(), draftPath)
		fmt.Printf("\n■ HALTED ON BUDGET — exhausted the iteration ceiling (%s).\n", result.Reason.LimitType)
		fmt.Printf("Also a normal terminal outcome: the climb ran out of budget while still\n")
		fmt.Printf("(possibly) improving. The file on disk is the best-so-far draft.\n")
		return nil
	case result.Kind == sporecore.RunFailure && result.Reason.Kind == sporecore.HaltHillClimbingMisconfigured:
		return fmt.Errorf("HillClimbing misconfigured: %s", result.Reason.Reason)
	case result.Kind == sporecore.RunSuccess:
		// HillClimbing does not normally return Success (it has no success
		// condition); surface it honestly if a future core revision does.
		reportBest(math.NaN(), draftPath)
		fmt.Printf("\n■ run returned Success after %d turn(s) — best-so-far draft on disk.\n", result.Turns)
		return nil
	default:
		return fmt.Errorf("run did not complete as expected: %s %+v", result.Kind, result.Reason)
	}
}

// reportBest prints the best-so-far metric (when known) and the final draft on
// disk.
func reportBest(bestMetric float64, draftPath string) {
	if !math.IsNaN(bestMetric) {
		total := uint32(math.Round(bestMetric * float64(totalMax)))
		fmt.Printf("\n── best score seen: %d/%d (normalized %.3f) ──\n", total, totalMax, bestMetric)
	}
	if code, err := os.ReadFile(draftPath); err == nil {
		fmt.Printf("\n── final draft (%s) ──\n%s\n", draftPath, string(code))
	} else {
		fmt.Printf("\n(no draft was written to %s)\n", draftPath)
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

// initGitWorkspace `git init`s the workspace and makes an initial commit if it
// is not already a repo, so RevertOnNoImprovement's `git reset --hard` has a
// baseline. Best-effort and idempotent: an existing repo is skipped; a missing
// `git` binary surfaces as an error (the revert would otherwise silently no-op).
func initGitWorkspace(root string) error {
	if _, err := os.Stat(filepath.Join(root, ".git")); err == nil {
		return nil
	}
	runGit := func(args ...string) error {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		cmd.Stdout = nil
		cmd.Stderr = nil
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("git %s failed: %w", strings.Join(args, " "), err)
		}
		return nil
	}
	if err := runGit("init"); err != nil {
		return err
	}
	// Local identity so the initial commit succeeds without global git config.
	if err := runGit("config", "user.email", "example@spore-core.invalid"); err != nil {
		return err
	}
	if err := runGit("config", "user.name", "spore-core example"); err != nil {
		return err
	}
	if err := runGit("add", "-A"); err != nil {
		return err
	}
	// An empty initial commit is fine if the dir is otherwise empty.
	return runGit("commit", "--allow-empty", "-m", "baseline")
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
