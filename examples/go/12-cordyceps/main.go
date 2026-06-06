// spore-core example 12 — cordyceps: a fully autonomous task-completion agent
// (the capstone).
//
// # The thesis: you give it a task; it does not stop until the job is done —
// and when a worker gets stuck or uncertain, it asks for help (a sibling helper,
// then a human) rather than giving up.
//
// This example composes everything the suite has built — subagents-as-tools (11),
// custom sandboxed tools (05), web_search (06), memory (07), task_list — plus two
// new capabilities:
//
//   - Architect-side skill loading (see skills.go): a load_skill tool activates
//     the bundled audit skill at runtime via a guideregistry, and a custom
//     context manager re-injects the skill body every turn (compaction-proof).
//     This is the pattern issue #115 will absorb into the harness; the live
//     loop's structural skill-injection path is not wired yet (see the README and
//     #115).
//   - A generalized consult / escalation ladder (issue #114): the analysis worker
//     escalates mid-loop to a research helper (kind=research, budget 5,
//     soft-fail) and then to a cloud-model advisor (kind=advice, budget 3,
//     escalate-to-human), resuming each time without ending its run.
//
// # Topology (depth-1)
//
//	orchestrator (ReAct, gemma4:e4b)
//	  tools: list_dir, grep, task_list, memory, write_file, bash_command,
//	         analysis_worker (SubagentTool, with consult handlers)
//	  ├── analysis_worker (Isolated) — audits ONE module
//	  │     tools: read_file, grep, research_best_practices, consult_advisor,
//	  │            load_skill
//	  ├── research_worker (Isolated) — web_search   [consult handler: research]
//	  └── advisor         (Isolated, cloud model)   [consult handler: advice]
//
// The orchestrator enumerates crates → modules, adds one task_list task per
// module, dispatches the analysis worker per task, accumulates findings in
// memory, finalizes the top 5, writes workspace/findings.md, and runs the y/N
// issue-filing flow. The audit is READ-ONLY; the only writes are
// workspace/findings.md and (approved) GitHub issues.
//
// # Run it
//
//	ollama serve &
//	ollama pull gemma4:e4b
//	export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
//	go run .
package main

import (
	"bufio"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/storage"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
)

// bundledAuditSkill is the audit skill, embedded so the example is
// self-contained even with an empty .spore/skills/. Go's //go:embed is the
// idiomatic equivalent of Rust's include_str!.
//
//go:embed skills/audit/SKILL.md
var bundledAuditSkill string

// defaultAuditPrompt is the pre-filled audit prompt the user presses enter to
// accept (from the issue). An empty line at the REPL ⇒ this verbatim.
const defaultAuditPrompt = "Audit the current repo for the rust language. Work sequentially by " +
	"identifying each crate, and each module in the crate, and adding a task to the tasklist for a " +
	"subagent to do the deep dive audit on."

// workerTimeout is the per-worker wall-clock cap. A worker can burn many internal
// ReAct turns (and mediated consults); this bounds how long the orchestrator
// waits on one delegation.
const workerTimeout = 300 * time.Second

// maxOrchestratorTurns is the orchestrator's turn budget.
const maxOrchestratorTurns uint32 = 64

const orchestratorPrompt = "You are the cordyceps orchestrator: an autonomous Rust-repo auditor. " +
	"You do not stop until the audit is complete. Work sequentially: (1) use `list_dir` to " +
	"enumerate the crates under `rust/crates/`, then the modules (`src/*.rs`) in each crate; " +
	"(2) for each module, add ONE task to the task list (`task_list`) describing the module to " +
	"audit; (3) for each task, call `analysis_worker` with an `instruction` naming the ONE module " +
	"to deep-dive audit; (4) accumulate the findings each worker returns into `memory` under a " +
	"stable key; (5) when every module is audited, pick the TOP 5 most important findings across " +
	"all modules and write them as a markdown report to `workspace/findings.md` using " +
	"`write_file`. The audit is READ-ONLY — never modify source files. Delegate the per-module " +
	"deep dives to `analysis_worker`; do not audit modules yourself. Finish by writing findings.md."

const analysisWorkerPrompt = "You are an analysis worker: you deep-dive audit exactly ONE Rust " +
	"module for real, actionable defects. BEFORE auditing, call `load_skill` with `skill_id` = " +
	"\"audit\" and follow the returned procedure and findings schema EXACTLY. Stay inside the one " +
	"module you were given. Grep first, read only narrow line ranges, and escalate with " +
	"`research_best_practices` (idiom questions) or `consult_advisor` (severity / is-this-real " +
	"questions) when genuinely unsure. Your FINAL answer must be a JSON array of " +
	"{file, line, severity, description} objects — and nothing else."

const researchPrompt = "You are a research worker. Use the web_search tool to gather current, " +
	"factual information on the Rust best-practice or idiom question you are given. Issue focused " +
	"queries, read the results, and return a concise, cited answer as plain text. Act using " +
	"web_search — do not answer from memory alone."

const advisorPrompt = "You are a senior Rust advisor. A worker is stuck on whether a finding is a " +
	"real defect, or on how to rank its severity. Use `read_file` and `grep` to investigate the " +
	"specific code in question, then give a crisp, decisive recommendation: is it a real defect, " +
	"what severity (low/medium/high/critical), and why. Be concrete and brief."

// instructionSchema is the single-parameter input schema every subagent tool
// advertises.
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
// to 06/11).
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

// buildInnerContextManager builds the standard compaction adapter (the same one
// the conversational preset installs), so the skill-injecting context manager can
// embed it and inherit every non-Assemble method.
func buildInnerContextManager(model, baseURL string) *contextmgr.StandardCompactionAdapter {
	mi := ollama.WithBaseURL(model, baseURL)
	return contextmgr.NewStandardCompactionAdapter(
		contextmgr.NewStandardContextManager(
			mi,
			contextmgr.NullCacheProvider{},
			contextmgr.DefaultCompactionConfig(),
		),
	)
}

// buildResearchHarness builds the research worker (web_search only). Used as the
// kind=research consult handler.
func buildResearchHarness(model, baseURL, endpoint string) (sporecore.Harness, error) {
	webSearch, err := buildWebSearch(endpoint)
	if err != nil {
		return nil, err
	}
	mi := ollama.WithBaseURL(model, baseURL)
	return observability.ConversationalBuilder(mi).
		Tool(webSearch).
		SystemPrompt(researchPrompt).
		Build(), nil
}

// buildAdvisorHarness builds the advisor (cloud model, read_file + grep). Used as
// the kind=advice consult handler. Rides the same Ollama endpoint via WithBaseURL;
// only the model id differs.
func buildAdvisorHarness(model, baseURL string, repoSandbox sporecore.SandboxProvider) sporecore.Harness {
	mi := ollama.WithBaseURL(model, baseURL)
	return observability.ConversationalBuilder(mi).
		Sandbox(repoSandbox).
		Tool(tools.StandardTools{}.ReadFile()).
		Tool(tools.StandardTools{}.Grep()).
		SystemPrompt(advisorPrompt).
		Build()
}

// buildAnalysisHarness builds the analysis worker: read_file, grep, the two
// consult tools, and load_skill. Isolated; audits ONE module. It shares the SAME
// run store as the orchestrator so load_skill's active-skill write and the
// context manager's read rendezvous (keyed by the worker's own session id, so
// each worker activates audit for itself).
func buildAnalysisHarness(
	model, baseURL string,
	repoSandbox sporecore.SandboxProvider,
	runStore sporecore.ToolRunStore,
	catalog *SkillCatalog,
) sporecore.Harness {
	mi := ollama.WithBaseURL(model, baseURL)
	inner := buildInnerContextManager(model, baseURL)
	skillCM := NewSkillInjectingContextManager(inner, runStore, catalog.Manifest())
	return observability.ConversationalBuilder(mi).
		Sandbox(repoSandbox).
		Storage(runStore, nil).
		ContextManager(skillCM).
		Tool(tools.StandardTools{}.ReadFile()).
		Tool(tools.StandardTools{}.Grep()).
		Tool(researchBestPracticesTool()).
		Tool(consultAdvisorTool()).
		Tool(loadSkillTool(catalog.Registry())).
		SystemPrompt(analysisWorkerPrompt).
		Build()
}

// buildAnalysisTool wraps the analysis worker as a SubagentTool with the consult
// handlers installed. The two handlers mediate by kind (research →
// research_worker, advice → advisor) with the per-kind budgets + overflow
// policies from #114.
func buildAnalysisTool(
	analysis sporecore.Harness,
	consultHandlers map[string]sporecore.ConsultHandlerEntry,
) (sporecore.StandardTool, error) {
	const name = "analysis_worker"
	const description = "Delegate a deep-dive audit of ONE Rust module: pass an `instruction` " +
		"naming the module; it loads the `audit` skill, audits the module (escalating via " +
		"consults when stuck), and returns a JSON array of {file, line, severity, description} findings."

	emptyChildRegistry := sporecore.NewStandardToolRegistry()
	subagent, err := tools.NewSubagentTool(
		name,
		description,
		instructionSchema(),
		workerTimeout,
		tools.Isolated{},
		analysis,
		emptyChildRegistry,
	)
	if err != nil {
		return sporecore.StandardTool{}, fmt.Errorf("failed to build analysis worker tool: %w", err)
	}
	subagent = subagent.WithConsultHandlers(consultHandlers)

	return tools.StandardTool{
		Implementation: subagent,
		Schema: sporecore.RegistryToolSchema{
			Name:        name,
			Description: "Deep-dive audit one module via a subagent; returns JSON findings.",
			Parameters:  instructionSchema(),
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
	ctx := context.Background()

	model := flagOrEnv("--model", "SPORE_OLLAMA_MODEL", "gemma4:e4b")
	advisorModel := flagOrEnv("--advisor-model", "SPORE_ADVISOR_MODEL", "minimax-m3:cloud")
	baseURL := os.Getenv("SPORE_OLLAMA_BASE_URL")
	if baseURL == "" {
		baseURL = ollama.DefaultBaseURL
	}

	// Required search backend (research worker) — fail fast like 06/11.
	endpoint := strings.TrimSpace(flagValue("--search-url"))
	if endpoint == "" {
		endpoint = strings.TrimSpace(os.Getenv("SPORE_WEB_SEARCH_ENDPOINT"))
	}
	if endpoint == "" {
		fmt.Fprintln(os.Stderr, "SPORE_WEB_SEARCH_ENDPOINT is not set.\n"+
			"Set it to a SearXNG JSON endpoint, e.g. "+
			"http://localhost:8888/search?format=json. See .env.example and the README.")
		os.Exit(2)
	}

	// Resolve the repo root (cwd) for the read-only audit sandbox, and this
	// example's workspace/ for the report write.
	repoRoot, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to resolve repo root: %w", err)
	}
	if abs, absErr := filepath.Abs(repoRoot); absErr == nil {
		repoRoot = abs
	}
	workspaceRoot, err := workspaceDir()
	if err != nil {
		return err
	}

	// Seed .spore/skills/audit/SKILL.md from the bundled copy if absent, so a
	// user can see the filesystem-registry shape (documented in the README).
	seedBundledAuditSkill(repoRoot)

	// One in-memory storage provider, shared by the orchestrator and the analysis
	// worker so load_skill (worker-side write) and the context manager (read)
	// rendezvous on run_store["active_skills"].
	store := storage.SingleStorageProvider(storage.NewInMemoryStorageProvider())
	runStore := store.Run()

	// Scan + register skills (bundled audit + any project/user skills).
	catalog := BootstrapCatalog(ctx, repoRoot, bundledAuditSkill)

	// The orchestrator can read the whole repo + write into its own workspace.
	// Workers and the advisor get a read-only-by-tool-set view rooted at the same
	// repo. For the read-only audit guarantee we rely on the prompt + skill
	// discipline + the workers' tool sets, not a read-only sandbox.
	orchestratorSandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: repoRoot})
	if err != nil {
		return fmt.Errorf("failed to create orchestrator sandbox: %w", err)
	}
	workerSandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: repoRoot})
	if err != nil {
		return fmt.Errorf("failed to create worker sandbox: %w", err)
	}

	// ---- Build the consult handlers (research + advice) ---------------------
	researchHandler, err := buildResearchHarness(model, baseURL, endpoint)
	if err != nil {
		return err
	}
	advisorHandler := buildAdvisorHarness(advisorModel, baseURL, workerSandbox)
	consultHandlers := map[string]sporecore.ConsultHandlerEntry{
		kindResearch: {
			Handler:  researchHandler,
			Budget:   5,
			Overflow: sporecore.ConsultOverflowPolicy{Kind: sporecore.ConsultOverflowSoftFail},
		},
		kindAdvice: {
			Handler:  advisorHandler,
			Budget:   3,
			Overflow: sporecore.ConsultOverflowPolicy{Kind: sporecore.ConsultOverflowEscalateToHuman},
		},
	}

	// ---- Build the analysis worker + wrap it (with consult handlers) --------
	analysis := buildAnalysisHarness(model, baseURL, workerSandbox, runStore, catalog)
	analysisTool, err := buildAnalysisTool(analysis, consultHandlers)
	if err != nil {
		return err
	}

	// ---- Build the orchestrator --------------------------------------------
	orchestratorModel := ollama.WithBaseURL(model, baseURL)
	orchestrator := observability.ConversationalBuilder(orchestratorModel).
		Sandbox(orchestratorSandbox).
		Storage(runStore, nil).
		Tool(tools.StandardTools{}.ListDir()).
		Tool(tools.StandardTools{}.Grep()).
		Tool(tools.StandardTools{}.TaskList()).
		Tool(tools.StandardTools{}.Memory()).
		Tool(tools.StandardTools{}.WriteFile()).
		Tool(tools.StandardTools{}.BashCommand()).
		Tool(analysisTool).
		SystemPrompt(orchestratorPrompt).
		Build()

	fmt.Printf("model        : %s\n", model)
	fmt.Printf("advisor model: %s\n", advisorModel)
	fmt.Printf("endpoint     : %s\n", endpoint)
	fmt.Printf("repo root    : %s\n", repoRoot)
	fmt.Printf("workspace    : %s\n", workspaceRoot)
	fmt.Printf("skills       : %s\n", strings.Join(catalog.Names(), ", "))
	fmt.Printf("strategy     : orchestrator=ReAct, workers=ReAct (isolated)\n\n")

	// ---- REPL: pre-filled prompt, enter accepts the default -----------------
	reader := bufio.NewScanner(os.Stdin)
	prompt := readAuditPrompt(reader)

	// Stream banners: mirror 11-multi-agent's ┌─ … └─ boundary style.
	callNames := map[string]string{}
	maxTurns := maxOrchestratorTurns
	task := sporecore.NewTask(prompt, sporecore.NewSessionID(), sporecore.LoopStrategy{
		Kind:          sporecore.StrategyReAct,
		MaxIterations: 64,
	}).WithBudget(sporecore.BudgetLimits{MaxTurns: &maxTurns})
	options := sporecore.NewHarnessRunOptions(task)
	options.OnStream = streamSink(callNames)

	// ---- Drive the orchestrator, handling the human-escalation ladder -------
	result := orchestrator.Run(ctx, options)
	for {
		switch result.Kind {
		case sporecore.RunSuccess:
			fmt.Printf("\norchestrator done (%d turn(s)): %s\n", result.Turns, truncate(result.Output, 400))
			findings := filepath.Join(workspaceRoot, "findings.md")
			if _, statErr := os.Stat(findings); statErr == nil {
				fmt.Printf("\nfindings.md written: %s\n", findings)
				runIssueFilingFlow(ctx, orchestrator, reader, findings)
			} else {
				fmt.Fprintln(os.Stderr, "\nwarning: orchestrator finished but workspace/findings.md was not written.")
			}
			return nil
		case sporecore.RunFailure:
			return fmt.Errorf("orchestrator failed after %d turn(s): %+v", result.Turns, result.Reason)
		case sporecore.RunWaitingForHuman:
			// The advice consult budget (3) was exhausted under EscalateToHuman:
			// the analysis_worker SubagentTool converted the over-budget consult
			// into a human pause, which bubbled up here.
			result = handleHumanEscalation(ctx, orchestrator, advisorHandler, reader, result)
		default:
			return fmt.Errorf("run ended unexpectedly: %s %+v", result.Kind, result.Reason)
		}
	}
}

// streamSink builds the orchestrator stream sink: boundary banners for the
// analysis worker, terse lines for the standard tools (mirrors 11-multi-agent).
func streamSink(callNames map[string]string) func(sporecore.HarnessStreamEvent) {
	return func(ev sporecore.HarnessStreamEvent) {
		switch ev.Kind {
		case sporecore.HarnessStreamTurnStart:
			fmt.Printf("orchestrator · turn %d\n", ev.Turn)
		case sporecore.HarnessStreamToolCall:
			callNames[ev.CallID] = ev.Name
			if ev.Name == "analysis_worker" {
				fmt.Println("┌─ orchestrator → analysis_worker")
				fmt.Printf("│  received: %s\n", truncate(instructionArg(ev.Args), 200))
			} else {
				fmt.Printf("  orchestrator → %s(%s)\n", ev.Name, truncate(string(ev.Args), 140))
			}
		case sporecore.HarnessStreamToolResult:
			name, ok := callNames[ev.CallID]
			if !ok {
				name = "<tool>"
			}
			delete(callNames, ev.CallID)
			if name == "analysis_worker" {
				tag := "findings"
				if ev.IsError {
					tag = "FAILED"
				}
				fmt.Println("└─ analysis_worker → orchestrator")
				fmt.Printf("   %s: %s\n", tag, truncate(ev.ResultContent, 300))
			} else {
				tag := "ok"
				if ev.IsError {
					tag = "err"
				}
				fmt.Printf("  %s → orchestrator [%s]: %s\n", name, tag, truncate(ev.ResultContent, 140))
			}
		}
	}
}

// handleHumanEscalation handles an advice-budget-exhausted human escalation with
// the three-choice ladder. Returns the next RunResult (the orchestrator resumed).
//
// IMPORTANT (honest mechanics): the worker's paused consult lives inside the
// orchestrator's PausedState child state, and the harness does NOT yet wire a
// child-consult resume through the parent (the #5/#115 follow-up). So every
// choice here resumes the ORCHESTRATOR with the human's decision injected as
// guidance; the specific module's in-flight worker audit is dropped. "+1 advisor
// turn" re-runs the advisor handler HOST-SIDE and injects its answer as that
// guidance — the closest we can get to a budget bump without a core primitive.
func handleHumanEscalation(
	ctx context.Context,
	orchestrator sporecore.Harness,
	advisorHandler sporecore.Harness,
	reader *bufio.Scanner,
	result sporecore.RunResult,
) sporecore.RunResult {
	contextText := "advice requested"
	if result.Request != nil {
		switch result.Request.Kind {
		case sporecore.HumanReqReview:
			contextText = result.Request.Content
		case sporecore.HumanReqClarification:
			contextText = result.Request.Question
		case sporecore.HumanReqToolApproval:
			contextText = "tool approval requested"
		}
	}
	fmt.Println("\n╔═ HUMAN ESCALATION (advisor budget exhausted) ═")
	fmt.Printf("║ %s\n", truncate(contextText, 400))
	fmt.Println("╚═══════════════════════════════════════════════")
	fmt.Println("Choose: [1] +1 advisor turn  [2] abort subagent & chat  [3] free-form answer")

	if result.State == nil {
		fmt.Fprintln(os.Stderr, "escalation arrived without a resumable state; aborting.")
		return sporecore.RunResult{Kind: sporecore.RunFailure}
	}
	state := *result.State

	choice := strings.TrimSpace(promptLine(reader, "> "))
	switch choice {
	case "1":
		// Re-run the advisor handler once, host-side, on the escalation context;
		// inject its output as the orchestrator's guidance.
		fmt.Println("(running advisor for one more turn…)")
		advTask := sporecore.NewTask(contextText, sporecore.NewSessionID(), sporecore.LoopStrategy{
			Kind:          sporecore.StrategyReAct,
			MaxIterations: 16,
		})
		advice := "advisor did not complete cleanly"
		if r := advisorHandler.Run(ctx, sporecore.NewHarnessRunOptions(advTask)); r.Kind == sporecore.RunSuccess {
			advice = r.Output
		}
		fmt.Printf("advisor: %s\n", truncate(advice, 300))
		return orchestrator.Resume(ctx, state, sporecore.HumanResponse{Kind: sporecore.HumanRespAnswer, Text: advice}, nil)
	case "2":
		fmt.Println("(aborting the stuck subagent; returning to the orchestrator…)")
		return orchestrator.Resume(ctx, state, sporecore.HumanResponse{Kind: sporecore.HumanRespHalt}, nil)
	default:
		text := choice
		if choice == "3" {
			text = promptLine(reader, "your answer> ")
		}
		return orchestrator.Resume(ctx, state, sporecore.HumanResponse{Kind: sporecore.HumanRespAnswer, Text: text}, nil)
	}
}

// runIssueFilingFlow presents the top-5 and offers to file them as issues. The
// model drives gh issue create via bash_command (no gh skill).
func runIssueFilingFlow(ctx context.Context, orchestrator sporecore.Harness, reader *bufio.Scanner, findings string) {
	report, _ := os.ReadFile(findings)
	fmt.Println("\n── top findings (workspace/findings.md) ──")
	fmt.Println(truncate(string(report), 1200))
	fmt.Println("──────────────────────────────────────────")

	answer := promptLine(reader, "File these as GitHub issues? [y/N] ")
	if !strings.EqualFold(strings.TrimSpace(answer), "y") {
		fmt.Println("Not filing. Done.")
		return
	}

	fmt.Println("(asking the orchestrator to file the top 5 via `gh issue create`…)")
	task := sporecore.NewTask(
		"Using `bash_command`, file the TOP 5 findings from workspace/findings.md as GitHub "+
			"issues via `gh issue create` — one issue per finding, with a clear title and a body "+
			"containing the file, line, severity, and description. Run `gh` once per finding. "+
			"Report the issue URLs when done.",
		sporecore.NewSessionID(),
		sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, MaxIterations: 24},
	)
	if r := orchestrator.Run(ctx, sporecore.NewHarnessRunOptions(task)); r.Kind == sporecore.RunSuccess {
		fmt.Printf("\nfiling done: %s\n", truncate(r.Output, 400))
	} else {
		fmt.Fprintf(os.Stderr, "\nfiling did not complete cleanly: %s %+v\n", r.Kind, r.Reason)
	}
}

// seedBundledAuditSkill seeds .spore/skills/audit/SKILL.md from the bundled copy
// if absent.
func seedBundledAuditSkill(repoRoot string) {
	dir := filepath.Join(repoRoot, ".spore", "skills", "audit")
	file := filepath.Join(dir, "SKILL.md")
	if _, err := os.Stat(file); err == nil {
		return
	}
	if os.MkdirAll(dir, 0o755) == nil {
		_ = os.WriteFile(file, []byte(bundledAuditSkill), 0o644)
	}
}

// readAuditPrompt reads the audit prompt from the REPL: prints the default,
// accepts an empty line as the default verbatim.
func readAuditPrompt(reader *bufio.Scanner) string {
	fmt.Println("Default audit prompt (press enter to accept, or type your own):")
	fmt.Printf("  %s\n", defaultAuditPrompt)
	line := promptLine(reader, "audit> ")
	if strings.TrimSpace(line) == "" {
		return defaultAuditPrompt
	}
	return line
}

// promptLine prints a prompt and reads one line from the scanner (trailing
// newline stripped).
func promptLine(reader *bufio.Scanner, prompt string) string {
	fmt.Print(prompt)
	if !reader.Scan() {
		return ""
	}
	return strings.TrimRight(reader.Text(), "\r\n")
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

// instructionArg pulls the `instruction` field out of a tool-call args blob for
// the boundary banner.
func instructionArg(args json.RawMessage) string {
	var probe struct {
		Instruction string `json:"instruction"`
	}
	if json.Unmarshal(args, &probe) == nil && probe.Instruction != "" {
		return probe.Instruction
	}
	return "<no instruction>"
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

// flagOrEnv resolves a value from a CLI flag, then an env var, then a default.
func flagOrEnv(flag, env, def string) string {
	if v := flagValue(flag); v != "" {
		return v
	}
	if v := os.Getenv(env); v != "" {
		return v
	}
	return def
}

// truncate keeps boundary lines readable.
func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}
